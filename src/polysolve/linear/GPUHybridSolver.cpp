
#include "GPUHybridSolver.hpp"
#include "../Utils.hpp"

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>

#include <iostream>

#include <mpi.h>
#include <metis.h>


#define CHECK_CUDA(call) \
    do { \
        cudaError_t status = call; \
        if (status != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorName(status) \
                      << " (" << cudaGetErrorString(status) << ")" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CUDSS(call) \
    do { \
        cudssStatus_t status = call; \
        if (status != CUDSS_STATUS_SUCCESS) { \
            std::cerr << "cuDSS Error at " << __FILE__ << ":" << __LINE__ \
                      << " code " << (int) status << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


namespace polysolve::linear
{
    GPUHybridSolver::GPUHybridSolver()
    {
        HYPRE_Init();
        CHECK_CUDA(cudaSetDevice(0));
        HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
        HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);
        HYPRE_SetSpGemmUseCusparse(false);
        HYPRE_SetUseGpuRand(true);

        CHECK_CUDSS(cudssCreate(&cudss_handle));
    }

    void GPUHybridSolver::set_parameters(const json &params)
    {
        if (params.contains("GPUHybridSolver"))
        {
            if (params["GPUHybridSolver"].contains("max_iter"))
            {
                max_iter_ = params["GPUHybridSolver"]["max_iter"];
            }
            if (params["GPUHybridSolver"].contains("tolerance"))
            {
                conv_tol_ = params["GPUHybridSolver"]["tolerance"];
            }
            if (params["GPUHybridSolver"].contains("theta"))
            {
                theta = params["GPUHybridSolver"]["theta"];
            }
            if (params["GPUHybridSolver"].contains("dimension"))
            {
                dimension_ = params["GPUHybridSolver"]["dimension"];
            }
            if (params["GPUHybridSolver"].contains("do_mixed_precond"))
            {
                do_mixed_precond = params["GPUHybridSolver"]["do_mixed_precond"];
            }
            if (params["GPUHybridSolver"].contains("use_absolute_tol"))
            {
                use_absolute_tol = params["GPUHybridSolver"]["use_absolute_tol"];
            }
            if (params["GPUHybridSolver"].contains("decompose_subdomains"))
            {
                decompose_subdomains = params["GPUHybridSolver"]["decompose_subdomains"];
            }
            if (params["GPUHybridSolver"].contains("min_subdomain_size"))
            {
                min_subdomain_size = params["GPUHybridSolver"]["min_subdomain_size"];
            }
            if (params["GPUHybridSolver"].contains("max_subdomain_size"))
            {
                max_subdomain_size = params["GPUHybridSolver"]["max_subdomain_size"];
            }
            if (params["GPUHybridSolver"].contains("bad_dof_threshold"))
            {
                bad_dof_threshold = params["GPUHybridSolver"]["bad_dof_threshold"];
            }
        }
    } 

    void GPUHybridSolver::get_info(json &params) const
    {
        params["num_iterations"] = num_iterations;
        params["final_res_norm"] = final_res_norm;
    }

    void GPUHybridSolver::check_settings() const
    {

    }

    void GPUHybridSolver::analyze_pattern(const StiffnessMatrix &A, const int precond_num)
    {
        check_settings();
    }

    void GPUHybridSolver::factorize(const StiffnessMatrix &Ain)
    {
        {
            POLYSOLVE_SCOPED_STOPWATCH("eigen matrix copy time", eigen_copy_time, *logger);
            sparse_A = Ain;
            sparse_A.makeCompressed();
        }

        if (do_mixed_precond)
        {
            prepare_dss();
        }

        if (has_matrix_)
        {
            POLYSOLVE_SCOPED_STOPWATCH("matrix destroy time", matrix_destroy_time, *logger);
            HYPRE_IJMatrixDestroy(A);
            has_matrix_ = false;
        }

        copy_matrix_to_hypre();
    }

    namespace {
        void HypreBoomerAMG_SetDefaultOptions(HYPRE_Solver &amg_precond)
        {
            // AMG coarsening options:
            int coarsen_type = 8; // 10 = HMIS, 8 = PMIS, 6 = Falgout, 0 = CLJP
            int agg_levels = 1;    // number of aggressive coarsening levels
            double theta = 0.25;   // strength threshold: 0.25, 0.5, 0.8

            // AMG interpolation options:
            int interp_type = 6; // 6 = extended+i, 0 = classical
            int Pmax = 4;        // max number of elements per row in P

            // AMG relaxation options:
            int relax_type = 18;   // 8 = l1-GS, 6 = symm. GS, 3 = GS, 18 = l1-Jacobi
            int relax_sweeps = 1; // relaxation sweeps on each level

            // Additional options:
            int print_level = 0; // print AMG iterations? 1 = no, 2 = yes
            int max_levels = 25; // max number of levels in AMG hierarchy

            int min_coarse_size = 5;

            HYPRE_BoomerAMGSetCoarsenType(amg_precond, coarsen_type);
            HYPRE_BoomerAMGSetAggNumLevels(amg_precond, agg_levels);
            HYPRE_BoomerAMGSetRelaxType(amg_precond, relax_type);

            HYPRE_BoomerAMGSetRelaxOrder(amg_precond, false);
            HYPRE_BoomerAMGSetRAP2(amg_precond, true);
            HYPRE_BoomerAMGSetKeepTranspose(amg_precond, true);
            
            HYPRE_BoomerAMGSetMinCoarseSize(amg_precond, min_coarse_size);
            HYPRE_BoomerAMGSetCycleRelaxType(amg_precond, relax_type, 3);
            HYPRE_BoomerAMGSetNumSweeps(amg_precond, relax_sweeps);
            HYPRE_BoomerAMGSetStrongThreshold(amg_precond, theta);
            HYPRE_BoomerAMGSetInterpType(amg_precond, interp_type);
            HYPRE_BoomerAMGSetPMaxElmts(amg_precond, Pmax);
            //print_level = 3;
            HYPRE_BoomerAMGSetPrintLevel(amg_precond, print_level);
            HYPRE_BoomerAMGSetMaxLevels(amg_precond, max_levels);

            // Use as a preconditioner (one V-cycle, zero tolerance)
            HYPRE_BoomerAMGSetMaxIter(amg_precond, 1);
            HYPRE_BoomerAMGSetTol(amg_precond, 0.0);
        }

        void HypreBoomerAMG_SetElasticityOptions(HYPRE_Solver &amg_precond, int dim, double theta)
        {
            // Make sure the systems AMG options are set
            HYPRE_BoomerAMGSetNumFunctions(amg_precond, dim);

            //HYPRE_BoomerAMGSetDofFunc(amg_precond, (HYPRE_Int*) dof_to_function.data());

            // More robust options with respect to convergence
            HYPRE_BoomerAMGSetAggNumLevels(amg_precond, 0);
            HYPRE_BoomerAMGSetStrongThreshold(amg_precond, theta);
        }
    }

    void GPUHybridSolver::solve(const Ref<const VectorXd> b, Ref<VectorXd> x)
    {
        Eigen::VectorXd result = x;
        Eigen::VectorXd rhs = b;

        HYPRE_ParVector par_b;
        HYPRE_ParVector par_x;
        init_hypre_vectors(b.size());            
        
        double *gpu_rhs;
        double *gpu_x;
        CHECK_CUDA(cudaMalloc(&gpu_rhs, rhs.size() * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&gpu_x, x.size() * sizeof(double)));
        CHECK_CUDA(cudaMemcpy(gpu_rhs, b.data(), rhs.size() * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(gpu_x, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice));

        set_hypre_vec(ij_b, par_b, gpu_rhs);
        set_hypre_vec(ij_x, par_x, gpu_x);
        
        HYPRE_Solver precond;

        HYPRE_BoomerAMGCreate(&precond);

        {
            POLYSOLVE_SCOPED_STOPWATCH("set AMG options", set_options_time, *logger);
            HypreBoomerAMG_SetDefaultOptions(precond);
            if (dimension_ > 1)
            {
                HypreBoomerAMG_SetElasticityOptions(
                    precond, 
                    dimension_, 
                    theta
                );
            }
        }

        {
            POLYSOLVE_SCOPED_STOPWATCH("AMG setup time", amg_setup_time, *logger);
            HYPRE_BoomerAMGSetup(precond, parcsr_A, par_b, par_x);
        }

        {
            POLYSOLVE_SCOPED_STOPWATCH("actual solve time", actual_solve_time, *logger);

            pcg_solve(gpu_rhs, gpu_x, par_b, par_x, precond);
        }

        CHECK_CUDA(cudaMemcpy(x.data(), gpu_x, x.size() * sizeof(double), cudaMemcpyDeviceToHost));
        final_res_norm = (b - sparse_A * x).norm();

        logger->debug("GPUHybrid solver Iterations: {}", num_iterations);
        logger->debug("GPUHybrid solver Final Relative Residual Norm: {}", final_res_norm);

        {
            POLYSOLVE_SCOPED_STOPWATCH("destroy time", destroy_time, *logger);
            HYPRE_BoomerAMGDestroy(precond);
            HYPRE_IJVectorDestroy(ij_x);
            HYPRE_IJVectorDestroy(ij_b);

            CHECK_CUDA(cudaFree(gpu_x));
            CHECK_CUDA(cudaFree(gpu_rhs));
        }
    }

    void GPUHybridSolver::copy_matrix_to_hypre()
    {
        POLYSOLVE_SCOPED_STOPWATCH("copy matrix time", matrix_copy_time, *logger);

        HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, sparse_A.rows() - 1, 0, sparse_A.cols() - 1, &A);
        HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
        HYPRE_IJMatrixInitialize(A);

        HYPRE_Int num_rows = sparse_A.rows();
        HYPRE_Int nnz = sparse_A.nonZeros();

        std::vector<HYPRE_Int> cpu_n_cols(num_rows, 0);
        std::vector<HYPRE_Int> cpu_rows(num_rows, 0);
        std::vector<HYPRE_Int> cpu_cols;
        std::vector<double> cpu_vals;

        cpu_cols.reserve(nnz);
        cpu_vals.reserve(nnz);

        // assuming symmetry
        for (HYPRE_Int k = 0; k < num_rows; ++k)
        {
            cpu_rows[k] = k;
            int counter = 0;
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(sparse_A, k); it; ++it)
            {
                cpu_cols.push_back((HYPRE_Int)it.col());
                cpu_vals.push_back(it.value());
                counter++;
            }
            cpu_n_cols[k] = counter;
        }

        HYPRE_Int *gpu_n_cols, *gpu_rows, *gpu_cols;
        double *gpu_vals;

        CHECK_CUDA(cudaMalloc(&gpu_n_cols, num_rows * sizeof(HYPRE_Int)));
        CHECK_CUDA(cudaMalloc(&gpu_rows, num_rows * sizeof(HYPRE_Int)));
        CHECK_CUDA(cudaMalloc(&gpu_cols, nnz * sizeof(HYPRE_Int)));
        CHECK_CUDA(cudaMalloc(&gpu_vals, nnz * sizeof(double)));

        CHECK_CUDA(cudaMemcpy(gpu_n_cols, cpu_n_cols.data(), num_rows * sizeof(HYPRE_Int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(gpu_rows, cpu_rows.data(), num_rows * sizeof(HYPRE_Int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(gpu_cols, cpu_cols.data(), nnz * sizeof(HYPRE_Int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(gpu_vals, cpu_vals.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));

        HYPRE_IJMatrixSetValues(A, num_rows, gpu_n_cols, gpu_rows, gpu_cols, gpu_vals);

        CHECK_CUDA(cudaFree(gpu_n_cols));
        CHECK_CUDA(cudaFree(gpu_rows));
        CHECK_CUDA(cudaFree(gpu_cols));
        CHECK_CUDA(cudaFree(gpu_vals));

        HYPRE_IJMatrixAssemble(A);

        void* temp_A = nullptr;
        HYPRE_IJMatrixGetObject(A, &temp_A);
        parcsr_A = static_cast<decltype(parcsr_A)>(temp_A);
    }

    void GPUHybridSolver::init_hypre_vectors(const int size)
    {
        HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, size - 1, &ij_x);
        HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
        HYPRE_IJVectorInitializeShell(ij_x);
        HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, size - 1, &ij_b);
        HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
        HYPRE_IJVectorInitializeShell(ij_b);
    }

    void GPUHybridSolver::matmul(double* x, double* result)
    {
        POLYSOLVE_SCOPED_STOPWATCH("matmul time", matmul_time, *logger);     
        
        HYPRE_ParVector par_x;
        HYPRE_ParVector par_result;

        set_hypre_vec(ij_x, par_x, x);
        set_hypre_vec(ij_b, par_result, result);

        HYPRE_ParCSRMatrixMatvec(1.0, parcsr_A, par_x, 0.0, par_result);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    double GPUHybridSolver::dot(double* a, double* b)
    {
        HYPRE_ParVector par_a;
        HYPRE_ParVector par_b;

        set_hypre_vec(ij_x, par_a, a);
        set_hypre_vec(ij_b, par_b, b);
        double result;
        HYPRE_ParVectorInnerProd(par_a, par_b, &result);
        return result;
    }

    void GPUHybridSolver::vector_copy(double* x, double* y)
    {
        HYPRE_ParVector par_x;
        HYPRE_ParVector par_y;

        set_hypre_vec(ij_x, par_x, x);
        set_hypre_vec(ij_b, par_y, y);

        HYPRE_ParVectorCopy(par_x, par_y);
    }

    void GPUHybridSolver::vector_add(double alpha, double* x, double* y)
    {
        HYPRE_ParVector par_x;
        HYPRE_ParVector par_y;

        set_hypre_vec(ij_x, par_x, x);
        set_hypre_vec(ij_b, par_y, y);

        hypre_ParVectorAxpy(alpha, par_x, par_y);
    }

    void GPUHybridSolver::vector_scale(double alpha, double* x)
    {
        HYPRE_ParVector par_x;

        set_hypre_vec(ij_x, par_x, x);

        HYPRE_ParVectorScale(alpha, par_x);
    }

    void GPUHybridSolver::set_hypre_vec(HYPRE_IJVector &my_ij_x, HYPRE_ParVector &par_x, double* x)
    {
        HYPRE_IJVectorSetData(my_ij_x, x);
        HYPRE_IJVectorAssemble(my_ij_x);
        HYPRE_IJVectorGetObject(my_ij_x, (void **)&par_x);
    }

    void GPUHybridSolver::custom_mixed_precond_iter(const HYPRE_Solver &precond, double* r, double* z, double* buffer, double* z2)
    {
        assert(bad_indices_.size() == 1);
        if (!do_mixed_precond || bad_indices_.size() == 0 || bad_indices_[0].size() == 0)
        {
            amg_precond_iter(precond, r, z);
            return;
        }
        else
        {
            CHECK_CUDA(cudaMemset(buffer, 0, sparse_A.rows() * sizeof(double)));
            CHECK_CUDA(cudaMemset(z2, 0, sparse_A.rows() * sizeof(double)));
            amg_precond_iter(precond, r, z);
            dss_precond_iter(z, r, z2);
            matmul(z2, z);
            vector_copy(r, buffer);
            vector_add(-1.0, z, buffer);
            CHECK_CUDA(cudaMemset(z, 0, sparse_A.rows() * sizeof(double)));
            amg_precond_iter(precond, buffer, z);
            vector_add(1.0, z2, z);
        }

    }

    void GPUHybridSolver::dss_precond_iter(double* z, double* r, double* next_z)
    {
       {
            POLYSOLVE_SCOPED_STOPWATCH("dss step time: ", dss_step_time, *logger);

            CHECK_CUDA(cudaMemset(next_z, 0, sparse_A.rows() * sizeof(double)));

            matmul(z, next_z);
            vector_scale(-1.0, next_z);
            vector_add(1.0, r, next_z);

            thrust::gather(
                thrust::device,
                all_bad_dof_map.begin(), 
                all_bad_dof_map.end(), 
                next_z, 
                d_b
            );
            
            CHECK_CUDA(cudaDeviceSynchronize());
            double test_time;
            {
                POLYSOLVE_SCOPED_STOPWATCH("dss backsub time: ", test_time, *logger);
                CHECK_CUDSS(cudssExecute(cudss_handle, CUDSS_PHASE_SOLVE, config, solverData, 
                                        batchMatrixA, batchMatrixX, batchMatrixB));
                CHECK_CUDA(cudaDeviceSynchronize());
            }
            
            CHECK_CUDA(cudaMemset(next_z, 0, sizeof(double) * sparse_A.rows()));
    
            thrust::scatter(
                thrust::device,
                d_x,
                d_x + all_bad_dof_map.size(),
                all_bad_dof_map.begin(),
                next_z
            );

            vector_add(1.0, z, next_z);
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    }

    void GPUHybridSolver::amg_precond_iter(const HYPRE_Solver &precond, double* b, double* x)
    {
        HYPRE_ParVector par_x;
        HYPRE_ParVector par_b;

        set_hypre_vec(ij_x, par_x, x);
        set_hypre_vec(ij_b, par_b, b);

        {
            POLYSOLVE_SCOPED_STOPWATCH("boomeramg solve time: ", solve_time, *logger);
            HYPRE_BoomerAMGSolve(precond, parcsr_A, par_b, par_x);
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    }

    void GPUHybridSolver::prepare_dss()
    {
        POLYSOLVE_SCOPED_STOPWATCH("prepare dss time", prepare_dss_time, *logger);

        select_bad_indices();
        std::vector<std::set<int>> overlap_extensions;

        if (decompose_subdomains)
        {
            decompose_subdomains_to_disjoint_subsets(overlap_extensions);
        }

        bad_indices_arrays.clear();
        owned_counts.clear();
        owned_counts.resize(bad_indices_.size());

        logger->trace("Num subdomains: {}", bad_indices_.size());

        int i = 0;
        for (auto& subdomain : bad_indices_)
        {
            logger->trace("Subdomain size: {}", subdomain.size() + overlap_extensions[i].size());
            std::vector<int> cpu_buff(subdomain.begin(), subdomain.end());
            bad_indices_arrays.emplace_back(cpu_buff.begin(), cpu_buff.end());
            //bad_indices_arrays.back().insert(bad_indices_arrays.back().end(), overlap_extensions[i].begin(), overlap_extensions[i].end());
            //owned_counts[i] = subdomain.size();
            ++i;
        }

        //h_node_multiplicity.clear();
        //h_node_multiplicity.resize(sparse_A.rows());
        //for (const auto& cpu_map : bad_indices_arrays) 
        //{
        //    for (int global_idx : cpu_map) 
        //    {
        //        h_node_multiplicity[global_idx]++;
        //    }
        //}

        //d_node_multiplicity = h_node_multiplicity;
        //raw_d_multiplicity = thrust::raw_pointer_cast(d_node_multiplicity.data());

        CHECK_CUDA(cudaDeviceSynchronize());
        factorize_submatrix();
    }

    void GPUHybridSolver::decompose_subdomains_to_disjoint_subsets(std::vector<std::set<int>> &overlap_extensions)
    {
        POLYSOLVE_SCOPED_STOPWATCH("subdomain decomposition time", decomp_time, *logger);
        std::vector<int> all_bad_dofs;
        std::vector<int> global_to_local(sparse_A.rows(), -1);
        for (auto &subdomain : bad_indices_)
        {
            for (auto index : subdomain)
            {
                global_to_local[index] = all_bad_dofs.size();
                all_bad_dofs.push_back(index);
            }
        }

        disjointSet decomposed_subdomains(all_bad_dofs.size());

        for (int k : all_bad_dofs)
        {
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(sparse_A, k); it; ++it)
            {
                if (global_to_local[it.col()] != -1)
                {
                    decomposed_subdomains.union_set(global_to_local[it.row()], global_to_local[it.col()]);
                }
            }
        }

        std::unordered_map<int, std::vector<int>> chosen_sets;
        for (auto index : all_bad_dofs)
        {
            chosen_sets[decomposed_subdomains.find_set(global_to_local[index])].push_back(index);
        }

        bad_indices_.clear();

        for (auto &kv : chosen_sets)
        {
            if (kv.second.size() < min_subdomain_size)
            {
                continue;
            }
            if (kv.second.size() > max_subdomain_size)
            {
                partition_subdomain(kv.second, overlap_extensions);
                continue;
            }
            bad_indices_.emplace_back(kv.second.begin(), kv.second.end());
            overlap_extensions.emplace_back();
        }
    }

    void GPUHybridSolver::partition_subdomain(std::vector<int>& subdomain, std::vector<std::set<int>> &overlap_extensions)
    {
        std::vector<idx_t> global_to_subdomain(sparse_A.rows(), -1);
        for (int i = 0; i < subdomain.size(); ++i)
        {
            global_to_subdomain[subdomain[i]] = i;
        }

        idx_t nvtxs = subdomain.size();
        idx_t ncon = 1;

        std::vector<idx_t> xadj;
        std::vector<idx_t> adjncy;
        std::vector<idx_t> adjwgt;

        xadj.push_back(0);

        for (int i = 0; i < subdomain.size(); ++i)
        {
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(sparse_A, subdomain[i]); it; ++it)
            {
                if (it.col() == it.row())
                {
                    continue;
                }

                if (global_to_subdomain[it.col()] != -1)
                {
                    adjncy.push_back(global_to_subdomain[it.col()]);

                    idx_t weight = abs(it.value()) * 1e4 + 1;
                    adjwgt.push_back(weight);
                }
            }
            xadj.push_back(adjncy.size());
        }

        idx_t nparts = (nvtxs + max_subdomain_size - 1) / max_subdomain_size;
        real_t tolerance = (real_t) max_subdomain_size / ((real_t) nvtxs / (real_t) nparts);
        real_t ubvec[1];
        ubvec[0] = tolerance;

        idx_t options[METIS_NOPTIONS];
        METIS_SetDefaultOptions(options);

        idx_t objval;
        std::vector<idx_t> part(nvtxs);

        int status = METIS_PartGraphKway(
            &nvtxs,
            &ncon,
            xadj.data(),
            adjncy.data(),
            nullptr,
            nullptr,
            adjwgt.data(),
            &nparts,
            nullptr,
            ubvec,
            options,
            &objval,
            part.data()
        );

        int new_num_subdomains = *std::max_element(part.begin(), part.end()) + 1;
        int start_idx = bad_indices_.size();
        
        // Allocate space for both core and overlap sets
        for (int i = 0; i < new_num_subdomains; ++i)
        {
            bad_indices_.emplace_back();
            overlap_extensions.emplace_back(); 
        }

        std::vector<std::vector<int>> boundaries(new_num_subdomains);
        
        // Step 1: Populate core (owned) DoFs into bad_indices_
        for (int i = 0; i < subdomain.size(); ++i)
        {
            int p = part[i];
            bad_indices_[start_idx + p].insert(subdomain[i]);
            boundaries[p].push_back(subdomain[i]); 
        }

        // Step 2: Expand boundaries and save ghosts to overlap_extensions
        int overlap_layers = 0;
        for (int layer = 0; layer < overlap_layers; ++layer)
        {
            std::vector<std::vector<int>> next_boundaries(new_num_subdomains);

            for (int p = 0; p < new_num_subdomains; ++p)
            {
                for (int u : boundaries[p])
                {
                    for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(sparse_A, u); it; ++it)
                    {
                        int v = it.col();
                        
                        if (u == v || global_to_subdomain[v] == -1) continue; // Skip diagonal/self-loops

                        // If 'v' is already an owned DoF in the core partition, skip it
                        if (bad_indices_[start_idx + p].count(v))
                        {
                            continue; 
                        }

                        // Try inserting into the overlap set. If it wasn't already there,
                        // .second is true, so we add it to the next boundary to keep searching.
                        if (overlap_extensions[start_idx + p].insert(v).second)
                        {
                            next_boundaries[p].push_back(v);
                        }
                    }
                }
            }
            
            boundaries = std::move(next_boundaries);
        }
    }

    void GPUHybridSolver::build_index_mappings()
    {
        index_mappings.clear();
        index_mappings.resize(bad_indices_.size());

        int i_counter = 0;
        for (int i = 0; i < bad_indices_.size(); ++i)
        {
            int j_counter = 0;
            for (auto j : bad_indices_arrays[i])
            {
                index_mappings[i_counter][j] = j_counter;
                ++j_counter;
            }
            ++i_counter;
        }
    }

    void GPUHybridSolver::select_bad_indices()
    {
        POLYSOLVE_SCOPED_STOPWATCH("bad dof selection time", bad_dof_selection_time, *logger);
        
        bad_indices_.clear();
        bad_indices_.resize(1);
        // 1. Extract diagonal magnitudes on CPU using Eigen
        Eigen::VectorXd sq_mags;
        {
            POLYSOLVE_SCOPED_STOPWATCH("select dofs from hess diagonal", select_dofs_from_diag_time, *logger);
            sq_mags = sparse_A.diagonal().cwiseAbs();
        }

        int n = sq_mags.size();
        if (n == 0) return; // Quick safety check

        // 2. Transfer data to GPU
        // Eigen::VectorXd stores data contiguously, so we can pass its pointer directly to Thrust
        thrust::device_vector<double> d_sq_mags(sq_mags.data(), sq_mags.data() + n);
        
        // Create and populate the indices array (0, 1, 2, ..., n-1)
        thrust::device_vector<int> d_indices(n);
        thrust::sequence(d_indices.begin(), d_indices.end());

        // 3. Argsort: Sorts d_sq_mags in place and permutes d_indices to match
        thrust::sort_by_key(d_sq_mags.begin(), d_sq_mags.end(), d_indices.begin());

        // 4. Compute Logarithms on GPU
        thrust::device_vector<double> d_log_sorted(n);
        thrust::transform(d_sq_mags.begin(), d_sq_mags.end(), d_log_sorted.begin(), 
                          [] __device__ (double val) { return log(val); });

        // Since the array is sorted and log() is monotonic, min and max are just the first and last elements
        double min_mag = d_log_sorted.front();
        double max_mag = d_log_sorted.back();

        // 5. Compute deviations on GPU
        thrust::device_vector<double> d_deviations(n);
        
        // We use a counting iterator to represent 'i' and pair it with 'd_log_sorted' in a binary transform
        thrust::transform(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(n),
            d_log_sorted.begin(),
            d_deviations.begin(),
            [min_mag, max_mag, n] __device__ (int i, double log_y) {
                // Use fabs() for device-side double absolute value
                double expected_y = (max_mag - min_mag) / (n - 1.0) * i + min_mag;
                return fabs(expected_y - log_y);
            }
        );

        // 6. Find the index of the maximum deviation
        auto max_dev_iter = thrust::max_element(d_deviations.begin(), d_deviations.end());
        int cutoff_index = thrust::distance(d_deviations.begin(), max_dev_iter);
        
        if (bad_dof_threshold < 1.0)
        {
            cutoff_index = n * (1.0 - bad_dof_threshold);
        }

        // Retrieve the cutoff value from the GPU
        const double cutoff = d_sq_mags[cutoff_index];
        logger->trace("Problematic threshold: {}, cutoff index: {}", cutoff, cutoff_index);

        // 7. Transfer ONLY the bad indices back to the CPU
        int num_bad = n - cutoff_index;
        std::vector<int> h_bad_indices(num_bad);
        thrust::copy(d_indices.begin() + cutoff_index, d_indices.end(), h_bad_indices.begin());

        // 8. Insert into your tracking structure
        for (int idx : h_bad_indices)
        {
            bad_indices_[0].insert(idx);
        }
    }

    void GPUHybridSolver::factorize_submatrix()
    {
        POLYSOLVE_SCOPED_STOPWATCH("assemble D", dss_assembly_time, *logger);

        int batchCount = bad_indices_arrays.size();
        if (batchCount == 0) return;

        build_index_mappings();
        free_device_memory();

        CHECK_CUDA(cudaDeviceSynchronize());
        logger->trace("Checkpoint 0");

        CHECK_CUDSS(cudssConfigCreate(&config));
        CHECK_CUDSS(cudssDataCreate(cudss_handle, &solverData));

        CHECK_CUDA(cudaDeviceSynchronize());
        logger->trace("Checkpoint 1");

        h_nrows.clear();
        h_ncols.clear();
        h_nnz.clear();
        h_vec_ncols.clear();
        h_ld.clear();

        h_csrRowOffsets_void.clear();
        h_csrColIndices_void.clear();
        h_csrValues_void.clear();

        int total_bad_dofs = 0;
        all_bad_dof_map.clear();

        CHECK_CUDA(cudaDeviceSynchronize());
        logger->trace("Checkpoint 2");

        for (auto &ba : bad_indices_arrays)
        {
            total_bad_dofs += ba.size();
        }
        all_bad_dof_map.reserve(total_bad_dofs);
        logger->trace("Total bad dofs: {}", total_bad_dofs);
        for (auto &ba : bad_indices_arrays)
        {
            all_bad_dof_map.insert(all_bad_dof_map.end(), ba.begin(), ba.end());
        }

        CHECK_CUDA(cudaDeviceSynchronize());
        logger->trace("Checkpoint 3");

        std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor>> cpu_matrices(batchCount);
        
        int total_row_offsets = 0; // sum of (nrows + 1)
        int total_nnz = 0;         // sum of non-zeros

        // --- PHASE 1: ASSEMBLE AND COUNT ---
        for (int i = 0; i < batchCount; ++i)
        {
            assemble_D(i, i, cpu_matrices[i]); // Assuming i_counter == i

            int m_nrows = cpu_matrices[i].rows();
            int m_ncols = cpu_matrices[i].cols();
            int m_nnz   = cpu_matrices[i].nonZeros();

            // Persist dimensions in host memory
            h_nrows.push_back(m_nrows);
            h_ncols.push_back(m_ncols);
            h_nnz.push_back(m_nnz);
            h_vec_ncols.push_back(1);
            h_ld.push_back(m_nrows);

            total_row_offsets += (m_nrows + 1);
            total_nnz += m_nnz;
        }

        // --- PHASE 2: PACK INTO CONTIGUOUS CPU MEMORY ---
        std::vector<int> h_all_rowOffsets(total_row_offsets);
        std::vector<int> h_all_colIndices(total_nnz);
        std::vector<double> h_all_values(total_nnz);

        int offset_rows = 0;
        int offset_nnz = 0;

        for (int i = 0; i < batchCount; ++i)
        {
            const auto& D = cpu_matrices[i];
            int current_rows = D.rows();
            int current_nnz = D.nonZeros();

            // Lightning-fast memcpy on the CPU side
            std::memcpy(&h_all_rowOffsets[offset_rows], D.outerIndexPtr(), (current_rows + 1) * sizeof(int));
            std::memcpy(&h_all_colIndices[offset_nnz], D.innerIndexPtr(), current_nnz * sizeof(int));
            std::memcpy(&h_all_values[offset_nnz], D.valuePtr(), current_nnz * sizeof(double));

            offset_rows += (current_rows + 1);
            offset_nnz += current_nnz;
        }

        // --- PHASE 3: THE 3 MEGA-ALLOCATIONS & TRANSFERS ---
        int* d_all_rowOffsets;
        int* d_all_colIndices;
        double* d_all_values;

        CHECK_CUDA(cudaMalloc(&d_all_rowOffsets, total_row_offsets * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_all_colIndices, total_nnz * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_all_values, total_nnz * sizeof(double)));

        CHECK_CUDA(cudaMemcpy(d_all_rowOffsets, h_all_rowOffsets.data(), total_row_offsets * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_all_colIndices, h_all_colIndices.data(), total_nnz * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_all_values, h_all_values.data(), total_nnz * sizeof(double), cudaMemcpyHostToDevice));

        // --- PHASE 4: ASSIGN SUB-POINTERS FOR CUDSS ---
        offset_rows = 0;
        offset_nnz = 0;

        for (int i = 0; i < batchCount; ++i)
        {
            int current_rows = h_nrows[i];
            int current_nnz = h_nnz[i];

            // Pointer arithmetic advances by the type size (int/double), so we just add the offset
            h_csrRowOffsets_void.push_back(static_cast<void*>(d_all_rowOffsets + offset_rows));
            h_csrColIndices_void.push_back(static_cast<void*>(d_all_colIndices + offset_nnz));
            h_csrValues_void.push_back(static_cast<void*>(d_all_values + offset_nnz));

            offset_rows += (current_rows + 1);
            offset_nnz += current_nnz;
        }

        CHECK_CUDA(cudaDeviceSynchronize());
        logger->trace("Checkpoint 4");

        CHECK_CUDA(cudaDeviceSynchronize());
        logger->trace("Checkpoint 5");

        CHECK_CUDA(cudaMalloc(&d_csrRowOffsets_void, batchCount * sizeof(void*)));
        CHECK_CUDA(cudaMalloc(&d_csrColIndices_void, batchCount * sizeof(void*)));
        CHECK_CUDA(cudaMalloc(&d_csrValues_void, batchCount * sizeof(void*)));

        CHECK_CUDA(cudaMalloc(&d_x, total_bad_dofs * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_b, total_bad_dofs * sizeof(double)));

        CHECK_CUDA(cudaDeviceSynchronize());
        logger->trace("Checkpoint 6");

        h_x_void.push_back(static_cast<void*>(d_x));
        h_b_void.push_back(static_cast<void*>(d_b));

        for (int i = 1; i < h_nrows.size(); ++i)
        {
            h_x_void.push_back(static_cast<void*>(h_x_void.back() + h_nrows[i - 1] * sizeof(double)));
            h_b_void.push_back(static_cast<void*>(h_b_void.back() + h_nrows[i - 1] * sizeof(double)));
        }

        CHECK_CUDA(cudaMalloc(&d_x_void, batchCount * sizeof(void*)));
        CHECK_CUDA(cudaMalloc(&d_b_void, batchCount * sizeof(void*)));

        CHECK_CUDA(cudaDeviceSynchronize());
        logger->trace("Checkpoint 7");

        // Copy Pointer Arrays to GPU
        CHECK_CUDA(cudaMemcpy(d_csrRowOffsets_void, h_csrRowOffsets_void.data(), batchCount * sizeof(void*), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_csrColIndices_void, h_csrColIndices_void.data(), batchCount * sizeof(void*), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_csrValues_void, h_csrValues_void.data(), batchCount * sizeof(void*), cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaMemcpy(d_x_void, h_x_void.data(), batchCount * sizeof(void*), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b_void, h_b_void.data(), batchCount * sizeof(void*), cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaDeviceSynchronize());
        logger->trace("Checkpoint 8");

        CHECK_CUDSS(cudssMatrixCreateBatchDn(
            &batchMatrixX, batchCount, h_nrows.data(), h_vec_ncols.data(), h_ld.data(), 
            d_x_void, CUDA_R_32I, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR
        ));

        CHECK_CUDSS(cudssMatrixCreateBatchDn(
            &batchMatrixB, batchCount, h_nrows.data(), h_vec_ncols.data(), h_ld.data(), 
            d_b_void, CUDA_R_32I, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR
        ));

        CHECK_CUDA(cudaDeviceSynchronize());
        logger->trace("Checkpoint 9");

        {
            POLYSOLVE_SCOPED_STOPWATCH("factorize D", dss_factorization_time, *logger);

            CHECK_CUDSS(cudssMatrixCreateBatchCsr(
                &batchMatrixA, batchCount, h_nrows.data(), h_ncols.data(), h_nnz.data(), 
                d_csrRowOffsets_void, nullptr, d_csrColIndices_void, d_csrValues_void, 
                CUDA_R_32I, CUDA_R_64F, CUDSS_MTYPE_SYMMETRIC, 
                CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO
            ));

            CHECK_CUDSS(cudssExecute(cudss_handle, CUDSS_PHASE_ANALYSIS, config, solverData, 
                                    batchMatrixA, nullptr, nullptr));
            CHECK_CUDSS(cudssExecute(cudss_handle, CUDSS_PHASE_FACTORIZATION, config, solverData, 
                                batchMatrixA, nullptr, nullptr));
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    }

    void GPUHybridSolver::assemble_D(int bad_i, int i, Eigen::SparseMatrix<double, Eigen::RowMajor>& D)
    {
        D.resize(bad_indices_arrays[i].size(), bad_indices_arrays[i].size());
        std::vector<Eigen::Triplet<double>> triplets;
        for (int k : bad_indices_arrays[i])
        {
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(sparse_A, k); it; ++it)
            {
                auto ind_it = index_mappings[bad_i].find(it.col());
                if (ind_it != index_mappings[bad_i].end())
                {
                    triplets.push_back(Eigen::Triplet<double>(index_mappings[bad_i][it.row()], index_mappings[bad_i][it.col()], it.value()));
                }
            }
        }

        {
            //POLYSOLVE_SCOPED_STOPWATCH("set D from triplets", set_from_triplets_time, *logger);
            D.setFromTriplets(triplets.begin(), triplets.end());
        }
    }

    void GPUHybridSolver::pcg_solve(double* rhs, double* result, HYPRE_ParVector &par_b, HYPRE_ParVector &par_x, HYPRE_Solver &precond)
    {
        double pre_loop_time;
        double bi_prod, eps, gamma, old_gamma;

        double* r;
        double* p;
        double* z;
        double* z2;
        double* buffer;

        CHECK_CUDA(cudaMalloc(&r, sparse_A.rows() * sizeof(double)));
        CHECK_CUDA(cudaMemset(r, 0, sparse_A.rows() * sizeof(double)));

        CHECK_CUDA(cudaMalloc(&p, sparse_A.rows() * sizeof(double)));
        CHECK_CUDA(cudaMemset(p, 0, sparse_A.rows() * sizeof(double)));

        CHECK_CUDA(cudaMalloc(&z, sparse_A.rows() * sizeof(double)));
        CHECK_CUDA(cudaMemset(z, 0, sparse_A.rows() * sizeof(double)));

        CHECK_CUDA(cudaMalloc(&z2, sparse_A.rows() * sizeof(double)));
        CHECK_CUDA(cudaMemset(z2, 0, sparse_A.rows() * sizeof(double)));

        CHECK_CUDA(cudaMalloc(&buffer, sparse_A.rows() * sizeof(double)));
        CHECK_CUDA(cudaMemset(buffer, 0, sparse_A.rows() * sizeof(double)));

        {
            POLYSOLVE_SCOPED_STOPWATCH("pre loop time: ", pre_loop_time, *logger);
        
            bi_prod = dot(rhs, rhs);
            logger->trace("GPUHybrid solver bi prod: {}", bi_prod);

            if (bi_prod > 0.0)
            {
                eps = conv_tol_ * conv_tol_;
            }
            else 
            {
                CHECK_CUDA(cudaMemset(result, 0, sparse_A.rows() * sizeof(double)));
                num_iterations = 0;
                final_res_norm = 0;
                return;
            }

            Eigen::VectorXd A_times_result;
            matmul(result, buffer);

            
            vector_copy(rhs, r);
            vector_add(-1.0, buffer, r);

            custom_mixed_precond_iter(precond, r, z, buffer, z2);

            vector_copy(z, p);

            gamma = dot(r, z);
            old_gamma = gamma;
        }

        for (int k = 0; k < max_iter_; ++k)
        {
            POLYSOLVE_SCOPED_STOPWATCH("main loop time: ", loop_time, *logger);
            num_iterations = k + 1;

            matmul(p, buffer);
            double sdotp = dot(p, buffer);

            if (sdotp == 0.0)
            {
                logger->debug("GPUHybrid solver error: zero sdotp value");
                break;
            }

            double alpha = gamma / sdotp;

            if (alpha <= 0.0)
            {
                logger->debug("GPUHybrid solver error: negative or zero alpha value. gamma: {}, sdotp: {}", gamma, sdotp);
                break;
            } 
            else if (alpha < __DBL_MIN__)
            {
                logger->debug("GPUHybrid solver error: subnormal alpha value");
                break;
            }

            vector_add(alpha, p, result);
            vector_add(-1.0 * alpha, buffer, r);
            
            double drob2 = alpha * alpha * dot(p, p);
            if (!use_absolute_tol) 
            {
                drob2 /= bi_prod;
            }

            if (drob2 < conv_tol_ * conv_tol_)
            {
                logger->debug("GPUHybrid solver converged: change in residual too small");
                //break;
            }

            double i_prod = dot(r, r);
            logger->trace("GPUHybrid solver i prod: {}", i_prod);
            if (!use_absolute_tol) 
            {
                i_prod /= bi_prod;
            }

            if (i_prod < eps)
            {
                logger->debug("GPUHybrid solver converged: residual too small");
                break;
            }

            CHECK_CUDA(cudaMemset(z, 0, sparse_A.rows() * sizeof(double)));
            custom_mixed_precond_iter(precond, r, z, buffer, z2);

            gamma = dot(r, z);
            
            double beta = gamma / old_gamma;
            old_gamma = gamma;

            vector_scale(beta, p);
            vector_add(1.0, z, p);
        }
    }

    GPUHybridSolver::~GPUHybridSolver()
    {
        if (has_matrix_)
        {
            HYPRE_IJMatrixDestroy(A);
            has_matrix_ = false;
        }
        free_device_memory();
        if (cudss_handle) {
            cudssDestroy(cudss_handle);
            cudss_handle = nullptr;
        }
    }

    void GPUHybridSolver::free_device_memory() 
    {
        // Destroy cuDSS Opaque Structures
        if (batchMatrixA) { CHECK_CUDSS(cudssMatrixDestroy(batchMatrixA)); batchMatrixA = nullptr; }
        if (batchMatrixX) { CHECK_CUDSS(cudssMatrixDestroy(batchMatrixX)); batchMatrixX = nullptr; }
        if (batchMatrixB) { CHECK_CUDSS(cudssMatrixDestroy(batchMatrixB)); batchMatrixB = nullptr; }
        if (solverData)   { CHECK_CUDSS(cudssDataDestroy(cudss_handle, solverData)); solverData = nullptr; }
        if (config)       { CHECK_CUDSS(cudssConfigDestroy(config)); config = nullptr; }

        if (d_all_rowOffsets)     { CHECK_CUDA(cudaFree(d_all_rowOffsets)); d_all_rowOffsets = nullptr; }
        if (d_all_colIndices)     { CHECK_CUDA(cudaFree(d_all_colIndices)); d_all_colIndices = nullptr; }
        if (d_all_values)         { CHECK_CUDA(cudaFree(d_all_values)); d_all_values = nullptr; }
        
        if (d_csrRowOffsets_void) { CHECK_CUDA(cudaFree(d_csrRowOffsets_void)); d_csrRowOffsets_void = nullptr; }
        if (d_csrColIndices_void) { CHECK_CUDA(cudaFree(d_csrColIndices_void)); d_csrColIndices_void = nullptr; }
        if (d_csrValues_void)     { CHECK_CUDA(cudaFree(d_csrValues_void)); d_csrValues_void = nullptr; }
        if (d_x_void)             { CHECK_CUDA(cudaFree(d_x_void)); d_x_void = nullptr; }
        if (d_b_void)             { CHECK_CUDA(cudaFree(d_b_void)); d_b_void = nullptr; }

        // Free GPU Data Arrays
        if (d_x)             { cudaFree(d_x); d_x = nullptr; }
        if (d_b)             { cudaFree(d_b); d_b = nullptr; }
    }
}