
#include "GPUHybridSolver.hpp"
#include "../Utils.hpp"

#include <cuda_runtime.h>


namespace polysolve::linear
{
    GPUHybridSolver::GPUHybridSolver()
    {
        HYPRE_Init();
        precond_num_ = 0;
        cudaSetDevice(0);
        HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST);
        HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST);
        HYPRE_SetSpGemmUseCusparse(false);
        HYPRE_SetUseGpuRand(true);
    }

    void GPUHybridSolver::set_parameters(const json &params)
    {
        if (params.contains("GPUHybridSolver"))
        {
            if (params["GPUHybridSolver"].contains("max_iter"))
            {
                max_iter_ = params["GPUHybridSolver"]["max_iter"];
            }
            if (params["GPUHybridSolver"].contains("pre_max_iter"))
            {
                pre_max_iter_ = params["GPUHybridSolver"]["pre_max_iter"];
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
            if (params["Experimental"].contains("do_mixed_precond"))
            {
                do_mixed_precond = params["Experimental"]["do_mixed_precond"];
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
        }
    } 

    void GPUHybridSolver::check_settings() const
    {

    }

    void GPUHybridSolver::analyze_pattern(const StiffnessMatrix &A, const int precond_num)
    {
        check_settings();
    }

    void GPUHybridSolver::factorize(const StiffnessMatrix &A)
    {
        {
            POLYSOLVE_SCOPED_STOPWATCH("eigen matrix copy time", eigen_copy_time, *logger);
            sparse_A = Ain;
        }

        prepare_dss();

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
            int relax_type = 6;   // 8 = l1-GS, 6 = symm. GS, 3 = GS, 18 = l1-Jacobi
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
        HYPRE_ParVector par_b;
        HYPRE_ParVector par_x;
        init_hypre_vectors();

        // TODO copy x and b
        
        /* AMG preconditioner */
        HYPRE_Solver precond;

        /* Now set up the AMG preconditioner and specify any parameters */
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

            HYPRE_BoomerAMGSetMaxIter(precond, amg_iters);
        }

        {
            POLYSOLVE_SCOPED_STOPWATCH("AMG setup time", amg_setup_time, *logger);
            HYPRE_BoomerAMGSetup(precond, parcsr_A, par_b, par_x);
        }

        {
            POLYSOLVE_SCOPED_STOPWATCH("actual solve time", actual_solve_time, *logger);

            pcg_solve(remapped_rhs, remapped_result, par_b, par_x, precond);
    
            matmul(remapped_result, sparse_A, A_times_result);
            final_res_norm = (remapped_rhs - A_times_result).norm();
        }

        logger->debug("Experimental solver Iterations: {}", num_iterations);
        logger->debug("Experimental solver Final Relative Residual Norm: {}", final_res_norm);

        result = remapped_result;

        {
            POLYSOLVE_SCOPED_STOPWATCH("destroy time", destroy_time, *logger);
            HYPRE_BoomerAMGDestroy(precond);
            HYPRE_IJVectorDestroy(ij_x);
            HYPRE_IJVectorDestroy(ij_b);
        }
    }

    void GPUHybridSolver::copy_matrix_to_hypre()
    {
        POLYSOLVE_SCOPED_STOPWATCH("copy matrix time", matrix_copy_time, *logger);

        HYPRE_IJMatrixCreate(0, 0, sparse_A.rows() - 1, 0, sparse_A.cols() - 1, &A);
        HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
        HYPRE_IJMatrixInitialize(A);

        for (HYPRE_Int k = 0; k <= spares_A.rows(); ++k)
        {
            HYPRE_Int row[1]; 
            int counter = 0;
            std::vector<HYPRE_Int> cols;
            std::vector<double> vals;
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(sparse_A, k); it; ++it)
            {
                ++counter;
                row[0] = it.row();
                cols.push_back((HYPRE_Int)it.col());
                vals.push_back(it.value());
            }
            HYPRE_Int n_cols[1] = {counter};
            HYPRE_IJMatrixSetValues(A, 1, n_cols, row, cols.data(), vals.data());
        }
        HYPRE_IJMatrixAssemble(A);
        HYPRE_IJMatrixGetObject(A, (void **)&parcsr_A);
    }

    void GPUHybridSolver::init_hypre_vectors()
    {
        HYPRE_IJVectorCreate(0, 0, rhs.size() - 1, &x);
        HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(ij_x);
        HYPRE_IJVectorInitializeShell(ij_x);
        HYPRE_IJVectorCreate(0, 0, rhs.size() - 1, &ij_b);
        HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(ij_b);
        HYPRE_IJVectorInitializeShell(ij_b);
    }

    void GPUHybridSolver::matmul(Eigen::VectorXd &x, Eigen::SparseMatrix<double, Eigen::RowMajor> &A, Eigen::VectorXd &result)
    {
        POLYSOLVE_SCOPED_STOPWATCH("matmul time", matmul_time, *logger);     
        
        result.resize(x.size());
        result.setZero();
        HYPRE_ParVector par_x;
        HYPRE_ParVector par_result;
        HYPRE_IJVectorSetData(ij_x, x.data());
        HYPRE_IJVectorAssemble(ij_x);
        HYPRE_IJVectorGetObject(ij_x, (void **)&par_x);
        HYPRE_IJVectorSetData(ij_b, result.data());
        HYPRE_IJVectorAssemble(ij_b);
        HYPRE_IJVectorGetObject(ij_b, (void **)&par_result);
        HYPRE_ParCSRMatrixMatvec(1.0, parcsr_A, par_x, 0.0, par_result);
    }

    double GPUHybridSolver::dot(Eigen::VectorXd &x, Eigen::VectorXd &y)
    {
        HYPRE_ParVector par_a;
        HYPRE_ParVector par_b;
        HYPRE_IJVectorSetData(ij_x, a.data());
        HYPRE_IJVectorAssemble(ij_x);
        HYPRE_IJVectorGetObject(ij_x, (void **)&par_a);
        HYPRE_IJVectorSetData(ij_b, b.data());
        HYPRE_IJVectorAssemble(ij_b);
        HYPRE_IJVectorGetObject(ij_b, (void **)&par_b);
        double result;
        HYPRE_ParVectorInnerProd(par_a, par_b, &result);
        return result;
    }

    void GPUHybridSolver::custom_mixed_precond_iter(const HYPRE_Solver &precond, Eigen::VectorXd &r, Eigen::VectorXd &z)
    {
        Eigen::VectorXd z1(z.size());
        Eigen::VectorXd z2(z.size());
        Eigen::VectorXd z3(z.size());
        z1.setZero();
        z2.setZero();
        z3.setZero();

        assert(bad_indices_.size() == 1);
        if (!do_mixed_precond || bad_indices_.size() == 0 || bad_indices_[0].size() == 0)
        {
            amg_precond_iter(precond, r, z1);
            z = z1;
            return;
        }
        else
        {
            amg_precond_iter(precond, r, z1);
            dss_precond_iter(z1, r, z2);
            Eigen::VectorXd A_times_z2;
            matmul(z2, sparse_A, A_times_z2);
            Eigen::VectorXd curr_r = r - A_times_z2;
            amg_precond_iter(precond, curr_r, z3);
            z = z2 + z3;
        }

    }

    void GPUHybridSolver::dss_precond_iter(Eigen::VectorXd &z, Eigen::VectorXd &r, Eigen::VectorXd &next_z)
    {
        {
            POLYSOLVE_SCOPED_STOPWATCH("dss step time: ", dss_step_time, *logger);

            double work_time;
            {
                POLYSOLVE_SCOPED_STOPWATCH("dss work time: ", work_time, *logger);
                next_z.setZero();
            
                int index_counter = 0;
                for (int index : bad_subdomain_assignments[myid])
                {
                    auto &subdomain = bad_indices_arrays[index];
                    Eigen::VectorXd sub_rhs, sub_result;

                    {
                        sub_rhs.resize(subdomain.size());
                        sub_result.resize(subdomain.size());
                    }

                    for (int i = 0; i < subdomain.size(); ++i)
                    {
                        sub_rhs(index_mappings[index_counter][subdomain[i]]) = r(subdomain[i]) - sparse_A.row(subdomain[i]).dot(z);
                    }

                    {
                        sub_result = D_solvers[index_counter]->solve(sub_rhs);
                    }

                    for (int i = 0; i < subdomain.size(); ++i)
                    {
                        next_z(subdomain[i]) = sub_result(index_mappings[index_counter][subdomain[i]]);
                    }
                    ++index_counter;
                }
            }

            double sum_time;
            {
                POLYSOLVE_SCOPED_STOPWATCH("dss sum time: ", sum_time, *logger);
                next_z += z;
            }
        }
    }

    void GPUHybridSolver::amg_precond_iter(const HYPRE_Solver &precond, Eigen::VectorXd& b, Eigen::VectorXd &x)
    {
        HYPRE_ParVector par_x;
        HYPRE_ParVector par_b;

        HYPRE_IJVectorSetData(ij_x, eigen_x.data());
        HYPRE_IJVectorSetData(ij_b, eigen_b.data());
        HYPRE_IJVectorAssemble(ij_x);
        HYPRE_IJVectorGetObject(ij_x, (void **)&par_x);
        HYPRE_IJVectorAssemble(ij_b);
        HYPRE_IJVectorGetObject(ij_b, (void **)&par_b);
        
        {
            POLYSOLVE_SCOPED_STOPWATCH("boomeramg solve time: ", solve_time, *logger);
            HYPRE_BoomerAMGSolve(precond, parcsr_A, par_b, par_x);
        }

    }

    void GPUHybridSolver::prepare_dss()
    {
        POLYSOLVE_SCOPED_STOPWATCH("prepare dss time", prepare_dss_time, *logger);

        select_bad_indices();

        if (decompose_subdomains)
        {
            decompose_subdomains_to_disjoint_subsets();
        }

        factorize_submatrix();
    }

    void GPUHybridSolver::decompose_subdomains_to_disjoint_subsets()
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
                partition_subdomain(kv.second);
                continue;
            }
            bad_indices_.emplace_back(kv.second.begin(), kv.second.end());
        }
    }

    void GPUHybridSolver::build_index_mappings()
    {
        index_mappings.clear();
        index_mappings.resize(bad_subdomain_assignments[myid].size());

        int i_counter = 0;
        for (int i : bad_subdomain_assignments[myid])
        {
            int j_counter = 0;
            for (auto j : bad_indices_[i])
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
        Eigen::VectorXd sq_mags(sparse_A.rows());
        double cutoff_threshold; 

        bad_indices_.clear();
        bad_indices_.resize(1);

        {
            POLYSOLVE_SCOPED_STOPWATCH("select dofs from hess diagonal", select_dofs_from_diag_time, *logger);
            //assert(rhs.size() % dimension_ == 0);
            sq_mags = sparse_A.diagonal().cwiseAbs();
        }

        std::vector<int> indices(sq_mags.size());
        std::iota(indices.begin(), indices.end(), 0); 
        std::sort(indices.begin(), indices.end(), [&sq_mags](size_t i1, size_t i2) {
            return sq_mags(i1) < sq_mags(i2);
        });
        Eigen::VectorXd sorted_sq_mags(sq_mags.size());
        for (size_t i = 0; i < indices.size(); ++i)
        {            
            sorted_sq_mags(i) = sq_mags(indices[i]);
        }

        Eigen::VectorXd log_sorted = sorted_sq_mags.array().log();
        double min_mag = log_sorted.minCoeff();
        double max_mag = log_sorted.maxCoeff();
        int n = log_sorted.size();
        double max_deviation = 0;
        int max_deviation_index = 0;
        for (int i = 0; i < n; ++i)
        {
            double y = log_sorted(i);
            double deviation = abs((max_mag - min_mag) / (log_sorted.size()- 1) * i + min_mag - log_sorted(i));
            if (deviation > max_deviation)
            {
                max_deviation = deviation;
                max_deviation_index = i;
            }
        }
        cutoff_index = max_deviation_index;

        const double cutoff = sorted_sq_mags(cutoff_index);
        logger->trace("Problematic threshold: {}, cutoff: {}", cutoff, cutoff_index);

        for (int i = cutoff_index; i < indices.size(); ++i)
        {
            bad_indices_[0].insert(indices[i]);
        }
    }

    void GPUHybridSolver::factorize_submatrix()
    {
        POLYSOLVE_SCOPED_STOPWATCH("assemble D", dss_assembly_time, *logger);
        D_solvers.clear();

        for (int i : bad_subdomain_assignments[myid])
        {   
            if (bad_indices_[i].size() > 1000)
            {
                D_solvers.push_back(std::make_unique<EigenWrapper<Eigen::PardisoLDLT<Eigen::SparseMatrix<double>>>>());
            }
            else 
            {
                D_solvers.push_back(std::make_unique<EigenWrapper<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>>>());
            }
        }

        build_index_mappings();

        int i_counter = 0;
        for (int i : bad_subdomain_assignments[myid])
        {
            Eigen::SparseMatrix<double> D;
            assemble_D(i_counter, i, D);

            {
                POLYSOLVE_SCOPED_STOPWATCH("factorize D", dss_factorization_time, *logger);
                D_solvers[i_counter]->compute(D);
            }

            ++i_counter;
        }
        
    }

    void GPUHybridSolver::assemble_D(int bad_i, int i, Eigen::SparseMatrix<double>& D)
    {
        D.resize(bad_indices_[i].size(), bad_indices_[i].size());
        std::vector<Eigen::Triplet<double>> triplets;
        for (int k : bad_indices_[i])
        {
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(sparse_A, k); it; ++it)
            {
                auto ind_it = index_mappings[bad_i].find(it.col());
                if (ind_it != index_mappings[bad_i].end())
                {
                    triplets.push_back(Eigen::Triplet<double>(index_mappings[bad_i][k], index_mappings[bad_i][it.col()], it.value()));
                }
            }
        }

        {
            POLYSOLVE_SCOPED_STOPWATCH("set D from triplets", set_from_triplets_time, *logger);
            D.setFromTriplets(triplets.begin(), triplets.end());
        }
    }

    void GPUHybridSolver::pcg_solve(Eigen::VectorXd &rhs, Eigen::VectorXd &result, HYPRE_ParVector &par_b, HYPRE_ParVector &par_x, HYPRE_Solver &precond)
    {
        double pre_loop_time;
        double bi_prod, eps, gamma, old_gamma;

        Eigen::VectorXd r, p, z;
        r.resize(rhs.size());
        p.resize(rhs.size());
        z.resize(rhs.size());
        r.setZero();
        {
            POLYSOLVE_SCOPED_STOPWATCH("pre loop time: ", pre_loop_time, *logger);
        
            bi_prod = dot(rhs, rhs);
            logger->trace("Experimental solver bi prod: {}", bi_prod);

            if (bi_prod > 0.0)
            {
                eps = conv_tol_ * conv_tol_;
            }
            else 
            {
                result.setZero();
                num_iterations = 0;
                final_res_norm = 0;
                logger->debug("Experimental solver Iterations: {}", num_iterations);
                logger->debug("Experimental solver Final Relative Residual Norm: {}", final_res_norm);
                return;
            }

            Eigen::VectorXd A_times_result;
            matmul(result, sparse_A, A_times_result);
            r += rhs - A_times_result;

            p.resize(r.size());
            z.resize(r.size());
            p.setZero();
            z.setZero();

            custom_mixed_precond_iter(precond, r, z);

            p = z;

            gamma = dot(r, z);
            old_gamma = gamma;
        }

        for (int k = 0; k < max_iter_; ++k)
        {
            POLYSOLVE_SCOPED_STOPWATCH("main loop time: ", loop_time, *logger);
            num_iterations = k + 1;

            Eigen::VectorXd A_times_p;
            matmul(p, sparse_A, A_times_p);
            double sdotp = dot(p, A_times_p);

            if (sdotp == 0.0)
            {
                logger->debug("Experimental solver error: zero sdotp value");
                break;
            }

            double alpha = gamma / sdotp;

            if (alpha <= 0.0)
            {
                logger->debug("Experimental solver error: negative or zero alpha value. gamma: {}, sdotp: {}", gamma, sdotp);
                break;
            } 
            else if (alpha < __DBL_MIN__)
            {
                logger->debug("Experimental solver error: subnormal alpha value");
                break;
            }

            result += alpha * p;
            r -= alpha * A_times_p;
            double drob2 = alpha * alpha * dot(p, p);
            if (!use_absolute_tol) 
            {
                drob2 /= bi_prod;
            }

            if (drob2 < conv_tol_ * conv_tol_)
            {
                logger->debug("Experimental solver converged: change in residual too small");
                //break;
            }

            double i_prod = dot(r, r);
            logger->trace("Experimental solver i prod: {}", i_prod);
            if (!use_absolute_tol) 
            {
                i_prod /= bi_prod;
            }

            if (i_prod < eps)
            {
                logger->debug("Experimental solver converged: residual too small");
                break;
            }

            z.setZero(); 

            custom_mixed_precond_iter(precond, r, z);
            
            gamma = dot(r, z);
            double beta = gamma / old_gamma;
            old_gamma = gamma;

            p = z + beta*p;
        }
    }

}