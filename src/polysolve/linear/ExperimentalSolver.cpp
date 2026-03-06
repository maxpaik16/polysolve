
////////////////////////////////////////////////////////////////////////////////
#include "ExperimentalSolver.hpp"

#include "../Utils.hpp"

#ifdef POLYSOLVE_WITH_ICHOL
#include "cholesky.h"
#include "chol_hierarchy.h"
#endif
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <unordered_map>

#include <metis.h>

#include <HYPRE_utilities.h>

#include <Eigen/SparseCholesky>
#include <Eigen/PardisoSupport>

#include <SymEigsSolver.h>
#include <MatOp/SparseSymMatProd.h>

namespace polysolve::linear
{

    ////////////////////////////////////////////////////////////////////////////////

    ExperimentalSolver::ExperimentalSolver()
    {
        // set number of threads from OMP environment variable
        const char* num_threads_val = std::getenv("OMP_NUM_THREADS");
        if (num_threads_val)
        {
            num_threads = std::stoi(num_threads_val);
        }

#ifdef HYPRE_WITH_MPI
        // check if MPI is initialized
        int done_already;
        MPI_Initialized(&done_already);

        if (!done_already)
        {
            // Initialize MPI 
            int argc = 1;
            char name[] = "";
            char *argv[] = {name};
            char **argvv = &argv[0];
            MPI_Init(&argc, &argvv);
            MPI_Comm_rank(MPI_COMM_WORLD, &myid);
            MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        }

        // get MPI rank information
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        Eigen::setNbThreads(1);
        
#else
        Eigen::setNbThreads(num_threads);
#endif
    }

    // Set solver parameters
    void ExperimentalSolver::set_parameters(const json &params)
    {
        if (params.contains("Experimental"))
        {
            if (params["Experimental"].contains("max_iter"))
            {
                max_iter_ = params["Experimental"]["max_iter"];
            }
            if (params["Experimental"].contains("pre_max_iter"))
            {
                pre_max_iter_ = params["Experimental"]["pre_max_iter"];
            }
            if (params["Experimental"].contains("tolerance"))
            {
                conv_tol_ = params["Experimental"]["tolerance"];
            }
            if (params["Experimental"].contains("theta"))
            {
                theta = params["Experimental"]["theta"];
            }
            if (params["Experimental"].contains("nodal_coarsening"))
            {
                nodal_coarsening = params["Experimental"]["nodal_coarsening"];
            }
            if (params["Experimental"].contains("interp_rbms"))
            {
                interp_rbms = params["Experimental"]["interp_rbms"];
            }
            if (params["Experimental"].contains("dimension"))
            {
                dimension_ = params["Experimental"]["dimension"];
            }
            if (params["Experimental"].contains("do_mixed_precond"))
            {
                do_mixed_precond = params["Experimental"]["do_mixed_precond"];
            }
            if (params["Experimental"].contains("print_subdomain_conditioning"))
            {
                print_subdomain_conditioning = params["Experimental"]["print_subdomain_conditioning"];
            }
            if (params["Experimental"].contains("print_conditioning"))
            {
                print_conditioning = params["Experimental"]["print_conditioning"];
            }
            if (params["Experimental"].contains("dss_in_middle"))
            {
                dss_in_middle = params["Experimental"]["dss_in_middle"];
            }
            if (params["Experimental"].contains("use_absolute_tol"))
            {
                use_absolute_tol = params["Experimental"]["use_absolute_tol"];
            }
            if (params["Experimental"].contains("select_bad_dofs_from_rhs"))
            {
                select_bad_dofs_from_rhs = params["Experimental"]["select_bad_dofs_from_rhs"];
            }
            if (params["Experimental"].contains("select_bad_dofs_from_row_norms"))
            {
                select_bad_dofs_from_row_norms = params["Experimental"]["select_bad_dofs_from_row_norms"];
            }
            if (params["Experimental"].contains("select_bad_dofs_from_diag"))
            {
                select_bad_dofs_from_diag = params["Experimental"]["select_bad_dofs_from_diag"];
            }
            if (params["Experimental"].contains("select_bad_dofs_from_amg"))
            {
                select_bad_dofs_from_amg = params["Experimental"]["select_bad_dofs_from_amg"];
            }
            if (params["Experimental"].contains("bad_dof_threshold"))
            {
                bad_dof_threshold = params["Experimental"]["bad_dof_threshold"];
            }
            if (params["Experimental"].contains("max_bad_dof_threshold"))
            {
                max_bad_dof_threshold = params["Experimental"]["max_bad_dof_threshold"];
            }
#ifdef POLYSOLVE_WITH_ICHOL
            if (params["Experimental"].contains("use_incomplete_cholesky_precond"))
            {
                use_incomplete_cholesky_precond = params["Experimental"]["use_incomplete_cholesky_precond"];
            }
            if (params["Experimental"].contains("rho"))
            {
                rho = params["Experimental"]["rho"];
            }
#endif
            if (params["Experimental"].contains("save_selection_criteria"))
            {
                save_selection_criteria = params["Experimental"]["save_selection_criteria"];
            }
            if (params["Experimental"].contains("save_problem"))
            {
                save_problem = params["Experimental"]["save_problem"];
            }
            if (params["Experimental"].contains("save_selected_indices"))
            {
                save_selected_indices = params["Experimental"]["save_selected_indices"];
            }
            if (params["Experimental"].contains("use_gmres"))
            {
                use_gmres = params["Experimental"]["use_gmres"];
            }
            if (params["Experimental"].contains("use_minres"))
            {
                use_minres = params["Experimental"]["use_minres"];
            }
            if (params["Experimental"].contains("m"))
            {
                m_ = params["Experimental"]["m"];
            }
            if (params["Experimental"].contains("amg_iters"))
            {
                amg_iters = params["Experimental"]["amg_iters"];
            }
            if (params["Experimental"].contains("project_d_option"))
            {
                project_d_option = params["Experimental"]["project_d_option"];
            }
            if (params["Experimental"].contains("problematic_subdomain_precond_only"))
            {
                problematic_subdomain_precond_only = params["Experimental"]["problematic_subdomain_precond_only"];
            }
            if (params["Experimental"].contains("jacobi_precond"))
            {
                jacobi_precond = params["Experimental"]["jacobi_precond"];
            }
            if (params["Experimental"].contains("use_problematic_subdomain_for_initial_guess"))
            {
                use_problematic_subdomain_for_initial_guess = params["Experimental"]["use_problematic_subdomain_for_initial_guess"];
            }
            if (params["Experimental"].contains("decompose_subdomains"))
            {
                decompose_subdomains = params["Experimental"]["decompose_subdomains"];
            }
            if (params["Experimental"].contains("adapt_bad_dof_threshold"))
            {
                adapt_bad_dof_threshold = params["Experimental"]["adapt_bad_dof_threshold"];
            }
            if (params["Experimental"].contains("bad_dof_threshold_inc_factor"))
            {
                bad_dof_threshold_inc_factor = params["Experimental"]["bad_dof_threshold_inc_factor"];
            }
            if (params["Experimental"].contains("adaptive_max_iters"))
            {
                adaptive_max_iters = params["Experimental"]["adaptive_max_iters"];
            }
            if (params["Experimental"].contains("min_subdomain_size"))
            {
                min_subdomain_size = params["Experimental"]["min_subdomain_size"];
            }
            if (params["Experimental"].contains("max_subdomain_size"))
            {
                max_subdomain_size = params["Experimental"]["max_subdomain_size"];
            }
            if (params["Experimental"].contains("jacobi_precondition_system"))
            {
                jacobi_precondition_system = params["Experimental"]["jacobi_precondition_system"];
            }
            if (params["Experimental"].contains("find_threshold_from_secant"))
            {
                find_threshold_from_secant = params["Experimental"]["find_threshold_from_secant"];
            }
        }
    }

    void ExperimentalSolver::check_settings() const
    {
        if (myid != 0)
        {
            return;
        }

        if (use_gmres || use_minres)
        {
            log_and_throw_error(*logger, "Only PCG has been thoroughly tested!");
        }
        if (use_incomplete_cholesky_precond && num_procs != 1)
        {
            log_and_throw_error(*logger, "Incomplete cholesky preconditioning incompatible with multiple MPI ranks!");
        }
        if (decompose_subdomains && print_conditioning)
        {
            log_and_throw_error(*logger, "Matrix conditioning check requires non-decomposed subdomain!");
        }
        if (problematic_subdomain_precond_only)
        {
            log_and_throw_error(*logger, "Problematic subdomain preconditioner only option has not been thoroughly tested!");
        }
        if (!dss_in_middle)
        {
            log_and_throw_error(*logger, "Only DSS in middle has been thoroughly tested!");
        }
        if (project_d_option != 0)
        {
            log_and_throw_error(*logger, "Selected relatively untested subdomain projection option ({})!", project_d_option);
        }
        if (adapt_bad_dof_threshold)
        {
            log_and_throw_error(*logger, "Adaptive bad DOF thresholding has been deprecated!");
        }
        if (select_bad_dofs_from_rhs)
        {
            log_and_throw_error(*logger, "Selecting bad DOFs from the RHS has been deprecated!");
        }
        if (select_bad_dofs_from_amg)
        {
            log_and_throw_error(*logger, "Selecting bad DOFs from AMG has been deprecated!");
        }
    }

    void ExperimentalSolver::get_info(json &params) const
    {
        params["num_iterations"] = num_iterations;
        params["final_res_norm"] = final_res_norm;
    }

    ////////////////////////////////////////////////////////////////////////////////

    void ExperimentalSolver::analyze_pattern(const StiffnessMatrix &A, const int precond_num) 
    { 
        check_settings();
    }

    void ExperimentalSolver::factorize(const StiffnessMatrix &Ain)
    {
        logger->trace("Num Threads for ExperimentalSolver: {}", num_threads);
        logger->trace("Eigen num threads: {}", Eigen::nbThreads());
        logger->trace("World size: {}, myid: {}", num_procs, myid);

        if (myid == 0)
        {
            {
                POLYSOLVE_SCOPED_STOPWATCH("eigen matrix copy time", eigen_copy_time, *logger);
                sparse_A = Ain;
            }

            if (do_mixed_precond || use_problematic_subdomain_for_initial_guess || problematic_subdomain_precond_only)
            {
                prepare_dss();
            }
            else
            {
                P.resize(sparse_A.rows());
                P.setIdentity();
                P_T = P.transpose();
                partition_ranks();
                sparse_A = P * sparse_A * P_T;
                scatter_matrix();
            }
        }
        else
        {   
            if (do_mixed_precond || use_problematic_subdomain_for_initial_guess || problematic_subdomain_precond_only)
            {
                prepare_dss();
            }
            else
            {
                scatter_matrix();
            }
        }

#ifdef HYPRE_WITH_MPI
        if (jacobi_precondition_system)
        {
            logger->trace("Using Jacobi preconditioned system. Diagonal info: min: {}, max: {}", sparse_A.diagonal().minCoeff(), sparse_A.diagonal().maxCoeff());
            Eigen::VectorXd sqrt_diag(sparse_A.rows());
            sqrt_diag.setZero();

            if (myid == 0)
            {
                sqrt_diag = sparse_A.diagonal().cwiseSqrt();
            }
            MPI_Allreduce(MPI_IN_PLACE, sqrt_diag.data(), sqrt_diag.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            sqrt_diag_A_inv = sqrt_diag.asDiagonal().inverse();
            sparse_A = sqrt_diag_A_inv * sparse_A * sqrt_diag_A_inv;
        }
#endif

#ifdef POLYSOLVE_WITH_ICHOL
        setup_ichol_precond();
#endif

        if (has_matrix_)
        {
            POLYSOLVE_SCOPED_STOPWATCH("matrix destroy time", matrix_destroy_time, *logger);
            HYPRE_IJMatrixDestroy(A);
            has_matrix_ = false;
        }

        if (save_problem && myid == 0)
        {
            save_problem_to_file("A.mat");
        }
        has_matrix_ = true;

        if (jacobi_precond)
        {
            diag_inv = sparse_A.diagonal().segment(starts[myid], ends[myid] - starts[myid] + 1).asDiagonal().inverse();
        }

        copy_matrix_to_hypre();
    }

    ////////////////////////////////////////////////////////////////////////////////

    namespace
    {

        void eigen_to_hypre_par_vec(HYPRE_ParVector &par_x, HYPRE_IJVector &ij_x, const Eigen::VectorXd &x, int start_i, int end_i)
        {


    #ifdef HYPRE_WITH_MPI
            HYPRE_IJVectorSetValues(ij_x, end_i - start_i + 1, nullptr, x.data() + start_i);
    #else
            HYPRE_IJVectorSetValues(ij_x, x.size(), nullptr, x.data());
    #endif

            HYPRE_IJVectorAssemble(ij_x);
            HYPRE_IJVectorGetObject(ij_x, (void **)&par_x);
        }

        void hypre_vec_to_eigen(const HYPRE_IJVector &ij_x, Eigen::Ref<Eigen::VectorXd> x, int start_i, int end_i, int num_procs)
        {
    #ifdef HYPRE_WITH_MPI
            x.setZero();
            HYPRE_IJVectorGetValues(ij_x, end_i - start_i + 1, nullptr, x.data() + start_i);
            
            std::vector<int> recv_counts(num_procs);
            std::vector<int> displs(num_procs);

            int local_size = end_i - start_i + 1;
            MPI_Allgather(&local_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

            displs[0] = 0;
            for (int i = 1; i < num_procs; ++i) {
                displs[i] = displs[i - 1] + recv_counts[i - 1];
            }

            MPI_Allgatherv(x.data() + start_i, local_size, MPI_DOUBLE,
                        x.data(), recv_counts.data(), displs.data(), 
                        MPI_DOUBLE, MPI_COMM_WORLD);
    #else
            HYPRE_IJVectorGetValues(ij_x, x.size(), nullptr, x.data());
    #endif            
        }

        void calculate_rbms(Eigen::VectorXd &rbm_xy, Eigen::VectorXd &rbm_zx, Eigen::VectorXd &rbm_yz, const Eigen::MatrixXd &positions, const int dim)
        {
            rbm_xy.resize(positions.size());
            rbm_xy.setZero();
            
            if (dim == 3)
            {
                rbm_zx.resize(positions.size());
                rbm_yz.resize(positions.size());
            
                rbm_zx.setZero();
                rbm_yz.setZero();
            }

            for (int i = 0; i < positions.rows(); ++i)
            {
                rbm_xy(0 + i*dim) = positions(i, 1);
                rbm_xy(1 + i*dim) = -1 * positions(i, 0);

                if (dim == 3)
                {
                    rbm_zx(1 + i*dim) = positions(i, 2);
                    rbm_zx(2 + i*dim) = -1 * positions(i, 1);

                    rbm_yz(2 + i*dim) = positions(i, 0);
                    rbm_yz(0 + i*dim) = -1 * positions(i, 2);
                }
            }
        }

        void HypreBoomerAMG_SetDefaultOptions(HYPRE_Solver &amg_precond)
        {
            // AMG coarsening options:
            int coarsen_type = 10; // 10 = HMIS, 8 = PMIS, 6 = Falgout, 0 = CLJP
            int agg_levels = 1;    // number of aggressive coarsening levels
            double theta = 0.25;   // strength threshold: 0.25, 0.5, 0.8

            // AMG interpolation options:
            int interp_type = 6; // 6 = extended+i, 0 = classical
            int Pmax = 4;        // max number of elements per row in P

            // AMG relaxation options:
            int relax_type = 8;   // 8 = l1-GS, 6 = symm. GS, 3 = GS, 18 = l1-Jacobi
            int relax_sweeps = 1; // relaxation sweeps on each level

            // Additional options:
            int print_level = 0; // print AMG iterations? 1 = no, 2 = yes
            int max_levels = 25; // max number of levels in AMG hierarchy

            int min_coarse_size = 5;

            HYPRE_BoomerAMGSetCoarsenType(amg_precond, coarsen_type);
            HYPRE_BoomerAMGSetAggNumLevels(amg_precond, agg_levels);
            HYPRE_BoomerAMGSetRelaxType(amg_precond, relax_type);
            
            //relax_type = 88;
            HYPRE_BoomerAMGSetMinCoarseSize(amg_precond, min_coarse_size);
            //HYPRE_BoomerAMGSetCycleRelaxType(amg_precond, relax_type, 1);
            //HYPRE_BoomerAMGSetCycleRelaxType(amg_precond, relax_type, 2);
            HYPRE_BoomerAMGSetCycleRelaxType(amg_precond, relax_type, 3);
            //HYPRE_BoomerAMGSetDebugFlag(amg_precond, 1);
            //HYPRE_BoomerAMGSetNodal(amg_precond, 0);
            //HYPRE_BoomerAMGSetNodalDiag(amg_precond, 0);
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

        void HypreBoomerAMG_SetElasticityOptions(HYPRE_Solver &amg_precond, int dim, double theta, bool nodal_coarsening, bool interp_rbms, const Eigen::MatrixXd &positions, std::vector<int> &dof_to_function, std::vector<HYPRE_IJVector> &rbms, std::vector<HYPRE_ParVector> &par_rbms)
        {
            // Make sure the systems AMG options are set
            HYPRE_BoomerAMGSetNumFunctions(amg_precond, dim);

            //HYPRE_BoomerAMGSetDofFunc(amg_precond, (HYPRE_Int*) dof_to_function.data());

            // More robust options with respect to convergence
            HYPRE_BoomerAMGSetAggNumLevels(amg_precond, 0);
            HYPRE_BoomerAMGSetStrongThreshold(amg_precond, theta);

            // Nodal coarsening options (nodal coarsening is required for this solver)
            // See hypre's new_ij driver and the paper for descriptions.
            int nodal = 4;        // strength reduction norm: 1, 3 or 4
            int nodal_diag = 1;   // diagonal in strength matrix: 0, 1 or 2
            int relax_coarse = 8; // smoother on the coarsest grid: 8, 99 or 29

            // Elasticity interpolation options
            int interp_vec_variant = 2;    // 1 = GM-1, 2 = GM-2, 3 = LN
            int q_max = 4;                 // max elements per row for each Q
            int smooth_interp_vectors = 1; // smooth the rigid-body modes?

            // Optionally pre-process the interpolation matrix through iterative weight
            // refinement (this is generally applicable for any system)
            int interp_refine = 1;

            if (nodal_coarsening) 
            {
                HYPRE_BoomerAMGSetNodal(amg_precond, nodal);
                HYPRE_BoomerAMGSetNodalDiag(amg_precond, nodal_diag);
                HYPRE_BoomerAMGSetCycleRelaxType(amg_precond, relax_coarse, 3);
            }

            if (interp_rbms)
            {
                if (dim != 2 && dim != 3)
                {
                    assert(false);
                }

                HYPRE_BoomerAMGSetInterpVecVariant(amg_precond, interp_vec_variant);
                HYPRE_BoomerAMGSetInterpVecQMax(amg_precond, q_max);

                // HYPRE_BoomerAMGSetSmoothInterpVectors(amg_precond, smooth_interp_vectors);
                // HYPRE_BoomerAMGSetInterpRefine(amg_precond, interp_refine);

                Eigen::VectorXd rbm_xy, rbm_zx, rbm_yz;
                calculate_rbms(rbm_xy, rbm_zx, rbm_yz, positions, dim);

                eigen_to_hypre_par_vec(par_rbms[0], rbms[0], rbm_xy, 0, positions.rows() - 1);
                if (dim == 3)
                {
                    eigen_to_hypre_par_vec(par_rbms[1], rbms[1], rbm_zx, 0, positions.rows() - 1);
                    eigen_to_hypre_par_vec(par_rbms[2], rbms[2], rbm_yz, 0, positions.rows() - 1);
                }
            
                HYPRE_BoomerAMGSetInterpVectors(amg_precond, par_rbms.size(), &(par_rbms[0]));
            }
        }

    } // anonymous namespace


    ////////////////////////////////////////////////////////////////////////////////

    void ExperimentalSolver::solve(const Eigen::Ref<const VectorXd> rhs, Eigen::Ref<VectorXd> result)
    {
        assert(result.size() == rhs.size());
        Eigen::VectorXd remapped_rhs; 
        Eigen::VectorXd remapped_result; 

#ifdef HYPRE_WITH_MPI
    
        {
            POLYSOLVE_SCOPED_STOPWATCH("copy problem to other ranks time", copy_to_ranks_time, *logger);
            int problem_size;
            if (myid == 0)
            {
                remapped_rhs = P * rhs;
                remapped_result = P * result;
                problem_size = remapped_rhs.size();
            }
            MPI_Bcast(&problem_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (myid != 0)
            {
                remapped_rhs.resize(problem_size);
                remapped_result.resize(problem_size); 
            }
            MPI_Bcast(remapped_rhs.data(), problem_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(remapped_result.data(), problem_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
#endif
        z1.resize(remapped_rhs.size());
        z2.resize(remapped_rhs.size());
        z3.resize(remapped_rhs.size());

        if (jacobi_precondition_system)
        {
            remapped_rhs = sqrt_diag_A_inv * remapped_rhs;
        }

#ifdef POLYSOLVE_WITH_ICHOL
        if (use_incomplete_cholesky_precond)
        {
            for (int i = 0; i < rhs.size(); ++i)
            {
                remapped_rhs(i) = rhs(remap_dof(i));
                remapped_result(i) = result(remap_dof(i));
            }
        }
#endif

        if (save_problem && myid == 0)
        {
            std::ofstream file("rhs.mat", std::ios_base::app);
            file << remapped_rhs.transpose();
            file << std::endl;
            file.close();
        }

        HYPRE_ParVector par_b;
        HYPRE_ParVector par_x;
        init_hypre_vectors();

        {
            POLYSOLVE_SCOPED_STOPWATCH("copy x and b", copy_b_and_x_time, *logger);
            eigen_to_hypre_par_vec(par_b, ij_b, remapped_rhs, starts[myid], ends[myid]);
            eigen_to_hypre_par_vec(par_x, ij_x, remapped_result, starts[myid], ends[myid]);
        }

        /* AMG preconditioner */
        HYPRE_Solver precond;

        /* Now set up the AMG preconditioner and specify any parameters */
        HYPRE_BoomerAMGCreate(&precond);

        const int num_rbms = dimension_ == 2 ? 1 : 3;
        std::vector<HYPRE_ParVector> par_rbms(num_rbms);
        std::vector<HYPRE_IJVector> rbms(num_rbms);
        
        {
            POLYSOLVE_SCOPED_STOPWATCH("set AMG options", set_options_time, *logger);
            HypreBoomerAMG_SetDefaultOptions(precond);
            if (dimension_ > 1)
            {
                HypreBoomerAMG_SetElasticityOptions(
                    precond, 
                    dimension_, 
                    theta, 
                    nodal_coarsening, 
                    interp_rbms, 
                    positions_,
                    dof_to_function_, 
                    rbms, 
                    par_rbms
                );
            }

            HYPRE_BoomerAMGSetMaxIter(precond, amg_iters);

#ifdef HYPRE_WITH_MPI
            MPI_Barrier(MPI_COMM_WORLD);
#endif
        }

        {
            POLYSOLVE_SCOPED_STOPWATCH("AMG setup time", amg_setup_time, *logger);
            HYPRE_BoomerAMGSetup(precond, parcsr_A, par_b, par_x);
        }

        /* Now setup and solve! */
        {
            POLYSOLVE_SCOPED_STOPWATCH("actual solve time", actual_solve_time, *logger);

            if (use_minres)
            {
                minres_solve(remapped_rhs, remapped_result, par_b, par_x, precond);
            } 
            else if (use_gmres)
            {
                gmres_solve(remapped_rhs, remapped_result, par_b, par_x, precond);
            } 
            else
            {
                pcg_solve(remapped_rhs, remapped_result, par_b, par_x, precond);
            }

            Eigen::VectorXd A_times_result;
            matmul(remapped_result, sparse_A, A_times_result);
            sync_vector(A_times_result);
            sync_vector(remapped_rhs);
            final_res_norm = (remapped_rhs - A_times_result).norm();
            
        }

        logger->debug("Experimental solver Iterations: {}", num_iterations);
        logger->debug("Experimental solver Final Relative Residual Norm: {}", final_res_norm);

        sync_vector(remapped_result);
        result = remapped_result;
        recover_solution(result);

        /* Destroy preconditioner */
        {
            POLYSOLVE_SCOPED_STOPWATCH("destroy time", destroy_time, *logger);
            HYPRE_BoomerAMGDestroy(precond);
            HYPRE_IJVectorDestroy(ij_x);
            HYPRE_IJVectorDestroy(ij_b);
        }
    }

    void ExperimentalSolver::pcg_solve(Eigen::VectorXd &rhs, Eigen::VectorXd &result, HYPRE_ParVector &par_b, HYPRE_ParVector &par_x, HYPRE_Solver &precond)
    {
#ifdef HYPRE_WITH_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        double pre_loop_time;
        double bi_prod, eps, gamma, old_gamma;

        if (use_problematic_subdomain_for_initial_guess)
        {
            if (myid == 0)
            {
                Eigen::VectorXd temp(result.size());
                temp.setZero();
                dss_precond_iter(temp, rhs, result);
            }
#ifdef HYPRE_WITH_MPI
            MPI_Bcast(result.data(), result.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
        }

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
#ifdef HYPRE_WITH_MPI
                MPI_Barrier(MPI_COMM_WORLD);
#endif
                return;
            }

            Eigen::VectorXd A_times_result;
            matmul(result, sparse_A, A_times_result);
            r += rhs - A_times_result;

            p.resize(r.size());
            z.resize(r.size());
            p.setZero();
            z.setZero();

#ifdef POLYSOLVE_WITH_ICHOL
            if (use_incomplete_cholesky_precond)
            {
                POLYSOLVE_SCOPED_STOPWATCH("ichol time: ", ichol_time, *logger);
                z = inc_chol_precond->solve(r);
            } else
#endif
            {
                custom_mixed_precond_iter(precond, r, z);
            }
            
            
            p = z;

            gamma = dot(r, z);
            old_gamma = gamma;
        }

        for (int k = 0; k < max_iter_; ++k)
        {

            if ((problematic_subdomain_precond_only || do_mixed_precond) && adapt_bad_dof_threshold && bad_dof_threshold < max_bad_dof_threshold && k >= adaptive_max_iters)
            {
                bad_dof_threshold *= bad_dof_threshold_inc_factor;
                bad_dof_threshold = std::min(bad_dof_threshold, max_bad_dof_threshold);
                logger->trace("Restarting with new bad dof threshold: {}", bad_dof_threshold);
                sync_vector(rhs);
                sync_vector(result);
                prepare_dss();
                pcg_solve(rhs, result, par_b, par_x, precond);
                return;
            }
            
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
                check_smallest_eigenvalue();
                break;
            } 
            else if (alpha < __DBL_MIN__)
            {
                logger->debug("Experimental solver error: subnormal alpha value");
                break;
            }

            result += alpha * p;
            r -= alpha * A_times_p;
            //r = rhs - (sparse_A * result);
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

#ifdef POLYSOLVE_WITH_ICHOL
            if (use_incomplete_cholesky_precond)
            {
                POLYSOLVE_SCOPED_STOPWATCH("ichol time: ", ichol_time, *logger);
                z = inc_chol_precond->solve(r);
            } else
#endif
            {
                custom_mixed_precond_iter(precond, r, z);
            }

            gamma = dot(r, z);
            double beta = gamma / old_gamma;
            old_gamma = gamma;

            p = z + beta*p;
        }
#ifdef HYPRE_WITH_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }

    void ExperimentalSolver::minres_solve(Eigen::VectorXd &rhs, Eigen::VectorXd &result, HYPRE_ParVector &par_b, HYPRE_ParVector &par_x, HYPRE_Solver &precond)
    {
        log_and_throw_error(*logger, "MINRES not yet implemented!");
    }

    void ExperimentalSolver::gmres_solve(Eigen::VectorXd &rhs, Eigen::VectorXd &result, HYPRE_ParVector &par_b, HYPRE_ParVector &par_x, HYPRE_Solver &precond)
    {
        log_and_throw_error(*logger, "GMRES not yet implemented!");
    }

    void ExperimentalSolver::custom_mixed_precond_iter(const HYPRE_Solver &precond, Eigen::VectorXd &r, Eigen::VectorXd &z)
    {
        z1.setZero();
        z2.setZero();
        z3.setZero();

        if (problematic_subdomain_precond_only)
        {
            if (bad_indices_.size() > 0 && bad_indices_[0].size() > 0)
            {
                Eigen::VectorXd z0(r.size());
                z0.setZero();
                dss_precond_iter(z0, r, z1);
                z = z1;
            }
            else 
            {
                z = r;
            }
            return;
        }

        assert(bad_indices_.size() == 1);
        if (!do_mixed_precond || bad_indices_.size() == 0 || bad_indices_[0].size() == 0)
        {
            amg_precond_iter(precond, r, z1);
            z = z1;
            return;
        }

        if (dss_in_middle)
        {
            amg_precond_iter(precond, r, z1);
            dss_precond_iter(z1, r, z2);
            Eigen::VectorXd A_times_z2;
            matmul(z2, sparse_A, A_times_z2);
            Eigen::VectorXd curr_r = r - A_times_z2;
            amg_precond_iter(precond, curr_r, z3);
            z = z2 + z3;
        }
        else
        {
            Eigen::VectorXd z0(r.size());
            z0.setZero();
            dss_precond_iter(z0, r, z1);
            Eigen::VectorXd A_times_z1;
            matmul(z2, sparse_A, A_times_z1);
            Eigen::VectorXd curr_r = r - A_times_z1;
            amg_precond_iter(precond, curr_r, z2);
            z2 += z1;
            dss_precond_iter(z2, r, z);
        }

    }

    void ExperimentalSolver::amg_precond_iter(const HYPRE_Solver &precond, Eigen::VectorXd& eigen_b, Eigen::VectorXd &eigen_x)
    {

        if (jacobi_precond)
        {
            POLYSOLVE_SCOPED_STOPWATCH("jacobi time: ", jacobi_time, *logger);
            eigen_x.segment(starts[myid], ends[myid] - starts[myid] + 1) = diag_inv * eigen_b.segment(starts[myid], ends[myid] - starts[myid] + 1);
#ifdef HYPRE_WITH_MPI
            // all_gather_vec(local_result, eigen_x);
#endif
            return;
        }

        HYPRE_ParVector par_x;
        HYPRE_ParVector par_b;
#ifdef HYPRE_WITH_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif

        HYPRE_IJVectorSetData(ij_x, eigen_x.data() + starts[myid]);
        HYPRE_IJVectorSetData(ij_b, eigen_b.data() + starts[myid]);
        HYPRE_IJVectorAssemble(ij_x);
        HYPRE_IJVectorGetObject(ij_x, (void **)&par_x);
        HYPRE_IJVectorAssemble(ij_b);
        HYPRE_IJVectorGetObject(ij_b, (void **)&par_b);
        

        {
            POLYSOLVE_SCOPED_STOPWATCH("boomeramg solve time: ", solve_time, *logger);
            HYPRE_BoomerAMGSolve(precond, parcsr_A, par_b, par_x);
        }
        
    }

    void ExperimentalSolver::dss_precond_iter(Eigen::VectorXd &z, Eigen::VectorXd &r, Eigen::VectorXd &next_z)
    {
        {
            POLYSOLVE_SCOPED_STOPWATCH("dss step time: ", dss_step_time, *logger);

            double sync_time;
            if (num_procs > 1)
            {
                POLYSOLVE_SCOPED_STOPWATCH("dss sync time: ", sync_time, *logger);
                sync_vector(z);
            }

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
                        //logger->trace("A row sum: {}", sparse_A.row(subdomain[i]).cwiseAbs().sum());
                        //logger->trace("Subdomain {}, dof {}, rhs value: {}, mystart/end: {}/{}", index, subdomain[i], sub_rhs(index_mappings[index_counter][subdomain[i]]), starts[myid], ends[myid]);
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
                MPI_Barrier(MPI_COMM_WORLD);
            }

            double sum_time;
            {
                POLYSOLVE_SCOPED_STOPWATCH("dss sum time: ", sum_time, *logger);
                next_z.segment(starts[myid], ends[myid] - starts[myid] + 1) += z.segment(starts[myid], ends[myid] - starts[myid] + 1);
            }
        }
    }

#ifdef POLYSOLVE_WITH_ICHOL
    int ExperimentalSolver::remap_dof(const int index)
    {
        int nod_index = index / dimension_;
        int func_offset = index % dimension_;
        return dimension_ * ichol_dof_remapping(nod_index) + func_offset;
    }

    void ExperimentalSolver::setup_ichol_precond()
    {
        double ichol_fac_time;
        if (use_incomplete_cholesky_precond)
        {
            POLYSOLVE_SCOPED_STOPWATCH("ichol factorization time", ichol_fac_time, *logger);
            pt.put<double>("nei_num.value", rho);
            pt.put<double>("alpha.value", 1e-4);
            pt.put<std::ptrdiff_t>("max_su_size.value", 64);
            pt.put<int>("num_threads.value", num_threads);
            pt.put<int>("subst_num_threads.value", num_threads);
            
            Eigen::Matrix<size_t, -1, -1> test_elements = elements_.cast<size_t>();
            mschol::chol_hierarchy builder(test_elements.transpose(), positions_.transpose(), positions_.cols() == 2 ? "trig" : "tets");
            
            std::vector<std::shared_ptr<mschol::chol_level>> levels;
            builder.build(levels, 125, dimension_);
            builder.get_dof_remapping(ichol_dof_remapping);

            std::vector<Eigen::Triplet<double>> triplets;
            triplets.reserve(sparse_A.nonZeros());

            Eigen::VectorXi old_to_new(ichol_dof_remapping.size());
            for (int i = 0; i < ichol_dof_remapping.size(); ++i)
            {
                old_to_new(ichol_dof_remapping[i]) = i; 
            }

            for (int k = 0; k < sparse_A.outerSize(); ++k)
            {
                for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(sparse_A, k); it; ++it)
                {   
                    int nod_index_i = it.row() / dimension_;
                    int func_offset_i = it.row() % dimension_;
                    int new_i = dimension_ * old_to_new(nod_index_i) + func_offset_i;

                    int nod_index_j = it.col() / dimension_;
                    int func_offset_j = it.col() % dimension_;
                    int new_j = dimension_ * old_to_new(nod_index_j) + func_offset_j;
                    
                    triplets.push_back(Eigen::Triplet<double>(new_i, new_j, it.value()));
                }
            }

            sparse_A.setFromTriplets(triplets.begin(), triplets.end());

            inc_chol_precond = std::make_shared<mschol::ichol_precond>(levels, pt);
            inc_chol_precond->analyse_pattern(sparse_A);
            inc_chol_precond->factorize(sparse_A);
        }
    }
#endif

    void ExperimentalSolver::select_bad_indices()
    {
        POLYSOLVE_SCOPED_STOPWATCH("bad dof selection time", bad_dof_selection_time, *logger);
        Eigen::VectorXd sq_mags(sparse_A.rows());
        double cutoff_threshold; 

        if (!(select_bad_dofs_from_amg || select_bad_dofs_from_rhs || select_bad_dofs_from_row_norms || select_bad_dofs_from_diag))
        {
            return; 
        }

        if (select_bad_dofs_from_amg + select_bad_dofs_from_rhs + select_bad_dofs_from_row_norms + select_bad_dofs_from_diag > 1)
        {
            logger->warn("Multiple selection methods specified, defaulting to row norms.");   
        }

        bad_indices_.clear();
        bad_indices_.resize(1);

        if (select_bad_dofs_from_rhs)
        {
            POLYSOLVE_SCOPED_STOPWATCH("select dofs from rhs", select_dofs_from_rhs_time, *logger);
            log_and_throw_error(*logger, "Selection from rhs not implemented!");
        }

        if (select_bad_dofs_from_amg)
        {
            POLYSOLVE_SCOPED_STOPWATCH("select dofs from amg", select_dofs_from_amg_time, *logger);
            HYPRE_ParVector test_par_b, test_par_x;
            HYPRE_IJVector test_x, test_b;
            HYPRE_IJVectorCreate(MPI_COMM_WORLD, starts[myid], ends[myid], &test_b);
            HYPRE_IJVectorSetObjectType(test_b, HYPRE_PARCSR);
            HYPRE_IJVectorInitialize(test_b);
            HYPRE_IJVectorCreate(MPI_COMM_WORLD, starts[myid], ends[myid], &test_x);
            HYPRE_IJVectorSetObjectType(test_x, HYPRE_PARCSR);
            HYPRE_IJVectorInitialize(test_x);

            Eigen::VectorXd test_result = Eigen::VectorXd::Random(sparse_A.rows());
            Eigen::VectorXd start_result = test_result;
            Eigen::VectorXd test_rhs(sparse_A.rows());
            test_rhs.setZero();

            eigen_to_hypre_par_vec(test_par_b, test_b, test_rhs, starts[myid], ends[myid]);
            eigen_to_hypre_par_vec(test_par_x, test_x, test_result, starts[myid], ends[myid]);

            HYPRE_Solver test_precond;
            HYPRE_BoomerAMGCreate(&test_precond);
            HypreBoomerAMG_SetDefaultOptions(test_precond);
            if (dimension_ > 1)
            {
                const int num_rbms = dimension_ == 2 ? 1 : 3;
                std::vector<HYPRE_ParVector> par_rbms(num_rbms);
                std::vector<HYPRE_IJVector> rbms(num_rbms);
                HypreBoomerAMG_SetElasticityOptions(
                    test_precond, 
                    dimension_, 
                    theta, 
                    nodal_coarsening, 
                    interp_rbms, 
                    positions_,
                    dof_to_function_, 
                    rbms, 
                    par_rbms
                );
            }
            HYPRE_BoomerAMGSetMaxIter(test_precond, 5);
            HYPRE_BoomerAMGSetup(test_precond, parcsr_A, test_par_b, test_par_x);
            HYPRE_BoomerAMGSolve(test_precond, parcsr_A, test_par_b, test_par_x);

            hypre_vec_to_eigen(test_x, test_result, starts[myid], ends[myid], num_procs);

            //assert(rhs.size() % dimension_ == 0);
            for (int i = 0; i < sparse_A.rows(); ++i)
            {
                sq_mags(i) = abs(test_result(i) / start_result(i));
            }
            
            HYPRE_IJVectorDestroy(test_x);
            HYPRE_IJVectorDestroy(test_b);
            HYPRE_BoomerAMGDestroy(test_precond);
        }

        if (select_bad_dofs_from_diag)
        {
            POLYSOLVE_SCOPED_STOPWATCH("select dofs from hess diagonal", select_dofs_from_diag_time, *logger);
            //assert(rhs.size() % dimension_ == 0);
            sq_mags = sparse_A.diagonal().cwiseAbs();
        
        }

        if (select_bad_dofs_from_row_norms)
        {
            POLYSOLVE_SCOPED_STOPWATCH("select dofs from hess row norms", select_dofs_from_row_norms_time, *logger);
            //assert(rhs.size() % dimension_ == 0);
            for (int i = 0; i < sparse_A.rows(); ++i)
            {
                sq_mags(i) = sparse_A.row(i).norm();
            }
        }

        if (save_selection_criteria)
        {
            std::ofstream file;
            file.open("criteria.txt", std::ios_base::app);
            file << sq_mags.transpose() << std::endl;
            file.close();
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

        int cutoff_index;
        if (find_threshold_from_secant)
        {
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
        }
        else
        {
            cutoff_index = indices.size() * (1 - bad_dof_threshold);
        }

        const double cutoff = sorted_sq_mags(cutoff_index);
        logger->trace("Problematic threshold: {}, cutoff: {}", cutoff, cutoff_index);

        for (int i = cutoff_index; i < indices.size(); ++i)
        {
            bad_indices_[0].insert(indices[i]);
        }

        if (save_selected_indices)
        {
            std::ofstream file;
            file.open("selected_indices.txt", std::ios_base::app);
            if (bad_indices_.size() > 0)
            {
                for (auto i : bad_indices_[0])
                {
                    file << i << " ";
                }
            }
            file << std::endl;;
            file.close();
        }

    }

    void ExperimentalSolver::factorize_submatrix()
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

            if (print_subdomain_conditioning)
            {
                Eigen::EigenSolver<Eigen::MatrixXd> es(D);
                auto abs_evs = es.eigenvalues().cwiseAbs();
                logger->trace("Condition number of subdomain: {}", abs_evs.maxCoeff() / abs_evs.minCoeff());
            }

            project_D(D);

            {
                POLYSOLVE_SCOPED_STOPWATCH("factorize D", dss_factorization_time, *logger);
                D_solvers[i_counter]->compute(D);
            }

            ++i_counter;
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        
    }

    void ExperimentalSolver::matmul(Eigen::VectorXd &x, Eigen::SparseMatrix<double, Eigen::RowMajor> &A, Eigen::VectorXd &result)
    {
        POLYSOLVE_SCOPED_STOPWATCH("matmul time", matmul_time, *logger);
#ifdef HYPRE_WITH_MPI
        if (num_procs == 1)
        {
            result = A * x;
        }
        else
        {            
            result.resize(x.size());
            result.setZero();
            HYPRE_ParVector par_x;
            HYPRE_ParVector par_result;
            HYPRE_IJVectorSetData(ij_x, x.data() + starts[myid]);
            HYPRE_IJVectorAssemble(ij_x);
            HYPRE_IJVectorGetObject(ij_x, (void **)&par_x);
            HYPRE_IJVectorSetData(ij_b, result.data() + starts[myid]);
            HYPRE_IJVectorAssemble(ij_b);
            HYPRE_IJVectorGetObject(ij_b, (void **)&par_result);
            HYPRE_ParCSRMatrixMatvec(1.0, parcsr_A, par_x, 0.0, par_result);
        }
#else
        result = A*x;
#endif
    }

    double ExperimentalSolver::dot(Eigen::VectorXd &a, Eigen::VectorXd &b)
    {
#ifdef HYPRE_WITH_MPI
        HYPRE_ParVector par_a;
        HYPRE_ParVector par_b;
        HYPRE_IJVectorSetData(ij_x, a.data() + starts[myid]);
        HYPRE_IJVectorAssemble(ij_x);
        HYPRE_IJVectorGetObject(ij_x, (void **)&par_a);
        HYPRE_IJVectorSetData(ij_b, b.data() + starts[myid]);
        HYPRE_IJVectorAssemble(ij_b);
        HYPRE_IJVectorGetObject(ij_b, (void **)&par_b);
        double result;
        HYPRE_ParVectorInnerProd(par_a, par_b, &result);
        return result;
#else
        result = a.dot(b);
#endif
        return result;
    }

    void ExperimentalSolver::prepare_dss()
    {
        POLYSOLVE_SCOPED_STOPWATCH("prepare dss time", prepare_dss_time, *logger);
#ifdef HYPRE_WITH_MPI
        if (myid == 0) 
        {
#endif
            select_bad_indices();

            if (save_selected_indices)
            {
                std::ofstream file;
                file.open("selected_indices.txt", std::ios_base::app);
                if (bad_indices_.size() > 0)
                {
                    for (auto i : bad_indices_[0])
                    {
                        file << i << " ";
                    }
                }
                file << std::endl;;
                file.close();
            }
#ifdef HYPRE_WITH_MPI
        }
#endif

        if (myid == 0)
        {
            if (decompose_subdomains)
            {
                decompose_subdomains_to_disjoint_subsets();
                load_balance_subdomains();
                compute_permutation_matrix();
            }
            else
            {
                load_balance_subdomains();
                P.resize(sparse_A.rows());
                P.setIdentity();
                P_T = P.transpose();
                partition_ranks();
            }
            sparse_A = P * sparse_A * P_T;
            /*
            std::ofstream file;
            file.open("remapped_A.txt");
            for (int k = 0; k < sparse_A.outerSize(); ++k)
            {
                for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(sparse_A, k); it; ++it)
                {   
                    file << it.row() + 1 << " " << it.col() + 1 << " " << it.value() << std::endl;
                }
            }
            file.close();
            std::ofstream P_file;
            P_file.open("P.txt");
            for (int i = 0; i < P.indices().size(); ++i)
            {
                P_file << i << " " << P.indices()[i] << std::endl;  
            }*/
        }

        scatter_matrix();
        share_bad_subdomains();
        factorize_submatrix();

        if (myid == 0 && print_conditioning)
        {
            POLYSOLVE_SCOPED_STOPWATCH("print conditioning time", print_cond_time, *logger);
            check_matrix_conditioning("Hessian", sparse_A);
            check_matrix_conditioning("Preconditioned Hessian", bad_indices_[0]);
        }
    }

    void ExperimentalSolver::compute_permutation_matrix()
    {
        P.resize(sparse_A.rows());

        std::vector<bool> good_dofs(sparse_A.rows(), true);
        for (auto &index_set : bad_indices_)
        {
            for (auto index : index_set)
            {
                good_dofs[index] = false;
            }
        }

        int target_size = sparse_A.rows() / num_procs;
        starts.clear();
        ends.clear();
        int current_index = 0;
        int current_global_index = 0;
        for (int i = 0; i < num_procs; ++i)
        {
            starts.push_back(current_index);

            for (auto bs_i : bad_subdomain_assignments[i])
            {
                for (auto index : bad_indices_[bs_i])
                {
                    P.indices()[index] = current_index;
                    ++current_index;
                }
            }

            int target_end = (i == num_procs - 1) ? sparse_A.rows() : starts.back() + std::max(target_size, current_index - starts.back()) - 1;
            ends.push_back(std::min(target_end, static_cast<int>(sparse_A.rows()) - 1));

            while (current_index <= ends.back())
            {
                if (good_dofs[current_global_index])
                {
                    P.indices()[current_global_index] = current_index;
                    ++current_index;
                }
                ++current_global_index;
            }
        }

        P_T = P.transpose();
        std::vector<std::set<int>> remapped_bad_indices;
        for (auto &subdomain : bad_indices_)
        {
            remapped_bad_indices.emplace_back();
            for (auto index : subdomain)
            {
                remapped_bad_indices.back().insert(P.indices()[index]);
            }
        }
        bad_indices_ = std::move(remapped_bad_indices);
    }

    void ExperimentalSolver::check_matrix_conditioning(const std::string name, const std::set<int>& subdomain)
    {
        if (subdomain.size() == 0)
        {
            return;
        }

        Eigen::MatrixXd preconditioned_A = sparse_A;

        for (int col = 0; col < preconditioned_A.cols(); ++col)
        {
            Eigen::VectorXd sub_rhs;
            Eigen::VectorXd sub_result;
            sub_rhs.resize(subdomain.size());
            sub_result.resize(subdomain.size());

            int i_counter = 0;
            for (auto &i : subdomain)
            {
                sub_rhs(i_counter) = preconditioned_A(i, col);
                ++i_counter;
            }

            sub_result = D_solvers[0]->solve(sub_rhs);
            i_counter = 0;
            for (auto &i : subdomain)
            {
                preconditioned_A(i, col) = sub_result(i_counter);
                ++i_counter;
            }
        }

        check_matrix_conditioning(name, preconditioned_A);
    }

    void ExperimentalSolver::GeneratePlaneRotation(double &dx, double &dy, double &cs, double &sn)
    {
        if (dy == 0.0) {
            cs = 1.0;
            sn = 0.0;
        } else if (abs(dy) > abs(dx)) {
            double temp = dx / dy;
            sn = 1.0 / sqrt( 1.0 + temp*temp );
            cs = temp * sn;
        } else {
            double temp = dy / dx;
            cs = 1.0 / sqrt( 1.0 + temp*temp );
            sn = temp * cs;
        }
    }

    void ExperimentalSolver::ApplyPlaneRotation(double &dx, double &dy, double &cs, double &sn)
    {
        double temp  =  cs * dx + sn * dy;
        dy = -sn * dx + cs * dy;
        dx = temp;
    }

    void ExperimentalSolver::Update(Eigen::VectorXd &x, int k, Eigen::MatrixXd &h, Eigen::VectorXd &s, Eigen::MatrixXd &v)
    {
        Eigen::VectorXd y = s;

        // Backsolve:  
        for (int i = k; i >= 0; i--) {
            y(i) /= h(i,i);
            for (int j = i - 1; j >= 0; j--)
                y(j) -= h(j,i) * y(i);
        }

        for (int j = 0; j <= k; j++)
            x += v.col(j) * y(j);
    }

    void ExperimentalSolver::all_gather_vec(const Eigen::VectorXd &local_part, Eigen::VectorXd &global_result)
    {
        std::vector<int> recv_counts(num_procs);
        std::vector<int> displs(num_procs);

        int local_size = ends[myid] - starts[myid] + 1;
        MPI_Allgather(&local_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        displs[0] = 0;
        for (int i = 1; i < num_procs; ++i) {
            displs[i] = displs[i - 1] + recv_counts[i - 1];
        }

        MPI_Allgatherv(local_part.data(), local_size, MPI_DOUBLE,
                    global_result.data(), recv_counts.data(), displs.data(), 
                    MPI_DOUBLE, MPI_COMM_WORLD);

    }

    void ExperimentalSolver::sync_vector(Eigen::VectorXd &vec)
    {
        Eigen::VectorXd local_part = vec.segment(starts[myid], ends[myid] - starts[myid] + 1);
        all_gather_vec(local_part, vec);
    }

    void ExperimentalSolver::check_matrix_conditioning(const std::string name, const Eigen::MatrixXd& mat)
    {
        Eigen::BDCSVD<Eigen::MatrixXd> svd(mat);
        double cond = svd.singularValues()(0) 
        / svd.singularValues()(svd.singularValues().size()-1);

        Eigen::LDLT<Eigen::MatrixXd> chol_decomp(mat);
        bool spd = !(chol_decomp.info() == Eigen::NumericalIssue);
        bool symm = mat.isApprox(mat.transpose());
        bool isPos = chol_decomp.isPositive();

        logger->trace("Analyzing {}...", name);
        logger->trace("SPD: {}, Symm: {}, isPos: {}", spd, symm, isPos);
        logger->trace("Condition number: {}", cond);
    }

    void ExperimentalSolver::save_problem_to_file(const std::string& file_name)
    {
        logger->trace("Saving problem to {}", file_name);
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(sparse_A.nonZeros());

        for (int k = 0; k < sparse_A.outerSize(); ++k)
        {
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(sparse_A, k); it; ++it)
            {   
                triplets.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
            }
        }

        std::ofstream file(file_name, std::ios_base::app);
        file << sparse_A.rows() << " " << sparse_A.cols() << " " << sparse_A.nonZeros() << std::endl;
        for (auto &trip : triplets)
        {
            file << trip.row() << " " << trip.col() << " " << trip.value() << " ";
        }
        file << std::endl;
        file.close();
    }

    void ExperimentalSolver::partition_ranks()
    {
        starts.clear();
        ends.clear();
        int local_size = sparse_A.rows() / num_procs;
        for (int i = 0; i < num_procs; ++i)
        {
            starts.push_back(i == 0 ? 0 : local_size * i + i);
            ends.push_back(i == (num_procs - 1) ? sparse_A.rows() - 1 : starts.back() + local_size);
        }   
    }

    void ExperimentalSolver::scatter_matrix()
    {
        POLYSOLVE_SCOPED_STOPWATCH("scatter matrix time", scatter_matrix_time, *logger);
    #ifdef HYPRE_WITH_MPI
        if (myid == 0 && num_procs > 1)
        {
            std::vector<int> row_vec;
            std::vector<int> col_vec;
            std::vector<double> val_vec;
            std::vector<int> counts;

            MPI_Bcast(starts.data(), num_procs, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(ends.data(), num_procs, MPI_INT, 0, MPI_COMM_WORLD);

            std::vector<int> displs;
            for (int k = 0; k < sparse_A.outerSize(); ++k)
            {
                for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(sparse_A, k); it; ++it)
                {
                    if (displs.size() < starts.size() && it.row() == starts[displs.size()])
                    {
                        displs.push_back(row_vec.size());
                    }

                    row_vec.push_back(it.row());
                    col_vec.push_back(it.col());
                    val_vec.push_back(it.value());
                }
            }

            std::vector<int> sizes(num_procs);
            int proc_index = 0;
            sizes[0] = 0;
            sizes.back() = row_vec.size() - displs.back(); 
            for (int i = num_procs - 2; i > 0; --i)
            {
                sizes[i] = displs[i + 1] - displs[i];
            }            

            int problem_size = sparse_A.rows();
            MPI_Bcast(&problem_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            int root_size;
            MPI_Scatter(sizes.data(), 1, MPI_INT, &root_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Scatterv(row_vec.data(), sizes.data(), displs.data(), MPI_INT, nullptr, root_size, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Scatterv(col_vec.data(), sizes.data(), displs.data(), MPI_INT, nullptr, root_size, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Scatterv(val_vec.data(), sizes.data(), displs.data(), MPI_DOUBLE, nullptr, root_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        else if (myid > 0)
        {
            starts.clear();
            ends.clear();
            starts.resize(num_procs);
            ends.resize(num_procs);

            MPI_Bcast(starts.data(), num_procs, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(ends.data(), num_procs, MPI_INT, 0, MPI_COMM_WORLD);

            int problem_size;
            MPI_Bcast(&problem_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            int my_size;
            MPI_Scatter(nullptr, 1, MPI_INT, &my_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            
            std::vector<int> row_vec(my_size);
            std::vector<int> col_vec(my_size);
            std::vector<double> val_vec(my_size);
            MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, row_vec.data(), my_size, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, col_vec.data(), my_size, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE, val_vec.data(), my_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            std::vector<Eigen::Triplet<double>> triplets;
            triplets.reserve(my_size);
            for (int i = 0; i < my_size; ++i)
            {
                triplets.push_back(Eigen::Triplet<double>(row_vec[i], col_vec[i], val_vec[i]));
            }

            sparse_A.resize(problem_size, problem_size);
            sparse_A.setFromTriplets(triplets.begin(), triplets.end());
        }
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        logger->trace("start {}, end {}", starts[myid], ends[myid]);
        local_result.resize(ends[myid] - starts[myid] + 1);
    }

    void ExperimentalSolver::copy_matrix_to_hypre()
    {
        POLYSOLVE_SCOPED_STOPWATCH("copy matrix time", matrix_copy_time, *logger);

#ifdef HYPRE_WITH_MPI
        HYPRE_IJMatrixCreate(MPI_COMM_WORLD, starts[myid], ends[myid], starts[myid], ends[myid], &A);
#else
        HYPRE_IJMatrixCreate(0, 0, sparse_A.rows() - 1, 0, sparse_A.cols() - 1, &A);
#endif
        HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
        HYPRE_IJMatrixInitialize(A);

        for (HYPRE_Int k = starts[myid]; k <= ends[myid]; ++k)
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

    void ExperimentalSolver::init_hypre_vectors()
    {
#ifdef HYPRE_WITH_MPI
        HYPRE_IJVectorCreate(MPI_COMM_WORLD, starts[myid], ends[myid], &ij_x);
#else
        HYPRE_IJVectorCreate(0, 0, rhs.size() - 1, &x);
#endif
        HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(ij_x);
        HYPRE_IJVectorInitializeShell(ij_x);

#ifdef HYPRE_WITH_MPI
        HYPRE_IJVectorCreate(MPI_COMM_WORLD, starts[myid], ends[myid], &ij_b);
#else
        HYPRE_IJVectorCreate(0, 0, rhs.size() - 1, &ij_b);
#endif
        HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(ij_b);
        HYPRE_IJVectorInitializeShell(ij_b);
    }

    void ExperimentalSolver::recover_solution(Eigen::Ref<VectorXd> result)
    {
#ifdef POLYSOLVE_WITH_ICHOL
        if (use_incomplete_cholesky_precond)
        {
            for (int i = 0; i < result.size(); ++i)
            {
                result(remap_dof(i)) = result(i);
            }
        }
#endif
        if (jacobi_precondition_system)
        {
            result = sqrt_diag_A_inv * result;
        }

        result = P_T * result;
    }

    void ExperimentalSolver::check_smallest_eigenvalue()
    {
        return;
        if (myid == 0 && bad_indices_.size() > 0 && bad_indices_[0].size() > 0)
        {
            polysolve::StiffnessMatrix A_copy = sparse_A;
            Spectra::SparseSymMatProd<double> op(A_copy);
            Spectra::SymEigsSolver<double, Spectra::SMALLEST_ALGE, Spectra::SparseSymMatProd<double>> eigs(&op, 1, 6);

            eigs.init();
            int nconv = eigs.compute();
            Eigen::VectorXd eigenvalues;
            Eigen::MatrixXd eigenvectors;
            if (eigs.info() == Spectra::SUCCESSFUL)
            {
                eigenvalues = eigs.eigenvalues();
                eigenvectors = eigs.eigenvectors();

                Eigen::VectorXd bad_evec = eigenvectors.col(0);
                Eigen::VectorXd projected_bad_evec = bad_evec;
                for (int i = 0; i < projected_bad_evec.size(); ++i)
                {
                    if (bad_indices_[0].count(i) == 0)
                    {
                        projected_bad_evec(i) = 0;
                    }
                }
                logger->debug("eigenvalue found: {}, evec norm: {}, projected evec norm: {}", eigenvalues(0), bad_evec.norm(), projected_bad_evec.norm()); 
            }
            else
            {
                logger->debug("could not find eigenvalue");
            }
        }
    }

    void ExperimentalSolver::project_D(Eigen::SparseMatrix<double>& D)
    {
        POLYSOLVE_SCOPED_STOPWATCH("project D", d_projection_time, *logger);
        if (project_d_option == 1)
        {
            // make diagonally dominant row by row
            for (int ri = 0; ri < D.rows(); ++ri)
            {
                double row_sum = D.row(ri).cwiseAbs().sum();
                D.coeffRef(ri, ri) = row_sum;
            }
        }
        else if (project_d_option == 2 && D.rows() > 0)
        {
            Eigen::SelfAdjointEigenSolver<Eigen::SparseMatrix<double>>
                eigensolver(D);
            if (eigensolver.info() != Eigen::Success) {
                logger->trace("unable to project matrix onto positive definite cone");
            }
            // Check if all eigen values are positive.
            // The eigenvalues are sorted in increasing order.
            else if (eigensolver.eigenvalues()[0] <= 0.0) 
            {

                Eigen::DiagonalMatrix<double, Eigen::Dynamic> Diag(eigensolver.eigenvalues());
                for (int ri = 0; ri < Diag.rows(); ri++) {
                    if (Diag.diagonal()[ri] <= 0.0) {
                        Diag.diagonal()[ri] = 0;
                    } 
                }
                Eigen::MatrixXd dense_D = eigensolver.eigenvectors() * Diag * eigensolver.eigenvectors().transpose();
                D = dense_D.sparseView(); 
            }
        }
    }

    void ExperimentalSolver::assemble_D(int bad_i, int i, Eigen::SparseMatrix<double>& D)
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
            // logger->trace("D symmetric: {}", D.isApprox(D.transpose()));
        }
    }
    
    void ExperimentalSolver::build_index_mappings()
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

    void ExperimentalSolver::decompose_subdomains_to_disjoint_subsets()
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
        //bad_indices_.reserve(chosen_sets.size());

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

    void ExperimentalSolver::partition_subdomain(std::vector<int>& subdomain)
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
                if (it.row() >= it.col())
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
        for (int i = 0; i < new_num_subdomains; ++i)
        {
            bad_indices_.emplace_back();
        }

        for (int i = 0; i < subdomain.size(); ++i)
        {
            bad_indices_[part[i] + bad_indices_.size() - new_num_subdomains].insert(subdomain[i]);
        }
    }

    void ExperimentalSolver::share_bad_subdomains()
    {
#ifdef HYPRE_WITH_MPI
        int num_subdomains;
        if (myid == 0)
        {
            num_subdomains = bad_indices_.size();
            MPI_Bcast(&num_subdomains, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (bad_indices_.size() > 0)
            {
                for (int i = 0; i < num_subdomains; ++i)
                {
                    int num_indices = bad_indices_[i].size();
                    MPI_Bcast(&num_indices, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    std::vector<int> subdomain_vec;
                    for (auto index : bad_indices_[i])
                    {
                        subdomain_vec.push_back(index);
                    }
                    MPI_Bcast(subdomain_vec.data(), num_indices, MPI_INT, 0, MPI_COMM_WORLD);
                    MPI_Barrier(MPI_COMM_WORLD);
                }
            }
            for (int i = 0; i < num_procs; ++i)
            {
                int local_size = bad_subdomain_assignments[i].size();
                MPI_Bcast(&local_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(bad_subdomain_assignments[i].data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);
            }
        } 
        else
        {
            MPI_Bcast(&num_subdomains, 1, MPI_INT, 0, MPI_COMM_WORLD);
            bad_indices_.clear();
            bad_indices_.resize(num_subdomains);
            for (int i = 0; i < num_subdomains; ++i)
            {
                int num_indices;
                MPI_Bcast(&num_indices, 1, MPI_INT, 0, MPI_COMM_WORLD);
                std::vector<int> subdomain_vec;
                subdomain_vec.resize(num_indices);
                MPI_Bcast(subdomain_vec.data(), num_indices, MPI_INT, 0, MPI_COMM_WORLD);
                for (auto index : subdomain_vec)
                {
                    bad_indices_[i].insert(index);
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
            bad_subdomain_assignments.resize(num_procs);
            for (int i = 0; i < num_procs; ++i)
            {
                int local_size;
                MPI_Bcast(&local_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
                bad_subdomain_assignments[i].resize(local_size);
                MPI_Bcast(bad_subdomain_assignments[i].data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);
            }
        }
#endif

        bad_indices_arrays.clear();
        bad_indices_arrays.resize(bad_indices_.size());
        for (int i = 0; i < bad_indices_.size(); ++i)
        {
            bad_indices_arrays[i].reserve(bad_indices_[i].size());
            for (auto index : bad_indices_[i])
            {
                bad_indices_arrays[i].push_back(index);
            }
        }
    }

    void ExperimentalSolver::load_balance_subdomains()
    {
        bad_subdomain_assignments.clear();
        bad_subdomain_assignments.resize(num_procs);

        std::vector<std::pair<int, int>> subdomain_sizes;
        subdomain_sizes.reserve(bad_indices_.size());
        
        for (auto &subdomain : bad_indices_)
        {
            subdomain_sizes.push_back(std::make_pair(subdomain_sizes.size(), subdomain.size()));
        }

        std::sort(subdomain_sizes.begin(), subdomain_sizes.end(), [](const std::pair<int, int>& l, const std::pair<int, int>& r) {return l.second > r.second;});
        std::vector<int> assigned_sizes(num_procs, 0);

        logger->trace("Num subdomains: {}", bad_indices_.size());

        for (auto [i, size] : subdomain_sizes)
        {
            logger->trace("Subdomain size: {}", size);
            int min_size = assigned_sizes[0];
            int chosen_proc = 0;
            for (int pi = 1; pi < num_procs; ++pi)
            {
                if (assigned_sizes[pi] < min_size)
                {
                    min_size = assigned_sizes[pi];
                    chosen_proc = pi;
                }
            }
            bad_subdomain_assignments[chosen_proc].push_back(i);
            assigned_sizes[chosen_proc] += size;
        }
    }
    
    ////////////////////////////////////////////////////////////////////////////////

    ExperimentalSolver::~ExperimentalSolver()
    {
        if (has_matrix_)
        {
            HYPRE_IJMatrixDestroy(A);
            has_matrix_ = false;
        }
    }

} // namespace polysolve::linear

