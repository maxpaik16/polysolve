
#include <fstream>
#include <iostream>
#include <string>
#include <ctime>
//#include <format>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <stdlib.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

#include "linear/Solver.hpp"
#include "Utils.hpp"

#include <mpi.h>
#include <HYPRE_struct_ls.h>


void usage(const std::string &executable)
{
    std::cout << "Usage: " << executable << " solver A_file b_file param_file num_trials [ind_file]" << std::endl;
}


int main(int argc, char **argv)
{

    if (argc != 6 && argc != 7)
    {
        const std::string executable(argv[0]);
        usage(executable);
        return 1;
    }

    int done_already;

    MPI_Initialized(&done_already);
    int myid = 0, num_procs = 1;
    if (!done_already)
    {
        // Initialize MPI 
        MPI_Init(&argc, &argv);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Create logger
    const std::string logger_name = "solver-" + std::to_string(myid);
    std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt(logger_name);
    if (myid == 0)
    {
        logger->set_level(spdlog::level::trace);
    } 
    else
    {
        logger->set_level(spdlog::level::off);
    }

    const char *num_threads_c_str = std::getenv("OMP_NUM_THREADS");
    const std::string num_threads_str(
        num_threads_c_str ? num_threads_c_str : "1"
    );
    const int num_threads = std::stoi(num_threads_str);
    Eigen::setNbThreads(num_threads);

    const std::string solver_str(argv[1]);
    const std::string A_file(argv[2]);
    const std::string b_file(argv[3]);
    const std::string param_file(argv[4]);
    const std::string num_files_str(argv[5]);

    std::string ind_file = "NONE";
    if (argc == 7)
    {
        ind_file = argv[6];
    }

    const int num_trials = std::stoi(num_files_str);

    auto solver = polysolve::linear::Solver::create(solver_str, "");
    solver->logger = logger.get();

    std::ifstream param_file_stream(param_file);
    polysolve::json params = polysolve::json::parse(param_file_stream);
    solver->set_parameters(params);

    Eigen::SparseMatrix<double> A;
    Eigen::VectorXd b;
    int rows, cols, nnzs;
    std::vector<std::set<int>> bad_indices;

    if (myid == 0)
    {
        logger->info("Starting run with {} trials", num_trials);

        logger->info("Solver: {}", solver_str);
        logger->info("Params set by {}", param_file);
        logger->info("Matrix: {}", A_file);
        logger->trace("Num threads: {}", Eigen::nbThreads());

        logger->info("Loading problem...");

        double load_problem_time;
        {
            POLYSOLVE_SCOPED_STOPWATCH("load problem time", load_problem_time, *logger);

            if (!Eigen::loadMarket(A, A_file)) {
                logger->error("Could not load Matrix Market file");
                return 1;
            }

            std::vector<Eigen::Triplet<double>> triplets;
            triplets.reserve(2 * A.nonZeros());

            for (int k = 0; k < A.outerSize(); ++k)
            {
                for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it)
                {   
                    triplets.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
                    if (it.col() != it.row())
                    {
                        triplets.push_back(Eigen::Triplet<double>(it.col(), it.row(), it.value()));
                    }
                }
            }

            A.setFromTriplets(triplets.begin(), triplets.end());
            
            rows = A.rows();
            cols = A.cols();
            nnzs = A.nonZeros();

            logger->trace("Symmetry: {}", A.isApprox(A.transpose()));


            if (b_file != "NONE")
            {
                std::ifstream b_file_object(b_file);
                b.resize(rows);
                double val;
                int i = 0;
                while (b_file_object >> val)
                {
                    b(i++) = val;
                }
                b_file_object.close();
            }   

            if (ind_file != "NONE")
            {
                bad_indices.resize(1);
                
                std::ifstream file(ind_file);

                int n;
                while (file >> n)
                {
                    bad_indices[0].insert(n);
                }
                
                file.close();
            }
        }

        logger->trace("Problem size: {}, nnzs: {}", rows, nnzs);
    }

    if (solver_str != "Experimental")
    {
        if (myid == 0)
        {
            Eigen::VectorXd x(rows);
            logger->info("Starting solve...");

            for (int solve_i = 0; solve_i < num_trials; ++solve_i)
            {
                double solve_time;
                x.setZero();
                if (b_file == "NONE")
                {
                    b = Eigen::VectorXd::Random(rows);
                }
                {
                    POLYSOLVE_SCOPED_STOPWATCH("total solve time", solve_time, *logger);
                    solver->analyze_pattern(A, A.rows());
                    solver->factorize(A);
                    solver->solve(b, x);
                    logger->trace("Residual: {}", (b - A*x).norm());
                }

            }   
            logger->flush();
            MPI_Abort(MPI_COMM_WORLD, 0);
            int finalized;
            MPI_Finalized(&finalized);
            if (!finalized)
                MPI_Finalize();
            HYPRE_Finalize();
            
            return 0;
        }
        else
        {
            while (true) { 
                 struct timespec ts = {1, 0}; // 1 second
                 nanosleep(&ts, NULL); 
             }
        }
    }

    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nnzs, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int start_factorize = 0, start_solve = 0;
    Eigen::VectorXd x(rows);

    if (myid != 0)
    {
        while (true)
        {
            MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&nnzs, 1, MPI_INT, 0, MPI_COMM_WORLD);

            A.resize(rows, cols);
            A.reserve(nnzs);

            MPI_Bcast(A.valuePtr(), nnzs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(A.innerIndexPtr(), nnzs, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(A.outerIndexPtr(), rows + 1, MPI_INT, 0, MPI_COMM_WORLD);

            MPI_Bcast(&start_factorize, 1, MPI_INT, 0, MPI_COMM_WORLD);
            solver->factorize(A);
            b.resize(rows);
            x.resize(rows); 
            MPI_Bcast(b.data(), rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(x.data(), rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&start_solve, 1, MPI_INT, 0, MPI_COMM_WORLD);
            solver->solve(b, x);
        }
    }

    logger->info("Starting solve 2...");

    solver->set_problematic_dofs(bad_indices);

    for (int solve_i = 0; solve_i < num_trials; ++solve_i)
    {
        double solve_time;
        x.setZero();
        if (b_file == "NONE")
        {
            b = Eigen::VectorXd::Random(rows);
        }
        {
            POLYSOLVE_SCOPED_STOPWATCH("total solve time", solve_time, *logger);
            solver->analyze_pattern(A, A.rows());
            solver->factorize(A);
            solver->solve(b, x);
            logger->trace("Residual: {}", (b - A*x).norm());
        }

    }   

    logger->flush();
	MPI_Abort(MPI_COMM_WORLD, 0);
	int finalized;
    MPI_Finalized(&finalized);
    if (!finalized)
        MPI_Finalize();
	HYPRE_Finalize();
    
    return 0;
}