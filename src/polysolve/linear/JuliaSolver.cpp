
////////////////////////////////////////////////////////////////////////////////
#include "JuliaSolver.hpp"

#include <fstream>
#include <cstdlib>
#include <iomanip>
#include <filesystem>
#include <unistd.h>
////////////////////////////////////////////////////////////////////////////////

namespace polysolve::linear
{

    ////////////////////////////////////////////////////////////////////////////////

    JuliaSolver::JuliaSolver()
    {
        julia_thread = std::thread(JuliaSolver::launch_julia_program);
    }

    // Set solver parameters
    void JuliaSolver::set_parameters(const json &params)
    {
        if (params.contains("Julia"))
        {
            if (params["Julia"].contains("options"))
            {
                options = params["Julia"]["options"];
            }
        }
    }

    void JuliaSolver::get_info(json &params) const
    {
        params["final_res_norm"] = final_res_norm;
    }

    ////////////////////////////////////////////////////////////////////////////////

    void JuliaSolver::factorize(const StiffnessMatrix &Ain)
    {
        A_ = Ain;
        has_matrix_ = true;
    }

    void JuliaSolver::solve(const Eigen::Ref<const VectorXd> rhs, Eigen::Ref<VectorXd> result)
    {
        std::ofstream options_file("cudss_options.txt");
        options_file << options;
        options_file.close();

        int nnz = A_.nonZeros();

        std::ofstream colptr_file("cudss_colptr.txt");
        for (int i = 0; i < rhs.size() + 1; ++i)
        {
            colptr_file << std::setprecision(12) << A_.outerIndexPtr()[i] << std::endl;
        }
        colptr_file.close();

        std::ofstream rowval_file("cudss_rowval.txt");
        for (int i = 0; i < nnz; ++i)
        {
            rowval_file << std::setprecision(12) << A_.innerIndexPtr()[i] << std::endl;
        }
        rowval_file.close();

        std::ofstream nzval_file("cudss_nzval.txt");
        for (int i = 0; i < nnz; ++i)
        {
            nzval_file << std::setprecision(12) << A_.valuePtr()[i] << std::endl;
        }
        nzval_file.close();

        std::ofstream rhs_file("cudss_rhs.txt");
        rhs_file << std::setprecision(12) << rhs;
        rhs_file.close();

        //TODO send solve signal to julia and wait for response
        std::ofstream start_file("cudss_start.txt");
        start_file.close();

        while (std::filesystem::exists("cudss_start.txt"))
        {
            sleep(0.01);
        }

        std::ifstream solution_file("cudss_solution.txt");
        int i = 0;
        double value;
        while (solution_file >> std::setprecision(12) >> value)
        {
            result(i++) = value;
        }
        solution_file.close();
    }

    void JuliaSolver::launch_julia_program()
    {
        int julia_ret_code = system("julia --project dump_cudss.jl");
    }

    ////////////////////////////////////////////////////////////////////////////////

    JuliaSolver::~JuliaSolver()
    {
        if (has_matrix_)
        {
            has_matrix_ = false;
        }
    }

} // namespace polysolve::linear


