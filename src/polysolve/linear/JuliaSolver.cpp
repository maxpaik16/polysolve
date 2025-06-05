
////////////////////////////////////////////////////////////////////////////////
#include "JuliaSolver.hpp"

#include <fstream>
#include <cstdlib>
#include <iomanip>
////////////////////////////////////////////////////////////////////////////////

namespace polysolve::linear
{

    ////////////////////////////////////////////////////////////////////////////////

    JuliaSolver::JuliaSolver()
    {
        
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
        params["num_iterations"] = num_iterations;
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

        int julia_ret_code = system("julia --project dump_cudss.jl");

        std::ifstream solution_file("cudss_solution.txt");
        int i = 0;
        double value;
        while (solution_file >> std::setprecision(12) >> value)
        {
            result(i++) = value;
        }
        solution_file.close();
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


