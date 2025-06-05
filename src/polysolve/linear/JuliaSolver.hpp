#pragma once

////////////////////////////////////////////////////////////////////////////////
#include "Solver.hpp"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>


namespace polysolve::linear
{

    class JuliaSolver : public Solver
    {

    public:
        JuliaSolver();
        ~JuliaSolver();

    private:
        POLYSOLVE_DELETE_MOVE_COPY(JuliaSolver)

    public:
        //////////////////////
        // Public interface //
        //////////////////////

        // Set solver parameters
        virtual void set_parameters(const json &params) override;

        // Retrieve memory information from Pardiso
        virtual void get_info(json &params) const override;

        // Factorize system matrix
        virtual void factorize(const StiffnessMatrix &A) override;

        // Solve the linear system Ax = b
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;

        // Name of the solver type (for debugging purposes)
        virtual std::string name() const override { return "Julia"; }

    protected:
        std::string options;
        int num_iterations;
        double final_res_norm;

    private:
        bool has_matrix_ = false;
        StiffnessMatrix A_;
    };

} // namespace polysolve::linear
