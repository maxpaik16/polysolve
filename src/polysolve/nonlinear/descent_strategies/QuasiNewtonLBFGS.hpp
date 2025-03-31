// L-BFGS solver (Using the LBFGSpp under MIT License).

#pragma once

#include <list>

#include "DescentStrategy.hpp"
#include <polysolve/Utils.hpp>

#include <polysolve/linear/Solver.hpp>

#include <LBFGSpp/BFGSMat.h>

namespace polysolve::nonlinear
{
    class QuasiNewtonLBFGS : public DescentStrategy
    {
    public:
        using Superclass = DescentStrategy;

        QuasiNewtonLBFGS(const json &solver_params,
            const json &linear_solver_params,
              const double characteristic_length,
              spdlog::logger &logger);

        std::string name() const override { return "QuasiNewton-L-BFGS"; }

    public:
        void reset(const int ndof) override;

        bool compute_update_direction(
            Problem &objFunc,
            const TVector &x,
            const TVector &grad,
            TVector &direction) override;

    private:
        /// The number of corrections to approximate the inverse Hessian matrix.
        /// The L-BFGS routine stores the computation results of previous \ref m
        /// iterations to approximate the inverse Hessian matrix of the current
        /// iteration. This parameter controls the size of the limited memories
        /// (corrections). The default value is \c 6. Values less than \c 3 are
        /// not recommended. Large values will result in excessive computing time.
        int m_history_size;
        int restart_interval;
        int interval_counter = 0;
        std::list<TVector> x_history;
        std::list<TVector> grad_history;

        Eigen::SparseMatrix<double> initial_hessian;

        std::unique_ptr<polysolve::linear::Solver> linear_solver; ///< Linear solver used to solve the linear system
    };
} // namespace polysolve::nonlinear
