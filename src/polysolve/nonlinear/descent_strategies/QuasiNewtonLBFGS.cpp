// L-BFGS solver (Using the LBFGSpp under MIT License).

#include "QuasiNewtonLBFGS.hpp"

#include <polysolve/Utils.hpp>

namespace polysolve::nonlinear
{
    QuasiNewtonLBFGS::QuasiNewtonLBFGS(const json &solver_params, const json &linear_solver_params,
                 const double characteristic_length,
                 spdlog::logger &logger)
        : Superclass(solver_params,
                     characteristic_length,
                     logger)
    {
        linear_solver = polysolve::linear::Solver::create(linear_solver_params, logger);
        m_history_size = extract_param("QuasiNewtonLBFGS", "history_size", solver_params);
        restart_interval = extract_param("QuasiNewtonLBFGS", "restart_interval", solver_params);
        always_use_first_hessian = solver_params["QuasiNewtonLBFGS"]["always_use_first_hessian"];
        if (m_history_size <= 0)
            log_and_throw_error(logger, "QuasiNewton-L-BFGS history_size must be >=1, instead got {}", m_history_size);
    }

    void QuasiNewtonLBFGS::reset(const int ndof)
    {
        Superclass::reset(ndof);
        if (!always_use_first_hessian)
        {
            initial_hessian.resize(0, 0);
        }
        x_history.clear();
        grad_history.clear();
        interval_counter = 0;
    }

    bool QuasiNewtonLBFGS::compute_update_direction(
        Problem &objFunc,
        const TVector &x,
        const TVector &grad,
        TVector &direction)
    {
        if (interval_counter % restart_interval == 0)
        {
            reset(x.size());
        }

        ++interval_counter;

        if (initial_hessian.rows() == 0)
        {
            double evaluate_hessian_time;
            {
                POLYSOLVE_SCOPED_STOPWATCH("evaluate hessian", evaluate_hessian_time, m_logger);
                m_logger.trace("Factorizing new initial Hessian");
                objFunc.hessian(x, initial_hessian);
            }

            try
            {
                double fact_time;
                {
                    POLYSOLVE_SCOPED_STOPWATCH("factorization time", fact_time, m_logger);
                    linear_solver->analyze_pattern(initial_hessian, initial_hessian.rows());
                    linear_solver->factorize(initial_hessian);
                }
            }
            catch (const std::runtime_error &err)
            {
                m_logger.debug("Unable to factorize Hessian: \"{}\";", err.what());
                return false;
            }
        }

        double update_time;
        {
            POLYSOLVE_SCOPED_STOPWATCH("update hessian time", update_time, m_logger);
        
            TVector q = -grad;

            const int curr_m = x_history.size();

            std::vector<TVector> s(curr_m);
            std::vector<TVector> t(curr_m);
            std::vector<double> rho(curr_m);
            std::vector<double> zeta(curr_m);

            Eigen::VectorXd last_x = x;
            int i_counter = curr_m - 1;
            for (auto curr_x = x_history.rbegin(); curr_x != x_history.rend(); ++curr_x)
            {
                s[i_counter] = last_x - *curr_x;
                last_x = *curr_x;
                --i_counter;
            }

            Eigen::VectorXd last_grad = grad;
            i_counter = curr_m - 1;
            for (auto curr_grad = grad_history.rbegin(); curr_grad != grad_history.rend(); ++curr_grad)
            {
                t[i_counter] = last_grad - *curr_grad;
                last_grad = *curr_grad;
                --i_counter;
            }

            for (int i = curr_m - 1; i >= 0; --i)
            {
                rho[i] = t[i].dot(s[i]);
                zeta[i] = s[i].dot(q) / rho[i];
                q  -= zeta[i] * t[i];
            }

            Eigen::VectorXd r(x.size());
            linear_solver->solve(q, r);

            for (int i = 0; i < curr_m; ++i)
            {
                double eta = t[i].dot(r) / rho[i];
                r += (zeta[i] - eta) * s[i];
            }

            direction = r;
        }

        if (x_history.size() == m_history_size)
        {
            x_history.pop_front();
            grad_history.pop_front();
        }
    
        x_history.push_back(x);
        grad_history.push_back(grad);

        return true;
    }
} // namespace polysolve::nonlinear
