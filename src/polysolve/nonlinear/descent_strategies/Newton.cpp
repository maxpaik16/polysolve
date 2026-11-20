#include "Newton.hpp"

#include <polysolve/Utils.hpp>

#if defined(SPDLOG_FMT_EXTERNAL)
#include <fmt/color.h>
#else
#include <spdlog/fmt/bundled/color.h>
#endif

#include <cmath>

namespace polysolve::nonlinear
{
    namespace
    {
        /// Options of one Newton strategy: solver_params[key] if present, else solver_params
        /// itself (the list form, where each entry holds its own options).
        const json &strategy_params(const std::string &key, const json &solver_params)
        {
            const auto it = solver_params.find(key);
            return it != solver_params.end() ? *it : solver_params;
        }

        template <typename T>
        T param_or(const json &params, const std::string &name, const T &default_value)
        {
            const auto it = params.find(name);
            return it != params.end() && !it->is_null() ? it->template get<T>() : default_value;
        }
    } // namespace

    std::vector<std::shared_ptr<DescentStrategy>> Newton::create_solver(
        const bool sparse,
        const json &solver_params,
        const json &linear_solver_params,
        const double characteristic_length,
        spdlog::logger &logger,
        const NormType norm_type,
        const int dimension)
    {
        // Copies stuff from main newton
        json proj_solver_params = R"({"ProjectedNewton": {}})"_json;
        proj_solver_params["ProjectedNewton"]["residual_tolerance"] = solver_params["Newton"]["residual_tolerance"];

        json reg_solver_params = R"({"RegularizedNewton": {}})"_json;
        reg_solver_params["RegularizedNewton"]["residual_tolerance"] = solver_params["Newton"]["residual_tolerance"];
        reg_solver_params["RegularizedNewton"]["reg_weight_min"] = solver_params["Newton"]["reg_weight_min"];
        reg_solver_params["RegularizedNewton"]["reg_weight_max"] = solver_params["Newton"]["reg_weight_max"];
        reg_solver_params["RegularizedNewton"]["reg_weight_inc"] = solver_params["Newton"]["reg_weight_inc"];

        // The fallback strategies use the same linear-solve perturbation as the main Newton.
        for (const std::string name : {"perturb_target_residual", "perturb_relative", "perturb_seed"})
        {
            if (solver_params["Newton"].contains(name))
            {
                proj_solver_params["ProjectedNewton"][name] = solver_params["Newton"][name];
                reg_solver_params["RegularizedNewton"][name] = solver_params["Newton"][name];
            }
        }

        std::vector<std::shared_ptr<DescentStrategy>> res;
        const bool force_psd_projection = solver_params["Newton"]["force_psd_projection"];
        if (!force_psd_projection)
            res.push_back(std::make_unique<Newton>(
                sparse,
                solver_params, linear_solver_params,
                characteristic_length, logger, norm_type, dimension));

        const bool use_psd_projection = solver_params["Newton"]["use_psd_projection"];
        if (use_psd_projection)
            res.push_back(std::make_unique<ProjectedNewton>(
                sparse,
                proj_solver_params, linear_solver_params,
                characteristic_length, logger, norm_type, dimension));

        const double reg_weight_min = solver_params["Newton"]["reg_weight_min"];
        if (reg_weight_min > 0)
            res.push_back(std::make_unique<RegularizedNewton>(
                sparse, solver_params["Newton"]["use_psd_projection_in_regularized"],
                reg_solver_params, linear_solver_params,
                characteristic_length, logger, norm_type, dimension));

        if (res.empty())
            log_and_throw_error(logger, "Newton needs to have at least one of force_psd_projection=false, reg_weight_min>0, or use_psd_projection=true");

        return res;
    }

    Newton::Newton(const bool sparse,
                   const std::string &params_key,
                   const json &solver_params,
                   const json &linear_solver_params,
                   const double characteristic_length,
                   spdlog::logger &logger,
                   const NormType norm_type,
                   const int dimension)
        : Superclass(solver_params, characteristic_length, logger),
          is_sparse(sparse), characteristic_length(characteristic_length),
          residual_tolerance(extract_param(params_key, "residual_tolerance", solver_params)), norm_type(norm_type)
    {
        const json &params = strategy_params(params_key, solver_params);
        perturb_target_residual = param_or<double>(params, "perturb_target_residual", 0.0);
        perturb_relative = param_or<bool>(params, "perturb_relative", true);
        perturb_rng.seed(param_or<uint64_t>(params, "perturb_seed", 0));

        linear_solver = polysolve::linear::Solver::create(linear_solver_params, logger, true, dimension);

        if (linear_solver->is_dense() == sparse)
            log_and_throw_error(logger, "Newton linear solver must be {}, instead got {}", sparse ? "sparse" : "dense", linear_solver->name());

        if (residual_tolerance <= 0)
            log_and_throw_error(logger, "Newton residual_tolerance must be > 0, instead got {}", residual_tolerance);

        if (perturb_target_residual < 0 || !std::isfinite(perturb_target_residual))
            log_and_throw_error(logger, "Newton perturb_target_residual must be >= 0, instead got {}", perturb_target_residual);

        if (perturbation_enabled())
            m_logger.info("[{}] perturbing linear solves to a {} residual of {:g}",
                          params_key, perturb_relative ? "relative" : "absolute", perturb_target_residual);
    }

    Newton::Newton(
        const bool sparse,
        const json &solver_params,
        const json &linear_solver_params,
        const double characteristic_length,
        spdlog::logger &logger,
        const NormType norm_type,
        const int dimension)
        : Newton(sparse, "Newton", solver_params, linear_solver_params, characteristic_length, logger, norm_type, dimension)
    {
    }

    ProjectedNewton::ProjectedNewton(
        const bool sparse,
        const json &solver_params,
        const json &linear_solver_params,
        const double characteristic_length,
        spdlog::logger &logger,
        const NormType norm_type,
        const int dimension)
        : Superclass(sparse, "ProjectedNewton", solver_params, linear_solver_params, characteristic_length, logger, norm_type, dimension)
    {
    }

    RegularizedNewton::RegularizedNewton(
        const bool sparse,
        const bool project_to_psd,
        const json &solver_params,
        const json &linear_solver_params,
        const double characteristic_length,
        spdlog::logger &logger,
        const NormType norm_type,
        const int dimension)
        : Superclass(sparse, "RegularizedNewton", solver_params, linear_solver_params, characteristic_length, logger, norm_type, dimension),
          project_to_psd(project_to_psd)
    {
        reg_weight_min = extract_param("RegularizedNewton", "reg_weight_min", solver_params);
        reg_weight_max = extract_param("RegularizedNewton", "reg_weight_max", solver_params);
        reg_weight_inc = extract_param("RegularizedNewton", "reg_weight_inc", solver_params);

        reg_weight = reg_weight_min;

        if (reg_weight_min <= 0)
            log_and_throw_error(logger, "Newton reg_weight_min must be  > 0, instead got {}", reg_weight_min);

        if (reg_weight_inc <= 1)
            log_and_throw_error(logger, "Newton reg_weight_inc must be  > 1, instead got {}", reg_weight_inc);

        if (reg_weight_max <= reg_weight_min)
            log_and_throw_error(logger, "Newton reg_weight_max must be  > {}, instead got {}", reg_weight_min, reg_weight_max);
    }

    // =======================================================================

    void Newton::reset(const int ndof)
    {
        Superclass::reset(ndof);
        if (track_info)
            internal_solver_info = json::array();
    }

    void RegularizedNewton::reset(const int ndof)
    {
        Superclass::reset(ndof);
        reg_weight = reg_weight_min;
    }

    // =======================================================================

    bool Newton::compute_update_direction(
        Problem &objFunc,
        const TVector &x,
        const TVector &grad,
        TVector &direction)
    {
        const double residual =
            is_sparse ? solve_sparse_linear_system(objFunc, x, grad, direction)
                      : solve_dense_linear_system(objFunc, x, grad, direction);

        double current_residual_tolerance = residual_tolerance;

        if (std::isnan(residual) || residual > current_residual_tolerance)
        {
            if (m_logger.should_log(spdlog::level::debug))
                m_logger.debug("[{}] large (or nan) linear solve residual {}>{} (‖∇f‖={})",
                               name(), residual, current_residual_tolerance, objFunc.grad_norm(grad, norm_type));

            return false;
        }
        else if (m_logger.should_log(spdlog::level::trace))
        {
            m_logger.trace("linear solve residual {}", residual);
        }

        return true;
    }

    // =======================================================================

    double Newton::solve_sparse_linear_system(Problem &objFunc,
                                              const TVector &x,
                                              const TVector &grad,
                                              TVector &direction)
    {
        polysolve::StiffnessMatrix hessian;

        {
            POLYSOLVE_SCOPED_STOPWATCH_IF(track_info, "assembly time", this->assembly_time, m_logger);
            compute_hessian(objFunc, x, hessian);
        }

        {
            POLYSOLVE_SCOPED_STOPWATCH_IF(track_info, "linear solve", this->inverting_time, m_logger);

            const Eigen::VectorXi block_mapping = objFunc.block_mapping();
            if (block_mapping.size() > 0)
                linear_solver->set_block_mapping(block_mapping);

            // TODO: get the correct size
            linear_solver->analyze_pattern(hessian, hessian.rows());

            try
            {
                linear_solver->factorize(hessian);
            }
            catch (const std::runtime_error &err)
            {
                // warn if using gradient descent
                m_logger.debug("Unable to factorize Hessian: \"{}\"", err.what());

                // Eigen::saveMarket(hessian, "problematic_hessian.mtx");
                return std::nan("");
            }

            linear_solver->solve(-grad, direction); // H Δx = -g
        }

        const TVector r0 = hessian * direction + grad;           // H Δx + g = 0
        const double residual = objFunc.grad_norm(r0, norm_type); // unperturbed residual

        json info;
        if (track_info)
            linear_solver->get_info(info);

        // Only perturb a solve that would be accepted: a rejected direction is discarded anyway.
        if (perturbation_enabled() && std::isfinite(residual) && residual <= residual_tolerance)
        {
            const TVector v = random_perturbation(direction.size());
            const TVector w = hessian * v;
            apply_perturbation(objFunc, grad, r0, v, w, direction, info);
        }

        if (track_info)
            internal_solver_info.push_back(info);

        return residual;
    }

    double Newton::solve_dense_linear_system(Problem &objFunc,
                                             const TVector &x,
                                             const TVector &grad,
                                             TVector &direction)
    {
        Eigen::MatrixXd hessian;

        {
            POLYSOLVE_SCOPED_STOPWATCH_IF(track_info, "assembly time", this->assembly_time, m_logger);
            compute_hessian(objFunc, x, hessian);
        }

        {
            POLYSOLVE_SCOPED_STOPWATCH_IF(track_info, "linear solve", this->inverting_time, m_logger);

            const Eigen::VectorXi block_mapping = objFunc.block_mapping();
            if (block_mapping.size() > 0)
                linear_solver->set_block_mapping(block_mapping);

            try
            {
                linear_solver->analyze_pattern_dense(hessian, hessian.rows());
                linear_solver->factorize_dense(hessian);
                m_neg_grad = -grad;
                linear_solver->solve(m_neg_grad, direction);
            }
            catch (const std::runtime_error &err)
            {
                // warn if using gradient descent
                m_logger.debug("Unable to factorize Hessian: \"{}\"",
                               err.what());

                return std::nan("");
            }
        }

        m_hessian_direction.noalias() = hessian * direction;
        const double residual = (m_hessian_direction + grad).norm(); // H Δx + g = 0

        json info;
        if (track_info)
            linear_solver->get_info(info);

        // Only perturb a solve that would be accepted: a rejected direction is discarded anyway.
        if (perturbation_enabled() && std::isfinite(residual) && residual <= residual_tolerance)
        {
            const TVector r0 = m_hessian_direction + grad;
            const TVector v = random_perturbation(direction.size());
            const TVector w = hessian * v;
            apply_perturbation(objFunc, grad, r0, v, w, direction, info);
        }

        if (track_info)
            internal_solver_info.push_back(info);

        return residual;
    }

    // =======================================================================

    Newton::TVector Newton::random_perturbation(const int size)
    {
        std::normal_distribution<double> normal(0.0, 1.0);
        TVector v(size);
        for (int i = 0; i < size; ++i)
            v[i] = normal(perturb_rng);
        return v;
    }

    void Newton::apply_perturbation(const Problem &objFunc,
                                    const TVector &grad,
                                    const TVector &r0,
                                    const TVector &v,
                                    const TVector &w,
                                    TVector &direction,
                                    json &info) const
    {
        const double r0_norm = objFunc.grad_norm(r0, norm_type);
        const double target = perturb_relative
                                  ? perturb_target_residual * objFunc.grad_norm(grad, norm_type)
                                  : perturb_target_residual;
        const double w_norm = objFunc.grad_norm(w, norm_type);

        double alpha = 0;
        double achieved = r0_norm;

        if (!(target > r0_norm))
        {
            m_logger.debug("[{}] linear solve residual {:g} already >= perturbation target {:g}; not perturbing",
                           name(), r0_norm, target);
        }
        else if (!(w_norm > 0) || !std::isfinite(w_norm))
        {
            m_logger.warn("[{}] ‖H v‖={:g} for the random perturbation v; not perturbing", name(), w_norm);
        }
        else
        {
            // f(a) = ‖r0 + a w‖ - target is convex with f(0) < 0, and by the triangle
            // inequality f(hi) >= 0 for hi = (target + ‖r0‖) / ‖w‖. Bisect for the root.
            TVector r(r0.size());
            const auto f = [&](const double a) {
                r = r0;
                r.noalias() += a * w;
                return objFunc.grad_norm(r, norm_type) - target;
            };

            double lo = 0, hi = (target + r0_norm) / w_norm;
            alpha = hi;
            for (int it = 0; it < 200; ++it)
            {
                alpha = 0.5 * (lo + hi);
                const double fa = f(alpha);
                if (std::abs(fa) <= 1e-12 * target || hi - lo <= 1e-15 * hi)
                    break;
                (fa < 0 ? lo : hi) = alpha;
            }

            direction.noalias() += alpha * v;
            achieved = f(alpha) + target;

            if (m_logger.should_log(spdlog::level::debug))
                m_logger.debug("[{}] perturbed linear solve: residual {:g} -> {:g} (target {:g}, ‖∇f‖={:g})",
                               name(), r0_norm, achieved, target, objFunc.grad_norm(grad, norm_type));
        }

        if (track_info)
        {
            info["perturbation"] = {
                {"unperturbed_residual", r0_norm},
                {"perturbed_residual", achieved},
                {"target_residual", target},
                {"alpha", alpha},
                {"perturbation_norm", alpha * v.norm()},
            };
        }
    }
    // =======================================================================

    void Newton::compute_hessian(Problem &objFunc,
                                 const TVector &x,
                                 polysolve::StiffnessMatrix &hessian)

    {
        objFunc.set_project_to_psd(false);
        objFunc.hessian(x, hessian);
    }

    void ProjectedNewton::compute_hessian(Problem &objFunc,
                                          const TVector &x,
                                          polysolve::StiffnessMatrix &hessian)

    {
        objFunc.set_project_to_psd(true);
        objFunc.hessian(x, hessian);
    }

    void RegularizedNewton::compute_hessian(Problem &objFunc,
                                            const TVector &x,
                                            polysolve::StiffnessMatrix &hessian)

    {
        if (x.size() != x_cache.size() || x != x_cache)
        {
            objFunc.set_project_to_psd(project_to_psd);
            objFunc.hessian(x, hessian_cache);
            x_cache = x;
        }
        hessian = hessian_cache;
        if (reg_weight > 0)
        {
            hessian += reg_weight * sparse_identity(hessian.rows(), hessian.cols());
        }
    }

    void Newton::compute_hessian(Problem &objFunc,
                                 const TVector &x,
                                 Eigen::MatrixXd &hessian)

    {
        objFunc.set_project_to_psd(false);
        objFunc.hessian(x, hessian);
    }

    void ProjectedNewton::compute_hessian(Problem &objFunc,
                                          const TVector &x,
                                          Eigen::MatrixXd &hessian)

    {
        objFunc.set_project_to_psd(true);
        objFunc.hessian(x, hessian);
    }

    void RegularizedNewton::compute_hessian(Problem &objFunc,
                                            const TVector &x,
                                            Eigen::MatrixXd &hessian)

    {
        objFunc.set_project_to_psd(project_to_psd);
        objFunc.hessian(x, hessian);
        if (reg_weight > 0)
        {
            for (int k = 0; k < x.size(); k++)
                hessian(k, k) += reg_weight;
        }
    }
    // =======================================================================

    bool RegularizedNewton::handle_error()
    {
        reg_weight *= reg_weight_inc;
        return reg_weight < reg_weight_max;
    }
    // =======================================================================

    void Newton::update_solver_info(json &solver_info, const double per_iteration)
    {
        Superclass::update_solver_info(solver_info, per_iteration);

        solver_info["internal_solver"] = internal_solver_info;
        solver_info["time_assembly"] = assembly_time / per_iteration;
        solver_info["time_inverting"] = inverting_time / per_iteration;
    }

    void Newton::reset_times()
    {
        assembly_time = 0;
        inverting_time = 0;
    }

    void Newton::log_times() const
    {
        if (assembly_time <= 0 && inverting_time <= 0)
            return; // nothing to log
        m_logger.debug(
            "[{}][{}] assembly: {:.2e}s; linear_solve: {:.2e}s",
            fmt::format(fmt::fg(fmt::terminal_color::magenta), "timing"),
            name(), assembly_time, inverting_time);
    }

    // =======================================================================

} // namespace polysolve::nonlinear
