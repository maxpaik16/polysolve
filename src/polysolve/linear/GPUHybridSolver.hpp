#pragma once

////////////////////////////////////////////////////////////////////////////////
#include "Solver.hpp"

#include <vector>
#include <deque>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <HYPRE.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_parcsr_mv.h>


namespace polysolve::linear
{
    class GPUHybridSolver : public Solver
    {

    public:
        GPUHybridSolver();
        ~GPUHybridSolver();

    private:
        POLYSOLVE_DELETE_MOVE_COPY(GPUHybridSolver)

    public:
        //////////////////////
        // Public interface //
        //////////////////////

        // Set solver parameters
        virtual void set_parameters(const json &params) override;

        // Retrieve solve information
        virtual void get_info(json &params) const override;

        void check_settings() const;

        // Analyze sparsity pattern
        virtual void analyze_pattern(const StiffnessMatrix &A, const int precond_num) override;

        // Factorize system matrix
        virtual void factorize(const StiffnessMatrix &A) override;

        // Solve the linear system Ax = b
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;

        // Name of the solver type (for debugging purposes)
        virtual std::string name() const override { return "GPUHybrid"; }

        // set convergence tolerance
        virtual void set_tolerance(const double tol) override { conv_tol_ = tol; }

    protected:
        // AMG settings
        double theta = 0.5;

        // General solver settings
        int dimension_ = 1; // 1 = scalar (Laplace), 2 or 3 = vector (Elasticity)
        int max_iter_ = 1000;
        int pre_max_iter_ = 1;
        double conv_tol_ = 1e-10;
        bool use_absolute_tol = false;
        bool do_mixed_precond = false;
        int num_threads = 1; 

        // solve information
        HYPRE_Int num_iterations;
        HYPRE_Complex final_res_norm;

    private:
        bool has_matrix_ = false;

        // temporary buffer
        Eigen::VectorXd local_result;

        // problem-specific data
        Eigen::SparseMatrix<double, Eigen::RowMajor> sparse_A;

        // Hypre variables
        HYPRE_IJMatrix A;
        HYPRE_ParCSRMatrix parcsr_A;
        HYPRE_IJVector ij_x;
        HYPRE_IJVector ij_b;

        // hybrid preconditioner data
        std::deque<std::unique_ptr<AbstractSolver>> D_solvers;
        std::vector<std::vector<int>> bad_indices_arrays;
        std::vector<std::vector<int>> bad_subdomain_assignments;
        std::vector<std::unordered_map<int, int>> index_mappings;

        // timing variables
        double copy_b_and_x_time;
        double set_options_time;
        double actual_solve_time;
        double factorization_time;
        double bad_dof_selection_time;
        double dss_factorization_time;
        double dss_assembly_time;
        double eigen_copy_time;
        double matrix_destroy_time;
        double matrix_copy_time;
        double amg_setup_time;
        double destroy_time;
        double loop_time;
        double solve_time;
        double copy_to_time;
        double copy_from_time;
        double dss_step_time;
        double select_dofs_from_diag_time;
        double set_from_triplets_time;
        double matmul_time;
        double prepare_dss_time;
        double decomp_time;

        void copy_matrix_to_hypre();

        // solve helpers
        void init_hypre_vectors();

        // matrix multiplication
        void matmul(Eigen::VectorXd &x, Eigen::SparseMatrix<double, Eigen::RowMajor> &A, Eigen::VectorXd &result);
        double dot(Eigen::VectorXd &x, Eigen::VectorXd &y);

        // preconditioning functions
        void custom_mixed_precond_iter(const HYPRE_Solver &precond, Eigen::VectorXd &r, Eigen::VectorXd &z);
        void amg_precond_iter(const HYPRE_Solver &precond, Eigen::VectorXd& b, Eigen::VectorXd &x);
        void dss_precond_iter(Eigen::VectorXd &z, Eigen::VectorXd &r, Eigen::VectorXd &next_z);

        // hybrid preconditioner preparation functions
        void prepare_dss();
        void decompose_subdomains_to_disjoint_subsets();
        void build_index_mappings();
        void select_bad_indices();
        void factorize_submatrix();
        void assemble_D(int bad_i, int i, Eigen::SparseMatrix<double>& D);

        // Krylov solve methods
        void pcg_solve(Eigen::VectorXd &rhs, Eigen::VectorXd &result, HYPRE_ParVector &par_b, HYPRE_ParVector &par_x, HYPRE_Solver &precond);

    };

    struct disjointSet {
        std::vector<int> parent, rank;

        disjointSet(int n) 
        {
            rank.assign(n, 0);
            for (int i = 0; i < n; i++)
                parent.push_back(i);
        }

        int find_set(int v) 
        {
            if (parent[v] != v)
                parent[v] = find_set(parent[v]);

            return parent[v];
        }

        void union_set(int x,int y) 
        {
            if (x == y)
            {
                return;
            }

            x = find_set(x);
            y = find_set(y);

            if (rank[x] > rank[y])
                parent[y] = x;
            else
            {
                parent[x] = y;
                if(rank[x] == rank[y])
                    rank[y]++;
            }
        }
};


} // namespace polysolve::linear
