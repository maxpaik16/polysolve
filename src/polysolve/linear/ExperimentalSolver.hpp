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

#define HYPRE_WITH_MPI 1

#ifdef POLYSOLVE_WITH_ICHOL
#include <boost/property_tree/ptree.hpp>
namespace mschol {
    class ichol_precond;
}
#endif

namespace polysolve::linear
{
    class AbstractSolver {

    public:
        virtual void compute(const Eigen::SparseMatrix<double>& A) = 0;
        virtual Eigen::VectorXd solve(const Eigen::VectorXd& b) = 0;
        virtual ~AbstractSolver() = default;
    };

    template <typename EigenSolverT>
    class EigenWrapper : public AbstractSolver {
        EigenSolverT solver;
    public:
        void compute(const Eigen::SparseMatrix<double>& A) override {
            solver.compute(A);
        }

        Eigen::VectorXd solve(const Eigen::VectorXd& b) override {
            return solver.solve(b);
        }
    };

    class ExperimentalSolver : public Solver
    {

    public:
        ExperimentalSolver();
        ~ExperimentalSolver();

    private:
        POLYSOLVE_DELETE_MOVE_COPY(ExperimentalSolver)

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
        virtual std::string name() const override { return "Experimental"; }

        // set convergence tolerance
        virtual void set_tolerance(const double tol) override { conv_tol_ = tol; }

    protected:
        // AMG settings
        double theta = 0.5;
        bool nodal_coarsening = false;
        bool interp_rbms = false;
        int amg_iters = 1;

        // Hybrid preconditioner settings
        bool dss_in_middle = true;
        double bad_dof_threshold = 0.1;
        double max_bad_dof_threshold = 0.5; 
        bool adapt_bad_dof_threshold = false;
        double bad_dof_threshold_inc_factor = 1.5;
        int adaptive_max_iters = 100;
        bool select_bad_dofs_from_rhs = false;
        bool select_bad_dofs_from_row_norms = false;
        bool select_bad_dofs_from_amg = false;
        bool select_bad_dofs_from_diag = false;
        bool decompose_subdomains = false;
        int min_subdomain_size = 1;
        int max_subdomain_size = 1e9;
        int project_d_option = 0;
        bool find_threshold_from_secant = false;

        // General solver settings
        int dimension_ = 1; // 1 = scalar (Laplace), 2 or 3 = vector (Elasticity)
        int max_iter_ = 1000;
        int pre_max_iter_ = 1;
        double conv_tol_ = 1e-10;
        bool use_absolute_tol = false;
        bool do_mixed_precond = false;
        bool use_incomplete_cholesky_precond = false;
        bool problematic_subdomain_precond_only = false;
        bool jacobi_precond = false;
        bool use_problematic_subdomain_for_initial_guess = false;
        bool jacobi_precondition_system = false;
        bool use_gmres = false;
        bool use_minres = false;
        int m_ = 10;
        int num_threads = 1; 

        // Debugging / analysis settings
        bool print_conditioning = false;
        bool print_subdomain_conditioning = false;
        bool save_selection_criteria = false;
        bool save_selected_indices = false;
        bool save_problem = false;

        // solve information
        HYPRE_Int num_iterations;
        HYPRE_Complex final_res_norm;

    private:
        bool has_matrix_ = false;

        // MPI rank distribution 
        int myid = 0;
        int num_procs = 1;
        std::vector<int> starts;
        std::vector<int> ends;

        // temporary buffer
        Eigen::VectorXd local_result;

        // problem-specific data
        Eigen::SparseMatrix<double, Eigen::RowMajor> sparse_A;
        Eigen::DiagonalMatrix<double, Eigen::Dynamic> diag_inv; 
        Eigen::DiagonalMatrix<double, Eigen::Dynamic> sqrt_diag_A_inv; 
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P;
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P_T;

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

        Eigen::VectorXd z1, z2, z3;

#ifdef POLYSOLVE_WITH_ICHOL
        // incomplete cholesky variables
        std::shared_ptr<mschol::ichol_precond> inc_chol_precond; 
        boost::property_tree::ptree pt;
        Eigen::VectorXi ichol_dof_remapping;
        double rho = 1.01;
        int remap_dof(const int index);
        void setup_ichol_precond();
#endif

        // timing variables
        double copy_b_and_x_time;
        double set_options_time;
        double actual_solve_time;
        double factorization_time;
        double bad_dof_selection_time;
        double dss_factorization_time;
        double dss_assembly_time;
        double eigen_copy_time;
        double ichol_fac_time;
        double matrix_destroy_time;
        double matrix_copy_time;
        double copy_to_ranks_time;
        double amg_setup_time;
        double destroy_time;
        double ichol_time;
        double loop_time;
        double jacobi_time;
        double solve_time;
        double copy_to_time;
        double copy_from_time;
        double dss_step_time;
        double select_dofs_from_rhs_time;
        double select_dofs_from_amg_time;
        double select_dofs_from_diag_time;
        double select_dofs_from_row_norms_time;
        double set_from_triplets_time;
        double d_projection_time;
        double matmul_time;
        double prepare_dss_time;
        double decomp_time;
        double print_cond_time;
        double scatter_matrix_time;

        // factorization helpers
        void scatter_matrix();
        void save_problem_to_file(const std::string& file_name);
        void partition_ranks();
        void copy_matrix_to_hypre();

        // solve helpers
        void init_hypre_vectors();
        void recover_solution(Eigen::Ref<VectorXd> result);
        void check_smallest_eigenvalue();

        // hybrid preconditioner helpers
        //void scatter_subsystem();
        void assemble_D(int bad_i, int i, Eigen::SparseMatrix<double>& D);
        void project_D(Eigen::SparseMatrix<double>& D);
        void build_index_mappings();
        void decompose_subdomains_to_disjoint_subsets();
        void partition_subdomain(std::vector<int>& subdomain);
        void share_bad_subdomains();
        void load_balance_subdomains();
        void compute_permutation_matrix();

        // matrix multiplication
        void matmul(Eigen::VectorXd &x, Eigen::SparseMatrix<double, Eigen::RowMajor> &A, Eigen::VectorXd &result);
        double dot(Eigen::VectorXd &x, Eigen::VectorXd &y);

        // log system conditioning
        void check_matrix_conditioning(const std::string name, const std::set<int>& subdomain);
        void check_matrix_conditioning(const std::string name, const Eigen::MatrixXd& mat);

        // preconditioning functions
        void custom_mixed_precond_iter(const HYPRE_Solver &precond, Eigen::VectorXd &r, Eigen::VectorXd &z);
        void amg_precond_iter(const HYPRE_Solver &precond, Eigen::VectorXd& b, Eigen::VectorXd &x);
        void dss_precond_iter(Eigen::VectorXd &z, Eigen::VectorXd &r, Eigen::VectorXd &next_z);

        // hybrid preconditioner preparation functions
        void prepare_dss();
        void select_bad_indices();
        void factorize_submatrix();

        // GMRES helpers
        void GeneratePlaneRotation(double &dx, double &dy, double &cs, double &sn);
        void ApplyPlaneRotation(double &dx, double &dy, double &cs, double &sn);
        void Update(Eigen::VectorXd &x, int k, Eigen::MatrixXd &h, Eigen::VectorXd &s, Eigen::MatrixXd &v);

        // MPI communication helpers
        void all_gather_vec(const Eigen::VectorXd &local_part, Eigen::VectorXd &global_result);
        void sync_vector(Eigen::VectorXd &vec);

        // Krylov solve methods
        void pcg_solve(Eigen::VectorXd &rhs, Eigen::VectorXd &result, HYPRE_ParVector &par_b, HYPRE_ParVector &par_x, HYPRE_Solver &precond);
        void minres_solve(Eigen::VectorXd &rhs, Eigen::VectorXd &result, HYPRE_ParVector &par_b, HYPRE_ParVector &par_x, HYPRE_Solver &precond);
        void gmres_solve(Eigen::VectorXd &rhs, Eigen::VectorXd &result, HYPRE_ParVector &par_b, HYPRE_ParVector &par_x, HYPRE_Solver &precond);

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
