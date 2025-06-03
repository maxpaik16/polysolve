#pragma once

#ifdef POLYSOLVE_WITH_CUSOLVER

////////////////////////////////////////////////////////////////////////////////
#include "Solver.hpp"

#include <cuda_runtime.h>
#include <cusolverSp.h>

#include <vector>

////////////////////////////////////////////////////////////////////////////////
//
// https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverSP-function-reference
//

namespace polysolve::linear
{
    template <typename T>
    class CuSolverSP : public Solver
    {

    public:
        CuSolverSP();
        ~CuSolverSP();

    private:
        POLYSOLVE_DELETE_MOVE_COPY(CuSolverSP)

    public:
        //////////////////////
        // Public interface //
        //////////////////////

        // Retrieve memory information from cuSolverSP
        virtual void get_info(json &params) const override;

        // Factorize system matrix (sparse)
        virtual void factorize(const StiffnessMatrix &A) override;

        // Factorize system matrix (dense, preferred)
        virtual void factorize_dense(const Eigen::MatrixXd &A) override;

        bool is_dense() const override { return true; }

        // Solve the linear system Ax = b
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;

        // Name of the solver type (for debugging purposes)
        virtual std::string name() const override { return "cuSolverSP"; }

    protected:
        void init();

    private:
        cusolverSpHandle_t cuHandle;
        cusolverSpParams_t cuParams;
        cudaStream_t stream;

        // device copies
        bool d_A_alloc = false;
        T *d_A;
        bool d_b_alloc = false;
        T *d_b;
        int64_t *d_Ipiv;

        // device work buffers
        size_t d_lwork = 0;     // size of workspace
        void *d_work = nullptr; // device workspace for getrf
        size_t h_lwork = 0;     // size of workspace
        void *h_work = nullptr; // host workspace for getrf
        int *d_info = nullptr;  // error info

        int numrows;
    };

} // namespace polysolve::linear

#endif
