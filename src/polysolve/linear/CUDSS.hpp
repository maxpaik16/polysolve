#pragma once

#include "Solver.hpp" 
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cudss.h>
#include <cuda_runtime.h>
#include <vector>

namespace polysolve::linear {
    class CUDSS : public Solver {
    public:
        CUDSS();
        ~CUDSS();

        // Core Solver Interface
        void analyze_pattern(const StiffnessMatrix &A, const int precond_num) override;
        void factorize(const StiffnessMatrix &A) override;
        void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;
        
        std::string name() const override { return "CUDSS"; }

    private:
        void free_device_memory();

        double pattern_analysis_time, factorization_time, solve_time;

        // cuDSS Handles and Descriptors
        cudssHandle_t cudss_handle = nullptr;
        cudssConfig_t config = nullptr;
        cudssData_t solverData = nullptr;
        
        cudssMatrix_t batchMatrixA = nullptr;
        cudssMatrix_t batchMatrixX = nullptr;
        cudssMatrix_t batchMatrixB = nullptr;

        // Dimensions (kept as class members to ensure pointers survive across phases)
        int m_nrows = 0;
        int m_ncols = 0;
        int m_nnz = 0;
        int m_batchCount = 1;

        std::vector<int> h_nrows;
        std::vector<int> h_ncols;
        std::vector<int> h_nnz;
        std::vector<int> h_vec_ncols;
        std::vector<int> h_ld;

        // Device memory pointers (Arrays)
        int* d_csrRowOffsets = nullptr;
        int* d_csrColIndices = nullptr;
        double* d_csrValues = nullptr;
        double* d_x = nullptr;
        double* d_b = nullptr;

        // Device memory pointers (Arrays of pointers for cuDSS Batch API)
        void **d_csrRowOffsets_void = nullptr;
        void **d_csrColIndices_void = nullptr;
        void **d_csrValues_void = nullptr;
        void **d_x_void = nullptr;
        void **d_b_void = nullptr;
    };
}