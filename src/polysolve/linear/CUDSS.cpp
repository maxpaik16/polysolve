#include "CUDSS.hpp"
#include <iostream>
#include <stdexcept>

#include "../Utils.hpp"

// Fallback macros just in case they aren't globally defined in your project.
#ifndef CHECK_CUDA
#define CHECK_CUDA(func)                                                       \
    {                                                                          \
        cudaError_t status = (func);                                           \
        if (status != cudaSuccess) {                                           \
            std::cerr << "CUDA Error: " << cudaGetErrorString(status)          \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            throw std::runtime_error("CUDA Error");                            \
        }                                                                      \
    }
#endif

#ifndef CHECK_CUDSS
#define CHECK_CUDSS(func)                                                      \
    {                                                                          \
        cudssStatus_t status = (func);                                         \
        if (status != CUDSS_STATUS_SUCCESS) {                                  \
            std::cerr << "cuDSS Error Code: " << status                        \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            throw std::runtime_error("cuDSS Error");                           \
        }                                                                      \
    }
#endif

namespace polysolve::linear {

    CUDSS::CUDSS() {
        CHECK_CUDSS(cudssCreate(&cudss_handle));
    }

    CUDSS::~CUDSS() {
        free_device_memory();
        if (cudss_handle) {
            cudssDestroy(cudss_handle);
            cudss_handle = nullptr;
        }
    }

    void CUDSS::free_device_memory() {
        // Destroy cuDSS Opaque Structures
        if (batchMatrixA) { cudssMatrixDestroy(batchMatrixA); batchMatrixA = nullptr; }
        if (batchMatrixX) { cudssMatrixDestroy(batchMatrixX); batchMatrixX = nullptr; }
        if (batchMatrixB) { cudssMatrixDestroy(batchMatrixB); batchMatrixB = nullptr; }
        if (solverData)   { cudssDataDestroy(cudss_handle, solverData); solverData = nullptr; }
        if (config)       { cudssConfigDestroy(config); config = nullptr; }

        // Free GPU Pointer Arrays
        if (d_csrRowOffsets_void) { cudaFree(d_csrRowOffsets_void); d_csrRowOffsets_void = nullptr; }
        if (d_csrColIndices_void) { cudaFree(d_csrColIndices_void); d_csrColIndices_void = nullptr; }
        if (d_csrValues_void)     { cudaFree(d_csrValues_void); d_csrValues_void = nullptr; }
        if (d_x_void)             { cudaFree(d_x_void); d_x_void = nullptr; }
        if (d_b_void)             { cudaFree(d_b_void); d_b_void = nullptr; }

        // Free GPU Data Arrays
        if (d_csrRowOffsets) { cudaFree(d_csrRowOffsets); d_csrRowOffsets = nullptr; }
        if (d_csrColIndices) { cudaFree(d_csrColIndices); d_csrColIndices = nullptr; }
        if (d_csrValues)     { cudaFree(d_csrValues); d_csrValues = nullptr; }
        if (d_x)             { cudaFree(d_x); d_x = nullptr; }
        if (d_b)             { cudaFree(d_b); d_b = nullptr; }
    }

    void CUDSS::analyze_pattern(const StiffnessMatrix &A, const int precond_num) {
        // Clean up previous allocations if re-analyzing
        free_device_memory();

        m_nrows = A.rows();
        m_ncols = A.cols();
        m_nnz   = A.nonZeros();

        // Persist dimensions in host memory
        h_nrows = { m_nrows };
        h_ncols = { m_ncols };
        h_nnz   = { m_nnz };
        h_vec_ncols = { 1 };
        h_ld    = { m_nrows };

        // Allocate Device Memory for Matrix A
        CHECK_CUDA(cudaMalloc(&d_csrRowOffsets, (m_nrows + 1) * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_csrColIndices, m_nnz * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_csrValues, m_nnz * sizeof(double)));

        // Copy Sparse Structure to GPU
        CHECK_CUDA(cudaMemcpy(d_csrRowOffsets, A.outerIndexPtr(), (m_nrows + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_csrColIndices, A.innerIndexPtr(), m_nnz * sizeof(int), cudaMemcpyHostToDevice));
        
        // Create Pointer Arrays on Host
        std::vector<void*> h_csrRowOffsets_void = { static_cast<void*>(d_csrRowOffsets) };
        std::vector<void*> h_csrColIndices_void = { static_cast<void*>(d_csrColIndices) };
        std::vector<void*> h_csrValues_void     = { static_cast<void*>(d_csrValues) };

        // Allocate Pointer Arrays on GPU
        CHECK_CUDA(cudaMalloc(&d_csrRowOffsets_void, m_batchCount * sizeof(void*)));
        CHECK_CUDA(cudaMalloc(&d_csrColIndices_void, m_batchCount * sizeof(void*)));
        CHECK_CUDA(cudaMalloc(&d_csrValues_void, m_batchCount * sizeof(void*)));

        // Copy Pointer Arrays to GPU
        CHECK_CUDA(cudaMemcpy(d_csrRowOffsets_void, h_csrRowOffsets_void.data(), m_batchCount * sizeof(void*), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_csrColIndices_void, h_csrColIndices_void.data(), m_batchCount * sizeof(void*), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_csrValues_void, h_csrValues_void.data(), m_batchCount * sizeof(void*), cudaMemcpyHostToDevice));

        // Create Matrix Descriptor
        CHECK_CUDSS(cudssMatrixCreateBatchCsr(
            &batchMatrixA, m_batchCount, h_nrows.data(), h_ncols.data(), h_nnz.data(), 
            d_csrRowOffsets_void, nullptr, d_csrColIndices_void, d_csrValues_void, 
            CUDA_R_32I, CUDA_R_64F, CUDSS_MTYPE_SYMMETRIC, 
            CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO
        ));

        // Setup Config and Solver Data
        CHECK_CUDSS(cudssConfigCreate(&config));
        CHECK_CUDSS(cudssDataCreate(cudss_handle, &solverData));

        {
            POLYSOLVE_SCOPED_STOPWATCH("actual pattern analysis time", pattern_analysis_time, *logger);

            // Execute Analysis Phase (X and B can be nullptr during analysis)
            CHECK_CUDSS(cudssExecute(cudss_handle, CUDSS_PHASE_ANALYSIS, config, solverData, 
                                    batchMatrixA, nullptr, nullptr));
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    }

    void CUDSS::factorize(const StiffnessMatrix &A) {
        POLYSOLVE_SCOPED_STOPWATCH("actual factorization time", factorization_time, *logger);
        // Copy the actual numerical values to the GPU
        CHECK_CUDA(cudaMemcpy(d_csrValues, A.valuePtr(), m_nnz * sizeof(double), cudaMemcpyHostToDevice));

        // Execute Factorization Phase
        CHECK_CUDSS(cudssExecute(cudss_handle, CUDSS_PHASE_FACTORIZATION, config, solverData, 
                                batchMatrixA, nullptr, nullptr));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    void CUDSS::solve(const Ref<const VectorXd> b, Ref<VectorXd> x) {
        // Lazily allocate X and B descriptors / memory on the first solve pass
        if (d_x == nullptr || d_b == nullptr) {
            CHECK_CUDA(cudaMalloc(&d_x, m_nrows * sizeof(double)));
            CHECK_CUDA(cudaMalloc(&d_b, m_nrows * sizeof(double)));

            std::vector<void*> h_x_void = { static_cast<void*>(d_x) };
            std::vector<void*> h_b_void = { static_cast<void*>(d_b) };

            CHECK_CUDA(cudaMalloc(&d_x_void, m_batchCount * sizeof(void*)));
            CHECK_CUDA(cudaMalloc(&d_b_void, m_batchCount * sizeof(void*)));

            CHECK_CUDA(cudaMemcpy(d_x_void, h_x_void.data(), m_batchCount * sizeof(void*), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_b_void, h_b_void.data(), m_batchCount * sizeof(void*), cudaMemcpyHostToDevice));

            CHECK_CUDSS(cudssMatrixCreateBatchDn(
                &batchMatrixX, m_batchCount, h_nrows.data(), h_vec_ncols.data(), h_ld.data(), 
                d_x_void, CUDA_R_32I, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR
            ));

            CHECK_CUDSS(cudssMatrixCreateBatchDn(
                &batchMatrixB, m_batchCount, h_nrows.data(), h_vec_ncols.data(), h_ld.data(), 
                d_b_void, CUDA_R_32I, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR
            ));
        }

        // Transfer Inputs to GPU
        CHECK_CUDA(cudaMemcpy(d_b, b.data(), m_nrows * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_x, x.data(), m_nrows * sizeof(double), cudaMemcpyHostToDevice)); // Optional: Copy initial guess

        {
            POLYSOLVE_SCOPED_STOPWATCH("actual solve time", solve_time, *logger);
            // Execute Solve Phase
            CHECK_CUDSS(cudssExecute(cudss_handle, CUDSS_PHASE_SOLVE, config, solverData, 
                                    batchMatrixA, batchMatrixX, batchMatrixB));
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        // Transfer Output back to Host
        CHECK_CUDA(cudaMemcpy(x.data(), d_x, m_nrows * sizeof(double), cudaMemcpyDeviceToHost));
    }
}