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
        CHECK_CUDSS(cudssSetThreadingLayer(cudss_handle, "/usr/lib/x86_64-linux-gnu/libcudss_mtlayer_gomp.so"));
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
        if (MatrixA) { cudssMatrixDestroy(MatrixA); MatrixA = nullptr; }
        if (MatrixX) { cudssMatrixDestroy(MatrixX); MatrixX = nullptr; }
        if (MatrixB) { cudssMatrixDestroy(MatrixB); MatrixB = nullptr; }
        if (solverData)   { cudssDataDestroy(cudss_handle, solverData); solverData = nullptr; }
        if (config)       { cudssConfigDestroy(config); config = nullptr; }

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

        // Allocate Device Memory for Matrix A
        CHECK_CUDA(cudaMalloc(&d_csrRowOffsets, (m_nrows + 1) * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_csrColIndices, m_nnz * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_csrValues, m_nnz * sizeof(double)));

        // Copy Sparse Structure to GPU
        CHECK_CUDA(cudaMemcpy(d_csrRowOffsets, A.outerIndexPtr(), (m_nrows + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_csrColIndices, A.innerIndexPtr(), m_nnz * sizeof(int), cudaMemcpyHostToDevice));
        
        // Create Matrix Descriptor
        CHECK_CUDSS(cudssMatrixCreateCsr(
            &MatrixA, m_nrows, m_ncols, m_nnz, 
            d_csrRowOffsets, nullptr, d_csrColIndices, d_csrValues, 
            CUDA_R_32I, CUDA_R_64F, CUDSS_MTYPE_SYMMETRIC, 
            CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO
        ));

        // Setup Config and Solver Data
        CHECK_CUDSS(cudssConfigCreate(&config));
        CHECK_CUDSS(cudssDataCreate(cudss_handle, &solverData));

        {
            POLYSOLVE_SCOPED_STOPWATCH("actual pattern analysis time", pattern_analysis_time, *logger);

            {
                POLYSOLVE_SCOPED_STOPWATCH("actual reordering time", reordering_time, *logger);
                CHECK_CUDSS(cudssExecute(cudss_handle, CUDSS_PHASE_REORDERING, config, solverData, 
                                   MatrixA, nullptr, nullptr));
                CHECK_CUDA(cudaDeviceSynchronize());
            }
            {
                POLYSOLVE_SCOPED_STOPWATCH("actual symbolic factorization time", symbolic_time, *logger);
                CHECK_CUDSS(cudssExecute(cudss_handle, CUDSS_PHASE_SYMBOLIC_FACTORIZATION, config, solverData, 
                                    MatrixA, nullptr, nullptr));
                CHECK_CUDA(cudaDeviceSynchronize());
            }
        }
    }

    void CUDSS::factorize(const StiffnessMatrix &A) {
        POLYSOLVE_SCOPED_STOPWATCH("actual factorization time", factorization_time, *logger);
        // Copy the actual numerical values to the GPU
        CHECK_CUDA(cudaMemcpy(d_csrValues, A.valuePtr(), m_nnz * sizeof(double), cudaMemcpyHostToDevice));

        // Execute Factorization Phase
        CHECK_CUDSS(cudssExecute(cudss_handle, CUDSS_PHASE_FACTORIZATION, config, solverData, 
                                MatrixA, nullptr, nullptr));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    void CUDSS::solve(const Ref<const VectorXd> b, Ref<VectorXd> x) {
        // Lazily allocate X and B descriptors / memory on the first solve pass
        if (d_x == nullptr || d_b == nullptr) {
            CHECK_CUDA(cudaMalloc(&d_x, m_nrows * sizeof(double)));
            CHECK_CUDA(cudaMalloc(&d_b, m_nrows * sizeof(double)));

            CHECK_CUDSS(cudssMatrixCreateDn(
                &MatrixX, m_nrows, 1, m_nrows, 
                d_x, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR
            ));

            CHECK_CUDSS(cudssMatrixCreateDn(
                &MatrixB, m_nrows, 1, m_nrows, 
                d_b, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR
            ));
        }

        // Transfer Inputs to GPU
        CHECK_CUDA(cudaMemcpy(d_b, b.data(), m_nrows * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_x, x.data(), m_nrows * sizeof(double), cudaMemcpyHostToDevice));

        {
            POLYSOLVE_SCOPED_STOPWATCH("actual solve time", solve_time, *logger);
            // Execute Solve Phase
            CHECK_CUDSS(cudssExecute(cudss_handle, CUDSS_PHASE_SOLVE, config, solverData, 
                                    MatrixA, MatrixX, MatrixB));
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        // Transfer Output back to Host
        CHECK_CUDA(cudaMemcpy(x.data(), d_x, m_nrows * sizeof(double), cudaMemcpyDeviceToHost));
    }
}