#pragma once

#include "BinaryMatrix.cuh"

FLOAT_VEC_TYPE* cuda_mul_vector(long aWidth, long aHeight, BINARY_TYPE* aData, long bHeight, FLOAT_VEC_TYPE* bData);

FLOAT_MAT_TYPE* cuda_mul_matrix(long aWidth, long aHeight, BINARY_TYPE* aData, long bWidth ,long bHeight, FLOAT_MAT_TYPE* bData);

FLOAT_MAT_TYPE* cublas_mul_matrix(long aWidth, long aHeight, FLOAT_MAT_TYPE* aData, long bWidth, long bHeight, FLOAT_MAT_TYPE* bData);