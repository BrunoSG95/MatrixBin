#pragma once

#include "BinaryMatrix.cuh"

float* cuda_mul_vector(long aWidth, long aHeight, BINARY_TYPE* aData, long bHeight, float* bData);

float* cuda_mul_matrix(long aWidth, long aHeight, BINARY_TYPE* aData, long bWidth ,long bHeight, float* bData);