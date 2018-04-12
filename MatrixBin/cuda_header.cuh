#pragma once

float* cuda_mul_vector(long aWidth, long aHeight, unsigned int* aData, long bHeight, float* bData);

float* cuda_mul_matrix(long aWidth, long aHeight, unsigned int* aData, long bWidth ,long bHeight, float* bData);