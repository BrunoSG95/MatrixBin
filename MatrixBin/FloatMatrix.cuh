#pragma once

#include "Matrix.cuh"

#define FLOAT_MAT_TYPE double

class FloatMatrix : public Matrix {
private:
	FLOAT_MAT_TYPE * data;
public:
	FloatMatrix(long width, long height, FLOAT_MAT_TYPE* data = nullptr);
	FLOAT_MAT_TYPE * getData();
};
