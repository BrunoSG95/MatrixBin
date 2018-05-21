#pragma once

#include "Matrix.cuh"

#define FLOAT_MAT_TYPE float

class FloatMatrix : public Matrix {
private:
	FLOAT_MAT_TYPE * data;
public:
	FloatMatrix(long width, long height, FLOAT_MAT_TYPE* data = nullptr);
	FloatMatrix operator*(FloatMatrix& b);
	bool operator==(FloatMatrix& b);
	FLOAT_MAT_TYPE * getData();
	~FloatMatrix();
};
