#pragma once

#include "Matrix.cuh"
#include "FloatMatrix.cuh"
#include "FloatVector.cuh"

#define BINARY_TYPE unsigned int
#define BITS_IN_BIN 8 * sizeof(BINARY_TYPE)

class BinaryMatrix : public Matrix {
	private:
		BINARY_TYPE * data;
	public:
		BinaryMatrix(long width, long height, BINARY_TYPE* data = nullptr);
		FloatMatrix operator*(FloatMatrix& b);
		FloatVector operator*(FloatVector& b);
		BINARY_TYPE * getData();
		bool assertMulVector(FloatVector &b, FloatVector &c);
		bool assertMulMatrix(FloatMatrix &b, FloatMatrix &c);
};
