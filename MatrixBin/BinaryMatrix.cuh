#pragma once

#include "Matrix.cuh"
#include "FloatMatrix.cuh"
#include "FloatVector.cuh"

class BinaryMatrix : public Matrix {
	private:
		unsigned int * data;
	public:
		BinaryMatrix(long width, long height, unsigned int* data = nullptr);
		FloatMatrix* operator*(FloatMatrix& b);
		FloatVector* operator*(FloatVector& b);
		unsigned int * getData();

};
