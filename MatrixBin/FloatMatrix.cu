#include "FloatMatrix.cuh"

FloatMatrix::FloatMatrix(long width, long height, FLOAT_MAT_TYPE* data) :Matrix(width, height) {
	this->data = data;
}

FLOAT_MAT_TYPE * FloatMatrix::getData() {
	return data;
}