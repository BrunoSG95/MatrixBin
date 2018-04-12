#include "FloatMatrix.cuh"

FloatMatrix::FloatMatrix(long width, long height, float* data) :Matrix(width, height) {
	this->data = data;
}

float * FloatMatrix::getData() {
	return data;
}