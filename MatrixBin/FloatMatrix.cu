#include "FloatMatrix.cuh"

#include <stdio.h>

#include "cuda_header.cuh"

FloatMatrix::FloatMatrix(long width, long height, FLOAT_MAT_TYPE* data) :Matrix(width, height) {
	this->data = data;
}

FLOAT_MAT_TYPE * FloatMatrix::getData() {
	return data;
}

FloatMatrix FloatMatrix::operator*(FloatMatrix& b) {
	FLOAT_MAT_TYPE* data = cublas_mul_matrix(this->getWidth(), this->getHeight(), this->getData(), b.getWidth(), b.getHeight(), b.getData());
	return FloatMatrix(b.getWidth(), this->getHeight(), data);
}

bool FloatMatrix::operator==(FloatMatrix& b) {
	if (this->getWidth() != b.getWidth() || this->getHeight() != b.getHeight())
		return false;
	for (unsigned long int i = 0; i < this->getWidth() * this->getHeight(); i++)
		if (this->data[i] != b.getData()[i]) {
			printf("Expected %f, got %f", data[i], b.getData()[i]);
			return false;
		}
	return true;
}

FloatMatrix::~FloatMatrix() {
	if (this->data)
		delete this->data;
}