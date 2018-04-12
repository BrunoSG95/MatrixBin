#include "BinaryMatrix.cuh"
#include "cuda_header.cuh"

BinaryMatrix::BinaryMatrix(long width, long height, unsigned int* data) :Matrix(width, height) {
	this->data = data;
}

FloatMatrix* BinaryMatrix::operator*(FloatMatrix& b) {
	//Call CUDA
	float* data = cuda_mul_matrix(this->getWidth(), this->getHeight(), this->getData(), b.getWidth(), b.getHeight(), b.getData());
	return new FloatMatrix(b.getWidth(), this->getHeight(), data);
}

FloatVector* BinaryMatrix::operator*(FloatVector& b) {
	//Call CUDA
	float* data = cuda_mul_vector(this->getWidth(), this->getHeight(), this->getData(), b.getHeight(), b.getData());
	return new FloatVector(this->getHeight(), data);
}

unsigned int * BinaryMatrix::getData() {
	return data;
}