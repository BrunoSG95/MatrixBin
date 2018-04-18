#include "BinaryMatrix.cuh"

#include "cuda_header.cuh"

#include "stdio.h"

BinaryMatrix::BinaryMatrix(long width, long height, BINARY_TYPE* data) :Matrix(width, height) {
	this->data = data;
}

FloatMatrix BinaryMatrix::operator*(FloatMatrix& b) {
	//Call CUDA
	float* data = cuda_mul_matrix(this->getWidth(), this->getHeight(), this->getData(), b.getWidth(), b.getHeight(), b.getData());
	return FloatMatrix(b.getWidth(), this->getHeight(), data);
}

FloatVector BinaryMatrix::operator*(FloatVector& b) {
	//Call CUDA
	float* data = cuda_mul_vector(this->getWidth(), this->getHeight(), this->getData(), b.getHeight(), b.getData());
	return FloatVector(this->getHeight(), data);
}

bool BinaryMatrix::assertMulMatrix(FloatMatrix &b, FloatMatrix &c) {
	float* newCData = new float[c.getWidth() * c.getHeight()];
	for (unsigned long int i = 0; i < c.getWidth() * c.getHeight(); i++) {
		newCData[i] = 0.0f;
	}
	for (unsigned long int i = 0; i < c.getWidth(); i++) {
		for (unsigned long int j = 0; j < c.getHeight(); j++) {
			for (unsigned long int k = 0; k < this->getWidth()/BITS_IN_BIN; k++) {
				BINARY_TYPE word = this->data[j * (this->getWidth() / BITS_IN_BIN) + k];
				for (unsigned int shift = 0; shift < BITS_IN_BIN; shift++) {
					if ((0x1 << shift) & word)
						newCData[j * c.getWidth() + i] += b.getData()[k * b.getWidth() + (i * BITS_IN_BIN + shift) ];
				}
			}
		}
	}

	for (unsigned long i = 0; i < c.getWidth() * c.getHeight(); i++) {
		if (c.getData()[i] != newCData[i]) {
			printf("Expected %f, got %f.\n", newCData[i], c.getData()[i]);
			delete[] newCData;
			return false;
		}
	}

	delete [] newCData;
	return true;
}

bool BinaryMatrix::assertMulVector(FloatVector &b, FloatVector &c) {
	printf("Assertin maxtrix*matrix on CPU\n");

	float* newCData = new float[c.getHeight()];
	for (unsigned long i = 0; i < c.getHeight(); i++) {
		newCData[i] = 0.0f;
	}
	for (unsigned long i = 0; i < c.getHeight(); i++){
		for (unsigned long k = 0; k < this->getWidth() / (8 * sizeof(BINARY_TYPE)); k++) {
			for (unsigned int shift = 0; shift < (8 * sizeof(BINARY_TYPE)); shift++) {
				if ((0x1 << shift) & this->data[i * this->getWidth() / (sizeof(BINARY_TYPE) * 8) + k])
					newCData[i] += b.getData()[k * (8 * sizeof(BINARY_TYPE)) + shift];
			}
		}
	}

	for (unsigned long i = 0; i < c.getHeight(); i++) {
		if (c.getData()[i] != newCData[i]) {
			delete[] newCData;
			return false;
		}
	}

	delete[] newCData;
	return true;
}

BINARY_TYPE * BinaryMatrix::getData() {
	return data;
}