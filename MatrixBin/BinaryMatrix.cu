#include "BinaryMatrix.cuh"

#include "cuda_header.cuh"

#include "stdio.h"

BinaryMatrix::BinaryMatrix(long width, long height, BINARY_TYPE* data) :Matrix(width, height) {
	this->data = data;
}

FloatMatrix BinaryMatrix::operator*(FloatMatrix& b) {
	//Call CUDA
	FLOAT_MAT_TYPE* data = cuda_mul_matrix(this->getWidth(), this->getHeight(), this->getData(), b.getWidth(), b.getHeight(), b.getData());
	return FloatMatrix(b.getWidth(), this->getHeight(), data);
}

FloatVector BinaryMatrix::operator*(FloatVector& b) {
	//Call CUDA
	FLOAT_VEC_TYPE* data = cuda_mul_vector(this->getWidth(), this->getHeight(), this->getData(), b.getHeight(), b.getData());
	return FloatVector(this->getHeight(), data);
}

bool BinaryMatrix::assertMulMatrix(FloatMatrix &b, FloatMatrix &c) {
	FLOAT_MAT_TYPE* newCData = new FLOAT_MAT_TYPE[c.getWidth() * c.getHeight()];
	for (unsigned long int i = 0; i < c.getWidth() * c.getHeight(); i++) {
		newCData[i] = 0.0f;
	}
	for (unsigned long int i = 0; i < c.getHeight(); i++) {
		for (unsigned long int j = 0; j < c.getWidth(); j++) {
			for (unsigned long int k = 0; k < this->getWidth()/unsigned int (BITS_IN_BIN); k++) {
				BINARY_TYPE word = this->data[i * (this->getWidth() / BITS_IN_BIN) + k];
				for (unsigned int shift = 0; shift < BITS_IN_BIN; shift++) {
					if ((0x1 << shift) & word) {
						newCData[i * c.getWidth() + j] += b.getData()[(k*BITS_IN_BIN + shift) * b.getWidth() + j];
					}
				}
			}
		}
	}

	for (unsigned long i = 0; i < c.getWidth() * c.getHeight(); i++) {
		if (c.getData()[i] != newCData[i]) {
			printf("Expected %f at index [%d, %d], got %f.\n", newCData[i], i / c.getWidth(), i % c.getWidth(), c.getData()[i]);
			delete[] newCData;
			return false;
		}
	}

	delete [] newCData;
	return true;
}

bool BinaryMatrix::assertMulVector(FloatVector &b, FloatVector &c) {
	printf("Asserting maxtrix*vector on CPU\n");

	FLOAT_VEC_TYPE* newCData = new FLOAT_VEC_TYPE[c.getHeight()];
	for (unsigned long int i = 0; i < c.getHeight(); i++) {
		newCData[i] = 0.0f;
	}
	for (unsigned long int i = 0; i < c.getHeight(); i++) {
		for (unsigned long int k = 0; k < this->getWidth() / unsigned int(BITS_IN_BIN); k++) {
			BINARY_TYPE word = this->data[i * (this->getWidth() / BITS_IN_BIN) + k];
			for (unsigned int shift = 0; shift < BITS_IN_BIN; shift++) {
				if ((0x1 << shift) & word) {
					newCData[i] += b.getData()[k*BITS_IN_BIN + shift];
				}
			}
		}
	}

	for (unsigned long i = 0; i < c.getHeight(); i++) {
		if (c.getData()[i] != newCData[i]) {
			printf("\tExpected %f at index [%d], got %f.\n", newCData[i], i, c.getData()[i]);
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

FloatMatrix* BinaryMatrix::toFloatMatrix() {
	FLOAT_MAT_TYPE* data = new FLOAT_MAT_TYPE[this->getWidth() * this->getHeight()];
	for (unsigned long int i = 0; i < this->getWidth() * this->getHeight(); i++) {
		data[i] = 0.0f;
		if (this->data[i / BITS_IN_BIN] & (0x1 << i%BITS_IN_BIN))
			data[i] = 1.0f;
	}
	return new FloatMatrix(this->getWidth(), this->getHeight(), data);
}

BinaryMatrix::~BinaryMatrix() {
	if (this->data)
		delete this->data;
}