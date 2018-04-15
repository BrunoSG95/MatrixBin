#include "BinaryMatrix.cuh"
#include "cuda_header.cuh"

BinaryMatrix::BinaryMatrix(long width, long height, BINARY_TYPE* data) :Matrix(width, height) {
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

bool BinaryMatrix::assertMulMatrix(FloatMatrix &b, FloatMatrix &c) {
	float* newCData = new float[c.getWidth() * c.getHeight()];
	for (unsigned long i = 0; i < c.getWidth() * c.getHeight(); i++) {
		newCData[i] = 0.0f;
	}
	for (unsigned long i = 0; i < c.getWidth(); i++) {
		for (unsigned long j = 0; j < c.getHeight(); j++) {
			for (unsigned long k = 0; k < this->getWidth()/(8*sizeof(BINARY_TYPE)); k++) {
				for (unsigned int shift = 0; shift < (8 * sizeof(BINARY_TYPE)); shift++) {
					if ((0x1 << shift) & this->data[j * this->getWidth()/(sizeof(BINARY_TYPE) * 8) + k])
						newCData[j * c.getWidth() + i] += b.getData()[(k * b.getWidth() + i)* 8 * sizeof(BINARY_TYPE)];
				}
			}
		}
	}

	for (unsigned long i = 0; i < c.getWidth() * c.getHeight(); i++) {
		if (c.getData()[i] != newCData[i]) {
			delete[] newCData;
			return false;
		}
	}

	delete [] newCData;
	return true;
}

bool BinaryMatrix::assertMulVector(FloatVector &b, FloatVector &c) {
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