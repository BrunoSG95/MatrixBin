#include "Test.h"

#include "BinaryMatrix.cuh"

bool runAllTests() {
	runBasicTest();
}

bool runBasicTest() {
	unsigned int width = 1024;
	unsigned int height = 1024;

	BINARY_TYPE* bdata = new BINARY_TYPE[(width*height)/BITS_IN_BIN];
	float* fdata = new float[width * height];
	
	for (unsigned long int i = 0; i < width*height; i++) {
		bdata[i] = 0x0;
		fdata[i] = 1.0f;
	}

	for (unsigned long int i = 0; i < (width * height); i++) {
		bdata[i/BITS_IN_BIN] |= 0x1 << i%BITS_IN_BIN;
	}

	BinaryMatrix bMat = BinaryMatrix(1024, 1024, bdata);
	FloatMatrix fMat = FloatMatrix(1024, 1024, fdata);

	FloatMatrix res = bMat * fMat;

	return bMat.assertMulMatrix(fMat, res);
}