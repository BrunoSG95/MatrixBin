#include "Test.h"

#include <stdio.h>

#include "BinaryMatrix.cuh"

bool runAllTests() {
	return runBasicTestMat() && runBasicTestVec();
}

bool runBasicTestMat() {
	printf("Running basic test:\n\tbinmat: 1024*1024 fmat: 1024*1024\n");

	unsigned int width = 1024;
	unsigned int height = 1024;

	BINARY_TYPE* bdata = new BINARY_TYPE[(width*height)/BITS_IN_BIN];
	FLOAT_MAT_TYPE* fdata = new FLOAT_MAT_TYPE[width * height];
	
	for (unsigned long int i = 0; i < width*height; i++) {
		bdata[i / BITS_IN_BIN] = 0x0;
		fdata[i] = 1.0;
	}

	for (unsigned long int i = 0; i < (width * height)/ BITS_IN_BIN; i++) {
		bdata[i] = 0xFFFFFFFF;
	}

	BinaryMatrix bMat = BinaryMatrix(1024, 1024, bdata);
	FloatMatrix fMat = FloatMatrix(1024, 1024, fdata);

	FloatMatrix res = bMat * fMat;

	return bMat.assertMulMatrix(fMat, res);
}

bool runBasicTestVec() {
	printf("Running basic test:\n\tbinmat: 1024*1024 fvec: 1024\n");

	unsigned int width = 1024;
	unsigned int height = 1024;

	BINARY_TYPE* bdata = new BINARY_TYPE[(width*height) / BITS_IN_BIN];
	FLOAT_VEC_TYPE* fdata = new FLOAT_VEC_TYPE[height];

	for (unsigned long int i = 0; i < height; i++) {
		fdata[i] = 1.0;
	}

	for (unsigned long int i = 0; i < (width * height) / BITS_IN_BIN; i++) {
		bdata[i] = 0xFFFFFFFF;
	}

	BinaryMatrix bMat = BinaryMatrix(1024, 1024, bdata);
	FloatVector fVec = FloatVector(1024, fdata);

	FloatVector res = bMat * fVec;

	return bMat.assertMulVector(fVec, res);
}