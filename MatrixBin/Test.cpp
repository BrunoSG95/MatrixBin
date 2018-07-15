#include "Test.h"

#include <stdio.h>

Test::Test() {
	a = nullptr;
	b = nullptr;
	v = nullptr;
	this->assert = false;
}

void Test::loadA(unsigned long int width, unsigned long int height) {
	if (a) {
		delete a;
		a = nullptr;
	}
	BINARY_TYPE* bdata = new BINARY_TYPE[(width*height) / BITS_IN_BIN];
	for (unsigned long int i = 0; i < (width * height) / BITS_IN_BIN; i++) {
		bdata[i] = 0xFFFFFFFF;
	}
	this->a = new BinaryMatrix(width, height, bdata);
}

void Test::loadB(unsigned long int width, unsigned long int height) {
	if (b) {
		delete b;
		b = nullptr;
	}
	FLOAT_MAT_TYPE* fdata = new FLOAT_MAT_TYPE[width * height];
	for (unsigned long int i = 0; i < width*height; i++) {
		fdata[i] = 1.0;
	}
	this->b = new FloatMatrix(width, height, fdata);
}

void Test::loadV(unsigned long int height) {
	if (v) {
		delete v;
		v = nullptr;
	}
	FLOAT_VEC_TYPE* fdata = new FLOAT_MAT_TYPE[height];
	for (unsigned long int i = 0; i < height; i++) {
		fdata[i] = 1.0;
	}
	this->v = new FloatVector(height, fdata);
}

bool Test::testMat() {
	if (a->getWidth() == b->getHeight()) {
		FloatMatrix res = *a * *b;
		if (assert)
			return a->assertMulMatrix(*b, res);
		else
			return true;
	}
	else {
		printf("Incompatible sizes");
	}
}

bool Test::testVec() {
	if (a && v && a->getWidth() == v->getHeight()) {
		FloatVector res = *a * *v;
		if (assert)
			return a->assertMulVector(*v, res);
		else
			return true;
	}
	else {
		printf("Incompatible sizes");
	}
}

bool Test::testCUDA() {
	if (a->getWidth() == b->getHeight()) {
		FloatMatrix* conv = this->a->toFloatMatrix();
		FloatMatrix result = conv->operator*(*b);
		delete conv;
		if (assert)
			return a->assertMulMatrix(*b, result);
		else
			return true;
	}
	else {
		printf("Incompatible sizes");
	}
}

Test::~Test() {
	if (a)
		delete a;
	if (b)
		delete b;
	if (v)
		delete v;
}