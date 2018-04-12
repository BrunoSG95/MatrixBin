#pragma once

#include "Matrix.cuh"

class FloatMatrix : public Matrix {
private:
	float * data;
public:
	FloatMatrix(long width, long height, float* data = nullptr);
	float * getData();

};
