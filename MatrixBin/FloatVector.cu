#include "FloatVector.cuh"

FloatVector::FloatVector(long height, FLOAT_VEC_TYPE* data) {
	this->height = height;
	this->data = data;
}

long FloatVector::getHeight() {
	return height;
}

FLOAT_VEC_TYPE * FloatVector::getData() {
	return data;
}