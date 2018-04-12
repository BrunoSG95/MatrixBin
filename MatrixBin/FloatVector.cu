#include "FloatVector.cuh"

FloatVector::FloatVector(long height, float* data) {
	this->height = height;
	this->data = data;
}

long FloatVector::getHeight() {
	return height;
}

float * FloatVector::getData() {
	return data;
}