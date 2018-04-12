#pragma once
#include "Matrix.cuh"

Matrix::Matrix(long width, long height) {
	this->width = width;
	this->height = height;
}

long Matrix::getWidth() {
	return width;
}

long Matrix::getHeight() {
	return height;
}