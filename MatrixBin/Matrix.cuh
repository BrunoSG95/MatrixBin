#pragma once

class Matrix {
	private:
		long width, height;
	public:
		Matrix(long width, long height);
		long getWidth();
		long getHeight();
};