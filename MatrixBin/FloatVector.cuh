#pragma once

class FloatVector{
private:
	long height;
	float * data;
public:
	FloatVector(long height, float* data = nullptr);
	long getHeight();
	float * getData();
};
