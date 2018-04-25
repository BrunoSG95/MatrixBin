#pragma once

#define FLOAT_VEC_TYPE float

class FloatVector{
private:
	long height;
	FLOAT_VEC_TYPE * data;
public:
	FloatVector(long height, FLOAT_VEC_TYPE* data = nullptr);
	long getHeight();
	FLOAT_VEC_TYPE * getData();
};
