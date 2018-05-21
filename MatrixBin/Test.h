#pragma once

#include "BinaryMatrix.cuh"
#include "FloatMatrix.cuh"

class Test {
	private: 
		BinaryMatrix* a;
		FloatMatrix* b;
		FloatVector* v;
	public: 
		Test();
		void loadA(unsigned long int width, unsigned long int height);
		void loadB(unsigned long int width, unsigned long int height);
		void loadV(unsigned long int height);
		bool testMat(); 
		bool testVec();
		bool testCUDA();
		~Test();
};