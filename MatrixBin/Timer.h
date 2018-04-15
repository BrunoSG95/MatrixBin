#pragma once

#include <time.h>
#include <string>

using namespace std;

class Timer {
	private:
		clock_t start_t;
	public:
		Timer();
		void start();
		void stop(string prefix);
};