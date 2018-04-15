#include "Timer.h"

Timer::Timer() {

}

void Timer::start() {
	this->start_t = clock();
}

void Timer::stop(string prefix) {
	string print = prefix + to_string(1000.0f * (clock() - this->start_t) / float(CLOCKS_PER_SEC)) + " ms.\n";
	printf(print.c_str());
}