#include <stdio.h>

#include "Test.h"

int main() {
	if (runAllTests()) {
		printf("Success");
	}
	else {
		printf("Failure");
	}
	getchar();
	return 0;
}