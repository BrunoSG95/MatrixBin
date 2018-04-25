#include <stdio.h>

#include "Test.h"

int main() {
	if (runAllTests()) {
		printf("\nSuccess\n");
	}
	else {
		printf("\nFailure\n");
	}
	getchar();
	return 0;
}