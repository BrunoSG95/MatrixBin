#include <stdio.h>

#include "Test.h"

void defMatrixSize(int* w, int* h) {
	printf("\tWidth: ");
	scanf("%d", w);
	printf("\tHeight: ");
	scanf("%d", h);
}

int main() {
	bool exit = false;
	Test test = Test();
	while (!exit) {
		printf("\n 1.Load matrix A \n 2.Load matrix B\n 3.Load vector\n 4.Multipy matrix-matrix\n 5.Multiply matrix-vector\n 6.Multiply matrix-matrix with CUDA\n 7.Toggle assertion\n 8.Exit\nSelect option: ");
		int opt = 0;
		scanf("%d", &opt);
		switch (opt) {
			case 1: {
				int width = 0;
				int height = 0;
				defMatrixSize(&width, &height);
				test.loadA(width, height);
			} break;
			case 2: {
				int width = 0;
				int height = 0;
				defMatrixSize(&width, &height);
				test.loadB(width, height);
			} break;
			case 3: {
				printf("\tHeight: ");
				int height = 0;
				scanf("%d", &height);
				test.loadV(height);
			} break;
			case 4: {
				printf("\nTesting...\n");
				if (test.testMat()) {
					printf("Success\n");
				}
				else {
					printf("Failure\n");
				}
			} break;
			case 5: {
				printf("\nTesting...\n");
				if (test.testVec()) {
					printf("Success\n");
				}
				else {
					printf("Failure\n");
				}
				
			} break;
			case 6: {
				printf("\nTesting...\n");
				if (test.testCUDA()) {
					printf("Success\n");
				}
				else {
					printf("Failure\n");
				}
			} break;
			case 7: {
				test.assert = !test.assert;
				if (test.assert)
					printf("\nAssertion ON\n");
				else
					printf("\nAssertion OFF\n");
			}break;
			case 8: {
				exit = true;
			} break;
			default: {
				printf("Invalid option.\n");
			}break;
		}
	}
	return 0;
}