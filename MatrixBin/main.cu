#include <stdio.h>

#include "Test.h"

int main() {
	bool exit = false;
	Test test = Test();
	while (!exit) {
		printf("\n 1.Load matrix A \n 2.Load matrix B\n 3.Load vector\n 4.Multipy matrix-matrix\n 5.Multiply matrix-vector\n 6.Multiply matrix-matrix with CUDA\n 7.Exit\nSelect option: ");
		int opt = 0;
		scanf("%d", &opt);
		switch (opt) {
			case 1: {
				printf("\nSelect size:\n 1. 1024x1024\n 2. 2048 x 2048\n 3.2048x1024\nSelect option: ");
				int opt2 = 0;
				scanf("%d", &opt2);
				int width, height;
				switch (opt2) {
					case 1: {
						width = 1024;
						height = 1024;
					} break;
					case 2: {
						width = 2048;
						height = 2048;
					} break;
					case 3: {
						width = 2048;
						height = 1024;
					} break;
					default: {
						width = 1024;
						height = 1024;
					} break;
				}
				test.loadA(width, height);
			} break;
			case 2: {
				printf("\nSelect size:\n 1. 1024x1024\n 2. 2048 x 2048\n 3.1024x2048\nSelect option: ");
				int opt2 = 0;
				scanf("%d", &opt2);
				int width, height;
				switch (opt2) {
				case 1: {
					width = 1024;
					height = 1024;
				} break;
				case 2: {
					width = 2048;
					height = 2048;
				} break;
				case 3: {
					width = 1024;
					height = 2048;
				} break;
				default: {
					width = 1024;
					height = 1024;
				} break;
				}
				test.loadB(width, height);
			} break;
			case 3: {
				printf("\nSelect size:\n 1. 1024\n 2. 2048\n 3.4096\nSelect option: ");
				int opt2 = 0;
				scanf("%d", &opt2);
				int height;
				switch (opt2) {
				case 1: {
					height = 1024;
				} break;
				case 2: {
					height = 2048;
				} break;
				case 3: {
					height = 4096;
				} break;
				default: {
					height = 1024;
				} break;
				}
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
				exit = true;
			}break;
			default: {
				printf("Ingrese una opción válida.\n");
			}break;
		}
	}
	return 0;
}