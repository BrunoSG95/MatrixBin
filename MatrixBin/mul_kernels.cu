#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_header.cuh"
#include "Timer.h"

#define BLOCK_THREADS_MATRIX 32
#define BLOCK_THREADS_VECTOR 32

__global__ void simpleMulVector(BINARY_TYPE* aData, FLOAT_VEC_TYPE* bData, FLOAT_VEC_TYPE* cData, unsigned long int aWidth, unsigned long int aHeight) {
	
	BINARY_TYPE masks[] = { 0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x100,0x200,0x400,0x800,0x1000,0x2000,0x4000,0x8000,0x10000,0x20000,0x40000,0x80000,0x100000,0x200000,0x400000,0x800000,0x1000000,0x2000000,0x4000000,0x8000000,0x10000000,0x20000000,0x40000000,0x80000000,0x100000000,0x200000000,0x400000000,0x800000000,0x1000000000,0x2000000000,0x4000000000,0x8000000000,0x10000000000,0x20000000000,0x40000000000,0x80000000000,0x100000000000,0x200000000000,0x400000000000,0x800000000000,0x1000000000000,0x2000000000000,0x4000000000000,0x8000000000000,0x10000000000000,0x20000000000000,0x40000000000000,0x80000000000000,0x100000000000000,0x200000000000000,0x400000000000000,0x800000000000000,0x1000000000000000,0x2000000000000000,0x4000000000000000,0x8000000000000000 };
	
	unsigned long int pos = blockIdx.x*blockDim.x + threadIdx.x;

	FLOAT_VEC_TYPE partial_result = 0;

	for (unsigned long int k = 0; k < aWidth / unsigned int(BITS_IN_BIN); k++) {

		BINARY_TYPE word = aData[pos * (aWidth / unsigned int(BITS_IN_BIN)) + k];

		for (unsigned long int shift = 0; shift < BITS_IN_BIN; shift++) {
			if (word & masks[shift]) {
				partial_result += bData[k * unsigned int(BITS_IN_BIN) + shift];
			}
		}
	}

	cData[pos] = partial_result;

}

FLOAT_VEC_TYPE* cuda_mul_vector(long aWidth, long aHeight, BINARY_TYPE* aData, long bHeight, FLOAT_VEC_TYPE* bData) {
	
	// Initialize result
	FLOAT_VEC_TYPE* cData = new FLOAT_VEC_TYPE[aHeight];

	size_t aSize = aWidth * aHeight / sizeof(BINARY_TYPE);
	size_t bSize = bHeight * sizeof(FLOAT_VEC_TYPE);
	size_t cSize = aHeight * sizeof(FLOAT_VEC_TYPE);

	BINARY_TYPE* A_dev;
	FLOAT_VEC_TYPE* B_dev, *C_dev;

	Timer alloc_dev, exec, alloc_host;

	// Allocate and copy (device)
	alloc_dev.start();

	cudaMalloc(&A_dev, aSize);
	cudaMalloc(&B_dev, bSize);
	cudaMalloc(&C_dev, cSize);

	cudaMemcpy(A_dev, aData, aSize, cudaMemcpyHostToDevice);
	cudaMemcpy(B_dev, bData, bSize, cudaMemcpyHostToDevice);
	cudaMemset(C_dev, 0, cSize);

	alloc_dev.stop("Allocate and copy to device: ");

	// Configure blocks of threads
	unsigned long int gridx = aHeight / BLOCK_THREADS_VECTOR;
	dim3 gridSize(gridx);
	dim3 blockSize(BLOCK_THREADS_VECTOR);

	exec.start();
	simpleMulVector<<<gridSize, blockSize>>>(A_dev, B_dev, C_dev, aWidth, aHeight);
	cudaDeviceSynchronize();

	exec.stop("Execution: ");

	// Copy to host
	alloc_host.start();

	cudaMemcpy(cData, C_dev, cSize, cudaMemcpyDeviceToHost);

	alloc_host.stop("Allocate and copy to host: ");

	// Free memory (device)
	cudaFree(A_dev);
	cudaFree(B_dev);
	cudaFree(C_dev);

	return cData;
}

__global__ void simpleMul(BINARY_TYPE* aData, FLOAT_MAT_TYPE* bData, FLOAT_MAT_TYPE* cData, unsigned long int aWidth, unsigned long int aHeight, unsigned long int bWidth) {

	BINARY_TYPE masks[] = { 0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x100,0x200,0x400,0x800,0x1000,0x2000,0x4000,0x8000,0x10000,0x20000,0x40000,0x80000,0x100000,0x200000,0x400000,0x800000,0x1000000,0x2000000,0x4000000,0x8000000,0x10000000,0x20000000,0x40000000,0x80000000,0x100000000,0x200000000,0x400000000,0x800000000,0x1000000000,0x2000000000,0x4000000000,0x8000000000,0x10000000000,0x20000000000,0x40000000000,0x80000000000,0x100000000000,0x200000000000,0x400000000000,0x800000000000,0x1000000000000,0x2000000000000,0x4000000000000,0x8000000000000,0x10000000000000,0x20000000000000,0x40000000000000,0x80000000000000,0x100000000000000,0x200000000000000,0x400000000000000,0x800000000000000,0x1000000000000000,0x2000000000000000,0x4000000000000000,0x8000000000000000 };

	unsigned long int row = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned long int col = blockIdx.x*blockDim.x + threadIdx.x;

	FLOAT_MAT_TYPE partial_result = 0;

	for (unsigned long int i = 0; i < aWidth/ unsigned int(BITS_IN_BIN); i++) {

		BINARY_TYPE word = aData[row * aWidth/ unsigned int(BITS_IN_BIN) + i];

		for (unsigned long int shift = 0; shift < BITS_IN_BIN; shift++) {
			if (word & masks[shift]){
				partial_result += bData[ (i * BITS_IN_BIN + shift) * bWidth + col];
			}
		}
	}

	cData[row * aWidth + col] = partial_result;

}

FLOAT_MAT_TYPE* cuda_mul_matrix(long aWidth, long aHeight, BINARY_TYPE* aData, long bWidth, long bHeight, FLOAT_MAT_TYPE* bData) {
	
	// Initialize result
	FLOAT_MAT_TYPE* cData = new FLOAT_MAT_TYPE[aHeight * bWidth];

	size_t aSize = aWidth * aHeight / sizeof(BINARY_TYPE);
	size_t bSize = bWidth * bHeight * sizeof(FLOAT_MAT_TYPE);
	size_t cSize = bWidth * aHeight * sizeof(FLOAT_MAT_TYPE);
	
	BINARY_TYPE* A_dev;
	FLOAT_MAT_TYPE* B_dev, *C_dev;

	Timer alloc_dev, exec, alloc_host;

	// Allocate and copy (device)
	alloc_dev.start();

	cudaMalloc(&A_dev, aSize);
	cudaMalloc(&B_dev, bSize);
	cudaMalloc(&C_dev, cSize);

	cudaMemcpy(A_dev, aData, aSize, cudaMemcpyHostToDevice);
	cudaMemcpy(B_dev, bData, bSize, cudaMemcpyHostToDevice);
	cudaMemset(C_dev, 0, cSize);

	alloc_dev.stop("Allocate and copy to device: ");

	// Configure blocks of threads
	unsigned long int gridx = bWidth / BLOCK_THREADS_MATRIX;
	unsigned long int gridy = aHeight / BLOCK_THREADS_MATRIX;
	dim3 gridSize(gridx, gridy);
	dim3 blockSize(BLOCK_THREADS_MATRIX, BLOCK_THREADS_MATRIX);

	exec.start();
	simpleMul<<<gridSize, blockSize>>>(A_dev, B_dev, C_dev, aWidth, aHeight, bWidth);
	cudaDeviceSynchronize();

	exec.stop("Execution: ");

	// Copy to host
	alloc_host.start();

	cudaMemcpy(cData, C_dev, cSize, cudaMemcpyDeviceToHost);

	alloc_host.stop("Allocate and copy to host: ");

	// Free memory (device)
	cudaFree(A_dev);
	cudaFree(B_dev);
	cudaFree(C_dev);
	
	return cData;
}