#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_header.cuh"
#include "Timer.h"

#define BLOCK_THREADS_MATRIX 32
#define BLOCK_THREADS_VECTOR 32

__global__ void simpleMulVector(BINARY_TYPE* aData, float* bData, float* cData, unsigned long int aWidth, unsigned long int aHeight) {

	unsigned long int pos = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned int bits_in_cell = 8 * sizeof(BINARY_TYPE);

	float partial_result = 0;

	for (unsigned long int i = 0; i < aHeight / bits_in_cell; i++) {

		BINARY_TYPE word = aData[pos * aWidth / bits_in_cell + i];

		for (unsigned long int shift = 0; shift < bits_in_cell; shift++) {
			if (word & (0x1 << shift)) {
				partial_result += bData[i * bits_in_cell + shift];
			}
		}
	}

	cData[pos] = partial_result;

}

float* cuda_mul_vector(long aWidth, long aHeight, BINARY_TYPE* aData, long bHeight, float* bData) {
	
	// Initialize result
	float* cData = new float[aHeight];

	size_t aSize = aWidth * aHeight / sizeof(BINARY_TYPE);
	size_t bSize = bHeight * sizeof(float);
	size_t cSize = aHeight * sizeof(float);

	BINARY_TYPE* A_dev;
	float* B_dev, *C_dev;

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
	unsigned long int gridx = aHeight / (BLOCK_THREADS_VECTOR*BLOCK_THREADS_VECTOR);
	dim3 gridSize(gridx);
	dim3 blockSize(BLOCK_THREADS_MATRIX, BLOCK_THREADS_MATRIX);

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

__global__ void simpleMul(BINARY_TYPE* aData, float* bData, float* cData, unsigned long int aWidth, unsigned long int aHeight, unsigned long int bWidth) {

	unsigned long int row = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned long int col = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned int bits_in_cell = 8 * sizeof(BINARY_TYPE);

	float partial_result = 0;

	for (unsigned long int i = 0; i < aHeight/bits_in_cell; i++) {

		BINARY_TYPE word = aData[row * aWidth/bits_in_cell + i];

		for (unsigned long int shift = 0; shift < bits_in_cell; shift++) {
			if (word & (0x1 << shift)){
				partial_result += bData[(i * bWidth + col)*bits_in_cell];
			}
		}
	}

	cData[row * aWidth + col] = partial_result;

}

float* cuda_mul_matrix(long aWidth, long aHeight, BINARY_TYPE* aData, long bWidth, long bHeight, float* bData) {
	
	// Initialize result
	float* cData = new float[aHeight * bWidth];

	size_t aSize = aWidth * aHeight / sizeof(BINARY_TYPE);
	size_t bSize = bWidth * bHeight * sizeof(float);
	size_t cSize = bWidth * aHeight * sizeof(float);
	
	BINARY_TYPE* A_dev;
	float* B_dev, *C_dev;

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