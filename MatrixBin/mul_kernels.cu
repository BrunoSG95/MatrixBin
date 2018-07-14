#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_header.cuh"
#include "Timer.h"

#define BLOCK_THREADS_MATRIX 32
#define BLOCK_THREADS_VECTOR 32
#define SHARED_LIMIT 512 // Limit (floats) shared memory size to mantain high occupancy

__global__ void sharedMulVector(BINARY_TYPE* aData, FLOAT_VEC_TYPE* bData, FLOAT_VEC_TYPE* cData, unsigned long int aWidth, unsigned long int aHeight, unsigned int sharedSize) {
	BINARY_TYPE masks[] = { 0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x100,0x200,0x400,0x800,0x1000,0x2000,0x4000,0x8000,0x10000,0x20000,0x40000,0x80000,0x100000,0x200000,0x400000,0x800000,0x1000000,0x2000000,0x4000000,0x8000000,0x10000000,0x20000000,0x40000000,0x80000000,0x100000000,0x200000000,0x400000000,0x800000000,0x1000000000,0x2000000000,0x4000000000,0x8000000000,0x10000000000,0x20000000000,0x40000000000,0x80000000000,0x100000000000,0x200000000000,0x400000000000,0x800000000000,0x1000000000000,0x2000000000000,0x4000000000000,0x8000000000000,0x10000000000000,0x20000000000000,0x40000000000000,0x80000000000000,0x100000000000000,0x200000000000000,0x400000000000000,0x800000000000000,0x1000000000000000,0x2000000000000000,0x4000000000000000,0x8000000000000000 };
	
	extern __shared__ float shared[];
	FLOAT_VEC_TYPE* vector = (FLOAT_VEC_TYPE*) shared;

	unsigned long int pos = blockIdx.x*blockDim.x + threadIdx.x;

	FLOAT_VEC_TYPE partial_result = 0;

	for (unsigned long int step = 0; step < aWidth / sharedSize; step++) {
		for (unsigned int i = threadIdx.x; i < sharedSize; i += blockDim.x) {
			vector[i] = bData[step * sharedSize + i];
		}
		__syncthreads();
		for (unsigned long int inner = 0; inner < sharedSize / unsigned int(BITS_IN_BIN); inner++) {
			unsigned long int offset = step * (sharedSize/ unsigned int(BITS_IN_BIN)) + inner;
			BINARY_TYPE word = aData[pos * (aWidth / unsigned int(BITS_IN_BIN)) + offset];
			# pragma unroll
			for (unsigned long int shift = 0; shift < BITS_IN_BIN; shift++) {
				if (word & masks[shift]) {
					partial_result += vector[inner * unsigned int(BITS_IN_BIN) + shift];
				}
			}
		}
		__syncthreads();
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
	unsigned int sharedSize = aWidth;
	if (sharedSize > unsigned int (SHARED_LIMIT))
		sharedSize = unsigned int(SHARED_LIMIT);
	unsigned int sharedSizeInBytes = sharedSize * sizeof(FLOAT_VEC_TYPE);
	exec.start();
	sharedMulVector<<<gridSize, blockSize, sharedSizeInBytes >>>(A_dev, B_dev, C_dev, aWidth, aHeight, sharedSize);
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

__global__ void optimizedMul(BINARY_TYPE* aData, FLOAT_MAT_TYPE* bData, FLOAT_MAT_TYPE* cData, unsigned long int aWidth, unsigned long int aHeight, unsigned long int bWidth) {
	
	// Declare tiles (A&B)
	extern __shared__ float shared[];
	BINARY_TYPE* tileA = (BINARY_TYPE*) shared;
	FLOAT_MAT_TYPE* tileB = (FLOAT_MAT_TYPE*)&tileA[((blockDim.x + 1) * blockDim.y)/ unsigned int (BITS_IN_BIN)]; // +1 for padding

	BINARY_TYPE masks[] = { 0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x100,0x200,0x400,0x800,0x1000,0x2000,0x4000,0x8000,0x10000,0x20000,0x40000,0x80000,0x100000,0x200000,0x400000,0x800000,0x1000000,0x2000000,0x4000000,0x8000000,0x10000000,0x20000000,0x40000000,0x80000000,0x100000000,0x200000000,0x400000000,0x800000000,0x1000000000,0x2000000000,0x4000000000,0x8000000000,0x10000000000,0x20000000000,0x40000000000,0x80000000000,0x100000000000,0x200000000000,0x400000000000,0x800000000000,0x1000000000000,0x2000000000000,0x4000000000000,0x8000000000000,0x10000000000000,0x20000000000000,0x40000000000000,0x80000000000000,0x100000000000000,0x200000000000000,0x400000000000000,0x800000000000000,0x1000000000000000,0x2000000000000000,0x4000000000000000,0x8000000000000000 };

	FLOAT_MAT_TYPE res = 0;

	// Get tile indices
	unsigned int tile_row, tile_column;
	tile_row = threadIdx.y;
	tile_column = threadIdx.x;

	// Get A indices
	unsigned int read_a_row, read_a_column;
	read_a_row = blockIdx.y * blockDim.y + threadIdx.y;

	// Get B indices
	unsigned int read_b_column = blockIdx.x * blockDim.x + threadIdx.x;

	for (unsigned int i = 0; i < aWidth / blockDim.x; i++) {

		//Load A in tile
		read_a_column = blockDim.x * i + threadIdx.x;
		if ( (tile_row * (blockDim.y + 1) + tile_column) % unsigned int(BITS_IN_BIN) == 0)
			tileA[ (tile_row * (blockDim.y + 1) + tile_column) / unsigned int(BITS_IN_BIN)] = aData[read_a_row * aWidth / unsigned int(BITS_IN_BIN) + read_a_column];

		//Load B in tile
		unsigned int read_b_row = blockDim.y * i + threadIdx.y;
		tileB[tile_row * (blockDim.x + 1) + tile_column] = bData[read_b_row * bWidth + read_b_column];

		//Calculate multiplication
		__syncthreads();

		for (unsigned int j = 0; j < blockDim.x / unsigned int (BITS_IN_BIN); j++) {
			#pragma unroll
			for (unsigned int shift = 0; shift < unsigned int(BITS_IN_BIN); shift++) {
				if (tileA[threadIdx.y * (blockDim.y + 1) / unsigned int(BITS_IN_BIN) + j] & masks[shift])
					res += tileB[(j * unsigned int(BITS_IN_BIN) + shift)  * (blockDim.x + 1) + threadIdx.x];
			}
		}
		__syncthreads();
	}

	cData[read_a_row * bWidth + read_b_column] = res;
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
	unsigned int sharedSize = (BLOCK_THREADS_MATRIX + 1)*BLOCK_THREADS_MATRIX* sizeof(FLOAT_MAT_TYPE) + (((BLOCK_THREADS_MATRIX + 1)*BLOCK_THREADS_MATRIX*sizeof(BINARY_TYPE)) / unsigned int (BITS_IN_BIN));
	exec.start();
	//simpleMul<<<gridSize, blockSize>>>(A_dev, B_dev, C_dev, aWidth, aHeight, bWidth);
	optimizedMul<<<gridSize, blockSize, sharedSize >>>(A_dev, B_dev, C_dev, aWidth, aHeight, bWidth);
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