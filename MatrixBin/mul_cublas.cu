#pragma comment(lib,"cublas.lib")

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#include "cuda_header.cuh"

#include "Timer.h"

FLOAT_MAT_TYPE* cublas_mul_matrix(long aWidth, long aHeight, FLOAT_MAT_TYPE* aData, long bWidth, long bHeight, FLOAT_MAT_TYPE* bData){
	FLOAT_MAT_TYPE* matrix_a_dev;
	FLOAT_MAT_TYPE* matrix_b_dev;
	FLOAT_MAT_TYPE* matrix_out;
	FLOAT_MAT_TYPE* res;

	long width_out, height_out;
	width_out = bWidth;
	height_out = aHeight;

	Timer alloc_dev, exec, alloc_host;

	alloc_dev.start();
	cudaMalloc((void**)&matrix_a_dev, aWidth * aHeight * sizeof(FLOAT_MAT_TYPE));
	cudaMalloc((void**)&matrix_b_dev, bWidth * bHeight * sizeof(FLOAT_MAT_TYPE));
	cudaMalloc((void**)&matrix_out, width_out * height_out * sizeof(FLOAT_MAT_TYPE));

	cudaMemcpy(matrix_a_dev, aData, aWidth * aHeight * sizeof(FLOAT_MAT_TYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(matrix_b_dev, bData, bWidth * bHeight * sizeof(FLOAT_MAT_TYPE), cudaMemcpyHostToDevice);
	alloc_dev.stop("Allocate and copy to device: ");

	const float alpha = 1.0f;
	const float beta = 0.0f;
	cublasHandle_t handle;

	exec.start();
	cublasCreate(&handle);
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, bWidth, aHeight, aWidth, &alpha, matrix_b_dev, bWidth, matrix_a_dev, aWidth, &beta, matrix_out, bWidth);
	cudaDeviceSynchronize();
	exec.stop("Execution: ");

	alloc_host.start();
	// copy result from device to host
	res = new FLOAT_MAT_TYPE[width_out * height_out];
	cudaMemcpy(res, matrix_out, width_out * height_out * sizeof(FLOAT_MAT_TYPE), cudaMemcpyDeviceToHost);

	cublasDestroy(handle);
	alloc_host.stop("Allocate and copy to host: ");

	cudaFree(matrix_a_dev);
	cudaFree(matrix_b_dev);
	cudaFree(matrix_out);

	return res;
}