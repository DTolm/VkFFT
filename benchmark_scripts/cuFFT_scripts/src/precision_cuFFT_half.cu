//general parts
#include <stdio.h>
#include <vector>
#include <memory>
#include <string.h>
#include <chrono>
#include <thread>
#include <iostream>
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>

//CUDA parts
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>
#include <cuda_fp16.h>
#include <cufftXt.h>

void launch_precision_cuFFT_half(void* inputC, void* output_cuFFT, uint64_t* dims)
{
	cufftHandle planHalf;
	half2* dataC;
	cudaMalloc((void**)&dataC, sizeof(half2) * dims[0] * dims[1] * dims[2]);
	cudaMemcpy(dataC, inputC, sizeof(half2) * dims[0] * dims[1] * dims[2], cudaMemcpyHostToDevice);
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return;
	}
	uint64_t sizeCUDA;
	cufftResult res = cufftCreate(&planHalf);
	size_t ws = 0;
	long long local_dims[3];
	switch (dims[3]) {
	case 1:
		local_dims[0] = (long long)dims[0];
		local_dims[1] = (long long)dims[1];
		local_dims[2] = (long long)dims[2];
		break;
	case 2:
		local_dims[0] = (long long)dims[1];
		local_dims[1] = (long long)dims[0];
		local_dims[2] = (long long)dims[2];
		break;
	case 3:
		local_dims[0] = (long long)dims[2];
		local_dims[1] = (long long)dims[1];
		local_dims[2] = (long long)dims[0];
		break;
	}
	res = cufftXtMakePlanMany(
		planHalf, dims[3], local_dims, NULL, 1, 1, CUDA_C_16F,
		NULL, 1, 1, CUDA_C_16F, 1, &ws, CUDA_C_16F);

	for (int i = 0; i < 1; i++) {
		res = cufftXtExec(planHalf, dataC, dataC, -1);
	}
	cudaDeviceSynchronize();
	cudaMemcpy(output_cuFFT, dataC, sizeof(half2) * dims[0] * dims[1] * dims[2], cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cufftDestroy(planHalf);
	cudaFree(dataC);
}
