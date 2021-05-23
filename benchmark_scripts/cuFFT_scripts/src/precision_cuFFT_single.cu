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

void launch_precision_cuFFT_single(void* inputC, void* output_cuFFT, uint64_t* dims)
{
	cufftHandle planC2C;
	cufftComplex* dataC;
	cudaMalloc((void**)&dataC, sizeof(cufftComplex) * dims[0] * dims[1] * dims[2]);
	cudaMemcpy(dataC, inputC, sizeof(cufftComplex) * dims[0] * dims[1] * dims[2], cudaMemcpyHostToDevice);
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return;
	}
	switch (dims[3]) {
	case 1:
		cufftPlan1d(&planC2C, dims[0], CUFFT_C2C, 1);
		break;
	case 2:
		cufftPlan2d(&planC2C, dims[1], dims[0], CUFFT_C2C);
		break;
	case 3:
		cufftPlan3d(&planC2C, dims[2], dims[1], dims[0], CUFFT_C2C);
		break;
	}
	for (int i = 0; i < 1; i++) {
		cufftExecC2C(planC2C, dataC, dataC, -1);
	}
	cudaDeviceSynchronize();
	cudaMemcpy(output_cuFFT, dataC, sizeof(cufftComplex) * dims[0] * dims[1] * dims[2], cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cufftDestroy(planC2C);
	cudaFree(dataC);
}
