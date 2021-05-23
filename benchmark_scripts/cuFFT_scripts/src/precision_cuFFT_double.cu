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

void launch_precision_cuFFT_double(void* inputC, void* output_cuFFT, uint64_t* dims)
{
	cufftHandle planZ2Z;
	cufftDoubleComplex* dataC;
	cudaMalloc((void**)&dataC, sizeof(cufftDoubleComplex) * dims[0] * dims[1] * dims[2]);
	cudaMemcpy(dataC, inputC, sizeof(cufftDoubleComplex) * dims[0] * dims[1] * dims[2], cudaMemcpyHostToDevice);
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return;
	}
	switch (dims[3]) {
	case 1:
		cufftPlan1d(&planZ2Z, dims[0], CUFFT_Z2Z, 1);
		break;
	case 2:
		cufftPlan2d(&planZ2Z, dims[1], dims[0], CUFFT_Z2Z);
		break;
	case 3:
		cufftPlan3d(&planZ2Z, dims[2], dims[1], dims[0], CUFFT_Z2Z);
		break;
	}
	for (int i = 0; i < 1; i++) {
		cufftExecZ2Z(planZ2Z, dataC, dataC, -1);
	}
	cudaDeviceSynchronize();
	cudaMemcpy(output_cuFFT, dataC, sizeof(cufftDoubleComplex) * dims[0] * dims[1] * dims[2], cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cufftDestroy(planZ2Z);
	cudaFree(dataC);
}
