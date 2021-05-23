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

void launch_precision_cuFFT_r2c(void* inputC, void* output_cuFFT, uint64_t* dims)
{
	cufftHandle planR2C;
	cufftReal* dataR;
	cufftComplex* dataC;
	cudaMalloc((void**)&dataR, sizeof(cufftComplex) * (dims[0]/2+1) * dims[1] * dims[2]);
	cudaMalloc((void**)&dataC, sizeof(cufftComplex) * (dims[0] / 2 + 1) * dims[1] * dims[2]);
	cudaMemcpy(dataR, inputC, sizeof(cufftReal) * dims[0] * dims[1] * dims[2], cudaMemcpyHostToDevice);
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return;
	}
	switch (dims[3]) {
	case 1:
		cufftPlan1d(&planR2C, dims[0], CUFFT_R2C, 1);
		break;
	case 2:
		cufftPlan2d(&planR2C, dims[1], dims[0], CUFFT_R2C);
		break;
	case 3:
		cufftPlan3d(&planR2C, dims[2], dims[1], dims[0], CUFFT_R2C);
		break;
	}
	for (int i = 0; i < 1; i++) {
		cufftExecR2C(planR2C, dataR, dataC);
	}
	cudaDeviceSynchronize();
	cufftDestroy(planR2C);
	switch (dims[3]) {
	case 1:
		cufftPlan1d(&planR2C, dims[0], CUFFT_C2R, 1);
		break;
	case 2:
		cufftPlan2d(&planR2C, dims[1], dims[0], CUFFT_C2R);
		break;
	case 3:
		cufftPlan3d(&planR2C, dims[2], dims[1], dims[0], CUFFT_C2R);
		break;
	}
	for (int i = 0; i < 1; i++) {
		cufftExecC2R(planR2C, dataC, dataR);
	}
	cudaDeviceSynchronize();
	cudaMemcpy(output_cuFFT, dataR, sizeof(float) * (dims[0] ) * dims[1] * dims[2], cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cufftDestroy(planR2C);
	cudaFree(dataR);
	cudaFree(dataC);
}
