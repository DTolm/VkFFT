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

//ROCM parts
#include "hip/hip_runtime.h"
#include <hipfft.h>

void launch_precision_rocFFT_r2c(void* inputC, void* output_rocFFT, uint64_t* dims)
{
	hipfftHandle planR2C;
	hipfftReal* dataR;
	hipfftComplex* dataC;
	hipMalloc((void**)&dataR, sizeof(hipfftComplex) * (dims[0]/2+1) * dims[1] * dims[2]);
	hipMalloc((void**)&dataC, sizeof(hipfftComplex) * (dims[0] / 2 + 1) * dims[1] * dims[2]);
	hipMemcpy(dataR, inputC, sizeof(hipfftReal) * dims[0] * dims[1] * dims[2], hipMemcpyHostToDevice);
	if (hipGetLastError() != hipSuccess) {
		fprintf(stderr, "ROCM error: Failed to allocate\n");
		return;
	}
	switch (dims[3]) {
	case 1:
		hipfftPlan1d(&planR2C, dims[0], HIPFFT_R2C, 1);
		break;
	case 2:
		hipfftPlan2d(&planR2C, dims[1], dims[0], HIPFFT_R2C);
		break;
	case 3:
		hipfftPlan3d(&planR2C, dims[2], dims[1], dims[0], HIPFFT_R2C);
		break;
	}
	for (int i = 0; i < 1; i++) {
		hipfftExecR2C(planR2C, dataR, dataC);
	}
	hipDeviceSynchronize();
	hipfftDestroy(planR2C);
	switch (dims[3]) {
	case 1:
		hipfftPlan1d(&planR2C, dims[0], HIPFFT_C2R, 1);
		break;
	case 2:
		hipfftPlan2d(&planR2C, dims[1], dims[0], HIPFFT_C2R);
		break;
	case 3:
		hipfftPlan3d(&planR2C, dims[2], dims[1], dims[0], HIPFFT_C2R);
		break;
	}
	for (int i = 0; i < 1; i++) {
		hipfftExecC2R(planR2C, dataC, dataR);
	}
	hipDeviceSynchronize();
	hipMemcpy(output_rocFFT, dataR, sizeof(float) * (dims[0] ) * dims[1] * dims[2], hipMemcpyDeviceToHost);
	hipDeviceSynchronize();
	hipfftDestroy(planR2C);
	hipFree(dataR);
	hipFree(dataC);
}
