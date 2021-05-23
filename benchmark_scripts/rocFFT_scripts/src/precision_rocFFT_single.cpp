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

void launch_precision_rocFFT_single(void* inputC, void* output_rocFFT, uint64_t* dims)
{
	hipfftHandle planC2C;
	hipfftComplex* dataC;
	hipMalloc((void**)&dataC, sizeof(hipfftComplex) * dims[0] * dims[1] * dims[2]);
	hipMemcpy(dataC, inputC, sizeof(hipfftComplex) * dims[0] * dims[1] * dims[2], hipMemcpyHostToDevice);
	if (hipGetLastError() != hipSuccess) {
		fprintf(stderr, "ROCM error: Failed to allocate\n");
		return;
	}
	switch (dims[3]) {
	case 1:
		hipfftPlan1d(&planC2C, dims[0], HIPFFT_C2C, 1);
		break;
	case 2:
		hipfftPlan2d(&planC2C, dims[1], dims[0], HIPFFT_C2C);
		break;
	case 3:
		hipfftPlan3d(&planC2C, dims[2], dims[1], dims[0], HIPFFT_C2C);
		break;
	}
	for (int i = 0; i < 1; i++) {
		hipfftExecC2C(planC2C, dataC, dataC, -1);
	}
	hipDeviceSynchronize();
	hipMemcpy(output_rocFFT, dataC, sizeof(hipfftComplex) * dims[0] * dims[1] * dims[2], hipMemcpyDeviceToHost);
	hipDeviceSynchronize();
	hipfftDestroy(planC2C);
	hipFree(dataC);
}
