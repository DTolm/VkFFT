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

void launch_precision_rocFFT_double(void* inputC, void* output_rocFFT, uint64_t* dims)
{
	hipfftHandle planZ2Z;
	hipfftDoubleComplex* dataC;
	hipMalloc((void**)&dataC, sizeof(hipfftDoubleComplex) * dims[0] * dims[1] * dims[2]);
	hipMemcpy(dataC, inputC, sizeof(hipfftDoubleComplex) * dims[0] * dims[1] * dims[2], hipMemcpyHostToDevice);
	if (hipGetLastError() != hipSuccess) {
		fprintf(stderr, "ROCM error: Failed to allocate\n");
		return;
	}
	switch (dims[3]) {
	case 1:
		hipfftPlan1d(&planZ2Z, dims[0], HIPFFT_Z2Z, 1);
		break;
	case 2:
		hipfftPlan2d(&planZ2Z, dims[1], dims[0], HIPFFT_Z2Z);
		break;
	case 3:
		hipfftPlan3d(&planZ2Z, dims[2], dims[1], dims[0], HIPFFT_Z2Z);
		break;
	}
	for (int i = 0; i < 1; i++) {
		hipfftExecZ2Z(planZ2Z, dataC, dataC, -1);
	}
	hipDeviceSynchronize();
	hipMemcpy(output_rocFFT, dataC, sizeof(hipfftDoubleComplex) * dims[0] * dims[1] * dims[2], hipMemcpyDeviceToHost);
	hipDeviceSynchronize();
	hipfftDestroy(planZ2Z);
	hipFree(dataC);
}
