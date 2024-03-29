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

#if(VKFFT_BACKEND==0)
#include "vulkan/vulkan.h"
#include "glslang_c_interface.h"
#elif(VKFFT_BACKEND==1)
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#elif(VKFFT_BACKEND==2)
#ifndef __HIP_PLATFORM_HCC__
#define __HIP_PLATFORM_HCC__
#endif
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_complex.h>
#elif(VKFFT_BACKEND==3)
#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif 
#elif(VKFFT_BACKEND==4)
#include <ze_api.h>
#elif(VKFFT_BACKEND==5)
#include "Foundation/Foundation.hpp"
#include "QuartzCore/QuartzCore.hpp"
#include "Metal/Metal.hpp"
#endif
#include "vkFFT.h"
#include "utils_VkFFT.h"
#include "fftw3.h"
#ifdef USE_cuFFT
#include "precision_cuFFT_double.h"
#endif	
#ifdef USE_rocFFT
#include "precision_rocFFT_double.h"
#endif	
#include <iostream>
#ifdef VKFFT_USE_DOUBLEDOUBLE_FP128
#include <quadmath.h>
#endif
VkFFTResult sample_19_precision_VkFFT_quadDoubleDouble_nonPow2(VkGPU* vkGPU, uint64_t file_output, FILE* output, uint64_t isCompilerInitialized)
{
	VkFFTResult resFFT = VKFFT_SUCCESS;
#ifdef VKFFT_USE_DOUBLEDOUBLE_FP128
#if(VKFFT_BACKEND==0)
	VkResult res = VK_SUCCESS;
#elif(VKFFT_BACKEND==1)
	cudaError_t res = cudaSuccess;
#elif(VKFFT_BACKEND==2)
	hipError_t res = hipSuccess;
#elif(VKFFT_BACKEND==3)
	cl_int res = CL_SUCCESS;
#elif(VKFFT_BACKEND==4)
	ze_result_t res = ZE_RESULT_SUCCESS;
#elif(VKFFT_BACKEND==5)
#endif
	if (file_output)
		fprintf(output, "19 - VkFFT/FFTW C2C precision test in double-double emulation of quad precision\n");
	printf("19 - VkFFT/FFTW C2C precision test in double-double emulation of quad precision\n");

	const int num_benchmark_samples = 349;
	const int num_runs = 1;

	uint64_t benchmark_dimensions[num_benchmark_samples][5] = {  {(uint64_t)pow(2,5), 1, 1, 1, 1},{(uint64_t)pow(2,6), 1, 1, 1, 1},{(uint64_t)pow(2,7), 1, 1, 1, 1},{(uint64_t)pow(2,8), 1, 1, 1, 1},{(uint64_t)pow(2,9), 1, 1, 1, 1},{(uint64_t)pow(2,10), 1, 1, 1, 1},
		{(uint64_t)pow(2,11), 1, 1, 1, 1},{(uint64_t)pow(2,12), 1, 1, 1, 1},{(uint64_t)pow(2,13), 1, 1, 1, 1},{(uint64_t)pow(2,14), 1, 1, 1, 1},{(uint64_t)pow(2,15), 1, 1, 1, 1},{(uint64_t)pow(2,16), 1, 1, 1, 1},{(uint64_t)pow(2,17), 1, 1, 1, 1},{(uint64_t)pow(2,18), 1, 1, 1, 1},
		{(uint64_t)pow(2,19), 1, 1, 1, 1},{(uint64_t)pow(2,20), 1, 1, 1, 1},{(uint64_t)pow(2,21), 1, 1, 1, 1},{(uint64_t)pow(2,22), 1, 1, 1, 1},{(uint64_t)pow(2,23), 1, 1, 1, 1}, {3, 1, 1, 1, 1}, {5, 1, 1, 1, 1},{6, 1, 1, 1, 1},{7, 1, 1, 1, 1},{9, 1, 1, 1, 1},{10, 1, 1, 1, 1},{11, 1, 1, 1, 1},{12, 1, 1, 1, 1},{13, 1, 1, 1, 1},{14, 1, 1, 1, 1},
		{15, 1, 1, 1, 1},{17, 1, 1, 1, 1},{19, 1, 1, 1, 1},{21, 1, 1, 1, 1},{22, 1, 1, 1, 1},{23, 1, 1, 1, 1},{24, 1, 1, 1, 1},{25, 1, 1, 1, 1},{26, 1, 1, 1, 1},{27, 1, 1, 1, 1},{28, 1, 1, 1, 1},{29, 1, 1, 1, 1},{30, 1, 1, 1, 1},{31, 1, 1, 1, 1},{33, 1, 1, 1, 1},{35, 1, 1, 1, 1},{37, 1, 1, 1, 1},{39, 1, 1, 1, 1},{41, 1, 1, 1, 1},{43, 1, 1, 1, 1},{42, 1, 1, 1, 1},{44, 1, 1, 1, 1},{45, 1, 1, 1, 1},{47, 1, 1, 1, 1},{49, 1, 1, 1, 1},{52, 1, 1, 1, 1},{53, 1, 1, 1, 1},{55, 1, 1, 1, 1},{56, 1, 1, 1, 1},{59, 1, 1, 1, 1},{60, 1, 1, 1, 1},{61, 1, 1, 1, 1},{65, 1, 1, 1, 1},{66, 1, 1, 1, 1},{67, 1, 1, 1, 1},{71, 1, 1, 1, 1},{73, 1, 1, 1, 1},{79, 1, 1, 1, 1},{81, 1, 1, 1, 1},{83, 1, 1, 1, 1},{89, 1, 1, 1, 1},{97, 1, 1, 1, 1},
		{121, 1, 1, 1, 1},{125, 1, 1, 1, 1},{137, 1, 1, 1, 1},{143, 1, 1, 1, 1},{169, 1, 1, 1, 1},{191, 1, 1, 1, 1},{243, 1, 1, 1, 1},{286, 1, 1, 1, 1},{343, 1, 1, 1, 1},{383, 1, 1, 1, 1},{429, 1, 1, 1, 1},{509, 1, 1, 1, 1},{572, 1, 1, 1, 1},{625, 1, 1, 1, 1},{720, 1, 1, 1, 1},{1080, 1, 1, 1, 1},{1001, 1, 1, 1, 1},{1213, 1, 1, 1, 1},{1287, 1, 1, 1, 1},{1400, 1, 1, 1, 1},{1440, 1, 1, 1, 1},{1920, 1, 1, 1, 1},{2160, 1, 1, 1, 1},{2731, 1, 1, 1, 1},{3024,1,1, 1, 1},{3500,1,1, 1, 1},
		{3840, 1, 1, 1, 1},{4000 , 1, 1, 1, 1},{4050, 1, 1, 1, 1},{4320 , 1, 1, 1, 1},{4391, 1, 1, 1, 1},{7000,1,1, 1, 1},{7680, 1, 1, 1, 1},{7879, 1, 1, 1, 1},{9000, 1, 1, 1, 1},{11587, 1, 1, 1, 1},{7680 * 5, 1, 1, 1, 1},
		{15319, 1, 1, 1, 1},{21269, 1, 1, 1, 1},{27283, 1, 1, 1, 1},{39829, 1, 1, 1, 1},{52733, 1, 1, 1, 1},{2000083, 1, 1, 1, 1},{4000067, 1, 1, 1, 1},{8003869, 1, 1, 1, 1},
		{(uint64_t)pow(3,10), 1, 1, 1, 1},{(uint64_t)pow(3,11), 1, 1, 1, 1},{(uint64_t)pow(3,12), 1, 1, 1, 1},{(uint64_t)pow(3,13), 1, 1, 1, 1},{(uint64_t)pow(3,14), 1, 1, 1, 1},{(uint64_t)pow(3,15), 1, 1, 1, 1},
		{(uint64_t)pow(5,5), 1, 1, 1, 1},{(uint64_t)pow(5,6), 1, 1, 1, 1},{(uint64_t)pow(5,7), 1, 1, 1, 1},{(uint64_t)pow(5,8), 1, 1, 1, 1},{(uint64_t)pow(5,9), 1, 1, 1, 1},
		{(uint64_t)pow(7,4), 1, 1, 1, 1},{(uint64_t)pow(7,5), 1, 1, 1, 1},{(uint64_t)pow(7,6), 1, 1, 1, 1},{(uint64_t)pow(7,7), 1, 1, 1, 1},{(uint64_t)pow(7,8), 1, 1, 1, 1},
		{(uint64_t)pow(11,3), 1, 1, 1, 1},{(uint64_t)pow(11,4), 1, 1, 1, 1},{(uint64_t)pow(11,5), 1, 1, 1, 1},{(uint64_t)pow(11,6), 1, 1, 1, 1},
		{(uint64_t)pow(13,3), 1, 1, 1, 1},{(uint64_t)pow(13,4), 1, 1, 1, 1},{(uint64_t)pow(13,5), 1, 1, 1, 1},{(uint64_t)pow(13,6), 1, 1, 1, 1},
		{8, 3, 1, 1, 2},{8, 5, 1, 1, 2},{8, 6, 1, 1, 2},{8, 7, 1, 1, 2},{8, 9, 1, 1, 2},{8, 10, 1, 1, 2},{8, 11, 1, 1, 2},{8, 12, 1, 1, 2},{8, 13, 1, 1, 2},{8, 14, 1, 1, 2},{8, 15, 1, 1, 2},{8, 17, 1, 1, 2},{8, 19, 1, 1, 2},{8, 21, 1, 1, 2},{8, 22, 1, 1, 2},{8, 23, 1, 1, 2},{8, 24, 1, 1, 2},
		{8, 25, 1, 1, 2},{8, 26, 1, 1, 2},{8, 27, 1, 1, 2},{8, 28, 1, 1, 2},{8, 29, 1, 1, 2},{8, 30, 1, 1, 2},{8, 31, 1, 1, 2},{8, 33, 1, 1, 2},{8, 35, 1, 1, 2},{8, 37, 1, 1, 2},{8, 39, 1, 1, 2},{8, 41, 1, 1, 2},{8, 43, 1, 1, 2},{8, 44, 1, 1, 2},{8, 45, 1, 1, 2},{8, 47, 1, 1, 2},{8, 49, 1, 1, 2},{8, 52, 1, 1, 2},{8, 53, 1, 1, 2},{8, 56, 1, 1, 2},{8, 59, 1, 1, 2},{8, 60, 1, 1, 2},{8, 61, 1, 1, 2},{8, 66, 1, 1, 2},{8, 67, 1, 1, 2},{8, 71, 1, 1, 2},{8, 73, 1, 1, 2},{8, 79, 1, 1, 2},{8, 81, 1, 1, 2},{8, 83, 1, 1, 2},{8, 89, 1, 1, 2},{8, 97, 1, 1, 2},{8, 125, 1, 1, 2},{8, 243, 1, 1, 2},{8, 343, 1, 1, 2},
		{8, 625, 1, 1, 2},{8, 720, 1, 1, 2},{8, 1080, 1, 1, 2},{8, 1287, 1, 1, 2},{8, 1400, 1, 1, 2},{8, 1440, 1, 1, 2},{8, 1920, 1, 1, 2},{8, 2160, 1, 1, 2},{8, 2731, 1, 1, 2},{8, 3024, 1, 1, 2},{8, 3500, 1, 1, 2},
		{8, 3840, 1, 1, 2},{8, 4000, 1, 1, 2},{8, 4050, 1, 1, 2},{8, 4320, 1, 1, 2},{8, 4391, 1, 1, 2},{8, 7000, 1, 1, 2},{8, 7680, 1, 1, 2},{8, 4050 * 3, 1, 1, 2},{8, 7680 * 5, 1, 1, 2}, {720, 480, 1, 1, 2},{1280, 720, 1, 1, 2},{1920, 1080, 1, 1, 2}, {2560, 1440, 1, 1, 2},{3840, 2160, 1, 1, 2},{7680, 4320, 1, 1, 2},
		{8,15319, 1, 1, 2},{8,21269, 1, 1, 2},{8,27283, 1, 1, 2},{8,39829, 1, 1, 2},{8,52733, 1, 1, 2},{8,2000083, 1, 1, 2},{8,4000067, 1, 1, 2},{8,8003869, 1, 1, 2},
		{8, (uint64_t)pow(3,10), 1, 1, 2},	{8, (uint64_t)pow(3,11), 1, 1, 2}, {8, (uint64_t)pow(3,12), 1, 1, 2}, {8, (uint64_t)pow(3,13), 1, 1, 2}, {8, (uint64_t)pow(3,14), 1, 1, 2}, {8, (uint64_t)pow(3,15), 1, 1, 2},
		{8, (uint64_t)pow(5,5), 1, 1, 2},	{8, (uint64_t)pow(5,6), 1, 1, 2}, {8, (uint64_t)pow(5,7), 1, 1, 2}, {8, (uint64_t)pow(5,8), 1, 1, 2}, {8, (uint64_t)pow(5,9), 1, 1, 2},
		{8, (uint64_t)pow(7,4), 1, 1, 2},{8, (uint64_t)pow(7,5), 1, 1, 2},{8, (uint64_t)pow(7,6), 1, 1, 2},{8, (uint64_t)pow(7,7), 1, 1, 2},{8, (uint64_t)pow(7,8), 1, 1, 2},
		{8, (uint64_t)pow(11,3), 1, 1, 2},{8, (uint64_t)pow(11,4), 1, 1, 2},{8, (uint64_t)pow(11,5), 1, 1, 2},{8, (uint64_t)pow(11,6), 1, 1, 2},
		{8, (uint64_t)pow(13,3), 1, 1, 2},{8, (uint64_t)pow(13,4), 1, 1, 2},{8, (uint64_t)pow(13,5), 1, 1, 2},{8, (uint64_t)pow(13,6), 1, 1, 2},
		{3, 3, 3, 1, 3},{5, 5, 5, 1, 3},{6, 6, 6, 1, 3},{7, 7, 7, 1, 3},{9, 9, 9, 1, 3},{10, 10, 10, 1, 3},{11, 11, 11, 1, 3},{12, 12, 12, 1, 3},{13, 13, 13, 1, 3},{14, 14, 14, 1, 3},
		{15, 15, 15, 1, 3},{17, 17, 17, 1, 3},{21, 21, 21, 1, 3},{22, 22, 22, 1, 3},{23, 23, 23, 1, 3},{24, 24, 24, 1, 3},{25, 25, 25, 1, 3},{26, 26, 26, 1, 3},{27, 27, 27, 1, 3},{28, 28, 28, 1, 3},{29, 29, 29, 1, 3},{30, 30, 30, 1, 3},{31, 31, 31, 1, 3},{33, 33, 33, 1, 3},{35, 35, 35, 1, 3},{37, 37, 37, 1, 3},{39, 39, 39, 1, 3},{41, 41, 41, 1, 3},{42, 42, 42, 1, 3},{43, 43, 43, 1, 3},{44, 44, 44, 1, 3},{45, 45, 45, 1, 3},{47, 47, 47, 1, 3},{49, 49, 49, 1, 3},{52, 52, 52, 1, 3},{53, 53, 53, 1, 3},{56, 56, 56, 1, 3},{59, 59, 59, 1, 3},{60, 60, 60, 1, 3},{61, 61, 61, 1, 3},{81, 81, 81, 1, 3},
		{121, 121, 121, 1, 3},{125, 125, 125, 1, 3},{143, 143, 143, 1, 3},{169, 169, 169, 1, 3},{243, 243, 243, 1, 3},
		{3, 3, 3, 3, 4},{5, 5, 5, 5, 4},{6, 6, 6, 6, 4},{7, 7, 7, 7, 4},{9, 9, 9, 9, 4},{10, 10, 10, 10, 4},{11, 11, 11, 11, 4},{12, 12, 12, 12, 4},{13, 13, 13, 13, 4},{14, 14, 14, 14, 4},
		{15, 15, 15, 15, 4},{17, 17, 17, 17, 4},{21, 21, 21, 21, 4},{22, 22, 22, 22, 4},{23, 23, 23, 23, 4},{24, 24, 24, 24, 4},{25, 25, 25, 25, 4},{26, 26, 26, 26, 4},{27, 27, 27, 27, 4},{28, 28, 28, 28, 4},{29, 29, 29, 29, 4},{30, 30, 30, 30, 4},{31, 31, 31, 31, 4},{33, 33, 33, 33, 4},{35, 35, 35, 35, 4},{37, 37, 37, 37, 4},{39, 39, 39, 39, 4},{41, 41, 41, 41, 4},{42, 42, 42, 42, 4},{43, 43, 43, 43, 4},{44, 44, 44, 44, 4},{45, 45, 45, 45, 4},{47, 47, 47, 47, 4},{49, 49, 49, 49, 4},{52, 52, 52, 52, 4},{53, 53, 53, 53, 4},{56, 56, 56, 56, 4},{59, 59, 59, 59, 4},{60, 60, 60, 60, 4},{61, 61, 61, 61, 4},{81, 81, 81, 81, 4},
		{3, 5, 7, 9, 4},{5, 3, 7, 9, 4},{9, 7, 5, 3, 4},{23, 25, 27, 29, 4},{25, 23, 27, 29, 4},{29, 27, 25, 23, 4},{123, 25, 127, 129, 4},{125, 123, 27, 129, 4},{129, 127, 125, 23, 4},
		{20000, 2, 3, 3, 4},{3, 20000, 2, 3, 4},{3, 2, 3, 20000, 4}
	};

	double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	for (int n = 0; n < num_benchmark_samples; n++) {
		for (int r = 0; r < 1; r++) {

			double* inputC;
			fftwq_complex* inputC_quad;
			uint64_t dims[4] = { benchmark_dimensions[n][0] , benchmark_dimensions[n][1] , benchmark_dimensions[n][2] , benchmark_dimensions[n][3]};

			inputC = (double*)(calloc(4 * dims[0] * dims[1] * dims[2] * dims[3], sizeof(double)));
			if (!inputC) return VKFFT_ERROR_MALLOC_FAILED;
			inputC_quad = (fftwq_complex*)(malloc(sizeof(fftwq_complex) * dims[0] * dims[1] * dims[2] * dims[3]));
			if (!inputC_quad) return VKFFT_ERROR_MALLOC_FAILED;
			for (uint64_t k = 0; k < dims[3]; k++) {
				for (uint64_t l = 0; l < dims[2]; l++) {
					for (uint64_t j = 0; j < dims[1]; j++) {
						for (uint64_t i = 0; i < dims[0]; i++) {
							inputC[4*(i + j * dims[0] + l * dims[0] * dims[1] + k * dims[0] * dims[1] * dims[2])] =  (double)(2 * ((double)rand()) / RAND_MAX - 1.0);
							inputC[4*(i + j * dims[0] + l * dims[0] * dims[1] + k * dims[0] * dims[1] * dims[2])+1] = 1e-17 * (double)(2 * ((double)rand()) / RAND_MAX - 1.0);
							inputC[4*(i + j * dims[0] + l * dims[0] * dims[1] + k * dims[0] * dims[1] * dims[2])+2] = (double)(2 * ((double)rand()) / RAND_MAX - 1.0);
							inputC[4*(i + j * dims[0] + l * dims[0] * dims[1] + k * dims[0] * dims[1] * dims[2])+3] = 1e-17 * (double)(2 * ((double)rand()) / RAND_MAX - 1.0);
							inputC_quad[i + j * dims[0] + l * dims[0] * dims[1] + k * dims[0] * dims[1] * dims[2]][0] = (__float128)inputC[4*(i + j * dims[0] + l * dims[0] * dims[1] + k * dims[0] * dims[1] * dims[2])]+(__float128)inputC[4*(i + j * dims[0] + l * dims[0] * dims[1] + k * dims[0] * dims[1] * dims[2])+1];
							inputC_quad[i + j * dims[0] + l * dims[0] * dims[1] + k * dims[0] * dims[1] * dims[2]][1] = (__float128)inputC[4*(i + j * dims[0] + l * dims[0] * dims[1] + k * dims[0] * dims[1] * dims[2])+2]+(__float128)inputC[4*(i + j * dims[0] + l * dims[0] * dims[1] + k * dims[0] * dims[1] * dims[2])+3];
						}
					}
				}
			}
			fftwq_plan p;

			fftwq_complex* output_FFTW = (fftwq_complex*)(malloc(sizeof(fftwq_complex) * dims[0] * dims[1] * dims[2]* dims[3]));
			if (!output_FFTW) return VKFFT_ERROR_MALLOC_FAILED;
			switch (benchmark_dimensions[n][4]) {
			case 1:
				p = fftwq_plan_dft_1d((int)benchmark_dimensions[n][0], inputC_quad, output_FFTW, -1, FFTW_ESTIMATE);
				break;
			case 2:
				p = fftwq_plan_dft_2d((int)benchmark_dimensions[n][1], (int)benchmark_dimensions[n][0], inputC_quad, output_FFTW, -1, FFTW_ESTIMATE);
				break;
			case 3:
				p = fftwq_plan_dft_3d((int)benchmark_dimensions[n][2], (int)benchmark_dimensions[n][1], (int)benchmark_dimensions[n][0], inputC_quad, output_FFTW, -1, FFTW_ESTIMATE);
				break;
			case 4:
				fftwq_iodim fftw_iodims[4];
				fftw_iodims[0].n = (int)benchmark_dimensions[n][3];
				fftw_iodims[0].is = (int)(benchmark_dimensions[n][2]*benchmark_dimensions[n][1]*benchmark_dimensions[n][0]);
				fftw_iodims[0].os = (int)(benchmark_dimensions[n][2]*benchmark_dimensions[n][1]*benchmark_dimensions[n][0]);
				fftw_iodims[1].n = (int)benchmark_dimensions[n][2];
				fftw_iodims[1].is = (int)(benchmark_dimensions[n][1]*benchmark_dimensions[n][0]);
				fftw_iodims[1].os = (int)(benchmark_dimensions[n][1]*benchmark_dimensions[n][0]);
				fftw_iodims[2].n = (int)benchmark_dimensions[n][1];
				fftw_iodims[2].is = (int)(benchmark_dimensions[n][0]);
				fftw_iodims[2].os = (int)(benchmark_dimensions[n][0]);
				fftw_iodims[3].n = (int)benchmark_dimensions[n][0];
				fftw_iodims[3].is = 1;
				fftw_iodims[3].os = 1;
				fftwq_iodim howmany_dims[1];
				howmany_dims[0].n = 1;
				howmany_dims[0].is = 1;
				howmany_dims[0].os = 1;

				p = fftwq_plan_guru_dft(4, fftw_iodims, 1, howmany_dims, inputC_quad, output_FFTW, -1, FFTW_ESTIMATE);
				break;
			}

			fftwq_execute(p);

			double totTime = 0;
			int num_iter = 1;

			//VkFFT part

			VkFFTConfiguration configuration = {};
			VkFFTApplication app = {};
			configuration.FFTdim = benchmark_dimensions[n][4]; //FFT dimension, 1D, 2D or 3D (default 1).
			configuration.size[0] = benchmark_dimensions[n][0]; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			configuration.size[1] = benchmark_dimensions[n][1];
			configuration.size[2] = benchmark_dimensions[n][2];
			configuration.size[3] = benchmark_dimensions[n][3];
            configuration.quadDoubleDoublePrecision = 1;
			//configuration.keepShaderCode = 1;
			//configuration.printMemoryLayout = 1;
			//configuration.disableReorderFourStep = 1;
			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
#if(VKFFT_BACKEND==5)
			configuration.device = vkGPU->device;
#else
			configuration.device = &vkGPU->device;
#endif
#if(VKFFT_BACKEND==0)
			configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
			configuration.fence = &vkGPU->fence;
			configuration.commandPool = &vkGPU->commandPool;
			configuration.physicalDevice = &vkGPU->physicalDevice;
			configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization
#elif(VKFFT_BACKEND==3)
			configuration.context = &vkGPU->context;
#elif(VKFFT_BACKEND==4)
			configuration.context = &vkGPU->context;
			configuration.commandQueue = &vkGPU->commandQueue;
			configuration.commandQueueID = vkGPU->commandQueueID;
#elif(VKFFT_BACKEND==5)
            configuration.queue = vkGPU->queue;
#endif			

			uint64_t numBuf = 1;

			//Allocate buffers for the input data. - we use 4 in this example
			uint64_t* bufferSize = (uint64_t*)malloc(sizeof(uint64_t) * numBuf);
			if (!bufferSize) return VKFFT_ERROR_MALLOC_FAILED;
			for (uint64_t i = 0; i < numBuf; i++) {
				bufferSize[i] = {};
				bufferSize[i] = (uint64_t)sizeof(double) * 2 * 2 * benchmark_dimensions[n][0] * benchmark_dimensions[n][1] * benchmark_dimensions[n][2] * benchmark_dimensions[n][3]/ numBuf;
			}
#if(VKFFT_BACKEND==0)
			VkBuffer* buffer = (VkBuffer*)malloc(numBuf * sizeof(VkBuffer));
			if (!buffer) return VKFFT_ERROR_MALLOC_FAILED;
			VkDeviceMemory* bufferDeviceMemory = (VkDeviceMemory*)malloc(numBuf * sizeof(VkDeviceMemory));
			if (!bufferDeviceMemory) return VKFFT_ERROR_MALLOC_FAILED;
#elif(VKFFT_BACKEND==1)
			cuDoubleComplex* buffer = 0;
#elif(VKFFT_BACKEND==2)
			hipDoubleComplex* buffer = 0;
#elif(VKFFT_BACKEND==3)
			cl_mem buffer = 0;
#elif(VKFFT_BACKEND==4)
			void* buffer = 0;
#elif(VKFFT_BACKEND==5)
            MTL::Buffer* buffer = 0;
#endif			
			for (uint64_t i = 0; i < numBuf; i++) {
#if(VKFFT_BACKEND==0)
				buffer[i] = {};
				bufferDeviceMemory[i] = {};
				resFFT = allocateBuffer(vkGPU, &buffer[i], &bufferDeviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize[i]);
				if (resFFT != VKFFT_SUCCESS) return resFFT;
#elif(VKFFT_BACKEND==1)
				res = cudaMalloc((void**)&buffer, bufferSize[i]);
				if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
#elif(VKFFT_BACKEND==2)
				res = hipMalloc((void**)&buffer, bufferSize[i]);
				if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
#elif(VKFFT_BACKEND==3)
				buffer = clCreateBuffer(vkGPU->context, CL_MEM_READ_WRITE, bufferSize[i], 0, &res);
				if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
#elif(VKFFT_BACKEND==4)
				ze_device_mem_alloc_desc_t device_desc = {};
				device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
				res = zeMemAllocDevice(vkGPU->context, &device_desc, bufferSize[i], sizeof(double), vkGPU->device, &buffer);
				if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
#elif(VKFFT_BACKEND==5)
                buffer = vkGPU->device->newBuffer(bufferSize[i], MTL::ResourceStorageModePrivate);
#endif
			}

			configuration.bufferNum = numBuf;
			/*
#if(VKFFT_BACKEND==0)
			configuration.buffer = buffer;
#elif(VKFFT_BACKEND==1)
			configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==2)
			configuration.buffer = (void**)&buffer;
#endif
			*/ // Can specify buffers at launch
			configuration.bufferSize = bufferSize;

			//Sample buffer transfer tool. Uses staging buffer (if needed) of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
			uint64_t shift = 0;
			for (uint64_t i = 0; i < numBuf; i++) {
#if(VKFFT_BACKEND==0)
				resFFT = transferDataFromCPU(vkGPU, (inputC + shift / sizeof(double)), &buffer[i], bufferSize[i]);
#else
                resFFT = transferDataFromCPU(vkGPU, (inputC + shift / sizeof(double)), &buffer, bufferSize[i]);
#endif
				if (resFFT != VKFFT_SUCCESS) return resFFT;
				shift += bufferSize[i];
			}
			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			resFFT = initializeVkFFT(&app, configuration);
			if (resFFT != VKFFT_SUCCESS) return resFFT;
			//Submit FFT+iFFT.
			//num_iter = 1;
			//specify buffers at launch
			VkFFTLaunchParams launchParams = {};
#if(VKFFT_BACKEND==0)
			launchParams.buffer = buffer;
#elif(VKFFT_BACKEND==1)
			launchParams.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==2)
			launchParams.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==3)
			launchParams.buffer = &buffer;
#elif(VKFFT_BACKEND==4)
			launchParams.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==5)
            launchParams.buffer = &buffer;
#endif
			resFFT = performVulkanFFT(vkGPU, &app, &launchParams, -1, num_iter);
			if (resFFT != VKFFT_SUCCESS) return resFFT;
			double* output_VkFFT = (double*)(malloc(4*sizeof(double) * benchmark_dimensions[n][0] * benchmark_dimensions[n][1] * benchmark_dimensions[n][2] * benchmark_dimensions[n][3]));
			if (!output_VkFFT) return VKFFT_ERROR_MALLOC_FAILED;
			//Transfer data from GPU using staging buffer.
			shift = 0;
			for (uint64_t i = 0; i < numBuf; i++) {
#if(VKFFT_BACKEND==0)
				resFFT = transferDataToCPU(vkGPU, (output_VkFFT + shift / sizeof(double)), &buffer[i],4* sizeof(double) * benchmark_dimensions[n][0] * benchmark_dimensions[n][1] * benchmark_dimensions[n][2] * benchmark_dimensions[n][3]);
#else
                resFFT = transferDataToCPU(vkGPU, (output_VkFFT + shift / sizeof(double)), &buffer, 4*sizeof(double) * benchmark_dimensions[n][0] * benchmark_dimensions[n][1] * benchmark_dimensions[n][2] * benchmark_dimensions[n][3]);
#endif
				if (resFFT != VKFFT_SUCCESS) return resFFT;
				shift += bufferSize[i];
			}
			__float128 avg_difference[2] = { 0,0 };
			__float128 max_difference[2] = { 0,0 };
			__float128 avg_eps[2] = { 0,0 };
			__float128 max_eps[2] = { 0,0 };
			for (uint64_t k = 0; k < dims[3]; k++) {
				for (uint64_t l = 0; l < dims[2]; l++) {
					for (uint64_t j = 0; j < dims[1]; j++) {
						for (uint64_t i = 0; i < dims[0]; i++) {
							uint64_t loc_i = i;
							uint64_t loc_j = j;
							uint64_t loc_l = l;

							//if (file_output) fprintf(output, "%.2e %.2e - %.2e %.2e \n", output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] / N, output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1] / N, output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][0], output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][1]);
							//if (i > dims[0] - 10)
							/*char buf[128];
							__float128 r;
							int n = quadmath_snprintf (buf, sizeof buf, "%+-#*.20Qe", 46, output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0]);
							printf("%s ", buf);
							n = quadmath_snprintf (buf, sizeof buf, "%+-#*.20Qe", 46, output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1]);
							printf("%s - ", buf);
							r = (__float128)output_VkFFT[4*(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]+ k * dims[0] * dims[1]* dims[2])] + (__float128)output_VkFFT[4*(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]+ k * dims[0] * dims[1]* dims[2])+1];
							n = quadmath_snprintf (buf, sizeof buf, "%+-#*.20Qe", 46, r);
							printf("%s", buf);
							r = (__float128)output_VkFFT[4*(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]+ k * dims[0] * dims[1]* dims[2])+2] + (__float128)output_VkFFT[4*(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]+ k * dims[0] * dims[1]* dims[2])+3];
							n = quadmath_snprintf (buf, sizeof buf, "%+-#*.20Qe", 46, r);
							printf("%s\n", buf);*/

							//printf("%.17Le %.17Le - %.17Le %.17Le \n", (long double)output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] , (long double)output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1], (long double)output_VkFFT[4*(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])], (long double)output_VkFFT[4*(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])+2]);
							//printf("%.2e %.2e \n", output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][0], output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][1]);

							__float128 current_data_norm = sqrtq(output_FFTW[i + j * dims[0] + l * dims[0] * dims[1] + k * dims[0] * dims[1]* dims[2]][0] * output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]+ k * dims[0] * dims[1]* dims[2]][0] + output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]+ k * dims[0] * dims[1]* dims[2]][1] * output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]+ k * dims[0] * dims[1]* dims[2]][1]);

							__float128 current_diff_x_VkFFT = ((__float128)output_VkFFT[4*(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]+ k * dims[0] * dims[1]* dims[2])] + (__float128)output_VkFFT[4*(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]+ k * dims[0] * dims[1]* dims[2])+1] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]+ k * dims[0] * dims[1]* dims[2]][0]);
							__float128 current_diff_y_VkFFT = ((__float128)output_VkFFT[4*(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]+ k * dims[0] * dims[1]* dims[2])+2] + (__float128)output_VkFFT[4*(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]+ k * dims[0] * dims[1]* dims[2])+3] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]+ k * dims[0] * dims[1]* dims[2]][1]);
							__float128 current_diff_norm_VkFFT = sqrtq(current_diff_x_VkFFT * current_diff_x_VkFFT + current_diff_y_VkFFT * current_diff_y_VkFFT);
							if (current_diff_norm_VkFFT > max_difference[1]) max_difference[1] = current_diff_norm_VkFFT;
							avg_difference[1] += current_diff_norm_VkFFT;
							if ((current_diff_norm_VkFFT / current_data_norm > max_eps[1])) {
								max_eps[1] = current_diff_norm_VkFFT / current_data_norm;
							}
							avg_eps[1] += current_diff_norm_VkFFT / current_data_norm;
						}
					}
				}
			}
			avg_difference[0] /= (dims[0] * dims[1] * dims[2]* dims[3]);
			avg_eps[0] /= (dims[0] * dims[1] * dims[2]* dims[3]);
			avg_difference[1] /= (dims[0] * dims[1] * dims[2]* dims[3]);
			avg_eps[1] /= (dims[0] * dims[1] * dims[2]* dims[3]);

			if (file_output)
				fprintf(output, "VkFFT System: %" PRIu64 "x%" PRIu64 "x%" PRIu64 "x%" PRIu64 " avg_difference: %.2e max_difference: %.2e avg_eps: %.2e max_eps: %.2e\n", dims[0], dims[1], dims[2],dims[3], (double)avg_difference[1], (double)max_difference[1], (double)avg_eps[1], (double)max_eps[1]);
			printf("VkFFT System: %" PRIu64 "x%" PRIu64 "x%" PRIu64 "x%" PRIu64 " avg_difference: %.2e max_difference: %.2e avg_eps: %.2e max_eps: %.2e\n", dims[0], dims[1], dims[2], dims[3], (double)avg_difference[1], (double)max_difference[1], (double)avg_eps[1], (double)max_eps[1]);
			free(output_VkFFT);
			for (uint64_t i = 0; i < numBuf; i++) {

#if(VKFFT_BACKEND==0)
				vkDestroyBuffer(vkGPU->device, buffer[i], NULL);
				vkFreeMemory(vkGPU->device, bufferDeviceMemory[i], NULL);
#elif(VKFFT_BACKEND==1)
				cudaFree(buffer);
#elif(VKFFT_BACKEND==2)
				hipFree(buffer);
#elif(VKFFT_BACKEND==3)
				clReleaseMemObject(buffer);
#elif(VKFFT_BACKEND==4)
				zeMemFree(vkGPU->context, buffer);
#elif(VKFFT_BACKEND==5)
                buffer->release();
#endif

			}
#if(VKFFT_BACKEND==0)
			free(buffer);
			free(bufferDeviceMemory);
#endif

			free(bufferSize);
			deleteVkFFT(&app);
			free(inputC);
			fftwq_destroy_plan(p);
			free(inputC_quad);
			free(output_FFTW);
		}
	}
#endif
	return resFFT;
}
