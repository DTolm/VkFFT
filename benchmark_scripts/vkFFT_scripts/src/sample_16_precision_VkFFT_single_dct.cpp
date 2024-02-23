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
#include "glslang/Include/glslang_c_interface.h"
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
#include "precision_cuFFT_single.h"
#endif	
#ifdef USE_rocFFT
#include "precision_rocFFT_single.h"
#endif	
VkFFTResult sample_16_precision_VkFFT_single_dct(VkGPU* vkGPU, uint64_t file_output, FILE* output, uint64_t isCompilerInitialized)
{
	VkFFTResult resFFT = VKFFT_SUCCESS;
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
		fprintf(output, "16 - VkFFT/FFTW R2R DCT-I, II, III and IV precision test in single precision\n");
	printf("16 - VkFFT/FFTW R2R DCT-I, II, III and IV precision test in single precision\n");

	const int num_benchmark_samples = 318;
	const int num_runs = 1;

	uint64_t benchmark_dimensions[num_benchmark_samples][5] = { {2, 1, 1, 1, 1},{3, 1, 1, 1, 1},{5, 1, 1, 1, 1},{6, 1, 1, 1, 1},{7, 1, 1, 1, 1},{8, 1, 1, 1, 1},{9, 1, 1, 1, 1},{10, 1, 1, 1, 1},{11, 1, 1, 1, 1},{12, 1, 1, 1, 1},{13, 1, 1, 1, 1},{14, 1, 1, 1, 1},
		{15, 1, 1, 1, 1},{16, 1, 1, 1, 1},{17, 1, 1, 1, 1},{21, 1, 1, 1, 1},{22, 1, 1, 1, 1},{23, 1, 1, 1, 1},{24, 1, 1, 1, 1},{25, 1, 1, 1, 1},{26, 1, 1, 1, 1},{27, 1, 1, 1, 1},{28, 1, 1, 1, 1},{29, 1, 1, 1, 1},{30, 1, 1, 1, 1},{31, 1, 1, 1, 1},{32, 1, 1, 1, 1},{33, 1, 1, 1, 1},{34, 1, 1, 1, 1},{35, 1, 1, 1, 1},{37, 1, 1, 1, 1},{38, 1, 1, 1, 1},{39, 1, 1, 1, 1},{41, 1, 1, 1, 1},{42, 1, 1, 1, 1},{44, 1, 1, 1, 1},{45, 1, 1, 1, 1},{46, 1, 1, 1, 1},{49, 1, 1, 1, 1},{52, 1, 1, 1, 1},{55, 1, 1, 1, 1},{56, 1, 1, 1, 1},{58, 1, 1, 1, 1},{60, 1, 1, 1, 1},{64, 1, 1, 1, 1},{65, 1, 1, 1, 1},{66, 1, 1, 1, 1},{81, 1, 1, 1, 1},
		{121, 1, 1, 1, 1},{125, 1, 1, 1, 1},{128, 1, 1, 1, 1},{143, 1, 1, 1, 1},{146, 1, 1, 1, 1}, {169, 1, 1, 1, 1},{243, 1, 1, 1, 1},{256, 1, 1, 1, 1},{283, 1, 1, 1, 1},{286, 1, 1, 1, 1},{343, 1, 1, 1, 1},{429, 1, 1, 1, 1},{512, 1, 1, 1, 1},{572, 1, 1, 1, 1},{625, 1, 1, 1, 1},{720, 1, 1, 1, 1},{1024, 1, 1, 1, 1},{1080, 1, 1, 1, 1},{1001, 1, 1, 1, 1},{1229, 1, 1, 1, 1},{1287, 1, 1, 1, 1},{1400, 1, 1, 1, 1},{1440, 1, 1, 1, 1},{1676, 1, 1, 1, 1},{1920, 1, 1, 1, 1},{2048, 1, 1, 1, 1},{2160, 1, 1, 1, 1},{3024,1,1, 1, 1},{3500,1,1, 1, 1},
		{3840, 1, 1, 1, 1},{4000 , 1, 1, 1, 1},{4050, 1, 1, 1, 1},{4096 , 1, 1, 1, 1} ,{4391, 1, 1, 1, 1},{7000,1,1, 1, 1},{7680, 1, 1, 1, 1},{7879, 1, 1, 1, 1},{9000, 1, 1, 1, 1},{11587, 1, 1, 1, 1},{7680 * 5, 1, 1, 1, 1},
		{15319, 1, 1, 1, 1},{21269, 1, 1, 1, 1},{27283, 1, 1, 1, 1},{39829, 1, 1, 1, 1},{52733, 1, 1, 1, 1},{2000083, 1, 1, 1, 1},{4000067, 1, 1, 1, 1},{8003869, 1, 1, 1, 1},
		{(uint64_t)pow(3,10), 1, 1, 1, 1},{(uint64_t)pow(3,11), 1, 1, 1, 1},{(uint64_t)pow(3,12), 1, 1, 1, 1},{(uint64_t)pow(3,13), 1, 1, 1, 1},{(uint64_t)pow(3,14), 1, 1, 1, 1},{(uint64_t)pow(3,15), 1, 1, 1, 1},
		{(uint64_t)pow(5,5), 1, 1, 1, 1},{(uint64_t)pow(5,6), 1, 1, 1, 1},{(uint64_t)pow(5,7), 1, 1, 1, 1},{(uint64_t)pow(5,8), 1, 1, 1, 1},{(uint64_t)pow(5,9), 1, 1, 1, 1},
		{(uint64_t)pow(7,4), 1, 1, 1, 1},{(uint64_t)pow(7,5), 1, 1, 1, 1},{(uint64_t)pow(7,6), 1, 1, 1, 1},{(uint64_t)pow(7,7), 1, 1, 1, 1},{(uint64_t)pow(7,8), 1, 1, 1, 1},
		{(uint64_t)pow(11,3), 1, 1, 1, 1},{(uint64_t)pow(11,4), 1, 1, 1, 1},{(uint64_t)pow(11,5), 1, 1, 1, 1},{(uint64_t)pow(11,6), 1, 1, 1, 1},
		{(uint64_t)pow(13,3), 1, 1, 1, 1},{(uint64_t)pow(13,4), 1, 1, 1, 1},{(uint64_t)pow(13,5), 1, 1, 1, 1},{(uint64_t)pow(13,6), 1, 1, 1, 1},
		{8, 3, 1, 1, 2},{8, 5, 1, 1, 2},{8, 6, 1, 1, 2},{8, 7, 1, 1, 2},{8, 8, 1, 1, 2},{8, 9, 1, 1, 2},{8, 10, 1, 1, 2},{8, 11, 1, 1, 2},{8, 12, 1, 1, 2},{8, 13, 1, 1, 2},{8, 14, 1, 1, 2},{8, 15, 1, 1, 2},{8, 16, 1, 1, 2},{8, 17, 1, 1, 2},{8, 21, 1, 1, 2},{8, 22, 1, 1, 2},{8, 23, 1, 1, 2},{8, 24, 1, 1, 2},
		{8, 25, 1, 1, 2},{8, 26, 1, 1, 2},{8, 27, 1, 1, 2},{8, 28, 1, 1, 2},{8, 29, 1, 1, 2},{8, 30, 1, 1, 2},{8, 31, 1, 1, 2},{8, 32, 1, 1, 2},{8, 33, 1, 1, 2},{8, 34, 1, 1, 2},{8, 35, 1, 1, 2},{8, 37, 1, 1, 2},{8, 38, 1, 1, 2},{8, 39, 1, 1, 2},{8, 41, 1, 1, 2},{8, 44, 1, 1, 2},{8, 45, 1, 1, 2},{8, 46, 1, 1, 2},{8, 49, 1, 1, 2},{8, 52, 1, 1, 2},{8, 56, 1, 1, 2},{8, 58, 1, 1, 2},{8, 60, 1, 1, 2},{8, 64, 1, 1, 2},{8, 66, 1, 1, 2},{8, 81, 1, 1, 2},{8, 125, 1, 1, 2},{8, 128, 1, 1, 2},{8, 243, 1, 1, 2},{8, 256, 1, 1, 2},{8, 343, 1, 1, 2},{8, 358, 1, 1, 2},{8, 429, 1, 1, 2},{8, 512, 1, 1, 2},{8, 1024, 1, 1, 2},
	    {720, 480, 1, 1, 2},{1280, 720, 1, 1, 2}, {8, 4320, 1, 1, 2},{8, 4391, 1, 1, 2},{8, 7000, 1, 1, 2},{8, 7680, 1, 1, 2},{8, 4050 * 3, 1, 1, 2},{8, 7680 * 5, 1, 1, 2}, {720, 480, 1, 1, 2},{1280, 720, 1, 1, 2},{1920, 1080, 1, 1, 2}, {2560, 1440, 1, 1, 2},{3840, 2160, 1, 1, 2},{7680, 4320, 1, 1, 2},
		{8,15319, 1, 1, 2},{8,21269, 1, 1, 2},{8,27283, 1, 1, 2},{8,39829, 1, 1, 2},{8,52733, 1, 1, 2},{8,2000083, 1, 1, 2},{8,4000067, 1, 1, 2},{8,8003869, 1, 1, 2},
		{8, (uint64_t)pow(3,10), 1, 1, 2},	{8, (uint64_t)pow(3,11), 1, 1, 2}, {8, (uint64_t)pow(3,12), 1, 1, 2}, {8, (uint64_t)pow(3,13), 1, 1, 2}, {8, (uint64_t)pow(3,14), 1, 1, 2}, {8, (uint64_t)pow(3,15), 1, 1, 2},
		{8, (uint64_t)pow(5,5), 1, 1, 2},	{8, (uint64_t)pow(5,6), 1, 1, 2}, {8, (uint64_t)pow(5,7), 1, 1, 2}, {8, (uint64_t)pow(5,8), 1, 1, 2}, {8, (uint64_t)pow(5,9), 1, 1, 2},
		{8, (uint64_t)pow(7,4), 1, 1, 2},{8, (uint64_t)pow(7,5), 1, 1, 2},{8, (uint64_t)pow(7,6), 1, 1, 2},{8, (uint64_t)pow(7,7), 1, 1, 2},{8, (uint64_t)pow(7,8), 1, 1, 2},
		{8, (uint64_t)pow(11,3), 1, 1, 2},{8, (uint64_t)pow(11,4), 1, 1, 2},{8, (uint64_t)pow(11,5), 1, 1, 2},{8, (uint64_t)pow(11,6), 1, 1, 2},
		{8, (uint64_t)pow(13,3), 1, 1, 2},{8, (uint64_t)pow(13,4), 1, 1, 2},{8, (uint64_t)pow(13,5), 1, 1, 2},{8, (uint64_t)pow(13,6), 1, 1, 2},
		{2, 2, 2, 1, 3},{3, 3, 3, 1, 3},{5, 5, 5, 1, 3},{6, 6, 6, 1, 3},{7, 7, 7, 1, 3},{8, 8, 8, 1, 3},{9, 9, 9, 1, 3},{10, 10, 10, 1, 3},{11, 11, 11, 1, 3},{12, 12, 12, 1, 3},{13, 13, 13, 1, 3},{14, 14, 14, 1, 3},
		{15, 15, 15, 1, 3},{16, 16, 16, 1, 3},{17, 17, 17, 1, 3},{21, 21, 21, 1, 3},{22, 22, 22, 1, 3},{23, 23, 23, 1, 3},{24, 24, 24, 1, 3},{25, 25, 25, 1, 3},{26, 26, 26, 1, 3},{27, 27, 27, 1, 3},{28, 28, 28, 1, 3},{29, 29, 29, 1, 3},{30, 30, 30, 1, 3},{31, 31, 31, 1, 3},{32, 32, 32, 1, 3},{33, 33, 33, 1, 3},{34, 34, 34, 1, 3},{35, 35, 35, 1, 3},{39, 39, 39, 1, 3},{42, 42, 42, 1, 3},{44, 44, 44, 1, 3},{45, 45, 45, 1, 3},{46, 46, 46, 1, 3},{49, 49, 49, 1, 3},{52, 52, 52, 1, 3},{56, 56, 56, 1, 3},{60, 60, 60, 1, 3},{64, 64, 64, 1, 3},{81, 81, 81, 1, 3},
		{121, 121, 121, 1, 3},{128, 128, 128, 1, 3},{125, 125, 125, 1, 3},{143, 143, 143, 1, 3},{169, 169, 169, 1, 3},{243, 243, 243, 1, 3},{256, 256, 256, 1, 3},
		{2, 2, 2, 2, 4},{3, 3, 3, 3, 4},{5, 5, 5, 5, 4},{6, 6, 6, 6, 4},{7, 7, 7, 7, 4},{9, 9, 9, 9, 4},{10, 10, 10, 10, 4},{11, 11, 11, 11, 4},{12, 12, 12, 12, 4},{13, 13, 13, 13, 4},{14, 14, 14, 14, 4},
		{15, 15, 15, 15, 4},{17, 17, 17, 17, 4},{21, 21, 21, 21, 4},{22, 22, 22, 22, 4},{23, 23, 23, 23, 4},{24, 24, 24, 24, 4},{25, 25, 25, 25, 4},{26, 26, 26, 26, 4},{27, 27, 27, 27, 4},{28, 28, 28, 28, 4},{29, 29, 29, 29, 4},{30, 30, 30, 30, 4},{31, 31, 31, 31, 4},{33, 33, 33, 33, 4},{35, 35, 35, 35, 4},{37, 37, 37, 37, 4},{39, 39, 39, 39, 4},{41, 41, 41, 41, 4},{42, 42, 42, 42, 4},{43, 43, 43, 43, 4},{44, 44, 44, 44, 4},{45, 45, 45, 45, 4},{47, 47, 47, 47, 4},{49, 49, 49, 49, 4},{52, 52, 52, 52, 4},{53, 53, 53, 53, 4},{56, 56, 56, 56, 4},{59, 59, 59, 59, 4},{60, 60, 60, 60, 4},{61, 61, 61, 61, 4},{81, 81, 81, 81, 4},
		{3, 5, 7, 9, 4},{5, 3, 7, 9, 4},{9, 7, 5, 3, 4},{23, 25, 27, 29, 4},{25, 23, 27, 29, 4},{29, 27, 25, 23, 4},{123, 25, 127, 129, 4},{125, 123, 27, 129, 4},{129, 127, 125, 23, 4}
	};

	double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples

	for (int n = 0; n < num_benchmark_samples; n++) {
		for (int r = 0; r < num_runs; r++) {
			for (int t = 1; t < 5; t++) {
				float* inputC;
				double* inputC_double;
				uint64_t dims[4] = { benchmark_dimensions[n][0] , benchmark_dimensions[n][1] , benchmark_dimensions[n][2] , benchmark_dimensions[n][3]};

				inputC = (float*)(malloc(sizeof(float) * dims[0] * dims[1] * dims[2] * dims[3]));
				if (!inputC) return VKFFT_ERROR_MALLOC_FAILED;
				inputC_double = (double*)(malloc(sizeof(double) * dims[0] * dims[1] * dims[2] * dims[3]));
				if (!inputC_double) return VKFFT_ERROR_MALLOC_FAILED;

				for (uint64_t k = 0; k < dims[3]; k++) {
					for (uint64_t l = 0; l < dims[2]; l++) {
						for (uint64_t j = 0; j < dims[1]; j++) {
							for (uint64_t i = 0; i < dims[0]; i++) {
								inputC[i + j * dims[0] + l * dims[0] * dims[1] + k * dims[0] * dims[1] * dims[2]] = (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
								inputC_double[i + j * dims[0] + l * dims[0] * dims[1] + k * dims[0] * dims[1] * dims[2]] = (double)inputC[i + j * dims[0] + l * dims[0] * dims[1] + k * dims[0] * dims[1] * dims[2]];
							}
						}
					}
				}

				fftw_plan p;

				double* output_FFTW = (double*)(malloc(sizeof(double) * dims[0] * dims[1] * dims[2] * dims[3]));

				enum fftw_r2r_kind_do_not_use_me dct_type;
				switch (t)
				{
				case 1:
					dct_type = FFTW_REDFT00;
					break;
				case 2:
					dct_type = FFTW_REDFT10;
					break;
				case 3:
					dct_type = FFTW_REDFT01;
					break;
				case 4:
					dct_type = FFTW_REDFT11;
					break;
				}

				if (!output_FFTW) return VKFFT_ERROR_MALLOC_FAILED;
				switch (benchmark_dimensions[n][4]) {
				case 1:
					p = fftw_plan_r2r_1d((int)benchmark_dimensions[n][0], inputC_double, output_FFTW, dct_type, FFTW_ESTIMATE);
					break;
				case 2:
					p = fftw_plan_r2r_2d((int)benchmark_dimensions[n][1], (int)benchmark_dimensions[n][0], inputC_double, output_FFTW, dct_type, dct_type, FFTW_ESTIMATE);
					break;
				case 3:
					p = fftw_plan_r2r_3d((int)benchmark_dimensions[n][2], (int)benchmark_dimensions[n][1], (int)benchmark_dimensions[n][0], inputC_double, output_FFTW, dct_type, dct_type, dct_type, FFTW_ESTIMATE);
					break;
				case 4:
					fftw_iodim fftw_iodims[4];
					fftw_iodims[0].n = (int)benchmark_dimensions[n][3];
					fftw_iodims[0].is = (int)(benchmark_dimensions[n][2] * benchmark_dimensions[n][1] * benchmark_dimensions[n][0]);
					fftw_iodims[0].os = (int)(benchmark_dimensions[n][2] * benchmark_dimensions[n][1] * benchmark_dimensions[n][0]);
					fftw_iodims[1].n = (int)benchmark_dimensions[n][2];
					fftw_iodims[1].is = (int)(benchmark_dimensions[n][1] * benchmark_dimensions[n][0]);
					fftw_iodims[1].os = (int)(benchmark_dimensions[n][1] * benchmark_dimensions[n][0]);
					fftw_iodims[2].n = (int)benchmark_dimensions[n][1];
					fftw_iodims[2].is = (int)(benchmark_dimensions[n][0]);
					fftw_iodims[2].os = (int)(benchmark_dimensions[n][0]);
					fftw_iodims[3].n = (int)benchmark_dimensions[n][0];
					fftw_iodims[3].is = 1;
					fftw_iodims[3].os = 1;
					fftw_iodim howmany_dims[1];
					howmany_dims[0].n = 1;
					howmany_dims[0].is = 1;
					howmany_dims[0].os = 1;
					fftw_r2r_kind_do_not_use_me dct_type_nd[4];
					for (int p = 0; p < 4; p++){
						switch (t)
						{
						case 1:
							dct_type_nd[p] = FFTW_REDFT00;
							break;
						case 2:
							dct_type_nd[p] = FFTW_REDFT10;
							break;
						case 3:
							dct_type_nd[p] = FFTW_REDFT01;
							break;
						case 4:
							dct_type_nd[p] = FFTW_REDFT11;
							break;
						}
					}
					p = fftw_plan_guru_r2r(4, fftw_iodims, 1, howmany_dims, inputC_double, output_FFTW, dct_type_nd, FFTW_ESTIMATE);
					break;
				}

				fftw_execute(p);

				float totTime = 0;
				int num_iter = 1;

				//VkFFT part

				VkFFTConfiguration configuration = {};
				VkFFTApplication app = {};
				configuration.FFTdim = benchmark_dimensions[n][4]; //FFT dimension, 1D, 2D or 3D (default 1).
				configuration.size[0] = benchmark_dimensions[n][0]; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
				configuration.size[1] = benchmark_dimensions[n][1];
				configuration.size[2] = benchmark_dimensions[n][2];
				configuration.size[3] = benchmark_dimensions[n][3];
				configuration.performDCT = t;// DCT type. Currently supported 1, 2, 3 and 4
				
				//configuration.keepShaderCode = 1;
				//configuration.useLUT = 1;
				//configuration.disableMergeSequencesR2C = 1;
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
					bufferSize[i] = (uint64_t)sizeof(float) * configuration.size[0] * configuration.size[1] * configuration.size[2] * configuration.size[3]/ numBuf;
				}
#if(VKFFT_BACKEND==0)
				VkBuffer* buffer = (VkBuffer*)malloc(numBuf * sizeof(VkBuffer));
				if (!buffer) return VKFFT_ERROR_MALLOC_FAILED;
				VkDeviceMemory* bufferDeviceMemory = (VkDeviceMemory*)malloc(numBuf * sizeof(VkDeviceMemory));
				if (!bufferDeviceMemory) return VKFFT_ERROR_MALLOC_FAILED;
#elif(VKFFT_BACKEND==1)
				cuFloatComplex* buffer = 0;
#elif(VKFFT_BACKEND==2)
				hipFloatComplex* buffer = 0;
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
					res = zeMemAllocDevice(vkGPU->context, &device_desc, bufferSize[i], sizeof(float), vkGPU->device, &buffer);
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
					resFFT = transferDataFromCPU(vkGPU, (inputC + shift / sizeof(float)), &buffer[i], bufferSize[i]);
					if (resFFT != VKFFT_SUCCESS) return resFFT;
#else
                    resFFT = transferDataFromCPU(vkGPU, (inputC + shift / sizeof(float)), &buffer, bufferSize[i]);
                    if (resFFT != VKFFT_SUCCESS) return resFFT;
#endif
					shift += bufferSize[i];
				}
				//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
				resFFT = initializeVkFFT(&app, configuration);
				if (resFFT == VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH_R2R) {
					if (file_output)
						fprintf(output, "VkFFT DCT-%d System: %" PRIu64 "x%" PRIu64 "x%" PRIu64 "x%" PRIu64 " - UNSUPPORTED\n", t, dims[0], dims[1], dims[2], dims[3]);
					printf("VkFFT DCT-%d System: %" PRIu64 "x%" PRIu64 "x%" PRIu64 "x%" PRIu64 " - UNSUPPORTED\n", t, dims[0], dims[1], dims[2], dims[3]);
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
					fftw_destroy_plan(p);
					free(inputC_double);
					free(output_FFTW);
					continue;
				}
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
				float* output_VkFFT = (float*)(malloc(sizeof(float) * dims[0] * dims[1] * dims[2] * dims[3]));
				if (!output_VkFFT) return VKFFT_ERROR_MALLOC_FAILED;
				//Transfer data from GPU using staging buffer.
				shift = 0;
				for (uint64_t i = 0; i < numBuf; i++) {
#if(VKFFT_BACKEND==0)
					resFFT = transferDataToCPU(vkGPU, (output_VkFFT + shift / sizeof(float)), &buffer[i], bufferSize[i]);
					if (resFFT != VKFFT_SUCCESS) return resFFT;
#else
                    resFFT = transferDataToCPU(vkGPU, (output_VkFFT + shift / sizeof(float)), &buffer, bufferSize[i]);
                    if (resFFT != VKFFT_SUCCESS) return resFFT;
#endif
					shift += bufferSize[i];
				}
				double avg_difference[2] = { 0,0 };
				double max_difference[2] = { 0,0 };
				double avg_eps[2] = { 0,0 };
				double max_eps[2] = { 0,0 };
				for (uint64_t k = 0; k < dims[3]; k++) {
					for (uint64_t l = 0; l < dims[2]; l++) {
						for (uint64_t j = 0; j < dims[1]; j++) {
							for (uint64_t i = 0; i < dims[0]; i++) {
								//if (file_output) fprintf(output, "%.2e %.2e - %.2e %.2e \n", output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] / N, output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1] / N, output_VkFFT[(i + j * dims[0] + l * dims[0] * dims[1])][0], output_VkFFT[(i + j * dims[0] + l * dims[0] * dims[1])][1]);

								//printf("%.2e - %.2e \n", output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]], output_VkFFT[(i + j * dims[0] + l * dims[0] * dims[1])]);
								double current_data_norm = sqrt(output_FFTW[i + j * dims[0] + l * dims[0] * dims[1] + k * dims[0] * dims[1] * dims[2]] * output_FFTW[i + j * dims[0] + l * dims[0] * dims[1] + k * dims[0] * dims[1] * dims[2]]);

								double current_diff_norm_VkFFT = sqrt((output_VkFFT[i + j * dims[0] + l * dims[0] * dims[1] + k * dims[0] * dims[1] * dims[2]] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1] + k * dims[0] * dims[1] * dims[2]]) * (output_VkFFT[i + j * dims[0] + l * dims[0] * dims[1] + k * dims[0] * dims[1] * dims[2]] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1] + k * dims[0] * dims[1] * dims[2]]));
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
				avg_difference[0] /= (dims[0] * dims[1] * dims[2] * dims[3]);
				avg_eps[0] /= (dims[0] * dims[1] * dims[2] * dims[3]);
				avg_difference[1] /= (dims[0] * dims[1] * dims[2] * dims[3]);
				avg_eps[1] /= (dims[0] * dims[1] * dims[2] * dims[3]);
				if (file_output)
					fprintf(output, "VkFFT DCT-%d System: %" PRIu64 "x%" PRIu64 "x%" PRIu64 "x%" PRIu64 " avg_difference: %.2e max_difference: %.2e avg_eps: %.2e max_eps: %.2e\n", t, dims[0], dims[1], dims[2], dims[3], avg_difference[1], max_difference[1], avg_eps[1], max_eps[1]);
				printf("VkFFT DCT-%d System: %" PRIu64 "x%" PRIu64 "x%" PRIu64 "x%" PRIu64 " avg_difference: %.2e max_difference: %.2e avg_eps: %.2e max_eps: %.2e\n", t, dims[0], dims[1], dims[2], dims[3], avg_difference[1], max_difference[1], avg_eps[1], max_eps[1]);
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
				fftw_destroy_plan(p);
				free(inputC_double);
				free(output_FFTW);
			}
		}
	}
	return resFFT;
}
