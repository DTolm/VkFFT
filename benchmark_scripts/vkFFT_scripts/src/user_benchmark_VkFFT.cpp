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
#endif
#include "vkFFT.h"
#include "utils_VkFFT.h"
#include "half.hpp"

VkFFTResult user_benchmark_VkFFT(VkGPU* vkGPU, uint64_t file_output, FILE* output, uint64_t isCompilerInitialized, VkFFTUserSystemParameters* userParams)
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
#endif
	const int num_runs = 3;
	double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	//memory allocated on the CPU once, makes benchmark completion faster + avoids performance issues connected to frequent allocation/deallocation.
	uint64_t storageComplexSize=8;
	switch (userParams->P) {
	case 0:
		storageComplexSize = (2 * sizeof(float));
		break;
	case 1:
		storageComplexSize = (2 * sizeof(double));
		break;
	case 2:
		storageComplexSize = (2 * 2);
		break;
	default:
		storageComplexSize = (2 * sizeof(float));
		break;
	}
	for (uint64_t n = 0; n < 2; n++) {
		double run_time[num_runs];
		for (uint64_t r = 0; r < num_runs; r++) {
			//Configuration + FFT application .
			VkFFTConfiguration configuration = {};
			VkFFTApplication app = {};
			//FFT + iFFT sample code.
			//Setting up FFT configuration for forward and inverse FFT.
			configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).
			//if (n == 1) configuration.keepShaderCode = 1;

			configuration.size[0] = userParams->X;
			configuration.size[1] = userParams->Y;
			configuration.size[2] = userParams->Z;
			if (userParams->Y > 1) configuration.FFTdim++;
			if (userParams->Z > 1) configuration.FFTdim++;
			configuration.numberBatches = userParams->B;
			configuration.performR2C = userParams->R2C;
			configuration.performDCT = userParams->DCT;
			if (userParams->P == 1) configuration.doublePrecision = 1;
			if (userParams->P == 2) configuration.halfPrecision = 1;
			if (userParams->saveApplicationToString && (n==0) && (r==0)) configuration.saveApplicationToString = 1;
			if (userParams->loadApplicationFromString || (userParams->saveApplicationToString && ((n != 0) || (r != 0)))) configuration.loadApplicationFromString = 1;
			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			configuration.device = &vkGPU->device;
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
#endif
			//Allocate buffer for the input data.
			uint64_t bufferSize = 0;
			if (userParams->R2C) {
				bufferSize = (uint64_t)(storageComplexSize / 2) * (configuration.size[0] + 2) * configuration.size[1] * configuration.size[2] * configuration.numberBatches;
			}
			else {
				if (userParams->DCT) {
					bufferSize = (uint64_t)(storageComplexSize / 2) * configuration.size[0] * configuration.size[1] * configuration.size[2] * configuration.numberBatches;
				}
				else {
					bufferSize = (uint64_t)storageComplexSize * configuration.size[0] * configuration.size[1] * configuration.size[2] * configuration.numberBatches;
				}
			}
#if(VKFFT_BACKEND==0)
			VkBuffer buffer = {};
			VkDeviceMemory bufferDeviceMemory = {};
			resFFT = allocateBuffer(vkGPU, &buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
			if (resFFT != VKFFT_SUCCESS) return resFFT;
			configuration.buffer = &buffer;
#elif(VKFFT_BACKEND==1)
			cuFloatComplex* buffer = 0;
			res = cudaMalloc((void**)&buffer, bufferSize);
			if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
			configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==2)
			hipFloatComplex* buffer = 0;
			res = hipMalloc((void**)&buffer, bufferSize);
			if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
			configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==3)
			cl_mem buffer = 0;
			buffer = clCreateBuffer(vkGPU->context, CL_MEM_READ_WRITE, bufferSize, 0, &res);
			if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
			configuration.buffer = &buffer;
#elif(VKFFT_BACKEND==4)
			void* buffer = 0;
			ze_device_mem_alloc_desc_t device_desc = {};
			device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
			res = zeMemAllocDevice(vkGPU->context, &device_desc, bufferSize, sizeof(float), vkGPU->device, &buffer);
			if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
			configuration.buffer = &buffer;
#endif

			configuration.bufferSize = &bufferSize;
			if (configuration.loadApplicationFromString) {
				FILE* kernelCache;
				uint64_t str_len;
				char fname[500];
				int VkFFT_version = VkFFTGetVersion();
				sprintf(fname, "VkFFT_binary_X%" PRIu64 "_Y%" PRIu64 "_Z%" PRIu64 "_P%" PRIu64 "_B%" PRIu64 "_N%" PRIu64 "_R2C%" PRIu64 "_DCT%" PRIu64 "_ver%d", userParams->X, userParams->Y, userParams->Z, userParams->P, userParams->B, userParams->N, userParams->R2C, userParams->DCT, VkFFT_version);
#if((VKFFT_BACKEND==0) || (VKFFT_BACKEND==2) || (VKFFT_BACKEND==4))
				kernelCache = fopen(fname, "rb"); //Vulkan and HIP backends load data as a uint32_t sequence
#else
				kernelCache = fopen(fname, "r");
#endif
				if (!kernelCache) return VKFFT_ERROR_EMPTY_FILE;
				fseek(kernelCache, 0, SEEK_END);
				str_len = ftell(kernelCache);
				fseek(kernelCache, 0, SEEK_SET);
				configuration.loadApplicationString = malloc(str_len);
				fread(configuration.loadApplicationString, str_len, 1, kernelCache);
				fclose(kernelCache);
			}
			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			resFFT = initializeVkFFT(&app, configuration);
			if (resFFT != VKFFT_SUCCESS) return resFFT;

			if (configuration.loadApplicationFromString)
				free(configuration.loadApplicationString);

			if (configuration.saveApplicationToString) {
				FILE* kernelCache;
				char fname[500];
				int VkFFT_version = VkFFTGetVersion();
				sprintf(fname, "VkFFT_binary_X%" PRIu64 "_Y%" PRIu64 "_Z%" PRIu64 "_P%" PRIu64 "_B%" PRIu64 "_N%" PRIu64 "_R2C%" PRIu64 "_DCT%" PRIu64 "_ver%d", userParams->X, userParams->Y, userParams->Z, userParams->P, userParams->B, userParams->N, userParams->R2C, userParams->DCT, VkFFT_version);
#if((VKFFT_BACKEND==0) || (VKFFT_BACKEND==2) || (VKFFT_BACKEND==4))
				kernelCache = fopen(fname, "wb"); //Vulkan and HIP backends save data as a uint32_t sequence
#else
				kernelCache = fopen(fname, "w");
#endif
				fwrite(app.saveApplicationString, app.applicationStringSize, 1, kernelCache);
				fclose(kernelCache);
			}

			//Submit FFT+iFFT.
			
			double totTime = 0;

			VkFFTLaunchParams launchParams = {};
			resFFT = performVulkanFFTiFFT(vkGPU, &app, &launchParams, userParams->N, &totTime);
			if (resFFT != VKFFT_SUCCESS) return resFFT;
			run_time[r] = totTime;
			if (n > 0) {
				if (r == num_runs - 1) {
					double std_error = 0;
					double avg_time = 0;
					for (uint64_t t = 0; t < num_runs; t++) {
						avg_time += run_time[t];
					}
					avg_time /= num_runs;
					for (uint64_t t = 0; t < num_runs; t++) {
						std_error += (run_time[t] - avg_time) * (run_time[t] - avg_time);
					}
					std_error = sqrt(std_error / num_runs);
					uint64_t num_tot_transfers = 0;
					for (uint64_t i = 0; i < configuration.FFTdim; i++)
						num_tot_transfers += app.localFFTPlan->numAxisUploads[i];
					num_tot_transfers *= 4;
					if (file_output)
						fprintf(output, "VkFFT System: %" PRIu64 "x%" PRIu64 "x%" PRIu64 " Batch: %" PRIu64 " Buffer: %" PRIu64 " MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %" PRIu64 " benchmark: %" PRIu64 " scaled bandwidth: %0.1f bandwidth: %0.1f\n", configuration.size[0], configuration.size[1], configuration.size[2], userParams->B, bufferSize / 1024 / 1024, avg_time, std_error, userParams->N, (uint64_t)(((double)bufferSize / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * 4 * configuration.FFTdim / avg_time, bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);

					printf("VkFFT System: %" PRIu64 "x%" PRIu64 "x%" PRIu64 " Batch: %" PRIu64 " Buffer: %" PRIu64 " MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %" PRIu64 " benchmark: %" PRIu64 " scaled bandwidth: %0.1f real bandwidth: %0.1f\n", configuration.size[0], configuration.size[1], configuration.size[2], userParams->B, bufferSize / 1024 / 1024, avg_time, std_error, userParams->N, (uint64_t)(((double)bufferSize / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * 4 * configuration.FFTdim / avg_time, bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);
					benchmark_result += ((double)bufferSize / 1024) / avg_time;
				}


			}

#if(VKFFT_BACKEND==0)
			vkDestroyBuffer(vkGPU->device, buffer, NULL);
			vkFreeMemory(vkGPU->device, bufferDeviceMemory, NULL);
#elif(VKFFT_BACKEND==1)
			cudaFree(buffer);
#elif(VKFFT_BACKEND==2)
			hipFree(buffer);
#elif(VKFFT_BACKEND==3)
			clReleaseMemObject(buffer);
#elif(VKFFT_BACKEND==4)
			zeMemFree(vkGPU->context, buffer);
#endif

			deleteVkFFT(&app);

		}
	}
	return resFFT;
}
