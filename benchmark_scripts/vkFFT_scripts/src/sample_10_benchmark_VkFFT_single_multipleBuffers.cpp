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

VkFFTResult sample_10_benchmark_VkFFT_single_multipleBuffers(VkGPU* vkGPU, uint64_t file_output, FILE* output, uint64_t isCompilerInitialized) 
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
#if(VKFFT_BACKEND==0)
	if (file_output)
		fprintf(output, "10 - VkFFT FFT + iFFT C2C benchmark 1D batched in single precision, multiple buffer split FFT\n");
	printf("10 - VkFFT FFT + iFFT C2C benchmark 1D batched in single precision, multiple buffer split FFT\n");
	const int num_runs = 3;
	double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	//memory allocated on the CPU once, makes benchmark completion faster + avoids performance issues connected to frequent allocation/deallocation.
	float* buffer_input = (float*)malloc((uint64_t)4 * 2 * (uint64_t)pow(2, 27));
	if (!buffer_input) return VKFFT_ERROR_MALLOC_FAILED;
	for (uint64_t i = 0; i < 2 * (uint64_t)pow(2, 27); i++) {
		buffer_input[i] = (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
	}
	for (uint64_t n = 0; n < 26; n++) {
		double run_time[num_runs];
		for (uint64_t r = 0; r < num_runs; r++) {
			//Configuration + FFT application .
			VkFFTConfiguration configuration = {};
			VkFFTApplication app = {};
			//FFT + iFFT sample code.
			//Setting up FFT configuration for forward and inverse FFT.
			configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).
			configuration.size[0] = 4 * (uint64_t)pow(2, n); //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			if (n == 0) configuration.size[0] = 4096;
			configuration.numberBatches = (uint64_t)64 * 32 * (uint64_t)pow(2, 16) / configuration.size[0];
			if (configuration.numberBatches < 1) configuration.numberBatches = 1;
			//configuration.numberBatches = (configuration.numberBatches > 32768) ? 32768 : configuration.numberBatches;
			uint64_t numBuf = 4;

			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			configuration.device = &vkGPU->device;
			configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
			configuration.fence = &vkGPU->fence;
			configuration.commandPool = &vkGPU->commandPool;
			configuration.physicalDevice = &vkGPU->physicalDevice;
			configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization

			//Allocate buffers for the input data. - we use 4 in this example
			uint64_t* bufferSize = (uint64_t*)malloc(sizeof(uint64_t) * numBuf);
			if (!bufferSize) return VKFFT_ERROR_MALLOC_FAILED;
			for (uint64_t i = 0; i < numBuf; i++) {
				bufferSize[i] = {};
				bufferSize[i] = (uint64_t)sizeof(float) * 2 * configuration.size[0] * configuration.numberBatches / numBuf;
			}

			VkBuffer* buffer = (VkBuffer*)malloc(numBuf * sizeof(VkBuffer));
			if (!buffer) return VKFFT_ERROR_MALLOC_FAILED;
			VkDeviceMemory* bufferDeviceMemory = (VkDeviceMemory*)malloc(numBuf * sizeof(VkDeviceMemory));
			if (!bufferDeviceMemory) return VKFFT_ERROR_MALLOC_FAILED;
			configuration.userTempBuffer = true; //user allocated temp buffer to reshuffle Four step FFT
			VkBuffer* tempBuffer = (VkBuffer*)malloc(numBuf * sizeof(VkBuffer));
			if (!tempBuffer) return VKFFT_ERROR_MALLOC_FAILED;
			VkDeviceMemory* tempBufferDeviceMemory = (VkDeviceMemory*)malloc(numBuf * sizeof(VkDeviceMemory));
			if (!tempBufferDeviceMemory) return VKFFT_ERROR_MALLOC_FAILED;
			for (uint64_t i = 0; i < numBuf; i++) {
				buffer[i] = {};
				tempBuffer[i] = {};
				bufferDeviceMemory[i] = {};
				tempBufferDeviceMemory[i] = {};
				resFFT = allocateBuffer(vkGPU, &buffer[i], &bufferDeviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize[i]);
				if (resFFT != VKFFT_SUCCESS) return resFFT;
				resFFT = allocateBuffer(vkGPU, &tempBuffer[i], &tempBufferDeviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize[i]);
				if (resFFT != VKFFT_SUCCESS) return resFFT;
			}

			configuration.bufferNum = numBuf;
			configuration.tempBufferNum = numBuf;

			configuration.buffer = buffer;
			configuration.tempBuffer = tempBuffer;

			configuration.bufferSize = bufferSize;
			configuration.tempBufferSize = bufferSize;

			//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
			/*float* buffer_input = (float*)malloc(bufferSize);
			for (uint64_t k = 0; k < configuration.size[2]; k++) {
				for (uint64_t j = 0; j < configuration.size[1]; j++) {
					for (uint64_t i = 0; i < configuration.size[0]; i++) {
						buffer_input[2 * (i + j * configuration.size[0] + k * (configuration.size[0]) * configuration.size[1] + v * (configuration.size[0]) * configuration.size[1] * configuration.size[2])] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						buffer_input[2 * (i + j * configuration.size[0] + k * (configuration.size[0]) * configuration.size[1] + v * (configuration.size[0]) * configuration.size[1] * configuration.size[2]) + 1] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						}
					}
				}
			*/
			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
			uint64_t shift = 0;
			for (uint64_t i = 0; i < numBuf; i++) {
				resFFT = transferDataFromCPU(vkGPU, (buffer_input + shift / sizeof(float)), &buffer[i], bufferSize[i]);
				if (resFFT != VKFFT_SUCCESS) return resFFT;
				shift += bufferSize[i];
			}

			//free(buffer_input);

			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			resFFT = initializeVkFFT(&app, configuration);
			if (resFFT != VKFFT_SUCCESS) return resFFT;

			//Submit FFT+iFFT.
			uint64_t num_iter = (((uint64_t)4096 * 1024.0 * 1024.0) / (numBuf * bufferSize[0]) > 1000) ? 1000 : (uint64_t)((uint64_t)4096 * 1024.0 * 1024.0) / (numBuf * bufferSize[0]);
			if (vkGPU->physicalDeviceProperties.vendorID == 0x8086) num_iter /= 4;
			if (num_iter == 0) num_iter = 1;
			if (vkGPU->physicalDeviceProperties.vendorID != 0x8086) num_iter *= 5;
			double totTime = 0;
			VkFFTLaunchParams launchParams = {};
			resFFT = performVulkanFFTiFFT(vkGPU, &app, &launchParams, num_iter, &totTime);
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
						fprintf(output, "VkFFT System: %" PRIu64 " %" PRIu64 "x%" PRIu64 " Buffer: %" PRIu64 " MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %" PRIu64 " benchmark: %" PRIu64 " bandwidth: %0.1f\n", (uint64_t)log2(configuration.size[0]), configuration.size[0], configuration.numberBatches, (numBuf * bufferSize[0]) / 1024 / 1024, avg_time, std_error, num_iter, (uint64_t)(((double)(numBuf * bufferSize[0]) / 1024) / avg_time), (numBuf * bufferSize[0]) / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);

					printf("VkFFT System: %" PRIu64 " %" PRIu64 "x%" PRIu64 " Buffer: %" PRIu64 " MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %" PRIu64 " benchmark: %" PRIu64 " bandwidth: %0.1f\n", (uint64_t)log2(configuration.size[0]), configuration.size[0], configuration.numberBatches, (numBuf * bufferSize[0]) / 1024 / 1024, avg_time, std_error, num_iter, (uint64_t)(((double)(numBuf * bufferSize[0]) / 1024) / avg_time), (numBuf * bufferSize[0]) / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);
					benchmark_result += ((double)numBuf * bufferSize[0] / 1024) / avg_time;
				}


			}
			for (uint64_t i = 0; i < numBuf; i++) {

				vkDestroyBuffer(vkGPU->device, buffer[i], NULL);
				vkDestroyBuffer(vkGPU->device, tempBuffer[i], NULL);
				vkFreeMemory(vkGPU->device, bufferDeviceMemory[i], NULL);
				vkFreeMemory(vkGPU->device, tempBufferDeviceMemory[i], NULL);

			}
			free(buffer);
			free(bufferDeviceMemory);
			free(tempBuffer);
			free(tempBufferDeviceMemory);
			free(bufferSize);
			deleteVkFFT(&app);

		}
	}
	free(buffer_input);
	benchmark_result /= 25;

	if (file_output) {
		fprintf(output, "Benchmark score VkFFT: %" PRIu64 "\n", (uint64_t)(benchmark_result));
		fprintf(output, "Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
	}
	printf("Benchmark score VkFFT: %" PRIu64 "\n", (uint64_t)(benchmark_result));
	printf("Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
#endif
	return resFFT;
}
