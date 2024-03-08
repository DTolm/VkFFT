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

#if(VKFFT_BACKEND_VULKAN)
#include "vulkan/vulkan.h"
#include "glslang/Include/glslang_c_interface.h"
#elif(VKFFT_BACKEND_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#elif(VKFFT_BACKEND_HIP)
#ifndef __HIP_PLATFORM_HCC__
#define __HIP_PLATFORM_HCC__
#endif
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_complex.h>
#elif(VKFFT_BACKEND_OPENCL)
#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif 
#elif(VKFFT_BACKEND_LEVEL_ZERO)
#include <ze_api.h>
#elif(VKFFT_BACKEND_METAL)
#include "Foundation/Foundation.hpp"
#include "QuartzCore/QuartzCore.hpp"
#include "Metal/Metal.hpp"
#endif
#include "vkFFT.h"
#include "utils_VkFFT.h"

VkFFTResult sample_1001_benchmark_VkFFT_double_2_4096(VkGPU* vkGPU, uint64_t file_output, FILE* output, uint64_t isCompilerInitialized)
{
	VkFFTResult resFFT = VKFFT_SUCCESS;
#if(VKFFT_BACKEND_VULKAN)
	VkResult res = VK_SUCCESS;
#elif(VKFFT_BACKEND_CUDA)
	cudaError_t res = cudaSuccess;
#elif(VKFFT_BACKEND_HIP)
	hipError_t res = hipSuccess;
#elif(VKFFT_BACKEND_OPENCL)
	cl_int res = CL_SUCCESS;
#elif(VKFFT_BACKEND_LEVEL_ZERO)
	ze_result_t res = ZE_RESULT_SUCCESS;
#elif(VKFFT_BACKEND_METAL)
#endif
	if (file_output)
		fprintf(output, "1001 - VkFFT FFT + iFFT C2C benchmark 1D batched in double precision: all supported systems from 2 to 4096\n");
	printf("1001 - VkFFT FFT + iFFT C2C benchmark 1D batched in double precision: all supported systems from 2 to 4096\n");
	const int num_runs = 3;
	double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	//memory allocated on the CPU once, makes benchmark completion faster + avoids performance issues connected to frequent allocation/deallocation.
	double* buffer_input = (double*)malloc((uint64_t)8 * 2 * (uint64_t)pow(2, 27));
	if (!buffer_input) return VKFFT_ERROR_MALLOC_FAILED;
	for (uint64_t i = 0; i < 2 * (uint64_t)pow(2, 27); i++) {
		buffer_input[i] = (double)(2 * ((double)rand()) / RAND_MAX - 1.0);
	}
	int num_systems = 0;
	for (int n = 1; n < 4097; n++) {
		double run_time[num_runs];
		for (uint64_t r = 0; r < num_runs; r++) {
			//Configuration + FFT application .
			VkFFTConfiguration configuration = {};
			VkFFTApplication app = {};
			//FFT + iFFT sample code.
			//Setting up FFT configuration for forward and inverse FFT.
			configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).
			configuration.size[0] = n;// 4 * pow(2, n); //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			if (n == 1) configuration.size[0] = 4096;
			uint64_t temp = configuration.size[0];

			/*for (uint64_t j = 2; j < 14; j++)
			{
				if (temp % j == 0) {
					temp /= j;
					j = 1;
				}
			}
			if (temp != 1) break;*/
			configuration.numberBatches = (uint64_t)pow(2, (uint64_t)log2((uint64_t)64 * 32 * (uint64_t)pow(2, 15) / configuration.size[0]));
			if (configuration.numberBatches < 1) configuration.numberBatches = 1;
			configuration.size[2] = 1;

			configuration.doublePrecision = true;
#if(!VKFFT_BACKEND_METAL)
			if (r==0) configuration.saveApplicationToString = 1;
			if (r!=0) configuration.loadApplicationFromString = 1;
#endif
			
			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
#if(VKFFT_BACKEND_METAL)
            configuration.device = vkGPU->device;
#else
            configuration.device = &vkGPU->device;
#endif
#if(VKFFT_BACKEND_VULKAN)
			configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
			configuration.fence = &vkGPU->fence;
			configuration.commandPool = &vkGPU->commandPool;
			configuration.physicalDevice = &vkGPU->physicalDevice;
			configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization
#elif(VKFFT_BACKEND_OPENCL)
			configuration.context = &vkGPU->context;
#elif(VKFFT_BACKEND_LEVEL_ZERO)
			configuration.context = &vkGPU->context;
			configuration.commandQueue = &vkGPU->commandQueue;
			configuration.commandQueueID = vkGPU->commandQueueID;
#elif(VKFFT_BACKEND_METAL)
            configuration.queue = vkGPU->queue;
#endif			

			//Allocate buffer for the input data.
			uint64_t bufferSize = (uint64_t)sizeof(double) * 2 * configuration.size[0] * configuration.numberBatches;
#if(VKFFT_BACKEND_VULKAN)
			VkBuffer buffer = {};
			VkDeviceMemory bufferDeviceMemory = {};
			resFFT = allocateBuffer(vkGPU, &buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
			if (resFFT != VKFFT_SUCCESS) return resFFT;
			configuration.buffer = &buffer;
#elif(VKFFT_BACKEND_CUDA)
			cuFloatComplex* buffer = 0;
			res = cudaMalloc((void**)&buffer, bufferSize);
			if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
			configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND_HIP)
			hipFloatComplex* buffer = 0;
			res = hipMalloc((void**)&buffer, bufferSize);
			if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
			configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND_OPENCL)
			cl_mem buffer = 0;
			buffer = clCreateBuffer(vkGPU->context, CL_MEM_READ_WRITE, bufferSize, 0, &res);
			if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
			configuration.buffer = &buffer;
#elif(VKFFT_BACKEND_LEVEL_ZERO)
			void* buffer = 0;
			ze_device_mem_alloc_desc_t device_desc = {};
			device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
			res = zeMemAllocDevice(vkGPU->context, &device_desc, bufferSize, sizeof(float), vkGPU->device, &buffer);
			if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
			configuration.buffer = &buffer;
#elif(VKFFT_BACKEND_METAL)
            MTL::Buffer* buffer = 0;
            buffer = vkGPU->device->newBuffer(bufferSize, MTL::ResourceStorageModePrivate);
            configuration.buffer = &buffer;
#endif

			configuration.bufferSize = &bufferSize;
			//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
			/*float* buffer_input = (float*)malloc(bufferSize);

			for (uint64_t k = 0; k < configuration.size[2]; k++) {
				for (uint64_t j = 0; j < configuration.size[1]; j++) {
					for (uint64_t i = 0; i < configuration.size[0]; i++) {
						buffer_input[2 * (i + j * configuration.size[0] + k * (configuration.size[0]) * configuration.size[1])] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						buffer_input[2 * (i + j * configuration.size[0] + k * (configuration.size[0]) * configuration.size[1]) + 1] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						}
					}
				}
			*/
			//Sample buffer transfer tool. Uses staging buffer (if needed) of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
            resFFT = transferDataFromCPU(vkGPU, buffer_input, &buffer, bufferSize);
            if (resFFT != VKFFT_SUCCESS) return resFFT;
			//free(buffer_input);

			if (configuration.loadApplicationFromString) {
				FILE* kernelCache;
				uint64_t str_len;
				char fname[500];
				int VkFFT_version = VkFFTGetVersion();
				sprintf(fname, "VkFFT_binary");
				kernelCache = fopen(fname, "rb");
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
				sprintf(fname, "VkFFT_binary");
				kernelCache = fopen(fname, "wb");
				fwrite(app.saveApplicationString, app.applicationStringSize, 1, kernelCache);
				fclose(kernelCache);
			}

			//Submit FFT+iFFT.
			uint64_t num_iter = (((uint64_t)4096 * 1024.0 * 1024.0) / bufferSize > 1000) ? 1000 : (uint64_t)((uint64_t)4096 * 1024.0 * 1024.0) / bufferSize;
#if(VKFFT_BACKEND_VULKAN)
			if (vkGPU->physicalDeviceProperties.vendorID == 0x8086) num_iter /= 4;
#elif(VKFFT_BACKEND_OPENCL)
			cl_uint vendorID;
			clGetDeviceInfo(vkGPU->device, CL_DEVICE_VENDOR_ID, sizeof(cl_int), &vendorID, 0);
			if (vendorID == 0x8086) num_iter /= 4;//smaller benchmark for Intel GPUs
#elif(VKFFT_BACKEND_LEVEL_ZERO)
			ze_device_properties_t device_properties;
			res = zeDeviceGetProperties(vkGPU->device, &device_properties);
			if (res != 0) return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
			if (device_properties.vendorId == 0x8086) num_iter /= 4;//smaller benchmark for Intel GPUs
#endif
			if (num_iter == 0) num_iter = 1;
			double totTime = 0;
			VkFFTLaunchParams launchParams = {};
			resFFT = performVulkanFFTiFFT(vkGPU, &app, &launchParams, num_iter, &totTime);
			if (resFFT != VKFFT_SUCCESS) return resFFT;
			run_time[r] = totTime;
			if (n > 1) {
				if (r == num_runs - 1) {
					num_systems++;
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
						fprintf(output, "VkFFT System: %" PRIu64 " %" PRIu64 " Buffer: %" PRIu64 " MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %" PRIu64 " benchmark: %" PRIu64 " scaled_bandwidth: %0.1f device_bandwidth: %0.1f\n", configuration.size[0], configuration.numberBatches, bufferSize / 1024 / 1024, avg_time, std_error, num_iter, (uint64_t)(((double)bufferSize * sizeof(float) / sizeof(double) / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * 4 / avg_time, bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);

					printf("VkFFT System: %" PRIu64 " %" PRIu64 " Buffer: %" PRIu64 " MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %" PRIu64 " benchmark: %" PRIu64 " scaled_bandwidth: %0.1f device_bandwidth: %0.1f\n", configuration.size[0], configuration.numberBatches, bufferSize / 1024 / 1024, avg_time, std_error, num_iter, (uint64_t)(((double)bufferSize * sizeof(float) / sizeof(double) / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * 4 / avg_time, bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);
					benchmark_result += ((double)bufferSize * sizeof(float) / sizeof(double) / 1024) / avg_time;
				}


			}

#if(VKFFT_BACKEND_VULKAN)
			vkDestroyBuffer(vkGPU->device, buffer, NULL);
			vkFreeMemory(vkGPU->device, bufferDeviceMemory, NULL);
#elif(VKFFT_BACKEND_CUDA)
			cudaFree(buffer);
#elif(VKFFT_BACKEND_HIP)
			hipFree(buffer);
#elif(VKFFT_BACKEND_OPENCL)
			clReleaseMemObject(buffer);
#elif(VKFFT_BACKEND_LEVEL_ZERO)
			zeMemFree(vkGPU->context, buffer);
#elif(VKFFT_BACKEND_METAL)
            buffer->release();
#endif
			deleteVkFFT(&app);

		}
	}
	free(buffer_input);
	benchmark_result /= (num_systems);
	if (file_output) {
		fprintf(output, "Benchmark score VkFFT: %" PRIu64 "\n", (uint64_t)(benchmark_result));
#if(VKFFT_BACKEND_VULKAN)
		fprintf(output, "Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
#endif
	}
	printf("Benchmark score VkFFT: %" PRIu64 "\n", (uint64_t)(benchmark_result));
#if(VKFFT_BACKEND_VULKAN)
	printf("Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
#endif
	return resFFT;
}
