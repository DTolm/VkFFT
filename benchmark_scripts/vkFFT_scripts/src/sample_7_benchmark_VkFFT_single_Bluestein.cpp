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

VkFFTResult sample_7_benchmark_VkFFT_single_Bluestein(VkGPU* vkGPU, uint64_t file_output, FILE* output, uint64_t isCompilerInitialized) 
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
	if (file_output)
		fprintf(output, "7 - VkFFT FFT + iFFT C2C Bluestein benchmark in single precision\n");
	printf("7 - VkFFT FFT + iFFT C2C Bluestein benchmark in single precision\n");

	const int num_benchmark_samples = 54;
	const int num_runs = 3;
	uint64_t benchmark_dimensions[num_benchmark_samples][4] = { {1024, 1024, 1, 2},
	{17, 17, 1, 2},{19, 19, 1, 2},{23, 23, 1, 2}, {29, 29, 1, 2},{31, 31, 1, 2},{37, 37, 1, 2},{41, 41, 1, 2},{43, 43, 1, 2},{47, 47, 1, 2},{53, 53, 1, 2},{59, 59, 1, 2},{61, 61, 1, 2},{67, 67, 1, 2},{71, 71, 1, 2},{73, 73, 1, 2},{79, 79, 1, 2},{83, 83, 1, 2},{89, 89, 1, 2},{97, 97, 1, 2},
	{17, 17, 17, 3},{19, 19, 19, 3},{23, 23, 23, 3}, {29, 29, 29, 3},{31, 31, 31, 3},{37, 37, 37, 3},{41, 41, 41, 3},{43, 43, 43, 3},{47, 47, 47, 3},{53, 53, 53, 3},{59, 59, 59, 3},{61, 61, 61, 3},{67, 67, 67, 3},{71, 71, 71, 3},{73, 73, 73, 3},{79, 79, 79, 3},{83, 83, 83, 3},{89, 89, 89, 3},{97, 97, 97, 3}, 
	{179, 179, 1, 2},{283, 283, 1, 2},{419, 419, 1, 2}, {547, 547, 1, 2},{661, 661, 1, 2},{811, 811, 1, 2},{947, 947, 1, 2},{1087, 1087, 1, 2},{1229, 1229, 1, 2},{1381, 1381, 1, 2},{1523, 1523, 1, 2},{2909, 2909, 1, 2},{4241, 4241, 1, 2},{6841, 6841, 1, 2},{7727, 7727, 1, 2}
	};
	double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	//memory allocated on the CPU once, makes benchmark completion faster + avoids performance issues connected to frequent allocation/deallocation.
	float* buffer_input = (float*)malloc((uint64_t)4 * 2 * (uint64_t)pow(2, 27));
	if (!buffer_input) return VKFFT_ERROR_MALLOC_FAILED;
	for (uint64_t i = 0; i < 2 * (uint64_t)pow(2, 27); i++) {
		buffer_input[i] = (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
	}
	for (uint64_t n = 0; n < num_benchmark_samples; n++) {
		double run_time[num_runs];
		for (uint64_t r = 0; r < num_runs; r++) {
			//Configuration + FFT application .
			VkFFTConfiguration configuration = {};
			VkFFTApplication app = {};
			//FFT + iFFT sample code.
			//Setting up FFT configuration for forward and inverse FFT.
			configuration.FFTdim = benchmark_dimensions[n][3]; //FFT dimension, 1D, 2D or 3D (default 1).
			configuration.size[0] = benchmark_dimensions[n][0]; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			configuration.size[1] = benchmark_dimensions[n][1];
			configuration.size[2] = benchmark_dimensions[n][2];
			if (r==0) configuration.saveApplicationToString = 1;
			if (r!=0) configuration.loadApplicationFromString = 1;
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
			uint64_t bufferSize = (uint64_t)sizeof(float) * 2 * configuration.size[0] * configuration.size[1] * configuration.size[2];
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
			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
#if(VKFFT_BACKEND==0)
			resFFT = transferDataFromCPU(vkGPU, buffer_input, &buffer, bufferSize);
			if (resFFT != VKFFT_SUCCESS) return resFFT;
#elif(VKFFT_BACKEND==1)
			res = cudaMemcpy(buffer, buffer_input, bufferSize, cudaMemcpyHostToDevice);
			if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==2)
			res = hipMemcpy(buffer, buffer_input, bufferSize, hipMemcpyHostToDevice);
			if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==3)
			res = clEnqueueWriteBuffer(vkGPU->commandQueue, buffer, CL_TRUE, 0, bufferSize, buffer_input, 0, NULL, NULL);
			if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==4)
			ze_command_queue_desc_t commandQueueCopyDesc = {
				ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
				0,
				vkGPU->commandQueueID,
				0, // index
				0, // flags
				ZE_COMMAND_QUEUE_MODE_DEFAULT,
				ZE_COMMAND_QUEUE_PRIORITY_NORMAL
			};
			ze_command_list_handle_t copyCommandList;
			res = zeCommandListCreateImmediate(vkGPU->context, vkGPU->device, &commandQueueCopyDesc, &copyCommandList);
			if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
			res = zeCommandListAppendMemoryCopy(copyCommandList, buffer, buffer_input, bufferSize, 0, 0, 0);
			if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_COPY;
			res = zeCommandQueueSynchronize(vkGPU->commandQueue, UINT32_MAX);
			if (res != 0) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
#endif
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
#if(VKFFT_BACKEND==0)
			if (vkGPU->physicalDeviceProperties.vendorID == 0x8086) num_iter /= 4;
#elif(VKFFT_BACKEND==3)
			cl_uint vendorID;
			clGetDeviceInfo(vkGPU->device, CL_DEVICE_VENDOR_ID, sizeof(cl_int), &vendorID, 0);
			if (vendorID == 0x8086) num_iter /= 4;//smaller benchmark for Intel GPUs
#elif(VKFFT_BACKEND==4)
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
						fprintf(output, "VkFFT System: %" PRIu64 "x%" PRIu64 "x%" PRIu64 " Buffer: %" PRIu64 " MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %" PRIu64 " benchmark: %" PRIu64 " bandwidth: %0.1f\n", benchmark_dimensions[n][0], benchmark_dimensions[n][1], benchmark_dimensions[n][2], bufferSize / 1024 / 1024, avg_time, std_error, num_iter, (uint64_t)(((double)bufferSize / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);
					printf("VkFFT System: %" PRIu64 "x%" PRIu64 "x%" PRIu64 " Buffer: %" PRIu64 " MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %" PRIu64 " benchmark: %" PRIu64 " bandwidth: %0.1f\n", benchmark_dimensions[n][0], benchmark_dimensions[n][1], benchmark_dimensions[n][2], bufferSize / 1024 / 1024, avg_time, std_error, num_iter, (uint64_t)(((double)bufferSize / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);
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
	free(buffer_input);
	benchmark_result /= ((int64_t)num_benchmark_samples - 1);

	if (file_output) {
		fprintf(output, "Benchmark score VkFFT: %" PRIu64 "\n", (uint64_t)(benchmark_result));
#if(VKFFT_BACKEND==0)
		fprintf(output, "Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
#endif
	}
	printf("Benchmark score VkFFT: %" PRIu64 "\n", (uint64_t)(benchmark_result));
#if(VKFFT_BACKEND==0)
	printf("Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
#endif
	return resFFT;
}
