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
#include "fftw3.h"
#ifdef USE_cuFFT
#include "precision_cuFFT_single.h"
#endif	
#ifdef USE_rocFFT
#include "precision_rocFFT_single.h"
#endif	
VkFFTResult sample_11_precision_VkFFT_single(VkGPU* vkGPU, uint64_t file_output, FILE* output, uint64_t isCompilerInitialized)
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
		fprintf(output, "11 - VkFFT/FFTW C2C precision test in single precision\n");
	printf("11 - VkFFT/FFTW C2C precision test in single precision\n");

	const int num_benchmark_samples = 63;
	const int num_runs = 1;

	uint64_t benchmark_dimensions[num_benchmark_samples][5] = {  {(uint64_t)pow(2,5), 1, 1, 1, 1}, {(uint64_t)pow(2,6), 1, 1, 1, 1},{(uint64_t)pow(2,7), 1, 1, 1, 1},{(uint64_t)pow(2,8), 1, 1, 1, 1},{(uint64_t)pow(2,9), 1, 1, 1, 1},{(uint64_t)pow(2,10), 1, 1, 1, 1},
		{(uint64_t)pow(2,11), 1, 1, 1, 1},{(uint64_t)pow(2,12), 1, 1, 1, 1},{(uint64_t)pow(2,13), 1, 1, 1, 1},{(uint64_t)pow(2,14), 1, 1, 1, 1},{(uint64_t)pow(2,15), 1, 1, 1, 1},{(uint64_t)pow(2,16), 1, 1, 1, 1},{(uint64_t)pow(2,17), 1, 1, 1, 1},{(uint64_t)pow(2,18), 1, 1, 1, 1},
		{(uint64_t)pow(2,19), 1, 1, 1, 1},{(uint64_t)pow(2,20), 1, 1, 1, 1},{(uint64_t)pow(2,21), 1, 1, 1, 1},{(uint64_t)pow(2,22), 1, 1, 1, 1},{(uint64_t)pow(2,23), 1, 1, 1, 1},{(uint64_t)pow(2,24), 1, 1, 1, 1},{(uint64_t)pow(2,25), 1, 1, 1, 1},{(uint64_t)pow(2,26), 1, 1, 1, 1},

		{8, (uint64_t)pow(2,3), 1, 1, 2},{8, (uint64_t)pow(2,4), 1, 1, 2},{8, (uint64_t)pow(2,5), 1, 1, 2},{8, (uint64_t)pow(2,6), 1, 1, 2},{8, (uint64_t)pow(2,7), 1, 1, 2},{8, (uint64_t)pow(2,8), 1, 1, 2},{8, (uint64_t)pow(2,9), 1, 1, 2},{8, (uint64_t)pow(2,10), 1, 1, 2},
		{8, (uint64_t)pow(2,11), 1, 1, 2},{8, (uint64_t)pow(2,12), 1, 1, 2},{8, (uint64_t)pow(2,13), 1, 1, 2},{8, (uint64_t)pow(2,14), 1, 1, 2},{8, (uint64_t)pow(2,15), 1, 1, 2},{8, (uint64_t)pow(2,16), 1, 1, 2},{8, (uint64_t)pow(2,17), 1, 1, 2},{8, (uint64_t)pow(2,18), 1, 1, 2},
		{8, (uint64_t)pow(2,19), 1, 1, 2},{8, (uint64_t)pow(2,20), 1, 1, 2},{8, (uint64_t)pow(2,21), 1, 1, 2},{8, (uint64_t)pow(2,22), 1, 1, 2},{8, (uint64_t)pow(2,23), 1, 1, 2},{8, (uint64_t)pow(2,24), 1, 1, 2},

		{ (uint64_t)pow(2,3), (uint64_t)pow(2,3), 1, 1, 2},{ (uint64_t)pow(2,4), (uint64_t)pow(2,4), 1, 1, 2},{ (uint64_t)pow(2,5), (uint64_t)pow(2,5), 1, 1, 2},{ (uint64_t)pow(2,6), (uint64_t)pow(2,6), 1, 1, 2},{ (uint64_t)pow(2,7), (uint64_t)pow(2,7), 1, 1, 2},{ (uint64_t)pow(2,8), (uint64_t)pow(2,8), 1, 1, 2},{ (uint64_t)pow(2,9), (uint64_t)pow(2,9), 1, 1, 2},
		{ (uint64_t)pow(2,10), (uint64_t)pow(2,10), 1, 1, 2},{ (uint64_t)pow(2,11), (uint64_t)pow(2,11), 1, 1, 2},{ (uint64_t)pow(2,12), (uint64_t)pow(2,12), 1, 1, 2},{ (uint64_t)pow(2,13), (uint64_t)pow(2,13), 1, 1, 2},{ (uint64_t)pow(2,14), (uint64_t)pow(2,13), 1, 1, 2},

		{ (uint64_t)pow(2,3), (uint64_t)pow(2,3), (uint64_t)pow(2,3), 1, 3},{ (uint64_t)pow(2,4), (uint64_t)pow(2,4), (uint64_t)pow(2,4), 1, 3},{ (uint64_t)pow(2,5), (uint64_t)pow(2,5), (uint64_t)pow(2,5), 1, 3},{ (uint64_t)pow(2,6), (uint64_t)pow(2,6), (uint64_t)pow(2,6), 1, 3},{ (uint64_t)pow(2,7), (uint64_t)pow(2,7), (uint64_t)pow(2,7), 1, 3},{ (uint64_t)pow(2,8), (uint64_t)pow(2,8), (uint64_t)pow(2,8), 1, 3},{ (uint64_t)pow(2,9), (uint64_t)pow(2,9), (uint64_t)pow(2,9), 1, 3},
	};

	double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples

	for (int n = 0; n < num_benchmark_samples; n++) {
		for (int r = 0; r < num_runs; r++) {

			fftwf_complex* inputC;
			fftw_complex* inputC_double;
			uint64_t dims[3] = { benchmark_dimensions[n][0] , benchmark_dimensions[n][1] ,benchmark_dimensions[n][2] };

			inputC = (fftwf_complex*)(malloc(sizeof(fftwf_complex) * dims[0] * dims[1] * dims[2]));
			if (!inputC) return VKFFT_ERROR_MALLOC_FAILED;
			inputC_double = (fftw_complex*)(malloc(sizeof(fftw_complex) * dims[0] * dims[1] * dims[2]));
			if (!inputC_double) return VKFFT_ERROR_MALLOC_FAILED;
			for (uint64_t l = 0; l < dims[2]; l++) {
				for (uint64_t j = 0; j < dims[1]; j++) {
					for (uint64_t i = 0; i < dims[0]; i++) {
						inputC[i + j * dims[0] + l * dims[0] * dims[1]][0] = (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
						inputC[i + j * dims[0] + l * dims[0] * dims[1]][1] = (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
						inputC_double[i + j * dims[0] + l * dims[0] * dims[1]][0] = (double)inputC[i + j * dims[0] + l * dims[0] * dims[1]][0];
						inputC_double[i + j * dims[0] + l * dims[0] * dims[1]][1] = (double)inputC[i + j * dims[0] + l * dims[0] * dims[1]][1];
					}
				}
			}

			fftw_plan p;

			fftw_complex* output_FFTW = (fftw_complex*)(malloc(sizeof(fftw_complex) * dims[0] * dims[1] * dims[2]));
			if (!output_FFTW) return VKFFT_ERROR_MALLOC_FAILED;
			switch (benchmark_dimensions[n][4]) {
			case 1:
				p = fftw_plan_dft_1d((int)benchmark_dimensions[n][0], inputC_double, output_FFTW, -1, FFTW_ESTIMATE);
				break;
			case 2:
				p = fftw_plan_dft_2d((int)benchmark_dimensions[n][1], (int)benchmark_dimensions[n][0], inputC_double, output_FFTW, -1, FFTW_ESTIMATE);
				break;
			case 3:
				p = fftw_plan_dft_3d((int)benchmark_dimensions[n][2], (int)benchmark_dimensions[n][1], (int)benchmark_dimensions[n][0], inputC_double, output_FFTW, -1, FFTW_ESTIMATE);
				break;
			}

			fftw_execute(p);

			float totTime = 0;
			int num_iter = 1;

#ifdef USE_cuFFT
			fftwf_complex* output_extFFT = (fftwf_complex*)(malloc(sizeof(fftwf_complex) * dims[0] * dims[1] * dims[2]));
			if (!output_extFFT) return VKFFT_ERROR_MALLOC_FAILED;
			launch_precision_cuFFT_single(inputC, (void*)output_extFFT, (int)vkGPU->device_id, benchmark_dimensions[n]);
#endif // USE_cuFFT
#ifdef USE_rocFFT
			fftwf_complex* output_extFFT = (fftwf_complex*)(malloc(sizeof(fftwf_complex) * dims[0] * dims[1] * dims[2]));
			if (!output_extFFT) return VKFFT_ERROR_MALLOC_FAILED;
			launch_precision_rocFFT_single(inputC, (void*)output_extFFT, (int)vkGPU->device_id, benchmark_dimensions[n]);
#endif // USE_rocFFT
			//VkFFT part

			VkFFTConfiguration configuration = {};
			VkFFTApplication app = {};
			configuration.FFTdim = benchmark_dimensions[n][4]; //FFT dimension, 1D, 2D or 3D (default 1).
			configuration.size[0] = benchmark_dimensions[n][0]; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			configuration.size[1] = benchmark_dimensions[n][1];
			configuration.size[2] = benchmark_dimensions[n][2];

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

			uint64_t numBuf = 1;

			//Allocate buffers for the input data. - we use 4 in this example
			uint64_t* bufferSize = (uint64_t*)malloc(sizeof(uint64_t) * numBuf);
			if (!bufferSize) return VKFFT_ERROR_MALLOC_FAILED;
			for (uint64_t i = 0; i < numBuf; i++) {
				bufferSize[i] = {};
				bufferSize[i] = (uint64_t)sizeof(float) * 2 * configuration.size[0] * configuration.size[1] * configuration.size[2] / numBuf;
			}
#if(VKFFT_BACKEND_VULKAN)
			VkBuffer* buffer = (VkBuffer*)malloc(numBuf * sizeof(VkBuffer));
			if (!buffer) return VKFFT_ERROR_MALLOC_FAILED;
			VkDeviceMemory* bufferDeviceMemory = (VkDeviceMemory*)malloc(numBuf * sizeof(VkDeviceMemory));
			if (!bufferDeviceMemory) return VKFFT_ERROR_MALLOC_FAILED;
#elif(VKFFT_BACKEND_CUDA)
			cuFloatComplex* buffer = 0;
#elif(VKFFT_BACKEND_HIP)
			hipFloatComplex* buffer = 0;
#elif(VKFFT_BACKEND_OPENCL)
			cl_mem buffer = 0;
#elif(VKFFT_BACKEND_LEVEL_ZERO)
			void* buffer = 0;
#elif(VKFFT_BACKEND_METAL)
            MTL::Buffer* buffer = 0;
#endif
			for (uint64_t i = 0; i < numBuf; i++) {
#if(VKFFT_BACKEND_VULKAN)
				buffer[i] = {};
				bufferDeviceMemory[i] = {};
				resFFT = allocateBuffer(vkGPU, &buffer[i], &bufferDeviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize[i]);
				if (resFFT != VKFFT_SUCCESS) return resFFT;
#elif(VKFFT_BACKEND_CUDA)
				res = cudaMalloc((void**)&buffer, bufferSize[i]);
				if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
#elif(VKFFT_BACKEND_HIP)
				res = hipMalloc((void**)&buffer, bufferSize[i]);
				if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
#elif(VKFFT_BACKEND_OPENCL)
				buffer = clCreateBuffer(vkGPU->context, CL_MEM_READ_WRITE, bufferSize[i], 0, &res);
				if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
#elif(VKFFT_BACKEND_LEVEL_ZERO)
				ze_device_mem_alloc_desc_t device_desc = {};
				device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
				res = zeMemAllocDevice(vkGPU->context, &device_desc, bufferSize[i], sizeof(float), vkGPU->device, &buffer);
				if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
#elif(VKFFT_BACKEND_METAL)
                buffer = vkGPU->device->newBuffer(bufferSize[i], MTL::ResourceStorageModePrivate);
#endif
			}

			configuration.bufferNum = numBuf;
			/*
#if(VKFFT_BACKEND_VULKAN)
			configuration.buffer = buffer;
#elif(VKFFT_BACKEND_CUDA)
			configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND_HIP)
			configuration.buffer = (void**)&buffer;
#endif
			*/ //Can specify buffers at launch
			configuration.bufferSize = bufferSize;

			//Sample buffer transfer tool. Uses staging buffer (if needed) of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
			uint64_t shift = 0;
			for (uint64_t i = 0; i < numBuf; i++) {
#if(VKFFT_BACKEND_VULKAN)
				resFFT = transferDataFromCPU(vkGPU, (inputC + shift / sizeof(fftwf_complex)), &buffer[i], bufferSize[i]);
				if (resFFT != VKFFT_SUCCESS) return resFFT;
#else
                resFFT = transferDataFromCPU(vkGPU, (inputC + shift / sizeof(fftwf_complex)), &buffer, bufferSize[i]);
                if (resFFT != VKFFT_SUCCESS) return resFFT;
#endif
				shift += bufferSize[i];
			}
			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			resFFT = initializeVkFFT(&app, configuration);
			if (resFFT != VKFFT_SUCCESS) return resFFT;

			//Submit FFT+iFFT.
			//num_iter = 1;
			//specify buffers at launch
			VkFFTLaunchParams launchParams = {};
#if(VKFFT_BACKEND_VULKAN)
			launchParams.buffer = buffer;
#elif(VKFFT_BACKEND_CUDA)
			launchParams.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND_HIP)
			launchParams.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND_OPENCL)
			launchParams.buffer = &buffer;
#elif(VKFFT_BACKEND_LEVEL_ZERO)
			launchParams.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND_METAL)
            launchParams.buffer = &buffer;
#endif
			resFFT = performVulkanFFT(vkGPU, &app, &launchParams, -1, num_iter);
			if (resFFT != VKFFT_SUCCESS) return resFFT;
			fftwf_complex* output_VkFFT = (fftwf_complex*)(malloc(sizeof(fftwf_complex) * dims[0] * dims[1] * dims[2]));
			if (!output_VkFFT) return VKFFT_ERROR_MALLOC_FAILED;
			//Transfer data from GPU using staging buffer.
			shift = 0;
			for (uint64_t i = 0; i < numBuf; i++) {
#if(VKFFT_BACKEND_VULKAN)
				resFFT = transferDataToCPU(vkGPU, (output_VkFFT + shift / sizeof(fftwf_complex)), &buffer[i], bufferSize[i]);
				if (resFFT != VKFFT_SUCCESS) return resFFT;
#else
                resFFT = transferDataToCPU(vkGPU, (output_VkFFT + shift / sizeof(fftwf_complex)), &buffer, bufferSize[i]);
                if (resFFT != VKFFT_SUCCESS) return resFFT;
#endif
				shift += bufferSize[i];
			}
			double avg_difference[2] = { 0,0 };
			double max_difference[2] = { 0,0 };
			double avg_eps[2] = { 0,0 };
			double max_eps[2] = { 0,0 };
			for (uint64_t l = 0; l < dims[2]; l++) {
				for (uint64_t j = 0; j < dims[1]; j++) {
					for (uint64_t i = 0; i < dims[0]; i++) {
						uint64_t loc_i = i;
						uint64_t loc_j = j;
						uint64_t loc_l = l;

						//if (file_output) fprintf(output, "%.2e %.2e - %.2e %.2e \n", output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] / N, output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1] / N, output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][0], output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][1]);

						//printf("%.2e %.2e - %.2e %.2e \n", output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0], output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1], output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][0], output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][1]);
						double current_data_norm = sqrt(output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] * output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] + output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1] * output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1]);
#if defined(USE_cuFFT) || defined(USE_rocFFT)
						double current_diff_x_extFFT = (output_extFFT[loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]][0] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0]);
						double current_diff_y_extFFT = (output_extFFT[loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]][1] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1]);
						double current_diff_norm_extFFT = sqrt(current_diff_x_extFFT * current_diff_x_extFFT + current_diff_y_extFFT * current_diff_y_extFFT);
						if (current_diff_norm_extFFT > max_difference[0]) max_difference[0] = current_diff_norm_extFFT;
						avg_difference[0] += current_diff_norm_extFFT;
						if ((current_diff_norm_extFFT / current_data_norm > max_eps[0])) {
							max_eps[0] = current_diff_norm_extFFT / current_data_norm;
						}
						avg_eps[0] += current_diff_norm_extFFT / current_data_norm;
#endif

						double current_diff_x_VkFFT = (output_VkFFT[loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]][0] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0]);
						double current_diff_y_VkFFT = (output_VkFFT[loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]][1] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1]);
						double current_diff_norm_VkFFT = sqrt(current_diff_x_VkFFT * current_diff_x_VkFFT + current_diff_y_VkFFT * current_diff_y_VkFFT);
						if (current_diff_norm_VkFFT > max_difference[1]) max_difference[1] = current_diff_norm_VkFFT;
						avg_difference[1] += current_diff_norm_VkFFT;
						if ((current_diff_norm_VkFFT / current_data_norm > max_eps[1])) {
							max_eps[1] = current_diff_norm_VkFFT / current_data_norm;
						}
						avg_eps[1] += current_diff_norm_VkFFT / current_data_norm;
					}
				}
			}
			avg_difference[0] /= (dims[0] * dims[1] * dims[2]);
			avg_eps[0] /= (dims[0] * dims[1] * dims[2]);
			avg_difference[1] /= (dims[0] * dims[1] * dims[2]);
			avg_eps[1] /= (dims[0] * dims[1] * dims[2]);
#ifdef USE_cuFFT
			if (file_output)
				fprintf(output, "cuFFT System: %" PRIu64 "x%" PRIu64 "x%" PRIu64 " avg_difference: %.2e max_difference: %.2e avg_eps: %.2e max_eps: %.2e\n", dims[0], dims[1], dims[2], avg_difference[0], max_difference[0], avg_eps[0], max_eps[0]);
			printf("cuFFT System: %" PRIu64 "x%" PRIu64 "x%" PRIu64 " avg_difference: %.2e max_difference: %.2e avg_eps: %.2e max_eps: %.2e\n", dims[0], dims[1], dims[2], avg_difference[0], max_difference[0], avg_eps[0], max_eps[0]);
#endif
#ifdef USE_rocFFT
			if (file_output)
				fprintf(output, "rocFFT System: %" PRIu64 "x%" PRIu64 "x%" PRIu64 " avg_difference: %.2e max_difference: %.2e avg_eps: %.2e max_eps: %.2e\n", dims[0], dims[1], dims[2], avg_difference[0], max_difference[0], avg_eps[0], max_eps[0]);
			printf("rocFFT System: %" PRIu64 "x%" PRIu64 "x%" PRIu64 " avg_difference: %.2e max_difference: %.2e avg_eps: %.2e max_eps: %.2e\n", dims[0], dims[1], dims[2], avg_difference[0], max_difference[0], avg_eps[0], max_eps[0]);
#endif
			if (file_output)
				fprintf(output, "VkFFT System: %" PRIu64 "x%" PRIu64 "x%" PRIu64 " avg_difference: %.2e max_difference: %.2e avg_eps: %.2e max_eps: %.2e\n", dims[0], dims[1], dims[2], avg_difference[1], max_difference[1], avg_eps[1], max_eps[1]);
			printf("VkFFT System: %" PRIu64 "x%" PRIu64 "x%" PRIu64 " avg_difference: %.2e max_difference: %.2e avg_eps: %.2e max_eps: %.2e\n", dims[0], dims[1], dims[2], avg_difference[1], max_difference[1], avg_eps[1], max_eps[1]);
			free(output_VkFFT);
			for (uint64_t i = 0; i < numBuf; i++) {
#if(VKFFT_BACKEND_VULKAN)
				vkDestroyBuffer(vkGPU->device, buffer[i], NULL);
				vkFreeMemory(vkGPU->device, bufferDeviceMemory[i], NULL);
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
			}
#if(VKFFT_BACKEND_VULKAN)
			free(buffer);
			free(bufferDeviceMemory);
#endif
#if defined(USE_cuFFT) || defined(USE_rocFFT)
			free(output_extFFT);
#endif
			free(bufferSize);
			deleteVkFFT(&app);
			free(inputC);
			fftw_destroy_plan(p);
			free(inputC_double);
			free(output_FFTW);
		}
	}
	return resFFT;
}
