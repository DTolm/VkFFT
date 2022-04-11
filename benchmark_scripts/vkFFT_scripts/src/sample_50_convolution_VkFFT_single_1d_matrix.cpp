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

VkFFTResult sample_50_convolution_VkFFT_single_1d_matrix(VkGPU* vkGPU, uint64_t file_output, FILE* output, uint64_t isCompilerInitialized)
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
		fprintf(output, "50 - VkFFT convolution example with identitiy kernel\n");
	printf("50 - VkFFT convolution example with identitiy kernel\n");
	//7 - convolution
	//Configuration + FFT application.
	VkFFTConfiguration configuration = {};
	VkFFTConfiguration convolution_configuration = {};
	VkFFTApplication app_convolution = {};
	VkFFTApplication app_kernel = {};
	//Convolution sample code
	//Setting up FFT configuration. FFT is performed in-place with no performance loss. 

	configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).
	configuration.size[0] = 1024 * 1024 * 8; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z. 
	configuration.size[1] = 1;
	configuration.size[2] = 1;

	configuration.kernelConvolution = true; //specify if this plan is used to create kernel for convolution
	configuration.coordinateFeatures = 9; //Specify dimensionality of the input feature vector (default 1). Each component is stored not as a vector, but as a separate system and padded on it's own according to other options (i.e. for x*y system of 3-vector, first x*y elements correspond to the first dimension, then goes x*y for the second, etc).
	//coordinateFeatures number is an important constant for convolution. If we perform 1x1 convolution, it is equal to number of features, but matrixConvolution should be equal to 1. For matrix convolution, it must be equal to matrixConvolution parameter. If we perform 2x2 convolution, it is equal to 3 for symmetric kernel (stored as xx, xy, yy) and 4 for nonsymmetric (stored as xx, xy, yx, yy). Similarly, 6 (stored as xx, xy, xz, yy, yz, zz) and 9 (stored as xx, xy, xz, yx, yy, yz, zx, zy, zz) for 3x3 convolutions. 
	configuration.normalize = 1;//normalize iFFT
	
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
	//In this example, we perform a convolution for a real vectorfield (3vector) with a symmetric kernel (6 values). We use configuration to initialize convolution kernel first from real data, then we create convolution_configuration for convolution. The buffer object from configuration is passed to convolution_configuration as kernel object.
	//1. Kernel forward FFT.
	uint64_t kernelSize = ((uint64_t)configuration.coordinateFeatures) * sizeof(float) * 2 * (configuration.size[0]) * configuration.size[1] * configuration.size[2];;

#if(VKFFT_BACKEND==0)
	VkBuffer kernel = {};
	VkDeviceMemory kernelDeviceMemory = {};
	resFFT = allocateBuffer(vkGPU, &kernel, &kernelDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, kernelSize);
	if (resFFT != VKFFT_SUCCESS) return resFFT;
	configuration.buffer = &kernel;
#elif(VKFFT_BACKEND==1)
	cuFloatComplex* kernel = 0;
	res = cudaMalloc((void**)&kernel, kernelSize);
	if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
	configuration.buffer = (void**)&kernel;
#elif(VKFFT_BACKEND==2)
	hipFloatComplex* kernel = 0;
	res = hipMalloc((void**)&kernel, kernelSize);
	if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
	configuration.buffer = (void**)&kernel;
#elif(VKFFT_BACKEND==3)
	cl_mem kernel = 0;
	kernel = clCreateBuffer(vkGPU->context, CL_MEM_READ_WRITE, kernelSize, 0, &res);
	if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
	configuration.buffer = &kernel;
#elif(VKFFT_BACKEND==4)
	void* kernel = 0;
	ze_device_mem_alloc_desc_t device_desc = {};
	device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
	res = zeMemAllocDevice(vkGPU->context, &device_desc, kernelSize, sizeof(float), vkGPU->device, &kernel);
	if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
	configuration.buffer = &kernel;
#endif

	configuration.bufferSize = &kernelSize;

	if (file_output)
		fprintf(output, "Total memory needed for kernel: %" PRIu64 " MB\n", kernelSize / 1024 / 1024);
	printf("Total memory needed for kernel: %" PRIu64 " MB\n", kernelSize / 1024 / 1024);
	//Fill kernel on CPU.
	float* kernel_input = (float*)malloc(kernelSize);
	if (!kernel_input) return VKFFT_ERROR_MALLOC_FAILED;
	for (uint64_t v = 0; v < configuration.coordinateFeatures; v++) {
		for (uint64_t k = 0; k < configuration.size[2]; k++) {
			for (uint64_t j = 0; j < configuration.size[1]; j++) {

				//for (uint64_t i = 0; i < configuration.size[0]; i++) {
				//	kernel_input[i + j * configuration.size[0] + k * (configuration.size[0] + 2) * configuration.size[1] + v * (configuration.size[0] + 2) * configuration.size[1] * configuration.size[2]] = 1;

				//Below is the test identity kernel for 3x3 nonsymmetric FFT
				for (uint64_t i = 0; i < configuration.size[0]; i++) {
					if ((v == 0) || (v == 4) || (v == 8))

						kernel_input[2 * (i + j * (configuration.size[0]) + k * (configuration.size[0]) * configuration.size[1] + v * (configuration.size[0]) * configuration.size[1] * configuration.size[2])] = 1;

					else
						kernel_input[2 * (i + j * (configuration.size[0]) + k * (configuration.size[0]) * configuration.size[1] + v * (configuration.size[0]) * configuration.size[1] * configuration.size[2])] = 0;
					kernel_input[2 * (i + j * (configuration.size[0]) + k * (configuration.size[0]) * configuration.size[1] + v * (configuration.size[0]) * configuration.size[1] * configuration.size[2]) + 1] = 0;

				}
			}
		}
	}
	//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
#if(VKFFT_BACKEND==0)
	resFFT = transferDataFromCPU(vkGPU, kernel_input, &kernel, kernelSize);
	if (resFFT != VKFFT_SUCCESS) return resFFT;
#elif(VKFFT_BACKEND==1)
	res = cudaMemcpy(kernel, kernel_input, kernelSize, cudaMemcpyHostToDevice);
	if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==2)
	res = hipMemcpy(kernel, kernel_input, kernelSize, hipMemcpyHostToDevice);
	if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==3)
	res = clEnqueueWriteBuffer(vkGPU->commandQueue, kernel, CL_TRUE, 0, kernelSize, kernel_input, 0, NULL, NULL);
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
	res = zeCommandListAppendMemoryCopy(copyCommandList, kernel, kernel_input, kernelSize, 0, 0, 0);
	if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_COPY;
	res = zeCommandQueueSynchronize(vkGPU->commandQueue, UINT32_MAX);
	if (res != 0) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
#endif

	//Initialize application responsible for the kernel. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
	resFFT = initializeVkFFT(&app_kernel, configuration);
	if (resFFT != VKFFT_SUCCESS) return resFFT;
	//Sample forward FFT command buffer allocation + execution performed on kernel. Second number determines how many times perform application in one submit. FFT can also be appended to user defined command buffers.

	//Uncomment the line below if you want to perform kernel FFT. In this sample we use predefined identitiy kernel.
	//performVulkanFFT(vkGPU, &app_kernel, -1, 1);

	//The kernel has been trasnformed.


	//2. Buffer convolution with transformed kernel.
	//Copy configuration, as it mostly remains unchanged. Change specific parts.
	convolution_configuration = configuration;
	configuration.kernelConvolution = false;
	convolution_configuration.performConvolution = true;
	convolution_configuration.symmetricKernel = false;//Specify if convolution kernel is symmetric. In this case we only pass upper triangle part of it in the form of: (xx, xy, yy) for 2d and (xx, xy, xz, yy, yz, zz) for 3d.
	convolution_configuration.matrixConvolution = 3;//we do matrix convolution, so kernel is 9 numbers (3x3), but vector dimension is 3
	convolution_configuration.coordinateFeatures = 3;//equal to matrixConvolution size

#if(VKFFT_BACKEND==0)
	convolution_configuration.kernel = &kernel;
#elif(VKFFT_BACKEND==1)
	convolution_configuration.kernel = (void**)&kernel;
#elif(VKFFT_BACKEND==2)
	convolution_configuration.kernel = (void**)&kernel;
#elif(VKFFT_BACKEND==3)
	convolution_configuration.kernel = &kernel;
#elif(VKFFT_BACKEND==4)
	convolution_configuration.kernel = (void**)&kernel;
#endif	

	//Allocate separate buffer for the input data.
	uint64_t bufferSize = ((uint64_t)convolution_configuration.coordinateFeatures) * sizeof(float) * 2 * (convolution_configuration.size[0]) * convolution_configuration.size[1] * convolution_configuration.size[2];;
#if(VKFFT_BACKEND==0)
	VkBuffer buffer = {};
	VkDeviceMemory bufferDeviceMemory = {};
	resFFT = allocateBuffer(vkGPU, &buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
	if (resFFT != VKFFT_SUCCESS) return resFFT;
	convolution_configuration.buffer = &buffer;
#elif(VKFFT_BACKEND==1)
	cuFloatComplex* buffer = 0;
	res = cudaMalloc((void**)&buffer, bufferSize);
	if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
	convolution_configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==2)
	hipFloatComplex* buffer = 0;
	res = hipMalloc((void**)&buffer, bufferSize);
	if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
	convolution_configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==3)
	cl_mem buffer = 0;
	buffer = clCreateBuffer(vkGPU->context, CL_MEM_READ_WRITE, bufferSize, 0, &res);
	if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
	configuration.buffer = &buffer;
#elif(VKFFT_BACKEND==4)
	void* buffer = 0;
	res = zeMemAllocDevice(vkGPU->context, &device_desc, bufferSize, sizeof(float), vkGPU->device, &buffer);
	if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
	configuration.buffer = &buffer;
#endif

	convolution_configuration.bufferSize = &bufferSize;
	convolution_configuration.kernelSize = &kernelSize;

	if (file_output)
		fprintf(output, "Total memory needed for buffer: %" PRIu64 " MB\n", bufferSize / 1024 / 1024);
	printf("Total memory needed for buffer: %" PRIu64 " MB\n", bufferSize / 1024 / 1024);
	//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
	float* buffer_input = (float*)malloc(bufferSize);
	if (!buffer_input) return VKFFT_ERROR_MALLOC_FAILED;
	for (uint64_t v = 0; v < convolution_configuration.coordinateFeatures; v++) {
		for (uint64_t k = 0; k < convolution_configuration.size[2]; k++) {
			for (uint64_t j = 0; j < convolution_configuration.size[1]; j++) {
				for (uint64_t i = 0; i < convolution_configuration.size[0]; i++) {
					buffer_input[2 * (i + j * convolution_configuration.size[0] + k * (convolution_configuration.size[0]) * convolution_configuration.size[1] + v * (convolution_configuration.size[0]) * convolution_configuration.size[1] * convolution_configuration.size[2])] = (float)(i % 8 - 3.5);
					buffer_input[2 * (i + j * convolution_configuration.size[0] + k * (convolution_configuration.size[0]) * convolution_configuration.size[1] + v * (convolution_configuration.size[0]) * convolution_configuration.size[1] * convolution_configuration.size[2]) + 1] = (float)(i % 4 - 1.5);
				}
			}
		}
	}
	//Transfer data to GPU using staging buffer.
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
	res = zeCommandListReset(copyCommandList);
	if (res != ZE_RESULT_SUCCESS)return VKFFT_ERROR_FAILED_TO_DESTROY_COMMAND_LIST;
	res = zeCommandListCreateImmediate(vkGPU->context, vkGPU->device, &commandQueueCopyDesc, &copyCommandList);
	if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
	res = zeCommandListAppendMemoryCopy(copyCommandList, buffer, buffer_input, bufferSize, 0, 0, 0);
	if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_COPY;
	res = zeCommandQueueSynchronize(vkGPU->commandQueue, UINT32_MAX);
	if (res != 0) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
#endif

	//Initialize application responsible for the convolution.
	resFFT = initializeVkFFT(&app_convolution, convolution_configuration);
	if (resFFT != VKFFT_SUCCESS) return resFFT;
	//Sample forward FFT command buffer allocation + execution performed on kernel. FFT can also be appended to user defined command buffers.
	VkFFTLaunchParams launchParams = {};
	resFFT = performVulkanFFT(vkGPU, &app_convolution, &launchParams, -1, 1);
	if (resFFT != VKFFT_SUCCESS) return resFFT;
	//The kernel has been trasnformed.

	float* buffer_output = (float*)malloc(bufferSize);
	if (!buffer_output) return VKFFT_ERROR_MALLOC_FAILED;
	//Transfer data from GPU using staging buffer.
#if(VKFFT_BACKEND==0)
	resFFT = transferDataToCPU(vkGPU, buffer_output, &buffer, bufferSize);
	if (resFFT != VKFFT_SUCCESS) return resFFT;
#elif(VKFFT_BACKEND==1)
	res = cudaMemcpy(buffer_output, buffer, bufferSize, cudaMemcpyDeviceToHost);
	if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==2)
	res = hipMemcpy(buffer_output, buffer, bufferSize, hipMemcpyDeviceToHost);
	if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==3)
	res = clEnqueueReadBuffer(vkGPU->commandQueue, buffer, CL_TRUE, 0, bufferSize, buffer_output, 0, NULL, NULL);
	if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==4)
	res = zeCommandListReset(copyCommandList);
	if (res != ZE_RESULT_SUCCESS)return VKFFT_ERROR_FAILED_TO_DESTROY_COMMAND_LIST;
	res = zeCommandListCreateImmediate(vkGPU->context, vkGPU->device, &commandQueueCopyDesc, &copyCommandList);
	if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
	res = zeCommandListAppendMemoryCopy(copyCommandList, buffer_output, buffer, bufferSize, 0, 0, 0);
	if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_COPY;
	res = zeCommandQueueSynchronize(vkGPU->commandQueue, UINT32_MAX);
	if (res != 0) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
#endif
	//Print data, if needed.
	for (uint64_t v = 0; v < convolution_configuration.coordinateFeatures; v++) {
		if (file_output)
			fprintf(output, "\ncoordinate: %" PRIu64 "\n\n", v);
		printf("\ncoordinate: %" PRIu64 "\n\n", v);
		for (uint64_t k = 0; k < convolution_configuration.size[2]; k++) {
			for (uint64_t j = 0; j < convolution_configuration.size[1]; j++) {
				for (uint64_t i = 0; i < convolution_configuration.size[0]; i++) {
					if (file_output)
						fprintf(output, "%.6f %.6f ", buffer_output[2 * (i + j * convolution_configuration.size[0] + k * (convolution_configuration.size[0]) * convolution_configuration.size[1] + v * (convolution_configuration.size[0]) * convolution_configuration.size[1] * convolution_configuration.size[2])], buffer_output[2 * (i + j * convolution_configuration.size[0] + k * (convolution_configuration.size[0]) * convolution_configuration.size[1] + v * (convolution_configuration.size[0]) * convolution_configuration.size[1] * convolution_configuration.size[2]) + 1]);
					printf("%.6f %.6f ", buffer_output[2 * (i + j * convolution_configuration.size[0] + k * (convolution_configuration.size[0]) * convolution_configuration.size[1] + v * (convolution_configuration.size[0]) * convolution_configuration.size[1] * convolution_configuration.size[2])], buffer_output[2 * (i + j * convolution_configuration.size[0] + k * (convolution_configuration.size[0]) * convolution_configuration.size[1] + v * (convolution_configuration.size[0]) * convolution_configuration.size[1] * convolution_configuration.size[2]) + 1]);
				}
				std::cout << "\n";
			}
		}
	}
	free(kernel_input);
	free(buffer_input);
	free(buffer_output);
#if(VKFFT_BACKEND==0)
	vkDestroyBuffer(vkGPU->device, buffer, NULL);
	vkFreeMemory(vkGPU->device, bufferDeviceMemory, NULL);
	vkDestroyBuffer(vkGPU->device, kernel, NULL);
	vkFreeMemory(vkGPU->device, kernelDeviceMemory, NULL);
#elif(VKFFT_BACKEND==1)
	cudaFree(buffer);
	cudaFree(kernel);
#elif(VKFFT_BACKEND==2)
	hipFree(buffer);
	hipFree(kernel);
#elif(VKFFT_BACKEND==3)
	clReleaseMemObject(buffer);
	clReleaseMemObject(kernel);
#elif(VKFFT_BACKEND==4)
	zeMemFree(vkGPU->context, buffer);
	zeMemFree(vkGPU->context, kernel);
#endif	
	deleteVkFFT(&app_kernel);
	deleteVkFFT(&app_convolution);
	return resFFT;
}
