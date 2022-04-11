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

VkFFTResult sample_52_convolution_VkFFT_single_2d_batched_r2c(VkGPU* vkGPU, uint64_t file_output, FILE* output, uint64_t isCompilerInitialized)
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
		fprintf(output, "52 - VkFFT batched convolution example with identitiy kernel\n");
	printf("52 - VkFFT batched convolution example with identitiy kernel\n");
	//Configuration + FFT application.
	VkFFTConfiguration configuration = {};
	VkFFTConfiguration convolution_configuration = {};
	VkFFTApplication app_convolution = {};
	VkFFTApplication app_kernel = {};
	//Convolution sample code
	//Setting up FFT configuration. FFT is performed in-place with no performance loss. 

	configuration.FFTdim = 2; //FFT dimension, 1D, 2D or 3D (default 1).
	configuration.size[0] = 32; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z. 
	configuration.size[1] = 32;
	configuration.size[2] = 1;

	configuration.kernelConvolution = true; //specify if this plan is used to create kernel for convolution
	configuration.performR2C = true; //Perform R2C/C2R transform. Can be combined with all other options. Reduces memory requirements by a factor of 2. Requires special input data alignment: for x*y*z system pad x*y plane to (x+2)*y with last 2*y elements reserved, total array dimensions are (x*y+2y)*z. Memory layout after R2C and before C2R can be found on github.
	configuration.coordinateFeatures = 2; //Specify dimensionality of the input feature vector (default 1). Each component is stored not as a vector, but as a separate system and padded on it's own according to other options (i.e. for x*y system of 3-vector, first x*y elements correspond to the first dimension, then goes x*y for the second, etc).
	//coordinateFeatures number is an important constant for convolution. If we perform 1x1 convolution, it is equal to number of features, but matrixConvolution should be equal to 1. For matrix convolution, it must be equal to matrixConvolution parameter. If we perform 2x2 convolution, it is equal to 3 for symmetric kernel (stored as xx, xy, yy) and 4 for nonsymmetric (stored as xx, xy, yx, yy). Similarly, 6 (stored as xx, xy, xz, yy, yz, zz) and 9 (stored as xx, xy, xz, yx, yy, yz, zx, zy, zz) for 3x3 convolutions. 
	configuration.normalize = 1;//normalize iFFT
	
	configuration.numberBatches = 2;
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
	uint64_t kernelSize = ((uint64_t)configuration.numberBatches) * configuration.coordinateFeatures * sizeof(float) * 2 * (configuration.size[0] / 2 + 1) * configuration.size[1] * configuration.size[2];;

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
	for (uint64_t f = 0; f < configuration.numberBatches; f++) {
		for (uint64_t v = 0; v < configuration.coordinateFeatures; v++) {
			for (uint64_t k = 0; k < configuration.size[2]; k++) {
				for (uint64_t j = 0; j < configuration.size[1]; j++) {

					//Below is the test identity kernel for 1x1 nonsymmetric FFT, multiplied by (f * configuration.coordinateFeatures + v + 1);
					for (uint64_t i = 0; i < configuration.size[0] / 2 + 1; i++) {

						kernel_input[2 * i + j * (configuration.size[0] + 2) + k * (configuration.size[0] + 2) * configuration.size[1] + v * (configuration.size[0] + 2) * configuration.size[1] * configuration.size[2] + f * configuration.coordinateFeatures * (configuration.size[0] + 2) * configuration.size[1] * configuration.size[2]] = (float)(f * configuration.coordinateFeatures + v + 1.0);
						kernel_input[2 * i + 1 + j * (configuration.size[0] + 2) + k * (configuration.size[0] + 2) * configuration.size[1] + v * (configuration.size[0] + 2) * configuration.size[1] * configuration.size[2] + f * configuration.coordinateFeatures * (configuration.size[0] + 2) * configuration.size[1] * configuration.size[2]] = 0;

					}
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
	convolution_configuration.kernelConvolution = false;
	convolution_configuration.performConvolution = true;
	convolution_configuration.symmetricKernel = false;//Specify if convolution kernel is symmetric. In this case we only pass upper triangle part of it in the form of: (xx, xy, yy) for 2d and (xx, xy, xz, yy, yz, zz) for 3d.

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

	convolution_configuration.kernelSize = &kernelSize;
	convolution_configuration.numberBatches = 1;//one batch - numberKernels convolutions
	convolution_configuration.numberKernels = configuration.numberBatches;// number of convolutions on a single input
	//Allocate separate buffer for the input data.
	uint64_t inputBufferSize = ((uint64_t)convolution_configuration.coordinateFeatures) * sizeof(float) * 2 * (convolution_configuration.size[0] / 2 + 1) * convolution_configuration.size[1] * convolution_configuration.size[2];;
	uint64_t bufferSize = convolution_configuration.numberKernels * convolution_configuration.coordinateFeatures * sizeof(float) * 2 * (convolution_configuration.size[0] / 2 + 1) * convolution_configuration.size[1] * convolution_configuration.size[2];;
	convolution_configuration.isInputFormatted = true; //if input is a different buffer, it doesn't have to be zeropadded/R2C padded	

#if(VKFFT_BACKEND==0)
	VkBuffer inputBuffer = {};
	VkBuffer buffer = {};
	VkDeviceMemory inputBufferDeviceMemory = {};
	VkDeviceMemory bufferDeviceMemory = {};
	resFFT = allocateBuffer(vkGPU, &inputBuffer, &inputBufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, inputBufferSize);
	if (resFFT != VKFFT_SUCCESS) return resFFT;
	resFFT = allocateBuffer(vkGPU, &buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
	if (resFFT != VKFFT_SUCCESS) return resFFT;
	convolution_configuration.inputBuffer = &inputBuffer;
	convolution_configuration.buffer = &buffer;
#elif(VKFFT_BACKEND==1)
	cuFloatComplex* inputBuffer = 0;
	cuFloatComplex* buffer = 0;
	res = cudaMalloc((void**)&inputBuffer, inputBufferSize);
	if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
	res = cudaMalloc((void**)&buffer, bufferSize);
	if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
	convolution_configuration.inputBuffer = (void**)&inputBuffer;
	convolution_configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==2)
	hipFloatComplex* inputBuffer = 0;
	hipFloatComplex* buffer = 0;
	res = hipMalloc((void**)&inputBuffer, inputBufferSize);
	if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
	res = hipMalloc((void**)&buffer, bufferSize);
	if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
	convolution_configuration.inputBuffer = (void**)&inputBuffer;
	convolution_configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==3)
	cl_mem inputBuffer = 0;
	cl_mem buffer = 0;
	inputBuffer = clCreateBuffer(vkGPU->context, CL_MEM_READ_WRITE, inputBufferSize, 0, &res);
	if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
	buffer = clCreateBuffer(vkGPU->context, CL_MEM_READ_WRITE, bufferSize, 0, &res);
	if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
	convolution_configuration.inputBuffer = &inputBuffer;
	convolution_configuration.buffer = &buffer;
#elif(VKFFT_BACKEND==4)
	void* inputBuffer = 0;
	void* buffer = 0;
	res = zeMemAllocDevice(vkGPU->context, &device_desc, inputBufferSize, sizeof(float), vkGPU->device, &inputBuffer);
	if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
	res = zeMemAllocDevice(vkGPU->context, &device_desc, bufferSize, sizeof(float), vkGPU->device, &buffer);
	if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
	convolution_configuration.inputBuffer = &inputBuffer;
	configuration.buffer = &buffer;
#endif

	convolution_configuration.inputBufferSize = &inputBufferSize;
	convolution_configuration.bufferSize = &bufferSize;


	if (file_output)
		fprintf(output, "Total memory needed for buffer: %" PRIu64 " MB\n", bufferSize / 1024 / 1024);
	printf("Total memory needed for buffer: %" PRIu64 " MB\n", bufferSize / 1024 / 1024);
	//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
	float* buffer_input = (float*)malloc(inputBufferSize);
	if (!buffer_input) return VKFFT_ERROR_MALLOC_FAILED;
	for (uint64_t v = 0; v < convolution_configuration.coordinateFeatures; v++) {
		for (uint64_t k = 0; k < convolution_configuration.size[2]; k++) {
			for (uint64_t j = 0; j < convolution_configuration.size[1]; j++) {
				for (uint64_t i = 0; i < convolution_configuration.size[0]; i++) {
					buffer_input[i + j * (convolution_configuration.size[0] + 2) + k * (convolution_configuration.size[0] + 2) * convolution_configuration.size[1] + v * (convolution_configuration.size[0] + 2) * convolution_configuration.size[1] * convolution_configuration.size[2]] = 1;
				}
			}
		}
	}
	//Transfer data to GPU using staging buffer.
#if(VKFFT_BACKEND==0)
	resFFT = transferDataFromCPU(vkGPU, buffer_input, &inputBuffer, inputBufferSize);
	if (resFFT != VKFFT_SUCCESS) return resFFT;
#elif(VKFFT_BACKEND==1)
	res = cudaMemcpy(inputBuffer, buffer_input, inputBufferSize, cudaMemcpyHostToDevice);
	if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==2)
	res = hipMemcpy(inputBuffer, buffer_input, inputBufferSize, hipMemcpyHostToDevice);
	if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==3)
	res = clEnqueueWriteBuffer(vkGPU->commandQueue, inputBuffer, CL_TRUE, 0, inputBufferSize, buffer_input, 0, NULL, NULL);
	if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==4)
	res = zeCommandListReset(copyCommandList);
	if (res != ZE_RESULT_SUCCESS)return VKFFT_ERROR_FAILED_TO_DESTROY_COMMAND_LIST;
	res = zeCommandListCreateImmediate(vkGPU->context, vkGPU->device, &commandQueueCopyDesc, &copyCommandList);
	if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
	res = zeCommandListAppendMemoryCopy(copyCommandList, inputBuffer, buffer_input, inputBufferSize, 0, 0, 0);
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
	for (uint64_t f = 0; f < convolution_configuration.numberKernels; f++) {
		if (file_output)
			fprintf(output, "\nKernel id: %" PRIu64 "\n\n", f);
		printf("\nKernel id: %" PRIu64 "\n\n", f);
		for (uint64_t v = 0; v < convolution_configuration.coordinateFeatures; v++) {
			if (file_output)
				fprintf(output, "\ncoordinate: %" PRIu64 "\n\n", v);
			printf("\ncoordinate: %" PRIu64 "\n\n", v);
			for (uint64_t k = 0; k < convolution_configuration.size[2]; k++) {
				for (uint64_t j = 0; j < convolution_configuration.size[1]; j++) {
					for (uint64_t i = 0; i < convolution_configuration.size[0]; i++) {
						if (file_output)
							fprintf(output, "%.6f ", buffer_output[i + j * (convolution_configuration.size[0] + 2) + k * (convolution_configuration.size[0] + 2) * convolution_configuration.size[1] + v * (convolution_configuration.size[0] + 2) * convolution_configuration.size[1] * convolution_configuration.size[2] + f * convolution_configuration.coordinateFeatures * (convolution_configuration.size[0] + 2) * convolution_configuration.size[1] * convolution_configuration.size[2]]);

						printf("%.6f ", buffer_output[i + j * (convolution_configuration.size[0] + 2) + k * (convolution_configuration.size[0] + 2) * convolution_configuration.size[1] + v * (convolution_configuration.size[0] + 2) * convolution_configuration.size[1] * convolution_configuration.size[2] + f * convolution_configuration.coordinateFeatures * (convolution_configuration.size[0] + 2) * convolution_configuration.size[1] * convolution_configuration.size[2]]);
					}
					std::cout << "\n";
				}
			}
		}
	}
	free(kernel_input);
	free(buffer_input);
	free(buffer_output);
#if(VKFFT_BACKEND==0)
	vkDestroyBuffer(vkGPU->device, inputBuffer, NULL);
	vkFreeMemory(vkGPU->device, inputBufferDeviceMemory, NULL);
	vkDestroyBuffer(vkGPU->device, buffer, NULL);
	vkFreeMemory(vkGPU->device, bufferDeviceMemory, NULL);
	vkDestroyBuffer(vkGPU->device, kernel, NULL);
	vkFreeMemory(vkGPU->device, kernelDeviceMemory, NULL);
#elif(VKFFT_BACKEND==1)
	cudaFree(inputBuffer);
	cudaFree(buffer);
	cudaFree(kernel);
#elif(VKFFT_BACKEND==2)
	hipFree(inputBuffer);
	hipFree(buffer);
	hipFree(kernel);
#elif(VKFFT_BACKEND==3)
	clReleaseMemObject(inputBuffer);
	clReleaseMemObject(buffer);
	clReleaseMemObject(kernel);
#elif(VKFFT_BACKEND==4)
	zeMemFree(vkGPU->context, inputBuffer);
	zeMemFree(vkGPU->context, buffer);
	zeMemFree(vkGPU->context, kernel);
#endif	
	deleteVkFFT(&app_kernel);
	deleteVkFFT(&app_convolution);
	return resFFT;
}
