#include <vector>
#include <memory>
#include <string.h>
#include <chrono>
#include <thread>
#include <iostream>
#include <algorithm>
#define __STDC_FORMAT_MACROS
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
#include "user_benchmark_VkFFT.h"
#include "sample_0_benchmark_VkFFT_single.h"
#include "sample_1_benchmark_VkFFT_double.h"
#if(VKFFT_BACKEND==0)
#include "sample_2_benchmark_VkFFT_half.h"
#endif
#include "sample_3_benchmark_VkFFT_single_3d.h"
#include "sample_4_benchmark_VkFFT_single_3d_zeropadding.h"
#include "sample_5_benchmark_VkFFT_single_disableReorderFourStep.h"
#include "sample_6_benchmark_VkFFT_single_r2c.h"
#include "sample_7_benchmark_VkFFT_single_Bluestein.h"
#include "sample_8_benchmark_VkFFT_double_Bluestein.h"
#include "sample_10_benchmark_VkFFT_single_multipleBuffers.h"
#ifdef USE_FFTW
#include "sample_11_precision_VkFFT_single.h"
#include "sample_12_precision_VkFFT_double.h"
#if(VKFFT_BACKEND==0)
#include "sample_13_precision_VkFFT_half.h"
#endif
#include "sample_14_precision_VkFFT_single_nonPow2.h"
#include "sample_15_precision_VkFFT_single_r2c.h"
#include "sample_16_precision_VkFFT_single_dct.h"
#include "sample_17_precision_VkFFT_double_dct.h"
#include "sample_18_precision_VkFFT_double_nonPow2.h"
#endif
#include "sample_50_convolution_VkFFT_single_1d_matrix.h"
#include "sample_51_convolution_VkFFT_single_3d_matrix_zeropadding_r2c.h"
#include "sample_52_convolution_VkFFT_single_2d_batched_r2c.h"

#include "sample_100_benchmark_VkFFT_single_nd_dct.h"
#include "sample_101_benchmark_VkFFT_double_nd_dct.h"
#include "sample_1000_VkFFT_single_2_4096.h"
#include "sample_1001_benchmark_VkFFT_double_2_4096.h"
#include "sample_1003_benchmark_VkFFT_single_3d_2_512.h"

#ifdef USE_cuFFT
#include "user_benchmark_cuFFT.h"
#include "sample_0_benchmark_cuFFT_single.h"
#include "sample_1_benchmark_cuFFT_double.h"
#include "sample_2_benchmark_cuFFT_half.h"
#include "sample_3_benchmark_cuFFT_single_3d.h"
#include "sample_6_benchmark_cuFFT_single_r2c.h"
#include "sample_7_benchmark_cuFFT_single_Bluestein.h"
#include "sample_8_benchmark_cuFFT_double_Bluestein.h"
#include "sample_1000_benchmark_cuFFT_single_2_4096.h"
#include "sample_1001_benchmark_cuFFT_double_2_4096.h"
#include "sample_1003_benchmark_cuFFT_single_3d_2_512.h"
#endif  
#ifdef USE_rocFFT
#include "user_benchmark_rocFFT.h"
#include "sample_0_benchmark_rocFFT_single.h"
#include "sample_1_benchmark_rocFFT_double.h"
#include "sample_3_benchmark_rocFFT_single_3d.h"
#include "sample_6_benchmark_rocFFT_single_r2c.h"
#include "sample_7_benchmark_rocFFT_single_Bluestein.h"
#include "sample_8_benchmark_rocFFT_double_Bluestein.h"
#include "sample_1000_benchmark_rocFFT_single_2_4096.h"
#include "sample_1001_benchmark_rocFFT_double_2_4096.h"
#include "sample_1003_benchmark_rocFFT_single_3d_2_512.h"
#endif 
#ifdef USE_FFTW
#include "fftw3.h"
#endif

VkFFTResult launchVkFFT(VkGPU* vkGPU, uint64_t sample_id, bool file_output, FILE* output, VkFFTUserSystemParameters* userParams) {
	//Sample Vulkan project GPU initialization.
	VkFFTResult resFFT = VKFFT_SUCCESS;

#if(VKFFT_BACKEND==0)
	VkResult res = VK_SUCCESS;
	//create instance - a connection between the application and the Vulkan library 
	res = createInstance(vkGPU, sample_id);
	if (res != 0) {
		//printf("Instance creation failed, error code: %" PRIu64 "\n", res);
		return VKFFT_ERROR_FAILED_TO_CREATE_INSTANCE;
	}
	//set up the debugging messenger 
	res = setupDebugMessenger(vkGPU);
	if (res != 0) {
		//printf("Debug messenger creation failed, error code: %" PRIu64 "\n", res);
		return VKFFT_ERROR_FAILED_TO_SETUP_DEBUG_MESSENGER;
	}
	//check if there are GPUs that support Vulkan and select one
	res = findPhysicalDevice(vkGPU);
	if (res != 0) {
		//printf("Physical device not found, error code: %" PRIu64 "\n", res);
		return VKFFT_ERROR_FAILED_TO_FIND_PHYSICAL_DEVICE;
	}
	//create logical device representation
	res = createDevice(vkGPU, sample_id);
	if (res != 0) {
		//printf("Device creation failed, error code: %" PRIu64 "\n", res);
		return VKFFT_ERROR_FAILED_TO_CREATE_DEVICE;
	}
	//create fence for synchronization 
	res = createFence(vkGPU);
	if (res != 0) {
		//printf("Fence creation failed, error code: %" PRIu64 "\n", res);
		return VKFFT_ERROR_FAILED_TO_CREATE_FENCE;
	}
	//create a place, command buffer memory is allocated from
	res = createCommandPool(vkGPU);
	if (res != 0) {
		//printf("Fence creation failed, error code: %" PRIu64 "\n", res);
		return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_POOL;
	}
	vkGetPhysicalDeviceProperties(vkGPU->physicalDevice, &vkGPU->physicalDeviceProperties);
	vkGetPhysicalDeviceMemoryProperties(vkGPU->physicalDevice, &vkGPU->physicalDeviceMemoryProperties);

	glslang_initialize_process();//compiler can be initialized before VkFFT
#elif(VKFFT_BACKEND==1)
	CUresult res = CUDA_SUCCESS;
	cudaError_t res2 = cudaSuccess;
	res = cuInit(0);
	if (res != CUDA_SUCCESS) return VKFFT_ERROR_FAILED_TO_INITIALIZE;
	res2 = cudaSetDevice((int)vkGPU->device_id);
	if (res2 != cudaSuccess) return VKFFT_ERROR_FAILED_TO_SET_DEVICE_ID;
	res = cuDeviceGet(&vkGPU->device, (int)vkGPU->device_id);
	if (res != CUDA_SUCCESS) return VKFFT_ERROR_FAILED_TO_GET_DEVICE;
	res = cuCtxCreate(&vkGPU->context, 0, (int)vkGPU->device);
	if (res != CUDA_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_CONTEXT;
#elif(VKFFT_BACKEND==2)
	hipError_t res = hipSuccess;
	res = hipInit(0);
	if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_INITIALIZE;
	res = hipSetDevice((int)vkGPU->device_id);
	if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_SET_DEVICE_ID;
	res = hipDeviceGet(&vkGPU->device, (int)vkGPU->device_id);
	if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_GET_DEVICE;
	res = hipCtxCreate(&vkGPU->context, 0, (int)vkGPU->device);
	if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_CREATE_CONTEXT;
#elif(VKFFT_BACKEND==3)
	cl_int res = CL_SUCCESS;
	cl_uint numPlatforms;
	res = clGetPlatformIDs(0, 0, &numPlatforms);
	if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_INITIALIZE;
	cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * numPlatforms);
	if (!platforms) return VKFFT_ERROR_MALLOC_FAILED;
	res = clGetPlatformIDs(numPlatforms, platforms, 0);
	if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_INITIALIZE;
	uint64_t k = 0;
	for (uint64_t j = 0; j < numPlatforms; j++) {
		cl_uint numDevices;
		res = clGetDeviceIDs(platforms[j], CL_DEVICE_TYPE_ALL, 0, 0, &numDevices);
		cl_device_id* deviceList = (cl_device_id*)malloc(sizeof(cl_device_id) * numDevices);
		if (!deviceList) return VKFFT_ERROR_MALLOC_FAILED;
		res = clGetDeviceIDs(platforms[j], CL_DEVICE_TYPE_ALL, numDevices, deviceList, 0);
		if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_GET_DEVICE;
		for (uint64_t i = 0; i < numDevices; i++) {
			if (k == vkGPU->device_id) {
				vkGPU->platform = platforms[j];
				vkGPU->device = deviceList[i];
				vkGPU->context = clCreateContext(NULL, 1, &vkGPU->device, NULL, NULL, &res);
				if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_CONTEXT;
				cl_command_queue commandQueue = clCreateCommandQueue(vkGPU->context, vkGPU->device, 0, &res);
				if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE;
				vkGPU->commandQueue = commandQueue;
				k++;
			}
			else {
				k++;
			}
		}
		free(deviceList);
	}
	free(platforms);
#elif(VKFFT_BACKEND==4)
	ze_result_t res = ZE_RESULT_SUCCESS;
	res = zeInit(0);
	if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_INITIALIZE;
	uint32_t numDrivers = 0;
	res = zeDriverGet(&numDrivers, 0);
	if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_INITIALIZE;
	ze_driver_handle_t* drivers = (ze_driver_handle_t*)malloc(numDrivers * sizeof(ze_driver_handle_t));
	if (!drivers) return VKFFT_ERROR_MALLOC_FAILED;
	res = zeDriverGet(&numDrivers, drivers);
	if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_INITIALIZE;
	uint64_t k = 0;
	for (uint64_t j = 0; j < numDrivers; j++) {
		uint32_t numDevices = 0;
		res = zeDeviceGet(drivers[j], &numDevices, nullptr);
		if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_GET_DEVICE;
		ze_device_handle_t* deviceList = (ze_device_handle_t*)malloc(numDevices * sizeof(ze_device_handle_t));
		if (!deviceList) return VKFFT_ERROR_MALLOC_FAILED;
		res = zeDeviceGet(drivers[j], &numDevices, deviceList);
		if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_GET_DEVICE;
		for (uint64_t i = 0; i < numDevices; i++) {
			if (k == vkGPU->device_id) {
				vkGPU->driver = drivers[j];
				vkGPU->device = deviceList[i];
				ze_context_desc_t contextDescription = {};
				contextDescription.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
				res = zeContextCreate(vkGPU->driver, &contextDescription, &vkGPU->context);
				if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_CONTEXT;

				uint32_t queueGroupCount = 0;
				res = zeDeviceGetCommandQueueGroupProperties(vkGPU->device, &queueGroupCount, 0);
				if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE;

				ze_command_queue_group_properties_t* cmdqueueGroupProperties = (ze_command_queue_group_properties_t*) malloc(queueGroupCount * sizeof(ze_command_queue_group_properties_t));
				if (!cmdqueueGroupProperties) return VKFFT_ERROR_MALLOC_FAILED;
				res = zeDeviceGetCommandQueueGroupProperties(vkGPU->device, &queueGroupCount, cmdqueueGroupProperties);
				if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE;

				uint32_t commandQueueID = -1;
				for (uint32_t i = 0; i < queueGroupCount; ++i) {
					if ((cmdqueueGroupProperties[i].flags && ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) && (cmdqueueGroupProperties[i].flags && ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY)) {
						commandQueueID = i;
						break;
					}
				}
				if (commandQueueID == -1) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE;
				vkGPU->commandQueueID = commandQueueID;
				ze_command_queue_desc_t commandQueueDescription = {};
				commandQueueDescription.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
				commandQueueDescription.ordinal = commandQueueID;
				commandQueueDescription.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
				commandQueueDescription.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
				res = zeCommandQueueCreate(vkGPU->context, vkGPU->device, &commandQueueDescription, &vkGPU->commandQueue);
				if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE;
				free(cmdqueueGroupProperties);
				k++;
			}
			else {
				k++;
			}
		}

		free(deviceList);
	}
	free(drivers);
#endif

	uint64_t isCompilerInitialized = 1;

	switch (sample_id) {
	case 0:
	{
		resFFT = sample_0_benchmark_VkFFT_single(vkGPU, file_output, output, isCompilerInitialized);
		break;
	}
	case 1:
	{
		resFFT = sample_1_benchmark_VkFFT_double(vkGPU, file_output, output, isCompilerInitialized);
		break;
	}
#if ((VKFFT_BACKEND==0)&&(VK_API_VERSION>10))
	case 2:
	{
		resFFT = sample_2_benchmark_VkFFT_half(vkGPU, file_output, output, isCompilerInitialized);
		break;
	}
#endif
	case 3:
	{
		resFFT = sample_3_benchmark_VkFFT_single_3d(vkGPU, file_output, output, isCompilerInitialized);
		break;
	}
	case 4:
	{
		resFFT = sample_4_benchmark_VkFFT_single_3d_zeropadding(vkGPU, file_output, output, isCompilerInitialized);
		break;
	}
	case 5:
	{
		resFFT = sample_5_benchmark_VkFFT_single_disableReorderFourStep(vkGPU, file_output, output, isCompilerInitialized);
		break;
	}
	case 6:
	{
		resFFT = sample_6_benchmark_VkFFT_single_r2c(vkGPU, file_output, output, isCompilerInitialized);
		break;
	}
	case 7:
	{
		resFFT = sample_7_benchmark_VkFFT_single_Bluestein(vkGPU, file_output, output, isCompilerInitialized);
		break;
	}
	case 8:
	{
		resFFT = sample_8_benchmark_VkFFT_double_Bluestein(vkGPU, file_output, output, isCompilerInitialized);
		break;
	}
#if(VKFFT_BACKEND==0)
	case 10:
	{
		resFFT = sample_10_benchmark_VkFFT_single_multipleBuffers(vkGPU, file_output, output, isCompilerInitialized);
		break;
	}
#endif
#ifdef USE_FFTW
	case 11:
	{
		resFFT = sample_11_precision_VkFFT_single(vkGPU, file_output, output, isCompilerInitialized);
		break;
	}
	case 12:
	{
		resFFT = sample_12_precision_VkFFT_double(vkGPU, file_output, output, isCompilerInitialized);
		break;
	}
#if ((VKFFT_BACKEND==0)&&(VK_API_VERSION>10))
	case 13:
	{
		resFFT = sample_13_precision_VkFFT_half(vkGPU, file_output, output, isCompilerInitialized);
		break;
	}
#endif
	case 14:
	{
		resFFT = sample_14_precision_VkFFT_single_nonPow2(vkGPU, file_output, output, isCompilerInitialized);
		break;
	}
	case 15:
	{
		resFFT = sample_15_precision_VkFFT_single_r2c(vkGPU, file_output, output, isCompilerInitialized);
		break;
	}
	case 16:
	{
		resFFT = sample_16_precision_VkFFT_single_dct(vkGPU, file_output, output, isCompilerInitialized);
		break;
	}
	case 17:
	{
		resFFT = sample_17_precision_VkFFT_double_dct(vkGPU, file_output, output, isCompilerInitialized);
		break;
	}
	case 18:
	{
		resFFT = sample_18_precision_VkFFT_double_nonPow2(vkGPU, file_output, output, isCompilerInitialized);
		break;
	}
#endif
	case 50:
	{
		resFFT = sample_50_convolution_VkFFT_single_1d_matrix(vkGPU, file_output, output, isCompilerInitialized);
		break;
	}
	case 51:
	{
		resFFT = sample_51_convolution_VkFFT_single_3d_matrix_zeropadding_r2c(vkGPU, file_output, output, isCompilerInitialized);
		break;
	}
	case 52:
	{
		resFFT = sample_52_convolution_VkFFT_single_2d_batched_r2c(vkGPU, file_output, output, isCompilerInitialized);
		break;
	}
	case 110:
	{
		resFFT = sample_100_benchmark_VkFFT_single_nd_dct(vkGPU, file_output, output, isCompilerInitialized, 1);
		break;
	}
	case 120:
	{
		resFFT = sample_100_benchmark_VkFFT_single_nd_dct(vkGPU, file_output, output, isCompilerInitialized, 2);
		break;
	}
	case 130:
	{
		resFFT = sample_100_benchmark_VkFFT_single_nd_dct(vkGPU, file_output, output, isCompilerInitialized, 3);
		break;
	}
	case 140:
	{
		resFFT = sample_100_benchmark_VkFFT_single_nd_dct(vkGPU, file_output, output, isCompilerInitialized, 4);
		break;
	}
	case 111:
	{
		resFFT = sample_101_benchmark_VkFFT_double_nd_dct(vkGPU, file_output, output, isCompilerInitialized, 1);
		break;
	}
	case 121:
	{
		resFFT = sample_101_benchmark_VkFFT_double_nd_dct(vkGPU, file_output, output, isCompilerInitialized, 2);
		break;
	}
	case 131:
	{
		resFFT = sample_101_benchmark_VkFFT_double_nd_dct(vkGPU, file_output, output, isCompilerInitialized, 3);
		break;
	}
	case 141:
	{
		resFFT = sample_101_benchmark_VkFFT_double_nd_dct(vkGPU, file_output, output, isCompilerInitialized, 4);
		break;
	}
	case 200: case 201:
	{
		resFFT = user_benchmark_VkFFT(vkGPU, file_output, output, isCompilerInitialized, userParams);
		break;
	}
#if ((VKFFT_BACKEND==0)&&(VK_API_VERSION>10))
	case 202:
	{
		resFFT = user_benchmark_VkFFT(vkGPU, file_output, output, isCompilerInitialized, userParams);
		break;
	}
#endif
	case 1000:
	{
		resFFT = sample_1000_VkFFT_single_2_4096(vkGPU, file_output, output, isCompilerInitialized);
		break;
	}
	case 1001:
	{
		resFFT = sample_1001_benchmark_VkFFT_double_2_4096(vkGPU, file_output, output, isCompilerInitialized);
		break;
	}
	case 1003:
	{
		resFFT = sample_1003_benchmark_VkFFT_single_3d_2_512(vkGPU, file_output, output, isCompilerInitialized);
		break;
	}
	}
#if(VKFFT_BACKEND==0)
	vkDestroyFence(vkGPU->device, vkGPU->fence, NULL);
	vkDestroyCommandPool(vkGPU->device, vkGPU->commandPool, NULL);
	vkDestroyDevice(vkGPU->device, NULL);
	DestroyDebugUtilsMessengerEXT(vkGPU, NULL);
	vkDestroyInstance(vkGPU->instance, NULL);
	glslang_finalize_process();//destroy compiler after use
#elif(VKFFT_BACKEND==1)
	res = cuCtxDestroy(vkGPU->context);
#elif(VKFFT_BACKEND==2)
	res = hipCtxDestroy(vkGPU->context);
#elif(VKFFT_BACKEND==3)
	res = clReleaseCommandQueue(vkGPU->commandQueue);
	if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_RELEASE_COMMAND_QUEUE;
	clReleaseContext(vkGPU->context);
#elif(VKFFT_BACKEND==4)
	res = zeCommandQueueDestroy(vkGPU->commandQueue);
	if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_RELEASE_COMMAND_QUEUE;
	res = zeContextDestroy(vkGPU->context);
#endif

	return resFFT;
}

bool findFlag(char** start, char** end, const std::string& flag) {
	return (std::find(start, end, flag) != end);
}
char* getFlagValue(char** start, char** end, const std::string& flag)
{
	char** value = std::find(start, end, flag);
	value++;
	if (value != end)
	{
		return *value;
	}
	return 0;
}
int main(int argc, char* argv[])
{
	VkGPU vkGPU = {};
#if(VKFFT_BACKEND==0)
	vkGPU.enableValidationLayers = 0;
#endif
	bool file_output = false;
	FILE* output = NULL;
	int sscanf_res = 0;
	if (findFlag(argv, argv + argc, "-h"))
	{
		//print help
		int version = VkFFTGetVersion();
		int version_decomposed[3];
		version_decomposed[0] = version / 10000;
		version_decomposed[1] = (version - version_decomposed[0] * 10000) / 100;
		version_decomposed[2] = (version - version_decomposed[0] * 10000 - version_decomposed[1] * 100);
		printf("VkFFT v%d.%d.%d (11-04-2022). Author: Tolmachev Dmitrii\n", version_decomposed[0], version_decomposed[1], version_decomposed[2]);
#if (VKFFT_BACKEND==0)
		printf("Vulkan backend\n");
#elif (VKFFT_BACKEND==1)
		printf("CUDA backend\n");
#elif (VKFFT_BACKEND==2)
		printf("HIP backend\n");
#elif (VKFFT_BACKEND==3)
		printf("OpenCL backend\n");
#elif (VKFFT_BACKEND==4)
		printf("Level Zero backend\n");
#endif
		printf("	-h: print help\n");
		printf("	-devices: print the list of available device ids, used as -d argument\n");
		printf("	-d X: select device (default 0)\n");
		printf("	-o NAME: specify output file path\n");
		printf("	-vkfft X: launch VkFFT sample X:\n");
		printf("		0 - FFT + iFFT C2C benchmark 1D batched in single precision\n");
		printf("		1 - FFT + iFFT C2C benchmark 1D batched in double precision LUT\n");
#if ((VKFFT_BACKEND==0)&&(VK_API_VERSION>10))
		printf("		2 - FFT + iFFT C2C benchmark 1D batched in half precision\n");
#endif
		printf("		3 - FFT + iFFT C2C multidimensional benchmark in single precision\n");
		printf("		4 - FFT + iFFT C2C multidimensional benchmark in single precision, native zeropadding\n");
		printf("		5 - FFT + iFFT C2C benchmark 1D batched in single precision, no reshuffling\n");
		printf("		6 - FFT + iFFT R2C / C2R benchmark\n");
		printf("		7 - FFT + iFFT C2C Bluestein benchmark in single precision\n");
		printf("		8 - FFT + iFFT C2C Bluestein benchmark in double precision\n");
#if (VKFFT_BACKEND==0)
		printf("		10 - multiple buffer(4 by default) split version of benchmark 0\n");
#endif
#ifdef USE_FFTW
#ifdef USE_cuFFT
		printf("		11 - VkFFT / cuFFT / FFTW C2C precision test in single precision\n");
		printf("		12 - VkFFT / cuFFT / FFTW C2C precision test in double precision\n");
#if ((VKFFT_BACKEND==0)&&(VK_API_VERSION>10))
		printf("		13 - VkFFT / cuFFT / FFTW C2C precision test in half precision\n");
#endif
		printf("		14 - VkFFT / FFTW C2C radix 3 / 5 / 7 / 11 / 13 / Bluestein precision test in single precision\n");
		printf("		15 - VkFFT / cuFFT / FFTW R2C+C2R precision test in single precision\n");
		printf("		16 - VkFFT / FFTW R2R DCT-I, II, III and IV precision test in single precision\n");
		printf("		17 - VkFFT / FFTW R2R DCT-I, II, III and IV precision test in double precision\n");
		printf("		18 - VkFFT / FFTW C2C radix 3 / 5 / 7 / 11 / 13 / Bluestein precision test in double precision\n");
#elif USE_rocFFT
		printf("		11 - VkFFT / rocFFT / FFTW C2C precision test in single precision\n");
		printf("		12 - VkFFT / rocFFT / FFTW C2C precision test in double precision\n");
#if ((VKFFT_BACKEND==0)&&(VK_API_VERSION>10))
		printf("		13 - VkFFT / FFTW C2C precision test in half precision\n");
#endif
		printf("		14 - VkFFT / FFTW C2C radix 3 / 5 / 7 / 11 / 13 / Bluestein precision test in single precision\n");
		printf("		15 - VkFFT / rocFFT / FFTW R2C+C2R precision test in single precision\n");
		printf("		16 - VkFFT / FFTW R2R DCT-I, II, III and IV precision test in single precision\n");
		printf("		17 - VkFFT / FFTW R2R DCT-I, II, III and IV precision test in double precision\n");
		printf("		18 - VkFFT / FFTW C2C radix 3 / 5 / 7 / 11 / 13 / Bluestein precision test in double precision\n");
#else
		printf("		11 - VkFFT / FFTW C2C precision test in single precision\n");
		printf("		12 - VkFFT / FFTW C2C precision test in double precision\n");
#if ((VKFFT_BACKEND==0)&&(VK_API_VERSION>10))
		printf("		13 - VkFFT / FFTW C2C precision test in half precision\n");
#endif
		printf("		14 - VkFFT / FFTW C2C radix 3 / 5 / 7 / 11 / 13 / Bluestein precision test in single precision\n");
		printf("		15 - VkFFT / FFTW R2C+C2R precision test in single precision\n");
		printf("		16 - VkFFT / FFTW R2R DCT-I, II, III and IV precision test in single precision\n");
		printf("		17 - VkFFT / FFTW R2R DCT-I, II, III and IV precision test in double precision\n");
		printf("		18 - VkFFT / FFTW C2C radix 3 / 5 / 7 / 11 / 13 / Bluestein precision test in double precision\n");
#endif
#endif
		printf("		50 - convolution example with identity kernel\n");
		printf("		51 - zeropadding convolution example with identity kernel\n");
		printf("		52 - batched convolution example with identity kernel\n");
		printf("		110 - VkFFT FFT + iFFT R2R DCT-1 multidimensional benchmark in single precision\n");
		printf("		111 - VkFFT FFT + iFFT R2R DCT-1 multidimensional benchmark in double precision\n");
		printf("		120 - VkFFT FFT + iFFT R2R DCT-2 multidimensional benchmark in single precision\n");
		printf("		121 - VkFFT FFT + iFFT R2R DCT-2 multidimensional benchmark in double precision\n");
		printf("		130 - VkFFT FFT + iFFT R2R DCT-3 multidimensional benchmark in single precision\n");
		printf("		131 - VkFFT FFT + iFFT R2R DCT-3 multidimensional benchmark in double precision\n");
		printf("		140 - VkFFT FFT + iFFT R2R DCT-4 multidimensional benchmark in single precision\n");
		printf("		141 - VkFFT FFT + iFFT R2R DCT-4 multidimensional benchmark in double precision\n");

		printf("		1000 - FFT + iFFT C2C benchmark 1D batched in single precision: all supported systems from 2 to 4096\n");
		printf("		1001 - FFT + iFFT C2C benchmark 1D batched in double precision: all supported systems from 2 to 4096\n");
		printf("		1003 - FFT + iFFT C2C multidimensional benchmark in single precision: all supported cubes from 2 to 512\n");
		printf("	-benchmark_vkfft: run VkFFT benchmark on a user-defined system:\n\
		-X uint, -Y uint, -Z uint - FFT dimensions (default Y and Z are 1)\n");
#if ((VKFFT_BACKEND==0)&&(VK_API_VERSION>10))
		printf("\
		-P uint - precision (0 - single, 1 - double, 2 - half) (default 0)\n");
#else
		printf("\
		-P uint - precision (0 - single, 1 - double) (default 0)\n");
#endif
		printf("\
		-B uint - number of batched systems (default 1)\n\
		-N uint - number of consecutive FFT+iFFT iterations (default 1)\n\
		-R2C uint - use R2C (0 - off, 1 - on) (default 0)\n\
		-DCT uint - perform DCT (0 - off, else type: 1, 2, 3 or 4) (default 0)\n\
		-save - save generated binaries\n\
		-load - load previously generated binaries\n");
#ifdef USE_cuFFT
		printf("	-cufft X: launch cuFFT sample X:\n");
		printf("		0 - FFT + iFFT C2C benchmark 1D batched in single precision\n");
		printf("		1 - FFT + iFFT C2C benchmark 1D batched in double precision LUT\n");
		printf("		2 - FFT + iFFT C2C benchmark 1D batched in half precision\n");
		printf("		3 - FFT + iFFT C2C multidimensional benchmark in single precision\n");
		printf("		6 - FFT + iFFT R2C / C2R benchmark\n");
		printf("		7 - FFT + iFFT C2C big prime benchmark in single precision (similar to VkFFT Bluestein)\n");
		printf("		8 - FFT + iFFT C2C big prime benchmark in double precision (similar to VkFFT Bluestein)\n");
		printf("		1000 - FFT + iFFT C2C benchmark 1D batched in single precision: all supported systems from 2 to 4096\n");
		printf("		1001 - FFT + iFFT C2C benchmark 1D batched in double precision: all supported systems from 2 to 4096\n");
		printf("		1003 - FFT + iFFT C2C multidimensional benchmark in single precision: all supported cubes from 2 to 512\n");
		printf("	-test: (or no -vkfft and -cufft keys) run vkfft benchmarks 0-6 and cufft benchmarks 0-6\n");
		printf("	-benchmark_cufft: run cuFFT benchmark on a user-defined system:\n\
		-X uint, -Y uint, -Z uint - FFT dimensions (default Y and Z are 1)\n\
		-P uint - precision (0 - single, 1 - double) (default 0)\n\
		-B uint - number of batched systems (default 1)\n\
		-N uint - number of consecutive FFT+iFFT iterations (default 1)\n\
		-R2C uint - use R2C (0 - off, 1 - on) (default 0)\n");
#elif USE_rocFFT
		printf("	-rocfft X: launch rocFFT sample X:\n");
		printf("		0 - FFT + iFFT C2C benchmark 1D batched in single precision\n");
		printf("		1 - FFT + iFFT C2C benchmark 1D batched in double precision LUT\n");
		printf("		3 - FFT + iFFT C2C multidimensional benchmark in single precision\n");
		printf("		6 - FFT + iFFT R2C / C2R benchmark\n");
		printf("		7 - FFT + iFFT C2C big prime benchmark in single precision (similar to VkFFT Bluestein)\n");
		printf("		8 - FFT + iFFT C2C big prime benchmark in double precision (similar to VkFFT Bluestein)\n");
		printf("		1000 - FFT + iFFT C2C benchmark 1D batched in single precision: all supported systems from 2 to 4096\n");
		printf("		1001 - FFT + iFFT C2C benchmark 1D batched in double precision: all supported systems from 2 to 4096\n");
		printf("		1003 - FFT + iFFT C2C multidimensional benchmark in single precision: all supported cubes from 2 to 512\n");
		printf("	-test: (or no -vkfft and -rocfft keys) run vkfft benchmarks 0-6 and rocfft benchmarks 0-6\n");
		printf("	-benchmark_rocfft: run rocFFT benchmark on a user-defined system:\n\
		-X uint, -Y uint, -Z uint - FFT dimensions (default Y and Z are 1)\n\
		-P uint - precision (0 - single, 1 - double) (default 0)\n\
		-B uint - number of batched systems (default 1)\n\
		-N uint - number of consecutive FFT+iFFT iterations (default 1)\n\
		-R2C uint - use R2C (0 - off, 1 - on) (default 0)\n");
#else
		printf("	-test: run vkfft benchmarks 0-6\n");
		printf("	-cufft command is disabled\n");
		printf("	-rocfft command is disabled\n");
#endif
		return 0;
	}
	if (findFlag(argv, argv + argc, "-devices"))
	{
		//print device list
		VkFFTResult resFFT = devices_list();
		return resFFT;
	}
	if (findFlag(argv, argv + argc, "-d"))
	{
		//select device_id
		char* value = getFlagValue(argv, argv + argc, "-d");
		if (value != 0) {
			sscanf_res = sscanf(value, "%" PRIu64 "", &vkGPU.device_id);
			if (sscanf_res <= 0) {
				printf("sscanf failed\n");
				return 1;
			}
		}
		else {
			printf("No device is selected with -d flag\n");
			return 1;
		}
	}
	if (findFlag(argv, argv + argc, "-o"))
	{
		//specify output file
		char* value = getFlagValue(argv, argv + argc, "-o");
		if (value != 0) {
			file_output = true;
			output = fopen(value, "a");
		}
		else {
			printf("No output file is selected with -o flag\n");
			return 1;
		}
	}
	if (findFlag(argv, argv + argc, "-benchmark_vkfft") || findFlag(argv, argv + argc, "-benchmark_cufft") || findFlag(argv, argv + argc, "-benchmark_rocfft"))
	{
		//select sample_id
		VkFFTUserSystemParameters userParams = {};
		userParams.X = 1;
		userParams.Y = 1;
		userParams.Z = 1;
		userParams.P = 0;
		userParams.B = 1;
		userParams.N = 1;
		userParams.R2C = 0;
		userParams.DCT = 0;
		if (findFlag(argv, argv + argc, "-X"))
		{
			char* value = getFlagValue(argv, argv + argc, "-X");
			if (value != 0) {
				sscanf_res = sscanf(value, "%" PRIu64 "", &userParams.X);
				if (sscanf_res <= 0) {
					printf("sscanf failed\n");
					return 1;
				}
			}
			else {
				printf("No dimension is selected with -X flag\n");
				return 1;
			}
		}
		else {
			printf("No -X flag is selected\n");
			return 1;
		}
		if (findFlag(argv, argv + argc, "-Y"))
		{
			char* value = getFlagValue(argv, argv + argc, "-Y");
			if (value != 0) {
				sscanf_res = sscanf(value, "%" PRIu64 "", &userParams.Y);
				if (sscanf_res <= 0) {
					printf("sscanf failed\n");
					return 1;
				}
			}
			else {
				printf("No dimension is selected with -Y flag\n");
				return 1;
			}
		}
		if (findFlag(argv, argv + argc, "-Z"))
		{
			char* value = getFlagValue(argv, argv + argc, "-Z");
			if (value != 0) {
				sscanf_res = sscanf(value, "%" PRIu64 "", &userParams.Z);
				if (sscanf_res <= 0) {
					printf("sscanf failed\n");
					return 1;
				}
			}
			else {
				printf("No dimension is selected with -Z flag\n");
				return 1;
			}
		}
		if (findFlag(argv, argv + argc, "-P"))
		{
			char* value = getFlagValue(argv, argv + argc, "-P");
			if (value != 0) {
				sscanf_res = sscanf(value, "%" PRIu64 "", &userParams.P);
				if (sscanf_res <= 0) {
					printf("sscanf failed\n");
					return 1;
				}
			}
			else {
				printf("No precision is selected with -P flag\n");
				return 1;
			}
		}
		if (findFlag(argv, argv + argc, "-B"))
		{
			char* value = getFlagValue(argv, argv + argc, "-B");
			if (value != 0) {
				sscanf_res = sscanf(value, "%" PRIu64 "", &userParams.B);
				if (sscanf_res <= 0) {
					printf("sscanf failed\n");
					return 1;
				}
			}
			else {
				printf("No batch is selected with -B flag\n");
				return 1;
			}
		}
		if (findFlag(argv, argv + argc, "-N"))
		{
			char* value = getFlagValue(argv, argv + argc, "-N");
			if (value != 0) {
				sscanf_res = sscanf(value, "%" PRIu64 "", &userParams.N);
				if (sscanf_res <= 0) {
					printf("sscanf failed\n");
					return 1;
				}
			}
			else {
				printf("No number of iterations is selected with -N flag\n");
				return 1;
			}
		}
		if (findFlag(argv, argv + argc, "-R2C"))
		{
			char* value = getFlagValue(argv, argv + argc, "-R2C");
			if (value != 0) {
				sscanf_res = sscanf(value, "%" PRIu64 "", &userParams.R2C);
				if (sscanf_res <= 0) {
					printf("sscanf failed\n");
					return 1;
				}
			}
			else {
				printf("No R2C parameter is selected with -R2C flag\n");
				return 1;
			}
		}
		if (findFlag(argv, argv + argc, "-DCT"))
		{
			char* value = getFlagValue(argv, argv + argc, "-DCT");
			if (value != 0) {
				sscanf_res = sscanf(value, "%" PRIu64 "", &userParams.DCT);
				if (sscanf_res <= 0) {
					printf("sscanf failed\n");
					return 1;
				}
			}
			else {
				printf("No DCT parameter is selected with -DCT flag\n");
				return 1;
			}
		}
		if (findFlag(argv, argv + argc, "-save"))
		{
			userParams.saveApplicationToString = 1;
		}
		if (findFlag(argv, argv + argc, "-load"))
		{
			userParams.loadApplicationFromString = 1;
		}
		if (findFlag(argv, argv + argc, "-benchmark_vkfft")) {
			VkFFTResult resFFT = launchVkFFT(&vkGPU, 200 + userParams.P, file_output, output, &userParams);
			if (resFFT != VKFFT_SUCCESS) return resFFT;
		}
		else {
#ifdef USE_cuFFT
			if (findFlag(argv, argv + argc, "-benchmark_cufft")) {
				user_benchmark_cuFFT(file_output, output, (cuFFTUserSystemParameters*)(&userParams));
			}
			return 0;
#elif USE_rocFFT
			if (findFlag(argv, argv + argc, "-benchmark_rocfft")) {
				user_benchmark_rocFFT(file_output, output, (rocFFTUserSystemParameters*)(&userParams));
			}
			return 0;
#endif
			return 1;
		}
		return 0;
	}

	if (findFlag(argv, argv + argc, "-vkfft"))
	{
		//select sample_id
		char* value = getFlagValue(argv, argv + argc, "-vkfft");
		if (value != 0) {
			uint64_t sample_id = 0;
			sscanf_res = sscanf(value, "%" PRIu64 "", &sample_id);
			if (sscanf_res <= 0) {
				printf("sscanf failed\n");
				return 1;
			}
			VkFFTResult resFFT = launchVkFFT(&vkGPU, sample_id, file_output, output, 0);
			if (resFFT != VKFFT_SUCCESS) return resFFT;
		}
		else {
			printf("No sample is selected with -vkfft flag\n");
			return 1;
		}
	}
#ifdef USE_cuFFT
	if (findFlag(argv, argv + argc, "-cufft"))
	{
		//select sample_id
		char* value = getFlagValue(argv, argv + argc, "-cufft");
		if (value != 0) {
			uint64_t sample_id = 0;
			sscanf_res = sscanf(value, "%" PRIu64 "", &sample_id);
			if (sscanf_res <= 0) {
				printf("sscanf failed\n");
				return 1;
			}
			switch (sample_id) {
			case 0:
				sample_0_benchmark_cuFFT_single(file_output, output);
				break;
			case 1:
				sample_1_benchmark_cuFFT_double(file_output, output);
				break;
			case 2:
				sample_2_benchmark_cuFFT_half(file_output, output);
				break;
			case 3:
				sample_3_benchmark_cuFFT_single_3d(file_output, output);
				break;
			case 6:
				sample_6_benchmark_cuFFT_single_r2c(file_output, output);
				break;
			case 7:
				sample_7_benchmark_cuFFT_single_Bluestein(file_output, output);
				break;
			case 8:
				sample_8_benchmark_cuFFT_double_Bluestein(file_output, output);
				break;
			case 1000:
				sample_1000_benchmark_cuFFT_single_2_4096(file_output, output);
				break;
			case 1001:
				sample_1001_benchmark_cuFFT_double_2_4096(file_output, output);
				break;
			case 1003:
				sample_1003_benchmark_cuFFT_single_3d_2_512(file_output, output);
				break;
			}
		}
		else {
			printf("No cuFFT script is selected with -cufft flag\n");
			return 1;
		}
	}
#elif USE_rocFFT
	if (findFlag(argv, argv + argc, "-rocfft"))
	{
		//select sample_id
		char* value = getFlagValue(argv, argv + argc, "-rocfft");
		if (value != 0) {
			uint64_t sample_id = 0;
			sscanf_res = sscanf(value, "%" PRIu64 "", &sample_id);
			if (sscanf_res <= 0) {
				printf("sscanf failed\n");
				return 1;
			}
			switch (sample_id) {
			case 0:
				sample_0_benchmark_rocFFT_single(file_output, output);
				break;
			case 1:
				sample_1_benchmark_rocFFT_double(file_output, output);
				break;
			case 3:
				sample_3_benchmark_rocFFT_single_3d(file_output, output);
				break;
			case 6:
				sample_6_benchmark_rocFFT_single_r2c(file_output, output);
				break;
			case 7:
				sample_7_benchmark_rocFFT_single_Bluestein(file_output, output);
				break;
			case 8:
				sample_8_benchmark_rocFFT_double_Bluestein(file_output, output);
				break;
			case 1000:
				sample_1000_benchmark_rocFFT_single_2_4096(file_output, output);
				break;
			case 1001:
				sample_1001_benchmark_rocFFT_double_2_4096(file_output, output);
				break;
			case 1003:
				sample_1003_benchmark_rocFFT_single_3d_2_512(file_output, output);
				break;
			}
		}
		else {
			printf("No rocFFT script is selected with -rocfft flag\n");
			return 1;
		}
	}
#endif
	if ((findFlag(argv, argv + argc, "-test")) || ((!findFlag(argv, argv + argc, "-cufft")) && (!findFlag(argv, argv + argc, "-rocfft")) && (!findFlag(argv, argv + argc, "-vkfft"))))
	{
		if (output == NULL) {
			file_output = true;
			output = fopen("result.txt", "a");
		}
		for (uint64_t i = 0; i < 9; i++) {
#if((VKFFT_BACKEND>0) || (VK_API_VERSION == 10))
			if (i == 2) i++;
#endif
			VkFFTResult resFFT = launchVkFFT(&vkGPU, i, file_output, output, 0);
			if (resFFT != VKFFT_SUCCESS) return resFFT;
		}
#ifdef USE_cuFFT
		sample_0_benchmark_cuFFT_single(file_output, output);
		sample_1_benchmark_cuFFT_double(file_output, output);
		sample_2_benchmark_cuFFT_half(file_output, output);
		sample_3_benchmark_cuFFT_single_3d(file_output, output);
		sample_6_benchmark_cuFFT_single_r2c(file_output, output);
		sample_7_benchmark_cuFFT_single_Bluestein(file_output, output);
		sample_8_benchmark_cuFFT_double_Bluestein(file_output, output);
#elif USE_rocFFT
		sample_0_benchmark_rocFFT_single(file_output, output);
		sample_1_benchmark_rocFFT_double(file_output, output);
		sample_3_benchmark_rocFFT_single_3d(file_output, output);
		sample_6_benchmark_rocFFT_single_r2c(file_output, output);
		sample_7_benchmark_rocFFT_single_Bluestein(file_output, output);
		sample_8_benchmark_rocFFT_double_Bluestein(file_output, output);
#endif
	}
	return 0;
}
