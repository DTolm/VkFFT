#include <vector>
#include <memory>
#include <string.h>
#include <chrono>
#include <thread>
#include <iostream>
#include <algorithm>
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
#define __HIP_PLATFORM_HCC__
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_complex.h>
#endif
#include "vkFFT.h"
#include "half.hpp"
#ifdef USE_cuFFT
#include "benchmark_cuFFT.h"
#include "benchmark_cuFFT_2_4096.h"
#include "benchmark_cuFFT_r2c.h"
#include "benchmark_cuFFT_double.h"
#include "benchmark_cuFFT_double_2_4096.h"
#include "benchmark_cuFFT_half.h"
#include "benchmark_cuFFT_3d.h"
#include "benchmark_cuFFT_3d_2_512.h"
#include "precision_cuFFT.h"
#include "precision_cuFFT_r2c.h"
#include "precision_cuFFT_double.h"
#include "precision_cuFFT_half.h"
#endif  
#ifdef USE_rocFFT
#include "benchmark_rocFFT.h"
#include "benchmark_rocFFT_2_4096.h"
#include "benchmark_rocFFT_r2c.h"
#include "benchmark_rocFFT_double.h"
#include "benchmark_rocFFT_double_2_4096.h"
#include "benchmark_rocFFT_3d.h"
#include "benchmark_rocFFT_3d_2_512.h"
#include "precision_rocFFT.h"
#include "precision_rocFFT_r2c.h"
#include "precision_rocFFT_double.h"
#endif 
#ifdef USE_FFTW
#include "fftw3.h"
#endif

using half_float::half;

typedef half half2[2];

const bool enableValidationLayers = false;

typedef struct {
#if(VKFFT_BACKEND==0)
	VkInstance instance;//a connection between the application and the Vulkan library 
	VkPhysicalDevice physicalDevice;//a handle for the graphics card used in the application
	VkPhysicalDeviceProperties physicalDeviceProperties;//bastic device properties
	VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;//bastic memory properties of the device
	VkDevice device;//a logical device, interacting with physical device
	VkDebugUtilsMessengerEXT debugMessenger;//extension for debugging
	uint32_t queueFamilyIndex;//if multiple queues are available, specify the used one
	VkQueue queue;//a place, where all operations are submitted
	VkCommandPool commandPool;//an opaque objects that command buffer memory is allocated from
	VkFence fence;//a vkGPU->fence used to synchronize dispatches
	std::vector<const char*> enabledDeviceExtensions;
#elif(VKFFT_BACKEND==1)
	CUdevice device;
	CUcontext context;
#elif(VKFFT_BACKEND==2)
	hipDevice_t device;
	hipCtx_t context;
#endif
	uint32_t device_id;//an id of a device, reported by Vulkan device list
} VkGPU;//an example structure containing Vulkan primitives
#if(VKFFT_BACKEND==0)
const char validationLayers[28] = "VK_LAYER_KHRONOS_validation";
/*static VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallbackFn(
	VkDebugReportFlagsEXT                       flags,
	VkDebugReportObjectTypeEXT                  objectType,
	uint64_t                                    object,
	size_t                                      location,
	int32_t                                     messageCode,
	const char* pLayerPrefix,
	const char* pMessage,
	void* pUserData) {

	printf("Debug Report: %s: %s\n", pLayerPrefix, pMessage);

	return VK_FALSE;
}*/

VkResult CreateDebugUtilsMessengerEXT(VkGPU* vkGPU, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
	//pointer to the function, as it is not part of the core. Function creates debugging messenger
	PFN_vkCreateDebugUtilsMessengerEXT func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(vkGPU->instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != NULL) {
		return func(vkGPU->instance, pCreateInfo, pAllocator, pDebugMessenger);
	}
	else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}
void DestroyDebugUtilsMessengerEXT(VkGPU* vkGPU, const VkAllocationCallbacks* pAllocator) {
	//pointer to the function, as it is not part of the core. Function destroys debugging messenger
	PFN_vkDestroyDebugUtilsMessengerEXT func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(vkGPU->instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != NULL) {
		func(vkGPU->instance, vkGPU->debugMessenger, pAllocator);
	}
}
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
	printf("validation layer: %s\n", pCallbackData->pMessage);
	return VK_FALSE;
}


VkResult setupDebugMessenger(VkGPU* vkGPU) {
	//function that sets up the debugging messenger 
	if (enableValidationLayers == 0) return VK_SUCCESS;

	VkDebugUtilsMessengerCreateInfoEXT createInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
	createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
	createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
	createInfo.pfnUserCallback = debugCallback;

	if (CreateDebugUtilsMessengerEXT(vkGPU, &createInfo, NULL, &vkGPU->debugMessenger) != VK_SUCCESS) {
		return VK_ERROR_INITIALIZATION_FAILED;
	}
	return VK_SUCCESS;
}
VkResult checkValidationLayerSupport() {
	//check if validation layers are supported when an instance is created
	uint32_t layerCount;
	vkEnumerateInstanceLayerProperties(&layerCount, NULL);

	VkLayerProperties* availableLayers = (VkLayerProperties*)malloc(sizeof(VkLayerProperties) * layerCount);
	vkEnumerateInstanceLayerProperties(&layerCount, availableLayers);

	for (uint32_t i = 0; i < layerCount; i++) {
		if (strcmp("VK_LAYER_KHRONOS_validation", availableLayers[i].layerName) == 0) {
			free(availableLayers);
			return VK_SUCCESS;
		}
	}
	free(availableLayers);
	return VK_ERROR_LAYER_NOT_PRESENT;
}

std::vector<const char*> getRequiredExtensions(uint32_t sample_id) {
	std::vector<const char*> extensions;

	if (enableValidationLayers) {
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}
	switch (sample_id) {
#if (VK_API_VERSION>10)
	case 2:
		extensions.push_back("VK_KHR_get_physical_device_properties2");
		break;
#endif
	default:
		break;
	}


	return extensions;
}

VkResult createInstance(VkGPU* vkGPU, uint32_t sample_id) {
	//create instance - a connection between the application and the Vulkan library 
	VkResult res = VK_SUCCESS;
	//check if validation layers are supported
	if (enableValidationLayers == 1) {
		res = checkValidationLayerSupport();
		if (res != VK_SUCCESS) return res;
	}

	VkApplicationInfo applicationInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
	applicationInfo.pApplicationName = "VkFFT";
	applicationInfo.applicationVersion = 1.0;
	applicationInfo.pEngineName = "VkFFT";
	applicationInfo.engineVersion = 1.0;
#if (VK_API_VERSION>=12)
	applicationInfo.apiVersion = VK_API_VERSION_1_2;
#elif (VK_API_VERSION==11)
	applicationInfo.apiVersion = VK_API_VERSION_1_1;
#else
	applicationInfo.apiVersion = VK_API_VERSION_1_0;
#endif

	VkInstanceCreateInfo createInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
	createInfo.flags = 0;
	createInfo.pApplicationInfo = &applicationInfo;

	auto extensions = getRequiredExtensions(sample_id);
	createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
	createInfo.ppEnabledExtensionNames = extensions.data();

	VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
	if (enableValidationLayers) {
		//query for the validation layer support in the instance
		createInfo.enabledLayerCount = 1;
		const char* validationLayers = "VK_LAYER_KHRONOS_validation";
		createInfo.ppEnabledLayerNames = &validationLayers;
		debugCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		debugCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		debugCreateInfo.pfnUserCallback = debugCallback;
		createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
	}
	else {
		createInfo.enabledLayerCount = 0;

		createInfo.pNext = nullptr;
	}

	res = vkCreateInstance(&createInfo, NULL, &vkGPU->instance);
	if (res != VK_SUCCESS) return res;

	return res;
}

VkResult findPhysicalDevice(VkGPU* vkGPU) {
	//check if there are GPUs that support Vulkan and select one
	VkResult res = VK_SUCCESS;
	uint32_t deviceCount;
	res = vkEnumeratePhysicalDevices(vkGPU->instance, &deviceCount, NULL);
	if (res != VK_SUCCESS) return res;
	if (deviceCount == 0) {
		return VK_ERROR_DEVICE_LOST;
	}

	VkPhysicalDevice* devices = (VkPhysicalDevice*)malloc(sizeof(VkPhysicalDevice) * deviceCount);
	res = vkEnumeratePhysicalDevices(vkGPU->instance, &deviceCount, devices);
	if (res != VK_SUCCESS) return res;
	vkGPU->physicalDevice = devices[vkGPU->device_id];
	free(devices);
	return VK_SUCCESS;
}
VkResult devices_list() {
	//this function creates an instance and prints the list of available devices
	VkResult res = VK_SUCCESS;
	VkInstance local_instance = { 0 };
	VkInstanceCreateInfo createInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
	createInfo.flags = 0;
	createInfo.pApplicationInfo = NULL;
	VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
	createInfo.enabledLayerCount = 0;
	createInfo.enabledExtensionCount = 0;
	createInfo.pNext = NULL;
	res = vkCreateInstance(&createInfo, NULL, &local_instance);
	if (res != VK_SUCCESS) return res;

	uint32_t deviceCount;
	res = vkEnumeratePhysicalDevices(local_instance, &deviceCount, NULL);
	if (res != VK_SUCCESS) return res;

	VkPhysicalDevice* devices = (VkPhysicalDevice*)malloc(sizeof(VkPhysicalDevice) * deviceCount);
	res = vkEnumeratePhysicalDevices(local_instance, &deviceCount, devices);
	if (res != VK_SUCCESS) return res;
	for (uint32_t i = 0; i < deviceCount; i++) {
		VkPhysicalDeviceProperties device_properties;
		vkGetPhysicalDeviceProperties(devices[i], &device_properties);
		printf("Device id: %d name: %s API:%d.%d.%d\n", i, device_properties.deviceName, (device_properties.apiVersion >> 22), ((device_properties.apiVersion >> 12) & 0x3ff), (device_properties.apiVersion & 0xfff));
	}
	free(devices);
	vkDestroyInstance(local_instance, NULL);
	return res;
}
VkResult getComputeQueueFamilyIndex(VkGPU* vkGPU) {
	//find a queue family for a selected GPU, select the first available for use
	uint32_t queueFamilyCount;
	vkGetPhysicalDeviceQueueFamilyProperties(vkGPU->physicalDevice, &queueFamilyCount, NULL);

	VkQueueFamilyProperties* queueFamilies = (VkQueueFamilyProperties*)malloc(sizeof(VkQueueFamilyProperties) * queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(vkGPU->physicalDevice, &queueFamilyCount, queueFamilies);
	uint32_t i = 0;
	for (; i < queueFamilyCount; i++) {
		VkQueueFamilyProperties props = queueFamilies[i];

		if (props.queueCount > 0 && (props.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
			break;
		}
	}
	free(queueFamilies);
	if (i == queueFamilyCount) {
		return VK_ERROR_INITIALIZATION_FAILED;
	}
	vkGPU->queueFamilyIndex = i;
	return VK_SUCCESS;
}

VkResult createDevice(VkGPU* vkGPU, uint32_t sample_id) {
	//create logical device representation
	VkResult res = VK_SUCCESS;
	VkDeviceQueueCreateInfo queueCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
	res = getComputeQueueFamilyIndex(vkGPU);
	if (res != VK_SUCCESS) return res;
	queueCreateInfo.queueFamilyIndex = vkGPU->queueFamilyIndex;
	queueCreateInfo.queueCount = 1;
	float queuePriorities = 1.0;
	queueCreateInfo.pQueuePriorities = &queuePriorities;
	VkDeviceCreateInfo deviceCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
	VkPhysicalDeviceFeatures deviceFeatures = {};
	switch (sample_id) {
	case 1: {
		deviceFeatures.shaderFloat64 = true;
		deviceCreateInfo.enabledExtensionCount = vkGPU->enabledDeviceExtensions.size();
		deviceCreateInfo.ppEnabledExtensionNames = vkGPU->enabledDeviceExtensions.data();
		deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
		deviceCreateInfo.queueCreateInfoCount = 1;
		deviceCreateInfo.pEnabledFeatures = &deviceFeatures;
		res = vkCreateDevice(vkGPU->physicalDevice, &deviceCreateInfo, NULL, &vkGPU->device);
		if (res != VK_SUCCESS) return res;
		vkGetDeviceQueue(vkGPU->device, vkGPU->queueFamilyIndex, 0, &vkGPU->queue);
		break;
	}
#if (VK_API_VERSION>10)
	case 2: {
		VkPhysicalDeviceFeatures2 deviceFeatures2 = {};
		VkPhysicalDevice16BitStorageFeatures shaderFloat16 = {};
		shaderFloat16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
		shaderFloat16.storageBuffer16BitAccess = true;
		/*VkPhysicalDeviceShaderFloat16Int8Features shaderFloat16 = {};
		shaderFloat16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
		shaderFloat16.shaderFloat16 = true;
		shaderFloat16.shaderInt8 = true;*/
		deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
		deviceFeatures2.pNext = &shaderFloat16;
		deviceFeatures2.features = deviceFeatures;
		vkGetPhysicalDeviceFeatures2(vkGPU->physicalDevice, &deviceFeatures2);
		deviceCreateInfo.pNext = &deviceFeatures2;
		vkGPU->enabledDeviceExtensions.push_back("VK_KHR_16bit_storage");
		deviceCreateInfo.enabledExtensionCount = vkGPU->enabledDeviceExtensions.size();
		deviceCreateInfo.ppEnabledExtensionNames = vkGPU->enabledDeviceExtensions.data();
		deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
		deviceCreateInfo.queueCreateInfoCount = 1;
		deviceCreateInfo.pEnabledFeatures = NULL;
		res = vkCreateDevice(vkGPU->physicalDevice, &deviceCreateInfo, NULL, &vkGPU->device);
		if (res != VK_SUCCESS) return res;
		vkGetDeviceQueue(vkGPU->device, vkGPU->queueFamilyIndex, 0, &vkGPU->queue);
		break;
	}
#endif
	default: {
		deviceCreateInfo.enabledExtensionCount = vkGPU->enabledDeviceExtensions.size();
		deviceCreateInfo.ppEnabledExtensionNames = vkGPU->enabledDeviceExtensions.data();
		deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
		deviceCreateInfo.queueCreateInfoCount = 1;
		deviceCreateInfo.pEnabledFeatures = NULL;
		deviceCreateInfo.pEnabledFeatures = &deviceFeatures;
		res = vkCreateDevice(vkGPU->physicalDevice, &deviceCreateInfo, NULL, &vkGPU->device);
		if (res != VK_SUCCESS) return res;
		vkGetDeviceQueue(vkGPU->device, vkGPU->queueFamilyIndex, 0, &vkGPU->queue);
		break;
	}
	}
	return res;
}
VkResult createFence(VkGPU* vkGPU) {
	//create fence for synchronization 
	VkResult res = VK_SUCCESS;
	VkFenceCreateInfo fenceCreateInfo = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
	fenceCreateInfo.flags = 0;
	res = vkCreateFence(vkGPU->device, &fenceCreateInfo, NULL, &vkGPU->fence);
	return res;
}
VkResult createCommandPool(VkGPU* vkGPU) {
	//create a place, command buffer memory is allocated from
	VkResult res = VK_SUCCESS;
	VkCommandPoolCreateInfo commandPoolCreateInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
	commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	commandPoolCreateInfo.queueFamilyIndex = vkGPU->queueFamilyIndex;
	res = vkCreateCommandPool(vkGPU->device, &commandPoolCreateInfo, NULL, &vkGPU->commandPool);
	return res;
}

VkResult findMemoryType(VkGPU* vkGPU, uint32_t memoryTypeBits, uint32_t memorySize, VkMemoryPropertyFlags properties, uint32_t* memoryTypeIndex) {
	VkPhysicalDeviceMemoryProperties memoryProperties = { 0 };

	vkGetPhysicalDeviceMemoryProperties(vkGPU->physicalDevice, &memoryProperties);

	for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
		if ((memoryTypeBits & (1 << i)) && ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) && (memoryProperties.memoryHeaps[memoryProperties.memoryTypes[i].heapIndex].size >= memorySize))
		{
			memoryTypeIndex[0] = i;
			return VK_SUCCESS;
		}
	}
	return VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

VkResult allocateFFTBuffer(VkGPU* vkGPU, VkBuffer* buffer, VkDeviceMemory* deviceMemory, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags propertyFlags, uint64_t size) {
	//allocate the buffer used by the GPU with specified properties
	VkResult res = VK_SUCCESS;
	uint32_t queueFamilyIndices;
	VkBufferCreateInfo bufferCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	bufferCreateInfo.queueFamilyIndexCount = 1;
	bufferCreateInfo.pQueueFamilyIndices = &queueFamilyIndices;
	bufferCreateInfo.size = size;
	bufferCreateInfo.usage = usageFlags;
	res = vkCreateBuffer(vkGPU->device, &bufferCreateInfo, NULL, buffer);
	if (res != VK_SUCCESS) return res;
	VkMemoryRequirements memoryRequirements = { 0 };
	vkGetBufferMemoryRequirements(vkGPU->device, buffer[0], &memoryRequirements);
	VkMemoryAllocateInfo memoryAllocateInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
	memoryAllocateInfo.allocationSize = memoryRequirements.size;
	res = findMemoryType(vkGPU, memoryRequirements.memoryTypeBits, memoryRequirements.size, propertyFlags, &memoryAllocateInfo.memoryTypeIndex);
	if (res != VK_SUCCESS) return res;
	res = vkAllocateMemory(vkGPU->device, &memoryAllocateInfo, NULL, deviceMemory);
	if (res != VK_SUCCESS) return res;
	res = vkBindBufferMemory(vkGPU->device, buffer[0], deviceMemory[0], 0);
	if (res != VK_SUCCESS) return res;
	return res;
}
VkResult transferDataFromCPU(VkGPU* vkGPU, void* arr, VkBuffer* buffer, uint64_t bufferSize) {
	//a function that transfers data from the CPU to the GPU using staging buffer, because the GPU memory is not host-coherent
	VkResult res = VK_SUCCESS;
	uint64_t stagingBufferSize = bufferSize;
	VkBuffer stagingBuffer = { 0 };
	VkDeviceMemory stagingBufferMemory = { 0 };
	res = allocateFFTBuffer(vkGPU, &stagingBuffer, &stagingBufferMemory, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBufferSize);
	if (res != VK_SUCCESS) return res;
	void* data;
	res = vkMapMemory(vkGPU->device, stagingBufferMemory, 0, stagingBufferSize, 0, &data);
	if (res != VK_SUCCESS) return res;
	memcpy(data, arr, stagingBufferSize);
	vkUnmapMemory(vkGPU->device, stagingBufferMemory);
	VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	commandBufferAllocateInfo.commandPool = vkGPU->commandPool;
	commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferAllocateInfo.commandBufferCount = 1;
	VkCommandBuffer commandBuffer = { 0 };
	res = vkAllocateCommandBuffers(vkGPU->device, &commandBufferAllocateInfo, &commandBuffer);
	if (res != VK_SUCCESS) return res;
	VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	res = vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
	if (res != VK_SUCCESS) return res;
	VkBufferCopy copyRegion = { 0 };
	copyRegion.srcOffset = 0;
	copyRegion.dstOffset = 0;
	copyRegion.size = stagingBufferSize;
	vkCmdCopyBuffer(commandBuffer, stagingBuffer, buffer[0], 1, &copyRegion);
	res = vkEndCommandBuffer(commandBuffer);
	if (res != VK_SUCCESS) return res;
	VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;
	res = vkQueueSubmit(vkGPU->queue, 1, &submitInfo, vkGPU->fence);
	if (res != VK_SUCCESS) return res;
	res = vkWaitForFences(vkGPU->device, 1, &vkGPU->fence, VK_TRUE, 100000000000);
	if (res != VK_SUCCESS) return res;
	res = vkResetFences(vkGPU->device, 1, &vkGPU->fence);
	if (res != VK_SUCCESS) return res;
	vkFreeCommandBuffers(vkGPU->device, vkGPU->commandPool, 1, &commandBuffer);
	vkDestroyBuffer(vkGPU->device, stagingBuffer, NULL);
	vkFreeMemory(vkGPU->device, stagingBufferMemory, NULL);
	return res;
}
VkResult transferDataToCPU(VkGPU* vkGPU, void* arr, VkBuffer* buffer, uint64_t bufferSize) {
	//a function that transfers data from the GPU to the CPU using staging buffer, because the GPU memory is not host-coherent
	VkResult res = VK_SUCCESS;
	uint64_t stagingBufferSize = bufferSize;
	VkBuffer stagingBuffer = { 0 };
	VkDeviceMemory stagingBufferMemory = { 0 };
	res = allocateFFTBuffer(vkGPU, &stagingBuffer, &stagingBufferMemory, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBufferSize);
	if (res != VK_SUCCESS) return res;
	VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	commandBufferAllocateInfo.commandPool = vkGPU->commandPool;
	commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferAllocateInfo.commandBufferCount = 1;
	VkCommandBuffer commandBuffer = { 0 };
	res = vkAllocateCommandBuffers(vkGPU->device, &commandBufferAllocateInfo, &commandBuffer);
	if (res != VK_SUCCESS) return res;
	VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	res = vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
	if (res != VK_SUCCESS) return res;
	VkBufferCopy copyRegion = { 0 };
	copyRegion.srcOffset = 0;
	copyRegion.dstOffset = 0;
	copyRegion.size = stagingBufferSize;
	vkCmdCopyBuffer(commandBuffer, buffer[0], stagingBuffer, 1, &copyRegion);
	vkEndCommandBuffer(commandBuffer);
	VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;
	res = vkQueueSubmit(vkGPU->queue, 1, &submitInfo, vkGPU->fence);
	if (res != VK_SUCCESS) return res;
	res = vkWaitForFences(vkGPU->device, 1, &vkGPU->fence, VK_TRUE, 100000000000);
	if (res != VK_SUCCESS) return res;
	res = vkResetFences(vkGPU->device, 1, &vkGPU->fence);
	if (res != VK_SUCCESS) return res;
	vkFreeCommandBuffers(vkGPU->device, vkGPU->commandPool, 1, &commandBuffer);
	void* data;
	res = vkMapMemory(vkGPU->device, stagingBufferMemory, 0, stagingBufferSize, 0, &data);
	if (res != VK_SUCCESS) return res;
	memcpy(arr, data, stagingBufferSize);
	vkUnmapMemory(vkGPU->device, stagingBufferMemory);
	vkDestroyBuffer(vkGPU->device, stagingBuffer, NULL);
	vkFreeMemory(vkGPU->device, stagingBufferMemory, NULL);
	return res;
}
#endif
void performVulkanFFT(VkGPU* vkGPU, VkFFTApplication* app, int inverse, uint32_t num_iter) {
#if(VKFFT_BACKEND==0)
	VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	commandBufferAllocateInfo.commandPool = vkGPU->commandPool;
	commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferAllocateInfo.commandBufferCount = 1;
	VkCommandBuffer commandBuffer = {};
	vkAllocateCommandBuffers(vkGPU->device, &commandBufferAllocateInfo, &commandBuffer);
	VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
	//Record commands num_iter times. Allows to perform multiple convolutions/transforms in one submit.
	for (uint32_t i = 0; i < num_iter; i++) {
		VkFFTAppend(app, inverse, &commandBuffer);
	}
	vkEndCommandBuffer(commandBuffer);
	VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;
	auto timeSubmit = std::chrono::system_clock::now();
	vkQueueSubmit(vkGPU->queue, 1, &submitInfo, vkGPU->fence);
	vkWaitForFences(vkGPU->device, 1, &vkGPU->fence, VK_TRUE, 100000000000);
	auto timeEnd = std::chrono::system_clock::now();
	double totTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001;
	//printf("Pure submit execution time per num_iter: %.3f ms\n", totTime / num_iter);
	vkResetFences(vkGPU->device, 1, &vkGPU->fence);
	vkFreeCommandBuffers(vkGPU->device, vkGPU->commandPool, 1, &commandBuffer);
#elif(VKFFT_BACKEND==1)

	auto timeSubmit = std::chrono::system_clock::now();
	for (uint32_t i = 0; i < num_iter; i++) {
		VkFFTAppend(app, inverse, NULL);
	}
	cudaDeviceSynchronize();
	auto timeEnd = std::chrono::system_clock::now();
	double totTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001;
#elif(VKFFT_BACKEND==2)
	auto timeSubmit = std::chrono::system_clock::now();
	for (uint32_t i = 0; i < num_iter; i++) {
		VkFFTAppend(app, inverse, NULL);
	}
	hipDeviceSynchronize();
	auto timeEnd = std::chrono::system_clock::now();
	double totTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001;
#endif
}
float performVulkanFFTiFFT(VkGPU* vkGPU, VkFFTApplication* app, uint32_t num_iter) {
#if(VKFFT_BACKEND==0)
	VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	commandBufferAllocateInfo.commandPool = vkGPU->commandPool;
	commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferAllocateInfo.commandBufferCount = 1;
	VkCommandBuffer commandBuffer = {};
	vkAllocateCommandBuffers(vkGPU->device, &commandBufferAllocateInfo, &commandBuffer);
	VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
	for (uint32_t i = 0; i < num_iter; i++) {
		VkFFTAppend(app, -1, &commandBuffer);
		VkFFTAppend(app, 1, &commandBuffer);
	}
	vkEndCommandBuffer(commandBuffer);
	VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;
	auto timeSubmit = std::chrono::high_resolution_clock::now();
	vkQueueSubmit(vkGPU->queue, 1, &submitInfo, vkGPU->fence);
	vkWaitForFences(vkGPU->device, 1, &vkGPU->fence, VK_TRUE, 100000000000);
	auto timeEnd = std::chrono::high_resolution_clock::now();
	float totTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001;
	vkResetFences(vkGPU->device, 1, &vkGPU->fence);
	vkFreeCommandBuffers(vkGPU->device, vkGPU->commandPool, 1, &commandBuffer);
#elif(VKFFT_BACKEND==1)
	auto timeSubmit = std::chrono::system_clock::now();
	for (uint32_t i = 0; i < num_iter; i++) {
		VkFFTAppend(app, -1, NULL);
		VkFFTAppend(app, 1, NULL);
	}
	cudaDeviceSynchronize();
	auto timeEnd = std::chrono::system_clock::now();
	double totTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001;
#elif(VKFFT_BACKEND==2)
	auto timeSubmit = std::chrono::system_clock::now();
	for (uint32_t i = 0; i < num_iter; i++) {
		VkFFTAppend(app, -1, NULL);
		VkFFTAppend(app, 1, NULL);
	}
	hipDeviceSynchronize();
	auto timeEnd = std::chrono::system_clock::now();
	double totTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001;
#endif
	return totTime / num_iter;
}
uint32_t sample_0(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	uint32_t res = 0;
	if (file_output)
		fprintf(output, "0 - VkFFT FFT + iFFT C2C benchmark 1D batched in single precision\n");
	printf("0 - VkFFT FFT + iFFT C2C benchmark 1D batched in single precision\n");
	const int num_runs = 3;
	double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	//memory allocated on the CPU once, makes benchmark completion faster + avoids performance issues connected to frequent allocation/deallocation.
	float* buffer_input = (float*)malloc((uint64_t)4 * 2 * pow(2, 27));
	for (uint64_t i = 0; i < 2 * pow(2, 27); i++) {
		buffer_input[i] = 2 * ((float)rand()) / RAND_MAX - 1.0;
	}
	for (uint32_t n = 0; n < 26; n++) {
		double run_time[num_runs];
		for (uint32_t r = 0; r < num_runs; r++) {
			//Configuration + FFT application .
			VkFFTConfiguration configuration = {};
			VkFFTApplication app = {};
			//FFT + iFFT sample code.
			//Setting up FFT configuration for forward and inverse FFT.
			configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).
			configuration.size[0] = 4 * pow(2, n); //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			if (n == 0) configuration.size[0] = 4096;
			configuration.size[1] = 64 * 32 * pow(2, 16) / configuration.size[0];
			if (configuration.size[1] < 1) configuration.size[1] = 1;
			configuration.size[2] = 1;

			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			configuration.device = &vkGPU->device;
#if(VKFFT_BACKEND==0)
			configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
			configuration.fence = &vkGPU->fence;
			configuration.commandPool = &vkGPU->commandPool;
			configuration.physicalDevice = &vkGPU->physicalDevice;
			configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization
#endif
			//Allocate buffer for the input data.
			uint64_t bufferSize = (uint64_t)sizeof(float) * 2 * configuration.size[0] * configuration.size[1] * configuration.size[2];;
#if(VKFFT_BACKEND==0)
			VkBuffer buffer = {};
			VkDeviceMemory bufferDeviceMemory = {};
			allocateFFTBuffer(vkGPU, &buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
			configuration.buffer = &buffer;
#elif(VKFFT_BACKEND==1)
			cuFloatComplex* buffer = 0;
			cudaMalloc((void**)&buffer, bufferSize);
			configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==2)
			hipFloatComplex* buffer = 0;
			hipMalloc((void**)&buffer, bufferSize);
			configuration.buffer = (void**)&buffer;
#endif

			configuration.bufferSize = &bufferSize;

			//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
			/*float* buffer_input = (float*)malloc(bufferSize);

			for (uint32_t k = 0; k < configuration.size[2]; k++) {
				for (uint32_t j = 0; j < configuration.size[1]; j++) {
					for (uint32_t i = 0; i < configuration.size[0]; i++) {
						buffer_input[2 * (i + j * configuration.size[0] + k * (configuration.size[0]) * configuration.size[1])] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						buffer_input[2 * (i + j * configuration.size[0] + k * (configuration.size[0]) * configuration.size[1]) + 1] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						}
					}
				}

			*/
			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
#if(VKFFT_BACKEND==0)
			transferDataFromCPU(vkGPU, buffer_input, &buffer, bufferSize);
#elif(VKFFT_BACKEND==1)
			cudaMemcpy(buffer, buffer_input, bufferSize, cudaMemcpyHostToDevice);
#elif(VKFFT_BACKEND==2)
			hipMemcpy(buffer, buffer_input, bufferSize, hipMemcpyHostToDevice);
#endif

			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			res = initializeVkFFT(&app, configuration);
			if (res != 0) return res;

			//Submit FFT+iFFT.
			uint32_t num_iter = ((3 * 4096 * 1024.0 * 1024.0) / bufferSize > 1000) ? 1000 : (3 * 4096 * 1024.0 * 1024.0) / bufferSize;
#if(VKFFT_BACKEND==0)
			if (vkGPU->physicalDeviceProperties.vendorID == 0x8086) num_iter /= 4;//smaller benchmark for Intel GPUs
#endif
			if (num_iter == 0) num_iter = 1;
			float totTime = performVulkanFFTiFFT(vkGPU, &app, num_iter);

			run_time[r] = totTime;
			if (n > 0) {
				if (r == num_runs - 1) {
					double std_error = 0;
					double avg_time = 0;
					for (uint32_t t = 0; t < num_runs; t++) {
						avg_time += run_time[t];
					}
					avg_time /= num_runs;
					for (uint32_t t = 0; t < num_runs; t++) {
						std_error += (run_time[t] - avg_time) * (run_time[t] - avg_time);
					}
					std_error = sqrt(std_error / num_runs);
					uint32_t num_tot_transfers = 0;
					for (uint32_t i = 0; i < configuration.FFTdim; i++)
						num_tot_transfers += app.localFFTPlan->numAxisUploads[i];
					num_tot_transfers *= 4;
					if (file_output)
						fprintf(output, "VkFFT System: %d %dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %d benchmark: %d bandwidth: %0.1f\n", (int)log2(configuration.size[0]), configuration.size[0], configuration.size[1], bufferSize / 1024 / 1024, avg_time, std_error, num_iter, (int)(((double)bufferSize / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);

					printf("VkFFT System: %d %dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %d benchmark: %d bandwidth: %0.1f\n", (int)log2(configuration.size[0]), configuration.size[0], configuration.size[1], bufferSize / 1024 / 1024, avg_time, std_error, num_iter, (int)(((double)bufferSize / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);
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
#endif

			deleteVkFFT(&app);

		}
	}
	free(buffer_input);
	benchmark_result /= (26 - 1);

	if (file_output) {
		fprintf(output, "Benchmark score VkFFT: %d\n", (int)(benchmark_result));
#if(VKFFT_BACKEND==0)
		fprintf(output, "Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
#endif
	}
	printf("Benchmark score VkFFT: %d\n", (int)(benchmark_result));
#if(VKFFT_BACKEND==0)
	printf("Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
#endif
	return res;
}
uint32_t sample_1(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	uint32_t res = 0;
	if (file_output)
		fprintf(output, "1 - VkFFT FFT + iFFT C2C benchmark 1D batched in double precision LUT\n");
	printf("1 - VkFFT FFT + iFFT C2C benchmark 1D batched in double precision LUT\n");
	const int num_runs = 3;
	double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	//memory allocated on the CPU once, makes benchmark completion faster + avoids performance issues connected to frequent allocation/deallocation.
	double* buffer_input = (double*)malloc((uint64_t)8 * 2 * pow(2, 27));
	for (uint64_t i = 0; i < 2 * pow(2, 27); i++) {
		buffer_input[i] = 2 * ((double)rand()) / RAND_MAX - 1.0;
	}
	for (uint32_t n = 0; n < 24; n++) {
		double run_time[num_runs];
		for (uint32_t r = 0; r < num_runs; r++) {
			//Configuration + FFT application .
			VkFFTConfiguration configuration = {};
			VkFFTApplication app = {};
			//FFT + iFFT sample code.
			//Setting up FFT configuration for forward and inverse FFT.
			configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).
			configuration.size[0] = 4 * pow(2, n); //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			if (n == 0) configuration.size[0] = 2048;
			configuration.size[1] = 64 * 32 * pow(2, 15) / configuration.size[0];
			if (configuration.size[1] < 1) configuration.size[1] = 1;
			configuration.size[2] = 1;


			configuration.doublePrecision = true;

			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			configuration.device = &vkGPU->device;
#if(VKFFT_BACKEND==0)
			configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
			configuration.fence = &vkGPU->fence;
			configuration.commandPool = &vkGPU->commandPool;
			configuration.physicalDevice = &vkGPU->physicalDevice;
			configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization
#endif			

			//Allocate buffer for the input data.
			uint64_t bufferSize = (uint64_t)sizeof(double) * 2 * configuration.size[0] * configuration.size[1] * configuration.size[2];;
#if(VKFFT_BACKEND==0)
			VkBuffer buffer = {};
			VkDeviceMemory bufferDeviceMemory = {};
			allocateFFTBuffer(vkGPU, &buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
			configuration.buffer = &buffer;
#elif(VKFFT_BACKEND==1)
			cuFloatComplex* buffer = 0;
			cudaMalloc((void**)&buffer, bufferSize);
			configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==2)
			hipFloatComplex* buffer = 0;
			hipMalloc((void**)&buffer, bufferSize);
			configuration.buffer = (void**)&buffer;
#endif

			configuration.bufferSize = &bufferSize;
			//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
			/*float* buffer_input = (float*)malloc(bufferSize);

			for (uint32_t k = 0; k < configuration.size[2]; k++) {
				for (uint32_t j = 0; j < configuration.size[1]; j++) {
					for (uint32_t i = 0; i < configuration.size[0]; i++) {
						buffer_input[2 * (i + j * configuration.size[0] + k * (configuration.size[0]) * configuration.size[1])] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						buffer_input[2 * (i + j * configuration.size[0] + k * (configuration.size[0]) * configuration.size[1]) + 1] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						}
					}
				}
			*/
			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
#if(VKFFT_BACKEND==0)
			transferDataFromCPU(vkGPU, buffer_input, &buffer, bufferSize);
#elif(VKFFT_BACKEND==1)
			cudaMemcpy(buffer, buffer_input, bufferSize, cudaMemcpyHostToDevice);
#elif(VKFFT_BACKEND==2)
			hipMemcpy(buffer, buffer_input, bufferSize, hipMemcpyHostToDevice);
#endif
			//free(buffer_input);

			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			res = initializeVkFFT(&app, configuration);
			if (res != 0) return res;

			//Submit FFT+iFFT.
			uint32_t num_iter = ((4096 * 1024.0 * 1024.0) / bufferSize > 1000) ? 1000 : (4096 * 1024.0 * 1024.0) / bufferSize;
#if(VKFFT_BACKEND==0)
			if (vkGPU->physicalDeviceProperties.vendorID == 0x8086) num_iter /= 4;
#endif
			if (num_iter == 0) num_iter = 1;

			float totTime = performVulkanFFTiFFT(vkGPU, &app, num_iter);

			run_time[r] = totTime;
			if (n > 0) {
				if (r == num_runs - 1) {
					double std_error = 0;
					double avg_time = 0;
					for (uint32_t t = 0; t < num_runs; t++) {
						avg_time += run_time[t];
					}
					avg_time /= num_runs;
					for (uint32_t t = 0; t < num_runs; t++) {
						std_error += (run_time[t] - avg_time) * (run_time[t] - avg_time);
					}
					std_error = sqrt(std_error / num_runs);
					uint32_t num_tot_transfers = 0;
					for (uint32_t i = 0; i < configuration.FFTdim; i++)
						num_tot_transfers += app.localFFTPlan->numAxisUploads[i];
					num_tot_transfers *= 4;
					if (file_output)
						fprintf(output, "VkFFT System: %d %dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %d benchmark: %d bandwidth: %0.1f\n", (int)log2(configuration.size[0]), configuration.size[0], configuration.size[1], bufferSize / 1024 / 1024, avg_time, std_error, num_iter, (int)(((double)bufferSize * sizeof(float) / sizeof(double) / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);

					printf("VkFFT System: %d %dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %d benchmark: %d bandwidth: %0.1f\n", (int)log2(configuration.size[0]), configuration.size[0], configuration.size[1], bufferSize / 1024 / 1024, avg_time, std_error, num_iter, (int)(((double)bufferSize * sizeof(float) / sizeof(double) / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);
					benchmark_result += ((double)bufferSize * sizeof(float) / sizeof(double) / 1024) / avg_time;
				}


			}

#if(VKFFT_BACKEND==0)
			vkDestroyBuffer(vkGPU->device, buffer, NULL);
			vkFreeMemory(vkGPU->device, bufferDeviceMemory, NULL);
#elif(VKFFT_BACKEND==1)
			cudaFree(buffer);
#elif(VKFFT_BACKEND==2)
			hipFree(buffer);
#endif
			deleteVkFFT(&app);

		}
	}
	free(buffer_input);
	benchmark_result /= (24 - 1);
	if (file_output) {
		fprintf(output, "Benchmark score VkFFT: %d\n", (int)(benchmark_result));
#if(VKFFT_BACKEND==0)
		fprintf(output, "Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
#endif
	}
	printf("Benchmark score VkFFT: %d\n", (int)(benchmark_result));
#if(VKFFT_BACKEND==0)
	printf("Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
#endif
	return res;
}
#if (VK_API_VERSION>10)
uint32_t sample_2(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	uint32_t res = 0;
	if (file_output)
		fprintf(output, "2 - VkFFT FFT + iFFT C2C benchmark 1D batched in half precision\n");
	printf("2 - VkFFT FFT + iFFT C2C benchmark 1D batched in half precision\n");
	const int num_runs = 3;
	double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	//memory allocated on the CPU once, makes benchmark completion faster + avoids performance issues connected to frequent allocation/deallocation.
	half* buffer_input = (half*)malloc((uint64_t)4 * 2 * pow(2, 27));
	for (uint64_t i = 0; i < 2 * pow(2, 27); i++) {
		buffer_input[i] = 2 * ((half)rand()) / RAND_MAX - 1.0;
	}
	for (uint32_t n = 0; n < 25; n++) {
		double run_time[num_runs];
		for (uint32_t r = 0; r < num_runs; r++) {
			//Configuration + FFT application .
			VkFFTConfiguration configuration = {};
			VkFFTApplication app = {};
			//FFT + iFFT sample code.
			//Setting up FFT configuration for forward and inverse FFT.
			configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).
			configuration.size[0] = 4 * pow(2, n); //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			if (n == 0) configuration.size[0] = 4096;
			configuration.size[1] = 64 * 32 * pow(2, 16) / configuration.size[0];
			if (configuration.size[1] < 1) configuration.size[1] = 1;
			configuration.size[2] = 1;

			configuration.halfPrecision = true;

			//PARAMETERS THAT CAN BE ADJUSTED FOR SPECIFIC GPU's - this configuration is by no means final form
#if(VKFFT_BACKEND==0)
			if (vkGPU->physicalDeviceProperties.vendorID == 0x8086) {
				if (n > 22)//128byte coalescing has a limit of 2^24 max size
					configuration.coalescedMemory = 64;
				else
					configuration.coalescedMemory = 128;
			}
#endif
			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			configuration.device = &vkGPU->device;
#if(VKFFT_BACKEND==0)
			configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
			configuration.fence = &vkGPU->fence;
			configuration.commandPool = &vkGPU->commandPool;
			configuration.physicalDevice = &vkGPU->physicalDevice;
			configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization
#endif

			//Allocate buffer for the input data.
			uint64_t bufferSize = (uint64_t)2 * sizeof(half) * configuration.size[0] * configuration.size[1] * configuration.size[2];;
#if(VKFFT_BACKEND==0)
			VkBuffer buffer = {};
			VkDeviceMemory bufferDeviceMemory = {};
			allocateFFTBuffer(vkGPU, &buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
			configuration.buffer = &buffer;
#elif(VKFFT_BACKEND==1)
			cuFloatComplex* buffer = 0;
			cudaMalloc((void**)&buffer, bufferSize);
			configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==2)
			hipFloatComplex* buffer = 0;
			hipMalloc((void**)&buffer, bufferSize);
			configuration.buffer = (void**)&buffer;
#endif

			configuration.bufferSize = &bufferSize;
			//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
			/*float* buffer_input = (float*)malloc(bufferSize);

			for (uint32_t k = 0; k < configuration.size[2]; k++) {
				for (uint32_t j = 0; j < configuration.size[1]; j++) {
					for (uint32_t i = 0; i < configuration.size[0]; i++) {
						buffer_input[2 * (i + j * configuration.size[0] + k * (configuration.size[0]) * configuration.size[1])] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						buffer_input[2 * (i + j * configuration.size[0] + k * (configuration.size[0]) * configuration.size[1]) + 1] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						}
					}
				}
			*/
			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
#if(VKFFT_BACKEND==0)
			transferDataFromCPU(vkGPU, buffer_input, &buffer, bufferSize);
#elif(VKFFT_BACKEND==1)
			cudaMemcpy(buffer, buffer_input, bufferSize, cudaMemcpyHostToDevice);
#elif(VKFFT_BACKEND==2)
			hipMemcpy(buffer, buffer_input, bufferSize, hipMemcpyHostToDevice);
#endif
			//free(buffer_input);

			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			res = initializeVkFFT(&app, configuration);
			if (res != 0) return res;

			//Submit FFT+iFFT.
			uint32_t num_iter = ((4096 * 1024.0 * 1024.0) / bufferSize > 1000) ? 1000 : (4096 * 1024.0 * 1024.0) / bufferSize;
#if(VKFFT_BACKEND==0)
			if (vkGPU->physicalDeviceProperties.vendorID == 0x8086) num_iter /= 4;
#endif
			if (num_iter == 0) num_iter = 1;
			float totTime = performVulkanFFTiFFT(vkGPU, &app, num_iter);

			run_time[r] = totTime;
			if (n > 0) {
				if (r == num_runs - 1) {
					double std_error = 0;
					double avg_time = 0;
					for (uint32_t t = 0; t < num_runs; t++) {
						avg_time += run_time[t];
					}
					avg_time /= num_runs;
					for (uint32_t t = 0; t < num_runs; t++) {
						std_error += (run_time[t] - avg_time) * (run_time[t] - avg_time);
					}
					std_error = sqrt(std_error / num_runs);
					uint32_t num_tot_transfers = 0;
					for (uint32_t i = 0; i < configuration.FFTdim; i++)
						num_tot_transfers += app.localFFTPlan->numAxisUploads[i];
					num_tot_transfers *= 4;
					if (file_output)
						fprintf(output, "VkFFT System: %d %dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %d benchmark: %d bandwidth: %0.1f\n", (int)log2(configuration.size[0]), configuration.size[0], configuration.size[1], bufferSize / 1024 / 1024, avg_time, std_error, num_iter, (int)(((double)bufferSize * sizeof(float) / sizeof(half) / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);

					printf("VkFFT System: %d %dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %d benchmark: %d bandwidth: %0.1f\n", (int)log2(configuration.size[0]), configuration.size[0], configuration.size[1], bufferSize / 1024 / 1024, avg_time, std_error, num_iter, (int)(((double)bufferSize * sizeof(float) / sizeof(half) / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);
					benchmark_result += ((double)bufferSize * sizeof(float) / sizeof(half) / 1024) / avg_time;
				}


			}

#if(VKFFT_BACKEND==0)
			vkDestroyBuffer(vkGPU->device, buffer, NULL);
			vkFreeMemory(vkGPU->device, bufferDeviceMemory, NULL);
#elif(VKFFT_BACKEND==1)
			cudaFree(buffer);
#elif(VKFFT_BACKEND==2)
			hipFree(buffer);
#endif
			deleteVkFFT(&app);

		}
	}
	free(buffer_input);
	benchmark_result /= (25 - 1);
	if (file_output) {
		fprintf(output, "Benchmark score VkFFT: %d\n", (int)(benchmark_result));
#if(VKFFT_BACKEND==0)
		fprintf(output, "Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
#endif
	}
	printf("Benchmark score VkFFT: %d\n", (int)(benchmark_result));
#if(VKFFT_BACKEND==0)
	printf("Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
#endif
	return res;
}
#endif
uint32_t sample_3(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	uint32_t res = 0;
	if (file_output)
		fprintf(output, "3 - VkFFT FFT + iFFT C2C multidimensional benchmark in single precision\n");
	printf("3 - VkFFT FFT + iFFT C2C multidimensional benchmark in single precision\n");

	const int num_benchmark_samples = 39;
	const int num_runs = 3;
	uint32_t benchmark_dimensions[num_benchmark_samples][4] = { {1024, 1024, 1, 2},
	{720, 480, 1, 2},{1280, 720, 1, 2},{1920, 1080, 1, 2}, {2560, 1440, 1, 2},{3840, 2160, 1, 2},{7680, 4320, 1, 2},
	{(uint32_t)pow(2,6), (uint32_t)pow(2,6), 1, 2},	{(uint32_t)pow(2,7), (uint32_t)pow(2,6), 1, 2}, {(uint32_t)pow(2,7), (uint32_t)pow(2,7), 1, 2},	{(uint32_t)pow(2,8), (uint32_t)pow(2,7), 1, 2},{(uint32_t)pow(2,8), (uint32_t)pow(2,8), 1, 2},
	{(uint32_t)pow(2,9), (uint32_t)pow(2,8), 1, 2},{(uint32_t)pow(2,9), (uint32_t)pow(2,9), 1, 2},	{(uint32_t)pow(2,10), (uint32_t)pow(2,9), 1, 2},{(uint32_t)pow(2,10), (uint32_t)pow(2,10), 1, 2},	{(uint32_t)pow(2,11), (uint32_t)pow(2,10), 1, 2},{(uint32_t)pow(2,11), (uint32_t)pow(2,11), 1, 2},
	{(uint32_t)pow(2,12), (uint32_t)pow(2,11), 1, 2},{(uint32_t)pow(2,12), (uint32_t)pow(2,12), 1, 2},	{(uint32_t)pow(2,13), (uint32_t)pow(2,12), 1, 2},	{(uint32_t)pow(2,13), (uint32_t)pow(2,13), 1, 2},{(uint32_t)pow(2,14), (uint32_t)pow(2,13), 1, 2},
	{(uint32_t)pow(2,4), (uint32_t)pow(2,4), (uint32_t)pow(2,4), 3} ,{(uint32_t)pow(2,5), (uint32_t)pow(2,4), (uint32_t)pow(2,4), 3} ,{(uint32_t)pow(2,5), (uint32_t)pow(2,5), (uint32_t)pow(2,4), 3} ,{(uint32_t)pow(2,5), (uint32_t)pow(2,5), (uint32_t)pow(2,5), 3},{(uint32_t)pow(2,6), (uint32_t)pow(2,5), (uint32_t)pow(2,5), 3} ,{(uint32_t)pow(2,6), (uint32_t)pow(2,6), (uint32_t)pow(2,5), 3} ,
	{(uint32_t)pow(2,6), (uint32_t)pow(2,6), (uint32_t)pow(2,6), 3},{(uint32_t)pow(2,7), (uint32_t)pow(2,6), (uint32_t)pow(2,6), 3} ,{(uint32_t)pow(2,7), (uint32_t)pow(2,7), (uint32_t)pow(2,6), 3} ,{(uint32_t)pow(2,7), (uint32_t)pow(2,7), (uint32_t)pow(2,7), 3},{(uint32_t)pow(2,8), (uint32_t)pow(2,7), (uint32_t)pow(2,7), 3} , {(uint32_t)pow(2,8), (uint32_t)pow(2,8), (uint32_t)pow(2,7), 3} ,
	{(uint32_t)pow(2,8), (uint32_t)pow(2,8), (uint32_t)pow(2,8), 3},{(uint32_t)pow(2,9), (uint32_t)pow(2,8), (uint32_t)pow(2,8), 3}, {(uint32_t)pow(2,9), (uint32_t)pow(2,9), (uint32_t)pow(2,8), 3},{(uint32_t)pow(2,9), (uint32_t)pow(2,9), (uint32_t)pow(2,9), 3}
	};
	double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	//memory allocated on the CPU once, makes benchmark completion faster + avoids performance issues connected to frequent allocation/deallocation.
	float* buffer_input = (float*)malloc((uint64_t)4 * 2 * pow(2, 27));
	for (uint64_t i = 0; i < 2 * pow(2, 27); i++) {
		buffer_input[i] = 2 * ((float)rand()) / RAND_MAX - 1.0;
	}
	for (uint32_t n = 0; n < num_benchmark_samples; n++) {
		double run_time[num_runs];
		for (uint32_t r = 0; r < num_runs; r++) {
			//Configuration + FFT application .
			VkFFTConfiguration configuration = {};
			VkFFTApplication app = {};
			//FFT + iFFT sample code.
			//Setting up FFT configuration for forward and inverse FFT.
			configuration.FFTdim = benchmark_dimensions[n][3]; //FFT dimension, 1D, 2D or 3D (default 1).
			configuration.size[0] = benchmark_dimensions[n][0]; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			configuration.size[1] = benchmark_dimensions[n][1];
			configuration.size[2] = benchmark_dimensions[n][2];

			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			configuration.device = &vkGPU->device;
#if(VKFFT_BACKEND==0)
			configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
			configuration.fence = &vkGPU->fence;
			configuration.commandPool = &vkGPU->commandPool;
			configuration.physicalDevice = &vkGPU->physicalDevice;
			configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization
#endif
			//Allocate buffer for the input data.
			uint64_t bufferSize = (uint64_t)sizeof(float) * 2 * configuration.size[0] * configuration.size[1] * configuration.size[2];;
#if(VKFFT_BACKEND==0)
			VkBuffer buffer = {};
			VkDeviceMemory bufferDeviceMemory = {};
			allocateFFTBuffer(vkGPU, &buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
			configuration.buffer = &buffer;
#elif(VKFFT_BACKEND==1)
			cuFloatComplex* buffer = 0;
			cudaMalloc((void**)&buffer, bufferSize);
			configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==2)
			hipFloatComplex* buffer = 0;
			hipMalloc((void**)&buffer, bufferSize);
			configuration.buffer = (void**)&buffer;
#endif

			configuration.bufferSize = &bufferSize;
			//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
			/*float* buffer_input = (float*)malloc(bufferSize);

			for (uint32_t k = 0; k < configuration.size[2]; k++) {
				for (uint32_t j = 0; j < configuration.size[1]; j++) {
					for (uint32_t i = 0; i < configuration.size[0]; i++) {
						buffer_input[2 * (i + j * configuration.size[0] + k * (configuration.size[0]) * configuration.size[1])] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						buffer_input[2 * (i + j * configuration.size[0] + k * (configuration.size[0]) * configuration.size[1]) + 1] = 2 * ((float)rand()) / RAND_MAX - 1.0;
					}
				}
			}
			*/
			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
#if(VKFFT_BACKEND==0)
			transferDataFromCPU(vkGPU, buffer_input, &buffer, bufferSize);
#elif(VKFFT_BACKEND==1)
			cudaMemcpy(buffer, buffer_input, bufferSize, cudaMemcpyHostToDevice);
#elif(VKFFT_BACKEND==2)
			hipMemcpy(buffer, buffer_input, bufferSize, hipMemcpyHostToDevice);
#endif
			//free(buffer_input);

			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			res = initializeVkFFT(&app, configuration);
			if (res != 0) return res;

			//Submit FFT+iFFT.
			uint32_t num_iter = ((4096 * 1024.0 * 1024.0) / bufferSize > 1000) ? 1000 : (4096 * 1024.0 * 1024.0) / bufferSize;
#if(VKFFT_BACKEND==0)
			if (vkGPU->physicalDeviceProperties.vendorID == 0x8086) num_iter /= 4;
#endif
			if (num_iter == 0) num_iter = 1;

			float totTime = performVulkanFFTiFFT(vkGPU, &app, num_iter);

			run_time[r] = totTime;
			if (n > 0) {
				if (r == num_runs - 1) {
					double std_error = 0;
					double avg_time = 0;
					for (uint32_t t = 0; t < num_runs; t++) {
						avg_time += run_time[t];
					}
					avg_time /= num_runs;
					for (uint32_t t = 0; t < num_runs; t++) {
						std_error += (run_time[t] - avg_time) * (run_time[t] - avg_time);
					}
					std_error = sqrt(std_error / num_runs);

					uint32_t num_tot_transfers = 0;
					for (uint32_t i = 0; i < configuration.FFTdim; i++)
						num_tot_transfers += app.localFFTPlan->numAxisUploads[i];
					num_tot_transfers *= 4;
					if (file_output)
						fprintf(output, "VkFFT System: %dx%dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %d benchmark: %d bandwidth: %0.1f\n", benchmark_dimensions[n][0], benchmark_dimensions[n][1], benchmark_dimensions[n][2], bufferSize / 1024 / 1024, avg_time, std_error, num_iter, (int)(((double)bufferSize / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);
					printf("VkFFT System: %dx%dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %d benchmark: %d bandwidth: %0.1f\n", benchmark_dimensions[n][0], benchmark_dimensions[n][1], benchmark_dimensions[n][2], bufferSize / 1024 / 1024, avg_time, std_error, num_iter, (int)(((double)bufferSize / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);
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
#endif
			deleteVkFFT(&app);

		}
	}
	free(buffer_input);
	benchmark_result /= (num_benchmark_samples - 1);

	if (file_output) {
		fprintf(output, "Benchmark score VkFFT: %d\n", (int)(benchmark_result));
#if(VKFFT_BACKEND==0)
		fprintf(output, "Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
#endif
	}
	printf("Benchmark score VkFFT: %d\n", (int)(benchmark_result));
#if(VKFFT_BACKEND==0)
	printf("Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
#endif
	return res;
}
uint32_t sample_4(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	uint32_t res = 0;
	if (file_output)
		fprintf(output, "4 - VkFFT FFT + iFFT C2C multidimensional benchmark in single precision, native zeropadding\n");
	printf("4 - VkFFT FFT + iFFT C2C multidimensional benchmark in single precision, native zeropadding\n");

	const int num_benchmark_samples = 39;
	const int num_runs = 3;
	uint32_t benchmark_dimensions[num_benchmark_samples][4] = { {1024, 1024, 1, 2},
	{720, 480, 1, 2},{1280, 720, 1, 2},{1920, 1080, 1, 2}, {2560, 1440, 1, 2},{3840, 2160, 1, 2},{7680, 4320, 1, 2},
	{(uint32_t)pow(2,6), (uint32_t)pow(2,6), 1, 2},	{(uint32_t)pow(2,7), (uint32_t)pow(2,6), 1, 2}, {(uint32_t)pow(2,7), (uint32_t)pow(2,7), 1, 2},	{(uint32_t)pow(2,8), (uint32_t)pow(2,7), 1, 2},{(uint32_t)pow(2,8), (uint32_t)pow(2,8), 1, 2},
	{(uint32_t)pow(2,9), (uint32_t)pow(2,8), 1, 2},{(uint32_t)pow(2,9), (uint32_t)pow(2,9), 1, 2},	{(uint32_t)pow(2,10), (uint32_t)pow(2,9), 1, 2},{(uint32_t)pow(2,10), (uint32_t)pow(2,10), 1, 2},	{(uint32_t)pow(2,11), (uint32_t)pow(2,10), 1, 2},{(uint32_t)pow(2,11), (uint32_t)pow(2,11), 1, 2},
	{(uint32_t)pow(2,12), (uint32_t)pow(2,11), 1, 2},{(uint32_t)pow(2,12), (uint32_t)pow(2,12), 1, 2},	{(uint32_t)pow(2,13), (uint32_t)pow(2,12), 1, 2},	{(uint32_t)pow(2,13), (uint32_t)pow(2,13), 1, 2},{(uint32_t)pow(2,14), (uint32_t)pow(2,13), 1, 2},
	{(uint32_t)pow(2,4), (uint32_t)pow(2,4), (uint32_t)pow(2,4), 3} ,{(uint32_t)pow(2,5), (uint32_t)pow(2,4), (uint32_t)pow(2,4), 3} ,{(uint32_t)pow(2,5), (uint32_t)pow(2,5), (uint32_t)pow(2,4), 3} ,{(uint32_t)pow(2,5), (uint32_t)pow(2,5), (uint32_t)pow(2,5), 3},{(uint32_t)pow(2,6), (uint32_t)pow(2,5), (uint32_t)pow(2,5), 3} ,{(uint32_t)pow(2,6), (uint32_t)pow(2,6), (uint32_t)pow(2,5), 3} ,
	{(uint32_t)pow(2,6), (uint32_t)pow(2,6), (uint32_t)pow(2,6), 3},{(uint32_t)pow(2,7), (uint32_t)pow(2,6), (uint32_t)pow(2,6), 3} ,{(uint32_t)pow(2,7), (uint32_t)pow(2,7), (uint32_t)pow(2,6), 3} ,{(uint32_t)pow(2,7), (uint32_t)pow(2,7), (uint32_t)pow(2,7), 3},{(uint32_t)pow(2,8), (uint32_t)pow(2,7), (uint32_t)pow(2,7), 3} , {(uint32_t)pow(2,8), (uint32_t)pow(2,8), (uint32_t)pow(2,7), 3} ,
	{(uint32_t)pow(2,8), (uint32_t)pow(2,8), (uint32_t)pow(2,8), 3},{(uint32_t)pow(2,9), (uint32_t)pow(2,8), (uint32_t)pow(2,8), 3}, {(uint32_t)pow(2,9), (uint32_t)pow(2,9), (uint32_t)pow(2,8), 3},{(uint32_t)pow(2,9), (uint32_t)pow(2,9), (uint32_t)pow(2,9), 3}
	};
	double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	//memory allocated on the CPU once, makes benchmark completion faster + avoids performance issues connected to frequent allocation/deallocation.
	float* buffer_input = (float*)malloc((uint64_t)4 * 2 * pow(2, 27));
	for (uint64_t i = 0; i < 2 * pow(2, 27); i++) {
		buffer_input[i] = 2 * ((float)rand()) / RAND_MAX - 1.0;
	}
	for (uint32_t n = 0; n < num_benchmark_samples; n++) {
		double run_time[num_runs];
		for (uint32_t r = 0; r < num_runs; r++) {
			//Configuration + FFT application .
			VkFFTConfiguration configuration = {};
			VkFFTApplication app = {};
			//FFT + iFFT sample code.
			//Setting up FFT configuration for forward and inverse FFT.
			configuration.FFTdim = benchmark_dimensions[n][3]; //FFT dimension, 1D, 2D or 3D (default 1).
			configuration.size[0] = benchmark_dimensions[n][0]; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			configuration.size[1] = benchmark_dimensions[n][1];
			configuration.size[2] = benchmark_dimensions[n][2];
			//PARAMETERS THAT CAN BE ADJUSTED FOR SPECIFIC GPU's - this configuration is by no means final form

			configuration.performZeropadding[0] = true; //Perform padding with zeros on GPU. Still need to properly align input data (no need to fill padding area with meaningful data) but this will increase performance due to the lower amount of the memory reads/writes and omitting sequences only consisting of zeros.
			configuration.performZeropadding[1] = true;
			configuration.performZeropadding[2] = true;
			configuration.fft_zeropad_left[0] = ceil(configuration.size[0] / 2.0);
			configuration.fft_zeropad_right[0] = configuration.size[0];
			configuration.fft_zeropad_left[1] = ceil(configuration.size[1] / 2.0);
			configuration.fft_zeropad_right[1] = configuration.size[1];
			configuration.fft_zeropad_left[2] = ceil(configuration.size[2] / 2.0);
			configuration.fft_zeropad_right[2] = configuration.size[2];

			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			configuration.device = &vkGPU->device;
#if(VKFFT_BACKEND==0)
			configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
			configuration.fence = &vkGPU->fence;
			configuration.commandPool = &vkGPU->commandPool;
			configuration.physicalDevice = &vkGPU->physicalDevice;
			configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization
#endif
			//Allocate buffer for the input data.
			uint64_t bufferSize = (uint64_t)sizeof(float) * 2 * configuration.size[0] * configuration.size[1] * configuration.size[2];;
#if(VKFFT_BACKEND==0)
			VkBuffer buffer = {};
			VkDeviceMemory bufferDeviceMemory = {};
			allocateFFTBuffer(vkGPU, &buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
			configuration.buffer = &buffer;
#elif(VKFFT_BACKEND==1)
			cuFloatComplex* buffer = 0;
			cudaMalloc((void**)&buffer, bufferSize);
			configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==2)
			hipFloatComplex* buffer = 0;
			hipMalloc((void**)&buffer, bufferSize);
			configuration.buffer = (void**)&buffer;
#endif

			configuration.bufferSize = &bufferSize;
			//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
			/*float* buffer_input = (float*)malloc(bufferSize);

			for (uint32_t k = 0; k < configuration.size[2]; k++) {
				for (uint32_t j = 0; j < configuration.size[1]; j++) {
					for (uint32_t i = 0; i < configuration.size[0]; i++) {
						buffer_input[2 * (i + j * configuration.size[0] + k * (configuration.size[0]) * configuration.size[1])] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						buffer_input[2 * (i + j * configuration.size[0] + k * (configuration.size[0]) * configuration.size[1]) + 1] = 2 * ((float)rand()) / RAND_MAX - 1.0;
					}
				}
			}
			*/
			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
#if(VKFFT_BACKEND==0)
			transferDataFromCPU(vkGPU, buffer_input, &buffer, bufferSize);
#elif(VKFFT_BACKEND==1)
			cudaMemcpy(buffer, buffer_input, bufferSize, cudaMemcpyHostToDevice);
#elif(VKFFT_BACKEND==2)
			hipMemcpy(buffer, buffer_input, bufferSize, hipMemcpyHostToDevice);
#endif
			//free(buffer_input);

			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			res = initializeVkFFT(&app, configuration);
			if (res != 0) return res;

			//Submit FFT+iFFT.
			uint32_t num_iter = ((4096 * 1024.0 * 1024.0) / bufferSize > 1000) ? 1000 : (4096 * 1024.0 * 1024.0) / bufferSize;
#if(VKFFT_BACKEND==0)
			if (vkGPU->physicalDeviceProperties.vendorID == 0x8086) num_iter /= 4;
#endif
			if (num_iter == 0) num_iter = 1;

			float totTime = performVulkanFFTiFFT(vkGPU, &app, num_iter);

			run_time[r] = totTime;
			if (n > 0) {
				if (r == num_runs - 1) {
					double std_error = 0;
					double avg_time = 0;
					for (uint32_t t = 0; t < num_runs; t++) {
						avg_time += run_time[t];
					}
					avg_time /= num_runs;
					for (uint32_t t = 0; t < num_runs; t++) {
						std_error += (run_time[t] - avg_time) * (run_time[t] - avg_time);
					}
					std_error = sqrt(std_error / num_runs);
					uint32_t num_tot_transfers = 0;
					for (uint32_t i = 0; i < configuration.FFTdim; i++)
						num_tot_transfers += app.localFFTPlan->numAxisUploads[i];
					num_tot_transfers *= 4;
					if (file_output)
						fprintf(output, "VkFFT System: %dx%dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %d benchmark: %d bandwidth: %0.1f\n", benchmark_dimensions[n][0], benchmark_dimensions[n][1], benchmark_dimensions[n][2], bufferSize / 1024 / 1024, avg_time, std_error, num_iter, (int)(((double)bufferSize / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);
					printf("VkFFT System: %dx%dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %d benchmark: %d bandwidth: %0.1f\n", benchmark_dimensions[n][0], benchmark_dimensions[n][1], benchmark_dimensions[n][2], bufferSize / 1024 / 1024, avg_time, std_error, num_iter, (int)(((double)bufferSize / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);
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
#endif
			deleteVkFFT(&app);

		}
	}
	free(buffer_input);
	benchmark_result /= (num_benchmark_samples - 1);

	if (file_output) {
		fprintf(output, "Benchmark score VkFFT: %d\n", (int)(benchmark_result));
#if(VKFFT_BACKEND==0)
		fprintf(output, "Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
#endif
	}
	printf("Benchmark score VkFFT: %d\n", (int)(benchmark_result));
#if(VKFFT_BACKEND==0)
	printf("Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
#endif
	return res;
}
uint32_t sample_5(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	uint32_t res = 0;
	if (file_output)
		fprintf(output, "5 - VkFFT FFT + iFFT C2C benchmark 1D batched in single precision, no reshuffling\n");
	printf("5 - VkFFT FFT + iFFT C2C benchmark 1D batched in single precision, no reshuffling\n");
	const int num_runs = 3;
	double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	//memory allocated on the CPU once, makes benchmark completion faster + avoids performance issues connected to frequent allocation/deallocation.
	float* buffer_input = (float*)malloc((uint64_t)4 * 2 * pow(2, 27));
	for (uint64_t i = 0; i < 2 * pow(2, 27); i++) {
		buffer_input[i] = 2 * ((float)rand()) / RAND_MAX - 1.0;
	}
	for (uint32_t n = 0; n < 26; n++) {
		double run_time[num_runs];
		for (uint32_t r = 0; r < num_runs; r++) {
			//Configuration + FFT application .
			VkFFTConfiguration configuration = {};
			VkFFTApplication app = {};
			//FFT + iFFT sample code.
			//Setting up FFT configuration for forward and inverse FFT.
			configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).
			configuration.size[0] = 4 * pow(2, n); //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			if (n == 0) configuration.size[0] = 4096;
			configuration.size[1] = 64 * 32 * pow(2, 16) / configuration.size[0];
			if (configuration.size[1] < 1) configuration.size[1] = 1;
			//configuration.size[1] = (configuration.size[1] > 32768) ? 32768 : configuration.size[1];
			configuration.size[2] = 1;

			configuration.disableReorderFourStep = true;

			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			configuration.device = &vkGPU->device;
#if(VKFFT_BACKEND==0)
			configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
			configuration.fence = &vkGPU->fence;
			configuration.commandPool = &vkGPU->commandPool;
			configuration.physicalDevice = &vkGPU->physicalDevice;
			configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization
#endif
			//Allocate buffer for the input data.
			uint64_t bufferSize = (uint64_t)sizeof(float) * 2 * configuration.size[0] * configuration.size[1] * configuration.size[2];;
#if(VKFFT_BACKEND==0)
			VkBuffer buffer = {};
			VkDeviceMemory bufferDeviceMemory = {};
			allocateFFTBuffer(vkGPU, &buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
			configuration.buffer = &buffer;
#elif(VKFFT_BACKEND==1)
			cuFloatComplex* buffer = 0;
			cudaMalloc((void**)&buffer, bufferSize);
			configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==2)
			hipFloatComplex* buffer = 0;
			hipMalloc((void**)&buffer, bufferSize);
			configuration.buffer = (void**)&buffer;
#endif

			configuration.bufferSize = &bufferSize;


			//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
			/*float* buffer_input = (float*)malloc(bufferSize);

			for (uint32_t k = 0; k < configuration.size[2]; k++) {
				for (uint32_t j = 0; j < configuration.size[1]; j++) {
					for (uint32_t i = 0; i < configuration.size[0]; i++) {
						buffer_input[2 * (i + j * configuration.size[0] + k * (configuration.size[0]) * configuration.size[1])] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						buffer_input[2 * (i + j * configuration.size[0] + k * (configuration.size[0]) * configuration.size[1]) + 1] = 2 * ((float)rand()) / RAND_MAX - 1.0;
					}
				}
			}
			*/
			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
#if(VKFFT_BACKEND==0)
			transferDataFromCPU(vkGPU, buffer_input, &buffer, bufferSize);
#elif(VKFFT_BACKEND==1)
			cudaMemcpy(buffer, buffer_input, bufferSize, cudaMemcpyHostToDevice);
#elif(VKFFT_BACKEND==2)
			hipMemcpy(buffer, buffer_input, bufferSize, hipMemcpyHostToDevice);
#endif
			//free(buffer_input);

			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			res = initializeVkFFT(&app, configuration);
			if (res != 0) return res;

			//Submit FFT+iFFT.
			uint32_t num_iter = ((3 * 4096 * 1024.0 * 1024.0) / bufferSize > 1000) ? 1000 : (3 * 4096 * 1024.0 * 1024.0) / bufferSize;
#if(VKFFT_BACKEND==0)
			if (vkGPU->physicalDeviceProperties.vendorID == 0x8086) num_iter /= 4;
#endif
			if (num_iter == 0) num_iter = 1;
			float totTime = performVulkanFFTiFFT(vkGPU, &app, num_iter);

			run_time[r] = totTime;
			if (n > 0) {
				if (r == num_runs - 1) {
					double std_error = 0;
					double avg_time = 0;
					for (uint32_t t = 0; t < num_runs; t++) {
						avg_time += run_time[t];
					}
					avg_time /= num_runs;
					for (uint32_t t = 0; t < num_runs; t++) {
						std_error += (run_time[t] - avg_time) * (run_time[t] - avg_time);
					}
					std_error = sqrt(std_error / num_runs);
					uint32_t num_tot_transfers = 0;
					for (uint32_t i = 0; i < configuration.FFTdim; i++)
						num_tot_transfers += app.localFFTPlan->numAxisUploads[i];
					num_tot_transfers *= 4;
					if (file_output)
						fprintf(output, "VkFFT System: %d %dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %d benchmark: %d bandwidth: %0.1f\n", (int)log2(configuration.size[0]), configuration.size[0], configuration.size[1], bufferSize / 1024 / 1024, avg_time, std_error, num_iter, (int)(((double)bufferSize / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);

					printf("VkFFT System: %d %dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %d benchmark: %d bandwidth: %0.1f\n", (int)log2(configuration.size[0]), configuration.size[0], configuration.size[1], bufferSize / 1024 / 1024, avg_time, std_error, num_iter, (int)(((double)bufferSize / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);
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
#endif
			deleteVkFFT(&app);

		}
	}
	free(buffer_input);
	benchmark_result /= (26 - 1);

	if (file_output) {
		fprintf(output, "Benchmark score VkFFT: %d\n", (int)(benchmark_result));
#if(VKFFT_BACKEND==0)
		fprintf(output, "Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
#endif
	}
	printf("Benchmark score VkFFT: %d\n", (int)(benchmark_result));
#if(VKFFT_BACKEND==0)
	printf("Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
#endif
	return res;
}
uint32_t sample_6(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	uint32_t res = 0;
	if (file_output)
		fprintf(output, "6 - VkFFT FFT + iFFT R2C/C2R benchmark\n");
	printf("6 - VkFFT FFT + iFFT R2C/C2R benchmark\n");
	const uint32_t num_benchmark_samples = 24;
	const uint32_t num_runs = 3;
	//printf("First %d runs are a warmup\n", num_runs);
	uint32_t benchmark_dimensions[num_benchmark_samples][4] = { {1024, 1024, 1, 2}, {64, 64, 1, 2}, {256, 256, 1, 2}, {1024, 256, 1, 2}, {512, 512, 1, 2}, {1024, 1024, 1, 2},  {4096, 256, 1, 2}, {2048, 1024, 1, 2},{4096, 2048, 1, 2}, {4096, 4096, 1, 2}, {720, 480, 1, 2},{1280, 720, 1, 2},{1920, 1080, 1, 2}, {2560, 1440, 1, 2},{3840, 2160, 1, 2},
																{32, 32, 32, 3}, {64, 64, 64, 3}, {256, 256, 32, 3},  {1024, 256, 32, 3},  {256, 256, 256, 3}, {2048, 1024, 8, 3},  {512, 512, 128, 3}, {2048, 256, 256, 3}, {4096, 512, 8, 3} };
	double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	float* buffer_input = (float*)malloc((uint64_t)4 * 2 * pow(2, 27));
	for (uint64_t i = 0; i < 2 * pow(2, 27); i++) {
		buffer_input[i] = 2 * ((float)rand()) / RAND_MAX - 1.0;
	}
	for (uint32_t n = 0; n < num_benchmark_samples; n++) {
		double run_time[num_runs];
		for (uint32_t r = 0; r < num_runs; r++) {
			//Configuration + FFT application .
			VkFFTConfiguration configuration = {};
			VkFFTApplication app = {};
			//FFT + iFFT sample code.
			//Setting up FFT configuration for forward and inverse FFT.

			configuration.FFTdim = benchmark_dimensions[n][3]; //FFT dimension, 1D, 2D or 3D (default 1).
			configuration.size[0] = benchmark_dimensions[n][0]; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			configuration.size[1] = benchmark_dimensions[n][1];
			configuration.size[2] = benchmark_dimensions[n][2];

			configuration.performR2C = true; //Perform R2C/C2R transform. Can be combined with all other options. Reduces memory requirements by a factor of 2. Requires special input data alignment: for x*y*z system pad x*y plane to (x+2)*y with last 2*y elements reserved, total array dimensions are (x*y+2y)*z. Memory layout after R2C and before C2R can be found on github.
			//configuration.disableMergeSequencesR2C = 1;
			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			configuration.device = &vkGPU->device;
#if(VKFFT_BACKEND==0)
			configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
			configuration.fence = &vkGPU->fence;
			configuration.commandPool = &vkGPU->commandPool;
			configuration.physicalDevice = &vkGPU->physicalDevice;
			configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization
#endif


			//Allocate buffer for the input data.
			uint64_t bufferSize = (uint64_t)sizeof(float) * 2 * (configuration.size[0] / 2 + 1) * configuration.size[1] * configuration.size[2];;
#if(VKFFT_BACKEND==0)
			VkBuffer buffer = {};
			VkDeviceMemory bufferDeviceMemory = {};
			allocateFFTBuffer(vkGPU, &buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
			configuration.buffer = &buffer;
#elif(VKFFT_BACKEND==1)
			cuFloatComplex* buffer = 0;
			cudaMalloc((void**)&buffer, bufferSize);
			configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==2)
			hipFloatComplex* buffer = 0;
			hipMalloc((void**)&buffer, bufferSize);
			configuration.buffer = (void**)&buffer;
#endif

			configuration.bufferSize = &bufferSize;
			//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
			/*float* buffer_input = (float*)malloc(bufferSize);

			for (uint32_t k = 0; k < configuration.size[2]; k++) {
				for (uint32_t j = 0; j < configuration.size[1]; j++) {
					for (uint32_t i = 0; i < configuration.size[0]; i++) {
						buffer_input[i + j * configuration.size[0] + k * (configuration.size[0] + 2) * configuration.size[1]] = i;//[-1,1]
						}
					}
				}
			*/
			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
#if(VKFFT_BACKEND==0)
			transferDataFromCPU(vkGPU, buffer_input, &buffer, bufferSize);
#elif(VKFFT_BACKEND==1)
			cudaMemcpy(buffer, buffer_input, bufferSize, cudaMemcpyHostToDevice);
#elif(VKFFT_BACKEND==2)
			hipMemcpy(buffer, buffer_input, bufferSize, hipMemcpyHostToDevice);
#endif
			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			res = initializeVkFFT(&app, configuration);
			if (res != 0) return res;

			//Submit FFT+iFFT.
			uint32_t num_iter = ((4096.0 * 1024.0 * 1024.0) / bufferSize > 1000) ? 1000 : (4096.0 * 1024.0 * 1024.0) / bufferSize;
#if(VKFFT_BACKEND==0)
			if (vkGPU->physicalDeviceProperties.vendorID == 0x8086) num_iter /= 4;
#endif
			if (num_iter == 0) num_iter = 1;

			//num_iter *= 5; //makes result more smooth, takes longer time
			float totTime = performVulkanFFTiFFT(vkGPU, &app, num_iter);
			//float* buffer_output = (float*)malloc(bufferSize);
			run_time[r] = totTime;
			if (n > 0) {
				if (r == num_runs - 1) {
					double std_error = 0;
					double avg_time = 0;
					for (uint32_t t = 0; t < num_runs; t++) {
						avg_time += run_time[t];
					}
					avg_time /= num_runs;
					for (uint32_t t = 0; t < num_runs; t++) {
						std_error += (run_time[t] - avg_time) * (run_time[t] - avg_time);
					}
					std_error = sqrt(std_error / num_runs);
					uint32_t num_tot_transfers = 0;
					for (uint32_t i = 0; i < configuration.FFTdim; i++)
						num_tot_transfers += app.localFFTPlan->numAxisUploads[i];
					num_tot_transfers *= 4;
					if (file_output)
						fprintf(output, "VkFFT System: %dx%dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %d benchmark: %d bandwidth: %0.1f\n", benchmark_dimensions[n][0], benchmark_dimensions[n][1], benchmark_dimensions[n][2], bufferSize / 1024 / 1024, avg_time, std_error, num_iter, (int)(((double)bufferSize / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);
					printf("VkFFT System: %dx%dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %d benchmark: %d bandwidth: %0.1f\n", benchmark_dimensions[n][0], benchmark_dimensions[n][1], benchmark_dimensions[n][2], bufferSize / 1024 / 1024, avg_time, std_error, num_iter, (int)(((double)bufferSize / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);
					benchmark_result += ((double)bufferSize / 1024) / avg_time;
				}

			}
			//printf("Benchmark score: %f\n", ((double)bufferSize / 1024) / totTime);
			//Transfer data from GPU using staging buffer.
			//transferDataToCPU(buffer_output, &buffer, bufferSize);
			//Print data, if needed.
			/*
				for (uint32_t k = 0; k < configuration.size[2]; k++) {
					for (uint32_t j = 0; j < configuration.size[1]; j++) {
						for (uint32_t i = 0; i < configuration.size[0]; i++) {
							printf("%.6f ", buffer_output[i + j * configuration.size[0] + k * (configuration.size[0] + 2) * configuration.size[1] + v * (configuration.size[0] + 2) * configuration.size[1] * configuration.size[2]]);
						}
						std::cout << "\n";
					}
				}
			*/
			//free(buffer_input);
			//free(buffer_output);
#if(VKFFT_BACKEND==0)
			vkDestroyBuffer(vkGPU->device, buffer, NULL);
			vkFreeMemory(vkGPU->device, bufferDeviceMemory, NULL);
#elif(VKFFT_BACKEND==1)
			cudaFree(buffer);
#elif(VKFFT_BACKEND==2)
			hipFree(buffer);
#endif
			deleteVkFFT(&app);

		}
	}
	free(buffer_input);
	benchmark_result /= ((num_benchmark_samples - 1));
	if (file_output) {
		fprintf(output, "Benchmark score VkFFT: %d\n", (int)(benchmark_result));
#if(VKFFT_BACKEND==0)
		fprintf(output, "Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
#endif
	}
	printf("Benchmark score VkFFT: %d\n", (int)(benchmark_result));
#if(VKFFT_BACKEND==0)
	printf("Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
#endif
	return res;
}
uint32_t sample_7(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	uint32_t res = 0;
	if (file_output)
		fprintf(output, "7 - VkFFT convolution example with identitiy kernel\n");
	printf("7 - VkFFT convolution example with identitiy kernel\n");
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
	configuration.disableReorderFourStep = true;//Set to false if you use convolution routine. Data reordering is not needed - no additional buffer - less memory usage

	//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
	configuration.device = &vkGPU->device;
#if(VKFFT_BACKEND==0)
	configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
	configuration.fence = &vkGPU->fence;
	configuration.commandPool = &vkGPU->commandPool;
	configuration.physicalDevice = &vkGPU->physicalDevice;
	configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization
#endif
	//In this example, we perform a convolution for a real vectorfield (3vector) with a symmetric kernel (6 values). We use configuration to initialize convolution kernel first from real data, then we create convolution_configuration for convolution. The buffer object from configuration is passed to convolution_configuration as kernel object.
	//1. Kernel forward FFT.
	uint64_t kernelSize = ((uint64_t)configuration.coordinateFeatures) * sizeof(float) * 2 * (configuration.size[0]) * configuration.size[1] * configuration.size[2];;

#if(VKFFT_BACKEND==0)
	VkBuffer kernel = {};
	VkDeviceMemory kernelDeviceMemory = {};
	allocateFFTBuffer(vkGPU, &kernel, &kernelDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, kernelSize);
	configuration.buffer = &kernel;
#elif(VKFFT_BACKEND==1)
	cuFloatComplex* kernel = 0;
	cudaMalloc((void**)&kernel, kernelSize);
	configuration.buffer = (void**)&kernel;
#elif(VKFFT_BACKEND==2)
	hipFloatComplex* kernel = 0;
	hipMalloc((void**)&kernel, kernelSize);
	configuration.buffer = (void**)&kernel;
#endif

	configuration.bufferSize = &kernelSize;

	if (file_output)
		fprintf(output, "Total memory needed for kernel: %d MB\n", kernelSize / 1024 / 1024);
	printf("Total memory needed for kernel: %d MB\n", kernelSize / 1024 / 1024);
	//Fill kernel on CPU.
	float* kernel_input = (float*)malloc(kernelSize);
	for (uint32_t v = 0; v < configuration.coordinateFeatures; v++) {
		for (uint32_t k = 0; k < configuration.size[2]; k++) {
			for (uint32_t j = 0; j < configuration.size[1]; j++) {

				//for (uint32_t i = 0; i < configuration.size[0]; i++) {
				//	kernel_input[i + j * configuration.size[0] + k * (configuration.size[0] + 2) * configuration.size[1] + v * (configuration.size[0] + 2) * configuration.size[1] * configuration.size[2]] = 1;

				//Below is the test identity kernel for 3x3 nonsymmetric FFT
				for (uint32_t i = 0; i < configuration.size[0]; i++) {
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
	transferDataFromCPU(vkGPU, kernel_input, &kernel, kernelSize);
#elif(VKFFT_BACKEND==1)
	cudaMemcpy(kernel, kernel_input, kernelSize, cudaMemcpyHostToDevice);
#elif(VKFFT_BACKEND==2)
	hipMemcpy(kernel, kernel_input, kernelSize, hipMemcpyHostToDevice);
#endif

	//Initialize application responsible for the kernel. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
	res = initializeVkFFT(&app_kernel, configuration);
	if (res != 0) return res;
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
#endif	

	//Allocate separate buffer for the input data.
	uint64_t bufferSize = ((uint64_t)convolution_configuration.coordinateFeatures) * sizeof(float) * 2 * (convolution_configuration.size[0]) * convolution_configuration.size[1] * convolution_configuration.size[2];;
#if(VKFFT_BACKEND==0)
	VkBuffer buffer = {};
	VkDeviceMemory bufferDeviceMemory = {};
	allocateFFTBuffer(vkGPU, &buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
	convolution_configuration.buffer = &buffer;
#elif(VKFFT_BACKEND==1)
	cuFloatComplex* buffer = 0;
	cudaMalloc((void**)&buffer, bufferSize);
	convolution_configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==2)
	hipFloatComplex* buffer = 0;
	hipMalloc((void**)&buffer, bufferSize);
	convolution_configuration.buffer = (void**)&buffer;
#endif

	convolution_configuration.bufferSize = &bufferSize;
	convolution_configuration.kernelSize = &kernelSize;

	if (file_output)
		fprintf(output, "Total memory needed for buffer: %d MB\n", bufferSize / 1024 / 1024);
	printf("Total memory needed for buffer: %d MB\n", bufferSize / 1024 / 1024);
	//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
	float* buffer_input = (float*)malloc(bufferSize);

	for (uint32_t v = 0; v < convolution_configuration.coordinateFeatures; v++) {
		for (uint32_t k = 0; k < convolution_configuration.size[2]; k++) {
			for (uint32_t j = 0; j < convolution_configuration.size[1]; j++) {
				for (uint32_t i = 0; i < convolution_configuration.size[0]; i++) {
					buffer_input[2 * (i + j * convolution_configuration.size[0] + k * (convolution_configuration.size[0]) * convolution_configuration.size[1] + v * (convolution_configuration.size[0]) * convolution_configuration.size[1] * convolution_configuration.size[2])] = i % 8 - 3.5;
					buffer_input[2 * (i + j * convolution_configuration.size[0] + k * (convolution_configuration.size[0]) * convolution_configuration.size[1] + v * (convolution_configuration.size[0]) * convolution_configuration.size[1] * convolution_configuration.size[2]) + 1] = i % 4 - 1.5;
				}
			}
		}
	}
	//Transfer data to GPU using staging buffer.
#if(VKFFT_BACKEND==0)
	transferDataFromCPU(vkGPU, buffer_input, &buffer, bufferSize);
#elif(VKFFT_BACKEND==1)
	cudaMemcpy(buffer, buffer_input, bufferSize, cudaMemcpyHostToDevice);
#elif(VKFFT_BACKEND==2)
	hipMemcpy(buffer, buffer_input, bufferSize, hipMemcpyHostToDevice);
#endif

	//Initialize application responsible for the convolution.
	res = initializeVkFFT(&app_convolution, convolution_configuration);
	if (res != 0) return res;
	//Sample forward FFT command buffer allocation + execution performed on kernel. FFT can also be appended to user defined command buffers.
	performVulkanFFT(vkGPU, &app_convolution, -1, 1);
	//The kernel has been trasnformed.

	float* buffer_output = (float*)malloc(bufferSize);
	//Transfer data from GPU using staging buffer.
#if(VKFFT_BACKEND==0)
	transferDataToCPU(vkGPU, buffer_output, &buffer, bufferSize);
#elif(VKFFT_BACKEND==1)
	cudaMemcpy(buffer_output, buffer, bufferSize, cudaMemcpyDeviceToHost);
#elif(VKFFT_BACKEND==2)
	hipMemcpy(buffer_output, buffer, bufferSize, hipMemcpyDeviceToHost);
#endif
	//Print data, if needed.
	for (uint32_t v = 0; v < convolution_configuration.coordinateFeatures; v++) {
		if (file_output)
			fprintf(output, "\ncoordinate: %d\n\n", v);
		printf("\ncoordinate: %d\n\n", v);
		for (uint32_t k = 0; k < convolution_configuration.size[2]; k++) {
			for (uint32_t j = 0; j < convolution_configuration.size[1]; j++) {
				for (uint32_t i = 0; i < convolution_configuration.size[0]; i++) {
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
#endif	
	deleteVkFFT(&app_kernel);
	deleteVkFFT(&app_convolution);
	return res;
}
uint32_t sample_8(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	uint32_t res = 0;
	if (file_output)
		fprintf(output, "8 - VkFFT zeropadding convolution example with identitiy kernel\n");
	printf("8 - VkFFT zeropadding convolution example with identitiy kernel\n");
	//Configuration + FFT application.
	VkFFTConfiguration configuration = {};
	VkFFTConfiguration convolution_configuration = {};
	VkFFTApplication app_convolution = {};
	VkFFTApplication app_kernel = {};
	//Zeropadding Convolution sample code
	//Setting up FFT configuration. FFT is performed in-place with no performance loss. 

	configuration.FFTdim = 3; //FFT dimension, 1D, 2D or 3D (default 1).
	configuration.size[0] = 32; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z. 
	configuration.size[1] = 32;
	configuration.size[2] = 32;

	configuration.normalize = 1;//normalize iFFT
	configuration.performZeropadding[0] = true; //Perform padding with zeros on GPU. Still need to properly align input data (no need to fill padding area with meaningful data) but this will increase performance due to the lower amount of the memory reads/writes and omitting sequences only consisting of zeros.
	configuration.performZeropadding[1] = true;
	configuration.performZeropadding[2] = true;
	configuration.fft_zeropad_left[0] = ceil(configuration.size[0] / 2.0);
	configuration.fft_zeropad_right[0] = configuration.size[0];
	configuration.fft_zeropad_left[1] = ceil(configuration.size[1] / 2.0);
	configuration.fft_zeropad_right[1] = configuration.size[1];
	configuration.fft_zeropad_left[2] = ceil(configuration.size[2] / 2.0);
	configuration.fft_zeropad_right[2] = configuration.size[2];
	configuration.kernelConvolution = true; //specify if this plan is used to create kernel for convolution
	configuration.performR2C = true; //Perform R2C/C2R transform. Can be combined with all other options. Reduces memory requirements by a factor of 2. Requires special input data alignment: for x*y*z system pad x*y plane to (x+2)*y with last 2*y elements reserved, total array dimensions are (x*y+2y)*z. Memory layout after R2C and before C2R can be found on github.
	configuration.coordinateFeatures = 9; //Specify dimensionality of the input feature vector (default 1). Each component is stored not as a vector, but as a separate system and padded on it's own according to other options (i.e. for x*y system of 3-vector, first x*y elements correspond to the first dimension, then goes x*y for the second, etc).
	//coordinateFeatures number is an important constant for convolution. If we perform 1x1 convolution, it is equal to number of features, but matrixConvolution should be equal to 1. For matrix convolution, it must be equal to matrixConvolution parameter. If we perform 2x2 convolution, it is equal to 3 for symmetric kernel (stored as xx, xy, yy) and 4 for nonsymmetric (stored as xx, xy, yx, yy). Similarly, 6 (stored as xx, xy, xz, yy, yz, zz) and 9 (stored as xx, xy, xz, yx, yy, yz, zx, zy, zz) for 3x3 convolutions. 

	configuration.disableReorderFourStep = true;//Set to false if you use convolution routine. Data reordering is not needed - no additional buffer - less memory usage

	//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
	configuration.device = &vkGPU->device;
#if(VKFFT_BACKEND==0)
	configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
	configuration.fence = &vkGPU->fence;
	configuration.commandPool = &vkGPU->commandPool;
	configuration.physicalDevice = &vkGPU->physicalDevice;
	configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization
#endif
	//In this example, we perform a convolution for a real vectorfield (3vector) with a symmetric kernel (6 values). We use configuration to initialize convolution kernel first from real data, then we create convolution_configuration for convolution. The buffer object from configuration is passed to convolution_configuration as kernel object.
	//1. Kernel forward FFT.
	uint64_t kernelSize = ((uint64_t)configuration.coordinateFeatures) * sizeof(float) * 2 * (configuration.size[0] / 2 + 1) * configuration.size[1] * configuration.size[2];
#if(VKFFT_BACKEND==0)
	VkBuffer kernel = {};
	VkDeviceMemory kernelDeviceMemory = {};
	allocateFFTBuffer(vkGPU, &kernel, &kernelDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, kernelSize);
	configuration.buffer = &kernel;
#elif(VKFFT_BACKEND==1)
	cuFloatComplex* kernel = 0;
	cudaMalloc((void**)&kernel, kernelSize);
	configuration.buffer = (void**)&kernel;
#elif(VKFFT_BACKEND==2)
	hipFloatComplex* kernel = 0;
	hipMalloc((void**)&kernel, kernelSize);
	configuration.buffer = (void**)&kernel;
#endif

	configuration.bufferSize = &kernelSize;

	if (file_output)
		fprintf(output, "Total memory needed for kernel: %d MB\n", kernelSize / 1024 / 1024);
	printf("Total memory needed for kernel: %d MB\n", kernelSize / 1024 / 1024);

	//Fill kernel on CPU.
	float* kernel_input = (float*)malloc(kernelSize);
	for (uint32_t v = 0; v < configuration.coordinateFeatures; v++) {
		for (uint32_t k = 0; k < configuration.size[2]; k++) {
			for (uint32_t j = 0; j < configuration.size[1]; j++) {

				//for (uint32_t i = 0; i < configuration.size[0]; i++) {
				//	kernel_input[i + j * configuration.size[0] + k * (configuration.size[0] + 2) * configuration.size[1] + v * (configuration.size[0] + 2) * configuration.size[1] * configuration.size[2]] = 1;

				//Below is the test identity kernel for 3x3 nonsymmetric FFT
				for (uint32_t i = 0; i < configuration.size[0] / 2 + 1; i++) {
					if ((v == 0) || (v == 4) || (v == 8))

						kernel_input[2 * i + j * (configuration.size[0] + 2) + k * (configuration.size[0] + 2) * configuration.size[1] + v * (configuration.size[0] + 2) * configuration.size[1] * configuration.size[2]] = 1;

					else
						kernel_input[2 * i + j * (configuration.size[0] + 2) + k * (configuration.size[0] + 2) * configuration.size[1] + v * (configuration.size[0] + 2) * configuration.size[1] * configuration.size[2]] = 0;
					kernel_input[2 * i + 1 + j * (configuration.size[0] + 2) + k * (configuration.size[0] + 2) * configuration.size[1] + v * (configuration.size[0] + 2) * configuration.size[1] * configuration.size[2]] = 0;

				}
			}
		}
	}
	//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
#if(VKFFT_BACKEND==0)
	transferDataFromCPU(vkGPU, kernel_input, &kernel, kernelSize);
#elif(VKFFT_BACKEND==1)
	cudaMemcpy(kernel, kernel_input, kernelSize, cudaMemcpyHostToDevice);
#elif(VKFFT_BACKEND==2)
	hipMemcpy(kernel, kernel_input, kernelSize, hipMemcpyHostToDevice);
#endif
	//Initialize application responsible for the kernel. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
	res = initializeVkFFT(&app_kernel, configuration);
	if (res != 0) return res;
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
	convolution_configuration.matrixConvolution = 3; //we do matrix convolution, so kernel is 9 numbers (3x3), but vector dimension is 3
	convolution_configuration.coordinateFeatures = 3;

#if(VKFFT_BACKEND==0)
	convolution_configuration.kernel = &kernel;
#elif(VKFFT_BACKEND==1)
	convolution_configuration.kernel = (void**)&kernel;
#elif(VKFFT_BACKEND==2)
	convolution_configuration.kernel = (void**)&kernel;
#endif	

	//Allocate separate buffer for the input data.
	uint64_t bufferSize = ((uint64_t)convolution_configuration.coordinateFeatures) * sizeof(float) * 2 * (convolution_configuration.size[0] / 2 + 1) * convolution_configuration.size[1] * convolution_configuration.size[2];;

#if(VKFFT_BACKEND==0)
	VkBuffer buffer = {};
	VkDeviceMemory bufferDeviceMemory = {};
	allocateFFTBuffer(vkGPU, &buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
	convolution_configuration.buffer = &buffer;
#elif(VKFFT_BACKEND==1)
	cuFloatComplex* buffer = 0;
	cudaMalloc((void**)&buffer, bufferSize);
	convolution_configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==2)
	hipFloatComplex* buffer = 0;
	hipMalloc((void**)&buffer, bufferSize);
	convolution_configuration.buffer = (void**)&buffer;
#endif

	convolution_configuration.bufferSize = &bufferSize;
	convolution_configuration.kernelSize = &kernelSize;

	if (file_output)
		fprintf(output, "Total memory needed for buffer: %d MB\n", bufferSize / 1024 / 1024);
	printf("Total memory needed for buffer: %d MB\n", bufferSize / 1024 / 1024);
	//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
	float* buffer_input = (float*)malloc(bufferSize);

	for (uint32_t v = 0; v < convolution_configuration.coordinateFeatures; v++) {
		for (uint32_t k = 0; k < ceil(convolution_configuration.size[2] / 2.0); k++) {
			for (uint32_t j = 0; j < ceil(convolution_configuration.size[1] / 2.0); j++) {
				for (uint32_t i = 0; i < ceil(convolution_configuration.size[0] / 2.0); i++) {
					buffer_input[i + j * (convolution_configuration.size[0] + 2) + k * (convolution_configuration.size[0] + 2) * convolution_configuration.size[1] + v * (convolution_configuration.size[0] + 2) * convolution_configuration.size[1] * convolution_configuration.size[2]] = i;
				}
			}
		}
	}
	//Transfer data to GPU using staging buffer.
#if(VKFFT_BACKEND==0)
	transferDataFromCPU(vkGPU, buffer_input, &buffer, bufferSize);
#elif(VKFFT_BACKEND==1)
	cudaMemcpy(buffer, buffer_input, bufferSize, cudaMemcpyHostToDevice);
#elif(VKFFT_BACKEND==2)
	hipMemcpy(buffer, buffer_input, bufferSize, hipMemcpyHostToDevice);
#endif
	//Initialize application responsible for the convolution.
	res = initializeVkFFT(&app_convolution, convolution_configuration);
	if (res != 0) return res;
	//Sample forward FFT command buffer allocation + execution performed on kernel. FFT can also be appended to user defined command buffers.
	performVulkanFFT(vkGPU, &app_convolution, -1, 1);
	//The kernel has been trasnformed.

	float* buffer_output = (float*)malloc(bufferSize);
	//Transfer data from GPU using staging buffer.
#if(VKFFT_BACKEND==0)
	transferDataToCPU(vkGPU, buffer_output, &buffer, bufferSize);
#elif(VKFFT_BACKEND==1)
	cudaMemcpy(buffer_output, buffer, bufferSize, cudaMemcpyDeviceToHost);
#elif(VKFFT_BACKEND==2)
	hipMemcpy(buffer_output, buffer, bufferSize, hipMemcpyDeviceToHost);
#endif

	//Print data, if needed.
	for (uint32_t v = 0; v < convolution_configuration.coordinateFeatures; v++) {
		if (file_output)
			fprintf(output, "\ncoordinate: %d\n\n", v);
		printf("\ncoordinate: %d\n\n", v);
		for (uint32_t k = 0; k < ceil(convolution_configuration.size[2] / 2.0); k++) {
			for (uint32_t j = 0; j < ceil(convolution_configuration.size[1] / 2.0); j++) {
				for (uint32_t i = 0; i < ceil(convolution_configuration.size[0] / 2.0); i++) {
					if (file_output)
						fprintf(output, "%.6f ", buffer_output[i + j * (convolution_configuration.size[0] + 2) + k * (convolution_configuration.size[0] + 2) * convolution_configuration.size[1] + v * (convolution_configuration.size[0] + 2) * convolution_configuration.size[1] * convolution_configuration.size[2]]);
					printf("%.6f ", buffer_output[i + j * (convolution_configuration.size[0] + 2) + k * (convolution_configuration.size[0] + 2) * convolution_configuration.size[1] + v * (convolution_configuration.size[0] + 2) * convolution_configuration.size[1] * convolution_configuration.size[2]]);
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
#endif	
	deleteVkFFT(&app_kernel);
	deleteVkFFT(&app_convolution);
	return res;
}
uint32_t sample_9(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	uint32_t res = 0;
	if (file_output)
		fprintf(output, "9 - VkFFT batched convolution example with identitiy kernel\n");
	printf("9 - VkFFT batched convolution example with identitiy kernel\n");
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
	configuration.disableReorderFourStep = true;//Set to false if you use convolution routine. Data reordering is not needed - no additional buffer - less memory usage

	configuration.numberBatches = 2;
	//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
	configuration.device = &vkGPU->device;
#if(VKFFT_BACKEND==0)
	configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
	configuration.fence = &vkGPU->fence;
	configuration.commandPool = &vkGPU->commandPool;
	configuration.physicalDevice = &vkGPU->physicalDevice;
	configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization
#endif
	//In this example, we perform a convolution for a real vectorfield (3vector) with a symmetric kernel (6 values). We use configuration to initialize convolution kernel first from real data, then we create convolution_configuration for convolution. The buffer object from configuration is passed to convolution_configuration as kernel object.
	//1. Kernel forward FFT.
	uint64_t kernelSize = ((uint64_t)configuration.numberBatches) * configuration.coordinateFeatures * sizeof(float) * 2 * (configuration.size[0] / 2 + 1) * configuration.size[1] * configuration.size[2];;

#if(VKFFT_BACKEND==0)
	VkBuffer kernel = {};
	VkDeviceMemory kernelDeviceMemory = {};
	allocateFFTBuffer(vkGPU, &kernel, &kernelDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, kernelSize);
	configuration.buffer = &kernel;
#elif(VKFFT_BACKEND==1)
	cuFloatComplex* kernel = 0;
	cudaMalloc((void**)&kernel, kernelSize);
	configuration.buffer = (void**)&kernel;
#elif(VKFFT_BACKEND==2)
	hipFloatComplex* kernel = 0;
	hipMalloc((void**)&kernel, kernelSize);
	configuration.buffer = (void**)&kernel;
#endif

	configuration.bufferSize = &kernelSize;

	if (file_output)
		fprintf(output, "Total memory needed for kernel: %d MB\n", kernelSize / 1024 / 1024);
	printf("Total memory needed for kernel: %d MB\n", kernelSize / 1024 / 1024);

	//Fill kernel on CPU.
	float* kernel_input = (float*)malloc(kernelSize);
	for (uint32_t f = 0; f < configuration.numberBatches; f++) {
		for (uint32_t v = 0; v < configuration.coordinateFeatures; v++) {
			for (uint32_t k = 0; k < configuration.size[2]; k++) {
				for (uint32_t j = 0; j < configuration.size[1]; j++) {

					//Below is the test identity kernel for 1x1 nonsymmetric FFT, multiplied by (f * configuration.coordinateFeatures + v + 1);
					for (uint32_t i = 0; i < configuration.size[0] / 2 + 1; i++) {

						kernel_input[2 * i + j * (configuration.size[0] + 2) + k * (configuration.size[0] + 2) * configuration.size[1] + v * (configuration.size[0] + 2) * configuration.size[1] * configuration.size[2] + f * configuration.coordinateFeatures * (configuration.size[0] + 2) * configuration.size[1] * configuration.size[2]] = f * configuration.coordinateFeatures + v + 1;
						kernel_input[2 * i + 1 + j * (configuration.size[0] + 2) + k * (configuration.size[0] + 2) * configuration.size[1] + v * (configuration.size[0] + 2) * configuration.size[1] * configuration.size[2] + f * configuration.coordinateFeatures * (configuration.size[0] + 2) * configuration.size[1] * configuration.size[2]] = 0;

					}
				}
			}
		}
	}
	//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
#if(VKFFT_BACKEND==0)
	transferDataFromCPU(vkGPU, kernel_input, &kernel, kernelSize);
#elif(VKFFT_BACKEND==1)
	cudaMemcpy(kernel, kernel_input, kernelSize, cudaMemcpyHostToDevice);
#elif(VKFFT_BACKEND==2)
	hipMemcpy(kernel, kernel_input, kernelSize, hipMemcpyHostToDevice);
#endif
	//Initialize application responsible for the kernel. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
	res = initializeVkFFT(&app_kernel, configuration);
	if (res != 0) return res;
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
	allocateFFTBuffer(vkGPU, &inputBuffer, &inputBufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, inputBufferSize);
	allocateFFTBuffer(vkGPU, &buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
	convolution_configuration.inputBuffer = &inputBuffer;
	convolution_configuration.buffer = &buffer;
#elif(VKFFT_BACKEND==1)
	cuFloatComplex* inputBuffer = 0;
	cuFloatComplex* buffer = 0;
	cudaMalloc((void**)&inputBuffer, inputBufferSize);
	cudaMalloc((void**)&buffer, bufferSize);
	convolution_configuration.inputBuffer = (void**)&inputBuffer;
	convolution_configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==2)
	hipFloatComplex* inputBuffer = 0;
	hipFloatComplex* buffer = 0;
	hipMalloc((void**)&inputBuffer, inputBufferSize);
	hipMalloc((void**)&buffer, bufferSize);
	convolution_configuration.inputBuffer = (void**)&inputBuffer;
	convolution_configuration.buffer = (void**)&buffer;
#endif

	convolution_configuration.inputBufferSize = &inputBufferSize;
	convolution_configuration.bufferSize = &bufferSize;


	if (file_output)
		fprintf(output, "Total memory needed for buffer: %d MB\n", bufferSize / 1024 / 1024);
	printf("Total memory needed for buffer: %d MB\n", bufferSize / 1024 / 1024);
	//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
	float* buffer_input = (float*)malloc(inputBufferSize);

	for (uint32_t v = 0; v < convolution_configuration.coordinateFeatures; v++) {
		for (uint32_t k = 0; k < convolution_configuration.size[2]; k++) {
			for (uint32_t j = 0; j < convolution_configuration.size[1]; j++) {
				for (uint32_t i = 0; i < convolution_configuration.size[0]; i++) {
					buffer_input[i + j * (convolution_configuration.size[0] + 2) + k * (convolution_configuration.size[0] + 2) * convolution_configuration.size[1] + v * (convolution_configuration.size[0] + 2) * convolution_configuration.size[1] * convolution_configuration.size[2]] = 1;
				}
			}
		}
	}
	//Transfer data to GPU using staging buffer.
#if(VKFFT_BACKEND==0)
	transferDataFromCPU(vkGPU, buffer_input, &inputBuffer, inputBufferSize);
#elif(VKFFT_BACKEND==1)
	cudaMemcpy(inputBuffer, buffer_input, inputBufferSize, cudaMemcpyHostToDevice);
#elif(VKFFT_BACKEND==2)
	hipMemcpy(inputBuffer, buffer_input, inputBufferSize, hipMemcpyHostToDevice);
#endif

	//Initialize application responsible for the convolution.
	res = initializeVkFFT(&app_convolution, convolution_configuration);
	if (res != 0) return res;
	//Sample forward FFT command buffer allocation + execution performed on kernel. FFT can also be appended to user defined command buffers.
	performVulkanFFT(vkGPU, &app_convolution, -1, 1);
	//The kernel has been trasnformed.

	float* buffer_output = (float*)malloc(bufferSize);
	//Transfer data from GPU using staging buffer.
#if(VKFFT_BACKEND==0)
	transferDataToCPU(vkGPU, buffer_output, &buffer, bufferSize);
#elif(VKFFT_BACKEND==1)
	cudaMemcpy(buffer_output, buffer, bufferSize, cudaMemcpyDeviceToHost);
#elif(VKFFT_BACKEND==2)
	hipMemcpy(buffer_output, buffer, bufferSize, hipMemcpyDeviceToHost);
#endif

	//Print data, if needed.
	for (uint32_t f = 0; f < convolution_configuration.numberKernels; f++) {
		if (file_output)
			fprintf(output, "\nKernel id: %d\n\n", f);
		printf("\nKernel id: %d\n\n", f);
		for (uint32_t v = 0; v < convolution_configuration.coordinateFeatures; v++) {
			if (file_output)
				fprintf(output, "\ncoordinate: %d\n\n", v);
			printf("\ncoordinate: %d\n\n", v);
			for (uint32_t k = 0; k < convolution_configuration.size[2]; k++) {
				for (uint32_t j = 0; j < convolution_configuration.size[1]; j++) {
					for (uint32_t i = 0; i < convolution_configuration.size[0]; i++) {
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
#endif	
	deleteVkFFT(&app_kernel);
	deleteVkFFT(&app_convolution);
	return res;
}
#if(VKFFT_BACKEND==0)
uint32_t sample_10(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	uint32_t res = 0;
	if (file_output)
		fprintf(output, "10 - VkFFT FFT + iFFT C2C benchmark 1D batched in single precision, multiple buffer split FFT\n");
	printf("10 - VkFFT FFT + iFFT C2C benchmark 1D batched in single precision, multiple buffer split FFT\n");
	const int num_runs = 3;
	double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	//memory allocated on the CPU once, makes benchmark completion faster + avoids performance issues connected to frequent allocation/deallocation.
	float* buffer_input = (float*)malloc((uint64_t)4 * 2 * pow(2, 27));
	for (uint64_t i = 0; i < 2 * pow(2, 27); i++) {
		buffer_input[i] = 2 * ((float)rand()) / RAND_MAX - 1.0;
	}
	for (uint32_t n = 0; n < 26; n++) {
		double run_time[num_runs];
		for (uint32_t r = 0; r < num_runs; r++) {
			//Configuration + FFT application .
			VkFFTConfiguration configuration = {};
			VkFFTApplication app = {};
			//FFT + iFFT sample code.
			//Setting up FFT configuration for forward and inverse FFT.
			configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).
			configuration.size[0] = 4 * pow(2, n); //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			if (n == 0) configuration.size[0] = 4096;
			configuration.size[1] = 64 * 32 * pow(2, 16) / configuration.size[0];
			if (configuration.size[1] < 1) configuration.size[1] = 1;
			//configuration.size[1] = (configuration.size[1] > 32768) ? 32768 : configuration.size[1];
			configuration.size[2] = 1;
			uint32_t numBuf = 4;

			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			configuration.device = &vkGPU->device;
			configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
			configuration.fence = &vkGPU->fence;
			configuration.commandPool = &vkGPU->commandPool;
			configuration.physicalDevice = &vkGPU->physicalDevice;
			configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization

			//Allocate buffers for the input data. - we use 4 in this example
			uint64_t* bufferSize = (uint64_t*)malloc(sizeof(uint64_t) * numBuf);
			for (uint32_t i = 0; i < numBuf; i++) {
				bufferSize[i] = {};
				bufferSize[i] = (uint64_t)sizeof(float) * 2 * configuration.size[0] * configuration.size[1] * configuration.size[2] / numBuf;
			}

			VkBuffer* buffer = (VkBuffer*)malloc(numBuf * sizeof(VkBuffer));
			VkDeviceMemory* bufferDeviceMemory = (VkDeviceMemory*)malloc(numBuf * sizeof(VkDeviceMemory));

			configuration.userTempBuffer = true; //user allocated temp buffer to reshuffle Four step FFT
			VkBuffer* tempBuffer = (VkBuffer*)malloc(numBuf * sizeof(VkBuffer));
			VkDeviceMemory* tempBufferDeviceMemory = (VkDeviceMemory*)malloc(numBuf * sizeof(VkDeviceMemory));

			for (uint32_t i = 0; i < numBuf; i++) {
				buffer[i] = {};
				tempBuffer[i] = {};
				bufferDeviceMemory[i] = {};
				tempBufferDeviceMemory[i] = {};
				allocateFFTBuffer(vkGPU, &buffer[i], &bufferDeviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize[i]);
				allocateFFTBuffer(vkGPU, &tempBuffer[i], &tempBufferDeviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize[i]);

			}

			configuration.bufferNum = numBuf;
			configuration.tempBufferNum = numBuf;

			configuration.buffer = buffer;
			configuration.tempBuffer = tempBuffer;

			configuration.bufferSize = bufferSize;
			configuration.tempBufferSize = bufferSize;

			//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
			/*float* buffer_input = (float*)malloc(bufferSize);
			for (uint32_t k = 0; k < configuration.size[2]; k++) {
				for (uint32_t j = 0; j < configuration.size[1]; j++) {
					for (uint32_t i = 0; i < configuration.size[0]; i++) {
						buffer_input[2 * (i + j * configuration.size[0] + k * (configuration.size[0]) * configuration.size[1] + v * (configuration.size[0]) * configuration.size[1] * configuration.size[2])] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						buffer_input[2 * (i + j * configuration.size[0] + k * (configuration.size[0]) * configuration.size[1] + v * (configuration.size[0]) * configuration.size[1] * configuration.size[2]) + 1] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						}
					}
				}
			*/
			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
			uint64_t shift = 0;
			for (uint32_t i = 0; i < numBuf; i++) {
				transferDataFromCPU(vkGPU, (buffer_input + shift / sizeof(float)), &buffer[i], bufferSize[i]);
				shift += bufferSize[i];
			}

			//free(buffer_input);

			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			res = initializeVkFFT(&app, configuration);
			if (res != 0) return res;

			//Submit FFT+iFFT.
			uint32_t num_iter = ((4096 * 1024.0 * 1024.0) / (numBuf * bufferSize[0]) > 1000) ? 1000 : (4096 * 1024.0 * 1024.0) / (numBuf * bufferSize[0]);
			if (vkGPU->physicalDeviceProperties.vendorID == 0x8086) num_iter /= 4;
			if (num_iter == 0) num_iter = 1;
			if (vkGPU->physicalDeviceProperties.vendorID != 0x8086) num_iter *= 5;
			float totTime = performVulkanFFTiFFT(vkGPU, &app, num_iter);

			run_time[r] = totTime;
			if (n > 0) {
				if (r == num_runs - 1) {
					double std_error = 0;
					double avg_time = 0;
					for (uint32_t t = 0; t < num_runs; t++) {
						avg_time += run_time[t];
					}
					avg_time /= num_runs;
					for (uint32_t t = 0; t < num_runs; t++) {
						std_error += (run_time[t] - avg_time) * (run_time[t] - avg_time);
					}
					std_error = sqrt(std_error / num_runs);
					uint32_t num_tot_transfers = 0;
					for (uint32_t i = 0; i < configuration.FFTdim; i++)
						num_tot_transfers += app.localFFTPlan->numAxisUploads[i];
					num_tot_transfers *= 4;
					if (file_output)
						fprintf(output, "VkFFT System: %d %dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %d benchmark: %d bandwidth: %0.1f\n", (int)log2(configuration.size[0]), configuration.size[0], configuration.size[1], (numBuf * bufferSize[0]) / 1024 / 1024, avg_time, std_error, num_iter, (int)(((double)(numBuf * bufferSize[0]) / 1024) / avg_time), (numBuf * bufferSize[0]) / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);

					printf("VkFFT System: %d %dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %d benchmark: %d bandwidth: %0.1f\n", (int)log2(configuration.size[0]), configuration.size[0], configuration.size[1], (numBuf * bufferSize[0]) / 1024 / 1024, avg_time, std_error, num_iter, (int)(((double)(numBuf * bufferSize[0]) / 1024) / avg_time), (numBuf * bufferSize[0]) / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);
					benchmark_result += ((double)numBuf * bufferSize[0] / 1024) / avg_time;
				}


			}
			for (uint32_t i = 0; i < numBuf; i++) {

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
	benchmark_result /= (26 - 1);

	if (file_output) {
		fprintf(output, "Benchmark score VkFFT: %d\n", (int)(benchmark_result));
		fprintf(output, "Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
	}
	printf("Benchmark score VkFFT: %d\n", (int)(benchmark_result));
	printf("Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
	return res;
}
#endif
#ifdef USE_FFTW
uint32_t sample_11(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	uint32_t res = 0;
	if (file_output)
		fprintf(output, "11 - VkFFT/FFTW C2C precision test in single precision\n");
	printf("11 - VkFFT/FFTW C2C precision test in single precision\n");

	const int num_benchmark_samples = 63;
	const int num_runs = 1;

	uint32_t benchmark_dimensions[num_benchmark_samples][4] = { {(uint32_t)pow(2,5), 1, 1, 1}, {(uint32_t)pow(2,6), 1, 1, 1},{(uint32_t)pow(2,7), 1, 1, 1},{(uint32_t)pow(2,8), 1, 1, 1},{(uint32_t)pow(2,9), 1, 1, 1},{(uint32_t)pow(2,10), 1, 1, 1},
		{(uint32_t)pow(2,11), 1, 1, 1},{(uint32_t)pow(2,12), 1, 1, 1},{(uint32_t)pow(2,13), 1, 1, 1},{(uint32_t)pow(2,14), 1, 1, 1},{(uint32_t)pow(2,15), 1, 1, 1},{(uint32_t)pow(2,16), 1, 1, 1},{(uint32_t)pow(2,17), 1, 1, 1},{(uint32_t)pow(2,18), 1, 1, 1},
		{(uint32_t)pow(2,19), 1, 1, 1},{(uint32_t)pow(2,20), 1, 1, 1},{(uint32_t)pow(2,21), 1, 1, 1},{(uint32_t)pow(2,22), 1, 1, 1},{(uint32_t)pow(2,23), 1, 1, 1},{(uint32_t)pow(2,24), 1, 1, 1},{(uint32_t)pow(2,25), 1, 1, 1},{(uint32_t)pow(2,26), 1, 1, 1},

		{8, (uint32_t)pow(2,3), 1, 2},{8, (uint32_t)pow(2,4), 1, 2},{8, (uint32_t)pow(2,5), 1, 2},{8, (uint32_t)pow(2,6), 1, 2},{8, (uint32_t)pow(2,7), 1, 2},{8, (uint32_t)pow(2,8), 1, 2},{8, (uint32_t)pow(2,9), 1, 2},{8, (uint32_t)pow(2,10), 1, 2},
		{8, (uint32_t)pow(2,11), 1, 2},{8, (uint32_t)pow(2,12), 1, 2},{8, (uint32_t)pow(2,13), 1, 2},{8, (uint32_t)pow(2,14), 1, 2},{8, (uint32_t)pow(2,15), 1, 2},{8, (uint32_t)pow(2,16), 1, 2},{8, (uint32_t)pow(2,17), 1, 2},{8, (uint32_t)pow(2,18), 1, 2},
		{8, (uint32_t)pow(2,19), 1, 2},{8, (uint32_t)pow(2,20), 1, 2},{8, (uint32_t)pow(2,21), 1, 2},{8, (uint32_t)pow(2,22), 1, 2},{8, (uint32_t)pow(2,23), 1, 2},{8, (uint32_t)pow(2,24), 1, 2},

		{ (uint32_t)pow(2,3), (uint32_t)pow(2,3), 1, 2},{ (uint32_t)pow(2,4), (uint32_t)pow(2,4), 1, 2},{ (uint32_t)pow(2,5), (uint32_t)pow(2,5), 1, 2},{ (uint32_t)pow(2,6), (uint32_t)pow(2,6), 1, 2},{ (uint32_t)pow(2,7), (uint32_t)pow(2,7), 1, 2},{ (uint32_t)pow(2,8), (uint32_t)pow(2,8), 1, 2},{ (uint32_t)pow(2,9), (uint32_t)pow(2,9), 1, 2},
		{ (uint32_t)pow(2,10), (uint32_t)pow(2,10), 1, 2},{ (uint32_t)pow(2,11), (uint32_t)pow(2,11), 1, 2},{ (uint32_t)pow(2,12), (uint32_t)pow(2,12), 1, 2},{ (uint32_t)pow(2,13), (uint32_t)pow(2,13), 1, 2},{ (uint32_t)pow(2,14), (uint32_t)pow(2,13), 1, 2},

		{ (uint32_t)pow(2,3), (uint32_t)pow(2,3), (uint32_t)pow(2,3), 3},{ (uint32_t)pow(2,4), (uint32_t)pow(2,4), (uint32_t)pow(2,4), 3},{ (uint32_t)pow(2,5), (uint32_t)pow(2,5), (uint32_t)pow(2,5), 3},{ (uint32_t)pow(2,6), (uint32_t)pow(2,6), (uint32_t)pow(2,6), 3},{ (uint32_t)pow(2,7), (uint32_t)pow(2,7), (uint32_t)pow(2,7), 3},{ (uint32_t)pow(2,8), (uint32_t)pow(2,8), (uint32_t)pow(2,8), 3},{ (uint32_t)pow(2,9), (uint32_t)pow(2,9), (uint32_t)pow(2,9), 3},
	};

	double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples

	for (int n = 0; n < num_benchmark_samples; n++) {
		for (int r = 0; r < num_runs; r++) {

			fftwf_complex* inputC;
			fftw_complex* inputC_double;
			uint32_t dims[3] = { benchmark_dimensions[n][0] , benchmark_dimensions[n][1] ,benchmark_dimensions[n][2] };

			inputC = (fftwf_complex*)(malloc(sizeof(fftwf_complex) * dims[0] * dims[1] * dims[2]));
			inputC_double = (fftw_complex*)(malloc(sizeof(fftw_complex) * dims[0] * dims[1] * dims[2]));
			for (int l = 0; l < dims[2]; l++) {
				for (int j = 0; j < dims[1]; j++) {
					for (int i = 0; i < dims[0]; i++) {
						inputC[i + j * dims[0] + l * dims[0] * dims[1]][0] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						inputC[i + j * dims[0] + l * dims[0] * dims[1]][1] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						inputC_double[i + j * dims[0] + l * dims[0] * dims[1]][0] = (double)inputC[i + j * dims[0] + l * dims[0] * dims[1]][0];
						inputC_double[i + j * dims[0] + l * dims[0] * dims[1]][1] = (double)inputC[i + j * dims[0] + l * dims[0] * dims[1]][1];
					}
				}
			}

			fftw_plan p;

			fftw_complex* output_FFTW = (fftw_complex*)(malloc(sizeof(fftw_complex) * dims[0] * dims[1] * dims[2]));

			switch (benchmark_dimensions[n][3]) {
			case 1:
				p = fftw_plan_dft_1d(benchmark_dimensions[n][0], inputC_double, output_FFTW, -1, FFTW_ESTIMATE);
				break;
			case 2:
				p = fftw_plan_dft_2d(benchmark_dimensions[n][1], benchmark_dimensions[n][0], inputC_double, output_FFTW, -1, FFTW_ESTIMATE);
				break;
			case 3:
				p = fftw_plan_dft_3d(benchmark_dimensions[n][2], benchmark_dimensions[n][1], benchmark_dimensions[n][0], inputC_double, output_FFTW, -1, FFTW_ESTIMATE);
				break;
			}

			fftw_execute(p);

			float totTime = 0;
			int num_iter = 1;

#ifdef USE_cuFFT
			fftwf_complex* output_extFFT = (fftwf_complex*)(malloc(sizeof(fftwf_complex) * dims[0] * dims[1] * dims[2]));
			launch_precision_cuFFT_single(inputC, (void*)output_extFFT, benchmark_dimensions[n]);
#endif // USE_cuFFT
#ifdef USE_rocFFT
			fftwf_complex* output_extFFT = (fftwf_complex*)(malloc(sizeof(fftwf_complex) * dims[0] * dims[1] * dims[2]));
			launch_precision_rocFFT_single(inputC, (void*)output_extFFT, benchmark_dimensions[n]);
#endif // USE_rocFFT
			//VkFFT part

			VkFFTConfiguration configuration = {};
			VkFFTApplication app = {};
			configuration.FFTdim = benchmark_dimensions[n][3]; //FFT dimension, 1D, 2D or 3D (default 1).
			configuration.size[0] = benchmark_dimensions[n][0]; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			configuration.size[1] = benchmark_dimensions[n][1];
			configuration.size[2] = benchmark_dimensions[n][2];

			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			configuration.device = &vkGPU->device;
#if(VKFFT_BACKEND==0)
			configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
			configuration.fence = &vkGPU->fence;
			configuration.commandPool = &vkGPU->commandPool;
			configuration.physicalDevice = &vkGPU->physicalDevice;
			configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization
#endif

			uint32_t numBuf = 1;

			//Allocate buffers for the input data. - we use 4 in this example
			uint64_t* bufferSize = (uint64_t*)malloc(sizeof(uint64_t) * numBuf);
			for (uint32_t i = 0; i < numBuf; i++) {
				bufferSize[i] = {};
				bufferSize[i] = (uint64_t)sizeof(float) * 2 * configuration.size[0] * configuration.size[1] * configuration.size[2] / numBuf;
			}
#if(VKFFT_BACKEND==0)
			VkBuffer* buffer = (VkBuffer*)malloc(numBuf * sizeof(VkBuffer));
			VkDeviceMemory* bufferDeviceMemory = (VkDeviceMemory*)malloc(numBuf * sizeof(VkDeviceMemory));
#elif(VKFFT_BACKEND==1)
			cuFloatComplex* buffer = 0;
#elif(VKFFT_BACKEND==2)
			hipFloatComplex* buffer = 0;
#endif
			for (uint32_t i = 0; i < numBuf; i++) {
#if(VKFFT_BACKEND==0)
				buffer[i] = {};
				bufferDeviceMemory[i] = {};
				allocateFFTBuffer(vkGPU, &buffer[i], &bufferDeviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize[i]);
#elif(VKFFT_BACKEND==1)
				cudaMalloc((void**)&buffer, bufferSize[i]);
#elif(VKFFT_BACKEND==2)
				hipMalloc((void**)&buffer, bufferSize[i]);
#endif
			}

			configuration.bufferNum = numBuf;

#if(VKFFT_BACKEND==0)
			configuration.buffer = buffer;
#elif(VKFFT_BACKEND==1)
			configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==2)
			configuration.buffer = (void**)&buffer;
#endif
			configuration.bufferSize = bufferSize;

			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
			uint64_t shift = 0;
			for (uint32_t i = 0; i < numBuf; i++) {
#if(VKFFT_BACKEND==0)
				transferDataFromCPU(vkGPU, (inputC + shift / sizeof(fftwf_complex)), &buffer[i], bufferSize[i]);
#elif(VKFFT_BACKEND==1)
				cudaMemcpy(buffer, inputC, bufferSize[i], cudaMemcpyHostToDevice);
#elif(VKFFT_BACKEND==2)
				hipMemcpy(buffer, inputC, bufferSize[i], hipMemcpyHostToDevice);
#endif
				shift += bufferSize[i];
			}
			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			res = initializeVkFFT(&app, configuration);
			if (res != 0) return res;

			//Submit FFT+iFFT.
			//num_iter = 1;
			performVulkanFFT(vkGPU, &app, -1, num_iter);
			fftwf_complex* output_VkFFT = (fftwf_complex*)(malloc(sizeof(fftwf_complex) * dims[0] * dims[1] * dims[2]));

			//Transfer data from GPU using staging buffer.
			shift = 0;
			for (uint32_t i = 0; i < numBuf; i++) {
#if(VKFFT_BACKEND==0)
				transferDataToCPU(vkGPU, (output_VkFFT + shift / sizeof(fftwf_complex)), &buffer[i], bufferSize[i]);
#elif(VKFFT_BACKEND==1)
				cudaMemcpy(output_VkFFT, buffer, bufferSize[i], cudaMemcpyDeviceToHost);
#elif(VKFFT_BACKEND==2)
				hipMemcpy(output_VkFFT, buffer, bufferSize[i], hipMemcpyDeviceToHost);
#endif
				shift += bufferSize[i];
			}
			float avg_difference[2] = { 0,0 };
			float max_difference[2] = { 0,0 };
			float avg_eps[2] = { 0,0 };
			float max_eps[2] = { 0,0 };
			for (int l = 0; l < dims[2]; l++) {
				for (int j = 0; j < dims[1]; j++) {
					for (int i = 0; i < dims[0]; i++) {
						int loc_i = i;
						int loc_j = j;
						int loc_l = l;

						//if (file_output) fprintf(output, "%f %f - %f %f \n", output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] / N, output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1] / N, output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][0], output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][1]);

						//printf("%f %f - %f %f \n", output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0], output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1], output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][0], output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][1]);
						float current_data_norm = sqrt(output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] * output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] + output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1] * output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1]);
#if defined(USE_cuFFT) || defined(USE_rocFFT)
						float current_diff_x_extFFT = (output_extFFT[loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]][0] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0]);
						float current_diff_y_extFFT = (output_extFFT[loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]][1] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1]);
						float current_diff_norm_extFFT = sqrt(current_diff_x_extFFT * current_diff_x_extFFT + current_diff_y_extFFT * current_diff_y_extFFT);
						if (current_diff_norm_extFFT > max_difference[0]) max_difference[0] = current_diff_norm_extFFT;
						avg_difference[0] += current_diff_norm_extFFT;
						if ((current_diff_norm_extFFT / current_data_norm > max_eps[0]) && (current_data_norm > 1e-10)) {
							max_eps[0] = current_diff_norm_extFFT / current_data_norm;
						}
						avg_eps[0] += (current_data_norm > 1e-10) ? current_diff_norm_extFFT / current_data_norm : 0;
#endif

						float current_diff_x_VkFFT = (output_VkFFT[loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]][0] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0]);
						float current_diff_y_VkFFT = (output_VkFFT[loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]][1] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1]);
						float current_diff_norm_VkFFT = sqrt(current_diff_x_VkFFT * current_diff_x_VkFFT + current_diff_y_VkFFT * current_diff_y_VkFFT);
						if (current_diff_norm_VkFFT > max_difference[1]) max_difference[1] = current_diff_norm_VkFFT;
						avg_difference[1] += current_diff_norm_VkFFT;
						if ((current_diff_norm_VkFFT / current_data_norm > max_eps[1]) && (current_data_norm > 1e-10)) {
							max_eps[1] = current_diff_norm_VkFFT / current_data_norm;
						}
						avg_eps[1] += (current_data_norm > 1e-10) ? current_diff_norm_VkFFT / current_data_norm : 0;
					}
				}
			}
			avg_difference[0] /= (dims[0] * dims[1] * dims[2]);
			avg_eps[0] /= (dims[0] * dims[1] * dims[2]);
			avg_difference[1] /= (dims[0] * dims[1] * dims[2]);
			avg_eps[1] /= (dims[0] * dims[1] * dims[2]);
#ifdef USE_cuFFT
			if (file_output)
				fprintf(output, "cuFFT System: %dx%dx%d avg_difference: %f max_difference: %f avg_eps: %f max_eps: %f\n", dims[0], dims[1], dims[2], avg_difference[0], max_difference[0], avg_eps[0], max_eps[0]);
			printf("cuFFT System: %dx%dx%d avg_difference: %f max_difference: %f avg_eps: %f max_eps: %f\n", dims[0], dims[1], dims[2], avg_difference[0], max_difference[0], avg_eps[0], max_eps[0]);
#endif
#ifdef USE_rocFFT
			if (file_output)
				fprintf(output, "rocFFT System: %dx%dx%d avg_difference: %f max_difference: %f avg_eps: %f max_eps: %f\n", dims[0], dims[1], dims[2], avg_difference[0], max_difference[0], avg_eps[0], max_eps[0]);
			printf("rocFFT System: %dx%dx%d avg_difference: %f max_difference: %f avg_eps: %f max_eps: %f\n", dims[0], dims[1], dims[2], avg_difference[0], max_difference[0], avg_eps[0], max_eps[0]);
#endif
			if (file_output)
				fprintf(output, "VkFFT System: %dx%dx%d avg_difference: %f max_difference: %f avg_eps: %f max_eps: %f\n", dims[0], dims[1], dims[2], avg_difference[1], max_difference[1], avg_eps[1], max_eps[1]);
			printf("VkFFT System: %dx%dx%d avg_difference: %f max_difference: %f avg_eps: %f max_eps: %f\n", dims[0], dims[1], dims[2], avg_difference[1], max_difference[1], avg_eps[1], max_eps[1]);
			free(output_VkFFT);
			for (uint32_t i = 0; i < numBuf; i++) {
#if(VKFFT_BACKEND==0)
				vkDestroyBuffer(vkGPU->device, buffer[i], NULL);
				vkFreeMemory(vkGPU->device, bufferDeviceMemory[i], NULL);
#elif(VKFFT_BACKEND==1)
				cudaFree(buffer);
#elif(VKFFT_BACKEND==2)
				hipFree(buffer);
#endif
			}
#if(VKFFT_BACKEND==0)
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
	return res;
}
uint32_t sample_12(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	uint32_t res = 0;
	if (file_output)
		fprintf(output, "12 - VkFFT/FFTW C2C precision test in double precision\n");
	printf("12 - VkFFT/FFTW C2C precision test in double precision\n");
	const int num_benchmark_samples = 60;
	const int num_runs = 1;

	uint32_t benchmark_dimensions[num_benchmark_samples][4] = { {(uint32_t)pow(2,5), 1, 1, 1},{(uint32_t)pow(2,6), 1, 1, 1},{(uint32_t)pow(2,7), 1, 1, 1},{(uint32_t)pow(2,8), 1, 1, 1},{(uint32_t)pow(2,9), 1, 1, 1},{(uint32_t)pow(2,10), 1, 1, 1},
		{(uint32_t)pow(2,11), 1, 1, 1},{(uint32_t)pow(2,12), 1, 1, 1},{(uint32_t)pow(2,13), 1, 1, 1},{(uint32_t)pow(2,14), 1, 1, 1},{(uint32_t)pow(2,15), 1, 1, 1},{(uint32_t)pow(2,16), 1, 1, 1},{(uint32_t)pow(2,17), 1, 1, 1},{(uint32_t)pow(2,18), 1, 1, 1},
		{(uint32_t)pow(2,19), 1, 1, 1},{(uint32_t)pow(2,20), 1, 1, 1},{(uint32_t)pow(2,21), 1, 1, 1},{(uint32_t)pow(2,22), 1, 1, 1},{(uint32_t)pow(2,23), 1, 1, 1},{(uint32_t)pow(2,24), 1, 1, 1},{(uint32_t)pow(2,25), 1, 1, 1},{(uint32_t)pow(2,26), 1, 1, 1},

		{8, (uint32_t)pow(2,3), 1, 2},{8, (uint32_t)pow(2,4), 1, 2},{8, (uint32_t)pow(2,5), 1, 2},{8, (uint32_t)pow(2,6), 1, 2},{8, (uint32_t)pow(2,7), 1, 2},{8, (uint32_t)pow(2,8), 1, 2},{8, (uint32_t)pow(2,9), 1, 2},{8, (uint32_t)pow(2,10), 1, 2},
		{8, (uint32_t)pow(2,11), 1, 2},{8, (uint32_t)pow(2,12), 1, 2},{8, (uint32_t)pow(2,13), 1, 2},{8, (uint32_t)pow(2,14), 1, 2},{8, (uint32_t)pow(2,15), 1, 2},{8, (uint32_t)pow(2,16), 1, 2},{8, (uint32_t)pow(2,17), 1, 2},{8, (uint32_t)pow(2,18), 1, 2},
		{8, (uint32_t)pow(2,19), 1, 2},{8, (uint32_t)pow(2,20), 1, 2},{8, (uint32_t)pow(2,21), 1, 2},{8, (uint32_t)pow(2,22), 1, 2},{8, (uint32_t)pow(2,23), 1, 2},

		{ (uint32_t)pow(2,3), (uint32_t)pow(2,3), 1, 2}, { (uint32_t)pow(2,4), (uint32_t)pow(2,4), 1, 2},{ (uint32_t)pow(2,5), (uint32_t)pow(2,5), 1, 2},{ (uint32_t)pow(2,6), (uint32_t)pow(2,6), 1, 2},{ (uint32_t)pow(2,7), (uint32_t)pow(2,7), 1, 2},{ (uint32_t)pow(2,8), (uint32_t)pow(2,8), 1, 2},{ (uint32_t)pow(2,9), (uint32_t)pow(2,9), 1, 2},
		{ (uint32_t)pow(2,10), (uint32_t)pow(2,10), 1, 2},{ (uint32_t)pow(2,11), (uint32_t)pow(2,11), 1, 2},{ (uint32_t)pow(2,12), (uint32_t)pow(2,12), 1, 2},{ (uint32_t)pow(2,13), (uint32_t)pow(2,13), 1, 2},

		{ (uint32_t)pow(2,3), (uint32_t)pow(2,3), (uint32_t)pow(2,3), 3},{ (uint32_t)pow(2,4), (uint32_t)pow(2,4), (uint32_t)pow(2,4), 3},{ (uint32_t)pow(2,5), (uint32_t)pow(2,5), (uint32_t)pow(2,5), 3},{ (uint32_t)pow(2,6), (uint32_t)pow(2,6), (uint32_t)pow(2,6), 3},{ (uint32_t)pow(2,7), (uint32_t)pow(2,7), (uint32_t)pow(2,7), 3},{ (uint32_t)pow(2,8), (uint32_t)pow(2,8), (uint32_t)pow(2,8), 3}
	};

	double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples

	for (int n = 0; n < num_benchmark_samples; n++) {
		for (int r = 0; r < num_runs; r++) {
			fftw_complex* inputC;
			fftw_complex* inputC_double;
			uint32_t dims[3] = { benchmark_dimensions[n][0] , benchmark_dimensions[n][1] ,benchmark_dimensions[n][2] };

			inputC = (fftw_complex*)(malloc(sizeof(fftw_complex) * dims[0] * dims[1] * dims[2]));
			inputC_double = (fftw_complex*)(malloc(sizeof(fftw_complex) * dims[0] * dims[1] * dims[2]));
			for (int l = 0; l < dims[2]; l++) {
				for (int j = 0; j < dims[1]; j++) {
					for (int i = 0; i < dims[0]; i++) {
						inputC[i + j * dims[0] + l * dims[0] * dims[1]][0] = 2 * ((double)rand()) / RAND_MAX - 1.0;
						inputC[i + j * dims[0] + l * dims[0] * dims[1]][1] = 2 * ((double)rand()) / RAND_MAX - 1.0;
						inputC_double[i + j * dims[0] + l * dims[0] * dims[1]][0] = (double)inputC[i + j * dims[0] + l * dims[0] * dims[1]][0];
						inputC_double[i + j * dims[0] + l * dims[0] * dims[1]][1] = (double)inputC[i + j * dims[0] + l * dims[0] * dims[1]][1];
					}
				}
			}

			fftw_plan p;

			fftw_complex* output_FFTW = (fftw_complex*)(malloc(sizeof(fftw_complex) * dims[0] * dims[1] * dims[2]));

			switch (benchmark_dimensions[n][3]) {
			case 1:
				p = fftw_plan_dft_1d(benchmark_dimensions[n][0], inputC_double, output_FFTW, -1, FFTW_ESTIMATE);
				break;
			case 2:
				p = fftw_plan_dft_2d(benchmark_dimensions[n][1], benchmark_dimensions[n][0], inputC_double, output_FFTW, -1, FFTW_ESTIMATE);
				break;
			case 3:
				p = fftw_plan_dft_3d(benchmark_dimensions[n][2], benchmark_dimensions[n][1], benchmark_dimensions[n][0], inputC_double, output_FFTW, -1, FFTW_ESTIMATE);
				break;
			}

			fftw_execute(p);

#ifdef USE_cuFFT
			fftw_complex* output_extFFT = (fftw_complex*)(malloc(sizeof(fftw_complex) * dims[0] * dims[1] * dims[2]));
			launch_precision_cuFFT_double(inputC, (void*)output_extFFT, benchmark_dimensions[n]);
#endif // USE_cuFFT
#ifdef USE_rocFFT
			fftw_complex* output_extFFT = (fftw_complex*)(malloc(sizeof(fftw_complex) * dims[0] * dims[1] * dims[2]));
			launch_precision_rocFFT_double(inputC, (void*)output_extFFT, benchmark_dimensions[n]);
#endif // USE_rocFFT
			float totTime = 0;
			int num_iter = 1;

			//VkFFT part

			VkFFTConfiguration configuration = {};
			VkFFTApplication app = {};

			configuration.FFTdim = benchmark_dimensions[n][3]; //FFT dimension, 1D, 2D or 3D (default 1).
			configuration.size[0] = benchmark_dimensions[n][0]; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			configuration.size[1] = benchmark_dimensions[n][1];
			configuration.size[2] = benchmark_dimensions[n][2];

			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			configuration.device = &vkGPU->device;
#if(VKFFT_BACKEND==0)
			configuration.queue = &vkGPU->queue;
			configuration.fence = &vkGPU->fence;
			configuration.commandPool = &vkGPU->commandPool;
			configuration.physicalDevice = &vkGPU->physicalDevice;
			configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization
#endif
			configuration.doublePrecision = true;

			uint32_t numBuf = 1;

			//Allocate buffers for the input data. - we use 4 in this example
			uint64_t* bufferSize = (uint64_t*)malloc(sizeof(uint64_t) * numBuf);
			for (uint32_t i = 0; i < numBuf; i++) {
				bufferSize[i] = {};
				bufferSize[i] = (uint64_t)sizeof(double) * 2 * configuration.size[0] * configuration.size[1] * configuration.size[2] / numBuf;
			}
#if(VKFFT_BACKEND==0)
			VkBuffer* buffer = (VkBuffer*)malloc(numBuf * sizeof(VkBuffer));
			VkDeviceMemory* bufferDeviceMemory = (VkDeviceMemory*)malloc(numBuf * sizeof(VkDeviceMemory));
#elif(VKFFT_BACKEND==1)
			cuFloatComplex* buffer = 0;
#elif(VKFFT_BACKEND==2)
			hipFloatComplex* buffer = 0;
#endif
			for (uint32_t i = 0; i < numBuf; i++) {
#if(VKFFT_BACKEND==0)
				buffer[i] = {};
				bufferDeviceMemory[i] = {};
				allocateFFTBuffer(vkGPU, &buffer[i], &bufferDeviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize[i]);
#elif(VKFFT_BACKEND==1)
				cudaMalloc((void**)&buffer, bufferSize[i]);
#elif(VKFFT_BACKEND==2)
				hipMalloc((void**)&buffer, bufferSize[i]);
#endif
			}

			configuration.bufferNum = numBuf;

#if(VKFFT_BACKEND==0)
			configuration.buffer = buffer;
#elif(VKFFT_BACKEND==1)
			configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==2)
			configuration.buffer = (void**)&buffer;
#endif

			configuration.bufferSize = bufferSize;

			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
			uint64_t shift = 0;
			for (uint32_t i = 0; i < numBuf; i++) {
#if(VKFFT_BACKEND==0)
				transferDataFromCPU(vkGPU, (inputC + shift / sizeof(fftw_complex)), &buffer[i], bufferSize[i]);
#elif(VKFFT_BACKEND==1)
				cudaMemcpy(buffer, inputC, bufferSize[i], cudaMemcpyHostToDevice);
#elif(VKFFT_BACKEND==2)
				hipMemcpy(buffer, inputC, bufferSize[i], hipMemcpyHostToDevice);
#endif
				shift += bufferSize[i];
			}
			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			res = initializeVkFFT(&app, configuration);
			if (res != 0) return res;
			//Submit FFT+iFFT.
			//num_iter = 1;
			performVulkanFFT(vkGPU, &app, -1, num_iter);

			fftw_complex* output_VkFFT = (fftw_complex*)(malloc(sizeof(fftw_complex) * dims[0] * dims[1] * dims[2]));

			//Transfer data from GPU using staging buffer.
			shift = 0;
			for (uint32_t i = 0; i < numBuf; i++) {
#if(VKFFT_BACKEND==0)
				transferDataToCPU(vkGPU, (output_VkFFT + shift / sizeof(fftw_complex)), &buffer[i], bufferSize[i]);
#elif(VKFFT_BACKEND==1)
				cudaMemcpy(output_VkFFT, buffer, bufferSize[i], cudaMemcpyDeviceToHost);
#elif(VKFFT_BACKEND==2)
				hipMemcpy(output_VkFFT, buffer, bufferSize[i], hipMemcpyDeviceToHost);
#endif
				shift += bufferSize[i];
			}
			double avg_difference[2] = { 0,0 };
			double max_difference[2] = { 0,0 };
			double avg_eps[2] = { 0,0 };
			double max_eps[2] = { 0,0 };
			for (int l = 0; l < dims[2]; l++) {
				for (int j = 0; j < dims[1]; j++) {
					for (int i = 0; i < dims[0]; i++) {
						int loc_i = i;
						int loc_j = j;
						int loc_l = l;

						//if (file_output) fprintf(output, "%f %f - %f %f \n", output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] / N, output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1] / N, output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][0], output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][1]);
						//printf("%f %f - %f %f \n", output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0], output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1], output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][0], output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][1]);
						double current_data_norm = sqrt(output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] * output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] + output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1] * output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1]);

#if defined(USE_cuFFT) || defined(USE_rocFFT)
						double current_diff_x_extFFT = (output_extFFT[loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]][0] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0]);
						double current_diff_y_extFFT = (output_extFFT[loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]][1] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1]);
						double current_diff_norm_extFFT = sqrt(current_diff_x_extFFT * current_diff_x_extFFT + current_diff_y_extFFT * current_diff_y_extFFT);
						if (current_diff_norm_extFFT > max_difference[0]) max_difference[0] = current_diff_norm_extFFT;
						avg_difference[0] += current_diff_norm_extFFT;
						if ((current_diff_norm_extFFT / current_data_norm > max_eps[0]) && (current_data_norm > 1e-10)) {
							max_eps[0] = current_diff_norm_extFFT / current_data_norm;
						}
						avg_eps[0] += (current_data_norm > 1e-10) ? current_diff_norm_extFFT / current_data_norm : 0;
#endif

						double current_diff_x_VkFFT = (output_VkFFT[loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]][0] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0]);
						double current_diff_y_VkFFT = (output_VkFFT[loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]][1] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1]);
						double current_diff_norm_VkFFT = sqrt(current_diff_x_VkFFT * current_diff_x_VkFFT + current_diff_y_VkFFT * current_diff_y_VkFFT);
						if (current_diff_norm_VkFFT > max_difference[1]) max_difference[1] = current_diff_norm_VkFFT;
						avg_difference[1] += current_diff_norm_VkFFT;
						if ((current_diff_norm_VkFFT / current_data_norm > max_eps[1]) && (current_data_norm > 1e-16)) {
							max_eps[1] = current_diff_norm_VkFFT / current_data_norm;
						}
						avg_eps[1] += (current_data_norm > 1e-10) ? current_diff_norm_VkFFT / current_data_norm : 0;
					}
				}
			}
			avg_difference[0] /= (dims[0] * dims[1] * dims[2]);
			avg_eps[0] /= (dims[0] * dims[1] * dims[2]);
			avg_difference[1] /= (dims[0] * dims[1] * dims[2]);
			avg_eps[1] /= (dims[0] * dims[1] * dims[2]);
#ifdef USE_cuFFT
			if (file_output)
				fprintf(output, "cuFFT System: %dx%dx%d avg_difference: %.15f max_difference: %.15f avg_eps: %.15f max_eps: %.15f\n", dims[0], dims[1], dims[2], avg_difference[0], max_difference[0], avg_eps[0], max_eps[0]);
			printf("cuFFT System: %dx%dx%d avg_difference: %.15f max_difference: %.15f avg_eps: %.15f max_eps: %.15f\n", dims[0], dims[1], dims[2], avg_difference[0], max_difference[0], avg_eps[0], max_eps[0]);
#endif
#ifdef USE_rocFFT
			if (file_output)
				fprintf(output, "rocFFT System: %dx%dx%d avg_difference: %.15f max_difference: %.15f avg_eps: %.15f max_eps: %.15f\n", dims[0], dims[1], dims[2], avg_difference[0], max_difference[0], avg_eps[0], max_eps[0]);
			printf("rocFFT System: %dx%dx%d avg_difference: %.15f max_difference: %.15f avg_eps: %.15f max_eps: %.15f\n", dims[0], dims[1], dims[2], avg_difference[0], max_difference[0], avg_eps[0], max_eps[0]);
#endif
			if (file_output)
				fprintf(output, "VkFFT System: %dx%dx%d avg_difference: %.15f max_difference: %.15f avg_eps: %.15f max_eps: %.15f\n", dims[0], dims[1], dims[2], avg_difference[1], max_difference[1], avg_eps[1], max_eps[1]);
			printf("VkFFT System: %dx%dx%d avg_difference: %.15f max_difference: %.15f avg_eps: %.15f max_eps: %.15f\n", dims[0], dims[1], dims[2], avg_difference[1], max_difference[1], avg_eps[1], max_eps[1]);
			free(output_VkFFT);
			for (uint32_t i = 0; i < numBuf; i++) {

#if(VKFFT_BACKEND==0)
				vkDestroyBuffer(vkGPU->device, buffer[i], NULL);
				vkFreeMemory(vkGPU->device, bufferDeviceMemory[i], NULL);
#elif(VKFFT_BACKEND==1)
				cudaFree(buffer);
#elif(VKFFT_BACKEND==2)
				hipFree(buffer);
#endif

			}
#if(VKFFT_BACKEND==0)
			free(buffer);
			free(bufferDeviceMemory);
#endif
#if defined(USE_cuFFT) || defined(USE_rocFFT)
			free(output_extFFT);
#endif
			deleteVkFFT(&app);
			free(inputC);
			fftw_destroy_plan(p);
			free(inputC_double);
			free(output_FFTW);
		}
	}
	return res;
}
#if ((VKFFT_BACKEND==0)&&(VK_API_VERSION>10))
uint32_t sample_13(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	uint32_t res = 0;
	if (file_output)
		fprintf(output, "13 - VkFFT/FFTW C2C precision test in half precision\n");
	printf("13 - VkFFT/FFTW C2C precision test in half precision\n");

	const int num_benchmark_samples = 61;
	const int num_runs = 1;

	uint32_t benchmark_dimensions[num_benchmark_samples][4] = { {(uint32_t)pow(2,5), 1, 1, 1},{(uint32_t)pow(2,6), 1, 1, 1},{(uint32_t)pow(2,7), 1, 1, 1},{(uint32_t)pow(2,8), 1, 1, 1},{(uint32_t)pow(2,9), 1, 1, 1},{(uint32_t)pow(2,10), 1, 1, 1},
		{(uint32_t)pow(2,11), 1, 1, 1},{(uint32_t)pow(2,12), 1, 1, 1},{(uint32_t)pow(2,13), 1, 1, 1},{(uint32_t)pow(2,14), 1, 1, 1},{(uint32_t)pow(2,15), 1, 1, 1},{(uint32_t)pow(2,16), 1, 1, 1},{(uint32_t)pow(2,17), 1, 1, 1},{(uint32_t)pow(2,18), 1, 1, 1},
		{(uint32_t)pow(2,19), 1, 1, 1},{(uint32_t)pow(2,20), 1, 1, 1},{(uint32_t)pow(2,21), 1, 1, 1},{(uint32_t)pow(2,22), 1, 1, 1},{(uint32_t)pow(2,23), 1, 1, 1},{(uint32_t)pow(2,24), 1, 1, 1},

		{8, (uint32_t)pow(2,3), 1, 2},{8, (uint32_t)pow(2,4), 1, 2},{8, (uint32_t)pow(2,5), 1, 2},{8, (uint32_t)pow(2,6), 1, 2},{8, (uint32_t)pow(2,7), 1, 2},{8, (uint32_t)pow(2,8), 1, 2},{8, (uint32_t)pow(2,9), 1, 2},{8, (uint32_t)pow(2,10), 1, 2},
		{8, (uint32_t)pow(2,11), 1, 2},{8, (uint32_t)pow(2,12), 1, 2},{8, (uint32_t)pow(2,13), 1, 2},{8, (uint32_t)pow(2,14), 1, 2},{8, (uint32_t)pow(2,15), 1, 2},{8, (uint32_t)pow(2,16), 1, 2},{8, (uint32_t)pow(2,17), 1, 2},{8, (uint32_t)pow(2,18), 1, 2},
		{8, (uint32_t)pow(2,19), 1, 2},{8, (uint32_t)pow(2,20), 1, 2},{8, (uint32_t)pow(2,21), 1, 2},{8, (uint32_t)pow(2,22), 1, 2},{8, (uint32_t)pow(2,23), 1, 2},{8, (uint32_t)pow(2,24), 1, 2},

		{ (uint32_t)pow(2,3), (uint32_t)pow(2,3), 1, 2},{ (uint32_t)pow(2,4), (uint32_t)pow(2,4), 1, 2},{ (uint32_t)pow(2,5), (uint32_t)pow(2,5), 1, 2},{ (uint32_t)pow(2,6), (uint32_t)pow(2,6), 1, 2},{ (uint32_t)pow(2,7), (uint32_t)pow(2,7), 1, 2},{ (uint32_t)pow(2,8), (uint32_t)pow(2,8), 1, 2},{ (uint32_t)pow(2,9), (uint32_t)pow(2,9), 1, 2},
		{ (uint32_t)pow(2,10), (uint32_t)pow(2,10), 1, 2},{ (uint32_t)pow(2,11), (uint32_t)pow(2,11), 1, 2},{ (uint32_t)pow(2,12), (uint32_t)pow(2,12), 1, 2},{ (uint32_t)pow(2,13), (uint32_t)pow(2,13), 1, 2},{ (uint32_t)pow(2,14), (uint32_t)pow(2,13), 1, 2},

		{ (uint32_t)pow(2,3), (uint32_t)pow(2,3), (uint32_t)pow(2,3), 3},{ (uint32_t)pow(2,4), (uint32_t)pow(2,4), (uint32_t)pow(2,4), 3},{ (uint32_t)pow(2,5), (uint32_t)pow(2,5), (uint32_t)pow(2,5), 3},{ (uint32_t)pow(2,6), (uint32_t)pow(2,6), (uint32_t)pow(2,6), 3},{ (uint32_t)pow(2,7), (uint32_t)pow(2,7), (uint32_t)pow(2,7), 3},{ (uint32_t)pow(2,8), (uint32_t)pow(2,8), (uint32_t)pow(2,8), 3},{ (uint32_t)pow(2,9), (uint32_t)pow(2,9), (uint32_t)pow(2,9), 3},
	};

	double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples

	for (int n = 0; n < num_benchmark_samples; n++) {
		for (int r = 0; r < num_runs; r++) {

			half2* inputC;
			fftw_complex* inputC_double;
			uint32_t dims[3] = { benchmark_dimensions[n][0] , benchmark_dimensions[n][1] ,benchmark_dimensions[n][2] };

			inputC = (half2*)(malloc(2 * sizeof(half) * dims[0] * dims[1] * dims[2]));
			inputC_double = (fftw_complex*)(malloc(sizeof(fftw_complex) * dims[0] * dims[1] * dims[2]));
			for (int l = 0; l < dims[2]; l++) {
				for (int j = 0; j < dims[1]; j++) {
					for (int i = 0; i < dims[0]; i++) {
						inputC[i + j * dims[0] + l * dims[0] * dims[1]][0] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						inputC[i + j * dims[0] + l * dims[0] * dims[1]][1] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						inputC_double[i + j * dims[0] + l * dims[0] * dims[1]][0] = (double)inputC[i + j * dims[0] + l * dims[0] * dims[1]][0];
						inputC_double[i + j * dims[0] + l * dims[0] * dims[1]][1] = (double)inputC[i + j * dims[0] + l * dims[0] * dims[1]][1];
					}
				}
			}

			fftw_plan p;

			fftw_complex* output_FFTW = (fftw_complex*)(malloc(sizeof(fftw_complex) * dims[0] * dims[1] * dims[2]));

			switch (benchmark_dimensions[n][3]) {
			case 1:
				p = fftw_plan_dft_1d(benchmark_dimensions[n][0], inputC_double, output_FFTW, -1, FFTW_ESTIMATE);
				break;
			case 2:
				p = fftw_plan_dft_2d(benchmark_dimensions[n][1], benchmark_dimensions[n][0], inputC_double, output_FFTW, -1, FFTW_ESTIMATE);
				break;
			case 3:
				p = fftw_plan_dft_3d(benchmark_dimensions[n][2], benchmark_dimensions[n][1], benchmark_dimensions[n][0], inputC_double, output_FFTW, -1, FFTW_ESTIMATE);
				break;
			}

			fftw_execute(p);

			float totTime = 0;
			int num_iter = 1;

#ifdef USE_cuFFT
			half2* output_extFFT = (half2*)(malloc(2 * sizeof(half) * dims[0] * dims[1] * dims[2]));
			launch_precision_cuFFT_half(inputC, (void*)output_extFFT, benchmark_dimensions[n]);
#endif // USE_cuFFT

			//VkFFT part

			VkFFTConfiguration configuration = {};
			VkFFTApplication app = {};
			configuration.FFTdim = benchmark_dimensions[n][3]; //FFT dimension, 1D, 2D or 3D (default 1).
			configuration.size[0] = benchmark_dimensions[n][0]; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			configuration.size[1] = benchmark_dimensions[n][1];
			configuration.size[2] = benchmark_dimensions[n][2];
			configuration.halfPrecision = true;

			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			configuration.device = &vkGPU->device;
#if(VKFFT_BACKEND==0)
			configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
			configuration.fence = &vkGPU->fence;
			configuration.commandPool = &vkGPU->commandPool;
			configuration.physicalDevice = &vkGPU->physicalDevice;
			configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization
#endif

			uint32_t numBuf = 1;

			//Allocate buffers for the input data. - we use 4 in this example
			uint64_t* bufferSize = (uint64_t*)malloc(sizeof(uint64_t) * numBuf);
			for (uint32_t i = 0; i < numBuf; i++) {
				bufferSize[i] = {};
				bufferSize[i] = (uint64_t)sizeof(half) * 2 * configuration.size[0] * configuration.size[1] * configuration.size[2] / numBuf;
			}
#if(VKFFT_BACKEND==0)
			VkBuffer* buffer = (VkBuffer*)malloc(numBuf * sizeof(VkBuffer));
			VkDeviceMemory* bufferDeviceMemory = (VkDeviceMemory*)malloc(numBuf * sizeof(VkDeviceMemory));
#elif(VKFFT_BACKEND==1)
			cuFloatComplex* buffer = 0;
#elif(VKFFT_BACKEND==2)
			hipFloatComplex* buffer = 0;
#endif			
			for (uint32_t i = 0; i < numBuf; i++) {
#if(VKFFT_BACKEND==0)
				buffer[i] = {};
				bufferDeviceMemory[i] = {};
				allocateFFTBuffer(vkGPU, &buffer[i], &bufferDeviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize[i]);
#elif(VKFFT_BACKEND==1)
				cudaMalloc((void**)&buffer, bufferSize[i]);
#elif(VKFFT_BACKEND==2)
				hipMalloc((void**)&buffer, bufferSize[i]);
#endif
			}

			configuration.bufferNum = numBuf;

#if(VKFFT_BACKEND==0)
			configuration.buffer = buffer;
#elif(VKFFT_BACKEND==1)
			configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==2)
			configuration.buffer = (void**)&buffer;
#endif

			configuration.bufferSize = bufferSize;

			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
			uint64_t shift = 0;
			for (uint32_t i = 0; i < numBuf; i++) {
#if(VKFFT_BACKEND==0)
				transferDataFromCPU(vkGPU, (inputC + shift / 2 / sizeof(half)), &buffer[i], bufferSize[i]);
#elif(VKFFT_BACKEND==1)
				cudaMemcpy(buffer, inputC, bufferSize[i], cudaMemcpyHostToDevice);
#elif(VKFFT_BACKEND==2)
				hipMemcpy(buffer, inputC, bufferSize[i], hipMemcpyHostToDevice);
#endif
				shift += bufferSize[i];
			}
			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			res = initializeVkFFT(&app, configuration);
			if (res != 0) return res;
			//Submit FFT+iFFT.
			//num_iter = 1;
			performVulkanFFT(vkGPU, &app, -1, num_iter);
			half2* output_VkFFT = (half2*)(malloc(2 * sizeof(half) * dims[0] * dims[1] * dims[2]));

			//Transfer data from GPU using staging buffer.
			shift = 0;
			for (uint32_t i = 0; i < numBuf; i++) {
#if(VKFFT_BACKEND==0)
				transferDataToCPU(vkGPU, (output_VkFFT + shift / 2 / sizeof(half)), &buffer[i], bufferSize[i]);
#elif(VKFFT_BACKEND==1)
				cudaMemcpy(output_VkFFT, buffer, bufferSize[i], cudaMemcpyDeviceToHost);
#elif(VKFFT_BACKEND==2)
				hipMemcpy(output_VkFFT, buffer, bufferSize[i], hipMemcpyDeviceToHost);
#endif
				shift += bufferSize[i];
			}
			float avg_difference[2] = { 0,0 };
			float max_difference[2] = { 0,0 };
			float avg_eps[2] = { 0,0 };
			float max_eps[2] = { 0,0 };
			for (int l = 0; l < dims[2]; l++) {
				for (int j = 0; j < dims[1]; j++) {
					for (int i = 0; i < dims[0]; i++) {
						int loc_i = i;
						int loc_j = j;
						int loc_l = l;

						//if (file_output) fprintf(output, "%f %f - %f %f \n", output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] / N, output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1] / N, output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][0], output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][1]);
						//printf("%f %f - %f %f \n", output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] / N, output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1] / N, output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][0], output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][1]);
						float current_data_norm = sqrt(output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] * output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] + output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1] * output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1]);
#ifdef USE_cuFFT
						float current_diff_x_extFFT = (output_extFFT[loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]][0] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0]);
						float current_diff_y_extFFT = (output_extFFT[loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]][1] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1]);
						float current_diff_norm_extFFT = sqrt(current_diff_x_extFFT * current_diff_x_extFFT + current_diff_y_extFFT * current_diff_y_extFFT);
						if (current_diff_norm_extFFT > max_difference[0]) max_difference[0] = current_diff_norm_extFFT;
						avg_difference[0] += current_diff_norm_extFFT;
						if ((current_diff_norm_extFFT / current_data_norm > max_eps[0]) && (current_data_norm > 1e-10)) {
							max_eps[0] = current_diff_norm_extFFT / current_data_norm;
						}
						avg_eps[0] += (current_data_norm > 1e-10) ? current_diff_norm_extFFT / current_data_norm : 0;
#endif

						float current_diff_x_VkFFT = (output_VkFFT[loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]][0] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0]);
						float current_diff_y_VkFFT = (output_VkFFT[loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]][1] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1]);
						float current_diff_norm_VkFFT = sqrt(current_diff_x_VkFFT * current_diff_x_VkFFT + current_diff_y_VkFFT * current_diff_y_VkFFT);
						if (current_diff_norm_VkFFT > max_difference[1]) max_difference[1] = current_diff_norm_VkFFT;
						avg_difference[1] += current_diff_norm_VkFFT;
						if ((current_diff_norm_VkFFT / current_data_norm > max_eps[1]) && (current_data_norm > 1e-10)) {
							max_eps[1] = current_diff_norm_VkFFT / current_data_norm;
						}
						avg_eps[1] += (current_data_norm > 1e-10) ? current_diff_norm_VkFFT / current_data_norm : 0;
					}
				}
			}
			avg_difference[0] /= (dims[0] * dims[1] * dims[2]);
			avg_eps[0] /= (dims[0] * dims[1] * dims[2]);
			avg_difference[1] /= (dims[0] * dims[1] * dims[2]);
			avg_eps[1] /= (dims[0] * dims[1] * dims[2]);
#ifdef USE_cuFFT
			if (file_output)
				fprintf(output, "cuFFT System: %dx%dx%d avg_difference: %f max_difference: %f avg_eps: %f max_eps: %f\n", dims[0], dims[1], dims[2], avg_difference[0], max_difference[0], avg_eps[0], max_eps[0]);
			printf("cuFFT System: %dx%dx%d avg_difference: %f max_difference: %f avg_eps: %f max_eps: %f\n", dims[0], dims[1], dims[2], avg_difference[0], max_difference[0], avg_eps[0], max_eps[0]);
#endif
			if (file_output)
				fprintf(output, "VkFFT System: %dx%dx%d avg_difference: %f max_difference: %f avg_eps: %f max_eps: %f\n", dims[0], dims[1], dims[2], avg_difference[1], max_difference[1], avg_eps[1], max_eps[1]);
			printf("VkFFT System: %dx%dx%d avg_difference: %f max_difference: %f avg_eps: %f max_eps: %f\n", dims[0], dims[1], dims[2], avg_difference[1], max_difference[1], avg_eps[1], max_eps[1]);
			free(output_VkFFT);
			for (uint32_t i = 0; i < numBuf; i++) {

#if(VKFFT_BACKEND==0)
				vkDestroyBuffer(vkGPU->device, buffer[i], NULL);
				vkFreeMemory(vkGPU->device, bufferDeviceMemory[i], NULL);
#elif(VKFFT_BACKEND==1)
				cudaFree(buffer);
#elif(VKFFT_BACKEND==2)
				hipFree(buffer);
#endif

			}
#if(VKFFT_BACKEND==0)
			free(buffer);
			free(bufferDeviceMemory);
#endif
#ifdef USE_cuFFT
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
	return res;
}
#endif

uint32_t sample_14(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	uint32_t res = 0;
	if (file_output)
		fprintf(output, "14 - VkFFT/FFTW C2C power 3/5/7/11/13 precision test in single precision\n");
	printf("14 - VkFFT/FFTW C2C power 3/5/7/11/13 precision test in single precision\n");

	const int num_benchmark_samples = 200;
	const int num_runs = 1;

	uint32_t benchmark_dimensions[num_benchmark_samples][4] = { {3, 1, 1, 1},{5, 1, 1, 1},{6, 1, 1, 1},{7, 1, 1, 1},{9, 1, 1, 1},{10, 1, 1, 1},{11, 1, 1, 1},{12, 1, 1, 1},{13, 1, 1, 1},{14, 1, 1, 1},
		{15, 1, 1, 1},{21, 1, 1, 1},{22, 1, 1, 1},{24, 1, 1, 1},{25, 1, 1, 1},{26, 1, 1, 1},{27, 1, 1, 1},{28, 1, 1, 1},{30, 1, 1, 1},{33, 1, 1, 1},{35, 1, 1, 1},{39, 1, 1, 1},{45, 1, 1, 1},{42, 1, 1, 1},{44, 1, 1, 1},{49, 1, 1, 1},{52, 1, 1, 1},{55, 1, 1, 1},{56, 1, 1, 1},{60, 1, 1, 1},{65, 1, 1, 1},{66, 1, 1, 1},{81, 1, 1, 1},
		{121, 1, 1, 1},{125, 1, 1, 1},{143, 1, 1, 1},{169, 1, 1, 1},{243, 1, 1, 1},{286, 1, 1, 1},{343, 1, 1, 1},{429, 1, 1, 1},{572, 1, 1, 1},{625, 1, 1, 1},{720, 1, 1, 1},{1080, 1, 1, 1},{1001, 1, 1, 1},{1287, 1, 1, 1},{1400, 1, 1, 1},{1440, 1, 1, 1},{1920, 1, 1, 1},{2160, 1, 1, 1},{3024,1,1,1},{3500,1,1,1},
		{3840, 1, 1, 1},{4000 , 1, 1, 1},{4050, 1, 1, 1},{4320 , 1, 1, 1},{7000,1,1,1},{7680, 1, 1, 1},{9000, 1, 1, 1},{7680 * 5, 1, 1, 1},
		{(uint32_t)pow(3,10), 1, 1, 1},{(uint32_t)pow(3,11), 1, 1, 1},{(uint32_t)pow(3,12), 1, 1, 1},{(uint32_t)pow(3,13), 1, 1, 1},{(uint32_t)pow(3,14), 1, 1, 1},{(uint32_t)pow(3,15), 1, 1, 1},
		{(uint32_t)pow(5,5), 1, 1, 1},{(uint32_t)pow(5,6), 1, 1, 1},{(uint32_t)pow(5,7), 1, 1, 1},{(uint32_t)pow(5,8), 1, 1, 1},{(uint32_t)pow(5,9), 1, 1, 1},
		{(uint32_t)pow(7,4), 1, 1, 1},{(uint32_t)pow(7,5), 1, 1, 1},{(uint32_t)pow(7,6), 1, 1, 1},{(uint32_t)pow(7,7), 1, 1, 1},{(uint32_t)pow(7,8), 1, 1, 1},
		{(uint32_t)pow(11,3), 1, 1, 1},{(uint32_t)pow(11,4), 1, 1, 1},{(uint32_t)pow(11,5), 1, 1, 1},{(uint32_t)pow(11,6), 1, 1, 1},
		{(uint32_t)pow(13,3), 1, 1, 1},{(uint32_t)pow(13,4), 1, 1, 1},{(uint32_t)pow(13,5), 1, 1, 1},{(uint32_t)pow(13,6), 1, 1, 1},
		{8, 3, 1, 2},{8, 5, 1, 2},{8, 6, 1, 2},{8, 7, 1, 2},{8, 9, 1, 2},{8, 10, 1, 2},{8, 11, 1, 2},{8, 12, 1, 2},{8, 13, 1, 2},{8, 14, 1, 2},{8, 15, 1, 2},{8, 21, 1, 2},{8, 22, 1, 2},{8, 24, 1, 2},
		{8, 25, 1, 2},{8, 26, 1, 2},{8, 27, 1, 2},{8, 28, 1, 2},{8, 30, 1, 2},{8, 33, 1, 2},{8, 35, 1, 2},{8, 39, 1, 2},{8, 44, 1, 2},{8, 45, 1, 2},{8, 49, 1, 2},{8, 52, 1, 2},{8, 56, 1, 2},{8, 60, 1, 2},{8, 66, 1, 2},{8, 81, 1, 2},{8, 125, 1, 2},{8, 243, 1, 2},{8, 343, 1, 2},
		{8, 625, 1, 2},{8, 720, 1, 2},{8, 1080, 1, 2},{8, 1400, 1, 2},{8, 1440, 1, 2},{8, 1920, 1, 2},{8, 2160, 1, 2},{8, 3024, 1, 2},{8, 3500, 1, 2},
		{8, 3840, 1, 2},{8, 4000, 1, 2},{8, 4050, 1, 2},{8, 4320, 1, 2},{8, 7000, 1, 2},{8, 7680, 1, 2},{8, 4050 * 3, 1, 2},{8, 7680 * 5, 1, 2}, {720, 480, 1, 2},{1280, 720, 1, 2},{1920, 1080, 1, 2}, {2560, 1440, 1, 2},{3840, 2160, 1, 2},{7680, 4320, 1, 2},
		{8, (uint32_t)pow(3,10), 1, 2},	{8, (uint32_t)pow(3,11), 1, 2}, {8, (uint32_t)pow(3,12), 1, 2}, {8, (uint32_t)pow(3,13), 1, 2}, {8, (uint32_t)pow(3,14), 1, 2}, {8, (uint32_t)pow(3,15), 1, 2},
		{8, (uint32_t)pow(5,5), 1, 2},	{8, (uint32_t)pow(5,6), 1, 2}, {8, (uint32_t)pow(5,7), 1, 2}, {8, (uint32_t)pow(5,8), 1, 2}, {8, (uint32_t)pow(5,9), 1, 2},
		{8, (uint32_t)pow(7,4), 1, 2},{8, (uint32_t)pow(7,5), 1, 2},{8, (uint32_t)pow(7,6), 1, 2},{8, (uint32_t)pow(7,7), 1, 2},{8, (uint32_t)pow(7,8), 1, 2},
		{8, (uint32_t)pow(11,3), 1, 2},{8, (uint32_t)pow(11,4), 1, 2},{8, (uint32_t)pow(11,5), 1, 2},{8, (uint32_t)pow(11,6), 1, 2},
		{8, (uint32_t)pow(13,3), 1, 2},{8, (uint32_t)pow(13,4), 1, 2},{8, (uint32_t)pow(13,5), 1, 2},{8, (uint32_t)pow(13,6), 1, 2},
		{3, 3, 3, 3},{5, 5, 5, 3},{6, 6, 6, 3},{7, 7, 7, 3},{9, 9, 9, 3},{10, 10, 10, 3},{11, 11, 11, 3},{12, 12, 12, 3},{13, 13, 13, 3},{14, 14, 14, 3},
		{15, 15, 15, 3},{21, 21, 21, 3},{22, 22, 22, 3},{24, 24, 24, 3},{25, 25, 25, 3},{26, 26, 26, 3},{27, 27, 27, 3},{28, 28, 28, 3},{30, 30, 30, 3},{33, 33, 33, 3},{35, 35, 35, 3},{39, 39, 39, 3},{42, 42, 42, 3},{44, 44, 44, 3},{45, 45, 45, 3},{49, 49, 49, 3},{52, 52, 52, 3},{56, 56, 56, 3},{60, 60, 60, 3},{81, 81, 81, 3},
		{121, 121, 121, 3},{125, 125, 125, 3},{143, 143, 143, 3},{169, 169, 169, 3},{243, 243, 243, 3}
	};

	double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples

	for (int n = 0; n < num_benchmark_samples; n++) {
		for (int r = 0; r < num_runs; r++) {

			fftwf_complex* inputC;
			fftw_complex* inputC_double;
			uint32_t dims[3] = { benchmark_dimensions[n][0] , benchmark_dimensions[n][1] ,benchmark_dimensions[n][2] };

			inputC = (fftwf_complex*)(malloc(sizeof(fftwf_complex) * dims[0] * dims[1] * dims[2]));
			inputC_double = (fftw_complex*)(malloc(sizeof(fftw_complex) * dims[0] * dims[1] * dims[2]));
			for (int l = 0; l < dims[2]; l++) {
				for (int j = 0; j < dims[1]; j++) {
					for (int i = 0; i < dims[0]; i++) {
						inputC[i + j * dims[0] + l * dims[0] * dims[1]][0] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						inputC[i + j * dims[0] + l * dims[0] * dims[1]][1] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						inputC_double[i + j * dims[0] + l * dims[0] * dims[1]][0] = (double)inputC[i + j * dims[0] + l * dims[0] * dims[1]][0];
						inputC_double[i + j * dims[0] + l * dims[0] * dims[1]][1] = (double)inputC[i + j * dims[0] + l * dims[0] * dims[1]][1];
					}
				}
			}

			fftw_plan p;

			fftw_complex* output_FFTW = (fftw_complex*)(malloc(sizeof(fftw_complex) * dims[0] * dims[1] * dims[2]));

			switch (benchmark_dimensions[n][3]) {
			case 1:
				p = fftw_plan_dft_1d(benchmark_dimensions[n][0], inputC_double, output_FFTW, -1, FFTW_ESTIMATE);
				break;
			case 2:
				p = fftw_plan_dft_2d(benchmark_dimensions[n][1], benchmark_dimensions[n][0], inputC_double, output_FFTW, -1, FFTW_ESTIMATE);
				break;
			case 3:
				p = fftw_plan_dft_3d(benchmark_dimensions[n][2], benchmark_dimensions[n][1], benchmark_dimensions[n][0], inputC_double, output_FFTW, -1, FFTW_ESTIMATE);
				break;
			}

			fftw_execute(p);

			float totTime = 0;
			int num_iter = 1;

			//VkFFT part

			VkFFTConfiguration configuration = {};
			VkFFTApplication app = {};
			configuration.FFTdim = benchmark_dimensions[n][3]; //FFT dimension, 1D, 2D or 3D (default 1).
			configuration.size[0] = benchmark_dimensions[n][0]; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			configuration.size[1] = benchmark_dimensions[n][1];
			configuration.size[2] = benchmark_dimensions[n][2];

			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			configuration.device = &vkGPU->device;
#if(VKFFT_BACKEND==0)
			configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
			configuration.fence = &vkGPU->fence;
			configuration.commandPool = &vkGPU->commandPool;
			configuration.physicalDevice = &vkGPU->physicalDevice;
			configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization
#endif			

			uint32_t numBuf = 1;

			//Allocate buffers for the input data. - we use 4 in this example
			uint64_t* bufferSize = (uint64_t*)malloc(sizeof(uint64_t) * numBuf);
			for (uint32_t i = 0; i < numBuf; i++) {
				bufferSize[i] = {};
				bufferSize[i] = (uint64_t)sizeof(float) * 2 * configuration.size[0] * configuration.size[1] * configuration.size[2] / numBuf;
			}
#if(VKFFT_BACKEND==0)
			VkBuffer* buffer = (VkBuffer*)malloc(numBuf * sizeof(VkBuffer));
			VkDeviceMemory* bufferDeviceMemory = (VkDeviceMemory*)malloc(numBuf * sizeof(VkDeviceMemory));
#elif(VKFFT_BACKEND==1)
			cuFloatComplex* buffer = 0;
#elif(VKFFT_BACKEND==2)
			hipFloatComplex* buffer = 0;
#endif			
			for (uint32_t i = 0; i < numBuf; i++) {
#if(VKFFT_BACKEND==0)
				buffer[i] = {};
				bufferDeviceMemory[i] = {};
				allocateFFTBuffer(vkGPU, &buffer[i], &bufferDeviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize[i]);
#elif(VKFFT_BACKEND==1)
				cudaMalloc((void**)&buffer, bufferSize[i]);
#elif(VKFFT_BACKEND==2)
				hipMalloc((void**)&buffer, bufferSize[i]);
#endif
			}

			configuration.bufferNum = numBuf;

#if(VKFFT_BACKEND==0)
			configuration.buffer = buffer;
#elif(VKFFT_BACKEND==1)
			configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==2)
			configuration.buffer = (void**)&buffer;
#endif

			configuration.bufferSize = bufferSize;

			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
			uint64_t shift = 0;
			for (uint32_t i = 0; i < numBuf; i++) {
#if(VKFFT_BACKEND==0)
				transferDataFromCPU(vkGPU, (inputC + shift / sizeof(fftwf_complex)), &buffer[i], bufferSize[i]);
#elif(VKFFT_BACKEND==1)
				cudaMemcpy(buffer, inputC, bufferSize[i], cudaMemcpyHostToDevice);
#elif(VKFFT_BACKEND==2)
				hipMemcpy(buffer, inputC, bufferSize[i], hipMemcpyHostToDevice);
#endif
				shift += bufferSize[i];
			}
			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			res = initializeVkFFT(&app, configuration);
			if (res != 0) return res;
			//Submit FFT+iFFT.
			//num_iter = 1;
			performVulkanFFT(vkGPU, &app, -1, num_iter);
			fftwf_complex* output_VkFFT = (fftwf_complex*)(malloc(sizeof(fftwf_complex) * dims[0] * dims[1] * dims[2]));

			//Transfer data from GPU using staging buffer.
			shift = 0;
			for (uint32_t i = 0; i < numBuf; i++) {
#if(VKFFT_BACKEND==0)
				transferDataToCPU(vkGPU, (output_VkFFT + shift / sizeof(fftwf_complex)), &buffer[i], bufferSize[i]);
#elif(VKFFT_BACKEND==1)
				cudaMemcpy(output_VkFFT, buffer, bufferSize[i], cudaMemcpyDeviceToHost);
#elif(VKFFT_BACKEND==2)
				hipMemcpy(output_VkFFT, buffer, bufferSize[i], hipMemcpyDeviceToHost);
#endif
				shift += bufferSize[i];
			}
			float avg_difference[2] = { 0,0 };
			float max_difference[2] = { 0,0 };
			float avg_eps[2] = { 0,0 };
			float max_eps[2] = { 0,0 };
			for (int l = 0; l < dims[2]; l++) {
				for (int j = 0; j < dims[1]; j++) {
					for (int i = 0; i < dims[0]; i++) {
						int loc_i = i;
						int loc_j = j;
						int loc_l = l;

						//if (file_output) fprintf(output, "%f %f - %f %f \n", output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] / N, output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1] / N, output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][0], output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][1]);
						//if (i > dims[0] - 10)
						//	printf("%f %f - %f %f \n", output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] / N, output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1] / N, output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][0], output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][1]);
						float current_data_norm = sqrt(output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] * output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] + output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1] * output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1]);

						float current_diff_x_VkFFT = (output_VkFFT[loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]][0] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0]);
						float current_diff_y_VkFFT = (output_VkFFT[loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]][1] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1]);
						float current_diff_norm_VkFFT = sqrt(current_diff_x_VkFFT * current_diff_x_VkFFT + current_diff_y_VkFFT * current_diff_y_VkFFT);
						if (current_diff_norm_VkFFT > max_difference[1]) max_difference[1] = current_diff_norm_VkFFT;
						avg_difference[1] += current_diff_norm_VkFFT;
						if ((current_diff_norm_VkFFT / current_data_norm > max_eps[1]) && (current_data_norm > 1e-10)) {
							max_eps[1] = current_diff_norm_VkFFT / current_data_norm;
						}
						avg_eps[1] += (current_data_norm > 1e-10) ? current_diff_norm_VkFFT / current_data_norm : 0;
					}
				}
			}
			avg_difference[0] /= (dims[0] * dims[1] * dims[2]);
			avg_eps[0] /= (dims[0] * dims[1] * dims[2]);
			avg_difference[1] /= (dims[0] * dims[1] * dims[2]);
			avg_eps[1] /= (dims[0] * dims[1] * dims[2]);
			if (file_output)
				fprintf(output, "VkFFT System: %dx%dx%d avg_difference: %f max_difference: %f avg_eps: %f max_eps: %f\n", dims[0], dims[1], dims[2], avg_difference[1], max_difference[1], avg_eps[1], max_eps[1]);
			printf("VkFFT System: %dx%dx%d avg_difference: %f max_difference: %f avg_eps: %f max_eps: %f\n", dims[0], dims[1], dims[2], avg_difference[1], max_difference[1], avg_eps[1], max_eps[1]);
			free(output_VkFFT);
			for (uint32_t i = 0; i < numBuf; i++) {

#if(VKFFT_BACKEND==0)
				vkDestroyBuffer(vkGPU->device, buffer[i], NULL);
				vkFreeMemory(vkGPU->device, bufferDeviceMemory[i], NULL);
#elif(VKFFT_BACKEND==1)
				cudaFree(buffer);
#elif(VKFFT_BACKEND==2)
				hipFree(buffer);
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
	return res;
}
uint32_t sample_15(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	uint32_t res = 0;
	if (file_output)
		fprintf(output, "15 - VkFFT / FFTW R2C+C2R precision test in single precision\n");
	printf("15 - VkFFT / FFTW R2C+C2R precision test in single precision\n");

	const uint32_t num_benchmark_samples = 24;
	const uint32_t num_runs = 1;
	uint32_t benchmark_dimensions[num_benchmark_samples][4] = { {1024, 1024, 1, 2}, {64, 64, 1, 2}, {256, 256, 1, 2}, {1024, 256, 1, 2}, {512, 512, 1, 2}, {1024, 1024, 1, 2},  {4096, 256, 1, 2}, {2048, 1024, 1, 2},{4096, 2048, 1, 2}, {4096, 4096, 1, 2}, {720, 480, 1, 2},{1280, 720, 1, 2},{1920, 1080, 1, 2}, {2560, 1440, 1, 2},{3840, 2160, 1, 2},
																{32, 32, 32, 3}, {64, 64, 64, 3}, {256, 256, 32, 3},  {1024, 256, 32, 3},  {256, 256, 256, 3}, {2048, 1024, 8, 3},  {512, 512, 128, 3}, {2048, 256, 256, 3}, {4096, 512, 8, 3} };


	double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples

	for (int n = 0; n < num_benchmark_samples; n++) {
		for (int r = 0; r < num_runs; r++) {

			float* inputC;
			double* inputC_double;
			uint32_t dims[3] = { benchmark_dimensions[n][0] , benchmark_dimensions[n][1] ,benchmark_dimensions[n][2] };

			inputC = (float*)(malloc(sizeof(float) * (dims[0]) * dims[1] * dims[2]));
			inputC_double = (double*)(malloc(sizeof(double) * (dims[0]) * dims[1] * dims[2]));
			for (int l = 0; l < dims[2]; l++) {
				for (int j = 0; j < dims[1]; j++) {
					for (int i = 0; i < dims[0]; i++) {
						inputC[i + j * (dims[0]) + l * (dims[0]) * dims[1]] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						inputC_double[i + j * (dims[0]) + l * (dims[0]) * dims[1]] = (double)inputC[i + j * (dims[0]) + l * (dims[0]) * dims[1]];
					}
				}
			}

			fftw_plan p;

			fftw_complex* output_FFTW = (fftw_complex*)(malloc(sizeof(fftw_complex) * (dims[0] / 2 + 1) * dims[1] * dims[2]));
			double* output_FFTWR = (double*)(malloc(sizeof(double) * (dims[0]) * dims[1] * dims[2]));
			switch (benchmark_dimensions[n][3]) {
			case 1:
				p = fftw_plan_dft_r2c_1d(benchmark_dimensions[n][0], inputC_double, output_FFTW, FFTW_ESTIMATE);
				break;
			case 2:
				p = fftw_plan_dft_r2c_2d(benchmark_dimensions[n][1], benchmark_dimensions[n][0], inputC_double, output_FFTW, FFTW_ESTIMATE);
				break;
			case 3:
				p = fftw_plan_dft_r2c_3d(benchmark_dimensions[n][2], benchmark_dimensions[n][1], benchmark_dimensions[n][0], inputC_double, output_FFTW, FFTW_ESTIMATE);
				break;
			}

			fftw_execute(p);
			fftw_destroy_plan(p);
			switch (benchmark_dimensions[n][3]) {
			case 1:
				p = fftw_plan_dft_c2r_1d(benchmark_dimensions[n][0], output_FFTW, output_FFTWR, FFTW_ESTIMATE);
				break;
			case 2:
				p = fftw_plan_dft_c2r_2d(benchmark_dimensions[n][1], benchmark_dimensions[n][0], output_FFTW, output_FFTWR, FFTW_ESTIMATE);
				break;
			case 3:
				p = fftw_plan_dft_c2r_3d(benchmark_dimensions[n][2], benchmark_dimensions[n][1], benchmark_dimensions[n][0], output_FFTW, output_FFTWR, FFTW_ESTIMATE);
				break;
			}
			fftw_execute(p);
#ifdef USE_cuFFT
			//fftwf_complex* output_extFFT = (fftwf_complex*)(malloc(sizeof(fftwf_complex) * (dims[0]/2+1) * dims[1] * dims[2]));
			float* output_extFFT = (float*)(malloc(sizeof(float) * (dims[0]) * dims[1] * dims[2]));
			launch_precision_cuFFT_r2c(inputC, (void*)output_extFFT, benchmark_dimensions[n]);
#endif // USE_cuFFT
#ifdef USE_rocFFT
			//fftwf_complex* output_extFFT = (fftwf_complex*)(malloc(sizeof(fftwf_complex) * (dims[0]/2+1) * dims[1] * dims[2]));
			float* output_extFFT = (float*)(malloc(sizeof(float) * (dims[0]) * dims[1] * dims[2]));
			launch_precision_rocFFT_r2c(inputC, (void*)output_extFFT, benchmark_dimensions[n]);
#endif // USE_rocFFT
			float totTime = 0;
			int num_iter = 1;

			//VkFFT part

			VkFFTConfiguration configuration = {};
			VkFFTApplication app = {};
			configuration.FFTdim = benchmark_dimensions[n][3]; //FFT dimension, 1D, 2D or 3D (default 1).
			configuration.size[0] = benchmark_dimensions[n][0]; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			configuration.size[1] = benchmark_dimensions[n][1];
			configuration.size[2] = benchmark_dimensions[n][2];
			configuration.performR2C = 1;
			//configuration.disableMergeSequencesR2C = 1;
			configuration.inverseReturnToInputBuffer = 1;

			//configuration.coalescedMemory = 64;
			//configuration.useLUT = 1;
			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			configuration.device = &vkGPU->device;
#if(VKFFT_BACKEND==0)
			configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
			configuration.fence = &vkGPU->fence;
			configuration.commandPool = &vkGPU->commandPool;
			configuration.physicalDevice = &vkGPU->physicalDevice;
			configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization
#endif			

			uint32_t numBuf = 1;

			uint64_t* inputBufferSize = (uint64_t*)malloc(sizeof(uint64_t) * numBuf);
			for (uint32_t i = 0; i < numBuf; i++) {
				inputBufferSize[i] = {};
				inputBufferSize[i] = (uint64_t)sizeof(float) * configuration.size[0] * configuration.size[1] * configuration.size[2] / numBuf;
			}
			uint64_t* bufferSize = (uint64_t*)malloc(sizeof(uint64_t) * numBuf);
			for (uint32_t i = 0; i < numBuf; i++) {
				bufferSize[i] = {};
				bufferSize[i] = (uint64_t)sizeof(float) * 2 * (configuration.size[0] / 2 + 1) * configuration.size[1] * configuration.size[2] / numBuf;
			}
#if(VKFFT_BACKEND==0)
			VkBuffer* ibuffer = (VkBuffer*)malloc(numBuf * sizeof(VkBuffer));
			VkDeviceMemory* ibufferDeviceMemory = (VkDeviceMemory*)malloc(numBuf * sizeof(VkDeviceMemory));
			VkBuffer* buffer = (VkBuffer*)malloc(numBuf * sizeof(VkBuffer));
			VkDeviceMemory* bufferDeviceMemory = (VkDeviceMemory*)malloc(numBuf * sizeof(VkDeviceMemory));
#elif(VKFFT_BACKEND==1)
			float* ibuffer = 0;
			cuFloatComplex* buffer = 0;
#elif(VKFFT_BACKEND==2)
			float* ibuffer = 0;
			hipFloatComplex* buffer = 0;
#endif
			for (uint32_t i = 0; i < numBuf; i++) {
#if(VKFFT_BACKEND==0)
				buffer[i] = {};
				bufferDeviceMemory[i] = {};
				allocateFFTBuffer(vkGPU, &buffer[i], &bufferDeviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize[i]);
				ibuffer[i] = {};
				ibufferDeviceMemory[i] = {};
				allocateFFTBuffer(vkGPU, &ibuffer[i], &ibufferDeviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, inputBufferSize[i]);
#elif(VKFFT_BACKEND==1)
				cudaMalloc((void**)&ibuffer, inputBufferSize[i]);
				cudaMalloc((void**)&buffer, bufferSize[i]);
#elif(VKFFT_BACKEND==2)
				hipMalloc((void**)&ibuffer, inputBufferSize[i]);
				hipMalloc((void**)&buffer, bufferSize[i]);
#endif
			}
			configuration.inputBufferNum = numBuf;
			configuration.bufferNum = numBuf;
			configuration.bufferSize = bufferSize;
#if(VKFFT_BACKEND==0)
			configuration.isInputFormatted = 1;
			configuration.inputBufferStride[0] = configuration.size[0];
			configuration.inputBufferStride[1] = configuration.size[0] * configuration.size[1];
			configuration.inputBufferStride[2] = configuration.size[0] * configuration.size[1] * configuration.size[2];
			configuration.inputBuffer = ibuffer;
			configuration.buffer = buffer;
#elif(VKFFT_BACKEND==1)
			configuration.isInputFormatted = 1;
			configuration.inputBufferStride[0] = configuration.size[0];
			configuration.inputBufferStride[1] = configuration.size[0] * configuration.size[1];
			configuration.inputBufferStride[2] = configuration.size[0] * configuration.size[1] * configuration.size[2];
			configuration.inputBuffer = (void**)&ibuffer;
			configuration.buffer = (void**)&buffer;

#elif(VKFFT_BACKEND==2)
			configuration.isInputFormatted = 1;
			configuration.inputBufferStride[0] = configuration.size[0];
			configuration.inputBufferStride[1] = configuration.size[0] * configuration.size[1];
			configuration.inputBufferStride[2] = configuration.size[0] * configuration.size[1] * configuration.size[2];
			configuration.inputBuffer = (void**)&ibuffer;
			configuration.buffer = (void**)&buffer;
#endif
			configuration.inputBufferSize = inputBufferSize;


			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
			uint64_t shift = 0;
			for (uint32_t i = 0; i < numBuf; i++) {
#if(VKFFT_BACKEND==0)
				transferDataFromCPU(vkGPU, (inputC + shift / sizeof(fftwf_complex)), &ibuffer[i], inputBufferSize[i]);
#elif(VKFFT_BACKEND==1)
				cudaMemcpy(ibuffer, inputC, inputBufferSize[i], cudaMemcpyHostToDevice);
#elif(VKFFT_BACKEND==2)
				hipMemcpy(ibuffer, inputC, inputBufferSize[i], hipMemcpyHostToDevice);
#endif
				shift += inputBufferSize[i];
			}
			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			res = initializeVkFFT(&app, configuration);
			if (res != 0) return res;

			//Submit FFT+iFFT.
			//num_iter = 1;
			performVulkanFFT(vkGPU, &app, -1, num_iter);
			performVulkanFFT(vkGPU, &app, 1, num_iter);
			//fftwf_complex* output_VkFFT = (fftwf_complex*)(malloc(sizeof(fftwf_complex) * (dims[0] / 2 + 1) * dims[1] * dims[2]));
			float* output_VkFFT = (float*)(malloc(bufferSize[0]));

			//Transfer data from GPU using staging buffer.
			shift = 0;
			for (uint32_t i = 0; i < numBuf; i++) {
#if(VKFFT_BACKEND==0)
				//transferDataToCPU(vkGPU, (output_VkFFT + shift / sizeof(fftwf_complex)), &buffer[i], bufferSize[i]);
				transferDataToCPU(vkGPU, (output_VkFFT + shift / sizeof(fftwf_complex)), &ibuffer[i], inputBufferSize[i]);
#elif(VKFFT_BACKEND==1)
				//cudaMemcpy(output_VkFFT, buffer, bufferSize[i], cudaMemcpyDeviceToHost);
				cudaMemcpy(output_VkFFT, ibuffer, inputBufferSize[i], cudaMemcpyDeviceToHost);
#elif(VKFFT_BACKEND==2)
				//hipMemcpy(output_VkFFT, buffer, bufferSize[i], hipMemcpyDeviceToHost);
				hipMemcpy(output_VkFFT, ibuffer, inputBufferSize[i], hipMemcpyDeviceToHost);
#endif
				shift += inputBufferSize[i];
			}
			float avg_difference[2] = { 0,0 };
			float max_difference[2] = { 0,0 };
			float avg_eps[2] = { 0,0 };
			float max_eps[2] = { 0,0 };

			for (int l = 0; l < dims[2]; l++) {
				for (int j = 0; j < dims[1]; j++) {
					for (int i = 0; i < dims[0]; i++) {
						int loc_i = i;
						int loc_j = j;
						int loc_l = l;

						//if (file_output) fprintf(output, "%f %f - %f %f \n", output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] / N, output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1] / N, output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][0], output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][1]);
						//if (i > dims[0] - 10)
						//printf("%f %f - %f %f\n", output_FFTW[i + j * (dims[0]) + l * (dims[0]) * dims[1]][0], output_FFTW[i + j * (dims[0]) + l * (dims[0]) * dims[1]][1], output_VkFFT[2*(loc_i + loc_j * (dims[0]) + loc_l * (dims[0]) * dims[1])], output_VkFFT[1+2 * (loc_i + loc_j * (dims[0]) + loc_l * (dims[0]) * dims[1])]);

						//printf("%f - %f \n", output_FFTWR[i + j * (dims[0]) + l * (dims[0]) * dims[1]], output_VkFFT[(loc_i + loc_j * (dims[0]) + loc_l * (dims[0]) * dims[1])]);
						float current_data_norm = sqrt(output_FFTWR[i + j * (dims[0]) + l * (dims[0]) * dims[1]] * output_FFTWR[i + j * (dims[0]) + l * (dims[0]) * dims[1]]);
#if defined(USE_cuFFT) || defined(USE_rocFFT)
						float current_diff_x_extFFT = (output_extFFT[loc_i + loc_j * (dims[0]) + loc_l * (dims[0]) * dims[1]] - output_FFTWR[i + j * (dims[0]) + l * (dims[0]) * dims[1]]);
						float current_diff_norm_extFFT = sqrt(current_diff_x_extFFT * current_diff_x_extFFT);
						if (current_diff_norm_extFFT > max_difference[0]) max_difference[0] = current_diff_norm_extFFT;
						avg_difference[0] += current_diff_norm_extFFT;
						if ((current_diff_norm_extFFT / current_data_norm > max_eps[0]) && (current_data_norm > 1e-10)) {
							max_eps[0] = current_diff_norm_extFFT / current_data_norm;
						}
						avg_eps[0] += (current_data_norm > 1e-10) ? current_diff_norm_extFFT / current_data_norm : 0;
#endif
						float current_diff_x_VkFFT = (output_VkFFT[loc_i + loc_j * (dims[0]) + loc_l * (dims[0]) * dims[1]] - output_FFTWR[i + j * (dims[0]) + l * (dims[0]) * dims[1]]);
						float current_diff_norm_VkFFT = sqrt(current_diff_x_VkFFT * current_diff_x_VkFFT);
						if (current_diff_norm_VkFFT > max_difference[1]) max_difference[1] = current_diff_norm_VkFFT;
						avg_difference[1] += current_diff_norm_VkFFT;
						if ((current_diff_norm_VkFFT / current_data_norm > max_eps[1]) && (current_data_norm > 1e-10)) {
							max_eps[1] = current_diff_norm_VkFFT / current_data_norm;
						}
						avg_eps[1] += (current_data_norm > 1e-10) ? current_diff_norm_VkFFT / current_data_norm : 0;
					}
				}
			}
			avg_difference[0] /= (dims[0] * dims[1] * dims[2]);
			avg_eps[0] /= (dims[0] * dims[1] * dims[2]);
			avg_difference[1] /= (dims[0] * dims[1] * dims[2]);
			avg_eps[1] /= (dims[0] * dims[1] * dims[2]);
#ifdef USE_cuFFT
			if (file_output)
				fprintf(output, "cuFFT System: %dx%dx%d avg_difference: %f max_difference: %f avg_eps: %f max_eps: %f\n", dims[0], dims[1], dims[2], avg_difference[0], max_difference[0], avg_eps[0], max_eps[0]);
			printf("cuFFT System: %dx%dx%d avg_difference: %f max_difference: %f avg_eps: %f max_eps: %f\n", dims[0], dims[1], dims[2], avg_difference[0], max_difference[0], avg_eps[0], max_eps[0]);
#endif
#ifdef USE_rocFFT
			if (file_output)
				fprintf(output, "rocFFT System: %dx%dx%d avg_difference: %f max_difference: %f avg_eps: %f max_eps: %f\n", dims[0], dims[1], dims[2], avg_difference[0], max_difference[0], avg_eps[0], max_eps[0]);
			printf("rocFFT System: %dx%dx%d avg_difference: %f max_difference: %f avg_eps: %f max_eps: %f\n", dims[0], dims[1], dims[2], avg_difference[0], max_difference[0], avg_eps[0], max_eps[0]);
#endif
			if (file_output)
				fprintf(output, "VkFFT System: %dx%dx%d avg_difference: %f max_difference: %f avg_eps: %f max_eps: %f\n", dims[0], dims[1], dims[2], avg_difference[1], max_difference[1], avg_eps[1], max_eps[1]);
			printf("VkFFT System: %dx%dx%d avg_difference: %f max_difference: %f avg_eps: %f max_eps: %f\n", dims[0], dims[1], dims[2], avg_difference[1], max_difference[1], avg_eps[1], max_eps[1]);
			free(output_VkFFT);
			for (uint32_t i = 0; i < numBuf; i++) {

#if(VKFFT_BACKEND==0)
				vkDestroyBuffer(vkGPU->device, buffer[i], NULL);
				vkFreeMemory(vkGPU->device, bufferDeviceMemory[i], NULL);
				vkDestroyBuffer(vkGPU->device, ibuffer[i], NULL);
				vkFreeMemory(vkGPU->device, ibufferDeviceMemory[i], NULL);
#elif(VKFFT_BACKEND==1)
				cudaFree(ibuffer);
				cudaFree(buffer);
#elif(VKFFT_BACKEND==2)
				hipFree(ibuffer);
				hipFree(buffer);
#endif

			}
#if(VKFFT_BACKEND==0)
			free(buffer);
			free(bufferDeviceMemory);
			free(ibuffer);
			free(ibufferDeviceMemory);
#endif
			free(inputBufferSize);
			free(bufferSize);
			deleteVkFFT(&app);
			free(inputC);
			fftw_destroy_plan(p);
			free(inputC_double);
			free(output_FFTW);
			free(output_FFTWR);
#if defined(USE_cuFFT) || defined(USE_rocFFT)
			free(output_extFFT);
#endif
		}
	}
	return res;
}
#endif
uint32_t sample_1000(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	uint32_t res = 0;
	if (file_output)
		fprintf(output, "1000 - VkFFT FFT + iFFT C2C benchmark 1D batched in single precision: all supported systems from 2 to 4096\n");
	printf("1000 - VkFFT FFT + iFFT C2C benchmark 1D batched in single precision: all supported systems from 2 to 4096\n");
	const int num_runs = 3;
	double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	//memory allocated on the CPU once, makes benchmark completion faster + avoids performance issues connected to frequent allocation/deallocation.
	float* buffer_input = (float*)malloc((uint64_t)4 * 2 * pow(2, 27));
	for (uint64_t i = 0; i < 2 * pow(2, 27); i++) {
		buffer_input[i] = 2 * ((float)rand()) / RAND_MAX - 1.0;
	}
	int num_systems = 0;
	for (uint32_t n = 1; n < 4097; n++) {
		double run_time[num_runs];
		for (uint32_t r = 0; r < num_runs; r++) {
			//Configuration + FFT application .
			VkFFTConfiguration configuration = {};
			VkFFTApplication app = {};
			//FFT + iFFT sample code.
			//Setting up FFT configuration for forward and inverse FFT.
			configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).
			configuration.size[0] = n;// 4 * pow(2, n); //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			if (n == 1) configuration.size[0] = 4096;
			uint32_t temp = configuration.size[0];

			for (uint32_t j = 2; j < 14; j++)
			{
				if (temp % j == 0) {
					temp /= j;
					j = 1;
				}
			}
			if (temp != 1) break;
			configuration.size[1] = pow(2, (uint32_t)log2(64 * 32 * pow(2, 16) / configuration.size[0]));
			if (configuration.size[1] < 1) configuration.size[1] = 1;
			configuration.size[2] = 1;
			num_systems++;
			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			configuration.device = &vkGPU->device;
#if(VKFFT_BACKEND==0)
			configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
			configuration.fence = &vkGPU->fence;
			configuration.commandPool = &vkGPU->commandPool;
			configuration.physicalDevice = &vkGPU->physicalDevice;
			configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization
#endif
			//Allocate buffer for the input data.
			uint64_t bufferSize = (uint64_t)sizeof(float) * 2 * configuration.size[0] * configuration.size[1] * configuration.size[2];;
#if(VKFFT_BACKEND==0)
			VkBuffer buffer = {};
			VkDeviceMemory bufferDeviceMemory = {};
			allocateFFTBuffer(vkGPU, &buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
			configuration.buffer = &buffer;
#elif(VKFFT_BACKEND==1)
			cuFloatComplex* buffer = 0;
			cudaMalloc((void**)&buffer, bufferSize);
			configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==2)
			hipFloatComplex* buffer = 0;
			hipMalloc((void**)&buffer, bufferSize);
			configuration.buffer = (void**)&buffer;
#endif

			configuration.bufferSize = &bufferSize;

			//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
			/*float* buffer_input = (float*)malloc(bufferSize);

			for (uint32_t k = 0; k < configuration.size[2]; k++) {
				for (uint32_t j = 0; j < configuration.size[1]; j++) {
					for (uint32_t i = 0; i < configuration.size[0]; i++) {
						buffer_input[2 * (i + j * configuration.size[0] + k * (configuration.size[0]) * configuration.size[1])] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						buffer_input[2 * (i + j * configuration.size[0] + k * (configuration.size[0]) * configuration.size[1]) + 1] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						}
					}
				}

			*/
			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
#if(VKFFT_BACKEND==0)
			transferDataFromCPU(vkGPU, buffer_input, &buffer, bufferSize);
#elif(VKFFT_BACKEND==1)
			cudaMemcpy(buffer, buffer_input, bufferSize, cudaMemcpyHostToDevice);
#elif(VKFFT_BACKEND==2)
			hipMemcpy(buffer, buffer_input, bufferSize, hipMemcpyHostToDevice);
#endif

			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			res = initializeVkFFT(&app, configuration);
			if (res != 0) return res;

			//Submit FFT+iFFT.
			uint32_t num_iter = ((3 * 4096 * 1024.0 * 1024.0) / bufferSize > 1000) ? 1000 : (3 * 4096 * 1024.0 * 1024.0) / bufferSize;
#if(VKFFT_BACKEND==0)
			if (vkGPU->physicalDeviceProperties.vendorID == 0x8086) num_iter /= 4;//smaller benchmark for Intel GPUs
#endif
			if (num_iter == 0) num_iter = 1;
			float totTime = performVulkanFFTiFFT(vkGPU, &app, num_iter);

			run_time[r] = totTime;
			if (n > 1) {
				if (r == num_runs - 1) {
					double std_error = 0;
					double avg_time = 0;
					for (uint32_t t = 0; t < num_runs; t++) {
						avg_time += run_time[t];
					}
					avg_time /= num_runs;
					for (uint32_t t = 0; t < num_runs; t++) {
						std_error += (run_time[t] - avg_time) * (run_time[t] - avg_time);
					}
					std_error = sqrt(std_error / num_runs);
					uint32_t num_tot_transfers = 0;
					for (uint32_t i = 0; i < configuration.FFTdim; i++)
						num_tot_transfers += app.localFFTPlan->numAxisUploads[i];
					num_tot_transfers *= 4;
					if (file_output)
						fprintf(output, "VkFFT System: %d %d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %d benchmark: %d bandwidth: %0.1f\n", configuration.size[0], configuration.size[1], bufferSize / 1024 / 1024, avg_time, std_error, num_iter, (int)(((double)bufferSize / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);

					printf("VkFFT System: %d %d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %d benchmark: %d bandwidth: %0.1f\n", configuration.size[0], configuration.size[1], bufferSize / 1024 / 1024, avg_time, std_error, num_iter, (int)(((double)bufferSize / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);
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
#endif

			deleteVkFFT(&app);

		}
	}
	free(buffer_input);
	benchmark_result /= (num_systems/num_runs - 1);

	if (file_output) {
		fprintf(output, "Benchmark score VkFFT: %d\n", (int)(benchmark_result));
#if(VKFFT_BACKEND==0)
		fprintf(output, "Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
#endif
	}
	printf("Benchmark score VkFFT: %d\n", (int)(benchmark_result));
#if(VKFFT_BACKEND==0)
	printf("Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
#endif
	return res;
}
uint32_t sample_1001(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	uint32_t res = 0;
	if (file_output)
		fprintf(output, "1001 - VkFFT FFT + iFFT C2C benchmark 1D batched in double precision: all supported systems from 2 to 4096\n");
	printf("1001 - VkFFT FFT + iFFT C2C benchmark 1D batched in double precision: all supported systems from 2 to 4096\n");
	const int num_runs = 3;
	double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	//memory allocated on the CPU once, makes benchmark completion faster + avoids performance issues connected to frequent allocation/deallocation.
	double* buffer_input = (double*)malloc((uint64_t)8 * 2 * pow(2, 27));
	for (uint64_t i = 0; i < 2 * pow(2, 27); i++) {
		buffer_input[i] = 2 * ((double)rand()) / RAND_MAX - 1.0;
	}
	int num_systems = 0;
	for (int n = 1; n < 4097; n++) {
		double run_time[num_runs];
		for (uint32_t r = 0; r < num_runs; r++) {
			//Configuration + FFT application .
			VkFFTConfiguration configuration = {};
			VkFFTApplication app = {};
			//FFT + iFFT sample code.
			//Setting up FFT configuration for forward and inverse FFT.
			configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).
			configuration.size[0] = n;// 4 * pow(2, n); //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			if (n == 1) configuration.size[0] = 4096;
			uint32_t temp = configuration.size[0];

			for (uint32_t j = 2; j < 14; j++)
			{
				if (temp % j == 0) {
					temp /= j;
					j = 1;
				}
			}
			if (temp != 1) break;
			configuration.size[1] = pow(2, (uint32_t)log2(64 * 32 * pow(2, 15) / configuration.size[0]));
			if (configuration.size[1] < 1) configuration.size[1] = 1;
			configuration.size[2] = 1;
			num_systems++;


			configuration.doublePrecision = true;

			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			configuration.device = &vkGPU->device;
#if(VKFFT_BACKEND==0)
			configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
			configuration.fence = &vkGPU->fence;
			configuration.commandPool = &vkGPU->commandPool;
			configuration.physicalDevice = &vkGPU->physicalDevice;
			configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization
#endif			

			//Allocate buffer for the input data.
			uint64_t bufferSize = (uint64_t)sizeof(double) * 2 * configuration.size[0] * configuration.size[1] * configuration.size[2];;
#if(VKFFT_BACKEND==0)
			VkBuffer buffer = {};
			VkDeviceMemory bufferDeviceMemory = {};
			allocateFFTBuffer(vkGPU, &buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
			configuration.buffer = &buffer;
#elif(VKFFT_BACKEND==1)
			cuFloatComplex* buffer = 0;
			cudaMalloc((void**)&buffer, bufferSize);
			configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==2)
			hipFloatComplex* buffer = 0;
			hipMalloc((void**)&buffer, bufferSize);
			configuration.buffer = (void**)&buffer;
#endif

			configuration.bufferSize = &bufferSize;
			//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
			/*float* buffer_input = (float*)malloc(bufferSize);

			for (uint32_t k = 0; k < configuration.size[2]; k++) {
				for (uint32_t j = 0; j < configuration.size[1]; j++) {
					for (uint32_t i = 0; i < configuration.size[0]; i++) {
						buffer_input[2 * (i + j * configuration.size[0] + k * (configuration.size[0]) * configuration.size[1])] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						buffer_input[2 * (i + j * configuration.size[0] + k * (configuration.size[0]) * configuration.size[1]) + 1] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						}
					}
				}
			*/
			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
#if(VKFFT_BACKEND==0)
			transferDataFromCPU(vkGPU, buffer_input, &buffer, bufferSize);
#elif(VKFFT_BACKEND==1)
			cudaMemcpy(buffer, buffer_input, bufferSize, cudaMemcpyHostToDevice);
#elif(VKFFT_BACKEND==2)
			hipMemcpy(buffer, buffer_input, bufferSize, hipMemcpyHostToDevice);
#endif
			//free(buffer_input);

			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			res = initializeVkFFT(&app, configuration);
			if (res != 0) return res;

			//Submit FFT+iFFT.
			uint32_t num_iter = ((4096 * 1024.0 * 1024.0) / bufferSize > 1000) ? 1000 : (4096 * 1024.0 * 1024.0) / bufferSize;
#if(VKFFT_BACKEND==0)
			if (vkGPU->physicalDeviceProperties.vendorID == 0x8086) num_iter /= 4;
#endif
			if (num_iter == 0) num_iter = 1;

			float totTime = performVulkanFFTiFFT(vkGPU, &app, num_iter);

			run_time[r] = totTime;
			if (n > 1) {
				if (r == num_runs - 1) {
					double std_error = 0;
					double avg_time = 0;
					for (uint32_t t = 0; t < num_runs; t++) {
						avg_time += run_time[t];
					}
					avg_time /= num_runs;
					for (uint32_t t = 0; t < num_runs; t++) {
						std_error += (run_time[t] - avg_time) * (run_time[t] - avg_time);
					}
					std_error = sqrt(std_error / num_runs);
					uint32_t num_tot_transfers = 0;
					for (uint32_t i = 0; i < configuration.FFTdim; i++)
						num_tot_transfers += app.localFFTPlan->numAxisUploads[i];
					num_tot_transfers *= 4;
					if (file_output)
						fprintf(output, "VkFFT System: %d %d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %d benchmark: %d bandwidth: %0.1f\n", configuration.size[0], configuration.size[1], bufferSize / 1024 / 1024, avg_time, std_error, num_iter, (int)(((double)bufferSize * sizeof(float) / sizeof(double) / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);

					printf("VkFFT System: %d %d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %d benchmark: %d bandwidth: %0.1f\n", configuration.size[0], configuration.size[1], bufferSize / 1024 / 1024, avg_time, std_error, num_iter, (int)(((double)bufferSize * sizeof(float) / sizeof(double) / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);
					benchmark_result += ((double)bufferSize * sizeof(float) / sizeof(double) / 1024) / avg_time;
				}


			}

#if(VKFFT_BACKEND==0)
			vkDestroyBuffer(vkGPU->device, buffer, NULL);
			vkFreeMemory(vkGPU->device, bufferDeviceMemory, NULL);
#elif(VKFFT_BACKEND==1)
			cudaFree(buffer);
#elif(VKFFT_BACKEND==2)
			hipFree(buffer);
#endif
			deleteVkFFT(&app);

		}
	}
	free(buffer_input);
	benchmark_result /= (num_systems/num_runs - 1);
	if (file_output) {
		fprintf(output, "Benchmark score VkFFT: %d\n", (int)(benchmark_result));
#if(VKFFT_BACKEND==0)
		fprintf(output, "Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
#endif
	}
	printf("Benchmark score VkFFT: %d\n", (int)(benchmark_result));
#if(VKFFT_BACKEND==0)
	printf("Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
#endif
	return res;
}
uint32_t sample_1003(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	uint32_t res = 0;
	if (file_output)
		fprintf(output, "1003 - VkFFT FFT + iFFT C2C multidimensional benchmark in single precision: all supported cubes from 2 to 512\n");
	printf("1003 - VkFFT FFT + iFFT C2C multidimensional benchmark in single precision: all supported cubes from 2 to 512\n");
	const int num_runs = 3;
	double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	//memory allocated on the CPU once, makes benchmark completion faster + avoids performance issues connected to frequent allocation/deallocation.
	float* buffer_input = (float*)malloc((uint64_t)4 * 2 * pow(2, 27));
	for (uint64_t i = 0; i < 2 * pow(2, 27); i++) {
		buffer_input[i] = 2 * ((float)rand()) / RAND_MAX - 1.0;
	}
	int num_systems = 0;
	for (int n = 1; n < 513; n++) {
		double run_time[num_runs];
		for (uint32_t r = 0; r < num_runs; r++) {
			//Configuration + FFT application .
			VkFFTConfiguration configuration = {};
			VkFFTApplication app = {};
			//FFT + iFFT sample code.
			//Setting up FFT configuration for forward and inverse FFT.
			configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).
			configuration.size[0] = n;// 4 * pow(2, n); //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			if (n == 1) configuration.size[0] = 512;
			uint32_t temp = configuration.size[0];

			for (uint32_t j = 2; j < 14; j++)
			{
				if (temp % j == 0) {
					temp /= j;
					j = 1;
				}
			}
			if (temp != 1) break;
			configuration.size[1] = configuration.size[0];
			configuration.size[2] = configuration.size[0];
			num_systems++;
			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			configuration.device = &vkGPU->device;
#if(VKFFT_BACKEND==0)
			configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
			configuration.fence = &vkGPU->fence;
			configuration.commandPool = &vkGPU->commandPool;
			configuration.physicalDevice = &vkGPU->physicalDevice;
			configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization
#endif
			//Allocate buffer for the input data.
			uint64_t bufferSize = (uint64_t)sizeof(float) * 2 * configuration.size[0] * configuration.size[1] * configuration.size[2];;
#if(VKFFT_BACKEND==0)
			VkBuffer buffer = {};
			VkDeviceMemory bufferDeviceMemory = {};
			allocateFFTBuffer(vkGPU, &buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
			configuration.buffer = &buffer;
#elif(VKFFT_BACKEND==1)
			cuFloatComplex* buffer = 0;
			cudaMalloc((void**)&buffer, bufferSize);
			configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==2)
			hipFloatComplex* buffer = 0;
			hipMalloc((void**)&buffer, bufferSize);
			configuration.buffer = (void**)&buffer;
#endif

			configuration.bufferSize = &bufferSize;

			//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
			/*float* buffer_input = (float*)malloc(bufferSize);

			for (uint32_t k = 0; k < configuration.size[2]; k++) {
				for (uint32_t j = 0; j < configuration.size[1]; j++) {
					for (uint32_t i = 0; i < configuration.size[0]; i++) {
						buffer_input[2 * (i + j * configuration.size[0] + k * (configuration.size[0]) * configuration.size[1])] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						buffer_input[2 * (i + j * configuration.size[0] + k * (configuration.size[0]) * configuration.size[1]) + 1] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						}
					}
				}

			*/
			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
#if(VKFFT_BACKEND==0)
			transferDataFromCPU(vkGPU, buffer_input, &buffer, bufferSize);
#elif(VKFFT_BACKEND==1)
			cudaMemcpy(buffer, buffer_input, bufferSize, cudaMemcpyHostToDevice);
#elif(VKFFT_BACKEND==2)
			hipMemcpy(buffer, buffer_input, bufferSize, hipMemcpyHostToDevice);
#endif

			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			res = initializeVkFFT(&app, configuration);
			if (res != 0) return res;

			//Submit FFT+iFFT.
			uint32_t num_iter = ((3 * 4096 * 1024.0 * 1024.0) / bufferSize > 1000) ? 1000 : (3 * 4096 * 1024.0 * 1024.0) / bufferSize;
#if(VKFFT_BACKEND==0)
			if (vkGPU->physicalDeviceProperties.vendorID == 0x8086) num_iter /= 4;//smaller benchmark for Intel GPUs
#endif
			if (num_iter == 0) num_iter = 1;
			float totTime = performVulkanFFTiFFT(vkGPU, &app, num_iter);

			run_time[r] = totTime;
			if (n > 1) {
				if (r == num_runs - 1) {
					double std_error = 0;
					double avg_time = 0;
					for (uint32_t t = 0; t < num_runs; t++) {
						avg_time += run_time[t];
					}
					avg_time /= num_runs;
					for (uint32_t t = 0; t < num_runs; t++) {
						std_error += (run_time[t] - avg_time) * (run_time[t] - avg_time);
					}
					std_error = sqrt(std_error / num_runs);
					uint32_t num_tot_transfers = 0;
					for (uint32_t i = 0; i < configuration.FFTdim; i++)
						num_tot_transfers += app.localFFTPlan->numAxisUploads[i];
					num_tot_transfers *= 4;
					if (file_output)
						fprintf(output, "VkFFT System: %d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %d benchmark: %d bandwidth: %0.1f\n", configuration.size[0], bufferSize / 1024 / 1024, avg_time, std_error, num_iter, (int)(((double)bufferSize / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);

					printf("VkFFT System: %d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %d benchmark: %d bandwidth: %0.1f\n", configuration.size[0], bufferSize / 1024 / 1024, avg_time, std_error, num_iter, (int)(((double)bufferSize / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);
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
#endif

			deleteVkFFT(&app);

		}
	}
	free(buffer_input);
	benchmark_result /= (num_systems/num_runs - 1);

	if (file_output) {
		fprintf(output, "Benchmark score VkFFT: %d\n", (int)(benchmark_result));
#if(VKFFT_BACKEND==0)
		fprintf(output, "Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
#endif
	}
	printf("Benchmark score VkFFT: %d\n", (int)(benchmark_result));
#if(VKFFT_BACKEND==0)
	printf("Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
#endif
	return res;
}
uint32_t launchVkFFT(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output) {
	//Sample Vulkan project GPU initialization.
	uint32_t res = 0;
#if(VKFFT_BACKEND==0)
	//create instance - a connection between the application and the Vulkan library 
	res = createInstance(vkGPU, sample_id);
	if (res != 0) {
		printf("Instance creation failed, error code: %d\n", res);
		return res;
	}
	//set up the debugging messenger 
	res = setupDebugMessenger(vkGPU);
	if (res != 0) {
		printf("Debug messenger creation failed, error code: %d\n", res);
		return res;
	}
	//check if there are GPUs that support Vulkan and select one
	res = findPhysicalDevice(vkGPU);
	if (res != 0) {
		printf("Physical device not found, error code: %d\n", res);
		return res;
	}
	//create logical device representation
	res = createDevice(vkGPU, sample_id);
	if (res != 0) {
		printf("Device creation failed, error code: %d\n", res);
		return res;
	}
	//create fence for synchronization 
	res = createFence(vkGPU);
	if (res != 0) {
		printf("Fence creation failed, error code: %d\n", res);
		return res;
	}
	//create a place, command buffer memory is allocated from
	res = createCommandPool(vkGPU);
	if (res != 0) {
		printf("Fence creation failed, error code: %d\n", res);
		return res;
	}
	vkGetPhysicalDeviceProperties(vkGPU->physicalDevice, &vkGPU->physicalDeviceProperties);
	vkGetPhysicalDeviceMemoryProperties(vkGPU->physicalDevice, &vkGPU->physicalDeviceMemoryProperties);

	glslang_initialize_process();//compiler can be initialized before VkFFT
#elif(VKFFT_BACKEND==1)
	cuInit(0);
	cudaSetDevice(vkGPU->device_id);
	cuDeviceGet(&vkGPU->device, vkGPU->device_id);
	cuCtxCreate(&vkGPU->context, 0, vkGPU->device);
#elif(VKFFT_BACKEND==2)
	hipInit(0);
	hipSetDevice(vkGPU->device_id);
	hipDeviceGet(&vkGPU->device, vkGPU->device_id);
	hipCtxCreate(&vkGPU->context, 0, vkGPU->device);
#endif

	uint32_t isCompilerInitialized = 1;

	switch (sample_id) {
	case 0:
	{
		sample_0(vkGPU, sample_id, file_output, output, isCompilerInitialized);
		break;
	}
	case 1:
	{
		sample_1(vkGPU, sample_id, file_output, output, isCompilerInitialized);
		break;
	}
#if ((VKFFT_BACKEND==0)&&(VK_API_VERSION>10))
	case 2:
	{
		sample_2(vkGPU, sample_id, file_output, output, isCompilerInitialized);
		break;
	}
#endif
	case 3:
	{
		sample_3(vkGPU, sample_id, file_output, output, isCompilerInitialized);
		break;
	}
	case 4:
	{
		sample_4(vkGPU, sample_id, file_output, output, isCompilerInitialized);
		break;
	}
	case 5:
	{
		sample_5(vkGPU, sample_id, file_output, output, isCompilerInitialized);
		break;
	}
	case 6:
	{
		sample_6(vkGPU, sample_id, file_output, output, isCompilerInitialized);
		break;
	}
	case 7:
	{
		sample_7(vkGPU, sample_id, file_output, output, isCompilerInitialized);
		break;
	}
	case 8:
	{
		sample_8(vkGPU, sample_id, file_output, output, isCompilerInitialized);
		break;
	}
	case 9:
	{
		sample_9(vkGPU, sample_id, file_output, output, isCompilerInitialized);
		break;
	}
#if(VKFFT_BACKEND==0)
	case 10:
	{
		sample_10(vkGPU, sample_id, file_output, output, isCompilerInitialized);
		break;
	}
#endif
#ifdef USE_FFTW
	case 11:
	{
		sample_11(vkGPU, sample_id, file_output, output, isCompilerInitialized);
		break;
	}
	case 12:
	{
		sample_12(vkGPU, sample_id, file_output, output, isCompilerInitialized);
		break;
	}
#if ((VKFFT_BACKEND==0)&&(VK_API_VERSION>10))
	case 13:
	{
		sample_13(vkGPU, sample_id, file_output, output, isCompilerInitialized);
		break;
	}
#endif
	case 14:
	{
		sample_14(vkGPU, sample_id, file_output, output, isCompilerInitialized);
		break;
	}
	case 15:
	{
		sample_15(vkGPU, sample_id, file_output, output, isCompilerInitialized);
		break;
	}
#endif
	case 1000:
	{
		sample_1000(vkGPU, sample_id, file_output, output, isCompilerInitialized);
		break;
	}
	case 1001:
	{
		sample_1001(vkGPU, sample_id, file_output, output, isCompilerInitialized);
		break;
	}
	case 1003:
	{
		sample_1003(vkGPU, sample_id, file_output, output, isCompilerInitialized);
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
	cuCtxDestroy(vkGPU->context);
#elif(VKFFT_BACKEND==2)
	hipCtxDestroy(vkGPU->context);
#endif

	return 0;
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
	bool file_output = false;
	FILE* output = NULL;
	if (findFlag(argv, argv + argc, "-h"))
	{
		//print help
		printf("VkFFT v1.1.10 (15-03-2021). Author: Tolmachev Dmitrii\n");
		printf("	-h: print help\n");
#if (VKFFT_BACKEND==0)
		printf("	-devices: print the list of available GPU devices\n");
#endif
		printf("	-d X: select GPU device (default 0)\n");
		printf("	-o NAME: specify output file path\n");
		printf("	-vkfft X: launch VkFFT sample X (0-14):\n");
		printf("		0 - FFT + iFFT C2C benchmark 1D batched in single precision\n");
		printf("		1 - FFT + iFFT C2C benchmark 1D batched in double precision LUT\n");
#if ((VKFFT_BACKEND==0)&&(VK_API_VERSION>10))
		printf("		2 - FFT + iFFT C2C benchmark 1D batched in half precision\n");
#endif
		printf("		3 - FFT + iFFT C2C multidimensional benchmark in single precision\n");
		printf("		4 - FFT + iFFT C2C multidimensional benchmark in single precision, native zeropadding\n");
		printf("		5 - FFT + iFFT C2C benchmark 1D batched in single precision, no reshuffling\n");
		printf("		6 - FFT + iFFT R2C / C2R benchmark\n");
		printf("		7 - convolution example with identitiy kernel\n");
		printf("		8 - zeropadding convolution example with identitiy kernel\n");
		printf("		9 - batched convolution example with identitiy kernel\n");
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
		printf("		14 - VkFFT / FFTW C2C power 3 / 5 / 7 / 11 / 13 precision test in single precision\n");
		printf("		15 - VkFFT / cuFFT / FFTW R2C+C2R precision test in single precision\n");
#elif USE_rocFFT
		printf("		11 - VkFFT / rocFFT / FFTW C2C precision test in single precision\n");
		printf("		12 - VkFFT / rocFFT / FFTW C2C precision test in double precision\n");
#if ((VKFFT_BACKEND==0)&&(VK_API_VERSION>10))
		printf("		13 - VkFFT / FFTW C2C precision test in half precision\n");
#endif
		printf("		14 - VkFFT / FFTW C2C power 3 / 5 / 7 / 11 / 13 precision test in single precision\n");
		printf("		15 - VkFFT / rocFFT / FFTW R2C+C2R precision test in single precision\n");
#else
		printf("		11 - VkFFT / FFTW C2C precision test in single precision\n");
		printf("		12 - VkFFT / FFTW C2C precision test in double precision\n");
#if ((VKFFT_BACKEND==0)&&(VK_API_VERSION>10))
		printf("		13 - VkFFT / FFTW C2C precision test in half precision\n");
#endif
		printf("		14 - VkFFT / FFTW C2C power 3 / 5 / 7 / 11 / 13 precision test in single precision\n");
		printf("		15 - VkFFT / FFTW R2C+C2R precision test in single precision\n");
#endif
#endif
		printf("		1000 - FFT + iFFT C2C benchmark 1D batched in single precision: all supported systems from 2 to 4096\n");
		printf("		1001 - FFT + iFFT C2C benchmark 1D batched in double precision: all supported systems from 2 to 4096\n");
		printf("		1003 - FFT + iFFT C2C multidimensional benchmark in single precision: all supported rocbes from 2 to 512\n");
#ifdef USE_cuFFT
		printf("	-cufft X: launch cuFFT sample X (0-3):\n");
		printf("		0 - FFT + iFFT C2C benchmark 1D batched in single precision\n");
		printf("		1 - FFT + iFFT C2C benchmark 1D batched in double precision LUT\n");
		printf("		2 - FFT + iFFT C2C benchmark 1D batched in half precision\n");
		printf("		3 - FFT + iFFT C2C multidimensional benchmark in single precision\n");
		printf("		4 - FFT + iFFT R2C / C2R benchmark\n");
		printf("		1000 - FFT + iFFT C2C benchmark 1D batched in single precision: all supported systems from 2 to 4096\n");
		printf("		1001 - FFT + iFFT C2C benchmark 1D batched in double precision: all supported systems from 2 to 4096\n");
		printf("		1003 - FFT + iFFT C2C multidimensional benchmark in single precision: all supported rocbes from 2 to 512\n");
		printf("	-test: (or no -vkfft and -cufft keys) run vkfft benchmarks 0-6 and cufft benchmarks 0-4\n");
#elif USE_rocFFT
		printf("	-rocfft X: launch rocFFT sample X (0-3):\n");
		printf("		0 - FFT + iFFT C2C benchmark 1D batched in single precision\n");
		printf("		1 - FFT + iFFT C2C benchmark 1D batched in double precision LUT\n");
		printf("		3 - FFT + iFFT C2C multidimensional benchmark in single precision\n");
		printf("		4 - FFT + iFFT R2C / C2R benchmark\n");
		printf("		1000 - FFT + iFFT C2C benchmark 1D batched in single precision: all supported systems from 2 to 4096\n");
		printf("		1001 - FFT + iFFT C2C benchmark 1D batched in double precision: all supported systems from 2 to 4096\n");
		printf("		1003 - FFT + iFFT C2C multidimensional benchmark in single precision: all supported rocbes from 2 to 512\n");
		printf("	-test: (or no -vkfft and -rocfft keys) run vkfft benchmarks 0-6 and rocfft benchmarks 0-4\n");
#else
		printf("	-test: run vkfft benchmarks 0-6\n");
		printf("	-cufft command is disabled\n");
		printf("	-rocfft command is disabled\n");
#endif
		return 0;
	}
#if(VKFFT_BACKEND==0)
	if (findFlag(argv, argv + argc, "-devices"))
	{
		//print device list
		uint32_t res = devices_list();
		return res;
	}
#endif
	if (findFlag(argv, argv + argc, "-d"))
	{
		//select device_id
		char* value = getFlagValue(argv, argv + argc, "-d");
		if (value != 0) {
			sscanf(value, "%d", &vkGPU.device_id);
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
	if (findFlag(argv, argv + argc, "-vkfft"))
	{
		//select sample_id
		char* value = getFlagValue(argv, argv + argc, "-vkfft");
		if (value != 0) {
			uint32_t sample_id = 0;
			sscanf(value, "%d", &sample_id);
			uint32_t res = launchVkFFT(&vkGPU, sample_id, file_output, output);
			if (res != 0) return res;
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
			uint32_t sample_id = 0;
			sscanf(value, "%d", &sample_id);
			switch (sample_id) {
			case 0:
				launch_benchmark_cuFFT_single(file_output, output);
				break;
			case 1:
				launch_benchmark_cuFFT_double(file_output, output);
				break;
			case 2:
				launch_benchmark_cuFFT_half(file_output, output);
				break;
			case 3:
				launch_benchmark_cuFFT_single_3d(file_output, output);
				break;
			case 4:
				launch_benchmark_cuFFT_single_r2c(file_output, output);
				break;
			case 1000:
				launch_benchmark_cuFFT_single_2_4096(file_output, output);
				break;
			case 1001:
				launch_benchmark_cuFFT_double_2_4096(file_output, output);
				break;
			case 1003:
				launch_benchmark_cuFFT_single_3d_2_512(file_output, output);
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
			uint32_t sample_id = 0;
			sscanf(value, "%d", &sample_id);
			switch (sample_id) {
			case 0:
				launch_benchmark_rocFFT_single(file_output, output);
				break;
			case 1:
				launch_benchmark_rocFFT_double(file_output, output);
				break;
			case 3:
				launch_benchmark_rocFFT_single_3d(file_output, output);
				break;
			case 4:
				launch_benchmark_rocFFT_single_r2c(file_output, output);
				break;
			case 1000:
				launch_benchmark_rocFFT_single_2_4096(file_output, output);
				break;
			case 1001:
				launch_benchmark_rocFFT_double_2_4096(file_output, output);
				break;
			case 1003:
				launch_benchmark_rocFFT_single_3d_2_512(file_output, output);
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
		for (uint32_t i = 0; i < 7; i++) {
#if((VKFFT_BACKEND>0) || (VK_API_VERSION == 10))
			if (i == 2) i++;
#endif
			uint32_t res = launchVkFFT(&vkGPU, i, file_output, output);
			if (res != 0) return res;
		}
#ifdef USE_cuFFT
		launch_benchmark_cuFFT_single(file_output, output);
		launch_benchmark_cuFFT_double(file_output, output);
		launch_benchmark_cuFFT_half(file_output, output);
		launch_benchmark_cuFFT_single_3d(file_output, output);
		launch_benchmark_cuFFT_single_r2c(file_output, output);
#elif USE_rocFFT
		launch_benchmark_rocFFT_single(file_output, output);
		launch_benchmark_rocFFT_double(file_output, output);
		launch_benchmark_rocFFT_single_3d(file_output, output);
		launch_benchmark_rocFFT_single_r2c(file_output, output);
#endif
	}
	return 0;
}
