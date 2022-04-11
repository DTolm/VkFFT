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
#if(VKFFT_BACKEND==0)

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
	if (vkGPU->enableValidationLayers == 0) return VK_SUCCESS;

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
	if (!availableLayers) return VK_INCOMPLETE;
	vkEnumerateInstanceLayerProperties(&layerCount, availableLayers);
	if (availableLayers) {
		for (uint64_t i = 0; i < layerCount; i++) {
			if (strcmp("VK_LAYER_KHRONOS_validation", availableLayers[i].layerName) == 0) {
				free(availableLayers);
				return VK_SUCCESS;
			}
		}
		free(availableLayers);
	}
	else {
		return VK_INCOMPLETE;
	}
	return VK_ERROR_LAYER_NOT_PRESENT;
}

std::vector<const char*> getRequiredExtensions(VkGPU* vkGPU, uint64_t sample_id) {
	std::vector<const char*> extensions;

	if (vkGPU->enableValidationLayers) {
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}
	switch (sample_id) {
#if (VK_API_VERSION>10)
	case 2: case 102:
		extensions.push_back("VK_KHR_get_physical_device_properties2");
		break;
#endif
	default:
		break;
	}


	return extensions;
}

VkResult createInstance(VkGPU* vkGPU, uint64_t sample_id) {
	//create instance - a connection between the application and the Vulkan library 
	VkResult res = VK_SUCCESS;
	//check if validation layers are supported
	if (vkGPU->enableValidationLayers == 1) {
		res = checkValidationLayerSupport();
		if (res != VK_SUCCESS) return res;
	}

	VkApplicationInfo applicationInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
	applicationInfo.pApplicationName = "VkFFT";
	applicationInfo.applicationVersion = (uint32_t)VkFFTGetVersion();
	applicationInfo.pEngineName = "VkFFT";
	applicationInfo.engineVersion = 1;
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

	auto extensions = getRequiredExtensions(vkGPU, sample_id);
	createInfo.enabledExtensionCount = (uint32_t)(extensions.size());
	createInfo.ppEnabledExtensionNames = extensions.data();

	VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
	if (vkGPU->enableValidationLayers) {
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
	if (!devices) return VK_INCOMPLETE;
	res = vkEnumeratePhysicalDevices(vkGPU->instance, &deviceCount, devices);
	if (res != VK_SUCCESS) return res;
	if (devices) {
		vkGPU->physicalDevice = devices[vkGPU->device_id];
		free(devices);
		return VK_SUCCESS;
	}
	else
		return VK_INCOMPLETE;
}
VkResult getComputeQueueFamilyIndex(VkGPU* vkGPU) {
	//find a queue family for a selected GPU, select the first available for use
	uint32_t queueFamilyCount;
	vkGetPhysicalDeviceQueueFamilyProperties(vkGPU->physicalDevice, &queueFamilyCount, NULL);

	VkQueueFamilyProperties* queueFamilies = (VkQueueFamilyProperties*)malloc(sizeof(VkQueueFamilyProperties) * queueFamilyCount);
	if (!queueFamilies) return VK_INCOMPLETE;
	if (queueFamilies) {
		vkGetPhysicalDeviceQueueFamilyProperties(vkGPU->physicalDevice, &queueFamilyCount, queueFamilies);
		uint64_t i = 0;
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
	else
		return VK_INCOMPLETE;
}

VkResult createDevice(VkGPU* vkGPU, uint64_t sample_id) {
	//create logical device representation
	VkResult res = VK_SUCCESS;
	VkDeviceQueueCreateInfo queueCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
	res = getComputeQueueFamilyIndex(vkGPU);
	if (res != VK_SUCCESS) return res;
	queueCreateInfo.queueFamilyIndex = (uint32_t)vkGPU->queueFamilyIndex;
	queueCreateInfo.queueCount = 1;
	float queuePriorities = 1.0;
	queueCreateInfo.pQueuePriorities = &queuePriorities;
	VkDeviceCreateInfo deviceCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
	VkPhysicalDeviceFeatures deviceFeatures = {};
	switch (sample_id) {
	case 1: case 12: case 17: case 18: case 101: case 201: case 1001: {
		deviceFeatures.shaderFloat64 = true;
		deviceCreateInfo.enabledExtensionCount = (uint32_t)vkGPU->enabledDeviceExtensions.size();
		deviceCreateInfo.ppEnabledExtensionNames = vkGPU->enabledDeviceExtensions.data();
		deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
		deviceCreateInfo.queueCreateInfoCount = 1;
		deviceCreateInfo.pEnabledFeatures = &deviceFeatures;
		res = vkCreateDevice(vkGPU->physicalDevice, &deviceCreateInfo, NULL, &vkGPU->device);
		if (res != VK_SUCCESS) return res;
		vkGetDeviceQueue(vkGPU->device, (uint32_t)vkGPU->queueFamilyIndex, 0, &vkGPU->queue);
		break;
	}
#if (VK_API_VERSION>10)
	case 2: case 102: {
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
		deviceCreateInfo.enabledExtensionCount = (uint32_t)vkGPU->enabledDeviceExtensions.size();
		deviceCreateInfo.ppEnabledExtensionNames = vkGPU->enabledDeviceExtensions.data();
		deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
		deviceCreateInfo.queueCreateInfoCount = 1;
		deviceCreateInfo.pEnabledFeatures = NULL;
		res = vkCreateDevice(vkGPU->physicalDevice, &deviceCreateInfo, NULL, &vkGPU->device);
		if (res != VK_SUCCESS) return res;
		vkGetDeviceQueue(vkGPU->device, (uint32_t)vkGPU->queueFamilyIndex, 0, &vkGPU->queue);
		break;
	}
#endif
	default: {
		deviceCreateInfo.enabledExtensionCount = (uint32_t)vkGPU->enabledDeviceExtensions.size();
		deviceCreateInfo.ppEnabledExtensionNames = vkGPU->enabledDeviceExtensions.data();
		deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
		deviceCreateInfo.queueCreateInfoCount = 1;
		deviceCreateInfo.pEnabledFeatures = NULL;
		deviceCreateInfo.pEnabledFeatures = &deviceFeatures;
		res = vkCreateDevice(vkGPU->physicalDevice, &deviceCreateInfo, NULL, &vkGPU->device);
		if (res != VK_SUCCESS) return res;
		vkGetDeviceQueue(vkGPU->device, (uint32_t)vkGPU->queueFamilyIndex, 0, &vkGPU->queue);
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
	commandPoolCreateInfo.queueFamilyIndex = (uint32_t)vkGPU->queueFamilyIndex;
	res = vkCreateCommandPool(vkGPU->device, &commandPoolCreateInfo, NULL, &vkGPU->commandPool);
	return res;
}

VkFFTResult findMemoryType(VkGPU* vkGPU, uint64_t memoryTypeBits, uint64_t memorySize, VkMemoryPropertyFlags properties, uint32_t* memoryTypeIndex) {
	VkPhysicalDeviceMemoryProperties memoryProperties = { 0 };

	vkGetPhysicalDeviceMemoryProperties(vkGPU->physicalDevice, &memoryProperties);

	for (uint64_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
		if ((memoryTypeBits & ((uint64_t)1 << i)) && ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) && (memoryProperties.memoryHeaps[memoryProperties.memoryTypes[i].heapIndex].size >= memorySize))
		{
			memoryTypeIndex[0] = (uint32_t)i;
			return VKFFT_SUCCESS;
		}
	}
	return VKFFT_ERROR_FAILED_TO_FIND_MEMORY;
}

VkFFTResult allocateBuffer(VkGPU* vkGPU, VkBuffer* buffer, VkDeviceMemory* deviceMemory, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags propertyFlags, uint64_t size) {
	//allocate the buffer used by the GPU with specified properties
	VkFFTResult resFFT = VKFFT_SUCCESS;
	VkResult res = VK_SUCCESS;
	uint32_t queueFamilyIndices;
	VkBufferCreateInfo bufferCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	bufferCreateInfo.queueFamilyIndexCount = 1;
	bufferCreateInfo.pQueueFamilyIndices = &queueFamilyIndices;
	bufferCreateInfo.size = size;
	bufferCreateInfo.usage = usageFlags;
	res = vkCreateBuffer(vkGPU->device, &bufferCreateInfo, NULL, buffer);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_BUFFER;
	VkMemoryRequirements memoryRequirements = { 0 };
	vkGetBufferMemoryRequirements(vkGPU->device, buffer[0], &memoryRequirements);
	VkMemoryAllocateInfo memoryAllocateInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
	memoryAllocateInfo.allocationSize = memoryRequirements.size;
	resFFT = findMemoryType(vkGPU, memoryRequirements.memoryTypeBits, memoryRequirements.size, propertyFlags, &memoryAllocateInfo.memoryTypeIndex);
	if (resFFT != VKFFT_SUCCESS) return resFFT;
	res = vkAllocateMemory(vkGPU->device, &memoryAllocateInfo, NULL, deviceMemory);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE_MEMORY;
	res = vkBindBufferMemory(vkGPU->device, buffer[0], deviceMemory[0], 0);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_BIND_BUFFER_MEMORY;
	return resFFT;
}
VkFFTResult transferDataFromCPU(VkGPU* vkGPU, void* arr, VkBuffer* buffer, uint64_t bufferSize) {
	//a function that transfers data from the CPU to the GPU using staging buffer, because the GPU memory is not host-coherent
	VkFFTResult resFFT = VKFFT_SUCCESS;
	VkResult res = VK_SUCCESS;
	uint64_t stagingBufferSize = bufferSize;
	VkBuffer stagingBuffer = { 0 };
	VkDeviceMemory stagingBufferMemory = { 0 };
	resFFT = allocateBuffer(vkGPU, &stagingBuffer, &stagingBufferMemory, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBufferSize);
	if (resFFT != VKFFT_SUCCESS) return resFFT;
	void* data;
	res = vkMapMemory(vkGPU->device, stagingBufferMemory, 0, stagingBufferSize, 0, &data);
	if (resFFT != VKFFT_SUCCESS) return resFFT;
	memcpy(data, arr, stagingBufferSize);
	vkUnmapMemory(vkGPU->device, stagingBufferMemory);
	VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	commandBufferAllocateInfo.commandPool = vkGPU->commandPool;
	commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferAllocateInfo.commandBufferCount = 1;
	VkCommandBuffer commandBuffer = { 0 };
	res = vkAllocateCommandBuffers(vkGPU->device, &commandBufferAllocateInfo, &commandBuffer);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE_COMMAND_BUFFERS;
	VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	res = vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_BEGIN_COMMAND_BUFFER;
	VkBufferCopy copyRegion = { 0 };
	copyRegion.srcOffset = 0;
	copyRegion.dstOffset = 0;
	copyRegion.size = stagingBufferSize;
	vkCmdCopyBuffer(commandBuffer, stagingBuffer, buffer[0], 1, &copyRegion);
	res = vkEndCommandBuffer(commandBuffer);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_END_COMMAND_BUFFER;
	VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;
	res = vkQueueSubmit(vkGPU->queue, 1, &submitInfo, vkGPU->fence);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_SUBMIT_QUEUE;
	res = vkWaitForFences(vkGPU->device, 1, &vkGPU->fence, VK_TRUE, 100000000000);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_WAIT_FOR_FENCES;
	res = vkResetFences(vkGPU->device, 1, &vkGPU->fence);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_RESET_FENCES;
	vkFreeCommandBuffers(vkGPU->device, vkGPU->commandPool, 1, &commandBuffer);
	vkDestroyBuffer(vkGPU->device, stagingBuffer, NULL);
	vkFreeMemory(vkGPU->device, stagingBufferMemory, NULL);
	return resFFT;
}
VkFFTResult transferDataToCPU(VkGPU* vkGPU, void* arr, VkBuffer* buffer, uint64_t bufferSize) {
	//a function that transfers data from the GPU to the CPU using staging buffer, because the GPU memory is not host-coherent
	VkFFTResult resFFT = VKFFT_SUCCESS;
	VkResult res = VK_SUCCESS;
	uint64_t stagingBufferSize = bufferSize;
	VkBuffer stagingBuffer = { 0 };
	VkDeviceMemory stagingBufferMemory = { 0 };
	resFFT = allocateBuffer(vkGPU, &stagingBuffer, &stagingBufferMemory, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBufferSize);
	if (resFFT != VKFFT_SUCCESS) return resFFT;
	VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	commandBufferAllocateInfo.commandPool = vkGPU->commandPool;
	commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferAllocateInfo.commandBufferCount = 1;
	VkCommandBuffer commandBuffer = { 0 };
	res = vkAllocateCommandBuffers(vkGPU->device, &commandBufferAllocateInfo, &commandBuffer);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE_COMMAND_BUFFERS;
	VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	res = vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_BEGIN_COMMAND_BUFFER;
	VkBufferCopy copyRegion = { 0 };
	copyRegion.srcOffset = 0;
	copyRegion.dstOffset = 0;
	copyRegion.size = stagingBufferSize;
	vkCmdCopyBuffer(commandBuffer, buffer[0], stagingBuffer, 1, &copyRegion);
	res = vkEndCommandBuffer(commandBuffer);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_END_COMMAND_BUFFER;
	VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;
	res = vkQueueSubmit(vkGPU->queue, 1, &submitInfo, vkGPU->fence);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_SUBMIT_QUEUE;
	res = vkWaitForFences(vkGPU->device, 1, &vkGPU->fence, VK_TRUE, 100000000000);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_WAIT_FOR_FENCES;
	res = vkResetFences(vkGPU->device, 1, &vkGPU->fence);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_RESET_FENCES;
	vkFreeCommandBuffers(vkGPU->device, vkGPU->commandPool, 1, &commandBuffer);
	void* data;
	res = vkMapMemory(vkGPU->device, stagingBufferMemory, 0, stagingBufferSize, 0, &data);
	if (resFFT != VKFFT_SUCCESS) return resFFT;
	memcpy(arr, data, stagingBufferSize);
	vkUnmapMemory(vkGPU->device, stagingBufferMemory);
	vkDestroyBuffer(vkGPU->device, stagingBuffer, NULL);
	vkFreeMemory(vkGPU->device, stagingBufferMemory, NULL);
	return resFFT;
}
#endif
VkFFTResult devices_list() {
	//this function creates an instance and prints the list of available devices
#if(VKFFT_BACKEND==0)
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
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_INSTANCE;

	uint32_t deviceCount;
	res = vkEnumeratePhysicalDevices(local_instance, &deviceCount, NULL);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_ENUMERATE_DEVICES;

	VkPhysicalDevice* devices = (VkPhysicalDevice*)malloc(sizeof(VkPhysicalDevice) * deviceCount);
	if (!devices) return VKFFT_ERROR_MALLOC_FAILED;
	if (devices) {
		res = vkEnumeratePhysicalDevices(local_instance, &deviceCount, devices);
		if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_ENUMERATE_DEVICES;
		for (uint64_t i = 0; i < deviceCount; i++) {
			VkPhysicalDeviceProperties device_properties;
			vkGetPhysicalDeviceProperties(devices[i], &device_properties);
			printf("Device id: %" PRIu64 " name: %s API:%d.%d.%d\n", i, device_properties.deviceName, (device_properties.apiVersion >> 22), ((device_properties.apiVersion >> 12) & 0x3ff), (device_properties.apiVersion & 0xfff));
		}
		free(devices);
	}
	else 
		return VKFFT_ERROR_FAILED_TO_ENUMERATE_DEVICES;
	vkDestroyInstance(local_instance, NULL);
#elif(VKFFT_BACKEND==1)
	CUresult res = CUDA_SUCCESS;
	res = cuInit(0);
	if (res != CUDA_SUCCESS) return VKFFT_ERROR_FAILED_TO_INITIALIZE;
	int numDevices;
	res = cuDeviceGetCount(&numDevices);
	if (res != CUDA_SUCCESS) return VKFFT_ERROR_FAILED_TO_SET_DEVICE_ID;
	for (uint64_t i = 0; i < numDevices; i++) {
		char deviceName[256];
		CUdevice device = {};
		res = cuDeviceGet(&device, (int)i);
		if (res != CUDA_SUCCESS) return VKFFT_ERROR_FAILED_TO_GET_DEVICE;
		res = cuDeviceGetName(deviceName, 256, device);
		if (res != CUDA_SUCCESS) return VKFFT_ERROR_FAILED_TO_GET_DEVICE;
		printf("Device id: %" PRIu64 " name: %s\n", i, deviceName);
	}
#elif(VKFFT_BACKEND==2)
	hipError_t res = hipSuccess;
	res = hipInit(0);
	if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_INITIALIZE;
	int numDevices;
	res = hipGetDeviceCount(&numDevices);
	if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_SET_DEVICE_ID;
	for (uint64_t i = 0; i < numDevices; i++) {
		char deviceName[256];
		hipDevice_t device = {};
		res = hipDeviceGet(&device, i);
		if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_GET_DEVICE;
		res = hipDeviceGetName(deviceName, 256, device);
		if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_GET_DEVICE;
		printf("Device id: %" PRIu64 " name: %s\n", i, deviceName);
	}
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
			char deviceName[256];
			char apiVersion[256];
			res = clGetDeviceInfo(deviceList[i], CL_DEVICE_NAME, 256 * sizeof(char), deviceName, 0);
			if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_ENUMERATE_DEVICES;
			res = clGetDeviceInfo(deviceList[i], CL_DEVICE_VERSION, 256 * sizeof(char), apiVersion, 0);
			if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_ENUMERATE_DEVICES;
			printf("Platform id: %" PRIu64 " Device id: %" PRIu64 " name: %s API:%s\n", j, k, deviceName, apiVersion);
			k++;
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
			ze_device_properties_t device_properties;
			res = zeDeviceGetProperties(deviceList[i], &device_properties);
			if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_ENUMERATE_DEVICES;
			printf("Driver id: %" PRIu64 " Device id: %" PRIu64 " name: %s\n", j, k, device_properties.name);
			k++;
		}

		free(deviceList);
	}
	free(drivers);
#endif
	return VKFFT_SUCCESS;
}
VkFFTResult performVulkanFFT(VkGPU* vkGPU, VkFFTApplication* app, VkFFTLaunchParams* launchParams, int inverse, uint64_t num_iter) {
	VkFFTResult resFFT = VKFFT_SUCCESS;
#if(VKFFT_BACKEND==0)
	VkResult res = VK_SUCCESS;
	VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	commandBufferAllocateInfo.commandPool = vkGPU->commandPool;
	commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferAllocateInfo.commandBufferCount = 1;
	VkCommandBuffer commandBuffer = {};
	res = vkAllocateCommandBuffers(vkGPU->device, &commandBufferAllocateInfo, &commandBuffer);
	if (res != 0) return VKFFT_ERROR_FAILED_TO_ALLOCATE_COMMAND_BUFFERS;
	VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	res = vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
	if (res != 0) return VKFFT_ERROR_FAILED_TO_BEGIN_COMMAND_BUFFER;
	launchParams->commandBuffer = &commandBuffer;
	//Record commands num_iter times. Allows to perform multiple convolutions/transforms in one submit.
	for (uint64_t i = 0; i < num_iter; i++) {
		resFFT = VkFFTAppend(app, inverse, launchParams);
		if (resFFT != VKFFT_SUCCESS) return resFFT;
	}
	res = vkEndCommandBuffer(commandBuffer);
	if (res != 0) return VKFFT_ERROR_FAILED_TO_END_COMMAND_BUFFER;
	VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;
	std::chrono::steady_clock::time_point timeSubmit = std::chrono::steady_clock::now();
	res = vkQueueSubmit(vkGPU->queue, 1, &submitInfo, vkGPU->fence);
	if (res != 0) return VKFFT_ERROR_FAILED_TO_SUBMIT_QUEUE;
	res = vkWaitForFences(vkGPU->device, 1, &vkGPU->fence, VK_TRUE, 100000000000);
	if (res != 0) return VKFFT_ERROR_FAILED_TO_WAIT_FOR_FENCES;
	std::chrono::steady_clock::time_point timeEnd = std::chrono::steady_clock::now();
	double totTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001;
	//printf("Pure submit execution time per num_iter: %.3f ms\n", totTime / num_iter);
	res = vkResetFences(vkGPU->device, 1, &vkGPU->fence);
	if (res != 0) return VKFFT_ERROR_FAILED_TO_RESET_FENCES;
	vkFreeCommandBuffers(vkGPU->device, vkGPU->commandPool, 1, &commandBuffer);
#elif(VKFFT_BACKEND==1)
	cudaError_t res = cudaSuccess;
	std::chrono::steady_clock::time_point timeSubmit = std::chrono::steady_clock::now();
	for (uint64_t i = 0; i < num_iter; i++) {
		resFFT = VkFFTAppend(app, inverse, launchParams);
		if (resFFT != VKFFT_SUCCESS) return resFFT;
	}
	res = cudaDeviceSynchronize();
	if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
	std::chrono::steady_clock::time_point timeEnd = std::chrono::steady_clock::now();
	double totTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001;
#elif(VKFFT_BACKEND==2)
	hipError_t res = hipSuccess;
	std::chrono::steady_clock::time_point timeSubmit = std::chrono::steady_clock::now();
	for (uint64_t i = 0; i < num_iter; i++) {
		resFFT = VkFFTAppend(app, inverse, launchParams);
		if (resFFT != VKFFT_SUCCESS) return resFFT;
	}
	res = hipDeviceSynchronize();
	if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
	std::chrono::steady_clock::time_point timeEnd = std::chrono::steady_clock::now();
	double totTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001;
#elif(VKFFT_BACKEND==3)
	cl_int res = CL_SUCCESS;
	launchParams->commandQueue = &vkGPU->commandQueue;
	std::chrono::steady_clock::time_point timeSubmit = std::chrono::steady_clock::now();
	for (uint64_t i = 0; i < num_iter; i++) {
		resFFT = VkFFTAppend(app, inverse, launchParams);
		if (resFFT != VKFFT_SUCCESS) return resFFT;
	}
	res = clFinish(vkGPU->commandQueue);
	if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
	std::chrono::steady_clock::time_point timeEnd = std::chrono::steady_clock::now();
	double totTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001;
#elif(VKFFT_BACKEND==4)
	ze_result_t res = ZE_RESULT_SUCCESS;
	ze_command_list_desc_t commandListDescription = {};
	commandListDescription.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
	ze_command_list_handle_t commandList = {};
	res = zeCommandListCreate(vkGPU->context, vkGPU->device, &commandListDescription, &commandList);
	if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
	
	launchParams->commandList = &commandList;
	//Record commands num_iter times. Allows to perform multiple convolutions/transforms in one submit.
	for (uint64_t i = 0; i < num_iter; i++) {
		resFFT = VkFFTAppend(app, inverse, launchParams);
		if (resFFT != VKFFT_SUCCESS) return resFFT;
	}
	res = zeCommandListClose(commandList);
	if (res != 0) return VKFFT_ERROR_FAILED_TO_END_COMMAND_BUFFER;
	
	std::chrono::steady_clock::time_point timeSubmit = std::chrono::steady_clock::now();
	res = zeCommandQueueExecuteCommandLists(vkGPU->commandQueue, 1, &commandList, 0);
	if (res != 0) return VKFFT_ERROR_FAILED_TO_SUBMIT_QUEUE;
	res = zeCommandQueueSynchronize(vkGPU->commandQueue, UINT32_MAX);
	if (res != 0) return VKFFT_ERROR_FAILED_TO_WAIT_FOR_FENCES;

	std::chrono::steady_clock::time_point timeEnd = std::chrono::steady_clock::now();
	double totTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001;
	//printf("Pure submit execution time per num_iter: %.3f ms\n", totTime / num_iter);
	res = zeCommandListDestroy(commandList);
	if (res != 0) return VKFFT_ERROR_FAILED_TO_DESTROY_COMMAND_LIST;
#endif
	return resFFT;
}
VkFFTResult performVulkanFFTiFFT(VkGPU* vkGPU, VkFFTApplication* app, VkFFTLaunchParams* launchParams, uint64_t num_iter, double* time_result) {
	VkFFTResult resFFT = VKFFT_SUCCESS;
#if(VKFFT_BACKEND==0)
	VkResult res = VK_SUCCESS;
	VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	commandBufferAllocateInfo.commandPool = vkGPU->commandPool;
	commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferAllocateInfo.commandBufferCount = 1;
	VkCommandBuffer commandBuffer = {};
	res = vkAllocateCommandBuffers(vkGPU->device, &commandBufferAllocateInfo, &commandBuffer);
	if (res != 0) return VKFFT_ERROR_FAILED_TO_ALLOCATE_COMMAND_BUFFERS;
	VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	res = vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
	if (res != 0) return VKFFT_ERROR_FAILED_TO_BEGIN_COMMAND_BUFFER;
	launchParams->commandBuffer = &commandBuffer;
	for (uint64_t i = 0; i < num_iter; i++) {
		resFFT = VkFFTAppend(app, -1, launchParams);
		if (resFFT != VKFFT_SUCCESS) return resFFT;
		resFFT = VkFFTAppend(app, 1, launchParams);
		if (resFFT != VKFFT_SUCCESS) return resFFT;
	}
	res = vkEndCommandBuffer(commandBuffer);
	if (res != 0) return VKFFT_ERROR_FAILED_TO_END_COMMAND_BUFFER;
	VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;
	std::chrono::steady_clock::time_point timeSubmit = std::chrono::steady_clock::now();
	res = vkQueueSubmit(vkGPU->queue, 1, &submitInfo, vkGPU->fence);
	if (res != 0) return VKFFT_ERROR_FAILED_TO_SUBMIT_QUEUE;
	res = vkWaitForFences(vkGPU->device, 1, &vkGPU->fence, VK_TRUE, 100000000000);
	if (res != 0) return VKFFT_ERROR_FAILED_TO_WAIT_FOR_FENCES;
	std::chrono::steady_clock::time_point timeEnd = std::chrono::steady_clock::now();
	double totTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001;
	time_result[0] = totTime / num_iter;
	res = vkResetFences(vkGPU->device, 1, &vkGPU->fence);
	if (res != 0) return VKFFT_ERROR_FAILED_TO_RESET_FENCES;
	vkFreeCommandBuffers(vkGPU->device, vkGPU->commandPool, 1, &commandBuffer);
#elif(VKFFT_BACKEND==1)
	cudaError_t res = cudaSuccess;
	std::chrono::steady_clock::time_point timeSubmit = std::chrono::steady_clock::now();
	for (uint64_t i = 0; i < num_iter; i++) {
		resFFT = VkFFTAppend(app, -1, launchParams);
		if (resFFT != VKFFT_SUCCESS) return resFFT;
		resFFT = VkFFTAppend(app, 1, launchParams);
		if (resFFT != VKFFT_SUCCESS) return resFFT;
	}
	res = cudaDeviceSynchronize();
	if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
	std::chrono::steady_clock::time_point timeEnd = std::chrono::steady_clock::now();
	double totTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001;
	time_result[0] = totTime / num_iter;
#elif(VKFFT_BACKEND==2)
	hipError_t res = hipSuccess;
	std::chrono::steady_clock::time_point timeSubmit = std::chrono::steady_clock::now();
	for (uint64_t i = 0; i < num_iter; i++) {
		resFFT = VkFFTAppend(app, -1, launchParams);
		if (resFFT != VKFFT_SUCCESS) return resFFT;
		resFFT = VkFFTAppend(app, 1, launchParams);
		if (resFFT != VKFFT_SUCCESS) return resFFT;
	}
	res = hipDeviceSynchronize();
	if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
	std::chrono::steady_clock::time_point timeEnd = std::chrono::steady_clock::now();
	double totTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001;
	time_result[0] = totTime / num_iter;
#elif(VKFFT_BACKEND==3)
	cl_int res = CL_SUCCESS;
	launchParams->commandQueue = &vkGPU->commandQueue;
	std::chrono::steady_clock::time_point timeSubmit = std::chrono::steady_clock::now();
	for (uint64_t i = 0; i < num_iter; i++) {
		resFFT = VkFFTAppend(app, -1, launchParams);
		if (resFFT != VKFFT_SUCCESS) return resFFT;
		resFFT = VkFFTAppend(app, 1, launchParams);
		if (resFFT != VKFFT_SUCCESS) return resFFT;
	}
	res = clFinish(vkGPU->commandQueue);
	if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
	std::chrono::steady_clock::time_point timeEnd = std::chrono::steady_clock::now();
	double totTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001;
	time_result[0] = totTime / num_iter;
#elif(VKFFT_BACKEND==4)
	ze_result_t res = ZE_RESULT_SUCCESS;
	ze_command_list_desc_t commandListDescription = {};
	commandListDescription.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
	ze_command_list_handle_t commandList = {};
	res = zeCommandListCreate(vkGPU->context, vkGPU->device, &commandListDescription, &commandList);
	if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;

	launchParams->commandList = &commandList;
	for (uint64_t i = 0; i < num_iter; i++) {
		resFFT = VkFFTAppend(app, -1, launchParams);
		if (resFFT != VKFFT_SUCCESS) return resFFT;
		resFFT = VkFFTAppend(app, 1, launchParams);
		if (resFFT != VKFFT_SUCCESS) return resFFT;
	}
	res = zeCommandListClose(commandList);
	if (res != 0) return VKFFT_ERROR_FAILED_TO_END_COMMAND_BUFFER;

	std::chrono::steady_clock::time_point timeSubmit = std::chrono::steady_clock::now();
	res = zeCommandQueueExecuteCommandLists(vkGPU->commandQueue, 1, &commandList, 0);
	if (res != 0) return VKFFT_ERROR_FAILED_TO_SUBMIT_QUEUE;
	res = zeCommandQueueSynchronize(vkGPU->commandQueue, UINT32_MAX);
	if (res != 0) return VKFFT_ERROR_FAILED_TO_WAIT_FOR_FENCES;

	std::chrono::steady_clock::time_point timeEnd = std::chrono::steady_clock::now();
	double totTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001;
	time_result[0] = totTime / num_iter;
	res = zeCommandListDestroy(commandList);
	if (res != 0) return VKFFT_ERROR_FAILED_TO_DESTROY_COMMAND_LIST;
#endif
	return resFFT;
}