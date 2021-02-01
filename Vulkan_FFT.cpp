#include <vector>
#include <memory>
#include <string.h>
#include <chrono>
#include <thread>
#include <iostream>
#include <algorithm>
#include "vkFFT.h"
#include "vulkan/vulkan.h"
#include "half.hpp"
#ifdef USE_cuFFT
#include "benchmark_cuFFT.h"
#include "benchmark_cuFFT_double.h"
#include "benchmark_cuFFT_half.h"
#include "benchmark_cuFFT_3d.h"
#include "precision_cuFFT.h"
#include "precision_cuFFT_double.h"
#include "precision_cuFFT_half.h"
#endif  
#ifdef USE_FFTW
#include "fftw3.h"
#endif
#include "glslang_c_interface.h"
using half_float::half;

typedef half half2[2];

const bool enableValidationLayers = false;

typedef struct {
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
	uint32_t device_id;//an id of a device, reported by Vulkan device list
	std::vector<const char*> enabledDeviceExtensions;
} VkGPU;//an example structure containing Vulkan primitives

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

VkResult allocateFFTBuffer(VkGPU* vkGPU, VkBuffer* buffer, VkDeviceMemory* deviceMemory, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags propertyFlags, VkDeviceSize size) {
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
VkResult transferDataFromCPU(VkGPU* vkGPU, void* arr, VkBuffer* buffer, VkDeviceSize bufferSize) {
	//a function that transfers data from the CPU to the GPU using staging buffer, because the GPU memory is not host-coherent
	VkResult res = VK_SUCCESS;
	VkDeviceSize stagingBufferSize = bufferSize;
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
VkResult transferDataToCPU(VkGPU* vkGPU, void* arr, VkBuffer* buffer, VkDeviceSize bufferSize) {
	//a function that transfers data from the GPU to the CPU using staging buffer, because the GPU memory is not host-coherent
	VkResult res = VK_SUCCESS;
	VkDeviceSize stagingBufferSize = bufferSize;
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

void performVulkanFFT(VkGPU* vkGPU, VkFFTApplication* app, uint32_t batch) {
	VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	commandBufferAllocateInfo.commandPool = vkGPU->commandPool;
	commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferAllocateInfo.commandBufferCount = 1;
	VkCommandBuffer commandBuffer = {};
	vkAllocateCommandBuffers(vkGPU->device, &commandBufferAllocateInfo, &commandBuffer);
	VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
	//Record commands batch times. Allows to perform multiple convolutions/transforms in one submit.
	for (uint32_t i = 0; i < batch; i++) {
		VkFFTAppend(app, commandBuffer);
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
	//printf("Pure submit execution time per batch: %.3f ms\n", totTime / batch);
	vkResetFences(vkGPU->device, 1, &vkGPU->fence);
	vkFreeCommandBuffers(vkGPU->device, vkGPU->commandPool, 1, &commandBuffer);
}
float performVulkanFFTiFFT(VkGPU* vkGPU, VkFFTApplication* app_forward, VkFFTApplication* app_inverse, uint32_t batch) {
	VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	commandBufferAllocateInfo.commandPool = vkGPU->commandPool;
	commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferAllocateInfo.commandBufferCount = 1;
	VkCommandBuffer commandBuffer = {};
	vkAllocateCommandBuffers(vkGPU->device, &commandBufferAllocateInfo, &commandBuffer);
	VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
	for (uint32_t i = 0; i < batch; i++) {
		VkFFTAppend(app_forward, commandBuffer);
		VkFFTAppend(app_inverse, commandBuffer);
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
	return totTime / batch;
}

VkResult sample_0(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	VkResult res = VK_SUCCESS;
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
			VkFFTConfiguration forward_configuration = defaultVkFFTConfiguration;
			VkFFTConfiguration inverse_configuration = defaultVkFFTConfiguration;
			VkFFTApplication app_forward = defaultVkFFTApplication;
			VkFFTApplication app_inverse = defaultVkFFTApplication;
			//FFT + iFFT sample code.
			//Setting up FFT configuration for forward and inverse FFT.
			forward_configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).
			forward_configuration.size[0] = 4 * pow(2, n); //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			if (n == 0) forward_configuration.size[0] = 4096;
			forward_configuration.size[1] = 64 * 32 * pow(2, 16) / forward_configuration.size[0];
			if (forward_configuration.size[1] < 1) forward_configuration.size[1] = 1;
			//forward_configuration.size[1] = (forward_configuration.size[1] > 32768) ? 32768 : forward_configuration.size[1];
			forward_configuration.size[2] = 1;
			for (uint32_t i = 0; i < 3; i++) {
				forward_configuration.bufferStride[i] = forward_configuration.size[i];//can specify arbitrary buffer strides, if buffer is bigger than data - can be used for upscaling - you put data in the bigger buffer (2x in each dimension) in the corner, transform forward, zeropad to full buffer size and inverse
			}

			//PARAMETERS THAT CAN BE ADJUSTED FOR SPECIFIC GPU's - this configuration is by no means final form
			switch (vkGPU->physicalDeviceProperties.vendorID) {
			case 0x10DE://NVIDIA
				forward_configuration.coalescedMemory = 32;//the coalesced memory is equal to 32 bytes between L2 and VRAM. 
				forward_configuration.useLUT = false;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 4;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			case 0x8086://INTEL
				forward_configuration.coalescedMemory = 64;
				forward_configuration.useLUT = true;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 2;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			case 0x1002://AMD
				forward_configuration.coalescedMemory = 32;
				forward_configuration.useLUT = false;
				forward_configuration.warpSize = 64;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = (vkGPU->physicalDeviceProperties.limits.maxComputeSharedMemorySize >= 65536) ? 2 : 4;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 19;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			default:
				forward_configuration.coalescedMemory = 64;
				forward_configuration.useLUT = false;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 1;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			}
			forward_configuration.reorderFourStep = true;
			forward_configuration.performZeropadding[0] = false; //Perform padding with zeros on GPU. Still need to properly align input data (no need to fill padding area with meaningful data) but this will increase performance due to the lower amount of the memory reads/writes and omitting sequences only consisting of zeros.
			forward_configuration.performZeropadding[1] = false;
			forward_configuration.performZeropadding[2] = false;
			forward_configuration.performConvolution = false; //Perform convolution with precomputed kernel. 
			forward_configuration.performR2C = false; //Perform C2C transform. Can be combined with all other options. 
			forward_configuration.coordinateFeatures = 1; //Specify dimensionality of the input feature vector (default 1). Each component is stored not as a vector, but as a separate system and padded on it's own according to other options (i.e. for x*y system of 3-vector, first x*y elements correspond to the first dimension, then goes x*y for the second, etc). 
			forward_configuration.inverse = false; //Direction of FFT. false - forward, true - inverse.
			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [VkDeviceSize *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [VkDeviceSize *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			forward_configuration.device = &vkGPU->device;
			forward_configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
			forward_configuration.fence = &vkGPU->fence;
			forward_configuration.commandPool = &vkGPU->commandPool;
			forward_configuration.physicalDevice = &vkGPU->physicalDevice;
			forward_configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization

			//Allocate buffer for the input data.
			VkDeviceSize bufferSize = ((uint64_t)forward_configuration.coordinateFeatures) * sizeof(float) * 2 * forward_configuration.bufferStride[0] * forward_configuration.bufferStride[1] * forward_configuration.bufferStride[2];;
			VkBuffer buffer = {};
			VkDeviceMemory bufferDeviceMemory = {};
			VkBuffer tempBuffer = {};
			VkDeviceMemory tempBufferDeviceMemory = {};
			allocateFFTBuffer(vkGPU, &buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
			allocateFFTBuffer(vkGPU, &tempBuffer, &tempBufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);

			forward_configuration.isInputFormatted = false; //set to true if input is a different buffer, so it can have zeropadding/R2C added . Have to specifiy corresponding inputBufferStride
			forward_configuration.isOutputFormatted = false;//set to true if output is a different buffer, so it can have zeropadding/C2R automatically removed. Have to specifiy corresponding outputBufferStride

			forward_configuration.bufferNum = 1;
			forward_configuration.tempBufferNum = 1;
			forward_configuration.inputBufferNum = 1;
			forward_configuration.outputBufferNum = 1;

			forward_configuration.buffer = &buffer;
			forward_configuration.tempBuffer = &tempBuffer;
			forward_configuration.inputBuffer = &buffer; //you can specify first buffer to read data from to be different from the buffer FFT is performed on. FFT is still in-place on the second buffer, this is here just for convenience.
			forward_configuration.outputBuffer = &buffer;

			forward_configuration.bufferSize = &bufferSize;
			forward_configuration.tempBufferSize = &bufferSize;
			forward_configuration.inputBufferSize = &bufferSize;
			forward_configuration.outputBufferSize = &bufferSize;

			//Now we will create a similar configuration for inverse FFT and change inverse parameter to true.
			inverse_configuration = forward_configuration;
			inverse_configuration.inputBuffer = &buffer;//If you continue working with previous data, select the FFT buffer as initial
			inverse_configuration.outputBuffer = &buffer;
			inverse_configuration.inverse = true;

			//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
			/*float* buffer_input = (float*)malloc(bufferSize);

			for (uint32_t v = 0; v < forward_configuration.coordinateFeatures; v++) {
				for (uint32_t k = 0; k < forward_configuration.size[2]; k++) {
					for (uint32_t j = 0; j < forward_configuration.size[1]; j++) {
						for (uint32_t i = 0; i < forward_configuration.size[0]; i++) {
							buffer_input[2 * (i + j * forward_configuration.size[0] + k * (forward_configuration.size[0]) * forward_configuration.size[1] + v * (forward_configuration.size[0]) * forward_configuration.size[1] * forward_configuration.size[2])] = 2 * ((float)rand()) / RAND_MAX - 1.0;
							buffer_input[2 * (i + j * forward_configuration.size[0] + k * (forward_configuration.size[0]) * forward_configuration.size[1] + v * (forward_configuration.size[0]) * forward_configuration.size[1] * forward_configuration.size[2]) + 1] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						}
					}
				}
			}
			*/
			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
			transferDataFromCPU(vkGPU, buffer_input, &buffer, bufferSize);
			//free(buffer_input);

			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			res = initializeVulkanFFT(&app_forward, forward_configuration);
			if (res != VK_SUCCESS) return res;
			res = initializeVulkanFFT(&app_inverse, inverse_configuration);
			if (res != VK_SUCCESS) return res;
			//Submit FFT+iFFT.
			uint32_t batch = ((3 * 4096 * 1024.0 * 1024.0) / bufferSize > 1000) ? 1000 : (3 * 4096 * 1024.0 * 1024.0) / bufferSize;
			if (vkGPU->physicalDeviceProperties.vendorID == 0x8086) batch /= 4;
			if (batch == 0) batch = 1;
			float totTime = performVulkanFFTiFFT(vkGPU, &app_forward, &app_inverse, batch);

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
					for (uint32_t i = 0; i < forward_configuration.FFTdim; i++)
						num_tot_transfers += app_forward.localFFTPlan->numAxisUploads[i];
					num_tot_transfers *= 4;
					if (file_output)
						fprintf(output, "VkFFT System: %d %dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f batch: %d benchmark: %d bandwidth: %0.1f\n", (int)log2(forward_configuration.size[0]), forward_configuration.size[0], forward_configuration.size[1], bufferSize / 1024 / 1024, avg_time, std_error, batch, (int)(((double)bufferSize / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);

					printf("VkFFT System: %d %dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f batch: %d benchmark: %d bandwidth: %0.1f\n", (int)log2(forward_configuration.size[0]), forward_configuration.size[0], forward_configuration.size[1], bufferSize / 1024 / 1024, avg_time, std_error, batch, (int)(((double)bufferSize / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);
					benchmark_result += ((double)bufferSize / 1024) / avg_time;
				}


			}

			vkDestroyBuffer(vkGPU->device, buffer, NULL);
			vkDestroyBuffer(vkGPU->device, tempBuffer, NULL);
			vkFreeMemory(vkGPU->device, bufferDeviceMemory, NULL);
			vkFreeMemory(vkGPU->device, tempBufferDeviceMemory, NULL);
			deleteVulkanFFT(&app_forward);
			deleteVulkanFFT(&app_inverse);
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
VkResult sample_1(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	VkResult res = VK_SUCCESS;
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
			VkFFTConfiguration forward_configuration = defaultVkFFTConfiguration;
			VkFFTConfiguration inverse_configuration = defaultVkFFTConfiguration;
			VkFFTApplication app_forward = defaultVkFFTApplication;
			VkFFTApplication app_inverse = defaultVkFFTApplication;
			//FFT + iFFT sample code.
			//Setting up FFT configuration for forward and inverse FFT.
			forward_configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).
			forward_configuration.size[0] = 4 * pow(2, n); //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			if (n == 0) forward_configuration.size[0] = 2048;
			forward_configuration.size[1] = 64 * 32 * pow(2, 15) / forward_configuration.size[0];
			if (forward_configuration.size[1] < 1) forward_configuration.size[1] = 1;
			forward_configuration.size[2] = 1;

			for (uint32_t i = 0; i < 3; i++) {
				forward_configuration.bufferStride[i] = forward_configuration.size[i];//can specify arbitrary buffer strides, if buffer is bigger than data - can be used for upscaling - you put data in the bigger buffer (2x in each dimension) in the corner, transform forward, zeropad to full buffer size and inverse
			}
			//PARAMETERS THAT CAN BE ADJUSTED FOR SPECIFIC GPU's - this configuration is by no means final form
			switch (vkGPU->physicalDeviceProperties.vendorID) {
			case 0x10DE://NVIDIA 
				forward_configuration.coalescedMemory = 32;//the coalesced memory is equal to 32 bytes between L2 and VRAM. 
				forward_configuration.useLUT = true;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 4;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			case 0x8086://INTEL
				forward_configuration.coalescedMemory = 64;
				forward_configuration.useLUT = true;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 1;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			case 0x1002://AMD
				forward_configuration.coalescedMemory = 32;
				forward_configuration.useLUT = true;
				forward_configuration.warpSize = 64;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = (vkGPU->physicalDeviceProperties.limits.maxComputeSharedMemorySize >= 65536) ? 2 : 4;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 20;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			default:
				forward_configuration.coalescedMemory = 64;
				forward_configuration.useLUT = true;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 1;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			}
			forward_configuration.reorderFourStep = true;

			forward_configuration.performZeropadding[0] = false; //Perform padding with zeros on GPU. Still need to properly align input data (no need to fill padding area with meaningful data) but this will increase performance due to the lower amount of the memory reads/writes and omitting sequences only consisting of zeros.
			forward_configuration.performZeropadding[1] = false;
			forward_configuration.performZeropadding[2] = false;
			forward_configuration.performConvolution = false; //Perform convolution with precomputed kernel. 
			forward_configuration.performR2C = false; //Perform C2C transform. Can be combined with all other options. 
			forward_configuration.coordinateFeatures = 1; //Specify dimensionality of the input feature vector (default 1). Each component is stored not as a vector, but as a separate system and padded on it's own according to other options (i.e. for x*y system of 3-vector, first x*y elements correspond to the first dimension, then goes x*y for the second, etc). 
			forward_configuration.inverse = false; //Direction of FFT. false - forward, true - inverse.
			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [VkDeviceSize *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [VkDeviceSize *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			forward_configuration.device = &vkGPU->device;
			forward_configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
			forward_configuration.fence = &vkGPU->fence;
			forward_configuration.commandPool = &vkGPU->commandPool;
			forward_configuration.physicalDevice = &vkGPU->physicalDevice;
			forward_configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization
			forward_configuration.doublePrecision = true;

			//Allocate buffer for the input data.
			VkDeviceSize bufferSize = ((uint64_t)forward_configuration.coordinateFeatures) * sizeof(double) * 2 * forward_configuration.bufferStride[0] * forward_configuration.bufferStride[1] * forward_configuration.bufferStride[2];;
			VkBuffer buffer = {};
			VkDeviceMemory bufferDeviceMemory = {};
			VkBuffer tempBuffer = {};
			VkDeviceMemory tempBufferDeviceMemory = {};
			allocateFFTBuffer(vkGPU, &buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
			allocateFFTBuffer(vkGPU, &tempBuffer, &tempBufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);

			forward_configuration.isInputFormatted = false; //set to true if input is a different buffer, so it can have zeropadding/R2C added
			forward_configuration.isOutputFormatted = false;//set to true if output is a different buffer, so it can have zeropadding/C2R automatically removed. Have to specifiy corresponding outputBufferStride

			forward_configuration.bufferNum = 1;
			forward_configuration.tempBufferNum = 1;
			forward_configuration.inputBufferNum = 1;
			forward_configuration.outputBufferNum = 1;

			forward_configuration.buffer = &buffer;
			forward_configuration.tempBuffer = &tempBuffer;
			forward_configuration.inputBuffer = &buffer; //you can specify first buffer to read data from to be different from the buffer FFT is performed on. FFT is still in-place on the second buffer, this is here just for convenience.
			forward_configuration.outputBuffer = &buffer;

			forward_configuration.bufferSize = &bufferSize;
			forward_configuration.tempBufferSize = &bufferSize;
			forward_configuration.inputBufferSize = &bufferSize;
			forward_configuration.outputBufferSize = &bufferSize;

			//Now we will create a similar configuration for inverse FFT and change inverse parameter to true.
			inverse_configuration = forward_configuration;
			inverse_configuration.inputBuffer = &buffer;//If you continue working with previous data, select the FFT buffer as initial
			inverse_configuration.outputBuffer = &buffer;
			inverse_configuration.inverse = true;


			//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
			/*float* buffer_input = (float*)malloc(bufferSize);

			for (uint32_t v = 0; v < forward_configuration.coordinateFeatures; v++) {
				for (uint32_t k = 0; k < forward_configuration.size[2]; k++) {
					for (uint32_t j = 0; j < forward_configuration.size[1]; j++) {
						for (uint32_t i = 0; i < forward_configuration.size[0]; i++) {
							buffer_input[2 * (i + j * forward_configuration.size[0] + k * (forward_configuration.size[0]) * forward_configuration.size[1] + v * (forward_configuration.size[0]) * forward_configuration.size[1] * forward_configuration.size[2])] = 2 * ((float)rand()) / RAND_MAX - 1.0;
							buffer_input[2 * (i + j * forward_configuration.size[0] + k * (forward_configuration.size[0]) * forward_configuration.size[1] + v * (forward_configuration.size[0]) * forward_configuration.size[1] * forward_configuration.size[2]) + 1] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						}
					}
				}
			}
			*/
			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
			transferDataFromCPU(vkGPU, buffer_input, &buffer, bufferSize);
			//free(buffer_input);

			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			res = initializeVulkanFFT(&app_forward, forward_configuration);
			if (res != VK_SUCCESS) return res;
			res = initializeVulkanFFT(&app_inverse, inverse_configuration);
			if (res != VK_SUCCESS) return res;
			//Submit FFT+iFFT.
			uint32_t batch = ((4096 * 1024.0 * 1024.0) / bufferSize > 1000) ? 1000 : (4096 * 1024.0 * 1024.0) / bufferSize;
			if (vkGPU->physicalDeviceProperties.vendorID == 0x8086) batch /= 4;
			if (batch == 0) batch = 1;

			float totTime = performVulkanFFTiFFT(vkGPU, &app_forward, &app_inverse, batch);

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
					for (uint32_t i = 0; i < forward_configuration.FFTdim; i++)
						num_tot_transfers += app_forward.localFFTPlan->numAxisUploads[i];
					num_tot_transfers *= 4;
					if (file_output)
						fprintf(output, "VkFFT System: %d %dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f batch: %d benchmark: %d bandwidth: %0.1f\n", (int)log2(forward_configuration.size[0]), forward_configuration.size[0], forward_configuration.size[1], bufferSize / 1024 / 1024, avg_time, std_error, batch, (int)(((double)bufferSize * sizeof(float) / sizeof(double) / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);

					printf("VkFFT System: %d %dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f batch: %d benchmark: %d bandwidth: %0.1f\n", (int)log2(forward_configuration.size[0]), forward_configuration.size[0], forward_configuration.size[1], bufferSize / 1024 / 1024, avg_time, std_error, batch, (int)(((double)bufferSize * sizeof(float) / sizeof(double) / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);
					benchmark_result += ((double)bufferSize * sizeof(float) / sizeof(double) / 1024) / avg_time;
				}


			}

			vkDestroyBuffer(vkGPU->device, buffer, NULL);
			vkDestroyBuffer(vkGPU->device, tempBuffer, NULL);
			vkFreeMemory(vkGPU->device, bufferDeviceMemory, NULL);
			vkFreeMemory(vkGPU->device, tempBufferDeviceMemory, NULL);
			deleteVulkanFFT(&app_forward);
			deleteVulkanFFT(&app_inverse);
		}
	}
	free(buffer_input);
	benchmark_result /= (24 - 1);
	if (file_output) {
		fprintf(output, "Benchmark score VkFFT: %d\n", (int)(benchmark_result));
		fprintf(output, "Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
	}
	printf("Benchmark score VkFFT: %d\n", (int)(benchmark_result));
	printf("Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
	return res;
}
#if (VK_API_VERSION>10)
VkResult sample_2(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	VkResult res = VK_SUCCESS;
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
			VkFFTConfiguration forward_configuration = defaultVkFFTConfiguration;
			VkFFTConfiguration inverse_configuration = defaultVkFFTConfiguration;
			VkFFTApplication app_forward = defaultVkFFTApplication;
			VkFFTApplication app_inverse = defaultVkFFTApplication;
			//FFT + iFFT sample code.
			//Setting up FFT configuration for forward and inverse FFT.
			forward_configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).
			forward_configuration.size[0] = 4 * pow(2, n); //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			if (n == 0) forward_configuration.size[0] = 4096;
			forward_configuration.size[1] = 64 * 32 * pow(2, 16) / forward_configuration.size[0];
			if (forward_configuration.size[1] < 1) forward_configuration.size[1] = 1;
			forward_configuration.size[2] = 1;
			for (uint32_t i = 0; i < 3; i++) {
				forward_configuration.bufferStride[i] = forward_configuration.size[i];//can specify arbitrary buffer strides, if buffer is bigger than data - can be used for upscaling - you put data in the bigger buffer (2x in each dimension) in the corner, transform forward, zeropad to full buffer size and inverse
			}
			//PARAMETERS THAT CAN BE ADJUSTED FOR SPECIFIC GPU's - this configuration is by no means final form
			switch (vkGPU->physicalDeviceProperties.vendorID) {
			case 0x10DE://NVIDIA 
				forward_configuration.coalescedMemory = 64;//have to set coalesce more, as calculations are still float, while uploads are half.
				forward_configuration.useLUT = false;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 4;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			case 0x8086://INTEL
				if (n > 22)//128byte coalescing has a limit of 2^24 max size
					forward_configuration.coalescedMemory = 64;
				else
					forward_configuration.coalescedMemory = 128;
				forward_configuration.useLUT = true;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 1;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			case 0x1002://AMD
				forward_configuration.coalescedMemory = 64;
				forward_configuration.useLUT = false;
				forward_configuration.warpSize = 64;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = (vkGPU->physicalDeviceProperties.limits.maxComputeSharedMemorySize >= 65536) ? 2 : 4;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			default:
				forward_configuration.coalescedMemory = 64;
				forward_configuration.useLUT = false;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 1;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			}
			forward_configuration.reorderFourStep = true;

			forward_configuration.performZeropadding[0] = false; //Perform padding with zeros on GPU. Still need to properly align input data (no need to fill padding area with meaningful data) but this will increase performance due to the lower amount of the memory reads/writes and omitting sequences only consisting of zeros.
			forward_configuration.performZeropadding[1] = false;
			forward_configuration.performZeropadding[2] = false;
			forward_configuration.performConvolution = false; //Perform convolution with precomputed kernel. 
			forward_configuration.performR2C = false; //Perform C2C transform. Can be combined with all other options. 
			forward_configuration.coordinateFeatures = 1; //Specify dimensionality of the input feature vector (default 1). Each component is stored not as a vector, but as a separate system and padded on it's own according to other options (i.e. for x*y system of 3-vector, first x*y elements correspond to the first dimension, then goes x*y for the second, etc). 
			forward_configuration.inverse = false; //Direction of FFT. false - forward, true - inverse.
			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [VkDeviceSize *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [VkDeviceSize *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			forward_configuration.device = &vkGPU->device;
			forward_configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
			forward_configuration.fence = &vkGPU->fence;
			forward_configuration.commandPool = &vkGPU->commandPool;
			forward_configuration.physicalDevice = &vkGPU->physicalDevice;
			forward_configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization
			forward_configuration.halfPrecision = true;

			//Allocate buffer for the input data.
			VkDeviceSize bufferSize = ((uint64_t)forward_configuration.coordinateFeatures) * 2 * sizeof(half) * forward_configuration.bufferStride[0] * forward_configuration.bufferStride[1] * forward_configuration.bufferStride[2];;
			VkBuffer buffer = {};
			VkDeviceMemory bufferDeviceMemory = {};
			VkBuffer tempBuffer = {};
			VkDeviceMemory tempBufferDeviceMemory = {};
			allocateFFTBuffer(vkGPU, &buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
			allocateFFTBuffer(vkGPU, &tempBuffer, &tempBufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);

			forward_configuration.isInputFormatted = false; //set to true if input is a different buffer, so it can have zeropadding/R2C added . Have to specifiy corresponding inputBufferStride
			forward_configuration.isOutputFormatted = false;//set to true if output is a different buffer, so it can have zeropadding/C2R automatically removed. Have to specifiy corresponding outputBufferStride

			forward_configuration.bufferNum = 1;
			forward_configuration.tempBufferNum = 1;
			forward_configuration.inputBufferNum = 1;
			forward_configuration.outputBufferNum = 1;

			forward_configuration.buffer = &buffer;
			forward_configuration.tempBuffer = &tempBuffer;
			forward_configuration.inputBuffer = &buffer; //you can specify first buffer to read data from to be different from the buffer FFT is performed on. FFT is still in-place on the second buffer, this is here just for convenience.
			forward_configuration.outputBuffer = &buffer;

			forward_configuration.bufferSize = &bufferSize;
			forward_configuration.tempBufferSize = &bufferSize;
			forward_configuration.inputBufferSize = &bufferSize;
			forward_configuration.outputBufferSize = &bufferSize;

			//Now we will create a similar configuration for inverse FFT and change inverse parameter to true.
			inverse_configuration = forward_configuration;
			inverse_configuration.inputBuffer = &buffer;//If you continue working with previous data, select the FFT buffer as initial
			inverse_configuration.outputBuffer = &buffer;
			inverse_configuration.inverse = true;


			//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
			/*float* buffer_input = (float*)malloc(bufferSize);

			for (uint32_t v = 0; v < forward_configuration.coordinateFeatures; v++) {
				for (uint32_t k = 0; k < forward_configuration.size[2]; k++) {
					for (uint32_t j = 0; j < forward_configuration.size[1]; j++) {
						for (uint32_t i = 0; i < forward_configuration.size[0]; i++) {
							buffer_input[2 * (i + j * forward_configuration.size[0] + k * (forward_configuration.size[0]) * forward_configuration.size[1] + v * (forward_configuration.size[0]) * forward_configuration.size[1] * forward_configuration.size[2])] = 2 * ((float)rand()) / RAND_MAX - 1.0;
							buffer_input[2 * (i + j * forward_configuration.size[0] + k * (forward_configuration.size[0]) * forward_configuration.size[1] + v * (forward_configuration.size[0]) * forward_configuration.size[1] * forward_configuration.size[2]) + 1] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						}
					}
				}
			}
			*/
			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
			transferDataFromCPU(vkGPU, buffer_input, &buffer, bufferSize);
			//free(buffer_input);

			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			res = initializeVulkanFFT(&app_forward, forward_configuration);
			if (res != VK_SUCCESS) return res;
			res = initializeVulkanFFT(&app_inverse, inverse_configuration);
			if (res != VK_SUCCESS) return res;
			//Submit FFT+iFFT.
			uint32_t batch = ((4096 * 1024.0 * 1024.0) / bufferSize > 1000) ? 1000 : (4096 * 1024.0 * 1024.0) / bufferSize;
			if (vkGPU->physicalDeviceProperties.vendorID == 0x8086) batch /= 4;
			if (batch == 0) batch = 1;
			float totTime = performVulkanFFTiFFT(vkGPU, &app_forward, &app_inverse, batch);

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
					for (uint32_t i = 0; i < forward_configuration.FFTdim; i++)
						num_tot_transfers += app_forward.localFFTPlan->numAxisUploads[i];
					num_tot_transfers *= 4;
					if (file_output)
						fprintf(output, "VkFFT System: %d %dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f batch: %d benchmark: %d bandwidth: %0.1f\n", (int)log2(forward_configuration.size[0]), forward_configuration.size[0], forward_configuration.size[1], bufferSize / 1024 / 1024, avg_time, std_error, batch, (int)(((double)bufferSize * sizeof(float) / sizeof(half) / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);

					printf("VkFFT System: %d %dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f batch: %d benchmark: %d bandwidth: %0.1f\n", (int)log2(forward_configuration.size[0]), forward_configuration.size[0], forward_configuration.size[1], bufferSize / 1024 / 1024, avg_time, std_error, batch, (int)(((double)bufferSize * sizeof(float) / sizeof(half) / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);
					benchmark_result += ((double)bufferSize * sizeof(float) / sizeof(half) / 1024) / avg_time;
				}


			}

			vkDestroyBuffer(vkGPU->device, buffer, NULL);
			vkDestroyBuffer(vkGPU->device, tempBuffer, NULL);
			vkFreeMemory(vkGPU->device, bufferDeviceMemory, NULL);
			vkFreeMemory(vkGPU->device, tempBufferDeviceMemory, NULL);
			deleteVulkanFFT(&app_forward);
			deleteVulkanFFT(&app_inverse);
		}
	}
	free(buffer_input);
	benchmark_result /= (25 - 1);
	if (file_output) {
		fprintf(output, "Benchmark score VkFFT: %d\n", (int)(benchmark_result));
		fprintf(output, "Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
	}
	printf("Benchmark score VkFFT: %d\n", (int)(benchmark_result));
	printf("Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
	return res;
}
#endif
VkResult sample_3(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	VkResult res = VK_SUCCESS;
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
			VkFFTConfiguration forward_configuration = defaultVkFFTConfiguration;
			VkFFTConfiguration inverse_configuration = defaultVkFFTConfiguration;
			VkFFTApplication app_forward = defaultVkFFTApplication;
			VkFFTApplication app_inverse = defaultVkFFTApplication;
			//FFT + iFFT sample code.
			//Setting up FFT configuration for forward and inverse FFT.
			forward_configuration.FFTdim = benchmark_dimensions[n][3]; //FFT dimension, 1D, 2D or 3D (default 1).
			forward_configuration.size[0] = benchmark_dimensions[n][0]; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			forward_configuration.size[1] = benchmark_dimensions[n][1];
			forward_configuration.size[2] = benchmark_dimensions[n][2];
			for (uint32_t i = 0; i < 3; i++) {
				forward_configuration.bufferStride[i] = forward_configuration.size[i];//can specify arbitrary buffer strides, if buffer is bigger than data - can be used for upscaling - you put data in the bigger buffer (2x in each dimension) in the corner, transform forward, zeropad to full buffer size and inverse
			}
			//PARAMETERS THAT CAN BE ADJUSTED FOR SPECIFIC GPU's - this configuration is by no means final form
			switch (vkGPU->physicalDeviceProperties.vendorID) {
			case 0x10DE://NVIDIA
				forward_configuration.coalescedMemory = 32;//the coalesced memory is equal to 32 bytes between L2 and VRAM. 
				forward_configuration.useLUT = false;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 4;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = 0;
				break;
			case 0x8086://INTEL
				forward_configuration.coalescedMemory = 64;
				forward_configuration.useLUT = true;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 2;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			case 0x1002://AMD
				forward_configuration.coalescedMemory = 32;
				forward_configuration.useLUT = false;
				forward_configuration.warpSize = 64;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = (vkGPU->physicalDeviceProperties.limits.maxComputeSharedMemorySize >= 65536) ? 2 : 4;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 19;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			default:
				forward_configuration.coalescedMemory = 64;
				forward_configuration.useLUT = false;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 1;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			}

			forward_configuration.performZeropadding[0] = false; //Perform padding with zeros on GPU. Still need to properly align input data (no need to fill padding area with meaningful data) but this will increase performance due to the lower amount of the memory reads/writes and omitting sequences only consisting of zeros.
			forward_configuration.performZeropadding[1] = false;
			forward_configuration.performZeropadding[2] = false;
			forward_configuration.performConvolution = false; //Perform convolution with precomputed kernel. 
			forward_configuration.performR2C = false; //Perform C2C transform. Can be combined with all other options. 
			forward_configuration.coordinateFeatures = 1; //Specify dimensionality of the input feature vector (default 1). Each component is stored not as a vector, but as a separate system and padded on it's own according to other options (i.e. for x*y system of 3-vector, first x*y elements correspond to the first dimension, then goes x*y for the second, etc). 
			forward_configuration.inverse = false; //Direction of FFT. false - forward, true - inverse.
			forward_configuration.reorderFourStep = true;//set to true if you want data to return to correct layout after FFT. Set to false if you use convolution routine. Requires additional tempBuffer of bufferSize (see below) to do reordering
			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [VkDeviceSize *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [VkDeviceSize *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			forward_configuration.device = &vkGPU->device;
			forward_configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
			forward_configuration.fence = &vkGPU->fence;
			forward_configuration.commandPool = &vkGPU->commandPool;
			forward_configuration.physicalDevice = &vkGPU->physicalDevice;
			forward_configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization

			//Allocate buffer for the input data.
			VkDeviceSize bufferSize = ((uint64_t)forward_configuration.coordinateFeatures) * sizeof(float) * 2 * forward_configuration.bufferStride[0] * forward_configuration.bufferStride[1] * forward_configuration.bufferStride[2];;
			VkBuffer buffer = {};
			VkDeviceMemory bufferDeviceMemory = {};
			VkBuffer tempBuffer = {};
			VkDeviceMemory tempBufferDeviceMemory = {};
			allocateFFTBuffer(vkGPU, &buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
			allocateFFTBuffer(vkGPU, &tempBuffer, &tempBufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);

			forward_configuration.isInputFormatted = false; //set to true if input is a different buffer, so it can have zeropadding/R2C added . Have to specifiy corresponding inputBufferStride
			forward_configuration.isOutputFormatted = false;//set to true if output is a different buffer, so it can have zeropadding/C2R automatically removed. Have to specifiy corresponding outputBufferStride

			forward_configuration.bufferNum = 1;
			forward_configuration.tempBufferNum = 1;
			forward_configuration.inputBufferNum = 1;
			forward_configuration.outputBufferNum = 1;

			forward_configuration.buffer = &buffer;
			forward_configuration.tempBuffer = &tempBuffer;
			forward_configuration.inputBuffer = &buffer; //you can specify first buffer to read data from to be different from the buffer FFT is performed on. FFT is still in-place on the second buffer, this is here just for convenience.
			forward_configuration.outputBuffer = &buffer;

			forward_configuration.bufferSize = &bufferSize;
			forward_configuration.tempBufferSize = &bufferSize;
			forward_configuration.inputBufferSize = &bufferSize;
			forward_configuration.outputBufferSize = &bufferSize;

			//Now we will create a similar configuration for inverse FFT and change inverse parameter to true.
			inverse_configuration = forward_configuration;
			inverse_configuration.inputBuffer = &buffer;//If you continue working with previous data, select the FFT buffer as initial
			inverse_configuration.outputBuffer = &buffer;
			inverse_configuration.inverse = true;


			//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
			/*float* buffer_input = (float*)malloc(bufferSize);

			for (uint32_t v = 0; v < forward_configuration.coordinateFeatures; v++) {
				for (uint32_t k = 0; k < forward_configuration.size[2]; k++) {
					for (uint32_t j = 0; j < forward_configuration.size[1]; j++) {
						for (uint32_t i = 0; i < forward_configuration.size[0]; i++) {
							buffer_input[2 * (i + j * forward_configuration.size[0] + k * (forward_configuration.size[0]) * forward_configuration.size[1] + v * (forward_configuration.size[0]) * forward_configuration.size[1] * forward_configuration.size[2])] = 2 * ((float)rand()) / RAND_MAX - 1.0;
							buffer_input[2 * (i + j * forward_configuration.size[0] + k * (forward_configuration.size[0]) * forward_configuration.size[1] + v * (forward_configuration.size[0]) * forward_configuration.size[1] * forward_configuration.size[2]) + 1] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						}
					}
				}
			}
			*/
			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
			transferDataFromCPU(vkGPU, buffer_input, &buffer, bufferSize);
			//free(buffer_input);

			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			res = initializeVulkanFFT(&app_forward, forward_configuration);
			if (res != VK_SUCCESS) return res;
			res = initializeVulkanFFT(&app_inverse, inverse_configuration);
			if (res != VK_SUCCESS) return res;
			//Submit FFT+iFFT.
			uint32_t batch = ((4096 * 1024.0 * 1024.0) / bufferSize > 1000) ? 1000 : (4096 * 1024.0 * 1024.0) / bufferSize;
			if (vkGPU->physicalDeviceProperties.vendorID == 0x8086) batch /= 4;
			if (batch == 0) batch = 1;

			float totTime = performVulkanFFTiFFT(vkGPU, &app_forward, &app_inverse, batch);

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
					for (uint32_t i = 0; i < forward_configuration.FFTdim; i++)
						num_tot_transfers += app_forward.localFFTPlan->numAxisUploads[i];
					num_tot_transfers *= 4;
					if (file_output)
						fprintf(output, "VkFFT System: %dx%dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f batch: %d benchmark: %d bandwidth: %0.1f\n", benchmark_dimensions[n][0], benchmark_dimensions[n][1], benchmark_dimensions[n][2], bufferSize / 1024 / 1024, avg_time, std_error, batch, (int)(((double)bufferSize / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);
					printf("VkFFT System: %dx%dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f batch: %d benchmark: %d bandwidth: %0.1f\n", benchmark_dimensions[n][0], benchmark_dimensions[n][1], benchmark_dimensions[n][2], bufferSize / 1024 / 1024, avg_time, std_error, batch, (int)(((double)bufferSize / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);
					benchmark_result += ((double)bufferSize / 1024) / avg_time;
				}


			}

			vkDestroyBuffer(vkGPU->device, buffer, NULL);
			vkFreeMemory(vkGPU->device, bufferDeviceMemory, NULL);
			vkDestroyBuffer(vkGPU->device, tempBuffer, NULL);
			vkFreeMemory(vkGPU->device, tempBufferDeviceMemory, NULL);
			deleteVulkanFFT(&app_forward);
			deleteVulkanFFT(&app_inverse);
		}
	}
	free(buffer_input);
	benchmark_result /= (num_benchmark_samples - 1);

	if (file_output) {
		fprintf(output, "Benchmark score VkFFT: %d\n", (int)(benchmark_result));
		fprintf(output, "Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
	}
	printf("Benchmark score VkFFT: %d\n", (int)(benchmark_result));
	printf("Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
	return res;
}
VkResult sample_4(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	VkResult res = VK_SUCCESS;
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
			VkFFTConfiguration forward_configuration = defaultVkFFTConfiguration;
			VkFFTConfiguration inverse_configuration = defaultVkFFTConfiguration;
			VkFFTApplication app_forward = defaultVkFFTApplication;
			VkFFTApplication app_inverse = defaultVkFFTApplication;
			//FFT + iFFT sample code.
			//Setting up FFT configuration for forward and inverse FFT.
			forward_configuration.FFTdim = benchmark_dimensions[n][3]; //FFT dimension, 1D, 2D or 3D (default 1).
			forward_configuration.size[0] = benchmark_dimensions[n][0]; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			forward_configuration.size[1] = benchmark_dimensions[n][1];
			forward_configuration.size[2] = benchmark_dimensions[n][2];
			for (uint32_t i = 0; i < 3; i++) {
				forward_configuration.bufferStride[i] = forward_configuration.size[i];//can specify arbitrary buffer strides, if buffer is bigger than data - can be used for upscaling - you put data in the bigger buffer (2x in each dimension) in the corner, transform forward, zeropad to full buffer size and inverse
			}
			//PARAMETERS THAT CAN BE ADJUSTED FOR SPECIFIC GPU's - this configuration is by no means final form
			switch (vkGPU->physicalDeviceProperties.vendorID) {
			case 0x10DE://NVIDIA
				forward_configuration.coalescedMemory = 32;//the coalesced memory is equal to 32 bytes between L2 and VRAM. 
				forward_configuration.useLUT = false;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 4;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			case 0x8086://INTEL
				forward_configuration.coalescedMemory = 64;
				forward_configuration.useLUT = true;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 2;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			case 0x1002://AMD
				forward_configuration.coalescedMemory = 32;
				forward_configuration.useLUT = false;
				forward_configuration.warpSize = 64;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = (vkGPU->physicalDeviceProperties.limits.maxComputeSharedMemorySize >= 65536) ? 2 : 4;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 19;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			default:
				forward_configuration.coalescedMemory = 64;
				forward_configuration.useLUT = false;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 1;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			}

			forward_configuration.performZeropadding[0] = true; //Perform padding with zeros on GPU. Still need to properly align input data (no need to fill padding area with meaningful data) but this will increase performance due to the lower amount of the memory reads/writes and omitting sequences only consisting of zeros.
			forward_configuration.performZeropadding[1] = true;
			forward_configuration.performZeropadding[2] = true;
			forward_configuration.fft_zeropad_left[0] = ceil(forward_configuration.size[0] / 2.0);
			forward_configuration.fft_zeropad_right[0] = forward_configuration.size[0];
			forward_configuration.fft_zeropad_left[1] = ceil(forward_configuration.size[1] / 2.0);
			forward_configuration.fft_zeropad_right[1] = forward_configuration.size[1];
			forward_configuration.fft_zeropad_left[2] = ceil(forward_configuration.size[2] / 2.0);
			forward_configuration.fft_zeropad_right[2] = forward_configuration.size[2];
			forward_configuration.performConvolution = false; //Perform convolution with precomputed kernel. 
			forward_configuration.performR2C = false; //Perform C2C transform. Can be combined with all other options. 
			forward_configuration.coordinateFeatures = 1; //Specify dimensionality of the input feature vector (default 1). Each component is stored not as a vector, but as a separate system and padded on it's own according to other options (i.e. for x*y system of 3-vector, first x*y elements correspond to the first dimension, then goes x*y for the second, etc). 
			forward_configuration.inverse = false; //Direction of FFT. false - forward, true - inverse.
			forward_configuration.reorderFourStep = true;//set to true if you want data to return to correct layout after FFT. Set to false if you use convolution routine. Requires additional tempBuffer of bufferSize (see below) to do reordering
			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [VkDeviceSize *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [VkDeviceSize *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			forward_configuration.device = &vkGPU->device;
			forward_configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
			forward_configuration.fence = &vkGPU->fence;
			forward_configuration.commandPool = &vkGPU->commandPool;
			forward_configuration.physicalDevice = &vkGPU->physicalDevice;
			forward_configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization

			//Allocate buffer for the input data.
			VkDeviceSize bufferSize = ((uint64_t)forward_configuration.coordinateFeatures) * sizeof(float) * 2 * forward_configuration.bufferStride[0] * forward_configuration.bufferStride[1] * forward_configuration.bufferStride[2];;
			VkBuffer buffer = {};
			VkDeviceMemory bufferDeviceMemory = {};
			VkBuffer tempBuffer = {};
			VkDeviceMemory tempBufferDeviceMemory = {};
			allocateFFTBuffer(vkGPU, &buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
			allocateFFTBuffer(vkGPU, &tempBuffer, &tempBufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);

			forward_configuration.isInputFormatted = false; //set to true if input is a different buffer, so it can have zeropadding/R2C added . Have to specifiy corresponding inputBufferStride
			forward_configuration.isOutputFormatted = false;//set to true if output is a different buffer, so it can have zeropadding/C2R automatically removed. Have to specifiy corresponding outputBufferStride

			forward_configuration.bufferNum = 1;
			forward_configuration.tempBufferNum = 1;
			forward_configuration.inputBufferNum = 1;
			forward_configuration.outputBufferNum = 1;

			forward_configuration.buffer = &buffer;
			forward_configuration.tempBuffer = &tempBuffer;
			forward_configuration.inputBuffer = &buffer; //you can specify first buffer to read data from to be different from the buffer FFT is performed on. FFT is still in-place on the second buffer, this is here just for convenience.
			forward_configuration.outputBuffer = &buffer;

			forward_configuration.bufferSize = &bufferSize;
			forward_configuration.tempBufferSize = &bufferSize;
			forward_configuration.inputBufferSize = &bufferSize;
			forward_configuration.outputBufferSize = &bufferSize;

			//Now we will create a similar configuration for inverse FFT and change inverse parameter to true.
			inverse_configuration = forward_configuration;
			inverse_configuration.inputBuffer = &buffer;//If you continue working with previous data, select the FFT buffer as initial
			inverse_configuration.outputBuffer = &buffer;
			inverse_configuration.inverse = true;


			//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
			/*float* buffer_input = (float*)malloc(bufferSize);

			for (uint32_t v = 0; v < forward_configuration.coordinateFeatures; v++) {
				for (uint32_t k = 0; k < forward_configuration.size[2]; k++) {
					for (uint32_t j = 0; j < forward_configuration.size[1]; j++) {
						for (uint32_t i = 0; i < forward_configuration.size[0]; i++) {
							buffer_input[2 * (i + j * forward_configuration.size[0] + k * (forward_configuration.size[0]) * forward_configuration.size[1] + v * (forward_configuration.size[0]) * forward_configuration.size[1] * forward_configuration.size[2])] = 2 * ((float)rand()) / RAND_MAX - 1.0;
							buffer_input[2 * (i + j * forward_configuration.size[0] + k * (forward_configuration.size[0]) * forward_configuration.size[1] + v * (forward_configuration.size[0]) * forward_configuration.size[1] * forward_configuration.size[2]) + 1] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						}
					}
				}
			}
			*/
			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
			transferDataFromCPU(vkGPU, buffer_input, &buffer, bufferSize);
			//free(buffer_input);

			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			res = initializeVulkanFFT(&app_forward, forward_configuration);
			if (res != VK_SUCCESS) return res;
			res = initializeVulkanFFT(&app_inverse, inverse_configuration);
			if (res != VK_SUCCESS) return res;
			//Submit FFT+iFFT.
			uint32_t batch = ((4096 * 1024.0 * 1024.0) / bufferSize > 1000) ? 1000 : (4096 * 1024.0 * 1024.0) / bufferSize;
			if (vkGPU->physicalDeviceProperties.vendorID == 0x8086) batch /= 4;
			if (batch == 0) batch = 1;

			float totTime = performVulkanFFTiFFT(vkGPU, &app_forward, &app_inverse, batch);

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
					for (uint32_t i = 0; i < forward_configuration.FFTdim; i++)
						num_tot_transfers += app_forward.localFFTPlan->numAxisUploads[i];
					num_tot_transfers *= 4;
					if (file_output)
						fprintf(output, "VkFFT System: %dx%dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f batch: %d benchmark: %d bandwidth: %0.1f\n", benchmark_dimensions[n][0], benchmark_dimensions[n][1], benchmark_dimensions[n][2], bufferSize / 1024 / 1024, avg_time, std_error, batch, (int)(((double)bufferSize / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);
					printf("VkFFT System: %dx%dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f batch: %d benchmark: %d bandwidth: %0.1f\n", benchmark_dimensions[n][0], benchmark_dimensions[n][1], benchmark_dimensions[n][2], bufferSize / 1024 / 1024, avg_time, std_error, batch, (int)(((double)bufferSize / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);
					benchmark_result += ((double)bufferSize / 1024) / avg_time;
				}


			}

			vkDestroyBuffer(vkGPU->device, buffer, NULL);
			vkFreeMemory(vkGPU->device, bufferDeviceMemory, NULL);
			vkDestroyBuffer(vkGPU->device, tempBuffer, NULL);
			vkFreeMemory(vkGPU->device, tempBufferDeviceMemory, NULL);
			deleteVulkanFFT(&app_forward);
			deleteVulkanFFT(&app_inverse);
		}
	}
	free(buffer_input);
	benchmark_result /= (num_benchmark_samples - 1);

	if (file_output) {
		fprintf(output, "Benchmark score VkFFT: %d\n", (int)(benchmark_result));
		fprintf(output, "Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
	}
	printf("Benchmark score VkFFT: %d\n", (int)(benchmark_result));
	printf("Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
	return res;
}
VkResult sample_5(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	VkResult res = VK_SUCCESS;
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
			VkFFTConfiguration forward_configuration = defaultVkFFTConfiguration;
			VkFFTConfiguration inverse_configuration = defaultVkFFTConfiguration;
			VkFFTApplication app_forward = defaultVkFFTApplication;
			VkFFTApplication app_inverse = defaultVkFFTApplication;
			//FFT + iFFT sample code.
			//Setting up FFT configuration for forward and inverse FFT.
			forward_configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).
			forward_configuration.size[0] = 4 * pow(2, n); //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			if (n == 0) forward_configuration.size[0] = 4096;
			forward_configuration.size[1] = 64 * 32 * pow(2, 16) / forward_configuration.size[0];
			if (forward_configuration.size[1] < 1) forward_configuration.size[1] = 1;
			//forward_configuration.size[1] = (forward_configuration.size[1] > 32768) ? 32768 : forward_configuration.size[1];
			forward_configuration.size[2] = 1;
			for (uint32_t i = 0; i < 3; i++) {
				forward_configuration.bufferStride[i] = forward_configuration.size[i];//can specify arbitrary buffer strides, if buffer is bigger than data - can be used for upscaling - you put data in the bigger buffer (2x in each dimension) in the corner, transform forward, zeropad to full buffer size and inverse
			}

			//PARAMETERS THAT CAN BE ADJUSTED FOR SPECIFIC GPU's - this configuration is by no means final form
			switch (vkGPU->physicalDeviceProperties.vendorID) {
			case 0x10DE://NVIDIA
				forward_configuration.coalescedMemory = 32;//the coalesced memory is equal to 32 bytes between L2 and VRAM. 
				forward_configuration.useLUT = false;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 4;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			case 0x8086://INTEL
				forward_configuration.coalescedMemory = 64;
				forward_configuration.useLUT = true;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 2;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			case 0x1002://AMD
				forward_configuration.coalescedMemory = 32;
				forward_configuration.useLUT = false;
				forward_configuration.warpSize = 64;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = (vkGPU->physicalDeviceProperties.limits.maxComputeSharedMemorySize >= 65536) ? 2 : 4;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 21;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			default:
				forward_configuration.coalescedMemory = 64;
				forward_configuration.useLUT = false;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 1;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			}
			forward_configuration.reorderFourStep = false;
			forward_configuration.performZeropadding[0] = false; //Perform padding with zeros on GPU. Still need to properly align input data (no need to fill padding area with meaningful data) but this will increase performance due to the lower amount of the memory reads/writes and omitting sequences only consisting of zeros.
			forward_configuration.performZeropadding[1] = false;
			forward_configuration.performZeropadding[2] = false;
			forward_configuration.performConvolution = false; //Perform convolution with precomputed kernel. 
			forward_configuration.performR2C = false; //Perform C2C transform. Can be combined with all other options. 
			forward_configuration.coordinateFeatures = 1; //Specify dimensionality of the input feature vector (default 1). Each component is stored not as a vector, but as a separate system and padded on it's own according to other options (i.e. for x*y system of 3-vector, first x*y elements correspond to the first dimension, then goes x*y for the second, etc). 
			forward_configuration.inverse = false; //Direction of FFT. false - forward, true - inverse.
			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [VkDeviceSize *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [VkDeviceSize *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			forward_configuration.device = &vkGPU->device;
			forward_configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
			forward_configuration.fence = &vkGPU->fence;
			forward_configuration.commandPool = &vkGPU->commandPool;
			forward_configuration.physicalDevice = &vkGPU->physicalDevice;
			forward_configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization

			//Allocate buffer for the input data.
			VkDeviceSize bufferSize = ((uint64_t)forward_configuration.coordinateFeatures) * sizeof(float) * 2 * forward_configuration.bufferStride[0] * forward_configuration.bufferStride[1] * forward_configuration.bufferStride[2];;
			VkBuffer buffer = {};
			VkDeviceMemory bufferDeviceMemory = {};
			allocateFFTBuffer(vkGPU, &buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);

			forward_configuration.buffer = &buffer;
			forward_configuration.isInputFormatted = false; //set to true if input is a different buffer, so it can have zeropadding/R2C added . Have to specifiy corresponding inputBufferStride
			forward_configuration.inputBuffer = &buffer; //you can specify first buffer to read data from to be different from the buffer FFT is performed on. FFT is still in-place on the second buffer, this is here just for convenience.
			forward_configuration.isOutputFormatted = false;//set to true if output is a different buffer, so it can have zeropadding/C2R automatically removed. Have to specifiy corresponding outputBufferStride
			forward_configuration.outputBuffer = &buffer;
			forward_configuration.bufferSize = &bufferSize;
			forward_configuration.tempBufferSize = &bufferSize;
			forward_configuration.inputBufferSize = &bufferSize;
			forward_configuration.outputBufferSize = &bufferSize;
			//Now we will create a similar configuration for inverse FFT and change inverse parameter to true.
			inverse_configuration = forward_configuration;
			inverse_configuration.inputBuffer = &buffer;//If you continue working with previous data, select the FFT buffer as initial
			inverse_configuration.outputBuffer = &buffer;
			inverse_configuration.inverse = true;

			//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
			/*float* buffer_input = (float*)malloc(bufferSize);

			for (uint32_t v = 0; v < forward_configuration.coordinateFeatures; v++) {
				for (uint32_t k = 0; k < forward_configuration.size[2]; k++) {
					for (uint32_t j = 0; j < forward_configuration.size[1]; j++) {
						for (uint32_t i = 0; i < forward_configuration.size[0]; i++) {
							buffer_input[2 * (i + j * forward_configuration.size[0] + k * (forward_configuration.size[0]) * forward_configuration.size[1] + v * (forward_configuration.size[0]) * forward_configuration.size[1] * forward_configuration.size[2])] = 2 * ((float)rand()) / RAND_MAX - 1.0;
							buffer_input[2 * (i + j * forward_configuration.size[0] + k * (forward_configuration.size[0]) * forward_configuration.size[1] + v * (forward_configuration.size[0]) * forward_configuration.size[1] * forward_configuration.size[2]) + 1] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						}
					}
				}
			}
			*/
			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
			transferDataFromCPU(vkGPU, buffer_input, &buffer, bufferSize);
			//free(buffer_input);

			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			res = initializeVulkanFFT(&app_forward, forward_configuration);
			if (res != VK_SUCCESS) return res;
			res = initializeVulkanFFT(&app_inverse, inverse_configuration);
			if (res != VK_SUCCESS) return res;
			//Submit FFT+iFFT.
			uint32_t batch = ((3 * 4096 * 1024.0 * 1024.0) / bufferSize > 1000) ? 1000 : (3 * 4096 * 1024.0 * 1024.0) / bufferSize;
			if (vkGPU->physicalDeviceProperties.vendorID == 0x8086) batch /= 4;
			if (batch == 0) batch = 1;
			float totTime = performVulkanFFTiFFT(vkGPU, &app_forward, &app_inverse, batch);

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
					for (uint32_t i = 0; i < forward_configuration.FFTdim; i++)
						num_tot_transfers += app_forward.localFFTPlan->numAxisUploads[i];
					num_tot_transfers *= 4;
					if (file_output)
						fprintf(output, "VkFFT System: %d %dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f batch: %d benchmark: %d bandwidth: %0.1f\n", (int)log2(forward_configuration.size[0]), forward_configuration.size[0], forward_configuration.size[1], bufferSize / 1024 / 1024, avg_time, std_error, batch, (int)(((double)bufferSize / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);

					printf("VkFFT System: %d %dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f batch: %d benchmark: %d bandwidth: %0.1f\n", (int)log2(forward_configuration.size[0]), forward_configuration.size[0], forward_configuration.size[1], bufferSize / 1024 / 1024, avg_time, std_error, batch, (int)(((double)bufferSize / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);
					benchmark_result += ((double)bufferSize / 1024) / avg_time;
				}


			}

			vkDestroyBuffer(vkGPU->device, buffer, NULL);
			vkFreeMemory(vkGPU->device, bufferDeviceMemory, NULL);
			deleteVulkanFFT(&app_forward);
			deleteVulkanFFT(&app_inverse);
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
VkResult sample_6(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	VkResult res = VK_SUCCESS;
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
			VkFFTConfiguration forward_configuration = defaultVkFFTConfiguration;
			VkFFTConfiguration inverse_configuration = defaultVkFFTConfiguration;
			VkFFTApplication app_forward;
			VkFFTApplication app_inverse;
			//FFT + iFFT sample code.
			//Setting up FFT configuration for forward and inverse FFT.
			switch (vkGPU->physicalDeviceProperties.vendorID) {
			case 0x10DE://NVIDIA
				forward_configuration.coalescedMemory = 32;
				forward_configuration.useLUT = false;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 1;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			case 0x8086://INTEL
				forward_configuration.coalescedMemory = 64;
				forward_configuration.useLUT = true;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 1;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			case 0x1002://AMD
				forward_configuration.coalescedMemory = 32;
				forward_configuration.useLUT = false;
				forward_configuration.warpSize = 64;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 1;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 19;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			default:
				forward_configuration.coalescedMemory = 64;
				forward_configuration.useLUT = false;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 1;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			}
			forward_configuration.FFTdim = benchmark_dimensions[n][3]; //FFT dimension, 1D, 2D or 3D (default 1).
			forward_configuration.size[0] = benchmark_dimensions[n][0]; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			forward_configuration.size[1] = benchmark_dimensions[n][1];
			forward_configuration.size[2] = benchmark_dimensions[n][2];
			for (uint32_t i = 0; i < 3; i++) {
				forward_configuration.bufferStride[i] = forward_configuration.size[i];//can specify arbitrary buffer strides, if buffer is bigger than data - can be used for upscaling - you put data in the bigger buffer (2x in each dimension) in the corner, transform forward, zeropad to full buffer size and inverse
			}
			forward_configuration.performZeropadding[0] = false; //Perform padding with zeros on GPU. Still need to properly align input data (no need to fill padding area with meaningful data) but this will increase performance due to the lower amount of the memory reads/writes and omitting sequences only consisting of zeros.
			forward_configuration.performZeropadding[1] = false;
			forward_configuration.performZeropadding[2] = false;
			forward_configuration.performConvolution = false; //Perform convolution with precomputed kernel. 
			forward_configuration.performR2C = true; //Perform R2C/C2R transform. Can be combined with all other options. Reduces memory requirements by a factor of 2. Requires special input data alignment: for x*y*z system pad x*y plane to (x+2)*y with last 2*y elements reserved, total array dimensions are (x*y+2y)*z. Memory layout after R2C and before C2R can be found on github.
			forward_configuration.coordinateFeatures = 1; //Specify dimensionality of the input feature vector (default 1). Each component is stored not as a vector, but as a separate system and padded on it's own according to other options (i.e. for x*y system of 3-vector, first x*y elements correspond to the first dimension, then goes x*y for the second, etc). 
			forward_configuration.inverse = false; //Direction of FFT. false - forward, true - inverse.
			forward_configuration.reorderFourStep = true;//set to true if you want data to return to correct layout after FFT. Set to false if you use convolution routine. Requires additional tempBuffer of bufferSize (see below) to do reordering
			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [VkDeviceSize *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [VkDeviceSize *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			forward_configuration.device = &vkGPU->device;
			forward_configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
			forward_configuration.fence = &vkGPU->fence;
			forward_configuration.commandPool = &vkGPU->commandPool;
			forward_configuration.physicalDevice = &vkGPU->physicalDevice;
			forward_configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization



			//Allocate buffer for the input data.
			VkDeviceSize bufferSize = ((uint64_t)forward_configuration.coordinateFeatures) * sizeof(float) * 2 * (forward_configuration.bufferStride[0] / 2 + 1) * forward_configuration.bufferStride[1] * forward_configuration.bufferStride[2];;
			VkBuffer buffer = {};
			VkDeviceMemory bufferDeviceMemory = {};
			VkBuffer tempBuffer = {};
			VkDeviceMemory tempBufferDeviceMemory = {};
			allocateFFTBuffer(vkGPU, &buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
			allocateFFTBuffer(vkGPU, &tempBuffer, &tempBufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);

			forward_configuration.isInputFormatted = false; //set to true if input is a different buffer, so it can have zeropadding/R2C added . Have to specifiy corresponding inputBufferStride
			forward_configuration.isOutputFormatted = false;//set to true if output is a different buffer, so it can have zeropadding/C2R automatically removed. Have to specifiy corresponding outputBufferStride

			forward_configuration.bufferNum = 1;
			forward_configuration.tempBufferNum = 1;
			forward_configuration.inputBufferNum = 1;
			forward_configuration.outputBufferNum = 1;

			forward_configuration.buffer = &buffer;
			forward_configuration.tempBuffer = &tempBuffer;
			forward_configuration.inputBuffer = &buffer; //you can specify first buffer to read data from to be different from the buffer FFT is performed on. FFT is still in-place on the second buffer, this is here just for convenience.
			forward_configuration.outputBuffer = &buffer;

			forward_configuration.bufferSize = &bufferSize;
			forward_configuration.tempBufferSize = &bufferSize;
			forward_configuration.inputBufferSize = &bufferSize;
			forward_configuration.outputBufferSize = &bufferSize;

			//Now we will create a similar configuration for inverse FFT and change inverse parameter to true.
			inverse_configuration = forward_configuration;
			inverse_configuration.inputBuffer = &buffer;//If you continue working with previous data, select the FFT buffer as initial
			inverse_configuration.outputBuffer = &buffer;
			inverse_configuration.inverse = true;


			//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
			/*float* buffer_input = (float*)malloc(bufferSize);

			for (uint32_t v = 0; v < forward_configuration.coordinateFeatures; v++) {
				for (uint32_t k = 0; k < forward_configuration.size[2]; k++) {
					for (uint32_t j = 0; j < forward_configuration.size[1]; j++) {
						for (uint32_t i = 0; i < forward_configuration.size[0]; i++) {
							buffer_input[i + j * forward_configuration.size[0] + k * (forward_configuration.size[0] + 2) * forward_configuration.size[1] + v * (forward_configuration.size[0] + 2) * forward_configuration.size[1] * forward_configuration.size[2]] = i;//[-1,1]
						}
					}
				}
			}*/
			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
			transferDataFromCPU(vkGPU, buffer_input, &buffer, bufferSize);
			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			res = initializeVulkanFFT(&app_forward, forward_configuration);
			if (res != VK_SUCCESS) return res;
			res = initializeVulkanFFT(&app_inverse, inverse_configuration);
			if (res != VK_SUCCESS) return res;
			//Submit FFT+iFFT.
			uint32_t batch = ((4096.0 * 1024.0 * 1024.0) / bufferSize > 1000) ? 1000 : (4096.0 * 1024.0 * 1024.0) / bufferSize;
			if (vkGPU->physicalDeviceProperties.vendorID == 0x8086) batch /= 4;
			if (batch == 0) batch = 1;

			//batch *= 5; //makes result more smooth, takes longer time
			float totTime = performVulkanFFTiFFT(vkGPU, &app_forward, &app_inverse, batch);
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
					for (uint32_t i = 0; i < forward_configuration.FFTdim; i++)
						num_tot_transfers += app_forward.localFFTPlan->numAxisUploads[i];
					num_tot_transfers *= 4;
					if (file_output)
						fprintf(output, "VkFFT System: %dx%dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f batch: %d benchmark: %d bandwidth: %0.1f\n", benchmark_dimensions[n][0], benchmark_dimensions[n][1], benchmark_dimensions[n][2], bufferSize / 1024 / 1024, avg_time, std_error, batch, (int)(((double)bufferSize / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);
					printf("VkFFT System: %dx%dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f batch: %d benchmark: %d bandwidth: %0.1f\n", benchmark_dimensions[n][0], benchmark_dimensions[n][1], benchmark_dimensions[n][2], bufferSize / 1024 / 1024, avg_time, std_error, batch, (int)(((double)bufferSize / 1024) / avg_time), bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);
					benchmark_result += ((double)bufferSize / 1024) / avg_time;
				}

			}
			//printf("Benchmark score: %f\n", ((double)bufferSize / 1024) / totTime);
			//Transfer data from GPU using staging buffer.
			//transferDataToCPU(buffer_output, &buffer, bufferSize);
			//Print data, if needed.
			/*for (uint32_t v = 0; v < inverse_configuration.coordinateFeatures; v++) {
				printf("\ncoordinate: %d\n\n", v);
				for (uint32_t k = 0; k < inverse_configuration.size[2]; k++) {
					for (uint32_t j = 0; j < inverse_configuration.size[1]; j++) {
						for (uint32_t i = 0; i < inverse_configuration.size[0]; i++) {
							printf("%.6f ", buffer_output[i + j * inverse_configuration.size[0] + k * (inverse_configuration.size[0] + 2) * inverse_configuration.size[1] + v * (inverse_configuration.size[0] + 2) * inverse_configuration.size[1] * inverse_configuration.size[2]]);
						}
						std::cout << "\n";
					}
				}
			}*/
			//free(buffer_input);
			//free(buffer_output);
			vkDestroyBuffer(vkGPU->device, buffer, NULL);
			vkFreeMemory(vkGPU->device, bufferDeviceMemory, NULL);
			vkDestroyBuffer(vkGPU->device, tempBuffer, NULL);
			vkFreeMemory(vkGPU->device, tempBufferDeviceMemory, NULL);
			deleteVulkanFFT(&app_forward);
			deleteVulkanFFT(&app_inverse);
		}
	}
	free(buffer_input);
	benchmark_result /= ((num_benchmark_samples - 1));
	if (file_output) {
		fprintf(output, "Benchmark score VkFFT: %d\n", (int)(benchmark_result));
		fprintf(output, "Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
	}
	printf("Benchmark score VkFFT: %d\n", (int)(benchmark_result));
	printf("Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
	return res;
}
VkResult sample_7(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	VkResult res = VK_SUCCESS;
	if (file_output)
		fprintf(output, "7 - VkFFT convolution example with identitiy kernel\n");
	printf("7 - VkFFT convolution example with identitiy kernel\n");
	//7 - convolution
	//Configuration + FFT application.
	VkFFTConfiguration forward_configuration = defaultVkFFTConfiguration;
	VkFFTConfiguration convolution_configuration = defaultVkFFTConfiguration;
	VkFFTApplication app_convolution;
	VkFFTApplication app_kernel;
	//Convolution sample code
	//Setting up FFT configuration. FFT is performed in-place with no performance loss. 
	switch (vkGPU->physicalDeviceProperties.vendorID) {
	case 0x10DE://NVIDIA
		forward_configuration.coalescedMemory = 32;
		forward_configuration.useLUT = false;
		forward_configuration.warpSize = 32;
		forward_configuration.registerBoostNonPow2 = 0;
		forward_configuration.registerBoost = 1;
		forward_configuration.registerBoost4Step = 1;
		forward_configuration.swapTo3Stage4Step = 0;
		forward_configuration.performHalfBandwidthBoost = false;
		break;
	case 0x8086://INTEL
		forward_configuration.coalescedMemory = 64;
		forward_configuration.useLUT = true;
		forward_configuration.warpSize = 32;
		forward_configuration.registerBoostNonPow2 = 0;
		forward_configuration.registerBoost = 1;
		forward_configuration.registerBoost4Step = 1;
		forward_configuration.swapTo3Stage4Step = 0;
		forward_configuration.performHalfBandwidthBoost = false;
		break;
	case 0x1002://AMD
		forward_configuration.coalescedMemory = 32;
		forward_configuration.useLUT = false;
		forward_configuration.warpSize = 64;
		forward_configuration.registerBoostNonPow2 = 0;
		forward_configuration.registerBoost = 1;
		forward_configuration.registerBoost4Step = 1;
		forward_configuration.swapTo3Stage4Step = 21;
		forward_configuration.performHalfBandwidthBoost = false;
	default:
		forward_configuration.coalescedMemory = 64;
		forward_configuration.useLUT = false;
		forward_configuration.warpSize = 32;
		forward_configuration.registerBoostNonPow2 = 0;
		forward_configuration.registerBoost = 1;
		forward_configuration.registerBoost4Step = 1;
		forward_configuration.swapTo3Stage4Step = 0;
		forward_configuration.performHalfBandwidthBoost = false;
		break;
	}
	forward_configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).
	forward_configuration.size[0] = 1024 * 1024 * 8; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z. 
	forward_configuration.size[1] = 1;
	forward_configuration.size[2] = 1;
	for (uint32_t i = 0; i < 3; i++) {
		forward_configuration.bufferStride[i] = forward_configuration.size[i];//can specify arbitrary buffer strides, if buffer is bigger than data - can be used for upscaling - you put data in the bigger buffer (2x in each dimension) in the corner, transform forward, zeropad to full buffer size and inverse
	}
	forward_configuration.performConvolution = false; //Perform convolution with precomputed kernel. As we perform forward FFT to get the kernel, it is set to false.
	forward_configuration.performR2C = false; //Perform R2C/C2R transform. Can be combined with all other options. Reduces memory requirements by a factor of 2. Requires special input data alignment: for x*y*z system pad x*y plane to (x+2)*y with last 2*y elements reserved, total array dimensions are (x*y+2y)*z. Memory layout after R2C and before C2R can be found on github.
	forward_configuration.coordinateFeatures = 9; //Specify dimensionality of the input feature vector (default 1). Each component is stored not as a vector, but as a separate system and padded on it's own according to other options (i.e. for x*y system of 3-vector, first x*y elements correspond to the first dimension, then goes x*y for the second, etc).
	//coordinateFeatures number is an important constant for convolution. If we perform 1x1 convolution, it is equal to number of features, but matrixConvolution should be equal to 1. For matrix convolution, it must be equal to matrixConvolution parameter. If we perform 2x2 convolution, it is equal to 3 for symmetric kernel (stored as xx, xy, yy) and 4 for nonsymmetric (stored as xx, xy, yx, yy). Similarly, 6 (stored as xx, xy, xz, yy, yz, zz) and 9 (stored as xx, xy, xz, yx, yy, yz, zx, zy, zz) for 3x3 convolutions. 
	forward_configuration.inverse = false; //Direction of FFT. false - forward, true - inverse.
	forward_configuration.reorderFourStep = false;//Set to false if you use convolution routine. Data reordering is not needed - no additional buffer - less memory usage

	//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [VkDeviceSize *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [VkDeviceSize *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
	forward_configuration.device = &vkGPU->device;
	forward_configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
	forward_configuration.fence = &vkGPU->fence;
	forward_configuration.commandPool = &vkGPU->commandPool;
	forward_configuration.physicalDevice = &vkGPU->physicalDevice;
	forward_configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization

	//In this example, we perform a convolution for a real vectorfield (3vector) with a symmetric kernel (6 values). We use forward_configuration to initialize convolution kernel first from real data, then we create convolution_configuration for convolution. The buffer object from forward_configuration is passed to convolution_configuration as kernel object.
	//1. Kernel forward FFT.
	VkDeviceSize kernelSize = ((uint64_t)forward_configuration.coordinateFeatures) * sizeof(float) * 2 * (forward_configuration.size[0]) * forward_configuration.size[1] * forward_configuration.size[2];;
	VkBuffer kernel = {};
	VkDeviceMemory kernelDeviceMemory = {};

	//Sample allocation tool.
	allocateFFTBuffer(vkGPU, &kernel, &kernelDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, kernelSize);
	forward_configuration.buffer = &kernel;
	forward_configuration.inputBuffer = &kernel;
	forward_configuration.outputBuffer = &kernel;
	forward_configuration.bufferSize = &kernelSize;
	forward_configuration.inputBufferSize = &kernelSize;
	forward_configuration.outputBufferSize = &kernelSize;

	if (file_output)
		fprintf(output, "Total memory needed for kernel: %d MB\n", kernelSize / 1024 / 1024);
	printf("Total memory needed for kernel: %d MB\n", kernelSize / 1024 / 1024);
	//Fill kernel on CPU.
	float* kernel_input = (float*)malloc(kernelSize);
	for (uint32_t v = 0; v < forward_configuration.coordinateFeatures; v++) {
		for (uint32_t k = 0; k < forward_configuration.size[2]; k++) {
			for (uint32_t j = 0; j < forward_configuration.size[1]; j++) {

				//for (uint32_t i = 0; i < forward_configuration.size[0]; i++) {
				//	kernel_input[i + j * forward_configuration.size[0] + k * (forward_configuration.size[0] + 2) * forward_configuration.size[1] + v * (forward_configuration.size[0] + 2) * forward_configuration.size[1] * forward_configuration.size[2]] = 1;

				//Below is the test identity kernel for 3x3 nonsymmetric FFT
				for (uint32_t i = 0; i < forward_configuration.size[0]; i++) {
					if ((v == 0) || (v == 4) || (v == 8))

						kernel_input[2 * (i + j * (forward_configuration.size[0]) + k * (forward_configuration.size[0]) * forward_configuration.size[1] + v * (forward_configuration.size[0]) * forward_configuration.size[1] * forward_configuration.size[2])] = 1;

					else
						kernel_input[2 * (i + j * (forward_configuration.size[0]) + k * (forward_configuration.size[0]) * forward_configuration.size[1] + v * (forward_configuration.size[0]) * forward_configuration.size[1] * forward_configuration.size[2])] = 0;
					kernel_input[2 * (i + j * (forward_configuration.size[0]) + k * (forward_configuration.size[0]) * forward_configuration.size[1] + v * (forward_configuration.size[0]) * forward_configuration.size[1] * forward_configuration.size[2]) + 1] = 0;

				}
			}
		}
	}
	//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
	transferDataFromCPU(vkGPU, kernel_input, &kernel, kernelSize);
	//Initialize application responsible for the kernel. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
	res = initializeVulkanFFT(&app_kernel, forward_configuration);
	if (res != VK_SUCCESS) return res;
	//Sample forward FFT command buffer allocation + execution performed on kernel. Second number determines how many times perform application in one submit. FFT can also be appended to user defined command buffers.

	//Uncomment the line below if you want to perform kernel FFT. In this sample we use predefined identitiy kernel.
	//performVulkanFFT(vkGPU, &app_kernel, 1);

	//The kernel has been trasnformed.


	//2. Buffer convolution with transformed kernel.
	//Copy configuration, as it mostly remains unchanged. Change specific parts.
	convolution_configuration = forward_configuration;
	convolution_configuration.performConvolution = true;
	convolution_configuration.symmetricKernel = false;//Specify if convolution kernel is symmetric. In this case we only pass upper triangle part of it in the form of: (xx, xy, yy) for 2d and (xx, xy, xz, yy, yz, zz) for 3d.
	convolution_configuration.matrixConvolution = 3;//we do matrix convolution, so kernel is 9 numbers (3x3), but vector dimension is 3
	convolution_configuration.coordinateFeatures = 3;//equal to matrixConvolution size
	convolution_configuration.kernel = &kernel;
	convolution_configuration.kernelSize = &kernelSize;

	//Allocate separate buffer for the input data.
	VkDeviceSize bufferSize = ((uint64_t)convolution_configuration.coordinateFeatures) * sizeof(float) * 2 * (convolution_configuration.bufferStride[0]) * convolution_configuration.bufferStride[1] * convolution_configuration.bufferStride[2];;
	VkBuffer buffer = {};
	VkDeviceMemory bufferDeviceMemory = {};

	allocateFFTBuffer(vkGPU, &buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
	convolution_configuration.buffer = &buffer;
	convolution_configuration.isInputFormatted = false; //if input is a different buffer, it doesn't have to be zeropadded/R2C padded	
	convolution_configuration.inputBuffer = &buffer;
	convolution_configuration.isOutputFormatted = false;//if output is a different buffer, it can have zeropadding/C2R automatically removed
	convolution_configuration.outputBuffer = &buffer;
	convolution_configuration.bufferSize = &bufferSize;
	convolution_configuration.inputBufferSize = &kernelSize;
	convolution_configuration.outputBufferSize = &kernelSize;

	if (file_output)
		fprintf(output, "Total memory needed for buffer: %d MB\n", bufferSize / 1024 / 1024);
	printf("Total memory needed for buffer: %d MB\n", bufferSize / 1024 / 1024);
	//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
	float* buffer_input = (float*)malloc(bufferSize);

	for (uint32_t v = 0; v < convolution_configuration.coordinateFeatures; v++) {
		for (uint32_t k = 0; k < convolution_configuration.size[2]; k++) {
			for (uint32_t j = 0; j < convolution_configuration.size[1]; j++) {
				for (uint32_t i = 0; i < convolution_configuration.size[0]; i++) {
					buffer_input[2 * (i + j * convolution_configuration.size[0] + k * (convolution_configuration.bufferStride[0]) * convolution_configuration.bufferStride[1] + v * (convolution_configuration.bufferStride[0]) * convolution_configuration.bufferStride[1] * convolution_configuration.bufferStride[2])] = i % 8 - 3.5;
					buffer_input[2 * (i + j * convolution_configuration.size[0] + k * (convolution_configuration.bufferStride[0]) * convolution_configuration.bufferStride[1] + v * (convolution_configuration.bufferStride[0]) * convolution_configuration.bufferStride[1] * convolution_configuration.bufferStride[2]) + 1] = i % 4 - 1.5;
				}
			}
		}
	}
	//Transfer data to GPU using staging buffer.
	transferDataFromCPU(vkGPU, buffer_input, &buffer, bufferSize);

	//Initialize application responsible for the convolution.
	res = initializeVulkanFFT(&app_convolution, convolution_configuration);
	if (res != VK_SUCCESS) return res;
	//Sample forward FFT command buffer allocation + execution performed on kernel. FFT can also be appended to user defined command buffers.
	performVulkanFFT(vkGPU, &app_convolution, 1);
	//The kernel has been trasnformed.

	float* buffer_output = (float*)malloc(bufferSize);
	//Transfer data from GPU using staging buffer.
	transferDataToCPU(vkGPU, buffer_output, &buffer, bufferSize);

	//Print data, if needed.
	for (uint32_t v = 0; v < convolution_configuration.coordinateFeatures; v++) {
		if (file_output)
			fprintf(output, "\ncoordinate: %d\n\n", v);
		printf("\ncoordinate: %d\n\n", v);
		for (uint32_t k = 0; k < convolution_configuration.size[2]; k++) {
			for (uint32_t j = 0; j < convolution_configuration.size[1]; j++) {
				for (uint32_t i = 0; i < convolution_configuration.size[0]; i++) {
					if (file_output)
						fprintf(output, "%.6f %.6f ", buffer_output[2 * (i + j * convolution_configuration.bufferStride[0] + k * (convolution_configuration.bufferStride[0]) * convolution_configuration.bufferStride[1] + v * (convolution_configuration.bufferStride[0]) * convolution_configuration.bufferStride[1] * convolution_configuration.bufferStride[2])], buffer_output[2 * (i + j * convolution_configuration.bufferStride[0] + k * (convolution_configuration.bufferStride[0]) * convolution_configuration.bufferStride[1] + v * (convolution_configuration.bufferStride[0]) * convolution_configuration.bufferStride[1] * convolution_configuration.bufferStride[2]) + 1]);
					printf("%.6f %.6f ", buffer_output[2 * (i + j * convolution_configuration.bufferStride[0] + k * (convolution_configuration.bufferStride[0]) * convolution_configuration.bufferStride[1] + v * (convolution_configuration.bufferStride[0]) * convolution_configuration.bufferStride[1] * convolution_configuration.bufferStride[2])], buffer_output[2 * (i + j * convolution_configuration.bufferStride[0] + k * (convolution_configuration.bufferStride[0]) * convolution_configuration.bufferStride[1] + v * (convolution_configuration.bufferStride[0]) * convolution_configuration.bufferStride[1] * convolution_configuration.bufferStride[2]) + 1]);
				}
				std::cout << "\n";
			}
		}
	}
	free(kernel_input);
	free(buffer_input);
	free(buffer_output);
	vkDestroyBuffer(vkGPU->device, buffer, NULL);
	vkFreeMemory(vkGPU->device, bufferDeviceMemory, NULL);
	vkDestroyBuffer(vkGPU->device, kernel, NULL);
	vkFreeMemory(vkGPU->device, kernelDeviceMemory, NULL);
	deleteVulkanFFT(&app_kernel);
	deleteVulkanFFT(&app_convolution);
	return res;
}
VkResult sample_8(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	VkResult res = VK_SUCCESS;
	if (file_output)
		fprintf(output, "8 - VkFFT zeropadding convolution example with identitiy kernel\n");
	printf("8 - VkFFT zeropadding convolution example with identitiy kernel\n");
	//Configuration + FFT application.
	VkFFTConfiguration forward_configuration = defaultVkFFTConfiguration;
	VkFFTConfiguration convolution_configuration = defaultVkFFTConfiguration;
	VkFFTApplication app_convolution;
	VkFFTApplication app_kernel;
	//Zeropadding Convolution sample code
	//Setting up FFT configuration. FFT is performed in-place with no performance loss. 
	switch (vkGPU->physicalDeviceProperties.vendorID) {
	case 0x10DE://NVIDIA
		forward_configuration.coalescedMemory = 32;
		forward_configuration.useLUT = false;
		forward_configuration.warpSize = 32;
		forward_configuration.registerBoostNonPow2 = 0;
		forward_configuration.registerBoost = 1;
		forward_configuration.registerBoost4Step = 1;
		forward_configuration.swapTo3Stage4Step = 0;
		forward_configuration.performHalfBandwidthBoost = false;
		break;
	case 0x8086://INTEL - this sample most likely won't work due to the low register count
		forward_configuration.coalescedMemory = 64;
		forward_configuration.useLUT = true;
		forward_configuration.warpSize = 32;
		forward_configuration.registerBoostNonPow2 = 0;
		forward_configuration.registerBoost = 1;
		forward_configuration.registerBoost4Step = 1;
		forward_configuration.swapTo3Stage4Step = 0;
		forward_configuration.performHalfBandwidthBoost = false;
		break;
	case 0x1002://AMD
		forward_configuration.coalescedMemory = 32;
		forward_configuration.useLUT = false;
		forward_configuration.warpSize = 64;
		forward_configuration.registerBoostNonPow2 = 0;
		forward_configuration.registerBoost = 1;
		forward_configuration.registerBoost4Step = 1;
		forward_configuration.swapTo3Stage4Step = 21;
		forward_configuration.performHalfBandwidthBoost = false;
	default:
		forward_configuration.coalescedMemory = 64;
		forward_configuration.useLUT = false;
		forward_configuration.warpSize = 32;
		forward_configuration.registerBoostNonPow2 = 0;
		forward_configuration.registerBoost = 1;
		forward_configuration.registerBoost4Step = 1;
		forward_configuration.swapTo3Stage4Step = 0;
		forward_configuration.performHalfBandwidthBoost = false;
		break;
	}
	forward_configuration.FFTdim = 3; //FFT dimension, 1D, 2D or 3D (default 1).
	forward_configuration.size[0] = 32; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z. 
	forward_configuration.size[1] = 32;
	forward_configuration.size[2] = 32;
	for (uint32_t i = 0; i < 3; i++) {
		forward_configuration.bufferStride[i] = forward_configuration.size[i];//can specify arbitrary buffer strides, if buffer is bigger than data - can be used for upscaling - you put data in the bigger buffer (2x in each dimension) in the corner, transform forward, zeropad to full buffer size and inverse
	}
	forward_configuration.performZeropadding[0] = true; //Perform padding with zeros on GPU. Still need to properly align input data (no need to fill padding area with meaningful data) but this will increase performance due to the lower amount of the memory reads/writes and omitting sequences only consisting of zeros.
	forward_configuration.performZeropadding[1] = true;
	forward_configuration.performZeropadding[2] = true;
	forward_configuration.fft_zeropad_left[0] = ceil(forward_configuration.size[0] / 2.0);
	forward_configuration.fft_zeropad_right[0] = forward_configuration.size[0];
	forward_configuration.fft_zeropad_left[1] = ceil(forward_configuration.size[1] / 2.0);
	forward_configuration.fft_zeropad_right[1] = forward_configuration.size[1];
	forward_configuration.fft_zeropad_left[2] = ceil(forward_configuration.size[2] / 2.0);
	forward_configuration.fft_zeropad_right[2] = forward_configuration.size[2];
	forward_configuration.performConvolution = false; //Perform convolution with precomputed kernel. As we perform forward FFT to get the kernel, it is set to false.
	forward_configuration.performR2C = true; //Perform R2C/C2R transform. Can be combined with all other options. Reduces memory requirements by a factor of 2. Requires special input data alignment: for x*y*z system pad x*y plane to (x+2)*y with last 2*y elements reserved, total array dimensions are (x*y+2y)*z. Memory layout after R2C and before C2R can be found on github.
	forward_configuration.coordinateFeatures = 9; //Specify dimensionality of the input feature vector (default 1). Each component is stored not as a vector, but as a separate system and padded on it's own according to other options (i.e. for x*y system of 3-vector, first x*y elements correspond to the first dimension, then goes x*y for the second, etc).
	//coordinateFeatures number is an important constant for convolution. If we perform 1x1 convolution, it is equal to number of features, but matrixConvolution should be equal to 1. For matrix convolution, it must be equal to matrixConvolution parameter. If we perform 2x2 convolution, it is equal to 3 for symmetric kernel (stored as xx, xy, yy) and 4 for nonsymmetric (stored as xx, xy, yx, yy). Similarly, 6 (stored as xx, xy, xz, yy, yz, zz) and 9 (stored as xx, xy, xz, yx, yy, yz, zx, zy, zz) for 3x3 convolutions. 
	forward_configuration.inverse = false; //Direction of FFT. false - forward, true - inverse.
	forward_configuration.reorderFourStep = false;//Set to false if you use convolution routine. Data reordering is not needed - no additional buffer - less memory usage

	//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [VkDeviceSize *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [VkDeviceSize *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
	forward_configuration.device = &vkGPU->device;
	forward_configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
	forward_configuration.fence = &vkGPU->fence;
	forward_configuration.commandPool = &vkGPU->commandPool;
	forward_configuration.physicalDevice = &vkGPU->physicalDevice;
	forward_configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization

	//In this example, we perform a convolution for a real vectorfield (3vector) with a symmetric kernel (6 values). We use forward_configuration to initialize convolution kernel first from real data, then we create convolution_configuration for convolution. The buffer object from forward_configuration is passed to convolution_configuration as kernel object.
	//1. Kernel forward FFT.
	VkDeviceSize kernelSize = ((uint64_t)forward_configuration.coordinateFeatures) * sizeof(float) * 2 * (forward_configuration.bufferStride[0] / 2 + 1) * forward_configuration.bufferStride[1] * forward_configuration.bufferStride[2];
	VkBuffer kernel = {};
	VkDeviceMemory kernelDeviceMemory = {};

	//Sample allocation tool.
	allocateFFTBuffer(vkGPU, &kernel, &kernelDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, kernelSize);
	forward_configuration.buffer = &kernel;
	forward_configuration.inputBuffer = &kernel;
	forward_configuration.outputBuffer = &kernel;
	forward_configuration.bufferSize = &kernelSize;
	forward_configuration.inputBufferSize = &kernelSize;
	forward_configuration.outputBufferSize = &kernelSize;
	if (file_output)
		fprintf(output, "Total memory needed for kernel: %d MB\n", kernelSize / 1024 / 1024);
	printf("Total memory needed for kernel: %d MB\n", kernelSize / 1024 / 1024);

	//Fill kernel on CPU.
	float* kernel_input = (float*)malloc(kernelSize);
	for (uint32_t v = 0; v < forward_configuration.coordinateFeatures; v++) {
		for (uint32_t k = 0; k < forward_configuration.size[2]; k++) {
			for (uint32_t j = 0; j < forward_configuration.size[1]; j++) {

				//for (uint32_t i = 0; i < forward_configuration.size[0]; i++) {
				//	kernel_input[i + j * forward_configuration.size[0] + k * (forward_configuration.size[0] + 2) * forward_configuration.size[1] + v * (forward_configuration.size[0] + 2) * forward_configuration.size[1] * forward_configuration.size[2]] = 1;

				//Below is the test identity kernel for 3x3 nonsymmetric FFT
				for (uint32_t i = 0; i < forward_configuration.size[0] / 2 + 1; i++) {
					if ((v == 0) || (v == 4) || (v == 8))

						kernel_input[2 * i + j * (forward_configuration.bufferStride[0] + 2) + k * (forward_configuration.bufferStride[0] + 2) * forward_configuration.bufferStride[1] + v * (forward_configuration.bufferStride[0] + 2) * forward_configuration.bufferStride[1] * forward_configuration.bufferStride[2]] = 1;

					else
						kernel_input[2 * i + j * (forward_configuration.bufferStride[0] + 2) + k * (forward_configuration.bufferStride[0] + 2) * forward_configuration.bufferStride[1] + v * (forward_configuration.bufferStride[0] + 2) * forward_configuration.bufferStride[1] * forward_configuration.bufferStride[2]] = 0;
					kernel_input[2 * i + 1 + j * (forward_configuration.bufferStride[0] + 2) + k * (forward_configuration.bufferStride[0] + 2) * forward_configuration.bufferStride[1] + v * (forward_configuration.bufferStride[0] + 2) * forward_configuration.bufferStride[1] * forward_configuration.bufferStride[2]] = 0;

				}
			}
		}
	}
	//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
	transferDataFromCPU(vkGPU, kernel_input, &kernel, kernelSize);
	//Initialize application responsible for the kernel. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
	res = initializeVulkanFFT(&app_kernel, forward_configuration);
	if (res != VK_SUCCESS) return res;
	//Sample forward FFT command buffer allocation + execution performed on kernel. Second number determines how many times perform application in one submit. FFT can also be appended to user defined command buffers.

	//Uncomment the line below if you want to perform kernel FFT. In this sample we use predefined identitiy kernel.
	//performVulkanFFT(vkGPU, &app_kernel, 1);

	//The kernel has been trasnformed.


	//2. Buffer convolution with transformed kernel.
	//Copy configuration, as it mostly remains unchanged. Change specific parts.
	convolution_configuration = forward_configuration;
	convolution_configuration.performConvolution = true;
	convolution_configuration.symmetricKernel = false;//Specify if convolution kernel is symmetric. In this case we only pass upper triangle part of it in the form of: (xx, xy, yy) for 2d and (xx, xy, xz, yy, yz, zz) for 3d.
	convolution_configuration.matrixConvolution = 3; //we do matrix convolution, so kernel is 9 numbers (3x3), but vector dimension is 3
	convolution_configuration.coordinateFeatures = 3;
	convolution_configuration.kernel = &kernel;
	convolution_configuration.kernelSize = &kernelSize;

	//Allocate separate buffer for the input data.
	VkDeviceSize bufferSize = ((uint64_t)convolution_configuration.coordinateFeatures) * sizeof(float) * 2 * (convolution_configuration.bufferStride[0] / 2 + 1) * convolution_configuration.bufferStride[1] * convolution_configuration.bufferStride[2];;
	VkBuffer buffer = {};
	VkDeviceMemory bufferDeviceMemory = {};

	allocateFFTBuffer(vkGPU, &buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
	convolution_configuration.buffer = &buffer;
	convolution_configuration.isInputFormatted = false; //if input is a different buffer, it doesn't have to be zeropadded/R2C padded	
	convolution_configuration.inputBuffer = &buffer;
	convolution_configuration.isOutputFormatted = false;//if output is a different buffer, it can have zeropadding/C2R automatically removed
	convolution_configuration.outputBuffer = &buffer;
	convolution_configuration.bufferSize = &bufferSize;
	convolution_configuration.inputBufferSize = &bufferSize;
	convolution_configuration.outputBufferSize = &bufferSize;
	if (file_output)
		fprintf(output, "Total memory needed for buffer: %d MB\n", bufferSize / 1024 / 1024);
	printf("Total memory needed for buffer: %d MB\n", bufferSize / 1024 / 1024);
	//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
	float* buffer_input = (float*)malloc(bufferSize);

	for (uint32_t v = 0; v < convolution_configuration.coordinateFeatures; v++) {
		for (uint32_t k = 0; k < ceil(convolution_configuration.size[2] / 2.0); k++) {
			for (uint32_t j = 0; j < convolution_configuration.size[1] / 2; j++) {
				for (uint32_t i = 0; i < convolution_configuration.size[0] / 2; i++) {
					buffer_input[i + j * convolution_configuration.bufferStride[0] + k * (convolution_configuration.bufferStride[0] + 2) * convolution_configuration.bufferStride[1] + v * (convolution_configuration.bufferStride[0] + 2) * convolution_configuration.bufferStride[1] * convolution_configuration.bufferStride[2]] = i;
				}
			}
		}
	}
	//Transfer data to GPU using staging buffer.
	transferDataFromCPU(vkGPU, buffer_input, &buffer, bufferSize);

	//Initialize application responsible for the convolution.
	res = initializeVulkanFFT(&app_convolution, convolution_configuration);
	if (res != VK_SUCCESS) return res;
	//Sample forward FFT command buffer allocation + execution performed on kernel. FFT can also be appended to user defined command buffers.
	performVulkanFFT(vkGPU, &app_convolution, 1);
	//The kernel has been trasnformed.

	float* buffer_output = (float*)malloc(bufferSize);
	//Transfer data from GPU using staging buffer.
	transferDataToCPU(vkGPU, buffer_output, &buffer, bufferSize);

	//Print data, if needed.
	for (uint32_t v = 0; v < convolution_configuration.coordinateFeatures; v++) {
		if (file_output)
			fprintf(output, "\ncoordinate: %d\n\n", v);
		printf("\ncoordinate: %d\n\n", v);
		for (uint32_t k = 0; k < ceil(convolution_configuration.size[2] / 2.0); k++) {
			for (uint32_t j = 0; j < convolution_configuration.size[1] / 2; j++) {
				for (uint32_t i = 0; i < convolution_configuration.size[0] / 2; i++) {
					if (file_output)
						fprintf(output, "%.6f ", buffer_output[i + j * convolution_configuration.bufferStride[0] + k * (convolution_configuration.bufferStride[0] + 2) * convolution_configuration.bufferStride[1] + v * (convolution_configuration.bufferStride[0] + 2) * convolution_configuration.bufferStride[1] * convolution_configuration.bufferStride[2]]);
					printf("%.6f ", buffer_output[i + j * convolution_configuration.bufferStride[0] + k * (convolution_configuration.bufferStride[0] + 2) * convolution_configuration.bufferStride[1] + v * (convolution_configuration.bufferStride[0] + 2) * convolution_configuration.bufferStride[1] * convolution_configuration.bufferStride[2]]);
				}
				std::cout << "\n";
			}
		}
	}
	free(kernel_input);
	free(buffer_input);
	free(buffer_output);
	vkDestroyBuffer(vkGPU->device, buffer, NULL);
	vkFreeMemory(vkGPU->device, bufferDeviceMemory, NULL);
	vkDestroyBuffer(vkGPU->device, kernel, NULL);
	vkFreeMemory(vkGPU->device, kernelDeviceMemory, NULL);
	deleteVulkanFFT(&app_kernel);
	deleteVulkanFFT(&app_convolution);
	return res;
}
VkResult sample_9(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	VkResult res = VK_SUCCESS;
	if (file_output)
		fprintf(output, "9 - VkFFT batched convolution example with identitiy kernel\n");
	printf("9 - VkFFT batched convolution example with identitiy kernel\n");
	//Configuration + FFT application.
	VkFFTConfiguration forward_configuration = defaultVkFFTConfiguration;
	VkFFTConfiguration convolution_configuration = defaultVkFFTConfiguration;
	VkFFTApplication app_convolution;
	VkFFTApplication app_kernel;
	//Convolution sample code
	//Setting up FFT configuration. FFT is performed in-place with no performance loss. 
	switch (vkGPU->physicalDeviceProperties.vendorID) {
	case 0x10DE://NVIDIA
		forward_configuration.coalescedMemory = 32;
		forward_configuration.useLUT = false;
		forward_configuration.warpSize = 32;
		forward_configuration.registerBoostNonPow2 = 0;
		forward_configuration.registerBoost = 1;
		forward_configuration.registerBoost4Step = 1;
		forward_configuration.swapTo3Stage4Step = 0;
		forward_configuration.performHalfBandwidthBoost = false;
		break;
	case 0x8086://INTEL
		forward_configuration.coalescedMemory = 64;
		forward_configuration.useLUT = true;
		forward_configuration.warpSize = 32;
		forward_configuration.registerBoostNonPow2 = 0;
		forward_configuration.registerBoost = 1;
		forward_configuration.registerBoost4Step = 1;
		forward_configuration.swapTo3Stage4Step = 0;
		forward_configuration.performHalfBandwidthBoost = false;
		break;
	case 0x1002://AMD
		forward_configuration.coalescedMemory = 32;
		forward_configuration.useLUT = false;
		forward_configuration.warpSize = 64;
		forward_configuration.registerBoostNonPow2 = 0;
		forward_configuration.registerBoost = 1;
		forward_configuration.registerBoost4Step = 1;
		forward_configuration.swapTo3Stage4Step = 21;
		forward_configuration.performHalfBandwidthBoost = false;
	default:
		forward_configuration.coalescedMemory = 64;
		forward_configuration.useLUT = false;
		forward_configuration.warpSize = 32;
		forward_configuration.registerBoostNonPow2 = 0;
		forward_configuration.registerBoost = 1;
		forward_configuration.registerBoost4Step = 1;
		forward_configuration.swapTo3Stage4Step = 0;
		forward_configuration.performHalfBandwidthBoost = false;
		break;
	}
	forward_configuration.FFTdim = 2; //FFT dimension, 1D, 2D or 3D (default 1).
	forward_configuration.size[0] = 32; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z. 
	forward_configuration.size[1] = 32;
	forward_configuration.size[2] = 1;
	for (uint32_t i = 0; i < 3; i++) {
		forward_configuration.bufferStride[i] = forward_configuration.size[i];//can specify arbitrary buffer strides, if buffer is bigger than data - can be used for upscaling - you put data in the bigger buffer (2x in each dimension) in the corner, transform forward, zeropad to full buffer size and inverse
	}
	forward_configuration.performConvolution = false; //Perform convolution with precomputed kernel. As we perform forward FFT to get the kernel, it is set to false.
	forward_configuration.performR2C = true; //Perform R2C/C2R transform. Can be combined with all other options. Reduces memory requirements by a factor of 2. Requires special input data alignment: for x*y*z system pad x*y plane to (x+2)*y with last 2*y elements reserved, total array dimensions are (x*y+2y)*z. Memory layout after R2C and before C2R can be found on github.
	forward_configuration.coordinateFeatures = 2; //Specify dimensionality of the input feature vector (default 1). Each component is stored not as a vector, but as a separate system and padded on it's own according to other options (i.e. for x*y system of 3-vector, first x*y elements correspond to the first dimension, then goes x*y for the second, etc).
	//coordinateFeatures number is an important constant for convolution. If we perform 1x1 convolution, it is equal to number of features, but matrixConvolution should be equal to 1. For matrix convolution, it must be equal to matrixConvolution parameter. If we perform 2x2 convolution, it is equal to 3 for symmetric kernel (stored as xx, xy, yy) and 4 for nonsymmetric (stored as xx, xy, yx, yy). Similarly, 6 (stored as xx, xy, xz, yy, yz, zz) and 9 (stored as xx, xy, xz, yx, yy, yz, zx, zy, zz) for 3x3 convolutions. 
	forward_configuration.inverse = false; //Direction of FFT. false - forward, true - inverse.
	forward_configuration.reorderFourStep = false;//Set to false if you use convolution routine. Data reordering is not needed - no additional buffer - less memory usage

	forward_configuration.numberBatches = 2;
	//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [VkDeviceSize *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [VkDeviceSize *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
	forward_configuration.device = &vkGPU->device;
	forward_configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
	forward_configuration.fence = &vkGPU->fence;
	forward_configuration.commandPool = &vkGPU->commandPool;
	forward_configuration.physicalDevice = &vkGPU->physicalDevice;
	forward_configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization

	//In this example, we perform a convolution for a real vectorfield (3vector) with a symmetric kernel (6 values). We use forward_configuration to initialize convolution kernel first from real data, then we create convolution_configuration for convolution. The buffer object from forward_configuration is passed to convolution_configuration as kernel object.
	//1. Kernel forward FFT.
	VkDeviceSize kernelSize = ((uint64_t)forward_configuration.numberBatches) * forward_configuration.coordinateFeatures * sizeof(float) * 2 * (forward_configuration.bufferStride[0] / 2 + 1) * forward_configuration.bufferStride[1] * forward_configuration.bufferStride[2];;
	VkBuffer kernel = {};
	VkDeviceMemory kernelDeviceMemory = {};

	//Sample allocation tool.
	allocateFFTBuffer(vkGPU, &kernel, &kernelDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, kernelSize);
	forward_configuration.buffer = &kernel;
	forward_configuration.inputBuffer = &kernel;
	forward_configuration.outputBuffer = &kernel;
	forward_configuration.bufferSize = &kernelSize;
	forward_configuration.inputBufferSize = &kernelSize;
	forward_configuration.outputBufferSize = &kernelSize;
	if (file_output)
		fprintf(output, "Total memory needed for kernel: %d MB\n", kernelSize / 1024 / 1024);
	printf("Total memory needed for kernel: %d MB\n", kernelSize / 1024 / 1024);

	//Fill kernel on CPU.
	float* kernel_input = (float*)malloc(kernelSize);
	for (uint32_t f = 0; f < forward_configuration.numberBatches; f++) {
		for (uint32_t v = 0; v < forward_configuration.coordinateFeatures; v++) {
			for (uint32_t k = 0; k < forward_configuration.size[2]; k++) {
				for (uint32_t j = 0; j < forward_configuration.size[1]; j++) {

					//Below is the test identity kernel for 1x1 nonsymmetric FFT, multiplied by (f * forward_configuration.coordinateFeatures + v + 1);
					for (uint32_t i = 0; i < forward_configuration.size[0] / 2 + 1; i++) {

						kernel_input[2 * i + j * (forward_configuration.bufferStride[0] + 2) + k * (forward_configuration.bufferStride[0] + 2) * forward_configuration.bufferStride[1] + v * (forward_configuration.bufferStride[0] + 2) * forward_configuration.bufferStride[1] * forward_configuration.bufferStride[2] + f * forward_configuration.coordinateFeatures * (forward_configuration.bufferStride[0] + 2) * forward_configuration.bufferStride[1] * forward_configuration.bufferStride[2]] = f * forward_configuration.coordinateFeatures + v + 1;
						kernel_input[2 * i + 1 + j * (forward_configuration.bufferStride[0] + 2) + k * (forward_configuration.bufferStride[0] + 2) * forward_configuration.bufferStride[1] + v * (forward_configuration.bufferStride[0] + 2) * forward_configuration.bufferStride[1] * forward_configuration.bufferStride[2] + f * forward_configuration.coordinateFeatures * (forward_configuration.bufferStride[0] + 2) * forward_configuration.bufferStride[1] * forward_configuration.bufferStride[2]] = 0;

					}
				}
			}
		}
	}
	//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
	transferDataFromCPU(vkGPU, kernel_input, &kernel, kernelSize);
	//Initialize application responsible for the kernel. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
	res = initializeVulkanFFT(&app_kernel, forward_configuration);
	if (res != VK_SUCCESS) return res;
	//Sample forward FFT command buffer allocation + execution performed on kernel. Second number determines how many times perform application in one submit. FFT can also be appended to user defined command buffers.

	//Uncomment the line below if you want to perform kernel FFT. In this sample we use predefined identitiy kernel.
	//performVulkanFFT(vkGPU, &app_kernel, 1);

	//The kernel has been trasnformed.


	//2. Buffer convolution with transformed kernel.
	//Copy configuration, as it mostly remains unchanged. Change specific parts.
	convolution_configuration = forward_configuration;
	convolution_configuration.performConvolution = true;
	convolution_configuration.symmetricKernel = false;//Specify if convolution kernel is symmetric. In this case we only pass upper triangle part of it in the form of: (xx, xy, yy) for 2d and (xx, xy, xz, yy, yz, zz) for 3d.
	convolution_configuration.kernel = &kernel;
	convolution_configuration.kernelSize = &kernelSize;
	convolution_configuration.numberBatches = 1;//one batch - numberKernels convolutions
	convolution_configuration.numberKernels = forward_configuration.numberBatches;// number of convolutions on a single input
	//Allocate separate buffer for the input data.
	VkDeviceSize bufferSize = ((uint64_t)convolution_configuration.coordinateFeatures) * sizeof(float) * 2 * (convolution_configuration.bufferStride[0] / 2 + 1) * convolution_configuration.bufferStride[1] * convolution_configuration.bufferStride[2];;
	VkBuffer buffer = {};
	VkDeviceMemory bufferDeviceMemory = {};

	allocateFFTBuffer(vkGPU, &buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
	VkDeviceSize outputBufferSize = convolution_configuration.numberKernels * convolution_configuration.coordinateFeatures * sizeof(float) * 2 * (convolution_configuration.bufferStride[0] / 2 + 1) * convolution_configuration.bufferStride[1] * convolution_configuration.bufferStride[2];;
	VkBuffer outputBuffer = {};
	VkDeviceMemory outputBufferDeviceMemory = {};

	allocateFFTBuffer(vkGPU, &outputBuffer, &outputBufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, outputBufferSize);

	convolution_configuration.buffer = &outputBuffer;
	convolution_configuration.isInputFormatted = true; //if input is a different buffer, it doesn't have to be zeropadded/R2C padded	
	convolution_configuration.inputBuffer = &buffer;
	convolution_configuration.inputBufferSize = &bufferSize;
	convolution_configuration.isOutputFormatted = false;//if output is a different buffer, it can have zeropadding/C2R automatically removed
	convolution_configuration.outputBuffer = &outputBuffer;
	convolution_configuration.bufferSize = &outputBufferSize;

	convolution_configuration.outputBufferSize = &outputBufferSize;
	if (file_output)
		fprintf(output, "Total memory needed for buffer: %d MB\n", bufferSize / 1024 / 1024);
	printf("Total memory needed for buffer: %d MB\n", bufferSize / 1024 / 1024);
	//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
	float* buffer_input = (float*)malloc(bufferSize);

	for (uint32_t v = 0; v < convolution_configuration.coordinateFeatures; v++) {
		for (uint32_t k = 0; k < convolution_configuration.size[2]; k++) {
			for (uint32_t j = 0; j < convolution_configuration.size[1]; j++) {
				for (uint32_t i = 0; i < convolution_configuration.size[0]; i++) {
					buffer_input[i + j * convolution_configuration.bufferStride[0] + k * (convolution_configuration.bufferStride[0] + 2) * convolution_configuration.bufferStride[1] + v * (convolution_configuration.bufferStride[0] + 2) * convolution_configuration.bufferStride[1] * convolution_configuration.bufferStride[2]] = 1;
				}
			}
		}
	}
	//Transfer data to GPU using staging buffer.
	transferDataFromCPU(vkGPU, buffer_input, &buffer, bufferSize);

	//Initialize application responsible for the convolution.
	res = initializeVulkanFFT(&app_convolution, convolution_configuration);
	if (res != VK_SUCCESS) return res;
	//Sample forward FFT command buffer allocation + execution performed on kernel. FFT can also be appended to user defined command buffers.
	performVulkanFFT(vkGPU, &app_convolution, 1);
	//The kernel has been trasnformed.

	float* buffer_output = (float*)malloc(outputBufferSize);
	//Transfer data from GPU using staging buffer.
	transferDataToCPU(vkGPU, buffer_output, &outputBuffer, outputBufferSize);

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
							fprintf(output, "%.6f ", buffer_output[i + j * convolution_configuration.bufferStride[0] + k * (convolution_configuration.bufferStride[0] + 2) * convolution_configuration.bufferStride[1] + v * (convolution_configuration.bufferStride[0] + 2) * convolution_configuration.bufferStride[1] * convolution_configuration.bufferStride[2] + f * convolution_configuration.coordinateFeatures * (convolution_configuration.bufferStride[0] + 2) * convolution_configuration.bufferStride[1] * convolution_configuration.bufferStride[2]]);

						printf("%.6f ", buffer_output[i + j * convolution_configuration.bufferStride[0] + k * (convolution_configuration.bufferStride[0] + 2) * convolution_configuration.bufferStride[1] + v * (convolution_configuration.bufferStride[0] + 2) * convolution_configuration.bufferStride[1] * convolution_configuration.bufferStride[2] + f * convolution_configuration.coordinateFeatures * (convolution_configuration.bufferStride[0] + 2) * convolution_configuration.bufferStride[1] * convolution_configuration.bufferStride[2]]);
					}
					std::cout << "\n";
				}
			}
		}
	}
	free(kernel_input);
	free(buffer_input);
	free(buffer_output);
	vkDestroyBuffer(vkGPU->device, buffer, NULL);
	vkFreeMemory(vkGPU->device, bufferDeviceMemory, NULL);
	vkDestroyBuffer(vkGPU->device, outputBuffer, NULL);
	vkFreeMemory(vkGPU->device, outputBufferDeviceMemory, NULL);
	vkDestroyBuffer(vkGPU->device, kernel, NULL);
	vkFreeMemory(vkGPU->device, kernelDeviceMemory, NULL);
	deleteVulkanFFT(&app_kernel);
	deleteVulkanFFT(&app_convolution);
	return res;
}
VkResult sample_10(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	VkResult res = VK_SUCCESS;
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
			VkFFTConfiguration forward_configuration = defaultVkFFTConfiguration;
			VkFFTConfiguration inverse_configuration = defaultVkFFTConfiguration;
			VkFFTApplication app_forward = defaultVkFFTApplication;
			VkFFTApplication app_inverse = defaultVkFFTApplication;
			//FFT + iFFT sample code.
			//Setting up FFT configuration for forward and inverse FFT.
			forward_configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).
			forward_configuration.size[0] = 4 * pow(2, n); //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			if (n == 0) forward_configuration.size[0] = 4096;
			forward_configuration.size[1] = 64 * 32 * pow(2, 16) / forward_configuration.size[0];
			if (forward_configuration.size[1] < 1) forward_configuration.size[1] = 1;
			//forward_configuration.size[1] = (forward_configuration.size[1] > 32768) ? 32768 : forward_configuration.size[1];
			forward_configuration.size[2] = 1;
			for (uint32_t i = 0; i < 3; i++) {
				forward_configuration.bufferStride[i] = forward_configuration.size[i];//can specify arbitrary buffer strides, if buffer is bigger than data - can be used for upscaling - you put data in the bigger buffer (2x in each dimension) in the corner, transform forward, zeropad to full buffer size and inverse
			}
			uint32_t numBuf = 4;
			//PARAMETERS THAT CAN BE ADJUSTED FOR SPECIFIC GPU's - this configuration is by no means final form
			switch (vkGPU->physicalDeviceProperties.vendorID) {
			case 0x10DE://NVIDIA
				forward_configuration.coalescedMemory = 32;//the coalesced memory is equal to 32 bytes between L2 and VRAM. 
				forward_configuration.useLUT = false;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 4;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			case 0x8086://INTEL
				forward_configuration.coalescedMemory = 64;
				forward_configuration.useLUT = true;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 2;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			case 0x1002://AMD
				forward_configuration.coalescedMemory = 32;
				forward_configuration.useLUT = false;
				forward_configuration.warpSize = 64;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = (vkGPU->physicalDeviceProperties.limits.maxComputeSharedMemorySize >= 65536) ? 2 : 4;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 19;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			default:
				forward_configuration.coalescedMemory = 64;
				forward_configuration.useLUT = false;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 1;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			}
			forward_configuration.reorderFourStep = true;
			forward_configuration.performZeropadding[0] = false; //Perform padding with zeros on GPU. Still need to properly align input data (no need to fill padding area with meaningful data) but this will increase performance due to the lower amount of the memory reads/writes and omitting sequences only consisting of zeros.
			forward_configuration.performZeropadding[1] = false;
			forward_configuration.performZeropadding[2] = false;
			forward_configuration.performConvolution = false; //Perform convolution with precomputed kernel. 
			forward_configuration.performR2C = false; //Perform C2C transform. Can be combined with all other options. 
			forward_configuration.coordinateFeatures = 1; //Specify dimensionality of the input feature vector (default 1). Each component is stored not as a vector, but as a separate system and padded on it's own according to other options (i.e. for x*y system of 3-vector, first x*y elements correspond to the first dimension, then goes x*y for the second, etc). 
			forward_configuration.inverse = false; //Direction of FFT. false - forward, true - inverse.
			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [VkDeviceSize *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [VkDeviceSize *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			forward_configuration.device = &vkGPU->device;
			forward_configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
			forward_configuration.fence = &vkGPU->fence;
			forward_configuration.commandPool = &vkGPU->commandPool;
			forward_configuration.physicalDevice = &vkGPU->physicalDevice;
			forward_configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization

			//Allocate buffers for the input data. - we use 4 in this example
			VkDeviceSize* bufferSize = (VkDeviceSize*)malloc(sizeof(VkDeviceSize) * numBuf);
			for (uint32_t i = 0; i < numBuf; i++) {
				bufferSize[i] = {};
				bufferSize[i] = ((uint64_t)forward_configuration.coordinateFeatures) * sizeof(float) * 2 * forward_configuration.bufferStride[0] * forward_configuration.bufferStride[1] * forward_configuration.bufferStride[2] / numBuf;
			}
			VkBuffer* buffer = (VkBuffer*)malloc(numBuf * sizeof(VkBuffer));
			VkDeviceMemory* bufferDeviceMemory = (VkDeviceMemory*)malloc(numBuf * sizeof(VkDeviceMemory));
			VkBuffer* tempBuffer = (VkBuffer*)malloc(numBuf * sizeof(VkBuffer));
			VkDeviceMemory* tempBufferDeviceMemory = (VkDeviceMemory*)malloc(numBuf * sizeof(VkDeviceMemory));
			for (uint32_t i = 0; i < numBuf; i++) {
				buffer[i] = {};
				bufferDeviceMemory[i] = {};
				tempBuffer[i] = {};
				tempBufferDeviceMemory[i] = {};
				allocateFFTBuffer(vkGPU, &buffer[i], &bufferDeviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize[i]);
				allocateFFTBuffer(vkGPU, &tempBuffer[i], &tempBufferDeviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize[i]);
			}
			forward_configuration.isInputFormatted = false; //set to true if input is a different buffer, so it can have zeropadding/R2C added . Have to specifiy corresponding inputBufferStride
			forward_configuration.isOutputFormatted = false;//set to true if output is a different buffer, so it can have zeropadding/C2R automatically removed. Have to specifiy corresponding outputBufferStride

			forward_configuration.bufferNum = numBuf;
			forward_configuration.tempBufferNum = numBuf;
			forward_configuration.inputBufferNum = numBuf;
			forward_configuration.outputBufferNum = numBuf;

			forward_configuration.buffer = buffer;
			forward_configuration.tempBuffer = tempBuffer;
			forward_configuration.inputBuffer = buffer; //you can specify first buffer to read data from to be different from the buffer FFT is performed on. FFT is still in-place on the second buffer, this is here just for convenience.
			forward_configuration.outputBuffer = buffer;

			forward_configuration.bufferSize = bufferSize;
			forward_configuration.tempBufferSize = bufferSize;
			forward_configuration.inputBufferSize = bufferSize;
			forward_configuration.outputBufferSize = bufferSize;

			//Now we will create a similar configuration for inverse FFT and change inverse parameter to true.
			inverse_configuration = forward_configuration;
			inverse_configuration.inputBuffer = buffer;//If you continue working with previous data, select the FFT buffer as initial
			inverse_configuration.outputBuffer = buffer;
			inverse_configuration.inverse = true;

			//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
			/*float* buffer_input = (float*)malloc(bufferSize);

			for (uint32_t v = 0; v < forward_configuration.coordinateFeatures; v++) {
				for (uint32_t k = 0; k < forward_configuration.size[2]; k++) {
					for (uint32_t j = 0; j < forward_configuration.size[1]; j++) {
						for (uint32_t i = 0; i < forward_configuration.size[0]; i++) {
							buffer_input[2 * (i + j * forward_configuration.size[0] + k * (forward_configuration.size[0]) * forward_configuration.size[1] + v * (forward_configuration.size[0]) * forward_configuration.size[1] * forward_configuration.size[2])] = 2 * ((float)rand()) / RAND_MAX - 1.0;
							buffer_input[2 * (i + j * forward_configuration.size[0] + k * (forward_configuration.size[0]) * forward_configuration.size[1] + v * (forward_configuration.size[0]) * forward_configuration.size[1] * forward_configuration.size[2]) + 1] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						}
					}
				}
			}
			*/
			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
			uint64_t shift = 0;
			for (uint32_t i = 0; i < numBuf; i++) {
				transferDataFromCPU(vkGPU, buffer_input + shift / sizeof(float), &buffer[i], bufferSize[i]);
				shift += bufferSize[i];
			}

			//free(buffer_input);

			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			res = initializeVulkanFFT(&app_forward, forward_configuration);
			if (res != VK_SUCCESS) return res;
			res = initializeVulkanFFT(&app_inverse, inverse_configuration);
			if (res != VK_SUCCESS) return res;
			//Submit FFT+iFFT.
			uint32_t batch = ((4096 * 1024.0 * 1024.0) / (numBuf * bufferSize[0]) > 1000) ? 1000 : (4096 * 1024.0 * 1024.0) / (numBuf * bufferSize[0]);
			if (vkGPU->physicalDeviceProperties.vendorID == 0x8086) batch /= 4;
			if (batch == 0) batch = 1;
			if (vkGPU->physicalDeviceProperties.vendorID != 0x8086) batch *= 5;
			float totTime = performVulkanFFTiFFT(vkGPU, &app_forward, &app_inverse, batch);

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
					for (uint32_t i = 0; i < forward_configuration.FFTdim; i++)
						num_tot_transfers += app_forward.localFFTPlan->numAxisUploads[i];
					num_tot_transfers *= 4;
					if (file_output)
						fprintf(output, "VkFFT System: %d %dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f batch: %d benchmark: %d bandwidth: %0.1f\n", (int)log2(forward_configuration.size[0]), forward_configuration.size[0], forward_configuration.size[1], (numBuf * bufferSize[0]) / 1024 / 1024, avg_time, std_error, batch, (int)(((double)(numBuf * bufferSize[0]) / 1024) / avg_time), (numBuf * bufferSize[0]) / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);

					printf("VkFFT System: %d %dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f batch: %d benchmark: %d bandwidth: %0.1f\n", (int)log2(forward_configuration.size[0]), forward_configuration.size[0], forward_configuration.size[1], (numBuf * bufferSize[0]) / 1024 / 1024, avg_time, std_error, batch, (int)(((double)(numBuf * bufferSize[0]) / 1024) / avg_time), (numBuf * bufferSize[0]) / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / avg_time);
					benchmark_result += ((double)numBuf * bufferSize[0] / 1024) / avg_time;
				}


			}
			for (uint32_t i = 0; i < numBuf; i++) {

				vkDestroyBuffer(vkGPU->device, buffer[i], NULL);
				vkDestroyBuffer(vkGPU->device, tempBuffer[i], NULL);
				vkFreeMemory(vkGPU->device, bufferDeviceMemory[i], NULL);
				vkFreeMemory(vkGPU->device, tempBufferDeviceMemory[i], NULL);

			}
			free(bufferSize);
			deleteVulkanFFT(&app_forward);
			deleteVulkanFFT(&app_inverse);
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
#ifdef USE_FFTW
VkResult sample_11(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	VkResult res = VK_SUCCESS;
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
			int batch = 1;

#ifdef USE_cuFFT
			fftwf_complex* output_cuFFT = (fftwf_complex*)(malloc(sizeof(fftwf_complex) * dims[0] * dims[1] * dims[2]));
			launch_precision_cuFFT_single(inputC, (void*)output_cuFFT, benchmark_dimensions[n]);
#endif // USE_cuFFT

			//VkFFT part

			VkFFTConfiguration forward_configuration = defaultVkFFTConfiguration;
			VkFFTApplication app_forward = defaultVkFFTApplication;
			//VkFFTApplication app_inverse = defaultVkFFTApplication;
			forward_configuration.FFTdim = benchmark_dimensions[n][3]; //FFT dimension, 1D, 2D or 3D (default 1).
			forward_configuration.size[0] = benchmark_dimensions[n][0]; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			forward_configuration.size[1] = benchmark_dimensions[n][1];
			forward_configuration.size[2] = benchmark_dimensions[n][2];
			for (uint32_t i = 0; i < 3; i++) {
				forward_configuration.bufferStride[i] = forward_configuration.size[i];//can specify arbitrary buffer strides, if buffer is bigger than data - can be used for upscaling - you put data in the bigger buffer (2x in each dimension) in the corner, transform forward, zeropad to full buffer size and inverse
			}
			//PARAMETERS THAT CAN BE ADJUSTED FOR SPECIFIC GPU's - this configuration is by no means final form
			switch (vkGPU->physicalDeviceProperties.vendorID) {
			case 0x10DE://NVIDIA
				forward_configuration.coalescedMemory = 32;//the coalesced memory is equal to 32 bytes between L2 and VRAM. 
				forward_configuration.useLUT = false;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 4;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = 0;
				break;
			case 0x8086://INTEL
				forward_configuration.coalescedMemory = 64;
				forward_configuration.useLUT = true;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 2;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			case 0x1002://AMD
				forward_configuration.coalescedMemory = 32;
				forward_configuration.useLUT = false;
				forward_configuration.warpSize = 64;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = (vkGPU->physicalDeviceProperties.limits.maxComputeSharedMemorySize >= 65536) ? 2 : 4;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 19;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			default:
				forward_configuration.coalescedMemory = 64;
				forward_configuration.useLUT = false;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 1;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			}
			forward_configuration.performZeropadding[0] = false; //Perform padding with zeros on GPU. Still need to properly align input data (no need to fill padding area with meaningful data) but this will increase performance due to the lower amount of the memory reads/writes and omitting sequences only consisting of zeros.
			forward_configuration.performZeropadding[1] = false;
			forward_configuration.performZeropadding[2] = false;
			forward_configuration.performConvolution = false; //Perform convolution with precomputed kernel. 
			forward_configuration.performR2C = false; //Perform C2C transform. Can be combined with all other options. 
			forward_configuration.coordinateFeatures = 1; //Specify dimensionality of the input feature vector (default 1). Each component is stored not as a vector, but as a separate system and padded on it's own according to other options (i.e. for x*y system of 3-vector, first x*y elements correspond to the first dimension, then goes x*y for the second, etc). 
			forward_configuration.inverse = false; //Direction of FFT. false - forward, true - inverse.
			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [VkDeviceSize *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [VkDeviceSize *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			forward_configuration.device = &vkGPU->device;
			forward_configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
			forward_configuration.fence = &vkGPU->fence;
			forward_configuration.commandPool = &vkGPU->commandPool;
			forward_configuration.physicalDevice = &vkGPU->physicalDevice;
			forward_configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization
			//forward_configuration.compiler = &compiler;
			//forward_configuration.options = &options;
			//forward_configuration.code0 = code0;
			forward_configuration.reorderFourStep = true;
			//Custom path to the folder with shaders, default is "shaders");


			uint32_t numBuf = 1;

			//Allocate buffers for the input data. - we use 4 in this example
			VkDeviceSize* bufferSize = (VkDeviceSize*)malloc(sizeof(VkDeviceSize) * numBuf);
			for (uint32_t i = 0; i < numBuf; i++) {
				bufferSize[i] = {};
				bufferSize[i] = ((uint64_t)forward_configuration.coordinateFeatures) * sizeof(float) * 2 * forward_configuration.bufferStride[0] * forward_configuration.bufferStride[1] * forward_configuration.bufferStride[2] / numBuf;
			}
			VkBuffer* buffer = (VkBuffer*)malloc(numBuf * sizeof(VkBuffer));
			VkDeviceMemory* bufferDeviceMemory = (VkDeviceMemory*)malloc(numBuf * sizeof(VkDeviceMemory));
			VkBuffer* tempBuffer = (VkBuffer*)malloc(numBuf * sizeof(VkBuffer));
			VkDeviceMemory* tempBufferDeviceMemory = (VkDeviceMemory*)malloc(numBuf * sizeof(VkDeviceMemory));
			for (uint32_t i = 0; i < numBuf; i++) {
				buffer[i] = {};
				bufferDeviceMemory[i] = {};
				tempBuffer[i] = {};
				tempBufferDeviceMemory[i] = {};
				allocateFFTBuffer(vkGPU, &buffer[i], &bufferDeviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize[i]);
				allocateFFTBuffer(vkGPU, &tempBuffer[i], &tempBufferDeviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize[i]);
			}
			forward_configuration.isInputFormatted = false; //set to true if input is a different buffer, so it can have zeropadding/R2C added . Have to specifiy corresponding inputBufferStride
			forward_configuration.isOutputFormatted = false;//set to true if output is a different buffer, so it can have zeropadding/C2R automatically removed. Have to specifiy corresponding outputBufferStride

			forward_configuration.bufferNum = numBuf;
			forward_configuration.tempBufferNum = numBuf;
			forward_configuration.inputBufferNum = numBuf;
			forward_configuration.outputBufferNum = numBuf;

			forward_configuration.buffer = buffer;
			forward_configuration.tempBuffer = tempBuffer;
			forward_configuration.inputBuffer = buffer; //you can specify first buffer to read data from to be different from the buffer FFT is performed on. FFT is still in-place on the second buffer, this is here just for convenience.
			forward_configuration.outputBuffer = buffer;

			forward_configuration.bufferSize = bufferSize;
			forward_configuration.tempBufferSize = bufferSize;
			forward_configuration.inputBufferSize = bufferSize;
			forward_configuration.outputBufferSize = bufferSize;

			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
			uint64_t shift = 0;
			for (uint32_t i = 0; i < numBuf; i++) {
				transferDataFromCPU(vkGPU, (float*)(inputC + shift / sizeof(fftwf_complex)), &buffer[i], bufferSize[i]);
				shift += bufferSize[i];
			}
			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			res = initializeVulkanFFT(&app_forward, forward_configuration);
			if (res != VK_SUCCESS) return res;
			//forward_configuration.inverse = true;
			//res = initializeVulkanFFT(&app_inverse,forward_configuration);
			//Submit FFT+iFFT.
			//batch = 1;
			performVulkanFFT(vkGPU, &app_forward, batch);
			//totTime = performVulkanFFT(vkGPU, &app_inverse, batch);
			fftwf_complex* output_VkFFT = (fftwf_complex*)(malloc(sizeof(fftwf_complex) * dims[0] * dims[1] * dims[2]));

			//Transfer data from GPU using staging buffer.
			shift = 0;
			for (uint32_t i = 0; i < numBuf; i++) {
				transferDataToCPU(vkGPU, (float*)(output_VkFFT + shift / sizeof(fftwf_complex)), &buffer[i], bufferSize[i]);
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

						int N = (forward_configuration.inverse) ? dims[0] * dims[1] * dims[2] : 1;

						//if (file_output) fprintf(output, "%f %f - %f %f \n", output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] / N, output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1] / N, output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][0], output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][1]);

						//printf("%f %f - %f %f \n", output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] / N, output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1] / N, output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][0], output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][1]);
						float current_data_norm = sqrt(output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] * output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] + output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1] * output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1]);
#ifdef USE_cuFFT
						float current_diff_x_cuFFT = (output_cuFFT[loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]][0] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0]);
						float current_diff_y_cuFFT = (output_cuFFT[loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]][1] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1]);
						float current_diff_norm_cuFFT = sqrt(current_diff_x_cuFFT * current_diff_x_cuFFT + current_diff_y_cuFFT * current_diff_y_cuFFT);
						if (current_diff_norm_cuFFT > max_difference[0]) max_difference[0] = current_diff_norm_cuFFT;
						avg_difference[0] += current_diff_norm_cuFFT;
						if ((current_diff_norm_cuFFT / current_data_norm > max_eps[0]) && (current_data_norm > 1e-10)) {
							max_eps[0] = current_diff_norm_cuFFT / current_data_norm;
						}
						avg_eps[0] += (current_data_norm > 1e-10) ? current_diff_norm_cuFFT / current_data_norm : 0;
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

				vkDestroyBuffer(vkGPU->device, buffer[i], NULL);
				vkDestroyBuffer(vkGPU->device, tempBuffer[i], NULL);
				vkFreeMemory(vkGPU->device, bufferDeviceMemory[i], NULL);
				vkFreeMemory(vkGPU->device, tempBufferDeviceMemory[i], NULL);

			}
#ifdef USE_cuFFT
			free(output_cuFFT);
#endif
			free(bufferSize);
			deleteVulkanFFT(&app_forward);
			free(inputC);
			fftw_destroy_plan(p);
			free(inputC_double);
			free(output_FFTW);
		}
	}
	return res;
}
VkResult sample_12(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	VkResult res = VK_SUCCESS;
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
			fftw_complex* output_cuFFT = (fftw_complex*)(malloc(sizeof(fftw_complex) * dims[0] * dims[1] * dims[2]));
			launch_precision_cuFFT_double(inputC, (void*)output_cuFFT, benchmark_dimensions[n]);
#endif // USE_cuFFT

			float totTime = 0;
			int batch = 1;

			//VkFFT part

			VkFFTConfiguration forward_configuration = defaultVkFFTConfiguration;
			VkFFTApplication app_forward = defaultVkFFTApplication;
			//VkFFTApplication app_inverse;
			forward_configuration.FFTdim = benchmark_dimensions[n][3]; //FFT dimension, 1D, 2D or 3D (default 1).
			forward_configuration.size[0] = benchmark_dimensions[n][0]; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			forward_configuration.size[1] = benchmark_dimensions[n][1];
			forward_configuration.size[2] = benchmark_dimensions[n][2];
			for (uint32_t i = 0; i < 3; i++) {
				forward_configuration.bufferStride[i] = forward_configuration.size[i];//can specify arbitrary buffer strides, if buffer is bigger than data - can be used for upscaling - you put data in the bigger buffer (2x in each dimension) in the corner, transform forward, zeropad to full buffer size and inverse
			}
			//PARAMETERS THAT CAN BE ADJUSTED FOR SPECIFIC GPU's - this configuration is by no means final form
			switch (vkGPU->physicalDeviceProperties.vendorID) {
			case 0x10DE://NVIDIA 
				forward_configuration.coalescedMemory = 32;//the coalesced memory is equal to 32 bytes between L2 and VRAM. 
				forward_configuration.useLUT = true;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 4;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = 0;
				break;
			case 0x8086://INTEL
				forward_configuration.coalescedMemory = 64;
				forward_configuration.useLUT = true;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 2;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			case 0x1002://AMD
				forward_configuration.coalescedMemory = 32;
				forward_configuration.useLUT = true;
				forward_configuration.warpSize = 64;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = (vkGPU->physicalDeviceProperties.limits.maxComputeSharedMemorySize >= 65536) ? 2 : 4;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 20;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			default:
				forward_configuration.coalescedMemory = 64;
				forward_configuration.useLUT = true;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 1;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			}
			forward_configuration.performZeropadding[0] = false; //Perform padding with zeros on GPU. Still need to properly align input data (no need to fill padding area with meaningful data) but this will increase performance due to the lower amount of the memory reads/writes and omitting sequences only consisting of zeros.
			forward_configuration.performZeropadding[1] = false;
			forward_configuration.performZeropadding[2] = false;
			forward_configuration.performConvolution = false; //Perform convolution with precomputed kernel. 
			forward_configuration.performR2C = false; //Perform C2C transform. Can be combined with all other options. 
			forward_configuration.coordinateFeatures = 1; //Specify dimensionality of the input feature vector (default 1). Each component is stored not as a vector, but as a separate system and padded on it's own according to other options (i.e. for x*y system of 3-vector, first x*y elements correspond to the first dimension, then goes x*y for the second, etc). 
			forward_configuration.inverse = false; //Direction of FFT. false - forward, true - inverse.
			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [VkDeviceSize *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [VkDeviceSize *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			forward_configuration.device = &vkGPU->device;
			forward_configuration.queue = &vkGPU->queue;
			forward_configuration.fence = &vkGPU->fence;
			forward_configuration.commandPool = &vkGPU->commandPool;
			forward_configuration.physicalDevice = &vkGPU->physicalDevice;
			forward_configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization
			//forward_configuration.compiler = &compiler;
			//forward_configuration.options = &options;
			//forward_configuration.code0 = code0;
			forward_configuration.doublePrecision = true;
			forward_configuration.reorderFourStep = true;

			//Custom path to the folder with shaders, default is "shaders");

			uint32_t numBuf = 1;

			//Allocate buffers for the input data. - we use 4 in this example
			VkDeviceSize* bufferSize = (VkDeviceSize*)malloc(sizeof(VkDeviceSize) * numBuf);
			for (uint32_t i = 0; i < numBuf; i++) {
				bufferSize[i] = {};
				bufferSize[i] = ((uint64_t)forward_configuration.coordinateFeatures) * sizeof(double) * 2 * forward_configuration.bufferStride[0] * forward_configuration.bufferStride[1] * forward_configuration.bufferStride[2] / numBuf;
			}
			VkBuffer* buffer = (VkBuffer*)malloc(numBuf * sizeof(VkBuffer));
			VkDeviceMemory* bufferDeviceMemory = (VkDeviceMemory*)malloc(numBuf * sizeof(VkDeviceMemory));
			VkBuffer* tempBuffer = (VkBuffer*)malloc(numBuf * sizeof(VkBuffer));
			VkDeviceMemory* tempBufferDeviceMemory = (VkDeviceMemory*)malloc(numBuf * sizeof(VkDeviceMemory));
			for (uint32_t i = 0; i < numBuf; i++) {
				buffer[i] = {};
				bufferDeviceMemory[i] = {};
				tempBuffer[i] = {};
				tempBufferDeviceMemory[i] = {};
				allocateFFTBuffer(vkGPU, &buffer[i], &bufferDeviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize[i]);
				allocateFFTBuffer(vkGPU, &tempBuffer[i], &tempBufferDeviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize[i]);
			}
			forward_configuration.isInputFormatted = false; //set to true if input is a different buffer, so it can have zeropadding/R2C added . Have to specifiy corresponding inputBufferStride
			forward_configuration.isOutputFormatted = false;//set to true if output is a different buffer, so it can have zeropadding/C2R automatically removed. Have to specifiy corresponding outputBufferStride

			forward_configuration.bufferNum = numBuf;
			forward_configuration.tempBufferNum = numBuf;
			forward_configuration.inputBufferNum = numBuf;
			forward_configuration.outputBufferNum = numBuf;

			forward_configuration.buffer = buffer;
			forward_configuration.tempBuffer = tempBuffer;
			forward_configuration.inputBuffer = buffer; //you can specify first buffer to read data from to be different from the buffer FFT is performed on. FFT is still in-place on the second buffer, this is here just for convenience.
			forward_configuration.outputBuffer = buffer;

			forward_configuration.bufferSize = bufferSize;
			forward_configuration.tempBufferSize = bufferSize;
			forward_configuration.inputBufferSize = bufferSize;
			forward_configuration.outputBufferSize = bufferSize;

			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
			uint64_t shift = 0;
			for (uint32_t i = 0; i < numBuf; i++) {
				transferDataFromCPU(vkGPU, (float*)(inputC + shift / sizeof(fftw_complex)), &buffer[i], bufferSize[i]);
				shift += bufferSize[i];
			}
			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			res = initializeVulkanFFT(&app_forward, forward_configuration);
			if (res != VK_SUCCESS) return res;
			//forward_configuration.inverse = true;
			//app_inverse.res = initializeVulkanFFT(forward_configuration);
			//Submit FFT+iFFT.
			//batch = 1;
			performVulkanFFT(vkGPU, &app_forward, batch);
			//totTime = performVulkanFFT(vkGPU, &app_inverse, batch);
			fftw_complex* output_VkFFT = (fftw_complex*)(malloc(sizeof(fftw_complex) * dims[0] * dims[1] * dims[2]));

			//Transfer data from GPU using staging buffer.
			shift = 0;
			for (uint32_t i = 0; i < numBuf; i++) {
				transferDataToCPU(vkGPU, (float*)(output_VkFFT + shift / sizeof(fftw_complex)), &buffer[i], bufferSize[i]);
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
						//int N = (forward_configuration.inverse) ? dims[0] * dims[1] * dims[2] : 1;

						//if (file_output) fprintf(output, "%f %f - %f %f \n", output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] / N, output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1] / N, output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][0], output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][1]);
						//printf("%f %f - %f %f \n", output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] / N, output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1] / N, output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][0], output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][1]);
						double current_data_norm = sqrt(output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] * output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] + output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1] * output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1]);

#ifdef USE_cuFFT
						double current_diff_x_cuFFT = (output_cuFFT[loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]][0] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0]);
						double current_diff_y_cuFFT = (output_cuFFT[loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]][1] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1]);
						double current_diff_norm_cuFFT = sqrt(current_diff_x_cuFFT * current_diff_x_cuFFT + current_diff_y_cuFFT * current_diff_y_cuFFT);
						if (current_diff_norm_cuFFT > max_difference[0]) max_difference[0] = current_diff_norm_cuFFT;
						avg_difference[0] += current_diff_norm_cuFFT;
						if ((current_diff_norm_cuFFT / current_data_norm > max_eps[0]) && (current_data_norm > 1e-10)) {
							max_eps[0] = current_diff_norm_cuFFT / current_data_norm;
						}
						avg_eps[0] += (current_data_norm > 1e-10) ? current_diff_norm_cuFFT / current_data_norm : 0;
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
			if (file_output)
				fprintf(output, "VkFFT System: %dx%dx%d avg_difference: %.15f max_difference: %.15f avg_eps: %.15f max_eps: %.15f\n", dims[0], dims[1], dims[2], avg_difference[1], max_difference[1], avg_eps[1], max_eps[1]);
			printf("VkFFT System: %dx%dx%d avg_difference: %.15f max_difference: %.15f avg_eps: %.15f max_eps: %.15f\n", dims[0], dims[1], dims[2], avg_difference[1], max_difference[1], avg_eps[1], max_eps[1]);
			free(output_VkFFT);
			for (uint32_t i = 0; i < numBuf; i++) {

				vkDestroyBuffer(vkGPU->device, buffer[i], NULL);
				vkDestroyBuffer(vkGPU->device, tempBuffer[i], NULL);
				vkFreeMemory(vkGPU->device, bufferDeviceMemory[i], NULL);
				vkFreeMemory(vkGPU->device, tempBufferDeviceMemory[i], NULL);

			}
#ifdef USE_cuFFT
			free(output_cuFFT);
#endif
			deleteVulkanFFT(&app_forward);
			free(inputC);
			fftw_destroy_plan(p);
			free(inputC_double);
			free(output_FFTW);
		}
	}
	return res;
}
#if (VK_API_VERSION>10)
VkResult sample_13(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	VkResult res = VK_SUCCESS;
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
			int batch = 1;

#ifdef USE_cuFFT
			half2* output_cuFFT = (half2*)(malloc(2 * sizeof(half) * dims[0] * dims[1] * dims[2]));
			launch_precision_cuFFT_half(inputC, (void*)output_cuFFT, benchmark_dimensions[n]);
#endif // USE_cuFFT

			//VkFFT part

			VkFFTConfiguration forward_configuration = defaultVkFFTConfiguration;
			VkFFTApplication app_forward = defaultVkFFTApplication;
			//VkFFTApplication app_inverse = defaultVkFFTApplication;
			forward_configuration.FFTdim = benchmark_dimensions[n][3]; //FFT dimension, 1D, 2D or 3D (default 1).
			forward_configuration.size[0] = benchmark_dimensions[n][0]; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			forward_configuration.size[1] = benchmark_dimensions[n][1];
			forward_configuration.size[2] = benchmark_dimensions[n][2];
			for (uint32_t i = 0; i < 3; i++) {
				forward_configuration.bufferStride[i] = forward_configuration.size[i];//can specify arbitrary buffer strides, if buffer is bigger than data - can be used for upscaling - you put data in the bigger buffer (2x in each dimension) in the corner, transform forward, zeropad to full buffer size and inverse
			}
			//PARAMETERS THAT CAN BE ADJUSTED FOR SPECIFIC GPU's - this configuration is by no means final form
			switch (vkGPU->physicalDeviceProperties.vendorID) {
			case 0x10DE://NVIDIA 
				forward_configuration.coalescedMemory = 64;//have to set coalesce more, as calculations are still float, while uploads are half.
				forward_configuration.useLUT = false;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 4;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			case 0x8086://INTEL
				forward_configuration.coalescedMemory = 128;
				forward_configuration.useLUT = true;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 2;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			case 0x1002://AMD
				forward_configuration.coalescedMemory = 64;
				forward_configuration.useLUT = false;
				forward_configuration.warpSize = 64;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = (vkGPU->physicalDeviceProperties.limits.maxComputeSharedMemorySize >= 65536) ? 2 : 4;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			default:
				forward_configuration.coalescedMemory = 64;
				forward_configuration.useLUT = false;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 1;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			}
			forward_configuration.performZeropadding[0] = false; //Perform padding with zeros on GPU. Still need to properly align input data (no need to fill padding area with meaningful data) but this will increase performance due to the lower amount of the memory reads/writes and omitting sequences only consisting of zeros.
			forward_configuration.performZeropadding[1] = false;
			forward_configuration.performZeropadding[2] = false;
			forward_configuration.performConvolution = false; //Perform convolution with precomputed kernel. 
			forward_configuration.performR2C = false; //Perform C2C transform. Can be combined with all other options. 
			forward_configuration.coordinateFeatures = 1; //Specify dimensionality of the input feature vector (default 1). Each component is stored not as a vector, but as a separate system and padded on it's own according to other options (i.e. for x*y system of 3-vector, first x*y elements correspond to the first dimension, then goes x*y for the second, etc). 
			forward_configuration.inverse = false; //Direction of FFT. false - forward, true - inverse.
			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [VkDeviceSize *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [VkDeviceSize *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			forward_configuration.device = &vkGPU->device;
			forward_configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
			forward_configuration.fence = &vkGPU->fence;
			forward_configuration.commandPool = &vkGPU->commandPool;
			forward_configuration.physicalDevice = &vkGPU->physicalDevice;
			forward_configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization
			//forward_configuration.compiler = &compiler;
			//forward_configuration.options = &options;
			//forward_configuration.code0 = code0;
			forward_configuration.reorderFourStep = true;
			forward_configuration.halfPrecision = true;
			//Custom path to the folder with shaders, default is "shaders");


			uint32_t numBuf = 1;

			//Allocate buffers for the input data. - we use 4 in this example
			VkDeviceSize* bufferSize = (VkDeviceSize*)malloc(sizeof(VkDeviceSize) * numBuf);
			for (uint32_t i = 0; i < numBuf; i++) {
				bufferSize[i] = {};
				bufferSize[i] = ((uint64_t)forward_configuration.coordinateFeatures) * sizeof(half) * 2 * forward_configuration.bufferStride[0] * forward_configuration.bufferStride[1] * forward_configuration.bufferStride[2] / numBuf;
			}
			VkBuffer* buffer = (VkBuffer*)malloc(numBuf * sizeof(VkBuffer));
			VkDeviceMemory* bufferDeviceMemory = (VkDeviceMemory*)malloc(numBuf * sizeof(VkDeviceMemory));
			VkBuffer* tempBuffer = (VkBuffer*)malloc(numBuf * sizeof(VkBuffer));
			VkDeviceMemory* tempBufferDeviceMemory = (VkDeviceMemory*)malloc(numBuf * sizeof(VkDeviceMemory));
			for (uint32_t i = 0; i < numBuf; i++) {
				buffer[i] = {};
				bufferDeviceMemory[i] = {};
				tempBuffer[i] = {};
				tempBufferDeviceMemory[i] = {};
				allocateFFTBuffer(vkGPU, &buffer[i], &bufferDeviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize[i]);
				allocateFFTBuffer(vkGPU, &tempBuffer[i], &tempBufferDeviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize[i]);
			}
			forward_configuration.isInputFormatted = false; //set to true if input is a different buffer, so it can have zeropadding/R2C added . Have to specifiy corresponding inputBufferStride
			forward_configuration.isOutputFormatted = false;//set to true if output is a different buffer, so it can have zeropadding/C2R automatically removed. Have to specifiy corresponding outputBufferStride

			forward_configuration.bufferNum = numBuf;
			forward_configuration.tempBufferNum = numBuf;
			forward_configuration.inputBufferNum = numBuf;
			forward_configuration.outputBufferNum = numBuf;

			forward_configuration.buffer = buffer;
			forward_configuration.tempBuffer = tempBuffer;
			forward_configuration.inputBuffer = buffer; //you can specify first buffer to read data from to be different from the buffer FFT is performed on. FFT is still in-place on the second buffer, this is here just for convenience.
			forward_configuration.outputBuffer = buffer;

			forward_configuration.bufferSize = bufferSize;
			forward_configuration.tempBufferSize = bufferSize;
			forward_configuration.inputBufferSize = bufferSize;
			forward_configuration.outputBufferSize = bufferSize;

			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
			uint64_t shift = 0;
			for (uint32_t i = 0; i < numBuf; i++) {
				transferDataFromCPU(vkGPU, (half2*)(inputC + shift / 2 / sizeof(half)), &buffer[i], bufferSize[i]);
				shift += bufferSize[i];
			}
			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			res = initializeVulkanFFT(&app_forward, forward_configuration);
			if (res != VK_SUCCESS) return res;
			//forward_configuration.inverse = true;
			//res = initializeVulkanFFT(&app_inverse,forward_configuration);
			//Submit FFT+iFFT.
			//batch = 1;
			performVulkanFFT(vkGPU, &app_forward, batch);
			//totTime = performVulkanFFT(vkGPU, &app_inverse, batch);
			half2* output_VkFFT = (half2*)(malloc(2 * sizeof(half) * dims[0] * dims[1] * dims[2]));

			//Transfer data from GPU using staging buffer.
			shift = 0;
			for (uint32_t i = 0; i < numBuf; i++) {
				transferDataToCPU(vkGPU, (half2*)(output_VkFFT + shift / 2 / sizeof(half)), &buffer[i], bufferSize[i]);
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

						int N = (forward_configuration.inverse) ? dims[0] * dims[1] * dims[2] : 1;

						//if (file_output) fprintf(output, "%f %f - %f %f \n", output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] / N, output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1] / N, output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][0], output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][1]);
						//printf("%f %f - %f %f \n", output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] / N, output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1] / N, output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][0], output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][1]);
						float current_data_norm = sqrt(output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] * output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] + output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1] * output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1]);
#ifdef USE_cuFFT
						float current_diff_x_cuFFT = (output_cuFFT[loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]][0] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0]);
						float current_diff_y_cuFFT = (output_cuFFT[loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]][1] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1]);
						float current_diff_norm_cuFFT = sqrt(current_diff_x_cuFFT * current_diff_x_cuFFT + current_diff_y_cuFFT * current_diff_y_cuFFT);
						if (current_diff_norm_cuFFT > max_difference[0]) max_difference[0] = current_diff_norm_cuFFT;
						avg_difference[0] += current_diff_norm_cuFFT;
						if ((current_diff_norm_cuFFT / current_data_norm > max_eps[0]) && (current_data_norm > 1e-10)) {
							max_eps[0] = current_diff_norm_cuFFT / current_data_norm;
						}
						avg_eps[0] += (current_data_norm > 1e-10) ? current_diff_norm_cuFFT / current_data_norm : 0;
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

				vkDestroyBuffer(vkGPU->device, buffer[i], NULL);
				vkDestroyBuffer(vkGPU->device, tempBuffer[i], NULL);
				vkFreeMemory(vkGPU->device, bufferDeviceMemory[i], NULL);
				vkFreeMemory(vkGPU->device, tempBufferDeviceMemory[i], NULL);

			}
#ifdef USE_cuFFT
			free(output_cuFFT);
#endif
			free(bufferSize);
			deleteVulkanFFT(&app_forward);
			free(inputC);
			fftw_destroy_plan(p);
			free(inputC_double);
			free(output_FFTW);
		}
	}
	return res;
}
#endif
VkResult sample_14(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output, uint32_t isCompilerInitialized) {
	VkResult res = VK_SUCCESS;
	if (file_output)
		fprintf(output, "14 - VkFFT/FFTW C2C power 3/5/7 precision test in single precision\n");
	printf("14 - VkFFT/FFTW C2C power 3/5/7 precision test in single precision\n");

	const int num_benchmark_samples = 145;
	const int num_runs = 1;

	uint32_t benchmark_dimensions[num_benchmark_samples][4] = { {3, 1, 1, 1},{5, 1, 1, 1},{6, 1, 1, 1},{7, 1, 1, 1},{9, 1, 1, 1},{10, 1, 1, 1},{12, 1, 1, 1},{14, 1, 1, 1},
		{15, 1, 1, 1},{21, 1, 1, 1},{24, 1, 1, 1},{25, 1, 1, 1},{27, 1, 1, 1},{28, 1, 1, 1},{30, 1, 1, 1},{35, 1, 1, 1},{45, 1, 1, 1},{42, 1, 1, 1},{49, 1, 1, 1},{56, 1, 1, 1},{60, 1, 1, 1},{81, 1, 1, 1},
		{125, 1, 1, 1},{243, 1, 1, 1},{343, 1, 1, 1},{625, 1, 1, 1},{720, 1, 1, 1},{1080, 1, 1, 1},{1400, 1, 1, 1},{1440, 1, 1, 1},{1920, 1, 1, 1},{2160, 1, 1, 1},{3024,1,1,1},{3500,1,1,1},
		{3840, 1, 1, 1},{4000 , 1, 1, 1},{4050, 1, 1, 1},{4320 , 1, 1, 1},{7000,1,1,1},{7680, 1, 1, 1},{9000, 1, 1, 1},{7680 * 5, 1, 1, 1},
		{(uint32_t)pow(3,10), 1, 1, 1},{(uint32_t)pow(3,11), 1, 1, 1},{(uint32_t)pow(3,12), 1, 1, 1},{(uint32_t)pow(3,13), 1, 1, 1},{(uint32_t)pow(3,14), 1, 1, 1},{(uint32_t)pow(3,15), 1, 1, 1},
		{(uint32_t)pow(5,5), 1, 1, 1},{(uint32_t)pow(5,6), 1, 1, 1},{(uint32_t)pow(5,7), 1, 1, 1},{(uint32_t)pow(5,8), 1, 1, 1},{(uint32_t)pow(5,9), 1, 1, 1},
		{(uint32_t)pow(7,4), 1, 1, 1},{(uint32_t)pow(7,5), 1, 1, 1},{(uint32_t)pow(7,6), 1, 1, 1},{(uint32_t)pow(7,7), 1, 1, 1},{(uint32_t)pow(7,8), 1, 1, 1},
		{8, 3, 1, 2},{8, 5, 1, 2},{8, 6, 1, 2},{8, 7, 1, 2},{8, 9, 1, 2},{8, 10, 1, 2},{8, 12, 1, 2},{8, 14, 1, 2},{8, 15, 1, 2},{8, 21, 1, 2},{8, 24, 1, 2},
		{8, 25, 1, 2},{8, 27, 1, 2},{8, 28, 1, 2},{8, 30, 1, 2},{8, 35, 1, 2},{8, 45, 1, 2},{8, 49, 1, 2},{8, 56, 1, 2},{8, 60, 1, 2},{8, 81, 1, 2},{8, 125, 1, 2},{8, 243, 1, 2},{8, 343, 1, 2},
		{8, 625, 1, 2},{8, 720, 1, 2},{8, 1080, 1, 2},{8, 1400, 1, 2},{8, 1440, 1, 2},{8, 1920, 1, 2},{8, 2160, 1, 2},{8, 3024, 1, 2},{8, 3500, 1, 2},
		{8, 3840, 1, 2},{8, 4000, 1, 2},{8, 4050, 1, 2},{8, 4320, 1, 2},{8, 7000, 1, 2},{8, 7680, 1, 2},{8, 4050 * 3, 1, 2},{8, 7680 * 5, 1, 2}, {720, 480, 1, 2},{1280, 720, 1, 2},{1920, 1080, 1, 2}, {2560, 1440, 1, 2},{3840, 2160, 1, 2},{7680, 4320, 1, 2},
		{8, (uint32_t)pow(3,10), 1, 2},	{8, (uint32_t)pow(3,11), 1, 2}, {8, (uint32_t)pow(3,12), 1, 2}, {8, (uint32_t)pow(3,13), 1, 2}, {8, (uint32_t)pow(3,14), 1, 2}, {8, (uint32_t)pow(3,15), 1, 2},
		{8, (uint32_t)pow(5,5), 1, 2},	{8, (uint32_t)pow(5,6), 1, 2}, {8, (uint32_t)pow(5,7), 1, 2}, {8, (uint32_t)pow(5,8), 1, 2}, {8, (uint32_t)pow(5,9), 1, 2},
		{8, (uint32_t)pow(7,4), 1, 2},{8, (uint32_t)pow(7,5), 1, 2},{8, (uint32_t)pow(7,6), 1, 2},{8, (uint32_t)pow(7,7), 1, 2},{8, (uint32_t)pow(7,8), 1, 2},
		{3, 3, 3, 3},{5, 5, 5, 3},{6, 6, 6, 3},{7, 7, 7, 3},{9, 9, 9, 3},{10, 10, 10, 3},{12, 12, 12, 3},{14, 14, 14, 3},
		{15, 15, 15, 3},{21, 21, 21, 3},{24, 24, 24, 3},{25, 25, 25, 3},{27, 27, 27, 3},{28, 28, 28, 3},{30, 30, 30, 3},{35, 35, 35, 3},{42, 42, 42, 3},{45, 45, 45, 3},{49, 49, 49, 3},{56, 56, 56, 3},{60, 60, 60, 3},{81, 81, 81, 3},
		{125, 125, 125, 3},{243, 243, 243, 3}
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
			int batch = 1;

			//VkFFT part

			VkFFTConfiguration forward_configuration = defaultVkFFTConfiguration;
			VkFFTApplication app_forward = defaultVkFFTApplication;
			//VkFFTApplication app_inverse = defaultVkFFTApplication;
			forward_configuration.FFTdim = benchmark_dimensions[n][3]; //FFT dimension, 1D, 2D or 3D (default 1).
			forward_configuration.size[0] = benchmark_dimensions[n][0]; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			forward_configuration.size[1] = benchmark_dimensions[n][1];
			forward_configuration.size[2] = benchmark_dimensions[n][2];
			for (uint32_t i = 0; i < 3; i++) {
				forward_configuration.bufferStride[i] = forward_configuration.size[i];//can specify arbitrary buffer strides, if buffer is bigger than data - can be used for upscaling - you put data in the bigger buffer (2x in each dimension) in the corner, transform forward, zeropad to full buffer size and inverse
			}
			//PARAMETERS THAT CAN BE ADJUSTED FOR SPECIFIC GPU's - this configuration is by no means final form
			switch (vkGPU->physicalDeviceProperties.vendorID) {
			case 0x10DE://NVIDIA
				forward_configuration.coalescedMemory = 32;//the coalesced memory is equal to 32 bytes between L2 and VRAM. 
				forward_configuration.useLUT = false;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 4;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			case 0x8086://INTEL
				forward_configuration.coalescedMemory = 64;
				forward_configuration.useLUT = true;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 2;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			case 0x1002://AMD
				forward_configuration.coalescedMemory = 32;
				forward_configuration.useLUT = false;
				forward_configuration.warpSize = 64;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = (vkGPU->physicalDeviceProperties.limits.maxComputeSharedMemorySize >= 65536) ? 2 : 4;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 19;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			default:
				forward_configuration.coalescedMemory = 64;
				forward_configuration.useLUT = false;
				forward_configuration.warpSize = 32;
				forward_configuration.registerBoostNonPow2 = 0;
				forward_configuration.registerBoost = 1;
				forward_configuration.registerBoost4Step = 1;
				forward_configuration.swapTo3Stage4Step = 0;
				forward_configuration.performHalfBandwidthBoost = false;
				break;
			}
			forward_configuration.performZeropadding[0] = false; //Perform padding with zeros on GPU. Still need to properly align input data (no need to fill padding area with meaningful data) but this will increase performance due to the lower amount of the memory reads/writes and omitting sequences only consisting of zeros.
			forward_configuration.performZeropadding[1] = false;
			forward_configuration.performZeropadding[2] = false;
			forward_configuration.performConvolution = false; //Perform convolution with precomputed kernel. 
			forward_configuration.performR2C = false; //Perform C2C transform. Can be combined with all other options. 
			forward_configuration.coordinateFeatures = 1; //Specify dimensionality of the input feature vector (default 1). Each component is stored not as a vector, but as a separate system and padded on it's own according to other options (i.e. for x*y system of 3-vector, first x*y elements correspond to the first dimension, then goes x*y for the second, etc). 
			forward_configuration.inverse = false; //Direction of FFT. false - forward, true - inverse.
			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [VkDeviceSize *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [VkDeviceSize *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			forward_configuration.device = &vkGPU->device;
			forward_configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
			forward_configuration.fence = &vkGPU->fence;
			forward_configuration.commandPool = &vkGPU->commandPool;
			forward_configuration.physicalDevice = &vkGPU->physicalDevice;
			forward_configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization
			//forward_configuration.compiler = &compiler;
			//forward_configuration.options = &options;
			//forward_configuration.code0 = code0;
			forward_configuration.reorderFourStep = true;
			//Custom path to the folder with shaders, default is "shaders");


			uint32_t numBuf = 1;

			//Allocate buffers for the input data. - we use 4 in this example
			VkDeviceSize* bufferSize = (VkDeviceSize*)malloc(sizeof(VkDeviceSize) * numBuf);
			for (uint32_t i = 0; i < numBuf; i++) {
				bufferSize[i] = {};
				bufferSize[i] = ((uint64_t)forward_configuration.coordinateFeatures) * sizeof(float) * 2 * forward_configuration.bufferStride[0] * forward_configuration.bufferStride[1] * forward_configuration.bufferStride[2] / numBuf;
			}
			VkBuffer* buffer = (VkBuffer*)malloc(numBuf * sizeof(VkBuffer));
			VkDeviceMemory* bufferDeviceMemory = (VkDeviceMemory*)malloc(numBuf * sizeof(VkDeviceMemory));
			VkBuffer* tempBuffer = (VkBuffer*)malloc(numBuf * sizeof(VkBuffer));
			VkDeviceMemory* tempBufferDeviceMemory = (VkDeviceMemory*)malloc(numBuf * sizeof(VkDeviceMemory));
			for (uint32_t i = 0; i < numBuf; i++) {
				buffer[i] = {};
				bufferDeviceMemory[i] = {};
				tempBuffer[i] = {};
				tempBufferDeviceMemory[i] = {};
				allocateFFTBuffer(vkGPU, &buffer[i], &bufferDeviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize[i]);
				allocateFFTBuffer(vkGPU, &tempBuffer[i], &tempBufferDeviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize[i]);
			}
			forward_configuration.isInputFormatted = false; //set to true if input is a different buffer, so it can have zeropadding/R2C added . Have to specifiy corresponding inputBufferStride
			forward_configuration.isOutputFormatted = false;//set to true if output is a different buffer, so it can have zeropadding/C2R automatically removed. Have to specifiy corresponding outputBufferStride

			forward_configuration.bufferNum = numBuf;
			forward_configuration.tempBufferNum = numBuf;
			forward_configuration.inputBufferNum = numBuf;
			forward_configuration.outputBufferNum = numBuf;

			forward_configuration.buffer = buffer;
			forward_configuration.tempBuffer = tempBuffer;
			forward_configuration.inputBuffer = buffer; //you can specify first buffer to read data from to be different from the buffer FFT is performed on. FFT is still in-place on the second buffer, this is here just for convenience.
			forward_configuration.outputBuffer = buffer;

			forward_configuration.bufferSize = bufferSize;
			forward_configuration.tempBufferSize = bufferSize;
			forward_configuration.inputBufferSize = bufferSize;
			forward_configuration.outputBufferSize = bufferSize;

			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
			uint64_t shift = 0;
			for (uint32_t i = 0; i < numBuf; i++) {
				transferDataFromCPU(vkGPU, (float*)(inputC + shift / sizeof(fftwf_complex)), &buffer[i], bufferSize[i]);
				shift += bufferSize[i];
			}
			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			res = initializeVulkanFFT(&app_forward, forward_configuration);
			if (res != VK_SUCCESS) return res;
			//forward_configuration.inverse = true;
			//res = initializeVulkanFFT(&app_inverse,forward_configuration);
			//Submit FFT+iFFT.
			//batch = 1;
			performVulkanFFT(vkGPU, &app_forward, batch);
			//totTime = performVulkanFFT(vkGPU, &app_inverse, batch);
			fftwf_complex* output_VkFFT = (fftwf_complex*)(malloc(sizeof(fftwf_complex) * dims[0] * dims[1] * dims[2]));

			//Transfer data from GPU using staging buffer.
			shift = 0;
			for (uint32_t i = 0; i < numBuf; i++) {
				transferDataToCPU(vkGPU, (float*)(output_VkFFT + shift / sizeof(fftwf_complex)), &buffer[i], bufferSize[i]);
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

						int N = (forward_configuration.inverse) ? dims[0] * dims[1] * dims[2] : 1;

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

				vkDestroyBuffer(vkGPU->device, buffer[i], NULL);
				vkDestroyBuffer(vkGPU->device, tempBuffer[i], NULL);
				vkFreeMemory(vkGPU->device, bufferDeviceMemory[i], NULL);
				vkFreeMemory(vkGPU->device, tempBufferDeviceMemory[i], NULL);

			}
			free(bufferSize);
			deleteVulkanFFT(&app_forward);
			free(inputC);
			fftw_destroy_plan(p);
			free(inputC_double);
			free(output_FFTW);
		}
	}
	return res;
}
#endif
VkResult launchVkFFT(VkGPU* vkGPU, uint32_t sample_id, bool file_output, FILE* output) {
	//Sample Vulkan project GPU initialization.
	VkResult res = VK_SUCCESS;
	//create instance - a connection between the application and the Vulkan library 
	res = createInstance(vkGPU, sample_id);
	if (res != VK_SUCCESS) {
		printf("Instance creation failed, error code: %d\n", res);
		return res;
	}
	//set up the debugging messenger 
	res = setupDebugMessenger(vkGPU);
	if (res != VK_SUCCESS) {
		printf("Debug messenger creation failed, error code: %d\n", res);
		return res;
	}
	//check if there are GPUs that support Vulkan and select one
	res = findPhysicalDevice(vkGPU);
	if (res != VK_SUCCESS) {
		printf("Physical device not found, error code: %d\n", res);
		return res;
	}
	//create logical device representation
	res = createDevice(vkGPU, sample_id);
	if (res != VK_SUCCESS) {
		printf("Device creation failed, error code: %d\n", res);
		return res;
	}
	//create fence for synchronization 
	res = createFence(vkGPU);
	if (res != VK_SUCCESS) {
		printf("Fence creation failed, error code: %d\n", res);
		return res;
	}
	//create a place, command buffer memory is allocated from
	res = createCommandPool(vkGPU);
	if (res != VK_SUCCESS) {
		printf("Fence creation failed, error code: %d\n", res);
		return res;
	}
	vkGetPhysicalDeviceProperties(vkGPU->physicalDevice, &vkGPU->physicalDeviceProperties);
	vkGetPhysicalDeviceMemoryProperties(vkGPU->physicalDevice, &vkGPU->physicalDeviceMemoryProperties);



	glslang_initialize_process();//compiler can be initialized before VkFFT
	uint32_t isCompilerInitialized = 1;


	//shaderc_compiler_t compiler = //shaderc_compiler_initialize();
	//shaderc_compile_options_t options = //shaderc_compile_options_initialize();
	////shaderc_compile_options_set_optimization_level(options, shaderc_optimization_level_performance);
	//char* code0 = (char*)malloc(sizeof(char) * 100000);

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
#if (VK_API_VERSION>10)
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
	case 10:
	{
		sample_10(vkGPU, sample_id, file_output, output, isCompilerInitialized);
		break;
	}
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
#if (VK_API_VERSION>10)
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
#endif
	}

	vkDestroyFence(vkGPU->device, vkGPU->fence, NULL);
	vkDestroyCommandPool(vkGPU->device, vkGPU->commandPool, NULL);
	vkDestroyDevice(vkGPU->device, NULL);
	DestroyDebugUtilsMessengerEXT(vkGPU, NULL);
	vkDestroyInstance(vkGPU->instance, NULL);
	glslang_finalize_process();//destroy compiler after use
	return VK_SUCCESS;
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
		printf("VkFFT v1.1.7 (31-01-2021). Author: Tolmachev Dmitrii\n");
		printf("	-h: print help\n");
		printf("	-devices: print the list of available GPU devices\n");
		printf("	-d X: select GPU device (default 0)\n");
		printf("	-o NAME: specify output file path\n");
		printf("	-vkfft X: launch VkFFT sample X (0-14):\n");
		printf("		0 - FFT + iFFT C2C benchmark 1D batched in single precision\n");
		printf("		1 - FFT + iFFT C2C benchmark 1D batched in double precision LUT\n");
#if (VK_API_VERSION>10)
		printf("		2 - FFT + iFFT C2C benchmark 1D batched in half precision\n");
#endif
		printf("		3 - FFT + iFFT C2C multidimensional benchmark in single precision\n");
		printf("		4 - FFT + iFFT C2C multidimensional benchmark in single precision, native zeropadding\n");
		printf("		5 - FFT + iFFT C2C benchmark 1D batched in single precision, no reshuffling\n");
		printf("		6 - FFT + iFFT R2C / C2R benchmark\n");
		printf("		7 - convolution example with identitiy kernel\n");
		printf("		8 - zeropadding convolution example with identitiy kernel\n");
		printf("		9 - batched convolution example with identitiy kernel\n");
		printf("		10 - multiple buffer(4 by default) split version of benchmark 0\n");
#ifdef USE_FFTW
#ifdef USE_cuFFT
		printf("		11 - VkFFT / cuFFT / FFTW C2C precision test in single precision\n");
		printf("		12 - VkFFT / cuFFT / FFTW C2C precision test in double precision\n");
#if (VK_API_VERSION>10)
		printf("		13 - VkFFT / cuFFT / FFTW C2C precision test in half precision\n");
#endif
		printf("		14 - VkFFT / FFTW C2C power 3 / 5 / 7 precision test in single precision\n");
#else
		printf("		11 - VkFFT / FFTW C2C precision test in single precision\n");
		printf("		12 - VkFFT / FFTW C2C precision test in double precision\n");
#if (VK_API_VERSION>10)
		printf("		13 - VkFFT / FFTW C2C precision test in half precision\n");
#endif
		printf("		14 - VkFFT / FFTW C2C power 3 / 5 / 7 precision test in single precision\n");
#endif
#endif
#ifdef USE_cuFFT
		printf("	-cufft X: launch cuFFT sample X (0-3):\n");
		printf("		0 - FFT + iFFT C2C benchmark 1D batched in single precision\n");
		printf("		1 - FFT + iFFT C2C benchmark 1D batched in double precision LUT\n");
		printf("		2 - FFT + iFFT C2C benchmark 1D batched in half precision\n");
		printf("		3 - FFT + iFFT C2C multidimensional benchmark in single precision\n");
		printf("	-test: (or no -vkfft and -cufft keys) run vkfft benchmarks 0-6 and cufft benchmarks 0-3\n");
#else
		printf("	-test: (or no -vkfft and -cufft keys) run vkfft benchmarks 0-6\n");
		printf("	-cufft command is disabled\n");
#endif
		return 0;
	}
	if (findFlag(argv, argv + argc, "-devices"))
	{
		//print device list
		VkResult res = devices_list();
		return res;
	}
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
			VkResult res = launchVkFFT(&vkGPU, sample_id, file_output, output);
			if (res != VK_SUCCESS) return res;
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
			}
		}
		else {
			printf("No cuFFT script is selected with -cufft flag\n");
			return 1;
		}
	}
#endif
	if ((findFlag(argv, argv + argc, "-test")) || ((!findFlag(argv, argv + argc, "-cufft")) && (!findFlag(argv, argv + argc, "-vkfft"))))
	{
		if (output == NULL) {
			file_output = true;
			output = fopen("result.txt", "a");
		}
		for (uint32_t i = 0; i < 7; i++) {
			if ((i == 2) && (VK_API_VERSION == 10)) i++;
			VkResult res = launchVkFFT(&vkGPU, i, file_output, output);
			if (res != VK_SUCCESS) return res;
		}
#ifdef USE_cuFFT
		launch_benchmark_cuFFT_single(file_output, output);
		launch_benchmark_cuFFT_double(file_output, output);
		launch_benchmark_cuFFT_half(file_output, output);
		launch_benchmark_cuFFT_single_3d(file_output, output);
#endif
	}
	return VK_SUCCESS;
}
