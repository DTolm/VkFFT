#include <vector>
#include <memory>
#include <string.h>
#include <chrono>
#include <thread>
#include <iostream>
#include <algorithm>
#include "vkFFT.h"
#include "vulkan/vulkan.h"

#ifdef USE_cuFFT
#include "benchmark_cuFFT.h"
#include "benchmark_cuFFT_double.h"
#include "benchmark_cuFFT_half.h"
#endif

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

VkInstance instance = {};
VkDebugReportCallbackEXT debugReportCallback = {};
VkPhysicalDevice physicalDevice = {};
VkPhysicalDeviceProperties physicalDeviceProperties = {};
VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties = {};
VkDevice device = {};
VkDebugUtilsMessengerEXT debugMessenger = {};
uint32_t queueFamilyIndex = {};
std::vector<const char*> enabledDeviceExtensions;
VkQueue queue = {};
VkCommandPool commandPool = {};
VkFence fence = {};

const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};
static VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallbackFn(
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
}

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	}
	else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr) {
		func(instance, debugMessenger, pAllocator);
	}
}
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
	std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

	return VK_FALSE;
}


void setupDebugMessenger() {
	if (!enableValidationLayers) return;

	VkDebugUtilsMessengerCreateInfoEXT createInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
	createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
	createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
	createInfo.pfnUserCallback = debugCallback;

	if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
		throw std::runtime_error("failed to set up debug messenger");
	}
}

std::vector<const char*> getRequiredExtensions(uint32_t sample_id) {
	std::vector<const char*> extensions;

	if (enableValidationLayers) {
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}
	switch (sample_id) {
	case 2:
		extensions.push_back("VK_KHR_get_physical_device_properties2");
		break;
	default:
		break;
	}


	return extensions;
}

bool checkValidationLayerSupport() {
	uint32_t layerCount;
	vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

	std::vector<VkLayerProperties> availableLayers(layerCount);
	vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

	for (const char* layerName : validationLayers) {
		bool layerFound = false;

		for (const auto& layerProperties : availableLayers) {
			if (strcmp(layerName, layerProperties.layerName) == 0) {
				layerFound = true;
				break;
			}
		}

		if (!layerFound) {
			return false;
		}
	}

	return true;
}

void createInstance(uint32_t sample_id) {
	if (enableValidationLayers && !checkValidationLayerSupport()) {
		throw std::runtime_error("validation layers creation failed");
	}

	VkApplicationInfo applicationInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
	applicationInfo.pApplicationName = "VkFFT";
	applicationInfo.applicationVersion = 1.0;
	applicationInfo.pEngineName = "VkFFT";
	applicationInfo.engineVersion = 1.0;
	switch (sample_id) {
	case 2:
		applicationInfo.apiVersion = VK_API_VERSION_1_1;
		break;
	default:
		applicationInfo.apiVersion = VK_API_VERSION_1_0;
		break;
	}

	VkInstanceCreateInfo createInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
	createInfo.flags = 0;
	createInfo.pApplicationInfo = &applicationInfo;

	auto extensions = getRequiredExtensions(sample_id);
	createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
	createInfo.ppEnabledExtensionNames = extensions.data();

	VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
	if (enableValidationLayers) {
		createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
		createInfo.ppEnabledLayerNames = validationLayers.data();
		debugCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		debugCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		debugCreateInfo.pfnUserCallback = debugCallback;
		createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
	}
	else {
		createInfo.enabledLayerCount = 0;

		createInfo.pNext = nullptr;
	}

	if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
		throw std::runtime_error("instance creation failed");
	}


}

void findPhysicalDevice(uint32_t deviceID) {

	uint32_t deviceCount;
	vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
	if (deviceCount == 0) {
		throw std::runtime_error("device with vulkan support not found");
	}

	std::vector<VkPhysicalDevice> devices(deviceCount);
	vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

	physicalDevice = devices[deviceID];

}
void devices_list() {
	VkInstance local_instance = {};
	VkInstanceCreateInfo createInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
	createInfo.flags = 0;
	createInfo.pApplicationInfo = NULL;
	VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
	createInfo.enabledLayerCount = 0;
	createInfo.enabledExtensionCount = 0;
	createInfo.pNext = NULL;
	if (vkCreateInstance(&createInfo, nullptr, &local_instance) != VK_SUCCESS) {
		throw std::runtime_error("instance creation failed");
	}

	uint32_t deviceCount;
	vkEnumeratePhysicalDevices(local_instance, &deviceCount, NULL);
	if (deviceCount == 0) {
		throw std::runtime_error("device with vulkan support not found");
	}

	std::vector<VkPhysicalDevice> devices(deviceCount);
	vkEnumeratePhysicalDevices(local_instance, &deviceCount, devices.data());
	for (uint32_t i = 0; i < devices.size(); i++) {
		VkPhysicalDeviceProperties device_properties;
		vkGetPhysicalDeviceProperties(devices[i], &device_properties);
		printf("Device id: %d name: %s API:%d.%d.%d\n", i, device_properties.deviceName, (device_properties.apiVersion >> 22), ((device_properties.apiVersion >> 12) & 0x3ff), (device_properties.apiVersion & 0xfff));
	}
	vkDestroyInstance(local_instance, NULL);
}
uint32_t getComputeQueueFamilyIndex() {
	uint32_t queueFamilyCount;

	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, NULL);

	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

	uint32_t i = 0;
	for (; i < queueFamilies.size(); ++i) {
		VkQueueFamilyProperties props = queueFamilies[i];

		if (props.queueCount > 0 && (props.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
			break;
		}
	}

	if (i == queueFamilies.size()) {
		throw std::runtime_error("queue family creation failed");
	}

	return i;
}

void createDevice(uint32_t sample_id) {

	VkDeviceQueueCreateInfo queueCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
	queueFamilyIndex = getComputeQueueFamilyIndex();
	queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
	queueCreateInfo.queueCount = 1;
	float queuePriorities = 1.0;
	queueCreateInfo.pQueuePriorities = &queuePriorities;
	VkDeviceCreateInfo deviceCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
	VkPhysicalDeviceFeatures deviceFeatures = {};
	switch (sample_id) {
	case 1: {
		deviceFeatures.shaderFloat64 = true;
		deviceCreateInfo.enabledExtensionCount = enabledDeviceExtensions.size();
		deviceCreateInfo.ppEnabledExtensionNames = enabledDeviceExtensions.data();
		deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
		deviceCreateInfo.queueCreateInfoCount = 1;
		deviceCreateInfo.pEnabledFeatures = &deviceFeatures;
		vkCreateDevice(physicalDevice, &deviceCreateInfo, NULL, &device);
		vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
		break;
	}
	case 2: {
		VkPhysicalDeviceFeatures2 deviceFeatures2 = {};
		VkPhysicalDeviceShaderFloat16Int8Features shaderFloat16 = {};
		shaderFloat16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
		shaderFloat16.shaderFloat16 = true;
		shaderFloat16.shaderInt8 = true;
		deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
		deviceFeatures2.pNext = &shaderFloat16;
		deviceFeatures2.features = deviceFeatures;
		vkGetPhysicalDeviceFeatures2(physicalDevice, &deviceFeatures2);
		deviceCreateInfo.pNext = &deviceFeatures2;
		enabledDeviceExtensions.push_back("VK_KHR_16bit_storage");
		deviceCreateInfo.enabledExtensionCount = enabledDeviceExtensions.size();
		deviceCreateInfo.ppEnabledExtensionNames = enabledDeviceExtensions.data();
		deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
		deviceCreateInfo.queueCreateInfoCount = 1;
		deviceCreateInfo.pEnabledFeatures = NULL;
		vkCreateDevice(physicalDevice, &deviceCreateInfo, NULL, &device);
		vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
		break;
	}
	default: {
		deviceCreateInfo.enabledExtensionCount = enabledDeviceExtensions.size();
		deviceCreateInfo.ppEnabledExtensionNames = enabledDeviceExtensions.data();
		deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
		deviceCreateInfo.queueCreateInfoCount = 1;
		deviceCreateInfo.pEnabledFeatures = NULL;
		deviceCreateInfo.pEnabledFeatures = &deviceFeatures;
		vkCreateDevice(physicalDevice, &deviceCreateInfo, NULL, &device);
		vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
		break;
	}
	}
}


uint32_t findMemoryType(uint32_t memoryTypeBits, VkMemoryPropertyFlags properties) {
	VkPhysicalDeviceMemoryProperties memoryProperties = {};

	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

	for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
		if ((memoryTypeBits & (1 << i)) &&
			((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties))
			return i;
	}
	return -1;
}
void allocateFFTBuffer(VkBuffer* buffer, VkDeviceMemory* deviceMemory, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags propertyFlags, VkDeviceSize size) {
	uint32_t queueFamilyIndices;
	VkBufferCreateInfo bufferCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	bufferCreateInfo.queueFamilyIndexCount = 1;
	bufferCreateInfo.pQueueFamilyIndices = &queueFamilyIndices;
	bufferCreateInfo.size = size;
	bufferCreateInfo.usage = usageFlags;
	vkCreateBuffer(device, &bufferCreateInfo, NULL, buffer);
	VkMemoryRequirements memoryRequirements = {};
	vkGetBufferMemoryRequirements(device, buffer[0], &memoryRequirements);
	VkMemoryAllocateInfo memoryAllocateInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
	memoryAllocateInfo.allocationSize = memoryRequirements.size;
	memoryAllocateInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, propertyFlags);
	vkAllocateMemory(device, &memoryAllocateInfo, NULL, deviceMemory);
	vkBindBufferMemory(device, buffer[0], deviceMemory[0], 0);
}
void transferDataFromCPU(void* arr, VkBuffer* buffer, VkDeviceSize bufferSize) {
	VkDeviceSize stagingBufferSize = bufferSize;
	VkBuffer stagingBuffer = {};
	VkDeviceMemory stagingBufferMemory = {};
	allocateFFTBuffer(&stagingBuffer, &stagingBufferMemory, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBufferSize);

	void* data;
	vkMapMemory(device, stagingBufferMemory, 0, stagingBufferSize, 0, &data);
	memcpy(data, arr, stagingBufferSize);
	vkUnmapMemory(device, stagingBufferMemory);
	VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	commandBufferAllocateInfo.commandPool = commandPool;
	commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferAllocateInfo.commandBufferCount = 1;
	VkCommandBuffer commandBuffer = {};
	vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer);
	VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
	VkBufferCopy copyRegion = {};
	copyRegion.srcOffset = 0;
	copyRegion.dstOffset = 0;
	copyRegion.size = stagingBufferSize;
	vkCmdCopyBuffer(commandBuffer, stagingBuffer, buffer[0], 1, &copyRegion);
	vkEndCommandBuffer(commandBuffer);
	VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;
	vkQueueSubmit(queue, 1, &submitInfo, fence);
	vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000);
	vkResetFences(device, 1, &fence);
	vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
	vkDestroyBuffer(device, stagingBuffer, NULL);
	vkFreeMemory(device, stagingBufferMemory, NULL);
}
void transferDataToCPU(void* arr, VkBuffer* buffer, VkDeviceSize bufferSize) {
	VkDeviceSize stagingBufferSize = bufferSize;
	VkBuffer stagingBuffer = {};
	VkDeviceMemory stagingBufferMemory = {};
	allocateFFTBuffer(&stagingBuffer, &stagingBufferMemory, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBufferSize);


	VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	commandBufferAllocateInfo.commandPool = commandPool;
	commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferAllocateInfo.commandBufferCount = 1;
	VkCommandBuffer commandBuffer = {};
	vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer);
	VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
	VkBufferCopy copyRegion = {};
	copyRegion.srcOffset = 0;
	copyRegion.dstOffset = 0;
	copyRegion.size = stagingBufferSize;
	vkCmdCopyBuffer(commandBuffer, buffer[0], stagingBuffer, 1, &copyRegion);
	vkEndCommandBuffer(commandBuffer);
	VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;
	vkQueueSubmit(queue, 1, &submitInfo, fence);
	vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000);
	vkResetFences(device, 1, &fence);
	vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
	void* data;
	vkMapMemory(device, stagingBufferMemory, 0, stagingBufferSize, 0, &data);
	memcpy(arr, data, stagingBufferSize);
	vkUnmapMemory(device, stagingBufferMemory);
	vkDestroyBuffer(device, stagingBuffer, NULL);
	vkFreeMemory(device, stagingBufferMemory, NULL);
}

void performVulkanFFT(VkFFTApplication* app, uint32_t batch) {
	VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	commandBufferAllocateInfo.commandPool = commandPool;
	commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferAllocateInfo.commandBufferCount = 1;
	VkCommandBuffer commandBuffer = {};
	vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer);
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
	vkQueueSubmit(queue, 1, &submitInfo, fence);
	vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000);
	auto timeEnd = std::chrono::system_clock::now();
	double totTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001;
	printf("Pure submit execution time per batch: %.3f ms\n", totTime / batch);
	vkResetFences(device, 1, &fence);
	vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}
float performVulkanFFTiFFT(VkFFTApplication* app_forward, VkFFTApplication* app_inverse, uint32_t batch) {
	VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	commandBufferAllocateInfo.commandPool = commandPool;
	commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferAllocateInfo.commandBufferCount = 1;
	VkCommandBuffer commandBuffer = {};
	vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer);
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
	auto timeSubmit = std::chrono::system_clock::now();
	vkQueueSubmit(queue, 1, &submitInfo, fence);
	vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000);
	auto timeEnd = std::chrono::system_clock::now();
	float totTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001;
	vkResetFences(device, 1, &fence);
	vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
	return totTime / batch;
}
int launchVkFFT(uint32_t device_id, uint32_t sample_id, bool file_output, FILE* output) {
	//Sample Vulkan project GPU initialization.
	createInstance(sample_id);
	setupDebugMessenger();
	findPhysicalDevice(device_id);
	createDevice(sample_id);

	VkFenceCreateInfo fenceCreateInfo = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
	fenceCreateInfo.flags = 0;
	vkCreateFence(device, &fenceCreateInfo, NULL, &fence);
	VkCommandPoolCreateInfo commandPoolCreateInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
	commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;
	vkCreateCommandPool(device, &commandPoolCreateInfo, NULL, &commandPool);
	vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &physicalDeviceMemoryProperties);

	switch (sample_id) {
	case 0:
	{
		//0 - FFT + iFFT C2C benchmark. 
		const int num_benchmark_samples = 26;
		const int num_runs = 7;

		uint32_t benchmark_dimensions[num_benchmark_samples][4] = { {1024, 1024, 1, 2},
		{32,32,1,2},{64,64,1,2},{256,256,1,2},{1024, 256, 1, 2}, {512, 512, 1, 2},  {1024, 1024, 1, 2} , {4096, 1024, 1, 2}, {2048, 2048, 1, 2}, {4096, 4096, 1, 2},
		{64,64,64,3}, {128,128,128, 3}, {256,256,256,3}, {512, 256, 64, 3}, {1024, 1024, 64, 3}, {4096, 256, 32, 3},  {2048, 256, 256, 3},{4096, 4096, 8, 3},
		{(uint32_t)pow(2,15), 64, 1, 2}, {(uint32_t)pow(2,16), 64, 1, 2}, {(uint32_t)pow(2,17), 64, 1, 2}, {(uint32_t)pow(2,18), 64, 1, 2},  {(uint32_t)pow(2,20), 64, 1, 2},  {(uint32_t)pow(2,22), 64, 1, 2},
		{(uint32_t)pow(2,13), (uint32_t)pow(2,13), 1, 2},{(uint32_t)pow(2,14), (uint32_t)pow(2,14), 1, 2},
		};
		double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
		//memory allocated on the CPU once, makes benchmark completion faster + avoids performance issues connected to frequent allocation/deallocation.
		float* buffer_input = (float*)malloc((uint64_t)4 * 2 * 4096 * 4096 * 8 * 2);
		for (uint64_t i = 0; i < 2 * 4096 * 4096 * 2 * 8; i++) {
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

				//PARAMETERS THAT CAN BE ADJUSTED FOR SPECIFIC GPU's - this configuration is by no means final form
				switch (physicalDeviceProperties.vendorID) {
				case 0x10DE://NVIDIA
					forward_configuration.coalescedMemory = 32;//the coalesced memory is equal to 32 bytes between L2 and VRAM. 
					forward_configuration.useLUT = false;
					if (forward_configuration.size[1] > 512)
						forward_configuration.registerBoost = 4;
					else
						forward_configuration.registerBoost = 1;
					break;
				case 0x8086://INTEL
					forward_configuration.coalescedMemory = 64;
					forward_configuration.useLUT = true;
					forward_configuration.registerBoost = 1;
					break;
				case 0x13B5://AMD
					forward_configuration.coalescedMemory = 32;
					forward_configuration.useLUT = false;
					if (forward_configuration.size[1] > 512)
						forward_configuration.registerBoost = 4;
					else
						forward_configuration.registerBoost = 1;
					break;
				default:
					forward_configuration.coalescedMemory = 64;
					forward_configuration.useLUT = false;
					forward_configuration.registerBoost = 1;
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
				forward_configuration.device = &device;
				forward_configuration.queue = &queue; //to allocate memory for LUT, we have to pass a queue, fence, commandPool and physicalDevice pointers 
				forward_configuration.fence = &fence;
				forward_configuration.commandPool = &commandPool;
				forward_configuration.physicalDevice = &physicalDevice;
				//Custom path to the floder with shaders, default is "shaders/". Max length - 256 chars.
				if (sizeof(SHADER_DIR) > 255) {
					printf("SHADER_DIR length must be <256\n");
					return 1;
				}
				sprintf(forward_configuration.shaderPath, SHADER_DIR);

				//Allocate buffer for the input data.
				VkDeviceSize bufferSize = ((uint64_t)forward_configuration.coordinateFeatures) * sizeof(float) * 2 * forward_configuration.size[0] * forward_configuration.size[1] * forward_configuration.size[2];;
				VkBuffer buffer = {};
				VkDeviceMemory bufferDeviceMemory = {};
				VkBuffer tempBuffer = {};//temp buffer, needed to unshuffle four-step fft
				VkDeviceMemory tempBufferDeviceMemory = {};
				allocateFFTBuffer(&buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
				allocateFFTBuffer(&tempBuffer, &tempBufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);

				forward_configuration.buffer = &buffer;
				forward_configuration.tempBuffer = &tempBuffer;
				forward_configuration.isInputFormatted = false; //set to true if input is a different buffer, so it can have zeropadding/R2C added  
				forward_configuration.inputBuffer = &buffer; //you can specify first buffer to read data from to be different from the buffer FFT is performed on. FFT is still in-place on the second buffer, this is here just for convenience.
				forward_configuration.isOutputFormatted = false;//set to true if output is a different buffer, so it can have zeropadding/C2R automatically removed
				forward_configuration.outputBuffer = &buffer;
				forward_configuration.bufferSize = &bufferSize;
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
				transferDataFromCPU(buffer_input, &buffer, bufferSize);
				//free(buffer_input);

				//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
				initializeVulkanFFT(&app_forward, forward_configuration);
				initializeVulkanFFT(&app_inverse, inverse_configuration);
				//Submit FFT+iFFT.
				uint32_t batch = ((4096 * 1024.0 * 1024.0) / bufferSize > 1000) ? 1000 : (4096 * 1024.0 * 1024.0) / bufferSize;
				if (batch == 0) batch = 1;

				float totTime = performVulkanFFTiFFT(&app_forward, &app_inverse, batch);

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
						if (file_output)
							fprintf(output, "VkFFT System: %dx%dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f batch: %d benchmark: %d\n", benchmark_dimensions[n][0], benchmark_dimensions[n][1], benchmark_dimensions[n][2], bufferSize / 1024 / 1024, avg_time, std_error, batch, (int)(((double)bufferSize / 1024) / avg_time));
						printf("VkFFT System: %dx%dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f batch: %d benchmark: %d\n", benchmark_dimensions[n][0], benchmark_dimensions[n][1], benchmark_dimensions[n][2], bufferSize / 1024 / 1024, avg_time, std_error, batch, (int)(((double)bufferSize / 1024) / avg_time));
						benchmark_result += ((double)bufferSize / 1024) / avg_time;
					}


				}

				vkDestroyBuffer(device, buffer, NULL);
				vkFreeMemory(device, bufferDeviceMemory, NULL);
				vkDestroyBuffer(device, tempBuffer, NULL);
				vkFreeMemory(device, tempBufferDeviceMemory, NULL);
				deleteVulkanFFT(&app_forward);
				deleteVulkanFFT(&app_inverse);
			}
		}
		free(buffer_input);
		benchmark_result /= (num_benchmark_samples - 1);

		if (file_output) {
			fprintf(output, "Benchmark score VkFFT: %d\n", (int)(benchmark_result));
			fprintf(output, "Device name: %s API:%d.%d.%d\n", physicalDeviceProperties.deviceName, (physicalDeviceProperties.apiVersion >> 22), ((physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (physicalDeviceProperties.apiVersion & 0xfff));
		}
		printf("Benchmark score VkFFT: %d\n", (int)(benchmark_result));
		printf("Device name: %s API:%d.%d.%d\n", physicalDeviceProperties.deviceName, (physicalDeviceProperties.apiVersion >> 22), ((physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (physicalDeviceProperties.apiVersion & 0xfff));
		vkDestroyFence(device, fence, NULL);
		vkDestroyCommandPool(device, commandPool, NULL);
		vkDestroyDevice(device, NULL);
		DestroyDebugUtilsMessengerEXT(instance, debugMessenger, NULL);
		vkDestroyInstance(instance, NULL);
		break;
	}
	case 1:
	{
		//1 - FFT + iFFT C2C benchmark in double precision. 
		const int num_benchmark_samples = 23;
		const int num_runs = 7;

		uint32_t benchmark_dimensions[num_benchmark_samples][4] = { {1024, 1024, 1, 2},
		{32,32,1,2},{64,64,1,2},{256,256,1,2},{1024, 256, 1, 2}, {512, 512, 1, 2},  {1024, 1024, 1, 2} , {4096, 1024, 1, 2}, {2048, 2048, 1, 2}, {4096, 4096, 1, 2},
		{64,64,64,3}, {128,128,128, 3}, {256,256,256,3}, {512, 256, 64, 3}, {1024, 1024, 64, 3}, {4096, 256, 32, 3},  {2048, 256, 256, 3},{4096, 4096, 8, 3},
		{(uint32_t)pow(2,15), 64, 1, 2}, {(uint32_t)pow(2,16), 64, 1, 2}, {(uint32_t)pow(2,17), 64, 1, 2}, {(uint32_t)pow(2,18), 64, 1, 2},  {(uint32_t)pow(2,20), 64, 1, 2},
		};
		double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
		//memory allocated on the CPU once, makes benchmark completion faster + avoids performance issues connected to frequent allocation/deallocation.
		double* buffer_input = (double*)malloc((uint64_t)8 * 2 * 4096 * 4096 * 8 * 2);
		for (uint64_t i = 0; i < 2 * 4096 * 4096 * 2 * 8; i++) {
			buffer_input[i] = 2 * ((double)rand()) / RAND_MAX - 1.0;
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

				//PARAMETERS THAT CAN BE ADJUSTED FOR SPECIFIC GPU's - this configuration is by no means final form
				switch (physicalDeviceProperties.vendorID) {
				case 0x10DE://NVIDIA
					forward_configuration.coalescedMemory = 32;//the coalesced memory is equal to 32 bytes between L2 and VRAM. 
					forward_configuration.useLUT = true;
					if (forward_configuration.size[1] > 512)
						forward_configuration.registerBoost = 1;//registerBoost is not implemented for double yet.
					else
						forward_configuration.registerBoost = 1;
					break;
				case 0x8086://INTEL
					forward_configuration.coalescedMemory = 64;
					forward_configuration.useLUT = true;
					forward_configuration.registerBoost = 1;
					break;
				case 0x13B5://AMD
					forward_configuration.coalescedMemory = 32;
					forward_configuration.useLUT = true;
					if (forward_configuration.size[1] > 512)
						forward_configuration.registerBoost = 1;
					else
						forward_configuration.registerBoost = 1;
					break;
				default:
					forward_configuration.coalescedMemory = 64;
					forward_configuration.useLUT = true;
					forward_configuration.registerBoost = 1;
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
				forward_configuration.device = &device;
				forward_configuration.queue = &queue; //to allocate memory for LUT, we have to pass a queue, fence, commandPool and physicalDevice pointers 
				forward_configuration.fence = &fence;
				forward_configuration.commandPool = &commandPool;
				forward_configuration.physicalDevice = &physicalDevice;
				//forward_configuration.useLUT = true;
				forward_configuration.doublePrecision = true;
				//Custom path to the floder with shaders, default is "shaders/". Max length - 256 chars.
				if (sizeof(SHADER_DIR) > 255) {
					printf("SHADER_DIR length must be <256\n");
					return 1;
				}
				sprintf(forward_configuration.shaderPath, SHADER_DIR);

				//Allocate buffer for the input data.
				VkDeviceSize bufferSize = ((uint64_t)forward_configuration.coordinateFeatures) * sizeof(double) * 2 * forward_configuration.size[0] * forward_configuration.size[1] * forward_configuration.size[2];;
				VkBuffer buffer = {};
				VkDeviceMemory bufferDeviceMemory = {};
				VkBuffer tempBuffer = {};//temp buffer, needed to unshuffle four-step fft
				VkDeviceMemory tempBufferDeviceMemory = {};
				allocateFFTBuffer(&buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
				allocateFFTBuffer(&tempBuffer, &tempBufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);

				forward_configuration.buffer = &buffer;
				forward_configuration.tempBuffer = &tempBuffer;
				forward_configuration.isInputFormatted = false; //set to true if input is a different buffer, so it can have zeropadding/R2C added  
				forward_configuration.inputBuffer = &buffer; //you can specify first buffer to read data from to be different from the buffer FFT is performed on. FFT is still in-place on the second buffer, this is here just for convenience.
				forward_configuration.isOutputFormatted = false;//set to true if output is a different buffer, so it can have zeropadding/C2R automatically removed
				forward_configuration.outputBuffer = &buffer;
				forward_configuration.bufferSize = &bufferSize;
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
				transferDataFromCPU(buffer_input, &buffer, bufferSize);
				//free(buffer_input);

				//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
				initializeVulkanFFT(&app_forward, forward_configuration);
				initializeVulkanFFT(&app_inverse, inverse_configuration);
				//Submit FFT+iFFT.
				uint32_t batch = ((4096 * 1024.0 * 1024.0) / bufferSize / 5 > 1000) ? 1000 : (4096 * 1024.0 * 1024.0) / bufferSize / 5;
				if (batch == 0) batch = 1;

				float totTime = performVulkanFFTiFFT(&app_forward, &app_inverse, batch);

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
						if (file_output)
							fprintf(output, "VkFFT System: %dx%dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f batch: %d benchmark: %d\n", benchmark_dimensions[n][0], benchmark_dimensions[n][1], benchmark_dimensions[n][2], bufferSize / 1024 / 1024, avg_time, std_error, batch, (int)(((double)bufferSize * sizeof(float) / sizeof(double) / 1024) / avg_time));

						printf("VkFFT System: %dx%dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f batch: %d benchmark: %d\n", benchmark_dimensions[n][0], benchmark_dimensions[n][1], benchmark_dimensions[n][2], bufferSize / 1024 / 1024, avg_time, std_error, batch, (int)(((double)bufferSize * sizeof(float) / sizeof(double) / 1024) / avg_time));
						benchmark_result += ((double)bufferSize * sizeof(float) / sizeof(double) / 1024) / avg_time;
					}


				}

				vkDestroyBuffer(device, buffer, NULL);
				vkFreeMemory(device, bufferDeviceMemory, NULL);
				vkDestroyBuffer(device, tempBuffer, NULL);
				vkFreeMemory(device, tempBufferDeviceMemory, NULL);
				deleteVulkanFFT(&app_forward);
				deleteVulkanFFT(&app_inverse);
			}
		}
		free(buffer_input);
		benchmark_result /= (num_benchmark_samples - 1);
		if (file_output) {
			fprintf(output, "Benchmark score VkFFT: %d\n", (int)(benchmark_result));
			fprintf(output, "Device name: %s API:%d.%d.%d\n", physicalDeviceProperties.deviceName, (physicalDeviceProperties.apiVersion >> 22), ((physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (physicalDeviceProperties.apiVersion & 0xfff));
		}
		printf("Benchmark score VkFFT: %d\n", (int)(benchmark_result));
		printf("Device name: %s API:%d.%d.%d\n", physicalDeviceProperties.deviceName, (physicalDeviceProperties.apiVersion >> 22), ((physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (physicalDeviceProperties.apiVersion & 0xfff));

		vkDestroyFence(device, fence, NULL);
		vkDestroyCommandPool(device, commandPool, NULL);
		vkDestroyDevice(device, NULL);
		DestroyDebugUtilsMessengerEXT(instance, debugMessenger, NULL);
		vkDestroyInstance(instance, NULL);
		break;
	}
	case 2:
	{
		//2 - FFT + iFFT C2C benchmark in half precision. 
		const int num_benchmark_samples = 26;
		const int num_runs = 7;

		uint32_t benchmark_dimensions[num_benchmark_samples][4] = { {1024, 1024, 1, 2},
		{32,32,1,2},{64,64,1,2},{256,256,1,2},{1024, 256, 1, 2}, {512, 512, 1, 2},  {1024, 1024, 1, 2} , {4096, 1024, 1, 2}, {2048, 2048, 1, 2}, {4096, 4096, 1, 2},
		{64,64,64,3}, {128,128,128, 3}, {256,256,256,3}, {512, 256, 64, 3}, {1024, 1024, 64, 3}, {4096, 256, 32, 3},  {2048, 256, 256, 3},{4096, 4096, 8, 3},
		{(uint32_t)pow(2,15), 64, 1, 2}, {(uint32_t)pow(2,16), 64, 1, 2}, {(uint32_t)pow(2,17), 64, 1, 2}, {(uint32_t)pow(2,18), 64, 1, 2},  {(uint32_t)pow(2,20), 64, 1, 2},  {(uint32_t)pow(2,22), 64, 1, 2},
		{(uint32_t)pow(2,13), (uint32_t)pow(2,13), 1, 2},{(uint32_t)pow(2,14), (uint32_t)pow(2,14), 1, 2},
		};
		double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
		//memory allocated on the CPU once, makes benchmark completion faster + avoids performance issues connected to frequent allocation/deallocation.
		//double* buffer_input = (double*)malloc((uint64_t)2 * 8 * 1024 * 1024 * 8 * 2);
		//for (uint64_t i = 0; i < 2 * 1024 * 1024 * 8 * 2; i++) {
			//buffer_input[i] = 2 * ((double)rand()) / RAND_MAX - 1.0;
		//}
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

				//PARAMETERS THAT CAN BE ADJUSTED FOR SPECIFIC GPU's - this configuration is by no means final form
				switch (physicalDeviceProperties.vendorID) {
				case 0x10DE://NVIDIA
					forward_configuration.coalescedMemory = 64;//have to set coalesce more, as calculations are still float, while uploads are half.
					forward_configuration.useLUT = false;
					if (forward_configuration.size[1] > 512)
						forward_configuration.registerBoost = 1;//registerBoost is less efficient for half precision, because computations are still in float
					else
						forward_configuration.registerBoost = 1;
					break;
				case 0x8086://INTEL
					forward_configuration.coalescedMemory = 128;
					forward_configuration.useLUT = true;
					forward_configuration.registerBoost = 1;
					break;
				case 0x13B5://AMD
					forward_configuration.coalescedMemory = 64;
					forward_configuration.useLUT = false;
					if (forward_configuration.size[1] > 512)
						forward_configuration.registerBoost = 1;
					else
						forward_configuration.registerBoost = 1;
					break;
				default:
					forward_configuration.coalescedMemory = 64;
					forward_configuration.useLUT = false;
					forward_configuration.registerBoost = 1;
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
				forward_configuration.device = &device;
				forward_configuration.queue = &queue; //to allocate memory for LUT, we have to pass a queue, fence, commandPool and physicalDevice pointers 
				forward_configuration.fence = &fence;
				forward_configuration.commandPool = &commandPool;
				forward_configuration.physicalDevice = &physicalDevice;
				//forward_configuration.useLUT = false;
				forward_configuration.halfPrecision = true;
				//Custom path to the floder with shaders, default is "shaders/". Max length - 256 chars.
				if (sizeof(SHADER_DIR) > 255) {
					printf("SHADER_DIR length must be <256\n");
					return 1;
				}
				sprintf(forward_configuration.shaderPath, SHADER_DIR);

				//Allocate buffer for the input data.
				VkDeviceSize bufferSize = ((uint64_t)forward_configuration.coordinateFeatures) * 2 * 2 * forward_configuration.size[0] * forward_configuration.size[1] * forward_configuration.size[2];;
				VkBuffer buffer = {};
				VkDeviceMemory bufferDeviceMemory = {};
				VkBuffer tempBuffer = {};//temp buffer, needed to unshuffle four-step fft
				VkDeviceMemory tempBufferDeviceMemory = {};
				allocateFFTBuffer(&buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
				allocateFFTBuffer(&tempBuffer, &tempBufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);

				forward_configuration.buffer = &buffer;
				forward_configuration.tempBuffer = &tempBuffer;
				forward_configuration.isInputFormatted = false; //set to true if input is a different buffer, so it can have zeropadding/R2C added  
				forward_configuration.inputBuffer = &buffer; //you can specify first buffer to read data from to be different from the buffer FFT is performed on. FFT is still in-place on the second buffer, this is here just for convenience.
				forward_configuration.isOutputFormatted = false;//set to true if output is a different buffer, so it can have zeropadding/C2R automatically removed
				forward_configuration.outputBuffer = &buffer;
				forward_configuration.bufferSize = &bufferSize;
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
				//transferDataFromCPU(buffer_input, &buffer, bufferSize);
				//free(buffer_input);

				//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
				initializeVulkanFFT(&app_forward, forward_configuration);
				initializeVulkanFFT(&app_inverse, inverse_configuration);
				//Submit FFT+iFFT.
				uint32_t batch = ((4096 * 1024.0 * 1024.0) / bufferSize > 1000) ? 1000 : (4096 * 1024.0 * 1024.0) / bufferSize;
				if (batch == 0) batch = 1;

				float totTime = performVulkanFFTiFFT(&app_forward, &app_inverse, batch);

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
						if (file_output)
							fprintf(output, "VkFFT System: %dx%dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f batch: %d benchmark: %d\n", benchmark_dimensions[n][0], benchmark_dimensions[n][1], benchmark_dimensions[n][2], bufferSize / 1024 / 1024, avg_time, std_error, batch, (int)(((double)bufferSize * sizeof(float) / 2 / 1024) / avg_time));

						printf("VkFFT System: %dx%dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f batch: %d benchmark: %d\n", benchmark_dimensions[n][0], benchmark_dimensions[n][1], benchmark_dimensions[n][2], bufferSize / 1024 / 1024, avg_time, std_error, batch, (int)(((double)bufferSize * sizeof(float) / 2 / 1024) / avg_time));
						benchmark_result += ((double)bufferSize * sizeof(float) / 2 / 1024) / avg_time;
					}


				}

				vkDestroyBuffer(device, buffer, NULL);
				vkFreeMemory(device, bufferDeviceMemory, NULL);
				vkDestroyBuffer(device, tempBuffer, NULL);
				vkFreeMemory(device, tempBufferDeviceMemory, NULL);
				deleteVulkanFFT(&app_forward);
				deleteVulkanFFT(&app_inverse);
			}
		}
		//free(buffer_input);
		benchmark_result /= (num_benchmark_samples - 1);
		if (file_output) {
			fprintf(output, "Benchmark score VkFFT: %d\n", (int)(benchmark_result));
			fprintf(output, "Device name: %s API:%d.%d.%d\n", physicalDeviceProperties.deviceName, (physicalDeviceProperties.apiVersion >> 22), ((physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (physicalDeviceProperties.apiVersion & 0xfff));
		}
		printf("Benchmark score VkFFT: %d\n", (int)(benchmark_result));
		printf("Device name: %s API:%d.%d.%d\n", physicalDeviceProperties.deviceName, (physicalDeviceProperties.apiVersion >> 22), ((physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (physicalDeviceProperties.apiVersion & 0xfff));
		vkDestroyFence(device, fence, NULL);
		vkDestroyCommandPool(device, commandPool, NULL);
		vkDestroyDevice(device, NULL);
		DestroyDebugUtilsMessengerEXT(instance, debugMessenger, NULL);
		vkDestroyInstance(instance, NULL);
		break;
	}
	case 3:
	{
		//3 - convolution
		//Configuration + FFT application.
		VkFFTConfiguration forward_configuration = defaultVkFFTConfiguration;
		VkFFTConfiguration convolution_configuration = defaultVkFFTConfiguration;
		VkFFTApplication app_convolution;
		VkFFTApplication app_kernel;
		//Convolution sample code
		//Setting up FFT configuration. FFT is performed in-place with no performance loss. 
		switch (physicalDeviceProperties.vendorID) {
		case 0x10DE://NVIDIA
			forward_configuration.coalescedMemory = 32;
			forward_configuration.registerBoost = 1;
			break;
		case 0x8086://INTEL
			forward_configuration.coalescedMemory = 64;
			forward_configuration.registerBoost = 1;
			break;
		case 0x13B5://AMD
			forward_configuration.coalescedMemory = 32;
			forward_configuration.registerBoost = 1;
		default:
			forward_configuration.coalescedMemory = 64;
			forward_configuration.registerBoost = 1;
			break;
		}
		forward_configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).
		forward_configuration.size[0] = 1024*1024*8; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z. 
		forward_configuration.size[1] = 1;
		forward_configuration.size[2] = 1;
		forward_configuration.performConvolution = false; //Perform convolution with precomputed kernel. As we perform forward FFT to get the kernel, it is set to false.
		forward_configuration.performR2C = false; //Perform R2C/C2R transform. Can be combined with all other options. Reduces memory requirements by a factor of 2. Requires special input data alignment: for x*y*z system pad x*y plane to (x+2)*y with last 2*y elements reserved, total array dimensions are (x*y+2y)*z. Memory layout after R2C and before C2R can be found on github.
		forward_configuration.coordinateFeatures = 9; //Specify dimensionality of the input feature vector (default 1). Each component is stored not as a vector, but as a separate system and padded on it's own according to other options (i.e. for x*y system of 3-vector, first x*y elements correspond to the first dimension, then goes x*y for the second, etc).
		//coordinateFeatures number is an important constant for convolution. If we perform 1x1 convolution, it is equal to number of features, but matrixConvolution should be equal to 1. For matrix convolution, it must be equal to matrixConvolution parameter. If we perform 2x2 convolution, it is equal to 3 for symmetric kernel (stored as xx, xy, yy) and 4 for nonsymmetric (stored as xx, xy, yx, yy). Similarly, 6 (stored as xx, xy, xz, yy, yz, zz) and 9 (stored as xx, xy, xz, yx, yy, yz, zx, zy, zz) for 3x3 convolutions. 
		forward_configuration.inverse = false; //Direction of FFT. false - forward, true - inverse.
		forward_configuration.reorderFourStep = false;//Set to false if you use convolution routine. Data reordering is not needed - no additional buffer - less memory usage

		//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [VkDeviceSize *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [VkDeviceSize *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
		forward_configuration.device = &device;
		forward_configuration.queue = &queue; //to allocate memory for LUT, we have to pass a queue, fence, commandPool and physicalDevice pointers 
		forward_configuration.fence = &fence;
		forward_configuration.commandPool = &commandPool;
		forward_configuration.physicalDevice = &physicalDevice;
		forward_configuration.useLUT = false;
		if (sizeof(SHADER_DIR) > 255) {
			printf("SHADER_DIR length must be <256\n");
			return 1;

		}
		sprintf(forward_configuration.shaderPath, SHADER_DIR);
		//In this example, we perform a convolution for a real vectorfield (3vector) with a symmetric kernel (6 values). We use forward_configuration to initialize convolution kernel first from real data, then we create convolution_configuration for convolution. The buffer object from forward_configuration is passed to convolution_configuration as kernel object.
		//1. Kernel forward FFT.
		VkDeviceSize kernelSize = ((uint64_t)forward_configuration.coordinateFeatures) * sizeof(float) * 2 * (forward_configuration.size[0]) * forward_configuration.size[1] * forward_configuration.size[2];;
		VkBuffer kernel = {};
		VkDeviceMemory kernelDeviceMemory = {};

		//Sample allocation tool.
		allocateFFTBuffer(&kernel, &kernelDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, kernelSize);
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
						kernel_input[2 * (i + j * (forward_configuration.size[0]) + k * (forward_configuration.size[0]) * forward_configuration.size[1] + v * (forward_configuration.size[0]) * forward_configuration.size[1] * forward_configuration.size[2])+1] = 0;

					}
				}
			}
		}
		//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
		transferDataFromCPU(kernel_input, &kernel, kernelSize);
		//Initialize application responsible for the kernel. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
		initializeVulkanFFT(&app_kernel, forward_configuration);
		//Sample forward FFT command buffer allocation + execution performed on kernel. Second number determines how many times perform application in one submit. FFT can also be appended to user defined command buffers.

		//Uncomment the line below if you want to perform kernel FFT. In this sample we use predefined identitiy kernel.
		//performVulkanFFT(&app_kernel, 1);

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
		VkDeviceSize bufferSize = ((uint64_t)convolution_configuration.coordinateFeatures) * sizeof(float) * 2 * (convolution_configuration.size[0]) * convolution_configuration.size[1] * convolution_configuration.size[2];;
		VkBuffer buffer = {};
		VkDeviceMemory bufferDeviceMemory = {};

		allocateFFTBuffer(&buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
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
						buffer_input[2 * (i + j * convolution_configuration.size[0] + k * (convolution_configuration.size[0]) * convolution_configuration.size[1] + v * (convolution_configuration.size[0]) * convolution_configuration.size[1] * convolution_configuration.size[2])] = i%8-3.5;
						buffer_input[2 * (i + j * convolution_configuration.size[0] + k * (convolution_configuration.size[0]) * convolution_configuration.size[1] + v * (convolution_configuration.size[0]) * convolution_configuration.size[1] * convolution_configuration.size[2])+1] = i % 4 - 1.5;
					}
				}
			}
		}
		//Transfer data to GPU using staging buffer.
		transferDataFromCPU(buffer_input, &buffer, bufferSize);

		//Initialize application responsible for the convolution.
		initializeVulkanFFT(&app_convolution, convolution_configuration);
		//Sample forward FFT command buffer allocation + execution performed on kernel. FFT can also be appended to user defined command buffers.
		performVulkanFFT(&app_convolution, 1);
		//The kernel has been trasnformed.

		float* buffer_output = (float*)malloc(bufferSize);
		//Transfer data from GPU using staging buffer.
		transferDataToCPU(buffer_output, &buffer, bufferSize);

		//Print data, if needed.
		for (uint32_t v = 0; v < convolution_configuration.coordinateFeatures; v++) {
			if (file_output)
				fprintf(output, "\ncoordinate: %d\n\n", v);
			printf("\ncoordinate: %d\n\n", v);
			for (uint32_t k = 0; k < convolution_configuration.size[2]; k++) {
				for (uint32_t j = 0; j < convolution_configuration.size[1]; j++) {
					for (uint32_t i = 0; i < convolution_configuration.size[0]; i++) {
						if (file_output)
							fprintf(output, "%.6f %.6f ", buffer_output[2*(i + j * convolution_configuration.size[0] + k * (convolution_configuration.size[0]) * convolution_configuration.size[1] + v * (convolution_configuration.size[0] ) * convolution_configuration.size[1] * convolution_configuration.size[2])], buffer_output[2 * (i + j * convolution_configuration.size[0] + k * (convolution_configuration.size[0]) * convolution_configuration.size[1] + v * (convolution_configuration.size[0]) * convolution_configuration.size[1] * convolution_configuration.size[2])+1]);
						printf("%.6f %.6f ", buffer_output[2 * (i + j * convolution_configuration.size[0] + k * (convolution_configuration.size[0]) * convolution_configuration.size[1] + v * (convolution_configuration.size[0]) * convolution_configuration.size[1] * convolution_configuration.size[2])], buffer_output[2 * (i + j * convolution_configuration.size[0] + k * (convolution_configuration.size[0]) * convolution_configuration.size[1] + v * (convolution_configuration.size[0]) * convolution_configuration.size[1] * convolution_configuration.size[2])+1]);
					}
					std::cout << "\n";
				}
			}
		}
		free(kernel_input);
		free(buffer_input);
		free(buffer_output);
		vkDestroyBuffer(device, buffer, NULL);
		vkFreeMemory(device, bufferDeviceMemory, NULL);
		vkDestroyBuffer(device, kernel, NULL);
		vkFreeMemory(device, kernelDeviceMemory, NULL);
		deleteVulkanFFT(&app_kernel);
		deleteVulkanFFT(&app_convolution);
		vkDestroyFence(device, fence, NULL);
		vkDestroyCommandPool(device, commandPool, NULL);
		vkDestroyDevice(device, NULL);
		DestroyDebugUtilsMessengerEXT(instance, debugMessenger, NULL);
		vkDestroyInstance(instance, NULL);
		break;
	}
	case 4:
	{
		//4 - zeropadding convolution
		//Configuration + FFT application.
		VkFFTConfiguration forward_configuration = defaultVkFFTConfiguration;
		VkFFTConfiguration convolution_configuration = defaultVkFFTConfiguration;
		VkFFTApplication app_convolution;
		VkFFTApplication app_kernel;
		//Zeropadding Convolution sample code
		//Setting up FFT configuration. FFT is performed in-place with no performance loss. 
		switch (physicalDeviceProperties.vendorID) {
		case 0x10DE://NVIDIA
			forward_configuration.coalescedMemory = 32;
			forward_configuration.registerBoost = 1;
			break;
		case 0x8086://INTEL
			forward_configuration.coalescedMemory = 64;
			forward_configuration.registerBoost = 1;
			break;
		case 0x13B5://AMD
			forward_configuration.coalescedMemory = 32;
			forward_configuration.registerBoost = 1;
		default:
			forward_configuration.coalescedMemory = 64;
			forward_configuration.registerBoost = 1;
			break;
		}
		forward_configuration.FFTdim = 3; //FFT dimension, 1D, 2D or 3D (default 1).
		forward_configuration.size[0] = 256; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z. 
		forward_configuration.size[1] = 256;
		forward_configuration.size[2] = 256;
		forward_configuration.performZeropadding[0] = true; //Perform padding with zeros on GPU. Still need to properly align input data (no need to fill padding area with meaningful data) but this will increase performance due to the lower amount of the memory reads/writes and omitting sequences only consisting of zeros.
		forward_configuration.performZeropadding[1] = true;
		forward_configuration.performZeropadding[2] = true;
		forward_configuration.performConvolution = false; //Perform convolution with precomputed kernel. As we perform forward FFT to get the kernel, it is set to false.
		forward_configuration.performR2C = true; //Perform R2C/C2R transform. Can be combined with all other options. Reduces memory requirements by a factor of 2. Requires special input data alignment: for x*y*z system pad x*y plane to (x+2)*y with last 2*y elements reserved, total array dimensions are (x*y+2y)*z. Memory layout after R2C and before C2R can be found on github.
		forward_configuration.coordinateFeatures = 9; //Specify dimensionality of the input feature vector (default 1). Each component is stored not as a vector, but as a separate system and padded on it's own according to other options (i.e. for x*y system of 3-vector, first x*y elements correspond to the first dimension, then goes x*y for the second, etc).
		//coordinateFeatures number is an important constant for convolution. If we perform 1x1 convolution, it is equal to number of features, but matrixConvolution should be equal to 1. For matrix convolution, it must be equal to matrixConvolution parameter. If we perform 2x2 convolution, it is equal to 3 for symmetric kernel (stored as xx, xy, yy) and 4 for nonsymmetric (stored as xx, xy, yx, yy). Similarly, 6 (stored as xx, xy, xz, yy, yz, zz) and 9 (stored as xx, xy, xz, yx, yy, yz, zx, zy, zz) for 3x3 convolutions. 
		forward_configuration.inverse = false; //Direction of FFT. false - forward, true - inverse.
		forward_configuration.reorderFourStep = false;//Set to false if you use convolution routine. Data reordering is not needed - no additional buffer - less memory usage

		//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [VkDeviceSize *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [VkDeviceSize *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
		forward_configuration.device = &device;
		forward_configuration.queue = &queue; //to allocate memory for LUT, we have to pass a queue, fence, commandPool and physicalDevice pointers 
		forward_configuration.fence = &fence;
		forward_configuration.commandPool = &commandPool;
		forward_configuration.physicalDevice = &physicalDevice;
		forward_configuration.useLUT = false;
		if (sizeof(SHADER_DIR) > 255) {
			printf("SHADER_DIR length must be <256\n");
			return 1;
		}
		sprintf(forward_configuration.shaderPath, SHADER_DIR);
		//In this example, we perform a convolution for a real vectorfield (3vector) with a symmetric kernel (6 values). We use forward_configuration to initialize convolution kernel first from real data, then we create convolution_configuration for convolution. The buffer object from forward_configuration is passed to convolution_configuration as kernel object.
		//1. Kernel forward FFT.
		VkDeviceSize kernelSize = ((uint64_t)forward_configuration.coordinateFeatures) * sizeof(float) * 2 * (forward_configuration.size[0] / 2 + 1) * forward_configuration.size[1] * forward_configuration.size[2];
		VkBuffer kernel = {};
		VkDeviceMemory kernelDeviceMemory = {};

		//Sample allocation tool.
		allocateFFTBuffer(&kernel, &kernelDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, kernelSize);
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

							kernel_input[2 * i + j * (forward_configuration.size[0] + 2) + k * (forward_configuration.size[0] + 2) * forward_configuration.size[1] + v * (forward_configuration.size[0] + 2) * forward_configuration.size[1] * forward_configuration.size[2]] = 1;

						else
							kernel_input[2 * i + j * (forward_configuration.size[0] + 2) + k * (forward_configuration.size[0] + 2) * forward_configuration.size[1] + v * (forward_configuration.size[0] + 2) * forward_configuration.size[1] * forward_configuration.size[2]] = 0;
						kernel_input[2 * i + 1 + j * (forward_configuration.size[0] + 2) + k * (forward_configuration.size[0] + 2) * forward_configuration.size[1] + v * (forward_configuration.size[0] + 2) * forward_configuration.size[1] * forward_configuration.size[2]] = 0;

					}
				}
			}
		}
		//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
		transferDataFromCPU(kernel_input, &kernel, kernelSize);
		//Initialize application responsible for the kernel. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
		initializeVulkanFFT(&app_kernel, forward_configuration);
		//Sample forward FFT command buffer allocation + execution performed on kernel. Second number determines how many times perform application in one submit. FFT can also be appended to user defined command buffers.

		//Uncomment the line below if you want to perform kernel FFT. In this sample we use predefined identitiy kernel.
		//performVulkanFFT(&app_kernel, 1);

		//The kernel has been trasnformed.


		//2. Buffer convolution with transformed kernel.
		//Copy configuration, as it mostly remains unchanged. Change specific parts.
		convolution_configuration = forward_configuration;
		convolution_configuration.performConvolution = true;
		convolution_configuration.symmetricKernel = false;//Specify if convolution kernel is symmetric. In this case we only pass upper triangle part of it in the form of: (xx, xy, yy) for 2d and (xx, xy, xz, yy, yz, zz) for 3d.
		convolution_configuration.matrixConvolution = 3; //we do matrix convolution, so kernel is 9 numbers (3x3), but vector dimension is 3
		convolution_configuration.coordinateFeatures = 3; //equal to matrixConvolution size
		convolution_configuration.kernel = &kernel;
		convolution_configuration.kernelSize = &kernelSize;

		//Allocate separate buffer for the input data.
		VkDeviceSize bufferSize = ((uint64_t)convolution_configuration.coordinateFeatures) * sizeof(float) * 2 * (convolution_configuration.size[0] / 2 + 1) * convolution_configuration.size[1] * convolution_configuration.size[2];;
		VkBuffer buffer = {};
		VkDeviceMemory bufferDeviceMemory = {};

		allocateFFTBuffer(&buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
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
			for (uint32_t k = 0; k < convolution_configuration.size[2] / 2; k++) {
				for (uint32_t j = 0; j < convolution_configuration.size[1] / 2; j++) {
					for (uint32_t i = 0; i < convolution_configuration.size[0] / 2; i++) {
						buffer_input[i + j * convolution_configuration.size[0] + k * (convolution_configuration.size[0] + 2) * convolution_configuration.size[1] + v * (convolution_configuration.size[0] + 2) * convolution_configuration.size[1] * convolution_configuration.size[2]] = i;
					}
				}
			}
		}
		//Transfer data to GPU using staging buffer.
		transferDataFromCPU(buffer_input, &buffer, bufferSize);

		//Initialize application responsible for the convolution.
		initializeVulkanFFT(&app_convolution, convolution_configuration);
		//Sample forward FFT command buffer allocation + execution performed on kernel. FFT can also be appended to user defined command buffers.
		performVulkanFFT(&app_convolution, 1);
		//The kernel has been trasnformed.

		float* buffer_output = (float*)malloc(bufferSize);
		//Transfer data from GPU using staging buffer.
		transferDataToCPU(buffer_output, &buffer, bufferSize);

		//Print data, if needed.
		for (uint32_t v = 0; v < convolution_configuration.coordinateFeatures; v++) {
			if (file_output)
				fprintf(output, "\ncoordinate: %d\n\n", v);
			printf("\ncoordinate: %d\n\n", v);
			for (uint32_t k = 0; k < convolution_configuration.size[2] / 2; k++) {
				for (uint32_t j = 0; j < convolution_configuration.size[1] / 2; j++) {
					for (uint32_t i = 0; i < convolution_configuration.size[0] / 2; i++) {
						if (file_output)
							fprintf(output, "%.6f ", buffer_output[i + j * convolution_configuration.size[0] + k * (convolution_configuration.size[0] + 2) * convolution_configuration.size[1] + v * (convolution_configuration.size[0] + 2) * convolution_configuration.size[1] * convolution_configuration.size[2]]);
						printf("%.6f ", buffer_output[i + j * convolution_configuration.size[0] + k * (convolution_configuration.size[0] + 2) * convolution_configuration.size[1] + v * (convolution_configuration.size[0] + 2) * convolution_configuration.size[1] * convolution_configuration.size[2]]);
					}
					std::cout << "\n";
				}
			}
		}
		free(kernel_input);
		free(buffer_input);
		free(buffer_output);
		vkDestroyBuffer(device, buffer, NULL);
		vkFreeMemory(device, bufferDeviceMemory, NULL);
		vkDestroyBuffer(device, kernel, NULL);
		vkFreeMemory(device, kernelDeviceMemory, NULL);
		deleteVulkanFFT(&app_kernel);
		deleteVulkanFFT(&app_convolution);
		vkDestroyFence(device, fence, NULL);
		vkDestroyCommandPool(device, commandPool, NULL);
		vkDestroyDevice(device, NULL);
		DestroyDebugUtilsMessengerEXT(instance, debugMessenger, NULL);
		vkDestroyInstance(instance, NULL);
		break;
	}
	case 5:
	{
		//5 - multiple batches (kernels) convolution
		//Configuration + FFT application.
		VkFFTConfiguration forward_configuration = defaultVkFFTConfiguration;
		VkFFTConfiguration convolution_configuration = defaultVkFFTConfiguration;
		VkFFTApplication app_convolution;
		VkFFTApplication app_kernel;
		//Convolution sample code
		//Setting up FFT configuration. FFT is performed in-place with no performance loss. 
		switch (physicalDeviceProperties.vendorID) {
		case 0x10DE://NVIDIA
			forward_configuration.coalescedMemory = 32;
			forward_configuration.registerBoost = 1;
			break;
		case 0x8086://INTEL
			forward_configuration.coalescedMemory = 64;
			forward_configuration.registerBoost = 1;
			break;
		case 0x13B5://AMD
			forward_configuration.coalescedMemory = 32;
			forward_configuration.registerBoost = 1;
		default:
			forward_configuration.coalescedMemory = 64;
			forward_configuration.registerBoost = 1;
			break;
		}
		forward_configuration.FFTdim = 2; //FFT dimension, 1D, 2D or 3D (default 1).
		forward_configuration.size[0] = 32; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z. 
		forward_configuration.size[1] = 32;
		forward_configuration.size[2] = 1;
		forward_configuration.performConvolution = false; //Perform convolution with precomputed kernel. As we perform forward FFT to get the kernel, it is set to false.
		forward_configuration.performR2C = true; //Perform R2C/C2R transform. Can be combined with all other options. Reduces memory requirements by a factor of 2. Requires special input data alignment: for x*y*z system pad x*y plane to (x+2)*y with last 2*y elements reserved, total array dimensions are (x*y+2y)*z. Memory layout after R2C and before C2R can be found on github.
		forward_configuration.coordinateFeatures = 3; //Specify dimensionality of the input feature vector (default 1). Each component is stored not as a vector, but as a separate system and padded on it's own according to other options (i.e. for x*y system of 3-vector, first x*y elements correspond to the first dimension, then goes x*y for the second, etc).
		//coordinateFeatures number is an important constant for convolution. If we perform 1x1 convolution, it is equal to number of features, but matrixConvolution should be equal to 1. For matrix convolution, it must be equal to matrixConvolution parameter. If we perform 2x2 convolution, it is equal to 3 for symmetric kernel (stored as xx, xy, yy) and 4 for nonsymmetric (stored as xx, xy, yx, yy). Similarly, 6 (stored as xx, xy, xz, yy, yz, zz) and 9 (stored as xx, xy, xz, yx, yy, yz, zx, zy, zz) for 3x3 convolutions. 
		forward_configuration.inverse = false; //Direction of FFT. false - forward, true - inverse.
		forward_configuration.reorderFourStep = false;//Set to false if you use convolution routine. Data reordering is not needed - no additional buffer - less memory usage

		forward_configuration.numberBatches = 2;
		//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [VkDeviceSize *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [VkDeviceSize *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
		forward_configuration.device = &device;
		forward_configuration.queue = &queue; //to allocate memory for LUT, we have to pass a queue, fence, commandPool and physicalDevice pointers 
		forward_configuration.fence = &fence;
		forward_configuration.commandPool = &commandPool;
		forward_configuration.physicalDevice = &physicalDevice;
		forward_configuration.useLUT = false;
		if (sizeof(SHADER_DIR) > 255) {
			printf("SHADER_DIR length must be <256\n");
			return 1;
		}
		sprintf(forward_configuration.shaderPath, SHADER_DIR);
		//In this example, we perform a convolution for a real vectorfield (3vector) with a symmetric kernel (6 values). We use forward_configuration to initialize convolution kernel first from real data, then we create convolution_configuration for convolution. The buffer object from forward_configuration is passed to convolution_configuration as kernel object.
		//1. Kernel forward FFT.
		VkDeviceSize kernelSize = ((uint64_t)forward_configuration.numberBatches) * forward_configuration.coordinateFeatures * sizeof(float) * 2 * (forward_configuration.size[0] / 2 + 1) * forward_configuration.size[1] * forward_configuration.size[2];;
		VkBuffer kernel = {};
		VkDeviceMemory kernelDeviceMemory = {};

		//Sample allocation tool.
		allocateFFTBuffer(&kernel, &kernelDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, kernelSize);
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

							kernel_input[2 * i + j * (forward_configuration.size[0] + 2) + k * (forward_configuration.size[0] + 2) * forward_configuration.size[1] + v * (forward_configuration.size[0] + 2) * forward_configuration.size[1] * forward_configuration.size[2] + f * forward_configuration.coordinateFeatures * (forward_configuration.size[0] + 2) * forward_configuration.size[1] * forward_configuration.size[2]] = f * forward_configuration.coordinateFeatures + v + 1;
							kernel_input[2 * i + 1 + j * (forward_configuration.size[0] + 2) + k * (forward_configuration.size[0] + 2) * forward_configuration.size[1] + v * (forward_configuration.size[0] + 2) * forward_configuration.size[1] * forward_configuration.size[2] + f * forward_configuration.coordinateFeatures * (forward_configuration.size[0] + 2) * forward_configuration.size[1] * forward_configuration.size[2]] = 0;

						}
					}
				}
			}
		}
		//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
		transferDataFromCPU(kernel_input, &kernel, kernelSize);
		//Initialize application responsible for the kernel. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
		initializeVulkanFFT(&app_kernel, forward_configuration);
		//Sample forward FFT command buffer allocation + execution performed on kernel. Second number determines how many times perform application in one submit. FFT can also be appended to user defined command buffers.

		//Uncomment the line below if you want to perform kernel FFT. In this sample we use predefined identitiy kernel.
		//performVulkanFFT(&app_kernel, 1);

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
		VkDeviceSize bufferSize = ((uint64_t)convolution_configuration.coordinateFeatures) * sizeof(float) * 2 * (convolution_configuration.size[0] / 2 + 1) * convolution_configuration.size[1] * convolution_configuration.size[2];;
		VkBuffer buffer = {};
		VkDeviceMemory bufferDeviceMemory = {};

		allocateFFTBuffer(&buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
		VkDeviceSize outputBufferSize = convolution_configuration.numberKernels * convolution_configuration.coordinateFeatures * sizeof(float) * 2 * (convolution_configuration.size[0] / 2 + 1) * convolution_configuration.size[1] * convolution_configuration.size[2];;
		VkBuffer outputBuffer = {};
		VkDeviceMemory outputBufferDeviceMemory = {};

		allocateFFTBuffer(&outputBuffer, &outputBufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, outputBufferSize);

		convolution_configuration.buffer = &buffer;
		convolution_configuration.isInputFormatted = false; //if input is a different buffer, it doesn't have to be zeropadded/R2C padded	
		convolution_configuration.inputBuffer = &buffer;
		convolution_configuration.isOutputFormatted = false;//if output is a different buffer, it can have zeropadding/C2R automatically removed
		convolution_configuration.outputBuffer = &outputBuffer;
		convolution_configuration.bufferSize = &bufferSize;
		convolution_configuration.inputBufferSize = &bufferSize;
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
						buffer_input[i + j * convolution_configuration.size[0] + k * (convolution_configuration.size[0] + 2) * convolution_configuration.size[1] + v * (convolution_configuration.size[0] + 2) * convolution_configuration.size[1] * convolution_configuration.size[2]] = 1;
					}
				}
			}
		}
		//Transfer data to GPU using staging buffer.
		transferDataFromCPU(buffer_input, &buffer, bufferSize);

		//Initialize application responsible for the convolution.
		initializeVulkanFFT(&app_convolution, convolution_configuration);
		//Sample forward FFT command buffer allocation + execution performed on kernel. FFT can also be appended to user defined command buffers.
		performVulkanFFT(&app_convolution, 1);
		//The kernel has been trasnformed.

		float* buffer_output = (float*)malloc(outputBufferSize);
		//Transfer data from GPU using staging buffer.
		transferDataToCPU(buffer_output, &outputBuffer, outputBufferSize);

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
								fprintf(output, "%.6f ", buffer_output[i + j * convolution_configuration.size[0] + k * (convolution_configuration.size[0] + 2) * convolution_configuration.size[1] + v * (convolution_configuration.size[0] + 2) * convolution_configuration.size[1] * convolution_configuration.size[2] + f * convolution_configuration.coordinateFeatures * (convolution_configuration.size[0] + 2) * convolution_configuration.size[1] * convolution_configuration.size[2]]);

							printf("%.6f ", buffer_output[i + j * convolution_configuration.size[0] + k * (convolution_configuration.size[0] + 2) * convolution_configuration.size[1] + v * (convolution_configuration.size[0] + 2) * convolution_configuration.size[1] * convolution_configuration.size[2] + f * convolution_configuration.coordinateFeatures * (convolution_configuration.size[0] + 2) * convolution_configuration.size[1] * convolution_configuration.size[2]]);
						}
						std::cout << "\n";
					}
				}
			}
		}
		free(kernel_input);
		free(buffer_input);
		free(buffer_output);
		vkDestroyBuffer(device, buffer, NULL);
		vkFreeMemory(device, bufferDeviceMemory, NULL);
		vkDestroyBuffer(device, outputBuffer, NULL);
		vkFreeMemory(device, outputBufferDeviceMemory, NULL);
		vkDestroyBuffer(device, kernel, NULL);
		vkFreeMemory(device, kernelDeviceMemory, NULL);
		deleteVulkanFFT(&app_kernel);
		deleteVulkanFFT(&app_convolution);
		vkDestroyFence(device, fence, NULL);
		vkDestroyCommandPool(device, commandPool, NULL);
		vkDestroyDevice(device, NULL);
		DestroyDebugUtilsMessengerEXT(instance, debugMessenger, NULL);
		vkDestroyInstance(instance, NULL);
		break;
	}
	case 6:
	{
		//6 - FFT + iFFT C2C benchmark for big systems. Done by using four-step FFT algorithm in two or three passes - up to pow(2,30). Test uses 4GB of memory.
		const int num_runs = 3;
		double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
		//memory allocated on the CPU once, makes benchmark completion faster + avoids performance issues connected to frequent allocation/deallocation.
		float* buffer_input = (float*)malloc((uint64_t)4 * 2 * 4096 * 4096 * 8 * 2);
		for (uint64_t i = 0; i < 2 * 4096 * 4096 * 2 * 8; i++) {
			buffer_input[i] = 2 * ((float)rand()) / RAND_MAX - 1.0;
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
				forward_configuration.size[0] = 32 * pow(2, n); //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
				if (n == 0) forward_configuration.size[0] = 4096;
				forward_configuration.size[1] = 64 * 32 * pow(2, 17) / forward_configuration.size[0];
				//forward_configuration.size[1] = (forward_configuration.size[1] > 32768) ? 32768 : forward_configuration.size[1];
				forward_configuration.size[2] = 1;


				//PARAMETERS THAT CAN BE ADJUSTED FOR SPECIFIC GPU's - this configuration is by no means final form
				switch (physicalDeviceProperties.vendorID) {
				case 0x10DE://NVIDIA
					forward_configuration.coalescedMemory = 32;//the coalesced memory is equal to 32 bytes between L2 and VRAM. But 16 behaves better (only affects seqence of 2048 elements in y axis - it is done in one upload this way)
					forward_configuration.useLUT = false;
					if (forward_configuration.size[1] > 512)
						forward_configuration.registerBoost = 4;
					else
						forward_configuration.registerBoost = 1;
					forward_configuration.registerBoost = 4;
					break;
				case 0x8086://INTEL
					forward_configuration.coalescedMemory = 64;
					forward_configuration.useLUT = true;
					forward_configuration.registerBoost = 1;

					break;
				case 0x13B5://AMD
					forward_configuration.coalescedMemory = 32;
					forward_configuration.useLUT = false;
					if (forward_configuration.size[1] > 512)
						forward_configuration.registerBoost = 4;
					else
						forward_configuration.registerBoost = 1;
					break;
				default:
					forward_configuration.coalescedMemory = 64;
					forward_configuration.useLUT = false;
					forward_configuration.registerBoost = 1;
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
				forward_configuration.device = &device;
				forward_configuration.queue = &queue; //to allocate memory for LUT, we have to pass a queue, fence, commandPool and physicalDevice pointers 
				forward_configuration.fence = &fence;
				forward_configuration.commandPool = &commandPool;
				forward_configuration.physicalDevice = &physicalDevice;

				//Custom path to the floder with shaders, default is "shaders/". Max length - 256 chars.
				if (sizeof(SHADER_DIR) > 255) {
					printf("SHADER_DIR length must be <256\n");
					return 1;
				}
				sprintf(forward_configuration.shaderPath, SHADER_DIR);

				//Allocate buffer for the input data.
				VkDeviceSize bufferSize = ((uint64_t)forward_configuration.coordinateFeatures) * sizeof(float) * 2 * forward_configuration.size[0] * forward_configuration.size[1] * forward_configuration.size[2];;
				VkBuffer buffer = {};
				VkDeviceMemory bufferDeviceMemory = {};
				VkBuffer tempBuffer = {};
				VkDeviceMemory tempBufferDeviceMemory = {};
				allocateFFTBuffer(&buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
				allocateFFTBuffer(&tempBuffer, &tempBufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);

				forward_configuration.buffer = &buffer;
				forward_configuration.tempBuffer = &tempBuffer;
				forward_configuration.isInputFormatted = false; //set to true if input is a different buffer, so it can have zeropadding/R2C added  
				forward_configuration.inputBuffer = &buffer; //you can specify first buffer to read data from to be different from the buffer FFT is performed on. FFT is still in-place on the second buffer, this is here just for convenience.
				forward_configuration.isOutputFormatted = false;//set to true if output is a different buffer, so it can have zeropadding/C2R automatically removed
				forward_configuration.outputBuffer = &buffer;
				forward_configuration.bufferSize = &bufferSize;
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
				transferDataFromCPU(buffer_input, &buffer, bufferSize);
				//free(buffer_input);

				//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
				initializeVulkanFFT(&app_forward, forward_configuration);
				initializeVulkanFFT(&app_inverse, inverse_configuration);
				//Submit FFT+iFFT.
				uint32_t batch = ((4096 * 1024.0 * 1024.0) / bufferSize > 1000) ? 1000 : (4096 * 1024.0 * 1024.0) / bufferSize;
				if (batch == 0) batch = 1;
				batch *= 5;
				float totTime = performVulkanFFTiFFT(&app_forward, &app_inverse, batch);

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
						if (file_output)
							fprintf(output, "VkFFT System: %d %dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f batch: %d benchmark: %d\n", forward_configuration.size[0], forward_configuration.size[0], forward_configuration.size[1], bufferSize / 1024 / 1024, avg_time, std_error, batch, (int)(((double)bufferSize / 1024) / avg_time));

						printf("VkFFT System: %d %dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f batch: %d benchmark: %d\n", forward_configuration.size[0], forward_configuration.size[0], forward_configuration.size[1], bufferSize / 1024 / 1024, avg_time, std_error, batch, (int)(((double)bufferSize / 1024) / avg_time));
						benchmark_result += ((double)bufferSize / 1024) / avg_time;
					}


				}

				vkDestroyBuffer(device, buffer, NULL);
				vkDestroyBuffer(device, tempBuffer, NULL);
				vkFreeMemory(device, bufferDeviceMemory, NULL);
				vkFreeMemory(device, tempBufferDeviceMemory, NULL);
				deleteVulkanFFT(&app_forward);
				deleteVulkanFFT(&app_inverse);
			}
		}
		free(buffer_input);
		benchmark_result /= (24 - 1);

		if (file_output) {
			fprintf(output, "Benchmark score VkFFT: %d\n", (int)(benchmark_result));
			fprintf(output, "Device name: %s API:%d.%d.%d\n", physicalDeviceProperties.deviceName, (physicalDeviceProperties.apiVersion >> 22), ((physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (physicalDeviceProperties.apiVersion & 0xfff));
		}
		printf("Benchmark score VkFFT: %d\n", (int)(benchmark_result));
		printf("Device name: %s API:%d.%d.%d\n", physicalDeviceProperties.deviceName, (physicalDeviceProperties.apiVersion >> 22), ((physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (physicalDeviceProperties.apiVersion & 0xfff));
		vkDestroyFence(device, fence, NULL);
		vkDestroyCommandPool(device, commandPool, NULL);
		vkDestroyDevice(device, NULL);
		DestroyDebugUtilsMessengerEXT(instance, debugMessenger, NULL);
		vkDestroyInstance(instance, NULL);
		break;
	}
	case 7:
	{
		//7 - FFT + iFFT R2C/C2R benchmark. Limited to one upload for now in x axis (16k with register overutilization), y and z are pow(2,30)
		const uint32_t num_benchmark_samples = 19;
		const uint32_t num_runs = 7;
		//printf("First %d runs are a warmup\n", num_runs);
		uint32_t benchmark_dimensions[num_benchmark_samples][4] = { {1024, 1024, 1, 2}, {64, 64, 1, 2}, {256, 256, 1, 2}, {1024, 256, 1, 2}, {512, 512, 1, 2}, {1024, 1024, 1, 2},  {4096, 256, 1, 2}, {2048, 1024, 1, 2},{4096, 2048, 1, 2}, {4096, 4096, 1, 2},
																	{32, 32, 32, 3}, {64, 64, 64, 3}, {256, 256, 32, 3},  {1024, 256, 32, 3},  {256, 256, 256, 3}, {2048, 1024, 8, 3},  {512, 512, 128, 3}, {2048, 256, 256, 3}, {4096, 4096, 8, 3} };
		double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
		float* buffer_input = (float*)malloc((uint64_t)4 * 2 * 4096 * 4096 * 8 * 2);
		for (uint64_t i = 0; i < 2 * 4096 * 4096 * 2 * 8; i++) {
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
				switch (physicalDeviceProperties.vendorID) {
				case 0x10DE://NVIDIA
					forward_configuration.coalescedMemory = 32;
					forward_configuration.registerBoost = 4;
					break;
				case 0x8086://INTEL
					forward_configuration.coalescedMemory = 64;
					forward_configuration.registerBoost = 4;
					break;
				case 0x13B5://AMD
					forward_configuration.coalescedMemory = 32;
					forward_configuration.registerBoost = 4;
					break;
				default:
					forward_configuration.coalescedMemory = 64;
					forward_configuration.registerBoost = 1;
					break;
				}
				forward_configuration.FFTdim = benchmark_dimensions[n][3]; //FFT dimension, 1D, 2D or 3D (default 1).
				forward_configuration.size[0] = benchmark_dimensions[n][0]; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
				forward_configuration.size[1] = benchmark_dimensions[n][1];
				forward_configuration.size[2] = benchmark_dimensions[n][2];
				forward_configuration.performZeropadding[0] = false; //Perform padding with zeros on GPU. Still need to properly align input data (no need to fill padding area with meaningful data) but this will increase performance due to the lower amount of the memory reads/writes and omitting sequences only consisting of zeros.
				forward_configuration.performZeropadding[1] = false;
				forward_configuration.performZeropadding[2] = false;
				forward_configuration.performConvolution = false; //Perform convolution with precomputed kernel. 
				forward_configuration.performR2C = true; //Perform R2C/C2R transform. Can be combined with all other options. Reduces memory requirements by a factor of 2. Requires special input data alignment: for x*y*z system pad x*y plane to (x+2)*y with last 2*y elements reserved, total array dimensions are (x*y+2y)*z. Memory layout after R2C and before C2R can be found on github.
				forward_configuration.coordinateFeatures = 1; //Specify dimensionality of the input feature vector (default 1). Each component is stored not as a vector, but as a separate system and padded on it's own according to other options (i.e. for x*y system of 3-vector, first x*y elements correspond to the first dimension, then goes x*y for the second, etc). 
				forward_configuration.inverse = false; //Direction of FFT. false - forward, true - inverse.
				forward_configuration.reorderFourStep = true;//set to true if you want data to return to correct layout after FFT. Set to false if you use convolution routine. Requires additional tempBuffer of bufferSize (see below) to do reordering
				//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [VkDeviceSize *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [VkDeviceSize *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
				forward_configuration.device = &device;
				forward_configuration.queue = &queue; //to allocate memory for LUT, we have to pass a queue, fence, commandPool and physicalDevice pointers 
				forward_configuration.fence = &fence;
				forward_configuration.commandPool = &commandPool;
				forward_configuration.physicalDevice = &physicalDevice;
				forward_configuration.useLUT = false;
				//Custom path to the floder with shaders, default is "shaders/");
				if (sizeof(SHADER_DIR) > 255) {
					printf("SHADER_DIR length must be <256\n");
					return 1;
				}
				sprintf(forward_configuration.shaderPath, SHADER_DIR);

				//Allocate buffer for the input data.
				VkDeviceSize bufferSize = ((uint64_t)forward_configuration.coordinateFeatures) * sizeof(float) * 2 * (forward_configuration.size[0] / 2 + 1) * forward_configuration.size[1] * forward_configuration.size[2];;
				VkBuffer buffer = {};
				VkDeviceMemory bufferDeviceMemory = {};

				VkBuffer tempBuffer = {};
				VkDeviceMemory tempBufferDeviceMemory = {};
				allocateFFTBuffer(&buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
				allocateFFTBuffer(&tempBuffer, &tempBufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
				forward_configuration.isInputFormatted = false; //set to true if input is a different buffer, so it can have zeropadding/R2C added  
				forward_configuration.inputBuffer = &buffer; //you can specify first buffer to read data from to be different from the buffer FFT is performed on. FFT is still in-place on the second buffer, this is here just for convenience.
				forward_configuration.isOutputFormatted = false;//set to true if output is a different buffer, so it can have zeropadding/C2R automatically removed
				forward_configuration.outputBuffer = &buffer;
				forward_configuration.bufferSize = &bufferSize;
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
				transferDataFromCPU(buffer_input, &buffer, bufferSize);
				//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
				initializeVulkanFFT(&app_forward, forward_configuration);
				initializeVulkanFFT(&app_inverse, inverse_configuration);
				//Submit FFT+iFFT.
				uint32_t batch = ((4096.0 * 1024.0 * 1024.0) / bufferSize > 1000) ? 1000 : (4096.0 * 1024.0 * 1024.0) / bufferSize;
				if (batch == 0) batch = 1;

				//batch *= 5; //makes result more smooth, takes longer time
				float totTime = performVulkanFFTiFFT(&app_forward, &app_inverse, batch);
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
						if (file_output)
							fprintf(output, "VkFFT System: %dx%dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error %0.3f batch: %d\n", benchmark_dimensions[n][0], benchmark_dimensions[n][1], benchmark_dimensions[n][2], bufferSize / 1024 / 1024, avg_time, std_error, batch);

						printf("VkFFT System: %dx%dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error %0.3f batch: %d\n", benchmark_dimensions[n][0], benchmark_dimensions[n][1], benchmark_dimensions[n][2], bufferSize / 1024 / 1024, avg_time, std_error, batch);
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
				vkDestroyBuffer(device, buffer, NULL);
				vkFreeMemory(device, bufferDeviceMemory, NULL);
				vkDestroyBuffer(device, tempBuffer, NULL);
				vkFreeMemory(device, tempBufferDeviceMemory, NULL);
				deleteVulkanFFT(&app_forward);
				deleteVulkanFFT(&app_inverse);
			}
		}
		free(buffer_input);
		benchmark_result /= ((num_benchmark_samples - 1));
		if (file_output) {
			fprintf(output, "Benchmark score VkFFT: %d\n", (int)(benchmark_result));
			printf("Device name: %s API:%d.%d.%d\n", physicalDeviceProperties.deviceName, (physicalDeviceProperties.apiVersion >> 22), ((physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (physicalDeviceProperties.apiVersion & 0xfff));
		}
		printf("Benchmark score VkFFT: %d\n", (int)(benchmark_result));
		printf("Device name: %s API:%d.%d.%d\n", physicalDeviceProperties.deviceName, (physicalDeviceProperties.apiVersion >> 22), ((physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (physicalDeviceProperties.apiVersion & 0xfff));

		vkDestroyFence(device, fence, NULL);
		vkDestroyCommandPool(device, commandPool, NULL);
		vkDestroyDevice(device, NULL);
		DestroyDebugUtilsMessengerEXT(instance, debugMessenger, NULL);
		vkDestroyInstance(instance, NULL);
		break;
	}
	}
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
	uint32_t device_id = 0;//device id
	bool file_output = false;
	FILE* output = NULL;
	if (findFlag(argv, argv + argc, "-h"))
	{
		//print help
		printf("VkFFT v1.0.3 (13-11-2020). Author: Tolmachev Dmitrii\n");
		printf("	-h: print help\n");
		printf("	-devices: print the list of available GPU devices\n");
		printf("	-d X: select GPU device (default 0)\n");
		printf("	-o NAME: specify output file path\n");
		printf("	-vkfft X: launch VkFFT sample X (0-7):\n		0 - FFT + iFFT C2C single precision benchmark\n		1 - double precision benchmark\n		2 - half precision benchmark\n		3 - convolution\n		4 - zeropadding convolution\n		5 - multiple feature(kernel) convolution\n		6 - Big FFTs (>16k)\n		7 - R2C/C2R benchmark\n");
#ifdef USE_cuFFT
		printf("	-cufft X: launch cuFFT sample X (0-2):\n		0 - FFT + iFFT C2C single precision benchmark\n		1 - double precision benchmark\n		2 - half precision benchmark\n");
#else
		printf("	-cufft command is disabled in CMakeLists.txt\n");
#endif
		return 0;
	}
	if (findFlag(argv, argv + argc, "-devices"))
	{
		//print device list
		devices_list();
		return 0;
	}
	if (findFlag(argv, argv + argc, "-d"))
	{
		//select device_id
		char* value = getFlagValue(argv, argv + argc, "-d");
		if (value != 0) {
			sscanf(value, "%d", &device_id);
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
			uint32_t sample_id = 0;//setting parameter for VkFFT samples. 0 - FFT + iFFT C2C benchmark. 1 - convolution. 2 - zeropadding convolution. 3 - multiple feature(kernel) convolution. 4 - Big FFTs (>16k) 5 - R2C/C2R benchmark 6 - double precision benchmark 7 - half precision benchmark
			sscanf(value, "%d", &sample_id);
			launchVkFFT(device_id, sample_id, file_output, output);
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
			uint32_t sample_id = 0;//setting parameter for cuFFT samples. 0 - single precision benchmark. 1 - double precision benchmark. 2 - half precision benchmark. 
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
			}
		}
		else {
			printf("No cuFFT script is selected with -cufft flag\n");
			return 1;
		}
	}
#endif
}
