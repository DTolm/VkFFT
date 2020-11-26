//general parts
#include <stdio.h>
#include <vector>
#include <memory>
#include <string.h>
#include <chrono>
#include <thread>
#include <iostream>

//CUDA parts
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>

//Vulkan parts
#include <vulkan/vulkan.h>
#include <vkFFT.h>

//FFTW
#include "fftw3.h"

#define GROUP 1

const bool enableValidationLayers = false;

VkInstance instance = {};
VkDebugReportCallbackEXT debugReportCallback = {};
VkPhysicalDevice physicalDevice = {};
VkPhysicalDeviceProperties physicalDeviceProperties = {};
VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties = {};
VkDevice device = {};
VkDebugUtilsMessengerEXT debugMessenger = {};
uint32_t queueFamilyIndex = {};
std::vector<const char*> enabledLayers;
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

std::vector<const char*> getRequiredExtensions() {
	std::vector<const char*> extensions;

	if (enableValidationLayers) {
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
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

void createInstance() {
	if (enableValidationLayers && !checkValidationLayerSupport()) {
		throw std::runtime_error("validation layers creation failed");
	}

	VkApplicationInfo applicationInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
	applicationInfo.pApplicationName = "VkFFT";
	applicationInfo.applicationVersion = 1.0;
	applicationInfo.pEngineName = "VkFFT";
	applicationInfo.engineVersion = 1.0;
	applicationInfo.apiVersion = VK_API_VERSION_1_0;

	VkInstanceCreateInfo createInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
	createInfo.flags = 0;
	createInfo.pApplicationInfo = &applicationInfo;

	auto extensions = getRequiredExtensions();
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

void createDevice() {

	VkDeviceQueueCreateInfo queueCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
	queueFamilyIndex = getComputeQueueFamilyIndex();
	queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
	queueCreateInfo.queueCount = 1;
	float queuePriorities = 1.0;
	queueCreateInfo.pQueuePriorities = &queuePriorities;
	VkDeviceCreateInfo deviceCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
	VkPhysicalDeviceFeatures deviceFeatures = {};
	deviceFeatures.shaderFloat64 = true;
	deviceCreateInfo.enabledLayerCount = enabledLayers.size();
	deviceCreateInfo.ppEnabledLayerNames = enabledLayers.data();
	deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
	deviceCreateInfo.queueCreateInfoCount = 1;
	deviceCreateInfo.pEnabledFeatures = &deviceFeatures;
	vkCreateDevice(physicalDevice, &deviceCreateInfo, NULL, &device);
	vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

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

float performVulkanFFT(VkFFTApplication* app_forward, uint32_t batch) {
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
	}
	vkEndCommandBuffer(commandBuffer);
	VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;
	auto timeSubmit = std::chrono::steady_clock::now();
	vkQueueSubmit(queue, 1, &submitInfo, fence);
	vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000);
	auto timeEnd = std::chrono::steady_clock::now();
	float totTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001;
	vkResetFences(device, 1, &fence);
	vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
	return totTime / batch;
}

void main()
{
	createInstance();
	setupDebugMessenger();
	findPhysicalDevice(0);
	createDevice();

	VkFenceCreateInfo fenceCreateInfo = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
	fenceCreateInfo.flags = 0;
	vkCreateFence(device, &fenceCreateInfo, NULL, &fence);
	VkCommandPoolCreateInfo commandPoolCreateInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
	commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;
	vkCreateCommandPool(device, &commandPoolCreateInfo, NULL, &commandPool);
	vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &physicalDeviceMemoryProperties);

	const int num_benchmark_samples = 60;
	const int num_runs = 1;

	int benchmark_dimensions[num_benchmark_samples][4] = { {(uint32_t)pow(2,5), 1, 1, 1},{(uint32_t)pow(2,6), 1, 1, 1},{(uint32_t)pow(2,7), 1, 1, 1},{(uint32_t)pow(2,8), 1, 1, 1},{(uint32_t)pow(2,9), 1, 1, 1},{(uint32_t)pow(2,10), 1, 1, 1},
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
			cufftHandle planZ2Z;
			cufftDoubleComplex* dataC;
			cufftDoubleComplex* inputC;
			fftw_complex* inputC_double;
			int dims[3] = { benchmark_dimensions[n][0] , benchmark_dimensions[n][1] ,benchmark_dimensions[n][2] };

			inputC = (cufftDoubleComplex*)(malloc(sizeof(cufftDoubleComplex) * dims[0] * dims[1] * dims[2]));
			inputC_double = (fftw_complex*)(malloc(sizeof(fftw_complex) * dims[0] * dims[1] * dims[2]));
			for (int l = 0; l < dims[2]; l++) {
				for (int j = 0; j < dims[1]; j++) {
					for (int i = 0; i < dims[0]; i++) {
						inputC[i + j * dims[0] + l * dims[0] * dims[1]].x = 2 * ((double)rand()) / RAND_MAX - 1.0;
						inputC[i + j * dims[0] + l * dims[0] * dims[1]].y = 2 * ((double)rand()) / RAND_MAX - 1.0;
						inputC_double[i + j * dims[0] + l * dims[0] * dims[1]][0] = (double)inputC[i + j * dims[0] + l * dims[0] * dims[1]].x;
						inputC_double[i + j * dims[0] + l * dims[0] * dims[1]][1] = (double)inputC[i + j * dims[0] + l * dims[0] * dims[1]].y;
					}
				}
			}

			fftw_plan p;

			fftw_complex* output_FFTW = (fftw_complex*)(malloc(sizeof(fftw_complex) * dims[0] * dims[1] * dims[2]));

			switch (benchmark_dimensions[n][3]) {
			case 1:
				p = fftw_plan_dft_1d(benchmark_dimensions[n][0], inputC_double, output_FFTW, 1, FFTW_ESTIMATE);
				break;
			case 2:
				p = fftw_plan_dft_2d(benchmark_dimensions[n][1], benchmark_dimensions[n][0], inputC_double, output_FFTW, 1, FFTW_ESTIMATE);
				break;
			case 3:
				p = fftw_plan_dft_3d(benchmark_dimensions[n][2], benchmark_dimensions[n][1], benchmark_dimensions[n][0], inputC_double, output_FFTW, 1, FFTW_ESTIMATE);
				break;
			}

			fftw_execute(p);

			cudaMalloc((void**)&dataC, sizeof(cufftDoubleComplex) * dims[0] * dims[1] * dims[2]);

			cudaMemcpy(dataC, inputC, sizeof(cufftDoubleComplex) * dims[0] * dims[1] * dims[2], cudaMemcpyHostToDevice);
			if (cudaGetLastError() != cudaSuccess) {
				fprintf(stderr, "Cuda error: Failed to allocate\n");
				return;
			}
			switch (benchmark_dimensions[n][3]) {
			case 1:
				cufftPlan1d(&planZ2Z, dims[0], CUFFT_Z2Z, 1);
				break;
			case 2:
				cufftPlan2d(&planZ2Z, dims[1], dims[0], CUFFT_Z2Z);
				break;
			case 3:
				cufftPlan3d(&planZ2Z, dims[2], dims[1], dims[0], CUFFT_Z2Z);
				break;
			}

			float totTime = 0;
			int batch = 1;
			auto timeSubmit = std::chrono::steady_clock::now();
			cudaDeviceSynchronize();
			for (int i = 0; i < batch; i++) {

				cufftExecZ2Z(planZ2Z, dataC, dataC, 1);

			}
			cudaDeviceSynchronize();
			auto timeEnd = std::chrono::steady_clock::now();
			totTime = (std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001) / batch;

			cufftDoubleComplex* output_cuFFT = (cufftDoubleComplex*)(malloc(sizeof(cufftDoubleComplex) * dims[0] * dims[1] * dims[2]));
			cudaMemcpy(output_cuFFT, dataC, sizeof(cufftDoubleComplex) * dims[0] * dims[1] * dims[2], cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();

			//VkFFT part

			VkFFTConfiguration forward_configuration = defaultVkFFTConfiguration;
			VkFFTApplication app_forward = defaultVkFFTApplication;
			//VkFFTApplication app_inverse;
			forward_configuration.coalescedMemory = 32;//in bits, for Nvidia compute capability >=6.0 is equal to 32, <6.0 is equal 128. For Intel use 64. Gonna work regardles, but if specified by user correctly, the performance will be higher. 

			forward_configuration.FFTdim = benchmark_dimensions[n][3]; //FFT dimension, 1D, 2D or 3D (default 1).
			forward_configuration.size[0] = benchmark_dimensions[n][0]; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			forward_configuration.size[1] = benchmark_dimensions[n][1];
			forward_configuration.size[2] = benchmark_dimensions[n][2];
			//registerBoost should be disabled on machines with <256KB register file
			forward_configuration.registerBoost = 1;
			forward_configuration.performZeropadding[0] = false; //Perform padding with zeros on GPU. Still need to properly align input data (no need to fill padding area with meaningful data) but this will increase performance due to the lower amount of the memory reads/writes and omitting sequences only consisting of zeros.
			forward_configuration.performZeropadding[1] = false;
			forward_configuration.performZeropadding[2] = false;
			forward_configuration.performConvolution = false; //Perform convolution with precomputed kernel. 
			forward_configuration.performR2C = false; //Perform C2C transform. Can be combined with all other options. 
			forward_configuration.coordinateFeatures = 1; //Specify dimensionality of the input feature vector (default 1). Each component is stored not as a vector, but as a separate system and padded on it's own according to other options (i.e. for x*y system of 3-vector, first x*y elements correspond to the first dimension, then goes x*y for the second, etc). 
			forward_configuration.inverse = false; //Direction of FFT. false - forward, true - inverse.
			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [VkDeviceSize *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [VkDeviceSize *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			forward_configuration.device = &device;
			forward_configuration.queue = &queue;
			forward_configuration.fence = &fence;
			forward_configuration.commandPool = &commandPool;
			forward_configuration.physicalDevice = &physicalDevice;
			forward_configuration.doublePrecision = true;
			forward_configuration.useLUT = true;
			forward_configuration.reorderFourStep = true;
			forward_configuration.devicePageSize = 2048;
			forward_configuration.localPageSize = 16;
			//Custom path to the folder with shaders, default is "shaders");
			sprintf(forward_configuration.shaderPath, "shaders\\");
			uint32_t numBuf = 2;

			//Allocate buffers for the input data. - we use 4 in this example
			VkDeviceSize* bufferSize = (VkDeviceSize*)malloc(sizeof(VkDeviceSize) * numBuf);
			for (uint32_t i = 0; i < numBuf; i++) {
				bufferSize[i] = {};
				bufferSize[i] = ((uint64_t)forward_configuration.coordinateFeatures) * sizeof(double) * 2 * forward_configuration.size[0] * forward_configuration.size[1] * forward_configuration.size[2] / numBuf;
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
				allocateFFTBuffer(&buffer[i], &bufferDeviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize[i]);
				allocateFFTBuffer(&tempBuffer[i], &tempBufferDeviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize[i]);
			}
			forward_configuration.isInputFormatted = false; //set to true if input is a different buffer, so it can have zeropadding/R2C added  
			forward_configuration.isOutputFormatted = false;//set to true if output is a different buffer, so it can have zeropadding/C2R automatically removed

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
				transferDataFromCPU((float*)(inputC + shift / sizeof(cufftDoubleComplex)), &buffer[i], bufferSize[i]);
				shift += bufferSize[i];
			}
			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			initializeVulkanFFT(&app_forward, forward_configuration);
			//forward_configuration.inverse = true;
			//app_inverse.initializeVulkanFFT(forward_configuration);
			//Submit FFT+iFFT.
			//batch = 1;
			totTime = performVulkanFFT(&app_forward, batch);
			//totTime = performVulkanFFT(&app_inverse, batch);
			fftw_complex* output_VkFFT = (fftw_complex*)(malloc(sizeof(fftw_complex) * dims[0] * dims[1] * dims[2]));

			//Transfer data from GPU using staging buffer.
			shift = 0;
			for (uint32_t i = 0; i < numBuf; i++) {
				transferDataToCPU((float*)(output_VkFFT + shift / sizeof(fftw_complex)), &buffer[i], bufferSize[i]);
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

						/*if (app_forward.localFFTPlan.numAxisUploads[0] > 1)
							loc_i = i / app_forward.localFFTPlan.axes[0][1].specializationConstants.fftDim + app_forward.localFFTPlan.axes[0][0].specializationConstants.fftDim * (i % app_forward.localFFTPlan.axes[0][1].specializationConstants.fftDim);
						if (app_forward.localFFTPlan.numAxisUploads[1] > 1)
							loc_j = j / app_forward.localFFTPlan.axes[1][1].specializationConstants.fftDim + app_forward.localFFTPlan.axes[1][0].specializationConstants.fftDim * (j % app_forward.localFFTPlan.axes[1][1].specializationConstants.fftDim);
						if (app_forward.localFFTPlan.numAxisUploads[2] > 1)
							loc_l = l / app_forward.localFFTPlan.axes[2][1].specializationConstants.fftDim + app_forward.localFFTPlan.axes[2][0].specializationConstants.fftDim * (l % app_forward.localFFTPlan.axes[2][1].specializationConstants.fftDim);
						*/
						//printf("%f %f - %f %f - %f %f\n", output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0], output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1], output_cuFFT[i + j * dims[0] + l * dims[0] * dims[1]].x, output_cuFFT[i + j * dims[0] + l * dims[0] * dims[1]].y, output_VkFFT[(loc_i + loc_j * dims[0]+ loc_l * dims[0] * dims[1])][0], output_VkFFT[(loc_i + loc_j * dims[0]+ loc_l * dims[0] * dims[1])][1]);

						double current_diff_x_cuFFT = (output_cuFFT[i + j * dims[0] + l * dims[0] * dims[1]].x - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0]);
						double current_diff_y_cuFFT = (output_cuFFT[i + j * dims[0] + l * dims[0] * dims[1]].y - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1]);
						double current_diff_x_VkFFT = (output_VkFFT[loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]][0] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0]);
						double current_diff_y_VkFFT = (output_VkFFT[loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]][1] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1]);

						double current_diff_norm_cuFFT = sqrt(current_diff_x_cuFFT * current_diff_x_cuFFT + current_diff_y_cuFFT * current_diff_y_cuFFT);
						double current_diff_norm_VkFFT = sqrt(current_diff_x_VkFFT * current_diff_x_VkFFT + current_diff_y_VkFFT * current_diff_y_VkFFT);
						double current_data_norm = sqrt(output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] * output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] + output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1] * output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1]);
						if (current_diff_norm_cuFFT > max_difference[0]) max_difference[0] = current_diff_norm_cuFFT;
						avg_difference[0] += current_diff_norm_cuFFT;

						if ((current_diff_norm_cuFFT / current_data_norm > max_eps[0]) && (current_data_norm > 1e-16)) {
							//printf("%f %f - %f %f\n", output_cuFFT[i + j * dims[0]].x, output_cuFFT[i + j * dims[0]].y, output_VkFFT[i + j * dims[0]].x, output_VkFFT[i + j * dims[0]].y);

							max_eps[0] = current_diff_norm_cuFFT / current_data_norm;
						}
						avg_eps[0] += (current_data_norm > 1e-10) ? current_diff_norm_cuFFT / current_data_norm : 0;

						if (current_diff_norm_VkFFT > max_difference[1]) max_difference[1] = current_diff_norm_VkFFT;
						avg_difference[1] += current_diff_norm_VkFFT;

						if ((current_diff_norm_VkFFT / current_data_norm > max_eps[1]) && (current_data_norm > 1e-16)) {
							//printf("%f %f - %f %f\n", output_cuFFT[i + j * dims[0]].x, output_cuFFT[i + j * dims[0]].y, output_VkFFT[i + j * dims[0]].x, output_VkFFT[i + j * dims[0]].y);

							max_eps[1] = current_diff_norm_VkFFT / current_data_norm;
						}
						avg_eps[1] += (current_data_norm > 1e-10) ? current_diff_norm_VkFFT / current_data_norm : 0;
					}
					//printf("\n");
				}
			}
			avg_difference[0] /= (dims[0] * dims[1] * dims[2]);
			avg_eps[0] /= (dims[0] * dims[1] * dims[2]);
			avg_difference[1] /= (dims[0] * dims[1] * dims[2]);
			avg_eps[1] /= (dims[0] * dims[1] * dims[2]);
			printf("cuFFT System: %dx%dx%d avg_difference: %.15f max_difference: %.15f avg_eps: %.15f max_eps: %.15f\n", dims[0], dims[1], dims[2], avg_difference[0], max_difference[0], avg_eps[0], max_eps[0]);
			printf("VkFFT System: %dx%dx%d avg_difference: %.15f max_difference: %.15f avg_eps: %.15f max_eps: %.15f\n", dims[0], dims[1], dims[2], avg_difference[1], max_difference[1], avg_eps[1], max_eps[1]);
			free(output_cuFFT);
			free(output_VkFFT);
			for (uint32_t i = 0; i < numBuf; i++) {

				vkDestroyBuffer(device, buffer[i], NULL);
				vkDestroyBuffer(device, tempBuffer[i], NULL);
				vkFreeMemory(device, bufferDeviceMemory[i], NULL);
				vkFreeMemory(device, tempBufferDeviceMemory[i], NULL);

			}
			deleteVulkanFFT(&app_forward);
			cufftDestroy(planZ2Z);
			cudaFree(dataC);
			free(inputC);
			fftw_destroy_plan(p);
			free(inputC_double);
			free(output_FFTW);
		}
	}

}
