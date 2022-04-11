#ifndef VKFFT_UTILS_H
#define VKFFT_UTILS_H
#include "vkFFT.h"
#include <vector>
typedef struct {
#if(VKFFT_BACKEND==0)
	VkInstance instance;//a connection between the application and the Vulkan library 
	VkPhysicalDevice physicalDevice;//a handle for the graphics card used in the application
	VkPhysicalDeviceProperties physicalDeviceProperties;//bastic device properties
	VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;//bastic memory properties of the device
	VkDevice device;//a logical device, interacting with physical device
	VkDebugUtilsMessengerEXT debugMessenger;//extension for debugging
	uint64_t queueFamilyIndex;//if multiple queues are available, specify the used one
	VkQueue queue;//a place, where all operations are submitted
	VkCommandPool commandPool;//an opaque objects that command buffer memory is allocated from
	VkFence fence;//a vkGPU->fence used to synchronize dispatches
	std::vector<const char*> enabledDeviceExtensions;
	uint64_t enableValidationLayers;
#elif(VKFFT_BACKEND==1)
	CUdevice device;
	CUcontext context;
#elif(VKFFT_BACKEND==2)
	hipDevice_t device;
	hipCtx_t context;
#elif(VKFFT_BACKEND==3)
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue commandQueue;
#elif(VKFFT_BACKEND==4)
	ze_driver_handle_t driver;
	ze_device_handle_t device;
	ze_context_handle_t context;
	ze_command_queue_handle_t commandQueue;
	uint32_t commandQueueID;
#endif
	uint64_t device_id;//an id of a device, reported by Vulkan device list
} VkGPU;//an example structure containing Vulkan primitives

typedef struct {
	uint64_t X;
	uint64_t Y;
	uint64_t Z;
	uint64_t P;
	uint64_t B;
	uint64_t N;
	uint64_t R2C;
	uint64_t DCT;
	uint64_t saveApplicationToString;
	uint64_t loadApplicationFromString;
} VkFFTUserSystemParameters;//an example structure used to pass user-defined system for benchmarking

#if(VKFFT_BACKEND==0)
VkResult CreateDebugUtilsMessengerEXT(VkGPU* vkGPU, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger);
void DestroyDebugUtilsMessengerEXT(VkGPU* vkGPU, const VkAllocationCallbacks* pAllocator);
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData);
VkResult setupDebugMessenger(VkGPU* vkGPU);
VkResult checkValidationLayerSupport();
std::vector<const char*> getRequiredExtensions(VkGPU* vkGPU, uint64_t sample_id);
VkResult createInstance(VkGPU* vkGPU, uint64_t sample_id);
VkResult findPhysicalDevice(VkGPU* vkGPU);
VkResult getComputeQueueFamilyIndex(VkGPU* vkGPU);
VkResult createDevice(VkGPU* vkGPU, uint64_t sample_id);
VkResult createFence(VkGPU* vkGPU);
VkResult createCommandPool(VkGPU* vkGPU);
VkFFTResult findMemoryType(VkGPU* vkGPU, uint64_t memoryTypeBits, uint64_t memorySize, VkMemoryPropertyFlags properties, uint32_t* memoryTypeIndex);
VkFFTResult allocateBuffer(VkGPU* vkGPU, VkBuffer* buffer, VkDeviceMemory* deviceMemory, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags propertyFlags, uint64_t size);
VkFFTResult transferDataFromCPU(VkGPU* vkGPU, void* arr, VkBuffer* buffer, uint64_t bufferSize);
VkFFTResult transferDataToCPU(VkGPU* vkGPU, void* arr, VkBuffer* buffer, uint64_t bufferSize);
#endif
VkFFTResult devices_list();
VkFFTResult performVulkanFFT(VkGPU* vkGPU, VkFFTApplication* app, VkFFTLaunchParams* launchParams, int inverse, uint64_t num_iter);
VkFFTResult performVulkanFFTiFFT(VkGPU* vkGPU, VkFFTApplication* app, VkFFTLaunchParams* launchParams, uint64_t num_iter, double* time_result);
#endif