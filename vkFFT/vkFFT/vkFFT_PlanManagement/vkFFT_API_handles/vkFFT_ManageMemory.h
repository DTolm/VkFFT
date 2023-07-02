// This file is part of VkFFT
//
// Copyright (C) 2021 - present Dmitrii Tolmachev <dtolm96@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
#ifndef VKFFT_MANAGEMEMORY_H
#define VKFFT_MANAGEMEMORY_H
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT/vkFFT_AppManagement/vkFFT_DeleteApp.h"

#if(VKFFT_BACKEND==0)
static inline VkFFTResult findMemoryType(VkFFTApplication* app, uint64_t memoryTypeBits, uint64_t memorySize, VkMemoryPropertyFlags properties, uint32_t* memoryTypeIndex) {
	VkPhysicalDeviceMemoryProperties memoryProperties = { 0 };

	vkGetPhysicalDeviceMemoryProperties(app->configuration.physicalDevice[0], &memoryProperties);

	for (uint64_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
		if ((memoryTypeBits & ((uint64_t)1 << i)) && ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) && (memoryProperties.memoryHeaps[memoryProperties.memoryTypes[i].heapIndex].size >= memorySize))
		{
			memoryTypeIndex[0] = (uint32_t)i;
			return VKFFT_SUCCESS;
		}
	}
	return VKFFT_ERROR_FAILED_TO_FIND_MEMORY;
}
static inline VkFFTResult allocateBufferVulkan(VkFFTApplication* app, VkBuffer* buffer, VkDeviceMemory* deviceMemory, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags propertyFlags, VkDeviceSize size) {
	VkFFTResult resFFT = VKFFT_SUCCESS;
	VkResult res = VK_SUCCESS;
	uint32_t queueFamilyIndices;
	VkBufferCreateInfo bufferCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	bufferCreateInfo.queueFamilyIndexCount = 1;
	bufferCreateInfo.pQueueFamilyIndices = &queueFamilyIndices;
	bufferCreateInfo.size = size;
	bufferCreateInfo.usage = usageFlags;
	res = vkCreateBuffer(app->configuration.device[0], &bufferCreateInfo, 0, buffer);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_BUFFER;
	VkMemoryRequirements memoryRequirements = { 0 };
	vkGetBufferMemoryRequirements(app->configuration.device[0], buffer[0], &memoryRequirements);
	VkMemoryAllocateInfo memoryAllocateInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
	memoryAllocateInfo.allocationSize = memoryRequirements.size;
	resFFT = findMemoryType(app, memoryRequirements.memoryTypeBits, memoryRequirements.size, propertyFlags, &memoryAllocateInfo.memoryTypeIndex);
	if (resFFT != VKFFT_SUCCESS) return resFFT;
	res = vkAllocateMemory(app->configuration.device[0], &memoryAllocateInfo, 0, deviceMemory);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE_MEMORY;
	res = vkBindBufferMemory(app->configuration.device[0], buffer[0], deviceMemory[0], 0);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_BIND_BUFFER_MEMORY;
	return resFFT;
}
#endif

static inline VkFFTResult VkFFT_TransferDataFromCPU(VkFFTApplication* app, void* cpu_arr, void* input_buffer, uint64_t transferSize) {
	VkFFTResult resFFT = VKFFT_SUCCESS;
#if(VKFFT_BACKEND==0)
	VkBuffer* buffer = (VkBuffer*)input_buffer;
	VkDeviceSize bufferSize = transferSize;
	VkResult res = VK_SUCCESS;
	VkDeviceSize stagingBufferSize = bufferSize;
	VkBuffer stagingBuffer = { 0 };
	VkDeviceMemory stagingBufferMemory = { 0 };
	resFFT = allocateBufferVulkan(app, &stagingBuffer, &stagingBufferMemory, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBufferSize);
	if (resFFT != VKFFT_SUCCESS) return resFFT;
	void* data;
	res = vkMapMemory(app->configuration.device[0], stagingBufferMemory, 0, stagingBufferSize, 0, &data);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_MAP_MEMORY;
	memcpy(data, cpu_arr, stagingBufferSize);
	vkUnmapMemory(app->configuration.device[0], stagingBufferMemory);
	VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	commandBufferAllocateInfo.commandPool = app->configuration.commandPool[0];
	commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferAllocateInfo.commandBufferCount = 1;
	VkCommandBuffer commandBuffer = { 0 };
	res = vkAllocateCommandBuffers(app->configuration.device[0], &commandBufferAllocateInfo, &commandBuffer);
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
	res = vkQueueSubmit(app->configuration.queue[0], 1, &submitInfo, app->configuration.fence[0]);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_SUBMIT_QUEUE;
	res = vkWaitForFences(app->configuration.device[0], 1, app->configuration.fence, VK_TRUE, 100000000000);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_WAIT_FOR_FENCES;
	res = vkResetFences(app->configuration.device[0], 1, app->configuration.fence);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_RESET_FENCES;
	vkFreeCommandBuffers(app->configuration.device[0], app->configuration.commandPool[0], 1, &commandBuffer);
	vkDestroyBuffer(app->configuration.device[0], stagingBuffer, 0);
	vkFreeMemory(app->configuration.device[0], stagingBufferMemory, 0);
	return resFFT;
#elif(VKFFT_BACKEND==1)
	cudaError_t res = cudaSuccess;
	void* buffer = ((void**)input_buffer)[0];
	res = cudaMemcpy(buffer, cpu_arr, transferSize, cudaMemcpyHostToDevice);
	if (res != cudaSuccess) {
		return VKFFT_ERROR_FAILED_TO_COPY;
	}
#elif(VKFFT_BACKEND==2)
	hipError_t res = hipSuccess;
	void* buffer = ((void**)input_buffer)[0];
	res = hipMemcpy(buffer, cpu_arr, transferSize, hipMemcpyHostToDevice);
	if (res != hipSuccess) {
		return VKFFT_ERROR_FAILED_TO_COPY;
	}
#elif(VKFFT_BACKEND==3)
	cl_int res = CL_SUCCESS;
	cl_mem* buffer = (cl_mem*)input_buffer;
	cl_command_queue commandQueue = clCreateCommandQueue(app->configuration.context[0], app->configuration.device[0], 0, &res);
	if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE;
	res = clEnqueueWriteBuffer(commandQueue, buffer[0], CL_TRUE, 0, transferSize, cpu_arr, 0, NULL, NULL);
	if (res != CL_SUCCESS) {
		return VKFFT_ERROR_FAILED_TO_COPY;
	}
	res = clReleaseCommandQueue(commandQueue);
	if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_RELEASE_COMMAND_QUEUE;
#elif(VKFFT_BACKEND==4)
	ze_result_t res = ZE_RESULT_SUCCESS;
	void* buffer = ((void**)input_buffer)[0];
	ze_command_queue_desc_t commandQueueCopyDesc = {
			ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
			0,
			app->configuration.commandQueueID,
			0, // index
			0, // flags
			ZE_COMMAND_QUEUE_MODE_DEFAULT,
			ZE_COMMAND_QUEUE_PRIORITY_NORMAL
	};
	ze_command_list_handle_t copyCommandList;
	res = zeCommandListCreateImmediate(app->configuration.context[0], app->configuration.device[0], &commandQueueCopyDesc, &copyCommandList);
	if (res != ZE_RESULT_SUCCESS) {
		return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
	}
	res = zeCommandListAppendMemoryCopy(copyCommandList, buffer, cpu_arr, transferSize, 0, 0, 0);
	if (res != ZE_RESULT_SUCCESS) {
		return VKFFT_ERROR_FAILED_TO_COPY;
	}
	res = zeCommandQueueSynchronize(app->configuration.commandQueue[0], UINT32_MAX);
	if (res != ZE_RESULT_SUCCESS) {
		return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
	}
#elif(VKFFT_BACKEND==5)
	MTL::Buffer* stagingBuffer = app->configuration.device->newBuffer(cpu_arr, transferSize, MTL::ResourceStorageModeShared);
	MTL::CommandBuffer* copyCommandBuffer = app->configuration.queue->commandBuffer();
	if (copyCommandBuffer == 0) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
	MTL::BlitCommandEncoder* blitCommandEncoder = copyCommandBuffer->blitCommandEncoder();
	if (blitCommandEncoder == 0) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
	MTL::Buffer* buffer = ((MTL::Buffer**)input_buffer)[0];
	blitCommandEncoder->copyFromBuffer((MTL::Buffer*)stagingBuffer, 0, (MTL::Buffer*)buffer, 0, transferSize);
	blitCommandEncoder->endEncoding();
	copyCommandBuffer->commit();
	copyCommandBuffer->waitUntilCompleted();
	blitCommandEncoder->release();
	copyCommandBuffer->release();
	stagingBuffer->release();
#endif
	return resFFT;
}
static inline VkFFTResult VkFFT_TransferDataToCPU(VkFFTApplication* app, void* cpu_arr, void* output_buffer, uint64_t transferSize) {
	VkFFTResult resFFT = VKFFT_SUCCESS;
#if(VKFFT_BACKEND==0)
	VkBuffer* buffer = (VkBuffer*)output_buffer;
	VkDeviceSize bufferSize = transferSize;
	VkResult res = VK_SUCCESS;
	uint64_t stagingBufferSize = bufferSize;
	VkBuffer stagingBuffer = { 0 };
	VkDeviceMemory stagingBufferMemory = { 0 };
	resFFT = allocateBufferVulkan(app, &stagingBuffer, &stagingBufferMemory, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBufferSize);
	if (resFFT != VKFFT_SUCCESS) return resFFT;
	VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	commandBufferAllocateInfo.commandPool = app->configuration.commandPool[0];
	commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferAllocateInfo.commandBufferCount = 1;
	VkCommandBuffer commandBuffer = { 0 };
	res = vkAllocateCommandBuffers(app->configuration.device[0], &commandBufferAllocateInfo, &commandBuffer);
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
	res = vkQueueSubmit(app->configuration.queue[0], 1, &submitInfo, app->configuration.fence[0]);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_SUBMIT_QUEUE;
	res = vkWaitForFences(app->configuration.device[0], 1, app->configuration.fence, VK_TRUE, 100000000000);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_WAIT_FOR_FENCES;
	res = vkResetFences(app->configuration.device[0], 1, app->configuration.fence);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_RESET_FENCES;
	vkFreeCommandBuffers(app->configuration.device[0], app->configuration.commandPool[0], 1, &commandBuffer);
	void* data;
	res = vkMapMemory(app->configuration.device[0], stagingBufferMemory, 0, stagingBufferSize, 0, &data);
	if (resFFT != VKFFT_SUCCESS) return resFFT;
	memcpy(cpu_arr, data, stagingBufferSize);
	vkUnmapMemory(app->configuration.device[0], stagingBufferMemory);
	vkDestroyBuffer(app->configuration.device[0], stagingBuffer, 0);
	vkFreeMemory(app->configuration.device[0], stagingBufferMemory, 0);
#elif(VKFFT_BACKEND==1)
	cudaError_t res = cudaSuccess;
	void* buffer = ((void**)output_buffer)[0];
	res = cudaMemcpy(cpu_arr, buffer, transferSize, cudaMemcpyDeviceToHost);
	if (res != cudaSuccess) {
		return VKFFT_ERROR_FAILED_TO_COPY;
	}
#elif(VKFFT_BACKEND==2)
	hipError_t res = hipSuccess;
	void* buffer = ((void**)output_buffer)[0];
	res = hipMemcpy(cpu_arr, buffer, transferSize, hipMemcpyDeviceToHost);
	if (res != hipSuccess) {
		return VKFFT_ERROR_FAILED_TO_COPY;
	}
#elif(VKFFT_BACKEND==3)
	cl_int res = CL_SUCCESS;
	cl_mem* buffer = (cl_mem*)output_buffer;
	cl_command_queue commandQueue = clCreateCommandQueue(app->configuration.context[0], app->configuration.device[0], 0, &res);
	if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE;
	res = clEnqueueReadBuffer(commandQueue, buffer[0], CL_TRUE, 0, transferSize, cpu_arr, 0, NULL, NULL);
	if (res != CL_SUCCESS) {
		return VKFFT_ERROR_FAILED_TO_COPY;
	}
	res = clReleaseCommandQueue(commandQueue);
	if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_RELEASE_COMMAND_QUEUE;
#elif(VKFFT_BACKEND==4)
	ze_result_t res = ZE_RESULT_SUCCESS;
	void* buffer = ((void**)output_buffer)[0];
	ze_command_queue_desc_t commandQueueCopyDesc = {
			ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
			0,
			app->configuration.commandQueueID,
			0, // index
			0, // flags
			ZE_COMMAND_QUEUE_MODE_DEFAULT,
			ZE_COMMAND_QUEUE_PRIORITY_NORMAL
	};
	ze_command_list_handle_t copyCommandList;
	res = zeCommandListCreateImmediate(app->configuration.context[0], app->configuration.device[0], &commandQueueCopyDesc, &copyCommandList);
	if (res != ZE_RESULT_SUCCESS) {
		return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
	}
	res = zeCommandListAppendMemoryCopy(copyCommandList, cpu_arr, buffer, transferSize, 0, 0, 0);
	if (res != ZE_RESULT_SUCCESS) {
		return VKFFT_ERROR_FAILED_TO_COPY;
	}
	res = zeCommandQueueSynchronize(app->configuration.commandQueue[0], UINT32_MAX);
	if (res != ZE_RESULT_SUCCESS) {
		return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
	}
#elif(VKFFT_BACKEND==5)
	MTL::Buffer* stagingBuffer = app->configuration.device->newBuffer(transferSize, MTL::ResourceStorageModeShared);
	MTL::CommandBuffer* copyCommandBuffer = app->configuration.queue->commandBuffer();
	if (copyCommandBuffer == 0) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
	MTL::BlitCommandEncoder* blitCommandEncoder = copyCommandBuffer->blitCommandEncoder();
	if (blitCommandEncoder == 0) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
	MTL::Buffer* buffer = ((MTL::Buffer**)output_buffer)[0];
	blitCommandEncoder->copyFromBuffer((MTL::Buffer*)buffer, 0, (MTL::Buffer*)stagingBuffer, 0, transferSize);
	blitCommandEncoder->endEncoding();
	copyCommandBuffer->commit();
	copyCommandBuffer->waitUntilCompleted();
	blitCommandEncoder->release();
	copyCommandBuffer->release();
	memcpy(cpu_arr, stagingBuffer->contents(), transferSize);
	stagingBuffer->release();
#endif
	return resFFT;
}

#endif
