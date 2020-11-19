// This file is part of VkFFT, a Vulkan Fast Fourier Transform library
//
// Copyright (C) 2020 Dmitrii Tolmachev <dtolm96@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/. 
#ifdef __cplusplus
extern "C" {
#endif
#include <memory.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "vulkan/vulkan.h"

	typedef struct {
		//WHDCN layout
		uint32_t size[3]; // WHD -system dimensions 
		uint32_t maxComputeWorkGroupCount[3]; // maxComputeWorkGroupCount from VkPhysicalDeviceLimits
		uint32_t coordinateFeatures; // C - coordinate, or dimension of features vector. In matrix convolution - size of vector
		uint32_t matrixConvolution; //if equal to 2 perform 2x2, if equal to 3 perform 3x3 matrix-vector convolution. Overrides coordinateFeatures

		uint32_t numberBatches;// N - used to perform multiple batches of initial data
		uint32_t numberKernels;// N - only used in convolution step - specify how many kernels were initialized before. Expands one input to multiple (batched) output
		uint32_t FFTdim; //FFT dimensionality (1, 2 or 3)
		uint32_t radix; //FFT radix (8)
		VkBool32 performZeropadding[3]; // perform zeropadding (0 - off, 1 - on)
		VkBool32 performTranspose[2]; //will be selected automatically
		VkBool32 performConvolution; //perform convolution in this application (0 - off, 1 - on)
		VkBool32 performR2C; //perform R2C/C2R decomposition (0 - off, 1 - on)
		VkBool32 inverse; //perform inverse FFT (0 - forward, 1 - inverse)
		VkBool32 symmetricKernel; //specify if kernel in 2x2 or 3x3 matrix convolution is symmetric
		VkBool32 isInputFormatted; //specify if input buffer is not padded for R2C if out-of-place mode is selected (only if numberBatches==1 and numberKernels==1) - 0 - padded, 1 - not padded
		VkBool32 isOutputFormatted; //specify if output buffer is not padded for R2C if out-of-place mode is selected (only if numberBatches==1 and numberKernels==1) - 0 - padded, 1 - not padded
		VkBool32 reorderFourStep; //1 to enable - unshuffles four step algorithm. Requires additional buffer allocation
		VkBool32 useLUT; //1 to enable - switches from calculating sincos to using precomputed LUT tables
		VkBool32 doublePrecision; //1 to enable
		VkBool32 halfPrecision; //1 to enable
		uint32_t sharedMemorySize;//in bytes. For now Vulkan is optimized for 32KB of shared memory
		uint32_t registerBoost; //specify if register file size is bigger than shared memory (on Nvidia 256KB register file can be used instead of 32KB of shared memory, set this constant to 4). Default 1, max 4
		uint32_t registerBoost4Step; //specify if register file overutilization should be used in big sequences (>2^14), same value or lower as above. Default 1, max 4
		uint32_t swapTo3Stage4Step; //specify at which power of 2 to switch from 2 upload to 3 upload 4-step FFT, in case if making max sequence size lower than coalesced sequence helps to combat TLB misses. Default 0 - disabled. Must be at least 17
		uint32_t performHalfBandwidthBoost;//try to reduce coalsesced number by a factor of 2 to get bigger sequence in one upload
		char shaderPath[256]; //path to shaders, can be selected automatically in CMake
		uint32_t coalescedMemory;//in bits, for Nvidia compute capability >=6.0 is equal to 32, <6.0 and Intel is equal 128. Gonna work regardles, but if specified by user correctly, the performance will be higher. 
		VkDevice* device;
		VkQueue* queue;
		VkCommandPool* commandPool;
		VkFence* fence;
		VkPhysicalDevice* physicalDevice;

		VkDeviceSize* bufferSize;
		VkDeviceSize* inputBufferSize;
		VkDeviceSize* outputBufferSize;

		VkBuffer* buffer;
		VkBuffer* tempBuffer;//needed if reorderFourStep is enabled to transpose the array. Same size as buffer
		VkBuffer* inputBuffer;
		VkBuffer* outputBuffer;

		VkDeviceSize* kernelSize;
		VkBuffer* kernel;
	} VkFFTConfiguration;

	VkFFTConfiguration defaultVkFFTConfiguration = { {1,1,1}, {65535,65535,65535},1,1,1,1,1,8,{0,0,0}, {0,0},0,0,0,0,0,0,0,0,0, 0, 32768, 1, 1, 0, 1,"shaders/", 32, 0,0,0,0,0, 0,0,0,0,0,0,0,0,0 };

	typedef struct {
		uint32_t localSize[3];
		uint32_t fftDim;
		VkBool32 inverse;
		VkBool32 zeropad[2];
		uint32_t inputStride[5];
		uint32_t outputStride[5];
		uint32_t fft_dim_full;
		uint32_t stageStartSize;
		uint32_t firstStageStartSize;
		uint32_t fft_dim_x;
		uint32_t numStages;
		uint32_t stageRadix[2];
		uint32_t inputOffset;
		uint32_t outputOffset;
		VkBool32 reorderFourStep;
		uint32_t performWorkGroupShift[3];
	} VkFFTSpecializationConstantsLayout;

	typedef struct {
		uint32_t coordinate;
		uint32_t batch;
		uint32_t workGroupShift[3];
	} VkFFTPushConstantsLayout;

	typedef struct {
		uint32_t localSize[3];
		uint32_t inputStride[5];
		uint32_t ratio;
		VkBool32 ratioDirection;
	} VkFFTTransposeSpecializationConstantsLayout;
	typedef struct {
		uint32_t axisBlock[4];
		uint32_t groupedBatch;
		VkFFTSpecializationConstantsLayout specializationConstants;
		VkFFTPushConstantsLayout pushConstants;
		VkDescriptorPool descriptorPool;
		VkDescriptorSetLayout descriptorSetLayout;
		VkDescriptorSet descriptorSet;
		VkPipelineLayout pipelineLayout;
		VkPipeline pipeline;
		VkDeviceSize bufferLUTSize;
		VkBuffer bufferLUT;
		VkDeviceMemory bufferLUTDeviceMemory;
	} VkFFTAxis;
	typedef struct {
		uint32_t transposeBlock[3];
		VkFFTTransposeSpecializationConstantsLayout specializationConstants;
		VkFFTPushConstantsLayout pushConstants;
		VkDescriptorPool descriptorPool;
		VkDescriptorSetLayout descriptorSetLayout;
		VkDescriptorSet descriptorSet;
		VkPipelineLayout pipelineLayout;
		VkPipeline pipeline;
	} VkFFTTranspose;
	typedef struct {
		uint32_t numAxisUploads[3];
		uint32_t axisSplit[3][5];
		uint32_t numSupportAxisUploads[2];
		VkFFTAxis axes[3][5];
		VkFFTAxis supportAxes[2][5];//Nx/2+1 for r2c/c2r
		VkFFTTranspose transpose[2];

	} VkFFTPlan;
	typedef struct {
		VkFFTConfiguration configuration;
		VkFFTPlan localFFTPlan;
		VkFFTPlan localFFTPlan_inverse_convolution; //additional inverse plan for convolution.
	} VkFFTApplication;
	VkFFTApplication defaultVkFFTApplication = { {}, {}, {} };
	uint32_t* VkFFTReadShader(uint32_t* length, const char* filename) {

		FILE* fp = fopen(filename, "rb");
		if (fp == NULL) {
			printf("Could not find or open file: %s\n", filename);
		}

		// get file size.
		fseek(fp, 0, SEEK_END);
		long filesize = ftell(fp);
		fseek(fp, 0, SEEK_SET);

		long filesizepadded = ((long)ceil(filesize / 4.0)) * 4;

		char* str = (char*)malloc(sizeof(char) * filesizepadded);
		fread(str, filesize, sizeof(char), fp);
		fclose(fp);

		for (long i = filesize; i < filesizepadded; i++) {
			str[i] = 0;
		}

		length[0] = filesizepadded;
		return (uint32_t*)str;
	}
	void VkFFTInitShader(VkFFTApplication* app, uint32_t shader_id, VkShaderModule* shaderModule) {
		char filename[512];
		char precision[20];
		if (app->configuration.doublePrecision)
			if (app->configuration.useLUT)
				sprintf(precision, "double/LUT/");
			else
				sprintf(precision, "double/sincos/");
		else
			if (app->configuration.halfPrecision)
				if (app->configuration.useLUT)
					sprintf(precision, "half/LUT/");
				else
					sprintf(precision, "half/sincos/");
			else
				if (app->configuration.useLUT)
					sprintf(precision, "float/LUT/");
				else
					sprintf(precision, "float/sincos/");
		switch (shader_id) {
		case 0:
			//printf("vkFFT_single_c2c\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "vkFFT_single_c2c.spv");
			break;
		case 1:
			//printf("vkFFT_single_c2r\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "vkFFT_single_c2r.spv");
			break;
		case 2:
			//printf("vkFFT_single_c2c_strided\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "vkFFT_single_c2c_strided.spv");
			break;
		case 3:
			//printf("vkFFT_single_r2c\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "vkFFT_single_r2c.spv");
			break;
		case 4:
			//printf("vkFFT_single_r2c_zp\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "vkFFT_single_r2c_zp.spv");
			break;
		case 5:
			//printf("vkFFT_single_c2c_afterR2C\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "vkFFT_single_c2c_afterR2C.spv");
			break;
		case 6:
			//printf("vkFFT_single_c2c_beforeC2R\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "vkFFT_single_c2c_beforeC2R.spv");
			break;
		case 7:
			//printf("vkFFT_grouped_c2c\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "vkFFT_grouped_c2c.spv");
			break;
		case 8:
			//printf("vkFFT_grouped_convolution_1x1\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "vkFFT_grouped_convolution_1x1.spv");
			break;
		case 9:
			//printf("vkFFT_single_convolution_1x1\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "vkFFT_single_convolution_1x1.spv");
			break;
		case 10:
			//printf("vkFFT_single_strided_convolution_1x1\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "vkFFT_single_strided_convolution_1x1.spv");
			break;
		case 11:
			//printf("vkFFT_grouped_convolution_symmetric_2x2\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "vkFFT_grouped_convolution_symmetric_2x2.spv");
			break;
		case 12:
			//printf("vkFFT_single_convolution_symmetric_2x2\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "vkFFT_single_convolution_symmetric_2x2.spv");
			break;
		case 13:
			//printf("vkFFT_single_strided_convolution_symmetric_2x2\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "vkFFT_single_strided_convolution_symmetric_2x2.spv");
			break;
		case 14:
			//printf("vkFFT_grouped_convolution_nonsymmetric_2x2\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "vkFFT_grouped_convolution_nonsymmetric_2x2.spv");
			break;
		case 15:
			//printf("vkFFT_single_convolution_nonsymmetric_2x2\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "vkFFT_single_convolution_nonsymmetric_2x2.spv");
			break;
		case 16:
			//printf("vkFFT_single_strided_convolution_nonsymmetric_2x2\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "vkFFT_single_strided_convolution_nonsymmetric_2x2.spv");
			break;
		case 17:
			//printf("vkFFT_grouped_convolution_symmetric_3x3\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "vkFFT_grouped_convolution_symmetric_3x3.spv");
			break;
		case 18:
			//printf("vkFFT_single_convolution_symmetric_3x3\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "vkFFT_single_convolution_symmetric_3x3.spv");
			break;
		case 19:
			//printf("vkFFT_single_strided_convolution_symmetric_3x3\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "vkFFT_single_strided_convolution_symmetric_3x3.spv");
			break;
		case 20:
			//printf("vkFFT_grouped_convolution_nonsymmetric_3x3\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "vkFFT_grouped_convolution_nonsymmetric_3x3.spv");
			break;
		case 21:
			//printf("vkFFT_single_convolution_nonsymmetric_3x3\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "vkFFT_single_convolution_nonsymmetric_3x3.spv");
			break;
		case 22:
			//printf("vkFFT_single_strided_convolution_nonsymmetric_3x3\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "vkFFT_single_strided_convolution_nonsymmetric_3x3.spv");
			break;
		case 23:
			//printf("vkFFT_single_c2r_8192\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "8192/vkFFT_single_c2r_8192.spv");
			break;
		case 24:
			//printf("vkFFT_single_r2c_8192\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "8192/vkFFT_single_r2c_8192.spv");
			break;
		case 25:
			//printf("vkFFT_single_c2c_8192\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "8192/vkFFT_single_c2c_8192.spv");
			break;
		case 26:
			//printf("vkFFT_grouped_strided_convolution_1x1\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "vkFFT_grouped_strided_convolution_1x1.spv");
			break;
		case 27:
			//printf("vkFFT_grouped_strided_convolution_symmetric_2x2\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "vkFFT_grouped_strided_convolution_symmetric_2x2.spv");
			break;
		case 28:
			//printf("vkFFT_grouped_strided_convolution_nonsymmetric_2x2\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "vkFFT_grouped_strided_convolution_nonsymmetric_2x2.spv");
			break;
		case 29:
			//printf("vkFFT_grouped_strided_convolution_symmetric_3x3\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "vkFFT_grouped_strided_convolution_symmetric_3x3.spv");
			break;
		case 30:
			//printf("vkFFT_grouped_strided_convolution_nonsymmetric_3x3\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "vkFFT_grouped_strided_convolution_nonsymmetric_3x3.spv");
			break;
		case 33:
			//printf("vkFFT_single_c2r_16384\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "16384/vkFFT_single_c2r_16384.spv");
			break;
		case 34:
			//printf("vkFFT_single_r2c_16384\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "16384/vkFFT_single_r2c_16384.spv");
			break;
		case 35:
			//printf("vkFFT_single_c2c_16384\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "16384/vkFFT_single_c2c_16384.spv");
			break;
		case 36:
			//printf("vkFFT_single_c2r_for_transposition_16384\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "16384/vkFFT_single_c2r_for_transposition_16384.spv");
			break;
		case 37:
			//printf("vkFFT_single_r2c_for_transposition_16384\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "16384/vkFFT_single_r2c_for_transposition_16384.spv");
			break;
		case 38:
			//printf("vkFFT_single_c2c_for_transposition_16384\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "16384/vkFFT_single_c2c_for_transposition_16384.spv");
			break;
		case 39:
			//printf("vkFFT_single_c2c_afterR2C_for_transposition_16384\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "16384/vkFFT_single_c2c_afterR2C_for_transposition_16384.spv");
			break;
		case 40:
			//printf("vkFFT_single_c2c_beforeC2R_for_transposition_16384\n");
			sprintf(filename, "%s%s%s", app->configuration.shaderPath, precision, "16384/vkFFT_single_c2c_beforeC2R_for_transposition_16384.spv");
			break;
		}


		uint32_t filelength;
		uint32_t* code = VkFFTReadShader(&filelength, filename);
		VkShaderModuleCreateInfo createInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
		createInfo.pCode = code;
		createInfo.codeSize = filelength;
		vkCreateShaderModule(app->configuration.device[0], &createInfo, NULL, shaderModule);
		free(code);

	}
	uint32_t findMemoryType(VkFFTApplication* app, uint32_t memoryTypeBits, uint32_t memorySize, VkMemoryPropertyFlags properties) {
		VkPhysicalDeviceMemoryProperties memoryProperties = { 0 };

		vkGetPhysicalDeviceMemoryProperties(app->configuration.physicalDevice[0], &memoryProperties);

		for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
			if ((memoryTypeBits & (1 << i)) && ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) && (memoryProperties.memoryHeaps[memoryProperties.memoryTypes[i].heapIndex].size >= memorySize))
				return i;
		}
		return -1;
	}
	void allocateFFTBuffer(VkFFTApplication* app, VkBuffer* buffer, VkDeviceMemory* deviceMemory, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags propertyFlags, VkDeviceSize size) {
		uint32_t queueFamilyIndices;
		VkBufferCreateInfo bufferCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
		bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		bufferCreateInfo.queueFamilyIndexCount = 1;
		bufferCreateInfo.pQueueFamilyIndices = &queueFamilyIndices;
		bufferCreateInfo.size = size;
		bufferCreateInfo.usage = usageFlags;
		vkCreateBuffer(app->configuration.device[0], &bufferCreateInfo, NULL, buffer);
		VkMemoryRequirements memoryRequirements = { 0 };
		vkGetBufferMemoryRequirements(app->configuration.device[0], buffer[0], &memoryRequirements);
		VkMemoryAllocateInfo memoryAllocateInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
		memoryAllocateInfo.allocationSize = memoryRequirements.size;
		memoryAllocateInfo.memoryTypeIndex = findMemoryType(app, memoryRequirements.memoryTypeBits, memoryRequirements.size, propertyFlags);
		vkAllocateMemory(app->configuration.device[0], &memoryAllocateInfo, NULL, deviceMemory);
		vkBindBufferMemory(app->configuration.device[0], buffer[0], deviceMemory[0], 0);
	}
	void transferDataFromCPU(VkFFTApplication* app, void* arr, VkBuffer* buffer, VkDeviceSize bufferSize) {
		VkDeviceSize stagingBufferSize = bufferSize;
		VkBuffer stagingBuffer = { 0 };
		VkDeviceMemory stagingBufferMemory = { 0 };
		allocateFFTBuffer(app, &stagingBuffer, &stagingBufferMemory, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBufferSize);

		void* data;
		vkMapMemory(app->configuration.device[0], stagingBufferMemory, 0, stagingBufferSize, 0, &data);
		memcpy(data, arr, stagingBufferSize);
		vkUnmapMemory(app->configuration.device[0], stagingBufferMemory);
		VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
		commandBufferAllocateInfo.commandPool = app->configuration.commandPool[0];
		commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		commandBufferAllocateInfo.commandBufferCount = 1;
		VkCommandBuffer commandBuffer = { 0 };
		vkAllocateCommandBuffers(app->configuration.device[0], &commandBufferAllocateInfo, &commandBuffer);
		VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
		VkBufferCopy copyRegion = { 0 };
		copyRegion.srcOffset = 0;
		copyRegion.dstOffset = 0;
		copyRegion.size = stagingBufferSize;
		vkCmdCopyBuffer(commandBuffer, stagingBuffer, buffer[0], 1, &copyRegion);
		vkEndCommandBuffer(commandBuffer);
		VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
		vkQueueSubmit(app->configuration.queue[0], 1, &submitInfo, app->configuration.fence[0]);
		vkWaitForFences(app->configuration.device[0], 1, app->configuration.fence, VK_TRUE, 100000000000);
		vkResetFences(app->configuration.device[0], 1, app->configuration.fence);
		vkFreeCommandBuffers(app->configuration.device[0], app->configuration.commandPool[0], 1, &commandBuffer);
		vkDestroyBuffer(app->configuration.device[0], stagingBuffer, NULL);
		vkFreeMemory(app->configuration.device[0], stagingBufferMemory, NULL);
	}
	void VkFFTScheduler(VkFFTApplication* app, VkFFTPlan* FFTPlan, uint32_t axis_id) {
		uint32_t complexSize;
		if (app->configuration.doublePrecision)
			complexSize = (2 * sizeof(double));
		else
			if (app->configuration.halfPrecision)
				complexSize = (2 * sizeof(float));
			else
				complexSize = (2 * sizeof(float));
		uint32_t registerBoost = app->configuration.registerBoost;

		uint32_t maxSequenceLengthSharedMemory = app->configuration.sharedMemorySize / complexSize;
		uint32_t maxSingleSizeNonStrided = maxSequenceLengthSharedMemory;
		if ((axis_id == 0) && (!app->configuration.performConvolution)) maxSingleSizeNonStrided *= registerBoost;
		uint32_t maxSingleSizeStrided = (app->configuration.coalescedMemory > complexSize) ? app->configuration.sharedMemorySize / (app->configuration.coalescedMemory) : app->configuration.sharedMemorySize / complexSize;
		uint32_t numPasses = 1;
		uint32_t numPassesHalfBandwidth = 1;
		uint32_t temp;
		temp = (axis_id == 0) ? app->configuration.size[axis_id] / maxSingleSizeNonStrided : app->configuration.size[axis_id] / maxSingleSizeStrided;
		if (temp > 1) {//more passes than one
			registerBoost = app->configuration.registerBoost4Step;
			temp = ((axis_id == 0) && (!app->configuration.reorderFourStep)) ? app->configuration.size[axis_id] / maxSingleSizeNonStrided : app->configuration.size[axis_id] / maxSingleSizeStrided;
			if (app->configuration.reorderFourStep)
				numPasses = (uint32_t)ceil(log2(app->configuration.size[axis_id]) / log2(maxSingleSizeStrided));
			else
				numPasses += (uint32_t)ceil(log2(temp) / log2(maxSingleSizeStrided));
		}
		uint32_t maxSingleSizeStridedHalfBandwidth = maxSingleSizeStrided;
		if (app->configuration.performHalfBandwidthBoost) {
			maxSingleSizeStridedHalfBandwidth = (app->configuration.coalescedMemory / 2 > complexSize) ? app->configuration.sharedMemorySize / (app->configuration.coalescedMemory / 2) : app->configuration.sharedMemorySize / complexSize;
			temp = (axis_id == 0) ? app->configuration.size[axis_id] / maxSingleSizeNonStrided : app->configuration.size[axis_id] / maxSingleSizeStridedHalfBandwidth;
			if (temp > 1) {//more passes than two
				temp = ((axis_id == 0) && (!app->configuration.reorderFourStep)) ? app->configuration.size[axis_id] / maxSingleSizeNonStrided : app->configuration.size[axis_id] / maxSingleSizeStridedHalfBandwidth;
				if (app->configuration.reorderFourStep)
					numPassesHalfBandwidth = (uint32_t)ceil(log2(app->configuration.size[axis_id]) / log2(maxSingleSizeStridedHalfBandwidth));
				else
					numPassesHalfBandwidth = 1 + (uint32_t)ceil(log2(temp) / log2(maxSingleSizeStridedHalfBandwidth));
				if ((numPassesHalfBandwidth == 2)&& (!app->configuration.reorderFourStep)&&(registerBoost>1)) {//switch back for two step and don't do half bandwidth on strided accesses if register boost and no 4-step reordering
					numPassesHalfBandwidth = 1 + (uint32_t)ceil(log2(temp) / log2(maxSingleSizeStrided));
				}
			}
			if (numPassesHalfBandwidth < numPasses) numPasses = numPassesHalfBandwidth;
		}
		if (((uint32_t)log2(app->configuration.size[axis_id]) >= app->configuration.swapTo3Stage4Step)&&(app->configuration.swapTo3Stage4Step>=17)) numPasses = 3;//Force set to 3 stage 4 step algorithm
		FFTPlan->numAxisUploads[axis_id] = numPasses;
		switch (numPasses) {
		case 1: {
			FFTPlan->axisSplit[axis_id][0] = app->configuration.size[axis_id];
			break;
		}
		case 2: {
			if ((axis_id == 0) && (!app->configuration.reorderFourStep)) {
				uint32_t maxPow8SharedMemory = (uint32_t)pow(8, ((uint32_t)log2(maxSequenceLengthSharedMemory)) / 3);
				//unit stride
				if (app->configuration.size[axis_id] / maxPow8SharedMemory <= maxSingleSizeStrided) {
					FFTPlan->axisSplit[axis_id][0] = maxPow8SharedMemory;
				}
				else {
					if (app->configuration.size[axis_id] / maxSequenceLengthSharedMemory <= maxSingleSizeStrided) {
						FFTPlan->axisSplit[axis_id][0] = maxSequenceLengthSharedMemory;
					}
					else {
						if (app->configuration.size[axis_id] / (maxSequenceLengthSharedMemory * registerBoost) < maxSingleSizeStridedHalfBandwidth) {
							for (uint32_t i = 1; i <= (uint32_t)log2(registerBoost); i++) {
								if (app->configuration.size[axis_id] / (maxSequenceLengthSharedMemory * (uint32_t)pow(2, i)) <= maxSingleSizeStrided) {
									FFTPlan->axisSplit[axis_id][0] = (maxSequenceLengthSharedMemory * (uint32_t)pow(2, i));
									i = (uint32_t)log2(registerBoost) + 1;
								}
							}
						}
						else {
							FFTPlan->axisSplit[axis_id][0] = (maxSequenceLengthSharedMemory * registerBoost);
						}
					}
				}
			}
			else {
				uint32_t maxPow8Strided = (uint32_t)pow(8, ((uint32_t)log2(maxSingleSizeStrided)) / 3);
				//all FFTs are considered as non-unit stride
				if (app->configuration.size[axis_id] / maxPow8Strided <= maxSingleSizeStrided) {
					FFTPlan->axisSplit[axis_id][0] = maxPow8Strided;
				}
				else {
					if (app->configuration.size[axis_id] / maxSingleSizeStrided < maxSingleSizeStridedHalfBandwidth) {
						FFTPlan->axisSplit[axis_id][0] = maxSingleSizeStrided;
					}
					else {
						FFTPlan->axisSplit[axis_id][0] = maxSingleSizeStridedHalfBandwidth;
					}
				}
			}
			FFTPlan->axisSplit[axis_id][1] = app->configuration.size[axis_id] / FFTPlan->axisSplit[axis_id][0];
			if (FFTPlan->axisSplit[axis_id][1] < 64) {
				FFTPlan->axisSplit[axis_id][0] = (FFTPlan->axisSplit[axis_id][1] == 0) ? FFTPlan->axisSplit[axis_id][0] / (64) : FFTPlan->axisSplit[axis_id][0] / (64 / FFTPlan->axisSplit[axis_id][1]);
				FFTPlan->axisSplit[axis_id][1] = 64;
			}
			if (FFTPlan->axisSplit[axis_id][1] > FFTPlan->axisSplit[axis_id][0]) {
				uint32_t swap = FFTPlan->axisSplit[axis_id][0];
				FFTPlan->axisSplit[axis_id][0] = FFTPlan->axisSplit[axis_id][1];
				FFTPlan->axisSplit[axis_id][1] = swap;
			}
			break;
		}
		case 3: {


			uint32_t maxPow8Strided = (uint32_t)pow(8, ((uint32_t)log2(maxSingleSizeStrided)) / 3);
			if ((axis_id == 0) && (!app->configuration.reorderFourStep)) {
				//unit stride
				uint32_t maxPow8SharedMemory = (uint32_t)pow(8, ((uint32_t)log2(maxSequenceLengthSharedMemory)) / 3);
				if (app->configuration.size[axis_id] / maxPow8SharedMemory <= maxPow8Strided * maxPow8Strided)
					FFTPlan->axisSplit[axis_id][0] = maxPow8SharedMemory;
				else {
					if (app->configuration.size[axis_id] / maxSequenceLengthSharedMemory <= maxSingleSizeStrided * maxSingleSizeStrided)
						FFTPlan->axisSplit[axis_id][0] = maxSequenceLengthSharedMemory;
					else {
						if (app->configuration.size[axis_id] / (maxSequenceLengthSharedMemory * registerBoost) <= maxSingleSizeStrided * maxSingleSizeStrided) {
							for (uint32_t i = 0; i <= (uint32_t)log2(registerBoost); i++) {
								if (app->configuration.size[axis_id] / (maxSequenceLengthSharedMemory * (uint32_t)pow(2, i)) <= maxSingleSizeStrided * maxSingleSizeStrided) {
									FFTPlan->axisSplit[axis_id][0] = (maxSequenceLengthSharedMemory * (uint32_t)pow(2, i));
									i = (uint32_t)log2(registerBoost) + 1;
								}
							}
						}
						else {
							FFTPlan->axisSplit[axis_id][0] = (maxSequenceLengthSharedMemory * registerBoost);
						}
					}
				}
			}
			else {
				//to account for TLB misses, it is best to coalesce the unit-strided stage to 128 bytes
				/*uint32_t log2axis = (uint32_t)log2(app->configuration.size[axis_id]);
				FFTPlan->axisSplit[axis_id][0] = (uint32_t)pow(2, (uint32_t)log2axis / 3);
				if (log2axis % 3 > 0) FFTPlan->axisSplit[axis_id][0] *= 2;
				FFTPlan->axisSplit[axis_id][1] = (uint32_t)pow(2, (uint32_t)log2axis / 3);
				if (log2axis % 3 > 1) FFTPlan->axisSplit[axis_id][1] *= 2;
				FFTPlan->axisSplit[axis_id][2] = app->configuration.size[axis_id] / FFTPlan->axisSplit[axis_id][0] / FFTPlan->axisSplit[axis_id][1];*/
				uint32_t maxSingleSizeStrided128 = app->configuration.sharedMemorySize / (128) ;
				uint32_t maxPow8_128 = (uint32_t)pow(8, ((uint32_t)log2(maxSingleSizeStrided128)) / 3);
				//unit stride
				if (app->configuration.size[axis_id] / maxPow8_128 <= maxPow8Strided * maxSingleSizeStrided)
					FFTPlan->axisSplit[axis_id][0] = maxPow8_128;
				//non-unit stride
				else {

					if ((app->configuration.size[axis_id] / (maxPow8_128*2) <= maxPow8Strided * maxSingleSizeStrided)&&(maxPow8_128 * 2<= maxSingleSizeStrided128)) {
						FFTPlan->axisSplit[axis_id][0] = maxPow8_128 * 2;
					}
					else {
						if ((app->configuration.size[axis_id] / (maxPow8_128 * 4) <= maxPow8Strided * maxSingleSizeStrided) && (maxPow8_128 * 4 <= maxSingleSizeStrided128)) {
							FFTPlan->axisSplit[axis_id][0] = maxPow8_128 * 4;
						}
						else {
							if (app->configuration.size[axis_id] / maxSingleSizeStrided <= maxSingleSizeStrided * maxSingleSizeStrided) {
								for (uint32_t i = 0; i <= (uint32_t)log2(maxSingleSizeStrided / maxSingleSizeStrided128); i++) {
									if (app->configuration.size[axis_id] / (maxSingleSizeStrided128 * (uint32_t)pow(2, i)) <= maxSingleSizeStrided * maxSingleSizeStrided) {
										FFTPlan->axisSplit[axis_id][0] = (maxSingleSizeStrided128 * (uint32_t)pow(2, i));
										i = (uint32_t)log2(maxSingleSizeStrided / maxSingleSizeStrided128) + 1;
									}
								}
							}
							else
								FFTPlan->axisSplit[axis_id][0] = maxSingleSizeStridedHalfBandwidth;
						}
					}
				}
			}
			if (app->configuration.size[axis_id] / FFTPlan->axisSplit[axis_id][0] / maxPow8Strided <= maxSingleSizeStrided) {
				FFTPlan->axisSplit[axis_id][1] = maxPow8Strided;
				FFTPlan->axisSplit[axis_id][2] = app->configuration.size[axis_id] / FFTPlan->axisSplit[axis_id][1] / FFTPlan->axisSplit[axis_id][0];
			}
			else {
				if (app->configuration.size[axis_id] / FFTPlan->axisSplit[axis_id][0] / maxSingleSizeStrided <= maxSingleSizeStrided) {
					FFTPlan->axisSplit[axis_id][1] = maxSingleSizeStrided;
					FFTPlan->axisSplit[axis_id][2] = app->configuration.size[axis_id] / FFTPlan->axisSplit[axis_id][1] / FFTPlan->axisSplit[axis_id][0];
				}
				else {
					FFTPlan->axisSplit[axis_id][1] = maxSingleSizeStridedHalfBandwidth;
					FFTPlan->axisSplit[axis_id][2] = app->configuration.size[axis_id] / FFTPlan->axisSplit[axis_id][1] / FFTPlan->axisSplit[axis_id][0];
				}
			}
			if (FFTPlan->axisSplit[axis_id][2] < 64) {
				FFTPlan->axisSplit[axis_id][1] = (FFTPlan->axisSplit[axis_id][2] == 0 ) ? FFTPlan->axisSplit[axis_id][1] / (64) : FFTPlan->axisSplit[axis_id][1] / (64 / FFTPlan->axisSplit[axis_id][2]);
				FFTPlan->axisSplit[axis_id][2] = 64;
			}
			if (FFTPlan->axisSplit[axis_id][2] > FFTPlan->axisSplit[axis_id][1]) {
				uint32_t swap = FFTPlan->axisSplit[axis_id][1];
				FFTPlan->axisSplit[axis_id][1] = FFTPlan->axisSplit[axis_id][2];
				FFTPlan->axisSplit[axis_id][2] = swap;
			}
			break;
		}
		default: {
			printf("sequence length exceeds boundaries\n");
			break;
		}
		}
	}
	void VkFFTPlanSupportAxis(VkFFTApplication* app, VkFFTPlan* FFTPlan, uint32_t axis_id, uint32_t axis_upload_id, VkBool32 inverse) {
		//get radix stages
		VkFFTAxis* axis = &FFTPlan->supportAxes[axis_id - 1][axis_upload_id];
		uint32_t maxSequenceLengthSharedMemory;
		uint32_t maxSingleSizeNonStrided;
		if (app->configuration.doublePrecision)
			maxSequenceLengthSharedMemory = app->configuration.sharedMemorySize / (2 * sizeof(double));
		else
			if (app->configuration.halfPrecision)
				maxSequenceLengthSharedMemory = app->configuration.sharedMemorySize / (2 * sizeof(float));
			else
				maxSequenceLengthSharedMemory = app->configuration.sharedMemorySize / (2 * sizeof(float));
		maxSingleSizeNonStrided = maxSequenceLengthSharedMemory;
		if (axis_id == 1) {
			//configure radix stages
			uint32_t logSize = log2(app->configuration.size[axis_id]);
			uint32_t numPasses[8][8];//4096-8k(256KB)-16k(256KB)-32k-64k - find correct strided FFT configuration - x axis | 256-512-1024-2048(256KB)-4096(256KB)-8k(future?)-16k(future?) - find correct strided FFT configuration
			for (uint32_t i = 0; i < 8; i++) {
				for (uint32_t j = 0; j < 8; j++) {
					numPasses[i][j] = 0;
				}
			}
			uint32_t temp = app->configuration.size[axis_id];
			uint32_t startStage = maxSingleSizeNonStrided;
			uint32_t continueStage = 256;
			uint32_t maxPassId[2] = { 0,0 };
			uint32_t minPassId[2] = { 0,0 };
			maxPassId[0] += log2(app->configuration.registerBoost);
			uint32_t maxSingleSizeStrided = app->configuration.sharedMemorySize / app->configuration.coalescedMemory;
			maxPassId[1] = log2(maxSingleSizeStrided / 256);
			minPassId[1] = (maxSingleSizeStrided >= 512) ? 1 : 0;
			//maxPassId[1] += log2(app->configuration.registerBoost);//in development
			for (uint32_t i = 0; i < 8; i++) {
				for (uint32_t j = 0; j < 8; j++) {
					temp /= startStage;
					numPasses[i][j]++;
					while (temp > 1)
					{
						temp /= continueStage;
						numPasses[i][j]++;
					}
					continueStage *= 2;
					temp = app->configuration.size[axis_id];
				}
				continueStage = 256;
				startStage *= 2;
			}
			uint32_t passId[2] = { minPassId[0], minPassId[1] };
			for (uint32_t i = minPassId[0]; i < maxPassId[0] + 1; i++) {
				for (uint32_t j = minPassId[1]; j < maxPassId[1] + 1; j++) {
					if (numPasses[i][j] < numPasses[passId[0]][passId[1]]) {
						passId[0] = i;
						passId[1] = j;
					}
				}
			}
			FFTPlan->numSupportAxisUploads[axis_id - 1] = numPasses[passId[0]][passId[1]];
			if (axis_upload_id >= numPasses[passId[0]][passId[1]])
				return;
			if (axis_upload_id == 0) {
				//first pass is non-strided, special case
				switch (app->configuration.radix) {
				case 8: {
					uint32_t logSize0Pass = (log2(maxSingleSizeNonStrided) + passId[0] < logSize) ? log2(maxSingleSizeNonStrided) + passId[0] : logSize; //4096 + shift
					if ((axis_upload_id + 1 == numPasses[passId[0]][passId[1]] - 1) && (logSize - logSize0Pass < maxSingleSizeStrided))
						logSize0Pass -= (3 - (logSize - logSize0Pass));

					uint32_t stage8 = logSize0Pass / 3;
					uint32_t stage4 = 0;
					uint32_t stage2 = 0;
					if (logSize0Pass % 3 == 2)
						stage4 = 1;
					if (logSize0Pass % 3 == 1)
						stage2 = 1;
					uint32_t totNumStages = stage8 + stage4 + stage2;

					axis->specializationConstants.numStages = stage8;
					axis->specializationConstants.fftDim = pow(8, stage8);
					axis->specializationConstants.stageRadix[0] = 8;
					axis->specializationConstants.stageRadix[1] = 8;

					if (stage4 == 1) {
						axis->specializationConstants.numStages++;
						axis->specializationConstants.stageRadix[1] = 4;
						axis->specializationConstants.fftDim *= 4;
					}
					if (stage2 == 1) {
						axis->specializationConstants.numStages++;
						axis->specializationConstants.stageRadix[1] = 2;
						axis->specializationConstants.fftDim *= 2;
					}
					axis->specializationConstants.stageStartSize = 1;
					if (numPasses[passId[0]][passId[1]] > 1) {
						uint32_t logSizeLaterPass = (logSize - log2(maxSingleSizeNonStrided) - passId[0] < 3) ? 3 : logSize - log2(maxSingleSizeNonStrided) - passId[0]; //4096 + shift
						stage8 = logSizeLaterPass / 3;
						stage4 = 0;
						stage2 = 0;
						if (logSizeLaterPass % 3 == 2)
							stage4 = 1;
						if (logSizeLaterPass % 3 == 1)
							stage2 = 1;
						totNumStages = stage8 + stage4 + stage2;
						uint32_t locNumStages = 0;
						locNumStages = stage8 / (numPasses[passId[0]][passId[1]] - 1);
						if (FFTPlan->numAxisUploads[axis_id] - 1 <= stage8 % (numPasses[passId[0]][passId[1]] - 1))
							locNumStages++;
						uint32_t tempFFTDim = pow(8, locNumStages);
						if (stage4 == 1) {
							tempFFTDim *= 4;
						}
						if (stage2 == 1) {
							tempFFTDim *= 2;
						}

						axis->specializationConstants.firstStageStartSize = app->configuration.size[axis_id] / tempFFTDim;
					}
					else
						axis->specializationConstants.firstStageStartSize = 1;
					if (app->configuration.performR2C)
						axis->specializationConstants.fft_dim_x = app->configuration.size[0] / 2;
					else
						axis->specializationConstants.fft_dim_x = app->configuration.size[0];

					break;
				}
				case 4: {
					uint32_t stage4 = logSize / 2;
					uint32_t stage2 = 0;
					if (logSize % 2 == 1)
						stage2 = 1;
					axis->specializationConstants.numStages = stage4 + stage2;


					axis->specializationConstants.stageRadix[0] = 4;
					axis->specializationConstants.stageRadix[1] = 4;
					if (logSize % 2 == 1)
						axis->specializationConstants.stageRadix[1] = 2;
					break;
				}
				case 2: {
					uint32_t stage2 = logSize;

					axis->specializationConstants.numStages = stage2;


					axis->specializationConstants.stageRadix[0] = 2;
					axis->specializationConstants.stageRadix[1] = 2;
					break;
				}
				}
			}
			else {
				//passes after first are done similar to strided passes in y and z
				uint32_t logSizeLaterPass = (logSize - log2(maxSingleSizeNonStrided) - passId[0] < 3) ? 3 : logSize - log2(maxSingleSizeNonStrided) - passId[0]; //4096 + shift
				switch (app->configuration.radix) {
				case 8: {
					uint32_t stage8 = logSizeLaterPass / 3;
					uint32_t stage4 = 0;
					uint32_t stage2 = 0;
					if (logSizeLaterPass % 3 == 2)
						stage4 = 1;
					if (logSizeLaterPass % 3 == 1)
						stage2 = 1;
					uint32_t totNumStages = stage8 + stage4 + stage2;
					uint32_t locNumStages = 0;
					if (passId[1] == minPassId[1]) {
						locNumStages = stage8 / (numPasses[passId[0]][passId[1]] - 1);
						if (axis_upload_id <= stage8 % (numPasses[passId[0]][passId[1]] - 1))
							locNumStages++;
						axis->specializationConstants.numStages = locNumStages;
						axis->specializationConstants.fftDim = pow(8, locNumStages);
						axis->specializationConstants.stageRadix[0] = 8;
						axis->specializationConstants.stageRadix[1] = 8;

						if (axis_upload_id == (numPasses[passId[0]][passId[1]] - 1)) {
							if (stage4 == 1) {
								axis->specializationConstants.numStages++;
								axis->specializationConstants.stageRadix[1] = 4;
								axis->specializationConstants.fftDim *= 4;
							}
							if (stage2 == 1) {
								axis->specializationConstants.numStages++;
								axis->specializationConstants.stageRadix[1] = 2;
								axis->specializationConstants.fftDim *= 2;
							}
						}
						axis->specializationConstants.firstStageStartSize = 1;
						axis->specializationConstants.stageStartSize = (axis_upload_id == 1) ? FFTPlan->supportAxes[axis_id - 1][axis_upload_id - 1].specializationConstants.fftDim : FFTPlan->supportAxes[axis_id - 1][axis_upload_id - 1].specializationConstants.stageStartSize * FFTPlan->supportAxes[axis_id - 1][axis_upload_id - 1].specializationConstants.fftDim;
						axis->specializationConstants.fft_dim_x = app->configuration.size[1];
					}
					else {
						if (axis_upload_id < numPasses[passId[0]][passId[1]] - 1) {
							uint32_t locLogSize = 8 + passId[1];
							if ((axis_upload_id + 1 == numPasses[passId[0]][passId[1]] - 1) && (logSizeLaterPass - (8 + passId[1]) * (numPasses[passId[0]][passId[1]] - 2) < 3))
								locLogSize -= (3 - (logSizeLaterPass - (8 + passId[1]) * (numPasses[passId[0]][passId[1]] - 2)));
							uint32_t locStage8 = locLogSize / 3;
							uint32_t locStage4 = 0;
							uint32_t locStage2 = 0;
							if (locLogSize % 3 == 2)
								locStage4 = 1;
							if (locLogSize % 3 == 1)
								locStage2 = 1;
							axis->specializationConstants.numStages = locStage8 + locStage4 + locStage2;
							axis->specializationConstants.fftDim = pow(2, locLogSize);
							axis->specializationConstants.stageRadix[0] = 8;
							axis->specializationConstants.stageRadix[1] = 8;

							if (locStage4 == 1) {
								axis->specializationConstants.stageRadix[1] = 4;
							}
							if (locStage2 == 1) {
								axis->specializationConstants.stageRadix[1] = 2;
							}
							axis->specializationConstants.firstStageStartSize = 1;
							axis->specializationConstants.stageStartSize = (axis_upload_id == 1) ? FFTPlan->supportAxes[axis_id - 1][axis_upload_id - 1].specializationConstants.fftDim : FFTPlan->supportAxes[axis_id - 1][axis_upload_id - 1].specializationConstants.stageStartSize * FFTPlan->supportAxes[axis_id - 1][axis_upload_id - 1].specializationConstants.fftDim;
							if (app->configuration.performR2C)
								axis->specializationConstants.fft_dim_x = app->configuration.size[0] / 2;
							else
								axis->specializationConstants.fft_dim_x = app->configuration.size[0];
						}
						else {
							uint32_t locLogSize = (logSizeLaterPass - (8 + passId[1]) * (numPasses[passId[0]][passId[1]] - 2) < 3) ? 3 : logSizeLaterPass - (8 + passId[1]) * (numPasses[passId[0]][passId[1]] - 2);
							uint32_t locStage8 = locLogSize / 3;
							uint32_t locStage4 = 0;
							uint32_t locStage2 = 0;
							if (locLogSize % 3 == 2)
								locStage4 = 1;
							if (locLogSize % 3 == 1)
								locStage2 = 1;
							axis->specializationConstants.numStages = locStage8 + locStage4 + locStage2;
							axis->specializationConstants.fftDim = pow(2, locLogSize);
							axis->specializationConstants.stageRadix[0] = 8;
							axis->specializationConstants.stageRadix[1] = 8;

							if (locStage4 == 1) {
								axis->specializationConstants.stageRadix[1] = 4;
							}
							if (locStage2 == 1) {
								axis->specializationConstants.stageRadix[1] = 2;
							}
							axis->specializationConstants.firstStageStartSize = 1;
							axis->specializationConstants.stageStartSize = (axis_upload_id == 1) ? FFTPlan->supportAxes[axis_id - 1][axis_upload_id - 1].specializationConstants.fftDim : FFTPlan->supportAxes[axis_id - 1][axis_upload_id - 1].specializationConstants.stageStartSize * FFTPlan->supportAxes[axis_id - 1][axis_upload_id - 1].specializationConstants.fftDim;
							if (app->configuration.performR2C)
								axis->specializationConstants.fft_dim_x = app->configuration.size[0] / 2;
							else
								axis->specializationConstants.fft_dim_x = app->configuration.size[0];
						}
					}


					break;
				}
				case 4: {
					uint32_t stage4 = logSize / 2;
					uint32_t stage2 = 0;
					if (logSize % 2 == 1)
						stage2 = 1;
					axis->specializationConstants.numStages = stage4 + stage2;


					axis->specializationConstants.stageRadix[0] = 4;
					axis->specializationConstants.stageRadix[1] = 4;
					if (logSize % 2 == 1)
						axis->specializationConstants.stageRadix[1] = 2;
					break;
				}
				case 2: {
					uint32_t stage2 = logSize;

					axis->specializationConstants.numStages = stage2;


					axis->specializationConstants.stageRadix[0] = 2;
					axis->specializationConstants.stageRadix[1] = 2;
					break;
				}
				}
			}
		}
		else {
			//configure radix stages
			uint32_t logSize = log2(app->configuration.size[axis_id]);
			uint32_t numPasses[8] = { 0,0,0,0,0,0,0,0 };//256-512-1024-2048(256KB)-4096(256KB)-8k(future?)-16k(future?) - find correct strided FFT configuration
			uint32_t temp = app->configuration.size[axis_id];
			uint32_t startStage = 256;
			uint32_t maxSingleSizeStrided = app->configuration.sharedMemorySize / app->configuration.coalescedMemory;
			uint32_t maxPassId = log2(maxSingleSizeStrided / 256);
			uint32_t minPassId = (maxSingleSizeStrided >= 512) ? 1 : 0;
			//maxPassId += log2(app->configuration.registerBoost); //in development
			for (uint32_t i = 0; i < 8; i++) {
				while (temp > 1)
				{
					temp /= startStage;
					numPasses[i]++;
				}
				temp = app->configuration.size[axis_id];
				startStage *= 2;
			}
			uint32_t passId = minPassId;
			for (uint32_t i = minPassId; i < maxPassId + 1; i++) {
				if (numPasses[i] < numPasses[passId]) {
					passId = i;
				}
			}
			FFTPlan->numSupportAxisUploads[axis_id - 1] = numPasses[passId];
			if (axis_upload_id >= numPasses[passId])
				return;
			switch (app->configuration.radix) {
			case 8: {
				uint32_t stage8 = logSize / 3;
				uint32_t stage4 = 0;
				uint32_t stage2 = 0;
				if (logSize % 3 == 2)
					stage4 = 1;
				if (logSize % 3 == 1)
					stage2 = 1;
				uint32_t totNumStages = stage8 + stage4 + stage2;
				uint32_t locNumStages = 0;
				if (passId == minPassId) {
					locNumStages = stage8 / numPasses[passId];
					if (axis_upload_id < stage8 % numPasses[passId])
						locNumStages++;
					axis->specializationConstants.numStages = locNumStages;
					axis->specializationConstants.fftDim = pow(8, locNumStages);
					axis->specializationConstants.stageRadix[0] = 8;
					axis->specializationConstants.stageRadix[1] = 8;

					if (axis_upload_id == numPasses[passId] - 1) {
						if (stage4 == 1) {
							axis->specializationConstants.numStages++;
							axis->specializationConstants.stageRadix[1] = 4;
							axis->specializationConstants.fftDim *= 4;
						}
						if (stage2 == 1) {
							axis->specializationConstants.numStages++;
							axis->specializationConstants.stageRadix[1] = 2;
							axis->specializationConstants.fftDim *= 2;
						}
					}
					axis->specializationConstants.firstStageStartSize = 1;
					axis->specializationConstants.stageStartSize = (axis_upload_id == 0) ? 1 : FFTPlan->supportAxes[axis_id - 1][axis_upload_id - 1].specializationConstants.stageStartSize * FFTPlan->supportAxes[axis_id - 1][axis_upload_id - 1].specializationConstants.fftDim;
					axis->specializationConstants.fft_dim_x = app->configuration.size[1];
				}
				else {
					if (axis_upload_id < numPasses[passId] - 1) {

						uint32_t locLogSize = 8 + passId;
						if ((axis_upload_id + 1 == numPasses[passId] - 1) && (logSize - (8 + passId) * (numPasses[passId] - 1) < 3))
							locLogSize -= (3 - (logSize - (8 + passId) * (numPasses[passId] - 1)));
						uint32_t locStage8 = locLogSize / 3;
						uint32_t locStage4 = 0;
						uint32_t locStage2 = 0;
						if (locLogSize % 3 == 2)
							locStage4 = 1;
						if (locLogSize % 3 == 1)
							locStage2 = 1;
						axis->specializationConstants.numStages = locStage8 + locStage4 + locStage2;
						axis->specializationConstants.fftDim = pow(2, locLogSize);
						axis->specializationConstants.stageRadix[0] = 8;
						axis->specializationConstants.stageRadix[1] = 8;

						if (locStage4 == 1) {
							axis->specializationConstants.stageRadix[1] = 4;
						}
						if (locStage2 == 1) {
							axis->specializationConstants.stageRadix[1] = 2;
						}
						axis->specializationConstants.firstStageStartSize = 1;
						axis->specializationConstants.stageStartSize = (axis_upload_id == 0) ? 1 : FFTPlan->axes[axis_id][axis_upload_id - 1].specializationConstants.stageStartSize * FFTPlan->axes[axis_id][axis_upload_id - 1].specializationConstants.fftDim;
						if (app->configuration.performR2C)
							axis->specializationConstants.fft_dim_x = app->configuration.size[0] / 2;
						else
							axis->specializationConstants.fft_dim_x = app->configuration.size[0];
					}
					else {
						uint32_t locLogSize = (logSize - (8 + passId) * (numPasses[passId] - 1) < 3) ? 3 : logSize - (8 + passId) * (numPasses[passId] - 1);
						uint32_t locStage8 = locLogSize / 3;
						uint32_t locStage4 = 0;
						uint32_t locStage2 = 0;
						if (locLogSize % 3 == 2)
							locStage4 = 1;
						if (locLogSize % 3 == 1)
							locStage2 = 1;
						axis->specializationConstants.numStages = locStage8 + locStage4 + locStage2;
						axis->specializationConstants.fftDim = pow(2, locLogSize);
						axis->specializationConstants.stageRadix[0] = 8;
						axis->specializationConstants.stageRadix[1] = 8;

						if (locStage4 == 1) {
							axis->specializationConstants.stageRadix[1] = 4;
						}
						if (locStage2 == 1) {
							axis->specializationConstants.stageRadix[1] = 2;
						}
						axis->specializationConstants.firstStageStartSize = 1;
						axis->specializationConstants.stageStartSize = (axis_upload_id == 0) ? 1 : FFTPlan->axes[axis_id][axis_upload_id - 1].specializationConstants.stageStartSize * FFTPlan->axes[axis_id][axis_upload_id - 1].specializationConstants.fftDim;
						if (app->configuration.performR2C)
							axis->specializationConstants.fft_dim_x = app->configuration.size[0] / 2;
						else
							axis->specializationConstants.fft_dim_x = app->configuration.size[0];
					}
				}


				break;
			}
			case 4: {
				uint32_t stage4 = logSize / 2;
				uint32_t stage2 = 0;
				if (logSize % 2 == 1)
					stage2 = 1;
				axis->specializationConstants.numStages = stage4 + stage2;


				axis->specializationConstants.stageRadix[0] = 4;
				axis->specializationConstants.stageRadix[1] = 4;
				if (logSize % 2 == 1)
					axis->specializationConstants.stageRadix[1] = 2;
				break;
			}
			case 2: {
				uint32_t stage2 = logSize;

				axis->specializationConstants.numStages = stage2;


				axis->specializationConstants.stageRadix[0] = 2;
				axis->specializationConstants.stageRadix[1] = 2;
				break;
			}
			}
		}
		axis->specializationConstants.reorderFourStep = 0;
		uint32_t passID = FFTPlan->numSupportAxisUploads[axis_id - 1] - 1 - axis_upload_id;
		axis->specializationConstants.fft_dim_full = app->configuration.size[axis_id];
		uint32_t maxBatchCoalesced;
		if (app->configuration.doublePrecision)
			maxBatchCoalesced = app->configuration.coalescedMemory / (2 * sizeof(double));
		else
			if (app->configuration.halfPrecision)
				maxBatchCoalesced = app->configuration.coalescedMemory / (2 * sizeof(float));
			else
				maxBatchCoalesced = app->configuration.coalescedMemory / (2 * sizeof(float));

		axis->groupedBatch = maxBatchCoalesced;
		if ((app->configuration.size[0] < 4096) && (app->configuration.size[1] < 512) && (app->configuration.size[2] == 1)) {
			if (app->configuration.sharedMemorySize / axis->specializationConstants.fftDim >= app->configuration.coalescedMemory) {
				if (1024 / axis->specializationConstants.fftDim < maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim)
					if (1024 / axis->specializationConstants.fftDim > axis->groupedBatch)
						axis->groupedBatch = 1024 / axis->specializationConstants.fftDim;
					else
						axis->groupedBatch = maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim;
			}
		}
		else {
			axis->groupedBatch = (app->configuration.sharedMemorySize / axis->specializationConstants.fftDim >= app->configuration.coalescedMemory) ? maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim : axis->groupedBatch;
		}
		//allocate LUT 
		if (app->configuration.useLUT) {
			double double_PI = 3.1415926535897932384626433832795;
			if (app->configuration.doublePrecision) {
				if (axis_upload_id > 0)
					axis->bufferLUTSize = (3 * (pow(8, (axis->specializationConstants.numStages)) - 1) / 7 + axis->specializationConstants.stageStartSize * axis->specializationConstants.fftDim) * 2 * sizeof(double);
				else
					axis->bufferLUTSize = (3 * (pow(8, (axis->specializationConstants.numStages)) - 1) / 7) * 2 * sizeof(double);

				double* tempLUT = (double*)malloc(axis->bufferLUTSize);
				for (uint32_t i = 0; i < axis->specializationConstants.numStages; i++) {
					for (uint32_t j = 0; j < pow(8, i); j++) {
						if (inverse) {
							tempLUT[2 * (j + (uint32_t)((pow(8, i) - 1) / 7))] = cos(-j * double_PI / pow(8, i));
							tempLUT[2 * (j + (uint32_t)((pow(8, i) - 1) / 7)) + 1] = sin(-j * double_PI / pow(8, i));
						}
						else {
							tempLUT[2 * (j + (uint32_t)((pow(8, i) - 1) / 7))] = cos(j * double_PI / pow(8, i));
							tempLUT[2 * (j + (uint32_t)((pow(8, i) - 1) / 7)) + 1] = sin(j * double_PI / pow(8, i));
						}
					}
				}
				for (uint32_t i = 0; i < axis->specializationConstants.numStages; i++) {
					for (uint32_t j = 0; j < pow(8, i); j++) {
						if (inverse) {
							tempLUT[(uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (j + (uint32_t)((pow(8, i) - 1) / 7))] = cos(-j * double_PI / pow(8, i) / 2);
							tempLUT[(uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (j + (uint32_t)((pow(8, i) - 1) / 7)) + 1] = sin(-j * double_PI / pow(8, i) / 2);
						}
						else {
							tempLUT[(uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (j + (uint32_t)((pow(8, i) - 1) / 7))] = cos(j * double_PI / pow(8, i) / 2);
							tempLUT[(uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (j + (uint32_t)((pow(8, i) - 1) / 7)) + 1] = sin(j * double_PI / pow(8, i) / 2);
						}
					}
				}
				for (uint32_t i = 0; i < axis->specializationConstants.numStages; i++) {
					for (uint32_t j = 0; j < pow(8, i); j++) {
						if (inverse) {
							tempLUT[2 * (uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (j + (uint32_t)((pow(8, i) - 1) / 7))] = cos(-j * double_PI / pow(8, i) / 4);
							tempLUT[2 * (uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (j + (uint32_t)((pow(8, i) - 1) / 7)) + 1] = sin(-j * double_PI / pow(8, i) / 4);
						}
						else {
							tempLUT[2 * (uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (j + (uint32_t)((pow(8, i) - 1) / 7))] = cos(j * double_PI / pow(8, i) / 4);
							tempLUT[2 * (uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (j + (uint32_t)((pow(8, i) - 1) / 7)) + 1] = sin(j * double_PI / pow(8, i) / 4);
						}
					}
				}
				if (axis_upload_id > 0)
					for (uint32_t i = 0; i < axis->specializationConstants.stageStartSize; i++) {
						for (uint32_t j = 0; j < axis->specializationConstants.fftDim; j++) {
							double angle = 2 * double_PI * ((i * j) / (double)(axis->specializationConstants.stageStartSize * axis->specializationConstants.fftDim));
							if (inverse) {
								tempLUT[3 * (uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize)] = cos(-angle);
								tempLUT[3 * (uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize) + 1] = sin(-angle);
							}
							else {
								tempLUT[3 * (uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize)] = cos(angle);
								tempLUT[3 * (uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize) + 1] = sin(angle);
							}
						}
					}
				allocateFFTBuffer(app, &axis->bufferLUT, &axis->bufferLUTDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, axis->bufferLUTSize);
				transferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
				free(tempLUT);
			}
			else {
				if (axis_upload_id > 0)
					axis->bufferLUTSize = (3 * (pow(8, (axis->specializationConstants.numStages)) - 1) / 7 + axis->specializationConstants.stageStartSize * axis->specializationConstants.fftDim) * 2 * sizeof(float);
				else
					axis->bufferLUTSize = (3 * (pow(8, (axis->specializationConstants.numStages)) - 1) / 7) * 2 * sizeof(float);
				float* tempLUT = (float*)malloc(axis->bufferLUTSize);
				for (uint32_t i = 0; i < axis->specializationConstants.numStages; i++) {
					for (uint32_t j = 0; j < pow(8, i); j++) {
						if (inverse) {
							tempLUT[2 * (j + (uint32_t)((pow(8, i) - 1) / 7))] = (float)cos(-j * double_PI / pow(8, i));
							tempLUT[2 * (j + (uint32_t)((pow(8, i) - 1) / 7)) + 1] = (float)sin(-j * double_PI / pow(8, i));
						}
						else {
							tempLUT[2 * (j + (uint32_t)((pow(8, i) - 1) / 7))] = (float)cos(j * double_PI / pow(8, i));
							tempLUT[2 * (j + (uint32_t)((pow(8, i) - 1) / 7)) + 1] = (float)sin(j * double_PI / pow(8, i));
						}
					}
				}
				for (uint32_t i = 0; i < axis->specializationConstants.numStages; i++) {
					for (uint32_t j = 0; j < pow(8, i); j++) {
						if (inverse) {
							tempLUT[(uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (j + (uint32_t)((pow(8, i) - 1) / 7))] = (float)cos(-j * double_PI / pow(8, i) / 2);
							tempLUT[(uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (j + (uint32_t)((pow(8, i) - 1) / 7)) + 1] = (float)sin(-j * double_PI / pow(8, i) / 2);
						}
						else {
							tempLUT[(uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (j + (uint32_t)((pow(8, i) - 1) / 7))] = (float)cos(j * double_PI / pow(8, i) / 2);
							tempLUT[(uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (j + (uint32_t)((pow(8, i) - 1) / 7)) + 1] = (float)sin(j * double_PI / pow(8, i) / 2);
						}
					}
				}
				for (uint32_t i = 0; i < axis->specializationConstants.numStages; i++) {
					for (uint32_t j = 0; j < pow(8, i); j++) {
						if (inverse) {
							tempLUT[2 * (uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (j + (uint32_t)((pow(8, i) - 1) / 7))] = (float)cos(-j * double_PI / pow(8, i) / 4);
							tempLUT[2 * (uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (j + (uint32_t)((pow(8, i) - 1) / 7)) + 1] = (float)sin(-j * double_PI / pow(8, i) / 4);
						}
						else {
							tempLUT[2 * (uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (j + (uint32_t)((pow(8, i) - 1) / 7))] = (float)cos(j * double_PI / pow(8, i) / 4);
							tempLUT[2 * (uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (j + (uint32_t)((pow(8, i) - 1) / 7)) + 1] = (float)sin(j * double_PI / pow(8, i) / 4);
						}
					}
				}
				if (axis_upload_id > 0)
					for (uint32_t i = 0; i < axis->specializationConstants.stageStartSize; i++) {
						for (uint32_t j = 0; j < axis->specializationConstants.fftDim; j++) {
							double angle = 2 * double_PI * ((i * j) / (double)(axis->specializationConstants.stageStartSize * axis->specializationConstants.fftDim));
							if (inverse) {
								tempLUT[3 * (uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize)] = (float)cos(-angle);
								tempLUT[3 * (uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize) + 1] = (float)sin(-angle);
							}
							else {
								tempLUT[3 * (uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize)] = (float)cos(angle);
								tempLUT[3 * (uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize) + 1] = (float)sin(angle);
							}
						}
					}
				allocateFFTBuffer(app, &axis->bufferLUT, &axis->bufferLUTDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, axis->bufferLUTSize);
				transferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
				free(tempLUT);
			}
		}
		//axis->groupedBatch = ((axis_upload_id>0)&&(axis->groupedBatch > axis->specializationConstants.stageStartSize)) ? axis->specializationConstants.stageStartSize : axis->groupedBatch;
		//configure strides
		//perform r2c
		axis->specializationConstants.inputStride[0] = 1;
		axis->specializationConstants.inputStride[3] = (app->configuration.size[0] / 2 + 1) * app->configuration.size[1] * app->configuration.size[2];

		if (axis_id == 1)
		{

			//don't transpose 0-1
			axis->specializationConstants.inputStride[1] = app->configuration.size[1];
			axis->specializationConstants.inputStride[2] = (app->configuration.size[0] / 2 + 1) * app->configuration.size[1];
			axis->specializationConstants.inputStride[3] = (app->configuration.size[0] / 2 + 1) * app->configuration.size[1] * app->configuration.size[2];
		}
		if (axis_id == 2)
		{

			//don't transpose 0-1, don't transpose 1-2
			axis->specializationConstants.inputStride[1] = (app->configuration.size[0] / 2 + 1) * app->configuration.size[1];
			axis->specializationConstants.inputStride[2] = app->configuration.size[1];

		}

		axis->specializationConstants.outputStride[0] = axis->specializationConstants.inputStride[0];
		axis->specializationConstants.outputStride[1] = axis->specializationConstants.inputStride[1];
		axis->specializationConstants.outputStride[2] = axis->specializationConstants.inputStride[2];
		axis->specializationConstants.outputStride[3] = axis->specializationConstants.inputStride[3];

		axis->specializationConstants.inputStride[3] = (app->configuration.coordinateFeatures == 1) ? 0 : axis->specializationConstants.inputStride[3];
		axis->specializationConstants.outputStride[3] = (app->configuration.coordinateFeatures == 1) ? 0 : axis->specializationConstants.outputStride[3];

		axis->specializationConstants.inputStride[4] = ((app->configuration.numberBatches == 1) && (app->configuration.numberKernels == 1)) ? 0 : axis->specializationConstants.inputStride[3] * app->configuration.coordinateFeatures;
		axis->specializationConstants.outputStride[4] = ((app->configuration.numberBatches == 1) && (app->configuration.numberKernels == 1)) ? 0 : axis->specializationConstants.outputStride[3] * app->configuration.coordinateFeatures;


		axis->specializationConstants.inverse = inverse;
		if (inverse)
			axis->specializationConstants.zeropad[0] = (axis_upload_id == 0) ? app->configuration.performZeropadding[axis_id] : 0;
		else
			axis->specializationConstants.zeropad[0] = (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) ? app->configuration.performZeropadding[axis_id] : 0;
		axis->specializationConstants.zeropad[1] = 0;
		axis->specializationConstants.inputOffset = app->configuration.size[0] * app->configuration.size[1] / 2;
		axis->specializationConstants.outputOffset = app->configuration.size[0] * app->configuration.size[1] / 2;

		VkDescriptorPoolSize descriptorPoolSize = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
		descriptorPoolSize.descriptorCount = 2;
		if ((axis_id == 1) && (axis_upload_id == 0) && (app->configuration.FFTdim == 2) && (app->configuration.performConvolution))
			descriptorPoolSize.descriptorCount = 3;
		if ((axis_id == 2) && (axis_upload_id == 0) && (app->configuration.FFTdim == 3) && (app->configuration.performConvolution))
			descriptorPoolSize.descriptorCount = 3;
		if (app->configuration.useLUT)
			descriptorPoolSize.descriptorCount++;
		VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
		descriptorPoolCreateInfo.poolSizeCount = 1;
		descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize;
		descriptorPoolCreateInfo.maxSets = 1;
		vkCreateDescriptorPool(app->configuration.device[0], &descriptorPoolCreateInfo, NULL, &axis->descriptorPool);

		const VkDescriptorType descriptorType[4] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
		VkDescriptorSetLayoutBinding* descriptorSetLayoutBindings;
		descriptorSetLayoutBindings = (VkDescriptorSetLayoutBinding*)malloc(descriptorPoolSize.descriptorCount * sizeof(VkDescriptorSetLayoutBinding));
		for (uint32_t i = 0; i < descriptorPoolSize.descriptorCount; ++i) {
			descriptorSetLayoutBindings[i].binding = i;
			descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
			descriptorSetLayoutBindings[i].descriptorCount = 1;
			descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		}

		VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
		descriptorSetLayoutCreateInfo.bindingCount = descriptorPoolSize.descriptorCount;
		descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;

		vkCreateDescriptorSetLayout(app->configuration.device[0], &descriptorSetLayoutCreateInfo, NULL, &axis->descriptorSetLayout);
		free(descriptorSetLayoutBindings);
		VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
		descriptorSetAllocateInfo.descriptorPool = axis->descriptorPool;
		descriptorSetAllocateInfo.descriptorSetCount = 1;
		descriptorSetAllocateInfo.pSetLayouts = &axis->descriptorSetLayout;
		vkAllocateDescriptorSets(app->configuration.device[0], &descriptorSetAllocateInfo, &axis->descriptorSet);
		for (uint32_t i = 0; i < descriptorPoolSize.descriptorCount; ++i) {
			VkDescriptorBufferInfo descriptorBufferInfo = { 0 };

			if (i == 0) {
				descriptorBufferInfo.buffer = app->configuration.buffer[0];
				descriptorBufferInfo.range = app->configuration.bufferSize[0];
				/*if (app->configuration.isInputFormatted && (
						((axis_id == 0) && (!inverse))
						|| ((axis_id == app->configuration.FFTdim-1) && (inverse)))
					) {
					descriptorBufferInfo.buffer = app->configuration.inputBuffer[0];
					descriptorBufferInfo.range = app->configuration.inputBufferSize[0];
				}
				else {
					if ((app->configuration.numberKernels > 1) && (inverse)) {
						descriptorBufferInfo.buffer = app->configuration.outputBuffer[0];
						descriptorBufferInfo.range = app->configuration.outputBufferSize[0];
					}
					else {
						descriptorBufferInfo.buffer = app->configuration.buffer[0];
						descriptorBufferInfo.range = app->configuration.bufferSize[0];
					}
				}*/
				descriptorBufferInfo.offset = 0;
			}
			if (i == 1) {
				descriptorBufferInfo.buffer = app->configuration.buffer[0];
				descriptorBufferInfo.range = app->configuration.bufferSize[0];
				/*if ((app->configuration.isOutputFormatted && (
						((axis_id == 0) && (inverse))
						|| ((axis_id == app->configuration.FFTdim-1) && (!inverse) && (!app->configuration.performConvolution))
						|| ((axis_id == 0) && (app->configuration.performConvolution) && (app->configuration.FFTdim == 1)))
					)||
					((app->configuration.numberKernels>1)&&(
						(inverse)
						||(axis_id== app->configuration.FFTdim-1)))
					) {
					descriptorBufferInfo.buffer = app->configuration.outputBuffer[0];
					descriptorBufferInfo.range = app->configuration.outputBufferSize[0];
				}
				else {
					descriptorBufferInfo.buffer = app->configuration.buffer[0];
					descriptorBufferInfo.range = app->configuration.bufferSize[0];
				}*/
				descriptorBufferInfo.offset = 0;
			}
			if ((i == 2) && (app->configuration.performConvolution)) {
				descriptorBufferInfo.buffer = app->configuration.kernel[0];
				descriptorBufferInfo.offset = 0;
				descriptorBufferInfo.range = app->configuration.kernelSize[0];
			}
			if ((i == descriptorPoolSize.descriptorCount - 1) && (app->configuration.useLUT)) {
				descriptorBufferInfo.buffer = axis->bufferLUT;
				descriptorBufferInfo.offset = 0;
				descriptorBufferInfo.range = axis->bufferLUTSize;
			}
			VkWriteDescriptorSet writeDescriptorSet = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
			writeDescriptorSet.dstSet = axis->descriptorSet;
			writeDescriptorSet.dstBinding = i;
			writeDescriptorSet.dstArrayElement = 0;
			writeDescriptorSet.descriptorType = descriptorType[i];
			writeDescriptorSet.descriptorCount = 1;
			writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
			vkUpdateDescriptorSets(app->configuration.device[0], 1, &writeDescriptorSet, 0, NULL);

		}

		{
			VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
			pipelineLayoutCreateInfo.setLayoutCount = 1;
			pipelineLayoutCreateInfo.pSetLayouts = &axis->descriptorSetLayout;

			VkPushConstantRange pushConstantRange = { VK_SHADER_STAGE_COMPUTE_BIT };
			pushConstantRange.offset = 0;
			pushConstantRange.size = sizeof(VkFFTPushConstantsLayout);
			// Push constant ranges are part of the pipeline layout
			pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
			pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;


			vkCreatePipelineLayout(app->configuration.device[0], &pipelineLayoutCreateInfo, NULL, &axis->pipelineLayout);
			if (axis_id == 1) {
				if (axis_upload_id == 0) {
					axis->axisBlock[0] = (axis->specializationConstants.fftDim / 8 > 1) ? axis->specializationConstants.fftDim / 8 : 1;
					if (axis->axisBlock[0] > 512) axis->axisBlock[0] = 512;
					axis->axisBlock[1] = 1;
					axis->axisBlock[2] = 1;
					axis->axisBlock[3] = axis->specializationConstants.fftDim;
				}
				else {
					axis->axisBlock[1] = (axis->specializationConstants.fftDim / 8 > 1) ? axis->specializationConstants.fftDim / 8 : 1;

					axis->axisBlock[0] = (axis->specializationConstants.stageStartSize > axis->groupedBatch) ? axis->groupedBatch : axis->specializationConstants.stageStartSize;

					axis->axisBlock[2] = 1;
					axis->axisBlock[3] = axis->specializationConstants.fftDim;
				}
			}
			if (axis_id == 2) {
				axis->axisBlock[1] = (axis->specializationConstants.fftDim / 8 > 1) ? axis->specializationConstants.fftDim / 8 : 1;

				axis->axisBlock[0] = (app->configuration.size[1] > axis->groupedBatch) ? axis->groupedBatch : app->configuration.size[1];
				/*if (axis->axisBlock[0] * axis->axisBlock[1] < 64)
					if (app->configuration.size[1] > 64 / axis->axisBlock[1])
						axis->axisBlock[0] = 64 / axis->axisBlock[1];
					else
						axis->axisBlock[0] = app->configuration.size[0];*/
				axis->axisBlock[2] = 1;
				axis->axisBlock[3] = axis->specializationConstants.fftDim;
			}
			uint32_t tempSize[3] = { app->configuration.size[0], app->configuration.size[1], app->configuration.size[2] };
			if (axis_id == 1) {
				if (axis_upload_id == 0)
					tempSize[0] = app->configuration.size[1] / axis->specializationConstants.fftDim;
				else
					tempSize[0] = app->configuration.size[1] / axis->specializationConstants.fftDim / axis->axisBlock[0];

				tempSize[1] = 1;
				tempSize[2] = app->configuration.size[2];
				if (app->configuration.performZeropadding[2]) tempSize[2] = ceil(tempSize[2] / 2.0);

				if (tempSize[0] > app->configuration.maxComputeWorkGroupCount[0]) axis->specializationConstants.performWorkGroupShift[0] = 1;
				else  axis->specializationConstants.performWorkGroupShift[0] = 0;
				if (tempSize[1] > app->configuration.maxComputeWorkGroupCount[1]) axis->specializationConstants.performWorkGroupShift[1] = 1;
				else  axis->specializationConstants.performWorkGroupShift[1] = 0;
				if (tempSize[2] > app->configuration.maxComputeWorkGroupCount[2]) axis->specializationConstants.performWorkGroupShift[2] = 1;
				else  axis->specializationConstants.performWorkGroupShift[2] = 0;

			}
			if (axis_id == 2) {
				tempSize[0] = app->configuration.size[1] / axis->axisBlock[0] * app->configuration.size[2] / axis->specializationConstants.fftDim;
				tempSize[1] = 1;
				tempSize[2] = 1;

				if (tempSize[0] > app->configuration.maxComputeWorkGroupCount[0]) axis->specializationConstants.performWorkGroupShift[0] = 1;
				else  axis->specializationConstants.performWorkGroupShift[0] = 0;
				if (tempSize[1] > app->configuration.maxComputeWorkGroupCount[1]) axis->specializationConstants.performWorkGroupShift[1] = 1;
				else  axis->specializationConstants.performWorkGroupShift[1] = 0;
				if (tempSize[2] > app->configuration.maxComputeWorkGroupCount[2]) axis->specializationConstants.performWorkGroupShift[2] = 1;
				else  axis->specializationConstants.performWorkGroupShift[2] = 0;

			}
			VkSpecializationMapEntry specializationMapEntries[30] = { {} };
			for (uint32_t i = 0; i < 30; i++) {
				specializationMapEntries[i].constantID = i + 1;
				specializationMapEntries[i].size = sizeof(uint32_t);
				specializationMapEntries[i].offset = i * sizeof(uint32_t);
			}
			VkSpecializationInfo specializationInfo = { 0 };
			specializationInfo.dataSize = 30 * sizeof(uint32_t);
			specializationInfo.mapEntryCount = 30;
			specializationInfo.pMapEntries = specializationMapEntries;
			axis->specializationConstants.localSize[0] = axis->axisBlock[0];
			axis->specializationConstants.localSize[1] = axis->axisBlock[1];
			axis->specializationConstants.localSize[2] = axis->axisBlock[2];
			axis->specializationConstants.fftDim = axis->axisBlock[3];
			specializationInfo.pData = &axis->specializationConstants;
			VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
			VkComputePipelineCreateInfo computePipelineCreateInfo = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };


			pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;


			if (axis_id == 1) {

				if ((app->configuration.FFTdim == 2) && (app->configuration.performConvolution) && (axis_upload_id == 0)) {
					if (axis_upload_id == 0) {
						switch (app->configuration.matrixConvolution) {
						case 1:
							VkFFTInitShader(app, 9, &pipelineShaderStageCreateInfo.module);
							break;
						case 2:
							if (app->configuration.symmetricKernel)
								VkFFTInitShader(app, 12, &pipelineShaderStageCreateInfo.module);
							else
								VkFFTInitShader(app, 15, &pipelineShaderStageCreateInfo.module);
							break;
						case 3:
							if (app->configuration.symmetricKernel)
								VkFFTInitShader(app, 18, &pipelineShaderStageCreateInfo.module);
							else
								VkFFTInitShader(app, 21, &pipelineShaderStageCreateInfo.module);
							break;
						}
					}
					else {
						switch (app->configuration.matrixConvolution) {
						case 1:
							VkFFTInitShader(app, 10, &pipelineShaderStageCreateInfo.module);
							break;
						case 2:
							if (app->configuration.symmetricKernel)
								VkFFTInitShader(app, 13, &pipelineShaderStageCreateInfo.module);
							else
								VkFFTInitShader(app, 16, &pipelineShaderStageCreateInfo.module);
							break;
						case 3:
							if (app->configuration.symmetricKernel)
								VkFFTInitShader(app, 19, &pipelineShaderStageCreateInfo.module);
							else
								VkFFTInitShader(app, 22, &pipelineShaderStageCreateInfo.module);
							break;
						}
					}

				}
				else {
					/*if (axis_upload_id == 0)
						VkFFTInitShader(app, 0, &pipelineShaderStageCreateInfo.module);
					else
						VkFFTInitShader(app, 2, &pipelineShaderStageCreateInfo.module);*/
					switch (app->configuration.registerBoost) {
					case 1:
					{
						if (axis_upload_id == 0)
							VkFFTInitShader(app, 0, &pipelineShaderStageCreateInfo.module);
						else
							VkFFTInitShader(app, 2, &pipelineShaderStageCreateInfo.module);
						break;
					}
					case 2:
					{
						switch (axis->specializationConstants.fftDim) {
						case 8192:
							VkFFTInitShader(app, 25, &pipelineShaderStageCreateInfo.module);
							break;
						default:
							if (axis_upload_id == 0)
								VkFFTInitShader(app, 0, &pipelineShaderStageCreateInfo.module);
							else
								VkFFTInitShader(app, 2, &pipelineShaderStageCreateInfo.module);
							break;
						}
						break;
					}
					case 4:
					{
						switch (axis->specializationConstants.fftDim) {
						case 8192:
							VkFFTInitShader(app, 25, &pipelineShaderStageCreateInfo.module);
							break;
						case 16384:
							VkFFTInitShader(app, 35, &pipelineShaderStageCreateInfo.module);
							break;
						default:
							if (axis_upload_id == 0)
								VkFFTInitShader(app, 0, &pipelineShaderStageCreateInfo.module);
							else
								VkFFTInitShader(app, 2, &pipelineShaderStageCreateInfo.module);
							break;
						}
						break;
					}
					}
				}

			}

			if (axis_id == 2) {
				if ((app->configuration.FFTdim == 3) && (app->configuration.performConvolution) && (axis_upload_id == 0)) {
					switch (app->configuration.matrixConvolution) {
					case 1:
						VkFFTInitShader(app, 8, &pipelineShaderStageCreateInfo.module);
						break;
					case 2:
						if (app->configuration.symmetricKernel)
							VkFFTInitShader(app, 11, &pipelineShaderStageCreateInfo.module);
						else
							VkFFTInitShader(app, 14, &pipelineShaderStageCreateInfo.module);
						break;
					case 3:
						if (app->configuration.symmetricKernel)
							VkFFTInitShader(app, 17, &pipelineShaderStageCreateInfo.module);
						else
							VkFFTInitShader(app, 20, &pipelineShaderStageCreateInfo.module);
						break;
					}

				}
				else {
					VkFFTInitShader(app, 7, &pipelineShaderStageCreateInfo.module);
				}
			}

			pipelineShaderStageCreateInfo.pName = "main";
			pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
			computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
			computePipelineCreateInfo.layout = axis->pipelineLayout;



			vkCreateComputePipelines(app->configuration.device[0], VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, &axis->pipeline);
			vkDestroyShaderModule(app->configuration.device[0], pipelineShaderStageCreateInfo.module, NULL);
		}


	}
	void VkFFTPlanAxis(VkFFTApplication* app, VkFFTPlan* FFTPlan, uint32_t axis_id, uint32_t axis_upload_id, VkBool32 inverse) {
		//get radix stages
		VkFFTAxis* axis = &FFTPlan->axes[axis_id][axis_upload_id];
		uint32_t complexSize;
		if (app->configuration.doublePrecision)
			complexSize = (2 * sizeof(double));
		else
			if (app->configuration.halfPrecision)
				complexSize = (2 * sizeof(float));
			else
				complexSize = (2 * sizeof(float));
		uint32_t maxSequenceLengthSharedMemory = app->configuration.sharedMemorySize / complexSize;
		uint32_t maxSingleSizeStrided = (app->configuration.coalescedMemory > complexSize) ? app->configuration.sharedMemorySize / (app->configuration.coalescedMemory) : app->configuration.sharedMemorySize / complexSize;

		uint32_t logSize = log2(app->localFFTPlan.axisSplit[axis_id][axis_upload_id]);

		uint32_t stage8 = logSize / 3;
		uint32_t stage4 = 0;
		uint32_t stage2 = 0;
		if (logSize % 3 == 2)
			stage4 = 1;
		if (logSize % 3 == 1)
			stage2 = 1;
		uint32_t totNumStages = stage8 + stage4 + stage2;

		axis->specializationConstants.numStages = stage8;
		axis->specializationConstants.fftDim = pow(8, stage8);
		axis->specializationConstants.stageRadix[0] = 8;
		axis->specializationConstants.stageRadix[1] = 8;

		if (stage4 == 1) {
			axis->specializationConstants.numStages++;
			axis->specializationConstants.stageRadix[1] = 4;
			axis->specializationConstants.fftDim *= 4;
		}
		if (stage2 == 1) {
			axis->specializationConstants.numStages++;
			axis->specializationConstants.stageRadix[1] = 2;
			axis->specializationConstants.fftDim *= 2;
		}

		axis->specializationConstants.stageStartSize = 1;
		for (uint32_t i = 0; i < axis_upload_id; i++)
			axis->specializationConstants.stageStartSize *= app->localFFTPlan.axisSplit[axis_id][i];
		axis->specializationConstants.firstStageStartSize = app->configuration.size[axis_id] / app->localFFTPlan.axisSplit[axis_id][app->localFFTPlan.numAxisUploads[axis_id] - 1];

		if (axis_id == 0) {
			//configure radix stages
			axis->specializationConstants.fft_dim_x = axis->specializationConstants.stageStartSize;
		}
		else {
			if (app->configuration.performR2C)
				axis->specializationConstants.fft_dim_x = app->configuration.size[0] / 2;
			else
				axis->specializationConstants.fft_dim_x = app->configuration.size[0];
		}

		axis->specializationConstants.reorderFourStep = (FFTPlan->numAxisUploads[axis_id] > 1) ? app->configuration.reorderFourStep : 0;
		//axis->groupedBatch = (4096 / axis->specializationConstants.fftDim >= app->configuration.coalescedMemory / 8) ? 4096 / axis->specializationConstants.fftDim : app->configuration.coalescedMemory / 8;
		uint32_t passID = FFTPlan->numAxisUploads[axis_id] - 1 - axis_upload_id;
		axis->specializationConstants.fft_dim_full = app->configuration.size[axis_id];
		uint32_t maxBatchCoalesced = app->configuration.coalescedMemory / complexSize;
		axis->groupedBatch = maxBatchCoalesced;
		if ((app->configuration.size[0] < 4096) && (app->configuration.size[1] < 512) && (app->configuration.size[2] == 1)) {
			if (app->configuration.sharedMemorySize / axis->specializationConstants.fftDim >= app->configuration.coalescedMemory) {
				if (1024 / axis->specializationConstants.fftDim < maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim)
					if (1024 / axis->specializationConstants.fftDim > axis->groupedBatch)
						axis->groupedBatch = 1024 / axis->specializationConstants.fftDim;
					else
						axis->groupedBatch = maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim;
			}
		}
		else {
			axis->groupedBatch = (app->configuration.sharedMemorySize / axis->specializationConstants.fftDim >= app->configuration.coalescedMemory) ? maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim : axis->groupedBatch;
		}
		//shared memory bank conflict resolve
		if ((FFTPlan->numAxisUploads[axis_id] == 3) && (axis_upload_id == 0) && (axis->specializationConstants.fftDim < maxSequenceLengthSharedMemory/16)) {
			axis->groupedBatch = ceil(axis->groupedBatch / 2.0);
		}
		//half bandiwdth technique
		if (!((axis_id == 0) && (FFTPlan->numAxisUploads[axis_id] == 1)) && !((axis_id == 0) && (axis_upload_id == 0) && (!app->configuration.reorderFourStep)) && (axis->specializationConstants.fftDim > maxSingleSizeStrided)) {
			axis->groupedBatch = ceil(axis->groupedBatch / 2.0);
		}
		//allocate LUT 
		if (app->configuration.useLUT) {
			double double_PI = 3.1415926535897932384626433832795;
			if (app->configuration.doublePrecision) {
				if (axis_upload_id > 0)
					axis->bufferLUTSize = (3 * (pow(8, (axis->specializationConstants.numStages)) - 1) / 7 + axis->specializationConstants.stageStartSize * axis->specializationConstants.fftDim) * 2 * sizeof(double);
				else
					axis->bufferLUTSize = (3 * (pow(8, (axis->specializationConstants.numStages)) - 1) / 7) * 2 * sizeof(double);
				double* tempLUT = (double*)malloc(axis->bufferLUTSize);
				for (uint32_t i = 0; i < axis->specializationConstants.numStages; i++) {
					for (uint32_t j = 0; j < pow(8, i); j++) {
						if (inverse) {
							tempLUT[2 * (j + (uint32_t)((pow(8, i) - 1) / 7))] = cos(-j * double_PI / pow(8, i));
							tempLUT[2 * (j + (uint32_t)((pow(8, i) - 1) / 7)) + 1] = sin(-j * double_PI / pow(8, i));
						}
						else {
							tempLUT[2 * (j + (uint32_t)((pow(8, i) - 1) / 7))] = cos(j * double_PI / pow(8, i));
							tempLUT[2 * (j + (uint32_t)((pow(8, i) - 1) / 7)) + 1] = sin(j * double_PI / pow(8, i));
						}
					}
				}
				for (uint32_t i = 0; i < axis->specializationConstants.numStages; i++) {
					for (uint32_t j = 0; j < pow(8, i); j++) {
						if (inverse) {
							tempLUT[(uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (j + (uint32_t)((pow(8, i) - 1) / 7))] = cos(-j * double_PI / pow(8, i) / 2);
							tempLUT[(uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (j + (uint32_t)((pow(8, i) - 1) / 7)) + 1] = sin(-j * double_PI / pow(8, i) / 2);
						}
						else {
							tempLUT[(uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (j + (uint32_t)((pow(8, i) - 1) / 7))] = cos(j * double_PI / pow(8, i) / 2);
							tempLUT[(uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (j + (uint32_t)((pow(8, i) - 1) / 7)) + 1] = sin(j * double_PI / pow(8, i) / 2);
						}
					}
				}
				for (uint32_t i = 0; i < axis->specializationConstants.numStages; i++) {
					for (uint32_t j = 0; j < pow(8, i); j++) {
						if (inverse) {
							tempLUT[2 * (uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (j + (uint32_t)((pow(8, i) - 1) / 7))] = cos(-j * double_PI / pow(8, i) / 4);
							tempLUT[2 * (uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (j + (uint32_t)((pow(8, i) - 1) / 7)) + 1] = sin(-j * double_PI / pow(8, i) / 4);
						}
						else {
							tempLUT[2 * (uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (j + (uint32_t)((pow(8, i) - 1) / 7))] = cos(j * double_PI / pow(8, i) / 4);
							tempLUT[2 * (uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (j + (uint32_t)((pow(8, i) - 1) / 7)) + 1] = sin(j * double_PI / pow(8, i) / 4);
						}
					}
				}
				if (axis_upload_id > 0)
					for (uint32_t i = 0; i < axis->specializationConstants.stageStartSize; i++) {
						for (uint32_t j = 0; j < axis->specializationConstants.fftDim; j++) {
							double angle = 2 * double_PI * ((i * j) / (double)(axis->specializationConstants.stageStartSize * axis->specializationConstants.fftDim));
							if (inverse) {
								tempLUT[3 * (uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize)] = cos(-angle);
								tempLUT[3 * (uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize) + 1] = sin(-angle);
							}
							else {
								tempLUT[3 * (uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize)] = cos(angle);
								tempLUT[3 * (uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize) + 1] = sin(angle);
							}
						}
					}
				allocateFFTBuffer(app, &axis->bufferLUT, &axis->bufferLUTDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, axis->bufferLUTSize);
				transferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
				free(tempLUT);
			}
			else {
				if (axis_upload_id > 0)
					axis->bufferLUTSize = (3 * (pow(8, (axis->specializationConstants.numStages)) - 1) / 7 + axis->specializationConstants.stageStartSize * axis->specializationConstants.fftDim) * 2 * sizeof(float);
				else
					axis->bufferLUTSize = (3 * (pow(8, (axis->specializationConstants.numStages)) - 1) / 7) * 2 * sizeof(float);
				float* tempLUT = (float*)malloc(axis->bufferLUTSize);
				for (uint32_t i = 0; i < axis->specializationConstants.numStages; i++) {
					for (uint32_t j = 0; j < pow(8, i); j++) {
						if (inverse) {
							tempLUT[2 * (j + (uint32_t)((pow(8, i) - 1) / 7))] = (float)cos(-j * double_PI / pow(8, i));
							tempLUT[2 * (j + (uint32_t)((pow(8, i) - 1) / 7)) + 1] = (float)sin(-j * double_PI / pow(8, i));
						}
						else {
							tempLUT[2 * (j + (uint32_t)((pow(8, i) - 1) / 7))] = (float)cos(j * double_PI / pow(8, i));
							tempLUT[2 * (j + (uint32_t)((pow(8, i) - 1) / 7)) + 1] = (float)sin(j * double_PI / pow(8, i));
						}
					}
				}
				for (uint32_t i = 0; i < axis->specializationConstants.numStages; i++) {
					for (uint32_t j = 0; j < pow(8, i); j++) {
						if (inverse) {
							tempLUT[(uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (j + (uint32_t)((pow(8, i) - 1) / 7))] = (float)cos(-j * double_PI / pow(8, i) / 2);
							tempLUT[(uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (j + (uint32_t)((pow(8, i) - 1) / 7)) + 1] = (float)sin(-j * double_PI / pow(8, i) / 2);
						}
						else {
							tempLUT[(uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (j + (uint32_t)((pow(8, i) - 1) / 7))] = (float)cos(j * double_PI / pow(8, i) / 2);
							tempLUT[(uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (j + (uint32_t)((pow(8, i) - 1) / 7)) + 1] = (float)sin(j * double_PI / pow(8, i) / 2);
						}
					}
				}
				for (uint32_t i = 0; i < axis->specializationConstants.numStages; i++) {
					for (uint32_t j = 0; j < pow(8, i); j++) {
						if (inverse) {
							tempLUT[2 * (uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (j + (uint32_t)((pow(8, i) - 1) / 7))] = (float)cos(-j * double_PI / pow(8, i) / 4);
							tempLUT[2 * (uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (j + (uint32_t)((pow(8, i) - 1) / 7)) + 1] = (float)sin(-j * double_PI / pow(8, i) / 4);
						}
						else {
							tempLUT[2 * (uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (j + (uint32_t)((pow(8, i) - 1) / 7))] = (float)cos(j * double_PI / pow(8, i) / 4);
							tempLUT[2 * (uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (j + (uint32_t)((pow(8, i) - 1) / 7)) + 1] = (float)sin(j * double_PI / pow(8, i) / 4);
						}
					}
				}
				if (axis_upload_id > 0)
					for (uint32_t i = 0; i < axis->specializationConstants.stageStartSize; i++) {
						for (uint32_t j = 0; j < axis->specializationConstants.fftDim; j++) {
							double angle = 2 * double_PI * ((i * j) / (double)(axis->specializationConstants.stageStartSize * axis->specializationConstants.fftDim));
							if (inverse) {
								tempLUT[3 * (uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize)] = (float)cos(-angle);
								tempLUT[3 * (uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize) + 1] = (float)sin(-angle);
							}
							else {
								tempLUT[3 * (uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize)] = (float)cos(angle);
								tempLUT[3 * (uint32_t)(pow(8, (axis->specializationConstants.numStages)) - 1) / 7 * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize) + 1] = (float)sin(angle);
							}
						}
					}
				allocateFFTBuffer(app, &axis->bufferLUT, &axis->bufferLUTDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, axis->bufferLUTSize);
				transferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
				free(tempLUT);
			}
		}
		//axis->groupedBatch = ((axis_upload_id > 0) && (axis->groupedBatch > axis->specializationConstants.stageStartSize)) ? axis->specializationConstants.stageStartSize : axis->groupedBatch;
		/*if (4096 / app->configuration.size[1] > app->configuration.coalescedMemory / 16) {
			app->configuration.performTranspose[0] = 0;
			FFTPlan->groupedBatch = 4096 / app->configuration.size[1];
		}
		else {
			app->configuration.performTranspose[0] = 1;
		}

		if (4096 / app->configuration.size[2] > app->configuration.coalescedMemory / 16) {
			app->configuration.performTranspose[1] = 0;
			FFTPlan->axes[2].groupedBatch = 4096 / app->configuration.size[2];
		}
		else {
			app->configuration.performTranspose[1] = 1;
		}*/
		//configure strides
		if (app->configuration.performR2C)
		{
			//perform r2c
			axis->specializationConstants.inputStride[0] = 1;
			axis->specializationConstants.inputStride[3] = (app->configuration.size[0] / 2 + 1) * app->configuration.size[1] * app->configuration.size[2];
			if (axis_id == 0) {
				axis->specializationConstants.inputStride[1] = app->configuration.size[0];
				axis->specializationConstants.inputStride[2] = (app->configuration.size[0] / 2 + 1) * app->configuration.size[1];
			}
			if (axis_id == 1)
			{
				if (app->configuration.performTranspose[0]) {
					//transpose 0-1
					axis->specializationConstants.inputStride[1] = app->configuration.size[1];
					axis->specializationConstants.inputStride[2] = (app->configuration.size[0] / 2 + 1) * app->configuration.size[1];
				}
				else {
					//don't transpose
					axis->specializationConstants.inputStride[1] = app->configuration.size[0] / 2;
					axis->specializationConstants.inputStride[2] = (app->configuration.size[0] / 2 + 1) * app->configuration.size[1];
				}
			}
			if (axis_id == 2)
			{

				if (app->configuration.performTranspose[1]) {
					//transpose 0-1, transpose 1-2
					axis->specializationConstants.inputStride[1] = (app->configuration.size[0] / 2 + 1) * app->configuration.size[2];
					axis->specializationConstants.inputStride[2] = app->configuration.size[2];
				}
				else {

					if (app->configuration.performTranspose[0]) {
						//transpose 0-1, don't transpose 1-2
						axis->specializationConstants.inputStride[1] = (app->configuration.size[0] / 2 + 1) * app->configuration.size[1];
						axis->specializationConstants.inputStride[2] = app->configuration.size[1];
					}
					else {
						//don't transpose
						axis->specializationConstants.inputStride[1] = (app->configuration.size[0] / 2 + 1) * app->configuration.size[1];
						axis->specializationConstants.inputStride[2] = app->configuration.size[0] / 2;
					}
				}
			}

			axis->specializationConstants.outputStride[0] = axis->specializationConstants.inputStride[0];
			axis->specializationConstants.outputStride[1] = axis->specializationConstants.inputStride[1];
			axis->specializationConstants.outputStride[2] = axis->specializationConstants.inputStride[2];
			axis->specializationConstants.outputStride[3] = axis->specializationConstants.inputStride[3];
			if (axis_id == 0) {
				if ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (app->configuration.isInputFormatted) && (!inverse)) {
					if (app->configuration.performZeropadding[0])
						axis->specializationConstants.inputStride[1] = app->configuration.size[0] / 2;

					if (app->configuration.performZeropadding[1])
						axis->specializationConstants.inputStride[2] = axis->specializationConstants.inputStride[1] * app->configuration.size[1] / 4;
					else
						axis->specializationConstants.inputStride[2] = axis->specializationConstants.inputStride[1] * app->configuration.size[1] / 2;

					if (app->configuration.performZeropadding[2])
						axis->specializationConstants.inputStride[3] = axis->specializationConstants.inputStride[2] * app->configuration.size[2] / 2;
					else
						axis->specializationConstants.inputStride[3] = axis->specializationConstants.inputStride[2] * app->configuration.size[2];
				}
				if ((axis_upload_id == 0) && (app->configuration.isOutputFormatted) && ((inverse) || ((app->configuration.performConvolution) && (app->configuration.FFTdim == 1)))) {
					if (app->configuration.performZeropadding[0])
						axis->specializationConstants.outputStride[1] = app->configuration.size[0] / 2;

					if (app->configuration.performZeropadding[1])
						axis->specializationConstants.outputStride[2] = axis->specializationConstants.outputStride[1] * app->configuration.size[1] / 4;
					else
						axis->specializationConstants.outputStride[2] = axis->specializationConstants.outputStride[1] * app->configuration.size[1] / 2;

					if (app->configuration.performZeropadding[2])
						axis->specializationConstants.outputStride[3] = axis->specializationConstants.outputStride[2] * app->configuration.size[2] / 2;
					else
						axis->specializationConstants.outputStride[3] = axis->specializationConstants.outputStride[2] * app->configuration.size[2];
				}
			}
		}
		else {
			//don't perform r2c
			axis->specializationConstants.inputStride[0] = 1;
			axis->specializationConstants.inputStride[3] = app->configuration.size[0] * app->configuration.size[1] * app->configuration.size[2];
			if (axis_id == 0) {
				axis->specializationConstants.inputStride[1] = app->configuration.size[0];
				axis->specializationConstants.inputStride[2] = app->configuration.size[0] * app->configuration.size[1];
			}
			if (axis_id == 1)
			{
				if (app->configuration.performTranspose[0]) {
					//transpose 0-1, no transpose 1-2
					axis->specializationConstants.inputStride[1] = app->configuration.size[1];
					axis->specializationConstants.inputStride[2] = app->configuration.size[0] * app->configuration.size[1];
				}
				else {
					//no transpose
					axis->specializationConstants.inputStride[1] = app->configuration.size[0];
					axis->specializationConstants.inputStride[2] = app->configuration.size[0] * app->configuration.size[1];
				}
			}
			if (axis_id == 2)
			{

				if (app->configuration.performTranspose[1]) {
					//transpose 0-1, transpose 1-2
					axis->specializationConstants.inputStride[1] = app->configuration.size[0] * app->configuration.size[2];
					axis->specializationConstants.inputStride[2] = app->configuration.size[2];
				}
				else {

					if (app->configuration.performTranspose[0]) {
						//transpose 0-1, no transpose 1-2
						axis->specializationConstants.inputStride[1] = app->configuration.size[0] * app->configuration.size[1];
						axis->specializationConstants.inputStride[2] = app->configuration.size[1];
					}
					else {
						//no transpose
						axis->specializationConstants.inputStride[1] = app->configuration.size[0] * app->configuration.size[1];
						axis->specializationConstants.inputStride[2] = app->configuration.size[0];
					}
				}
			}

			axis->specializationConstants.outputStride[0] = axis->specializationConstants.inputStride[0];
			axis->specializationConstants.outputStride[1] = axis->specializationConstants.inputStride[1];
			axis->specializationConstants.outputStride[2] = axis->specializationConstants.inputStride[2];
			axis->specializationConstants.outputStride[3] = axis->specializationConstants.inputStride[3];
			if (axis_id == 0) {
				if ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (app->configuration.isInputFormatted) && (!inverse)) {
					if (app->configuration.performZeropadding[0])
						axis->specializationConstants.inputStride[1] = app->configuration.size[0] / 2;

					if (app->configuration.performZeropadding[1])
						axis->specializationConstants.inputStride[2] = axis->specializationConstants.inputStride[1] * app->configuration.size[1] / 2;
					else
						axis->specializationConstants.inputStride[2] = axis->specializationConstants.inputStride[1] * app->configuration.size[1];

					if (app->configuration.performZeropadding[2])
						axis->specializationConstants.inputStride[3] = axis->specializationConstants.inputStride[2] * app->configuration.size[2] / 2;
					else
						axis->specializationConstants.inputStride[3] = axis->specializationConstants.inputStride[2] * app->configuration.size[2];
				}
				if ((axis_upload_id == 0) && (app->configuration.isOutputFormatted) && ((inverse) || ((app->configuration.performConvolution) && (app->configuration.FFTdim == 1)))) {
					if (app->configuration.performZeropadding[0])
						axis->specializationConstants.outputStride[1] = app->configuration.size[0] / 2;

					if (app->configuration.performZeropadding[1])
						axis->specializationConstants.outputStride[2] = axis->specializationConstants.outputStride[1] * app->configuration.size[1] / 2;
					else
						axis->specializationConstants.outputStride[2] = axis->specializationConstants.outputStride[1] * app->configuration.size[1];

					if (app->configuration.performZeropadding[2])
						axis->specializationConstants.outputStride[3] = axis->specializationConstants.outputStride[2] * app->configuration.size[2] / 2;
					else
						axis->specializationConstants.outputStride[3] = axis->specializationConstants.outputStride[2] * app->configuration.size[2];
				}
			}
		}
		axis->specializationConstants.inputStride[3] = (app->configuration.coordinateFeatures == 1) ? 0 : axis->specializationConstants.inputStride[3];
		axis->specializationConstants.outputStride[3] = (app->configuration.coordinateFeatures == 1) ? 0 : axis->specializationConstants.outputStride[3];

		axis->specializationConstants.inputStride[4] = ((app->configuration.numberBatches == 1) && (app->configuration.numberKernels == 1)) ? 0 : axis->specializationConstants.inputStride[3] * app->configuration.coordinateFeatures;
		axis->specializationConstants.outputStride[4] = ((app->configuration.numberBatches == 1) && (app->configuration.numberKernels == 1)) ? 0 : axis->specializationConstants.outputStride[3] * app->configuration.coordinateFeatures;

		axis->specializationConstants.inverse = inverse;
		if (inverse)
			axis->specializationConstants.zeropad[0] = (axis_upload_id == 0) ? app->configuration.performZeropadding[axis_id] : 0;
		else
			axis->specializationConstants.zeropad[0] = (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) ? app->configuration.performZeropadding[axis_id] : 0;
		if (axis_id == 0)
			axis->specializationConstants.zeropad[1] = app->configuration.performZeropadding[axis_id + 1];
		else
			axis->specializationConstants.zeropad[1] = 0;

		axis->specializationConstants.inputOffset = 0;
		axis->specializationConstants.outputOffset = 0;

		VkDescriptorPoolSize descriptorPoolSize = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
		descriptorPoolSize.descriptorCount = 2;
		if ((axis_id == 0) && (axis_upload_id == 0) && (app->configuration.FFTdim == 1) && (app->configuration.performConvolution))
			descriptorPoolSize.descriptorCount = 3;
		if ((axis_id == 1) && (axis_upload_id == 0) && (app->configuration.FFTdim == 2) && (app->configuration.performConvolution))
			descriptorPoolSize.descriptorCount = 3;
		if ((axis_id == 2) && (axis_upload_id == 0) && (app->configuration.FFTdim == 3) && (app->configuration.performConvolution))
			descriptorPoolSize.descriptorCount = 3;

		if (app->configuration.useLUT)
			descriptorPoolSize.descriptorCount++;
		VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
		descriptorPoolCreateInfo.poolSizeCount = 1;
		descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize;
		descriptorPoolCreateInfo.maxSets = 1;
		vkCreateDescriptorPool(app->configuration.device[0], &descriptorPoolCreateInfo, NULL, &axis->descriptorPool);

		const VkDescriptorType descriptorType[4] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
		VkDescriptorSetLayoutBinding* descriptorSetLayoutBindings;
		descriptorSetLayoutBindings = (VkDescriptorSetLayoutBinding*)malloc(descriptorPoolSize.descriptorCount * sizeof(VkDescriptorSetLayoutBinding));
		for (uint32_t i = 0; i < descriptorPoolSize.descriptorCount; ++i) {
			descriptorSetLayoutBindings[i].binding = i;
			descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
			descriptorSetLayoutBindings[i].descriptorCount = 1;
			descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		}

		VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
		descriptorSetLayoutCreateInfo.bindingCount = descriptorPoolSize.descriptorCount;
		descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;

		vkCreateDescriptorSetLayout(app->configuration.device[0], &descriptorSetLayoutCreateInfo, NULL, &axis->descriptorSetLayout);
		free(descriptorSetLayoutBindings);
		VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
		descriptorSetAllocateInfo.descriptorPool = axis->descriptorPool;
		descriptorSetAllocateInfo.descriptorSetCount = 1;
		descriptorSetAllocateInfo.pSetLayouts = &axis->descriptorSetLayout;
		vkAllocateDescriptorSets(app->configuration.device[0], &descriptorSetAllocateInfo, &axis->descriptorSet);
		for (uint32_t i = 0; i < descriptorPoolSize.descriptorCount; ++i) {
			VkDescriptorBufferInfo descriptorBufferInfo = { 0 };

			if (i == 0) {
				if ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (app->configuration.isInputFormatted) && (
					((axis_id == 0) && (!inverse))
					|| ((axis_id == app->configuration.FFTdim - 1) && (inverse) && (!app->configuration.performConvolution)))
					) {
					descriptorBufferInfo.buffer = app->configuration.inputBuffer[0];
					descriptorBufferInfo.range = app->configuration.inputBufferSize[0];
				}
				else {
					if ((axis_upload_id == 0) && (app->configuration.numberKernels > 1) && (inverse) && (!app->configuration.performConvolution)) {
						descriptorBufferInfo.buffer = app->configuration.outputBuffer[0];
						descriptorBufferInfo.range = app->configuration.outputBufferSize[0];
					}
					else {
						if ((FFTPlan->axes[axis_id]->specializationConstants.reorderFourStep == 1) && (FFTPlan->numAxisUploads[axis_id] > 1))
							if (axis_upload_id > 0)
								descriptorBufferInfo.buffer = app->configuration.buffer[0];
							else
								descriptorBufferInfo.buffer = app->configuration.tempBuffer[0];
						else
							descriptorBufferInfo.buffer = app->configuration.buffer[0];
						descriptorBufferInfo.range = app->configuration.bufferSize[0];
					}
				}
				descriptorBufferInfo.offset = 0;
			}
			if (i == 1) {
				if ((axis_upload_id == 0) && (app->configuration.isOutputFormatted && (
					((axis_id == 0) && (inverse))
					|| ((axis_id == app->configuration.FFTdim - 1) && (!inverse) && (!app->configuration.performConvolution))
					|| ((axis_id == 0) && (app->configuration.performConvolution) && (app->configuration.FFTdim == 1)))
					) ||
					((app->configuration.numberKernels > 1) && (
						(inverse)
						|| (axis_id == app->configuration.FFTdim - 1)))
					) {
					descriptorBufferInfo.buffer = app->configuration.outputBuffer[0];
					descriptorBufferInfo.range = app->configuration.outputBufferSize[0];
				}
				else {
					if ((FFTPlan->axes[axis_id]->specializationConstants.reorderFourStep == 1) && (FFTPlan->numAxisUploads[axis_id] > 1))
						if (axis_upload_id == 1)
							descriptorBufferInfo.buffer = app->configuration.tempBuffer[0];
						else
							descriptorBufferInfo.buffer = app->configuration.buffer[0];
					else
						descriptorBufferInfo.buffer = app->configuration.buffer[0];
					descriptorBufferInfo.range = app->configuration.bufferSize[0];
				}
				descriptorBufferInfo.offset = 0;
			}
			if ((i == 2) && (app->configuration.performConvolution)) {
				descriptorBufferInfo.buffer = app->configuration.kernel[0];
				descriptorBufferInfo.offset = 0;
				descriptorBufferInfo.range = app->configuration.kernelSize[0];
			}
			if ((i == descriptorPoolSize.descriptorCount - 1) && (app->configuration.useLUT)) {
				descriptorBufferInfo.buffer = axis->bufferLUT;
				descriptorBufferInfo.offset = 0;
				descriptorBufferInfo.range = axis->bufferLUTSize;
			}
			VkWriteDescriptorSet writeDescriptorSet = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
			writeDescriptorSet.dstSet = axis->descriptorSet;
			writeDescriptorSet.dstBinding = i;
			writeDescriptorSet.dstArrayElement = 0;
			writeDescriptorSet.descriptorType = descriptorType[i];
			writeDescriptorSet.descriptorCount = 1;
			writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
			vkUpdateDescriptorSets(app->configuration.device[0], 1, &writeDescriptorSet, 0, NULL);

		}

		{
			VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
			pipelineLayoutCreateInfo.setLayoutCount = 1;
			pipelineLayoutCreateInfo.pSetLayouts = &axis->descriptorSetLayout;

			VkPushConstantRange pushConstantRange = { VK_SHADER_STAGE_COMPUTE_BIT };
			pushConstantRange.offset = 0;
			pushConstantRange.size = sizeof(VkFFTPushConstantsLayout);
			// Push constant ranges are part of the pipeline layout
			pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
			pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;

			vkCreatePipelineLayout(app->configuration.device[0], &pipelineLayoutCreateInfo, NULL, &axis->pipelineLayout);
			if (!inverse) {
				if (axis_id == 0) {

					if (axis_upload_id == 0) {
						axis->axisBlock[0] = (axis->specializationConstants.fftDim / 8 > 1) ? axis->specializationConstants.fftDim / 8 : 1;
						if (axis->axisBlock[0] > 512) axis->axisBlock[0] = 512;
						if (app->configuration.reorderFourStep && (FFTPlan->numAxisUploads[axis_id] > 1))
							axis->axisBlock[1] = axis->groupedBatch;
						else
							axis->axisBlock[1] = 1;
						axis->axisBlock[2] = 1;
						axis->axisBlock[3] = axis->specializationConstants.fftDim;
					}
					else {
						axis->axisBlock[1] = (axis->specializationConstants.fftDim / 8 > 1) ? axis->specializationConstants.fftDim / 8 : 1;

						axis->axisBlock[0] = (axis->specializationConstants.stageStartSize > axis->groupedBatch) ? axis->groupedBatch : axis->specializationConstants.stageStartSize;

						axis->axisBlock[2] = 1;
						axis->axisBlock[3] = axis->specializationConstants.fftDim;
					}

				}
				if (axis_id == 1) {

					axis->axisBlock[1] = (axis->specializationConstants.fftDim / 8 > 1) ? axis->specializationConstants.fftDim / 8 : 1;

					if (app->configuration.performR2C) {
						if (axis_upload_id == 0) {
							for (uint32_t i = 0; i < 8; i++)
								VkFFTPlanSupportAxis(app, FFTPlan, 1, i, inverse);
						}
						axis->axisBlock[0] = (app->configuration.size[0] / 2 > axis->groupedBatch) ? axis->groupedBatch : app->configuration.size[0] / 2;
						/*if (axis->axisBlock[0] * axis->axisBlock[1] < 64)
							if (app->configuration.size[0]/2 > 64 / axis->axisBlock[1])
								axis->axisBlock[0] = 64 / axis->axisBlock[1];
							else
								axis->axisBlock[0] = app->configuration.size[0]/2;*/
					}
					else {
						axis->axisBlock[0] = (app->configuration.size[0] > axis->groupedBatch) ? axis->groupedBatch : app->configuration.size[0];
						/*if (axis->axisBlock[0] * axis->axisBlock[1] < 64)
							if (app->configuration.size[0] > 64 / axis->axisBlock[1])
								axis->axisBlock[0] = 64 / axis->axisBlock[1];
							else
								axis->axisBlock[0] = app->configuration.size[0];*/
					}

					axis->axisBlock[2] = 1;
					axis->axisBlock[3] = axis->specializationConstants.fftDim;

				}
				if (axis_id == 2) {
					axis->axisBlock[1] = (axis->specializationConstants.fftDim / 8 > 1) ? axis->specializationConstants.fftDim / 8 : 1;

					if (app->configuration.performR2C) {
						if (axis_upload_id == 0) {
							for (uint32_t i = 0; i < 8; i++)
								VkFFTPlanSupportAxis(app, FFTPlan, 2, i, inverse);
						}
						axis->axisBlock[0] = (app->configuration.size[0] / 2 > axis->groupedBatch) ? axis->groupedBatch : app->configuration.size[0] / 2;
						/*if (axis->axisBlock[0] * axis->axisBlock[1] < 64)
							if (app->configuration.size[0] / 2 > 64 / axis->axisBlock[1])
								axis->axisBlock[0] = 64 / axis->axisBlock[1];
							else
								axis->axisBlock[0] = app->configuration.size[0] / 2;*/
					}
					else {
						axis->axisBlock[0] = (app->configuration.size[0] > axis->groupedBatch) ? axis->groupedBatch : app->configuration.size[0];
						/*if (axis->axisBlock[0] * axis->axisBlock[1] < 64)
							if (app->configuration.size[0] > 64 / axis->axisBlock[1])
								axis->axisBlock[0] = 64 / axis->axisBlock[1];
							else
								axis->axisBlock[0] = app->configuration.size[0];*/
					}
					axis->axisBlock[2] = 1;
					axis->axisBlock[3] = axis->specializationConstants.fftDim;
				}
			}
			else {
				if (axis_id == 0) {
					if (axis_upload_id == 0) {
						axis->axisBlock[0] = (axis->specializationConstants.fftDim / 8 > 1) ? axis->specializationConstants.fftDim / 8 : 1;
						if (axis->axisBlock[0] > 512) axis->axisBlock[0] = 512;

						if (app->configuration.reorderFourStep && (FFTPlan->numAxisUploads[axis_id] > 1))
							axis->axisBlock[1] = axis->groupedBatch;
						else
							axis->axisBlock[1] = 1;
						axis->axisBlock[2] = 1;
						axis->axisBlock[3] = axis->specializationConstants.fftDim;
					}
					else {
						axis->axisBlock[1] = (axis->specializationConstants.fftDim / 8 > 1) ? axis->specializationConstants.fftDim / 8 : 1;

						axis->axisBlock[0] = (axis->specializationConstants.stageStartSize > axis->groupedBatch) ? axis->groupedBatch : axis->specializationConstants.stageStartSize;

						axis->axisBlock[2] = 1;
						axis->axisBlock[3] = axis->specializationConstants.fftDim;
					}
				}
				if (axis_id == 1) {

					axis->axisBlock[1] = (axis->specializationConstants.fftDim / 8 > 1) ? axis->specializationConstants.fftDim / 8 : 1;

					if (app->configuration.performR2C) {
						if (axis_upload_id == 0) {
							for (uint32_t i = 0; i < 8; i++)
								VkFFTPlanSupportAxis(app, FFTPlan, 1, i, inverse);
						}
						axis->axisBlock[0] = (app->configuration.size[0] / 2 > axis->groupedBatch) ? axis->groupedBatch : app->configuration.size[0] / 2;
						/*if (axis->axisBlock[0] * axis->axisBlock[1] < 64)
							if (app->configuration.size[0] / 2 > 64 / axis->axisBlock[1])
								axis->axisBlock[0] = 64 / axis->axisBlock[1];
							else
								axis->axisBlock[0] = app->configuration.size[0] / 2;*/
					}
					else {
						axis->axisBlock[0] = (app->configuration.size[0] > axis->groupedBatch) ? axis->groupedBatch : app->configuration.size[0];
						/*if (axis->axisBlock[0] * axis->axisBlock[1] < 64)
							if (app->configuration.size[0] > 64 / axis->axisBlock[1])
								axis->axisBlock[0] = 64 / axis->axisBlock[1];
							else
								axis->axisBlock[0] = app->configuration.size[0];*/
					}
					axis->axisBlock[2] = 1;
					axis->axisBlock[3] = axis->specializationConstants.fftDim;

				}
				if (axis_id == 2) {

					axis->axisBlock[1] = (axis->specializationConstants.fftDim / 8 > 1) ? axis->specializationConstants.fftDim / 8 : 1;

					if (app->configuration.performR2C) {
						if (axis_upload_id == 0) {
							for (uint32_t i = 0; i < 8; i++)
								VkFFTPlanSupportAxis(app, FFTPlan, 2, i, inverse);
						}
						axis->axisBlock[0] = (app->configuration.size[0] / 2 > axis->groupedBatch) ? axis->groupedBatch : app->configuration.size[0] / 2;
						/*if (axis->axisBlock[0] * axis->axisBlock[1] < 64)
							if (app->configuration.size[0] / 2 > 64 / axis->axisBlock[1])
								axis->axisBlock[0] = 64 / axis->axisBlock[1];
							else
								axis->axisBlock[0] = app->configuration.size[0] / 2;*/
					}
					else {
						axis->axisBlock[0] = (app->configuration.size[0] > axis->groupedBatch) ? axis->groupedBatch : app->configuration.size[0];
						/*if (axis->axisBlock[0] * axis->axisBlock[1] < 64)
							if (app->configuration.size[0] > 64 / axis->axisBlock[1])
								axis->axisBlock[0] = 64 / axis->axisBlock[1];
							else
								axis->axisBlock[0] = app->configuration.size[0];*/
					}
					axis->axisBlock[2] = 1;
					axis->axisBlock[3] = axis->specializationConstants.fftDim;

				}

			}


			uint32_t tempSize[3] = { app->configuration.size[0], app->configuration.size[1], app->configuration.size[2] };


			if (axis_id == 0) {
				if (axis_upload_id == 0)
					tempSize[0] = app->configuration.size[0] / axis->specializationConstants.fftDim / axis->axisBlock[1];
				else
					tempSize[0] = app->configuration.size[0] / axis->specializationConstants.fftDim / axis->axisBlock[0];
				if (app->configuration.performR2C == 1) tempSize[1] = ceil(tempSize[1] / 2.0);
				if (app->configuration.performZeropadding[1]) tempSize[1] = ceil(tempSize[1] / 2.0);
				if (app->configuration.performZeropadding[2]) tempSize[2] = ceil(tempSize[2] / 2.0);
				if (tempSize[0] > app->configuration.maxComputeWorkGroupCount[0]) axis->specializationConstants.performWorkGroupShift[0] = 1;
				else  axis->specializationConstants.performWorkGroupShift[0] = 0;
				if (tempSize[1] > app->configuration.maxComputeWorkGroupCount[1]) axis->specializationConstants.performWorkGroupShift[1] = 1;
				else  axis->specializationConstants.performWorkGroupShift[1] = 0;
				if (tempSize[2] > app->configuration.maxComputeWorkGroupCount[2]) axis->specializationConstants.performWorkGroupShift[2] = 1;
				else  axis->specializationConstants.performWorkGroupShift[2] = 0;
			}
			if (axis_id == 1) {
				tempSize[0] = app->configuration.size[0] / axis->axisBlock[0] * app->configuration.size[1] / axis->specializationConstants.fftDim;
				tempSize[1] = 1;
				tempSize[2] = app->configuration.size[2];
				if (app->configuration.performR2C == 1) tempSize[0] = ceil(tempSize[0] / 2.0);
				if (app->configuration.performZeropadding[2]) tempSize[2] = ceil(tempSize[2] / 2.0);

				if (tempSize[0] > app->configuration.maxComputeWorkGroupCount[0]) axis->specializationConstants.performWorkGroupShift[0] = 1;
				else  axis->specializationConstants.performWorkGroupShift[0] = 0;
				if (tempSize[1] > app->configuration.maxComputeWorkGroupCount[1]) axis->specializationConstants.performWorkGroupShift[1] = 1;
				else  axis->specializationConstants.performWorkGroupShift[1] = 0;
				if (tempSize[2] > app->configuration.maxComputeWorkGroupCount[2]) axis->specializationConstants.performWorkGroupShift[2] = 1;
				else  axis->specializationConstants.performWorkGroupShift[2] = 0;

			}
			if (axis_id == 2) {
				tempSize[0] = app->configuration.size[0] / axis->axisBlock[0] * app->configuration.size[2] / axis->specializationConstants.fftDim;
				tempSize[1] = 1;
				tempSize[2] = app->configuration.size[1];
				if (app->configuration.performR2C == 1) tempSize[0] = ceil(tempSize[0] / 2.0);

				if (tempSize[0] > app->configuration.maxComputeWorkGroupCount[0]) axis->specializationConstants.performWorkGroupShift[0] = 1;
				else  axis->specializationConstants.performWorkGroupShift[0] = 0;
				if (tempSize[1] > app->configuration.maxComputeWorkGroupCount[1]) axis->specializationConstants.performWorkGroupShift[1] = 1;
				else  axis->specializationConstants.performWorkGroupShift[1] = 0;
				if (tempSize[2] > app->configuration.maxComputeWorkGroupCount[2]) axis->specializationConstants.performWorkGroupShift[2] = 1;
				else  axis->specializationConstants.performWorkGroupShift[2] = 0;

			}
			VkSpecializationMapEntry specializationMapEntries[30] = { {} };
			for (uint32_t i = 0; i < 30; i++) {
				specializationMapEntries[i].constantID = i + 1;
				specializationMapEntries[i].size = sizeof(uint32_t);
				specializationMapEntries[i].offset = i * sizeof(uint32_t);
			}
			VkSpecializationInfo specializationInfo = { 0 };
			specializationInfo.dataSize = 30 * sizeof(uint32_t);
			specializationInfo.mapEntryCount = 30;
			specializationInfo.pMapEntries = specializationMapEntries;
			axis->specializationConstants.localSize[0] = axis->axisBlock[0];
			axis->specializationConstants.localSize[1] = axis->axisBlock[1];
			axis->specializationConstants.localSize[2] = axis->axisBlock[2];
			specializationInfo.pData = &axis->specializationConstants;
			VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };

			VkComputePipelineCreateInfo computePipelineCreateInfo = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };


			pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			uint32_t registerBoost = (FFTPlan->numAxisUploads[axis_id] > 1) ? app->configuration.registerBoost4Step : app->configuration.registerBoost;

			if (app->configuration.performR2C) {
				if (axis_id == 0) {
					if (inverse) {
						switch (registerBoost) {
						case 1:
						{
							VkFFTInitShader(app, 1, &pipelineShaderStageCreateInfo.module);
							break;
						}
						case 2:
						{
							switch (axis->specializationConstants.fftDim) {
							case 8192:
								VkFFTInitShader(app, 23, &pipelineShaderStageCreateInfo.module);
								break;
							default:
								VkFFTInitShader(app, 1, &pipelineShaderStageCreateInfo.module);
								break;
							}
							break;
						}
						case 4:
						{
							switch (axis->specializationConstants.fftDim) {
							case 8192:
								VkFFTInitShader(app, 23, &pipelineShaderStageCreateInfo.module);
								break;
							case 16384:
								VkFFTInitShader(app, 33, &pipelineShaderStageCreateInfo.module);
								break;
							default:
								VkFFTInitShader(app, 1, &pipelineShaderStageCreateInfo.module);
								break;
							}
							break;
						}
						}
					}
					else {
						switch (registerBoost) {
						case 1:
						{
							VkFFTInitShader(app, 3, &pipelineShaderStageCreateInfo.module);
							break;
						}
						case 2:
						{
							switch (axis->specializationConstants.fftDim) {
							case 8192:
								VkFFTInitShader(app, 24, &pipelineShaderStageCreateInfo.module);
								break;
							default:
								VkFFTInitShader(app, 3, &pipelineShaderStageCreateInfo.module);
								break;
							}
							break;
						}
						case 4:
						{
							switch (axis->specializationConstants.fftDim) {
							case 8192:
								VkFFTInitShader(app, 24, &pipelineShaderStageCreateInfo.module);
								break;
							case 16384:
								VkFFTInitShader(app, 34, &pipelineShaderStageCreateInfo.module);
								break;
							default:
								VkFFTInitShader(app, 3, &pipelineShaderStageCreateInfo.module);
								break;
							}
							break;
						}
						}

					}
				}
				if (axis_id == 1) {

					if ((app->configuration.FFTdim == 2) && (app->configuration.performConvolution) && (axis_upload_id == 0)) {


						switch (app->configuration.matrixConvolution) {
						case 1:
							VkFFTInitShader(app, 8, &pipelineShaderStageCreateInfo.module);
							break;
						case 2:
							if (app->configuration.symmetricKernel)
								VkFFTInitShader(app, 11, &pipelineShaderStageCreateInfo.module);
							else
								VkFFTInitShader(app, 14, &pipelineShaderStageCreateInfo.module);
							break;
						case 3:
							if (app->configuration.symmetricKernel)
								VkFFTInitShader(app, 17, &pipelineShaderStageCreateInfo.module);
							else
								VkFFTInitShader(app, 20, &pipelineShaderStageCreateInfo.module);
							break;
						}

					}
					else {
						VkFFTInitShader(app, 7, &pipelineShaderStageCreateInfo.module);
					}

				}

				if (axis_id == 2) {
					if ((app->configuration.FFTdim == 3) && (app->configuration.performConvolution) && (axis_upload_id == 0)) {

						switch (app->configuration.matrixConvolution) {
						case 1:
							VkFFTInitShader(app, 8, &pipelineShaderStageCreateInfo.module);
							break;
						case 2:
							if (app->configuration.symmetricKernel)
								VkFFTInitShader(app, 11, &pipelineShaderStageCreateInfo.module);
							else
								VkFFTInitShader(app, 14, &pipelineShaderStageCreateInfo.module);
							break;
						case 3:
							if (app->configuration.symmetricKernel)
								VkFFTInitShader(app, 17, &pipelineShaderStageCreateInfo.module);
							else
								VkFFTInitShader(app, 20, &pipelineShaderStageCreateInfo.module);
							break;
						}

					}
					else {

						VkFFTInitShader(app, 7, &pipelineShaderStageCreateInfo.module);
					}
				}
			}
			else {
				if (axis_id == 0) {
					if ((app->configuration.FFTdim == 1) && (app->configuration.performConvolution) && (axis_upload_id == 0)) {
						if (axis_upload_id == 0) {
							switch (app->configuration.matrixConvolution) {
							case 1:
								VkFFTInitShader(app, 9, &pipelineShaderStageCreateInfo.module);
								break;
							case 2:
								if (app->configuration.symmetricKernel)
									VkFFTInitShader(app, 12, &pipelineShaderStageCreateInfo.module);
								else
									VkFFTInitShader(app, 15, &pipelineShaderStageCreateInfo.module);
								break;
							case 3:
								if (app->configuration.symmetricKernel)
									VkFFTInitShader(app, 18, &pipelineShaderStageCreateInfo.module);
								else
									VkFFTInitShader(app, 21, &pipelineShaderStageCreateInfo.module);
								break;
							}
						}
						else {
							switch (app->configuration.matrixConvolution) {
							case 1:
								VkFFTInitShader(app, 10, &pipelineShaderStageCreateInfo.module);
								break;
							case 2:
								if (app->configuration.symmetricKernel)
									VkFFTInitShader(app, 13, &pipelineShaderStageCreateInfo.module);
								else
									VkFFTInitShader(app, 16, &pipelineShaderStageCreateInfo.module);
								break;
							case 3:
								if (app->configuration.symmetricKernel)
									VkFFTInitShader(app, 19, &pipelineShaderStageCreateInfo.module);
								else
									VkFFTInitShader(app, 22, &pipelineShaderStageCreateInfo.module);
								break;
							}
						}
					}
					else {
						switch (registerBoost) {
						case 1:
						{
							if (axis_upload_id == 0)
								VkFFTInitShader(app, 0, &pipelineShaderStageCreateInfo.module);
							else
								VkFFTInitShader(app, 2, &pipelineShaderStageCreateInfo.module);
							break;
						}
						case 2:
						{
							switch (axis->specializationConstants.fftDim) {
							case 8192:
								VkFFTInitShader(app, 25, &pipelineShaderStageCreateInfo.module);
								break;
							default:
								if (axis_upload_id == 0)
									VkFFTInitShader(app, 0, &pipelineShaderStageCreateInfo.module);
								else
									VkFFTInitShader(app, 2, &pipelineShaderStageCreateInfo.module);
								break;
							}
							break;
						}
						case 4:
						{
							switch (axis->specializationConstants.fftDim) {
							case 8192:
								VkFFTInitShader(app, 25, &pipelineShaderStageCreateInfo.module);
								break;
							case 16384:
								VkFFTInitShader(app, 35, &pipelineShaderStageCreateInfo.module);
								break;
							default:
								if (axis_upload_id == 0)
									VkFFTInitShader(app, 0, &pipelineShaderStageCreateInfo.module);
								else
									VkFFTInitShader(app, 2, &pipelineShaderStageCreateInfo.module);
								break;
							}
							break;
						}
						}
					}
				}
				if (axis_id == 1) {

					if ((app->configuration.FFTdim == 2) && (app->configuration.performConvolution) && (axis_upload_id == 0)) {

						switch (app->configuration.matrixConvolution) {
						case 1:
							VkFFTInitShader(app, 8, &pipelineShaderStageCreateInfo.module);
							break;
						case 2:
							if (app->configuration.symmetricKernel)
								VkFFTInitShader(app, 11, &pipelineShaderStageCreateInfo.module);
							else
								VkFFTInitShader(app, 14, &pipelineShaderStageCreateInfo.module);
							break;
						case 3:
							if (app->configuration.symmetricKernel)
								VkFFTInitShader(app, 17, &pipelineShaderStageCreateInfo.module);
							else
								VkFFTInitShader(app, 20, &pipelineShaderStageCreateInfo.module);
							break;
						}

					}
					else {
						VkFFTInitShader(app, 7, &pipelineShaderStageCreateInfo.module);

					}

				}

				if (axis_id == 2) {
					if ((app->configuration.FFTdim == 3) && (app->configuration.performConvolution) && (axis_upload_id == 0)) {

						switch (app->configuration.matrixConvolution) {
						case 1:
							VkFFTInitShader(app, 8, &pipelineShaderStageCreateInfo.module);
							break;
						case 2:
							if (app->configuration.symmetricKernel)
								VkFFTInitShader(app, 11, &pipelineShaderStageCreateInfo.module);
							else
								VkFFTInitShader(app, 14, &pipelineShaderStageCreateInfo.module);
							break;
						case 3:
							if (app->configuration.symmetricKernel)
								VkFFTInitShader(app, 17, &pipelineShaderStageCreateInfo.module);
							else
								VkFFTInitShader(app, 20, &pipelineShaderStageCreateInfo.module);
							break;
						}

					}
					else {
						VkFFTInitShader(app, 7, &pipelineShaderStageCreateInfo.module);
					}
				}
			}

			pipelineShaderStageCreateInfo.pName = "main";
			pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
			computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
			computePipelineCreateInfo.layout = axis->pipelineLayout;



			vkCreateComputePipelines(app->configuration.device[0], VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, &axis->pipeline);
			vkDestroyShaderModule(app->configuration.device[0], pipelineShaderStageCreateInfo.module, NULL);
		}


	}
	void VkFFTPlanTranspose(VkFFTApplication* app, VkFFTPlan* FFTPlan, uint32_t axis_id, VkBool32 inverse) {
		if (axis_id == 0) {
			if (app->configuration.performR2C) {
				FFTPlan->transpose[0].specializationConstants.ratio = (app->configuration.size[0] / app->configuration.size[1] / 2 >= 1) ? app->configuration.size[0] / app->configuration.size[1] / 2 : 2 * app->configuration.size[1] / app->configuration.size[0];
				FFTPlan->transpose[0].specializationConstants.ratioDirection = (app->configuration.size[0] / app->configuration.size[1] / 2 >= 1) ? 1 : 0;
			}
			else {
				FFTPlan->transpose[0].specializationConstants.ratio = (app->configuration.size[0] / app->configuration.size[1] >= 1) ? app->configuration.size[0] / app->configuration.size[1] : app->configuration.size[1] / app->configuration.size[0];
				FFTPlan->transpose[0].specializationConstants.ratioDirection = (app->configuration.size[0] / app->configuration.size[1] >= 1) ? 1 : 0;

			}
		}
		if (axis_id == 1) {
			FFTPlan->transpose[1].specializationConstants.ratio = (app->configuration.size[1] / app->configuration.size[2] >= 1) ? app->configuration.size[1] / app->configuration.size[2] : app->configuration.size[2] / app->configuration.size[1];
			FFTPlan->transpose[1].specializationConstants.ratioDirection = (app->configuration.size[1] / app->configuration.size[2] >= 1) ? 1 : 0;
		}

		if (axis_id == 0) {
			if (app->configuration.performR2C) {
				FFTPlan->transpose[axis_id].specializationConstants.inputStride[0] = 1;
				FFTPlan->transpose[axis_id].specializationConstants.inputStride[1] = (FFTPlan->transpose[0].specializationConstants.ratioDirection) ? app->configuration.size[0] / 2 : app->configuration.size[1];
				FFTPlan->transpose[axis_id].specializationConstants.inputStride[2] = (app->configuration.size[0] / 2 + 1) * app->configuration.size[1];
				FFTPlan->transpose[axis_id].specializationConstants.inputStride[3] = (app->configuration.size[0] / 2 + 1) * app->configuration.size[1] * app->configuration.size[2];
			}
			else {
				FFTPlan->transpose[axis_id].specializationConstants.inputStride[0] = 1;
				FFTPlan->transpose[axis_id].specializationConstants.inputStride[1] = (FFTPlan->transpose[0].specializationConstants.ratioDirection) ? app->configuration.size[0] : app->configuration.size[1];
				FFTPlan->transpose[axis_id].specializationConstants.inputStride[2] = app->configuration.size[0] * app->configuration.size[1];
				FFTPlan->transpose[axis_id].specializationConstants.inputStride[3] = app->configuration.size[0] * app->configuration.size[1] * app->configuration.size[2];
			}
		}
		if (axis_id == 1) {
			if (app->configuration.performR2C) {
				FFTPlan->transpose[axis_id].specializationConstants.inputStride[0] = 1;
				FFTPlan->transpose[axis_id].specializationConstants.inputStride[1] = (FFTPlan->transpose[1].specializationConstants.ratioDirection) ? (app->configuration.size[0] / 2 + 1) * app->configuration.size[1] : (app->configuration.size[0] / 2 + 1) * app->configuration.size[2];
				FFTPlan->transpose[axis_id].specializationConstants.inputStride[2] = (FFTPlan->transpose[1].specializationConstants.ratioDirection) ? app->configuration.size[1] : app->configuration.size[2];
				FFTPlan->transpose[axis_id].specializationConstants.inputStride[3] = (app->configuration.size[0] / 2 + 1) * app->configuration.size[1] * app->configuration.size[2];
			}
			else {
				FFTPlan->transpose[axis_id].specializationConstants.inputStride[0] = 1;
				FFTPlan->transpose[axis_id].specializationConstants.inputStride[1] = (FFTPlan->transpose[1].specializationConstants.ratioDirection) ? app->configuration.size[0] * app->configuration.size[1] : app->configuration.size[0] * app->configuration.size[2];
				FFTPlan->transpose[axis_id].specializationConstants.inputStride[2] = (FFTPlan->transpose[1].specializationConstants.ratioDirection) ? app->configuration.size[1] : app->configuration.size[2];
				FFTPlan->transpose[axis_id].specializationConstants.inputStride[3] = app->configuration.size[0] * app->configuration.size[1] * app->configuration.size[2];
			}
		}
		FFTPlan->transpose[axis_id].specializationConstants.inputStride[4] = FFTPlan->transpose[axis_id].specializationConstants.inputStride[3] * app->configuration.coordinateFeatures;
		VkDescriptorPoolSize descriptorPoolSize = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
		descriptorPoolSize.descriptorCount = 2;
		//collection->descriptorNum = 3;

		VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
		descriptorPoolCreateInfo.poolSizeCount = 1;
		descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize;
		descriptorPoolCreateInfo.maxSets = 1;
		vkCreateDescriptorPool(app->configuration.device[0], &descriptorPoolCreateInfo, NULL, &FFTPlan->transpose[axis_id].descriptorPool);

		const VkDescriptorType descriptorType[2] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
		VkDescriptorSetLayoutBinding* descriptorSetLayoutBindings;
		descriptorSetLayoutBindings = (VkDescriptorSetLayoutBinding*)malloc(descriptorPoolSize.descriptorCount * sizeof(VkDescriptorSetLayoutBinding));
		for (uint32_t i = 0; i < descriptorPoolSize.descriptorCount; ++i) {
			descriptorSetLayoutBindings[i].binding = i;
			descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
			descriptorSetLayoutBindings[i].descriptorCount = 1;
			descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		}

		VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
		descriptorSetLayoutCreateInfo.bindingCount = descriptorPoolSize.descriptorCount;
		descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;

		vkCreateDescriptorSetLayout(app->configuration.device[0], &descriptorSetLayoutCreateInfo, NULL, &FFTPlan->transpose[axis_id].descriptorSetLayout);
		free(descriptorSetLayoutBindings);
		VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
		descriptorSetAllocateInfo.descriptorPool = FFTPlan->transpose[axis_id].descriptorPool;
		descriptorSetAllocateInfo.descriptorSetCount = 1;
		descriptorSetAllocateInfo.pSetLayouts = &FFTPlan->transpose[axis_id].descriptorSetLayout;
		vkAllocateDescriptorSets(app->configuration.device[0], &descriptorSetAllocateInfo, &FFTPlan->transpose[axis_id].descriptorSet);
		for (uint32_t i = 0; i < descriptorPoolSize.descriptorCount; ++i) {


			VkDescriptorBufferInfo descriptorBufferInfo = { 0 };
			if (i == 0) {
				if ((app->configuration.numberKernels > 1) && (inverse)) {
					descriptorBufferInfo.buffer = app->configuration.outputBuffer[0];
					descriptorBufferInfo.range = app->configuration.outputBufferSize[0];
				}
				else {
					descriptorBufferInfo.buffer = app->configuration.buffer[0];
					descriptorBufferInfo.range = app->configuration.bufferSize[0];
				}
				descriptorBufferInfo.offset = 0;
			}
			if (i == 1) {
				if ((app->configuration.numberKernels > 1) && (inverse)) {
					descriptorBufferInfo.buffer = app->configuration.outputBuffer[0];
					descriptorBufferInfo.range = app->configuration.outputBufferSize[0];
				}
				else {
					descriptorBufferInfo.buffer = app->configuration.buffer[0];
					descriptorBufferInfo.range = app->configuration.bufferSize[0];
				}
				descriptorBufferInfo.offset = 0;
			}

			VkWriteDescriptorSet writeDescriptorSet = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
			writeDescriptorSet.dstSet = FFTPlan->transpose[axis_id].descriptorSet;
			writeDescriptorSet.dstBinding = i;
			writeDescriptorSet.dstArrayElement = 0;
			writeDescriptorSet.descriptorType = descriptorType[i];
			writeDescriptorSet.descriptorCount = 1;
			writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
			vkUpdateDescriptorSets(app->configuration.device[0], 1, &writeDescriptorSet, 0, NULL);
		}



		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
		pipelineLayoutCreateInfo.setLayoutCount = 1;
		pipelineLayoutCreateInfo.pSetLayouts = &FFTPlan->transpose[axis_id].descriptorSetLayout;

		VkPushConstantRange pushConstantRange = { VK_SHADER_STAGE_COMPUTE_BIT };
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(VkFFTPushConstantsLayout);
		// Push constant ranges are part of the pipeline layout
		pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
		pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;

		vkCreatePipelineLayout(app->configuration.device[0], &pipelineLayoutCreateInfo, NULL, &FFTPlan->transpose[axis_id].pipelineLayout);
		VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };

		VkComputePipelineCreateInfo computePipelineCreateInfo = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };

		uint32_t max_dim = 1;
		if (FFTPlan->axes[axis_id][0].axisBlock[1] * app->configuration.size[axis_id] < pow(2, floor(log2(sqrt(1024 * FFTPlan->transpose[axis_id].specializationConstants.ratio)))))
			max_dim = FFTPlan->axes[axis_id][0].axisBlock[1] * app->configuration.size[axis_id];
		else
			max_dim = pow(2, floor(log2(sqrt(1024 * FFTPlan->transpose[axis_id].specializationConstants.ratio))));
		FFTPlan->transpose[axis_id].transposeBlock[0] = max_dim;
		FFTPlan->transpose[axis_id].transposeBlock[1] = max_dim / FFTPlan->transpose[axis_id].specializationConstants.ratio;
		FFTPlan->transpose[axis_id].transposeBlock[2] = 1;

		VkSpecializationMapEntry specializationMapEntries[12] = { {} };
		for (uint32_t i = 0; i < 12; i++) {
			specializationMapEntries[i].constantID = i + 1;
			specializationMapEntries[i].size = sizeof(uint32_t);
			specializationMapEntries[i].offset = i * sizeof(uint32_t);
		}
		VkSpecializationInfo specializationInfo = { 0 };
		specializationInfo.dataSize = 12 * sizeof(uint32_t);
		specializationInfo.mapEntryCount = 12;
		specializationInfo.pMapEntries = specializationMapEntries;
		FFTPlan->transpose[axis_id].specializationConstants.localSize[0] = FFTPlan->transpose[axis_id].transposeBlock[0];
		FFTPlan->transpose[axis_id].specializationConstants.localSize[1] = FFTPlan->transpose[axis_id].transposeBlock[1];
		FFTPlan->transpose[axis_id].specializationConstants.localSize[2] = FFTPlan->transpose[axis_id].transposeBlock[2];

		specializationInfo.pData = &FFTPlan->transpose[axis_id].specializationConstants;

		pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;

		uint32_t filelength;
		//printf("vkFFT_transpose_inplace\n");
		char filename[512];
		sprintf(filename, "%s%s", app->configuration.shaderPath, "vkFFT_transpose_inplace.spv");

		uint32_t* code = VkFFTReadShader(&filelength, filename);
		VkShaderModuleCreateInfo createInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
		createInfo.pCode = code;
		createInfo.codeSize = filelength;
		vkCreateShaderModule(app->configuration.device[0], &createInfo, NULL, &pipelineShaderStageCreateInfo.module);
		free(code);

		pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
		pipelineShaderStageCreateInfo.pName = "main";
		computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
		computePipelineCreateInfo.layout = FFTPlan->transpose[axis_id].pipelineLayout;


		vkCreateComputePipelines(app->configuration.device[0], VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, &FFTPlan->transpose[axis_id].pipeline);
		vkDestroyShaderModule(app->configuration.device[0], pipelineShaderStageCreateInfo.module, NULL);

	}
	void deleteAxis(VkFFTApplication* app, VkFFTAxis* axis) {
		if (app->configuration.useLUT) {
			vkDestroyBuffer(app->configuration.device[0], axis->bufferLUT, NULL);
			vkFreeMemory(app->configuration.device[0], axis->bufferLUTDeviceMemory, NULL);
		}
		vkDestroyDescriptorPool(app->configuration.device[0], axis->descriptorPool, NULL);
		vkDestroyDescriptorSetLayout(app->configuration.device[0], axis->descriptorSetLayout, NULL);
		vkDestroyPipelineLayout(app->configuration.device[0], axis->pipelineLayout, NULL);
		vkDestroyPipeline(app->configuration.device[0], axis->pipeline, NULL);


	}
	void deleteTranspose(VkFFTApplication* app, VkFFTTranspose* transpose) {
		vkDestroyDescriptorPool(app->configuration.device[0], transpose->descriptorPool, NULL);
		vkDestroyDescriptorSetLayout(app->configuration.device[0], transpose->descriptorSetLayout, NULL);
		vkDestroyPipelineLayout(app->configuration.device[0], transpose->pipelineLayout, NULL);
		vkDestroyPipeline(app->configuration.device[0], transpose->pipeline, NULL);


	}
	void initializeVulkanFFT(VkFFTApplication* app, VkFFTConfiguration inputLaunchConfiguration) {
		VkPhysicalDeviceProperties physicalDeviceProperties = {};
		vkGetPhysicalDeviceProperties(inputLaunchConfiguration.physicalDevice[0], &physicalDeviceProperties);
		app->configuration = inputLaunchConfiguration;
		app->configuration.maxComputeWorkGroupCount[0] = physicalDeviceProperties.limits.maxComputeWorkGroupCount[0];
		app->configuration.maxComputeWorkGroupCount[1] = physicalDeviceProperties.limits.maxComputeWorkGroupCount[1];
		app->configuration.maxComputeWorkGroupCount[2] = physicalDeviceProperties.limits.maxComputeWorkGroupCount[2];

		if (app->configuration.matrixConvolution > 1) app->configuration.coordinateFeatures = app->configuration.matrixConvolution;

		if (app->configuration.performConvolution) {

			app->configuration.inverse = 0;
			for (uint32_t i = 0; i < app->configuration.FFTdim; i++) {
				VkFFTScheduler(app, &app->localFFTPlan, i);
				for (uint32_t j = 0; j < app->localFFTPlan.numAxisUploads[i]; j++)
					VkFFTPlanAxis(app, &app->localFFTPlan_inverse_convolution, i, j, 1);
			}

		}
		for (uint32_t i = 0; i < app->configuration.FFTdim; i++) {
			VkFFTScheduler(app, &app->localFFTPlan, i);
			for (uint32_t j = 0; j < app->localFFTPlan.numAxisUploads[i]; j++)
				VkFFTPlanAxis(app, &app->localFFTPlan, i, j, app->configuration.inverse);
		}

	}
	void dispatchEnchanced(VkFFTApplication* app, VkCommandBuffer commandBuffer, VkFFTAxis* axis, uint32_t* dispatchBlock) {
		uint32_t maxBlockPow2Size[3] = { (uint32_t)pow(2,(uint32_t)log2(app->configuration.maxComputeWorkGroupCount[0])),(uint32_t)pow(2,(uint32_t)log2(app->configuration.maxComputeWorkGroupCount[1])),(uint32_t)pow(2,(uint32_t)log2(app->configuration.maxComputeWorkGroupCount[2])) };
		uint32_t blockNumber[3] = { (uint32_t)ceil(dispatchBlock[0] / (float)maxBlockPow2Size[0]),(uint32_t)ceil(dispatchBlock[1] / (float)maxBlockPow2Size[1]),(uint32_t)ceil(dispatchBlock[2] / (float)maxBlockPow2Size[2]) };
		for (uint32_t i = 0; i < 3; i++)
			if (blockNumber[i] == 1) maxBlockPow2Size[i] = dispatchBlock[i];
		for (uint32_t i = 0; i < blockNumber[0]; i++) {
			for (uint32_t j = 0; j < blockNumber[1]; j++) {
				for (uint32_t k = 0; k < blockNumber[2]; k++) {
					axis->pushConstants.workGroupShift[0] = i * maxBlockPow2Size[0];
					axis->pushConstants.workGroupShift[1] = j * maxBlockPow2Size[1];
					axis->pushConstants.workGroupShift[2] = k * maxBlockPow2Size[2];
					vkCmdPushConstants(commandBuffer, axis->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTPushConstantsLayout), &axis->pushConstants);
					vkCmdDispatch(commandBuffer, maxBlockPow2Size[0], maxBlockPow2Size[1], maxBlockPow2Size[2]);
				}
			}
		}
	}
	void VkFFTAppend(VkFFTApplication* app, VkCommandBuffer commandBuffer) {
		VkMemoryBarrier memory_barrier = {
				VK_STRUCTURE_TYPE_MEMORY_BARRIER,
				0,
				VK_ACCESS_SHADER_WRITE_BIT,
				VK_ACCESS_SHADER_READ_BIT,
		};
		if (!app->configuration.inverse) {
			//FFT axis 0
			for (uint32_t j = 0; j < app->configuration.numberBatches; j++) {
				for (int l = app->localFFTPlan.numAxisUploads[0] - 1; l >= 0; l--) {
					VkFFTAxis* axis = &app->localFFTPlan.axes[0][l];
					axis->pushConstants.batch = j;
					uint32_t maxCoordinate = ((app->configuration.matrixConvolution) > 1 && (app->configuration.performConvolution) && (app->configuration.FFTdim == 1)) ? 1 : app->configuration.coordinateFeatures;
					for (uint32_t i = 0; i < maxCoordinate; i++) {
						axis->pushConstants.coordinate = i;


						vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
						vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
						uint32_t dispatchBlock[3];
						if (l == 0)
							dispatchBlock[0] = app->configuration.size[0] / axis->specializationConstants.fftDim / axis->axisBlock[1];
						else
							dispatchBlock[0] = app->configuration.size[0] / axis->specializationConstants.fftDim / axis->axisBlock[0];

						dispatchBlock[1] = app->configuration.size[1];
						dispatchBlock[2] = app->configuration.size[2];
						if (app->configuration.performR2C == 1) dispatchBlock[1] = ceil(dispatchBlock[1] / 2.0);
						if (app->configuration.performZeropadding[1]) dispatchBlock[1] = ceil(dispatchBlock[1] / 2.0);
						if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
						dispatchEnchanced(app, commandBuffer, axis, dispatchBlock);
					}
					vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);
				}
			}

			if (app->configuration.FFTdim > 1) {

				//FFT axis 1
				if ((app->configuration.FFTdim == 2) && (app->configuration.performConvolution)) {
					if (app->configuration.performR2C == 1) {
						for (int l = app->localFFTPlan.numSupportAxisUploads[0] - 1; l >= 0; l--) {
							VkFFTAxis* axis = &app->localFFTPlan.supportAxes[0][l];
							uint32_t maxCoordinate = ((app->configuration.matrixConvolution > 1) && (l == 0)) ? 1 : app->configuration.coordinateFeatures;
							for (uint32_t i = 0; i < maxCoordinate; i++) {
								axis->pushConstants.coordinate = i;

								axis->pushConstants.batch = ((l == 0) && (app->configuration.matrixConvolution == 1)) ? app->configuration.numberKernels : 0;


								vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
								vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
								uint32_t dispatchBlock[3];
								if (l == 0)
									dispatchBlock[0] = app->configuration.size[1] / axis->specializationConstants.fftDim;
								else
									dispatchBlock[0] = app->configuration.size[1] / axis->specializationConstants.fftDim / axis->axisBlock[0];

								dispatchBlock[1] = 1;
								dispatchBlock[2] = app->configuration.size[2];
								if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
								dispatchEnchanced(app, commandBuffer, axis, dispatchBlock);

							}
							if (l > 0)
								vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

						}

					}

					for (int l = app->localFFTPlan.numAxisUploads[1] - 1; l >= 0; l--) {
						VkFFTAxis* axis = &app->localFFTPlan.axes[1][l];
						uint32_t maxCoordinate = ((app->configuration.matrixConvolution > 1) && (l == 0)) ? 1 : app->configuration.coordinateFeatures;
						for (uint32_t i = 0; i < maxCoordinate; i++) {

							axis->pushConstants.coordinate = i;
							axis->pushConstants.batch = ((l == 0) && (app->configuration.matrixConvolution == 1)) ? app->configuration.numberKernels : 0;

							vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
							vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
							uint32_t dispatchBlock[3];
							dispatchBlock[0] = app->configuration.size[0] / axis->axisBlock[0] * app->configuration.size[1] / axis->specializationConstants.fftDim;
							dispatchBlock[1] = 1;
							dispatchBlock[2] = app->configuration.size[2];
							if (app->configuration.performR2C == 1) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);
							if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
							dispatchEnchanced(app, commandBuffer, axis, dispatchBlock);
						}
						vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

					}
				}
				else {
					if (app->configuration.performR2C == 1) {
						for (uint32_t j = 0; j < app->configuration.numberBatches; j++) {
							for (int l = app->localFFTPlan.numSupportAxisUploads[0] - 1; l >= 0; l--) {
								VkFFTAxis* axis = &app->localFFTPlan.supportAxes[0][l];
								axis->pushConstants.batch = j;
								for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {
									axis->pushConstants.coordinate = i;

									vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
									vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
									uint32_t dispatchBlock[3];
									if (l == 0)
										dispatchBlock[0] = app->configuration.size[1] / axis->specializationConstants.fftDim;
									else
										dispatchBlock[0] = app->configuration.size[1] / axis->specializationConstants.fftDim / axis->axisBlock[0];

									dispatchBlock[1] = 1;
									dispatchBlock[2] = app->configuration.size[2];
									if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
									dispatchEnchanced(app, commandBuffer, axis, dispatchBlock);

								}
								if (l > 0)
									vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

							}
						}
					}
					for (uint32_t j = 0; j < app->configuration.numberBatches; j++) {
						for (int l = app->localFFTPlan.numAxisUploads[1] - 1; l >= 0; l--) {
							VkFFTAxis* axis = &app->localFFTPlan.axes[1][l];
							axis->pushConstants.batch = j;
							for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {
								axis->pushConstants.coordinate = i;

								vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
								vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
								uint32_t dispatchBlock[3];

								dispatchBlock[0] = app->configuration.size[0] / axis->axisBlock[0] * app->configuration.size[1] / axis->specializationConstants.fftDim;
								dispatchBlock[1] = 1;
								dispatchBlock[2] = app->configuration.size[2];
								if (app->configuration.performR2C == 1) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);
								if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
								dispatchEnchanced(app, commandBuffer, axis, dispatchBlock);
							}
							vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

						}
					}

				}
			}
			//FFT axis 2
			if (app->configuration.FFTdim > 2) {
				if ((app->configuration.FFTdim == 3) && (app->configuration.performConvolution)) {

					if (app->configuration.performR2C == 1) {

						for (int l = app->localFFTPlan.numSupportAxisUploads[1] - 1; l >= 0; l--) {
							VkFFTAxis* axis = &app->localFFTPlan.supportAxes[1][l];
							uint32_t maxCoordinate = ((app->configuration.matrixConvolution > 1) && (l == 0)) ? 1 : app->configuration.coordinateFeatures;
							for (uint32_t i = 0; i < maxCoordinate; i++) {
								axis->pushConstants.coordinate = i;

								axis->pushConstants.batch = ((l == 0) && (app->configuration.matrixConvolution == 1)) ? app->configuration.numberKernels : 0;


								vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
								vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
								uint32_t dispatchBlock[3];
								dispatchBlock[0] = app->configuration.size[1] / axis->axisBlock[0] * app->configuration.size[2] / axis->specializationConstants.fftDim;
								dispatchBlock[1] = 1;
								dispatchBlock[2] = 1;
								dispatchEnchanced(app, commandBuffer, axis, dispatchBlock);

							}
							if (l > 0)
								vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

						}
					}

					for (int l = app->localFFTPlan.numAxisUploads[2] - 1; l >= 0; l--) {

						VkFFTAxis* axis = &app->localFFTPlan.axes[2][l];
						uint32_t maxCoordinate = ((app->configuration.matrixConvolution > 1) && (l == 0)) ? 1 : app->configuration.coordinateFeatures;
						for (uint32_t i = 0; i < maxCoordinate; i++) {
							axis->pushConstants.coordinate = i;
							axis->pushConstants.batch = ((l == 0) && (app->configuration.matrixConvolution == 1)) ? app->configuration.numberKernels : 0;


							vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
							vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
							uint32_t dispatchBlock[3];
							dispatchBlock[0] = app->configuration.size[0] / axis->axisBlock[0] * app->configuration.size[2] / axis->specializationConstants.fftDim;
							dispatchBlock[1] = 1;
							dispatchBlock[2] = app->configuration.size[1];
							if (app->configuration.performR2C == 1) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);
							dispatchEnchanced(app, commandBuffer, axis, dispatchBlock);

						}
						vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

					}
				}
				else {
					if (app->configuration.performR2C == 1) {
						for (uint32_t j = 0; j < app->configuration.numberBatches; j++) {
							for (int l = app->localFFTPlan.numSupportAxisUploads[1] - 1; l >= 0; l--) {
								VkFFTAxis* axis = &app->localFFTPlan.supportAxes[1][l];
								axis->pushConstants.batch = j;
								for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {
									axis->pushConstants.coordinate = i;


									vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
									vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
									uint32_t dispatchBlock[3];
									dispatchBlock[0] = app->configuration.size[1] / axis->axisBlock[0] * app->configuration.size[2] / axis->specializationConstants.fftDim;
									dispatchBlock[1] = 1;
									dispatchBlock[2] = 1;
									dispatchEnchanced(app, commandBuffer, axis, dispatchBlock);

								}
								if (l > 0)
									vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

							}
						}
					}
					for (uint32_t j = 0; j < app->configuration.numberBatches; j++) {
						for (int l = app->localFFTPlan.numAxisUploads[2] - 1; l >= 0; l--) {
							VkFFTAxis* axis = &app->localFFTPlan.axes[2][l];
							axis->pushConstants.batch = j;
							for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {
								axis->pushConstants.coordinate = i;

								vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
								vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
								uint32_t dispatchBlock[3];
								dispatchBlock[0] = app->configuration.size[0] / axis->axisBlock[0] * app->configuration.size[2] / axis->specializationConstants.fftDim;
								dispatchBlock[1] = 1;
								dispatchBlock[2] = app->configuration.size[1];
								if (app->configuration.performR2C == 1) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);
								dispatchEnchanced(app, commandBuffer, axis, dispatchBlock);
							}
							vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

						}
					}

				}

			}
		}
		if (app->configuration.performConvolution) {
			if (app->configuration.FFTdim > 2) {

				//multiple upload ifft leftovers
				if (app->configuration.FFTdim == 3) {
					if (app->configuration.performR2C == 1) {
						for (uint32_t j = 0; j < app->configuration.numberKernels; j++) {
							for (int l = 1; l < app->localFFTPlan_inverse_convolution.numSupportAxisUploads[1]; l++) {
								VkFFTAxis* axis = &app->localFFTPlan_inverse_convolution.supportAxes[1][l];
								uint32_t maxCoordinate = app->configuration.coordinateFeatures;
								for (uint32_t i = 0; i < maxCoordinate; i++) {
									axis->pushConstants.coordinate = i;
									axis->pushConstants.batch = j;

									vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
									vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
									uint32_t dispatchBlock[3];
									dispatchBlock[0] = app->configuration.size[1] / axis->axisBlock[0] * app->configuration.size[2] / axis->specializationConstants.fftDim;
									dispatchBlock[1] = 1;
									dispatchBlock[2] = 1;
									dispatchEnchanced(app, commandBuffer, axis, dispatchBlock);

								}
								if (l > 0)
									vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

							}
						}
					}
					for (uint32_t j = 0; j < app->configuration.numberKernels; j++) {
						for (int l = 1; l < app->localFFTPlan_inverse_convolution.numAxisUploads[2]; l++) {
							VkFFTAxis* axis = &app->localFFTPlan_inverse_convolution.axes[2][l];
							uint32_t maxCoordinate = app->configuration.coordinateFeatures;
							for (uint32_t i = 0; i < maxCoordinate; i++) {
								axis->pushConstants.coordinate = i;
								axis->pushConstants.batch = j;

								vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
								vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
								uint32_t dispatchBlock[3];
								dispatchBlock[0] = app->configuration.size[0] / axis->axisBlock[0] * app->configuration.size[2] / axis->specializationConstants.fftDim;
								dispatchBlock[1] = 1;
								dispatchBlock[2] = app->configuration.size[1];
								if (app->configuration.performR2C == 1) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);
								dispatchEnchanced(app, commandBuffer, axis, dispatchBlock);
							}
							vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

						}
					}
				}
				if (app->configuration.performR2C == 1) {
					for (uint32_t j = 0; j < app->configuration.numberKernels; j++) {
						for (int l = 0; l < app->localFFTPlan_inverse_convolution.numSupportAxisUploads[0]; l++) {
							VkFFTAxis* axis = &app->localFFTPlan_inverse_convolution.supportAxes[0][l];
							axis->pushConstants.batch = j;
							for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {

								axis->pushConstants.coordinate = i;

								vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
								vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
								uint32_t dispatchBlock[3];
								if (l == 0)
									dispatchBlock[0] = app->configuration.size[1] / axis->specializationConstants.fftDim;
								else
									dispatchBlock[0] = app->configuration.size[1] / axis->specializationConstants.fftDim / axis->axisBlock[0];
								dispatchBlock[1] = 1;
								dispatchBlock[2] = app->configuration.size[2];
								if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
								dispatchEnchanced(app, commandBuffer, axis, dispatchBlock);
							}
							if (l < app->localFFTPlan_inverse_convolution.numSupportAxisUploads[0] - 1)
								vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

						}
					}
				}
				for (uint32_t j = 0; j < app->configuration.numberKernels; j++) {
					for (int l = 0; l < app->localFFTPlan_inverse_convolution.numAxisUploads[1]; l++) {
						VkFFTAxis* axis = &app->localFFTPlan_inverse_convolution.axes[1][l];
						axis->pushConstants.batch = j;
						for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {
							axis->pushConstants.coordinate = i;

							vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
							vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
							uint32_t dispatchBlock[3];
							dispatchBlock[0] = app->configuration.size[0] / axis->axisBlock[0] * app->configuration.size[1] / axis->specializationConstants.fftDim;
							dispatchBlock[1] = 1;
							dispatchBlock[2] = app->configuration.size[2];
							if (app->configuration.performR2C == 1) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);
							if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
							dispatchEnchanced(app, commandBuffer, axis, dispatchBlock);

						}
						vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

					}
				}

			}
			if (app->configuration.FFTdim > 1) {
				if (app->configuration.FFTdim == 2) {
					if (app->configuration.performR2C == 1) {
						for (uint32_t j = 0; j < app->configuration.numberKernels; j++) {
							for (int l = 1; l < app->localFFTPlan_inverse_convolution.numSupportAxisUploads[0]; l++) {
								VkFFTAxis* axis = &app->localFFTPlan_inverse_convolution.supportAxes[0][l];
								uint32_t maxCoordinate = app->configuration.coordinateFeatures;
								for (uint32_t i = 0; i < maxCoordinate; i++) {
									axis->pushConstants.coordinate = i;
									axis->pushConstants.batch = j;

									vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
									vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
									uint32_t dispatchBlock[3];
									if (l == 0)
										dispatchBlock[0] = app->configuration.size[1] / axis->specializationConstants.fftDim;
									else
										dispatchBlock[0] = app->configuration.size[1] / axis->specializationConstants.fftDim / axis->axisBlock[0];
									dispatchBlock[1] = 1;
									dispatchBlock[2] = app->configuration.size[2];
									if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
									dispatchEnchanced(app, commandBuffer, axis, dispatchBlock);

								}
								if (l > 0)
									vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

							}
						}

					}
					for (uint32_t j = 0; j < app->configuration.numberKernels; j++) {
						for (int l = 1; l < app->localFFTPlan_inverse_convolution.numAxisUploads[1]; l++) {
							VkFFTAxis* axis = &app->localFFTPlan_inverse_convolution.axes[1][l];
							uint32_t maxCoordinate = app->configuration.coordinateFeatures;
							for (uint32_t i = 0; i < maxCoordinate; i++) {

								axis->pushConstants.coordinate = i;
								axis->pushConstants.batch = j;

								vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
								vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
								uint32_t dispatchBlock[3];
								dispatchBlock[0] = app->configuration.size[0] / axis->axisBlock[0] * app->configuration.size[1] / axis->specializationConstants.fftDim;
								dispatchBlock[1] = 1;
								dispatchBlock[2] = app->configuration.size[2];
								if (app->configuration.performR2C == 1) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);
								if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
								dispatchEnchanced(app, commandBuffer, axis, dispatchBlock);

							}
							vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

						}
					}
				}
				for (uint32_t j = 0; j < app->configuration.numberKernels; j++) {
					for (int l = 0; l < app->localFFTPlan_inverse_convolution.numAxisUploads[0]; l++) {
						VkFFTAxis* axis = &app->localFFTPlan_inverse_convolution.axes[0][l];
						axis->pushConstants.batch = j;
						for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {
							axis->pushConstants.coordinate = i;

							vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
							vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
							uint32_t dispatchBlock[3];
							if (l == 0)
								dispatchBlock[0] = app->configuration.size[0] / axis->specializationConstants.fftDim / axis->axisBlock[1];
							else
								dispatchBlock[0] = app->configuration.size[0] / axis->specializationConstants.fftDim / axis->axisBlock[0];
							dispatchBlock[1] = app->configuration.size[1];
							dispatchBlock[2] = app->configuration.size[2];
							if (app->configuration.performR2C == 1) dispatchBlock[1] = ceil(dispatchBlock[1] / 2.0);
							if (app->configuration.performZeropadding[1]) dispatchBlock[1] = ceil(dispatchBlock[1] / 2.0);
							if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
							dispatchEnchanced(app, commandBuffer, axis, dispatchBlock);
						}
						vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

					}
				}


			}
			if (app->configuration.FFTdim == 1) {
				for (uint32_t j = 0; j < app->configuration.numberKernels; j++) {
					for (int l = 1; l < app->localFFTPlan_inverse_convolution.numAxisUploads[0]; l++) {
						VkFFTAxis* axis = &app->localFFTPlan_inverse_convolution.axes[0][l];
						uint32_t maxCoordinate = app->configuration.coordinateFeatures;
						for (uint32_t i = 0; i < maxCoordinate; i++) {

							axis->pushConstants.coordinate = i;
							axis->pushConstants.batch = j;

							vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
							vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
							uint32_t dispatchBlock[3];
							dispatchBlock[0] = app->configuration.size[0] / axis->axisBlock[0] * app->configuration.size[1] / axis->specializationConstants.fftDim;
							dispatchBlock[1] = 1;
							dispatchBlock[2] = app->configuration.size[2];
							if (app->configuration.performR2C == 1) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);
							if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
							dispatchEnchanced(app, commandBuffer, axis, dispatchBlock);

						}
						vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

					}
				}
			}
		}

		if (app->configuration.inverse) {
			//we start from axis 2 and go back to axis 0
			//FFT axis 2
			if (app->configuration.FFTdim > 2) {
				if (app->configuration.performR2C == 1) {
					for (uint32_t j = 0; j < app->configuration.numberBatches; j++) {
						for (int l = app->localFFTPlan.numSupportAxisUploads[1] - 1; l >= 0; l--) {
							if (!app->configuration.reorderFourStep) l = app->localFFTPlan.numSupportAxisUploads[1] - 1 - l;
							VkFFTAxis* axis = &app->localFFTPlan.supportAxes[1][l];
							axis->pushConstants.batch = j;
							for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {
								axis->pushConstants.coordinate = i;


								vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
								vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
								uint32_t dispatchBlock[3];
								dispatchBlock[0] = app->configuration.size[1] / axis->axisBlock[0] * app->configuration.size[2] / axis->specializationConstants.fftDim;
								dispatchBlock[1] = 1;
								dispatchBlock[2] = 1;
								dispatchEnchanced(app, commandBuffer, axis, dispatchBlock);

							}
							if (!app->configuration.reorderFourStep) l = app->localFFTPlan.numSupportAxisUploads[1] - 1 - l;
							if (l > 0)
								vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

						}
					}
				}

				for (uint32_t j = 0; j < app->configuration.numberBatches; j++) {
					for (int l = app->localFFTPlan.numAxisUploads[2] - 1; l >= 0; l--) {
						if (!app->configuration.reorderFourStep) l = app->localFFTPlan.numAxisUploads[2] - 1 - l;
						VkFFTAxis* axis = &app->localFFTPlan.axes[2][l];
						axis->pushConstants.batch = j;
						for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {
							axis->pushConstants.coordinate = i;

							vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
							vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
							uint32_t dispatchBlock[3];
							dispatchBlock[0] = app->configuration.size[0] / axis->axisBlock[0] * app->configuration.size[2] / axis->specializationConstants.fftDim;
							dispatchBlock[1] = 1;
							dispatchBlock[2] = app->configuration.size[1];
							if (app->configuration.performR2C == 1) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);
							dispatchEnchanced(app, commandBuffer, axis, dispatchBlock);
						}
						vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);
						if (!app->configuration.reorderFourStep) l = app->localFFTPlan.numAxisUploads[2] - 1 - l;
					}
				}

			}
			if (app->configuration.FFTdim > 1) {

				//FFT axis 1
				if (app->configuration.performR2C == 1) {
					for (uint32_t j = 0; j < app->configuration.numberBatches; j++) {
						for (int l = app->localFFTPlan.numSupportAxisUploads[0] - 1; l >= 0; l--) {
							if (!app->configuration.reorderFourStep) l = app->localFFTPlan.numSupportAxisUploads[0] - 1 - l;
							VkFFTAxis* axis = &app->localFFTPlan.supportAxes[0][l];
							axis->pushConstants.batch = j;
							for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {
								axis->pushConstants.coordinate = i;

								vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
								vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
								uint32_t dispatchBlock[3];
								if (l == 0)
									dispatchBlock[0] = app->configuration.size[1] / axis->specializationConstants.fftDim;
								else
									dispatchBlock[0] = app->configuration.size[1] / axis->specializationConstants.fftDim / axis->axisBlock[0];
								dispatchBlock[1] = 1;
								dispatchBlock[2] = app->configuration.size[2];
								if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
								dispatchEnchanced(app, commandBuffer, axis, dispatchBlock);

							}
							if (!app->configuration.reorderFourStep) l = app->localFFTPlan.numSupportAxisUploads[0] - 1 - l;
							if (l > 0)
								vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

						}
					}
				}
				for (uint32_t j = 0; j < app->configuration.numberBatches; j++) {
					for (int l = app->localFFTPlan.numAxisUploads[1] - 1; l >= 0; l--) {
						if (!app->configuration.reorderFourStep) l = app->localFFTPlan.numAxisUploads[1] - 1 - l;
						VkFFTAxis* axis = &app->localFFTPlan.axes[1][l];
						axis->pushConstants.batch = j;
						for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {
							axis->pushConstants.coordinate = i;

							vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
							vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
							uint32_t dispatchBlock[3];
							dispatchBlock[0] = app->configuration.size[0] / axis->axisBlock[0] * app->configuration.size[1] / axis->specializationConstants.fftDim;
							dispatchBlock[1] = 1;
							dispatchBlock[2] = app->configuration.size[2];
							if (app->configuration.performR2C == 1) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);
							if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
							dispatchEnchanced(app, commandBuffer, axis, dispatchBlock);

						}
						if (!app->configuration.reorderFourStep) l = app->localFFTPlan.numAxisUploads[1] - 1 - l;

						vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

					}
				}

			}
			//FFT axis 0
			for (uint32_t j = 0; j < app->configuration.numberBatches; j++) {
				for (int l = app->localFFTPlan.numAxisUploads[0] - 1; l >= 0; l--) {
					if (!app->configuration.reorderFourStep) l = app->localFFTPlan.numAxisUploads[0] - 1 - l;
					VkFFTAxis* axis = &app->localFFTPlan.axes[0][l];
					axis->pushConstants.batch = j;
					uint32_t maxCoordinate = ((app->configuration.matrixConvolution) > 1 && (app->configuration.performConvolution) && (app->configuration.FFTdim == 1)) ? 1 : app->configuration.coordinateFeatures;
					for (uint32_t i = 0; i < maxCoordinate; i++) {
						axis->pushConstants.coordinate = i;

						vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
						vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
						uint32_t dispatchBlock[3];
						if (l == 0)
							dispatchBlock[0] = app->configuration.size[0] / axis->specializationConstants.fftDim / axis->axisBlock[1];
						else
							dispatchBlock[0] = app->configuration.size[0] / axis->specializationConstants.fftDim / axis->axisBlock[0];
						dispatchBlock[1] = app->configuration.size[1];
						dispatchBlock[2] = app->configuration.size[2];
						if (app->configuration.performR2C == 1) dispatchBlock[1] = ceil(dispatchBlock[1] / 2.0);
						if (app->configuration.performZeropadding[1]) dispatchBlock[1] = ceil(dispatchBlock[1] / 2.0);
						if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
						dispatchEnchanced(app, commandBuffer, axis, dispatchBlock);
					}
					if (!app->configuration.reorderFourStep) l = app->localFFTPlan.numAxisUploads[0] - 1 - l;
					vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

				}
			}


		}

	}
	void deleteVulkanFFT(VkFFTApplication* app) {
		for (uint32_t i = 0; i < app->configuration.FFTdim; i++) {
			for (uint32_t j = 0; j < app->localFFTPlan.numAxisUploads[i]; j++)
				deleteAxis(app, &app->localFFTPlan.axes[i][j]);
		}

		for (uint32_t i = 0; i < app->configuration.FFTdim - 1; i++) {
			if (app->configuration.performTranspose[i])
				deleteTranspose(app, &app->localFFTPlan.transpose[i]);
			else {
				for (uint32_t j = 0; j < app->localFFTPlan.numSupportAxisUploads[i]; j++)
					deleteAxis(app, &app->localFFTPlan.supportAxes[i][j]);
			}
		}
		if (app->configuration.performConvolution) {
			for (uint32_t i = 0; i < app->configuration.FFTdim; i++) {
				for (uint32_t j = 0; j < app->localFFTPlan_inverse_convolution.numAxisUploads[i]; j++)
					deleteAxis(app, &app->localFFTPlan_inverse_convolution.axes[i][j]);
			}
			for (uint32_t i = 0; i < app->configuration.FFTdim - 1; i++) {
				if (app->configuration.performTranspose[i])
					deleteTranspose(app, &app->localFFTPlan_inverse_convolution.transpose[i]);
				else {
					for (uint32_t j = 0; j < app->localFFTPlan_inverse_convolution.numSupportAxisUploads[i]; j++)
						deleteAxis(app, &app->localFFTPlan_inverse_convolution.supportAxes[i][j]);
				}
			}
		}
	}
#ifdef __cplusplus
}
#endif