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
#ifndef VKFFT_DELETEPLAN_H
#define VKFFT_DELETEPLAN_H
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"

static inline void deleteAxis(VkFFTApplication* app, VkFFTAxis* axis, int isInverseBluesteinAxes) {
	if (axis->specializationConstants.numRaderPrimes && (!isInverseBluesteinAxes)) {
		free(axis->specializationConstants.raderContainer);
		axis->specializationConstants.raderContainer = 0;
		axis->specializationConstants.numRaderPrimes = 0;
	}
#if(VKFFT_BACKEND==0)
	if ((app->configuration.useLUT == 1) && (!axis->referenceLUT)) {
		if (axis->bufferLUT != 0) {
			vkDestroyBuffer(app->configuration.device[0], axis->bufferLUT, 0);
			axis->bufferLUT = 0;
		}
		if (axis->bufferLUTDeviceMemory != 0) {
			vkFreeMemory(app->configuration.device[0], axis->bufferLUTDeviceMemory, 0);
			axis->bufferLUTDeviceMemory = 0;
		}
	}
	if (axis->descriptorPool != 0) {
		vkDestroyDescriptorPool(app->configuration.device[0], axis->descriptorPool, 0);
		axis->descriptorPool = 0;
	}
	if (axis->descriptorSetLayout != 0) {
		vkDestroyDescriptorSetLayout(app->configuration.device[0], axis->descriptorSetLayout, 0);
		axis->descriptorSetLayout = 0;
	}
	if (axis->pipelineLayout != 0) {
		vkDestroyPipelineLayout(app->configuration.device[0], axis->pipelineLayout, 0);
		axis->pipelineLayout = 0;
	}
	if (axis->pipeline != 0) {
		vkDestroyPipeline(app->configuration.device[0], axis->pipeline, 0);
		axis->pipeline = 0;
	}
#elif(VKFFT_BACKEND==1)
	CUresult res = CUDA_SUCCESS;
	cudaError_t res_t = cudaSuccess;
	if ((app->configuration.useLUT == 1) && (!axis->referenceLUT) && (axis->bufferLUT != 0)) {
		res_t = cudaFree(axis->bufferLUT);
		if (res_t == cudaSuccess) axis->bufferLUT = 0;
	}
	if (axis->VkFFTModule != 0) {
		res = cuModuleUnload(axis->VkFFTModule);
		if (res == CUDA_SUCCESS) axis->VkFFTModule = 0;
	}
#elif(VKFFT_BACKEND==2)
	hipError_t res = hipSuccess;
	if ((app->configuration.useLUT == 1) && (!axis->referenceLUT) && (axis->bufferLUT != 0)) {
		res = hipFree(axis->bufferLUT);
		if (res == hipSuccess) axis->bufferLUT = 0;
	}
	if (axis->VkFFTModule != 0) {
		res = hipModuleUnload(axis->VkFFTModule);
		if (res == hipSuccess) axis->VkFFTModule = 0;
	}
#elif(VKFFT_BACKEND==3)
	cl_int res = 0;
	if ((app->configuration.useLUT == 1) && (!axis->referenceLUT) && (axis->bufferLUT != 0)) {
		res = clReleaseMemObject(axis->bufferLUT);
		if (res == 0) axis->bufferLUT = 0;
	}
	if (axis->program != 0) {
		res = clReleaseProgram(axis->program);
		if (res == 0) axis->program = 0;
	}
	if (axis->kernel != 0) {
		res = clReleaseKernel(axis->kernel);
		if (res == 0) axis->kernel = 0;
	}
#elif(VKFFT_BACKEND==4)
	ze_result_t res = ZE_RESULT_SUCCESS;
	if ((app->configuration.useLUT == 1) && (!axis->referenceLUT) && (axis->bufferLUT != 0)) {
		res = zeMemFree(app->configuration.context[0], axis->bufferLUT);
		if (res == ZE_RESULT_SUCCESS) axis->bufferLUT = 0;
	}
	if (axis->VkFFTKernel != 0) {
		res = zeKernelDestroy(axis->VkFFTKernel);
		if (res == ZE_RESULT_SUCCESS)axis->VkFFTKernel = 0;
	}
	if (axis->VkFFTModule != 0) {
		res = zeModuleDestroy(axis->VkFFTModule);
		if (res == ZE_RESULT_SUCCESS)axis->VkFFTModule = 0;
	}
#elif(VKFFT_BACKEND==5)
	if (axis->pushConstants.dataUintBuffer) {
		axis->pushConstants.dataUintBuffer->release();
		axis->pushConstants.dataUintBuffer = 0;
	}
	if ((app->configuration.useLUT == 1) && (!axis->referenceLUT) && (axis->bufferLUT != 0)) {
		((MTL::Buffer*)axis->bufferLUT)->release();
		//free(axis->bufferLUT);
		axis->bufferLUT = 0;
	}
	if (axis->pipeline != 0) {
		axis->pipeline->release();
		//free(axis->pipeline);
		axis->pipeline = 0;
	}
	if (axis->library != 0) {
		axis->library->release();
		//free(axis->library);
		axis->library = 0;
	}
#endif
	if (app->configuration.saveApplicationToString) {
		if (axis->binary != 0) {
			free(axis->binary);
			axis->binary = 0;
		}
	}
}

#endif
