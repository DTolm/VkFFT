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
#ifndef VKFFT_DISPATCHPLAN_H
#define VKFFT_DISPATCHPLAN_H
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"

static inline VkFFTResult VkFFT_DispatchPlan(VkFFTApplication* app, VkFFTAxis* axis, uint64_t* dispatchBlock) {
	VkFFTResult resFFT = VKFFT_SUCCESS;
	if (axis->specializationConstants.swapComputeWorkGroupID == 1) {
		uint64_t temp = dispatchBlock[0];
		dispatchBlock[0] = dispatchBlock[1];
		dispatchBlock[1] = temp;
	}
	if (axis->specializationConstants.swapComputeWorkGroupID == 2) {
		uint64_t temp = dispatchBlock[0];
		dispatchBlock[0] = dispatchBlock[2];
		dispatchBlock[2] = temp;
	}
	uint64_t blockNumber[3] = { (uint64_t)ceil(dispatchBlock[0] / (double)app->configuration.maxComputeWorkGroupCount[0]),(uint64_t)ceil(dispatchBlock[1] / (double)app->configuration.maxComputeWorkGroupCount[1]),(uint64_t)ceil(dispatchBlock[2] / (double)app->configuration.maxComputeWorkGroupCount[2]) };
	uint64_t blockSize[3] = { (uint64_t)ceil(dispatchBlock[0] / (double)blockNumber[0]), (uint64_t)ceil(dispatchBlock[1] / (double)blockNumber[1]), (uint64_t)ceil(dispatchBlock[2] / (double)blockNumber[2]) };
	uint64_t lastBlockSize[3] = { blockSize[0],blockSize[1],blockSize[2] };
	uint64_t dispatchSize[3] = { 1,1,1 };
	if (blockNumber[0] == 0) blockNumber[0] = 1;
	if (blockNumber[1] == 0) blockNumber[1] = 1;
	if (blockNumber[2] == 0) blockNumber[2] = 1;
	if ((blockNumber[0] > 1) && (blockNumber[0] * blockSize[0] != dispatchBlock[0])) {
		lastBlockSize[0] = dispatchBlock[0] % blockSize[0];
	}
	if ((blockNumber[1] > 1) && (blockNumber[1] * blockSize[1] != dispatchBlock[1])) {
		lastBlockSize[1] = dispatchBlock[1] % blockSize[1];
	}
	if ((blockNumber[2] > 1) && (blockNumber[2] * blockSize[2] != dispatchBlock[2])) {
		lastBlockSize[2] = dispatchBlock[2] % blockSize[2];
	}
	if (app->configuration.specifyOffsetsAtLaunch) {
		axis->updatePushConstants = 1;
	}
	//printf("%" PRIu64 " %" PRIu64 " %" PRIu64 "\n", dispatchBlock[0], dispatchBlock[1], dispatchBlock[2]);
	//printf("%" PRIu64 " %" PRIu64 " %" PRIu64 "\n", blockNumber[0], blockNumber[1], blockNumber[2]);
	for (uint64_t i = 0; i < 3; i++)
		if (blockNumber[i] == 1) blockSize[i] = dispatchBlock[i];
	for (uint64_t i = 0; i < blockNumber[0]; i++) {
		for (uint64_t j = 0; j < blockNumber[1]; j++) {
			for (uint64_t k = 0; k < blockNumber[2]; k++) {
				if (axis->pushConstants.workGroupShift[0] != i * blockSize[0]) {
					axis->pushConstants.workGroupShift[0] = i * blockSize[0];
					axis->updatePushConstants = 1;
				}
				if (axis->pushConstants.workGroupShift[1] != j * blockSize[1]) {
					axis->pushConstants.workGroupShift[1] = j * blockSize[1];
					axis->updatePushConstants = 1;
				}
				if (axis->pushConstants.workGroupShift[2] != k * blockSize[2]) {
					axis->pushConstants.workGroupShift[2] = k * blockSize[2];
					axis->updatePushConstants = 1;
				}
				if (axis->updatePushConstants) {
					if (app->configuration.useUint64) {
						uint64_t offset = 0;
						uint64_t temp = 0;
						if (axis->specializationConstants.performWorkGroupShift[0]) {
							memcpy(&axis->pushConstants.data[offset], &axis->pushConstants.workGroupShift[0], sizeof(uint64_t));
							offset+=sizeof(uint64_t);
						}
						if (axis->specializationConstants.performWorkGroupShift[1]) {
							memcpy(&axis->pushConstants.data[offset], &axis->pushConstants.workGroupShift[1], sizeof(uint64_t));
							offset += sizeof(uint64_t);
						}
						if (axis->specializationConstants.performWorkGroupShift[2]) {
							memcpy(&axis->pushConstants.data[offset], &axis->pushConstants.workGroupShift[2], sizeof(uint64_t));
							offset += sizeof(uint64_t);
						}
						if (axis->specializationConstants.performPostCompilationInputOffset) {
							temp = axis->specializationConstants.inputOffset.data.i / axis->specializationConstants.inputNumberByteSize;
							memcpy(&axis->pushConstants.data[offset], &temp, sizeof(uint64_t));
							offset += sizeof(uint64_t);
						}
						if (axis->specializationConstants.performPostCompilationOutputOffset) {
							temp = axis->specializationConstants.outputOffset.data.i / axis->specializationConstants.outputNumberByteSize;
							memcpy(&axis->pushConstants.data[offset], &temp, sizeof(uint64_t));
							offset += sizeof(uint64_t);
						}
						if (axis->specializationConstants.performPostCompilationKernelOffset) {
							if (axis->specializationConstants.kernelNumberByteSize != 0)
								temp = axis->specializationConstants.kernelOffset.data.i / axis->specializationConstants.kernelNumberByteSize;
							else
								temp = 0;
							memcpy(&axis->pushConstants.data[offset], &temp, sizeof(uint64_t));
							offset += sizeof(uint64_t);
						}
					}
					else {
						uint64_t offset = 0;
						uint32_t temp = 0;
						if (axis->specializationConstants.performWorkGroupShift[0]) {
							temp = (uint32_t)axis->pushConstants.workGroupShift[0];
							memcpy(&axis->pushConstants.data[offset], &temp, sizeof(uint32_t));
							offset += sizeof(uint32_t);
						}
						if (axis->specializationConstants.performWorkGroupShift[1]) {
							temp = (uint32_t)axis->pushConstants.workGroupShift[1];
							memcpy(&axis->pushConstants.data[offset], &temp, sizeof(uint32_t));
							offset += sizeof(uint32_t);
						}
						if (axis->specializationConstants.performWorkGroupShift[2]) {
							temp = (uint32_t)axis->pushConstants.workGroupShift[2];
							memcpy(&axis->pushConstants.data[offset], &temp, sizeof(uint32_t));
							offset += sizeof(uint32_t);
						}
						if (axis->specializationConstants.performPostCompilationInputOffset) {
							temp = (uint32_t)(axis->specializationConstants.inputOffset.data.i / axis->specializationConstants.inputNumberByteSize);
							memcpy(&axis->pushConstants.data[offset], &temp, sizeof(uint32_t));
							offset += sizeof(uint32_t);
						}
						if (axis->specializationConstants.performPostCompilationOutputOffset) {
							temp = (uint32_t)(axis->specializationConstants.outputOffset.data.i / axis->specializationConstants.outputNumberByteSize);
							memcpy(&axis->pushConstants.data[offset], &temp, sizeof(uint32_t));
							offset += sizeof(uint32_t);
						}
						if (axis->specializationConstants.performPostCompilationKernelOffset) {
							if (axis->specializationConstants.kernelNumberByteSize != 0)
								temp = (uint32_t)(axis->specializationConstants.kernelOffset.data.i / axis->specializationConstants.kernelNumberByteSize);
							else
								temp = 0;
							memcpy(&axis->pushConstants.data[offset], &temp, sizeof(uint32_t));
							offset += sizeof(uint32_t);
						}
					}
				}
				dispatchSize[0] = (i == blockNumber[0] - 1) ? lastBlockSize[0] : blockSize[0];
				dispatchSize[1] = (j == blockNumber[1] - 1) ? lastBlockSize[1] : blockSize[1];
				dispatchSize[2] = (k == blockNumber[2] - 1) ? lastBlockSize[2] : blockSize[2];
#if(VKFFT_BACKEND==0)
				if (axis->pushConstants.structSize > 0) {
					vkCmdPushConstants(app->configuration.commandBuffer[0], axis->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)axis->pushConstants.structSize, axis->pushConstants.data);
				}
				vkCmdDispatch(app->configuration.commandBuffer[0], (uint32_t)dispatchSize[0], (uint32_t)dispatchSize[1], (uint32_t)dispatchSize[2]);
#elif(VKFFT_BACKEND==1)
				void* args[10];
				CUresult result = CUDA_SUCCESS;
				args[0] = axis->inputBuffer;
				args[1] = axis->outputBuffer;
				uint64_t args_id = 2;
				if (axis->specializationConstants.convolutionStep) {
					args[args_id] = app->configuration.kernel;
					args_id++;
				}
				if (axis->specializationConstants.LUT) {
					args[args_id] = &axis->bufferLUT;
					args_id++;
				}
				if (axis->specializationConstants.raderUintLUT) {
					args[args_id] = &axis->bufferRaderUintLUT;
					args_id++;
				}
				if (axis->specializationConstants.useBluesteinFFT && axis->specializationConstants.BluesteinConvolutionStep) {
					if (axis->specializationConstants.inverseBluestein)
						args[args_id] = &app->bufferBluesteinIFFT[axis->specializationConstants.axis_id];
					else
						args[args_id] = &app->bufferBluesteinFFT[axis->specializationConstants.axis_id];
					args_id++;
				}
				if (axis->specializationConstants.useBluesteinFFT && (axis->specializationConstants.BluesteinPreMultiplication || axis->specializationConstants.BluesteinPostMultiplication)) {
					args[args_id] = &app->bufferBluestein[axis->specializationConstants.axis_id];
					args_id++;
				}
				//args[args_id] = &axis->pushConstants;
				if (axis->updatePushConstants) {
					axis->updatePushConstants = 0;
					if (axis->pushConstants.structSize > 0) {
						result = cuMemcpyHtoD(axis->consts_addr, axis->pushConstants.data, axis->pushConstants.structSize);
						if (result != CUDA_SUCCESS) {
							printf("cuMemcpyHtoD error: %d\n", result);
							return VKFFT_ERROR_FAILED_TO_COPY;
						}
					}
				}
				if (app->configuration.num_streams >= 1) {
					result = cuLaunchKernel(axis->VkFFTKernel,
						(unsigned int)dispatchSize[0], (unsigned int)dispatchSize[1], (unsigned int)dispatchSize[2],     // grid dim
						(unsigned int)axis->specializationConstants.localSize[0].data.i, (unsigned int)axis->specializationConstants.localSize[1].data.i, (unsigned int)axis->specializationConstants.localSize[2].data.i,   // block dim
						(unsigned int)axis->specializationConstants.usedSharedMemory.data.i, app->configuration.stream[app->configuration.streamID],             // shared mem and stream
						args, 0);
				}
				else {
					result = cuLaunchKernel(axis->VkFFTKernel,
						(unsigned int)dispatchSize[0], (unsigned int)dispatchSize[1], (unsigned int)dispatchSize[2],     // grid dim
						(unsigned int)axis->specializationConstants.localSize[0].data.i, (unsigned int)axis->specializationConstants.localSize[1].data.i, (unsigned int)axis->specializationConstants.localSize[2].data.i,   // block dim
						(unsigned int)axis->specializationConstants.usedSharedMemory.data.i, 0,             // shared mem and stream
						args, 0);
				}
				if (result != CUDA_SUCCESS) {
					printf("cuLaunchKernel error: %d, %" PRIu64 " %" PRIu64 " %" PRIu64 " - %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", result, dispatchSize[0], dispatchSize[1], dispatchSize[2], axis->specializationConstants.localSize[0].data.i, axis->specializationConstants.localSize[1].data.i, axis->specializationConstants.localSize[2].data.i);
					return VKFFT_ERROR_FAILED_TO_LAUNCH_KERNEL;
				}
				if (app->configuration.num_streams > 1) {
					app->configuration.streamID = app->configuration.streamCounter % app->configuration.num_streams;
					if (app->configuration.streamCounter == 0) {
						cudaError_t res2 = cudaEventRecord(app->configuration.stream_event[app->configuration.streamID], app->configuration.stream[app->configuration.streamID]);
						if (res2 != cudaSuccess) return VKFFT_ERROR_FAILED_TO_EVENT_RECORD;
					}
					app->configuration.streamCounter++;
				}
#elif(VKFFT_BACKEND==2)
				hipError_t result = hipSuccess;
				void* args[10];
				args[0] = axis->inputBuffer;
				args[1] = axis->outputBuffer;
				uint64_t args_id = 2;
				if (axis->specializationConstants.convolutionStep) {
					args[args_id] = app->configuration.kernel;
					args_id++;
				}
				if (axis->specializationConstants.LUT) {
					args[args_id] = &axis->bufferLUT;
					args_id++;
				}
				if (axis->specializationConstants.raderUintLUT) {
					args[args_id] = &axis->bufferRaderUintLUT;
					args_id++;
				}
				if (axis->specializationConstants.useBluesteinFFT && axis->specializationConstants.BluesteinConvolutionStep) {
					if (axis->specializationConstants.inverseBluestein)
						args[args_id] = &app->bufferBluesteinIFFT[axis->specializationConstants.axis_id];
					else
						args[args_id] = &app->bufferBluesteinFFT[axis->specializationConstants.axis_id];
					args_id++;
				}
				if (axis->specializationConstants.useBluesteinFFT && (axis->specializationConstants.BluesteinPreMultiplication || axis->specializationConstants.BluesteinPostMultiplication)) {
					args[args_id] = &app->bufferBluestein[axis->specializationConstants.axis_id];
					args_id++;
				}
				//args[args_id] = &axis->pushConstants;
				if (axis->updatePushConstants) {
					axis->updatePushConstants = 0;
					if (axis->pushConstants.structSize > 0) {
						result = hipMemcpyHtoD(axis->consts_addr, axis->pushConstants.data, axis->pushConstants.structSize);
						if (result != hipSuccess) {
							printf("hipMemcpyHtoD error: %d\n", result);
							return VKFFT_ERROR_FAILED_TO_COPY;
						}
					}
				}
				//printf("%" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n",maxBlockSize[0], maxBlockSize[1], maxBlockSize[2], axis->specializationConstants.localSize[0], axis->specializationConstants.localSize[1], axis->specializationConstants.localSize[2]);
				if (app->configuration.num_streams >= 1) {
					result = hipModuleLaunchKernel(axis->VkFFTKernel,
						(unsigned int)dispatchSize[0], (unsigned int)dispatchSize[1], (unsigned int)dispatchSize[2],     // grid dim
						(unsigned int)axis->specializationConstants.localSize[0].data.i, (unsigned int)axis->specializationConstants.localSize[1].data.i, (unsigned int)axis->specializationConstants.localSize[2].data.i,   // block dim
						(unsigned int)axis->specializationConstants.usedSharedMemory.data.i, app->configuration.stream[app->configuration.streamID],             // shared mem and stream
						args, 0);
				}
				else {
					result = hipModuleLaunchKernel(axis->VkFFTKernel,
						(unsigned int)dispatchSize[0], (unsigned int)dispatchSize[1], (unsigned int)dispatchSize[2],     // grid dim
						(unsigned int)axis->specializationConstants.localSize[0].data.i, (unsigned int)axis->specializationConstants.localSize[1].data.i, (unsigned int)axis->specializationConstants.localSize[2].data.i,   // block dim
						(unsigned int)axis->specializationConstants.usedSharedMemory.data.i, 0,             // shared mem and stream
						args, 0);
				}
				if (result != hipSuccess) {
					printf("hipModuleLaunchKernel error: %d, %" PRIu64 " %" PRIu64 " %" PRIu64 " - %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", result, dispatchSize[0], dispatchSize[1], dispatchSize[2], axis->specializationConstants.localSize[0].data.i, axis->specializationConstants.localSize[1].data.i, axis->specializationConstants.localSize[2].data.i);
					return VKFFT_ERROR_FAILED_TO_LAUNCH_KERNEL;
				}
				if (app->configuration.num_streams > 1) {
					app->configuration.streamID = app->configuration.streamCounter % app->configuration.num_streams;
					if (app->configuration.streamCounter == 0) {
						result = hipEventRecord(app->configuration.stream_event[app->configuration.streamID], app->configuration.stream[app->configuration.streamID]);
						if (result != hipSuccess) return VKFFT_ERROR_FAILED_TO_EVENT_RECORD;
					}
					app->configuration.streamCounter++;
				}
#elif(VKFFT_BACKEND==3)
				cl_int result = CL_SUCCESS;
				void* args[10];
				args[0] = axis->inputBuffer;
				result = clSetKernelArg(axis->kernel, 0, sizeof(cl_mem), args[0]);
				if (result != CL_SUCCESS) {
					return VKFFT_ERROR_FAILED_TO_SET_KERNEL_ARG;
				}
				args[1] = axis->outputBuffer;
				result = clSetKernelArg(axis->kernel, 1, sizeof(cl_mem), args[1]);
				if (result != CL_SUCCESS) {
					return VKFFT_ERROR_FAILED_TO_SET_KERNEL_ARG;
				}
				uint64_t args_id = 2;
				if (axis->specializationConstants.convolutionStep) {
					args[args_id] = app->configuration.kernel;
					result = clSetKernelArg(axis->kernel, (cl_uint)args_id, sizeof(cl_mem), args[args_id]);
					if (result != CL_SUCCESS) {
						return VKFFT_ERROR_FAILED_TO_SET_KERNEL_ARG;
					}
					args_id++;
				}
				if (axis->specializationConstants.LUT) {
					args[args_id] = &axis->bufferLUT;
					result = clSetKernelArg(axis->kernel, (cl_uint)args_id, sizeof(cl_mem), args[args_id]);
					if (result != CL_SUCCESS) {
						return VKFFT_ERROR_FAILED_TO_SET_KERNEL_ARG;
					}
					args_id++;
				}
				if (axis->specializationConstants.raderUintLUT) {
					args[args_id] = &axis->bufferRaderUintLUT;
					result = clSetKernelArg(axis->kernel, (cl_uint)args_id, sizeof(cl_mem), args[args_id]);
					if (result != CL_SUCCESS) {
						return VKFFT_ERROR_FAILED_TO_SET_KERNEL_ARG;
					}
					args_id++;
				}
				if (axis->specializationConstants.useBluesteinFFT && axis->specializationConstants.BluesteinConvolutionStep) {
					if (axis->specializationConstants.inverseBluestein)
						args[args_id] = &app->bufferBluesteinIFFT[axis->specializationConstants.axis_id];
					else
						args[args_id] = &app->bufferBluesteinFFT[axis->specializationConstants.axis_id];
					result = clSetKernelArg(axis->kernel, (cl_uint)args_id, sizeof(cl_mem), args[args_id]);
					if (result != CL_SUCCESS) {
						return VKFFT_ERROR_FAILED_TO_SET_KERNEL_ARG;
					}
					args_id++;
				}
				if (axis->specializationConstants.useBluesteinFFT && (axis->specializationConstants.BluesteinPreMultiplication || axis->specializationConstants.BluesteinPostMultiplication)) {
					args[args_id] = &app->bufferBluestein[axis->specializationConstants.axis_id];
					result = clSetKernelArg(axis->kernel, (cl_uint)args_id, sizeof(cl_mem), args[args_id]);
					if (result != CL_SUCCESS) {
						return VKFFT_ERROR_FAILED_TO_SET_KERNEL_ARG;
					}
					args_id++;
				}

				if (axis->pushConstants.structSize > 0) {
					result = clSetKernelArg(axis->kernel, (cl_uint)args_id, axis->pushConstants.structSize, axis->pushConstants.data);
					if (result != CL_SUCCESS) {
						return VKFFT_ERROR_FAILED_TO_SET_KERNEL_ARG;
					}
					args_id++;
				}
				size_t local_work_size[3] = { (size_t)axis->specializationConstants.localSize[0].data.i , (size_t)axis->specializationConstants.localSize[1].data.i ,(size_t)axis->specializationConstants.localSize[2].data.i };
				size_t global_work_size[3] = { (size_t)dispatchSize[0] * local_work_size[0] , (size_t)dispatchSize[1] * local_work_size[1] ,(size_t)dispatchSize[2] * local_work_size[2] };
				result = clEnqueueNDRangeKernel(app->configuration.commandQueue[0], axis->kernel, 3, 0, global_work_size, local_work_size, 0, 0, 0);
				//printf("%" PRIu64 " %" PRIu64 " %" PRIu64 " - %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", maxBlockSize[0], maxBlockSize[1], maxBlockSize[2], axis->specializationConstants.localSize[0], axis->specializationConstants.localSize[1], axis->specializationConstants.localSize[2]);

				if (result != CL_SUCCESS) {
					return VKFFT_ERROR_FAILED_TO_LAUNCH_KERNEL;
				}
#elif(VKFFT_BACKEND==4)
				ze_result_t result = ZE_RESULT_SUCCESS;
				void* args[10];
				args[0] = axis->inputBuffer;
				result = zeKernelSetArgumentValue(axis->VkFFTKernel, 0, sizeof(void*), args[0]);
				if (result != ZE_RESULT_SUCCESS) {
					return VKFFT_ERROR_FAILED_TO_SET_KERNEL_ARG;
				}
				args[1] = axis->outputBuffer;
				result = zeKernelSetArgumentValue(axis->VkFFTKernel, 1, sizeof(void*), args[1]);
				if (result != ZE_RESULT_SUCCESS) {
					return VKFFT_ERROR_FAILED_TO_SET_KERNEL_ARG;
				}
				uint64_t args_id = 2;
				if (axis->specializationConstants.convolutionStep) {
					args[args_id] = app->configuration.kernel;
					result = zeKernelSetArgumentValue(axis->VkFFTKernel, (uint32_t)args_id, sizeof(void*), args[args_id]);
					if (result != ZE_RESULT_SUCCESS) {
						return VKFFT_ERROR_FAILED_TO_SET_KERNEL_ARG;
					}
					args_id++;
				}
				if (axis->specializationConstants.LUT) {
					args[args_id] = &axis->bufferLUT;
					result = zeKernelSetArgumentValue(axis->VkFFTKernel, (uint32_t)args_id, sizeof(void*), args[args_id]);
					if (result != ZE_RESULT_SUCCESS) {
						return VKFFT_ERROR_FAILED_TO_SET_KERNEL_ARG;
					}
					args_id++;
				}
				if (axis->specializationConstants.raderUintLUT) {
					args[args_id] = &axis->bufferRaderUintLUT;
					result = zeKernelSetArgumentValue(axis->VkFFTKernel, (uint32_t)args_id, sizeof(void*), args[args_id]);
					if (result != ZE_RESULT_SUCCESS) {
						return VKFFT_ERROR_FAILED_TO_SET_KERNEL_ARG;
					}
					args_id++;
				}
				if (axis->specializationConstants.useBluesteinFFT && axis->specializationConstants.BluesteinConvolutionStep) {
					if (axis->specializationConstants.inverseBluestein)
						args[args_id] = &app->bufferBluesteinIFFT[axis->specializationConstants.axis_id];
					else
						args[args_id] = &app->bufferBluesteinFFT[axis->specializationConstants.axis_id];
					result = zeKernelSetArgumentValue(axis->VkFFTKernel, (uint32_t)args_id, sizeof(void*), args[args_id]);
					if (result != ZE_RESULT_SUCCESS) {
						return VKFFT_ERROR_FAILED_TO_SET_KERNEL_ARG;
					}
					args_id++;
				}
				if (axis->specializationConstants.useBluesteinFFT && (axis->specializationConstants.BluesteinPreMultiplication || axis->specializationConstants.BluesteinPostMultiplication)) {
					args[args_id] = &app->bufferBluestein[axis->specializationConstants.axis_id];
					result = zeKernelSetArgumentValue(axis->VkFFTKernel, (uint32_t)args_id, sizeof(void*), args[args_id]);
					if (result != ZE_RESULT_SUCCESS) {
						return VKFFT_ERROR_FAILED_TO_SET_KERNEL_ARG;
					}
					args_id++;
				}

				if (axis->pushConstants.structSize > 0) {
					result = zeKernelSetArgumentValue(axis->VkFFTKernel, (uint32_t)args_id, axis->pushConstants.structSize, axis->pushConstants.data);

					if (result != ZE_RESULT_SUCCESS) {
						return VKFFT_ERROR_FAILED_TO_SET_KERNEL_ARG;
					}
					args_id++;
				}
				size_t local_work_size[3] = { (size_t)axis->specializationConstants.localSize[0].data.i , (size_t)axis->specializationConstants.localSize[1].data.i ,(size_t)axis->specializationConstants.localSize[2].data.i };
				ze_group_count_t launchArgs = { (uint32_t)dispatchSize[0], (uint32_t)dispatchSize[1],(uint32_t)dispatchSize[2] };
				result = zeCommandListAppendLaunchKernel(app->configuration.commandList[0], axis->VkFFTKernel, &launchArgs, 0, 0, 0);
				//printf("%" PRIu64 " %" PRIu64 " %" PRIu64 " - %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", maxBlockSize[0], maxBlockSize[1], maxBlockSize[2], axis->specializationConstants.localSize[0], axis->specializationConstants.localSize[1], axis->specializationConstants.localSize[2]);

				if (result != ZE_RESULT_SUCCESS) {
					return VKFFT_ERROR_FAILED_TO_LAUNCH_KERNEL;
				}
#elif(VKFFT_BACKEND==5)
				app->configuration.commandEncoder->setComputePipelineState(axis->pipeline);
				void* args[10];
				app->configuration.commandEncoder->setBuffer(axis->inputBuffer[0], 0, 0);
				app->configuration.commandEncoder->setBuffer(axis->outputBuffer[0], 0, 1);
				app->configuration.commandEncoder->setThreadgroupMemoryLength((uint64_t)ceil(axis->specializationConstants.usedSharedMemory.data.i / 16.0) * 16, 0);

				uint64_t args_id = 2;
				if (axis->specializationConstants.convolutionStep) {
					app->configuration.commandEncoder->setBuffer(app->configuration.kernel[0], 0, args_id);
					args_id++;
				}
				if (axis->specializationConstants.LUT) {
					app->configuration.commandEncoder->setBuffer(axis->bufferLUT, 0, args_id);
					args_id++;
				}
				if (axis->specializationConstants.raderUintLUT) {
					app->configuration.commandEncoder->setBuffer(axis->bufferRaderUintLUT, 0, args_id);
					args_id++;
				}
				if (axis->specializationConstants.useBluesteinFFT && axis->specializationConstants.BluesteinConvolutionStep) {
					if (axis->specializationConstants.inverseBluestein)
						app->configuration.commandEncoder->setBuffer(app->bufferBluesteinIFFT[axis->specializationConstants.axis_id], 0, args_id);
					else
						app->configuration.commandEncoder->setBuffer(app->bufferBluesteinFFT[axis->specializationConstants.axis_id], 0, args_id);
					args_id++;
				}
				if (axis->specializationConstants.useBluesteinFFT && (axis->specializationConstants.BluesteinPreMultiplication || axis->specializationConstants.BluesteinPostMultiplication)) {
					app->configuration.commandEncoder->setBuffer(app->bufferBluestein[axis->specializationConstants.axis_id], 0, args_id);
					args_id++;
				}
				//args[args_id] = &axis->pushConstants;
				if (axis->pushConstants.structSize > 0) {
					if (!axis->pushConstants.dataUintBuffer) {
						axis->pushConstants.dataUintBuffer = app->configuration.device->newBuffer(axis->pushConstants.structSize, MTL::ResourceStorageModeShared);
						memcpy(axis->pushConstants.dataUintBuffer->contents(), axis->pushConstants.data, axis->pushConstants.structSize);
						axis->updatePushConstants = 0;
					}
					else if (axis->updatePushConstants) {
						memcpy(axis->pushConstants.dataUintBuffer->contents(), axis->pushConstants.data, axis->pushConstants.structSize);
						axis->updatePushConstants = 0;
					}
					app->configuration.commandEncoder->setBuffer(axis->pushConstants.dataUintBuffer, 0, args_id);
					args_id++;
				}
				MTL::Size threadsPerGrid = { dispatchSize[0] * axis->specializationConstants.localSize[0].data.i , dispatchSize[1] * axis->specializationConstants.localSize[1].data.i ,dispatchSize[2] * axis->specializationConstants.localSize[2].data.i };
				MTL::Size threadsPerThreadgroup = { (NS::UInteger) axis->specializationConstants.localSize[0].data.i, (NS::UInteger) axis->specializationConstants.localSize[1].data.i, (NS::UInteger) axis->specializationConstants.localSize[2].data.i };

				app->configuration.commandEncoder->dispatchThreads(threadsPerGrid, threadsPerThreadgroup);

#endif
			}
		}
	}
	return resFFT;
}

#endif
