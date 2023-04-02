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
#include "vkFFT_Structs/vkFFT_Structs.h"

static inline VkFFTResult VkFFT_LaunchKernel(VkFFTApplication* app, VkFFTPlan* plan, uint64_t* dispatchBlock) {
	VkFFTResult resFFT = VKFFT_SUCCESS;
	uint64_t maxBlockSize[3] = { (uint64_t)pow(2,(uint64_t)log2(app->configuration.maxComputeWorkGroupCount[0])),(uint64_t)pow(2,(uint64_t)log2(app->configuration.maxComputeWorkGroupCount[1])),(uint64_t)pow(2,(uint64_t)log2(app->configuration.maxComputeWorkGroupCount[2])) };
	uint64_t blockNumber[3] = { (uint64_t)ceil(dispatchBlock[0] / (double)maxBlockSize[0]),(uint64_t)ceil(dispatchBlock[1] / (double)maxBlockSize[1]),(uint64_t)ceil(dispatchBlock[2] / (double)maxBlockSize[2]) };
	if (blockNumber[0] == 0) blockNumber[0] = 1;
	if (blockNumber[1] == 0) blockNumber[1] = 1;
	if (blockNumber[2] == 0) blockNumber[2] = 1;
	if ((blockNumber[0] > 1) && (blockNumber[0] * maxBlockSize[0] != dispatchBlock[0])) {
		for (uint64_t i = app->configuration.maxComputeWorkGroupCount[0]; i > 0; i--) {
			if (dispatchBlock[0] % i == 0) {
				maxBlockSize[0] = i;
				blockNumber[0] = dispatchBlock[0] / i;
				i = 1;
			}
		}
	}
	if ((blockNumber[1] > 1) && (blockNumber[1] * maxBlockSize[1] != dispatchBlock[1])) {
		for (uint64_t i = app->configuration.maxComputeWorkGroupCount[1]; i > 0; i--) {
			if (dispatchBlock[1] % i == 0) {
				maxBlockSize[1] = i;
				blockNumber[1] = dispatchBlock[1] / i;
				i = 1;
			}
		}
	}
	if ((blockNumber[2] > 1) && (blockNumber[2] * maxBlockSize[2] != dispatchBlock[2])) {
		for (uint64_t i = app->configuration.maxComputeWorkGroupCount[2]; i > 0; i--) {
			if (dispatchBlock[2] % i == 0) {
				maxBlockSize[2] = i;
				blockNumber[2] = dispatchBlock[2] / i;
				i = 1;
			}
		}
	}
	if (app->configuration.specifyOffsetsAtLaunch) {
		axis->updatePushConstants = 1;
	}
	//printf("%" PRIu64 " %" PRIu64 " %" PRIu64 "\n", dispatchBlock[0], dispatchBlock[1], dispatchBlock[2]);
	//printf("%" PRIu64 " %" PRIu64 " %" PRIu64 "\n", blockNumber[0], blockNumber[1], blockNumber[2]);
	for (uint64_t i = 0; i < 3; i++)
		if (blockNumber[i] == 1) maxBlockSize[i] = dispatchBlock[i];
	for (uint64_t i = 0; i < blockNumber[0]; i++) {
		for (uint64_t j = 0; j < blockNumber[1]; j++) {
			for (uint64_t k = 0; k < blockNumber[2]; k++) {
				if (axis->pushConstants.workGroupShift[0] != i * maxBlockSize[0]) {
					axis->pushConstants.workGroupShift[0] = i * maxBlockSize[0];
					axis->updatePushConstants = 1;
				}
				if (axis->pushConstants.workGroupShift[1] != j * maxBlockSize[1]) {
					axis->pushConstants.workGroupShift[1] = j * maxBlockSize[1];
					axis->updatePushConstants = 1;
				}
				if (axis->pushConstants.workGroupShift[2] != k * maxBlockSize[2]) {
					axis->pushConstants.workGroupShift[2] = k * maxBlockSize[2];
					axis->updatePushConstants = 1;
				}
				if (axis->updatePushConstants) {
					if (app->configuration.useUint64) {
						uint64_t pushConstID = 0;
						if (axis->specializationConstants.performWorkGroupShift[0]) {
							axis->pushConstants.dataUint64[pushConstID] = axis->pushConstants.workGroupShift[0];
							pushConstID++;
						}
						if (axis->specializationConstants.performWorkGroupShift[1]) {
							axis->pushConstants.dataUint64[pushConstID] = axis->pushConstants.workGroupShift[1];
							pushConstID++;
						}
						if (axis->specializationConstants.performWorkGroupShift[2]) {
							axis->pushConstants.dataUint64[pushConstID] = axis->pushConstants.workGroupShift[2];
							pushConstID++;
						}
						if (axis->specializationConstants.performPostCompilationInputOffset) {
							axis->pushConstants.dataUint64[pushConstID] = axis->specializationConstants.inputOffset / axis->specializationConstants.inputNumberByteSize;
							pushConstID++;
						}
						if (axis->specializationConstants.performPostCompilationOutputOffset) {
							axis->pushConstants.dataUint64[pushConstID] = axis->specializationConstants.outputOffset / axis->specializationConstants.outputNumberByteSize;
							pushConstID++;
						}
						if (axis->specializationConstants.performPostCompilationKernelOffset) {
							if (axis->specializationConstants.kernelNumberByteSize != 0)
								axis->pushConstants.dataUint64[pushConstID] = axis->specializationConstants.kernelOffset / axis->specializationConstants.kernelNumberByteSize;
							else
								axis->pushConstants.dataUint64[pushConstID] = 0;
							pushConstID++;
						}
					}
					else {
						uint64_t pushConstID = 0;
						if (axis->specializationConstants.performWorkGroupShift[0]) {
							axis->pushConstants.dataUint32[pushConstID] = (uint32_t)axis->pushConstants.workGroupShift[0];
							pushConstID++;
						}
						if (axis->specializationConstants.performWorkGroupShift[1]) {
							axis->pushConstants.dataUint32[pushConstID] = (uint32_t)axis->pushConstants.workGroupShift[1];
							pushConstID++;
						}
						if (axis->specializationConstants.performWorkGroupShift[2]) {
							axis->pushConstants.dataUint32[pushConstID] = (uint32_t)axis->pushConstants.workGroupShift[2];
							pushConstID++;
						}
						if (axis->specializationConstants.performPostCompilationInputOffset) {
							axis->pushConstants.dataUint32[pushConstID] = (uint32_t)(axis->specializationConstants.inputOffset / axis->specializationConstants.inputNumberByteSize);
							pushConstID++;
						}
						if (axis->specializationConstants.performPostCompilationOutputOffset) {
							axis->pushConstants.dataUint32[pushConstID] = (uint32_t)(axis->specializationConstants.outputOffset / axis->specializationConstants.outputNumberByteSize);
							pushConstID++;
						}
						if (axis->specializationConstants.performPostCompilationKernelOffset) {
							axis->pushConstants.dataUint32[pushConstID] = (uint32_t)(axis->specializationConstants.kernelOffset / axis->specializationConstants.kernelNumberByteSize);
							pushConstID++;
						}
					}
				}
#if(VKFFT_BACKEND==0)
				if (axis->pushConstants.structSize > 0) {
					if (app->configuration.useUint64) {
						vkCmdPushConstants(app->configuration.commandBuffer[0], axis->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)axis->pushConstants.structSize, axis->pushConstants.dataUint64);
					}
					else {
						vkCmdPushConstants(app->configuration.commandBuffer[0], axis->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)axis->pushConstants.structSize, axis->pushConstants.dataUint32);
					}
				}
				vkCmdDispatch(app->configuration.commandBuffer[0], (uint32_t)maxBlockSize[0], (uint32_t)maxBlockSize[1], (uint32_t)maxBlockSize[2]);
#elif(VKFFT_BACKEND==1)
				void* args[6];
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
						if (app->configuration.useUint64) {
							result = cuMemcpyHtoD(axis->consts_addr, axis->pushConstants.dataUint64, axis->pushConstants.structSize);
						}
						else {
							result = cuMemcpyHtoD(axis->consts_addr, axis->pushConstants.dataUint32, axis->pushConstants.structSize);
						}
						if (result != CUDA_SUCCESS) {
							printf("cuMemcpyHtoD error: %d\n", result);
							return VKFFT_ERROR_FAILED_TO_COPY;
						}
					}
				}
				if (app->configuration.num_streams >= 1) {
					result = cuLaunchKernel(axis->VkFFTKernel,
						(unsigned int)maxBlockSize[0], (unsigned int)maxBlockSize[1], (unsigned int)maxBlockSize[2],     // grid dim
						(unsigned int)axis->specializationConstants.localSize[0], (unsigned int)axis->specializationConstants.localSize[1], (unsigned int)axis->specializationConstants.localSize[2],   // block dim
						(unsigned int)axis->specializationConstants.usedSharedMemory, app->configuration.stream[app->configuration.streamID],             // shared mem and stream
						args, 0);
				}
				else {
					result = cuLaunchKernel(axis->VkFFTKernel,
						(unsigned int)maxBlockSize[0], (unsigned int)maxBlockSize[1], (unsigned int)maxBlockSize[2],     // grid dim
						(unsigned int)axis->specializationConstants.localSize[0], (unsigned int)axis->specializationConstants.localSize[1], (unsigned int)axis->specializationConstants.localSize[2],   // block dim
						(unsigned int)axis->specializationConstants.usedSharedMemory, 0,             // shared mem and stream
						args, 0);
				}
				if (result != CUDA_SUCCESS) {
					printf("cuLaunchKernel error: %d, %" PRIu64 " %" PRIu64 " %" PRIu64 " - %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", result, maxBlockSize[0], maxBlockSize[1], maxBlockSize[2], axis->specializationConstants.localSize[0], axis->specializationConstants.localSize[1], axis->specializationConstants.localSize[2]);
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
				void* args[6];
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
						if (app->configuration.useUint64) {
							result = hipMemcpyHtoD(axis->consts_addr, axis->pushConstants.dataUint64, axis->pushConstants.structSize);
						}
						else {
							result = hipMemcpyHtoD(axis->consts_addr, axis->pushConstants.dataUint32, axis->pushConstants.structSize);
						}
						if (result != hipSuccess) {
							printf("hipMemcpyHtoD error: %d\n", result);
							return VKFFT_ERROR_FAILED_TO_COPY;
						}
					}
				}
				//printf("%" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n",maxBlockSize[0], maxBlockSize[1], maxBlockSize[2], axis->specializationConstants.localSize[0], axis->specializationConstants.localSize[1], axis->specializationConstants.localSize[2]);
				if (app->configuration.num_streams >= 1) {
					result = hipModuleLaunchKernel(axis->VkFFTKernel,
						(unsigned int)maxBlockSize[0], (unsigned int)maxBlockSize[1], (unsigned int)maxBlockSize[2],     // grid dim
						(unsigned int)axis->specializationConstants.localSize[0], (unsigned int)axis->specializationConstants.localSize[1], (unsigned int)axis->specializationConstants.localSize[2],   // block dim
						(unsigned int)axis->specializationConstants.usedSharedMemory, app->configuration.stream[app->configuration.streamID],             // shared mem and stream
						args, 0);
				}
				else {
					result = hipModuleLaunchKernel(axis->VkFFTKernel,
						(unsigned int)maxBlockSize[0], (unsigned int)maxBlockSize[1], (unsigned int)maxBlockSize[2],     // grid dim
						(unsigned int)axis->specializationConstants.localSize[0], (unsigned int)axis->specializationConstants.localSize[1], (unsigned int)axis->specializationConstants.localSize[2],   // block dim
						(unsigned int)axis->specializationConstants.usedSharedMemory, 0,             // shared mem and stream
						args, 0);
				}
				if (result != hipSuccess) {
					printf("hipModuleLaunchKernel error: %d, %" PRIu64 " %" PRIu64 " %" PRIu64 " - %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", result, maxBlockSize[0], maxBlockSize[1], maxBlockSize[2], axis->specializationConstants.localSize[0], axis->specializationConstants.localSize[1], axis->specializationConstants.localSize[2]);
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
				void* args[6];
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
					if (app->configuration.useUint64) {
						result = clSetKernelArg(axis->kernel, (cl_uint)args_id, axis->pushConstants.structSize, axis->pushConstants.dataUint64);
					}
					else {
						result = clSetKernelArg(axis->kernel, (cl_uint)args_id, axis->pushConstants.structSize, axis->pushConstants.dataUint32);
					}
					if (result != CL_SUCCESS) {
						return VKFFT_ERROR_FAILED_TO_SET_KERNEL_ARG;
					}
					args_id++;
				}
				size_t local_work_size[3] = { (size_t)axis->specializationConstants.localSize[0], (size_t)axis->specializationConstants.localSize[1],(size_t)axis->specializationConstants.localSize[2] };
				size_t global_work_size[3] = { (size_t)maxBlockSize[0] * local_work_size[0] , (size_t)maxBlockSize[1] * local_work_size[1] ,(size_t)maxBlockSize[2] * local_work_size[2] };
				result = clEnqueueNDRangeKernel(app->configuration.commandQueue[0], axis->kernel, 3, 0, global_work_size, local_work_size, 0, 0, 0);
				//printf("%" PRIu64 " %" PRIu64 " %" PRIu64 " - %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", maxBlockSize[0], maxBlockSize[1], maxBlockSize[2], axis->specializationConstants.localSize[0], axis->specializationConstants.localSize[1], axis->specializationConstants.localSize[2]);

				if (result != CL_SUCCESS) {
					return VKFFT_ERROR_FAILED_TO_LAUNCH_KERNEL;
				}
#endif
			}
		}
	}
	return resFFT;
}

#endif
