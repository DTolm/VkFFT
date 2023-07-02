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
#ifndef VKFFT_RUNAPP_H
#define VKFFT_RUNAPP_H
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT/vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_DispatchPlan.h"
#include "vkFFT/vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_UpdateBuffers.h"

static inline VkFFTResult VkFFTSync(VkFFTApplication* app) {
#if(VKFFT_BACKEND==0)
	vkCmdPipelineBarrier(app->configuration.commandBuffer[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, app->configuration.memory_barrier, 0, 0, 0, 0);
#elif(VKFFT_BACKEND==1)
	if (app->configuration.num_streams > 1) {
		cudaError_t res = cudaSuccess;
		for (uint64_t s = 0; s < app->configuration.num_streams; s++) {
			res = cudaEventSynchronize(app->configuration.stream_event[s]);
			if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
		}
		app->configuration.streamCounter = 0;
	}
#elif(VKFFT_BACKEND==2)
	if (app->configuration.num_streams > 1) {
		hipError_t res = hipSuccess;
		for (uint64_t s = 0; s < app->configuration.num_streams; s++) {
			res = hipEventSynchronize(app->configuration.stream_event[s]);
			if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
		}
		app->configuration.streamCounter = 0;
	}
#elif(VKFFT_BACKEND==3)
#elif(VKFFT_BACKEND==4)
	ze_result_t res = ZE_RESULT_SUCCESS;
	res = zeCommandListAppendBarrier(app->configuration.commandList[0], nullptr, 0, nullptr);
	if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_SUBMIT_BARRIER;
#elif(VKFFT_BACKEND==5)
#endif
	return VKFFT_SUCCESS;
}
static inline void printDebugInformation(VkFFTApplication* app, VkFFTAxis* axis) {
	if (app->configuration.keepShaderCode) printf("%s\n", axis->specializationConstants.code0);
	if (app->configuration.printMemoryLayout) {
		if ((axis->inputBuffer == app->configuration.inputBuffer) && (app->configuration.inputBuffer != app->configuration.buffer))
			printf("read: inputBuffer\n");
		if (axis->inputBuffer == app->configuration.buffer)
			printf("read: buffer\n");
		if (axis->inputBuffer == app->configuration.tempBuffer)
			printf("read: tempBuffer\n");
		if ((axis->inputBuffer == app->configuration.outputBuffer) && (app->configuration.outputBuffer != app->configuration.buffer))
			printf("read: outputBuffer\n");
		if ((axis->outputBuffer == app->configuration.inputBuffer) && (app->configuration.inputBuffer != app->configuration.buffer))
			printf("write: inputBuffer\n");
		if (axis->outputBuffer == app->configuration.buffer)
			printf("write: buffer\n");
		if (axis->outputBuffer == app->configuration.tempBuffer)
			printf("write: tempBuffer\n");
		if ((axis->outputBuffer == app->configuration.outputBuffer) && (app->configuration.outputBuffer != app->configuration.buffer))
			printf("write: outputBuffer\n");
	}
}
static inline VkFFTResult VkFFTAppend(VkFFTApplication* app, int inverse, VkFFTLaunchParams* launchParams) {
	VkFFTResult resFFT = VKFFT_SUCCESS;
#if(VKFFT_BACKEND==0)
	app->configuration.commandBuffer = launchParams->commandBuffer;
	VkMemoryBarrier memory_barrier = {
			VK_STRUCTURE_TYPE_MEMORY_BARRIER,
			0,
			VK_ACCESS_SHADER_WRITE_BIT,
			VK_ACCESS_SHADER_READ_BIT,
	};
	app->configuration.memory_barrier = &memory_barrier;
#elif(VKFFT_BACKEND==1)
	app->configuration.streamCounter = 0;
#elif(VKFFT_BACKEND==2)
	app->configuration.streamCounter = 0;
#elif(VKFFT_BACKEND==3)
	app->configuration.commandQueue = launchParams->commandQueue;
#elif(VKFFT_BACKEND==4)
	app->configuration.commandList = launchParams->commandList;
#elif(VKFFT_BACKEND==5)
	app->configuration.commandBuffer = launchParams->commandBuffer;
	app->configuration.commandEncoder = launchParams->commandEncoder;
#endif
	uint64_t localSize0[3];
	if ((inverse != 1) && (app->configuration.makeInversePlanOnly)) return VKFFT_ERROR_ONLY_INVERSE_FFT_INITIALIZED;
	if ((inverse == 1) && (app->configuration.makeForwardPlanOnly)) return VKFFT_ERROR_ONLY_FORWARD_FFT_INITIALIZED;
	if ((inverse != 1) && (!app->configuration.makeInversePlanOnly) && (!app->localFFTPlan)) return VKFFT_ERROR_PLAN_NOT_INITIALIZED;
	if ((inverse == 1) && (!app->configuration.makeForwardPlanOnly) && (!app->localFFTPlan_inverse)) return VKFFT_ERROR_PLAN_NOT_INITIALIZED;
	if (inverse == 1) {
		localSize0[0] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0];
		localSize0[1] = app->localFFTPlan_inverse->actualFFTSizePerAxis[1][0];
		localSize0[2] = app->localFFTPlan_inverse->actualFFTSizePerAxis[2][0];
	}
	else {
		localSize0[0] = app->localFFTPlan->actualFFTSizePerAxis[0][0];
		localSize0[1] = app->localFFTPlan->actualFFTSizePerAxis[1][0];
		localSize0[2] = app->localFFTPlan->actualFFTSizePerAxis[2][0];
	}
	resFFT = VkFFTCheckUpdateBufferSet(app, 0, 0, launchParams);
	if (resFFT != VKFFT_SUCCESS) {
		return resFFT;
	}
	if (inverse != 1) {
		//FFT axis 0
		if (!app->configuration.omitDimension[0]) {
			for (int64_t l = (int64_t)app->localFFTPlan->numAxisUploads[0] - 1; l >= 0; l--) {
				VkFFTAxis* axis = &app->localFFTPlan->axes[0][l];
				resFFT = VkFFTUpdateBufferSet(app, app->localFFTPlan, axis, 0, l, 0);
				if (resFFT != VKFFT_SUCCESS) return resFFT;
				uint64_t maxCoordinate = ((app->configuration.matrixConvolution > 1) && (app->configuration.performConvolution) && (app->configuration.FFTdim == 1) && (l == 0)) ? 1 : app->configuration.coordinateFeatures;
#if(VKFFT_BACKEND==0)
				vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
				vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
				uint64_t dispatchBlock[3];
				if (l == 0) {
					if (app->localFFTPlan->numAxisUploads[0] > 2) {
						dispatchBlock[0] = (uint64_t)ceil((uint64_t)ceil(app->localFFTPlan->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)axis->axisBlock[1]) / (double)app->localFFTPlan->axisSplit[0][1]) * app->localFFTPlan->axisSplit[0][1];
						dispatchBlock[1] = app->localFFTPlan->actualFFTSizePerAxis[0][1];
					}
					else {
						if (app->localFFTPlan->numAxisUploads[0] > 1) {
							dispatchBlock[0] = (uint64_t)ceil((uint64_t)ceil(app->localFFTPlan->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)axis->axisBlock[1]));
							dispatchBlock[1] = app->localFFTPlan->actualFFTSizePerAxis[0][1];
						}
						else {
							dispatchBlock[0] = app->localFFTPlan->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i;
							dispatchBlock[1] = (uint64_t)ceil(app->localFFTPlan->actualFFTSizePerAxis[0][1] / (double)axis->axisBlock[1]);
						}
					}
				}
				else {
					dispatchBlock[0] = (uint64_t)ceil(app->localFFTPlan->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)axis->axisBlock[0]);
					dispatchBlock[1] = app->localFFTPlan->actualFFTSizePerAxis[0][1];
				}
				dispatchBlock[2] = app->localFFTPlan->actualFFTSizePerAxis[0][2] * maxCoordinate * app->configuration.numberBatches;
				if (axis->specializationConstants.mergeSequencesR2C == 1) dispatchBlock[1] = (uint64_t)ceil(dispatchBlock[1] / 2.0);
				//if (app->configuration.performZeropadding[1]) dispatchBlock[1] = (uint64_t)ceil(dispatchBlock[1] / 2.0);
				//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = (uint64_t)ceil(dispatchBlock[2] / 2.0);
				resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
				if (resFFT != VKFFT_SUCCESS) return resFFT;
				printDebugInformation(app, axis);
				resFFT = VkFFTSync(app);
				if (resFFT != VKFFT_SUCCESS) return resFFT;
			}
			if (app->useBluesteinFFT[0] && (app->localFFTPlan->numAxisUploads[0] > 1)) {
				for (int64_t l = 1; l < (int64_t)app->localFFTPlan->numAxisUploads[0]; l++) {
					VkFFTAxis* axis = &app->localFFTPlan->inverseBluesteinAxes[0][l];
					resFFT = VkFFTUpdateBufferSet(app, app->localFFTPlan, axis, 0, l, 0);
					if (resFFT != VKFFT_SUCCESS) return resFFT;
					uint64_t maxCoordinate = ((app->configuration.matrixConvolution > 1) && (app->configuration.performConvolution) && (app->configuration.FFTdim == 1)) ? 1 : app->configuration.coordinateFeatures;
#if(VKFFT_BACKEND==0)
					vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
					vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
					uint64_t dispatchBlock[3];
					if (l == 0) {
						if (app->localFFTPlan->numAxisUploads[0] > 2) {
							dispatchBlock[0] = (uint64_t)ceil((uint64_t)ceil(app->localFFTPlan->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)axis->axisBlock[1]) / (double)app->localFFTPlan->axisSplit[0][1]) * app->localFFTPlan->axisSplit[0][1];
							dispatchBlock[1] = app->localFFTPlan->actualFFTSizePerAxis[0][1];
						}
						else {
							if (app->localFFTPlan->numAxisUploads[0] > 1) {
								dispatchBlock[0] = (uint64_t)ceil((uint64_t)ceil(app->localFFTPlan->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)axis->axisBlock[1]));
								dispatchBlock[1] = app->localFFTPlan->actualFFTSizePerAxis[0][1];
							}
							else {
								dispatchBlock[0] = app->localFFTPlan->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i;
								dispatchBlock[1] = (uint64_t)ceil(app->localFFTPlan->actualFFTSizePerAxis[0][1] / (double)axis->axisBlock[1]);
							}
						}
					}
					else {
						dispatchBlock[0] = (uint64_t)ceil(app->localFFTPlan->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)axis->axisBlock[0]);
						dispatchBlock[1] = app->localFFTPlan->actualFFTSizePerAxis[0][1];
					}
					dispatchBlock[2] = app->localFFTPlan->actualFFTSizePerAxis[0][2] * maxCoordinate * app->configuration.numberBatches;
					if (axis->specializationConstants.mergeSequencesR2C == 1) dispatchBlock[1] = (uint64_t)ceil(dispatchBlock[1] / 2.0);
					//if (app->configuration.performZeropadding[1]) dispatchBlock[1] = (uint64_t)ceil(dispatchBlock[1] / 2.0);
					//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = (uint64_t)ceil(dispatchBlock[2] / 2.0);
					resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
					if (resFFT != VKFFT_SUCCESS) return resFFT;
					printDebugInformation(app, axis);
					resFFT = VkFFTSync(app);
					if (resFFT != VKFFT_SUCCESS) return resFFT;
				}
			}
			if (app->localFFTPlan->multiUploadR2C) {
				VkFFTAxis* axis = &app->localFFTPlan->R2Cdecomposition;
				resFFT = VkFFTUpdateBufferSetR2CMultiUploadDecomposition(app, app->localFFTPlan, axis, 0, 0, 0);
				if (resFFT != VKFFT_SUCCESS) return resFFT;
				uint64_t maxCoordinate = ((app->configuration.matrixConvolution > 1) && (app->configuration.performConvolution) && (app->configuration.FFTdim == 1)) ? 1 : app->configuration.coordinateFeatures;

#if(VKFFT_BACKEND==0)
				vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
				vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
				uint64_t dispatchBlock[3];

				dispatchBlock[0] = (uint64_t)ceil(((app->configuration.size[0] / 2 + 1) * app->configuration.size[1] * app->configuration.size[2]) / (double)(2 * axis->axisBlock[0]));
				dispatchBlock[1] = 1;
				dispatchBlock[2] = maxCoordinate * axis->specializationConstants.numBatches.data.i;
				resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
				if (resFFT != VKFFT_SUCCESS) return resFFT;
				printDebugInformation(app, axis);
				resFFT = VkFFTSync(app);
				if (resFFT != VKFFT_SUCCESS) return resFFT;
				//app->configuration.size[0] *= 2;
			}
		}
		if (app->configuration.FFTdim > 1) {

			//FFT axis 1
			if (!app->configuration.omitDimension[1]) {
				if ((app->configuration.FFTdim == 2) && (app->configuration.performConvolution)) {

					for (int64_t l = (int64_t)app->localFFTPlan->numAxisUploads[1] - 1; l >= 0; l--) {
						VkFFTAxis* axis = &app->localFFTPlan->axes[1][l];
						resFFT = VkFFTUpdateBufferSet(app, app->localFFTPlan, axis, 1, l, 0);
						if (resFFT != VKFFT_SUCCESS) return resFFT;
						uint64_t maxCoordinate = ((app->configuration.matrixConvolution > 1) && (l == 0)) ? 1 : app->configuration.coordinateFeatures;

#if(VKFFT_BACKEND==0)
						vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
						vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
						uint64_t dispatchBlock[3];
						dispatchBlock[0] = (uint64_t)ceil(localSize0[1] / (double)axis->axisBlock[0] * app->localFFTPlan->actualFFTSizePerAxis[1][1] / (double)axis->specializationConstants.fftDim.data.i);
						dispatchBlock[1] = 1;
						dispatchBlock[2] = app->localFFTPlan->actualFFTSizePerAxis[1][2] * maxCoordinate * app->configuration.numberBatches;
						//if (app->configuration.mergeSequencesR2C == 1) dispatchBlock[0] = (uint64_t)ceil(dispatchBlock[0] / 2.0);
						//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = (uint64_t)ceil(dispatchBlock[2] / 2.0);
						resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
						if (resFFT != VKFFT_SUCCESS) return resFFT;
						printDebugInformation(app, axis);
						resFFT = VkFFTSync(app);
						if (resFFT != VKFFT_SUCCESS) return resFFT;
					}
				}
				else {

					for (int64_t l = (int64_t)app->localFFTPlan->numAxisUploads[1] - 1; l >= 0; l--) {
						VkFFTAxis* axis = &app->localFFTPlan->axes[1][l];
						resFFT = VkFFTUpdateBufferSet(app, app->localFFTPlan, axis, 1, l, 0);
						if (resFFT != VKFFT_SUCCESS) return resFFT;

#if(VKFFT_BACKEND==0)
						vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
						vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
						uint64_t dispatchBlock[3];

						dispatchBlock[0] = (uint64_t)ceil(localSize0[1] / (double)axis->axisBlock[0] * app->localFFTPlan->actualFFTSizePerAxis[1][1] / (double)axis->specializationConstants.fftDim.data.i);
						dispatchBlock[1] = 1;
						dispatchBlock[2] = app->localFFTPlan->actualFFTSizePerAxis[1][2] * app->configuration.coordinateFeatures * app->configuration.numberBatches;
						//if (app->configuration.mergeSequencesR2C == 1) dispatchBlock[0] = (uint64_t)ceil(dispatchBlock[0] / 2.0);
						//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = (uint64_t)ceil(dispatchBlock[2] / 2.0);
						resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
						if (resFFT != VKFFT_SUCCESS) return resFFT;
						printDebugInformation(app, axis);

						resFFT = VkFFTSync(app);
						if (resFFT != VKFFT_SUCCESS) return resFFT;
					}
					if (app->useBluesteinFFT[1] && (app->localFFTPlan->numAxisUploads[1] > 1)) {
						for (int64_t l = 1; l < (int64_t)app->localFFTPlan->numAxisUploads[1]; l++) {
							VkFFTAxis* axis = &app->localFFTPlan->inverseBluesteinAxes[1][l];
							resFFT = VkFFTUpdateBufferSet(app, app->localFFTPlan, axis, 1, l, 0);
							if (resFFT != VKFFT_SUCCESS) return resFFT;
#if(VKFFT_BACKEND==0)
							vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
							vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
							uint64_t dispatchBlock[3];
							dispatchBlock[0] = (uint64_t)ceil(localSize0[1] / (double)axis->axisBlock[0] * app->localFFTPlan->actualFFTSizePerAxis[1][1] / (double)axis->specializationConstants.fftDim.data.i);
							dispatchBlock[1] = 1;
							dispatchBlock[2] = app->localFFTPlan->actualFFTSizePerAxis[1][2] * app->configuration.coordinateFeatures * app->configuration.numberBatches;

							//if (app->configuration.performZeropadding[1]) dispatchBlock[1] = (uint64_t)ceil(dispatchBlock[1] / 2.0);
							//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = (uint64_t)ceil(dispatchBlock[2] / 2.0);
							resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
							if (resFFT != VKFFT_SUCCESS) return resFFT;
							printDebugInformation(app, axis);
							resFFT = VkFFTSync(app);
							if (resFFT != VKFFT_SUCCESS) return resFFT;
						}
					}
				}
			}
		}
		//FFT axis 2
		if (app->configuration.FFTdim > 2) {
			if (!app->configuration.omitDimension[2]) {
				if ((app->configuration.FFTdim == 3) && (app->configuration.performConvolution)) {

					for (int64_t l = (int64_t)app->localFFTPlan->numAxisUploads[2] - 1; l >= 0; l--) {

						VkFFTAxis* axis = &app->localFFTPlan->axes[2][l];
						resFFT = VkFFTUpdateBufferSet(app, app->localFFTPlan, axis, 2, l, 0);
						if (resFFT != VKFFT_SUCCESS) return resFFT;
						uint64_t maxCoordinate = ((app->configuration.matrixConvolution > 1) && (l == 0)) ? 1 : app->configuration.coordinateFeatures;
#if(VKFFT_BACKEND==0)
						vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
						vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
						uint64_t dispatchBlock[3];
						dispatchBlock[0] = (uint64_t)ceil(localSize0[2] / (double)axis->axisBlock[0] * app->localFFTPlan->actualFFTSizePerAxis[2][2] / (double)axis->specializationConstants.fftDim.data.i);
						dispatchBlock[1] = 1;
						dispatchBlock[2] = app->localFFTPlan->actualFFTSizePerAxis[2][1] * maxCoordinate * app->configuration.numberBatches;
						//if (app->configuration.mergeSequencesR2C == 1) dispatchBlock[0] = (uint64_t)ceil(dispatchBlock[0] / 2.0);
						resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
						if (resFFT != VKFFT_SUCCESS) return resFFT;
						printDebugInformation(app, axis);
						resFFT = VkFFTSync(app);
						if (resFFT != VKFFT_SUCCESS) return resFFT;
					}
				}
				else {

					for (int64_t l = (int64_t)app->localFFTPlan->numAxisUploads[2] - 1; l >= 0; l--) {
						VkFFTAxis* axis = &app->localFFTPlan->axes[2][l];
						resFFT = VkFFTUpdateBufferSet(app, app->localFFTPlan, axis, 2, l, 0);
						if (resFFT != VKFFT_SUCCESS) return resFFT;
#if(VKFFT_BACKEND==0)
						vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
						vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
						uint64_t dispatchBlock[3];
						dispatchBlock[0] = (uint64_t)ceil(localSize0[2] / (double)axis->axisBlock[0] * app->localFFTPlan->actualFFTSizePerAxis[2][2] / (double)axis->specializationConstants.fftDim.data.i);
						dispatchBlock[1] = 1;
						dispatchBlock[2] = app->localFFTPlan->actualFFTSizePerAxis[2][1] * app->configuration.coordinateFeatures * app->configuration.numberBatches;
						//if (app->configuration.mergeSequencesR2C == 1) dispatchBlock[0] = (uint64_t)ceil(dispatchBlock[0] / 2.0);
						resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
						if (resFFT != VKFFT_SUCCESS) return resFFT;
						printDebugInformation(app, axis);
						resFFT = VkFFTSync(app);
						if (resFFT != VKFFT_SUCCESS) return resFFT;
					}
					if (app->useBluesteinFFT[2] && (app->localFFTPlan->numAxisUploads[2] > 1)) {
						for (int64_t l = 1; l < (int64_t)app->localFFTPlan->numAxisUploads[2]; l++) {
							VkFFTAxis* axis = &app->localFFTPlan->inverseBluesteinAxes[2][l];
							resFFT = VkFFTUpdateBufferSet(app, app->localFFTPlan, axis, 2, l, 0);
							if (resFFT != VKFFT_SUCCESS) return resFFT;
#if(VKFFT_BACKEND==0)
							vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
							vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
							uint64_t dispatchBlock[3];
							dispatchBlock[0] = (uint64_t)ceil(localSize0[2] / (double)axis->axisBlock[0] * app->localFFTPlan->actualFFTSizePerAxis[2][2] / (double)axis->specializationConstants.fftDim.data.i);
							dispatchBlock[1] = 1;
							dispatchBlock[2] = app->localFFTPlan->actualFFTSizePerAxis[2][1] * app->configuration.coordinateFeatures * app->configuration.numberBatches;

							//if (app->configuration.performZeropadding[1]) dispatchBlock[1] = (uint64_t)ceil(dispatchBlock[1] / 2.0);
							//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = (uint64_t)ceil(dispatchBlock[2] / 2.0);
							resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
							if (resFFT != VKFFT_SUCCESS) return resFFT;
							printDebugInformation(app, axis);
							resFFT = VkFFTSync(app);
							if (resFFT != VKFFT_SUCCESS) return resFFT;
						}
					}
				}
			}
		}
	}
	if (app->configuration.performConvolution) {
		if (app->configuration.FFTdim > 2) {

			//multiple upload ifft leftovers
			if (app->configuration.FFTdim == 3) {

				for (int64_t l = (int64_t)1; l < (int64_t)app->localFFTPlan_inverse->numAxisUploads[2]; l++) {
					VkFFTAxis* axis = &app->localFFTPlan_inverse->axes[2][l];
					resFFT = VkFFTUpdateBufferSet(app, app->localFFTPlan_inverse, axis, 2, l, 1);
					if (resFFT != VKFFT_SUCCESS) return resFFT;

#if(VKFFT_BACKEND==0)
					vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
					vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
					uint64_t dispatchBlock[3];
					dispatchBlock[0] = (uint64_t)ceil(localSize0[2] / (double)axis->axisBlock[0] * app->localFFTPlan_inverse->actualFFTSizePerAxis[2][2] / (double)axis->specializationConstants.fftDim.data.i);
					dispatchBlock[1] = 1;
					dispatchBlock[2] = app->localFFTPlan_inverse->actualFFTSizePerAxis[2][1] * app->configuration.coordinateFeatures * app->configuration.numberKernels;
					//if (app->configuration.mergeSequencesR2C == 1) dispatchBlock[0] = (uint64_t)ceil(dispatchBlock[0] / 2.0);
					resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
					if (resFFT != VKFFT_SUCCESS) return resFFT;
					printDebugInformation(app, axis);
					resFFT = VkFFTSync(app);
					if (resFFT != VKFFT_SUCCESS) return resFFT;
				}
			}

			for (int64_t l = 0; l < (int64_t)app->localFFTPlan_inverse->numAxisUploads[1]; l++) {
				VkFFTAxis* axis = &app->localFFTPlan_inverse->axes[1][l];
				resFFT = VkFFTUpdateBufferSet(app, app->localFFTPlan_inverse, axis, 1, l, 1);
				if (resFFT != VKFFT_SUCCESS) return resFFT;

#if(VKFFT_BACKEND==0)
				vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
				vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
				uint64_t dispatchBlock[3];
				dispatchBlock[0] = (uint64_t)ceil(localSize0[2] / (double)axis->axisBlock[0] * app->localFFTPlan_inverse->actualFFTSizePerAxis[1][1] / (double)axis->specializationConstants.fftDim.data.i);
				dispatchBlock[1] = 1;
				dispatchBlock[2] = app->localFFTPlan_inverse->actualFFTSizePerAxis[1][2] * app->configuration.coordinateFeatures * app->configuration.numberKernels;
				//if (app->configuration.mergeSequencesR2C == 1) dispatchBlock[0] = (uint64_t)ceil(dispatchBlock[0] / 2.0);
				//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = (uint64_t)ceil(dispatchBlock[2] / 2.0);
				resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
				if (resFFT != VKFFT_SUCCESS) return resFFT;
				printDebugInformation(app, axis);
				resFFT = VkFFTSync(app);
				if (resFFT != VKFFT_SUCCESS) return resFFT;
			}

		}
		if (app->configuration.FFTdim > 1) {
			if (app->configuration.FFTdim == 2) {

				for (int64_t l = (int64_t)1; l < (int64_t)app->localFFTPlan_inverse->numAxisUploads[1]; l++) {
					VkFFTAxis* axis = &app->localFFTPlan_inverse->axes[1][l];
					resFFT = VkFFTUpdateBufferSet(app, app->localFFTPlan_inverse, axis, 1, l, 1);
					if (resFFT != VKFFT_SUCCESS) return resFFT;

#if(VKFFT_BACKEND==0)
					vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
					vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
					uint64_t dispatchBlock[3];
					dispatchBlock[0] = (uint64_t)ceil(localSize0[1] / (double)axis->axisBlock[0] * app->localFFTPlan_inverse->actualFFTSizePerAxis[1][1] / (double)axis->specializationConstants.fftDim.data.i);
					dispatchBlock[1] = 1;
					dispatchBlock[2] = app->localFFTPlan_inverse->actualFFTSizePerAxis[1][2] * app->configuration.coordinateFeatures * app->configuration.numberKernels;
					//if (app->configuration.mergeSequencesR2C == 1) dispatchBlock[0] = (uint64_t)ceil(dispatchBlock[0] / 2.0);
					//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = (uint64_t)ceil(dispatchBlock[2] / 2.0);
					resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
					if (resFFT != VKFFT_SUCCESS) return resFFT;
					printDebugInformation(app, axis);
					resFFT = VkFFTSync(app);
					if (resFFT != VKFFT_SUCCESS) return resFFT;
				}
			}
			if (app->localFFTPlan_inverse->multiUploadR2C) {
				//app->configuration.size[0] /= 2;
				VkFFTAxis* axis = &app->localFFTPlan_inverse->R2Cdecomposition;
				resFFT = VkFFTUpdateBufferSetR2CMultiUploadDecomposition(app, app->localFFTPlan_inverse, axis, 0, 0, 1);
				if (resFFT != VKFFT_SUCCESS) return resFFT;

#if(VKFFT_BACKEND==0)
				vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
				vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
				uint64_t dispatchBlock[3];

				dispatchBlock[0] = (uint64_t)ceil(((app->configuration.size[0] / 2 + 1) * app->configuration.size[1] * app->configuration.size[2]) / (double)(2 * axis->axisBlock[0]));
				dispatchBlock[1] = 1;
				dispatchBlock[2] = app->configuration.coordinateFeatures * app->configuration.numberBatches;
				resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
				if (resFFT != VKFFT_SUCCESS) return resFFT;
				printDebugInformation(app, axis);

				resFFT = VkFFTSync(app);
				if (resFFT != VKFFT_SUCCESS) return resFFT;
			}
			for (int64_t l = 0; l < (int64_t)app->localFFTPlan_inverse->numAxisUploads[0]; l++) {
				VkFFTAxis* axis = &app->localFFTPlan_inverse->axes[0][l];
				resFFT = VkFFTUpdateBufferSet(app, app->localFFTPlan_inverse, axis, 0, l, 1);
				if (resFFT != VKFFT_SUCCESS) return resFFT;

#if(VKFFT_BACKEND==0)
				vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
				vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
				uint64_t dispatchBlock[3];
				if (l == 0) {
					if (app->localFFTPlan_inverse->numAxisUploads[0] > 2) {
						dispatchBlock[0] = (uint64_t)ceil((uint64_t)ceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)axis->axisBlock[1]) / (double)app->localFFTPlan_inverse->axisSplit[0][1]) * app->localFFTPlan_inverse->axisSplit[0][1];
						dispatchBlock[1] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1];
					}
					else {
						if (app->localFFTPlan_inverse->numAxisUploads[0] > 1) {
							dispatchBlock[0] = (uint64_t)ceil((uint64_t)ceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)axis->axisBlock[1]));
							dispatchBlock[1] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1];
						}
						else {
							dispatchBlock[0] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i;
							dispatchBlock[1] = (uint64_t)ceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1] / (double)axis->axisBlock[1]);
						}
					}
				}
				else {
					dispatchBlock[0] = (uint64_t)ceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)axis->axisBlock[0]);
					dispatchBlock[1] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1];
				}
				dispatchBlock[2] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][2] * app->configuration.coordinateFeatures * app->configuration.numberKernels;
				if (axis->specializationConstants.mergeSequencesR2C == 1) dispatchBlock[1] = (uint64_t)ceil(dispatchBlock[1] / 2.0);
				//if (app->configuration.performZeropadding[1]) dispatchBlock[1] = (uint64_t)ceil(dispatchBlock[1] / 2.0);
				//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = (uint64_t)ceil(dispatchBlock[2] / 2.0);
				resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
				if (resFFT != VKFFT_SUCCESS) return resFFT;
				printDebugInformation(app, axis);
				resFFT = VkFFTSync(app);
				if (resFFT != VKFFT_SUCCESS) return resFFT;
			}
		}
		if (app->configuration.FFTdim == 1) {
			for (int64_t l = (int64_t)1; l < (int64_t)app->localFFTPlan_inverse->numAxisUploads[0]; l++) {
				VkFFTAxis* axis = &app->localFFTPlan_inverse->axes[0][l];
				resFFT = VkFFTUpdateBufferSet(app, app->localFFTPlan_inverse, axis, 0, l, 1);
				if (resFFT != VKFFT_SUCCESS) return resFFT;

#if(VKFFT_BACKEND==0)
				vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
				vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
				uint64_t dispatchBlock[3];
				dispatchBlock[0] = (uint64_t)ceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / (double)axis->axisBlock[0] * app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1] / (double)axis->specializationConstants.fftDim.data.i);
				dispatchBlock[1] = 1;
				dispatchBlock[2] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][2] * app->configuration.coordinateFeatures * app->configuration.numberKernels;
				//if (app->configuration.mergeSequencesR2C == 1) dispatchBlock[0] = (uint64_t)ceil(dispatchBlock[0] / 2.0);
				//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = (uint64_t)ceil(dispatchBlock[2] / 2.0);
				resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
				if (resFFT != VKFFT_SUCCESS) return resFFT;
				printDebugInformation(app, axis);
				resFFT = VkFFTSync(app);
				if (resFFT != VKFFT_SUCCESS) return resFFT;
			}
		}
	}

	if (inverse == 1) {
		//we start from axis 2 and go back to axis 0
		//FFT axis 2
		if (app->configuration.FFTdim > 2) {
			if (!app->configuration.omitDimension[2]) {
				for (int64_t l = (int64_t)app->localFFTPlan_inverse->numAxisUploads[2] - 1; l >= 0; l--) {
					//if ((!app->configuration.reorderFourStep) && (!app->useBluesteinFFT[2])) l = app->localFFTPlan_inverse->numAxisUploads[2] - 1 - l;
					VkFFTAxis* axis = &app->localFFTPlan_inverse->axes[2][l];
					resFFT = VkFFTUpdateBufferSet(app, app->localFFTPlan_inverse, axis, 2, l, 1);
					if (resFFT != VKFFT_SUCCESS) return resFFT;

#if(VKFFT_BACKEND==0)
					vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
					vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
					uint64_t dispatchBlock[3];
					dispatchBlock[0] = (uint64_t)ceil(localSize0[2] / (double)axis->axisBlock[0] * app->localFFTPlan_inverse->actualFFTSizePerAxis[2][2] / (double)axis->specializationConstants.fftDim.data.i);
					dispatchBlock[1] = 1;
					dispatchBlock[2] = app->localFFTPlan_inverse->actualFFTSizePerAxis[2][1] * app->configuration.coordinateFeatures * app->configuration.numberBatches;
					//if (app->configuration.performZeropaddingInverse[0]) dispatchBlock[0] = (uint64_t)ceil(dispatchBlock[0] / 2.0);
					//if (app->configuration.performZeropaddingInverse[1]) dispatchBlock[1] = (uint64_t)ceil(dispatchBlock[1] / 2.0);

					//if (app->configuration.mergeSequencesR2C == 1) dispatchBlock[0] = (uint64_t)ceil(dispatchBlock[0] / 2.0);
					resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
					if (resFFT != VKFFT_SUCCESS) return resFFT;
					printDebugInformation(app, axis);
					resFFT = VkFFTSync(app);
					if (resFFT != VKFFT_SUCCESS) return resFFT;
					//if ((!app->configuration.reorderFourStep) && (!app->useBluesteinFFT[2])) l = app->localFFTPlan_inverse->numAxisUploads[2] - 1 - l;
				}
				if (app->useBluesteinFFT[2] && (app->localFFTPlan_inverse->numAxisUploads[2] > 1)) {
					for (int64_t l = 1; l < (int64_t)app->localFFTPlan_inverse->numAxisUploads[2]; l++) {
						VkFFTAxis* axis = &app->localFFTPlan_inverse->inverseBluesteinAxes[2][l];
						resFFT = VkFFTUpdateBufferSet(app, app->localFFTPlan_inverse, axis, 2, l, 1);
						if (resFFT != VKFFT_SUCCESS) return resFFT;
#if(VKFFT_BACKEND==0)
						vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
						vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
						uint64_t dispatchBlock[3];
						dispatchBlock[0] = (uint64_t)ceil(localSize0[2] / (double)axis->axisBlock[0] * app->localFFTPlan_inverse->actualFFTSizePerAxis[2][2] / (double)axis->specializationConstants.fftDim.data.i);
						dispatchBlock[1] = 1;
						dispatchBlock[2] = app->localFFTPlan_inverse->actualFFTSizePerAxis[2][1] * app->configuration.coordinateFeatures * app->configuration.numberBatches;

						//if (app->configuration.performZeropadding[1]) dispatchBlock[1] = (uint64_t)ceil(dispatchBlock[1] / 2.0);
						//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = (uint64_t)ceil(dispatchBlock[2] / 2.0);
						resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
						if (resFFT != VKFFT_SUCCESS) return resFFT;
						printDebugInformation(app, axis);
						resFFT = VkFFTSync(app);
						if (resFFT != VKFFT_SUCCESS) return resFFT;
					}
				}
			}
		}
		if (app->configuration.FFTdim > 1) {

			//FFT axis 1
			if (!app->configuration.omitDimension[1]) {
				for (int64_t l = (int64_t)app->localFFTPlan_inverse->numAxisUploads[1] - 1; l >= 0; l--) {
					//if ((!app->configuration.reorderFourStep) && (!app->useBluesteinFFT[1])) l = app->localFFTPlan_inverse->numAxisUploads[1] - 1 - l;
					VkFFTAxis* axis = &app->localFFTPlan_inverse->axes[1][l];
					resFFT = VkFFTUpdateBufferSet(app, app->localFFTPlan_inverse, axis, 1, l, 1);
					if (resFFT != VKFFT_SUCCESS) return resFFT;

#if(VKFFT_BACKEND==0)
					vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
					vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
					uint64_t dispatchBlock[3];
					dispatchBlock[0] = (uint64_t)ceil(localSize0[1] / (double)axis->axisBlock[0] * app->localFFTPlan_inverse->actualFFTSizePerAxis[1][1] / (double)axis->specializationConstants.fftDim.data.i);
					dispatchBlock[1] = 1;
					dispatchBlock[2] = app->localFFTPlan_inverse->actualFFTSizePerAxis[1][2] * app->configuration.coordinateFeatures * app->configuration.numberBatches;
					//if (app->configuration.mergeSequencesR2C == 1) dispatchBlock[0] = (uint64_t)ceil(dispatchBlock[0] / 2.0);
					//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = (uint64_t)ceil(dispatchBlock[2] / 2.0);
					//if (app->configuration.performZeropaddingInverse[0]) dispatchBlock[0] = (uint64_t)ceil(dispatchBlock[0] / 2.0);

					resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
					if (resFFT != VKFFT_SUCCESS) return resFFT;
					printDebugInformation(app, axis);
					//if ((!app->configuration.reorderFourStep) && (!app->useBluesteinFFT[1])) l = app->localFFTPlan_inverse->numAxisUploads[1] - 1 - l;
					resFFT = VkFFTSync(app);
					if (resFFT != VKFFT_SUCCESS) return resFFT;
				}
				if (app->useBluesteinFFT[1] && (app->localFFTPlan_inverse->numAxisUploads[1] > 1)) {
					for (int64_t l = 1; l < (int64_t)app->localFFTPlan_inverse->numAxisUploads[1]; l++) {
						VkFFTAxis* axis = &app->localFFTPlan_inverse->inverseBluesteinAxes[1][l];
						resFFT = VkFFTUpdateBufferSet(app, app->localFFTPlan_inverse, axis, 1, l, 1);
						if (resFFT != VKFFT_SUCCESS) return resFFT;
#if(VKFFT_BACKEND==0)
						vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
						vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
						uint64_t dispatchBlock[3];
						dispatchBlock[0] = (uint64_t)ceil(localSize0[1] / (double)axis->axisBlock[0] * app->localFFTPlan_inverse->actualFFTSizePerAxis[1][1] / (double)axis->specializationConstants.fftDim.data.i);
						dispatchBlock[1] = 1;
						dispatchBlock[2] = app->localFFTPlan_inverse->actualFFTSizePerAxis[1][2] * app->configuration.coordinateFeatures * app->configuration.numberBatches;

						//if (app->configuration.performZeropadding[1]) dispatchBlock[1] = (uint64_t)ceil(dispatchBlock[1] / 2.0);
						//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = (uint64_t)ceil(dispatchBlock[2] / 2.0);
						resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
						if (resFFT != VKFFT_SUCCESS) return resFFT;
						printDebugInformation(app, axis);
						resFFT = VkFFTSync(app);
						if (resFFT != VKFFT_SUCCESS) return resFFT;
					}
				}
			}
		}
		if (!app->configuration.omitDimension[0]) {
			if (app->localFFTPlan_inverse->multiUploadR2C) {
				//app->configuration.size[0] /= 2;
				VkFFTAxis* axis = &app->localFFTPlan_inverse->R2Cdecomposition;
				resFFT = VkFFTUpdateBufferSetR2CMultiUploadDecomposition(app, app->localFFTPlan_inverse, axis, 0, 0, 1);
				if (resFFT != VKFFT_SUCCESS) return resFFT;

#if(VKFFT_BACKEND==0)
				vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
				vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
				uint64_t dispatchBlock[3];

				dispatchBlock[0] = (uint64_t)ceil(((app->configuration.size[0] / 2 + 1) * app->configuration.size[1] * app->configuration.size[2]) / (double)(2 * axis->axisBlock[0]));
				dispatchBlock[1] = 1;
				dispatchBlock[2] = app->configuration.coordinateFeatures * axis->specializationConstants.numBatches.data.i;
				resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
				if (resFFT != VKFFT_SUCCESS) return resFFT;
				printDebugInformation(app, axis);

				resFFT = VkFFTSync(app);
				if (resFFT != VKFFT_SUCCESS) return resFFT;
			}
			//FFT axis 0
			for (int64_t l = (int64_t)app->localFFTPlan_inverse->numAxisUploads[0] - 1; l >= 0; l--) {
				//if ((!app->configuration.reorderFourStep) && (!app->useBluesteinFFT[0])) l = app->localFFTPlan_inverse->numAxisUploads[0] - 1 - l;
				VkFFTAxis* axis = &app->localFFTPlan_inverse->axes[0][l];
				resFFT = VkFFTUpdateBufferSet(app, app->localFFTPlan_inverse, axis, 0, l, 1);
				if (resFFT != VKFFT_SUCCESS) return resFFT;
#if(VKFFT_BACKEND==0)
				vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
				vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
				uint64_t dispatchBlock[3];
				if (l == 0) {
					if (app->localFFTPlan_inverse->numAxisUploads[0] > 2) {
						dispatchBlock[0] = (uint64_t)ceil((uint64_t)ceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)axis->axisBlock[1]) / (double)app->localFFTPlan_inverse->axisSplit[0][1]) * app->localFFTPlan_inverse->axisSplit[0][1];
						dispatchBlock[1] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1];
					}
					else {
						if (app->localFFTPlan_inverse->numAxisUploads[0] > 1) {
							dispatchBlock[0] = (uint64_t)ceil((uint64_t)ceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)axis->axisBlock[1]));
							dispatchBlock[1] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1];
						}
						else {
							dispatchBlock[0] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i;
							dispatchBlock[1] = (uint64_t)ceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1] / (double)axis->axisBlock[1]);
						}
					}
				}
				else {
					dispatchBlock[0] = (uint64_t)ceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)axis->axisBlock[0]);
					dispatchBlock[1] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1];
				}
				dispatchBlock[2] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][2] * app->configuration.coordinateFeatures * app->configuration.numberBatches;
				if (axis->specializationConstants.mergeSequencesR2C == 1) dispatchBlock[1] = (uint64_t)ceil(dispatchBlock[1] / 2.0);
				//if (app->configuration.performZeropadding[1]) dispatchBlock[1] = (uint64_t)ceil(dispatchBlock[1] / 2.0);
				//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = (uint64_t)ceil(dispatchBlock[2] / 2.0);
				resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
				if (resFFT != VKFFT_SUCCESS) return resFFT;
				printDebugInformation(app, axis);
				//if ((!app->configuration.reorderFourStep) && (!app->useBluesteinFFT[0])) l = app->localFFTPlan_inverse->numAxisUploads[0] - 1 - l;
				resFFT = VkFFTSync(app);
				if (resFFT != VKFFT_SUCCESS) return resFFT;
			}
			if (app->useBluesteinFFT[0] && (app->localFFTPlan_inverse->numAxisUploads[0] > 1)) {
				for (int64_t l = 1; l < (int64_t)app->localFFTPlan_inverse->numAxisUploads[0]; l++) {
					VkFFTAxis* axis = &app->localFFTPlan_inverse->inverseBluesteinAxes[0][l];
					resFFT = VkFFTUpdateBufferSet(app, app->localFFTPlan_inverse, axis, 0, l, 1);
					if (resFFT != VKFFT_SUCCESS) return resFFT;

#if(VKFFT_BACKEND==0)
					vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
					vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
					uint64_t dispatchBlock[3];
					if (l == 0) {
						if (app->localFFTPlan_inverse->numAxisUploads[0] > 2) {
							dispatchBlock[0] = (uint64_t)ceil((uint64_t)ceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)axis->axisBlock[1]) / (double)app->localFFTPlan_inverse->axisSplit[0][1]) * app->localFFTPlan_inverse->axisSplit[0][1];
							dispatchBlock[1] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1];
						}
						else {
							if (app->localFFTPlan_inverse->numAxisUploads[0] > 1) {
								dispatchBlock[0] = (uint64_t)ceil((uint64_t)ceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)axis->axisBlock[1]));
								dispatchBlock[1] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1];
							}
							else {
								dispatchBlock[0] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i;
								dispatchBlock[1] = (uint64_t)ceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1] / (double)axis->axisBlock[1]);
							}
						}
					}
					else {
						dispatchBlock[0] = (uint64_t)ceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)axis->axisBlock[0]);
						dispatchBlock[1] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1];
					}
					dispatchBlock[2] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][2] * app->configuration.coordinateFeatures * app->configuration.numberBatches;
					if (axis->specializationConstants.mergeSequencesR2C == 1) dispatchBlock[1] = (uint64_t)ceil(dispatchBlock[1] / 2.0);
					//if (app->configuration.performZeropadding[1]) dispatchBlock[1] = (uint64_t)ceil(dispatchBlock[1] / 2.0);
					//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = (uint64_t)ceil(dispatchBlock[2] / 2.0);
					resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
					if (resFFT != VKFFT_SUCCESS) return resFFT;
					printDebugInformation(app, axis);
					resFFT = VkFFTSync(app);
					if (resFFT != VKFFT_SUCCESS) return resFFT;
				}
			}
		}
		//if (app->localFFTPlan_inverse->multiUploadR2C) app->configuration.size[0] *= 2;

	}
	return resFFT;
}


#endif
