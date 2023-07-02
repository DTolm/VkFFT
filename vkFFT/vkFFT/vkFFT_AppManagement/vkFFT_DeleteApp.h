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
#ifndef VKFFT_DELETEAPP_H
#define VKFFT_DELETEAPP_H
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT/vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_DeletePlan.h"
#include "vkFFT/vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_UpdateBuffers.h"

static inline void deleteVkFFT(VkFFTApplication* app) {
#if(VKFFT_BACKEND==0)
	if (app->configuration.isCompilerInitialized) {
		glslang_finalize_process();
		app->configuration.isCompilerInitialized = 0;
	}
#elif(VKFFT_BACKEND==1)
	if (app->configuration.num_streams > 1) {
		cudaError_t res_t = cudaSuccess;
		for (uint64_t i = 0; i < app->configuration.num_streams; i++) {
			if (app->configuration.stream_event[i] != 0) {
				res_t = cudaEventDestroy(app->configuration.stream_event[i]);
				if (res_t == cudaSuccess) app->configuration.stream_event[i] = 0;
			}
		}
		if (app->configuration.stream_event != 0) {
			free(app->configuration.stream_event);
			app->configuration.stream_event = 0;
		}
	}
#elif(VKFFT_BACKEND==2)
	if (app->configuration.num_streams > 1) {
		hipError_t res_t = hipSuccess;
		for (uint64_t i = 0; i < app->configuration.num_streams; i++) {
			if (app->configuration.stream_event[i] != 0) {
				res_t = hipEventDestroy(app->configuration.stream_event[i]);
				if (res_t == hipSuccess) app->configuration.stream_event[i] = 0;
			}
		}
		if (app->configuration.stream_event != 0) {
			free(app->configuration.stream_event);
			app->configuration.stream_event = 0;
		}
	}
#endif
	if (app->numRaderFFTPrimes) {
		for (uint64_t i = 0; i < app->numRaderFFTPrimes; i++) {
			free(app->raderFFTkernel[i]);
			app->raderFFTkernel[i] = 0;
		}
	}
	if (!app->configuration.userTempBuffer) {
		if (app->configuration.allocateTempBuffer) {
			app->configuration.allocateTempBuffer = 0;
#if(VKFFT_BACKEND==0)
			if (app->configuration.tempBuffer[0] != 0) {
				vkDestroyBuffer(app->configuration.device[0], app->configuration.tempBuffer[0], 0);
				app->configuration.tempBuffer[0] = 0;
			}
			if (app->configuration.tempBufferDeviceMemory != 0) {
				vkFreeMemory(app->configuration.device[0], app->configuration.tempBufferDeviceMemory, 0);
				app->configuration.tempBufferDeviceMemory = 0;
			}
#elif(VKFFT_BACKEND==1)
			cudaError_t res_t = cudaSuccess;
			if (app->configuration.tempBuffer[0] != 0) {
				res_t = cudaFree(app->configuration.tempBuffer[0]);
				if (res_t == cudaSuccess) app->configuration.tempBuffer[0] = 0;
			}
#elif(VKFFT_BACKEND==2)
			hipError_t res_t = hipSuccess;
			if (app->configuration.tempBuffer[0] != 0) {
				res_t = hipFree(app->configuration.tempBuffer[0]);
				if (res_t == hipSuccess) app->configuration.tempBuffer[0] = 0;
			}
#elif(VKFFT_BACKEND==3)
			cl_int res = 0;
			if (app->configuration.tempBuffer[0] != 0) {
				res = clReleaseMemObject(app->configuration.tempBuffer[0]);
				if (res == 0) app->configuration.tempBuffer[0] = 0;
			}
#elif(VKFFT_BACKEND==4)
			ze_result_t res = ZE_RESULT_SUCCESS;
			if (app->configuration.tempBuffer[0] != 0) {
				res = zeMemFree(app->configuration.context[0], app->configuration.tempBuffer[0]);
				if (res == ZE_RESULT_SUCCESS) app->configuration.tempBuffer[0] = 0;
			}
#elif(VKFFT_BACKEND==5)
			if (app->configuration.tempBuffer[0] != 0) {
				((MTL::Buffer*)app->configuration.tempBuffer[0])->release();
			}
#endif
			if (app->configuration.tempBuffer != 0) {
				free(app->configuration.tempBuffer);
				app->configuration.tempBuffer = 0;
			}
		}
		if (app->configuration.tempBufferSize != 0) {
			free(app->configuration.tempBufferSize);
			app->configuration.tempBufferSize = 0;
		}
	}
	for (uint64_t i = 0; i < app->configuration.FFTdim; i++) {
		if (app->configuration.useRaderUintLUT) {
			for (uint64_t j = 0; j < 4; j++) {
				if (app->bufferRaderUintLUT[i][j]) {
#if(VKFFT_BACKEND==0)
					vkDestroyBuffer(app->configuration.device[0], app->bufferRaderUintLUT[i][j], 0);
					app->bufferRaderUintLUT[i][j] = 0;
					vkFreeMemory(app->configuration.device[0], app->bufferRaderUintLUTDeviceMemory[i][j], 0);
					app->bufferRaderUintLUTDeviceMemory[i][j] = 0;
#elif(VKFFT_BACKEND==1)
					cudaError_t res_t = cudaSuccess;
					res_t = cudaFree(app->bufferRaderUintLUT[i][j]);
					if (res_t == cudaSuccess) app->bufferRaderUintLUT[i][j] = 0;
#elif(VKFFT_BACKEND==2)
					hipError_t res_t = hipSuccess;
					res_t = hipFree(app->bufferRaderUintLUT[i][j]);
					if (res_t == hipSuccess) app->bufferRaderUintLUT[i][j] = 0;
#elif(VKFFT_BACKEND==3)
					cl_int res = 0;
					res = clReleaseMemObject(app->bufferRaderUintLUT[i][j]);
					if (res == 0) app->bufferRaderUintLUT[i][j] = 0;
#elif(VKFFT_BACKEND==4)
					ze_result_t res = ZE_RESULT_SUCCESS;
					res = zeMemFree(app->configuration.context[0], app->bufferRaderUintLUT[i][j]);
					if (res == ZE_RESULT_SUCCESS) app->bufferRaderUintLUT[i][j] = 0;
#elif(VKFFT_BACKEND==5)
					if (app->bufferRaderUintLUT[i][j] != 0) {
						((MTL::Buffer*)app->bufferRaderUintLUT[i][j])->release();
						//free(app->bufferRaderUintLUT[i][j]);
						app->bufferRaderUintLUT[i][j] = 0;
					}
#endif
				}
			}
		}
		if (app->useBluesteinFFT[i]) {
#if(VKFFT_BACKEND==0)
			if (app->bufferBluestein[i] != 0) {
				vkDestroyBuffer(app->configuration.device[0], app->bufferBluestein[i], 0);
				app->bufferBluestein[i] = 0;
			}
			if (app->bufferBluesteinDeviceMemory[i] != 0) {
				vkFreeMemory(app->configuration.device[0], app->bufferBluesteinDeviceMemory[i], 0);
				app->bufferBluesteinDeviceMemory[i] = 0;
			}
			if (app->bufferBluesteinFFT[i] != 0) {
				vkDestroyBuffer(app->configuration.device[0], app->bufferBluesteinFFT[i], 0);
				app->bufferBluesteinFFT[i] = 0;
			}
			if (app->bufferBluesteinFFTDeviceMemory[i] != 0) {
				vkFreeMemory(app->configuration.device[0], app->bufferBluesteinFFTDeviceMemory[i], 0);
				app->bufferBluesteinFFTDeviceMemory[i] = 0;
			}
			if (app->bufferBluesteinIFFT[i] != 0) {
				vkDestroyBuffer(app->configuration.device[0], app->bufferBluesteinIFFT[i], 0);
				app->bufferBluesteinIFFT[i] = 0;
			}
			if (app->bufferBluesteinIFFTDeviceMemory[i] != 0) {
				vkFreeMemory(app->configuration.device[0], app->bufferBluesteinIFFTDeviceMemory[i], 0);
				app->bufferBluesteinIFFTDeviceMemory[i] = 0;
			}
#elif(VKFFT_BACKEND==1)
			cudaError_t res_t = cudaSuccess;
			if (app->bufferBluestein[i] != 0) {
				res_t = cudaFree(app->bufferBluestein[i]);
				if (res_t == cudaSuccess) app->bufferBluestein[i] = 0;
			}
			if (app->bufferBluesteinFFT[i] != 0) {
				res_t = cudaFree(app->bufferBluesteinFFT[i]);
				if (res_t == cudaSuccess) app->bufferBluesteinFFT[i] = 0;
			}
			if (app->bufferBluesteinIFFT[i] != 0) {
				res_t = cudaFree(app->bufferBluesteinIFFT[i]);
				if (res_t == cudaSuccess) app->bufferBluesteinIFFT[i] = 0;
			}
#elif(VKFFT_BACKEND==2)
			hipError_t res_t = hipSuccess;
			if (app->bufferBluestein[i] != 0) {
				res_t = hipFree(app->bufferBluestein[i]);
				if (res_t == hipSuccess) app->bufferBluestein[i] = 0;
			}
			if (app->bufferBluesteinFFT[i] != 0) {
				res_t = hipFree(app->bufferBluesteinFFT[i]);
				if (res_t == hipSuccess) app->bufferBluesteinFFT[i] = 0;
			}
			if (app->bufferBluesteinIFFT[i] != 0) {
				res_t = hipFree(app->bufferBluesteinIFFT[i]);
				if (res_t == hipSuccess) app->bufferBluesteinIFFT[i] = 0;
			}
#elif(VKFFT_BACKEND==3)
			cl_int res = 0;
			if (app->bufferBluestein[i] != 0) {
				res = clReleaseMemObject(app->bufferBluestein[i]);
				if (res == 0) app->bufferBluestein[i] = 0;
			}
			if (app->bufferBluesteinFFT[i] != 0) {
				res = clReleaseMemObject(app->bufferBluesteinFFT[i]);
				if (res == 0) app->bufferBluesteinFFT[i] = 0;
			}
			if (app->bufferBluesteinIFFT[i] != 0) {
				res = clReleaseMemObject(app->bufferBluesteinIFFT[i]);
				if (res == 0) app->bufferBluesteinIFFT[i] = 0;
			}
#elif(VKFFT_BACKEND==4)
			ze_result_t res = ZE_RESULT_SUCCESS;
			if (app->bufferBluestein[i] != 0) {
				res = zeMemFree(app->configuration.context[0], app->bufferBluestein[i]);
				if (res == ZE_RESULT_SUCCESS) app->bufferBluestein[i] = 0;
			}
			if (app->bufferBluesteinFFT[i] != 0) {
				res = zeMemFree(app->configuration.context[0], app->bufferBluesteinFFT[i]);
				if (res == ZE_RESULT_SUCCESS) app->bufferBluesteinFFT[i] = 0;
			}
			if (app->bufferBluesteinIFFT[i] != 0) {
				res = zeMemFree(app->configuration.context[0], app->bufferBluesteinIFFT[i]);
				if (res == ZE_RESULT_SUCCESS) app->bufferBluesteinIFFT[i] = 0;
			}
#elif(VKFFT_BACKEND==5)
			if (app->bufferBluestein[i] != 0) {
				((MTL::Buffer*)app->bufferBluestein[i])->release();
				//free(app->bufferBluestein[i]);
				app->bufferBluestein[i] = 0;
			}
			if (app->bufferBluesteinFFT[i] != 0) {
				((MTL::Buffer*)app->bufferBluesteinFFT[i])->release();
				//free(app->bufferBluesteinFFT[i]);
				app->bufferBluesteinFFT[i] = 0;
			}
			if (app->bufferBluesteinIFFT[i] != 0) {
				((MTL::Buffer*)app->bufferBluesteinIFFT[i])->release();
				//free(app->bufferBluesteinIFFT[i]);
				app->bufferBluesteinIFFT[i] = 0;
			}
#endif
		}
	}
	if (!app->configuration.makeInversePlanOnly) {
		if (app->localFFTPlan != 0) {
			for (uint64_t i = 0; i < app->configuration.FFTdim; i++) {
				if (app->localFFTPlan->numAxisUploads[i] > 0) {
					for (uint64_t j = 0; j < app->localFFTPlan->numAxisUploads[i]; j++)
						deleteAxis(app, &app->localFFTPlan->axes[i][j]);
				}
			}
			if (app->localFFTPlan->multiUploadR2C) {
				deleteAxis(app, &app->localFFTPlan->R2Cdecomposition);
			}
			if (app->localFFTPlan != 0) {
				free(app->localFFTPlan);
				app->localFFTPlan = 0;
			}
		}
	}
	if (!app->configuration.makeForwardPlanOnly) {
		if (app->localFFTPlan_inverse != 0) {
			for (uint64_t i = 0; i < app->configuration.FFTdim; i++) {
				if (app->localFFTPlan_inverse->numAxisUploads[i] > 0) {
					for (uint64_t j = 0; j < app->localFFTPlan_inverse->numAxisUploads[i]; j++)
						deleteAxis(app, &app->localFFTPlan_inverse->axes[i][j]);
				}
			}
			if (app->localFFTPlan_inverse->multiUploadR2C) {
				deleteAxis(app, &app->localFFTPlan_inverse->R2Cdecomposition);
			}
			if (app->localFFTPlan_inverse != 0) {
				free(app->localFFTPlan_inverse);
				app->localFFTPlan_inverse = 0;
			}
		}
	}
	if (app->configuration.saveApplicationToString) {
		if (app->saveApplicationString != 0) {
			free(app->saveApplicationString);
			app->saveApplicationString = 0;
		}
		for (uint64_t i = 0; i < app->configuration.FFTdim; i++) {
			if (app->applicationBluesteinString[i] != 0) {
				free(app->applicationBluesteinString[i]);
				app->applicationBluesteinString[i] = 0;
			}
		}
	}
	if (app->configuration.autoCustomBluesteinPaddingPattern) {
		if (app->configuration.primeSizes != 0) {
			free(app->configuration.primeSizes);
			app->configuration.primeSizes = 0;
		}
		if (app->configuration.paddedSizes != 0) {
			free(app->configuration.paddedSizes);
			app->configuration.paddedSizes = 0;
		}
	}
}
#endif