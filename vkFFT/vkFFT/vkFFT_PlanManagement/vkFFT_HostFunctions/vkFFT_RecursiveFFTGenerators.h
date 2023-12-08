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
#ifndef VKFFT_RECURSIVEFFTGENERATORS_H
#define VKFFT_RECURSIVEFFTGENERATORS_H
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"

#include "vkFFT/vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_ManageMemory.h"
#include "vkFFT/vkFFT_AppManagement/vkFFT_InitializeApp.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_MathUtils/vkFFT_MathUtils.h"
#ifdef VkFFT_use_FP128_Bluestein_RaderFFT
#include "fftw3.h"
#endif

static inline VkFFTResult initializeVkFFT(VkFFTApplication* app, VkFFTConfiguration inputLaunchConfiguration);

static inline VkFFTResult VkFFTGeneratePhaseVectors(VkFFTApplication* app, VkFFTPlan* FFTPlan, pfUINT axis_id) {
	//generate two arrays used for Bluestein convolution and post-convolution multiplication
	VkFFTResult resFFT = VKFFT_SUCCESS;
	pfUINT bufferSize = (pfUINT)sizeof(float) * 2 * FFTPlan->actualFFTSizePerAxis[axis_id][axis_id];
	if (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory) bufferSize *= sizeof(double) / sizeof(float);
	if (app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) bufferSize *= 4;
	app->bufferBluesteinSize[axis_id] = bufferSize;
#if(VKFFT_BACKEND==0)
	VkResult res = VK_SUCCESS;
	resFFT = allocateBufferVulkan(app, &app->bufferBluestein[axis_id], &app->bufferBluesteinDeviceMemory[axis_id], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
	if (resFFT != VKFFT_SUCCESS) return resFFT;
	if (!app->configuration.makeInversePlanOnly) {
		resFFT = allocateBufferVulkan(app, &app->bufferBluesteinFFT[axis_id], &app->bufferBluesteinFFTDeviceMemory[axis_id], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
		if (resFFT != VKFFT_SUCCESS) return resFFT;
	}
	if (!app->configuration.makeForwardPlanOnly) {
		resFFT = allocateBufferVulkan(app, &app->bufferBluesteinIFFT[axis_id], &app->bufferBluesteinIFFTDeviceMemory[axis_id], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
		if (resFFT != VKFFT_SUCCESS) return resFFT;
	}
#elif(VKFFT_BACKEND==1)
	cudaError_t res = cudaSuccess;
	res = cudaMalloc((void**)&app->bufferBluestein[axis_id], bufferSize);
	if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
	if (!app->configuration.makeInversePlanOnly) {
		res = cudaMalloc((void**)&app->bufferBluesteinFFT[axis_id], bufferSize);
		if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
	}
	if (!app->configuration.makeForwardPlanOnly) {
		res = cudaMalloc((void**)&app->bufferBluesteinIFFT[axis_id], bufferSize);
		if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
	}
#elif(VKFFT_BACKEND==2)
	hipError_t res = hipSuccess;
	res = hipMalloc((void**)&app->bufferBluestein[axis_id], bufferSize);
	if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
	if (!app->configuration.makeInversePlanOnly) {
		res = hipMalloc((void**)&app->bufferBluesteinFFT[axis_id], bufferSize);
		if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
	}
	if (!app->configuration.makeForwardPlanOnly) {
		res = hipMalloc((void**)&app->bufferBluesteinIFFT[axis_id], bufferSize);
		if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
	}
#elif(VKFFT_BACKEND==3)
	cl_int res = CL_SUCCESS;
	app->bufferBluestein[axis_id] = clCreateBuffer(app->configuration.context[0], CL_MEM_READ_WRITE, bufferSize, 0, &res);
	if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
	if (!app->configuration.makeInversePlanOnly) {
		app->bufferBluesteinFFT[axis_id] = clCreateBuffer(app->configuration.context[0], CL_MEM_READ_WRITE, bufferSize, 0, &res);
		if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
	}
	if (!app->configuration.makeForwardPlanOnly) {
		app->bufferBluesteinIFFT[axis_id] = clCreateBuffer(app->configuration.context[0], CL_MEM_READ_WRITE, bufferSize, 0, &res);
		if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
	}
	cl_command_queue commandQueue = clCreateCommandQueue(app->configuration.context[0], app->configuration.device[0], 0, &res);
	if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE;
#elif(VKFFT_BACKEND==4)
	ze_result_t res = ZE_RESULT_SUCCESS;

	ze_device_mem_alloc_desc_t device_desc = VKFFT_ZERO_INIT;
	device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
	res = zeMemAllocDevice(app->configuration.context[0], &device_desc, bufferSize, sizeof(float), app->configuration.device[0], &app->bufferBluestein[axis_id]);
	if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;

	if (!app->configuration.makeInversePlanOnly) {
		res = zeMemAllocDevice(app->configuration.context[0], &device_desc, bufferSize, sizeof(float), app->configuration.device[0], &app->bufferBluesteinFFT[axis_id]);
		if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
	}
	if (!app->configuration.makeForwardPlanOnly) {
		res = zeMemAllocDevice(app->configuration.context[0], &device_desc, bufferSize, sizeof(float), app->configuration.device[0], &app->bufferBluesteinIFFT[axis_id]);
		if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
	}
#elif(VKFFT_BACKEND==5)
	app->bufferBluestein[axis_id] = app->configuration.device->newBuffer(bufferSize, MTL::ResourceStorageModePrivate);

	if (!app->configuration.makeInversePlanOnly) {
		app->bufferBluesteinFFT[axis_id] = app->configuration.device->newBuffer(bufferSize, MTL::ResourceStorageModePrivate);
	}
	if (!app->configuration.makeForwardPlanOnly) {
		app->bufferBluesteinIFFT[axis_id] = app->configuration.device->newBuffer(bufferSize, MTL::ResourceStorageModePrivate);
	}
#endif
#ifdef VkFFT_use_FP128_Bluestein_RaderFFT
	if (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory) {
		double* phaseVectors_fp64 = (double*)malloc(bufferSize);
		if (!phaseVectors_fp64) {
			return VKFFT_ERROR_MALLOC_FAILED;
		}
		pfLD* phaseVectors_fp128 = (pfLD*)malloc(2 * bufferSize);
		if (!phaseVectors_fp128) {
			free(phaseVectors_fp64);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
		pfLD* phaseVectors_fp128_out = (pfLD*)malloc(2 * bufferSize);
		if (!phaseVectors_fp128) {
			free(phaseVectors_fp64);
			free(phaseVectors_fp128);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
		pfUINT phaseVectorsNonZeroSize = ((((app->configuration.performDCT == 4) || (app->configuration.performDST == 4)) && (app->configuration.size[axis_id] % 2 == 0)) || ((FFTPlan->bigSequenceEvenR2C) && (axis_id == 0))) ? app->configuration.size[axis_id] / 2 : app->configuration.size[axis_id];
		if (app->configuration.performDCT == 1) phaseVectorsNonZeroSize = 2 * app->configuration.size[axis_id] - 2;
		if (app->configuration.performDST == 1) phaseVectorsNonZeroSize = 2 * app->configuration.size[axis_id] + 2;
		pfLD double_PI = pfFPinit("3.14159265358979323846264338327950288419716939937510");
		for (pfUINT i = 0; i < FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]; i++) {
			pfUINT rm = (i * i) % (2 * phaseVectorsNonZeroSize);
			pfLD angle = double_PI * rm / phaseVectorsNonZeroSize;
			phaseVectors_fp128[2 * i] = (i < phaseVectorsNonZeroSize) ? pfcos(angle) : 0;
			phaseVectors_fp128[2 * i + 1] = (i < phaseVectorsNonZeroSize) ? -pfsin(angle) : 0;
		}
		for (pfUINT i = 1; i < phaseVectorsNonZeroSize; i++) {
			phaseVectors_fp128[2 * (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - i)] = phaseVectors_fp128[2 * i];
			phaseVectors_fp128[2 * (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - i) + 1] = phaseVectors_fp128[2 * i + 1];
		}
		if ((FFTPlan->numAxisUploads[axis_id] > 1) && (!app->configuration.makeForwardPlanOnly)) {
			fftwl_plan p;
			p = fftwl_plan_dft_1d((int)(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]), (fftwl_complex*)phaseVectors_fp128, (fftwl_complex*)phaseVectors_fp128_out, -1, FFTW_ESTIMATE);
			fftwl_execute(p);
			fftwl_destroy_plan(p);
			for (pfUINT i = 0; i < FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]; i++) {
				pfUINT out = 0;
				if (FFTPlan->numAxisUploads[axis_id] == 1) {
					out = i;
				}
				else if (FFTPlan->numAxisUploads[axis_id] == 2) {
					out = i / FFTPlan->axisSplit[axis_id][1] + (i % FFTPlan->axisSplit[axis_id][1]) * FFTPlan->axisSplit[axis_id][0];
				}
				else {
					out = (i / FFTPlan->axisSplit[axis_id][2]) / FFTPlan->axisSplit[axis_id][1] + ((i / FFTPlan->axisSplit[axis_id][2]) % FFTPlan->axisSplit[axis_id][1]) * FFTPlan->axisSplit[axis_id][0] + (i % FFTPlan->axisSplit[axis_id][2]) * FFTPlan->axisSplit[axis_id][1] * FFTPlan->axisSplit[axis_id][0];
				}
				phaseVectors_fp64[2 * out] = (double)phaseVectors_fp128_out[2 * i];
				phaseVectors_fp64[2 * out + 1] = (double)phaseVectors_fp128_out[2 * i + 1];
			}
			resFFT = VkFFT_TransferDataFromCPU(app, phaseVectors_fp64, &app->bufferBluesteinIFFT[axis_id], bufferSize);
			if (resFFT != VKFFT_SUCCESS) {
				free(phaseVectors_fp64);
				free(phaseVectors_fp128);
				free(phaseVectors_fp128_out);
				return resFFT;
			}
		}
		for (pfUINT i = 0; i < FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]; i++) {
			phaseVectors_fp128[2 * i + 1] = -phaseVectors_fp128[2 * i + 1];
		}
		for (pfUINT i = 0; i < FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]; i++) {
			phaseVectors_fp64[2 * i] = (double)phaseVectors_fp128[2 * i];
			phaseVectors_fp64[2 * i + 1] = (double)phaseVectors_fp128[2 * i + 1];
		}
		resFFT = VkFFT_TransferDataFromCPU(app, phaseVectors_fp64, &app->bufferBluestein[axis_id], bufferSize);
		if (resFFT != VKFFT_SUCCESS) {
			free(phaseVectors_fp64);
			free(phaseVectors_fp128);
			free(phaseVectors_fp128_out);
			return resFFT;
		}
		if (!app->configuration.makeInversePlanOnly) {
			fftwl_plan p;
			p = fftwl_plan_dft_1d((int)(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]), (fftwl_complex*)phaseVectors_fp128, (fftwl_complex*)phaseVectors_fp128_out, -1, FFTW_ESTIMATE);
			fftwl_execute(p);
			fftwl_destroy_plan(p);
			for (pfUINT i = 0; i < FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]; i++) {
				pfUINT out = 0;
				if (FFTPlan->numAxisUploads[axis_id] == 1) {
					out = i;
				}
				else if (FFTPlan->numAxisUploads[axis_id] == 2) {
					out = i / FFTPlan->axisSplit[axis_id][1] + (i % FFTPlan->axisSplit[axis_id][1]) * FFTPlan->axisSplit[axis_id][0];
				}
				else {
					out = (i / FFTPlan->axisSplit[axis_id][2]) / FFTPlan->axisSplit[axis_id][1] + ((i / FFTPlan->axisSplit[axis_id][2]) % FFTPlan->axisSplit[axis_id][1]) * FFTPlan->axisSplit[axis_id][0] + (i % FFTPlan->axisSplit[axis_id][2]) * FFTPlan->axisSplit[axis_id][1] * FFTPlan->axisSplit[axis_id][0];
				}
				phaseVectors_fp64[2 * out] = (double)phaseVectors_fp128_out[2 * i];
				phaseVectors_fp64[2 * out + 1] = (double)phaseVectors_fp128_out[2 * i + 1];
			}
			resFFT = VkFFT_TransferDataFromCPU(app, phaseVectors_fp64, &app->bufferBluesteinFFT[axis_id], bufferSize);
			if (resFFT != VKFFT_SUCCESS) {
				free(phaseVectors_fp64);
				free(phaseVectors_fp128);
				free(phaseVectors_fp128_out);
				return resFFT;
			}
		}
		if ((FFTPlan->numAxisUploads[axis_id] == 1) && (!app->configuration.makeForwardPlanOnly)) {
			fftwl_plan p;
			p = fftwl_plan_dft_1d((int)(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]), (fftwl_complex*)phaseVectors_fp128, (fftwl_complex*)phaseVectors_fp128_out, 1, FFTW_ESTIMATE);
			fftwl_execute(p);
			fftwl_destroy_plan(p);

			for (pfUINT i = 0; i < FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]; i++) {
				phaseVectors_fp64[2 * i] = (double)phaseVectors_fp128_out[2 * i];
				phaseVectors_fp64[2 * i + 1] = (double)phaseVectors_fp128_out[2 * i + 1];
			}
			resFFT = VkFFT_TransferDataFromCPU(app, phaseVectors_fp64, &app->bufferBluesteinIFFT[axis_id], bufferSize);
			if (resFFT != VKFFT_SUCCESS) {
				free(phaseVectors_fp64);
				free(phaseVectors_fp128);
				free(phaseVectors_fp128_out);
				return resFFT;
			}
		}
		free(phaseVectors_fp64);
		free(phaseVectors_fp128);
		free(phaseVectors_fp128_out);
	}
	else {
#endif
		VkFFTApplication kernelPreparationApplication = VKFFT_ZERO_INIT;
		VkFFTConfiguration kernelPreparationConfiguration = VKFFT_ZERO_INIT;
		kernelPreparationConfiguration.FFTdim = 1;
		kernelPreparationConfiguration.size[0] = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id];
		kernelPreparationConfiguration.size[1] = 1;
		kernelPreparationConfiguration.size[2] = 1;
		kernelPreparationConfiguration.doublePrecision = (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory);
		kernelPreparationConfiguration.quadDoubleDoublePrecision = (app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory);
		kernelPreparationConfiguration.useLUT = 1;
		kernelPreparationConfiguration.useLUT_4step = 1;
		kernelPreparationConfiguration.registerBoost = 1;
		kernelPreparationConfiguration.disableReorderFourStep = 1;
		kernelPreparationConfiguration.fixMinRaderPrimeFFT = 17;
		kernelPreparationConfiguration.fixMinRaderPrimeMult = 17;
		kernelPreparationConfiguration.fixMaxRaderPrimeFFT = 17;
		kernelPreparationConfiguration.fixMaxRaderPrimeMult = 17;
		kernelPreparationConfiguration.saveApplicationToString = app->configuration.saveApplicationToString;
		kernelPreparationConfiguration.loadApplicationFromString = app->configuration.loadApplicationFromString;
		kernelPreparationConfiguration.sharedMemorySize = app->configuration.sharedMemorySize;
		if (kernelPreparationConfiguration.loadApplicationFromString) {
			kernelPreparationConfiguration.loadApplicationString = (void*)((char*)app->configuration.loadApplicationString + app->currentApplicationStringPos);
		}
		kernelPreparationConfiguration.performBandwidthBoost = (app->configuration.performBandwidthBoost > 0) ? app->configuration.performBandwidthBoost : 1;
		if (axis_id == 0) kernelPreparationConfiguration.performBandwidthBoost = 0;
		if (axis_id > 0) kernelPreparationConfiguration.considerAllAxesStrided = 1;
		if (app->configuration.tempBuffer) {
			kernelPreparationConfiguration.userTempBuffer = 1;
			kernelPreparationConfiguration.tempBuffer = app->configuration.tempBuffer;
			kernelPreparationConfiguration.tempBufferSize = app->configuration.tempBufferSize;
			kernelPreparationConfiguration.tempBufferNum = app->configuration.tempBufferNum;
		}
		kernelPreparationConfiguration.device = app->configuration.device;
#if(VKFFT_BACKEND==0)
		kernelPreparationConfiguration.queue = app->configuration.queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
		kernelPreparationConfiguration.fence = app->configuration.fence;
		kernelPreparationConfiguration.commandPool = app->configuration.commandPool;
		kernelPreparationConfiguration.physicalDevice = app->configuration.physicalDevice;
		kernelPreparationConfiguration.isCompilerInitialized = 1;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization
		if (app->configuration.tempBuffer) {
			kernelPreparationConfiguration.tempBufferDeviceMemory = app->configuration.tempBufferDeviceMemory;
		}
		if (app->configuration.stagingBuffer != 0)	kernelPreparationConfiguration.stagingBuffer = app->configuration.stagingBuffer;
		if (app->configuration.stagingBufferMemory != 0)	kernelPreparationConfiguration.stagingBufferMemory = app->configuration.stagingBufferMemory;
#elif(VKFFT_BACKEND==3)
		kernelPreparationConfiguration.context = app->configuration.context;
#elif(VKFFT_BACKEND==4)
		kernelPreparationConfiguration.context = app->configuration.context;
		kernelPreparationConfiguration.commandQueue = app->configuration.commandQueue;
		kernelPreparationConfiguration.commandQueueID = app->configuration.commandQueueID;
#elif(VKFFT_BACKEND==5)
		kernelPreparationConfiguration.device = app->configuration.device;
		kernelPreparationConfiguration.queue = app->configuration.queue;
#endif			

		kernelPreparationConfiguration.inputBufferSize = &app->bufferBluesteinSize[axis_id];
		kernelPreparationConfiguration.bufferSize = &app->bufferBluesteinSize[axis_id];
		kernelPreparationConfiguration.isInputFormatted = 1;
		resFFT = initializeVkFFT(&kernelPreparationApplication, kernelPreparationConfiguration);
		if (resFFT != VKFFT_SUCCESS) return resFFT;
		if (kernelPreparationConfiguration.loadApplicationFromString) {
			app->currentApplicationStringPos += kernelPreparationApplication.currentApplicationStringPos;
		}
		void* phaseVectors = malloc(bufferSize);
		if (!phaseVectors) {
			deleteVkFFT(&kernelPreparationApplication);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
		pfUINT phaseVectorsNonZeroSize = ((((app->configuration.performDCT == 4) || (app->configuration.performDST == 4)) && (app->configuration.size[axis_id] % 2 == 0)) || ((FFTPlan->bigSequenceEvenR2C) && (axis_id == 0))) ? app->configuration.size[axis_id] / 2 : app->configuration.size[axis_id];
		if (app->configuration.performDCT == 1) phaseVectorsNonZeroSize = 2 * app->configuration.size[axis_id] - 2;
		if (app->configuration.performDST == 1) phaseVectorsNonZeroSize = 2 * app->configuration.size[axis_id] + 2;
		if ((FFTPlan->numAxisUploads[axis_id] > 1) && (!app->configuration.makeForwardPlanOnly)) {
			if (app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) {
				PfContainer in = VKFFT_ZERO_INIT;
				PfContainer temp1 = VKFFT_ZERO_INIT;
				in.type = 22;
				pfLD double_PI = pfFPinit("3.14159265358979323846264338327950288419716939937510");
				double* phaseVectors_cast = (double*)phaseVectors;
				for (pfUINT i = 0; i < FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]; i++) {
					pfUINT rm = (i * i) % (2 * phaseVectorsNonZeroSize);
					pfLD angle = double_PI * rm / phaseVectorsNonZeroSize;
					in.data.d = pfcos(angle);
					PfConvToDoubleDouble(&FFTPlan->axes[axis_id][0].specializationConstants, &temp1, &in);
					phaseVectors_cast[4 * i] = (i < phaseVectorsNonZeroSize) ? (double)temp1.data.dd[0].data.d : 0;
					phaseVectors_cast[4 * i + 1] = (i < phaseVectorsNonZeroSize) ? (double)temp1.data.dd[1].data.d : 0;
					in.data.d = pfsin(angle);
					PfConvToDoubleDouble(&FFTPlan->axes[axis_id][0].specializationConstants, &temp1, &in);
					phaseVectors_cast[4 * i + 2] = (i < phaseVectorsNonZeroSize) ? (double)-temp1.data.dd[0].data.d : 0;
					phaseVectors_cast[4 * i + 3] = (i < phaseVectorsNonZeroSize) ? (double)-temp1.data.dd[1].data.d : 0;
				}
				for (pfUINT i = 1; i < phaseVectorsNonZeroSize; i++) {
					phaseVectors_cast[4 * (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - i)] = phaseVectors_cast[4 * i];
					phaseVectors_cast[4 * (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - i) + 1] = phaseVectors_cast[4 * i + 1];
					phaseVectors_cast[4 * (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - i) + 2] = phaseVectors_cast[4 * i + 2];
					phaseVectors_cast[4 * (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - i) + 3] = phaseVectors_cast[4 * i + 3];
				}
				PfDeallocateContainer(&FFTPlan->axes[axis_id][0].specializationConstants, &temp1);
			}
			else if (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory) {
				pfLD double_PI = pfFPinit("3.14159265358979323846264338327950288419716939937510");
				double* phaseVectors_cast = (double*)phaseVectors;
				for (pfUINT i = 0; i < FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]; i++) {
					pfUINT rm = (i * i) % (2 * phaseVectorsNonZeroSize);
					pfLD angle = double_PI * rm / phaseVectorsNonZeroSize;
					phaseVectors_cast[2 * i] = (i < phaseVectorsNonZeroSize) ? (double)pfcos(angle) : 0;
					phaseVectors_cast[2 * i + 1] = (i < phaseVectorsNonZeroSize) ? (double)-pfsin(angle) : 0;
				}
				for (pfUINT i = 1; i < phaseVectorsNonZeroSize; i++) {
					phaseVectors_cast[2 * (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - i)] = phaseVectors_cast[2 * i];
					phaseVectors_cast[2 * (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - i) + 1] = phaseVectors_cast[2 * i + 1];
				}
			}
			else {
				double double_PI = 3.14159265358979323846264338327950288419716939937510;
				float* phaseVectors_cast = (float*)phaseVectors;
				for (pfUINT i = 0; i < FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]; i++) {
					pfUINT rm = (i * i) % (2 * phaseVectorsNonZeroSize);
					double angle = double_PI * rm / phaseVectorsNonZeroSize;
					phaseVectors_cast[2 * i] = (i < phaseVectorsNonZeroSize) ? (float)pfcos(angle) : 0;
					phaseVectors_cast[2 * i + 1] = (i < phaseVectorsNonZeroSize) ? (float)-pfsin(angle) : 0;
				}
				for (pfUINT i = 1; i < phaseVectorsNonZeroSize; i++) {
					phaseVectors_cast[2 * (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - i)] = phaseVectors_cast[2 * i];
					phaseVectors_cast[2 * (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - i) + 1] = phaseVectors_cast[2 * i + 1];
				}
			}
			resFFT = VkFFT_TransferDataFromCPU(app, phaseVectors, &app->bufferBluestein[axis_id], bufferSize);
			if (resFFT != VKFFT_SUCCESS) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return resFFT;
			}
#if(VKFFT_BACKEND==0)
			{
				VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
				commandBufferAllocateInfo.commandPool = kernelPreparationApplication.configuration.commandPool[0];
				commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
				commandBufferAllocateInfo.commandBufferCount = 1;
				VkCommandBuffer commandBuffer = VKFFT_ZERO_INIT;
				res = vkAllocateCommandBuffers(kernelPreparationApplication.configuration.device[0], &commandBufferAllocateInfo, &commandBuffer);
				if (res != 0) {
					free(phaseVectors);
					deleteVkFFT(&kernelPreparationApplication);
					return VKFFT_ERROR_FAILED_TO_ALLOCATE_COMMAND_BUFFERS;
				}
				VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
				commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
				res = vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
				if (res != 0) {
					free(phaseVectors);
					deleteVkFFT(&kernelPreparationApplication);
					return VKFFT_ERROR_FAILED_TO_BEGIN_COMMAND_BUFFER;
				}
				VkFFTLaunchParams launchParams = VKFFT_ZERO_INIT;
				launchParams.commandBuffer = &commandBuffer;
				launchParams.inputBuffer = &app->bufferBluestein[axis_id];
				launchParams.buffer = &app->bufferBluesteinIFFT[axis_id];
				//Record commands
				resFFT = VkFFTAppend(&kernelPreparationApplication, -1, &launchParams);
				if (resFFT != VKFFT_SUCCESS) {
					free(phaseVectors);
					deleteVkFFT(&kernelPreparationApplication);
					return resFFT;
				}
				res = vkEndCommandBuffer(commandBuffer);
				if (res != 0) {
					free(phaseVectors);
					deleteVkFFT(&kernelPreparationApplication);
					return VKFFT_ERROR_FAILED_TO_END_COMMAND_BUFFER;
				}
				VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
				submitInfo.commandBufferCount = 1;
				submitInfo.pCommandBuffers = &commandBuffer;
				res = vkQueueSubmit(kernelPreparationApplication.configuration.queue[0], 1, &submitInfo, kernelPreparationApplication.configuration.fence[0]);
				if (res != 0) {
					free(phaseVectors);
					deleteVkFFT(&kernelPreparationApplication);
					return VKFFT_ERROR_FAILED_TO_SUBMIT_QUEUE;
				}
				res = vkWaitForFences(kernelPreparationApplication.configuration.device[0], 1, kernelPreparationApplication.configuration.fence, VK_TRUE, 100000000000);
				if (res != 0) {
					free(phaseVectors);
					deleteVkFFT(&kernelPreparationApplication);
					return VKFFT_ERROR_FAILED_TO_WAIT_FOR_FENCES;
				}
				res = vkResetFences(kernelPreparationApplication.configuration.device[0], 1, kernelPreparationApplication.configuration.fence);
				if (res != 0) {
					free(phaseVectors);
					deleteVkFFT(&kernelPreparationApplication);
					return VKFFT_ERROR_FAILED_TO_RESET_FENCES;
				}
				vkFreeCommandBuffers(kernelPreparationApplication.configuration.device[0], kernelPreparationApplication.configuration.commandPool[0], 1, &commandBuffer);
			}
#elif(VKFFT_BACKEND==1)
			VkFFTLaunchParams launchParams = VKFFT_ZERO_INIT;
			launchParams.inputBuffer = &app->bufferBluestein[axis_id];
			launchParams.buffer = &app->bufferBluesteinIFFT[axis_id];
			resFFT = VkFFTAppend(&kernelPreparationApplication, -1, &launchParams);
			if (resFFT != VKFFT_SUCCESS) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return resFFT;
			}
			res = cudaDeviceSynchronize();
			if (res != cudaSuccess) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
			}
#elif(VKFFT_BACKEND==2)
			VkFFTLaunchParams launchParams = VKFFT_ZERO_INIT;
			launchParams.inputBuffer = &app->bufferBluestein[axis_id];
			launchParams.buffer = &app->bufferBluesteinIFFT[axis_id];
			resFFT = VkFFTAppend(&kernelPreparationApplication, -1, &launchParams);
			if (resFFT != VKFFT_SUCCESS) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return resFFT;
			}
			res = hipDeviceSynchronize();
			if (res != hipSuccess) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
			}
#elif(VKFFT_BACKEND==3)
			VkFFTLaunchParams launchParams = VKFFT_ZERO_INIT;
			launchParams.commandQueue = &commandQueue;
			launchParams.inputBuffer = &app->bufferBluestein[axis_id];
			launchParams.buffer = &app->bufferBluesteinIFFT[axis_id];
			resFFT = VkFFTAppend(&kernelPreparationApplication, -1, &launchParams);
			if (resFFT != VKFFT_SUCCESS) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return resFFT;
			}
			res = clFinish(commandQueue);
			if (res != CL_SUCCESS) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
			}
#elif(VKFFT_BACKEND==4)
			ze_command_list_desc_t commandListDescription = VKFFT_ZERO_INIT;
			commandListDescription.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
			ze_command_list_handle_t commandList = VKFFT_ZERO_INIT;
			res = zeCommandListCreate(app->configuration.context[0], app->configuration.device[0], &commandListDescription, &commandList);
			if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
			VkFFTLaunchParams launchParams = VKFFT_ZERO_INIT;
			launchParams.commandList = &commandList;
			launchParams.inputBuffer = &app->bufferBluestein[axis_id];
			launchParams.buffer = &app->bufferBluesteinIFFT[axis_id];
			resFFT = VkFFTAppend(&kernelPreparationApplication, -1, &launchParams);
			if (resFFT != VKFFT_SUCCESS) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return resFFT;
			}
			res = zeCommandListClose(commandList);
			if (res != ZE_RESULT_SUCCESS) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_FAILED_TO_END_COMMAND_BUFFER;
			}
			res = zeCommandQueueExecuteCommandLists(app->configuration.commandQueue[0], 1, &commandList, 0);
			if (res != ZE_RESULT_SUCCESS) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_FAILED_TO_SUBMIT_QUEUE;
			}
			res = zeCommandQueueSynchronize(app->configuration.commandQueue[0], UINT32_MAX);
			if (res != ZE_RESULT_SUCCESS) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
			}
			res = zeCommandListDestroy(commandList);
			if (res != ZE_RESULT_SUCCESS) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_FAILED_TO_DESTROY_COMMAND_LIST;
			}
#elif(VKFFT_BACKEND==5)
			VkFFTLaunchParams launchParams = VKFFT_ZERO_INIT;
			MTL::CommandBuffer* commandBuffer = app->configuration.queue->commandBuffer();
			if (commandBuffer == 0) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
			MTL::ComputeCommandEncoder* commandEncoder = commandBuffer->computeCommandEncoder();
			if (commandEncoder == 0) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;

			launchParams.commandBuffer = commandBuffer;
			launchParams.commandEncoder = commandEncoder;
			launchParams.inputBuffer = &app->bufferBluestein[axis_id];
			launchParams.buffer = &app->bufferBluesteinIFFT[axis_id];
			resFFT = VkFFTAppend(&kernelPreparationApplication, -1, &launchParams);
			if (resFFT != VKFFT_SUCCESS) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return resFFT;
			}
			commandEncoder->endEncoding();
			commandBuffer->commit();
			commandBuffer->waitUntilCompleted();
			commandEncoder->release();
			commandBuffer->release();
#endif
		}
		if ((FFTPlan->numAxisUploads[axis_id] > 1) && (!app->configuration.makeForwardPlanOnly)) {
			if (app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) {
				double* phaseVectors_cast = (double*)phaseVectors;
				for (pfUINT i = 0; i < FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]; i++) {
					phaseVectors_cast[4 * i + 2] = -phaseVectors_cast[4 * i + 2];
					phaseVectors_cast[4 * i + 3] = -phaseVectors_cast[4 * i + 3];
				}
			}
			else if (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory) {
				double* phaseVectors_cast = (double*)phaseVectors;
				for (pfUINT i = 0; i < FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]; i++) {
					phaseVectors_cast[2 * i + 1] = -phaseVectors_cast[2 * i + 1];
				}

			}
			else {
				float* phaseVectors_cast = (float*)phaseVectors;
				for (pfUINT i = 0; i < FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]; i++) {
					phaseVectors_cast[2 * i + 1] = -phaseVectors_cast[2 * i + 1];
				}
			}
		}
		else {
			if (app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) {
				PfContainer in = VKFFT_ZERO_INIT;
				PfContainer temp1 = VKFFT_ZERO_INIT;
				in.type = 22;
				pfLD double_PI = pfFPinit("3.14159265358979323846264338327950288419716939937510");
				double* phaseVectors_cast = (double*)phaseVectors;
				for (pfUINT i = 0; i < FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]; i++) {
					pfUINT rm = (i * i) % (2 * phaseVectorsNonZeroSize);
					pfLD angle = double_PI * rm / phaseVectorsNonZeroSize;
					in.data.d = pfcos(angle);
					PfConvToDoubleDouble(&FFTPlan->axes[axis_id][0].specializationConstants, &temp1, &in);
					phaseVectors_cast[4 * i] = (i < phaseVectorsNonZeroSize) ? (double)temp1.data.dd[0].data.d : 0;
					phaseVectors_cast[4 * i + 1] = (i < phaseVectorsNonZeroSize) ? (double)temp1.data.dd[1].data.d : 0;
					in.data.d = pfsin(angle);
					PfConvToDoubleDouble(&FFTPlan->axes[axis_id][0].specializationConstants, &temp1, &in);
					phaseVectors_cast[4 * i + 2] = (i < phaseVectorsNonZeroSize) ? (double)temp1.data.dd[0].data.d : 0;
					phaseVectors_cast[4 * i + 3] = (i < phaseVectorsNonZeroSize) ? (double)temp1.data.dd[1].data.d : 0;
				}
				for (pfUINT i = 1; i < phaseVectorsNonZeroSize; i++) {
					phaseVectors_cast[4 * (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - i)] = phaseVectors_cast[4 * i];
					phaseVectors_cast[4 * (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - i) + 1] = phaseVectors_cast[4 * i + 1];
					phaseVectors_cast[4 * (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - i) + 2] = phaseVectors_cast[4 * i + 2];
					phaseVectors_cast[4 * (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - i) + 3] = phaseVectors_cast[4 * i + 3];
				}
				PfDeallocateContainer(&FFTPlan->axes[axis_id][0].specializationConstants, &temp1);
			}
			else if (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory) {
				pfLD double_PI = pfFPinit("3.14159265358979323846264338327950288419716939937510");
				double* phaseVectors_cast = (double*)phaseVectors;
				for (pfUINT i = 0; i < FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]; i++) {
					pfUINT rm = (i * i) % (2 * phaseVectorsNonZeroSize);
					pfLD angle = double_PI * rm / phaseVectorsNonZeroSize;
					phaseVectors_cast[2 * i] = (i < phaseVectorsNonZeroSize) ? (double)pfcos(angle) : 0;
					phaseVectors_cast[2 * i + 1] = (i < phaseVectorsNonZeroSize) ? (double)pfsin(angle) : 0;
				}
				for (pfUINT i = 1; i < phaseVectorsNonZeroSize; i++) {
					phaseVectors_cast[2 * (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - i)] = phaseVectors_cast[2 * i];
					phaseVectors_cast[2 * (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - i) + 1] = phaseVectors_cast[2 * i + 1];
				}
			}
			else {
				double double_PI = 3.14159265358979323846264338327950288419716939937510;
				float* phaseVectors_cast = (float*)phaseVectors;
				for (pfUINT i = 0; i < FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]; i++) {
					pfUINT rm = (i * i) % (2 * phaseVectorsNonZeroSize);
					double angle = double_PI * rm / phaseVectorsNonZeroSize;
					phaseVectors_cast[2 * i] = (i < phaseVectorsNonZeroSize) ? (float)pfcos(angle) : 0;
					phaseVectors_cast[2 * i + 1] = (i < phaseVectorsNonZeroSize) ? (float)pfsin(angle) : 0;
				}
				for (pfUINT i = 1; i < phaseVectorsNonZeroSize; i++) {
					phaseVectors_cast[2 * (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - i)] = phaseVectors_cast[2 * i];
					phaseVectors_cast[2 * (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - i) + 1] = phaseVectors_cast[2 * i + 1];
				}
			}
		}
		resFFT = VkFFT_TransferDataFromCPU(app, phaseVectors, &app->bufferBluestein[axis_id], bufferSize);
		if (resFFT != VKFFT_SUCCESS) {
			free(phaseVectors);
			deleteVkFFT(&kernelPreparationApplication);
			return resFFT;
		}
#if(VKFFT_BACKEND==0)
		if (!app->configuration.makeInversePlanOnly) {
			VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
			commandBufferAllocateInfo.commandPool = kernelPreparationApplication.configuration.commandPool[0];
			commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			commandBufferAllocateInfo.commandBufferCount = 1;
			VkCommandBuffer commandBuffer = VKFFT_ZERO_INIT;
			res = vkAllocateCommandBuffers(kernelPreparationApplication.configuration.device[0], &commandBufferAllocateInfo, &commandBuffer);
			if (res != 0) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_FAILED_TO_ALLOCATE_COMMAND_BUFFERS;
			}
			VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
			commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			res = vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
			if (res != 0) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_FAILED_TO_BEGIN_COMMAND_BUFFER;
			}
			VkFFTLaunchParams launchParams = VKFFT_ZERO_INIT;
			launchParams.commandBuffer = &commandBuffer;
			launchParams.inputBuffer = &app->bufferBluestein[axis_id];
			launchParams.buffer = &app->bufferBluesteinFFT[axis_id];
			//Record commands
			resFFT = VkFFTAppend(&kernelPreparationApplication, -1, &launchParams);
			if (resFFT != VKFFT_SUCCESS) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return resFFT;
			}
			res = vkEndCommandBuffer(commandBuffer);
			if (res != 0) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_FAILED_TO_END_COMMAND_BUFFER;
			}
			VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &commandBuffer;
			res = vkQueueSubmit(kernelPreparationApplication.configuration.queue[0], 1, &submitInfo, kernelPreparationApplication.configuration.fence[0]);
			if (res != 0) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_FAILED_TO_SUBMIT_QUEUE;
			}
			res = vkWaitForFences(kernelPreparationApplication.configuration.device[0], 1, kernelPreparationApplication.configuration.fence, VK_TRUE, 100000000000);
			if (res != 0) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_FAILED_TO_WAIT_FOR_FENCES;
			}
			res = vkResetFences(kernelPreparationApplication.configuration.device[0], 1, kernelPreparationApplication.configuration.fence);
			if (res != 0) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_FAILED_TO_RESET_FENCES;
			}
			vkFreeCommandBuffers(kernelPreparationApplication.configuration.device[0], kernelPreparationApplication.configuration.commandPool[0], 1, &commandBuffer);
		}
		if ((FFTPlan->numAxisUploads[axis_id] == 1) && (!app->configuration.makeForwardPlanOnly)) {
			VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
			commandBufferAllocateInfo.commandPool = kernelPreparationApplication.configuration.commandPool[0];
			commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			commandBufferAllocateInfo.commandBufferCount = 1;
			VkCommandBuffer commandBuffer = VKFFT_ZERO_INIT;
			res = vkAllocateCommandBuffers(kernelPreparationApplication.configuration.device[0], &commandBufferAllocateInfo, &commandBuffer);
			if (res != 0) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_FAILED_TO_ALLOCATE_COMMAND_BUFFERS;
			}
			VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
			commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			res = vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
			if (res != 0) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_FAILED_TO_BEGIN_COMMAND_BUFFER;
			}
			VkFFTLaunchParams launchParams = VKFFT_ZERO_INIT;
			launchParams.commandBuffer = &commandBuffer;
			launchParams.inputBuffer = &app->bufferBluestein[axis_id];
			launchParams.buffer = &app->bufferBluesteinIFFT[axis_id];
			//Record commands
			resFFT = VkFFTAppend(&kernelPreparationApplication, 1, &launchParams);
			if (resFFT != VKFFT_SUCCESS) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return resFFT;
			}
			res = vkEndCommandBuffer(commandBuffer);
			if (res != 0) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_FAILED_TO_END_COMMAND_BUFFER;
			}
			VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &commandBuffer;
			res = vkQueueSubmit(kernelPreparationApplication.configuration.queue[0], 1, &submitInfo, kernelPreparationApplication.configuration.fence[0]);
			if (res != 0) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_FAILED_TO_SUBMIT_QUEUE;
			}
			res = vkWaitForFences(kernelPreparationApplication.configuration.device[0], 1, kernelPreparationApplication.configuration.fence, VK_TRUE, 100000000000);
			if (res != 0) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_FAILED_TO_WAIT_FOR_FENCES;
			}
			res = vkResetFences(kernelPreparationApplication.configuration.device[0], 1, kernelPreparationApplication.configuration.fence);
			if (res != 0) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_FAILED_TO_RESET_FENCES;
			}
			vkFreeCommandBuffers(kernelPreparationApplication.configuration.device[0], kernelPreparationApplication.configuration.commandPool[0], 1, &commandBuffer);
		}
#elif(VKFFT_BACKEND==1)
		VkFFTLaunchParams launchParams = VKFFT_ZERO_INIT;
		launchParams.inputBuffer = &app->bufferBluestein[axis_id];
		if (!app->configuration.makeInversePlanOnly) {
			launchParams.buffer = &app->bufferBluesteinFFT[axis_id];
			resFFT = VkFFTAppend(&kernelPreparationApplication, -1, &launchParams);
			if (resFFT != VKFFT_SUCCESS) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return resFFT;
			}
			res = cudaDeviceSynchronize();
			if (res != cudaSuccess) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
			}
		}
		if ((FFTPlan->numAxisUploads[axis_id] == 1) && (!app->configuration.makeForwardPlanOnly)) {
			launchParams.buffer = &app->bufferBluesteinIFFT[axis_id];
			resFFT = VkFFTAppend(&kernelPreparationApplication, 1, &launchParams);
			if (resFFT != VKFFT_SUCCESS) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return resFFT;
			}
			res = cudaDeviceSynchronize();
			if (res != cudaSuccess) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
			}
		}
#elif(VKFFT_BACKEND==2)
		VkFFTLaunchParams launchParams = VKFFT_ZERO_INIT;
		launchParams.inputBuffer = &app->bufferBluestein[axis_id];
		if (!app->configuration.makeInversePlanOnly) {
			launchParams.buffer = &app->bufferBluesteinFFT[axis_id];
			resFFT = VkFFTAppend(&kernelPreparationApplication, -1, &launchParams);
			if (resFFT != VKFFT_SUCCESS) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return resFFT;
			}
			res = hipDeviceSynchronize();
			if (res != hipSuccess) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
			}
		}
		if ((FFTPlan->numAxisUploads[axis_id] == 1) && (!app->configuration.makeForwardPlanOnly)) {
			launchParams.buffer = &app->bufferBluesteinIFFT[axis_id];
			resFFT = VkFFTAppend(&kernelPreparationApplication, 1, &launchParams);
			if (resFFT != VKFFT_SUCCESS) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return resFFT;
			}
			res = hipDeviceSynchronize();
			if (res != hipSuccess) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
			}
		}
#elif(VKFFT_BACKEND==3)
		VkFFTLaunchParams launchParams = VKFFT_ZERO_INIT;
		launchParams.commandQueue = &commandQueue;
		launchParams.inputBuffer = &app->bufferBluestein[axis_id];
		if (!app->configuration.makeInversePlanOnly) {
			launchParams.buffer = &app->bufferBluesteinFFT[axis_id];
			resFFT = VkFFTAppend(&kernelPreparationApplication, -1, &launchParams);
			if (resFFT != VKFFT_SUCCESS) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return resFFT;
			}
			res = clFinish(commandQueue);
			if (res != CL_SUCCESS) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
			}
		}
		if ((FFTPlan->numAxisUploads[axis_id] == 1) && (!app->configuration.makeForwardPlanOnly)) {
			launchParams.buffer = &app->bufferBluesteinIFFT[axis_id];
			resFFT = VkFFTAppend(&kernelPreparationApplication, 1, &launchParams);
			if (resFFT != VKFFT_SUCCESS) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return resFFT;
			}
			res = clFinish(commandQueue);
			if (res != CL_SUCCESS) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
			}
		}
#elif(VKFFT_BACKEND==4)
		ze_command_list_desc_t commandListDescription = VKFFT_ZERO_INIT;
		commandListDescription.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
		ze_command_list_handle_t commandList = VKFFT_ZERO_INIT;
		res = zeCommandListCreate(app->configuration.context[0], app->configuration.device[0], &commandListDescription, &commandList);
		if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
		VkFFTLaunchParams launchParams = VKFFT_ZERO_INIT;
		launchParams.commandList = &commandList;
		launchParams.inputBuffer = &app->bufferBluestein[axis_id];

		if (!app->configuration.makeInversePlanOnly) {
			launchParams.buffer = &app->bufferBluesteinFFT[axis_id];
			resFFT = VkFFTAppend(&kernelPreparationApplication, -1, &launchParams);
			if (resFFT != VKFFT_SUCCESS) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return resFFT;
			}

			res = zeCommandListClose(commandList);
			if (res != ZE_RESULT_SUCCESS) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_FAILED_TO_END_COMMAND_BUFFER;
			}
			res = zeCommandQueueExecuteCommandLists(app->configuration.commandQueue[0], 1, &commandList, 0);
			if (res != ZE_RESULT_SUCCESS) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_FAILED_TO_SUBMIT_QUEUE;
			}
			res = zeCommandQueueSynchronize(app->configuration.commandQueue[0], UINT32_MAX);
			if (res != ZE_RESULT_SUCCESS) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
			}
			res = zeCommandListReset(commandList);
			if (res != ZE_RESULT_SUCCESS) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_FAILED_TO_DESTROY_COMMAND_LIST;
			}
		}
		if ((FFTPlan->numAxisUploads[axis_id] == 1) && (!app->configuration.makeForwardPlanOnly)) {
			launchParams.buffer = &app->bufferBluesteinIFFT[axis_id];
			resFFT = VkFFTAppend(&kernelPreparationApplication, 1, &launchParams);
			if (resFFT != VKFFT_SUCCESS) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return resFFT;
			}
			res = zeCommandListClose(commandList);
			if (res != ZE_RESULT_SUCCESS) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_FAILED_TO_END_COMMAND_BUFFER;
			}
			res = zeCommandQueueExecuteCommandLists(app->configuration.commandQueue[0], 1, &commandList, 0);
			if (res != ZE_RESULT_SUCCESS) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_FAILED_TO_SUBMIT_QUEUE;
			}
			res = zeCommandQueueSynchronize(app->configuration.commandQueue[0], UINT32_MAX);
			if (res != ZE_RESULT_SUCCESS) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
			}
		}
		res = zeCommandListDestroy(commandList);
		if (res != ZE_RESULT_SUCCESS) {
			free(phaseVectors);
			deleteVkFFT(&kernelPreparationApplication);
			return VKFFT_ERROR_FAILED_TO_DESTROY_COMMAND_LIST;
		}
#elif(VKFFT_BACKEND==5)
		VkFFTLaunchParams launchParams = VKFFT_ZERO_INIT;
		launchParams.inputBuffer = &app->bufferBluestein[axis_id];
		if (!app->configuration.makeInversePlanOnly) {
			MTL::CommandBuffer* commandBuffer = app->configuration.queue->commandBuffer();
			if (commandBuffer == 0) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
			MTL::ComputeCommandEncoder* commandEncoder = commandBuffer->computeCommandEncoder();
			if (commandEncoder == 0) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;

			launchParams.commandBuffer = commandBuffer;
			launchParams.commandEncoder = commandEncoder;
			launchParams.buffer = &app->bufferBluesteinFFT[axis_id];
			resFFT = VkFFTAppend(&kernelPreparationApplication, -1, &launchParams);
			if (resFFT != VKFFT_SUCCESS) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return resFFT;
			}
			commandEncoder->endEncoding();
			commandBuffer->commit();
			commandBuffer->waitUntilCompleted();
			commandEncoder->release();
			commandBuffer->release();
		}
		if ((FFTPlan->numAxisUploads[axis_id] == 1) && (!app->configuration.makeForwardPlanOnly)) {
			MTL::CommandBuffer* commandBuffer = app->configuration.queue->commandBuffer();
			if (commandBuffer == 0) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
			MTL::ComputeCommandEncoder* commandEncoder = commandBuffer->computeCommandEncoder();
			if (commandEncoder == 0) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;

			launchParams.commandBuffer = commandBuffer;
			launchParams.commandEncoder = commandEncoder;
			launchParams.buffer = &app->bufferBluesteinIFFT[axis_id];
			resFFT = VkFFTAppend(&kernelPreparationApplication, 1, &launchParams);
			if (resFFT != VKFFT_SUCCESS) {
				free(phaseVectors);
				deleteVkFFT(&kernelPreparationApplication);
				return resFFT;
			}
			commandEncoder->endEncoding();
			commandBuffer->commit();
			commandBuffer->waitUntilCompleted();
			commandEncoder->release();
			commandBuffer->release();
		}
#endif
#if(VKFFT_BACKEND==0)
		kernelPreparationApplication.configuration.isCompilerInitialized = 0;
#elif(VKFFT_BACKEND==3)
		res = clReleaseCommandQueue(commandQueue);
		if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_RELEASE_COMMAND_QUEUE;
#endif
		if (kernelPreparationConfiguration.saveApplicationToString) {
			app->applicationBluesteinStringSize[axis_id] = kernelPreparationApplication.applicationStringSize;
			app->applicationBluesteinString[axis_id] = calloc(app->applicationBluesteinStringSize[axis_id], 1);
			if (!app->applicationBluesteinString[axis_id]) {
				deleteVkFFT(&kernelPreparationApplication);
				return VKFFT_ERROR_MALLOC_FAILED;
			}
			memcpy(app->applicationBluesteinString[axis_id], kernelPreparationApplication.saveApplicationString, app->applicationBluesteinStringSize[axis_id]);
		}
		deleteVkFFT(&kernelPreparationApplication);
		free(phaseVectors);
#ifdef VkFFT_use_FP128_Bluestein_RaderFFT
	}
#endif
	return resFFT;
}
static inline VkFFTResult VkFFTGenerateRaderFFTKernel(VkFFTApplication* app, VkFFTAxis* axis) {
	//generate Rader FFTKernel
	VkFFTResult resFFT = VKFFT_SUCCESS;
	if (axis->specializationConstants.useRader) {
		for (pfUINT i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
			if (axis->specializationConstants.raderContainer[i].type == 0) {
				for (pfUINT j = 0; j < app->numRaderFFTPrimes; j++) {
					if (app->rader_primes[j] == axis->specializationConstants.raderContainer[i].prime) {
						axis->specializationConstants.raderContainer[i].raderFFTkernel = app->raderFFTkernel[j];
					}
				}
				if (axis->specializationConstants.raderContainer[i].raderFFTkernel) continue;

				pfUINT write_id = app->numRaderFFTPrimes;
				app->rader_primes[write_id] = axis->specializationConstants.raderContainer[i].prime;
				app->numRaderFFTPrimes++;

				if (app->configuration.loadApplicationFromString) continue;

#ifdef VkFFT_use_FP128_Bluestein_RaderFFT
				if (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory) {
					pfLD double_PI = pfFPinit("3.14159265358979323846264338327950288419716939937510");
					double* raderFFTkernel = (double*)malloc((axis->specializationConstants.raderContainer[i].prime - 1) * sizeof(double) * 2);
					if (!raderFFTkernel) return VKFFT_ERROR_MALLOC_FAILED;
					axis->specializationConstants.raderContainer[i].raderFFTkernel = (void*)raderFFTkernel;
					app->raderFFTkernel[write_id] = (void*)raderFFTkernel;
					app->rader_buffer_size[write_id] = (axis->specializationConstants.raderContainer[i].prime - 1) * sizeof(double) * 2;

					pfLD* raderFFTkernel_temp = (pfLD*)malloc((axis->specializationConstants.raderContainer[i].prime - 1) * sizeof(pfLD) * 2);
					if (!raderFFTkernel_temp) return VKFFT_ERROR_MALLOC_FAILED;
					for (pfUINT j = 0; j < (axis->specializationConstants.raderContainer[i].prime - 1); j++) {//fix later
						pfUINT g_pow = 1;
						for (pfUINT t = 0; t < axis->specializationConstants.raderContainer[i].prime - 1 - j; t++) {
							g_pow = (g_pow * axis->specializationConstants.raderContainer[i].generator) % axis->specializationConstants.raderContainer[i].prime;
						}
						raderFFTkernel_temp[2 * j] = pfcos(2.0 * g_pow * double_PI / axis->specializationConstants.raderContainer[i].prime);
						raderFFTkernel_temp[2 * j + 1] = -pfsin(2.0 * g_pow * double_PI / axis->specializationConstants.raderContainer[i].prime);
					}
					fftwl_plan p;
					p = fftwl_plan_dft_1d((int)(axis->specializationConstants.raderContainer[i].prime - 1), (fftwl_complex*)raderFFTkernel_temp, (fftwl_complex*)raderFFTkernel_temp, -1, FFTW_ESTIMATE);
					fftwl_execute(p);
					fftwl_destroy_plan(p);
					for (pfUINT j = 0; j < (axis->specializationConstants.raderContainer[i].prime - 1); j++) {//fix later
						raderFFTkernel[2 * j] = (double)raderFFTkernel_temp[2 * j];
						raderFFTkernel[2 * j + 1] = (double)raderFFTkernel_temp[2 * j + 1];
					}
					free(raderFFTkernel_temp);
					continue;
				}
#endif
				if (app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) {
					PfContainer in = VKFFT_ZERO_INIT;
					PfContainer temp1 = VKFFT_ZERO_INIT;
					in.type = 22;
					pfLD double_PI = pfFPinit("3.14159265358979323846264338327950288419716939937510");
					double* raderFFTkernel = (double*)malloc((axis->specializationConstants.raderContainer[i].prime - 1) * sizeof(double) * 4);
					if (!raderFFTkernel) return VKFFT_ERROR_MALLOC_FAILED;
					axis->specializationConstants.raderContainer[i].raderFFTkernel = (void*)raderFFTkernel;
					app->raderFFTkernel[write_id] = (void*)raderFFTkernel;
					app->rader_buffer_size[write_id] = (axis->specializationConstants.raderContainer[i].prime - 1) * sizeof(double) * 4;
					for (pfUINT j = 0; j < (axis->specializationConstants.raderContainer[i].prime - 1); j++) {//fix later
						pfUINT g_pow = 1;
						for (pfUINT t = 0; t < axis->specializationConstants.raderContainer[i].prime - 1 - j; t++) {
							g_pow = (g_pow * axis->specializationConstants.raderContainer[i].generator) % axis->specializationConstants.raderContainer[i].prime;
						}
						pfLD angle = g_pow * double_PI * pfFPinit("2.0") / axis->specializationConstants.raderContainer[i].prime;
						in.data.d = pfcos(angle);
						PfConvToDoubleDouble(&axis->specializationConstants, &temp1, &in);
						raderFFTkernel[4 * j] = (double)temp1.data.dd[0].data.d;
						raderFFTkernel[4 * j + 1] = (double)temp1.data.dd[1].data.d;
						in.data.d = -pfsin(angle);
						PfConvToDoubleDouble(&axis->specializationConstants, &temp1, &in);
						raderFFTkernel[4 * j + 2] = (double)temp1.data.dd[0].data.d;
						raderFFTkernel[4 * j + 3] = (double)temp1.data.dd[1].data.d;
					}
					PfDeallocateContainer(&axis->specializationConstants, &temp1);
				}
				else if (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory) {
					pfLD double_PI = pfFPinit("3.14159265358979323846264338327950288419716939937510");
					double* raderFFTkernel = (double*)malloc((axis->specializationConstants.raderContainer[i].prime - 1) * sizeof(double) * 2);
					if (!raderFFTkernel) return VKFFT_ERROR_MALLOC_FAILED;
					axis->specializationConstants.raderContainer[i].raderFFTkernel = (void*)raderFFTkernel;
					app->raderFFTkernel[write_id] = (void*)raderFFTkernel;
					app->rader_buffer_size[write_id] = (axis->specializationConstants.raderContainer[i].prime - 1) * sizeof(double) * 2;
					for (pfUINT j = 0; j < (axis->specializationConstants.raderContainer[i].prime - 1); j++) {//fix later
						pfUINT g_pow = 1;
						for (pfUINT t = 0; t < axis->specializationConstants.raderContainer[i].prime - 1 - j; t++) {
							g_pow = (g_pow * axis->specializationConstants.raderContainer[i].generator) % axis->specializationConstants.raderContainer[i].prime;
						}
						raderFFTkernel[2 * j] = (double)pfcos(2.0 * g_pow * double_PI / axis->specializationConstants.raderContainer[i].prime);
						raderFFTkernel[2 * j + 1] = (double)-pfsin(2.0 * g_pow * double_PI / axis->specializationConstants.raderContainer[i].prime);
					}
				}
				else {
					double double_PI = 3.14159265358979323846264338327950288419716939937510;
					float* raderFFTkernel = (float*)malloc((axis->specializationConstants.raderContainer[i].prime - 1) * sizeof(float) * 2);
					if (!raderFFTkernel) return VKFFT_ERROR_MALLOC_FAILED;
					axis->specializationConstants.raderContainer[i].raderFFTkernel = (void*)raderFFTkernel;
					app->raderFFTkernel[write_id] = (void*)raderFFTkernel;
					app->rader_buffer_size[write_id] = (axis->specializationConstants.raderContainer[i].prime - 1) * sizeof(float) * 2;
					for (pfUINT j = 0; j < (axis->specializationConstants.raderContainer[i].prime - 1); j++) {//fix later
						pfUINT g_pow = 1;
						for (pfUINT t = 0; t < axis->specializationConstants.raderContainer[i].prime - 1 - j; t++) {
							g_pow = (g_pow * axis->specializationConstants.raderContainer[i].generator) % axis->specializationConstants.raderContainer[i].prime;
						}
						raderFFTkernel[2 * j] = (float)pfcos(2.0 * g_pow * double_PI / axis->specializationConstants.raderContainer[i].prime);
						raderFFTkernel[2 * j + 1] = (float)(-pfsin(2.0 * g_pow * double_PI / axis->specializationConstants.raderContainer[i].prime));
					}
				}

				VkFFTApplication kernelPreparationApplication = VKFFT_ZERO_INIT;
				VkFFTConfiguration kernelPreparationConfiguration = VKFFT_ZERO_INIT;

				kernelPreparationConfiguration.FFTdim = 1;
				kernelPreparationConfiguration.size[0] = axis->specializationConstants.raderContainer[i].prime - 1;
				kernelPreparationConfiguration.size[1] = 1;
				kernelPreparationConfiguration.size[2] = 1;
				kernelPreparationConfiguration.doublePrecision = (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory);
				kernelPreparationConfiguration.quadDoubleDoublePrecision = (app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory);
				kernelPreparationConfiguration.useLUT = 1;
				kernelPreparationConfiguration.fixMinRaderPrimeFFT = 17;
				kernelPreparationConfiguration.fixMinRaderPrimeMult = 17;
				kernelPreparationConfiguration.fixMaxRaderPrimeFFT = 17;
				kernelPreparationConfiguration.fixMaxRaderPrimeMult = 17;

				kernelPreparationConfiguration.device = app->configuration.device;
#if(VKFFT_BACKEND==0)
				kernelPreparationConfiguration.queue = app->configuration.queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
				kernelPreparationConfiguration.fence = app->configuration.fence;
				kernelPreparationConfiguration.commandPool = app->configuration.commandPool;
				kernelPreparationConfiguration.physicalDevice = app->configuration.physicalDevice;
				kernelPreparationConfiguration.isCompilerInitialized = 1;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization
				if (app->configuration.stagingBuffer != 0)	kernelPreparationConfiguration.stagingBuffer = app->configuration.stagingBuffer;
				if (app->configuration.stagingBufferMemory != 0)	kernelPreparationConfiguration.stagingBufferMemory = app->configuration.stagingBufferMemory;
#elif(VKFFT_BACKEND==3)
				kernelPreparationConfiguration.context = app->configuration.context;
#elif(VKFFT_BACKEND==4)
				kernelPreparationConfiguration.context = app->configuration.context;
				kernelPreparationConfiguration.commandQueue = app->configuration.commandQueue;
				kernelPreparationConfiguration.commandQueueID = app->configuration.commandQueueID;
#elif(VKFFT_BACKEND==5)
				kernelPreparationConfiguration.device = app->configuration.device;
				kernelPreparationConfiguration.queue = app->configuration.queue;
#endif			

				pfUINT bufferSize = (pfUINT)sizeof(float) * 2 * kernelPreparationConfiguration.size[0] * kernelPreparationConfiguration.size[1] * kernelPreparationConfiguration.size[2];
				if (kernelPreparationConfiguration.doublePrecision) bufferSize *= sizeof(double) / sizeof(float);
				if (kernelPreparationConfiguration.quadDoubleDoublePrecision) bufferSize *= 2 * sizeof(double) / sizeof(float);

				kernelPreparationConfiguration.bufferSize = &bufferSize;
				resFFT = initializeVkFFT(&kernelPreparationApplication, kernelPreparationConfiguration);
				if (resFFT != VKFFT_SUCCESS) return resFFT;

#if(VKFFT_BACKEND==0)
				VkDeviceMemory bufferRaderFFTDeviceMemory;
				VkBuffer bufferRaderFFT;
#elif(VKFFT_BACKEND==1)
				void* bufferRaderFFT;
#elif(VKFFT_BACKEND==2)
				void* bufferRaderFFT;
#elif(VKFFT_BACKEND==3)
				cl_mem bufferRaderFFT;
#elif(VKFFT_BACKEND==4)
				void* bufferRaderFFT;
#elif(VKFFT_BACKEND==5)
				MTL::Buffer* bufferRaderFFT;
#endif
#if(VKFFT_BACKEND==0)
				VkResult res = VK_SUCCESS;
				resFFT = allocateBufferVulkan(app, &bufferRaderFFT, &bufferRaderFFTDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
				if (resFFT != VKFFT_SUCCESS) return resFFT;
#elif(VKFFT_BACKEND==1)
				cudaError_t res = cudaSuccess;
				res = cudaMalloc(&bufferRaderFFT, bufferSize);
				if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
#elif(VKFFT_BACKEND==2)
				hipError_t res = hipSuccess;
				res = hipMalloc(&bufferRaderFFT, bufferSize);
				if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
#elif(VKFFT_BACKEND==3)
				cl_int res = CL_SUCCESS;
				bufferRaderFFT = clCreateBuffer(app->configuration.context[0], CL_MEM_READ_WRITE, bufferSize, 0, &res);
				if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
				cl_command_queue commandQueue = clCreateCommandQueue(app->configuration.context[0], app->configuration.device[0], 0, &res);
				if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE;
#elif(VKFFT_BACKEND==4)
				ze_result_t res = ZE_RESULT_SUCCESS;
				ze_device_mem_alloc_desc_t device_desc = VKFFT_ZERO_INIT;
				device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
				res = zeMemAllocDevice(app->configuration.context[0], &device_desc, bufferSize, sizeof(float), app->configuration.device[0], &bufferRaderFFT);
				if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
#elif(VKFFT_BACKEND==5)
				bufferRaderFFT = app->configuration.device->newBuffer(bufferSize, MTL::ResourceStorageModePrivate);
#endif

				resFFT = VkFFT_TransferDataFromCPU(app, axis->specializationConstants.raderContainer[i].raderFFTkernel, &bufferRaderFFT, bufferSize);
				if (resFFT != VKFFT_SUCCESS) {
					free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
					deleteVkFFT(&kernelPreparationApplication);
					return resFFT;
				}
#if(VKFFT_BACKEND==0)
				{
					VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
					commandBufferAllocateInfo.commandPool = kernelPreparationApplication.configuration.commandPool[0];
					commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
					commandBufferAllocateInfo.commandBufferCount = 1;
					VkCommandBuffer commandBuffer = VKFFT_ZERO_INIT;
					res = vkAllocateCommandBuffers(kernelPreparationApplication.configuration.device[0], &commandBufferAllocateInfo, &commandBuffer);
					if (res != 0) {
						free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
						deleteVkFFT(&kernelPreparationApplication);
						return VKFFT_ERROR_FAILED_TO_ALLOCATE_COMMAND_BUFFERS;
					}
					VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
					commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
					res = vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
					if (res != 0) {
						free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
						deleteVkFFT(&kernelPreparationApplication);
						return VKFFT_ERROR_FAILED_TO_BEGIN_COMMAND_BUFFER;
					}
					VkFFTLaunchParams launchParams = VKFFT_ZERO_INIT;
					launchParams.commandBuffer = &commandBuffer;
					launchParams.buffer = &bufferRaderFFT;
					//Record commands
					resFFT = VkFFTAppend(&kernelPreparationApplication, -1, &launchParams);
					if (resFFT != VKFFT_SUCCESS) {
						free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
						deleteVkFFT(&kernelPreparationApplication);
						return resFFT;
					}
					res = vkEndCommandBuffer(commandBuffer);
					if (res != 0) {
						free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
						deleteVkFFT(&kernelPreparationApplication);
						return VKFFT_ERROR_FAILED_TO_END_COMMAND_BUFFER;
					}
					VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
					submitInfo.commandBufferCount = 1;
					submitInfo.pCommandBuffers = &commandBuffer;
					res = vkQueueSubmit(kernelPreparationApplication.configuration.queue[0], 1, &submitInfo, kernelPreparationApplication.configuration.fence[0]);
					if (res != 0) {
						free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
						deleteVkFFT(&kernelPreparationApplication);
						return VKFFT_ERROR_FAILED_TO_SUBMIT_QUEUE;
					}
					res = vkWaitForFences(kernelPreparationApplication.configuration.device[0], 1, kernelPreparationApplication.configuration.fence, VK_TRUE, 100000000000);
					if (res != 0) {
						free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
						deleteVkFFT(&kernelPreparationApplication);
						return VKFFT_ERROR_FAILED_TO_WAIT_FOR_FENCES;
					}
					res = vkResetFences(kernelPreparationApplication.configuration.device[0], 1, kernelPreparationApplication.configuration.fence);
					if (res != 0) {
						free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
						deleteVkFFT(&kernelPreparationApplication);
						return VKFFT_ERROR_FAILED_TO_RESET_FENCES;
					}
					vkFreeCommandBuffers(kernelPreparationApplication.configuration.device[0], kernelPreparationApplication.configuration.commandPool[0], 1, &commandBuffer);
				}
#elif(VKFFT_BACKEND==1)
				VkFFTLaunchParams launchParams = VKFFT_ZERO_INIT;
				launchParams.buffer = &bufferRaderFFT;
				resFFT = VkFFTAppend(&kernelPreparationApplication, -1, &launchParams);
				if (resFFT != VKFFT_SUCCESS) {
					free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
					deleteVkFFT(&kernelPreparationApplication);
					return resFFT;
				}
				res = cudaDeviceSynchronize();
				if (res != cudaSuccess) {
					free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
					deleteVkFFT(&kernelPreparationApplication);
					return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
				}
#elif(VKFFT_BACKEND==2)
				VkFFTLaunchParams launchParams = VKFFT_ZERO_INIT;
				launchParams.buffer = &bufferRaderFFT;
				resFFT = VkFFTAppend(&kernelPreparationApplication, -1, &launchParams);
				if (resFFT != VKFFT_SUCCESS) {
					free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
					deleteVkFFT(&kernelPreparationApplication);
					return resFFT;
				}
				res = hipDeviceSynchronize();
				if (res != hipSuccess) {
					free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
					deleteVkFFT(&kernelPreparationApplication);
					return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
				}
#elif(VKFFT_BACKEND==3)
				VkFFTLaunchParams launchParams = VKFFT_ZERO_INIT;
				launchParams.commandQueue = &commandQueue;
				launchParams.buffer = &bufferRaderFFT;
				resFFT = VkFFTAppend(&kernelPreparationApplication, -1, &launchParams);
				if (resFFT != VKFFT_SUCCESS) {
					free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
					deleteVkFFT(&kernelPreparationApplication);
					return resFFT;
				}
				res = clFinish(commandQueue);
				if (res != CL_SUCCESS) {
					free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
					deleteVkFFT(&kernelPreparationApplication);
					return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
				}
#elif(VKFFT_BACKEND==4)
				ze_command_list_desc_t commandListDescription = VKFFT_ZERO_INIT;
				commandListDescription.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
				ze_command_list_handle_t commandList = VKFFT_ZERO_INIT;
				res = zeCommandListCreate(app->configuration.context[0], app->configuration.device[0], &commandListDescription, &commandList);
				if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
				VkFFTLaunchParams launchParams = VKFFT_ZERO_INIT;
				launchParams.commandList = &commandList;
				launchParams.buffer = &bufferRaderFFT;
				resFFT = VkFFTAppend(&kernelPreparationApplication, -1, &launchParams);
				if (resFFT != VKFFT_SUCCESS) {
					free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
					deleteVkFFT(&kernelPreparationApplication);
					return resFFT;
				}
				res = zeCommandListClose(commandList);
				if (res != ZE_RESULT_SUCCESS) {
					free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
					deleteVkFFT(&kernelPreparationApplication);
					return VKFFT_ERROR_FAILED_TO_END_COMMAND_BUFFER;
				}
				res = zeCommandQueueExecuteCommandLists(app->configuration.commandQueue[0], 1, &commandList, 0);
				if (res != ZE_RESULT_SUCCESS) {
					free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
					deleteVkFFT(&kernelPreparationApplication);
					return VKFFT_ERROR_FAILED_TO_SUBMIT_QUEUE;
				}
				res = zeCommandQueueSynchronize(app->configuration.commandQueue[0], UINT32_MAX);
				if (res != ZE_RESULT_SUCCESS) {
					free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
					deleteVkFFT(&kernelPreparationApplication);
					return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
				}
				res = zeCommandListDestroy(commandList);
				if (res != ZE_RESULT_SUCCESS) {
					free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
					deleteVkFFT(&kernelPreparationApplication);
					return VKFFT_ERROR_FAILED_TO_DESTROY_COMMAND_LIST;
				}
#elif(VKFFT_BACKEND==5)
				VkFFTLaunchParams launchParams = VKFFT_ZERO_INIT;
				MTL::CommandBuffer* commandBuffer = app->configuration.queue->commandBuffer();
				if (commandBuffer == 0) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
				MTL::ComputeCommandEncoder* commandEncoder = commandBuffer->computeCommandEncoder();
				if (commandEncoder == 0) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;

				launchParams.commandBuffer = commandBuffer;
				launchParams.commandEncoder = commandEncoder;
				launchParams.buffer = &bufferRaderFFT;
				resFFT = VkFFTAppend(&kernelPreparationApplication, -1, &launchParams);
				if (resFFT != VKFFT_SUCCESS) {
					free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
					deleteVkFFT(&kernelPreparationApplication);
					return resFFT;
				}
				commandEncoder->endEncoding();
				commandBuffer->commit();
				commandBuffer->waitUntilCompleted();
				commandEncoder->release();
				commandBuffer->release();
#endif
				resFFT = VkFFT_TransferDataToCPU(&kernelPreparationApplication, axis->specializationConstants.raderContainer[i].raderFFTkernel, &bufferRaderFFT, bufferSize);
				if (resFFT != VKFFT_SUCCESS) {
					free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
					deleteVkFFT(&kernelPreparationApplication);
					return resFFT;
				}

#if(VKFFT_BACKEND==0)
				kernelPreparationApplication.configuration.isCompilerInitialized = 0;
#elif(VKFFT_BACKEND==3)
				res = clReleaseCommandQueue(commandQueue);
				if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_RELEASE_COMMAND_QUEUE;
#endif
#if(VKFFT_BACKEND==0)
				vkDestroyBuffer(app->configuration.device[0], bufferRaderFFT, 0);
				vkFreeMemory(app->configuration.device[0], bufferRaderFFTDeviceMemory, 0);
#elif(VKFFT_BACKEND==1)
				cudaFree(bufferRaderFFT);
#elif(VKFFT_BACKEND==2)
				hipFree(bufferRaderFFT);
#elif(VKFFT_BACKEND==3)
				clReleaseMemObject(bufferRaderFFT);
#elif(VKFFT_BACKEND==4)
				zeMemFree(app->configuration.context[0], bufferRaderFFT);
#elif(VKFFT_BACKEND==5)
				bufferRaderFFT->release();
#endif
				deleteVkFFT(&kernelPreparationApplication);
			}
		}
		if (app->configuration.loadApplicationFromString) {
			pfUINT offset = 0;
			for (pfUINT i = 0; i < app->numRaderFFTPrimes; i++) {
				pfUINT current_size = 0;
				if (app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) {
					current_size = (app->rader_primes[i] - 1) * sizeof(double) * 4;
				}
				else if (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory) {
					current_size = (app->rader_primes[i] - 1) * sizeof(double) * 2;
				}
				else {
					current_size = (app->rader_primes[i] - 1) * sizeof(float) * 2;
				}
				if (!app->raderFFTkernel[i]) {
					app->raderFFTkernel[i] = (void*)malloc(current_size);
					if (!app->raderFFTkernel[i]) return VKFFT_ERROR_MALLOC_FAILED;
					memcpy(app->raderFFTkernel[i], (char*)app->configuration.loadApplicationString + app->applicationStringOffsetRader + offset, current_size);
				}
				for (pfUINT j = 0; j < axis->specializationConstants.numRaderPrimes; j++) {
					if ((app->rader_primes[i] == axis->specializationConstants.raderContainer[j].prime) && (axis->specializationConstants.raderContainer[j].type == 0))
						axis->specializationConstants.raderContainer[j].raderFFTkernel = app->raderFFTkernel[i];
				}
				offset += current_size;
			}
		}
	}
	return resFFT;
}

#endif
