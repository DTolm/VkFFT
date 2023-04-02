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
#ifndef VKFFT_PLAN_FFT_H
#define VKFFT_PLAN_FFT_H
#include "vkFFT_Structs/vkFFT_Structs.h"

#include "vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_ManageMemory.h"
#include "vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_InitAPIParameters.h"
#include "vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_CompileKernel.h"
#include "vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_ManageLUT.h"
#include "vkFFT_CodeGen/vkFFT_KernelsLevel2/vkFFT_FFT.h"
#include "vkFFT_AppManagement/vkFFT_DeleteApp.h"
static inline VkFFTResult VkFFTPlanAxis(VkFFTApplication* app, VkFFTPlan* FFTPlan, uint64_t axis_id, uint64_t axis_upload_id, uint64_t inverse, uint64_t reverseBluesteinMultiUpload) {
	//get radix stages
	VkFFTResult resFFT = VKFFT_SUCCESS;
#if(VKFFT_BACKEND==0)
	VkResult res = VK_SUCCESS;
#elif(VKFFT_BACKEND==1)
	cudaError_t res = cudaSuccess;
#elif(VKFFT_BACKEND==2)
	hipError_t res = hipSuccess;
#elif(VKFFT_BACKEND==3)
	cl_int res = CL_SUCCESS;
#elif(VKFFT_BACKEND==4)
	ze_result_t res = ZE_RESULT_SUCCESS;
#elif(VKFFT_BACKEND==5)
#endif
	VkFFTAxis* axis = (reverseBluesteinMultiUpload) ? &FFTPlan->inverseBluesteinAxes[axis_id][axis_upload_id] : &FFTPlan->axes[axis_id][axis_upload_id];

	axis->specializationConstants.sourceFFTSize.type = 31;
	axis->specializationConstants.sourceFFTSize.data.i = app->configuration.size[axis_id];
	axis->specializationConstants.axis_id = axis_id;
	axis->specializationConstants.axis_upload_id = axis_upload_id;

	if ((app->configuration.FFTdim == 1) && (FFTPlan->actualFFTSizePerAxis[axis_id][1] == 1) && ((app->configuration.numberBatches > 1) || (app->actualNumBatches > 1)) && (!app->configuration.performConvolution) && (app->configuration.coordinateFeatures == 1)) {
		if (app->configuration.numberBatches > 1) {
			app->actualNumBatches = app->configuration.numberBatches;
			app->configuration.numberBatches = 1;
		}
		FFTPlan->actualFFTSizePerAxis[axis_id][1] = app->actualNumBatches;
	}
	axis->specializationConstants.numBatches.type = 31;
	axis->specializationConstants.numBatches.data.i = app->configuration.numberBatches;
	axis->specializationConstants.warpSize = app->configuration.warpSize;
	axis->specializationConstants.numSharedBanks = app->configuration.numSharedBanks;
	axis->specializationConstants.useUint64 = app->configuration.useUint64;
#if(VKFFT_BACKEND==2)
	axis->specializationConstants.useStrict32BitAddress = app->configuration.useStrict32BitAddress;
#endif
	axis->specializationConstants.disableSetLocale = app->configuration.disableSetLocale;

	axis->specializationConstants.numAxisUploads = FFTPlan->numAxisUploads[axis_id];
	axis->specializationConstants.fixMinRaderPrimeMult = app->configuration.fixMinRaderPrimeMult;
	axis->specializationConstants.fixMaxRaderPrimeMult = app->configuration.fixMaxRaderPrimeMult;
	axis->specializationConstants.fixMinRaderPrimeFFT = app->configuration.fixMinRaderPrimeFFT;
	axis->specializationConstants.fixMaxRaderPrimeFFT = app->configuration.fixMaxRaderPrimeFFT;

	axis->specializationConstants.raderUintLUT = (axis->specializationConstants.useRader) ? app->configuration.useRaderUintLUT : 0;
	axis->specializationConstants.inline_rader_g_pow = (axis->specializationConstants.raderUintLUT) ? 2 : 1;
	axis->specializationConstants.inline_rader_kernel = (app->configuration.useLUT == 1) ? 0 : 1;
	axis->specializationConstants.supportAxis = 0;
	axis->specializationConstants.symmetricKernel = app->configuration.symmetricKernel;
	axis->specializationConstants.conjugateConvolution = app->configuration.conjugateConvolution;
	axis->specializationConstants.crossPowerSpectrumNormalization = app->configuration.crossPowerSpectrumNormalization;

	axis->specializationConstants.maxCodeLength = app->configuration.maxCodeLength;
	axis->specializationConstants.maxTempLength = app->configuration.maxTempLength;

	axis->specializationConstants.double_PI = 3.14159265358979323846264338327950288419716939937510L;

	if (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory) {
		axis->specializationConstants.precision = 1;
		axis->specializationConstants.complexSize = 16;
	}
	else {
		if (app->configuration.halfPrecision) {
			axis->specializationConstants.precision = 0;
			axis->specializationConstants.complexSize = 8;
		}
		else {
			axis->specializationConstants.precision = 0;
			axis->specializationConstants.complexSize = 8;
		}
	}
	
	uint64_t allowedSharedMemory = app->configuration.sharedMemorySize;
	uint64_t allowedSharedMemoryPow2 = app->configuration.sharedMemorySizePow2;

	if (axis->specializationConstants.useRaderMult) {
		allowedSharedMemory -= (axis->specializationConstants.useRaderMult - 1) * axis->specializationConstants.complexSize;
		allowedSharedMemoryPow2 -= (axis->specializationConstants.useRaderMult - 1) * axis->specializationConstants.complexSize;
	}

	uint64_t maxSequenceLengthSharedMemory = allowedSharedMemory / axis->specializationConstants.complexSize;
	uint64_t maxSequenceLengthSharedMemoryPow2 = allowedSharedMemoryPow2 / axis->specializationConstants.complexSize;
	uint64_t maxSingleSizeStrided = (app->configuration.coalescedMemory > axis->specializationConstants.complexSize) ? allowedSharedMemory / (app->configuration.coalescedMemory) : allowedSharedMemory / axis->specializationConstants.complexSize;
	uint64_t maxSingleSizeStridedPow2 = (app->configuration.coalescedMemory > axis->specializationConstants.complexSize) ? allowedSharedMemoryPow2 / (app->configuration.coalescedMemory) : allowedSharedMemoryPow2 / axis->specializationConstants.complexSize;

	axis->specializationConstants.stageStartSize.type = 31;
	axis->specializationConstants.stageStartSize.data.i = 1;
	for (uint64_t i = 0; i < axis_upload_id; i++)
		axis->specializationConstants.stageStartSize.data.i *= FFTPlan->axisSplit[axis_id][i];

	axis->specializationConstants.firstStageStartSize.type = 31;
	axis->specializationConstants.firstStageStartSize.data.i = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / FFTPlan->axisSplit[axis_id][FFTPlan->numAxisUploads[axis_id] - 1];
	axis->specializationConstants.dispatchZactualFFTSize.type = 31;
	axis->specializationConstants.dispatchZactualFFTSize.data.i = (axis_id < 2) ? FFTPlan->actualFFTSizePerAxis[axis_id][2] : FFTPlan->actualFFTSizePerAxis[axis_id][1];
	axis->specializationConstants.fft_dim_x.type = 31;
	if (axis_id == 0) {
		//configure radix stages
		axis->specializationConstants.fft_dim_x.data.i = axis->specializationConstants.stageStartSize.data.i;
	}
	else {
		axis->specializationConstants.fft_dim_x.data.i = FFTPlan->actualFFTSizePerAxis[axis_id][0];
	}
	if (app->useBluesteinFFT[axis_id]) {
		axis->specializationConstants.useBluesteinFFT = 1;
	}

	if (app->configuration.performDCT == 3) {
		axis->specializationConstants.actualInverse = inverse;
		axis->specializationConstants.inverse = !inverse;
	}
	else {
		if (app->configuration.performDCT == 4) {
			axis->specializationConstants.actualInverse = inverse;
			axis->specializationConstants.inverse = 1;
		}
		else {
			axis->specializationConstants.actualInverse = inverse;
			axis->specializationConstants.inverse = inverse;
		}
	}
	if (app->useBluesteinFFT[axis_id]) {
		axis->specializationConstants.actualInverse = inverse;
		axis->specializationConstants.inverse = reverseBluesteinMultiUpload;
		if (app->configuration.performDCT == 3) {
			axis->specializationConstants.inverseBluestein = !inverse;
		}
		else {
			if (app->configuration.performDCT == 4) {
				axis->specializationConstants.inverseBluestein = 1;
			}
			else {
				axis->specializationConstants.inverseBluestein = inverse;
			}
		}
	}
	axis->specializationConstants.reverseBluesteinMultiUpload = reverseBluesteinMultiUpload;

	axis->specializationConstants.reorderFourStep = ((FFTPlan->numAxisUploads[axis_id] > 1) && (!app->useBluesteinFFT[axis_id])) ? app->configuration.reorderFourStep : 0;

	if ((axis_id == 0) && ((FFTPlan->numAxisUploads[axis_id] == 1) || ((axis_upload_id == 0) && (!axis->specializationConstants.reorderFourStep)))) {
		maxSequenceLengthSharedMemory *= axis->specializationConstants.registerBoost;
		maxSequenceLengthSharedMemoryPow2 = (uint64_t)pow(2, (uint64_t)log2(maxSequenceLengthSharedMemory));
	}
	else {
		maxSingleSizeStrided *= axis->specializationConstants.registerBoost;
		maxSingleSizeStridedPow2 = (uint64_t)pow(2, (uint64_t)log2(maxSingleSizeStrided));
	}
	axis->specializationConstants.maxSingleSizeStrided.type = 31;
	axis->specializationConstants.maxSingleSizeStrided.data.i = maxSingleSizeStrided;

	axis->specializationConstants.performR2C = FFTPlan->actualPerformR2CPerAxis[axis_id];
	axis->specializationConstants.performR2CmultiUpload = FFTPlan->multiUploadR2C;
	if (app->configuration.performDCT == 3) {
		axis->specializationConstants.performDCT = 2;
	}
	else {
		axis->specializationConstants.performDCT = app->configuration.performDCT;
	}
	if ((axis->specializationConstants.performR2CmultiUpload) && (app->configuration.size[0] % 2 != 0)) return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH_R2C;
	axis->specializationConstants.mergeSequencesR2C = ((axis->specializationConstants.fftDim.data.i < maxSequenceLengthSharedMemory) && (FFTPlan->actualFFTSizePerAxis[axis_id][1] > 1) && ((FFTPlan->actualPerformR2CPerAxis[axis_id]) || (((app->configuration.performDCT == 3) || (app->configuration.performDCT == 2) || (app->configuration.performDCT == 1) || ((app->configuration.performDCT == 4) && ((app->configuration.size[axis_id] % 2) != 0))) && (axis_id == 0)))) ? (1 - app->configuration.disableMergeSequencesR2C) : 0;
	//uint64_t passID = FFTPlan->numAxisUploads[axis_id] - 1 - axis_upload_id;
	axis->specializationConstants.fft_dim_full.type = 31;
	axis->specializationConstants.fft_dim_full.data.i = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id];
	if ((FFTPlan->numAxisUploads[axis_id] > 1) && (axis->specializationConstants.reorderFourStep || app->useBluesteinFFT[axis_id]) && (!app->configuration.userTempBuffer) && (app->configuration.allocateTempBuffer == 0)) {
		app->configuration.allocateTempBuffer = 1;
#if(VKFFT_BACKEND==0)
		app->configuration.tempBuffer = (VkBuffer*)malloc(sizeof(VkBuffer));
		if (!app->configuration.tempBuffer) {
			deleteVkFFT(app);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
		resFFT = allocateBufferVulkan(app, app->configuration.tempBuffer, &app->configuration.tempBufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, app->configuration.tempBufferSize[0]);
		if (resFFT != VKFFT_SUCCESS) {
			deleteVkFFT(app);
			return resFFT;
		}
#elif(VKFFT_BACKEND==1)
		app->configuration.tempBuffer = (void**)malloc(sizeof(void*));
		if (!app->configuration.tempBuffer) {
			deleteVkFFT(app);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
		res = cudaMalloc(app->configuration.tempBuffer, app->configuration.tempBufferSize[0]);
		if (res != cudaSuccess) {
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_TO_ALLOCATE;
		}
#elif(VKFFT_BACKEND==2)
		app->configuration.tempBuffer = (void**)malloc(sizeof(void*));
		if (!app->configuration.tempBuffer) {
			deleteVkFFT(app);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
		res = hipMalloc(app->configuration.tempBuffer, app->configuration.tempBufferSize[0]);
		if (res != hipSuccess) {
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_TO_ALLOCATE;
		}
#elif(VKFFT_BACKEND==3)
		app->configuration.tempBuffer = (cl_mem*)malloc(sizeof(cl_mem));
		if (!app->configuration.tempBuffer) {
			deleteVkFFT(app);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
		app->configuration.tempBuffer[0] = clCreateBuffer(app->configuration.context[0], CL_MEM_READ_WRITE, app->configuration.tempBufferSize[0], 0, &res);
		if (res != CL_SUCCESS) {
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_TO_ALLOCATE;
		}
#elif(VKFFT_BACKEND==4)
		app->configuration.tempBuffer = (void**)malloc(sizeof(void*));
		if (!app->configuration.tempBuffer) {
			deleteVkFFT(app);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
		ze_device_mem_alloc_desc_t device_desc = {};
		device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
		res = zeMemAllocDevice(app->configuration.context[0], &device_desc, app->configuration.tempBufferSize[0], sizeof(float), app->configuration.device[0], app->configuration.tempBuffer);
		if (res != ZE_RESULT_SUCCESS) {
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_TO_ALLOCATE;
		}
#elif(VKFFT_BACKEND==5)
		app->configuration.tempBuffer = (MTL::Buffer**)malloc(sizeof(MTL::Buffer*));
		if (!app->configuration.tempBuffer) {
			deleteVkFFT(app);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
		app->configuration.tempBuffer[0] = app->configuration.device->newBuffer(app->configuration.tempBufferSize[0], MTL::ResourceStorageModePrivate);
#endif
	}
	//generate Rader Kernels
	resFFT = VkFFTGenerateRaderFFTKernel(app, axis);
	if (resFFT != VKFFT_SUCCESS) {
		deleteVkFFT(app);
		return resFFT;
	}
	resFFT = VkFFT_AllocateLUT(app, FFTPlan, axis, inverse);
	if (resFFT != VKFFT_SUCCESS) {
		deleteVkFFT(app);
		return resFFT;
	}
	axis->specializationConstants.additionalRaderSharedSize.type = 31;
	if (axis->specializationConstants.useRaderMult)	axis->specializationConstants.additionalRaderSharedSize.data.i = (axis->specializationConstants.useRaderMult - 1);

	resFFT = VkFFT_AllocateRaderUintLUT(app, axis);
	if (resFFT != VKFFT_SUCCESS) {
		deleteVkFFT(app);
		return resFFT;
	}
	
	//configure strides

	VkContainer* axisStride = axis->specializationConstants.inputStride;
	uint64_t* usedStride = app->configuration.bufferStride;
	if ((!inverse) && (axis_id == app->firstAxis) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (app->configuration.isInputFormatted)) usedStride = app->configuration.inputBufferStride;
	if ((inverse) && (axis_id == app->lastAxis) && ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && ((app->useBluesteinFFT[axis_id] && (reverseBluesteinMultiUpload == 0)) || (!app->useBluesteinFFT[axis_id])) && (!app->configuration.performConvolution)) && (app->configuration.isInputFormatted) && (!app->configuration.inverseReturnToInputBuffer)) usedStride = app->configuration.inputBufferStride;
	axisStride[0].type = 31;
	axisStride[0].data.i = 1;

	if (axis_id == 0) {
		axisStride[1].type = 31;
		axisStride[1].data.i = usedStride[0];
		axisStride[2].type = 31;
		axisStride[2].data.i = usedStride[1];
	}
	if (axis_id == 1)
	{
		axisStride[1].type = 31;
		axisStride[1].data.i = usedStride[0];
		axisStride[2].type = 31; 
		axisStride[2].data.i = usedStride[1];
	}
	if (axis_id == 2)
	{
		axisStride[1].type = 31;
		axisStride[1].data.i = usedStride[1];
		axisStride[2].type = 31;
		axisStride[2].data.i = usedStride[0];
	}

	axisStride[3].type = 31;
	axisStride[3].data.i = usedStride[2];

	axisStride[4].type = 31;
	axisStride[4].data.i = axisStride[3].data.i * app->configuration.coordinateFeatures;
	if (app->useBluesteinFFT[axis_id] && (FFTPlan->numAxisUploads[axis_id] > 1) && (!((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (reverseBluesteinMultiUpload == 0)))) {
		axisStride[0].data.i = 1;

		if (axis_id == 0) {
			axisStride[1].data.i = FFTPlan->actualFFTSizePerAxis[axis_id][0];
			axisStride[2].data.i = FFTPlan->actualFFTSizePerAxis[axis_id][0] * FFTPlan->actualFFTSizePerAxis[axis_id][1];
		}
		if (axis_id == 1)
		{
			axisStride[1].data.i = FFTPlan->actualFFTSizePerAxis[axis_id][0];
			axisStride[2].data.i = FFTPlan->actualFFTSizePerAxis[axis_id][0] * FFTPlan->actualFFTSizePerAxis[axis_id][1];
		}
		if (axis_id == 2)
		{
			axisStride[1].data.i = FFTPlan->actualFFTSizePerAxis[axis_id][0] * FFTPlan->actualFFTSizePerAxis[axis_id][1];
			axisStride[2].data.i = FFTPlan->actualFFTSizePerAxis[axis_id][0];
		}

		axisStride[3].data.i = axisStride[2].data.i * FFTPlan->actualFFTSizePerAxis[axis_id][2];

		axisStride[4].type = 31;
		axisStride[4].data.i = axisStride[3].data.i * app->configuration.coordinateFeatures;
	}
	if ((!inverse) && (axis_id == 0) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (reverseBluesteinMultiUpload == 0) && (axis->specializationConstants.performR2C || FFTPlan->multiUploadR2C) && (!(app->configuration.isInputFormatted))) {
		axisStride[1].data.i *= 2;
		axisStride[2].data.i *= 2;
		axisStride[3].data.i *= 2;
		axisStride[4].data.i *= 2;
	}
	if ((FFTPlan->multiUploadR2C) && (!inverse) && (axis_id == 0) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (reverseBluesteinMultiUpload == 0)) {
		for (uint64_t i = 1; i < 5; i++) {
			axisStride[i].data.i /= 2;
		}
	}
	axisStride = axis->specializationConstants.outputStride;
	usedStride = app->configuration.bufferStride;
	if ((!inverse) && (axis_id == app->lastAxis) && (axis_upload_id == 0) && (app->configuration.isOutputFormatted)) usedStride = app->configuration.outputBufferStride;
	if ((inverse) && (axis_id == app->firstAxis) && (((axis_upload_id == 0) && (!app->configuration.performConvolution)) || ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && ((reverseBluesteinMultiUpload == 1) || (app->configuration.performConvolution)))) && ((app->configuration.isOutputFormatted))) usedStride = app->configuration.outputBufferStride;
	if ((inverse) && (axis_id == app->firstAxis) && (((axis_upload_id == 0) && (app->configuration.isInputFormatted)) || ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (!axis->specializationConstants.reorderFourStep))) && (app->configuration.inverseReturnToInputBuffer)) usedStride = app->configuration.inputBufferStride;

	axisStride[0].type = 31; 
	axisStride[0].data.i = 1;

	if (axis_id == 0) {
		axisStride[1].type = 31;
		axisStride[1].data.i = usedStride[0];
		axisStride[2].type = 31;
		axisStride[2].data.i = usedStride[1];
	}
	if (axis_id == 1)
	{
		axisStride[1].type = 31;
		axisStride[1].data.i = usedStride[0];
		axisStride[2].type = 31;
		axisStride[2].data.i = usedStride[1];
	}
	if (axis_id == 2)
	{
		axisStride[1].type = 31;
		axisStride[1].data.i = usedStride[1];
		axisStride[2].type = 31;
		axisStride[2].data.i = usedStride[0];
	}

	axisStride[3].type = 31;
	axisStride[3].data.i = usedStride[2];

	axisStride[4].type = 31;
	axisStride[4].data.i = axisStride[3].data.i * app->configuration.coordinateFeatures;
	if (app->useBluesteinFFT[axis_id] && (FFTPlan->numAxisUploads[axis_id] > 1) && (!((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (reverseBluesteinMultiUpload == 1)))) {
		axisStride[0].data.i = 1;

		if (axis_id == 0) {
			axisStride[1].data.i = FFTPlan->actualFFTSizePerAxis[axis_id][0];
			axisStride[2].data.i = FFTPlan->actualFFTSizePerAxis[axis_id][0] * FFTPlan->actualFFTSizePerAxis[axis_id][1];
		}
		if (axis_id == 1)
		{
			axisStride[1].data.i = FFTPlan->actualFFTSizePerAxis[axis_id][0];
			axisStride[2].data.i = FFTPlan->actualFFTSizePerAxis[axis_id][0] * FFTPlan->actualFFTSizePerAxis[axis_id][1];
		}
		if (axis_id == 2)
		{
			axisStride[1].data.i = FFTPlan->actualFFTSizePerAxis[axis_id][0] * FFTPlan->actualFFTSizePerAxis[axis_id][1];
			axisStride[2].data.i = FFTPlan->actualFFTSizePerAxis[axis_id][0];
		}

		axisStride[3].data.i = axisStride[2].data.i * FFTPlan->actualFFTSizePerAxis[axis_id][2];

		axisStride[4].data.i = axisStride[3].data.i * app->configuration.coordinateFeatures;
	}
	if ((inverse) && (axis_id == 0) && (((!app->useBluesteinFFT[axis_id]) && (axis_upload_id == 0)) || ((app->useBluesteinFFT[axis_id]) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && ((reverseBluesteinMultiUpload == 1) || (FFTPlan->numAxisUploads[axis_id] == 1)))) && (axis->specializationConstants.performR2C || FFTPlan->multiUploadR2C) && (!((app->configuration.isInputFormatted) && (app->configuration.inverseReturnToInputBuffer))) && (!app->configuration.isOutputFormatted)) {
		axisStride[1].data.i *= 2;
		axisStride[2].data.i *= 2;
		axisStride[3].data.i *= 2;
		axisStride[4].data.i *= 2;
	}
	if ((FFTPlan->multiUploadR2C) && (inverse) && (axis_id == 0) && (((!app->useBluesteinFFT[axis_id]) && (axis_upload_id == 0)) || ((app->useBluesteinFFT[axis_id]) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && ((reverseBluesteinMultiUpload == 1) || (FFTPlan->numAxisUploads[axis_id] == 1))))) {
		for (uint64_t i = 1; i < 5; i++) {
			axisStride[i].data.i /= 2;
		}
	}

	resFFT = VkFFTConfigureDescriptors(app, FFTPlan,axis,axis_id,axis_upload_id,inverse);
	if (resFFT != VKFFT_SUCCESS) {
		deleteVkFFT(app);
		return resFFT;
	}
	
	if (app->configuration.specifyOffsetsAtLaunch) {
		axis->specializationConstants.performPostCompilationInputOffset = 1;
		axis->specializationConstants.performPostCompilationOutputOffset = 1;
		if (app->configuration.performConvolution)
			axis->specializationConstants.performPostCompilationKernelOffset = 1;
	}
	
	resFFT = VkFFTCheckUpdateBufferSet(app, axis, 1, 0);
	if (resFFT != VKFFT_SUCCESS) {
		deleteVkFFT(app);
		return resFFT;
	}
	resFFT = VkFFTUpdateBufferSet(app, FFTPlan, axis, axis_id, axis_upload_id, inverse);
	if (resFFT != VKFFT_SUCCESS) {
		deleteVkFFT(app);
		return resFFT;
	}
	
	{
		uint64_t maxBatchCoalesced = app->configuration.coalescedMemory / axis->specializationConstants.complexSize;
		axis->groupedBatch = maxBatchCoalesced;
		/*if ((FFTPlan->actualFFTSizePerAxis[axis_id][0] < 4096) && (FFTPlan->actualFFTSizePerAxis[axis_id][1] < 512) && (FFTPlan->actualFFTSizePerAxis[axis_id][2] == 1)) {
			if (app->configuration.sharedMemorySize / axis->specializationConstants.fftDim >= app->configuration.coalescedMemory) {
				if (1024 / axis->specializationConstants.fftDim < maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim) {
					if (1024 / axis->specializationConstants.fftDim > axis->groupedBatch)
						axis->groupedBatch = 1024 / axis->specializationConstants.fftDim;
					else
						axis->groupedBatch = maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim;
				}
			}
		}
		else {
			axis->groupedBatch = (app->configuration.sharedMemorySize / axis->specializationConstants.fftDim >= app->configuration.coalescedMemory) ? maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim : axis->groupedBatch;
		}*/
		//if (axis->groupedBatch * (uint64_t)ceil(axis->specializationConstants.fftDim / 8.0) < app->configuration.warpSize) axis->groupedBatch = app->configuration.warpSize / (uint64_t)ceil(axis->specializationConstants.fftDim / 8.0);
		//axis->groupedBatch = (app->configuration.sharedMemorySize / axis->specializationConstants.fftDim >= app->configuration.coalescedMemory) ? maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim : axis->groupedBatch;
		if (((FFTPlan->numAxisUploads[axis_id] == 1) && (axis_id == 0)) || ((axis_id == 0) && (!axis->specializationConstants.reorderFourStep) && (axis_upload_id == 0))) {
			axis->groupedBatch = (maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim.data.i > axis->groupedBatch) ? maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim.data.i : axis->groupedBatch;
		}
		else {
			axis->groupedBatch = (maxSingleSizeStrided / axis->specializationConstants.fftDim.data.i > 1) ? maxSingleSizeStrided / axis->specializationConstants.fftDim.data.i * axis->groupedBatch : axis->groupedBatch;
		}
		//axis->groupedBatch = 8;
		//shared memory bank conflict resolve
//#if(VKFFT_BACKEND!=2)//for some reason, hip doesn't get performance increase from having variable shared memory strides.
		if (app->configuration.vendorID == 0x10DE) {
			if (FFTPlan->numAxisUploads[axis_id] == 2) {
				if ((axis_upload_id > 0) || (axis->specializationConstants.fftDim.data.i <= 512)) {
					if (axis->specializationConstants.fftDim.data.i * (64 / axis->specializationConstants.complexSize) <= maxSequenceLengthSharedMemory) {
						axis->groupedBatch = 64 / axis->specializationConstants.complexSize;
						maxBatchCoalesced = 64 / axis->specializationConstants.complexSize;
					}
					if (axis->specializationConstants.fftDim.data.i * (128 / axis->specializationConstants.complexSize) <= maxSequenceLengthSharedMemory) {
						axis->groupedBatch = 128 / axis->specializationConstants.complexSize;
						maxBatchCoalesced = 128 / axis->specializationConstants.complexSize;
					}
				}
			}
			//#endif
			if (FFTPlan->numAxisUploads[axis_id] == 3) {
				if (axis->specializationConstants.fftDim.data.i * (64 / axis->specializationConstants.complexSize) <= maxSequenceLengthSharedMemory) {
					axis->groupedBatch = 64 / axis->specializationConstants.complexSize;
					maxBatchCoalesced = 64 / axis->specializationConstants.complexSize;
				}
				if (axis->specializationConstants.fftDim.data.i * (128 / axis->specializationConstants.complexSize) <= maxSequenceLengthSharedMemory) {
					axis->groupedBatch = 128 / axis->specializationConstants.complexSize;
					maxBatchCoalesced = 128 / axis->specializationConstants.complexSize;
				}
			}
		}
		else {
			if ((FFTPlan->numAxisUploads[axis_id] == 2) && (axis_upload_id == 0) && (axis->specializationConstants.fftDim.data.i * maxBatchCoalesced <= maxSequenceLengthSharedMemory)) {
				axis->groupedBatch = (uint64_t)ceil(axis->groupedBatch / 2.0);
			}
			//#endif
			if ((FFTPlan->numAxisUploads[axis_id] == 3) && (axis_upload_id == 0) && (axis->specializationConstants.fftDim.data.i < maxSequenceLengthSharedMemory / (2 * axis->specializationConstants.complexSize))) {
				axis->groupedBatch = (uint64_t)ceil(axis->groupedBatch / 2.0);
			}
		}
		if (axis->groupedBatch < maxBatchCoalesced) axis->groupedBatch = maxBatchCoalesced;
		axis->groupedBatch = (axis->groupedBatch / maxBatchCoalesced) * maxBatchCoalesced;
		//half bandiwdth technique
		if (!((axis_id == 0) && (FFTPlan->numAxisUploads[axis_id] == 1)) && !((axis_id == 0) && (axis_upload_id == 0) && (!axis->specializationConstants.reorderFourStep)) && (axis->specializationConstants.fftDim.data.i > maxSingleSizeStrided)) {
			axis->groupedBatch = maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim.data.i;
			if (axis->groupedBatch == 0) axis->groupedBatch = 1;
		}

		if ((app->configuration.halfThreads) && (axis->groupedBatch * axis->specializationConstants.fftDim.data.i * axis->specializationConstants.complexSize >= app->configuration.sharedMemorySize))
			axis->groupedBatch = (uint64_t)ceil(axis->groupedBatch / 2.0);
		if (axis->groupedBatch > app->configuration.warpSize) axis->groupedBatch = (axis->groupedBatch / app->configuration.warpSize) * app->configuration.warpSize;
		if (axis->groupedBatch > 2 * maxBatchCoalesced) axis->groupedBatch = (axis->groupedBatch / (2 * maxBatchCoalesced)) * (2 * maxBatchCoalesced);
		if (axis->groupedBatch > 4 * maxBatchCoalesced) axis->groupedBatch = (axis->groupedBatch / (4 * maxBatchCoalesced)) * (4 * maxBatchCoalesced);
		//uint64_t maxThreadNum = (axis_id) ? (maxSingleSizeStrided * app->configuration.coalescedMemory / axis->specializationConstants.complexSize) / (axis->specializationConstants.min_registers_per_thread * axis->specializationConstants.registerBoost) : maxSequenceLengthSharedMemory / (axis->specializationConstants.min_registers_per_thread * axis->specializationConstants.registerBoost);
		//if (maxThreadNum > app->configuration.maxThreadsNum) maxThreadNum = app->configuration.maxThreadsNum;
		uint64_t maxThreadNum = app->configuration.maxThreadsNum;
		axis->specializationConstants.axisSwapped = 0;
		uint64_t r2cmult = (axis->specializationConstants.mergeSequencesR2C) ? 2 : 1;
		if (axis_id == 0) {
			if (axis_upload_id == 0) {
				axis->axisBlock[0] = (((uint64_t)ceil(axis->specializationConstants.fftDim.data.i / (double)axis->specializationConstants.min_registers_per_thread)) / axis->specializationConstants.registerBoost > 1) ? ((uint64_t)ceil(axis->specializationConstants.fftDim.data.i / (double)axis->specializationConstants.min_registers_per_thread)) / axis->specializationConstants.registerBoost : 1;
				if (axis->specializationConstants.useRaderMult) {
					uint64_t locMaxBatchCoalesced = ((axis_id == 0) && (((axis_upload_id == 0) && ((!app->configuration.reorderFourStep) || (app->useBluesteinFFT[axis_id]))) || (axis->specializationConstants.numAxisUploads == 1))) ? 1 : maxBatchCoalesced;
					uint64_t final_rader_thread_count = 0;
					for (uint64_t i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
						if (axis->specializationConstants.raderContainer[i].type == 1) {
							uint64_t temp_rader = (uint64_t)ceil((axis->specializationConstants.fftDim.data.i / (double)((axis->specializationConstants.rader_min_registers / 2) * 2)) / (double)((axis->specializationConstants.raderContainer[i].prime + 1) / 2));
							uint64_t active_rader = (uint64_t)ceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)temp_rader);
							if (active_rader > 1) {
								if ((((double)active_rader - (axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)temp_rader) >= 0.5) && ((((uint64_t)ceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)(active_rader - 1)) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2)) * locMaxBatchCoalesced) <= app->configuration.maxThreadsNum)) active_rader--;
							}
							uint64_t local_estimate_rader_threadnum = (uint64_t)ceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)active_rader) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2);

							uint64_t temp_rader_thread_count = ((uint64_t)ceil(axis->axisBlock[0] / (double)((axis->specializationConstants.raderContainer[i].prime + 1) / 2))) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2);
							if (temp_rader_thread_count < local_estimate_rader_threadnum) temp_rader_thread_count = local_estimate_rader_threadnum;
							if (temp_rader_thread_count > final_rader_thread_count) final_rader_thread_count = temp_rader_thread_count;
						}
					}
					axis->axisBlock[0] = final_rader_thread_count;
					if (axis->axisBlock[0] * axis->groupedBatch > maxThreadNum) axis->groupedBatch = locMaxBatchCoalesced;
				}
				if (axis->specializationConstants.useRaderFFT) {
					if (axis->axisBlock[0] < axis->specializationConstants.minRaderFFTThreadNum) axis->axisBlock[0] = axis->specializationConstants.minRaderFFTThreadNum;
				}
				if (axis->axisBlock[0] > maxThreadNum) axis->axisBlock[0] = maxThreadNum;
				if (axis->axisBlock[0] > app->configuration.maxComputeWorkGroupSize[0]) axis->axisBlock[0] = app->configuration.maxComputeWorkGroupSize[0];
				if (axis->specializationConstants.reorderFourStep && (FFTPlan->numAxisUploads[axis_id] > 1))
					axis->axisBlock[1] = axis->groupedBatch;
				else {
					//axis->axisBlock[1] = (axis->axisBlock[0] < app->configuration.warpSize) ? app->configuration.warpSize / axis->axisBlock[0] : 1;
					uint64_t estimate_batch = (((axis->axisBlock[0] / app->configuration.warpSize) == 1) && ((axis->axisBlock[0] / (double)app->configuration.warpSize) < 1.5)) ? app->configuration.aimThreads / app->configuration.warpSize : app->configuration.aimThreads / axis->axisBlock[0];
					if (estimate_batch == 0) estimate_batch = 1;
					axis->axisBlock[1] = ((axis->axisBlock[0] < app->configuration.aimThreads) && ((axis->axisBlock[0] < app->configuration.warpSize) || (axis->specializationConstants.useRader))) ? estimate_batch : 1;
				}

				uint64_t currentAxisBlock1 = axis->axisBlock[1];
				for (uint64_t i = currentAxisBlock1; i < 2 * currentAxisBlock1; i++) {
					if (((FFTPlan->numAxisUploads[0] > 1) && (!(((FFTPlan->actualFFTSizePerAxis[axis_id][0] / axis->specializationConstants.fftDim.data.i) % axis->axisBlock[1]) == 0))) || ((FFTPlan->numAxisUploads[0] == 1) && (!(((FFTPlan->actualFFTSizePerAxis[axis_id][1] / r2cmult) % axis->axisBlock[1]) == 0)))) {
						if (i * axis->specializationConstants.fftDim.data.i * axis->specializationConstants.complexSize <= allowedSharedMemory) axis->axisBlock[1] = i;
						i = 2 * currentAxisBlock1;
					}
				}
				if (((axis->specializationConstants.fftDim.data.i % 2 == 0) || (axis->axisBlock[0] < app->configuration.numSharedBanks / 4)) && (!(((!axis->specializationConstants.reorderFourStep) || (axis->specializationConstants.useBluesteinFFT)) && (FFTPlan->numAxisUploads[0] > 1))) && (axis->axisBlock[1] > 1) && (axis->axisBlock[1] * axis->specializationConstants.fftDim.data.i < maxSequenceLengthSharedMemoryPow2) && (!((app->configuration.performZeropadding[0] || app->configuration.performZeropadding[1] || app->configuration.performZeropadding[2])))) {
					//we plan to swap - this reduces bank conflicts
					axis->axisBlock[1] = (uint64_t)pow(2, (uint64_t)ceil(log2((double)axis->axisBlock[1])));
				}
				if ((FFTPlan->numAxisUploads[0] > 1) && ((uint64_t)ceil(FFTPlan->actualFFTSizePerAxis[axis_id][0] / axis->specializationConstants.fftDim.data.i) < axis->axisBlock[1])) axis->axisBlock[1] = (uint64_t)ceil(FFTPlan->actualFFTSizePerAxis[axis_id][0] / axis->specializationConstants.fftDim.data.i);
				if ((axis->specializationConstants.mergeSequencesR2C != 0) && (axis->specializationConstants.fftDim.data.i * axis->axisBlock[1] >= maxSequenceLengthSharedMemory)) {
					axis->specializationConstants.mergeSequencesR2C = 0;
					/*if ((!inverse) && (axis_id == 0) && (axis_upload_id == 0) && (!(app->configuration.isInputFormatted))) {
						axis->specializationConstants.inputStride[1] /= 2;
						axis->specializationConstants.inputStride[2] /= 2;
						axis->specializationConstants.inputStride[3] /= 2;
						axis->specializationConstants.inputStride[4] /= 2;
					}
					if ((inverse) && (axis_id == 0) && (axis_upload_id == 0) && (!((app->configuration.isInputFormatted) && (app->configuration.inverseReturnToInputBuffer))) && (!app->configuration.isOutputFormatted)) {
						axis->specializationConstants.outputStride[1] /= 2;
						axis->specializationConstants.outputStride[2] /= 2;
						axis->specializationConstants.outputStride[3] /= 2;
						axis->specializationConstants.outputStride[4] /= 2;
					}*/
					r2cmult = 1;
				}
				if ((FFTPlan->numAxisUploads[0] == 1) && ((uint64_t)ceil(FFTPlan->actualFFTSizePerAxis[axis_id][1] / (double)r2cmult) < axis->axisBlock[1])) axis->axisBlock[1] = (uint64_t)ceil(FFTPlan->actualFFTSizePerAxis[axis_id][1] / (double)r2cmult);
				if (app->configuration.vendorID == 0x10DE) {
					while ((axis->axisBlock[1] * axis->axisBlock[0] >= 2 * app->configuration.aimThreads) && (axis->axisBlock[1] > maxBatchCoalesced)) {
						axis->axisBlock[1] /= 2;
						if (axis->axisBlock[1] < maxBatchCoalesced) axis->axisBlock[1] = maxBatchCoalesced;
					}
				}
				if (axis->axisBlock[1] > app->configuration.maxComputeWorkGroupSize[1]) axis->axisBlock[1] = app->configuration.maxComputeWorkGroupSize[1];
				//if (axis->axisBlock[0] * axis->axisBlock[1] > app->configuration.maxThreadsNum) axis->axisBlock[1] /= 2;
				if (axis->axisBlock[0] * axis->axisBlock[1] > maxThreadNum) {
					for (uint64_t i = 1; i <= axis->axisBlock[1]; i++) {
						if ((axis->axisBlock[1] / i) * axis->axisBlock[0] <= maxThreadNum)
						{
							axis->axisBlock[1] /= i;
							i = axis->axisBlock[1] + 1;
						}

					}
				}
				while ((axis->axisBlock[1] * (axis->specializationConstants.fftDim.data.i / axis->specializationConstants.registerBoost)) > maxSequenceLengthSharedMemory) axis->axisBlock[1] /= 2;
				if (((axis->specializationConstants.fftDim.data.i % 2 == 0) || (axis->axisBlock[0] < app->configuration.numSharedBanks / 4)) && (!(((!axis->specializationConstants.reorderFourStep) || (axis->specializationConstants.useBluesteinFFT)) && (FFTPlan->numAxisUploads[0] > 1))) && (axis->axisBlock[1] > 1) && (axis->axisBlock[1] * axis->specializationConstants.fftDim.data.i < maxSequenceLengthSharedMemory) && (!((app->configuration.performZeropadding[0] || app->configuration.performZeropadding[1] || app->configuration.performZeropadding[2])))) {
					/*#if (VKFFT_BACKEND==0)
										if (((axis->specializationConstants.fftDim & (axis->specializationConstants.fftDim - 1)) != 0)) {
											uint64_t temp = axis->axisBlock[1];
											axis->axisBlock[1] = axis->axisBlock[0];
											axis->axisBlock[0] = temp;
											axis->specializationConstants.axisSwapped = 1;
										}
					#else*/
					uint64_t temp = axis->axisBlock[1];
					axis->axisBlock[1] = axis->axisBlock[0];
					axis->axisBlock[0] = temp;
					axis->specializationConstants.axisSwapped = 1;
					//#endif
				}
				axis->axisBlock[2] = 1;
				axis->axisBlock[3] = axis->specializationConstants.fftDim.data.i;
			}
			else {
				axis->axisBlock[1] = ((uint64_t)ceil(axis->specializationConstants.fftDim.data.i / (double)axis->specializationConstants.min_registers_per_thread) / axis->specializationConstants.registerBoost > 1) ? (uint64_t)ceil(axis->specializationConstants.fftDim.data.i / (double)axis->specializationConstants.min_registers_per_thread) / axis->specializationConstants.registerBoost : 1;
				if (axis->specializationConstants.useRaderMult) {
					uint64_t final_rader_thread_count = 0;
					for (uint64_t i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
						if (axis->specializationConstants.raderContainer[i].type == 1) {
							uint64_t temp_rader = (uint64_t)ceil((axis->specializationConstants.fftDim.data.i / (double)((axis->specializationConstants.rader_min_registers / 2) * 2)) / (double)((axis->specializationConstants.raderContainer[i].prime + 1) / 2));
							uint64_t active_rader = (uint64_t)ceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)temp_rader);
							if (active_rader > 1) {
								if ((((double)active_rader - (axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)temp_rader) >= 0.5) && ((((uint64_t)ceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)(active_rader - 1)) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2)) * maxBatchCoalesced) <= app->configuration.maxThreadsNum)) active_rader--;
							}
							uint64_t local_estimate_rader_threadnum = (uint64_t)ceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)active_rader) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2);

							uint64_t temp_rader_thread_count = ((uint64_t)ceil(axis->axisBlock[1] / (double)((axis->specializationConstants.raderContainer[i].prime + 1) / 2))) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2);
							if (temp_rader_thread_count < local_estimate_rader_threadnum) temp_rader_thread_count = local_estimate_rader_threadnum;
							if (temp_rader_thread_count > final_rader_thread_count) final_rader_thread_count = temp_rader_thread_count;
						}
					}
					axis->axisBlock[1] = final_rader_thread_count;
					if (axis->groupedBatch * axis->axisBlock[1] > maxThreadNum) axis->groupedBatch = maxBatchCoalesced;
				}
				if (axis->specializationConstants.useRaderFFT) {
					if (axis->axisBlock[1] < axis->specializationConstants.minRaderFFTThreadNum) axis->axisBlock[1] = axis->specializationConstants.minRaderFFTThreadNum;
				}

				uint64_t scale = app->configuration.aimThreads / axis->axisBlock[1] / axis->groupedBatch;
				if ((scale > 1) && ((axis->specializationConstants.fftDim.data.i * axis->groupedBatch * scale <= maxSequenceLengthSharedMemory))) axis->groupedBatch *= scale;

				axis->axisBlock[0] = (axis->specializationConstants.stageStartSize.data.i > axis->groupedBatch) ? axis->groupedBatch : axis->specializationConstants.stageStartSize.data.i;
				if (app->configuration.vendorID == 0x10DE) {
					while ((axis->axisBlock[1] * axis->axisBlock[0] >= 2 * app->configuration.aimThreads) && (axis->axisBlock[0] > maxBatchCoalesced)) {
						axis->axisBlock[0] /= 2;
						if (axis->axisBlock[0] < maxBatchCoalesced) axis->axisBlock[0] = maxBatchCoalesced;
					}
				}
				if (axis->axisBlock[0] > app->configuration.maxComputeWorkGroupSize[0]) axis->axisBlock[0] = app->configuration.maxComputeWorkGroupSize[0];
				if (axis->axisBlock[0] * axis->axisBlock[1] > maxThreadNum) {
					for (uint64_t i = 1; i <= axis->axisBlock[0]; i++) {
						if ((axis->axisBlock[0] / i) * axis->axisBlock[1] <= maxThreadNum)
						{
							axis->axisBlock[0] /= i;
							i = axis->axisBlock[0] + 1;
						}

					}
				}
				axis->axisBlock[2] = 1;
				axis->axisBlock[3] = axis->specializationConstants.fftDim.data.i;
			}

		}
		if (axis_id == 1) {

			axis->axisBlock[1] = ((uint64_t)ceil(axis->specializationConstants.fftDim.data.i / (double)axis->specializationConstants.min_registers_per_thread) / axis->specializationConstants.registerBoost > 1) ? ((uint64_t)ceil(axis->specializationConstants.fftDim.data.i / (double)axis->specializationConstants.min_registers_per_thread)) / axis->specializationConstants.registerBoost : 1;
			if (axis->specializationConstants.useRaderMult) {
				uint64_t final_rader_thread_count = 0;
				for (uint64_t i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
					if (axis->specializationConstants.raderContainer[i].type == 1) {
						uint64_t temp_rader = (uint64_t)ceil((axis->specializationConstants.fftDim.data.i / (double)((axis->specializationConstants.rader_min_registers / 2) * 2)) / (double)((axis->specializationConstants.raderContainer[i].prime + 1) / 2));
						uint64_t active_rader = (uint64_t)ceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)temp_rader);
						if (active_rader > 1) {
							if ((((double)active_rader - (axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)temp_rader) >= 0.5) && ((((uint64_t)ceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)(active_rader - 1)) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2)) * maxBatchCoalesced) <= app->configuration.maxThreadsNum)) active_rader--;
						}
						uint64_t local_estimate_rader_threadnum = (uint64_t)ceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)active_rader) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2);

						uint64_t temp_rader_thread_count = ((uint64_t)ceil(axis->axisBlock[1] / (double)((axis->specializationConstants.raderContainer[i].prime + 1) / 2))) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2);
						if (temp_rader_thread_count < local_estimate_rader_threadnum) temp_rader_thread_count = local_estimate_rader_threadnum;
						if (temp_rader_thread_count > final_rader_thread_count) final_rader_thread_count = temp_rader_thread_count;
					}
				}
				axis->axisBlock[1] = final_rader_thread_count;
				if (axis->groupedBatch * axis->axisBlock[1] > maxThreadNum) axis->groupedBatch = maxBatchCoalesced;
			}
			if (axis->specializationConstants.useRaderFFT) {
				if (axis->axisBlock[1] < axis->specializationConstants.minRaderFFTThreadNum) axis->axisBlock[1] = axis->specializationConstants.minRaderFFTThreadNum;
			}

			axis->axisBlock[0] = (FFTPlan->actualFFTSizePerAxis[axis_id][0] > axis->groupedBatch) ? axis->groupedBatch : FFTPlan->actualFFTSizePerAxis[axis_id][0];
			if (app->configuration.vendorID == 0x10DE) {
				while ((axis->axisBlock[1] * axis->axisBlock[0] >= 2 * app->configuration.aimThreads) && (axis->axisBlock[0] > maxBatchCoalesced)) {
					axis->axisBlock[0] /= 2;
					if (axis->axisBlock[0] < maxBatchCoalesced) axis->axisBlock[0] = maxBatchCoalesced;
				}
			}
			if (axis->axisBlock[0] > app->configuration.maxComputeWorkGroupSize[0]) axis->axisBlock[0] = app->configuration.maxComputeWorkGroupSize[0];
			if (axis->axisBlock[0] * axis->axisBlock[1] > maxThreadNum) {
				for (uint64_t i = 1; i <= axis->axisBlock[0]; i++) {
					if ((axis->axisBlock[0] / i) * axis->axisBlock[1] <= maxThreadNum)
					{
						axis->axisBlock[0] /= i;
						i = axis->axisBlock[0] + 1;
					}

				}
			}
			axis->axisBlock[2] = 1;
			axis->axisBlock[3] = axis->specializationConstants.fftDim.data.i;

		}
		if (axis_id == 2) {
			axis->axisBlock[1] = ((uint64_t)ceil(axis->specializationConstants.fftDim.data.i / (double)axis->specializationConstants.min_registers_per_thread) / axis->specializationConstants.registerBoost > 1) ? ((uint64_t)ceil(axis->specializationConstants.fftDim.data.i / (double)axis->specializationConstants.min_registers_per_thread)) / axis->specializationConstants.registerBoost : 1;
			if (axis->specializationConstants.useRaderMult) {
				uint64_t final_rader_thread_count = 0;
				for (uint64_t i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
					if (axis->specializationConstants.raderContainer[i].type == 1) {
						uint64_t temp_rader = (uint64_t)ceil((axis->specializationConstants.fftDim.data.i / (double)((axis->specializationConstants.rader_min_registers / 2) * 2)) / (double)((axis->specializationConstants.raderContainer[i].prime + 1) / 2));
						uint64_t active_rader = (uint64_t)ceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)temp_rader);
						if (active_rader > 1) {
							if ((((double)active_rader - (axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)temp_rader) >= 0.5) && ((((uint64_t)ceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)(active_rader - 1)) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2)) * maxBatchCoalesced) <= app->configuration.maxThreadsNum)) active_rader--;
						}
						uint64_t local_estimate_rader_threadnum = (uint64_t)ceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)active_rader) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2);

						uint64_t temp_rader_thread_count = ((uint64_t)ceil(axis->axisBlock[1] / (double)((axis->specializationConstants.raderContainer[i].prime + 1) / 2))) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2);
						if (temp_rader_thread_count < local_estimate_rader_threadnum) temp_rader_thread_count = local_estimate_rader_threadnum;
						if (temp_rader_thread_count > final_rader_thread_count) final_rader_thread_count = temp_rader_thread_count;
					}
				}
				axis->axisBlock[1] = final_rader_thread_count;
				if (axis->groupedBatch * axis->axisBlock[1] > maxThreadNum) axis->groupedBatch = maxBatchCoalesced;
			}
			if (axis->specializationConstants.useRaderFFT) {
				if (axis->axisBlock[1] < axis->specializationConstants.minRaderFFTThreadNum) axis->axisBlock[1] = axis->specializationConstants.minRaderFFTThreadNum;
			}

			axis->axisBlock[0] = (FFTPlan->actualFFTSizePerAxis[axis_id][0] > axis->groupedBatch) ? axis->groupedBatch : FFTPlan->actualFFTSizePerAxis[axis_id][0];
			if (app->configuration.vendorID == 0x10DE) {
				while ((axis->axisBlock[1] * axis->axisBlock[0] >= 2 * app->configuration.aimThreads) && (axis->axisBlock[0] > maxBatchCoalesced)) {
					axis->axisBlock[0] /= 2;
					if (axis->axisBlock[0] < maxBatchCoalesced) axis->axisBlock[0] = maxBatchCoalesced;
				}
			}
			if (axis->axisBlock[0] > app->configuration.maxComputeWorkGroupSize[0]) axis->axisBlock[0] = app->configuration.maxComputeWorkGroupSize[0];
			if (axis->axisBlock[0] * axis->axisBlock[1] > maxThreadNum) {
				for (uint64_t i = 1; i <= axis->axisBlock[0]; i++) {
					if ((axis->axisBlock[0] / i) * axis->axisBlock[1] <= maxThreadNum)
					{
						axis->axisBlock[0] /= i;
						i = axis->axisBlock[0] + 1;
					}

				}
			}
			axis->axisBlock[2] = 1;
			axis->axisBlock[3] = axis->specializationConstants.fftDim.data.i;
		}
		if ((axis->specializationConstants.axisSwapped) || (!((axis_id == 0) && (axis_upload_id == 0)))) axis->specializationConstants.stridedSharedLayout = 1;

		/*VkSpecializationMapEntry specializationMapEntries[36] = { {} };
		for (uint64_t i = 0; i < 36; i++) {
			specializationMapEntries[i].constantID = i + 1;
			specializationMapEntries[i].size = sizeof(uint64_t);
			specializationMapEntries[i].offset = i * sizeof(uint64_t);
		}
		VkSpecializationInfo specializationInfo = { 0 };
		specializationInfo.dataSize = 36 * sizeof(uint64_t);
		specializationInfo.mapEntryCount = 36;
		specializationInfo.pMapEntries = specializationMapEntries;*/
		axis->specializationConstants.localSize[0].type = 31;
		axis->specializationConstants.localSize[1].type = 31;
		axis->specializationConstants.localSize[2].type = 31;
		axis->specializationConstants.localSize[0].data.i = axis->axisBlock[0];
		axis->specializationConstants.localSize[1].data.i = axis->axisBlock[1];
		axis->specializationConstants.localSize[2].data.i = axis->axisBlock[2];
		axis->specializationConstants.numSubgroups = (uint64_t)ceil(axis->axisBlock[0] * axis->axisBlock[1] * axis->axisBlock[2] / (double)app->configuration.warpSize);
		//specializationInfo.pData = &axis->specializationConstants;
		//uint64_t registerBoost = (FFTPlan->numAxisUploads[axis_id] > 1) ? app->configuration.registerBoost4Step : app->configuration.registerBoost;

		axis->specializationConstants.numCoordinates = (app->configuration.matrixConvolution > 1) ? 1 : app->configuration.coordinateFeatures;
		axis->specializationConstants.matrixConvolution = app->configuration.matrixConvolution;
		axis->specializationConstants.coordinate.type = 31;
		axis->specializationConstants.coordinate.data.i = 0;
		axis->specializationConstants.batchID.type = 31;
		axis->specializationConstants.batchID.data.i = 0;
		axis->specializationConstants.numKernels.type = 31;
		axis->specializationConstants.numKernels.data.i = app->configuration.numberKernels;
		axis->specializationConstants.sharedMemSize = app->configuration.sharedMemorySize;
		axis->specializationConstants.sharedMemSizePow2 = app->configuration.sharedMemorySizePow2;
		axis->specializationConstants.normalize = (reverseBluesteinMultiUpload) ? 1 : app->configuration.normalize;
		axis->specializationConstants.size[0].type = 31;
		axis->specializationConstants.size[1].type = 31;
		axis->specializationConstants.size[2].type = 31;
		axis->specializationConstants.size[0].data.i = FFTPlan->actualFFTSizePerAxis[axis_id][0];
		axis->specializationConstants.size[1].data.i = FFTPlan->actualFFTSizePerAxis[axis_id][1];
		axis->specializationConstants.size[2].data.i = FFTPlan->actualFFTSizePerAxis[axis_id][2];

		for (uint64_t i = 0; i < 3; i++) {
			axis->specializationConstants.frequencyZeropadding = app->configuration.frequencyZeroPadding;
			axis->specializationConstants.performZeropaddingFull[i] = app->configuration.performZeropadding[i]; // don't read if input is zeropadded (0 - off, 1 - on)
			axis->specializationConstants.fft_zeropad_left_full[i].type = 31;
			axis->specializationConstants.fft_zeropad_left_full[i].data.i = app->configuration.fft_zeropad_left[i];
			axis->specializationConstants.fft_zeropad_right_full[i].type = 31;
			axis->specializationConstants.fft_zeropad_right_full[i].data.i = app->configuration.fft_zeropad_right[i];
		}
		if (axis->specializationConstants.useBluesteinFFT && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && ((reverseBluesteinMultiUpload == 0) || (FFTPlan->numAxisUploads[axis_id] == 1))) {
			axis->specializationConstants.zeropadBluestein[0] = 1;
			axis->specializationConstants.fft_zeropad_Bluestein_left_read[axis_id].type = 31;
			axis->specializationConstants.fft_zeropad_Bluestein_left_read[axis_id].data.i = app->configuration.size[axis_id];
			if ((FFTPlan->multiUploadR2C) && (axis_id == 0)) axis->specializationConstants.fft_zeropad_Bluestein_left_read[axis_id].data.i /= 2;
			if (app->configuration.performDCT == 1) axis->specializationConstants.fft_zeropad_Bluestein_left_read[axis_id].data.i = 2 * axis->specializationConstants.fft_zeropad_Bluestein_left_read[axis_id].data.i - 2;
			if ((app->configuration.performDCT == 4) && (app->configuration.size[axis_id] % 2 == 0)) axis->specializationConstants.fft_zeropad_Bluestein_left_read[axis_id].data.i /= 2;
			axis->specializationConstants.fft_zeropad_Bluestein_right_read[axis_id].type = 31;
			axis->specializationConstants.fft_zeropad_Bluestein_right_read[axis_id].data.i = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id];
		}
		if (axis->specializationConstants.useBluesteinFFT && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && ((reverseBluesteinMultiUpload == 1) || (FFTPlan->numAxisUploads[axis_id] == 1))) {
			axis->specializationConstants.zeropadBluestein[1] = 1;
			axis->specializationConstants.fft_zeropad_Bluestein_left_write[axis_id].type = 31;
			axis->specializationConstants.fft_zeropad_Bluestein_left_write[axis_id].data.i = app->configuration.size[axis_id];
			if ((FFTPlan->multiUploadR2C) && (axis_id == 0)) axis->specializationConstants.fft_zeropad_Bluestein_left_write[axis_id].data.i /= 2;
			if (app->configuration.performDCT == 1) axis->specializationConstants.fft_zeropad_Bluestein_left_write[axis_id].data.i = 2 * axis->specializationConstants.fft_zeropad_Bluestein_left_write[axis_id].data.i - 2;
			if ((app->configuration.performDCT == 4) && (app->configuration.size[axis_id] % 2 == 0)) axis->specializationConstants.fft_zeropad_Bluestein_left_write[axis_id].data.i /= 2;
			axis->specializationConstants.fft_zeropad_Bluestein_right_write[axis_id].type = 31;
			axis->specializationConstants.fft_zeropad_Bluestein_right_write[axis_id].data.i = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id];
		}
		uint64_t zeropad_r2c_multiupload_scale = ((axis_id == 0) && (FFTPlan->multiUploadR2C)) ? 2 : 1;
		if ((inverse)) {
			if ((app->configuration.frequencyZeroPadding) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (reverseBluesteinMultiUpload != 1)) {
				axis->specializationConstants.zeropad[0] = app->configuration.performZeropadding[axis_id];
				axis->specializationConstants.fft_zeropad_left_read[axis_id].type = 31;
				axis->specializationConstants.fft_zeropad_left_read[axis_id].data.i = app->configuration.fft_zeropad_left[axis_id] / zeropad_r2c_multiupload_scale;
				axis->specializationConstants.fft_zeropad_right_read[axis_id].type = 31;
				axis->specializationConstants.fft_zeropad_right_read[axis_id].data.i = app->configuration.fft_zeropad_right[axis_id] / zeropad_r2c_multiupload_scale;
			}
			else
				axis->specializationConstants.zeropad[0] = 0;
			if ((!app->configuration.frequencyZeroPadding) && (((axis_upload_id == 0) && (!((axis->specializationConstants.useBluesteinFFT) || (app->configuration.performConvolution)))) || ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && ((((reverseBluesteinMultiUpload == 1) || (FFTPlan->numAxisUploads[axis_id] == 1)) || (app->configuration.performConvolution)))))) {
				axis->specializationConstants.zeropad[1] = app->configuration.performZeropadding[axis_id];
				axis->specializationConstants.fft_zeropad_left_write[axis_id].type = 31;
				axis->specializationConstants.fft_zeropad_left_write[axis_id].data.i = app->configuration.fft_zeropad_left[axis_id] / zeropad_r2c_multiupload_scale;
				axis->specializationConstants.fft_zeropad_right_write[axis_id].type = 31;
				axis->specializationConstants.fft_zeropad_right_write[axis_id].data.i = app->configuration.fft_zeropad_right[axis_id] / zeropad_r2c_multiupload_scale;
			}
			else
				axis->specializationConstants.zeropad[1] = 0;
		}
		else {
			if ((!app->configuration.frequencyZeroPadding) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (reverseBluesteinMultiUpload != 1)) {
				axis->specializationConstants.zeropad[0] = app->configuration.performZeropadding[axis_id];
				axis->specializationConstants.fft_zeropad_left_read[axis_id].type = 31;
				axis->specializationConstants.fft_zeropad_left_read[axis_id].data.i = app->configuration.fft_zeropad_left[axis_id] / zeropad_r2c_multiupload_scale;
				axis->specializationConstants.fft_zeropad_right_read[axis_id].type = 31;
				axis->specializationConstants.fft_zeropad_right_read[axis_id].data.i = app->configuration.fft_zeropad_right[axis_id] / zeropad_r2c_multiupload_scale;
			}
			else
				axis->specializationConstants.zeropad[0] = 0;
			if (((app->configuration.frequencyZeroPadding) && (((axis_upload_id == 0) && (!axis->specializationConstants.useBluesteinFFT)) || ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (axis->specializationConstants.useBluesteinFFT && ((reverseBluesteinMultiUpload == 1) || (FFTPlan->numAxisUploads[axis_id] == 1)))))) || (((!app->configuration.frequencyZeroPadding) && (app->configuration.FFTdim - 1 == axis_id) && (axis_upload_id == 0) && (FFTPlan->numAxisUploads[axis_id] == 1) && (app->configuration.performConvolution)))) {
				axis->specializationConstants.zeropad[1] = app->configuration.performZeropadding[axis_id];
				axis->specializationConstants.fft_zeropad_left_write[axis_id].type = 31;
				axis->specializationConstants.fft_zeropad_left_write[axis_id].data.i = app->configuration.fft_zeropad_left[axis_id] / zeropad_r2c_multiupload_scale;
				axis->specializationConstants.fft_zeropad_right_write[axis_id].type = 31;
				axis->specializationConstants.fft_zeropad_right_write[axis_id].data.i = app->configuration.fft_zeropad_right[axis_id] / zeropad_r2c_multiupload_scale;
			}
			else
				axis->specializationConstants.zeropad[1] = 0;
		}
		if ((app->configuration.FFTdim - 1 == axis_id) && (axis_upload_id == 0) && (app->configuration.performConvolution)) {
			axis->specializationConstants.convolutionStep = 1;
		}
		else
			axis->specializationConstants.convolutionStep = 0;
		if (app->useBluesteinFFT[axis_id] && (axis_upload_id == 0))
			axis->specializationConstants.BluesteinConvolutionStep = 1;
		else
			axis->specializationConstants.BluesteinConvolutionStep = 0;

		if (app->useBluesteinFFT[axis_id] && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (reverseBluesteinMultiUpload == 0))
			axis->specializationConstants.BluesteinPreMultiplication = 1;
		else
			axis->specializationConstants.BluesteinPreMultiplication = 0;
		if (app->useBluesteinFFT[axis_id] && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && ((reverseBluesteinMultiUpload == 1) || (FFTPlan->numAxisUploads[axis_id] == 1)))
			axis->specializationConstants.BluesteinPostMultiplication = 1;
		else
			axis->specializationConstants.BluesteinPostMultiplication = 0;


		uint64_t tempSize[3] = { FFTPlan->actualFFTSizePerAxis[axis_id][0], FFTPlan->actualFFTSizePerAxis[axis_id][1], FFTPlan->actualFFTSizePerAxis[axis_id][2] };


		if (axis_id == 0) {
			if (axis_upload_id == 0)
				tempSize[0] = FFTPlan->actualFFTSizePerAxis[axis_id][0] / axis->specializationConstants.fftDim.data.i / axis->axisBlock[1];
			else
				tempSize[0] = FFTPlan->actualFFTSizePerAxis[axis_id][0] / axis->specializationConstants.fftDim.data.i / axis->axisBlock[0];
			if ((FFTPlan->actualPerformR2CPerAxis[axis_id] == 1) && (axis->specializationConstants.mergeSequencesR2C)) tempSize[1] = (uint64_t)ceil(tempSize[1] / 2.0);
			tempSize[2] *= app->configuration.numberKernels * app->configuration.numberBatches;
			if (!(axis->specializationConstants.convolutionStep && (app->configuration.matrixConvolution > 1))) tempSize[2] *= app->configuration.coordinateFeatures;
			//if (app->configuration.performZeropadding[1]) tempSize[1] = (uint64_t)ceil(tempSize[1] / 2.0);
			//if (app->configuration.performZeropadding[2]) tempSize[2] = (uint64_t)ceil(tempSize[2] / 2.0);
		}
		if (axis_id == 1) {
			tempSize[0] = (uint64_t)ceil(FFTPlan->actualFFTSizePerAxis[axis_id][0] / (double)axis->axisBlock[0] * FFTPlan->actualFFTSizePerAxis[axis_id][1] / (double)axis->specializationConstants.fftDim.data.i);
			tempSize[1] = 1;
			tempSize[2] = FFTPlan->actualFFTSizePerAxis[axis_id][2];
			tempSize[2] *= app->configuration.numberKernels * app->configuration.numberBatches;
			if (!(axis->specializationConstants.convolutionStep && (app->configuration.matrixConvolution > 1))) tempSize[2] *= app->configuration.coordinateFeatures;
			//if (app->configuration.actualPerformR2C == 1) tempSize[0] = (uint64_t)ceil(tempSize[0] / 2.0);
			//if (app->configuration.performZeropadding[2]) tempSize[2] = (uint64_t)ceil(tempSize[2] / 2.0);
		}
		if (axis_id == 2) {
			tempSize[0] = (uint64_t)ceil(FFTPlan->actualFFTSizePerAxis[axis_id][0] / (double)axis->axisBlock[0] * FFTPlan->actualFFTSizePerAxis[axis_id][2] / (double)axis->specializationConstants.fftDim.data.i);
			tempSize[1] = 1;
			tempSize[2] = FFTPlan->actualFFTSizePerAxis[axis_id][1];
			tempSize[2] *= app->configuration.numberKernels * app->configuration.numberBatches;
			if (!(axis->specializationConstants.convolutionStep && (app->configuration.matrixConvolution > 1))) tempSize[2] *= app->configuration.coordinateFeatures;
			//if (app->configuration.actualPerformR2C == 1) tempSize[0] = (uint64_t)ceil(tempSize[0] / 2.0);

		}
		if ((app->configuration.maxComputeWorkGroupCount[0] > app->configuration.maxComputeWorkGroupCount[1]) && (tempSize[1] > app->configuration.maxComputeWorkGroupCount[1]) && (tempSize[1] > tempSize[0]) && (tempSize[1] >= tempSize[2])) {
			uint64_t temp_tempSize = tempSize[0];
			tempSize[0] = tempSize[1];
			tempSize[1] = temp_tempSize;
			axis->specializationConstants.swapComputeWorkGroupID = 1;
		}
		else {
			if ((app->configuration.maxComputeWorkGroupCount[0] > app->configuration.maxComputeWorkGroupCount[2]) && (tempSize[2] > app->configuration.maxComputeWorkGroupCount[2]) && (tempSize[2] > tempSize[0]) && (tempSize[2] >= tempSize[1])) {
				uint64_t temp_tempSize = tempSize[0];
				tempSize[0] = tempSize[2];
				tempSize[2] = temp_tempSize;
				axis->specializationConstants.swapComputeWorkGroupID = 2;
			}
		}
		if (tempSize[0] > app->configuration.maxComputeWorkGroupCount[0]) axis->specializationConstants.performWorkGroupShift[0] = 1;
		else  axis->specializationConstants.performWorkGroupShift[0] = 0;
		if (tempSize[1] > app->configuration.maxComputeWorkGroupCount[1]) axis->specializationConstants.performWorkGroupShift[1] = 1;
		else  axis->specializationConstants.performWorkGroupShift[1] = 0;
		if (tempSize[2] > app->configuration.maxComputeWorkGroupCount[2]) axis->specializationConstants.performWorkGroupShift[2] = 1;
		else  axis->specializationConstants.performWorkGroupShift[2] = 0;

		axis->specializationConstants.LUT = (app->configuration.useLUT == 1) ? 1 : 0;
		axis->specializationConstants.LUT_4step = (app->configuration.useLUT_4step == 1) ? 1 : 0;
		{
			axis->pushConstants.structSize = 0;
			if (axis->specializationConstants.performWorkGroupShift[0]) {
				axis->pushConstants.performWorkGroupShift[0] = 1;
				axis->pushConstants.structSize += 1;
			}
			if (axis->specializationConstants.performWorkGroupShift[1]) {
				axis->pushConstants.performWorkGroupShift[1] = 1;
				axis->pushConstants.structSize += 1;
			}
			if (axis->specializationConstants.performWorkGroupShift[2]) {
				axis->pushConstants.performWorkGroupShift[2] = 1;
				axis->pushConstants.structSize += 1;
			}
			if (axis->specializationConstants.performPostCompilationInputOffset) {
				axis->pushConstants.performPostCompilationInputOffset = 1;
				axis->pushConstants.structSize += 1;
			}
			if (axis->specializationConstants.performPostCompilationOutputOffset) {
				axis->pushConstants.performPostCompilationOutputOffset = 1;
				axis->pushConstants.structSize += 1;
			}
			if (axis->specializationConstants.performPostCompilationKernelOffset) {
				axis->pushConstants.performPostCompilationKernelOffset = 1;
				axis->pushConstants.structSize += 1;
			}
			if (app->configuration.useUint64)
				axis->pushConstants.structSize *= sizeof(uint64_t);
			else
				axis->pushConstants.structSize *= sizeof(uint32_t);
			axis->specializationConstants.pushConstantsStructSize = axis->pushConstants.structSize;
		}
		//uint64_t LUT = app->configuration.useLUT;
		uint64_t type = 0;
		if ((axis_id == 0) && (axis_upload_id == 0)) type = 0;
		if (axis_id != 0) type = 1;
		if ((axis_id == 0) && (axis_upload_id > 0)) type = 2;
		//if ((axis->specializationConstants.fftDim == 8 * maxSequenceLengthSharedMemory) && (app->configuration.registerBoost >= 8)) axis->specializationConstants.registerBoost = 8;
		if ((axis_id == 0) && (!axis->specializationConstants.actualInverse) && (FFTPlan->actualPerformR2CPerAxis[axis_id])) type = 5;
		if ((axis_id == 0) && (axis->specializationConstants.actualInverse) && (FFTPlan->actualPerformR2CPerAxis[axis_id])) type = 6;
		if ((axis_id == 0) && (app->configuration.performDCT == 1)) type = 110;
		if ((axis_id != 0) && (app->configuration.performDCT == 1)) type = 111;
		if ((axis_id == 0) && (((app->configuration.performDCT == 2) && (!inverse)) || ((app->configuration.performDCT == 3) && (inverse)))) type = 120;
		if ((axis_id != 0) && (((app->configuration.performDCT == 2) && (!inverse)) || ((app->configuration.performDCT == 3) && (inverse)))) type = 121;
		if ((axis_id == 0) && (((app->configuration.performDCT == 2) && (inverse)) || ((app->configuration.performDCT == 3) && (!inverse)))) type = 130;
		if ((axis_id != 0) && (((app->configuration.performDCT == 2) && (inverse)) || ((app->configuration.performDCT == 3) && (!inverse)))) type = 131;
		if ((axis_id == 0) && (app->configuration.performDCT == 4) && ((app->configuration.size[axis_id] % 2) == 0)) type = 142;
		if ((axis_id == 0) && (app->configuration.performDCT == 4) && ((app->configuration.size[axis_id] % 2) == 1)) type = 144;
		if ((axis_id != 0) && (app->configuration.performDCT == 4) && ((app->configuration.size[axis_id] % 2) == 0)) type = 143;
		if ((axis_id != 0) && (app->configuration.performDCT == 4) && ((app->configuration.size[axis_id] % 2) == 1)) type = 145;
		
		resFFT = initMemoryParametersAPI(app, &axis->specializationConstants);
		if (resFFT != VKFFT_SUCCESS) {
			deleteVkFFT(app);
			return resFFT;
		}

		switch (type) {
		case 0: case 1: case 2: case 3: case 4: case 6:
			axis->specializationConstants.inputMemoryCode = axis->specializationConstants.vecTypeInputMemoryCode;
			switch ((axis->specializationConstants.inputMemoryCode % 100) / 10) {
			case 0:
				axis->specializationConstants.inputNumberByteSize = 4;
				break;
			case 1:
				axis->specializationConstants.inputNumberByteSize = 8;
				break;
			case 2:
				axis->specializationConstants.inputNumberByteSize = 16;
				break;
			}
			break;
		case 5: case 110: case 111: case 120: case 121: case 130: case 131: case 140: case 141: case 142: case 143: case 144: case 145:
			axis->specializationConstants.inputMemoryCode = axis->specializationConstants.floatTypeInputMemoryCode;
			switch ((axis->specializationConstants.inputMemoryCode % 100) / 10) {
			case 0:
				axis->specializationConstants.inputNumberByteSize = 2;
				break;
			case 1:
				axis->specializationConstants.inputNumberByteSize = 4;
				break;
			case 2:
				axis->specializationConstants.inputNumberByteSize = 8;
				break;
			}
			break;
		}
		switch (type) {
		case 0: case 1: case 2: case 3: case 4: case 5:
			axis->specializationConstants.outputMemoryCode = axis->specializationConstants.vecTypeOutputMemoryCode;
			switch ((axis->specializationConstants.outputMemoryCode % 100) / 10) {
			case 0:
				axis->specializationConstants.outputNumberByteSize = 4;
				break;
			case 1:
				axis->specializationConstants.outputNumberByteSize = 8;
				break;
			case 2:
				axis->specializationConstants.outputNumberByteSize = 16;
				break;
			}
			break;
		case 6: case 110: case 111: case 120: case 121: case 130: case 131: case 140: case 141: case 142: case 143: case 144: case 145:
			axis->specializationConstants.outputMemoryCode = axis->specializationConstants.floatTypeOutputMemoryCode;
			switch ((axis->specializationConstants.outputMemoryCode % 100) / 10) {
			case 0:
				axis->specializationConstants.outputNumberByteSize = 2;
				break;
			case 1:
				axis->specializationConstants.outputNumberByteSize = 4;
				break;
			case 2:
				axis->specializationConstants.outputNumberByteSize = 8;
				break;
			}
			break;
		}
		switch ((axis->specializationConstants.vecTypeCode % 100) / 10) {
		case 0:
			axis->specializationConstants.kernelNumberByteSize = 4;
			break;
		case 1:
			axis->specializationConstants.kernelNumberByteSize = 8;
			break;
		case 2:
			axis->specializationConstants.kernelNumberByteSize = 16;
			break;
		}

		resFFT = initParametersAPI(app, &axis->specializationConstants);
		if (resFFT != VKFFT_SUCCESS) {
			deleteVkFFT(app);
			return resFFT;
		}

		axis->specializationConstants.code0 = (char*)malloc(sizeof(char) * app->configuration.maxCodeLength);
		char* code0 = axis->specializationConstants.code0;
		if (!code0) {
			deleteVkFFT(app);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
#if(VKFFT_BACKEND==0)
		sprintf(axis->VkFFTFunctionName, "main");
#else
		sprintf(axis->VkFFTFunctionName, "VkFFT_main");
#endif
		resFFT = VkShaderGen_FFT(&axis->specializationConstants, type);
		if (resFFT != VKFFT_SUCCESS) {
			deleteVkFFT(app);
			return resFFT;
		}
		resFFT = VkFFT_CompileKernel(app, axis);
		if (resFFT != VKFFT_SUCCESS) {
			deleteVkFFT(app);
			return resFFT;
		}
		if (!app->configuration.keepShaderCode) {
			free(code0);
			code0 = 0;
			axis->specializationConstants.code0 = 0;
		}
	}
	freeMemoryParametersAPI(app, &axis->specializationConstants);
	freeParametersAPI(app, &axis->specializationConstants);
	if (axis->specializationConstants.axisSwapped) {//swap back for correct dispatch
		uint64_t temp = axis->axisBlock[1];
		axis->axisBlock[1] = axis->axisBlock[0];
		axis->axisBlock[0] = temp;
		axis->specializationConstants.axisSwapped = 0;
	}
	return resFFT;
}

#endif
