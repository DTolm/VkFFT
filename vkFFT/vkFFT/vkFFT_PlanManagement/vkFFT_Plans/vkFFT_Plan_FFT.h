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
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"

#include "vkFFT/vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_ManageMemory.h"
#include "vkFFT/vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_InitAPIParameters.h"
#include "vkFFT/vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_CompileKernel.h"
#include "vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_ManageLUT.h"
#include "vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_AxisBlockSplitter.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel2/vkFFT_FFT.h"
#include "vkFFT/vkFFT_AppManagement/vkFFT_DeleteApp.h"
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
	axis->specializationConstants.axis_id = (int)axis_id;
	axis->specializationConstants.axis_upload_id = (int)axis_upload_id;
    axis->specializationConstants.numFFTdims = (int)app->configuration.FFTdim;
	if ((app->configuration.FFTdim == 1) && (FFTPlan->actualFFTSizePerAxis[axis_id][1] == 1) && ((app->configuration.numberBatches > 1) || (app->actualNumBatches > 1)) && (!app->configuration.performConvolution) && (app->configuration.coordinateFeatures == 1)) {
		if (app->configuration.numberBatches > 1) {
			app->actualNumBatches = app->configuration.numberBatches;
			app->configuration.numberBatches = 1;
		}
		FFTPlan->actualFFTSizePerAxis[axis_id][1] = app->actualNumBatches;
	}
	axis->specializationConstants.numBatches.type = 31;
	axis->specializationConstants.numBatches.data.i = (int64_t)app->configuration.numberBatches;
	axis->specializationConstants.warpSize = (int)app->configuration.warpSize;
	axis->specializationConstants.numSharedBanks = (int)app->configuration.numSharedBanks;
	axis->specializationConstants.useUint64 = (int)app->configuration.useUint64;
#if(VKFFT_BACKEND==2)
	axis->specializationConstants.useStrict32BitAddress = app->configuration.useStrict32BitAddress;
#endif
	axis->specializationConstants.disableSetLocale = (int)app->configuration.disableSetLocale;

	axis->specializationConstants.numAxisUploads = (int)FFTPlan->numAxisUploads[axis_id];
	axis->specializationConstants.fixMinRaderPrimeMult = (int)app->configuration.fixMinRaderPrimeMult;
	axis->specializationConstants.fixMaxRaderPrimeMult = (int)app->configuration.fixMaxRaderPrimeMult;
	axis->specializationConstants.fixMinRaderPrimeFFT = (int)app->configuration.fixMinRaderPrimeFFT;
	axis->specializationConstants.fixMaxRaderPrimeFFT = (int)app->configuration.fixMaxRaderPrimeFFT;

	axis->specializationConstants.raderUintLUT = (axis->specializationConstants.useRader) ? (int)app->configuration.useRaderUintLUT : 0;
	axis->specializationConstants.inline_rader_g_pow = (axis->specializationConstants.raderUintLUT) ? 2 : 1;
	axis->specializationConstants.inline_rader_kernel = (app->configuration.useLUT == 1) ? 0 : 1;
	axis->specializationConstants.supportAxis = 0;
	axis->specializationConstants.symmetricKernel = (int)app->configuration.symmetricKernel;
	axis->specializationConstants.conjugateConvolution = (int)app->configuration.conjugateConvolution;
	axis->specializationConstants.crossPowerSpectrumNormalization = (int)app->configuration.crossPowerSpectrumNormalization;

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
    axis->specializationConstants.dispatchZactualFFTSize.data.i = 1;
    for (int i = 1; i < app->configuration.FFTdim; i++){
        if (((axis_id>0)||(i>1)) && (i != axis_id)){
            axis->specializationConstants.dispatchZactualFFTSize.data.i *= FFTPlan->actualFFTSizePerAxis[axis_id][i];
        }
    }
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
		axis->specializationConstants.actualInverse = (int)inverse;
		axis->specializationConstants.inverse = (int)!inverse;
	}
	else {
		if (app->configuration.performDCT == 4) {
			axis->specializationConstants.actualInverse = (int)inverse;
			axis->specializationConstants.inverse = 1;
		}
		else {
			axis->specializationConstants.actualInverse = (int)inverse;
			axis->specializationConstants.inverse = (int)inverse;
		}
	}
	if (app->useBluesteinFFT[axis_id]) {
		axis->specializationConstants.actualInverse = (int)inverse;
		axis->specializationConstants.inverse = (int)reverseBluesteinMultiUpload;
		if (app->configuration.performDCT == 3) {
			axis->specializationConstants.inverseBluestein = (int)!inverse;
		}
		else {
			if (app->configuration.performDCT == 4) {
				axis->specializationConstants.inverseBluestein = 1;
			}
			else {
				axis->specializationConstants.inverseBluestein = (int)inverse;
			}
		}
	}
	axis->specializationConstants.reverseBluesteinMultiUpload = (int)reverseBluesteinMultiUpload;

	axis->specializationConstants.reorderFourStep = ((FFTPlan->numAxisUploads[axis_id] > 1) && (!app->useBluesteinFFT[axis_id])) ? (int)app->configuration.reorderFourStep : 0;

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

	axis->specializationConstants.performR2C = (int)FFTPlan->actualPerformR2CPerAxis[axis_id];
	axis->specializationConstants.performR2CmultiUpload = (int)FFTPlan->multiUploadR2C;
	if (app->configuration.performDCT == 3) {
		axis->specializationConstants.performDCT = 2;
	}
	else {
		axis->specializationConstants.performDCT = (int)app->configuration.performDCT;
	}
	if ((axis->specializationConstants.performR2CmultiUpload) && (app->configuration.size[0] % 2 != 0)) return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH_R2C;
	uint64_t additionalR2Cshared = 0;
	if ((axis->specializationConstants.performR2C || ((axis->specializationConstants.performDCT == 2) || ((axis->specializationConstants.performDCT == 4) && ((axis->specializationConstants.fftDim.data.i % 2) != 0)))) && (axis->specializationConstants.axis_id == 0) && (!axis->specializationConstants.performR2CmultiUpload)) {
		additionalR2Cshared = ((axis->specializationConstants.fftDim.data.i % 2) == 0) ? 2 : 1;
		if ((axis->specializationConstants.performDCT == 2) || ((axis->specializationConstants.performDCT == 4) && ((axis->specializationConstants.fftDim.data.i % 2) != 0))) additionalR2Cshared = 1;
	}
	axis->specializationConstants.mergeSequencesR2C = (((axis->specializationConstants.fftDim.data.i + additionalR2Cshared) <= maxSequenceLengthSharedMemory) && (FFTPlan->actualFFTSizePerAxis[axis_id][1] > 1) && ((FFTPlan->actualPerformR2CPerAxis[axis_id]) || (((app->configuration.performDCT == 3) || (app->configuration.performDCT == 2) || (app->configuration.performDCT == 1) || ((app->configuration.performDCT == 4) && ((app->configuration.size[axis_id] % 2) != 0))) && (axis_id == 0)))) ? (1 - (int)app->configuration.disableMergeSequencesR2C) : 0;
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
		ze_device_mem_alloc_desc_t device_desc = VKFFT_ZERO_INIT;
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

	PfContainer* axisStride = axis->specializationConstants.inputStride;
	uint64_t* usedStride = app->configuration.bufferStride;
	if ((!inverse) && (axis_id == app->firstAxis) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (app->configuration.isInputFormatted)) usedStride = app->configuration.inputBufferStride;
	if ((inverse) && (axis_id == app->lastAxis) && ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && ((app->useBluesteinFFT[axis_id] && (reverseBluesteinMultiUpload == 0)) || (!app->useBluesteinFFT[axis_id])) && (!app->configuration.performConvolution)) && (app->configuration.isInputFormatted) && (!app->configuration.inverseReturnToInputBuffer)) usedStride = app->configuration.inputBufferStride;
	axisStride[0].type = 31;
	axisStride[0].data.i = 1;

    if (axis_id == 0) {
        for (int i = 1; i < app->configuration.FFTdim; i++){
            axisStride[i].type = 31;
            axisStride[i].data.i = usedStride[i-1];
        }
    }else{
        int locStrideOrder = 2;
        for (int i = 1; i < app->configuration.FFTdim; i++){
            if(i==axis_id){
                axisStride[1].type = 31;
                axisStride[1].data.i = usedStride[i-1];
            }else{
                axisStride[locStrideOrder].type = 31;
                axisStride[locStrideOrder].data.i = usedStride[i-1];
                locStrideOrder++;
            }
        }
        
    }

	axisStride[app->configuration.FFTdim].type = 31;
	axisStride[app->configuration.FFTdim].data.i = usedStride[app->configuration.FFTdim-1];

	axisStride[app->configuration.FFTdim+1].type = 31;
	axisStride[app->configuration.FFTdim+1].data.i = axisStride[app->configuration.FFTdim].data.i * app->configuration.coordinateFeatures;
	if (app->useBluesteinFFT[axis_id] && (FFTPlan->numAxisUploads[axis_id] > 1) && (!((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (reverseBluesteinMultiUpload == 0)))) {
		axisStride[0].data.i = 1;
        int64_t prevStride = axisStride[0].data.i;
        
		if (axis_id == 0) {
            for (int i = 1; i < app->configuration.FFTdim; i++){
                axisStride[i].data.i = prevStride * FFTPlan->actualFFTSizePerAxis[axis_id][i-1];
                prevStride = axisStride[i].data.i;
            }
        }else{
            int locStrideOrder = 2;
            for (int i = 1; i < app->configuration.FFTdim; i++){
                if(i==axis_id){
                    axisStride[1].data.i = prevStride * FFTPlan->actualFFTSizePerAxis[axis_id][i-1];
                    prevStride = axisStride[1].data.i;
                }else{
                    axisStride[locStrideOrder].data.i = prevStride * FFTPlan->actualFFTSizePerAxis[axis_id][i-1];
                    prevStride = axisStride[locStrideOrder].data.i;
                    locStrideOrder++;
                }
            }
            
        }

		axisStride[app->configuration.FFTdim].data.i = prevStride * FFTPlan->actualFFTSizePerAxis[axis_id][app->configuration.FFTdim-1];

		axisStride[app->configuration.FFTdim+1].data.i = axisStride[app->configuration.FFTdim].data.i * app->configuration.coordinateFeatures;
	}
	if ((!inverse) && (axis_id == 0) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (reverseBluesteinMultiUpload == 0) && (axis->specializationConstants.performR2C || FFTPlan->multiUploadR2C) && (!(app->configuration.isInputFormatted))) {
        for (int i = 1; i < (app->configuration.FFTdim+2); i++){
            axisStride[i].data.i *= 2;
        }
	}
	if ((FFTPlan->multiUploadR2C) && (!inverse) && (axis_id == 0) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (reverseBluesteinMultiUpload == 0)) {
		for (uint64_t i = 1; i < (app->configuration.FFTdim+2); i++) {
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
        for (int i = 1; i < app->configuration.FFTdim; i++){
            axisStride[i].type = 31;
            axisStride[i].data.i = usedStride[i-1];
        }
    }else{
        int locStrideOrder = 2;
        for (int i = 1; i < app->configuration.FFTdim; i++){
            if(i==axis_id){
                axisStride[1].type = 31;
                axisStride[1].data.i = usedStride[i-1];
            }else{
                axisStride[locStrideOrder].type = 31;
                axisStride[locStrideOrder].data.i = usedStride[i-1];
                locStrideOrder++;
            }
        }
        
    }

    axisStride[app->configuration.FFTdim].type = 31;
    axisStride[app->configuration.FFTdim].data.i = usedStride[app->configuration.FFTdim-1];

    axisStride[app->configuration.FFTdim+1].type = 31;
    axisStride[app->configuration.FFTdim+1].data.i = axisStride[app->configuration.FFTdim].data.i * app->configuration.coordinateFeatures;
	if (app->useBluesteinFFT[axis_id] && (FFTPlan->numAxisUploads[axis_id] > 1) && (!((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (reverseBluesteinMultiUpload == 1)))) {
		axisStride[0].data.i = 1;
        int64_t prevStride = axisStride[0].data.i;
        
        if (axis_id == 0) {
            for (int i = 1; i < app->configuration.FFTdim; i++){
                axisStride[i].data.i = prevStride * FFTPlan->actualFFTSizePerAxis[axis_id][i-1];
                prevStride = axisStride[i].data.i;
            }
        }else{
            int locStrideOrder = 2;
            for (int i = 1; i < app->configuration.FFTdim; i++){
                if(i==axis_id){
                    axisStride[1].data.i = prevStride * FFTPlan->actualFFTSizePerAxis[axis_id][i-1];
                    prevStride = axisStride[1].data.i;
                }else{
                    axisStride[locStrideOrder].data.i = prevStride * FFTPlan->actualFFTSizePerAxis[axis_id][i-1];
                    prevStride = axisStride[locStrideOrder].data.i;
                    locStrideOrder++;
                }
            }
            
        }

        axisStride[app->configuration.FFTdim].data.i = prevStride * FFTPlan->actualFFTSizePerAxis[axis_id][app->configuration.FFTdim-1];

        axisStride[app->configuration.FFTdim+1].data.i = axisStride[app->configuration.FFTdim].data.i * app->configuration.coordinateFeatures;
	}
	if ((inverse) && (axis_id == 0) && (((!app->useBluesteinFFT[axis_id]) && (axis_upload_id == 0)) || ((app->useBluesteinFFT[axis_id]) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && ((reverseBluesteinMultiUpload == 1) || (FFTPlan->numAxisUploads[axis_id] == 1)))) && (axis->specializationConstants.performR2C || FFTPlan->multiUploadR2C) && (!((app->configuration.isInputFormatted) && (app->configuration.inverseReturnToInputBuffer))) && (!app->configuration.isOutputFormatted)) {
        for (int i = 1; i < (app->configuration.FFTdim+2); i++){
            axisStride[i].data.i *= 2;
        }
	}
	if ((FFTPlan->multiUploadR2C) && (inverse) && (axis_id == 0) && (((!app->useBluesteinFFT[axis_id]) && (axis_upload_id == 0)) || ((app->useBluesteinFFT[axis_id]) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && ((reverseBluesteinMultiUpload == 1) || (FFTPlan->numAxisUploads[axis_id] == 1))))) {
		for (uint64_t i = 1; i < (app->configuration.FFTdim+2); i++) {
			axisStride[i].data.i /= 2;
		}
	}

	resFFT = VkFFTConfigureDescriptors(app, FFTPlan, axis, axis_id, axis_upload_id, inverse);
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
	else {
		axis->specializationConstants.inputOffset.type = 31;
		axis->specializationConstants.inputOffset.data.i = app->configuration.inputBufferOffset;
		axis->specializationConstants.outputOffset.type = 31;
		axis->specializationConstants.outputOffset.data.i = app->configuration.outputBufferOffset;
		axis->specializationConstants.kernelOffset.type = 31;
		axis->specializationConstants.kernelOffset.data.i = app->configuration.kernelOffset;
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
		VkFFTSplitAxisBlock(app, FFTPlan, axis, axis_id, axis_upload_id, allowedSharedMemory, allowedSharedMemoryPow2);
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
		axis->specializationConstants.numSubgroups = (int)ceil(axis->axisBlock[0] * axis->axisBlock[1] * axis->axisBlock[2] / (double)app->configuration.warpSize);
		//specializationInfo.pData = &axis->specializationConstants;
		//uint64_t registerBoost = (FFTPlan->numAxisUploads[axis_id] > 1) ? app->configuration.registerBoost4Step : app->configuration.registerBoost;

		axis->specializationConstants.numCoordinates = (app->configuration.matrixConvolution > 1) ? 1 : (int)app->configuration.coordinateFeatures;
		axis->specializationConstants.matrixConvolution = (int)app->configuration.matrixConvolution;
		axis->specializationConstants.coordinate.type = 31;
		axis->specializationConstants.coordinate.data.i = 0;
		axis->specializationConstants.batchID.type = 31;
		axis->specializationConstants.batchID.data.i = 0;
		axis->specializationConstants.numKernels.type = 31;
		axis->specializationConstants.numKernels.data.i = (int)app->configuration.numberKernels;
		axis->specializationConstants.sharedMemSize = (int)app->configuration.sharedMemorySize;
		axis->specializationConstants.sharedMemSizePow2 = (int)app->configuration.sharedMemorySizePow2;
		axis->specializationConstants.normalize = (reverseBluesteinMultiUpload) ? 1 : (int)app->configuration.normalize;
        for (uint64_t i = 0; i < VKFFT_MAX_FFT_DIMENSIONS; i++) {
            axis->specializationConstants.size[i].type = 31;
            axis->specializationConstants.size[i].data.i = (int64_t)FFTPlan->actualFFTSizePerAxis[axis_id][i];
        }
		
		for (uint64_t i = 0; i < VKFFT_MAX_FFT_DIMENSIONS; i++) {
			axis->specializationConstants.frequencyZeropadding = (int)app->configuration.frequencyZeroPadding;
			axis->specializationConstants.performZeropaddingFull[i] = (int)app->configuration.performZeropadding[i]; // don't read if input is zeropadded (0 - off, 1 - on)
			axis->specializationConstants.fft_zeropad_left_full[i].type = 31;
			axis->specializationConstants.fft_zeropad_left_full[i].data.i = (int64_t)app->configuration.fft_zeropad_left[i];
			axis->specializationConstants.fft_zeropad_right_full[i].type = 31;
			axis->specializationConstants.fft_zeropad_right_full[i].data.i = (int64_t)app->configuration.fft_zeropad_right[i];
		}
		if (axis->specializationConstants.useBluesteinFFT && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && ((reverseBluesteinMultiUpload == 0) || (FFTPlan->numAxisUploads[axis_id] == 1))) {
			axis->specializationConstants.zeropadBluestein[0] = 1;
			axis->specializationConstants.fft_zeropad_Bluestein_left_read[axis_id].type = 31;
			axis->specializationConstants.fft_zeropad_Bluestein_left_read[axis_id].data.i = (int64_t)app->configuration.size[axis_id];
			if ((FFTPlan->multiUploadR2C) && (axis_id == 0)) axis->specializationConstants.fft_zeropad_Bluestein_left_read[axis_id].data.i /= 2;
			if (app->configuration.performDCT == 1) axis->specializationConstants.fft_zeropad_Bluestein_left_read[axis_id].data.i = 2 * axis->specializationConstants.fft_zeropad_Bluestein_left_read[axis_id].data.i - 2;
			if ((app->configuration.performDCT == 4) && (app->configuration.size[axis_id] % 2 == 0)) axis->specializationConstants.fft_zeropad_Bluestein_left_read[axis_id].data.i /= 2;
			axis->specializationConstants.fft_zeropad_Bluestein_right_read[axis_id].type = 31;
			axis->specializationConstants.fft_zeropad_Bluestein_right_read[axis_id].data.i = (int64_t)FFTPlan->actualFFTSizePerAxis[axis_id][axis_id];
		}
		if (axis->specializationConstants.useBluesteinFFT && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && ((reverseBluesteinMultiUpload == 1) || (FFTPlan->numAxisUploads[axis_id] == 1))) {
			axis->specializationConstants.zeropadBluestein[1] = 1;
			axis->specializationConstants.fft_zeropad_Bluestein_left_write[axis_id].type = 31;
			axis->specializationConstants.fft_zeropad_Bluestein_left_write[axis_id].data.i = (int64_t)app->configuration.size[axis_id];
			if ((FFTPlan->multiUploadR2C) && (axis_id == 0)) axis->specializationConstants.fft_zeropad_Bluestein_left_write[axis_id].data.i /= 2;
			if (app->configuration.performDCT == 1) axis->specializationConstants.fft_zeropad_Bluestein_left_write[axis_id].data.i = 2 * axis->specializationConstants.fft_zeropad_Bluestein_left_write[axis_id].data.i - 2;
			if ((app->configuration.performDCT == 4) && (app->configuration.size[axis_id] % 2 == 0)) axis->specializationConstants.fft_zeropad_Bluestein_left_write[axis_id].data.i /= 2;
			axis->specializationConstants.fft_zeropad_Bluestein_right_write[axis_id].type = 31;
			axis->specializationConstants.fft_zeropad_Bluestein_right_write[axis_id].data.i = (int64_t)FFTPlan->actualFFTSizePerAxis[axis_id][axis_id];
		}
		uint64_t zeropad_r2c_multiupload_scale = ((axis_id == 0) && (FFTPlan->multiUploadR2C)) ? 2 : 1;
		if ((inverse)) {
			if ((app->configuration.frequencyZeroPadding) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (reverseBluesteinMultiUpload != 1)) {
				axis->specializationConstants.zeropad[0] = (int)app->configuration.performZeropadding[axis_id];
				axis->specializationConstants.fft_zeropad_left_read[axis_id].type = 31;
				axis->specializationConstants.fft_zeropad_left_read[axis_id].data.i = (int64_t)app->configuration.fft_zeropad_left[axis_id] / zeropad_r2c_multiupload_scale;
				axis->specializationConstants.fft_zeropad_right_read[axis_id].type = 31;
				axis->specializationConstants.fft_zeropad_right_read[axis_id].data.i = (int64_t)app->configuration.fft_zeropad_right[axis_id] / zeropad_r2c_multiupload_scale;
			}
			else
				axis->specializationConstants.zeropad[0] = 0;
			if ((!app->configuration.frequencyZeroPadding) && (((axis_upload_id == 0) && (!((axis->specializationConstants.useBluesteinFFT) || (app->configuration.performConvolution)))) || ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && ((((reverseBluesteinMultiUpload == 1) || (FFTPlan->numAxisUploads[axis_id] == 1)) || (app->configuration.performConvolution)))))) {
				axis->specializationConstants.zeropad[1] = (int)app->configuration.performZeropadding[axis_id];
				axis->specializationConstants.fft_zeropad_left_write[axis_id].type = 31;
				axis->specializationConstants.fft_zeropad_left_write[axis_id].data.i = (int64_t)app->configuration.fft_zeropad_left[axis_id] / zeropad_r2c_multiupload_scale;
				axis->specializationConstants.fft_zeropad_right_write[axis_id].type = 31;
				axis->specializationConstants.fft_zeropad_right_write[axis_id].data.i = (int64_t)app->configuration.fft_zeropad_right[axis_id] / zeropad_r2c_multiupload_scale;
			}
			else
				axis->specializationConstants.zeropad[1] = 0;
		}
		else {
			if ((!app->configuration.frequencyZeroPadding) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (reverseBluesteinMultiUpload != 1)) {
				axis->specializationConstants.zeropad[0] = (int)app->configuration.performZeropadding[axis_id];
				axis->specializationConstants.fft_zeropad_left_read[axis_id].type = 31;
				axis->specializationConstants.fft_zeropad_left_read[axis_id].data.i = (int64_t)app->configuration.fft_zeropad_left[axis_id] / zeropad_r2c_multiupload_scale;
				axis->specializationConstants.fft_zeropad_right_read[axis_id].type = 31;
				axis->specializationConstants.fft_zeropad_right_read[axis_id].data.i = (int64_t)app->configuration.fft_zeropad_right[axis_id] / zeropad_r2c_multiupload_scale;
			}
			else
				axis->specializationConstants.zeropad[0] = 0;
			if (((app->configuration.frequencyZeroPadding) && (((axis_upload_id == 0) && (!axis->specializationConstants.useBluesteinFFT)) || ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (axis->specializationConstants.useBluesteinFFT && ((reverseBluesteinMultiUpload == 1) || (FFTPlan->numAxisUploads[axis_id] == 1)))))) || (((!app->configuration.frequencyZeroPadding) && (app->configuration.FFTdim - 1 == axis_id) && (axis_upload_id == 0) && (FFTPlan->numAxisUploads[axis_id] == 1) && (app->configuration.performConvolution)))) {
				axis->specializationConstants.zeropad[1] = (int)app->configuration.performZeropadding[axis_id];
				axis->specializationConstants.fft_zeropad_left_write[axis_id].type = 31;
				axis->specializationConstants.fft_zeropad_left_write[axis_id].data.i = (int64_t)app->configuration.fft_zeropad_left[axis_id] / zeropad_r2c_multiupload_scale;
				axis->specializationConstants.fft_zeropad_right_write[axis_id].type = 31;
				axis->specializationConstants.fft_zeropad_right_write[axis_id].data.i = (int64_t)app->configuration.fft_zeropad_right[axis_id] / zeropad_r2c_multiupload_scale;
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


        uint64_t tempSize[3];

		if (axis_id == 0) {
			if (axis_upload_id == 0)
				tempSize[0] = FFTPlan->actualFFTSizePerAxis[axis_id][0] / axis->specializationConstants.fftDim.data.i / axis->axisBlock[1];
			else
				tempSize[0] = FFTPlan->actualFFTSizePerAxis[axis_id][0] / axis->specializationConstants.fftDim.data.i / axis->axisBlock[0];
            tempSize[1] = FFTPlan->actualFFTSizePerAxis[axis_id][1];
			if ((FFTPlan->actualPerformR2CPerAxis[axis_id] == 1) && (axis->specializationConstants.mergeSequencesR2C)) tempSize[1] = (uint64_t)ceil(tempSize[1] / 2.0);
            
			//if (app->configuration.performZeropadding[1]) tempSize[1] = (uint64_t)ceil(tempSize[1] / 2.0);
			//if (app->configuration.performZeropadding[2]) tempSize[2] = (uint64_t)ceil(tempSize[2] / 2.0);
        }else{
			tempSize[0] = (uint64_t)ceil(FFTPlan->actualFFTSizePerAxis[axis_id][0] / (double)axis->axisBlock[0] * FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (double)axis->specializationConstants.fftDim.data.i);
			tempSize[1] = 1;
			
			//if (app->configuration.actualPerformR2C == 1) tempSize[0] = (uint64_t)ceil(tempSize[0] / 2.0);
			//if (app->configuration.performZeropadding[2]) tempSize[2] = (uint64_t)ceil(tempSize[2] / 2.0);
		}
        tempSize[2] = 1;
        for (uint64_t i = 1; i < app->configuration.FFTdim; i++) {
            if (i!=axis_id)
                tempSize[2] *= FFTPlan->actualFFTSizePerAxis[axis_id][i];
        }
        tempSize[2] *= app->configuration.numberKernels * app->configuration.numberBatches;
        if (!(axis->specializationConstants.convolutionStep && (app->configuration.matrixConvolution > 1))) tempSize[2] *= app->configuration.coordinateFeatures;
        
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
			axis->specializationConstants.pushConstantsStructSize = (int)axis->pushConstants.structSize;
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
		resFFT = shaderGen_FFT(&axis->specializationConstants, (int)type);
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
