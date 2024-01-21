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
static inline VkFFTResult VkFFTPlanAxis(VkFFTApplication* app, VkFFTPlan* FFTPlan, pfUINT axis_id, pfUINT axis_upload_id, pfUINT inverse, pfUINT reverseBluesteinMultiUpload) {
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
	axis->specializationConstants.numBatches.data.i = (pfINT)app->configuration.numberBatches;
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

	axis->specializationConstants.double_PI = pfFPinit("3.14159265358979323846264338327950288419716939937510");
	axis->specializationConstants.storeSharedComplexComponentsSeparately = 0;
	
	if (app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) {
		axis->specializationConstants.precision = 3;
		axis->specializationConstants.complexSize = 32;
		axis->specializationConstants.storeSharedComplexComponentsSeparately = 1;
	}
	else {
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
	}
	
	pfUINT allowedSharedMemory = app->configuration.sharedMemorySize;
	pfUINT allowedSharedMemoryPow2 = app->configuration.sharedMemorySizePow2;

	if (axis->specializationConstants.useRaderMult) {
		allowedSharedMemory -= (axis->specializationConstants.useRaderMult - 1) * axis->specializationConstants.complexSize;
		allowedSharedMemoryPow2 -= (axis->specializationConstants.useRaderMult - 1) * axis->specializationConstants.complexSize;
	}

	pfUINT maxSequenceLengthSharedMemory = allowedSharedMemory / axis->specializationConstants.complexSize;
	pfUINT maxSequenceLengthSharedMemoryPow2 = allowedSharedMemoryPow2 / axis->specializationConstants.complexSize;
	pfUINT maxSingleSizeStrided = (app->configuration.coalescedMemory > axis->specializationConstants.complexSize) ? allowedSharedMemory / (app->configuration.coalescedMemory) : allowedSharedMemory / axis->specializationConstants.complexSize;
	pfUINT maxSingleSizeStridedPow2 = (app->configuration.coalescedMemory > axis->specializationConstants.complexSize) ? allowedSharedMemoryPow2 / (app->configuration.coalescedMemory) : allowedSharedMemoryPow2 / axis->specializationConstants.complexSize;

	axis->specializationConstants.stageStartSize.type = 31;
	axis->specializationConstants.stageStartSize.data.i = 1;
	for (pfUINT i = 0; i < axis_upload_id; i++)
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

	if ((app->configuration.performDCT == 3) || (app->configuration.performDST == 3)) {
		axis->specializationConstants.actualInverse = (int)inverse;
		axis->specializationConstants.inverse = (int)!inverse;
	}
	else {
		if ((app->configuration.performDCT == 4) || (app->configuration.performDST == 4)) {
			axis->specializationConstants.actualInverse = (int)inverse;
			axis->specializationConstants.inverse = 1;
		}
		else if ((app->configuration.performDCT == 1) || (app->configuration.performDST == 1)) {
			axis->specializationConstants.actualInverse = (int)inverse;
			axis->specializationConstants.inverse = 0;
		}
		else{
			axis->specializationConstants.actualInverse = (int)inverse;
			axis->specializationConstants.inverse = (int)inverse;
		}
	}
	if (app->useBluesteinFFT[axis_id]) {
		axis->specializationConstants.actualInverse = (int)inverse;
		axis->specializationConstants.inverse = (int)reverseBluesteinMultiUpload;
		if ((app->configuration.performDCT == 3) || (app->configuration.performDST == 3)) {
			axis->specializationConstants.inverseBluestein = (int)!inverse;
		}
		else {
			if ((app->configuration.performDCT == 4) || (app->configuration.performDST == 4)) {
				axis->specializationConstants.inverseBluestein = 1;
			}
			else if ((app->configuration.performDCT == 1) || (app->configuration.performDST == 1)) {
				axis->specializationConstants.inverseBluestein = 0;
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
		maxSequenceLengthSharedMemoryPow2 = (pfUINT)pow(2, (pfUINT)log2(maxSequenceLengthSharedMemory));
	}
	else {
		maxSingleSizeStrided *= axis->specializationConstants.registerBoost;
		maxSingleSizeStridedPow2 = (pfUINT)pow(2, (pfUINT)log2(maxSingleSizeStrided));
	}
	axis->specializationConstants.maxSingleSizeStrided.type = 31;
	axis->specializationConstants.maxSingleSizeStrided.data.i = maxSingleSizeStrided;

	axis->specializationConstants.performR2C = (int)FFTPlan->actualPerformR2CPerAxis[axis_id];
	axis->specializationConstants.performR2CmultiUpload = ((axis->specializationConstants.performR2C || (FFTPlan->bigSequenceEvenR2C)) && (axis->specializationConstants.numAxisUploads > 1)) ? 1 : 0;
	
	if (app->configuration.performDCT > 0)
		axis->specializationConstants.performDCT = (int)app->configuration.performDCT;
	if (app->configuration.performDST > 0)
		axis->specializationConstants.performDST = (int)app->configuration.performDST;

	axis->specializationConstants.performR2RmultiUpload = (((axis->specializationConstants.performDCT>0) || (axis->specializationConstants.performDST>0)) && (axis->specializationConstants.numAxisUploads > 1)) ? 1 : 0;
	if (axis->specializationConstants.performR2C || axis->specializationConstants.performDCT || axis->specializationConstants.performDST) axis->specializationConstants.forceCallbackVersionRealTransforms = app->configuration.forceCallbackVersionRealTransforms;
	if (((axis->specializationConstants.performR2C && (!FFTPlan->bigSequenceEvenR2C)) || axis->specializationConstants.performDCT || axis->specializationConstants.performDST) && (axis->specializationConstants.numAxisUploads > 1)) axis->specializationConstants.forceCallbackVersionRealTransforms = 1;
	//if ((axis->specializationConstants.performR2CmultiUpload) && (app->configuration.size[0] % 2 != 0)) return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH_R2C;
	//pfUINT passID = FFTPlan->numAxisUploads[axis_id] - 1 - axis_upload_id;
	axis->specializationConstants.fft_dim_full.type = 31;
	axis->specializationConstants.fft_dim_full.data.i = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id];

	pfUINT additionalR2Cshared = 0;
	if ((axis->specializationConstants.performR2C || ((axis->specializationConstants.performDCT == 2) || (axis->specializationConstants.performDST == 2) || (axis->specializationConstants.performDCT == 3) || (axis->specializationConstants.performDST == 3) || (((axis->specializationConstants.performDCT == 4) || (axis->specializationConstants.performDST == 4)) && ((axis->specializationConstants.fft_dim_full.data.i % 2) != 0)))) && (axis->specializationConstants.axis_id == 0) && (!axis->specializationConstants.performR2CmultiUpload) && (!axis->specializationConstants.performR2RmultiUpload)) {
		additionalR2Cshared = ((axis->specializationConstants.fft_dim_full.data.i % 2) == 0) ? 2 : 1;
		if ((axis->specializationConstants.performDCT == 2) || (axis->specializationConstants.performDST == 2) || (axis->specializationConstants.performDCT == 3) || (axis->specializationConstants.performDST == 3) || (((axis->specializationConstants.performDCT == 4) || (axis->specializationConstants.performDST == 4)) && ((axis->specializationConstants.fft_dim_full.data.i % 2) != 0))) additionalR2Cshared = 1;
	}
	axis->specializationConstants.mergeSequencesR2C = ((!axis->specializationConstants.performR2CmultiUpload) && (!axis->specializationConstants.performR2RmultiUpload) && ((axis->specializationConstants.fft_dim_full.data.i + additionalR2Cshared) <= maxSequenceLengthSharedMemory) && (FFTPlan->actualFFTSizePerAxis[axis_id][1] > 1) && ((FFTPlan->actualPerformR2CPerAxis[axis_id]) || ((((axis->specializationConstants.performDCT == 3) || (axis->specializationConstants.performDST == 3)) || ((axis->specializationConstants.performDCT == 2) || (axis->specializationConstants.performDST == 2)) || ((axis->specializationConstants.performDCT == 1) || (axis->specializationConstants.performDST == 1)) || (((axis->specializationConstants.performDCT == 4) || (axis->specializationConstants.performDST == 4)) && ((app->configuration.size[axis_id] % 2) != 0))) && (axis_id == 0)))) ? (1 - (int)app->configuration.disableMergeSequencesR2C) : 0;
	
	if ((FFTPlan->numAxisUploads[axis_id] > 1) && (axis->specializationConstants.reorderFourStep || app->useBluesteinFFT[axis_id]) && (!app->configuration.userTempBuffer) && (app->configuration.allocateTempBuffer == 0)) {
		app->configuration.allocateTempBuffer = 1;
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
	pfUINT* usedStride = app->configuration.bufferStride;
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
	if ((FFTPlan->numAxisUploads[axis_id] > 1) && ((app->useBluesteinFFT[axis_id] && (!((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (reverseBluesteinMultiUpload == 0))))) || ((!app->useBluesteinFFT[axis_id]) && (!((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1))))) {
		axisStride[0].data.i = 1;
        pfINT prevStride = axisStride[0].data.i;
        
		if (axis_id == 0) {
            for (int i = 1; i < app->configuration.FFTdim; i++){
				if (FFTPlan->bigSequenceEvenR2C && (i == 1))
					axisStride[i].data.i = prevStride * (FFTPlan->actualFFTSizePerAxis[axis_id][i-1]+1);
				else
					axisStride[i].data.i = prevStride * FFTPlan->actualFFTSizePerAxis[axis_id][i-1];
                prevStride = axisStride[i].data.i;
            }
			if (FFTPlan->bigSequenceEvenR2C && (app->configuration.FFTdim == 1)) {
				axisStride[app->configuration.FFTdim].data.i = prevStride * (FFTPlan->actualFFTSizePerAxis[axis_id][app->configuration.FFTdim-1]+1);

				axisStride[app->configuration.FFTdim+1].data.i = axisStride[app->configuration.FFTdim].data.i * app->configuration.coordinateFeatures;
			}
			else {
				axisStride[app->configuration.FFTdim].data.i = prevStride * FFTPlan->actualFFTSizePerAxis[axis_id][app->configuration.FFTdim-1];

				axisStride[app->configuration.FFTdim+1].data.i = axisStride[app->configuration.FFTdim].data.i * app->configuration.coordinateFeatures;
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
            axisStride[app->configuration.FFTdim].data.i = prevStride * FFTPlan->actualFFTSizePerAxis[axis_id][app->configuration.FFTdim-1];

			axisStride[app->configuration.FFTdim+1].data.i = axisStride[app->configuration.FFTdim].data.i * app->configuration.coordinateFeatures;
        }
	}
	if ((!inverse) && (axis_id == 0) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (reverseBluesteinMultiUpload == 0) && (axis->specializationConstants.performR2C || FFTPlan->bigSequenceEvenR2C) && (!(app->configuration.isInputFormatted))) {
        for (int i = 1; i < (app->configuration.FFTdim+2); i++){
            axisStride[i].data.i *= 2;
        }
	}
	if ((FFTPlan->bigSequenceEvenR2C) && (!inverse) && (axis_id == 0) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (reverseBluesteinMultiUpload == 0)) {
		for (pfUINT i = 1; i < (app->configuration.FFTdim+2); i++) {
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
	if ((FFTPlan->numAxisUploads[axis_id] > 1) && ((app->useBluesteinFFT[axis_id] && (!((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (reverseBluesteinMultiUpload == 1)))) || ((!app->useBluesteinFFT[axis_id]) && (axis_upload_id != 0)))) {
		axisStride[0].data.i = 1;
        pfINT prevStride = axisStride[0].data.i;
        
        if (axis_id == 0) {
            for (int i = 1; i < app->configuration.FFTdim; i++){
                if (FFTPlan->bigSequenceEvenR2C && (i == 1))
					axisStride[i].data.i = prevStride * (FFTPlan->actualFFTSizePerAxis[axis_id][i-1]+1);
				else
					axisStride[i].data.i = prevStride * FFTPlan->actualFFTSizePerAxis[axis_id][i-1];
                prevStride = axisStride[i].data.i;
            }
			
			if (FFTPlan->bigSequenceEvenR2C && (app->configuration.FFTdim == 1)) {
				axisStride[app->configuration.FFTdim].data.i = prevStride * (FFTPlan->actualFFTSizePerAxis[axis_id][app->configuration.FFTdim - 1]+1);

				axisStride[app->configuration.FFTdim + 1].data.i = axisStride[app->configuration.FFTdim].data.i * app->configuration.coordinateFeatures;
			}
			else {
				axisStride[app->configuration.FFTdim].data.i = prevStride * FFTPlan->actualFFTSizePerAxis[axis_id][app->configuration.FFTdim - 1];

				axisStride[app->configuration.FFTdim + 1].data.i = axisStride[app->configuration.FFTdim].data.i * app->configuration.coordinateFeatures;
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
			axisStride[app->configuration.FFTdim].data.i = prevStride * FFTPlan->actualFFTSizePerAxis[axis_id][app->configuration.FFTdim-1];

			axisStride[app->configuration.FFTdim+1].data.i = axisStride[app->configuration.FFTdim].data.i * app->configuration.coordinateFeatures;
        }

	}
	if ((inverse) && (axis_id == 0) && (((!app->useBluesteinFFT[axis_id]) && (axis_upload_id == 0)) || ((app->useBluesteinFFT[axis_id]) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && ((reverseBluesteinMultiUpload == 1) || (FFTPlan->numAxisUploads[axis_id] == 1)))) && (axis->specializationConstants.performR2C || FFTPlan->bigSequenceEvenR2C) && (!((app->configuration.isInputFormatted) && (app->configuration.inverseReturnToInputBuffer))) && (!app->configuration.isOutputFormatted)) {
        for (int i = 1; i < (app->configuration.FFTdim+2); i++){
            axisStride[i].data.i *= 2;
        }
	}
	if ((FFTPlan->bigSequenceEvenR2C) && (inverse) && (axis_id == 0) && (((!app->useBluesteinFFT[axis_id]) && (axis_upload_id == 0)) || ((app->useBluesteinFFT[axis_id]) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && ((reverseBluesteinMultiUpload == 1) || (FFTPlan->numAxisUploads[axis_id] == 1))))) {
		for (pfUINT i = 1; i < (app->configuration.FFTdim+2); i++) {
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
		for (pfUINT i = 0; i < 36; i++) {
			specializationMapEntries[i].constantID = i + 1;
			specializationMapEntries[i].size = sizeof(pfUINT);
			specializationMapEntries[i].offset = i * sizeof(pfUINT);
		}
		VkSpecializationInfo specializationInfo = { 0 };
		specializationInfo.dataSize = 36 * sizeof(pfUINT);
		specializationInfo.mapEntryCount = 36;
		specializationInfo.pMapEntries = specializationMapEntries;*/
		axis->specializationConstants.localSize[0].type = 31;
		axis->specializationConstants.localSize[1].type = 31;
		axis->specializationConstants.localSize[2].type = 31;
		axis->specializationConstants.localSize[0].data.i = axis->axisBlock[0];
		axis->specializationConstants.localSize[1].data.i = axis->axisBlock[1];
		axis->specializationConstants.localSize[2].data.i = axis->axisBlock[2];
		axis->specializationConstants.numSubgroups = (int)pfceil(axis->axisBlock[0] * axis->axisBlock[1] * axis->axisBlock[2] / (double)app->configuration.warpSize);
		//specializationInfo.pData = &axis->specializationConstants;
		//pfUINT registerBoost = (FFTPlan->numAxisUploads[axis_id] > 1) ? app->configuration.registerBoost4Step : app->configuration.registerBoost;

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
        for (pfUINT i = 0; i < VKFFT_MAX_FFT_DIMENSIONS; i++) {
            axis->specializationConstants.size[i].type = 31;
            axis->specializationConstants.size[i].data.i = (pfINT)FFTPlan->actualFFTSizePerAxis[axis_id][i];
        }
		
		for (pfUINT i = 0; i < VKFFT_MAX_FFT_DIMENSIONS; i++) {
			axis->specializationConstants.frequencyZeropadding = (int)app->configuration.frequencyZeroPadding;
			axis->specializationConstants.performZeropaddingFull[i] = (int)app->configuration.performZeropadding[i]; // don't read if input is zeropadded (0 - off, 1 - on)
			axis->specializationConstants.fft_zeropad_left_full[i].type = 31;
			axis->specializationConstants.fft_zeropad_left_full[i].data.i = (pfINT)app->configuration.fft_zeropad_left[i];
			axis->specializationConstants.fft_zeropad_right_full[i].type = 31;
			axis->specializationConstants.fft_zeropad_right_full[i].data.i = (pfINT)app->configuration.fft_zeropad_right[i];
		}
		if (axis->specializationConstants.useBluesteinFFT && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && ((reverseBluesteinMultiUpload == 0) || (FFTPlan->numAxisUploads[axis_id] == 1))) {
			axis->specializationConstants.zeropadBluestein[0] = 1;
			axis->specializationConstants.fft_zeropad_Bluestein_left_read[axis_id].type = 31;
			axis->specializationConstants.fft_zeropad_Bluestein_left_read[axis_id].data.i = (pfINT)app->configuration.size[axis_id];
			if ((FFTPlan->bigSequenceEvenR2C) && (axis_id == 0)) axis->specializationConstants.fft_zeropad_Bluestein_left_read[axis_id].data.i /= 2;
			//if (axis->specializationConstants.performDCT == 1) axis->specializationConstants.fft_zeropad_Bluestein_left_read[axis_id].data.i = 2 * axis->specializationConstants.fft_zeropad_Bluestein_left_read[axis_id].data.i - 2;
			//if (axis->specializationConstants.performDST == 1) axis->specializationConstants.fft_zeropad_Bluestein_left_read[axis_id].data.i = 2 * axis->specializationConstants.fft_zeropad_Bluestein_left_read[axis_id].data.i + 2;
			if (((axis->specializationConstants.performDCT == 4) || (axis->specializationConstants.performDST == 4)) && (app->configuration.size[axis_id] % 2 == 0) && (!axis->specializationConstants.forceCallbackVersionRealTransforms)) axis->specializationConstants.fft_zeropad_Bluestein_left_read[axis_id].data.i /= 2;
			axis->specializationConstants.fft_zeropad_Bluestein_right_read[axis_id].type = 31;
			axis->specializationConstants.fft_zeropad_Bluestein_right_read[axis_id].data.i = (pfINT)FFTPlan->actualFFTSizePerAxis[axis_id][axis_id];
		}
		if (axis->specializationConstants.useBluesteinFFT && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && ((reverseBluesteinMultiUpload == 1) || (FFTPlan->numAxisUploads[axis_id] == 1))) {
			axis->specializationConstants.zeropadBluestein[1] = 1;
			axis->specializationConstants.fft_zeropad_Bluestein_left_write[axis_id].type = 31;
			axis->specializationConstants.fft_zeropad_Bluestein_left_write[axis_id].data.i = (pfINT)app->configuration.size[axis_id];
			if ((FFTPlan->bigSequenceEvenR2C) && (axis_id == 0)) axis->specializationConstants.fft_zeropad_Bluestein_left_write[axis_id].data.i /= 2;
			//if (axis->specializationConstants.performDCT == 1) axis->specializationConstants.fft_zeropad_Bluestein_left_write[axis_id].data.i = 2 * axis->specializationConstants.fft_zeropad_Bluestein_left_write[axis_id].data.i - 2;
			//if (axis->specializationConstants.performDST == 1) axis->specializationConstants.fft_zeropad_Bluestein_left_write[axis_id].data.i = 2 * axis->specializationConstants.fft_zeropad_Bluestein_left_write[axis_id].data.i + 2;
			if (((axis->specializationConstants.performDCT == 4) || (axis->specializationConstants.performDST == 4)) && (app->configuration.size[axis_id] % 2 == 0) && (!axis->specializationConstants.forceCallbackVersionRealTransforms)) axis->specializationConstants.fft_zeropad_Bluestein_left_write[axis_id].data.i /= 2;
			axis->specializationConstants.fft_zeropad_Bluestein_right_write[axis_id].type = 31;
			axis->specializationConstants.fft_zeropad_Bluestein_right_write[axis_id].data.i = (pfINT)FFTPlan->actualFFTSizePerAxis[axis_id][axis_id];
		}
		pfUINT zeropad_even_r2c_multiupload_scale = ((axis_id == 0) && (FFTPlan->bigSequenceEvenR2C)) ? 2 : 1;
		if ((inverse)) {
			if ((app->configuration.frequencyZeroPadding) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (reverseBluesteinMultiUpload != 1)) {
				axis->specializationConstants.zeropad[0] = (int)app->configuration.performZeropadding[axis_id];
				axis->specializationConstants.fft_zeropad_left_read[axis_id].type = 31;
				axis->specializationConstants.fft_zeropad_left_read[axis_id].data.i = (pfINT)app->configuration.fft_zeropad_left[axis_id] / zeropad_even_r2c_multiupload_scale;
				axis->specializationConstants.fft_zeropad_right_read[axis_id].type = 31;
				axis->specializationConstants.fft_zeropad_right_read[axis_id].data.i = (pfINT)app->configuration.fft_zeropad_right[axis_id] / zeropad_even_r2c_multiupload_scale;
			}
			else
				axis->specializationConstants.zeropad[0] = 0;
			if ((!app->configuration.frequencyZeroPadding) && (((axis_upload_id == 0) && (!((axis->specializationConstants.useBluesteinFFT) || (app->configuration.performConvolution)))) || ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && ((((reverseBluesteinMultiUpload == 1) || (FFTPlan->numAxisUploads[axis_id] == 1)) || (app->configuration.performConvolution)))))) {
				axis->specializationConstants.zeropad[1] = (int)app->configuration.performZeropadding[axis_id];
				axis->specializationConstants.fft_zeropad_left_write[axis_id].type = 31;
				axis->specializationConstants.fft_zeropad_left_write[axis_id].data.i = (pfINT)app->configuration.fft_zeropad_left[axis_id] / zeropad_even_r2c_multiupload_scale;
				axis->specializationConstants.fft_zeropad_right_write[axis_id].type = 31;
				axis->specializationConstants.fft_zeropad_right_write[axis_id].data.i = (pfINT)app->configuration.fft_zeropad_right[axis_id] / zeropad_even_r2c_multiupload_scale;
			}
			else
				axis->specializationConstants.zeropad[1] = 0;
		}
		else {
			if ((!app->configuration.frequencyZeroPadding) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (reverseBluesteinMultiUpload != 1)) {
				axis->specializationConstants.zeropad[0] = (int)app->configuration.performZeropadding[axis_id];
				axis->specializationConstants.fft_zeropad_left_read[axis_id].type = 31;
				axis->specializationConstants.fft_zeropad_left_read[axis_id].data.i = (pfINT)app->configuration.fft_zeropad_left[axis_id] / zeropad_even_r2c_multiupload_scale;
				axis->specializationConstants.fft_zeropad_right_read[axis_id].type = 31;
				axis->specializationConstants.fft_zeropad_right_read[axis_id].data.i = (pfINT)app->configuration.fft_zeropad_right[axis_id] / zeropad_even_r2c_multiupload_scale;
			}
			else
				axis->specializationConstants.zeropad[0] = 0;
			if (((app->configuration.frequencyZeroPadding) && (((axis_upload_id == 0) && (!axis->specializationConstants.useBluesteinFFT)) || ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (axis->specializationConstants.useBluesteinFFT && ((reverseBluesteinMultiUpload == 1) || (FFTPlan->numAxisUploads[axis_id] == 1)))))) || (((!app->configuration.frequencyZeroPadding) && (app->configuration.FFTdim - 1 == axis_id) && (axis_upload_id == 0) && (FFTPlan->numAxisUploads[axis_id] == 1) && (app->configuration.performConvolution)))) {
				axis->specializationConstants.zeropad[1] = (int)app->configuration.performZeropadding[axis_id];
				axis->specializationConstants.fft_zeropad_left_write[axis_id].type = 31;
				axis->specializationConstants.fft_zeropad_left_write[axis_id].data.i = (pfINT)app->configuration.fft_zeropad_left[axis_id] / zeropad_even_r2c_multiupload_scale;
				axis->specializationConstants.fft_zeropad_right_write[axis_id].type = 31;
				axis->specializationConstants.fft_zeropad_right_write[axis_id].data.i = (pfINT)app->configuration.fft_zeropad_right[axis_id] / zeropad_even_r2c_multiupload_scale;
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


        pfUINT tempSize[3];

		if (axis_id == 0) {
			if (axis_upload_id == 0) {
				pfUINT batchingSize = (axis->specializationConstants.axisSwapped) ? axis->axisBlock[0] : axis->axisBlock[1];
                if (FFTPlan->numAxisUploads[0] > 2) {
                    tempSize[0] = (pfUINT)pfceil((pfUINT)pfceil(FFTPlan->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)batchingSize) / (double)FFTPlan->axisSplit[0][1]) * FFTPlan->axisSplit[0][1];
                    tempSize[1] = FFTPlan->actualFFTSizePerAxis[0][1];
                }
                else {
                    if (FFTPlan->numAxisUploads[0] > 1) {
                        tempSize[0] = (pfUINT)pfceil((pfUINT)pfceil(FFTPlan->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)batchingSize));
                        tempSize[1] = FFTPlan->actualFFTSizePerAxis[0][1];
                    }
                    else {
                        tempSize[0] = FFTPlan->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i;
                        tempSize[1] = (pfUINT)pfceil(FFTPlan->actualFFTSizePerAxis[0][1] / (double)batchingSize);
                    }
                }
            }
            else {
				pfUINT fftDimensionSize = (axis->specializationConstants.axisSwapped) ? axis->axisBlock[1] : axis->axisBlock[0];

                tempSize[0] = (pfUINT)pfceil(FFTPlan->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)fftDimensionSize);
                tempSize[1] = FFTPlan->actualFFTSizePerAxis[0][1];
            }
			if ((FFTPlan->actualPerformR2CPerAxis[axis_id] == 1) && (axis->specializationConstants.mergeSequencesR2C)) tempSize[1] = (pfUINT)pfceil(tempSize[1] / 2.0);
			//if (app->configuration.performZeropadding[1]) tempSize[1] = (pfUINT)pfceil(tempSize[1] / 2.0);
			//if (app->configuration.performZeropadding[2]) tempSize[2] = (pfUINT)pfceil(tempSize[2] / 2.0);
        }else{
			tempSize[0] = (pfUINT)pfceil(FFTPlan->actualFFTSizePerAxis[axis_id][0] / (double)axis->axisBlock[0] * FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (double)axis->specializationConstants.fftDim.data.i);
			tempSize[1] = 1;
			
			//if (app->configuration.actualPerformR2C == 1) tempSize[0] = (pfUINT)pfceil(tempSize[0] / 2.0);
			//if (app->configuration.performZeropadding[2]) tempSize[2] = (pfUINT)pfceil(tempSize[2] / 2.0);
		}
        tempSize[2] = 1;
        for (pfUINT i = 1; i < app->configuration.FFTdim; i++) {
            if (i!=axis_id)
                tempSize[2] *= FFTPlan->actualFFTSizePerAxis[axis_id][i];
        }
        tempSize[2] *= app->configuration.numberKernels * app->configuration.numberBatches;
        if (!(axis->specializationConstants.convolutionStep && (app->configuration.matrixConvolution > 1))) tempSize[2] *= app->configuration.coordinateFeatures;
        
		if ((app->configuration.maxComputeWorkGroupCount[0] > app->configuration.maxComputeWorkGroupCount[1]) && (tempSize[1] > app->configuration.maxComputeWorkGroupCount[1]) && (tempSize[1] > tempSize[0]) && (tempSize[1] >= tempSize[2])) {
			pfUINT temp_tempSize = tempSize[0];
			tempSize[0] = tempSize[1];
			tempSize[1] = temp_tempSize;
			axis->specializationConstants.swapComputeWorkGroupID = 1;
		}
		else {
			if ((app->configuration.maxComputeWorkGroupCount[0] > app->configuration.maxComputeWorkGroupCount[2]) && (tempSize[2] > app->configuration.maxComputeWorkGroupCount[2]) && (tempSize[2] > tempSize[0]) && (tempSize[2] >= tempSize[1])) {
				pfUINT temp_tempSize = tempSize[0];
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
				axis->pushConstants.structSize *= sizeof(pfUINT);
			else
				axis->pushConstants.structSize *= sizeof(uint32_t);
			axis->specializationConstants.pushConstantsStructSize = (int)axis->pushConstants.structSize;
		}
		//pfUINT LUT = app->configuration.useLUT;
		pfUINT type = 0;
		if ((axis_id == 0) && (axis_upload_id == 0)) type = 0;
		if (axis_id != 0) type = 1;
		if ((axis_id == 0) && (axis_upload_id > 0)) type = 2;
		//if ((axis->specializationConstants.fftDim == 8 * maxSequenceLengthSharedMemory) && (app->configuration.registerBoost >= 8)) axis->specializationConstants.registerBoost = 8;
		if ((!axis->specializationConstants.actualInverse) && (FFTPlan->actualPerformR2CPerAxis[axis_id])) type += 500;
		if ((axis->specializationConstants.actualInverse) && (FFTPlan->actualPerformR2CPerAxis[axis_id])) type += 600;
		if (FFTPlan->actualPerformR2CPerAxis[axis_id] && ((axis->specializationConstants.numAxisUploads > 1) || (axis->specializationConstants.forceCallbackVersionRealTransforms))) type += 200;
		if (((axis->specializationConstants.performDCT == 1) || (axis->specializationConstants.performDST == 1))) type += 1100;
		if (((((axis->specializationConstants.performDCT == 2) || (axis->specializationConstants.performDST == 2)) && (!inverse)) || (((axis->specializationConstants.performDCT == 3) || (axis->specializationConstants.performDST == 3)) && (inverse)))) type += 1200;
		if (((((axis->specializationConstants.performDCT == 2) || (axis->specializationConstants.performDST == 2)) && (inverse)) || (((axis->specializationConstants.performDCT == 3) || (axis->specializationConstants.performDST == 3)) && (!inverse)))) type += 1300;
		if (((axis->specializationConstants.performDCT == 4) || (axis->specializationConstants.performDST == 4)) && ((app->configuration.size[axis_id] % 2) == 0)) type += 1400;
		if (((axis->specializationConstants.performDCT == 4) || (axis->specializationConstants.performDST == 4)) && ((app->configuration.size[axis_id] % 2) == 1)) type += 1420;
		
		if (((axis->specializationConstants.performDCT) || (axis->specializationConstants.performDST)) && ((axis->specializationConstants.numAxisUploads > 1) || (axis->specializationConstants.forceCallbackVersionRealTransforms))) type += 10;
		
		resFFT = initMemoryParametersAPI(app, &axis->specializationConstants);
		if (resFFT != VKFFT_SUCCESS) {
			deleteVkFFT(app);
			return resFFT;
		}

		switch (type/10) {
		case 50: case 70: case 110: case 111: case 120: case 121: case 130: case 131: case 140: case 141: case 142: case 143:
			if ((axis->specializationConstants.axis_upload_id == (axis->specializationConstants.numAxisUploads-1)) && (type != 501) && (type != 701) && (!((app->useBluesteinFFT[axis_id] && (reverseBluesteinMultiUpload == 1))))) {
				axis->specializationConstants.inputMemoryCode = axis->specializationConstants.floatTypeInputMemoryCode;
				axis->specializationConstants.inputNumberByteSize = (1 << (1 + (axis->specializationConstants.inputMemoryCode % 100) / 10));
			}
			else {
				axis->specializationConstants.inputMemoryCode = axis->specializationConstants.vecTypeInputMemoryCode;
				axis->specializationConstants.inputNumberByteSize = 2 * (1 << (1 + (axis->specializationConstants.inputMemoryCode % 100) / 10));
			}
			break;
		default:
			axis->specializationConstants.inputMemoryCode = axis->specializationConstants.vecTypeInputMemoryCode;
			axis->specializationConstants.inputNumberByteSize = 2 * (1 << (1 + (axis->specializationConstants.inputMemoryCode % 100) / 10));
			break;
		}
		switch (type/10) {
		case 60: case 80: case 110: case 111: case 120: case 121: case 130: case 131: case 140: case 141: case 142: case 143:
			if (((axis->specializationConstants.axis_upload_id == 0) && (type != 601) && (type != 801) && (!((app->useBluesteinFFT[axis_id] && (reverseBluesteinMultiUpload == 0) && (axis->specializationConstants.numAxisUploads > 1))))) || ((axis->specializationConstants.axis_upload_id == (axis->specializationConstants.numAxisUploads-1)) && (app->useBluesteinFFT[axis_id] && (reverseBluesteinMultiUpload == 1)))) {
				axis->specializationConstants.outputMemoryCode = axis->specializationConstants.floatTypeOutputMemoryCode;
				axis->specializationConstants.outputNumberByteSize = (1 << (1 + (axis->specializationConstants.outputMemoryCode % 100) / 10));
			}
			else {
				axis->specializationConstants.outputMemoryCode = axis->specializationConstants.vecTypeOutputMemoryCode;
				axis->specializationConstants.outputNumberByteSize = 2 * (1 << (1 + (axis->specializationConstants.outputMemoryCode % 100) / 10));
			}
			break;
		default:
			axis->specializationConstants.outputMemoryCode = axis->specializationConstants.vecTypeOutputMemoryCode;
			axis->specializationConstants.outputNumberByteSize = 2 * (1 << (1 + (axis->specializationConstants.outputMemoryCode % 100) / 10));
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
		case 3:
			axis->specializationConstants.kernelNumberByteSize = 32;
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
		pfUINT temp = axis->axisBlock[1];
		axis->axisBlock[1] = axis->axisBlock[0];
		axis->axisBlock[0] = temp;
		axis->specializationConstants.axisSwapped = 0;
	}
	return resFFT;
}

#endif
