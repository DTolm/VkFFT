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
#ifndef VKFFT_PLAN_R2C_EVEN_DECOMPOSITION_H
#define VKFFT_PLAN_R2C_EVEN_DECOMPOSITION_H
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT/vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_InitAPIParameters.h"
#include "vkFFT/vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_CompileKernel.h"
#include "vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_ManageLUT.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel2/vkFFT_R2C_even_decomposition.h"
#include "vkFFT/vkFFT_AppManagement/vkFFT_DeleteApp.h"
static inline VkFFTResult VkFFTPlanR2CMultiUploadDecomposition(VkFFTApplication* app, VkFFTPlan* FFTPlan, pfUINT inverse) {
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
	VkFFTAxis* axis = &FFTPlan->R2Cdecomposition;
	axis->specializationConstants.sourceFFTSize.type = 31;
	axis->specializationConstants.sourceFFTSize.data.i = (pfINT)app->configuration.size[0];
    axis->specializationConstants.numFFTdims = (int)app->configuration.FFTdim;
    
	axis->specializationConstants.warpSize = (int)app->configuration.warpSize;
	axis->specializationConstants.numSharedBanks = (int)app->configuration.numSharedBanks;
	axis->specializationConstants.useUint64 = (int)app->configuration.useUint64;
#if(VKFFT_BACKEND==2)
	axis->specializationConstants.useStrict32BitAddress = app->configuration.useStrict32BitAddress;
#endif
	axis->specializationConstants.disableSetLocale = (int)app->configuration.disableSetLocale;

	axis->specializationConstants.numAxisUploads = (int)FFTPlan->numAxisUploads[0];
	axis->specializationConstants.reorderFourStep = ((FFTPlan->numAxisUploads[0] > 1) && (!app->useBluesteinFFT[0])) ? (int)app->configuration.reorderFourStep : 0;

	axis->specializationConstants.maxCodeLength = (int)app->configuration.maxCodeLength;
	axis->specializationConstants.maxTempLength = (int)app->configuration.maxTempLength;

	axis->specializationConstants.double_PI = pfFPinit("3.14159265358979323846264338327950288419716939937510");
	if (app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) {
		axis->specializationConstants.precision = 3;
		axis->specializationConstants.complexSize = (4 * sizeof(double));
	}
	else
	{
		if (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory) {
			axis->specializationConstants.precision = 1;
			axis->specializationConstants.complexSize = (2 * sizeof(double));
		}
		else {
			if (app->configuration.halfPrecision) {
				axis->specializationConstants.precision = 0;
				axis->specializationConstants.complexSize = (2 * sizeof(float));
			}
			else {
				axis->specializationConstants.precision = 0;
				axis->specializationConstants.complexSize = (2 * sizeof(float));
			}
		}
	}
	axis->specializationConstants.supportAxis = 0;
	axis->specializationConstants.symmetricKernel = (int)app->configuration.symmetricKernel;
	axis->specializationConstants.conjugateConvolution = (int)app->configuration.conjugateConvolution;
	axis->specializationConstants.crossPowerSpectrumNormalization = (int)app->configuration.crossPowerSpectrumNormalization;
	axis->specializationConstants.fft_dim_full.type = 31;
	axis->specializationConstants.fft_dim_full.data.i = (int)app->configuration.size[0];
	axis->specializationConstants.dispatchZactualFFTSize.type = 31;
	axis->specializationConstants.dispatchZactualFFTSize.data.i = 1;
	//allocate LUT
	resFFT = VkFFT_AllocateLUT_R2C(app, FFTPlan, axis, inverse);
	if (resFFT != VKFFT_SUCCESS) {
		deleteVkFFT(app);
		return resFFT;
	}
	//configure strides
	PfContainer* axisStride = axis->specializationConstants.inputStride;
	PfContainer* usedStride = 0;
	if (app->useBluesteinFFT[0] && (FFTPlan->numAxisUploads[0] > 1)) {
		if (inverse)
			usedStride = FFTPlan->axes[0][FFTPlan->numAxisUploads[0] - 1].specializationConstants.inputStride;
		else
			usedStride = FFTPlan->inverseBluesteinAxes[0][FFTPlan->numAxisUploads[0] - 1].specializationConstants.outputStride;
	}
	else {
		if (inverse)
			usedStride = FFTPlan->axes[0][FFTPlan->numAxisUploads[0] - 1].specializationConstants.inputStride;
		else
			usedStride = FFTPlan->axes[0][0].specializationConstants.outputStride;
	}
	for (int i = 0; i < app->configuration.FFTdim+2; i++){
        axisStride[i].type = 31;
        axisStride[i].data.i = usedStride[i].data.i;
    }

	axisStride = axis->specializationConstants.outputStride;
	usedStride = axis->specializationConstants.inputStride;

    for (int i = 0; i < app->configuration.FFTdim+2; i++){
        axisStride[i].type = 31;
        axisStride[i].data.i = usedStride[i].data.i;
    }
    
	axis->specializationConstants.inverse = (int)inverse;
	pfUINT axis_id = 0;
	pfUINT axis_upload_id = 0;

	resFFT = VkFFTConfigureDescriptorsR2CMultiUploadDecomposition(app, FFTPlan, axis, axis_id, axis_upload_id, inverse);
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
	resFFT = VkFFTUpdateBufferSetR2CMultiUploadDecomposition(app, FFTPlan, axis, axis_id, axis_upload_id, inverse);
	if (resFFT != VKFFT_SUCCESS) {
		deleteVkFFT(app);
		return resFFT;
	}
	{
		axis->axisBlock[0] = 128;
		if (axis->axisBlock[0] > app->configuration.maxThreadsNum) axis->axisBlock[0] = app->configuration.maxThreadsNum;
		axis->axisBlock[1] = 1;
		axis->axisBlock[2] = 1;

        pfUINT tempSize[3] = {1, 1, 1};
        for (int i = 0; i < app->configuration.FFTdim; i++){
            tempSize[0] *= app->configuration.size[i];
        }
        tempSize[0] = (pfUINT)pfceil(tempSize[0]/ (pfLD)(2 * axis->axisBlock[0]));
       
		tempSize[2] *= app->configuration.numberKernels * app->configuration.numberBatches * app->configuration.coordinateFeatures;
        
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

		axis->specializationConstants.localSize[0].type = 31;
		axis->specializationConstants.localSize[1].type = 31;
		axis->specializationConstants.localSize[2].type = 31;
		axis->specializationConstants.localSize[0].data.i = axis->axisBlock[0];
		axis->specializationConstants.localSize[1].data.i = axis->axisBlock[1];
		axis->specializationConstants.localSize[2].data.i = axis->axisBlock[2];

		axis->specializationConstants.numCoordinates = (app->configuration.matrixConvolution > 1) ? 1 : (int)app->configuration.coordinateFeatures;
		axis->specializationConstants.matrixConvolution = (int)app->configuration.matrixConvolution;
        for (pfUINT i = 0; i < VKFFT_MAX_FFT_DIMENSIONS; i++) {
            axis->specializationConstants.size[i].type = 31;
            axis->specializationConstants.size[i].data.i = (pfINT)app->configuration.size[i];
        }

		axis->specializationConstants.registers_per_thread = 4;

		axis->specializationConstants.numBatches.type = 31;
		axis->specializationConstants.numBatches.data.i = (pfINT)app->configuration.numberBatches;
		if ((app->configuration.FFTdim == 1) && (app->configuration.size[1] == 1) && ((app->configuration.numberBatches == 1) && (app->actualNumBatches > 1)) && (!app->configuration.performConvolution) && (app->configuration.coordinateFeatures == 1)) {
			axis->specializationConstants.numBatches.data.i = (pfINT)app->actualNumBatches;
		}

		axis->specializationConstants.numKernels.type = 31;
		axis->specializationConstants.numKernels.data.i = (pfINT)app->configuration.numberKernels;
		axis->specializationConstants.sharedMemSize = (int)app->configuration.sharedMemorySize;
		axis->specializationConstants.sharedMemSizePow2 = (int)app->configuration.sharedMemorySizePow2;
		axis->specializationConstants.normalize = (int)app->configuration.normalize;
		axis->specializationConstants.axis_id = 0;
		axis->specializationConstants.axis_upload_id = 0;

		for (pfUINT i = 0; i < VKFFT_MAX_FFT_DIMENSIONS; i++) {
			axis->specializationConstants.frequencyZeropadding = (int)app->configuration.frequencyZeroPadding;
			axis->specializationConstants.performZeropaddingFull[i] = (int)app->configuration.performZeropadding[i]; // don't read if input is zeropadded (0 - off, 1 - on)
			axis->specializationConstants.fft_zeropad_left_full[i].type = 31;
			axis->specializationConstants.fft_zeropad_left_full[i].data.i = (pfINT)app->configuration.fft_zeropad_left[i];
			axis->specializationConstants.fft_zeropad_right_full[i].type = 31;
			axis->specializationConstants.fft_zeropad_right_full[i].data.i = (pfINT)app->configuration.fft_zeropad_right[i];
		}
		/*if ((inverse)) {
			if ((app->configuration.frequencyZeroPadding) &&  (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1)) {
				axis->specializationConstants.zeropad[0] = app->configuration.performZeropadding[axis_id];
				axis->specializationConstants.fft_zeropad_left_read[axis_id] = app->configuration.fft_zeropad_left[axis_id];
				axis->specializationConstants.fft_zeropad_right_read[axis_id] = app->configuration.fft_zeropad_right[axis_id];
			}
			else
				axis->specializationConstants.zeropad[0] = 0;
			if ((!app->configuration.frequencyZeroPadding) && (axis_upload_id == 0)) {
				axis->specializationConstants.zeropad[1] = app->configuration.performZeropadding[axis_id];
				axis->specializationConstants.fft_zeropad_left_write[axis_id] = app->configuration.fft_zeropad_left[axis_id];
				axis->specializationConstants.fft_zeropad_right_write[axis_id] = app->configuration.fft_zeropad_right[axis_id];
			}
			else
				axis->specializationConstants.zeropad[1] = 0;
		}
		else {
			if ((!app->configuration.frequencyZeroPadding) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1)) {
				axis->specializationConstants.zeropad[0] = app->configuration.performZeropadding[axis_id];
				axis->specializationConstants.fft_zeropad_left_read[axis_id] = app->configuration.fft_zeropad_left[axis_id];
				axis->specializationConstants.fft_zeropad_right_read[axis_id] = app->configuration.fft_zeropad_right[axis_id];
			}
			else
				axis->specializationConstants.zeropad[0] = 0;
			if (((app->configuration.frequencyZeroPadding) && (axis_upload_id == 0)) || (((app->configuration.FFTdim - 1 == axis_id) && (axis_upload_id == 0) && (app->configuration.performConvolution)))) {
				axis->specializationConstants.zeropad[1] = app->configuration.performZeropadding[axis_id];
				axis->specializationConstants.fft_zeropad_left_write[axis_id] = app->configuration.fft_zeropad_left[axis_id];
				axis->specializationConstants.fft_zeropad_right_write[axis_id] = app->configuration.fft_zeropad_right[axis_id];
			}
			else
				axis->specializationConstants.zeropad[1] = 0;
		}*/
		if ((app->configuration.FFTdim == 1) && (app->configuration.performConvolution)) {
			axis->specializationConstants.convolutionStep = 1;
		}
		else
			axis->specializationConstants.convolutionStep = 0;
		
		axis->specializationConstants.LUT = (app->configuration.useLUT == 1) ? 1 : 0;
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

		resFFT = initMemoryParametersAPI(app, &axis->specializationConstants);
		if (resFFT != VKFFT_SUCCESS) {
			deleteVkFFT(app);
			return resFFT;
		}

		pfUINT type = 0;

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
		case 3:
			axis->specializationConstants.inputNumberByteSize = 32;
			break;
		}

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
		case 3:
			axis->specializationConstants.outputNumberByteSize = 32;
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
		sprintf(axis->VkFFTFunctionName, "VkFFT_main_R2C");
#endif
		resFFT = shaderGen_R2C_even_decomposition(&axis->specializationConstants, (int)type);
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
	return resFFT;
}

#endif
