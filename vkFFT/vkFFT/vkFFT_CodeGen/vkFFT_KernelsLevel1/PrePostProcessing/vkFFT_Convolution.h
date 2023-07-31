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
#ifndef VKFFT_CONVOLUTION_H
#define VKFFT_CONVOLUTION_H
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_StringManagement/vkFFT_StringManager.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_MathUtils/vkFFT_MathUtils.h"

#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_Zeropad.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_KernelUtils.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryTransfers/vkFFT_Transfers.h"

#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel1/vkFFT_ReadWrite.h"
static inline void appendRegisterStorage(VkFFTSpecializationConstantsLayout* sc, int readType, int readWrite) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_int1 = VKFFT_ZERO_INIT;
	temp_int1.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 32;

	PfContainer used_registers = VKFFT_ZERO_INIT;
	used_registers.type = 31;

	PfContainer localSize = VKFFT_ZERO_INIT;
	localSize.type = 31;

	PfContainer* localInvocationID = VKFFT_ZERO_INIT;

	if (sc->stridedSharedLayout) {
		localSize.data.i = sc->localSize[1].data.i;
		localInvocationID = &sc->gl_LocalInvocationID_y;
		PfDivCeil(sc, &used_registers, &sc->fftDim, &sc->localSize[1]);
	}
	else {
		localSize.data.i = sc->localSize[0].data.i;
		localInvocationID = &sc->gl_LocalInvocationID_x;
		PfDivCeil(sc, &used_registers, &sc->fftDim, &sc->localSize[0]);
	}

	if (((!sc->writeFromRegisters) && (readWrite == 0)) || ((!sc->readToRegisters) && (readWrite == 1)) || ((sc->convolutionStep) && ((sc->matrixConvolution > 1) || (sc->numKernels.data.i > 1)))) {
		appendBarrierVkFFT(sc);

		if (sc->useDisableThreads) {
			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
		}

		for (uint64_t i = 0; i < (uint64_t)used_registers.data.i; i++) {
			if (localSize.data.i * ((1 + (int64_t)i)) > sc->fftDim.data.i) {
				temp_int.data.i = sc->fftDim.data.i - i * localSize.data.i;
				PfIf_lt_start(sc, localInvocationID, &temp_int);
			}
			if (sc->stridedSharedLayout) {
				temp_int.data.i = i * sc->localSize[1].data.i;
				PfAdd(sc, &sc->sdataID, &sc->gl_LocalInvocationID_y, &temp_int);
				PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);

				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);
			}
			else {
				PfMul(sc, &sc->sdataID, &sc->gl_LocalInvocationID_y, &sc->sharedStride, 0);

				temp_int.data.i = i * sc->localSize[0].data.i;
				PfAdd(sc, &sc->tempInt, &sc->gl_LocalInvocationID_x, &temp_int);
				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);

			}
			if (readWrite)
				appendRegistersToShared(sc, &sc->sdataID, &sc->regIDs[sc->coordinate.data.i * sc->registers_per_thread + i]);
			else
				appendSharedToRegisters(sc, &sc->regIDs[sc->coordinate.data.i * sc->registers_per_thread + i], &sc->sdataID);

			if (localSize.data.i * ((1 + (int64_t)i)) > sc->fftDim.data.i) {
				PfIf_end(sc);
			}
		}

		if (sc->useDisableThreads) {
			PfIf_end(sc);
		}
	}
	return;
}

static inline void appendPreparationBatchedKernelConvolution(VkFFTSpecializationConstantsLayout* sc, int dataType) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_int1 = VKFFT_ZERO_INIT;
	temp_int1.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 32; 
	
	for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
		//sc->tempLen = sprintf(sc->tempStr, "			temp%s[i]=temp[i];\n", separateRegisterStore);
		for (uint64_t j = 0; j < sc->matrixConvolution; j++) {
			PfMov(sc, &sc->regIDs_copy[i + j * sc->registers_per_thread], &sc->regIDs[i + j * sc->registers_per_thread]);
		}
	}
	return;
}


static inline void appendKernelConvolution(VkFFTSpecializationConstantsLayout* sc, uint64_t strideType) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_int1 = VKFFT_ZERO_INIT;
	temp_int1.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 32;

	PfContainer localSize = VKFFT_ZERO_INIT;
	localSize.type = 31;

	PfContainer batching_localSize = VKFFT_ZERO_INIT;
	batching_localSize.type = 31;

	PfContainer* localInvocationID = VKFFT_ZERO_INIT;
	PfContainer* batchingInvocationID = VKFFT_ZERO_INIT;

	if (sc->stridedSharedLayout) {
		batching_localSize.data.i = sc->localSize[0].data.i;
		localSize.data.i = sc->localSize[1].data.i;
		localInvocationID = &sc->gl_LocalInvocationID_y;
		batchingInvocationID = &sc->gl_LocalInvocationID_x;
	}
	else {
		batching_localSize.data.i = sc->localSize[1].data.i;
		localSize.data.i = sc->localSize[0].data.i;
		localInvocationID = &sc->gl_LocalInvocationID_x;
		batchingInvocationID = &sc->gl_LocalInvocationID_y;
	}
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	PfContainer used_registers = VKFFT_ZERO_INIT;
	used_registers.type = 31;

	if (sc->stridedSharedLayout) {
		PfDivCeil(sc, &used_registers, &sc->fftDim, &sc->localSize[1]);
	}
	else {
		PfDivCeil(sc, &used_registers, &sc->fftDim, &sc->localSize[0]);
	}
	
	appendKernelOffset(sc, 0, (int)strideType);
	if (strideType == 0) {
		if (sc->fftDim.data.i != sc->fft_dim_full.data.i) {
			PfDiv(sc, &temp_int, &sc->firstStageStartSize, &sc->fftDim);
			PfMod(sc, &sc->tempInt, &sc->shiftX, &temp_int);

			PfDiv(sc, &sc->tempInt, &sc->shiftX, &temp_int);
			PfMul(sc, &temp_int, &batching_localSize, &sc->firstStageStartSize, 0);
			PfMul(sc, &sc->tempInt, &sc->tempInt, &temp_int, 0);

			//sc->tempLen = sprintf(sc->tempStr, "		%s numActiveThreads = ((%s/%" PRIu64 ")==%" PRIu64 ") ? %" PRIu64 " : %" PRIu64 ";\n", uintType, sc->gl_WorkGroupID_x, sc->firstStageStartSize / sc->fftDim, ((uint64_t)floor(sc->fft_dim_full / ((double)sc->localSize[0] * sc->fftDim))) / (sc->firstStageStartSize / sc->fftDim), (uint64_t)ceil(((sc->fft_dim_full - (sc->firstStageStartSize / sc->fftDim) * ((((uint64_t)floor(sc->fft_dim_full / ((double)sc->localSize[0] * sc->fftDim))) / (sc->firstStageStartSize / sc->fftDim)) * sc->localSize[0] * sc->fftDim)) / (sc->firstStageStartSize / sc->fftDim)) / (double)used_registers_read), sc->localSize[0] * sc->localSize[1]);// sc->fft_dim_full, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fftDim, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->localSize[0] * sc->firstStageStartSize, sc->fft_dim_full / (sc->localSize[0] * sc->fftDim));
			temp_int.data.i = sc->firstStageStartSize.data.i / sc->fftDim.data.i;
			PfDiv(sc, &sc->tempInt, &sc->gl_WorkGroupID_x, &temp_int);
			temp_int1.data.i = ((int64_t)floor(sc->fft_dim_full.data.i / ((long double)batching_localSize.data.i * sc->fftDim.data.i))) / (sc->firstStageStartSize.data.i / sc->fftDim.data.i);
			PfIf_eq_start(sc, &sc->tempInt, &temp_int1);
			temp_int.data.i = (int64_t)ceil(((sc->fft_dim_full.data.i - (sc->firstStageStartSize.data.i / sc->fftDim.data.i) * ((((int64_t)floor(sc->fft_dim_full.data.i / ((long double)batching_localSize.data.i * sc->fftDim.data.i))) / (sc->firstStageStartSize.data.i / sc->fftDim.data.i)) * batching_localSize.data.i * sc->fftDim.data.i)) / (sc->firstStageStartSize.data.i / sc->fftDim.data.i)) / (long double)used_registers.data.i);
			PfMov(sc, &sc->sdataID, &temp_int);
			PfIf_else(sc);
			temp_int.data.i = sc->localSize[0].data.i * sc->localSize[1].data.i;
			PfMov(sc, &sc->sdataID, &temp_int);
			PfIf_end(sc);

			if (sc->stridedSharedLayout) {

				PfMul(sc, &sc->tempInt, &sc->gl_LocalInvocationID_x, &sc->firstStageStartSize, 0);
				PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->tempInt2);
			}
			else {
				PfMul(sc, &sc->tempInt, &sc->gl_LocalInvocationID_y, &sc->firstStageStartSize, 0);
				PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->tempInt2);
			}

			/*sc->useDisableThreads = 1;
			PfIf_ge_start(sc, &sc->tempInt, &sc->fft_dim_full);
			temp_int.data.i = 0;
			PfMov(sc, &sc->disableThreads, &temp_int);
			PfIf_end(sc);*/

			PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);
			PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
			PfIf_lt_start(sc, &sc->combinedID, &sc->sdataID);
		}
	}
	else {
		if (sc->axis_id > 0) {
			PfMod(sc, &sc->inoutID_x, &sc->shiftX, &sc->fft_dim_x);
			//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		disableThreads = (((%s%s) / %" PRIu64 ") %% (%" PRIu64 ")+((%s%s) / %" PRIu64 ") * (%" PRIu64 ") < %" PRIu64 ") ? 1 : 0;\n", &sc->gl_GlobalInvocationID_x, shiftX, &sc->fft_dim_x, &sc->stageStartSize, &sc->gl_GlobalInvocationID_x, shiftX, &sc->fft_dim_x * &sc->stageStartSize, &sc->fftDim * &sc->stageStartSize, &sc->size[&sc->axis_id]);

			if (sc->numAxisUploads > 1) {
				PfIf_lt_start(sc, &sc->tempInt2, &sc->fft_dim_full);
			}
			else {
				PfIf_lt_start(sc, &sc->tempInt2, &sc->sourceFFTSize);
			}
		}
		else {
			//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		disableThreads = (((%s%s) / %" PRIu64 ") * (%" PRIu64 ") < %" PRIu64 ") ? 1 : 0;\n", &sc->gl_GlobalInvocationID_x, shiftX, &sc->stageStartSize, &sc->stageStartSize * &sc->fftDim, &sc->fft_dim_full);
			PfIf_lt_start(sc, &sc->tempInt2, &sc->fft_dim_full);
		}

		if (sc->axis_id > 0) {
			PfMul(sc, &sc->inoutID_y, &sc->gl_LocalInvocationID_y, &sc->stageStartSize, 0);
			PfAdd(sc, &sc->inoutID_y, &sc->inoutID_y, &sc->tempInt2);

		}
		else {
			//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		inoutID = (%s%s) %% (%" PRIu64 ") + %" PRIu64 " * (%s + %" PRIu64 ") + ((%s%s) / %" PRIu64 ") * (%" PRIu64 ");\n", &sc->gl_GlobalInvocationID_x, shiftX, &sc->stageStartSize, &sc->stageStartSize, &sc->gl_LocalInvocationID_y, (i + k * used_registers) * &sc->localSize[1], &sc->gl_GlobalInvocationID_x, shiftX, &sc->stageStartSize, &sc->stageStartSize * &sc->fftDim);
			PfMod(sc, &sc->inoutID_x, &sc->shiftX, &sc->stageStartSize);

			PfMul(sc, &sc->tempInt, &sc->gl_LocalInvocationID_y, &sc->stageStartSize, 0);

			PfAdd(sc, &sc->inoutID_x, &sc->inoutID_x, &sc->tempInt);

			PfAdd(sc, &sc->inoutID_x, &sc->inoutID_x, &sc->tempInt2);
		}
		PfMov(sc, &sc->inoutID, &sc->inoutID_x);

		if (sc->axis_id > 0) {
			PfMul(sc, &sc->tempInt, &sc->inoutID_y, &sc->inputStride[1], 0);
			PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
		}
		PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->blockInvocationID);
	}

	for (uint64_t i = 0; i < (uint64_t)used_registers.data.i; i++) {
		for (int64_t j = 0; j < sc->matrixConvolution; j++) {
			PfSetToZero(sc, &sc->temp_conv[j]);
		}
		if (localSize.data.i * ((1 + (int64_t)i)) > sc->fftDim.data.i) {
			temp_int.data.i = sc->fftDim.data.i - i * localSize.data.i;
			PfIf_lt_start(sc, localInvocationID, &temp_int);
		}
		switch (strideType) {
		case 0:
		{
			if (sc->fftDim.data.i == sc->fft_dim_full.data.i) {
				if (sc->localSize[1].data.i == 1) {
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		combinedID = %s + %" PRIu64 ";\n", &sc->gl_LocalInvocationID_x, (i + k * used_registers) * &sc->localSize[0]);
					temp_int.data.i = (i)*sc->localSize[0].data.i;

					PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
				}
				else {
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		combinedID = (%s + %" PRIu64 " * %s) + %" PRIu64 ";\n", &sc->gl_LocalInvocationID_x, &sc->localSize[0], &sc->gl_LocalInvocationID_y, (i + k * used_registers) * &sc->localSize[0] * &sc->localSize[1]);
					PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

					temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

					PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
					PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
				}
				PfMod(sc, &sc->inoutID_x, &sc->combinedID, &sc->fftDim);
				PfDiv(sc, &sc->inoutID_y, &sc->combinedID, &sc->fftDim);

				PfAdd(sc, &sc->inoutID_y, &sc->inoutID_y, &sc->shiftY);

				if ((sc->size[1].data.i % temp_int.data.i) != 0) {
#if (VKFFT_BACKEND!=2) //AMD compiler fix
					PfIf_lt_start(sc, &sc->inoutID_y, &sc->size[1]);
#else
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(!(combinedID / %" PRIu64 " + (%s%s)*%" PRIu64 "< %" PRIu64 ")) %s = 0; {\n", &sc->fftDim, &sc->gl_WorkGroupID_y, shiftY, &sc->localSize[0], &sc->size[&sc->axis_id + 1], &sc->inoutID);
					PfIf_ge_start(sc, &sc->inoutID_y, &sc->size[1]);
					PfSetToZero(sc, &sc->inoutID_x);
					PfSetToZero(sc, &sc->inoutID_y);
					PfIf_end(sc);
#endif
				}

			}
			else {
				PfMod(sc, &sc->inoutID_x, &sc->combinedID, &sc->fftDim);

				PfDiv(sc, &sc->tempInt, &sc->combinedID, &sc->fftDim);
				PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->firstStageStartSize, 0);
				PfAdd(sc, &sc->inoutID_x, &sc->inoutID_x, &sc->tempInt);

				PfAdd(sc, &sc->inoutID_x, &sc->inoutID_x, &sc->tempInt2);
			}

			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;

			temp_int1.data.i = sc->fftDim.data.i * batching_localSize.data.i;

			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
			PfMov(sc, &sc->inoutID, &sc->inoutID_x);

			if (sc->fftDim.data.i == sc->fft_dim_full.data.i) {
				PfMul(sc, &sc->tempInt, &sc->inoutID_y, &sc->inputStride[1], 0);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
			}
			PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->blockInvocationID);
		}
		break;
		case 1:
		{
			temp_int1.data.i = (i + 1) * sc->localSize[1].data.i;

			if (temp_int1.data.i > sc->fftDim.data.i) {
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(%s < %" PRIu64 "){\n", &sc->gl_LocalInvocationID_y, &sc->fftDim - (i + k * used_registers) * &sc->localSize[1]);
				temp_int1.data.i = sc->localSize[1].data.i - (temp_int1.data.i - sc->fftDim.data.i);
				PfIf_lt_start(sc, &sc->gl_LocalInvocationID_y, &temp_int1);
			}
			if (i > 0) {
				if (sc->axis_id > 0) {
					temp_int1.data.i = sc->stageStartSize.data.i * sc->inputStride[1].data.i * sc->localSize[1].data.i;
					PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int1);

				}
				else {
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		inoutID = (%s%s) %% (%" PRIu64 ") + %" PRIu64 " * (%s + %" PRIu64 ") + ((%s%s) / %" PRIu64 ") * (%" PRIu64 ");\n", &sc->gl_GlobalInvocationID_x, shiftX, &sc->stageStartSize, &sc->stageStartSize, &sc->gl_LocalInvocationID_y, (i + k * used_registers) * &sc->localSize[1], &sc->gl_GlobalInvocationID_x, shiftX, &sc->stageStartSize, &sc->stageStartSize * &sc->fftDim);
					temp_int1.data.i = sc->stageStartSize.data.i * sc->inputStride[0].data.i * sc->localSize[1].data.i;
					PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int1);
				}
			}
		}
		break;
		}
		for (uint64_t j = 0; j < sc->matrixConvolution; j++) {
			for (uint64_t l = 0; l < sc->matrixConvolution; l++) {
				uint64_t k = 0;
				if (sc->symmetricKernel) {
					k = (l < j) ? (l * sc->matrixConvolution - l * l + j) : (j * sc->matrixConvolution - j * j + l);
				}
				else {
					k = (j * sc->matrixConvolution + l);
				}
				temp_int.data.i = k * sc->inputStride[sc->numFFTdims].data.i;
				PfAdd(sc, &sc->tempInt, &sc->inoutID, &temp_int);
				appendGlobalToRegisters(sc, &sc->temp, &sc->kernelStruct, &sc->tempInt);

				if (sc->numKernels.data.i > 1) {
					if (sc->conjugateConvolution == 1) {
						PfConjugate(sc, &sc->regIDs_copy[i + l * sc->registers_per_thread], &sc->regIDs_copy[i + l * sc->registers_per_thread]);
					}
					PfMul(sc, &sc->w, &sc->temp, &sc->regIDs_copy[i + l * sc->registers_per_thread], 0);
					if (sc->conjugateConvolution == 1) {
						PfConjugate(sc, &sc->regIDs_copy[i + l * sc->registers_per_thread], &sc->regIDs_copy[i + l * sc->registers_per_thread]);
					}
				}
				else {
					if (sc->conjugateConvolution == 1) {
						PfConjugate(sc, &sc->regIDs[i + l * sc->registers_per_thread], &sc->regIDs[i + l * sc->registers_per_thread]);
					}
					PfMul(sc, &sc->w, &sc->temp, &sc->regIDs[i + l * sc->registers_per_thread], 0);
					if (sc->conjugateConvolution == 1) {
						PfConjugate(sc, &sc->regIDs[i + l * sc->registers_per_thread], &sc->regIDs[i + l * sc->registers_per_thread]);
					}
				}
				PfAdd(sc, &sc->temp_conv[j], &sc->temp_conv[j], &sc->w);
			}
		}
		if (sc->crossPowerSpectrumNormalization) {
			PfNorm(sc, &sc->tempFloat, &sc->temp_conv[0]);
			PfRsqrt(sc, &sc->tempFloat, &sc->tempFloat);
			PfMul(sc, &sc->regIDs[i], &sc->temp_conv[0], &sc->tempFloat, 0);
		}
		else {
			PfMov(sc, &sc->regIDs[i], &sc->temp_conv[0]);
		}
		for (uint64_t l = 1; l < sc->matrixConvolution; l++) {
			if (sc->crossPowerSpectrumNormalization) {
				PfNorm(sc, &sc->tempFloat, &sc->temp_conv[l]);
				PfRsqrt(sc, &sc->tempFloat, &sc->tempFloat);
				PfMul(sc, &sc->regIDs[i + l * sc->registers_per_thread], &sc->temp_conv[l], &sc->tempFloat, 0);
			}
			else {
				PfMov(sc, &sc->regIDs[i + l * sc->registers_per_thread], &sc->temp_conv[l]);
			}
		}
		

		if(strideType == 0)
		{
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;

			temp_int1.data.i = sc->fftDim.data.i * batching_localSize.data.i;

			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_end(sc);
			}
			if (sc->fftDim.data.i == sc->fft_dim_full.data.i) {
				temp_int.data.i = batching_localSize.data.i;
				//we switched to reading 2x more data, but we still need to check out of bounds for odd size1

				if ((sc->size[1].data.i % temp_int.data.i) != 0) {
#if (VKFFT_BACKEND!=2) //AMD compiler fix
					PfIf_end(sc);
#endif
				}
			}
		}
		else {
			temp_int1.data.i = (i + 1) * sc->localSize[1].data.i;

			if (temp_int1.data.i > sc->fftDim.data.i) {
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(%s < %" PRIu64 "){\n", &sc->gl_LocalInvocationID_y, &sc->fftDim - (i + k * used_registers) * &sc->localSize[1]);
				temp_int1.data.i = sc->localSize[1].data.i - (temp_int1.data.i - sc->fftDim.data.i);
				PfIf_lt_start(sc, &sc->gl_LocalInvocationID_y, &temp_int1);
			}
		}
	}

	if (!((sc->fftDim.data.i == sc->fft_dim_full.data.i) && (strideType == 0))) {
		PfIf_end(sc);
	}
	if (sc->useDisableThreads) {
		PfIf_end(sc);
	}

	return;
}

#endif
