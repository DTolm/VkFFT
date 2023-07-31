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
#ifndef VKFFT_BLUESTEIN_H
#define VKFFT_BLUESTEIN_H
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_StringManagement/vkFFT_StringManager.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_MathUtils/vkFFT_MathUtils.h"

#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_Zeropad.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_KernelUtils.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryTransfers/vkFFT_Transfers.h"

static inline void appendBluesteinMultiplication(VkFFTSpecializationConstantsLayout* sc, uint64_t strideType, uint64_t pre_or_post_multiplication) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_int1 = VKFFT_ZERO_INIT;
	temp_int1.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 32;

	
	//char index_y[2000] = "";
	//char requestBatch[100] = "";
	//char separateRegisterStore[100] = "";
	if (!((sc->readToRegisters && (pre_or_post_multiplication == 0)) || (sc->writeFromRegisters && (pre_or_post_multiplication == 1)))) {
		appendBarrierVkFFT(sc);
		
	}
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}

	PfContainer used_registers = VKFFT_ZERO_INIT;
	used_registers.type = 31;

	PfContainer batching_localSize = VKFFT_ZERO_INIT;
	batching_localSize.type = 31;

	PfContainer localSize = VKFFT_ZERO_INIT;
	localSize.type = 31;

	PfContainer* localInvocationID = VKFFT_ZERO_INIT;
	
	PfContainer* batching_localInvocationID = VKFFT_ZERO_INIT;

	if (sc->stridedSharedLayout) {
		localSize.data.i = sc->localSize[1].data.i;
		batching_localSize.data.i = sc->localSize[0].data.i;
		localInvocationID = &sc->gl_LocalInvocationID_y;
		batching_localInvocationID = &sc->gl_LocalInvocationID_x;
	}
	else {
		localSize.data.i = sc->localSize[0].data.i;
		batching_localSize.data.i = sc->localSize[1].data.i;
		localInvocationID = &sc->gl_LocalInvocationID_x;
		batching_localInvocationID = &sc->gl_LocalInvocationID_y;
	}
		
	PfDivCeil(sc, &used_registers, &sc->fftDim, &localSize);

	for (uint64_t i = 0; i < (uint64_t)used_registers.data.i; i++) {
		if (localSize.data.i * ((1 + (int64_t)i)) > sc->fftDim.data.i) {
			PfContainer current_group_cut;
			current_group_cut.type = 31;
			current_group_cut .data.i = sc->fftDim.data.i - i * localSize.data.i;
			PfIf_lt_start(sc, localInvocationID, &current_group_cut);
		}

		if (sc->fftDim.data.i == sc->fft_dim_full.data.i) {
			switch (strideType) {
			case 0: case 2: case 5: case 6: case 110: case 120: case 130: case 140: case 142: case 144:
			{
				temp_int.data.i = i * sc->localSize[0].data.i;
				PfAdd(sc, &sc->inoutID, &sc->gl_LocalInvocationID_x, &temp_int);
				break;
			}
			case 1: case 111: case 121: case 131: case 141: case 143: case 145:
			{
				temp_int.data.i = i * sc->localSize[1].data.i;
				PfAdd(sc, &sc->inoutID, &sc->gl_LocalInvocationID_y, &temp_int);
				break;
			}
			}
		}
		else {
			switch (strideType) {
			case 0: case 2: case 5: case 6: case 110: case 120: case 130: case 140: case 142: case 144:
			{
				PfMod(sc, &sc->inoutID, &sc->shiftX, &sc->stageStartSize);

				temp_int.data.i = i * sc->localSize[1].data.i;
				PfAdd(sc, &sc->tempInt, &sc->gl_LocalInvocationID_y, &temp_int);
				PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->stageStartSize, 0);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);

				PfDiv(sc, &sc->tempInt, &sc->shiftX, &sc->stageStartSize);
				temp_int.data.i = sc->stageStartSize.data.i * sc->fftDim.data.i;
				PfMul(sc, &sc->tempInt, &sc->tempInt, &temp_int, 0);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
				break;
			}
			case 1: case 111: case 121: case 131: case 141: case 143: case 145:
			{
				temp_int.data.i = i * sc->localSize[1].data.i;
				PfAdd(sc, &sc->inoutID, &sc->gl_LocalInvocationID_y, &temp_int);
				PfMul(sc, &sc->inoutID, &sc->inoutID, &sc->stageStartSize, 0);

				PfDiv(sc, &sc->tempInt, &sc->shiftX, &sc->fft_dim_x);
				PfMod(sc, &sc->tempInt, &sc->tempInt, &sc->stageStartSize);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);

				temp_int.data.i = sc->stageStartSize.data.i * sc->fft_dim_x.data.i;
				PfDiv(sc, &sc->tempInt, &sc->shiftX, &temp_int);
				temp_int.data.i = sc->stageStartSize.data.i * sc->fftDim.data.i;
				PfMul(sc, &sc->tempInt, &sc->tempInt, &temp_int, 0);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
				break;
			}
			}
		}
		
		if ((sc->zeropadBluestein[0]) && (pre_or_post_multiplication == 0)) {
			PfMod(sc, &sc->tempInt, &sc->inoutID, &sc->fft_dim_full);
			PfIf_lt_start(sc, &sc->tempInt, &sc->fft_zeropad_Bluestein_left_read[sc->axis_id]);
		}
		if ((sc->zeropadBluestein[1]) && (pre_or_post_multiplication == 1)) {
			PfMod(sc, &sc->tempInt, &sc->inoutID, &sc->fft_dim_full);
			PfIf_lt_start(sc, &sc->tempInt, &sc->fft_zeropad_Bluestein_left_write[sc->axis_id]);		
		}

		appendGlobalToRegisters(sc, &sc->w, &sc->BluesteinStruct, &sc->inoutID);
		
		//uint64_t k = 0;
		if (!((sc->readToRegisters && (pre_or_post_multiplication == 0)) || (sc->writeFromRegisters && (pre_or_post_multiplication == 1)))) {
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
			appendSharedToRegisters(sc, &sc->regIDs[i], &sc->sdataID);
		}
		
		if (!sc->inverseBluestein)
			PfConjugate(sc, &sc->w, &sc->w);
		
		PfMul(sc, &sc->regIDs[i], &sc->regIDs[i], &sc->w, &sc->temp);

		if (!((sc->readToRegisters && (pre_or_post_multiplication == 0)) || (sc->writeFromRegisters && (pre_or_post_multiplication == 1)))) {
			appendRegistersToShared(sc, &sc->sdataID, &sc->regIDs[i]);
		}
		if ((sc->zeropadBluestein[0]) && (pre_or_post_multiplication == 0)) {
			PfIf_end(sc);
		}
		if ((sc->zeropadBluestein[1]) && (pre_or_post_multiplication == 1)) {
			PfIf_end(sc);
		}
		if (localSize.data.i * ((1 + (int64_t)i)) > sc->fftDim.data.i) {
			PfIf_end(sc);
		}
	}
	if (sc->useDisableThreads) {
		PfIf_end(sc);
	}
	return;
}

static inline void appendBluesteinConvolution(VkFFTSpecializationConstantsLayout* sc, uint64_t strideType) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_int1 = VKFFT_ZERO_INIT;
	temp_int1.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 32;
	
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
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

	for (uint64_t i = 0; i < (uint64_t)used_registers.data.i; i++) {
		if (localSize.data.i * ((1 + (int64_t)i)) > sc->fftDim.data.i) {
			temp_int.data.i = sc->fftDim.data.i - i * localSize.data.i;
			PfIf_lt_start(sc, localInvocationID, &temp_int);
		}
		if (sc->fftDim.data.i == sc->fft_dim_full.data.i) {
			switch (strideType) {
			case 0: case 2: case 5: case 6: case 110: case 120: case 130: case 140: case 142: case 144:
			{
				temp_int.data.i = i * sc->localSize[0].data.i;
				PfAdd(sc, &sc->inoutID, &sc->gl_LocalInvocationID_x, &temp_int);
				break;
			}
			case 1: case 111: case 121: case 131: case 141: case 143: case 145:
			{
				temp_int.data.i = i * sc->localSize[1].data.i;
				PfAdd(sc, &sc->inoutID, &sc->gl_LocalInvocationID_y, &temp_int);
				break;
			}
			}
		}
		else {
			switch (strideType) {
			case 0: case 2: case 5: case 6: case 110: case 120: case 130: case 140: case 142: case 144:
			{
				temp_int.data.i = sc->firstStageStartSize.data.i / sc->fftDim.data.i;
				PfMod(sc, &sc->inoutID, &sc->shiftX, &temp_int);
				PfMul(sc, &sc->inoutID, &sc->inoutID, &sc->fftDim, 0);

				PfDiv(sc, &sc->tempInt, &sc->shiftX, &temp_int);
				temp_int.data.i = sc->localSize[1].data.i * sc->firstStageStartSize.data.i;
				PfMul(sc, &sc->tempInt, &sc->tempInt, &temp_int, 0);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);

				temp_int.data.i = i * sc->localSize[0].data.i;
				PfAdd(sc, &sc->tempInt, &sc->gl_LocalInvocationID_x, &temp_int);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);

				PfMul(sc, &sc->tempInt, &sc->gl_LocalInvocationID_y, &sc->firstStageStartSize, 0);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
				break;
			}
			case 1: case 111: case 121: case 131: case 141: case 143: case 145:
			{
				temp_int.data.i = i * sc->localSize[1].data.i;
				PfAdd(sc, &sc->inoutID, &sc->gl_LocalInvocationID_y, &temp_int);
				PfMul(sc, &sc->inoutID, &sc->inoutID, &sc->stageStartSize, 0);

				PfDiv(sc, &sc->tempInt, &sc->shiftX, &sc->fft_dim_x);
				PfMod(sc, &sc->tempInt, &sc->tempInt, &sc->stageStartSize);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);

				temp_int.data.i = sc->stageStartSize.data.i * sc->fft_dim_x.data.i;
				PfDiv(sc, &sc->tempInt, &sc->shiftX, &sc->fft_dim_x);
				temp_int.data.i = sc->stageStartSize.data.i * sc->fftDim.data.i;
				PfMul(sc, &sc->tempInt, &sc->tempInt, &temp_int, 0);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
				break;
			}
			}
		}
		PfIf_lt_start(sc, &sc->inoutID, &sc->fft_dim_full);
		
		appendGlobalToRegisters(sc, &sc->w, &sc->BluesteinConvolutionKernelStruct, &sc->inoutID);

		if ((sc->inverseBluestein) && (sc->fftDim.data.i == sc->fft_dim_full.data.i))
			PfConjugate(sc, &sc->w, &sc->w);
		
		PfMul(sc, &sc->regIDs[i], &sc->regIDs[i], &sc->w, &sc->temp);
		
		PfIf_end(sc);
		if (localSize.data.i * ((1 + (int64_t)i)) > sc->fftDim.data.i) {
			PfIf_end(sc);
		}
	}
	if (sc->useDisableThreads) {
		PfIf_end(sc);
	}
	
	return;
}

#endif
