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
#ifndef VKFFT_R2R_H
#define VKFFT_R2R_H
#include "vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT_CodeGen/vkFFT_StringManagement/vkFFT_StringManager.h"
#include "vkFFT_CodeGen/vkFFT_MathUtils/vkFFT_MathUtils.h"

static inline void appendDCTI_read(VkFFTSpecializationConstantsLayout* sc, int type, int readWrite) {
	if (sc->res != VKFFT_SUCCESS) return;
	VkContainer temp_int = {};
	temp_int.type = 31;
	VkContainer temp_int1 = {};
	temp_int1.type = 31;
	VkContainer temp_double = {};
	temp_double.type = 32;

	VkContainer used_registers = {};
	used_registers.type = 31;
	
	VkContainer fftDim = {};
	fftDim.type = 31;

	VkContainer localSize = {};
	localSize.type = 31;

	VkContainer batching_localSize = {};
	batching_localSize.type = 31;

	VkContainer* localInvocationID = {0};
	VkContainer* batchingInvocationID = {0};

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

	if (sc->zeropadBluestein[readWrite]) {
		if (readWrite) {
			fftDim.data.i = (sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i + 2) / 2;
		}
		else {
			fftDim.data.i = (sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i + 2) / 2;
		}
	}
	else {
		fftDim.data.i = (sc->fftDim.data.i + 2) / 2;
	}

	if (sc->stridedSharedLayout) {
		VkDivCeil(sc, &used_registers, &fftDim, &sc->localSize[1]);
	}
	else {
		VkDivCeil(sc, &used_registers, &fftDim, &sc->localSize[0]);
	}

	appendBarrierVkFFT(sc);
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		VkIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	for (uint64_t i = 0; i < used_registers.data.i; i++) {
		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i)*sc->localSize[1].data.i;

			VkAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				VkIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		else {
			if (sc->localSize[1].data.i == 1) {
				temp_int.data.i = (i)*sc->localSize[0].data.i;

				VkAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
			}
			else {
				VkMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

				temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

				VkAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
				VkAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
			}
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				VkIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		if (sc->stridedSharedLayout) {
			temp_int.data.i = 0;
			VkIf_gt_start(sc, &sc->combinedID, &temp_int);
			temp_int.data.i = fftDim.data.i - 1;
			VkIf_lt_start(sc, &sc->combinedID, &temp_int);
		}else{
			VkMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			temp_int.data.i = 0;
			VkIf_gt_start(sc, &sc->tempInt, &temp_int);
			temp_int.data.i = fftDim.data.i - 1;
			VkIf_lt_start(sc, &sc->tempInt, &temp_int);
		}
		if (sc->stridedSharedLayout) {
			VkMul(sc, &sc->tempInt, &sc->combinedID, &sc->sharedStride, 0);

			temp_int.data.i = (2 * fftDim.data.i - 2) * sc->sharedStride.data.i;
			VkAdd(sc, &sc->inoutID, &sc->gl_LocalInvocationID_x, &temp_int);
			VkSub(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);

			VkAdd(sc, &sc->sdataID, &sc->gl_LocalInvocationID_x, &sc->tempInt);
		}
		else {
			VkDiv(sc, &sc->sdataID, &sc->combinedID, &fftDim);
			VkMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);

			temp_int.data.i = (2 * fftDim.data.i - 2);
			VkAdd(sc, &sc->inoutID, &sc->sdataID, &temp_int);
			VkSub(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);

			VkAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
		}
		appendSharedToRegisters(sc, &sc->temp, &sc->sdataID);
		appendRegistersToShared(sc, &sc->inoutID, &sc->temp);
		VkIf_end(sc);
		VkIf_end(sc);
		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				VkIf_end(sc);
			}
		}
		else {
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				VkIf_end(sc);
			}
		}
	}
	if (sc->useDisableThreads) {
		VkIf_end(sc);
	}
	return;
}

static inline void appendDCTII_read_III_write(VkFFTSpecializationConstantsLayout* sc, int type, int readWrite) {
	if (sc->res != VKFFT_SUCCESS) return;
	VkContainer temp_int = {};
	temp_int.type = 31;
	VkContainer temp_int1 = {};
	temp_int1.type = 31;
	VkContainer temp_double = {};
	temp_double.type = 32;

	VkContainer used_registers = {};
	used_registers.type = 31;

	VkContainer fftDim = {};
	fftDim.type = 31;

	VkContainer localSize = {};
	localSize.type = 31;

	VkContainer batching_localSize = {};
	batching_localSize.type = 31;

	VkContainer* localInvocationID = {0};
	VkContainer* batchingInvocationID = {0};

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

	if (sc->zeropadBluestein[readWrite]) {
		if (readWrite) {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i;
		}
		else {
			appendSetSMToZero(sc);
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i;
		}
	}
	else {
		fftDim.data.i = sc->fftDim.data.i;
	}

	if (sc->stridedSharedLayout) {
		VkDivCeil(sc, &used_registers, &fftDim, &sc->localSize[1]);
	}
	else {
		VkDivCeil(sc, &used_registers, &fftDim, &sc->localSize[0]);
	}

	appendBarrierVkFFT(sc);
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		VkIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	for (uint64_t i = 0; i < used_registers.data.i; i++) {
		if (sc->axis_id > 0) {
			temp_int.data.i = (i)*sc->localSize[1].data.i;

			VkAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				VkIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		else {
			if (sc->localSize[1].data.i == 1) {
				temp_int.data.i = (i)*sc->localSize[0].data.i;

				VkAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
			}
			else {
				VkMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

				temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

				VkAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
				VkAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
			}

			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				VkIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		if (sc->axis_id > 0){
			temp_int.data.i = 2;
			VkMod(sc, &sc->sdataID, &sc->combinedID, &temp_int);
		}
		else {
			VkMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			temp_int.data.i = 2;
			VkMod(sc, &sc->sdataID, &sc->tempInt, &temp_int);
		}
		VkMul(sc, &sc->blockInvocationID, &sc->sdataID, &temp_int, 0);
		temp_int.data.i = 2;
		if (sc->axis_id > 0) {
			VkDiv(sc, &sc->tempInt, &sc->combinedID, &temp_int);
		}
		else {
			VkDiv(sc, &sc->tempInt, &sc->tempInt, &temp_int);
		}
		VkMul(sc, &sc->blockInvocationID, &sc->blockInvocationID, &sc->tempInt, 0);
		
		temp_int.data.i = fftDim.data.i - 1;
		VkMul(sc, &sc->sdataID, &sc->sdataID, &temp_int, 0);
		VkAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
		VkSub(sc, &sc->sdataID, &sc->sdataID, &sc->blockInvocationID);

		if (sc->axis_id > 0) {
			VkMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
			VkAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);
		}
		else {
			if (sc->stridedSharedLayout) {
				VkMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
				VkDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
				VkAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
			}
			else {
				VkDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
				VkMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
				VkAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
			}
		}
		if(readWrite)
			appendSharedToRegisters(sc, &sc->regIDs[i], &sc->sdataID);
		else
			appendRegistersToShared(sc, &sc->sdataID, &sc->regIDs[i]);

		if (sc->axis_id > 0) {
			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				VkIf_end(sc);
			}
		}
		else {
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				VkIf_end(sc);
			}
		}
	}
	if (sc->useDisableThreads) {
		VkIf_end(sc);
	}
	if (readWrite)
		sc->writeFromRegisters = 1;
	else
		sc->readToRegisters = 0;
	return;
}
static inline void appendDCTII_write_III_read(VkFFTSpecializationConstantsLayout* sc, int type, int readWrite) {
	if (sc->res != VKFFT_SUCCESS) return;
	VkContainer temp_int = {};
	temp_int.type = 31;
	VkContainer temp_int1 = {};
	temp_int1.type = 31;
	VkContainer temp_double = {};
	temp_double.type = 32;

	VkContainer used_registers = {};
	used_registers.type = 31;

	VkContainer fftDim = {};
	fftDim.type = 31;

	VkContainer fftDim_half = {};
	fftDim_half.type = 31;

	VkContainer localSize = {};
	localSize.type = 31;

	VkContainer batching_localSize = {};
	batching_localSize.type = 31;

	VkContainer* localInvocationID = {0};
	VkContainer* batchingInvocationID = {0};

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

	if (sc->zeropadBluestein[readWrite]) {
		if (readWrite) {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i;
		}
		else {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i;
		}
	}
	else {
		fftDim.data.i = sc->fftDim.data.i;
	}

	temp_int.data.i = 2;
	VkDiv(sc, &fftDim_half, &fftDim, &temp_int);
	VkInc(sc, &fftDim_half);

	if (sc->stridedSharedLayout) {
		VkDivCeil(sc, &used_registers, &fftDim_half, &sc->localSize[1]);
	}
	else {
		VkDivCeil(sc, &used_registers, &fftDim_half, &sc->localSize[0]);
	}

	appendBarrierVkFFT(sc);
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		VkIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	for (uint64_t i = 0; i < used_registers.data.i; i++) {
		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i)*sc->localSize[1].data.i;

			VkAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim_half.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				VkIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		else {
			if (sc->localSize[1].data.i == 1) {
				temp_int.data.i = (i)*sc->localSize[0].data.i;

				VkAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
			}
			else {
				VkMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

				temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

				VkAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
				VkAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
			}
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim_half.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim_half * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim_half * &sc->localSize[0]);
				VkIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}

		if (sc->LUT) {
			if (sc->stridedSharedLayout) {
				VkAdd(sc, &sc->tempInt, &sc->combinedID, &sc->startDCT3LUT);
			}
			else {
				VkMod(sc, &sc->tempInt, &sc->combinedID, &fftDim_half);
				VkAdd(sc, &sc->tempInt, &sc->tempInt, &sc->startDCT3LUT);
			}
			appendGlobalToRegisters(sc, &sc->mult, &sc->LUTStruct, &sc->tempInt);
			if ((!sc->mergeSequencesR2C) && (readWrite)) {
				temp_double.data.d = 2.0l;
				VkMul(sc, &sc->mult, &sc->mult, &temp_double, 0);
			}
			if (readWrite)
				VkConjugate(sc, &sc->mult, &sc->mult);
		}
		else {
			if (readWrite)
				temp_double.data.d = -sc->double_PI / 2.0 / fftDim.data.i;
			else
				temp_double.data.d = sc->double_PI / 2.0 / fftDim.data.i;
			if (sc->stridedSharedLayout) {
				VkMul(sc, &sc->tempFloat, &sc->combinedID, &temp_double, 0);
			}
			else {
				VkMod(sc, &sc->tempInt, &sc->combinedID, &fftDim_half);
				VkMul(sc, &sc->tempFloat, &sc->tempInt, &temp_double, 0);
			}

			VkSinCos(sc, &sc->mult, &sc->tempFloat);
			if ((!sc->mergeSequencesR2C) && (readWrite)) {
				temp_double.data.d = 2.0l;
				VkMul(sc, &sc->mult, &sc->mult, &temp_double, 0);
			}
		}

		if (sc->stridedSharedLayout) {
			VkMul(sc, &sc->sdataID, &sc->combinedID, &sc->sharedStride, 0);

			temp_int.data.i = fftDim.data.i * sc->sharedStride.data.i;
			VkSub(sc, &sc->inoutID, &temp_int, &sc->sdataID);

			temp_int.data.i = 0;
			VkIf_eq_start(sc, &sc->sdataID, &temp_int);
			VkMov(sc, &sc->inoutID, &sc->sdataID);
			VkIf_end(sc);

			VkAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);
			VkAdd(sc, &sc->inoutID, &sc->inoutID, &sc->gl_LocalInvocationID_x);
		}
		else {
			VkMod(sc, &sc->sdataID, &sc->combinedID, &fftDim_half);
			if (sc->stridedSharedLayout) {
				VkMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);

				temp_int.data.i = fftDim.data.i * sc->sharedStride.data.i;
				VkSub(sc, &sc->inoutID, &temp_int, &sc->sdataID);

				temp_int.data.i = 0;
				VkIf_eq_start(sc, &sc->sdataID, &temp_int);
				VkMov(sc, &sc->inoutID, &sc->sdataID);
				VkIf_end(sc);

				VkDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim_half);
				VkAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
				VkAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
			}
			else {
				temp_int.data.i = fftDim.data.i;
				VkSub(sc, &sc->inoutID, &temp_int, &sc->sdataID);

				temp_int.data.i = 0;
				VkIf_eq_start(sc, &sc->sdataID, &temp_int);
				VkMov(sc, &sc->inoutID, &sc->sdataID);
				VkIf_end(sc);

				VkDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim_half);
				VkMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);

				VkAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
				VkAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
			}
		}

		if (readWrite) {
			appendSharedToRegisters(sc, &sc->temp, &sc->sdataID);
			if (sc->mergeSequencesR2C) {
				appendSharedToRegisters(sc, &sc->w, &sc->inoutID);

				VkAdd_x(sc, &sc->regIDs[0], &sc->temp, &sc->w);
				VkSub_y(sc, &sc->regIDs[0], &sc->temp, &sc->w);
				VkSub_x(sc, &sc->regIDs[1], &sc->w, &sc->temp);
				VkAdd_y(sc, &sc->regIDs[1], &sc->temp, &sc->w);

				VkMul(sc, &sc->temp, &sc->regIDs[0], &sc->mult, 0);
				VkConjugate(sc, &sc->mult, &sc->mult);
				VkMul(sc, &sc->w, &sc->regIDs[1], &sc->mult, 0);
				VkMov_x(sc, &sc->regIDs[0], &sc->temp);
				VkMov_y(sc, &sc->regIDs[0], &sc->w);
				VkMov_x_Neg_y(sc, &sc->regIDs[1], &sc->temp);
				VkMov_y_Neg_x(sc, &sc->regIDs[1], &sc->w);

				appendRegistersToShared(sc, &sc->inoutID, &sc->regIDs[1]);
				appendRegistersToShared(sc, &sc->sdataID, &sc->regIDs[0]);
			}
			else {
				VkMul(sc, &sc->regIDs[0], &sc->temp, &sc->mult, 0);
				VkMov_x_Neg_y(sc, &sc->w, &sc->regIDs[0]);

				appendRegistersToShared(sc, &sc->inoutID, &sc->w);
				appendRegistersToShared(sc, &sc->sdataID, &sc->regIDs[0]);
			}
		}
		else {
			appendSharedToRegisters(sc, &sc->temp, &sc->sdataID);
			appendSharedToRegisters(sc, &sc->w, &sc->inoutID);

			VkMod(sc, &sc->tempInt, &sc->combinedID, &fftDim_half);
			temp_int.data.i = 0;
			VkIf_eq_start(sc, &sc->tempInt, &temp_int);
			VkSetToZero(sc, &sc->w);
			VkIf_end(sc);

			VkMov_x_y(sc, &sc->regIDs[0], &sc->w);
			VkMov_y_x(sc, &sc->regIDs[0], &sc->w);

			VkSub_x(sc, &sc->regIDs[1], &sc->temp, &sc->regIDs[0]);
			VkAdd_y(sc, &sc->regIDs[1], &sc->temp, &sc->regIDs[0]);

			VkAdd_x(sc, &sc->w, &sc->temp, &sc->regIDs[0]);
			VkSub_y(sc, &sc->w, &sc->temp, &sc->regIDs[0]);

			VkMul(sc, &sc->regIDs[0], &sc->w, &sc->mult, 0);
			VkConjugate(sc, &sc->mult, &sc->mult);

			VkMul(sc, &sc->temp, &sc->regIDs[1], &sc->mult, 0);

			appendRegistersToShared(sc, &sc->inoutID, &sc->temp);
			appendRegistersToShared(sc, &sc->sdataID, &sc->regIDs[0]);
		}
		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim_half.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				VkIf_end(sc);
			}
		}
		else {
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim_half.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				VkIf_end(sc);
			}
		}
	}
	if (sc->useDisableThreads) {
		VkIf_end(sc);
	}
	return;
}

static inline void appendDCTIV_even_read(VkFFTSpecializationConstantsLayout* sc, int type, int readWrite) {
	if (sc->res != VKFFT_SUCCESS) return;
	VkContainer temp_int = {};
	temp_int.type = 31;
	VkContainer temp_int1 = {};
	temp_int1.type = 31;
	VkContainer temp_double = {};
	temp_double.type = 32;

	VkContainer used_registers = {};
	used_registers.type = 31;

	VkContainer fftDim = {};
	fftDim.type = 31;
	VkContainer fftDim_half = {};
	fftDim_half.type = 31;
	VkContainer localSize = {};
	localSize.type = 31;

	VkContainer batching_localSize = {};
	batching_localSize.type = 31;

	VkContainer* localInvocationID = {0};
	VkContainer* batchingInvocationID = {0};

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

	if (sc->zeropadBluestein[readWrite]) {
		if (readWrite) {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i;
		}
		else {
			if (sc->readToRegisters == 1) {
				appendSetSMToZero(sc);
				appendBarrierVkFFT(sc);
			}
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i;
		}
	}
	else
		fftDim.data.i = sc->fftDim.data.i;

	fftDim.data.i = 2 * fftDim.data.i;	

	if (sc->stridedSharedLayout) {
		VkDivCeil(sc, &used_registers, &fftDim, &sc->localSize[1]);
	}
	else {
		VkDivCeil(sc, &used_registers, &fftDim, &sc->localSize[0]);
	}
	if (sc->readToRegisters == 1) {
		if (sc->useDisableThreads) {
			temp_int.data.i = 0;
			VkIf_gt_start(sc, &sc->disableThreads, &temp_int);
		}
		for (uint64_t i = 0; i < used_registers.data.i; i++) {
			if (sc->axis_id > 0) {
				temp_int.data.i = (i)*sc->localSize[1].data.i;

				VkAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

				temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					//check that we only read fftDim * local batch data
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
					VkIf_lt_start(sc, &sc->combinedID, &temp_int1);
				}
			}
			else {
				if (sc->localSize[1].data.i == 1) {
					temp_int.data.i = (i)*sc->localSize[0].data.i;

					VkAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
				}
				else {
					VkMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

					temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

					VkAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
					VkAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
				}
				temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					//check that we only read fftDim * local batch data
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
					VkIf_lt_start(sc, &sc->combinedID, &temp_int1);
				}
			}
			if (sc->axis_id > 0) {
				temp_int1.data.i = 2;
				VkDiv(sc, &sc->sdataID, &sc->combinedID, &temp_int1);

				VkMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
				VkAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);
			}
			else {
				if (sc->stridedSharedLayout) {
					temp_int.data.i = fftDim.data.i;
					VkMod(sc, &sc->sdataID, &sc->combinedID, &temp_int);

					temp_int1.data.i = 2;
					VkDiv(sc, &sc->sdataID, &sc->sdataID, &temp_int1);

					VkMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
					VkDiv(sc, &sc->tempInt, &sc->combinedID, &temp_int);
					VkAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
				}
				else {
					temp_int.data.i = fftDim.data.i;
					VkMod(sc, &sc->sdataID, &sc->combinedID, &temp_int);

					temp_int1.data.i = 2;
					VkDiv(sc, &sc->sdataID, &sc->sdataID, &temp_int1);

					VkDiv(sc, &sc->tempInt, &sc->combinedID, &temp_int);
					VkMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
					VkAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
				}
			}
			if (sc->axis_id > 0) {
				temp_int.data.i = 2;
				VkMod(sc, &sc->tempInt, &sc->combinedID, &temp_int);
			}
			else {
				VkMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
				temp_int.data.i = 2;
				VkMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			}
			temp_int.data.i = 0;
			VkIf_eq_start(sc, &sc->tempInt, &temp_int);
			if (i < used_registers.data.i / 2) {
				appendRegistersToShared_x_x(sc, &sc->sdataID, &sc->regIDs[i]);
			}
			else {
				appendRegistersToShared_x_y(sc, &sc->sdataID, &sc->regIDs[i - used_registers.data.i / 2]);
			}
#if(!((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5)))
			VkIf_else(sc);
			if (i < used_registers.data.i / 2) {
				appendRegistersToShared_y_x(sc, &sc->sdataID, &sc->regIDs[i]);
			}
			else {
				appendRegistersToShared_y_y(sc, &sc->sdataID, &sc->regIDs[i - used_registers.data.i / 2]);
			}
#endif
			VkIf_end(sc);
			if (sc->axis_id > 0) {
				temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					VkIf_end(sc);
				}
			}
			else {
				temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					VkIf_end(sc);
				}
			}
		}
		if (sc->useDisableThreads) {
			VkIf_end(sc);
		}
#if(((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5)))
		appendBarrierVkFFT(sc);
		if (sc->useDisableThreads) {
			temp_int.data.i = 0;
			VkIf_gt_start(sc, &sc->disableThreads, &temp_int);
		}
		for (uint64_t i = 0; i < used_registers.data.i; i++) {
			if (sc->axis_id > 0) {
				temp_int.data.i = (i)*sc->localSize[1].data.i;

				VkAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

				temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					//check that we only read fftDim * local batch data
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
					VkIf_lt_start(sc, &sc->combinedID, &temp_int1);
				}
			}
			else {
				if (sc->localSize[1].data.i == 1) {
					temp_int.data.i = (i)*sc->localSize[0].data.i;

					VkAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
				}
				else {
					VkMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

					temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

					VkAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
					VkAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
				}
				temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					//check that we only read fftDim * local batch data
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
					VkIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
			}
			if (sc->axis_id > 0) {
				temp_int1.data.i = 2;
				VkDiv(sc, &sc->sdataID, &sc->combinedID, &temp_int1);

				VkMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
				VkAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);
			}
			else {
				if (sc->stridedSharedLayout) {
					temp_int.data.i = fftDim.data.i;
					VkMod(sc, &sc->sdataID, &sc->combinedID, &temp_int);

					temp_int1.data.i = 2;
					VkDiv(sc, &sc->sdataID, &sc->sdataID, &temp_int1);

					VkMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
					VkDiv(sc, &sc->tempInt, &sc->combinedID, &temp_int);
					VkAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
				}
				else {
					temp_int.data.i = fftDim.data.i;
					VkMod(sc, &sc->sdataID, &sc->combinedID, &temp_int);

					temp_int1.data.i = 2;
					VkDiv(sc, &sc->sdataID, &sc->sdataID, &temp_int1);

					VkDiv(sc, &sc->tempInt, &sc->combinedID, &temp_int);
					VkMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
					VkAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
				}
			}
			if (sc->axis_id > 0) {
				temp_int.data.i = 2;
				VkMod(sc, &sc->tempInt, &sc->combinedID, &temp_int);
			}
			else {
				VkMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
				temp_int.data.i = 2;
				VkMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			}
			temp_int.data.i = 1;
			VkIf_eq_start(sc, &sc->tempInt, &temp_int);
			if (i < used_registers.data.i / 2) {
				appendRegistersToShared_y_x(sc, &sc->sdataID, &sc->regIDs[i]);
			}
			else {
				appendRegistersToShared_y_y(sc, &sc->sdataID, &sc->regIDs[i - used_registers.data.i / 2]);
			}
			VkIf_end(sc);

			if (sc->axis_id > 0) {
				temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					VkIf_end(sc);
				}
			}
			else {
				temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					VkIf_end(sc);
				}
			}
		}
		if (sc->useDisableThreads) {
			VkIf_end(sc);
		}
#endif
	}
	appendBarrierVkFFT(sc);
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		VkIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	fftDim.data.i = fftDim.data.i / 2;

	if (sc->stridedSharedLayout) {
		VkDivCeil(sc, &used_registers, &fftDim, &sc->localSize[1]);
	}
	else {
		VkDivCeil(sc, &used_registers, &fftDim, &sc->localSize[0]);
	}
	for (uint64_t i = 0; i < used_registers.data.i; i++) {
		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i)*sc->localSize[1].data.i;

			VkAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				VkIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		else {
			if (sc->localSize[1].data.i == 1) {
				temp_int.data.i = (i)*sc->localSize[0].data.i;

				VkAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
			}
			else {
				VkMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

				temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

				VkAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
				VkAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
			}
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				VkIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}

		if (sc->stridedSharedLayout) {
			VkMul(sc, &sc->sdataID, &sc->combinedID, &sc->sharedStride, 0);
			VkAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);

			temp_int.data.i = 0;
			VkIf_gt_start(sc, &sc->combinedID, &temp_int);
		}
		else {
			VkMod(sc, &sc->sdataID, &sc->combinedID, &fftDim);
			VkDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			VkMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
			VkAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);

			VkMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			temp_int.data.i = 0;
			VkIf_gt_start(sc, &sc->tempInt, &temp_int);
		}

		if (sc->stridedSharedLayout) {
			VkSub(sc, &sc->tempInt, &sc->sdataID, &sc->sharedStride);
			appendSharedToRegisters_y_y(sc, &sc->w, &sc->tempInt);
		}
		else {
			temp_int.data.i = 1;
			VkSub(sc, &sc->tempInt, &sc->sdataID, &temp_int);
			appendSharedToRegisters_y_y(sc, &sc->w, &sc->tempInt);
		}

		appendSharedToRegisters_x_x(sc, &sc->w, &sc->sdataID);
		
		VkMov_x_y(sc, &sc->regIDs[i], &sc->w);
		VkMov_y_Neg_x(sc, &sc->regIDs[i], &sc->w);
		VkAdd(sc, &sc->regIDs[i], &sc->regIDs[i], &sc->w);
		
		VkIf_else(sc);

		appendSharedToRegisters_x_x(sc, &sc->regIDs[i], &sc->sdataID);
		if (sc->stridedSharedLayout) {
			temp_int.data.i = (fftDim.data.i - 1) * sc->sharedStride.data.i;
			VkAdd(sc, &sc->sdataID, &sc->sdataID, &temp_int);
		}
		else {
			temp_int.data.i = (fftDim.data.i - 1);
			VkAdd(sc, &sc->sdataID, &sc->sdataID, &temp_int);
		}
		appendSharedToRegisters_y_y(sc, &sc->regIDs[i], &sc->sdataID);
		temp_double.data.d = 2.0l;
		VkMul(sc, &sc->regIDs[i], &sc->regIDs[i], &temp_double, 0);
		
		VkIf_end(sc);
		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				VkIf_end(sc);
			}
		}
		else {
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				VkIf_end(sc);
			}
		}
	}
	if (sc->useDisableThreads) {
		VkIf_end(sc);
	}
	appendBarrierVkFFT(sc);
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		VkIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	for (uint64_t i = 0; i < used_registers.data.i; i++) {
		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i)*sc->localSize[1].data.i;

			VkAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				VkIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		else {
			if (sc->localSize[1].data.i == 1) {
				temp_int.data.i = (i)*sc->localSize[0].data.i;

				VkAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
			}
			else {
				VkMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

				temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

				VkAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
				VkAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
			}

			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				VkIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		if (sc->stridedSharedLayout) {
			VkMul(sc, &sc->sdataID, &sc->combinedID, &sc->sharedStride, 0);
			VkAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);

			temp_int.data.i = 0;
			VkIf_gt_start(sc, &sc->combinedID, &temp_int);
		}
		else {
			VkMod(sc, &sc->sdataID, &sc->combinedID, &fftDim);
			VkDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			VkMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
			VkAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);

			VkMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			temp_int.data.i = 0;
			VkIf_gt_start(sc, &sc->tempInt, &temp_int);
		}

		appendRegistersToShared_x_x(sc, &sc->sdataID, &sc->regIDs[i]);

#if(!((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5)))//OpenCL, Level Zero and Metal are  not handling barrier with thread-conditional writes to local memory - so this is a work-around
		if (sc->stridedSharedLayout) {
			VkSub(sc, &sc->sdataID, &fftDim, &sc->combinedID);

			VkMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
			VkAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);
		}
		else {
			VkMod(sc, &sc->sdataID, &sc->combinedID, &fftDim);
			VkSub(sc, &sc->sdataID, &fftDim, &sc->sdataID);

			VkDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			VkMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
			VkAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
		}

		appendRegistersToShared_y_y(sc, &sc->sdataID, &sc->regIDs[i]);
#endif
		
		VkIf_else(sc);

		appendRegistersToShared(sc, &sc->sdataID, &sc->regIDs[i]);

		VkIf_end(sc);

		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				VkIf_end(sc);
			}
		}
		else {
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				VkIf_end(sc);
			}
		}
	}
	if (sc->useDisableThreads) {
		VkIf_end(sc);
	}
#if(((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5)))//OpenCL, Level Zero and Metal are  not handling barrier with thread-conditional writes to local memory - so this is a work-around

	appendBarrierVkFFT(sc);
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		VkIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	for (uint64_t i = 0; i < used_registers.data.i; i++) {
		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i)*sc->localSize[1].data.i;

			VkAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				VkIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		else {
			if (sc->localSize[1].data.i == 1) {
				temp_int.data.i = (i)*sc->localSize[0].data.i;

				VkAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
			}
			else {
				VkMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

				temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

				VkAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
				VkAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
			}

			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				VkIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}

		if (sc->stridedSharedLayout) {
			VkSub(sc, &sc->sdataID, &fftDim, &sc->combinedID);

			VkMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
			VkAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);

			temp_int.data.i = 0;
			VkIf_gt_start(sc, &sc->combinedID, &temp_int);
		}
		else {
			VkMod(sc, &sc->sdataID, &sc->combinedID, &fftDim);
			VkSub(sc, &sc->sdataID, &fftDim, &sc->sdataID);

			VkDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			VkMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
			VkAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
			
			VkMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			temp_int.data.i = 0;
			VkIf_gt_start(sc, &sc->tempInt, &temp_int);
		}

		appendRegistersToShared_y_y(sc, &sc->sdataID, &sc->regIDs[i]);

		VkIf_end(sc);

		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				VkIf_end(sc);
			}
		}
		else {
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				VkIf_end(sc);
			}
		}
	}
	if (sc->useDisableThreads) {
		VkIf_end(sc);
	}
#endif

	appendDCTII_write_III_read(sc, type, 0);

	sc->readToRegisters = 0;

	return;
}
static inline void appendDCTIV_even_write(VkFFTSpecializationConstantsLayout* sc, int type, int readWrite) {
	if (sc->res != VKFFT_SUCCESS) return;
	VkContainer temp_int = {};
	temp_int.type = 31;
	VkContainer temp_int1 = {};
	temp_int1.type = 31;
	VkContainer temp_int2 = {};
	temp_int2.type = 31;
	VkContainer temp_double = {};
	temp_double.type = 32;

	VkContainer used_registers = {};
	used_registers.type = 31;

	VkContainer fftDim = {};
	fftDim.type = 31;

	VkContainer localSize = {};
	localSize.type = 31;

	VkContainer batching_localSize = {};
	batching_localSize.type = 31;

	VkContainer* localInvocationID = {0};
	VkContainer* batchingInvocationID = {0};

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

	if (sc->zeropadBluestein[readWrite]) {
		fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i;
	}
	else {
		fftDim.data.i = sc->fftDim.data.i;
	}

	if (sc->stridedSharedLayout) {
		VkDivCeil(sc, &used_registers, &fftDim, &sc->localSize[1]);
	}
	else {
		VkDivCeil(sc, &used_registers, &fftDim, &sc->localSize[0]);
	}

	appendDCTII_read_III_write(sc, type, 1);

	appendBarrierVkFFT(sc);
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		VkIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	for (uint64_t i = 0; i < used_registers.data.i; i++) {
		if (sc->axis_id > 0) {
			temp_int.data.i = (i)*sc->localSize[1].data.i;

			VkAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				VkIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		else {
			if (sc->localSize[1].data.i == 1) {
				temp_int.data.i = (i)*sc->localSize[0].data.i;

				VkAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
			}
			else {
				VkMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

				temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

				VkAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
				VkAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
			}

			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				VkIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		if (sc->axis_id > 0) {
			temp_int.data.i = 2;
			VkMod(sc, &sc->sdataID, &sc->combinedID, &temp_int);
		}
		else {
			VkMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			temp_int.data.i = 2;
			VkMod(sc, &sc->sdataID, &sc->tempInt, &temp_int);
		}
		VkMul(sc, &sc->blockInvocationID, &sc->sdataID, &temp_int, 0);
		temp_double.data.d = 1;
		VkSub(sc, &sc->tempFloat, &temp_double, &sc->blockInvocationID);
		VkMul_y(sc, &sc->regIDs[i], &sc->regIDs[i], &sc->tempFloat, 0);

		if (sc->LUT) {
			if (sc->axis_id > 0) {
				VkAdd(sc, &sc->tempInt, &sc->combinedID, &sc->startDCT4LUT);
			}
			else {
				VkAdd(sc, &sc->tempInt, &sc->tempInt, &sc->startDCT4LUT);
			}
			appendGlobalToRegisters(sc, &sc->mult, &sc->LUTStruct, &sc->tempInt);
		}
		else {
			temp_int.data.i = 2;
			if (sc->axis_id > 0) {
				VkMul(sc, &sc->tempInt, &sc->combinedID, &temp_int, 0);
			}
			else {
				VkMul(sc, &sc->tempInt, &sc->tempInt, &temp_int, 0);
			}
			VkInc(sc, &sc->tempInt);
			if (readWrite)
				temp_double.data.d = -sc->double_PI / 8.0 / fftDim.data.i;
			else
				temp_double.data.d = sc->double_PI / 8.0 / fftDim.data.i;
			VkMul(sc, &sc->tempFloat, &sc->tempInt, &temp_double, 0);

			VkSinCos(sc, &sc->mult, &sc->tempFloat);
		}

		VkMul(sc, &sc->regIDs[i], &sc->regIDs[i], &sc->mult, &sc->temp);
		VkConjugate(sc, &sc->regIDs[i], &sc->regIDs[i]);

		if (sc->axis_id > 0) {
			VkMul(sc, &sc->tempInt, &sc->combinedID, &sc->sharedStride, 0);

			temp_int.data.i = (fftDim.data.i - 1) * sc->sharedStride.data.i;
			VkAdd(sc, &sc->sdataID, &sc->gl_LocalInvocationID_x, &temp_int);
			VkSub(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
		}
		else {
			VkMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			if (sc->stridedSharedLayout) {
				temp_int.data.i = (fftDim.data.i-1);
				VkSub(sc, &sc->sdataID, &temp_int, &sc->tempInt);
				VkMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
				
				VkDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
				VkAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
			}
			else {
				VkDiv(sc, &sc->sdataID, &sc->combinedID, &fftDim);
				VkMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);

				temp_int.data.i = (fftDim.data.i-1);
				VkAdd(sc, &sc->sdataID, &sc->sdataID, &temp_int);
				VkSub(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
			}
		}
		appendRegistersToShared_y_y(sc, &sc->sdataID, &sc->regIDs[i]);
		if (sc->axis_id > 0) {
			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				VkIf_end(sc);
			}
		}
		else {
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				VkIf_end(sc);
			}
		}
	}
	if (sc->useDisableThreads) {
		VkIf_end(sc);
	}
	appendBarrierVkFFT(sc);
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		VkIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	for (uint64_t i = 0; i < used_registers.data.i; i++) {
		if (sc->axis_id > 0) {
			temp_int.data.i = (i)*sc->localSize[1].data.i;

			VkAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				VkIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		else {
			if (sc->localSize[1].data.i == 1) {
				temp_int.data.i = (i)*sc->localSize[0].data.i;

				VkAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
			}
			else {
				VkMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

				temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

				VkAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
				VkAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
			}

			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				VkIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}

		if (sc->axis_id > 0) {
			VkMul(sc, &sc->tempInt, &sc->combinedID, &sc->sharedStride, 0);

			VkAdd(sc, &sc->sdataID, &sc->gl_LocalInvocationID_x, &sc->tempInt);
		}
		else {
			VkMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			if (sc->stridedSharedLayout) {
				VkMul(sc, &sc->sdataID, &sc->tempInt, &sc->sharedStride, 0);

				VkDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
				VkAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
			}
			else {
				VkDiv(sc, &sc->sdataID, &sc->combinedID, &fftDim);
				VkMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);

				VkAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
			}
		}
		appendSharedToRegisters_y_y(sc, &sc->regIDs[i], &sc->sdataID);
		if (sc->axis_id > 0) {
			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				VkIf_end(sc);
			}
		}
		else {
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				VkIf_end(sc);
			}
		}
	}
	if (sc->useDisableThreads) {
		VkIf_end(sc);
	}
	return;
}

static inline void appendDCTIV_odd_read(VkFFTSpecializationConstantsLayout* sc, int type, int readWrite) {
	if (sc->res != VKFFT_SUCCESS) return;
	VkContainer temp_int = {};
	temp_int.type = 31;
	VkContainer temp_int1 = {};
	temp_int1.type = 31;
	VkContainer temp_double = {};
	temp_double.type = 32;

	VkContainer used_registers = {};
	used_registers.type = 31;

	VkContainer fftDim = {};
	fftDim.type = 31;

	VkContainer localSize = {};
	localSize.type = 31;

	VkContainer batching_localSize = {};
	batching_localSize.type = 31;

	VkContainer* localInvocationID = {0};
	VkContainer* batchingInvocationID = {0};

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

	if (sc->zeropadBluestein[readWrite]) {
		if (readWrite) {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i;
		}
		else {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i;
		}
	}
	else {
		fftDim.data.i = sc->fftDim.data.i;
	}

	if (sc->stridedSharedLayout) {
		VkDivCeil(sc, &used_registers, &fftDim, &sc->localSize[1]);
	}
	else {
		VkDivCeil(sc, &used_registers, &fftDim, &sc->localSize[0]);
	}

	appendBarrierVkFFT(sc);
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		VkIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	for (uint64_t i = 0; i < used_registers.data.i; i++) {
		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i)*sc->localSize[1].data.i;

			VkAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				VkIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		else {
			if (sc->localSize[1].data.i == 1) {
				temp_int.data.i = (i)*sc->localSize[0].data.i;

				VkAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
			}
			else {
				VkMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

				temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

				VkAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
				VkAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
			}

			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				VkIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		if (sc->stridedSharedLayout) {
			temp_int.data.i = 4;
			VkMul(sc, &sc->inoutID, &sc->combinedID, &temp_int,0);
		}
		else {
			VkMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			temp_int.data.i = 4;
			VkMul(sc, &sc->inoutID, &sc->tempInt, &temp_int,0);
		}
		temp_int.data.i = fftDim.data.i / 2;
		VkAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int);
		
		VkIf_lt_start(sc, &sc->inoutID, &fftDim);
		VkMov(sc, &sc->sdataID, &sc->inoutID);
		VkIf_end(sc);
		
		temp_int.data.i = fftDim.data.i * 2;
		VkIf_lt_start(sc, &sc->inoutID, &temp_int);
		VkIf_ge_start(sc, &sc->inoutID, &fftDim);
		temp_int.data.i = fftDim.data.i * 2 - 1;
		VkSub(sc, &sc->sdataID, &temp_int, &sc->inoutID);
		VkIf_end(sc);
		VkIf_end(sc);

		temp_int.data.i = fftDim.data.i * 3;
		VkIf_lt_start(sc, &sc->inoutID, &temp_int);
		temp_int.data.i = fftDim.data.i * 2;
		VkIf_ge_start(sc, &sc->inoutID, &temp_int);
		temp_int.data.i = fftDim.data.i * 2;
		VkSub(sc, &sc->sdataID, &sc->inoutID, &temp_int);
		VkIf_end(sc);
		VkIf_end(sc);

		temp_int.data.i = fftDim.data.i * 4;
		VkIf_lt_start(sc, &sc->inoutID, &temp_int);
		temp_int.data.i = fftDim.data.i * 3;
		VkIf_ge_start(sc, &sc->inoutID, &temp_int);
		temp_int.data.i = fftDim.data.i * 4 - 1;
		VkSub(sc, &sc->sdataID, &temp_int, &sc->inoutID);
		VkIf_end(sc);
		VkIf_end(sc);

		temp_int.data.i = fftDim.data.i * 4;
		VkIf_ge_start(sc, &sc->inoutID, &temp_int);
		temp_int.data.i = fftDim.data.i * 4;
		VkSub(sc, &sc->sdataID, &sc->inoutID, &temp_int);
		VkIf_end(sc);
		
		if (sc->stridedSharedLayout) {
			VkMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
			VkAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);
		}
		else {
			VkDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			VkMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
			VkAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
		}
		appendSharedToRegisters(sc, &sc->regIDs[i], &sc->sdataID);

		temp_int.data.i = fftDim.data.i * 2;
		VkIf_lt_start(sc, &sc->inoutID, &temp_int);
		VkIf_ge_start(sc, &sc->inoutID, &fftDim);
		VkMov_x_Neg_x(sc, &sc->regIDs[i], &sc->regIDs[i]);
		VkMov_y_Neg_y(sc, &sc->regIDs[i], &sc->regIDs[i]);
		VkIf_end(sc);
		VkIf_end(sc);

		temp_int.data.i = fftDim.data.i * 3;
		VkIf_lt_start(sc, &sc->inoutID, &temp_int);
		temp_int.data.i = fftDim.data.i * 2;
		VkIf_ge_start(sc, &sc->inoutID, &temp_int);
		VkMov_x_Neg_x(sc, &sc->regIDs[i], &sc->regIDs[i]);
		VkMov_y_Neg_y(sc, &sc->regIDs[i], &sc->regIDs[i]);
		VkIf_end(sc);
		VkIf_end(sc);

		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				VkIf_end(sc);
			}
		}
		else {
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				VkIf_end(sc);
			}
		}
	}
	if (sc->useDisableThreads) {
		VkIf_end(sc);
	}
	int64_t registers_first_stage = (sc->stageRadix[0] < sc->fixMinRaderPrimeMult) ? sc->registers_per_thread_per_radix[sc->stageRadix[0]] : 1;
	if ((sc->rader_generator[0] > 0) || ((sc->fftDim.data.i / registers_first_stage) != localSize.data.i))
		sc->readToRegisters = 0;
	else
		sc->readToRegisters = 0; // can be switched to 1 if the indexing in previous step is aligned to 1 stage of fft (here it is combined)

	if (!sc->readToRegisters) {

		appendBarrierVkFFT(sc);
		if (sc->useDisableThreads) {
			temp_int.data.i = 0;
			VkIf_gt_start(sc, &sc->disableThreads, &temp_int);
		}
		for (uint64_t i = 0; i < used_registers.data.i; i++) {
			if (sc->stridedSharedLayout) {
				temp_int.data.i = (i)*sc->localSize[1].data.i;

				VkAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

				temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					//check that we only read fftDim * local batch data
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
					VkIf_lt_start(sc, &sc->combinedID, &temp_int1);
				}
			}
			else {
				if (sc->localSize[1].data.i == 1) {
					temp_int.data.i = (i)*sc->localSize[0].data.i;

					VkAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
				}
				else {
					VkMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

					temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

					VkAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
					VkAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
				}

				temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					//check that we only read fftDim * local batch data
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
					VkIf_lt_start(sc, &sc->combinedID, &temp_int1);
				}
			}
			
			if (sc->stridedSharedLayout) {
				VkMul(sc, &sc->tempInt, &sc->combinedID, &sc->sharedStride, 0);
				VkAdd(sc, &sc->sdataID, &sc->gl_LocalInvocationID_x, &sc->tempInt);
			}
			else {
				VkDiv(sc, &sc->sdataID, &sc->combinedID, &fftDim);
				VkMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);

				VkMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
				VkAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
			}

			appendRegistersToShared(sc, &sc->sdataID, &sc->regIDs[i]);

			if (sc->stridedSharedLayout) {
				temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					VkIf_end(sc);
				}
			}
			else {
				temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					VkIf_end(sc);
				}
			}
		}
		if (sc->useDisableThreads) {
			VkIf_end(sc);
		}
	}
	return;
}

static inline void appendDCTIV_odd_write(VkFFTSpecializationConstantsLayout* sc, int type, int readWrite) {
	if (sc->res != VKFFT_SUCCESS) return;
	VkContainer temp_int = {};
	temp_int.type = 31;
	VkContainer temp_int1 = {};
	temp_int1.type = 31;
	VkContainer temp_double = {};
	temp_double.type = 32;

	VkContainer used_registers = {};
	used_registers.type = 31;

	VkContainer fftDim = {};
	fftDim.type = 31;

	VkContainer localSize = {};
	localSize.type = 31;

	VkContainer batching_localSize = {};
	batching_localSize.type = 31;

	VkContainer* localInvocationID = {0};
	VkContainer* batchingInvocationID = {0};

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

	if (sc->zeropadBluestein[readWrite]) {
		if (readWrite) {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i;
		}
		else {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i;
		}
	}
	else {
		fftDim.data.i = sc->fftDim.data.i;
	}

	if (sc->stridedSharedLayout) {
		VkDivCeil(sc, &used_registers, &fftDim, &sc->localSize[1]);
	}
	else {
		VkDivCeil(sc, &used_registers, &fftDim, &sc->localSize[0]);
	}

	appendBarrierVkFFT(sc);
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		VkIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	for (uint64_t i = 0; i < used_registers.data.i; i++) {
		if (sc->axis_id > 0) {
			temp_int.data.i = (i)*sc->localSize[1].data.i;

			VkAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				VkIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		else {
			if (sc->localSize[1].data.i == 1) {
				temp_int.data.i = (i)*sc->localSize[0].data.i;

				VkAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
			}
			else {
				VkMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

				temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

				VkAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
				VkAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
			}

			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				VkIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		if (sc->axis_id > 0) {
			VkMov(sc, &sc->sdataID, &sc->combinedID);
		}
		else {
			VkMod(sc, &sc->sdataID, &sc->combinedID, &fftDim);
		}

		temp_int.data.i = fftDim.data.i / 4;
		VkIf_lt_start(sc, &sc->sdataID, &temp_int);

			temp_int.data.i = 2;
			VkMul(sc, &sc->inoutID, &sc->sdataID, &temp_int, 0);
			VkInc(sc, &sc->inoutID);
			if (sc->mergeSequencesR2C) {
				VkSub(sc, &sc->tempInt, &fftDim, &sc->inoutID);
				VkIf_eq_start(sc, &sc->tempInt, &fftDim);
				VkSetToZero(sc, &sc->tempInt);
				VkIf_end(sc);
			}

			VkIf_eq_start(sc, &sc->inoutID, &fftDim);
			VkSetToZero(sc, &sc->inoutID);
			VkIf_end(sc);

		VkIf_end(sc);

		temp_int.data.i = fftDim.data.i / 2;
		VkIf_lt_start(sc, &sc->sdataID, &temp_int);
		temp_int.data.i = fftDim.data.i / 4;
		VkIf_ge_start(sc, &sc->sdataID, &temp_int);

			temp_int.data.i = 2;
			VkMul(sc, &sc->inoutID, &sc->sdataID, &temp_int, 0);
			if (sc->mergeSequencesR2C) {
				temp_int.data.i = fftDim.data.i - 2 * (fftDim.data.i / 2);
				VkAdd(sc, &sc->tempInt, &temp_int, &sc->inoutID);

				VkIf_eq_start(sc, &sc->tempInt, &fftDim);
				VkSetToZero(sc, &sc->tempInt);
				VkIf_end(sc);
			}
			temp_int.data.i = 2 * (fftDim.data.i / 2);
			VkSub(sc, &sc->inoutID, &temp_int, &sc->inoutID);

			VkIf_eq_start(sc, &sc->inoutID, &fftDim);
			VkSetToZero(sc, &sc->inoutID);
			VkIf_end(sc);

		VkIf_end(sc);
		VkIf_end(sc);

		temp_int.data.i = 3 * fftDim.data.i / 4;
		VkIf_lt_start(sc, &sc->sdataID, &temp_int);
		temp_int.data.i = fftDim.data.i / 2;
		VkIf_ge_start(sc, &sc->sdataID, &temp_int);

			temp_int.data.i = 2;
			VkMul(sc, &sc->inoutID, &sc->sdataID, &temp_int, 0);
			if (sc->mergeSequencesR2C) {
				temp_int.data.i = fftDim.data.i + 2 * (fftDim.data.i / 2);
				VkSub(sc, &sc->tempInt, &temp_int, &sc->inoutID);
				VkIf_eq_start(sc, &sc->tempInt, &fftDim);
				VkSetToZero(sc, &sc->tempInt);
				VkIf_end(sc);
			}
			temp_int.data.i = 2 * (fftDim.data.i / 2);
			VkSub(sc, &sc->inoutID, &sc->inoutID, &temp_int);

			VkIf_eq_start(sc, &sc->inoutID, &fftDim);
			VkSetToZero(sc, &sc->inoutID);
			VkIf_end(sc);

		VkIf_end(sc);
		VkIf_end(sc);

		temp_int.data.i = 3 * fftDim.data.i / 4;
		VkIf_ge_start(sc, &sc->sdataID, &temp_int);

			temp_int.data.i = 2;
			VkMul(sc, &sc->inoutID, &sc->sdataID, &temp_int, 0);
			if (sc->mergeSequencesR2C) {
				temp_int.data.i = fftDim.data.i - 1;
				VkSub(sc, &sc->tempInt, &sc->inoutID, &temp_int);
				VkIf_eq_start(sc, &sc->tempInt, &fftDim);
				VkSetToZero(sc, &sc->tempInt);
				VkIf_end(sc);
			}
			temp_int.data.i = 2 * fftDim.data.i - 1;
			VkSub(sc, &sc->inoutID, &temp_int, &sc->inoutID);

			VkIf_eq_start(sc, &sc->inoutID, &fftDim);
			VkSetToZero(sc, &sc->inoutID);
			VkIf_end(sc);

		VkIf_end(sc);

		if (sc->axis_id > 0) {
			VkMul(sc, &sc->inoutID, &sc->inoutID, &sc->sharedStride, 0);
			VkAdd(sc, &sc->inoutID, &sc->inoutID, &sc->gl_LocalInvocationID_x);
			if (sc->mergeSequencesR2C) {
				VkMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
				VkAdd(sc, &sc->tempInt, &sc->tempInt, &sc->gl_LocalInvocationID_x);
			}
		}
		else {
			VkDiv(sc, &sc->blockInvocationID, &sc->combinedID, &fftDim);

			if (sc->stridedSharedLayout) {
				VkMul(sc, &sc->inoutID, &sc->inoutID, &sc->sharedStride, 0);
				VkAdd(sc, &sc->inoutID, &sc->inoutID, &sc->blockInvocationID);
				if (sc->mergeSequencesR2C) {
					VkMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
					VkAdd(sc, &sc->tempInt, &sc->tempInt, &sc->blockInvocationID);
				}
			}
			else {
				VkMul(sc, &sc->blockInvocationID, &sc->blockInvocationID, &sc->sharedStride, 0);
				VkAdd(sc, &sc->inoutID, &sc->inoutID, &sc->blockInvocationID);
				if (sc->mergeSequencesR2C) {
					VkAdd(sc, &sc->tempInt, &sc->tempInt, &sc->blockInvocationID);
				}
			}
		}
		appendSharedToRegisters(sc, &sc->temp, &sc->inoutID);
		if (sc->mergeSequencesR2C) {
			appendSharedToRegisters(sc, &sc->w, &sc->tempInt);
		}

		if (sc->mergeSequencesR2C) {
			VkAdd_x(sc, &sc->regIDs[i], &sc->temp, &sc->w);
			VkSub_y(sc, &sc->regIDs[i], &sc->temp, &sc->w);

			VkAdd_y(sc, &sc->w, &sc->temp, &sc->w);
			VkSub_x(sc, &sc->w, &sc->w, &sc->temp);
			VkMov_x_y(sc, &sc->temp, &sc->w);
			VkMov_y_x(sc, &sc->temp, &sc->w);
		}
		
		temp_int.data.i = fftDim.data.i / 4;
		VkIf_lt_start(sc, &sc->sdataID, &temp_int);

			temp_int.data.i = 1;
			VkAdd(sc, &sc->tempInt, &sc->sdataID, &temp_int);
			temp_int.data.i = 2;
			VkDiv(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			VkMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			temp_int.data.i = 0;
			VkIf_gt_start(sc, &sc->tempInt, &temp_int);
			if (sc->mergeSequencesR2C) {
				VkMov_x_Neg_x(sc, &sc->w, &sc->regIDs[i]);
				VkMov_y_Neg_x(sc, &sc->w, &sc->temp);
			}
			else {
				VkMov_x_Neg_x(sc, &sc->w, &sc->temp);
			}
			VkIf_else(sc);
			if (sc->mergeSequencesR2C) {
				VkMov_x(sc, &sc->w, &sc->regIDs[i]);
				VkMov_y_x(sc, &sc->w, &sc->temp);
			}
			else {
				VkMov_x(sc, &sc->w, &sc->temp);
			}
			VkIf_end(sc);

			temp_int.data.i = 2;
			VkDiv(sc, &sc->tempInt, &sc->sdataID, &temp_int);
			VkMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			temp_int.data.i = 0;
			VkIf_gt_start(sc, &sc->tempInt, &temp_int);
			if (sc->mergeSequencesR2C) {
				VkAdd_x_y(sc, &sc->w, &sc->w, &sc->regIDs[i]);
				VkAdd_y(sc, &sc->w, &sc->w, &sc->temp);
			}
			else {
				VkAdd_x_y(sc, &sc->w, &sc->w, &sc->temp);
			}
			VkIf_else(sc);
			if (sc->mergeSequencesR2C) {
				VkSub_x_y(sc, &sc->w, &sc->w, &sc->regIDs[i]);
				VkSub_y(sc, &sc->w, &sc->w, &sc->temp);
			}
			else {
				VkSub_x_y(sc, &sc->w, &sc->w, &sc->temp);
			}
			VkIf_end(sc);

		VkIf_end(sc);


		temp_int.data.i = fftDim.data.i / 2;
		VkIf_lt_start(sc, &sc->sdataID, &temp_int);
		temp_int.data.i = fftDim.data.i / 4;
		VkIf_ge_start(sc, &sc->sdataID, &temp_int);

			temp_int.data.i = 1;
			VkAdd(sc, &sc->tempInt, &sc->sdataID, &temp_int);
			temp_int.data.i = 2;
			VkDiv(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			VkMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			temp_int.data.i = 0;
			VkIf_gt_start(sc, &sc->tempInt, &temp_int);
			if (sc->mergeSequencesR2C) {
				VkMov_x_Neg_x(sc, &sc->w, &sc->regIDs[i]);
				VkMov_y_Neg_x(sc, &sc->w, &sc->temp);
			}
			else {
				VkMov_x_Neg_x(sc, &sc->w, &sc->temp);
			}
			VkIf_else(sc);
			if (sc->mergeSequencesR2C) {
				VkMov_x(sc, &sc->w, &sc->regIDs[i]);
				VkMov_y_x(sc, &sc->w, &sc->temp);
			}
			else {
				VkMov_x(sc, &sc->w, &sc->temp);
			}
			VkIf_end(sc);

			temp_int.data.i = 2;
			VkDiv(sc, &sc->tempInt, &sc->sdataID, &temp_int);
			VkMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			temp_int.data.i = 0;
			VkIf_gt_start(sc, &sc->tempInt, &temp_int);
			if (sc->mergeSequencesR2C) {
				VkSub_x_y(sc, &sc->w, &sc->w, &sc->regIDs[i]);
				VkSub_y(sc, &sc->w, &sc->w, &sc->temp);
			}
			else {
				VkSub_x_y(sc, &sc->w, &sc->w, &sc->temp);
			}
			VkIf_else(sc);
			if (sc->mergeSequencesR2C) {
				VkAdd_x_y(sc, &sc->w, &sc->w, &sc->regIDs[i]);
				VkAdd_y(sc, &sc->w, &sc->w, &sc->temp);
			}
			else {
				VkAdd_x_y(sc, &sc->w, &sc->w, &sc->temp);
			}
			VkIf_end(sc);

		VkIf_end(sc);
		VkIf_end(sc);


		temp_int.data.i = 3 * fftDim.data.i / 4;
		VkIf_lt_start(sc, &sc->sdataID, &temp_int);
		temp_int.data.i = fftDim.data.i / 2;
		VkIf_ge_start(sc, &sc->sdataID, &temp_int);
		
			temp_int.data.i = 1;
			VkAdd(sc, &sc->tempInt, &sc->sdataID, &temp_int);
			temp_int.data.i = 2;
			VkDiv(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			VkMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			temp_int.data.i = 0;
			VkIf_gt_start(sc, &sc->tempInt, &temp_int);
			if (sc->mergeSequencesR2C) {
				VkMov_x_Neg_x(sc, &sc->w, &sc->regIDs[i]);
				VkMov_y_Neg_x(sc, &sc->w, &sc->temp);
			}
			else {
				VkMov_x_Neg_x(sc, &sc->w, &sc->temp);
			}
			VkIf_else(sc);
			if (sc->mergeSequencesR2C) {
				VkMov_x(sc, &sc->w, &sc->regIDs[i]);
				VkMov_y_x(sc, &sc->w, &sc->temp);
			}
			else {
				VkMov_x(sc, &sc->w, &sc->temp);
			}
			VkIf_end(sc);

			temp_int.data.i = 2;
			VkDiv(sc, &sc->tempInt, &sc->sdataID, &temp_int);
			VkMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			temp_int.data.i = 0;
			VkIf_gt_start(sc, &sc->tempInt, &temp_int);
			if (sc->mergeSequencesR2C) {
				VkAdd_x_y(sc, &sc->w, &sc->w, &sc->regIDs[i]);
				VkAdd_y(sc, &sc->w, &sc->w, &sc->temp);
			}
			else {
				VkAdd_x_y(sc, &sc->w, &sc->w, &sc->temp);
			}
			VkIf_else(sc);
			if (sc->mergeSequencesR2C) {
				VkSub_x_y(sc, &sc->w, &sc->w, &sc->regIDs[i]);
				VkSub_y(sc, &sc->w, &sc->w, &sc->temp);
			}
			else {
				VkSub_x_y(sc, &sc->w, &sc->w, &sc->temp);
			}
			VkIf_end(sc);

		VkIf_end(sc);
		VkIf_end(sc);


		temp_int.data.i = 3 * fftDim.data.i / 4;
		VkIf_ge_start(sc, &sc->sdataID, &temp_int);
		
			temp_int.data.i = 1;
			VkAdd(sc, &sc->tempInt, &sc->sdataID, &temp_int);
			temp_int.data.i = 2;
			VkDiv(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			VkMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			temp_int.data.i = 0;
			VkIf_gt_start(sc, &sc->tempInt, &temp_int);
			if (sc->mergeSequencesR2C) {
				VkMov_x_Neg_x(sc, &sc->w, &sc->regIDs[i]);
				VkMov_y_Neg_x(sc, &sc->w, &sc->temp);
			}
			else {
				VkMov_x_Neg_x(sc, &sc->w, &sc->temp);
			}
			VkIf_else(sc);
			if (sc->mergeSequencesR2C) {
				VkMov_x(sc, &sc->w, &sc->regIDs[i]);
				VkMov_y_x(sc, &sc->w, &sc->temp);
			}
			else {
				VkMov_x(sc, &sc->w, &sc->temp);
			}
			VkIf_end(sc);

			temp_int.data.i = 2;
			VkDiv(sc, &sc->tempInt, &sc->sdataID, &temp_int);
			VkMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			temp_int.data.i = 0;
			VkIf_gt_start(sc, &sc->tempInt, &temp_int);
			if (sc->mergeSequencesR2C) {
				VkSub_x_y(sc, &sc->w, &sc->w, &sc->regIDs[i]);
				VkSub_y(sc, &sc->w, &sc->w, &sc->temp);
			}
			else {
				VkSub_x_y(sc, &sc->w, &sc->w, &sc->temp);
			}
			VkIf_else(sc);
			if (sc->mergeSequencesR2C) {
				VkAdd_x_y(sc, &sc->w, &sc->w, &sc->regIDs[i]);
				VkAdd_y(sc, &sc->w, &sc->w, &sc->temp);
			}
			else {
				VkAdd_x_y(sc, &sc->w, &sc->w, &sc->temp);
			}
			VkIf_end(sc);

		VkIf_end(sc);

		temp_double.data.d = sqrt(2);
		if (sc->mergeSequencesR2C) {
			temp_double.data.d *= 0.5;
		}

		if (sc->mergeSequencesR2C) {
			VkMul(sc, &sc->regIDs[i], &sc->w, &temp_double, 0);
		}
		else {
			VkMul_x(sc, &sc->regIDs[i], &sc->w, &temp_double, 0);
		}

		if (sc->axis_id > 0) {
			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				VkIf_end(sc);
			}
		}
		else {
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				VkIf_end(sc);
			}
		}
	}
	if (sc->useDisableThreads) {
		VkIf_end(sc);
	}
	sc->writeFromRegisters = 1;
	
	return;
}

#endif
