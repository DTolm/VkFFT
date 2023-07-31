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
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_StringManagement/vkFFT_StringManager.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_MathUtils/vkFFT_MathUtils.h"

static inline void appendDCTI_read(VkFFTSpecializationConstantsLayout* sc, int type, int readWrite) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_int1 = VKFFT_ZERO_INIT;
	temp_int1.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 32;

	PfContainer used_registers = VKFFT_ZERO_INIT;
	used_registers.type = 31;
	
	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;

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
		PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[1]);
	}
	else {
		PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[0]);
	}

	appendBarrierVkFFT(sc);
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	for (uint64_t i = 0; i < (uint64_t)used_registers.data.i; i++) {
		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i)*sc->localSize[1].data.i;

			PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		else {
			if (sc->localSize[1].data.i == 1) {
				temp_int.data.i = (i)*sc->localSize[0].data.i;

				PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
			}
			else {
				PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

				temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

				PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
				PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
			}
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		if (sc->stridedSharedLayout) {
			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->combinedID, &temp_int);
			temp_int.data.i = fftDim.data.i - 1;
			PfIf_lt_start(sc, &sc->combinedID, &temp_int);
		}else{
			PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->tempInt, &temp_int);
			temp_int.data.i = fftDim.data.i - 1;
			PfIf_lt_start(sc, &sc->tempInt, &temp_int);
		}
		if (sc->stridedSharedLayout) {
			PfMul(sc, &sc->tempInt, &sc->combinedID, &sc->sharedStride, 0);

			temp_int.data.i = (2 * fftDim.data.i - 2) * sc->sharedStride.data.i;
			PfAdd(sc, &sc->inoutID, &sc->gl_LocalInvocationID_x, &temp_int);
			PfSub(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);

			PfAdd(sc, &sc->sdataID, &sc->gl_LocalInvocationID_x, &sc->tempInt);
		}
		else {
			PfDiv(sc, &sc->sdataID, &sc->combinedID, &fftDim);
			PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);

			temp_int.data.i = (2 * fftDim.data.i - 2);
			PfAdd(sc, &sc->inoutID, &sc->sdataID, &temp_int);
			PfSub(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);

			PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
		}
		appendSharedToRegisters(sc, &sc->temp, &sc->sdataID);
		appendRegistersToShared(sc, &sc->inoutID, &sc->temp);
		PfIf_end(sc);
		PfIf_end(sc);
		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_end(sc);
			}
		}
		else {
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
	}
	if (sc->useDisableThreads) {
		PfIf_end(sc);
	}
	return;
}

static inline void appendDCTII_read_III_write(VkFFTSpecializationConstantsLayout* sc, int type, int readWrite) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_int1 = VKFFT_ZERO_INIT;
	temp_int1.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 32;

	PfContainer used_registers = VKFFT_ZERO_INIT;
	used_registers.type = 31;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;

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
		PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[1]);
	}
	else {
		PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[0]);
	}

	appendBarrierVkFFT(sc);
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	for (uint64_t i = 0; i < (uint64_t)used_registers.data.i; i++) {
		if (sc->axis_id > 0) {
			temp_int.data.i = (i)*sc->localSize[1].data.i;

			PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		else {
			if (sc->localSize[1].data.i == 1) {
				temp_int.data.i = (i)*sc->localSize[0].data.i;

				PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
			}
			else {
				PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

				temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

				PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
				PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
			}

			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		if (sc->axis_id > 0){
			temp_int.data.i = 2;
			PfMod(sc, &sc->sdataID, &sc->combinedID, &temp_int);
		}
		else {
			PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			temp_int.data.i = 2;
			PfMod(sc, &sc->sdataID, &sc->tempInt, &temp_int);
		}
		PfMul(sc, &sc->blockInvocationID, &sc->sdataID, &temp_int, 0);
		temp_int.data.i = 2;
		if (sc->axis_id > 0) {
			PfDiv(sc, &sc->tempInt, &sc->combinedID, &temp_int);
		}
		else {
			PfDiv(sc, &sc->tempInt, &sc->tempInt, &temp_int);
		}
		PfMul(sc, &sc->blockInvocationID, &sc->blockInvocationID, &sc->tempInt, 0);
		
		temp_int.data.i = fftDim.data.i - 1;
		PfMul(sc, &sc->sdataID, &sc->sdataID, &temp_int, 0);
		PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
		PfSub(sc, &sc->sdataID, &sc->sdataID, &sc->blockInvocationID);

		if (sc->axis_id > 0) {
			PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
			PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);
		}
		else {
			if (sc->stridedSharedLayout) {
				PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
				PfDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
			}
			else {
				PfDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
				PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
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
				PfIf_end(sc);
			}
		}
		else {
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
	}
	if (sc->useDisableThreads) {
		PfIf_end(sc);
	}
	if (readWrite)
		sc->writeFromRegisters = 1;
	else
		sc->readToRegisters = 0;
	return;
}
static inline void appendDCTII_write_III_read(VkFFTSpecializationConstantsLayout* sc, int type, int readWrite) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_int1 = VKFFT_ZERO_INIT;
	temp_int1.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 32;

	PfContainer used_registers = VKFFT_ZERO_INIT;
	used_registers.type = 31;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;

	PfContainer fftDim_half = VKFFT_ZERO_INIT;
	fftDim_half.type = 31;

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
	PfDiv(sc, &fftDim_half, &fftDim, &temp_int);
	PfInc(sc, &fftDim_half);

	if (sc->stridedSharedLayout) {
		PfDivCeil(sc, &used_registers, &fftDim_half, &sc->localSize[1]);
	}
	else {
		PfDivCeil(sc, &used_registers, &fftDim_half, &sc->localSize[0]);
	}

	appendBarrierVkFFT(sc);
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	for (uint64_t i = 0; i < (uint64_t)used_registers.data.i; i++) {
		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i)*sc->localSize[1].data.i;

			PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim_half.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		else {
			if (sc->localSize[1].data.i == 1) {
				temp_int.data.i = (i)*sc->localSize[0].data.i;

				PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
			}
			else {
				PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

				temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

				PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
				PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
			}
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim_half.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim_half * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim_half * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}

		if (sc->LUT) {
			if (sc->stridedSharedLayout) {
				PfAdd(sc, &sc->tempInt, &sc->combinedID, &sc->startDCT3LUT);
			}
			else {
				PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim_half);
				PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->startDCT3LUT);
			}
			appendGlobalToRegisters(sc, &sc->mult, &sc->LUTStruct, &sc->tempInt);
			if ((!sc->mergeSequencesR2C) && (readWrite)) {
				temp_double.data.d = 2.0l;
				PfMul(sc, &sc->mult, &sc->mult, &temp_double, 0);
			}
			if (readWrite)
				PfConjugate(sc, &sc->mult, &sc->mult);
		}
		else {
			if (readWrite)
				temp_double.data.d = -sc->double_PI / 2.0 / fftDim.data.i;
			else
				temp_double.data.d = sc->double_PI / 2.0 / fftDim.data.i;
			if (sc->stridedSharedLayout) {
				PfMul(sc, &sc->tempFloat, &sc->combinedID, &temp_double, 0);
			}
			else {
				PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim_half);
				PfMul(sc, &sc->tempFloat, &sc->tempInt, &temp_double, 0);
			}

			PfSinCos(sc, &sc->mult, &sc->tempFloat);
			if ((!sc->mergeSequencesR2C) && (readWrite)) {
				temp_double.data.d = 2.0l;
				PfMul(sc, &sc->mult, &sc->mult, &temp_double, 0);
			}
		}

		if (sc->stridedSharedLayout) {
			PfMul(sc, &sc->sdataID, &sc->combinedID, &sc->sharedStride, 0);

			temp_int.data.i = fftDim.data.i * sc->sharedStride.data.i;
			PfSub(sc, &sc->inoutID, &temp_int, &sc->sdataID);

			temp_int.data.i = 0;
			PfIf_eq_start(sc, &sc->sdataID, &temp_int);
			PfMov(sc, &sc->inoutID, &sc->sdataID);
			PfIf_end(sc);

			PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);
			PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->gl_LocalInvocationID_x);
		}
		else {
			PfMod(sc, &sc->sdataID, &sc->combinedID, &fftDim_half);
			if (sc->stridedSharedLayout) {
				PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);

				temp_int.data.i = fftDim.data.i * sc->sharedStride.data.i;
				PfSub(sc, &sc->inoutID, &temp_int, &sc->sdataID);

				temp_int.data.i = 0;
				PfIf_eq_start(sc, &sc->sdataID, &temp_int);
				PfMov(sc, &sc->inoutID, &sc->sdataID);
				PfIf_end(sc);

				PfDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim_half);
				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
			}
			else {
				temp_int.data.i = fftDim.data.i;
				PfSub(sc, &sc->inoutID, &temp_int, &sc->sdataID);

				temp_int.data.i = 0;
				PfIf_eq_start(sc, &sc->sdataID, &temp_int);
				PfMov(sc, &sc->inoutID, &sc->sdataID);
				PfIf_end(sc);

				PfDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim_half);
				PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);

				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
			}
		}

		if (readWrite) {
			appendSharedToRegisters(sc, &sc->temp, &sc->sdataID);
			if (sc->mergeSequencesR2C) {
				appendSharedToRegisters(sc, &sc->w, &sc->inoutID);

				PfAdd_x(sc, &sc->regIDs[0], &sc->temp, &sc->w);
				PfSub_y(sc, &sc->regIDs[0], &sc->temp, &sc->w);
				PfSub_x(sc, &sc->regIDs[1], &sc->w, &sc->temp);
				PfAdd_y(sc, &sc->regIDs[1], &sc->temp, &sc->w);

				PfMul(sc, &sc->temp, &sc->regIDs[0], &sc->mult, 0);
				PfConjugate(sc, &sc->mult, &sc->mult);
				PfMul(sc, &sc->w, &sc->regIDs[1], &sc->mult, 0);
				PfMov_x(sc, &sc->regIDs[0], &sc->temp);
				PfMov_y(sc, &sc->regIDs[0], &sc->w);
				PfMov_x_Neg_y(sc, &sc->regIDs[1], &sc->temp);
				PfMov_y_Neg_x(sc, &sc->regIDs[1], &sc->w);

				appendRegistersToShared(sc, &sc->inoutID, &sc->regIDs[1]);
				appendRegistersToShared(sc, &sc->sdataID, &sc->regIDs[0]);
			}
			else {
				PfMul(sc, &sc->regIDs[0], &sc->temp, &sc->mult, 0);
				PfMov_x_Neg_y(sc, &sc->w, &sc->regIDs[0]);

				appendRegistersToShared(sc, &sc->inoutID, &sc->w);
				appendRegistersToShared(sc, &sc->sdataID, &sc->regIDs[0]);
			}
		}
		else {
			appendSharedToRegisters(sc, &sc->temp, &sc->sdataID);
			appendSharedToRegisters(sc, &sc->w, &sc->inoutID);

			PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim_half);
			temp_int.data.i = 0;
			PfIf_eq_start(sc, &sc->tempInt, &temp_int);
			PfSetToZero(sc, &sc->w);
			PfIf_end(sc);

			PfMov_x_y(sc, &sc->regIDs[0], &sc->w);
			PfMov_y_x(sc, &sc->regIDs[0], &sc->w);

			PfSub_x(sc, &sc->regIDs[1], &sc->temp, &sc->regIDs[0]);
			PfAdd_y(sc, &sc->regIDs[1], &sc->temp, &sc->regIDs[0]);

			PfAdd_x(sc, &sc->w, &sc->temp, &sc->regIDs[0]);
			PfSub_y(sc, &sc->w, &sc->temp, &sc->regIDs[0]);

			PfMul(sc, &sc->regIDs[0], &sc->w, &sc->mult, 0);
			PfConjugate(sc, &sc->mult, &sc->mult);

			PfMul(sc, &sc->temp, &sc->regIDs[1], &sc->mult, 0);

			appendRegistersToShared(sc, &sc->inoutID, &sc->temp);
			appendRegistersToShared(sc, &sc->sdataID, &sc->regIDs[0]);
		}
		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim_half.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
		else {
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim_half.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
	}
	if (sc->useDisableThreads) {
		PfIf_end(sc);
	}
	return;
}

static inline void appendDCTIV_even_read(VkFFTSpecializationConstantsLayout* sc, int type, int readWrite) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_int1 = VKFFT_ZERO_INIT;
	temp_int1.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 32;

	PfContainer used_registers = VKFFT_ZERO_INIT;
	used_registers.type = 31;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;
	PfContainer fftDim_half = VKFFT_ZERO_INIT;
	fftDim_half.type = 31;
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
		PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[1]);
	}
	else {
		PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[0]);
	}
	if (sc->readToRegisters == 1) {
		if (sc->useDisableThreads) {
			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
		}
		for (uint64_t i = 0; i < (uint64_t)used_registers.data.i; i++) {
			if (sc->axis_id > 0) {
				temp_int.data.i = (i)*sc->localSize[1].data.i;

				PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

				temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					//check that we only read fftDim * local batch data
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
					PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
				}
			}
			else {
				if (sc->localSize[1].data.i == 1) {
					temp_int.data.i = (i)*sc->localSize[0].data.i;

					PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
				}
				else {
					PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

					temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

					PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
					PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
				}
				temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					//check that we only read fftDim * local batch data
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
					PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
				}
			}
			if (sc->axis_id > 0) {
				temp_int1.data.i = 2;
				PfDiv(sc, &sc->sdataID, &sc->combinedID, &temp_int1);

				PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);
			}
			else {
				if (sc->stridedSharedLayout) {
					temp_int.data.i = fftDim.data.i;
					PfMod(sc, &sc->sdataID, &sc->combinedID, &temp_int);

					temp_int1.data.i = 2;
					PfDiv(sc, &sc->sdataID, &sc->sdataID, &temp_int1);

					PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
					PfDiv(sc, &sc->tempInt, &sc->combinedID, &temp_int);
					PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
				}
				else {
					temp_int.data.i = fftDim.data.i;
					PfMod(sc, &sc->sdataID, &sc->combinedID, &temp_int);

					temp_int1.data.i = 2;
					PfDiv(sc, &sc->sdataID, &sc->sdataID, &temp_int1);

					PfDiv(sc, &sc->tempInt, &sc->combinedID, &temp_int);
					PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
					PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
				}
			}
			if (sc->axis_id > 0) {
				temp_int.data.i = 2;
				PfMod(sc, &sc->tempInt, &sc->combinedID, &temp_int);
			}
			else {
				PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
				temp_int.data.i = 2;
				PfMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			}
			temp_int.data.i = 0;
			PfIf_eq_start(sc, &sc->tempInt, &temp_int);
			if (i < (uint64_t)used_registers.data.i / 2) {
				appendRegistersToShared_x_x(sc, &sc->sdataID, &sc->regIDs[i]);
			}
			else {
				appendRegistersToShared_x_y(sc, &sc->sdataID, &sc->regIDs[i - used_registers.data.i / 2]);
			}
#if(!((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5)))
			PfIf_else(sc);
			if (i < (uint64_t)used_registers.data.i / 2) {
				appendRegistersToShared_y_x(sc, &sc->sdataID, &sc->regIDs[i]);
			}
			else {
				appendRegistersToShared_y_y(sc, &sc->sdataID, &sc->regIDs[i - used_registers.data.i / 2]);
			}
#endif
			PfIf_end(sc);
			if (sc->axis_id > 0) {
				temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					PfIf_end(sc);
				}
			}
			else {
				temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					PfIf_end(sc);
				}
			}
		}
		if (sc->useDisableThreads) {
			PfIf_end(sc);
		}
#if(((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5)))
		appendBarrierVkFFT(sc);
		if (sc->useDisableThreads) {
			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
		}
		for (uint64_t i = 0; i < (uint64_t)used_registers.data.i; i++) {
			if (sc->axis_id > 0) {
				temp_int.data.i = (i)*sc->localSize[1].data.i;

				PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

				temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					//check that we only read fftDim * local batch data
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
					PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
				}
			}
			else {
				if (sc->localSize[1].data.i == 1) {
					temp_int.data.i = (i)*sc->localSize[0].data.i;

					PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
				}
				else {
					PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

					temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

					PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
					PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
				}
				temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					//check that we only read fftDim * local batch data
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
					PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
			}
			if (sc->axis_id > 0) {
				temp_int1.data.i = 2;
				PfDiv(sc, &sc->sdataID, &sc->combinedID, &temp_int1);

				PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);
			}
			else {
				if (sc->stridedSharedLayout) {
					temp_int.data.i = fftDim.data.i;
					PfMod(sc, &sc->sdataID, &sc->combinedID, &temp_int);

					temp_int1.data.i = 2;
					PfDiv(sc, &sc->sdataID, &sc->sdataID, &temp_int1);

					PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
					PfDiv(sc, &sc->tempInt, &sc->combinedID, &temp_int);
					PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
				}
				else {
					temp_int.data.i = fftDim.data.i;
					PfMod(sc, &sc->sdataID, &sc->combinedID, &temp_int);

					temp_int1.data.i = 2;
					PfDiv(sc, &sc->sdataID, &sc->sdataID, &temp_int1);

					PfDiv(sc, &sc->tempInt, &sc->combinedID, &temp_int);
					PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
					PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
				}
			}
			if (sc->axis_id > 0) {
				temp_int.data.i = 2;
				PfMod(sc, &sc->tempInt, &sc->combinedID, &temp_int);
			}
			else {
				PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
				temp_int.data.i = 2;
				PfMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			}
			temp_int.data.i = 1;
			PfIf_eq_start(sc, &sc->tempInt, &temp_int);
			if ((int64_t)i < used_registers.data.i / 2) {
				appendRegistersToShared_y_x(sc, &sc->sdataID, &sc->regIDs[i]);
			}
			else {
				appendRegistersToShared_y_y(sc, &sc->sdataID, &sc->regIDs[i - used_registers.data.i / 2]);
			}
			PfIf_end(sc);

			if (sc->axis_id > 0) {
				temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					PfIf_end(sc);
				}
			}
			else {
				temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					PfIf_end(sc);
				}
			}
		}
		if (sc->useDisableThreads) {
			PfIf_end(sc);
		}
#endif
	}
	appendBarrierVkFFT(sc);
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	fftDim.data.i = fftDim.data.i / 2;

	if (sc->stridedSharedLayout) {
		PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[1]);
	}
	else {
		PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[0]);
	}
	for (uint64_t i = 0; i < (uint64_t)used_registers.data.i; i++) {
		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i)*sc->localSize[1].data.i;

			PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		else {
			if (sc->localSize[1].data.i == 1) {
				temp_int.data.i = (i)*sc->localSize[0].data.i;

				PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
			}
			else {
				PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

				temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

				PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
				PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
			}
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}

		if (sc->stridedSharedLayout) {
			PfMul(sc, &sc->sdataID, &sc->combinedID, &sc->sharedStride, 0);
			PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);

			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->combinedID, &temp_int);
		}
		else {
			PfMod(sc, &sc->sdataID, &sc->combinedID, &fftDim);
			PfDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
			PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);

			PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->tempInt, &temp_int);
		}

		if (sc->stridedSharedLayout) {
			PfSub(sc, &sc->tempInt, &sc->sdataID, &sc->sharedStride);
			appendSharedToRegisters_y_y(sc, &sc->w, &sc->tempInt);
		}
		else {
			temp_int.data.i = 1;
			PfSub(sc, &sc->tempInt, &sc->sdataID, &temp_int);
			appendSharedToRegisters_y_y(sc, &sc->w, &sc->tempInt);
		}

		appendSharedToRegisters_x_x(sc, &sc->w, &sc->sdataID);
		
		PfMov_x_y(sc, &sc->regIDs[i], &sc->w);
		PfMov_y_Neg_x(sc, &sc->regIDs[i], &sc->w);
		PfAdd(sc, &sc->regIDs[i], &sc->regIDs[i], &sc->w);
		
		PfIf_else(sc);

		appendSharedToRegisters_x_x(sc, &sc->regIDs[i], &sc->sdataID);
		if (sc->stridedSharedLayout) {
			temp_int.data.i = (fftDim.data.i - 1) * sc->sharedStride.data.i;
			PfAdd(sc, &sc->sdataID, &sc->sdataID, &temp_int);
		}
		else {
			temp_int.data.i = (fftDim.data.i - 1);
			PfAdd(sc, &sc->sdataID, &sc->sdataID, &temp_int);
		}
		appendSharedToRegisters_y_y(sc, &sc->regIDs[i], &sc->sdataID);
		temp_double.data.d = 2.0l;
		PfMul(sc, &sc->regIDs[i], &sc->regIDs[i], &temp_double, 0);
		
		PfIf_end(sc);
		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_end(sc);
			}
		}
		else {
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
	}
	if (sc->useDisableThreads) {
		PfIf_end(sc);
	}
	appendBarrierVkFFT(sc);
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	for (uint64_t i = 0; i < (uint64_t)used_registers.data.i; i++) {
		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i)*sc->localSize[1].data.i;

			PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		else {
			if (sc->localSize[1].data.i == 1) {
				temp_int.data.i = (i)*sc->localSize[0].data.i;

				PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
			}
			else {
				PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

				temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

				PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
				PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
			}

			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		if (sc->stridedSharedLayout) {
			PfMul(sc, &sc->sdataID, &sc->combinedID, &sc->sharedStride, 0);
			PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);

			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->combinedID, &temp_int);
		}
		else {
			PfMod(sc, &sc->sdataID, &sc->combinedID, &fftDim);
			PfDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
			PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);

			PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->tempInt, &temp_int);
		}

		appendRegistersToShared_x_x(sc, &sc->sdataID, &sc->regIDs[i]);

#if(!((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5)))//OpenCL, Level Zero and Metal are  not handling barrier with thread-conditional writes to local memory - so this is a work-around
		if (sc->stridedSharedLayout) {
			PfSub(sc, &sc->sdataID, &fftDim, &sc->combinedID);

			PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
			PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);
		}
		else {
			PfMod(sc, &sc->sdataID, &sc->combinedID, &fftDim);
			PfSub(sc, &sc->sdataID, &fftDim, &sc->sdataID);

			PfDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
			PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
		}

		appendRegistersToShared_y_y(sc, &sc->sdataID, &sc->regIDs[i]);
#endif
		
		PfIf_else(sc);

		appendRegistersToShared(sc, &sc->sdataID, &sc->regIDs[i]);

		PfIf_end(sc);

		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_end(sc);
			}
		}
		else {
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
	}
	if (sc->useDisableThreads) {
		PfIf_end(sc);
	}
#if(((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5)))//OpenCL, Level Zero and Metal are  not handling barrier with thread-conditional writes to local memory - so this is a work-around

	appendBarrierVkFFT(sc);
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	for (uint64_t i = 0; i < (uint64_t)used_registers.data.i; i++) {
		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i)*sc->localSize[1].data.i;

			PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		else {
			if (sc->localSize[1].data.i == 1) {
				temp_int.data.i = (i)*sc->localSize[0].data.i;

				PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
			}
			else {
				PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

				temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

				PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
				PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
			}

			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}

		if (sc->stridedSharedLayout) {
			PfSub(sc, &sc->sdataID, &fftDim, &sc->combinedID);

			PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
			PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);

			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->combinedID, &temp_int);
		}
		else {
			PfMod(sc, &sc->sdataID, &sc->combinedID, &fftDim);
			PfSub(sc, &sc->sdataID, &fftDim, &sc->sdataID);

			PfDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
			PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
			
			PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->tempInt, &temp_int);
		}

		appendRegistersToShared_y_y(sc, &sc->sdataID, &sc->regIDs[i]);

		PfIf_end(sc);

		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
		else {
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
	}
	if (sc->useDisableThreads) {
		PfIf_end(sc);
	}
#endif

	appendDCTII_write_III_read(sc, type, 0);

	sc->readToRegisters = 0;

	return;
}
static inline void appendDCTIV_even_write(VkFFTSpecializationConstantsLayout* sc, int type, int readWrite) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_int1 = VKFFT_ZERO_INIT;
	temp_int1.type = 31;
	PfContainer temp_int2 = VKFFT_ZERO_INIT;
	temp_int2.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 32;

	PfContainer used_registers = VKFFT_ZERO_INIT;
	used_registers.type = 31;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;

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

	if (sc->zeropadBluestein[readWrite]) {
		fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i;
	}
	else {
		fftDim.data.i = sc->fftDim.data.i;
	}

	if (sc->stridedSharedLayout) {
		PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[1]);
	}
	else {
		PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[0]);
	}

	appendDCTII_read_III_write(sc, type, 1);

	appendBarrierVkFFT(sc);
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	for (uint64_t i = 0; i < (uint64_t)used_registers.data.i; i++) {
		if (sc->axis_id > 0) {
			temp_int.data.i = (i)*sc->localSize[1].data.i;

			PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		else {
			if (sc->localSize[1].data.i == 1) {
				temp_int.data.i = (i)*sc->localSize[0].data.i;

				PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
			}
			else {
				PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

				temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

				PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
				PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
			}

			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		if (sc->axis_id > 0) {
			temp_int.data.i = 2;
			PfMod(sc, &sc->sdataID, &sc->combinedID, &temp_int);
		}
		else {
			PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			temp_int.data.i = 2;
			PfMod(sc, &sc->sdataID, &sc->tempInt, &temp_int);
		}
		PfMul(sc, &sc->blockInvocationID, &sc->sdataID, &temp_int, 0);
		temp_double.data.d = 1;
		PfSub(sc, &sc->tempFloat, &temp_double, &sc->blockInvocationID);
		PfMul_y(sc, &sc->regIDs[i], &sc->regIDs[i], &sc->tempFloat, 0);

		if (sc->LUT) {
			if (sc->axis_id > 0) {
				PfAdd(sc, &sc->tempInt, &sc->combinedID, &sc->startDCT4LUT);
			}
			else {
				PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->startDCT4LUT);
			}
			appendGlobalToRegisters(sc, &sc->mult, &sc->LUTStruct, &sc->tempInt);
		}
		else {
			temp_int.data.i = 2;
			if (sc->axis_id > 0) {
				PfMul(sc, &sc->tempInt, &sc->combinedID, &temp_int, 0);
			}
			else {
				PfMul(sc, &sc->tempInt, &sc->tempInt, &temp_int, 0);
			}
			PfInc(sc, &sc->tempInt);
			if (readWrite)
				temp_double.data.d = -sc->double_PI / 8.0 / fftDim.data.i;
			else
				temp_double.data.d = sc->double_PI / 8.0 / fftDim.data.i;
			PfMul(sc, &sc->tempFloat, &sc->tempInt, &temp_double, 0);

			PfSinCos(sc, &sc->mult, &sc->tempFloat);
		}

		PfMul(sc, &sc->regIDs[i], &sc->regIDs[i], &sc->mult, &sc->temp);
		PfConjugate(sc, &sc->regIDs[i], &sc->regIDs[i]);

		if (sc->axis_id > 0) {
			PfMul(sc, &sc->tempInt, &sc->combinedID, &sc->sharedStride, 0);

			temp_int.data.i = (fftDim.data.i - 1) * sc->sharedStride.data.i;
			PfAdd(sc, &sc->sdataID, &sc->gl_LocalInvocationID_x, &temp_int);
			PfSub(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
		}
		else {
			PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			if (sc->stridedSharedLayout) {
				temp_int.data.i = (fftDim.data.i-1);
				PfSub(sc, &sc->sdataID, &temp_int, &sc->tempInt);
				PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
				
				PfDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
			}
			else {
				PfDiv(sc, &sc->sdataID, &sc->combinedID, &fftDim);
				PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);

				temp_int.data.i = (fftDim.data.i-1);
				PfAdd(sc, &sc->sdataID, &sc->sdataID, &temp_int);
				PfSub(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
			}
		}
		appendRegistersToShared_y_y(sc, &sc->sdataID, &sc->regIDs[i]);
		if (sc->axis_id > 0) {
			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
		else {
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
	}
	if (sc->useDisableThreads) {
		PfIf_end(sc);
	}
	appendBarrierVkFFT(sc);
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	for (uint64_t i = 0; i < (uint64_t)used_registers.data.i; i++) {
		if (sc->axis_id > 0) {
			temp_int.data.i = (i)*sc->localSize[1].data.i;

			PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		else {
			if (sc->localSize[1].data.i == 1) {
				temp_int.data.i = (i)*sc->localSize[0].data.i;

				PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
			}
			else {
				PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

				temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

				PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
				PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
			}

			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}

		if (sc->axis_id > 0) {
			PfMul(sc, &sc->tempInt, &sc->combinedID, &sc->sharedStride, 0);

			PfAdd(sc, &sc->sdataID, &sc->gl_LocalInvocationID_x, &sc->tempInt);
		}
		else {
			PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			if (sc->stridedSharedLayout) {
				PfMul(sc, &sc->sdataID, &sc->tempInt, &sc->sharedStride, 0);

				PfDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
			}
			else {
				PfDiv(sc, &sc->sdataID, &sc->combinedID, &fftDim);
				PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);

				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
			}
		}
		appendSharedToRegisters_y_y(sc, &sc->regIDs[i], &sc->sdataID);
		if (sc->axis_id > 0) {
			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
		else {
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
	}
	if (sc->useDisableThreads) {
		PfIf_end(sc);
	}
	return;
}

static inline void appendDCTIV_odd_read(VkFFTSpecializationConstantsLayout* sc, int type, int readWrite) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_int1 = VKFFT_ZERO_INIT;
	temp_int1.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 32;

	PfContainer used_registers = VKFFT_ZERO_INIT;
	used_registers.type = 31;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;

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
		PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[1]);
	}
	else {
		PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[0]);
	}

	appendBarrierVkFFT(sc);
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	for (uint64_t i = 0; i < (uint64_t)used_registers.data.i; i++) {
		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i)*sc->localSize[1].data.i;

			PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		else {
			if (sc->localSize[1].data.i == 1) {
				temp_int.data.i = (i)*sc->localSize[0].data.i;

				PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
			}
			else {
				PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

				temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

				PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
				PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
			}

			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		if (sc->stridedSharedLayout) {
			temp_int.data.i = 4;
			PfMul(sc, &sc->inoutID, &sc->combinedID, &temp_int,0);
		}
		else {
			PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			temp_int.data.i = 4;
			PfMul(sc, &sc->inoutID, &sc->tempInt, &temp_int,0);
		}
		temp_int.data.i = fftDim.data.i / 2;
		PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int);
		
		PfIf_lt_start(sc, &sc->inoutID, &fftDim);
		PfMov(sc, &sc->sdataID, &sc->inoutID);
		PfIf_end(sc);
		
		temp_int.data.i = fftDim.data.i * 2;
		PfIf_lt_start(sc, &sc->inoutID, &temp_int);
		PfIf_ge_start(sc, &sc->inoutID, &fftDim);
		temp_int.data.i = fftDim.data.i * 2 - 1;
		PfSub(sc, &sc->sdataID, &temp_int, &sc->inoutID);
		PfIf_end(sc);
		PfIf_end(sc);

		temp_int.data.i = fftDim.data.i * 3;
		PfIf_lt_start(sc, &sc->inoutID, &temp_int);
		temp_int.data.i = fftDim.data.i * 2;
		PfIf_ge_start(sc, &sc->inoutID, &temp_int);
		temp_int.data.i = fftDim.data.i * 2;
		PfSub(sc, &sc->sdataID, &sc->inoutID, &temp_int);
		PfIf_end(sc);
		PfIf_end(sc);

		temp_int.data.i = fftDim.data.i * 4;
		PfIf_lt_start(sc, &sc->inoutID, &temp_int);
		temp_int.data.i = fftDim.data.i * 3;
		PfIf_ge_start(sc, &sc->inoutID, &temp_int);
		temp_int.data.i = fftDim.data.i * 4 - 1;
		PfSub(sc, &sc->sdataID, &temp_int, &sc->inoutID);
		PfIf_end(sc);
		PfIf_end(sc);

		temp_int.data.i = fftDim.data.i * 4;
		PfIf_ge_start(sc, &sc->inoutID, &temp_int);
		temp_int.data.i = fftDim.data.i * 4;
		PfSub(sc, &sc->sdataID, &sc->inoutID, &temp_int);
		PfIf_end(sc);
		
		if (sc->stridedSharedLayout) {
			PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
			PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);
		}
		else {
			PfDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
			PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
		}
		appendSharedToRegisters(sc, &sc->regIDs[i], &sc->sdataID);

		temp_int.data.i = fftDim.data.i * 2;
		PfIf_lt_start(sc, &sc->inoutID, &temp_int);
		PfIf_ge_start(sc, &sc->inoutID, &fftDim);
		PfMov_x_Neg_x(sc, &sc->regIDs[i], &sc->regIDs[i]);
		PfMov_y_Neg_y(sc, &sc->regIDs[i], &sc->regIDs[i]);
		PfIf_end(sc);
		PfIf_end(sc);

		temp_int.data.i = fftDim.data.i * 3;
		PfIf_lt_start(sc, &sc->inoutID, &temp_int);
		temp_int.data.i = fftDim.data.i * 2;
		PfIf_ge_start(sc, &sc->inoutID, &temp_int);
		PfMov_x_Neg_x(sc, &sc->regIDs[i], &sc->regIDs[i]);
		PfMov_y_Neg_y(sc, &sc->regIDs[i], &sc->regIDs[i]);
		PfIf_end(sc);
		PfIf_end(sc);

		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
		else {
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
	}
	if (sc->useDisableThreads) {
		PfIf_end(sc);
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
			PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
		}
		for (uint64_t i = 0; i < (uint64_t)used_registers.data.i; i++) {
			if (sc->stridedSharedLayout) {
				temp_int.data.i = (i)*sc->localSize[1].data.i;

				PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

				temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					//check that we only read fftDim * local batch data
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
					PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
				}
			}
			else {
				if (sc->localSize[1].data.i == 1) {
					temp_int.data.i = (i)*sc->localSize[0].data.i;

					PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
				}
				else {
					PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

					temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

					PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
					PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
				}

				temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					//check that we only read fftDim * local batch data
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
					PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
				}
			}
			
			if (sc->stridedSharedLayout) {
				PfMul(sc, &sc->tempInt, &sc->combinedID, &sc->sharedStride, 0);
				PfAdd(sc, &sc->sdataID, &sc->gl_LocalInvocationID_x, &sc->tempInt);
			}
			else {
				PfDiv(sc, &sc->sdataID, &sc->combinedID, &fftDim);
				PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);

				PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
			}

			appendRegistersToShared(sc, &sc->sdataID, &sc->regIDs[i]);

			if (sc->stridedSharedLayout) {
				temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					PfIf_end(sc);
				}
			}
			else {
				temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					PfIf_end(sc);
				}
			}
		}
		if (sc->useDisableThreads) {
			PfIf_end(sc);
		}
	}
	return;
}

static inline void appendDCTIV_odd_write(VkFFTSpecializationConstantsLayout* sc, int type, int readWrite) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_int1 = VKFFT_ZERO_INIT;
	temp_int1.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 32;

	PfContainer used_registers = VKFFT_ZERO_INIT;
	used_registers.type = 31;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;

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
		PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[1]);
	}
	else {
		PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[0]);
	}

	appendBarrierVkFFT(sc);
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	for (uint64_t i = 0; i < (uint64_t)used_registers.data.i; i++) {
		if (sc->axis_id > 0) {
			temp_int.data.i = (i)*sc->localSize[1].data.i;

			PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		else {
			if (sc->localSize[1].data.i == 1) {
				temp_int.data.i = (i)*sc->localSize[0].data.i;

				PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
			}
			else {
				PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

				temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

				PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
				PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
			}

			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		if (sc->axis_id > 0) {
			PfMov(sc, &sc->sdataID, &sc->combinedID);
		}
		else {
			PfMod(sc, &sc->sdataID, &sc->combinedID, &fftDim);
		}

		temp_int.data.i = fftDim.data.i / 4;
		PfIf_lt_start(sc, &sc->sdataID, &temp_int);

			temp_int.data.i = 2;
			PfMul(sc, &sc->inoutID, &sc->sdataID, &temp_int, 0);
			PfInc(sc, &sc->inoutID);
			if (sc->mergeSequencesR2C) {
				PfSub(sc, &sc->tempInt, &fftDim, &sc->inoutID);
				PfIf_eq_start(sc, &sc->tempInt, &fftDim);
				PfSetToZero(sc, &sc->tempInt);
				PfIf_end(sc);
			}

			PfIf_eq_start(sc, &sc->inoutID, &fftDim);
			PfSetToZero(sc, &sc->inoutID);
			PfIf_end(sc);

		PfIf_end(sc);

		temp_int.data.i = fftDim.data.i / 2;
		PfIf_lt_start(sc, &sc->sdataID, &temp_int);
		temp_int.data.i = fftDim.data.i / 4;
		PfIf_ge_start(sc, &sc->sdataID, &temp_int);

			temp_int.data.i = 2;
			PfMul(sc, &sc->inoutID, &sc->sdataID, &temp_int, 0);
			if (sc->mergeSequencesR2C) {
				temp_int.data.i = fftDim.data.i - 2 * (fftDim.data.i / 2);
				PfAdd(sc, &sc->tempInt, &temp_int, &sc->inoutID);

				PfIf_eq_start(sc, &sc->tempInt, &fftDim);
				PfSetToZero(sc, &sc->tempInt);
				PfIf_end(sc);
			}
			temp_int.data.i = 2 * (fftDim.data.i / 2);
			PfSub(sc, &sc->inoutID, &temp_int, &sc->inoutID);

			PfIf_eq_start(sc, &sc->inoutID, &fftDim);
			PfSetToZero(sc, &sc->inoutID);
			PfIf_end(sc);

		PfIf_end(sc);
		PfIf_end(sc);

		temp_int.data.i = 3 * fftDim.data.i / 4;
		PfIf_lt_start(sc, &sc->sdataID, &temp_int);
		temp_int.data.i = fftDim.data.i / 2;
		PfIf_ge_start(sc, &sc->sdataID, &temp_int);

			temp_int.data.i = 2;
			PfMul(sc, &sc->inoutID, &sc->sdataID, &temp_int, 0);
			if (sc->mergeSequencesR2C) {
				temp_int.data.i = fftDim.data.i + 2 * (fftDim.data.i / 2);
				PfSub(sc, &sc->tempInt, &temp_int, &sc->inoutID);
				PfIf_eq_start(sc, &sc->tempInt, &fftDim);
				PfSetToZero(sc, &sc->tempInt);
				PfIf_end(sc);
			}
			temp_int.data.i = 2 * (fftDim.data.i / 2);
			PfSub(sc, &sc->inoutID, &sc->inoutID, &temp_int);

			PfIf_eq_start(sc, &sc->inoutID, &fftDim);
			PfSetToZero(sc, &sc->inoutID);
			PfIf_end(sc);

		PfIf_end(sc);
		PfIf_end(sc);

		temp_int.data.i = 3 * fftDim.data.i / 4;
		PfIf_ge_start(sc, &sc->sdataID, &temp_int);

			temp_int.data.i = 2;
			PfMul(sc, &sc->inoutID, &sc->sdataID, &temp_int, 0);
			if (sc->mergeSequencesR2C) {
				temp_int.data.i = fftDim.data.i - 1;
				PfSub(sc, &sc->tempInt, &sc->inoutID, &temp_int);
				PfIf_eq_start(sc, &sc->tempInt, &fftDim);
				PfSetToZero(sc, &sc->tempInt);
				PfIf_end(sc);
			}
			temp_int.data.i = 2 * fftDim.data.i - 1;
			PfSub(sc, &sc->inoutID, &temp_int, &sc->inoutID);

			PfIf_eq_start(sc, &sc->inoutID, &fftDim);
			PfSetToZero(sc, &sc->inoutID);
			PfIf_end(sc);

		PfIf_end(sc);

		if (sc->axis_id > 0) {
			PfMul(sc, &sc->inoutID, &sc->inoutID, &sc->sharedStride, 0);
			PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->gl_LocalInvocationID_x);
			if (sc->mergeSequencesR2C) {
				PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
				PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->gl_LocalInvocationID_x);
			}
		}
		else {
			PfDiv(sc, &sc->blockInvocationID, &sc->combinedID, &fftDim);

			if (sc->stridedSharedLayout) {
				PfMul(sc, &sc->inoutID, &sc->inoutID, &sc->sharedStride, 0);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->blockInvocationID);
				if (sc->mergeSequencesR2C) {
					PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
					PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->blockInvocationID);
				}
			}
			else {
				PfMul(sc, &sc->blockInvocationID, &sc->blockInvocationID, &sc->sharedStride, 0);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->blockInvocationID);
				if (sc->mergeSequencesR2C) {
					PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->blockInvocationID);
				}
			}
		}
		appendSharedToRegisters(sc, &sc->temp, &sc->inoutID);
		if (sc->mergeSequencesR2C) {
			appendSharedToRegisters(sc, &sc->w, &sc->tempInt);
		}

		if (sc->mergeSequencesR2C) {
			PfAdd_x(sc, &sc->regIDs[i], &sc->temp, &sc->w);
			PfSub_y(sc, &sc->regIDs[i], &sc->temp, &sc->w);

			PfAdd_y(sc, &sc->w, &sc->temp, &sc->w);
			PfSub_x(sc, &sc->w, &sc->w, &sc->temp);
			PfMov_x_y(sc, &sc->temp, &sc->w);
			PfMov_y_x(sc, &sc->temp, &sc->w);
		}
		
		temp_int.data.i = fftDim.data.i / 4;
		PfIf_lt_start(sc, &sc->sdataID, &temp_int);

			temp_int.data.i = 1;
			PfAdd(sc, &sc->tempInt, &sc->sdataID, &temp_int);
			temp_int.data.i = 2;
			PfDiv(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			PfMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->tempInt, &temp_int);
			if (sc->mergeSequencesR2C) {
				PfMov_x_Neg_x(sc, &sc->w, &sc->regIDs[i]);
				PfMov_y_Neg_x(sc, &sc->w, &sc->temp);
			}
			else {
				PfMov_x_Neg_x(sc, &sc->w, &sc->temp);
			}
			PfIf_else(sc);
			if (sc->mergeSequencesR2C) {
				PfMov_x(sc, &sc->w, &sc->regIDs[i]);
				PfMov_y_x(sc, &sc->w, &sc->temp);
			}
			else {
				PfMov_x(sc, &sc->w, &sc->temp);
			}
			PfIf_end(sc);

			temp_int.data.i = 2;
			PfDiv(sc, &sc->tempInt, &sc->sdataID, &temp_int);
			PfMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->tempInt, &temp_int);
			if (sc->mergeSequencesR2C) {
				PfAdd_x_y(sc, &sc->w, &sc->w, &sc->regIDs[i]);
				PfAdd_y(sc, &sc->w, &sc->w, &sc->temp);
			}
			else {
				PfAdd_x_y(sc, &sc->w, &sc->w, &sc->temp);
			}
			PfIf_else(sc);
			if (sc->mergeSequencesR2C) {
				PfSub_x_y(sc, &sc->w, &sc->w, &sc->regIDs[i]);
				PfSub_y(sc, &sc->w, &sc->w, &sc->temp);
			}
			else {
				PfSub_x_y(sc, &sc->w, &sc->w, &sc->temp);
			}
			PfIf_end(sc);

		PfIf_end(sc);


		temp_int.data.i = fftDim.data.i / 2;
		PfIf_lt_start(sc, &sc->sdataID, &temp_int);
		temp_int.data.i = fftDim.data.i / 4;
		PfIf_ge_start(sc, &sc->sdataID, &temp_int);

			temp_int.data.i = 1;
			PfAdd(sc, &sc->tempInt, &sc->sdataID, &temp_int);
			temp_int.data.i = 2;
			PfDiv(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			PfMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->tempInt, &temp_int);
			if (sc->mergeSequencesR2C) {
				PfMov_x_Neg_x(sc, &sc->w, &sc->regIDs[i]);
				PfMov_y_Neg_x(sc, &sc->w, &sc->temp);
			}
			else {
				PfMov_x_Neg_x(sc, &sc->w, &sc->temp);
			}
			PfIf_else(sc);
			if (sc->mergeSequencesR2C) {
				PfMov_x(sc, &sc->w, &sc->regIDs[i]);
				PfMov_y_x(sc, &sc->w, &sc->temp);
			}
			else {
				PfMov_x(sc, &sc->w, &sc->temp);
			}
			PfIf_end(sc);

			temp_int.data.i = 2;
			PfDiv(sc, &sc->tempInt, &sc->sdataID, &temp_int);
			PfMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->tempInt, &temp_int);
			if (sc->mergeSequencesR2C) {
				PfSub_x_y(sc, &sc->w, &sc->w, &sc->regIDs[i]);
				PfSub_y(sc, &sc->w, &sc->w, &sc->temp);
			}
			else {
				PfSub_x_y(sc, &sc->w, &sc->w, &sc->temp);
			}
			PfIf_else(sc);
			if (sc->mergeSequencesR2C) {
				PfAdd_x_y(sc, &sc->w, &sc->w, &sc->regIDs[i]);
				PfAdd_y(sc, &sc->w, &sc->w, &sc->temp);
			}
			else {
				PfAdd_x_y(sc, &sc->w, &sc->w, &sc->temp);
			}
			PfIf_end(sc);

		PfIf_end(sc);
		PfIf_end(sc);


		temp_int.data.i = 3 * fftDim.data.i / 4;
		PfIf_lt_start(sc, &sc->sdataID, &temp_int);
		temp_int.data.i = fftDim.data.i / 2;
		PfIf_ge_start(sc, &sc->sdataID, &temp_int);
		
			temp_int.data.i = 1;
			PfAdd(sc, &sc->tempInt, &sc->sdataID, &temp_int);
			temp_int.data.i = 2;
			PfDiv(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			PfMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->tempInt, &temp_int);
			if (sc->mergeSequencesR2C) {
				PfMov_x_Neg_x(sc, &sc->w, &sc->regIDs[i]);
				PfMov_y_Neg_x(sc, &sc->w, &sc->temp);
			}
			else {
				PfMov_x_Neg_x(sc, &sc->w, &sc->temp);
			}
			PfIf_else(sc);
			if (sc->mergeSequencesR2C) {
				PfMov_x(sc, &sc->w, &sc->regIDs[i]);
				PfMov_y_x(sc, &sc->w, &sc->temp);
			}
			else {
				PfMov_x(sc, &sc->w, &sc->temp);
			}
			PfIf_end(sc);

			temp_int.data.i = 2;
			PfDiv(sc, &sc->tempInt, &sc->sdataID, &temp_int);
			PfMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->tempInt, &temp_int);
			if (sc->mergeSequencesR2C) {
				PfAdd_x_y(sc, &sc->w, &sc->w, &sc->regIDs[i]);
				PfAdd_y(sc, &sc->w, &sc->w, &sc->temp);
			}
			else {
				PfAdd_x_y(sc, &sc->w, &sc->w, &sc->temp);
			}
			PfIf_else(sc);
			if (sc->mergeSequencesR2C) {
				PfSub_x_y(sc, &sc->w, &sc->w, &sc->regIDs[i]);
				PfSub_y(sc, &sc->w, &sc->w, &sc->temp);
			}
			else {
				PfSub_x_y(sc, &sc->w, &sc->w, &sc->temp);
			}
			PfIf_end(sc);

		PfIf_end(sc);
		PfIf_end(sc);


		temp_int.data.i = 3 * fftDim.data.i / 4;
		PfIf_ge_start(sc, &sc->sdataID, &temp_int);
		
			temp_int.data.i = 1;
			PfAdd(sc, &sc->tempInt, &sc->sdataID, &temp_int);
			temp_int.data.i = 2;
			PfDiv(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			PfMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->tempInt, &temp_int);
			if (sc->mergeSequencesR2C) {
				PfMov_x_Neg_x(sc, &sc->w, &sc->regIDs[i]);
				PfMov_y_Neg_x(sc, &sc->w, &sc->temp);
			}
			else {
				PfMov_x_Neg_x(sc, &sc->w, &sc->temp);
			}
			PfIf_else(sc);
			if (sc->mergeSequencesR2C) {
				PfMov_x(sc, &sc->w, &sc->regIDs[i]);
				PfMov_y_x(sc, &sc->w, &sc->temp);
			}
			else {
				PfMov_x(sc, &sc->w, &sc->temp);
			}
			PfIf_end(sc);

			temp_int.data.i = 2;
			PfDiv(sc, &sc->tempInt, &sc->sdataID, &temp_int);
			PfMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->tempInt, &temp_int);
			if (sc->mergeSequencesR2C) {
				PfSub_x_y(sc, &sc->w, &sc->w, &sc->regIDs[i]);
				PfSub_y(sc, &sc->w, &sc->w, &sc->temp);
			}
			else {
				PfSub_x_y(sc, &sc->w, &sc->w, &sc->temp);
			}
			PfIf_else(sc);
			if (sc->mergeSequencesR2C) {
				PfAdd_x_y(sc, &sc->w, &sc->w, &sc->regIDs[i]);
				PfAdd_y(sc, &sc->w, &sc->w, &sc->temp);
			}
			else {
				PfAdd_x_y(sc, &sc->w, &sc->w, &sc->temp);
			}
			PfIf_end(sc);

		PfIf_end(sc);

		temp_double.data.d = sqrt(2);
		if (sc->mergeSequencesR2C) {
			temp_double.data.d *= 0.5;
		}

		if (sc->mergeSequencesR2C) {
			PfMul(sc, &sc->regIDs[i], &sc->w, &temp_double, 0);
		}
		else {
			PfMul_x(sc, &sc->regIDs[i], &sc->w, &temp_double, 0);
		}

		if (sc->axis_id > 0) {
			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
		else {
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
	}
	if (sc->useDisableThreads) {
		PfIf_end(sc);
	}
	sc->writeFromRegisters = 1;
	
	return;
}

#endif
