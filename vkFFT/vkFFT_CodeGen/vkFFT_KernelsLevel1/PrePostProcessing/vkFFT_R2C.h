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
#ifndef VKFFT_R2C_H
#define VKFFT_R2C_H
#include "vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT_CodeGen/vkFFT_StringManagement/vkFFT_StringManager.h"
#include "vkFFT_CodeGen/vkFFT_MathUtils/vkFFT_MathUtils.h"

static inline void appendC2R_read(VkFFTSpecializationConstantsLayout* sc, int type, int readWrite) {
	if (sc->res != VKFFT_SUCCESS) return;
	VkContainer temp_int = {};
	temp_int.type = 31;
	VkContainer temp_int1 = {};
	temp_int1.type = 31;
	VkContainer temp_double = {};
	temp_double.type = 32;

	VkContainer used_registers = {};
	used_registers.type = 31;
	VkContainer mult = {};
	mult.type = 31;

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

	if ((type == 6) && (readWrite == 0)) {
		if (sc->zeropadBluestein[readWrite]) {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i;
		}
		else {
			fftDim.data.i = sc->fftDim.data.i;
		}
	}

	VkDivCeil(sc, &used_registers, &fftDim, &localSize);

	appendBarrierVkFFT(sc);
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		VkIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	for (uint64_t i = 0; i < used_registers.data.i; i++) {
		if (i < ((fftDim.data.i / 2 + 1) / localSize.data.i)) {
			temp_int.data.i = i * localSize.data.i;
			if (sc->stridedSharedLayout) {
				VkAdd(sc, &sc->sdataID, localInvocationID, &temp_int);
				VkMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
				VkAdd(sc, &sc->sdataID, &sc->sdataID, batchingInvocationID);
			}
			else {
				VkMul(sc, &sc->sdataID, batchingInvocationID, &sc->sharedStride, 0);
				VkAdd(sc, &sc->sdataID, &sc->sdataID, &temp_int);
				VkAdd(sc, &sc->sdataID, &sc->sdataID, localInvocationID);
			}
			appendSharedToRegisters(sc, &sc->regIDs[i], &sc->sdataID);
			if (sc->mergeSequencesR2C) {
				temp_int.data.i = ((int64_t)ceil(fftDim.data.i / 2.0) + (1 - fftDim.data.i % 2));

				if (sc->stridedSharedLayout)
					temp_int.data.i *= sc->sharedStride.data.i;

				VkAdd(sc, &sc->sdataID, &sc->sdataID, &temp_int);
				appendSharedToRegisters(sc, &sc->temp, &sc->sdataID);

				VkShuffleComplex(sc, &sc->regIDs[i], &sc->regIDs[i], &sc->temp, 0);
			}
		}
		else {
			if (i >= (int64_t)ceil((fftDim.data.i / 2 + 1) / (long double)localSize.data.i)) {
				if ((1 + i) * localSize.data.i > fftDim.data.i) {
					temp_int.data.i = fftDim.data.i - (i)*localSize.data.i;
					VkIf_lt_start(sc, localInvocationID, &temp_int);
				}
				if ((((int64_t)ceil(fftDim.data.i / 2.0) - 1 - (localSize.data.i - ((fftDim.data.i / 2) % localSize.data.i + 1))) > (i - ((int64_t)ceil((fftDim.data.i / 2 + 1) / (long double)localSize.data.i))) * localSize.data.i) && ((uint64_t)ceil(fftDim.data.i / 2.0) - 1 > (localSize.data.i - ((fftDim.data.i / 2) % localSize.data.i + 1)))) {
					if (sc->zeropadBluestein[0]) {
						temp_int.data.i = ((int64_t)ceil(fftDim.data.i / 2.0) - 1 - (localSize.data.i - ((fftDim.data.i / 2) % localSize.data.i + 1))) - (i - ((int64_t)ceil((fftDim.data.i / 2 + 1) / (long double)localSize.data.i))) * localSize.data.i;
						VkIf_gt_start(sc, &temp_int, localInvocationID);
					}

					temp_int.data.i = ((int64_t)ceil(fftDim.data.i / 2.0) - 1 - (localSize.data.i - ((fftDim.data.i / 2) % localSize.data.i + 1))) - (i - ((int64_t)ceil((fftDim.data.i / 2 + 1) / (long double)localSize.data.i))) * localSize.data.i;
					if (sc->stridedSharedLayout) {
						VkSub(sc, &sc->sdataID, &temp_int, localInvocationID);
						VkMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
						VkAdd(sc, &sc->sdataID, &sc->sdataID, batchingInvocationID);
					}
					else {
						VkMul(sc, &sc->sdataID, batchingInvocationID, &sc->sharedStride, 0);
						VkAdd(sc, &sc->sdataID, &sc->sdataID, &temp_int);
						VkSub(sc, &sc->sdataID, &sc->sdataID, localInvocationID);
					}
					appendSharedToRegisters(sc, &sc->regIDs[i], &sc->sdataID);
					if (sc->mergeSequencesR2C) {
						temp_int.data.i = ((int64_t)ceil(fftDim.data.i / 2.0) + (1 - fftDim.data.i % 2));

						if (sc->stridedSharedLayout)
							temp_int.data.i *= sc->sharedStride.data.i;

						VkAdd(sc, &sc->sdataID, &sc->sdataID, &temp_int);
						appendSharedToRegisters(sc, &sc->temp, &sc->sdataID);

						VkShuffleComplexInv(sc, &sc->regIDs[i], &sc->regIDs[i], &sc->temp, 0);
					}
					VkConjugate(sc, &sc->regIDs[i], &sc->regIDs[i]);

					if (sc->zeropadBluestein[0]) {
						VkIf_else(sc);

						VkSetToZero(sc, &sc->regIDs[i]);

						VkIf_end(sc);
					}
				}
				else {
					VkSetToZero(sc, &sc->regIDs[i]);
				}
				if ((1 + i) * localSize.data.i > fftDim.data.i) {
					VkIf_end(sc);
				}
			}
			else {
				if (localSize.data.i > fftDim.data.i) {
					VkIf_lt_start(sc, localInvocationID, &fftDim);
				}
				temp_int.data.i = (fftDim.data.i / 2 + 1) % localSize.data.i;
				VkIf_lt_start(sc, localInvocationID, &temp_int);

				temp_int.data.i = i * localSize.data.i;
				if (sc->stridedSharedLayout)
				{
					VkAdd(sc, &sc->sdataID, localInvocationID, &temp_int);
					VkMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
					VkAdd(sc, &sc->sdataID, &sc->sdataID, batchingInvocationID);
				}
				else {
					VkMul(sc, &sc->sdataID, batchingInvocationID, &sc->sharedStride, 0);
					VkAdd(sc, &sc->sdataID, &sc->sdataID, &temp_int);
					VkAdd(sc, &sc->sdataID, &sc->sdataID, localInvocationID);
				}
				appendSharedToRegisters(sc, &sc->regIDs[i], &sc->sdataID);
				if (sc->mergeSequencesR2C) {
					temp_int.data.i = ((int64_t)ceil(fftDim.data.i / 2.0) + (1 - fftDim.data.i % 2));

					if (sc->stridedSharedLayout)
						temp_int.data.i *= sc->sharedStride.data.i;

					VkAdd(sc, &sc->sdataID, &sc->sdataID, &temp_int);
					appendSharedToRegisters(sc, &sc->temp, &sc->sdataID);

					VkShuffleComplex(sc, &sc->regIDs[i], &sc->regIDs[i], &sc->temp, 0);
				}
				VkIf_else(sc);

				temp_int.data.i = ((int64_t)ceil(fftDim.data.i / 2.0) - 1 - (localSize.data.i - ((fftDim.data.i / 2) % localSize.data.i + 1))) - (i - ((int64_t)ceil((fftDim.data.i / 2 + 1) / (long double)localSize.data.i))) * localSize.data.i;

				if (sc->stridedSharedLayout)
				{
					VkSub(sc, &sc->sdataID, &temp_int, localInvocationID);
					VkMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
					VkAdd(sc, &sc->sdataID, &sc->sdataID, batchingInvocationID);
				}
				else {
					VkMul(sc, &sc->sdataID, batchingInvocationID, &sc->sharedStride, 0);
					VkAdd(sc, &sc->sdataID, &sc->sdataID, &temp_int);
					VkSub(sc, &sc->sdataID, &sc->sdataID, localInvocationID);
				}
				appendSharedToRegisters(sc, &sc->regIDs[i], &sc->sdataID);
				if (sc->mergeSequencesR2C) {
					temp_int.data.i = ((int64_t)ceil(fftDim.data.i / 2.0) + (1 - fftDim.data.i % 2));

					if (sc->stridedSharedLayout)
						temp_int.data.i *= sc->sharedStride.data.i;

					VkAdd(sc, &sc->sdataID, &sc->sdataID, &temp_int);
					appendSharedToRegisters(sc, &sc->temp, &sc->sdataID);

					VkShuffleComplexInv(sc, &sc->regIDs[i], &sc->regIDs[i], &sc->temp, 0);
				}
				VkConjugate(sc, &sc->regIDs[i], &sc->regIDs[i]);

				VkIf_end(sc);

				if (localSize.data.i > fftDim.data.i) {
					VkIf_else(sc);

					VkSetToZero(sc, &sc->regIDs[i]);

					VkIf_end(sc);
				}
			}
		}
	}
	if (sc->useDisableThreads) {
		VkIf_end(sc);
	}
	if ((sc->rader_generator[0] > 0) || ((sc->fftDim.data.i % sc->localSize[0].data.i) && (!sc->stridedSharedLayout)) || ((sc->fftDim.data.i % sc->localSize[1].data.i) && (sc->stridedSharedLayout)))
		sc->readToRegisters = 0;
	else
		sc->readToRegisters = 1;

	if (sc->zeropadBluestein[0]) {
		fftDim.data.i = sc->fft_dim_full.data.i;
		VkDivCeil(sc, &used_registers, &fftDim, &localSize);
	}
	if (!sc->readToRegisters) {
		appendBarrierVkFFT(sc);
		if (sc->useDisableThreads) {
			temp_int.data.i = 0;
			VkIf_gt_start(sc, &sc->disableThreads, &temp_int);
		}
		for (uint64_t k = 0; k < sc->registerBoost; k++) {
			for (uint64_t i = 0; i < used_registers.data.i; i++) {
				if ((1 + i + k * used_registers.data.i) * localSize.data.i > fftDim.data.i) {
					temp_int.data.i = fftDim.data.i - (i + k * used_registers.data.i) * localSize.data.i;
					VkIf_lt_start(sc, localInvocationID, &temp_int);
				}

				temp_int.data.i = (i + k * used_registers.data.i) * localSize.data.i;

				if (sc->stridedSharedLayout)
				{
					VkAdd(sc, &sc->sdataID, &temp_int, localInvocationID);
					VkMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
					VkAdd(sc, &sc->sdataID, &sc->sdataID, batchingInvocationID);
				}
				else {
					VkMul(sc, &sc->sdataID, batchingInvocationID, &sc->sharedStride, 0);
					VkAdd(sc, &sc->sdataID, &sc->sdataID, &temp_int);
					VkAdd(sc, &sc->sdataID, &sc->sdataID, localInvocationID);
				}
				appendRegistersToShared(sc, &sc->sdataID, &sc->regIDs[i + k * used_registers.data.i]);
				if ((1 + i + k * used_registers.data.i) * localSize.data.i > fftDim.data.i) {
					sc->tempLen = sprintf(sc->tempStr, "		}\n");
					VkAppendLine(sc);
				}
			}
		}
		if (sc->useDisableThreads) {
			VkIf_end(sc);
		}
	}
	return;
}

static inline void appendR2C_write(VkFFTSpecializationConstantsLayout* sc, int type, int readWrite) {
	if (sc->res != VKFFT_SUCCESS) return;
	VkContainer temp_int = {};
	temp_int.type = 31;
	VkContainer temp_int1 = {};
	temp_int1.type = 31;
	VkContainer temp_double = {};
	temp_double.type = 32;

	VkContainer used_registers = {};
	used_registers.type = 31;
	VkContainer mult = {};
	mult.type = 31;

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

	if ((type == 5) && (readWrite == 1)) {
		if (sc->zeropadBluestein[readWrite]) {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i;
		}
		else {
			fftDim.data.i = sc->fftDim.data.i;
		}
	}

	fftDim_half.data.i = fftDim.data.i / 2 + 1;

	if (sc->mergeSequencesR2C)
		mult.data.i = 2;
	else
		mult.data.i = 1;

	if ((type == 5) && (readWrite == 1)) {
		VkMul(sc, &used_registers, &fftDim_half, &mult, 0);
		VkDivCeil(sc, &used_registers, &used_registers, &localSize);

		//mult.data.i = 1;
	}

	appendBarrierVkFFT(sc);
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		VkIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	//we actually construct 2x used_registers here, if mult = 2
	for (uint64_t i = 0; i < used_registers.data.i; i++) {

		if (sc->localSize[1].data.i == 1) {
			//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		combinedID = %s + %" PRIu64 ";\n", &sc->gl_LocalInvocationID_x, (i + k * used_registers) * &sc->localSize[0]);
			temp_int.data.i = (i)*sc->localSize[0].data.i;

			VkAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
		}
		else {
			//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		combinedID = (%s + %" PRIu64 " * %s) + %" PRIu64 ";\n", &sc->gl_LocalInvocationID_x, &sc->localSize[0], &sc->gl_LocalInvocationID_y, (i + k * used_registers) * &sc->localSize[0] * &sc->localSize[1]);
			VkMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

			temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

			VkAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
			VkAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
		}

		temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
		temp_int1.data.i = mult.data.i * fftDim_half.data.i * batching_localSize.data.i;
		if (temp_int.data.i > temp_int1.data.i) {
			//check that we only read fftDim * local batch data
			//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
			VkIf_lt_start(sc, &sc->combinedID, &temp_int1);
		}

		VkMod(sc, &sc->sdataID, &sc->combinedID, &fftDim_half);

		if (sc->stridedSharedLayout) {
			VkMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
			temp_int.data.i = 2 * fftDim_half.data.i;
			VkDiv(sc, &sc->tempInt, &sc->combinedID, &temp_int);
			VkAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
		}
		else {
			temp_int.data.i = 2 * fftDim_half.data.i;
			VkDiv(sc, &sc->tempInt, &sc->combinedID, &temp_int);
			VkMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
			VkAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
		}

		appendSharedToRegisters(sc, &sc->temp, &sc->sdataID);

		VkMod(sc, &sc->sdataID, &sc->combinedID, &fftDim_half);
		VkSub(sc, &sc->sdataID, &fftDim, &sc->sdataID);

		VkIf_eq_start(sc, &sc->sdataID, &fftDim);
		VkSetToZero(sc, &sc->sdataID);
		VkIf_end(sc);

		if (sc->stridedSharedLayout) {
			VkMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);

			VkAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
		}
		else {
			VkAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
		}

		appendSharedToRegisters(sc, &sc->w, &sc->sdataID);

		VkDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim_half);
		temp_int.data.i = 2;
		VkMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
		temp_int.data.i = 0;
		VkIf_eq_start(sc, &sc->tempInt, &temp_int);

		VkAdd_x(sc, &sc->regIDs[i], &sc->temp, &sc->w);
		VkSub_y(sc, &sc->regIDs[i], &sc->temp, &sc->w);
		VkIf_else(sc);
		VkAdd_y(sc, &sc->temp, &sc->temp, &sc->w);
		VkSub_x(sc, &sc->temp, &sc->w, &sc->temp);
		VkMov_x_y(sc, &sc->regIDs[i], &sc->temp);
		VkMov_y_x(sc, &sc->regIDs[i], &sc->temp);
		VkIf_end(sc);
		temp_double.data.d = 0.5l;
		VkMul(sc, &sc->regIDs[i], &sc->regIDs[i], &temp_double, 0);

		temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
		temp_int1.data.i = mult.data.i * fftDim_half.data.i * batching_localSize.data.i;
		if (temp_int.data.i > temp_int1.data.i) {
			//check that we only read fftDim * local batch data
			//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
			VkIf_end(sc);
		}
	}
	if (sc->useDisableThreads) {
		VkIf_end(sc);
	}
	sc->writeFromRegisters = 1;

	return;
}
#endif
