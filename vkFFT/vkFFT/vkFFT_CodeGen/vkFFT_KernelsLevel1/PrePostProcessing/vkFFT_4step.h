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
#ifndef VKFFT_4STEP_H
#define VKFFT_4STEP_H
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_StringManagement/vkFFT_StringManager.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_MathUtils/vkFFT_MathUtils.h"

#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_Zeropad.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_KernelUtils.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryTransfers/vkFFT_Transfers.h"
static inline void appendReorder4Step(VkFFTSpecializationConstantsLayout* sc, int type, int readWrite) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_int1 = VKFFT_ZERO_INIT;
	temp_int1.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 32;
	uint64_t logicalRegistersPerThread;
	if (readWrite==0)
		logicalRegistersPerThread = (sc->rader_generator[0] > 0) ? sc->min_registers_per_thread : sc->registers_per_thread_per_radix[sc->stageRadix[0]];// (sc->registers_per_thread % sc->stageRadix[sc->numStages - 1] == 0) ? sc->registers_per_thread : sc->min_registers_per_thread;
	else
		logicalRegistersPerThread = (sc->rader_generator[sc->numStages - 1] > 0) ? sc->min_registers_per_thread : sc->registers_per_thread_per_radix[sc->stageRadix[sc->numStages - 1]];// (sc->registers_per_thread % sc->stageRadix[sc->numStages - 1] == 0) ? sc->registers_per_thread : sc->min_registers_per_thread;
	switch (type) {
	case 1: case 2: {//grouped_c2c
		if ((sc->stageStartSize.data.i > 1) && (((!sc->reorderFourStep) && (sc->inverse) && (readWrite==0)) || ((!((sc->stageStartSize.data.i > 1) && (!sc->reorderFourStep) && (sc->inverse))) && (readWrite == 1)))) {
			if (((!sc->readToRegisters) && (readWrite==0))|| ((!sc->writeFromRegisters) && (readWrite == 1))) {
				appendBarrierVkFFT(sc);
			}
			if (sc->useDisableThreads) {
				temp_int.data.i = 0;
				PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
			}
			PfDivCeil(sc, &temp_int1, &sc->fftDim, &sc->localSize[1]);
			if (type == 1) {
				PfDiv(sc, &sc->inoutID, &sc->shiftX, &sc->fft_dim_x);
				PfMod(sc, &sc->inoutID, &sc->inoutID, &sc->stageStartSize);
			}
			else {
				PfMod(sc, &sc->inoutID, &sc->shiftX, &sc->stageStartSize);
			}
			for (uint64_t i = 0; i < (uint64_t)temp_int1.data.i; i++) {
				PfMod(sc, &temp_int, &sc->fftDim, &sc->localSize[1]);
				if ((temp_int.data.i != 0) && (i == (temp_int1.data.i - 1))) {
					PfIf_lt_start(sc, &sc->gl_LocalInvocationID_y, &temp_int);				
				}
				uint64_t id = (i / logicalRegistersPerThread) * sc->registers_per_thread + i % logicalRegistersPerThread;

				if ((sc->LUT) && (sc->LUT_4step)) {
					temp_int.data.i = i * sc->localSize[1].data.i;
					PfAdd(sc, &sc->tempInt, &sc->gl_LocalInvocationID_y, &temp_int);
					PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->stageStartSize, 0);
					PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->inoutID);
					temp_int.data.i = sc->maxStageSumLUT;
					PfAdd(sc, &sc->tempInt, &sc->tempInt, &temp_int);
					appendGlobalToRegisters(sc, &sc->mult, &sc->LUTStruct, &sc->tempInt);
					
					if (!sc->inverse) {
						PfConjugate(sc, &sc->mult, &sc->mult);
					}
				}
				else {
					temp_int.data.i = i * sc->localSize[1].data.i;
					PfAdd(sc, &sc->tempInt, &sc->gl_LocalInvocationID_y, &temp_int);
					PfMul(sc, &sc->tempInt, &sc->inoutID, &sc->tempInt, 0);
					temp_double.data.d = 2 * sc->double_PI/ (long double)(sc->stageStartSize.data.i * sc->fftDim.data.i);
					PfMul(sc, &sc->angle, &sc->tempInt, &temp_double, 0);
					PfSinCos(sc, &sc->mult, &sc->angle);
					if ((!sc->inverse) && (readWrite == 1)) {
						PfConjugate(sc, &sc->mult, &sc->mult);
					}
				}
				if (((sc->readToRegisters) && (readWrite == 0)) || ((sc->writeFromRegisters) && (readWrite == 1))) {
					PfMul(sc, &sc->regIDs[id], &sc->regIDs[id], &sc->mult, &sc->temp);
				}
				else {
					temp_int.data.i = i * sc->localSize[1].data.i;
					PfAdd(sc, &sc->sdataID, &sc->gl_LocalInvocationID_y, &temp_int);
					PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
					PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);
					appendSharedToRegisters(sc, &sc->w, &sc->sdataID);
					PfMul(sc, &sc->w, &sc->w, &sc->mult, &sc->temp);
					appendRegistersToShared(sc, &sc->sdataID, &sc->w);
				}
				PfMod(sc, &temp_int, &sc->fftDim, &sc->localSize[1]);
				if ((temp_int.data.i != 0) && (i == (temp_int1.data.i - 1))) {
					PfIf_end(sc);
				}
			}
			if (sc->useDisableThreads) {
				PfIf_end(sc);
			}
		}

		break;
	}
	}
	return;
}

#endif
