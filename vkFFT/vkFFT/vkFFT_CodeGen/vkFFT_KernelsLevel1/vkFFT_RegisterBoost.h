// This file is part of VkFFT
//
// Copyright (C) 2021 - present Dmitrii Tolmachev <dtolm96@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, &including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, &iNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
#ifndef VKFFT_REGISTERBOOST_H
#define VKFFT_REGISTERBOOST_H

#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_StringManagement/vkFFT_StringManager.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_MathUtils/vkFFT_MathUtils.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_KernelUtils.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_Zeropad.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryTransfers/vkFFT_Transfers.h"
static inline void appendBoostThreadDataReorder(VkFFTSpecializationConstantsLayout* sc, int type, int start) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (sc->fftDim.data.i == 1) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_int1 = VKFFT_ZERO_INIT;
	temp_int1.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 22;

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
	pfINT logicalStoragePerThread;
	if (start == 1) {
		logicalStoragePerThread = sc->registers_per_thread_per_radix[sc->stageRadix[0]] * sc->registerBoost;// (sc->registers_per_thread % sc->stageRadix[0] == 0) ? sc->registers_per_thread * sc->registerBoost : sc->min_registers_per_thread * sc->registerBoost;
	}
	else {
		logicalStoragePerThread = sc->registers_per_thread_per_radix[sc->stageRadix[sc->numStages - 1]] * sc->registerBoost;// (sc->registers_per_thread % sc->stageRadix[sc->numStages - 1] == 0) ? sc->registers_per_thread * sc->registerBoost : sc->min_registers_per_thread * sc->registerBoost;
	}
	pfINT logicalGroupSize = sc->fftDim.data.i / logicalStoragePerThread;
	if ((sc->registerBoost > 1) && (logicalStoragePerThread != sc->min_registers_per_thread * sc->registerBoost)) {
		for (pfINT k = 0; k < sc->registerBoost; k++) {
			if (k > 0) {
				appendBarrierVkFFT(sc);
			}
			if (sc->useDisableThreads) {
				temp_int.data.i = 0;
				PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
			}
			if (start == 0) {
				temp_int.data.i = logicalStoragePerThread;
				PfDivCeil(sc, &temp_int1, &sc->fftDim, &temp_int);
				PfIf_lt_start(sc, localInvocationID, &temp_int1);

				for (pfUINT i = 0; i < (pfUINT)logicalStoragePerThread / sc->registerBoost; i++) {
					temp_int.data.i = i * logicalGroupSize;
					PfAdd(sc, &sc->sdataID, localInvocationID, &temp_int);
					if (sc->stridedSharedLayout) {
						PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
						PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);
					}
					appendRegistersToShared(sc, &sc->sdataID, &sc->regIDs[i + k * sc->registers_per_thread]);
				}
				PfIf_end(sc);
			}
			else
			{
				for (pfUINT i = 0; i < sc->min_registers_per_thread; i++) {
					temp_int.data.i = i * localSize.data.i;
					PfAdd(sc, &sc->sdataID, localInvocationID, &temp_int);
					if (sc->stridedSharedLayout) {
						PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
						PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);
					}
					appendRegistersToShared(sc, &sc->sdataID, &sc->regIDs[i + k * sc->registers_per_thread]);
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

			if (start == 1) {
				temp_int.data.i = logicalStoragePerThread;
				PfDivCeil(sc, &temp_int1, &sc->fftDim, &temp_int);
				PfIf_lt_start(sc, localInvocationID, &temp_int1);

				for (pfUINT i = 0; i < (pfUINT)logicalStoragePerThread / sc->registerBoost; i++) {
					temp_int.data.i = i * logicalGroupSize;
					PfAdd(sc, &sc->sdataID, localInvocationID, &temp_int);
					if (sc->stridedSharedLayout) {
						PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
						PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);
					}

					appendSharedToRegisters(sc, &sc->regIDs[i + k * sc->registers_per_thread], &sc->sdataID);
				}
				PfIf_end(sc);
			}
			else {
				for (pfUINT i = 0; i < sc->min_registers_per_thread; i++) {
					temp_int.data.i = i * localSize.data.i;
					PfAdd(sc, &sc->sdataID, localInvocationID, &temp_int);
					if (sc->stridedSharedLayout) {
						PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
						PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);
					}
					appendSharedToRegisters(sc, &sc->regIDs[i + k * sc->registers_per_thread], &sc->sdataID);
				}
			}
			if (sc->useDisableThreads) {
				PfIf_end(sc);
			}
		}
	}
	return;
}

#endif
