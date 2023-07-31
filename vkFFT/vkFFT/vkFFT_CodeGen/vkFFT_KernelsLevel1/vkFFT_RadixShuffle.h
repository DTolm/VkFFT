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
#ifndef VKFFT_RADIXSHUFFLE_H
#define VKFFT_RADIXSHUFFLE_H

#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_StringManagement/vkFFT_StringManager.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_MathUtils/vkFFT_MathUtils.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryTransfers/vkFFT_Transfers.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_KernelUtils.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_Zeropad.h"

#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel1/vkFFT_RadixKernels.h"

static inline void appendRadixShuffleNonStrided(VkFFTSpecializationConstantsLayout* sc, PfContainer* stageSize, PfContainer* stageSizeSum, PfContainer* stageAngle, PfContainer* stageRadix, PfContainer* stageRadixNext) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_int1 = VKFFT_ZERO_INIT;
	temp_int1.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 32; 
	
	PfContainer stageNormalization;
	stageNormalization.type = 32;
	PfContainer normalizationValue;
	normalizationValue.type = 31;
	normalizationValue.data.i = 1;

	if ((((sc->actualInverse) && (sc->normalize)) || (sc->convolutionStep && (stageAngle->data.d > 0))) && (stageSize->data.i == 1) && (sc->axis_upload_id == 0) && (!(sc->useBluesteinFFT && (stageAngle->data.d < 0)))) {
		if ((sc->performDCT) && (sc->actualInverse)) {
			if (sc->performDCT == 1)
				normalizationValue.data.i = (sc->sourceFFTSize.data.i - 1) * 2;
			else
				normalizationValue.data.i = sc->sourceFFTSize.data.i * 2;
		}
		else
			normalizationValue.data.i = sc->sourceFFTSize.data.i;
	}
	if (sc->useBluesteinFFT && (stageAngle->data.d > 0) && (stageSize->data.i == 1) && (sc->axis_upload_id == 0)) {
		normalizationValue.data.i *= sc->fft_dim_full.data.i;
	}
	if (normalizationValue.data.i != 1) {
		stageNormalization.data.d = 1.0 / (long double)(normalizationValue.data.i);
	}
	
	PfContainer logicalStoragePerThread;
	logicalStoragePerThread.type = 31;
	logicalStoragePerThread.data.i = sc->registers_per_thread_per_radix[stageRadix->data.i] * sc->registerBoost;// (sc->registers_per_thread % stageRadix->data.i == 0) ? sc->registers_per_thread * sc->registerBoost : sc->min_registers_per_thread * sc->registerBoost;
	PfContainer logicalStoragePerThreadNext;
	logicalStoragePerThreadNext.type = 31;
	logicalStoragePerThreadNext.data.i = sc->registers_per_thread_per_radix[stageRadixNext->data.i] * sc->registerBoost;// (sc->registers_per_thread % stageRadixNext->data.i == 0) ? sc->registers_per_thread * sc->registerBoost : sc->min_registers_per_thread * sc->registerBoost;
	PfContainer logicalRegistersPerThread; 
	logicalRegistersPerThread.type = 31;
	logicalRegistersPerThread.data.i = sc->registers_per_thread_per_radix[stageRadix->data.i];// (sc->registers_per_thread % stageRadix->data.i == 0) ? sc->registers_per_thread : sc->min_registers_per_thread;
	PfContainer logicalRegistersPerThreadNext; 
	logicalRegistersPerThreadNext.type = 31;
	logicalRegistersPerThreadNext.data.i = sc->registers_per_thread_per_radix[stageRadixNext->data.i];// (sc->registers_per_thread % stageRadixNext->data.i == 0) ? sc->registers_per_thread : sc->min_registers_per_thread;

	PfContainer logicalGroupSize;
	logicalGroupSize.type = 31;
	PfDivCeil(sc, &logicalGroupSize, &sc->fftDim, &logicalStoragePerThread);
	PfContainer logicalGroupSizeNext;
	logicalGroupSizeNext.type = 31;
	PfDivCeil(sc, &logicalGroupSizeNext, &sc->fftDim, &logicalStoragePerThreadNext);
	
	if ((!((sc->writeFromRegisters == 1) && (stageSize->data.i == sc->fftDim.data.i / stageRadix->data.i) && (!(((sc->convolutionStep) || (sc->useBluesteinFFT && sc->BluesteinConvolutionStep)) && (stageAngle->data.d < 0) && ((sc->matrixConvolution > 1) || (sc->numKernels.data.i > 1)))))) && (((sc->registerBoost == 1) && ((sc->localSize[0].data.i * logicalStoragePerThread.data.i > sc->fftDim.data.i) || (stageSize->data.i < sc->fftDim.data.i / stageRadix->data.i) || ((sc->reorderFourStep) && (sc->fftDim.data.i < sc->fft_dim_full.data.i) && (sc->localSize[1].data.i > 1)) || (sc->localSize[1].data.i > 1) || ((sc->performR2C) && (!sc->actualInverse) && (sc->axis_id == 0)) || ((sc->convolutionStep) && ((sc->matrixConvolution > 1) || (sc->numKernels.data.i > 1)) && (stageAngle->data.d < 0)))) || (sc->performDCT)))
	{
		appendBarrierVkFFT(sc);
	}
	//if ((sc->localSize[0] * logicalStoragePerThread > sc->fftDim) || (stageSize->data.i < sc->fftDim / stageRadix->data.i) || ((sc->reorderFourStep) && (sc->fftDim < sc->fft_dim_full) && (sc->localSize[1] > 1)) || (sc->localSize[1] > 1) || ((sc->performR2C) && (!sc->actualInverse) && (sc->axis_id == 0)) || ((sc->convolutionStep) && ((sc->matrixConvolution > 1) || (sc->numKernels > 1)) && (stageAngle->data.d < 0)) || (sc->registerBoost > 1) || (sc->performDCT)) {
	if ((!((sc->writeFromRegisters == 1) && (stageSize->data.i == sc->fftDim.data.i / stageRadix->data.i) && (!(((sc->convolutionStep) || (sc->useBluesteinFFT && sc->BluesteinConvolutionStep)) && (stageAngle->data.d < 0) && ((sc->matrixConvolution > 1) || (sc->numKernels.data.i > 1)))))) && ((sc->localSize[0].data.i * logicalStoragePerThread.data.i > sc->fftDim.data.i) || (stageSize->data.i < sc->fftDim.data.i / stageRadix->data.i) || ((sc->reorderFourStep) && (sc->fftDim.data.i < sc->fft_dim_full.data.i) && (sc->localSize[1].data.i > 1)) || (sc->localSize[1].data.i > 1) || ((sc->performR2C) && (!sc->actualInverse) && (sc->axis_id == 0)) || ((sc->convolutionStep) && ((sc->matrixConvolution > 1) || (sc->numKernels.data.i > 1)) && (stageAngle->data.d < 0)) || (sc->registerBoost > 1) || (sc->performDCT))) {
		if (!((sc->registerBoost > 1) && (stageSize->data.i * stageRadix->data.i == sc->fftDim.data.i / sc->stageRadix[sc->numStages - 1]) && (sc->stageRadix[sc->numStages - 1] == sc->registerBoost))) {
			PfContainer* tempID;
			tempID = (PfContainer*)calloc(sc->registers_per_thread * sc->registerBoost, sizeof(PfContainer));
			if (tempID) {
				for (uint64_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
					PfAllocateContainerFlexible(sc, &tempID[i], 50);
					tempID[i].type = sc->regIDs[0].type;
				}

				for (uint64_t k = 0; k < sc->registerBoost; ++k) {
					uint64_t t = 0;
					if (sc->registerBoost > 1) {
						appendBarrierVkFFT(sc);
											
						if (logicalGroupSize.data.i * logicalStoragePerThread.data.i > sc->fftDim.data.i) {
							PfDivCeil(sc, &temp_int, &logicalStoragePerThread, &sc->fftDim);
							PfIf_lt_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);	
						}

						if (sc->useDisableThreads) {
							temp_int.data.i = 0;
							PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
						}
					}
					else {
						if (sc->useDisableThreads) {
							temp_int.data.i = 0;
							PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
						}
					}
					if (logicalGroupSize.data.i != sc->localSize[0].data.i) {
						PfIf_lt_start(sc, &sc->gl_LocalInvocationID_x, &logicalGroupSize);
					}
					for (uint64_t j = 0; j < (uint64_t)logicalRegistersPerThread.data.i / stageRadix->data.i; j++) {
						if (logicalGroupSize.data.i * ((int64_t)(j + k * logicalRegistersPerThread.data.i / stageRadix->data.i) * stageRadix->data.i) <= sc->fftDim.data.i) {
							if (logicalGroupSize.data.i * ((int64_t)(1 + j + k * logicalRegistersPerThread.data.i / stageRadix->data.i) * stageRadix->data.i) > sc->fftDim.data.i) {
								PfContainer current_group_cut;
								current_group_cut.type = 31;
								current_group_cut.data.i = sc->fftDim.data.i / stageRadix->data.i - (j + k * logicalRegistersPerThread.data.i / stageRadix->data.i) * logicalGroupSize.data.i;
								PfIf_lt_start(sc, &sc->gl_LocalInvocationID_x, &current_group_cut);
							}
							temp_int.data.i =  j * logicalGroupSize.data.i;
							PfAdd(sc, &sc->stageInvocationID, &sc->gl_LocalInvocationID_x, &temp_int);
							PfMov(sc, &sc->blockInvocationID, &sc->stageInvocationID);
							PfMod(sc, &sc->stageInvocationID, &sc->stageInvocationID, stageSize);
							PfSub(sc, &sc->blockInvocationID, &sc->blockInvocationID, &sc->stageInvocationID);
							PfMul(sc, &sc->inoutID, &sc->blockInvocationID, stageRadix, 0);
							PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->stageInvocationID);
						}
						/*sc->tempLen = sprintf(sc->tempStr, "\
		stageInvocationID = (gl_LocalInvocationID.x + %" PRIu64 ") %% (%" PRIu64 ");\n\
		blockInvocationID = (gl_LocalInvocationID.x + %" PRIu64 ") - stageInvocationID;\n\
		inoutID = stageInvocationID + blockInvocationID * %" PRIu64 ";\n", j * logicalGroupSize, stageSize->data.i, j * logicalGroupSize, stageRadix->data.i);*/
						
						for (uint64_t i = 0; i < (uint64_t)stageRadix->data.i; i++) {
							PfContainer id;
							id.type = 31;
							id.data.i = j + k * logicalRegistersPerThread.data.i / stageRadix->data.i + i * logicalStoragePerThread.data.i / stageRadix->data.i;
							id.data.i = (id.data.i / logicalRegistersPerThread.data.i) * sc->registers_per_thread + id.data.i % logicalRegistersPerThread.data.i;
							PfCopyContainer(sc, &tempID[t + k * sc->registers_per_thread], &sc->regIDs[id.data.i]);
							t++;
							if (logicalGroupSize.data.i * ((int64_t)(j + k * logicalRegistersPerThread.data.i / stageRadix->data.i) * stageRadix->data.i) <= sc->fftDim.data.i) {
								temp_int.data.i = i * stageSize->data.i;
								PfAdd(sc, &sc->sdataID, &sc->inoutID, &temp_int);
									
								if ((stageSize->data.i <= sc->numSharedBanks / 2) && (sc->fftDim.data.i > sc->numSharedBanks / 2) && (sc->sharedStrideBankConflictFirstStages.data.i != sc->fftDim.data.i / sc->registerBoost) && ((sc->fftDim.data.i & (sc->fftDim.data.i - 1)) == 0) && (stageSize->data.i * stageRadix->data.i != sc->fftDim.data.i)) {
									if (sc->resolveBankConflictFirstStages == 0) {
										sc->resolveBankConflictFirstStages = 1;
										PfMov(sc, &sc->sharedStride, &sc->sharedStrideBankConflictFirstStages);
									}
									temp_int.data.i = sc->numSharedBanks / 2;
									PfDiv(sc, &sc->tempInt, &sc->sdataID, &temp_int);
									temp_int.data.i = sc->numSharedBanks / 2 + 1;
									PfMul(sc, &sc->tempInt, &sc->tempInt, &temp_int, 0);
									temp_int.data.i = sc->numSharedBanks / 2;
									PfMod(sc, &sc->sdataID, &sc->sdataID, &temp_int);
									PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
								}
								else {
									if (sc->resolveBankConflictFirstStages == 1) {
										sc->resolveBankConflictFirstStages = 0;
										PfMov(sc, &sc->sharedStride, &sc->sharedStrideReadWriteConflict);	
									}
								}
								if (sc->localSize[1].data.i > 1) {
									PfMul(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &sc->sharedStride, 0);
									PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->combinedID);
								}
								//sprintf(sc->sdataID, "sharedStride * gl_LocalInvocationID.y + inoutID + %" PRIu64 "", i * stageSize->data.i);
								if (normalizationValue.data.i != 1) {
									PfMul(sc, &sc->regIDs[id.data.i], &sc->regIDs[id.data.i], &stageNormalization, 0);	
								}
								appendRegistersToShared(sc, &sc->sdataID, &sc->regIDs[id.data.i]);
									
							}
							/*sc->tempLen = sprintf(sc->tempStr, "\
sdata[sharedStride * gl_LocalInvocationID.y + inoutID + %" PRIu64 "] = temp%s%s;\n", i * stageSize->data.i, sc->regIDs[id], stageNormalization);*/
						}
						if (logicalGroupSize.data.i * ((int64_t)(j + k * logicalRegistersPerThread.data.i / stageRadix->data.i) * stageRadix->data.i) <= sc->fftDim.data.i) {
							if (logicalGroupSize.data.i * ((int64_t)(1 + j + k * logicalRegistersPerThread.data.i / stageRadix->data.i) * stageRadix->data.i) > sc->fftDim.data.i) {
								PfIf_end(sc);
							}
						}
					}
					if (logicalGroupSize.data.i != sc->localSize[0].data.i) {
						PfIf_end(sc);
					}
					for (uint64_t j = logicalRegistersPerThread.data.i; j < sc->registers_per_thread; j++) {
						PfCopyContainer(sc, &tempID[t + k * sc->registers_per_thread], &sc->regIDs[t + k * sc->registers_per_thread]);
						t++;
					}
					t = 0;

					if (sc->useDisableThreads) {
						PfIf_end(sc);
					}
					if (sc->registerBoost > 1) {
						//registerBoost
						if (logicalGroupSize.data.i * logicalStoragePerThread.data.i > sc->fftDim.data.i)
						{
							PfIf_end(sc);
						}
						appendBarrierVkFFT(sc);
						
						if (logicalGroupSize.data.i * logicalStoragePerThreadNext.data.i > sc->fftDim.data.i) {
							PfDivCeil(sc, &temp_int, &sc->fftDim, &logicalRegistersPerThreadNext);
							PfIf_lt_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);
						}

						if (sc->useDisableThreads) {
							temp_int.data.i = 0;
							PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
						}
						for (uint64_t j = 0; j < (uint64_t)logicalRegistersPerThreadNext.data.i / stageRadixNext->data.i; j++) {
							for (uint64_t i = 0; i < (uint64_t)stageRadixNext->data.i; i++) {
								PfContainer id;
								id.type = 31;
								id.data.i = j + k * logicalRegistersPerThreadNext.data.i / stageRadixNext->data.i + i * logicalStoragePerThreadNext.data.i / stageRadixNext->data.i;
								id.data.i = (id.data.i / logicalRegistersPerThreadNext.data.i) * sc->registers_per_thread + id.data.i % logicalRegistersPerThreadNext.data.i;
								//resID[t + k * sc->registers_per_thread] = sc->regIDs[id];
								temp_int.data.i = t * logicalGroupSizeNext.data.i;
								PfAdd(sc, &sc->sdataID, &sc->gl_LocalInvocationID_x, &temp_int);
								
								if (sc->localSize[1].data.i > 1) {
									PfMul(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &sc->sharedStride, 0);
									PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->combinedID);
								}
								if (sc->resolveBankConflictFirstStages == 1) {
									temp_int.data.i = sc->numSharedBanks / 2;
									PfDiv(sc, &sc->tempInt, &sc->sdataID, &temp_int);
									temp_int.data.i = sc->numSharedBanks / 2 + 1;
									PfMul(sc, &sc->tempInt, &sc->tempInt, &temp_int, 0);
									temp_int.data.i = sc->numSharedBanks / 2;
									PfMod(sc, &sc->sdataID, &sc->sdataID, &temp_int);
									PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
								}
								//sprintf(sc->sdataID, "sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + %" PRIu64 "", t * logicalGroupSizeNext);
								appendSharedToRegisters(sc, &tempID[t + k * sc->registers_per_thread], &sc->sdataID);
								
								t++;
							}

						}
						if (sc->useDisableThreads) {
							PfIf_end(sc);
						}
						if (logicalGroupSize.data.i * logicalStoragePerThreadNext.data.i > sc->fftDim.data.i)
						{
							PfIf_end(sc);
						}
					}
				}
				for (uint64_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
					PfCopyContainer(sc, &sc->regIDs[i], &tempID[i]);
				}
				for (uint64_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
					PfDeallocateContainer(sc, &tempID[i]);
				}
				free(tempID);
				tempID = 0;
			}
			else {
				sc->res = VKFFT_ERROR_MALLOC_FAILED;
				return;
			}
		}
		else {
			//registerBoost
			PfContainer* tempID;
			tempID = (PfContainer*)calloc(sc->registers_per_thread * sc->registerBoost, sizeof(PfContainer));
			if (tempID) {
				//resID = (char**)malloc(sizeof(char*) * sc->registers_per_thread * sc->registerBoost);
				for (uint64_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
					PfAllocateContainerFlexible(sc, &tempID[i], 50);
					tempID[i].type = sc->regIDs[0].type;
				}
				if (sc->useDisableThreads) {
					temp_int.data.i = 0;
					PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
				}
				for (uint64_t k = 0; k < (uint64_t)sc->registerBoost; ++k) {
					for (uint64_t j = 0; j < (uint64_t)logicalRegistersPerThread.data.i / stageRadix->data.i; j++) {
						for (uint64_t i = 0; i < (uint64_t)stageRadix->data.i; i++) {
							PfContainer id;
							id.type = 31;
							id.data.i = j + k * logicalRegistersPerThread.data.i / stageRadix->data.i + i * logicalStoragePerThread.data.i / stageRadix->data.i;
							id.data.i = (id.data.i / logicalRegistersPerThread.data.i) * sc->registers_per_thread + id.data.i % logicalRegistersPerThread.data.i;
							PfCopyContainer(sc, &tempID[j + i * logicalRegistersPerThread.data.i / stageRadix->data.i + k * sc->registers_per_thread], &sc->regIDs[id.data.i]);
						}
					}
					for (uint64_t j = logicalRegistersPerThread.data.i; j < sc->registers_per_thread; j++) {
						PfCopyContainer(sc, &tempID[j + k * sc->registers_per_thread], &sc->regIDs[j + k * sc->registers_per_thread]);
					}
				}
				for (uint64_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
					PfCopyContainer(sc, &sc->regIDs[i], &tempID[i]);
				}
				for (uint64_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
					PfDeallocateContainer(sc, &tempID[i]);
				}
				free(tempID);
				tempID = 0;
			}
			else
			{
				sc->res = VKFFT_ERROR_MALLOC_FAILED;
				return;
			}
		}
	}
	else {
		if (sc->useDisableThreads) {
			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
		}

		if (((sc->actualInverse) && (sc->normalize)) || ((sc->convolutionStep || sc->useBluesteinFFT) && (stageAngle->data.d > 0))) {
			for (uint64_t i = 0; i < (uint64_t)logicalStoragePerThread.data.i; i++) {
				PfContainer id;
				id.type = 31;
				id.data.i = (i / logicalRegistersPerThread.data.i) * sc->registers_per_thread + i % logicalRegistersPerThread.data.i;
				
				if (normalizationValue.data.i != 1) {
					PfMul(sc, &sc->regIDs[id.data.i], &sc->regIDs[id.data.i], &stageNormalization, 0);
				}
				/*sc->tempLen = sprintf(sc->tempStr, "\
	temp%s = temp%s%s;\n", sc->regIDs[(i / logicalRegistersPerThread) * sc->registers_per_thread + i % logicalRegistersPerThread], sc->regIDs[(i / logicalRegistersPerThread) * sc->registers_per_thread + i % logicalRegistersPerThread], stageNormalization);*/
			}
		}

		if (sc->useDisableThreads) {
			PfIf_end(sc);
		}

	}
	return;
}
static inline void appendRadixShuffleStrided(VkFFTSpecializationConstantsLayout* sc, PfContainer* stageSize, PfContainer* stageSizeSum, PfContainer* stageAngle, PfContainer* stageRadix, PfContainer* stageRadixNext) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_int1 = VKFFT_ZERO_INIT;
	temp_int1.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 32;

	PfContainer stageNormalization;
	stageNormalization.type = 32;
	PfContainer normalizationValue;
	normalizationValue.type = 31;
	normalizationValue.data.i = 1;

	if ((((sc->actualInverse) && (sc->normalize)) || (sc->convolutionStep && (stageAngle->data.d > 0))) && (stageSize->data.i == 1) && (sc->axis_upload_id == 0) && (!(sc->useBluesteinFFT && (stageAngle->data.d < 0)))) {
		if ((sc->performDCT) && (sc->actualInverse)) {
			if (sc->performDCT == 1)
				normalizationValue.data.i = (sc->sourceFFTSize.data.i - 1) * 2;
			else
				normalizationValue.data.i = sc->sourceFFTSize.data.i * 2;
		}
		else
			normalizationValue.data.i = sc->sourceFFTSize.data.i;
	}
	if (sc->useBluesteinFFT && (stageAngle->data.d > 0) && (stageSize->data.i == 1) && (sc->axis_upload_id == 0)) {
		normalizationValue.data.i *= sc->fft_dim_full.data.i;
	}
	if (normalizationValue.data.i != 1) {
		stageNormalization.data.d = 1.0 / (long double)(normalizationValue.data.i);
	}
	char tempNum[50] = "";

	PfContainer logicalStoragePerThread;
	logicalStoragePerThread.type = 31;
	logicalStoragePerThread.data.i = sc->registers_per_thread_per_radix[stageRadix->data.i] * sc->registerBoost;// (sc->registers_per_thread % stageRadix->data.i == 0) ? sc->registers_per_thread * sc->registerBoost : sc->min_registers_per_thread * sc->registerBoost;
	PfContainer logicalStoragePerThreadNext;
	logicalStoragePerThreadNext.type = 31;
	logicalStoragePerThreadNext.data.i = sc->registers_per_thread_per_radix[stageRadixNext->data.i] * sc->registerBoost;// (sc->registers_per_thread % stageRadixNext->data.i == 0) ? sc->registers_per_thread * sc->registerBoost : sc->min_registers_per_thread * sc->registerBoost;
	PfContainer logicalRegistersPerThread;
	logicalRegistersPerThread.type = 31;
	logicalRegistersPerThread.data.i = sc->registers_per_thread_per_radix[stageRadix->data.i];// (sc->registers_per_thread % stageRadix->data.i == 0) ? sc->registers_per_thread : sc->min_registers_per_thread;
	PfContainer logicalRegistersPerThreadNext;
	logicalRegistersPerThreadNext.type = 31;
	logicalRegistersPerThreadNext.data.i = sc->registers_per_thread_per_radix[stageRadixNext->data.i];// (sc->registers_per_thread % stageRadixNext->data.i == 0) ? sc->registers_per_thread : sc->min_registers_per_thread;

	PfContainer logicalGroupSize;
	logicalGroupSize.type = 31;
	PfDivCeil(sc, &logicalGroupSize, &sc->fftDim, &logicalStoragePerThread);
	PfContainer logicalGroupSizeNext;
	logicalGroupSizeNext.type = 31;
	PfDivCeil(sc, &logicalGroupSizeNext, &sc->fftDim, &logicalStoragePerThreadNext);

	if ((!((sc->writeFromRegisters == 1) && (stageSize->data.i == sc->fftDim.data.i / stageRadix->data.i) && (!(((sc->convolutionStep) || (sc->useBluesteinFFT && sc->BluesteinConvolutionStep)) && (stageAngle->data.d < 0) && ((sc->matrixConvolution > 1) || (sc->numKernels.data.i > 1)))))) && (((sc->axis_id == 0) && (sc->axis_upload_id == 0)) || (sc->localSize[1].data.i * logicalStoragePerThread.data.i > sc->fftDim.data.i) || (stageSize->data.i < sc->fftDim.data.i / stageRadix->data.i) || ((sc->convolutionStep) && ((sc->matrixConvolution > 1) || (sc->numKernels.data.i > 1)) && (stageAngle->data.d < 0)) || (sc->performDCT)))
	{
		appendBarrierVkFFT(sc);
	}
	if (stageSize->data.i == sc->fftDim.data.i / stageRadix->data.i) {
		PfMov(sc, &sc->sharedStride, &sc->sharedStrideReadWriteConflict);
	}
	//if ((sc->localSize[0] * logicalStoragePerThread > sc->fftDim) || (stageSize->data.i < sc->fftDim / stageRadix->data.i) || ((sc->reorderFourStep) && (sc->fftDim < sc->fft_dim_full) && (sc->localSize[1] > 1)) || (sc->localSize[1] > 1) || ((sc->performR2C) && (!sc->actualInverse) && (sc->axis_id == 0)) || ((sc->convolutionStep) && ((sc->matrixConvolution > 1) || (sc->numKernels > 1)) && (stageAngle->data.d < 0)) || (sc->registerBoost > 1) || (sc->performDCT)) {
	if ((!((sc->writeFromRegisters == 1) && (stageSize->data.i == sc->fftDim.data.i / stageRadix->data.i) && (!(((sc->convolutionStep) || (sc->useBluesteinFFT && sc->BluesteinConvolutionStep)) && (stageAngle->data.d < 0) && ((sc->matrixConvolution > 1) || (sc->numKernels.data.i > 1)))))) && (((sc->axis_id == 0) && (sc->axis_upload_id == 0)) || (sc->localSize[1].data.i * logicalStoragePerThread.data.i > sc->fftDim.data.i) || (stageSize->data.i < sc->fftDim.data.i / stageRadix->data.i) || ((sc->convolutionStep) && ((sc->matrixConvolution > 1) || (sc->numKernels.data.i > 1)) && (stageAngle->data.d < 0)) || (sc->performDCT))) {
		if (!((sc->registerBoost > 1) && (stageSize->data.i * stageRadix->data.i == sc->fftDim.data.i / sc->stageRadix[sc->numStages - 1]) && (sc->stageRadix[sc->numStages - 1] == sc->registerBoost))) {
			PfContainer* tempID;
			tempID = (PfContainer*)calloc(sc->registers_per_thread * sc->registerBoost, sizeof(PfContainer));
			if (tempID) {
				for (uint64_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
					PfAllocateContainerFlexible(sc, &tempID[i], 50);
					tempID[i].type = sc->regIDs[0].type;
				}

				for (uint64_t k = 0; k < sc->registerBoost; ++k) {
					uint64_t t = 0;
					if (sc->registerBoost > 1) {
						appendBarrierVkFFT(sc);

						if (logicalGroupSize.data.i * logicalStoragePerThread.data.i > sc->fftDim.data.i) {
							PfDivCeil(sc, &temp_int, &sc->fftDim, &logicalStoragePerThread);
							PfIf_lt_start(sc, &sc->gl_LocalInvocationID_y, &temp_int);
						}
						if (sc->useDisableThreads) {
							temp_int.data.i = 0;
							PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
						}
					}
					else {
						if (sc->useDisableThreads) {
							temp_int.data.i = 0;
							PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
						}
					}
					if (logicalGroupSize.data.i != sc->localSize[1].data.i) {
						PfIf_lt_start(sc, &sc->gl_LocalInvocationID_y, &logicalGroupSize);
					}
					for (uint64_t j = 0; j < (uint64_t)logicalRegistersPerThread.data.i / stageRadix->data.i; j++) {
						if (logicalGroupSize.data.i * ((int64_t)(j + k * logicalRegistersPerThread.data.i / stageRadix->data.i) * stageRadix->data.i) <= sc->fftDim.data.i) {
							if (logicalGroupSize.data.i * ((int64_t)(1 + j + k * logicalRegistersPerThread.data.i / stageRadix->data.i) * stageRadix->data.i) > sc->fftDim.data.i) {
								PfContainer current_group_cut;
								current_group_cut.type = 31;
								current_group_cut.data.i = sc->fftDim.data.i / stageRadix->data.i - (j + k * logicalRegistersPerThread.data.i / stageRadix->data.i) * logicalGroupSize.data.i;
								PfIf_lt_start(sc, &sc->gl_LocalInvocationID_y, &current_group_cut);
							}
							temp_int.data.i = j * logicalGroupSize.data.i;
							PfAdd(sc, &sc->stageInvocationID, &sc->gl_LocalInvocationID_y, &temp_int);
							PfMov(sc, &sc->blockInvocationID, &sc->stageInvocationID);
							PfMod(sc, &sc->stageInvocationID, &sc->stageInvocationID, stageSize);
							PfSub(sc, &sc->blockInvocationID, &sc->blockInvocationID, &sc->stageInvocationID);
							PfMul(sc, &sc->inoutID, &sc->blockInvocationID, stageRadix, 0);
							PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->stageInvocationID);
						}
						/*sc->tempLen = sprintf(sc->tempStr, "\
		stageInvocationID = (gl_LocalInvocationID.x + %" PRIu64 ") %% (%" PRIu64 ");\n\
		blockInvocationID = (gl_LocalInvocationID.x + %" PRIu64 ") - stageInvocationID;\n\
		inoutID = stageInvocationID + blockInvocationID * %" PRIu64 ";\n", j * logicalGroupSize, stageSize->data.i, j * logicalGroupSize, stageRadix->data.i);*/

						for (uint64_t i = 0; i < (uint64_t)stageRadix->data.i; i++) {
							PfContainer id;
							id.type = 31;
							id.data.i = j + k * logicalRegistersPerThread.data.i / stageRadix->data.i + i * logicalStoragePerThread.data.i / stageRadix->data.i;
							id.data.i = (id.data.i / logicalRegistersPerThread.data.i) * sc->registers_per_thread + id.data.i % logicalRegistersPerThread.data.i;
							PfCopyContainer(sc, &tempID[t + k * sc->registers_per_thread], &sc->regIDs[id.data.i]);
							t++;
							if (logicalGroupSize.data.i * ((int64_t)(j + k * logicalRegistersPerThread.data.i / stageRadix->data.i) * stageRadix->data.i) <= sc->fftDim.data.i) {
								temp_int.data.i = i * stageSize->data.i;
								PfAdd(sc, &sc->sdataID, &sc->inoutID, &temp_int);

								PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
								PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);
								//sprintf(sc->sdataID, "sharedStride * gl_LocalInvocationID.y + inoutID + %" PRIu64 "", i * stageSize->data.i);
								if (normalizationValue.data.i != 1) {
									PfMul(sc, &sc->regIDs[id.data.i], &sc->regIDs[id.data.i], &stageNormalization, 0);
								}
								appendRegistersToShared(sc, &sc->sdataID, &sc->regIDs[id.data.i]);

							}
							/*sc->tempLen = sprintf(sc->tempStr, "\
sdata[sharedStride * gl_LocalInvocationID.y + inoutID + %" PRIu64 "] = temp%s%s;\n", i * stageSize->data.i, sc->regIDs[id], stageNormalization);*/
						}
						if (logicalGroupSize.data.i * ((int64_t)(j + k * logicalRegistersPerThread.data.i / stageRadix->data.i) * stageRadix->data.i) <= sc->fftDim.data.i) {
							if (logicalGroupSize.data.i * ((int64_t)(1 + j + k * logicalRegistersPerThread.data.i / stageRadix->data.i) * stageRadix->data.i) > sc->fftDim.data.i) {
								PfIf_end(sc);
							}
						}
					}
					if (logicalGroupSize.data.i != sc->localSize[1].data.i) {
						PfIf_end(sc);
					}
					for (uint64_t j = logicalRegistersPerThread.data.i; j < sc->registers_per_thread; j++) {
						PfCopyContainer(sc, &tempID[t + k * sc->registers_per_thread], &sc->regIDs[t + k * sc->registers_per_thread]);
						t++;
					}
					t = 0;
					if (sc->useDisableThreads) {
						PfIf_end(sc);
					}
					if (sc->registerBoost > 1) {
						//registerBoost
						if (logicalGroupSize.data.i * logicalStoragePerThread.data.i > sc->fftDim.data.i)
						{
							PfIf_end(sc);
						}
						appendBarrierVkFFT(sc);

						if (sc->useDisableThreads) {
							temp_int.data.i = 0;
							PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
						}

						if (logicalGroupSize.data.i * logicalRegistersPerThreadNext.data.i > sc->fftDim.data.i) {
							PfDivCeil(sc, &temp_int, &sc->fftDim, &logicalRegistersPerThreadNext);
							PfIf_lt_start(sc, &sc->gl_LocalInvocationID_y, &temp_int);
						}
						for (uint64_t j = 0; j < (uint64_t)logicalRegistersPerThreadNext.data.i / stageRadixNext->data.i; j++) {
							for (uint64_t i = 0; i < (uint64_t)stageRadixNext->data.i; i++) {
								PfContainer id;
								id.type = 31;
								id.data.i = j + k * logicalRegistersPerThreadNext.data.i / stageRadixNext->data.i + i * logicalStoragePerThreadNext.data.i / stageRadixNext->data.i;
								id.data.i = (id.data.i / logicalRegistersPerThreadNext.data.i) * sc->registers_per_thread + id.data.i % logicalRegistersPerThreadNext.data.i;
								//resID[t + k * sc->registers_per_thread] = sc->regIDs[id];
								temp_int.data.i = t * logicalGroupSizeNext.data.i;
								PfAdd(sc, &sc->sdataID, &sc->gl_LocalInvocationID_y, &temp_int);
								PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
								PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);

								//sprintf(sc->sdataID, "sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + %" PRIu64 "", t * logicalGroupSizeNext);
								appendSharedToRegisters(sc, &tempID[t + k * sc->registers_per_thread], &sc->sdataID);

								t++;
							}

						}
						
						if (sc->useDisableThreads) {
							PfIf_end(sc);
						}
						if (logicalGroupSize.data.i * logicalRegistersPerThreadNext.data.i > sc->fftDim.data.i)
						{
							PfIf_end(sc);
						}
					}

				}
				for (uint64_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
					PfCopyContainer(sc, &sc->regIDs[i], &tempID[i]);
				}
				for (uint64_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
					PfDeallocateContainer(sc, &tempID[i]);
				}
				free(tempID);
				tempID = 0;
			}
			else {
				sc->res = VKFFT_ERROR_MALLOC_FAILED;
				return;
			}
		}
		else {
			//registerBoost
			PfContainer* tempID;
			tempID = (PfContainer*)calloc(sc->registers_per_thread * sc->registerBoost, sizeof(PfContainer));
			if (tempID) {
				//resID = (char**)malloc(sizeof(char*) * sc->registers_per_thread * sc->registerBoost);
				for (uint64_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
					PfAllocateContainerFlexible(sc, &tempID[i], 50);
					tempID[i].type = sc->regIDs[0].type;
				}
				if (sc->useDisableThreads) {
					temp_int.data.i = 0;
					PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
				}
				for (uint64_t k = 0; k < sc->registerBoost; ++k) {
					for (uint64_t j = 0; j < (uint64_t)logicalRegistersPerThread.data.i / stageRadix->data.i; j++) {
						for (uint64_t i = 0; i < (uint64_t)stageRadix->data.i; i++) {
							PfContainer id;
							id.type = 31;
							id.data.i = j + k * logicalRegistersPerThread.data.i / stageRadix->data.i + i * logicalStoragePerThread.data.i / stageRadix->data.i;
							id.data.i = (id.data.i / logicalRegistersPerThread.data.i) * sc->registers_per_thread + id.data.i % logicalRegistersPerThread.data.i;
							PfCopyContainer(sc, &tempID[j + i * logicalRegistersPerThread.data.i / stageRadix->data.i + k * sc->registers_per_thread], &sc->regIDs[id.data.i]);
						}
					}
					for (uint64_t j = logicalRegistersPerThread.data.i; j < sc->registers_per_thread; j++) {
						PfCopyContainer(sc, &tempID[j + k * sc->registers_per_thread], &sc->regIDs[j + k * sc->registers_per_thread]);
					}
				}
				for (uint64_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
					PfCopyContainer(sc, &sc->regIDs[i], &tempID[i]);
				}
				for (uint64_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
					PfDeallocateContainer(sc, &tempID[i]);
				}
				free(tempID);
				tempID = 0;
			}
			else
			{
				sc->res = VKFFT_ERROR_MALLOC_FAILED;
				return;
			}
		}
	}
	else {
		
		if (sc->useDisableThreads) {
			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
		}
		if (sc->localSize[1].data.i * logicalStoragePerThread.data.i > sc->fftDim.data.i) {
			PfDivCeil(sc, &temp_int, &sc->fftDim, &logicalStoragePerThread);
			PfIf_lt_start(sc, &sc->gl_LocalInvocationID_y, &temp_int);
		}
		if (((sc->actualInverse) && (sc->normalize)) || ((sc->convolutionStep || sc->useBluesteinFFT) && (stageAngle->data.d > 0))) {
			for (uint64_t i = 0; i < (uint64_t)logicalStoragePerThread.data.i; i++) {
				PfContainer id;
				id.type = 31;
				id.data.i = (i / logicalRegistersPerThread.data.i) * sc->registers_per_thread + i % logicalRegistersPerThread.data.i;

				if (normalizationValue.data.i != 1) {
					PfMul(sc, &sc->regIDs[id.data.i], &sc->regIDs[id.data.i], &stageNormalization, 0);
				}
				/*sc->tempLen = sprintf(sc->tempStr, "\
	temp%s = temp%s%s;\n", sc->regIDs[(i / logicalRegistersPerThread) * sc->registers_per_thread + i % logicalRegistersPerThread], sc->regIDs[(i / logicalRegistersPerThread) * sc->registers_per_thread + i % logicalRegistersPerThread], stageNormalization);*/
			}
		}
		if (sc->localSize[1].data.i * logicalStoragePerThread.data.i > sc->fftDim.data.i) {
			PfIf_end(sc);
		}
		if (sc->useDisableThreads) {
			PfIf_end(sc);
		}

	}
	return;
}
static inline void appendRadixShuffle(VkFFTSpecializationConstantsLayout* sc, PfContainer* stageSize, PfContainer* stageSizeSum, PfContainer* stageAngle, PfContainer* stageRadix, PfContainer* stageRadixNext, uint64_t stageID, uint64_t shuffleType) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (sc->rader_generator[stageID] == 0) {
		switch (shuffleType) {
		case 0: case 5: case 6: case 110: case 120: case 130: case 140: case 142: case 144: {
			appendRadixShuffleNonStrided(sc, stageSize, stageSizeSum, stageAngle, stageRadix, stageRadixNext);
			
			//appendBarrierVkFFT(sc, 1);
			break;
		}
		case 1: case 2: case 111: case 121: case 131: case 141: case 143: case 145: {
			appendRadixShuffleStrided(sc, stageSize, stageSizeSum, stageAngle, stageRadix, stageRadixNext);
			
			//appendBarrierVkFFT(sc, 1);
			break;
		}
		}
	}
	return;
}

#endif
