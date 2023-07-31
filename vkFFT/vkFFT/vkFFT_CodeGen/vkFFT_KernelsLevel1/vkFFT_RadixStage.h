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
#ifndef VKFFT_RADIXSTAGE_H
#define VKFFT_RADIXSTAGE_H

#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_StringManagement/vkFFT_StringManager.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_MathUtils/vkFFT_MathUtils.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryTransfers/vkFFT_Transfers.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_KernelUtils.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_Zeropad.h"

#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel1/vkFFT_RadixKernels.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel1/vkFFT_RaderKernels.h"

static inline void appendRadixStageNonStrided(VkFFTSpecializationConstantsLayout* sc, PfContainer* stageSize, PfContainer* stageSizeSum, PfContainer* stageAngle, PfContainer* stageRadix) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_int1 = VKFFT_ZERO_INIT;
	temp_int1.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 32;
	/*char convolutionInverse[10] = "";
	if (sc->convolutionStep) {
		if (stageAngle->data.d < 0)
			sprintf(convolutionInverse, ", 0");
		else
			sprintf(convolutionInverse, ", 1");
	}*/
	PfContainer logicalStoragePerThread;
	logicalStoragePerThread.type = 31;
	logicalStoragePerThread.data.i = sc->registers_per_thread_per_radix[stageRadix->data.i] * sc->registerBoost;// (sc->registers_per_thread % stageRadix->data.i == 0) ? sc->registers_per_thread * sc->registerBoost : sc->min_registers_per_thread * sc->registerBoost;
	PfContainer logicalRegistersPerThread;
	logicalRegistersPerThread.type = 31;
	logicalRegistersPerThread.data.i = sc->registers_per_thread_per_radix[stageRadix->data.i];// (sc->registers_per_thread % stageRadix->data.i == 0) ? sc->registers_per_thread : sc->min_registers_per_thread;
	PfContainer logicalGroupSize;
	logicalGroupSize.type = 31;
	PfDivCeil(sc, &logicalGroupSize, &sc->fftDim, &logicalStoragePerThread);

	if ((!((sc->readToRegisters == 1) && (stageSize->data.i == 1) && (!(((sc->convolutionStep) || (sc->useBluesteinFFT && sc->BluesteinConvolutionStep)) && (stageAngle->data.d > 0) && ((sc->matrixConvolution > 1) || (sc->numKernels.data.i > 1)))))) && ((sc->localSize[0].data.i * logicalStoragePerThread.data.i > sc->fftDim.data.i) || (stageSize->data.i > 1) || ((sc->localSize[1].data.i > 1) && (!(sc->performR2C && (sc->actualInverse)))) || ((sc->convolutionStep) && ((sc->matrixConvolution > 1) || (sc->numKernels.data.i > 1)) && (stageAngle->data.d > 0)) || (sc->performDCT)))
	{
		appendBarrierVkFFT(sc);
		
	}
	
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	//upload second stage of LUT to sm
	uint64_t numLUTelementsStage = 0;
	switch (stageRadix->data.i) {
	case 2:
		numLUTelementsStage = 1;
		break;
	case 4:
		numLUTelementsStage = 2;
		break;
	case 8:
		numLUTelementsStage = 3;
		break;
	case 16:
		numLUTelementsStage = 4;
		break;
	case 32:
		numLUTelementsStage = 5;
		break;
	default:
		if (stageRadix->data.i < sc->fixMinRaderPrimeMult)
			numLUTelementsStage = stageRadix->data.i - 1;
		else
			numLUTelementsStage = stageRadix->data.i;
		break;
	}
	if ((sc->LUT) && (stageSize->data.i > 1) && ((((numLUTelementsStage >= 4) && (sc->fftDim.data.i >= 1024)) || (((numLUTelementsStage >= 3) && (sc->fftDim.data.i < 1024)))) || (logicalRegistersPerThread.data.i / stageRadix->data.i > 1)) && (sc->registerBoost == 1) && (stageSize->data.i < sc->warpSize))
		sc->useCoalescedLUTUploadToSM = 1;
	else
		sc->useCoalescedLUTUploadToSM = 0;

	for (uint64_t k = 0; k < sc->registerBoost; k++) {
		if (logicalGroupSize.data.i != sc->localSize[0].data.i) {
			PfIf_lt_start(sc, &sc->gl_LocalInvocationID_x, &logicalGroupSize);
		}
		for (uint64_t j = 0; j < (uint64_t)(logicalRegistersPerThread.data.i / stageRadix->data.i); j++) {
			if (logicalGroupSize.data.i * ((int64_t)(j + k * logicalRegistersPerThread.data.i / stageRadix->data.i) * stageRadix->data.i) > sc->fftDim.data.i) continue;
			if (logicalGroupSize.data.i * ((int64_t)(1 + j + k * logicalRegistersPerThread.data.i / stageRadix->data.i) * stageRadix->data.i) > sc->fftDim.data.i) {
				PfContainer current_group_cut;
				current_group_cut.type = 31;
				current_group_cut.data.i = sc->fftDim.data.i / stageRadix->data.i - (j + k * logicalRegistersPerThread.data.i / stageRadix->data.i) * logicalGroupSize.data.i;
				PfIf_lt_start(sc, &sc->gl_LocalInvocationID_x, &current_group_cut);
			}

			temp_int.data.i = (j + k * logicalRegistersPerThread.data.i / stageRadix->data.i) * logicalGroupSize.data.i;
			PfAdd(sc, &sc->stageInvocationID, &sc->gl_LocalInvocationID_x, &temp_int);
			temp_int.data.i = stageSize->data.i;
			PfMod(sc, &sc->stageInvocationID, &sc->stageInvocationID, &temp_int);
			
			
			if (sc->LUT) {
				temp_int.data.i = stageSizeSum->data.i;
				PfAdd(sc, &sc->LUTId, &sc->stageInvocationID, &temp_int);
			}
			else {
				temp_double.data.d = stageAngle->data.d;
				PfMul(sc, &sc->angle, &sc->stageInvocationID, &temp_double, 0);
			}
				
			if ((!((sc->readToRegisters == 1) && (stageSize->data.i == 1) && (!(((sc->convolutionStep) || (sc->useBluesteinFFT && sc->BluesteinConvolutionStep)) && (stageAngle->data.d > 0) && ((sc->matrixConvolution > 1) || (sc->numKernels.data.i > 1)))))) && ((sc->registerBoost == 1) && ((sc->localSize[0].data.i * logicalStoragePerThread.data.i > sc->fftDim.data.i) || (stageSize->data.i > 1) || ((sc->localSize[1].data.i > 1) && (!(sc->performR2C && (sc->actualInverse)))) || ((sc->convolutionStep) && ((sc->matrixConvolution > 1) || (sc->numKernels.data.i > 1)) && (stageAngle->data.d > 0)) || (sc->performDCT)))) {
				//if(sc->readToRegisters==0){
				for (uint64_t i = 0; i < (uint64_t)stageRadix->data.i; i++) {
					PfContainer id;
					id.type = 31;
					id.data.i = j + i * logicalRegistersPerThread.data.i / stageRadix->data.i;
					id.data.i = (id.data.i / logicalRegistersPerThread.data.i) * sc->registers_per_thread + id.data.i % logicalRegistersPerThread.data.i;

					temp_int.data.i = j * logicalGroupSize.data.i + i * sc->fftDim.data.i / stageRadix->data.i;
					PfAdd(sc, &sc->sdataID, &sc->gl_LocalInvocationID_x, &temp_int);					

					if (sc->resolveBankConflictFirstStages == 1) {
						temp_int.data.i = sc->numSharedBanks / 2;
						PfDiv(sc, &sc->tempInt, &sc->sdataID, &temp_int);
						temp_int.data.i = sc->numSharedBanks / 2 + 1;
						PfMul(sc, &sc->tempInt, &sc->tempInt, &temp_int, 0);
						temp_int.data.i = sc->numSharedBanks / 2;
						PfMod(sc, &sc->sdataID, &sc->sdataID, &temp_int);
						PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
					}

					if (sc->localSize[1].data.i > 1) {
						PfMul(sc, &sc->tempInt, &sc->sharedStride, &sc->gl_LocalInvocationID_y, 0);
						PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
					}

					appendSharedToRegisters(sc, &sc->regIDs[id.data.i], &sc->sdataID);
					
				}
			}
			if (!sc->useCoalescedLUTUploadToSM) {
				PfContainer* regID = (PfContainer*)calloc(stageRadix->data.i, sizeof(PfContainer));
				if (regID) {
					for (uint64_t i = 0; i < (uint64_t)stageRadix->data.i; i++) {
						PfContainer id;
						id.type = 31;
						id.data.i = j + k * logicalRegistersPerThread.data.i / stageRadix->data.i + i * logicalStoragePerThread.data.i / stageRadix->data.i;
						id.data.i = (id.data.i / logicalRegistersPerThread.data.i) * sc->registers_per_thread + id.data.i % logicalRegistersPerThread.data.i;
						PfAllocateContainerFlexible(sc, &regID[i], 50);
						regID[i].type = sc->regIDs[id.data.i].type;
						PfCopyContainer(sc, &regID[i], &sc->regIDs[id.data.i]);
					}

					inlineRadixKernelVkFFT(sc, stageRadix->data.i, stageSize->data.i, stageSizeSum->data.i, stageAngle->data.d, regID);
					
					for (uint64_t i = 0; i < (uint64_t)stageRadix->data.i; i++) {
						PfContainer id;
						id.type = 31; 
						id.data.i = j + k * logicalRegistersPerThread.data.i / stageRadix->data.i + i * logicalStoragePerThread.data.i / stageRadix->data.i;
						id.data.i = (id.data.i / logicalRegistersPerThread.data.i) * sc->registers_per_thread + id.data.i % logicalRegistersPerThread.data.i;
						PfCopyContainer(sc, &sc->regIDs[id.data.i], &regID[i]);
						PfDeallocateContainer(sc, &regID[i]);
					}
					free(regID);
					regID = 0;
				}
				else {
					sc->res = VKFFT_ERROR_MALLOC_FAILED;
					return;
				}
			}

			if (logicalGroupSize.data.i * ((int64_t)(1 + j + k * logicalRegistersPerThread.data.i / stageRadix->data.i) * stageRadix->data.i) > sc->fftDim.data.i) {
				PfIf_end(sc);
			}
		}
		if (logicalGroupSize.data.i != sc->localSize[0].data.i) {
			PfIf_end(sc);			
		}
		if (sc->useCoalescedLUTUploadToSM) {
			if (sc->useDisableThreads) {
				PfIf_end(sc);
			}
			
			appendBarrierVkFFT(sc);
			

			sc->useCoalescedLUTUploadToSM = 1;
			PfMov(sc, &sc->sdataID, &sc->gl_LocalInvocationID_x);
			
			if (sc->localSize[1].data.i > 1) {
				PfMul(sc, &sc->tempInt, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);
				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
			}

			for (uint64_t i = 0; i < (uint64_t)ceil(numLUTelementsStage * stageSize->data.i / ((double)sc->localSize[0].data.i * sc->localSize[1].data.i)); i++) {
				if (i > 0) {
					temp_int.data.i = sc->localSize[0].data.i * sc->localSize[1].data.i;
					PfAdd(sc, &sc->sdataID, &sc->sdataID, &temp_int);					
				}
				if (i == (uint64_t)ceil(numLUTelementsStage * stageSize->data.i / ((double)sc->localSize[0].data.i * sc->localSize[1].data.i)) - 1) {
					temp_int.data.i = numLUTelementsStage * stageSize->data.i;
					PfIf_lt_start(sc, &sc->sdataID, &temp_int);					
				}
				temp_int.data.i = stageSizeSum->data.i;
				PfAdd(sc, &sc->inoutID, &sc->sdataID, &temp_int);
				appendGlobalToShared(sc, &sc->sdataID, &sc->LUTStruct, &sc->inoutID);
				
				if (i == (uint64_t)ceil(numLUTelementsStage * stageSize->data.i / ((double)sc->localSize[0].data.i * sc->localSize[1].data.i)) - 1) {
					PfIf_end(sc);
				}
			}
			appendBarrierVkFFT(sc);
			
			if (sc->useDisableThreads) {
				temp_int.data.i = 0;
				PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
			}

			if (logicalGroupSize.data.i != sc->localSize[0].data.i) {
				PfIf_lt_start(sc, &sc->gl_LocalInvocationID_x, &logicalGroupSize);
			}
			for (uint64_t j = 0; j < (uint64_t)logicalRegistersPerThread.data.i / stageRadix->data.i; j++) {
				if (logicalGroupSize.data.i * ((int64_t)(j + k * logicalRegistersPerThread.data.i / stageRadix->data.i) * stageRadix->data.i) > sc->fftDim.data.i) continue;
				if (logicalGroupSize.data.i * ((int64_t)(1 + j + k * logicalRegistersPerThread.data.i / stageRadix->data.i) * stageRadix->data.i) > sc->fftDim.data.i) {
					PfContainer current_group_cut;
					current_group_cut.type = 31;
					current_group_cut.data.i = sc->fftDim.data.i / stageRadix->data.i - (j + k * logicalRegistersPerThread.data.i / stageRadix->data.i) * logicalGroupSize.data.i;
					PfIf_lt_start(sc, &sc->gl_LocalInvocationID_x, &current_group_cut);
				}
				PfContainer* regID = (PfContainer*)calloc(stageRadix->data.i, sizeof(PfContainer));
				if (regID) {
					for (uint64_t i = 0; i < (uint64_t)stageRadix->data.i; i++) {
						PfContainer id;
						id.type = 31;
						id.data.i = j + k * logicalRegistersPerThread.data.i / stageRadix->data.i + i * logicalStoragePerThread.data.i / stageRadix->data.i;
						id.data.i = (id.data.i / logicalRegistersPerThread.data.i) * sc->registers_per_thread + id.data.i % logicalRegistersPerThread.data.i;
						PfAllocateContainerFlexible(sc, &regID[i], 50);
						regID[i].type = sc->regIDs[id.data.i].type;
						PfCopyContainer(sc, &regID[i], &sc->regIDs[id.data.i]);
					}

					temp_int.data.i = (j + k * logicalRegistersPerThread.data.i / stageRadix->data.i) * logicalGroupSize.data.i;
					PfAdd(sc, &sc->stageInvocationID, &sc->gl_LocalInvocationID_x, &temp_int);
					temp_int.data.i = stageSize->data.i;
					PfMod(sc, &sc->stageInvocationID, &sc->stageInvocationID, &temp_int);
					
					inlineRadixKernelVkFFT(sc, stageRadix->data.i, stageSize->data.i, stageSizeSum->data.i, stageAngle->data.d, regID);
					
					for (uint64_t i = 0; i < (uint64_t)stageRadix->data.i; i++) {
						PfContainer id;
						id.type = 31;
						id.data.i = j + k * logicalRegistersPerThread.data.i / stageRadix->data.i + i * logicalStoragePerThread.data.i / stageRadix->data.i;
						id.data.i = (id.data.i / logicalRegistersPerThread.data.i) * sc->registers_per_thread + id.data.i % logicalRegistersPerThread.data.i;
						PfCopyContainer(sc, &sc->regIDs[id.data.i], &regID[i]);
						PfDeallocateContainer(sc, &regID[i]);
					}
					free(regID);
					regID = 0;
				}
				else
				{
					sc->res = VKFFT_ERROR_MALLOC_FAILED;
					return;
				}
				if (logicalGroupSize.data.i * ((int64_t)(1 + j + k * logicalRegistersPerThread.data.i / stageRadix->data.i) * stageRadix->data.i) > sc->fftDim.data.i) {
					PfIf_end(sc);
				}
			}
			if (logicalGroupSize.data.i != sc->localSize[0].data.i) {
				PfIf_end(sc);
			}
		}
	}

	if (sc->useDisableThreads) {
		PfIf_end(sc);
	}
	
	return;
}

static inline void appendRadixStageStrided(VkFFTSpecializationConstantsLayout* sc, PfContainer* stageSize, PfContainer* stageSizeSum, PfContainer* stageAngle, PfContainer* stageRadix) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_int1 = VKFFT_ZERO_INIT;
	temp_int1.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 32;
	/*char convolutionInverse[10] = "";
	if (sc->convolutionStep) {
		if (stageAngle->data.d < 0)
			sprintf(convolutionInverse, ", 0");
		else
			sprintf(convolutionInverse, ", 1");
	}*/
	PfContainer logicalStoragePerThread;
	logicalStoragePerThread.type = 31;
	logicalStoragePerThread.data.i = sc->registers_per_thread_per_radix[stageRadix->data.i] * sc->registerBoost;// (sc->registers_per_thread % stageRadix->data.i == 0) ? sc->registers_per_thread * sc->registerBoost : sc->min_registers_per_thread * sc->registerBoost;
	PfContainer logicalRegistersPerThread;
	logicalRegistersPerThread.type = 31;
	logicalRegistersPerThread.data.i = sc->registers_per_thread_per_radix[stageRadix->data.i];// (sc->registers_per_thread % stageRadix->data.i == 0) ? sc->registers_per_thread : sc->min_registers_per_thread;
	PfContainer logicalGroupSize;
	logicalGroupSize.type = 31;
	PfDivCeil(sc, &logicalGroupSize, &sc->fftDim, &logicalStoragePerThread);

	if ((!((sc->readToRegisters == 1) && (stageSize->data.i == 1) && (!(((sc->convolutionStep) || (sc->useBluesteinFFT && sc->BluesteinConvolutionStep)) && (stageAngle->data.d > 0) && ((sc->matrixConvolution > 1) || (sc->numKernels.data.i > 1)))))) && (((sc->axis_id == 0) && (sc->axis_upload_id == 0) && (!(sc->performR2C && (sc->actualInverse)))) || (sc->localSize[1].data.i * logicalStoragePerThread.data.i > sc->fftDim.data.i) || (stageSize->data.i > 1) || ((sc->convolutionStep) && ((sc->matrixConvolution > 1) || (sc->numKernels.data.i > 1)) && (stageAngle->data.d > 0)) || (sc->performDCT)))
	{
		appendBarrierVkFFT(sc);

	}
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	//upload second stage of LUT to sm
	uint64_t numLUTelementsStage = 0;
	switch (stageRadix->data.i) {
	case 2:
		numLUTelementsStage = 1;
		break;
	case 4:
		numLUTelementsStage = 2;
		break;
	case 8:
		numLUTelementsStage = 3;
		break;
	case 16:
		numLUTelementsStage = 4;
		break;
	case 32:
		numLUTelementsStage = 5;
		break;
	default:
		if (stageRadix->data.i < sc->fixMinRaderPrimeMult)
			numLUTelementsStage = stageRadix->data.i - 1;
		else
			numLUTelementsStage = stageRadix->data.i;
		break;
	}
	if ((sc->LUT) && (stageSize->data.i > 1) && ((((numLUTelementsStage >= 4) && (sc->fftDim.data.i >= 1024)) || (((numLUTelementsStage >= 3) && (sc->fftDim.data.i < 1024)))) || (logicalRegistersPerThread.data.i / stageRadix->data.i > 1)) && (sc->registerBoost == 1) && (stageSize->data.i < sc->warpSize))
		sc->useCoalescedLUTUploadToSM = 1;
	else
		sc->useCoalescedLUTUploadToSM = 0;

	for (uint64_t k = 0; k < sc->registerBoost; k++) {
		if (logicalGroupSize.data.i != sc->localSize[1].data.i) {
			PfIf_lt_start(sc, &sc->gl_LocalInvocationID_y, &logicalGroupSize);
		}
		for (uint64_t j = 0; j < (uint64_t)logicalRegistersPerThread.data.i / stageRadix->data.i; j++) {
			if (logicalGroupSize.data.i * ((int64_t)(j + k * logicalRegistersPerThread.data.i / stageRadix->data.i) * stageRadix->data.i) > sc->fftDim.data.i) continue;
			if (logicalGroupSize.data.i * ((int64_t)(1 + j + k * logicalRegistersPerThread.data.i / stageRadix->data.i) * stageRadix->data.i) > sc->fftDim.data.i) {
				PfContainer current_group_cut;
				current_group_cut.type = 31;
				current_group_cut.data.i = sc->fftDim.data.i / stageRadix->data.i - (j + k * logicalRegistersPerThread.data.i / stageRadix->data.i) * logicalGroupSize.data.i;
				PfIf_lt_start(sc, &sc->gl_LocalInvocationID_y, &current_group_cut);
			}

			temp_int.data.i = (j + k * logicalRegistersPerThread.data.i / stageRadix->data.i) * logicalGroupSize.data.i;
			PfAdd(sc, &sc->stageInvocationID, &sc->gl_LocalInvocationID_y, &temp_int);
			temp_int.data.i = stageSize->data.i;
			PfMod(sc, &sc->stageInvocationID, &sc->stageInvocationID, &temp_int);


			if (sc->LUT) {
				temp_int.data.i = stageSizeSum->data.i;
				PfAdd(sc, &sc->LUTId, &sc->stageInvocationID, &temp_int);
			}
			else {
				temp_double.data.d = stageAngle->data.d;
				PfMul(sc, &sc->angle, &sc->stageInvocationID, &temp_double, 0);
			}

			if ((!((sc->readToRegisters == 1) && (stageSize->data.i == 1) && (!(((sc->convolutionStep) || (sc->useBluesteinFFT && sc->BluesteinConvolutionStep)) && (stageAngle->data.d > 0) && ((sc->matrixConvolution > 1) || (sc->numKernels.data.i > 1)))))) && ((sc->registerBoost == 1) && (((sc->axis_id == 0) && (sc->axis_upload_id == 0) && (!(sc->performR2C && (sc->actualInverse)))) || (sc->localSize[1].data.i * logicalStoragePerThread.data.i > sc->fftDim.data.i) || (stageSize->data.i > 1) || ((sc->convolutionStep) && ((sc->matrixConvolution > 1) || (sc->numKernels.data.i > 1)) && (stageAngle->data.d > 0)) || (sc->performDCT)))) {
				//if(sc->readToRegisters==0){
				for (uint64_t i = 0; i < (uint64_t)stageRadix->data.i; i++) {
					PfContainer id;
					id.type = 31;
					id.data.i = j + i * logicalRegistersPerThread.data.i / stageRadix->data.i;
					id.data.i = (id.data.i / logicalRegistersPerThread.data.i) * sc->registers_per_thread + id.data.i % logicalRegistersPerThread.data.i;

					temp_int.data.i = j * logicalGroupSize.data.i + i * sc->fftDim.data.i / stageRadix->data.i;
					PfAdd(sc, &sc->sdataID, &sc->gl_LocalInvocationID_y, &temp_int);
					PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
					PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);

					appendSharedToRegisters(sc, &sc->regIDs[id.data.i], &sc->sdataID);

				}
			}
			if (!sc->useCoalescedLUTUploadToSM) {
				PfContainer* regID = (PfContainer*)calloc(stageRadix->data.i, sizeof(PfContainer));
				if (regID) {
					for (uint64_t i = 0; i < (uint64_t)stageRadix->data.i; i++) {
						PfContainer id;
						id.type = 31;
						id.data.i = j + k * logicalRegistersPerThread.data.i / stageRadix->data.i + i * logicalStoragePerThread.data.i / stageRadix->data.i;
						id.data.i = (id.data.i / logicalRegistersPerThread.data.i) * sc->registers_per_thread + id.data.i % logicalRegistersPerThread.data.i;
						PfAllocateContainerFlexible(sc, &regID[i], 50);
						regID[i].type = sc->regIDs[id.data.i].type;
						PfCopyContainer(sc, &regID[i], &sc->regIDs[id.data.i]);
					}

					inlineRadixKernelVkFFT(sc, stageRadix->data.i, stageSize->data.i, stageSizeSum->data.i, stageAngle->data.d, regID);

					for (uint64_t i = 0; i < (uint64_t)stageRadix->data.i; i++) {
						PfContainer id;
						id.type = 31;
						id.data.i = j + k * logicalRegistersPerThread.data.i / stageRadix->data.i + i * logicalStoragePerThread.data.i / stageRadix->data.i;
						id.data.i = (id.data.i / logicalRegistersPerThread.data.i) * sc->registers_per_thread + id.data.i % logicalRegistersPerThread.data.i;
						PfCopyContainer(sc, &sc->regIDs[id.data.i], &regID[i]);
						PfDeallocateContainer(sc, &regID[i]);
					}
					free(regID);
					regID = 0;
				}
				else {
					sc->res = VKFFT_ERROR_MALLOC_FAILED;
					return;
				}
			}

			if (logicalGroupSize.data.i * ((int64_t)(1 + j + k * logicalRegistersPerThread.data.i / stageRadix->data.i) * stageRadix->data.i) > sc->fftDim.data.i) {
				PfIf_end(sc);
			}
		}
		if (logicalGroupSize.data.i != sc->localSize[1].data.i) {
			PfIf_end(sc);
		}
		if (sc->useCoalescedLUTUploadToSM) {
			if (sc->useDisableThreads) {
				PfIf_end(sc);
			}

			appendBarrierVkFFT(sc);


			sc->useCoalescedLUTUploadToSM = 1;
			PfMov(sc, &sc->sdataID, &sc->gl_LocalInvocationID_x);

			PfMul(sc, &sc->tempInt, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);
			PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);

			for (uint64_t i = 0; i < (uint64_t)ceil(numLUTelementsStage * stageSize->data.i / ((double)sc->localSize[0].data.i * sc->localSize[1].data.i)); i++) {
				if (i > 0) {
					temp_int.data.i = sc->localSize[0].data.i * sc->localSize[1].data.i;
					PfAdd(sc, &sc->sdataID, &sc->sdataID, &temp_int);
				}
				if (i == (uint64_t)ceil(numLUTelementsStage * stageSize->data.i / ((double)sc->localSize[0].data.i * sc->localSize[1].data.i)) - 1) {
					temp_int.data.i = numLUTelementsStage * stageSize->data.i;
					PfIf_lt_start(sc, &sc->sdataID, &temp_int);
				}
				temp_int.data.i = stageSizeSum->data.i;
				PfAdd(sc, &sc->inoutID, &sc->sdataID, &temp_int);
				appendGlobalToShared(sc, &sc->sdataID, &sc->LUTStruct, &sc->inoutID);

				if (i == (uint64_t)ceil(numLUTelementsStage * stageSize->data.i / ((double)sc->localSize[0].data.i * sc->localSize[1].data.i)) - 1) {
					PfIf_end(sc);
				}
			}
			appendBarrierVkFFT(sc);

			if (sc->useDisableThreads) {
				temp_int.data.i = 0;
				PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
			}

			if (logicalGroupSize.data.i != sc->localSize[1].data.i) {
				PfIf_lt_start(sc, &sc->gl_LocalInvocationID_y, &logicalGroupSize);
			}
			for (uint64_t j = 0; j < (uint64_t)logicalRegistersPerThread.data.i / stageRadix->data.i; j++) {
				if (logicalGroupSize.data.i * ((int64_t)(j + k * logicalRegistersPerThread.data.i / stageRadix->data.i) * stageRadix->data.i) > sc->fftDim.data.i) continue;
				if (logicalGroupSize.data.i * ((int64_t)(1 + j + k * logicalRegistersPerThread.data.i / stageRadix->data.i) * stageRadix->data.i) > sc->fftDim.data.i) {
					PfContainer current_group_cut;
					current_group_cut.type = 31;
					current_group_cut.data.i = sc->fftDim.data.i / stageRadix->data.i - (j + k * logicalRegistersPerThread.data.i / stageRadix->data.i) * logicalGroupSize.data.i;
					PfIf_lt_start(sc, &sc->gl_LocalInvocationID_y, &current_group_cut);
				}
				PfContainer* regID = (PfContainer*)calloc(stageRadix->data.i, sizeof(PfContainer));
				if (regID) {
					for (uint64_t i = 0; i < (uint64_t)stageRadix->data.i; i++) {
						PfContainer id;
						id.type = 31;
						id.data.i = j + k * logicalRegistersPerThread.data.i / stageRadix->data.i + i * logicalStoragePerThread.data.i / stageRadix->data.i;
						id.data.i = (id.data.i / logicalRegistersPerThread.data.i) * sc->registers_per_thread + id.data.i % logicalRegistersPerThread.data.i;
						PfAllocateContainerFlexible(sc, &regID[i], 50);
						regID[i].type = sc->regIDs[id.data.i].type;
						PfCopyContainer(sc, &regID[i], &sc->regIDs[id.data.i]);
					}

					temp_int.data.i = (j + k * logicalRegistersPerThread.data.i / stageRadix->data.i) * logicalGroupSize.data.i;
					PfAdd(sc, &sc->stageInvocationID, &sc->gl_LocalInvocationID_y, &temp_int);
					temp_int.data.i = stageSize->data.i;
					PfMod(sc, &sc->stageInvocationID, &sc->stageInvocationID, &temp_int);

					inlineRadixKernelVkFFT(sc, stageRadix->data.i, stageSize->data.i, stageSizeSum->data.i, stageAngle->data.d, regID);

					for (uint64_t i = 0; i < (uint64_t)stageRadix->data.i; i++) {
						PfContainer id;
						id.type = 31;
						id.data.i = j + k * logicalRegistersPerThread.data.i / stageRadix->data.i + i * logicalStoragePerThread.data.i / stageRadix->data.i;
						id.data.i = (id.data.i / logicalRegistersPerThread.data.i) * sc->registers_per_thread + id.data.i % logicalRegistersPerThread.data.i;
						PfCopyContainer(sc, &sc->regIDs[id.data.i], &regID[i]);
						PfDeallocateContainer(sc, &regID[i]);
					}
					free(regID);
					regID = 0;
				}
				else
				{
					sc->res = VKFFT_ERROR_MALLOC_FAILED;
					return;
				}
				if (logicalGroupSize.data.i * ((int64_t)(1 + j + k * logicalRegistersPerThread.data.i / stageRadix->data.i) * stageRadix->data.i) > sc->fftDim.data.i) {
					PfIf_end(sc);
				}
			}
			if (logicalGroupSize.data.i != sc->localSize[1].data.i) {
				PfIf_end(sc);
			}
		}
	}

	if (sc->useDisableThreads) {
		PfIf_end(sc);
	}

	if (stageSize->data.i == 1) {
		PfMov(sc, &sc->sharedStride, &sc->localSize[0]);
	}
	return;
}

static inline void appendRadixStage(VkFFTSpecializationConstantsLayout* sc, PfContainer* stageSize, PfContainer* stageSizeSum, PfContainer* stageAngle, PfContainer* stageRadix, int stageID, int shuffleType) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (sc->rader_generator[stageID]) {
		for (uint64_t i = 0; i < sc->numRaderPrimes; i++) {
			if (sc->raderContainer[i].prime == stageRadix->data.i) {
				sc->currentRaderContainer = &sc->raderContainer[i];
			}
		}
		if (sc->currentRaderContainer->type) {
			switch (shuffleType) {
			case 0: case 5: case 6: case 110: case 120: case 130: case 140: case 142: case 144: {
				appendMultRaderStage(sc, stageSize, stageSizeSum, stageAngle, stageRadix, stageID);
				break;
			}
			case 1: case 2: case 111: case 121: case 131: case 141: case 143: case 145: {
				appendMultRaderStage(sc, stageSize, stageSizeSum, stageAngle, stageRadix, stageID);
				break;
			}
			}
		}
		else {
			switch (shuffleType) {
			case 0: case 5: case 6: case 110: case 120: case 130: case 140: case 142: case 144: {
				appendFFTRaderStage(sc, stageSize, stageSizeSum, stageAngle, stageRadix, stageID);
				break;
			}
			case 1: case 2: case 111: case 121: case 131: case 141: case 143: case 145: {
				appendFFTRaderStage(sc, stageSize, stageSizeSum, stageAngle, stageRadix, stageID);	
				break;
			}
			}
		}
	}
	else {
		switch (shuffleType) {
		case 0: case 5: case 6: case 110: case 120: case 130: case 140: case 142: case 144: {
			appendRadixStageNonStrided(sc, stageSize, stageSizeSum, stageAngle, stageRadix);
			
			break;
		}
		case 1: case 2: case 111: case 121: case 131: case 141: case 143: case 145: {
			appendRadixStageStrided(sc, stageSize, stageSizeSum, stageAngle, stageRadix);
			
			break;
		}
		}
	}
	return;
}

#endif
