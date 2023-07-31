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
#ifndef VKFFT_SHAREDMEMORY_H
#define VKFFT_SHAREDMEMORY_H
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_StringManagement/vkFFT_StringManager.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_MathUtils/vkFFT_MathUtils.h"

static inline void appendSharedMemoryVkFFT(VkFFTSpecializationConstantsLayout* sc, int type) {
	if (sc->res != VKFFT_SUCCESS) return;
	
	if (sc->useRaderMult) {
		sc->sharedMemSize -= (int)(sc->additionalRaderSharedSize.data.i * sc->complexSize);
		sc->sharedMemSizePow2 -= (int)(sc->additionalRaderSharedSize.data.i * sc->complexSize);
	}
	PfContainer maxSequenceSharedMemory;
	maxSequenceSharedMemory.type = 31;
	maxSequenceSharedMemory.data.i = sc->sharedMemSize / sc->complexSize;
	//maxSequenceSharedMemoryPow2 = sc->sharedMemSizePow2 / sc->complexSize;
	uint64_t additionalR2Cshared = 0;
	if ((sc->performR2C || ((sc->performDCT == 2) || ((sc->performDCT == 4) && ((sc->fftDim.data.i % 2) != 0)))) && (sc->mergeSequencesR2C) && (sc->axis_id == 0) && (!sc->performR2CmultiUpload)) {
		additionalR2Cshared = (sc->fftDim.data.i % 2 == 0) ? 2 : 1;
		if ((sc->performDCT == 2) || ((sc->performDCT == 4) && ((sc->fftDim.data.i % 2) != 0))) additionalR2Cshared = 1;
	}
	switch (type) {
	case 0: case 5: case 6: case 110: case 120: case 130: case 140: case 142: case 144://single_c2c + single_r2c
	{
		sc->resolveBankConflictFirstStages = 0;
		sc->sharedStrideBankConflictFirstStages.type = 31;
		sc->sharedStrideBankConflictFirstStages.data.i = ((sc->fftDim.data.i > sc->numSharedBanks / 2) && ((sc->fftDim.data.i & (sc->fftDim.data.i - 1)) == 0)) ? (sc->fftDim.data.i / sc->registerBoost + additionalR2Cshared) * (sc->numSharedBanks / 2 + 1) / (sc->numSharedBanks / 2) : sc->fftDim.data.i / sc->registerBoost + additionalR2Cshared;
		sc->sharedStrideReadWriteConflict.type = 31;
		sc->sharedStrideReadWriteConflict.data.i = ((sc->numSharedBanks / 2 <= sc->localSize[1].data.i)) ? sc->fftDim.data.i / sc->registerBoost + additionalR2Cshared + 1 : sc->fftDim.data.i / sc->registerBoost + additionalR2Cshared + (sc->numSharedBanks / 2) / sc->localSize[1].data.i;
		if ((uint64_t)sc->sharedStrideReadWriteConflict.data.i < (sc->fftDim.data.i / sc->registerBoost + additionalR2Cshared)) sc->sharedStrideReadWriteConflict.data.i = sc->fftDim.data.i / sc->registerBoost + additionalR2Cshared;
		if (sc->useRaderFFT) {
			uint64_t max_stride = sc->fftDim.data.i / sc->registerBoost + additionalR2Cshared;
			uint64_t max_shift = 0;
			for (uint64_t i = 0; i < sc->numRaderPrimes; i++) {

				for (uint64_t j = 0; j < sc->raderContainer[i].numStages; j++) {
					if (sc->raderContainer[i].containerFFTNum < 8) {
						uint64_t subLogicalGroupSize = (uint64_t)ceil(sc->raderContainer[i].containerFFTDim / (double)sc->raderContainer[i].registers_per_thread_per_radix[sc->raderContainer[i].stageRadix[j]]); // hopefully it is not <1, will fix 
						uint64_t shift = (subLogicalGroupSize > (sc->raderContainer[i].containerFFTDim % (sc->numSharedBanks / 2))) ? subLogicalGroupSize - sc->raderContainer[i].containerFFTDim % (sc->numSharedBanks / 2) : 0;
						if (j == 0) shift = (sc->raderContainer[i].containerFFTDim % (sc->numSharedBanks / 2)) ? 0 : 1;
						uint64_t loc_stride = sc->raderContainer[i].containerFFTDim + shift;
						if (sc->raderContainer[i].containerFFTNum * (loc_stride + 1) > max_stride) {
							max_stride = sc->raderContainer[i].containerFFTNum * (loc_stride + 1);
							if (shift > max_shift) max_shift = shift;
						}
					}
				}
			}
			sc->sharedShiftRaderFFT.type = 31;
			sc->sharedShiftRaderFFT.data.i = max_shift;
			sc->sharedStrideRaderFFT.type = 31;
			sc->sharedStrideRaderFFT.data.i = max_stride;
		}

		sc->maxSharedStride.type = 31;
		sc->maxSharedStride.data.i = (sc->sharedStrideBankConflictFirstStages.data.i < sc->sharedStrideReadWriteConflict.data.i) ? sc->sharedStrideReadWriteConflict.data.i : sc->sharedStrideBankConflictFirstStages.data.i;

		if (sc->useRaderFFT)
			sc->maxSharedStride.data.i = (sc->maxSharedStride.data.i < sc->sharedStrideRaderFFT.data.i) ? sc->sharedStrideRaderFFT.data.i : sc->maxSharedStride.data.i;

		sc->usedSharedMemory.type = 31;
		sc->usedSharedMemory.data.i = sc->complexSize * sc->localSize[1].data.i * sc->maxSharedStride.data.i;
		sc->maxSharedStride.data.i = ((sc->sharedMemSize < sc->usedSharedMemory.data.i)) ? sc->fftDim.data.i / sc->registerBoost + additionalR2Cshared : sc->maxSharedStride.data.i;

		sc->sharedStrideBankConflictFirstStages.data.i = (sc->maxSharedStride.data.i == (sc->fftDim.data.i / sc->registerBoost + additionalR2Cshared)) ? sc->fftDim.data.i / sc->registerBoost + additionalR2Cshared : sc->sharedStrideBankConflictFirstStages.data.i;
		sc->sharedStrideReadWriteConflict.data.i = (sc->maxSharedStride.data.i == (sc->fftDim.data.i / sc->registerBoost + additionalR2Cshared)) ? sc->fftDim.data.i / sc->registerBoost + additionalR2Cshared : sc->sharedStrideReadWriteConflict.data.i;
		if (sc->useRaderFFT) {
			sc->sharedStrideRaderFFT.data.i = (sc->maxSharedStride.data.i == (sc->fftDim.data.i / sc->registerBoost + additionalR2Cshared)) ? sc->fftDim.data.i / sc->registerBoost + additionalR2Cshared : sc->sharedStrideRaderFFT.data.i;
			sc->sharedShiftRaderFFT.data.i = (sc->maxSharedStride.data.i == (sc->fftDim.data.i / sc->registerBoost + additionalR2Cshared)) ? 0 : sc->sharedShiftRaderFFT.data.i;
		}
		//sc->maxSharedStride += mergeR2C;
		//printf("%" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", sc->maxSharedStride, sc->sharedStrideBankConflictFirstStages, sc->sharedStrideReadWriteConflict, sc->localSize[1], sc->fftDim);
		sc->sharedStride.type = 31;
		PfMov(sc, &sc->sharedStride, &sc->sharedStrideReadWriteConflict);


		sc->usedSharedMemory.data.i = sc->complexSize * sc->localSize[1].data.i * sc->maxSharedStride.data.i;
		if (sc->useRaderMult) {
			for (uint64_t i = 0; i < 20; i++) {
				sc->RaderKernelOffsetShared[i].type = 31;
				sc->RaderKernelOffsetShared[i].data.i += sc->usedSharedMemory.data.i / sc->complexSize;
			}
			sc->usedSharedMemory.data.i += sc->additionalRaderSharedSize.data.i * sc->complexSize;
		}
		PfContainer* vecType;
		PfGetTypeFromCode(sc, sc->vecTypeCode, &vecType);
#if(VKFFT_BACKEND==0)
		sc->tempLen = sprintf(sc->tempStr, "shared %s sdata[%" PRIu64 "];// sharedStride - fft size,  gl_WorkGroupSize.y - grouped consecutive ffts\n\n", vecType->data.s, sc->usedSharedMemory.data.i / sc->complexSize);
		PfAppendLine(sc);
		
#elif(VKFFT_BACKEND==1)
		//sc->tempLen = sprintf(sc->tempStr, "%s %s sdata[%" PRIu64 "];// sharedStride - fft size,  gl_WorkGroupSize.y - grouped consecutive ffts\n\n", sharedDefinitions, vecType, sc->localSize[1] * sc->maxSharedStride);
		sc->tempLen = sprintf(sc->tempStr, "%s* sdata = (%s*)shared;\n\n", vecType->data.s, vecType->data.s);
		PfAppendLine(sc);
		
		//sc->tempLen = sprintf(sc->tempStr, "%s %s sdata[];// sharedStride - fft size,  gl_WorkGroupSize.y - grouped consecutive ffts\n\n", sharedDefinitions, vecType);
#elif(VKFFT_BACKEND==2)
		//sc->tempLen = sprintf(sc->tempStr, "%s %s sdata[%" PRIu64 "];// sharedStride - fft size,  gl_WorkGroupSize.y - grouped consecutive ffts\n\n", sharedDefinitions, vecType, sc->localSize[1] * sc->maxSharedStride);
		sc->tempLen = sprintf(sc->tempStr, "%s* sdata = (%s*)shared;\n\n", vecType->data.s, vecType->data.s);
		PfAppendLine(sc);
		
		//sc->tempLen = sprintf(sc->tempStr, "%s %s sdata[];// sharedStride - fft size,  gl_WorkGroupSize.y - grouped consecutive ffts\n\n", sharedDefinitions, vecType);
#elif((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
		sc->tempLen = sprintf(sc->tempStr, "__local %s sdata[%" PRIu64 "];// sharedStride - fft size,  gl_WorkGroupSize.y - grouped consecutive ffts\n\n", vecType->data.s, sc->usedSharedMemory.data.i / sc->complexSize);
		PfAppendLine(sc);
		
#endif
		break;
				}
	case 1: case 2: case 111: case 121: case 131: case 141: case 143: case 145://grouped_c2c + single_c2c_strided
	{
		uint64_t shift = (sc->fftDim.data.i < (sc->numSharedBanks / 2)) ? (sc->numSharedBanks / 2) / sc->fftDim.data.i : 1;
		sc->sharedStrideReadWriteConflict.type = 31;
		sc->sharedStrideReadWriteConflict.data.i = ((sc->axisSwapped) && ((sc->localSize[0].data.i % 4) == 0)) ? sc->localSize[0].data.i + shift : sc->localSize[0].data.i;
		sc->maxSharedStride.type = 31;
		sc->maxSharedStride.data.i = ((maxSequenceSharedMemory.data.i < sc->sharedStrideReadWriteConflict.data.i * (sc->fftDim.data.i / sc->registerBoost + (int64_t)additionalR2Cshared))) ? sc->localSize[0].data.i : sc->sharedStrideReadWriteConflict.data.i;
		sc->sharedStrideReadWriteConflict.data.i = (sc->maxSharedStride.data.i == sc->localSize[0].data.i) ? sc->localSize[0].data.i : sc->sharedStrideReadWriteConflict.data.i;
		
		sc->sharedStride.type = 31;
		PfMov(sc, &sc->sharedStride, &sc->sharedStrideReadWriteConflict);
		
		sc->usedSharedMemory.type = 31;
		sc->usedSharedMemory.data.i = sc->complexSize * sc->maxSharedStride.data.i * (sc->fftDim.data.i / sc->registerBoost + additionalR2Cshared);
		if (sc->useRaderMult) {
			for (uint64_t i = 0; i < 20; i++) {
				sc->RaderKernelOffsetShared[i].type = 31;
				sc->RaderKernelOffsetShared[i].data.i += sc->usedSharedMemory.data.i / sc->complexSize;
			}
			sc->usedSharedMemory.data.i += sc->additionalRaderSharedSize.data.i * sc->complexSize;
		}
		PfContainer* vecType;
		PfGetTypeFromCode(sc, sc->vecTypeCode, &vecType);
#if(VKFFT_BACKEND==0)
		sc->tempLen = sprintf(sc->tempStr, "shared %s sdata[%" PRIu64 "];\n\n", vecType->data.s, sc->usedSharedMemory.data.i / sc->complexSize);
		PfAppendLine(sc);
		
#elif(VKFFT_BACKEND==1)
		//sc->tempLen = sprintf(sc->tempStr, "%s %s sdata[%" PRIu64 "];\n\n", sharedDefinitions, vecType, sc->maxSharedStride * (sc->fftDim + mergeR2C) / sc->registerBoost);
		sc->tempLen = sprintf(sc->tempStr, "%s* sdata = (%s*)shared;\n\n", vecType->data.s, vecType->data.s);
		PfAppendLine(sc);
		
		//sc->tempLen = sprintf(sc->tempStr, "%s %s sdata[];\n\n", sharedDefinitions, vecType);
#elif(VKFFT_BACKEND==2)
		//sc->tempLen = sprintf(sc->tempStr, "%s %s sdata[%" PRIu64 "];\n\n", sharedDefinitions, vecType, sc->maxSharedStride * (sc->fftDim + mergeR2C) / sc->registerBoost);
		sc->tempLen = sprintf(sc->tempStr, "%s* sdata = (%s*)shared;\n\n", vecType->data.s, vecType->data.s);
		PfAppendLine(sc);
		
		//sc->tempLen = sprintf(sc->tempStr, "%s %s sdata[];\n\n", sharedDefinitions, vecType);
#elif((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
		sc->tempLen = sprintf(sc->tempStr, "__local %s sdata[%" PRIu64 "];\n\n", vecType->data.s, sc->usedSharedMemory.data.i / sc->complexSize);
		PfAppendLine(sc);
		
#endif
		break;
	}
			}
	if (sc->useRaderMult) {
		sc->sharedMemSize += (int)(sc->additionalRaderSharedSize.data.i * sc->complexSize);
		sc->sharedMemSizePow2 += (int)(sc->additionalRaderSharedSize.data.i * sc->complexSize);
	}
	return;
}
#endif
