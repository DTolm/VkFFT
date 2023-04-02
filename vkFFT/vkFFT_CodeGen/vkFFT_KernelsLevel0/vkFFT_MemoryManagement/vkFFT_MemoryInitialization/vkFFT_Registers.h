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
#ifndef VKFFT_REGISTERS_H
#define VKFFT_REGISTERS_H
#include "vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT_CodeGen/vkFFT_StringManagement/vkFFT_StringManager.h"
#include "vkFFT_CodeGen/vkFFT_MathUtils/vkFFT_MathUtils.h"

static inline void appendRegisterInitialization(VkFFTSpecializationConstantsLayout* sc, int type) {

	if (sc->res != VKFFT_SUCCESS) return;
	VkContainer temp_int = {};
	temp_int.type = 31;

	//sc->tempLen = sprintf(sc->tempStr, "	uint dum=gl_LocalInvocationID.x;\n");
	int additional_registers_c2r = 0;
	if ((sc->mergeSequencesR2C == 1) && (type == 5))
		additional_registers_c2r = 2;

	int64_t max_coordinate = 1;
	if ((sc->convolutionStep) && (sc->matrixConvolution > 1)) {
		max_coordinate = sc->matrixConvolution;
	}

	int logicalStoragePerThread = (sc->registers_per_thread + additional_registers_c2r) * sc->registerBoost * max_coordinate;
	int logicalRegistersPerThread = sc->registers_per_thread;

	sc->regIDs = (VkContainer*)calloc(logicalStoragePerThread, sizeof(VkContainer));
	if (sc->regIDs == 0) sc->res = VKFFT_ERROR_MALLOC_FAILED;

	for (int i = 0; i < logicalStoragePerThread; i++) {
		VkAllocateContainerFlexible(sc, &sc->regIDs[i], 50);
		sc->regIDs[i].type = 100 + sc->vecTypeCode;
		sprintf(sc->regIDs[i].data.s, "temp_%d", i);
		VkDefine(sc, &sc->regIDs[i]);
		VkSetToZero(sc, &sc->regIDs[i]);
	}
	if (sc->convolutionStep) {
		if (sc->numKernels.data.i > 1) {
			sc->regIDs_copy = (VkContainer*)calloc(logicalStoragePerThread, sizeof(VkContainer));
			if (sc->regIDs_copy == 0) sc->res = VKFFT_ERROR_MALLOC_FAILED;

			for (int i = 0; i < logicalStoragePerThread; i++) {
				VkAllocateContainerFlexible(sc, &sc->regIDs_copy[i], 50);
				sc->regIDs_copy[i].type = 100 + sc->vecTypeCode;
				sprintf(sc->regIDs_copy[i].data.s, "temp_copy_%d", i);
				VkDefine(sc, &sc->regIDs_copy[i]);
				VkSetToZero(sc, &sc->regIDs_copy[i]);
			}
		}
		sc->temp_conv = (VkContainer*)calloc(sc->matrixConvolution, sizeof(VkContainer));
		if (sc->temp_conv == 0) sc->res = VKFFT_ERROR_MALLOC_FAILED;

		for (int j = 0; j < sc->matrixConvolution; j++) {
			VkAllocateContainerFlexible(sc, &sc->temp_conv[j], 50);
			sc->temp_conv[j].type = 100 + sc->vecTypeCode;
			sprintf(sc->temp_conv[j].data.s, "temp_conv_%d", j);
			VkDefine(sc, &sc->temp_conv[j]);
			VkSetToZero(sc, &sc->temp_conv[j]);
		}
	}


	VkAllocateContainerFlexible(sc, &sc->w, 50);
	sc->w.type = 100 + sc->vecTypeCode;
	sprintf(sc->w.data.s, "w");
	VkDefine(sc, &sc->w);
	VkSetToZero(sc, &sc->w);

	int maxNonPow2Radix = sc->maxNonPow2Radix;
	for (int i = 0; i < sc->usedLocRegs; i++) {
		VkAllocateContainerFlexible(sc, &sc->locID[i], 50);
		sc->locID[i].type = 100 + sc->vecTypeCode;
		sprintf(sc->locID[i].data.s, "loc_%d", i);
		VkDefine(sc, &sc->locID[i]);
		VkSetToZero(sc, &sc->locID[i]);	
	}
	VkAllocateContainerFlexible(sc, &sc->temp, 50);
	sc->temp.type = 100 + sc->vecTypeCode;
	sprintf(sc->temp.data.s, "loc_0"); 
	//VkDefine(sc, &sc->temp);
	//VkSetToZero(sc, &sc->temp);

	VkAllocateContainerFlexible(sc, &sc->tempFloat, 50);
	sc->tempFloat.type = 100 + sc->floatTypeCode;
	sprintf(sc->tempFloat.data.s, "%s.x", sc->temp.data.s);

	VkAllocateContainerFlexible(sc, &sc->tempInt, 50);
	sc->tempInt.type = 100 + sc->uintTypeCode;
	sprintf(sc->tempInt.data.s, "tempInt");
	VkDefine(sc, &sc->tempInt);
	VkSetToZero(sc, &sc->tempInt);

	VkAllocateContainerFlexible(sc, &sc->tempInt2, 50);
	sc->tempInt2.type = 100 + sc->uintTypeCode;
	sprintf(sc->tempInt2.data.s, "tempInt2");
	VkDefine(sc, &sc->tempInt2);
	VkSetToZero(sc, &sc->tempInt2);

	VkAllocateContainerFlexible(sc, &sc->shiftX, 50);
	sc->shiftX.type = 100 + sc->uintTypeCode;
	sprintf(sc->shiftX.data.s, "shiftX");
	VkDefine(sc, &sc->shiftX);
	VkSetToZero(sc, &sc->shiftX);

	VkAllocateContainerFlexible(sc, &sc->shiftY, 50);
	sc->shiftY.type = 100 + sc->uintTypeCode;
	sprintf(sc->shiftY.data.s, "shiftY");
	VkDefine(sc, &sc->shiftY);
	VkSetToZero(sc, &sc->shiftY);

	VkAllocateContainerFlexible(sc, &sc->shiftZ, 50);
	sc->shiftZ.type = 100 + sc->uintTypeCode;
	sprintf(sc->shiftZ.data.s, "shiftZ");
	VkDefine(sc, &sc->shiftZ);
	VkSetToZero(sc, &sc->shiftZ);

	if (sc->useRaderFFT) {
		for (int i = 0; i < 2; i++) {
			VkAllocateContainerFlexible(sc, &sc->x0[i], 50);
			sc->x0[i].type = 100 + sc->vecTypeCode;
			sprintf(sc->x0[i].data.s, "x0_%d", i);
			VkDefine(sc, &sc->x0[i]);
			VkSetToZero(sc, &sc->x0[i]);
		}
	}
	if (sc->useRaderMult) {
		int rader_fft_regs = (sc->useRaderFFT) ? 2 : 0;
		int rader_mult_regs = sc->raderRegisters / 2 - rader_fft_regs;
		if (rader_mult_regs <= sc->usedLocRegs - 1) {
			for (int i = 0; i < rader_mult_regs; i++) {
				VkAllocateContainerFlexible(sc, &sc->x0[i + rader_fft_regs], 50);
				sc->x0[i + rader_fft_regs].type = 100 + sc->vecTypeCode;
				sprintf(sc->x0[i + rader_fft_regs].data.s, "%s", sc->locID[i + 1].data.s);
			}
		}
		else {
			for (int i = 0; i < sc->usedLocRegs - 1; i++) {
				VkAllocateContainerFlexible(sc, &sc->x0[i + rader_fft_regs], 50);
				sc->x0[i + rader_fft_regs].type = 100 + sc->vecTypeCode;
				sprintf(sc->x0[i + rader_fft_regs].data.s, "%s", sc->locID[i + 1].data.s);
			}
			for (int i = sc->usedLocRegs - 1; i < rader_mult_regs; i++) {
				VkAllocateContainerFlexible(sc, &sc->x0[i + rader_fft_regs], 50);
				sc->x0[i + rader_fft_regs].type = 100 + sc->vecTypeCode;
				sprintf(sc->x0[i + rader_fft_regs].data.s, "x0_%d", i + rader_fft_regs);	
				VkDefine(sc, &sc->x0[i + rader_fft_regs]);
				VkSetToZero(sc, &sc->x0[i + rader_fft_regs]);
			}
		}
	}
	//sc->tempLen = sprintf(sc->tempStr, "	%s temp2;\n", vecType);
	//VkAppendLine(sc);
	//
	int useRadix8plus = 0;
	for (int i = 0; i < sc->numStages; i++)
		if ((sc->stageRadix[i] == 8) || (sc->stageRadix[i] == 16) || (sc->stageRadix[i] == 32) || (sc->useRaderFFT)) useRadix8plus = 1;
	if (useRadix8plus == 1) {
		if (maxNonPow2Radix > 1) {
			VkAllocateContainerFlexible(sc, &sc->iw, 50);
			sc->iw.type = 100 + sc->vecTypeCode;
			sprintf(sc->iw.data.s, "%s", sc->locID[1].data.s);
		}
		else {
			VkAllocateContainerFlexible(sc, &sc->iw, 50);
			sc->iw.type = 100 + sc->vecTypeCode;
			sprintf(sc->iw.data.s, "iw");
			VkDefine(sc, &sc->iw);
			VkSetToZero(sc, &sc->iw);
		}
	}
	//sc->tempLen = sprintf(sc->tempStr, "	%s %s;\n", vecType, sc->tempReg);
	VkAllocateContainerFlexible(sc, &sc->stageInvocationID, 50);
	sc->stageInvocationID.type = 100 + sc->uintTypeCode;
	sprintf(sc->stageInvocationID.data.s, "stageInvocationID");
	VkDefine(sc, &sc->stageInvocationID);
	VkSetToZero(sc, &sc->stageInvocationID);

	VkAllocateContainerFlexible(sc, &sc->blockInvocationID, 50);
	sc->blockInvocationID.type = 100 + sc->uintTypeCode;
	sprintf(sc->blockInvocationID.data.s, "blockInvocationID");
	VkDefine(sc, &sc->blockInvocationID);
	VkSetToZero(sc, &sc->blockInvocationID);
	
	VkAllocateContainerFlexible(sc, &sc->sdataID, 50);
	sc->sdataID.type = 100 + sc->uintTypeCode;
	sprintf(sc->sdataID.data.s, "sdataID");
	VkDefine(sc, &sc->sdataID);
	VkSetToZero(sc, &sc->sdataID);
	
	VkAllocateContainerFlexible(sc, &sc->combinedID, 50);
	sc->combinedID.type = 100 + sc->uintTypeCode;
	sprintf(sc->combinedID.data.s, "combinedID");
	VkDefine(sc, &sc->combinedID);
	VkSetToZero(sc, &sc->combinedID);
	
	VkAllocateContainerFlexible(sc, &sc->inoutID, 50);
	sc->inoutID.type = 100 + sc->uintTypeCode;
	sprintf(sc->inoutID.data.s, "inoutID");
	VkDefine(sc, &sc->inoutID);
	VkSetToZero(sc, &sc->inoutID);
	
	VkAllocateContainerFlexible(sc, &sc->inoutID_x, 50);
	sc->inoutID_x.type = 100 + sc->uintTypeCode;
	sprintf(sc->inoutID_x.data.s, "inoutID_x");
	VkDefine(sc, &sc->inoutID_x);
	VkSetToZero(sc, &sc->inoutID_x);

	VkAllocateContainerFlexible(sc, &sc->inoutID_y, 50);
	sc->inoutID_y.type = 100 + sc->uintTypeCode;
	sprintf(sc->inoutID_y.data.s, "inoutID_y");
	VkDefine(sc, &sc->inoutID_y);
	VkSetToZero(sc, &sc->inoutID_y);

	if ((sc->fftDim.data.i < sc->fft_dim_full.data.i) || (type == 1) || (type == 111) || (type == 121) || (type == 131) || (type == 143) || (type == 145) || (type == 2) || (sc->performZeropaddingFull[0]) || (sc->performZeropaddingFull[1]) || (sc->performZeropaddingFull[2])) {
		VkAllocateContainerFlexible(sc, &sc->disableThreads, 50);
		sc->disableThreads.type = 101;
		sprintf(sc->disableThreads.data.s, "disableThreads");
		VkDefine(sc, &sc->disableThreads);
		temp_int.data.i = 1;
		VkMov(sc, &sc->disableThreads, &temp_int);
	}
	//initialize subgroups ids
	if (sc->useRader) {
		VkAllocateContainerFlexible(sc, &sc->raderIDx, 50);
		sc->raderIDx.type = 100 + sc->uintTypeCode;
		sprintf(sc->raderIDx.data.s, "raderIDx");
		VkDefine(sc, &sc->raderIDx);
		VkSetToZero(sc, &sc->raderIDx);
		
		VkAllocateContainerFlexible(sc, &sc->raderIDx2, 50);
		sc->raderIDx2.type = 100 + sc->uintTypeCode;
		sprintf(sc->raderIDx2.data.s, "raderIDx2");
		VkDefine(sc, &sc->raderIDx2);
		VkSetToZero(sc, &sc->raderIDx2);
		
		/*#if((VKFFT_BACKEND==1)||(VKFFT_BACKEND==2))
				sprintf(sc->gl_SubgroupInvocationID, "gl_SubgroupInvocationID");
				sprintf(sc->gl_SubgroupID, "gl_SubgroupID");
				if (sc->localSize[1] == 1) {
					sc->tempLen = sprintf(sc->tempStr, "	%s %s=(threadIdx.x %% %" PRIu64 ");\n", uintType, sc->gl_SubgroupInvocationID, sc->warpSize);
					VkAppendLine(sc);
					
					sc->tempLen = sprintf(sc->tempStr, "	%s %s=(threadIdx.x / %" PRIu64 ");\n", uintType, sc->gl_SubgroupID, sc->warpSize);
					VkAppendLine(sc);
					
				}
				else {
					sc->tempLen = sprintf(sc->tempStr, "	%s %s=((threadIdx.x+threadIdx.y*blockDim.x) %% %" PRIu64 ");\n", uintType, sc->gl_SubgroupInvocationID, sc->warpSize);
					VkAppendLine(sc);
					
					sc->tempLen = sprintf(sc->tempStr, "	%s %s=((threadIdx.x+threadIdx.y*blockDim.x) / %" PRIu64 ");\n", uintType, sc->gl_SubgroupID, sc->warpSize);
					VkAppendLine(sc);
					
				}
		#endif*/
	}
	if (sc->LUT) {
		VkAllocateContainerFlexible(sc, &sc->LUTId, 50);
		sc->LUTId.type = 100 + sc->uintTypeCode;
		sprintf(sc->LUTId.data.s, "LUTId");
		VkDefine(sc, &sc->LUTId);
		VkSetToZero(sc, &sc->LUTId);
		
		if ((!sc->LUT_4step)&&(sc->numAxisUploads>1)) {
			VkAllocateContainerFlexible(sc, &sc->angle, 50);
			sc->angle.type = 100 + sc->floatTypeCode;
			sprintf(sc->angle.data.s, "angle");
			VkDefine(sc, &sc->angle);
			VkSetToZero(sc, &sc->angle);
		}
	}
	else {
		VkAllocateContainerFlexible(sc, &sc->angle, 50);
		sc->angle.type = 100 + sc->floatTypeCode;
		sprintf(sc->angle.data.s, "angle");
		VkDefine(sc, &sc->angle);
		VkSetToZero(sc, &sc->angle);
	}
	if (((sc->stageStartSize.data.i > 1) && (!((sc->stageStartSize.data.i > 1) && (!sc->reorderFourStep) && (sc->inverse)))) || (((sc->stageStartSize.data.i > 1) && (!sc->reorderFourStep) && (sc->inverse))) || (sc->performDCT)) {
		VkAllocateContainerFlexible(sc, &sc->mult, 50);
		sc->mult.type = 100 + sc->vecTypeCode;
		sprintf(sc->mult.data.s, "mult");
		VkDefine(sc, &sc->mult);
		VkSetToZero(sc, &sc->mult);
	}
	return;
}

static inline void appendRegisterInitialization_R2C(VkFFTSpecializationConstantsLayout* sc, int type) {

	if (sc->res != VKFFT_SUCCESS) return;
	VkContainer temp_int = {};
	temp_int.type = 31;

	sc->regIDs = (VkContainer*)calloc(sc->registers_per_thread, sizeof(VkContainer));

	for (int i = 0; i < sc->registers_per_thread; i++) {
		VkAllocateContainerFlexible(sc, &sc->regIDs[i], 50);
		sc->regIDs[i].type = 100 + sc->vecTypeCode;
		sprintf(sc->regIDs[i].data.s, "temp_%d", i);
		VkDefine(sc, &sc->regIDs[i]);
		VkSetToZero(sc, &sc->regIDs[i]);
	}
	
	VkAllocateContainerFlexible(sc, &sc->w, 50);
	sc->w.type = 100 + sc->vecTypeCode;
	sprintf(sc->w.data.s, "w");
	VkDefine(sc, &sc->w);
	VkSetToZero(sc, &sc->w);

	VkAllocateContainerFlexible(sc, &sc->tempInt, 50);
	sc->tempInt.type = 100 + sc->uintTypeCode;
	sprintf(sc->tempInt.data.s, "tempInt");
	VkDefine(sc, &sc->tempInt);
	VkSetToZero(sc, &sc->tempInt);

	VkAllocateContainerFlexible(sc, &sc->tempInt2, 50);
	sc->tempInt2.type = 100 + sc->uintTypeCode;
	sprintf(sc->tempInt2.data.s, "tempInt2");
	VkDefine(sc, &sc->tempInt2);
	VkSetToZero(sc, &sc->tempInt2);

	VkAllocateContainerFlexible(sc, &sc->shiftX, 50);
	sc->shiftX.type = 100 + sc->uintTypeCode;
	sprintf(sc->shiftX.data.s, "shiftX");
	VkDefine(sc, &sc->shiftX);
	VkSetToZero(sc, &sc->shiftX);

	VkAllocateContainerFlexible(sc, &sc->shiftY, 50);
	sc->shiftY.type = 100 + sc->uintTypeCode;
	sprintf(sc->shiftY.data.s, "shiftY");
	VkDefine(sc, &sc->shiftY);
	VkSetToZero(sc, &sc->shiftY);

	VkAllocateContainerFlexible(sc, &sc->shiftZ, 50);
	sc->shiftZ.type = 100 + sc->uintTypeCode;
	sprintf(sc->shiftZ.data.s, "shiftZ");
	VkDefine(sc, &sc->shiftZ);
	VkSetToZero(sc, &sc->shiftZ);

	VkAllocateContainerFlexible(sc, &sc->inoutID, 50);
	sc->inoutID.type = 100 + sc->uintTypeCode;
	sprintf(sc->inoutID.data.s, "inoutID");
	VkDefine(sc, &sc->inoutID);
	VkSetToZero(sc, &sc->inoutID);

	VkAllocateContainerFlexible(sc, &sc->inoutID_x, 50);
	sc->inoutID_x.type = 100 + sc->uintTypeCode;
	sprintf(sc->inoutID_x.data.s, "inoutID_x");
	VkDefine(sc, &sc->inoutID_x);
	VkSetToZero(sc, &sc->inoutID_x);

	VkAllocateContainerFlexible(sc, &sc->inoutID_y, 50);
	sc->inoutID_y.type = 100 + sc->uintTypeCode;
	sprintf(sc->inoutID_y.data.s, "inoutID_y");
	VkDefine(sc, &sc->inoutID_y);
	VkSetToZero(sc, &sc->inoutID_y);

	if (sc->LUT) {
		VkAllocateContainerFlexible(sc, &sc->LUTId, 50);
		sc->LUTId.type = 100 + sc->uintTypeCode;
		sprintf(sc->LUTId.data.s, "LUTId");
		VkDefine(sc, &sc->LUTId);
		VkSetToZero(sc, &sc->LUTId);

	}
	else {
		VkAllocateContainerFlexible(sc, &sc->angle, 50);
		sc->angle.type = 100 + sc->floatTypeCode;
		sprintf(sc->angle.data.s, "angle");
		VkDefine(sc, &sc->angle);
		VkSetToZero(sc, &sc->angle);
	}

	return;
}

static inline void freeRegisterInitialization(VkFFTSpecializationConstantsLayout* sc, int type) {

	if (sc->res != VKFFT_SUCCESS) return;
	VkContainer temp_int = {};
	temp_int.type = 31;

	//sc->tempLen = sprintf(sc->tempStr, "	uint dum=gl_LocalInvocationID.x;\n");
	int logicalStoragePerThread = sc->registers_per_thread * sc->registerBoost;
	int logicalRegistersPerThread = sc->registers_per_thread;

	for (uint64_t i = 0; i < logicalStoragePerThread; i++) {
		VkDeallocateContainer(sc, &sc->regIDs[i]);
	}

	if (sc->convolutionStep) {
		if (sc->numKernels.data.i > 1) {
			for (int i = 0; i < logicalStoragePerThread; i++) {
				VkDeallocateContainer(sc, &sc->regIDs_copy[i]);
			}
			free(sc->regIDs_copy);
		}
		
		for (int j = 0; j < sc->matrixConvolution; j++) {
			VkDeallocateContainer(sc, &sc->temp_conv[j]);
		}
		free(sc->temp_conv);
	}

	free(sc->regIDs);
	//sc->tempLen = sprintf(sc->tempStr, "	uint dum=gl_LocalInvocationID.y;//gl_LocalInvocationID.x/gl_WorkGroupSize.x;\n");
	//sc->tempLen = sprintf(sc->tempStr, "	dum=dum/gl_LocalInvocationID.x-1;\n");
	//sc->tempLen = sprintf(sc->tempStr, "	dummy=dummy/gl_LocalInvocationID.x-1;\n");
	if (sc->registerBoost > 1) {
		/*for (uint64_t i = 1; i < sc->registerBoost; i++) {
			//sc->tempLen = sprintf(sc->tempStr, "	%s temp%" PRIu64 "[%" PRIu64 "];\n", vecType, i, logicalRegistersPerThread);
			for (uint64_t j = 0; j < sc->registers_per_thread; j++) {
				sc->tempLen = sprintf(sc->tempStr, "	%s temp_%" PRIu64 ";\n", vecType, j + i * sc->registers_per_thread);
				VkAppendLine(sc);

				sc->tempLen = sprintf(sc->tempStr, "	temp_%" PRIu64 ".x=0;\n", j + i * sc->registers_per_thread);
				VkAppendLine(sc);

				sc->tempLen = sprintf(sc->tempStr, "	temp_%" PRIu64 ".y=0;\n", j + i * sc->registers_per_thread);
				VkAppendLine(sc);

			}
		}*/
	}
	VkDeallocateContainer(sc, &sc->w);
	
	uint64_t maxNonPow2Radix = sc->maxNonPow2Radix;
	for (uint64_t i = 0; i < sc->usedLocRegs; i++) {
		VkDeallocateContainer(sc, &sc->locID[i]);
	}
	VkDeallocateContainer(sc, &sc->temp);
	VkDeallocateContainer(sc, &sc->tempFloat);

	VkDeallocateContainer(sc, &sc->tempInt);

	VkDeallocateContainer(sc, &sc->tempInt2);
	VkDeallocateContainer(sc, &sc->shiftX);
	VkDeallocateContainer(sc, &sc->shiftZ);
	VkDeallocateContainer(sc, &sc->shiftY);
	if (sc->useRaderFFT) {
		for (int i = 0; i < 2; i++) {
			VkDeallocateContainer(sc, &sc->x0[i]);
		}
	}
	if (sc->useRaderMult) {
		int rader_fft_regs = (sc->useRaderFFT) ? 2 : 0;
		int rader_mult_regs = sc->raderRegisters / 2 - rader_fft_regs;
		if (rader_mult_regs <= sc->usedLocRegs - 1) {
			for (int i = 0; i < rader_mult_regs; i++) {
				VkDeallocateContainer(sc, &sc->x0[i + rader_fft_regs]);
			}
		}
		else {
			for (int i = 0; i < sc->usedLocRegs - 1; i++) {
				VkDeallocateContainer(sc, &sc->x0[i + rader_fft_regs]);
			}
			for (int i = sc->usedLocRegs - 1; i < rader_mult_regs; i++) {
				VkDeallocateContainer(sc, &sc->x0[i + rader_fft_regs]);
			}
		}
	}
	//sc->tempLen = sprintf(sc->tempStr, "	%s temp2;\n", vecType);
	//VkAppendLine(sc);
	//
	int useRadix8plus = 0;
	for (int i = 0; i < sc->numStages; i++)
		if ((sc->stageRadix[i] == 8) || (sc->stageRadix[i] == 16) || (sc->stageRadix[i] == 32) || (sc->useRaderFFT)) useRadix8plus = 1;
	if (useRadix8plus == 1) {
		if (maxNonPow2Radix > 1) {
			VkDeallocateContainer(sc, &sc->iw);
		}
		else {
			VkDeallocateContainer(sc, &sc->iw);
		}
	}
	//sc->tempLen = sprintf(sc->tempStr, "	%s %s;\n", vecType, sc->tempReg);
	VkDeallocateContainer(sc, &sc->stageInvocationID);
	VkDeallocateContainer(sc, &sc->blockInvocationID);
	VkDeallocateContainer(sc, &sc->sdataID);
	VkDeallocateContainer(sc, &sc->combinedID);
	VkDeallocateContainer(sc, &sc->inoutID);
	VkDeallocateContainer(sc, &sc->inoutID_x);
	VkDeallocateContainer(sc, &sc->inoutID_y);

	if ((sc->performZeropaddingFull[0]) || (sc->performZeropaddingFull[1]) || (sc->performZeropaddingFull[2])) {
		VkDeallocateContainer(sc, &sc->disableThreads);
	}
	//initialize subgroups ids
	if (sc->useRader) {
		VkDeallocateContainer(sc, &sc->raderIDx);

		VkDeallocateContainer(sc, &sc->raderIDx2);

		/*#if((VKFFT_BACKEND==1)||(VKFFT_BACKEND==2))
				sprintf(sc->gl_SubgroupInvocationID, "gl_SubgroupInvocationID");
				sprintf(sc->gl_SubgroupID, "gl_SubgroupID");
				if (sc->localSize[1] == 1) {
					sc->tempLen = sprintf(sc->tempStr, "	%s %s=(threadIdx.x %% %" PRIu64 ");\n", uintType, sc->gl_SubgroupInvocationID, sc->warpSize);
					VkAppendLine(sc);

					sc->tempLen = sprintf(sc->tempStr, "	%s %s=(threadIdx.x / %" PRIu64 ");\n", uintType, sc->gl_SubgroupID, sc->warpSize);
					VkAppendLine(sc);

				}
				else {
					sc->tempLen = sprintf(sc->tempStr, "	%s %s=((threadIdx.x+threadIdx.y*blockDim.x) %% %" PRIu64 ");\n", uintType, sc->gl_SubgroupInvocationID, sc->warpSize);
					VkAppendLine(sc);

					sc->tempLen = sprintf(sc->tempStr, "	%s %s=((threadIdx.x+threadIdx.y*blockDim.x) / %" PRIu64 ");\n", uintType, sc->gl_SubgroupID, sc->warpSize);
					VkAppendLine(sc);

				}
		#endif*/
	}
	if (sc->LUT) {
		VkDeallocateContainer(sc, &sc->LUTId);

		if (!sc->LUT_4step) {
			VkDeallocateContainer(sc, &sc->angle);
		}
	}
	else {
		VkDeallocateContainer(sc, &sc->angle);
	}
	if (((sc->stageStartSize.data.i > 1) && (!((sc->stageStartSize.data.i > 1) && (!sc->reorderFourStep) && (sc->inverse)))) || (((sc->stageStartSize.data.i > 1) && (!sc->reorderFourStep) && (sc->inverse))) || (sc->performDCT)) {
		VkDeallocateContainer(sc, &sc->mult);
	}
	return;
}

static inline void freeRegisterInitialization_R2C(VkFFTSpecializationConstantsLayout* sc, int type) {

	if (sc->res != VKFFT_SUCCESS) return;
	VkContainer temp_int = {};
	temp_int.type = 31;

	for (int i = 0; i < sc->registers_per_thread; i++) {
		VkDeallocateContainer(sc, &sc->regIDs[i]);
	}

	free(sc->regIDs);

	VkDeallocateContainer(sc, &sc->w);
	
	VkDeallocateContainer(sc, &sc->tempInt);
	
	VkDeallocateContainer(sc, &sc->tempInt2);
	
	VkDeallocateContainer(sc, &sc->shiftX);
	
	VkDeallocateContainer(sc, &sc->shiftY);
	
	VkDeallocateContainer(sc, &sc->shiftZ);
	
	VkDeallocateContainer(sc, &sc->inoutID);
	
	VkDeallocateContainer(sc, &sc->inoutID_x);
	
	if (sc->LUT) {
		VkDeallocateContainer(sc, &sc->LUTId);
	}
	else {
		VkDeallocateContainer(sc, &sc->angle);
	}

	return;
}
#endif
