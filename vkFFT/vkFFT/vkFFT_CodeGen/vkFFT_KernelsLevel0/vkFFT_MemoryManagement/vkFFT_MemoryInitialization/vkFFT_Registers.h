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
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_StringManagement/vkFFT_StringManager.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_MathUtils/vkFFT_MathUtils.h"

static inline void appendRegisterInitialization(VkFFTSpecializationConstantsLayout* sc, int type) {

	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	char name[50];
	//sc->tempLen = sprintf(sc->tempStr, "	uint dum=gl_LocalInvocationID.x;\n");
	int additional_registers_c2r = 0;
	if ((sc->mergeSequencesR2C == 1) && (type == 500))
		additional_registers_c2r = 2;

	pfINT max_coordinate = 1;
	if ((sc->convolutionStep) && (sc->matrixConvolution > 1)) {
		max_coordinate = sc->matrixConvolution;
	}

	int logicalStoragePerThread = (sc->registers_per_thread + additional_registers_c2r) * sc->registerBoost * (int)max_coordinate;
	int logicalRegistersPerThread = sc->registers_per_thread;

	sc->regIDs = (PfContainer*)calloc(logicalStoragePerThread, sizeof(PfContainer));
	if (sc->regIDs == 0) sc->res = VKFFT_ERROR_MALLOC_FAILED;

	for (int i = 0; i < logicalStoragePerThread; i++) {
		sc->regIDs[i].type = 100 + sc->vecTypeCode;
		PfAllocateContainerFlexible(sc, &sc->regIDs[i], 50);

		sprintf(name, "temp_%d", i);
		PfDefine(sc, &sc->regIDs[i], name);
		PfSetToZero(sc, &sc->regIDs[i]);
	}
	if (sc->convolutionStep) {
		if (sc->numKernels.data.i > 1) {
			sc->regIDs_copy = (PfContainer*)calloc(logicalStoragePerThread, sizeof(PfContainer));
			if (sc->regIDs_copy == 0) sc->res = VKFFT_ERROR_MALLOC_FAILED;

			for (int i = 0; i < logicalStoragePerThread; i++) {
				sc->regIDs_copy[i].type = 100 + sc->vecTypeCode;
				PfAllocateContainerFlexible(sc, &sc->regIDs_copy[i], 50);
				sprintf(name, "temp_copy_%d", i);
				PfDefine(sc, &sc->regIDs_copy[i], name);
				PfSetToZero(sc, &sc->regIDs_copy[i]);
			}
		}
		sc->temp_conv = (PfContainer*)calloc(sc->matrixConvolution, sizeof(PfContainer));
		if (sc->temp_conv == 0) sc->res = VKFFT_ERROR_MALLOC_FAILED;

		for (int j = 0; j < sc->matrixConvolution; j++) {
			sc->temp_conv[j].type = 100 + sc->vecTypeCode;
			PfAllocateContainerFlexible(sc, &sc->temp_conv[j], 50);
			sprintf(name, "temp_conv_%d", j);
			PfDefine(sc, &sc->temp_conv[j], name);
			PfSetToZero(sc, &sc->temp_conv[j]);
		}
	}


	sc->w.type = 100 + sc->vecTypeCode;
	PfAllocateContainerFlexible(sc, &sc->w, 50);
	sprintf(name, "w");
	PfDefine(sc, &sc->w, name);
	PfSetToZero(sc, &sc->w);

	if (((sc->floatTypeCode % 100) / 10) == 3) {
		sc->tempQuad.type = 100 + sc->vecTypeCode;
		PfAllocateContainerFlexible(sc, &sc->tempQuad, 50);
		sprintf(name, "tempQuad");
		PfDefine(sc, &sc->tempQuad, name);
		PfSetToZero(sc, &sc->tempQuad);

		sc->tempQuad2.type = 100 + sc->vecTypeCode;
		PfAllocateContainerFlexible(sc, &sc->tempQuad2, 50);
		sprintf(name, "tempQuad2");
		PfDefine(sc, &sc->tempQuad2, name);
		PfSetToZero(sc, &sc->tempQuad2);

		sc->tempQuad3.type = 100 + sc->vecTypeCode;
		PfAllocateContainerFlexible(sc, &sc->tempQuad3, 50);
		sprintf(name, "tempQuad3");
		PfDefine(sc, &sc->tempQuad3, name);
		PfSetToZero(sc, &sc->tempQuad3);

		sc->tempIntQuad.type = 100 + sc->uintTypeCode;
		PfAllocateContainerFlexible(sc, &sc->tempIntQuad, 50);
		sprintf(name, "tempIntQuad");
		PfDefine(sc, &sc->tempIntQuad, name);
		PfSetToZero(sc, &sc->tempIntQuad);
	}

	int maxNonPow2Radix = sc->maxNonPow2Radix;
	for (int i = 0; i < sc->usedLocRegs; i++) {
		sc->locID[i].type = 100 + sc->vecTypeCode;
		PfAllocateContainerFlexible(sc, &sc->locID[i], 50);
		sprintf(name, "loc_%d", i);
		PfDefine(sc, &sc->locID[i], name);
		PfSetToZero(sc, &sc->locID[i]);	
	}
	sc->temp.type = 100 + sc->vecTypeCode;
	PfAllocateContainerFlexible(sc, &sc->temp, 50);
	sprintf(name, "loc_0"); 
	PfSetContainerName(sc, &sc->temp, name);
	//PfDefineReference(sc, &sc->temp, name);
	//PfSetToZero(sc, &sc->temp);

	sc->tempFloat.type = 100 + sc->floatTypeCode;
	PfAllocateContainerFlexible(sc, &sc->tempFloat, 50);
	sprintf(name, "loc_0");
	if (((sc->floatTypeCode % 100) / 10) == 3) {
		sprintf(sc->tempFloat.data.dd[0].name, "%s.x.x\n", name);
		sprintf(sc->tempFloat.data.dd[1].name, "%s.x.y\n", name);
	}
	else {
		sprintf(sc->tempFloat.name, "%s.x", sc->temp.name);
	}
	//PfDefineReference(sc, &sc->tempFloat, name);

	sc->tempInt.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->tempInt, 50);
	sprintf(name, "tempInt");
	PfDefine(sc, &sc->tempInt, name);
	PfSetToZero(sc, &sc->tempInt);

	sc->tempInt2.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->tempInt2, 50);
	sprintf(name, "tempInt2");
	PfDefine(sc, &sc->tempInt2, name);
	PfSetToZero(sc, &sc->tempInt2);

	sc->shiftX.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->shiftX, 50);
	sprintf(name, "shiftX");
	PfDefine(sc, &sc->shiftX, name);
	PfSetToZero(sc, &sc->shiftX);

	sc->shiftY.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->shiftY, 50);
	sprintf(name, "shiftY");
	PfDefine(sc, &sc->shiftY, name);
	PfSetToZero(sc, &sc->shiftY);

	sc->shiftZ.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->shiftZ, 50);
	sprintf(name, "shiftZ");
	PfDefine(sc, &sc->shiftZ, name);
	PfSetToZero(sc, &sc->shiftZ);

	if (sc->useRaderFFT) {
		for (int i = 0; i < 2; i++) {
			sc->x0[i].type = 100 + sc->vecTypeCode;
			PfAllocateContainerFlexible(sc, &sc->x0[i], 50);
			sprintf(name, "x0_%d", i);
			PfDefine(sc, &sc->x0[i], name);
			PfSetToZero(sc, &sc->x0[i]);
		}
	}
	if (sc->useRaderMult) {
		int rader_fft_regs = (sc->useRaderFFT) ? 2 : 0;
		int rader_mult_regs = sc->raderRegisters / 2 - rader_fft_regs;
		if (rader_mult_regs <= sc->usedLocRegs - 1) {
			for (int i = 0; i < rader_mult_regs; i++) {
				sc->x0[i + rader_fft_regs].type = 100 + sc->vecTypeCode;
				PfAllocateContainerFlexible(sc, &sc->x0[i + rader_fft_regs], 50);
				PfCopyContainer(sc, &sc->x0[i + rader_fft_regs], &sc->locID[i + 1]);
			}
		}
		else {
			for (int i = 0; i < sc->usedLocRegs - 1; i++) {
				sc->x0[i + rader_fft_regs].type = 100 + sc->vecTypeCode;
				PfAllocateContainerFlexible(sc, &sc->x0[i + rader_fft_regs], 50);
				PfCopyContainer(sc, &sc->x0[i + rader_fft_regs], &sc->locID[i + 1]);
			}
			for (int i = sc->usedLocRegs - 1; i < rader_mult_regs; i++) {
				sc->x0[i + rader_fft_regs].type = 100 + sc->vecTypeCode;
				PfAllocateContainerFlexible(sc, &sc->x0[i + rader_fft_regs], 50);
				sprintf(name, "x0_%d", i + rader_fft_regs);	
				PfDefine(sc, &sc->x0[i + rader_fft_regs], name);
				PfSetToZero(sc, &sc->x0[i + rader_fft_regs]);
			}
		}
	}
	//sc->tempLen = sprintf(sc->tempStr, "	%s temp2;\n", vecType);
	//PfAppendLine(sc);
	//
	int useRadix8plus = 0;
	for (int i = 0; i < sc->numStages; i++)
		if ((sc->stageRadix[i] == 8) || (sc->stageRadix[i] == 16) || (sc->stageRadix[i] == 32) || (sc->useRaderFFT)) useRadix8plus = 1;
	if (useRadix8plus == 1) {
		if (maxNonPow2Radix > 1) {
			sc->iw.type = 100 + sc->vecTypeCode;
			PfAllocateContainerFlexible(sc, &sc->iw, 50);
			sprintf(name, "%s", sc->locID[1].name);
			PfSetContainerName(sc, &sc->iw, name);
		}
		else {
			sc->iw.type = 100 + sc->vecTypeCode;
			PfAllocateContainerFlexible(sc, &sc->iw, 50);
			sprintf(name, "iw");
			PfDefine(sc, &sc->iw, name);
			PfSetToZero(sc, &sc->iw);
		}
	}
	//sc->tempLen = sprintf(sc->tempStr, "	%s %s;\n", vecType, sc->tempReg);
	sc->stageInvocationID.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->stageInvocationID, 50);
	sprintf(name, "stageInvocationID");
	PfDefine(sc, &sc->stageInvocationID, name);
	PfSetToZero(sc, &sc->stageInvocationID);

	sc->blockInvocationID.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->blockInvocationID, 50);
	sprintf(name, "blockInvocationID");
	PfDefine(sc, &sc->blockInvocationID, name);
	PfSetToZero(sc, &sc->blockInvocationID);
	
	sc->sdataID.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->sdataID, 50);
	sprintf(name, "sdataID");
	PfDefine(sc, &sc->sdataID, name);
	PfSetToZero(sc, &sc->sdataID);
	
	sc->combinedID.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->combinedID, 50);
	sprintf(name, "combinedID");
	PfDefine(sc, &sc->combinedID, name);
	PfSetToZero(sc, &sc->combinedID);
	
	sc->inoutID.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->inoutID, 50);
	sprintf(name, "inoutID");
	PfDefine(sc, &sc->inoutID, name);
	PfSetToZero(sc, &sc->inoutID);
	
	if (((type/10)  == 121) || ((type/10)  == 131) || ((type/10) == 141) || ((type/10)  == 143)) {
		sc->inoutID2.type = 100 + sc->uintTypeCode;
		PfAllocateContainerFlexible(sc, &sc->inoutID2, 50);
		sprintf(name, "inoutID2");
		PfDefine(sc, &sc->inoutID2, name);
		PfSetToZero(sc, &sc->inoutID2);
	}

	sc->inoutID_x.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->inoutID_x, 50);
	sprintf(name, "inoutID_x");
	PfDefine(sc, &sc->inoutID_x, name);
	PfSetToZero(sc, &sc->inoutID_x);

	sc->inoutID_y.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->inoutID_y, 50);
	sprintf(name, "inoutID_y");
	PfDefine(sc, &sc->inoutID_y, name);
	PfSetToZero(sc, &sc->inoutID_y);

	if ((sc->fftDim.data.i < sc->fft_dim_full.data.i) || ((type % 10) == 1) || ((type%10) == 2) || (sc->performZeropaddingFull[0]) || (sc->performZeropaddingFull[1]) || (sc->performZeropaddingFull[2])) {
		sc->disableThreads.type = 101;
		PfAllocateContainerFlexible(sc, &sc->disableThreads, 50);
		sprintf(name, "disableThreads");
		PfDefine(sc, &sc->disableThreads, name);
		temp_int.data.i = 1;
		PfMov(sc, &sc->disableThreads, &temp_int);
	}
	//initialize subgroups ids
	if (sc->useRader) {
		sc->raderIDx.type = 100 + sc->uintTypeCode;
		PfAllocateContainerFlexible(sc, &sc->raderIDx, 50);
		sprintf(name, "raderIDx");
		PfDefine(sc, &sc->raderIDx, name);
		PfSetToZero(sc, &sc->raderIDx);
		
		sc->raderIDx2.type = 100 + sc->uintTypeCode;
		PfAllocateContainerFlexible(sc, &sc->raderIDx2, 50);
		sprintf(name, "raderIDx2");
		PfDefine(sc, &sc->raderIDx2, name);
		PfSetToZero(sc, &sc->raderIDx2);
		
		/*#if((VKFFT_BACKEND==1)||(VKFFT_BACKEND==2))
				sprintf(sc->gl_SubgroupInvocationID, "gl_SubgroupInvocationID");
				sprintf(sc->gl_SubgroupID, "gl_SubgroupID");
				if (sc->localSize[1] == 1) {
					sc->tempLen = sprintf(sc->tempStr, "	%s %s=(threadIdx.x %% %" PRIu64 ");\n", uintType, sc->gl_SubgroupInvocationID, sc->warpSize);
					PfAppendLine(sc);
					
					sc->tempLen = sprintf(sc->tempStr, "	%s %s=(threadIdx.x / %" PRIu64 ");\n", uintType, sc->gl_SubgroupID, sc->warpSize);
					PfAppendLine(sc);
					
				}
				else {
					sc->tempLen = sprintf(sc->tempStr, "	%s %s=((threadIdx.x+threadIdx.y*blockDim.x) %% %" PRIu64 ");\n", uintType, sc->gl_SubgroupInvocationID, sc->warpSize);
					PfAppendLine(sc);
					
					sc->tempLen = sprintf(sc->tempStr, "	%s %s=((threadIdx.x+threadIdx.y*blockDim.x) / %" PRIu64 ");\n", uintType, sc->gl_SubgroupID, sc->warpSize);
					PfAppendLine(sc);
					
				}
		#endif*/
	}
	if (sc->LUT) {
		sc->LUTId.type = 100 + sc->uintTypeCode;
		PfAllocateContainerFlexible(sc, &sc->LUTId, 50);
		sprintf(name, "LUTId");
		PfDefine(sc, &sc->LUTId, name);
		PfSetToZero(sc, &sc->LUTId);
		
		if ((!sc->LUT_4step)&&(sc->numAxisUploads>1)) {
			sc->angle.type = 100 + sc->floatTypeCode;
			PfAllocateContainerFlexible(sc, &sc->angle, 50);
			sprintf(name, "angle");
			PfDefine(sc, &sc->angle, name);
			PfSetToZero(sc, &sc->angle);
		}
	}
	else {
		sc->angle.type = 100 + sc->floatTypeCode;
		PfAllocateContainerFlexible(sc, &sc->angle, 50);
		sprintf(name, "angle");
		PfDefine(sc, &sc->angle, name);
		PfSetToZero(sc, &sc->angle);
	}
	if (((sc->stageStartSize.data.i > 1) && (!((sc->stageStartSize.data.i > 1) && (!sc->reorderFourStep) && (sc->inverse)))) || (((sc->stageStartSize.data.i > 1) && (!sc->reorderFourStep) && (sc->inverse))) || (sc->performDCT) || (sc->performDST)) {
		sc->mult.type = 100 + sc->vecTypeCode;
		PfAllocateContainerFlexible(sc, &sc->mult, 50);
		sprintf(name, "mult");
		PfDefine(sc, &sc->mult, name);
		PfSetToZero(sc, &sc->mult);
	}
	return;
}

static inline void appendRegisterInitialization_R2C(VkFFTSpecializationConstantsLayout* sc, int type) {

	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	char name[50];

	sc->regIDs = (PfContainer*)calloc(sc->registers_per_thread, sizeof(PfContainer));

	for (int i = 0; i < sc->registers_per_thread; i++) {
		sc->regIDs[i].type = 100 + sc->vecTypeCode;
		PfAllocateContainerFlexible(sc, &sc->regIDs[i], 50);
		sprintf(name, "temp_%d", i);
		PfDefine(sc, &sc->regIDs[i], name);
		PfSetToZero(sc, &sc->regIDs[i]);
	}
	
	sc->w.type = 100 + sc->vecTypeCode;
	PfAllocateContainerFlexible(sc, &sc->w, 50);
	sprintf(name, "w");
	PfDefine(sc, &sc->w, name);
	PfSetToZero(sc, &sc->w);

	if (((sc->floatTypeCode % 100) / 10) == 3) {
		sc->tempQuad.type = 100 + sc->vecTypeCode;
		PfAllocateContainerFlexible(sc, &sc->tempQuad, 50);
		sprintf(name, "tempQuad");
		PfDefine(sc, &sc->tempQuad, name);
		PfSetToZero(sc, &sc->tempQuad);

		sc->tempQuad2.type = 100 + sc->vecTypeCode;
		PfAllocateContainerFlexible(sc, &sc->tempQuad2, 50);
		sprintf(name, "tempQuad2");
		PfDefine(sc, &sc->tempQuad2, name);
		PfSetToZero(sc, &sc->tempQuad2);

		sc->tempQuad3.type = 100 + sc->vecTypeCode;
		PfAllocateContainerFlexible(sc, &sc->tempQuad3, 50);
		sprintf(name, "tempQuad3");
		PfDefine(sc, &sc->tempQuad3, name);
		PfSetToZero(sc, &sc->tempQuad3);

		sc->tempIntQuad.type = 100 + sc->uintTypeCode;
		PfAllocateContainerFlexible(sc, &sc->tempIntQuad, 50);
		sprintf(name, "tempIntQuad");
		PfDefine(sc, &sc->tempIntQuad, name);
		PfSetToZero(sc, &sc->tempIntQuad);
	}

	sc->tempInt.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->tempInt, 50);
	sprintf(name, "tempInt");
	PfDefine(sc, &sc->tempInt, name);
	PfSetToZero(sc, &sc->tempInt);

	sc->tempInt2.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->tempInt2, 50);
	sprintf(name, "tempInt2");
	PfDefine(sc, &sc->tempInt2, name);
	PfSetToZero(sc, &sc->tempInt2);

	sc->shiftX.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->shiftX, 50);
	sprintf(name, "shiftX");
	PfDefine(sc, &sc->shiftX, name);
	PfSetToZero(sc, &sc->shiftX);

	sc->shiftY.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->shiftY, 50);
	sprintf(name, "shiftY");
	PfDefine(sc, &sc->shiftY, name);
	PfSetToZero(sc, &sc->shiftY);

	sc->shiftZ.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->shiftZ, 50);
	sprintf(name, "shiftZ");
	PfDefine(sc, &sc->shiftZ, name);
	PfSetToZero(sc, &sc->shiftZ);

	sc->inoutID.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->inoutID, 50);
	sprintf(name, "inoutID");
	PfDefine(sc, &sc->inoutID, name);
	PfSetToZero(sc, &sc->inoutID);

	sc->inoutID_x.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->inoutID_x, 50);
	sprintf(name, "inoutID_x");
	PfDefine(sc, &sc->inoutID_x, name);
	PfSetToZero(sc, &sc->inoutID_x);

	sc->inoutID_y.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->inoutID_y, 50);
	sprintf(name, "inoutID_y");
	PfDefine(sc, &sc->inoutID_y, name);
	PfSetToZero(sc, &sc->inoutID_y);

	if (sc->LUT) {
		sc->LUTId.type = 100 + sc->uintTypeCode;
		PfAllocateContainerFlexible(sc, &sc->LUTId, 50);
		sprintf(name, "LUTId");
		PfDefine(sc, &sc->LUTId, name);
		PfSetToZero(sc, &sc->LUTId);

	}
	else {
		sc->angle.type = 100 + sc->floatTypeCode;
		PfAllocateContainerFlexible(sc, &sc->angle, 50);
		sprintf(name, "angle");
		PfDefine(sc, &sc->angle, name);
		PfSetToZero(sc, &sc->angle);
	}

	return;
}

static inline void freeRegisterInitialization(VkFFTSpecializationConstantsLayout* sc, int type) {

	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;

	int additional_registers_c2r = 0;
	if ((sc->mergeSequencesR2C == 1) && (type == 500))
		additional_registers_c2r = 2;

	pfINT max_coordinate = 1;
	if ((sc->convolutionStep) && (sc->matrixConvolution > 1)) {
		max_coordinate = sc->matrixConvolution;
	}

	int logicalStoragePerThread = (sc->registers_per_thread + additional_registers_c2r) * sc->registerBoost * (int)max_coordinate;
	int logicalRegistersPerThread = sc->registers_per_thread;

	for (pfUINT i = 0; i < logicalStoragePerThread; i++) {
		PfDeallocateContainer(sc, &sc->regIDs[i]);
	}

	if (sc->convolutionStep) {
		if (sc->numKernels.data.i > 1) {
			for (int i = 0; i < logicalStoragePerThread; i++) {
				PfDeallocateContainer(sc, &sc->regIDs_copy[i]);
			}
			free(sc->regIDs_copy);
		}
		
		for (int j = 0; j < sc->matrixConvolution; j++) {
			PfDeallocateContainer(sc, &sc->temp_conv[j]);
		}
		free(sc->temp_conv);
	}

	free(sc->regIDs);
	//sc->tempLen = sprintf(sc->tempStr, "	uint dum=gl_LocalInvocationID.y;//gl_LocalInvocationID.x/gl_WorkGroupSize.x;\n");
	//sc->tempLen = sprintf(sc->tempStr, "	dum=dum/gl_LocalInvocationID.x-1;\n");
	//sc->tempLen = sprintf(sc->tempStr, "	dummy=dummy/gl_LocalInvocationID.x-1;\n");
	if (sc->registerBoost > 1) {
		/*for (pfUINT i = 1; i < sc->registerBoost; i++) {
			//sc->tempLen = sprintf(sc->tempStr, "	%s temp%" PRIu64 "[%" PRIu64 "];\n", vecType, i, logicalRegistersPerThread);
			for (pfUINT j = 0; j < sc->registers_per_thread; j++) {
				sc->tempLen = sprintf(sc->tempStr, "	%s temp_%" PRIu64 ";\n", vecType, j + i * sc->registers_per_thread);
				PfAppendLine(sc);

				sc->tempLen = sprintf(sc->tempStr, "	temp_%" PRIu64 ".x=0;\n", j + i * sc->registers_per_thread);
				PfAppendLine(sc);

				sc->tempLen = sprintf(sc->tempStr, "	temp_%" PRIu64 ".y=0;\n", j + i * sc->registers_per_thread);
				PfAppendLine(sc);

			}
		}*/
	}
	PfDeallocateContainer(sc, &sc->w);
	
	pfUINT maxNonPow2Radix = sc->maxNonPow2Radix;
	for (pfUINT i = 0; i < sc->usedLocRegs; i++) {
		PfDeallocateContainer(sc, &sc->locID[i]);
	}

	PfDeallocateContainer(sc, &sc->temp);
	PfDeallocateContainer(sc, &sc->tempFloat);

	if (((sc->floatTypeCode % 100) / 10) == 3) {
		PfDeallocateContainer(sc, &sc->tempQuad);
		PfDeallocateContainer(sc, &sc->tempQuad2);
		PfDeallocateContainer(sc, &sc->tempQuad3);
		PfDeallocateContainer(sc, &sc->tempIntQuad);
	}
	PfDeallocateContainer(sc, &sc->tempInt);

	PfDeallocateContainer(sc, &sc->tempInt2);
	PfDeallocateContainer(sc, &sc->shiftX);
	PfDeallocateContainer(sc, &sc->shiftZ);
	PfDeallocateContainer(sc, &sc->shiftY);
	if (sc->useRaderFFT) {
		for (int i = 0; i < 2; i++) {
			PfDeallocateContainer(sc, &sc->x0[i]);
		}
	}
	if (sc->useRaderMult) {
		int rader_fft_regs = (sc->useRaderFFT) ? 2 : 0;
		int rader_mult_regs = sc->raderRegisters / 2 - rader_fft_regs;
		if (rader_mult_regs <= sc->usedLocRegs - 1) {
			for (int i = 0; i < rader_mult_regs; i++) {
				PfDeallocateContainer(sc, &sc->x0[i + rader_fft_regs]);
			}
		}
		else {
			for (int i = 0; i < sc->usedLocRegs - 1; i++) {
				PfDeallocateContainer(sc, &sc->x0[i + rader_fft_regs]);
			}
			for (int i = sc->usedLocRegs - 1; i < rader_mult_regs; i++) {
				PfDeallocateContainer(sc, &sc->x0[i + rader_fft_regs]);
			}
		}
	}
	//sc->tempLen = sprintf(sc->tempStr, "	%s temp2;\n", vecType);
	//PfAppendLine(sc);
	//
	int useRadix8plus = 0;
	for (int i = 0; i < sc->numStages; i++)
		if ((sc->stageRadix[i] == 8) || (sc->stageRadix[i] == 16) || (sc->stageRadix[i] == 32) || (sc->useRaderFFT)) useRadix8plus = 1;
	if (useRadix8plus == 1) {
		if (maxNonPow2Radix > 1) {
			PfDeallocateContainer(sc, &sc->iw);
		}
		else {
			PfDeallocateContainer(sc, &sc->iw);
		}
	}
	//sc->tempLen = sprintf(sc->tempStr, "	%s %s;\n", vecType, sc->tempReg);
	PfDeallocateContainer(sc, &sc->stageInvocationID);
	PfDeallocateContainer(sc, &sc->blockInvocationID);
	PfDeallocateContainer(sc, &sc->sdataID);
	PfDeallocateContainer(sc, &sc->combinedID);
	PfDeallocateContainer(sc, &sc->inoutID);
	if (((type/10)  == 121) || ((type/10)  == 131) || ((type/10) == 141) || ((type/10)  == 143)) 
		PfDeallocateContainer(sc, &sc->inoutID2);
	PfDeallocateContainer(sc, &sc->inoutID_x);
	PfDeallocateContainer(sc, &sc->inoutID_y);

	if ((sc->performZeropaddingFull[0]) || (sc->performZeropaddingFull[1]) || (sc->performZeropaddingFull[2])) {
		PfDeallocateContainer(sc, &sc->disableThreads);
	}
	//initialize subgroups ids
	if (sc->useRader) {
		PfDeallocateContainer(sc, &sc->raderIDx);

		PfDeallocateContainer(sc, &sc->raderIDx2);

		/*#if((VKFFT_BACKEND==1)||(VKFFT_BACKEND==2))
				sprintf(sc->gl_SubgroupInvocationID, "gl_SubgroupInvocationID");
				sprintf(sc->gl_SubgroupID, "gl_SubgroupID");
				if (sc->localSize[1] == 1) {
					sc->tempLen = sprintf(sc->tempStr, "	%s %s=(threadIdx.x %% %" PRIu64 ");\n", uintType, sc->gl_SubgroupInvocationID, sc->warpSize);
					PfAppendLine(sc);

					sc->tempLen = sprintf(sc->tempStr, "	%s %s=(threadIdx.x / %" PRIu64 ");\n", uintType, sc->gl_SubgroupID, sc->warpSize);
					PfAppendLine(sc);

				}
				else {
					sc->tempLen = sprintf(sc->tempStr, "	%s %s=((threadIdx.x+threadIdx.y*blockDim.x) %% %" PRIu64 ");\n", uintType, sc->gl_SubgroupInvocationID, sc->warpSize);
					PfAppendLine(sc);

					sc->tempLen = sprintf(sc->tempStr, "	%s %s=((threadIdx.x+threadIdx.y*blockDim.x) / %" PRIu64 ");\n", uintType, sc->gl_SubgroupID, sc->warpSize);
					PfAppendLine(sc);

				}
		#endif*/
	}
	if (sc->LUT) {
		PfDeallocateContainer(sc, &sc->LUTId);

		if (!sc->LUT_4step) {
			PfDeallocateContainer(sc, &sc->angle);
		}
	}
	else {
		PfDeallocateContainer(sc, &sc->angle);
	}
	if (((sc->stageStartSize.data.i > 1) && (!((sc->stageStartSize.data.i > 1) && (!sc->reorderFourStep) && (sc->inverse)))) || (((sc->stageStartSize.data.i > 1) && (!sc->reorderFourStep) && (sc->inverse))) || (sc->performDCT) || (sc->performDST)) {
		PfDeallocateContainer(sc, &sc->mult);
	}
	return;
}

static inline void freeRegisterInitialization_R2C(VkFFTSpecializationConstantsLayout* sc, int type) {

	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;

	for (int i = 0; i < sc->registers_per_thread; i++) {
		PfDeallocateContainer(sc, &sc->regIDs[i]);
	}

	free(sc->regIDs);

	PfDeallocateContainer(sc, &sc->w);
	
	if (((sc->floatTypeCode % 100) / 10) == 3) {
		PfDeallocateContainer(sc, &sc->tempQuad);
		PfDeallocateContainer(sc, &sc->tempQuad2);
		PfDeallocateContainer(sc, &sc->tempQuad3);
		PfDeallocateContainer(sc, &sc->tempIntQuad);
	}
	PfDeallocateContainer(sc, &sc->tempInt);
	
	PfDeallocateContainer(sc, &sc->tempInt2);
	
	PfDeallocateContainer(sc, &sc->shiftX);
	
	PfDeallocateContainer(sc, &sc->shiftY);
	
	PfDeallocateContainer(sc, &sc->shiftZ);
	
	PfDeallocateContainer(sc, &sc->inoutID);
	
	PfDeallocateContainer(sc, &sc->inoutID_x);
	
	if (sc->LUT) {
		PfDeallocateContainer(sc, &sc->LUTId);
	}
	else {
		PfDeallocateContainer(sc, &sc->angle);
	}

	return;
}
#endif
