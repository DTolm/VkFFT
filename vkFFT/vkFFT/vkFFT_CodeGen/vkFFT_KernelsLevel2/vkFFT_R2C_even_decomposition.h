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
#ifndef VKFFT_R2C_EVEN_DECOMPOSITION_H
#define VKFFT_R2C_EVEN_DECOMPOSITION_H

#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_StringManagement/vkFFT_StringManager.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryInitialization/vkFFT_InputOutput.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryInitialization/vkFFT_InputOutputLayout.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryInitialization/vkFFT_SharedMemory.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryInitialization/vkFFT_Registers.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryInitialization/vkFFT_PushConstants.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryInitialization/vkFFT_Constants.h"

#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel1/vkFFT_ReadWrite.h"

#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_KernelUtils.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_KernelStartEnd.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_MathUtils/vkFFT_MathUtils.h"

static inline VkFFTResult shaderGen_R2C_even_decomposition(VkFFTSpecializationConstantsLayout* sc, int type) {
	
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_int1 = VKFFT_ZERO_INIT;
	temp_int1.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 32;
	appendVersion(sc);
	appendExtensions(sc);
	appendLayoutVkFFT(sc);

	if (((!sc->LUT) || (!sc->LUT_4step)) && (sc->floatTypeCode == 2)) {
		appendSinCos20(sc);
	}
	if ((sc->floatTypeCode != sc->floatTypeInputMemoryCode) || (sc->floatTypeCode != sc->floatTypeOutputMemoryCode)) {
		appendConversion(sc);
	}
	appendPushConstants(sc);

	int id = 0;
	appendInputLayoutVkFFT(sc, id, type);
	id++;
	appendOutputLayoutVkFFT(sc, id, type);
	id++;
	if (sc->LUT) {
		appendLUTLayoutVkFFT(sc, id);
		id++;
	}
	appendKernelStart_R2C(sc, type);

	appendRegisterInitialization_R2C(sc, type);
	if (sc->performWorkGroupShift[0]) {
		PfMul(sc, &sc->shiftX, &sc->workGroupShiftX, &sc->localSize[0], 0);
		PfAdd(sc, &sc->shiftX, &sc->shiftX, &sc->gl_GlobalInvocationID_x);
	}
	else {
		PfMov(sc, &sc->shiftX, &sc->gl_GlobalInvocationID_x);
	}
	temp_int.data.i = 4;
	PfDivCeil(sc, &temp_int, &sc->size[0], &temp_int);
	PfMod(sc, &sc->inoutID_x, &sc->shiftX, &temp_int);

	PfDiv(sc, &sc->tempInt, &sc->shiftX, &temp_int);
	for (int i = 1; i < sc->numFFTdims; i++){
        PfMod(sc, &sc->shiftY, &sc->tempInt, &sc->size[i]);
        PfMul(sc, &sc->shiftY, &sc->shiftY, &sc->inputStride[i], 0);
        PfAdd(sc, &sc->shiftZ, &sc->shiftZ, &sc->shiftY);
		if (i!=(sc->numFFTdims-1))
			PfDiv(sc, &sc->tempInt, &sc->tempInt, &sc->size[i]);
    }

	appendOffset(sc, 0, 0);
	for(int i = 1; i < sc->numFFTdims; i++)
		temp_int.data.i *= sc->size[i].data.i;
	PfIf_lt_start(sc, &sc->shiftX, &temp_int);	

	PfAdd(sc, &sc->inoutID, &sc->inoutID_x, &sc->shiftZ);

	appendGlobalToRegisters(sc, &sc->regIDs[0], &sc->inputsStruct, &sc->inoutID);
	
	
	if (sc->size[0].data.i % 4 == 0) {
		temp_int.data.i = 0;
		PfIf_eq_start(sc, &sc->inoutID_x, &temp_int);

		temp_int.data.i = sc->size[0].data.i / 2;
		PfAdd(sc, &sc->tempInt, &temp_int, &sc->shiftZ);	

		temp_int.data.i = 4;
		PfDivCeil(sc, &temp_int, &sc->size[0], &temp_int);
		PfAdd(sc, &sc->tempInt2, &temp_int, &sc->shiftZ);
		
		appendGlobalToRegisters(sc, &sc->w, &sc->inputsStruct, &sc->tempInt2);

		PfIf_else(sc);

		temp_int.data.i = sc->size[0].data.i / 2;
		PfSub(sc, &sc->tempInt, &temp_int, &sc->inoutID_x);
		PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->shiftZ);

		PfIf_end(sc);	
	}
	else {
		temp_int.data.i = sc->size[0].data.i / 2;
		PfSub(sc, &sc->tempInt, &temp_int, &sc->inoutID_x);
		PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->shiftZ);		
	}
	appendGlobalToRegisters(sc, &sc->regIDs[1], &sc->inputsStruct, &sc->tempInt);

	temp_int.data.i = 0;
	PfIf_eq_start(sc, &sc->inoutID_x, &temp_int);
	
	if (sc->size[0].data.i % 4 == 0) {
		if (!sc->inverse) {
			PfMov_x_y(sc, &sc->regIDs[2], &sc->regIDs[0]);
			PfMov_x_Neg_y(sc, &sc->regIDs[3], &sc->regIDs[0]);
			PfAdd_x(sc, &sc->regIDs[2], &sc->regIDs[2], &sc->regIDs[0]);
			PfAdd_x(sc, &sc->regIDs[3], &sc->regIDs[3], &sc->regIDs[0]);
		}
		else {
			PfSub_x(sc, &sc->regIDs[2], &sc->regIDs[0], &sc->regIDs[1]);
			PfMov_y_x(sc, &sc->regIDs[2], &sc->regIDs[2]);
			PfAdd_x(sc, &sc->regIDs[2], &sc->regIDs[0], &sc->regIDs[1]);			
		}
		PfConjugate(sc, &sc->w, &sc->w);
		
		if (sc->inverse) {
			temp_double.data.d = 2;
			PfMul(sc, &sc->w, &sc->w, &temp_double, 0);
		}

		appendRegistersToGlobal(sc, &sc->outputsStruct, &sc->inoutID, &sc->regIDs[2]);
		
		if (!sc->inverse) {
			appendRegistersToGlobal(sc, &sc->outputsStruct, &sc->tempInt, &sc->regIDs[3]);		
		}

		appendRegistersToGlobal(sc, &sc->outputsStruct, &sc->tempInt2, &sc->w);
	}
	else {
		if (!sc->inverse) {
			PfMov_x_y(sc, &sc->regIDs[2], &sc->regIDs[0]);
			PfMov_x_Neg_y(sc, &sc->regIDs[3], &sc->regIDs[0]);
			PfAdd_x(sc, &sc->regIDs[2], &sc->regIDs[2], &sc->regIDs[0]);
			PfAdd_x(sc, &sc->regIDs[3], &sc->regIDs[3], &sc->regIDs[0]);			
		}
		else {
			PfSub_x(sc, &sc->regIDs[2], &sc->regIDs[0], &sc->regIDs[1]);
			PfMov_y_x(sc, &sc->regIDs[2], &sc->regIDs[2]);
			PfAdd_x(sc, &sc->regIDs[2], &sc->regIDs[0], &sc->regIDs[1]);
		}
		appendRegistersToGlobal(sc, &sc->outputsStruct, &sc->inoutID, &sc->regIDs[2]);

		if (!sc->inverse) {
			appendRegistersToGlobal(sc, &sc->outputsStruct, &sc->tempInt, &sc->regIDs[3]);
		}
	}
	PfIf_else(sc);
	PfAdd(sc, &sc->regIDs[2], &sc->regIDs[0], &sc->regIDs[1]);
	PfSub(sc, &sc->regIDs[3], &sc->regIDs[0], &sc->regIDs[1]);

	if (!sc->inverse) {
		temp_double.data.d = 0.5l;
		PfMul(sc, &sc->regIDs[2], &sc->regIDs[2], &temp_double,0);
		PfMul(sc, &sc->regIDs[3], &sc->regIDs[3], &temp_double, 0);

		
	}
	if (sc->LUT) {
		appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->inoutID_x);
	}
	else {
		temp_double.data.d = sc->double_PI / (sc->size[0].data.i / 2);
		PfMul(sc, &sc->angle, &sc->inoutID_x, &temp_double, 0);
		
		PfSinCos(sc, &sc->w, &sc->angle);
	}
	if (!sc->inverse) {
		PfConjugate(sc, &sc->w, &sc->w);
		PfMov_x(sc, &sc->regIDs[0], &sc->regIDs[3]);
		PfMov_y(sc, &sc->regIDs[0], &sc->regIDs[2]);
		PfMul(sc, &sc->regIDs[1], &sc->regIDs[0], &sc->w, 0);
		PfMov_x_y(sc, &sc->regIDs[0], &sc->regIDs[1]);
		PfMov_y_Neg_x(sc, &sc->regIDs[0], &sc->regIDs[1]);
		
		PfSub_x(sc, &sc->regIDs[1], &sc->regIDs[2], &sc->regIDs[0]);
		PfSub_y(sc, &sc->regIDs[1], &sc->regIDs[0], &sc->regIDs[3]);
		
		PfAdd_x(sc, &sc->regIDs[0], &sc->regIDs[2], &sc->regIDs[0]);
		PfAdd_y(sc, &sc->regIDs[0], &sc->regIDs[0], &sc->regIDs[3]);
		
	}
	else {
		PfMov_x(sc, &sc->regIDs[0], &sc->regIDs[3]);
		PfMov_y(sc, &sc->regIDs[0], &sc->regIDs[2]);
		PfMul(sc, &sc->regIDs[1], &sc->regIDs[0], &sc->w, 0);
		PfMov_x_y(sc, &sc->regIDs[0], &sc->regIDs[1]);
		PfMov_y_x(sc, &sc->regIDs[0], &sc->regIDs[1]);

		PfAdd_x(sc, &sc->regIDs[1], &sc->regIDs[2], &sc->regIDs[0]);
		PfSub_y(sc, &sc->regIDs[1], &sc->regIDs[0], &sc->regIDs[3]);

		PfSub_x(sc, &sc->regIDs[0], &sc->regIDs[2], &sc->regIDs[0]);
		PfAdd_y(sc, &sc->regIDs[0], &sc->regIDs[0], &sc->regIDs[3]);		
	}
	
	appendRegistersToGlobal(sc, &sc->outputsStruct, &sc->inoutID, &sc->regIDs[0]);
	appendRegistersToGlobal(sc, &sc->outputsStruct, &sc->tempInt, &sc->regIDs[1]);

	
	PfIf_end(sc);
	PfIf_end(sc);	

	appendKernelEnd(sc);

	freeRegisterInitialization_R2C(sc, type);

	return sc->res;
}
#endif
