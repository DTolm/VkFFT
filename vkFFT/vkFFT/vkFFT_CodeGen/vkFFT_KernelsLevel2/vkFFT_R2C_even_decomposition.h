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

static inline VkFFTResult VkShaderGen_R2C_even_decomposition(VkFFTSpecializationConstantsLayout* sc, int type) {
	
	VkContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	VkContainer temp_int1 = VKFFT_ZERO_INIT;
	temp_int1.type = 31;
	VkContainer temp_double = VKFFT_ZERO_INIT;
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
		VkMul(sc, &sc->shiftX, &sc->workGroupShiftX, &sc->localSize[0], 0);
		VkAdd(sc, &sc->shiftX, &sc->shiftX, &sc->gl_GlobalInvocationID_x);
	}
	else {
		VkMov(sc, &sc->shiftX, &sc->gl_GlobalInvocationID_x);
	}
	temp_int.data.i = 4;
	VkDivCeil(sc, &temp_int, &sc->size[0], &temp_int);
	VkMod(sc, &sc->inoutID_x, &sc->shiftX, &temp_int);

	VkDiv(sc, &sc->tempInt, &sc->shiftX, &temp_int);
	for (int i = 1; i < sc->numFFTdims; i++){
        VkMod(sc, &sc->shiftY, &sc->tempInt, &sc->size[i]);
        VkMul(sc, &sc->shiftY, &sc->shiftY, &sc->inputStride[i], 0);
        VkAdd(sc, &sc->shiftZ, &sc->shiftZ, &sc->shiftY);
		if (i!=(sc->numFFTdims-1))
			VkDiv(sc, &sc->tempInt, &sc->tempInt, &sc->size[i]);
    }

	appendOffset(sc, 0, 0);
	for(int i = 1; i < sc->numFFTdims; i++)
		temp_int.data.i *= sc->size[i].data.i;
	VkIf_lt_start(sc, &sc->shiftX, &temp_int);	

	VkAdd(sc, &sc->inoutID, &sc->inoutID_x, &sc->shiftZ);

	appendGlobalToRegisters(sc, &sc->regIDs[0], &sc->inputsStruct, &sc->inoutID);
	
	
	if (sc->size[0].data.i % 4 == 0) {
		temp_int.data.i = 0;
		VkIf_eq_start(sc, &sc->inoutID_x, &temp_int);

		temp_int.data.i = sc->size[0].data.i / 2;
		VkAdd(sc, &sc->tempInt, &temp_int, &sc->shiftZ);	

		temp_int.data.i = 4;
		VkDivCeil(sc, &temp_int, &sc->size[0], &temp_int);
		VkAdd(sc, &sc->tempInt2, &temp_int, &sc->shiftZ);
		
		appendGlobalToRegisters(sc, &sc->w, &sc->inputsStruct, &sc->tempInt2);

		VkIf_else(sc);

		temp_int.data.i = sc->size[0].data.i / 2;
		VkSub(sc, &sc->tempInt, &temp_int, &sc->inoutID_x);
		VkAdd(sc, &sc->tempInt, &sc->tempInt, &sc->shiftZ);

		VkIf_end(sc);	
	}
	else {
		temp_int.data.i = sc->size[0].data.i / 2;
		VkSub(sc, &sc->tempInt, &temp_int, &sc->inoutID_x);
		VkAdd(sc, &sc->tempInt, &sc->tempInt, &sc->shiftZ);		
	}
	appendGlobalToRegisters(sc, &sc->regIDs[1], &sc->inputsStruct, &sc->tempInt);

	temp_int.data.i = 0;
	VkIf_eq_start(sc, &sc->inoutID_x, &temp_int);
	
	if (sc->size[0].data.i % 4 == 0) {
		if (!sc->inverse) {
			VkMov_x_y(sc, &sc->regIDs[2], &sc->regIDs[0]);
			VkMov_x_Neg_y(sc, &sc->regIDs[3], &sc->regIDs[0]);
			VkAdd_x(sc, &sc->regIDs[2], &sc->regIDs[2], &sc->regIDs[0]);
			VkAdd_x(sc, &sc->regIDs[3], &sc->regIDs[3], &sc->regIDs[0]);
		}
		else {
			VkSub_x(sc, &sc->regIDs[2], &sc->regIDs[0], &sc->regIDs[1]);
			VkMov_y_x(sc, &sc->regIDs[2], &sc->regIDs[2]);
			VkAdd_x(sc, &sc->regIDs[2], &sc->regIDs[0], &sc->regIDs[1]);			
		}
		VkConjugate(sc, &sc->w, &sc->w);
		
		if (sc->inverse) {
			temp_double.data.d = 2;
			VkMul(sc, &sc->w, &sc->w, &temp_double, 0);
		}

		appendRegistersToGlobal(sc, &sc->outputsStruct, &sc->inoutID, &sc->regIDs[2]);
		
		if (!sc->inverse) {
			appendRegistersToGlobal(sc, &sc->outputsStruct, &sc->tempInt, &sc->regIDs[3]);		
		}

		appendRegistersToGlobal(sc, &sc->outputsStruct, &sc->tempInt2, &sc->w);
	}
	else {
		if (!sc->inverse) {
			VkMov_x_y(sc, &sc->regIDs[2], &sc->regIDs[0]);
			VkMov_x_Neg_y(sc, &sc->regIDs[3], &sc->regIDs[0]);
			VkAdd_x(sc, &sc->regIDs[2], &sc->regIDs[2], &sc->regIDs[0]);
			VkAdd_x(sc, &sc->regIDs[3], &sc->regIDs[3], &sc->regIDs[0]);			
		}
		else {
			VkSub_x(sc, &sc->regIDs[2], &sc->regIDs[0], &sc->regIDs[1]);
			VkMov_y_x(sc, &sc->regIDs[2], &sc->regIDs[2]);
			VkAdd_x(sc, &sc->regIDs[2], &sc->regIDs[0], &sc->regIDs[1]);
		}
		appendRegistersToGlobal(sc, &sc->outputsStruct, &sc->inoutID, &sc->regIDs[2]);

		if (!sc->inverse) {
			appendRegistersToGlobal(sc, &sc->outputsStruct, &sc->tempInt, &sc->regIDs[3]);
		}
	}
	VkIf_else(sc);
	VkAdd(sc, &sc->regIDs[2], &sc->regIDs[0], &sc->regIDs[1]);
	VkSub(sc, &sc->regIDs[3], &sc->regIDs[0], &sc->regIDs[1]);

	if (!sc->inverse) {
		temp_double.data.d = 0.5l;
		VkMul(sc, &sc->regIDs[2], &sc->regIDs[2], &temp_double,0);
		VkMul(sc, &sc->regIDs[3], &sc->regIDs[3], &temp_double, 0);

		
	}
	if (sc->LUT) {
		appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->inoutID_x);
	}
	else {
		temp_double.data.d = sc->double_PI / (sc->size[0].data.i / 2);
		VkMul(sc, &sc->angle, &sc->inoutID_x, &temp_double, 0);
		
		VkSinCos(sc, &sc->w, &sc->angle);
	}
	if (!sc->inverse) {
		VkConjugate(sc, &sc->w, &sc->w);
		VkMov_x(sc, &sc->regIDs[0], &sc->regIDs[3]);
		VkMov_y(sc, &sc->regIDs[0], &sc->regIDs[2]);
		VkMul(sc, &sc->regIDs[1], &sc->regIDs[0], &sc->w, 0);
		VkMov_x_y(sc, &sc->regIDs[0], &sc->regIDs[1]);
		VkMov_y_Neg_x(sc, &sc->regIDs[0], &sc->regIDs[1]);
		
		VkSub_x(sc, &sc->regIDs[1], &sc->regIDs[2], &sc->regIDs[0]);
		VkSub_y(sc, &sc->regIDs[1], &sc->regIDs[0], &sc->regIDs[3]);
		
		VkAdd_x(sc, &sc->regIDs[0], &sc->regIDs[2], &sc->regIDs[0]);
		VkAdd_y(sc, &sc->regIDs[0], &sc->regIDs[0], &sc->regIDs[3]);
		
	}
	else {
		VkMov_x(sc, &sc->regIDs[0], &sc->regIDs[3]);
		VkMov_y(sc, &sc->regIDs[0], &sc->regIDs[2]);
		VkMul(sc, &sc->regIDs[1], &sc->regIDs[0], &sc->w, 0);
		VkMov_x_y(sc, &sc->regIDs[0], &sc->regIDs[1]);
		VkMov_y_x(sc, &sc->regIDs[0], &sc->regIDs[1]);

		VkAdd_x(sc, &sc->regIDs[1], &sc->regIDs[2], &sc->regIDs[0]);
		VkSub_y(sc, &sc->regIDs[1], &sc->regIDs[0], &sc->regIDs[3]);

		VkSub_x(sc, &sc->regIDs[0], &sc->regIDs[2], &sc->regIDs[0]);
		VkAdd_y(sc, &sc->regIDs[0], &sc->regIDs[0], &sc->regIDs[3]);		
	}
	
	appendRegistersToGlobal(sc, &sc->outputsStruct, &sc->inoutID, &sc->regIDs[0]);
	appendRegistersToGlobal(sc, &sc->outputsStruct, &sc->tempInt, &sc->regIDs[1]);

	
	VkIf_end(sc);
	VkIf_end(sc);	

	appendKernelEnd(sc);

	freeRegisterInitialization_R2C(sc, type);

	return sc->res;
}
#endif
