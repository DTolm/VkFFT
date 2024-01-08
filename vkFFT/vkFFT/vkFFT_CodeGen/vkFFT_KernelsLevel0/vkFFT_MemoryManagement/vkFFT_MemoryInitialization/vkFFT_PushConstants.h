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
#ifndef VKFFT_PUSHCONSTANTS_H
#define VKFFT_PUSHCONSTANTS_H
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_StringManagement/vkFFT_StringManager.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_MathUtils/vkFFT_MathUtils.h"

static inline void appendPushConstant(VkFFTSpecializationConstantsLayout* sc, PfContainer* var) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (var->type > 100) {
		PfContainer* varType = VKFFT_ZERO_INIT;
		PfGetTypeFromCode(sc, var->type, &varType);
		sc->tempLen = sprintf(sc->tempStr, "	%s %s;\n", varType->name, var->name);
		PfAppendLine(sc);
	}
	else {
		sc->res = VKFFT_ERROR_MATH_FAILED;
	}
	return;
}
static inline void appendPushConstants(VkFFTSpecializationConstantsLayout* sc) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (sc->pushConstantsStructSize == 0)
		return;
#if(VKFFT_BACKEND==0)
	sc->tempLen = sprintf(sc->tempStr, "layout(push_constant) uniform PushConsts\n{\n");
	PfAppendLine(sc);
	
#elif(VKFFT_BACKEND==1)
	sc->tempLen = sprintf(sc->tempStr, "	typedef struct {\n");
	PfAppendLine(sc);
	
#elif(VKFFT_BACKEND==2)
	sc->tempLen = sprintf(sc->tempStr, "	typedef struct {\n");
	PfAppendLine(sc);
	
#elif(VKFFT_BACKEND==3)
	sc->tempLen = sprintf(sc->tempStr, "	typedef struct {\n");
	PfAppendLine(sc);
	
#endif
	char tempCopyStr[60];
	if (sc->performWorkGroupShift[0]) {
		appendPushConstant(sc, &sc->workGroupShiftX);
		sprintf(tempCopyStr, "consts.%s", sc->workGroupShiftX.name);
		sprintf(sc->workGroupShiftX.name, "%s", tempCopyStr);
	}
	if (sc->performWorkGroupShift[1]) {
		appendPushConstant(sc, &sc->workGroupShiftY);
		sprintf(tempCopyStr, "consts.%s", sc->workGroupShiftY.name);
		sprintf(sc->workGroupShiftY.name, "%s", tempCopyStr);
	}
	if (sc->performWorkGroupShift[2]) {
		appendPushConstant(sc, &sc->workGroupShiftZ);
		sprintf(tempCopyStr, "consts.%s", sc->workGroupShiftZ.name);
		sprintf(sc->workGroupShiftZ.name, "%s", tempCopyStr);
	}
	if (sc->performPostCompilationInputOffset) {
		appendPushConstant(sc, &sc->inputOffset);
		sprintf(tempCopyStr, "consts.%s", sc->inputOffset.name);
		sprintf(sc->inputOffset.name, "%s", tempCopyStr);
	}
	if (sc->performPostCompilationOutputOffset) {
		appendPushConstant(sc, &sc->outputOffset);
		sprintf(tempCopyStr, "consts.%s", sc->outputOffset.name);
		sprintf(sc->outputOffset.name, "%s", tempCopyStr);
	}
	if (sc->performPostCompilationKernelOffset) {
		appendPushConstant(sc, &sc->kernelOffset);
		sprintf(tempCopyStr, "consts.%s", sc->kernelOffset.name);
		sprintf(sc->kernelOffset.name, "%s", tempCopyStr);
	}
#if(VKFFT_BACKEND==0)
	sc->tempLen = sprintf(sc->tempStr, "} consts;\n\n");
	PfAppendLine(sc);
	
#elif(VKFFT_BACKEND==1)
	sc->tempLen = sprintf(sc->tempStr, "	}PushConsts;\n");
	PfAppendLine(sc);
	//sc->tempLen = sprintf(sc->tempStr, "	__constant__ PushConsts consts;\n");
	//PfAppendLine(sc);
#elif(VKFFT_BACKEND==2)
	sc->tempLen = sprintf(sc->tempStr, "	}PushConsts;\n");
	PfAppendLine(sc);
	
	//sc->tempLen = sprintf(sc->tempStr, "	__constant__ PushConsts consts;\n");
	//PfAppendLine(sc);
	
#elif(VKFFT_BACKEND==3)
	sc->tempLen = sprintf(sc->tempStr, "	}PushConsts;\n");
	PfAppendLine(sc);
	
#endif
	return;
}

#endif
