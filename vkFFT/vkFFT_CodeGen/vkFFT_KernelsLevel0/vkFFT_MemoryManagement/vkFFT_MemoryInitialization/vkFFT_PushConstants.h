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
#include "vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT_CodeGen/vkFFT_StringManagement/vkFFT_StringManager.h"
#include "vkFFT_CodeGen/vkFFT_MathUtils/vkFFT_MathUtils.h"
static inline VkFFTResult appendPushConstant(VkFFTSpecializationConstantsLayout* sc, const char* type, const char* name) {
	VkFFTResult res = VKFFT_SUCCESS;
	sc->tempLen = sprintf(sc->tempStr, "	%s %s;\n", type, name);
	VkAppendLine(sc);
	
	return res;
}
static inline VkFFTResult appendPushConstants(VkFFTSpecializationConstantsLayout* sc) {
	VkFFTResult res = VKFFT_SUCCESS;
	if (sc->pushConstantsStructSize == 0)
		return res;
	VkContainer* intType;
	VkGetTypeFromCode(sc, sc->intTypeCode, &intType);
#if(VKFFT_BACKEND==0)
	sc->tempLen = sprintf(sc->tempStr, "layout(push_constant) uniform PushConsts\n{\n");
	VkAppendLine(sc);
	
#elif(VKFFT_BACKEND==1)
	sc->tempLen = sprintf(sc->tempStr, "	typedef struct {\n");
	VkAppendLine(sc);
	
#elif(VKFFT_BACKEND==2)
	sc->tempLen = sprintf(sc->tempStr, "	typedef struct {\n");
	VkAppendLine(sc);
	
#elif(VKFFT_BACKEND==3)
	sc->tempLen = sprintf(sc->tempStr, "	typedef struct {\n");
	VkAppendLine(sc);
	
#endif
	if (sc->performWorkGroupShift[0]) {
		res = appendPushConstant(sc, intType->data.s, "workGroupShiftX");
		
	}
	if (sc->performWorkGroupShift[1]) {
		res = appendPushConstant(sc, intType->data.s, "workGroupShiftY");
		
	}
	if (sc->performWorkGroupShift[2]) {
		res = appendPushConstant(sc, intType->data.s, "workGroupShiftZ");
		
	}
	if (sc->performPostCompilationInputOffset) {
		res = appendPushConstant(sc, intType->data.s, "inputOffset");
		
	}
	if (sc->performPostCompilationOutputOffset) {
		res = appendPushConstant(sc, intType->data.s, "outputOffset");
		
	}
	if (sc->performPostCompilationKernelOffset) {
		res = appendPushConstant(sc, intType->data.s, "kernelOffset");
		
	}
#if(VKFFT_BACKEND==0)
	sc->tempLen = sprintf(sc->tempStr, "} consts;\n\n");
	VkAppendLine(sc);
	
#elif(VKFFT_BACKEND==1)
	sc->tempLen = sprintf(sc->tempStr, "	}PushConsts;\n");
	VkAppendLine(sc);
	sc->tempLen = sprintf(sc->tempStr, "	__constant__ PushConsts consts;\n");
	VkAppendLine(sc);
#elif(VKFFT_BACKEND==2)
	sc->tempLen = sprintf(sc->tempStr, "	}PushConsts;\n");
	VkAppendLine(sc);
	
	sc->tempLen = sprintf(sc->tempStr, "	__constant__ PushConsts consts;\n");
	VkAppendLine(sc);
	
#elif(VKFFT_BACKEND==3)
	sc->tempLen = sprintf(sc->tempStr, "	}PushConsts;\n");
	VkAppendLine(sc);
	
#endif
	return res;
}
#endif
