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
#ifndef VKFFT_TRANSFERS_H
#define VKFFT_TRANSFERS_H
#include "vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT_CodeGen/vkFFT_StringManagement/vkFFT_StringManager.h"

static inline void appendSharedToRegisters(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* sdataID)
{
	if (sc->res != VKFFT_SUCCESS) return;
	sc->tempLen = sprintf(sc->tempStr, "\
%s = sdata[%s];\n", out->data.s, sdataID->data.s);
	VkAppendLine(sc);
	return;
}
static inline void appendSharedToRegisters_x_x(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* sdataID)
{
	if (sc->res != VKFFT_SUCCESS) return;
	sc->tempLen = sprintf(sc->tempStr, "\
%s.x = sdata[%s].x;\n", out->data.s, sdataID->data.s);
	VkAppendLine(sc);
	return;
}
static inline void appendSharedToRegisters_x_y(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* sdataID)
{
	if (sc->res != VKFFT_SUCCESS) return;
	sc->tempLen = sprintf(sc->tempStr, "\
%s.x = sdata[%s].y;\n", out->data.s, sdataID->data.s);
	VkAppendLine(sc);
	return;
}
static inline void appendSharedToRegisters_y_x(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* sdataID)
{
	if (sc->res != VKFFT_SUCCESS) return;
	sc->tempLen = sprintf(sc->tempStr, "\
%s.y = sdata[%s].x;\n", out->data.s, sdataID->data.s);
	VkAppendLine(sc);
	return;
}
static inline void appendSharedToRegisters_y_y(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* sdataID)
{
	if (sc->res != VKFFT_SUCCESS) return;
	sc->tempLen = sprintf(sc->tempStr, "\
%s.y = sdata[%s].y;\n", out->data.s, sdataID->data.s);
	VkAppendLine(sc);
	return;
}

static inline void appendRegistersToShared(VkFFTSpecializationConstantsLayout* sc, VkContainer* sdataID, VkContainer* out)
{
	if (sc->res != VKFFT_SUCCESS) return;
	sc->tempLen = sprintf(sc->tempStr, "\
sdata[%s] = %s;\n", sdataID->data.s, out->data.s);
	VkAppendLine(sc);
	return;
}
static inline void appendRegistersToShared_x_x(VkFFTSpecializationConstantsLayout* sc, VkContainer* sdataID, VkContainer* out)
{
	if (sc->res != VKFFT_SUCCESS) return;
	sc->tempLen = sprintf(sc->tempStr, "\
sdata[%s].x = %s.x;\n", sdataID->data.s, out->data.s);
	VkAppendLine(sc);
	return;
}
static inline void appendRegistersToShared_x_y(VkFFTSpecializationConstantsLayout* sc, VkContainer* sdataID, VkContainer* out)
{
	if (sc->res != VKFFT_SUCCESS) return;
	sc->tempLen = sprintf(sc->tempStr, "\
sdata[%s].x = %s.y;\n", sdataID->data.s, out->data.s);
	VkAppendLine(sc);
	return;
}
static inline void appendRegistersToShared_y_y(VkFFTSpecializationConstantsLayout* sc, VkContainer* sdataID, VkContainer* out)
{
	if (sc->res != VKFFT_SUCCESS) return;
	sc->tempLen = sprintf(sc->tempStr, "\
sdata[%s].y = %s.y;\n", sdataID->data.s, out->data.s);
	VkAppendLine(sc);
	return;
}
static inline void appendRegistersToShared_y_x(VkFFTSpecializationConstantsLayout* sc, VkContainer* sdataID, VkContainer* out)
{
	if (sc->res != VKFFT_SUCCESS) return;
	sc->tempLen = sprintf(sc->tempStr, "\
sdata[%s].y = %s.x;\n", sdataID->data.s, out->data.s);
	VkAppendLine(sc);
	return;
}

static inline void appendConstantToRegisters(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* constantBufferName, VkContainer* inoutID)
{
	if (sc->res != VKFFT_SUCCESS) return;
	sc->tempLen = sprintf(sc->tempStr, "\
%s = %s[%s];\n", out->data.s, constantBufferName->data.s, inoutID->data.s);
	VkAppendLine(sc);
	return;
}
static inline void appendConstantToRegisters_x(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* constantBufferName, VkContainer* inoutID)
{
	if (sc->res != VKFFT_SUCCESS) return;
	sc->tempLen = sprintf(sc->tempStr, "\
%s.x = %s[%s];\n", out->data.s, constantBufferName->data.s, inoutID->data.s);
	VkAppendLine(sc);
	return;
}
static inline void appendConstantToRegisters_y(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* constantBufferName, VkContainer* inoutID)
{
	if (sc->res != VKFFT_SUCCESS) return;
	sc->tempLen = sprintf(sc->tempStr, "\
%s.y = %s[%s];\n", out->data.s, constantBufferName->data.s, inoutID->data.s);
	VkAppendLine(sc);
	return;
}

static inline void appendGlobalToRegisters(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* bufferName, VkContainer* inoutID)
{
	if (sc->res != VKFFT_SUCCESS) return;
	sc->tempLen = sprintf(sc->tempStr, "%s", out->data.s);
	VkAppendLine(sc);
	sc->tempLen = sprintf(sc->tempStr, " = ");
	VkAppendLine(sc);
	VkAppendConversionStart(sc, out, bufferName);
	if ((!(strcmp(bufferName->data.s, sc->inputsStruct.data.s))) && (sc->inputBufferBlockNum != 1)) {
		sc->tempLen = sprintf(sc->tempStr, "inputBlocks[%s / %" PRIu64 "].%s[%s %% %" PRIu64 "]", inoutID->data.s, sc->inputBufferBlockSize, bufferName->data.s, inoutID->data.s, sc->inputBufferBlockSize);
	}
	else if ((!(strcmp(bufferName->data.s, sc->outputsStruct.data.s))) && (sc->outputBufferBlockNum != 1)) {
		sc->tempLen = sprintf(sc->tempStr, "outputBlocks[%s / %" PRIu64 "].%s[%s %% %" PRIu64 "]", inoutID->data.s, sc->outputBufferBlockSize, bufferName->data.s, inoutID->data.s, sc->outputBufferBlockSize);
	}
	else if ((!(strcmp(bufferName->data.s, sc->kernelStruct.data.s))) && (sc->kernelBlockNum != 1)) {
		sc->tempLen = sprintf(sc->tempStr, "kernelBlocks[%s / %" PRIu64 "].%s[%s %% %" PRIu64 "]", inoutID->data.s, sc->kernelBlockSize, bufferName->data.s, inoutID->data.s, sc->kernelBlockSize);
	}
	else {
		sc->tempLen = sprintf(sc->tempStr, "%s[%s]", bufferName->data.s, inoutID->data.s);
	}
	VkAppendLine(sc);
	VkAppendConversionEnd(sc, out, bufferName);
	sc->tempLen = sprintf(sc->tempStr, ";\n");
	VkAppendLine(sc);
	return;
}
static inline void appendGlobalToRegisters_x(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* bufferName, VkContainer* inoutID)
{
	if (sc->res != VKFFT_SUCCESS) return;
	VkContainer* floatType;
	VkGetTypeFromCode(sc, out->type - 1, &floatType);

	sc->tempLen = sprintf(sc->tempStr, "%s.x", out->data.s);
	VkAppendLine(sc);
	sc->tempLen = sprintf(sc->tempStr, " = ");
	VkAppendLine(sc);
	VkAppendConversionStart(sc, floatType, bufferName);
	if ((!(strcmp(bufferName->data.s, sc->inputsStruct.data.s))) && (sc->inputBufferBlockNum != 1)) {
		sc->tempLen = sprintf(sc->tempStr, "inputBlocks[%s / %" PRIu64 "].%s[%s %% %" PRIu64 "]", inoutID->data.s, sc->inputBufferBlockSize, bufferName->data.s, inoutID->data.s, sc->inputBufferBlockSize);
	}
	else if ((!(strcmp(bufferName->data.s, sc->outputsStruct.data.s))) && (sc->outputBufferBlockNum != 1)) {
		sc->tempLen = sprintf(sc->tempStr, "outputBlocks[%s / %" PRIu64 "].%s[%s %% %" PRIu64 "]", inoutID->data.s, sc->outputBufferBlockSize, bufferName->data.s, inoutID->data.s, sc->outputBufferBlockSize);
	}
	else if ((!(strcmp(bufferName->data.s, sc->kernelStruct.data.s))) && (sc->kernelBlockNum != 1)) {
		sc->tempLen = sprintf(sc->tempStr, "kernelBlocks[%s / %" PRIu64 "].%s[%s %% %" PRIu64 "]", inoutID->data.s, sc->kernelBlockSize, bufferName->data.s, inoutID->data.s, sc->kernelBlockSize);
	}
	else {
		sc->tempLen = sprintf(sc->tempStr, "%s[%s]", bufferName->data.s, inoutID->data.s);
	}
	VkAppendLine(sc);
	VkAppendConversionEnd(sc, floatType, bufferName);
	sc->tempLen = sprintf(sc->tempStr, ";\n");
	VkAppendLine(sc);
	return;
}
static inline void appendGlobalToRegisters_y(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* bufferName, VkContainer* inoutID)
{
	if (sc->res != VKFFT_SUCCESS) return;
	VkContainer* floatType;
	VkGetTypeFromCode(sc, out->type - 1, &floatType);

	sc->tempLen = sprintf(sc->tempStr, "%s.y", out->data.s);
	VkAppendLine(sc);
	sc->tempLen = sprintf(sc->tempStr, " = ");
	VkAppendLine(sc);
	VkAppendConversionStart(sc, floatType, bufferName);
	if ((!(strcmp(bufferName->data.s, sc->inputsStruct.data.s))) && (sc->inputBufferBlockNum != 1)) {
		sc->tempLen = sprintf(sc->tempStr, "inputBlocks[%s / %" PRIu64 "].%s[%s %% %" PRIu64 "]", inoutID->data.s, sc->inputBufferBlockSize, bufferName->data.s, inoutID->data.s, sc->inputBufferBlockSize);
	}
	else if ((!(strcmp(bufferName->data.s, sc->outputsStruct.data.s))) && (sc->outputBufferBlockNum != 1)) {
		sc->tempLen = sprintf(sc->tempStr, "outputBlocks[%s / %" PRIu64 "].%s[%s %% %" PRIu64 "]", inoutID->data.s, sc->outputBufferBlockSize, bufferName->data.s, inoutID->data.s, sc->outputBufferBlockSize);
	}
	else if ((!(strcmp(bufferName->data.s, sc->kernelStruct.data.s))) && (sc->kernelBlockNum != 1)) {
		sc->tempLen = sprintf(sc->tempStr, "kernelBlocks[%s / %" PRIu64 "].%s[%s %% %" PRIu64 "]", inoutID->data.s, sc->kernelBlockSize, bufferName->data.s, inoutID->data.s, sc->kernelBlockSize);
	}
	else {
		sc->tempLen = sprintf(sc->tempStr, "%s[%s]", bufferName->data.s, inoutID->data.s);
	}
	VkAppendLine(sc);
	VkAppendConversionEnd(sc, floatType, bufferName);
	sc->tempLen = sprintf(sc->tempStr, ";\n");
	VkAppendLine(sc);
	return;
}

static inline void appendRegistersToGlobal(VkFFTSpecializationConstantsLayout* sc, VkContainer* bufferName, VkContainer* inoutID, VkContainer* in)
{
	if (sc->res != VKFFT_SUCCESS) return;
	if ((!(strcmp(bufferName->data.s, sc->inputsStruct.data.s))) && (sc->inputBufferBlockNum != 1)) {
		sc->tempLen = sprintf(sc->tempStr, "inputBlocks[%s / %" PRIu64 "].%s[%s %% %" PRIu64 "]", inoutID->data.s, sc->inputBufferBlockSize, bufferName->data.s, inoutID->data.s, sc->inputBufferBlockSize);
	}
	else if ((!(strcmp(bufferName->data.s, sc->outputsStruct.data.s))) && (sc->outputBufferBlockNum != 1)) {
		sc->tempLen = sprintf(sc->tempStr, "outputBlocks[%s / %" PRIu64 "].%s[%s %% %" PRIu64 "]", inoutID->data.s, sc->outputBufferBlockSize, bufferName->data.s, inoutID->data.s, sc->outputBufferBlockSize);
	}
	else if ((!(strcmp(bufferName->data.s, sc->kernelStruct.data.s))) && (sc->kernelBlockNum != 1)) {
		sc->tempLen = sprintf(sc->tempStr, "kernelBlocks[%s / %" PRIu64 "].%s[%s %% %" PRIu64 "]", inoutID->data.s, sc->kernelBlockSize, bufferName->data.s, inoutID->data.s, sc->kernelBlockSize);
	}
	else {
		sc->tempLen = sprintf(sc->tempStr, "%s[%s]", bufferName->data.s, inoutID->data.s);
	}
	VkAppendLine(sc);
	sc->tempLen = sprintf(sc->tempStr, " = ");
	VkAppendLine(sc);
	VkAppendConversionStart(sc, bufferName, in);
	sc->tempLen = sprintf(sc->tempStr, "%s", in->data.s);
	VkAppendLine(sc);
	VkAppendConversionEnd(sc, bufferName, in);
	sc->tempLen = sprintf(sc->tempStr, ";\n");
	VkAppendLine(sc);
	return;
}
static inline void appendRegistersToGlobal_x(VkFFTSpecializationConstantsLayout* sc, VkContainer* bufferName, VkContainer* inoutID, VkContainer* in)
{
	if (sc->res != VKFFT_SUCCESS) return;
	VkContainer* floatType;
	VkGetTypeFromCode(sc, in->type - 1, &floatType);
	
	if ((!(strcmp(bufferName->data.s, sc->inputsStruct.data.s))) && (sc->inputBufferBlockNum != 1)) {
		sc->tempLen = sprintf(sc->tempStr, "inputBlocks[%s / %" PRIu64 "].%s[%s %% %" PRIu64 "]", inoutID->data.s, sc->inputBufferBlockSize, bufferName->data.s, inoutID->data.s, sc->inputBufferBlockSize);
	}
	else if ((!(strcmp(bufferName->data.s, sc->outputsStruct.data.s))) && (sc->outputBufferBlockNum != 1)) {
		sc->tempLen = sprintf(sc->tempStr, "outputBlocks[%s / %" PRIu64 "].%s[%s %% %" PRIu64 "]", inoutID->data.s, sc->outputBufferBlockSize, bufferName->data.s, inoutID->data.s, sc->outputBufferBlockSize);
	}
	else if ((!(strcmp(bufferName->data.s, sc->kernelStruct.data.s))) && (sc->kernelBlockNum != 1)) {
		sc->tempLen = sprintf(sc->tempStr, "kernelBlocks[%s / %" PRIu64 "].%s[%s %% %" PRIu64 "]", inoutID->data.s, sc->kernelBlockSize, bufferName->data.s, inoutID->data.s, sc->kernelBlockSize);
	}
	else {
		sc->tempLen = sprintf(sc->tempStr, "%s[%s]", bufferName->data.s, inoutID->data.s);
	}
	VkAppendLine(sc);
	sc->tempLen = sprintf(sc->tempStr, " = ");
	VkAppendLine(sc);
	VkAppendConversionStart(sc, bufferName, floatType);
	sc->tempLen = sprintf(sc->tempStr, "%s.x", in->data.s);
	VkAppendLine(sc);
	VkAppendConversionEnd(sc, bufferName, floatType);
	sc->tempLen = sprintf(sc->tempStr, ";\n");
	VkAppendLine(sc);
	return;
}
static inline void appendRegistersToGlobal_y(VkFFTSpecializationConstantsLayout* sc, VkContainer* bufferName, VkContainer* inoutID, VkContainer* in)
{
	if (sc->res != VKFFT_SUCCESS) return;
	VkContainer* floatType;
	VkGetTypeFromCode(sc, in->type - 1, &floatType);

	if ((!(strcmp(bufferName->data.s, sc->inputsStruct.data.s))) && (sc->inputBufferBlockNum != 1)) {
		sc->tempLen = sprintf(sc->tempStr, "inputBlocks[%s / %" PRIu64 "].%s[%s %% %" PRIu64 "]", inoutID->data.s, sc->inputBufferBlockSize, bufferName->data.s, inoutID->data.s, sc->inputBufferBlockSize);
	}
	else if ((!(strcmp(bufferName->data.s, sc->outputsStruct.data.s))) && (sc->outputBufferBlockNum != 1)) {
		sc->tempLen = sprintf(sc->tempStr, "outputBlocks[%s / %" PRIu64 "].%s[%s %% %" PRIu64 "]", inoutID->data.s, sc->outputBufferBlockSize, bufferName->data.s, inoutID->data.s, sc->outputBufferBlockSize);
	}
	else if ((!(strcmp(bufferName->data.s, sc->kernelStruct.data.s))) && (sc->kernelBlockNum != 1)) {
		sc->tempLen = sprintf(sc->tempStr, "kernelBlocks[%s / %" PRIu64 "].%s[%s %% %" PRIu64 "]", inoutID->data.s, sc->kernelBlockSize, bufferName->data.s, inoutID->data.s, sc->kernelBlockSize);
	}
	else {
		sc->tempLen = sprintf(sc->tempStr, "%s[%s]", bufferName->data.s, inoutID->data.s);
	}
	VkAppendLine(sc);
	sc->tempLen = sprintf(sc->tempStr, " = ");
	VkAppendLine(sc);
	VkAppendConversionStart(sc, bufferName, floatType);
	sc->tempLen = sprintf(sc->tempStr, "%s.y", in->data.s);
	VkAppendLine(sc);
	VkAppendConversionEnd(sc, bufferName, floatType);
	sc->tempLen = sprintf(sc->tempStr, ";\n");
	VkAppendLine(sc);
	return;
}

static inline void appendGlobalToShared(VkFFTSpecializationConstantsLayout* sc, VkContainer* sdataID, VkContainer* bufferName, VkContainer* inoutID)
{
	if (sc->res != VKFFT_SUCCESS) return;
	sc->tempLen = sprintf(sc->tempStr, "sdata[%s]", sdataID->data.s);
	VkAppendLine(sc);
	sc->tempLen = sprintf(sc->tempStr, " = ");
	VkAppendLine(sc);
	VkAppendConversionStart(sc, &sc->sdataStruct, bufferName);
	if ((!(strcmp(bufferName->data.s, sc->inputsStruct.data.s))) && (sc->inputBufferBlockNum != 1)) {
		sc->tempLen = sprintf(sc->tempStr, "inputBlocks[%s / %" PRIu64 "].%s[%s %% %" PRIu64 "]", inoutID->data.s, sc->inputBufferBlockSize, bufferName->data.s, inoutID->data.s, sc->inputBufferBlockSize);
	}
	else if ((!(strcmp(bufferName->data.s, sc->outputsStruct.data.s))) && (sc->outputBufferBlockNum != 1)) {
		sc->tempLen = sprintf(sc->tempStr, "outputBlocks[%s / %" PRIu64 "].%s[%s %% %" PRIu64 "]", inoutID->data.s, sc->outputBufferBlockSize, bufferName->data.s, inoutID->data.s, sc->outputBufferBlockSize);
	}
	else if ((!(strcmp(bufferName->data.s, sc->kernelStruct.data.s))) && (sc->kernelBlockNum != 1)) {
		sc->tempLen = sprintf(sc->tempStr, "kernelBlocks[%s / %" PRIu64 "].%s[%s %% %" PRIu64 "]", inoutID->data.s, sc->kernelBlockSize, bufferName->data.s, inoutID->data.s, sc->kernelBlockSize);
	}
	else {
		sc->tempLen = sprintf(sc->tempStr, "%s[%s]", bufferName->data.s, inoutID->data.s);
	}
	VkAppendLine(sc);
	VkAppendConversionEnd(sc, &sc->sdataStruct, bufferName);
	sc->tempLen = sprintf(sc->tempStr, ";\n");
	VkAppendLine(sc);
	return;
}
static inline void appendSharedToGlobal(VkFFTSpecializationConstantsLayout* sc, VkContainer* bufferName, VkContainer* inoutID, VkContainer* sdataID)
{
	if (sc->res != VKFFT_SUCCESS) return;
	if ((!(strcmp(bufferName->data.s, sc->inputsStruct.data.s))) && (sc->inputBufferBlockNum != 1)) {
		sc->tempLen = sprintf(sc->tempStr, "inputBlocks[%s / %" PRIu64 "].%s[%s %% %" PRIu64 "]", inoutID->data.s, sc->inputBufferBlockSize, bufferName->data.s, inoutID->data.s, sc->inputBufferBlockSize);
	}
	else if ((!(strcmp(bufferName->data.s, sc->outputsStruct.data.s))) && (sc->outputBufferBlockNum != 1)) {
		sc->tempLen = sprintf(sc->tempStr, "outputBlocks[%s / %" PRIu64 "].%s[%s %% %" PRIu64 "]", inoutID->data.s, sc->outputBufferBlockSize, bufferName->data.s, inoutID->data.s, sc->outputBufferBlockSize);
	}
	else if ((!(strcmp(bufferName->data.s, sc->kernelStruct.data.s))) && (sc->kernelBlockNum != 1)) {
		sc->tempLen = sprintf(sc->tempStr, "kernelBlocks[%s / %" PRIu64 "].%s[%s %% %" PRIu64 "]", inoutID->data.s, sc->kernelBlockSize, bufferName->data.s, inoutID->data.s, sc->kernelBlockSize);
	}
	else {
		sc->tempLen = sprintf(sc->tempStr, "%s[%s]", bufferName->data.s, inoutID->data.s);
	}
	VkAppendLine(sc);
	sc->tempLen = sprintf(sc->tempStr, " = ");
	VkAppendLine(sc);
	VkAppendConversionStart(sc, bufferName, &sc->sdataStruct);
	sc->tempLen = sprintf(sc->tempStr, "sdata[%s]", sdataID->data.s);
	VkAppendLine(sc);
	VkAppendConversionEnd(sc, bufferName, &sc->sdataStruct);
	sc->tempLen = sprintf(sc->tempStr, ";\n");
	VkAppendLine(sc);
	return;
}
static inline void appendSetSMToZero(VkFFTSpecializationConstantsLayout* sc) {
	VkFFTResult res = VKFFT_SUCCESS;

	VkContainer temp_int = {};
	temp_int.type = 31;

	VkContainer temp_int1 = {};
	temp_int1.type = 31;

	VkContainer used_registers = {};
	used_registers.type = 31;
	temp_int.data.i = sc->localSize[0].data.i * sc->localSize[1].data.i;
	temp_int1.data.i = sc->usedSharedMemory.data.i / sc->complexSize;
	VkDivCeil(sc, &used_registers, &temp_int1, &temp_int);
	for (int64_t i = 0; i < used_registers.data.i; i++) {
		if (sc->localSize[1].data.i == 1) {
			temp_int.data.i = (i)*sc->localSize[0].data.i;

			VkAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
		}
		else {
			VkMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

			temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

			VkAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
			VkAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
		}

		temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
		temp_int1.data.i = sc->usedSharedMemory.data.i / sc->complexSize;
		if (temp_int.data.i > temp_int1.data.i) {
			//check that we only read fftDim * local batch data
			//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
			VkIf_lt_start(sc, &sc->combinedID, &temp_int1);
		}
	
		VkSetToZeroShared(sc, &sc->combinedID);

		temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
		temp_int1.data.i = sc->usedSharedMemory.data.i / sc->complexSize;
		if (temp_int.data.i > temp_int1.data.i) {
			//check that we only read fftDim * local batch data
			//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
			VkIf_end(sc);
		}
	}


	//res = appendZeropadEnd(sc);
	//if (res != VKFFT_SUCCESS) return res;
	return;
}

#endif
