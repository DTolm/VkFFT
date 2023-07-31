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
#ifndef VKFFT_INPUTOUTPUT_H
#define VKFFT_INPUTOUTPUT_H
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_StringManagement/vkFFT_StringManager.h"
/*
static inline VkFFTResult indexInputVkFFT(VkFFTSpecializationConstantsLayout* sc, PfContainer* output, PfContainer* index_x, PfContainer* index_y, int coordinate, int batchID) {
	VkFFTResult res = VKFFT_SUCCESS;
	switch (sc->inputType % 1000) {
	case 0: case 2: case 3: case 4:case 5: case 6: case 110: case 120: case 130: case 140: case 142: case 144: {//single_c2c + single_c2c_strided
		PfContainer locOffset;
		if (sc->inputOffset.type == 1) {
			locOffset.type = 1;
			locOffset.data.i = sc->inputOffset.data.i / sc->inputNumberByteSize;
		}
		else {
			if (sc->inputOffset.type == 1001) {
				if (sc->performPostCompilationInputOffset) {
					locOffset.type = 1001;
					if (sc->inputType < 1000) {
						locOffset.data.s = sc->inputOffset.data.s;
					}
					else {
						locOffset.data.s = sc->kernelOffset.data.s;
					}
				}
			}
		}
		if (sc->inputStride[0].data.i != 1)
			PfMul(sc, index_x, index_x, &sc->inputStride[0], 0);

		int mult = (sc->mergeSequencesR2C) ? 2 : 1;
		if (sc->size[1].data.i > 1) {
			if (sc->numAxisUploads == 1) {
				if (sc->axisSwapped) {
					if (sc->performWorkGroupShift[1])
						sprintf(shiftY, " + (%s + consts.workGroupShiftY) * %" PRIu64 "", sc->gl_WorkGroupID_y, mult * sc->localSize[0].data.i * sc->inputStride[1].data.i);
					else
						sprintf(shiftY, " + %s * %" PRIu64 "", sc->gl_WorkGroupID_y, mult * sc->localSize[0] * sc->inputStride[1]);
				}
				else {
					if (sc->performWorkGroupShift[1])
						sprintf(shiftY, " + (%s + consts.workGroupShiftY) * %" PRIu64 "", sc->gl_WorkGroupID_y, mult * sc->localSize[1] * sc->inputStride[1]);
					else
						sprintf(shiftY, " + %s * %" PRIu64 "", sc->gl_WorkGroupID_y, mult * sc->localSize[1] * sc->inputStride[1]);
				}
			}
			else {
				if (sc->performWorkGroupShift[1])
					sprintf(shiftY, " + (%s + consts.workGroupShiftY) * %" PRIu64 "", sc->gl_WorkGroupID_y, sc->inputStride[1]);
				else
					sprintf(shiftY, " + %s * %" PRIu64 "", sc->gl_WorkGroupID_y, sc->inputStride[1]);
			}
		}
		char shiftZ[500] = "";
		if (sc->size[2] > 1) {
			if (sc->numCoordinates * sc->matrixConvolution * sc->numBatches > 1) {
				if (sc->performWorkGroupShift[2])
					sprintf(shiftZ, " + ((%s + consts.workGroupShiftZ * %s) %% %" PRIu64 ") * %" PRIu64 "", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z, sc->dispatchZactualFFTSize, sc->inputStride[2]);
				else
					sprintf(shiftZ, " + (%s %% %" PRIu64 ") * %" PRIu64 "", sc->gl_GlobalInvocationID_z, sc->dispatchZactualFFTSize, sc->inputStride[2]);
			}
			else {
				if (sc->performWorkGroupShift[2])
					sprintf(shiftZ, " + (%s + consts.workGroupShiftZ * %s) * %" PRIu64 "", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z, sc->inputStride[2]);
				else
					sprintf(shiftZ, " + %s * %" PRIu64 "", sc->gl_GlobalInvocationID_z, sc->inputStride[2]);
			}
		}
		char shiftCoordinate[500] = "";
		uint64_t maxCoordinate = sc->numCoordinates * sc->matrixConvolution;
		if (sc->numCoordinates * sc->matrixConvolution > 1) {
			sprintf(shiftCoordinate, " + ((%s / %" PRIu64 ") %% %" PRIu64 ") * %" PRIu64 "", sc->gl_GlobalInvocationID_z, sc->dispatchZactualFFTSize, maxCoordinate, sc->inputStride[3]);
		}
		if ((sc->matrixConvolution > 1) && (sc->convolutionStep)) {
			maxCoordinate = 1;
			sprintf(shiftCoordinate, " + %s * %" PRIu64 "", coordinate, sc->inputStride[3]);
		}
		char shiftBatch[500] = "";
		if ((sc->numBatches > 1) || (sc->numKernels > 1)) {
			if (sc->convolutionStep && (sc->numKernels > 1)) {
				sprintf(shiftBatch, " + %s * %" PRIu64 "", batchID, sc->inputStride[4]);
			}
			else
				sprintf(shiftBatch, " + (%s / %" PRIu64 ") * %" PRIu64 "", sc->gl_GlobalInvocationID_z, sc->dispatchZactualFFTSize * maxCoordinate, sc->inputStride[4]);
		}
		sc->tempLen = sprintf(sc->tempStr, "%s%s%s%s%s%s", inputOffset, shiftX, shiftY, shiftZ, shiftCoordinate, shiftBatch);
		res = PfAppendLine(sc);
		if (res != VKFFT_SUCCESS) return res;
		break;
	}
	case 1: case 111: case 121: case 131: case 141: case 143: case 145: {//grouped_c2c
		char inputOffset[30] = "";
		if (sc->inputOffset > 0) {
			sprintf(inputOffset, "%" PRIu64 " + ", sc->inputOffset / sc->inputNumberByteSize);
		}
		else {
			if (sc->performPostCompilationInputOffset) {
				if (sc->inputType < 1000)
					sprintf(inputOffset, "consts.inputOffset + ");
				else
					sprintf(inputOffset, "consts.kernelOffset + ");
			}
		}
		char shiftX[500] = "";
		if (sc->inputStride[0] == 1)
			sprintf(shiftX, "(%s)", index_x);
		else
			sprintf(shiftX, "(%s) * %" PRIu64 "", index_x, sc->inputStride[0]);

		char shiftY[500] = "";
		if (index_y)
			sprintf(shiftY, " + (%s) * %" PRIu64 "", index_y, sc->inputStride[1]);

		char shiftZ[500] = "";
		if (sc->size[2] > 1) {
			if (sc->numCoordinates * sc->matrixConvolution * sc->numBatches > 1) {
				if (sc->performWorkGroupShift[2])
					sprintf(shiftZ, " + ((%s + consts.workGroupShiftZ * %s) %% %" PRIu64 ") * %" PRIu64 "", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z, sc->dispatchZactualFFTSize, sc->inputStride[2]);
				else
					sprintf(shiftZ, " + (%s %% %" PRIu64 ") * %" PRIu64 "", sc->gl_GlobalInvocationID_z, sc->dispatchZactualFFTSize, sc->inputStride[2]);
			}
			else {
				if (sc->performWorkGroupShift[2])
					sprintf(shiftZ, " + (%s + consts.workGroupShiftZ * %s) * %" PRIu64 "", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z, sc->inputStride[2]);
				else
					sprintf(shiftZ, " + %s * %" PRIu64 "", sc->gl_GlobalInvocationID_z, sc->inputStride[2]);
			}
		}
		char shiftCoordinate[500] = "";
		uint64_t maxCoordinate = sc->numCoordinates * sc->matrixConvolution;
		if (sc->numCoordinates * sc->matrixConvolution > 1) {
			sprintf(shiftCoordinate, " + ((%s / %" PRIu64 ") %% %" PRIu64 ") * %" PRIu64 "", sc->gl_GlobalInvocationID_z, sc->dispatchZactualFFTSize, maxCoordinate, sc->inputStride[3]);
		}
		if ((sc->matrixConvolution > 1) && (sc->convolutionStep)) {
			maxCoordinate = 1;
			sprintf(shiftCoordinate, " + %s * %" PRIu64 "", coordinate, sc->inputStride[3]);
		}
		char shiftBatch[500] = "";
		if ((sc->numBatches > 1) || (sc->numKernels > 1)) {
			if (sc->convolutionStep && (sc->numKernels > 1)) {
				sprintf(shiftBatch, " + %s * %" PRIu64 "", batchID, sc->inputStride[4]);
			}
			else
				sprintf(shiftBatch, " + (%s / %" PRIu64 ") * %" PRIu64 "", sc->gl_GlobalInvocationID_z, sc->dispatchZactualFFTSize * maxCoordinate, sc->inputStride[4]);
		}
		sc->tempLen = sprintf(sc->tempStr, "%s%s%s%s%s%s", inputOffset, shiftX, shiftY, shiftZ, shiftCoordinate, shiftBatch);
		res = PfAppendLine(sc);
		if (res != VKFFT_SUCCESS) return res;
		break;
	}
	}
	return res;
}
static inline VkFFTResult indexOutputVkFFT(VkFFTSpecializationConstantsLayout* sc, const char* index_x, const char* index_y, const char* coordinate, const char* batchID) {
	VkFFTResult res = VKFFT_SUCCESS;
	switch (sc->outputType % 1000) {//single_c2c + single_c2c_strided
	case 0: case 2: case 3: case 4: case 5: case 6: case 110: case 120: case 130: case 140: case 142: case 144: {
		char outputOffset[30] = "";
		if (sc->outputOffset > 0) {
			sprintf(outputOffset, "%" PRIu64 " + ", sc->outputOffset / sc->outputNumberByteSize);
		}
		else {
			if (sc->performPostCompilationOutputOffset) {
				if (sc->outputType < 1000)
					sprintf(outputOffset, "consts.outputOffset + ");
				else
					sprintf(outputOffset, "consts.kernelOffset + ");
			}
		}
		char shiftX[500] = "";
		if (sc->numAxisUploads == 1)
			sprintf(shiftX, "(%s)", index_x);
		else
			sprintf(shiftX, "(%s) * %" PRIu64 "", index_x, sc->outputStride[0]);
		char shiftY[500] = "";
		uint64_t mult = (sc->mergeSequencesR2C) ? 2 : 1;
		if (sc->size[1] > 1) {
			if (sc->numAxisUploads == 1) {
				if (sc->axisSwapped) {
					if (sc->performWorkGroupShift[1])
						sprintf(shiftY, " + (%s + consts.workGroupShiftY) * %" PRIu64 "", sc->gl_WorkGroupID_y, mult * sc->localSize[0] * sc->outputStride[1]);
					else
						sprintf(shiftY, " + %s * %" PRIu64 "", sc->gl_WorkGroupID_y, mult * sc->localSize[0] * sc->outputStride[1]);
				}
				else {
					if (sc->performWorkGroupShift[1])
						sprintf(shiftY, " + (%s + consts.workGroupShiftY) * %" PRIu64 "", sc->gl_WorkGroupID_y, mult * sc->localSize[1] * sc->outputStride[1]);
					else
						sprintf(shiftY, " + %s * %" PRIu64 "", sc->gl_WorkGroupID_y, mult * sc->localSize[1] * sc->outputStride[1]);
				}
			}
			else {
				if (sc->performWorkGroupShift[1])
					sprintf(shiftY, " + (%s + consts.workGroupShiftY) * %" PRIu64 "", sc->gl_WorkGroupID_y, sc->outputStride[1]);
				else
					sprintf(shiftY, " + %s * %" PRIu64 "", sc->gl_WorkGroupID_y, sc->outputStride[1]);
			}
		}
		char shiftZ[500] = "";
		if (sc->size[2] > 1) {
			if (sc->numCoordinates * sc->matrixConvolution * sc->numBatches > 1) {
				if (sc->performWorkGroupShift[2])
					sprintf(shiftZ, " + ((%s + consts.workGroupShiftZ * %s) %% %" PRIu64 ") * %" PRIu64 "", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z, sc->dispatchZactualFFTSize, sc->outputStride[2]);
				else
					sprintf(shiftZ, " + (%s %% %" PRIu64 ") * %" PRIu64 "", sc->gl_GlobalInvocationID_z, sc->dispatchZactualFFTSize, sc->outputStride[2]);
			}
			else {
				if (sc->performWorkGroupShift[2])
					sprintf(shiftZ, " + (%s + consts.workGroupShiftZ * %s) * %" PRIu64 "", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z, sc->outputStride[2]);
				else
					sprintf(shiftZ, " + %s * %" PRIu64 "", sc->gl_GlobalInvocationID_z, sc->outputStride[2]);
			}
		}
		char shiftCoordinate[500] = "";
		uint64_t maxCoordinate = sc->numCoordinates * sc->matrixConvolution;
		if (sc->numCoordinates * sc->matrixConvolution > 1) {
			sprintf(shiftCoordinate, " + ((%s / %" PRIu64 ") %% %" PRIu64 ") * %" PRIu64 "", sc->gl_GlobalInvocationID_z, sc->dispatchZactualFFTSize, maxCoordinate, sc->outputStride[3]);
		}
		if ((sc->matrixConvolution > 1) && (sc->convolutionStep)) {
			maxCoordinate = 1;
			sprintf(shiftCoordinate, " + %s * %" PRIu64 "", coordinate, sc->outputStride[3]);
		}
		char shiftBatch[500] = "";
		if ((sc->numBatches > 1) || (sc->numKernels > 1)) {
			if (sc->convolutionStep && (sc->numKernels > 1)) {
				sprintf(shiftBatch, " + %s * %" PRIu64 "", batchID, sc->outputStride[4]);
			}
			else
				sprintf(shiftBatch, " + (%s / %" PRIu64 ") * %" PRIu64 "", sc->gl_GlobalInvocationID_z, sc->dispatchZactualFFTSize * maxCoordinate, sc->outputStride[4]);
		}
		sc->tempLen = sprintf(sc->tempStr, "%s%s%s%s%s%s", outputOffset, shiftX, shiftY, shiftZ, shiftCoordinate, shiftBatch);
		res = PfAppendLine(sc);
		if (res != VKFFT_SUCCESS) return res;
		break;
	}
	case 1: case 111: case 121: case 131: case 141: case 143: case 145: {//grouped_c2c
		char outputOffset[30] = "";
		if (sc->outputOffset > 0) {
			sprintf(outputOffset, "%" PRIu64 " + ", sc->outputOffset / sc->outputNumberByteSize);
		}
		else {
			if (sc->performPostCompilationOutputOffset) {
				if (sc->outputType < 1000)
					sprintf(outputOffset, "consts.outputOffset + ");
				else
					sprintf(outputOffset, "consts.kernelOffset + ");
			}
		}
		char shiftX[500] = "";
		if (sc->numAxisUploads == 1)
			sprintf(shiftX, "(%s)", index_x);
		else
			sprintf(shiftX, "(%s) * %" PRIu64 "", index_x, sc->outputStride[0]);
		char shiftY[500] = "";
		if (index_y)
			sprintf(shiftY, " + (%s) * %" PRIu64 "", index_y, sc->outputStride[1]);
		char shiftZ[500] = "";
		if (sc->size[2] > 1) {
			if (sc->numCoordinates * sc->matrixConvolution * sc->numBatches > 1) {
				if (sc->performWorkGroupShift[2])
					sprintf(shiftZ, " + ((%s + consts.workGroupShiftZ * %s) %% %" PRIu64 ") * %" PRIu64 "", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z, sc->dispatchZactualFFTSize, sc->outputStride[2]);
				else
					sprintf(shiftZ, " + (%s %% %" PRIu64 ") * %" PRIu64 "", sc->gl_GlobalInvocationID_z, sc->dispatchZactualFFTSize, sc->outputStride[2]);
			}
			else {
				if (sc->performWorkGroupShift[2])
					sprintf(shiftZ, " + (%s + consts.workGroupShiftZ * %s) * %" PRIu64 "", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z, sc->outputStride[2]);
				else
					sprintf(shiftZ, " + %s * %" PRIu64 "", sc->gl_GlobalInvocationID_z, sc->outputStride[2]);
			}
		}
		char shiftCoordinate[500] = "";
		uint64_t maxCoordinate = sc->numCoordinates * sc->matrixConvolution;
		if (sc->numCoordinates * sc->matrixConvolution > 1) {
			sprintf(shiftCoordinate, " + ((%s / %" PRIu64 ") %% %" PRIu64 ") * %" PRIu64 "", sc->gl_GlobalInvocationID_z, sc->dispatchZactualFFTSize, maxCoordinate, sc->outputStride[3]);
		}
		if ((sc->matrixConvolution > 1) && (sc->convolutionStep)) {
			maxCoordinate = 1;
			sprintf(shiftCoordinate, " + %s * %" PRIu64 "", coordinate, sc->outputStride[3]);
		}
		char shiftBatch[500] = "";
		if ((sc->numBatches > 1) || (sc->numKernels > 1)) {
			if (sc->convolutionStep && (sc->numKernels > 1)) {
				sprintf(shiftBatch, " + %s * %" PRIu64 "", batchID, sc->outputStride[4]);
			}
			else
				sprintf(shiftBatch, " + (%s / %" PRIu64 ") * %" PRIu64 "", sc->gl_GlobalInvocationID_z, sc->dispatchZactualFFTSize * maxCoordinate, sc->outputStride[4]);
		}
		sc->tempLen = sprintf(sc->tempStr, "%s%s%s%s%s%s", outputOffset, shiftX, shiftY, shiftZ, shiftCoordinate, shiftBatch);
		res = PfAppendLine(sc);
		if (res != VKFFT_SUCCESS) return res;
		break;

	}
	}
	return res;
}
*/
#endif