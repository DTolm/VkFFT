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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
#ifndef VKFFT_SHADERGEN_FFT_H
#define VKFFT_SHADERGEN_FFT_H

#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_StringManagement/vkFFT_StringManager.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryInitialization/vkFFT_InputOutput.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryInitialization/vkFFT_InputOutputLayout.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryInitialization/vkFFT_SharedMemory.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryInitialization/vkFFT_Registers.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryInitialization/vkFFT_PushConstants.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryInitialization/vkFFT_Constants.h"

#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel1/PrePostProcessing/vkFFT_4step.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel1/PrePostProcessing/vkFFT_Bluestein.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel1/PrePostProcessing/vkFFT_Convolution.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel1/PrePostProcessing/vkFFT_R2C.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel1/PrePostProcessing/vkFFT_R2R.h"

#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel1/vkFFT_RegisterBoost.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel1/vkFFT_ReadWrite.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel1/vkFFT_RadixStage.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel1/vkFFT_RadixShuffle.h"

#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_KernelUtils.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_KernelStartEnd.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_MathUtils/vkFFT_MathUtils.h"
static inline VkFFTResult shaderGen_FFT(VkFFTSpecializationConstantsLayout* sc, int type) {
	
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_int1 = VKFFT_ZERO_INIT;
	temp_int1.type = 31;
	appendVersion(sc);
	appendExtensions(sc);
	appendLayoutVkFFT(sc);

	appendConstantsVkFFT(sc);
	//additional functions
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
	if (sc->convolutionStep) {
		appendKernelLayoutVkFFT(sc, id);
		id++;
	}
	if (sc->LUT) {
		appendLUTLayoutVkFFT(sc, id);
		id++;
	}
	if (sc->raderUintLUT) {
		appendRaderUintLUTLayoutVkFFT(sc, id);
		id++;
	}
	if (sc->useBluesteinFFT) {
		appendBluesteinLayoutVkFFT(sc, id);
		if (sc->BluesteinConvolutionStep)
			id++;
		if (sc->BluesteinPreMultiplication || sc->BluesteinPostMultiplication)
			id++;
	}
	int locType = (((type == 0) || (type == 5) || (type == 6) || (type == 110) || (type == 120) || (type == 130) || (type == 140) || (type == 142) || (type == 144)) && (sc->axisSwapped)) ? 1 : type;
	
	appendKernelStart(sc, type);
	
	setReadToRegisters(sc, type);
	setWriteFromRegisters(sc, type);

	appendRegisterInitialization(sc, type);
	
	PfContainer stageSize;
	stageSize.type = 31;
	PfContainer stageSizeSum;
	stageSizeSum.type = 31;
	PfContainer stageAngle;
	stageAngle.type = 32;
	
	int64_t max_coordinate = 0;
	if ((sc->convolutionStep) && (sc->matrixConvolution > 1)) {
		max_coordinate = sc->matrixConvolution - 1;
	}
	for (sc->coordinate.data.i = max_coordinate; sc->coordinate.data.i >= 0; sc->coordinate.data.i--) {

		appendReadDataVkFFT(sc, type);

		//pre-processing
		//r2c, r2r
		if (type == 6) {
			appendC2R_read(sc, type, 0);
		}
		if ((type == 110) || (type == 111)) {
			appendDCTI_read(sc, type, 0);
		}
		if ((type == 120) || (type == 121)) {
			appendDCTII_read_III_write(sc, type, 0);
		}
		if ((type == 130) || (type == 131)) {
			appendDCTII_write_III_read(sc, type, 0);
		}
		if ((type == 142) || (type == 143)) {
			appendDCTIV_even_read(sc, type, 0);
		}
		if ((type == 144) || (type == 145)) {
			appendDCTIV_odd_read(sc, type, 0);
		}
		if (sc->useBluesteinFFT && sc->BluesteinPreMultiplication) {
			appendBluesteinMultiplication(sc, locType, 0);
		}

		appendReorder4Step(sc, locType, 0);

		if (!sc->useRader) {
			appendBoostThreadDataReorder(sc, locType, 1);
		}

		//main FFT loop
		stageSize.data.i = 1;
		stageSizeSum.data.i = 0;
		stageAngle.data.d = (sc->inverse) ? sc->double_PI : -sc->double_PI;

		for (int i = 0; i < sc->numStages; i++) {
			if ((i == sc->numStages - 1) && (sc->registerBoost > 1)) {
				temp_int.data.i = sc->stageRadix[i];
				appendRadixStage(sc, &stageSize, &stageSizeSum, &stageAngle, &temp_int, i, locType);
			}
			else {
				temp_int.data.i = sc->stageRadix[i];
				appendRadixStage(sc, &stageSize, &stageSizeSum, &stageAngle, &temp_int, i, locType);
				if (i > 0) {
					switch (sc->stageRadix[i]) {
					case 2:
						stageSizeSum.data.i += stageSize.data.i;
						break;
					case 3:
						stageSizeSum.data.i += stageSize.data.i * 2;
						break;
					case 4:
						stageSizeSum.data.i += stageSize.data.i * 2;
						break;
					case 5:
						stageSizeSum.data.i += stageSize.data.i * 4;
						break;
					case 6:
						stageSizeSum.data.i += stageSize.data.i * 5;
						break;
					case 7:
						stageSizeSum.data.i += stageSize.data.i * 6;
						break;
					case 8:
						stageSizeSum.data.i += stageSize.data.i * 3;
						break;
					case 9:
						stageSizeSum.data.i += stageSize.data.i * 8;
						break;
					case 10:
						stageSizeSum.data.i += stageSize.data.i * 9;
						break;
					case 11:
						stageSizeSum.data.i += stageSize.data.i * 10;
						break;
					case 12:
						stageSizeSum.data.i += stageSize.data.i * 11;
						break;
					case 13:
						stageSizeSum.data.i += stageSize.data.i * 12;
						break;
					case 14:
						stageSizeSum.data.i += stageSize.data.i * 13;
						break;
					case 15:
						stageSizeSum.data.i += stageSize.data.i * 14;
						break;
					case 16:
						stageSizeSum.data.i += stageSize.data.i * 4;
						break;
					case 32:
						stageSizeSum.data.i += stageSize.data.i * 5;
						break;
					default:
						stageSizeSum.data.i += stageSize.data.i * (sc->stageRadix[i]);
						break;
					}
				}

				if ((i == sc->numStages - 1) || (sc->registerBoost == 1)) {
					temp_int.data.i = sc->stageRadix[i];
					appendRadixShuffle(sc, &stageSize, &stageSizeSum, &stageAngle, &temp_int, &temp_int, i, locType);
				}
				else {
					temp_int.data.i = sc->stageRadix[i];
					temp_int1.data.i = sc->stageRadix[i + 1];
					appendRadixShuffle(sc, &stageSize, &stageSizeSum, &stageAngle, &temp_int, &temp_int1, i, locType);
				}

				stageSize.data.i *= sc->stageRadix[i];
				stageAngle.data.d /= sc->stageRadix[i];
			}
		}

		if ((sc->convolutionStep) || (sc->useBluesteinFFT && sc->BluesteinConvolutionStep)) {
			appendRegisterStorage(sc, locType, 0);
		}
	}
	if ((sc->convolutionStep) || (sc->useBluesteinFFT && sc->BluesteinConvolutionStep)) {
		if (sc->numKernels.data.i > 1) {
			appendPreparationBatchedKernelConvolution(sc, locType);
		}
	}
	for (sc->batchID.data.i = 0; sc->batchID.data.i < sc->numKernels.data.i; sc->batchID.data.i++) {
		if ((sc->convolutionStep) || (sc->useBluesteinFFT && sc->BluesteinConvolutionStep)) {
			sc->coordinate.data.i = 0;
			if (sc->useBluesteinFFT && sc->BluesteinConvolutionStep)
			{
				appendBluesteinConvolution(sc, locType);
			}
			else {
				appendKernelConvolution(sc, locType);
			}
			appendBarrierVkFFT(sc);

		}
		for (sc->coordinate.data.i = 0; sc->coordinate.data.i < max_coordinate+1; sc->coordinate.data.i++) {

			if ((sc->convolutionStep) || (sc->useBluesteinFFT && sc->BluesteinConvolutionStep)) {
				appendRegisterStorage(sc, locType, 1);

				stageSize.data.i = 1;
				stageSizeSum.data.i = 0;
				stageAngle.data.d = sc->double_PI;
				sc->inverse = 1;
				for (uint64_t i = 0; i < (uint64_t)sc->numStages; i++) {
					temp_int.data.i = sc->stageRadix[i];
					appendRadixStage(sc, &stageSize, &stageSizeSum, &stageAngle, &temp_int, (int)i, locType);
					if (i > 0) {
						switch (sc->stageRadix[i]) {
						case 2:
							stageSizeSum.data.i += stageSize.data.i;
							break;
						case 3:
							stageSizeSum.data.i += stageSize.data.i * 2;
							break;
						case 4:
							stageSizeSum.data.i += stageSize.data.i * 2;
							break;
						case 5:
							stageSizeSum.data.i += stageSize.data.i * 4;
							break;
						case 6:
							stageSizeSum.data.i += stageSize.data.i * 5;
							break;
						case 7:
							stageSizeSum.data.i += stageSize.data.i * 6;
							break;
						case 8:
							stageSizeSum.data.i += stageSize.data.i * 3;
							break;
						case 9:
							stageSizeSum.data.i += stageSize.data.i * 8;
							break;
						case 10:
							stageSizeSum.data.i += stageSize.data.i * 9;
							break;
						case 11:
							stageSizeSum.data.i += stageSize.data.i * 10;
							break;
						case 12:
							stageSizeSum.data.i += stageSize.data.i * 11;
							break;
						case 13:
							stageSizeSum.data.i += stageSize.data.i * 12;
							break;
						case 14:
							stageSizeSum.data.i += stageSize.data.i * 13;
							break;
						case 15:
							stageSizeSum.data.i += stageSize.data.i * 14;
							break;
						case 16:
							stageSizeSum.data.i += stageSize.data.i * 4;
							break;
						case 32:
							stageSizeSum.data.i += stageSize.data.i * 5;
							break;
						default:
							stageSizeSum.data.i += stageSize.data.i * (sc->stageRadix[i]);
							break;
						}
					}
					if (i == sc->numStages - 1) {
						temp_int.data.i = sc->stageRadix[i];
						temp_int1.data.i = sc->stageRadix[i];

						appendRadixShuffle(sc, &stageSize, &stageSizeSum, &stageAngle, &temp_int, &temp_int1, i, locType);
					}
					else {
						temp_int.data.i = sc->stageRadix[i];
						temp_int1.data.i = sc->stageRadix[i + 1];

						appendRadixShuffle(sc, &stageSize, &stageSizeSum, &stageAngle, &temp_int, &temp_int1, i, locType);
					}
					stageSize.data.i *= sc->stageRadix[i];
					stageAngle.data.d /= sc->stageRadix[i];
				}
			}

			if (!sc->useRader) {
				appendBoostThreadDataReorder(sc, locType, 0);
			}

			//post-processing
			//r2c, r2r
			appendReorder4Step(sc, locType, 1);
			if (sc->useBluesteinFFT && sc->BluesteinPostMultiplication) {
				appendBluesteinMultiplication(sc, locType, 1);
			}
			if ((type == 5) && (sc->mergeSequencesR2C)) {
				appendR2C_write(sc, type, 1);
			}
			if ((type == 120) || (type == 121)) {
				appendDCTII_write_III_read(sc, type, 1);
			}
			if ((type == 130) || (type == 131)) {
				appendDCTII_read_III_write(sc, type, 1);
			}
			if ((type == 142) || (type == 143)) {
				appendDCTIV_even_write(sc, type, 1);
			}
			if ((type == 144) || (type == 145)) {
				appendDCTIV_odd_write(sc, type, 1);
			}
			appendWriteDataVkFFT(sc, type);
		}
	}
	appendKernelEnd(sc);

	freeRegisterInitialization(sc, type);
	
	return sc->res;
}

#endif
