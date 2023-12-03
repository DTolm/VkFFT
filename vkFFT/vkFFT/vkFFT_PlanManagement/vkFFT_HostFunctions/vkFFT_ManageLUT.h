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
#ifndef VKFFT_MANAGELUT_H
#define VKFFT_MANAGELUT_H
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT/vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_ManageMemory.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_MathUtils/vkFFT_MathUtils.h"

static inline VkFFTResult VkFFT_AllocateLUT(VkFFTApplication* app, VkFFTPlan* FFTPlan, VkFFTAxis* axis, pfUINT inverse){
	VkFFTResult resFFT = VKFFT_SUCCESS;
#if(VKFFT_BACKEND==0)
	VkResult res = VK_SUCCESS;
#elif(VKFFT_BACKEND==1)
	cudaError_t res = cudaSuccess;
#elif(VKFFT_BACKEND==2)
	hipError_t res = hipSuccess;
#elif(VKFFT_BACKEND==3)
	cl_int res = CL_SUCCESS;
#elif(VKFFT_BACKEND==4)
	ze_result_t res = ZE_RESULT_SUCCESS;
#elif(VKFFT_BACKEND==5)
#endif
	//allocate LUT
	if (app->configuration.useLUT == 1) {
		pfUINT dimMult = 1;
		pfUINT maxStageSum = 0;
		for (pfUINT i = 0; i < axis->specializationConstants.numStages; i++) {
			if (i > 0) {
				switch (axis->specializationConstants.stageRadix[i]) {
				case 2:
					maxStageSum += dimMult;
					break;
				case 3:
					maxStageSum += dimMult * 2;
					break;
				case 4:
					maxStageSum += dimMult * 2;
					break;
				case 5:
					maxStageSum += dimMult * 4;
					break;
				case 6:
					maxStageSum += dimMult * 5;
					break;
				case 7:
						maxStageSum += dimMult * 6;
					break;
				case 8:
					maxStageSum += dimMult * 3;
					break;
				case 9:
					maxStageSum += dimMult * 8;
					break;
				case 10:
					maxStageSum += dimMult * 9;
					break;
				case 11:
					if (app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory)
						maxStageSum += dimMult * 11;
					else 
						maxStageSum += dimMult * 10;
					break;
				case 12:
					maxStageSum += dimMult * 11;
					break;
				case 13:
					if (app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory)
						maxStageSum += dimMult * 13;
					else 
						maxStageSum += dimMult * 12;
					break;
				case 14:
					maxStageSum += dimMult * 13;
					break;
				case 15:
					maxStageSum += dimMult * 14;
					break;
				case 16:
					maxStageSum += dimMult * 4;
					break;
				case 32:
					maxStageSum += dimMult * 5;
					break;
				default:
					maxStageSum += dimMult * (axis->specializationConstants.stageRadix[i]);
					break;
				}
			}
			dimMult *= axis->specializationConstants.stageRadix[i];
		}
		axis->specializationConstants.maxStageSumLUT = (int)maxStageSum;

		dimMult = 1;
		for (pfUINT k = 0; k < axis->specializationConstants.numRaderPrimes; k++) {
			if (axis->specializationConstants.raderContainer[k].type == 0) {
				axis->specializationConstants.raderContainer[k].RaderRadixOffsetLUT = maxStageSum;
				for (pfUINT i = 0; i < axis->specializationConstants.raderContainer[k].numStages; i++) {
					if (i > 0) {
						switch (axis->specializationConstants.raderContainer[k].stageRadix[i]) {
						case 2:
							maxStageSum += dimMult;
							break;
						case 3:
							maxStageSum += dimMult * 2;
							break;
						case 4:
							maxStageSum += dimMult * 2;
							break;
						case 5:
							maxStageSum += dimMult * 4;
							break;
						case 6:
							maxStageSum += dimMult * 5;
							break;
						case 7:
								maxStageSum += dimMult * 6;
							break;
						case 8:
							maxStageSum += dimMult * 3;
							break;
						case 9:
							maxStageSum += dimMult * 8;
							break;
						case 10:
							maxStageSum += dimMult * 9;
							break;
						case 11:
							if (app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory)
								maxStageSum += dimMult * 11;
							else 
								maxStageSum += dimMult * 10;
							break;
						case 12:
							maxStageSum += dimMult * 11;
							break;
						case 13:
							if (app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory)
								maxStageSum += dimMult * 13;
							else 
								maxStageSum += dimMult * 12;
							break;
						case 14:
							maxStageSum += dimMult * 13;
							break;
						case 15:
							maxStageSum += dimMult * 14;
							break;
						case 16:
							maxStageSum += dimMult * 4;
							break;
						case 32:
							maxStageSum += dimMult * 5;
							break;
						default:
							maxStageSum += dimMult * (axis->specializationConstants.raderContainer[k].stageRadix[i]);
							break;
						}
					}
					dimMult *= axis->specializationConstants.raderContainer[k].stageRadix[i];
				}
				axis->specializationConstants.maxStageSumLUT = (int)maxStageSum;
				dimMult = 1;
			}
		}
		//iFFT LUT
		dimMult = 1;
		for (pfUINT k = 0; k < axis->specializationConstants.numRaderPrimes; k++) {
			if (axis->specializationConstants.raderContainer[k].type == 0) {
				axis->specializationConstants.raderContainer[k].RaderRadixOffsetLUTiFFT = maxStageSum;
				for (pfINT i = axis->specializationConstants.raderContainer[k].numStages - 1; i >= 0; i--) {
					if (i < (pfINT)axis->specializationConstants.raderContainer[k].numStages - 1) {
						switch (axis->specializationConstants.raderContainer[k].stageRadix[i]) {
						case 2:
							maxStageSum += dimMult;
							break;
						case 3:
							maxStageSum += dimMult * 2;
							break;
						case 4:
							maxStageSum += dimMult * 2;
							break;
						case 5:
							maxStageSum += dimMult * 4;
							break;
						case 6:
							maxStageSum += dimMult * 5;
							break;
						case 7:
								maxStageSum += dimMult * 6;
							break;
						case 8:
							maxStageSum += dimMult * 3;
							break;
						case 9:
							maxStageSum += dimMult * 8;
							break;
						case 10:
							maxStageSum += dimMult * 9;
							break;
						case 11:
							if (app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory)
								maxStageSum += dimMult * 11;
							else 
								maxStageSum += dimMult * 10;
							break;
						case 12:
							maxStageSum += dimMult * 11;
							break;
						case 13:
							if (app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory)
								maxStageSum += dimMult * 13;
							else 
								maxStageSum += dimMult * 12;
							break;
						case 14:
							maxStageSum += dimMult * 13;
							break;
						case 15:
							maxStageSum += dimMult * 14;
							break;
						case 16:
							maxStageSum += dimMult * 4;
							break;
						case 32:
							maxStageSum += dimMult * 5;
							break;
						default:
							maxStageSum += dimMult * (axis->specializationConstants.raderContainer[k].stageRadix[i]);
							break;
						}
					}
					dimMult *= axis->specializationConstants.raderContainer[k].stageRadix[i];
				}
				axis->specializationConstants.maxStageSumLUT = (int)maxStageSum;
				dimMult = 1;
			}
		}

		pfUINT currentLUTPos = maxStageSum;
		if ((app->configuration.useLUT_4step == 1) && (axis->specializationConstants.axis_upload_id > 0)) currentLUTPos += axis->specializationConstants.stageStartSize.data.i * axis->specializationConstants.fftDim.data.i;
		pfUINT disableReferenceLUT_DCT = 0;
		if ((((((axis->specializationConstants.performDCT == 3) || (axis->specializationConstants.performDST == 3)) && (axis->specializationConstants.actualInverse == 0)) || (((axis->specializationConstants.performDCT == 2) || (axis->specializationConstants.performDST == 2)) && (axis->specializationConstants.actualInverse == 1)))) || (((((axis->specializationConstants.performDCT == 2) || (axis->specializationConstants.performDST == 2)) && (axis->specializationConstants.actualInverse == 0)) || (((axis->specializationConstants.performDCT == 3) || (axis->specializationConstants.performDST == 3)) && (axis->specializationConstants.actualInverse == 1))))) {
			disableReferenceLUT_DCT = 1;
		}
		if ((((((axis->specializationConstants.performDCT == 3) || (axis->specializationConstants.performDST == 3)) && (axis->specializationConstants.actualInverse == 0)) || (((axis->specializationConstants.performDCT == 2) || (axis->specializationConstants.performDST == 2)) && (axis->specializationConstants.actualInverse == 1))) && ((axis->specializationConstants.axis_upload_id == (axis->specializationConstants.numAxisUploads-1)) && (!((axis->specializationConstants.useBluesteinFFT && (axis->specializationConstants.reverseBluesteinMultiUpload == 1)))))) || (((((axis->specializationConstants.performDCT == 2) || (axis->specializationConstants.performDST == 2)) && (axis->specializationConstants.actualInverse == 0)) || (((axis->specializationConstants.performDCT == 3) || (axis->specializationConstants.performDST == 3)) && (axis->specializationConstants.actualInverse == 1))) && (((axis->specializationConstants.axis_upload_id == 0) && (!((axis->specializationConstants.useBluesteinFFT && (axis->specializationConstants.reverseBluesteinMultiUpload == 0) && (axis->specializationConstants.numAxisUploads > 1))))) || ((axis->specializationConstants.axis_upload_id == (axis->specializationConstants.numAxisUploads-1)) && (axis->specializationConstants.useBluesteinFFT && (axis->specializationConstants.reverseBluesteinMultiUpload == 1)))))) {
			axis->specializationConstants.startDCT3LUT.type = 31;
			axis->specializationConstants.startDCT3LUT.data.i = currentLUTPos;
			currentLUTPos += (app->configuration.size[axis->specializationConstants.axis_id] / 2 + 2);
			disableReferenceLUT_DCT = 1;
		}
		if (((axis->specializationConstants.performDCT == 4) || (axis->specializationConstants.performDST == 4)) && (app->configuration.size[axis->specializationConstants.axis_id] % 2 == 0) && ((axis->specializationConstants.axis_upload_id == (axis->specializationConstants.numAxisUploads-1)) && (!((axis->specializationConstants.useBluesteinFFT && (axis->specializationConstants.reverseBluesteinMultiUpload == 1)))))) {
			axis->specializationConstants.startDCT3LUT.type = 31;
			axis->specializationConstants.startDCT3LUT.data.i = currentLUTPos;
			currentLUTPos += (app->configuration.size[axis->specializationConstants.axis_id] / 4 + 2);
			disableReferenceLUT_DCT = 1;
		}
		if (((axis->specializationConstants.performDCT == 4) || (axis->specializationConstants.performDST == 4)) && (app->configuration.size[axis->specializationConstants.axis_id] % 2 == 0) && (((axis->specializationConstants.axis_upload_id == 0) && (!((axis->specializationConstants.useBluesteinFFT && (axis->specializationConstants.reverseBluesteinMultiUpload == 0) && (axis->specializationConstants.numAxisUploads > 1))))) || ((axis->specializationConstants.axis_upload_id == (axis->specializationConstants.numAxisUploads-1)) && (axis->specializationConstants.useBluesteinFFT && (axis->specializationConstants.reverseBluesteinMultiUpload == 1))))) {
			axis->specializationConstants.startDCT4LUT.type = 31;
			axis->specializationConstants.startDCT4LUT.data.i = currentLUTPos;
			currentLUTPos += app->configuration.size[axis->specializationConstants.axis_id] / 2;
			disableReferenceLUT_DCT = 1;
		}
		if (axis->specializationConstants.useRader) {
			for (pfUINT i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
				if (!axis->specializationConstants.inline_rader_kernel) {
					axis->specializationConstants.raderContainer[i].RaderKernelOffsetLUT = currentLUTPos;
					currentLUTPos += (axis->specializationConstants.raderContainer[i].prime - 1);
				}
			}
		}

		if (app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) {
			pfLD double_PI = pfFPinit("3.14159265358979323846264338327950288419716939937510");
			axis->bufferLUTSize = currentLUTPos * 4 * sizeof(double);			
			if (axis->bufferLUTSize == 0) axis->bufferLUTSize = 2 * sizeof(double);
			double* tempLUT = (double*)malloc(axis->bufferLUTSize);
			if (!tempLUT) {
				deleteVkFFT(app);
				return VKFFT_ERROR_MALLOC_FAILED;
			}
			pfUINT localStageSize = axis->specializationConstants.stageRadix[0];
			pfUINT localStageSum = 0;

			PfContainer in = VKFFT_ZERO_INIT;
			PfContainer temp1 = VKFFT_ZERO_INIT;
			in.type = 22;

			for (pfUINT i = 1; i < axis->specializationConstants.numStages; i++) {
				if ((axis->specializationConstants.stageRadix[i] & (axis->specializationConstants.stageRadix[i] - 1)) == 0) {
					for (pfUINT k = 0; k < log2(axis->specializationConstants.stageRadix[i]); k++) {
						for (pfUINT j = 0; j < localStageSize; j++) {
							in.data.d = pfcos(j * double_PI / localStageSize / pow(2, k));
							PfConvToDoubleDouble(&axis->specializationConstants, &temp1, &in);
							tempLUT[4 * (j + localStageSum)] = (double)temp1.data.dd[0].data.d;
							tempLUT[4 * (j + localStageSum) + 1] = (double)temp1.data.dd[1].data.d;

							in.data.d = pfsin(j * double_PI / localStageSize / pow(2, k));
							PfConvToDoubleDouble(&axis->specializationConstants, &temp1, &in);
							tempLUT[4 * (j + localStageSum) + 2] = (double)temp1.data.dd[0].data.d;
							tempLUT[4 * (j + localStageSum) + 3] = (double)temp1.data.dd[1].data.d;
						}
						localStageSum += localStageSize;
					}
				}
				else if (axis->specializationConstants.rader_generator[i] > 0) {
					for (pfUINT j = 0; j < localStageSize; j++) {
						for (pfINT k = (axis->specializationConstants.stageRadix[i] - 1); k >= 0; k--) {
							in.data.d = pfcos(j * pfFPinit("2.0") * k / axis->specializationConstants.stageRadix[i] * double_PI / localStageSize);
							PfConvToDoubleDouble(&axis->specializationConstants, &temp1, &in);
							tempLUT[4* (k + localStageSum)] = (double)temp1.data.dd[0].data.d;
							tempLUT[4 * (k + localStageSum) + 1] = (double)temp1.data.dd[1].data.d;

							in.data.d = pfsin(j * pfFPinit("2.0") * k / axis->specializationConstants.stageRadix[i] * double_PI / localStageSize);
							PfConvToDoubleDouble(&axis->specializationConstants, &temp1, &in);
							tempLUT[4 * (k + localStageSum) + 2] = (double)temp1.data.dd[0].data.d;
							tempLUT[4 * (k + localStageSum) + 3] = (double)temp1.data.dd[1].data.d;
						}
						localStageSum += (axis->specializationConstants.stageRadix[i]);
					}
				}
				else {
					for (pfUINT k = (axis->specializationConstants.stageRadix[i] - 1); k > 0; k--) {
						for (pfUINT j = 0; j < localStageSize; j++) {
							in.data.d = pfcos(j * pfFPinit("2.0") * k / axis->specializationConstants.stageRadix[i] * double_PI / localStageSize);
							PfConvToDoubleDouble(&axis->specializationConstants, &temp1, &in);
							tempLUT[4 * (j + localStageSum)] = (double)temp1.data.dd[0].data.d;
							tempLUT[4 * (j + localStageSum) + 1] = (double)temp1.data.dd[1].data.d;
							in.data.d = pfsin(j * pfFPinit("2.0") * k / axis->specializationConstants.stageRadix[i] * double_PI / localStageSize);
							PfConvToDoubleDouble(&axis->specializationConstants, &temp1, &in);
							tempLUT[4 * (j + localStageSum) + 2] = (double)temp1.data.dd[0].data.d;
							tempLUT[4 * (j + localStageSum) + 3] = (double)temp1.data.dd[1].data.d;
						}
						localStageSum += localStageSize;
					}
				}
				localStageSize *= axis->specializationConstants.stageRadix[i];
			}


			if (axis->specializationConstants.useRader) {
				for (pfUINT i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
					if (axis->specializationConstants.raderContainer[i].type) {
						if (!axis->specializationConstants.inline_rader_kernel) {
							for (pfUINT j = 0; j < (axis->specializationConstants.raderContainer[i].prime - 1); j++) {//fix later
								pfUINT g_pow = 1;
								for (pfUINT t = 0; t < axis->specializationConstants.raderContainer[i].prime - 1 - j; t++) {
									g_pow = (g_pow * axis->specializationConstants.raderContainer[i].generator) % axis->specializationConstants.raderContainer[i].prime;
								}
								in.data.d = pfcos(pfFPinit("2.0") * g_pow * double_PI / axis->specializationConstants.raderContainer[i].prime);
								PfConvToDoubleDouble(&axis->specializationConstants, &temp1, &in);
								tempLUT[4 * (j + axis->specializationConstants.raderContainer[i].RaderKernelOffsetLUT)] = (double)temp1.data.dd[0].data.d;
								tempLUT[4 * (j + axis->specializationConstants.raderContainer[i].RaderKernelOffsetLUT) + 1] = (double)temp1.data.dd[1].data.d;
								in.data.d = (-pfsin(pfFPinit("2.0") * g_pow * double_PI / axis->specializationConstants.raderContainer[i].prime));
								PfConvToDoubleDouble(&axis->specializationConstants, &temp1, &in);
								tempLUT[4 * (j + axis->specializationConstants.raderContainer[i].RaderKernelOffsetLUT) + 2] = (double)temp1.data.dd[0].data.d;
								tempLUT[4 * (j + axis->specializationConstants.raderContainer[i].RaderKernelOffsetLUT) + 3] = (double)temp1.data.dd[1].data.d;
							}
						}
					}
					else {
						localStageSize = axis->specializationConstants.raderContainer[i].stageRadix[0];
						localStageSum = 0;
						for (pfUINT l = 1; l < axis->specializationConstants.raderContainer[i].numStages; l++) {
							if ((axis->specializationConstants.raderContainer[i].stageRadix[l] & (axis->specializationConstants.raderContainer[i].stageRadix[l] - 1)) == 0) {
								for (pfUINT k = 0; k < log2(axis->specializationConstants.raderContainer[i].stageRadix[l]); k++) {
									for (pfUINT j = 0; j < localStageSize; j++) {
										in.data.d = pfcos(j * double_PI / localStageSize / pow(2, k));
										PfConvToDoubleDouble(&axis->specializationConstants, &temp1, &in);
										tempLUT[4 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUT)] = (double)temp1.data.dd[0].data.d;
										tempLUT[4 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUT) + 1] = (double)temp1.data.dd[1].data.d;
										in.data.d = pfsin(j * double_PI / localStageSize / pow(2, k));
										PfConvToDoubleDouble(&axis->specializationConstants, &temp1, &in);
										tempLUT[4 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUT) + 2] = (double)temp1.data.dd[0].data.d;
										tempLUT[4 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUT) + 3] = (double)temp1.data.dd[1].data.d;
									}
									localStageSum += localStageSize;
								}
							}
							else {
								for (pfUINT k = (axis->specializationConstants.raderContainer[i].stageRadix[l] - 1); k > 0; k--) {
									for (pfUINT j = 0; j < localStageSize; j++) {
										in.data.d = pfcos(j * pfFPinit("2.0") * k / axis->specializationConstants.raderContainer[i].stageRadix[l] * double_PI / localStageSize);
										PfConvToDoubleDouble(&axis->specializationConstants, &temp1, &in);
										tempLUT[4 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUT)] = (double)temp1.data.dd[0].data.d;
										tempLUT[4 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUT) + 1] = (double)temp1.data.dd[1].data.d;
										in.data.d = pfsin(j * pfFPinit("2.0") * k / axis->specializationConstants.raderContainer[i].stageRadix[l] * double_PI / localStageSize);
										PfConvToDoubleDouble(&axis->specializationConstants, &temp1, &in);
										tempLUT[4 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUT) + 2] = (double)temp1.data.dd[0].data.d;
										tempLUT[4 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUT) + 3] = (double)temp1.data.dd[1].data.d;
									}
									localStageSum += localStageSize;
								}
							}
							localStageSize *= axis->specializationConstants.raderContainer[i].stageRadix[l];
						}

						localStageSize = axis->specializationConstants.raderContainer[i].stageRadix[axis->specializationConstants.raderContainer[i].numStages - 1];
						localStageSum = 0;
						for (pfINT l = (pfINT)axis->specializationConstants.raderContainer[i].numStages - 2; l >= 0; l--) {
							if ((axis->specializationConstants.raderContainer[i].stageRadix[l] & (axis->specializationConstants.raderContainer[i].stageRadix[l] - 1)) == 0) {
								for (pfUINT k = 0; k < log2(axis->specializationConstants.raderContainer[i].stageRadix[l]); k++) {
									for (pfUINT j = 0; j < localStageSize; j++) {
										in.data.d = pfcos(j * double_PI / localStageSize / pow(2, k));
										PfConvToDoubleDouble(&axis->specializationConstants, &temp1, &in);
										tempLUT[4 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUTiFFT)] = (double)temp1.data.dd[0].data.d;
										tempLUT[4 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUTiFFT) + 1] = (double)temp1.data.dd[1].data.d;
										in.data.d = pfsin(j * double_PI / localStageSize / pow(2, k));
										PfConvToDoubleDouble(&axis->specializationConstants, &temp1, &in);
										tempLUT[4 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUTiFFT) + 2] = (double)temp1.data.dd[0].data.d;
										tempLUT[4 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUTiFFT) + 3] = (double)temp1.data.dd[1].data.d;
									}
									localStageSum += localStageSize;
								}
							}
							else {
								for (pfUINT k = (axis->specializationConstants.raderContainer[i].stageRadix[l] - 1); k > 0; k--) {
									for (pfUINT j = 0; j < localStageSize; j++) {
										in.data.d = pfcos(j * pfFPinit("2.0") * k / axis->specializationConstants.raderContainer[i].stageRadix[l] * double_PI / localStageSize);
										PfConvToDoubleDouble(&axis->specializationConstants, &temp1, &in);
										tempLUT[4 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUTiFFT)] = (double)temp1.data.dd[0].data.d;
										tempLUT[4 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUTiFFT) + 1] = (double)temp1.data.dd[1].data.d;
										in.data.d = pfsin(j * pfFPinit("2.0") * k / axis->specializationConstants.raderContainer[i].stageRadix[l] * double_PI / localStageSize);
										PfConvToDoubleDouble(&axis->specializationConstants, &temp1, &in);
										tempLUT[4 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUTiFFT) + 2] = (double)temp1.data.dd[0].data.d;
										tempLUT[4 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUTiFFT) + 3] = (double)temp1.data.dd[1].data.d;
									}
									localStageSum += localStageSize;
								}
							}
							localStageSize *= axis->specializationConstants.raderContainer[i].stageRadix[l];
						}

						if (!axis->specializationConstants.inline_rader_kernel) {
							double* raderFFTkernel = (double*)axis->specializationConstants.raderContainer[i].raderFFTkernel;
							for (pfUINT j = 0; j < (axis->specializationConstants.raderContainer[i].prime - 1); j++) {//fix later
								in.data.d = (((pfLD)raderFFTkernel[4 * j] + (pfLD)raderFFTkernel[4 * j + 1])/ (pfLD)(axis->specializationConstants.raderContainer[i].prime - 1));
								PfConvToDoubleDouble(&axis->specializationConstants, &temp1, &in);
								tempLUT[4 * (j + axis->specializationConstants.raderContainer[i].RaderKernelOffsetLUT)] = (double)temp1.data.dd[0].data.d;
								tempLUT[4 * (j + axis->specializationConstants.raderContainer[i].RaderKernelOffsetLUT) + 1] = (double)temp1.data.dd[1].data.d;
								in.data.d = (((pfLD)raderFFTkernel[4 * j + 2] + (pfLD)raderFFTkernel[4 * j + 3])/ (pfLD)(axis->specializationConstants.raderContainer[i].prime - 1));
								PfConvToDoubleDouble(&axis->specializationConstants, &temp1, &in);
								tempLUT[4 * (j + axis->specializationConstants.raderContainer[i].RaderKernelOffsetLUT) + 2] = (double)temp1.data.dd[0].data.d;
								tempLUT[4 * (j + axis->specializationConstants.raderContainer[i].RaderKernelOffsetLUT) + 3] = (double)temp1.data.dd[1].data.d;
							}
						}
					}
				}
			}
			if ((axis->specializationConstants.axis_upload_id > 0) && (app->configuration.useLUT_4step == 1)) {
				for (pfUINT i = 0; i < (pfUINT)axis->specializationConstants.stageStartSize.data.i; i++) {
					for (pfUINT j = 0; j < (pfUINT)axis->specializationConstants.fftDim.data.i; j++) {
						pfLD angle = pfFPinit("2.0") * double_PI * ((i * j) / (pfLD)(axis->specializationConstants.stageStartSize.data.i * axis->specializationConstants.fftDim.data.i));
						in.data.d = pfcos(angle);
						PfConvToDoubleDouble(&axis->specializationConstants, &temp1, &in);
						tempLUT[maxStageSum * 4 + 4 * (i + j * axis->specializationConstants.stageStartSize.data.i)] = (double)temp1.data.dd[0].data.d;
						tempLUT[maxStageSum * 4 + 4 * (i + j * axis->specializationConstants.stageStartSize.data.i) + 1] = (double)temp1.data.dd[1].data.d;
						in.data.d = pfsin(angle);
						PfConvToDoubleDouble(&axis->specializationConstants, &temp1, &in);
						tempLUT[maxStageSum * 4 + 4 * (i + j * axis->specializationConstants.stageStartSize.data.i) + 2] = (double)temp1.data.dd[0].data.d;
						tempLUT[maxStageSum * 4 + 4 * (i + j * axis->specializationConstants.stageStartSize.data.i) + 3] = (double)temp1.data.dd[1].data.d;
					}
				}
			}
			if ((((((axis->specializationConstants.performDCT == 3) || (axis->specializationConstants.performDST == 3)) && (axis->specializationConstants.actualInverse == 0)) || (((axis->specializationConstants.performDCT == 2) || (axis->specializationConstants.performDST == 2)) && (axis->specializationConstants.actualInverse == 1))) && ((axis->specializationConstants.axis_upload_id == (axis->specializationConstants.numAxisUploads-1)) && (!((axis->specializationConstants.useBluesteinFFT && (axis->specializationConstants.reverseBluesteinMultiUpload == 1)))))) || (((((axis->specializationConstants.performDCT == 2) || (axis->specializationConstants.performDST == 2)) && (axis->specializationConstants.actualInverse == 0)) || (((axis->specializationConstants.performDCT == 3) || (axis->specializationConstants.performDST == 3)) && (axis->specializationConstants.actualInverse == 1))) && (((axis->specializationConstants.axis_upload_id == 0) && (!((axis->specializationConstants.useBluesteinFFT && (axis->specializationConstants.reverseBluesteinMultiUpload == 0) && (axis->specializationConstants.numAxisUploads > 1))))) || ((axis->specializationConstants.axis_upload_id == (axis->specializationConstants.numAxisUploads-1)) && (axis->specializationConstants.useBluesteinFFT && (axis->specializationConstants.reverseBluesteinMultiUpload == 1)))))) {
				for (pfUINT j = 0; j < app->configuration.size[axis->specializationConstants.axis_id] / 2 + 2; j++) {
					pfLD angle = (double_PI / pfFPinit("2.0") / (pfLD)(app->configuration.size[axis->specializationConstants.axis_id])) * j;
					in.data.d = pfcos(angle);
					PfConvToDoubleDouble(&axis->specializationConstants, &temp1, &in);
					tempLUT[4 * axis->specializationConstants.startDCT3LUT.data.i + 4 * j] = (double)temp1.data.dd[0].data.d;
					tempLUT[4 * axis->specializationConstants.startDCT3LUT.data.i + 4 * j + 1] = (double)temp1.data.dd[1].data.d;
					in.data.d = pfsin(angle);
					PfConvToDoubleDouble(&axis->specializationConstants, &temp1, &in);
					tempLUT[4 * axis->specializationConstants.startDCT3LUT.data.i + 4 * j + 2] = (double)temp1.data.dd[0].data.d;
					tempLUT[4 * axis->specializationConstants.startDCT3LUT.data.i + 4 * j + 3] = (double)temp1.data.dd[1].data.d;
				}
			}
			if (((axis->specializationConstants.performDCT == 4) || (axis->specializationConstants.performDST == 4)) && (app->configuration.size[axis->specializationConstants.axis_id] % 2 == 0) && ((axis->specializationConstants.axis_upload_id == (axis->specializationConstants.numAxisUploads - 1)) && (!((axis->specializationConstants.useBluesteinFFT && (axis->specializationConstants.reverseBluesteinMultiUpload == 1)))))) {
				for (pfUINT j = 0; j < app->configuration.size[axis->specializationConstants.axis_id] / 4 + 2; j++) {
					pfLD angle = (double_PI / pfFPinit("2.0") / (pfLD)(app->configuration.size[axis->specializationConstants.axis_id] / 2)) * j;
					in.data.d = pfcos(angle);
					PfConvToDoubleDouble(&axis->specializationConstants, &temp1, &in);
					tempLUT[4 * axis->specializationConstants.startDCT3LUT.data.i + 4 * j] = (double)temp1.data.dd[0].data.d;
					tempLUT[4 * axis->specializationConstants.startDCT3LUT.data.i + 4 * j + 1] = (double)temp1.data.dd[1].data.d;
					in.data.d = pfsin(angle);
					PfConvToDoubleDouble(&axis->specializationConstants, &temp1, &in);
					tempLUT[4 * axis->specializationConstants.startDCT3LUT.data.i + 4 * j + 2] = (double)temp1.data.dd[0].data.d;
					tempLUT[4 * axis->specializationConstants.startDCT3LUT.data.i + 4 * j + 3] = (double)temp1.data.dd[1].data.d;
				}
			}
			if (((axis->specializationConstants.performDCT == 4) || (axis->specializationConstants.performDST == 4)) && (app->configuration.size[axis->specializationConstants.axis_id] % 2 == 0) && (((axis->specializationConstants.axis_upload_id == 0) && (!((axis->specializationConstants.useBluesteinFFT && (axis->specializationConstants.reverseBluesteinMultiUpload == 0) && (axis->specializationConstants.numAxisUploads > 1))))) || ((axis->specializationConstants.axis_upload_id == (axis->specializationConstants.numAxisUploads-1)) && (axis->specializationConstants.useBluesteinFFT && (axis->specializationConstants.reverseBluesteinMultiUpload == 1))))) {
				for (pfUINT j = 0; j < app->configuration.size[axis->specializationConstants.axis_id] / 2; j++) {
					pfLD angle = (-double_PI / pfFPinit("8.0") / (pfLD)(app->configuration.size[axis->specializationConstants.axis_id] / 2)) * (2 * j + 1);
					in.data.d = pfcos(angle);
					PfConvToDoubleDouble(&axis->specializationConstants, &temp1, &in);
					tempLUT[4 * axis->specializationConstants.startDCT4LUT.data.i + 4 * j] = (double)temp1.data.dd[0].data.d;
					tempLUT[4 * axis->specializationConstants.startDCT4LUT.data.i + 4 * j + 1] = (double)temp1.data.dd[1].data.d;
					in.data.d = pfsin(angle);
					PfConvToDoubleDouble(&axis->specializationConstants, &temp1, &in);
					tempLUT[4 * axis->specializationConstants.startDCT4LUT.data.i + 4 * j + 2] = (double)temp1.data.dd[0].data.d;
					tempLUT[4 * axis->specializationConstants.startDCT4LUT.data.i + 4 * j + 3] = (double)temp1.data.dd[1].data.d;
				}
			}
			PfDeallocateContainer(&axis->specializationConstants, &temp1);
			axis->referenceLUT = 0;
			if ((axis->specializationConstants.reverseBluesteinMultiUpload == 1) && (!disableReferenceLUT_DCT)) {
				axis->bufferLUT = FFTPlan->axes[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id].bufferLUT;
#if(VKFFT_BACKEND==0)
				axis->bufferLUTDeviceMemory = FFTPlan->axes[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id].bufferLUTDeviceMemory;
#endif
				axis->bufferLUTSize = FFTPlan->axes[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id].bufferLUTSize;
				axis->referenceLUT = 1;
			}
			else {
				if ((!inverse) && (!app->configuration.makeForwardPlanOnly) && (!disableReferenceLUT_DCT)) {
					axis->bufferLUT = app->localFFTPlan_inverse->axes[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id].bufferLUT;
#if(VKFFT_BACKEND==0)
					axis->bufferLUTDeviceMemory = app->localFFTPlan_inverse->axes[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id].bufferLUTDeviceMemory;
#endif
					axis->bufferLUTSize = app->localFFTPlan_inverse->axes[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id].bufferLUTSize;
					axis->referenceLUT = 1;
				}
				else {
					pfUINT checkRadixOrder = 1;
					for (pfUINT i = 0; i < axis->specializationConstants.numStages; i++)
						if (FFTPlan->axes[0][0].specializationConstants.stageRadix[i] != axis->specializationConstants.stageRadix[i]) checkRadixOrder = 0;
					if (checkRadixOrder) {
						for (pfUINT i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
							if (axis->specializationConstants.raderContainer[i].type == 0) {
								for (pfUINT k = 0; k < axis->specializationConstants.raderContainer[i].numStages; k++) {
									if (FFTPlan->axes[0][0].specializationConstants.raderContainer[i].stageRadix[k] != axis->specializationConstants.raderContainer[i].stageRadix[k]) checkRadixOrder = 0;
								}
							}
						}
					}
					if (checkRadixOrder && (!disableReferenceLUT_DCT) && (axis->specializationConstants.axis_id >= 1) && (!((!axis->specializationConstants.reorderFourStep) && (FFTPlan->numAxisUploads[axis->specializationConstants.axis_id] > 1))) && ((axis->specializationConstants.fft_dim_full.data.i == FFTPlan->axes[0][0].specializationConstants.fft_dim_full.data.i) && (FFTPlan->numAxisUploads[axis->specializationConstants.axis_id] == 1) && (axis->specializationConstants.fft_dim_full.data.i < axis->specializationConstants.maxSingleSizeStrided.data.i / axis->specializationConstants.registerBoost)) && (((!axis->specializationConstants.performDCT) && (!axis->specializationConstants.performDST)) || (app->configuration.size[axis->specializationConstants.axis_id] == app->configuration.size[0]))) {
						axis->bufferLUT = FFTPlan->axes[0][axis->specializationConstants.axis_upload_id].bufferLUT;
#if(VKFFT_BACKEND==0)
						axis->bufferLUTDeviceMemory = FFTPlan->axes[0][axis->specializationConstants.axis_upload_id].bufferLUTDeviceMemory;
#endif
						axis->bufferLUTSize = FFTPlan->axes[0][axis->specializationConstants.axis_upload_id].bufferLUTSize;
						axis->referenceLUT = 1;
					}
					else {
                        for (int p = 1; p < axis->specializationConstants.axis_id; p++){
                            if(axis->referenceLUT == 0){
                                checkRadixOrder = 1;
                                for (pfUINT i = 0; i < axis->specializationConstants.numStages; i++)
                                    if (FFTPlan->axes[p][0].specializationConstants.stageRadix[i] != axis->specializationConstants.stageRadix[i]) checkRadixOrder = 0;
                                if (checkRadixOrder) {
                                    for (pfUINT i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
                                        if (axis->specializationConstants.raderContainer[i].type == 0) {
                                            for (pfUINT k = 0; k < axis->specializationConstants.raderContainer[i].numStages; k++) {
                                                if (FFTPlan->axes[p][0].specializationConstants.raderContainer[i].stageRadix[k] != axis->specializationConstants.raderContainer[i].stageRadix[k]) checkRadixOrder = 0;
                                            }
                                        }
                                    }
                                }
                                if (checkRadixOrder && (!disableReferenceLUT_DCT) && (axis->specializationConstants.fft_dim_full.data.i == FFTPlan->axes[p][0].specializationConstants.fft_dim_full.data.i) && (((!axis->specializationConstants.performDCT) && (!axis->specializationConstants.performDST)) || (app->configuration.size[axis->specializationConstants.axis_id] == app->configuration.size[p]))) {
                                    axis->bufferLUT = FFTPlan->axes[p][axis->specializationConstants.axis_upload_id].bufferLUT;
#if(VKFFT_BACKEND==0)
                                    axis->bufferLUTDeviceMemory = FFTPlan->axes[p][axis->specializationConstants.axis_upload_id].bufferLUTDeviceMemory;
#endif
                                    axis->bufferLUTSize = FFTPlan->axes[p][axis->specializationConstants.axis_upload_id].bufferLUTSize;
                                    axis->referenceLUT = 1;
                                }
                            }
                        }
                        if(axis->referenceLUT == 0){
#if(VKFFT_BACKEND==0)
							resFFT = allocateBufferVulkan(app, &axis->bufferLUT, &axis->bufferLUTDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, axis->bufferLUTSize);
							if (resFFT != VKFFT_SUCCESS) {
								deleteVkFFT(app);
								free(tempLUT);
								tempLUT = 0;
								return resFFT;
							}
							resFFT = VkFFT_TransferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
							if (resFFT != VKFFT_SUCCESS) {
								deleteVkFFT(app);
								free(tempLUT);
								tempLUT = 0;
								return resFFT;
							}
#elif(VKFFT_BACKEND==1)
							res = cudaMalloc((void**)&axis->bufferLUT, axis->bufferLUTSize);
							if (res != cudaSuccess) {
								deleteVkFFT(app);
								free(tempLUT);
								tempLUT = 0;
								return VKFFT_ERROR_FAILED_TO_ALLOCATE;
							}
							resFFT = VkFFT_TransferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
							if (resFFT != VKFFT_SUCCESS) {
								deleteVkFFT(app);
								free(tempLUT);
								tempLUT = 0;
								return resFFT;
							}
#elif(VKFFT_BACKEND==2)
							res = hipMalloc((void**)&axis->bufferLUT, axis->bufferLUTSize);
							if (res != hipSuccess) {
								deleteVkFFT(app);
								free(tempLUT);
								tempLUT = 0;
								return VKFFT_ERROR_FAILED_TO_ALLOCATE;
							}
							resFFT = VkFFT_TransferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
							if (resFFT != VKFFT_SUCCESS) {
								deleteVkFFT(app);
								free(tempLUT);
								tempLUT = 0;
								return resFFT;
							}
#elif(VKFFT_BACKEND==3)
							axis->bufferLUT = clCreateBuffer(app->configuration.context[0], CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, axis->bufferLUTSize, tempLUT, &res);
							if (res != CL_SUCCESS) {
								deleteVkFFT(app);
								free(tempLUT);
								tempLUT = 0;
								return VKFFT_ERROR_FAILED_TO_ALLOCATE;
							}
#elif(VKFFT_BACKEND==4)
							ze_device_mem_alloc_desc_t device_desc = VKFFT_ZERO_INIT;
							device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
							res = zeMemAllocDevice(app->configuration.context[0], &device_desc, axis->bufferLUTSize, sizeof(float), app->configuration.device[0], &axis->bufferLUT);
							if (res != ZE_RESULT_SUCCESS) {
								deleteVkFFT(app);
								free(tempLUT);
								tempLUT = 0;
								return VKFFT_ERROR_FAILED_TO_ALLOCATE;
							}
							resFFT = VkFFT_TransferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
							if (resFFT != VKFFT_SUCCESS) {
								deleteVkFFT(app);
								free(tempLUT);
								tempLUT = 0;
								return resFFT;
							}
#elif(VKFFT_BACKEND==5)
							axis->bufferLUT = app->configuration.device->newBuffer(axis->bufferLUTSize, MTL::ResourceStorageModePrivate);
							resFFT = VkFFT_TransferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
							if (resFFT != VKFFT_SUCCESS) {
								deleteVkFFT(app);
								free(tempLUT);
								tempLUT = 0;
								return resFFT;
							}
#endif
						}
					}
				}
			}
			free(tempLUT);
			tempLUT = 0;		
		}
		else if (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory) {
			pfLD double_PI = pfFPinit("3.14159265358979323846264338327950288419716939937510");
			axis->bufferLUTSize = currentLUTPos * 2 * sizeof(double);
			if (axis->bufferLUTSize == 0) axis->bufferLUTSize = sizeof(double);
			double* tempLUT = (double*)malloc(axis->bufferLUTSize);
			if (!tempLUT) {
				deleteVkFFT(app);
				return VKFFT_ERROR_MALLOC_FAILED;
			}
			pfUINT localStageSize = axis->specializationConstants.stageRadix[0];
			pfUINT localStageSum = 0;
			for (pfUINT i = 1; i < axis->specializationConstants.numStages; i++) {
				if ((axis->specializationConstants.stageRadix[i] & (axis->specializationConstants.stageRadix[i] - 1)) == 0) {
					for (pfUINT k = 0; k < log2(axis->specializationConstants.stageRadix[i]); k++) {
						for (pfUINT j = 0; j < localStageSize; j++) {
							tempLUT[2 * (j + localStageSum)] = (double)pfcos(j * double_PI / localStageSize / pow(2, k));
							tempLUT[2 * (j + localStageSum) + 1] = (double)pfsin(j * double_PI / localStageSize / pow(2, k));
						}
						localStageSum += localStageSize;
					}
				}
				else if (axis->specializationConstants.rader_generator[i] > 0) {
					for (pfUINT j = 0; j < localStageSize; j++) {
						for (pfINT k = (axis->specializationConstants.stageRadix[i] - 1); k >= 0; k--) {
							tempLUT[2 * (k + localStageSum)] = (double)pfcos(j * 2.0 * k / axis->specializationConstants.stageRadix[i] * double_PI / localStageSize);
							tempLUT[2 * (k + localStageSum) + 1] = (double)pfsin(j * 2.0 * k / axis->specializationConstants.stageRadix[i] * double_PI / localStageSize);
						}
						localStageSum += (axis->specializationConstants.stageRadix[i]);
					}
				}
				else {
					for (pfUINT k = (axis->specializationConstants.stageRadix[i] - 1); k > 0; k--) {
						for (pfUINT j = 0; j < localStageSize; j++) {
							tempLUT[2 * (j + localStageSum)] = (double)pfcos(j * 2.0 * k / axis->specializationConstants.stageRadix[i] * double_PI / localStageSize);
							tempLUT[2 * (j + localStageSum) + 1] = (double)pfsin(j * 2.0 * k / axis->specializationConstants.stageRadix[i] * double_PI / localStageSize);
						}
						localStageSum += localStageSize;
					}
				}
				localStageSize *= axis->specializationConstants.stageRadix[i];
			}


			if (axis->specializationConstants.useRader) {
				for (pfUINT i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
					if (axis->specializationConstants.raderContainer[i].type) {
						if (!axis->specializationConstants.inline_rader_kernel) {
							for (pfUINT j = 0; j < (axis->specializationConstants.raderContainer[i].prime - 1); j++) {//fix later
								pfUINT g_pow = 1;
								for (pfUINT t = 0; t < axis->specializationConstants.raderContainer[i].prime - 1 - j; t++) {
									g_pow = (g_pow * axis->specializationConstants.raderContainer[i].generator) % axis->specializationConstants.raderContainer[i].prime;
								}
								tempLUT[2 * (j + axis->specializationConstants.raderContainer[i].RaderKernelOffsetLUT)] = (double)pfcos(2.0 * g_pow * double_PI / axis->specializationConstants.raderContainer[i].prime);
								tempLUT[2 * (j + axis->specializationConstants.raderContainer[i].RaderKernelOffsetLUT) + 1] = (double)(-pfsin(2.0 * g_pow * double_PI / axis->specializationConstants.raderContainer[i].prime));
							}
						}
					}
					else {
						localStageSize = axis->specializationConstants.raderContainer[i].stageRadix[0];
						localStageSum = 0;
						for (pfUINT l = 1; l < axis->specializationConstants.raderContainer[i].numStages; l++) {
							if ((axis->specializationConstants.raderContainer[i].stageRadix[l] & (axis->specializationConstants.raderContainer[i].stageRadix[l] - 1)) == 0) {
								for (pfUINT k = 0; k < log2(axis->specializationConstants.raderContainer[i].stageRadix[l]); k++) {
									for (pfUINT j = 0; j < localStageSize; j++) {
										tempLUT[2 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUT)] = (double)pfcos(j * double_PI / localStageSize / pow(2, k));
										tempLUT[2 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUT) + 1] = (double)pfsin(j * double_PI / localStageSize / pow(2, k));
									}
									localStageSum += localStageSize;
								}
							}
							else {
								for (pfUINT k = (axis->specializationConstants.raderContainer[i].stageRadix[l] - 1); k > 0; k--) {
									for (pfUINT j = 0; j < localStageSize; j++) {
										tempLUT[2 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUT)] = (double)pfcos(j * 2.0 * k / axis->specializationConstants.raderContainer[i].stageRadix[l] * double_PI / localStageSize);
										tempLUT[2 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUT) + 1] = (double)pfsin(j * 2.0 * k / axis->specializationConstants.raderContainer[i].stageRadix[l] * double_PI / localStageSize);
									}
									localStageSum += localStageSize;
								}
							}
							localStageSize *= axis->specializationConstants.raderContainer[i].stageRadix[l];
						}

						localStageSize = axis->specializationConstants.raderContainer[i].stageRadix[axis->specializationConstants.raderContainer[i].numStages - 1];
						localStageSum = 0;
						for (pfINT l = (pfINT)axis->specializationConstants.raderContainer[i].numStages - 2; l >= 0; l--) {
							if ((axis->specializationConstants.raderContainer[i].stageRadix[l] & (axis->specializationConstants.raderContainer[i].stageRadix[l] - 1)) == 0) {
								for (pfUINT k = 0; k < log2(axis->specializationConstants.raderContainer[i].stageRadix[l]); k++) {
									for (pfUINT j = 0; j < localStageSize; j++) {
										tempLUT[2 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUTiFFT)] = (double)pfcos(j * double_PI / localStageSize / pow(2, k));
										tempLUT[2 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUTiFFT) + 1] = (double)pfsin(j * double_PI / localStageSize / pow(2, k));
									}
									localStageSum += localStageSize;
								}
							}
							else {
								for (pfUINT k = (axis->specializationConstants.raderContainer[i].stageRadix[l] - 1); k > 0; k--) {
									for (pfUINT j = 0; j < localStageSize; j++) {
										tempLUT[2 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUTiFFT)] = (double)pfcos(j * 2.0 * k / axis->specializationConstants.raderContainer[i].stageRadix[l] * double_PI / localStageSize);
										tempLUT[2 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUTiFFT) + 1] = (double)pfsin(j * 2.0 * k / axis->specializationConstants.raderContainer[i].stageRadix[l] * double_PI / localStageSize);
									}
									localStageSum += localStageSize;
								}
							}
							localStageSize *= axis->specializationConstants.raderContainer[i].stageRadix[l];
						}

						if (!axis->specializationConstants.inline_rader_kernel) {
							double* raderFFTkernel = (double*)axis->specializationConstants.raderContainer[i].raderFFTkernel;
							for (pfUINT j = 0; j < (axis->specializationConstants.raderContainer[i].prime - 1); j++) {//fix later
								tempLUT[2 * (j + axis->specializationConstants.raderContainer[i].RaderKernelOffsetLUT)] = (double)(raderFFTkernel[2 * j] / (pfLD)(axis->specializationConstants.raderContainer[i].prime - 1));
								tempLUT[2 * (j + axis->specializationConstants.raderContainer[i].RaderKernelOffsetLUT) + 1] = (double)(raderFFTkernel[2 * j + 1] / (pfLD)(axis->specializationConstants.raderContainer[i].prime - 1));
							}
						}
					}
				}
			}
			if ((axis->specializationConstants.axis_upload_id > 0) && (app->configuration.useLUT_4step == 1)) {
				for (pfUINT i = 0; i < (pfUINT)axis->specializationConstants.stageStartSize.data.i; i++) {
					for (pfUINT j = 0; j < (pfUINT)axis->specializationConstants.fftDim.data.i; j++) {
						pfLD angle = 2 * double_PI * ((i * j) / (pfLD)(axis->specializationConstants.stageStartSize.data.i * axis->specializationConstants.fftDim.data.i));
						tempLUT[maxStageSum * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize.data.i)] = (double)pfcos(angle);
						tempLUT[maxStageSum * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize.data.i) + 1] = (double)pfsin(angle);
					}
				}
			}
			if ((((((axis->specializationConstants.performDCT == 3) || (axis->specializationConstants.performDST == 3)) && (axis->specializationConstants.actualInverse == 0)) || (((axis->specializationConstants.performDCT == 2) || (axis->specializationConstants.performDST == 2)) && (axis->specializationConstants.actualInverse == 1))) && ((axis->specializationConstants.axis_upload_id == (axis->specializationConstants.numAxisUploads-1)) && (!((axis->specializationConstants.useBluesteinFFT && (axis->specializationConstants.reverseBluesteinMultiUpload == 1)))))) || (((((axis->specializationConstants.performDCT == 2) || (axis->specializationConstants.performDST == 2)) && (axis->specializationConstants.actualInverse == 0)) || (((axis->specializationConstants.performDCT == 3) || (axis->specializationConstants.performDST == 3)) && (axis->specializationConstants.actualInverse == 1))) && (((axis->specializationConstants.axis_upload_id == 0) && (!((axis->specializationConstants.useBluesteinFFT && (axis->specializationConstants.reverseBluesteinMultiUpload == 0) && (axis->specializationConstants.numAxisUploads > 1))))) || ((axis->specializationConstants.axis_upload_id == (axis->specializationConstants.numAxisUploads-1)) && (axis->specializationConstants.useBluesteinFFT && (axis->specializationConstants.reverseBluesteinMultiUpload == 1)))))) {
				for (pfUINT j = 0; j < app->configuration.size[axis->specializationConstants.axis_id] / 2 + 2; j++) {
					pfLD angle = (double_PI / 2.0 / (pfLD)(app->configuration.size[axis->specializationConstants.axis_id])) * j;
					tempLUT[2 * axis->specializationConstants.startDCT3LUT.data.i + 2 * j] = (double)pfcos(angle);
					tempLUT[2 * axis->specializationConstants.startDCT3LUT.data.i + 2 * j + 1] = (double)pfsin(angle);
				}
			}
			if (((axis->specializationConstants.performDCT == 4) || (axis->specializationConstants.performDST == 4)) && (app->configuration.size[axis->specializationConstants.axis_id] % 2 == 0) && ((axis->specializationConstants.axis_upload_id == (axis->specializationConstants.numAxisUploads-1)) && (!((axis->specializationConstants.useBluesteinFFT && (axis->specializationConstants.reverseBluesteinMultiUpload == 1)))))) {
				for (pfUINT j = 0; j < app->configuration.size[axis->specializationConstants.axis_id] / 4 + 2; j++) {
					pfLD angle = (double_PI / 2.0 / (pfLD)(app->configuration.size[axis->specializationConstants.axis_id] / 2)) * j;
					tempLUT[2 * axis->specializationConstants.startDCT3LUT.data.i + 2 * j] = (double)pfcos(angle);
					tempLUT[2 * axis->specializationConstants.startDCT3LUT.data.i + 2 * j + 1] = (double)pfsin(angle);
				}
			}
			if (((axis->specializationConstants.performDCT == 4) || (axis->specializationConstants.performDST == 4)) && (app->configuration.size[axis->specializationConstants.axis_id] % 2 == 0) && (((axis->specializationConstants.axis_upload_id == 0) && (!((axis->specializationConstants.useBluesteinFFT && (axis->specializationConstants.reverseBluesteinMultiUpload == 0) && (axis->specializationConstants.numAxisUploads > 1))))) || ((axis->specializationConstants.axis_upload_id == (axis->specializationConstants.numAxisUploads-1)) && (axis->specializationConstants.useBluesteinFFT && (axis->specializationConstants.reverseBluesteinMultiUpload == 1))))) {
				for (pfUINT j = 0; j < app->configuration.size[axis->specializationConstants.axis_id] / 2; j++) {
					pfLD angle = (-double_PI / 8.0 / (pfLD)(app->configuration.size[axis->specializationConstants.axis_id] / 2)) * (2 * j + 1);
					tempLUT[2 * axis->specializationConstants.startDCT4LUT.data.i + 2 * j] = (double)pfcos(angle);
					tempLUT[2 * axis->specializationConstants.startDCT4LUT.data.i + 2 * j + 1] = (double)pfsin(angle);
				}
			}
			axis->referenceLUT = 0;
			if ((axis->specializationConstants.reverseBluesteinMultiUpload == 1) && (!disableReferenceLUT_DCT)) {
				axis->bufferLUT = FFTPlan->axes[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id].bufferLUT;
#if(VKFFT_BACKEND==0)
				axis->bufferLUTDeviceMemory = FFTPlan->axes[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id].bufferLUTDeviceMemory;
#endif
				axis->bufferLUTSize = FFTPlan->axes[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id].bufferLUTSize;
				axis->referenceLUT = 1;
			}
			else {
				if ((!inverse) && (!app->configuration.makeForwardPlanOnly) && (!disableReferenceLUT_DCT)) {
					axis->bufferLUT = app->localFFTPlan_inverse->axes[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id].bufferLUT;
#if(VKFFT_BACKEND==0)
					axis->bufferLUTDeviceMemory = app->localFFTPlan_inverse->axes[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id].bufferLUTDeviceMemory;
#endif
					axis->bufferLUTSize = app->localFFTPlan_inverse->axes[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id].bufferLUTSize;
					axis->referenceLUT = 1;
				}
				else {
					pfUINT checkRadixOrder = 1;
					for (pfUINT i = 0; i < axis->specializationConstants.numStages; i++)
						if (FFTPlan->axes[0][0].specializationConstants.stageRadix[i] != axis->specializationConstants.stageRadix[i]) checkRadixOrder = 0;
					if (checkRadixOrder) {
						for (pfUINT i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
							if (axis->specializationConstants.raderContainer[i].type == 0) {
								for (pfUINT k = 0; k < axis->specializationConstants.raderContainer[i].numStages; k++) {
									if (FFTPlan->axes[0][0].specializationConstants.raderContainer[i].stageRadix[k] != axis->specializationConstants.raderContainer[i].stageRadix[k]) checkRadixOrder = 0;
								}
							}
						}
					}
					if (checkRadixOrder && (!disableReferenceLUT_DCT) && (axis->specializationConstants.axis_id >= 1) && (!((!axis->specializationConstants.reorderFourStep) && (FFTPlan->numAxisUploads[axis->specializationConstants.axis_id] > 1))) && ((axis->specializationConstants.fft_dim_full.data.i == FFTPlan->axes[0][0].specializationConstants.fft_dim_full.data.i) && (FFTPlan->numAxisUploads[axis->specializationConstants.axis_id] == 1) && (axis->specializationConstants.fft_dim_full.data.i < axis->specializationConstants.maxSingleSizeStrided.data.i / axis->specializationConstants.registerBoost)) && ((((!axis->specializationConstants.performDCT) && (!axis->specializationConstants.performDST)) && (!axis->specializationConstants.performDST)) || (app->configuration.size[axis->specializationConstants.axis_id] == app->configuration.size[0]))) {
						axis->bufferLUT = FFTPlan->axes[0][axis->specializationConstants.axis_upload_id].bufferLUT;
#if(VKFFT_BACKEND==0)
						axis->bufferLUTDeviceMemory = FFTPlan->axes[0][axis->specializationConstants.axis_upload_id].bufferLUTDeviceMemory;
#endif
						axis->bufferLUTSize = FFTPlan->axes[0][axis->specializationConstants.axis_upload_id].bufferLUTSize;
						axis->referenceLUT = 1;
					}
					else {
                        for (int p = 1; p < axis->specializationConstants.axis_id; p++){
                            if(axis->referenceLUT == 0){
                                checkRadixOrder = 1;
                                for (pfUINT i = 0; i < axis->specializationConstants.numStages; i++)
                                    if (FFTPlan->axes[p][0].specializationConstants.stageRadix[i] != axis->specializationConstants.stageRadix[i]) checkRadixOrder = 0;
                                if (checkRadixOrder) {
                                    for (pfUINT i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
                                        if (axis->specializationConstants.raderContainer[i].type == 0) {
                                            for (pfUINT k = 0; k < axis->specializationConstants.raderContainer[i].numStages; k++) {
                                                if (FFTPlan->axes[p][0].specializationConstants.raderContainer[i].stageRadix[k] != axis->specializationConstants.raderContainer[i].stageRadix[k]) checkRadixOrder = 0;
                                            }
                                        }
                                    }
                                }
                                if (checkRadixOrder && (!disableReferenceLUT_DCT) && (axis->specializationConstants.fft_dim_full.data.i == FFTPlan->axes[p][0].specializationConstants.fft_dim_full.data.i) && (((!axis->specializationConstants.performDCT) && (!axis->specializationConstants.performDST)) || (app->configuration.size[axis->specializationConstants.axis_id] == app->configuration.size[p]))) {
                                    axis->bufferLUT = FFTPlan->axes[p][axis->specializationConstants.axis_upload_id].bufferLUT;
#if(VKFFT_BACKEND==0)
                                    axis->bufferLUTDeviceMemory = FFTPlan->axes[p][axis->specializationConstants.axis_upload_id].bufferLUTDeviceMemory;
#endif
                                    axis->bufferLUTSize = FFTPlan->axes[p][axis->specializationConstants.axis_upload_id].bufferLUTSize;
                                    axis->referenceLUT = 1;
                                }
                            }
                        }
                        if(axis->referenceLUT == 0){
#if(VKFFT_BACKEND==0)
							resFFT = allocateBufferVulkan(app, &axis->bufferLUT, &axis->bufferLUTDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, axis->bufferLUTSize);
							if (resFFT != VKFFT_SUCCESS) {
								deleteVkFFT(app);
								free(tempLUT);
								tempLUT = 0;
								return resFFT;
							}
							resFFT = VkFFT_TransferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
							if (resFFT != VKFFT_SUCCESS) {
								deleteVkFFT(app);
								free(tempLUT);
								tempLUT = 0;
								return resFFT;
							}
#elif(VKFFT_BACKEND==1)
							res = cudaMalloc((void**)&axis->bufferLUT, axis->bufferLUTSize);
							if (res != cudaSuccess) {
								deleteVkFFT(app);
								free(tempLUT);
								tempLUT = 0;
								return VKFFT_ERROR_FAILED_TO_ALLOCATE;
							}
							resFFT = VkFFT_TransferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
							if (resFFT != VKFFT_SUCCESS) {
								deleteVkFFT(app);
								free(tempLUT);
								tempLUT = 0;
								return resFFT;
							}
#elif(VKFFT_BACKEND==2)
							res = hipMalloc((void**)&axis->bufferLUT, axis->bufferLUTSize);
							if (res != hipSuccess) {
								deleteVkFFT(app);
								free(tempLUT);
								tempLUT = 0;
								return VKFFT_ERROR_FAILED_TO_ALLOCATE;
							}
							resFFT = VkFFT_TransferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
							if (resFFT != VKFFT_SUCCESS) {
								deleteVkFFT(app);
								free(tempLUT);
								tempLUT = 0;
								return resFFT;
							}
#elif(VKFFT_BACKEND==3)
							axis->bufferLUT = clCreateBuffer(app->configuration.context[0], CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, axis->bufferLUTSize, tempLUT, &res);
							if (res != CL_SUCCESS) {
								deleteVkFFT(app);
								free(tempLUT);
								tempLUT = 0;
								return VKFFT_ERROR_FAILED_TO_ALLOCATE;
							}
#elif(VKFFT_BACKEND==4)
							ze_device_mem_alloc_desc_t device_desc = VKFFT_ZERO_INIT;
							device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
							res = zeMemAllocDevice(app->configuration.context[0], &device_desc, axis->bufferLUTSize, sizeof(float), app->configuration.device[0], &axis->bufferLUT);
							if (res != ZE_RESULT_SUCCESS) {
								deleteVkFFT(app);
								free(tempLUT);
								tempLUT = 0;
								return VKFFT_ERROR_FAILED_TO_ALLOCATE;
							}
							resFFT = VkFFT_TransferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
							if (resFFT != VKFFT_SUCCESS) {
								deleteVkFFT(app);
								free(tempLUT);
								tempLUT = 0;
								return resFFT;
							}
#elif(VKFFT_BACKEND==5)
							axis->bufferLUT = app->configuration.device->newBuffer(axis->bufferLUTSize, MTL::ResourceStorageModePrivate);
							resFFT = VkFFT_TransferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
							if (resFFT != VKFFT_SUCCESS) {
								deleteVkFFT(app);
								free(tempLUT);
								tempLUT = 0;
								return resFFT;
							}
#endif
						}
					}
				}
			}
			free(tempLUT);
			tempLUT = 0;
		}
		else {
			double double_PI = 3.14159265358979323846264338327950288419716939937510;
			axis->bufferLUTSize = currentLUTPos * 2 * sizeof(float);		
			if (axis->bufferLUTSize == 0) axis->bufferLUTSize = sizeof(float);
			float* tempLUT = (float*)malloc(axis->bufferLUTSize);
			if (!tempLUT) {
				deleteVkFFT(app);
				return VKFFT_ERROR_MALLOC_FAILED;
			}
			pfUINT localStageSize = axis->specializationConstants.stageRadix[0];
			pfUINT localStageSum = 0;
			for (pfUINT i = 1; i < axis->specializationConstants.numStages; i++) {
				if ((axis->specializationConstants.stageRadix[i] & (axis->specializationConstants.stageRadix[i] - 1)) == 0) {
					for (pfUINT k = 0; k < log2(axis->specializationConstants.stageRadix[i]); k++) {
						for (pfUINT j = 0; j < localStageSize; j++) {
							tempLUT[2 * (j + localStageSum)] = (float)pfcos(j * double_PI / localStageSize / pow(2, k));
							tempLUT[2 * (j + localStageSum) + 1] = (float)pfsin(j * double_PI / localStageSize / pow(2, k));
						}
						localStageSum += localStageSize;
					}
				}
				else if (axis->specializationConstants.rader_generator[i] > 0) {
					for (pfUINT j = 0; j < localStageSize; j++) {
						for (pfINT k = (axis->specializationConstants.stageRadix[i] - 1); k >= 0; k--) {
							tempLUT[2 * (k + localStageSum)] = (float)pfcos(j * 2.0 * k / axis->specializationConstants.stageRadix[i] * double_PI / localStageSize);
							tempLUT[2 * (k + localStageSum) + 1] = (float)pfsin(j * 2.0 * k / axis->specializationConstants.stageRadix[i] * double_PI / localStageSize);
						}
						localStageSum += (axis->specializationConstants.stageRadix[i]);
					}
				}
				else {
					for (pfUINT k = (axis->specializationConstants.stageRadix[i] - 1); k > 0; k--) {
						for (pfUINT j = 0; j < localStageSize; j++) {
							tempLUT[2 * (j + localStageSum)] = (float)pfcos(j * 2.0 * k / axis->specializationConstants.stageRadix[i] * double_PI / localStageSize);
							tempLUT[2 * (j + localStageSum) + 1] = (float)pfsin(j * 2.0 * k / axis->specializationConstants.stageRadix[i] * double_PI / localStageSize);
						}
						localStageSum += localStageSize;
					}
				}
				localStageSize *= axis->specializationConstants.stageRadix[i];
			}

			if (axis->specializationConstants.useRader) {
				for (pfUINT i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
					if (axis->specializationConstants.raderContainer[i].type) {
						if (!axis->specializationConstants.inline_rader_kernel) {
							for (pfUINT j = 0; j < (axis->specializationConstants.raderContainer[i].prime - 1); j++) {//fix later
								pfUINT g_pow = 1;
								for (pfUINT t = 0; t < axis->specializationConstants.raderContainer[i].prime - 1 - j; t++) {
									g_pow = (g_pow * axis->specializationConstants.raderContainer[i].generator) % axis->specializationConstants.raderContainer[i].prime;
								}
								tempLUT[2 * (j + axis->specializationConstants.raderContainer[i].RaderKernelOffsetLUT)] = (float)(pfcos(2.0 * g_pow * double_PI / axis->specializationConstants.raderContainer[i].prime));
								tempLUT[2 * (j + axis->specializationConstants.raderContainer[i].RaderKernelOffsetLUT) + 1] = (float)(-pfsin(2.0 * g_pow * double_PI / axis->specializationConstants.raderContainer[i].prime));
							}
						}
					}
					else {
						localStageSize = axis->specializationConstants.raderContainer[i].stageRadix[0];
						localStageSum = 0;
						for (pfUINT l = 1; l < axis->specializationConstants.raderContainer[i].numStages; l++) {
							if ((axis->specializationConstants.raderContainer[i].stageRadix[l] & (axis->specializationConstants.raderContainer[i].stageRadix[l] - 1)) == 0) {
								for (pfUINT k = 0; k < log2(axis->specializationConstants.raderContainer[i].stageRadix[l]); k++) {
									for (pfUINT j = 0; j < localStageSize; j++) {
										tempLUT[2 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUT)] = (float)pfcos(j * double_PI / localStageSize / pow(2, k));
										tempLUT[2 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUT) + 1] = (float)pfsin(j * double_PI / localStageSize / pow(2, k));
									}
									localStageSum += localStageSize;
								}
							}
							else {
								for (pfUINT k = (axis->specializationConstants.raderContainer[i].stageRadix[l] - 1); k > 0; k--) {
									for (pfUINT j = 0; j < localStageSize; j++) {
										tempLUT[2 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUT)] = (float)pfcos(j * 2.0 * k / axis->specializationConstants.raderContainer[i].stageRadix[l] * double_PI / localStageSize);
										tempLUT[2 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUT) + 1] = (float)pfsin(j * 2.0 * k / axis->specializationConstants.raderContainer[i].stageRadix[l] * double_PI / localStageSize);
									}
									localStageSum += localStageSize;
								}
							}
							localStageSize *= axis->specializationConstants.raderContainer[i].stageRadix[l];
						}
						localStageSize = axis->specializationConstants.raderContainer[i].stageRadix[axis->specializationConstants.raderContainer[i].numStages - 1];
						localStageSum = 0;
						for (pfINT l = (pfINT)axis->specializationConstants.raderContainer[i].numStages - 2; l >= 0; l--) {
							if ((axis->specializationConstants.raderContainer[i].stageRadix[l] & (axis->specializationConstants.raderContainer[i].stageRadix[l] - 1)) == 0) {
								for (pfUINT k = 0; k < log2(axis->specializationConstants.raderContainer[i].stageRadix[l]); k++) {
									for (pfUINT j = 0; j < localStageSize; j++) {
										tempLUT[2 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUTiFFT)] = (float)pfcos(j * double_PI / localStageSize / pow(2, k));
										tempLUT[2 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUTiFFT) + 1] = (float)pfsin(j * double_PI / localStageSize / pow(2, k));
									}
									localStageSum += localStageSize;
								}
							}
							else {
								for (pfUINT k = (axis->specializationConstants.raderContainer[i].stageRadix[l] - 1); k > 0; k--) {
									for (pfUINT j = 0; j < localStageSize; j++) {
										tempLUT[2 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUTiFFT)] = (float)pfcos(j * 2.0 * k / axis->specializationConstants.raderContainer[i].stageRadix[l] * double_PI / localStageSize);
										tempLUT[2 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUTiFFT) + 1] = (float)pfsin(j * 2.0 * k / axis->specializationConstants.raderContainer[i].stageRadix[l] * double_PI / localStageSize);
									}
									localStageSum += localStageSize;
								}
							}
							localStageSize *= axis->specializationConstants.raderContainer[i].stageRadix[l];
						}
						if (!axis->specializationConstants.inline_rader_kernel) {
							float* raderFFTkernel = (float*)axis->specializationConstants.raderContainer[i].raderFFTkernel;
							for (pfUINT j = 0; j < (axis->specializationConstants.raderContainer[i].prime - 1); j++) {//fix later
								tempLUT[2 * (j + axis->specializationConstants.raderContainer[i].RaderKernelOffsetLUT)] = (float)(raderFFTkernel[2 * j] / (axis->specializationConstants.raderContainer[i].prime - 1));
								tempLUT[2 * (j + axis->specializationConstants.raderContainer[i].RaderKernelOffsetLUT) + 1] = (float)(raderFFTkernel[2 * j + 1] / (axis->specializationConstants.raderContainer[i].prime - 1));
							}
						}
					}
				}
			}

			if ((axis->specializationConstants.axis_upload_id > 0) && (app->configuration.useLUT_4step == 1)) {
				for (pfUINT i = 0; i < (pfUINT)axis->specializationConstants.stageStartSize.data.i; i++) {
					for (pfUINT j = 0; j < (pfUINT)axis->specializationConstants.fftDim.data.i; j++) {
						double angle = 2 * double_PI * ((i * j) / (double)(axis->specializationConstants.stageStartSize.data.i * axis->specializationConstants.fftDim.data.i));
						tempLUT[maxStageSum * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize.data.i)] = (float)pfcos(angle);
						tempLUT[maxStageSum * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize.data.i) + 1] = (float)pfsin(angle);
					}
				}
			}
			if ((((((axis->specializationConstants.performDCT == 3) || (axis->specializationConstants.performDST == 3)) && (axis->specializationConstants.actualInverse == 0)) || (((axis->specializationConstants.performDCT == 2) || (axis->specializationConstants.performDST == 2)) && (axis->specializationConstants.actualInverse == 1))) && ((axis->specializationConstants.axis_upload_id == (axis->specializationConstants.numAxisUploads-1)) && (!((axis->specializationConstants.useBluesteinFFT && (axis->specializationConstants.reverseBluesteinMultiUpload == 1)))))) || (((((axis->specializationConstants.performDCT == 2) || (axis->specializationConstants.performDST == 2)) && (axis->specializationConstants.actualInverse == 0)) || (((axis->specializationConstants.performDCT == 3) || (axis->specializationConstants.performDST == 3)) && (axis->specializationConstants.actualInverse == 1))) && (((axis->specializationConstants.axis_upload_id == 0) && (!((axis->specializationConstants.useBluesteinFFT && (axis->specializationConstants.reverseBluesteinMultiUpload == 0) && (axis->specializationConstants.numAxisUploads > 1))))) || ((axis->specializationConstants.axis_upload_id == (axis->specializationConstants.numAxisUploads-1)) && (axis->specializationConstants.useBluesteinFFT && (axis->specializationConstants.reverseBluesteinMultiUpload == 1)))))) {
				for (pfUINT j = 0; j < app->configuration.size[axis->specializationConstants.axis_id] / 2 + 2; j++) {
					double angle = (double_PI / 2.0 / (double)(app->configuration.size[axis->specializationConstants.axis_id])) * j;
					tempLUT[2 * axis->specializationConstants.startDCT3LUT.data.i + 2 * j] = (float)pfcos(angle);
					tempLUT[2 * axis->specializationConstants.startDCT3LUT.data.i + 2 * j + 1] = (float)pfsin(angle);
				}
			}
			if (((axis->specializationConstants.performDCT == 4) || (axis->specializationConstants.performDST == 4)) && (app->configuration.size[axis->specializationConstants.axis_id] % 2 == 0) && ((axis->specializationConstants.axis_upload_id == (axis->specializationConstants.numAxisUploads-1)) && (!((axis->specializationConstants.useBluesteinFFT && (axis->specializationConstants.reverseBluesteinMultiUpload == 1)))))) {
				for (pfUINT j = 0; j < app->configuration.size[axis->specializationConstants.axis_id] / 4 + 2; j++) {
					double angle = (double_PI / 2.0 / (double)(app->configuration.size[axis->specializationConstants.axis_id] / 2)) * j;
					tempLUT[2 * axis->specializationConstants.startDCT3LUT.data.i + 2 * j] = (float)pfcos(angle);
					tempLUT[2 * axis->specializationConstants.startDCT3LUT.data.i + 2 * j + 1] = (float)pfsin(angle);
				}
			}
			if (((axis->specializationConstants.performDCT == 4) || (axis->specializationConstants.performDST == 4)) && (app->configuration.size[axis->specializationConstants.axis_id] % 2 == 0) && (((axis->specializationConstants.axis_upload_id == 0) && (!((axis->specializationConstants.useBluesteinFFT && (axis->specializationConstants.reverseBluesteinMultiUpload == 0) && (axis->specializationConstants.numAxisUploads > 1))))) || ((axis->specializationConstants.axis_upload_id == (axis->specializationConstants.numAxisUploads-1)) && (axis->specializationConstants.useBluesteinFFT && (axis->specializationConstants.reverseBluesteinMultiUpload == 1))))) {
				for (pfUINT j = 0; j < app->configuration.size[axis->specializationConstants.axis_id] / 2; j++) {
					double angle = (-double_PI / 8.0 / (double)(app->configuration.size[axis->specializationConstants.axis_id] / 2)) * (2 * j + 1);
					tempLUT[2 * axis->specializationConstants.startDCT4LUT.data.i + 2 * j] = (float)pfcos(angle);
					tempLUT[2 * axis->specializationConstants.startDCT4LUT.data.i + 2 * j + 1] = (float)pfsin(angle);
				}
			}
			axis->referenceLUT = 0;

			if ((axis->specializationConstants.reverseBluesteinMultiUpload == 1) && (!disableReferenceLUT_DCT)) {
				axis->bufferLUT = FFTPlan->axes[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id].bufferLUT;
#if(VKFFT_BACKEND==0)
				axis->bufferLUTDeviceMemory = FFTPlan->axes[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id].bufferLUTDeviceMemory;
#endif
				axis->bufferLUTSize = FFTPlan->axes[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id].bufferLUTSize;
				axis->referenceLUT = 1;
			}
			else {
				if ((!inverse) && (!app->configuration.makeForwardPlanOnly) && (!disableReferenceLUT_DCT)) {
					axis->bufferLUT = app->localFFTPlan_inverse->axes[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id].bufferLUT;
#if(VKFFT_BACKEND==0)
					axis->bufferLUTDeviceMemory = app->localFFTPlan_inverse->axes[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id].bufferLUTDeviceMemory;
#endif
					axis->bufferLUTSize = app->localFFTPlan_inverse->axes[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id].bufferLUTSize;
					axis->referenceLUT = 1;
				}
				else {
					pfUINT checkRadixOrder = 1;
					for (pfUINT i = 0; i < axis->specializationConstants.numStages; i++)
						if (FFTPlan->axes[0][0].specializationConstants.stageRadix[i] != axis->specializationConstants.stageRadix[i]) checkRadixOrder = 0;
					if (checkRadixOrder) {
						for (pfUINT i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
							if (axis->specializationConstants.raderContainer[i].type == 0) {
								for (pfUINT k = 0; k < axis->specializationConstants.raderContainer[i].numStages; k++) {
									if (FFTPlan->axes[0][0].specializationConstants.raderContainer[i].stageRadix[k] != axis->specializationConstants.raderContainer[i].stageRadix[k]) checkRadixOrder = 0;
								}
							}
						}
					}
					if (checkRadixOrder && (!disableReferenceLUT_DCT) && (axis->specializationConstants.axis_id >= 1) && (!((!axis->specializationConstants.reorderFourStep) && (FFTPlan->numAxisUploads[axis->specializationConstants.axis_id] > 1))) && ((axis->specializationConstants.fft_dim_full.data.i == FFTPlan->axes[0][0].specializationConstants.fft_dim_full.data.i) && (FFTPlan->numAxisUploads[axis->specializationConstants.axis_id] == 1) && (axis->specializationConstants.fft_dim_full.data.i < axis->specializationConstants.maxSingleSizeStrided.data.i / axis->specializationConstants.registerBoost)) && ((((!axis->specializationConstants.performDCT) && (!axis->specializationConstants.performDST)) && (!axis->specializationConstants.performDST)) || (app->configuration.size[axis->specializationConstants.axis_id] == app->configuration.size[0]))) {
						axis->bufferLUT = FFTPlan->axes[0][axis->specializationConstants.axis_upload_id].bufferLUT;
#if(VKFFT_BACKEND==0)
						axis->bufferLUTDeviceMemory = FFTPlan->axes[0][axis->specializationConstants.axis_upload_id].bufferLUTDeviceMemory;
#endif
						axis->bufferLUTSize = FFTPlan->axes[0][axis->specializationConstants.axis_upload_id].bufferLUTSize;
						axis->referenceLUT = 1;
					}
					else {
                        for (int p = 1; p < axis->specializationConstants.axis_id; p++){
                            if(axis->referenceLUT == 0){
                                checkRadixOrder = 1;
                                for (pfUINT i = 0; i < axis->specializationConstants.numStages; i++)
                                    if (FFTPlan->axes[p][0].specializationConstants.stageRadix[i] != axis->specializationConstants.stageRadix[i]) checkRadixOrder = 0;
                                if (checkRadixOrder) {
                                    for (pfUINT i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
                                        if (axis->specializationConstants.raderContainer[i].type == 0) {
                                            for (pfUINT k = 0; k < axis->specializationConstants.raderContainer[i].numStages; k++) {
                                                if (FFTPlan->axes[p][0].specializationConstants.raderContainer[i].stageRadix[k] != axis->specializationConstants.raderContainer[i].stageRadix[k]) checkRadixOrder = 0;
                                            }
                                        }
                                    }
                                }
                                if (checkRadixOrder && (!disableReferenceLUT_DCT) && (axis->specializationConstants.fft_dim_full.data.i == FFTPlan->axes[p][0].specializationConstants.fft_dim_full.data.i) && (((!axis->specializationConstants.performDCT) && (!axis->specializationConstants.performDST)) || (app->configuration.size[axis->specializationConstants.axis_id] == app->configuration.size[p]))) {
                                    axis->bufferLUT = FFTPlan->axes[p][axis->specializationConstants.axis_upload_id].bufferLUT;
#if(VKFFT_BACKEND==0)
                                    axis->bufferLUTDeviceMemory = FFTPlan->axes[p][axis->specializationConstants.axis_upload_id].bufferLUTDeviceMemory;
#endif
                                    axis->bufferLUTSize = FFTPlan->axes[p][axis->specializationConstants.axis_upload_id].bufferLUTSize;
                                    axis->referenceLUT = 1;
                                }
                            }
                        }
                        if(axis->referenceLUT == 0){
#if(VKFFT_BACKEND==0)
							resFFT = allocateBufferVulkan(app, &axis->bufferLUT, &axis->bufferLUTDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, axis->bufferLUTSize);
							if (resFFT != VKFFT_SUCCESS) {
								deleteVkFFT(app);
								free(tempLUT);
								tempLUT = 0;
								return resFFT;
							}
							resFFT = VkFFT_TransferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
							if (resFFT != VKFFT_SUCCESS) {
								deleteVkFFT(app);
								free(tempLUT);
								tempLUT = 0;
								return resFFT;
							}
#elif(VKFFT_BACKEND==1)
							res = cudaMalloc((void**)&axis->bufferLUT, axis->bufferLUTSize);
							if (res != cudaSuccess) {
								deleteVkFFT(app);
								free(tempLUT);
								tempLUT = 0;
								return VKFFT_ERROR_FAILED_TO_ALLOCATE;
							}
							resFFT = VkFFT_TransferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
							if (resFFT != VKFFT_SUCCESS) {
								deleteVkFFT(app);
								free(tempLUT);
								tempLUT = 0;
								return resFFT;
							}
#elif(VKFFT_BACKEND==2)
							res = hipMalloc((void**)&axis->bufferLUT, axis->bufferLUTSize);
							if (res != hipSuccess) {
								deleteVkFFT(app);
								free(tempLUT);
								tempLUT = 0;
								return VKFFT_ERROR_FAILED_TO_ALLOCATE;
							}
							resFFT = VkFFT_TransferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
							if (resFFT != VKFFT_SUCCESS) {
								deleteVkFFT(app);
								free(tempLUT);
								tempLUT = 0;
								return resFFT;
							}
#elif(VKFFT_BACKEND==3)
							axis->bufferLUT = clCreateBuffer(app->configuration.context[0], CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, axis->bufferLUTSize, tempLUT, &res);
							if (res != CL_SUCCESS) {
								deleteVkFFT(app);
								free(tempLUT);
								tempLUT = 0;
								return VKFFT_ERROR_FAILED_TO_ALLOCATE;
							}
#elif(VKFFT_BACKEND==4)
							ze_device_mem_alloc_desc_t device_desc = VKFFT_ZERO_INIT;
							device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
							res = zeMemAllocDevice(app->configuration.context[0], &device_desc, axis->bufferLUTSize, sizeof(float), app->configuration.device[0], &axis->bufferLUT);
							if (res != ZE_RESULT_SUCCESS) {
								deleteVkFFT(app);
								free(tempLUT);
								tempLUT = 0;
								return VKFFT_ERROR_FAILED_TO_ALLOCATE;
							}
							resFFT = VkFFT_TransferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
							if (resFFT != VKFFT_SUCCESS) {
								deleteVkFFT(app);
								free(tempLUT);
								tempLUT = 0;
								return resFFT;
							}
#elif(VKFFT_BACKEND==5)
							axis->bufferLUT = app->configuration.device->newBuffer(axis->bufferLUTSize, MTL::ResourceStorageModePrivate);
							resFFT = VkFFT_TransferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
							if (resFFT != VKFFT_SUCCESS) {
								deleteVkFFT(app);
								free(tempLUT);
								tempLUT = 0;
								return resFFT;
							}
#endif
						}
					}
				}
			}
			free(tempLUT);
			tempLUT = 0;
		}
	}
	return resFFT;
}

static inline VkFFTResult VkFFT_AllocateRaderUintLUT(VkFFTApplication* app, VkFFTAxis* axis){
	VkFFTResult resFFT = VKFFT_SUCCESS;
#if(VKFFT_BACKEND==0)
	VkResult res = VK_SUCCESS;
#elif(VKFFT_BACKEND==1)
	cudaError_t res = cudaSuccess;
#elif(VKFFT_BACKEND==2)
	hipError_t res = hipSuccess;
#elif(VKFFT_BACKEND==3)
	cl_int res = CL_SUCCESS;
#elif(VKFFT_BACKEND==4)
	ze_result_t res = ZE_RESULT_SUCCESS;
#elif(VKFFT_BACKEND==5)
#endif
	//allocate RaderUintLUT
	if (axis->specializationConstants.raderUintLUT) {
		if (app->bufferRaderUintLUT[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id] == 0) {
			app->bufferRaderUintLUTSize[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id] = 0;
			for (pfUINT i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
				app->bufferRaderUintLUTSize[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id] += axis->specializationConstants.raderContainer[i].prime * sizeof(uint32_t);
			}
			uint32_t* tempRaderUintLUT = (uint32_t*)malloc(app->bufferRaderUintLUTSize[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id]);
			if (!tempRaderUintLUT) {
				deleteVkFFT(app);
				return VKFFT_ERROR_MALLOC_FAILED;
			}
			pfUINT current_offset = 0;
			for (pfUINT i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
				if (axis->specializationConstants.raderContainer[i].prime > 0) {
					axis->specializationConstants.raderContainer[i].raderUintLUToffset = (int)current_offset;
					pfUINT g_pow = 1;
					tempRaderUintLUT[current_offset] = 1;
					current_offset++;
					for (pfUINT t = 0; t < axis->specializationConstants.raderContainer[i].prime - 1; t++) {
						g_pow = (g_pow * axis->specializationConstants.raderContainer[i].generator) % axis->specializationConstants.raderContainer[i].prime;
						tempRaderUintLUT[current_offset] = (uint32_t)g_pow;
						current_offset++;
					}
				}
			}

#if(VKFFT_BACKEND==0)
			resFFT = allocateBufferVulkan(app, &app->bufferRaderUintLUT[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id], &app->bufferRaderUintLUTDeviceMemory[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, app->bufferRaderUintLUTSize[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id]);
			if (resFFT != VKFFT_SUCCESS) {
				deleteVkFFT(app);
				free(tempRaderUintLUT);
				tempRaderUintLUT = 0;
				return resFFT;
			}
			resFFT = VkFFT_TransferDataFromCPU(app, tempRaderUintLUT, &app->bufferRaderUintLUT[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id], app->bufferRaderUintLUTSize[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id]);
			if (resFFT != VKFFT_SUCCESS) {
				deleteVkFFT(app);
				free(tempRaderUintLUT);
				tempRaderUintLUT = 0;
				return resFFT;
			}
#elif(VKFFT_BACKEND==1)
			res = cudaMalloc((void**)&app->bufferRaderUintLUT[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id], app->bufferRaderUintLUTSize[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id]);
			if (res != cudaSuccess) {
				deleteVkFFT(app);
				free(tempRaderUintLUT);
				tempRaderUintLUT = 0;
				return VKFFT_ERROR_FAILED_TO_ALLOCATE;
			}
			resFFT = VkFFT_TransferDataFromCPU(app, tempRaderUintLUT, &app->bufferRaderUintLUT[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id], app->bufferRaderUintLUTSize[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id]);
			if (resFFT != VKFFT_SUCCESS) {
				deleteVkFFT(app);
				free(tempRaderUintLUT);
				tempRaderUintLUT = 0;
				return resFFT;
			}
#elif(VKFFT_BACKEND==2)
			res = hipMalloc((void**)&app->bufferRaderUintLUT[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id], app->bufferRaderUintLUTSize[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id]);
			if (res != hipSuccess) {
				deleteVkFFT(app);
				free(tempRaderUintLUT);
				tempRaderUintLUT = 0;
				return VKFFT_ERROR_FAILED_TO_ALLOCATE;
			}
			resFFT = VkFFT_TransferDataFromCPU(app, tempRaderUintLUT, &app->bufferRaderUintLUT[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id], app->bufferRaderUintLUTSize[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id]);
			if (resFFT != VKFFT_SUCCESS) {
				deleteVkFFT(app);
				free(tempRaderUintLUT);
				tempRaderUintLUT = 0;
				return resFFT;
			}
#elif(VKFFT_BACKEND==3)
			app->bufferRaderUintLUT[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id] = clCreateBuffer(app->configuration.context[0], CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, app->bufferRaderUintLUTSize[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id], tempRaderUintLUT, &res);
			if (res != CL_SUCCESS) {
				deleteVkFFT(app);
				free(tempRaderUintLUT);
				tempRaderUintLUT = 0;
				return VKFFT_ERROR_FAILED_TO_ALLOCATE;
			}
#elif(VKFFT_BACKEND==4)
			ze_device_mem_alloc_desc_t device_desc = VKFFT_ZERO_INIT;
			device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
			res = zeMemAllocDevice(app->configuration.context[0], &device_desc, app->bufferRaderUintLUTSize[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id], sizeof(uint32_t), app->configuration.device[0], &app->bufferRaderUintLUT[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id]);
			if (res != ZE_RESULT_SUCCESS) {
				deleteVkFFT(app);
				free(tempRaderUintLUT);
				tempRaderUintLUT = 0;
				return VKFFT_ERROR_FAILED_TO_ALLOCATE;
			}
			resFFT = VkFFT_TransferDataFromCPU(app, tempRaderUintLUT, &app->bufferRaderUintLUT[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id], app->bufferRaderUintLUTSize[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id]);
			if (resFFT != VKFFT_SUCCESS) {
				deleteVkFFT(app);
				free(tempRaderUintLUT);
				tempRaderUintLUT = 0;
				return resFFT;
			}
#elif(VKFFT_BACKEND==5)
			app->bufferRaderUintLUT[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id] = app->configuration.device->newBuffer(app->bufferRaderUintLUTSize[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id], MTL::ResourceStorageModePrivate);
			resFFT = VkFFT_TransferDataFromCPU(app, tempRaderUintLUT, &app->bufferRaderUintLUT[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id], app->bufferRaderUintLUTSize[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id]);
			if (resFFT != VKFFT_SUCCESS) {
				deleteVkFFT(app);
				free(tempRaderUintLUT);
				tempRaderUintLUT = 0;
				return resFFT;
			}
#endif
			free(tempRaderUintLUT);
			tempRaderUintLUT = 0;
		}
		else {
			pfUINT current_offset = 0;
			for (pfUINT i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
				if (axis->specializationConstants.raderContainer[i].prime > 0) {
					axis->specializationConstants.raderContainer[i].raderUintLUToffset = (int)current_offset;
					pfUINT g_pow = 1;
					current_offset += axis->specializationConstants.raderContainer[i].prime;
				}
			}
		}

		axis->bufferRaderUintLUT = app->bufferRaderUintLUT[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id];
#if(VKFFT_BACKEND==0)
		axis->bufferRaderUintLUTDeviceMemory = app->bufferRaderUintLUTDeviceMemory[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id];
#endif
		axis->bufferRaderUintLUTSize = app->bufferRaderUintLUTSize[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id];
	}
	return resFFT;
}

static inline VkFFTResult VkFFT_AllocateLUT_R2C(VkFFTApplication* app, VkFFTPlan* FFTPlan, VkFFTAxis* axis, pfUINT inverse) {
	VkFFTResult resFFT = VKFFT_SUCCESS;
#if(VKFFT_BACKEND==0)
	VkResult res = VK_SUCCESS;
#elif(VKFFT_BACKEND==1)
	cudaError_t res = cudaSuccess;
#elif(VKFFT_BACKEND==2)
	hipError_t res = hipSuccess;
#elif(VKFFT_BACKEND==3)
	cl_int res = CL_SUCCESS;
#elif(VKFFT_BACKEND==4)
	ze_result_t res = ZE_RESULT_SUCCESS;
#elif(VKFFT_BACKEND==5)
	#endif
	if (app->configuration.useLUT == 1) {
		if (app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) {
			PfContainer in = VKFFT_ZERO_INIT;
			PfContainer temp1 = VKFFT_ZERO_INIT;
			in.type = 22;
			
			pfLD double_PI = pfFPinit("3.14159265358979323846264338327950288419716939937510");
			axis->bufferLUTSize = (app->configuration.size[0] / 2) * 4 * sizeof(double);
			double* tempLUT = (double*)malloc(axis->bufferLUTSize);
			if (!tempLUT) {
				deleteVkFFT(app);
				return VKFFT_ERROR_MALLOC_FAILED;
			}
			for (pfUINT i = 0; i < app->configuration.size[0] / 2; i++) {
				pfLD angle = double_PI * i / (app->configuration.size[0] / 2);
				
				in.data.d = pfcos(angle);
				PfConvToDoubleDouble(&axis->specializationConstants, &temp1, &in);
				tempLUT[4 * i] = (double)temp1.data.dd[0].data.d;
				tempLUT[4 * i + 1] = (double)temp1.data.dd[1].data.d;
				in.data.d = pfsin(angle);
				PfConvToDoubleDouble(&axis->specializationConstants, &temp1, &in);
				tempLUT[4 * i + 2] = (double)temp1.data.dd[0].data.d;
				tempLUT[4 * i + 3] = (double)temp1.data.dd[1].data.d;
			}
			axis->referenceLUT = 0;
			PfDeallocateContainer(&axis->specializationConstants, &temp1);
			if ((!inverse) && (!app->configuration.makeForwardPlanOnly)) {
				axis->bufferLUT = app->localFFTPlan_inverse->R2Cdecomposition.bufferLUT;
#if(VKFFT_BACKEND==0)
				axis->bufferLUTDeviceMemory = app->localFFTPlan_inverse->R2Cdecomposition.bufferLUTDeviceMemory;
#endif
				axis->bufferLUTSize = app->localFFTPlan_inverse->R2Cdecomposition.bufferLUTSize;
				axis->referenceLUT = 1;
			}
			else {
#if(VKFFT_BACKEND==0)
				resFFT = allocateBufferVulkan(app, &axis->bufferLUT, &axis->bufferLUTDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, axis->bufferLUTSize);
				if (resFFT != VKFFT_SUCCESS) {
					deleteVkFFT(app);
					free(tempLUT);
					tempLUT = 0;
					return resFFT;
				}
				resFFT = VkFFT_TransferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
				if (resFFT != VKFFT_SUCCESS) {
					deleteVkFFT(app);
					free(tempLUT);
					tempLUT = 0;
					return resFFT;
				}
#elif(VKFFT_BACKEND==1)
				res = cudaMalloc((void**)&axis->bufferLUT, axis->bufferLUTSize);
				if (res != cudaSuccess) {
					deleteVkFFT(app);
					free(tempLUT);
					tempLUT = 0;
					return VKFFT_ERROR_FAILED_TO_ALLOCATE;
				}
				resFFT = VkFFT_TransferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
				if (resFFT != VKFFT_SUCCESS) {
					deleteVkFFT(app);
					free(tempLUT);
					tempLUT = 0;
					return resFFT;
				}
#elif(VKFFT_BACKEND==2)
				res = hipMalloc((void**)&axis->bufferLUT, axis->bufferLUTSize);
				if (res != hipSuccess) {
					deleteVkFFT(app);
					free(tempLUT);
					tempLUT = 0;
					return VKFFT_ERROR_FAILED_TO_ALLOCATE;
				}
				resFFT = VkFFT_TransferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
				if (resFFT != VKFFT_SUCCESS) {
					deleteVkFFT(app);
					free(tempLUT);
					tempLUT = 0;
					return resFFT;
				}
#elif(VKFFT_BACKEND==3)
				axis->bufferLUT = clCreateBuffer(app->configuration.context[0], CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, axis->bufferLUTSize, tempLUT, &res);
				if (res != CL_SUCCESS) {
					deleteVkFFT(app);
					free(tempLUT);
					tempLUT = 0;
					return VKFFT_ERROR_FAILED_TO_ALLOCATE;
				}
#elif(VKFFT_BACKEND==4)
				ze_device_mem_alloc_desc_t device_desc = VKFFT_ZERO_INIT;
				device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
				res = zeMemAllocDevice(app->configuration.context[0], &device_desc, axis->bufferLUTSize, sizeof(float), app->configuration.device[0], &axis->bufferLUT);
				if (res != ZE_RESULT_SUCCESS) {
					deleteVkFFT(app);
					free(tempLUT);
					tempLUT = 0;
					return VKFFT_ERROR_FAILED_TO_ALLOCATE;
				}
				resFFT = VkFFT_TransferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
				if (resFFT != VKFFT_SUCCESS) {
					deleteVkFFT(app);
					free(tempLUT);
					tempLUT = 0;
					return resFFT;
				}
#elif(VKFFT_BACKEND==5)
				axis->bufferLUT = app->configuration.device->newBuffer(axis->bufferLUTSize, MTL::ResourceStorageModePrivate);

				resFFT = VkFFT_TransferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
				if (resFFT != VKFFT_SUCCESS) {
					deleteVkFFT(app);
					free(tempLUT);
					tempLUT = 0;
					return resFFT;
				}
#endif
				free(tempLUT);
				tempLUT = 0;
			}
		}
		else if (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory) {
			pfLD double_PI = pfFPinit("3.14159265358979323846264338327950288419716939937510");
			axis->bufferLUTSize = (app->configuration.size[0] / 2) * 2 * sizeof(double);
			double* tempLUT = (double*)malloc(axis->bufferLUTSize);
			if (!tempLUT) {
				deleteVkFFT(app);
				return VKFFT_ERROR_MALLOC_FAILED;
			}
			for (pfUINT i = 0; i < app->configuration.size[0] / 2; i++) {
				pfLD angle = double_PI * i / (app->configuration.size[0] / 2);
				tempLUT[2 * i] = (double)pfcos(angle);
				tempLUT[2 * i + 1] = (double)pfsin(angle);
			}
			axis->referenceLUT = 0;
			if ((!inverse) && (!app->configuration.makeForwardPlanOnly)) {
				axis->bufferLUT = app->localFFTPlan_inverse->R2Cdecomposition.bufferLUT;
#if(VKFFT_BACKEND==0)
				axis->bufferLUTDeviceMemory = app->localFFTPlan_inverse->R2Cdecomposition.bufferLUTDeviceMemory;
#endif
				axis->bufferLUTSize = app->localFFTPlan_inverse->R2Cdecomposition.bufferLUTSize;
				axis->referenceLUT = 1;
			}
			else {
#if(VKFFT_BACKEND==0)
				resFFT = allocateBufferVulkan(app, &axis->bufferLUT, &axis->bufferLUTDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, axis->bufferLUTSize);
				if (resFFT != VKFFT_SUCCESS) {
					deleteVkFFT(app);
					free(tempLUT);
					tempLUT = 0;
					return resFFT;
				}
				resFFT = VkFFT_TransferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
				if (resFFT != VKFFT_SUCCESS) {
					deleteVkFFT(app);
					free(tempLUT);
					tempLUT = 0;
					return resFFT;
				}
#elif(VKFFT_BACKEND==1)
				res = cudaMalloc((void**)&axis->bufferLUT, axis->bufferLUTSize);
				if (res != cudaSuccess) {
					deleteVkFFT(app);
					free(tempLUT);
					tempLUT = 0;
					return VKFFT_ERROR_FAILED_TO_ALLOCATE;
				}
				resFFT = VkFFT_TransferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
				if (resFFT != VKFFT_SUCCESS) {
					deleteVkFFT(app);
					free(tempLUT);
					tempLUT = 0;
					return resFFT;
				}
#elif(VKFFT_BACKEND==2)
				res = hipMalloc((void**)&axis->bufferLUT, axis->bufferLUTSize);
				if (res != hipSuccess) {
					deleteVkFFT(app);
					free(tempLUT);
					tempLUT = 0;
					return VKFFT_ERROR_FAILED_TO_ALLOCATE;
				}
				resFFT = VkFFT_TransferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
				if (resFFT != VKFFT_SUCCESS) {
					deleteVkFFT(app);
					free(tempLUT);
					tempLUT = 0;
					return resFFT;
				}
#elif(VKFFT_BACKEND==3)
				axis->bufferLUT = clCreateBuffer(app->configuration.context[0], CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, axis->bufferLUTSize, tempLUT, &res);
				if (res != CL_SUCCESS) {
					deleteVkFFT(app);
					free(tempLUT);
					tempLUT = 0;
					return VKFFT_ERROR_FAILED_TO_ALLOCATE;
				}
#elif(VKFFT_BACKEND==4)
				ze_device_mem_alloc_desc_t device_desc = VKFFT_ZERO_INIT;
				device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
				res = zeMemAllocDevice(app->configuration.context[0], &device_desc, axis->bufferLUTSize, sizeof(float), app->configuration.device[0], &axis->bufferLUT);
				if (res != ZE_RESULT_SUCCESS) {
					deleteVkFFT(app);
					free(tempLUT);
					tempLUT = 0;
					return VKFFT_ERROR_FAILED_TO_ALLOCATE;
				}
				resFFT = VkFFT_TransferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
				if (resFFT != VKFFT_SUCCESS) {
					deleteVkFFT(app);
					free(tempLUT);
					tempLUT = 0;
					return resFFT;
				}
#elif(VKFFT_BACKEND==5)
				axis->bufferLUT = app->configuration.device->newBuffer(axis->bufferLUTSize, MTL::ResourceStorageModePrivate);

				resFFT = VkFFT_TransferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
				if (resFFT != VKFFT_SUCCESS) {
					deleteVkFFT(app);
					free(tempLUT);
					tempLUT = 0;
					return resFFT;
				}
#endif
				free(tempLUT);
				tempLUT = 0;
			}
		}
		else {
			double double_PI = 3.14159265358979323846264338327950288419716939937510;
			axis->bufferLUTSize = (app->configuration.size[0] / 2) * 2 * sizeof(float);
			float* tempLUT = (float*)malloc(axis->bufferLUTSize);
			if (!tempLUT) {
				deleteVkFFT(app);
				return VKFFT_ERROR_MALLOC_FAILED;
			}
			for (pfUINT i = 0; i < app->configuration.size[0] / 2; i++) {
				double angle = double_PI * i / (app->configuration.size[0] / 2);
				tempLUT[2 * i] = (float)pfcos(angle);
				tempLUT[2 * i + 1] = (float)pfsin(angle);
			}
			axis->referenceLUT = 0;
			if ((!inverse) && (!app->configuration.makeForwardPlanOnly)) {
				axis->bufferLUT = app->localFFTPlan_inverse->R2Cdecomposition.bufferLUT;
#if(VKFFT_BACKEND==0)
				axis->bufferLUTDeviceMemory = app->localFFTPlan_inverse->R2Cdecomposition.bufferLUTDeviceMemory;
#endif
				axis->bufferLUTSize = app->localFFTPlan_inverse->R2Cdecomposition.bufferLUTSize;
				axis->referenceLUT = 1;
			}
			else {
#if(VKFFT_BACKEND==0)
				resFFT = allocateBufferVulkan(app, &axis->bufferLUT, &axis->bufferLUTDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, axis->bufferLUTSize);
				if (resFFT != VKFFT_SUCCESS) {
					deleteVkFFT(app);
					free(tempLUT);
					tempLUT = 0;
					return resFFT;
				}
				resFFT = VkFFT_TransferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
				if (resFFT != VKFFT_SUCCESS) {
					deleteVkFFT(app);
					free(tempLUT);
					tempLUT = 0;
					return resFFT;
				}
#elif(VKFFT_BACKEND==1)
				res = cudaMalloc((void**)&axis->bufferLUT, axis->bufferLUTSize);
				if (res != cudaSuccess) {
					deleteVkFFT(app);
					free(tempLUT);
					tempLUT = 0;
					return VKFFT_ERROR_FAILED_TO_ALLOCATE;
				}
				resFFT = VkFFT_TransferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
				if (resFFT != VKFFT_SUCCESS) {
					deleteVkFFT(app);
					free(tempLUT);
					tempLUT = 0;
					return resFFT;
				}
#elif(VKFFT_BACKEND==2)
				res = hipMalloc((void**)&axis->bufferLUT, axis->bufferLUTSize);
				if (res != hipSuccess) {
					deleteVkFFT(app);
					free(tempLUT);
					tempLUT = 0;
					return VKFFT_ERROR_FAILED_TO_ALLOCATE;
				}
				resFFT = VkFFT_TransferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
				if (resFFT != VKFFT_SUCCESS) {
					deleteVkFFT(app);
					free(tempLUT);
					tempLUT = 0;
					return resFFT;
				}
#elif(VKFFT_BACKEND==3)
				axis->bufferLUT = clCreateBuffer(app->configuration.context[0], CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, axis->bufferLUTSize, tempLUT, &res);
				if (res != CL_SUCCESS) {
					deleteVkFFT(app);
					free(tempLUT);
					tempLUT = 0;
					return VKFFT_ERROR_FAILED_TO_ALLOCATE;
				}
#elif(VKFFT_BACKEND==4)
				ze_device_mem_alloc_desc_t device_desc = VKFFT_ZERO_INIT;
				device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
				res = zeMemAllocDevice(app->configuration.context[0], &device_desc, axis->bufferLUTSize, sizeof(float), app->configuration.device[0], &axis->bufferLUT);
				if (res != ZE_RESULT_SUCCESS) {
					deleteVkFFT(app);
					free(tempLUT);
					tempLUT = 0;
					return VKFFT_ERROR_FAILED_TO_ALLOCATE;
				}
				resFFT = VkFFT_TransferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
				if (resFFT != VKFFT_SUCCESS) {
					deleteVkFFT(app);
					free(tempLUT);
					tempLUT = 0;
					return resFFT;
				}
#elif(VKFFT_BACKEND==5)
				axis->bufferLUT = app->configuration.device->newBuffer(axis->bufferLUTSize, MTL::ResourceStorageModePrivate);

				resFFT = VkFFT_TransferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
				if (resFFT != VKFFT_SUCCESS) {
					deleteVkFFT(app);
					free(tempLUT);
					tempLUT = 0;
					return resFFT;
				}
#endif
				free(tempLUT);
				tempLUT = 0;
			}
		}
	}
	return resFFT;
}
#endif
