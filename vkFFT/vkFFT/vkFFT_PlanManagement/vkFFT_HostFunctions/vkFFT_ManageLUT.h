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

static inline VkFFTResult VkFFT_AllocateLUT(VkFFTApplication* app, VkFFTPlan* FFTPlan, VkFFTAxis* axis, uint64_t inverse){
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
		uint64_t dimMult = 1;
		uint64_t maxStageSum = 0;
		for (uint64_t i = 0; i < axis->specializationConstants.numStages; i++) {
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
					maxStageSum += dimMult * 10;
					break;
				case 12:
					maxStageSum += dimMult * 11;
					break;
				case 13:
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
		for (uint64_t k = 0; k < axis->specializationConstants.numRaderPrimes; k++) {
			if (axis->specializationConstants.raderContainer[k].type == 0) {
				axis->specializationConstants.raderContainer[k].RaderRadixOffsetLUT = maxStageSum;
				for (uint64_t i = 0; i < axis->specializationConstants.raderContainer[k].numStages; i++) {
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
							maxStageSum += dimMult * 10;
							break;
						case 12:
							maxStageSum += dimMult * 11;
							break;
						case 13:
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
		for (uint64_t k = 0; k < axis->specializationConstants.numRaderPrimes; k++) {
			if (axis->specializationConstants.raderContainer[k].type == 0) {
				axis->specializationConstants.raderContainer[k].RaderRadixOffsetLUTiFFT = maxStageSum;
				for (int64_t i = axis->specializationConstants.raderContainer[k].numStages - 1; i >= 0; i--) {
					if (i < (int64_t)axis->specializationConstants.raderContainer[k].numStages - 1) {
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
							maxStageSum += dimMult * 10;
							break;
						case 12:
							maxStageSum += dimMult * 11;
							break;
						case 13:
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

		if (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory) {
			long double double_PI = 3.14159265358979323846264338327950288419716939937510L;
			if (axis->specializationConstants.axis_upload_id > 0) {
				if ((app->configuration.performDCT == 2) || (app->configuration.performDCT == 3)) {
					axis->specializationConstants.startDCT3LUT.type = 31;
					axis->specializationConstants.startDCT3LUT.data.i = (maxStageSum);
					if (app->configuration.useLUT_4step == 1) axis->specializationConstants.startDCT3LUT.data.i += axis->specializationConstants.stageStartSize.data.i * axis->specializationConstants.fftDim.data.i;
					axis->bufferLUTSize = (maxStageSum + (app->configuration.size[axis->specializationConstants.axis_id] / 2 + 2)) * 2 * sizeof(double);
				}
				else {
					if ((app->configuration.performDCT == 4) && (app->configuration.size[axis->specializationConstants.axis_id] % 2 == 0)) {
						axis->specializationConstants.startDCT3LUT.type = 31;
						axis->specializationConstants.startDCT3LUT.data.i = (maxStageSum);
						if (app->configuration.useLUT_4step == 1) axis->specializationConstants.startDCT3LUT.data.i += axis->specializationConstants.stageStartSize.data.i * axis->specializationConstants.fftDim.data.i;
						axis->specializationConstants.startDCT4LUT.type = 31;
						axis->specializationConstants.startDCT4LUT.data.i = (axis->specializationConstants.startDCT3LUT.data.i + (app->configuration.size[axis->specializationConstants.axis_id] / 4 + 2));
						axis->bufferLUTSize = (maxStageSum + (app->configuration.size[axis->specializationConstants.axis_id] / 4 + 2) + app->configuration.size[axis->specializationConstants.axis_id] / 2) * 2 * sizeof(double);
					}
					else
						axis->bufferLUTSize = (maxStageSum) * 2 * sizeof(double);
				}
				if (app->configuration.useLUT_4step == 1) axis->bufferLUTSize += axis->specializationConstants.stageStartSize.data.i * axis->specializationConstants.fftDim.data.i * 2 * sizeof(double);
			}
			else {
				if ((app->configuration.performDCT == 2) || (app->configuration.performDCT == 3)) {
					axis->specializationConstants.startDCT3LUT.type = 31;
					axis->specializationConstants.startDCT3LUT.data.i = (maxStageSum);
					axis->bufferLUTSize = (maxStageSum + (app->configuration.size[axis->specializationConstants.axis_id] / 2 + 2)) * 2 * sizeof(double);
				}
				else {
					if ((app->configuration.performDCT == 4) && (app->configuration.size[axis->specializationConstants.axis_id] % 2 == 0)) {
						axis->specializationConstants.startDCT3LUT.type = 31;
						axis->specializationConstants.startDCT3LUT.data.i = (maxStageSum);
						axis->specializationConstants.startDCT4LUT.type = 31;
						axis->specializationConstants.startDCT4LUT.data.i = (axis->specializationConstants.startDCT3LUT.data.i + (app->configuration.size[axis->specializationConstants.axis_id] / 4 + 2));
						axis->bufferLUTSize = (maxStageSum + (app->configuration.size[axis->specializationConstants.axis_id] / 4 + 2) + app->configuration.size[axis->specializationConstants.axis_id] / 2) * 2 * sizeof(double);

					}
					else
						axis->bufferLUTSize = (maxStageSum) * 2 * sizeof(double);
				}
			}
			if (axis->specializationConstants.useRader) {
				for (uint64_t i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
					if (!axis->specializationConstants.inline_rader_kernel) {
						axis->specializationConstants.raderContainer[i].RaderKernelOffsetLUT = axis->bufferLUTSize / (2 * sizeof(double));
						axis->bufferLUTSize += (axis->specializationConstants.raderContainer[i].prime - 1) * 2 * sizeof(double);
					}
				}
			}
			if (axis->bufferLUTSize == 0) axis->bufferLUTSize = sizeof(double);
			double* tempLUT = (double*)malloc(axis->bufferLUTSize);
			if (!tempLUT) {
				deleteVkFFT(app);
				return VKFFT_ERROR_MALLOC_FAILED;
			}
			uint64_t localStageSize = axis->specializationConstants.stageRadix[0];
			uint64_t localStageSum = 0;
			for (uint64_t i = 1; i < axis->specializationConstants.numStages; i++) {
				if ((axis->specializationConstants.stageRadix[i] & (axis->specializationConstants.stageRadix[i] - 1)) == 0) {
					for (uint64_t k = 0; k < log2(axis->specializationConstants.stageRadix[i]); k++) {
						for (uint64_t j = 0; j < localStageSize; j++) {
							tempLUT[2 * (j + localStageSum)] = (double)cos(j * double_PI / localStageSize / pow(2, k));
							tempLUT[2 * (j + localStageSum) + 1] = (double)sin(j * double_PI / localStageSize / pow(2, k));
						}
						localStageSum += localStageSize;
					}
				}
				else if (axis->specializationConstants.rader_generator[i] > 0) {
					for (uint64_t j = 0; j < localStageSize; j++) {
						for (int64_t k = (axis->specializationConstants.stageRadix[i] - 1); k >= 0; k--) {
							tempLUT[2 * (k + localStageSum)] = (double)cos(j * 2.0 * k / axis->specializationConstants.stageRadix[i] * double_PI / localStageSize);
							tempLUT[2 * (k + localStageSum) + 1] = (double)sin(j * 2.0 * k / axis->specializationConstants.stageRadix[i] * double_PI / localStageSize);
						}
						localStageSum += (axis->specializationConstants.stageRadix[i]);
					}
				}
				else {
					for (uint64_t k = (axis->specializationConstants.stageRadix[i] - 1); k > 0; k--) {
						for (uint64_t j = 0; j < localStageSize; j++) {
							tempLUT[2 * (j + localStageSum)] = (double)cos(j * 2.0 * k / axis->specializationConstants.stageRadix[i] * double_PI / localStageSize);
							tempLUT[2 * (j + localStageSum) + 1] = (double)sin(j * 2.0 * k / axis->specializationConstants.stageRadix[i] * double_PI / localStageSize);
						}
						localStageSum += localStageSize;
					}
				}
				localStageSize *= axis->specializationConstants.stageRadix[i];
			}


			if (axis->specializationConstants.useRader) {
				for (uint64_t i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
					if (axis->specializationConstants.raderContainer[i].type) {
						if (!axis->specializationConstants.inline_rader_kernel) {
							for (uint64_t j = 0; j < (axis->specializationConstants.raderContainer[i].prime - 1); j++) {//fix later
								uint64_t g_pow = 1;
								for (uint64_t t = 0; t < axis->specializationConstants.raderContainer[i].prime - 1 - j; t++) {
									g_pow = (g_pow * axis->specializationConstants.raderContainer[i].generator) % axis->specializationConstants.raderContainer[i].prime;
								}
								tempLUT[2 * (j + axis->specializationConstants.raderContainer[i].RaderKernelOffsetLUT)] = (double)cos(2.0 * g_pow * double_PI / axis->specializationConstants.raderContainer[i].prime);
								tempLUT[2 * (j + axis->specializationConstants.raderContainer[i].RaderKernelOffsetLUT) + 1] = (double)(-sin(2.0 * g_pow * double_PI / axis->specializationConstants.raderContainer[i].prime));
							}
						}
					}
					else {
						localStageSize = axis->specializationConstants.raderContainer[i].stageRadix[0];
						localStageSum = 0;
						for (uint64_t l = 1; l < axis->specializationConstants.raderContainer[i].numStages; l++) {
							if ((axis->specializationConstants.raderContainer[i].stageRadix[l] & (axis->specializationConstants.raderContainer[i].stageRadix[l] - 1)) == 0) {
								for (uint64_t k = 0; k < log2(axis->specializationConstants.raderContainer[i].stageRadix[l]); k++) {
									for (uint64_t j = 0; j < localStageSize; j++) {
										tempLUT[2 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUT)] = (double)cos(j * double_PI / localStageSize / pow(2, k));
										tempLUT[2 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUT) + 1] = (double)sin(j * double_PI / localStageSize / pow(2, k));
									}
									localStageSum += localStageSize;
								}
							}
							else {
								for (uint64_t k = (axis->specializationConstants.raderContainer[i].stageRadix[l] - 1); k > 0; k--) {
									for (uint64_t j = 0; j < localStageSize; j++) {
										tempLUT[2 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUT)] = (double)cos(j * 2.0 * k / axis->specializationConstants.raderContainer[i].stageRadix[l] * double_PI / localStageSize);
										tempLUT[2 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUT) + 1] = (double)sin(j * 2.0 * k / axis->specializationConstants.raderContainer[i].stageRadix[l] * double_PI / localStageSize);
									}
									localStageSum += localStageSize;
								}
							}
							localStageSize *= axis->specializationConstants.raderContainer[i].stageRadix[l];
						}

						localStageSize = axis->specializationConstants.raderContainer[i].stageRadix[axis->specializationConstants.raderContainer[i].numStages - 1];
						localStageSum = 0;
						for (int64_t l = (int64_t)axis->specializationConstants.raderContainer[i].numStages - 2; l >= 0; l--) {
							if ((axis->specializationConstants.raderContainer[i].stageRadix[l] & (axis->specializationConstants.raderContainer[i].stageRadix[l] - 1)) == 0) {
								for (uint64_t k = 0; k < log2(axis->specializationConstants.raderContainer[i].stageRadix[l]); k++) {
									for (uint64_t j = 0; j < localStageSize; j++) {
										tempLUT[2 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUTiFFT)] = (double)cos(j * double_PI / localStageSize / pow(2, k));
										tempLUT[2 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUTiFFT) + 1] = (double)sin(j * double_PI / localStageSize / pow(2, k));
									}
									localStageSum += localStageSize;
								}
							}
							else {
								for (uint64_t k = (axis->specializationConstants.raderContainer[i].stageRadix[l] - 1); k > 0; k--) {
									for (uint64_t j = 0; j < localStageSize; j++) {
										tempLUT[2 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUTiFFT)] = (double)cos(j * 2.0 * k / axis->specializationConstants.raderContainer[i].stageRadix[l] * double_PI / localStageSize);
										tempLUT[2 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUTiFFT) + 1] = (double)sin(j * 2.0 * k / axis->specializationConstants.raderContainer[i].stageRadix[l] * double_PI / localStageSize);
									}
									localStageSum += localStageSize;
								}
							}
							localStageSize *= axis->specializationConstants.raderContainer[i].stageRadix[l];
						}

						if (!axis->specializationConstants.inline_rader_kernel) {
							double* raderFFTkernel = (double*)axis->specializationConstants.raderContainer[i].raderFFTkernel;
							for (uint64_t j = 0; j < (axis->specializationConstants.raderContainer[i].prime - 1); j++) {//fix later
								tempLUT[2 * (j + axis->specializationConstants.raderContainer[i].RaderKernelOffsetLUT)] = (double)(raderFFTkernel[2 * j] / (long double)(axis->specializationConstants.raderContainer[i].prime - 1));
								tempLUT[2 * (j + axis->specializationConstants.raderContainer[i].RaderKernelOffsetLUT) + 1] = (double)(raderFFTkernel[2 * j + 1] / (long double)(axis->specializationConstants.raderContainer[i].prime - 1));
							}
						}
					}
				}
			}
			if ((axis->specializationConstants.axis_upload_id > 0) && (app->configuration.useLUT_4step == 1)) {
				for (uint64_t i = 0; i < (uint64_t)axis->specializationConstants.stageStartSize.data.i; i++) {
					for (uint64_t j = 0; j < (uint64_t)axis->specializationConstants.fftDim.data.i; j++) {
						long double angle = 2 * double_PI * ((i * j) / (long double)(axis->specializationConstants.stageStartSize.data.i * axis->specializationConstants.fftDim.data.i));
						tempLUT[maxStageSum * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize.data.i)] = (double)cos(angle);
						tempLUT[maxStageSum * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize.data.i) + 1] = (double)sin(angle);
					}
				}
			}
			if ((app->configuration.performDCT == 2) || (app->configuration.performDCT == 3)) {
				for (uint64_t j = 0; j < app->configuration.size[axis->specializationConstants.axis_id] / 2 + 2; j++) {
					long double angle = (double_PI / 2.0 / (long double)(app->configuration.size[axis->specializationConstants.axis_id])) * j;
					tempLUT[2 * axis->specializationConstants.startDCT3LUT.data.i + 2 * j] = (double)cos(angle);
					tempLUT[2 * axis->specializationConstants.startDCT3LUT.data.i + 2 * j + 1] = (double)sin(angle);
				}
			}
			if ((app->configuration.performDCT == 4) && (app->configuration.size[axis->specializationConstants.axis_id] % 2 == 0)) {
				for (uint64_t j = 0; j < app->configuration.size[axis->specializationConstants.axis_id] / 4 + 2; j++) {
					long double angle = (double_PI / 2.0 / (long double)(app->configuration.size[axis->specializationConstants.axis_id] / 2)) * j;
					tempLUT[2 * axis->specializationConstants.startDCT3LUT.data.i + 2 * j] = (double)cos(angle);
					tempLUT[2 * axis->specializationConstants.startDCT3LUT.data.i + 2 * j + 1] = (double)sin(angle);
				}
				for (uint64_t j = 0; j < app->configuration.size[axis->specializationConstants.axis_id] / 2; j++) {
					long double angle = (-double_PI / 8.0 / (long double)(app->configuration.size[axis->specializationConstants.axis_id] / 2)) * (2 * j + 1);
					tempLUT[2 * axis->specializationConstants.startDCT4LUT.data.i + 2 * j] = (double)cos(angle);
					tempLUT[2 * axis->specializationConstants.startDCT4LUT.data.i + 2 * j + 1] = (double)sin(angle);
				}
			}
			axis->referenceLUT = 0;
			if (axis->specializationConstants.reverseBluesteinMultiUpload == 1) {
				axis->bufferLUT = FFTPlan->axes[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id].bufferLUT;
#if(VKFFT_BACKEND==0)
				axis->bufferLUTDeviceMemory = FFTPlan->axes[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id].bufferLUTDeviceMemory;
#endif
				axis->bufferLUTSize = FFTPlan->axes[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id].bufferLUTSize;
				axis->referenceLUT = 1;
			}
			else {
				if ((!inverse) && (!app->configuration.makeForwardPlanOnly)) {
					axis->bufferLUT = app->localFFTPlan_inverse->axes[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id].bufferLUT;
#if(VKFFT_BACKEND==0)
					axis->bufferLUTDeviceMemory = app->localFFTPlan_inverse->axes[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id].bufferLUTDeviceMemory;
#endif
					axis->bufferLUTSize = app->localFFTPlan_inverse->axes[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id].bufferLUTSize;
					axis->referenceLUT = 1;
				}
				else {
					uint64_t checkRadixOrder = 1;
					for (uint64_t i = 0; i < axis->specializationConstants.numStages; i++)
						if (FFTPlan->axes[0][0].specializationConstants.stageRadix[i] != axis->specializationConstants.stageRadix[i]) checkRadixOrder = 0;
					if (checkRadixOrder) {
						for (uint64_t i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
							if (axis->specializationConstants.raderContainer[i].type == 0) {
								for (uint64_t k = 0; k < axis->specializationConstants.raderContainer[i].numStages; k++) {
									if (FFTPlan->axes[0][0].specializationConstants.raderContainer[i].stageRadix[k] != axis->specializationConstants.raderContainer[i].stageRadix[k]) checkRadixOrder = 0;
								}
							}
						}
					}
					if (checkRadixOrder && (axis->specializationConstants.axis_id >= 1) && (!((!axis->specializationConstants.reorderFourStep) && (FFTPlan->numAxisUploads[axis->specializationConstants.axis_id] > 1))) && ((axis->specializationConstants.fft_dim_full.data.i == FFTPlan->axes[0][0].specializationConstants.fft_dim_full.data.i) && (FFTPlan->numAxisUploads[axis->specializationConstants.axis_id] == 1) && (axis->specializationConstants.fft_dim_full.data.i < axis->specializationConstants.maxSingleSizeStrided.data.i / axis->specializationConstants.registerBoost)) && ((!app->configuration.performDCT) || (app->configuration.size[axis->specializationConstants.axis_id] == app->configuration.size[0]))) {
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
                                for (uint64_t i = 0; i < axis->specializationConstants.numStages; i++)
                                    if (FFTPlan->axes[p][0].specializationConstants.stageRadix[i] != axis->specializationConstants.stageRadix[i]) checkRadixOrder = 0;
                                if (checkRadixOrder) {
                                    for (uint64_t i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
                                        if (axis->specializationConstants.raderContainer[i].type == 0) {
                                            for (uint64_t k = 0; k < axis->specializationConstants.raderContainer[i].numStages; k++) {
                                                if (FFTPlan->axes[p][0].specializationConstants.raderContainer[i].stageRadix[k] != axis->specializationConstants.raderContainer[i].stageRadix[k]) checkRadixOrder = 0;
                                            }
                                        }
                                    }
                                }
                                if (checkRadixOrder && (axis->specializationConstants.fft_dim_full.data.i == FFTPlan->axes[p][0].specializationConstants.fft_dim_full.data.i) && ((!app->configuration.performDCT) || (app->configuration.size[axis->specializationConstants.axis_id] == app->configuration.size[p]))) {
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
			if (axis->specializationConstants.axis_upload_id > 0) {
				if ((app->configuration.performDCT == 2) || (app->configuration.performDCT == 3)) {
					axis->specializationConstants.startDCT3LUT.type = 31;
					axis->specializationConstants.startDCT3LUT.data.i = (maxStageSum);
					if (app->configuration.useLUT_4step == 1) axis->specializationConstants.startDCT3LUT.data.i += axis->specializationConstants.stageStartSize.data.i * axis->specializationConstants.fftDim.data.i;
					axis->bufferLUTSize = (maxStageSum + (app->configuration.size[axis->specializationConstants.axis_id] / 2 + 2)) * 2 * sizeof(float);
				}
				else {
					if ((app->configuration.performDCT == 4) && (app->configuration.size[axis->specializationConstants.axis_id] % 2 == 0)) {
						axis->specializationConstants.startDCT3LUT.type = 31;
						axis->specializationConstants.startDCT3LUT.data.i = (maxStageSum);
						if (app->configuration.useLUT_4step == 1) axis->specializationConstants.startDCT3LUT.data.i += axis->specializationConstants.stageStartSize.data.i * axis->specializationConstants.fftDim.data.i;
						axis->specializationConstants.startDCT4LUT.type = 31;
						axis->specializationConstants.startDCT4LUT.data.i = (axis->specializationConstants.startDCT3LUT.data.i + (axis->specializationConstants.fftDim.data.i / 4 + 2));
						axis->bufferLUTSize = (maxStageSum + (app->configuration.size[axis->specializationConstants.axis_id] / 4 + 2) + app->configuration.size[axis->specializationConstants.axis_id] / 2) * 2 * sizeof(float);
					}
					else
						axis->bufferLUTSize = (maxStageSum) * 2 * sizeof(float);
				}
				if (app->configuration.useLUT_4step == 1) axis->bufferLUTSize += axis->specializationConstants.stageStartSize.data.i * axis->specializationConstants.fftDim.data.i * 2 * sizeof(float);
			}
			else {
				if ((app->configuration.performDCT == 2) || (app->configuration.performDCT == 3)) {
					axis->specializationConstants.startDCT3LUT.type = 31;
					axis->specializationConstants.startDCT3LUT.data.i = (maxStageSum);
					axis->bufferLUTSize = (maxStageSum + (app->configuration.size[axis->specializationConstants.axis_id] / 2 + 2)) * 2 * sizeof(float);
				}
				else {
					if ((app->configuration.performDCT == 4) && (app->configuration.size[axis->specializationConstants.axis_id] % 2 == 0)) {
						axis->specializationConstants.startDCT3LUT.type = 31;
						axis->specializationConstants.startDCT3LUT.data.i = (maxStageSum);
						axis->specializationConstants.startDCT4LUT.type = 31;
						axis->specializationConstants.startDCT4LUT.data.i = (axis->specializationConstants.startDCT3LUT.data.i + (app->configuration.size[axis->specializationConstants.axis_id] / 4 + 2));
						axis->bufferLUTSize = (maxStageSum + (app->configuration.size[axis->specializationConstants.axis_id] / 4 + 2) + app->configuration.size[axis->specializationConstants.axis_id] / 2) * 2 * sizeof(float);
					}
					else
						axis->bufferLUTSize = (maxStageSum) * 2 * sizeof(float);
				}
			}
			if (axis->specializationConstants.useRader) {
				for (uint64_t i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
					if (!axis->specializationConstants.inline_rader_kernel) {
						axis->specializationConstants.raderContainer[i].RaderKernelOffsetLUT = axis->bufferLUTSize / (2 * sizeof(float));
						axis->bufferLUTSize += (axis->specializationConstants.raderContainer[i].prime - 1) * 2 * sizeof(float);
					}
				}
			}
			if (axis->bufferLUTSize == 0) axis->bufferLUTSize = sizeof(float);
			float* tempLUT = (float*)malloc(axis->bufferLUTSize);
			if (!tempLUT) {
				deleteVkFFT(app);
				return VKFFT_ERROR_MALLOC_FAILED;
			}
			uint64_t localStageSize = axis->specializationConstants.stageRadix[0];
			uint64_t localStageSum = 0;
			for (uint64_t i = 1; i < axis->specializationConstants.numStages; i++) {
				if ((axis->specializationConstants.stageRadix[i] & (axis->specializationConstants.stageRadix[i] - 1)) == 0) {
					for (uint64_t k = 0; k < log2(axis->specializationConstants.stageRadix[i]); k++) {
						for (uint64_t j = 0; j < localStageSize; j++) {
							tempLUT[2 * (j + localStageSum)] = (float)cos(j * double_PI / localStageSize / pow(2, k));
							tempLUT[2 * (j + localStageSum) + 1] = (float)sin(j * double_PI / localStageSize / pow(2, k));
						}
						localStageSum += localStageSize;
					}
				}
				else if (axis->specializationConstants.rader_generator[i] > 0) {
					for (uint64_t j = 0; j < localStageSize; j++) {
						for (int64_t k = (axis->specializationConstants.stageRadix[i] - 1); k >= 0; k--) {
							tempLUT[2 * (k + localStageSum)] = (float)cos(j * 2.0 * k / axis->specializationConstants.stageRadix[i] * double_PI / localStageSize);
							tempLUT[2 * (k + localStageSum) + 1] = (float)sin(j * 2.0 * k / axis->specializationConstants.stageRadix[i] * double_PI / localStageSize);
						}
						localStageSum += (axis->specializationConstants.stageRadix[i]);
					}
				}
				else {
					for (uint64_t k = (axis->specializationConstants.stageRadix[i] - 1); k > 0; k--) {
						for (uint64_t j = 0; j < localStageSize; j++) {
							tempLUT[2 * (j + localStageSum)] = (float)cos(j * 2.0 * k / axis->specializationConstants.stageRadix[i] * double_PI / localStageSize);
							tempLUT[2 * (j + localStageSum) + 1] = (float)sin(j * 2.0 * k / axis->specializationConstants.stageRadix[i] * double_PI / localStageSize);
						}
						localStageSum += localStageSize;
					}
				}
				localStageSize *= axis->specializationConstants.stageRadix[i];
			}

			if (axis->specializationConstants.useRader) {
				for (uint64_t i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
					if (axis->specializationConstants.raderContainer[i].type) {
						if (!axis->specializationConstants.inline_rader_kernel) {
							for (uint64_t j = 0; j < (axis->specializationConstants.raderContainer[i].prime - 1); j++) {//fix later
								uint64_t g_pow = 1;
								for (uint64_t t = 0; t < axis->specializationConstants.raderContainer[i].prime - 1 - j; t++) {
									g_pow = (g_pow * axis->specializationConstants.raderContainer[i].generator) % axis->specializationConstants.raderContainer[i].prime;
								}
								tempLUT[2 * (j + axis->specializationConstants.raderContainer[i].RaderKernelOffsetLUT)] = (float)(cos(2.0 * g_pow * double_PI / axis->specializationConstants.raderContainer[i].prime));
								tempLUT[2 * (j + axis->specializationConstants.raderContainer[i].RaderKernelOffsetLUT) + 1] = (float)(-sin(2.0 * g_pow * double_PI / axis->specializationConstants.raderContainer[i].prime));
							}
						}
					}
					else {
						localStageSize = axis->specializationConstants.raderContainer[i].stageRadix[0];
						localStageSum = 0;
						for (uint64_t l = 1; l < axis->specializationConstants.raderContainer[i].numStages; l++) {
							if ((axis->specializationConstants.raderContainer[i].stageRadix[l] & (axis->specializationConstants.raderContainer[i].stageRadix[l] - 1)) == 0) {
								for (uint64_t k = 0; k < log2(axis->specializationConstants.raderContainer[i].stageRadix[l]); k++) {
									for (uint64_t j = 0; j < localStageSize; j++) {
										tempLUT[2 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUT)] = (float)cos(j * double_PI / localStageSize / pow(2, k));
										tempLUT[2 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUT) + 1] = (float)sin(j * double_PI / localStageSize / pow(2, k));
									}
									localStageSum += localStageSize;
								}
							}
							else {
								for (uint64_t k = (axis->specializationConstants.raderContainer[i].stageRadix[l] - 1); k > 0; k--) {
									for (uint64_t j = 0; j < localStageSize; j++) {
										tempLUT[2 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUT)] = (float)cos(j * 2.0 * k / axis->specializationConstants.raderContainer[i].stageRadix[l] * double_PI / localStageSize);
										tempLUT[2 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUT) + 1] = (float)sin(j * 2.0 * k / axis->specializationConstants.raderContainer[i].stageRadix[l] * double_PI / localStageSize);
									}
									localStageSum += localStageSize;
								}
							}
							localStageSize *= axis->specializationConstants.raderContainer[i].stageRadix[l];
						}
						localStageSize = axis->specializationConstants.raderContainer[i].stageRadix[axis->specializationConstants.raderContainer[i].numStages - 1];
						localStageSum = 0;
						for (int64_t l = (int64_t)axis->specializationConstants.raderContainer[i].numStages - 2; l >= 0; l--) {
							if ((axis->specializationConstants.raderContainer[i].stageRadix[l] & (axis->specializationConstants.raderContainer[i].stageRadix[l] - 1)) == 0) {
								for (uint64_t k = 0; k < log2(axis->specializationConstants.raderContainer[i].stageRadix[l]); k++) {
									for (uint64_t j = 0; j < localStageSize; j++) {
										tempLUT[2 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUTiFFT)] = (float)cos(j * double_PI / localStageSize / pow(2, k));
										tempLUT[2 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUTiFFT) + 1] = (float)sin(j * double_PI / localStageSize / pow(2, k));
									}
									localStageSum += localStageSize;
								}
							}
							else {
								for (uint64_t k = (axis->specializationConstants.raderContainer[i].stageRadix[l] - 1); k > 0; k--) {
									for (uint64_t j = 0; j < localStageSize; j++) {
										tempLUT[2 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUTiFFT)] = (float)cos(j * 2.0 * k / axis->specializationConstants.raderContainer[i].stageRadix[l] * double_PI / localStageSize);
										tempLUT[2 * (j + localStageSum + axis->specializationConstants.raderContainer[i].RaderRadixOffsetLUTiFFT) + 1] = (float)sin(j * 2.0 * k / axis->specializationConstants.raderContainer[i].stageRadix[l] * double_PI / localStageSize);
									}
									localStageSum += localStageSize;
								}
							}
							localStageSize *= axis->specializationConstants.raderContainer[i].stageRadix[l];
						}
						if (!axis->specializationConstants.inline_rader_kernel) {
							float* raderFFTkernel = (float*)axis->specializationConstants.raderContainer[i].raderFFTkernel;
							for (uint64_t j = 0; j < (axis->specializationConstants.raderContainer[i].prime - 1); j++) {//fix later
								tempLUT[2 * (j + axis->specializationConstants.raderContainer[i].RaderKernelOffsetLUT)] = (float)(raderFFTkernel[2 * j] / (axis->specializationConstants.raderContainer[i].prime - 1));
								tempLUT[2 * (j + axis->specializationConstants.raderContainer[i].RaderKernelOffsetLUT) + 1] = (float)(raderFFTkernel[2 * j + 1] / (axis->specializationConstants.raderContainer[i].prime - 1));
							}
						}
					}
				}
			}

			if ((axis->specializationConstants.axis_upload_id > 0) && (app->configuration.useLUT_4step == 1)) {
				for (uint64_t i = 0; i < (uint64_t)axis->specializationConstants.stageStartSize.data.i; i++) {
					for (uint64_t j = 0; j < (uint64_t)axis->specializationConstants.fftDim.data.i; j++) {
						double angle = 2 * double_PI * ((i * j) / (double)(axis->specializationConstants.stageStartSize.data.i * axis->specializationConstants.fftDim.data.i));
						tempLUT[maxStageSum * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize.data.i)] = (float)cos(angle);
						tempLUT[maxStageSum * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize.data.i) + 1] = (float)sin(angle);
					}
				}
			}
			if ((app->configuration.performDCT == 2) || (app->configuration.performDCT == 3)) {
				for (uint64_t j = 0; j < app->configuration.size[axis->specializationConstants.axis_id] / 2 + 2; j++) {
					double angle = (double_PI / 2.0 / (double)(app->configuration.size[axis->specializationConstants.axis_id])) * j;
					tempLUT[2 * axis->specializationConstants.startDCT3LUT.data.i + 2 * j] = (float)cos(angle);
					tempLUT[2 * axis->specializationConstants.startDCT3LUT.data.i + 2 * j + 1] = (float)sin(angle);
				}
			}
			if ((app->configuration.performDCT == 4) && (app->configuration.size[axis->specializationConstants.axis_id] % 2 == 0)) {
				for (uint64_t j = 0; j < app->configuration.size[axis->specializationConstants.axis_id] / 4 + 2; j++) {
					double angle = (double_PI / 2.0 / (double)(app->configuration.size[axis->specializationConstants.axis_id] / 2)) * j;
					tempLUT[2 * axis->specializationConstants.startDCT3LUT.data.i + 2 * j] = (float)cos(angle);
					tempLUT[2 * axis->specializationConstants.startDCT3LUT.data.i + 2 * j + 1] = (float)sin(angle);
				}
				for (uint64_t j = 0; j < app->configuration.size[axis->specializationConstants.axis_id] / 2; j++) {
					double angle = (-double_PI / 8.0 / (double)(app->configuration.size[axis->specializationConstants.axis_id] / 2)) * (2 * j + 1);
					tempLUT[2 * axis->specializationConstants.startDCT4LUT.data.i + 2 * j] = (float)cos(angle);
					tempLUT[2 * axis->specializationConstants.startDCT4LUT.data.i + 2 * j + 1] = (float)sin(angle);
				}
			}
			axis->referenceLUT = 0;

			if (axis->specializationConstants.reverseBluesteinMultiUpload == 1) {
				axis->bufferLUT = FFTPlan->axes[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id].bufferLUT;
#if(VKFFT_BACKEND==0)
				axis->bufferLUTDeviceMemory = FFTPlan->axes[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id].bufferLUTDeviceMemory;
#endif
				axis->bufferLUTSize = FFTPlan->axes[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id].bufferLUTSize;
				axis->referenceLUT = 1;
			}
			else {
				if ((!inverse) && (!app->configuration.makeForwardPlanOnly)) {
					axis->bufferLUT = app->localFFTPlan_inverse->axes[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id].bufferLUT;
#if(VKFFT_BACKEND==0)
					axis->bufferLUTDeviceMemory = app->localFFTPlan_inverse->axes[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id].bufferLUTDeviceMemory;
#endif
					axis->bufferLUTSize = app->localFFTPlan_inverse->axes[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id].bufferLUTSize;
					axis->referenceLUT = 1;
				}
				else {
					uint64_t checkRadixOrder = 1;
					for (uint64_t i = 0; i < axis->specializationConstants.numStages; i++)
						if (FFTPlan->axes[0][0].specializationConstants.stageRadix[i] != axis->specializationConstants.stageRadix[i]) checkRadixOrder = 0;
					if (checkRadixOrder) {
						for (uint64_t i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
							if (axis->specializationConstants.raderContainer[i].type == 0) {
								for (uint64_t k = 0; k < axis->specializationConstants.raderContainer[i].numStages; k++) {
									if (FFTPlan->axes[0][0].specializationConstants.raderContainer[i].stageRadix[k] != axis->specializationConstants.raderContainer[i].stageRadix[k]) checkRadixOrder = 0;
								}
							}
						}
					}
					if (checkRadixOrder && (axis->specializationConstants.axis_id >= 1) && (!((!axis->specializationConstants.reorderFourStep) && (FFTPlan->numAxisUploads[axis->specializationConstants.axis_id] > 1))) && ((axis->specializationConstants.fft_dim_full.data.i == FFTPlan->axes[0][0].specializationConstants.fft_dim_full.data.i) && (FFTPlan->numAxisUploads[axis->specializationConstants.axis_id] == 1) && (axis->specializationConstants.fft_dim_full.data.i < axis->specializationConstants.maxSingleSizeStrided.data.i / axis->specializationConstants.registerBoost)) && ((!app->configuration.performDCT) || (app->configuration.size[axis->specializationConstants.axis_id] == app->configuration.size[0]))) {
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
                                for (uint64_t i = 0; i < axis->specializationConstants.numStages; i++)
                                    if (FFTPlan->axes[p][0].specializationConstants.stageRadix[i] != axis->specializationConstants.stageRadix[i]) checkRadixOrder = 0;
                                if (checkRadixOrder) {
                                    for (uint64_t i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
                                        if (axis->specializationConstants.raderContainer[i].type == 0) {
                                            for (uint64_t k = 0; k < axis->specializationConstants.raderContainer[i].numStages; k++) {
                                                if (FFTPlan->axes[p][0].specializationConstants.raderContainer[i].stageRadix[k] != axis->specializationConstants.raderContainer[i].stageRadix[k]) checkRadixOrder = 0;
                                            }
                                        }
                                    }
                                }
                                if (checkRadixOrder && (axis->specializationConstants.fft_dim_full.data.i == FFTPlan->axes[p][0].specializationConstants.fft_dim_full.data.i) && ((!app->configuration.performDCT) || (app->configuration.size[axis->specializationConstants.axis_id] == app->configuration.size[p]))) {
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
			for (uint64_t i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
				app->bufferRaderUintLUTSize[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id] += axis->specializationConstants.raderContainer[i].prime * sizeof(uint32_t);
			}
			uint32_t* tempRaderUintLUT = (uint32_t*)malloc(app->bufferRaderUintLUTSize[axis->specializationConstants.axis_id][axis->specializationConstants.axis_upload_id]);
			if (!tempRaderUintLUT) {
				deleteVkFFT(app);
				return VKFFT_ERROR_MALLOC_FAILED;
			}
			uint64_t current_offset = 0;
			for (uint64_t i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
				if (axis->specializationConstants.raderContainer[i].prime > 0) {
					axis->specializationConstants.raderContainer[i].raderUintLUToffset = (int)current_offset;
					uint64_t g_pow = 1;
					tempRaderUintLUT[current_offset] = 1;
					current_offset++;
					for (uint64_t t = 0; t < axis->specializationConstants.raderContainer[i].prime - 1; t++) {
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
			uint64_t current_offset = 0;
			for (uint64_t i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
				if (axis->specializationConstants.raderContainer[i].prime > 0) {
					axis->specializationConstants.raderContainer[i].raderUintLUToffset = (int)current_offset;
					uint64_t g_pow = 1;
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

static inline VkFFTResult VkFFT_AllocateLUT_R2C(VkFFTApplication* app, VkFFTPlan* FFTPlan, VkFFTAxis* axis, uint64_t inverse) {
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
		if (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory) {
			long double double_PI = 3.14159265358979323846264338327950288419716939937510L;
			axis->bufferLUTSize = (app->configuration.size[0] / 2) * 2 * sizeof(double);
			double* tempLUT = (double*)malloc(axis->bufferLUTSize);
			if (!tempLUT) {
				deleteVkFFT(app);
				return VKFFT_ERROR_MALLOC_FAILED;
			}
			for (uint64_t i = 0; i < app->configuration.size[0] / 2; i++) {
				long double angle = double_PI * i / (app->configuration.size[0] / 2);
				tempLUT[2 * i] = (double)cos(angle);
				tempLUT[2 * i + 1] = (double)sin(angle);
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
			for (uint64_t i = 0; i < app->configuration.size[0] / 2; i++) {
				double angle = double_PI * i / (app->configuration.size[0] / 2);
				tempLUT[2 * i] = (float)cos(angle);
				tempLUT[2 * i + 1] = (float)sin(angle);
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
