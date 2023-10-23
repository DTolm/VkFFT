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
#ifndef VKFFT_AXISBLOCKSPLITTER_H
#define VKFFT_AXISBLOCKSPLITTER_H
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"

static inline VkFFTResult VkFFTSplitAxisBlock(VkFFTApplication* app, VkFFTPlan* FFTPlan, VkFFTAxis* axis, pfUINT axis_id, pfUINT axis_upload_id, pfUINT allowedSharedMemory, pfUINT allowedSharedMemoryPow2) {
	pfUINT maxBatchCoalesced = app->configuration.coalescedMemory / axis->specializationConstants.complexSize;
	axis->groupedBatch = maxBatchCoalesced;

	pfUINT maxSequenceLengthSharedMemory = allowedSharedMemory / axis->specializationConstants.complexSize;
	pfUINT maxSequenceLengthSharedMemoryPow2 = allowedSharedMemoryPow2 / axis->specializationConstants.complexSize;
	pfUINT maxSingleSizeStrided = (app->configuration.coalescedMemory > axis->specializationConstants.complexSize) ? allowedSharedMemory / (app->configuration.coalescedMemory) : allowedSharedMemory / axis->specializationConstants.complexSize;
	pfUINT maxSingleSizeStridedPow2 = (app->configuration.coalescedMemory > axis->specializationConstants.complexSize) ? allowedSharedMemoryPow2 / (app->configuration.coalescedMemory) : allowedSharedMemoryPow2 / axis->specializationConstants.complexSize;
	if (((FFTPlan->numAxisUploads[axis_id] == 1) && (axis_id == 0)) || ((axis_id == 0) && (!axis->specializationConstants.reorderFourStep) && (axis_upload_id == 0))) {
		axis->groupedBatch = (maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim.data.i > axis->groupedBatch) ? maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim.data.i : axis->groupedBatch;
	}
	else {
		axis->groupedBatch = (maxSingleSizeStrided / axis->specializationConstants.fftDim.data.i > 1) ? maxSingleSizeStrided / axis->specializationConstants.fftDim.data.i * axis->groupedBatch : axis->groupedBatch;
	}
	
	if (app->configuration.groupedBatch[axis_id])
	{
		pfUINT maxThreadNum = app->configuration.maxThreadsNum;
		axis->specializationConstants.axisSwapped = 0;
		pfUINT r2cmult = (axis->specializationConstants.mergeSequencesR2C) ? 2 : 1;
		if (axis_id == 0) {
			if (axis_upload_id == 0) {
				axis->axisBlock[0] = (((pfUINT)pfceil(axis->specializationConstants.fftDim.data.i / (double)axis->specializationConstants.min_registers_per_thread)) / axis->specializationConstants.registerBoost > 1) ? ((pfUINT)pfceil(axis->specializationConstants.fftDim.data.i / (double)axis->specializationConstants.min_registers_per_thread)) / axis->specializationConstants.registerBoost : 1;
				if (axis->specializationConstants.useRaderMult) {
					pfUINT locMaxBatchCoalesced = ((axis_id == 0) && (((axis_upload_id == 0) && ((!app->configuration.reorderFourStep) || (app->useBluesteinFFT[axis_id]))) || (axis->specializationConstants.numAxisUploads == 1))) ? 1 : maxBatchCoalesced;
					pfUINT final_rader_thread_count = 0;
					for (pfUINT i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
						if (axis->specializationConstants.raderContainer[i].type == 1) {
							pfUINT temp_rader = (pfUINT)pfceil((axis->specializationConstants.fftDim.data.i / (double)((axis->specializationConstants.rader_min_registers / 2) * 2)) / (double)((axis->specializationConstants.raderContainer[i].prime + 1) / 2));
							pfUINT active_rader = (pfUINT)pfceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)temp_rader);
							if (active_rader > 1) {
								if ((((double)active_rader - (axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)temp_rader) >= 0.5) && ((((pfUINT)pfceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)(active_rader - 1)) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2)) * locMaxBatchCoalesced) <= app->configuration.maxThreadsNum)) active_rader--;
							}
							pfUINT local_estimate_rader_threadnum = (pfUINT)pfceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)active_rader) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2);

							pfUINT temp_rader_thread_count = ((pfUINT)pfceil(axis->axisBlock[0] / (double)((axis->specializationConstants.raderContainer[i].prime + 1) / 2))) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2);
							if (temp_rader_thread_count < local_estimate_rader_threadnum) temp_rader_thread_count = local_estimate_rader_threadnum;
							if (temp_rader_thread_count > final_rader_thread_count) final_rader_thread_count = temp_rader_thread_count;
						}
					}
					axis->axisBlock[0] = final_rader_thread_count;
					if (axis->axisBlock[0] * axis->groupedBatch > maxThreadNum) axis->groupedBatch = locMaxBatchCoalesced;
				}
				if (axis->specializationConstants.useRaderFFT) {
					if (axis->axisBlock[0] < axis->specializationConstants.minRaderFFTThreadNum) axis->axisBlock[0] = axis->specializationConstants.minRaderFFTThreadNum;
				}
				if (axis->axisBlock[0] > maxThreadNum) axis->axisBlock[0] = maxThreadNum;
				if (axis->axisBlock[0] > app->configuration.maxComputeWorkGroupSize[0]) axis->axisBlock[0] = app->configuration.maxComputeWorkGroupSize[0];
				axis->axisBlock[1] = (axis->groupedBatch > app->configuration.groupedBatch[0]) ? app->configuration.groupedBatch[0] : axis->groupedBatch;
				if (axis->axisBlock[0] * axis->axisBlock[1] > maxThreadNum) {
					axis->axisBlock[1] = maxThreadNum / axis->axisBlock[0];
				}
				while ((axis->axisBlock[1] * (axis->specializationConstants.fftDim.data.i / axis->specializationConstants.registerBoost)) > maxSequenceLengthSharedMemory) axis->axisBlock[1] /= 2;
				
				axis->groupedBatch = axis->axisBlock[1];
				if (((axis->specializationConstants.fftDim.data.i % 2 == 0) || (axis->axisBlock[0] < app->configuration.numSharedBanks / 4)) && (!(((!axis->specializationConstants.reorderFourStep) || (axis->specializationConstants.useBluesteinFFT)) && (FFTPlan->numAxisUploads[0] > 1))) && (axis->axisBlock[1] > 1) && (axis->axisBlock[1] * axis->specializationConstants.fftDim.data.i < maxSequenceLengthSharedMemory) && (!((app->configuration.performZeropadding[0] || app->configuration.performZeropadding[1] || app->configuration.performZeropadding[2])))) {
					/*#if (VKFFT_BACKEND==0)
										if (((axis->specializationConstants.fftDim & (axis->specializationConstants.fftDim - 1)) != 0)) {
											pfUINT temp = axis->axisBlock[1];
											axis->axisBlock[1] = axis->axisBlock[0];
											axis->axisBlock[0] = temp;
											axis->specializationConstants.axisSwapped = 1;
										}
					#else*/
					pfUINT temp = axis->axisBlock[1];
					axis->axisBlock[1] = axis->axisBlock[0];
					axis->axisBlock[0] = temp;
					axis->specializationConstants.axisSwapped = 1;
					//#endif
				}
				axis->axisBlock[2] = 1;
				axis->axisBlock[3] = axis->specializationConstants.fftDim.data.i;
			}
			else {
				axis->axisBlock[1] = ((pfUINT)pfceil(axis->specializationConstants.fftDim.data.i / (double)axis->specializationConstants.min_registers_per_thread) / axis->specializationConstants.registerBoost > 1) ? (pfUINT)pfceil(axis->specializationConstants.fftDim.data.i / (double)axis->specializationConstants.min_registers_per_thread) / axis->specializationConstants.registerBoost : 1;
				if (axis->specializationConstants.useRaderMult) {
					pfUINT final_rader_thread_count = 0;
					for (pfUINT i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
						if (axis->specializationConstants.raderContainer[i].type == 1) {
							pfUINT temp_rader = (pfUINT)pfceil((axis->specializationConstants.fftDim.data.i / (double)((axis->specializationConstants.rader_min_registers / 2) * 2)) / (double)((axis->specializationConstants.raderContainer[i].prime + 1) / 2));
							pfUINT active_rader = (pfUINT)pfceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)temp_rader);
							if (active_rader > 1) {
								if ((((double)active_rader - (axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)temp_rader) >= 0.5) && ((((pfUINT)pfceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)(active_rader - 1)) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2)) * maxBatchCoalesced) <= app->configuration.maxThreadsNum)) active_rader--;
							}
							pfUINT local_estimate_rader_threadnum = (pfUINT)pfceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)active_rader) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2);

							pfUINT temp_rader_thread_count = ((pfUINT)pfceil(axis->axisBlock[1] / (double)((axis->specializationConstants.raderContainer[i].prime + 1) / 2))) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2);
							if (temp_rader_thread_count < local_estimate_rader_threadnum) temp_rader_thread_count = local_estimate_rader_threadnum;
							if (temp_rader_thread_count > final_rader_thread_count) final_rader_thread_count = temp_rader_thread_count;
						}
					}
					axis->axisBlock[1] = final_rader_thread_count;
					if (axis->groupedBatch * axis->axisBlock[1] > maxThreadNum) axis->groupedBatch = maxBatchCoalesced;
				}
				if (axis->specializationConstants.useRaderFFT) {
					if (axis->axisBlock[1] < axis->specializationConstants.minRaderFFTThreadNum) axis->axisBlock[1] = axis->specializationConstants.minRaderFFTThreadNum;
				}

				pfUINT scale = app->configuration.aimThreads / axis->axisBlock[1] / axis->groupedBatch;
				if ((scale > 1) && ((axis->specializationConstants.fftDim.data.i * axis->groupedBatch * scale <= maxSequenceLengthSharedMemory))) axis->groupedBatch *= scale;

				axis->axisBlock[0] = ((pfUINT)axis->specializationConstants.stageStartSize.data.i > axis->groupedBatch) ? axis->groupedBatch : (pfUINT)axis->specializationConstants.stageStartSize.data.i;
				if (app->configuration.vendorID == 0x10DE) {
					while ((axis->axisBlock[1] * axis->axisBlock[0] >= 2 * app->configuration.aimThreads) && (axis->axisBlock[0] > maxBatchCoalesced)) {
						axis->axisBlock[0] /= 2;
						if (axis->axisBlock[0] < maxBatchCoalesced) axis->axisBlock[0] = maxBatchCoalesced;
					}
				}
				if (axis->axisBlock[0] > app->configuration.maxComputeWorkGroupSize[0]) axis->axisBlock[0] = app->configuration.maxComputeWorkGroupSize[0];
				if (axis->axisBlock[0] * axis->axisBlock[1] > maxThreadNum) {
					for (pfUINT i = 1; i <= axis->axisBlock[0]; i++) {
						if ((axis->axisBlock[0] / i) * axis->axisBlock[1] <= maxThreadNum)
						{
							axis->axisBlock[0] /= i;
							i = axis->axisBlock[0] + 1;
						}

					}
				}
				axis->axisBlock[2] = 1;
				axis->axisBlock[3] = axis->specializationConstants.fftDim.data.i;
				axis->groupedBatch = axis->axisBlock[0];
			}

		}
		if (axis_id >= 1) {

			axis->axisBlock[1] = ((pfUINT)pfceil(axis->specializationConstants.fftDim.data.i / (double)axis->specializationConstants.min_registers_per_thread) / axis->specializationConstants.registerBoost > 1) ? ((pfUINT)pfceil(axis->specializationConstants.fftDim.data.i / (double)axis->specializationConstants.min_registers_per_thread)) / axis->specializationConstants.registerBoost : 1;
			if (axis->specializationConstants.useRaderMult) {
				pfUINT final_rader_thread_count = 0;
				for (pfUINT i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
					if (axis->specializationConstants.raderContainer[i].type == 1) {
						pfUINT temp_rader = (pfUINT)pfceil((axis->specializationConstants.fftDim.data.i / (double)((axis->specializationConstants.rader_min_registers / 2) * 2)) / (double)((axis->specializationConstants.raderContainer[i].prime + 1) / 2));
						pfUINT active_rader = (pfUINT)pfceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)temp_rader);
						if (active_rader > 1) {
							if ((((double)active_rader - (axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)temp_rader) >= 0.5) && ((((pfUINT)pfceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)(active_rader - 1)) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2)) * maxBatchCoalesced) <= app->configuration.maxThreadsNum)) active_rader--;
						}
						pfUINT local_estimate_rader_threadnum = (pfUINT)pfceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)active_rader) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2);

						pfUINT temp_rader_thread_count = ((pfUINT)pfceil(axis->axisBlock[1] / (double)((axis->specializationConstants.raderContainer[i].prime + 1) / 2))) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2);
						if (temp_rader_thread_count < local_estimate_rader_threadnum) temp_rader_thread_count = local_estimate_rader_threadnum;
						if (temp_rader_thread_count > final_rader_thread_count) final_rader_thread_count = temp_rader_thread_count;
					}
				}
				axis->axisBlock[1] = final_rader_thread_count;
				if (axis->groupedBatch * axis->axisBlock[1] > maxThreadNum) axis->groupedBatch = maxBatchCoalesced;
			}
			if (axis->specializationConstants.useRaderFFT) {
				if (axis->axisBlock[1] < axis->specializationConstants.minRaderFFTThreadNum) axis->axisBlock[1] = axis->specializationConstants.minRaderFFTThreadNum;
			}

			axis->axisBlock[0] = (FFTPlan->actualFFTSizePerAxis[axis_id][0] > axis->groupedBatch) ? axis->groupedBatch : FFTPlan->actualFFTSizePerAxis[axis_id][0];
			
			axis->axisBlock[0] = (axis->axisBlock[0] > app->configuration.groupedBatch[1]) ? app->configuration.groupedBatch[axis_id] : axis->axisBlock[0];

			
			if (axis->axisBlock[0] > app->configuration.maxComputeWorkGroupSize[0]) axis->axisBlock[0] = app->configuration.maxComputeWorkGroupSize[0];
			if (axis->axisBlock[0] * axis->axisBlock[1] > maxThreadNum) {
				axis->axisBlock[0] = maxThreadNum / axis->axisBlock[0];
			}
			axis->axisBlock[2] = 1;
			axis->axisBlock[3] = axis->specializationConstants.fftDim.data.i;
			axis->groupedBatch = axis->axisBlock[0];

		}
		
		return VKFFT_SUCCESS;
	}
	/*if ((FFTPlan->actualFFTSizePerAxis[axis_id][0] < 4096) && (FFTPlan->actualFFTSizePerAxis[axis_id][1] < 512) && (FFTPlan->actualFFTSizePerAxis[axis_id][2] == 1)) {
		if (app->configuration.sharedMemorySize / axis->specializationConstants.fftDim >= app->configuration.coalescedMemory) {
			if (1024 / axis->specializationConstants.fftDim < maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim) {
				if (1024 / axis->specializationConstants.fftDim > axis->groupedBatch)
					axis->groupedBatch = 1024 / axis->specializationConstants.fftDim;
				else
					axis->groupedBatch = maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim;
			}
		}
	}
	else {
		axis->groupedBatch = (app->configuration.sharedMemorySize / axis->specializationConstants.fftDim >= app->configuration.coalescedMemory) ? maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim : axis->groupedBatch;
	}*/
	//if (axis->groupedBatch * (pfUINT)pfceil(axis->specializationConstants.fftDim / 8.0) < app->configuration.warpSize) axis->groupedBatch = app->configuration.warpSize / (pfUINT)pfceil(axis->specializationConstants.fftDim / 8.0);
	//axis->groupedBatch = (app->configuration.sharedMemorySize / axis->specializationConstants.fftDim >= app->configuration.coalescedMemory) ? maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim : axis->groupedBatch;
	//axis->groupedBatch = 8;
	//shared memory bank conflict resolve
//#if(VKFFT_BACKEND!=2)//for some reason, hip doesn't get performance increase from having variable shared memory strides.
	if (app->configuration.vendorID == 0x10DE) {
		if (FFTPlan->numAxisUploads[axis_id] == 2) {
			if ((axis_upload_id > 0) || (axis->specializationConstants.fftDim.data.i <= 512)) {
				if ((pfUINT)(axis->specializationConstants.fftDim.data.i * (64 / axis->specializationConstants.complexSize)) <= maxSequenceLengthSharedMemory) {
					axis->groupedBatch = 64 / axis->specializationConstants.complexSize;
					maxBatchCoalesced = 64 / axis->specializationConstants.complexSize;
				}
				if ((pfUINT)(axis->specializationConstants.fftDim.data.i * (128 / axis->specializationConstants.complexSize)) <= maxSequenceLengthSharedMemory) {
					axis->groupedBatch = 128 / axis->specializationConstants.complexSize;
					maxBatchCoalesced = 128 / axis->specializationConstants.complexSize;
				}
			}
		}
		//#endif
		if (FFTPlan->numAxisUploads[axis_id] == 3) {
			if ((pfUINT)(axis->specializationConstants.fftDim.data.i * (64 / axis->specializationConstants.complexSize)) <= maxSequenceLengthSharedMemory) {
				axis->groupedBatch = 64 / axis->specializationConstants.complexSize;
				maxBatchCoalesced = 64 / axis->specializationConstants.complexSize;
			}
			if ((pfUINT)(axis->specializationConstants.fftDim.data.i * (128 / axis->specializationConstants.complexSize)) <= maxSequenceLengthSharedMemory) {
				axis->groupedBatch = 128 / axis->specializationConstants.complexSize;
				maxBatchCoalesced = 128 / axis->specializationConstants.complexSize;
			}
		}
	}
	else {
		if ((FFTPlan->numAxisUploads[axis_id] == 2) && (axis_upload_id == 0) && (axis->specializationConstants.fftDim.data.i * maxBatchCoalesced <= maxSequenceLengthSharedMemory)) {
			axis->groupedBatch = (pfUINT)pfceil(axis->groupedBatch / 2.0);
		}
		//#endif
		if ((FFTPlan->numAxisUploads[axis_id] == 3) && (axis_upload_id == 0) && ((pfUINT)axis->specializationConstants.fftDim.data.i < maxSequenceLengthSharedMemory / (2 * axis->specializationConstants.complexSize))) {
			axis->groupedBatch = (pfUINT)pfceil(axis->groupedBatch / 2.0);
		}
	}
	if (axis->groupedBatch < maxBatchCoalesced) axis->groupedBatch = maxBatchCoalesced;
	axis->groupedBatch = (axis->groupedBatch / maxBatchCoalesced) * maxBatchCoalesced;
	//half bandiwdth technique
	if (!((axis_id == 0) && (FFTPlan->numAxisUploads[axis_id] == 1)) && !((axis_id == 0) && (axis_upload_id == 0) && (!axis->specializationConstants.reorderFourStep)) && ((pfUINT)axis->specializationConstants.fftDim.data.i > maxSingleSizeStrided)) {
		axis->groupedBatch = maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim.data.i;
		if (axis->groupedBatch == 0) axis->groupedBatch = 1;
	}

	if ((app->configuration.halfThreads) && (axis->groupedBatch * axis->specializationConstants.fftDim.data.i * axis->specializationConstants.complexSize >= app->configuration.sharedMemorySize))
		axis->groupedBatch = (pfUINT)pfceil(axis->groupedBatch / 2.0);
	if (axis->groupedBatch > app->configuration.warpSize) axis->groupedBatch = (axis->groupedBatch / app->configuration.warpSize) * app->configuration.warpSize;
	if (axis->groupedBatch > 2 * maxBatchCoalesced) axis->groupedBatch = (axis->groupedBatch / (2 * maxBatchCoalesced)) * (2 * maxBatchCoalesced);
	if (axis->groupedBatch > 4 * maxBatchCoalesced) axis->groupedBatch = (axis->groupedBatch / (4 * maxBatchCoalesced)) * (4 * maxBatchCoalesced);
	//pfUINT maxThreadNum = (axis_id) ? (maxSingleSizeStrided * app->configuration.coalescedMemory / axis->specializationConstants.complexSize) / (axis->specializationConstants.min_registers_per_thread * axis->specializationConstants.registerBoost) : maxSequenceLengthSharedMemory / (axis->specializationConstants.min_registers_per_thread * axis->specializationConstants.registerBoost);
	//if (maxThreadNum > app->configuration.maxThreadsNum) maxThreadNum = app->configuration.maxThreadsNum;
	pfUINT maxThreadNum = app->configuration.maxThreadsNum;
	axis->specializationConstants.axisSwapped = 0;
	pfUINT r2cmult = (axis->specializationConstants.mergeSequencesR2C) ? 2 : 1;
	if (axis_id == 0) {
		if (axis_upload_id == 0) {
			axis->axisBlock[0] = (((pfUINT)pfceil(axis->specializationConstants.fftDim.data.i / (double)axis->specializationConstants.min_registers_per_thread)) / axis->specializationConstants.registerBoost > 1) ? ((pfUINT)pfceil(axis->specializationConstants.fftDim.data.i / (double)axis->specializationConstants.min_registers_per_thread)) / axis->specializationConstants.registerBoost : 1;
			if (axis->specializationConstants.useRaderMult) {
				pfUINT locMaxBatchCoalesced = ((axis_id == 0) && (((axis_upload_id == 0) && ((!app->configuration.reorderFourStep) || (app->useBluesteinFFT[axis_id]))) || (axis->specializationConstants.numAxisUploads == 1))) ? 1 : maxBatchCoalesced;
				pfUINT final_rader_thread_count = 0;
				for (pfUINT i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
					if (axis->specializationConstants.raderContainer[i].type == 1) {
						pfUINT temp_rader = (pfUINT)pfceil((axis->specializationConstants.fftDim.data.i / (double)((axis->specializationConstants.rader_min_registers / 2) * 2)) / (double)((axis->specializationConstants.raderContainer[i].prime + 1) / 2));
						pfUINT active_rader = (pfUINT)pfceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)temp_rader);
						if (active_rader > 1) {
							if ((((double)active_rader - (axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)temp_rader) >= 0.5) && ((((pfUINT)pfceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)(active_rader - 1)) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2)) * locMaxBatchCoalesced) <= app->configuration.maxThreadsNum)) active_rader--;
						}
						pfUINT local_estimate_rader_threadnum = (pfUINT)pfceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)active_rader) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2);

						pfUINT temp_rader_thread_count = ((pfUINT)pfceil(axis->axisBlock[0] / (double)((axis->specializationConstants.raderContainer[i].prime + 1) / 2))) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2);
						if (temp_rader_thread_count < local_estimate_rader_threadnum) temp_rader_thread_count = local_estimate_rader_threadnum;
						if (temp_rader_thread_count > final_rader_thread_count) final_rader_thread_count = temp_rader_thread_count;
					}
				}
				axis->axisBlock[0] = final_rader_thread_count;
				if (axis->axisBlock[0] * axis->groupedBatch > maxThreadNum) axis->groupedBatch = locMaxBatchCoalesced;
			}
			if (axis->specializationConstants.useRaderFFT) {
				if (axis->axisBlock[0] < axis->specializationConstants.minRaderFFTThreadNum) axis->axisBlock[0] = axis->specializationConstants.minRaderFFTThreadNum;
			}
			if (axis->axisBlock[0] > maxThreadNum) axis->axisBlock[0] = maxThreadNum;
			if (axis->axisBlock[0] > app->configuration.maxComputeWorkGroupSize[0]) axis->axisBlock[0] = app->configuration.maxComputeWorkGroupSize[0];
			if (axis->specializationConstants.reorderFourStep && (FFTPlan->numAxisUploads[axis_id] > 1))
				axis->axisBlock[1] = axis->groupedBatch;
			else {
				//axis->axisBlock[1] = (axis->axisBlock[0] < app->configuration.warpSize) ? app->configuration.warpSize / axis->axisBlock[0] : 1;
				pfUINT estimate_batch = (((axis->axisBlock[0] / app->configuration.warpSize) == 1) && ((axis->axisBlock[0] / (double)app->configuration.warpSize) < 1.5)) ? app->configuration.aimThreads / app->configuration.warpSize : app->configuration.aimThreads / axis->axisBlock[0];
				if (estimate_batch == 0) estimate_batch = 1;
				axis->axisBlock[1] = ((axis->axisBlock[0] < app->configuration.aimThreads) && ((axis->axisBlock[0] < app->configuration.warpSize) || (axis->specializationConstants.useRader))) ? estimate_batch : 1;
			}

			pfUINT currentAxisBlock1 = axis->axisBlock[1];
			for (pfUINT i = currentAxisBlock1; i < 2 * currentAxisBlock1; i++) {
				if (((FFTPlan->numAxisUploads[0] > 1) && (!(((FFTPlan->actualFFTSizePerAxis[axis_id][0] / axis->specializationConstants.fftDim.data.i) % axis->axisBlock[1]) == 0))) || ((FFTPlan->numAxisUploads[0] == 1) && (!(((FFTPlan->actualFFTSizePerAxis[axis_id][1] / r2cmult) % axis->axisBlock[1]) == 0)))) {
					if (i * axis->specializationConstants.fftDim.data.i * axis->specializationConstants.complexSize <= allowedSharedMemory) axis->axisBlock[1] = i;
					i = 2 * currentAxisBlock1;
				}
			}
			if (((axis->specializationConstants.fftDim.data.i % 2 == 0) || (axis->axisBlock[0] < app->configuration.numSharedBanks / 4)) && (!(((!axis->specializationConstants.reorderFourStep) || (axis->specializationConstants.useBluesteinFFT)) && (FFTPlan->numAxisUploads[0] > 1))) && (axis->axisBlock[1] > 1) && (axis->axisBlock[1] * axis->specializationConstants.fftDim.data.i < maxSequenceLengthSharedMemoryPow2) && (!((app->configuration.performZeropadding[0] || app->configuration.performZeropadding[1] || app->configuration.performZeropadding[2])))) {
				//we plan to swap - this reduces bank conflicts
				axis->axisBlock[1] = (pfUINT)pow(2, (pfUINT)pfceil(log2((double)axis->axisBlock[1])));
			}
			if ((FFTPlan->numAxisUploads[0] > 1) && ((pfUINT)pfceil(FFTPlan->actualFFTSizePerAxis[axis_id][0] / axis->specializationConstants.fftDim.data.i) < axis->axisBlock[1])) axis->axisBlock[1] = (pfUINT)pfceil(FFTPlan->actualFFTSizePerAxis[axis_id][0] / axis->specializationConstants.fftDim.data.i);
			if ((axis->specializationConstants.mergeSequencesR2C != 0) && (axis->specializationConstants.fftDim.data.i * axis->axisBlock[1] >= maxSequenceLengthSharedMemory)) {
				axis->specializationConstants.mergeSequencesR2C = 0;
				/*if ((!inverse) && (axis_id == 0) && (axis_upload_id == 0) && (!(app->configuration.isInputFormatted))) {
					axis->specializationConstants.inputStride[1] /= 2;
					axis->specializationConstants.inputStride[2] /= 2;
					axis->specializationConstants.inputStride[3] /= 2;
					axis->specializationConstants.inputStride[4] /= 2;
				}
				if ((inverse) && (axis_id == 0) && (axis_upload_id == 0) && (!((app->configuration.isInputFormatted) && (app->configuration.inverseReturnToInputBuffer))) && (!app->configuration.isOutputFormatted)) {
					axis->specializationConstants.outputStride[1] /= 2;
					axis->specializationConstants.outputStride[2] /= 2;
					axis->specializationConstants.outputStride[3] /= 2;
					axis->specializationConstants.outputStride[4] /= 2;
				}*/
				r2cmult = 1;
			}
			if ((FFTPlan->numAxisUploads[0] == 1) && ((pfUINT)pfceil(FFTPlan->actualFFTSizePerAxis[axis_id][1] / (double)r2cmult) < axis->axisBlock[1])) axis->axisBlock[1] = (pfUINT)pfceil(FFTPlan->actualFFTSizePerAxis[axis_id][1] / (double)r2cmult);
			if (app->configuration.vendorID == 0x10DE) {
				while ((axis->axisBlock[1] * axis->axisBlock[0] >= 2 * app->configuration.aimThreads) && (axis->axisBlock[1] > maxBatchCoalesced)) {
					axis->axisBlock[1] /= 2;
					if (axis->axisBlock[1] < maxBatchCoalesced) axis->axisBlock[1] = maxBatchCoalesced;
				}
			}
			if (axis->axisBlock[1] > app->configuration.maxComputeWorkGroupSize[1]) axis->axisBlock[1] = app->configuration.maxComputeWorkGroupSize[1];
			//if (axis->axisBlock[0] * axis->axisBlock[1] > app->configuration.maxThreadsNum) axis->axisBlock[1] /= 2;
			if (axis->axisBlock[0] * axis->axisBlock[1] > maxThreadNum) {
				for (pfUINT i = 1; i <= axis->axisBlock[1]; i++) {
					if ((axis->axisBlock[1] / i) * axis->axisBlock[0] <= maxThreadNum)
					{
						axis->axisBlock[1] /= i;
						i = axis->axisBlock[1] + 1;
					}

				}
			}
			while ((axis->axisBlock[1] * (axis->specializationConstants.fftDim.data.i / axis->specializationConstants.registerBoost)) > maxSequenceLengthSharedMemory) axis->axisBlock[1] /= 2;
			axis->groupedBatch = axis->axisBlock[1];
			if (((axis->specializationConstants.fftDim.data.i % 2 == 0) || (axis->axisBlock[0] < app->configuration.numSharedBanks / 4)) && (!(((!axis->specializationConstants.reorderFourStep) || (axis->specializationConstants.useBluesteinFFT)) && (FFTPlan->numAxisUploads[0] > 1))) && (axis->axisBlock[1] > 1) && (axis->axisBlock[1] * axis->specializationConstants.fftDim.data.i < maxSequenceLengthSharedMemory) && (!((app->configuration.performZeropadding[0] || app->configuration.performZeropadding[1] || app->configuration.performZeropadding[2])))) {
				/*#if (VKFFT_BACKEND==0)
									if (((axis->specializationConstants.fftDim & (axis->specializationConstants.fftDim - 1)) != 0)) {
										pfUINT temp = axis->axisBlock[1];
										axis->axisBlock[1] = axis->axisBlock[0];
										axis->axisBlock[0] = temp;
										axis->specializationConstants.axisSwapped = 1;
									}
				#else*/
				pfUINT temp = axis->axisBlock[1];
				axis->axisBlock[1] = axis->axisBlock[0];
				axis->axisBlock[0] = temp;
				axis->specializationConstants.axisSwapped = 1;
				//#endif
			}
			axis->axisBlock[2] = 1;
			axis->axisBlock[3] = axis->specializationConstants.fftDim.data.i;
		}
		else {
			axis->axisBlock[1] = ((pfUINT)pfceil(axis->specializationConstants.fftDim.data.i / (double)axis->specializationConstants.min_registers_per_thread) / axis->specializationConstants.registerBoost > 1) ? (pfUINT)pfceil(axis->specializationConstants.fftDim.data.i / (double)axis->specializationConstants.min_registers_per_thread) / axis->specializationConstants.registerBoost : 1;
			if (axis->specializationConstants.useRaderMult) {
				pfUINT final_rader_thread_count = 0;
				for (pfUINT i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
					if (axis->specializationConstants.raderContainer[i].type == 1) {
						pfUINT temp_rader = (pfUINT)pfceil((axis->specializationConstants.fftDim.data.i / (double)((axis->specializationConstants.rader_min_registers / 2) * 2)) / (double)((axis->specializationConstants.raderContainer[i].prime + 1) / 2));
						pfUINT active_rader = (pfUINT)pfceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)temp_rader);
						if (active_rader > 1) {
							if ((((double)active_rader - (axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)temp_rader) >= 0.5) && ((((pfUINT)pfceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)(active_rader - 1)) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2)) * maxBatchCoalesced) <= app->configuration.maxThreadsNum)) active_rader--;
						}
						pfUINT local_estimate_rader_threadnum = (pfUINT)pfceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)active_rader) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2);

						pfUINT temp_rader_thread_count = ((pfUINT)pfceil(axis->axisBlock[1] / (double)((axis->specializationConstants.raderContainer[i].prime + 1) / 2))) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2);
						if (temp_rader_thread_count < local_estimate_rader_threadnum) temp_rader_thread_count = local_estimate_rader_threadnum;
						if (temp_rader_thread_count > final_rader_thread_count) final_rader_thread_count = temp_rader_thread_count;
					}
				}
				axis->axisBlock[1] = final_rader_thread_count;
				if (axis->groupedBatch * axis->axisBlock[1] > maxThreadNum) axis->groupedBatch = maxBatchCoalesced;
			}
			if (axis->specializationConstants.useRaderFFT) {
				if (axis->axisBlock[1] < axis->specializationConstants.minRaderFFTThreadNum) axis->axisBlock[1] = axis->specializationConstants.minRaderFFTThreadNum;
			}

			pfUINT scale = app->configuration.aimThreads / axis->axisBlock[1] / axis->groupedBatch;
			if ((scale > 1) && ((axis->specializationConstants.fftDim.data.i * axis->groupedBatch * scale <= maxSequenceLengthSharedMemory))) axis->groupedBatch *= scale;

			axis->axisBlock[0] = ((pfUINT)axis->specializationConstants.stageStartSize.data.i > axis->groupedBatch) ? axis->groupedBatch : axis->specializationConstants.stageStartSize.data.i;
			if (app->configuration.vendorID == 0x10DE) {
				while ((axis->axisBlock[1] * axis->axisBlock[0] >= 2 * app->configuration.aimThreads) && (axis->axisBlock[0] > maxBatchCoalesced)) {
					axis->axisBlock[0] /= 2;
					if (axis->axisBlock[0] < maxBatchCoalesced) axis->axisBlock[0] = maxBatchCoalesced;
				}
			}
			if (axis->axisBlock[0] > app->configuration.maxComputeWorkGroupSize[0]) axis->axisBlock[0] = app->configuration.maxComputeWorkGroupSize[0];
			if (axis->axisBlock[0] * axis->axisBlock[1] > maxThreadNum) {
				for (pfUINT i = 1; i <= axis->axisBlock[0]; i++) {
					if ((axis->axisBlock[0] / i) * axis->axisBlock[1] <= maxThreadNum)
					{
						axis->axisBlock[0] /= i;
						i = axis->axisBlock[0] + 1;
					}

				}
			}
			axis->axisBlock[2] = 1;
			axis->axisBlock[3] = axis->specializationConstants.fftDim.data.i;
			axis->groupedBatch = axis->axisBlock[0];
		}

	}
	if (axis_id >= 1) {

		axis->axisBlock[1] = ((pfUINT)pfceil(axis->specializationConstants.fftDim.data.i / (double)axis->specializationConstants.min_registers_per_thread) / axis->specializationConstants.registerBoost > 1) ? ((pfUINT)pfceil(axis->specializationConstants.fftDim.data.i / (double)axis->specializationConstants.min_registers_per_thread)) / axis->specializationConstants.registerBoost : 1;
		if (axis->specializationConstants.useRaderMult) {
			pfUINT final_rader_thread_count = 0;
			for (pfUINT i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
				if (axis->specializationConstants.raderContainer[i].type == 1) {
					pfUINT temp_rader = (pfUINT)pfceil((axis->specializationConstants.fftDim.data.i / (double)((axis->specializationConstants.rader_min_registers / 2) * 2)) / (double)((axis->specializationConstants.raderContainer[i].prime + 1) / 2));
					pfUINT active_rader = (pfUINT)pfceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)temp_rader);
					if (active_rader > 1) {
						if ((((double)active_rader - (axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)temp_rader) >= 0.5) && ((((pfUINT)pfceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)(active_rader - 1)) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2)) * maxBatchCoalesced) <= app->configuration.maxThreadsNum)) active_rader--;
					}
					pfUINT local_estimate_rader_threadnum = (pfUINT)pfceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)active_rader) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2);

					pfUINT temp_rader_thread_count = ((pfUINT)pfceil(axis->axisBlock[1] / (double)((axis->specializationConstants.raderContainer[i].prime + 1) / 2))) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2);
					if (temp_rader_thread_count < local_estimate_rader_threadnum) temp_rader_thread_count = local_estimate_rader_threadnum;
					if (temp_rader_thread_count > final_rader_thread_count) final_rader_thread_count = temp_rader_thread_count;
				}
			}
			axis->axisBlock[1] = final_rader_thread_count;
			if (axis->groupedBatch * axis->axisBlock[1] > maxThreadNum) axis->groupedBatch = maxBatchCoalesced;
		}
		if (axis->specializationConstants.useRaderFFT) {
			if (axis->axisBlock[1] < axis->specializationConstants.minRaderFFTThreadNum) axis->axisBlock[1] = axis->specializationConstants.minRaderFFTThreadNum;
		}

		axis->axisBlock[0] = (FFTPlan->actualFFTSizePerAxis[axis_id][0] > axis->groupedBatch) ? axis->groupedBatch : FFTPlan->actualFFTSizePerAxis[axis_id][0];
		if (app->configuration.vendorID == 0x10DE) {
			while ((axis->axisBlock[1] * axis->axisBlock[0] >= 2 * app->configuration.aimThreads) && (axis->axisBlock[0] > maxBatchCoalesced)) {
				axis->axisBlock[0] /= 2;
				if (axis->axisBlock[0] < maxBatchCoalesced) axis->axisBlock[0] = maxBatchCoalesced;
			}
		}
		if (axis->axisBlock[0] > app->configuration.maxComputeWorkGroupSize[0]) axis->axisBlock[0] = app->configuration.maxComputeWorkGroupSize[0];
		if (axis->axisBlock[0] * axis->axisBlock[1] > maxThreadNum) {
			for (pfUINT i = 1; i <= axis->axisBlock[0]; i++) {
				if ((axis->axisBlock[0] / i) * axis->axisBlock[1] <= maxThreadNum)
				{
					axis->axisBlock[0] /= i;
					i = axis->axisBlock[0] + 1;
				}

			}
		}
		axis->axisBlock[2] = 1;
		axis->axisBlock[3] = axis->specializationConstants.fftDim.data.i;
		axis->groupedBatch = axis->axisBlock[0];

	}
	return VKFFT_SUCCESS;
}
#endif
