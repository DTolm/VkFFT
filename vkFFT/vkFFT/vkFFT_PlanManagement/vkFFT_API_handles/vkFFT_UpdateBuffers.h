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
#ifndef VKFFT_UPDATEBUFFERS_H
#define VKFFT_UPDATEBUFFERS_H
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT/vkFFT_AppManagement/vkFFT_DeleteApp.h"

static inline VkFFTResult VkFFTConfigureDescriptors(VkFFTApplication* app, VkFFTPlan* FFTPlan, VkFFTAxis* axis, pfUINT axis_id, pfUINT axis_upload_id, pfUINT inverse) {
	pfUINT initPageSize = -1;
	pfUINT locBufferNum = 1;
	pfUINT locBufferSize = -1;
	if ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (app->configuration.isInputFormatted) && (!axis->specializationConstants.reverseBluesteinMultiUpload) && (
		((axis_id == app->firstAxis) && (!inverse))
		|| ((axis_id == app->lastAxis) && (inverse) && (!((axis_id == 0) && (axis->specializationConstants.performR2CmultiUpload))) && (!app->configuration.performConvolution) && (!app->configuration.inverseReturnToInputBuffer)))
		) {
		pfUINT totalSize = 0;
		pfUINT locPageSize = initPageSize;
		locBufferNum = app->configuration.inputBufferNum;
		if (app->configuration.inputBufferSize) {
			locBufferSize = app->configuration.inputBufferSize[0];
			for (pfUINT i = 0; i < app->configuration.inputBufferNum; i++) {
				totalSize += app->configuration.inputBufferSize[i];
				if (app->configuration.inputBufferSize[i] < locPageSize) locPageSize = app->configuration.inputBufferSize[i];
			}
		}
		axis->specializationConstants.inputBufferBlockSize = (locBufferNum == 1) ? locBufferSize : locPageSize;
		axis->specializationConstants.inputBufferBlockNum = (locBufferNum == 1) ? 1 : (pfUINT)pfceil(totalSize / (double)(axis->specializationConstants.inputBufferBlockSize));
		//if (axis->specializationConstants.inputBufferBlockNum == 1) axis->specializationConstants.inputBufferBlockSize = totalSize / axis->specializationConstants.complexSize;

	}
	else {
		if ((axis_upload_id == 0) && (app->configuration.numberKernels > 1) && (inverse) && (!app->configuration.performConvolution)) {
			pfUINT totalSize = 0;
			pfUINT locPageSize = initPageSize;
			locBufferNum = app->configuration.outputBufferNum;
			if (app->configuration.outputBufferSize) {
				locBufferSize = app->configuration.outputBufferSize[0];
				for (pfUINT i = 0; i < app->configuration.outputBufferNum; i++) {
					totalSize += app->configuration.outputBufferSize[i];
					if (app->configuration.outputBufferSize[i] < locPageSize) locPageSize = app->configuration.outputBufferSize[i];
				}
			}
			axis->specializationConstants.inputBufferBlockSize = (locBufferNum == 1) ? locBufferSize : locPageSize;
			axis->specializationConstants.inputBufferBlockNum = (locBufferNum == 1) ? 1 : (pfUINT)pfceil(totalSize / (double)(axis->specializationConstants.inputBufferBlockSize));
			//if (axis->specializationConstants.inputBufferBlockNum == 1) axis->specializationConstants.outputBufferBlockSize = totalSize / axis->specializationConstants.complexSize;

		}
		else {
			pfUINT totalSize = 0;
			pfUINT locPageSize = initPageSize;
			if (((axis->specializationConstants.reorderFourStep == 1) || (app->useBluesteinFFT[axis_id])) && (FFTPlan->numAxisUploads[axis_id] > 1)) {
				if ((((axis->specializationConstants.reorderFourStep == 1) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1)) || (app->useBluesteinFFT[axis_id] && (axis->specializationConstants.reverseBluesteinMultiUpload == 0) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1))) && (!((axis_id == 0) && (axis->specializationConstants.performR2CmultiUpload) && (axis->specializationConstants.reorderFourStep == 1) && (inverse == 1)))) {
                    locBufferNum = app->configuration.bufferNum;
					if (app->configuration.bufferSize) {
						locBufferSize = app->configuration.bufferSize[0];
						for (pfUINT i = 0; i < app->configuration.bufferNum; i++) {
							totalSize += app->configuration.bufferSize[i];
							if (app->configuration.bufferSize[i] < locPageSize) locPageSize = app->configuration.bufferSize[i];

						}
					}
				}
				else {
					locBufferNum = app->configuration.tempBufferNum;
					if (app->configuration.tempBufferSize) {
						locBufferSize = app->configuration.tempBufferSize[0];
						for (pfUINT i = 0; i < app->configuration.tempBufferNum; i++) {
							totalSize += app->configuration.tempBufferSize[i];
							if (app->configuration.tempBufferSize[i] < locPageSize) locPageSize = app->configuration.tempBufferSize[i];

						}
					}
				}
			}
			else {
				locBufferNum = app->configuration.bufferNum;
				if (app->configuration.bufferSize) {
					locBufferSize = app->configuration.bufferSize[0];
					for (pfUINT i = 0; i < app->configuration.bufferNum; i++) {
						totalSize += app->configuration.bufferSize[i];
						if (app->configuration.bufferSize[i] < locPageSize) locPageSize = app->configuration.bufferSize[i];

					}
				}
			}

			axis->specializationConstants.inputBufferBlockSize = (locBufferNum == 1) ? locBufferSize : locPageSize;
			axis->specializationConstants.inputBufferBlockNum = (locBufferNum == 1) ? 1 : (pfUINT)pfceil(totalSize / (double)(axis->specializationConstants.inputBufferBlockSize));
			//if (axis->specializationConstants.inputBufferBlockNum == 1) axis->specializationConstants.inputBufferBlockSize = totalSize / axis->specializationConstants.complexSize;

		}
	}
	initPageSize = -1;
	locBufferNum = 1;
	locBufferSize = -1;
	if (((axis_upload_id == 0) && (!app->useBluesteinFFT[axis_id]) && (app->configuration.isOutputFormatted && (
		((axis_id == app->firstAxis) && (inverse))
		|| ((axis_id == app->lastAxis) && (!inverse) && (!app->configuration.performConvolution))
		|| ((axis_id == app->firstAxis) && (app->configuration.performConvolution) && (app->configuration.FFTdim == 1)))
		)) ||
		((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (app->useBluesteinFFT[axis_id]) && (axis->specializationConstants.reverseBluesteinMultiUpload || (FFTPlan->numAxisUploads[axis_id] == 1)) && (app->configuration.isOutputFormatted && (
			((axis_id == app->firstAxis) && (inverse))
			|| ((axis_id == app->lastAxis) && (!inverse) && (!app->configuration.performConvolution)))
			)) ||
		((app->configuration.numberKernels > 1) && (
			(inverse)
			|| (axis_id == app->lastAxis)))
		) {
		pfUINT totalSize = 0;
		pfUINT locPageSize = initPageSize;
		locBufferNum = app->configuration.outputBufferNum;
		if (app->configuration.outputBufferSize) {
			locBufferSize = app->configuration.outputBufferSize[0];
			for (pfUINT i = 0; i < app->configuration.outputBufferNum; i++) {
				totalSize += app->configuration.outputBufferSize[i];
				if (app->configuration.outputBufferSize[i] < locPageSize) locPageSize = app->configuration.outputBufferSize[i];
			}
		}
		axis->specializationConstants.outputBufferBlockSize = (locBufferNum == 1) ? locBufferSize : locPageSize;
		axis->specializationConstants.outputBufferBlockNum = (locBufferNum == 1) ? 1 : (pfUINT)pfceil(totalSize / (double)(axis->specializationConstants.outputBufferBlockSize));
		//if (axis->specializationConstants.outputBufferBlockNum == 1) axis->specializationConstants.outputBufferBlockSize = totalSize / axis->specializationConstants.complexSize;

	}
	else {
		pfUINT totalSize = 0;
		pfUINT locPageSize = initPageSize;
        if (((axis->specializationConstants.reorderFourStep == 1) || (app->useBluesteinFFT[axis_id])) && (FFTPlan->numAxisUploads[axis_id] > 1)) {
            if ((inverse) && (axis_id == app->firstAxis) && (
                ((axis_upload_id == 0) && (app->configuration.isInputFormatted) && (app->configuration.inverseReturnToInputBuffer) && (!app->useBluesteinFFT[axis_id]))
                || ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (app->configuration.isInputFormatted) && (axis->specializationConstants.actualInverse) && (app->configuration.inverseReturnToInputBuffer) && (app->useBluesteinFFT[axis_id]) && (axis->specializationConstants.reverseBluesteinMultiUpload || (FFTPlan->numAxisUploads[axis_id] == 1))))
                ) {
                    locBufferNum = app->configuration.inputBufferNum;
                    if (app->configuration.inputBufferSize) {
                        locBufferSize = app->configuration.inputBufferSize[0];
                        for (pfUINT i = 0; i < app->configuration.inputBufferNum; i++) {
                            totalSize += app->configuration.inputBufferSize[i];
                            if (app->configuration.inputBufferSize[i] < locPageSize) locPageSize = app->configuration.inputBufferSize[i];
                        }
                    } 
                }
                else{
                    if (((axis->specializationConstants.reorderFourStep == 1) && (axis_upload_id > 0)) || (app->useBluesteinFFT[axis_id] && (!((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (axis->specializationConstants.reverseBluesteinMultiUpload == 1))))) {
								
                        locBufferNum = app->configuration.tempBufferNum;
                        if (app->configuration.tempBufferSize) {
                            locBufferSize = app->configuration.tempBufferSize[0];
                            for (pfUINT i = 0; i < app->configuration.tempBufferNum; i++) {
                                totalSize += app->configuration.tempBufferSize[i];
                                if (app->configuration.tempBufferSize[i] < locPageSize) locPageSize = app->configuration.tempBufferSize[i];
					        }
				        }
                    }
                    else {
                        locBufferNum = app->configuration.bufferNum;
                        if (app->configuration.bufferSize) {
                            locBufferSize = app->configuration.bufferSize[0];
                            for (pfUINT i = 0; i < app->configuration.bufferNum; i++) {
                                totalSize += app->configuration.bufferSize[i];
                                if (app->configuration.bufferSize[i] < locPageSize) locPageSize = app->configuration.bufferSize[i];
                            }
                        }
                    }
                }
		}
		else {
            if ((inverse) && (axis_id == app->firstAxis) && (axis_upload_id == 0) && (app->configuration.isInputFormatted) && (app->configuration.inverseReturnToInputBuffer)) {				
                locBufferNum = app->configuration.inputBufferNum;
                if (app->configuration.inputBufferSize) {
                    locBufferSize = app->configuration.inputBufferSize[0];
                    for (pfUINT i = 0; i < app->configuration.inputBufferNum; i++) {
                        totalSize += app->configuration.inputBufferSize[i];
                        if (app->configuration.inputBufferSize[i] < locPageSize) locPageSize = app->configuration.inputBufferSize[i];
                    }
                } 
            }
            else {
                locBufferNum = app->configuration.bufferNum;
                if (app->configuration.bufferSize) {
                    locBufferSize = app->configuration.bufferSize[0];
                    for (pfUINT i = 0; i < app->configuration.bufferNum; i++) {
                        totalSize += app->configuration.bufferSize[i];
                        if (app->configuration.bufferSize[i] < locPageSize) locPageSize = app->configuration.bufferSize[i];
                    }
                }
            }
        }
		axis->specializationConstants.outputBufferBlockSize = (locBufferNum == 1) ? locBufferSize : locPageSize;
		axis->specializationConstants.outputBufferBlockNum = (locBufferNum == 1) ? 1 : (pfUINT)pfceil(totalSize / (double)(axis->specializationConstants.outputBufferBlockSize));
		//if (axis->specializationConstants.outputBufferBlockNum == 1) axis->specializationConstants.outputBufferBlockSize = totalSize / axis->specializationConstants.complexSize;

	}
	if (axis->specializationConstants.inputBufferBlockNum == 0) axis->specializationConstants.inputBufferBlockNum = 1;
	if (axis->specializationConstants.outputBufferBlockNum == 0) axis->specializationConstants.outputBufferBlockNum = 1;
	if (app->configuration.performConvolution) {
		pfUINT totalSize = 0;
		pfUINT locPageSize = initPageSize;
		locBufferNum = app->configuration.kernelNum;
		if (app->configuration.kernelSize) {
			locBufferSize = app->configuration.kernelSize[0];
			for (pfUINT i = 0; i < app->configuration.kernelNum; i++) {
				totalSize += app->configuration.kernelSize[i];
				if (app->configuration.kernelSize[i] < locPageSize) locPageSize = app->configuration.kernelSize[i];
			}
		}
		axis->specializationConstants.kernelBlockSize = (locBufferNum == 1) ? locBufferSize : locPageSize;
		axis->specializationConstants.kernelBlockNum = (locBufferNum == 1) ? 1 : (pfUINT)pfceil(totalSize / (double)(axis->specializationConstants.kernelBlockSize));
		//if (axis->specializationConstants.kernelBlockNum == 1) axis->specializationConstants.inputBufferBlockSize = totalSize / axis->specializationConstants.complexSize;
		if (axis->specializationConstants.kernelBlockNum == 0) axis->specializationConstants.kernelBlockNum = 1;
	}
	else {
		axis->specializationConstants.kernelBlockSize = 0;
		axis->specializationConstants.kernelBlockNum = 0;
	}
	axis->numBindings = 2;
	axis->specializationConstants.numBuffersBound[0] = (int)axis->specializationConstants.inputBufferBlockNum;
	axis->specializationConstants.numBuffersBound[1] = (int)axis->specializationConstants.outputBufferBlockNum;
	axis->specializationConstants.numBuffersBound[2] = 0;
	axis->specializationConstants.numBuffersBound[3] = 0;
#if(VKFFT_BACKEND==0)
	VkDescriptorPoolSize descriptorPoolSize = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
	descriptorPoolSize.descriptorCount = (uint32_t)(axis->specializationConstants.inputBufferBlockNum + axis->specializationConstants.outputBufferBlockNum);
#endif
	axis->specializationConstants.convolutionBindingID = -1;
	if ((axis_id == (app->configuration.FFTdim-1)) && (axis_upload_id == 0) && (app->configuration.performConvolution)) {
		axis->specializationConstants.convolutionBindingID = (int)axis->numBindings;
		axis->specializationConstants.numBuffersBound[axis->numBindings] = (int)axis->specializationConstants.kernelBlockNum;
#if(VKFFT_BACKEND==0)
		descriptorPoolSize.descriptorCount += (uint32_t)axis->specializationConstants.kernelBlockNum;
#endif
		axis->numBindings++;
	}
	if (app->configuration.useLUT == 1) {
		axis->specializationConstants.LUTBindingID = (int)axis->numBindings;
		axis->specializationConstants.numBuffersBound[axis->numBindings] = 1;
#if(VKFFT_BACKEND==0)
		descriptorPoolSize.descriptorCount++;
#endif
		axis->numBindings++;
	}
	if (axis->specializationConstants.raderUintLUT) {
		axis->specializationConstants.RaderUintLUTBindingID = (int)axis->numBindings;
		axis->specializationConstants.numBuffersBound[axis->numBindings] = 1;
#if(VKFFT_BACKEND==0)
		descriptorPoolSize.descriptorCount++;
#endif
		axis->numBindings++;
	}
	if ((app->useBluesteinFFT[axis_id]) && (axis_upload_id == 0)) {
		if (axis->specializationConstants.inverseBluestein)
			axis->bufferBluesteinFFT = &app->bufferBluesteinIFFT[axis_id];
		else
			axis->bufferBluesteinFFT = &app->bufferBluesteinFFT[axis_id];
		axis->specializationConstants.BluesteinConvolutionBindingID = (int)axis->numBindings;
		axis->specializationConstants.numBuffersBound[axis->numBindings] = 1;
#if(VKFFT_BACKEND==0)
		descriptorPoolSize.descriptorCount++;
#endif
		axis->numBindings++;
	}
	if ((app->useBluesteinFFT[axis_id]) && (axis_upload_id == (FFTPlan->numAxisUploads[axis_id] - 1))) {
		axis->bufferBluestein = &app->bufferBluestein[axis_id];
		axis->specializationConstants.BluesteinMultiplicationBindingID = (int)axis->numBindings;
		axis->specializationConstants.numBuffersBound[axis->numBindings] = 1;
#if(VKFFT_BACKEND==0)
		descriptorPoolSize.descriptorCount++;
#endif
		axis->numBindings++;
	}
#if(VKFFT_BACKEND==0)
	VkResult res = VK_SUCCESS;
	VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
	descriptorPoolCreateInfo.poolSizeCount = 1;
	descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize;
	descriptorPoolCreateInfo.maxSets = 1;
	res = vkCreateDescriptorPool(app->configuration.device[0], &descriptorPoolCreateInfo, 0, &axis->descriptorPool);
	if (res != VK_SUCCESS) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_CREATE_DESCRIPTOR_POOL;
	}
	const VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	VkDescriptorSetLayoutBinding* descriptorSetLayoutBindings;
	descriptorSetLayoutBindings = (VkDescriptorSetLayoutBinding*)malloc(axis->numBindings * sizeof(VkDescriptorSetLayoutBinding));
	if (!descriptorSetLayoutBindings) {
		deleteVkFFT(app);
		return VKFFT_ERROR_MALLOC_FAILED;
	}
	for (pfUINT i = 0; i < axis->numBindings; ++i) {
		descriptorSetLayoutBindings[i].binding = (uint32_t)i;
		descriptorSetLayoutBindings[i].descriptorType = descriptorType;
		descriptorSetLayoutBindings[i].descriptorCount = (uint32_t)axis->specializationConstants.numBuffersBound[i];
		descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	}

	VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
	descriptorSetLayoutCreateInfo.bindingCount = (uint32_t)axis->numBindings;
	descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;

	res = vkCreateDescriptorSetLayout(app->configuration.device[0], &descriptorSetLayoutCreateInfo, 0, &axis->descriptorSetLayout);
	if (res != VK_SUCCESS) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_CREATE_DESCRIPTOR_SET_LAYOUT;
	}
	free(descriptorSetLayoutBindings);
	descriptorSetLayoutBindings = 0;
	VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
	descriptorSetAllocateInfo.descriptorPool = axis->descriptorPool;
	descriptorSetAllocateInfo.descriptorSetCount = 1;
	descriptorSetAllocateInfo.pSetLayouts = &axis->descriptorSetLayout;
	res = vkAllocateDescriptorSets(app->configuration.device[0], &descriptorSetAllocateInfo, &axis->descriptorSet);
	if (res != VK_SUCCESS) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_ALLOCATE_DESCRIPTOR_SETS;
	}
#endif
	return VKFFT_SUCCESS;
}
static inline VkFFTResult VkFFTConfigureDescriptorsR2CMultiUploadDecomposition(VkFFTApplication* app, VkFFTPlan* FFTPlan, VkFFTAxis* axis, pfUINT axis_id, pfUINT axis_upload_id, pfUINT inverse) {
	pfUINT initPageSize = -1;
	pfUINT locBufferNum = 1;
	pfUINT locBufferSize = 0;
	
	{
		if (inverse) {
			if ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (app->configuration.isInputFormatted) && (!axis->specializationConstants.reverseBluesteinMultiUpload) && (
				((axis_id == app->firstAxis) && (!inverse))
				|| ((axis_id == app->lastAxis) && (inverse) && (!app->configuration.performConvolution) && (!app->configuration.inverseReturnToInputBuffer)))
				) {
				pfUINT totalSize = 0;
				pfUINT locPageSize = initPageSize;
				locBufferNum = app->configuration.inputBufferNum;
				if (app->configuration.inputBufferSize) {
					locBufferSize = app->configuration.inputBufferSize[0];
					for (pfUINT i = 0; i < app->configuration.inputBufferNum; i++) {
						totalSize += app->configuration.inputBufferSize[i];
						if (app->configuration.inputBufferSize[i] < locPageSize) locPageSize = app->configuration.inputBufferSize[i];
					}
				}
				axis->specializationConstants.inputBufferBlockSize = (locBufferNum == 1) ? locBufferSize : locPageSize;
				axis->specializationConstants.inputBufferBlockNum = (locBufferNum == 1) ? 1 : (pfUINT)pfceil(totalSize / (double)(axis->specializationConstants.inputBufferBlockSize));
				//if (axis->specializationConstants.inputBufferBlockNum == 1) axis->specializationConstants.inputBufferBlockSize = totalSize / axis->specializationConstants.complexSize;

			}
			else {
				if ((axis_upload_id == 0) && (app->configuration.numberKernels > 1) && (inverse) && (!app->configuration.performConvolution)) {
					pfUINT totalSize = 0;
					pfUINT locPageSize = initPageSize;
					locBufferNum = app->configuration.outputBufferNum;
					if (app->configuration.outputBufferSize) {
						locBufferSize = app->configuration.outputBufferSize[0];
						for (pfUINT i = 0; i < app->configuration.outputBufferNum; i++) {
							totalSize += app->configuration.outputBufferSize[i];
							if (app->configuration.outputBufferSize[i] < locPageSize) locPageSize = app->configuration.outputBufferSize[i];
						}
					}
					axis->specializationConstants.inputBufferBlockSize = (locBufferNum == 1) ? locBufferSize : locPageSize;
					axis->specializationConstants.inputBufferBlockNum = (locBufferNum == 1) ? 1 : (pfUINT)pfceil(totalSize / (double)(axis->specializationConstants.inputBufferBlockSize));
					//if (axis->specializationConstants.inputBufferBlockNum == 1) axis->specializationConstants.outputBufferBlockSize = totalSize / axis->specializationConstants.complexSize;

				}
				else {
					pfUINT totalSize = 0;
					pfUINT locPageSize = initPageSize;
					locBufferNum = app->configuration.bufferNum;
					if (app->configuration.bufferSize) {
						locBufferSize = app->configuration.bufferSize[0];
						for (pfUINT i = 0; i < app->configuration.bufferNum; i++) {
							totalSize += app->configuration.bufferSize[i];
							if (app->configuration.bufferSize[i] < locPageSize) locPageSize = app->configuration.bufferSize[i];

						}
					}
					axis->specializationConstants.inputBufferBlockSize = (locBufferNum == 1) ? locBufferSize : locPageSize;
					axis->specializationConstants.inputBufferBlockNum = (locBufferNum == 1) ? 1 : (pfUINT)pfceil(totalSize / (double)(axis->specializationConstants.inputBufferBlockSize));
					//if (axis->specializationConstants.inputBufferBlockNum == 1) axis->specializationConstants.inputBufferBlockSize = totalSize / axis->specializationConstants.complexSize;

				}
			}
		}
		else {
			if (((axis_upload_id == 0) && (!app->useBluesteinFFT[axis_id]) && (app->configuration.isOutputFormatted && (
				((axis_id == app->firstAxis) && (inverse))
				|| ((axis_id == app->lastAxis) && (!inverse) && (!app->configuration.performConvolution))
				|| ((axis_id == app->firstAxis) && (app->configuration.performConvolution) && (app->configuration.FFTdim == 1)))
				)) ||
				((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (app->useBluesteinFFT[axis_id]) && (axis->specializationConstants.reverseBluesteinMultiUpload || (FFTPlan->numAxisUploads[axis_id] == 1)) && (app->configuration.isOutputFormatted && (
					((axis_id == app->firstAxis) && (inverse))
					|| ((axis_id == app->lastAxis) && (!inverse) && (!app->configuration.performConvolution)))
					)) ||
				((app->configuration.numberKernels > 1) && (
					(inverse)
					|| (axis_id == app->lastAxis)))
				) {
				pfUINT totalSize = 0;
				pfUINT locPageSize = initPageSize;
				locBufferNum = app->configuration.outputBufferNum;
				if (app->configuration.outputBufferSize) {
					locBufferSize = app->configuration.outputBufferSize[0];
					for (pfUINT i = 0; i < app->configuration.outputBufferNum; i++) {
						totalSize += app->configuration.outputBufferSize[i];
						if (app->configuration.outputBufferSize[i] < locPageSize) locPageSize = app->configuration.outputBufferSize[i];
					}
				}
				axis->specializationConstants.inputBufferBlockSize = (locBufferNum == 1) ? locBufferSize : locPageSize;
				axis->specializationConstants.outputBufferBlockNum = (locBufferNum == 1) ? 1 : (pfUINT)pfceil(totalSize / (double)(axis->specializationConstants.inputBufferBlockSize));
				//if (axis->specializationConstants.outputBufferBlockNum == 1) axis->specializationConstants.outputBufferBlockSize = totalSize / axis->specializationConstants.complexSize;

			}
			else {
				pfUINT totalSize = 0;
				pfUINT locPageSize = initPageSize;

				locBufferNum = app->configuration.bufferNum;
				if (app->configuration.bufferSize) {
					locBufferSize = app->configuration.bufferSize[0];
					for (pfUINT i = 0; i < app->configuration.bufferNum; i++) {
						totalSize += app->configuration.bufferSize[i];
						if (app->configuration.bufferSize[i] < locPageSize) locPageSize = app->configuration.bufferSize[i];
					}
				}
				axis->specializationConstants.inputBufferBlockSize = (locBufferNum == 1) ? locBufferSize : locPageSize;
				axis->specializationConstants.outputBufferBlockNum = (locBufferNum == 1) ? 1 : (pfUINT)pfceil(totalSize / (double)(axis->specializationConstants.inputBufferBlockSize));
				//if (axis->specializationConstants.outputBufferBlockNum == 1) axis->specializationConstants.outputBufferBlockSize = totalSize / axis->specializationConstants.complexSize;

			}
		}
	}
	initPageSize = -1;
	locBufferNum = 1;
	locBufferSize = -1;
	{
		if (inverse) {
			if ((axis_upload_id == 0) && (app->configuration.numberKernels > 1) && (inverse) && (!app->configuration.performConvolution)) {
				pfUINT totalSize = 0;
				pfUINT locPageSize = initPageSize;
				locBufferNum = app->configuration.outputBufferNum;
				if (app->configuration.outputBufferSize) {
					locBufferSize = app->configuration.outputBufferSize[0];
					for (pfUINT i = 0; i < app->configuration.outputBufferNum; i++) {
						totalSize += app->configuration.outputBufferSize[i];
						if (app->configuration.outputBufferSize[i] < locPageSize) locPageSize = app->configuration.outputBufferSize[i];
					}
				}
				axis->specializationConstants.outputBufferBlockSize = (locBufferNum == 1) ? locBufferSize : locPageSize;
				axis->specializationConstants.outputBufferBlockNum = (locBufferNum == 1) ? 1 : (pfUINT)pfceil(totalSize / (double)(axis->specializationConstants.outputBufferBlockSize));
				//if (axis->specializationConstants.outputBufferBlockNum == 1) axis->specializationConstants.outputBufferBlockSize = totalSize / axis->specializationConstants.complexSize;

			}
			else {
				pfUINT totalSize = 0;
				pfUINT locPageSize = initPageSize;
				locBufferNum = app->configuration.bufferNum;
				if (app->configuration.bufferSize) {
					locBufferSize = app->configuration.bufferSize[0];
					for (pfUINT i = 0; i < app->configuration.bufferNum; i++) {
						totalSize += app->configuration.bufferSize[i];
						if (app->configuration.bufferSize[i] < locPageSize) locPageSize = app->configuration.bufferSize[i];

					}
				}
				axis->specializationConstants.outputBufferBlockSize = (locBufferNum == 1) ? locBufferSize : locPageSize;
				axis->specializationConstants.outputBufferBlockNum = (locBufferNum == 1) ? 1 : (pfUINT)pfceil(totalSize / (double)(axis->specializationConstants.outputBufferBlockSize));
				//if (axis->specializationConstants.outputBufferBlockNum == 1) axis->specializationConstants.outputBufferBlockSize = totalSize / axis->specializationConstants.complexSize;

			}
		}
		else {
			if (((axis_upload_id == 0) && (!app->useBluesteinFFT[axis_id]) && (app->configuration.isOutputFormatted && (
				((axis_id == app->firstAxis) && (inverse))
				|| ((axis_id == app->lastAxis) && (!inverse) && (!app->configuration.performConvolution))
				|| ((axis_id == app->firstAxis) && (app->configuration.performConvolution) && (app->configuration.FFTdim == 1)))
				)) ||
				((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (app->useBluesteinFFT[axis_id]) && (axis->specializationConstants.reverseBluesteinMultiUpload || (FFTPlan->numAxisUploads[axis_id] == 1)) && (app->configuration.isOutputFormatted && (
					((axis_id == app->firstAxis) && (inverse))
					|| ((axis_id == app->lastAxis) && (!inverse) && (!app->configuration.performConvolution)))
					)) ||
				((app->configuration.numberKernels > 1) && (
					(inverse)
					|| (axis_id == app->lastAxis)))
				) {
				pfUINT totalSize = 0;
				pfUINT locPageSize = initPageSize;
				locBufferNum = app->configuration.outputBufferNum;
				if (app->configuration.outputBufferSize) {
					locBufferSize = app->configuration.outputBufferSize[0];
					for (pfUINT i = 0; i < app->configuration.outputBufferNum; i++) {
						totalSize += app->configuration.outputBufferSize[i];
						if (app->configuration.outputBufferSize[i] < locPageSize) locPageSize = app->configuration.outputBufferSize[i];
					}
				}
				axis->specializationConstants.outputBufferBlockSize = (locBufferNum == 1) ? locBufferSize : locPageSize;
				axis->specializationConstants.outputBufferBlockNum = (locBufferNum == 1) ? 1 : (pfUINT)pfceil(totalSize / (double)(axis->specializationConstants.outputBufferBlockSize));
				//if (axis->specializationConstants.outputBufferBlockNum == 1) axis->specializationConstants.outputBufferBlockSize = totalSize / axis->specializationConstants.complexSize;

			}
			else {
				pfUINT totalSize = 0;
				pfUINT locPageSize = initPageSize;

				locBufferNum = app->configuration.bufferNum;
				if (app->configuration.bufferSize) {
					locBufferSize = app->configuration.bufferSize[0];
					for (pfUINT i = 0; i < app->configuration.bufferNum; i++) {
						totalSize += app->configuration.bufferSize[i];
						if (app->configuration.bufferSize[i] < locPageSize) locPageSize = app->configuration.bufferSize[i];
					}
				}
				axis->specializationConstants.outputBufferBlockSize = (locBufferNum == 1) ? locBufferSize : locPageSize;
				axis->specializationConstants.outputBufferBlockNum = (locBufferNum == 1) ? 1 : (pfUINT)pfceil(totalSize / (double)(axis->specializationConstants.outputBufferBlockSize));
				//if (axis->specializationConstants.outputBufferBlockNum == 1) axis->specializationConstants.outputBufferBlockSize = totalSize / axis->specializationConstants.complexSize;

			}
		}
	}

	if (axis->specializationConstants.inputBufferBlockNum == 0) axis->specializationConstants.inputBufferBlockNum = 1;
	if (axis->specializationConstants.outputBufferBlockNum == 0) axis->specializationConstants.outputBufferBlockNum = 1;
	if (app->configuration.performConvolution) {
		//need fixing (not used now)
		pfUINT totalSize = 0;
		pfUINT locPageSize = initPageSize;
		if (app->configuration.kernelSize) {
			for (pfUINT i = 0; i < app->configuration.kernelNum; i++) {
				totalSize += app->configuration.kernelSize[i];
				if (app->configuration.kernelSize[i] < locPageSize) locPageSize = app->configuration.kernelSize[i];
			}
		}
		axis->specializationConstants.kernelBlockSize = locPageSize;
		axis->specializationConstants.kernelBlockNum = (pfUINT)pfceil(totalSize / (double)(axis->specializationConstants.kernelBlockSize));
		//if (axis->specializationConstants.kernelBlockNum == 1) axis->specializationConstants.inputBufferBlockSize = totalSize / axis->specializationConstants.complexSize;
		if (axis->specializationConstants.kernelBlockNum == 0) axis->specializationConstants.kernelBlockNum = 1;
	}
	else {
		axis->specializationConstants.kernelBlockSize = 0;
		axis->specializationConstants.kernelBlockNum = 0;
	}
	axis->numBindings = 2;
	axis->specializationConstants.numBuffersBound[0] = (int)axis->specializationConstants.inputBufferBlockNum;
	axis->specializationConstants.numBuffersBound[1] = (int)axis->specializationConstants.outputBufferBlockNum;
	axis->specializationConstants.numBuffersBound[2] = 0;
	axis->specializationConstants.numBuffersBound[3] = 0;

#if(VKFFT_BACKEND==0)
	VkDescriptorPoolSize descriptorPoolSize = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
	descriptorPoolSize.descriptorCount = (uint32_t)(axis->specializationConstants.numBuffersBound[0] + axis->specializationConstants.numBuffersBound[1]);
#endif
	if ((axis_id == (app->configuration.FFTdim-1)) && (axis_upload_id == 0) && (app->configuration.performConvolution)) {
		axis->specializationConstants.numBuffersBound[axis->numBindings] = (int)axis->specializationConstants.kernelBlockNum;
#if(VKFFT_BACKEND==0)
		descriptorPoolSize.descriptorCount += (uint32_t)axis->specializationConstants.kernelBlockNum;
#endif
		axis->numBindings++;
	}

	if (app->configuration.useLUT == 1) {
		axis->specializationConstants.numBuffersBound[axis->numBindings] = 1;
#if(VKFFT_BACKEND==0)
		descriptorPoolSize.descriptorCount++;
#endif
		axis->numBindings++;
	}
#if(VKFFT_BACKEND==0)
	VkResult res = VK_SUCCESS;
	VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
	descriptorPoolCreateInfo.poolSizeCount = 1;
	descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize;
	descriptorPoolCreateInfo.maxSets = 1;
	res = vkCreateDescriptorPool(app->configuration.device[0], &descriptorPoolCreateInfo, 0, &axis->descriptorPool);
	if (res != VK_SUCCESS) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_CREATE_DESCRIPTOR_POOL;
	}
	const VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	VkDescriptorSetLayoutBinding* descriptorSetLayoutBindings;
	descriptorSetLayoutBindings = (VkDescriptorSetLayoutBinding*)malloc(axis->numBindings * sizeof(VkDescriptorSetLayoutBinding));
	if (!descriptorSetLayoutBindings) {
		deleteVkFFT(app);
		return VKFFT_ERROR_MALLOC_FAILED;
	}
	for (pfUINT i = 0; i < axis->numBindings; ++i) {
		descriptorSetLayoutBindings[i].binding = (uint32_t)i;
		descriptorSetLayoutBindings[i].descriptorType = descriptorType;
		descriptorSetLayoutBindings[i].descriptorCount = (uint32_t)axis->specializationConstants.numBuffersBound[i];
		descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	}

	VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
	descriptorSetLayoutCreateInfo.bindingCount = (uint32_t)axis->numBindings;
	descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;

	res = vkCreateDescriptorSetLayout(app->configuration.device[0], &descriptorSetLayoutCreateInfo, 0, &axis->descriptorSetLayout);
	if (res != VK_SUCCESS) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_CREATE_DESCRIPTOR_SET_LAYOUT;
	}
	free(descriptorSetLayoutBindings);
	descriptorSetLayoutBindings = 0;
	VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
	descriptorSetAllocateInfo.descriptorPool = axis->descriptorPool;
	descriptorSetAllocateInfo.descriptorSetCount = 1;
	descriptorSetAllocateInfo.pSetLayouts = &axis->descriptorSetLayout;
	res = vkAllocateDescriptorSets(app->configuration.device[0], &descriptorSetAllocateInfo, &axis->descriptorSet);
	if (res != VK_SUCCESS) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_ALLOCATE_DESCRIPTOR_SETS;
	}
#endif
	return VKFFT_SUCCESS;
}
static inline VkFFTResult VkFFTCheckUpdateBufferSet(VkFFTApplication* app, VkFFTAxis* axis, pfUINT planStage, VkFFTLaunchParams* launchParams) {
	pfUINT performBufferSetUpdate = planStage;
	pfUINT performOffsetUpdate = planStage;
	if (!planStage) {
		if (launchParams != 0) {
			if ((launchParams->buffer != 0) && (app->configuration.buffer != launchParams->buffer)) {
				app->configuration.buffer = launchParams->buffer;
				performBufferSetUpdate = 1;
			}
			if ((launchParams->inputBuffer != 0) && (app->configuration.inputBuffer != launchParams->inputBuffer)) {
				app->configuration.inputBuffer = launchParams->inputBuffer;
				performBufferSetUpdate = 1;
			}
			if ((launchParams->outputBuffer != 0) && (app->configuration.outputBuffer != launchParams->outputBuffer)) {
				app->configuration.outputBuffer = launchParams->outputBuffer;
				performBufferSetUpdate = 1;
			}
			if ((launchParams->tempBuffer != 0) && (app->configuration.tempBuffer != launchParams->tempBuffer)) {
				app->configuration.tempBuffer = launchParams->tempBuffer;
				performBufferSetUpdate = 1;
			}
			if ((launchParams->kernel != 0) && (app->configuration.kernel != launchParams->kernel)) {
				app->configuration.kernel = launchParams->kernel;
				performBufferSetUpdate = 1;
			}
			if (app->configuration.inputBuffer == 0) app->configuration.inputBuffer = app->configuration.buffer;
			if (app->configuration.outputBuffer == 0) app->configuration.outputBuffer = app->configuration.buffer;

			if (app->configuration.bufferOffset != launchParams->bufferOffset) {
				app->configuration.bufferOffset = launchParams->bufferOffset;
				performOffsetUpdate = 1;
			}
			if (app->configuration.inputBufferOffset != launchParams->inputBufferOffset) {
				app->configuration.inputBufferOffset = launchParams->inputBufferOffset;
				performOffsetUpdate = 1;
			}
			if (app->configuration.outputBufferOffset != launchParams->outputBufferOffset) {
				app->configuration.outputBufferOffset = launchParams->outputBufferOffset;
				performOffsetUpdate = 1;
			}
			if (app->configuration.tempBufferOffset != launchParams->tempBufferOffset) {
				app->configuration.tempBufferOffset = launchParams->tempBufferOffset;
				performOffsetUpdate = 1;
			}
			if (app->configuration.kernelOffset != launchParams->kernelOffset) {
				app->configuration.kernelOffset = launchParams->kernelOffset;
				performOffsetUpdate = 1;
			}
		}
	}
	if (planStage) {
		if (app->configuration.buffer == 0) {
			performBufferSetUpdate = 0;
		}
		if ((app->configuration.isInputFormatted) && (app->configuration.inputBuffer == 0)) {
			performBufferSetUpdate = 0;
		}
		if ((app->configuration.isOutputFormatted) && (app->configuration.outputBuffer == 0)) {
			performBufferSetUpdate = 0;
		}
		if (((app->configuration.userTempBuffer) && (app->configuration.tempBuffer == 0)) || (app->configuration.allocateTempBuffer)){
			performBufferSetUpdate = 0;
		}
		if ((app->configuration.performConvolution) && (app->configuration.kernel == 0)) {
			performBufferSetUpdate = 0;
		}
	}
	else {
		if (app->configuration.buffer == 0) {
			return VKFFT_ERROR_EMPTY_buffer;
		}
		if ((app->configuration.isInputFormatted) && (app->configuration.inputBuffer == 0)) {
			return VKFFT_ERROR_EMPTY_inputBuffer;
		}
		if ((app->configuration.isOutputFormatted) && (app->configuration.outputBuffer == 0)) {
			return VKFFT_ERROR_EMPTY_outputBuffer;
		}
		if ((app->configuration.userTempBuffer) && (app->configuration.tempBuffer == 0)) {
			return VKFFT_ERROR_EMPTY_tempBuffer;
		}
		if ((app->configuration.performConvolution) && (app->configuration.kernel == 0)) {
			return VKFFT_ERROR_EMPTY_kernel;
		}
	}
	if (performBufferSetUpdate) {
		if (planStage) axis->specializationConstants.performBufferSetUpdate = 1;
		else {
			if (!app->configuration.makeInversePlanOnly) {
				for (pfUINT i = 0; i < app->configuration.FFTdim; i++) {
					for (pfUINT j = 0; j < app->localFFTPlan->numAxisUploads[i]; j++)
						app->localFFTPlan->axes[i][j].specializationConstants.performBufferSetUpdate = 1;
					if (app->useBluesteinFFT[i] && (app->localFFTPlan->numAxisUploads[i] > 1)) {
						for (pfUINT j = 1; j < app->localFFTPlan->numAxisUploads[i]; j++)
							app->localFFTPlan->inverseBluesteinAxes[i][j].specializationConstants.performBufferSetUpdate = 1;
					}
				}
				if (app->localFFTPlan->bigSequenceEvenR2C) {
					app->localFFTPlan->R2Cdecomposition.specializationConstants.performBufferSetUpdate = 1;
				}
			}
			if (!app->configuration.makeForwardPlanOnly) {
				for (pfUINT i = 0; i < app->configuration.FFTdim; i++) {
					for (pfUINT j = 0; j < app->localFFTPlan_inverse->numAxisUploads[i]; j++)
						app->localFFTPlan_inverse->axes[i][j].specializationConstants.performBufferSetUpdate = 1;
					if (app->useBluesteinFFT[i] && (app->localFFTPlan_inverse->numAxisUploads[i] > 1)) {
						for (pfUINT j = 1; j < app->localFFTPlan_inverse->numAxisUploads[i]; j++)
							app->localFFTPlan_inverse->inverseBluesteinAxes[i][j].specializationConstants.performBufferSetUpdate = 1;
					}
				}
				if (app->localFFTPlan_inverse->bigSequenceEvenR2C) {
					app->localFFTPlan_inverse->R2Cdecomposition.specializationConstants.performBufferSetUpdate = 1;
				}
			}
		}
	}
	if (performOffsetUpdate) {
		if (planStage) axis->specializationConstants.performOffsetUpdate = 1;
		else {
			if (!app->configuration.makeInversePlanOnly) {
				for (pfUINT i = 0; i < app->configuration.FFTdim; i++) {
					for (pfUINT j = 0; j < app->localFFTPlan->numAxisUploads[i]; j++)
						app->localFFTPlan->axes[i][j].specializationConstants.performOffsetUpdate = 1;
					if (app->useBluesteinFFT[i] && (app->localFFTPlan->numAxisUploads[i] > 1)) {
						for (pfUINT j = 1; j < app->localFFTPlan->numAxisUploads[i]; j++)
							app->localFFTPlan->inverseBluesteinAxes[i][j].specializationConstants.performOffsetUpdate = 1;
					}
				}
				if (app->localFFTPlan->bigSequenceEvenR2C) {
					app->localFFTPlan->R2Cdecomposition.specializationConstants.performOffsetUpdate = 1;
				}
			}
			if (!app->configuration.makeForwardPlanOnly) {
				for (pfUINT i = 0; i < app->configuration.FFTdim; i++) {
					for (pfUINT j = 0; j < app->localFFTPlan_inverse->numAxisUploads[i]; j++)
						app->localFFTPlan_inverse->axes[i][j].specializationConstants.performOffsetUpdate = 1;
					if (app->useBluesteinFFT[i] && (app->localFFTPlan_inverse->numAxisUploads[i] > 1)) {
						for (pfUINT j = 1; j < app->localFFTPlan_inverse->numAxisUploads[i]; j++)
							app->localFFTPlan_inverse->inverseBluesteinAxes[i][j].specializationConstants.performOffsetUpdate = 1;
					}
				}
				if (app->localFFTPlan_inverse->bigSequenceEvenR2C) {
					app->localFFTPlan_inverse->R2Cdecomposition.specializationConstants.performOffsetUpdate = 1;
				}
			}
		}
	}
	return VKFFT_SUCCESS;
}
static inline VkFFTResult VkFFTUpdateBufferSet(VkFFTApplication* app, VkFFTPlan* FFTPlan, VkFFTAxis* axis, pfUINT axis_id, pfUINT axis_upload_id, pfUINT inverse) {
	if (axis->specializationConstants.performOffsetUpdate || axis->specializationConstants.performBufferSetUpdate) {
		axis->specializationConstants.inputOffset.type = 31;
		axis->specializationConstants.outputOffset.type = 31;
		axis->specializationConstants.kernelOffset.type = 31;
#if(VKFFT_BACKEND==0)
		const VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
#endif
		for (pfUINT i = 0; i < axis->numBindings; ++i) {
			for (pfUINT j = 0; j < axis->specializationConstants.numBuffersBound[i]; ++j) {
#if(VKFFT_BACKEND==0)
				VkDescriptorBufferInfo descriptorBufferInfo = { 0 };
#endif
				if (i == 0) {
					if ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (app->configuration.isInputFormatted) && (!axis->specializationConstants.reverseBluesteinMultiUpload) && (
						((axis_id == app->firstAxis) && (!inverse))
						|| ((axis_id == app->lastAxis) && (inverse) && (!((axis_id == 0) && (axis->specializationConstants.performR2CmultiUpload))) && (!app->configuration.performConvolution) && (!app->configuration.inverseReturnToInputBuffer)))
						) {
						if (axis->specializationConstants.performBufferSetUpdate) {
							pfUINT bufferId = 0;
							pfUINT offset = j;
							if (app->configuration.inputBufferSize)
							{
								for (pfUINT l = 0; l < app->configuration.inputBufferNum; ++l) {
									if (offset >= (pfUINT)pfceil(app->configuration.inputBufferSize[l] / (double)(axis->specializationConstants.inputBufferBlockSize))) {
										bufferId++;
										offset -= (pfUINT)pfceil(app->configuration.inputBufferSize[l] / (double)(axis->specializationConstants.inputBufferBlockSize));
									}
									else {
										l = app->configuration.inputBufferNum;
									}

								}
							}
							axis->inputBuffer = app->configuration.inputBuffer;
#if(VKFFT_BACKEND==0)
							descriptorBufferInfo.buffer = app->configuration.inputBuffer[bufferId];
							descriptorBufferInfo.range = (axis->specializationConstants.inputBufferBlockSize);
							descriptorBufferInfo.offset = offset * (axis->specializationConstants.inputBufferBlockSize);
#endif
						}
						if (axis->specializationConstants.performOffsetUpdate) {
							axis->specializationConstants.inputOffset.data.i = app->configuration.inputBufferOffset;
						}
					}
					else {
						if ((axis_upload_id == 0) && (app->configuration.numberKernels > 1) && (inverse) && (!app->configuration.performConvolution)) {
							if (axis->specializationConstants.performBufferSetUpdate) {
								pfUINT bufferId = 0;
								pfUINT offset = j;
								if (app->configuration.outputBufferSize)
								{
									for (pfUINT l = 0; l < app->configuration.outputBufferNum; ++l) {
										if (offset >= (pfUINT)pfceil(app->configuration.outputBufferSize[l] / (double)(axis->specializationConstants.inputBufferBlockSize))) {
											bufferId++;
											offset -= (pfUINT)pfceil(app->configuration.outputBufferSize[l] / (double)(axis->specializationConstants.inputBufferBlockSize));
										}
										else {
											l = app->configuration.outputBufferNum;
										}

									}
								}
								axis->inputBuffer = app->configuration.outputBuffer;
#if(VKFFT_BACKEND==0)
								descriptorBufferInfo.buffer = app->configuration.outputBuffer[bufferId];
								descriptorBufferInfo.range = (axis->specializationConstants.inputBufferBlockSize);
								descriptorBufferInfo.offset = offset * (axis->specializationConstants.inputBufferBlockSize);
#endif
							}
							if (axis->specializationConstants.performOffsetUpdate) {
								axis->specializationConstants.inputOffset.data.i = app->configuration.outputBufferOffset;
							}
						}
						else {
							pfUINT bufferId = 0;
							pfUINT offset = j;
							if (((axis->specializationConstants.reorderFourStep == 1) || (app->useBluesteinFFT[axis_id])) && (FFTPlan->numAxisUploads[axis_id] > 1)) {
								if ((((axis->specializationConstants.reorderFourStep == 1) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1)) || (app->useBluesteinFFT[axis_id] && (axis->specializationConstants.reverseBluesteinMultiUpload == 0) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1))) && (!((axis_id == 0) && (FFTPlan->bigSequenceEvenR2C) && (axis->specializationConstants.reorderFourStep == 1) && (inverse == 1)))) {
									if (axis->specializationConstants.performBufferSetUpdate) {
										if (app->configuration.bufferSize)
										{
											for (pfUINT l = 0; l < app->configuration.bufferNum; ++l) {
												if (offset >= (pfUINT)pfceil(app->configuration.bufferSize[l] / (double)(axis->specializationConstants.inputBufferBlockSize))) {
													bufferId++;
													offset -= (pfUINT)pfceil(app->configuration.bufferSize[l] / (double)(axis->specializationConstants.inputBufferBlockSize));
												}
												else {
													l = app->configuration.bufferNum;
												}

											}
										}
										axis->inputBuffer = app->configuration.buffer;
#if(VKFFT_BACKEND==0)
										descriptorBufferInfo.buffer = app->configuration.buffer[bufferId];
#endif
									}
									if (axis->specializationConstants.performOffsetUpdate) {
										axis->specializationConstants.inputOffset.data.i = app->configuration.bufferOffset;
									}
								}
								else {
									if (axis->specializationConstants.performBufferSetUpdate) {
										if (app->configuration.tempBufferSize) {
											for (pfUINT l = 0; l < app->configuration.tempBufferNum; ++l) {
												if (offset >= (pfUINT)pfceil(app->configuration.tempBufferSize[l] / (double)(axis->specializationConstants.inputBufferBlockSize))) {
													bufferId++;
													offset -= (pfUINT)pfceil(app->configuration.tempBufferSize[l] / (double)(axis->specializationConstants.inputBufferBlockSize));
												}
												else {
													l = app->configuration.tempBufferNum;
												}

											}
										}
										axis->inputBuffer = app->configuration.tempBuffer;
#if(VKFFT_BACKEND==0)
										descriptorBufferInfo.buffer = app->configuration.tempBuffer[bufferId];
#endif
									}
									if (axis->specializationConstants.performOffsetUpdate) {
										axis->specializationConstants.inputOffset.data.i = app->configuration.tempBufferOffset;
									}
								}
							}
							else {
								if (axis->specializationConstants.performBufferSetUpdate) {
									if (app->configuration.bufferSize) {
										for (pfUINT l = 0; l < app->configuration.bufferNum; ++l) {
											if (offset >= (pfUINT)pfceil(app->configuration.bufferSize[l] / (double)(axis->specializationConstants.inputBufferBlockSize))) {
												bufferId++;
												offset -= (pfUINT)pfceil(app->configuration.bufferSize[l] / (double)(axis->specializationConstants.inputBufferBlockSize));
											}
											else {
												l = app->configuration.bufferNum;
											}

										}
									}
									axis->inputBuffer = app->configuration.buffer;
#if(VKFFT_BACKEND==0)
									descriptorBufferInfo.buffer = app->configuration.buffer[bufferId];
#endif
								}
								if (axis->specializationConstants.performOffsetUpdate) {
									axis->specializationConstants.inputOffset.data.i = app->configuration.bufferOffset;
								}
							}
#if(VKFFT_BACKEND==0)
							if (axis->specializationConstants.performBufferSetUpdate) {
								descriptorBufferInfo.range = (axis->specializationConstants.inputBufferBlockSize);
								descriptorBufferInfo.offset = offset * (axis->specializationConstants.inputBufferBlockSize);
							}
#endif
						}
					}
					//descriptorBufferInfo.offset = 0;
				}
				if (i == 1) {
					if (((axis_upload_id == 0) && (!app->useBluesteinFFT[axis_id]) && (app->configuration.isOutputFormatted && (
						((axis_id == app->firstAxis) && (inverse))
						|| ((axis_id == app->lastAxis) && (!inverse) && (!app->configuration.performConvolution))
						|| ((axis_id == app->firstAxis) && (app->configuration.performConvolution) && (app->configuration.FFTdim == 1)))
						)) ||
						((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (app->useBluesteinFFT[axis_id]) && (axis->specializationConstants.reverseBluesteinMultiUpload || (FFTPlan->numAxisUploads[axis_id] == 1)) && (app->configuration.isOutputFormatted && (
							((axis_id == app->firstAxis) && (inverse))
							|| ((axis_id == app->lastAxis) && (!inverse) && (!app->configuration.performConvolution)))
							)) ||
						((app->configuration.numberKernels > 1) && (
							(inverse)
							|| (axis_id == app->lastAxis)))
						) {
						if (axis->specializationConstants.performBufferSetUpdate) {
							pfUINT bufferId = 0;
							pfUINT offset = j;
							if (app->configuration.outputBufferSize) {
								for (pfUINT l = 0; l < app->configuration.outputBufferNum; ++l) {
									if (offset >= (pfUINT)pfceil(app->configuration.outputBufferSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize))) {
										bufferId++;
										offset -= (pfUINT)pfceil(app->configuration.outputBufferSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize));
									}
									else {
										l = app->configuration.outputBufferNum;
									}

								}
							}
							axis->outputBuffer = app->configuration.outputBuffer;
#if(VKFFT_BACKEND==0)
							descriptorBufferInfo.buffer = app->configuration.outputBuffer[bufferId];
							descriptorBufferInfo.range = (axis->specializationConstants.outputBufferBlockSize);
							descriptorBufferInfo.offset = offset * (axis->specializationConstants.outputBufferBlockSize);
#endif
						}
						if (axis->specializationConstants.performOffsetUpdate) {
							axis->specializationConstants.outputOffset.data.i = app->configuration.outputBufferOffset;
						}
					}
					else {
						pfUINT bufferId = 0;
						pfUINT offset = j;

						if (((axis->specializationConstants.reorderFourStep == 1) || (app->useBluesteinFFT[axis_id])) && (FFTPlan->numAxisUploads[axis_id] > 1)) {
							if ((inverse) && (axis_id == app->firstAxis) && (
								((axis_upload_id == 0) && (app->configuration.isInputFormatted) && (app->configuration.inverseReturnToInputBuffer) && (!app->useBluesteinFFT[axis_id]))
								|| ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (app->configuration.isInputFormatted) && (axis->specializationConstants.actualInverse) && (app->configuration.inverseReturnToInputBuffer) && (app->useBluesteinFFT[axis_id]) && (axis->specializationConstants.reverseBluesteinMultiUpload || (FFTPlan->numAxisUploads[axis_id] == 1))))
								) {
								if (axis->specializationConstants.performBufferSetUpdate) {
									if (app->configuration.inputBufferSize) {
										for (pfUINT l = 0; l < app->configuration.inputBufferNum; ++l) {
											if (offset >= (pfUINT)pfceil(app->configuration.inputBufferSize[l] / (double)(axis->specializationConstants.inputBufferBlockSize))) {
												bufferId++;
												offset -= (pfUINT)pfceil(app->configuration.inputBufferSize[l] / (double)(axis->specializationConstants.inputBufferBlockSize));
											}
											else {
												l = app->configuration.inputBufferNum;
											}

										}
									}
									axis->outputBuffer = app->configuration.inputBuffer;
#if(VKFFT_BACKEND==0)
									descriptorBufferInfo.buffer = app->configuration.inputBuffer[bufferId];
#endif
								}
								if (axis->specializationConstants.performOffsetUpdate) {
									axis->specializationConstants.outputOffset.data.i = app->configuration.inputBufferOffset;
								}
							}
							else {
								if (((axis->specializationConstants.reorderFourStep == 1) && (axis_upload_id > 0)) || (app->useBluesteinFFT[axis_id] && (!((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (axis->specializationConstants.reverseBluesteinMultiUpload == 1))))) {
									if (axis->specializationConstants.performBufferSetUpdate) {
										if (app->configuration.tempBufferSize) {
											for (pfUINT l = 0; l < app->configuration.tempBufferNum; ++l) {
												if (offset >= (pfUINT)pfceil(app->configuration.tempBufferSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize))) {
													bufferId++;
													offset -= (pfUINT)pfceil(app->configuration.tempBufferSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize));
												}
												else {
													l = app->configuration.tempBufferNum;
												}

											}
										}
										axis->outputBuffer = app->configuration.tempBuffer;
#if(VKFFT_BACKEND==0)
										descriptorBufferInfo.buffer = app->configuration.tempBuffer[bufferId];
#endif
									}
									if (axis->specializationConstants.performOffsetUpdate) {
										axis->specializationConstants.outputOffset.data.i = app->configuration.tempBufferOffset;
									}
								}
								else {
									if (axis->specializationConstants.performBufferSetUpdate) {
										if (app->configuration.bufferSize) {
											for (pfUINT l = 0; l < app->configuration.bufferNum; ++l) {
												if (offset >= (pfUINT)pfceil(app->configuration.bufferSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize))) {
													bufferId++;
													offset -= (pfUINT)pfceil(app->configuration.bufferSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize));
												}
												else {
													l = app->configuration.bufferNum;
												}

											}
										}
										axis->outputBuffer = app->configuration.buffer;
#if(VKFFT_BACKEND==0)
										descriptorBufferInfo.buffer = app->configuration.buffer[bufferId];
#endif
									}
									if (axis->specializationConstants.performOffsetUpdate) {
										axis->specializationConstants.outputOffset.data.i = app->configuration.bufferOffset;
									}
								}
							}
						}
						else {
							if ((inverse) && (axis_id == app->firstAxis) && (axis_upload_id == 0) && (app->configuration.isInputFormatted) && (app->configuration.inverseReturnToInputBuffer)) {
								if (axis->specializationConstants.performBufferSetUpdate) {
									if (app->configuration.inputBufferSize) {
										for (pfUINT l = 0; l < app->configuration.inputBufferNum; ++l) {
											if (offset >= (pfUINT)pfceil(app->configuration.inputBufferSize[l] / (double)(axis->specializationConstants.inputBufferBlockSize))) {
												bufferId++;
												offset -= (pfUINT)pfceil(app->configuration.inputBufferSize[l] / (double)(axis->specializationConstants.inputBufferBlockSize));
											}
											else {
												l = app->configuration.inputBufferNum;
											}

										}
									}
									axis->outputBuffer = app->configuration.inputBuffer;
#if(VKFFT_BACKEND==0)
									descriptorBufferInfo.buffer = app->configuration.inputBuffer[bufferId];
#endif
								}
								if (axis->specializationConstants.performOffsetUpdate) {
									axis->specializationConstants.outputOffset.data.i = app->configuration.inputBufferOffset;
								}
							}
							else {
								if (axis->specializationConstants.performBufferSetUpdate) {
									if (app->configuration.bufferSize) {
										for (pfUINT l = 0; l < app->configuration.bufferNum; ++l) {
											if (offset >= (pfUINT)pfceil(app->configuration.bufferSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize))) {
												bufferId++;
												offset -= (pfUINT)pfceil(app->configuration.bufferSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize));
											}
											else {
												l = app->configuration.bufferNum;
											}

										}
									}
									axis->outputBuffer = app->configuration.buffer;
#if(VKFFT_BACKEND==0)
									descriptorBufferInfo.buffer = app->configuration.buffer[bufferId];
#endif
								}
								if (axis->specializationConstants.performOffsetUpdate) {
									axis->specializationConstants.outputOffset.data.i = app->configuration.bufferOffset;
								}
							}
						}
#if(VKFFT_BACKEND==0)
						if (axis->specializationConstants.performBufferSetUpdate) {
							descriptorBufferInfo.range = (axis->specializationConstants.outputBufferBlockSize);
							descriptorBufferInfo.offset = offset * (axis->specializationConstants.outputBufferBlockSize);
						}
#endif
					}
					//descriptorBufferInfo.offset = 0;
				}
				if ((i == axis->specializationConstants.convolutionBindingID) && (app->configuration.performConvolution)) {
					if (axis->specializationConstants.performBufferSetUpdate) {
						pfUINT bufferId = 0;
						pfUINT offset = j;
						if (app->configuration.kernelSize) {
							for (pfUINT l = 0; l < app->configuration.kernelNum; ++l) {
								if (offset >= (pfUINT)pfceil(app->configuration.kernelSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize))) {
									bufferId++;
									offset -= (pfUINT)pfceil(app->configuration.kernelSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize));
								}
								else {
									l = app->configuration.kernelNum;
								}

							}
						}
#if(VKFFT_BACKEND==0)
						descriptorBufferInfo.buffer = app->configuration.kernel[bufferId];
						descriptorBufferInfo.range = (axis->specializationConstants.kernelBlockSize);
						descriptorBufferInfo.offset = offset * (axis->specializationConstants.kernelBlockSize);
#endif
					}
					if (axis->specializationConstants.performOffsetUpdate) {
						axis->specializationConstants.kernelOffset.data.i = app->configuration.kernelOffset;
					}
				}
				if ((i == axis->specializationConstants.LUTBindingID) && (app->configuration.useLUT == 1)) {
#if(VKFFT_BACKEND==0)
					if (axis->specializationConstants.performBufferSetUpdate) {
						descriptorBufferInfo.buffer = axis->bufferLUT;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = axis->bufferLUTSize;
					}
#endif
				}
				if ((i == axis->specializationConstants.RaderUintLUTBindingID) && (axis->specializationConstants.raderUintLUT)) {
#if(VKFFT_BACKEND==0)
					if (axis->specializationConstants.performBufferSetUpdate) {
						descriptorBufferInfo.buffer = axis->bufferRaderUintLUT;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = axis->bufferRaderUintLUTSize;
					}
#endif
				}
				if ((i == axis->specializationConstants.BluesteinConvolutionBindingID) && (app->useBluesteinFFT[axis_id]) && (axis_upload_id == 0)) {
#if(VKFFT_BACKEND==0)
					if (axis->specializationConstants.performBufferSetUpdate) {
						if (axis->specializationConstants.inverseBluestein)
							descriptorBufferInfo.buffer = app->bufferBluesteinIFFT[axis_id];
						else
							descriptorBufferInfo.buffer = app->bufferBluesteinFFT[axis_id];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = app->bufferBluesteinSize[axis_id];
					}
#endif
				}
				if ((i == axis->specializationConstants.BluesteinMultiplicationBindingID) && (app->useBluesteinFFT[axis_id]) && (axis_upload_id == (FFTPlan->numAxisUploads[axis_id] - 1))) {
#if(VKFFT_BACKEND==0)
					if (axis->specializationConstants.performBufferSetUpdate) {
						descriptorBufferInfo.buffer = app->bufferBluestein[axis_id];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = app->bufferBluesteinSize[axis_id];
					}
#endif
				}
#if(VKFFT_BACKEND==0)
				if (axis->specializationConstants.performBufferSetUpdate) {
					VkWriteDescriptorSet writeDescriptorSet = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
					writeDescriptorSet.dstSet = axis->descriptorSet;
					writeDescriptorSet.dstBinding = (uint32_t)i;
					writeDescriptorSet.dstArrayElement = (uint32_t)j;
					writeDescriptorSet.descriptorType = descriptorType;
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(app->configuration.device[0], 1, &writeDescriptorSet, 0, 0);
				}
#endif
			}
		}
	}
	if (axis->specializationConstants.performBufferSetUpdate) {
		axis->specializationConstants.performBufferSetUpdate = 0;
	}
	if (axis->specializationConstants.performOffsetUpdate) {
		axis->specializationConstants.performOffsetUpdate = 0;
	}
	return VKFFT_SUCCESS;
}
static inline VkFFTResult VkFFTUpdateBufferSetR2CMultiUploadDecomposition(VkFFTApplication* app, VkFFTPlan* FFTPlan, VkFFTAxis* axis, pfUINT axis_id, pfUINT axis_upload_id, pfUINT inverse) {
	if (axis->specializationConstants.performOffsetUpdate || axis->specializationConstants.performBufferSetUpdate) {
#if(VKFFT_BACKEND==0)
		const VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
#endif
		for (pfUINT i = 0; i < axis->numBindings; ++i) {
			for (pfUINT j = 0; j < axis->specializationConstants.numBuffersBound[i]; ++j) {
#if(VKFFT_BACKEND==0)
				VkDescriptorBufferInfo descriptorBufferInfo = { 0 };
#endif
				if (i == 0) {
					if (inverse) {
						if ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (app->configuration.isInputFormatted) && (!axis->specializationConstants.reverseBluesteinMultiUpload) && (
							((axis_id == app->firstAxis) && (!inverse))
							|| ((axis_id == app->lastAxis) && (inverse) && (!app->configuration.performConvolution) && (!app->configuration.inverseReturnToInputBuffer)))
							) {
							if (axis->specializationConstants.performBufferSetUpdate) {
								pfUINT bufferId = 0;
								pfUINT offset = j;
								if (app->configuration.inputBufferSize) {
									for (pfUINT l = 0; l < app->configuration.inputBufferNum; ++l) {
										if (offset >= (pfUINT)pfceil(app->configuration.inputBufferSize[l] / (double)(axis->specializationConstants.inputBufferBlockSize))) {
											bufferId++;
											offset -= (pfUINT)pfceil(app->configuration.inputBufferSize[l] / (double)(axis->specializationConstants.inputBufferBlockSize));
										}
										else {
											l = app->configuration.inputBufferNum;
										}

									}
								}
								axis->inputBuffer = app->configuration.inputBuffer;
#if(VKFFT_BACKEND==0)
								descriptorBufferInfo.buffer = app->configuration.inputBuffer[bufferId];
								descriptorBufferInfo.range = (axis->specializationConstants.inputBufferBlockSize);
								descriptorBufferInfo.offset = offset * (axis->specializationConstants.inputBufferBlockSize);
#endif
							}
							if (axis->specializationConstants.performOffsetUpdate) {
								axis->specializationConstants.inputOffset.data.i = app->configuration.inputBufferOffset;
							}
						}
						else {
							if ((axis_upload_id == 0) && (app->configuration.numberKernels > 1) && (inverse) && (!app->configuration.performConvolution)) {
								if (axis->specializationConstants.performBufferSetUpdate) {
									pfUINT bufferId = 0;
									pfUINT offset = j;
									if (app->configuration.outputBufferSize) {
										for (pfUINT l = 0; l < app->configuration.outputBufferNum; ++l) {
											if (offset >= (pfUINT)pfceil(app->configuration.outputBufferSize[l] / (double)(axis->specializationConstants.inputBufferBlockSize))) {
												bufferId++;
												offset -= (pfUINT)pfceil(app->configuration.outputBufferSize[l] / (double)(axis->specializationConstants.inputBufferBlockSize));
											}
											else {
												l = app->configuration.outputBufferNum;
											}

										}
									}
									axis->inputBuffer = app->configuration.outputBuffer;
#if(VKFFT_BACKEND==0)
									descriptorBufferInfo.buffer = app->configuration.outputBuffer[bufferId];
									descriptorBufferInfo.range = (axis->specializationConstants.inputBufferBlockSize);
									descriptorBufferInfo.offset = offset * (axis->specializationConstants.inputBufferBlockSize);
#endif
								}
								if (axis->specializationConstants.performOffsetUpdate) {
									axis->specializationConstants.inputOffset.data.i = app->configuration.outputBufferOffset;
								}
							}
							else {
								if (axis->specializationConstants.performBufferSetUpdate) {
									pfUINT bufferId = 0;
									pfUINT offset = j;
									if (app->configuration.bufferSize) {
										for (pfUINT l = 0; l < app->configuration.bufferNum; ++l) {
											if (offset >= (pfUINT)pfceil(app->configuration.bufferSize[l] / (double)(axis->specializationConstants.inputBufferBlockSize))) {
												bufferId++;
												offset -= (pfUINT)pfceil(app->configuration.bufferSize[l] / (double)(axis->specializationConstants.inputBufferBlockSize));
											}
											else {
												l = app->configuration.bufferNum;
											}

										}
									}
									axis->inputBuffer = app->configuration.buffer;
#if(VKFFT_BACKEND==0)
									descriptorBufferInfo.buffer = app->configuration.buffer[bufferId];
									descriptorBufferInfo.range = (axis->specializationConstants.inputBufferBlockSize);
									descriptorBufferInfo.offset = offset * (axis->specializationConstants.inputBufferBlockSize);
#endif
								}
								if (axis->specializationConstants.performOffsetUpdate) {
									axis->specializationConstants.inputOffset.data.i = app->configuration.bufferOffset;
								}
							}
						}
					}
					else {
						if (((axis_upload_id == 0) && (!app->useBluesteinFFT[axis_id]) && (app->configuration.isOutputFormatted && (
							((axis_id == app->firstAxis) && (inverse))
							|| ((axis_id == app->lastAxis) && (!inverse) && (!app->configuration.performConvolution))
							|| ((axis_id == app->firstAxis) && (app->configuration.performConvolution) && (app->configuration.FFTdim == 1)))
							)) ||
							((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (app->useBluesteinFFT[axis_id]) && (axis->specializationConstants.reverseBluesteinMultiUpload || (FFTPlan->numAxisUploads[axis_id] == 1)) && (app->configuration.isOutputFormatted && (
								((axis_id == app->firstAxis) && (inverse))
								|| ((axis_id == app->lastAxis) && (!inverse) && (!app->configuration.performConvolution)))
								)) ||
							((app->configuration.numberKernels > 1) && (
								(inverse)
								|| (axis_id == app->lastAxis)))
							) {
							if (axis->specializationConstants.performBufferSetUpdate) {
								pfUINT bufferId = 0;
								pfUINT offset = j;
								if (app->configuration.outputBufferSize) {
									for (pfUINT l = 0; l < app->configuration.outputBufferNum; ++l) {
										if (offset >= (pfUINT)pfceil(app->configuration.outputBufferSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize))) {
											bufferId++;
											offset -= (pfUINT)pfceil(app->configuration.outputBufferSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize));
										}
										else {
											l = app->configuration.outputBufferNum;
										}

									}
								}
								axis->inputBuffer = app->configuration.outputBuffer;
#if(VKFFT_BACKEND==0)
								descriptorBufferInfo.buffer = app->configuration.outputBuffer[bufferId];
								descriptorBufferInfo.range = (axis->specializationConstants.outputBufferBlockSize);
								descriptorBufferInfo.offset = offset * (axis->specializationConstants.outputBufferBlockSize);
#endif
							}
							if (axis->specializationConstants.performOffsetUpdate) {
								axis->specializationConstants.inputOffset.data.i = app->configuration.outputBufferOffset;
							}
						}
						else {
							if (axis->specializationConstants.performBufferSetUpdate) {
								pfUINT bufferId = 0;
								pfUINT offset = j;
								if (app->configuration.bufferSize) {
									for (pfUINT l = 0; l < app->configuration.bufferNum; ++l) {
										if (offset >= (pfUINT)pfceil(app->configuration.bufferSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize))) {
											bufferId++;
											offset -= (pfUINT)pfceil(app->configuration.bufferSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize));
										}
										else {
											l = app->configuration.bufferNum;
										}

									}
								}
								axis->inputBuffer = app->configuration.buffer;
#if(VKFFT_BACKEND==0)
								descriptorBufferInfo.buffer = app->configuration.buffer[bufferId];
								descriptorBufferInfo.range = (axis->specializationConstants.outputBufferBlockSize);
								descriptorBufferInfo.offset = offset * (axis->specializationConstants.outputBufferBlockSize);
#endif
							}
							if (axis->specializationConstants.performOffsetUpdate) {
								axis->specializationConstants.inputOffset.data.i = app->configuration.bufferOffset;
							}
						}
					}
				}
				if (i == 1) {
					if (inverse) {
						if ((axis_upload_id == 0) && (app->configuration.numberKernels > 1) && (inverse) && (!app->configuration.performConvolution)) {
							if (axis->specializationConstants.performBufferSetUpdate) {
								pfUINT bufferId = 0;
								pfUINT offset = j;
								if (app->configuration.outputBufferSize) {
									for (pfUINT l = 0; l < app->configuration.outputBufferNum; ++l) {
										if (offset >= (pfUINT)pfceil(app->configuration.outputBufferSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize))) {
											bufferId++;
											offset -= (pfUINT)pfceil(app->configuration.outputBufferSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize));
										}
										else {
											l = app->configuration.outputBufferNum;
										}

									}
								}
								axis->outputBuffer = app->configuration.outputBuffer;
#if(VKFFT_BACKEND==0)
								descriptorBufferInfo.buffer = app->configuration.outputBuffer[bufferId];
								descriptorBufferInfo.range = (axis->specializationConstants.outputBufferBlockSize);
								descriptorBufferInfo.offset = offset * (axis->specializationConstants.outputBufferBlockSize);
#endif
							}
							if (axis->specializationConstants.performOffsetUpdate) {
								axis->specializationConstants.outputOffset.data.i = app->configuration.outputBufferOffset;
							}
						}
						else {
							pfUINT bufferId = 0;
							pfUINT offset = j;
							if (axis->specializationConstants.reorderFourStep == 1) {
								if (axis->specializationConstants.performBufferSetUpdate) {
									if (app->configuration.tempBufferSize) {
										for (pfUINT l = 0; l < app->configuration.tempBufferNum; ++l) {
											if (offset >= (pfUINT)pfceil(app->configuration.tempBufferSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize))) {
												bufferId++;
												offset -= (pfUINT)pfceil(app->configuration.tempBufferSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize));
											}
											else {
												l = app->configuration.tempBufferNum;
											}

										}
									}
									axis->outputBuffer = app->configuration.tempBuffer;
#if(VKFFT_BACKEND==0)
									descriptorBufferInfo.buffer = app->configuration.tempBuffer[bufferId];
									descriptorBufferInfo.range = (axis->specializationConstants.outputBufferBlockSize);
									descriptorBufferInfo.offset = offset * (axis->specializationConstants.outputBufferBlockSize);
#endif
								}
								if (axis->specializationConstants.performOffsetUpdate) {
									axis->specializationConstants.outputOffset.data.i = app->configuration.tempBufferOffset;
								}
							}
							else {
								if (axis->specializationConstants.performBufferSetUpdate) {
									if (app->configuration.bufferSize) {
										for (pfUINT l = 0; l < app->configuration.bufferNum; ++l) {
											if (offset >= (pfUINT)pfceil(app->configuration.bufferSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize))) {
												bufferId++;
												offset -= (pfUINT)pfceil(app->configuration.bufferSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize));
											}
											else {
												l = app->configuration.bufferNum;
											}

										}
									}
									axis->outputBuffer = app->configuration.buffer;
#if(VKFFT_BACKEND==0)
									descriptorBufferInfo.buffer = app->configuration.buffer[bufferId];
									descriptorBufferInfo.range = (axis->specializationConstants.outputBufferBlockSize);
									descriptorBufferInfo.offset = offset * (axis->specializationConstants.outputBufferBlockSize);
#endif
								}
								if (axis->specializationConstants.performOffsetUpdate) {
									axis->specializationConstants.outputOffset.data.i = app->configuration.bufferOffset;
								}
							}
						}
					}
					else {
						if (((axis_upload_id == 0) && (!app->useBluesteinFFT[axis_id]) && (app->configuration.isOutputFormatted && (
							((axis_id == app->firstAxis) && (inverse))
							|| ((axis_id == app->lastAxis) && (!inverse) && (!app->configuration.performConvolution))
							|| ((axis_id == app->firstAxis) && (app->configuration.performConvolution) && (app->configuration.FFTdim == 1)))
							)) ||
							((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (app->useBluesteinFFT[axis_id]) && (axis->specializationConstants.reverseBluesteinMultiUpload || (FFTPlan->numAxisUploads[axis_id] == 1)) && (app->configuration.isOutputFormatted && (
								((axis_id == app->firstAxis) && (inverse))
								|| ((axis_id == app->lastAxis) && (!inverse) && (!app->configuration.performConvolution)))
								)) ||
							((app->configuration.numberKernels > 1) && (
								(inverse)
								|| (axis_id == app->lastAxis)))
							) {
							if (axis->specializationConstants.performBufferSetUpdate) {
								pfUINT bufferId = 0;
								pfUINT offset = j;
								if (app->configuration.outputBufferSize) {
									for (pfUINT l = 0; l < app->configuration.outputBufferNum; ++l) {
										if (offset >= (pfUINT)pfceil(app->configuration.outputBufferSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize))) {
											bufferId++;
											offset -= (pfUINT)pfceil(app->configuration.outputBufferSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize));
										}
										else {
											l = app->configuration.outputBufferNum;
										}

									}
								}
								axis->outputBuffer = app->configuration.outputBuffer;
#if(VKFFT_BACKEND==0)
								descriptorBufferInfo.buffer = app->configuration.outputBuffer[bufferId];
								descriptorBufferInfo.range = (axis->specializationConstants.outputBufferBlockSize);
								descriptorBufferInfo.offset = offset * (axis->specializationConstants.outputBufferBlockSize);
#endif
							}
							if (axis->specializationConstants.performOffsetUpdate) {
								axis->specializationConstants.outputOffset.data.i = app->configuration.outputBufferOffset;
							}
						}
						else {
							if (axis->specializationConstants.performBufferSetUpdate) {
								pfUINT bufferId = 0;
								pfUINT offset = j;
								if (app->configuration.bufferSize) {
									for (pfUINT l = 0; l < app->configuration.bufferNum; ++l) {
										if (offset >= (pfUINT)pfceil(app->configuration.bufferSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize))) {
											bufferId++;
											offset -= (pfUINT)pfceil(app->configuration.bufferSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize));
										}
										else {
											l = app->configuration.bufferNum;
										}

									}
								}
								axis->outputBuffer = app->configuration.buffer;
#if(VKFFT_BACKEND==0)
								descriptorBufferInfo.buffer = app->configuration.buffer[bufferId];
								descriptorBufferInfo.range = (axis->specializationConstants.outputBufferBlockSize);
								descriptorBufferInfo.offset = offset * (axis->specializationConstants.outputBufferBlockSize);
#endif
							}
							if (axis->specializationConstants.performOffsetUpdate) {
								axis->specializationConstants.outputOffset.data.i = app->configuration.bufferOffset;
							}
						}
					}
				}
				if ((i == 2) && (app->configuration.performConvolution)) {
					if (axis->specializationConstants.performBufferSetUpdate) {
						pfUINT bufferId = 0;
						pfUINT offset = j;
						if (app->configuration.kernelSize) {
							for (pfUINT l = 0; l < app->configuration.kernelNum; ++l) {
								if (offset >= (pfUINT)pfceil(app->configuration.kernelSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize))) {
									bufferId++;
									offset -= (pfUINT)pfceil(app->configuration.kernelSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize));
								}
								else {
									l = app->configuration.kernelNum;
								}

							}
						}
#if(VKFFT_BACKEND==0)
						descriptorBufferInfo.buffer = app->configuration.kernel[bufferId];
						descriptorBufferInfo.range = (axis->specializationConstants.kernelBlockSize);
						descriptorBufferInfo.offset = offset * (axis->specializationConstants.kernelBlockSize);
#endif
					}
					if (axis->specializationConstants.performOffsetUpdate) {
						axis->specializationConstants.kernelOffset.data.i = app->configuration.kernelOffset;
					}
				}
				if ((i == axis->numBindings - 1) && (app->configuration.useLUT == 1)) {
#if(VKFFT_BACKEND==0)
					if (axis->specializationConstants.performBufferSetUpdate) {
						descriptorBufferInfo.buffer = axis->bufferLUT;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = axis->bufferLUTSize;
					}
#endif
				}
#if(VKFFT_BACKEND==0)
				if (axis->specializationConstants.performBufferSetUpdate) {
					VkWriteDescriptorSet writeDescriptorSet = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
					writeDescriptorSet.dstSet = axis->descriptorSet;
					writeDescriptorSet.dstBinding = (uint32_t)i;
					writeDescriptorSet.dstArrayElement = (uint32_t)j;
					writeDescriptorSet.descriptorType = descriptorType;
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(app->configuration.device[0], 1, &writeDescriptorSet, 0, 0);
				}
#endif
			}
		}
	}
	if (axis->specializationConstants.performBufferSetUpdate) {
		axis->specializationConstants.performBufferSetUpdate = 0;
	}
	if (axis->specializationConstants.performOffsetUpdate) {
		axis->specializationConstants.performOffsetUpdate = 0;
	}
	return VKFFT_SUCCESS;
}

#endif
