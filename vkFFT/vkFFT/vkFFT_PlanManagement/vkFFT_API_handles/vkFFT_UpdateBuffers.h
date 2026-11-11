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
static inline void VkFFTSetBlockParameters(int* axisSeparateComplexComponents, pfUINT* initPageSize, pfUINT* locBufferNum, pfUINT* locBufferSize, pfUINT bufferNum, pfUINT* bufferSize, pfUINT separateComplexComponents, pfUINT* bufferBlockSize, pfUINT* bufferBlockNum){
	if (separateComplexComponents){
		if (bufferSize)
			locBufferSize[0] = bufferSize[0];
		bufferBlockSize[0] = locBufferSize[0];
		bufferBlockNum[0] = 2;
		axisSeparateComplexComponents[0] = 1;
	}
	else
	{	
		pfUINT totalSize = 0;
		pfUINT locPageSize = initPageSize[0];
		locBufferNum[0] = bufferNum;
		if (bufferSize) {
			locBufferSize[0] = bufferSize[0];
			for (pfUINT i = 0; i < bufferNum; i++) {
				totalSize += bufferSize[i];
				if (bufferSize[i] < locPageSize) locPageSize = bufferSize[i];
			}
		}
		bufferBlockSize[0] = (locBufferNum[0] == 1) ? locBufferSize[0] : locPageSize;
		bufferBlockNum[0] = (locBufferNum[0] == 1) ? 1 : (pfUINT)pfceil(totalSize / (double)(bufferBlockSize[0]));
	}
		
}
static inline VkFFTResult VkFFTConfigureDescriptors(VkFFTApplication* app, VkFFTPlan* FFTPlan, VkFFTAxis* axis, pfUINT axis_id, pfUINT axis_upload_id, pfUINT inverse) {
	pfUINT initPageSize = -1;
	pfUINT locBufferNum = 1;
	pfUINT locBufferSize = -1;
	if ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (app->configuration.isInputFormatted) && (!axis->specializationConstants.reverseBluesteinMultiUpload) && (
		((axis_id == app->firstAxis) && (!inverse))
		|| ((axis_id == app->lastAxis) && (inverse) && (!((axis_id == 0) && (axis->specializationConstants.performR2CmultiUpload))) && (!app->configuration.performConvolution) && (!app->configuration.inverseReturnToInputBuffer)))
		) {
		VkFFTSetBlockParameters(&axis->specializationConstants.inputBufferSeparateComplexComponents, &initPageSize, &locBufferNum, &locBufferSize, app->configuration.inputBufferNum, app->configuration.inputBufferSize, app->configuration.inputBufferSeparateComplexComponents, &axis->specializationConstants.inputBufferBlockSize, &axis->specializationConstants.inputBufferBlockNum);
	}
	else {
		if ((axis_upload_id == 0) && (app->configuration.numberKernels > 1) && (inverse) && (!app->configuration.performConvolution)) {
			VkFFTSetBlockParameters(&axis->specializationConstants.inputBufferSeparateComplexComponents, &initPageSize, &locBufferNum, &locBufferSize, app->configuration.outputBufferNum, app->configuration.outputBufferSize, app->configuration.outputBufferSeparateComplexComponents, &axis->specializationConstants.inputBufferBlockSize, &axis->specializationConstants.inputBufferBlockNum);
		}
		else {
			if (((axis->specializationConstants.reorderFourStep) || (app->useBluesteinFFT[axis_id])) && (FFTPlan->numAxisUploads[axis_id] > 1)) {
				int parity_reorderFourStep_startBuffer = 0;// ((axis_id == 0) && (FFTPlan->bigSequenceEvenR2C) && (axis->specializationConstants.reorderFourStep == 2) && (inverse == 1)) ? 1 : 0;
				if ((((axis->specializationConstants.reorderFourStep == 1) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1)) || ((axis->specializationConstants.reorderFourStep == 2) && (((FFTPlan->numAxisUploads[axis_id] - 1 - axis_upload_id)%2) == parity_reorderFourStep_startBuffer)) || (app->useBluesteinFFT[axis_id] && (axis->specializationConstants.reverseBluesteinMultiUpload == 0) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1)))) {
                    VkFFTSetBlockParameters(&axis->specializationConstants.inputBufferSeparateComplexComponents, &initPageSize, &locBufferNum, &locBufferSize, app->configuration.bufferNum, app->configuration.bufferSize, app->configuration.bufferSeparateComplexComponents, &axis->specializationConstants.inputBufferBlockSize, &axis->specializationConstants.inputBufferBlockNum);
				}
				else {
					axis->specializationConstants.tempBufferInput = 1;
					VkFFTSetBlockParameters(&axis->specializationConstants.inputBufferSeparateComplexComponents, &initPageSize, &locBufferNum, &locBufferSize, app->configuration.tempBufferNum, app->configuration.tempBufferSize, app->configuration.tempBufferSeparateComplexComponents, &axis->specializationConstants.inputBufferBlockSize, &axis->specializationConstants.inputBufferBlockNum);		
				}
			}
			else {
				if ((axis->specializationConstants.optimizePow2StridesTempBuffer) && (((axis_id > 0) && (inverse == 0) && (FFTPlan->numAxisUploads[axis_id - 1] == 1)) || ((axis_id < (app->configuration.FFTdim - 1)) && (inverse == 1) && (FFTPlan->numAxisUploads[axis_id + 1] == 1)))) {
					axis->specializationConstants.tempBufferInput = 1;
					VkFFTSetBlockParameters(&axis->specializationConstants.inputBufferSeparateComplexComponents, &initPageSize, &locBufferNum, &locBufferSize, app->configuration.tempBufferNum, app->configuration.tempBufferSize, app->configuration.tempBufferSeparateComplexComponents, &axis->specializationConstants.inputBufferBlockSize, &axis->specializationConstants.inputBufferBlockNum);
				}
				else
					VkFFTSetBlockParameters(&axis->specializationConstants.inputBufferSeparateComplexComponents, &initPageSize, &locBufferNum, &locBufferSize, app->configuration.bufferNum, app->configuration.bufferSize, app->configuration.bufferSeparateComplexComponents, &axis->specializationConstants.inputBufferBlockSize, &axis->specializationConstants.inputBufferBlockNum);
			}
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
		VkFFTSetBlockParameters(&axis->specializationConstants.outputBufferSeparateComplexComponents, &initPageSize, &locBufferNum, &locBufferSize, app->configuration.outputBufferNum, app->configuration.outputBufferSize, app->configuration.outputBufferSeparateComplexComponents, &axis->specializationConstants.outputBufferBlockSize, &axis->specializationConstants.outputBufferBlockNum);
	}
	else {
		if (((axis->specializationConstants.reorderFourStep) || (app->useBluesteinFFT[axis_id])) && (FFTPlan->numAxisUploads[axis_id] > 1)) {
            if ((inverse) && (axis_id == app->firstAxis) && (
                ((axis_upload_id == 0) && (app->configuration.isInputFormatted) && (app->configuration.inverseReturnToInputBuffer) && (!app->useBluesteinFFT[axis_id]))
                || ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (app->configuration.isInputFormatted) && (axis->specializationConstants.actualInverse) && (app->configuration.inverseReturnToInputBuffer) && (app->useBluesteinFFT[axis_id]) && (axis->specializationConstants.reverseBluesteinMultiUpload || (FFTPlan->numAxisUploads[axis_id] == 1))))
                ) {
                    VkFFTSetBlockParameters(&axis->specializationConstants.outputBufferSeparateComplexComponents, &initPageSize, &locBufferNum, &locBufferSize, app->configuration.inputBufferNum, app->configuration.inputBufferSize, app->configuration.inputBufferSeparateComplexComponents, &axis->specializationConstants.outputBufferBlockSize, &axis->specializationConstants.outputBufferBlockNum);
                }
                else{
				int parity_reorderFourStep_startBuffer = 0;// ((axis_id == 0) && (FFTPlan->bigSequenceEvenR2C) && (axis->specializationConstants.reorderFourStep == 2) && (inverse == 1)) ? 1 : 0;
					if (((axis->specializationConstants.reorderFourStep == 1) && (axis_upload_id > 0)) || ((axis->specializationConstants.reorderFourStep == 2) && ((((FFTPlan->numAxisUploads[axis_id] - 1 - axis_upload_id)%2) == parity_reorderFourStep_startBuffer) && (axis_upload_id != 0))) || (app->useBluesteinFFT[axis_id] && (!((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (axis->specializationConstants.reverseBluesteinMultiUpload == 1))))) {
						axis->specializationConstants.tempBufferOutput = 1;
						VkFFTSetBlockParameters(&axis->specializationConstants.outputBufferSeparateComplexComponents, &initPageSize, &locBufferNum, &locBufferSize, app->configuration.tempBufferNum, app->configuration.tempBufferSize, app->configuration.tempBufferSeparateComplexComponents, &axis->specializationConstants.outputBufferBlockSize, &axis->specializationConstants.outputBufferBlockNum);
                    }
                    else {
                        VkFFTSetBlockParameters(&axis->specializationConstants.outputBufferSeparateComplexComponents, &initPageSize, &locBufferNum, &locBufferSize, app->configuration.bufferNum, app->configuration.bufferSize, app->configuration.bufferSeparateComplexComponents, &axis->specializationConstants.outputBufferBlockSize, &axis->specializationConstants.outputBufferBlockNum);
                    }
                }
		}
		else {
            if ((inverse) && (axis_id == app->firstAxis) && (axis_upload_id == 0) && (app->configuration.isInputFormatted) && (app->configuration.inverseReturnToInputBuffer)) {				
                VkFFTSetBlockParameters(&axis->specializationConstants.outputBufferSeparateComplexComponents, &initPageSize, &locBufferNum, &locBufferSize, app->configuration.inputBufferNum, app->configuration.inputBufferSize, app->configuration.inputBufferSeparateComplexComponents, &axis->specializationConstants.outputBufferBlockSize, &axis->specializationConstants.outputBufferBlockNum);
            }
            else {
				if ((axis->specializationConstants.optimizePow2StridesTempBuffer) && (((axis_id < (app->configuration.FFTdim - 1)) && (inverse == 0) && (FFTPlan->numAxisUploads[axis_id + 1] == 1)) || ((axis_id > 0) && (inverse == 1) && (FFTPlan->numAxisUploads[axis_id - 1] == 1)))) {
					axis->specializationConstants.tempBufferOutput = 1;
					VkFFTSetBlockParameters(&axis->specializationConstants.outputBufferSeparateComplexComponents, &initPageSize, &locBufferNum, &locBufferSize, app->configuration.tempBufferNum, app->configuration.tempBufferSize, app->configuration.tempBufferSeparateComplexComponents, &axis->specializationConstants.outputBufferBlockSize, &axis->specializationConstants.outputBufferBlockNum);
				}
				else {
					VkFFTSetBlockParameters(&axis->specializationConstants.outputBufferSeparateComplexComponents, &initPageSize, &locBufferNum, &locBufferSize, app->configuration.bufferNum, app->configuration.bufferSize, app->configuration.bufferSeparateComplexComponents, &axis->specializationConstants.outputBufferBlockSize, &axis->specializationConstants.outputBufferBlockNum);
				}
            }
        }
	}

	if (axis->specializationConstants.inputBufferBlockNum == 0) axis->specializationConstants.inputBufferBlockNum = 1;
	if (axis->specializationConstants.outputBufferBlockNum == 0) axis->specializationConstants.outputBufferBlockNum = 1;

	initPageSize = -1;
	locBufferNum = 1;
	locBufferSize = -1;

	if (app->configuration.performConvolution) {
		VkFFTSetBlockParameters(&axis->specializationConstants.kernelSeparateComplexComponents, &initPageSize, &locBufferNum, &locBufferSize, app->configuration.kernelNum, app->configuration.kernelSize, app->configuration.kernelSeparateComplexComponents, &axis->specializationConstants.kernelBlockSize, &axis->specializationConstants.kernelBlockNum);
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
	if (app->configuration.usePushDescriptors)
		descriptorSetLayoutCreateInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;

	res = vkCreateDescriptorSetLayout(app->configuration.device[0], &descriptorSetLayoutCreateInfo, 0, &axis->descriptorSetLayout);
	if (res != VK_SUCCESS) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_CREATE_DESCRIPTOR_SET_LAYOUT;
	}
	free(descriptorSetLayoutBindings);
	descriptorSetLayoutBindings = 0;
	if (!app->configuration.usePushDescriptors) {
		VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
		descriptorSetAllocateInfo.descriptorPool = axis->descriptorPool;
		descriptorSetAllocateInfo.descriptorSetCount = 1;
		descriptorSetAllocateInfo.pSetLayouts = &axis->descriptorSetLayout;
		res = vkAllocateDescriptorSets(app->configuration.device[0], &descriptorSetAllocateInfo, &axis->descriptorSet);
		if (res != VK_SUCCESS) {
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_TO_ALLOCATE_DESCRIPTOR_SETS;
		}
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
				VkFFTSetBlockParameters(&axis->specializationConstants.inputBufferSeparateComplexComponents, &initPageSize, &locBufferNum, &locBufferSize, app->configuration.inputBufferNum, app->configuration.inputBufferSize, app->configuration.inputBufferSeparateComplexComponents, &axis->specializationConstants.inputBufferBlockSize, &axis->specializationConstants.inputBufferBlockNum);
			}
			else {
				if ((axis_upload_id == 0) && (app->configuration.numberKernels > 1) && (inverse) && (!app->configuration.performConvolution)) {
					VkFFTSetBlockParameters(&axis->specializationConstants.inputBufferSeparateComplexComponents, &initPageSize, &locBufferNum, &locBufferSize, app->configuration.outputBufferNum, app->configuration.outputBufferSize, app->configuration.outputBufferSeparateComplexComponents, &axis->specializationConstants.inputBufferBlockSize, &axis->specializationConstants.inputBufferBlockNum);
				}
				else {
					VkFFTSetBlockParameters(&axis->specializationConstants.inputBufferSeparateComplexComponents, &initPageSize, &locBufferNum, &locBufferSize, app->configuration.bufferNum, app->configuration.bufferSize, app->configuration.bufferSeparateComplexComponents, &axis->specializationConstants.inputBufferBlockSize, &axis->specializationConstants.inputBufferBlockNum);
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
				VkFFTSetBlockParameters(&axis->specializationConstants.inputBufferSeparateComplexComponents, &initPageSize, &locBufferNum, &locBufferSize, app->configuration.outputBufferNum, app->configuration.outputBufferSize, app->configuration.outputBufferSeparateComplexComponents, &axis->specializationConstants.inputBufferBlockSize, &axis->specializationConstants.inputBufferBlockNum);
			}
			else {
				VkFFTSetBlockParameters(&axis->specializationConstants.inputBufferSeparateComplexComponents, &initPageSize, &locBufferNum, &locBufferSize, app->configuration.bufferNum, app->configuration.bufferSize, app->configuration.bufferSeparateComplexComponents, &axis->specializationConstants.inputBufferBlockSize, &axis->specializationConstants.inputBufferBlockNum);
			}
		}
	}
	initPageSize = -1;
	locBufferNum = 1;
	locBufferSize = -1;
	{
		if (inverse) {
			if ((axis_upload_id == 0) && (app->configuration.numberKernels > 1) && (inverse) && (!app->configuration.performConvolution)) {
				VkFFTSetBlockParameters(&axis->specializationConstants.outputBufferSeparateComplexComponents, &initPageSize, &locBufferNum, &locBufferSize, app->configuration.outputBufferNum, app->configuration.outputBufferSize, app->configuration.outputBufferSeparateComplexComponents, &axis->specializationConstants.outputBufferBlockSize, &axis->specializationConstants.outputBufferBlockNum);
			}
			else {
				VkFFTSetBlockParameters(&axis->specializationConstants.outputBufferSeparateComplexComponents, &initPageSize, &locBufferNum, &locBufferSize, app->configuration.bufferNum, app->configuration.bufferSize, app->configuration.bufferSeparateComplexComponents, &axis->specializationConstants.outputBufferBlockSize, &axis->specializationConstants.outputBufferBlockNum);
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
				VkFFTSetBlockParameters(&axis->specializationConstants.outputBufferSeparateComplexComponents, &initPageSize, &locBufferNum, &locBufferSize, app->configuration.outputBufferNum, app->configuration.outputBufferSize, app->configuration.outputBufferSeparateComplexComponents, &axis->specializationConstants.outputBufferBlockSize, &axis->specializationConstants.outputBufferBlockNum);
			}
			else {
				VkFFTSetBlockParameters(&axis->specializationConstants.outputBufferSeparateComplexComponents, &initPageSize, &locBufferNum, &locBufferSize, app->configuration.bufferNum, app->configuration.bufferSize, app->configuration.bufferSeparateComplexComponents, &axis->specializationConstants.outputBufferBlockSize, &axis->specializationConstants.outputBufferBlockNum);
			}
		}
	}

	if (axis->specializationConstants.inputBufferBlockNum == 0) axis->specializationConstants.inputBufferBlockNum = 1;
	if (axis->specializationConstants.outputBufferBlockNum == 0) axis->specializationConstants.outputBufferBlockNum = 1;
	if (app->configuration.performConvolution) {
		//need fixing (not used now)
		VkFFTSetBlockParameters(&axis->specializationConstants.kernelSeparateComplexComponents, &initPageSize, &locBufferNum, &locBufferSize, app->configuration.kernelNum, app->configuration.kernelSize, app->configuration.kernelSeparateComplexComponents, &axis->specializationConstants.kernelBlockSize, &axis->specializationConstants.kernelBlockNum);
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
	if (app->configuration.usePushDescriptors)
		descriptorSetLayoutCreateInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;

	res = vkCreateDescriptorSetLayout(app->configuration.device[0], &descriptorSetLayoutCreateInfo, 0, &axis->descriptorSetLayout);
	if (res != VK_SUCCESS) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_CREATE_DESCRIPTOR_SET_LAYOUT;
	}
	free(descriptorSetLayoutBindings);
	descriptorSetLayoutBindings = 0;
	if (!app->configuration.usePushDescriptors) {
		VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
		descriptorSetAllocateInfo.descriptorPool = axis->descriptorPool;
		descriptorSetAllocateInfo.descriptorSetCount = 1;
		descriptorSetAllocateInfo.pSetLayouts = &axis->descriptorSetLayout;
		res = vkAllocateDescriptorSets(app->configuration.device[0], &descriptorSetAllocateInfo, &axis->descriptorSet);
		if (res != VK_SUCCESS) {
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_TO_ALLOCATE_DESCRIPTOR_SETS;
		}
	}
#endif
	return VKFFT_SUCCESS;
}
static inline VkFFTResult VkFFTCheckUpdateBufferSet(VkFFTApplication* app, VkFFTAxis* axis, pfUINT planStage, VkFFTLaunchParams* launchParams) {
	pfUINT performBufferSetUpdate = planStage;
	pfUINT performOffsetUpdate = planStage;
	if (!planStage) {
		if (launchParams != 0) {
			if (launchParams->buffer != 0) {
				for (pfUINT i = 0; i < app->configuration.bufferNum; i++) {
					if (app->configuration.buffer[i] != launchParams->buffer[i]) {
#if(VKFFT_BACKEND==0)
						memcpy((void*)&app->configuration.buffer[i], (const void*)&launchParams->buffer[i], sizeof(const VkBuffer));
#elif((VKFFT_BACKEND==1) || (VKFFT_BACKEND==2) || (VKFFT_BACKEND==4))
						memcpy((void*)&app->configuration.buffer[i], (const void*)&launchParams->buffer[i], sizeof(void* const));
#elif(VKFFT_BACKEND==3)
						memcpy((void*)&app->configuration.buffer[i], (const void*)&launchParams->buffer[i], sizeof(const cl_mem));
#elif(VKFFT_BACKEND==5)
						memcpy((void*)&app->configuration.buffer[i], (const void*)&launchParams->buffer[i], sizeof(MTL::Buffer* const));
#endif
						performBufferSetUpdate = 1;
					}
				}
			}
			if (launchParams->inputBuffer != 0) {
				for (pfUINT i = 0; i < app->configuration.inputBufferNum; i++) {
					if (app->configuration.inputBuffer[i] != launchParams->inputBuffer[i]) {
#if(VKFFT_BACKEND==0)
						memcpy((void*)&app->configuration.inputBuffer[i], (const void*)&launchParams->inputBuffer[i], sizeof(const VkBuffer));
#elif((VKFFT_BACKEND==1) || (VKFFT_BACKEND==2) || (VKFFT_BACKEND==4))
						memcpy((void*)&app->configuration.inputBuffer[i], (const void*)&launchParams->inputBuffer[i], sizeof(void* const));
#elif(VKFFT_BACKEND==3)
						memcpy((void*)&app->configuration.inputBuffer[i], (const void*)&launchParams->inputBuffer[i], sizeof(const cl_mem));
#elif(VKFFT_BACKEND==5)
						memcpy((void*)&app->configuration.inputBuffer[i], (const void*)&launchParams->inputBuffer[i], sizeof(MTL::Buffer* const));
#endif
						performBufferSetUpdate = 1;
					}
				}
			}
			if (launchParams->outputBuffer != 0) {
				for (pfUINT i = 0; i < app->configuration.outputBufferNum; i++) {
					if (app->configuration.outputBuffer[i] != launchParams->outputBuffer[i]) {
#if(VKFFT_BACKEND==0)
						memcpy((void*)&app->configuration.outputBuffer[i], (const void*)&launchParams->outputBuffer[i], sizeof(const VkBuffer));
#elif((VKFFT_BACKEND==1) || (VKFFT_BACKEND==2) || (VKFFT_BACKEND==4))
						memcpy((void*)&app->configuration.outputBuffer[i], (const void*)&launchParams->outputBuffer[i], sizeof(void* const));
#elif(VKFFT_BACKEND==3)
						memcpy((void*)&app->configuration.outputBuffer[i], (const void*)&launchParams->outputBuffer[i], sizeof(const cl_mem));
#elif(VKFFT_BACKEND==5)
						memcpy((void*)&app->configuration.outputBuffer[i], (const void*)&launchParams->outputBuffer[i], sizeof(MTL::Buffer* const));
#endif
						performBufferSetUpdate = 1;
					}
				}
			}
			if (launchParams->tempBuffer != 0) {
				for (pfUINT i = 0; i < app->configuration.tempBufferNum; i++) {
					if (app->configuration.tempBuffer[i] != launchParams->tempBuffer[i]) {
#if(VKFFT_BACKEND==0)
						memcpy((void*)&app->configuration.tempBuffer[i], (const void*)&launchParams->tempBuffer[i], sizeof(VkBuffer));
#elif((VKFFT_BACKEND==1) || (VKFFT_BACKEND==2) || (VKFFT_BACKEND==4))
						memcpy((void*)&app->configuration.tempBuffer[i], (const void*)&launchParams->tempBuffer[i], sizeof(void*));
#elif(VKFFT_BACKEND==3)
						memcpy((void*)&app->configuration.tempBuffer[i], (const void*)&launchParams->tempBuffer[i], sizeof(cl_mem));
#elif(VKFFT_BACKEND==5)
						memcpy((void*)&app->configuration.tempBuffer[i], (const void*)&launchParams->tempBuffer[i], sizeof(MTL::Buffer*));
#endif
						performBufferSetUpdate = 1;
					}
				}
			}
			if (launchParams->kernel != 0) {
				for (pfUINT i = 0; i < app->configuration.kernelNum; i++) {
					if (app->configuration.kernel[i] != launchParams->kernel[i]) {
#if(VKFFT_BACKEND==0)
						memcpy((void*)&app->configuration.kernel[i], (const void*)&launchParams->kernel[i], sizeof(const VkBuffer));
#elif((VKFFT_BACKEND==1) || (VKFFT_BACKEND==2) || (VKFFT_BACKEND==4))
						memcpy((void*)&app->configuration.kernel[i], (const void*)&launchParams->kernel[i], sizeof(void* const));
#elif(VKFFT_BACKEND==3)
						memcpy((void*)&app->configuration.kernel[i], (const void*)&launchParams->kernel[i], sizeof(const cl_mem));
#elif(VKFFT_BACKEND==5)
						memcpy((void*)&app->configuration.kernel[i], (const void*)&launchParams->kernel[i], sizeof(MTL::Buffer* const));
#endif
						performBufferSetUpdate = 1;
					}
				}
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

			if (app->configuration.bufferOffsetImaginary != launchParams->bufferOffsetImaginary) {
				app->configuration.bufferOffsetImaginary = launchParams->bufferOffsetImaginary;
				performOffsetUpdate = 1;
			}
			if (app->configuration.inputBufferOffsetImaginary != launchParams->inputBufferOffsetImaginary) {
				app->configuration.inputBufferOffsetImaginary = launchParams->inputBufferOffsetImaginary;
				performOffsetUpdate = 1;
			}
			if (app->configuration.outputBufferOffsetImaginary != launchParams->outputBufferOffsetImaginary) {
				app->configuration.outputBufferOffsetImaginary = launchParams->outputBufferOffsetImaginary;
				performOffsetUpdate = 1;
			}
			if (app->configuration.tempBufferOffsetImaginary != launchParams->tempBufferOffsetImaginary) {
				app->configuration.tempBufferOffsetImaginary = launchParams->tempBufferOffsetImaginary;
				performOffsetUpdate = 1;
			}
			if (app->configuration.kernelOffsetImaginary != launchParams->kernelOffsetImaginary) {
				app->configuration.kernelOffsetImaginary = launchParams->kernelOffsetImaginary;
				performOffsetUpdate = 1;
			}
		}
	}
	if (planStage) {
		for (pfUINT i = 0; i < app->configuration.bufferNum; i++) {
			if (app->configuration.buffer[i] == 0) {
				performBufferSetUpdate = 0;
			}
		}
		if (app->configuration.isInputFormatted) {
			for (pfUINT i = 0; i < app->configuration.inputBufferNum; i++) {
				if (app->configuration.inputBuffer[i] == 0) {
					performBufferSetUpdate = 0;
				}
			}
		}
		if (app->configuration.isOutputFormatted) {
			for (pfUINT i = 0; i < app->configuration.outputBufferNum; i++) {
				if (app->configuration.outputBuffer[i] == 0) {
					performBufferSetUpdate = 0;
				}
			}
		}
		if ((app->configuration.userTempBuffer) || (app->configuration.allocateTempBuffer)){
			for (pfUINT i = 0; i < app->configuration.tempBufferNum; i++) {
				if (app->configuration.tempBuffer[i] == 0) {
					performBufferSetUpdate = 0;
				}
			}
		}
		if (app->configuration.performConvolution) {
			for (pfUINT i = 0; i < app->configuration.kernelNum; i++) {
				if (app->configuration.kernel[i] == 0) {
					performBufferSetUpdate = 0;
				}
			}
		}
#if(VKFFT_BACKEND==0)
		if (app->configuration.usePushDescriptors) {
			performBufferSetUpdate = 0;
		}
#endif
	}
	else {
		for (pfUINT i = 0; i < app->configuration.bufferNum; i++) {
			if (app->configuration.buffer[i] == 0) {
				return VKFFT_ERROR_EMPTY_buffer;
			}
		}
		if (app->configuration.isInputFormatted) {
			for (pfUINT i = 0; i < app->configuration.inputBufferNum; i++) {
				if (app->configuration.inputBuffer[i] == 0) {
					return VKFFT_ERROR_EMPTY_inputBuffer;
				}
			}
		}
		if (app->configuration.isOutputFormatted) {
			for (pfUINT i = 0; i < app->configuration.outputBufferNum; i++) {
				if (app->configuration.outputBuffer[i] == 0) {
					return VKFFT_ERROR_EMPTY_outputBuffer;
				}
			}
		}
		if ((app->configuration.userTempBuffer) || (app->configuration.allocateTempBuffer)){
			for (pfUINT i = 0; i < app->configuration.tempBufferNum; i++) {
				if (app->configuration.tempBuffer[i] == 0) {
					return VKFFT_ERROR_EMPTY_tempBuffer;
				}
			}
		}
		if (app->configuration.performConvolution) {
			for (pfUINT i = 0; i < app->configuration.kernelNum; i++) {
				if (app->configuration.kernel[i] == 0) {
					return VKFFT_ERROR_EMPTY_kernel;
				}
			}
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
static inline void VkFFTSetBufferParameters(void*const** axisBuffer, pfUINT* axisBufferNum, void* const* appBuffer, pfUINT bufferID, pfUINT bufferNum, pfUINT* bufferSize, pfUINT* bufferBlockSize, pfUINT separateComplexComponents, void* descriptorBufferInfo) {
	if(separateComplexComponents){
		axisBuffer[0] = appBuffer;
		axisBufferNum[0] = bufferNum;
#if(VKFFT_BACKEND==0)
		VkDescriptorBufferInfo* localDescriptorBufferInfo = (VkDescriptorBufferInfo*) descriptorBufferInfo; 
		localDescriptorBufferInfo->buffer = ((VkBuffer*)appBuffer)[bufferID];
		localDescriptorBufferInfo->range = (bufferSize[bufferID]);
		localDescriptorBufferInfo->offset = 0;
#endif
	}
	else
	{
		pfUINT localBufferID = 0;
		pfUINT offset = bufferID;
		if (bufferSize)
		{
			for (pfUINT l = 0; l < bufferNum; ++l) {
				if (offset >= (pfUINT)pfceil(bufferSize[l] / (double)(bufferBlockSize[0]))) {
					localBufferID++;
					offset -= (pfUINT)pfceil(bufferSize[l] / (double)(bufferBlockSize[0]));
				}
				else {
					l = bufferNum;
				}

			}
		}
		axisBuffer[0] = appBuffer;
		axisBufferNum[0] = bufferNum;
#if(VKFFT_BACKEND==0)
		VkDescriptorBufferInfo* localDescriptorBufferInfo = (VkDescriptorBufferInfo*) descriptorBufferInfo; 
		localDescriptorBufferInfo->buffer = ((const VkBuffer*)appBuffer)[localBufferID];
		localDescriptorBufferInfo->range = (bufferBlockSize[0]);
		localDescriptorBufferInfo->offset = offset * (bufferBlockSize[0]);
#endif
	}
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
#else
				int descriptorBufferInfo = 0;
#endif
				if (i == 0) {
					if ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (app->configuration.isInputFormatted) && (!axis->specializationConstants.reverseBluesteinMultiUpload) && (
						((axis_id == app->firstAxis) && (!inverse))
						|| ((axis_id == app->lastAxis) && (inverse) && (!((axis_id == 0) && (axis->specializationConstants.performR2CmultiUpload))) && (!app->configuration.performConvolution) && (!app->configuration.inverseReturnToInputBuffer)))
						) {
						if (axis->specializationConstants.performBufferSetUpdate) {
							VkFFTSetBufferParameters((void*const**)&axis->inputBuffer, &axis->specializationConstants.inputBufferNum, (void*const*)app->configuration.inputBuffer, j, app->configuration.inputBufferNum, app->configuration.inputBufferSize, &axis->specializationConstants.inputBufferBlockSize, axis->specializationConstants.inputBufferSeparateComplexComponents, &descriptorBufferInfo);
						}
						if (axis->specializationConstants.performOffsetUpdate) {
							axis->specializationConstants.inputOffset.data.i = app->configuration.inputBufferOffset;
							axis->specializationConstants.inputOffsetImaginary.data.i = app->configuration.inputBufferOffsetImaginary;
						}
					}
					else {
						if ((axis_upload_id == 0) && (app->configuration.numberKernels > 1) && (inverse) && (!app->configuration.performConvolution)) {
							if (axis->specializationConstants.performBufferSetUpdate) {
								VkFFTSetBufferParameters((void*const**)&axis->inputBuffer, &axis->specializationConstants.inputBufferNum, (void*const*)app->configuration.outputBuffer, j, app->configuration.outputBufferNum, app->configuration.outputBufferSize, &axis->specializationConstants.inputBufferBlockSize, axis->specializationConstants.inputBufferSeparateComplexComponents, &descriptorBufferInfo);
							}
							if (axis->specializationConstants.performOffsetUpdate) {
								axis->specializationConstants.inputOffset.data.i = app->configuration.outputBufferOffset;
								axis->specializationConstants.inputOffsetImaginary.data.i = app->configuration.outputBufferOffsetImaginary;
							}
						}
						else {
							if (((axis->specializationConstants.reorderFourStep) || (app->useBluesteinFFT[axis_id])) && (FFTPlan->numAxisUploads[axis_id] > 1)) {
								int parity_reorderFourStep_startBuffer = 0;// ((axis_id == 0) && (FFTPlan->bigSequenceEvenR2C) && (axis->specializationConstants.reorderFourStep == 2) && (inverse == 1)) ? 1 : 0;
								if ((((axis->specializationConstants.reorderFourStep == 1) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1)) || ((axis->specializationConstants.reorderFourStep == 2) && (((FFTPlan->numAxisUploads[axis_id] - 1 - axis_upload_id)%2) == parity_reorderFourStep_startBuffer)) || (app->useBluesteinFFT[axis_id] && (axis->specializationConstants.reverseBluesteinMultiUpload == 0) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1)))) {
									if (axis->specializationConstants.performBufferSetUpdate) {
										VkFFTSetBufferParameters((void*const**)&axis->inputBuffer, &axis->specializationConstants.inputBufferNum, (void*const*)app->configuration.buffer, j, app->configuration.bufferNum, app->configuration.bufferSize, &axis->specializationConstants.inputBufferBlockSize, axis->specializationConstants.inputBufferSeparateComplexComponents, &descriptorBufferInfo);
									}
									if (axis->specializationConstants.performOffsetUpdate) {
										axis->specializationConstants.inputOffset.data.i = app->configuration.bufferOffset;
										axis->specializationConstants.inputOffsetImaginary.data.i = app->configuration.bufferOffsetImaginary;
									}
								}
								else {
									if (axis->specializationConstants.performBufferSetUpdate) {
										VkFFTSetBufferParameters((void*const**)&axis->inputBuffer, &axis->specializationConstants.inputBufferNum, (void*const*)app->configuration.tempBuffer, j, app->configuration.tempBufferNum, app->configuration.tempBufferSize, &axis->specializationConstants.inputBufferBlockSize, axis->specializationConstants.inputBufferSeparateComplexComponents, &descriptorBufferInfo);
									}
									if (axis->specializationConstants.performOffsetUpdate) {
										axis->specializationConstants.inputOffset.data.i = app->configuration.tempBufferOffset;
										axis->specializationConstants.inputOffsetImaginary.data.i = app->configuration.tempBufferOffsetImaginary;
									}
								}
							}
							else {
								if ((axis->specializationConstants.optimizePow2StridesTempBuffer) && (((axis_id > 0) && (inverse == 0) && (FFTPlan->numAxisUploads[axis_id - 1] == 1)) || ((axis_id < (app->configuration.FFTdim - 1)) && (inverse == 1) && (FFTPlan->numAxisUploads[axis_id + 1] == 1)))) {
									if (axis->specializationConstants.performBufferSetUpdate) {
										VkFFTSetBufferParameters((void*const**)&axis->inputBuffer, &axis->specializationConstants.inputBufferNum, (void*const*)app->configuration.tempBuffer, j, app->configuration.tempBufferNum, app->configuration.tempBufferSize, &axis->specializationConstants.inputBufferBlockSize, axis->specializationConstants.inputBufferSeparateComplexComponents, &descriptorBufferInfo);
									}
									if (axis->specializationConstants.performOffsetUpdate) {
										axis->specializationConstants.inputOffset.data.i = app->configuration.tempBufferOffset;
										axis->specializationConstants.inputOffsetImaginary.data.i = app->configuration.tempBufferOffsetImaginary;
									}
								}
								else{
									if (axis->specializationConstants.performBufferSetUpdate) {
										VkFFTSetBufferParameters((void* const**)&axis->inputBuffer, &axis->specializationConstants.inputBufferNum, (void* const*)app->configuration.buffer, j, app->configuration.bufferNum, app->configuration.bufferSize, &axis->specializationConstants.inputBufferBlockSize, axis->specializationConstants.inputBufferSeparateComplexComponents, &descriptorBufferInfo);
									}
									if (axis->specializationConstants.performOffsetUpdate) {
										axis->specializationConstants.inputOffset.data.i = app->configuration.bufferOffset;
										axis->specializationConstants.inputOffsetImaginary.data.i = app->configuration.bufferOffsetImaginary;
									}
								}
							}
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
							VkFFTSetBufferParameters((void*const**)&axis->outputBuffer, &axis->specializationConstants.outputBufferNum, (void*const*)app->configuration.outputBuffer, j, app->configuration.outputBufferNum, app->configuration.outputBufferSize, &axis->specializationConstants.outputBufferBlockSize, axis->specializationConstants.outputBufferSeparateComplexComponents, &descriptorBufferInfo);
						}
						if (axis->specializationConstants.performOffsetUpdate) {
							axis->specializationConstants.outputOffset.data.i = app->configuration.outputBufferOffset;
							axis->specializationConstants.outputOffsetImaginary.data.i = app->configuration.outputBufferOffsetImaginary;
						}
					}
					else {
						if (((axis->specializationConstants.reorderFourStep) || (app->useBluesteinFFT[axis_id])) && (FFTPlan->numAxisUploads[axis_id] > 1)) {
							if ((inverse) && (axis_id == app->firstAxis) && (
								((axis_upload_id == 0) && (app->configuration.isInputFormatted) && (app->configuration.inverseReturnToInputBuffer) && (!app->useBluesteinFFT[axis_id]))
								|| ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (app->configuration.isInputFormatted) && (axis->specializationConstants.actualInverse) && (app->configuration.inverseReturnToInputBuffer) && (app->useBluesteinFFT[axis_id]) && (axis->specializationConstants.reverseBluesteinMultiUpload || (FFTPlan->numAxisUploads[axis_id] == 1))))
								) {
								if (axis->specializationConstants.performBufferSetUpdate) {
									VkFFTSetBufferParameters((void*const**)&axis->outputBuffer, &axis->specializationConstants.outputBufferNum, (void*const*)app->configuration.inputBuffer, j, app->configuration.inputBufferNum, app->configuration.inputBufferSize, &axis->specializationConstants.outputBufferBlockSize, axis->specializationConstants.outputBufferSeparateComplexComponents, &descriptorBufferInfo);
								}
								if (axis->specializationConstants.performOffsetUpdate) {
									axis->specializationConstants.outputOffset.data.i = app->configuration.inputBufferOffset;
									axis->specializationConstants.outputOffsetImaginary.data.i = app->configuration.inputBufferOffsetImaginary;
								}
							}
							else {
								int parity_reorderFourStep_startBuffer = 0;// ((axis_id == 0) && (FFTPlan->bigSequenceEvenR2C) && (axis->specializationConstants.reorderFourStep == 2) && (inverse == 1)) ? 1 : 0;
								if (((axis->specializationConstants.reorderFourStep == 1) && (axis_upload_id > 0)) || ((axis->specializationConstants.reorderFourStep == 2) && ((((FFTPlan->numAxisUploads[axis_id] - 1 - axis_upload_id)%2) == parity_reorderFourStep_startBuffer) && (axis_upload_id != 0))) || (app->useBluesteinFFT[axis_id] && (!((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (axis->specializationConstants.reverseBluesteinMultiUpload == 1))))) {
									if (axis->specializationConstants.performBufferSetUpdate) {
										VkFFTSetBufferParameters((void*const**)&axis->outputBuffer, &axis->specializationConstants.outputBufferNum, (void*const*)app->configuration.tempBuffer, j, app->configuration.tempBufferNum, app->configuration.tempBufferSize, &axis->specializationConstants.outputBufferBlockSize, axis->specializationConstants.outputBufferSeparateComplexComponents, &descriptorBufferInfo);
									}
									if (axis->specializationConstants.performOffsetUpdate) {
										axis->specializationConstants.outputOffset.data.i = app->configuration.tempBufferOffset;
										axis->specializationConstants.outputOffsetImaginary.data.i = app->configuration.tempBufferOffsetImaginary;
									}
								}
								else {
									if (axis->specializationConstants.performBufferSetUpdate) {
										VkFFTSetBufferParameters((void*const**)&axis->outputBuffer, &axis->specializationConstants.outputBufferNum, (void*const*)app->configuration.buffer, j, app->configuration.bufferNum, app->configuration.bufferSize, &axis->specializationConstants.outputBufferBlockSize, axis->specializationConstants.outputBufferSeparateComplexComponents, &descriptorBufferInfo);
									}
									if (axis->specializationConstants.performOffsetUpdate) {
										axis->specializationConstants.outputOffset.data.i = app->configuration.bufferOffset;
										axis->specializationConstants.outputOffsetImaginary.data.i = app->configuration.bufferOffsetImaginary;
									}
								}
							}
						}
						else {
							if ((inverse) && (axis_id == app->firstAxis) && (axis_upload_id == 0) && (app->configuration.isInputFormatted) && (app->configuration.inverseReturnToInputBuffer)) {
								if (axis->specializationConstants.performBufferSetUpdate) {
									VkFFTSetBufferParameters((void*const**)&axis->outputBuffer, &axis->specializationConstants.outputBufferNum, (void*const*)app->configuration.inputBuffer, j, app->configuration.inputBufferNum, app->configuration.inputBufferSize, &axis->specializationConstants.outputBufferBlockSize, axis->specializationConstants.outputBufferSeparateComplexComponents, &descriptorBufferInfo);
								}
								if (axis->specializationConstants.performOffsetUpdate) {
									axis->specializationConstants.outputOffset.data.i = app->configuration.inputBufferOffset;
									axis->specializationConstants.outputOffsetImaginary.data.i = app->configuration.inputBufferOffsetImaginary;
								}
							}
							else {
								if ((axis->specializationConstants.optimizePow2StridesTempBuffer) && (((axis_id < (app->configuration.FFTdim - 1)) && (inverse == 0) && (FFTPlan->numAxisUploads[axis_id + 1] == 1)) || ((axis_id > 0) && (inverse == 1) && (FFTPlan->numAxisUploads[axis_id - 1] == 1)))) {
									if (axis->specializationConstants.performBufferSetUpdate) {
										VkFFTSetBufferParameters((void*const**)&axis->outputBuffer, &axis->specializationConstants.outputBufferNum, (void*const*)app->configuration.tempBuffer, j, app->configuration.tempBufferNum, app->configuration.tempBufferSize, &axis->specializationConstants.outputBufferBlockSize, axis->specializationConstants.outputBufferSeparateComplexComponents, &descriptorBufferInfo);
									}
									if (axis->specializationConstants.performOffsetUpdate) {
										axis->specializationConstants.outputOffset.data.i = app->configuration.tempBufferOffset;
										axis->specializationConstants.outputOffsetImaginary.data.i = app->configuration.tempBufferOffsetImaginary;
									}
								}
								else {
									if (axis->specializationConstants.performBufferSetUpdate) {
										VkFFTSetBufferParameters((void* const**)&axis->outputBuffer, &axis->specializationConstants.outputBufferNum, (void* const*)app->configuration.buffer, j, app->configuration.bufferNum, app->configuration.bufferSize, &axis->specializationConstants.outputBufferBlockSize, axis->specializationConstants.outputBufferSeparateComplexComponents, &descriptorBufferInfo);
									}
									if (axis->specializationConstants.performOffsetUpdate) {
										axis->specializationConstants.outputOffset.data.i = app->configuration.bufferOffset;
										axis->specializationConstants.outputOffsetImaginary.data.i = app->configuration.bufferOffsetImaginary;
									}
								}
							}
						}
					}
					//descriptorBufferInfo.offset = 0;
				}
				if ((i == axis->specializationConstants.convolutionBindingID) && (app->configuration.performConvolution)) {
					if (axis->specializationConstants.performBufferSetUpdate) {
						VkFFTSetBufferParameters((void*const**)&axis->kernel, &axis->specializationConstants.kernelNum, (void*const*)app->configuration.kernel, j, app->configuration.kernelNum, app->configuration.kernelSize, &axis->specializationConstants.kernelBlockSize, axis->specializationConstants.kernelSeparateComplexComponents, &descriptorBufferInfo);
					}
					if (axis->specializationConstants.performOffsetUpdate) {
						axis->specializationConstants.kernelOffset.data.i = app->configuration.kernelOffset;
						axis->specializationConstants.kernelOffsetImaginary.data.i = app->configuration.kernelOffsetImaginary;
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
				if ((axis->specializationConstants.performBufferSetUpdate) && (descriptorBufferInfo.buffer != VK_NULL_HANDLE)) {
					VkWriteDescriptorSet writeDescriptorSet = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
					writeDescriptorSet.dstSet = axis->descriptorSet;
					writeDescriptorSet.dstBinding = (uint32_t)i;
					writeDescriptorSet.dstArrayElement = (uint32_t)j;
					writeDescriptorSet.descriptorType = descriptorType;
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					if (app->configuration.usePushDescriptors){
						PFN_vkCmdPushDescriptorSetKHR vkCmdPushDescriptorSetKHR = VKFFT_ZERO_INIT;
						vkCmdPushDescriptorSetKHR = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(app->configuration.device[0], "vkCmdPushDescriptorSetKHR");
						if (!vkCmdPushDescriptorSetKHR) {
							return VKFFT_SUCCESS;
						}
						vkCmdPushDescriptorSetKHR(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &writeDescriptorSet);
					}
					else{
						vkUpdateDescriptorSets(app->configuration.device[0], 1, &writeDescriptorSet, 0, 0);
					}
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
#else
				int descriptorBufferInfo = 0;
#endif
				if (i == 0) {
					if (inverse) {
						if ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (app->configuration.isInputFormatted) && (!axis->specializationConstants.reverseBluesteinMultiUpload) && (
							((axis_id == app->firstAxis) && (!inverse))
							|| ((axis_id == app->lastAxis) && (inverse) && (!app->configuration.performConvolution) && (!app->configuration.inverseReturnToInputBuffer)))
							) {
							if (axis->specializationConstants.performBufferSetUpdate) {
								VkFFTSetBufferParameters((void*const**)&axis->inputBuffer, &axis->specializationConstants.inputBufferNum, (void*const*)app->configuration.inputBuffer, j, app->configuration.inputBufferNum, app->configuration.inputBufferSize, &axis->specializationConstants.inputBufferBlockSize, axis->specializationConstants.inputBufferSeparateComplexComponents, &descriptorBufferInfo);
							}
							if (axis->specializationConstants.performOffsetUpdate) {
								axis->specializationConstants.inputOffset.data.i = app->configuration.inputBufferOffset;
								axis->specializationConstants.inputOffsetImaginary.data.i = app->configuration.inputBufferOffsetImaginary;
							}
						}
						else {
							if ((axis_upload_id == 0) && (app->configuration.numberKernels > 1) && (inverse) && (!app->configuration.performConvolution)) {
								if (axis->specializationConstants.performBufferSetUpdate) {
									VkFFTSetBufferParameters((void*const**)&axis->inputBuffer, &axis->specializationConstants.inputBufferNum, (void*const*)app->configuration.outputBuffer, j, app->configuration.outputBufferNum, app->configuration.outputBufferSize, &axis->specializationConstants.inputBufferBlockSize, axis->specializationConstants.inputBufferSeparateComplexComponents, &descriptorBufferInfo);
								}
								if (axis->specializationConstants.performOffsetUpdate) {
									axis->specializationConstants.inputOffset.data.i = app->configuration.outputBufferOffset;
									axis->specializationConstants.inputOffsetImaginary.data.i = app->configuration.outputBufferOffsetImaginary;
								}
							}
							else {
								if (axis->specializationConstants.performBufferSetUpdate) {
									VkFFTSetBufferParameters((void*const**)&axis->inputBuffer, &axis->specializationConstants.inputBufferNum, (void*const*)app->configuration.buffer, j, app->configuration.bufferNum, app->configuration.bufferSize, &axis->specializationConstants.inputBufferBlockSize, axis->specializationConstants.inputBufferSeparateComplexComponents, &descriptorBufferInfo);
								}
								if (axis->specializationConstants.performOffsetUpdate) {
									axis->specializationConstants.inputOffset.data.i = app->configuration.bufferOffset;
									axis->specializationConstants.inputOffsetImaginary.data.i = app->configuration.bufferOffsetImaginary;
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
								VkFFTSetBufferParameters((void*const**)&axis->inputBuffer, &axis->specializationConstants.inputBufferNum, (void*const*)app->configuration.outputBuffer, j, app->configuration.outputBufferNum, app->configuration.outputBufferSize, &axis->specializationConstants.inputBufferBlockSize, axis->specializationConstants.inputBufferSeparateComplexComponents, &descriptorBufferInfo);
							}
							if (axis->specializationConstants.performOffsetUpdate) {
								axis->specializationConstants.inputOffset.data.i = app->configuration.outputBufferOffset;
								axis->specializationConstants.inputOffsetImaginary.data.i = app->configuration.outputBufferOffsetImaginary;
							}
						}
						else {
							if (axis->specializationConstants.performBufferSetUpdate) {
								VkFFTSetBufferParameters((void*const**)&axis->inputBuffer, &axis->specializationConstants.inputBufferNum, (void*const*)app->configuration.buffer, j, app->configuration.bufferNum, app->configuration.bufferSize, &axis->specializationConstants.inputBufferBlockSize, axis->specializationConstants.inputBufferSeparateComplexComponents, &descriptorBufferInfo);
							}
							if (axis->specializationConstants.performOffsetUpdate) {
								axis->specializationConstants.inputOffset.data.i = app->configuration.bufferOffset;
								axis->specializationConstants.inputOffsetImaginary.data.i = app->configuration.bufferOffsetImaginary;
							}
						}
					}
				}
				if (i == 1) {
					if (inverse) {
						if ((axis_upload_id == 0) && (app->configuration.numberKernels > 1) && (inverse) && (!app->configuration.performConvolution)) {
							if (axis->specializationConstants.performBufferSetUpdate) {
								VkFFTSetBufferParameters((void*const**)&axis->outputBuffer, &axis->specializationConstants.outputBufferNum, (void*const*)app->configuration.outputBuffer, j, app->configuration.outputBufferNum, app->configuration.outputBufferSize, &axis->specializationConstants.outputBufferBlockSize, axis->specializationConstants.outputBufferSeparateComplexComponents, &descriptorBufferInfo);
							}
							if (axis->specializationConstants.performOffsetUpdate) {
								axis->specializationConstants.outputOffset.data.i = app->configuration.outputBufferOffset;
								axis->specializationConstants.outputOffsetImaginary.data.i = app->configuration.outputBufferOffsetImaginary;
							}
						}
						else {
							pfUINT bufferId = 0;
							pfUINT offset = j;
							if (axis->specializationConstants.performBufferSetUpdate) {
								VkFFTSetBufferParameters((void*const**)&axis->outputBuffer, &axis->specializationConstants.outputBufferNum, (void*const*)app->configuration.buffer, j, app->configuration.bufferNum, app->configuration.bufferSize, &axis->specializationConstants.outputBufferBlockSize, axis->specializationConstants.outputBufferSeparateComplexComponents, &descriptorBufferInfo);
							}
							if (axis->specializationConstants.performOffsetUpdate) {
								axis->specializationConstants.outputOffset.data.i = app->configuration.bufferOffset;
								axis->specializationConstants.outputOffsetImaginary.data.i = app->configuration.bufferOffsetImaginary;
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
								VkFFTSetBufferParameters((void*const**)&axis->outputBuffer, &axis->specializationConstants.outputBufferNum, (void*const*)app->configuration.outputBuffer, j, app->configuration.outputBufferNum, app->configuration.outputBufferSize, &axis->specializationConstants.outputBufferBlockSize, axis->specializationConstants.outputBufferSeparateComplexComponents, &descriptorBufferInfo);
							}
							if (axis->specializationConstants.performOffsetUpdate) {
								axis->specializationConstants.outputOffset.data.i = app->configuration.outputBufferOffset;
								axis->specializationConstants.outputOffsetImaginary.data.i = app->configuration.outputBufferOffsetImaginary;
							}
						}
						else {
							if (axis->specializationConstants.performBufferSetUpdate) {
								VkFFTSetBufferParameters((void*const**)&axis->outputBuffer, &axis->specializationConstants.outputBufferNum, (void*const*)app->configuration.buffer, j, app->configuration.bufferNum, app->configuration.bufferSize, &axis->specializationConstants.outputBufferBlockSize, axis->specializationConstants.outputBufferSeparateComplexComponents, &descriptorBufferInfo);
							}
							if (axis->specializationConstants.performOffsetUpdate) {
								axis->specializationConstants.outputOffset.data.i = app->configuration.bufferOffset;
								axis->specializationConstants.outputOffsetImaginary.data.i = app->configuration.bufferOffsetImaginary;
							}
						}
					}
				}
				if ((i == 2) && (app->configuration.performConvolution)) {
					if (axis->specializationConstants.performBufferSetUpdate) {
						VkFFTSetBufferParameters((void*const**)&axis->kernel, &axis->specializationConstants.kernelNum, (void*const*)app->configuration.kernel, j, app->configuration.kernelNum, app->configuration.kernelSize, &axis->specializationConstants.kernelBlockSize, axis->specializationConstants.kernelSeparateComplexComponents, &descriptorBufferInfo);
					}
					if (axis->specializationConstants.performOffsetUpdate) {
						axis->specializationConstants.kernelOffset.data.i = app->configuration.kernelOffset;
						axis->specializationConstants.kernelOffsetImaginary.data.i = app->configuration.kernelOffsetImaginary;
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
					if (app->configuration.usePushDescriptors){
						PFN_vkCmdPushDescriptorSetKHR vkCmdPushDescriptorSetKHR = VKFFT_ZERO_INIT;
						vkCmdPushDescriptorSetKHR = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(app->configuration.device[0], "vkCmdPushDescriptorSetKHR");
						if (!vkCmdPushDescriptorSetKHR) {
							return VKFFT_SUCCESS;
						}
						vkCmdPushDescriptorSetKHR(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &writeDescriptorSet);
					}
					else{
						vkUpdateDescriptorSets(app->configuration.device[0], 1, &writeDescriptorSet, 0, 0);
					}
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
