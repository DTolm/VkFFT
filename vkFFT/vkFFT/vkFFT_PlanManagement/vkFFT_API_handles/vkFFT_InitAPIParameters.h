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
#ifndef VKFFT_INITAPIPARAMETERS_H
#define VKFFT_INITAPIPARAMETERS_H
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_StringManagement/vkFFT_StringManager.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_MathUtils/vkFFT_MathUtils.h"

static inline VkFFTResult initMemoryParametersAPI(VkFFTApplication* app, VkFFTSpecializationConstantsLayout* sc) {
	VkFFTResult res = VKFFT_SUCCESS;

	PfAllocateContainerFlexible(sc, &sc->halfLiteral, 50);
	sc->halfLiteral.type = 100;	
	PfAllocateContainerFlexible(sc, &sc->floatLiteral, 50);
	sc->floatLiteral.type = 110;	
	PfAllocateContainerFlexible(sc, &sc->doubleLiteral, 50);
	sc->doubleLiteral.type = 120;
	PfAllocateContainerFlexible(sc, &sc->halfDef, 50);
	sc->halfDef.type = 100;
	PfAllocateContainerFlexible(sc, &sc->floatDef, 50);
	sc->floatDef.type = 110;
	PfAllocateContainerFlexible(sc, &sc->doubleDef, 50);
	sc->doubleDef.type = 120;
	PfAllocateContainerFlexible(sc, &sc->half2Def, 50);
	sc->half2Def.type = 100;
	PfAllocateContainerFlexible(sc, &sc->float2Def, 50);
	sc->float2Def.type = 110;
	PfAllocateContainerFlexible(sc, &sc->double2Def, 50);
	sc->double2Def.type = 120;

	PfAllocateContainerFlexible(sc, &sc->uintDef, 50);
	sc->uintDef.type = 100;
	PfAllocateContainerFlexible(sc, &sc->intDef, 50);
	sc->intDef.type = 110;
	PfAllocateContainerFlexible(sc, &sc->uint64Def, 50);
	sc->uint64Def.type = 120;
	PfAllocateContainerFlexible(sc, &sc->int64Def, 50);
	sc->int64Def.type = 130;

#if(VKFFT_BACKEND==0)
	sprintf(sc->halfLiteral.data.s, "h");
	sprintf(sc->floatLiteral.data.s, "f");
	sprintf(sc->doubleLiteral.data.s, "LF");
	sprintf(sc->halfDef.data.s, "half");
	sprintf(sc->floatDef.data.s, "float");
	sprintf(sc->doubleDef.data.s, "double");
	sprintf(sc->half2Def.data.s, "f16vec2");
	sprintf(sc->float2Def.data.s, "vec2");
	sprintf(sc->double2Def.data.s, "dvec2");

	sprintf(sc->intDef.data.s, "int");
	sprintf(sc->uintDef.data.s, "uint");
	sprintf(sc->int64Def.data.s, "int64_t");
	sprintf(sc->uint64Def.data.s, "uint64_t");
#elif(VKFFT_BACKEND==1)
	sprintf(sc->halfLiteral.data.s, "h");
	sprintf(sc->floatLiteral.data.s, "f");
	sprintf(sc->doubleLiteral.data.s, "l");
	sprintf(sc->halfDef.data.s, "half");
	sprintf(sc->floatDef.data.s, "float");
	sprintf(sc->doubleDef.data.s, "double");
	sprintf(sc->half2Def.data.s, "half2");
	sprintf(sc->float2Def.data.s, "float2");
	sprintf(sc->double2Def.data.s, "double2");

	sprintf(sc->intDef.data.s, "int");
	sprintf(sc->uintDef.data.s, "unsigned int");
	sprintf(sc->int64Def.data.s, "long long");
	sprintf(sc->uint64Def.data.s, "unsigned long long");
#elif(VKFFT_BACKEND==2)
	sprintf(sc->halfLiteral.data.s, "h");
	sprintf(sc->floatLiteral.data.s, "f");
	sprintf(sc->doubleLiteral.data.s, "l");
	sprintf(sc->halfDef.data.s, "half");
	sprintf(sc->floatDef.data.s, "float");
	sprintf(sc->doubleDef.data.s, "double");
	sprintf(sc->half2Def.data.s, "half2");
	sprintf(sc->float2Def.data.s, "float2");
	sprintf(sc->double2Def.data.s, "double2");

	sprintf(sc->intDef.data.s, "int");
	sprintf(sc->uintDef.data.s, "unsigned int");
	sprintf(sc->int64Def.data.s, "long long");
	sprintf(sc->uint64Def.data.s, "unsigned long long");
#elif((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
	sprintf(sc->halfLiteral.data.s, "h");
	sprintf(sc->floatLiteral.data.s, "f");
	sprintf(sc->halfDef.data.s, "half");
	sprintf(sc->floatDef.data.s, "float");
	sprintf(sc->doubleDef.data.s, "double");
	sprintf(sc->half2Def.data.s, "half2");
	sprintf(sc->float2Def.data.s, "float2");
	sprintf(sc->double2Def.data.s, "double2");

	sprintf(sc->intDef.data.s, "int");
	sprintf(sc->uintDef.data.s, "unsigned int");
	sprintf(sc->int64Def.data.s, "long long");
	sprintf(sc->uint64Def.data.s, "unsigned long long");
#elif(VKFFT_BACKEND==5)
	sprintf(sc->halfLiteral.data.s, "h");
	sprintf(sc->floatLiteral.data.s, "f");
	sprintf(sc->halfDef.data.s, "half");
	sprintf(sc->floatDef.data.s, "float");
	sprintf(sc->doubleDef.data.s, "double");
	sprintf(sc->half2Def.data.s, "half2");
	sprintf(sc->float2Def.data.s, "float2");
	sprintf(sc->double2Def.data.s, "double2");

	sprintf(sc->intDef.data.s, "int");
	sprintf(sc->uintDef.data.s, "uint");
	sprintf(sc->int64Def.data.s, "long");
	sprintf(sc->uint64Def.data.s, "ulong");
#endif



	if (app->configuration.halfPrecision) {
		sc->floatTypeCode = 12;
		sc->vecTypeCode = 13;
		if (app->configuration.halfPrecisionMemoryOnly) {
			//only out of place mode, input/output buffer must be different
			sc->floatTypeKernelMemoryCode = 12;
			sc->vecTypeKernelMemoryCode = 13;
			if ((sc->axis_id == app->firstAxis) && (sc->axis_upload_id == sc->numAxisUploads - 1) && (!sc->actualInverse)) {
				sc->floatTypeInputMemoryCode = 02;
				sc->vecTypeInputMemoryCode = 03;
			}
			else {
				sc->floatTypeInputMemoryCode = 12;
				sc->vecTypeInputMemoryCode = 13;
			}
			if ((sc->axis_id == app->firstAxis) && (((!sc->reorderFourStep) && (sc->axis_upload_id == sc->numAxisUploads - 1)) || ((sc->reorderFourStep) && (sc->axis_upload_id == 0))) && (sc->actualInverse)) {
				sc->floatTypeOutputMemoryCode = 02;
				sc->vecTypeOutputMemoryCode = 03;
			}
			else {
				sc->floatTypeOutputMemoryCode = 12;
				sc->vecTypeOutputMemoryCode = 13;
			}
		}
		else {
			sc->floatTypeKernelMemoryCode = 02;
			sc->floatTypeInputMemoryCode = 02;
			sc->floatTypeOutputMemoryCode = 02;

			sc->vecTypeKernelMemoryCode = 03;
			sc->vecTypeInputMemoryCode = 03;
			sc->vecTypeOutputMemoryCode = 03;
		}
	}
	else if (app->configuration.doublePrecision) {
		sc->floatTypeCode = 22;
		sc->vecTypeCode = 23;

		sc->floatTypeKernelMemoryCode = 22;
		sc->floatTypeInputMemoryCode = 22;
		sc->floatTypeOutputMemoryCode = 22;

		sc->vecTypeKernelMemoryCode = 23;
		sc->vecTypeInputMemoryCode = 23;
		sc->vecTypeOutputMemoryCode = 23;
	}
	else {
		if (app->configuration.doublePrecisionFloatMemory) {
			sc->floatTypeCode = 22;
			sc->vecTypeCode = 23;

			sc->floatTypeKernelMemoryCode = 12;
			sc->floatTypeInputMemoryCode = 12;
			sc->floatTypeOutputMemoryCode = 12;

			sc->vecTypeKernelMemoryCode = 13;
			sc->vecTypeInputMemoryCode = 13;
			sc->vecTypeOutputMemoryCode = 13;
		}
		else {
			sc->floatTypeCode = 12;
			sc->vecTypeCode = 13;

			sc->floatTypeKernelMemoryCode = 12;
			sc->floatTypeInputMemoryCode = 12;
			sc->floatTypeOutputMemoryCode = 12;

			sc->vecTypeKernelMemoryCode = 13;
			sc->vecTypeInputMemoryCode = 13;
			sc->vecTypeOutputMemoryCode = 13;
		}
	}
	
	if (!app->configuration.useUint64) {
		sc->intTypeCode = 11;
		sc->uintTypeCode = 01;
	}
	else {
		sc->intTypeCode = 31;
		sc->uintTypeCode = 21;
	}

	sc->uintType32Code = 01;

	return res;
}

static inline VkFFTResult initParametersAPI(VkFFTApplication* app, VkFFTSpecializationConstantsLayout* sc) {
	VkFFTResult res = VKFFT_SUCCESS;
	sc->tempStr = (char*)calloc(sc->maxTempLength, sizeof(char));
	if (!sc->tempStr) return VKFFT_ERROR_MALLOC_FAILED;
	sc->tempLen = 0;
	sc->currentLen = 0;
	PfAllocateContainerFlexible(sc, &sc->inputsStruct, 50);
	sc->inputsStruct.type = 200 + sc->inputMemoryCode;
	PfAllocateContainerFlexible(sc, &sc->outputsStruct, 50);
	sc->outputsStruct.type = 200 + sc->outputMemoryCode;

	PfAllocateContainerFlexible(sc, &sc->sdataStruct, 50);
	sc->sdataStruct.type = 200 + sc->vecTypeCode;
	sprintf(sc->sdataStruct.data.s, "sdata");

	PfAllocateContainerFlexible(sc, &sc->LUTStruct, 50);
	sc->LUTStruct.type = 200 + sc->vecTypeCode;
	sprintf(sc->LUTStruct.data.s, "twiddleLUT");

	PfAllocateContainerFlexible(sc, &sc->BluesteinStruct, 50);
	sc->BluesteinStruct.type = 200 + sc->vecTypeCode;
	sprintf(sc->BluesteinStruct.data.s, "BluesteinMultiplication");

	PfAllocateContainerFlexible(sc, &sc->BluesteinConvolutionKernelStruct, 50);
	sc->BluesteinConvolutionKernelStruct.type = 200 + sc->vecTypeCode;
	sprintf(sc->BluesteinConvolutionKernelStruct.data.s, "BluesteinConvolutionKernel");
	
	PfAllocateContainerFlexible(sc, &sc->kernelStruct, 50);
	sc->kernelStruct.type = 200 + sc->vecTypeCode;
	sprintf(sc->kernelStruct.data.s, "kernel_obj");

	for (int i = 0; i < sc->numRaderPrimes; i++) {
		if (sc->raderContainer[i].prime > 0) {
			if (sc->inline_rader_g_pow == 1) {
				PfAllocateContainerFlexible(sc, &sc->raderContainer[i].g_powConstantStruct, 50);
				sc->BluesteinConvolutionKernelStruct.type = 201;
				sprintf(sc->raderContainer[i].g_powConstantStruct.data.s, "g_pow_%d", sc->raderContainer[i].prime);
			}
			if (sc->inline_rader_kernel) {
				PfAllocateContainerFlexible(sc, &sc->raderContainer[i].r_rader_kernelConstantStruct, 50);
				sc->BluesteinConvolutionKernelStruct.type = 200 + sc->floatTypeCode;
				sprintf(sc->raderContainer[i].r_rader_kernelConstantStruct.data.s, "r_rader_kernel_%d", sc->raderContainer[i].prime);

				PfAllocateContainerFlexible(sc, &sc->raderContainer[i].i_rader_kernelConstantStruct, 50);
				sc->BluesteinConvolutionKernelStruct.type = 200 + sc->floatTypeCode;
				sprintf(sc->raderContainer[i].i_rader_kernelConstantStruct.data.s, "i_rader_kernel_%d", sc->raderContainer[i].prime);
			}
		}
	}
	if (sc->inline_rader_g_pow == 2) {
		PfAllocateContainerFlexible(sc, &sc->g_powStruct, 50);
		sc->g_powStruct.type = 201;
		sprintf(sc->g_powStruct.data.s, "g_pow");
	}
	PfAllocateContainerFlexible(sc, &sc->gl_LocalInvocationID_x, 50);
	sc->gl_LocalInvocationID_x.type = 101;
	PfAllocateContainerFlexible(sc, &sc->gl_LocalInvocationID_y, 50);
	sc->gl_LocalInvocationID_y.type = 101;
	PfAllocateContainerFlexible(sc, &sc->gl_LocalInvocationID_z, 50);
	sc->gl_LocalInvocationID_z.type = 101;
	PfAllocateContainerFlexible(sc, &sc->gl_GlobalInvocationID_x, 50);
	sc->gl_GlobalInvocationID_x.type = 101;
	PfAllocateContainerFlexible(sc, &sc->gl_GlobalInvocationID_y, 50);
	sc->gl_GlobalInvocationID_y.type = 101;
	PfAllocateContainerFlexible(sc, &sc->gl_GlobalInvocationID_z, 50);
	sc->gl_GlobalInvocationID_z.type = 101;
	PfAllocateContainerFlexible(sc, &sc->gl_WorkGroupSize_x, 50);
	sc->gl_WorkGroupSize_x.type = 101;
	PfAllocateContainerFlexible(sc, &sc->gl_WorkGroupSize_y, 50);
	sc->gl_WorkGroupSize_y.type = 101;
	PfAllocateContainerFlexible(sc, &sc->gl_WorkGroupSize_z, 50);
	sc->gl_WorkGroupSize_z.type = 101;
	PfAllocateContainerFlexible(sc, &sc->gl_WorkGroupID_x, 50);
	sc->gl_WorkGroupID_x.type = 101;
	PfAllocateContainerFlexible(sc, &sc->gl_WorkGroupID_y, 50);
	sc->gl_WorkGroupID_y.type = 101;
	PfAllocateContainerFlexible(sc, &sc->gl_WorkGroupID_z, 50);
	sc->gl_WorkGroupID_z.type = 101;

	//PfAllocateContainerFlexible(sc, &sc->cosDef, 50);
	//sc->cosDef.type = 100;
	//PfAllocateContainerFlexible(sc, &sc->sinDef, 50);
	//sc->sinDef.type = 100;

	PfAllocateContainerFlexible(sc, &sc->constDef, 50);
	sc->constDef.type = 100;

	PfAllocateContainerFlexible(sc, &sc->functionDef, 50);
	sc->functionDef.type = 100;

	if (sc->performWorkGroupShift[0]) {
		PfAllocateContainerFlexible(sc, &sc->workGroupShiftX, 50);
		sc->workGroupShiftX.type = 100 + sc->uintTypeCode;
		sprintf(sc->workGroupShiftX.data.s, "workGroupShiftX");
	}
	if (sc->performWorkGroupShift[1]) {
		PfAllocateContainerFlexible(sc, &sc->workGroupShiftY, 50);
		sc->workGroupShiftY.type = 100 + sc->uintTypeCode;
		sprintf(sc->workGroupShiftY.data.s, "workGroupShiftY");
	}
	if (sc->performWorkGroupShift[2]) {
		PfAllocateContainerFlexible(sc, &sc->workGroupShiftZ, 50);
		sc->workGroupShiftZ.type = 100 + sc->uintTypeCode;
		sprintf(sc->workGroupShiftZ.data.s, "workGroupShiftZ");
	}
	if (sc->performPostCompilationInputOffset) {
		PfAllocateContainerFlexible(sc, &sc->inputOffset, 50);
		sc->inputOffset.type = 100 + sc->uintTypeCode;
		sprintf(sc->inputOffset.data.s, "inputOffset");
	}
	if (sc->performPostCompilationOutputOffset) {
		PfAllocateContainerFlexible(sc, &sc->outputOffset, 50);
		sc->outputOffset.type = 100 + sc->uintTypeCode;
		sprintf(sc->outputOffset.data.s, "outputOffset");
	}
	if (sc->performPostCompilationKernelOffset) {
		PfAllocateContainerFlexible(sc, &sc->kernelOffset, 50);
		sc->kernelOffset.type = 100 + sc->uintTypeCode;
		sprintf(sc->kernelOffset.data.s, "kernelOffset");
	}
#if(VKFFT_BACKEND==0)
	sprintf(sc->inputsStruct.data.s, "inputs");
	sprintf(sc->outputsStruct.data.s, "outputs");
	sprintf(sc->gl_LocalInvocationID_x.data.s, "gl_LocalInvocationID.x");
	sprintf(sc->gl_LocalInvocationID_y.data.s, "gl_LocalInvocationID.y");
	sprintf(sc->gl_LocalInvocationID_z.data.s, "gl_LocalInvocationID.z");
	switch (sc->swapComputeWorkGroupID) {
	case 0:
		sprintf(sc->gl_GlobalInvocationID_x.data.s, "gl_GlobalInvocationID.x");
		sprintf(sc->gl_GlobalInvocationID_y.data.s, "gl_GlobalInvocationID.y");
		sprintf(sc->gl_GlobalInvocationID_z.data.s, "gl_GlobalInvocationID.z");
		sprintf(sc->gl_WorkGroupID_x.data.s, "gl_WorkGroupID.x");
		sprintf(sc->gl_WorkGroupID_y.data.s, "gl_WorkGroupID.y");
		sprintf(sc->gl_WorkGroupID_z.data.s, "gl_WorkGroupID.z");
		break;
	case 1:
		sprintf(sc->gl_GlobalInvocationID_x.data.s, "(gl_LocalInvocationID.x + gl_WorkGroupID.y * %" PRIi64 ")", sc->localSize[0].data.i);
		sprintf(sc->gl_GlobalInvocationID_y.data.s, "(gl_LocalInvocationID.y + gl_WorkGroupID.x * %" PRIi64 ")", sc->localSize[1].data.i);
		sprintf(sc->gl_GlobalInvocationID_z.data.s, "gl_GlobalInvocationID.z");
		sprintf(sc->gl_WorkGroupID_x.data.s, "gl_WorkGroupID.y");
		sprintf(sc->gl_WorkGroupID_y.data.s, "gl_WorkGroupID.x");
		sprintf(sc->gl_WorkGroupID_z.data.s, "gl_WorkGroupID.z");
		break;
	case 2:
		sprintf(sc->gl_GlobalInvocationID_x.data.s, "(gl_LocalInvocationID.x + gl_WorkGroupID.z * %" PRIi64 ")", sc->localSize[0].data.i);
		sprintf(sc->gl_GlobalInvocationID_y.data.s, "gl_GlobalInvocationID.y");
		sprintf(sc->gl_GlobalInvocationID_z.data.s, "(gl_LocalInvocationID.z + gl_WorkGroupID.x * %" PRIi64 ")", sc->localSize[2].data.i);
		sprintf(sc->gl_WorkGroupID_x.data.s, "gl_WorkGroupID.z");
		sprintf(sc->gl_WorkGroupID_y.data.s, "gl_WorkGroupID.y");
		sprintf(sc->gl_WorkGroupID_z.data.s, "gl_WorkGroupID.x");
		break;
	}
	sprintf(sc->gl_WorkGroupSize_x.data.s, "%" PRIi64 "", sc->localSize[0].data.i);
	sprintf(sc->gl_WorkGroupSize_y.data.s, "%" PRIi64 "", sc->localSize[1].data.i);
	sprintf(sc->gl_WorkGroupSize_z.data.s, "%" PRIi64 "", sc->localSize[2].data.i);
	//sprintf(sc->cosDef.data.s, "cos");
	//sprintf(sc->sinDef.data.s, "sin");
	sprintf(sc->constDef.data.s, "const");
#elif((VKFFT_BACKEND==1) ||(VKFFT_BACKEND==2))
	sprintf(sc->inputsStruct.data.s, "inputs");
	sprintf(sc->outputsStruct.data.s, "outputs");
	sprintf(sc->gl_LocalInvocationID_x.data.s, "threadIdx.x");
	sprintf(sc->gl_LocalInvocationID_y.data.s, "threadIdx.y");
	sprintf(sc->gl_LocalInvocationID_z.data.s, "threadIdx.z");
	switch (sc->swapComputeWorkGroupID) {
	case 0:
		sprintf(sc->gl_GlobalInvocationID_x.data.s, "(threadIdx.x + blockIdx.x * %" PRIi64 ")", sc->localSize[0].data.i);
		sprintf(sc->gl_GlobalInvocationID_y.data.s, "(threadIdx.y + blockIdx.y * %" PRIi64 ")", sc->localSize[1].data.i);
		sprintf(sc->gl_GlobalInvocationID_z.data.s, "(threadIdx.z + blockIdx.z * %" PRIi64 ")", sc->localSize[2].data.i);
		sprintf(sc->gl_WorkGroupID_x.data.s, "blockIdx.x");
		sprintf(sc->gl_WorkGroupID_y.data.s, "blockIdx.y");
		sprintf(sc->gl_WorkGroupID_z.data.s, "blockIdx.z");
		break;
	case 1:
		sprintf(sc->gl_GlobalInvocationID_x.data.s, "(threadIdx.x + blockIdx.y * %" PRIi64 ")", sc->localSize[0].data.i);
		sprintf(sc->gl_GlobalInvocationID_y.data.s, "(threadIdx.y + blockIdx.x * %" PRIi64 ")", sc->localSize[1].data.i);
		sprintf(sc->gl_GlobalInvocationID_z.data.s, "(threadIdx.z + blockIdx.z * %" PRIi64 ")", sc->localSize[2].data.i);
		sprintf(sc->gl_WorkGroupID_x.data.s, "blockIdx.y");
		sprintf(sc->gl_WorkGroupID_y.data.s, "blockIdx.x");
		sprintf(sc->gl_WorkGroupID_z.data.s, "blockIdx.z");
		break;
	case 2:
		sprintf(sc->gl_GlobalInvocationID_x.data.s, "(threadIdx.x + blockIdx.z * %" PRIi64 ")", sc->localSize[0].data.i);
		sprintf(sc->gl_GlobalInvocationID_y.data.s, "(threadIdx.y + blockIdx.y * %" PRIi64 ")", sc->localSize[1].data.i);
		sprintf(sc->gl_GlobalInvocationID_z.data.s, "(threadIdx.z + blockIdx.x * %" PRIi64 ")", sc->localSize[2].data.i);
		sprintf(sc->gl_WorkGroupID_x.data.s, "blockIdx.z");
		sprintf(sc->gl_WorkGroupID_y.data.s, "blockIdx.y");
		sprintf(sc->gl_WorkGroupID_z.data.s, "blockIdx.x");
		break;
	}
	sprintf(sc->gl_WorkGroupSize_x.data.s, "%" PRIi64 "", sc->localSize[0].data.i);
	sprintf(sc->gl_WorkGroupSize_y.data.s, "%" PRIi64 "", sc->localSize[1].data.i);
	sprintf(sc->gl_WorkGroupSize_z.data.s, "%" PRIi64 "", sc->localSize[2].data.i);
	//sprintf(sc->cosDef.data.s, "__cosf");
	//sprintf(sc->sinDef.data.s, "__sinf");
	sprintf(sc->constDef.data.s, "const");
	sprintf(sc->functionDef.data.s, "__device__ static __inline__ ");
#elif((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
	sprintf(sc->inputsStruct.data.s, "inputs");
	sprintf(sc->outputsStruct.data.s, "outputs");
	sprintf(sc->gl_LocalInvocationID_x.data.s, "get_local_id(0)");
	sprintf(sc->gl_LocalInvocationID_y.data.s, "get_local_id(1)");
	sprintf(sc->gl_LocalInvocationID_z.data.s, "get_local_id(2)");
	switch (sc->swapComputeWorkGroupID) {
	case 0:
		sprintf(sc->gl_GlobalInvocationID_x.data.s, "get_global_id(0)");
		sprintf(sc->gl_GlobalInvocationID_y.data.s, "get_global_id(1)");
		sprintf(sc->gl_GlobalInvocationID_z.data.s, "get_global_id(2)");
		sprintf(sc->gl_WorkGroupID_x.data.s, "get_group_id(0)");
		sprintf(sc->gl_WorkGroupID_y.data.s, "get_group_id(1)");
		sprintf(sc->gl_WorkGroupID_z.data.s, "get_group_id(2)");
		break;
	case 1:
		sprintf(sc->gl_GlobalInvocationID_x.data.s, "(get_local_id(0) + get_group_id(1) * %" PRIi64 ")", sc->localSize[0].data.i);
		sprintf(sc->gl_GlobalInvocationID_y.data.s, "(get_local_id(1) + get_group_id(0) * %" PRIi64 ")", sc->localSize[1].data.i);
		sprintf(sc->gl_GlobalInvocationID_z.data.s, "get_global_id(2)");
		sprintf(sc->gl_WorkGroupID_x.data.s, "get_group_id(1)");
		sprintf(sc->gl_WorkGroupID_y.data.s, "get_group_id(0)");
		sprintf(sc->gl_WorkGroupID_z.data.s, "get_group_id(2)");
		break;
	case 2:
		sprintf(sc->gl_GlobalInvocationID_x.data.s, "(get_local_id(0) + get_group_id(2) * %" PRIi64 ")", sc->localSize[0].data.i);
		sprintf(sc->gl_GlobalInvocationID_y.data.s, "get_global_id(1)");
		sprintf(sc->gl_GlobalInvocationID_z.data.s, "(get_local_id(2) + get_group_id(0) * %" PRIi64 ")", sc->localSize[2].data.i);
		sprintf(sc->gl_WorkGroupID_x.data.s, "get_group_id(2)");
		sprintf(sc->gl_WorkGroupID_y.data.s, "get_group_id(1)");
		sprintf(sc->gl_WorkGroupID_z.data.s, "get_group_id(0)");
		break;
	}
	sprintf(sc->gl_WorkGroupSize_x.data.s, "%" PRIi64 "", sc->localSize[0].data.i);
	sprintf(sc->gl_WorkGroupSize_y.data.s, "%" PRIi64 "", sc->localSize[1].data.i);
	sprintf(sc->gl_WorkGroupSize_z.data.s, "%" PRIi64 "", sc->localSize[2].data.i);
	//sprintf(sc->cosDef.data.s, "native_cos");
	//sprintf(sc->sinDef.data.s, "native_sin");
	sprintf(sc->constDef.data.s, "__constant");
	sprintf(sc->functionDef.data.s, "static __inline__ ");
#elif(VKFFT_BACKEND==5)
	sprintf(sc->inputsStruct.data.s, "inputs");
	sprintf(sc->outputsStruct.data.s, "outputs");
	sprintf(sc->gl_LocalInvocationID_x.data.s, "thread_position_in_threadgroup.x");
	sprintf(sc->gl_LocalInvocationID_y.data.s, "thread_position_in_threadgroup.y");
	sprintf(sc->gl_LocalInvocationID_z.data.s, "thread_position_in_threadgroup.z");
	switch (sc->swapComputeWorkGroupID) {
	case 0:
		sprintf(sc->gl_GlobalInvocationID_x.data.s, "thread_position_in_grid.x");
		sprintf(sc->gl_GlobalInvocationID_y.data.s, "thread_position_in_grid.y");
		sprintf(sc->gl_GlobalInvocationID_z.data.s, "thread_position_in_grid.z");
		sprintf(sc->gl_WorkGroupID_x.data.s, "threadgroup_position_in_grid.x");
		sprintf(sc->gl_WorkGroupID_y.data.s, "threadgroup_position_in_grid.y");
		sprintf(sc->gl_WorkGroupID_z.data.s, "threadgroup_position_in_grid.z");
		break;
	case 1:
		sprintf(sc->gl_GlobalInvocationID_x.data.s, "(thread_position_in_threadgroup.x + threadgroup_position_in_grid.y * %" PRIi64 ")", sc->localSize[0].data.i);
		sprintf(sc->gl_GlobalInvocationID_y.data.s, "(thread_position_in_threadgroup.y + threadgroup_position_in_grid.x * %" PRIi64 ")", sc->localSize[1].data.i);
		sprintf(sc->gl_GlobalInvocationID_z.data.s, "thread_position_in_threadgroup.z");
		sprintf(sc->gl_WorkGroupID_x.data.s, "threadgroup_position_in_grid.y");
		sprintf(sc->gl_WorkGroupID_y.data.s, "threadgroup_position_in_grid.x");
		sprintf(sc->gl_WorkGroupID_z.data.s, "threadgroup_position_in_grid.z");
		break;
	case 2:
		sprintf(sc->gl_GlobalInvocationID_x.data.s, "(thread_position_in_threadgroup.x + threadgroup_position_in_grid.z * %" PRIi64 ")", sc->localSize[0].data.i);
		sprintf(sc->gl_GlobalInvocationID_y.data.s, "thread_position_in_threadgroup.y");
		sprintf(sc->gl_GlobalInvocationID_z.data.s, "(thread_position_in_threadgroup.z + threadgroup_position_in_grid.x * %" PRIi64 ")", sc->localSize[2].data.i);
		sprintf(sc->gl_WorkGroupID_x.data.s, "threadgroup_position_in_grid.z");
		sprintf(sc->gl_WorkGroupID_y.data.s, "threadgroup_position_in_grid.y");
		sprintf(sc->gl_WorkGroupID_z.data.s, "threadgroup_position_in_grid.x");
		break;
}
	sprintf(sc->gl_WorkGroupSize_x.data.s, "%" PRIi64 "", sc->localSize[0].data.i);
	sprintf(sc->gl_WorkGroupSize_y.data.s, "%" PRIi64 "", sc->localSize[1].data.i);
	sprintf(sc->gl_WorkGroupSize_z.data.s, "%" PRIi64 "", sc->localSize[2].data.i);
	//sprintf(sc->cosDef.data.s, "native_cos");
	//sprintf(sc->sinDef.data.s, "native_sin");
	sprintf(sc->constDef.data.s, "constant");
#endif
	return res;
}

static inline VkFFTResult freeMemoryParametersAPI(VkFFTApplication* app, VkFFTSpecializationConstantsLayout* sc) {
	VkFFTResult res = VKFFT_SUCCESS;

	PfDeallocateContainer(sc, &sc->halfLiteral);
	PfDeallocateContainer(sc, &sc->floatLiteral);
	PfDeallocateContainer(sc, &sc->doubleLiteral);
	PfDeallocateContainer(sc, &sc->halfDef);
	PfDeallocateContainer(sc, &sc->floatDef);
	PfDeallocateContainer(sc, &sc->doubleDef);
	PfDeallocateContainer(sc, &sc->half2Def);
	PfDeallocateContainer(sc, &sc->float2Def);
	PfDeallocateContainer(sc, &sc->double2Def);
	PfDeallocateContainer(sc, &sc->intDef);
	PfDeallocateContainer(sc, &sc->uintDef);
	PfDeallocateContainer(sc, &sc->int64Def);
	PfDeallocateContainer(sc, &sc->uint64Def);
	return res;
}

static inline VkFFTResult freeParametersAPI(VkFFTApplication* app, VkFFTSpecializationConstantsLayout* sc) {
	VkFFTResult res = VKFFT_SUCCESS;
	free(sc->tempStr);
	sc->tempStr = 0;
	PfDeallocateContainer(sc, &sc->inputsStruct);
	PfDeallocateContainer(sc, &sc->outputsStruct);
	PfDeallocateContainer(sc, &sc->sdataStruct);
	PfDeallocateContainer(sc, &sc->LUTStruct);
	PfDeallocateContainer(sc, &sc->BluesteinStruct);
	PfDeallocateContainer(sc, &sc->BluesteinConvolutionKernelStruct);
	PfDeallocateContainer(sc, &sc->kernelStruct);
	for (int i = 0; i < sc->numRaderPrimes; i++) {
		if (sc->raderContainer[i].prime > 0) {
			if (sc->inline_rader_g_pow == 1) {
				PfDeallocateContainer(sc, &sc->raderContainer[i].g_powConstantStruct);
			}
			if (sc->inline_rader_kernel) {
				PfDeallocateContainer(sc, &sc->raderContainer[i].r_rader_kernelConstantStruct);
				PfDeallocateContainer(sc, &sc->raderContainer[i].i_rader_kernelConstantStruct);
			}
		}
	}
	if (sc->inline_rader_g_pow == 2) {
		PfDeallocateContainer(sc, &sc->g_powStruct);
	}

	PfDeallocateContainer(sc, &sc->gl_LocalInvocationID_x);
	PfDeallocateContainer(sc, &sc->gl_LocalInvocationID_y);
	PfDeallocateContainer(sc, &sc->gl_LocalInvocationID_z);
	PfDeallocateContainer(sc, &sc->gl_GlobalInvocationID_x);
	PfDeallocateContainer(sc, &sc->gl_GlobalInvocationID_y);
	PfDeallocateContainer(sc, &sc->gl_GlobalInvocationID_z);
	PfDeallocateContainer(sc, &sc->gl_WorkGroupSize_x);
	PfDeallocateContainer(sc, &sc->gl_WorkGroupSize_y);
	PfDeallocateContainer(sc, &sc->gl_WorkGroupSize_z);
	PfDeallocateContainer(sc, &sc->gl_WorkGroupID_x);
	PfDeallocateContainer(sc, &sc->gl_WorkGroupID_y);
	PfDeallocateContainer(sc, &sc->gl_WorkGroupID_z);
	//PfDeallocateContainer(sc, &sc->cosDef);
	//PfDeallocateContainer(sc, &sc->sinDef);

	PfDeallocateContainer(sc, &sc->constDef);
	PfDeallocateContainer(sc, &sc->functionDef);

	if (sc->performWorkGroupShift[0]) {
		PfDeallocateContainer(sc, &sc->workGroupShiftX);
	}
	if (sc->performWorkGroupShift[1]) {
		PfDeallocateContainer(sc, &sc->workGroupShiftY);
	}
	if (sc->performWorkGroupShift[2]) {
		PfDeallocateContainer(sc, &sc->workGroupShiftZ);
	}
	if (sc->performPostCompilationInputOffset) {
		PfDeallocateContainer(sc, &sc->inputOffset);
	}
	if (sc->performPostCompilationOutputOffset) {
		PfDeallocateContainer(sc, &sc->outputOffset);
	}
	if (sc->performPostCompilationKernelOffset) {
		PfDeallocateContainer(sc, &sc->kernelOffset);
	}
	return res;
}

#endif
