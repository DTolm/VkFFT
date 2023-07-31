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
#ifndef VKFFT_KERNELSTARTEND_H
#define VKFFT_KERNELSTARTEND_H
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_StringManagement/vkFFT_StringManager.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryInitialization/vkFFT_SharedMemory.h"
static inline void appendKernelStart(VkFFTSpecializationConstantsLayout* sc, int64_t type) {
	if (sc->res != VKFFT_SUCCESS) return;
	uint64_t locType = (((type == 0) || (type == 5) || (type == 6) || (type == 110) || (type == 120) || (type == 130) || (type == 140) || (type == 142) || (type == 144)) && (sc->axisSwapped)) ? 1 : type;
	PfContainer* floatType;
	PfGetTypeFromCode(sc, sc->floatTypeCode, &floatType); 
	PfContainer* floatTypeInputMemory;
	PfGetTypeFromCode(sc, sc->floatTypeInputMemoryCode, &floatTypeInputMemory);
	PfContainer* floatTypeOutputMemory;
	PfGetTypeFromCode(sc, sc->floatTypeOutputMemoryCode, &floatTypeOutputMemory);
	PfContainer* floatTypeKernelMemory;
	PfGetTypeFromCode(sc, sc->floatTypeKernelMemoryCode, &floatTypeKernelMemory);

	PfContainer* vecType;
	PfGetTypeFromCode(sc, sc->vecTypeCode, &vecType);
	PfContainer* vecTypeInputMemory;
	PfGetTypeFromCode(sc, sc->vecTypeInputMemoryCode, &vecTypeInputMemory);
	PfContainer* vecTypeOutputMemory;
	PfGetTypeFromCode(sc, sc->vecTypeOutputMemoryCode, &vecTypeOutputMemory);
	PfContainer* vecTypeKernelMemory;
	PfGetTypeFromCode(sc, sc->vecTypeKernelMemoryCode, &vecTypeKernelMemory);

	PfContainer* uintType;
	PfGetTypeFromCode(sc, sc->uintTypeCode, &uintType);

	PfContainer* uintType32;
	PfGetTypeFromCode(sc, sc->uintType32Code, &uintType32);
#if(VKFFT_BACKEND==0)
	appendSharedMemoryVkFFT(sc, (int)locType);
	sc->tempLen = sprintf(sc->tempStr, "void main() {\n");
	PfAppendLine(sc);
#elif(VKFFT_BACKEND==1)
	sc->tempLen = sprintf(sc->tempStr, "extern __shared__ float shared[];\n");
	PfAppendLine(sc);
	
	sc->tempLen = sprintf(sc->tempStr, "extern \"C\" __global__ void __launch_bounds__(%" PRIi64 ") VkFFT_main ", sc->localSize[0].data.i * sc->localSize[1].data.i * sc->localSize[2].data.i);
	PfAppendLine(sc);
	switch (type) {
	case 5:
	{
		sc->tempLen = sprintf(sc->tempStr, "(%s* inputs, %s* outputs", floatTypeInputMemory->data.s, vecTypeOutputMemory->data.s);
		break;
	}
	case 6:
	{
		sc->tempLen = sprintf(sc->tempStr, "(%s* inputs, %s* outputs", vecTypeInputMemory->data.s, floatTypeOutputMemory->data.s);
		break;
	}
	case 110:case 111:case 120:case 121:case 130:case 131:case 140:case 141:case 142:case 143:case 144:case 145:
	{
		sc->tempLen = sprintf(sc->tempStr, "(%s* inputs, %s* outputs", floatTypeInputMemory->data.s, floatTypeOutputMemory->data.s);
		break;
	}
	default:
	{
		sc->tempLen = sprintf(sc->tempStr, "(%s* inputs, %s* outputs", vecTypeInputMemory->data.s, vecTypeOutputMemory->data.s);
		break;
	}
	}
	PfAppendLine(sc);

	if (sc->convolutionStep) {
		sc->tempLen = sprintf(sc->tempStr, ", %s* kernel_obj", vecType->data.s);
		PfAppendLine(sc);
	}
	if (sc->LUT) {
		sc->tempLen = sprintf(sc->tempStr, ", %s* twiddleLUT", vecType->data.s);
		PfAppendLine(sc);
	}
	if (sc->raderUintLUT) {
		sc->tempLen = sprintf(sc->tempStr, ", %s* g_pow", uintType32->data.s);
		PfAppendLine(sc);
	}
	if (sc->BluesteinConvolutionStep) {
		sc->tempLen = sprintf(sc->tempStr, ", %s* BluesteinConvolutionKernel", vecType->data.s);
		PfAppendLine(sc);
	}
	if (sc->BluesteinPreMultiplication || sc->BluesteinPostMultiplication) {
		sc->tempLen = sprintf(sc->tempStr, ", %s* BluesteinMultiplication", vecType->data.s);
		PfAppendLine(sc);
	}
	sc->tempLen = sprintf(sc->tempStr, ") {\n");
	PfAppendLine(sc);
	//sc->tempLen = sprintf(sc->tempStr, ", const PushConsts consts) {\n");
	appendSharedMemoryVkFFT(sc, (int)locType);
	
#elif(VKFFT_BACKEND==2)
	sc->tempLen = sprintf(sc->tempStr, "extern __shared__ float shared[];\n");
	PfAppendLine(sc);
	if (!sc->useUint64 && sc->useStrict32BitAddress > 0) {
		// These wrappers help hipcc to generate faster code for load and store operations where
		// 64-bit scalar + 32-bit vector registers are used instead of 64-bit vector saving a few
		// instructions for computing 64-bit vector addresses.
		sc->tempLen = sprintf(sc->tempStr,
			"template<typename T>\n"
			"struct Inputs\n"
			"{\n"
			"	const T* buffer;\n"
			"	inline __device__ Inputs(const T* buffer) : buffer(buffer) {}\n"
			"	inline __device__ const T& operator[](unsigned int idx) const { return *reinterpret_cast<const T*>(reinterpret_cast<const char*>(buffer) + idx * static_cast<unsigned int>(sizeof(T))); }\n"
			"};\n"
			"template<typename T>\n"
			"struct Outputs\n"
			"{\n"
			"	T* buffer;\n"
			"	inline __device__ Outputs(T* buffer) : buffer(buffer) {}\n"
			"	inline __device__ T& operator[](unsigned int idx) const { return *reinterpret_cast<T*>(reinterpret_cast<char*>(buffer) + idx * static_cast<unsigned int>(sizeof(T))); }\n"
			"};\n"
		);
	}
	else {
		sc->tempLen = sprintf(sc->tempStr,
			"template<typename T>\n"
			"using Inputs = const T*;\n"
			"template<typename T>\n"
			"using Outputs = T*;\n"
		);
	}
	PfAppendLine(sc);
	sc->tempLen = sprintf(sc->tempStr, "extern \"C\" __launch_bounds__(%" PRIi64 ") __global__ void VkFFT_main ", sc->localSize[0].data.i * sc->localSize[1].data.i * sc->localSize[2].data.i);
	PfAppendLine(sc);
	switch (type) {
	case 5:
	{
		sc->tempLen = sprintf(sc->tempStr, "(const Inputs<%s> inputs, Outputs<%s> outputs", floatTypeInputMemory->data.s, vecTypeOutputMemory->data.s);
		break;
	}
	case 6:
	{
		sc->tempLen = sprintf(sc->tempStr, "(const Inputs<%s> inputs, Outputs<%s> outputs", vecTypeInputMemory->data.s, floatTypeOutputMemory->data.s);
		break;
	}
	case 110:case 111:case 120:case 121:case 130:case 131:case 140:case 141:case 142:case 143:case 144:case 145:
	{
		sc->tempLen = sprintf(sc->tempStr, "(const Inputs<%s> inputs, Outputs<%s> outputs", floatTypeInputMemory->data.s, floatTypeOutputMemory->data.s);
		break;
	}
	default:
	{
		sc->tempLen = sprintf(sc->tempStr, "(const Inputs<%s> inputs, Outputs<%s> outputs", vecTypeInputMemory->data.s, vecTypeOutputMemory->data.s);
		break;
	}
	}
	PfAppendLine(sc);
	if (sc->convolutionStep) {
		sc->tempLen = sprintf(sc->tempStr, ", const Inputs<%s> kernel_obj", vecType->data.s);
		PfAppendLine(sc);
	}
	if (sc->LUT) {
		sc->tempLen = sprintf(sc->tempStr, ", const Inputs<%s> twiddleLUT", vecType->data.s);
		PfAppendLine(sc);
	}
	if (sc->raderUintLUT) {
		sc->tempLen = sprintf(sc->tempStr, ", const Inputs<%s> g_pow", uintType32->data.s);
		PfAppendLine(sc);
	}
	if (sc->BluesteinConvolutionStep) {
		sc->tempLen = sprintf(sc->tempStr, ", const Inputs<%s> BluesteinConvolutionKernel", vecType->data.s);
		PfAppendLine(sc);
	}
	if (sc->BluesteinPreMultiplication || sc->BluesteinPostMultiplication) {
		sc->tempLen = sprintf(sc->tempStr, ", const Inputs<%s> BluesteinMultiplication", vecType->data.s);
		PfAppendLine(sc);
	}
	sc->tempLen = sprintf(sc->tempStr, ") {\n");
	PfAppendLine(sc);
	//sc->tempLen = sprintf(sc->tempStr, ", const PushConsts consts) {\n");
	appendSharedMemoryVkFFT(sc, (int)locType);
#elif((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
	sc->tempLen = sprintf(sc->tempStr, "__kernel __attribute__((reqd_work_group_size(%" PRIi64 ", %" PRIi64 ", %" PRIi64 "))) void VkFFT_main ", sc->localSize[0].data.i, sc->localSize[1].data.i, sc->localSize[2].data.i);
	PfAppendLine(sc);
	switch (type) {
	case 5:
	{
		sc->tempLen = sprintf(sc->tempStr, "(__global %s* inputs, __global %s* outputs", floatTypeInputMemory->data.s, vecTypeOutputMemory->data.s);
		break;
	}
	case 6:
	{
		sc->tempLen = sprintf(sc->tempStr, "(__global %s* inputs, __global %s* outputs", vecTypeInputMemory->data.s, floatTypeOutputMemory->data.s);
		break;
	}
	case 110:case 111:case 120:case 121:case 130:case 131:case 140:case 141:case 142:case 143:case 144:case 145:
	{
		sc->tempLen = sprintf(sc->tempStr, "(__global %s* inputs, __global %s* outputs", floatTypeInputMemory->data.s, floatTypeOutputMemory->data.s);
		break;
	}
	default:
	{
		sc->tempLen = sprintf(sc->tempStr, "(__global %s* inputs, __global %s* outputs", vecTypeInputMemory->data.s, vecTypeOutputMemory->data.s);
		break;
	}
	}
	PfAppendLine(sc);
	int args_id = 2;
	if (sc->convolutionStep) {
		sc->tempLen = sprintf(sc->tempStr, ", __global %s* kernel_obj", vecType->data.s);
		PfAppendLine(sc);
		args_id++;
	}
	if (sc->LUT) {
		sc->tempLen = sprintf(sc->tempStr, ", __global %s* twiddleLUT", vecType->data.s);
		PfAppendLine(sc);
		args_id++;
	}
	if (sc->raderUintLUT) {
		sc->tempLen = sprintf(sc->tempStr, ", __global %s* g_pow", uintType32->data.s);
		PfAppendLine(sc);
		args_id++;
	}
	if (sc->BluesteinConvolutionStep) {
		sc->tempLen = sprintf(sc->tempStr, ", __global %s* BluesteinConvolutionKernel", vecType->data.s);
		PfAppendLine(sc);
		args_id++;
	}
	if (sc->BluesteinPreMultiplication || sc->BluesteinPostMultiplication) {
		sc->tempLen = sprintf(sc->tempStr, ", __global %s* BluesteinMultiplication", vecType->data.s);
		PfAppendLine(sc);
		args_id++;
	}
	if (sc->pushConstantsStructSize > 0) {
		sc->tempLen = sprintf(sc->tempStr, ", PushConsts consts");
		PfAppendLine(sc);
	}
	sc->tempLen = sprintf(sc->tempStr, ") {\n");
	PfAppendLine(sc);
	//sc->tempLen = sprintf(sc->tempStr, ", const PushConsts consts) {\n");
	appendSharedMemoryVkFFT(sc, (int)locType);
#elif(VKFFT_BACKEND==5)
	sc->tempLen = sprintf(sc->tempStr, "kernel void VkFFT_main ");
	PfAppendLine(sc);
	
	sc->tempLen = sprintf(sc->tempStr, "(%s3 thread_position_in_grid [[thread_position_in_grid]], ", uintType->data.s);
	PfAppendLine(sc);
	
	sc->tempLen = sprintf(sc->tempStr, "%s3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], ", uintType->data.s);
	PfAppendLine(sc);
	
	sc->tempLen = sprintf(sc->tempStr, "%s3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], ", uintType->data.s);
	PfAppendLine(sc);
	
	sc->tempLen = sprintf(sc->tempStr, "threadgroup %s* sdata [[threadgroup(0)]], ", vecType->data.s);
	PfAppendLine(sc);
	switch (type) {
	case 5:
	{
		sc->tempLen = sprintf(sc->tempStr, "device %s* inputs[[buffer(0)]], device %s* outputs[[buffer(1)]]", floatTypeInputMemory->data.s, vecTypeOutputMemory->data.s);
		break;
	}
	case 6:
	{
		sc->tempLen = sprintf(sc->tempStr, "device %s* inputs[[buffer(0)]], device %s* outputs[[buffer(1)]]", vecTypeInputMemory->data.s, floatTypeOutputMemory->data.s);
		break;
	}
	case 110:case 111:case 120:case 121:case 130:case 131:case 140:case 141:case 142:case 143:case 144:case 145:
	{
		sc->tempLen = sprintf(sc->tempStr, "device %s* inputs[[buffer(0)]], device %s* outputs[[buffer(1)]]", floatTypeInputMemory->data.s, floatTypeOutputMemory->data.s);
		break;
	}
	default:
	{
		sc->tempLen = sprintf(sc->tempStr, "device %s* inputs[[buffer(0)]], device %s* outputs[[buffer(1)]]", vecTypeInputMemory->data.s, vecTypeOutputMemory->data.s);
		break;
	}
	}
	PfAppendLine(sc);
	int args_id = 2;
	if (sc->convolutionStep) {
		sc->tempLen = sprintf(sc->tempStr, ", constant %s* kernel_obj[[buffer(%d)]]", vecType->data.s, args_id);
		PfAppendLine(sc);
		args_id++;
	}
	if (sc->LUT) {
		sc->tempLen = sprintf(sc->tempStr, ", constant %s* twiddleLUT[[buffer(%d)]]", vecType->data.s, args_id);
		PfAppendLine(sc);
		args_id++;
	}
	if (sc->raderUintLUT) {
		sc->tempLen = sprintf(sc->tempStr, ", constant %s* g_pow[[buffer(%d)]]", uintType32->data.s, args_id);
		PfAppendLine(sc);
		args_id++;
	}
	if (sc->BluesteinConvolutionStep) {
		sc->tempLen = sprintf(sc->tempStr, ", constant %s* BluesteinConvolutionKernel[[buffer(%d)]]", vecType->data.s, args_id);
		PfAppendLine(sc);
		args_id++;
	}
	if (sc->BluesteinPreMultiplication || sc->BluesteinPostMultiplication) {
		sc->tempLen = sprintf(sc->tempStr, ", constant %s* BluesteinMultiplication[[buffer(%d)]]", vecType->data.s, args_id);
		PfAppendLine(sc);
		args_id++;
	}
	if (sc->pushConstantsStructSize > 0) {
		sc->tempLen = sprintf(sc->tempStr, ", constant PushConsts& consts[[buffer(%d)]]", args_id);
		PfAppendLine(sc);
		
		args_id++;
	}
	sc->tempLen = sprintf(sc->tempStr, ") {\n");
	PfAppendLine(sc);
	
	//sc->tempLen = sprintf(sc->tempStr, ", const PushConsts consts) {\n");
	appendSharedMemoryVkFFT(sc, (int)locType);
#endif
	return;
}

static inline void appendKernelStart_R2C(VkFFTSpecializationConstantsLayout* sc, int64_t type) {
	if (sc->res != VKFFT_SUCCESS) return;
	uint64_t locType = (((type == 0) || (type == 5) || (type == 6) || (type == 110) || (type == 120) || (type == 130) || (type == 140) || (type == 142) || (type == 144)) && (sc->axisSwapped)) ? 1 : type;
	PfContainer* floatType;
	PfGetTypeFromCode(sc, sc->floatTypeCode, &floatType);
	PfContainer* floatTypeInputMemory;
	PfGetTypeFromCode(sc, sc->floatTypeInputMemoryCode, &floatTypeInputMemory);
	PfContainer* floatTypeOutputMemory;
	PfGetTypeFromCode(sc, sc->floatTypeOutputMemoryCode, &floatTypeOutputMemory);
	PfContainer* floatTypeKernelMemory;
	PfGetTypeFromCode(sc, sc->floatTypeKernelMemoryCode, &floatTypeKernelMemory);

	PfContainer* vecType;
	PfGetTypeFromCode(sc, sc->vecTypeCode, &vecType);
	PfContainer* vecTypeInputMemory;
	PfGetTypeFromCode(sc, sc->vecTypeInputMemoryCode, &vecTypeInputMemory);
	PfContainer* vecTypeOutputMemory;
	PfGetTypeFromCode(sc, sc->vecTypeOutputMemoryCode, &vecTypeOutputMemory);
	PfContainer* vecTypeKernelMemory;
	PfGetTypeFromCode(sc, sc->vecTypeKernelMemoryCode, &vecTypeKernelMemory);

	PfContainer* uintType;
	PfGetTypeFromCode(sc, sc->uintTypeCode, &uintType);

	PfContainer* uintType32;
	PfGetTypeFromCode(sc, sc->uintType32Code, &uintType32);
#if(VKFFT_BACKEND==0)
	sc->tempLen = sprintf(sc->tempStr, "void main() {\n");
	PfAppendLine(sc);
#elif(VKFFT_BACKEND==1)
	
	sc->tempLen = sprintf(sc->tempStr, "extern \"C\" __global__ void __launch_bounds__(%" PRIi64 ") VkFFT_main_R2C ", sc->localSize[0].data.i * sc->localSize[1].data.i * sc->localSize[2].data.i);
	PfAppendLine(sc);

	sc->tempLen = sprintf(sc->tempStr, "(%s* inputs, %s* outputs", vecTypeInputMemory->data.s, vecTypeOutputMemory->data.s);
	PfAppendLine(sc);

	if (sc->LUT) {
		sc->tempLen = sprintf(sc->tempStr, ", %s* twiddleLUT", vecType->data.s);
		PfAppendLine(sc);
	}
	sc->tempLen = sprintf(sc->tempStr, ") {\n");
	PfAppendLine(sc);
	//sc->tempLen = sprintf(sc->tempStr, ", const PushConsts consts) {\n");
#elif(VKFFT_BACKEND==2)
	if (!sc->useUint64 && sc->useStrict32BitAddress > 0) {
		// These wrappers help hipcc to generate faster code for load and store operations where
		// 64-bit scalar + 32-bit vector registers are used instead of 64-bit vector saving a few
		// instructions for computing 64-bit vector addresses.
		sc->tempLen = sprintf(sc->tempStr,
			"template<typename T>\n"
			"struct Inputs\n"
			"{\n"
			"	const T* buffer;\n"
			"	inline __device__ Inputs(const T* buffer) : buffer(buffer) {}\n"
			"	inline __device__ const T& operator[](unsigned int idx) const { return *reinterpret_cast<const T*>(reinterpret_cast<const char*>(buffer) + idx * static_cast<unsigned int>(sizeof(T))); }\n"
			"};\n"
			"template<typename T>\n"
			"struct Outputs\n"
			"{\n"
			"	T* buffer;\n"
			"	inline __device__ Outputs(T* buffer) : buffer(buffer) {}\n"
			"	inline __device__ T& operator[](unsigned int idx) const { return *reinterpret_cast<T*>(reinterpret_cast<char*>(buffer) + idx * static_cast<unsigned int>(sizeof(T))); }\n"
			"};\n"
		);
	}
	else {
		sc->tempLen = sprintf(sc->tempStr,
			"template<typename T>\n"
			"using Inputs = const T*;\n"
			"template<typename T>\n"
			"using Outputs = T*;\n"
		);
	}
	PfAppendLine(sc);
	sc->tempLen = sprintf(sc->tempStr, "extern \"C\" __launch_bounds__(%" PRIi64 ") __global__ void VkFFT_main_R2C ", sc->localSize[0].data.i * sc->localSize[1].data.i * sc->localSize[2].data.i);
	PfAppendLine(sc);
	
	sc->tempLen = sprintf(sc->tempStr, "(const Inputs<%s> inputs, Outputs<%s> outputs", vecTypeInputMemory->data.s, vecTypeOutputMemory->data.s);
	PfAppendLine(sc);
	
	if (sc->LUT) {
		sc->tempLen = sprintf(sc->tempStr, ", const Inputs<%s> twiddleLUT", vecType->data.s);
		PfAppendLine(sc);
	}
	sc->tempLen = sprintf(sc->tempStr, ") {\n");
	PfAppendLine(sc);
	//sc->tempLen = sprintf(sc->tempStr, ", const PushConsts consts) {\n");
#elif((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
	sc->tempLen = sprintf(sc->tempStr, "__kernel __attribute__((reqd_work_group_size(%" PRIi64 ", %" PRIi64 ", %" PRIi64 "))) void VkFFT_main_R2C ", sc->localSize[0].data.i, sc->localSize[1].data.i, sc->localSize[2].data.i);
	PfAppendLine(sc);
	sc->tempLen = sprintf(sc->tempStr, "(__global %s* inputs, __global %s* outputs", vecTypeInputMemory->data.s, vecTypeOutputMemory->data.s);
	PfAppendLine(sc);
	int args_id = 2;
	if (sc->LUT) {
		sc->tempLen = sprintf(sc->tempStr, ", __global %s* twiddleLUT", vecType->data.s);
		PfAppendLine(sc);
		args_id++;
	}
	if (sc->pushConstantsStructSize > 0) {
		sc->tempLen = sprintf(sc->tempStr, ", PushConsts consts");
		PfAppendLine(sc);
	}
	sc->tempLen = sprintf(sc->tempStr, ") {\n");
	PfAppendLine(sc);
	//sc->tempLen = sprintf(sc->tempStr, ", const PushConsts consts) {\n");
#elif(VKFFT_BACKEND==5)
	sc->tempLen = sprintf(sc->tempStr, "kernel void VkFFT_main_R2C ");
	PfAppendLine(sc);

	sc->tempLen = sprintf(sc->tempStr, "(%s3 thread_position_in_grid [[thread_position_in_grid]], ", uintType->data.s);
	PfAppendLine(sc);

	sc->tempLen = sprintf(sc->tempStr, "%s3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], ", uintType->data.s);
	PfAppendLine(sc);

	sc->tempLen = sprintf(sc->tempStr, "%s3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], ", uintType->data.s);
	PfAppendLine(sc);

	sc->tempLen = sprintf(sc->tempStr, "device %s* inputs[[buffer(0)]], device %s* outputs[[buffer(1)]]", vecTypeInputMemory->data.s, vecTypeOutputMemory->data.s);
	PfAppendLine(sc);
	int args_id = 2;
	
	if (sc->LUT) {
		sc->tempLen = sprintf(sc->tempStr, ", constant %s* twiddleLUT[[buffer(%d)]]", vecType->data.s, args_id);
		PfAppendLine(sc);
		args_id++;
	}
	
	if (sc->pushConstantsStructSize > 0) {
		sc->tempLen = sprintf(sc->tempStr, ", constant PushConsts& consts[[buffer(%d)]]", args_id);
		PfAppendLine(sc);

		args_id++;
	}
	sc->tempLen = sprintf(sc->tempStr, ") {\n");
	PfAppendLine(sc);

	//sc->tempLen = sprintf(sc->tempStr, ", const PushConsts consts) {\n");
#endif
	return;
}
static inline void appendKernelEnd(VkFFTSpecializationConstantsLayout* sc) {
	if (sc->res != VKFFT_SUCCESS) return;
	sc->tempLen = sprintf(sc->tempStr, "}\n");
	PfAppendLine(sc);
	return;
}
#endif
