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
#ifndef VKFFT_KERNELUTILS_H
#define VKFFT_KERNELUTILS_H
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_StringManagement/vkFFT_StringManager.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_MathUtils/vkFFT_MathUtils.h"

static inline void appendLicense(VkFFTSpecializationConstantsLayout* sc) {
	if (sc->res != VKFFT_SUCCESS) return;
	sc->tempLen = sprintf(sc->tempStr, "\
// This file is part of VkFFT\n\
//\n\
// Copyright (C) 2021 - present Dmitrii Tolmachev <dtolm96@gmail.com>\n\
//\n\
// Permission is hereby granted, free of charge, to any person obtaining a copy\n\
// of this software and associated documentation files (the \"Software\"), to deal\n\
// in the Software without restriction, including without limitation the rights\n\
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n\
// copies of the Software, and to permit persons to whom the Software is\n\
// furnished to do so, subject to the following conditions:\n\
//\n\
// The above copyright notice and this permission notice shall be included in\n\
// all copies or substantial portions of the Software.\n\
//\n\
// THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n\
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n\
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE\n\
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n\
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n\
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN\n\
// THE SOFTWARE.\n");
	PfAppendLine(sc);
	return;
}

static inline void appendVersion(VkFFTSpecializationConstantsLayout* sc) {
	if (sc->res != VKFFT_SUCCESS) return;
#if(VKFFT_BACKEND==0)
	sc->tempLen = sprintf(sc->tempStr, "#version 450\n\n");
	PfAppendLine(sc);
#endif
	return;
}
static inline void appendExtensions(VkFFTSpecializationConstantsLayout* sc) {
	if (sc->res != VKFFT_SUCCESS) return;
#if(VKFFT_BACKEND==0)
	//sc->tempLen = sprintf(sc->tempStr, "#extension GL_EXT_debug_printf : require\n\n");
	//PfAppendLine(sc);
	//

	if ((((sc->floatTypeCode/10)%10) == 2) || (sc->useUint64)) {
		sc->tempLen = sprintf(sc->tempStr, "\
#extension GL_ARB_gpu_shader_fp64 : enable\n\
#extension GL_ARB_gpu_shader_int64 : enable\n\n");
		PfAppendLine(sc);
	}
	if ((((sc->floatTypeInputMemoryCode / 10) % 10) == 0) || (((sc->floatTypeOutputMemoryCode / 10) % 10) == 0) || (((sc->floatTypeCode / 10) % 10) == 0)) {
		sc->tempLen = sprintf(sc->tempStr, "#extension GL_EXT_shader_16bit_storage : require\n\n");
		PfAppendLine(sc);
	}
#elif(VKFFT_BACKEND==1)
	if ((((sc->floatTypeInputMemoryCode / 10) % 10) == 0) || (((sc->floatTypeOutputMemoryCode / 10) % 10) == 0) || (((sc->floatTypeCode / 10) % 10) == 0)) {
		sc->tempLen = sprintf(sc->tempStr, "\
#include <%s/include/cuda_fp16.h>\n", CUDA_TOOLKIT_ROOT_DIR);
		PfAppendLine(sc);
	}
#elif(VKFFT_BACKEND==2)
#ifdef VKFFT_OLD_ROCM
	sc->tempLen = sprintf(sc->tempStr, "\
#include <hip/hip_runtime.h>\n");
	PfAppendLine(sc);
#endif
	if ((((sc->floatTypeInputMemoryCode / 10) % 10) == 0) || (((sc->floatTypeOutputMemoryCode / 10) % 10) == 0) || (((sc->floatTypeCode / 10) % 10) == 0)) {
		sc->tempLen = sprintf(sc->tempStr, "\
#include <hip/hip_fp16.h>\n");
		PfAppendLine(sc);
	}
#elif((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
	if ((((sc->floatTypeCode / 10) % 10) == 2) || (sc->useUint64)) {
		sc->tempLen = sprintf(sc->tempStr, "\
#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n\n");
		PfAppendLine(sc);
	}
    if ((((sc->floatTypeInputMemoryCode / 10) % 10) == 0) || (((sc->floatTypeOutputMemoryCode / 10) % 10) == 0) || (((sc->floatTypeCode / 10) % 10) == 0)) {
        sc->tempLen = sprintf(sc->tempStr, "\
#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n\n");
        PfAppendLine(sc);
    }
#elif(VKFFT_BACKEND==5)
	sc->tempLen = sprintf(sc->tempStr, "\
#include <metal_math>\n\
using namespace metal;\n");
	PfAppendLine(sc);
#endif
	return;
}

static inline void appendSinCos20(VkFFTSpecializationConstantsLayout* sc) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer* vecType;
	PfGetTypeFromCode(sc, sc->vecTypeCode, &vecType);
	PfContainer* floatType;
	PfGetTypeFromCode(sc, sc->floatTypeCode, &floatType);
	PfContainer temp_double;
	temp_double.type = 32;
	PfContainer temp_name;
	PfAllocateContainerFlexible(sc, &temp_name, 50);
	temp_name.type = 100 + sc->floatTypeCode;
#if(VKFFT_BACKEND==0)
	temp_double.data.d = 0.63661977236758134307553505349006l;
	sprintf(temp_name.data.s, "loc_2_PI");
	PfDefineConstant(sc, &temp_name, &temp_double);
	temp_double.data.d = 1.5707963267948966192313216916398l;
	sprintf(temp_name.data.s, "loc_PI_2"); 
	PfDefineConstant(sc, &temp_name, &temp_double); 
	temp_double.data.d = 0.99999999999999999999962122687403772l;
	sprintf(temp_name.data.s, "a1");
	PfDefineConstant(sc, &temp_name, &temp_double);
	temp_double.data.d = -0.166666666666666666637194166219637268l;
	sprintf(temp_name.data.s, "a3");
	PfDefineConstant(sc, &temp_name, &temp_double);
	temp_double.data.d = 0.00833333333333333295212653322266277182l;
	sprintf(temp_name.data.s, "a5");
	PfDefineConstant(sc, &temp_name, &temp_double);
	temp_double.data.d = -0.000198412698412696489459896530659927773l;
	sprintf(temp_name.data.s, "a7");
	PfDefineConstant(sc, &temp_name, &temp_double);
	temp_double.data.d = 2.75573192239364018847578909205399262e-6l;
	sprintf(temp_name.data.s, "a9");
	PfDefineConstant(sc, &temp_name, &temp_double);
	temp_double.data.d = -2.50521083781017605729370231280411712e-8l;
	sprintf(temp_name.data.s, "a11");
	PfDefineConstant(sc, &temp_name, &temp_double);
	temp_double.data.d = 1.60590431721336942356660057796782021e-10l;
	sprintf(temp_name.data.s, "a13");
	PfDefineConstant(sc, &temp_name, &temp_double);
	temp_double.data.d = -7.64712637907716970380859898835680587e-13l;
	sprintf(temp_name.data.s, "a15");
	PfDefineConstant(sc, &temp_name, &temp_double);
	temp_double.data.d = 2.81018528153898622636194976499656274e-15l;
	sprintf(temp_name.data.s, "a17");
	PfDefineConstant(sc, &temp_name, &temp_double);
	temp_double.data.d = -7.97989713648499642889739108679114937e-18l;
	sprintf(temp_name.data.s, "ab");
	PfDefineConstant(sc, &temp_name, &temp_double);

	sc->tempLen = sprintf(sc->tempStr, "\
%s%s sincos_20(double x)\n\
{\n\
	//minimax coefs for sin for 0..pi/2 range\n\
	double y = abs(x * loc_2_PI);\n\
	double q = floor(y);\n\
	int quadrant = int(q);\n\
	double t = (quadrant & 1) != 0 ? 1 - y + q : y - q;\n\
	t *= loc_PI_2;\n\
	double t2 = t * t;\n\
	double r = fma(fma(fma(fma(fma(fma(fma(fma(fma(ab, t2, a17), t2, a15), t2, a13), t2, a11), t2, a9), t2, a7), t2, a5), t2, a3), t2 * t, t);\n\
	%s cos_sin;\n\
	cos_sin.x = ((quadrant == 0) || (quadrant == 3)) ? sqrt(1 - r * r) : -sqrt(1 - r * r);\n\
	r = x < 0 ? -r : r;\n\
	cos_sin.y = (quadrant & 2) != 0 ? -r : r;\n\
	return cos_sin;\n\
}\n\n", sc->functionDef.data.s, vecType->data.s, vecType->data.s);
	PfAppendLine(sc);
#endif
	PfDeallocateContainer(sc, &temp_name);

	return;
}

static inline void appendConversion(VkFFTSpecializationConstantsLayout* sc) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer* vecType;
	PfGetTypeFromCode(sc, sc->vecTypeCode, &vecType);
	PfContainer* floatType;
	PfGetTypeFromCode(sc, sc->floatTypeCode, &floatType);

	PfContainer* vecTypeDifferent;
	PfContainer* floatTypeDifferent;
	if (sc->floatTypeInputMemoryCode != sc->floatTypeCode) {
		PfGetTypeFromCode(sc, sc->vecTypeInputMemoryCode, &vecTypeDifferent);
		PfGetTypeFromCode(sc, sc->floatTypeInputMemoryCode, &floatTypeDifferent);
	}
	if (sc->floatTypeOutputMemoryCode != sc->floatTypeCode) {
		PfGetTypeFromCode(sc, sc->vecTypeOutputMemoryCode, &vecTypeDifferent);
		PfGetTypeFromCode(sc, sc->floatTypeOutputMemoryCode, &floatTypeDifferent);
	}

#if(VKFFT_BACKEND==0)
#else
	sc->tempLen = sprintf(sc->tempStr, "\
%s%s conv_%s(%s input)\n\
{\n\
	%s ret_val;\n\
	ret_val.x = (%s) input.x;\n\
	ret_val.y = (%s) input.y;\n\
	return ret_val;\n\
}\n\n", sc->functionDef.data.s, vecType->data.s, vecType->data.s, vecTypeDifferent->data.s, vecType->data.s, floatType->data.s, floatType->data.s);
	PfAppendLine(sc);
	sc->tempLen = sprintf(sc->tempStr, "\
%s%s conv_%s(%s input)\n\
{\n\
	%s ret_val;\n\
	ret_val.x = (%s) input.x;\n\
	ret_val.y = (%s) input.y;\n\
	return ret_val;\n\
}\n\n", sc->functionDef.data.s, vecTypeDifferent->data.s, vecTypeDifferent->data.s, vecType->data.s, vecTypeDifferent->data.s, floatTypeDifferent->data.s, floatTypeDifferent->data.s);
	PfAppendLine(sc);
#endif
	
	return;
}

static inline void appendBarrierVkFFT(VkFFTSpecializationConstantsLayout* sc) {
	if (sc->res != VKFFT_SUCCESS) return;
#if(VKFFT_BACKEND==0)
	sc->tempLen = sprintf(sc->tempStr, "barrier();\n\n");
	PfAppendLine(sc);
#elif(VKFFT_BACKEND==1)
	sc->tempLen = sprintf(sc->tempStr, "__syncthreads();\n\n");
	PfAppendLine(sc);
#elif(VKFFT_BACKEND==2)
	sc->tempLen = sprintf(sc->tempStr, "__syncthreads();\n\n");
	PfAppendLine(sc);
#elif((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
	sc->tempLen = sprintf(sc->tempStr, "barrier(CLK_LOCAL_MEM_FENCE);\n\n");
	PfAppendLine(sc);
#elif(VKFFT_BACKEND==5)
	sc->tempLen = sprintf(sc->tempStr, "threadgroup_barrier(mem_flags::mem_none);\n\n");
	PfAppendLine(sc);
#endif
	return;
}
#endif
