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

	if ((((sc->floatTypeCode/10)%10) == 2) || (((sc->floatTypeCode/10)%10) == 3) ||(sc->useUint64)) {
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
	if ((((sc->floatTypeCode / 10) % 10) == 2) || (((sc->floatTypeCode/10)%10) == 3) || (sc->useUint64)) {
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
static inline void appendQuadDoubleDoubleStruct(VkFFTSpecializationConstantsLayout* sc) {
#if(VKFFT_BACKEND==0)	
	/*sc->tempLen = sprintf(sc->tempStr, "\
struct pf_quad {\n\
%s x;\n\
%s y;\n\
};\n", sc->doubleDef.name, sc->doubleDef.name);
	PfAppendLine(sc);*/
	sc->tempLen = sprintf(sc->tempStr, "\
struct pf_quad2 {\n\
%s x;\n\
%s y;\n\
};\n", sc->quadDef.name, sc->quadDef.name);
	PfAppendLine(sc);
#else	
	/*sc->tempLen = sprintf(sc->tempStr, "\
typedef struct pf_quad {\n\
%s x;\n\
%s y;\n\
};\n", sc->doubleDef.name, sc->doubleDef.name);
	PfAppendLine(sc);*/
	sc->tempLen = sprintf(sc->tempStr, "\
typedef struct pf_quad2 {\n\
%s x;\n\
%s y;\n\
};\n", sc->quadDef.name, sc->quadDef.name);
	PfAppendLine(sc);
#endif
}
static inline void appendSinCos20(VkFFTSpecializationConstantsLayout* sc) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer* vecType;
	PfGetTypeFromCode(sc, sc->vecTypeCode, &vecType);
	PfContainer* floatType;
	PfGetTypeFromCode(sc, sc->floatTypeCode, &floatType);
	PfContainer temp_double;
	temp_double.type = 22;
	PfContainer temp_name = VKFFT_ZERO_INIT;
	temp_name.type = 100 + sc->floatTypeCode;
	PfAllocateContainerFlexible(sc, &temp_name, 50);
#if(VKFFT_BACKEND==0)
	temp_double.data.d = pfFPinit("0.63661977236758134307553505349006");
	sprintf(temp_name.name, "loc_2_PI");
	PfDefineConstant(sc, &temp_name, &temp_double);
	temp_double.data.d = pfFPinit("1.5707963267948966192313216916398");
	sprintf(temp_name.name, "loc_PI_2"); 
	PfDefineConstant(sc, &temp_name, &temp_double); 
	temp_double.data.d = pfFPinit("0.99999999999999999999962122687403772");
	sprintf(temp_name.name, "a1");
	PfDefineConstant(sc, &temp_name, &temp_double);
	temp_double.data.d = pfFPinit("-0.166666666666666666637194166219637268");
	sprintf(temp_name.name, "a3");
	PfDefineConstant(sc, &temp_name, &temp_double);
	temp_double.data.d = pfFPinit("0.00833333333333333295212653322266277182");
	sprintf(temp_name.name, "a5");
	PfDefineConstant(sc, &temp_name, &temp_double);
	temp_double.data.d = pfFPinit("-0.000198412698412696489459896530659927773");
	sprintf(temp_name.name, "a7");
	PfDefineConstant(sc, &temp_name, &temp_double);
	temp_double.data.d = pfFPinit("2.75573192239364018847578909205399262e-6");
	sprintf(temp_name.name, "a9");
	PfDefineConstant(sc, &temp_name, &temp_double);
	temp_double.data.d = pfFPinit("-2.50521083781017605729370231280411712e-8");
	sprintf(temp_name.name, "a11");
	PfDefineConstant(sc, &temp_name, &temp_double);
	temp_double.data.d = pfFPinit("1.60590431721336942356660057796782021e-10");
	sprintf(temp_name.name, "a13");
	PfDefineConstant(sc, &temp_name, &temp_double);
	temp_double.data.d = pfFPinit("-7.64712637907716970380859898835680587e-13");
	sprintf(temp_name.name, "a15");
	PfDefineConstant(sc, &temp_name, &temp_double);
	temp_double.data.d = pfFPinit("2.81018528153898622636194976499656274e-15");
	sprintf(temp_name.name, "a17");
	PfDefineConstant(sc, &temp_name, &temp_double);
	temp_double.data.d = pfFPinit("-7.97989713648499642889739108679114937e-18");
	sprintf(temp_name.name, "ab");
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
}\n\n", sc->functionDef.name, vecType->name, vecType->name);
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
	if (((sc->vecTypeCode % 100) / 10) == 3) {
		sc->tempLen = sprintf(sc->tempStr, "\
%s%s conv_%s_to_pf_quad(%s input)\n\
{\n\
	%s ret_val;\n\
	ret_val.x = (%s) input;\n\
	ret_val.y = (%s) 0;\n\
	return ret_val;\n\
}\n\n", sc->functionDef.name, sc->quadDef.name, sc->doubleDef.name, sc->doubleDef.name, sc->quadDef.name, sc->doubleDef.name, sc->doubleDef.name);
		PfAppendLine(sc);
		sc->tempLen = sprintf(sc->tempStr, "\
%s%s conv_pf_quad_to_%s(%s input)\n\
{\n\
	%s ret_val;\n\
	ret_val = (%s) input.x;\n\
	return ret_val;\n\
}\n\n", sc->functionDef.name, sc->doubleDef.name, sc->doubleDef.name, sc->quadDef.name, sc->doubleDef.name, sc->doubleDef.name);
		PfAppendLine(sc);

		sc->tempLen = sprintf(sc->tempStr, "\
%s%s conv_%s_to_%s(%s input)\n\
{\n\
	%s ret_val;\n\
	ret_val.x.x = (%s) input.x;\n\
	ret_val.y.x = (%s) input.y;\n\
	ret_val.x.y = (%s) 0;\n\
	ret_val.y.y = (%s) 0;\n\
	return ret_val;\n\
}\n\n", sc->functionDef.name, sc->quad2Def.name, sc->double2Def.name, sc->quad2Def.name, sc->double2Def.name, sc->quad2Def.name, sc->doubleDef.name, sc->doubleDef.name, sc->doubleDef.name, sc->doubleDef.name);
		PfAppendLine(sc);
		sc->tempLen = sprintf(sc->tempStr, "\
%s%s conv_%s_to_%s(%s input)\n\
{\n\
	%s ret_val;\n\
	ret_val.x = (%s) input.x.x;\n\
	ret_val.y = (%s) input.y.x;\n\
	return ret_val;\n\
}\n\n", sc->functionDef.name, sc->double2Def.name, sc->quad2Def.name, sc->double2Def.name, sc->quad2Def.name, sc->double2Def.name, sc->doubleDef.name, sc->doubleDef.name);
		PfAppendLine(sc);
	}
	else {
#if(VKFFT_BACKEND==0)
#else
		sc->tempLen = sprintf(sc->tempStr, "\
%s%s conv_%s(%s input)\n\
{\n\
	%s ret_val;\n\
	ret_val.x = (%s) input.x;\n\
	ret_val.y = (%s) input.y;\n\
	return ret_val;\n\
}\n\n", sc->functionDef.name, vecType->name, vecType->name, vecTypeDifferent->name, vecType->name, floatType->name, floatType->name);
		PfAppendLine(sc);
		sc->tempLen = sprintf(sc->tempStr, "\
%s%s conv_%s(%s input)\n\
{\n\
	%s ret_val;\n\
	ret_val.x = (%s) input.x;\n\
	ret_val.y = (%s) input.y;\n\
	return ret_val;\n\
}\n\n", sc->functionDef.name, vecTypeDifferent->name, vecTypeDifferent->name, vecType->name, vecTypeDifferent->name, floatTypeDifferent->name, floatTypeDifferent->name);
		PfAppendLine(sc);
#endif
	}
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
