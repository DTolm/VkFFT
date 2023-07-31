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
#ifndef VKFFT_RADIXKERNELS_H
#define VKFFT_RADIXKERNELS_H

#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_StringManagement/vkFFT_StringManager.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_MathUtils/vkFFT_MathUtils.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryTransfers/vkFFT_Transfers.h"

static inline void inlineRadixKernelVkFFT(VkFFTSpecializationConstantsLayout* sc, int64_t radix, int64_t stageSize, int64_t stageSizeSum, long double stageAngle, PfContainer* regID) {
	if (sc->res != VKFFT_SUCCESS) return;

	PfContainer temp_complex;
	temp_complex.type = 33;
	PfContainer temp_double;
	temp_double.type = 32;
	PfContainer temp_int;
	temp_int.type = 31;
	//sprintf(temp, "loc_0");

	switch (radix) {
	case 2: {
		if (stageSize == 1) {
			temp_complex.data.c[0] = 1;
			temp_complex.data.c[1] = 0;
			PfMov(sc, &sc->w, &temp_complex);		
		}
		else {
			if (sc->LUT) {
				if (sc->useCoalescedLUTUploadToSM) {
					appendSharedToRegisters(sc, &sc->w, &sc->stageInvocationID);
				}
				else {
					appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->LUTId);
				}
				if (stageAngle < 0) {
					PfConjugate(sc, &sc->w, &sc->w);
				}
			}
			else {
				PfSinCos(sc, &sc->w, &sc->angle);
			}
		}
		PfMul(sc, &sc->temp, &regID[1], &sc->w, 0);
		
		PfSub(sc, &regID[1], &regID[0], &sc->temp);
		
		PfAdd(sc, &regID[0], &regID[0], &sc->temp);
		
		break;
	}
	case 3: {

		PfContainer tf[2];
		for (int64_t i = 0; i < 2; i++){
			tf[i].type = 32;
		}
		
		tf[0].data.d = -0.5;
		tf[1].data.d = -0.8660254037844386467637231707529;

		if (stageSize == 1) {
			temp_complex.data.c[0] = 1;
			temp_complex.data.c[1] = 0;
			PfMov(sc, &sc->w, &temp_complex);		
		}
		else {
			if (sc->LUT) {
				if (sc->useCoalescedLUTUploadToSM) {
					appendSharedToRegisters(sc, &sc->w, &sc->stageInvocationID);
				}
				else {
					appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->LUTId);
				}
				if (stageAngle < 0) {
					PfConjugate(sc, &sc->w, &sc->w);
				}
			}
			else { 
				temp_double.data.d = 4.0 / 3.0;
				PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
				PfSinCos(sc, &sc->w, &sc->tempFloat);
			}
		}
		PfMul(sc, &sc->locID[2], &regID[2], &sc->w, 0);
		if (stageSize == 1) {
			temp_complex.data.c[0] = 1;
			temp_complex.data.c[1] = 0;
			PfMov(sc, &sc->w, &temp_complex);		
		}
		else {
			if (sc->LUT) {
				if (sc->useCoalescedLUTUploadToSM) {
					temp_int.data.i = stageSize;
					PfAdd(sc, &sc->sdataID, &sc->stageInvocationID, &temp_int);
					appendSharedToRegisters(sc, &sc->w, &sc->sdataID);
				}
				else {
					temp_double.data.d = 4.0 / 3.0;
					temp_int.data.i = stageSize;
					PfAdd(sc, &sc->inoutID, &sc->LUTId, &temp_int);
					appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->inoutID);
					
				}
				if (stageAngle < 0) {
					PfConjugate(sc, &sc->w, &sc->w);
					
				}
			}
			else {
				temp_double.data.d = 2.0 / 3.0;
				PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
				PfSinCos(sc, &sc->w, &sc->tempFloat);
			}
		}
		PfMul(sc, &sc->locID[1], &regID[1], &sc->w, 0);
		
		PfAdd(sc, &regID[1], &sc->locID[1], &sc->locID[2]);
		
		PfSub(sc, &regID[2], &sc->locID[1], &sc->locID[2]);
		
		PfAdd(sc, &sc->locID[0], &regID[0], &regID[1]);
		
		PfFMA(sc, &sc->locID[1], &regID[1], &tf[0], &regID[0]);
		
		PfMul(sc, &sc->locID[2], &regID[2], &tf[1], 0);
		
		PfMov(sc, &regID[0], &sc->locID[0]);
		
		if (stageAngle < 0)
		{
			PfShuffleComplex(sc, &regID[1], &sc->locID[1], &sc->locID[2], 0);
			
			PfShuffleComplexInv(sc, &regID[2], &sc->locID[1], &sc->locID[2], 0);
			
		}
		else {
			PfShuffleComplexInv(sc, &regID[1], &sc->locID[1], &sc->locID[2], 0);
			
			PfShuffleComplex(sc, &regID[2], &sc->locID[1], &sc->locID[2], 0);
			
		}

		break;
	}
	case 4: {
		/*if (&sc->LUT)
			&sc->tempLen = sprintf(&sc->tempStr, "void radix4(inout %s temp_0, inout %s temp_1, inout %s temp_2, inout %s temp_3, %s LUTId%s) {\n", vecType, vecType, vecType, vecType, uintType, convolutionInverse);
		else
			&sc->tempLen = sprintf(&sc->tempStr, "void radix4(inout %s temp_0, inout %s temp_1, inout %s temp_2, inout %s temp_3, %s angle%s) {\n", vecType, vecType, vecType, vecType, floatType, convolutionInverse);
		*/
		//PfAppendLine(sc, "	{\n");
		//&sc->tempLen = sprintf(&sc->tempStr, "	%s %s;\n", vecType, &sc->temp);
		//		
		if (stageSize == 1) {
			temp_complex.data.c[0] = 1;
			temp_complex.data.c[1] = 0;
			PfMov(sc, &sc->w, &temp_complex);	
			
		}
		else {
			if (sc->LUT) {
				if (sc->useCoalescedLUTUploadToSM) {
					appendSharedToRegisters(sc, &sc->w, &sc->stageInvocationID);
										
				}
				else {
					appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->LUTId);
										
				}
				if (stageAngle < 0) {
					PfConjugate(sc, &sc->w, &sc->w);
					
				}
			}
			else {
				PfSinCos(sc, &sc->w, &sc->angle);
			}
		}
		PfMul(sc, &sc->temp, &regID[2], &sc->w, 0);
		
		PfSub(sc, &regID[2], &regID[0], &sc->temp);
		
		PfAdd(sc, &regID[0], &regID[0], &sc->temp);
		
		PfMul(sc, &sc->temp, &regID[3], &sc->w, 0);
		
		PfSub(sc, &regID[3], &regID[1], &sc->temp);
		
		PfAdd(sc, &regID[1], &regID[1], &sc->temp);
		
		/*&sc->tempLen = sprintf(&sc->tempStr, "\
temp.x=temp%s.x*w.x-temp%s.y*w.y;\n\
temp.y = temp%s.y * w.x + temp%s.x * w.y;\n\
temp%s = temp%s - temp;\n\
temp%s = temp%s + temp;\n\n\
temp.x=temp%s.x*w.x-temp%s.y*w.y;\n\
temp.y = temp%s.y * w.x + temp%s.x * w.y;\n\
temp%s = temp%s - temp;\n\
temp%s = temp%s + temp;\n\n\
//DIF 2nd stage with angle\n", &regID[2], &regID[2], &regID[2], &regID[2], &regID[2], &regID[0], &regID[0], &regID[0], &regID[3], &regID[3], &regID[3], &regID[3], &regID[3], &regID[1], &regID[1], &regID[1]);*/
		if (stageSize == 1) {
			temp_complex.data.c[0] = 1;
			temp_complex.data.c[1] = 0;
			PfMov(sc, &sc->w, &temp_complex);	
			
		}
		else {
			if (sc->LUT) {
				if (sc->useCoalescedLUTUploadToSM) {
					temp_int.data.i = stageSize;
					PfAdd(sc, &sc->sdataID, &sc->stageInvocationID, &temp_int);
					appendSharedToRegisters(sc, &sc->w, &sc->sdataID);
				}
				else {
					temp_int.data.i = stageSize;
					PfAdd(sc, &sc->inoutID, &sc->LUTId, &temp_int);
					appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->inoutID);
					
				}
				if (stageAngle < 0) {
					PfConjugate(sc, &sc->w, &sc->w);
					
				}
			}
			else {
				temp_double.data.d = 0.5;
				PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
				PfSinCos(sc, &sc->w, &sc->tempFloat);
				
			}
		}
		PfMul(sc, &sc->temp, &regID[1], &sc->w, 0);
		
		PfSub(sc, &regID[1], &regID[0], &sc->temp);
		
		PfAdd(sc, &regID[0], &regID[0], &sc->temp);
		
		/*&sc->tempLen = sprintf(&sc->tempStr, "\
temp.x = temp%s.x * w.x - temp%s.y * w.y;\n\
temp.y = temp%s.y * w.x + temp%s.x * w.y;\n\
temp%s = temp%s - temp;\n\
temp%s = temp%s + temp;\n\n", &regID[1], &regID[1], &regID[1], &regID[1], &regID[1], &regID[0], &regID[0], &regID[0]);*/
		if (stageAngle < 0) {
			PfMov_x(sc, &sc->temp, &sc->w);
			
			PfMov_x_y(sc, &sc->w, &sc->w);
			PfMov_y_Neg_x(sc, &sc->w, &sc->temp);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(w.y, -w.x);\n\n", vecType);
		}
		else {
			PfMov_x(sc, &sc->temp, &sc->w);
			
			PfMov_x_Neg_y(sc, &sc->w, &sc->w);
			PfMov_y_x(sc, &sc->w, &sc->temp);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(-w.y, w.x);\n\n", vecType);
		}
		PfMul(sc, &sc->temp, &regID[3], &sc->w, 0);
		
		PfSub(sc, &regID[3], &regID[2], &sc->temp);
		
		PfAdd(sc, &regID[2], &regID[2], &sc->temp);
		
		//PfMov(sc, &sc->temp, &regID[1]);
		//

		uint64_t permute2[4] = { 0,2,1,3 };
		PfPermute(sc, permute2, 4, 1, regID, &sc->temp);
		

		/*PfMov(sc, &regID[1], &regID[2]);
		
		PfMov(sc, &regID[2], &sc->temp);
		*/
		/*PfAppendLine(sc, "	}\n");
		&sc->tempLen = sprintf(&sc->tempStr, "\
temp.x = temp%s.x * w.x - temp%s.y * w.y;\n\
temp.y = temp%s.y * w.x + temp%s.x * w.y;\n\
temp%s = temp%s - temp;\n\
temp%s = temp%s + temp;\n\n\
temp = temp%s;\n\
temp%s = temp%s;\n\
temp%s = temp;\n\
}\n", &regID[3], &regID[3], &regID[3], &regID[3], &regID[3], &regID[2], &regID[2], &regID[2], &regID[1], &regID[1], &regID[2], &regID[2]);*/
		break;
	}
	case 5: {
		/*if (sc->LUT) {
			&sc->tempLen = sprintf(&sc->tempStr, "void radix5(inout %s temp_0, inout %s temp_1, inout %s temp_2, inout %s temp_3, inout %s temp_4, %s LUTId) {\n", vecType, vecType, vecType, vecType, vecType, uintType);
		}
		else {
			&sc->tempLen = sprintf(&sc->tempStr, "void radix5(inout %s temp_0, inout %s temp_1, inout %s temp_2, inout %s temp_3, inout %s temp_4, %s angle) {\n", vecType, vecType, vecType, vecType, vecType, floatType);
		}*/
		PfContainer tf[5];
		for (int64_t i = 0; i < 5; i++){
			tf[i].type = 32;
		}
		tf[0].data.d = -0.5;
		tf[1].data.d = 1.538841768587626701285145288018455;
		tf[2].data.d = -0.363271264002680442947733378740309;
		tf[3].data.d = -0.809016994374947424102293417182819;
		tf[4].data.d = -0.587785252292473129168705954639073;

		/*for (uint64_t i = 0; i < 5; i++) {
			&sc->locID[i], (char*)malloc(sizeof(char) * 50);
			sprintf(&sc->locID[i], loc_%" PRIu64 "", i);
			&sc->tempLen = sprintf(&sc->tempStr, "	%s %s;\n", vecType, &sc->locID[i]);
			
			}*/
			/*&sc->tempLen = sprintf(&sc->tempStr, "	{\n\
	%s loc_0;\n	%s loc_1;\n	%s loc_2;\n	%s loc_3;\n	%s loc_4;\n", vecType, vecType, vecType, vecType, vecType);*/

		for (uint64_t i = radix - 1; i > 0; i--) {
			if (stageSize == 1) {
				temp_complex.data.c[0] = 1;
				temp_complex.data.c[1] = 0;
				PfMov(sc, &sc->w, &temp_complex);	
				
			}
			else {
				if (i == radix - 1) {
					if (sc->LUT) {
						if (sc->useCoalescedLUTUploadToSM) {
							appendSharedToRegisters(sc, &sc->w, &sc->stageInvocationID);
														
						}
						else {
							appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->LUTId);
														
						}
						if (stageAngle < 0) {
							PfConjugate(sc, &sc->w, &sc->w);
					
						}
					}
					else {
						temp_double.data.d = 2.0 * i / radix;
						PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
						PfSinCos(sc, &sc->w, &sc->tempFloat);
					}
				}
				else {
					if (sc->LUT) {
						if (sc->useCoalescedLUTUploadToSM) {
							temp_int.data.i = (radix - 1 - i) * stageSize;
							PfAdd(sc, &sc->sdataID, &sc->stageInvocationID, &temp_int);
							appendSharedToRegisters(sc, &sc->w, &sc->sdataID);
						}
						else {
							temp_int.data.i = (radix - 1 - i) * stageSize;
							PfAdd(sc, &sc->inoutID, &sc->LUTId, &temp_int);
							appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->inoutID);
							
						}
						if (stageAngle < 0) {
							PfConjugate(sc, &sc->w, &sc->w);
					
						}
					}
					else {
						temp_double.data.d = 2.0 * i / radix;
						PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
						PfSinCos(sc, &sc->w, &sc->tempFloat);
					}
				}
			}
			PfMul(sc, &sc->locID[i], &regID[i], &sc->w, 0);
			
			/*&sc->tempLen = sprintf(&sc->tempStr, "\
loc_%" PRIu64 ".x = temp%s.x * w.x - temp%s.y * w.y;\n\
loc_%" PRIu64 ".y = temp%s.y * w.x + temp%s.x * w.y;\n", i, &regID[i], &regID[i], i, &regID[i], &regID[i]);*/
		}
		PfAdd(sc, &regID[1], &sc->locID[1], &sc->locID[4]);
		
		PfAdd(sc, &regID[2], &sc->locID[2], &sc->locID[3]);
		
		PfSub(sc, &regID[3], &sc->locID[2], &sc->locID[3]);
		
		PfSub(sc, &regID[4], &sc->locID[1], &sc->locID[4]);
		
		PfSub(sc, &sc->locID[3], &regID[1], &regID[2]);
		
		PfAdd(sc, &sc->locID[4], &regID[3], &regID[4]);
		
		/*&sc->tempLen = sprintf(&sc->tempStr, "\
temp%s = loc_1 + loc_4;\n\
temp%s = loc_2 + loc_3;\n\
temp%s = loc_2 - loc_3;\n\
temp%s = loc_1 - loc_4;\n\
loc_3 = temp%s - temp%s;\n\
loc_4 = temp%s + temp%s;\n", &regID[1], &regID[2], &regID[3], &regID[4], &regID[1], &regID[2], &regID[3], &regID[4]);*/
		PfAdd(sc, &sc->locID[0], &regID[0], &regID[1]);
		
		PfAdd(sc, &sc->locID[0], &sc->locID[0], &regID[2]);
		
		PfFMA(sc, &sc->locID[1], &regID[1], &tf[0], &regID[0]);
		
		PfFMA(sc, &sc->locID[2], &regID[2], &tf[0], &regID[0]);
		
		PfMul(sc, &regID[3], &regID[3], &tf[1], 0);
		
		PfMul(sc, &regID[4], &regID[4], &tf[2], 0);
		
		PfMul(sc, &sc->locID[3], &sc->locID[3], &tf[3], 0);
		
		PfMul(sc, &sc->locID[4], &sc->locID[4], &tf[4], 0);
		
		/*&sc->tempLen = sprintf(&sc->tempStr, "\
loc_0 = temp%s + temp%s + temp%s;\n\
loc_1 = temp%s - 0.5 * temp%s;\n\
loc_2 = temp%s - 0.5 * temp%s;\n\
temp%s *= 1.538841768587626701285145288018455;\n\
temp%s *= -0.363271264002680442947733378740309;\n\
loc_3 *= -0.809016994374947424102293417182819;\n\
loc_4 *= -0.587785252292473129168705954639073;\n", &regID[0], &regID[1], &regID[2], &regID[0], &regID[1], &regID[0], &regID[2], &regID[3], &regID[4]);*/
		PfSub(sc, &sc->locID[1], &sc->locID[1], &sc->locID[3]);
		
		PfAdd(sc, &sc->locID[2], &sc->locID[2], &sc->locID[3]);
		
		PfAdd(sc, &sc->locID[3], &regID[3], &sc->locID[4]);
		
		PfAdd(sc, &sc->locID[4], &sc->locID[4], &regID[4]);
		
		PfMov(sc, &regID[0], &sc->locID[0]);
		
		/*&sc->tempLen = sprintf(&sc->tempStr, "\
loc_1 -= loc_3;\n\
loc_2 += loc_3;\n\
loc_3 = temp%s+loc_4;\n\
loc_4 += temp%s;\n\
temp%s = loc_0;\n", &regID[3], &regID[4], &regID[0]);*/

		if (stageAngle < 0)
		{
			PfShuffleComplex(sc, &regID[1], &sc->locID[1], &sc->locID[4], 0);
			
			PfShuffleComplex(sc, &regID[2], &sc->locID[2], &sc->locID[3], 0);
			
			PfShuffleComplexInv(sc, &regID[3], &sc->locID[2], &sc->locID[3], 0);
			
			PfShuffleComplexInv(sc, &regID[4], &sc->locID[1], &sc->locID[4], 0);
			
			/*&sc->tempLen = sprintf(&sc->tempStr, "\
temp%s.x = loc_1.x - loc_4.y; \n\
temp%s.y = loc_1.y + loc_4.x; \n\
temp%s.x = loc_2.x - loc_3.y; \n\
temp%s.y = loc_2.y + loc_3.x; \n\
temp%s.x = loc_2.x + loc_3.y; \n\
temp%s.y = loc_2.y - loc_3.x; \n\
temp%s.x = loc_1.x + loc_4.y; \n\
temp%s.y = loc_1.y - loc_4.x; \n", &regID[1], &regID[1], &regID[2], &regID[2], &regID[3], &regID[3], &regID[4], &regID[4]);*/
		}
		else {
			PfShuffleComplexInv(sc, &regID[1], &sc->locID[1], &sc->locID[4], 0);
			
			PfShuffleComplexInv(sc, &regID[2], &sc->locID[2], &sc->locID[3], 0);
			
			PfShuffleComplex(sc, &regID[3], &sc->locID[2], &sc->locID[3], 0);
			
			PfShuffleComplex(sc, &regID[4], &sc->locID[1], &sc->locID[4], 0);
			
			/*&sc->tempLen = sprintf(&sc->tempStr, "\
temp%s.x = loc_1.x + loc_4.y; \n\
temp%s.y = loc_1.y - loc_4.x; \n\
temp%s.x = loc_2.x + loc_3.y; \n\
temp%s.y = loc_2.y - loc_3.x; \n\
temp%s.x = loc_2.x - loc_3.y; \n\
temp%s.y = loc_2.y + loc_3.x; \n\
temp%s.x = loc_1.x - loc_4.y; \n\
temp%s.y = loc_1.y + loc_4.x; \n", &regID[1], &regID[1], &regID[2], &regID[2], &regID[3], &regID[3], &regID[4], &regID[4]);*/
		}

		//PfAppendLine(sc, "	}\n");
		break;
	}
	case 6: {
		PfContainer tf[2];
		for (int64_t i = 0; i < 2; i++){
			tf[i].type = 32;
		}
		//PfAppendLine(sc, "	{\n");
		

		tf[0].data.d = -0.5;
		tf[1].data.d = -0.8660254037844386467637231707529;
		for (uint64_t i = radix - 1; i > 0; i--) {
			if (stageSize == 1) {
				temp_complex.data.c[0] = 1;
				temp_complex.data.c[1] = 0;
				PfMov(sc, &sc->w, &temp_complex);	
				
			}
			else {
				if (i == radix - 1) {
					if (sc->LUT) {
						if (sc->useCoalescedLUTUploadToSM) {
							appendSharedToRegisters(sc, &sc->w, &sc->stageInvocationID);
														
						}
						else {
							appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->LUTId);
														
						}
						if (stageAngle < 0) {
							PfConjugate(sc, &sc->w, &sc->w);
					
						}
					}
					else {
						temp_double.data.d = 2.0 * i / radix;
						PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
						PfSinCos(sc, &sc->w, &sc->tempFloat);
					}
				}
				else {
					if (sc->LUT) {
						if (sc->useCoalescedLUTUploadToSM) {
							temp_int.data.i = (radix - 1 - i) * stageSize;
							PfAdd(sc, &sc->sdataID, &sc->stageInvocationID, &temp_int);
							appendSharedToRegisters(sc, &sc->w, &sc->sdataID);
						}
						else {
							temp_int.data.i = (radix - 1 - i) * stageSize;
							PfAdd(sc, &sc->inoutID, &sc->LUTId, &temp_int);
							appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->inoutID);
							
						}
						if (stageAngle < 0) {
							PfConjugate(sc, &sc->w, &sc->w);
					
						}
					}
					else {
						temp_double.data.d = 2.0 * i / radix;
						PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
						PfSinCos(sc, &sc->w, &sc->tempFloat);
					}
				}
			}
			PfMul(sc, &regID[i], &regID[i], &sc->w, &sc->temp);
			
		}
		//important
		//PfMov(sc, &regID[1], &sc->locID[1]);
		//

		//uint64_t P = 3;
		uint64_t Q = 2;
		for (uint64_t i = 0; i < Q; i++) {
			PfMov(sc, &sc->locID[0], &regID[i]);
			
			PfMov(sc, &sc->locID[1], &regID[i + Q]);
			
			PfMov(sc, &sc->locID[2], &regID[i + 2 * Q]);
			

			PfAdd(sc, &regID[i + Q], &sc->locID[1], &sc->locID[2]);
			
			PfSub(sc, &regID[i + 2 * Q], &sc->locID[1], &sc->locID[2]);
			

			PfAdd(sc, &sc->locID[0], &regID[i], &regID[i + Q]);
			
			PfFMA(sc, &sc->locID[1], &regID[i + Q], &tf[0], &regID[i]);
			
			PfMul(sc, &sc->locID[2], &regID[i + 2 * Q], &tf[1], 0);
			
			PfMov(sc, &regID[i], &sc->locID[0]);
			
			if (stageAngle < 0)
			{
				PfShuffleComplex(sc, &regID[i + Q], &sc->locID[1], &sc->locID[2], 0);
				
				PfShuffleComplexInv(sc, &regID[i + 2 * Q], &sc->locID[1], &sc->locID[2], 0);
				
			}
			else {
				PfShuffleComplexInv(sc, &regID[i + Q], &sc->locID[1], &sc->locID[2], 0);
				
				PfShuffleComplex(sc, &regID[i + 2 * Q], &sc->locID[1], &sc->locID[2], 0);
				
			}
		}

		PfMov(sc, &sc->temp, &regID[1]);
		
		PfSub(sc, &regID[1], &regID[0], &sc->temp);
		
		PfAdd(sc, &regID[0], &regID[0], &sc->temp);
		
		if (stageAngle < 0) {
			temp_complex.data.c[0] =  -0.5;
			temp_complex.data.c[1] = 0.8660254037844386467637231707529;
			PfMov(sc, &sc->w, &temp_complex);	
			
		}
		else {
			temp_complex.data.c[0] =  -0.5;
			temp_complex.data.c[1] = -0.8660254037844386467637231707529;
			PfMov(sc, &sc->w, &temp_complex);
			
		}

		PfMul(sc, &sc->temp, &regID[3], &sc->w, 0);
		
		PfSub(sc, &regID[3], &regID[2], &sc->temp);
		
		PfAdd(sc, &regID[2], &regID[2], &sc->temp);
		

		PfConjugate(sc, &sc->w, &sc->w);
		

		PfMul(sc, &sc->temp, &regID[5], &sc->w, 0);
		
		PfSub(sc, &regID[5], &regID[4], &sc->temp);
		
		PfAdd(sc, &regID[4], &regID[4], &sc->temp);
		

		uint64_t permute2[6] = { 0,3,4,1,2,5 };
		PfPermute(sc, permute2, 6, 1, regID, &sc->temp);
		

		/*PfMov(sc, &sc->temp, &regID[1]);
		
		PfMov(sc, &regID[1], &regID[3]);
		
		PfMov(sc, &regID[3], &sc->temp);
		

		PfMov(sc, &sc->temp, &regID[2]);
		
		PfMov(sc, &regID[2], &regID[4]);
		
		PfMov(sc, &regID[4], &sc->temp);
		*/
		break;
	}
	case 7: {
		/*if (sc->LUT) {
			&sc->tempLen = sprintf(&sc->tempStr, "void radix5(inout %s temp_0, inout %s temp_1, inout %s temp_2, inout %s temp_3, inout %s temp_4, %s LUTId) {\n", vecType, vecType, vecType, vecType, vecType, uintType);
		}
		else {
			&sc->tempLen = sprintf(&sc->tempStr, "void radix5(inout %s temp_0, inout %s temp_1, inout %s temp_2, inout %s temp_3, inout %s temp_4, %s angle) {\n", vecType, vecType, vecType, vecType, vecType, floatType);
		}*/
		PfContainer tf[8];
		for (int64_t i = 0; i < 8; i++){
			tf[i].type = 32;
		}
		//PfAppendLine(sc, "	{\n");
		tf[0].data.d = -1.16666666666666651863693004997913;
		tf[1].data.d = 0.79015646852540022404554065360571;
		tf[2].data.d = 0.05585426728964774240049351305970;
		tf[3].data.d = 0.73430220123575240531721419756650;
		if (stageAngle < 0) {
			tf[4].data.d = 0.44095855184409837868031445395900;
			tf[5].data.d = 0.34087293062393136944265847887436;
			tf[6].data.d = -0.53396936033772524066165487965918;
			tf[7].data.d = 0.87484229096165666561546458979137;
		}
		else {
			tf[4].data.d = -0.44095855184409837868031445395900;
			tf[5].data.d = -0.34087293062393136944265847887436;
			tf[6].data.d = 0.53396936033772524066165487965918;
			tf[7].data.d = -0.87484229096165666561546458979137;
		}
		/*for (uint64_t i = 0; i < 7; i++) {
			&sc->locID[i], (char*)malloc(sizeof(char) * 50);
			sprintf(&sc->locID[i], loc_%" PRIu64 "", i);
			&sc->tempLen = sprintf(&sc->tempStr, "	%s %s;\n", vecType, &sc->locID[i]);
			
			}*/
		for (uint64_t i = radix - 1; i > 0; i--) {
			if (stageSize == 1) {
				temp_complex.data.c[0] = 1;
				temp_complex.data.c[1] = 0;
				PfMov(sc, &sc->w, &temp_complex);	
				
			}
			else {
				if (i == radix - 1) {
					if (sc->LUT) {
						if (sc->useCoalescedLUTUploadToSM) {
							appendSharedToRegisters(sc, &sc->w, &sc->stageInvocationID);
														
						}
						else {
							appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->LUTId);
														
						}
						if (stageAngle < 0) {
							PfConjugate(sc, &sc->w, &sc->w);
					
						}
					}
					else {
						temp_double.data.d = 2.0 * i / radix;
						PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
						PfSinCos(sc, &sc->w, &sc->tempFloat);
					}
				}
				else {
					if (sc->LUT) {
						if (sc->useCoalescedLUTUploadToSM) {
							temp_int.data.i = (radix - 1 - i) * stageSize;
							PfAdd(sc, &sc->sdataID, &sc->stageInvocationID, &temp_int);
							appendSharedToRegisters(sc, &sc->w, &sc->sdataID);
						}
						else {
							temp_int.data.i = (radix - 1 - i) * stageSize;
							PfAdd(sc, &sc->inoutID, &sc->LUTId, &temp_int);
							appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->inoutID);
							
						}
						if (stageAngle < 0) {
							PfConjugate(sc, &sc->w, &sc->w);
					
						}
					}
					else {
						temp_double.data.d = 2.0 * i / radix;
						PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
						PfSinCos(sc, &sc->w, &sc->tempFloat);
					}
				}
			}
			PfMul(sc, &sc->locID[i], &regID[i], &sc->w, 0);
			
			/*&sc->tempLen = sprintf(&sc->tempStr, "\
loc_%" PRIu64 ".x = temp%s.x * w.x - temp%s.y * w.y;\n\
loc_%" PRIu64 ".y = temp%s.y * w.x + temp%s.x * w.y;\n", i, &regID[i], &regID[i], i, &regID[i], &regID[i]);*/
		}
		PfMov(sc, &sc->locID[0], &regID[0]);
		
		PfAdd(sc, &regID[0], &sc->locID[1], &sc->locID[6]);
		
		PfSub(sc, &regID[1], &sc->locID[1], &sc->locID[6]);
		
		PfAdd(sc, &regID[2], &sc->locID[2], &sc->locID[5]);
		
		PfSub(sc, &regID[3], &sc->locID[2], &sc->locID[5]);
		
		PfAdd(sc, &regID[4], &sc->locID[4], &sc->locID[3]);
		
		PfSub(sc, &regID[5], &sc->locID[4], &sc->locID[3]);
		
		/*&sc->tempLen = sprintf(&sc->tempStr, "\
loc_0 = temp%s;\n\
temp%s = loc_1 + loc_6;\n\
temp%s = loc_1 - loc_6;\n\
temp%s = loc_2 + loc_5;\n\
temp%s = loc_2 - loc_5;\n\
temp%s = loc_4 + loc_3;\n\
temp%s = loc_4 - loc_3;\n", &regID[0], &regID[0], &regID[1], &regID[2], &regID[3], &regID[4], &regID[5]);*/
		PfAdd(sc, &sc->locID[5], &regID[1], &regID[3]);
		
		PfAdd(sc, &sc->locID[5], &sc->locID[5], &regID[5]);
		
		PfAdd(sc, &sc->locID[1], &regID[0], &regID[2]);
		
		PfAdd(sc, &sc->locID[1], &sc->locID[1], &regID[4]);
		
		PfAdd(sc, &sc->locID[0], &sc->locID[0], &sc->locID[1]);
		
		/*&sc->tempLen = sprintf(&sc->tempStr, "\
loc_5 = temp%s + temp%s + temp%s;\n\
loc_1 = temp%s + temp%s + temp%s;\n\
loc_0 += loc_1;\n", &regID[1], &regID[3], &regID[5], &regID[0], &regID[2], &regID[4]);*/
		PfSub(sc, &sc->locID[2], &regID[0], &regID[4]);
		
		PfSub(sc, &sc->locID[3], &regID[4], &regID[2]);
		
		PfSub(sc, &sc->locID[4], &regID[2], &regID[0]);
		
		/*&sc->tempLen = sprintf(&sc->tempStr, "\
loc_2 = temp%s - temp%s;\n\
loc_3 = temp%s - temp%s;\n\
loc_4 = temp%s - temp%s;\n", &regID[0], &regID[4], &regID[4], &regID[2], &regID[2], &regID[0]);*/
		PfSub(sc, &regID[0], &regID[1], &regID[5]);
		
		PfSub(sc, &regID[2], &regID[5], &regID[3]);
		
		PfSub(sc, &regID[4], &regID[3], &regID[1]);
		
		/*&sc->tempLen = sprintf(&sc->tempStr, "\
temp%s = temp%s - temp%s;\n\
temp%s = temp%s - temp%s;\n\
temp%s = temp%s - temp%s;\n", &regID[0], &regID[1], &regID[5], &regID[2], &regID[5], &regID[3], &regID[4], &regID[3], &regID[1]);*/

		PfMul(sc, &sc->locID[1], &sc->locID[1], &tf[0], 0);
		
		PfMul(sc, &sc->locID[2], &sc->locID[2], &tf[1], 0);
		
		PfMul(sc, &sc->locID[3], &sc->locID[3], &tf[2], 0);
		
		PfMul(sc, &sc->locID[4], &sc->locID[4], &tf[3], 0);
		
		PfMul(sc, &sc->locID[5], &sc->locID[5], &tf[4], 0);
		
		PfMul(sc, &regID[0], &regID[0], &tf[5], 0);
		
		PfMul(sc, &regID[2], &regID[2], &tf[6], 0);
		
		PfMul(sc, &regID[4], &regID[4], &tf[7], 0);
		
		/*&sc->tempLen = sprintf(&sc->tempStr, "\
loc_1 *= -1.16666666666666651863693004997913;\n\
loc_2 *= 0.79015646852540022404554065360571;\n\
loc_3 *= 0.05585426728964774240049351305970;\n\
loc_4 *= 0.73430220123575240531721419756650;\n\
loc_5 *= 0.44095855184409837868031445395900;\n\
temp%s *= 0.34087293062393136944265847887436;\n\
temp%s *= -0.53396936033772524066165487965918;\n\
temp%s *= 0.87484229096165666561546458979137;\n", &regID[0], &regID[2], &regID[4]);*/

		PfSub(sc, &regID[5], &regID[4], &regID[2]);
		
		PfAddInv(sc, &regID[6], &regID[4], &regID[0]);
		
		PfAdd(sc, &regID[4], &regID[0], &regID[2]);
		
		/*&sc->tempLen = sprintf(&sc->tempStr, "\
temp%s = temp%s - temp%s;\n\
temp%s = - temp%s - temp%s;\n\
temp%s = temp%s + temp%s;\n", &regID[5], &regID[4], &regID[2], &regID[6], &regID[4], &regID[0], &regID[4], &regID[0], &regID[2]);*/
		PfAdd(sc, &regID[0], &sc->locID[0], &sc->locID[1]);
		
		PfAdd(sc, &regID[1], &sc->locID[2], &sc->locID[3]);
		
		PfSub(sc, &regID[2], &sc->locID[4], &sc->locID[3]);
		
		PfAddInv(sc, &regID[3], &sc->locID[2], &sc->locID[4]);
		
		/*&sc->tempLen = sprintf(&sc->tempStr, "\
temp%s = loc_0 + loc_1;\n\
temp%s = loc_2 + loc_3;\n\
temp%s = loc_4 - loc_3;\n\
temp%s = - loc_2 - loc_4;\n", &regID[0], &regID[1], &regID[2], &regID[3]);*/
		PfAdd(sc, &sc->locID[1], &regID[0], &regID[1]);
		
		PfAdd(sc, &sc->locID[2], &regID[0], &regID[2]);
		
		PfAdd(sc, &sc->locID[3], &regID[0], &regID[3]);
		
		PfAdd(sc, &sc->locID[4], &regID[4], &sc->locID[5]);
		
		PfAdd(sc, &sc->locID[6], &regID[6], &sc->locID[5]);
		
		PfAdd(sc, &sc->locID[5], &sc->locID[5], &regID[5]);
		
		PfMov(sc, &regID[0], &sc->locID[0]);
		
		/*&sc->tempLen = sprintf(&sc->tempStr, "\
loc_1 = temp%s + temp%s;\n\
loc_2 = temp%s + temp%s;\n\
loc_3 = temp%s + temp%s;\n\
loc_4 = temp%s + loc_5;\n\
loc_6 = temp%s + loc_5;\n\
loc_5 += temp%s;\n\
temp%s = loc_0;\n", &regID[0], &regID[1], &regID[0], &regID[2], &regID[0], &regID[3], &regID[4], &regID[6], &regID[5], &regID[0]);*/
		PfShuffleComplexInv(sc, &regID[1], &sc->locID[1], &sc->locID[4], 0);
		
		PfShuffleComplexInv(sc, &regID[2], &sc->locID[3], &sc->locID[6], 0);
		
		PfShuffleComplex(sc, &regID[3], &sc->locID[2], &sc->locID[5], 0);
		
		PfShuffleComplexInv(sc, &regID[4], &sc->locID[2], &sc->locID[5], 0);
		
		PfShuffleComplex(sc, &regID[5], &sc->locID[3], &sc->locID[6], 0);
		
		PfShuffleComplex(sc, &regID[6], &sc->locID[1], &sc->locID[4], 0);
		

		/*&sc->tempLen = sprintf(&sc->tempStr, "\
temp%s.x = loc_1.x + loc_4.y; \n\
temp%s.y = loc_1.y - loc_4.x; \n\
temp%s.x = loc_3.x + loc_6.y; \n\
temp%s.y = loc_3.y - loc_6.x; \n\
temp%s.x = loc_2.x - loc_5.y; \n\
temp%s.y = loc_2.y + loc_5.x; \n\
temp%s.x = loc_2.x + loc_5.y; \n\
temp%s.y = loc_2.y - loc_5.x; \n\
temp%s.x = loc_3.x - loc_6.y; \n\
temp%s.y = loc_3.y + loc_6.x; \n\
temp%s.x = loc_1.x - loc_4.y; \n\
temp%s.y = loc_1.y + loc_4.x; \n", &regID[1], &regID[1], &regID[2], &regID[2], &regID[3], &regID[3], &regID[4], &regID[4], &regID[5], &regID[5], &regID[6], &regID[6]);
		PfAppendLine(sc, "	}\n");*/
		/*for (uint64_t i = 0; i < 7; i++) {
			free(&sc->locID[i]);
		}*/
		break;
	}
	case 8: {
		/*if (&sc->LUT)
			&sc->tempLen = sprintf(&sc->tempStr, "void radix8(inout %s temp_0, inout %s temp_1, inout %s temp_2, inout %s temp_3, inout %s temp_4, inout %s temp_5, inout %s temp_6, inout %s temp_7, %s LUTId%s) {\n", vecType, vecType, vecType, vecType, vecType, vecType, vecType, vecType, uintType, convolutionInverse);
		else
			&sc->tempLen = sprintf(&sc->tempStr, "void radix8(inout %s temp_0, inout %s temp_1, inout %s temp_2, inout %s temp_3, inout %s temp_4, inout %s temp_5, inout %s temp_6, inout %s temp_7, %s angle%s) {\n", vecType, vecType, vecType, vecType, vecType, vecType, vecType, vecType, floatType, convolutionInverse);
		*/
		//PfAppendLine(sc, "	{\n");
		/*&sc->tempLen = sprintf(&sc->tempStr, "	%s %s;\n", vecType, &sc->temp);
		
			&sc->tempLen = sprintf(&sc->tempStr, "	%s %s;\n", vecType, iw);
			*/
		if (stageSize == 1) {
			temp_complex.data.c[0] = 1;
			temp_complex.data.c[1] = 0;
			PfMov(sc, &sc->w, &temp_complex);	
			
		}
		else {
			if (sc->LUT) {
				if (sc->useCoalescedLUTUploadToSM) {
					appendSharedToRegisters(sc, &sc->w, &sc->stageInvocationID);
										
				}
				else {
					appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->LUTId);
										
				}
				if (stageAngle < 0) {
					PfConjugate(sc, &sc->w, &sc->w);
					
				}
			}
			else {
				PfSinCos(sc, &sc->w, &sc->angle);
			}
		}
		for (uint64_t i = 0; i < 4; i++) {
			PfMul(sc, &sc->temp, &regID[i + 4], &sc->w, 0);
			
			PfSub(sc, &regID[i + 4], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
			/*&sc->tempLen = sprintf(&sc->tempStr, "\
temp.x=temp%s.x*w.x-temp%s.y*w.y;\n\
temp.y = temp%s.y * w.x + temp%s.x * w.y;\n\
temp%s = temp%s - temp;\n\
temp%s = temp%s + temp;\n\n", &regID[i + 4], &regID[i + 4], &regID[i + 4], &regID[i + 4], &regID[i + 4], &regID[i + 0], &regID[i + 0], &regID[i + 0]);*/
		}
		if (stageSize == 1) {
			temp_complex.data.c[0] = 1;
			temp_complex.data.c[1] = 0;
			PfMov(sc, &sc->w, &temp_complex);	
			
		}
		else {
			if (sc->LUT) {
				if (sc->useCoalescedLUTUploadToSM) {
					temp_int.data.i = stageSize;
					PfAdd(sc, &sc->sdataID, &sc->stageInvocationID, &temp_int);
					appendSharedToRegisters(sc, &sc->w, &sc->sdataID);
				}
				else {
					temp_int.data.i = stageSize;
					PfAdd(sc, &sc->inoutID, &sc->LUTId, &temp_int);
					appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->inoutID);
					
				}
				if (stageAngle < 0) {
					PfConjugate(sc, &sc->w, &sc->w);
					
				}
			}
			else {
				temp_double.data.d = 0.5;
				PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
				PfSinCos(sc, &sc->w, &sc->tempFloat);
			}
		}
		for (uint64_t i = 0; i < 2; i++) {
			PfMul(sc, &sc->temp, &regID[i + 2], &sc->w, 0);
			
			PfSub(sc, &regID[i + 2], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
			/*&sc->tempLen = sprintf(&sc->tempStr, "\
temp.x=temp%s.x*w.x-temp%s.y*w.y;\n\
temp.y = temp%s.y * w.x + temp%s.x * w.y;\n\
temp%s = temp%s - temp;\n\
temp%s = temp%s + temp;\n\n", &regID[i + 2], &regID[i + 2], &regID[i + 2], &regID[i + 2], &regID[i + 2], &regID[i + 0], &regID[i + 0], &regID[i + 0]);*/
		}
		if (stageAngle < 0) {
			
			PfMov_x_y(sc, &sc->iw, &sc->w);
			PfMov_y_Neg_x(sc, &sc->iw, &sc->w);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(w.y, -w.x);\n\n", vecType);
		}
		else {
			
			PfMov_x_Neg_y(sc, &sc->iw, &sc->w);
			PfMov_y_x(sc, &sc->iw, &sc->w);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	iw = %s(-w.y, w.x);\n\n", vecType);
		}

		for (uint64_t i = 4; i < 6; i++) {
			PfMul(sc, &sc->temp, &regID[i + 2], &sc->iw, 0);
			
			PfSub(sc, &regID[i + 2], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
			/*&sc->tempLen = sprintf(&sc->tempStr, "\
temp.x = temp%s.x * iw.x - temp%s.y * iw.y;\n\
temp.y = temp%s.y * iw.x + temp%s.x * iw.y;\n\
temp%s = temp%s - temp;\n\
temp%s = temp%s + temp;\n\n", &regID[i + 2], &regID[i + 2], &regID[i + 2], &regID[i + 2], &regID[i + 2], &regID[i + 0], &regID[i + 0], &regID[i + 0]);*/
		}
		if (stageSize == 1) {
			temp_complex.data.c[0] = 1;
			temp_complex.data.c[1] = 0;
			PfMov(sc, &sc->w, &temp_complex);	
			
		}
		else {
			if (sc->LUT) {
				if (sc->useCoalescedLUTUploadToSM) {
					temp_int.data.i = 2 * stageSize;
					PfAdd(sc, &sc->sdataID, &sc->stageInvocationID, &temp_int);
					appendSharedToRegisters(sc, &sc->w, &sc->sdataID);
				}
				else {
					temp_int.data.i = 2 * stageSize;
					PfAdd(sc, &sc->inoutID, &sc->LUTId, &temp_int);
					appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->inoutID);
					
				}
				if (stageAngle < 0) {
					PfConjugate(sc, &sc->w, &sc->w);
					
				}
			}
			else {
				temp_double.data.d = 0.25;
				PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
				PfSinCos(sc, &sc->w, &sc->tempFloat);
			}
		}
		PfMul(sc, &sc->temp, &regID[1], &sc->w, 0);
		
		PfSub(sc, &regID[1], &regID[0], &sc->temp);
		
		PfAdd(sc, &regID[0], &regID[0], &sc->temp);
		
		/*&sc->tempLen = sprintf(&sc->tempStr, "\
temp.x=temp%s.x*w.x-temp%s.y*w.y;\n\
temp.y = temp%s.y * w.x + temp%s.x * w.y;\n\
temp%s = temp%s - temp;\n\
temp%s = temp%s + temp;\n\n", &regID[1], &regID[1], &regID[1], &regID[1], &regID[1], &regID[0], &regID[0], &regID[0]);*/
		if (stageAngle < 0) {
			
			PfMov_x_y(sc, &sc->iw, &sc->w);
			PfMov_y_Neg_x(sc, &sc->iw, &sc->w);
			
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(w.y, -w.x);\n\n", vecType);
		}
		else {
			PfMov_x_Neg_y(sc, &sc->iw, &sc->w);
			PfMov_y_x(sc, &sc->iw, &sc->w);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	iw = %s(-w.y, w.x);\n\n", vecType);
		}
		PfMul(sc, &sc->temp, &regID[3], &sc->iw, 0);
		
		PfSub(sc, &regID[3], &regID[2], &sc->temp);
		
		PfAdd(sc, &regID[2], &regID[2], &sc->temp);
		
		/*&sc->tempLen = sprintf(&sc->tempStr, "\
temp.x = temp%s.x * iw.x - temp%s.y * iw.y;\n\
temp.y = temp%s.y * iw.x + temp%s.x * iw.y;\n\
temp%s = temp%s - temp;\n\
temp%s = temp%s + temp;\n\n", &regID[3], &regID[3], &regID[3], &regID[3], &regID[3], &regID[2], &regID[2], &regID[2]);*/
		if (stageAngle < 0) {
			temp_complex.data.c[0] = 0.70710678118654752440084436210485;
			temp_complex.data.c[1] = -0.70710678118654752440084436210485;
			PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
		
		}
		else {
			temp_complex.data.c[0] = 0.70710678118654752440084436210485;
			temp_complex.data.c[1] = 0.70710678118654752440084436210485;
			PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
		}
		PfMul(sc, &sc->temp, &regID[5], &sc->iw, 0);
		
		PfSub(sc, &regID[5], &regID[4], &sc->temp);
		
		PfAdd(sc, &regID[4], &regID[4], &sc->temp);
		
		/*&sc->tempLen = sprintf(&sc->tempStr, "\
temp.x = temp%s.x * iw.x - temp%s.y * iw.y;\n\
temp.y = temp%s.y * iw.x + temp%s.x * iw.y;\n\
temp%s = temp%s - temp;\n\
temp%s = temp%s + temp;\n\n", &regID[5], &regID[5], &regID[5], &regID[5], &regID[5], &regID[4], &regID[4], &regID[4]);*/
		if (stageAngle < 0) {
			PfMov_x_y(sc, &sc->w, &sc->iw);
			PfMov_y_Neg_x(sc, &sc->w, &sc->iw);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(iw.y, -iw.x);\n\n", vecType);
		}
		else {
			PfMov_x_Neg_y(sc, &sc->w, &sc->iw);
			PfMov_y_x(sc, &sc->w, &sc->iw);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(-iw.y, iw.x);\n\n", vecType);
		}
		PfMul(sc, &sc->temp, &regID[7], &sc->w, 0);
		
		PfSub(sc, &regID[7], &regID[6], &sc->temp);
		
		PfAdd(sc, &regID[6], &regID[6], &sc->temp);
		

		uint64_t permute2[8] = { 0,4,2,6,1,5,3,7 };
		PfPermute(sc, permute2, 8, 1, regID, &sc->temp);
		
		/*
		
		PfMov(sc, &sc->temp, &regID[1]);
		
		PfMov(sc, &regID[1], &regID[4]);
		
		PfMov(sc, &regID[4], &sc->temp);
		
		PfMov(sc, &sc->temp, &regID[3]);
		
		PfMov(sc, &regID[3], &regID[6]);
		
		PfMov(sc, &regID[6], &sc->temp);
		*/
		/*&sc->tempLen = sprintf(&sc->tempStr, "\
temp.x = temp%s.x * w.x - temp%s.y * w.y;\n\
temp.y = temp%s.y * w.x + temp%s.x * w.y;\n\
temp%s = temp%s - temp;\n\
temp%s = temp%s + temp;\n\n\
temp = temp%s;\n\
temp%s = temp%s;\n\
temp%s = temp;\n\n\
temp = temp%s;\n\
temp%s = temp%s;\n\
temp%s = temp;\n\
}\n\n", &regID[7], &regID[7], &regID[7], &regID[7], &regID[7], &regID[6], &regID[6], &regID[6], &regID[1], &regID[1], &regID[4], &regID[4], &regID[3], &regID[3], &regID[6], &regID[6]);
			//PfAppendLine(sc, "	}\n");*/

		break;
	}
	case 9: {
		PfContainer tf[2];
		//PfAppendLine(sc, "	{\n");
		for (int64_t i = 0; i < 2; i++){
			tf[i].type = 32;
		}

		tf[0].data.d = -0.5;
		tf[1].data.d = -0.8660254037844386467637231707529;
		for (uint64_t i = radix - 1; i > 0; i--) {
			if (stageSize == 1) {
				temp_complex.data.c[0] = 1;
				temp_complex.data.c[1] = 0;
				PfMov(sc, &sc->w, &temp_complex);	
				
			}
			else {
				if (i == radix - 1) {
					if (sc->LUT) {
						if (sc->useCoalescedLUTUploadToSM) {
							appendSharedToRegisters(sc, &sc->w, &sc->stageInvocationID);
														
						}
						else {
							appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->LUTId);
														
						}
						if (stageAngle < 0) {
							PfConjugate(sc, &sc->w, &sc->w);
					
						}
					}
					else {
						temp_double.data.d = 2.0 * i / radix;
						PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
						PfSinCos(sc, &sc->w, &sc->tempFloat);
					}
				}
				else {
					if (sc->LUT) {
						if (sc->useCoalescedLUTUploadToSM) {
							temp_int.data.i = (radix - 1 - i) * stageSize;
							PfAdd(sc, &sc->sdataID, &sc->stageInvocationID, &temp_int);
							appendSharedToRegisters(sc, &sc->w, &sc->sdataID);
						}
						else {
							temp_int.data.i = (radix - 1 - i) * stageSize;
							PfAdd(sc, &sc->inoutID, &sc->LUTId, &temp_int);
							appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->inoutID);
							
						}
						if (stageAngle < 0) {
							PfConjugate(sc, &sc->w, &sc->w);
					
						}
					}
					else {
						temp_double.data.d = 2.0 * i / radix;
						PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
						PfSinCos(sc, &sc->w, &sc->tempFloat);
					}
				}
			}
			PfMul(sc, &regID[i], &regID[i], &sc->w, &sc->temp);
			
		}
		//important
		//PfMov(sc, &regID[1], &sc->locID[1]);
		//
		//PfMov(sc, &regID[2], &sc->locID[2]);
		//
		uint64_t P = 3;
		uint64_t Q = 3;
		for (uint64_t i = 0; i < Q; i++) {
			PfMov(sc, &sc->locID[0], &regID[i]);
			
			PfMov(sc, &sc->locID[1], &regID[i + Q]);
			
			PfMov(sc, &sc->locID[2], &regID[i + 2 * Q]);
			
			PfAdd(sc, &regID[i + Q], &sc->locID[1], &sc->locID[2]);
			
			PfSub(sc, &regID[i + 2 * Q], &sc->locID[1], &sc->locID[2]);
			

			PfAdd(sc, &sc->locID[0], &regID[i], &regID[i + Q]);
			
			PfFMA(sc, &sc->locID[1], &regID[i + Q], &tf[0], &regID[i]);
			
			PfMul(sc, &sc->locID[2], &regID[i + 2 * Q], &tf[1], 0);
			
			PfMov(sc, &regID[i], &sc->locID[0]);
			
			if (stageAngle < 0)
			{
				PfShuffleComplex(sc, &regID[i + Q], &sc->locID[1], &sc->locID[2], 0);
				
				PfShuffleComplexInv(sc, &regID[i + 2 * Q], &sc->locID[1], &sc->locID[2], 0);
				
			}
			else {
				PfShuffleComplexInv(sc, &regID[i + Q], &sc->locID[1], &sc->locID[2], 0);
				
				PfShuffleComplex(sc, &regID[i + 2 * Q], &sc->locID[1], &sc->locID[2], 0);
				
			}
		}


		for (uint64_t i = 0; i < P; i++) {
			if (i > 0) {
				if (stageAngle < 0) {
					temp_complex.data.c[0] = cos(2 * i * sc->double_PI / radix);
					temp_complex.data.c[1] = -sin(2 * i * sc->double_PI / radix);
					PfMov(sc, &sc->w, &temp_complex);	
					
				}
				else {
					temp_complex.data.c[0] = cos(2 * i * sc->double_PI / radix);
					temp_complex.data.c[1] = sin(2 * i * sc->double_PI / radix);
					PfMov(sc, &sc->w, &temp_complex);	
					
				}
				PfMul(sc, &sc->locID[1], &regID[Q * i + 1], &sc->w, &sc->temp);
				
				if (stageAngle < 0) {
					temp_complex.data.c[0] = cos(4 * i * sc->double_PI / radix);
					temp_complex.data.c[1] = -sin(4 * i * sc->double_PI / radix);
					PfMov(sc, &sc->w, &temp_complex);	
					
				}
				else {
					temp_complex.data.c[0] = cos(4 * i * sc->double_PI / radix);
					temp_complex.data.c[1] = sin(4 * i * sc->double_PI / radix);
					PfMov(sc, &sc->w, &temp_complex);	
					
				}
				PfMul(sc, &sc->locID[2], &regID[Q * i + 2], &sc->w, &sc->temp);
				
			}
			else {
				PfMov(sc, &sc->locID[1], &regID[1]);
				
				PfMov(sc, &sc->locID[2], &regID[2]);
				
			}

			PfAdd(sc, &regID[Q * i + 1], &sc->locID[1], &sc->locID[2]);
			
			PfSub(sc, &regID[Q * i + 2], &sc->locID[1], &sc->locID[2]);
			

			PfAdd(sc, &sc->locID[0], &regID[Q * i], &regID[Q * i + 1]);
			
			PfFMA(sc, &sc->locID[1], &regID[Q * i + 1], &tf[0], &regID[Q * i]);
			
			PfMul(sc, &sc->locID[2], &regID[Q * i + 2], &tf[1], 0);
			
			PfMov(sc, &regID[Q * i], &sc->locID[0]);
			
			if (stageAngle < 0)
			{
				PfShuffleComplex(sc, &regID[Q * i + 1], &sc->locID[1], &sc->locID[2], 0);
				
				PfShuffleComplexInv(sc, &regID[Q * i + 2], &sc->locID[1], &sc->locID[2], 0);
				
			}
			else {
				PfShuffleComplexInv(sc, &regID[Q * i + 1], &sc->locID[1], &sc->locID[2], 0);
				
				PfShuffleComplex(sc, &regID[Q * i + 2], &sc->locID[1], &sc->locID[2], 0);
				
			}
		}

		uint64_t permute2[9] = { 0,3,6,1,4,7,2,5,8 };
		PfPermute(sc, permute2, 9, 1, regID, &sc->temp);
		

		/*PfMov(sc, &sc->temp, &regID[1]);
		
		PfMov(sc, &regID[1], &regID[3]);
		
		PfMov(sc, &regID[3], &sc->temp);
		

		PfMov(sc, &sc->temp, &regID[2]);
		
		PfMov(sc, &regID[2], &regID[4]);
		
		PfMov(sc, &regID[4], &sc->temp);
		*/
		break;
	}
	case 10: {
		PfContainer tf[5];
		for (int64_t i = 0; i < 5; i++){
			tf[i].type = 32;
		}
		//PfAppendLine(sc, "	{\n");
		
		tf[0].data.d = -0.5;
		tf[1].data.d = 1.538841768587626701285145288018455;
		tf[2].data.d = -0.363271264002680442947733378740309;
		tf[3].data.d = -0.809016994374947424102293417182819;
		tf[4].data.d = -0.587785252292473129168705954639073;
		for (uint64_t i = radix - 1; i > 0; i--) {
			if (stageSize == 1) {
				temp_complex.data.c[0] = 1;
				temp_complex.data.c[1] = 0;
				PfMov(sc, &sc->w, &temp_complex);	
				
			}
			else {
				if (i == radix - 1) {
					if (sc->LUT) {
						if (sc->useCoalescedLUTUploadToSM) {
							appendSharedToRegisters(sc, &sc->w, &sc->stageInvocationID);
														
						}
						else {
							appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->LUTId);
														
						}
						if (stageAngle < 0) {
							PfConjugate(sc, &sc->w, &sc->w);
					
						}
					}
					else {
						temp_double.data.d = 2.0 * i / radix;
						PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
						PfSinCos(sc, &sc->w, &sc->tempFloat);
					}
				}
				else {
					if (sc->LUT) {
						if (sc->useCoalescedLUTUploadToSM) {
							temp_int.data.i = (radix - 1 - i) * stageSize;
							PfAdd(sc, &sc->sdataID, &sc->stageInvocationID, &temp_int);
							appendSharedToRegisters(sc, &sc->w, &sc->sdataID);
						}
						else {
							temp_int.data.i = (radix - 1 - i) * stageSize;
							PfAdd(sc, &sc->inoutID, &sc->LUTId, &temp_int);
							appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->inoutID);
							
						}
						if (stageAngle < 0) {
							PfConjugate(sc, &sc->w, &sc->w);
					
						}
					}
					else {
						temp_double.data.d = 2.0 * i / radix;
						PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
						PfSinCos(sc, &sc->w, &sc->tempFloat);
					}
				}
			}
			PfMul(sc, &regID[i], &regID[i], &sc->w, &sc->temp);
			
		}
		//important
		//PfMov(sc, &regID[1], &sc->locID[1]);
		//

		uint64_t P = 5;
		uint64_t Q = 2;
		for (uint64_t i = 0; i < Q; i++) {
			PfMov(sc, &sc->locID[0], &regID[i]);
			
			PfMov(sc, &sc->locID[1], &regID[i + Q]);
			
			PfMov(sc, &sc->locID[2], &regID[i + 2 * Q]);
			
			PfMov(sc, &sc->locID[3], &regID[i + 3 * Q]);
			
			PfMov(sc, &sc->locID[4], &regID[i + 4 * Q]);
			

			PfAdd(sc, &regID[i + Q], &sc->locID[1], &sc->locID[4]);
			
			PfAdd(sc, &regID[i + 2 * Q], &sc->locID[2], &sc->locID[3]);
			
			PfSub(sc, &regID[i + 3 * Q], &sc->locID[2], &sc->locID[3]);
			
			PfSub(sc, &regID[i + 4 * Q], &sc->locID[1], &sc->locID[4]);
			
			PfSub(sc, &sc->locID[3], &regID[i + Q], &regID[i + 2 * Q]);
			
			PfAdd(sc, &sc->locID[4], &regID[i + 3 * Q], &regID[i + 4 * Q]);
			

			PfAdd(sc, &sc->locID[0], &regID[i], &regID[i + Q]);
			
			PfAdd(sc, &sc->locID[0], &sc->locID[0], &regID[i + 2 * Q]);
			
			PfFMA(sc, &sc->locID[1], &regID[i + Q], &tf[0], &regID[i]);
			
			PfFMA(sc, &sc->locID[2], &regID[i + 2 * Q], &tf[0], &regID[i]);
			
			PfMul(sc, &regID[i + 3 * Q], &regID[i + 3 * Q], &tf[1], 0);
			
			PfMul(sc, &regID[i + 4 * Q], &regID[i + 4 * Q], &tf[2], 0);
			
			PfMul(sc, &sc->locID[3], &sc->locID[3], &tf[3], 0);
			
			PfMul(sc, &sc->locID[4], &sc->locID[4], &tf[4], 0);
			

			PfSub(sc, &sc->locID[1], &sc->locID[1], &sc->locID[3]);
			
			PfAdd(sc, &sc->locID[2], &sc->locID[2], &sc->locID[3]);
			
			PfAdd(sc, &sc->locID[3], &regID[i + 3 * Q], &sc->locID[4]);
			
			PfAdd(sc, &sc->locID[4], &sc->locID[4], &regID[i + 4 * Q]);
			
			PfMov(sc, &regID[i], &sc->locID[0]);
			

			if (stageAngle < 0)
			{
				PfShuffleComplex(sc, &regID[i + Q], &sc->locID[1], &sc->locID[4], 0);
				
				PfShuffleComplex(sc, &regID[i + 2 * Q], &sc->locID[2], &sc->locID[3], 0);
				
				PfShuffleComplexInv(sc, &regID[i + 3 * Q], &sc->locID[2], &sc->locID[3], 0);
				
				PfShuffleComplexInv(sc, &regID[i + 4 * Q], &sc->locID[1], &sc->locID[4], 0);
				
			}
			else {
				PfShuffleComplexInv(sc, &regID[i + Q], &sc->locID[1], &sc->locID[4], 0);
				
				PfShuffleComplexInv(sc, &regID[i + 2 * Q], &sc->locID[2], &sc->locID[3], 0);
				
				PfShuffleComplex(sc, &regID[i + 3 * Q], &sc->locID[2], &sc->locID[3], 0);
				
				PfShuffleComplex(sc, &regID[i + 4 * Q], &sc->locID[1], &sc->locID[4], 0);
				
			}

		}


		for (uint64_t i = 0; i < P; i++) {
			if (i > 0) {
				if (stageAngle < 0) {
					temp_complex.data.c[0] = cos(2 * i * sc->double_PI / radix);
					temp_complex.data.c[1] = -sin(2 * i * sc->double_PI / radix);
					PfMov(sc, &sc->w, &temp_complex);	
					
				}
				else {
					temp_complex.data.c[0] = cos(2 * i * sc->double_PI / radix);
					temp_complex.data.c[1] = sin(2 * i * sc->double_PI / radix);
					PfMov(sc, &sc->w, &temp_complex);	
					
				}
				PfMul(sc, &sc->temp, &regID[Q * i + 1], &sc->w, 0);
			}
			else {
				PfMov(sc, &sc->temp, &regID[Q * i + 1]);
				
			}
			PfSub(sc, &regID[Q * i + 1], &regID[Q * i], &sc->temp);
			
			PfAdd(sc, &regID[Q * i], &regID[Q * i], &sc->temp);
			
		}

		uint64_t permute2[10] = { 0, 2, 4, 6, 8, 1, 3, 5, 7, 9 };
		PfPermute(sc, permute2, 10, 1, regID, &sc->temp);
		break;
	}
	case 11: {
		PfContainer tf_x[20];
		PfContainer tf_y[20];
		for (int64_t i = 0; i < 20; i++){
			tf_x[i].type = 32;
			tf_y[i].type = 32;
		}
		
		tf_x[0].data.d = 8.4125353283118116886306336876800e-01;
		tf_x[1].data.d = -9.5949297361449738990105129410324e-01;
		tf_x[2].data.d = -1.4231483827328514046015907335008e-01;
		tf_x[3].data.d = -6.5486073394528506407246543075118e-01;
		tf_x[4].data.d = 4.1541501300188642567903264668505e-01;
		tf_x[5].data.d = 8.4125353283118116886306336876800e-01;
		tf_x[6].data.d = -9.5949297361449738990105129410324e-01;
		tf_x[7].data.d = -1.4231483827328514046015907335008e-01;
		tf_x[8].data.d = -6.5486073394528506407246543075118e-01;
		tf_x[9].data.d = 4.1541501300188642567903264668505e-01;
		if (stageAngle < 0) {
			tf_y[0].data.d = -5.4064081745559758210122047739077e-01;
			tf_y[1].data.d = 2.8173255684142969773359373164556e-01;
			tf_y[2].data.d = -9.8982144188093273235937163967435e-01;
			tf_y[3].data.d = 7.5574957435425828375808593451168e-01;
			tf_y[4].data.d = 9.0963199535451837136413102968824e-01;
			tf_y[5].data.d = 5.4064081745559758210122047739077e-01;
			tf_y[6].data.d = -2.8173255684142969773359373164556e-01;
			tf_y[7].data.d = 9.8982144188093273235937163967435e-01;
			tf_y[8].data.d = -7.5574957435425828375808593451168e-01;
			tf_y[9].data.d = -9.0963199535451837136413102968824e-01;
		}
		else {
			tf_y[0].data.d = 5.4064081745559758210122047739077e-01;
			tf_y[1].data.d = -2.8173255684142969773359373164556e-01;
			tf_y[2].data.d = 9.8982144188093273235937163967435e-01;
			tf_y[3].data.d = -7.5574957435425828375808593451168e-01;
			tf_y[4].data.d = -9.0963199535451837136413102968824e-01;
			tf_y[5].data.d = -5.4064081745559758210122047739077e-01;
			tf_y[6].data.d = 2.8173255684142969773359373164556e-01;
			tf_y[7].data.d = -9.8982144188093273235937163967435e-01;
			tf_y[8].data.d = 7.5574957435425828375808593451168e-01;
			tf_y[9].data.d = 9.0963199535451837136413102968824e-01;
		}
		for (uint64_t i = radix - 1; i > 0; i--) {
			if (stageSize == 1) {
				temp_complex.data.c[0] = 1;
				temp_complex.data.c[1] = 0;
			PfMov(sc, &sc->w, &temp_complex);	
				
			}
			else {
				if (i == radix - 1) {
					if (sc->LUT) {
						if (sc->useCoalescedLUTUploadToSM) {
							appendSharedToRegisters(sc, &sc->w, &sc->stageInvocationID);
														
						}
						else {
							appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->LUTId);
														
						}
						if (stageAngle < 0) {
							PfConjugate(sc, &sc->w, &sc->w);
					
						}
					}
					else {
						temp_double.data.d = 2.0 * i / radix;
						PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
						PfSinCos(sc, &sc->w, &sc->tempFloat);
					}
				}
				else {
					if (sc->LUT) {
						if (sc->useCoalescedLUTUploadToSM) {
							temp_int.data.i = (radix - 1 - i) * stageSize;
							PfAdd(sc, &sc->sdataID, &sc->stageInvocationID, &temp_int);
							appendSharedToRegisters(sc, &sc->w, &sc->sdataID);
						}
						else {
							temp_int.data.i = (radix - 1 - i) * stageSize;
							PfAdd(sc, &sc->inoutID, &sc->LUTId, &temp_int);
							appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->inoutID);
							
						}
						if (stageAngle < 0) {
							PfConjugate(sc, &sc->w, &sc->w);
							
						}
					}
					else {
						temp_double.data.d = 2.0 * i / radix;
						PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
						PfSinCos(sc, &sc->w, &sc->tempFloat);
					}
				}
			}
			PfMul(sc, &sc->locID[i], &regID[i], &sc->w, 0);
			
		}
		PfMov(sc, &sc->locID[0], &regID[0]);
		
		uint64_t permute[11] = { 0,1,2,4,8,5,10,9,7,3,6 };
		PfPermute(sc, permute, 11, 0, 0, &sc->w);
		
		for (uint64_t i = 0; i < 5; i++) {
			PfSub_x(sc, &regID[i + 6], &sc->locID[i + 1], &sc->locID[i + 6]);
			
			PfAdd_x(sc, &regID[i + 1], &sc->locID[i + 1], &sc->locID[i + 6]);
			
			PfAdd_y(sc, &regID[i + 6], &sc->locID[i + 1], &sc->locID[i + 6]);
			
			PfSub_y(sc, &regID[i + 1], &sc->locID[i + 1], &sc->locID[i + 6]);
			
		}
		for (uint64_t i = 0; i < 5; i++) {
			PfAdd_x(sc, &regID[0], &regID[0], &regID[i + 1]);
			
			PfAdd_y(sc, &regID[0], &regID[0], &regID[i + 6]);
			
		}
		for (uint64_t i = 1; i < 6; i++) {
			PfMov(sc, &sc->locID[i], &sc->locID[0]);
			
			
		}
		for (uint64_t i = 6; i < 11; i++) {
			PfSetToZero(sc, &sc->locID[i]);
		}
		for (uint64_t i = 0; i < 5; i++) {
			for (uint64_t j = 0; j < 5; j++) {
				uint64_t id = ((10 - i) + j) % 10;
				PfFMA3_const_w(sc, &sc->locID[j + 1], &sc->locID[j + 6], &regID[i + 1], &tf_x[id], &tf_y[id], &regID[i + 6], &sc->w);
				
			}
		}
		for (uint64_t i = 1; i < 6; i++) {
			PfSub_x(sc, &regID[i], &sc->locID[i], &sc->locID[i + 5]);
			
			PfAdd_y(sc, &regID[i], &sc->locID[i], &sc->locID[i + 5]);
			
		}
		for (uint64_t i = 1; i < 6; i++) {
			PfAdd_x(sc, &regID[i + 5], &sc->locID[i], &sc->locID[i + 5]);
			
			PfSub_y(sc, &regID[i + 5], &sc->locID[i], &sc->locID[i + 5]);
			
		}

		uint64_t permute2[11] = { 0,1,10,3,9,7,2,4,8,5,6 };
		PfPermute(sc, permute2, 11, 1, regID, &sc->w);
		break;
	}
	case 12: {
		PfContainer tf[2];
		for (int64_t i = 0; i < 2; i++){
			tf[i].type = 32;
		}
		//PfAppendLine(sc, "	{\n");
		
		tf[0].data.d = -0.5;
		tf[1].data.d = -0.8660254037844386467637231707529;
		for (uint64_t i = radix - 1; i > 0; i--) {
			if (stageSize == 1) {
				temp_complex.data.c[0] = 1;
				temp_complex.data.c[1] = 0;
				PfMov(sc, &sc->w, &temp_complex);	

			}
			else {
				if (i == radix - 1) {
					if (sc->LUT) {
						if (sc->useCoalescedLUTUploadToSM) {
							appendSharedToRegisters(sc, &sc->w, &sc->stageInvocationID);
														
						}
						else {
							appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->LUTId);
														
						}
						if (stageAngle < 0) {
							PfConjugate(sc, &sc->w, &sc->w);
							
						}
					}
					else {
						temp_double.data.d = 2.0 * i / radix;
						PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
						PfSinCos(sc, &sc->w, &sc->tempFloat);
					}
				}
				else {
					if (sc->LUT) {
						if (sc->useCoalescedLUTUploadToSM) {
							temp_int.data.i = (radix - 1 - i) * stageSize;
							PfAdd(sc, &sc->sdataID, &sc->stageInvocationID, &temp_int);
							appendSharedToRegisters(sc, &sc->w, &sc->sdataID);
						}
						else {
							temp_int.data.i = (radix - 1 - i) * stageSize;
							PfAdd(sc, &sc->inoutID, &sc->LUTId, &temp_int);
							appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->inoutID);
							
						}
						if (stageAngle < 0) {
							PfConjugate(sc, &sc->w, &sc->w);
							
						}
					}
					else {
						temp_double.data.d = 2.0 * i / radix;
						PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
						PfSinCos(sc, &sc->w, &sc->tempFloat);
					}
				}
			}
			PfMul(sc, &regID[i], &regID[i], &sc->w, &sc->temp);
			
		}
		//important
		//PfMov(sc, &regID[1], &sc->locID[1]);
		//
		//PfMov(sc, &regID[2], &sc->locID[2]);
		//
		uint64_t P = 3;
		uint64_t Q = 4;
		for (uint64_t i = 0; i < Q; i++) {
			PfMov(sc, &sc->locID[0], &regID[i]);
			
			PfMov(sc, &sc->locID[1], &regID[i + Q]);
			
			PfMov(sc, &sc->locID[2], &regID[i + 2 * Q]);
			
			PfAdd(sc, &regID[i + Q], &sc->locID[1], &sc->locID[2]);
			
			PfSub(sc, &regID[i + 2 * Q], &sc->locID[1], &sc->locID[2]);
			

			PfAdd(sc, &sc->locID[0], &regID[i], &regID[i + Q]);
			
			PfFMA(sc, &sc->locID[1], &regID[i + Q], &tf[0], &regID[i]);
			
			PfMul(sc, &sc->locID[2], &regID[i + 2 * Q], &tf[1], 0);
			
			PfMov(sc, &regID[i], &sc->locID[0]);
			
			if (stageAngle < 0)
			{
				PfShuffleComplex(sc, &regID[i + Q], &sc->locID[1], &sc->locID[2], 0);
				
				PfShuffleComplexInv(sc, &regID[i + 2 * Q], &sc->locID[1], &sc->locID[2], 0);
				
			}
			else {
				PfShuffleComplexInv(sc, &regID[i + Q], &sc->locID[1], &sc->locID[2], 0);
				
				PfShuffleComplex(sc, &regID[i + 2 * Q], &sc->locID[1], &sc->locID[2], 0);
				
			}
		}


		for (uint64_t i = 0; i < P; i++) {
			for (uint64_t j = 0; j < Q; j++) {
				if (i > 0) {
					if (stageAngle < 0) {
						temp_complex.data.c[0] = cos(2 * i * j * sc->double_PI / radix);
						temp_complex.data.c[1] = -sin(2 * i * j * sc->double_PI / radix);
						PfMov(sc, &sc->w, &temp_complex);	
						
					}
					else {
						temp_complex.data.c[0] = cos(2 * i * j * sc->double_PI / radix);
						temp_complex.data.c[1] = sin(2 * i * j * sc->double_PI / radix);
						PfMov(sc, &sc->w, &temp_complex);	
						
					}
					PfMul(sc, &regID[Q * i + j], &regID[Q * i + j], &sc->w, &sc->temp);
					
				}
			}
			PfMov(sc, &sc->temp, &regID[Q * i + 2]);
			
			PfSub(sc, &regID[Q * i + 2], &regID[Q * i], &regID[Q * i + 2]);
			
			PfAdd(sc, &regID[Q * i], &regID[Q * i], &sc->temp);
			

			PfMov(sc, &sc->temp, &regID[Q * i + 3]);
			
			PfSub(sc, &regID[Q * i + 3], &regID[Q * i + 1], &regID[Q * i + 3]);
			
			PfAdd(sc, &regID[Q * i + 1], &regID[Q * i + 1], &sc->temp);
			

			PfMov(sc, &sc->temp, &regID[Q * i + 1]);
			
			PfSub(sc, &regID[Q * i + 1], &regID[Q * i], &regID[Q * i + 1]);
			
			PfAdd(sc, &regID[Q * i], &regID[Q * i], &sc->temp);
			

			if (stageAngle < 0) {
				PfMov_x_y(sc, &sc->temp, &regID[Q * i + 3]);
				PfMov_y_Neg_x(sc, &sc->temp, &regID[Q * i + 3]);
				
			}
			else {
				PfMov_x_Neg_y(sc, &sc->temp, &regID[Q * i + 3]);
				PfMov_y_x(sc, &sc->temp, &regID[Q * i + 3]);
				
			}
			PfSub(sc, &regID[Q * i + 3], &regID[Q * i + 2], &sc->temp);
			
			PfAdd(sc, &regID[Q * i + 2], &regID[Q * i + 2], &sc->temp);
			
		}

		uint64_t permute2[12] = { 0,4,8,2,6,10,1,5,9,3,7,11 };
		PfPermute(sc, permute2, 12, 1, regID, &sc->temp);
		
		break;
	}
	case 13: {
		PfContainer tf_x[20];
		for (int64_t i = 0; i < 20; i++){
			tf_x[i].type = 32;
		}
		PfContainer tf_y[20];
		for (int64_t i = 0; i < 20; i++){
			tf_y[i].type = 32;
		}
		
		tf_x[0].data.d = 8.8545602565320989587194927539215e-01;
		tf_x[1].data.d = -9.7094181742605202719252621701429e-01;
		tf_x[2].data.d = 1.2053668025532305345994812592614e-01;
		tf_x[3].data.d = -7.4851074817110109868448578063216e-01;
		tf_x[4].data.d = -3.5460488704253562600274447824678e-01;
		tf_x[5].data.d = 5.6806474673115580237845248512407e-01;
		tf_x[6].data.d = 8.8545602565320989608878970988926e-01;
		tf_x[7].data.d = -9.7094181742605202719252621701429e-01;
		tf_x[8].data.d = 1.2053668025532305324988395500707e-01;
		tf_x[9].data.d = -7.4851074817110109863027567200788e-01;
		tf_x[10].data.d = -3.5460488704253562600274447824678e-01;
		tf_x[11].data.d = 5.6806474673115580248687270237262e-01;
		if (stageAngle < 0) {
			tf_y[0].data.d = -4.6472317204376854566250792943904e-01;
			tf_y[1].data.d = 2.3931566428755776706062234626682e-01;
			tf_y[2].data.d = 9.9270887409805399278096144088934e-01;
			tf_y[3].data.d = -6.6312265824079520232193704631918e-01;
			tf_y[4].data.d = 9.3501624268541482344965776185575e-01;
			tf_y[5].data.d = 8.2298386589365639468820687318917e-01;
			tf_y[6].data.d = 4.6472317204376854531014222338126e-01;
			tf_y[7].data.d = -2.3931566428755776695220212901827e-01;
			tf_y[8].data.d = -9.9270887409805399283517154951362e-01;
			tf_y[9].data.d = 6.6312265824079520243035726356773e-01;
			tf_y[10].data.d = -9.3501624268541482344965776185575e-01;
			tf_y[11].data.d = -8.2298386589365639457978665594062e-01;
		}
		else {
			tf_y[0].data.d = 4.6472317204376854566250792943904e-01;
			tf_y[1].data.d = -2.3931566428755776706062234626682e-01;
			tf_y[2].data.d = -9.9270887409805399278096144088934e-01;
			tf_y[3].data.d = 6.6312265824079520232193704631918e-01;
			tf_y[4].data.d = -9.3501624268541482344965776185575e-01;
			tf_y[5].data.d = -8.2298386589365639468820687318917e-01;
			tf_y[6].data.d = -4.6472317204376854531014222338126e-01;
			tf_y[7].data.d = 2.3931566428755776695220212901827e-01;
			tf_y[8].data.d = 9.9270887409805399283517154951362e-01;
			tf_y[9].data.d = -6.6312265824079520243035726356773e-01;
			tf_y[10].data.d = 9.3501624268541482344965776185575e-01;
			tf_y[11].data.d = 8.2298386589365639457978665594062e-01;
		}
		for (uint64_t i = radix - 1; i > 0; i--) {
			if (stageSize == 1) {
				temp_complex.data.c[0] = 1;
				temp_complex.data.c[1] = 0;
				PfMov(sc, &sc->w, &temp_complex);	
				
			}
			else {
				if (i == radix - 1) {
					if (sc->LUT) {
						if (sc->useCoalescedLUTUploadToSM) {
							appendSharedToRegisters(sc, &sc->w, &sc->stageInvocationID);
														
						}
						else {
							appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->LUTId);
														
						}
						if (stageAngle < 0) {
							PfConjugate(sc, &sc->w, &sc->w);
							
						}
					}
					else {
						temp_double.data.d = 2.0 * i / radix;
						PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
						PfSinCos(sc, &sc->w, &sc->tempFloat);
					}
				}
				else {
					if (sc->LUT) {
						if (sc->useCoalescedLUTUploadToSM) {
							temp_int.data.i = (radix - 1 - i) * stageSize;
							PfAdd(sc, &sc->sdataID, &sc->stageInvocationID, &temp_int);
							appendSharedToRegisters(sc, &sc->w, &sc->sdataID);
						}
						else {
							temp_int.data.i = (radix - 1 - i) * stageSize;
							PfAdd(sc, &sc->inoutID, &sc->LUTId, &temp_int);
							appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->inoutID);
							
						}
						if (stageAngle < 0) {
							PfConjugate(sc, &sc->w, &sc->w);
							
						}
					}
					else {
						temp_double.data.d = 2.0 * i / radix;
						PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
						PfSinCos(sc, &sc->w, &sc->tempFloat);
					}
				}
			}
			PfMul(sc, &sc->locID[i], &regID[i], &sc->w, 0);
			
		}
		PfMov(sc, &sc->locID[0], &regID[0]);
		
		uint64_t permute[13] = { 0, 1, 2, 4, 8, 3, 6, 12, 11, 9, 5, 10, 7 };
		PfPermute(sc, permute, 13, 0, 0, &sc->w);
		
		for (uint64_t i = 0; i < 6; i++) {
			PfSub_x(sc, &regID[i + 7], &sc->locID[i + 1], &sc->locID[i + 7]);
			
			PfAdd_x(sc, &regID[i + 1], &sc->locID[i + 1], &sc->locID[i + 7]);
			
			PfAdd_y(sc, &regID[i + 7], &sc->locID[i + 1], &sc->locID[i + 7]);
			
			PfSub_y(sc, &regID[i + 1], &sc->locID[i + 1], &sc->locID[i + 7]);
			
		}
		for (uint64_t i = 0; i < 6; i++) {
			PfAdd_x(sc, &regID[0], &regID[0], &regID[i + 1]);
			
			PfAdd_y(sc, &regID[0], &regID[0], &regID[i + 7]);
			
		}
		for (uint64_t i = 1; i < 7; i++) {
			PfMov(sc, &sc->locID[i], &sc->locID[0]);
			
		}
		for (uint64_t i = 7; i < 13; i++) {
			PfSetToZero(sc, &sc->locID[i]);
		}
		for (uint64_t i = 0; i < 6; i++) {
			for (uint64_t j = 0; j < 6; j++) {
				uint64_t id = ((12 - i) + j) % 12;
				PfFMA3_const_w(sc, &sc->locID[j + 1], &sc->locID[j + 7], &regID[i + 1], &tf_x[id], &tf_y[id], &regID[i + 7], &sc->w);
				
			}
		}
		for (uint64_t i = 1; i < 7; i++) {
			PfSub_x(sc, &regID[i], &sc->locID[i], &sc->locID[i + 6]);
			
			PfAdd_y(sc, &regID[i], &sc->locID[i], &sc->locID[i + 6]);
			
		}
		for (uint64_t i = 1; i < 7; i++) {
			PfAdd_x(sc, &regID[i + 6], &sc->locID[i], &sc->locID[i + 6]);
			
			PfSub_y(sc, &regID[i + 6], &sc->locID[i], &sc->locID[i + 6]);
			
		}

		uint64_t permute2[13] = { 0,1,12,9,11,4,8,2,10,5,3,6,7 };
		PfPermute(sc, permute2, 13, 1, regID, &sc->w);
		//
		break;
	}
	case 14: {
		PfContainer tf[8];
		for (int64_t i = 0; i < 8; i++){
			tf[i].type = 32;
		}
		//PfAppendLine(sc, "	{\n");
		
		tf[0].data.d = -1.16666666666666651863693004997913;
		tf[1].data.d = 0.79015646852540022404554065360571;
		tf[2].data.d = 0.05585426728964774240049351305970;
		tf[3].data.d = 0.73430220123575240531721419756650;
		if (stageAngle < 0) {
			tf[4].data.d = 0.44095855184409837868031445395900;
			tf[5].data.d = 0.34087293062393136944265847887436;
			tf[6].data.d = -0.53396936033772524066165487965918;
			tf[7].data.d = 0.87484229096165666561546458979137;
		}
		else {
			tf[4].data.d = -0.44095855184409837868031445395900;
			tf[5].data.d = -0.34087293062393136944265847887436;
			tf[6].data.d = 0.53396936033772524066165487965918;
			tf[7].data.d = -0.87484229096165666561546458979137;
		}
		for (uint64_t i = radix - 1; i > 0; i--) {
			if (stageSize == 1) {
				temp_complex.data.c[0] = 1;
				temp_complex.data.c[1] = 0;
				PfMov(sc, &sc->w, &temp_complex);	
				
			}
			else {
				if (i == radix - 1) {
					if (sc->LUT) {
						if (sc->useCoalescedLUTUploadToSM) {
							appendSharedToRegisters(sc, &sc->w, &sc->stageInvocationID);
														
						}
						else {
							appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->LUTId);
														
						}
						if (stageAngle < 0) {
							PfConjugate(sc, &sc->w, &sc->w);
							
						}
					}
					else {
						temp_double.data.d = 2.0 * i / radix;
						PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
						PfSinCos(sc, &sc->w, &sc->tempFloat);
					}
				}
				else {
					if (sc->LUT) {
						if (sc->useCoalescedLUTUploadToSM) {
							temp_int.data.i = (radix - 1 - i) * stageSize;
							PfAdd(sc, &sc->sdataID, &sc->stageInvocationID, &temp_int);
							appendSharedToRegisters(sc, &sc->w, &sc->sdataID);
						}
						else {
							temp_int.data.i = (radix - 1 - i) * stageSize;
							PfAdd(sc, &sc->inoutID, &sc->LUTId, &temp_int);
							appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->inoutID);
							
						}
						if (stageAngle < 0) {
							PfConjugate(sc, &sc->w, &sc->w);
							
						}
					}
					else {
						temp_double.data.d = 2.0 * i / radix;
						PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
						PfSinCos(sc, &sc->w, &sc->tempFloat);
					}
				}
			}
			PfMul(sc, &regID[i], &regID[i], &sc->w, &sc->temp);
			
		}
		//important
		//PfMov(sc, &regID[1], &sc->locID[1]);
		//

		uint64_t P = 7;
		uint64_t Q = 2;
		for (uint64_t i = 0; i < Q; i++) {
			PfMov(sc, &sc->locID[0], &regID[i]);
			
			PfMov(sc, &sc->locID[1], &regID[i + Q]);
			
			PfMov(sc, &sc->locID[2], &regID[i + 2 * Q]);
			
			PfMov(sc, &sc->locID[3], &regID[i + 3 * Q]);
			
			PfMov(sc, &sc->locID[4], &regID[i + 4 * Q]);
			
			PfMov(sc, &sc->locID[5], &regID[i + 5 * Q]);
			
			PfMov(sc, &sc->locID[6], &regID[i + 6 * Q]);
			

			PfAdd(sc, &regID[i], &sc->locID[1], &sc->locID[6]);
			
			PfSub(sc, &regID[i + Q], &sc->locID[1], &sc->locID[6]);
			
			PfAdd(sc, &regID[i + 2 * Q], &sc->locID[2], &sc->locID[5]);
			
			PfSub(sc, &regID[i + 3 * Q], &sc->locID[2], &sc->locID[5]);
			
			PfAdd(sc, &regID[i + 4 * Q], &sc->locID[4], &sc->locID[3]);
			
			PfSub(sc, &regID[i + 5 * Q], &sc->locID[4], &sc->locID[3]);
			

			PfAdd(sc, &sc->locID[5], &regID[i + Q], &regID[i + 3 * Q]);
			
			PfAdd(sc, &sc->locID[5], &sc->locID[5], &regID[i + 5 * Q]);
			
			PfAdd(sc, &sc->locID[1], &regID[i], &regID[i + 2 * Q]);
			
			PfAdd(sc, &sc->locID[1], &sc->locID[1], &regID[i + 4 * Q]);
			
			PfAdd(sc, &sc->locID[0], &sc->locID[0], &sc->locID[1]);
			

			PfSub(sc, &sc->locID[2], &regID[i], &regID[i + 4 * Q]);
			
			PfSub(sc, &sc->locID[3], &regID[i + 4 * Q], &regID[i + 2 * Q]);
			
			PfSub(sc, &sc->locID[4], &regID[i + 2 * Q], &regID[i]);
			

			PfSub(sc, &regID[i], &regID[i + Q], &regID[i + 5 * Q]);
			
			PfSub(sc, &regID[i + 2 * Q], &regID[i + 5 * Q], &regID[i + 3 * Q]);
			
			PfSub(sc, &regID[i + 4 * Q], &regID[i + 3 * Q], &regID[i + Q]);
			

			PfMul(sc, &sc->locID[1], &sc->locID[1], &tf[0], 0);
			
			PfMul(sc, &sc->locID[2], &sc->locID[2], &tf[1], 0);
			
			PfMul(sc, &sc->locID[3], &sc->locID[3], &tf[2], 0);
			
			PfMul(sc, &sc->locID[4], &sc->locID[4], &tf[3], 0);
			
			PfMul(sc, &sc->locID[5], &sc->locID[5], &tf[4], 0);
			
			PfMul(sc, &regID[i], &regID[i], &tf[5], 0);
			
			PfMul(sc, &regID[i + 2 * Q], &regID[i + 2 * Q], &tf[6], 0);
			
			PfMul(sc, &regID[i + 4 * Q], &regID[i + 4 * Q], &tf[7], 0);
			

			PfSub(sc, &regID[i + 5 * Q], &regID[i + 4 * Q], &regID[i + 2 * Q]);
			
			PfAddInv(sc, &regID[i + 6 * Q], &regID[i + 4 * Q], &regID[i]);
			
			PfAdd(sc, &regID[i + 4 * Q], &regID[i], &regID[i + 2 * Q]);
			
			PfAdd(sc, &regID[i], &sc->locID[0], &sc->locID[1]);
			
			PfAdd(sc, &regID[i + Q], &sc->locID[2], &sc->locID[3]);
			
			PfSub(sc, &regID[i + 2 * Q], &sc->locID[4], &sc->locID[3]);
			
			PfAddInv(sc, &regID[i + 3 * Q], &sc->locID[2], &sc->locID[4]);
			
			PfAdd(sc, &sc->locID[1], &regID[i], &regID[i + Q]);
			
			PfAdd(sc, &sc->locID[2], &regID[i], &regID[i + 2 * Q]);
			
			PfAdd(sc, &sc->locID[3], &regID[i], &regID[i + 3 * Q]);
			
			PfAdd(sc, &sc->locID[4], &regID[i + 4 * Q], &sc->locID[5]);
			
			PfAdd(sc, &sc->locID[6], &regID[i + 6 * Q], &sc->locID[5]);
			
			PfAdd(sc, &sc->locID[5], &sc->locID[5], &regID[i + 5 * Q]);
			
			PfMov(sc, &regID[i], &sc->locID[0]);
			
			PfShuffleComplexInv(sc, &regID[i + Q], &sc->locID[1], &sc->locID[4], 0);
			
			PfShuffleComplexInv(sc, &regID[i + 2 * Q], &sc->locID[3], &sc->locID[6], 0);
			
			PfShuffleComplex(sc, &regID[i + 3 * Q], &sc->locID[2], &sc->locID[5], 0);
			
			PfShuffleComplexInv(sc, &regID[i + 4 * Q], &sc->locID[2], &sc->locID[5], 0);
			
			PfShuffleComplex(sc, &regID[i + 5 * Q], &sc->locID[3], &sc->locID[6], 0);
			
			PfShuffleComplex(sc, &regID[i + 6 * Q], &sc->locID[1], &sc->locID[4], 0);
			

		}


		for (uint64_t i = 0; i < P; i++) {
			if (i > 0) {
				if (stageAngle < 0) {
					temp_complex.data.c[0] = cos(2 * i * sc->double_PI / radix);
					temp_complex.data.c[1] = -sin(2 * i * sc->double_PI / radix);
					PfMov(sc, &sc->w, &temp_complex);	
					
				}
				else {
					temp_complex.data.c[0] = cos(2 * i * sc->double_PI / radix);
					temp_complex.data.c[1] = sin(2 * i * sc->double_PI / radix);
					PfMov(sc, &sc->w, &temp_complex);	
					
				}
				PfMul(sc, &sc->temp, &regID[Q * i + 1], &sc->w, 0);
				
			}
			else {
				PfMov(sc, &sc->temp, &regID[Q * i + 1]);
				
			}
			PfSub(sc, &regID[Q * i + 1], &regID[Q * i], &sc->temp);
			
			PfAdd(sc, &regID[Q * i], &regID[Q * i], &sc->temp);
			
		}

		uint64_t permute2[14] = { 0,2,4,6,8,10,12,1,3,5,7,9,11,13 };
		PfPermute(sc, permute2, 14, 1, regID, &sc->temp);
		
		break;
	}
	case 15: {
		PfContainer tf[5];
		for (int64_t i = 0; i < 5; i++){
			tf[i].type = 32;
		}
		//PfAppendLine(sc, "	{\n");
		
		tf[0].data.d = -0.5;
		tf[1].data.d = 1.538841768587626701285145288018455;
		tf[2].data.d = -0.363271264002680442947733378740309;
		tf[3].data.d = -0.809016994374947424102293417182819;
		tf[4].data.d = -0.587785252292473129168705954639073;

		PfContainer tf2[2];
		for (int64_t i = 0; i < 2; i++){
			tf2[i].type = 32;
		}
		//PfAppendLine(sc, "	{\n");
		

		tf2[0].data.d = -0.5;
		tf2[1].data.d = -0.8660254037844386467637231707529;

		for (uint64_t i = radix - 1; i > 0; i--) {
			if (stageSize == 1) {
				temp_complex.data.c[0] = 1;
				temp_complex.data.c[1] = 0;
				PfMov(sc, &sc->w, &temp_complex);	
				
			}
			else {
				if (i == radix - 1) {
					if (sc->LUT) {
						if (sc->useCoalescedLUTUploadToSM) {
							appendSharedToRegisters(sc, &sc->w, &sc->stageInvocationID);
														
						}
						else {
							appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->LUTId);
														
						}
						if (stageAngle < 0) {
							PfConjugate(sc, &sc->w, &sc->w);
							
						}
					}
					else {
						temp_double.data.d = 2.0 * i / radix;
						PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
						PfSinCos(sc, &sc->w, &sc->tempFloat);
					}
				}
				else {
					if (sc->LUT) {
						if (sc->useCoalescedLUTUploadToSM) {
							temp_int.data.i = (radix - 1 - i) * stageSize;
							PfAdd(sc, &sc->sdataID, &sc->stageInvocationID, &temp_int);
							appendSharedToRegisters(sc, &sc->w, &sc->sdataID);
						}
						else {
							temp_int.data.i = (radix - 1 - i) * stageSize;
							PfAdd(sc, &sc->inoutID, &sc->LUTId, &temp_int);
							appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->inoutID);
							
						}
						if (stageAngle < 0) {
							PfConjugate(sc, &sc->w, &sc->w);
							
						}
					}
					else {
						temp_double.data.d = 2.0 * i / radix;
						PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
						PfSinCos(sc, &sc->w, &sc->tempFloat);
					}
				}
			}
			PfMul(sc, &regID[i], &regID[i], &sc->w, &sc->temp);
			
		}
		//important
		//PfMov(sc, &regID[1], &sc->locID[1]);
		//

		uint64_t P = 5;
		uint64_t Q = 3;
		for (uint64_t i = 0; i < Q; i++) {
			PfMov(sc, &sc->locID[0], &regID[i]);
			
			PfMov(sc, &sc->locID[1], &regID[i + Q]);
			
			PfMov(sc, &sc->locID[2], &regID[i + 2 * Q]);
			
			PfMov(sc, &sc->locID[3], &regID[i + 3 * Q]);
			
			PfMov(sc, &sc->locID[4], &regID[i + 4 * Q]);
			

			PfAdd(sc, &regID[i + Q], &sc->locID[1], &sc->locID[4]);
			
			PfAdd(sc, &regID[i + 2 * Q], &sc->locID[2], &sc->locID[3]);
			
			PfSub(sc, &regID[i + 3 * Q], &sc->locID[2], &sc->locID[3]);
			
			PfSub(sc, &regID[i + 4 * Q], &sc->locID[1], &sc->locID[4]);
			
			PfSub(sc, &sc->locID[3], &regID[i + Q], &regID[i + 2 * Q]);
			
			PfAdd(sc, &sc->locID[4], &regID[i + 3 * Q], &regID[i + 4 * Q]);
			

			PfAdd(sc, &sc->locID[0], &regID[i], &regID[i + Q]);
			
			PfAdd(sc, &sc->locID[0], &sc->locID[0], &regID[i + 2 * Q]);
			
			PfFMA(sc, &sc->locID[1], &regID[i + Q], &tf[0], &regID[i]);
			
			PfFMA(sc, &sc->locID[2], &regID[i + 2 * Q], &tf[0], &regID[i]);
			
			PfMul(sc, &regID[i + 3 * Q], &regID[i + 3 * Q], &tf[1], 0);
			
			PfMul(sc, &regID[i + 4 * Q], &regID[i + 4 * Q], &tf[2], 0);
			
			PfMul(sc, &sc->locID[3], &sc->locID[3], &tf[3], 0);
			
			PfMul(sc, &sc->locID[4], &sc->locID[4], &tf[4], 0);
			

			PfSub(sc, &sc->locID[1], &sc->locID[1], &sc->locID[3]);
			
			PfAdd(sc, &sc->locID[2], &sc->locID[2], &sc->locID[3]);
			
			PfAdd(sc, &sc->locID[3], &regID[i + 3 * Q], &sc->locID[4]);
			
			PfAdd(sc, &sc->locID[4], &sc->locID[4], &regID[i + 4 * Q]);
			
			PfMov(sc, &regID[i], &sc->locID[0]);
			

			if (stageAngle < 0)
			{
				PfShuffleComplex(sc, &regID[i + Q], &sc->locID[1], &sc->locID[4], 0);
				
				PfShuffleComplex(sc, &regID[i + 2 * Q], &sc->locID[2], &sc->locID[3], 0);
				
				PfShuffleComplexInv(sc, &regID[i + 3 * Q], &sc->locID[2], &sc->locID[3], 0);
				
				PfShuffleComplexInv(sc, &regID[i + 4 * Q], &sc->locID[1], &sc->locID[4], 0);
				
			}
			else {
				PfShuffleComplexInv(sc, &regID[i + Q], &sc->locID[1], &sc->locID[4], 0);
				
				PfShuffleComplexInv(sc, &regID[i + 2 * Q], &sc->locID[2], &sc->locID[3], 0);
				
				PfShuffleComplex(sc, &regID[i + 3 * Q], &sc->locID[2], &sc->locID[3], 0);
				
				PfShuffleComplex(sc, &regID[i + 4 * Q], &sc->locID[1], &sc->locID[4], 0);
				
			}

		}


		for (uint64_t i = 0; i < P; i++) {
			if (i > 0) {
				if (stageAngle < 0) {
					temp_complex.data.c[0] = cos(2 * i * sc->double_PI / radix);
					temp_complex.data.c[1] = -sin(2 * i * sc->double_PI / radix);
					PfMov(sc, &sc->w, &temp_complex);	
					
				}
				else {
					temp_complex.data.c[0] = cos(2 * i * sc->double_PI / radix);
					temp_complex.data.c[1] = sin(2 * i * sc->double_PI / radix);
					PfMov(sc, &sc->w, &temp_complex);	
					
				}
				PfMul(sc, &sc->locID[1], &regID[Q * i + 1], &sc->w, &sc->temp);
				
				if (stageAngle < 0) {
					temp_complex.data.c[0] = cos(4 * i * sc->double_PI / radix);
					temp_complex.data.c[1] = -sin(4 * i * sc->double_PI / radix);
					PfMov(sc, &sc->w, &temp_complex);	
					
				}
				else {
					temp_complex.data.c[0] = cos(4 * i * sc->double_PI / radix);
					temp_complex.data.c[1] = sin(4 * i * sc->double_PI / radix);
					PfMov(sc, &sc->w, &temp_complex);	
					
				}
				PfMul(sc, &sc->locID[2], &regID[Q * i + 2], &sc->w, &sc->temp);
				
			}
			else {
				PfMov(sc, &sc->locID[1], &regID[1]);
				
				PfMov(sc, &sc->locID[2], &regID[2]);
				
			}

			PfAdd(sc, &regID[Q * i + 1], &sc->locID[1], &sc->locID[2]);
			
			PfSub(sc, &regID[Q * i + 2], &sc->locID[1], &sc->locID[2]);
			

			PfAdd(sc, &sc->locID[0], &regID[Q * i], &regID[Q * i + 1]);
			
			PfFMA(sc, &sc->locID[1], &regID[Q * i + 1], &tf2[0], &regID[Q * i]);
			
			PfMul(sc, &sc->locID[2], &regID[Q * i + 2], &tf2[1], 0);
			
			PfMov(sc, &regID[Q * i], &sc->locID[0]);
			
			if (stageAngle < 0)
			{
				PfShuffleComplex(sc, &regID[Q * i + 1], &sc->locID[1], &sc->locID[2], 0);
				
				PfShuffleComplexInv(sc, &regID[Q * i + 2], &sc->locID[1], &sc->locID[2], 0);
				
			}
			else {
				PfShuffleComplexInv(sc, &regID[Q * i + 1], &sc->locID[1], &sc->locID[2], 0);
				
				PfShuffleComplex(sc, &regID[Q * i + 2], &sc->locID[1], &sc->locID[2], 0);
				
			}
		}

		uint64_t permute2[15] = { 0, 3, 6, 9, 12, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14 };
		PfPermute(sc, permute2, 15, 1, regID, &sc->temp);
		
		break;
	}
	case 16: {
		
		if (stageSize == 1) {
			temp_complex.data.c[0] = 1;
			temp_complex.data.c[1] = 0;
			PfMov(sc, &sc->w, &temp_complex);	
			
		}
		else {
			if (sc->LUT) {
				if (sc->useCoalescedLUTUploadToSM) {
					appendSharedToRegisters(sc, &sc->w, &sc->stageInvocationID);
										
				}
				else {
					appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->LUTId);
										
				}
				if (stageAngle < 0) {
					PfConjugate(sc, &sc->w, &sc->w);
					
				}
			}
			else {
				PfSinCos(sc, &sc->w, &sc->angle);
			}
		}
		for (uint64_t i = 0; i < 8; i++) {
			PfMul(sc, &sc->temp, &regID[i + 8], &sc->w, 0);
			
			PfSub(sc, &regID[i + 8], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageSize == 1) {
			temp_complex.data.c[0] = 1;
			temp_complex.data.c[1] = 0;
			PfMov(sc, &sc->w, &temp_complex);	
			
		}
		else {
			if (sc->LUT) {
				if (sc->useCoalescedLUTUploadToSM) {
					temp_int.data.i = stageSize;
					PfAdd(sc, &sc->sdataID, &sc->stageInvocationID, &temp_int);
					appendSharedToRegisters(sc, &sc->w, &sc->sdataID);
				}
				else {
					temp_int.data.i = stageSize;
					PfAdd(sc, &sc->inoutID, &sc->LUTId, &temp_int);
					appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->inoutID);
					
				}
				if (stageAngle < 0) {
					PfConjugate(sc, &sc->w, &sc->w);
					
				}
			}
			else {
				temp_double.data.d = 0.5;
				PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
				PfSinCos(sc, &sc->w, &sc->tempFloat);
			}
		}
		for (uint64_t i = 0; i < 4; i++) {
			PfMul(sc, &sc->temp, &regID[i + 4], &sc->w, 0);
			
			PfSub(sc, &regID[i + 4], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageAngle < 0) {
			PfMov_x_y(sc, &sc->iw, &sc->w);
			PfMov_y_Neg_x(sc, &sc->iw, &sc->w);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(w.y, -w.x);\n\n", vecType);
		}
		else {
			PfMov_x_Neg_y(sc, &sc->iw, &sc->w);
			PfMov_y_x(sc, &sc->iw, &sc->w);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	iw = %s(-w.y, w.x);\n\n", vecType);
		}

		for (uint64_t i = 8; i < 12; i++) {
			PfMul(sc, &sc->temp, &regID[i + 4], &sc->iw, 0);
			
			PfSub(sc, &regID[i + 4], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageSize == 1) {
			temp_complex.data.c[0] = 1;
			temp_complex.data.c[1] = 0;
			PfMov(sc, &sc->w, &temp_complex);	
			
		}
		else {
			if (sc->LUT) {
				if (sc->useCoalescedLUTUploadToSM) {
					temp_int.data.i = 2 * stageSize;
					PfAdd(sc, &sc->sdataID, &sc->stageInvocationID, &temp_int);
					appendSharedToRegisters(sc, &sc->w, &sc->sdataID);
				}
				else {
					temp_int.data.i = 2 * stageSize;
					PfAdd(sc, &sc->inoutID, &sc->LUTId, &temp_int);
					appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->inoutID);
					
				}
				if (stageAngle < 0) {
					PfConjugate(sc, &sc->w, &sc->w);
					
				}
			}
			else {
				temp_double.data.d = 0.25;
				PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
				PfSinCos(sc, &sc->w, &sc->tempFloat);
			}
		}
		for (uint64_t i = 0; i < 2; i++) {
			PfMul(sc, &sc->temp, &regID[i + 2], &sc->w, 0);
			
			PfSub(sc, &regID[i + 2], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageAngle < 0) {
			PfMov_x_y(sc, &sc->iw, &sc->w);
			PfMov_y_Neg_x(sc, &sc->iw, &sc->w);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(w.y, -w.x);\n\n", vecType);
		}
		else {
			PfMov_x_Neg_y(sc, &sc->iw, &sc->w);
			PfMov_y_x(sc, &sc->iw, &sc->w);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	iw = %s(-w.y, w.x);\n\n", vecType);
		}
		for (uint64_t i = 4; i < 6; i++) {
			PfMul(sc, &sc->temp, &regID[i + 2], &sc->iw, 0);
			
			PfSub(sc, &regID[i + 2], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageAngle < 0) {
			temp_complex.data.c[0] = 0.70710678118654752440084436210485;
			temp_complex.data.c[1] = -0.70710678118654752440084436210485;
			PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);

		}
		else {
			temp_complex.data.c[0] = 0.70710678118654752440084436210485;
			temp_complex.data.c[1] = 0.70710678118654752440084436210485;
			PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
		}
		for (uint64_t i = 8; i < 10; i++) {
			PfMul(sc, &sc->temp, &regID[i + 2], &sc->iw, 0);
			
			PfSub(sc, &regID[i + 2], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageAngle < 0) {
			PfMov_x_y(sc, &sc->w, &sc->iw);
			PfMov_y_Neg_x(sc, &sc->w, &sc->iw);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(iw.y, -iw.x);\n\n", vecType);
		}
		else {
			PfMov_x_Neg_y(sc, &sc->w, &sc->iw);
			PfMov_y_x(sc, &sc->w, &sc->iw);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(-iw.y, iw.x);\n\n", vecType);
		}
		for (uint64_t i = 12; i < 14; i++) {
			PfMul(sc, &sc->temp, &regID[i + 2], &sc->w, 0);
			
			PfSub(sc, &regID[i + 2], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}

		if (stageSize == 1) {
			temp_complex.data.c[0] = 1;
			temp_complex.data.c[1] = 0;
			PfMov(sc, &sc->w, &temp_complex);	
			
		}
		else {
			if (sc->LUT) {
				if (sc->useCoalescedLUTUploadToSM) {
					temp_int.data.i = 3 * stageSize;
					PfAdd(sc, &sc->sdataID, &sc->stageInvocationID, &temp_int);
					appendSharedToRegisters(sc, &sc->w, &sc->sdataID);
				}
				else {
					temp_int.data.i = 3 * stageSize;
					PfAdd(sc, &sc->inoutID, &sc->LUTId, &temp_int);
					appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->inoutID);
					
				}
				if (stageAngle < 0) {
					PfConjugate(sc, &sc->w, &sc->w);
					
				}
			}
			else {
				temp_double.data.d = 0.125;
				PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
				PfSinCos(sc, &sc->w, &sc->tempFloat);
			}
		}

		for (uint64_t i = 0; i < 1; i++) {
			PfMul(sc, &sc->temp, &regID[i + 1], &sc->w, 0);
			
			PfSub(sc, &regID[i + 1], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageAngle < 0) {
			PfMov_x_y(sc, &sc->iw, &sc->w);
			PfMov_y_Neg_x(sc, &sc->iw, &sc->w);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(w.y, -w.x);\n\n", vecType);
		}
		else {
			PfMov_x_Neg_y(sc, &sc->iw, &sc->w);
			PfMov_y_x(sc, &sc->iw, &sc->w);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	iw = %s(-w.y, w.x);\n\n", vecType);
		}
		for (uint64_t i = 2; i < 3; i++) {
			PfMul(sc, &sc->temp, &regID[i + 1], &sc->iw, 0);
			
			PfSub(sc, &regID[i + 1], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}


		if (stageAngle < 0) {
			temp_complex.data.c[0] = 0.70710678118654752440084436210485;
			temp_complex.data.c[1] = -0.70710678118654752440084436210485;
			PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);

		}
		else {
			temp_complex.data.c[0] = 0.70710678118654752440084436210485;
			temp_complex.data.c[1] = 0.70710678118654752440084436210485;
			PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
		}
		for (uint64_t i = 4; i < 5; i++) {
			PfMul(sc, &sc->temp, &regID[i + 1], &sc->iw, 0);
			
			PfSub(sc, &regID[i + 1], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageAngle < 0) {
			PfMov_x_y(sc, &sc->temp, &sc->iw);
			PfMov_y_Neg_x(sc, &sc->temp, &sc->iw);
			
			PfMov(sc, &sc->iw, &sc->temp);
			
		}
		else {
			PfMov_x_Neg_y(sc, &sc->temp, &sc->iw);
			PfMov_y_x(sc, &sc->temp, &sc->iw);
			
			PfMov(sc, &sc->iw, &sc->temp);
			
		}
		for (uint64_t i = 6; i < 7; i++) {
			PfMul(sc, &sc->temp, &regID[i + 1], &sc->iw, 0);
			
			PfSub(sc, &regID[i + 1], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}


		for (uint64_t j = 0; j < 2; j++) {
			if (stageAngle < 0) {
				temp_complex.data.c[0] = cos((2 * j + 1) * sc->double_PI / 8);
				temp_complex.data.c[1] = -sin((2 * j + 1) * sc->double_PI / 8);
				PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
			}
			else {
				temp_complex.data.c[0] = cos((2 * j + 1) * sc->double_PI / 8);
				temp_complex.data.c[1] = sin((2 * j + 1) * sc->double_PI / 8);
				PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
			}
			for (uint64_t i = 8 + 4 * j; i < 9 + 4 * j; i++) {
				PfMul(sc, &sc->temp, &regID[i + 1], &sc->iw, 0);
				
				PfSub(sc, &regID[i + 1], &regID[i], &sc->temp);
				
				PfAdd(sc, &regID[i], &regID[i], &sc->temp);
				
			}
			if (stageAngle < 0) {
				PfMov_x_y(sc, &sc->temp, &sc->iw);
				PfMov_y_Neg_x(sc, &sc->temp, &sc->iw);
				
				PfMov(sc, &sc->iw, &sc->temp);
				
			}
			else {
				PfMov_x_Neg_y(sc, &sc->temp, &sc->iw);
				PfMov_y_x(sc, &sc->temp, &sc->iw);
				
				PfMov(sc, &sc->iw, &sc->temp);
				
			}
			for (uint64_t i = 10 + 4 * j; i < 11 + 4 * j; i++) {
				PfMul(sc, &sc->temp, &regID[i + 1], &sc->iw, 0);
				
				PfSub(sc, &regID[i + 1], &regID[i], &sc->temp);
				
				PfAdd(sc, &regID[i], &regID[i], &sc->temp);
				
			}
		}

		uint64_t permute2[16] = { 0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15 };
		PfPermute(sc, permute2, 16, 1, regID, &sc->temp);
		

		/*PfMov(sc, &sc->temp, &regID[1]);
		
		PfMov(sc, &regID[1], &regID[8]);
		
		PfMov(sc, &regID[8], &sc->temp);
		

		PfMov(sc, &sc->temp, &regID[2]);
		
		PfMov(sc, &regID[2], &regID[4]);
		
		PfMov(sc, &regID[4], &sc->temp);
		

		PfMov(sc, &sc->temp, &regID[3]);
		
		PfMov(sc, &regID[3], &regID[12]);
		
		PfMov(sc, &regID[12], &sc->temp);
		

		PfMov(sc, &sc->temp, &regID[5]);
		
		PfMov(sc, &regID[5], &regID[10]);
		
		PfMov(sc, &regID[10], &sc->temp);
		

		PfMov(sc, &sc->temp, &regID[7]);
		
		PfMov(sc, &regID[7], &regID[14]);
		
		PfMov(sc, &regID[14], &sc->temp);
		

		PfMov(sc, &sc->temp, &regID[11]);
		
		PfMov(sc, &regID[11], &regID[13]);
		
		PfMov(sc, &regID[13], &sc->temp);
		*/
		break;
	}
	case 32: {
		
		if (stageSize == 1) {
			temp_complex.data.c[0] = 1;
			temp_complex.data.c[1] = 0;
			PfMov(sc, &sc->w, &temp_complex);	
			
		}
		else {
			if (sc->LUT) {
				if (sc->useCoalescedLUTUploadToSM) {
					appendSharedToRegisters(sc, &sc->w, &sc->stageInvocationID);
										
				}
				else {
					appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->LUTId);
										
				}
				if (stageAngle < 0) {
					PfConjugate(sc, &sc->w, &sc->w);
					
				}
			}
			else {
				PfSinCos(sc, &sc->w, &sc->angle);
			}
		}
		for (uint64_t i = 0; i < 16; i++) {
			PfMul(sc, &sc->temp, &regID[i + 16], &sc->w, 0);
			
			PfSub(sc, &regID[i + 16], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageSize == 1) {
			temp_complex.data.c[0] = 1;
			temp_complex.data.c[1] = 0;
			PfMov(sc, &sc->w, &temp_complex);	
			
		}
		else {
			if (sc->LUT) {
				if (sc->useCoalescedLUTUploadToSM) {
					temp_int.data.i = stageSize;
					PfAdd(sc, &sc->sdataID, &sc->stageInvocationID, &temp_int);
					appendSharedToRegisters(sc, &sc->w, &sc->sdataID);
				}
				else {
					temp_int.data.i = stageSize;
					PfAdd(sc, &sc->inoutID, &sc->LUTId, &temp_int);
					appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->inoutID);
				}
				if (stageAngle < 0) {
					PfConjugate(sc, &sc->w, &sc->w);
					
				}
			}
			else {
				temp_double.data.d = 0.5;
				PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
				PfSinCos(sc, &sc->w, &sc->tempFloat);
			}
		}
		for (uint64_t i = 0; i < 8; i++) {
			PfMul(sc, &sc->temp, &regID[i + 8], &sc->w, 0);
			
			PfSub(sc, &regID[i + 8], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageAngle < 0) {
			PfMov_x_y(sc, &sc->iw, &sc->w);
			PfMov_y_Neg_x(sc, &sc->iw, &sc->w);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(w.y, -w.x);\n\n", vecType);
		}
		else {
			PfMov_x_Neg_y(sc, &sc->iw, &sc->w);
			PfMov_y_x(sc, &sc->iw, &sc->w);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	iw = %s(-w.y, w.x);\n\n", vecType);
		}

		for (uint64_t i = 16; i < 24; i++) {
			PfMul(sc, &sc->temp, &regID[i + 8], &sc->iw, 0);
			
			PfSub(sc, &regID[i + 8], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageSize == 1) {
			temp_complex.data.c[0] = 1;
			temp_complex.data.c[1] = 0;
			PfMov(sc, &sc->w, &temp_complex);	
			
		}
		else {
			if (sc->LUT) {
				if (sc->useCoalescedLUTUploadToSM) {
					temp_int.data.i = 2 * stageSize;
					PfAdd(sc, &sc->sdataID, &sc->stageInvocationID, &temp_int);
					appendSharedToRegisters(sc, &sc->w, &sc->sdataID);
				}
				else {
					temp_int.data.i = 2 * stageSize;
					PfAdd(sc, &sc->inoutID, &sc->LUTId, &temp_int);
					appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->inoutID);
				}
				if (stageAngle < 0) {
					PfConjugate(sc, &sc->w, &sc->w);
					
				}
			}
			else {
				temp_double.data.d = 0.25;
				PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
				PfSinCos(sc, &sc->w, &sc->tempFloat);
			}
		}
		for (uint64_t i = 0; i < 4; i++) {
			PfMul(sc, &sc->temp, &regID[i + 4], &sc->w, 0);
			
			PfSub(sc, &regID[i + 4], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageAngle < 0) {
			PfMov_x_y(sc, &sc->iw, &sc->w);
			PfMov_y_Neg_x(sc, &sc->iw, &sc->w);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(w.y, -w.x);\n\n", vecType);
		}
		else {
			PfMov_x_Neg_y(sc, &sc->iw, &sc->w);
			PfMov_y_x(sc, &sc->iw, &sc->w);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	iw = %s(-w.y, w.x);\n\n", vecType);
		}
		for (uint64_t i = 8; i < 12; i++) {
			PfMul(sc, &sc->temp, &regID[i + 4], &sc->iw, 0);
			
			PfSub(sc, &regID[i + 4], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageAngle < 0) {
			temp_complex.data.c[0] = 0.70710678118654752440084436210485;
			temp_complex.data.c[1] = -0.70710678118654752440084436210485;
			PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);

		}
		else {
			temp_complex.data.c[0] = 0.70710678118654752440084436210485;
			temp_complex.data.c[1] = 0.70710678118654752440084436210485;
			PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
		}
		for (uint64_t i = 16; i < 20; i++) {
			PfMul(sc, &sc->temp, &regID[i + 4], &sc->iw, 0);
			
			PfSub(sc, &regID[i + 4], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageAngle < 0) {
			PfMov_x_y(sc, &sc->w, &sc->iw);
			PfMov_y_Neg_x(sc, &sc->w, &sc->iw);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(iw.y, -iw.x);\n\n", vecType);
		}
		else {
			PfMov_x_Neg_y(sc, &sc->w, &sc->iw);
			PfMov_y_x(sc, &sc->w, &sc->iw);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(-iw.y, iw.x);\n\n", vecType);
		}
		for (uint64_t i = 24; i < 28; i++) {
			PfMul(sc, &sc->temp, &regID[i + 4], &sc->w, 0);
			
			PfSub(sc, &regID[i + 4], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}

		if (stageSize == 1) {
			temp_complex.data.c[0] = 1;
			temp_complex.data.c[1] = 0;
			PfMov(sc, &sc->w, &temp_complex);	
			
		}
		else {
			if (sc->LUT) {
				if (sc->useCoalescedLUTUploadToSM) {
					temp_int.data.i = 3 * stageSize;
					PfAdd(sc, &sc->sdataID, &sc->stageInvocationID, &temp_int);
					appendSharedToRegisters(sc, &sc->w, &sc->sdataID);
				}
				else {
					temp_int.data.i = 3 * stageSize;
					PfAdd(sc, &sc->inoutID, &sc->LUTId, &temp_int);
					appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->inoutID);
				}
				if (stageAngle < 0) {
					PfConjugate(sc, &sc->w, &sc->w);
					
				}
			}
			else {
				temp_double.data.d = 0.125;
				PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
				PfSinCos(sc, &sc->w, &sc->tempFloat);
			}
		}

		for (uint64_t i = 0; i < 2; i++) {
			PfMul(sc, &sc->temp, &regID[i + 2], &sc->w, 0);
			
			PfSub(sc, &regID[i + 2], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageAngle < 0) {
			PfMov_x_y(sc, &sc->iw, &sc->w);
			PfMov_y_Neg_x(sc, &sc->iw, &sc->w);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(w.y, -w.x);\n\n", vecType);
		}
		else {
			PfMov_x_Neg_y(sc, &sc->iw, &sc->w);
			PfMov_y_x(sc, &sc->iw, &sc->w);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	iw = %s(-w.y, w.x);\n\n", vecType);
		}
		for (uint64_t i = 4; i < 6; i++) {
			PfMul(sc, &sc->temp, &regID[i + 2], &sc->iw, 0);
			
			PfSub(sc, &regID[i + 2], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}


		if (stageAngle < 0) {
			temp_complex.data.c[0] = 0.70710678118654752440084436210485;
			temp_complex.data.c[1] = -0.70710678118654752440084436210485;
			PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);

		}
		else {
			temp_complex.data.c[0] = 0.70710678118654752440084436210485;
			temp_complex.data.c[1] = 0.70710678118654752440084436210485;
			PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
		}
		for (uint64_t i = 8; i < 10; i++) {
			PfMul(sc, &sc->temp, &regID[i + 2], &sc->iw, 0);
			
			PfSub(sc, &regID[i + 2], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageAngle < 0) {
			PfMov_x_y(sc, &sc->temp, &sc->iw);
			PfMov_y_Neg_x(sc, &sc->temp, &sc->iw);
			
			PfMov(sc, &sc->iw, &sc->temp);
			
		}
		else {
			PfMov_x_Neg_y(sc, &sc->temp, &sc->iw);
			PfMov_y_x(sc, &sc->temp, &sc->iw);
			
			PfMov(sc, &sc->iw, &sc->temp);
			
		}
		for (uint64_t i = 12; i < 14; i++) {
			PfMul(sc, &sc->temp, &regID[i + 2], &sc->iw, 0);
			
			PfSub(sc, &regID[i + 2], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}


		for (uint64_t j = 0; j < 2; j++) {
			if (stageAngle < 0) {
				temp_complex.data.c[0] = cos((2 * j + 1) * sc->double_PI / 8);
				temp_complex.data.c[1] = -sin((2 * j + 1) * sc->double_PI / 8);
				PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
			}
			else {
				temp_complex.data.c[0] = cos((2 * j + 1) * sc->double_PI / 8);
				temp_complex.data.c[1] = sin((2 * j + 1) * sc->double_PI / 8);
				PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
			}
			for (uint64_t i = 16 + 8 * j; i < 18 + 8 * j; i++) {
				PfMul(sc, &sc->temp, &regID[i + 2], &sc->iw, 0);
				
				PfSub(sc, &regID[i + 2], &regID[i], &sc->temp);
				
				PfAdd(sc, &regID[i], &regID[i], &sc->temp);
				
			}
			if (stageAngle < 0) {
				PfMov_x_y(sc, &sc->temp, &sc->iw);
				PfMov_y_Neg_x(sc, &sc->temp, &sc->iw);
			
				PfMov(sc, &sc->iw, &sc->temp);
			}
			else {
				PfMov_x_Neg_y(sc, &sc->temp, &sc->iw);
				PfMov_y_x(sc, &sc->temp, &sc->iw);
			
				PfMov(sc, &sc->iw, &sc->temp);
				
			}
			for (uint64_t i = 20 + 8 * j; i < 22 + 8 * j; i++) {
				PfMul(sc, &sc->temp, &regID[i + 2], &sc->iw, 0);
				
				PfSub(sc, &regID[i + 2], &regID[i], &sc->temp);
				
				PfAdd(sc, &regID[i], &regID[i], &sc->temp);
				
			}
		}

		if (stageSize == 1) {
			temp_complex.data.c[0] = 1;
			temp_complex.data.c[1] = 0;
			PfMov(sc, &sc->w, &temp_complex);	
			
		}
		else {
			if (sc->LUT) {
				if (sc->useCoalescedLUTUploadToSM) {
					temp_int.data.i = 4 * stageSize;
					PfAdd(sc, &sc->sdataID, &sc->stageInvocationID, &temp_int);
					appendSharedToRegisters(sc, &sc->w, &sc->sdataID);
				}
				else {
					temp_int.data.i = 4 * stageSize;
					PfAdd(sc, &sc->inoutID, &sc->LUTId, &temp_int);
					appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->inoutID);
				}
				if (stageAngle < 0) {
					PfConjugate(sc, &sc->w, &sc->w);
				}
			}
			else {
				temp_double.data.d = 0.0625;
				PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
				PfSinCos(sc, &sc->w, &sc->tempFloat);
			}
		}

		for (uint64_t i = 0; i < 1; i++) {
			PfMul(sc, &sc->temp, &regID[i + 1], &sc->w, 0);
			
			PfSub(sc, &regID[i + 1], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageAngle < 0) {
			PfMov_x_y(sc, &sc->iw, &sc->w);
			PfMov_y_Neg_x(sc, &sc->iw, &sc->w);
			
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(w.y, -w.x);\n\n", vecType);
		}
		else {
			PfMov_x_Neg_y(sc, &sc->iw, &sc->w);
			PfMov_y_x(sc, &sc->iw, &sc->w);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	iw = %s(-w.y, w.x);\n\n", vecType);
		}
		for (uint64_t i = 2; i < 3; i++) {
			PfMul(sc, &sc->temp, &regID[i + 1], &sc->iw, 0);
			
			PfSub(sc, &regID[i + 1], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}


		if (stageAngle < 0) {
			temp_complex.data.c[0] = 0.70710678118654752440084436210485;
			temp_complex.data.c[1] = -0.70710678118654752440084436210485;
			PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);

		}
		else {
			temp_complex.data.c[0] = 0.70710678118654752440084436210485;
			temp_complex.data.c[1] = 0.70710678118654752440084436210485;
			PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
		}
		for (uint64_t i = 4; i < 5; i++) {
			PfMul(sc, &sc->temp, &regID[i + 1], &sc->iw, 0);
			
			PfSub(sc, &regID[i + 1], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageAngle < 0) {
			PfMov_x_y(sc, &sc->temp, &sc->iw);
			PfMov_y_Neg_x(sc, &sc->temp, &sc->iw);
			
			PfMov(sc, &sc->iw, &sc->temp);
			
		}
		else {
			PfMov_x_Neg_y(sc, &sc->temp, &sc->iw);
			PfMov_y_x(sc, &sc->temp, &sc->iw);
			
			
			PfMov(sc, &sc->iw, &sc->temp);
			
		}
		for (uint64_t i = 6; i < 7; i++) {
			PfMul(sc, &sc->temp, &regID[i + 1], &sc->iw, 0);
			
			PfSub(sc, &regID[i + 1], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}


		for (uint64_t j = 0; j < 2; j++) {
			if (stageAngle < 0) {
				temp_complex.data.c[0] = cos((2 * j + 1) * sc->double_PI / 8);
				temp_complex.data.c[1] = -sin((2 * j + 1) * sc->double_PI / 8);
				PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
			}
			else {
				temp_complex.data.c[0] = cos((2 * j + 1) * sc->double_PI / 8);
				temp_complex.data.c[1] = sin((2 * j + 1) * sc->double_PI / 8);
				PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
			}
			for (uint64_t i = 8 + 4 * j; i < 9 + 4 * j; i++) {
				PfMul(sc, &sc->temp, &regID[i + 1], &sc->iw, 0);
				
				PfSub(sc, &regID[i + 1], &regID[i], &sc->temp);
				
				PfAdd(sc, &regID[i], &regID[i], &sc->temp);
				
			}
			if (stageAngle < 0) {
				PfMov_x_y(sc, &sc->temp, &sc->iw);
				PfMov_y_Neg_x(sc, &sc->temp, &sc->iw);
				
				PfMov(sc, &sc->iw, &sc->temp);
				
			}
			else {
				PfMov_x_Neg_y(sc, &sc->temp, &sc->iw);
				PfMov_y_x(sc, &sc->temp, &sc->iw);
				
				PfMov(sc, &sc->iw, &sc->temp);
				
			}
			for (uint64_t i = 10 + 4 * j; i < 11 + 4 * j; i++) {
				PfMul(sc, &sc->temp, &regID[i + 1], &sc->iw, 0);
				
				PfSub(sc, &regID[i + 1], &regID[i], &sc->temp);
				
				PfAdd(sc, &regID[i], &regID[i], &sc->temp);
				
			}
		}

		for (uint64_t j = 0; j < 4; j++) {
			if ((j == 1) || (j == 2)) {
				if (stageAngle < 0) {
					temp_complex.data.c[0] = cos((7 - 2 * j) * sc->double_PI / 16);
					temp_complex.data.c[1] = -sin((7 - 2 * j) * sc->double_PI / 16);
					PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
				}
				else {
					temp_complex.data.c[0] = cos((7 - 2 * j) * sc->double_PI / 16);
					temp_complex.data.c[1] = sin((7 - 2 * j) * sc->double_PI / 16);
					PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
				}
			}
			else {
				if (stageAngle < 0) {
					temp_complex.data.c[0] = cos((2 * j + 1) * sc->double_PI / 16);
					temp_complex.data.c[1] = -sin((2 * j + 1) * sc->double_PI / 16);
					PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
				}
				else {
					temp_complex.data.c[0] = cos((2 * j + 1) * sc->double_PI / 16);
					temp_complex.data.c[1] = sin((2 * j + 1) * sc->double_PI / 16);
					PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
				}
			}
			for (uint64_t i = 16 + 4 * j; i < 17 + 4 * j; i++) {
				PfMul(sc, &sc->temp, &regID[i + 1], &sc->iw, 0);
				
				PfSub(sc, &regID[i + 1], &regID[i], &sc->temp);
				
				PfAdd(sc, &regID[i], &regID[i], &sc->temp);
				
			}
			if (stageAngle < 0) {
				PfMov_x_y(sc, &sc->temp, &sc->iw);
				PfMov_y_Neg_x(sc, &sc->temp, &sc->iw);
				
				PfMov(sc, &sc->iw, &sc->temp);
				
			}
			else {
				PfMov_x_Neg_y(sc, &sc->temp, &sc->iw);
				PfMov_y_x(sc, &sc->temp, &sc->iw);
				
				PfMov(sc, &sc->iw, &sc->temp);
				
			}
			for (uint64_t i = 18 + 4 * j; i < 19 + 4 * j; i++) {
				PfMul(sc, &sc->temp, &regID[i + 1], &sc->iw, 0);
				
				PfSub(sc, &regID[i + 1], &regID[i], &sc->temp);
				
				PfAdd(sc, &regID[i], &regID[i], &sc->temp);
				
			}
		}

		uint64_t permute2[32] = { 0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30,1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31 };
		PfPermute(sc, permute2, 32, 1, regID, &sc->temp);
		

		/*PfMov(sc, &sc->temp, &regID[1]);
		
		PfMov(sc, &regID[1], &regID[16]);
		
		PfMov(sc, &regID[16], &sc->temp);
		

		PfMov(sc, &sc->temp, &regID[2]);
		
		PfMov(sc, &regID[2], &regID[8]);
		
		PfMov(sc, &regID[8], &sc->temp);
		

		PfMov(sc, &sc->temp, &regID[3]);
		
		PfMov(sc, &regID[3], &regID[24]);
		
		PfMov(sc, &regID[24], &sc->temp);
		

		PfMov(sc, &sc->temp, &regID[5]);
		
		PfMov(sc, &regID[5], &regID[20]);
		
		PfMov(sc, &regID[20], &sc->temp);
		

		PfMov(sc, &sc->temp, &regID[6]);
		
		PfMov(sc, &regID[6], &regID[12]);
		
		PfMov(sc, &regID[12], &sc->temp);
		

		PfMov(sc, &sc->temp, &regID[7]);
		
		PfMov(sc, &regID[7], &regID[28]);
		
		PfMov(sc, &regID[28], &sc->temp);
		

		PfMov(sc, &sc->temp, &regID[9]);
		
		PfMov(sc, &regID[9], &regID[18]);
		
		PfMov(sc, &regID[18], &sc->temp);
		

		PfMov(sc, &sc->temp, &regID[11]);
		
		PfMov(sc, &regID[11], &regID[26]);
		
		PfMov(sc, &regID[26], &sc->temp);
		

		PfMov(sc, &sc->temp, &regID[13]);
		
		PfMov(sc, &regID[13], &regID[22]);
		
		PfMov(sc, &regID[22], &sc->temp);
		

		PfMov(sc, &sc->temp, &regID[15]);
		
		PfMov(sc, &regID[15], &regID[30]);
		
		PfMov(sc, &regID[30], &sc->temp);
		

		PfMov(sc, &sc->temp, &regID[19]);
		
		PfMov(sc, &regID[19], &regID[25]);
		
		PfMov(sc, &regID[25], &sc->temp);
		

		PfMov(sc, &sc->temp, &regID[23]);
		
		PfMov(sc, &regID[23], &regID[29]);
		
		PfMov(sc, &regID[29], &sc->temp);
		*/

		break;
	}
	}
	return;
}

#endif
