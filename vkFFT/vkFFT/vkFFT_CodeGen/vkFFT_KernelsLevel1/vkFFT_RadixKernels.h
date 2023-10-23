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

static inline void inlineRadixKernelVkFFT(VkFFTSpecializationConstantsLayout* sc, pfINT radix, pfINT stageSize, pfINT stageSizeSum, pfLD stageAngle, PfContainer* regID) {
	if (sc->res != VKFFT_SUCCESS) return;

	PfContainer temp_complex = VKFFT_ZERO_INIT;
	temp_complex.type = 23;
	PfAllocateContainerFlexible(sc, &temp_complex, 50);
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 22;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	//sprintf(temp, "loc_0");

	switch (radix) {
	case 2: {
		if (stageSize == 1) {
			temp_complex.data.c[0].data.d = pfFPinit("1.0");
			temp_complex.data.c[1].data.d = pfFPinit("0.0");
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

		PfContainer tf[2] = VKFFT_ZERO_INIT;
		for (pfINT i = 0; i < 2; i++){
			tf[i].type = 22;
		}
		
		tf[0].data.d = pfFPinit("-0.5");
		tf[1].data.d = pfFPinit("-0.8660254037844386467637231707529361834714");

		if (stageSize == 1) {
			temp_complex.data.c[0].data.d = pfFPinit("1.0");
			temp_complex.data.c[1].data.d = pfFPinit("0.0");
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
				temp_double.data.d = pfFPinit("4.0") / 3.0;
				PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
				PfSinCos(sc, &sc->w, &sc->tempFloat);
			}
		}
		PfMul(sc, &sc->locID[2], &regID[2], &sc->w, 0);
		if (stageSize == 1) {
			temp_complex.data.c[0].data.d = pfFPinit("1.0");
			temp_complex.data.c[1].data.d = pfFPinit("0.0");
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
					temp_double.data.d = pfFPinit("4.0") / 3.0;
					temp_int.data.i = stageSize;
					PfAdd(sc, &sc->inoutID, &sc->LUTId, &temp_int);
					appendGlobalToRegisters(sc, &sc->w, &sc->LUTStruct, &sc->inoutID);
					
				}
				if (stageAngle < 0) {
					PfConjugate(sc, &sc->w, &sc->w);
					
				}
			}
			else {
				temp_double.data.d = pfFPinit("2.0") / 3.0;
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
			PfShuffleComplex(sc, &regID[1], &sc->locID[1], &sc->locID[2], &sc->locID[0]);
			
			PfShuffleComplexInv(sc, &regID[2], &sc->locID[1], &sc->locID[2], &sc->locID[0]);
			
		}
		else {
			PfShuffleComplexInv(sc, &regID[1], &sc->locID[1], &sc->locID[2], &sc->locID[0]);
			
			PfShuffleComplex(sc, &regID[2], &sc->locID[1], &sc->locID[2], &sc->locID[0]);
			
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
			temp_complex.data.c[0].data.d = pfFPinit("1.0");
			temp_complex.data.c[1].data.d = pfFPinit("0.0");
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
			temp_complex.data.c[0].data.d = pfFPinit("1.0");
			temp_complex.data.c[1].data.d = pfFPinit("0.0");
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
				temp_double.data.d = pfFPinit("0.5");
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
			PfMov(sc, &sc->temp.data.c[0], &sc->w.data.c[0]);
			
			PfMov(sc, &sc->w.data.c[0], &sc->w.data.c[1]);
			PfMovNeg(sc, &sc->w.data.c[1], &sc->temp.data.c[0]);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(w.y, -w.x);\n\n", vecType);
		}
		else {
			PfMov(sc, &sc->temp.data.c[0], &sc->w.data.c[0]);
			
			PfMovNeg(sc, &sc->w.data.c[0], &sc->w.data.c[1]);
			PfMov(sc, &sc->w.data.c[1], &sc->temp.data.c[0]);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(-w.y, w.x);\n\n", vecType);
		}
		PfMul(sc, &sc->temp, &regID[3], &sc->w, 0);
		
		PfSub(sc, &regID[3], &regID[2], &sc->temp);
		
		PfAdd(sc, &regID[2], &regID[2], &sc->temp);
		
		//PfMov(sc, &sc->temp, &regID[1]);
		//

		pfUINT permute2[4] = { 0,2,1,3 };
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
		PfContainer tf[5] = VKFFT_ZERO_INIT;
		for (pfINT i = 0; i < 5; i++){
			tf[i].type = 22;
		}
		tf[0].data.d = pfFPinit("-0.5");
		tf[1].data.d = pfFPinit("1.538841768587626701285145288018455");
		tf[2].data.d = pfFPinit("-0.363271264002680442947733378740309");
		tf[3].data.d = pfFPinit("-0.809016994374947424102293417182819");
		tf[4].data.d = pfFPinit("-0.587785252292473129168705954639073");

		/*for (pfUINT i = 0; i < 5; i++) {
			&sc->locID[i], (char*)malloc(sizeof(char) * 50);
			sprintf(&sc->locID[i], loc_%" PRIu64 "", i);
			&sc->tempLen = sprintf(&sc->tempStr, "	%s %s;\n", vecType, &sc->locID[i]);
			
			}*/
			/*&sc->tempLen = sprintf(&sc->tempStr, "	{\n\
	%s loc_0;\n	%s loc_1;\n	%s loc_2;\n	%s loc_3;\n	%s loc_4;\n", vecType, vecType, vecType, vecType, vecType);*/

		for (pfUINT i = radix - 1; i > 0; i--) {
			if (stageSize == 1) {
				temp_complex.data.c[0].data.d = pfFPinit("1.0");
				temp_complex.data.c[1].data.d = pfFPinit("0.0");
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
						temp_double.data.d = pfFPinit("2.0") * i / radix;
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
						temp_double.data.d = pfFPinit("2.0") * i / radix;
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
		
		PfMul(sc, &regID[3], &regID[3], &tf[1], &regID[0]);
		
		PfMul(sc, &regID[4], &regID[4], &tf[2], &regID[0]);
		
		PfMul(sc, &sc->locID[3], &sc->locID[3], &tf[3], &regID[0]);
		
		PfMul(sc, &sc->locID[4], &sc->locID[4], &tf[4], &regID[0]);
		
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
			PfShuffleComplex(sc, &regID[1], &sc->locID[1], &sc->locID[4], &sc->locID[0]);
			
			PfShuffleComplex(sc, &regID[2], &sc->locID[2], &sc->locID[3], &sc->locID[0]);
			
			PfShuffleComplexInv(sc, &regID[3], &sc->locID[2], &sc->locID[3], &sc->locID[0]);
			
			PfShuffleComplexInv(sc, &regID[4], &sc->locID[1], &sc->locID[4], &sc->locID[0]);
			
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
			PfShuffleComplexInv(sc, &regID[1], &sc->locID[1], &sc->locID[4], &sc->locID[0]);
			
			PfShuffleComplexInv(sc, &regID[2], &sc->locID[2], &sc->locID[3], &sc->locID[0]);
			
			PfShuffleComplex(sc, &regID[3], &sc->locID[2], &sc->locID[3], &sc->locID[0]);
			
			PfShuffleComplex(sc, &regID[4], &sc->locID[1], &sc->locID[4], &sc->locID[0]);
			
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
		PfContainer tf[2] = VKFFT_ZERO_INIT;
		for (pfINT i = 0; i < 2; i++){
			tf[i].type = 22;
		}
		//PfAppendLine(sc, "	{\n");
		

		tf[0].data.d = pfFPinit("-0.5");
		tf[1].data.d = pfFPinit("-0.8660254037844386467637231707529361834714");
		for (pfUINT i = radix - 1; i > 0; i--) {
			if (stageSize == 1) {
				temp_complex.data.c[0].data.d = pfFPinit("1.0");
				temp_complex.data.c[1].data.d = pfFPinit("0.0");
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
						temp_double.data.d = pfFPinit("2.0") * i / radix;
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
						temp_double.data.d = pfFPinit("2.0") * i / radix;
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

		//pfUINT P = 3;
		pfUINT Q = 2;
		for (pfUINT i = 0; i < Q; i++) {
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
				PfShuffleComplex(sc, &regID[i + Q], &sc->locID[1], &sc->locID[2], &sc->locID[0]);
				
				PfShuffleComplexInv(sc, &regID[i + 2 * Q], &sc->locID[1], &sc->locID[2], &sc->locID[0]);
				
			}
			else {
				PfShuffleComplexInv(sc, &regID[i + Q], &sc->locID[1], &sc->locID[2], &sc->locID[0]);
				
				PfShuffleComplex(sc, &regID[i + 2 * Q], &sc->locID[1], &sc->locID[2], &sc->locID[0]);
				
			}
		}

		PfMov(sc, &sc->temp, &regID[1]);
		
		PfSub(sc, &regID[1], &regID[0], &sc->temp);
		
		PfAdd(sc, &regID[0], &regID[0], &sc->temp);
		
		if (stageAngle < 0) {
			temp_complex.data.c[0].data.d = pfFPinit("-0.5");
			temp_complex.data.c[1].data.d = pfFPinit("0.8660254037844386467637231707529361834714");
			PfMov(sc, &sc->w, &temp_complex);	
			
		}
		else {
			temp_complex.data.c[0].data.d = pfFPinit("-0.5");
			temp_complex.data.c[1].data.d = pfFPinit("-0.8660254037844386467637231707529361834714");
			PfMov(sc, &sc->w, &temp_complex);
			
		}

		PfMul(sc, &sc->temp, &regID[3], &sc->w, 0);
		
		PfSub(sc, &regID[3], &regID[2], &sc->temp);
		
		PfAdd(sc, &regID[2], &regID[2], &sc->temp);
		

		PfConjugate(sc, &sc->w, &sc->w);
		

		PfMul(sc, &sc->temp, &regID[5], &sc->w, 0);
		
		PfSub(sc, &regID[5], &regID[4], &sc->temp);
		
		PfAdd(sc, &regID[4], &regID[4], &sc->temp);
		

		pfUINT permute2[6] = { 0,3,4,1,2,5 };
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
		PfContainer tf_x[6] = VKFFT_ZERO_INIT;
		PfContainer tf_y[6] = VKFFT_ZERO_INIT;
		for (pfINT i = 0; i < 6; i++){
			tf_x[i].type = 22;
			tf_y[i].type = 22;
		}
		
		tf_x[0].data.d = pfFPinit("0.6234898018587335305250048840042398106322747308964021053655");
		tf_x[1].data.d = pfFPinit("-0.222520933956314404288902564496794759466355568764544955311");
		tf_x[2].data.d = pfFPinit("-0.900968867902419126236102319507445051165919162131857150053");
		tf_x[3].data.d = tf_x[0].data.d;
		tf_x[4].data.d = tf_x[1].data.d;
		tf_x[5].data.d = tf_x[2].data.d;
		if (stageAngle < 0) {
			tf_y[0].data.d = pfFPinit("-0.7818314824680298087084445266740577502323345187086875289806");
			tf_y[1].data.d = pfFPinit("0.9749279121818236070181316829939312172327858006199974376480");
			tf_y[2].data.d = pfFPinit("0.4338837391175581204757683328483587546099907277874598764445");
			tf_y[3].data.d = -tf_y[0].data.d;
			tf_y[4].data.d = -tf_y[1].data.d;
			tf_y[5].data.d = -tf_y[2].data.d;
		}
		else {
			tf_y[0].data.d = pfFPinit("0.7818314824680298087084445266740577502323345187086875289806");
			tf_y[1].data.d = pfFPinit("-0.9749279121818236070181316829939312172327858006199974376480");
			tf_y[2].data.d = pfFPinit("-0.4338837391175581204757683328483587546099907277874598764445");
			tf_y[3].data.d = -tf_y[0].data.d;
			tf_y[4].data.d = -tf_y[1].data.d;
			tf_y[5].data.d = -tf_y[2].data.d;
		}
		for (pfUINT i = radix - 1; i > 0; i--) {
			if (stageSize == 1) {
				temp_complex.data.c[0].data.d = pfFPinit("1.0");
				temp_complex.data.c[1].data.d = pfFPinit("0.0");
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
						temp_double.data.d = pfFPinit("2.0") * i / radix;
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
						temp_double.data.d = pfFPinit("2.0") * i / radix;
						PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
						PfSinCos(sc, &sc->w, &sc->tempFloat);
					}
				}
			}
			PfMul(sc, &sc->locID[i], &regID[i], &sc->w, 0);
			
		}
		PfMov(sc, &sc->locID[0], &regID[0]);
		
		pfUINT permute[7] = { 0, 1, 3, 2, 6, 4, 5 };
		PfPermute(sc, permute, 7, 0, 0, &sc->w);
		
		for (pfUINT i = 0; i < 3; i++) {
			PfSub(sc, &regID[i + 4].data.c[0], &sc->locID[i + 1].data.c[0], &sc->locID[i + 4].data.c[0]);
			
			PfAdd(sc, &regID[i + 1].data.c[0], &sc->locID[i + 1].data.c[0], &sc->locID[i + 4].data.c[0]);
			
			PfAdd(sc, &regID[i + 4].data.c[1], &sc->locID[i + 1].data.c[1], &sc->locID[i + 4].data.c[1]);
			
			PfSub(sc, &regID[i + 1].data.c[1], &sc->locID[i + 1].data.c[1], &sc->locID[i + 4].data.c[1]);
			
		}
		for (pfUINT i = 0; i < 3; i++) {
			PfAdd(sc, &regID[0].data.c[0], &regID[0].data.c[0], &regID[i + 1].data.c[0]);
			
			PfAdd(sc, &regID[0].data.c[1], &regID[0].data.c[1], &regID[i + 4].data.c[1]);
			
		}
		for (pfUINT i = 1; i < 4; i++) {
			PfMov(sc, &sc->locID[i], &sc->locID[0]);
			
			
		}
		for (pfUINT i = 4; i < 7; i++) {
			PfSetToZero(sc, &sc->locID[i]);
		}
		for (pfUINT i = 0; i < 3; i++) {
			for (pfUINT j = 0; j < 3; j++) {
				pfUINT id = ((6 - i) + j) % 6;
				PfFMA3_const_w(sc, &sc->locID[j + 1], &sc->locID[j + 4], &regID[i + 1], &tf_x[id], &tf_y[id], &regID[i + 4], &sc->w, &sc->locID[0]);
				
			}
		}
		for (pfUINT i = 1; i < 4; i++) {
			PfSub(sc, &regID[i].data.c[0], &sc->locID[i].data.c[0], &sc->locID[i + 3].data.c[0]);
			
			PfAdd(sc, &regID[i].data.c[1], &sc->locID[i].data.c[1], &sc->locID[i + 3].data.c[1]);
			
		}
		for (pfUINT i = 1; i < 4; i++) {
			PfAdd(sc, &regID[i + 3].data.c[0], &sc->locID[i].data.c[0], &sc->locID[i + 3].data.c[0]);
			
			PfSub(sc, &regID[i + 3].data.c[1], &sc->locID[i].data.c[1], &sc->locID[i + 3].data.c[1]);
			
		}
		pfUINT permute2[7] = { 0, 1, 5, 6, 3, 2, 4 };
		PfPermute(sc, permute2, 7, 1, regID, &sc->w);
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
			temp_complex.data.c[0].data.d = pfFPinit("1.0");
			temp_complex.data.c[1].data.d = pfFPinit("0.0");
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
		for (pfUINT i = 0; i < 4; i++) {
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
			temp_complex.data.c[0].data.d = pfFPinit("1.0");
			temp_complex.data.c[1].data.d = pfFPinit("0.0");
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
				temp_double.data.d = pfFPinit("0.5");
				PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
				PfSinCos(sc, &sc->w, &sc->tempFloat);
			}
		}
		for (pfUINT i = 0; i < 2; i++) {
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
			
			PfMov(sc, &sc->iw.data.c[0], &sc->w.data.c[1]);
			PfMovNeg(sc, &sc->iw.data.c[1], &sc->w.data.c[0]);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(w.y, -w.x);\n\n", vecType);
		}
		else {
			
			PfMovNeg(sc, &sc->iw.data.c[0], &sc->w.data.c[1]);
			PfMov(sc, &sc->iw.data.c[1], &sc->w.data.c[0]);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	iw = %s(-w.y, w.x);\n\n", vecType);
		}

		for (pfUINT i = 4; i < 6; i++) {
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
			temp_complex.data.c[0].data.d = pfFPinit("1.0");
			temp_complex.data.c[1].data.d = pfFPinit("0.0");
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
				temp_double.data.d = pfFPinit("0.25");
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
			
			PfMov(sc, &sc->iw.data.c[0], &sc->w.data.c[1]);
			PfMovNeg(sc, &sc->iw.data.c[1], &sc->w.data.c[0]);
			
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(w.y, -w.x);\n\n", vecType);
		}
		else {
			PfMovNeg(sc, &sc->iw.data.c[0], &sc->w.data.c[1]);
			PfMov(sc, &sc->iw.data.c[1], &sc->w.data.c[0]);
			
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
			temp_complex.data.c[0].data.d = pfFPinit("0.70710678118654752440084436210485");
			temp_complex.data.c[1].data.d = pfFPinit("-0.70710678118654752440084436210485");
			PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
		
		}
		else {
			temp_complex.data.c[0].data.d = pfFPinit("0.70710678118654752440084436210485");
			temp_complex.data.c[1].data.d = pfFPinit("0.70710678118654752440084436210485");
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
			PfMov(sc, &sc->w.data.c[0], &sc->iw.data.c[1]);
			PfMovNeg(sc, &sc->w.data.c[1], &sc->iw.data.c[0]);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(iw.y, -iw.x);\n\n", vecType);
		}
		else {
			PfMovNeg(sc, &sc->w.data.c[0], &sc->iw.data.c[1]);
			PfMov(sc, &sc->w.data.c[1], &sc->iw.data.c[0]);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(-iw.y, iw.x);\n\n", vecType);
		}
		PfMul(sc, &sc->temp, &regID[7], &sc->w, 0);
		
		PfSub(sc, &regID[7], &regID[6], &sc->temp);
		
		PfAdd(sc, &regID[6], &regID[6], &sc->temp);
		

		pfUINT permute2[8] = { 0,4,2,6,1,5,3,7 };
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
		PfContainer tf[2] = VKFFT_ZERO_INIT;
		//PfAppendLine(sc, "	{\n");
		for (pfINT i = 0; i < 2; i++){
			tf[i].type = 22;
		}

		tf[0].data.d = pfFPinit("-0.5");
		tf[1].data.d = pfFPinit("-0.8660254037844386467637231707529361834714");
		for (pfUINT i = radix - 1; i > 0; i--) {
			if (stageSize == 1) {
				temp_complex.data.c[0].data.d = pfFPinit("1.0");
				temp_complex.data.c[1].data.d = pfFPinit("0.0");
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
						temp_double.data.d = pfFPinit("2.0") * i / radix;
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
						temp_double.data.d = pfFPinit("2.0") * i / radix;
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
		pfUINT P = 3;
		pfUINT Q = 3;
		for (pfUINT i = 0; i < Q; i++) {
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
				PfShuffleComplex(sc, &regID[i + Q], &sc->locID[1], &sc->locID[2], &sc->locID[0]);
				
				PfShuffleComplexInv(sc, &regID[i + 2 * Q], &sc->locID[1], &sc->locID[2], &sc->locID[0]);
				
			}
			else {
				PfShuffleComplexInv(sc, &regID[i + Q], &sc->locID[1], &sc->locID[2], &sc->locID[0]);
				
				PfShuffleComplex(sc, &regID[i + 2 * Q], &sc->locID[1], &sc->locID[2], &sc->locID[0]);
				
			}
		}


		for (pfUINT i = 0; i < P; i++) {
			if (i > 0) {
				if (stageAngle < 0) {
					temp_complex.data.c[0].data.d = pfcos(2 * i * sc->double_PI / radix);
					temp_complex.data.c[1].data.d = -pfsin(2 * i * sc->double_PI / radix);
					PfMov(sc, &sc->w, &temp_complex);	
					
				}
				else {
					temp_complex.data.c[0].data.d = pfcos(2 * i * sc->double_PI / radix);
					temp_complex.data.c[1].data.d = pfsin(2 * i * sc->double_PI / radix);
					PfMov(sc, &sc->w, &temp_complex);	
					
				}
				PfMul(sc, &sc->locID[1], &regID[Q * i + 1], &sc->w, &sc->temp);
				
				if (stageAngle < 0) {
					temp_complex.data.c[0].data.d = pfcos(4 * i * sc->double_PI / radix);
					temp_complex.data.c[1].data.d = -pfsin(4 * i * sc->double_PI / radix);
					PfMov(sc, &sc->w, &temp_complex);	
					
				}
				else {
					temp_complex.data.c[0].data.d = pfcos(4 * i * sc->double_PI / radix);
					temp_complex.data.c[1].data.d = pfsin(4 * i * sc->double_PI / radix);
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
				PfShuffleComplex(sc, &regID[Q * i + 1], &sc->locID[1], &sc->locID[2], &sc->locID[0]);
				
				PfShuffleComplexInv(sc, &regID[Q * i + 2], &sc->locID[1], &sc->locID[2], &sc->locID[0]);
				
			}
			else {
				PfShuffleComplexInv(sc, &regID[Q * i + 1], &sc->locID[1], &sc->locID[2], &sc->locID[0]);
				
				PfShuffleComplex(sc, &regID[Q * i + 2], &sc->locID[1], &sc->locID[2], &sc->locID[0]);
				
			}
		}

		pfUINT permute2[9] = { 0,3,6,1,4,7,2,5,8 };
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
		PfContainer tf[5] = VKFFT_ZERO_INIT;
		for (pfINT i = 0; i < 5; i++){
			tf[i].type = 22;
		}
		//PfAppendLine(sc, "	{\n");
		
		tf[0].data.d = pfFPinit("-0.5");
		tf[1].data.d = pfFPinit("1.538841768587626701285145288018455");
		tf[2].data.d = pfFPinit("-0.363271264002680442947733378740309");
		tf[3].data.d = pfFPinit("-0.809016994374947424102293417182819");
		tf[4].data.d = pfFPinit("-0.587785252292473129168705954639073");
		for (pfUINT i = radix - 1; i > 0; i--) {
			if (stageSize == 1) {
				temp_complex.data.c[0].data.d = pfFPinit("1.0");
				temp_complex.data.c[1].data.d = pfFPinit("0.0");
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
						temp_double.data.d = pfFPinit("2.0") * i / radix;
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
						temp_double.data.d = pfFPinit("2.0") * i / radix;
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

		pfUINT P = 5;
		pfUINT Q = 2;
		for (pfUINT i = 0; i < Q; i++) {
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
			
			PfMul(sc, &regID[i + 3 * Q], &regID[i + 3 * Q], &tf[1], &regID[i]);
			
			PfMul(sc, &regID[i + 4 * Q], &regID[i + 4 * Q], &tf[2], &regID[i]);
			
			PfMul(sc, &sc->locID[3], &sc->locID[3], &tf[3], &regID[i]);
			
			PfMul(sc, &sc->locID[4], &sc->locID[4], &tf[4], &regID[i]);
			

			PfSub(sc, &sc->locID[1], &sc->locID[1], &sc->locID[3]);
			
			PfAdd(sc, &sc->locID[2], &sc->locID[2], &sc->locID[3]);
			
			PfAdd(sc, &sc->locID[3], &regID[i + 3 * Q], &sc->locID[4]);
			
			PfAdd(sc, &sc->locID[4], &sc->locID[4], &regID[i + 4 * Q]);
			
			PfMov(sc, &regID[i], &sc->locID[0]);
			

			if (stageAngle < 0)
			{
				PfShuffleComplex(sc, &regID[i + Q], &sc->locID[1], &sc->locID[4], &sc->locID[0]);
				
				PfShuffleComplex(sc, &regID[i + 2 * Q], &sc->locID[2], &sc->locID[3], &sc->locID[0]);
				
				PfShuffleComplexInv(sc, &regID[i + 3 * Q], &sc->locID[2], &sc->locID[3], &sc->locID[0]);
				
				PfShuffleComplexInv(sc, &regID[i + 4 * Q], &sc->locID[1], &sc->locID[4], &sc->locID[0]);
				
			}
			else {
				PfShuffleComplexInv(sc, &regID[i + Q], &sc->locID[1], &sc->locID[4], &sc->locID[0]);
				
				PfShuffleComplexInv(sc, &regID[i + 2 * Q], &sc->locID[2], &sc->locID[3], &sc->locID[0]);
				
				PfShuffleComplex(sc, &regID[i + 3 * Q], &sc->locID[2], &sc->locID[3], &sc->locID[0]);
				
				PfShuffleComplex(sc, &regID[i + 4 * Q], &sc->locID[1], &sc->locID[4], &sc->locID[0]);
				
			}

		}


		for (pfUINT i = 0; i < P; i++) {
			if (i > 0) {
				if (stageAngle < 0) {
					temp_complex.data.c[0].data.d = pfcos(2 * i * sc->double_PI / radix);
					temp_complex.data.c[1].data.d = -pfsin(2 * i * sc->double_PI / radix);
					PfMov(sc, &sc->w, &temp_complex);	
					
				}
				else {
					temp_complex.data.c[0].data.d = pfcos(2 * i * sc->double_PI / radix);
					temp_complex.data.c[1].data.d = pfsin(2 * i * sc->double_PI / radix);
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

		pfUINT permute2[10] = { 0, 2, 4, 6, 8, 1, 3, 5, 7, 9 };
		PfPermute(sc, permute2, 10, 1, regID, &sc->temp);
		break;
	}
	case 11: {
		PfContainer tf_x[20] = VKFFT_ZERO_INIT;
		PfContainer tf_y[20] = VKFFT_ZERO_INIT;
		for (pfINT i = 0; i < 20; i++){
			tf_x[i].type = 22;
			tf_y[i].type = 22;
		}
		
		tf_x[0].data.d = pfFPinit("0.8412535328311811688618116489193677175132924984205378986426");
		tf_x[1].data.d = pfFPinit("-0.959492973614497389890368057066327699062454848422161955044");
		tf_x[2].data.d = pfFPinit("-0.142314838273285140443792668616369668791051361125984328418");
		tf_x[3].data.d = pfFPinit("-0.654860733945285064056925072466293553183791199336928427606");
		tf_x[4].data.d = pfFPinit("0.4154150130018864255292741492296232035240049104645368124262");
		tf_x[5].data.d = tf_x[0].data.d;
		tf_x[6].data.d = tf_x[1].data.d;
		tf_x[7].data.d = tf_x[2].data.d;
		tf_x[8].data.d = tf_x[3].data.d;
		tf_x[9].data.d = tf_x[4].data.d;
		if (stageAngle < 0) {
			tf_y[0].data.d = pfFPinit("-0.5406408174555975821076359543186916954317706078981138400357");
			tf_y[1].data.d = pfFPinit("0.2817325568414296977114179153466168990357778989732668718310");
			tf_y[2].data.d = pfFPinit("-0.9898214418809327323760920377767187873765193719487166878386");
			tf_y[3].data.d = pfFPinit("0.7557495743542582837740358439723444201797174451692235695799");
			tf_y[4].data.d = pfFPinit("0.9096319953545183714117153830790284600602410511946441707561");
			tf_y[5].data.d = -tf_y[0].data.d;
			tf_y[6].data.d = -tf_y[1].data.d;
			tf_y[7].data.d = -tf_y[2].data.d;
			tf_y[8].data.d = -tf_y[3].data.d;
			tf_y[9].data.d = -tf_y[4].data.d;
		}
		else {
			tf_y[0].data.d = pfFPinit("0.5406408174555975821076359543186916954317706078981138400357");
			tf_y[1].data.d = pfFPinit("-0.2817325568414296977114179153466168990357778989732668718310");
			tf_y[2].data.d = pfFPinit("0.9898214418809327323760920377767187873765193719487166878386");
			tf_y[3].data.d = pfFPinit("-0.7557495743542582837740358439723444201797174451692235695799");
			tf_y[4].data.d = pfFPinit("-0.9096319953545183714117153830790284600602410511946441707561");
			tf_y[5].data.d = -tf_y[0].data.d;
			tf_y[6].data.d = -tf_y[1].data.d;
			tf_y[7].data.d = -tf_y[2].data.d;
			tf_y[8].data.d = -tf_y[3].data.d;
			tf_y[9].data.d = -tf_y[4].data.d;
		}
		for (pfUINT i = radix - 1; i > 0; i--) {
			if (stageSize == 1) {
				temp_complex.data.c[0].data.d = pfFPinit("1.0");
				temp_complex.data.c[1].data.d = pfFPinit("0.0");
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
						temp_double.data.d = pfFPinit("2.0") * i / radix;
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
						temp_double.data.d = pfFPinit("2.0") * i / radix;
						PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
						PfSinCos(sc, &sc->w, &sc->tempFloat);
					}
				}
			}
			PfMul(sc, &sc->locID[i], &regID[i], &sc->w, 0);
			
		}
		PfMov(sc, &sc->locID[0], &regID[0]);
		
		pfUINT permute[11] = { 0,1,2,4,8,5,10,9,7,3,6 };
		PfPermute(sc, permute, 11, 0, 0, &sc->w);
		
		for (pfUINT i = 0; i < 5; i++) {
			PfSub(sc, &regID[i + 6].data.c[0], &sc->locID[i + 1].data.c[0], &sc->locID[i + 6].data.c[0]);
			
			PfAdd(sc, &regID[i + 1].data.c[0], &sc->locID[i + 1].data.c[0], &sc->locID[i + 6].data.c[0]);
			
			PfAdd(sc, &regID[i + 6].data.c[1], &sc->locID[i + 1].data.c[1], &sc->locID[i + 6].data.c[1]);
			
			PfSub(sc, &regID[i + 1].data.c[1], &sc->locID[i + 1].data.c[1], &sc->locID[i + 6].data.c[1]);
			
		}
		for (pfUINT i = 0; i < 5; i++) {
			PfAdd(sc, &regID[0].data.c[0], &regID[0].data.c[0], &regID[i + 1].data.c[0]);
			
			PfAdd(sc, &regID[0].data.c[1], &regID[0].data.c[1], &regID[i + 6].data.c[1]);
			
		}
		for (pfUINT i = 1; i < 6; i++) {
			PfMov(sc, &sc->locID[i], &sc->locID[0]);
			
			
		}
		for (pfUINT i = 6; i < 11; i++) {
			PfSetToZero(sc, &sc->locID[i]);
		}
		for (pfUINT i = 0; i < 5; i++) {
			for (pfUINT j = 0; j < 5; j++) {
				pfUINT id = ((10 - i) + j) % 10;
				PfFMA3_const_w(sc, &sc->locID[j + 1], &sc->locID[j + 6], &regID[i + 1], &tf_x[id], &tf_y[id], &regID[i + 6], &sc->w, &sc->locID[0]);
				
			}
		}
		for (pfUINT i = 1; i < 6; i++) {
			PfSub(sc, &regID[i].data.c[0], &sc->locID[i].data.c[0], &sc->locID[i + 5].data.c[0]);
			
			PfAdd(sc, &regID[i].data.c[1], &sc->locID[i].data.c[1], &sc->locID[i + 5].data.c[1]);
			
		}
		for (pfUINT i = 1; i < 6; i++) {
			PfAdd(sc, &regID[i + 5].data.c[0], &sc->locID[i].data.c[0], &sc->locID[i + 5].data.c[0]);
			
			PfSub(sc, &regID[i + 5].data.c[1], &sc->locID[i].data.c[1], &sc->locID[i + 5].data.c[1]);
			
		}

		pfUINT permute2[11] = { 0,1,10,3,9,7,2,4,8,5,6 };
		PfPermute(sc, permute2, 11, 1, regID, &sc->w);
		break;
	}
	case 12: {
		PfContainer tf[2] = VKFFT_ZERO_INIT;
		for (pfINT i = 0; i < 2; i++){
			tf[i].type = 22;
		}
		//PfAppendLine(sc, "	{\n");
		
		tf[0].data.d = pfFPinit("-0.5");
		tf[1].data.d = pfFPinit("-0.8660254037844386467637231707529361834714");
		for (pfUINT i = radix - 1; i > 0; i--) {
			if (stageSize == 1) {
				temp_complex.data.c[0].data.d = pfFPinit("1.0");
				temp_complex.data.c[1].data.d = pfFPinit("0.0");
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
						temp_double.data.d = pfFPinit("2.0") * i / radix;
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
						temp_double.data.d = pfFPinit("2.0") * i / radix;
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
		pfUINT P = 3;
		pfUINT Q = 4;
		for (pfUINT i = 0; i < Q; i++) {
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
				PfShuffleComplex(sc, &regID[i + Q], &sc->locID[1], &sc->locID[2], &sc->locID[0]);
				
				PfShuffleComplexInv(sc, &regID[i + 2 * Q], &sc->locID[1], &sc->locID[2], &sc->locID[0]);
				
			}
			else {
				PfShuffleComplexInv(sc, &regID[i + Q], &sc->locID[1], &sc->locID[2], &sc->locID[0]);
				
				PfShuffleComplex(sc, &regID[i + 2 * Q], &sc->locID[1], &sc->locID[2], &sc->locID[0]);
				
			}
		}


		for (pfUINT i = 0; i < P; i++) {
			for (pfUINT j = 0; j < Q; j++) {
				if (i > 0) {
					if (stageAngle < 0) {
						temp_complex.data.c[0].data.d = pfcos(2 * i * j * sc->double_PI / radix);
						temp_complex.data.c[1].data.d = -pfsin(2 * i * j * sc->double_PI / radix);
						PfMov(sc, &sc->w, &temp_complex);	
						
					}
					else {
						temp_complex.data.c[0].data.d = pfcos(2 * i * j * sc->double_PI / radix);
						temp_complex.data.c[1].data.d = pfsin(2 * i * j * sc->double_PI / radix);
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
				PfMov(sc, &sc->temp.data.c[0], &regID[Q * i + 3].data.c[1]);
				PfMovNeg(sc, &sc->temp.data.c[1], &regID[Q * i + 3].data.c[0]);
				
			}
			else {
				PfMovNeg(sc, &sc->temp.data.c[0], &regID[Q * i + 3].data.c[1]);
				PfMov(sc, &sc->temp.data.c[1], &regID[Q * i + 3].data.c[0]);
				
			}
			PfSub(sc, &regID[Q * i + 3], &regID[Q * i + 2], &sc->temp);
			
			PfAdd(sc, &regID[Q * i + 2], &regID[Q * i + 2], &sc->temp);
			
		}

		pfUINT permute2[12] = { 0,4,8,2,6,10,1,5,9,3,7,11 };
		PfPermute(sc, permute2, 12, 1, regID, &sc->temp);
		
		break;
	}
	case 13: {
		PfContainer tf_x[20] = VKFFT_ZERO_INIT;
		for (pfINT i = 0; i < 20; i++){
			tf_x[i].type = 22;
		}
		PfContainer tf_y[20] = VKFFT_ZERO_INIT;
		for (pfINT i = 0; i < 20; i++){
			tf_y[i].type = 22;
		}
		
		tf_x[0].data.d = pfFPinit("0.8854560256532098959003755220150988786054984163475349018024");
		tf_x[1].data.d = pfFPinit("-0.970941817426052027156982276293789227249865105739003588587");
		tf_x[2].data.d = pfFPinit("0.1205366802553230533490676874525435822736811592275714047969");
		tf_x[3].data.d = pfFPinit("-0.748510748171101098634630599701351383846451590175826134069");
		tf_x[4].data.d = pfFPinit("-0.354604887042535625969637892600018474316355432113794753421");
		tf_x[5].data.d = pfFPinit("0.5680647467311558025118075591275166245334925524535181694796");
		tf_x[6].data.d = tf_x[0].data.d;
		tf_x[7].data.d = tf_x[1].data.d;
		tf_x[8].data.d = tf_x[2].data.d;
		tf_x[9].data.d = tf_x[3].data.d;
		tf_x[10].data.d = tf_x[4].data.d;
		tf_x[11].data.d = tf_x[5].data.d;
		if (stageAngle < 0) {
			tf_y[0].data.d = pfFPinit("-0.4647231720437685456560153351331047775577358653324689769540");
			tf_y[1].data.d = pfFPinit("0.2393156642875577671487537262602118952031730227383060133551");
			tf_y[2].data.d = pfFPinit("0.9927088740980539928007516494925201793436756329701668557709");
			tf_y[3].data.d = pfFPinit("-0.6631226582407952023767854926667662795247641070441061881807");
			tf_y[4].data.d = pfFPinit("0.9350162426854148234397845998378307290505174695784318706963");
			tf_y[5].data.d = pfFPinit("0.8229838658936563945796174234393819906550676930875738058270");
			tf_y[6].data.d = -tf_y[0].data.d;
			tf_y[7].data.d = -tf_y[1].data.d;
			tf_y[8].data.d = -tf_y[2].data.d;
			tf_y[9].data.d = -tf_y[3].data.d;
			tf_y[10].data.d = -tf_y[4].data.d;
			tf_y[11].data.d = -tf_y[5].data.d;
		}
		else {
			tf_y[0].data.d = pfFPinit("0.4647231720437685456560153351331047775577358653324689769540");
			tf_y[1].data.d = pfFPinit("-0.2393156642875577671487537262602118952031730227383060133551");
			tf_y[2].data.d = pfFPinit("-0.9927088740980539928007516494925201793436756329701668557709");
			tf_y[3].data.d = pfFPinit("0.6631226582407952023767854926667662795247641070441061881807");
			tf_y[4].data.d = pfFPinit("-0.9350162426854148234397845998378307290505174695784318706963");
			tf_y[5].data.d = pfFPinit("-0.8229838658936563945796174234393819906550676930875738058270");
			tf_y[6].data.d = -tf_y[0].data.d;
			tf_y[7].data.d = -tf_y[1].data.d;
			tf_y[8].data.d = -tf_y[2].data.d;
			tf_y[9].data.d = -tf_y[3].data.d;
			tf_y[10].data.d = -tf_y[4].data.d;
			tf_y[11].data.d = -tf_y[5].data.d;
		}
		for (pfUINT i = radix - 1; i > 0; i--) {
			if (stageSize == 1) {
				temp_complex.data.c[0].data.d = pfFPinit("1.0");
				temp_complex.data.c[1].data.d = pfFPinit("0.0");
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
						temp_double.data.d = pfFPinit("2.0") * i / radix;
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
						temp_double.data.d = pfFPinit("2.0") * i / radix;
						PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
						PfSinCos(sc, &sc->w, &sc->tempFloat);
					}
				}
			}
			PfMul(sc, &sc->locID[i], &regID[i], &sc->w, 0);
			
		}
		PfMov(sc, &sc->locID[0], &regID[0]);
		
		pfUINT permute[13] = { 0, 1, 2, 4, 8, 3, 6, 12, 11, 9, 5, 10, 7 };
		PfPermute(sc, permute, 13, 0, 0, &sc->w);
		
		for (pfUINT i = 0; i < 6; i++) {
			PfSub(sc, &regID[i + 7].data.c[0], &sc->locID[i + 1].data.c[0], &sc->locID[i + 7].data.c[0]);
			
			PfAdd(sc, &regID[i + 1].data.c[0], &sc->locID[i + 1].data.c[0], &sc->locID[i + 7].data.c[0]);
			
			PfAdd(sc, &regID[i + 7].data.c[1], &sc->locID[i + 1].data.c[1], &sc->locID[i + 7].data.c[1]);
			
			PfSub(sc, &regID[i + 1].data.c[1], &sc->locID[i + 1].data.c[1], &sc->locID[i + 7].data.c[1]);
			
		}
		for (pfUINT i = 0; i < 6; i++) {
			PfAdd(sc, &regID[0].data.c[0], &regID[0].data.c[0], &regID[i + 1].data.c[0]);
			
			PfAdd(sc, &regID[0].data.c[1], &regID[0].data.c[1], &regID[i + 7].data.c[1]);
			
		}
		for (pfUINT i = 1; i < 7; i++) {
			PfMov(sc, &sc->locID[i], &sc->locID[0]);
			
		}
		for (pfUINT i = 7; i < 13; i++) {
			PfSetToZero(sc, &sc->locID[i]);
		}
		for (pfUINT i = 0; i < 6; i++) {
			for (pfUINT j = 0; j < 6; j++) {
				pfUINT id = ((12 - i) + j) % 12;
				PfFMA3_const_w(sc, &sc->locID[j + 1], &sc->locID[j + 7], &regID[i + 1], &tf_x[id], &tf_y[id], &regID[i + 7], &sc->w, &sc->locID[0]);
				
			}
		}
		for (pfUINT i = 1; i < 7; i++) {
			PfSub(sc, &regID[i].data.c[0], &sc->locID[i].data.c[0], &sc->locID[i + 6].data.c[0]);
			
			PfAdd(sc, &regID[i].data.c[1], &sc->locID[i].data.c[1], &sc->locID[i + 6].data.c[1]);
			
		}
		for (pfUINT i = 1; i < 7; i++) {
			PfAdd(sc, &regID[i + 6].data.c[0], &sc->locID[i].data.c[0], &sc->locID[i + 6].data.c[0]);
			
			PfSub(sc, &regID[i + 6].data.c[1], &sc->locID[i].data.c[1], &sc->locID[i + 6].data.c[1]);
			
		}

		pfUINT permute2[13] = { 0,1,12,9,11,4,8,2,10,5,3,6,7 };
		PfPermute(sc, permute2, 13, 1, regID, &sc->w);
		//
		break;
	}
	case 14: {
		PfContainer tf_x[6] = VKFFT_ZERO_INIT;
		PfContainer tf_y[6] = VKFFT_ZERO_INIT;
		for (pfINT i = 0; i < 6; i++){
			tf_x[i].type = 22;
			tf_y[i].type = 22;
		}
		
		tf_x[0].data.d = pfFPinit("0.6234898018587335305250048840042398106322747308964021053655");
		tf_x[1].data.d = pfFPinit("-0.222520933956314404288902564496794759466355568764544955311");
		tf_x[2].data.d = pfFPinit("-0.900968867902419126236102319507445051165919162131857150053");
		tf_x[3].data.d = tf_x[0].data.d;
		tf_x[4].data.d = tf_x[1].data.d;
		tf_x[5].data.d = tf_x[2].data.d;
		if (stageAngle < 0) {
			tf_y[0].data.d = pfFPinit("-0.7818314824680298087084445266740577502323345187086875289806");
			tf_y[1].data.d = pfFPinit("0.9749279121818236070181316829939312172327858006199974376480");
			tf_y[2].data.d = pfFPinit("0.4338837391175581204757683328483587546099907277874598764445");
			tf_y[3].data.d = -tf_y[0].data.d;
			tf_y[4].data.d = -tf_y[1].data.d;
			tf_y[5].data.d = -tf_y[2].data.d;
		}
		else {
			tf_y[0].data.d = pfFPinit("0.7818314824680298087084445266740577502323345187086875289806");
			tf_y[1].data.d = pfFPinit("-0.9749279121818236070181316829939312172327858006199974376480");
			tf_y[2].data.d = pfFPinit("-0.4338837391175581204757683328483587546099907277874598764445");
			tf_y[3].data.d = -tf_y[0].data.d;
			tf_y[4].data.d = -tf_y[1].data.d;
			tf_y[5].data.d = -tf_y[2].data.d;
		}

		for (pfUINT i = radix - 1; i > 0; i--) {
			if (stageSize == 1) {
				temp_complex.data.c[0].data.d = pfFPinit("1.0");
				temp_complex.data.c[1].data.d = pfFPinit("0.0");
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
						temp_double.data.d = pfFPinit("2.0") * i / radix;
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
						temp_double.data.d = pfFPinit("2.0") * i / radix;
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

		pfUINT P = 7;
		pfUINT Q = 2;
		PfContainer tempID[7] = VKFFT_ZERO_INIT;

		for (int t = 0; t < 7; t++) {
			tempID[t].type = 100 + sc->vecTypeCode;
			PfAllocateContainerFlexible(sc, &tempID[t], 50);
		}

		for (pfUINT i = 0; i < Q; i++) {
			pfUINT permute[7] = { 0, 1, 3, 2, 6, 4, 5 };
			
			for (pfUINT t = 0; t < 7; t++)
				PfCopyContainer(sc, &tempID[t], &regID[i + Q * t]);
			for (pfUINT t = 0; t < 7; t++)
				PfCopyContainer(sc, &regID[i + Q * t], &tempID[permute[t]]);

			PfMov(sc, &sc->locID[0], &regID[i]);		
			PfMov(sc, &sc->locID[1], &regID[i + Q]);	
			PfMov(sc, &sc->locID[2], &regID[i + 2 * Q]);	
			PfMov(sc, &sc->locID[3], &regID[i + 3 * Q]);	
			PfMov(sc, &sc->locID[4], &regID[i + 4 * Q]);	
			PfMov(sc, &sc->locID[5], &regID[i + 5 * Q]);	
			PfMov(sc, &sc->locID[6], &regID[i + 6 * Q]);
		
			for (pfUINT t = 0; t < 3; t++) {
				PfSub(sc, &regID[i + Q * (t + 4)].data.c[0], &sc->locID[t + 1].data.c[0], &sc->locID[t + 4].data.c[0]);
			
				PfAdd(sc, &regID[i + Q * (t + 1)].data.c[0], &sc->locID[t + 1].data.c[0], &sc->locID[t + 4].data.c[0]);
			
				PfAdd(sc, &regID[i + Q * (t + 4)].data.c[1], &sc->locID[t + 1].data.c[1], &sc->locID[t + 4].data.c[1]);
			
				PfSub(sc, &regID[i + Q * (t + 1)].data.c[1], &sc->locID[t + 1].data.c[1], &sc->locID[t + 4].data.c[1]);
			
			}
			for (pfUINT t = 0; t < 3; t++) {
				PfAdd(sc, &regID[i].data.c[0], &regID[i].data.c[0], &regID[i + Q * (t + 1)].data.c[0]);
			
				PfAdd(sc, &regID[i].data.c[1], &regID[i].data.c[1], &regID[i + Q * (t + 4)].data.c[1]);
			
			}
			for (pfUINT t = 1; t < 4; t++) {
				PfMov(sc, &sc->locID[t], &sc->locID[0]);
			
			
			}
			for (pfUINT t = 4; t < 7; t++) {
				PfSetToZero(sc, &sc->locID[t]);
			}
			for (pfUINT t = 0; t < 3; t++) {
				for (pfUINT j = 0; j < 3; j++) {
					pfUINT id = ((6 - t) + j) % 6;
					PfFMA3_const_w(sc, &sc->locID[j + 1], &sc->locID[j + 4], &regID[i + Q * (t + 1)], &tf_x[id], &tf_y[id], &regID[i + Q * (t + 4)], &sc->w, &sc->locID[0]);
				
				}
			}
			for (pfUINT t = 1; t < 4; t++) {
				PfSub(sc, &regID[i + Q * t].data.c[0], &sc->locID[t].data.c[0], &sc->locID[t + 3].data.c[0]);
			
				PfAdd(sc, &regID[i + Q * t].data.c[1], &sc->locID[t].data.c[1], &sc->locID[t + 3].data.c[1]);
			
			}
			for (pfUINT t = 1; t < 4; t++) {
				PfAdd(sc, &regID[i + Q * (t + 3)].data.c[0], &sc->locID[t].data.c[0], &sc->locID[t + 3].data.c[0]);
			
				PfSub(sc, &regID[i + Q * (t + 3)].data.c[1], &sc->locID[t].data.c[1], &sc->locID[t + 3].data.c[1]);
			
			}
			pfUINT permute2[7] = { 0, 1, 5, 6, 3, 2, 4 };
						
			for (pfUINT t = 0; t < 7; t++)
				PfCopyContainer(sc, &tempID[t], &regID[i + Q * t]);
			for (pfUINT t = 0; t < 7; t++)
				PfCopyContainer(sc, &regID[i + Q * t], &tempID[permute2[t]]);

		}

		for (int t = 0; t < 7; t++) {
			PfDeallocateContainer(sc, &tempID[t]);
		}

		for (pfUINT i = 0; i < P; i++) {
			if (i > 0) {
				if (stageAngle < 0) {
					temp_complex.data.c[0].data.d = pfcos(2 * i * sc->double_PI / radix);
					temp_complex.data.c[1].data.d = -pfsin(2 * i * sc->double_PI / radix);
					PfMov(sc, &sc->w, &temp_complex);	
					
				}
				else {
					temp_complex.data.c[0].data.d = pfcos(2 * i * sc->double_PI / radix);
					temp_complex.data.c[1].data.d = pfsin(2 * i * sc->double_PI / radix);
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

		pfUINT permute2[14] = { 0,2,4,6,8,10,12,1,3,5,7,9,11,13 };
		PfPermute(sc, permute2, 14, 1, regID, &sc->temp);
		
		break;
	}
	case 15: {
		PfContainer tf[5] = VKFFT_ZERO_INIT;
		for (pfINT i = 0; i < 5; i++){
			tf[i].type = 22;
		}
		//PfAppendLine(sc, "	{\n");
		
		tf[0].data.d = pfFPinit("-0.5");
		tf[1].data.d = pfFPinit("1.538841768587626701285145288018455");
		tf[2].data.d = pfFPinit("-0.363271264002680442947733378740309");
		tf[3].data.d = pfFPinit("-0.809016994374947424102293417182819");
		tf[4].data.d = pfFPinit("-0.587785252292473129168705954639073");

		PfContainer tf2[2] = VKFFT_ZERO_INIT;
		for (pfINT i = 0; i < 2; i++){
			tf2[i].type = 22;
		}
		//PfAppendLine(sc, "	{\n");
		

		tf2[0].data.d = pfFPinit("-0.5");
		tf2[1].data.d = pfFPinit("-0.8660254037844386467637231707529361834714");

		for (pfUINT i = radix - 1; i > 0; i--) {
			if (stageSize == 1) {
				temp_complex.data.c[0].data.d = pfFPinit("1.0");
				temp_complex.data.c[1].data.d = pfFPinit("0.0");
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
						temp_double.data.d = pfFPinit("2.0") * i / radix;
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
						temp_double.data.d = pfFPinit("2.0") * i / radix;
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

		pfUINT P = 5;
		pfUINT Q = 3;
		for (pfUINT i = 0; i < Q; i++) {
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
			
			PfMul(sc, &regID[i + 3 * Q], &regID[i + 3 * Q], &tf[1], &regID[i]);
			
			PfMul(sc, &regID[i + 4 * Q], &regID[i + 4 * Q], &tf[2], &regID[i]);
			
			PfMul(sc, &sc->locID[3], &sc->locID[3], &tf[3], &regID[i]);
			
			PfMul(sc, &sc->locID[4], &sc->locID[4], &tf[4], &regID[i]);
			

			PfSub(sc, &sc->locID[1], &sc->locID[1], &sc->locID[3]);
			
			PfAdd(sc, &sc->locID[2], &sc->locID[2], &sc->locID[3]);
			
			PfAdd(sc, &sc->locID[3], &regID[i + 3 * Q], &sc->locID[4]);
			
			PfAdd(sc, &sc->locID[4], &sc->locID[4], &regID[i + 4 * Q]);
			
			PfMov(sc, &regID[i], &sc->locID[0]);
			

			if (stageAngle < 0)
			{
				PfShuffleComplex(sc, &regID[i + Q], &sc->locID[1], &sc->locID[4], &sc->locID[0]);
				
				PfShuffleComplex(sc, &regID[i + 2 * Q], &sc->locID[2], &sc->locID[3], &sc->locID[0]);
				
				PfShuffleComplexInv(sc, &regID[i + 3 * Q], &sc->locID[2], &sc->locID[3], &sc->locID[0]);
				
				PfShuffleComplexInv(sc, &regID[i + 4 * Q], &sc->locID[1], &sc->locID[4], &sc->locID[0]);
				
			}
			else {
				PfShuffleComplexInv(sc, &regID[i + Q], &sc->locID[1], &sc->locID[4], &sc->locID[0]);
				
				PfShuffleComplexInv(sc, &regID[i + 2 * Q], &sc->locID[2], &sc->locID[3], &sc->locID[0]);
				
				PfShuffleComplex(sc, &regID[i + 3 * Q], &sc->locID[2], &sc->locID[3], &sc->locID[0]);
				
				PfShuffleComplex(sc, &regID[i + 4 * Q], &sc->locID[1], &sc->locID[4], &sc->locID[0]);
				
			}

		}


		for (pfUINT i = 0; i < P; i++) {
			if (i > 0) {
				if (stageAngle < 0) {
					temp_complex.data.c[0].data.d = pfcos(2 * i * sc->double_PI / radix);
					temp_complex.data.c[1].data.d = -pfsin(2 * i * sc->double_PI / radix);
					PfMov(sc, &sc->w, &temp_complex);	
					
				}
				else {
					temp_complex.data.c[0].data.d = pfcos(2 * i * sc->double_PI / radix);
					temp_complex.data.c[1].data.d = pfsin(2 * i * sc->double_PI / radix);
					PfMov(sc, &sc->w, &temp_complex);	
					
				}
				PfMul(sc, &sc->locID[1], &regID[Q * i + 1], &sc->w, &sc->temp);
				
				if (stageAngle < 0) {
					temp_complex.data.c[0].data.d = pfcos(4 * i * sc->double_PI / radix);
					temp_complex.data.c[1].data.d = -pfsin(4 * i * sc->double_PI / radix);
					PfMov(sc, &sc->w, &temp_complex);	
					
				}
				else {
					temp_complex.data.c[0].data.d = pfcos(4 * i * sc->double_PI / radix);
					temp_complex.data.c[1].data.d = pfsin(4 * i * sc->double_PI / radix);
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
				PfShuffleComplex(sc, &regID[Q * i + 1], &sc->locID[1], &sc->locID[2], &sc->locID[0]);
				
				PfShuffleComplexInv(sc, &regID[Q * i + 2], &sc->locID[1], &sc->locID[2], &sc->locID[0]);
				
			}
			else {
				PfShuffleComplexInv(sc, &regID[Q * i + 1], &sc->locID[1], &sc->locID[2], &sc->locID[0]);
				
				PfShuffleComplex(sc, &regID[Q * i + 2], &sc->locID[1], &sc->locID[2], &sc->locID[0]);
				
			}
		}

		pfUINT permute2[15] = { 0, 3, 6, 9, 12, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14 };
		PfPermute(sc, permute2, 15, 1, regID, &sc->temp);
		
		break;
	}
	case 16: {
		
		if (stageSize == 1) {
			temp_complex.data.c[0].data.d = pfFPinit("1.0");
			temp_complex.data.c[1].data.d = pfFPinit("0.0");
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
		for (pfUINT i = 0; i < 8; i++) {
			PfMul(sc, &sc->temp, &regID[i + 8], &sc->w, 0);
			
			PfSub(sc, &regID[i + 8], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageSize == 1) {
			temp_complex.data.c[0].data.d = pfFPinit("1.0");
			temp_complex.data.c[1].data.d = pfFPinit("0.0");
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
				temp_double.data.d = pfFPinit("0.5");
				PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
				PfSinCos(sc, &sc->w, &sc->tempFloat);
			}
		}
		for (pfUINT i = 0; i < 4; i++) {
			PfMul(sc, &sc->temp, &regID[i + 4], &sc->w, 0);
			
			PfSub(sc, &regID[i + 4], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageAngle < 0) {
			PfMov(sc, &sc->iw.data.c[0], &sc->w.data.c[1]);
			PfMovNeg(sc, &sc->iw.data.c[1], &sc->w.data.c[0]);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(w.y, -w.x);\n\n", vecType);
		}
		else {
			PfMovNeg(sc, &sc->iw.data.c[0], &sc->w.data.c[1]);
			PfMov(sc, &sc->iw.data.c[1], &sc->w.data.c[0]);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	iw = %s(-w.y, w.x);\n\n", vecType);
		}

		for (pfUINT i = 8; i < 12; i++) {
			PfMul(sc, &sc->temp, &regID[i + 4], &sc->iw, 0);
			
			PfSub(sc, &regID[i + 4], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageSize == 1) {
			temp_complex.data.c[0].data.d = pfFPinit("1.0");
			temp_complex.data.c[1].data.d = pfFPinit("0.0");
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
				temp_double.data.d = pfFPinit("0.25");
				PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
				PfSinCos(sc, &sc->w, &sc->tempFloat);
			}
		}
		for (pfUINT i = 0; i < 2; i++) {
			PfMul(sc, &sc->temp, &regID[i + 2], &sc->w, 0);
			
			PfSub(sc, &regID[i + 2], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageAngle < 0) {
			PfMov(sc, &sc->iw.data.c[0], &sc->w.data.c[1]);
			PfMovNeg(sc, &sc->iw.data.c[1], &sc->w.data.c[0]);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(w.y, -w.x);\n\n", vecType);
		}
		else {
			PfMovNeg(sc, &sc->iw.data.c[0], &sc->w.data.c[1]);
			PfMov(sc, &sc->iw.data.c[1], &sc->w.data.c[0]);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	iw = %s(-w.y, w.x);\n\n", vecType);
		}
		for (pfUINT i = 4; i < 6; i++) {
			PfMul(sc, &sc->temp, &regID[i + 2], &sc->iw, 0);
			
			PfSub(sc, &regID[i + 2], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageAngle < 0) {
			temp_complex.data.c[0].data.d = pfFPinit("0.70710678118654752440084436210485");
			temp_complex.data.c[1].data.d = pfFPinit("-0.70710678118654752440084436210485");
			PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);

		}
		else {
			temp_complex.data.c[0].data.d = pfFPinit("0.70710678118654752440084436210485");
			temp_complex.data.c[1].data.d = pfFPinit("0.70710678118654752440084436210485");
			PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
		}
		for (pfUINT i = 8; i < 10; i++) {
			PfMul(sc, &sc->temp, &regID[i + 2], &sc->iw, 0);
			
			PfSub(sc, &regID[i + 2], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageAngle < 0) {
			PfMov(sc, &sc->w.data.c[0], &sc->iw.data.c[1]);
			PfMovNeg(sc, &sc->w.data.c[1], &sc->iw.data.c[0]);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(iw.y, -iw.x);\n\n", vecType);
		}
		else {
			PfMovNeg(sc, &sc->w.data.c[0], &sc->iw.data.c[1]);
			PfMov(sc, &sc->w.data.c[1], &sc->iw.data.c[0]);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(-iw.y, iw.x);\n\n", vecType);
		}
		for (pfUINT i = 12; i < 14; i++) {
			PfMul(sc, &sc->temp, &regID[i + 2], &sc->w, 0);
			
			PfSub(sc, &regID[i + 2], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}

		if (stageSize == 1) {
			temp_complex.data.c[0].data.d = pfFPinit("1.0");
			temp_complex.data.c[1].data.d = pfFPinit("0.0");
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
				temp_double.data.d = pfFPinit("0.125");
				PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
				PfSinCos(sc, &sc->w, &sc->tempFloat);
			}
		}

		for (pfUINT i = 0; i < 1; i++) {
			PfMul(sc, &sc->temp, &regID[i + 1], &sc->w, 0);
			
			PfSub(sc, &regID[i + 1], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageAngle < 0) {
			PfMov(sc, &sc->iw.data.c[0], &sc->w.data.c[1]);
			PfMovNeg(sc, &sc->iw.data.c[1], &sc->w.data.c[0]);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(w.y, -w.x);\n\n", vecType);
		}
		else {
			PfMovNeg(sc, &sc->iw.data.c[0], &sc->w.data.c[1]);
			PfMov(sc, &sc->iw.data.c[1], &sc->w.data.c[0]);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	iw = %s(-w.y, w.x);\n\n", vecType);
		}
		for (pfUINT i = 2; i < 3; i++) {
			PfMul(sc, &sc->temp, &regID[i + 1], &sc->iw, 0);
			
			PfSub(sc, &regID[i + 1], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}


		if (stageAngle < 0) {
			temp_complex.data.c[0].data.d = pfFPinit("0.70710678118654752440084436210485");
			temp_complex.data.c[1].data.d = pfFPinit("-0.70710678118654752440084436210485");
			PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);

		}
		else {
			temp_complex.data.c[0].data.d = pfFPinit("0.70710678118654752440084436210485");
			temp_complex.data.c[1].data.d = pfFPinit("0.70710678118654752440084436210485");
			PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
		}
		for (pfUINT i = 4; i < 5; i++) {
			PfMul(sc, &sc->temp, &regID[i + 1], &sc->iw, 0);
			
			PfSub(sc, &regID[i + 1], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageAngle < 0) {
			PfMov(sc, &sc->temp.data.c[0], &sc->iw.data.c[1]);
			PfMovNeg(sc, &sc->temp.data.c[1], &sc->iw.data.c[0]);
			
			PfMov(sc, &sc->iw, &sc->temp);
			
		}
		else {
			PfMovNeg(sc, &sc->temp.data.c[0], &sc->iw.data.c[1]);
			PfMov(sc, &sc->temp.data.c[1], &sc->iw.data.c[0]);
			
			PfMov(sc, &sc->iw, &sc->temp);
			
		}
		for (pfUINT i = 6; i < 7; i++) {
			PfMul(sc, &sc->temp, &regID[i + 1], &sc->iw, 0);
			
			PfSub(sc, &regID[i + 1], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}


		for (pfUINT j = 0; j < 2; j++) {
			if (stageAngle < 0) {
				temp_complex.data.c[0].data.d = pfcos((2 * j + 1) * sc->double_PI / 8);
				temp_complex.data.c[1].data.d = -pfsin((2 * j + 1) * sc->double_PI / 8);
				PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
			}
			else {
				temp_complex.data.c[0].data.d = pfcos((2 * j + 1) * sc->double_PI / 8);
				temp_complex.data.c[1].data.d = pfsin((2 * j + 1) * sc->double_PI / 8);
				PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
			}
			for (pfUINT i = 8 + 4 * j; i < 9 + 4 * j; i++) {
				PfMul(sc, &sc->temp, &regID[i + 1], &sc->iw, 0);
				
				PfSub(sc, &regID[i + 1], &regID[i], &sc->temp);
				
				PfAdd(sc, &regID[i], &regID[i], &sc->temp);
				
			}
			if (stageAngle < 0) {
				PfMov(sc, &sc->temp.data.c[0], &sc->iw.data.c[1]);
				PfMovNeg(sc, &sc->temp.data.c[1], &sc->iw.data.c[0]);
				
				PfMov(sc, &sc->iw, &sc->temp);
				
			}
			else {
				PfMovNeg(sc, &sc->temp.data.c[0], &sc->iw.data.c[1]);
				PfMov(sc, &sc->temp.data.c[1], &sc->iw.data.c[0]);
				
				PfMov(sc, &sc->iw, &sc->temp);
				
			}
			for (pfUINT i = 10 + 4 * j; i < 11 + 4 * j; i++) {
				PfMul(sc, &sc->temp, &regID[i + 1], &sc->iw, 0);
				
				PfSub(sc, &regID[i + 1], &regID[i], &sc->temp);
				
				PfAdd(sc, &regID[i], &regID[i], &sc->temp);
				
			}
		}

		pfUINT permute2[16] = { 0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15 };
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
			temp_complex.data.c[0].data.d = pfFPinit("1.0");
			temp_complex.data.c[1].data.d = pfFPinit("0.0");
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
		for (pfUINT i = 0; i < 16; i++) {
			PfMul(sc, &sc->temp, &regID[i + 16], &sc->w, 0);
			
			PfSub(sc, &regID[i + 16], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageSize == 1) {
			temp_complex.data.c[0].data.d = pfFPinit("1.0");
			temp_complex.data.c[1].data.d = pfFPinit("0.0");
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
				temp_double.data.d = pfFPinit("0.5");
				PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
				PfSinCos(sc, &sc->w, &sc->tempFloat);
			}
		}
		for (pfUINT i = 0; i < 8; i++) {
			PfMul(sc, &sc->temp, &regID[i + 8], &sc->w, 0);
			
			PfSub(sc, &regID[i + 8], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageAngle < 0) {
			PfMov(sc, &sc->iw.data.c[0], &sc->w.data.c[1]);
			PfMovNeg(sc, &sc->iw.data.c[1], &sc->w.data.c[0]);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(w.y, -w.x);\n\n", vecType);
		}
		else {
			PfMovNeg(sc, &sc->iw.data.c[0], &sc->w.data.c[1]);
			PfMov(sc, &sc->iw.data.c[1], &sc->w.data.c[0]);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	iw = %s(-w.y, w.x);\n\n", vecType);
		}

		for (pfUINT i = 16; i < 24; i++) {
			PfMul(sc, &sc->temp, &regID[i + 8], &sc->iw, 0);
			
			PfSub(sc, &regID[i + 8], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageSize == 1) {
			temp_complex.data.c[0].data.d = pfFPinit("1.0");
			temp_complex.data.c[1].data.d = pfFPinit("0.0");
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
				temp_double.data.d = pfFPinit("0.25");
				PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
				PfSinCos(sc, &sc->w, &sc->tempFloat);
			}
		}
		for (pfUINT i = 0; i < 4; i++) {
			PfMul(sc, &sc->temp, &regID[i + 4], &sc->w, 0);
			
			PfSub(sc, &regID[i + 4], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageAngle < 0) {
			PfMov(sc, &sc->iw.data.c[0], &sc->w.data.c[1]);
			PfMovNeg(sc, &sc->iw.data.c[1], &sc->w.data.c[0]);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(w.y, -w.x);\n\n", vecType);
		}
		else {
			PfMovNeg(sc, &sc->iw.data.c[0], &sc->w.data.c[1]);
			PfMov(sc, &sc->iw.data.c[1], &sc->w.data.c[0]);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	iw = %s(-w.y, w.x);\n\n", vecType);
		}
		for (pfUINT i = 8; i < 12; i++) {
			PfMul(sc, &sc->temp, &regID[i + 4], &sc->iw, 0);
			
			PfSub(sc, &regID[i + 4], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageAngle < 0) {
			temp_complex.data.c[0].data.d = pfFPinit("0.70710678118654752440084436210485");
			temp_complex.data.c[1].data.d = pfFPinit("-0.70710678118654752440084436210485");
			PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);

		}
		else {
			temp_complex.data.c[0].data.d = pfFPinit("0.70710678118654752440084436210485");
			temp_complex.data.c[1].data.d = pfFPinit("0.70710678118654752440084436210485");
			PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
		}
		for (pfUINT i = 16; i < 20; i++) {
			PfMul(sc, &sc->temp, &regID[i + 4], &sc->iw, 0);
			
			PfSub(sc, &regID[i + 4], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageAngle < 0) {
			PfMov(sc, &sc->w.data.c[0], &sc->iw.data.c[1]);
			PfMovNeg(sc, &sc->w.data.c[1], &sc->iw.data.c[0]);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(iw.y, -iw.x);\n\n", vecType);
		}
		else {
			PfMovNeg(sc, &sc->w.data.c[0], &sc->iw.data.c[1]);
			PfMov(sc, &sc->w.data.c[1], &sc->iw.data.c[0]);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(-iw.y, iw.x);\n\n", vecType);
		}
		for (pfUINT i = 24; i < 28; i++) {
			PfMul(sc, &sc->temp, &regID[i + 4], &sc->w, 0);
			
			PfSub(sc, &regID[i + 4], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}

		if (stageSize == 1) {
			temp_complex.data.c[0].data.d = pfFPinit("1.0");
			temp_complex.data.c[1].data.d = pfFPinit("0.0");
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
				temp_double.data.d = pfFPinit("0.125");
				PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
				PfSinCos(sc, &sc->w, &sc->tempFloat);
			}
		}

		for (pfUINT i = 0; i < 2; i++) {
			PfMul(sc, &sc->temp, &regID[i + 2], &sc->w, 0);
			
			PfSub(sc, &regID[i + 2], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageAngle < 0) {
			PfMov(sc, &sc->iw.data.c[0], &sc->w.data.c[1]);
			PfMovNeg(sc, &sc->iw.data.c[1], &sc->w.data.c[0]);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(w.y, -w.x);\n\n", vecType);
		}
		else {
			PfMovNeg(sc, &sc->iw.data.c[0], &sc->w.data.c[1]);
			PfMov(sc, &sc->iw.data.c[1], &sc->w.data.c[0]);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	iw = %s(-w.y, w.x);\n\n", vecType);
		}
		for (pfUINT i = 4; i < 6; i++) {
			PfMul(sc, &sc->temp, &regID[i + 2], &sc->iw, 0);
			
			PfSub(sc, &regID[i + 2], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}


		if (stageAngle < 0) {
			temp_complex.data.c[0].data.d = pfFPinit("0.70710678118654752440084436210485");
			temp_complex.data.c[1].data.d = pfFPinit("-0.70710678118654752440084436210485");
			PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);

		}
		else {
			temp_complex.data.c[0].data.d = pfFPinit("0.70710678118654752440084436210485");
			temp_complex.data.c[1].data.d = pfFPinit("0.70710678118654752440084436210485");
			PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
		}
		for (pfUINT i = 8; i < 10; i++) {
			PfMul(sc, &sc->temp, &regID[i + 2], &sc->iw, 0);
			
			PfSub(sc, &regID[i + 2], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageAngle < 0) {
			PfMov(sc, &sc->temp.data.c[0], &sc->iw.data.c[1]);
			PfMovNeg(sc, &sc->temp.data.c[1], &sc->iw.data.c[0]);
			
			PfMov(sc, &sc->iw, &sc->temp);
			
		}
		else {
			PfMovNeg(sc, &sc->temp.data.c[0], &sc->iw.data.c[1]);
			PfMov(sc, &sc->temp.data.c[1], &sc->iw.data.c[0]);
			
			PfMov(sc, &sc->iw, &sc->temp);
			
		}
		for (pfUINT i = 12; i < 14; i++) {
			PfMul(sc, &sc->temp, &regID[i + 2], &sc->iw, 0);
			
			PfSub(sc, &regID[i + 2], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}


		for (pfUINT j = 0; j < 2; j++) {
			if (stageAngle < 0) {
				temp_complex.data.c[0].data.d = pfcos((2 * j + 1) * sc->double_PI / 8);
				temp_complex.data.c[1].data.d = -pfsin((2 * j + 1) * sc->double_PI / 8);
				PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
			}
			else {
				temp_complex.data.c[0].data.d = pfcos((2 * j + 1) * sc->double_PI / 8);
				temp_complex.data.c[1].data.d = pfsin((2 * j + 1) * sc->double_PI / 8);
				PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
			}
			for (pfUINT i = 16 + 8 * j; i < 18 + 8 * j; i++) {
				PfMul(sc, &sc->temp, &regID[i + 2], &sc->iw, 0);
				
				PfSub(sc, &regID[i + 2], &regID[i], &sc->temp);
				
				PfAdd(sc, &regID[i], &regID[i], &sc->temp);
				
			}
			if (stageAngle < 0) {
				PfMov(sc, &sc->temp.data.c[0], &sc->iw.data.c[1]);
				PfMovNeg(sc, &sc->temp.data.c[1], &sc->iw.data.c[0]);
			
				PfMov(sc, &sc->iw, &sc->temp);
			}
			else {
				PfMovNeg(sc, &sc->temp.data.c[0], &sc->iw.data.c[1]);
				PfMov(sc, &sc->temp.data.c[1], &sc->iw.data.c[0]);
			
				PfMov(sc, &sc->iw, &sc->temp);
				
			}
			for (pfUINT i = 20 + 8 * j; i < 22 + 8 * j; i++) {
				PfMul(sc, &sc->temp, &regID[i + 2], &sc->iw, 0);
				
				PfSub(sc, &regID[i + 2], &regID[i], &sc->temp);
				
				PfAdd(sc, &regID[i], &regID[i], &sc->temp);
				
			}
		}

		if (stageSize == 1) {
			temp_complex.data.c[0].data.d = pfFPinit("1.0");
			temp_complex.data.c[1].data.d = pfFPinit("0.0");
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
				temp_double.data.d = pfFPinit("0.0625");
				PfMul(sc, &sc->tempFloat, &sc->angle, &temp_double, 0);
				PfSinCos(sc, &sc->w, &sc->tempFloat);
			}
		}

		for (pfUINT i = 0; i < 1; i++) {
			PfMul(sc, &sc->temp, &regID[i + 1], &sc->w, 0);
			
			PfSub(sc, &regID[i + 1], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageAngle < 0) {
			PfMov(sc, &sc->iw.data.c[0], &sc->w.data.c[1]);
			PfMovNeg(sc, &sc->iw.data.c[1], &sc->w.data.c[0]);
			
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	w = %s(w.y, -w.x);\n\n", vecType);
		}
		else {
			PfMovNeg(sc, &sc->iw.data.c[0], &sc->w.data.c[1]);
			PfMov(sc, &sc->iw.data.c[1], &sc->w.data.c[0]);
			
			//&sc->tempLen = sprintf(&sc->tempStr, "	iw = %s(-w.y, w.x);\n\n", vecType);
		}
		for (pfUINT i = 2; i < 3; i++) {
			PfMul(sc, &sc->temp, &regID[i + 1], &sc->iw, 0);
			
			PfSub(sc, &regID[i + 1], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}


		if (stageAngle < 0) {
			temp_complex.data.c[0].data.d = pfFPinit("0.70710678118654752440084436210485");
			temp_complex.data.c[1].data.d = pfFPinit("-0.70710678118654752440084436210485");
			PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);

		}
		else {
			temp_complex.data.c[0].data.d = pfFPinit("0.70710678118654752440084436210485");
			temp_complex.data.c[1].data.d = pfFPinit("0.70710678118654752440084436210485");
			PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
		}
		for (pfUINT i = 4; i < 5; i++) {
			PfMul(sc, &sc->temp, &regID[i + 1], &sc->iw, 0);
			
			PfSub(sc, &regID[i + 1], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}
		if (stageAngle < 0) {
			PfMov(sc, &sc->temp.data.c[0], &sc->iw.data.c[1]);
			PfMovNeg(sc, &sc->temp.data.c[1], &sc->iw.data.c[0]);
			
			PfMov(sc, &sc->iw, &sc->temp);
			
		}
		else {
			PfMovNeg(sc, &sc->temp.data.c[0], &sc->iw.data.c[1]);
			PfMov(sc, &sc->temp.data.c[1], &sc->iw.data.c[0]);
			
			
			PfMov(sc, &sc->iw, &sc->temp);
			
		}
		for (pfUINT i = 6; i < 7; i++) {
			PfMul(sc, &sc->temp, &regID[i + 1], &sc->iw, 0);
			
			PfSub(sc, &regID[i + 1], &regID[i], &sc->temp);
			
			PfAdd(sc, &regID[i], &regID[i], &sc->temp);
			
		}


		for (pfUINT j = 0; j < 2; j++) {
			if (stageAngle < 0) {
				temp_complex.data.c[0].data.d = pfcos((2 * j + 1) * sc->double_PI / 8);
				temp_complex.data.c[1].data.d = -pfsin((2 * j + 1) * sc->double_PI / 8);
				PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
			}
			else {
				temp_complex.data.c[0].data.d = pfcos((2 * j + 1) * sc->double_PI / 8);
				temp_complex.data.c[1].data.d = pfsin((2 * j + 1) * sc->double_PI / 8);
				PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
			}
			for (pfUINT i = 8 + 4 * j; i < 9 + 4 * j; i++) {
				PfMul(sc, &sc->temp, &regID[i + 1], &sc->iw, 0);
				
				PfSub(sc, &regID[i + 1], &regID[i], &sc->temp);
				
				PfAdd(sc, &regID[i], &regID[i], &sc->temp);
				
			}
			if (stageAngle < 0) {
				PfMov(sc, &sc->temp.data.c[0], &sc->iw.data.c[1]);
				PfMovNeg(sc, &sc->temp.data.c[1], &sc->iw.data.c[0]);
				
				PfMov(sc, &sc->iw, &sc->temp);
				
			}
			else {
				PfMovNeg(sc, &sc->temp.data.c[0], &sc->iw.data.c[1]);
				PfMov(sc, &sc->temp.data.c[1], &sc->iw.data.c[0]);
				
				PfMov(sc, &sc->iw, &sc->temp);
				
			}
			for (pfUINT i = 10 + 4 * j; i < 11 + 4 * j; i++) {
				PfMul(sc, &sc->temp, &regID[i + 1], &sc->iw, 0);
				
				PfSub(sc, &regID[i + 1], &regID[i], &sc->temp);
				
				PfAdd(sc, &regID[i], &regID[i], &sc->temp);
				
			}
		}

		for (pfUINT j = 0; j < 4; j++) {
			if ((j == 1) || (j == 2)) {
				if (stageAngle < 0) {
					temp_complex.data.c[0].data.d = pfcos((7 - 2 * j) * sc->double_PI / 16);
					temp_complex.data.c[1].data.d = -pfsin((7 - 2 * j) * sc->double_PI / 16);
					PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
				}
				else {
					temp_complex.data.c[0].data.d = pfcos((7 - 2 * j) * sc->double_PI / 16);
					temp_complex.data.c[1].data.d = pfsin((7 - 2 * j) * sc->double_PI / 16);
					PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
				}
			}
			else {
				if (stageAngle < 0) {
					temp_complex.data.c[0].data.d = pfcos((2 * j + 1) * sc->double_PI / 16);
					temp_complex.data.c[1].data.d = -pfsin((2 * j + 1) * sc->double_PI / 16);
					PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
				}
				else {
					temp_complex.data.c[0].data.d = pfcos((2 * j + 1) * sc->double_PI / 16);
					temp_complex.data.c[1].data.d = pfsin((2 * j + 1) * sc->double_PI / 16);
					PfMul(sc, &sc->iw, &sc->w, &temp_complex, 0);
				}
			}
			for (pfUINT i = 16 + 4 * j; i < 17 + 4 * j; i++) {
				PfMul(sc, &sc->temp, &regID[i + 1], &sc->iw, 0);
				
				PfSub(sc, &regID[i + 1], &regID[i], &sc->temp);
				
				PfAdd(sc, &regID[i], &regID[i], &sc->temp);
				
			}
			if (stageAngle < 0) {
				PfMov(sc, &sc->temp.data.c[0], &sc->iw.data.c[1]);
				PfMovNeg(sc, &sc->temp.data.c[1], &sc->iw.data.c[0]);
				
				PfMov(sc, &sc->iw, &sc->temp);
				
			}
			else {
				PfMovNeg(sc, &sc->temp.data.c[0], &sc->iw.data.c[1]);
				PfMov(sc, &sc->temp.data.c[1], &sc->iw.data.c[0]);
				
				PfMov(sc, &sc->iw, &sc->temp);
				
			}
			for (pfUINT i = 18 + 4 * j; i < 19 + 4 * j; i++) {
				PfMul(sc, &sc->temp, &regID[i + 1], &sc->iw, 0);
				
				PfSub(sc, &regID[i + 1], &regID[i], &sc->temp);
				
				PfAdd(sc, &regID[i], &regID[i], &sc->temp);
				
			}
		}

		pfUINT permute2[32] = { 0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30,1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31 };
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
	PfDeallocateContainer(sc, &temp_complex);
	return;
}

#endif
