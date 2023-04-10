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
#ifndef VKFFT_MATHUTILS_H
#define VKFFT_MATHUTILS_H
#include "vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT_CodeGen/vkFFT_StringManagement/vkFFT_StringManager.h"

//register manipulation functions: mov, add, sub, etc.
static inline void VkCopyContainer(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in) {
	if (sc->res != VKFFT_SUCCESS) return;

	if (out->type > 100) {
		if (in->type > 100) {
			if (out->type == in->type) {
				int len = 0;
				len = sprintf(out->data.s, "%s", in->data.s);
				if (len > out->size) sc->res = VKFFT_ERROR_MATH_FAILED;
				return;
			}
		}
		else {
		}
	}
	else {
		if (in->type > 100) {
		}
		else {
			if (out->type == in->type) {
				switch (out->type % 10) {
				case 1:
					out->data.i = in->data.i;
					return;
				case 2:
					out->data.d = in->data.d;
					return;
				case 3:
					out->data.c[0] = in->data.c[0];
					out->data.c[1] = in->data.c[1];
					return;
				}
			}
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}
static inline void VkAllocateContainerFlexible(VkFFTSpecializationConstantsLayout* sc, VkContainer* container, int size) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (container->size != 0) return;

	container->type = 100;
	container->data.s = (char*)calloc(size, sizeof(char));
	container->size = size;

	if (container->data.s == 0) sc->res = VKFFT_ERROR_MALLOC_FAILED;
	return;
}
static inline void VkDeallocateContainer(VkFFTSpecializationConstantsLayout* sc, VkContainer* container) {
	if (container->type > 0) {
		free(container->data.s);
		container->data.s = 0;
		container->type = 0;
	}
}

static inline void VkGetTypeFromCode(VkFFTSpecializationConstantsLayout* sc, int code, VkContainer** type) {
	if (sc->res != VKFFT_SUCCESS) return;
	switch (code % 10) {
	case 1:
		switch ((code % 100) / 10) {
		case 0:
			type[0] = &sc->uintDef;
			return;
		case 1:
			type[0] = &sc->intDef;
			return;
		case 2:
			type[0] = &sc->uint64Def;
			return;
		case 3:
			type[0] = &sc->int64Def;
			return;
		}
		break;
	case 2:
		switch ((code % 100) / 10) {
		case 0:
			type[0] = &sc->halfDef;
			return;
		case 1:
			type[0] = &sc->floatDef;
			return;
		case 2:
			type[0] = &sc->doubleDef;
			return;
		}
		break;
	case 3:
		switch ((code % 100) / 10) {
		case 0:
			type[0] = &sc->half2Def;
			return;
		case 1:
			type[0] = &sc->float2Def;
			return;
		case 2:
			type[0] = &sc->double2Def;
			return;
		}
		break;
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}
static inline void VkAppendNumberLiteral(VkFFTSpecializationConstantsLayout* sc, VkContainer* number) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (((number->type % 10) == 2) || ((number->type % 10) == 3)) {
		switch ((number->type % 100) / 10) {
		case 0:
			sc->tempLen = sprintf(sc->tempStr, "%s", sc->halfLiteral.data.s);
			VkAppendLine(sc);
			return;
		case 1:
			sc->tempLen = sprintf(sc->tempStr, "%s", sc->floatLiteral.data.s);
			VkAppendLine(sc);
			return;
		case 2:
			sc->tempLen = sprintf(sc->tempStr, "%s", sc->doubleLiteral.data.s);
			VkAppendLine(sc);
			return;
		}
	}
	return;
}
static inline void VkAppendConversionStart(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (((out->type % 100) / 10) == ((in->type % 100) / 10))
		return;
	if ((out->type < 100) || (in->type < 100))
		return;
	switch (in->type % 10) {
	case 1:
		return;
	case 2:
		switch ((out->type % 100) / 10) {
		case 0:
#if((VKFFT_BACKEND==0)||(VKFFT_BACKEND==5))
			sc->tempLen = sprintf(sc->tempStr, "half(");
			VkAppendLine(sc);
#elif((VKFFT_BACKEND==1)||(VKFFT_BACKEND==2)||(VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
			sc->tempLen = sprintf(sc->tempStr, "(half)");
			VkAppendLine(sc);
#endif
			return;
		case 1:
#if((VKFFT_BACKEND==0)||(VKFFT_BACKEND==5))
			sc->tempLen = sprintf(sc->tempStr, "float(");
			VkAppendLine(sc);
#elif((VKFFT_BACKEND==1)||(VKFFT_BACKEND==2)||(VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
			sc->tempLen = sprintf(sc->tempStr, "(float)");
			VkAppendLine(sc);
#endif
			return;
		case 2:
#if((VKFFT_BACKEND==0)||(VKFFT_BACKEND==5))
			sc->tempLen = sprintf(sc->tempStr, "double(");
			VkAppendLine(sc);
#elif((VKFFT_BACKEND==1)||(VKFFT_BACKEND==2)||(VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
			sc->tempLen = sprintf(sc->tempStr, "(double)");
			VkAppendLine(sc);
#endif
			return;
		}
	case 3:
		switch ((out->type % 100) / 10) {
		case 0:
#if(VKFFT_BACKEND==0)
			sc->tempLen = sprintf(sc->tempStr, "f16vec2(");
			VkAppendLine(sc);
#elif((VKFFT_BACKEND==1)||(VKFFT_BACKEND==2)||(VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5))
			sc->tempLen = sprintf(sc->tempStr, "conv_half2(");
			VkAppendLine(sc);
#endif
			return;
		case 1:
#if(VKFFT_BACKEND==0)
			sc->tempLen = sprintf(sc->tempStr, "vec2(");
			VkAppendLine(sc);
#elif((VKFFT_BACKEND==1)||(VKFFT_BACKEND==2)||(VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5))
			sc->tempLen = sprintf(sc->tempStr, "conv_float2(");
			VkAppendLine(sc);
#endif
			return;
		case 2:
#if(VKFFT_BACKEND==0)
			sc->tempLen = sprintf(sc->tempStr, "dvec2(");
			VkAppendLine(sc);
#elif((VKFFT_BACKEND==1)||(VKFFT_BACKEND==2)||(VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5))
			sc->tempLen = sprintf(sc->tempStr, "conv_double2(");
			VkAppendLine(sc);
#endif
			return;
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}
static inline void VkAppendConversionEnd(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (((out->type % 100) / 10) == ((in->type % 100) / 10))
		return;
	if ((out->type < 100) || (in->type < 100))
		return;
	switch (in->type % 10) {
	case 1:
		return;
	case 2:
		switch ((out->type % 100) / 10) {
		case 0:
#if((VKFFT_BACKEND==0)||(VKFFT_BACKEND==5))
			sc->tempLen = sprintf(sc->tempStr, ")");
			VkAppendLine(sc);
#elif((VKFFT_BACKEND==1)||(VKFFT_BACKEND==2)||(VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
#endif
			return;
		case 1:
#if((VKFFT_BACKEND==0)||(VKFFT_BACKEND==5))
			sc->tempLen = sprintf(sc->tempStr, ")");
			VkAppendLine(sc);
#elif((VKFFT_BACKEND==1)||(VKFFT_BACKEND==2)||(VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
#endif
			return;
		case 2:
#if((VKFFT_BACKEND==0)||(VKFFT_BACKEND==5))
			sc->tempLen = sprintf(sc->tempStr, ")");
			VkAppendLine(sc);
#elif((VKFFT_BACKEND==1)||(VKFFT_BACKEND==2)||(VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
#endif
			return;
		}
	case 3:
		switch ((out->type % 100) / 10) {
		case 0:
#if(VKFFT_BACKEND==0)
			sc->tempLen = sprintf(sc->tempStr, ")");
			VkAppendLine(sc);
#elif((VKFFT_BACKEND==1)||(VKFFT_BACKEND==2)||(VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5))
			sc->tempLen = sprintf(sc->tempStr, ")");
			VkAppendLine(sc);
#endif
			return;
		case 1:
#if(VKFFT_BACKEND==0)
			sc->tempLen = sprintf(sc->tempStr, ")");
			VkAppendLine(sc);
#elif((VKFFT_BACKEND==1)||(VKFFT_BACKEND==2)||(VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5))
			sc->tempLen = sprintf(sc->tempStr, ")");
			VkAppendLine(sc);
#endif
			return;
		case 2:
#if(VKFFT_BACKEND==0)
			sc->tempLen = sprintf(sc->tempStr, ")");
			VkAppendLine(sc);
#elif((VKFFT_BACKEND==1)||(VKFFT_BACKEND==2)||(VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5))
			sc->tempLen = sprintf(sc->tempStr, ")");
			VkAppendLine(sc);
#endif
			return;
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}

static inline void VkDefine(VkFFTSpecializationConstantsLayout* sc, VkContainer* name) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (name->type > 100) {
		switch (name->type % 10) {
		case 1:
			switch ((name->type % 100) / 10) {
			case 0:
				sc->tempLen = sprintf(sc->tempStr, "%s %s;\n", sc->uintDef.data.s, name->data.s);
				VkAppendLine(sc);
				return;
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "%s %s;\n", sc->intDef.data.s, name->data.s);
				VkAppendLine(sc);
				return;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "%s %s;\n", sc->uint64Def.data.s, name->data.s);
				VkAppendLine(sc);
				return;
			case 3:
				sc->tempLen = sprintf(sc->tempStr, "%s %s;\n", sc->int64Def.data.s, name->data.s);
				VkAppendLine(sc);
				return;
			}
			break;
		case 2:
			switch ((name->type % 100) / 10) {
			case 0:
				sc->tempLen = sprintf(sc->tempStr, "%s %s;\n", sc->halfDef.data.s, name->data.s);
				VkAppendLine(sc);
				return;
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "%s %s;\n", sc->floatDef.data.s, name->data.s);
				VkAppendLine(sc);
				return;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "%s %s;\n", sc->doubleDef.data.s, name->data.s);
				VkAppendLine(sc);
				return;
			}
			break;
		case 3:
			switch ((name->type % 100) / 10) {
			case 0:
				sc->tempLen = sprintf(sc->tempStr, "%s %s;\n", sc->half2Def.data.s, name->data.s);
				VkAppendLine(sc);
				return;
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "%s %s;\n", sc->float2Def.data.s, name->data.s);
				VkAppendLine(sc);
				return;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "%s %s;\n", sc->double2Def.data.s, name->data.s);
				VkAppendLine(sc);
				return;
			}
			break;
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}
static inline void VkDefineConstant(VkFFTSpecializationConstantsLayout* sc, VkContainer* name, VkContainer* value) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (name->type > 100) {
		switch (name->type % 10) {
		case 1:
			switch ((name->type % 100) / 10) {
			case 0:
				sc->tempLen = sprintf(sc->tempStr, "%s %s %s", sc->constDef.data.s, sc->uintDef.data.s, name->data.s);
				VkAppendLine(sc);
				break;
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "%s %s %s", sc->constDef.data.s, sc->intDef.data.s, name->data.s);
				VkAppendLine(sc);
				break;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "%s %s %s", sc->constDef.data.s, sc->uint64Def.data.s, name->data.s);
				VkAppendLine(sc);
				break;
			case 3:
				sc->tempLen = sprintf(sc->tempStr, "%s %s %s", sc->constDef.data.s, sc->int64Def.data.s, name->data.s);
				VkAppendLine(sc);
				break;
			}
			break;
		case 2:
			switch ((name->type % 100) / 10) {
			case 0:
				sc->tempLen = sprintf(sc->tempStr, "%s %s %s", sc->constDef.data.s, sc->halfDef.data.s, name->data.s);
				VkAppendLine(sc);
				break;
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "%s %s %s", sc->constDef.data.s, sc->floatDef.data.s, name->data.s);
				VkAppendLine(sc);
				break;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "%s %s %s", sc->constDef.data.s, sc->doubleDef.data.s, name->data.s);
				VkAppendLine(sc);
				break;
			}
			break;
		case 3:
			switch ((name->type % 100) / 10) {
			case 0:
				sc->tempLen = sprintf(sc->tempStr, "%s %s %s", sc->constDef.data.s, sc->half2Def.data.s, name->data.s);
				VkAppendLine(sc);
				break;
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "%s %s %s", sc->constDef.data.s, sc->float2Def.data.s, name->data.s);
				VkAppendLine(sc);
				break;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "%s %s %s", sc->constDef.data.s, sc->double2Def.data.s, name->data.s);
				VkAppendLine(sc);
				break;
			}
			break;
		}
		if (value->type < 100) {
			switch (value->type % 10) {
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", value->data.i);
				VkAppendLine(sc);
				break;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "%.17Le", value->data.d);
				VkAppendLine(sc);
				break;
			case 3:
				//fix
				sc->res = VKFFT_ERROR_MATH_FAILED;
				break;
			}
			VkAppendNumberLiteral(sc, name);
			return;
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}

static inline void VkSetToZero(VkFFTSpecializationConstantsLayout* sc, VkContainer* out) {
	if (sc->res != VKFFT_SUCCESS) return;
	//out
	if (out->type > 100) {
		switch (out->type % 10) {
		case 1: case 2:
			sc->tempLen = sprintf(sc->tempStr, "%s", out->data.s);
			VkAppendLine(sc);
			break;
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.x", out->data.s);
			VkAppendLine(sc);
			break;
		}
		sc->tempLen = sprintf(sc->tempStr, " = ");
		VkAppendLine(sc);
		switch (out->type % 10) {
		case 1:
			sc->tempLen = sprintf(sc->tempStr, "0");
			VkAppendLine(sc);
			break;
		case 2: case 3:
			sc->tempLen = sprintf(sc->tempStr, "0.0");
			VkAppendLine(sc);
			break;
		}
		VkAppendNumberLiteral(sc, out);
		sc->tempLen = sprintf(sc->tempStr, ";\n");
		VkAppendLine(sc);

		switch (out->type % 10) {
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.y", out->data.s);
			VkAppendLine(sc);
			sc->tempLen = sprintf(sc->tempStr, " = ");
			VkAppendLine(sc);
			sc->tempLen = sprintf(sc->tempStr, "0.0");
			VkAppendLine(sc);
			VkAppendNumberLiteral(sc, out);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
			break;
		}
		return;
	}
	else {
		switch (out->type % 10) {
		case 1: 
			out->data.i = 0;
			return;
		case 2:
			out->data.d = 0;
			return;
		case 3:
			out->data.c[0] = 0;
			out->data.c[1] = 0;
			return;
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}
static inline void VkSetToZeroShared(VkFFTSpecializationConstantsLayout* sc, VkContainer* sdataID) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (sdataID->type > 100) {
		switch (sdataID->type % 10) {
		case 1: 
			sc->tempLen = sprintf(sc->tempStr, "\
sdata[%s].x = 0;\n\
sdata[%s].y = 0;\n", sdataID->data.s, sdataID->data.s);
			VkAppendLine(sc);
			return;
		}
	}
	else {
		switch (sdataID->type % 10) {
		case 1:
			sc->tempLen = sprintf(sc->tempStr, "\
sdata[%" PRIi64 "].x = 0;\n\
sdata[%" PRIi64 "].y = 0;\n", sdataID->data.i, sdataID->data.i);
			VkAppendLine(sc);
			return;
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}


static inline void VkMov(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
		if ((out->type > 100) && (in->type > 100) && ((out->type % 10) == (in->type % 10))) {
			//packed instructions workaround if all values are in registers
			sc->tempLen = sprintf(sc->tempStr, "%s", out->data.s);
			VkAppendLine(sc);
			sc->tempLen = sprintf(sc->tempStr, " = ");
			VkAppendLine(sc);
			VkAppendConversionStart(sc, out, in);
			sc->tempLen = sprintf(sc->tempStr, "%s", in->data.s);
			VkAppendLine(sc);
			VkAppendConversionEnd(sc, out, in);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
			return;
		}
		switch (out->type % 10) {
		case 1: case 2:
			sc->tempLen = sprintf(sc->tempStr, "%s", out->data.s);
			VkAppendLine(sc);
			break;
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.x", out->data.s);
			VkAppendLine(sc);
			break;
		}
		sc->tempLen = sprintf(sc->tempStr, " = ");
		VkAppendLine(sc);
		VkAppendConversionStart(sc, out, in);
		if (in->type > 100) {
			switch (in->type % 10) {
			case 1: case 2:
				sc->tempLen = sprintf(sc->tempStr, "%s", in->data.s);
				VkAppendLine(sc);
				break;
			case 3:
				sc->tempLen = sprintf(sc->tempStr, "%s.x", in->data.s);
				VkAppendLine(sc);
				break;
			}
		}
		else {
			switch (in->type % 10) {
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in->data.i);
				VkAppendLine(sc);
				break;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "%.17Le", in->data.d);
				VkAppendLine(sc);
				break;
			case 3:
				sc->tempLen = sprintf(sc->tempStr, "%.17Le", in->data.c[0]);
				VkAppendLine(sc);
				break;
			}
			VkAppendNumberLiteral(sc, out);
		}
		VkAppendConversionEnd(sc, out, in);
		sc->tempLen = sprintf(sc->tempStr, ";\n");
		VkAppendLine(sc);

		switch (out->type % 10) {
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.y", out->data.s);
			VkAppendLine(sc);
			sc->tempLen = sprintf(sc->tempStr, " = ");
			VkAppendLine(sc);
			VkAppendConversionStart(sc, out, in);
			if (in->type > 100) {
				switch (in->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.y", in->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in->data.c[1]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
			break;
		}
		return;
	}
	else {
		if (in->type > 100) {
		}
		else {
			switch (out->type % 10) {
			case 1:
				switch (in->type % 10) {
				case 1:
					out->data.i = in->data.i;
					return;
				case 2:
					out->data.i = (int64_t)in->data.d;
					return;
				}
				return;
			case 2:
				switch (in->type % 10) {
				case 1:
					out->data.d = (double)in->data.i;
					return;
				case 2:
					out->data.d = in->data.d;
					return;
				}
				return;
			case 3:
				out->data.c[0] = in->data.c[0];
				out->data.c[1] = in->data.c[1];
				return;
			}
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}

static inline void VkMov_x(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
		//in_1 has to be same type as out
		switch (out->type % 10) {
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.x", out->data.s);
			VkAppendLine(sc);
			break;
		default:
			sc->res = VKFFT_ERROR_MATH_FAILED;
			return;
		}
		sc->tempLen = sprintf(sc->tempStr, " = ");
		VkAppendLine(sc);
		VkAppendConversionStart(sc, out, in);
		if (in->type > 100) {
			switch (in->type % 10) {
			case 1: case 2:
				sc->tempLen = sprintf(sc->tempStr, "%s", in->data.s);
				VkAppendLine(sc);
				break;
			case 3:
				sc->tempLen = sprintf(sc->tempStr, "%s.x", in->data.s);
				VkAppendLine(sc);
				break;
			}
		}
		else {
			switch (in->type % 10) {
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in->data.i);
				VkAppendLine(sc);
				break;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "%.17Le", in->data.d);
				VkAppendLine(sc);
				break;
			case 3:
				sc->tempLen = sprintf(sc->tempStr, "%.17Le", in->data.c[0]);
				VkAppendLine(sc);
				break;
			}
			VkAppendNumberLiteral(sc, out);
		}
		VkAppendConversionEnd(sc, out, in);
		sc->tempLen = sprintf(sc->tempStr, ";\n");
		VkAppendLine(sc);
		return;
	}
	else {
		switch (out->type % 10) {
		case 3:
			if (in->type > 100) {
			}
			else {
				switch (in->type % 10) {
				case 1:
					out->data.c[0] = in->data.i;
					return;
				case 2:
					out->data.c[0] = in->data.d;
					return;
				case 3:
					out->data.c[0] = in->data.c[0];
					return;
				}
			}
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}
static inline void VkMov_y(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
		//in_1 has to be same type as out
		switch (out->type % 10) {
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.y", out->data.s);
			VkAppendLine(sc);
			break;
		default:
			sc->res = VKFFT_ERROR_MATH_FAILED;
			return;
		}
		sc->tempLen = sprintf(sc->tempStr, " = ");
		VkAppendLine(sc);
		VkAppendConversionStart(sc, out, in);
		if (in->type > 100) {
			switch (in->type % 10) {
			case 1: case 2:
				sc->tempLen = sprintf(sc->tempStr, "%s", in->data.s);
				VkAppendLine(sc);
				break;
			case 3:
				sc->tempLen = sprintf(sc->tempStr, "%s.y", in->data.s);
				VkAppendLine(sc);
				break;
			}
		}
		else {
			switch (in->type % 10) {
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in->data.i);
				VkAppendLine(sc);
				break;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "%.17Le", in->data.d);
				VkAppendLine(sc);
				break;
			case 3:
				sc->tempLen = sprintf(sc->tempStr, "%.17Le", in->data.c[1]);
				VkAppendLine(sc);
				break;
			}
			VkAppendNumberLiteral(sc, out);
		}
		VkAppendConversionEnd(sc, out, in);
		sc->tempLen = sprintf(sc->tempStr, ";\n");
		VkAppendLine(sc);
		return;
	}
	else {
		switch (out->type % 10) {
		case 3:
			if (in->type > 100) {
			}
			else {
				switch (in->type % 10) {
				case 1:
					out->data.c[1] = in->data.i;
					return;
				case 2:
					out->data.c[1] = in->data.d;
					return;
				case 3:
					out->data.c[1] = in->data.c[1];
					return;
				}
			}
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}

static inline void VkMov_x_Neg_x(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
		//in_1 has to be same type as out
		switch (out->type % 10) {
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.x", out->data.s);
			VkAppendLine(sc);
			break;
		default:
			sc->res = VKFFT_ERROR_MATH_FAILED;
			return;
		}
		sc->tempLen = sprintf(sc->tempStr, " = ");
		VkAppendLine(sc);
		VkAppendConversionStart(sc, out, in);
		if (in->type > 100) {
			switch (in->type % 10) {
			case 1: case 2:
				sc->tempLen = sprintf(sc->tempStr, "-%s", in->data.s);
				VkAppendLine(sc);
				break;
			case 3:
				sc->tempLen = sprintf(sc->tempStr, "-%s.x", in->data.s);
				VkAppendLine(sc);
				break;
			}
		}
		else {
			switch (in->type % 10) {
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", -in->data.i);
				VkAppendLine(sc);
				break;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "%.17Le", -in->data.d);
				VkAppendLine(sc);
				break;
			case 3:
				sc->tempLen = sprintf(sc->tempStr, "%.17Le", -in->data.c[0]);
				VkAppendLine(sc);
				break;
			}
			VkAppendNumberLiteral(sc, out);
		}
		VkAppendConversionEnd(sc, out, in);
		sc->tempLen = sprintf(sc->tempStr, ";\n");
		VkAppendLine(sc);
		return;
	}
	else {
		switch (out->type % 10) {
		case 3:
			if (in->type > 100) {
			}
			else {
				switch (in->type % 10) {
				case 1:
					out->data.c[0] = -in->data.i;
					return;
				case 2:
					out->data.c[0] = -in->data.d;
					return;
				case 3:
					out->data.c[0] = -in->data.c[0];
					return;
				}
			}
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}
static inline void VkMov_y_Neg_y(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
		//in_1 has to be same type as out
		switch (out->type % 10) {
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.y", out->data.s);
			VkAppendLine(sc);
			break;
		default:
			sc->res = VKFFT_ERROR_MATH_FAILED;
			return;
		}
		sc->tempLen = sprintf(sc->tempStr, " = ");
		VkAppendLine(sc);
		VkAppendConversionStart(sc, out, in);
		if (in->type > 100) {
			switch (in->type % 10) {
			case 1: case 2:
				sc->tempLen = sprintf(sc->tempStr, "-%s", in->data.s);
				VkAppendLine(sc);
				break;
			case 3:
				sc->tempLen = sprintf(sc->tempStr, "-%s.y", in->data.s);
				VkAppendLine(sc);
				break;
			}
		}
		else {
			switch (in->type % 10) {
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", -in->data.i);
				VkAppendLine(sc);
				break;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "%.17Le", -in->data.d);
				VkAppendLine(sc);
				break;
			case 3:
				sc->tempLen = sprintf(sc->tempStr, "%.17Le", -in->data.c[1]);
				VkAppendLine(sc);
				break;
			}
			VkAppendNumberLiteral(sc, out);
		}
		VkAppendConversionEnd(sc, out, in);
		sc->tempLen = sprintf(sc->tempStr, ";\n");
		VkAppendLine(sc);
		return;
	}
	else {
		switch (out->type % 10) {
		case 3:
			if (in->type > 100) {
			}
			else {
				switch (in->type % 10) {
				case 1:
					out->data.c[1] = -in->data.i;
					return;
				case 2:
					out->data.c[1] = -in->data.d;
					return;
				case 3:
					out->data.c[1] = -in->data.c[1];
					return;
				}
			}
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}

static inline void VkMov_x_y(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
		//in_1 has to be same type as out
		switch (out->type % 10) {
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.x", out->data.s);
			VkAppendLine(sc);
			break;
		default:
			sc->res = VKFFT_ERROR_MATH_FAILED;
			return;
		}
		sc->tempLen = sprintf(sc->tempStr, " = ");
		VkAppendLine(sc);
		VkAppendConversionStart(sc, out, in);
		if (in->type > 100) {
			switch (in->type % 10) {
			case 3:
				sc->tempLen = sprintf(sc->tempStr, "%s.y", in->data.s);
				VkAppendLine(sc);
				break;
			default:
				sc->res = VKFFT_ERROR_MATH_FAILED;
				return;
			}
		}
		else {
			switch (in->type % 10) {
			case 3:
				sc->tempLen = sprintf(sc->tempStr, "%.17Le", in->data.c[1]);
				VkAppendLine(sc);
				break;
			default:
				sc->res = VKFFT_ERROR_MATH_FAILED;
				return;
			}
			VkAppendNumberLiteral(sc, out);
		}
		VkAppendConversionEnd(sc, out, in);
		sc->tempLen = sprintf(sc->tempStr, ";\n");
		VkAppendLine(sc);
		return;
	}
	else {
		switch (out->type % 10) {
		case 3:
			if (in->type > 100) {
			}
			else {
				switch (in->type % 10) {
				case 3:
					out->data.c[0] = in->data.c[1];
					return;
				}
			}
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}
static inline void VkMov_x_Neg_y(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
		//in_1 has to be same type as out
		switch (out->type % 10) {
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.x", out->data.s);
			VkAppendLine(sc);
			break;
		default:
			sc->res = VKFFT_ERROR_MATH_FAILED;
			return;
		}
		sc->tempLen = sprintf(sc->tempStr, " = ");
		VkAppendLine(sc);
		VkAppendConversionStart(sc, out, in);
		if (in->type > 100) {
			switch (in->type % 10) {
			case 3:
				sc->tempLen = sprintf(sc->tempStr, "-%s.y", in->data.s);
				VkAppendLine(sc);
				break;
			default:
				sc->res = VKFFT_ERROR_MATH_FAILED;
				return;
			}
		}
		else {
			switch (in->type % 10) {
			case 3:
				sc->tempLen = sprintf(sc->tempStr, "%.17Le", -in->data.c[1]);
				VkAppendLine(sc);
				break;
			default:
				sc->res = VKFFT_ERROR_MATH_FAILED;
				return;
			}
			VkAppendNumberLiteral(sc, out);
		}
		VkAppendConversionEnd(sc, out, in);
		sc->tempLen = sprintf(sc->tempStr, ";\n");
		VkAppendLine(sc);
		return;
	}
	else {
		switch (out->type % 10) {
		case 3:
			if (in->type > 100) {
			}
			else {
				switch (in->type % 10) {
				case 3:
					out->data.c[0] = -in->data.c[1];
					return;
				}
			}
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}

static inline void VkMov_y_x(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
		//in_1 has to be same type as out
		switch (out->type % 10) {
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.y", out->data.s);
			VkAppendLine(sc);
			break;
		default:
			sc->res = VKFFT_ERROR_MATH_FAILED;
			return;
		}
		sc->tempLen = sprintf(sc->tempStr, " = ");
		VkAppendLine(sc);
		VkAppendConversionStart(sc, out, in);
		if (in->type > 100) {
			switch (in->type % 10) {
			case 3:
				sc->tempLen = sprintf(sc->tempStr, "%s.x", in->data.s);
				VkAppendLine(sc);
				break;
			default:
				sc->res = VKFFT_ERROR_MATH_FAILED;
				return;
			}
		}
		else {
			switch (in->type % 10) {
			case 3:
				sc->tempLen = sprintf(sc->tempStr, "%.17Le", in->data.c[0]);
				VkAppendLine(sc);
				break;
			default:
				sc->res = VKFFT_ERROR_MATH_FAILED;
				return;
			}
			VkAppendNumberLiteral(sc, out);
		}
		VkAppendConversionEnd(sc, out, in);
		sc->tempLen = sprintf(sc->tempStr, ";\n");
		VkAppendLine(sc);
		return;
	}
	else {
		switch (out->type % 10) {
		case 3:
			if (in->type > 100) {
			}
			else {
				switch (in->type % 10) {
				case 3:
					out->data.c[1] = in->data.c[0];
					return;
				}
			}
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}
static inline void VkMov_y_Neg_x(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
		//in_1 has to be same type as out
		switch (out->type % 10) {
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.y", out->data.s);
			VkAppendLine(sc);
			break;
		default:
			sc->res = VKFFT_ERROR_MATH_FAILED;
			return;
		}
		sc->tempLen = sprintf(sc->tempStr, " = ");
		VkAppendLine(sc);
		VkAppendConversionStart(sc, out, in);
		if (in->type > 100) {
			switch (in->type % 10) {
			case 3:
				sc->tempLen = sprintf(sc->tempStr, "-%s.x", in->data.s);
				VkAppendLine(sc);
				break;
			default:
				sc->res = VKFFT_ERROR_MATH_FAILED;
				return;
			}
		}
		else {
			switch (in->type % 10) {
			case 3:
				sc->tempLen = sprintf(sc->tempStr, "%.17Le", -in->data.c[0]);
				VkAppendLine(sc);
				break;
			default:
				sc->res = VKFFT_ERROR_MATH_FAILED;
				return;
			}
			VkAppendNumberLiteral(sc, out);
		}
		VkAppendConversionEnd(sc, out, in);
		sc->tempLen = sprintf(sc->tempStr, ";\n");
		VkAppendLine(sc);
		return;
	}
	else {
		switch (out->type % 10) {
		case 3:
			if (in->type > 100) {
			}
			else {
				switch (in->type % 10) {
				case 3:
					out->data.c[1] = -in->data.c[0];
					return;
				}
			}
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}

static inline void VkAdd(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in_1, VkContainer* in_2) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
#if(VKFFT_BACKEND == 2)
		if ((in_1->type > 100) && (in_2->type > 100)) {
			//packed instructions workaround if all values are in registers
			sc->tempLen = sprintf(sc->tempStr, "%s", out->data.s);
			VkAppendLine(sc);
			sc->tempLen = sprintf(sc->tempStr, " = ");
			VkAppendLine(sc);
			VkAppendConversionStart(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, "%s", in_1->data.s);
			VkAppendLine(sc);
			VkAppendConversionEnd(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, " + ");
			VkAppendLine(sc);
			VkAppendConversionStart(sc, out, in_2);
			sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.s);
			VkAppendLine(sc);
			VkAppendConversionEnd(sc, out, in_2);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
			return;
		}
#endif
		switch (out->type % 10) {
		case 1: case 2:
			sc->tempLen = sprintf(sc->tempStr, "%s", out->data.s);
			VkAppendLine(sc);
			break;
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.x", out->data.s);
			VkAppendLine(sc);
			break;
		}
		sc->tempLen = sprintf(sc->tempStr, " = ");
		VkAppendLine(sc);
		if ((in_1->type < 100) && (in_2->type < 100)) {
			switch (in_1->type % 10) {
			case 1: 
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i + in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i + in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i + in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				break;
			case 2:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d + (long double)in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d + in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d + in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				break;
			case 3:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] + (long double)in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] + in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] + in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				break;
			}
			VkAppendNumberLiteral(sc, out);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
		}
		else {
			VkAppendConversionStart(sc, out, in_1);
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in_1->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.x", in_1->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, " + ");
			VkAppendLine(sc);
			VkAppendConversionStart(sc, out, in_2);
			if (in_2->type > 100) {
				switch (in_2->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.x", in_2->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_2);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
		}
		switch (out->type % 10) {
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.y", out->data.s); 
			VkAppendLine(sc);
			sc->tempLen = sprintf(sc->tempStr, " = ");
			VkAppendLine(sc);
			if ((in_1->type < 100) && (in_2->type < 100)) {
				switch (in_1->type % 10) {
				case 1:
					switch (in_2->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i + in_2->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i + in_2->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i + in_2->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					break;
				case 2:
					switch (in_2->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d + (long double)in_2->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d + in_2->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d + in_2->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					break;
				case 3:
					switch (in_2->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] + (long double)in_2->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] + in_2->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] + in_2->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					break;
				}
				VkAppendNumberLiteral(sc, out);
				sc->tempLen = sprintf(sc->tempStr, ";\n");
				VkAppendLine(sc);
			}
			else {
				VkAppendConversionStart(sc, out, in_1);
				if (in_1->type > 100) {
					switch (in_1->type % 10) {
					case 1: case 2:
						sc->tempLen = sprintf(sc->tempStr, "%s", in_1->data.s);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%s.y", in_1->data.s);
						VkAppendLine(sc);
						break;
					}
				}
				else {
					switch (in_1->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					VkAppendNumberLiteral(sc, out);
				}
				VkAppendConversionEnd(sc, out, in_1);
				sc->tempLen = sprintf(sc->tempStr, " + ");
				VkAppendLine(sc);
				VkAppendConversionStart(sc, out, in_2);
				if (in_2->type > 100) {
					switch (in_2->type % 10) {
					case 1: case 2:
						sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.s);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%s.y", in_2->data.s);
						VkAppendLine(sc);
						break;
					}
				}
				else {
					switch (in_2->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_2->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					VkAppendNumberLiteral(sc, out);
				}
				VkAppendConversionEnd(sc, out, in_2);
				sc->tempLen = sprintf(sc->tempStr, ";\n");
				VkAppendLine(sc);
			}
			break;
		}
		return;
	}
	else {
		switch (out->type % 10) {
		case 1:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = in_1->data.i + in_2->data.i;
							return;
						case 2:
							out->data.i = in_1->data.i + in_2->data.d;
							return;
						case 3:
							break;
						}
					}
					break;
				case 2:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = (int64_t)in_1->data.d + in_2->data.i;
							return;
						case 2:
							out->data.i = (int64_t)(in_1->data.d + in_2->data.d);
							return;
						case 3:
							break;
						}
					}
					break;
				case 3:
					break;
				}
			}
		break;
		case 2:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.d = (long double)(in_1->data.i + in_2->data.i);
							return;
						case 2:
							out->data.d = (long double)in_1->data.i + in_2->data.d;
							return;
						case 3:
							break;
						}
					}
					break;
				case 2:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.d = in_1->data.d + (long double)in_2->data.i;
							return;
						case 2:
							out->data.d = in_1->data.d + in_2->data.d;
							return;
						case 3:
							break;
						}
					}
					break;
				case 3:
					break;
				}
			}
		break;
		case 3:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
				case 2:
					break;
				case 3:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.c[0] = in_1->data.c[0] + (long double)in_2->data.i;
							out->data.c[1] = in_1->data.c[1] + (long double)in_2->data.i;
							return;
						case 2:
							out->data.c[0] = in_1->data.c[0] + in_2->data.d;
							out->data.c[1] = in_1->data.c[1] + in_2->data.d;
							return;
						case 3:
							out->data.c[0] = in_1->data.c[0] + in_2->data.c[0];
							out->data.c[1] = in_1->data.c[1] + in_2->data.c[1];
							return;
						}
					}
					break;
				}
			}
		break;
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}

static inline void VkAdd_x(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in_1, VkContainer* in_2) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
		switch (out->type % 10) {
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.x", out->data.s);
			VkAppendLine(sc);
			break;
		default:
			sc->res = VKFFT_ERROR_MATH_FAILED;
			return;
		}
		sc->tempLen = sprintf(sc->tempStr, " = ");
		VkAppendLine(sc);
		if ((in_1->type < 100) && (in_2->type < 100)) {
			switch (in_1->type % 10) {
			case 1:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i + in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i + in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i + in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				break;
			case 2:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d + (long double)in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d + in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d + in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				break;
			case 3:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] + (long double)in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] + in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] + in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				break;
			}
			VkAppendNumberLiteral(sc, out);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
		}
		else {
			VkAppendConversionStart(sc, out, in_1);
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in_1->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.x", in_1->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, " + ");
			VkAppendLine(sc);
			VkAppendConversionStart(sc, out, in_2);
			if (in_2->type > 100) {
				switch (in_2->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.x", in_2->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_2);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
		}
		return;
	}
	else {
		switch (out->type % 10) {
		case 1:
			break;
		case 2:
			break;
		case 3:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
				case 2:
					break;
				case 3:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.c[0] = in_1->data.c[0] + (double)in_2->data.i;
							return;
						case 2:
							out->data.c[0] = in_1->data.c[0] + in_2->data.d;
							return;
						case 3:
							out->data.c[0] = in_1->data.c[0] + in_2->data.c[0];
							return;
						}
					}
				}
			}
			break;

		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}
static inline void VkAdd_y(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in_1, VkContainer* in_2) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
		switch (out->type % 10) {
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.y", out->data.s);
			VkAppendLine(sc);
			break;
		default:
			sc->res = VKFFT_ERROR_MATH_FAILED;
			return;
		}
		sc->tempLen = sprintf(sc->tempStr, " = ");
		VkAppendLine(sc);
		if ((in_1->type < 100) && (in_2->type < 100)) {
			switch (in_1->type % 10) {
			case 1:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i + in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i + in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i + in_2->data.c[1]);
					VkAppendLine(sc);
					break;
				}
				break;
			case 2:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d + (long double)in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d + in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d + in_2->data.c[1]);
					VkAppendLine(sc);
					break;
				}
				break;
			case 3:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] + (long double)in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] + in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] + in_2->data.c[1]);
					VkAppendLine(sc);
					break;
				}
				break;
			}
			VkAppendNumberLiteral(sc, out);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
		}
		else {
			VkAppendConversionStart(sc, out, in_1);
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in_1->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.y", in_1->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, " + ");
			VkAppendLine(sc);
			VkAppendConversionStart(sc, out, in_2);
			if (in_2->type > 100) {
				switch (in_2->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.y", in_2->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.c[1]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_2);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
		}
		return;
	}
	else {
		switch (out->type % 10) {
		case 1:
			break;
		case 2:
			break;
		case 3:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
				case 2:
					break;
				case 3:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.c[1] = in_1->data.c[1] + (double)in_2->data.i;
							return;
						case 2:
							out->data.c[1] = in_1->data.c[1] + in_2->data.d;
							return;
						case 3:
							out->data.c[1] = in_1->data.c[1] + in_2->data.c[1];
							return;
						}
					}
				}
			}
			break;

		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}
static inline void VkAdd_x_y(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in_1, VkContainer* in_2) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
		switch (out->type % 10) {
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.x", out->data.s);
			VkAppendLine(sc);
			break;
		default:
			sc->res = VKFFT_ERROR_MATH_FAILED;
			return;
		}
		sc->tempLen = sprintf(sc->tempStr, " = ");
		VkAppendLine(sc);
		if ((in_1->type < 100) && (in_2->type < 100)) {
			switch (in_1->type % 10) {
			case 1:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i + in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i + in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i + in_2->data.c[1]);
					VkAppendLine(sc);
					break;
		}
				break;
			case 2:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d + (long double)in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d + in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d + in_2->data.c[1]);
					VkAppendLine(sc);
					break;
				}
				break;
			case 3:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] + (long double)in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] + in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] + in_2->data.c[1]);
					VkAppendLine(sc);
					break;
				}
				break;
	}
			VkAppendNumberLiteral(sc, out);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
}
		else {
			VkAppendConversionStart(sc, out, in_1);
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in_1->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.x", in_1->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, " + ");
			VkAppendLine(sc);
			VkAppendConversionStart(sc, out, in_2);
			if (in_2->type > 100) {
				switch (in_2->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.y", in_2->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.c[1]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_2);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
		}
		return;
	}
	else {
		switch (out->type % 10) {
		case 1:
			break;
		case 2:
			break;
		case 3:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
				case 2:
					break;
				case 3:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.c[0] = in_1->data.c[0] + (double)in_2->data.i;
							return;
						case 2:
							out->data.c[0] = in_1->data.c[0] + in_2->data.d;
							return;
						case 3:
							out->data.c[0] = in_1->data.c[0] + in_2->data.c[1];
							return;
						}
					}
				}
			}
			break;

		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}
static inline void VkAdd_y_x(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in_1, VkContainer* in_2) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
		switch (out->type % 10) {
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.y", out->data.s);
			VkAppendLine(sc);
			break;
		default:
			sc->res = VKFFT_ERROR_MATH_FAILED;
			return;
		}
		sc->tempLen = sprintf(sc->tempStr, " = ");
		VkAppendLine(sc);
		if ((in_1->type < 100) && (in_2->type < 100)) {
			switch (in_1->type % 10) {
			case 1:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i + in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i + in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i + in_2->data.c[0]);
					VkAppendLine(sc);
					break;
	}
				break;
			case 2:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d + (long double)in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d + in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d + in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				break;
			case 3:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] + (long double)in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] + in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] + in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				break;
}
			VkAppendNumberLiteral(sc, out);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
}
		else {
			VkAppendConversionStart(sc, out, in_1);
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in_1->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.y", in_1->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, " + ");
			VkAppendLine(sc);
			VkAppendConversionStart(sc, out, in_2);
			if (in_2->type > 100) {
				switch (in_2->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.x", in_2->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_2);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
		}
		return;
	}
	else {
		switch (out->type % 10) {
		case 1:
			break;
		case 2:
			break;
		case 3:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
				case 2:
					break;
				case 3:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.c[1] = in_1->data.c[1] + (double)in_2->data.i;
							return;
						case 2:
							out->data.c[1] = in_1->data.c[1] + in_2->data.d;
							return;
						case 3:
							out->data.c[1] = in_1->data.c[1] + in_2->data.c[0];
							return;
						}
					}
				}
			}
			break;

		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}

static inline void VkAddInv(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in_1, VkContainer* in_2) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
#if(VKFFT_BACKEND == 2)
		if ((in_1->type > 100) && (in_2->type > 100)) {
			//packed instructions workaround if all values are in registers
			sc->tempLen = sprintf(sc->tempStr, "%s", out->data.s);
			VkAppendLine(sc);
			sc->tempLen = sprintf(sc->tempStr, " = ");
			VkAppendLine(sc);
			VkAppendConversionStart(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, "-%s", in_1->data.s);
			VkAppendLine(sc);
			VkAppendConversionEnd(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, " - ");
			VkAppendLine(sc);
			VkAppendConversionStart(sc, out, in_2);
			sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.s);
			VkAppendLine(sc);
			VkAppendConversionEnd(sc, out, in_2);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
			return;
		}
#endif
		switch (out->type % 10) {
		case 1: case 2:
			sc->tempLen = sprintf(sc->tempStr, "%s", out->data.s);
			VkAppendLine(sc);
			break;
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.x", out->data.s);
			VkAppendLine(sc);
			break;
		}
		sc->tempLen = sprintf(sc->tempStr, " = ");
		VkAppendLine(sc);
		if ((in_1->type < 100) && (in_2->type < 100)) {
			switch (in_1->type % 10) {
			case 1:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", -in_1->data.i - in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", -(long double)in_1->data.i - in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", -(long double)in_1->data.i - in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				break;
			case 2:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", -in_1->data.d - (long double)in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", -in_1->data.d - in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", -in_1->data.d - in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				break;
			case 3:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", -in_1->data.c[0] - (long double)in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", -in_1->data.c[0] - in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", -in_1->data.c[0] - in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				break;
			}
			VkAppendNumberLiteral(sc, out);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
		}
		else {
			VkAppendConversionStart(sc, out, in_1);
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "-%s", in_1->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "-%s.x", in_1->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", -in_1->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", -in_1->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", -in_1->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, " + ");
			VkAppendLine(sc);
			VkAppendConversionStart(sc, out, in_2);
			if (in_2->type > 100) {
				switch (in_2->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "(-%s)", in_2->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "(-%s.x)", in_2->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "(%" PRIi64 ")", -in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "(%.17Le)", -in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "(%.17Le)", -in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_2);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
		}
		switch (out->type % 10) {
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.y", out->data.s);
			VkAppendLine(sc);
			sc->tempLen = sprintf(sc->tempStr, " = ");
			VkAppendLine(sc);
			if ((in_1->type < 100) && (in_2->type < 100)) {
				switch (in_1->type % 10) {
				case 1:
					switch (in_2->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", -in_1->data.i - in_2->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", -(long double)in_1->data.i - in_2->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", -(long double)in_1->data.i - in_2->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					break;
				case 2:
					switch (in_2->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", -in_1->data.d - (long double)in_2->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", -in_1->data.d - in_2->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", -in_1->data.d - in_2->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					break;
				case 3:
					switch (in_2->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", -in_1->data.c[1] - (long double)in_2->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", -in_1->data.c[1] - in_2->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", -in_1->data.c[1] - in_2->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					break;
				}
				VkAppendNumberLiteral(sc, out);
				sc->tempLen = sprintf(sc->tempStr, ";\n");
				VkAppendLine(sc);
			}
			else {
				VkAppendConversionStart(sc, out, in_1);
				if (in_1->type > 100) {
					switch (in_1->type % 10) {
					case 1: case 2:
						sc->tempLen = sprintf(sc->tempStr, "-%s", in_1->data.s);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "-%s.y", in_1->data.s);
						VkAppendLine(sc);
						break;
					}
				}
				else {
					switch (in_1->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", -in_1->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", -in_1->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", -in_1->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					VkAppendNumberLiteral(sc, out);
				}
				VkAppendConversionEnd(sc, out, in_1);
				sc->tempLen = sprintf(sc->tempStr, " + ");
				VkAppendLine(sc);
				VkAppendConversionStart(sc, out, in_2);
				if (in_2->type > 100) {
					switch (in_2->type % 10) {
					case 1: case 2:
						sc->tempLen = sprintf(sc->tempStr, "(-%s)", in_2->data.s);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "(-%s.y)", in_2->data.s);
						VkAppendLine(sc);
						break;
					}
				}
				else {
					switch (in_2->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "(%" PRIi64 ")", -in_2->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "(%.17Le)", -in_2->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "(%.17Le)", -in_2->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					VkAppendNumberLiteral(sc, out);
				}
				VkAppendConversionEnd(sc, out, in_2);
				sc->tempLen = sprintf(sc->tempStr, ";\n");
				VkAppendLine(sc);
			}
			break;
		}
		return;
	}
	else {
		switch (out->type % 10) {
		case 1:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = -in_1->data.i - in_2->data.i;
							return;
						case 2:
							out->data.i = -in_1->data.i - in_2->data.d;
							return;
						case 3:
							break;
						}
					}
					break;
				case 2:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = -(int64_t)in_1->data.d - in_2->data.i;
							return;
						case 2:
							out->data.i = -(int64_t)(in_1->data.d + in_2->data.d);
							return;
						case 3:
							break;
						}
					}
					break;
				case 3:
					break;
				}
			}
			break;
		case 2:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.d = -(long double)(in_1->data.i + in_2->data.i);
							return;
						case 2:
							out->data.d = -(long double)in_1->data.i - in_2->data.d;
							return;
						case 3:
							break;
						}
					}
					break;
				case 2:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.d = -in_1->data.d - (long double)in_2->data.i;
							return;
						case 2:
							out->data.d = -in_1->data.d - in_2->data.d;
							return;
						case 3:
							break;
						}
					}
					break;
				case 3:
					break;
				}
			}
			break;
		case 3:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
				case 2:
					break;
				case 3:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.c[0] = -in_1->data.c[0] - (long double)in_2->data.i;
							out->data.c[1] = -in_1->data.c[1] - (long double)in_2->data.i;
							return;
						case 2:
							out->data.c[0] = -in_1->data.c[0] - in_2->data.d;
							out->data.c[1] = -in_1->data.c[1] - in_2->data.d;
							return;
						case 3:
							out->data.c[0] = -in_1->data.c[0] - in_2->data.c[0];
							out->data.c[1] = -in_1->data.c[1] - in_2->data.c[1];
							return;
						}
					}
					break;
				}
			}
			break;
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}

static inline void VkInc(VkFFTSpecializationConstantsLayout* sc, VkContainer* out) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
		//in_1 has to be same type as out
		switch (out->type % 10) {
		case 1:
		case 2:
			sc->tempLen = sprintf(sc->tempStr, "\
%s = %s + 1;\n", out->data.s, out->data.s);
			VkAppendLine(sc);
			return;
		case 3:
			break;
		}
	}
	else {
		switch (out->type % 10) {
		case 1:
			out->data.i = out->data.i + 1;
			return;
		case 2:
			out->data.d = out->data.d + 1;
			return;
			break;
		case 3:
			break;
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}

static inline void VkSub(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in_1, VkContainer* in_2) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
#if(VKFFT_BACKEND == 2)
		if ((in_1->type > 100) && (in_2->type > 100)) {
			//packed instructions workaround if all values are in registers
			sc->tempLen = sprintf(sc->tempStr, "%s", out->data.s);
			VkAppendLine(sc);
			sc->tempLen = sprintf(sc->tempStr, " = ");
			VkAppendLine(sc);
			VkAppendConversionStart(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, "%s", in_1->data.s);
			VkAppendLine(sc);
			VkAppendConversionEnd(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, " - ");
			VkAppendLine(sc);
			VkAppendConversionStart(sc, out, in_2);
			sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.s);
			VkAppendLine(sc);
			VkAppendConversionEnd(sc, out, in_2);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
			return;
		}
#endif
		switch (out->type % 10) {
		case 1: case 2:
			sc->tempLen = sprintf(sc->tempStr, "%s", out->data.s);
			VkAppendLine(sc);
			break;
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.x", out->data.s);
			VkAppendLine(sc);
			break;
		}
		sc->tempLen = sprintf(sc->tempStr, " = ");
		VkAppendLine(sc);
		if ((in_1->type < 100) && (in_2->type < 100)) {
			switch (in_1->type % 10) {
			case 1:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i - in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i - in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i - in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				break;
			case 2:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d - (long double)in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d - in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d - in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				break;
			case 3:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] - (long double)in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] - in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] - in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				break;
			}
			VkAppendNumberLiteral(sc, out);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
		}
		else {
			VkAppendConversionStart(sc, out, in_1);
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in_1->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.x", in_1->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, " - ");
			VkAppendLine(sc);
			VkAppendConversionStart(sc, out, in_2);
			if (in_2->type > 100) {
				switch (in_2->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.x", in_2->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_2);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
		}
		switch (out->type % 10) {
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.y", out->data.s);
			VkAppendLine(sc);
			sc->tempLen = sprintf(sc->tempStr, " = ");
			VkAppendLine(sc);
			if ((in_1->type < 100) && (in_2->type < 100)) {
				switch (in_1->type % 10) {
				case 1:
					switch (in_2->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i - in_2->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i - in_2->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i - in_2->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					break;
				case 2:
					switch (in_2->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d - (long double)in_2->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d - in_2->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d - in_2->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					break;
				case 3:
					switch (in_2->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] - (long double)in_2->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] - in_2->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] - in_2->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					break;
				}
				VkAppendNumberLiteral(sc, out);
				sc->tempLen = sprintf(sc->tempStr, ";\n");
				VkAppendLine(sc);
			}
			else {
				VkAppendConversionStart(sc, out, in_1);
				if (in_1->type > 100) {
					switch (in_1->type % 10) {
					case 1: case 2:
						sc->tempLen = sprintf(sc->tempStr, "%s", in_1->data.s);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%s.y", in_1->data.s);
						VkAppendLine(sc);
						break;
					}
				}
				else {
					switch (in_1->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					VkAppendNumberLiteral(sc, out);
				}
				VkAppendConversionEnd(sc, out, in_1);
				sc->tempLen = sprintf(sc->tempStr, " - ");
				VkAppendLine(sc);
				VkAppendConversionStart(sc, out, in_2);
				if (in_2->type > 100) {
					switch (in_2->type % 10) {
					case 1: case 2:
						sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.s);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%s.y", in_2->data.s);
						VkAppendLine(sc);
						break;
					}
				}
				else {
					switch (in_2->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_2->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					VkAppendNumberLiteral(sc, out);
				}
				VkAppendConversionEnd(sc, out, in_2);
				sc->tempLen = sprintf(sc->tempStr, ";\n");
				VkAppendLine(sc);
			}
			break;
		}
		return;
	}
	else {
		switch (out->type % 10) {
		case 1:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = in_1->data.i - in_2->data.i;
							return;
						case 2:
							out->data.i = in_1->data.i - in_2->data.d;
							return;
						case 3:
							break;
						}
					}
					break;
				case 2:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = (int64_t)in_1->data.d - in_2->data.i;
							return;
						case 2:
							out->data.i = (int64_t)(in_1->data.d - in_2->data.d);
							return;
						case 3:
							break;
						}
					}
					break;
				case 3:
					break;
				}
			}
			break;
		case 2:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.d = (long double)(in_1->data.i - in_2->data.i);
							return;
						case 2:
							out->data.d = (long double)in_1->data.i - in_2->data.d;
							return;
						case 3:
							break;
						}
					}
					break;
				case 2:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.d = in_1->data.d - (long double)in_2->data.i;
							return;
						case 2:
							out->data.d = in_1->data.d - in_2->data.d;
							return;
						case 3:
							break;
						}
					}
					break;
				case 3:
					break;
				}
			}
			break;
		case 3:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
				case 2:
					break;
				case 3:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.c[0] = in_1->data.c[0] - (long double)in_2->data.i;
							out->data.c[1] = in_1->data.c[1] - (long double)in_2->data.i;
							return;
						case 2:
							out->data.c[0] = in_1->data.c[0] - in_2->data.d;
							out->data.c[1] = in_1->data.c[1] - in_2->data.d;
							return;
						case 3:
							out->data.c[0] = in_1->data.c[0] - in_2->data.c[0];
							out->data.c[1] = in_1->data.c[1] - in_2->data.c[1];
							return;
						}
					}
					break;
				}
			}
			break;
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}

static inline void VkSub_x(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in_1, VkContainer* in_2) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
		switch (out->type % 10) {
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.x", out->data.s);
			VkAppendLine(sc);
			break;
		default:
			sc->res = VKFFT_ERROR_MATH_FAILED;
			return;
		}
		sc->tempLen = sprintf(sc->tempStr, " = ");
		VkAppendLine(sc);
		if ((in_1->type < 100) && (in_2->type < 100)) {
			switch (in_1->type % 10) {
			case 1:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i - in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i - in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i - in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				break;
			case 2:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d - (long double)in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d - in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d - in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				break;
			case 3:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] - (long double)in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] - in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] - in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				break;
			}
			VkAppendNumberLiteral(sc, out);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
		}
		else {
			VkAppendConversionStart(sc, out, in_1);
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in_1->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.x", in_1->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, " - ");
			VkAppendLine(sc);
			VkAppendConversionStart(sc, out, in_2);
			if (in_2->type > 100) {
				switch (in_2->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.x", in_2->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_2);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
		}
		return;
	}
	else {
		switch (out->type % 10) {
		case 1:
			break;
		case 2:
			break;
		case 3:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
				case 2:
					break;
				case 3:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.c[0] = in_1->data.c[0] - (double)in_2->data.i;
							return;
						case 2:
							out->data.c[0] = in_1->data.c[0] - in_2->data.d;
							return;
						case 3:
							out->data.c[0] = in_1->data.c[0] - in_2->data.c[0];
							return;
						}
					}
				}
			}
			break;

		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}
static inline void VkSub_y(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in_1, VkContainer* in_2) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
		switch (out->type % 10) {
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.y", out->data.s);
			VkAppendLine(sc);
			break;
		default:
			sc->res = VKFFT_ERROR_MATH_FAILED;
			return;
		}
		sc->tempLen = sprintf(sc->tempStr, " = ");
		VkAppendLine(sc);
		if ((in_1->type < 100) && (in_2->type < 100)) {
			switch (in_1->type % 10) {
			case 1:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i - in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i - in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i - in_2->data.c[1]);
					VkAppendLine(sc);
					break;
				}
				break;
			case 2:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d - (long double)in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d - in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d - in_2->data.c[1]);
					VkAppendLine(sc);
					break;
				}
				break;
			case 3:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] - (long double)in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] - in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] - in_2->data.c[1]);
					VkAppendLine(sc);
					break;
				}
				break;
			}
			VkAppendNumberLiteral(sc, out);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
		}
		else {
			VkAppendConversionStart(sc, out, in_1);
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in_1->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.y", in_1->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, " - ");
			VkAppendLine(sc);
			VkAppendConversionStart(sc, out, in_2);
			if (in_2->type > 100) {
				switch (in_2->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.y", in_2->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.c[1]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_2);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
		}
		return;
	}
	else {
		switch (out->type % 10) {
		case 1:
			break;
		case 2:
			break;
		case 3:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
				case 2:
					break;
				case 3:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.c[1] = in_1->data.c[1] - (double)in_2->data.i;
							return;
						case 2:
							out->data.c[1] = in_1->data.c[1] - in_2->data.d;
							return;
						case 3:
							out->data.c[1] = in_1->data.c[1] - in_2->data.c[1];
							return;
						}
					}
				}
			}
			break;

		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}
static inline void VkSub_x_y(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in_1, VkContainer* in_2) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
		switch (out->type % 10) {
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.x", out->data.s);
			VkAppendLine(sc);
			break;
		default:
			sc->res = VKFFT_ERROR_MATH_FAILED;
			return;
		}
		sc->tempLen = sprintf(sc->tempStr, " = ");
		VkAppendLine(sc);
		if ((in_1->type < 100) && (in_2->type < 100)) {
			switch (in_1->type % 10) {
			case 1:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i - in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i - in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i - in_2->data.c[1]);
					VkAppendLine(sc);
					break;
				}
				break;
			case 2:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d - (long double)in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d - in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d - in_2->data.c[1]);
					VkAppendLine(sc);
					break;
		}
				break;
			case 3:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] - (long double)in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] - in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] - in_2->data.c[1]);
					VkAppendLine(sc);
					break;
				}
				break;
	}
			VkAppendNumberLiteral(sc, out);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
}
		else {
			VkAppendConversionStart(sc, out, in_1);
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in_1->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.x", in_1->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, " - ");
			VkAppendLine(sc);
			VkAppendConversionStart(sc, out, in_2);
			if (in_2->type > 100) {
				switch (in_2->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.y", in_2->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.c[1]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_2);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
		}
		return;
	}
	else {
		switch (out->type % 10) {
		case 1:
			break;
		case 2:
			break;
		case 3:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
				case 2:
					break;
				case 3:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.c[0] = in_1->data.c[0] - (double)in_2->data.i;
							return;
						case 2:
							out->data.c[0] = in_1->data.c[0] - in_2->data.d;
							return;
						case 3:
							out->data.c[0] = in_1->data.c[0] - in_2->data.c[1];
							return;
						}
					}
				}
			}
			break;

		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}
static inline void VkSub_y_x(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in_1, VkContainer* in_2) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
		switch (out->type % 10) {
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.y", out->data.s);
			VkAppendLine(sc);
			break;
		default:
			sc->res = VKFFT_ERROR_MATH_FAILED;
			return;
		}
		sc->tempLen = sprintf(sc->tempStr, " = ");
		VkAppendLine(sc);
		if ((in_1->type < 100) && (in_2->type < 100)) {
			switch (in_1->type % 10) {
			case 1:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i - in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i - in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i - in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				break;
			case 2:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d - (long double)in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d - in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d - in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				break;
			case 3:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] - (long double)in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] - in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] - in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				break;
			}
			VkAppendNumberLiteral(sc, out);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
		}
		else {
			VkAppendConversionStart(sc, out, in_1);
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in_1->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.y", in_1->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, " - ");
			VkAppendLine(sc);
			VkAppendConversionStart(sc, out, in_2);
			if (in_2->type > 100) {
				switch (in_2->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.x", in_2->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_2);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
		}
		return;
	}
	else {
		switch (out->type % 10) {
		case 1:
			break;
		case 2:
			break;
		case 3:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
				case 2:
					break;
				case 3:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.c[1] = in_1->data.c[1] - (double)in_2->data.i;
							return;
						case 2:
							out->data.c[1] = in_1->data.c[1] - in_2->data.d;
							return;
						case 3:
							out->data.c[1] = in_1->data.c[1] - in_2->data.c[0];
							return;
						}
					}
				}
			}
			break;

		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}

static inline void VkFMA(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in_1, VkContainer* in_2, VkContainer* in_3) {
	//fma inlining is not correct if all three numbers are complex for now
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
#if(VKFFT_BACKEND == 2)
		if ((in_1->type > 100) && (in_2->type > 100) && (in_3->type > 100)) {
			//packed instructions workaround if all values are in registers
			if (((in_1->type % 10) != 3) || ((in_2->type % 10) != 3)) {
				sc->tempLen = sprintf(sc->tempStr, "%s", out->data.s);
				VkAppendLine(sc);
				sc->tempLen = sprintf(sc->tempStr, " = ");
				VkAppendLine(sc);
				VkAppendConversionStart(sc, out, in_1);
				sc->tempLen = sprintf(sc->tempStr, "%s", in_1->data.s);
				VkAppendLine(sc);
				VkAppendConversionEnd(sc, out, in_1);
				sc->tempLen = sprintf(sc->tempStr, " * ");
				VkAppendLine(sc);
				VkAppendConversionStart(sc, out, in_2);
				sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.s);
				VkAppendLine(sc);
				VkAppendConversionEnd(sc, out, in_2);
				sc->tempLen = sprintf(sc->tempStr, " + ");
				VkAppendLine(sc);
				VkAppendConversionStart(sc, out, in_1);
				sc->tempLen = sprintf(sc->tempStr, "%s", in_3->data.s);
				VkAppendLine(sc);
				VkAppendConversionEnd(sc, out, in_1);
				sc->tempLen = sprintf(sc->tempStr, ";\n");
				VkAppendLine(sc);
				return;
			}
		}
#endif
		switch (out->type % 10) {
		case 1: case 2:
			sc->tempLen = sprintf(sc->tempStr, "%s", out->data.s);
			VkAppendLine(sc);
			break;
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.x", out->data.s);
			VkAppendLine(sc);
			break;
		}
		sc->tempLen = sprintf(sc->tempStr, " = ");
		VkAppendLine(sc);
		if ((in_1->type < 100) && (in_2->type < 100) && (in_3->type < 100)) {
			switch (in_1->type % 10) {
			case 1:
				switch (in_2->type % 10) {
				case 1:
					switch (in_3->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i * in_2->data.i + in_3->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)(in_1->data.i * in_2->data.i) + in_3->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)(in_1->data.i * in_2->data.i) + in_3->data.c[0]);
						VkAppendLine(sc);
						break;
}
					break;
				case 2:
					switch (in_3->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i * in_2->data.d + (long double)in_3->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i * in_2->data.d + in_3->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i * in_2->data.d + in_3->data.c[0]);
						VkAppendLine(sc);
						break;
					}
					break;
				case 3:
					switch (in_3->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i * in_2->data.c[0] + (long double)in_3->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i * in_2->data.c[0] + in_3->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i * in_2->data.c[0] + in_3->data.c[0]);
						VkAppendLine(sc);
						break;
					}
					break;
				}
				break;
			case 2:
				switch (in_2->type % 10) {
				case 1:
					switch (in_3->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * (long double)in_2->data.i + (long double)in_3->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * in_2->data.i + in_3->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * in_2->data.i + in_3->data.c[0]);
						VkAppendLine(sc);
						break;
					}
					break;
				case 2:
					switch (in_3->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * in_2->data.d + (long double)in_3->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * in_2->data.d + in_3->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * in_2->data.d + in_3->data.c[0]);
						VkAppendLine(sc);
						break;
					}
					break;
				case 3:
					switch (in_3->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * in_2->data.c[0] + (long double)in_3->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * in_2->data.c[0] + in_3->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * in_2->data.c[0] + in_3->data.c[0]);
						VkAppendLine(sc);
						break;
					}
					break;
				}
				break;
			case 3:
				switch (in_2->type % 10) {
				case 1:
					switch (in_3->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] * (long double)in_2->data.i + (long double)in_3->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] * in_2->data.i + in_3->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] * in_2->data.i + in_3->data.c[0]);
						VkAppendLine(sc);
						break;
					}
					break;
				case 2:
					switch (in_3->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] * in_2->data.d + (long double)in_3->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] * in_2->data.d + in_3->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] * in_2->data.d + in_3->data.c[0]);
						VkAppendLine(sc);
						break;
					}
					break;
				case 3:
					switch (in_3->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] * in_2->data.c[0] - in_1->data.c[1] * in_2->data.c[1] + (long double)in_3->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] * in_2->data.c[0] - in_1->data.c[1] * in_2->data.c[1] + in_3->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] * in_2->data.c[0] - in_1->data.c[1] * in_2->data.c[1] + in_3->data.c[0]);
						VkAppendLine(sc);
						break;
					}
					break;
				}
				break;
			}
			VkAppendNumberLiteral(sc, out);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
		}
		else if ((in_1->type < 100) && (in_2->type < 100) && (in_3->type > 100)) {
			switch (in_1->type % 10) {
			case 1:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i * in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i * in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i * in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				break;
			case 2:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * (long double)in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				break;
			case 3:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] * (long double)in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] * in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] * in_2->data.c[0] - in_1->data.c[1] * in_2->data.c[1]);
					VkAppendLine(sc);
					break;
				}
				break;
			}
			VkAppendNumberLiteral(sc, out);
			sc->tempLen = sprintf(sc->tempStr, " + ");
			VkAppendLine(sc);
			VkAppendConversionStart(sc, out, in_3);
			switch (in_3->type % 10) {
			case 1: case 2:
				sc->tempLen = sprintf(sc->tempStr, "%s", in_3->data.s);
				VkAppendLine(sc);
				break;
			case 3:
				sc->tempLen = sprintf(sc->tempStr, "%s.x", in_3->data.s);
				VkAppendLine(sc);
				break;
			}
			VkAppendConversionEnd(sc, out, in_3);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
		}
		else {
			sc->tempLen = sprintf(sc->tempStr, "fma(");
			VkAppendLine(sc);
			VkAppendConversionStart(sc, out, in_1);
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in_1->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.x", in_1->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, ", ");
			VkAppendLine(sc);
			VkAppendConversionStart(sc, out, in_2);
			if (in_2->type > 100) {
				switch (in_2->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.x", in_2->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_2);
			sc->tempLen = sprintf(sc->tempStr, ", ");
			VkAppendLine(sc);
			VkAppendConversionStart(sc, out, in_3);
			if (in_3->type > 100) {
				switch (in_3->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.x", in_3->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_3->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_3->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_3->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_3->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_3);
			sc->tempLen = sprintf(sc->tempStr, ");\n");
			VkAppendLine(sc);
		}
		switch (out->type % 10) {
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.y", out->data.s);
			VkAppendLine(sc);
			sc->tempLen = sprintf(sc->tempStr, " = ");
			VkAppendLine(sc);
			if ((in_1->type < 100) && (in_2->type < 100) && (in_3->type < 100)) {
				switch (in_1->type % 10) {
				case 1:
					switch (in_2->type % 10) {
					case 1:
						switch (in_3->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i * in_2->data.i + in_3->data.i);
							VkAppendLine(sc);
							break;
						case 2:
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)(in_1->data.i * in_2->data.i) + in_3->data.d);
							VkAppendLine(sc);
							break;
						case 3:
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)(in_1->data.i * in_2->data.i) + in_3->data.c[1]);
							VkAppendLine(sc);
							break;
						}
						break;
					case 2:
						switch (in_3->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i * in_2->data.d + (long double)in_3->data.i);
							VkAppendLine(sc);
							break;
						case 2:
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i * in_2->data.d + in_3->data.d);
							VkAppendLine(sc);
							break;
						case 3:
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i * in_2->data.d + in_3->data.c[1]);
							VkAppendLine(sc);
							break;
						}
						break;
					case 3:
						switch (in_3->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i * in_2->data.c[1] + (long double)in_3->data.i);
							VkAppendLine(sc);
							break;
						case 2:
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i * in_2->data.c[1] + in_3->data.d);
							VkAppendLine(sc);
							break;
						case 3:
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i * in_2->data.c[1] + in_3->data.c[1]);
							VkAppendLine(sc);
							break;
						}
						break;
					}
					break;
				case 2:
					switch (in_2->type % 10) {
					case 1:
						switch (in_3->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * (long double)in_2->data.i + (long double)in_3->data.i);
							VkAppendLine(sc);
							break;
						case 2:
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * in_2->data.i + in_3->data.d);
							VkAppendLine(sc);
							break;
						case 3:
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * in_2->data.i + in_3->data.c[1]);
							VkAppendLine(sc);
							break;
						}
						break;
					case 2:
						switch (in_3->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * in_2->data.d + (long double)in_3->data.i);
							VkAppendLine(sc);
							break;
						case 2:
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * in_2->data.d + in_3->data.d);
							VkAppendLine(sc);
							break;
						case 3:
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * in_2->data.d + in_3->data.c[1]);
							VkAppendLine(sc);
							break;
						}
						break;
					case 3:
						switch (in_3->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * in_2->data.c[1] + (long double)in_3->data.i);
							VkAppendLine(sc);
							break;
						case 2:
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * in_2->data.c[1] + in_3->data.d);
							VkAppendLine(sc);
							break;
						case 3:
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * in_2->data.c[1] + in_3->data.c[1]);
							VkAppendLine(sc);
							break;
						}
						break;
					}
					break;
				case 3:
					switch (in_2->type % 10) {
					case 1:
						switch (in_3->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] * (long double)in_2->data.i + (long double)in_3->data.i);
							VkAppendLine(sc);
							break;
						case 2:
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] * in_2->data.i + in_3->data.d);
							VkAppendLine(sc);
							break;
						case 3:
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] * in_2->data.i + in_3->data.c[0]);
							VkAppendLine(sc);
							break;
						}
						break;
					case 2:
						switch (in_3->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] * in_2->data.d + (long double)in_3->data.i);
							VkAppendLine(sc);
							break;
						case 2:
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] * in_2->data.d + in_3->data.d);
							VkAppendLine(sc);
							break;
						case 3:
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] * in_2->data.d + in_3->data.c[0]);
							VkAppendLine(sc);
							break;
						}
						break;
					case 3:
						switch (in_3->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] * in_2->data.c[1] + in_1->data.c[1] * in_2->data.c[0] + (long double)in_3->data.i);
							VkAppendLine(sc);
							break;
						case 2:
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] * in_2->data.c[1] + in_1->data.c[1] * in_2->data.c[0] + in_3->data.d);
							VkAppendLine(sc);
							break;
						case 3:
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] * in_2->data.c[1] + in_1->data.c[1] * in_2->data.c[0] + in_3->data.c[1]);
							VkAppendLine(sc);
							break;
						}
						break;
					}
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			else if ((in_1->type < 100) && (in_2->type < 100) && (in_3->type > 100)) {
				switch (in_1->type % 10) {
				case 1:
					switch (in_2->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i * in_2->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i * in_2->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i * in_2->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					break;
				case 2:
					switch (in_2->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * (long double)in_2->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * in_2->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * in_2->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					break;
				case 3:
					switch (in_2->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] * (long double)in_2->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] * in_2->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] * in_2->data.c[1] + in_1->data.c[1] * in_2->data.c[0]);
						VkAppendLine(sc);
						break;
					}
					break;
				}
				VkAppendNumberLiteral(sc, out);
				sc->tempLen = sprintf(sc->tempStr, " + ");
				VkAppendLine(sc);
				VkAppendConversionStart(sc, out, in_3);
				switch (in_3->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in_3->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.y", in_3->data.s);
					VkAppendLine(sc);
					break;
				}
				VkAppendConversionEnd(sc, out, in_3);
			}
			else {
				sc->tempLen = sprintf(sc->tempStr, "fma(");
				VkAppendLine(sc);
				VkAppendConversionStart(sc, out, in_1);
				if (in_1->type > 100) {
					switch (in_1->type % 10) {
					case 1: case 2:
						sc->tempLen = sprintf(sc->tempStr, "%s", in_1->data.s);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%s.y", in_1->data.s);
						VkAppendLine(sc);
						break;
					}
				}
				else {
					switch (in_1->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					VkAppendNumberLiteral(sc, out);
				}
				VkAppendConversionEnd(sc, out, in_1);
				sc->tempLen = sprintf(sc->tempStr, ", ");
				VkAppendLine(sc);
				VkAppendConversionStart(sc, out, in_2);
				if (in_2->type > 100) {
					switch (in_2->type % 10) {
					case 1: case 2:
						sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.s);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%s.y", in_2->data.s);
						VkAppendLine(sc);
						break;
					}
				}
				else {
					switch (in_2->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_2->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					VkAppendNumberLiteral(sc, out);
				}
				VkAppendConversionEnd(sc, out, in_2);
				sc->tempLen = sprintf(sc->tempStr, ", ");
				VkAppendLine(sc);
				VkAppendConversionStart(sc, out, in_3);
				if (in_3->type > 100) {
					switch (in_3->type % 10) {
					case 1: case 2:
						sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.s);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%s.y", in_3->data.s);
						VkAppendLine(sc);
						break;
					}
				}
				else {
					switch (in_3->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_3->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_3->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_3->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					VkAppendNumberLiteral(sc, out);
				}
				VkAppendConversionEnd(sc, out, in_3);
				sc->tempLen = sprintf(sc->tempStr, ");\n");
				VkAppendLine(sc);
			}
		}

		return;
	}
	else {
		switch (out->type % 10) {
		case 1:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							if (in_3->type > 100) {
							}
							else {
								switch (in_3->type % 10) {
								case 1:
									out->data.i = in_1->data.i * in_2->data.i + in_3->data.i;
									return;
								case 2:
									out->data.i = in_1->data.i * in_2->data.i + in_3->data.d;
									return;
								case 3:
									break;
								}
							}
							break;
						case 2:
							if (in_3->type > 100) {
							}
							else {
								switch (in_3->type % 10) {
								case 1:
									out->data.i = in_1->data.i * in_2->data.d + in_3->data.i;
									return;
								case 2:
									out->data.i = in_1->data.i * in_2->data.d + in_3->data.d;
									return;
								case 3:
									break;
								}
							}
							break;
						case 3:
							break;
						}
					}
					break;
				case 2:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							if (in_3->type > 100) {
							}
							else {
								switch (in_3->type % 10) {
								case 1:
									out->data.i = in_1->data.d * in_2->data.i + in_3->data.i;
									return;
								case 2:
									out->data.i = in_1->data.d * in_2->data.i + in_3->data.d;
									return;
								case 3:
									break;
								}
							}
							break;
						case 2:
							if (in_3->type > 100) {
							}
							else {
								switch (in_3->type % 10) {
								case 1:
									out->data.i = in_1->data.d * in_2->data.d + in_3->data.i;
									return;
								case 2:
									out->data.i = in_1->data.d * in_2->data.d + in_3->data.d;
									return;
								case 3:
									break;
								}
							}
							break;
						case 3:
							break;
						}
					}
					break;
				case 3:
					break;
				}
			}
			break;
		case 2:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							if (in_3->type > 100) {
							}
							else {
								switch (in_3->type % 10) {
								case 1:
									out->data.d = in_1->data.i * in_2->data.i + in_3->data.i;
									return;
								case 2:
									out->data.d = in_1->data.i * in_2->data.i + in_3->data.d;
									return;
								case 3:
									break;
								}
							}
							break;
						case 2:
							if (in_3->type > 100) {
							}
							else {
								switch (in_3->type % 10) {
								case 1:
									out->data.d = in_1->data.i * in_2->data.d + in_3->data.i;
									return;
								case 2:
									out->data.d = in_1->data.i * in_2->data.d + in_3->data.d;
									return;
								case 3:
									break;
								}
							}
							break;
						case 3:
							break;
						}
					}
					break;
				case 2:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							if (in_3->type > 100) {
							}
							else {
								switch (in_3->type % 10) {
								case 1:
									out->data.d = in_1->data.d * in_2->data.i + in_3->data.i;
									return;
								case 2:
									out->data.d = in_1->data.d * in_2->data.i + in_3->data.d;
									return;
								case 3:
									break;
								}
							}
							break;
						case 2:
							if (in_3->type > 100) {
							}
							else {
								switch (in_3->type % 10) {
								case 1:
									out->data.d = in_1->data.d * in_2->data.d + in_3->data.i;
									return;
								case 2:
									out->data.d = in_1->data.d * in_2->data.d + in_3->data.d;
									return;
								case 3:
									break;
								}
							}
							break;
						case 3:
							break;
						}
					}
					break;
				case 3:
					break;
				}
			}
			break;
		case 3:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
				case 2:
					break;
				case 3:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							if (in_3->type > 100) {
							}
							else {
								switch (in_3->type % 10) {
								case 1:
									out->data.c[0] = in_1->data.c[0] * in_2->data.i + in_3->data.i;
									out->data.c[1] = in_1->data.c[1] * in_2->data.i + in_3->data.i;
									return;
								case 2:
									out->data.c[0] = in_1->data.c[0] * in_2->data.i + in_3->data.d;
									out->data.c[1] = in_1->data.c[1] * in_2->data.i + in_3->data.d;
									return;
								case 3:
									break;
								}
							}
							break;
						case 2:
							if (in_3->type > 100) {
							}
							else {
								switch (in_3->type % 10) {
								case 1:
									out->data.c[0] = in_1->data.c[0] * in_2->data.d + in_3->data.i;
									out->data.c[1] = in_1->data.c[1] * in_2->data.d + in_3->data.i;
									return;
								case 2:
									out->data.c[0] = in_1->data.c[0] * in_2->data.d + in_3->data.d;
									out->data.c[1] = in_1->data.c[1] * in_2->data.d + in_3->data.d;
									return;
								case 3:
									break;
								}
							}
							break;
						case 3:
							break;
						}
					}
					break;
				}
			}
			break;
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}

static inline void VkMul(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in_1, VkContainer* in_2, VkContainer* temp) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
#if(VKFFT_BACKEND == 2)
		if ((in_1->type > 100) && (in_2->type > 100)) {
			//packed instructions workaround if all values are in registers
			if (((in_1->type % 10) != 3) || ((in_2->type % 10) != 3)) {
				sc->tempLen = sprintf(sc->tempStr, "%s", out->data.s);
				VkAppendLine(sc);
				sc->tempLen = sprintf(sc->tempStr, " = ");
				VkAppendLine(sc);
				VkAppendConversionStart(sc, out, in_1);
				sc->tempLen = sprintf(sc->tempStr, "%s", in_1->data.s);
				VkAppendLine(sc);
				VkAppendConversionEnd(sc, out, in_1);
				sc->tempLen = sprintf(sc->tempStr, " * ");
				VkAppendLine(sc);
				VkAppendConversionStart(sc, out, in_2);
				sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.s);
				VkAppendLine(sc);
				VkAppendConversionEnd(sc, out, in_2);
				sc->tempLen = sprintf(sc->tempStr, ";\n");
				VkAppendLine(sc);
				return;
			}
			else {
				if ((((out->type % 100) / 10) < 2) && (out->type == in_1->type) && (out->type == in_2->type)) {
					if ((strcmp(out->data.s, in_1->data.s)) && (strcmp(out->data.s, in_2->data.s))) {
						VkMov_x_Neg_y(sc, out, in_1);
						VkMov_y_x(sc, out, in_1);
						sc->tempLen = sprintf(sc->tempStr, "%s", out->data.s);
						VkAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, " = ");
						VkAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "%s", out->data.s);
						VkAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, " * ");
						VkAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "%s.y", in_2->data.s);
						VkAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, ";\n");
						VkAppendLine(sc);
						
						sc->tempLen = sprintf(sc->tempStr, "%s", out->data.s);
						VkAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, " = ");
						VkAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "%s", in_1->data.s);
						VkAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, " * ");
						VkAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "%s.x", in_2->data.s);
						VkAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, " + ");
						VkAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "%s", out->data.s);
						VkAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, ";\n");
						VkAppendLine(sc);
					}
					else {
						VkMov_x_Neg_y(sc, temp, in_1);
						VkMov_y_x(sc, temp, in_1);
						sc->tempLen = sprintf(sc->tempStr, "%s", temp->data.s);
						VkAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, " = ");
						VkAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "%s", temp->data.s);
						VkAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, " * ");
						VkAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "%s.y", in_2->data.s);
						VkAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, ";\n");
						VkAppendLine(sc);

						sc->tempLen = sprintf(sc->tempStr, "%s", out->data.s);
						VkAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, " = ");
						VkAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "%s", in_1->data.s);
						VkAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, " * ");
						VkAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "%s.x", in_2->data.s);
						VkAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, " + ");
						VkAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "%s", temp->data.s);
						VkAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, ";\n");
						VkAppendLine(sc);
					}
					return;
				}
			}
		}
#endif
		switch (out->type % 10) {
		case 1: case 2:
			sc->tempLen = sprintf(sc->tempStr, "%s", out->data.s);
			VkAppendLine(sc);
			break;
		case 3:
			if ((in_1->type < 100) || (in_2->type < 100) || ((in_1->type % 10) != 3) || ((in_2->type % 10) != 3) || ((strcmp(out->data.s, in_1->data.s)) && (strcmp(out->data.s, in_2->data.s)))) {
				sc->tempLen = sprintf(sc->tempStr, "%s.x", out->data.s);
			}
			else {
				sc->tempLen = sprintf(sc->tempStr, "%s.x", temp->data.s);
			}
			VkAppendLine(sc);
			break;
		}
		sc->tempLen = sprintf(sc->tempStr, " = ");
		VkAppendLine(sc);
		if ((in_1->type < 100) && (in_2->type < 100)) {
			switch (in_1->type % 10) {
			case 1:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i * in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i * in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i * in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				break;
			case 2:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * (long double)in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				break;
			case 3:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] * (long double)in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] * in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] * in_2->data.c[0] - in_1->data.c[1] * in_2->data.c[1]);
					VkAppendLine(sc);
					break;
				}
				break;
			}
			VkAppendNumberLiteral(sc, out);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
		}
		else {
			VkAppendConversionStart(sc, out, in_1);
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in_1->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.x", in_1->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, " * ");
			VkAppendLine(sc);
			VkAppendConversionStart(sc, out, in_2);
			if (in_2->type > 100) {
				switch (in_2->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.x", in_2->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_2);
			if (((in_1->type % 10) == 3) && ((in_2->type % 10) == 3)) {
				sc->tempLen = sprintf(sc->tempStr, " - ");
				VkAppendLine(sc);
				VkAppendConversionStart(sc, out, in_1);
				if (in_1->type > 100) {
					sc->tempLen = sprintf(sc->tempStr, "%s.y", in_1->data.s);
					VkAppendLine(sc);
				}
				else {
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1]);
					VkAppendLine(sc);
					VkAppendNumberLiteral(sc, out);
				}
				VkAppendConversionEnd(sc, out, in_1);
				sc->tempLen = sprintf(sc->tempStr, " * ");
				VkAppendLine(sc);
				VkAppendConversionStart(sc, out, in_2);
				if (in_2->type > 100) {
					sc->tempLen = sprintf(sc->tempStr, "%s.y", in_2->data.s);
					VkAppendLine(sc);
				}
				else {
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.c[1]);
					VkAppendLine(sc);
					VkAppendNumberLiteral(sc, out);
				}
				VkAppendConversionEnd(sc, out, in_2);
			}
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
		}
		switch (out->type % 10) {
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.y", out->data.s);
			/*if ((in_1->type < 100) || (in_2->type < 100) || ((in_1->type % 10) != 3) || ((in_2->type % 10) != 3) || ((strcmp(out->data.s, in_1->data.s)) && (strcmp(out->data.s, in_2->data.s)))) {
				sc->tempLen = sprintf(sc->tempStr, "%s.y", out->data.s);
			}
			else {
				sc->tempLen = sprintf(sc->tempStr, "%s.y", temp->data.s);
			}*/
			VkAppendLine(sc);
			sc->tempLen = sprintf(sc->tempStr, " = ");
			VkAppendLine(sc);
			if ((in_1->type < 100) && (in_2->type < 100)) {
				switch (in_1->type % 10) {
				case 1:
					switch (in_2->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i * in_2->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i * in_2->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i * in_2->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					break;
				case 2:
					switch (in_2->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * (long double)in_2->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * in_2->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * in_2->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					break;
				case 3:
					switch (in_2->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] * (long double)in_2->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] * in_2->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] * in_2->data.c[1] + in_1->data.c[1] * in_2->data.c[0]);
						VkAppendLine(sc);
						break;
					}
					break;
				}
				VkAppendNumberLiteral(sc, out);
				sc->tempLen = sprintf(sc->tempStr, ";\n");
				VkAppendLine(sc);
			}
			else {
				VkAppendConversionStart(sc, out, in_1);
				if (in_1->type > 100) {
					switch (in_1->type % 10) {
					case 1: case 2:
						sc->tempLen = sprintf(sc->tempStr, "%s", in_1->data.s);
						VkAppendLine(sc);
						break;
					case 3:
						if ((in_2->type % 10) == 3)
							sc->tempLen = sprintf(sc->tempStr, "%s.x", in_1->data.s);
						else
							sc->tempLen = sprintf(sc->tempStr, "%s.y", in_1->data.s);
						VkAppendLine(sc);
						break;
					}
				}
				else {
					switch (in_1->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						if ((in_2->type % 10) == 3)
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0]);
						else
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					VkAppendNumberLiteral(sc, out);
				}
				VkAppendConversionEnd(sc, out, in_1);
				sc->tempLen = sprintf(sc->tempStr, " * ");
				VkAppendLine(sc);
				VkAppendConversionStart(sc, out, in_2);
				if (in_2->type > 100) {
					switch (in_2->type % 10) {
					case 1: case 2:
						sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.s);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%s.y", in_2->data.s);
						VkAppendLine(sc);
						break;
					}
				}
				else {
					switch (in_2->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_2->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					VkAppendNumberLiteral(sc, out);
				}
				VkAppendConversionEnd(sc, out, in_2);
				if (((in_1->type % 10) == 3) && ((in_2->type % 10) == 3)) {
					sc->tempLen = sprintf(sc->tempStr, " + ");
					VkAppendLine(sc);
					VkAppendConversionStart(sc, out, in_1);
					if (in_1->type > 100) {
						sc->tempLen = sprintf(sc->tempStr, "%s.y", in_1->data.s);
						VkAppendLine(sc);
					}
					else {
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1]);
						VkAppendLine(sc);
						VkAppendNumberLiteral(sc, out);
					}
					VkAppendConversionEnd(sc, out, in_1);
					sc->tempLen = sprintf(sc->tempStr, " * ");
					VkAppendLine(sc);
					VkAppendConversionStart(sc, out, in_2);
					if (in_2->type > 100) {
						sc->tempLen = sprintf(sc->tempStr, "%s.x", in_2->data.s);
						VkAppendLine(sc);
					}
					else {
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.c[0]);
						VkAppendLine(sc);
						VkAppendNumberLiteral(sc, out);
					}
					VkAppendConversionEnd(sc, out, in_2);
				}
				sc->tempLen = sprintf(sc->tempStr, ";\n");
				VkAppendLine(sc);
			}
			if ((in_1->type < 100) || (in_2->type < 100) || ((in_1->type % 10) != 3) || ((in_2->type % 10) != 3) || ((strcmp(out->data.s, in_1->data.s)) && (strcmp(out->data.s, in_2->data.s)))) {
			}
			else {
				VkMov_x(sc, out, temp);
			}
			break;
		}

		return;
	}
	else {
		switch (out->type % 10) {
		case 1:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = in_1->data.i * in_2->data.i;
							return;
						case 2:
							out->data.i = in_1->data.i * in_2->data.d;
							return;
						case 3:
							break;
						}
					}
					break;
				case 2:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = (int64_t)(in_1->data.d * in_2->data.i);
							return;
						case 2:
							out->data.i = (int64_t)(in_1->data.d * in_2->data.d);
							return;
						case 3:
							break;
						}
					}
					break;
				case 3:
					break;
				}
			}
			break;
		case 2:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.d = (long double)(in_1->data.i * in_2->data.i);
							return;
						case 2:
							out->data.d = (long double)in_1->data.i * in_2->data.d;
							return;
						case 3:
							break;
						}
					}
					break;
				case 2:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.d = in_1->data.d * (long double)in_2->data.i;
							return;
						case 2:
							out->data.d = in_1->data.d * in_2->data.d;
							return;
						case 3:
							break;
						}
					}
					break;
				case 3:
					break;
				}
			}
			break;
		case 3:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
				case 2:
					break;
				case 3:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.c[0] = in_1->data.c[0] * (long double)in_2->data.i;
							out->data.c[1] = in_1->data.c[1] * (long double)in_2->data.i;
							return;
						case 2:
							out->data.c[0] = in_1->data.c[0] * in_2->data.d;
							out->data.c[1] = in_1->data.c[1] * in_2->data.d;
							return;
						case 3:
							out->data.c[0] = in_1->data.c[0] * in_2->data.c[0] - in_1->data.c[1] * in_2->data.c[1];
							out->data.c[1] = in_1->data.c[1] * in_2->data.c[0] + in_1->data.c[0] * in_2->data.c[1];
							return;
						}
					}
					break;
				}
			}
			break;
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}

static inline void VkMul_x(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in_1, VkContainer* in_2, VkContainer* temp) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
		switch (out->type % 10) {
		case 1: case 2:
			sc->tempLen = sprintf(sc->tempStr, "%s", out->data.s);
			VkAppendLine(sc);
			break;
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.x", out->data.s);
			VkAppendLine(sc);
			break;
		}
		sc->tempLen = sprintf(sc->tempStr, " = ");
		VkAppendLine(sc);
		if ((in_1->type < 100) && (in_2->type < 100)) {
			switch (in_1->type % 10) {
			case 1:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i * in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i * in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i * in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				break;
			case 2:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * (long double)in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				break;
			case 3:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] * (long double)in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] * in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] * in_2->data.c[0] - in_1->data.c[1] * in_2->data.c[1]);
					VkAppendLine(sc);
					break;
				}
				break;
			}
			VkAppendNumberLiteral(sc, out);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
		}
		else {
			VkAppendConversionStart(sc, out, in_1);
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in_1->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.x", in_1->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, " * ");
			VkAppendLine(sc);
			VkAppendConversionStart(sc, out, in_2);
			if (in_2->type > 100) {
				switch (in_2->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.x", in_2->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_2);
			if (((in_1->type % 10) == 3) && ((in_2->type % 10) == 3)) {
				sc->tempLen = sprintf(sc->tempStr, " - ");
				VkAppendLine(sc);
				VkAppendConversionStart(sc, out, in_1);
				if (in_1->type > 100) {
					sc->tempLen = sprintf(sc->tempStr, "%s.y", in_1->data.s);
					VkAppendLine(sc);
				}
				else {
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1]);
					VkAppendLine(sc);
					VkAppendNumberLiteral(sc, out);
				}
				VkAppendConversionEnd(sc, out, in_1);
				sc->tempLen = sprintf(sc->tempStr, " * ");
				VkAppendLine(sc);
				VkAppendConversionStart(sc, out, in_2);
				if (in_2->type > 100) {
					sc->tempLen = sprintf(sc->tempStr, "%s.y", in_2->data.s);
					VkAppendLine(sc);
				}
				else {
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.c[1]);
					VkAppendLine(sc);
					VkAppendNumberLiteral(sc, out);
				}
				VkAppendConversionEnd(sc, out, in_2);
			}
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
		}

		return;
	}
	else {
		switch (out->type % 10) {
		case 1:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = in_1->data.i * in_2->data.i;
							return;
						case 2:
							out->data.i = in_1->data.i * in_2->data.d;
							return;
						case 3:
							break;
						}
					}
					break;
				case 2:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = (int64_t)(in_1->data.d * in_2->data.i);
							return;
						case 2:
							out->data.i = (int64_t)(in_1->data.d * in_2->data.d);
							return;
						case 3:
							break;
						}
					}
					break;
				case 3:
					break;
				}
			}
			break;
		case 2:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.d = (long double)(in_1->data.i * in_2->data.i);
							return;
						case 2:
							out->data.d = (long double)in_1->data.i * in_2->data.d;
							return;
						case 3:
							break;
						}
					}
					break;
				case 2:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.d = in_1->data.d * (long double)in_2->data.i;
							return;
						case 2:
							out->data.d = in_1->data.d * in_2->data.d;
							return;
						case 3:
							break;
						}
					}
					break;
				case 3:
					break;
				}
			}
			break;
		case 3:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
				case 2:
					break;
				case 3:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.c[0] = in_1->data.c[0] * (long double)in_2->data.i;
							return;
						case 2:
							out->data.c[0] = in_1->data.c[0] * in_2->data.d;
							return;
						case 3:
							out->data.c[0] = in_1->data.c[0] * in_2->data.c[0] - in_1->data.c[1] * in_2->data.c[1];
							return;
						}
					}
					break;
				}
			}
			break;
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}

static inline void VkMul_y(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in_1, VkContainer* in_2, VkContainer* temp) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
		switch (out->type % 10) {
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.y", out->data.s);
			/*if ((in_1->type < 100) || (in_2->type < 100) || ((in_1->type % 10) != 3) || ((in_2->type % 10) != 3) || ((strcmp(out->data.s, in_1->data.s)) && (strcmp(out->data.s, in_2->data.s)))) {
				sc->tempLen = sprintf(sc->tempStr, "%s.y", out->data.s);
			}
			else {
				sc->tempLen = sprintf(sc->tempStr, "%s.y", temp->data.s);
			}*/
			VkAppendLine(sc);
			sc->tempLen = sprintf(sc->tempStr, " = ");
			VkAppendLine(sc);
			if ((in_1->type < 100) && (in_2->type < 100)) {
				switch (in_1->type % 10) {
				case 1:
					switch (in_2->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i * in_2->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i * in_2->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i * in_2->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					break;
				case 2:
					switch (in_2->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * (long double)in_2->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * in_2->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d * in_2->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					break;
				case 3:
					switch (in_2->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] * (long double)in_2->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] * in_2->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] * in_2->data.c[1] + in_1->data.c[1] * in_2->data.c[0]);
						VkAppendLine(sc);
						break;
					}
					break;
				}
				VkAppendNumberLiteral(sc, out);
				sc->tempLen = sprintf(sc->tempStr, ";\n");
				VkAppendLine(sc);
			}
			else {
				VkAppendConversionStart(sc, out, in_1);
				if (in_1->type > 100) {
					switch (in_1->type % 10) {
					case 1: case 2:
						sc->tempLen = sprintf(sc->tempStr, "%s", in_1->data.s);
						VkAppendLine(sc);
						break;
					case 3:
						if ((in_2->type % 10) == 3)
							sc->tempLen = sprintf(sc->tempStr, "%s.x", in_1->data.s);
						else
							sc->tempLen = sprintf(sc->tempStr, "%s.y", in_1->data.s);
						VkAppendLine(sc);
						break;
					}
				}
				else {
					switch (in_1->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						if ((in_2->type % 10) == 3)
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0]);
						else
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					VkAppendNumberLiteral(sc, out);
				}
				VkAppendConversionEnd(sc, out, in_1);
				sc->tempLen = sprintf(sc->tempStr, " * ");
				VkAppendLine(sc);
				VkAppendConversionStart(sc, out, in_2);
				if (in_2->type > 100) {
					switch (in_2->type % 10) {
					case 1: case 2:
						sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.s);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%s.y", in_2->data.s);
						VkAppendLine(sc);
						break;
					}
				}
				else {
					switch (in_2->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_2->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					VkAppendNumberLiteral(sc, out);
				}
				VkAppendConversionEnd(sc, out, in_2);
				if (((in_1->type % 10) == 3) && ((in_2->type % 10) == 3)) {
					sc->tempLen = sprintf(sc->tempStr, " + ");
					VkAppendLine(sc);
					VkAppendConversionStart(sc, out, in_1);
					if (in_1->type > 100) {
						sc->tempLen = sprintf(sc->tempStr, "%s.y", in_1->data.s);
						VkAppendLine(sc);
					}
					else {
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1]);
						VkAppendLine(sc);
						VkAppendNumberLiteral(sc, out);
					}
					VkAppendConversionEnd(sc, out, in_1);
					sc->tempLen = sprintf(sc->tempStr, " * ");
					VkAppendLine(sc);
					VkAppendConversionStart(sc, out, in_2);
					if (in_2->type > 100) {
						sc->tempLen = sprintf(sc->tempStr, "%s.x", in_2->data.s);
						VkAppendLine(sc);
					}
					else {
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.c[0]);
						VkAppendLine(sc);
						VkAppendNumberLiteral(sc, out);
					}
					VkAppendConversionEnd(sc, out, in_2);
				}
				sc->tempLen = sprintf(sc->tempStr, ";\n");
				VkAppendLine(sc);
			}
			break;
		}

		return;
	}
	else {
		switch (out->type % 10) {
		case 1:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = in_1->data.i * in_2->data.i;
							return;
						case 2:
							out->data.i = in_1->data.i * in_2->data.d;
							return;
						case 3:
							break;
						}
					}
					break;
				case 2:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = (int64_t)(in_1->data.d * in_2->data.i);
							return;
						case 2:
							out->data.i = (int64_t)(in_1->data.d * in_2->data.d);
							return;
						case 3:
							break;
						}
					}
					break;
				case 3:
					break;
				}
			}
			break;
		case 2:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.d = (long double)(in_1->data.i * in_2->data.i);
							return;
						case 2:
							out->data.d = (long double)in_1->data.i * in_2->data.d;
							return;
						case 3:
							break;
						}
					}
					break;
				case 2:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.d = in_1->data.d * (long double)in_2->data.i;
							return;
						case 2:
							out->data.d = in_1->data.d * in_2->data.d;
							return;
						case 3:
							break;
						}
					}
					break;
				case 3:
					break;
				}
			}
			break;
		case 3:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
				case 2:
					break;
				case 3:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.c[1] = in_1->data.c[1] * (long double)in_2->data.i;
							return;
						case 2:
							out->data.c[1] = in_1->data.c[1] * in_2->data.d;
							return;
						case 3:
							out->data.c[1] = in_1->data.c[1] * in_2->data.c[0] + in_1->data.c[0] * in_2->data.c[1];
							return;
						}
					}
					break;
				}
			}
			break;
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}

static inline void VkFMA3(VkFFTSpecializationConstantsLayout* sc, VkContainer* out_1, VkContainer* out_2, VkContainer* in_1, VkContainer* in_num, VkContainer* in_conj) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out_1->type > 100) {
		//in_1 has to be same type as out
		switch (out_1->type % 10) {
		case 1:
		case 2:
			break;
		case 3:
			switch (in_num->type % 10) {
			case 1:
				break;
			case 2:
				break;
			case 3:
				sc->tempLen = sprintf(sc->tempStr, "\
%s.x = fma(%s.x, %s.x, %s.x);\n\
%s.y = fma(%s.y, %s.x, %s.y);\n", out_1->data.s, in_1->data.s, in_num->data.s, out_1->data.s, out_1->data.s, in_conj->data.s, in_num->data.s, out_1->data.s);
				VkAppendLine(sc);
				sc->tempLen = sprintf(sc->tempStr, "\
%s.x = fma(%s.y, %s.y, %s.x);\n\
%s.y = fma(%s.x, %s.y, %s.y);\n", out_2->data.s, in_1->data.s, in_num->data.s, out_2->data.s, out_2->data.s, in_conj->data.s, in_num->data.s, out_2->data.s);
				VkAppendLine(sc);
				return;
			}
			break;
		}
	}
	else {
		switch (out_1->type % 10) {
		case 1:
			break;
		case 2:
			break;
		case 3:
			switch (in_num->type % 10) {
			case 1:
				break;
			case 2:
				break;
			case 3:
				out_1->data.c[0] = in_1->data.c[0] * in_num->data.c[0] + out_1->data.c[0];
				out_1->data.c[1] = in_conj->data.c[1] * in_num->data.c[0] + out_1->data.c[1];
				out_2->data.c[0] = in_1->data.c[1] * in_num->data.c[1] + out_2->data.c[0];
				out_2->data.c[1] = in_conj->data.c[0] * in_num->data.c[1] + out_2->data.c[1];
				return;
			}
			break;
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}
static inline void VkFMA3_const_w(VkFFTSpecializationConstantsLayout* sc, VkContainer* out_1, VkContainer* out_2, VkContainer* in_1, VkContainer* in_num_x, VkContainer* in_num_y, VkContainer* in_conj, VkContainer* temp) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out_1->type > 100) {
#if(VKFFT_BACKEND==2)
		if (((out_1->type%100)/10) < 2) {
			VkMov_x(sc, temp, in_1);
			VkMov_y(sc, temp, in_conj);
			VkFMA(sc, out_1, temp, in_num_x, out_1);

			VkMov_x_y(sc, temp, in_1);
			VkMov_y_x(sc, temp, in_conj);
			VkFMA(sc, out_2, temp, in_num_y, out_2);
			return;
		}
#endif
		//in_1 has to be same type as out
		switch (out_1->type % 10) {
		case 1:
		case 2:
			break;
		case 3:
			switch (in_num_x->type % 10) {
			case 1:
				break;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "%s.x = fma(%s.x, %.17Le", out_1->data.s, in_1->data.s, in_num_x->data.d);
				VkAppendLine(sc);
				VkAppendNumberLiteral(sc, out_1);
				sc->tempLen = sprintf(sc->tempStr, ", %s.x);\n", out_1->data.s);
				VkAppendLine(sc);
				sc->tempLen = sprintf(sc->tempStr, "%s.y = fma(%s.y, %.17Le", out_1->data.s, in_conj->data.s, in_num_x->data.d);
				VkAppendLine(sc);
				VkAppendNumberLiteral(sc, out_1);
				sc->tempLen = sprintf(sc->tempStr, ", %s.y);\n", out_1->data.s);
				VkAppendLine(sc);
				sc->tempLen = sprintf(sc->tempStr, "%s.x = fma(%s.y, %.17Le", out_2->data.s, in_1->data.s, in_num_y->data.d);
				VkAppendLine(sc);
				VkAppendNumberLiteral(sc, out_1);
				sc->tempLen = sprintf(sc->tempStr, ", %s.x);\n", out_2->data.s);
				VkAppendLine(sc);
				sc->tempLen = sprintf(sc->tempStr, "%s.y = fma(%s.x, %.17Le", out_2->data.s, in_conj->data.s, in_num_y->data.d);
				VkAppendLine(sc);
				VkAppendNumberLiteral(sc, out_1);
				sc->tempLen = sprintf(sc->tempStr, ", %s.y);\n", out_2->data.s);
				VkAppendLine(sc);
				return;
			case 3:
				break;
			}
			break;
		}
	}
	else {
		switch (out_1->type % 10) {
		case 1:
			break;
		case 2:
			break;
		case 3:
			switch (in_num_x->type % 10) {
			case 1:
				break;
			case 2:
				out_1->data.c[0] = in_1->data.c[0] * in_num_x->data.d + out_1->data.c[0];
				out_1->data.c[1] = in_conj->data.c[1] * in_num_x->data.d + out_1->data.c[1];
				out_2->data.c[0] = in_1->data.c[1] * in_num_y->data.d + out_2->data.c[0];
				out_2->data.c[1] = in_conj->data.c[0] * in_num_y->data.d + out_2->data.c[1];
				return;
			case 3:
				break;
			}
			break;
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}

static inline void VkDiv(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in_1, VkContainer* in_2) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
		switch (out->type % 10) {
		case 1: case 2:
			sc->tempLen = sprintf(sc->tempStr, "%s", out->data.s);
			VkAppendLine(sc);
			break;
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.x", out->data.s);
			VkAppendLine(sc);
			break;
		}
		sc->tempLen = sprintf(sc->tempStr, " = ");
		VkAppendLine(sc);
		if ((in_1->type < 100) && (in_2->type < 100)) {
			switch (in_1->type % 10) {
			case 1:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i / in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i / in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i / in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				break;
			case 2:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d / (long double)in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d / in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d / in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				break;
			case 3:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] / (long double)in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] / in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->res = VKFFT_ERROR_MATH_FAILED; 
					break;
				}
				break;
			}
			VkAppendNumberLiteral(sc, out);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
		}
		else {
			VkAppendConversionStart(sc, out, in_1);
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in_1->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.x", in_1->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, " / ");
			VkAppendLine(sc);
			VkAppendConversionStart(sc, out, in_2);
			if (in_2->type > 100) {
				switch (in_2->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.x", in_2->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_2);
			if (((in_1->type % 10) == 3) && ((in_2->type % 10) == 3)) {
				sc->res = VKFFT_ERROR_MATH_FAILED;
			}
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
		}
		switch (out->type % 10) {
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.y", out->data.s);
			VkAppendLine(sc);
			sc->tempLen = sprintf(sc->tempStr, " = ");
			VkAppendLine(sc);
			if ((in_1->type < 100) && (in_2->type < 100)) {
				switch (in_1->type % 10) {
				case 1:
					switch (in_2->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i / in_2->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i / in_2->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)in_1->data.i / in_2->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					break;
				case 2:
					switch (in_2->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d / (long double)in_2->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d / in_2->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d / in_2->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					break;
				case 3:
					switch (in_2->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] / (long double)in_2->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] / in_2->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->res = VKFFT_ERROR_MATH_FAILED; 
						break;
					}
					break;
				}
				VkAppendNumberLiteral(sc, out);
				sc->tempLen = sprintf(sc->tempStr, ";\n");
				VkAppendLine(sc);
			}
			else {
				VkAppendConversionStart(sc, out, in_1);
				if (in_1->type > 100) {
					switch (in_1->type % 10) {
					case 1: case 2:
						sc->tempLen = sprintf(sc->tempStr, "%s", in_1->data.s);
						VkAppendLine(sc);
						break;
					case 3:
						if ((in_2->type % 10) == 3)
							sc->tempLen = sprintf(sc->tempStr, "%s.x", in_1->data.s);
						else
							sc->tempLen = sprintf(sc->tempStr, "%s.y", in_1->data.s);
						VkAppendLine(sc);
						break;
					}
				}
				else {
					switch (in_1->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						if ((in_2->type % 10) == 3)
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0]);
						else
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					VkAppendNumberLiteral(sc, out);
				}
				VkAppendConversionEnd(sc, out, in_1);
				sc->tempLen = sprintf(sc->tempStr, " / ");
				VkAppendLine(sc);
				VkAppendConversionStart(sc, out, in_2);
				if (in_2->type > 100) {
					switch (in_2->type % 10) {
					case 1: case 2:
						sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.s);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%s.y", in_2->data.s);
						VkAppendLine(sc);
						break;
					}
				}
				else {
					switch (in_2->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_2->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					VkAppendNumberLiteral(sc, out);
				}
				VkAppendConversionEnd(sc, out, in_2);
				if (((in_1->type % 10) == 3) && ((in_2->type % 10) == 3)) {
					sc->res = VKFFT_ERROR_MATH_FAILED;
				}
				sc->tempLen = sprintf(sc->tempStr, ";\n");
				VkAppendLine(sc);
			}
			break;
		}
		return;
	}
	else {
		switch (out->type % 10) {
		case 1:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = in_1->data.i / in_2->data.i;
							return;
						case 2:
							out->data.i = in_1->data.i / in_2->data.d;
							return;
						case 3:
							break;
						}
					}
					break;
				case 2:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = (int64_t)(in_1->data.d / in_2->data.i);
							return;
						case 2:
							out->data.i = (int64_t)(in_1->data.d / in_2->data.d);
							return;
						case 3:
							break;
						}
					}
					break;
				case 3:
					break;
				}
			}
			break;
		case 2:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.d = (long double)(in_1->data.i / in_2->data.i);
							return;
						case 2:
							out->data.d = (long double)in_1->data.i / in_2->data.d;
							return;
						case 3:
							break;
						}
					}
					break;
				case 2:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.d = in_1->data.d / (long double)in_2->data.i;
							return;
						case 2:
							out->data.d = in_1->data.d / in_2->data.d;
							return;
						case 3:
							break;
						}
					}
					break;
				case 3:
					break;
				}
			}
			break;
		case 3:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
				case 2:
					break;
				case 3:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.c[0] = in_1->data.c[0] / (long double)in_2->data.i;
							out->data.c[1] = in_1->data.c[1] / (long double)in_2->data.i;
							return;
						case 2:
							out->data.c[0] = in_1->data.c[0] / in_2->data.d;
							out->data.c[1] = in_1->data.c[1] / in_2->data.d;
							return;
						case 3:
							sc->res = VKFFT_ERROR_MATH_FAILED; 
							return;
						}
					}
					break;
				}
			}
			break;
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}
static inline void VkDivCeil(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in_1, VkContainer* in_2) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
		switch (out->type % 10) {
		case 1: case 2:
			sc->tempLen = sprintf(sc->tempStr, "%s", out->data.s);
			VkAppendLine(sc);
			break;
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.x", out->data.s);
			VkAppendLine(sc);
			break;
		}
		sc->tempLen = sprintf(sc->tempStr, " = ");
		VkAppendLine(sc);
		if ((in_1->type < 100) && (in_2->type < 100)) {
			switch (in_1->type % 10) {
			case 1:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", (int64_t)ceil(in_1->data.i / (long double)in_2->data.i));
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", ceil((long double)in_1->data.i / in_2->data.d));
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", ceil((long double)in_1->data.i / in_2->data.c[0]));
					VkAppendLine(sc);
					break;
				}
				break;
			case 2:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", ceil(in_1->data.d / (long double)in_2->data.i));
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", ceil(in_1->data.d / in_2->data.d));
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", ceil(in_1->data.d / in_2->data.c[0]));
					VkAppendLine(sc);
					break;
				}
				break;
			case 3:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", ceil(in_1->data.c[0] / (long double)in_2->data.i));
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", ceil(in_1->data.c[0] / in_2->data.d));
					VkAppendLine(sc);
					break;
				case 3:
					sc->res = VKFFT_ERROR_MATH_FAILED;
					break;
				}
				break;
			}
			VkAppendNumberLiteral(sc, out);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
		}
		else {
			sc->tempLen = sprintf(sc->tempStr, "ceil(");
			VkAppendLine(sc);
			VkAppendConversionStart(sc, out, in_1);
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in_1->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.x", in_1->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, " / ");
			VkAppendLine(sc);
			VkAppendConversionStart(sc, out, in_2);
			if (in_2->type > 100) {
				switch (in_2->type % 10) {
				case 1: case 2:
					sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.s);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.x", in_2->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_2->data.i);
					VkAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.d);
					VkAppendLine(sc);
					break;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_2);
			if (((in_1->type % 10) == 3) && ((in_2->type % 10) == 3)) {
				sc->res = VKFFT_ERROR_MATH_FAILED;
			}
			sc->tempLen = sprintf(sc->tempStr, ");\n");
			VkAppendLine(sc);
		}
		switch (out->type % 10) {
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s.y", out->data.s);
			VkAppendLine(sc);
			sc->tempLen = sprintf(sc->tempStr, " = ");
			VkAppendLine(sc);
			if ((in_1->type < 100) && (in_2->type < 100)) {
				switch (in_1->type % 10) {
				case 1:
					switch (in_2->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", (int64_t)ceil(in_1->data.i / (long double)in_2->data.i));
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", ceil((long double)in_1->data.i / in_2->data.d));
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", ceil((long double)in_1->data.i / in_2->data.c[1]));
						VkAppendLine(sc);
						break;
					}
					break;
				case 2:
					switch (in_2->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", ceil(in_1->data.d / (long double)in_2->data.i));
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", ceil(in_1->data.d / in_2->data.d));
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", ceil(in_1->data.d / in_2->data.c[1]));
						VkAppendLine(sc);
						break;
					}
					break;
				case 3:
					switch (in_2->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", ceil(in_1->data.c[1] / (long double)in_2->data.i));
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", ceil(in_1->data.c[1] / in_2->data.d));
						VkAppendLine(sc);
						break;
					case 3:
						sc->res = VKFFT_ERROR_MATH_FAILED;
						break;
					}
					break;
				}
				VkAppendNumberLiteral(sc, out);
				sc->tempLen = sprintf(sc->tempStr, ";\n");
				VkAppendLine(sc);
			}
			else {
				sc->tempLen = sprintf(sc->tempStr, "ceil(");
				VkAppendLine(sc);
				VkAppendConversionStart(sc, out, in_1);
				if (in_1->type > 100) {
					switch (in_1->type % 10) {
					case 1: case 2:
						sc->tempLen = sprintf(sc->tempStr, "%s", in_1->data.s);
						VkAppendLine(sc);
						break;
					case 3:
						if ((in_2->type % 10) == 3)
							sc->tempLen = sprintf(sc->tempStr, "%s.x", in_1->data.s);
						else
							sc->tempLen = sprintf(sc->tempStr, "%s.y", in_1->data.s);
						VkAppendLine(sc);
						break;
					}
				}
				else {
					switch (in_1->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						if ((in_2->type % 10) == 3)
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0]);
						else
							sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					VkAppendNumberLiteral(sc, out);
				}
				VkAppendConversionEnd(sc, out, in_1);
				sc->tempLen = sprintf(sc->tempStr, " / ");
				VkAppendLine(sc);
				VkAppendConversionStart(sc, out, in_2);
				if (in_2->type > 100) {
					switch (in_2->type % 10) {
					case 1: case 2:
						sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.s);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%s.y", in_2->data.s);
						VkAppendLine(sc);
						break;
					}
				}
				else {
					switch (in_2->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_2->data.i);
						VkAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.d);
						VkAppendLine(sc);
						break;
					case 3:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.c[1]);
						VkAppendLine(sc);
						break;
					}
					VkAppendNumberLiteral(sc, out);
				}
				VkAppendConversionEnd(sc, out, in_2);
				if (((in_1->type % 10) == 3) && ((in_2->type % 10) == 3)) {
					sc->res = VKFFT_ERROR_MATH_FAILED;
				}
				sc->tempLen = sprintf(sc->tempStr, ");\n");
				VkAppendLine(sc);
			}
			break;
		}
		return;
	}
	else {
		switch (out->type % 10) {
		case 1:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = in_1->data.i / in_2->data.i + (in_1->data.i % in_2->data.i != 0);
							return;
						case 2:
							out->data.i = (int64_t)ceil(in_1->data.i / in_2->data.d);
							return;
						case 3:
							break;
						}
					}
					break;
				case 2:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = (int64_t)ceil(in_1->data.d / in_2->data.i);
							return;
						case 2:
							out->data.i = (int64_t)ceil(in_1->data.d / in_2->data.d);
							return;
						case 3:
							break;
						}
					}
					break;
				case 3:
					break;
				}
			}
			break;
		case 2:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.d = (long double)(in_1->data.i / in_2->data.i + (in_1->data.i % in_2->data.i != 0));
							return;
						case 2:
							out->data.d = (long double)ceil(in_1->data.i / in_2->data.d);
							return;
						case 3:
							break;
						}
					}
					break;
				case 2:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.d = ceil(in_1->data.d / in_2->data.i);
							return;
						case 2:
							out->data.d = ceil(in_1->data.d / in_2->data.d);
							return;
						case 3:
							break;
						}
					}
					break;
				case 3:
					break;
				}
			}
			break;
		case 3:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
				case 2:
					break;
				case 3:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.c[0] = ceil(in_1->data.c[0] / in_2->data.i);
							out->data.c[1] = ceil(in_1->data.c[1] / in_2->data.i);
							return;
						case 2:
							out->data.c[0] = ceil(in_1->data.c[0] / in_2->data.d);
							out->data.c[1] = ceil(in_1->data.c[1] / in_2->data.d);
							return;
						case 3:
							break;
						}
					}
					break;
				}
			}
			break;
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}

static inline void VkMod(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in_1, VkContainer* in_2) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
		//in_1 has to be same type as out
		switch (out->type % 10) {
		case 1:
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
						switch (in_2->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "\
%s = %s %% %s;\n", out->data.s, in_1->data.s, in_2->data.s);
							VkAppendLine(sc);
							return;
						case 2:
							break;
						case 3:
							break;
						}
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "\
%s = %s %% %" PRIi64 ";\n", out->data.s, in_1->data.s, in_2->data.i);
							VkAppendLine(sc);
							return;
						case 2:
							break;
						case 3:
							break;
						}
					}
					break;
				case 2:
					break;
				case 3:
					break;
				}
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
						switch (in_2->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "\
%s = %" PRIi64 " %% %s;\n", out->data.s, in_1->data.i, in_2->data.s);
							VkAppendLine(sc);
							return;
						case 2:
							break;
						case 3:
							break;
						}
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "\
%s = %" PRIi64 ";\n", out->data.s, in_1->data.i % in_2->data.i);
							VkAppendLine(sc);
							return;
						case 2:
							break;
						case 3:
							break;
						}
					}
					break;
				case 2:
					break;
				case 3:
					break;
				}
			}
		break;
		case 2:
			break;
		case 3:
			break;
		}
	}
	else {
		switch (out->type % 10) {
		case 1:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = in_1->data.i % in_2->data.i;
							return;
						}
					}
				break;
				}
			}
			break;
		case 2:
			break;
		case 3:
			break;
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}

static inline void VkAnd(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in_1, VkContainer* in_2) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
		//in_1 has to be same type as out
		switch (out->type % 10) {
		case 1:
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
						switch (in_2->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "\
%s = %s && %s;\n", out->data.s, in_1->data.s, in_2->data.s);
							VkAppendLine(sc);
							return;
						case 2:
							break;
						case 3:
							break;
						}
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "\
%s = %s && %" PRIi64 ";\n", out->data.s, in_1->data.s, in_2->data.i);
							VkAppendLine(sc);
							return;
						case 2:
							break;
						case 3:
							break;
						}
					}
					break;
				case 2:
					break;
				case 3:
					break;
				}
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
						switch (in_2->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "\
%s = %" PRIi64 " && %s;\n", out->data.s, in_1->data.i, in_2->data.s);
							VkAppendLine(sc);
							return;
						case 2:
							break;
						case 3:
							break;
						}
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "\
%s = %d;\n", out->data.s, in_1->data.i && in_2->data.i);
							VkAppendLine(sc);
							return;
						case 2:
							break;
						case 3:
							break;
						}
					}
					break;
				case 2:
					break;
				case 3:
					break;
				}
			}
			break;
		case 2:
			break;
		case 3:
			break;
		}
	}
	else {
		switch (out->type % 10) {
		case 1:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = in_1->data.i && in_2->data.i;
							return;
						}
					}
					break;
				}
			}
			break;
		case 2:
			break;
		case 3:
			break;
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}
static inline void VkOr(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in_1, VkContainer* in_2) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
		//in_1 has to be same type as out
		switch (out->type % 10) {
		case 1:
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
						switch (in_2->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "\
%s = %s || %s;\n", out->data.s, in_1->data.s, in_2->data.s);
							VkAppendLine(sc);
							return;
						case 2:
							break;
						case 3:
							break;
						}
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "\
%s = %s || %" PRIi64 ";\n", out->data.s, in_1->data.s, in_2->data.i);
							VkAppendLine(sc);
							return;
						case 2:
							break;
						case 3:
							break;
						}
					}
					break;
				case 2:
					break;
				case 3:
					break;
				}
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
						switch (in_2->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "\
%s = %" PRIi64 " || %s;\n", out->data.s, in_1->data.i, in_2->data.s);
							VkAppendLine(sc);
							return;
						case 2:
							break;
						case 3:
							break;
						}
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "\
%s = %d;\n", out->data.s, in_1->data.i || in_2->data.i);
							VkAppendLine(sc);
							return;
						case 2:
							break;
						case 3:
							break;
						}
					}
					break;
				case 2:
					break;
				case 3:
					break;
				}
			}
			break;
		case 2:
			break;
		case 3:
			break;
		}
	}
	else {
		switch (out->type % 10) {
		case 1:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = in_1->data.i || in_2->data.i;
							return;
						}
					}
					break;
				}
			}
			break;
		case 2:
			break;
		case 3:
			break;
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}


static inline void VkSinCos(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in_1) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
		//in_1 has to be same type as out
		switch (out->type % 10) {
		case 3:
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 2:
					switch ((out->type / 10) % 10) {
					case 0: case 1:
#if(VKFFT_BACKEND==0)
						sc->tempLen = sprintf(sc->tempStr, "\
%s.x = cos(%s);\n", out->data.s, in_1->data.s);
						VkAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "\
%s.y = sin(%s);\n", out->data.s, in_1->data.s);
						VkAppendLine(sc);
#elif ((VKFFT_BACKEND == 1) || (VKFFT_BACKEND == 2))
						sc->tempLen = sprintf(sc->tempStr, "\
__sincosf(%s, &%s.y, &%s.x);\n", in_1->data.s, out->data.s, out->data.s);
						VkAppendLine(sc);
#elif ((VKFFT_BACKEND == 3) || (VKFFT_BACKEND == 4))
						sc->tempLen = sprintf(sc->tempStr, "\
%s.x = native_cos(%s);\n", out->data.s, in_1->data.s);
						VkAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "\
%s.y = native_sin(%s);\n", out->data.s, in_1->data.s);
						VkAppendLine(sc);
#elif (VKFFT_BACKEND == 5)
						sc->tempLen = sprintf(sc->tempStr, "\
%s.x = cos(%s);\n", out->data.s, in_1->data.s);
						VkAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "\
%s.y = sin(%s);\n", out->data.s, in_1->data.s);
						VkAppendLine(sc);
#endif
						return;
					case 2:
#if(VKFFT_BACKEND==0)
						sc->tempLen = sprintf(sc->tempStr, "\
%s = sincos20(%s);\n", out->data.s, in_1->data.s);
						VkAppendLine(sc);
#elif ((VKFFT_BACKEND == 1) || (VKFFT_BACKEND == 2))
						sc->tempLen = sprintf(sc->tempStr, "\
sincos(%s, &%s.y, &%s.x);\n", in_1->data.s, out->data.s, out->data.s);
						VkAppendLine(sc);
#elif ((VKFFT_BACKEND == 3) || (VKFFT_BACKEND == 4) || (VKFFT_BACKEND == 5))
						sc->tempLen = sprintf(sc->tempStr, "\
%s.y = sincos(%s, &%s.x);\n", out->data.s, in_1->data.s, out->data.s);
						VkAppendLine(sc);
#endif
						return;
					}
				}
			}
			else {
				switch (in_1->type % 10) {
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "\
%s.x = %.17Le;\n", out->data.s, cos(in_1->data.d));
					VkAppendLine(sc);
					sc->tempLen = sprintf(sc->tempStr, "\
%s.y = %.17Le;\n", out->data.s, sin(in_1->data.d));
					VkAppendLine(sc);
					return;
				}
			}
		}
	}
	else {
		switch (out->type % 10) {
		case 3:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 2:
					out->data.c[0] = cos(in_1->data.d);
					out->data.c[1] = sin(in_1->data.d);
					return;
				}
			}
			break;
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}
static inline void VkNorm(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in_1) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
		//in_1 has to be same type as out
		switch (out->type % 10) {
		case 2:
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "\
%s = %s.x*%s.x + %s.y * %s.y;\n", out->data.s, in_1->data.s, in_1->data.s, in_1->data.s, in_1->data.s);
					VkAppendLine(sc);
				}
			}
			else {
				switch (in_1->type % 10) {
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "\
%s = %.17Le;\n", out->data.s, in_1->data.c[0] * in_1->data.c[0] + in_1->data.c[1] * in_1->data.c[1]);
					VkAppendLine(sc);
					return;
				}
			}
		}
	}
	else {
		switch (out->type % 10) {
		case 2:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 3:
					out->data.d = in_1->data.c[0] * in_1->data.c[0] + in_1->data.c[1] * in_1->data.c[1];
					return;
				}
			}
			break;
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}
static inline void VkRsqrt(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in_1) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
		//in_1 has to be same type as out
		switch (out->type % 10) {
		case 2:
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 2:
#if(VKFFT_BACKEND==0)
					sc->tempLen = sprintf(sc->tempStr, "\
%s = inversesqrt(%s);\n", out->data.s, in_1->data.s);
					VkAppendLine(sc);
#else
					sc->tempLen = sprintf(sc->tempStr, "\
%s = rsqrt(%s);\n", out->data.s, in_1->data.s);
					VkAppendLine(sc);
#endif
				}
			}
			else {
				switch (in_1->type % 10) {
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "\
%s = %.17Le;\n", out->data.s, 1.0l / sqrt(in_1->data.d));
					VkAppendLine(sc);
					return;
				}
			}
		}
	}
	else {
		switch (out->type % 10) {
		case 2:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 2:
					out->data.d = 1.0l / sqrt(in_1->data.d);
					return;
				}
			}
			break;
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}

static inline void VkConjugate(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in_1) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
		//in_1 has to be same type as out
		switch (out->type % 10) {
		case 3:
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 3:
					if (strcmp(out->data.s, in_1->data.s)) {
						sc->tempLen = sprintf(sc->tempStr, "%s.x = ", out->data.s);
						VkAppendLine(sc);
						VkAppendConversionStart(sc, out, in_1);
						sc->tempLen = sprintf(sc->tempStr, "%s.x", in_1->data.s);
						VkAppendLine(sc);
						VkAppendConversionEnd(sc, out, in_1);
						sc->tempLen = sprintf(sc->tempStr, ";\n");
						VkAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "%s.y = ", out->data.s);
						VkAppendLine(sc);
						VkAppendConversionStart(sc, out, in_1);
						sc->tempLen = sprintf(sc->tempStr, "-%s.y", in_1->data.s);
						VkAppendLine(sc);
						VkAppendConversionEnd(sc, out, in_1);
						sc->tempLen = sprintf(sc->tempStr, ";\n");
						VkAppendLine(sc);
					}
					else {
						sc->tempLen = sprintf(sc->tempStr, "%s.y = ", out->data.s);
						VkAppendLine(sc);
						VkAppendConversionStart(sc, out, in_1);
						sc->tempLen = sprintf(sc->tempStr, "-%s.y", in_1->data.s);
						VkAppendLine(sc);
						VkAppendConversionEnd(sc, out, in_1);
						sc->tempLen = sprintf(sc->tempStr, ";\n");
						VkAppendLine(sc);
					}
					return;
				}
			}
			else {
				switch (in_1->type % 10) {
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.x = %.17Le", out->data.s, in_1->data.c[0]);
					VkAppendLine(sc);
					VkAppendNumberLiteral(sc, out);
					sc->tempLen = sprintf(sc->tempStr, ";\n");
					VkAppendLine(sc);
					sc->tempLen = sprintf(sc->tempStr, "%s.x = %.17Le", out->data.s, -in_1->data.c[1]);
					VkAppendLine(sc);
					VkAppendNumberLiteral(sc, out);
					sc->tempLen = sprintf(sc->tempStr, ";\n");
					VkAppendLine(sc);
					return;
				}
			}
			break;
		}
	}
	else {
		switch (out->type % 10) {
		case 3:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 3:
					out->data.c[0] = in_1->data.c[0];
					out->data.c[1] = -in_1->data.c[1];
					return;
				}
			}
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}

static inline void VkShuffleComplex(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in_1, VkContainer* in_2, VkContainer* temp) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
		if (strcmp(out->data.s, in_2->data.s)) {
			sc->tempLen = sprintf(sc->tempStr, "%s.x", out->data.s);
		}
		else {
			sc->tempLen = sprintf(sc->tempStr, "%s.x", temp->data.s);
		}
		VkAppendLine(sc);
		sc->tempLen = sprintf(sc->tempStr, " = ");
		VkAppendLine(sc);
		if ((in_1->type < 100) && (in_2->type < 100)) {
			switch (in_1->type % 10) {
			case 3:
				switch (in_2->type % 10) {
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] - in_2->data.c[1]);
					VkAppendLine(sc);
					break;
				}
				break;
			}
			VkAppendNumberLiteral(sc, out);
		}
		else {
			VkAppendConversionStart(sc, out, in_1);
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.x", in_1->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_1->type % 10) {
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, " - ");
			VkAppendLine(sc);
			VkAppendConversionStart(sc, out, in_2);
			if (in_2->type > 100) {
				switch (in_2->type % 10) {
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.y", in_2->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_2->type % 10) {
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.c[1]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_2);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
		}
		if (strcmp(out->data.s, in_2->data.s)) {
			sc->tempLen = sprintf(sc->tempStr, "%s.y", out->data.s);
		}
		else {
			sc->tempLen = sprintf(sc->tempStr, "%s.y", temp->data.s);
		}
		VkAppendLine(sc);
		sc->tempLen = sprintf(sc->tempStr, " = ");
		VkAppendLine(sc);
		if ((in_1->type < 100) && (in_2->type < 100)) {
			switch (in_1->type % 10) {
			case 3:
				switch (in_2->type % 10) {
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] + in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				break;
			}
			VkAppendNumberLiteral(sc, out);
		}
		else {
			VkAppendConversionStart(sc, out, in_1);
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.y", in_1->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_1->type % 10) {
					;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, " + ");
			VkAppendLine(sc);
			VkAppendConversionStart(sc, out, in_2);
			if (in_2->type > 100) {
				switch (in_2->type % 10) {
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.x", in_2->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_2->type % 10) {
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_2);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
		}
		if (!strcmp(out->data.s, in_2->data.s)) {
			VkMov(sc, out, temp);
		}
		return;
	}
	else {
		switch (out->type % 10) {
		case 3:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 3:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 3:
							out->data.c[0] = in_1->data.c[0] - in_2->data.c[1];
							out->data.c[1] = in_1->data.c[1] + in_2->data.c[0];
							return;
						}
					}
					break;
				}
			}
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}
static inline void VkShuffleComplexInv(VkFFTSpecializationConstantsLayout* sc, VkContainer* out, VkContainer* in_1, VkContainer* in_2, VkContainer* temp) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (out->type > 100) {
		if (strcmp(out->data.s, in_2->data.s)) {
			sc->tempLen = sprintf(sc->tempStr, "%s.x", out->data.s);
		}
		else {
			sc->tempLen = sprintf(sc->tempStr, "%s.x", temp->data.s);
		}
		VkAppendLine(sc);
		sc->tempLen = sprintf(sc->tempStr, " = ");
		VkAppendLine(sc);
		if ((in_1->type < 100) && (in_2->type < 100)) {
			switch (in_1->type % 10) {
			case 3:
				switch (in_2->type % 10) {
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0] + in_2->data.c[1]);
					VkAppendLine(sc);
					break;
				}
				break;
			}
			VkAppendNumberLiteral(sc, out);
		}
		else {
			VkAppendConversionStart(sc, out, in_1);
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.x", in_1->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_1->type % 10) {
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, " + ");
			VkAppendLine(sc);
			VkAppendConversionStart(sc, out, in_2);
			if (in_2->type > 100) {
				switch (in_2->type % 10) {
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.y", in_2->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_2->type % 10) {
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.c[1]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_2);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
		}
		if (strcmp(out->data.s, in_2->data.s)) {
			sc->tempLen = sprintf(sc->tempStr, "%s.y", out->data.s);
		}
		else {
			sc->tempLen = sprintf(sc->tempStr, "%s.y", temp->data.s);
		}
		VkAppendLine(sc);
		sc->tempLen = sprintf(sc->tempStr, " = ");
		VkAppendLine(sc);
		if ((in_1->type < 100) && (in_2->type < 100)) {
			switch (in_1->type % 10) {
			case 3:
				switch (in_2->type % 10) {
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1] - in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				break;
			}
			VkAppendNumberLiteral(sc, out);
		}
		else {
			VkAppendConversionStart(sc, out, in_1);
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.y", in_1->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_1->type % 10) {
					;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_1->data.c[1]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, " - ");
			VkAppendLine(sc);
			VkAppendConversionStart(sc, out, in_2);
			if (in_2->type > 100) {
				switch (in_2->type % 10) {
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%s.x", in_2->data.s);
					VkAppendLine(sc);
					break;
				}
			}
			else {
				switch (in_2->type % 10) {
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", in_2->data.c[0]);
					VkAppendLine(sc);
					break;
				}
				VkAppendNumberLiteral(sc, out);
			}
			VkAppendConversionEnd(sc, out, in_2);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			VkAppendLine(sc);
		}
		if (!strcmp(out->data.s, in_2->data.s)) {
			VkMov(sc, out, temp);
		}
		return;
	}
	else {
		switch (out->type % 10) {
		case 3:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 3:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 3:
							out->data.c[0] = in_1->data.c[0] + in_2->data.c[1];
							out->data.c[1] = in_1->data.c[1] - in_2->data.c[0];
							return;
						}
					}
					break;
				}
			}
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}

//logic functions: if, ge, gt, le, lt, etc.
static inline void VkIf_eq_start(VkFFTSpecializationConstantsLayout* sc, VkContainer* left, VkContainer* right) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (left->type > 100) {
		if (right->type > 100) {
			sc->tempLen = sprintf(sc->tempStr, "\
if (%s == %s) {\n", left->data.s, right->data.s);
			VkAppendLine(sc);
			return;
		}
		else {
			switch (right->type % 10) {
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%s == %" PRIi64 ") {\n", left->data.s, right->data.i);
				VkAppendLine(sc);
				return;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%s == %.17Le) {\n", left->data.s, right->data.d);
				VkAppendLine(sc);
				return;
			case 3:
				break;
			}
		}
	}
	else {
		if (right->type > 100) {
			switch (left->type % 10) {
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%" PRIi64 " == %s) {\n", left->data.i, right->data.s);
				VkAppendLine(sc);
				return;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%.17Le == %s) {\n", left->data.d, right->data.s);
				VkAppendLine(sc);
				return;
			case 3:
				break;
			}
		}
		else {
			switch (left->type % 10) {
			case 1:
				switch (right->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.i == right->data.i));
					VkAppendLine(sc);
					return;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.i == right->data.d));
					VkAppendLine(sc);
					return;
				case 3:
					break;
				}
				break;
			case 2:
				switch (right->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.d == right->data.i));
					VkAppendLine(sc);
					return;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.d == right->data.d));
					VkAppendLine(sc);
					return;
				case 3:
					break;
				}
				return;
			case 3:
				break;
			}
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}
static inline void VkIf_lt_start(VkFFTSpecializationConstantsLayout* sc, VkContainer* left, VkContainer* right) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (left->type > 100) {
		if (right->type > 100) {
			sc->tempLen = sprintf(sc->tempStr, "\
if (%s < %s) {\n", left->data.s, right->data.s);
			VkAppendLine(sc);
			return;
		}
		else {
			switch (right->type % 10) {
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%s < %" PRIi64 ") {\n", left->data.s, right->data.i);
				VkAppendLine(sc);
				return;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%s < %.17Le) {\n", left->data.s, right->data.d);
				VkAppendLine(sc);
				return;
			case 3:
				break;
			}
		}
	}
	else {
		if (right->type > 100) {
			switch (left->type % 10) {
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%" PRIi64 " < %s) {\n", left->data.i, right->data.s);
				VkAppendLine(sc);
				return;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%.17Le < %s) {\n", left->data.d, right->data.s);
				VkAppendLine(sc);
				return;
			case 3:
				break;
			}
		}
		else {
			switch (left->type % 10) {
			case 1:
				switch (right->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.i < right->data.i));
					VkAppendLine(sc);
					return;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.i < right->data.d));
					VkAppendLine(sc);
					return;
				case 3:
					break;
				}
				break;
			case 2:
				switch (right->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.d < right->data.i));
					VkAppendLine(sc);
					return;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.d < right->data.d));
					VkAppendLine(sc);
					return;
				case 3:
					break;
				}
				return;
			case 3:
				break;
			}
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}
static inline void VkIf_le_start(VkFFTSpecializationConstantsLayout* sc, VkContainer* left, VkContainer* right) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (left->type > 100) {
		if (right->type > 100) {
			sc->tempLen = sprintf(sc->tempStr, "\
if (%s <= %s) {\n", left->data.s, right->data.s);
			VkAppendLine(sc);
			return;
}
		else {
			switch (right->type % 10) {
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%s <= %" PRIi64 ") {\n", left->data.s, right->data.i);
				VkAppendLine(sc);
				return;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%s <= %.17Le) {\n", left->data.s, right->data.d);
				VkAppendLine(sc);
				return;
			case 3:
				break;
			}
		}
	}
	else {
		if (right->type > 100) {
			switch (left->type % 10) {
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%" PRIi64 " <= %s) {\n", left->data.i, right->data.s);
				VkAppendLine(sc);
				return;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%.17Le <= %s) {\n", left->data.d, right->data.s);
				VkAppendLine(sc);
				return;
			case 3:
				break;
			}
		}
		else {
			switch (left->type % 10) {
			case 1:
				switch (right->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.i <= right->data.i));
					VkAppendLine(sc);
					return;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.i <= right->data.d));
					VkAppendLine(sc);
					return;
				case 3:
					break;
				}
				break;
			case 2:
				switch (right->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.d <= right->data.i));
					VkAppendLine(sc);
					return;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.d <= right->data.d));
					VkAppendLine(sc);
					return;
				case 3:
					break;
				}
				return;
			case 3:
				break;
			}
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}
static inline void VkIf_gt_start(VkFFTSpecializationConstantsLayout* sc, VkContainer* left, VkContainer* right) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (left->type > 100) {
		if (right->type > 100) {
			sc->tempLen = sprintf(sc->tempStr, "\
if (%s > %s) {\n", left->data.s, right->data.s);
			VkAppendLine(sc);
			return;
}
		else {
			switch (right->type % 10) {
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%s > %" PRIi64 ") {\n", left->data.s, right->data.i);
				VkAppendLine(sc);
				return;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%s > %.17Le) {\n", left->data.s, right->data.d);
				VkAppendLine(sc);
				return;
			case 3:
				break;
			}
		}
	}
	else {
		if (right->type > 100) {
			switch (left->type % 10) {
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%" PRIi64 " > %s) {\n", left->data.i, right->data.s);
				VkAppendLine(sc);
				return;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%.17Le > %s) {\n", left->data.d, right->data.s);
				VkAppendLine(sc);
				return;
			case 3:
				break;
			}
		}
		else {
			switch (left->type % 10) {
			case 1:
				switch (right->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.i > right->data.i));
					VkAppendLine(sc);
					return;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.i > right->data.d));
					VkAppendLine(sc);
					return;
				case 3:
					break;
				}
				break;
			case 2:
				switch (right->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.d > right->data.i));
					VkAppendLine(sc);
					return;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.d > right->data.d));
					VkAppendLine(sc);
					return;
				case 3:
					break;
				}
				return;
			case 3:
				break;
			}
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}
static inline void VkIf_ge_start(VkFFTSpecializationConstantsLayout* sc, VkContainer* left, VkContainer* right) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (left->type > 100) {
		if (right->type > 100) {
			sc->tempLen = sprintf(sc->tempStr, "\
if (%s >= %s) {\n", left->data.s, right->data.s);
			VkAppendLine(sc);
			return;
}
		else {
			switch (right->type % 10) {
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%s >= %" PRIi64 ") {\n", left->data.s, right->data.i);
				VkAppendLine(sc);
				return;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%s >= %.17Le) {\n", left->data.s, right->data.d);
				VkAppendLine(sc);
				return;
			case 3:
				break;
			}
		}
	}
	else {
		if (right->type > 100) {
			switch (left->type % 10) {
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%" PRIi64 " >= %s) {\n", left->data.i, right->data.s);
				VkAppendLine(sc);
				return;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%.17Le >= %s) {\n", left->data.d, right->data.s);
				VkAppendLine(sc);
				return;
			case 3:
				break;
			}
		}
		else {
			switch (left->type % 10) {
			case 1:
				switch (right->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.i >= right->data.i));
					VkAppendLine(sc);
					return;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.i >= right->data.d));
					VkAppendLine(sc);
					return;
				case 3:
					break;
				}
				break;
			case 2:
				switch (right->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.d >= right->data.i));
					VkAppendLine(sc);
					return;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.d >= right->data.d));
					VkAppendLine(sc);
					return;
				case 3:
					break;
				}
				return;
			case 3:
				break;
			}
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}

static inline void VkIf_start(VkFFTSpecializationConstantsLayout* sc) {
	if (sc->res != VKFFT_SUCCESS) return;
	sc->tempLen = sprintf(sc->tempStr, "\
{\n");
	VkAppendLine(sc);
	return;
}
static inline void VkIfTrue(VkFFTSpecializationConstantsLayout* sc, VkContainer* in) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (in->type > 100) {
		switch (in->type % 10) {
		case 1:
			sc->tempLen = sprintf(sc->tempStr, "\
if (%s) {\n", in->data.s);
			VkAppendLine(sc);
			return;
		case 2:
			break;
		case 3:
			break;
		}
	}
	else {
		switch (in->type % 10) {
		case 1:
			sc->tempLen = sprintf(sc->tempStr, "\
if (%" PRIi64 ") {\n", in->data.i);
			VkAppendLine(sc);
			return;
		case 2:
			break;
		case 3:
			break;
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}
static inline void VkIfFalse(VkFFTSpecializationConstantsLayout* sc, VkContainer* in) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (in->type > 100) {
		switch (in->type % 10) {
		case 1:
			sc->tempLen = sprintf(sc->tempStr, "\
if (!%s) {\n", in->data.s);
			VkAppendLine(sc);
			return;
		case 2:
			break;
		case 3:
			break;
		}
	}
	else {
		switch (in->type % 10) {
		case 1:
			sc->tempLen = sprintf(sc->tempStr, "\
if (!%" PRIi64 ") {\n", in->data.i);
			VkAppendLine(sc);
			return;
		case 2:
			break;
		case 3:
			break;
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}
static inline void VkIf_else(VkFFTSpecializationConstantsLayout* sc) {
	if (sc->res != VKFFT_SUCCESS) return;
	sc->tempLen = sprintf(sc->tempStr, "\
}else{\n");
	VkAppendLine(sc);
	return;
}
static inline void VkIf_end(VkFFTSpecializationConstantsLayout* sc) {
	if (sc->res != VKFFT_SUCCESS) return;
	sc->tempLen = sprintf(sc->tempStr, "\
}\n");
	VkAppendLine(sc);
	return;
}

static inline void VkPrintReg(VkFFTSpecializationConstantsLayout* sc, VkContainer* inoutID, VkContainer* in) {
	if (sc->res != VKFFT_SUCCESS) return;
	if (in->type > 100) {
		switch (in->type % 10) {
		case 1:
			sc->tempLen = sprintf(sc->tempStr, "printf(\"%%d %%d\\n\", %s, %s);", inoutID->data.s, in->data.s);
			VkAppendLine(sc);
			return;
		case 2:
			sc->tempLen = sprintf(sc->tempStr, "printf(\"%%d %%f\\n\", %s, %s);", inoutID->data.s, in->data.s);
			VkAppendLine(sc); return;
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "printf(\"%%d %%f %%f\\n\", %s, %s.x, %s.y);", inoutID->data.s, in->data.s, in->data.s);
			VkAppendLine(sc);
			return;
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}

static inline void VkPermute(VkFFTSpecializationConstantsLayout* sc, uint64_t* permute, uint64_t num_elem, uint64_t type, VkContainer* regIDs, VkContainer* temp) {
	if (sc->res != VKFFT_SUCCESS) return;
	char* temp_ID[33];
	if (type == 0) {
		if (sc->locID[0].type > 100) {
			for (uint64_t i = 0; i < num_elem; i++)
				temp_ID[i] = sc->locID[i].data.s;
			for (uint64_t i = 0; i < num_elem; i++)
				sc->locID[i].data.s = temp_ID[permute[i]];
			return;
		}
	}
	if (type == 1) {
		if (regIDs[0].type > 100) {
			for (uint64_t i = 0; i < num_elem; i++)
				temp_ID[i] = regIDs[i].data.s;
			for (uint64_t i = 0; i < num_elem; i++)
				regIDs[i].data.s = temp_ID[permute[i]];
			return;
		}
	}
	sc->res = VKFFT_ERROR_MATH_FAILED;
	return;
}
static inline void VkSubgroupAdd(VkFFTSpecializationConstantsLayout* sc, VkContainer* in, VkContainer* out, uint64_t subWarpSplit) {
	if (sc->res != VKFFT_SUCCESS) return;

#if (VKFFT_BACKEND==0)
	/*sc->tempLen = sprintf(sc->tempStr, "	%s.x = subgroupAdd(%s.x);\n", out, in);
	res = VkAppendLine(sc);
	if (res != VKFFT_SUCCESS) return res;
	sc->tempLen = sprintf(sc->tempStr, "	%s.y = subgroupAdd(%s.y);\n", out, in);
	res = VkAppendLine(sc);
	if (res != VKFFT_SUCCESS) return res;*/
#elif (VKFFT_BACKEND==1)
	//v1
	/*for (int i = 1; i < sc->warpSize / subWarpSplit; i *= 2) {
		sc->tempLen = sprintf(sc->tempStr, "	%s.x += __shfl_xor_sync(0xffffffff, %s.x, %d);\n", out, in, i);
		res = VkAppendLine(sc);
		if (res != VKFFT_SUCCESS) return res;
		sc->tempLen = sprintf(sc->tempStr, "	%s.y += __shfl_xor_sync(0xffffffff, %s.y, %d);\n", out, in, i);
		res = VkAppendLine(sc);
		if (res != VKFFT_SUCCESS) return res;
	}
	//v2
	for (int i = (int)sc->warpSize / 2 / subWarpSplit; i > 0; i /= 2) {
		sc->tempLen = sprintf(sc->tempStr, "	%s.x += __shfl_down_sync(0xffffffff, %s.x, %d);\n", out, in, i);
		res = VkAppendLine(sc);
		if (res != VKFFT_SUCCESS) return res;
		sc->tempLen = sprintf(sc->tempStr, "	%s.y += __shfl_down_sync(0xffffffff, %s.y, %d);\n", out, in, i);
		res = VkAppendLine(sc);
		if (res != VKFFT_SUCCESS) return res;
	}*/
#endif
	return;
}

#endif