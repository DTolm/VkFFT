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
#ifndef VKFFT_R2R_H
#define VKFFT_R2R_H
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_StringManagement/vkFFT_StringManager.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_MathUtils/vkFFT_MathUtils.h"

static inline void appendDCTI_read_get_inoutID (VkFFTSpecializationConstantsLayout* sc, PfContainer* inoutID, int readWrite, PfContainer* tempInt) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;


	if (sc->zeropadBluestein[readWrite]) {
		if (readWrite) {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i;
		}
		else {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i;
		}
	}
	else {
		if(sc->performDCT)
			fftDim.data.i = (sc->fft_dim_full.data.i + 2) / 2;
		if(sc->performDST)
			fftDim.data.i = (sc->fft_dim_full.data.i - 2) / 2;
	}
	if (readWrite == 0) {
		if(sc->performDCT)
			temp_int.data.i = (2 * fftDim.data.i - 2);
		if(sc->performDST)
			temp_int.data.i = (2 * fftDim.data.i + 2);
		if (sc->performDCT == 1) {
			PfIf_lt_start(sc, inoutID, &temp_int);
			PfIf_ge_start(sc, inoutID, &fftDim);
			PfSub(sc, tempInt, &temp_int, inoutID);
			PfIf_else(sc);
			PfMov(sc, tempInt, inoutID);
			PfIf_end(sc);
		}
		if (sc->performDST == 1) {
			PfIf_lt_start(sc, inoutID, &temp_int);

			temp_int.data.i = 0;
			PfIf_neq_start(sc, inoutID, &temp_int);
			temp_int.data.i = fftDim.data.i + 1;
			PfIf_neq_start(sc, inoutID, &temp_int);

			temp_int.data.i = fftDim.data.i + 2;
			PfIf_ge_start(sc, inoutID, &temp_int);
			temp_int.data.i = (2 * fftDim.data.i + 1);
			PfSub(sc, tempInt, &temp_int, inoutID);
			PfIf_else(sc);
			temp_int.data.i = 1;
			PfSub(sc, tempInt, inoutID, &temp_int);
			PfIf_end(sc);
		}
		PfSwapContainers(sc, tempInt, inoutID);
		
	}
	return;
}
static inline void appendDCTI_read_set_inoutID (VkFFTSpecializationConstantsLayout* sc, PfContainer* inoutID, int readWrite, PfContainer* tempInt) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;


	if (sc->zeropadBluestein[readWrite]) {
		if (readWrite) {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i;
		}
		else {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i;
		}
	}
	else {
		if(sc->performDCT)
			fftDim.data.i = (sc->fft_dim_full.data.i + 2) / 2;
		if(sc->performDST)
			fftDim.data.i = (sc->fft_dim_full.data.i - 2) / 2;
	}
	if (readWrite == 0) {
		PfSwapContainers(sc, inoutID, tempInt);
		if (sc->performDCT == 1) {
			PfIf_end(sc);
		}
		if (sc->performDST == 1) {
			PfIf_end(sc);
			PfIf_end(sc);
			PfIf_end(sc);
		}
	}
	return;
}

static inline void appendDCTI_write_get_inoutID (VkFFTSpecializationConstantsLayout* sc, PfContainer* inoutID, int readWrite, PfContainer* tempInt) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;


	if (sc->zeropadBluestein[readWrite]) {
		if (readWrite) {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i;
		}
		else {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i;
		}
	}
	else {
		if(sc->performDCT)
			fftDim.data.i = (sc->fft_dim_full.data.i + 2) / 2;
		if(sc->performDST)
			fftDim.data.i = (sc->fft_dim_full.data.i - 2) / 2;
	}
	if (readWrite == 1) {
		if(sc->performDCT)
			temp_int.data.i = (2 * fftDim.data.i - 2);
		if(sc->performDST)
			temp_int.data.i = (2 * fftDim.data.i + 2);
		if (sc->performDST == 1) {
			temp_int.data.i = 0;
			PfIf_neq_start(sc, inoutID, &temp_int);
			temp_int.data.i = 1;
			PfSub(sc, tempInt, inoutID, &temp_int);
			PfSwapContainers(sc, tempInt, inoutID);
		}		
	}
	return;
}
static inline void appendDCTI_write_set_inoutID (VkFFTSpecializationConstantsLayout* sc, PfContainer* inoutID, int readWrite, PfContainer* tempInt) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;


	if (sc->zeropadBluestein[readWrite]) {
		if (readWrite) {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i;
		}
		else {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i;
		}
	}
	else {
		if(sc->performDCT)
			fftDim.data.i = (sc->fft_dim_full.data.i + 2) / 2;
		if(sc->performDST)
			fftDim.data.i = (sc->fft_dim_full.data.i - 2) / 2;
	}
	if (readWrite == 1) {
		if (sc->performDST == 1) {
			PfIf_end(sc);
			PfSwapContainers(sc, tempInt, inoutID);
		}
	}
	return;
}

static inline void appendDCTII_read_III_write_get_inoutID (VkFFTSpecializationConstantsLayout* sc, PfContainer* inoutID, int readWrite, PfContainer* tempInt, PfContainer* tempInt2) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;

	if (sc->zeropadBluestein[readWrite]) {
		if (readWrite) {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i;
		}
		else {
			//appendSetSMToZero(sc);
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i;
		}
	}
	else {
		fftDim.data.i = sc->fft_dim_full.data.i;
	}
	if (((readWrite == 0) && ((((sc->performDCT == 2) || (sc->performDST == 2)) && (sc->actualInverse == 0)) || (((sc->performDCT == 3) || (sc->performDST == 3)) && (sc->actualInverse == 1)))) || ((readWrite == 1) && ((((sc->performDCT == 2) || (sc->performDST == 2)) && (sc->actualInverse == 1)) || (((sc->performDCT == 3) || (sc->performDST == 3)) && (sc->actualInverse == 0))))) {
		PfIf_lt_start(sc, inoutID, &fftDim);
		temp_int.data.i = 2;
		PfDivCeil(sc, &temp_int, &fftDim, &temp_int);
		PfIf_lt_start(sc, inoutID, &temp_int);
		temp_int.data.i = 2;
		PfMul(sc, tempInt, inoutID, &temp_int, 0);
		PfIf_else(sc);
		PfMul(sc, tempInt, inoutID, &temp_int, 0);
		temp_int.data.i = 2*fftDim.data.i - 1;
		PfSub(sc, tempInt, &temp_int, tempInt);
		PfIf_end(sc);
		
		PfSwapContainers(sc, tempInt, inoutID);
	}
	return;
}
static inline void appendDCTII_read_III_write_set_inoutID (VkFFTSpecializationConstantsLayout* sc, PfContainer* inoutID, int readWrite, PfContainer* tempInt) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;

	if (sc->zeropadBluestein[readWrite]) {
		if (readWrite) {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i;
		}
		else {
			//appendSetSMToZero(sc);
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i;
		}
	}
	else {
		fftDim.data.i = sc->fft_dim_full.data.i;
	}
	if (((readWrite == 0) && ((((sc->performDCT == 2) || (sc->performDST == 2)) && (sc->actualInverse == 0)) || (((sc->performDCT == 3) || (sc->performDST == 3)) && (sc->actualInverse == 1)))) || ((readWrite == 1) && ((((sc->performDCT == 2) || (sc->performDST == 2)) && (sc->actualInverse == 1)) || (((sc->performDCT == 3) || (sc->performDST == 3)) && (sc->actualInverse == 0))))) {
		PfSwapContainers(sc, tempInt, inoutID);
		PfIf_end(sc);
	}
	return;
}

static inline void appendDCTII_write_III_read_get_inoutID (VkFFTSpecializationConstantsLayout* sc, PfContainer* inoutID, int readWrite, PfContainer* tempInt, PfContainer* tempInt2) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;

	if (sc->zeropadBluestein[readWrite]) {
		if (readWrite) {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i;
		}
		else {
			//appendSetSMToZero(sc);
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i;
		}
	}
	else {
		fftDim.data.i = sc->fft_dim_full.data.i;
	}
	
	if ((readWrite == 1) && (((sc->performDST == 2) && (sc->actualInverse == 0)) || ((sc->performDST == 3) && (sc->actualInverse == 1)))) {
		temp_int.data.i = fftDim.data.i;
		PfIf_lt_start(sc, inoutID, &temp_int);
		temp_int.data.i = fftDim.data.i - 1;
		PfSub(sc, tempInt, &temp_int, inoutID);
		PfIf_eq_start(sc, inoutID, &temp_int);
		PfSetToZero(sc, tempInt);
		PfIf_end(sc);
		PfSwapContainers(sc, tempInt, inoutID);
	}
	if ((readWrite == 0) && (((sc->performDST == 2) && (sc->actualInverse == 1)) || ((sc->performDST == 3) && (sc->actualInverse == 0)))) {
		temp_int.data.i = fftDim.data.i;
		PfIf_lt_start(sc, inoutID, &temp_int);
		temp_int.data.i = fftDim.data.i - 1;
		PfSub(sc, tempInt, &temp_int, inoutID);
		PfIf_eq_start(sc, inoutID, &temp_int);
		PfSetToZero(sc, tempInt);
		PfIf_end(sc);

		if (sc->axis_id > 0) {
			PfSub(sc, &sc->inoutID2, &fftDim, inoutID);
			PfIf_eq_start(sc, &sc->inoutID2, &fftDim);
			PfSetToZero(sc, &sc->inoutID2);
			PfIf_end(sc);
			temp_int.data.i = fftDim.data.i - 1;
			PfSub(sc, &sc->inoutID2, &temp_int, &sc->inoutID2);
		}
		else {
			PfSub(sc, &sc->inoutID2, &fftDim, inoutID);
			PfIf_eq_start(sc, &sc->inoutID2, &fftDim);
			PfSetToZero(sc, &sc->inoutID2);
			PfIf_end(sc);
			temp_int.data.i = fftDim.data.i - 1;
			PfSub(sc, &sc->inoutID2, &temp_int, &sc->inoutID2);
		}

		PfSwapContainers(sc, tempInt, inoutID);
	}
	if ((readWrite == 0) && (((sc->performDCT == 2) && (sc->actualInverse == 1)) || ((sc->performDCT == 3) && (sc->actualInverse == 0)))) {
		temp_int.data.i = fftDim.data.i;
		PfIf_lt_start(sc, inoutID, &temp_int);
		
		if (sc->axis_id > 0) {
			PfSub(sc, &sc->inoutID2, &fftDim, inoutID);
			PfIf_eq_start(sc, &sc->inoutID2, &fftDim);
			PfSetToZero(sc, &sc->inoutID2);
			PfIf_end(sc);
		}
		else {
			PfSub(sc, &sc->inoutID2, &fftDim, inoutID);
			PfIf_eq_start(sc, &sc->inoutID2, &fftDim);
			PfSetToZero(sc, &sc->inoutID2);
			PfIf_end(sc);
		}
		//PfSwapContainers(sc, tempInt, inoutID);
	}
	return;
}
static inline void appendDCTII_write_III_read_set_inoutID (VkFFTSpecializationConstantsLayout* sc, PfContainer* inoutID, int readWrite, PfContainer* tempInt) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;

	if (sc->zeropadBluestein[readWrite]) {
		if (readWrite) {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i;
		}
		else {
			//appendSetSMToZero(sc);
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i;
		}
	}
	else {
		fftDim.data.i = sc->fft_dim_full.data.i;
	}

	if (((readWrite == 1) && (((sc->performDST == 2) && (sc->actualInverse == 0)) || ((sc->performDST == 3) && (sc->actualInverse == 1)))) || ((readWrite == 0) && (((sc->performDST == 2) && (sc->actualInverse == 1)) || ((sc->performDST == 3) && (sc->actualInverse == 0))))) {
		PfSwapContainers(sc, tempInt, inoutID);
		PfIf_end(sc);
	}
	if ((readWrite == 0) && (((sc->performDCT == 2) && (sc->actualInverse == 1)) || ((sc->performDCT == 3) && (sc->actualInverse == 0)))) {
		//PfSwapContainers(sc, tempInt, inoutID);
		PfIf_end(sc);
	}
	return;
}

static inline void appendDCTIV_even_read_get_inoutID (VkFFTSpecializationConstantsLayout* sc, PfContainer* inoutID, int readWrite, PfContainer* tempInt) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;

	if (readWrite == 0) {
		temp_int.data.i = 2;
		PfMul(sc, tempInt, inoutID, &temp_int, 0);

		if (sc->zeropadBluestein[readWrite]) {
			if (readWrite) {
				temp_int.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i - 1;
			}
			else {
				temp_int.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i - 1;
			}
		}
		else
			temp_int.data.i = 2*sc->fft_dim_full.data.i - 1;
		PfIf_le_start(sc, tempInt, &temp_int);
		PfSub(sc, &sc->inoutID2, &temp_int, tempInt);
		PfIf_else(sc);
		temp_int.data.i += 1;
        PfMov(sc, &sc->inoutID2, &temp_int);
        PfIf_end(sc);
		PfSwapContainers(sc, tempInt, inoutID);
	}
	return;
}
static inline void appendDCTIV_even_read_set_inoutID (VkFFTSpecializationConstantsLayout* sc, PfContainer* inoutID, int readWrite, PfContainer* tempInt) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;

	if (readWrite == 0) {
		PfSwapContainers(sc, inoutID, tempInt);
	}
	return;
}

static inline void appendDCTIV_odd_read_get_inoutID (VkFFTSpecializationConstantsLayout* sc, PfContainer* inoutID, int readWrite, PfContainer* tempInt, PfContainer* tempInt2) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;
	if (sc->zeropadBluestein[readWrite]) {
		if (readWrite) {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i;
		}
		else {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i;
		}
	}
	else {
		fftDim.data.i = sc->fft_dim_full.data.i;
	}

	if (readWrite == 0) {
		PfIf_lt_start(sc, inoutID, &fftDim);
		temp_int.data.i = 4;
		PfMul(sc, &sc->inoutID2, inoutID, &temp_int,0);

		temp_int.data.i = fftDim.data.i / 2;
		PfAdd(sc, &sc->inoutID2, &sc->inoutID2, &temp_int);
		
		PfIf_lt_start(sc, &sc->inoutID2, &fftDim);
		PfMov(sc, tempInt, &sc->inoutID2);
		PfIf_end(sc);
		
		temp_int.data.i = fftDim.data.i * 2;
		PfIf_lt_start(sc, &sc->inoutID2, &temp_int);
		PfIf_ge_start(sc, &sc->inoutID2, &fftDim);
		temp_int.data.i = fftDim.data.i * 2 - 1;
		PfSub(sc, tempInt, &temp_int, &sc->inoutID2);
		PfIf_end(sc);
		PfIf_end(sc);

		temp_int.data.i = fftDim.data.i * 3;
		PfIf_lt_start(sc, &sc->inoutID2, &temp_int);
		temp_int.data.i = fftDim.data.i * 2;
		PfIf_ge_start(sc, &sc->inoutID2, &temp_int);
		temp_int.data.i = fftDim.data.i * 2;
		PfSub(sc, tempInt, &sc->inoutID2, &temp_int);
		PfIf_end(sc);
		PfIf_end(sc);

		temp_int.data.i = fftDim.data.i * 4;
		PfIf_lt_start(sc, &sc->inoutID2, &temp_int);
		temp_int.data.i = fftDim.data.i * 3;
		PfIf_ge_start(sc, &sc->inoutID2, &temp_int);
		temp_int.data.i = fftDim.data.i * 4 - 1;
		PfSub(sc, tempInt, &temp_int, &sc->inoutID2);
		PfIf_end(sc);
		PfIf_end(sc);

		temp_int.data.i = fftDim.data.i * 4;
		PfIf_ge_start(sc, &sc->inoutID2, &temp_int);
		temp_int.data.i = fftDim.data.i * 4;
		PfSub(sc, tempInt, &sc->inoutID2, &temp_int);
		PfIf_end(sc);

		PfSwapContainers(sc, inoutID, tempInt);
	}
	return;
}
static inline void appendDCTIV_odd_read_set_inoutID (VkFFTSpecializationConstantsLayout* sc, PfContainer* inoutID, int readWrite, PfContainer* tempInt) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;

	if (readWrite == 0) {
		PfSwapContainers(sc, inoutID, tempInt);
		PfIf_end(sc);
	}
	return;
}

static inline void appendDCTIV_even_write_get_inoutID (VkFFTSpecializationConstantsLayout* sc, PfContainer* inoutID, int readWrite, PfContainer* tempInt) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;
	if (sc->zeropadBluestein[readWrite]) {
		if (readWrite) {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i/2;
		}
		else {
			//appendSetSMToZero(sc);
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i/2;
		}
	}
	else {
		fftDim.data.i = sc->fft_dim_full.data.i;
	}
	if (readWrite == 1) {
		temp_int.data.i = 2;
		PfDivCeil(sc, &temp_int, &fftDim, &temp_int);
		PfIf_lt_start(sc, inoutID, &temp_int);
		temp_int.data.i = 2;
		PfMul(sc, tempInt, inoutID, &temp_int, 0);
		PfIf_else(sc);
		PfMul(sc, tempInt, inoutID, &temp_int, 0);
		temp_int.data.i = 2*fftDim.data.i - 1;
		PfSub(sc, tempInt, &temp_int, tempInt);
		PfIf_end(sc);
		
		if (sc->performDST) {
			temp_int.data.i = 2*fftDim.data.i - 1;
			PfSub(sc, tempInt, &temp_int, tempInt);
		}
		PfSwapContainers(sc, tempInt, inoutID);
	}
	return;
}
static inline void appendDCTIV_even_write_set_inoutID (VkFFTSpecializationConstantsLayout* sc, PfContainer* inoutID, int readWrite, PfContainer* tempInt) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;

	if (readWrite == 1) {
		PfSwapContainers(sc, inoutID, tempInt);
	}
	return;
}

static inline void appendDCTIV_odd_write_get_inoutID (VkFFTSpecializationConstantsLayout* sc, PfContainer* inoutID, int readWrite, PfContainer* tempInt) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;
	if (sc->zeropadBluestein[readWrite]) {
		if (readWrite) {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i;
		}
		else {
			//appendSetSMToZero(sc);
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i;
		}
	}
	else {
		fftDim.data.i = sc->fft_dim_full.data.i;
	}
	if (readWrite == 1) {
		PfIf_lt_start(sc, inoutID, &fftDim);
		/*temp_int.data.i = 2;
		PfMod(sc, &sc->tempInt, inoutID, &temp_int);

		temp_int.data.i = 1;
		PfIf_eq_start(sc, &sc->tempInt, &temp_int);
		temp_int.data.i = 1;
		PfSub(sc, tempInt, inoutID, &temp_int);
		temp_int.data.i = 2;
		PfDiv(sc, tempInt, tempInt, &temp_int);
		PfIf_else(sc);
		temp_int.data.i = fftDim.data.i - 1;
		PfSub(sc, tempInt, &temp_int, inoutID);
		temp_int.data.i = 2;
		PfDiv(sc, tempInt, tempInt, &temp_int);
		PfIf_end(sc);*/
		temp_int.data.i = 2;
		PfMod(sc, &sc->tempInt, inoutID, &temp_int);

		temp_int.data.i = 1;
		PfIf_eq_start(sc, &sc->tempInt, &temp_int);

		temp_int.data.i = (fftDim.data.i+1) / 2;
		PfIf_lt_start(sc, inoutID, &temp_int);

		temp_int.data.i = 1;
		PfSub(sc, tempInt, inoutID, &temp_int);

		temp_int.data.i = 2;
		PfDiv(sc, tempInt, tempInt, &temp_int);

		PfIf_else(sc);

		temp_int.data.i = 2*fftDim.data.i-1;
		PfSub(sc, tempInt, &temp_int, inoutID);

		temp_int.data.i = 2;
		PfDiv(sc, tempInt, tempInt, &temp_int);

		PfIf_end(sc);
		PfIf_else(sc);

		temp_int.data.i = (fftDim.data.i+1) / 2;
		PfIf_lt_start(sc, inoutID, &temp_int);

		temp_int.data.i = fftDim.data.i - 1;
		PfSub(sc, tempInt, &temp_int, inoutID);
		temp_int.data.i = 2;
		PfDiv(sc, tempInt, tempInt, &temp_int);
		
		PfIf_else(sc);

		temp_int.data.i = fftDim.data.i - 1;
		PfAdd(sc, tempInt, &temp_int, inoutID);
		temp_int.data.i = 2;
		PfDiv(sc, tempInt, tempInt, &temp_int);

		PfIf_end(sc);
		PfIf_end(sc);
		/*temp_int.data.i = fftDim.data.i / 4;
		PfIf_lt_start(sc, inoutID, &temp_int);

		temp_int.data.i = 2;
		PfMul(sc, tempInt, inoutID, &temp_int, 0);
		PfInc(sc, tempInt);
		
		PfIf_eq_start(sc, tempInt, &fftDim);
		PfSetToZero(sc, tempInt);
		PfIf_end(sc);

		PfIf_end(sc);

		temp_int.data.i = fftDim.data.i / 2;
		PfIf_lt_start(sc, inoutID, &temp_int);
		temp_int.data.i = fftDim.data.i / 4;
		PfIf_ge_start(sc, inoutID, &temp_int);

		temp_int.data.i = 2;
		PfMul(sc, tempInt, inoutID, &temp_int, 0);

		temp_int.data.i = 2 * (fftDim.data.i / 2);
		PfSub(sc, tempInt, &temp_int, tempInt);

		PfIf_eq_start(sc, tempInt, &fftDim);
		PfSetToZero(sc, tempInt);
		PfIf_end(sc);

		PfIf_end(sc);
		PfIf_end(sc);

		temp_int.data.i = 3 * fftDim.data.i / 4;
		PfIf_lt_start(sc, inoutID, &temp_int);
		temp_int.data.i = fftDim.data.i / 2;
		PfIf_ge_start(sc, inoutID, &temp_int);

		temp_int.data.i = 2;
		PfMul(sc, tempInt, inoutID, &temp_int, 0);

		temp_int.data.i = 2 * (fftDim.data.i / 2);
		PfSub(sc, tempInt, tempInt, &temp_int);

		PfIf_eq_start(sc, tempInt, &fftDim);
		PfSetToZero(sc, tempInt);
		PfIf_end(sc);

		PfIf_end(sc);
		PfIf_end(sc);

		temp_int.data.i = 3 * fftDim.data.i / 4;
		PfIf_ge_start(sc, inoutID, &temp_int);

		temp_int.data.i = 2;
		PfMul(sc, tempInt, inoutID, &temp_int, 0);

		temp_int.data.i = 2 * fftDim.data.i - 1;
		PfSub(sc, tempInt, &temp_int, tempInt);

		PfIf_eq_start(sc, tempInt, &fftDim);
		PfSetToZero(sc, tempInt);
		PfIf_end(sc);

		PfIf_end(sc);*/
		if (sc->performDST) {
			temp_int.data.i = fftDim.data.i - 1;
			PfSub(sc, tempInt, &temp_int, tempInt);
		}
		PfSwapContainers(sc, tempInt, inoutID);
	}
	return;
}
static inline void appendDCTIV_odd_write_set_inoutID (VkFFTSpecializationConstantsLayout* sc, PfContainer* inoutID, int readWrite, PfContainer* tempInt) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;

	if (readWrite == 1) {
		PfSwapContainers(sc, inoutID, tempInt);
		PfIf_end(sc);
	}
	return;
}

static inline void appendDCTI_processing (VkFFTSpecializationConstantsLayout* sc, PfContainer* inoutID, PfContainer* regID, int readWrite, PfContainer* tempInt, PfContainer* tempInt2) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 22;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;

	if (sc->zeropadBluestein[readWrite]) {
		if (readWrite) {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i;
		}
		else {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i;
		}
	}
	else {
		if(sc->performDCT)
			fftDim.data.i = (sc->fft_dim_full.data.i + 2) / 2;
		if(sc->performDST)
			fftDim.data.i = (sc->fft_dim_full.data.i - 2) / 2;
	}
	if ((readWrite == 0) && (sc->performDST == 1)) {
		temp_int.data.i = fftDim.data.i + 2;
		PfIf_ge_start(sc, &sc->stageInvocationID, &temp_int);
		PfMovNeg(sc, &regID->data.c[0], &regID->data.c[0]);
		PfIf_end(sc);
	}
	return;
}

static inline void appendDCTII_read_III_write_processing (VkFFTSpecializationConstantsLayout* sc, PfContainer* inoutID, PfContainer* regID, int readWrite, PfContainer* tempInt, PfContainer* tempInt2) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 22;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;

	if (sc->zeropadBluestein[readWrite]) {
		if (readWrite) {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i;
		}
		else {
			//appendSetSMToZero(sc);
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i;
		}
	}
	else {
		fftDim.data.i = sc->fft_dim_full.data.i;
	}
	if (((readWrite == 0) && (((sc->performDST == 2) && (sc->actualInverse == 0)) || ((sc->performDST == 3) && (sc->actualInverse == 1)))) || ((readWrite == 1) && (((sc->performDST == 2) && (sc->actualInverse == 1)) || ((sc->performDST == 3) && (sc->actualInverse == 0))))) {
		PfIf_lt_start(sc, inoutID, &fftDim);
		temp_int.data.i = 2;
		PfMod(sc, &sc->tempInt, inoutID, &temp_int);

		temp_int.data.i = 1;
		PfIf_eq_start(sc, &sc->tempInt, &temp_int);
		PfMovNeg(sc, regID, regID);
		PfIf_end(sc);

		PfIf_end(sc);
	}
	return;
}

static inline void appendDCTII_write_III_read_processing (VkFFTSpecializationConstantsLayout* sc, PfContainer* inoutID, PfContainer* regID, int readWrite, PfContainer* tempInt, PfContainer* tempInt2) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 22;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;

	if (sc->zeropadBluestein[readWrite]) {
		if (readWrite) {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i;
		}
		else {
			//appendSetSMToZero(sc);
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i;
		}
	}
	else {
		fftDim.data.i = sc->fft_dim_full.data.i;
	}
	if (((readWrite == 1) && ((((sc->performDCT == 2) || (sc->performDST == 2)) && (sc->actualInverse == 0)) || (((sc->performDCT == 3) || (sc->performDST == 3)) && (sc->actualInverse == 1)))) || ((readWrite == 0) && ((((sc->performDCT == 2) || (sc->performDST == 2)) && (sc->actualInverse == 1)) || (((sc->performDCT == 3) || (sc->performDST == 3)) && (sc->actualInverse == 0))))) {
		if (sc->performDST)
			PfSwapContainers(sc, inoutID, &sc->stageInvocationID);
		PfIf_lt_start(sc, inoutID, &fftDim);
		if (sc->LUT) {
			temp_int.data.i = fftDim.data.i / 2 + 1;
			PfIf_lt_start(sc, inoutID, &temp_int);
			PfMov(sc, &sc->tempInt, inoutID);
			PfIf_else(sc);
			PfSub(sc, &sc->tempInt, &fftDim, inoutID);
			PfIf_end(sc);
			PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->startDCT3LUT);
			appendGlobalToRegisters(sc, &sc->mult, &sc->LUTStruct, &sc->tempInt);
			
			PfIf_ge_start(sc, inoutID, &temp_int);
			PfMov(sc, &sc->w.data.c[0], &sc->mult.data.c[1]);
			PfMov(sc, &sc->mult.data.c[1], &sc->mult.data.c[0]);
			PfMov(sc, &sc->mult.data.c[0], &sc->w.data.c[0]);
			PfIf_end(sc);
			if (readWrite) {
				temp_double.data.d = pfFPinit("2.0");
				PfMul(sc, &sc->mult, &sc->mult, &temp_double, 0);
				PfConjugate(sc, &sc->mult, &sc->mult);
			}
		}
		else {
			if (readWrite)
				temp_double.data.d = -sc->double_PI / pfFPinit("2.0") / fftDim.data.i;
			else
				temp_double.data.d = sc->double_PI / pfFPinit("2.0") / fftDim.data.i;
			PfMul(sc, &sc->w.data.c[0], inoutID, &temp_double, 0);
			PfSinCos(sc, &sc->mult, &sc->w.data.c[0]);
			if (readWrite) {
				temp_double.data.d = pfFPinit("2.0");
				PfMul(sc, &sc->mult, &sc->mult, &temp_double, 0);
			}
		}
		if (readWrite) {
			PfMul(sc, regID, regID, &sc->mult, &sc->w);
		}
		else {
			PfMovNeg(sc, &regID->data.c[1], &regID->data.c[1]);

			PfMul(sc, regID, regID, &sc->mult, &sc->w);
			//PfConjugate(sc, &sc->mult, &sc->mult);

			//PfMul(sc, &sc->temp, &sc->regIDs[1], &sc->mult, 0);
		}
		PfIf_end(sc);
		if (sc->performDST)
			PfSwapContainers(sc, &sc->stageInvocationID, inoutID);
	}
	return;
}

static inline void appendDCTIV_even_read_processing (VkFFTSpecializationConstantsLayout* sc, PfContainer* inoutID, PfContainer* regID, int readWrite, PfContainer* tempInt, PfContainer* tempInt2) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 22;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;

	if (sc->zeropadBluestein[readWrite]) {
		if (readWrite) {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i/2;
		}
		else {
			//appendSetSMToZero(sc);
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i/2;
		}
	}
	else {
		fftDim.data.i = sc->fft_dim_full.data.i;
	}
	if (readWrite == 0) {
		temp_int.data.i = 2 * fftDim.data.i;
		PfIf_lt_start(sc, inoutID, &temp_int);
		if (sc->performDST == 4) {
			PfMovNeg(sc, &regID->data.c[1], &regID->data.c[1]);
		}
		if (sc->LUT) {
			temp_int.data.i = 2;
			PfDiv(sc, &sc->tempInt, inoutID, &temp_int);
			temp_int.data.i = fftDim.data.i/2 + 1;
			PfIf_ge_start(sc, &sc->tempInt, &temp_int);
			PfSub(sc, &sc->tempInt, &fftDim, &sc->tempInt);
			PfIf_end(sc);
			PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->startDCT3LUT);
			appendGlobalToRegisters(sc, &sc->mult, &sc->LUTStruct, &sc->tempInt);
			
			temp_int.data.i = 2*(fftDim.data.i/2 + 1);
			PfIf_ge_start(sc, inoutID, &temp_int);
			PfMov(sc, &sc->w.data.c[0], &sc->mult.data.c[1]);
			PfMov(sc, &sc->mult.data.c[1], &sc->mult.data.c[0]);
			PfMov(sc, &sc->mult.data.c[0], &sc->w.data.c[0]);
			PfIf_end(sc);
			temp_double.data.d = pfFPinit("2.0");
			PfMul(sc, &sc->mult, &sc->mult, &temp_double, 0);
		}
		else {
			temp_double.data.d = sc->double_PI / pfFPinit("4.0") / fftDim.data.i;

			PfMul(sc, &sc->w.data.c[0], inoutID, &temp_double, 0);
			PfSinCos(sc, &sc->mult, &sc->w.data.c[0]);
			temp_double.data.d = pfFPinit("2.0");
			PfMul(sc, &sc->mult, &sc->mult, &temp_double, 0);
		}
		PfMovNeg(sc, &regID->data.c[1], &regID->data.c[1]);
		PfMul(sc, regID, regID, &sc->mult, &sc->w);
		PfIf_end(sc);
	}
	return;
}
static inline void appendDCTIV_odd_read_processing (VkFFTSpecializationConstantsLayout* sc, PfContainer* inoutID, PfContainer* regID, int readWrite, PfContainer* tempInt, PfContainer* tempInt2) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 22;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;

	if (sc->zeropadBluestein[readWrite]) {
		if (readWrite) {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i;
		}
		else {
			//appendSetSMToZero(sc);
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i;
		}
	}
	else {
		fftDim.data.i = sc->fft_dim_full.data.i;
	}
	if (readWrite == 0) {
		if (sc->performDST)  {
			temp_int.data.i = 2;
			PfMod(sc, &sc->tempInt, inoutID, &temp_int);
			temp_int.data.i = 1;
			PfIf_eq_start(sc, &sc->tempInt, &temp_int);
			PfMovNeg(sc, regID, regID);
			PfIf_end(sc);
		}

		temp_int.data.i = fftDim.data.i * 2;
		PfIf_lt_start(sc, &sc->inoutID2, &temp_int); // &sc->stageInvocationID has original inoutID
		PfIf_ge_start(sc, &sc->inoutID2, &fftDim);
		PfMovNeg(sc, &regID->data.c[0], &regID->data.c[0]);
		PfMovNeg(sc, &regID->data.c[1], &regID->data.c[1]);
		PfIf_end(sc);
		PfIf_end(sc);

		temp_int.data.i = fftDim.data.i * 3;
		PfIf_lt_start(sc, &sc->inoutID2, &temp_int);
		temp_int.data.i = fftDim.data.i * 2;
		PfIf_ge_start(sc, &sc->inoutID2, &temp_int);
		PfMovNeg(sc, &regID->data.c[0], &regID->data.c[0]);
		PfMovNeg(sc, &regID->data.c[1], &regID->data.c[1]);
		PfIf_end(sc);
		PfIf_end(sc);

	}
	return;
}

static inline void appendDCTIV_even_write_processing (VkFFTSpecializationConstantsLayout* sc, PfContainer* inoutID, PfContainer* regID, int readWrite, PfContainer* tempInt, PfContainer* tempInt2) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 22;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;

	if (sc->zeropadBluestein[readWrite]) {
		if (readWrite) {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i/2;
		}
		else {
			//appendSetSMToZero(sc);
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i/2;
		}
	}
	else {
		fftDim.data.i = sc->fft_dim_full.data.i;
	}
	if (readWrite == 1) {
		if (sc->performDST) {
			temp_int.data.i = 2*fftDim.data.i - 1;
			PfSub(sc, inoutID, &temp_int, inoutID);
		}
		temp_int.data.i = 2;
		PfMod(sc, &sc->tempInt, inoutID, &temp_int);
		temp_int.data.i = 1;
		PfIf_eq_start(sc, &sc->tempInt, &temp_int);
		PfMovNeg(sc, &regID->data.c[1], &regID->data.c[1]);
		PfIf_end(sc);
		if (sc->LUT) {
			PfAdd(sc, &sc->tempInt, inoutID, &sc->startDCT4LUT);
			appendGlobalToRegisters(sc, &sc->mult, &sc->LUTStruct, &sc->tempInt);
		}
		else {
			temp_int.data.i = 2;
			PfMul(sc, &sc->tempInt, inoutID, &temp_int, 0);
			PfInc(sc, &sc->tempInt);
			temp_double.data.d = -sc->double_PI / pfFPinit("8.0") / fftDim.data.i;
			PfMul(sc, &sc->w.data.c[0], &sc->tempInt, &temp_double, 0);
			PfSinCos(sc, &sc->mult, &sc->w.data.c[0]);
		}
			
		PfMovNeg(sc, &regID->data.c[1], &regID->data.c[1]);
		PfMul(sc, regID, regID, &sc->mult, &sc->w);
		PfMovNeg(sc, &regID->data.c[1], &regID->data.c[1]);
		if (sc->performDST) {
			temp_int.data.i = 2*fftDim.data.i - 1;
			PfSub(sc, inoutID, &temp_int, inoutID);
		}
	}

	return;
}
static inline void appendDCTIV_odd_write_processing (VkFFTSpecializationConstantsLayout* sc, PfContainer* inoutID, PfContainer* regID, int readWrite, PfContainer* tempInt, PfContainer* tempInt2) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 22;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;

	if (sc->zeropadBluestein[readWrite]) {
		if (readWrite) {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i;
		}
		else {
			//appendSetSMToZero(sc);
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i;
		}
	}
	else {
		fftDim.data.i = sc->fft_dim_full.data.i;
	}
	if (readWrite == 1) {
		if (sc->performDST) {
			temp_int.data.i = fftDim.data.i - 1;
			PfSub(sc, inoutID, &temp_int, inoutID);
		}
		temp_int.data.i = (fftDim.data.i+1) / 4;
		PfIf_ge_start(sc, inoutID, &temp_int);
		temp_int.data.i = (3*fftDim.data.i) / 4;
		if(((fftDim.data.i-1) % 4) == 0)
			PfIf_le_start(sc, inoutID, &temp_int);
		else 
			PfIf_lt_start(sc, inoutID, &temp_int);
		PfMovNeg(sc, &regID->data.c[1], &regID->data.c[1]);
		PfIf_end(sc);
		PfIf_end(sc);

		temp_int.data.i = 4;
		PfMod(sc, &sc->tempInt, inoutID, &temp_int);


		temp_int.data.i = 0;
		PfIf_eq_start(sc, &sc->tempInt, &temp_int);
		PfSub(sc, &regID->data.c[0], &regID->data.c[0], &regID->data.c[1]);
		PfIf_end(sc);
		temp_int.data.i = 1;
		PfIf_eq_start(sc, &sc->tempInt, &temp_int);
		PfMovNeg(sc, &regID->data.c[0], &regID->data.c[0]);
		PfSub(sc, &regID->data.c[0], &regID->data.c[0], &regID->data.c[1]);
		PfIf_end(sc);
		temp_int.data.i = 2;
		PfIf_eq_start(sc, &sc->tempInt, &temp_int);
		PfSub(sc, &regID->data.c[0], &regID->data.c[1], &regID->data.c[0]);
		PfIf_end(sc);
		temp_int.data.i = 3;
		PfIf_eq_start(sc, &sc->tempInt, &temp_int);
		PfAdd(sc, &regID->data.c[0], &regID->data.c[0], &regID->data.c[1]);
		PfIf_end(sc);

		/*
		temp_int.data.i = 2;
		PfMod(sc, &sc->tempInt, &sc->stageInvocationID, &temp_int);

		temp_int.data.i = 1;
		PfIf_eq_start(sc, &sc->tempInt, &temp_int);

		temp_int.data.i = (fftDim.data.i+1) / 2;
		PfIf_lt_start(sc, &sc->stageInvocationID, &temp_int);
		temp_int.data.i = 2;
		PfDiv(sc, &sc->tempInt, &sc->stageInvocationID, &temp_int);
		temp_int.data.i = 4;
		PfMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);

		temp_int.data.i = 0;
		PfIf_eq_start(sc, &sc->tempInt, &temp_int);
		PfSub(sc, &regID->data.c[0], &regID->data.c[0], &regID->data.c[1]);
		PfIf_end(sc);
		temp_int.data.i = 1;
		PfIf_eq_start(sc, &sc->tempInt, &temp_int);
		PfMovNeg(sc, &regID->data.c[0], &regID->data.c[0]);
		PfSub(sc, &regID->data.c[0], &regID->data.c[0], &regID->data.c[1]);
		PfIf_end(sc);
		temp_int.data.i = 2;
		PfIf_eq_start(sc, &sc->tempInt, &temp_int);
		PfSub(sc, &regID->data.c[0], &regID->data.c[1], &regID->data.c[0]);
		PfIf_end(sc);
		temp_int.data.i = 3;
		PfIf_eq_start(sc, &sc->tempInt, &temp_int);
		PfAdd(sc, &regID->data.c[0], &regID->data.c[0], &regID->data.c[1]);
		PfIf_end(sc);

		PfIf_else(sc);
		PfSetToZero(sc, regID);
		//PfSub(sc, &regID->data.c[0], &regID->data.c[0], &regID->data.c[1]);
		
		PfIf_end(sc);
		PfIf_else(sc);

		temp_int.data.i = (fftDim.data.i+1) / 2;
		PfIf_lt_start(sc, &sc->stageInvocationID, &temp_int);
		PfSetToZero(sc, regID);
		//PfSub(sc, &regID->data.c[0], &regID->data.c[0], &regID->data.c[1]);
		
		PfIf_else(sc);
		PfSetToZero(sc, regID);
		//PfAdd(sc, &regID->data.c[0], &regID->data.c[0], &regID->data.c[1]);
		
		PfIf_end(sc);
		PfIf_end(sc);
		
		temp_double.data.d = pfFPinit("1.41421356237309504880168872420969807856967");
		PfMul(sc, &regID->data.c[0], &regID->data.c[0], &temp_double, 0);
		/*temp_int.data.i = fftDim.data.i / 4;
		PfIf_lt_start(sc, &sc->stageInvocationID, &temp_int);

		temp_int.data.i = 1;
		PfAdd(sc, &sc->tempInt, &sc->stageInvocationID, &temp_int);
		temp_int.data.i = 2;
		PfDiv(sc, &sc->tempInt, &sc->tempInt, &temp_int);
		PfMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->tempInt, &temp_int);
		PfMovNeg(sc, &sc->w.data.c[0], &regID->data.c[0]);
		PfIf_else(sc);
		PfMov(sc, &sc->w.data.c[0], &regID->data.c[0]);
		PfIf_end(sc);

		temp_int.data.i = 2;
		PfDiv(sc, &sc->tempInt, &sc->stageInvocationID, &temp_int);
		PfMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->tempInt, &temp_int);
		PfAdd(sc, &sc->w.data.c[0], &sc->w.data.c[0], &regID->data.c[1]);
		PfIf_else(sc);
		PfSub(sc, &sc->w.data.c[0], &sc->w.data.c[0], &regID->data.c[1]);
		PfIf_end(sc);

		PfIf_end(sc);


		temp_int.data.i = fftDim.data.i / 2;
		PfIf_lt_start(sc, &sc->stageInvocationID, &temp_int);
		temp_int.data.i = fftDim.data.i / 4;
		PfIf_ge_start(sc, &sc->stageInvocationID, &temp_int);

		temp_int.data.i = 1;
		PfAdd(sc, &sc->tempInt, &sc->stageInvocationID, &temp_int);
		temp_int.data.i = 2;
		PfDiv(sc, &sc->tempInt, &sc->tempInt, &temp_int);
		PfMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->tempInt, &temp_int);
		PfMovNeg(sc, &sc->w.data.c[0], &regID->data.c[0]);
		PfIf_else(sc);
		PfMov(sc, &sc->w.data.c[0], &regID->data.c[0]);
		PfIf_end(sc);

		temp_int.data.i = 2;
		PfDiv(sc, &sc->tempInt, &sc->stageInvocationID, &temp_int);
		PfMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->tempInt, &temp_int);
		PfSub(sc, &sc->w.data.c[0], &sc->w.data.c[0], &regID->data.c[1]);
		PfIf_else(sc);
		PfAdd(sc, &sc->w.data.c[0], &sc->w.data.c[0], &regID->data.c[1]);
		PfIf_end(sc);

		PfIf_end(sc);
		PfIf_end(sc);


		temp_int.data.i = 3 * fftDim.data.i / 4;
		PfIf_lt_start(sc, &sc->stageInvocationID, &temp_int);
		temp_int.data.i = fftDim.data.i / 2;
		PfIf_ge_start(sc, &sc->stageInvocationID, &temp_int);
		
		temp_int.data.i = 1;
		PfAdd(sc, &sc->tempInt, &sc->stageInvocationID, &temp_int);
		temp_int.data.i = 2;
		PfDiv(sc, &sc->tempInt, &sc->tempInt, &temp_int);
		PfMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->tempInt, &temp_int);
		PfMovNeg(sc, &sc->w.data.c[0], &regID->data.c[0]);

		PfIf_else(sc);
		PfMov(sc, &sc->w.data.c[0], &regID->data.c[0]);
		PfIf_end(sc);

		temp_int.data.i = 2;
		PfDiv(sc, &sc->tempInt, &sc->stageInvocationID, &temp_int);
		PfMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->tempInt, &temp_int);
		PfAdd(sc, &sc->w.data.c[0], &sc->w.data.c[0], &regID->data.c[1]);
		PfIf_else(sc);
		PfSub(sc, &sc->w.data.c[0], &sc->w.data.c[0], &regID->data.c[1]);
		PfIf_end(sc);

		PfIf_end(sc);
		PfIf_end(sc);


		temp_int.data.i = 3 * fftDim.data.i / 4;
		PfIf_ge_start(sc, &sc->stageInvocationID, &temp_int);
		
		temp_int.data.i = 1;
		PfAdd(sc, &sc->tempInt, &sc->stageInvocationID, &temp_int);
		temp_int.data.i = 2;
		PfDiv(sc, &sc->tempInt, &sc->tempInt, &temp_int);
		PfMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->tempInt, &temp_int);
		PfMovNeg(sc, &sc->w.data.c[0], &regID->data.c[0]);
		PfIf_else(sc);
		PfMov(sc, &sc->w.data.c[0], &regID->data.c[0]);
		PfIf_end(sc);

		temp_int.data.i = 2;
		PfDiv(sc, &sc->tempInt, &sc->stageInvocationID, &temp_int);
		PfMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->tempInt, &temp_int);
		PfSub(sc, &sc->w.data.c[0], &sc->w.data.c[0], &regID->data.c[1]);
		PfIf_else(sc);
		PfAdd(sc, &sc->w.data.c[0], &sc->w.data.c[0], &regID->data.c[1]);
		PfIf_end(sc);

		PfIf_end(sc);
		*/
		temp_double.data.d = pfFPinit("1.41421356237309504880168872420969807856967");
		PfMul(sc, &regID->data.c[0], &regID->data.c[0], &temp_double, 0);
		if (sc->performDST) {
			temp_int.data.i = fftDim.data.i - 1;
			PfSub(sc, inoutID, &temp_int, inoutID);
		}
	}

	return;
}

static inline void append_inoutID_preprocessing_multiupload_R2R(VkFFTSpecializationConstantsLayout* sc, PfContainer* inoutID, int readWrite, int type,  PfContainer* tempInt, PfContainer* tempInt2) {
	if (((type/10) == 111) && ((sc->inputMemoryCode % 10) == 2))
		appendDCTI_read_get_inoutID(sc, inoutID, readWrite, tempInt);
	if (((type/10) == 111) && ((sc->outputMemoryCode % 10) == 2))
		appendDCTI_write_get_inoutID(sc, inoutID, readWrite, tempInt);
	if (((type/10) == 121) && ((sc->inputMemoryCode % 10) == 2))
		appendDCTII_read_III_write_get_inoutID(sc, inoutID, readWrite, tempInt, tempInt2);
	if (((type/10) == 131) && ((sc->inputMemoryCode % 10) == 2))
		appendDCTII_write_III_read_get_inoutID(sc, inoutID, readWrite, tempInt, tempInt2);
	if (((type/10) == 121) && ((sc->outputMemoryCode % 10) == 2))
		appendDCTII_write_III_read_get_inoutID(sc, inoutID, readWrite, tempInt, tempInt2);
	if (((type/10) == 131) && ((sc->outputMemoryCode % 10) == 2))
		appendDCTII_read_III_write_get_inoutID(sc, inoutID, readWrite, tempInt, tempInt2);
	if (((type/10) == 141) && ((sc->inputMemoryCode % 10) == 2))
		appendDCTIV_even_read_get_inoutID(sc, inoutID, readWrite, tempInt);
	if (((type/10) == 141) && ((sc->outputMemoryCode % 10) == 2))
		appendDCTIV_even_write_get_inoutID(sc, inoutID, readWrite, tempInt);
	if (((type/10) == 143) && ((sc->inputMemoryCode % 10) == 2))
		appendDCTIV_odd_read_get_inoutID(sc, inoutID, readWrite, tempInt, tempInt2);
	if (((type/10) == 143) && ((sc->outputMemoryCode % 10) == 2))
		appendDCTIV_odd_write_get_inoutID(sc, inoutID, readWrite, tempInt);
}
static inline void append_processing_multiupload_R2R(VkFFTSpecializationConstantsLayout* sc, PfContainer* inoutID, PfContainer* regID, int readWrite, int type, PfContainer* tempInt, PfContainer* tempInt2) {
	if (((type/10) == 111) && ((sc->inputMemoryCode % 10) == 2))
		appendDCTI_processing(sc, inoutID, regID, readWrite, tempInt, tempInt2);
	if (((type/10) == 121) && ((sc->inputMemoryCode % 10) == 2))
		appendDCTII_read_III_write_processing(sc, inoutID, regID, readWrite, tempInt, tempInt2);
	if (((type/10) == 121) && ((sc->outputMemoryCode % 10) == 2))
		appendDCTII_write_III_read_processing(sc, inoutID, regID, readWrite, tempInt, tempInt2);
	if (((type/10) == 131) && ((sc->inputMemoryCode % 10) == 2))
		appendDCTII_write_III_read_processing(sc, inoutID, regID, readWrite, tempInt, tempInt2);
	if (((type/10) == 131) && ((sc->outputMemoryCode % 10) == 2))
		appendDCTII_read_III_write_processing(sc, inoutID, regID, readWrite, tempInt, tempInt2);
	if (((type/10) == 141) && ((sc->inputMemoryCode % 10) == 2))
		appendDCTIV_even_read_processing(sc, inoutID, regID, readWrite, tempInt, tempInt2);
	if (((type/10) == 141) && ((sc->outputMemoryCode % 10) == 2))
		appendDCTIV_even_write_processing(sc, inoutID, regID, readWrite, tempInt, tempInt2);
	if (((type/10) == 143) && ((sc->inputMemoryCode % 10) == 2))
		appendDCTIV_odd_read_processing(sc, inoutID, regID, readWrite, tempInt, tempInt2);
	if (((type/10) == 143) && ((sc->outputMemoryCode % 10) == 2))
		appendDCTIV_odd_write_processing(sc, inoutID, regID, readWrite, tempInt, tempInt2);
}
static inline void append_inoutID_postprocessing_multiupload_R2R(VkFFTSpecializationConstantsLayout* sc, PfContainer* inoutID, int readWrite, int type, PfContainer* tempInt) {
	if (((type/10) == 111) && ((sc->inputMemoryCode % 10) == 2))
		appendDCTI_read_set_inoutID(sc, inoutID, readWrite, tempInt);
	if (((type/10) == 111) && ((sc->outputMemoryCode % 10) == 2))
		appendDCTI_write_set_inoutID(sc, inoutID, readWrite, tempInt);
	if (((type/10) == 121) && ((sc->inputMemoryCode % 10) == 2))
		appendDCTII_read_III_write_set_inoutID(sc, inoutID, readWrite, tempInt);
	if (((type/10) == 131) && ((sc->inputMemoryCode % 10) == 2))
		appendDCTII_write_III_read_set_inoutID(sc, inoutID, readWrite, tempInt);
	if (((type/10) == 121) && ((sc->outputMemoryCode % 10) == 2))
		appendDCTII_write_III_read_set_inoutID(sc, inoutID, readWrite, tempInt);
	if (((type/10) == 131) && ((sc->outputMemoryCode % 10) == 2))
		appendDCTII_read_III_write_set_inoutID(sc, inoutID, readWrite, tempInt);
	if (((type/10) == 141) && ((sc->inputMemoryCode % 10) == 2))
		appendDCTIV_even_read_set_inoutID(sc, inoutID, readWrite, tempInt);
	if (((type/10) == 141) && ((sc->outputMemoryCode % 10) == 2))
		appendDCTIV_even_write_set_inoutID(sc, inoutID, readWrite, tempInt);
	if (((type/10) == 143) && ((sc->inputMemoryCode % 10) == 2))
		appendDCTIV_odd_read_set_inoutID(sc, inoutID, readWrite, tempInt);
	if (((type/10) == 143) && ((sc->outputMemoryCode % 10) == 2))
		appendDCTIV_odd_write_set_inoutID(sc, inoutID, readWrite, tempInt);
}

static inline void appendDCTI_read(VkFFTSpecializationConstantsLayout* sc, int type, int readWrite) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_int1 = VKFFT_ZERO_INIT;
	temp_int1.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 22;

	PfContainer used_registers = VKFFT_ZERO_INIT;
	used_registers.type = 31;
	
	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;

	PfContainer localSize = VKFFT_ZERO_INIT;
	localSize.type = 31;

	PfContainer batching_localSize = VKFFT_ZERO_INIT;
	batching_localSize.type = 31;

	PfContainer* localInvocationID = VKFFT_ZERO_INIT;
	PfContainer* batchingInvocationID = VKFFT_ZERO_INIT;

	if (sc->stridedSharedLayout) {
		batching_localSize.data.i = sc->localSize[0].data.i;
		localSize.data.i = sc->localSize[1].data.i;
		localInvocationID = &sc->gl_LocalInvocationID_y;
		batchingInvocationID = &sc->gl_LocalInvocationID_x;
	}
	else {
		batching_localSize.data.i = sc->localSize[1].data.i;
		localSize.data.i = sc->localSize[0].data.i;
		localInvocationID = &sc->gl_LocalInvocationID_x;
		batchingInvocationID = &sc->gl_LocalInvocationID_y;
	}

	if (sc->zeropadBluestein[readWrite]) {
		if (readWrite) {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i;
		}
		else {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i;
		}
	}
	else {
		if(sc->performDCT)
			fftDim.data.i = (sc->fftDim.data.i + 2) / 2;
		if(sc->performDST)
			fftDim.data.i = (sc->fftDim.data.i - 2) / 2;
	}

	if (sc->stridedSharedLayout) {
		PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[1]);
	}
	else {
		PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[0]);
	}

	appendBarrierVkFFT(sc);
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	for (pfUINT i = 0; i < (pfUINT)used_registers.data.i; i++) {
		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i)*sc->localSize[1].data.i;

			PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		else {
			if (sc->localSize[1].data.i == 1) {
				temp_int.data.i = (i)*sc->localSize[0].data.i;

				PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
			}
			else {
				PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

				temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

				PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
				PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
			}
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		if (sc->performDCT) {
			if (sc->stridedSharedLayout) {
				temp_int.data.i = 0;
				PfIf_gt_start(sc, &sc->combinedID, &temp_int);
				temp_int.data.i = fftDim.data.i - 1;
				PfIf_lt_start(sc, &sc->combinedID, &temp_int);
			}
			else {
				PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
				temp_int.data.i = 0;
				PfIf_gt_start(sc, &sc->tempInt, &temp_int);
				temp_int.data.i = fftDim.data.i - 1;
				PfIf_lt_start(sc, &sc->tempInt, &temp_int);
			}
		}
		if (sc->performDST) {
			if (!sc->stridedSharedLayout) {
				PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			}
		}
		if (sc->stridedSharedLayout) {
			PfMul(sc, &sc->tempInt, &sc->combinedID, &sc->sharedStride, 0);
			if (sc->performDCT)
				temp_int.data.i = (2 * fftDim.data.i - 2) * sc->sharedStride.data.i;
			if (sc->performDST)
				temp_int.data.i = (2 * fftDim.data.i + 1) * sc->sharedStride.data.i;
			PfAdd(sc, &sc->inoutID, &sc->gl_LocalInvocationID_x, &temp_int);
			PfSub(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);

			PfAdd(sc, &sc->sdataID, &sc->gl_LocalInvocationID_x, &sc->tempInt);

			if (sc->performDST) {
				if (i == 0) {
					temp_int.data.i = 0;
					PfIf_eq_start(sc, &sc->combinedID, &temp_int);
					PfSetToZeroShared(sc, &sc->sdataID);
					temp_int.data.i = (fftDim.data.i + 1) * sc->sharedStride.data.i;
					PfAdd(sc, &sc->tempInt, &sc->sdataID, &temp_int);
					PfSetToZeroShared(sc, &sc->tempInt);
					PfIf_end(sc);
				}
				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride);
			}
		}
		else {
			PfDiv(sc, &sc->sdataID, &sc->combinedID, &fftDim);
			PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
			if (sc->performDCT)
				temp_int.data.i = (2 * fftDim.data.i - 2);
			if (sc->performDST)
				temp_int.data.i = (2 * fftDim.data.i + 1);

			PfAdd(sc, &sc->inoutID, &sc->sdataID, &temp_int);
			PfSub(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);

			PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);

			if (sc->performDST) {
				if (i == 0) {
					temp_int.data.i = 0;
					PfIf_eq_start(sc, &sc->tempInt, &temp_int);
					PfSetToZeroShared(sc, &sc->sdataID);
					temp_int.data.i = (fftDim.data.i + 1);
					PfAdd(sc, &sc->tempInt, &sc->sdataID, &temp_int);
					PfSetToZeroShared(sc, &sc->tempInt);
					PfIf_end(sc);
				}
				PfInc(sc, &sc->sdataID);
			}
		}
		appendSharedToRegisters(sc, &sc->temp, &sc->sdataID);
		if (sc->performDST)
			PfMovNeg(sc, &sc->temp, &sc->temp);
		appendRegistersToShared(sc, &sc->inoutID, &sc->temp);

		if (sc->performDCT) {
			PfIf_end(sc);
			PfIf_end(sc);
		}
		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_end(sc);
			}
		}
		else {
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
	}
	if (sc->useDisableThreads) {
		PfIf_end(sc);
	}
	return;
}

static inline void appendDCTII_read_III_write(VkFFTSpecializationConstantsLayout* sc, int type, int readWrite) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_int1 = VKFFT_ZERO_INIT;
	temp_int1.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 22;

	PfContainer used_registers = VKFFT_ZERO_INIT;
	used_registers.type = 31;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;

	PfContainer localSize = VKFFT_ZERO_INIT;
	localSize.type = 31;

	PfContainer batching_localSize = VKFFT_ZERO_INIT;
	batching_localSize.type = 31;

	PfContainer* localInvocationID = VKFFT_ZERO_INIT;
	PfContainer* batchingInvocationID = VKFFT_ZERO_INIT;

	if (sc->stridedSharedLayout) {
		batching_localSize.data.i = sc->localSize[0].data.i;
		localSize.data.i = sc->localSize[1].data.i;
		localInvocationID = &sc->gl_LocalInvocationID_y;
		batchingInvocationID = &sc->gl_LocalInvocationID_x;
	}
	else {
		batching_localSize.data.i = sc->localSize[1].data.i;
		localSize.data.i = sc->localSize[0].data.i;
		localInvocationID = &sc->gl_LocalInvocationID_x;
		batchingInvocationID = &sc->gl_LocalInvocationID_y;
	}

	if (sc->zeropadBluestein[readWrite]) {
		if (readWrite) {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i;
		}
		else {
			appendSetSMToZero(sc);
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i;
		}
	}
	else {
		fftDim.data.i = sc->fftDim.data.i;
	}

	if (sc->stridedSharedLayout) {
		PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[1]);
	}
	else {
		PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[0]);
	}

	appendBarrierVkFFT(sc);
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	for (pfUINT i = 0; i < (pfUINT)used_registers.data.i; i++) {
		if (sc->axis_id > 0) {
			temp_int.data.i = (i)*sc->localSize[1].data.i;

			PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		else {
			if (sc->localSize[1].data.i == 1) {
				temp_int.data.i = (i)*sc->localSize[0].data.i;

				PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
			}
			else {
				PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

				temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

				PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
				PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
			}

			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		if (sc->axis_id > 0){
			temp_int.data.i = 2;
			PfMod(sc, &sc->sdataID, &sc->combinedID, &temp_int);
		}
		else {
			PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			temp_int.data.i = 2;
			PfMod(sc, &sc->sdataID, &sc->tempInt, &temp_int);
		}
		if (((sc->performDST == 2) && (sc->actualInverse == 0)) || ((sc->performDST == 3) && (sc->actualInverse == 1))) {
			temp_int.data.i = 1;
			PfIf_eq_start(sc, &sc->sdataID, &temp_int);
			PfMovNeg(sc, &sc->regIDs[i], &sc->regIDs[i]);
			PfIf_end(sc);
		}
		temp_int.data.i = 2;
		PfMul(sc, &sc->blockInvocationID, &sc->sdataID, &temp_int, 0);
		temp_int.data.i = 2;
		if (sc->axis_id > 0) {
			PfDiv(sc, &sc->tempInt, &sc->combinedID, &temp_int);
		}
		else {
			PfDiv(sc, &sc->tempInt, &sc->tempInt, &temp_int);
		}
		PfMul(sc, &sc->blockInvocationID, &sc->blockInvocationID, &sc->tempInt, 0);
		
		temp_int.data.i = fftDim.data.i - 1;
		PfMul(sc, &sc->sdataID, &sc->sdataID, &temp_int, 0);
		PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
		PfSub(sc, &sc->sdataID, &sc->sdataID, &sc->blockInvocationID);

		if (sc->axis_id > 0) {
			PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
			PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);
		}
		else {
			if (sc->stridedSharedLayout) {
				PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
				PfDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
			}
			else {
				PfDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
				PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
			}
		}
		if(readWrite)
			appendSharedToRegisters(sc, &sc->regIDs[i], &sc->sdataID);
		else
			appendRegistersToShared(sc, &sc->sdataID, &sc->regIDs[i]);

		if (((sc->performDST == 2) && (sc->actualInverse == 1)) || ((sc->performDST == 3) && (sc->actualInverse == 0)))  {
			if (sc->axis_id > 0){
				temp_int.data.i = 2;
				PfMod(sc, &sc->sdataID, &sc->combinedID, &temp_int);
			}
			else {
				PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
				temp_int.data.i = 2;
				PfMod(sc, &sc->sdataID, &sc->tempInt, &temp_int);
			}
			temp_int.data.i = 1;
			PfIf_eq_start(sc, &sc->sdataID, &temp_int);
			PfMovNeg(sc, &sc->regIDs[i], &sc->regIDs[i]);
			PfIf_end(sc);
		}
		if (sc->axis_id > 0) {
			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
		else {
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
	}
	if (sc->useDisableThreads) {
		PfIf_end(sc);
	}
	if (readWrite)
		sc->writeFromRegisters = 1;
	else
		sc->readToRegisters = 0;
	return;
}
static inline void appendDCTII_write_III_read(VkFFTSpecializationConstantsLayout* sc, int type, int readWrite) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_int1 = VKFFT_ZERO_INIT;
	temp_int1.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 22;

	PfContainer used_registers = VKFFT_ZERO_INIT;
	used_registers.type = 31;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;

	PfContainer fftDim_half = VKFFT_ZERO_INIT;
	fftDim_half.type = 31;

	PfContainer localSize = VKFFT_ZERO_INIT;
	localSize.type = 31;

	PfContainer batching_localSize = VKFFT_ZERO_INIT;
	batching_localSize.type = 31;

	PfContainer* localInvocationID = VKFFT_ZERO_INIT;
	PfContainer* batchingInvocationID = VKFFT_ZERO_INIT;

	if (sc->stridedSharedLayout) {
		batching_localSize.data.i = sc->localSize[0].data.i;
		localSize.data.i = sc->localSize[1].data.i;
		localInvocationID = &sc->gl_LocalInvocationID_y;
		batchingInvocationID = &sc->gl_LocalInvocationID_x;
	}
	else {
		batching_localSize.data.i = sc->localSize[1].data.i;
		localSize.data.i = sc->localSize[0].data.i;
		localInvocationID = &sc->gl_LocalInvocationID_x;
		batchingInvocationID = &sc->gl_LocalInvocationID_y;
	}

	if (sc->zeropadBluestein[readWrite]) {
		if (readWrite) {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i;
		}
		else {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i;
		}
	}
	else {
		fftDim.data.i = sc->fftDim.data.i;
	}

	if (((sc->performDST == 2) && (sc->actualInverse == 1)) || ((sc->performDST == 3) && (sc->actualInverse == 0))) {
		appendBarrierVkFFT(sc);
		if (sc->stridedSharedLayout) {
			PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[1]);
		}
		else {
			PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[0]);
		}
		if (sc->useDisableThreads) {
			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
		}
		for (pfUINT i = 0; i < (pfUINT)used_registers.data.i; i++) {
			if (sc->stridedSharedLayout) {
				temp_int.data.i = (i)*sc->localSize[1].data.i;

				PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

				temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					//check that we only read fftDim * local batch data
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
					PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
				}
			}
			else {
				if (sc->localSize[1].data.i == 1) {
					temp_int.data.i = (i)*sc->localSize[0].data.i;

					PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
				}
				else {
					PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

					temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

					PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
					PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
				}
				temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					//check that we only read fftDim * local batch data
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim_half * &sc->localSize[0]);
					PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
				}
			}
			if (sc->stridedSharedLayout) {
				temp_int.data.i = fftDim.data.i - 1;
				PfSub(sc, &sc->combinedID, &temp_int, &sc->combinedID);

				PfMul(sc, &sc->sdataID, &sc->combinedID, &sc->sharedStride, 0);

				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);
			}
			else {
				PfMod(sc, &sc->sdataID, &sc->combinedID, &fftDim);
				temp_int.data.i = fftDim.data.i - 1;
				PfSub(sc, &sc->sdataID, &temp_int, &sc->sdataID);

				PfDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
				PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);

				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
			}
			appendSharedToRegisters(sc, &sc->regIDs[i], &sc->sdataID);
			if (sc->stridedSharedLayout) {
				temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					PfIf_end(sc);
				}
			}
			else {
				temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					PfIf_end(sc);
				}
			}
		}
		if (sc->useDisableThreads) {
			PfIf_end(sc);
		}
		appendBarrierVkFFT(sc);
		
		if (sc->useDisableThreads) {
			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
		}
		for (pfUINT i = 0; i < (pfUINT)used_registers.data.i; i++) {
			if (sc->stridedSharedLayout) {
				temp_int.data.i = (i)*sc->localSize[1].data.i;

				PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

				temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					//check that we only read fftDim * local batch data
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
					PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
				}
			}
			else {
				if (sc->localSize[1].data.i == 1) {
					temp_int.data.i = (i)*sc->localSize[0].data.i;

					PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
				}
				else {
					PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

					temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

					PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
					PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
				}
				temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					//check that we only read fftDim * local batch data
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim_half * &sc->localSize[0]);
					PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
				}
			}
			if (sc->stridedSharedLayout) {
				PfMul(sc, &sc->sdataID, &sc->combinedID, &sc->sharedStride, 0);

				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);
			}
			else {
				PfMod(sc, &sc->sdataID, &sc->combinedID, &fftDim);
				
				PfDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
				PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);

				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
			}
			appendRegistersToShared(sc, &sc->sdataID, &sc->regIDs[i]);
			if (sc->stridedSharedLayout) {
				temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					PfIf_end(sc);
				}
			}
			else {
				temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					PfIf_end(sc);
				}
			}
		}
		if (sc->useDisableThreads) {
			PfIf_end(sc);
		}
	}

	temp_int.data.i = 2;
	PfDiv(sc, &fftDim_half, &fftDim, &temp_int);
	PfInc(sc, &fftDim_half);

	if (sc->stridedSharedLayout) {
		PfDivCeil(sc, &used_registers, &fftDim_half, &sc->localSize[1]);
	}
	else {
		PfDivCeil(sc, &used_registers, &fftDim_half, &sc->localSize[0]);
	}

	appendBarrierVkFFT(sc);
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	for (pfUINT i = 0; i < (pfUINT)used_registers.data.i; i++) {
		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i)*sc->localSize[1].data.i;

			PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim_half.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		else {
			if (sc->localSize[1].data.i == 1) {
				temp_int.data.i = (i)*sc->localSize[0].data.i;

				PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
			}
			else {
				PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

				temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

				PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
				PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
			}
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim_half.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim_half * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim_half * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}

		if (sc->LUT) {
			if (sc->stridedSharedLayout) {
				PfAdd(sc, &sc->tempInt, &sc->combinedID, &sc->startDCT3LUT);
			}
			else {
				PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim_half);
				PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->startDCT3LUT);
			}
			appendGlobalToRegisters(sc, &sc->mult, &sc->LUTStruct, &sc->tempInt);
			if ((!sc->mergeSequencesR2C) && (readWrite)) {
				temp_double.data.d = pfFPinit("2.0");
				PfMul(sc, &sc->mult, &sc->mult, &temp_double, 0);
			}
			if (readWrite)
				PfConjugate(sc, &sc->mult, &sc->mult);
		}
		else {
			if (readWrite)
				temp_double.data.d = -sc->double_PI / pfFPinit("2.0") / fftDim.data.i;
			else
				temp_double.data.d = sc->double_PI / pfFPinit("2.0") / fftDim.data.i;
			if (sc->stridedSharedLayout) {
				PfMul(sc, &sc->tempFloat, &sc->combinedID, &temp_double, 0);
			}
			else {
				PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim_half);
				PfMul(sc, &sc->tempFloat, &sc->tempInt, &temp_double, 0);
			}

			PfSinCos(sc, &sc->mult, &sc->tempFloat);
			if ((!sc->mergeSequencesR2C) && (readWrite)) {
				temp_double.data.d = pfFPinit("2.0");
				PfMul(sc, &sc->mult, &sc->mult, &temp_double, 0);
			}
		}

		if (sc->stridedSharedLayout) {
			PfMul(sc, &sc->sdataID, &sc->combinedID, &sc->sharedStride, 0);

			temp_int.data.i = fftDim.data.i * sc->sharedStride.data.i;
			PfSub(sc, &sc->inoutID, &temp_int, &sc->sdataID);

			temp_int.data.i = 0;
			PfIf_eq_start(sc, &sc->sdataID, &temp_int);
			PfMov(sc, &sc->inoutID, &sc->sdataID);
			PfIf_end(sc);

			PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);
			PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->gl_LocalInvocationID_x);
		}
		else {
			PfMod(sc, &sc->sdataID, &sc->combinedID, &fftDim_half);
			if (sc->stridedSharedLayout) {
				PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);

				temp_int.data.i = fftDim.data.i * sc->sharedStride.data.i;
				PfSub(sc, &sc->inoutID, &temp_int, &sc->sdataID);

				temp_int.data.i = 0;
				PfIf_eq_start(sc, &sc->sdataID, &temp_int);
				PfMov(sc, &sc->inoutID, &sc->sdataID);
				PfIf_end(sc);

				PfDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim_half);
				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
			}
			else {
				temp_int.data.i = fftDim.data.i;
				PfSub(sc, &sc->inoutID, &temp_int, &sc->sdataID);

				temp_int.data.i = 0;
				PfIf_eq_start(sc, &sc->sdataID, &temp_int);
				PfMov(sc, &sc->inoutID, &sc->sdataID);
				PfIf_end(sc);

				PfDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim_half);
				PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);

				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
			}
		}

		if (readWrite) {
			appendSharedToRegisters(sc, &sc->temp, &sc->sdataID);
			if (sc->mergeSequencesR2C) {
				appendSharedToRegisters(sc, &sc->w, &sc->inoutID);

				PfAdd(sc, &sc->regIDs[0].data.c[0], &sc->temp.data.c[0], &sc->w.data.c[0]);
				PfSub(sc, &sc->regIDs[0].data.c[1], &sc->temp.data.c[1], &sc->w.data.c[1]);
				PfSub(sc, &sc->regIDs[1].data.c[0], &sc->w.data.c[0], &sc->temp.data.c[0]);
				PfAdd(sc, &sc->regIDs[1].data.c[1], &sc->temp.data.c[1], &sc->w.data.c[1]);

				PfMul(sc, &sc->temp, &sc->regIDs[0], &sc->mult, 0);
				PfConjugate(sc, &sc->mult, &sc->mult);
				PfMul(sc, &sc->w, &sc->regIDs[1], &sc->mult, 0);
				PfMov(sc, &sc->regIDs[0].data.c[0], &sc->temp.data.c[0]);
				PfMov(sc, &sc->regIDs[0].data.c[1], &sc->w.data.c[1]);
				PfMovNeg(sc, &sc->regIDs[1].data.c[0], &sc->temp.data.c[1]);
				PfMovNeg(sc, &sc->regIDs[1].data.c[1], &sc->w.data.c[0]);

				appendRegistersToShared(sc, &sc->inoutID, &sc->regIDs[1]);
				appendRegistersToShared(sc, &sc->sdataID, &sc->regIDs[0]);
			}
			else {
				PfMul(sc, &sc->regIDs[0], &sc->temp, &sc->mult, 0);
				PfMovNeg(sc, &sc->w.data.c[0], &sc->regIDs[0].data.c[1]);

				appendRegistersToShared(sc, &sc->inoutID, &sc->w);
				appendRegistersToShared(sc, &sc->sdataID, &sc->regIDs[0]);
			}
		}
		else {
			appendSharedToRegisters(sc, &sc->temp, &sc->sdataID);
			appendSharedToRegisters(sc, &sc->w, &sc->inoutID);

			PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim_half);
			temp_int.data.i = 0;
			PfIf_eq_start(sc, &sc->tempInt, &temp_int);
			PfSetToZero(sc, &sc->w);
			PfIf_end(sc);

			PfMov(sc, &sc->regIDs[0].data.c[0], &sc->w.data.c[1]);
			PfMov(sc, &sc->regIDs[0].data.c[1], &sc->w.data.c[0]);

			PfSub(sc, &sc->regIDs[1].data.c[0], &sc->temp.data.c[0], &sc->regIDs[0].data.c[0]);
			PfAdd(sc, &sc->regIDs[1].data.c[1], &sc->temp.data.c[1], &sc->regIDs[0].data.c[1]);

			PfAdd(sc, &sc->w.data.c[0], &sc->temp.data.c[0], &sc->regIDs[0].data.c[0]);
			PfSub(sc, &sc->w.data.c[1], &sc->temp.data.c[1], &sc->regIDs[0].data.c[1]);

			PfMul(sc, &sc->regIDs[0], &sc->w, &sc->mult, 0);
			PfConjugate(sc, &sc->mult, &sc->mult);

			PfMul(sc, &sc->temp, &sc->regIDs[1], &sc->mult, 0);

			appendRegistersToShared(sc, &sc->inoutID, &sc->temp);
			appendRegistersToShared(sc, &sc->sdataID, &sc->regIDs[0]);
		}
		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim_half.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
		else {
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim_half.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
	}
	if (sc->useDisableThreads) {
		PfIf_end(sc);
	}
	if (((sc->performDST == 2) && (sc->actualInverse == 0)) || ((sc->performDST == 3) && (sc->actualInverse == 1))) {
		appendBarrierVkFFT(sc);
		if (sc->stridedSharedLayout) {
			PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[1]);
		}
		else {
			PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[0]);
		}
		if (sc->useDisableThreads) {
			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
		}
		for (pfUINT i = 0; i < (pfUINT)used_registers.data.i; i++) {
			if (sc->stridedSharedLayout) {
				temp_int.data.i = (i)*sc->localSize[1].data.i;

				PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

				temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					//check that we only read fftDim * local batch data
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
					PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
				}
			}
			else {
				if (sc->localSize[1].data.i == 1) {
					temp_int.data.i = (i)*sc->localSize[0].data.i;

					PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
				}
				else {
					PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

					temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

					PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
					PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
				}
				temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					//check that we only read fftDim * local batch data
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim_half * &sc->localSize[0]);
					PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
				}
			}
			if (sc->stridedSharedLayout) {
				temp_int.data.i = fftDim.data.i - 1;
				PfSub(sc, &sc->combinedID, &temp_int, &sc->combinedID);

				PfMul(sc, &sc->sdataID, &sc->combinedID, &sc->sharedStride, 0);

				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);
			}
			else {
				PfMod(sc, &sc->sdataID, &sc->combinedID, &fftDim);
				temp_int.data.i = fftDim.data.i - 1;
				PfSub(sc, &sc->sdataID, &temp_int, &sc->sdataID);

				PfDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
				PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);

				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
			}
			appendSharedToRegisters(sc, &sc->regIDs[i], &sc->sdataID);
			if (sc->stridedSharedLayout) {
				temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					PfIf_end(sc);
				}
			}
			else {
				temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					PfIf_end(sc);
				}
			}
		}
		if (sc->useDisableThreads) {
			PfIf_end(sc);
		}
		appendBarrierVkFFT(sc);
		
		if (sc->useDisableThreads) {
			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
		}
		for (pfUINT i = 0; i < (pfUINT)used_registers.data.i; i++) {
			if (sc->stridedSharedLayout) {
				temp_int.data.i = (i)*sc->localSize[1].data.i;

				PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

				temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					//check that we only read fftDim * local batch data
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
					PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
				}
			}
			else {
				if (sc->localSize[1].data.i == 1) {
					temp_int.data.i = (i)*sc->localSize[0].data.i;

					PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
				}
				else {
					PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

					temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

					PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
					PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
				}
				temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					//check that we only read fftDim * local batch data
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim_half * &sc->localSize[0]);
					PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
				}
			}
			if (sc->stridedSharedLayout) {
				PfMul(sc, &sc->sdataID, &sc->combinedID, &sc->sharedStride, 0);

				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);
			}
			else {
				PfMod(sc, &sc->sdataID, &sc->combinedID, &fftDim);
				
				PfDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
				PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);

				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
			}
			appendRegistersToShared(sc, &sc->sdataID, &sc->regIDs[i]);
			if (sc->stridedSharedLayout) {
				temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					PfIf_end(sc);
				}
			}
			else {
				temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					PfIf_end(sc);
				}
			}
		}
		if (sc->useDisableThreads) {
			PfIf_end(sc);
		}
	}
	return;
}

static inline void appendDCTIV_even_read(VkFFTSpecializationConstantsLayout* sc, int type, int readWrite)  {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_int1 = VKFFT_ZERO_INIT;
	temp_int1.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 22;

	PfContainer used_registers = VKFFT_ZERO_INIT;
	used_registers.type = 31;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;
	PfContainer fftDim_half = VKFFT_ZERO_INIT;
	fftDim_half.type = 31;
	PfContainer localSize = VKFFT_ZERO_INIT;
	localSize.type = 31;

	PfContainer batching_localSize = VKFFT_ZERO_INIT;
	batching_localSize.type = 31;

	PfContainer* localInvocationID = VKFFT_ZERO_INIT;
	PfContainer* batchingInvocationID = VKFFT_ZERO_INIT;

	if (sc->stridedSharedLayout) {
		batching_localSize.data.i = sc->localSize[0].data.i;
		localSize.data.i = sc->localSize[1].data.i;
		localInvocationID = &sc->gl_LocalInvocationID_y;
		batchingInvocationID = &sc->gl_LocalInvocationID_x;
	}
	else {
		batching_localSize.data.i = sc->localSize[1].data.i;
		localSize.data.i = sc->localSize[0].data.i;
		localInvocationID = &sc->gl_LocalInvocationID_x;
		batchingInvocationID = &sc->gl_LocalInvocationID_y;
	}

	if (sc->zeropadBluestein[readWrite]) {
		if (readWrite) {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i;
		}
		else {
			if (sc->readToRegisters == 1) {
				appendSetSMToZero(sc);
				appendBarrierVkFFT(sc);
			}
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i;
		}
	}
	else
		fftDim.data.i = sc->fftDim.data.i;

	fftDim.data.i = 2 * fftDim.data.i;	

	if (sc->stridedSharedLayout) {
		PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[1]);
	}
	else {
		PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[0]);
	}
	if (sc->readToRegisters == 1) {
		if (sc->useDisableThreads) {
			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
		}
		for (pfUINT i = 0; i < (pfUINT)used_registers.data.i; i++) {
			if (sc->axis_id > 0) {
				temp_int.data.i = (i)*sc->localSize[1].data.i;

				PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

				temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					//check that we only read fftDim * local batch data
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
					PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
				}
			}
			else {
				if (sc->localSize[1].data.i == 1) {
					temp_int.data.i = (i)*sc->localSize[0].data.i;

					PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
				}
				else {
					PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

					temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

					PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
					PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
				}
				temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					//check that we only read fftDim * local batch data
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
					PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
				}
			}
			if (sc->axis_id > 0) {
				temp_int1.data.i = 2;
				PfDiv(sc, &sc->sdataID, &sc->combinedID, &temp_int1);

				PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);
			}
			else {
				if (sc->stridedSharedLayout) {
					temp_int.data.i = fftDim.data.i;
					PfMod(sc, &sc->sdataID, &sc->combinedID, &temp_int);

					temp_int1.data.i = 2;
					PfDiv(sc, &sc->sdataID, &sc->sdataID, &temp_int1);

					PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
					PfDiv(sc, &sc->tempInt, &sc->combinedID, &temp_int);
					PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
				}
				else {
					temp_int.data.i = fftDim.data.i;
					PfMod(sc, &sc->sdataID, &sc->combinedID, &temp_int);

					temp_int1.data.i = 2;
					PfDiv(sc, &sc->sdataID, &sc->sdataID, &temp_int1);

					PfDiv(sc, &sc->tempInt, &sc->combinedID, &temp_int);
					PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
					PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
				}
			}
			if (sc->axis_id > 0) {
				temp_int.data.i = 2;
				PfMod(sc, &sc->tempInt, &sc->combinedID, &temp_int);
			}
			else {
				PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
				temp_int.data.i = 2;
				PfMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			}
			if (sc->performDST)  {
				if (i < (pfUINT)used_registers.data.i / 2) {
					temp_int.data.i = 1;
					PfIf_eq_start(sc, &sc->tempInt, &temp_int);
					PfMovNeg(sc, &sc->regIDs[i].data.c[0], &sc->regIDs[i].data.c[0]);
					PfIf_end(sc);
				}
				else {
					temp_int.data.i = 1;
					PfIf_eq_start(sc, &sc->tempInt, &temp_int);
					PfMovNeg(sc, &sc->regIDs[i - used_registers.data.i / 2].data.c[1], &sc->regIDs[i - used_registers.data.i / 2].data.c[1]);
					PfIf_end(sc);
				}
			}
			temp_int.data.i = 0;
			PfIf_eq_start(sc, &sc->tempInt, &temp_int);
			if (i < (pfUINT)used_registers.data.i / 2) {
				appendRegistersToShared_x_x(sc, &sc->sdataID, &sc->regIDs[i]);
			}
			else {
				appendRegistersToShared_x_y(sc, &sc->sdataID, &sc->regIDs[i - used_registers.data.i / 2]);
			}
#if(!((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5)))
			PfIf_else(sc);
			if (i < (pfUINT)used_registers.data.i / 2) {
				appendRegistersToShared_y_x(sc, &sc->sdataID, &sc->regIDs[i]);
			}
			else {
				appendRegistersToShared_y_y(sc, &sc->sdataID, &sc->regIDs[i - used_registers.data.i / 2]);
			}
#endif
			PfIf_end(sc);
			if (sc->axis_id > 0) {
				temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					PfIf_end(sc);
				}
			}
			else {
				temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					PfIf_end(sc);
				}
			}
		}
		if (sc->useDisableThreads) {
			PfIf_end(sc);
		}
#if(((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5)))
		appendBarrierVkFFT(sc);
		if (sc->useDisableThreads) {
			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
		}
		for (pfUINT i = 0; i < (pfUINT)used_registers.data.i; i++) {
			if (sc->axis_id > 0) {
				temp_int.data.i = (i)*sc->localSize[1].data.i;

				PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

				temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					//check that we only read fftDim * local batch data
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
					PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
				}
			}
			else {
				if (sc->localSize[1].data.i == 1) {
					temp_int.data.i = (i)*sc->localSize[0].data.i;

					PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
				}
				else {
					PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

					temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

					PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
					PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
				}
				temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					//check that we only read fftDim * local batch data
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
					PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
			}
			if (sc->axis_id > 0) {
				temp_int1.data.i = 2;
				PfDiv(sc, &sc->sdataID, &sc->combinedID, &temp_int1);

				PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);
			}
			else {
				if (sc->stridedSharedLayout) {
					temp_int.data.i = fftDim.data.i;
					PfMod(sc, &sc->sdataID, &sc->combinedID, &temp_int);

					temp_int1.data.i = 2;
					PfDiv(sc, &sc->sdataID, &sc->sdataID, &temp_int1);

					PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
					PfDiv(sc, &sc->tempInt, &sc->combinedID, &temp_int);
					PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
				}
				else {
					temp_int.data.i = fftDim.data.i;
					PfMod(sc, &sc->sdataID, &sc->combinedID, &temp_int);

					temp_int1.data.i = 2;
					PfDiv(sc, &sc->sdataID, &sc->sdataID, &temp_int1);

					PfDiv(sc, &sc->tempInt, &sc->combinedID, &temp_int);
					PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
					PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
				}
			}
			if (sc->axis_id > 0) {
				temp_int.data.i = 2;
				PfMod(sc, &sc->tempInt, &sc->combinedID, &temp_int);
			}
			else {
				PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
				temp_int.data.i = 2;
				PfMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			}
			temp_int.data.i = 1;
			PfIf_eq_start(sc, &sc->tempInt, &temp_int);
			if ((pfINT)i < used_registers.data.i / 2) {
				appendRegistersToShared_y_x(sc, &sc->sdataID, &sc->regIDs[i]);
			}
			else {
				appendRegistersToShared_y_y(sc, &sc->sdataID, &sc->regIDs[i - used_registers.data.i / 2]);
			}
			PfIf_end(sc);

			if (sc->axis_id > 0) {
				temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					PfIf_end(sc);
				}
			}
			else {
				temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					PfIf_end(sc);
				}
			}
		}
		if (sc->useDisableThreads) {
			PfIf_end(sc);
		}
#endif
	}
	appendBarrierVkFFT(sc);
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	fftDim.data.i = fftDim.data.i / 2;

	if (sc->stridedSharedLayout) {
		PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[1]);
	}
	else {
		PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[0]);
	}
	for (pfUINT i = 0; i < (pfUINT)used_registers.data.i; i++) {
		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i)*sc->localSize[1].data.i;

			PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		else {
			if (sc->localSize[1].data.i == 1) {
				temp_int.data.i = (i)*sc->localSize[0].data.i;

				PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
			}
			else {
				PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

				temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

				PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
				PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
			}
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}

		if (sc->stridedSharedLayout) {
			PfMul(sc, &sc->sdataID, &sc->combinedID, &sc->sharedStride, 0);
			PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);

			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->combinedID, &temp_int);
		}
		else {
			PfMod(sc, &sc->sdataID, &sc->combinedID, &fftDim);
			PfDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
			PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);

			PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->tempInt, &temp_int);
		}

		if (sc->stridedSharedLayout) {
			PfSub(sc, &sc->tempInt, &sc->sdataID, &sc->sharedStride);
			appendSharedToRegisters_y_y(sc, &sc->w, &sc->tempInt);
		}
		else {
			temp_int.data.i = 1;
			PfSub(sc, &sc->tempInt, &sc->sdataID, &temp_int);
			appendSharedToRegisters_y_y(sc, &sc->w, &sc->tempInt);
		}

		appendSharedToRegisters_x_x(sc, &sc->w, &sc->sdataID);
		
		PfMov(sc, &sc->regIDs[i].data.c[0], &sc->w.data.c[1]);
		PfMovNeg(sc, &sc->regIDs[i].data.c[1], &sc->w.data.c[0]);
		PfAdd(sc, &sc->regIDs[i], &sc->regIDs[i], &sc->w);
		
		PfIf_else(sc);

		appendSharedToRegisters_x_x(sc, &sc->regIDs[i], &sc->sdataID);
		if (sc->stridedSharedLayout) {
			temp_int.data.i = (fftDim.data.i - 1) * sc->sharedStride.data.i;
			PfAdd(sc, &sc->sdataID, &sc->sdataID, &temp_int);
		}
		else {
			temp_int.data.i = (fftDim.data.i - 1);
			PfAdd(sc, &sc->sdataID, &sc->sdataID, &temp_int);
		}
		appendSharedToRegisters_y_y(sc, &sc->regIDs[i], &sc->sdataID);
		temp_double.data.d = pfFPinit("2.0");
		PfMul(sc, &sc->regIDs[i], &sc->regIDs[i], &temp_double, 0);
		
		PfIf_end(sc);
		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_end(sc);
			}
		}
		else {
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
	}
	if (sc->useDisableThreads) {
		PfIf_end(sc);
	}
	appendBarrierVkFFT(sc);
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	for (pfUINT i = 0; i < (pfUINT)used_registers.data.i; i++) {
		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i)*sc->localSize[1].data.i;

			PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		else {
			if (sc->localSize[1].data.i == 1) {
				temp_int.data.i = (i)*sc->localSize[0].data.i;

				PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
			}
			else {
				PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

				temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

				PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
				PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
			}

			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		if (sc->stridedSharedLayout) {
			PfMul(sc, &sc->sdataID, &sc->combinedID, &sc->sharedStride, 0);
			PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);

			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->combinedID, &temp_int);
		}
		else {
			PfMod(sc, &sc->sdataID, &sc->combinedID, &fftDim);
			PfDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
			PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);

			PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->tempInt, &temp_int);
		}

		appendRegistersToShared_x_x(sc, &sc->sdataID, &sc->regIDs[i]);

#if(!((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5)))//OpenCL, Level Zero and Metal are  not handling barrier with thread-conditional writes to local memory - so this is a work-around
		if (sc->stridedSharedLayout) {
			PfSub(sc, &sc->sdataID, &fftDim, &sc->combinedID);

			PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
			PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);
		}
		else {
			PfMod(sc, &sc->sdataID, &sc->combinedID, &fftDim);
			PfSub(sc, &sc->sdataID, &fftDim, &sc->sdataID);

			PfDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
			PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
		}

		appendRegistersToShared_y_y(sc, &sc->sdataID, &sc->regIDs[i]);
#endif
		
		PfIf_else(sc);

		appendRegistersToShared(sc, &sc->sdataID, &sc->regIDs[i]);

		PfIf_end(sc);

		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_end(sc);
			}
		}
		else {
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
	}
	if (sc->useDisableThreads) {
		PfIf_end(sc);
	}
#if(((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5)))//OpenCL, Level Zero and Metal are  not handling barrier with thread-conditional writes to local memory - so this is a work-around

	appendBarrierVkFFT(sc);
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	for (pfUINT i = 0; i < (pfUINT)used_registers.data.i; i++) {
		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i)*sc->localSize[1].data.i;

			PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		else {
			if (sc->localSize[1].data.i == 1) {
				temp_int.data.i = (i)*sc->localSize[0].data.i;

				PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
			}
			else {
				PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

				temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

				PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
				PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
			}

			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}

		if (sc->stridedSharedLayout) {
			PfSub(sc, &sc->sdataID, &fftDim, &sc->combinedID);

			PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
			PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);

			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->combinedID, &temp_int);
		}
		else {
			PfMod(sc, &sc->sdataID, &sc->combinedID, &fftDim);
			PfSub(sc, &sc->sdataID, &fftDim, &sc->sdataID);

			PfDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
			PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
			
			PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->tempInt, &temp_int);
		}

		appendRegistersToShared_y_y(sc, &sc->sdataID, &sc->regIDs[i]);

		PfIf_end(sc);

		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
		else {
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
	}
	if (sc->useDisableThreads) {
		PfIf_end(sc);
	}
#endif

	appendDCTII_write_III_read(sc, type, 0);

	sc->readToRegisters = 0;

	return;
}
static inline void appendDCTIV_even_write(VkFFTSpecializationConstantsLayout* sc, int type, int readWrite) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_int1 = VKFFT_ZERO_INIT;
	temp_int1.type = 31;
	PfContainer temp_int2 = VKFFT_ZERO_INIT;
	temp_int2.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 22;

	PfContainer used_registers = VKFFT_ZERO_INIT;
	used_registers.type = 31;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;

	PfContainer localSize = VKFFT_ZERO_INIT;
	localSize.type = 31;

	PfContainer batching_localSize = VKFFT_ZERO_INIT;
	batching_localSize.type = 31;

	PfContainer* localInvocationID = VKFFT_ZERO_INIT;
	PfContainer* batchingInvocationID = VKFFT_ZERO_INIT;

	if (sc->stridedSharedLayout) {
		batching_localSize.data.i = sc->localSize[0].data.i;
		localSize.data.i = sc->localSize[1].data.i;
		localInvocationID = &sc->gl_LocalInvocationID_y;
		batchingInvocationID = &sc->gl_LocalInvocationID_x;
	}
	else {
		batching_localSize.data.i = sc->localSize[1].data.i;
		localSize.data.i = sc->localSize[0].data.i;
		localInvocationID = &sc->gl_LocalInvocationID_x;
		batchingInvocationID = &sc->gl_LocalInvocationID_y;
	}

	if (sc->zeropadBluestein[readWrite]) {
		fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i;
	}
	else {
		fftDim.data.i = sc->fftDim.data.i;
	}

	if (sc->stridedSharedLayout) {
		PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[1]);
	}
	else {
		PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[0]);
	}

	appendDCTII_read_III_write(sc, type, 1);

	appendBarrierVkFFT(sc);
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	for (pfUINT i = 0; i < (pfUINT)used_registers.data.i; i++) {
		if (sc->axis_id > 0) {
			temp_int.data.i = (i)*sc->localSize[1].data.i;

			PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		else {
			if (sc->localSize[1].data.i == 1) {
				temp_int.data.i = (i)*sc->localSize[0].data.i;

				PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
			}
			else {
				PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

				temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

				PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
				PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
			}

			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		if (sc->axis_id > 0) {
			temp_int.data.i = 2;
			PfMod(sc, &sc->sdataID, &sc->combinedID, &temp_int);
		}
		else {
			PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			temp_int.data.i = 2;
			PfMod(sc, &sc->sdataID, &sc->tempInt, &temp_int);
		}
		PfMul(sc, &sc->blockInvocationID, &sc->sdataID, &temp_int, 0);
		temp_double.data.d = pfFPinit("1.0");
		if (sc->precision == 3) {
			PfSetToZero(sc, &sc->tempFloat);
			PfMov(sc, &sc->tempFloat.data.dd[0], &sc->blockInvocationID);
			PfSub(sc, &sc->tempFloat, &temp_double, &sc->tempFloat);
		} else {
			PfSub(sc, &sc->tempFloat, &temp_double, &sc->blockInvocationID);
		}
		PfMul(sc, &sc->regIDs[i].data.c[1], &sc->regIDs[i].data.c[1], &sc->tempFloat, 0);

		if (sc->LUT) {
			if (sc->axis_id > 0) {
				PfAdd(sc, &sc->tempInt, &sc->combinedID, &sc->startDCT4LUT);
			}
			else {
				PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->startDCT4LUT);
			}
			appendGlobalToRegisters(sc, &sc->mult, &sc->LUTStruct, &sc->tempInt);
		}
		else {
			temp_int.data.i = 2;
			if (sc->axis_id > 0) {
				PfMul(sc, &sc->tempInt, &sc->combinedID, &temp_int, 0);
			}
			else {
				PfMul(sc, &sc->tempInt, &sc->tempInt, &temp_int, 0);
			}
			PfInc(sc, &sc->tempInt);
			if (readWrite)
				temp_double.data.d = -sc->double_PI / pfFPinit("8.0") / fftDim.data.i;
			else
				temp_double.data.d = sc->double_PI / pfFPinit("8.0") / fftDim.data.i;
			PfMul(sc, &sc->tempFloat, &sc->tempInt, &temp_double, 0);

			PfSinCos(sc, &sc->mult, &sc->tempFloat);
		}

		PfMul(sc, &sc->regIDs[i], &sc->regIDs[i], &sc->mult, &sc->temp);
		PfConjugate(sc, &sc->regIDs[i], &sc->regIDs[i]);

		if (sc->axis_id > 0) {
			PfMul(sc, &sc->tempInt, &sc->combinedID, &sc->sharedStride, 0);

			temp_int.data.i = (fftDim.data.i - 1) * sc->sharedStride.data.i;
			PfAdd(sc, &sc->sdataID, &sc->gl_LocalInvocationID_x, &temp_int);
			PfSub(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
		}
		else {
			PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			if (sc->stridedSharedLayout) {
				temp_int.data.i = (fftDim.data.i-1);
				PfSub(sc, &sc->sdataID, &temp_int, &sc->tempInt);
				PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
				
				PfDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
			}
			else {
				PfDiv(sc, &sc->sdataID, &sc->combinedID, &fftDim);
				PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);

				temp_int.data.i = (fftDim.data.i-1);
				PfAdd(sc, &sc->sdataID, &sc->sdataID, &temp_int);
				PfSub(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
			}
		}
		if (sc->performDST) 
			appendRegistersToShared(sc, &sc->sdataID, &sc->regIDs[i]);
		else
			appendRegistersToShared_y_y(sc, &sc->sdataID, &sc->regIDs[i]);
		if (sc->axis_id > 0) {
			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
		else {
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
	}
	if (sc->useDisableThreads) {
		PfIf_end(sc);
	}
	appendBarrierVkFFT(sc);
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	for (pfUINT i = 0; i < (pfUINT)used_registers.data.i; i++) {
		if (sc->axis_id > 0) {
			temp_int.data.i = (i)*sc->localSize[1].data.i;

			PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		else {
			if (sc->localSize[1].data.i == 1) {
				temp_int.data.i = (i)*sc->localSize[0].data.i;

				PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
			}
			else {
				PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

				temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

				PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
				PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
			}

			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}

		if (sc->axis_id > 0) {
			if (sc->performDST) {
				temp_int.data.i = fftDim.data.i - 1;
				PfSub(sc, &sc->tempInt, &temp_int, &sc->combinedID);
				PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
			}
			else
				PfMul(sc, &sc->tempInt, &sc->combinedID, &sc->sharedStride, 0);

			PfAdd(sc, &sc->sdataID, &sc->gl_LocalInvocationID_x, &sc->tempInt);
		}
		else {
			PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			if (sc->performDST) {
				temp_int.data.i = fftDim.data.i - 1;
				PfSub(sc, &sc->tempInt, &temp_int, &sc->tempInt);
			}
			if (sc->stridedSharedLayout) {
				PfMul(sc, &sc->sdataID, &sc->tempInt, &sc->sharedStride, 0);

				PfDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
			}
			else {
				PfDiv(sc, &sc->sdataID, &sc->combinedID, &fftDim);
				PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);

				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
			}
		}
		if (sc->performDST) {
			appendSharedToRegisters_x_y(sc, &sc->regIDs[i], &sc->sdataID);
			if (sc->axis_id > 0) {
				PfMul(sc, &sc->tempInt, &sc->combinedID, &sc->sharedStride, 0);

				PfAdd(sc, &sc->sdataID, &sc->gl_LocalInvocationID_x, &sc->tempInt);
			}
			else {
				PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
				if (sc->stridedSharedLayout) {
					PfMul(sc, &sc->sdataID, &sc->tempInt, &sc->sharedStride, 0);

					PfDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
					PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
				}
				else {
					PfDiv(sc, &sc->sdataID, &sc->combinedID, &fftDim);
					PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);

					PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
				}
			}
			appendSharedToRegisters_y_x(sc, &sc->regIDs[i], &sc->sdataID);
		}
		else
			appendSharedToRegisters_y_y(sc, &sc->regIDs[i], &sc->sdataID);
		if (sc->axis_id > 0) {
			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
		else {
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
	}
	if (sc->useDisableThreads) {
		PfIf_end(sc);
	}
	return;
}

static inline void appendDCTIV_odd_read(VkFFTSpecializationConstantsLayout* sc, int type, int readWrite) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_int1 = VKFFT_ZERO_INIT;
	temp_int1.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 22;

	PfContainer used_registers = VKFFT_ZERO_INIT;
	used_registers.type = 31;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;

	PfContainer localSize = VKFFT_ZERO_INIT;
	localSize.type = 31;

	PfContainer batching_localSize = VKFFT_ZERO_INIT;
	batching_localSize.type = 31;

	PfContainer* localInvocationID = VKFFT_ZERO_INIT;
	PfContainer* batchingInvocationID = VKFFT_ZERO_INIT;

	if (sc->stridedSharedLayout) {
		batching_localSize.data.i = sc->localSize[0].data.i;
		localSize.data.i = sc->localSize[1].data.i;
		localInvocationID = &sc->gl_LocalInvocationID_y;
		batchingInvocationID = &sc->gl_LocalInvocationID_x;
	}
	else {
		batching_localSize.data.i = sc->localSize[1].data.i;
		localSize.data.i = sc->localSize[0].data.i;
		localInvocationID = &sc->gl_LocalInvocationID_x;
		batchingInvocationID = &sc->gl_LocalInvocationID_y;
	}

	if (sc->zeropadBluestein[readWrite]) {
		if (readWrite) {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i;
		}
		else {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i;
		}
	}
	else {
		fftDim.data.i = sc->fftDim.data.i;
	}

	if (sc->stridedSharedLayout) {
		PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[1]);
	}
	else {
		PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[0]);
	}

	appendBarrierVkFFT(sc);
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	for (pfUINT i = 0; i < (pfUINT)used_registers.data.i; i++) {
		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i)*sc->localSize[1].data.i;

			PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		else {
			if (sc->localSize[1].data.i == 1) {
				temp_int.data.i = (i)*sc->localSize[0].data.i;

				PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
			}
			else {
				PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

				temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

				PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
				PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
			}

			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		if (sc->stridedSharedLayout) {
			temp_int.data.i = 4;
			PfMul(sc, &sc->inoutID, &sc->combinedID, &temp_int,0);
		}
		else {
			PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			temp_int.data.i = 4;
			PfMul(sc, &sc->inoutID, &sc->tempInt, &temp_int,0);
		}
		temp_int.data.i = fftDim.data.i / 2;
		PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int);
		
		PfIf_lt_start(sc, &sc->inoutID, &fftDim);
		PfMov(sc, &sc->sdataID, &sc->inoutID);
		PfIf_end(sc);
		
		temp_int.data.i = fftDim.data.i * 2;
		PfIf_lt_start(sc, &sc->inoutID, &temp_int);
		PfIf_ge_start(sc, &sc->inoutID, &fftDim);
		temp_int.data.i = fftDim.data.i * 2 - 1;
		PfSub(sc, &sc->sdataID, &temp_int, &sc->inoutID);
		PfIf_end(sc);
		PfIf_end(sc);

		temp_int.data.i = fftDim.data.i * 3;
		PfIf_lt_start(sc, &sc->inoutID, &temp_int);
		temp_int.data.i = fftDim.data.i * 2;
		PfIf_ge_start(sc, &sc->inoutID, &temp_int);
		temp_int.data.i = fftDim.data.i * 2;
		PfSub(sc, &sc->sdataID, &sc->inoutID, &temp_int);
		PfIf_end(sc);
		PfIf_end(sc);

		temp_int.data.i = fftDim.data.i * 4;
		PfIf_lt_start(sc, &sc->inoutID, &temp_int);
		temp_int.data.i = fftDim.data.i * 3;
		PfIf_ge_start(sc, &sc->inoutID, &temp_int);
		temp_int.data.i = fftDim.data.i * 4 - 1;
		PfSub(sc, &sc->sdataID, &temp_int, &sc->inoutID);
		PfIf_end(sc);
		PfIf_end(sc);

		temp_int.data.i = fftDim.data.i * 4;
		PfIf_ge_start(sc, &sc->inoutID, &temp_int);
		temp_int.data.i = fftDim.data.i * 4;
		PfSub(sc, &sc->sdataID, &sc->inoutID, &temp_int);
		PfIf_end(sc);

		if (sc->performDST) {
			temp_int.data.i = 2;
			PfMod(sc, &sc->blockInvocationID, &sc->sdataID, &temp_int);
		}

		if (sc->stridedSharedLayout) {
			PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
			PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);
		}
		else {
			PfDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
			PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
			PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
		}
		appendSharedToRegisters(sc, &sc->regIDs[i], &sc->sdataID);

		if (sc->performDST)  {
			temp_int.data.i = 1;
			PfIf_eq_start(sc, &sc->blockInvocationID, &temp_int);
			PfMovNeg(sc, &sc->regIDs[i], &sc->regIDs[i]);
			PfIf_end(sc);
		}

		temp_int.data.i = fftDim.data.i * 2;
		PfIf_lt_start(sc, &sc->inoutID, &temp_int);
		PfIf_ge_start(sc, &sc->inoutID, &fftDim);
		PfMovNeg(sc, &sc->regIDs[i].data.c[0], &sc->regIDs[i].data.c[0]);
		PfMovNeg(sc, &sc->regIDs[i].data.c[1], &sc->regIDs[i].data.c[1]);
		PfIf_end(sc);
		PfIf_end(sc);

		temp_int.data.i = fftDim.data.i * 3;
		PfIf_lt_start(sc, &sc->inoutID, &temp_int);
		temp_int.data.i = fftDim.data.i * 2;
		PfIf_ge_start(sc, &sc->inoutID, &temp_int);
		PfMovNeg(sc, &sc->regIDs[i].data.c[0], &sc->regIDs[i].data.c[0]);
		PfMovNeg(sc, &sc->regIDs[i].data.c[1], &sc->regIDs[i].data.c[1]);
		PfIf_end(sc);
		PfIf_end(sc);

		if (sc->stridedSharedLayout) {
			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
		else {
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
	}
	if (sc->useDisableThreads) {
		PfIf_end(sc);
	}
	pfINT registers_first_stage = (sc->stageRadix[0] < sc->fixMinRaderPrimeMult) ? sc->registers_per_thread_per_radix[sc->stageRadix[0]] : 1;
	if ((sc->rader_generator[0] > 0) || ((sc->fftDim.data.i / registers_first_stage) != localSize.data.i))
		sc->readToRegisters = 0;
	else
		sc->readToRegisters = 0; // can be switched to 1 if the indexing in previous step is aligned to 1 stage of fft (here it is combined)

	if (!sc->readToRegisters) {

		appendBarrierVkFFT(sc);
		if (sc->useDisableThreads) {
			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
		}
		for (pfUINT i = 0; i < (pfUINT)used_registers.data.i; i++) {
			if (sc->stridedSharedLayout) {
				temp_int.data.i = (i)*sc->localSize[1].data.i;

				PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

				temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					//check that we only read fftDim * local batch data
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
					PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
				}
			}
			else {
				if (sc->localSize[1].data.i == 1) {
					temp_int.data.i = (i)*sc->localSize[0].data.i;

					PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
				}
				else {
					PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

					temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

					PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
					PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
				}

				temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					//check that we only read fftDim * local batch data
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
					PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
				}
			}
			
			if (sc->stridedSharedLayout) {
				PfMul(sc, &sc->tempInt, &sc->combinedID, &sc->sharedStride, 0);
				PfAdd(sc, &sc->sdataID, &sc->gl_LocalInvocationID_x, &sc->tempInt);
			}
			else {
				PfDiv(sc, &sc->sdataID, &sc->combinedID, &fftDim);
				PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);

				PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
			}

			appendRegistersToShared(sc, &sc->sdataID, &sc->regIDs[i]);

			if (sc->stridedSharedLayout) {
				temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					PfIf_end(sc);
				}
			}
			else {
				temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					PfIf_end(sc);
				}
			}
		}
		if (sc->useDisableThreads) {
			PfIf_end(sc);
		}
	}
	return;
}

static inline void appendDCTIV_odd_write(VkFFTSpecializationConstantsLayout* sc, int type, int readWrite) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	PfContainer temp_int1 = VKFFT_ZERO_INIT;
	temp_int1.type = 31;
	PfContainer temp_double = VKFFT_ZERO_INIT;
	temp_double.type = 22;

	PfContainer used_registers = VKFFT_ZERO_INIT;
	used_registers.type = 31;

	PfContainer fftDim = VKFFT_ZERO_INIT;
	fftDim.type = 31;

	PfContainer localSize = VKFFT_ZERO_INIT;
	localSize.type = 31;

	PfContainer batching_localSize = VKFFT_ZERO_INIT;
	batching_localSize.type = 31;

	PfContainer* localInvocationID = VKFFT_ZERO_INIT;
	PfContainer* batchingInvocationID = VKFFT_ZERO_INIT;

	if (sc->stridedSharedLayout) {
		batching_localSize.data.i = sc->localSize[0].data.i;
		localSize.data.i = sc->localSize[1].data.i;
		localInvocationID = &sc->gl_LocalInvocationID_y;
		batchingInvocationID = &sc->gl_LocalInvocationID_x;
	}
	else {
		batching_localSize.data.i = sc->localSize[1].data.i;
		localSize.data.i = sc->localSize[0].data.i;
		localInvocationID = &sc->gl_LocalInvocationID_x;
		batchingInvocationID = &sc->gl_LocalInvocationID_y;
	}

	if (sc->zeropadBluestein[readWrite]) {
		if (readWrite) {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i;
		}
		else {
			fftDim.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i;
		}
	}
	else {
		fftDim.data.i = sc->fftDim.data.i;
	}

	if (sc->stridedSharedLayout) {
		PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[1]);
	}
	else {
		PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[0]);
	}

	appendBarrierVkFFT(sc);
	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	for (pfUINT i = 0; i < (pfUINT)used_registers.data.i; i++) {
		if (sc->axis_id > 0) {
			temp_int.data.i = (i)*sc->localSize[1].data.i;

			PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		else {
			if (sc->localSize[1].data.i == 1) {
				temp_int.data.i = (i)*sc->localSize[0].data.i;

				PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
			}
			else {
				PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

				temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

				PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
				PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
			}

			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}
		}
		if (sc->axis_id > 0) {
			PfMov(sc, &sc->sdataID, &sc->combinedID);
		}
		else {
			PfMod(sc, &sc->sdataID, &sc->combinedID, &fftDim);
		}

		temp_int.data.i = fftDim.data.i / 4;
		PfIf_lt_start(sc, &sc->sdataID, &temp_int);

		temp_int.data.i = 2;
		PfMul(sc, &sc->inoutID, &sc->sdataID, &temp_int, 0);
		PfInc(sc, &sc->inoutID);
		if (sc->mergeSequencesR2C) {
			PfSub(sc, &sc->tempInt, &fftDim, &sc->inoutID);
			PfIf_eq_start(sc, &sc->tempInt, &fftDim);
			PfSetToZero(sc, &sc->tempInt);
			PfIf_end(sc);
		}

		PfIf_eq_start(sc, &sc->inoutID, &fftDim);
		PfSetToZero(sc, &sc->inoutID);
		PfIf_end(sc);

		PfIf_end(sc);

		temp_int.data.i = fftDim.data.i / 2;
		PfIf_lt_start(sc, &sc->sdataID, &temp_int);
		temp_int.data.i = fftDim.data.i / 4;
		PfIf_ge_start(sc, &sc->sdataID, &temp_int);

		temp_int.data.i = 2;
		PfMul(sc, &sc->inoutID, &sc->sdataID, &temp_int, 0);
		if (sc->mergeSequencesR2C) {
			temp_int.data.i = fftDim.data.i - 2 * (fftDim.data.i / 2);
			PfAdd(sc, &sc->tempInt, &temp_int, &sc->inoutID);

			PfIf_eq_start(sc, &sc->tempInt, &fftDim);
			PfSetToZero(sc, &sc->tempInt);
			PfIf_end(sc);
		}
		temp_int.data.i = 2 * (fftDim.data.i / 2);
		PfSub(sc, &sc->inoutID, &temp_int, &sc->inoutID);

		PfIf_eq_start(sc, &sc->inoutID, &fftDim);
		PfSetToZero(sc, &sc->inoutID);
		PfIf_end(sc);

		PfIf_end(sc);
		PfIf_end(sc);

		temp_int.data.i = 3 * fftDim.data.i / 4;
		PfIf_lt_start(sc, &sc->sdataID, &temp_int);
		temp_int.data.i = fftDim.data.i / 2;
		PfIf_ge_start(sc, &sc->sdataID, &temp_int);

		temp_int.data.i = 2;
		PfMul(sc, &sc->inoutID, &sc->sdataID, &temp_int, 0);
		if (sc->mergeSequencesR2C) {
			temp_int.data.i = fftDim.data.i + 2 * (fftDim.data.i / 2);
			PfSub(sc, &sc->tempInt, &temp_int, &sc->inoutID);
			PfIf_eq_start(sc, &sc->tempInt, &fftDim);
			PfSetToZero(sc, &sc->tempInt);
			PfIf_end(sc);
		}
		temp_int.data.i = 2 * (fftDim.data.i / 2);
		PfSub(sc, &sc->inoutID, &sc->inoutID, &temp_int);

		PfIf_eq_start(sc, &sc->inoutID, &fftDim);
		PfSetToZero(sc, &sc->inoutID);
		PfIf_end(sc);

		PfIf_end(sc);
		PfIf_end(sc);

		temp_int.data.i = 3 * fftDim.data.i / 4;
		PfIf_ge_start(sc, &sc->sdataID, &temp_int);

		temp_int.data.i = 2;
		PfMul(sc, &sc->inoutID, &sc->sdataID, &temp_int, 0);
		if (sc->mergeSequencesR2C) {
			temp_int.data.i = fftDim.data.i - 1;
			PfSub(sc, &sc->tempInt, &sc->inoutID, &temp_int);
			PfIf_eq_start(sc, &sc->tempInt, &fftDim);
			PfSetToZero(sc, &sc->tempInt);
			PfIf_end(sc);
		}
		temp_int.data.i = 2 * fftDim.data.i - 1;
		PfSub(sc, &sc->inoutID, &temp_int, &sc->inoutID);

		PfIf_eq_start(sc, &sc->inoutID, &fftDim);
		PfSetToZero(sc, &sc->inoutID);
		PfIf_end(sc);

		PfIf_end(sc);

		if (sc->axis_id > 0) {
			PfMul(sc, &sc->inoutID, &sc->inoutID, &sc->sharedStride, 0);
			PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->gl_LocalInvocationID_x);
			if (sc->mergeSequencesR2C) {
				PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
				PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->gl_LocalInvocationID_x);
			}
		}
		else {
			PfDiv(sc, &sc->blockInvocationID, &sc->combinedID, &fftDim);

			if (sc->stridedSharedLayout) {
				PfMul(sc, &sc->inoutID, &sc->inoutID, &sc->sharedStride, 0);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->blockInvocationID);
				if (sc->mergeSequencesR2C) {
					PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
					PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->blockInvocationID);
				}
			}
			else {
				PfMul(sc, &sc->blockInvocationID, &sc->blockInvocationID, &sc->sharedStride, 0);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->blockInvocationID);
				if (sc->mergeSequencesR2C) {
					PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->blockInvocationID);
				}
			}
		}
		appendSharedToRegisters(sc, &sc->temp, &sc->inoutID);
		if (sc->mergeSequencesR2C) {
			appendSharedToRegisters(sc, &sc->w, &sc->tempInt);
		}

		if (sc->mergeSequencesR2C) {
			PfAdd(sc, &sc->regIDs[i].data.c[0], &sc->temp.data.c[0], &sc->w.data.c[0]);
			PfSub(sc, &sc->regIDs[i].data.c[1], &sc->temp.data.c[1], &sc->w.data.c[1]);

			PfAdd(sc, &sc->w.data.c[1], &sc->temp.data.c[1], &sc->w.data.c[1]);
			PfSub(sc, &sc->w.data.c[0], &sc->w.data.c[0], &sc->temp.data.c[0]);
			PfMov(sc, &sc->temp.data.c[0], &sc->w.data.c[1]);
			PfMov(sc, &sc->temp.data.c[1], &sc->w.data.c[0]);
		}
		
		temp_int.data.i = fftDim.data.i / 4;
		PfIf_lt_start(sc, &sc->sdataID, &temp_int);

		temp_int.data.i = 1;
		PfAdd(sc, &sc->tempInt, &sc->sdataID, &temp_int);
		temp_int.data.i = 2;
		PfDiv(sc, &sc->tempInt, &sc->tempInt, &temp_int);
		PfMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->tempInt, &temp_int);
		if (sc->mergeSequencesR2C) {
			PfMovNeg(sc, &sc->w.data.c[0], &sc->regIDs[i].data.c[0]);
			PfMovNeg(sc, &sc->w.data.c[1], &sc->temp.data.c[0]);
		}
		else {
			PfMovNeg(sc, &sc->w.data.c[0], &sc->temp.data.c[0]);
		}
		PfIf_else(sc);
		if (sc->mergeSequencesR2C) {
			PfMov(sc, &sc->w.data.c[0], &sc->regIDs[i].data.c[0]);
			PfMov(sc, &sc->w.data.c[1], &sc->temp.data.c[0]);
		}
		else {
			PfMov(sc, &sc->w.data.c[0], &sc->temp.data.c[0]);
		}
		PfIf_end(sc);

		temp_int.data.i = 2;
		PfDiv(sc, &sc->tempInt, &sc->sdataID, &temp_int);
		PfMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->tempInt, &temp_int);
		if (sc->mergeSequencesR2C) {
			PfAdd(sc, &sc->w.data.c[0], &sc->w.data.c[0], &sc->regIDs[i].data.c[1]);
			PfAdd(sc, &sc->w.data.c[1], &sc->w.data.c[1], &sc->temp.data.c[1]);
		}
		else {
			PfAdd(sc, &sc->w.data.c[0], &sc->w.data.c[0], &sc->temp.data.c[1]);
		}
		PfIf_else(sc);
		if (sc->mergeSequencesR2C) {
			PfSub(sc, &sc->w.data.c[0], &sc->w.data.c[0], &sc->regIDs[i].data.c[1]);
			PfSub(sc, &sc->w.data.c[1], &sc->w.data.c[1], &sc->temp.data.c[1]);
		}
		else {
			PfSub(sc, &sc->w.data.c[0], &sc->w.data.c[0], &sc->temp.data.c[1]);
		}
		PfIf_end(sc);

		PfIf_end(sc);


		temp_int.data.i = fftDim.data.i / 2;
		PfIf_lt_start(sc, &sc->sdataID, &temp_int);
		temp_int.data.i = fftDim.data.i / 4;
		PfIf_ge_start(sc, &sc->sdataID, &temp_int);

		temp_int.data.i = 1;
		PfAdd(sc, &sc->tempInt, &sc->sdataID, &temp_int);
		temp_int.data.i = 2;
		PfDiv(sc, &sc->tempInt, &sc->tempInt, &temp_int);
		PfMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->tempInt, &temp_int);
		if (sc->mergeSequencesR2C) {
			PfMovNeg(sc, &sc->w.data.c[0], &sc->regIDs[i].data.c[0]);
			PfMovNeg(sc, &sc->w.data.c[1], &sc->temp.data.c[0]);
		}
		else {
			PfMovNeg(sc, &sc->w.data.c[0], &sc->temp.data.c[0]);
		}
		PfIf_else(sc);
		if (sc->mergeSequencesR2C) {
			PfMov(sc, &sc->w.data.c[0], &sc->regIDs[i].data.c[0]);
			PfMov(sc, &sc->w.data.c[1], &sc->temp.data.c[0]);
		}
		else {
			PfMov(sc, &sc->w.data.c[0], &sc->temp.data.c[0]);
		}
		PfIf_end(sc);

		temp_int.data.i = 2;
		PfDiv(sc, &sc->tempInt, &sc->sdataID, &temp_int);
		PfMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->tempInt, &temp_int);
		if (sc->mergeSequencesR2C) {
			PfSub(sc, &sc->w.data.c[0], &sc->w.data.c[0], &sc->regIDs[i].data.c[1]);
			PfSub(sc, &sc->w.data.c[1], &sc->w.data.c[1], &sc->temp.data.c[1]);
		}
		else {
			PfSub(sc, &sc->w.data.c[0], &sc->w.data.c[0], &sc->temp.data.c[1]);
		}
		PfIf_else(sc);
		if (sc->mergeSequencesR2C) {
			PfAdd(sc, &sc->w.data.c[0], &sc->w.data.c[0], &sc->regIDs[i].data.c[1]);
			PfAdd(sc, &sc->w.data.c[1], &sc->w.data.c[1], &sc->temp.data.c[1]);
		}
		else {
			PfAdd(sc, &sc->w.data.c[0], &sc->w.data.c[0], &sc->temp.data.c[1]);
		}
		PfIf_end(sc);

		PfIf_end(sc);
		PfIf_end(sc);


		temp_int.data.i = 3 * fftDim.data.i / 4;
		PfIf_lt_start(sc, &sc->sdataID, &temp_int);
		temp_int.data.i = fftDim.data.i / 2;
		PfIf_ge_start(sc, &sc->sdataID, &temp_int);
		
		temp_int.data.i = 1;
		PfAdd(sc, &sc->tempInt, &sc->sdataID, &temp_int);
		temp_int.data.i = 2;
		PfDiv(sc, &sc->tempInt, &sc->tempInt, &temp_int);
		PfMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->tempInt, &temp_int);
		if (sc->mergeSequencesR2C) {
			PfMovNeg(sc, &sc->w.data.c[0], &sc->regIDs[i].data.c[0]);
			PfMovNeg(sc, &sc->w.data.c[1], &sc->temp.data.c[0]);
		}
		else {
			PfMovNeg(sc, &sc->w.data.c[0], &sc->temp.data.c[0]);
		}
		PfIf_else(sc);
		if (sc->mergeSequencesR2C) {
			PfMov(sc, &sc->w.data.c[0], &sc->regIDs[i].data.c[0]);
			PfMov(sc, &sc->w.data.c[1], &sc->temp.data.c[0]);
		}
		else {
			PfMov(sc, &sc->w.data.c[0], &sc->temp.data.c[0]);
		}
		PfIf_end(sc);

		temp_int.data.i = 2;
		PfDiv(sc, &sc->tempInt, &sc->sdataID, &temp_int);
		PfMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->tempInt, &temp_int);
		if (sc->mergeSequencesR2C) {
			PfAdd(sc, &sc->w.data.c[0], &sc->w.data.c[0], &sc->regIDs[i].data.c[1]);
			PfAdd(sc, &sc->w.data.c[1], &sc->w.data.c[1], &sc->temp.data.c[1]);
		}
		else {
			PfAdd(sc, &sc->w.data.c[0], &sc->w.data.c[0], &sc->temp.data.c[1]);
		}
		PfIf_else(sc);
		if (sc->mergeSequencesR2C) {
			PfSub(sc, &sc->w.data.c[0], &sc->w.data.c[0], &sc->regIDs[i].data.c[1]);
			PfSub(sc, &sc->w.data.c[1], &sc->w.data.c[1], &sc->temp.data.c[1]);
		}
		else {
			PfSub(sc, &sc->w.data.c[0], &sc->w.data.c[0], &sc->temp.data.c[1]);
		}
		PfIf_end(sc);

		PfIf_end(sc);
		PfIf_end(sc);


		temp_int.data.i = 3 * fftDim.data.i / 4;
		PfIf_ge_start(sc, &sc->sdataID, &temp_int);
		
		temp_int.data.i = 1;
		PfAdd(sc, &sc->tempInt, &sc->sdataID, &temp_int);
		temp_int.data.i = 2;
		PfDiv(sc, &sc->tempInt, &sc->tempInt, &temp_int);
		PfMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->tempInt, &temp_int);
		if (sc->mergeSequencesR2C) {
			PfMovNeg(sc, &sc->w.data.c[0], &sc->regIDs[i].data.c[0]);
			PfMovNeg(sc, &sc->w.data.c[1], &sc->temp.data.c[0]);
		}
		else {
			PfMovNeg(sc, &sc->w.data.c[0], &sc->temp.data.c[0]);
		}
		PfIf_else(sc);
		if (sc->mergeSequencesR2C) {
			PfMov(sc, &sc->w.data.c[0], &sc->regIDs[i].data.c[0]);
			PfMov(sc, &sc->w.data.c[1], &sc->temp.data.c[0]);
		}
		else {
			PfMov(sc, &sc->w.data.c[0], &sc->temp.data.c[0]);
		}
		PfIf_end(sc);

		temp_int.data.i = 2;
		PfDiv(sc, &sc->tempInt, &sc->sdataID, &temp_int);
		PfMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->tempInt, &temp_int);
		if (sc->mergeSequencesR2C) {
			PfSub(sc, &sc->w.data.c[0], &sc->w.data.c[0], &sc->regIDs[i].data.c[1]);
			PfSub(sc, &sc->w.data.c[1], &sc->w.data.c[1], &sc->temp.data.c[1]);
		}
		else {
			PfSub(sc, &sc->w.data.c[0], &sc->w.data.c[0], &sc->temp.data.c[1]);
		}
		PfIf_else(sc);
		if (sc->mergeSequencesR2C) {
			PfAdd(sc, &sc->w.data.c[0], &sc->w.data.c[0], &sc->regIDs[i].data.c[1]);
			PfAdd(sc, &sc->w.data.c[1], &sc->w.data.c[1], &sc->temp.data.c[1]);
		}
		else {
			PfAdd(sc, &sc->w.data.c[0], &sc->w.data.c[0], &sc->temp.data.c[1]);
		}
		PfIf_end(sc);

		PfIf_end(sc);

		temp_double.data.d = pfFPinit("1.41421356237309504880168872420969807856967");
		if (sc->mergeSequencesR2C) {
			temp_double.data.d *= pfFPinit("0.5");
		}

		if (sc->mergeSequencesR2C) {
			PfMul(sc, &sc->regIDs[i], &sc->w, &temp_double, 0);
		}
		else {
			PfMul(sc, &sc->regIDs[i].data.c[0], &sc->w.data.c[0], &temp_double, 0);
		}

		if (sc->axis_id > 0) {
			temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
		else {
			temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}
		}
	}
	if (sc->useDisableThreads) {
		PfIf_end(sc);
	}

	if (sc->performDST) {
		appendBarrierVkFFT(sc);
		if (sc->useDisableThreads) {
			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
		}
		for (pfUINT i = 0; i < (pfUINT)used_registers.data.i; i++) {
			if (sc->axis_id > 0) {
				temp_int.data.i = (i)*sc->localSize[1].data.i;

				PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

				temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					//check that we only read fftDim * local batch data
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
					PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
				}
			}
			else {
				if (sc->localSize[1].data.i == 1) {
					temp_int.data.i = (i)*sc->localSize[0].data.i;

					PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
				}
				else {
					PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

					temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

					PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
					PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
				}
				temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					//check that we only read fftDim * local batch data
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim_half * &sc->localSize[0]);
					PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
				}
			}
			if (sc->axis_id > 0) {
				temp_int.data.i = fftDim.data.i - 1;
				PfSub(sc, &sc->combinedID, &temp_int, &sc->combinedID);

				PfMul(sc, &sc->sdataID, &sc->combinedID, &sc->sharedStride, 0);

				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);
			}
			else {
				if (sc->stridedSharedLayout) {
					PfMod(sc, &sc->sdataID, &sc->combinedID, &fftDim);
					temp_int.data.i = fftDim.data.i - 1;
					PfSub(sc, &sc->sdataID, &temp_int, &sc->sdataID);
					PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);

					PfDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
					
					PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
				}
				else {
					PfMod(sc, &sc->sdataID, &sc->combinedID, &fftDim);
					temp_int.data.i = fftDim.data.i - 1;
					PfSub(sc, &sc->sdataID, &temp_int, &sc->sdataID);

					PfDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
					PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);

					PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
				}
			}
			appendRegistersToShared(sc, &sc->sdataID, &sc->regIDs[i]);
			if (sc->axis_id > 0) {
				temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					PfIf_end(sc);
				}
			}
			else {
				temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					PfIf_end(sc);
				}
			}
		}
		if (sc->useDisableThreads) {
			PfIf_end(sc);
		}
		appendBarrierVkFFT(sc);
		
		if (sc->useDisableThreads) {
			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
		}
		for (pfUINT i = 0; i < (pfUINT)used_registers.data.i; i++) {
			if (sc->axis_id > 0) {
				temp_int.data.i = (i)*sc->localSize[1].data.i;

				PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);

				temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					//check that we only read fftDim * local batch data
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
					PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
				}
			}
			else {
				if (sc->localSize[1].data.i == 1) {
					temp_int.data.i = (i)*sc->localSize[0].data.i;

					PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
				}
				else {
					PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);

					temp_int.data.i = (i)*sc->localSize[0].data.i * sc->localSize[1].data.i;

					PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
					PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
				}
				temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					//check that we only read fftDim * local batch data
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim_half * &sc->localSize[0]);
					PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
				}
			}
			if (sc->axis_id > 0) {
				PfMul(sc, &sc->sdataID, &sc->combinedID, &sc->sharedStride, 0);

				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);
			}
			else {
				if (sc->stridedSharedLayout) {
					PfMod(sc, &sc->sdataID, &sc->combinedID, &fftDim);
					PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);

					PfDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
					
					PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
				}
				else {
					PfMod(sc, &sc->sdataID, &sc->combinedID, &fftDim);
					
					PfDiv(sc, &sc->tempInt, &sc->combinedID, &fftDim);
					PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);

					PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
				}
			}
			appendSharedToRegisters(sc, &sc->regIDs[i], &sc->sdataID);
			if (sc->axis_id > 0) {
				temp_int.data.i = (i + 1) * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					PfIf_end(sc);
				}
			}
			else {
				temp_int.data.i = (i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;
				temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;
				if (temp_int.data.i > temp_int1.data.i) {
					PfIf_end(sc);
				}
			}
		}
		if (sc->useDisableThreads) {
			PfIf_end(sc);
		}
	}
	sc->writeFromRegisters = 1;
	
	return;
}

#endif
