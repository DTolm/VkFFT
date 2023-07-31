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
#ifndef VKFFT_ZEROPAD_H
#define VKFFT_ZEROPAD_H
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_StringManagement/vkFFT_StringManager.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_MathUtils/vkFFT_MathUtils.h"

static inline void checkZeropadStart(VkFFTSpecializationConstantsLayout* sc, PfContainer* location, int axisCheck) {
	//return if sequence is full of zeros from the start
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
	if ((sc->frequencyZeropadding)) {
        for (int i =0; i <sc->axis_id; i++){
            if (axisCheck == i) {
                if (sc->performZeropaddingFull[i]) {
                    if (sc->fft_zeropad_left_full[i].data.i < sc->fft_zeropad_right_full[i].data.i) {
                        PfIf_ge_start(sc, location, &sc->fft_zeropad_left_full[i]);
                        PfIf_lt_start(sc, location, &sc->fft_zeropad_right_full[i]);
                    }
                }
            }
        }
	}
	else {
        for (int i = (sc->axis_id+1); i < sc->numFFTdims; i++){
            if (axisCheck == i) {
                if (sc->performZeropaddingFull[i]) {
                    if (sc->fft_zeropad_left_full[i].data.i < sc->fft_zeropad_right_full[i].data.i) {
                        PfIf_ge_start(sc, location, &sc->fft_zeropad_left_full[i]);
                        PfIf_lt_start(sc, location, &sc->fft_zeropad_right_full[i]);
                    }
                }
            }
        }
	}
	return;
}
static inline void checkZeropadEnd(VkFFTSpecializationConstantsLayout* sc, int axisCheck) {
	//return if sequence is full of zeros from the start
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
    if ((sc->frequencyZeropadding)) {
        for (int i =0; i <sc->axis_id; i++){
            if (axisCheck == i) {
                if (sc->performZeropaddingFull[i]) {
                    if (sc->fft_zeropad_left_full[i].data.i < sc->fft_zeropad_right_full[i].data.i) {
                        PfIf_end(sc);
                        PfIf_end(sc);
                    }
                }
            }
        }
    }
    else {
        for (int i = (sc->axis_id+1); i < sc->numFFTdims; i++){
            if (axisCheck == i) {
                if (sc->performZeropaddingFull[i]) {
                    if (sc->fft_zeropad_left_full[i].data.i < sc->fft_zeropad_right_full[i].data.i) {
                        PfIf_end(sc);
                        PfIf_end(sc);
                    }
                }
            }
        }
    }
	return;
}

static inline void checkZeropad(VkFFTSpecializationConstantsLayout* sc, PfContainer* location, int axisCheck) {
	//return if sequence is full of zeros from the start
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer temp_int = VKFFT_ZERO_INIT;
	temp_int.type = 31;
    if ((sc->frequencyZeropadding)) {
        for (int i =0; i <sc->axis_id; i++){
            if (axisCheck == i) {
                if (sc->performZeropaddingFull[i]) {
                    if (sc->fft_zeropad_left_full[i].data.i < sc->fft_zeropad_right_full[i].data.i) {
                        sc->useDisableThreads = 1;
                        PfIf_ge_start(sc, location, &sc->fft_zeropad_left_full[i]);
                        PfIf_lt_start(sc, location, &sc->fft_zeropad_right_full[i]);
                        temp_int.data.i = 0;
                        PfMov(sc, &sc->disableThreads, &temp_int);
                        PfIf_end(sc);
                        PfIf_end(sc);
                    }
                }
            }
        }
    }
    else {
        for (int i = (sc->axis_id+1); i < sc->numFFTdims; i++){
            if (axisCheck == i) {
                if (sc->performZeropaddingFull[i]) {
                    if (sc->fft_zeropad_left_full[i].data.i < sc->fft_zeropad_right_full[i].data.i) {
                        sc->useDisableThreads = 1;
                        PfIf_ge_start(sc, location, &sc->fft_zeropad_left_full[i]);
                        PfIf_lt_start(sc, location, &sc->fft_zeropad_right_full[i]);
                        temp_int.data.i = 0;
                        PfMov(sc, &sc->disableThreads, &temp_int);
                        PfIf_end(sc);
                        PfIf_end(sc);
                    }
                }
            }
        }
    }
	return;
}

#endif
