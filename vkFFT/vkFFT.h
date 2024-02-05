// This file is part of VkFFT, a Vulkan Fast Fourier Transform library
//
// Copyright (C) 2020 - present Dmitrii Tolmachev <dtolm96@gmail.com>
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

#ifndef VKFFT_H
#define VKFFT_H

#include <locale.h>
#include <memory.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>
#if(VKFFT_BACKEND==0)
#include "vulkan/vulkan.h"
#include "glslang_c_interface.h"
#elif(VKFFT_BACKEND==1)
#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#ifndef CUDA_TOOLKIT_ROOT_DIR
#define CUDA_TOOLKIT_ROOT_DIR ""
#endif
#elif(VKFFT_BACKEND==2)
#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_complex.h>
#elif(VKFFT_BACKEND==3)
#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#endif

#ifdef __cplusplus
#define VKFFT_ZERO_INIT {}
#else
#define VKFFT_ZERO_INIT {0}
#endif

#ifndef VKFFT_MAX_FFT_DIMENSIONS
#define VKFFT_MAX_FFT_DIMENSIONS 4
#endif

#ifdef VKFFT_USE_DOUBLEDOUBLE_FP128

#define VKFFT_USE_QUADMATH_FP128 // for now the only implementation, but defining these functions as mpfr should also be possible
#include <quadmath.h>
//#define pfQ __float128
#define pfLD __float128
#define pfUINT uint64_t
#define pfINT int64_t
#define pfsin sinq
#define pfcos cosq
#define pfceil ceilq
#define pffloor floorq
#define pfsqrt sqrtq

#define pfFPinit(x) strtoflt128(x, 0)

#else
#define pfLD long double
#define pfUINT uint64_t
#define pfINT int64_t
#define pfsin sin
#define pfcos cos
#define pfceil ceil
#define pffloor floor
#define pfsqrt sqrt

#define pfFPinit(x) strtold(x, 0)

#endif

#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT/vkFFT_AppManagement/vkFFT_RunApp.h"
#include "vkFFT/vkFFT_AppManagement/vkFFT_InitializeApp.h"
#include "vkFFT/vkFFT_AppManagement/vkFFT_DeleteApp.h"

static inline int VkFFTGetVersion() {
	return 10304; //X.XX.XX format
}
#endif
