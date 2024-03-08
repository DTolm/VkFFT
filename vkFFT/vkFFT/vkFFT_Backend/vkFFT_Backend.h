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

#ifndef VKFFT_BACKEND_H
#define VKFFT_BACKEND_H

#ifndef VKFFT_BACKEND
# error "VKFFT_BACKEND is undefined"
#endif

#define VKFFT_BACKEND_VULKAN     ((VKFFT_BACKEND==0) ? 1 : 0)
#define VKFFT_BACKEND_CUDA       ((VKFFT_BACKEND==1) ? 1 : 0)
#define VKFFT_BACKEND_HIP        ((VKFFT_BACKEND==2) ? 1 : 0)
#define VKFFT_BACKEND_OPENCL     ((VKFFT_BACKEND==3) ? 1 : 0)
#define VKFFT_BACKEND_LEVEL_ZERO ((VKFFT_BACKEND==4) ? 1 : 0)
#define VKFFT_BACKEND_METAL      ((VKFFT_BACKEND==5) ? 1 : 0)

#endif /* VKFFT_BACKEND_H */