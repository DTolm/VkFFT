[![Build Status](https://travis-ci.com/DTolm/VkFFT.svg?token=nMgUQeqx7PXMeCFaXqsb&branch=master)](https://travis-ci.com/github/DTolm/VkFFT)
# VkFFT - Vulkan Fast Fourier Transform library
VkFFT is an efficient GPU-accelerated multidimensional Fast Fourier Transform library for Vulkan projects. VkFFT aims to provide community with an open-source alternative to Nvidia's cuFFT library, while achieving better performance. VkFFT is written in C language.

## I am looking for a PhD position/job that may be interested in my set of skills. Contact me by email: <d.tolmachev@fz-juelich.de> | <dtolm96@gmail.com>

## If you tried the provided VkFFT and cuFFT benchmark scripts, I would really appreciate if you send me the results, via E-mail or GitHub issue

## Currently supported features:
  - 1D/2D/3D systems
  - Forward and inverse directions of FFT
  - Support for big FFT dimension sizes. Current limits: C2C - (2^24, 2^15, 2^15), C2R/R2C - (2^14, 2^15, 2^15) with register overutilization. (will be increased later). y and z axis are capped due to Vulkan maxComputeWorkGroupCount and will be increased later. x axis size will also be improved, after tests of the >2 passes big FFTs algorithm.
  - Radix-2/4/8 FFT, only power of two systems. 
  - All transformations are performed in-place with no performance loss. Out-of-place transforms are supported by selecting different input/output buffers.
  - No transpositions. Note: data is not reshuffled after the four stage FFT algorithm (for big sequences). Doesn't matter for convolutions - they return to the input ordering.
  - Complex to complex (C2C), real to complex (R2C) and complex to real (C2R) transformations. R2C and C2R are optimized to run up to 2x times faster than C2C (2D and 3D case only)
  - 1x1, 2x2, 3x3 convolutions with symmetric or nonsymmetric kernel (only for one upload last size for now - 1k in the last dimension on Nvidia. Will be changed in the next update)
  - Native zero padding to model open systems (up to 2x faster than simply padding input array with zeros)
  - WHDCN layout - data is stored in the following order (sorted by increase in strides): the width, the height, the depth, the coordinate (the number of feature maps), the batch number
  - Multiple feature/batch convolutions - one input, multiple kernels
  - Works on Nvidia, AMD and Intel GPUs (tested on Nvidia GTX 1660 Ti and Intel UHD 620)
  - Header-only (+precompiled shaders) library with Vulkan interface, which allows to append VkFFT directly to user's command buffer
## Future release plan
 - ##### Almost ready: 
   - Double and half-precision arithmetics
 - ##### Planned
    - Publication based on implemented optimizations
    - Mobile GPU support
	  - Radix 3,5... support
 - ##### Ambitious
    - Multiple GPU job splitting

## Installation
Include the vkFFT.h file and specify path to the shaders folder in CMake or from C interface. Sample CMakeLists.txt file configures project based on Vulkan_FFT.cpp file, which contains five examples on how to use VkFFT to perform FFT, iFFT and convolution calculations, use zero padding, multiple feature/batch convolutions and C2C FFTs of big systems.
## How to use VkFFT
VkFFT.h is a library which can append FFT, iFFT or convolution calculation to the user defined command buffer. It operates on storage buffers allocated by user and doesn't require any additional memory by itself. All computations are fully based on Vulkan compute shaders with no CPU usage except for FFT planning. VkFFT creates and optimizes memory layout by itself and performs FFT with the best chosen parameters. For an example application, see Vulkan_FFT.cpp file, which has comments explaining the VkFFT configuration process.\
Picture below is now outdated. VkFFT achieves striding by grouping nearby FFTs instead of transpositions now.
![alt text](https://github.com/dtolm/VkFFT/blob/master/FFT_memory_layout.png?raw=true)
## Benchmark results in comparison to cuFFT
To measure how Vulkan FFT implementation works in comparison to cuFFT, we will perform a number of 2D and 3D tests. The test will consist of performing R2C FFT and inverse C2R FFT consecutively multiple times to calculate average time required. cuFFT uses out-of-place configuration while VkFFT uses in-place. The results are obtained on Nvidia 1660 Ti graphics card with no other GPU load. Launching example 0 from Vulkan_FFT.cpp performs VkFFT benchmark, benchmark_cuFFT.cu file contains similar benchmark script for cuFFT library. 
![alt text](https://github.com/DTolm/VkFFT/blob/master/vkfft_benchmark_1.png?raw=true)
![alt text](https://github.com/DTolm/VkFFT/blob/master/vkfft_benchmark_2.png?raw=true)
## Precision comparison of cuFFT/VkFFT/FFTW
To measure how VkFFT (single precision) results compare to cuFFT (single precision) and FFTW (double precision), a set of ~50 systems was filled with random complex data on the scale of [-1,1] and one C2C transform was performed on each system. The precision_cuFFT_VkFFT_FFTW.cu script contains the comparison code, which calculates for each value of the transformed system:

- Max difference between cuFFT/VkFFT result and FFTW result
- Average difference between cuFFT/VkFFT result and FFTW result
- Max ratio of the difference between cuFFT/VkFFT result and FFTW result to the FFTW result
- Average ratio of the difference between cuFFT/VkFFT result and FFTW result to the FFTW result

The precision_cuFFT_VkFFT_FFTW.txt file contains the results for Nvidia's 1660Ti GPU and AMD Ryzen 2700 CPU. On average, the results fluctuate both for cuFFT and VkFFT with no clear winner. Max ratio stays in range of 2% for both cuFFT and VkFFT, while average ratio stays below 1e-6. 

## Contact information
Initial version of VkFFT is developed by Tolmachev Dmitrii\
Peter Grünberg Institute and Institute for Advanced Simulation, Forschungszentrum Jülich,  D-52425 Jülich, Germany\
E-mail 1: <d.tolmachev@fz-juelich.de>\
E-mail 2: <dtolm96@gmail.com>