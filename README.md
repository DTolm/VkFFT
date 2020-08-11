[![Build Status](https://travis-ci.com/DTolm/VkFFT.svg?token=nMgUQeqx7PXMeCFaXqsb&branch=master)](https://travis-ci.com/github/DTolm/VkFFT)
# VkFFT - Vulkan Fast Fourier Transform library
VkFFT is an efficient GPU-accelerated multidimensional Fast Fourier Transform library for Vulkan projects. VkFFT aims to provide community with an open-source alternative to Nvidia's cuFFT library, while achieving better performance. VkFFT is written in C language.
## Currently supported features:
  - 1D/2D/3D systems
  - Forward and inverse directions of FFT
  - Maximum dimension size is 4096, 32-bit float 
  - Radix-2/4/8 FFT, only power of two systems
  - All transformations are performed in-place with no performance loss. Out-of-place transforms are supported by selecting different input/output buffers.
  - Complex to complex (C2C), real to complex (R2C) and complex to real (C2R) transformations. R2C and C2R are optimized to run up to 2x times faster than C2C (2D and 3D case only)
  - 1x1, 2x2, 3x3 convolutions with symmetric or nonsymmetric kernel
  - Native zero padding to model open systems (up to 2x faster than simply padding input array with zeros)
  - WHDCN layout - data is stored in the following order (sorted by increase in strides): the width, the height, the depth, the coordinate (the number of feature maps), the batch number
  - Multiple feature/batch convolutions - one input, multiple kernels
  - Works on Nvidia, AMD and Intel GPUs (tested on Nvidia GTX 1660 Ti and Intel UHD 620)
  - Header-only (+precompiled shaders) library with Vulkan interface, which allows to append VkFFT directly to user's command buffer
## Future release plan
 - ##### Almost ready: 
   - Double and half-precision arithmetics
   - 8192 and 16384 dimension sizes
 - ##### Planned
    - Publication based on implemented optimizations
    - Mobile GPU support
 - ##### Ambitious
    - Multiple GPU job splitting

## Installation
Include the vkFFT.h file and specify path to the shaders folder in CMake or from C interface. Sample CMakeLists.txt file configures project based on Vulkan_FFT.cpp file, which contains four examples on how to use VkFFT to perform FFT, iFFT and convolution calculations, use zero padding and multiple feature/batch convolutions.
## How to use VkFFT
VkFFT.h is a library which can append FFT, iFFT or convolution calculation to the user defined command buffer. It operates on storage buffers allocated by user and doesn't require any additional memory by itself. All computations are fully based on Vulkan compute shaders with no CPU usage except for FFT planning. VkFFT creates and optimizes memory layout by itself and performs FFT with the best chosen parameters. For an example application, see Vulkan_FFT.cpp file, which has comments explaining the VkFFT configuration process.\
Picture below shows how data is restructured during the R2C transform depending on the system dimensions. This layout has minimal transfers between on-chip memory and graphics card (one read and one write per FFT axis + transposition if axis dimension is ≥ 256). If convolution is performed, it is embedded into the last FFT axis, which reduces memory transfers even further.
![alt text](https://github.com/dtolm/VkFFT/blob/master/FFT_memory_layout.png?raw=true)
## Benchmark results in comparison to cuFFT
To measure how Vulkan FFT implementation works in comparison to cuFFT, we will perform a number of 2D and 3D tests. The test will consist of performing R2C FFT and inverse C2R FFT consecutively multiple times to calculate average time required. cuFFT uses out-of-place configuration while VkFFT uses in-place. The results are obtained on Nvidia 1660 Ti graphics card with no other GPU load. Launching example 0 from Vulkan_FFT.cpp performs VkFFT benchmark, benchmark_cuFFT.cu file contains similar benchmark script for cuFFT library. 
![alt text](https://github.com/DTolm/VkFFT/blob/master/vkfft_benchmark_1.png?raw=true)
![alt text](https://github.com/DTolm/VkFFT/blob/master/vkfft_benchmark_2.png?raw=true)
## Contact information
Initial version of VkFFT is developed by Tolmachev Dmitrii\
Peter Grünberg Institute and Institute for Advanced Simulation, Forschungszentrum Jülich,  D-52425 Jülich, Germany\
Email 1: <d.tolmachev@fz-juelich.de>\
Email 2: <dtolm96@gmail.com>