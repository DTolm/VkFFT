[![Build Status](https://travis-ci.com/DTolm/VkFFT.svg?token=nMgUQeqx7PXMeCFaXqsb&branch=master)](https://travis-ci.com/github/DTolm/VkFFT)
# VkFFT - Vulkan Fast Fourier Transform library
VkFFT is an efficient GPU-accelerated multidimensional Fast Fourier Transform library for Vulkan projects. VkFFT aims to provide community with an open-source alternative to Nvidia's cuFFT library, while achieving better performance. VkFFT is written in C language.

## I am looking for a PhD position/job that may be interested in my set of skills. Contact me by email: <d.tolmachev@fz-juelich.de> | <dtolm96@gmail.com>

## Added Windows executables for benchmark: versions with CUDA benchmark (requires CUDA 11) and without (requires only graphics drivers). Both require FFTW dll placed in the same location as executable. Uses ~3.5GB of VRAM. 

## Benchmark results of VkFFT can be found here: https://openbenchmarking.org/test/pts/vkfft

## Currently supported features:
  - 1D/2D/3D systems
  - Forward and inverse directions of FFT
  - Support for big FFT dimension sizes. Current limits in single and half precision: C2C - (2^32, 2^32, 2^32). C2R/R2C with register overutilization - (2^14, 2^32, 2^32). (will be increased later). Current limits in double precision: C2C - (2^32, 2^32, 2^32), C2R/R2C - (2^11, 2^32, 2^32) with no register overutilization. Memory should be allocated as a single buffer in one heap.
  - Radix-2/4/8 FFT, only power of two systems. 
  - Single, double and half precision support. Double precision uses CPU generated LUT tables. Half precision still does all computations in single and only uses half precision to store data.
  - All transformations are performed in-place with no performance loss. Out-of-place transforms are supported by selecting different input/output buffers.
  - No additional transposition uploads. Note: data can be reshuffled after the four step FFT algorithm with additional buffer (for big sequences). Doesn't matter for convolutions - they return to the input ordering (saves memory).
  - Complex to complex (C2C), real to complex (R2C) and complex to real (C2R) transformations. R2C and C2R are optimized to run up to 2x times faster than C2C (2D and 3D case only)
  - 1x1, 2x2, 3x3 convolutions with symmetric or nonsymmetric kernel (no register overutilization)
  - Native zero padding to model open systems (up to 2x faster than simply padding input array with zeros). Can specify the range of sequences filled with zeros and the direction where zeropadding is applied (read or write stage)
  - WHDCN layout - data is stored in the following order (sorted by increase in strides): the width, the height, the depth, the coordinate (the number of feature maps), the batch number
  - Multiple feature/batch convolutions - one input, multiple kernels
  - Multiple input/output/temporary buffer split. Allows to use data split between different memory allocations and mitigate 4GB single allocation limit.
  - Works on Nvidia, AMD and Intel GPUs (tested on Nvidia RTX 3080, GTX 1660 Ti, AMD Radeon VII and Intel UHD 620)
  - Header-only library with Vulkan interface, which allows to append VkFFT directly to user's command buffer. Shaders are compiled once during the plan creation stage
## Future release plan
 - ##### Planned
    - Publication based on implemented optimizations
    - Mobile GPU support
	- Radix 3,5... support
 - ##### Ambitious
    - Multiple GPU job splitting

## Installation
Include the vkFFT.h file and glslang compiler. Sample CMakeLists.txt file configures project based on Vulkan_FFT.cpp file, which contains examples on how to use VkFFT to perform FFT, iFFT and convolution calculations, use zero padding, multiple feature/batch convolutions, C2C FFTs of big systems, R2C/C2R transforms, double precision FFTs, half precision FFTs.\
For single and double precision, Vulkan 1.0 is required. For half precision, Vulkan 1.1 is required.
## Command-line interface
VkFFT has a command-line interface with the following set of commands:\
-h: print help\
-devices: print the list of available GPU devices\
-d X: select GPU device (default 0)\
-o NAME: specify output file path\
-vkfft X: launch VkFFT sample X (0-13) (if FFTW is enabled in CMakeLists.txt)\
-cufft X: launch cuFFT sample X (0-3) (if enabled in CMakeLists.txt)\
-test: (or no other keys) launch all VkFFT and cuFFT benchmarks\
So, the command to launch single precision benchmark of VkFFT and cuFFT and save log to output.txt file on device 0 will look like this on Windows:\
.\Vulkan_FFT.exe -d 0 -o output.txt -vkfft 0 -cufft 0\
For double precision benchmark, replace -vkfft 0 -cufft 0 with -vkfft 1 -cufft 1. For half precision benchmark, replace -vkfft 0 -cufft 0 with -vkfft 2 -cufft 2.
## How to use VkFFT
VkFFT.h is a library which can append FFT, iFFT or convolution calculation to the user defined command buffer. It operates on storage buffers allocated by user and doesn't require any additional memory by itself. All computations are fully based on Vulkan compute shaders with no CPU usage except for FFT planning. VkFFT creates and optimizes memory layout by itself and performs FFT with the best chosen parameters. For an example application, see Vulkan_FFT.cpp file, which has comments explaining the VkFFT configuration process.\
VkFFT achieves striding by grouping nearby FFTs instead of transpositions.
![alt text](https://github.com/dtolm/VkFFT/blob/master/FFT_memory_layout.png?raw=true)
## Benchmark results in comparison to cuFFT
To measure how Vulkan FFT implementation works in comparison to cuFFT, we will perform a number of 1D, 2D and 3D tests, ranging from the small systems to the big ones. The test will consist of performing C2C FFT and inverse C2C FFT consecutively multiple times to calculate average time required. The results are obtained on Nvidia RTX 3080, AMD Radeon VII and AMD Radeon 6800XT graphics cards with no other GPU load. Launching -test key from Vulkan_FFT.cpp performs VkFFT/cuFFT benchmark. The overall benchmark score is calculated as an averaged performance score over presented set of systems (the bigger - the better): sum(system_size/average_iteration_time) /num_benchmark_samples

The stable flat lines present for small sequence lengths indicate that time scales linearly with the system size, so the bigger the bandwidth the better the result will be. The stepwise drops occur once the amount of transfers increases from to 2x and to 3x when compute unit can't hold full sequence and splits it into combination of smaller ones. Radeon VII is faster than RTX 3080 below 2^18 (=2MB - page file size on AMD due to it having HBM2 memory with a higher bandwidth, however, this GPU apparently has TLB miss problems on large buffer sizes. On RTX 3080, VkFFT is faster than cuFFT in single precision batched 1D FFTs on the range from 2^3 to 2^27:
![alt text](https://github.com/DTolm/VkFFT/blob/master/vkfft_benchmark_single.png?raw=true)
In double precision Radeon VII is able to get advantage due to its high double precision core count. Radeon RX 6800XT can store LUT in L3 cache and has higher double precision core count as well:
![alt text](https://github.com/DTolm/VkFFT/blob/master/vkfft_benchmark_double.png?raw=true)
In half precision mode, VkFFT only uses it for data storage, all computations are performed in single.It still proves to be enough to get stable 2x performance gain on RTX 3080: 
![alt text](https://github.com/DTolm/VkFFT/blob/master/vkfft_benchmark_half.png?raw=true)
Multidimensional systems are optimized as well. Benchmark shows Radeon RX 6800XT can store systems up to 128MB in L3 cache for big performance gains. Native support for zero padding allows to transfer less data and get up to 3x performance boost in multidimensional FFTs:
![alt text](https://github.com/DTolm/VkFFT/blob/master/vkfft_benchmark_2d.png?raw=true)
![alt text](https://github.com/DTolm/VkFFT/blob/master/vkfft_benchmark_3d.png?raw=true)
## Precision comparison of cuFFT/VkFFT/FFTW
To measure how VkFFT (single/double/half precision) results compare to cuFFT (single/double/half precision) and FFTW (double precision), a set of ~60 systems covering full FFT range was filled with random complex data on the scale of [-1,1] and one C2C transform was performed on each system. Samples 11(single), 12(double), 13(half) calculate for each value of the transformed system:

- Max difference between cuFFT/VkFFT result and FFTW result
- Average difference between cuFFT/VkFFT result and FFTW result
- Max ratio of the difference between cuFFT/VkFFT result and FFTW result to the FFTW result
- Average ratio of the difference between cuFFT/VkFFT result and FFTW result to the FFTW result

FFTW is required to launch these samples (specify in CMakeLists include and library directories). If cuFFT is disabled, only FFTW/VkFFT results are calculated.\
The precision_cuFFT_VkFFT_FFTW.txt file contains the single precision results for Nvidia's 1660Ti GPU and AMD Ryzen 2700 CPU. On average, the results fluctuate both for cuFFT and VkFFT with no clear winner in single precision. Max ratio stays in range of 2% for both cuFFT and VkFFT, while average ratio stays below 1e-6.\
The precision_cuFFT_VkFFT_FFTW_double.txt file contains the double precision results for Nvidia's 1660Ti GPU and AMD Ryzen 2700 CPU. On average, VkFFT is more precise than cuFFT in double precision (see: max_difference and max_eps coloumns), however it is also ~20% slower (vkfft_benchmark_double.png). Note that double precision is still in testing and these results may change in the future. Max ratio stays in range of 5e-10% for both cuFFT and VkFFT, while average ratio stays below 1e-15. Overall, double precision is ~7 times slower than single on Nvidia's 1660Ti GPU.\
The precision_cuFFT_VkFFT_FFTW_half.txt file contains the half precision results for Nvidia's 1660Ti GPU and AMD Ryzen 2700 CPU. On average, VkFFT is at least two times more precise than cuFFT in half precision (see: max_difference and max_eps coloumns), while being faster on average (vkfft_benchmark_half.png). Note that half precision is still in testing and is only used to store data in VkFFT. cuFFT script can probably also be improved. Average ratio stays in range of 0.2% for both cuFFT and VkFFT. Overall, half precision of VkFFT is ~50%-100% times faster than single on Nvidia's 1660Ti GPU.
## Contact information
Initial version of VkFFT is developed by Tolmachev Dmitrii\
Peter Grünberg Institute and Institute for Advanced Simulation, Forschungszentrum Jülich,  D-52425 Jülich, Germany\
E-mail 1: <d.tolmachev@fz-juelich.de>\
E-mail 2: <dtolm96@gmail.com>