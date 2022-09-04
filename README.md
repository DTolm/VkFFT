[![Build Status](https://travis-ci.com/DTolm/VkFFT.svg?token=nMgUQeqx7PXMeCFaXqsb&branch=master)](https://travis-ci.com/github/DTolm/VkFFT)
# VkFFT - Vulkan/CUDA/HIP/OpenCL/Level Zero Fast Fourier Transform library
VkFFT is an efficient GPU-accelerated multidimensional Fast Fourier Transform library for Vulkan/CUDA/HIP/OpenCL/Level Zero projects. VkFFT aims to provide the community with an open-source alternative to Nvidia's cuFFT library while achieving better performance. VkFFT is written in C language and supports Vulkan, CUDA, HIP, OpenCL and Level Zero as backends.

## Check out my panel at Nvidia's GTC 2021 in Higher Education and Research category: https://gtc21.event.nvidia.com/

## Python interface to VkFFT can be found here: https://github.com/vincefn/pyvkfft

## Rust bindings to VkFFT can be found here: https://github.com/semio-ai/vkfft-rs

## Benchmark results of VkFFT can be found here: https://openbenchmarking.org/test/pts/vkfft

## Currently supported features:
  - 1D/2D/3D systems
  - Forward and inverse directions of FFT
  - Support for big FFT dimension sizes. Current limits: C2C or even C2R/R2C - (2^32, 2^32, 2^32).  Odd C2R/R2C - (2^12, 2^32, 2^32). R2R - (2^12, 2^12, 2^12). Depends on the amount of shared memory on the device. (will be increased later).
  - Radix-2/3/4/5/7/8/11/13 FFT. Sequences using radix 3, 5, 7, 11 and 13 have comparable performance to that of powers of 2.
  - Rader's FFT algorithm for primes from 17 up to max shared memory length (~10000). Inlined and done without additional memory transfers.
  - Bluestein's FFT algorithm for all other sequences. Full coverage of C2C range, single upload (2^12, 2^12, 2^12) for R2C/C2R/R2R. Optimized to have as few memory transfers as possible by using zero padding and merged convolution support of VkFFT
  - Single, double and half precision support. Double precision uses CPU-generated LUT tables. Half precision still does all computations in single and only uses half precision to store data.
  - All transformations are performed in-place with no performance loss. Out-of-place transforms are supported by selecting different input/output buffers.
  - No additional transposition uploads. Note: Data can be reshuffled after the Four Step FFT algorithm with an additional buffer (for big sequences). Doesn't matter for convolutions - they return to the input ordering (saves memory).
  - Complex to complex (C2C), real to complex (R2C), complex to real (C2R) transformations and real to real (R2R) Discrete Cosine Transformations of types I, II, III and IV. R2R, R2C and C2R are optimized to run up to 2x times faster than C2C and take 2x less memory
  - 1x1, 2x2, 3x3 convolutions with symmetric or nonsymmetric kernel (no register overutilization)
  - Native zero padding to model open systems (up to 2x faster than simply padding input array with zeros). Can specify the range of sequences filled with zeros and the direction where zero padding is applied (read or write stage)
  - WHDCN layout - data is stored in the following order (sorted by increase in strides): the width, the height, the depth, the coordinate (the number of feature maps), the batch number
  - Multiple feature/batch convolutions - one input, multiple kernels
  - Multiple input/output/temporary buffer split. Allows using data split between different memory allocations and mitigates 4GB single allocation limit.
  - Works on Nvidia, AMD and Intel GPUs. And Raspberry Pi 4 GPU.
  - Works on Windows, Linux and macOS
  - VkFFT supports Vulkan, CUDA, HIP, OpenCL and Level Zero as backend to cover wide range of APIs
  - Header-only library with Vulkan interface, which allows appending VkFFT directly to user's command buffer. Kernels are compiled at run-time
## Future release plan
 - ##### Planned
    - Publication based on implemented optimizations
    - Test mobile GPUs (they should work)
 - ##### Ambitious
    - Multiple GPU job splitting

## Installation
Vulkan version:
Include the vkFFT.h file and glslang compiler. Provide the library with correctly chosen VKFFT_BACKEND definition (VKFFT_BACKEND=0 for Vulkan). Sample CMakeLists.txt file configures project based on Vulkan_FFT.cpp file, which contains examples on how to use VkFFT to perform FFT, iFFT and convolution calculations, use zero padding, multiple feature/batch convolutions, C2C FFTs of big systems, R2C/C2R transforms, R2R DCT-I, II, III and IV, double precision FFTs, half precision FFTs.\
For single and double precision, Vulkan 1.0 is required. For half precision, Vulkan 1.1 is required.

CUDA/HIP:
Include the vkFFT.h file and make sure your system has NVRTC/HIPRTC built. Provide the library with correctly chosen VKFFT_BACKEND definition. Only single/double precision for now.\
To build CUDA/HIP version of the benchmark, replace VKFFT_BACKEND in CMakeLists (line 5) with the correct one and optionally enable FFTW. VKFFT_BACKEND=1 for CUDA, VKFFT_BACKEND=2 for HIP.

OpenCL:
Include the vkFFT.h file. Provide the library with correctly chosen VKFFT_BACKEND definition. Only single/double precision for now.\
To build OpenCL version of the benchmark, replace VKFFT_BACKEND in CMakeLists (line 5) with the value 3 and optionally enable FFTW.

Level Zero:
Include the vkFFT.h file. Provide the library with correctly chosen VKFFT_BACKEND definition. Clang and llvm-spirv must be valid system calls. Only single/double precision for now.\
To build Level Zero version of the benchmark, replace VKFFT_BACKEND in CMakeLists (line 5) with the value 4 and optionally enable FFTW.
## Command-line interface
VkFFT has a command-line interface with the following set of commands:\
-h: print help\
-devices: print the list of available GPU devices\
-d X: select GPU device (default 0)\
-o NAME: specify output file path\
-vkfft X: launch VkFFT sample X (0-17, 100, 101, 200, 201, 1000-1003) (if FFTW is enabled in CMakeLists.txt)\
-cufft X: launch cuFFT sample X (0-4, 1000-1003) (if enabled in CMakeLists.txt)\
-rocfft X: launch rocFFT sample X (0-4, 1000-1003) (if enabled in CMakeLists.txt)\
-test: (or no other keys) launch all VkFFT and cuFFT benchmarks\
So, the command to launch single precision benchmark of VkFFT and cuFFT and save log to output.txt file on device 0 will look like this on Windows:\
.\Vulkan_FFT.exe -d 0 -o output.txt -vkfft 0 -cufft 0\
For double precision benchmark, replace -vkfft 0 -cufft 0 with -vkfft 1 -cufft 1. For half precision benchmark, replace -vkfft 0 -cufft 0 with -vkfft 2 -cufft 2.
## How to use VkFFT
VkFFT.h is a library that can append FFT, iFFT or convolution calculation to the user-defined command buffer. It operates on storage buffers allocated by the user and doesn't require any additional memory by itself (except for LUT, if they are enabled). All computations are fully based on Vulkan compute shaders with no CPU usage except for FFT planning. VkFFT creates and optimizes memory layout by itself and performs FFT with the best-chosen parameters. For an example application, see Vulkan_FFT.cpp file, which has comments explaining the VkFFT configuration process.\
VkFFT achieves striding by grouping nearby FFTs instead of transpositions. \
Explicit VkFFT documentation can be found in the documentation folder.
## Benchmark results in comparison to cuFFT
The test configuration below takes multiple 1D FFTs of all lengths from the range of 2 to 4096, batch them together so the full system takes from 500MB to 1GB of data and perform multiple consecutive FFTs/iFFTs (-vkfft 1001 key). After that time per a single FFT is obtained by averaging the result.   Total system size will be divided by the time taken by a single transform upload+download, resulting in the estimation of an achieved global bandwidth. The GPUs used in this comparison are Nvidia A100 and AMD MI250. The performance was compared against Nvidia cuFFT (CUDA 11.7 version) and AMD rocFFT (ROCm 5.2 version) libraries in double precision: 
![alt text](https://github.com/DTolm/VkFFT/blob/master/benchmark_plot/fp64_cuda_a100.png?raw=true)
![alt text](https://github.com/DTolm/VkFFT/blob/master/benchmark_plot/fp64_hip_mi250.png?raw=true)
## Precision comparison of cuFFT/VkFFT/FFTW
To measure how VkFFT (single/double/half precision) results compare to cuFFT/rocFFT (single/double/half precision) and FFTW (double precision), a set of ~60 systems covering full FFT range was filled with random complex data on the scale of [-1,1] and one C2C transform was performed on each system. Samples 11(single), 12(double), 13(half) calculate for each value of the transformed system:

- Max difference between cuFFT/rocFFT/VkFFT result and FFTW result
- Average difference between cuFFT/rocFFT/VkFFT result and FFTW result
- Max ratio of the difference between cuFFT/rocFFT/VkFFT result and FFTW result to the FFTW result
- Average ratio of the difference between cuFFT/rocFFT/VkFFT result and FFTW result to the FFTW result

FFTW is required to launch these samples (specify in CMakeLists include and library directories). If cuFFT is disabled, only FFTW/VkFFT results are calculated.\
The precision_cuFFT_VkFFT_FFTW.txt file contains the single precision results for Nvidia's 1660Ti GPU and AMD Ryzen 2700 CPU. On average, the results fluctuate both for cuFFT and VkFFT with no clear winner in single precision. Max ratio stays in the range of 2% for both cuFFT and VkFFT, while the average ratio stays below 1e-6.\
The precision_cuFFT_VkFFT_FFTW_double.txt file contains the double precision results for Nvidia's 1660Ti GPU and AMD Ryzen 2700 CPU. On average, VkFFT is more precise than cuFFT in double precision (see: max_difference and max_eps columns), however, it is also ~20% slower (vkfft_benchmark_double.png). Note that double precision is still in testing and these results may change in the future. Max ratio stays in the range of 5e-10% for both cuFFT and VkFFT, while the average ratio stays below 1e-15. Overall, double precision is ~7 times slower than single on Nvidia's 1660Ti GPU.\
The precision_cuFFT_VkFFT_FFTW_half.txt file contains the half precision results for Nvidia's 1660Ti GPU and AMD Ryzen 2700 CPU. On average, VkFFT is at least two times more precise than cuFFT in half precision (see: max_difference and max_eps columns), while being faster on average (vkfft_benchmark_half.png). Note that half precision is still in testing and is only used to store data in VkFFT. cuFFT script can probably also be improved. The average ratio stays in the range of 0.2% for both cuFFT and VkFFT. Overall, half precision of VkFFT is ~50%-100% times faster than single on Nvidia's 1660Ti GPU.
## Contact information
The initial version of VkFFT is developed by Tolmachev Dmitrii\
E-mail 1: <dtolm96@gmail.com>