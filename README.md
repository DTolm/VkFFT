# VkFFT - Vulkan/CUDA/HIP/OpenCL/Level Zero/Metal Fast Fourier Transform library
VkFFT is an efficient GPU-accelerated multidimensional Fast Fourier Transform library for Vulkan/CUDA/HIP/OpenCL/Level Zero/Metal projects. VkFFT aims to provide the community with an open-source alternative to Nvidia's cuFFT library while achieving better performance. VkFFT is written in C language and supports Vulkan, CUDA, HIP, OpenCL, Level Zero and Metal as backends.

## The white paper of VkFFT is out - if you use VkFFT, you can cite it: https://ieeexplore.ieee.org/document/10036080

## Currently supported features:
  - 1D/2D/3D/ND systems - specify VKFFT_MAX_FFT_DIMENSIONS for arbitrary number of dimensions.
  - Forward and inverse directions of FFT.
  - Support for big FFT dimension sizes. Current limits: approximately 2^32 in all dimensions for all types of transforms. Depends on the amount of shared memory available on the device.
  - Radix-2/3/4/5/7/8/11/13 FFT. Sequences using radix 3, 5, 7, 11 and 13 have comparable performance to that of powers of 2.
  - Rader's FFT algorithm for primes from 17 up to max shared memory length (~10000). Inlined and done without additional memory transfers.
  - Bluestein's FFT algorithm for all other sequences. Optimized to have as few memory transfers as possible by using zero padding and merged convolution support of VkFFT.
  - Single, double, half and quad (double-double) precision support. Double and quad precision uses CPU-generated LUT tables. Half precision still does all computations in single and only uses half precision to store data.
  - All transformations are performed in-place with no performance loss. Out-of-place transforms are supported by selecting different input/output buffers.
  - No additional transposition uploads. Note: Data can be reshuffled after the Four Step FFT algorithm with an additional buffer (for big sequences). Doesn't matter for convolutions - they return to the input ordering (saves memory).
  - Complex to complex (C2C), real to complex (R2C), complex to real (C2R) transformations and real to real (R2R) Discrete Cosine Transformations of types I, II, III and IV. R2R, R2C and C2R are optimized to run up to 2x times faster than C2C and take 2x less memory.
  - 1x1, 2x2, 3x3 convolutions with symmetric or nonsymmetric kernel (no register overutilization).
  - Native zero padding to model open systems (up to 2x faster than simply padding input array with zeros). Can specify the range of sequences filled with zeros and the direction where zero padding is applied (read or write stage).
  - WHD+CN layout - data is stored in the following order (sorted by increase in strides): the width, the height, the depth, other dimensions, the coordinate (the number of feature maps), the batch number.
  - Multiple feature/batch convolutions - one input, multiple kernels.
  - Multiple input/output/temporary buffer split. Allows using data split between different memory allocations and mitigates 4GB single allocation limit.
  - Works on Nvidia, AMD, Intel and Apple GPUs. And Raspberry Pi 4 GPU.
  - Works on Windows, Linux and macOS.
  - VkFFT supports Vulkan, CUDA, HIP, OpenCL, Level Zero and Metal as backend to cover wide range of APIs.
  - Header-only library, which allows appending VkFFT directly to user's command buffer. Kernels are compiled at run-time.
## Future release plan
 - ##### Ambitious
    - Multiple GPU job splitting

## Installation
Vulkan version:
Include the vkFFT.h file and glslang compiler. Provide the library with correctly chosen VKFFT_BACKEND definition (VKFFT_BACKEND=0 for Vulkan). Sample CMakeLists.txt file configures project based on Vulkan_FFT.cpp file, which contains examples on how to use VkFFT to perform FFT, iFFT and convolution calculations, use zero padding, multiple feature/batch convolutions, C2C FFTs of big systems, R2C/C2R transforms, R2R DCT-I, II, III and IV, double precision FFTs, half precision FFTs.\
For single and double precision, Vulkan 1.0 is required. For half precision, Vulkan 1.1 is required.

CUDA/HIP:
Include the vkFFT.h file and make sure your system has NVRTC/HIPRTC built. Provide the library with correctly chosen VKFFT_BACKEND definition.\
To build CUDA/HIP version of the benchmark, replace VKFFT_BACKEND in CMakeLists (line 5) with the correct one and optionally enable FFTW. VKFFT_BACKEND=1 for CUDA, VKFFT_BACKEND=2 for HIP.

OpenCL:
Include the vkFFT.h file. Provide the library with correctly chosen VKFFT_BACKEND definition.\
To build OpenCL version of the benchmark, replace VKFFT_BACKEND in CMakeLists (line 5) with the value 3 and optionally enable FFTW.

Level Zero:
Include the vkFFT.h file. Provide the library with correctly chosen VKFFT_BACKEND definition. Clang and llvm-spirv must be valid system calls.\
To build Level Zero version of the benchmark, replace VKFFT_BACKEND in CMakeLists (line 5) with the value 4 and optionally enable FFTW.

Metal:
Include the vkFFT.h file. Provide the library with correctly chosen VKFFT_BACKEND definition. VkFFT uses metal-cpp as a C++ bindings to Apple's libraries - Foundation.hpp, QuartzCore.hpp and Metal.hpp.\
To build Metal version of the benchmark, replace VKFFT_BACKEND in CMakeLists (line 5) with the value 5 and optionally enable FFTW.

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
.\VkFFT_TestSuite.exe -d 0 -o output.txt -vkfft 0 -cufft 0\
For double precision benchmark, replace -vkfft 0 -cufft 0 with -vkfft 1 -cufft 1. For half precision benchmark, replace -vkfft 0 -cufft 0 with -vkfft 2 -cufft 2.
## How to use VkFFT
VkFFT.h is a library that can append FFT, iFFT or convolution calculation to the user-defined command buffer. It operates on storage buffers allocated by the user and doesn't require any additional memory by itself (except for LUT, if they are enabled). All computations are fully based on Vulkan compute shaders with no CPU usage except for FFT planning. VkFFT creates and optimizes memory layout by itself and performs FFT with the best-chosen parameters. For an example application, see VkFFT_TestSuite.cpp file, which has comments explaining the VkFFT configuration process.\
VkFFT achieves striding by grouping nearby FFTs instead of transpositions. \
Explicit VkFFT documentation can be found in the documentation folder.
## Benchmark results in comparison to cuFFT
The test configuration below takes multiple 1D FFTs of all lengths from the range of 2 to 4096, batch them together so the full system takes from 500MB to 1GB of data and perform multiple consecutive FFTs/iFFTs (-vkfft 1001 key). After that time per a single FFT is obtained by averaging the result.   Total system size will be divided by the time taken by a single transform upload+download, resulting in the estimation of an achieved global bandwidth. The GPUs used in this comparison are Nvidia A100 and AMD MI250. The performance was compared against Nvidia cuFFT (CUDA 11.7 version) and AMD rocFFT (ROCm 5.2 version) libraries in double precision: 
![alt text](https://github.com/DTolm/VkFFT/blob/master/benchmark_plot/fp64_cuda_a100.png?raw=true)
![alt text](https://github.com/DTolm/VkFFT/blob/master/benchmark_plot/fp64_hip_mi250.png?raw=true)
## Precision comparison of cuFFT/VkFFT/FFTW
![alt text](https://github.com/DTolm/VkFFT/blob/master/precision_results/FP64_precision.png?raw=true)
![alt text](https://github.com/DTolm/VkFFT/blob/master/precision_results/FP32_precision.png?raw=true)

Above, VkFFT precision is verified by comparing its results with FP128 version of FFTW. We test all FFT lengths from the [2, 100000] range. We perform tests in single and double precision on random input data from [-1;1] range.

For both precisions, all tested libraries exhibit logarithmic error scaling. The main source of error is imprecise twiddle factor computation – sines and cosines used by FFT algorithms. For FP64 they are calculated on the CPU either in FP128 or in FP64 and stored in the lookup tables. With FP128 precomputation (left) VkFFT is more precise than cuFFT and rocFFT. 

For FP32, twiddle factors can be calculated on-the-fly in FP32 or precomputed in FP64/FP32. With FP32 twiddle factors (right) VkFFT is slightly less precise in Bluestein’s and Rader’s algorithms. If needed, this can be solved with FP64 precomputation. 

## VkFFT - a story of Vulkan Compute GPU HPC library development: https://youtu.be/FQuJJ0m-my0

## VkFFT and beyond – a platform for runtime GPU code generation: https://youtu.be/lHlFPqlOezo

## Check out my poster at SC22: https://sc22.supercomputing.org/presentation/?id=rpost143&sess=sess273

## Check out my panel at Nvidia's GTC 2021 in Higher Education and Research category: https://gtc21.event.nvidia.com/

## Python interface to VkFFT can be found here: https://github.com/vincefn/pyvkfft

## Rust bindings to VkFFT can be found here: https://github.com/semio-ai/vkfft-rs

## Benchmark results of VkFFT can be found here: https://openbenchmarking.org/test/pts/vkfft

## Contact information
The initial version of VkFFT is developed by Tolmachev Dmitrii\
E-mail 1: <dtolm96@gmail.com>
