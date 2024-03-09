cmake_minimum_required(VERSION 3.17)
project(VkFFT_TestSuite LANGUAGES CXX)

########################################################################################################################
# Include CMake modules
########################################################################################################################

include(FetchContent)

########################################################################################################################
# Set up project options
########################################################################################################################

# Set the default build type to Release
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

if (NOT DEFINED GLSLANG_GIT_TAG)
	set(GLSLANG_GIT_TAG "12.3.1")
endif()

# Enable Unity build by default (will work for CXX only, not CUDA or HIP)
set(CMAKE_UNITY_BUILD ON CACHE BOOL "Enable Unity build" FORCE)

# Set the default backend to Vulkan
set(VKFFT_BACKEND 0 CACHE STRING "0 - Vulkan, 1 - CUDA, 2 - HIP, 3 - OpenCL, 4 - Level Zero, 5 - Metal")

# Parse the backend option and set VKFFT_BACKEND_NUM to the corresponding value and define VKFFT_BACKEND_*
if((VKFFT_BACKEND EQUAL 0) OR (VKFFT_BACKEND STREQUAL "Vulkan"))
	set(VKFFT_BACKEND_NUM 0)
	set(VKFFT_BACKEND_VULKAN ON)
elseif((VKFFT_BACKEND EQUAL 1) OR (VKFFT_BACKEND STREQUAL "CUDA"))
	set(VKFFT_BACKEND_NUM 1)
	set(VKFFT_BACKEND_CUDA ON)
elseif((VKFFT_BACKEND EQUAL 2) OR (VKFFT_BACKEND STREQUAL "HIP"))
	set(VKFFT_BACKEND_NUM 2)
	set(VKFFT_BACKEND_HIP ON)
elseif((VKFFT_BACKEND EQUAL 3) OR (VKFFT_BACKEND STREQUAL "OpenCL"))
	set(VKFFT_BACKEND_NUM 3)
	set(VKFFT_BACKEND_OPENCL ON)
elseif((VKFFT_BACKEND EQUAL 4) OR (VKFFT_BACKEND STREQUAL "LevelZero"))
	set(VKFFT_BACKEND_NUM 4)
	set(VKFFT_BACKEND_LEVEL_ZERO ON)
elseif((VKFFT_BACKEND EQUAL 5) OR (VKFFT_BACKEND STREQUAL "Metal"))
	set(VKFFT_BACKEND_NUM 5)
	set(VKFFT_BACKEND_METAL ON)
else()
	message(FATAL_ERROR "Invalid VKFFT_BACKEND value")
endif()

# Enable cuFFT benchmark by default if CUDA is enabled
if(VKFFT_BACKEND_CUDA)
	option(build_VkFFT_cuFFT_benchmark "Build VkFFT cuFFT benchmark" ON)
else()
	option(build_VkFFT_cuFFT_benchmark "Build VkFFT cuFFT benchmark" OFF)
endif()

# Enable rocFFT benchmark by default if HIP is enabled
if(VKFFT_BACKEND_HIP)
	option(build_VkFFT_rocFFT_benchmark "Build VkFFT rocFFT benchmark" ON)
else()
	option(build_VkFFT_rocFFT_benchmark "Build VkFFT rocFFT benchmark" OFF)
endif()

option(build_VkFFT_FFTW_precision "Build VkFFT FFTW precision comparison" OFF)
option(VkFFT_use_FP128_Bluestein_RaderFFT "Use LD for Bluestein and Rader FFT kernel calculations. Currently requires LD FFT library, like FFTWl, will be reworked" OFF)
option(VkFFT_use_FP128_double_double "Build VkFFT quad double-double" OFF)

# Configure MSVC
if(MSVC)
	set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
	add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

########################################################################################################################
# Find and configure dependencies
########################################################################################################################

if(VKFFT_BACKEND_VULKAN)
	find_package(Vulkan REQUIRED)

	add_compile_definitions(VK_API_VERSION=11) # 10 - Vulkan 1.0, 11 - Vulkan 1.1, 12 - Vulkan 1.2

	find_package(glslang CONFIG)
	if(NOT ${glslang_FOUND})
		set(ENABLE_OPT 0)
		FetchContent_Declare(glslang-main GIT_TAG ${GLSLANG_GIT_TAG}
			                                GIT_REPOSITORY https://github.com/KhronosGroup/glslang
			                                SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/glslang-main)
		FetchContent_GetProperties(glslang-main)

		if(NOT glslang-main_POPULATED)
			FetchContent_Populate(glslang-main)
		endif()

		add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/glslang-main)
	endif (NOT ${glslang_FOUND})

elseif(VKFFT_BACKEND_CUDA)
	find_package(CUDAToolkit 9.0 REQUIRED)
	enable_language(CUDA)

	set(CMAKE_CUDA_STANDARD               11)
	set(CMAKE_CUDA_STANDARD_REQUIRED      ON)
	set(CMAKE_CUDA_SEPARABLE_COMPILATION  ON)
	set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)
	set(CMAKE_CUDA_ARCHITECTURES          60 70 75 80 86 90)

elseif(VKFFT_BACKEND_HIP)
	list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)

	find_package(hip)
	enable_language(HIP)

	#add_compile_definitions(VKFFT_OLD_ROCM) #ROCm versions before 4.5 needed kernel include of hiprtc
elseif(VKFFT_BACKEND_OPENCL)
	find_package(OpenCL REQUIRED)

elseif(VKFFT_BACKEND_LEVEL_ZERO)
	set(LevelZero_LIBRARY "/usr/lib/x86_64-linux-gnu/")
	set(LevelZero_INCLUDE_DIR "/usr/include/")

	find_library(LevelZero_LIB NAMES "ze_loader"
		                         PATHS ${LevelZero_LIBRARY}
		                         PATH_SUFFIXES "lib" "lib64"
		                         NO_DEFAULT_PATH)
	find_path(LevelZero_INCLUDES NAMES "ze_api.h"
		                           PATHS ${LevelZero_INCLUDE_DIR}
		                           PATH_SUFFIXES "include" "level_zero"
		                           NO_DEFAULT_PATH)

	include_directories(${LevelZero_INCLUDES})

elseif(VKFFT_BACKEND_METAL)
	find_library(FOUNDATION_LIB Foundation REQUIRED)
	find_library(QUARTZ_CORE_LIB QuartzCore REQUIRED)	
	find_library(METAL_LIB Metal REQUIRED)

	include_directories(${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp)

	add_compile_options(-WMTL_IGNORE_WARNINGS)
endif()

if(build_VkFFT_FFTW_precision OR VkFFT_use_FP128_Bluestein_RaderFFT)
	add_compile_definitions(USE_FFTW)

	set(FFTW3_LIB_DIR "/usr/lib/x86_64-linux-gnu/")
	set(FFTW3_INCLUDE_DIR "/usr/include/")

	find_library(FFTW_LIB NAMES "libfftw3-3" "fftw3" 
		                    PATHS ${FFTW3_LIB_DIR}
		                    PATH_SUFFIXES "lib" "lib64"
		                    NO_DEFAULT_PATH)
	find_path(FFTW_INCLUDES NAMES "fftw3.h"
		                      PATHS ${FFTW3_INCLUDE_DIR}
		                      PATH_SUFFIXES "include"
		                      NO_DEFAULT_PATH)
													
	if(VkFFT_use_FP128_double_double)
		find_library(FFTWQ_LIB NAMES "libfftw3q" "fftw3q"
			                     PATHS ${FFTW3_LIB_DIR}
			                     PATH_SUFFIXES "lib" "lib64"
			                     NO_DEFAULT_PATH)
	endif()

	if(VkFFT_use_FP128_Bluestein_RaderFFT)        
		find_library(FFTWL_LIB NAMES "libfftw3l" "fftw3l"
			                     PATHS ${FFTW3_LIB_DIR}
			                     PATH_SUFFIXES "lib" "lib64"
			                     NO_DEFAULT_PATH)
	endif()
endif()

########################################################################################################################
# Add cuFFT benchmark target
########################################################################################################################

if(build_VkFFT_cuFFT_benchmark)
	add_compile_definitions(USE_cuFFT)

	set(cuFFT_scripts_DIR ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_scripts/cuFFT_scripts)
	set(cuFFT_scripts_SRC ${cuFFT_scripts_DIR}/src/user_benchmark_cuFFT.cu
	                      ${cuFFT_scripts_DIR}/src/sample_0_benchmark_cuFFT_single.cu
	                      ${cuFFT_scripts_DIR}/src/sample_1_benchmark_cuFFT_double.cu
	                      ${cuFFT_scripts_DIR}/src/sample_2_benchmark_cuFFT_half.cu
	                      ${cuFFT_scripts_DIR}/src/sample_3_benchmark_cuFFT_single_3d.cu
	                      ${cuFFT_scripts_DIR}/src/sample_6_benchmark_cuFFT_single_r2c.cu
	                      ${cuFFT_scripts_DIR}/src/sample_7_benchmark_cuFFT_single_Bluestein.cu
	                      ${cuFFT_scripts_DIR}/src/sample_8_benchmark_cuFFT_double_Bluestein.cu
	                      ${cuFFT_scripts_DIR}/src/sample_1000_benchmark_cuFFT_single_2_4096.cu
	                      ${cuFFT_scripts_DIR}/src/sample_1001_benchmark_cuFFT_double_2_4096.cu
	                      ${cuFFT_scripts_DIR}/src/sample_1003_benchmark_cuFFT_single_3d_2_512.cu)

	if(build_VkFFT_FFTW_precision)
		list(APPEND cuFFT_scripts_SRC ${cuFFT_scripts_DIR}/src/precision_cuFFT_single.cu
		                              ${cuFFT_scripts_DIR}/src/precision_cuFFT_r2c.cu
		                              ${cuFFT_scripts_DIR}/src/precision_cuFFT_double.cu
		                              ${cuFFT_scripts_DIR}/src/precision_cuFFT_half.cu)
	endif()

	add_library(cuFFT_scripts STATIC ${cuFFT_scripts_SRC})
	target_compile_features(cuFFT_scripts PUBLIC cxx_std_11)

	target_include_directories(cuFFT_scripts PUBLIC ${cuFFT_scripts_DIR}/include)
	target_include_directories(cuFFT_scripts PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
	target_link_libraries(cuFFT_scripts PRIVATE CUDA::cudart CUDA::cufft)
endif()

########################################################################################################################
# Add rocFFT benchmark target
########################################################################################################################

if(build_VkFFT_rocFFT_benchmark)
	add_compile_definitions(USE_rocFFT)

	list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)

	find_package(hipfft)

	set(rocFFT_scripts_DIR ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_scripts/rocFFT_scripts)
	set(rocFFT_scripts_SRC ${rocFFT_scripts_DIR}/src/user_benchmark_rocFFT.cpp
	                       ${rocFFT_scripts_DIR}/src/sample_0_benchmark_rocFFT_single.cpp
	                       ${rocFFT_scripts_DIR}/src/sample_1_benchmark_rocFFT_double.cpp
	                       ${rocFFT_scripts_DIR}/src/sample_3_benchmark_rocFFT_single_3d.cpp
	                       ${rocFFT_scripts_DIR}/src/sample_6_benchmark_rocFFT_single_r2c.cpp
	                       ${rocFFT_scripts_DIR}/src/sample_7_benchmark_rocFFT_single_Bluestein.cpp
	                       ${rocFFT_scripts_DIR}/src/sample_8_benchmark_rocFFT_double_Bluestein.cpp
	                       ${rocFFT_scripts_DIR}/src/sample_1000_benchmark_rocFFT_single_2_4096.cpp
	                       ${rocFFT_scripts_DIR}/src/sample_1001_benchmark_rocFFT_double_2_4096.cpp
	                       ${rocFFT_scripts_DIR}/src/sample_1003_benchmark_rocFFT_single_3d_2_512.cpp)

	if(build_VkFFT_FFTW_precision)
		list(APPEND rocFFT_scripts_SRC ${rocFFT_scripts_DIR}/src/precision_rocFFT_single.cpp
		                               ${rocFFT_scripts_DIR}/src/precision_rocFFT_r2c.cpp
		                               ${rocFFT_scripts_DIR}/src/precision_rocFFT_double.cpp)
	endif()

	add_library(rocFFT_scripts STATIC ${rocFFT_scripts_SRC})
	target_compile_features(rocFFT_scripts PUBLIC cxx_std_11)

	target_include_directories(rocFFT_scripts PUBLIC ${rocFFT_scripts_DIR}/include)
	target_link_libraries(rocFFT_scripts PRIVATE hip::host hip::hipfft)
endif()

########################################################################################################################
# Add VkFFT library target
########################################################################################################################

add_library(VkFFT INTERFACE)
target_include_directories(VkFFT INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/vkFFT/)
target_compile_definitions(VkFFT INTERFACE VKFFT_BACKEND=${VKFFT_BACKEND_NUM})

########################################################################################################################
# Add half library target
########################################################################################################################

add_library(half INTERFACE)
target_include_directories(half INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/half_lib)

########################################################################################################################
# Add VkFFT_TestSuite target
########################################################################################################################

set(${PROJECT_NAME}_DIR ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_scripts/vkFFT_scripts)
set(${PROJECT_NAME}_SRC ${CMAKE_CURRENT_SOURCE_DIR}/VkFFT_TestSuite.cpp
                        ${${PROJECT_NAME}_DIR}/src/utils_VkFFT.cpp
                        ${${PROJECT_NAME}_DIR}/src/user_benchmark_VkFFT.cpp
                        ${${PROJECT_NAME}_DIR}/src/sample_0_benchmark_VkFFT_single.cpp
                        ${${PROJECT_NAME}_DIR}/src/sample_1_benchmark_VkFFT_double.cpp
                        ${${PROJECT_NAME}_DIR}/src/sample_2_benchmark_VkFFT_half.cpp
                        ${${PROJECT_NAME}_DIR}/src/sample_3_benchmark_VkFFT_single_3d.cpp
                        ${${PROJECT_NAME}_DIR}/src/sample_4_benchmark_VkFFT_single_3d_zeropadding.cpp
                        ${${PROJECT_NAME}_DIR}/src/sample_5_benchmark_VkFFT_single_disableReorderFourStep.cpp
                        ${${PROJECT_NAME}_DIR}/src/sample_6_benchmark_VkFFT_single_r2c.cpp
                        ${${PROJECT_NAME}_DIR}/src/sample_7_benchmark_VkFFT_single_Bluestein.cpp
                        ${${PROJECT_NAME}_DIR}/src/sample_8_benchmark_VkFFT_double_Bluestein.cpp
                        ${${PROJECT_NAME}_DIR}/src/sample_9_benchmark_VkFFT_quadDoubleDouble.cpp
                        ${${PROJECT_NAME}_DIR}/src/sample_10_benchmark_VkFFT_single_multipleBuffers.cpp
                        ${${PROJECT_NAME}_DIR}/src/sample_50_convolution_VkFFT_single_1d_matrix.cpp
                        ${${PROJECT_NAME}_DIR}/src/sample_51_convolution_VkFFT_single_3d_matrix_zeropadding_r2c.cpp
                        ${${PROJECT_NAME}_DIR}/src/sample_52_convolution_VkFFT_single_2d_batched_r2c.cpp
                        ${${PROJECT_NAME}_DIR}/src/sample_53_convolution_VkFFT_single_2d_Nimages_1kernel.cpp
                        ${${PROJECT_NAME}_DIR}/src/sample_100_benchmark_VkFFT_single_nd_dct.cpp
                        ${${PROJECT_NAME}_DIR}/src/sample_101_benchmark_VkFFT_double_nd_dct.cpp
                        ${${PROJECT_NAME}_DIR}/src/sample_1000_benchmark_VkFFT_single_2_4096.cpp
                        ${${PROJECT_NAME}_DIR}/src/sample_1001_benchmark_VkFFT_double_2_4096.cpp
                        ${${PROJECT_NAME}_DIR}/src/sample_1002_benchmark_VkFFT_half_2_4096.cpp
                        ${${PROJECT_NAME}_DIR}/src/sample_1003_benchmark_VkFFT_single_3d_2_512.cpp
                        ${${PROJECT_NAME}_DIR}/src/sample_1004_benchmark_VkFFT_quadDoubleDouble_2_4096.cpp)

if(build_VkFFT_FFTW_precision)
	list(APPEND ${PROJECT_NAME}_SRC ${${PROJECT_NAME}_DIR}/src/sample_11_precision_VkFFT_single.cpp
		                              ${${PROJECT_NAME}_DIR}/src/sample_12_precision_VkFFT_double.cpp
		                              ${${PROJECT_NAME}_DIR}/src/sample_13_precision_VkFFT_half.cpp
		                              ${${PROJECT_NAME}_DIR}/src/sample_14_precision_VkFFT_single_nonPow2.cpp
		                              ${${PROJECT_NAME}_DIR}/src/sample_15_precision_VkFFT_single_r2c.cpp
		                              ${${PROJECT_NAME}_DIR}/src/sample_16_precision_VkFFT_single_dct.cpp
		                              ${${PROJECT_NAME}_DIR}/src/sample_17_precision_VkFFT_double_dct.cpp
		                              ${${PROJECT_NAME}_DIR}/src/sample_18_precision_VkFFT_double_nonPow2.cpp
		                              ${${PROJECT_NAME}_DIR}/src/sample_19_precision_VkFFT_quadDoubleDouble_nonPow2.cpp)
endif()

add_executable(${PROJECT_NAME} ${${PROJECT_NAME}_SRC})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
add_compile_definitions(VKFFT_BACKEND=${VKFFT_BACKEND_NUM})
target_include_directories(${PROJECT_NAME} PUBLIC ${${PROJECT_NAME}_DIR}/include)

if(VkFFT_use_FP128_Bluestein_RaderFFT)
  target_compile_definitions(${PROJECT_NAME} PUBLIC VkFFT_use_FP128_Bluestein_RaderFFT)
endif()

if(VkFFT_use_FP128_double_double)
	target_compile_definitions(${PROJECT_NAME} PUBLIC VKFFT_USE_DOUBLEDOUBLE_FP128)
	target_link_libraries(${PROJECT_NAME} PUBLIC quadmath)
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC VkFFT half)

# Link the correct backend libraries
if(VKFFT_BACKEND_VULKAN)
	if (${glslang_FOUND})
		target_link_libraries(${PROJECT_NAME} PUBLIC glslang::SPIRV glslang::glslang Vulkan::Vulkan)
	else (${glslang_FOUND})
		target_link_libraries(${PROJECT_NAME} PUBLIC SPIRV glslang Vulkan::Vulkan)
	endif (${glslang_FOUND})
elseif(VKFFT_BACKEND_CUDA)
	target_compile_definitions(${PROJECT_NAME} PUBLIC CUDA_TOOLKIT_ROOT_DIR="${CUDAToolkit_ROOT}")
	target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart CUDA::cuda_driver CUDA::nvrtc)
	target_include_directories(${PROJECT_NAME} PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
elseif(VKFFT_BACKEND_HIP)
	target_link_libraries(${PROJECT_NAME} PUBLIC hip::host)
elseif(VKFFT_BACKEND_OPENCL)
	target_link_libraries(${PROJECT_NAME} PUBLIC OpenCL::OpenCL)
elseif(VKFFT_BACKEND_LEVEL_ZERO)
	target_link_libraries(${PROJECT_NAME} PUBLIC ze_loader)
elseif(VKFFT_BACKEND_METAL)
	target_link_libraries(${PROJECT_NAME} PUBLIC ${FOUNDATION_LIB} ${QUARTZ_CORE_LIB} ${METAL_LIB})
endif()

# Link FFTW libraries if needed
if(build_VkFFT_FFTW_precision OR VkFFT_use_FP128_Bluestein_RaderFFT)
	target_include_directories(${PROJECT_NAME} PUBLIC ${FFTW_INCLUDES})
	target_link_libraries(${PROJECT_NAME} PUBLIC ${FFTW_LIB})
		
	if (VkFFT_use_FP128_double_double)
		target_link_libraries(${PROJECT_NAME} PUBLIC ${FFTWQ_LIB})
	endif()

	if(VkFFT_use_FP128_Bluestein_RaderFFT)        
		target_link_libraries(${PROJECT_NAME} PUBLIC ${FFTWL_LIB})
	endif()
endif()

# Link cuFFT benchmark library if needed
if(build_VkFFT_cuFFT_benchmark)
	target_link_libraries(${PROJECT_NAME} PUBLIC cuFFT_scripts)
endif()

# Link rocFFT benchmark library if needed
if(build_VkFFT_rocFFT_benchmark)
	target_link_libraries(${PROJECT_NAME} PUBLIC rocFFT_scripts)
endif()

########################################################################################################################
# Install targets
########################################################################################################################

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/vkFFT/" DESTINATION include)
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
