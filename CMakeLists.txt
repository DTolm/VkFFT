cmake_minimum_required(VERSION 3.11)
project(Vulkan_FFT)
set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
include(FetchContent)
set(VKFFT_BACKEND 0 CACHE STRING "0 - Vulkan, 1 - CUDA, 2 - HIP, 3 - OpenCL, 4 - Level Zero")

if(${VKFFT_BACKEND} EQUAL 1)
	option(build_VkFFT_cuFFT_benchmark "Build VkFFT cuFFT benchmark" ON)
else()
	option(build_VkFFT_cuFFT_benchmark "Build VkFFT cuFFT benchmark" OFF)
endif()

if(${VKFFT_BACKEND} EQUAL 2)
	option(build_VkFFT_rocFFT_benchmark "Build VkFFT rocFFT benchmark" ON)
else()
	option(build_VkFFT_rocFFT_benchmark "Build VkFFT rocFFT benchmark" OFF)
endif()

option(build_VkFFT_FFTW_precision "Build VkFFT FFTW precision comparison" OFF)
if (MSVC)
	set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()
if(build_VkFFT_FFTW_precision)
	add_executable(${PROJECT_NAME} Vulkan_FFT.cpp
		benchmark_scripts/vkFFT_scripts/src/utils_VkFFT.cpp
		benchmark_scripts/vkFFT_scripts/src/user_benchmark_VkFFT.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_0_benchmark_VkFFT_single.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_1_benchmark_VkFFT_double.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_2_benchmark_VkFFT_half.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_3_benchmark_VkFFT_single_3d.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_4_benchmark_VkFFT_single_3d_zeropadding.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_5_benchmark_VkFFT_single_disableReorderFourStep.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_6_benchmark_VkFFT_single_r2c.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_7_benchmark_VkFFT_single_Bluestein.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_8_benchmark_VkFFT_double_Bluestein.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_10_benchmark_VkFFT_single_multipleBuffers.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_11_precision_VkFFT_single.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_12_precision_VkFFT_double.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_13_precision_VkFFT_half.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_14_precision_VkFFT_single_nonPow2.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_15_precision_VkFFT_single_r2c.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_16_precision_VkFFT_single_dct.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_17_precision_VkFFT_double_dct.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_18_precision_VkFFT_double_nonPow2.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_50_convolution_VkFFT_single_1d_matrix.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_51_convolution_VkFFT_single_3d_matrix_zeropadding_r2c.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_52_convolution_VkFFT_single_2d_batched_r2c.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_100_benchmark_VkFFT_single_nd_dct.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_101_benchmark_VkFFT_double_nd_dct.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_1000_VkFFT_single_2_4096.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_1001_benchmark_VkFFT_double_2_4096.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_1003_benchmark_VkFFT_single_3d_2_512.cpp)
else()		
	add_executable(${PROJECT_NAME} Vulkan_FFT.cpp
		benchmark_scripts/vkFFT_scripts/src/utils_VkFFT.cpp
		benchmark_scripts/vkFFT_scripts/src/user_benchmark_VkFFT.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_0_benchmark_VkFFT_single.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_1_benchmark_VkFFT_double.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_2_benchmark_VkFFT_half.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_3_benchmark_VkFFT_single_3d.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_4_benchmark_VkFFT_single_3d_zeropadding.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_5_benchmark_VkFFT_single_disableReorderFourStep.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_6_benchmark_VkFFT_single_r2c.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_7_benchmark_VkFFT_single_Bluestein.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_8_benchmark_VkFFT_double_Bluestein.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_10_benchmark_VkFFT_single_multipleBuffers.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_50_convolution_VkFFT_single_1d_matrix.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_51_convolution_VkFFT_single_3d_matrix_zeropadding_r2c.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_52_convolution_VkFFT_single_2d_batched_r2c.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_100_benchmark_VkFFT_single_nd_dct.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_101_benchmark_VkFFT_double_nd_dct.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_1000_VkFFT_single_2_4096.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_1001_benchmark_VkFFT_double_2_4096.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_1003_benchmark_VkFFT_single_3d_2_512.cpp)
endif()
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
add_definitions(-DVKFFT_BACKEND=${VKFFT_BACKEND})
if(${VKFFT_BACKEND} EQUAL 0)
	find_package(Vulkan REQUIRED)
elseif(${VKFFT_BACKEND} EQUAL 1)
	find_package(CUDA 9.0 REQUIRED)
	enable_language(CUDA)
if (MSVC)
else()
		set_source_files_properties(Vulkan_FFT.cpp PROPERTIES LANGUAGE CUDA)
endif()
	set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 35 60 70 75 80 86)
	target_compile_options(${PROJECT_NAME} PUBLIC "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:
	-std=c++11
	-DVKFFT_BACKEND=${VKFFT_BACKEND}
	-gencode arch=compute_35,code=compute_35
	-gencode arch=compute_60,code=compute_60
	-gencode arch=compute_70,code=compute_70 
	-gencode arch=compute_75,code=compute_75 
	-gencode arch=compute_80,code=compute_80 
	-gencode arch=compute_86,code=compute_86>")
	set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
	set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
elseif(${VKFFT_BACKEND} EQUAL 2)
	list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
	find_package(hip)
	#target_compile_definitions(${PROJECT_NAME} PUBLIC -DVKFFT_OLD_ROCM) #ROCm versions before 4.5 needed kernel include of hiprtc
elseif(${VKFFT_BACKEND} EQUAL 3)
	find_package(OpenCL REQUIRED)
elseif(${VKFFT_BACKEND} EQUAL 4)
	set(LevelZero_LIBRARY "/usr/lib/x86_64-linux-gnu/")
	set(LevelZero_INCLUDE_DIR "/usr/include/")
	find_library(
		LevelZero_LIB
		NAMES "ze_loader"
		PATHS ${LevelZero_LIBRARY}
		PATH_SUFFIXES "lib" "lib64"
		NO_DEFAULT_PATH
	  )
	find_path(
		LevelZero_INCLUDES
		NAMES "ze_api.h"
		PATHS ${LevelZero_INCLUDE_DIR}
		PATH_SUFFIXES "include" 
		NO_DEFAULT_PATH
	  )
	target_include_directories(${PROJECT_NAME} PUBLIC ${LevelZero_INCLUDES})
endif()

target_compile_definitions(${PROJECT_NAME} PUBLIC -DVK_API_VERSION=11)#10 - Vulkan 1.0, 11 - Vulkan 1.1, 12 - Vulkan 1.2 
if(${VKFFT_BACKEND} EQUAL 0)
	FetchContent_Declare(
		glslang-master
		GIT_REPOSITORY https://github.com/KhronosGroup/glslang
		SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/glslang-master
	)
	FetchContent_GetProperties(glslang-master)
	if(NOT glslang-master_POPULATED)
		FetchContent_Populate(glslang-master)
	endif()
	target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/glslang-master/glslang/Include/)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/glslang-master)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vkFFT/)
add_library(VkFFT INTERFACE)
target_compile_definitions(VkFFT INTERFACE -DVKFFT_BACKEND=${VKFFT_BACKEND})

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/half_lib/)
add_library(half INTERFACE)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_scripts/vkFFT_scripts/include/)

if(${VKFFT_BACKEND} EQUAL 0)
	target_link_libraries(${PROJECT_NAME} PUBLIC SPIRV glslang Vulkan::Vulkan VkFFT half)
elseif(${VKFFT_BACKEND} EQUAL 1)
	find_library(CUDA_NVRTC_LIB libnvrtc nvrtc HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${LIBNVRTC_LIBRARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64" /usr/lib64 /usr/local/cuda/lib64)
	target_link_libraries(${PROJECT_NAME} PUBLIC ${CUDA_LIBRARIES} cuda ${CUDA_NVRTC_LIB} VkFFT half)
elseif(${VKFFT_BACKEND} EQUAL 2)
	target_link_libraries(${PROJECT_NAME} PUBLIC hip::host VkFFT half)
elseif(${VKFFT_BACKEND} EQUAL 3)
	target_link_libraries(${PROJECT_NAME} PUBLIC OpenCL::OpenCL VkFFT half)
elseif(${VKFFT_BACKEND} EQUAL 4)
	target_link_libraries(${PROJECT_NAME} PUBLIC LevelZero VkFFT half)
endif()

if(build_VkFFT_FFTW_precision)
	add_definitions(-DUSE_FFTW)
	set(FFTW3_LIB_DIR "/usr/lib/x86_64-linux-gnu/")
	set(FFTW3_INCLUDE_DIR "/usr/include/")
	find_library(
		FFTW_LIB
		NAMES "libfftw3-3" "fftw3"
		PATHS ${FFTW3_LIB_DIR}
		PATH_SUFFIXES "lib" "lib64"
		NO_DEFAULT_PATH
	  )
	find_path(
		FFTW_INCLUDES
		NAMES "fftw3.h"
		PATHS ${FFTW3_INCLUDE_DIR}
		PATH_SUFFIXES "include"
		NO_DEFAULT_PATH
	  )
	
	target_link_libraries (${PROJECT_NAME} PUBLIC ${FFTW_LIB})
	target_include_directories(${PROJECT_NAME} PUBLIC ${FFTW_INCLUDES})
endif()

if(build_VkFFT_cuFFT_benchmark)
	add_definitions(-DUSE_cuFFT)
	find_package(CUDA 9.0 REQUIRED)
	enable_language(CUDA)	
	if(build_VkFFT_FFTW_precision)
		add_library(cuFFT_scripts STATIC
		benchmark_scripts/cuFFT_scripts/src/user_benchmark_cuFFT.cu
		benchmark_scripts/cuFFT_scripts/src/sample_0_benchmark_cuFFT_single.cu
		benchmark_scripts/cuFFT_scripts/src/sample_1_benchmark_cuFFT_double.cu
		benchmark_scripts/cuFFT_scripts/src/sample_2_benchmark_cuFFT_half.cu
		benchmark_scripts/cuFFT_scripts/src/sample_3_benchmark_cuFFT_single_3d.cu
		benchmark_scripts/cuFFT_scripts/src/sample_6_benchmark_cuFFT_single_r2c.cu
		benchmark_scripts/cuFFT_scripts/src/sample_7_benchmark_cuFFT_single_Bluestein.cu
		benchmark_scripts/cuFFT_scripts/src/sample_8_benchmark_cuFFT_double_Bluestein.cu
		benchmark_scripts/cuFFT_scripts/src/sample_1000_benchmark_cuFFT_single_2_4096.cu
		benchmark_scripts/cuFFT_scripts/src/sample_1001_benchmark_cuFFT_double_2_4096.cu
		benchmark_scripts/cuFFT_scripts/src/sample_1003_benchmark_cuFFT_single_3d_2_512.cu
		benchmark_scripts/cuFFT_scripts/src/precision_cuFFT_single.cu
		benchmark_scripts/cuFFT_scripts/src/precision_cuFFT_r2c.cu
		benchmark_scripts/cuFFT_scripts/src/precision_cuFFT_double.cu
		benchmark_scripts/cuFFT_scripts/src/precision_cuFFT_half.cu)
	else()
		add_library(cuFFT_scripts STATIC
		benchmark_scripts/cuFFT_scripts/src/user_benchmark_cuFFT.cu
		benchmark_scripts/cuFFT_scripts/src/sample_0_benchmark_cuFFT_single.cu
		benchmark_scripts/cuFFT_scripts/src/sample_1_benchmark_cuFFT_double.cu
		benchmark_scripts/cuFFT_scripts/src/sample_2_benchmark_cuFFT_half.cu
		benchmark_scripts/cuFFT_scripts/src/sample_3_benchmark_cuFFT_single_3d.cu
		benchmark_scripts/cuFFT_scripts/src/sample_6_benchmark_cuFFT_single_r2c.cu
		benchmark_scripts/cuFFT_scripts/src/sample_7_benchmark_cuFFT_single_Bluestein.cu
		benchmark_scripts/cuFFT_scripts/src/sample_8_benchmark_cuFFT_double_Bluestein.cu
		benchmark_scripts/cuFFT_scripts/src/sample_1000_benchmark_cuFFT_single_2_4096.cu
		benchmark_scripts/cuFFT_scripts/src/sample_1001_benchmark_cuFFT_double_2_4096.cu
		benchmark_scripts/cuFFT_scripts/src/sample_1003_benchmark_cuFFT_single_3d_2_512.cu)
	endif()
	set_property(TARGET cuFFT_scripts PROPERTY CUDA_ARCHITECTURES 35 60 70 75 80 86)
	CUDA_ADD_CUFFT_TO_TARGET(cuFFT_scripts)
	target_compile_options(cuFFT_scripts PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:
	-std=c++11
	-gencode arch=compute_35,code=compute_35
	-gencode arch=compute_60,code=compute_60
	-gencode arch=compute_70,code=compute_70 
	-gencode arch=compute_75,code=compute_75 
	-gencode arch=compute_80,code=compute_80 
	-gencode arch=compute_86,code=compute_86>")
	target_include_directories(cuFFT_scripts PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_scripts/cuFFT_scripts/include)
	set_target_properties(cuFFT_scripts PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
	set_target_properties(cuFFT_scripts PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
	target_link_libraries(${PROJECT_NAME} PUBLIC cuFFT_scripts)
endif()
if(build_VkFFT_rocFFT_benchmark)
	add_definitions(-DUSE_rocFFT)
	list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
	find_package(hip)
    find_package(hipfft)

	if(build_VkFFT_FFTW_precision)
		add_library(rocFFT_scripts STATIC
		benchmark_scripts/rocFFT_scripts/src/user_benchmark_rocFFT.cpp
		benchmark_scripts/rocFFT_scripts/src/sample_0_benchmark_rocFFT_single.cpp
		benchmark_scripts/rocFFT_scripts/src/sample_1_benchmark_rocFFT_double.cpp
		benchmark_scripts/rocFFT_scripts/src/sample_3_benchmark_rocFFT_single_3d.cpp
		benchmark_scripts/rocFFT_scripts/src/sample_6_benchmark_rocFFT_single_r2c.cpp
		benchmark_scripts/rocFFT_scripts/src/sample_7_benchmark_rocFFT_single_Bluestein.cpp
		benchmark_scripts/rocFFT_scripts/src/sample_8_benchmark_rocFFT_double_Bluestein.cpp
		benchmark_scripts/rocFFT_scripts/src/sample_1000_benchmark_rocFFT_single_2_4096.cpp
		benchmark_scripts/rocFFT_scripts/src/sample_1001_benchmark_rocFFT_double_2_4096.cpp
		benchmark_scripts/rocFFT_scripts/src/sample_1003_benchmark_rocFFT_single_3d_2_512.cpp
		benchmark_scripts/rocFFT_scripts/src/precision_rocFFT_single.cpp
		benchmark_scripts/rocFFT_scripts/src/precision_rocFFT_r2c.cpp
		benchmark_scripts/rocFFT_scripts/src/precision_rocFFT_double.cpp)
	else()
		add_library(rocFFT_scripts STATIC
		benchmark_scripts/rocFFT_scripts/src/user_benchmark_rocFFT.cpp
		benchmark_scripts/rocFFT_scripts/src/sample_0_benchmark_rocFFT_single.cpp
		benchmark_scripts/rocFFT_scripts/src/sample_1_benchmark_rocFFT_double.cpp
		benchmark_scripts/rocFFT_scripts/src/sample_3_benchmark_rocFFT_single_3d.cpp
		benchmark_scripts/rocFFT_scripts/src/sample_6_benchmark_rocFFT_single_r2c.cpp
		benchmark_scripts/rocFFT_scripts/src/sample_7_benchmark_rocFFT_single_Bluestein.cpp
		benchmark_scripts/rocFFT_scripts/src/sample_8_benchmark_rocFFT_double_Bluestein.cpp
		benchmark_scripts/rocFFT_scripts/src/sample_1000_benchmark_rocFFT_single_2_4096.cpp
		benchmark_scripts/rocFFT_scripts/src/sample_1001_benchmark_rocFFT_double_2_4096.cpp
		benchmark_scripts/rocFFT_scripts/src/sample_1003_benchmark_rocFFT_single_3d_2_512.cpp)
	endif()
	target_include_directories(rocFFT_scripts PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_scripts/rocFFT_scripts/include)
	target_link_libraries(rocFFT_scripts PRIVATE hip::host hip::hipfft)
	target_link_libraries(${PROJECT_NAME} PUBLIC rocFFT_scripts)
endif()
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/vkFFT/vkFFT.h" DESTINATION include)
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
