cmake_minimum_required(VERSION 3.10)
project(Vulkan_FFT)
set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
set(VKFFT_BACKEND 0)#0 - Vulkan, 1 - CUDA, 2 - HIP
option(build_VkFFT_cuFFT_benchmark "Build VkFFT cuFFT benchmark" OFF)
option(build_VkFFT_FFTW_precision "Build VkFFT FFTW precision comparison" OFF)
if (MSVC)
	set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

add_executable(${PROJECT_NAME} Vulkan_FFT.cpp)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
add_definitions(-DVKFFT_BACKEND=${VKFFT_BACKEND})
if(${VKFFT_BACKEND} EQUAL 0)
	find_package(Vulkan REQUIRED)
elseif(${VKFFT_BACKEND} EQUAL 1)
	find_package(CUDA 11.0 REQUIRED)
	enable_language(CUDA)
	#set_source_files_properties(Vulkan_FFT.cpp PROPERTIES LANGUAGE CUDA)
	set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 35 60 70 75 80 86)
	target_compile_options(${PROJECT_NAME} PUBLIC "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:
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
    	find_library(HIP_hiprtc_LIBRARY hiprtc
      		PATHS
      		ENV ROCM_PATH
       		ENV HIP_PATH
      		/opt/rocm
      		PATH_SUFFIXES lib
       		NO_DEFAULT_PATH)
	add_library(hiprt UNKNOWN IMPORTED)
	set_target_properties(hiprt PROPERTIES
		INTERFACE_LINK_LIBRARIES ${HIP_hiprtc_LIBRARY}
	)
endif()
target_compile_definitions(${PROJECT_NAME} PUBLIC -DVK_API_VERSION=10)#10 - Vulkan 1.0, 11 - Vulkan 1.1, 12 - Vulkan 1.2 
if(${VKFFT_BACKEND} EQUAL 0)
	target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/glslang-master/glslang/Include/)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/glslang-master)
endif()
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vkFFT/)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/half_lib/)

add_library(VkFFT INTERFACE)
target_compile_definitions(VkFFT INTERFACE -DVKFFT_BACKEND=${VKFFT_BACKEND})
add_library(half INTERFACE)
if(${VKFFT_BACKEND} EQUAL 0)
	target_link_libraries(${PROJECT_NAME} PUBLIC SPIRV glslang Vulkan::Vulkan VkFFT half)
elseif(${VKFFT_BACKEND} EQUAL 1)
	find_library(CUDA_NVRTC_LIB libnvrtc nvrtc HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${LIBNVRTC_LIBRARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64" /usr/lib64 /usr/local/cuda/lib64)
	target_link_libraries(${PROJECT_NAME} PUBLIC cuda ${CUDA_NVRTC_LIB} VkFFT half)
elseif(${VKFFT_BACKEND} EQUAL 2)
	target_link_libraries(${PROJECT_NAME} PUBLIC hip::host VkFFT half)
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
	find_package(CUDA 11.0 REQUIRED)
	enable_language(CUDA)	
	if(build_VkFFT_FFTW_precision)
		add_library(cufft_scripts STATIC
		cufft_scripts/benchmark_cuFFT.cu
		cufft_scripts/benchmark_cuFFT_double.cu
		cufft_scripts/benchmark_cuFFT_half.cu
		cufft_scripts/benchmark_cuFFT_3d.cu
		cufft_scripts/precision_cuFFT.cu
		cufft_scripts/precision_cuFFT_double.cu
		cufft_scripts/precision_cuFFT_half.cu)
	else()
		add_library(cufft_scripts STATIC
		cufft_scripts/benchmark_cuFFT.cu
		cufft_scripts/benchmark_cuFFT_double.cu
		cufft_scripts/benchmark_cuFFT_half.cu
		cufft_scripts/benchmark_cuFFT_3d.cu)
	endif()
	set_property(TARGET cufft_scripts PROPERTY CUDA_ARCHITECTURES 35 60 70 75 80 86)
	CUDA_ADD_CUFFT_TO_TARGET(cufft_scripts)
	target_compile_options(cufft_scripts PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:
	-gencode arch=compute_35,code=compute_35
	-gencode arch=compute_60,code=compute_60
	-gencode arch=compute_70,code=compute_70 
	-gencode arch=compute_75,code=compute_75 
	-gencode arch=compute_80,code=compute_80 
	-gencode arch=compute_86,code=compute_86>")
	target_include_directories(cufft_scripts PUBLIC "cufft_scripts/")
	set_target_properties(cufft_scripts PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
	set_target_properties(cufft_scripts PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
	target_link_libraries(${PROJECT_NAME} PUBLIC cufft_scripts)
endif()