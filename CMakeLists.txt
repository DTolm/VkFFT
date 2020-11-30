cmake_minimum_required(VERSION 3.11)
project(Vulkan_FFT)
set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)
option(build_VkFFT_cuFFT_benchmark "Build VkFFT cuFFT benchmark" OFF)
option(build_VkFFT_FFTW_precision "Build VkFFT FFTW precision comparison" OFF)
if (MSVC)
	set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

add_executable(${PROJECT_NAME} Vulkan_FFT.cpp)

target_compile_definitions(${PROJECT_NAME} PUBLIC -DSHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vkFFT/shaders/")
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

find_package(Vulkan REQUIRED)
add_library(VkFFT INTERFACE)
target_include_directories(VkFFT INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/vkFFT/lib/)

add_library(Half INTERFACE)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Half/)
target_link_libraries(${PROJECT_NAME} PUBLIC Vulkan::Vulkan VkFFT Half)

if(build_VkFFT_FFTW_precision)
	add_definitions(-DUSE_FFTW)
	set(FFTW3_LIB_DIR "")
	set(FFTW3_INCLUDE_DIR "")
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


#Build shaders routine

find_program(
	GLSL_VALIDATOR
	glslangValidator
	DOC "Vulkan Shader Compiler (glslangValidator)"
	REQUIRED
)

file(GLOB_RECURSE COMP_SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.comp"
    )

foreach(INPUT_SHADER ${COMP_SOURCE_FILES})
	get_filename_component(DIR ${INPUT_SHADER} DIRECTORY)
	get_filename_component(FILE_NAME ${INPUT_SHADER} NAME_WE)
	set(OUTPUT_BINARY "${DIR}/${FILE_NAME}.spv")

	add_custom_command(
		OUTPUT ${OUTPUT_BINARY}
		COMMAND ${GLSL_VALIDATOR} -V ${INPUT_SHADER} -o ${OUTPUT_BINARY}
		DEPENDS ${INPUT_SHADER})
	list(APPEND SPIRV_BINARY_FILES ${OUTPUT_BINARY})
endforeach(INPUT_SHADER)

add_custom_target(
    compile_shaders
    DEPENDS ${SPIRV_BINARY_FILES}
    )
add_dependencies(VkFFT compile_shaders)