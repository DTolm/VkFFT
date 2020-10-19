cmake_minimum_required(VERSION 3.11)
project(Vulkan_FFT)
set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)
option(build_VkFFT_cuFFT_benchmark "Build VkFFT cuFFT benchmark" OFF)

if (MSVC)
	set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

add_executable(${PROJECT_NAME} Vulkan_FFT.cpp)

target_compile_definitions(${PROJECT_NAME} PUBLIC -DSHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders/")
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

find_package(Vulkan REQUIRED)
add_library(VkFFT INTERFACE)
target_include_directories(VkFFT INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/lib/)
target_link_libraries(${PROJECT_NAME} PUBLIC Vulkan::Vulkan VkFFT)

if(build_VkFFT_cuFFT_benchmark)
	add_definitions(-DUSE_cuFFT)
	find_package(CUDA 8.0 REQUIRED)
	enable_language(CUDA)
	add_library(benchmark_scripts STATIC
	benchmark_scripts/benchmark_cuFFT.cu
	benchmark_scripts/benchmark_cuFFT_double.cu
	benchmark_scripts/benchmark_cuFFT_half.cu)
	CUDA_ADD_CUFFT_TO_TARGET(benchmark_scripts)
	target_include_directories(benchmark_scripts PUBLIC "benchmark_scripts/")
	set_target_properties(benchmark_scripts PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
	set_target_properties(benchmark_scripts PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
	target_link_libraries(${PROJECT_NAME} PRIVATE benchmark_scripts)
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