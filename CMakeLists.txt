cmake_minimum_required(VERSION 3.11)
project(Vulkan_FFT)

if (MSVC)
	set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT Vulkan_FFT)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

add_executable(Vulkan_FFT Vulkan_FFT.cpp)
target_compile_definitions(Vulkan_FFT PRIVATE -DSHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders/")
target_compile_features(Vulkan_FFT PUBLIC cxx_std_11)

find_package(Vulkan REQUIRED)

add_library(VkFFT INTERFACE)
target_include_directories(VkFFT INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/lib/)

target_link_libraries(Vulkan_FFT
		PRIVATE
			Vulkan::Vulkan
			VkFFT
		)

option(VKFFT_BUILD_SHADERS "Compile/Build VkFFT shaders into SPIRV" OFF)
set(GLSL_VALIDATOR glslangValidator CACHE STRING "Path to glslangValidator, default is 'glslangValidator'")

# expects first argument to be the target name of the shader, used to make new
# custom targets to name avoid conflicts
function(target_shader_postbuild SHADER_TARGET)
	# each additional argument contains the path of a shader source
	foreach (SHADER_SOURCE_FILEPATH ${ARGN})
		get_filename_component(SHADER_SOURCE_FILENAME ${SHADER_SOURCE_FILEPATH} NAME_WE)
		set(SHADER_TARGET_NAME "${SHADER_TARGET}_${SHADER_SOURCE_FILENAME}")
		set(SHADER_BINARY_FILEPATH "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER_SOURCE_FILENAME}.spv")

	#	creating custom target to make sure that library depends on it.
		add_custom_target(${SHADER_TARGET_NAME} DEPENDS ${SHADER_BINARY_FILEPATH})
		add_dependencies(${SHADER_TARGET} ${SHADER_TARGET_NAME})
		add_custom_command(
				COMMAND ${GLSL_VALIDATOR} -V ${SHADER_SOURCE_FILEPATH} -o ${SHADER_BINARY_FILEPATH}

				DEPENDS ${SHADER_SOURCE_FILEPATH}
				OUTPUT ${SHADER_BINARY_FILEPATH}
				COMMENT "Compiling SPIRV for \nsource: \n\t${SHADER_SOURCE_FILEPATH} \nbinary: \n\t${SHADER_BINARY_FILEPATH} \n"
		)
	endforeach (SHADER_SOURCE_FILEPATH)
endfunction()

message("VKFFT_BUILD_SHADERS=${VKFFT_BUILD_SHADERS}")
if(VKFFT_BUILD_SHADERS)
	message("GLSL_VALIDATOR=${GLSL_VALIDATOR}")
	target_shader_postbuild(VkFFT
			${CMAKE_CURRENT_SOURCE_DIR}/shaders/vkFFT_grouped_convolution_symmetric_2x2.comp
			${CMAKE_CURRENT_SOURCE_DIR}/shaders/vkFFT_grouped_c2c.comp
			${CMAKE_CURRENT_SOURCE_DIR}/shaders/vkFFT_single_c2c.comp
			${CMAKE_CURRENT_SOURCE_DIR}/shaders/vkFFT_single_c2c_beforeC2R.comp
			${CMAKE_CURRENT_SOURCE_DIR}/shaders/vkFFT_grouped_convolution_nonsymmetric_2x2.comp
			${CMAKE_CURRENT_SOURCE_DIR}/shaders/vkFFT_grouped_convolution_symmetric_3x3.comp
			${CMAKE_CURRENT_SOURCE_DIR}/shaders/vkFFT_single_c2c_afterR2C.comp
			${CMAKE_CURRENT_SOURCE_DIR}/shaders/vkFFT_single_convolution_1x1.comp
			${CMAKE_CURRENT_SOURCE_DIR}/shaders/vkFFT_single_c2r.comp
			${CMAKE_CURRENT_SOURCE_DIR}/shaders/vkFFT_grouped_convolution_1x1.comp
			${CMAKE_CURRENT_SOURCE_DIR}/shaders/vkFFT_grouped_convolution_nonsymmetric_3x3.comp
			${CMAKE_CURRENT_SOURCE_DIR}/shaders/vkFFT_single_convolution_afterR2C_1x1.comp
			${CMAKE_CURRENT_SOURCE_DIR}/shaders/vkFFT_single_convolution_afterR2C_nonsymmetric_2x2.comp
			${CMAKE_CURRENT_SOURCE_DIR}/shaders/vkFFT_single_convolution_afterR2C_nonsymmetric_3x3.comp
			${CMAKE_CURRENT_SOURCE_DIR}/shaders/vkFFT_single_convolution_afterR2C_symmetric_2x2.comp
			${CMAKE_CURRENT_SOURCE_DIR}/shaders/vkFFT_single_convolution_afterR2C_symmetric_3x3.comp
			${CMAKE_CURRENT_SOURCE_DIR}/shaders/vkFFT_single_convolution_nonsymmetric_2x2.comp
			${CMAKE_CURRENT_SOURCE_DIR}/shaders/vkFFT_single_convolution_nonsymmetric_3x3.comp
			${CMAKE_CURRENT_SOURCE_DIR}/shaders/vkFFT_single_convolution_symmetric_2x2.comp
			${CMAKE_CURRENT_SOURCE_DIR}/shaders/vkFFT_single_convolution_symmetric_3x3.comp
			${CMAKE_CURRENT_SOURCE_DIR}/shaders/vkFFT_single_r2c.comp
			${CMAKE_CURRENT_SOURCE_DIR}/shaders/vkFFT_transpose_inplace.comp
			)
endif(VKFFT_BUILD_SHADERS)