cmake_minimum_required(VERSION 3.11)
project(Vulkan_FFT)

if (MSVC)
	set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

add_executable(${PROJECT_NAME} Vulkan_FFT.cpp)
target_compile_definitions(${PROJECT_NAME} PRIVATE -DSHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders/")
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

find_package(Vulkan REQUIRED)

add_library(VkFFT INTERFACE)
target_include_directories(VkFFT INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/lib/)

target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan VkFFT)


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