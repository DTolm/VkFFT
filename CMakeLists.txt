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