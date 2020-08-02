cmake_minimum_required(VERSION 3.11)
project(Vulkan_FFT)
set(CMAKE_CXX_STANDARD 11)
add_definitions(-D_CRT_SECURE_NO_WARNINGS)
add_definitions(-DSHADER_DIR="${PROJECT_SOURCE_DIR}/shaders/")
add_executable(Vulkan_FFT Vulkan_FFT.cpp)
if (MSVC)
	set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT Vulkan_FFT)
endif()
find_package(Vulkan REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${Vulkan_INCLUDE_DIRS})

add_library(VkFFT INTERFACE)
target_include_directories(VkFFT INTERFACE ${PROJECT_SOURCE_DIR}/lib/)

target_link_libraries(${PROJECT_NAME} Vulkan::Vulkan)
target_link_libraries(${PROJECT_NAME} VkFFT)


