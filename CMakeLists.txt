cmake_minimum_required(VERSION 3.11)
project(Vulkan_FFT)

add_executable(Vulkan_FFT Vulkan_FFT.cpp)
target_compile_definitions(Vulkan_FFT PRIVATE -DSHADER_DIR="${PROJECT_SOURCE_DIR}/shaders/")
target_compile_features(Vulkan_FFT PUBLIC cxx_std_11)
if (MSVC)
	set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT Vulkan_FFT)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()
find_package(Vulkan REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${Vulkan_INCLUDE_DIRS})

add_library(VkFFT INTERFACE)
target_include_directories(VkFFT INTERFACE ${PROJECT_SOURCE_DIR}/lib/)

target_link_libraries(${PROJECT_NAME} Vulkan::Vulkan)
target_link_libraries(${PROJECT_NAME} VkFFT)


