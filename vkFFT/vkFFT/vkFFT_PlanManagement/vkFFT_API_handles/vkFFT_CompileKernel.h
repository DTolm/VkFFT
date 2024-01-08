// This file is part of VkFFT
//
// Copyright (C) 2021 - present Dmitrii Tolmachev <dtolm96@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
#ifndef VKFFT_COMPILEKERNEL_H
#define VKFFT_COMPILEKERNEL_H
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"

static inline VkFFTResult VkFFT_CompileKernel(VkFFTApplication* app, VkFFTAxis* axis) {
#if(VKFFT_BACKEND==0)
	VkResult res = VK_SUCCESS;
#elif(VKFFT_BACKEND==1)
	cudaError_t res = cudaSuccess;
#elif(VKFFT_BACKEND==2)
	hipError_t res = hipSuccess;
#elif(VKFFT_BACKEND==3)
	cl_int res = CL_SUCCESS;
#elif(VKFFT_BACKEND==4)
	ze_result_t res = ZE_RESULT_SUCCESS;
#elif(VKFFT_BACKEND==5)
#endif
	char* code0 = axis->specializationConstants.code0;
#if(VKFFT_BACKEND==0)
	uint32_t* code;
	pfUINT codeSize;
	if (app->configuration.loadApplicationFromString) {
		char* localStrPointer = (char*)app->configuration.loadApplicationString + app->currentApplicationStringPos;
		memcpy(&codeSize, localStrPointer, sizeof(pfUINT));
		code = (uint32_t*)malloc(codeSize);
		if (!code) {
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
		memcpy(code, localStrPointer + sizeof(pfUINT), codeSize);
		app->currentApplicationStringPos += codeSize + sizeof(pfUINT);
	}
	else
	{
		glslang_resource_t default_resource = VKFFT_ZERO_INIT;
		default_resource.max_lights = 32;
		default_resource.max_clip_planes = 6;
		default_resource.max_texture_units = 32;
		default_resource.max_texture_coords = 32;
		default_resource.max_vertex_attribs = 64;
		default_resource.max_vertex_uniform_components = 4096;
		default_resource.max_varying_floats = 64;
		default_resource.max_vertex_texture_image_units = 32;
		default_resource.max_combined_texture_image_units = 80;
		default_resource.max_texture_image_units = 32;
		default_resource.max_fragment_uniform_components = 4096;
		default_resource.max_draw_buffers = 32;
		default_resource.max_vertex_uniform_vectors = 128;
		default_resource.max_varying_vectors = 8;
		default_resource.max_fragment_uniform_vectors = 16;
		default_resource.max_vertex_output_vectors = 16;
		default_resource.max_fragment_input_vectors = 15;
		default_resource.min_program_texel_offset = -8;
		default_resource.max_program_texel_offset = 7;
		default_resource.max_clip_distances = 8;
		default_resource.max_compute_work_group_count_x = (int)app->configuration.maxComputeWorkGroupCount[0];
		default_resource.max_compute_work_group_count_y = (int)app->configuration.maxComputeWorkGroupCount[1];
		default_resource.max_compute_work_group_count_z = (int)app->configuration.maxComputeWorkGroupCount[2];
		default_resource.max_compute_work_group_size_x = (int)app->configuration.maxComputeWorkGroupSize[0];
		default_resource.max_compute_work_group_size_y = (int)app->configuration.maxComputeWorkGroupSize[1];
		default_resource.max_compute_work_group_size_z = (int)app->configuration.maxComputeWorkGroupSize[2];
		default_resource.max_compute_uniform_components = 1024;
		default_resource.max_compute_texture_image_units = 16;
		default_resource.max_compute_image_uniforms = 8;
		default_resource.max_compute_atomic_counters = 8;
		default_resource.max_compute_atomic_counter_buffers = 1;
		default_resource.max_varying_components = 60;
		default_resource.max_vertex_output_components = 64;
		default_resource.max_geometry_input_components = 64;
		default_resource.max_geometry_output_components = 128;
		default_resource.max_fragment_input_components = 128;
		default_resource.max_image_units = 8;
		default_resource.max_combined_image_units_and_fragment_outputs = 8;
		default_resource.max_combined_shader_output_resources = 8;
		default_resource.max_image_samples = 0;
		default_resource.max_vertex_image_uniforms = 0;
		default_resource.max_tess_control_image_uniforms = 0;
		default_resource.max_tess_evaluation_image_uniforms = 0;
		default_resource.max_geometry_image_uniforms = 0;
		default_resource.max_fragment_image_uniforms = 8;
		default_resource.max_combined_image_uniforms = 8;
		default_resource.max_geometry_texture_image_units = 16;
		default_resource.max_geometry_output_vertices = 256;
		default_resource.max_geometry_total_output_components = 1024;
		default_resource.max_geometry_uniform_components = 1024;
		default_resource.max_geometry_varying_components = 64;
		default_resource.max_tess_control_input_components = 128;
		default_resource.max_tess_control_output_components = 128;
		default_resource.max_tess_control_texture_image_units = 16;
		default_resource.max_tess_control_uniform_components = 1024;
		default_resource.max_tess_control_total_output_components = 4096;
		default_resource.max_tess_evaluation_input_components = 128;
		default_resource.max_tess_evaluation_output_components = 128;
		default_resource.max_tess_evaluation_texture_image_units = 16;
		default_resource.max_tess_evaluation_uniform_components = 1024;
		default_resource.max_tess_patch_components = 120;
		default_resource.max_patch_vertices = 32;
		default_resource.max_tess_gen_level = 64;
		default_resource.max_viewports = 16;
		default_resource.max_vertex_atomic_counters = 0;
		default_resource.max_tess_control_atomic_counters = 0;
		default_resource.max_tess_evaluation_atomic_counters = 0;
		default_resource.max_geometry_atomic_counters = 0;
		default_resource.max_fragment_atomic_counters = 8;
		default_resource.max_combined_atomic_counters = 8;
		default_resource.max_atomic_counter_bindings = 1;
		default_resource.max_vertex_atomic_counter_buffers = 0;
		default_resource.max_tess_control_atomic_counter_buffers = 0;
		default_resource.max_tess_evaluation_atomic_counter_buffers = 0;
		default_resource.max_geometry_atomic_counter_buffers = 0;
		default_resource.max_fragment_atomic_counter_buffers = 1;
		default_resource.max_combined_atomic_counter_buffers = 1;
		default_resource.max_atomic_counter_buffer_size = 16384;
		default_resource.max_transform_feedback_buffers = 4;
		default_resource.max_transform_feedback_interleaved_components = 64;
		default_resource.max_cull_distances = 8;
		default_resource.max_combined_clip_and_cull_distances = 8;
		default_resource.max_samples = 4;
		default_resource.max_mesh_output_vertices_nv = 256;
		default_resource.max_mesh_output_primitives_nv = 512;
		default_resource.max_mesh_work_group_size_x_nv = 32;
		default_resource.max_mesh_work_group_size_y_nv = 1;
		default_resource.max_mesh_work_group_size_z_nv = 1;
		default_resource.max_task_work_group_size_x_nv = 32;
		default_resource.max_task_work_group_size_y_nv = 1;
		default_resource.max_task_work_group_size_z_nv = 1;
		default_resource.max_mesh_view_count_nv = 4;

		default_resource.limits.non_inductive_for_loops = 1;
		default_resource.limits.while_loops = 1;
		default_resource.limits.do_while_loops = 1;
		default_resource.limits.general_uniform_indexing = 1;
		default_resource.limits.general_attribute_matrix_vector_indexing = 1;
		default_resource.limits.general_varying_indexing = 1;
		default_resource.limits.general_sampler_indexing = 1;
		default_resource.limits.general_variable_indexing = 1;
		default_resource.limits.general_constant_matrix_vector_indexing = 1;

		glslang_target_client_version_t client_version = (app->configuration.halfPrecision) ? GLSLANG_TARGET_VULKAN_1_1 : GLSLANG_TARGET_VULKAN_1_0;
		glslang_target_language_version_t target_language_version = (app->configuration.halfPrecision) ? GLSLANG_TARGET_SPV_1_3 : GLSLANG_TARGET_SPV_1_0;
		glslang_input_t input =
		{
			GLSLANG_SOURCE_GLSL,
			GLSLANG_STAGE_COMPUTE,
			GLSLANG_CLIENT_VULKAN,
			client_version,
			GLSLANG_TARGET_SPV,
			target_language_version,
			code0,
			450,
			GLSLANG_NO_PROFILE,
			1,
			0,
			GLSLANG_MSG_DEFAULT_BIT,
			(const glslang_resource_t*)&default_resource,
		};
		//printf("%s\n", code0);
		glslang_shader_t* shader = glslang_shader_create((const glslang_input_t*)&input);
		const char* err;
		if (!glslang_shader_preprocess(shader, &input))
		{
			err = glslang_shader_get_info_log(shader);
			printf("%s\n", code0);
			printf("%s\n", err);
			glslang_shader_delete(shader);
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_SHADER_PREPROCESS;

		}

		if (!glslang_shader_parse(shader, &input))
		{
			err = glslang_shader_get_info_log(shader);
			printf("%s\n", code0);
			printf("%s\n", err);
			glslang_shader_delete(shader);
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_SHADER_PARSE;

		}
		glslang_program_t* program = glslang_program_create();
		glslang_program_add_shader(program, shader);
		if (!glslang_program_link(program, GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT))
		{
			err = glslang_program_get_info_log(program);
			printf("%s\n", code0);
			printf("%s\n", err);
			glslang_shader_delete(shader);
			glslang_program_delete(program);
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_SHADER_LINK;

		}

		glslang_program_SPIRV_generate(program, input.stage);

		if (glslang_program_SPIRV_get_messages(program))
		{
			printf("%s", glslang_program_SPIRV_get_messages(program));
			glslang_shader_delete(shader);
			glslang_program_delete(program);
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_SPIRV_GENERATE;
		}

		glslang_shader_delete(shader);
		uint32_t* tempCode = glslang_program_SPIRV_get_ptr(program);
		codeSize = glslang_program_SPIRV_get_size(program) * sizeof(uint32_t);
		axis->binarySize = codeSize;
		code = (uint32_t*)malloc(codeSize);
		if (!code) {
			free(code0);
			code0 = 0;
			glslang_program_delete(program);
			deleteVkFFT(app);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
		axis->binary = code;
		memcpy(code, tempCode, codeSize);
		glslang_program_delete(program);
	}
	VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
	VkComputePipelineCreateInfo computePipelineCreateInfo = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
	pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	VkShaderModuleCreateInfo createInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
	createInfo.pCode = code;
	createInfo.codeSize = codeSize;
	res = vkCreateShaderModule(app->configuration.device[0], &createInfo, 0, &pipelineShaderStageCreateInfo.module);
	if (res != VK_SUCCESS) {
		free(code0);
		code0 = 0;
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_CREATE_SHADER_MODULE;
	}
	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
	pipelineLayoutCreateInfo.setLayoutCount = 1;
	pipelineLayoutCreateInfo.pSetLayouts = &axis->descriptorSetLayout;
	VkPushConstantRange pushConstantRange = { VK_SHADER_STAGE_COMPUTE_BIT };
	pushConstantRange.offset = 0;
	pushConstantRange.size = (uint32_t)axis->pushConstants.structSize;
	// Push constant ranges are part of the pipeline layout
	if (axis->pushConstants.structSize) {
		pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
		pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
	}
	res = vkCreatePipelineLayout(app->configuration.device[0], &pipelineLayoutCreateInfo, 0, &axis->pipelineLayout);
	if (res != VK_SUCCESS) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_CREATE_PIPELINE_LAYOUT;
	}
	pipelineShaderStageCreateInfo.pName = "main";
	pipelineShaderStageCreateInfo.pSpecializationInfo = 0;// &specializationInfo;
	computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
	computePipelineCreateInfo.layout = axis->pipelineLayout;
	if (app->configuration.pipelineCache)
		res = vkCreateComputePipelines(app->configuration.device[0], app->configuration.pipelineCache[0], 1, &computePipelineCreateInfo, 0, &axis->pipeline);
	else
		res = vkCreateComputePipelines(app->configuration.device[0], 0, 1, &computePipelineCreateInfo, 0, &axis->pipeline);
	if (res != VK_SUCCESS) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_CREATE_PIPELINE;
	}
	vkDestroyShaderModule(app->configuration.device[0], pipelineShaderStageCreateInfo.module, 0);
	if (!app->configuration.saveApplicationToString) {
		free(code);
		code = 0;
	}
#elif(VKFFT_BACKEND==1)
	char* code;
	pfUINT codeSize;
	if (app->configuration.loadApplicationFromString) {
		char* localStrPointer = (char*)app->configuration.loadApplicationString + app->currentApplicationStringPos;
		memcpy(&codeSize, localStrPointer, sizeof(pfUINT));
		code = (char*)malloc(codeSize);
		if (!code) {
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
		memcpy(code, localStrPointer + sizeof(pfUINT), codeSize);
		app->currentApplicationStringPos += codeSize + sizeof(pfUINT);
	}
	else {
		nvrtcProgram prog;
		nvrtcResult result = nvrtcCreateProgram(&prog,         // prog
			code0,         // buffer
			"VkFFT.cu",    // name
			0,             // numHeaders
			0,          // headers
			0);        // includeNames
		//free(includeNames);
		//free(headers);
		if (result != NVRTC_SUCCESS) {
			printf("nvrtcCreateProgram error: %s\n", nvrtcGetErrorString(result));
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_TO_CREATE_PROGRAM;
		}
		int numOpts = 1;
		char* opts[5];
		opts[0] = (char*)malloc(sizeof(char) * 50);
		if (!opts[0]) {
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
#if (CUDA_VERSION >= 11030)
		sprintf(opts[0], "--gpu-architecture=sm_%" PRIu64 "%" PRIu64 "", app->configuration.computeCapabilityMajor, app->configuration.computeCapabilityMinor);
#else
		sprintf(opts[0], "--gpu-architecture=compute_%" PRIu64 "%" PRIu64 "", app->configuration.computeCapabilityMajor, app->configuration.computeCapabilityMinor);
#endif
		if (app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory){
			opts[1] = (char*)malloc(sizeof(char) * 50);
			if (!opts[1]) {
				free(code0);
				code0 = 0;
				deleteVkFFT(app);
				return VKFFT_ERROR_MALLOC_FAILED;
			}
			numOpts++;
			sprintf(opts[1], "-fmad=false");
		}
		//result = nvrtcAddNameExpression(prog, "&consts");
		//if (result != NVRTC_SUCCESS) printf("1.5 error: %s\n", nvrtcGetErrorString(result));
		result = nvrtcCompileProgram(prog,  // prog
			numOpts,     // numOptions
			(const char* const*)opts); // options

		free(opts[0]);
		if (app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory)
			free(opts[1]);

		if (result != NVRTC_SUCCESS) {
			printf("nvrtcCompileProgram error: %s\n", nvrtcGetErrorString(result));
			char* log = (char*)malloc(sizeof(char) * 4000000);
			if (!log) {
				free(code0);
				code0 = 0;
				deleteVkFFT(app);
				return VKFFT_ERROR_FAILED_TO_COMPILE_PROGRAM;
			}
			else {
				nvrtcGetProgramLog(prog, log);
				printf("%s\n", log);
				free(log);
				log = 0;
				printf("%s\n", code0);
				free(code0);
				code0 = 0;
				deleteVkFFT(app);
				return VKFFT_ERROR_FAILED_TO_COMPILE_PROGRAM;
			}
		}
#if (CUDA_VERSION >= 11030)
		result = nvrtcGetCUBINSize(prog, &codeSize);
#else
		result = nvrtcGetPTXSize(prog, &codeSize);
#endif
		if (result != NVRTC_SUCCESS) {
#if (CUDA_VERSION >= 11030)
			printf("nvrtcGetCUBINSize error: %s\n", nvrtcGetErrorString(result));
#else
			printf("nvrtcGetPTXSize error: %s\n", nvrtcGetErrorString(result));
#endif
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_TO_GET_CODE_SIZE;
		}
		axis->binarySize = codeSize;
		code = (char*)malloc(codeSize);
		if (!code) {
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
		axis->binary = code;
#if (CUDA_VERSION >= 11030)
		result = nvrtcGetCUBIN(prog, code);
#else
		result = nvrtcGetPTX(prog, code);
#endif
		if (result != NVRTC_SUCCESS) {
#if (CUDA_VERSION >= 11030)
			printf("nvrtcGetCUBIN error: %s\n", nvrtcGetErrorString(result));
#else
			printf("nvrtcGetPTX error: %s\n", nvrtcGetErrorString(result));
#endif
			free(code);
			code = 0;
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_TO_GET_CODE;
		}
		result = nvrtcDestroyProgram(&prog);
		if (result != NVRTC_SUCCESS) {
			printf("nvrtcDestroyProgram error: %s\n", nvrtcGetErrorString(result));
			free(code);
			code = 0;
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_TO_DESTROY_PROGRAM;
		}
	}
	CUresult result2 = cuModuleLoadDataEx(&axis->VkFFTModule, code, 0, 0, 0);

	if (result2 != CUDA_SUCCESS) {
		printf("cuModuleLoadDataEx error: %d\n", result2);
		free(code);
		code = 0;
		free(code0);
		code0 = 0;
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_LOAD_MODULE;
	}
	result2 = cuModuleGetFunction(&axis->VkFFTKernel, axis->VkFFTModule, axis->VkFFTFunctionName);
	if (result2 != CUDA_SUCCESS) {
		printf("cuModuleGetFunction error: %d\n", result2);
		free(code);
		code = 0;
		free(code0);
		code0 = 0;
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_GET_FUNCTION;
	}
	if ((pfUINT)axis->specializationConstants.usedSharedMemory.data.i > app->configuration.sharedMemorySizeStatic) {
		result2 = cuFuncSetAttribute(axis->VkFFTKernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)axis->specializationConstants.usedSharedMemory.data.i);
		if (result2 != CUDA_SUCCESS) {
			printf("cuFuncSetAttribute error: %d\n", result2);
			free(code);
			code = 0;
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_TO_SET_DYNAMIC_SHARED_MEMORY;
		}
	}
	/*if (axis->pushConstants.structSize) {
		size_t size = axis->pushConstants.structSize;
		result2 = cuModuleGetGlobal(&axis->consts_addr, &size, axis->VkFFTModule, "consts");
		if (result2 != CUDA_SUCCESS) {
			printf("cuModuleGetGlobal error: %d\n", result2);
			free(code);
			code = 0;
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_TO_MODULE_GET_GLOBAL;
		}
	}*/
	if (!app->configuration.saveApplicationToString) {
		free(code);
		code = 0;
	}
#elif(VKFFT_BACKEND==2)
	uint32_t* code;
	pfUINT codeSize;
	if (app->configuration.loadApplicationFromString) {
		char* localStrPointer = (char*)app->configuration.loadApplicationString + app->currentApplicationStringPos;
		memcpy(&codeSize, localStrPointer, sizeof(pfUINT));
		code = (uint32_t*)malloc(codeSize);
		if (!code) {
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
		memcpy(code, localStrPointer + sizeof(pfUINT), codeSize);
		app->currentApplicationStringPos += codeSize + sizeof(pfUINT);
	}
	else
	{
		hiprtcProgram prog;
		enum hiprtcResult result = hiprtcCreateProgram(&prog,         // prog
			code0,         // buffer
			"VkFFT.hip",    // name
			0,             // numHeaders
			0,          // headers
			0);        // includeNames
		if (result != HIPRTC_SUCCESS) {
			printf("hiprtcCreateProgram error: %s\n", hiprtcGetErrorString(result));
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_TO_CREATE_PROGRAM;
		}
		/*if (axis->pushConstants.structSize) {
			result = hiprtcAddNameExpression(prog, "&consts");
			if (result != HIPRTC_SUCCESS) {
				printf("hiprtcAddNameExpression error: %s\n", hiprtcGetErrorString(result));
				free(code0);
				code0 = 0;
				deleteVkFFT(app);
				return VKFFT_ERROR_FAILED_TO_ADD_NAME_EXPRESSION;
			}
		}*/
		int numOpts = 0;
		char* opts[5];
		if (app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory){
			opts[0] = (char*)malloc(sizeof(char) * 50);
			if (!opts[0]) {
				free(code0);
				code0 = 0;
				deleteVkFFT(app);
				return VKFFT_ERROR_MALLOC_FAILED;
			}
			numOpts++;
			sprintf(opts[0], "-ffp-contract=off");
		}

		result = hiprtcCompileProgram(prog,  // prog
			numOpts,     // numOptions
			(const char**)opts); // options

		if (app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory)
			free(opts[0]);	
		if (result != HIPRTC_SUCCESS) {
			printf("hiprtcCompileProgram error: %s\n", hiprtcGetErrorString(result));
			char* log = (char*)malloc(sizeof(char) * 100000);
			if (!log) {
				free(code0);
				code0 = 0;
				deleteVkFFT(app);
				return VKFFT_ERROR_FAILED_TO_COMPILE_PROGRAM;
			}
			else {
				hiprtcGetProgramLog(prog, log);
				printf("%s\n", log);
				free(log);
				log = 0;
				printf("%s\n", code0);
				free(code0);
				code0 = 0;
				deleteVkFFT(app);
				return VKFFT_ERROR_FAILED_TO_COMPILE_PROGRAM;
			}
		}
		result = hiprtcGetCodeSize(prog, &codeSize);
		if (result != HIPRTC_SUCCESS) {
			printf("hiprtcGetCodeSize error: %s\n", hiprtcGetErrorString(result));
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_TO_GET_CODE;
		}
		axis->binarySize = codeSize;
		code = (uint32_t*)malloc(codeSize);
		if (!code) {
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
		axis->binary = code;
		result = hiprtcGetCode(prog, (char*)code);
		if (result != HIPRTC_SUCCESS) {
			printf("hiprtcGetCode error: %s\n", hiprtcGetErrorString(result));
			free(code);
			code = 0;
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_TO_GET_CODE_SIZE;
		}
		//printf("%s\n", code);
		// Destroy the program.
		result = hiprtcDestroyProgram(&prog);
		if (result != HIPRTC_SUCCESS) {
			printf("hiprtcDestroyProgram error: %s\n", hiprtcGetErrorString(result));
			free(code);
			code = 0;
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_TO_DESTROY_PROGRAM;
		}
	}
	hipError_t result2 = hipModuleLoadDataEx(&axis->VkFFTModule, code, 0, 0, 0);

	if (result2 != hipSuccess) {
		printf("hipModuleLoadDataEx error: %d\n", result2);
		free(code);
		code = 0;
		free(code0);
		code0 = 0;
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_LOAD_MODULE;
	}
	result2 = hipModuleGetFunction(&axis->VkFFTKernel, axis->VkFFTModule, axis->VkFFTFunctionName);
	if (result2 != hipSuccess) {
		printf("hipModuleGetFunction error: %d\n", result2);
		free(code);
		code = 0;
		free(code0);
		code0 = 0;
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_GET_FUNCTION;
	}
	if ((pfUINT)axis->specializationConstants.usedSharedMemory.data.i > app->configuration.sharedMemorySizeStatic) {
		result2 = hipFuncSetAttribute(axis->VkFFTKernel, hipFuncAttributeMaxDynamicSharedMemorySize, (int)axis->specializationConstants.usedSharedMemory.data.i);
		//result2 = hipFuncSetCacheConfig(axis->VkFFTKernel, hipFuncCachePreferShared);
		if (result2 != hipSuccess) {
			printf("hipFuncSetAttribute error: %d\n", result2);
			free(code);
			code = 0;
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_TO_SET_DYNAMIC_SHARED_MEMORY;
		}
	}
	/*if (axis->pushConstants.structSize) {
		size_t size = axis->pushConstants.structSize;
		result2 = hipModuleGetGlobal(&axis->consts_addr, &size, axis->VkFFTModule, "consts");
		if (result2 != hipSuccess) {
			printf("hipModuleGetGlobal error: %d\n", result2);
			free(code);
			code = 0;
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_TO_MODULE_GET_GLOBAL;
		}
	}*/
	if (!app->configuration.saveApplicationToString) {
		free(code);
		code = 0;
	}
#elif(VKFFT_BACKEND==3)
	if (app->configuration.loadApplicationFromString) {
		char* code;
		pfUINT codeSize;
		char* localStrPointer = (char*)app->configuration.loadApplicationString + app->currentApplicationStringPos;
		memcpy(&codeSize, localStrPointer, sizeof(pfUINT));
		size_t codeSize_size_t = (size_t)codeSize;
		code = (char*)malloc(codeSize);
		if (!code) {
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
		memcpy(code, localStrPointer + sizeof(pfUINT), codeSize);
		app->currentApplicationStringPos += codeSize + sizeof(pfUINT);
		const unsigned char* temp_code = (const unsigned char*)code;
		axis->program = clCreateProgramWithBinary(app->configuration.context[0], 1, app->configuration.device, &codeSize_size_t, (const unsigned char**)(&temp_code), 0, &res);
		if (res != CL_SUCCESS) {
			free(code);
			code = 0;
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_TO_CREATE_PROGRAM;
		}
		free(code);
		code = 0;
	}
	else {
		size_t codelen = strlen(code0);
		const char* temp_code = (const char*)code0;
		axis->program = clCreateProgramWithSource(app->configuration.context[0], 1, (const char**)&temp_code, &codelen, &res);
		if (res != CL_SUCCESS) {
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_TO_CREATE_PROGRAM;
		}
	}
	res = clBuildProgram(axis->program, 1, app->configuration.device, 0, 0, 0);
	if (res != CL_SUCCESS) {
		size_t log_size;
		clGetProgramBuildInfo(axis->program, app->configuration.device[0], CL_PROGRAM_BUILD_LOG, 0, 0, &log_size);
		char* log = (char*)malloc(log_size);
		if (!log) {
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_TO_COMPILE_PROGRAM;
		}
		else {
			clGetProgramBuildInfo(axis->program, app->configuration.device[0], CL_PROGRAM_BUILD_LOG, log_size, log, 0);
			printf("%s\n", log);
			free(log);
			log = 0;
			printf("%s\n", code0);
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_TO_COMPILE_PROGRAM;
		}
	}
	if (app->configuration.saveApplicationToString) {
		size_t codeSize;
		res = clGetProgramInfo(axis->program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &codeSize, NULL);
		if (res != CL_SUCCESS) {
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_TO_COMPILE_PROGRAM;
		}
		axis->binarySize = (pfUINT)codeSize;
		axis->binary = (char*)malloc(axis->binarySize);
		if (!axis->binary) {
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
		res = clGetProgramInfo(axis->program, CL_PROGRAM_BINARIES, sizeof(unsigned char*), &axis->binary, NULL);
		if (res != CL_SUCCESS) {
			if (app->configuration.saveApplicationToString) {
				free(axis->binary);
				axis->binary = 0;
			}
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_TO_COMPILE_PROGRAM;
		}
	}
	axis->kernel = clCreateKernel(axis->program, axis->VkFFTFunctionName, &res);
	if (res != CL_SUCCESS) {
		if (app->configuration.saveApplicationToString) {
			free(axis->binary);
			axis->binary = 0;
		}
		free(code0);
		code0 = 0;
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_CREATE_SHADER_MODULE;
	}
#elif(VKFFT_BACKEND==4)
	uint32_t* code;
	pfUINT codeSize;
	if (app->configuration.loadApplicationFromString) {
		char* localStrPointer = (char*)app->configuration.loadApplicationString + app->currentApplicationStringPos;
		memcpy(&codeSize, localStrPointer, sizeof(pfUINT));
		code = (uint32_t*)malloc(codeSize);
		if (!code) {
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
		memcpy(code, localStrPointer + sizeof(pfUINT), codeSize);
		app->currentApplicationStringPos += codeSize + sizeof(pfUINT);

		const char* pBuildFlags = (app->configuration.useUint64) ? "-ze-opt-greater-than-4GB-buffer-required" : 0;
		ze_module_desc_t moduleDesc = {
			ZE_STRUCTURE_TYPE_MODULE_DESC,
			0,
			ZE_MODULE_FORMAT_NATIVE,
			codeSize,
			(uint8_t*)code,
			pBuildFlags,
			0
		};
		res = zeModuleCreate(app->configuration.context[0], app->configuration.device[0], &moduleDesc, &axis->VkFFTModule, 0);
		if (res != ZE_RESULT_SUCCESS) {
			free(code);
			code = 0;
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_TO_CREATE_PROGRAM;
		}
		free(code);
		code = 0;
	}
	else {
		size_t codelen = strlen(code0);
		pfUINT successOpen = 0;
		FILE* temp;
		char fname_cl[100];
		char fname_bc[100];
		char fname_spv[100];
		int name_id = 0;
		while (!successOpen) {
			sprintf(fname_cl, "VkFFT_temp_cl_%d.cl", name_id);
			temp = fopen(fname_cl, "r");
			if (temp != 0) {
				fclose(temp);
				name_id++;
			}
			else {
				successOpen = 1;
				sprintf(fname_bc, "VkFFT_temp_bc_%d.spv", name_id);
				sprintf(fname_spv, "VkFFT_temp_cl_%d.spv", name_id);
			}
		}
		temp = fopen(fname_cl, "w");
		fwrite(code0, 1, codelen, temp);
		fclose(temp);
		char system_call[500];
		sprintf(system_call, "clang -c -target spir64 -O0 -emit-llvm -o %s %s", fname_bc, fname_cl);
		system(system_call);
		sprintf(system_call, "llvm-spirv -o %s %s", fname_spv, fname_bc);
		system(system_call);
		temp = fopen(fname_spv, "rb");
		fseek(temp, 0L, SEEK_END);
		pfUINT spv_size = ftell(temp);
		rewind(temp);

		uint8_t* spv_binary = (uint8_t*)malloc(spv_size);
		if (!spv_binary) {
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
		fread(spv_binary, 1, spv_size, temp);
		fclose(temp);
		remove(fname_cl);
		remove(fname_bc);
		remove(fname_spv);
		const char* pBuildFlags = (app->configuration.useUint64) ? "-ze-opt-greater-than-4GB-buffer-required" : 0;

		ze_module_desc_t moduleDesc = {
			ZE_STRUCTURE_TYPE_MODULE_DESC,
			0,
			ZE_MODULE_FORMAT_IL_SPIRV,
			spv_size,
			spv_binary,
			pBuildFlags,
			0
		};
		res = zeModuleCreate(app->configuration.context[0], app->configuration.device[0], &moduleDesc, &axis->VkFFTModule, 0);
		if (res != ZE_RESULT_SUCCESS) {
			free(spv_binary);
			spv_binary = 0;
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_TO_CREATE_PROGRAM;
		}
		free(spv_binary);
		spv_binary = 0;
		if (app->configuration.saveApplicationToString) {
			size_t codeSize;
			res = zeModuleGetNativeBinary(axis->VkFFTModule, &codeSize, 0);
			if (res != ZE_RESULT_SUCCESS) {
				free(code0);
				code0 = 0;
				deleteVkFFT(app);
				return VKFFT_ERROR_FAILED_TO_COMPILE_PROGRAM;
			}
			axis->binarySize = codeSize;
			axis->binary = (char*)malloc(axis->binarySize);
			if (!axis->binary) {
				free(code0);
				code0 = 0;
				deleteVkFFT(app);
				return VKFFT_ERROR_MALLOC_FAILED;
			}
			res = zeModuleGetNativeBinary(axis->VkFFTModule, &codeSize, (uint8_t*)axis->binary);
			if (res != ZE_RESULT_SUCCESS) {
				free(axis->binary);
				axis->binary = 0;
				free(code0);
				code0 = 0;
				deleteVkFFT(app);
				return VKFFT_ERROR_FAILED_TO_COMPILE_PROGRAM;
			}
		}
	}
	ze_kernel_desc_t kernelDesc = {
		ZE_STRUCTURE_TYPE_KERNEL_DESC,
		0,
		0, // flags
		axis->VkFFTFunctionName
	};
	res = zeKernelCreate(axis->VkFFTModule, &kernelDesc, &axis->VkFFTKernel);
	if (res != ZE_RESULT_SUCCESS) {
		if (app->configuration.saveApplicationToString) {
			free(axis->binary);
			axis->binary = 0;
		}
		free(code0);
		code0 = 0;
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_CREATE_SHADER_MODULE;
	}
#elif(VKFFT_BACKEND==5)
	NS::Error* error;
	if (app->configuration.loadApplicationFromString) {
		char* code;
		pfUINT codeSize;
		char* localStrPointer = (char*)app->configuration.loadApplicationString + app->currentApplicationStringPos;
		memcpy(&codeSize, localStrPointer, sizeof(pfUINT));
		size_t codeSize_size_t = (size_t)codeSize;
		code = (char*)malloc(codeSize);
		if (!code) {
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
		memcpy(code, localStrPointer + sizeof(pfUINT), codeSize);
		app->currentApplicationStringPos += codeSize + sizeof(pfUINT);
		dispatch_data_t data = dispatch_data_create(code, codeSize, 0, 0);
		axis->library = app->configuration.device->newLibrary(data, &error);
		free(code);
		code = 0;
	}
	else {
		size_t codelen = strlen(code0);
		MTL::CompileOptions* compileOptions = MTL::CompileOptions::alloc();
		compileOptions->setFastMathEnabled(true);
		NS::String* str = NS::String::string(code0, NS::UTF8StringEncoding);
		axis->library = app->configuration.device->newLibrary(str, compileOptions, &error);
		if (error) {
			printf("%s\n%s\n", error->debugDescription()->cString(NS::ASCIIStringEncoding), error->localizedDescription()->cString(NS::ASCIIStringEncoding));
			free(code0);
			code0 = 0;
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_TO_COMPILE_PROGRAM;
		}
		compileOptions->release();
		if (app->configuration.saveApplicationToString) {

		}
		str->release();
	}
	//const char function_name[20] = "VkFFT_main_R2C";
	NS::String* str = NS::String::string(axis->VkFFTFunctionName, NS::UTF8StringEncoding);
	MTL::Function* function = axis->library->newFunction(str);
	axis->pipeline = app->configuration.device->newComputePipelineState(function, &error);
	function->release();
	str->release();
#endif
	return VKFFT_SUCCESS;
}
#endif
