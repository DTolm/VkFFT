// This file is part of VkFFT, a Vulkan Fast Fourier Transform library
//
// Copyright (C) 2020 Dmitrii Tolmachev <dtolm96@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/. 
#ifndef VKFFT_H
#define VKFFT_H
#ifdef __cplusplus
extern "C" {
#endif

#include <memory.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vulkan/vulkan.h"
#include "glslang_c_interface.h"
	//#include "shaderc/shaderc.h"
	typedef struct {
		//WHDCN layout
		uint32_t size[3]; // WHD -system dimensions 
		uint32_t bufferStride[3];
		uint32_t inputBufferStride[3];
		uint32_t outputBufferStride[3];
		uint32_t maxComputeWorkGroupCount[3]; // maxComputeWorkGroupCount from VkPhysicalDeviceLimits
		uint32_t coordinateFeatures; // C - coordinate, or dimension of features vector. In matrix convolution - size of vector
		uint32_t matrixConvolution; //if equal to 2 perform 2x2, if equal to 3 perform 3x3 matrix-vector convolution. Overrides coordinateFeatures

		uint32_t numberBatches;// N - used to perform multiple batches of initial data
		uint32_t numberKernels;// N - only used in convolution step - specify how many kernels were initialized before. Expands one input to multiple (batched) output
		uint32_t FFTdim; //FFT dimensionality (1, 2 or 3)
		uint32_t radix; //FFT radix (8)
		VkBool32 performZeropaddingInput[3]; // don't read if input is zeropadded (0 - off, 1 - on)
		VkBool32 performZeropaddingOutput[3]; // don't write if output is zeropadded (0 - off, 1 - on)
		uint32_t fft_zeropad_left_read[3];
		uint32_t fft_zeropad_left_write[3];
		uint32_t fft_zeropad_right_read[3];
		uint32_t fft_zeropad_right_write[3];
		VkBool32 performTranspose[2]; //will be selected automatically
		VkBool32 performConvolution; //perform convolution in this application (0 - off, 1 - on)
		VkBool32 performR2C; //perform R2C/C2R decomposition (0 - off, 1 - on)
		VkBool32 inverse; //perform inverse FFT (0 - forward, 1 - inverse)
		VkBool32 symmetricKernel; //specify if kernel in 2x2 or 3x3 matrix convolution is symmetric
		VkBool32 isInputFormatted; //specify if input buffer is not padded for R2C if out-of-place mode is selected (only if numberBatches==1 and numberKernels==1) - 0 - padded, 1 - not padded
		VkBool32 isOutputFormatted; //specify if output buffer is not padded for R2C if out-of-place mode is selected (only if numberBatches==1 and numberKernels==1) - 0 - padded, 1 - not padded
		VkBool32 reorderFourStep; //1 to enable - unshuffles four step algorithm. Requires additional buffer allocation
		VkBool32 useLUT; //1 to enable - switches from calculating sincos to using precomputed LUT tables
		VkBool32 doublePrecision; //1 to enable
		VkBool32 halfPrecision; //1 to enable
		uint32_t devicePageSize;//in KB, the size of a page on the GPU. Setting to 0 disables local buffer split in pages
		uint32_t localPageSize;//in KB, the size to split page into if sequence spans multiple devicePageSize pages
		uint32_t sharedMemorySize;//in bytes. Vulkan aims at power of 2 shared memory sizes
		uint32_t sharedMemorySizePow2;//in bytes. For now Vulkan is optimized for 32KB of shared memory
		uint32_t warpSize;//number of threads per warp/wavefront. Default 32
		uint32_t registerBoost; //specify if register file size is bigger than shared memory (on Nvidia 256KB register file can be used instead of 32KB of shared memory, set this constant to 4). Default 1, max 4
		uint32_t registerBoost4Step; //specify if register file overutilization should be used in big sequences (>2^14), same value or lower as above. Default 1, max 4
		uint32_t swapTo3Stage4Step; //specify at which power of 2 to switch from 2 upload to 3 upload 4-step FFT, in case if making max sequence size lower than coalesced sequence helps to combat TLB misses. Default 0 - disabled. Must be at least 17
		uint32_t performHalfBandwidthBoost;//try to reduce coalsesced number by a factor of 2 to get bigger sequence in one upload
		char shaderPath[256]; //path to shaders, can be selected automatically in CMake
		uint32_t coalescedMemory;//in bits, for Nvidia compute capability >=6.0 is equal to 32, <6.0 and Intel is equal 128. Gonna work regardles, but if specified by user correctly, the performance will be higher. 
		VkDevice* device;
		VkQueue* queue;
		VkCommandPool* commandPool;
		VkFence* fence;
		VkPhysicalDevice* physicalDevice;

		uint32_t bufferNum;
		uint32_t tempBufferNum;
		uint32_t inputBufferNum;
		uint32_t outputBufferNum;
		uint32_t kernelNum;

		VkDeviceSize* bufferSize;
		VkDeviceSize* tempBufferSize;
		VkDeviceSize* inputBufferSize;
		VkDeviceSize* outputBufferSize;
		VkDeviceSize* kernelSize;

		VkBuffer* buffer;
		VkBuffer* tempBuffer;//needed if reorderFourStep is enabled to transpose the array. Same size as buffer
		VkBuffer* inputBuffer;
		VkBuffer* outputBuffer;
		VkBuffer* kernel;

		uint32_t isCompilerInitialized;

		uint32_t halfThreads;
	} VkFFTConfiguration;

	static VkFFTConfiguration defaultVkFFTConfiguration = { {1,1,1}, {1,1,1}, {1,1,1}, {1,1,1}, {65535,65535,65535},1,1,1,1,1,8,{0,0,0},{0,0,0},{0,0,0},{0,0,0},{0,0,0},{0,0,0}, {0,0},0,0,0,0,0,0,0,0,0, 0, 0, 0, 32768, 32768, 32, 1, 1, 0, 1,"shaders/", 32, 0,0,0,0,0, 1,1,1,1,1, 0,0,0,0,0, 0,0,0,0,0,0, 0 };

	typedef struct {
		uint32_t size[3];
		uint32_t localSize[3];
		uint32_t fftDim;
		VkBool32 inverse;
		VkBool32 zeropad[2];
		uint32_t axis_id;
		uint32_t axis_upload_id;
		uint32_t registers_per_thread;
		uint32_t min_registers_per_thread;
		uint32_t readToRegisters;
		uint32_t writeFromRegisters;
		uint32_t LUT;
		uint32_t performR2C;
		VkBool32 performZeropaddingInput[3]; // don't read if input is zeropadded (0 - off, 1 - on)
		VkBool32 performZeropaddingOutput[3]; // don't write if output is zeropadded (0 - off, 1 - on)
		uint32_t fft_zeropad_left_read[3];
		uint32_t fft_zeropad_left_write[3];
		uint32_t fft_zeropad_right_read[3];
		uint32_t fft_zeropad_right_write[3];
		uint32_t inputStride[5];
		uint32_t outputStride[5];
		uint32_t fft_dim_full;
		uint32_t stageStartSize;
		uint32_t firstStageStartSize;
		uint32_t fft_dim_x;
		uint32_t numStages;
		uint32_t stageRadix[20];
		uint32_t inputOffset;
		uint32_t outputOffset;
		VkBool32 reorderFourStep;
		uint32_t performWorkGroupShift[3];
		uint32_t inputBufferBlockNum;
		uint32_t inputBufferBlockSize;
		uint32_t outputBufferBlockNum;
		uint32_t outputBufferBlockSize;
		uint32_t kernelBlockNum;
		uint32_t kernelBlockSize;
		uint32_t numCoordinates;
		uint32_t matrixConvolution; //if equal to 2 perform 2x2, if equal to 3 perform 3x3 matrix-vector convolution. Overrides coordinateFeatures
		uint32_t numBatches;
		uint32_t numKernels;
		uint32_t sharedMemSize;
		uint32_t sharedMemSizePow2;
		uint32_t normalize;
		uint32_t complexSize;
		uint32_t maxStageSumLUT;
		uint32_t unroll;
		uint32_t convolutionStep;
		uint32_t symmetricKernel;
		uint32_t supportAxis;
		uint32_t cacheShuffle;
	} VkFFTSpecializationConstantsLayout;

	typedef struct {
		uint32_t coordinate;
		uint32_t batch;
		uint32_t workGroupShift[3];
	} VkFFTPushConstantsLayout;

	typedef struct {
		uint32_t localSize[3];
		uint32_t inputStride[5];
		uint32_t ratio;
		VkBool32 ratioDirection;
	} VkFFTTransposeSpecializationConstantsLayout;
	typedef struct {
		uint32_t axisBlock[4];
		uint32_t groupedBatch;
		VkFFTSpecializationConstantsLayout specializationConstants;
		VkFFTPushConstantsLayout pushConstants;
		VkDescriptorPool descriptorPool;
		VkDescriptorSetLayout descriptorSetLayout;
		VkDescriptorSet descriptorSet;
		VkPipelineLayout pipelineLayout;
		VkPipeline pipeline;
		VkDeviceSize bufferLUTSize;
		VkBuffer bufferLUT;
		VkDeviceMemory bufferLUTDeviceMemory;
	} VkFFTAxis;
	typedef struct {
		uint32_t transposeBlock[3];
		VkFFTTransposeSpecializationConstantsLayout specializationConstants;
		VkFFTPushConstantsLayout pushConstants;
		VkDescriptorPool descriptorPool;
		VkDescriptorSetLayout descriptorSetLayout;
		VkDescriptorSet descriptorSet;
		VkPipelineLayout pipelineLayout;
		VkPipeline pipeline;
	} VkFFTTranspose;
	typedef struct {
		uint32_t numAxisUploads[3];
		uint32_t axisSplit[3][5];
		uint32_t numSupportAxisUploads[2];
		VkFFTAxis axes[3][5];
		VkFFTAxis supportAxes[2][5];//Nx/2+1 for r2c/c2r
		VkFFTTranspose transpose[2];

	} VkFFTPlan;
	typedef struct {
		VkFFTConfiguration configuration;
		VkFFTPlan localFFTPlan;
		VkFFTPlan localFFTPlan_inverse_convolution; //additional inverse plan for convolution.
	} VkFFTApplication;
	static VkFFTApplication defaultVkFFTApplication = { {}, {}, {} };

	static inline void appendLicense(char* output) {
		sprintf(output + strlen(output), "\
// This file is part of VkFFT, a Vulkan Fast Fourier Transform library\n\
//\n\
// Copyright (C) 2020 Dmitrii Tolmachev <dtolm96@gmail.com>\n\
//\n\
// This Source Code Form is subject to the terms of the Mozilla Public\n\
// License, v. 2.0. If a copy of the MPL was not distributed with this\n\
// file, You can obtain one at https://mozilla.org/MPL/2.0/. \n");
	}
	static inline void appendVersion(char* output) {
		sprintf(output, "#version 450\n\n");
	}
	static inline void appendExtensions(char* output, const char* floatType, const char* floatTypeMemory) {
		if (!strcmp(floatType, "double"))
			sprintf(output + strlen(output), "\
#extension GL_ARB_gpu_shader_fp64 : enable\n\
#extension GL_ARB_gpu_shader_int64 : enable\n\n");
		if (!strcmp(floatTypeMemory, "half"))
			sprintf(output + strlen(output), "#extension GL_EXT_shader_16bit_storage : require\n\n");
	}
	static inline void appendLayoutVkFFT(char* output, VkFFTSpecializationConstantsLayout sc) {
		sprintf(output + strlen(output), "layout (local_size_x = %d, local_size_y = %d, local_size_z = %d) in;\n", sc.localSize[0], sc.localSize[1], sc.localSize[2]);
		//sprintf(output+strlen(output), "layout (local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;\n");
	}
	static inline void appendConstant(char* output, const char* type, const char* name, const char* defaultVal) {
		sprintf(output + strlen(output), "const %s %s = %s;\n", type, name, defaultVal);
	}
	static inline void appendPushConstant(char* output, const char* type, const char* name) {
		sprintf(output + strlen(output), "	%s %s;\n", type, name);
	}
	static inline void appendBarrierVkFFT(char* output, uint32_t numTab) {
		char tabs[100];
		for (uint32_t i = 0; i < numTab; i++)
			sprintf(tabs, "	");
		sprintf(output + strlen(output), "%s\n%smemoryBarrierShared();\nbarrier();\n\n", tabs, tabs);
	}
	static inline void appendPushConstantsVkFFT(char* output, VkFFTSpecializationConstantsLayout sc, const char* floatType, const char* uintType) {
		sprintf(output + strlen(output), "layout(push_constant) uniform PushConsts\n{\n");
		appendPushConstant(output, uintType, "coordinate");
		appendPushConstant(output, uintType, "batchID");
		appendPushConstant(output, uintType, "workGroupShiftX");
		appendPushConstant(output, uintType, "workGroupShiftY");
		appendPushConstant(output, uintType, "workGroupShiftZ");
		sprintf(output + strlen(output), "} consts;\n\n");
	}
	static inline void appendConstantsVkFFT(char* output, const char* floatType, const char* uintType) {
		appendConstant(output, floatType, "M_PI", "3.1415926535897932384626433832795");
		appendConstant(output, floatType, "M_SQRT1_2", "0.70710678118654752440084436210485");
	}
	static inline void appendSinCos20(char* output, const char* floatType, const char* uintType) {
		appendConstant(output, floatType, "M_2_PI", "0.63661977236758134307553505349006");
		appendConstant(output, floatType, "M_PI_2", "1.5707963267948966192313216916398");
		appendConstant(output, floatType, "a1", "0.99999999999999999999962122687403772");
		appendConstant(output, floatType, "a3", "-0.166666666666666666637194166219637268");
		appendConstant(output, floatType, "a5", "0.00833333333333333295212653322266277182");
		appendConstant(output, floatType, "a7", "-0.000198412698412696489459896530659927773");
		appendConstant(output, floatType, "a9", "2.75573192239364018847578909205399262e-6");
		appendConstant(output, floatType, "a11", "-2.50521083781017605729370231280411712e-8");
		appendConstant(output, floatType, "a13", "1.60590431721336942356660057796782021e-10");
		appendConstant(output, floatType, "a15", "-7.64712637907716970380859898835680587e-13");
		appendConstant(output, floatType, "a17", "2.81018528153898622636194976499656274e-15");
		appendConstant(output, floatType, "ab", "-7.97989713648499642889739108679114937e-18");
		sprintf(output + strlen(output), "\
dvec2 sincos_20(double x)\n\
{\n\
	//minimax coefs for sin for 0..pi/2 range\n\
	double y = abs(x * M_2_PI);\n\
	double q = floor(y);\n\
	int quadrant = int(q);\n\
	double t = (quadrant & 1) != 0 ? 1 - y + q : y - q;\n\
	t *= M_PI_2;\n\
	double t2 = t * t;\n\
	double r = fma(fma(fma(fma(fma(fma(fma(fma(fma(ab, t2, a17), t2, a15), t2, a13), t2, a11), t2, a9), t2, a7), t2, a5), t2, a3), t2 * t, t);\n\
	dvec2 cos_sin;\n\
	cos_sin.x = ((quadrant == 0) || (quadrant == 3)) ? sqrt(1 - r * r) : -sqrt(1 - r * r);\n\
	r = x < 0 ? -r : r;\n\
	cos_sin.y = (quadrant & 2) != 0 ? -r : r;\n\
	return cos_sin;\n\
}\n\n");
	}
	static inline void appendInputLayoutVkFFT(char* output, VkFFTSpecializationConstantsLayout sc, uint32_t id, const char* floatTypeMemory, uint32_t inputType) {
		char vecType[10];
		switch (inputType) {
		case 0: case 1: case 2: case 3: case 4: case 6: {
			if (!strcmp(floatTypeMemory, "half")) sprintf(vecType, "f16vec2");
			if (!strcmp(floatTypeMemory, "float")) sprintf(vecType, "vec2");
			if (!strcmp(floatTypeMemory, "double")) sprintf(vecType, "dvec2");
			sprintf(output + strlen(output), "\
layout(std430, binding = %d) buffer DataIn{\n\
	%s inputs[%d];\n\
} inputBlocks[%d];\n\n", id, vecType, sc.inputBufferBlockSize, sc.inputBufferBlockNum);
			break;
		}
		case 5:
		{
			if (!strcmp(floatTypeMemory, "half")) sprintf(vecType, "float16_t");
			if (!strcmp(floatTypeMemory, "float")) sprintf(vecType, "float");
			if (!strcmp(floatTypeMemory, "double")) sprintf(vecType, "double");
			sprintf(output + strlen(output), "\
layout(std430, binding = %d) buffer DataIn{\n\
	%s inputs[%d];\n\
} inputBlocks[%d];\n\n", id, vecType, 2 * sc.inputBufferBlockSize, sc.inputBufferBlockNum);
			break;
		}
		}


	}
	static inline void appendOutputLayoutVkFFT(char* output, VkFFTSpecializationConstantsLayout sc, uint32_t id, const char* floatTypeMemory, uint32_t outputType) {
		char vecType[10];
		switch (outputType) {
		case 0: case 1: case 2: case 3: case 4: case 5: {
			if (!strcmp(floatTypeMemory, "half")) sprintf(vecType, "f16vec2");
			if (!strcmp(floatTypeMemory, "float")) sprintf(vecType, "vec2");
			if (!strcmp(floatTypeMemory, "double")) sprintf(vecType, "dvec2");
			sprintf(output + strlen(output), "\
layout(std430, binding = %d) buffer DataOut{\n\
	%s outputs[%d];\n\
} outputBlocks[%d];\n\n", id, vecType, sc.outputBufferBlockSize, sc.outputBufferBlockNum);
			break;
		}
		case 6:
		{
			if (!strcmp(floatTypeMemory, "half")) sprintf(vecType, "float16_t");
			if (!strcmp(floatTypeMemory, "float")) sprintf(vecType, "float");
			if (!strcmp(floatTypeMemory, "double")) sprintf(vecType, "double");
			sprintf(output + strlen(output), "\
layout(std430, binding = %d) buffer DataOut{\n\
	%s outputs[%d];\n\
} outputBlocks[%d];\n\n", id, vecType, 2 * sc.outputBufferBlockSize, sc.outputBufferBlockNum);
			break;
		}
		}
	}
	static inline void appendKernelLayoutVkFFT(char* output, VkFFTSpecializationConstantsLayout sc, uint32_t id, const char* floatTypeMemory) {
		char vecType[10];
		if (!strcmp(floatTypeMemory, "half")) sprintf(vecType, "f16vec2");
		if (!strcmp(floatTypeMemory, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatTypeMemory, "double")) sprintf(vecType, "dvec2");

		sprintf(output + strlen(output), "\
layout(std430, binding = %d) buffer Kernel_FFT{\n\
	%s kernel[%d];\n\
} kernelBlocks[%d];\n\n", id, vecType, sc.kernelBlockSize, sc.kernelBlockNum);

	}
	static inline void appendLUTLayoutVkFFT(char* output, VkFFTSpecializationConstantsLayout sc, uint32_t id, const char* floatType) {
		char vecType[10];
		if (!strcmp(floatType, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "dvec2");
		sprintf(output + strlen(output), "\
layout(std430, binding = %d) readonly buffer DataLUT {\n\
%s twiddleLUT[];\n\
};\n", id, vecType);
	}
	static inline void appendIndexInputVkFFT(char* output, VkFFTSpecializationConstantsLayout sc, const char* uintType, uint32_t inputType) {
		switch (inputType) {
		case 0: case 2: case 3: case 4: {//single_c2c + single_c2c_strided
			char inputOffset[30] = "";
			if (sc.inputOffset > 0)
				sprintf(inputOffset, "%d + ", sc.inputOffset);
			char shiftX[100] = "";
			if (sc.inputStride[0] == 1)
				sprintf(shiftX, "index");
			else
				sprintf(shiftX, "index * %d", sc.inputStride[0]);
			char shiftY[100] = "";
			if (sc.size[1] > 1) {
				if (sc.fftDim == sc.fft_dim_full) {
					if (sc.performWorkGroupShift[1])
						sprintf(shiftY, " + (gl_WorkGroupID.y + consts.workGroupShiftY) * %d", sc.localSize[1] * sc.inputStride[1]);
					else
						sprintf(shiftY, " + gl_WorkGroupID.y * %d", sc.localSize[1] * sc.inputStride[1]);
				}
				else {
					if (sc.performWorkGroupShift[1])
						sprintf(shiftY, " + (gl_WorkGroupID.y + consts.workGroupShiftY) * %d", sc.inputStride[1]);
					else
						sprintf(shiftY, " + gl_WorkGroupID.y * %d", sc.inputStride[1]);
				}
			}
			char shiftZ[100] = "";
			if (sc.size[2] > 1) {
				if (sc.performWorkGroupShift[2])
					sprintf(shiftZ, " + (gl_GlobalInvocationID.z + consts.workGroupShiftZ * gl_WorkGroupSize.z) * %d", sc.inputStride[2]);
				else
					sprintf(shiftZ, " + gl_GlobalInvocationID.z * %d", sc.inputStride[2]);
			}
			char shiftCoordinate[100] = "";
			char requestCoordinate[100] = "";
			if (sc.numCoordinates * sc.matrixConvolution > 1) {
				sprintf(shiftCoordinate, " + consts.coordinate * %d", sc.inputStride[3]);
			}
			if ((sc.matrixConvolution > 1) && (sc.convolutionStep)) {
				sprintf(shiftCoordinate, " + coordinate * %d", sc.inputStride[3]);
				sprintf(requestCoordinate, ", %s coordinate", uintType);
			}
			char shiftBatch[100] = "";
			char requestBatch[100] = "";
			if ((sc.numBatches > 1) || (sc.numKernels > 1)) {
				if (sc.convolutionStep) {
					sprintf(shiftBatch, " + batchID * %d", sc.inputStride[4]);
					sprintf(requestBatch, ", %s batchID", uintType);
				}
				else
					sprintf(shiftBatch, " + consts.batchID * %d", sc.inputStride[4]);
			}
			sprintf(output + strlen(output), "\
%s indexInput(%s index%s%s) {\n\
	return %s%s%s%s%s%s;\n\
}\n\n", uintType, uintType, requestCoordinate, requestBatch, inputOffset, shiftX, shiftY, shiftZ, shiftCoordinate, shiftBatch);
			break;
		}
		case 1: {//grouped_c2c
			char inputOffset[30] = "";
			if (sc.inputOffset > 0)
				sprintf(inputOffset, "%d + ", sc.inputOffset);
			char shiftX[100] = "";
			if (sc.inputStride[0] == 1)
				sprintf(shiftX, "index_x");
			else
				sprintf(shiftX, "index_x * %d", sc.inputStride[0]);

			char shiftY[100] = "";
			sprintf(shiftY, " + index_y * %d", sc.inputStride[1]);

			char shiftZ[100] = "";
			if (sc.size[2] > 1) {
				if (sc.performWorkGroupShift[2])
					sprintf(shiftZ, " + (gl_GlobalInvocationID.z + consts.workGroupShiftZ * gl_WorkGroupSize.z) * %d", sc.inputStride[2]);
				else
					sprintf(shiftZ, " + gl_GlobalInvocationID.z * %d", sc.inputStride[2]);
			}
			char shiftCoordinate[100] = "";
			char requestCoordinate[100] = "";
			if (sc.numCoordinates * sc.matrixConvolution > 1) {
				sprintf(shiftCoordinate, " + consts.coordinate * %d", sc.outputStride[3]);
			}
			if ((sc.matrixConvolution > 1) && (sc.convolutionStep)) {
				sprintf(shiftCoordinate, " + coordinate * %d", sc.inputStride[3]);
				sprintf(requestCoordinate, ", %s coordinate", uintType);
			}
			char shiftBatch[100] = "";
			char requestBatch[100] = "";
			if ((sc.numBatches > 1) || (sc.numKernels > 1)) {
				if (sc.convolutionStep) {
					sprintf(shiftBatch, " + batchID * %d", sc.inputStride[4]);
					sprintf(requestBatch, ", %s batchID", uintType);
				}
				else
					sprintf(shiftBatch, " + consts.batchID * %d", sc.inputStride[4]);
			}
			sprintf(output + strlen(output), "\
%s indexInput(%s index_x, %s index_y%s%s) {\n\
	return %s%s%s%s%s%s;\n\
}\n\n", uintType, uintType, uintType, requestCoordinate, requestBatch, inputOffset, shiftX, shiftY, shiftZ, shiftCoordinate, shiftBatch);
			break;
		}
		case 5: {//single_r2c
			char inputOffset[30] = "";
			if (sc.inputOffset > 0)
				sprintf(inputOffset, "%d + ", sc.inputOffset);
			char shiftX[100] = "";
			if (sc.inputStride[0] == 1)
				sprintf(shiftX, "index");
			else
				sprintf(shiftX, "index * %d", sc.inputStride[0]);
			char shiftY[100] = "";
			if (sc.size[1] > 1) {
				if (sc.fftDim == sc.fft_dim_full) {
					if (sc.performWorkGroupShift[1])
						sprintf(shiftY, " + (gl_WorkGroupID.y + consts.workGroupShiftY) * %d", 2 * sc.localSize[1] * sc.inputStride[1]);
					else
						sprintf(shiftY, " + gl_WorkGroupID.y * %d", 2 * sc.localSize[1] * sc.inputStride[1]);
				}
				else {
					if (sc.performWorkGroupShift[1])
						sprintf(shiftY, " + (gl_WorkGroupID.y + consts.workGroupShiftY) * %d", 2 * sc.inputStride[1]);
					else
						sprintf(shiftY, " + gl_WorkGroupID.y * %d", 2 * sc.inputStride[1]);
				}
			}
			char shiftZ[100] = "";
			if (sc.size[2] > 1) {
				if (sc.performWorkGroupShift[2])
					sprintf(shiftZ, " + (gl_GlobalInvocationID.z + consts.workGroupShiftZ * gl_WorkGroupSize.z) * %d", 2 * sc.inputStride[2]);
				else
					sprintf(shiftZ, " + gl_GlobalInvocationID.z * %d", 2 * sc.inputStride[2]);
			}
			char shiftCoordinate[100] = "";
			if (sc.numCoordinates * sc.matrixConvolution > 1) {
				sprintf(shiftCoordinate, " + consts.coordinate * %d", 2 * sc.inputStride[3]);
			}
			char shiftBatch[100] = "";
			if ((sc.numBatches > 1) || (sc.numKernels > 1)) {
				sprintf(shiftBatch, " + consts.batchID * %d", 2 * sc.inputStride[4]);
			}
			sprintf(output + strlen(output), "\
%s indexInput(%s index) {\n\
	return %s%s%s%s%s%s;\n\
}\n\n", uintType, uintType, inputOffset, shiftX, shiftY, shiftZ, shiftCoordinate, shiftBatch);
			break;
		}
		case 6: {//single_c2r
			char inputOffset[30] = "";
			if (sc.inputOffset > 0)
				sprintf(inputOffset, "%d + ", sc.inputOffset);
			char shiftX[100] = "";
			if (sc.inputStride[0] == 1)
				sprintf(shiftX, "index_x");
			else
				sprintf(shiftX, "index_x * %d", sc.inputStride[0]);
			char shiftY[100] = "";
			sprintf(shiftY, " + index_y * %d", sc.inputStride[1]);
			char shiftZ[100] = "";
			if (sc.size[2] > 1) {
				if (sc.performWorkGroupShift[2])
					sprintf(shiftZ, " + (gl_GlobalInvocationID.z + consts.workGroupShiftZ * gl_WorkGroupSize.z) * %d", sc.inputStride[2]);
				else
					sprintf(shiftZ, " + gl_GlobalInvocationID.z * %d", sc.inputStride[2]);
			}
			char shiftCoordinate[100] = "";
			if (sc.numCoordinates * sc.matrixConvolution > 1) {
				sprintf(shiftCoordinate, " + consts.coordinate * %d", sc.inputStride[3]);
			}
			char shiftBatch[100] = "";
			if ((sc.numBatches > 1) || (sc.numKernels > 1)) {
				sprintf(shiftBatch, " + consts.batchID * %d", sc.inputStride[4]);
			}
			sprintf(output + strlen(output), "\
%s indexInput(%s index_x, %s index_y) {\n\
	return %s%s%s%s%s%s;\n\
}\n\n", uintType, uintType, uintType, inputOffset, shiftX, shiftY, shiftZ, shiftCoordinate, shiftBatch);
			break;
		}
		}
	}
	static inline void appendIndexOutputVkFFT(char* output, VkFFTSpecializationConstantsLayout sc, const char* uintType, uint32_t outputType) {
		switch (outputType) {//single_c2c + single_c2c_strided
		case 0: case 2: case 3: case 4: {
			char outputOffset[30] = "";
			if (sc.outputOffset > 0)
				sprintf(outputOffset, "%d + ", sc.outputOffset);
			char shiftX[100] = "";
			if (sc.fftDim == sc.fft_dim_full)
				sprintf(shiftX, "index");
			else
				sprintf(shiftX, "index * %d", sc.outputStride[0]);
			char shiftY[100] = "";
			if (sc.size[1] > 1) {
				if (sc.fftDim == sc.fft_dim_full) {
					if (sc.performWorkGroupShift[1])
						sprintf(shiftY, " + (gl_WorkGroupID.y + consts.workGroupShiftY) * %d", sc.localSize[1] * sc.outputStride[1]);
					else
						sprintf(shiftY, " + gl_WorkGroupID.y * %d", sc.localSize[1] * sc.outputStride[1]);
				}
				else {
					if (sc.performWorkGroupShift[1])
						sprintf(shiftY, " + (gl_WorkGroupID.y + consts.workGroupShiftY) * %d", sc.outputStride[1]);
					else
						sprintf(shiftY, " + gl_WorkGroupID.y * %d", sc.outputStride[1]);
				}
			}
			char shiftZ[100] = "";
			if (sc.size[2] > 1) {
				if (sc.performWorkGroupShift[2])
					sprintf(shiftZ, " + (gl_GlobalInvocationID.z + consts.workGroupShiftZ * gl_WorkGroupSize.z) * %d", sc.outputStride[2]);
				else
					sprintf(shiftZ, " + gl_GlobalInvocationID.z * %d", sc.outputStride[2]);
			}
			char shiftCoordinate[100] = "";
			char requestCoordinate[100] = "";
			if (sc.numCoordinates * sc.matrixConvolution > 1) {
				sprintf(shiftCoordinate, " + consts.coordinate * %d", sc.outputStride[3]);
			}
			if ((sc.matrixConvolution > 1) && (sc.convolutionStep)) {
				sprintf(shiftCoordinate, " + coordinate * %d", sc.outputStride[3]);
				sprintf(requestCoordinate, ", %s coordinate", uintType);
			}
			char shiftBatch[100] = "";
			char requestBatch[100] = "";
			if ((sc.numBatches > 1) || (sc.numKernels > 1)) {
				if (sc.convolutionStep) {
					sprintf(shiftBatch, " + batchID * %d", sc.outputStride[4]);
					sprintf(requestBatch, ", %s batchID", uintType);
				}
				else
					sprintf(shiftBatch, " + consts.batchID * %d", sc.outputStride[4]);
			}
			sprintf(output + strlen(output), "\
%s indexOutput(%s index%s%s) {\n\
	return %s%s%s%s%s%s;\n\
}\n\n", uintType, uintType, requestCoordinate, requestBatch, outputOffset, shiftX, shiftY, shiftZ, shiftCoordinate, shiftBatch);
			break;
		}
		case 1: {//grouped_c2c
			char outputOffset[30] = "";
			if (sc.outputOffset > 0)
				sprintf(outputOffset, "%d + ", sc.outputOffset);
			char shiftX[100] = "";
			if (sc.fftDim == sc.fft_dim_full)
				sprintf(shiftX, "index_x");
			else
				sprintf(shiftX, "index_x * %d", sc.outputStride[0]);
			char shiftY[100] = "";
			sprintf(shiftY, " + index_y * %d", sc.outputStride[1]);
			char shiftZ[100] = "";
			if (sc.size[2] > 1) {
				if (sc.performWorkGroupShift[2])
					sprintf(shiftZ, " + (gl_GlobalInvocationID.z + consts.workGroupShiftZ * gl_WorkGroupSize.z) * %d", sc.outputStride[2]);
				else
					sprintf(shiftZ, " + gl_GlobalInvocationID.z * %d", sc.outputStride[2]);
			}
			char shiftCoordinate[100] = "";
			char requestCoordinate[100] = "";
			if (sc.numCoordinates * sc.matrixConvolution > 1) {
				sprintf(shiftCoordinate, " + consts.coordinate * %d", sc.outputStride[3]);
			}
			if ((sc.matrixConvolution > 1) && (sc.convolutionStep)) {
				sprintf(shiftCoordinate, " + coordinate * %d", sc.outputStride[3]);
				sprintf(requestCoordinate, ", %s coordinate", uintType);
			}
			char shiftBatch[100] = "";
			char requestBatch[100] = "";
			if ((sc.numBatches > 1) || (sc.numKernels > 1)) {
				if (sc.convolutionStep) {
					sprintf(shiftBatch, " + batchID * %d", sc.outputStride[4]);
					sprintf(requestBatch, ", %s batchID", uintType);
				}
				else
					sprintf(shiftBatch, " + consts.batchID * %d", sc.outputStride[4]);
			}
			sprintf(output + strlen(output), "\
%s indexOutput(%s index_x, %s index_y%s%s) {\n\
	return %s%s%s%s%s%s;\n\
}\n\n", uintType, uintType, uintType, requestCoordinate, requestBatch, outputOffset, shiftX, shiftY, shiftZ, shiftCoordinate, shiftBatch);
			break;

		}
		case 5: {//single_r2c
			char outputOffset[30] = "";
			if (sc.outputOffset > 0)
				sprintf(outputOffset, "%d + ", sc.outputOffset);
			char shiftX[100] = "";
			if (sc.outputStride[0] == 1)
				sprintf(shiftX, "index_x");
			else
				sprintf(shiftX, "index_x * %d", sc.outputStride[0]);
			char shiftY[100] = "";
			sprintf(shiftY, " + index_y * %d", sc.outputStride[1]);
			char shiftZ[100] = "";
			if (sc.size[2] > 1) {
				if (sc.performWorkGroupShift[2])
					sprintf(shiftZ, " + (gl_GlobalInvocationID.z + consts.workGroupShiftZ * gl_WorkGroupSize.z) * %d", sc.outputStride[2]);
				else
					sprintf(shiftZ, " + gl_GlobalInvocationID.z * %d", sc.outputStride[2]);
			}
			char shiftCoordinate[100] = "";
			if (sc.numCoordinates * sc.matrixConvolution > 1) {
				sprintf(shiftCoordinate, " + consts.coordinate * %d", sc.outputStride[3]);
			}
			char shiftBatch[100] = "";
			if ((sc.numBatches > 1) || (sc.numKernels > 1)) {
				sprintf(shiftBatch, " + consts.batchID * %d", sc.outputStride[4]);
			}
			sprintf(output + strlen(output), "\
%s indexOutput(%s index_x, %s index_y) {\n\
	return %s%s%s%s%s%s;\n\
}\n\n", uintType, uintType, uintType, outputOffset, shiftX, shiftY, shiftZ, shiftCoordinate, shiftBatch);
			break;
		}
		case 6: {//single_c2r
			char outputOffset[30] = "";
			if (sc.outputOffset > 0)
				sprintf(outputOffset, "%d + ", sc.outputOffset);
			char shiftX[100] = "";
			if (sc.outputStride[0] == 1)
				sprintf(shiftX, "index");
			else
				sprintf(shiftX, "index * %d", sc.outputStride[0]);
			char shiftY[100] = "";
			if (sc.size[1] > 1) {
				if (sc.fftDim == sc.fft_dim_full) {
					if (sc.performWorkGroupShift[1])
						sprintf(shiftY, " + (gl_WorkGroupID.y + consts.workGroupShiftY) * %d", 2 * sc.localSize[1] * sc.outputStride[1]);
					else
						sprintf(shiftY, " + gl_WorkGroupID.y * %d", 2 * sc.localSize[1] * sc.outputStride[1]);
				}
				else {
					if (sc.performWorkGroupShift[1])
						sprintf(shiftY, " + (gl_WorkGroupID.y + consts.workGroupShiftY) * %d", 2 * sc.outputStride[1]);
					else
						sprintf(shiftY, " + gl_WorkGroupID.y * %d", 2 * sc.outputStride[1]);
				}
			}
			char shiftZ[100] = "";
			if (sc.size[2] > 1) {
				if (sc.performWorkGroupShift[2])
					sprintf(shiftZ, " + (gl_GlobalInvocationID.z + consts.workGroupShiftZ * gl_WorkGroupSize.z) * %d", 2 * sc.outputStride[2]);
				else
					sprintf(shiftZ, " + gl_GlobalInvocationID.z * %d", 2 * sc.outputStride[2]);
			}
			char shiftCoordinate[100] = "";
			if (sc.numCoordinates * sc.matrixConvolution > 1) {
				sprintf(shiftCoordinate, " + consts.coordinate * %d", 2 * sc.outputStride[3]);
			}
			char shiftBatch[100] = "";
			if ((sc.numBatches > 1) || (sc.numKernels > 1)) {
				sprintf(shiftBatch, " + consts.batchID * %d", 2 * sc.outputStride[4]);
			}
			sprintf(output + strlen(output), "\
%s indexOutput(%s index) {\n\
	return %s%s%s%s%s%s;\n\
}\n\n", uintType, uintType, outputOffset, shiftX, shiftY, shiftZ, shiftCoordinate, shiftBatch);
			break;
		}
		}
	}

	static inline void inlineRadixKernelVkFFT(char* output, VkFFTSpecializationConstantsLayout sc, const char* floatType, const char* uintType, uint32_t radix, double stageAngle, uint32_t* regID) {
		char vecType[10];
		if (!strcmp(floatType, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "dvec2");
		char convolutionInverse[30] = "";
		if (sc.convolutionStep) sprintf(convolutionInverse, ", uint inverse");
		switch (radix) {
		case 2: {
			/*if (sc.LUT) {
				sprintf(output + strlen(output), "void radix2(inout %s temp_0, inout %s temp_1, %s LUTId) {\n", vecType, vecType, uintType);
			}
			else {
				sprintf(output + strlen(output), "void radix2(inout %s temp_0, inout %s temp_1, %s angle) {\n", vecType, vecType, floatType);
			}*/
			sprintf(output + strlen(output), "	{\n\
	%s temp;\n", vecType);
			if (sc.LUT)
				sprintf(output + strlen(output), "	%s w = twiddleLUT[LUTId];\n\n", vecType);
			else {
				if (!strcmp(floatType, "float"))
					sprintf(output + strlen(output), "	%s w = %s(cos(angle), sin(angle));\n\n", vecType, vecType);
				if (!strcmp(floatType, "double"))
					sprintf(output + strlen(output), "	%s w = sincos_20(angle);\n", vecType);
			}
			sprintf(output + strlen(output), "\
	temp.x = temp_%d.x * w.x - temp_%d.y * w.y;\n\
	temp.y = temp_%d.y * w.x + temp_%d.x * w.y;\n\
	temp_%d = temp_%d - temp;\n\
	temp_%d = temp_%d + temp;\n\
}\n", regID[1], regID[1], regID[1], regID[1], regID[1], regID[0], regID[0], regID[0]);
			break;
		}
		case 3: {
			/*	if (sc.LUT) {
					sprintf(output + strlen(output), "void radix3(inout %s temp_0, inout %s temp_1, inout %s temp_2, %s LUTId) {\n", vecType, vecType, vecType, uintType);
				}
				else {
					sprintf(output + strlen(output), "void radix3(inout %s temp_0, inout %s temp_1, inout %s temp_2, %s angle) {\n", vecType, vecType, vecType, floatType);
				}*/
			sprintf(output + strlen(output), "	{\n\
	%s loc_0;\n	%s loc_1;\n	%s loc_2;\n", vecType, vecType, vecType);
			if (sc.LUT)
				sprintf(output + strlen(output), "	%s w = twiddleLUT[LUTId];\n\n", vecType);
			else {
				if (!strcmp(floatType, "float"))
					sprintf(output + strlen(output), "	%s w = %s(cos(angle*%.17f), sin(angle*%.17f));\n\n", vecType, vecType, 4.0 / 3.0, 4.0 / 3.0);
				if (!strcmp(floatType, "double"))
					sprintf(output + strlen(output), "	%s w = sincos_20(angle*%.17f);\n", vecType, 4.0 / 3.0);
			}
			sprintf(output + strlen(output), "\
	loc_2.x = temp_%d.x * w.x - temp_%d.y * w.y;\n\
	loc_2.y = temp_%d.y * w.x + temp_%d.x * w.y;\n", regID[2], regID[2], regID[2], regID[2]);
			if (sc.LUT)
				sprintf(output + strlen(output), "	w = twiddleLUT[LUTId];\n\n");
			else {
				if (!strcmp(floatType, "float"))
					sprintf(output + strlen(output), "	w = %s(cos(angle*%.17f), sin(angle*%.17f));\n\n", vecType, 2.0 / 3.0, 2.0 / 3.0);
				if (!strcmp(floatType, "double"))
					sprintf(output + strlen(output), "	w=sincos_20(angle*%.17f);\n", 2.0 / 3.0);
			}
			sprintf(output + strlen(output), "\
	loc_1.x = temp_%d.x * w.x - temp_%d.y * w.y;\n\
	loc_1.y = temp_%d.y * w.x + temp_%d.x * w.y;\n", regID[1], regID[1], regID[1], regID[1]);
			sprintf(output + strlen(output), "\
	temp_%d = loc_1 + loc_2;\n\
	temp_%d = loc_1 - loc_2;\n", regID[1], regID[2]);
			sprintf(output + strlen(output), "\
	loc_0 = temp_%d + temp_%d;\n\
	loc_1 = temp_%d - 0.5 * temp_%d;\n\
	loc_2 = -0.8660254037844386467637231707529 * temp_%d;\n\
	temp_%d = loc_0;\n", regID[0], regID[1], regID[0], regID[1], regID[2], regID[0]);

			if (stageAngle < 0)
			{
				sprintf(output + strlen(output), "\
	temp_%d.x = loc_1.x - loc_2.y; \n\
	temp_%d.y = loc_1.y + loc_2.x; \n\
	temp_%d.x = loc_1.x + loc_2.y; \n\
	temp_%d.y = loc_1.y - loc_2.x; \n", regID[1], regID[1], regID[2], regID[2]);
			}
			else {
				sprintf(output + strlen(output), "\
	temp_%d.x = loc_1.x + loc_2.y; \n\
	temp_%d.y = loc_1.y - loc_2.x; \n\
	temp_%d.x = loc_1.x - loc_2.y; \n\
	temp_%d.y = loc_1.y + loc_2.x; \n", regID[1], regID[1], regID[2], regID[2]);
			}

			sprintf(output + strlen(output), "\
}\n");
			break;
		}
		case 4: {
			/*if (sc.LUT)
				sprintf(output + strlen(output), "void radix4(inout %s temp_0, inout %s temp_1, inout %s temp_2, inout %s temp_3, %s LUTId%s) {\n", vecType, vecType, vecType, vecType, uintType, convolutionInverse);
			else
				sprintf(output + strlen(output), "void radix4(inout %s temp_0, inout %s temp_1, inout %s temp_2, inout %s temp_3, %s angle%s) {\n", vecType, vecType, vecType, vecType, floatType, convolutionInverse);
			*/
			sprintf(output + strlen(output), "\
	//DIF 1st stage with double angle\n\
	{\n\
	%s temp;\n", vecType);
			if (sc.LUT)
				sprintf(output + strlen(output), "	%s w = twiddleLUT[LUTId];\n\n", vecType);
			else {
				if (!strcmp(floatType, "float"))
					sprintf(output + strlen(output), "	%s w = %s(cos(angle), sin(angle));\n\n", vecType, vecType);
				if (!strcmp(floatType, "double"))
					sprintf(output + strlen(output), "	%s w = sincos_20(angle);\n", vecType);
			}
			sprintf(output + strlen(output), "\
	temp.x=temp_%d.x*w.x-temp_%d.y*w.y;\n\
	temp.y = temp_%d.y * w.x + temp_%d.x * w.y;\n\
	temp_%d = temp_%d - temp;\n\
	temp_%d = temp_%d + temp;\n\n\
	temp.x=temp_%d.x*w.x-temp_%d.y*w.y;\n\
	temp.y = temp_%d.y * w.x + temp_%d.x * w.y;\n\
	temp_%d = temp_%d - temp;\n\
	temp_%d = temp_%d + temp;\n\n\
	//DIF 2nd stage with angle\n", regID[2], regID[2], regID[2], regID[2], regID[2], regID[0], regID[0], regID[0], regID[3], regID[3], regID[3], regID[3], regID[3], regID[1], regID[1], regID[1]);
			if (sc.LUT)
				sprintf(output + strlen(output), "	w=twiddleLUT[LUTId+%d];\n\n", sc.maxStageSumLUT);
			else {
				if (!strcmp(floatType, "float"))
					sprintf(output + strlen(output), "	w = %s(cos(0.5*angle), sin(0.5*angle));\n\n", vecType);
				if (!strcmp(floatType, "double"))
					sprintf(output + strlen(output), "	w=normalize(w + %s(1.0, 0.0));\n", vecType);
			}
			sprintf(output + strlen(output), "\
	temp.x = temp_%d.x * w.x - temp_%d.y * w.y;\n\
	temp.y = temp_%d.y * w.x + temp_%d.x * w.y;\n\
	temp_%d = temp_%d - temp;\n\
	temp_%d = temp_%d + temp;\n\n", regID[1], regID[1], regID[1], regID[1], regID[1], regID[0], regID[0], regID[0]);
			if (stageAngle < 0)
				sprintf(output + strlen(output), "	w = %s(w.y, -w.x);\n\n", vecType);
			else
				sprintf(output + strlen(output), "	w = %s(-w.y, w.x);\n\n", vecType);
			sprintf(output + strlen(output), "\
	temp.x = temp_%d.x * w.x - temp_%d.y * w.y;\n\
	temp.y = temp_%d.y * w.x + temp_%d.x * w.y;\n\
	temp_%d = temp_%d - temp;\n\
	temp_%d = temp_%d + temp;\n\n\
	temp = temp_%d;\n\
	temp_%d = temp_%d;\n\
	temp_%d = temp;\n\
}\n", regID[3], regID[3], regID[3], regID[3], regID[3], regID[2], regID[2], regID[2], regID[1], regID[1], regID[2], regID[2]);
			break;
		}
		case 5: {
			/*if (sc.LUT) {
				sprintf(output + strlen(output), "void radix5(inout %s temp_0, inout %s temp_1, inout %s temp_2, inout %s temp_3, inout %s temp_4, %s LUTId) {\n", vecType, vecType, vecType, vecType, vecType, uintType);
			}
			else {
				sprintf(output + strlen(output), "void radix5(inout %s temp_0, inout %s temp_1, inout %s temp_2, inout %s temp_3, inout %s temp_4, %s angle) {\n", vecType, vecType, vecType, vecType, vecType, floatType);
			}*/
			sprintf(output + strlen(output), "	{\n\
	%s loc_0;\n	%s loc_1;\n	%s loc_2;\n	%s loc_3;\n	%s loc_4;\n", vecType, vecType, vecType, vecType, vecType);
			for (uint32_t i = radix - 1; i > 0; i--) {
				if (i == radix - 1) {
					if (sc.LUT)
						sprintf(output + strlen(output), "	%s w = twiddleLUT[LUTId];\n\n", vecType);
					else {
						if (!strcmp(floatType, "float"))
							sprintf(output + strlen(output), "	%s w = %s(cos(angle*%.17f), sin(angle*%.17f));\n\n", vecType, vecType, 2.0 * i / radix, 2.0 * i / radix);
						if (!strcmp(floatType, "double"))
							sprintf(output + strlen(output), "	%s w = sincos_20(angle*%.17f);\n", vecType, 2.0 * i / radix);
					}
				}
				else {
					if (sc.LUT)
						sprintf(output + strlen(output), "	w = twiddleLUT[LUTId];\n\n");
					else {
						if (!strcmp(floatType, "float"))
							sprintf(output + strlen(output), "	w = %s(cos(angle*%.17f), sin(angle*%.17f));\n\n", vecType, 2.0 * i / radix, 2.0 * i / radix);
						if (!strcmp(floatType, "double"))
							sprintf(output + strlen(output), "	w = sincos_20(angle*%.17f);\n", 2.0 * i / radix);
					}
				}
				sprintf(output + strlen(output), "\
	loc_%d.x = temp_%d.x * w.x - temp_%d.y * w.y;\n\
	loc_%d.y = temp_%d.y * w.x + temp_%d.x * w.y;\n", i, regID[i], regID[i], i, regID[i], regID[i]);
			}
			sprintf(output + strlen(output), "\
	temp_%d = loc_1 + loc_4;\n\
	temp_%d = loc_2 + loc_3;\n\
	temp_%d = loc_2 - loc_3;\n\
	temp_%d = loc_1 - loc_4;\n\
	loc_3 = temp_%d - temp_%d;\n\
	loc_4 = temp_%d + temp_%d;\n", regID[1], regID[2], regID[3], regID[4], regID[1], regID[2], regID[3], regID[4]);
			sprintf(output + strlen(output), "\
	loc_0 = temp_%d + temp_%d + temp_%d;\n\
	loc_1 = temp_%d - 0.5 * temp_%d;\n\
	loc_2 = temp_%d - 0.5 * temp_%d;\n\
	temp_%d *= 1.538841768587626701285145288018455;\n\
	temp_%d *= -0.363271264002680442947733378740309;\n\
	loc_3 *= -0.809016994374947424102293417182819;\n\
	loc_4 *= -0.587785252292473129168705954639073;\n", regID[0], regID[1], regID[2], regID[0], regID[1], regID[0], regID[2], regID[3], regID[4]);
			sprintf(output + strlen(output), "\
	loc_1 -= loc_3;\n\
	loc_2 += loc_3;\n\
	loc_3 = temp_%d+loc_4;\n\
	loc_4 += temp_%d;\n\
	temp_%d = loc_0;\n", regID[3], regID[4], regID[0]);

			if (stageAngle < 0)
			{
				sprintf(output + strlen(output), "\
	temp_%d.x = loc_1.x - loc_4.y; \n\
	temp_%d.y = loc_1.y + loc_4.x; \n\
	temp_%d.x = loc_2.x - loc_3.y; \n\
	temp_%d.y = loc_2.y + loc_3.x; \n\
	temp_%d.x = loc_2.x + loc_3.y; \n\
	temp_%d.y = loc_2.y - loc_3.x; \n\
	temp_%d.x = loc_1.x + loc_4.y; \n\
	temp_%d.y = loc_1.y - loc_4.x; \n", regID[1], regID[1], regID[2], regID[2], regID[3], regID[3], regID[4], regID[4]);
			}
			else {
				sprintf(output + strlen(output), "\
	temp_%d.x = loc_1.x + loc_4.y; \n\
	temp_%d.y = loc_1.y - loc_4.x; \n\
	temp_%d.x = loc_2.x + loc_3.y; \n\
	temp_%d.y = loc_2.y - loc_3.x; \n\
	temp_%d.x = loc_2.x - loc_3.y; \n\
	temp_%d.y = loc_2.y + loc_3.x; \n\
	temp_%d.x = loc_1.x - loc_4.y; \n\
	temp_%d.y = loc_1.y + loc_4.x; \n", regID[1], regID[1], regID[2], regID[2], regID[3], regID[3], regID[4], regID[4]);
			}


			sprintf(output + strlen(output), "\
}\n\n");
			break;
		}
		case 8: {
			/*if (sc.LUT)
				sprintf(output + strlen(output), "void radix8(inout %s temp_0, inout %s temp_1, inout %s temp_2, inout %s temp_3, inout %s temp_4, inout %s temp_5, inout %s temp_6, inout %s temp_7, %s LUTId%s) {\n", vecType, vecType, vecType, vecType, vecType, vecType, vecType, vecType, uintType, convolutionInverse);
			else
				sprintf(output + strlen(output), "void radix8(inout %s temp_0, inout %s temp_1, inout %s temp_2, inout %s temp_3, inout %s temp_4, inout %s temp_5, inout %s temp_6, inout %s temp_7, %s angle%s) {\n", vecType, vecType, vecType, vecType, vecType, vecType, vecType, vecType, floatType, convolutionInverse);
			*/
			sprintf(output + strlen(output), "\
	//DIF 1st stage with quadruple angle\n\
	{\n\
	%s temp;\n", vecType);
			if (sc.LUT)
				sprintf(output + strlen(output), "	%s w = twiddleLUT[LUTId];\n\n", vecType);
			else {
				if (!strcmp(floatType, "float"))
					sprintf(output + strlen(output), "	%s w = %s(cos(angle), sin(angle));\n\n", vecType, vecType);
				if (!strcmp(floatType, "double"))
					sprintf(output + strlen(output), "	%s w = sincos_20(angle);\n", vecType);
			}
			for (uint32_t i = 0; i < 4; i++) {
				sprintf(output + strlen(output), "\
	temp.x=temp_%d.x*w.x-temp_%d.y*w.y;\n\
	temp.y = temp_%d.y * w.x + temp_%d.x * w.y;\n\
	temp_%d = temp_%d - temp;\n\
	temp_%d = temp_%d + temp;\n\n", regID[i + 4], regID[i + 4], regID[i + 4], regID[i + 4], regID[i + 4], regID[i + 0], regID[i + 0], regID[i + 0]);
			}
			if (sc.LUT)
				sprintf(output + strlen(output), "	w=twiddleLUT[LUTId+%d];\n\n", sc.maxStageSumLUT);
			else {
				if (!strcmp(floatType, "float"))
					sprintf(output + strlen(output), "	w = %s(cos(0.5*angle), sin(0.5*angle));\n\n", vecType);
				if (!strcmp(floatType, "double"))
					sprintf(output + strlen(output), "	w=normalize(w + %s(1.0, 0.0));\n", vecType);
			}
			for (uint32_t i = 0; i < 2; i++) {
				sprintf(output + strlen(output), "\
	temp.x=temp_%d.x*w.x-temp_%d.y*w.y;\n\
	temp.y = temp_%d.y * w.x + temp_%d.x * w.y;\n\
	temp_%d = temp_%d - temp;\n\
	temp_%d = temp_%d + temp;\n\n", regID[i + 2], regID[i + 2], regID[i + 2], regID[i + 2], regID[i + 2], regID[i + 0], regID[i + 0], regID[i + 0]);
			}
			if (stageAngle < 0)
				sprintf(output + strlen(output), "	%s iw = %s(w.y, -w.x);\n\n", vecType, vecType);
			else
				sprintf(output + strlen(output), "	%s iw = %s(-w.y, w.x);\n\n", vecType, vecType);

			for (uint32_t i = 4; i < 6; i++) {
				sprintf(output + strlen(output), "\
	temp.x = temp_%d.x * iw.x - temp_%d.y * iw.y;\n\
	temp.y = temp_%d.y * iw.x + temp_%d.x * iw.y;\n\
	temp_%d = temp_%d - temp;\n\
	temp_%d = temp_%d + temp;\n\n", regID[i + 2], regID[i + 2], regID[i + 2], regID[i + 2], regID[i + 2], regID[i + 0], regID[i + 0], regID[i + 0]);
			}

			if (sc.LUT)
				sprintf(output + strlen(output), "	w=twiddleLUT[LUTId+%d];\n\n", 2 * sc.maxStageSumLUT);
			else {
				if (!strcmp(floatType, "float"))
					sprintf(output + strlen(output), "	w = %s(cos(0.25*angle), sin(0.25*angle));\n\n", vecType);
				if (!strcmp(floatType, "double"))
					sprintf(output + strlen(output), "	w=normalize(w + %s(1.0, 0.0));\n", vecType);
			}
			sprintf(output + strlen(output), "\
	temp.x=temp_%d.x*w.x-temp_%d.y*w.y;\n\
	temp.y = temp_%d.y * w.x + temp_%d.x * w.y;\n\
	temp_%d = temp_%d - temp;\n\
	temp_%d = temp_%d + temp;\n\n", regID[1], regID[1], regID[1], regID[1], regID[1], regID[0], regID[0], regID[0]);
			if (stageAngle < 0)
				sprintf(output + strlen(output), "	iw = %s(w.y, -w.x);\n\n", vecType);
			else
				sprintf(output + strlen(output), "	iw = %s(-w.y, w.x);\n\n", vecType);

			sprintf(output + strlen(output), "\
	temp.x = temp_%d.x * iw.x - temp_%d.y * iw.y;\n\
	temp.y = temp_%d.y * iw.x + temp_%d.x * iw.y;\n\
	temp_%d = temp_%d - temp;\n\
	temp_%d = temp_%d + temp;\n\n", regID[3], regID[3], regID[3], regID[3], regID[3], regID[2], regID[2], regID[2]);
			if (stageAngle < 0) {
				sprintf(output + strlen(output), "	iw.x = w.x * M_SQRT1_2 + w.y * M_SQRT1_2;\n");
				sprintf(output + strlen(output), "	iw.y = w.y * M_SQRT1_2 - w.x * M_SQRT1_2;\n\n");
			}
			else {
				sprintf(output + strlen(output), "	iw.x = w.x * M_SQRT1_2 - w.y * M_SQRT1_2;\n");
				sprintf(output + strlen(output), "	iw.y = w.y * M_SQRT1_2 + w.x * M_SQRT1_2;\n\n");
			}

			sprintf(output + strlen(output), "\
	temp.x = temp_%d.x * iw.x - temp_%d.y * iw.y;\n\
	temp.y = temp_%d.y * iw.x + temp_%d.x * iw.y;\n\
	temp_%d = temp_%d - temp;\n\
	temp_%d = temp_%d + temp;\n\n", regID[5], regID[5], regID[5], regID[5], regID[5], regID[4], regID[4], regID[4]);
			if (stageAngle < 0)
				sprintf(output + strlen(output), "	w = %s(iw.y, -iw.x);\n\n", vecType);
			else
				sprintf(output + strlen(output), "	w = %s(-iw.y, iw.x);\n\n", vecType);
			sprintf(output + strlen(output), "\
	temp.x = temp_%d.x * w.x - temp_%d.y * w.y;\n\
	temp.y = temp_%d.y * w.x + temp_%d.x * w.y;\n\
	temp_%d = temp_%d - temp;\n\
	temp_%d = temp_%d + temp;\n\n\
	temp = temp_%d;\n\
	temp_%d = temp_%d;\n\
	temp_%d = temp;\n\n\
	temp = temp_%d;\n\
	temp_%d = temp_%d;\n\
	temp_%d = temp;\n\
}\n\n", regID[7], regID[7], regID[7], regID[7], regID[7], regID[6], regID[6], regID[6], regID[1], regID[1], regID[4], regID[4], regID[3], regID[3], regID[6], regID[6]);
			break;
		}
		}
	}
	static inline void appendSharedMemoryVkFFT(char* output, VkFFTSpecializationConstantsLayout sc, const char* floatType, const char* uintType, uint32_t sharedType) {
		char vecType[10];
		uint32_t maxSequenceSharedMemory = 0;
		uint32_t maxSequenceSharedMemoryPow2 = 0;
		if (!strcmp(floatType, "float"))
		{
			sprintf(vecType, "vec2");
			maxSequenceSharedMemory = sc.sharedMemSize / 8;
			maxSequenceSharedMemoryPow2 = sc.sharedMemSizePow2 / 8;
		}
		if (!strcmp(floatType, "double")) {
			sprintf(vecType, "dvec2");
			maxSequenceSharedMemory = sc.sharedMemSize / 16;
			maxSequenceSharedMemoryPow2 = sc.sharedMemSizePow2 / 16;
		}
		switch (sharedType) {
		case 0: case 5: case 6://single_c2c + single_r2c
		{
			uint32_t sharedStride = ((maxSequenceSharedMemory - sc.localSize[1] * sc.fftDim) * (sc.localSize[1] - 1) == 0) ? sc.fftDim : sc.fftDim + 1;
			sprintf(output + strlen(output), "const %s sharedStride = %d; //to avoid bank conflict if we transpose\n", uintType, sharedStride);
			sprintf(output + strlen(output), "shared %s sdata[%d];// sharedStride - fft size,  gl_WorkGroupSize.y - grouped consequential ffts\n\n", vecType, sc.localSize[1] * sharedStride);
			break;
		}
		case 1: case 2://grouped_c2c + single_c2c_strided
		{
			sprintf(output + strlen(output), "shared %s sdata[%d];\n\n", vecType, sc.localSize[0] * sc.fftDim);
			break;
		}
		case 3: case 4: //registerBoost
		{
			sprintf(output + strlen(output), "shared %s sdata[%d];\n\n", vecType, maxSequenceSharedMemoryPow2);
			break;
		}
		}
	}
	static inline void appendInitialization(char* output, VkFFTSpecializationConstantsLayout sc, const char* floatType, const char* uintType, uint32_t initType) {
		char vecType[10];
		if (!strcmp(floatType, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "dvec2");
		switch (initType) {
		case 0: case 1: case 2: case 5: case 6:
		{
			if (sc.convolutionStep) {
				for (uint32_t i = 0; i < sc.registers_per_thread; i++)
					sprintf(output + strlen(output), "	%s temp_%d;\n", vecType, i);
				for (uint32_t j = 1; j < sc.matrixConvolution; j++) {
					for (uint32_t i = 0; i < sc.registers_per_thread; i++)
						sprintf(output + strlen(output), "	%s temp_%d_%d;\n", vecType, i, j);
				}
			}
			else {
				for (uint32_t i = 0; i < sc.registers_per_thread; i++)
					sprintf(output + strlen(output), "	%s temp_%d;\n", vecType, i);
			}
			//sprintf(output + strlen(output), "	%s temp[8];\n", vecType);
			break;
		}
		case 3://registerBoost - 2x
		{
			for (uint32_t i = 0; i < 2 * sc.registers_per_thread; i++)
				sprintf(output + strlen(output), "	%s temp_%d;\n", vecType, i);
			sprintf(output + strlen(output), "	%s sort0;\n", vecType);
			break;
		}
		case 4://registerBoost - 4x
		{
			for (uint32_t i = 0; i < 4 * sc.registers_per_thread; i++)
				sprintf(output + strlen(output), "	%s temp_%d;\n", vecType, i);
			sprintf(output + strlen(output), "	%s sort0;\n", vecType);
			break;
		}
		}
		sprintf(output + strlen(output), "	%s stageInvocationID;\n", uintType);
		sprintf(output + strlen(output), "	%s blockInvocationID;\n", uintType);
		sprintf(output + strlen(output), "	%s combinedId;\n", uintType);
		sprintf(output + strlen(output), "	%s inoutID;\n", uintType);
		if (sc.LUT)
			sprintf(output + strlen(output), "	%s LUTId=0;\n", uintType);
		else
			sprintf(output + strlen(output), "	%s angle=0;\n", floatType);
		if (((sc.stageStartSize > 1) && (!((sc.stageStartSize > 1) && (!sc.reorderFourStep) && (sc.inverse)))) || ((!sc.reorderFourStep) && (sc.inverse))) {
			sprintf(output + strlen(output), "	%s mult=vec2(0,0);\n", vecType);
		}
		if (sc.cacheShuffle) {
			sprintf(output + strlen(output), "\
	uint tshuffle= ((gl_LocalInvocationID.x>>1))%%(%d);\n\
	%s shuffle[%d];\n", sc.registers_per_thread, vecType, sc.registers_per_thread);
			for (uint32_t i = 0; i < sc.registers_per_thread; i++) {
				sprintf(output + strlen(output), "\
	shuffle[%d]=%s(0,0);\n", i, vecType);
			}
		}

	}
	static inline void appendZeropadReturn(char* output, VkFFTSpecializationConstantsLayout sc) {
		//return if sequence is full of zeros from the start
		if (sc.inverse) {
			switch (sc.axis_id) {
			case 0: {
				break;
			}
			case 1: {
				if (!sc.supportAxis) {
					char idX[100] = "";
					if (sc.performWorkGroupShift[0])
						sprintf(idX, "(gl_GlobalInvocationID.x + consts.workGroupShiftX * gl_WorkGroupSize.x)");
					else
						sprintf(idX, "gl_GlobalInvocationID.x");
					if (sc.performZeropaddingInput[0])
						sprintf(output + strlen(output), "		if((%s >= %d)&&(%s < %d)) return;\n", idX, sc.fft_zeropad_left_read[0], idX, sc.fft_zeropad_right_read[0]);
				}
				break;
			}
			case 2: {
				if (!sc.supportAxis) {
					char idY[100] = "";
					if (sc.performWorkGroupShift[1])//y axis is along z workgroup here
						sprintf(idY, "(gl_GlobalInvocationID.z + consts.workGroupShiftZ * gl_WorkGroupSize.z)");
					else
						sprintf(idY, "gl_GlobalInvocationID.z");

					char idX[100] = "";
					if (sc.performWorkGroupShift[0])
						sprintf(idX, "(gl_GlobalInvocationID.x + consts.workGroupShiftX * gl_WorkGroupSize.x)");
					else
						sprintf(idX, "gl_GlobalInvocationID.x");
					if (sc.fft_zeropad_left_read[0] < sc.fft_zeropad_right_read[0]) {
						if (sc.performZeropaddingInput[0])
							sprintf(output + strlen(output), "		if((%s >= %d)&&(%s < %d)) return;\n", idX, sc.fft_zeropad_left_read[0], idX, sc.fft_zeropad_right_read[0]);
					}

					if (sc.fft_zeropad_left_read[1] < sc.fft_zeropad_right_read[1]) {
						if (sc.performZeropaddingInput[1])
							sprintf(output + strlen(output), "		if((%s >= %d)&&(%s < %d)) return;\n", idY, sc.fft_zeropad_left_read[1], idY, sc.fft_zeropad_right_read[1]);
					}
				}
				else {
					char idY[100] = "";
					if (sc.performWorkGroupShift[1])//for support axes y is along x workgroup
						sprintf(idY, "(gl_GlobalInvocationID.x + consts.workGroupShiftX * gl_WorkGroupSize.x)");
					else
						sprintf(idY, "gl_GlobalInvocationID.x");
					if (sc.fft_zeropad_left_read[1] < sc.fft_zeropad_right_read[1]) {
						if (sc.performZeropaddingInput[1])
							sprintf(output + strlen(output), "		if((%s >= %d)&&(%s < %d)) return;\n", idY, sc.fft_zeropad_left_read[1], idY, sc.fft_zeropad_right_read[1]);
					}
				}
				break;
			}
			}
		}
		else {
			switch (sc.axis_id) {
			case 0: {
				char idY[100] = "";
				if (sc.performWorkGroupShift[1])
					sprintf(idY, "(gl_GlobalInvocationID.y + consts.workGroupShiftY * gl_WorkGroupSize.y)");
				else
					sprintf(idY, "gl_GlobalInvocationID.y");

				char idZ[100] = "";
				if (sc.performWorkGroupShift[2])
					sprintf(idZ, "(gl_GlobalInvocationID.z + consts.workGroupShiftZ * gl_WorkGroupSize.z)");
				else
					sprintf(idZ, "gl_GlobalInvocationID.z");
				if (sc.fft_zeropad_left_read[1] < sc.fft_zeropad_right_read[1]) {
					if (sc.performZeropaddingInput[1])
						sprintf(output + strlen(output), "		if((%s >= %d)&&(%s < %d)) return;\n", idY, sc.fft_zeropad_left_read[1], idY, sc.fft_zeropad_right_read[1]);
				}
				if (sc.fft_zeropad_left_read[2] < sc.fft_zeropad_right_read[2]) {
					if (sc.performZeropaddingInput[2])
						sprintf(output + strlen(output), "		if((%s >= %d)&&(%s < %d)) return;\n", idZ, sc.fft_zeropad_left_read[2], idZ, sc.fft_zeropad_right_read[2]);
				}
				break;
			}
			case 1: {
				char idZ[100] = "";
				if (sc.performWorkGroupShift[2])
					sprintf(idZ, "(gl_GlobalInvocationID.z + consts.workGroupShiftZ * gl_WorkGroupSize.z)");
				else
					sprintf(idZ, "gl_GlobalInvocationID.z");
				if (sc.fft_zeropad_left_read[2] < sc.fft_zeropad_right_read[2]) {
					if (sc.performZeropaddingInput[2])
						sprintf(output + strlen(output), "		if((%s >= %d)&&(%s < %d)) return;\n", idZ, sc.fft_zeropad_left_read[2], idZ, sc.fft_zeropad_right_read[2]);
				}
				break;
			}
			case 2: {

				break;
			}
			}
		}
	}
	static inline void appendReadDataVkFFT(char* output, VkFFTSpecializationConstantsLayout sc, const char* floatType, const char* floatTypeMemory, const char* uintType, uint32_t readType) {
		char vecType[10];
		if (!strcmp(floatType, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "dvec2");
		char convTypeLeft[10] = "";
		char convTypeRight[10] = "";
		if ((!strcmp(floatType, "float")) && (strcmp(floatTypeMemory, "float"))) {
			if (readType == 5) {
				sprintf(convTypeLeft, "float(");
				sprintf(convTypeRight, ")");
			}
			else {
				sprintf(convTypeLeft, "vec2(");
				sprintf(convTypeRight, ")");
			}
		}
		if ((!strcmp(floatType, "double")) && (strcmp(floatTypeMemory, "double"))) {
			if (readType == 5) {
				sprintf(convTypeLeft, "double(");
				sprintf(convTypeRight, ")");
			}
			else {
				sprintf(convTypeLeft, "dvec2(");
				sprintf(convTypeRight, ")");
			}
		}
		char requestCoordinate[100] = "";
		if (sc.convolutionStep) {
			if (sc.matrixConvolution > 1) {
				sprintf(requestCoordinate, ", coordinate");
			}
		}
		char requestBatch[100] = "";
		if (sc.convolutionStep) {
			if (sc.numKernels > 1) {
				sprintf(requestBatch, ", 0");//if one buffer - multiple kernel convolution
			}
		}

		switch (readType) {
		case 0://single_c2c
		{
			//sprintf(output + strlen(output), "	return;\n");
			char shiftX[100] = "";
			if (sc.performWorkGroupShift[0])
				sprintf(shiftX, " + consts.workGroupShiftX ");
			char shiftY[100] = "";
			if (sc.performWorkGroupShift[1])
				sprintf(shiftY, " + consts.workGroupShiftY*gl_WorkGroupSize.y ");
			char shiftY2[100] = "";
			if (sc.performWorkGroupShift[1])
				sprintf(shiftY, " + consts.workGroupShiftY ");
			if (sc.fftDim < sc.fft_dim_full) {
				sprintf(output + strlen(output), "		if(gl_LocalInvocationID.y * %d + (((gl_WorkGroupID.x%s) %% %d) * %d + ((gl_WorkGroupID.x%s) / %d) * %d) < %d) {\n", sc.firstStageStartSize, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.localSize[1] * sc.firstStageStartSize, sc.fft_dim_full);
			}
			else {
				sprintf(output + strlen(output), "		{ \n");
			}

			if ((sc.localSize[1] > 1) || ((sc.performR2C) && (sc.inverse)) || (sc.localSize[0] * sc.stageRadix[0] * (sc.registers_per_thread / sc.stageRadix[0]) > sc.fftDim))
				sc.readToRegisters = 0;
			else
				sc.readToRegisters = 1;
			if (sc.zeropad[0]) {
				if (sc.fftDim == sc.fft_dim_full) {
					for (uint32_t i = 0; i < sc.min_registers_per_thread; i++) {

						if (sc.localSize[1] == 1)
							sprintf(output + strlen(output), "		combinedId = gl_LocalInvocationID.x + %d;\n", i * sc.localSize[0]);
						else
							sprintf(output + strlen(output), "		combinedId = (gl_LocalInvocationID.x + %d * gl_LocalInvocationID.y) + %d;\n", sc.localSize[0], i * sc.localSize[0] * sc.localSize[1]);

						if (sc.inputStride[0] > 1)
							sprintf(output + strlen(output), "		inoutID = (combinedId %% %d) * %d + (combinedId / %d) * %d;\n", sc.fftDim, sc.inputStride[0], sc.fftDim, sc.inputStride[1]);
						else
							sprintf(output + strlen(output), "		inoutID = (combinedId %% %d) + (combinedId / %d) * %d;\n", sc.fftDim, sc.fftDim, sc.inputStride[1]);

						if (sc.size[sc.axis_id + 1] % sc.localSize[1] != 0)
							sprintf(output + strlen(output), "		if(combinedId / %d + (gl_WorkGroupID.y%s)*gl_WorkGroupSize.y< %d){", sc.fftDim, shiftY2, sc.size[sc.axis_id + 1]);

						sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d)){\n", sc.fft_dim_full, sc.fft_zeropad_left_read[sc.axis_id], sc.fft_dim_full, sc.fft_zeropad_right_read[sc.axis_id]);
						if (sc.readToRegisters) {
							if (sc.inputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		temp_%d = %sinputBlocks[0].inputs[indexInput(inoutID%s%s)]%s;\n", i, convTypeLeft, requestCoordinate, requestBatch, convTypeRight);
							else
								sprintf(output + strlen(output), "		temp_%d = %sinputBlocks[indexInput(inoutID%s%s) / %d].inputs[indexInput(inoutID%s%s) %% %d]%s;\n", i, convTypeLeft, requestCoordinate, requestBatch, sc.inputBufferBlockSize, requestCoordinate, requestBatch, sc.inputBufferBlockSize, convTypeRight);
						}
						else {
							if (sc.inputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		sdata[(combinedId %% %d) + (combinedId / %d) * sharedStride] = %sinputBlocks[0].inputs[indexInput(inoutID%s%s)]%s;\n", sc.fftDim, sc.fftDim, convTypeLeft, requestCoordinate, requestBatch, convTypeRight);
							else
								sprintf(output + strlen(output), "		sdata[(combinedId %% %d) + (combinedId / %d) * sharedStride] = %sinputBlocks[indexInput(inoutID%s%s) / %d].inputs[indexInput(inoutID%s%s) %% %d]%s;\n", sc.fftDim, sc.fftDim, convTypeLeft, requestCoordinate, requestBatch, sc.inputBufferBlockSize, requestCoordinate, requestBatch, sc.inputBufferBlockSize, convTypeRight);

						}
						sprintf(output + strlen(output), "		}else{\n");
						if (sc.localSize[1] == 1)
							sprintf(output + strlen(output), "			temp_%d = %s(0,0);\n", i, vecType);
						else
							sprintf(output + strlen(output), "			sdata[(combinedId %% %d) + (combinedId / %d) * sharedStride] = %s(0,0);\n", sc.fftDim, sc.fftDim, vecType);
						sprintf(output + strlen(output), "		}\n");
						if (sc.size[sc.axis_id + 1] % sc.localSize[1] != 0)
							sprintf(output + strlen(output), "		}");

					}

				}
				else {
					for (uint32_t i = 0; i < sc.min_registers_per_thread; i++) {
						sprintf(output + strlen(output), "		inoutID = gl_LocalInvocationID.x+%d+gl_LocalInvocationID.y * %d + (((gl_WorkGroupID.x%s) %% %d) * %d + ((gl_WorkGroupID.x%s) / %d) * %d);\n", i * sc.localSize[0], sc.firstStageStartSize, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.localSize[1] * sc.firstStageStartSize);
						sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d)){\n", sc.fft_dim_full, sc.fft_zeropad_left_read[sc.axis_id], sc.fft_dim_full, sc.fft_zeropad_right_read[sc.axis_id]);
						if (sc.readToRegisters) {
							if (sc.inputBufferBlockNum == 1)
								sprintf(output + strlen(output), "			temp_%d = %sinputBlocks[0].inputs[indexInput(inoutID%s%s)]%s;\n", i, convTypeLeft, requestCoordinate, requestBatch, convTypeRight);
							else
								sprintf(output + strlen(output), "			temp_%d = %sinputBlocks[indexInput(inoutID%s%s) / %d].inputs[indexInput(inoutID%s%s) %% %d]%s;\n", i, convTypeLeft, requestCoordinate, requestBatch, sc.inputBufferBlockSize, requestCoordinate, requestBatch, sc.inputBufferBlockSize, convTypeRight);
						}
						else {
							if (sc.inputBufferBlockNum == 1)
								sprintf(output + strlen(output), "			sdata[sharedStride*gl_LocalInvocationID.y + (gl_LocalInvocationID.x + %d)] = %sinputBlocks[0].inputs[indexInput(inoutID%s%s)]%s;\n", i * sc.localSize[0], convTypeLeft, requestCoordinate, requestBatch, convTypeRight);
							else
								sprintf(output + strlen(output), "			sdata[sharedStride*gl_LocalInvocationID.y + (gl_LocalInvocationID.x + %d)] = %sinputBlocks[indexInput(inoutID%s%s) / %d].inputs[indexInput(inoutID%s%s) %% %d]%s;\n", i * sc.localSize[0], convTypeLeft, requestCoordinate, requestBatch, sc.inputBufferBlockSize, requestCoordinate, requestBatch, sc.inputBufferBlockSize, convTypeRight);
						}
						sprintf(output + strlen(output), "		}\n");
						sprintf(output + strlen(output), "		else\n");
						if (sc.localSize[1] == 1)
							sprintf(output + strlen(output), "			temp_%d = %s(0,0);\n", i, vecType);
						else
							sprintf(output + strlen(output), "			sdata[sharedStride*gl_LocalInvocationID.y + (gl_LocalInvocationID.x + %d)] = %s(0,0);\n", i * sc.localSize[0], vecType);
					}
				}
			}
			else {
				if (sc.fftDim == sc.fft_dim_full) {
					for (uint32_t i = 0; i < sc.min_registers_per_thread; i++) {
						if (sc.localSize[1] == 1)
							sprintf(output + strlen(output), "		combinedId = gl_LocalInvocationID.x + %d;\n", i * sc.localSize[0]);
						else
							sprintf(output + strlen(output), "		combinedId = (gl_LocalInvocationID.x + %d * gl_LocalInvocationID.y) + %d;\n", sc.localSize[0], i * sc.localSize[0] * sc.localSize[1]);

						if (sc.inputStride[0] > 1)
							sprintf(output + strlen(output), "		inoutID = indexInput((combinedId %% %d) * %d + (combinedId / %d) * %d%s%s);\n", sc.fftDim, sc.inputStride[0], sc.fftDim, sc.inputStride[1], requestCoordinate, requestBatch);
						else
							sprintf(output + strlen(output), "		inoutID = indexInput((combinedId %% %d) + (combinedId / %d) * %d%s%s);\n", sc.fftDim, sc.fftDim, sc.inputStride[1], requestCoordinate, requestBatch);
						if (sc.size[sc.axis_id + 1] % sc.localSize[1] != 0)
							sprintf(output + strlen(output), "		if(combinedId / %d + (gl_WorkGroupID.y%s)*gl_WorkGroupSize.y< %d){", sc.fftDim, shiftY2, sc.size[sc.axis_id + 1]);
						if (sc.readToRegisters) {
							if (sc.inputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		temp_%d = %sinputBlocks[0].inputs[inoutID]%s;\n", i, convTypeLeft, convTypeRight);
							else
								sprintf(output + strlen(output), "		temp_%d = %sinputBlocks[inoutID / %d].inputs[inoutID %% %d]%s;\n", i, convTypeLeft, sc.inputBufferBlockSize, sc.inputBufferBlockSize, convTypeRight);
						}
						else {
							if (sc.inputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		sdata[(combinedId %% %d) + (combinedId / %d) * sharedStride] = %sinputBlocks[0].inputs[inoutID]%s;\n", sc.fftDim, sc.fftDim, convTypeLeft, convTypeRight);
							else
								sprintf(output + strlen(output), "		sdata[(combinedId %% %d) + (combinedId / %d) * sharedStride] = %sinputBlocks[inoutID / %d].inputs[inoutID %% %d]%s;\n", sc.fftDim, sc.fftDim, convTypeLeft, sc.inputBufferBlockSize, sc.inputBufferBlockSize, convTypeRight);
						}
						if (sc.size[sc.axis_id + 1] % sc.localSize[1] != 0)
							sprintf(output + strlen(output), "		}");
					}

				}
				else {
					for (uint32_t i = 0; i < sc.min_registers_per_thread; i++) {
						sprintf(output + strlen(output), "		inoutID = indexInput(gl_LocalInvocationID.x+%d+gl_LocalInvocationID.y * %d + (((gl_WorkGroupID.x%s) %% %d) * %d + ((gl_WorkGroupID.x%s) / %d) * %d)%s%s);\n", i * sc.localSize[0], sc.firstStageStartSize, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.localSize[1] * sc.firstStageStartSize, requestCoordinate, requestBatch);
						if (sc.readToRegisters) {
							if (sc.inputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		temp_%d = %sinputBlocks[0].inputs[inoutID]%s;\n", i, convTypeLeft, convTypeRight);
							else
								sprintf(output + strlen(output), "		temp_%d = %sinputBlocks[inoutID / %d].inputs[inoutID %% %d]%s;\n", i, convTypeLeft, sc.inputBufferBlockSize, sc.inputBufferBlockSize, convTypeRight);
						}
						else {
							if (sc.inputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		sdata[sharedStride*gl_LocalInvocationID.y + (gl_LocalInvocationID.x + %d)] = %sinputBlocks[0].inputs[inoutID]%s;\n", i * sc.localSize[0], convTypeLeft, convTypeRight);
							else
								sprintf(output + strlen(output), "		sdata[sharedStride*gl_LocalInvocationID.y + (gl_LocalInvocationID.x + %d)] = %sinputBlocks[inoutID / %d].inputs[inoutID %% %d]%s;\n", i * sc.localSize[0], convTypeLeft, sc.inputBufferBlockSize, sc.inputBufferBlockSize, convTypeRight);
						}
					}
				}
			}
			sprintf(output + strlen(output), "	}\n");
			break;
		}
		case 1://grouped_c2c
		{
			if (sc.localSize[1] * sc.stageRadix[0] * (sc.registers_per_thread / sc.stageRadix[0]) > sc.fftDim)
				sc.readToRegisters = 0;
			else
				sc.readToRegisters = 1;
			char shiftX[100] = "";
			if (sc.performWorkGroupShift[0])
				sprintf(shiftX, " + consts.workGroupShiftX * gl_WorkGroupSize.x ");
			sprintf(output + strlen(output), "		if (((gl_GlobalInvocationID.x%s) / %d) %% (%d)+((gl_GlobalInvocationID.x%s) / %d) * (%d) < %d) {;\n", shiftX, sc.fft_dim_x, sc.stageStartSize, shiftX, sc.fft_dim_x* sc.stageStartSize, sc.fftDim* sc.stageStartSize, sc.size[sc.axis_id]);


			if (sc.zeropad[0]) {
				for (uint32_t i = 0; i < sc.min_registers_per_thread; i++) {
					sprintf(output + strlen(output), "		inoutID = (%d * (gl_LocalInvocationID.y + %d) + ((gl_GlobalInvocationID.x%s) / %d) %% (%d)+((gl_GlobalInvocationID.x%s) / %d) * (%d));\n", sc.stageStartSize, i * sc.localSize[1], shiftX, sc.fft_dim_x, sc.stageStartSize, shiftX, sc.fft_dim_x * sc.stageStartSize, sc.fftDim * sc.stageStartSize);

					sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d)){\n", sc.fft_dim_full, sc.fft_zeropad_left_read[sc.axis_id], sc.fft_dim_full, sc.fft_zeropad_right_read[sc.axis_id]);
					if (sc.readToRegisters) {
						if (sc.inputBufferBlockNum == 1)
							sprintf(output + strlen(output), "			temp_%d=%sinputBlocks[0].inputs[indexInput((gl_GlobalInvocationID.x%s) %% (%d), inoutID%s%s)]%s;\n", i, convTypeLeft, shiftX, sc.fft_dim_x, requestCoordinate, requestBatch, convTypeRight);
						else
							sprintf(output + strlen(output), "			temp_%d=%sinputBlocks[indexInput((gl_GlobalInvocationID.x%s) %% (%d), inoutID%s%s) / %d].inputs[indexInput((gl_GlobalInvocationID.x%s) %% (%d), inoutID%s%s) %% %d]%s;\n", i, convTypeLeft, shiftX, sc.fft_dim_x, requestCoordinate, requestBatch, sc.inputBufferBlockSize, shiftX, sc.fft_dim_x, requestCoordinate, requestBatch, sc.inputBufferBlockSize, convTypeRight);
					}
					else {
						if (sc.inputBufferBlockNum == 1)
							sprintf(output + strlen(output), "			sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+%d)+gl_LocalInvocationID.x]=%sinputBlocks[0].inputs[indexInput((gl_GlobalInvocationID.x%s) %% (%d), inoutID%s%s)]%s;\n", i * sc.localSize[1], convTypeLeft, shiftX, sc.fft_dim_x, requestCoordinate, requestBatch, convTypeRight);
						else
							sprintf(output + strlen(output), "			sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+%d)+gl_LocalInvocationID.x]=%sinputBlocks[indexInput((gl_GlobalInvocationID.x%s) %% (%d), inoutID%s%s) / %d].inputs[indexInput((gl_GlobalInvocationID.x%s) %% (%d), inoutID%s%s) %% %d]%s;\n", i * sc.localSize[1], convTypeLeft, shiftX, sc.fft_dim_x, requestCoordinate, requestBatch, sc.inputBufferBlockSize, shiftX, sc.fft_dim_x, requestCoordinate, requestBatch, sc.inputBufferBlockSize, convTypeRight);
					}
					sprintf(output + strlen(output), "		}\n");
					sprintf(output + strlen(output), "		else\n");
					sprintf(output + strlen(output), "			temp_%d = %s(0,0);\n", i, vecType);
				}

			}
			else {
				for (uint32_t i = 0; i < sc.min_registers_per_thread; i++) {

					sprintf(output + strlen(output), "		inoutID = indexInput((gl_GlobalInvocationID.x%s) %% (%d), %d * (gl_LocalInvocationID.y + %d) + ((gl_GlobalInvocationID.x%s) / %d) %% (%d)+((gl_GlobalInvocationID.x%s) / %d) * (%d)%s%s);\n", shiftX, sc.fft_dim_x, sc.stageStartSize, i * sc.localSize[1], shiftX, sc.fft_dim_x, sc.stageStartSize, shiftX, sc.fft_dim_x * sc.stageStartSize, sc.fftDim * sc.stageStartSize, requestCoordinate, requestBatch);
					if (sc.readToRegisters) {
						if (sc.inputBufferBlockNum == 1)
							sprintf(output + strlen(output), "		temp_%d = %sinputBlocks[0].inputs[inoutID]%s;\n", i, convTypeLeft, convTypeRight);
						else
							sprintf(output + strlen(output), "		temp_%d = %sinputBlocks[inoutID / %d].inputs[inoutID %% %d]%s;\n", i, convTypeLeft, sc.inputBufferBlockSize, sc.inputBufferBlockSize, convTypeRight);
					}
					else {
						if (sc.inputBufferBlockNum == 1)
							sprintf(output + strlen(output), "		sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+%d)+gl_LocalInvocationID.x] = %sinputBlocks[0].inputs[inoutID]%s;\n", i * sc.localSize[1], convTypeLeft, convTypeRight);
						else
							sprintf(output + strlen(output), "		sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+%d)+gl_LocalInvocationID.x] = %sinputBlocks[inoutID / %d].inputs[inoutID %% %d]%s;\n", i * sc.localSize[1], convTypeLeft, sc.inputBufferBlockSize, sc.inputBufferBlockSize, convTypeRight);
					}
				}

			}
			sprintf(output + strlen(output), "	}\n");
			break;
		}
		case 2://single_c2c_strided
		{
			if (sc.localSize[1] * sc.stageRadix[0] * (sc.registers_per_thread / sc.stageRadix[0]) > sc.fftDim)
				sc.readToRegisters = 0;
			else
				sc.readToRegisters = 1;
			char shiftX[100] = "";
			if (sc.performWorkGroupShift[0])
				sprintf(shiftX, " + consts.workGroupShiftX * gl_WorkGroupSize.x ");

			//sprintf(output + strlen(output), "		if(gl_GlobalInvolcationID.x%s >= %d) return; \n", shiftX, sc.size[0] / axis->specializationConstants.fftDim);
			sprintf(output + strlen(output), "		if (((gl_GlobalInvocationID.x%s) / %d) * (%d) < %d) {;\n", shiftX, sc.stageStartSize, sc.stageStartSize* sc.fftDim, sc.fft_dim_full);

			if (sc.zeropad[0]) {
				for (uint32_t i = 0; i < sc.min_registers_per_thread; i++) {
					sprintf(output + strlen(output), "		inoutID = (gl_GlobalInvocationID.x%s) %% (%d) + %d * (gl_LocalInvocationID.y + %d) + ((gl_GlobalInvocationID.x%s) / %d) * (%d);\n", shiftX, sc.stageStartSize, sc.stageStartSize, i * sc.localSize[1], shiftX, sc.stageStartSize, sc.stageStartSize * sc.fftDim);
					sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d))\n", sc.fft_dim_full, sc.fft_zeropad_left_read[sc.axis_id], sc.fft_dim_full, sc.fft_zeropad_right_read[sc.axis_id]);
					if (sc.readToRegisters) {
						if (sc.inputBufferBlockNum == 1)
							sprintf(output + strlen(output), "			temp_%d=%sinputBlocks[0].inputs[indexInput(inoutID)]%s;\n", i, convTypeLeft, convTypeRight);
						else
							sprintf(output + strlen(output), "			temp_%d=%sinputBlocks[indexInput(inoutID) / %d].inputs[indexInput(inoutID) %% %d]%s;\n", i, convTypeLeft, sc.inputBufferBlockSize, sc.inputBufferBlockSize, convTypeRight);
					}
					else {
						if (sc.inputBufferBlockNum == 1)
							sprintf(output + strlen(output), "			sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+%d)+gl_LocalInvocationID.x]=%sinputBlocks[0].inputs[indexInput(inoutID)]%s;\n", i * sc.localSize[1], convTypeLeft, convTypeRight);
						else
							sprintf(output + strlen(output), "			sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+%d)+gl_LocalInvocationID.x]=%sinputBlocks[indexInput(inoutID) / %d].inputs[indexInput(inoutID) %% %d]%s;\n", i * sc.localSize[1], convTypeLeft, sc.inputBufferBlockSize, sc.inputBufferBlockSize, convTypeRight);
					}
					sprintf(output + strlen(output), "		else\n");
					sprintf(output + strlen(output), "			temp_%d = %s(0,0);\n", i, vecType);
				}
			}
			else {
				for (uint32_t i = 0; i < sc.min_registers_per_thread; i++) {

					sprintf(output + strlen(output), "		inoutID = indexInput((gl_GlobalInvocationID.x%s) %% (%d) + %d * (gl_LocalInvocationID.y + %d) + ((gl_GlobalInvocationID.x%s) / %d) * (%d));\n", shiftX, sc.stageStartSize, sc.stageStartSize, i * sc.localSize[1], shiftX, sc.stageStartSize, sc.stageStartSize * sc.fftDim);
					if (sc.readToRegisters) {
						if (sc.inputBufferBlockNum == 1)
							sprintf(output + strlen(output), "		temp_%d = %sinputBlocks[0].inputs[inoutID]%s;\n", i, convTypeLeft, convTypeRight);
						else
							sprintf(output + strlen(output), "		temp_%d = %sinputBlocks[inoutID / %d].inputs[inoutID %% %d]%s;\n", i, convTypeLeft, sc.inputBufferBlockSize, sc.inputBufferBlockSize, convTypeRight);
					}
					else {
						if (sc.inputBufferBlockNum == 1)
							sprintf(output + strlen(output), "		sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+%d)+gl_LocalInvocationID.x] = %sinputBlocks[0].inputs[inoutID]%s;\n", i * sc.localSize[1], convTypeLeft, convTypeRight);
						else
							sprintf(output + strlen(output), "		sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+%d)+gl_LocalInvocationID.x] = %sinputBlocks[inoutID / %d].inputs[inoutID %% %d]%s;\n", i * sc.localSize[1], convTypeLeft, sc.inputBufferBlockSize, sc.inputBufferBlockSize, convTypeRight);
					}
				}

			}
			sprintf(output + strlen(output), "	}\n");
			break;
		}
		case 3://single_c2c - registerBoost - 2x
		{
			char shiftX[100] = "";
			if (sc.performWorkGroupShift[0])
				sprintf(shiftX, " + consts.workGroupShiftX ");
			if (sc.zeropad[0]) {
				if (sc.fftDim == sc.fft_dim_full) {
					for (uint32_t j = 0; j < 2; j++) {
						for (uint32_t i = 0; i < sc.registers_per_thread; i++) {

							sprintf(output + strlen(output), "		combinedId = gl_LocalInvocationID.x + %d;\n", (i + j * sc.registers_per_thread) * sc.localSize[0]);

							if (sc.inputStride[0] > 1)
								sprintf(output + strlen(output), "		inoutID = (combinedId %% %d) * %d + (combinedId / %d) * %d;\n", sc.fftDim, sc.inputStride[0], sc.fftDim, sc.inputStride[1]);
							else
								sprintf(output + strlen(output), "		inoutID = (combinedId %% %d) + (combinedId / %d) * %d;\n", sc.fftDim, sc.fftDim, sc.inputStride[1]);

							sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d)){\n", sc.fft_dim_full, sc.fft_zeropad_left_read[sc.axis_id], sc.fft_dim_full, sc.fft_zeropad_right_read[sc.axis_id]);
							if (sc.inputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		temp_%d = %sinputBlocks[0].inputs[indexInput(inoutID%s%s)]%s;\n", (i + j * sc.registers_per_thread), convTypeLeft, requestCoordinate, requestBatch, convTypeRight);
							else
								sprintf(output + strlen(output), "		temp_%d = %sinputBlocks[indexInput(inoutID%s%s) / %d].inputs[indexInput(inoutID%s%s) %% %d]%s;\n", (i + j * sc.registers_per_thread), convTypeLeft, requestCoordinate, requestBatch, sc.inputBufferBlockSize, requestCoordinate, requestBatch, sc.inputBufferBlockSize, convTypeRight);

							sprintf(output + strlen(output), "		}else{\n");
							sprintf(output + strlen(output), "			temp_%d = %s(0,0);\n", (i + j * sc.registers_per_thread), vecType);
							sprintf(output + strlen(output), "		}\n");
						}
					}
				}
				else {
					for (uint32_t j = 0; j < 2; j++) {
						for (uint32_t i = 0; i < sc.registers_per_thread; i++) {
							sprintf(output + strlen(output), "		inoutID = gl_LocalInvocationID.x+%d+gl_LocalInvocationID.y * %d + (((gl_WorkGroupID.x%s) %% %d) * %d + ((gl_WorkGroupID.x%s) / %d) * %d);\n", (i + j * sc.registers_per_thread) * sc.localSize[0], sc.firstStageStartSize, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.localSize[1] * sc.firstStageStartSize);
							sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d))\n", sc.fft_dim_full, sc.fft_zeropad_left_read[sc.axis_id], sc.fft_dim_full, sc.fft_zeropad_right_read[sc.axis_id]);

							if (sc.inputBufferBlockNum == 1)
								sprintf(output + strlen(output), "			temp_%d = %sinputBlocks[0].inputs[indexInput(inoutID)]%s;\n", (i + j * sc.registers_per_thread), convTypeLeft, convTypeRight);
							else
								sprintf(output + strlen(output), "			temp_%d = %sinputBlocks[indexInput(inoutID) / %d].inputs[indexInput(inoutID) %% %d]%s;\n", (i + j * sc.registers_per_thread), convTypeLeft, sc.inputBufferBlockSize, sc.inputBufferBlockSize, convTypeRight);
							sprintf(output + strlen(output), "		else\n");
							sprintf(output + strlen(output), "			temp_%d = %s(0,0);\n", (i + j * sc.registers_per_thread), vecType);
						}
					}
				}
			}
			else {
				if (sc.fftDim == sc.fft_dim_full) {
					for (uint32_t j = 0; j < 2; j++) {
						for (uint32_t i = 0; i < sc.registers_per_thread; i++) {
							sprintf(output + strlen(output), "		combinedId = gl_LocalInvocationID.x + %d;\n", (i + j * sc.registers_per_thread) * sc.localSize[0]);

							if (sc.inputStride[0] > 1)
								sprintf(output + strlen(output), "		inoutID = indexInput((combinedId %% %d) * %d + (combinedId / %d) * %d%s%s);\n", sc.fftDim, sc.inputStride[0], sc.fftDim, sc.inputStride[1], requestCoordinate, requestBatch);
							else
								sprintf(output + strlen(output), "		inoutID = indexInput((combinedId %% %d) + (combinedId / %d) * %d%s%s);\n", sc.fftDim, sc.fftDim, sc.inputStride[1], requestCoordinate, requestBatch);

							if (sc.inputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		temp_%d = %sinputBlocks[0].inputs[inoutID]%s;\n", (i + j * sc.registers_per_thread), convTypeLeft, convTypeRight);
							else
								sprintf(output + strlen(output), "		temp_%d = %sinputBlocks[inoutID / %d].inputs[inoutID %% %d]%s;\n", (i + j * sc.registers_per_thread), convTypeLeft, sc.inputBufferBlockSize, sc.inputBufferBlockSize, convTypeRight);

						}
					}

				}
				else {
					for (uint32_t j = 0; j < 2; j++) {
						for (uint32_t i = 0; i < sc.registers_per_thread; i++) {
							sprintf(output + strlen(output), "		inoutID = indexInput(gl_LocalInvocationID.x+%d+gl_LocalInvocationID.y * %d + (((gl_WorkGroupID.x%s) %% %d) * %d + ((gl_WorkGroupID.x%s) / %d) * %d));\n", (i + j * sc.registers_per_thread) * sc.localSize[0], sc.firstStageStartSize, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.localSize[1] * sc.firstStageStartSize);

							if (sc.inputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		temp_%d = %sinputBlocks[0].inputs[inoutID]%s;\n", (i + j * sc.registers_per_thread), convTypeLeft, convTypeRight);
							else
								sprintf(output + strlen(output), "		temp_%d = %sinputBlocks[inoutID / %d].inputs[inoutID %% %d]%s;\n", (i + j * sc.registers_per_thread), convTypeLeft, sc.inputBufferBlockSize, sc.inputBufferBlockSize, convTypeRight);
						}
					}
				}
			}
			break;
		}
		case 4://single_c2c - registerBoost - 4x
		{
			char shiftX[100] = "";
			if (sc.performWorkGroupShift[0])
				sprintf(shiftX, " + consts.workGroupShiftX ");
			if (sc.zeropad[0]) {
				if (sc.fftDim == sc.fft_dim_full) {
					for (uint32_t j = 0; j < 4; j++) {
						for (uint32_t i = 0; i < sc.registers_per_thread; i++) {

							sprintf(output + strlen(output), "		combinedId = gl_LocalInvocationID.x + %d;\n", (i + j * sc.registers_per_thread) * sc.localSize[0]);

							if (sc.inputStride[0] > 1)
								sprintf(output + strlen(output), "		inoutID = (combinedId %% %d) * %d + (combinedId / %d) * %d;\n", sc.fftDim, sc.inputStride[0], sc.fftDim, sc.inputStride[1]);
							else
								sprintf(output + strlen(output), "		inoutID = (combinedId %% %d) + (combinedId / %d) * %d;\n", sc.fftDim, sc.fftDim, sc.inputStride[1]);

							sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d)){\n", sc.fft_dim_full, sc.fft_zeropad_left_read[sc.axis_id], sc.fft_dim_full, sc.fft_zeropad_right_read[sc.axis_id]);
							if (sc.readToRegisters) {
								if (sc.inputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		temp_%d = %sinputBlocks[0].inputs[indexInput(inoutID%s%s)]%s;\n", (i + j * sc.registers_per_thread), convTypeLeft, requestCoordinate, requestBatch, convTypeRight);
								else
									sprintf(output + strlen(output), "		temp_%d = %sinputBlocks[indexInput(inoutID%s%s) / %d].inputs[indexInput(inoutID%s%s) %% %d]%s;\n", (i + j * sc.registers_per_thread), convTypeLeft, requestCoordinate, requestBatch, sc.inputBufferBlockSize, requestCoordinate, requestBatch, sc.inputBufferBlockSize, convTypeRight);
							}
							else {
								if (sc.inputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		sdata[(combinedId %% %d) + (combinedId / %d) * sharedStride] = %sinputBlocks[0].inputs[indexInput(inoutID%s%s)]%s;\n", sc.fftDim, sc.fftDim, convTypeLeft, requestCoordinate, requestBatch, convTypeRight);
								else
									sprintf(output + strlen(output), "		sdata[(combinedId %% %d) + (combinedId / %d) * sharedStride] = %sinputBlocks[indexInput(inoutID%s%s) / %d].inputs[indexInput(inoutID%s%s) %% %d]%s;\n", sc.fftDim, sc.fftDim, convTypeLeft, requestCoordinate, requestBatch, sc.inputBufferBlockSize, requestCoordinate, requestBatch, sc.inputBufferBlockSize, convTypeRight);

							}
							sprintf(output + strlen(output), "		}else{\n");
							sprintf(output + strlen(output), "			temp_%d = %s(0,0);\n", (i + j * sc.registers_per_thread), vecType);
							sprintf(output + strlen(output), "		}\n");
						}
					}
				}
				else {
					for (uint32_t j = 0; j < 4; j++) {
						for (uint32_t i = 0; i < sc.registers_per_thread; i++) {
							sprintf(output + strlen(output), "		inoutID = gl_LocalInvocationID.x+%d+gl_LocalInvocationID.y * %d + (((gl_WorkGroupID.x%s) %% %d) * %d + ((gl_WorkGroupID.x%s) / %d) * %d);\n", (i + j * sc.registers_per_thread) * sc.localSize[0], sc.firstStageStartSize, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.localSize[1] * sc.firstStageStartSize);
							sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d))\n", sc.fft_dim_full, sc.fft_zeropad_left_read[sc.axis_id], sc.fft_dim_full, sc.fft_zeropad_right_read[sc.axis_id]);

							if (sc.inputBufferBlockNum == 1)
								sprintf(output + strlen(output), "			temp_%d = %sinputBlocks[0].inputs[indexInput(inoutID)]%s;\n", (i + j * sc.registers_per_thread), convTypeLeft, convTypeRight);
							else
								sprintf(output + strlen(output), "			temp_%d = %sinputBlocks[indexInput(inoutID) / %d].inputs[indexInput(inoutID) %% %d]%s;\n", (i + j * sc.registers_per_thread), convTypeLeft, sc.inputBufferBlockSize, sc.inputBufferBlockSize, convTypeRight);
							sprintf(output + strlen(output), "		else\n");
							sprintf(output + strlen(output), "			temp_%d = %s(0,0);\n", (i + j * sc.registers_per_thread), vecType);
						}
					}
				}
			}
			else {
				if (sc.fftDim == sc.fft_dim_full) {
					for (uint32_t j = 0; j < 4; j++) {
						for (uint32_t i = 0; i < sc.registers_per_thread; i++) {
							sprintf(output + strlen(output), "		combinedId = gl_LocalInvocationID.x + %d;\n", (i + j * sc.registers_per_thread) * sc.localSize[0]);

							if (sc.inputStride[0] > 1)
								sprintf(output + strlen(output), "		inoutID = indexInput((combinedId %% %d) * %d + (combinedId / %d) * %d%s%s);\n", sc.fftDim, sc.inputStride[0], sc.fftDim, sc.inputStride[1], requestCoordinate, requestBatch);
							else
								sprintf(output + strlen(output), "		inoutID = indexInput((combinedId %% %d) + (combinedId / %d) * %d%s%s);\n", sc.fftDim, sc.fftDim, sc.inputStride[1], requestCoordinate, requestBatch);

							if (sc.inputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		temp_%d = %sinputBlocks[0].inputs[inoutID]%s;\n", (i + j * sc.registers_per_thread), convTypeLeft, convTypeRight);
							else
								sprintf(output + strlen(output), "		temp_%d = %sinputBlocks[inoutID / %d].inputs[inoutID %% %d]%s;\n", (i + j * sc.registers_per_thread), convTypeLeft, sc.inputBufferBlockSize, sc.inputBufferBlockSize, convTypeRight);

						}
					}
				}
				else {
					for (uint32_t j = 0; j < 4; j++) {
						for (uint32_t i = 0; i < sc.registers_per_thread; i++) {
							sprintf(output + strlen(output), "		inoutID = indexInput(gl_LocalInvocationID.x+%d+gl_LocalInvocationID.y * %d + (((gl_WorkGroupID.x%s) %% %d) * %d + ((gl_WorkGroupID.x%s) / %d) * %d));\n", (i + j * sc.registers_per_thread) * sc.localSize[0], sc.firstStageStartSize, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.localSize[1] * sc.firstStageStartSize);

							if (sc.inputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		temp_%d = %sinputBlocks[0].inputs[inoutID]%s;\n", (i + j * sc.registers_per_thread), convTypeLeft, convTypeRight);
							else
								sprintf(output + strlen(output), "		temp_%d = %sinputBlocks[inoutID / %d].inputs[inoutID %% %d]%s;\n", (i + j * sc.registers_per_thread), convTypeLeft, sc.inputBufferBlockSize, sc.inputBufferBlockSize, convTypeRight);
						}
					}
				}
			}
			break;
		}
		case 5://single_r2c
		{
			if ((sc.localSize[1] > 1) || ((sc.performR2C) && (sc.inverse)) || (sc.localSize[0] * sc.stageRadix[0] * (sc.registers_per_thread / sc.stageRadix[0]) > sc.fftDim))
				sc.readToRegisters = 0;
			else
				sc.readToRegisters = 1;
			char shiftX[100] = "";
			if (sc.performWorkGroupShift[0])
				sprintf(shiftX, " + consts.workGroupShiftX ");
			char shiftY[100] = "";
			if (sc.performWorkGroupShift[1])
				sprintf(shiftY, " + consts.workGroupShiftY ");
			if (sc.zeropad[0]) {
				if (sc.fftDim == sc.fft_dim_full) {
					for (uint32_t i = 0; i < sc.min_registers_per_thread; i++) {

						if (sc.localSize[1] == 1)
							sprintf(output + strlen(output), "		combinedId = gl_LocalInvocationID.x + %d;\n", i * sc.localSize[0]);
						else
							sprintf(output + strlen(output), "		combinedId = (gl_LocalInvocationID.x + %d * gl_LocalInvocationID.y) + %d;\n", sc.localSize[0], i * sc.localSize[0] * sc.localSize[1]);

						if (sc.inputStride[0] > 1)
							sprintf(output + strlen(output), "		inoutID = (combinedId %% %d) * %d + (combinedId / %d) * %d;\n", sc.fftDim, sc.inputStride[0], sc.fftDim, 2 * sc.inputStride[1]);
						else
							sprintf(output + strlen(output), "		inoutID = (combinedId %% %d) + (combinedId / %d) * %d;\n", sc.fftDim, sc.fftDim, 2 * sc.inputStride[1]);
						if ((uint32_t)ceil(sc.size[1] / 2.0) % sc.localSize[1] != 0)
							sprintf(output + strlen(output), "		if(combinedId / %d + (gl_WorkGroupID.y%s)*gl_WorkGroupSize.y< %d){", sc.fftDim, shiftY, (uint32_t)ceil(sc.size[1] / 2.0));

						sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d)){\n", sc.fft_dim_full, sc.fft_zeropad_left_read[sc.axis_id], sc.fft_dim_full, sc.fft_zeropad_right_read[sc.axis_id]);
						if (sc.readToRegisters) {
							if (sc.inputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		temp_%d.x = %sinputBlocks[0].inputs[indexInput(inoutID)]%s;\n", i, convTypeLeft, convTypeRight);
							else
								sprintf(output + strlen(output), "		temp_%d.x = %sinputBlocks[indexInput(inoutID) / %d].inputs[indexInput(inoutID) %% %d]%s;\n", i, convTypeLeft, sc.inputBufferBlockSize, sc.inputBufferBlockSize, convTypeRight);
							if (sc.inputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		temp_%d.y = %sinputBlocks[0].inputs[(indexInput(inoutID) + %d)]%s;\n", i, convTypeLeft, sc.inputStride[1], convTypeRight);
							else
								sprintf(output + strlen(output), "		temp_%d.y = %sinputBlocks[(indexInput(inoutID) + %d)/ %d].inputs[(indexInput(inoutID) + %d) %% %d]%s;\n", i, convTypeLeft, sc.inputStride[1], sc.inputBufferBlockSize, sc.inputStride[1], sc.inputBufferBlockSize, convTypeRight);
						}
						else {
							if (sc.inputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		sdata[(combinedId %% %d) + (combinedId / %d) * sharedStride].x = %sinputBlocks[0].inputs[indexInput(inoutID)]%s;\n", sc.fftDim, sc.fftDim, convTypeLeft, convTypeRight);
							else
								sprintf(output + strlen(output), "		sdata[(combinedId %% %d) + (combinedId / %d) * sharedStride].x = %sinputBlocks[indexInput(inoutID) / %d].inputs[indexInput(inoutID) %% %d]%s;\n", sc.fftDim, sc.fftDim, convTypeLeft, sc.inputBufferBlockSize, sc.inputBufferBlockSize, convTypeRight);
							if (sc.inputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		sdata[(combinedId %% %d) + (combinedId / %d) * sharedStride].y = %sinputBlocks[0].inputs[(indexInput(inoutID) + %d)]%s;\n", sc.fftDim, sc.fftDim, convTypeLeft, sc.inputStride[1], convTypeRight);
							else
								sprintf(output + strlen(output), "		sdata[(combinedId %% %d) + (combinedId / %d) * sharedStride].y = %sinputBlocks[(indexInput(inoutID) + %d)/ %d].inputs[(indexInput(inoutID) + %d) %% %d]%s;\n", sc.fftDim, sc.fftDim, convTypeLeft, sc.inputStride[1], sc.inputBufferBlockSize, sc.inputStride[1], sc.inputBufferBlockSize, convTypeRight);

						}
						sprintf(output + strlen(output), "	}else{\n");
						if (sc.localSize[1] == 1)
							sprintf(output + strlen(output), "		temp_%d = %s(0,0);\n", i, vecType);
						else
							sprintf(output + strlen(output), "		sdata[(combinedId %% %d) + (combinedId / %d) * sharedStride] = %s(0,0);\n", sc.fftDim, sc.fftDim, vecType);
						sprintf(output + strlen(output), "	}\n");
						if ((uint32_t)ceil(sc.size[1] / 2.0) % sc.localSize[1] != 0)
							sprintf(output + strlen(output), "		}");
					}
				}
				else {
					//Not implemented
				}
			}
			else {
				if (sc.fftDim == sc.fft_dim_full) {
					for (uint32_t i = 0; i < sc.min_registers_per_thread; i++) {

						if (sc.localSize[1] == 1)
							sprintf(output + strlen(output), "		combinedId = gl_LocalInvocationID.x + %d;\n", i * sc.localSize[0]);
						else
							sprintf(output + strlen(output), "		combinedId = (gl_LocalInvocationID.x + %d * gl_LocalInvocationID.y) + %d;\n", sc.localSize[0], i * sc.localSize[0] * sc.localSize[1]);

						if (sc.inputStride[0] > 1)
							sprintf(output + strlen(output), "		inoutID = indexInput((combinedId %% %d) * %d + (combinedId / %d) * %d);\n", sc.fftDim, sc.inputStride[0], sc.fftDim, 2 * sc.inputStride[1]);
						else
							sprintf(output + strlen(output), "		inoutID = indexInput((combinedId %% %d) + (combinedId / %d) * %d);\n", sc.fftDim, sc.fftDim, 2 * sc.inputStride[1]);
						if ((uint32_t)ceil(sc.size[1] / 2.0) % sc.localSize[1] != 0)
							sprintf(output + strlen(output), "		if(combinedId / %d + (gl_WorkGroupID.y%s)*gl_WorkGroupSize.y< %d){", sc.fftDim, shiftY, (uint32_t)ceil(sc.size[sc.axis_id + 1] / 2.0));

						if (sc.readToRegisters) {
							if (sc.inputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		temp_%d.x = %sinputBlocks[0].inputs[inoutID]%s;\n", i, convTypeLeft, convTypeRight);
							else
								sprintf(output + strlen(output), "		temp_%d.x = %sinputBlocks[inoutID / %d].inputs[inoutID %% %d]%s;\n", i, convTypeLeft, sc.inputBufferBlockSize, sc.inputBufferBlockSize, convTypeRight);
							sprintf(output + strlen(output), "		inoutID += %d;\n", sc.inputStride[1]);
							if (sc.inputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		temp_%d.y = %sinputBlocks[0].inputs[inoutID]%s;\n", i, convTypeLeft, convTypeRight);
							else
								sprintf(output + strlen(output), "		temp_%d.y = %sinputBlocks[inoutID / %d].inputs[inoutID %% %d]%s;\n", i, convTypeLeft, sc.inputBufferBlockSize, sc.inputBufferBlockSize, convTypeRight);
						}
						else {
							if (sc.inputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		sdata[(combinedId %% %d) + (combinedId / %d) * sharedStride].x = %sinputBlocks[0].inputs[inoutID]%s;\n", sc.fftDim, sc.fftDim, convTypeLeft, convTypeRight);
							else
								sprintf(output + strlen(output), "		sdata[(combinedId %% %d) + (combinedId / %d) * sharedStride].x = %sinputBlocks[inoutID / %d].inputs[inoutID %% %d]%s;\n", sc.fftDim, sc.fftDim, convTypeLeft, sc.inputBufferBlockSize, sc.inputBufferBlockSize, convTypeRight);
							sprintf(output + strlen(output), "		inoutID += %d;\n", sc.inputStride[1]);
							if (sc.inputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		sdata[(combinedId %% %d) + (combinedId / %d) * sharedStride].y = %sinputBlocks[0].inputs[inoutID]%s;\n", sc.fftDim, sc.fftDim, convTypeLeft, convTypeRight);
							else
								sprintf(output + strlen(output), "		sdata[(combinedId %% %d) + (combinedId / %d) * sharedStride].y = %sinputBlocks[inoutID / %d].inputs[inoutID %% %d]%s;\n", sc.fftDim, sc.fftDim, convTypeLeft, sc.inputBufferBlockSize, sc.inputBufferBlockSize, convTypeRight);

						}
						if ((uint32_t)ceil(sc.size[1] / 2.0) % sc.localSize[1] != 0)
							sprintf(output + strlen(output), "		}");
					}

				}
				else {
					//Not implemented
				}
			}
			break;
		}
		case 6: {//single_c2r
			if ((sc.localSize[1] > 1) || ((sc.performR2C) && (sc.inverse)) || (sc.localSize[0] * sc.stageRadix[0] * (sc.registers_per_thread / sc.stageRadix[0]) > sc.fftDim))
				sc.readToRegisters = 0;
			else
				sc.readToRegisters = 1;
			char shiftY[100] = "";
			if (sc.performWorkGroupShift[1])
				sprintf(shiftY, " + consts.workGroupShiftY * %d", sc.localSize[1]);
			if (sc.zeropad[0]) {
				if (sc.fftDim == sc.fft_dim_full) {
					for (uint32_t i = 0; i < ceil(sc.min_registers_per_thread / 2.0); i++) {
						if ((uint32_t)ceil(sc.size[1] / 2.0) % sc.localSize[1] != 0)
							sprintf(output + strlen(output), "		if(gl_GlobalInvocationID.y%s < %d){", shiftY, (uint32_t)ceil(sc.size[1] / 2.0));

						sprintf(output + strlen(output), "		inoutID = gl_LocalInvocationID.x+%d;\n", i * sc.localSize[0]);

						sprintf(output + strlen(output), "		if((inoutID < %d)||(inoutID >= %d)){\n", sc.fft_zeropad_left_read[sc.axis_id], sc.fft_zeropad_right_read[sc.axis_id]);

						if (sc.inputBufferBlockNum == 1)
							sprintf(output + strlen(output), "		temp_0 = %sinputBlocks[0].inputs[indexInput(inoutID, (gl_GlobalInvocationID.y%s))]%s;\n", convTypeLeft, shiftY, convTypeRight);
						else
							sprintf(output + strlen(output), "		temp_0 = %sinputBlocks[indexInput(inoutID, (gl_GlobalInvocationID.y%s)) / %d].inputs[indexInput(inoutID, (gl_GlobalInvocationID.y%s)) %% %d]%s;\n", convTypeLeft, shiftY, sc.inputBufferBlockSize, shiftY, sc.inputBufferBlockSize, convTypeRight);

						sprintf(output + strlen(output), "		inoutID = indexInput(gl_LocalInvocationID.x+%d, (gl_GlobalInvocationID.y%s));\n", sc.inputStride[1] / 2 + i * sc.localSize[0], shiftY);

						if (sc.inputBufferBlockNum == 1)
							sprintf(output + strlen(output), "		temp_1 = %sinputBlocks[0].inputs[inoutID]%s;\n", convTypeLeft, convTypeRight);
						else
							sprintf(output + strlen(output), "		temp_1 = %sinputBlocks[inoutID / %d].inputs[inoutID %% %d]%s;\n", convTypeLeft, sc.inputBufferBlockSize, sc.inputBufferBlockSize, convTypeRight);

						sprintf(output + strlen(output), "		}else{\n");
						sprintf(output + strlen(output), "			temp_0 = %s(0,0);", vecType);
						sprintf(output + strlen(output), "			temp_1 = %s(0,0);", vecType);
						sprintf(output + strlen(output), "		}\n");
						if (sc.localSize[1] == 1)
							sprintf(output + strlen(output), "\
		sdata[gl_LocalInvocationID.x+%d].x=(temp_0.x-temp_1.y);\n\
		sdata[gl_LocalInvocationID.x + %d].y = (temp_0.y + temp_1.x);\n\
		sdata[%d - gl_LocalInvocationID.x].x = (temp_0.x + temp_1.y);\n\
		sdata[%d - gl_LocalInvocationID.x].y = (-temp_0.y + temp_1.x);\n", i * sc.localSize[0] + 1, i * sc.localSize[0] + 1, sc.fftDim - i * sc.localSize[0] - 1, sc.fftDim - i * sc.localSize[0] - 1);
						else
							sprintf(output + strlen(output), "\
		sdata[sharedStride*gl_LocalInvocationID.y + gl_LocalInvocationID.x+%d].x=(temp_0.x-temp_1.y);\n\
		sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + %d].y = (temp_0.y + temp_1.x);\n\
		sdata[sharedStride * gl_LocalInvocationID.y + %d - gl_LocalInvocationID.x].x = (temp_0.x + temp_1.y);\n\
		sdata[sharedStride * gl_LocalInvocationID.y + %d - gl_LocalInvocationID.x].y = (-temp_0.y + temp_1.x);\n", i * sc.localSize[0] + 1, i * sc.localSize[0] + 1, sc.fftDim - i * sc.localSize[0] - 1, sc.fftDim - i * sc.localSize[0] - 1);
					}
					sprintf(output + strlen(output), "\
	if (gl_LocalInvocationID.x==0) \n\
	{\n");
					sprintf(output + strlen(output), "		inoutID = indexInput(2 * (gl_GlobalInvocationID.y%s), %d);\n", shiftY, sc.inputStride[2] / (sc.inputStride[1] + 2));
					if (sc.inputBufferBlockNum == 1)
						sprintf(output + strlen(output), "		temp_0 = %sinputBlocks[0].inputs[inoutID]%s;\n", convTypeLeft, convTypeRight);
					else
						sprintf(output + strlen(output), "		temp_0 = %sinputBlocks[inoutID / %d].inputs[inoutID %% %d]%s;\n", convTypeLeft, sc.inputBufferBlockSize, sc.inputBufferBlockSize, convTypeRight);
					sprintf(output + strlen(output), "		inoutID = indexInput(2 * (gl_GlobalInvocationID.y%s) + 1, %d);\n", shiftY, sc.inputStride[2] / (sc.inputStride[1] + 2));
					if (sc.inputBufferBlockNum == 1)
						sprintf(output + strlen(output), "		temp_1 = %sinputBlocks[0].inputs[inoutID]%s;\n", convTypeLeft, convTypeRight);
					else
						sprintf(output + strlen(output), "		temp_1 = %sinputBlocks[inoutID / %d].inputs[inoutID %% %d]%s;\n", convTypeLeft, sc.inputBufferBlockSize, sc.inputBufferBlockSize, convTypeRight);
					if (sc.localSize[1] == 1)
						sprintf(output + strlen(output), "\
		sdata[0].x = (temp_0.x - temp_1.y);\n\
		sdata[0].y = (temp_0.y + temp_1.x);\n");
					else
						sprintf(output + strlen(output), "\
		sdata[sharedStride * gl_LocalInvocationID.y].x = (temp_0.x - temp_1.y);\n\
		sdata[sharedStride * gl_LocalInvocationID.y].y = (temp_0.y + temp_1.x);\n");
					sprintf(output + strlen(output), "	}\n");
					if ((uint32_t)ceil(sc.size[1] / 2.0) % sc.localSize[1] != 0)
						sprintf(output + strlen(output), "		}");
				}
			}
			else {
				if (sc.fftDim == sc.fft_dim_full) {
					for (uint32_t i = 0; i < ceil(sc.min_registers_per_thread / 2.0); i++) {
						if ((uint32_t)ceil(sc.size[1] / 2.0) % sc.localSize[1] != 0)
							sprintf(output + strlen(output), "		if(gl_GlobalInvocationID.y%s < %d){", shiftY, (uint32_t)ceil(sc.size[1] / 2.0));

						sprintf(output + strlen(output), "		inoutID = indexInput(gl_LocalInvocationID.x + %d, (gl_GlobalInvocationID.y%s));\n", i * sc.localSize[0], shiftY);

						if (sc.inputBufferBlockNum == 1)
							sprintf(output + strlen(output), "		temp_0 = %sinputBlocks[0].inputs[inoutID]%s;\n", convTypeLeft, convTypeRight);
						else
							sprintf(output + strlen(output), "		temp_0 = %sinputBlocks[inoutID / %d].inputs[inoutID %% %d]%s;\n", convTypeLeft, sc.inputBufferBlockSize, sc.inputBufferBlockSize, convTypeRight);

						sprintf(output + strlen(output), "		inoutID = indexInput(gl_LocalInvocationID.x+%d, (gl_GlobalInvocationID.y%s));\n", sc.inputStride[1] / 2 + i * sc.localSize[0], shiftY);

						if (sc.inputBufferBlockNum == 1)
							sprintf(output + strlen(output), "		temp_1 = %sinputBlocks[0].inputs[inoutID]%s;\n", convTypeLeft, convTypeRight);
						else
							sprintf(output + strlen(output), "		temp_1 = %sinputBlocks[inoutID / %d].inputs[inoutID %% %d]%s;\n", convTypeLeft, sc.inputBufferBlockSize, sc.inputBufferBlockSize, convTypeRight);
						if (sc.localSize[1] == 1)
							sprintf(output + strlen(output), "\
		sdata[gl_LocalInvocationID.x+%d].x=(temp_0.x-temp_1.y);\n\
		sdata[gl_LocalInvocationID.x + %d].y = (temp_0.y + temp_1.x);\n\
		sdata[%d - gl_LocalInvocationID.x].x = (temp_0.x + temp_1.y);\n\
		sdata[%d - gl_LocalInvocationID.x].y = (-temp_0.y + temp_1.x);\n", i * sc.localSize[0] + 1, i * sc.localSize[0] + 1, sc.fftDim - i * sc.localSize[0] - 1, sc.fftDim - i * sc.localSize[0] - 1);
						else
							sprintf(output + strlen(output), "\
		sdata[sharedStride*gl_LocalInvocationID.y + gl_LocalInvocationID.x+%d].x=(temp_0.x-temp_1.y);\n\
		sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + %d].y = (temp_0.y + temp_1.x);\n\
		sdata[sharedStride * gl_LocalInvocationID.y + %d - gl_LocalInvocationID.x].x = (temp_0.x + temp_1.y);\n\
		sdata[sharedStride * gl_LocalInvocationID.y + %d - gl_LocalInvocationID.x].y = (-temp_0.y + temp_1.x);\n", i * sc.localSize[0] + 1, i * sc.localSize[0] + 1, sc.fftDim - i * sc.localSize[0] - 1, sc.fftDim - i * sc.localSize[0] - 1);
					}
					sprintf(output + strlen(output), "\
	if (gl_LocalInvocationID.x==0) \n\
	{\n");
					sprintf(output + strlen(output), "		inoutID = indexInput(2 * (gl_GlobalInvocationID.y%s), %d);\n", shiftY, sc.inputStride[2] / (sc.inputStride[1] + 2));
					if (sc.inputBufferBlockNum == 1)
						sprintf(output + strlen(output), "		temp_0 = %sinputBlocks[0].inputs[inoutID]%s;\n", convTypeLeft, convTypeRight);
					else
						sprintf(output + strlen(output), "		temp_0 = %sinputBlocks[inoutID / %d].inputs[inoutID %% %d]%s;\n", convTypeLeft, sc.inputBufferBlockSize, sc.inputBufferBlockSize, convTypeRight);
					sprintf(output + strlen(output), "		inoutID = indexInput(2 * (gl_GlobalInvocationID.y%s) + 1, %d);\n", shiftY, sc.inputStride[2] / (sc.inputStride[1] + 2));
					if (sc.inputBufferBlockNum == 1)
						sprintf(output + strlen(output), "		temp_1 = %sinputBlocks[0].inputs[inoutID]%s;\n", convTypeLeft, convTypeRight);
					else
						sprintf(output + strlen(output), "		temp_1 = %sinputBlocks[inoutID / %d].inputs[inoutID %% %d]%s;\n", convTypeLeft, sc.inputBufferBlockSize, sc.inputBufferBlockSize, convTypeRight);
					if (sc.localSize[1] == 1)
						sprintf(output + strlen(output), "\
		sdata[0].x = (temp_0.x - temp_1.y);\n\
		sdata[0].y = (temp_0.y + temp_1.x);\n");
					else
						sprintf(output + strlen(output), "\
		sdata[sharedStride * gl_LocalInvocationID.y].x = (temp_0.x - temp_1.y);\n\
		sdata[sharedStride * gl_LocalInvocationID.y].y = (temp_0.y + temp_1.x);\n");
					sprintf(output + strlen(output), "	}\n");
					if ((uint32_t)ceil(sc.size[1] / 2.0) % sc.localSize[1] != 0)
						sprintf(output + strlen(output), "		}");
				}
				else {
					//Not implemented
				}
			}
			break;
		}
		}
	}

	static inline void appendReorder4StepRead(char* output, VkFFTSpecializationConstantsLayout sc, const char* floatType, const char* uintType, uint32_t reorderType) {
		char vecType[10];
		char LFending[4] = "";
		if (!strcmp(floatType, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatType, "double")) {
			sprintf(vecType, "dvec2");
			sprintf(LFending, "LF");
		}
		switch (reorderType) {
		case 1: {//grouped_c2c
			char shiftX[100] = "";
			if (sc.performWorkGroupShift[0])
				sprintf(shiftX, " + consts.workGroupShiftX * gl_WorkGroupSize.x ");
			if ((sc.stageStartSize > 1) && (!sc.reorderFourStep) && (sc.inverse)) {
				if (sc.localSize[1] * sc.stageRadix[0] * (sc.registers_per_thread / sc.stageRadix[0]) > sc.fftDim) {
					appendBarrierVkFFT(output, 1);
					sc.readToRegisters = 0;
				}
				else
					sc.readToRegisters = 1;
				for (uint32_t i = 0; i < sc.fftDim / sc.localSize[1]; i++) {
					if (sc.LUT)
						sprintf(output + strlen(output), "		mult = twiddleLUT[%d+(((gl_GlobalInvocationID.x%s)/%d) %% (%d))+%d*(gl_LocalInvocationID.y+%d)];\n", 3 * sc.maxStageSumLUT, shiftX, sc.fft_dim_x, sc.stageStartSize, sc.stageStartSize, i * sc.localSize[1]);
					else {
						sprintf(output + strlen(output), "		angle = 2 * M_PI * (((((gl_GlobalInvocationID.x%s) / %d) %% (%d)) * (gl_LocalInvocationID.y + %d)) / %f%s;\n", shiftX, sc.fft_dim_x, sc.stageStartSize, i * sc.localSize[1], (double)(sc.stageStartSize * sc.fftDim), LFending);
						if (!strcmp(floatType, "float"))
							sprintf(output + strlen(output), "		mult = %s(cos(angle), -sin(angle));\n", vecType);
						if (!strcmp(floatType, "double"))
							sprintf(output + strlen(output), "		mult = sincos_20(-angle);\n");
					}
					if (sc.readToRegisters) {
						sprintf(output + strlen(output), "\
		temp_%d = %s(temp_%d.x * mult.x - temp_%d.y * mult.y, temp_%d.y * mult.x + temp_%d.x * mult.y);\n", i, vecType, i, i, i, i);
					}
					else {
						sprintf(output + strlen(output), "\
		sdata[gl_WorkGroupSize.x*(%d+gl_LocalInvocationID.y) + gl_LocalInvocationID.x] = %s(sdata[gl_WorkGroupSize.x*(%d+gl_LocalInvocationID.y) + gl_LocalInvocationID.x].x * mult.x - sdata[gl_WorkGroupSize.x*(%d+gl_LocalInvocationID.y) + gl_LocalInvocationID.x].y * mult.y, sdata[gl_WorkGroupSize.x*(%d+gl_LocalInvocationID.y) + gl_LocalInvocationID.x].y * mult.x + sdata[gl_WorkGroupSize.x*(%d+gl_LocalInvocationID.y) + gl_LocalInvocationID.x].x * mult.y);\n", i * sc.localSize[1], vecType, i * sc.localSize[1], i * sc.localSize[1], i * sc.localSize[1], i * sc.localSize[1]);
					}
				}
				//appendBarrierVkFFT(output, 1);
			}

			break;
		}
		case 2: {//single_c2c_strided
			char shiftX[100] = "";
			if (sc.performWorkGroupShift[0])
				sprintf(shiftX, " + consts.workGroupShiftX * gl_WorkGroupSize.x ");
			if ((!sc.reorderFourStep) && (sc.inverse)) {
				if (sc.localSize[1] * sc.stageRadix[0] * (sc.registers_per_thread / sc.stageRadix[0]) > sc.fftDim) {
					appendBarrierVkFFT(output, 1);
					sc.readToRegisters = 0;
				}
				else
					sc.readToRegisters = 1;
				for (uint32_t i = 0; i < sc.fftDim / sc.localSize[1]; i++) {
					if (sc.LUT)
						sprintf(output + strlen(output), "		mult = twiddleLUT[%d + ((gl_GlobalInvocationID.x%s) %% (%d)) + (gl_LocalInvocationID.y + %d) * %d];\n", 3 * sc.maxStageSumLUT, shiftX, sc.stageStartSize, i * sc.localSize[1], sc.stageStartSize);
					else {
						sprintf(output + strlen(output), "		angle = 2 * M_PI * ((((gl_GlobalInvocationID.x%s) %% (%d)) * (gl_LocalInvocationID.y + %d)) / %f%s);\n", shiftX, sc.stageStartSize, i * sc.localSize[1], (double)(sc.stageStartSize * sc.fftDim), LFending);

						if (!strcmp(floatType, "float"))
							sprintf(output + strlen(output), "		mult = %s(cos(angle), -sin(angle));\n", vecType);
						if (!strcmp(floatType, "double"))
							sprintf(output + strlen(output), "		mult = sincos_20(-angle);\n");
					}
					if (sc.readToRegisters) {
						sprintf(output + strlen(output), "\
		temp_%d = %s(temp_%d.x * mult.x - temp_%d.y * mult.y, temp_%d.y * mult.x + temp_%d.x * mult.y);\n", i, vecType, i, i, i, i);
					}
					else {
						sprintf(output + strlen(output), "\
		sdata[gl_WorkGroupSize.x*(%d+gl_LocalInvocationID.y) + gl_LocalInvocationID.x] = %s(sdata[gl_WorkGroupSize.x*(%d+gl_LocalInvocationID.y) + gl_LocalInvocationID.x].x * mult.x - sdata[gl_WorkGroupSize.x*(%d+gl_LocalInvocationID.y) + gl_LocalInvocationID.x].y * mult.y, sdata[gl_WorkGroupSize.x*(%d+gl_LocalInvocationID.y) + gl_LocalInvocationID.x].y * mult.x + sdata[gl_WorkGroupSize.x*(%d+gl_LocalInvocationID.y) + gl_LocalInvocationID.x].x * mult.y);\n", i * sc.localSize[1], vecType, i * sc.localSize[1], i * sc.localSize[1], i * sc.localSize[1], i * sc.localSize[1]);
					}
				}
			}
			//appendBarrierVkFFT(output, 1);
			break;
		}
		}

	}
	static inline void appendReorder4StepWrite(char* output, VkFFTSpecializationConstantsLayout sc, const char* floatType, const char* uintType, uint32_t reorderType) {
		char vecType[10];
		char LFending[4] = "";
		if (!strcmp(floatType, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatType, "double")) {
			sprintf(vecType, "dvec2");
			sprintf(LFending, "LF");
		}
		switch (reorderType) {
		case 1: {//grouped_c2c
			char shiftX[100] = "";
			if (sc.performWorkGroupShift[0])
				sprintf(shiftX, " + consts.workGroupShiftX * gl_WorkGroupSize.x ");
			if ((sc.stageStartSize > 1) && (!((sc.stageStartSize > 1) && (!sc.reorderFourStep) && (sc.inverse)))) {
				if (sc.localSize[1] * sc.stageRadix[sc.numStages - 1] * (sc.registers_per_thread / sc.stageRadix[sc.numStages - 1]) > sc.fftDim) {
					appendBarrierVkFFT(output, 1);
					sc.writeFromRegisters = 0;
				}
				else
					sc.writeFromRegisters = 1;
				for (uint32_t i = 0; i < sc.fftDim / sc.localSize[1]; i++) {
					if (sc.LUT)
						sprintf(output + strlen(output), "		mult = twiddleLUT[%d+(((gl_GlobalInvocationID.x%s)/%d) %% (%d))+%d*(gl_LocalInvocationID.y+%d)];\n", 3 * sc.maxStageSumLUT, shiftX, sc.fft_dim_x, sc.stageStartSize, sc.stageStartSize, i * sc.localSize[1]);
					else {
						sprintf(output + strlen(output), "		angle = 2 * M_PI * ((((gl_GlobalInvocationID.x%s) / %d) %% (%d)) * (gl_LocalInvocationID.y + %d)) / %f%s;\n", shiftX, sc.fft_dim_x, sc.stageStartSize, i * sc.localSize[1], (double)(sc.stageStartSize * sc.fftDim), LFending);
						if (sc.inverse) {
							if (!strcmp(floatType, "float"))
								sprintf(output + strlen(output), "		mult = %s(cos(angle), -sin(angle));\n", vecType);
							if (!strcmp(floatType, "double"))
								sprintf(output + strlen(output), "		mult = sincos_20(-angle);\n");
						}
						else {
							if (!strcmp(floatType, "float"))
								sprintf(output + strlen(output), "		mult = %s(cos(angle), sin(angle));\n", vecType);
							if (!strcmp(floatType, "double"))
								sprintf(output + strlen(output), "		mult = sincos_20(angle);\n");
						}
					}
					if (sc.writeFromRegisters) {
						sprintf(output + strlen(output), "\
		temp_%d = %s(temp_%d.x * mult.x - temp_%d.y * mult.y, temp_%d.y * mult.x + temp_%d.x * mult.y);\n", i, vecType, i, i, i, i);
					}
					else {
						sprintf(output + strlen(output), "\
		sdata[gl_WorkGroupSize.x*(%d+gl_LocalInvocationID.y) + gl_LocalInvocationID.x] = %s(sdata[gl_WorkGroupSize.x*(%d+gl_LocalInvocationID.y) + gl_LocalInvocationID.x].x * mult.x - sdata[gl_WorkGroupSize.x*(%d+gl_LocalInvocationID.y) + gl_LocalInvocationID.x].y * mult.y, sdata[gl_WorkGroupSize.x*(%d+gl_LocalInvocationID.y) + gl_LocalInvocationID.x].y * mult.x + sdata[gl_WorkGroupSize.x*(%d+gl_LocalInvocationID.y) + gl_LocalInvocationID.x].x * mult.y);\n", i * sc.localSize[1], vecType, i * sc.localSize[1], i * sc.localSize[1], i * sc.localSize[1], i * sc.localSize[1]);
					}
				}
				//appendBarrierVkFFT(output, 1);
			}
			break;
		}
		case 2: {//single_c2c_strided
			char shiftX[100] = "";
			if (sc.performWorkGroupShift[0])
				sprintf(shiftX, " + consts.workGroupShiftX * gl_WorkGroupSize.x ");
			if (!((!sc.reorderFourStep) && (sc.inverse))) {
				if (sc.localSize[1] * sc.stageRadix[sc.numStages - 1] * (sc.registers_per_thread / sc.stageRadix[sc.numStages - 1]) > sc.fftDim) {
					appendBarrierVkFFT(output, 1);
					sc.writeFromRegisters = 0;
				}
				else
					sc.writeFromRegisters = 1;
				for (uint32_t i = 0; i < sc.fftDim / sc.localSize[1]; i++) {
					if (sc.LUT)
						sprintf(output + strlen(output), "		mult = twiddleLUT[%d + ((gl_GlobalInvocationID.x%s) %% (%d)) + (gl_LocalInvocationID.y + %d) * %d];\n", 3 * sc.maxStageSumLUT, shiftX, sc.stageStartSize, i * sc.localSize[1], sc.stageStartSize);
					else {
						sprintf(output + strlen(output), "		angle = 2 * M_PI * ((((gl_GlobalInvocationID.x%s) %% (%d)) * (gl_LocalInvocationID.y + %d)) / %f%s);\n", shiftX, sc.stageStartSize, i * sc.localSize[1], (double)(sc.stageStartSize * sc.fftDim), LFending);
						if (sc.inverse) {
							if (!strcmp(floatType, "float"))
								sprintf(output + strlen(output), "		mult = %s(cos(angle), -sin(angle));\n", vecType);
							if (!strcmp(floatType, "double"))
								sprintf(output + strlen(output), "		mult = sincos_20(-angle);\n");
						}
						else {
							if (!strcmp(floatType, "float"))
								sprintf(output + strlen(output), "		mult = %s(cos(angle), sin(angle));\n", vecType);
							if (!strcmp(floatType, "double"))
								sprintf(output + strlen(output), "		mult = sincos_20(angle);\n");
						}
					}
					if (sc.writeFromRegisters) {
						sprintf(output + strlen(output), "\
		temp_%d = %s(temp_%d.x * mult.x - temp_%d.y * mult.y, temp_%d.y * mult.x + temp_%d.x * mult.y);\n", i, vecType, i, i, i, i);
					}
					else {
						sprintf(output + strlen(output), "\
		sdata[gl_WorkGroupSize.x*(%d+gl_LocalInvocationID.y) + gl_LocalInvocationID.x] = %s(sdata[gl_WorkGroupSize.x*(%d+gl_LocalInvocationID.y) + gl_LocalInvocationID.x].x * mult.x - sdata[gl_WorkGroupSize.x*(%d+gl_LocalInvocationID.y) + gl_LocalInvocationID.x].y * mult.y, sdata[gl_WorkGroupSize.x*(%d+gl_LocalInvocationID.y) + gl_LocalInvocationID.x].y * mult.x + sdata[gl_WorkGroupSize.x*(%d+gl_LocalInvocationID.y) + gl_LocalInvocationID.x].x * mult.y);\n", i * sc.localSize[1], vecType, i * sc.localSize[1], i * sc.localSize[1], i * sc.localSize[1], i * sc.localSize[1]);
					}
				}
			}
			//appendBarrierVkFFT(output, 1);
			break;
		}
		}

	}

	static inline void appendRadixStageNonStrided(char* output, VkFFTSpecializationConstantsLayout sc, const char* floatType, const char* uintType, uint32_t stageSize, uint32_t stageSizeSum, double stageAngle, uint32_t stageRadix) {
		char vecType[10];
		char LFending[4] = "";
		if (!strcmp(floatType, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatType, "double")) {
			sprintf(vecType, "dvec2");
			sprintf(LFending, "LF");
		}
		char convolutionInverse[10] = "";
		if (sc.convolutionStep) {
			if (stageAngle > 0)
				sprintf(convolutionInverse, ", 0");
			else
				sprintf(convolutionInverse, ", 1");
		}
		uint32_t logicalRegistersPerThread = stageRadix * (sc.registers_per_thread / stageRadix);
		uint32_t logicalGroupSize = sc.fftDim / logicalRegistersPerThread;
		if ((sc.localSize[0] * logicalRegistersPerThread > sc.fftDim) || (stageSize > 1) || (sc.localSize[1] > 1) || ((sc.performR2C) && (sc.inverse)) || ((sc.convolutionStep) && ((sc.matrixConvolution > 1) || (sc.numKernels > 1)) && (stageAngle < 0)))
			appendBarrierVkFFT(output, 1);

		if (sc.localSize[0] * logicalRegistersPerThread > sc.fftDim)
			sprintf(output + strlen(output), "\
		if (gl_LocalInvocationID.x * %d < %d) {\n", logicalRegistersPerThread, sc.fftDim);
		for (uint32_t j = 0; j < logicalRegistersPerThread / stageRadix; j++) {
			sprintf(output + strlen(output), "\
		stageInvocationID = (gl_LocalInvocationID.x+ %d) %% (%d);\n", j * logicalGroupSize, stageSize);
			if (sc.LUT)
				sprintf(output + strlen(output), "		LUTId = stageInvocationID + %d;\n", stageSizeSum);
			else
				sprintf(output + strlen(output), "		angle = stageInvocationID * %.17f%s;\n", stageAngle, LFending);
			if ((sc.localSize[0] * logicalRegistersPerThread > sc.fftDim) || (stageSize > 1) || (sc.localSize[1] > 1) || ((sc.performR2C) && (sc.inverse)) || ((sc.convolutionStep) && ((sc.matrixConvolution > 1) || (sc.numKernels > 1)) && (stageAngle < 0))) {
				for (uint32_t i = 0; i < stageRadix; i++) {
					sprintf(output + strlen(output), "\
		temp_%d = sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + %d];\n", j + i * logicalRegistersPerThread / stageRadix, j * logicalGroupSize + i * sc.fftDim / stageRadix);
				}
			}
			uint32_t* regID = (uint32_t*)malloc(sizeof(uint32_t) * stageRadix);
			for (uint32_t i = 0; i < stageRadix; i++) {
				regID[i] = j + i * logicalRegistersPerThread / stageRadix;
			}
			inlineRadixKernelVkFFT(output, sc, floatType, uintType, stageRadix, stageAngle, regID);
			/*sprintf(output + strlen(output), "		radix%d(", stageRadix);
			for (uint32_t i = 0; i < stageRadix; i++) {
				sprintf(output + strlen(output), "temp_%d, ", j + i * logicalRegistersPerThread / stageRadix);
			}
			if (sc.LUT)
				sprintf(output + strlen(output), "LUTId%s);\n", convolutionInverse);
			else
				sprintf(output + strlen(output), "angle%s);\n", convolutionInverse);*/
			free(regID);
		}
		if ((stageSize == 1) && (sc.cacheShuffle)) {
			for (uint32_t i = 0; i < logicalRegistersPerThread; i++) {
				sprintf(output + strlen(output), "\
		shuffle[%d]=temp_%d;\n", i, i);
			}
			for (uint32_t i = 0; i < logicalRegistersPerThread; i++) {
				sprintf(output + strlen(output), "\
		temp_%d=shuffle[(%d+tshuffle)%%(%d)];\n", i, i, logicalRegistersPerThread);
			}
		}
		if (sc.localSize[0] * logicalRegistersPerThread > sc.fftDim)
			sprintf(output + strlen(output), "		}\n");

	}
	static inline void appendRadixStageNonStridedBoost2x(char* output, VkFFTSpecializationConstantsLayout sc, const char* floatType, const char* uintType, uint32_t stageSize, uint32_t stageSizeSum, double stageAngle, uint32_t stageRadix) {
		char vecType[10];
		char LFending[4] = "";
		if (!strcmp(floatType, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatType, "double")) {
			sprintf(vecType, "dvec2");
			sprintf(LFending, "LF");
		}
		switch (stageRadix) {
		case 2:
		{
			for (uint32_t j = 0; j < 2; j++) {
				for (uint32_t l = 0; l < 4; l++) {
					sprintf(output + strlen(output), "\
		stageInvocationID = (gl_LocalInvocationID.x + %d) %% (%d);\n", (l + 4 * j) * sc.localSize[0], stageSize);
					if (sc.LUT)
						sprintf(output + strlen(output), "		LUTId = stageInvocationID + %d;\n", stageSizeSum);
					else
						sprintf(output + strlen(output), "		angle = stageInvocationID * %.17f%s;\n", stageAngle, LFending);
					sprintf(output + strlen(output), "		radix%d(", stageRadix);
					for (uint32_t i = 0; i < stageRadix; i++) {
						sprintf(output + strlen(output), "temp_%d, ", j * sc.registers_per_thread + l * 2 + i);
					}
					if (sc.LUT)
						sprintf(output + strlen(output), "LUTId);\n");
					else
						sprintf(output + strlen(output), "angle);\n");
				}
			}
			break;
		}
		case 4:
		{
			for (uint32_t j = 0; j < 2; j++) {
				for (uint32_t l = 0; l < 2; l++) {
					sprintf(output + strlen(output), "\
		stageInvocationID = (gl_LocalInvocationID.x + %d) %% (%d);\n", (l + 2 * j) * sc.localSize[0], stageSize);
					if (sc.LUT)
						sprintf(output + strlen(output), "		LUTId = stageInvocationID + %d;\n", stageSizeSum);
					else
						sprintf(output + strlen(output), "		angle = stageInvocationID * %.17f%s;\n", stageAngle, LFending);
					sprintf(output + strlen(output), "		radix%d(", stageRadix);
					for (uint32_t i = 0; i < stageRadix; i++) {
						sprintf(output + strlen(output), "temp_%d, ", j * sc.registers_per_thread + l * 4 + i);
					}
					if (sc.LUT)
						sprintf(output + strlen(output), "LUTId);\n");
					else
						sprintf(output + strlen(output), "angle);\n");
				}
			}
			break;
		}
		case 8:
		{
			for (uint32_t j = 0; j < 2; j++) {
				sprintf(output + strlen(output), "\
		stageInvocationID = (gl_LocalInvocationID.x + %d) %% (%d);\n", j * sc.localSize[0], stageSize);
				if (sc.LUT)
					sprintf(output + strlen(output), "		LUTId = stageInvocationID + %d;\n", stageSizeSum);
				else
					sprintf(output + strlen(output), "		angle = stageInvocationID * %.17f%s;\n", stageAngle, LFending);
				sprintf(output + strlen(output), "		radix%d(", stageRadix);
				for (uint32_t i = 0; i < stageRadix; i++) {
					sprintf(output + strlen(output), "temp_%d, ", j * sc.registers_per_thread + i);
				}
				if (sc.LUT)
					sprintf(output + strlen(output), "LUTId);\n");
				else
					sprintf(output + strlen(output), "angle);\n");
				if ((stageSize == 1) && (sc.cacheShuffle)) {
					for (uint32_t i = 0; i < sc.registers_per_thread; i++) {
						sprintf(output + strlen(output), "\
		shuffle[%d]=temp_%d;\n", i, j * sc.registers_per_thread + i);
					}
					for (uint32_t i = 0; i < sc.registers_per_thread; i++) {
						sprintf(output + strlen(output), "\
		temp_%d=shuffle[(%d+tshuffle)%%(%d)];\n", j * sc.registers_per_thread + i, i, sc.registers_per_thread);
					}
				}
			}
			break;
		}
		}
	}
	static inline void appendRadixStageNonStridedBoost4x(char* output, VkFFTSpecializationConstantsLayout sc, const char* floatType, const char* uintType, uint32_t stageSize, uint32_t stageSizeSum, double stageAngle, uint32_t stageRadix) {
		char vecType[10];
		char LFending[4] = "";
		if (!strcmp(floatType, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatType, "double")) {
			sprintf(vecType, "dvec2");
			sprintf(LFending, "LF");
		}
		switch (stageRadix) {
		case 2:
		{
			for (uint32_t j = 0; j < 4; j++) {
				for (uint32_t l = 0; l < 4; l++) {
					sprintf(output + strlen(output), "\
		stageInvocationID = (gl_LocalInvocationID.x + %d) %% (%d);\n", (l + 4 * j) * sc.localSize[0], stageSize);
					if (sc.LUT)
						sprintf(output + strlen(output), "		LUTId = stageInvocationID + %d;\n", stageSizeSum);
					else
						sprintf(output + strlen(output), "		angle = stageInvocationID * %.17f%s;\n", stageAngle, LFending);
					sprintf(output + strlen(output), "		radix%d(", stageRadix);
					for (uint32_t i = 0; i < stageRadix; i++) {
						sprintf(output + strlen(output), "temp_%d, ", j * sc.registers_per_thread + l * 2 + i);
					}
					if (sc.LUT)
						sprintf(output + strlen(output), "LUTId);\n");
					else
						sprintf(output + strlen(output), "angle);\n");
				}
			}
			break;
		}
		case 4:
		{
			for (uint32_t j = 0; j < 4; j++) {
				for (uint32_t l = 0; l < 2; l++) {
					sprintf(output + strlen(output), "\
		stageInvocationID = (gl_LocalInvocationID.x + %d) %% (%d);\n", (l + 2 * j) * sc.localSize[0], stageSize);
					if (sc.LUT)
						sprintf(output + strlen(output), "		LUTId = stageInvocationID + %d;\n", stageSizeSum);
					else
						sprintf(output + strlen(output), "		angle = stageInvocationID * %.17f%s;\n", stageAngle, LFending);
					sprintf(output + strlen(output), "		radix%d(", stageRadix);
					for (uint32_t i = 0; i < stageRadix; i++) {
						sprintf(output + strlen(output), "temp_%d, ", j * sc.registers_per_thread + l * 4 + i);
					}
					if (sc.LUT)
						sprintf(output + strlen(output), "LUTId);\n");
					else
						sprintf(output + strlen(output), "angle);\n");
				}
			}
			break;
		}
		case 8:
		{
			for (uint32_t j = 0; j < 4; j++) {
				sprintf(output + strlen(output), "\
		stageInvocationID = (gl_LocalInvocationID.x + %d) %% (%d);\n", j * sc.localSize[0], stageSize);
				if (sc.LUT)
					sprintf(output + strlen(output), "		LUTId = stageInvocationID + %d;\n", stageSizeSum);
				else
					sprintf(output + strlen(output), "		angle = stageInvocationID * %.17f%s;\n", stageAngle, LFending);
				sprintf(output + strlen(output), "		radix%d(", stageRadix);
				for (uint32_t i = 0; i < stageRadix; i++) {
					sprintf(output + strlen(output), "temp_%d, ", j * sc.registers_per_thread + i);
				}
				if (sc.LUT)
					sprintf(output + strlen(output), "LUTId);\n");
				else
					sprintf(output + strlen(output), "angle);\n");
				if ((stageSize == 1) && (sc.cacheShuffle)) {
					for (uint32_t i = 0; i < sc.registers_per_thread; i++) {
						sprintf(output + strlen(output), "\
		shuffle[%d]=temp_%d;\n", i, j * sc.registers_per_thread + i);
					}
					for (uint32_t i = 0; i < sc.registers_per_thread; i++) {
						sprintf(output + strlen(output), "\
		temp_%d=shuffle[(%d+tshuffle)%%(%d)];\n", j * sc.registers_per_thread + i, i, sc.registers_per_thread);
					}
				}
			}
			break;
		}
		}
	}
	static inline void appendRadixStageStrided(char* output, VkFFTSpecializationConstantsLayout sc, const char* floatType, const char* uintType, uint32_t stageSize, uint32_t stageSizeSum, double stageAngle, uint32_t stageRadix) {
		char vecType[10];
		char LFending[4] = "";
		if (!strcmp(floatType, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatType, "double")) {
			sprintf(vecType, "dvec2");
			sprintf(LFending, "LF");
		}
		char convolutionInverse[10] = "";
		if (sc.convolutionStep) {
			if (stageAngle > 0)
				sprintf(convolutionInverse, ", 0");
			else
				sprintf(convolutionInverse, ", 1");
		}
		uint32_t logicalRegistersPerThread = stageRadix * (sc.registers_per_thread / stageRadix);
		uint32_t logicalGroupSize = sc.fftDim / logicalRegistersPerThread;
		if ((sc.localSize[1] * logicalRegistersPerThread > sc.fftDim) || (stageSize > 1) || ((sc.convolutionStep) && ((sc.matrixConvolution > 1) || (sc.numKernels > 1)) && (stageAngle < 0)))
			appendBarrierVkFFT(output, 1);
		if (sc.localSize[1] * logicalRegistersPerThread > sc.fftDim)
			sprintf(output + strlen(output), "\
		if (gl_LocalInvocationID.y * %d < %d) {\n", logicalRegistersPerThread, sc.fftDim);
		for (uint32_t j = 0; j < logicalRegistersPerThread / stageRadix; j++) {
			sprintf(output + strlen(output), "\
		stageInvocationID = (gl_LocalInvocationID.y+ %d) %% (%d);\n", j * logicalGroupSize, stageSize);
			if (sc.LUT)
				sprintf(output + strlen(output), "		LUTId = stageInvocationID + %d;\n", stageSizeSum);
			else
				sprintf(output + strlen(output), "		angle = stageInvocationID * %.17f%s;\n", stageAngle, LFending);
			if ((sc.localSize[1] * logicalRegistersPerThread > sc.fftDim) || (stageSize > 1) || ((sc.convolutionStep) && ((sc.matrixConvolution > 1) || (sc.numKernels > 1)) && (stageAngle < 0))) {
				for (uint32_t i = 0; i < stageRadix; i++) {
					sprintf(output + strlen(output), "\
		temp_%d = sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+%d)+gl_LocalInvocationID.x];\n", j + i * logicalRegistersPerThread / stageRadix, j * logicalGroupSize + i * sc.fftDim / stageRadix);
				}
			}
			uint32_t* regID = (uint32_t*)malloc(sizeof(uint32_t) * stageRadix);
			for (uint32_t i = 0; i < stageRadix; i++) {
				regID[i] = j + i * logicalRegistersPerThread / stageRadix;
			}
			inlineRadixKernelVkFFT(output, sc, floatType, uintType, stageRadix, stageAngle, regID);
			/*sprintf(output + strlen(output), "		radix%d(", stageRadix);
			for (uint32_t i = 0; i < stageRadix; i++) {
				sprintf(output + strlen(output), "temp_%d, ", j + i * logicalRegistersPerThread / stageRadix);
			}
			if (sc.LUT)
				sprintf(output + strlen(output), "LUTId%s);\n", convolutionInverse);
			else
				sprintf(output + strlen(output), "angle%s);\n", convolutionInverse);*/
		}
		if (sc.localSize[1] * logicalRegistersPerThread > sc.fftDim)
			sprintf(output + strlen(output), "		}\n");
	}
	static inline void appendRadixStage(char* output, VkFFTSpecializationConstantsLayout sc, const char* floatType, const char* uintType, uint32_t stageSize, uint32_t stageSizeSum, double stageAngle, uint32_t stageRadix, uint32_t shuffleType) {
		switch (shuffleType) {
		case 0: case 5: case 6: {
			appendRadixStageNonStrided(output, sc, floatType, uintType, stageSize, stageSizeSum, stageAngle, stageRadix);
			//appendBarrierVkFFT(output, 1);
			break;
		}
		case 1: case 2: {
			appendRadixStageStrided(output, sc, floatType, uintType, stageSize, stageSizeSum, stageAngle, stageRadix);
			//appendBarrierVkFFT(output, 1);
			break;
		}
		case 3: {
			appendRadixStageNonStridedBoost2x(output, sc, floatType, uintType, stageSize, stageSizeSum, stageAngle, stageRadix);
			break;
		}
		case 4: {
			appendRadixStageNonStridedBoost4x(output, sc, floatType, uintType, stageSize, stageSizeSum, stageAngle, stageRadix);
			break;
		}
		}
	}

	static inline void appendRegisterBoostShuffle(char* output, VkFFTSpecializationConstantsLayout sc, const char* floatType, uint32_t stageRadix, uint32_t shuffleType, uint32_t lastRadix) {
		char vecType[10];
		if (!strcmp(floatType, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "dvec2");
		switch (shuffleType) {
		case 3:
		{
			if (stageRadix == 2) {
				sprintf(output + strlen(output), "\
	sort0=temp_1;\n\
	temp_1=temp_8;\n\
	temp_8=temp_4;\n\
	temp_4=temp_2;\n\
	temp_2=sort0;\n\
	sort0=temp_3;\n\
	temp_3=temp_9;\n\
	temp_9=temp_12;\n\
	temp_12=temp_6;\n\
	temp_6=sort0;\n\
	sort0=temp_5;\n\
	temp_5=temp_10;\n\
	temp_10=sort0;\n\
	sort0=temp_7;\n\
	temp_7=temp_11;\n\
	temp_11=temp_13;\n\
	temp_13=temp_14;\n\
	temp_14=sort0;\n");
			}
			if (stageRadix == 4) {
				sprintf(output + strlen(output), "\
	sort0=temp_1;\n\
	temp_1=temp_4;\n\
	temp_4=sort0;\n\
	sort0=temp_2;\n\
	temp_2=temp_8;\n\
	temp_8=sort0;\n\
	sort0=temp_3;\n\
	temp_3=temp_12;\n\
	temp_12=sort0;\n\
	sort0=temp_6;\n\
	temp_6=temp_9;\n\
	temp_9=sort0;\n\
	sort0=temp_7;\n\
	temp_7=temp_13;\n\
	temp_13=sort0;\n\
	sort0=temp_11;\n\
	temp_11=temp_14;\n\
	temp_14=sort0;\n");
			}
			if (stageRadix == 8) {
				sprintf(output + strlen(output), "\
	sort0=temp_1;\n\
	temp_1=temp_2;\n\
	temp_2=temp_4;\n\
	temp_4=temp_8;\n\
	temp_8=sort0;\n\
	sort0=temp_3;\n\
	temp_3=temp_6;\n\
	temp_6=temp_12;\n\
	temp_12=temp_9;\n\
	temp_9=sort0;\n\
	sort0=temp_5;\n\
	temp_5=temp_10;\n\
	temp_10=sort0;\n\
	sort0=temp_7;\n\
	temp_7=temp_14;\n\
	temp_14=temp_13;\n\
	temp_13=temp_11;\n\
	temp_11=sort0;\n");
			}
			if ((sc.inverse) && (sc.normalize) && lastRadix) {
				for (uint32_t j = 0; j < 2; j++) {
					for (uint32_t i = 0; i < sc.registers_per_thread; i++) {
						sprintf(output + strlen(output), "		temp_%d /= %d;\n", i + j * sc.registers_per_thread, stageRadix);
					}
				}
			}
			break;
		}
		case 4:
		{
			if (stageRadix == 2) {
				sprintf(output + strlen(output), "\
	sort0=temp_1;\n\
	temp_1=temp_8;\n\
	temp_8=temp_2;\n\
	temp_2=temp_16;\n\
	temp_16=temp_4;\n\
	temp_4=sort0;\n\
	sort0=temp_3;\n\
	temp_3=temp_24;\n\
	temp_24=temp_6;\n\
	temp_6=temp_17;\n\
	temp_17=temp_12;\n\
	temp_12=sort0;\n\
	sort0=temp_5;\n\
	temp_5=temp_9;\n\
	temp_9=temp_10;\n\
	temp_10=temp_18;\n\
	temp_18=temp_20;\n\
	temp_20=sort0;\n\
	sort0=temp_7;\n\
	temp_7=temp_25;\n\
	temp_25=temp_14;\n\
	temp_14=temp_19;\n\
	temp_19=temp_28;\n\
	temp_28=sort0;\n\
	sort0=temp_11;\n\
	temp_11=temp_26;\n\
	temp_26=temp_22;\n\
	temp_22=temp_21;\n\
	temp_21=temp_13;\n\
	temp_13=sort0;\n\
	sort0=temp_15;\n\
	temp_15=temp_27;\n\
	temp_27=temp_30;\n\
	temp_30=temp_23;\n\
	temp_23=temp_29;\n\
	temp_29=sort0;\n");
			}
			if (stageRadix == 4) {
				sprintf(output + strlen(output), "\
	sort0=temp_1;\n\
	temp_1=temp_8;\n\
	temp_8=temp_2;\n\
	temp_2=temp_16;\n\
	temp_16=temp_4;\n\
	temp_4=sort0;\n\
	sort0=temp_3;\n\
	temp_3=temp_24;\n\
	temp_24=temp_6;\n\
	temp_6=temp_17;\n\
	temp_17=temp_12;\n\
	temp_12=sort0;\n\
	sort0=temp_5;\n\
	temp_5=temp_9;\n\
	temp_9=temp_10;\n\
	temp_10=temp_18;\n\
	temp_18=temp_20;\n\
	temp_20=sort0;\n\
	sort0=temp_7;\n\
	temp_7=temp_25;\n\
	temp_25=temp_14;\n\
	temp_14=temp_19;\n\
	temp_19=temp_28;\n\
	temp_28=sort0;\n\
	sort0=temp_11;\n\
	temp_11=temp_26;\n\
	temp_26=temp_22;\n\
	temp_22=temp_21;\n\
	temp_21=temp_13;\n\
	temp_13=sort0;\n\
	sort0=temp_15;\n\
	temp_15=temp_27;\n\
	temp_27=temp_30;\n\
	temp_30=temp_23;\n\
	temp_23=temp_29;\n\
	temp_29=sort0;\n");

			}
			if (stageRadix == 8) {
				sprintf(output + strlen(output), "\
	sort0=temp_1;\n\
	temp_1=temp_4;\n\
	temp_4=temp_16;\n\
	temp_16=temp_2;\n\
	temp_2=temp_8;\n\
	temp_8=sort0;\n\
	sort0=temp_3;\n\
	temp_3=temp_12;\n\
	temp_12=temp_17;\n\
	temp_17=temp_6;\n\
	temp_6=temp_24;\n\
	temp_24=sort0;\n\
	sort0=temp_5;\n\
	temp_5=temp_20;\n\
	temp_20=temp_18;\n\
	temp_18=temp_10;\n\
	temp_10=temp_9;\n\
	temp_9=sort0;\n\
	sort0=temp_7;\n\
	temp_7=temp_28;\n\
	temp_28=temp_19;\n\
	temp_19=temp_14;\n\
	temp_14=temp_25;\n\
	temp_25=sort0;\n\
	sort0=temp_11;\n\
	temp_11=temp_13;\n\
	temp_13=temp_21;\n\
	temp_21=temp_22;\n\
	temp_22=temp_26;\n\
	temp_26=sort0;\n\
	sort0=temp_15;\n\
	temp_15=temp_29;\n\
	temp_29=temp_23;\n\
	temp_23=temp_30;\n\
	temp_30=temp_27;\n\
	temp_27=sort0;\n");
			}
			if ((sc.inverse) && (sc.normalize) && lastRadix) {
				for (uint32_t j = 0; j < 4; j++) {
					for (uint32_t i = 0; i < sc.registers_per_thread; i++) {
						sprintf(output + strlen(output), "		temp_%d /= %d;\n", i + j * sc.registers_per_thread, stageRadix);
					}
				}
			}
			break;
		}
		}
	}

	static inline void appendRadixShuffleNonStrided(char* output, VkFFTSpecializationConstantsLayout sc, const char* floatType, const char* uintType, uint32_t stageSize, uint32_t stageSizeSum, double stageAngle, uint32_t stageRadix) {
		char vecType[10];
		if (!strcmp(floatType, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "dvec2");
		char stageNormalization[10] = "";
		if (((sc.inverse) && (sc.normalize)) || ((sc.convolutionStep) && (stageAngle < 0)))
			sprintf(stageNormalization, " / %d", stageRadix);
		uint32_t logicalRegistersPerThread = stageRadix * (sc.registers_per_thread / stageRadix);
		uint32_t logicalGroupSize = sc.fftDim / logicalRegistersPerThread;
		if ((sc.localSize[0] * logicalRegistersPerThread > sc.fftDim) || (stageSize < sc.fftDim / stageRadix) || ((sc.reorderFourStep) && (sc.fftDim < sc.fft_dim_full) && (sc.localSize[1] > 1)) || (sc.localSize[1] > 1) || ((sc.performR2C) && (!sc.inverse) && (sc.axis_id == 0)) || ((sc.convolutionStep) && ((sc.matrixConvolution > 1) || (sc.numKernels > 1)) && (stageAngle > 0)))
			appendBarrierVkFFT(output, 1);
		if (sc.localSize[0] * logicalRegistersPerThread > sc.fftDim)
			sprintf(output + strlen(output), "\
	if (gl_GlobalInvocationID.x * %d < %d) {\n", logicalRegistersPerThread, sc.fftDim);
		if ((sc.localSize[0] * logicalRegistersPerThread > sc.fftDim) || (stageSize < sc.fftDim / stageRadix) || ((sc.reorderFourStep) && (sc.fftDim < sc.fft_dim_full) && (sc.localSize[1] > 1)) || (sc.localSize[1] > 1) || ((sc.performR2C) && (!sc.inverse) && (sc.axis_id == 0)) || ((sc.convolutionStep) && ((sc.matrixConvolution > 1) || (sc.numKernels > 1)) && (stageAngle > 0))) {
			//appendBarrierVkFFT(output, 1);
			for (uint32_t j = 0; j < logicalRegistersPerThread / stageRadix; j++) {
				sprintf(output + strlen(output), "\
		stageInvocationID = (gl_LocalInvocationID.x + %d) %% (%d);\n\
		blockInvocationID = (gl_LocalInvocationID.x + %d) - stageInvocationID;\n\
		inoutID = stageInvocationID + blockInvocationID * %d;\n", j * logicalGroupSize, stageSize, j * logicalGroupSize, stageRadix);
				if ((stageSize == 1) && (sc.cacheShuffle)) {
					for (uint32_t i = 0; i < stageRadix; i++) {
						sprintf(output + strlen(output), "\
	sdata[sharedStride * gl_LocalInvocationID.y + inoutID + ((%d+tshuffle) %% (%d))*%d] = temp_%d%s;\n", i, logicalRegistersPerThread, stageSize, j + i * logicalRegistersPerThread / stageRadix, stageNormalization);
					}
				}
				else {
					for (uint32_t i = 0; i < stageRadix; i++) {
						sprintf(output + strlen(output), "\
	sdata[sharedStride * gl_LocalInvocationID.y + inoutID + %d] = temp_%d%s;\n", i * stageSize, j + i * logicalRegistersPerThread / stageRadix, stageNormalization);
					}
				}
			}
		}
		else {
			if (((sc.inverse) && (sc.normalize)) || ((sc.convolutionStep) && (stageAngle < 0))) {
				for (uint32_t i = 0; i < logicalRegistersPerThread; i++) {
					sprintf(output + strlen(output), "\
		temp_%d = temp_%d%s;\n", i, i, stageNormalization);
				}
			}
		}
		if (sc.localSize[0] * logicalRegistersPerThread > sc.fftDim)
			sprintf(output + strlen(output), "	}\n");

	}
	static inline void appendRadixShuffleNonStridedBoost2x(char* output, VkFFTSpecializationConstantsLayout sc, const char* floatType, const char* uintType, uint32_t stageSize, uint32_t stageSizeSum, double stageAngle, uint32_t stageRadix) {
		char vecType[10];
		if (!strcmp(floatType, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "dvec2");
		switch (stageRadix) {
		case 2:
		{
			//not implemented yet
			break;
		}
		case 4:
		{
			//not implemented yet
			break;
		}
		case 8:
		{
			char stageNormalization[10] = "";
			if ((sc.inverse) && (sc.normalize))
				sprintf(stageNormalization, " * 0.125");
			sprintf(output + strlen(output), "\
		stageInvocationID = (gl_LocalInvocationID.x) %% (%d);\n\
		blockInvocationID = (gl_LocalInvocationID.x) - stageInvocationID;\n\
		inoutID = stageInvocationID + blockInvocationID * 8;\n", stageSize);
			for (uint32_t j = 0; j < 2; ++j) {
				appendBarrierVkFFT(output, 2);
				if ((stageSize == 1) && (sc.cacheShuffle)) {
					for (uint32_t i = 0; i < sc.registers_per_thread; i++) {
						sprintf(output + strlen(output), "\
			sdata[inoutID + ((%d+tshuffle)%%(%d))*%d] = temp_%d%s;\n", i, sc.registers_per_thread, stageSize, i + j * sc.registers_per_thread, stageNormalization);
					}
				}
				else {
					for (uint32_t i = 0; i < sc.registers_per_thread; ++i) {
						sprintf(output + strlen(output), "\
			sdata[inoutID + %d] = temp_%d%s;\n", i * stageSize, i + j * sc.registers_per_thread, stageNormalization);
					}
				}
				appendBarrierVkFFT(output, 2);
				for (uint32_t i = 0; i < sc.registers_per_thread; ++i) {
					sprintf(output + strlen(output), "\
			temp_%d = sdata[(gl_LocalInvocationID.x)+%d];\n", i + j * sc.registers_per_thread, i * sc.localSize[0]);
				}
			}
			break;
		}
		}
	}
	static inline void appendRadixShuffleNonStridedBoost4x(char* output, VkFFTSpecializationConstantsLayout sc, const char* floatType, const char* uintType, uint32_t stageSize, uint32_t stageSizeSum, double stageAngle, uint32_t stageRadix) {
		char vecType[10];
		if (!strcmp(floatType, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "dvec2");
		switch (stageRadix) {
		case 2:
		{
			//not implemented yet
			break;
		}
		case 4:
		{
			//not implemented yet

			char stageNormalization[10] = "";
			if ((sc.inverse) && (sc.normalize))
				sprintf(stageNormalization, " * 0.25");
			sprintf(output + strlen(output), "\
{\n\
		stageInvocationID = (gl_LocalInvocationID.x) %% (%d);\n\
		blockInvocationID = (gl_LocalInvocationID.x) - stageInvocationID;\n\
		inoutID = stageInvocationID + blockInvocationID * 4;\n\
		stageInvocationID = (gl_LocalInvocationID.x + gl_WorkGroupSize.x) %% (%d);\n\
		blockInvocationID = (gl_LocalInvocationID.x + gl_WorkGroupSize.x) - stageInvocationID;\n\
		%s inoutID2 = stageInvocationID + blockInvocationID * 4;\n", stageSize, stageSize, uintType);
			for (uint32_t j = 0; j < 4; ++j) {
				appendBarrierVkFFT(output, 2);
				for (uint32_t i = 0; i < 4; ++i) {
					sprintf(output + strlen(output), "\
			sdata[inoutID + %d] = temp_%d%s;\n\
			sdata[inoutID2 + %d] = temp_%d%s;\n", i * stageSize, 2 * i + j * 8, stageNormalization, i * stageSize, 2 * i + j * 8 + 1, stageNormalization);
				}
				appendBarrierVkFFT(output, 2);
				for (uint32_t i = 0; i < 4; ++i) {
					sprintf(output + strlen(output), "\
			temp_%d = sdata[(gl_LocalInvocationID.x)+%d];\n\
			temp_%d = sdata[(gl_LocalInvocationID.x)+%d];\n", 2 * i + j * 8, i * sc.localSize[0], 2 * i + j * 8 + 1, (i + 4) * sc.localSize[0]);
				}
			}
			sprintf(output + strlen(output), "\
}\n");
			break;
		}
		case 8:
		{
			char stageNormalization[10] = "";
			if ((sc.inverse) && (sc.normalize))
				sprintf(stageNormalization, " * 0.125");
			sprintf(output + strlen(output), "\
		stageInvocationID = (gl_LocalInvocationID.x) %% (%d);\n\
		blockInvocationID = (gl_LocalInvocationID.x) - stageInvocationID;\n\
		inoutID = stageInvocationID + blockInvocationID * 8;\n", stageSize);
			for (uint32_t j = 0; j < 4; ++j) {
				appendBarrierVkFFT(output, 2);
				if ((stageSize == 1) && (sc.cacheShuffle)) {
					for (uint32_t i = 0; i < sc.registers_per_thread; i++) {
						sprintf(output + strlen(output), "\
			sdata[inoutID + ((%d+tshuffle)%%(%d))*%d] = temp_%d%s;\n", i, sc.registers_per_thread, stageSize, i + j * sc.registers_per_thread, stageNormalization);
					}
				}
				else {
					for (uint32_t i = 0; i < sc.registers_per_thread; ++i) {
						sprintf(output + strlen(output), "\
			sdata[inoutID + %d] = temp_%d%s;\n", i * stageSize, i + j * sc.registers_per_thread, stageNormalization);
					}
				}
				appendBarrierVkFFT(output, 2);
				for (uint32_t i = 0; i < sc.registers_per_thread; ++i) {
					sprintf(output + strlen(output), "\
			temp_%d = sdata[(gl_LocalInvocationID.x)+%d];\n", i + j * sc.registers_per_thread, i * sc.localSize[0]);
				}
			}
			break;
		}
		}
	}

	static inline void appendRadixShuffleStrided(char* output, VkFFTSpecializationConstantsLayout sc, const char* floatType, const char* uintType, uint32_t stageSize, uint32_t stageSizeSum, double stageAngle, uint32_t stageRadix) {
		char vecType[10];
		if (!strcmp(floatType, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "dvec2");

		char stageNormalization[10] = "";
		uint32_t logicalRegistersPerThread = stageRadix * (sc.registers_per_thread / stageRadix);
		uint32_t logicalGroupSize = sc.fftDim / logicalRegistersPerThread;
		if ((sc.localSize[1] * logicalRegistersPerThread > sc.fftDim) || (stageSize < sc.fftDim / stageRadix) || ((sc.convolutionStep) && ((sc.matrixConvolution > 1) || (sc.numKernels > 1)) && (stageAngle > 0)))
			appendBarrierVkFFT(output, 2);
		if (((sc.inverse) && (sc.normalize)) || ((sc.convolutionStep) && (stageAngle < 0)))
			sprintf(stageNormalization, " / %d", stageRadix);
		if (sc.localSize[1] * logicalRegistersPerThread > sc.fftDim)
			sprintf(output + strlen(output), "\
	if (gl_GlobalInvocationID.y * %d < %d) {\n", logicalRegistersPerThread, sc.fftDim);
		if ((sc.localSize[1] * logicalRegistersPerThread > sc.fftDim) || (stageSize < sc.fftDim / stageRadix) || ((sc.convolutionStep) && ((sc.matrixConvolution > 1) || (sc.numKernels > 1)) && (stageAngle > 0))) {
			//appendBarrierVkFFT(output, 2);
			for (uint32_t j = 0; j < logicalRegistersPerThread / stageRadix; j++) {
				sprintf(output + strlen(output), "\
		stageInvocationID = (gl_LocalInvocationID.y + %d) %% (%d);\n\
		blockInvocationID = (gl_LocalInvocationID.y + %d) - stageInvocationID;\n\
		inoutID = stageInvocationID + blockInvocationID * %d;\n", j * logicalGroupSize, stageSize, j * logicalGroupSize, stageRadix);
				for (uint32_t i = 0; i < stageRadix; i++) {
					sprintf(output + strlen(output), "\
		sdata[gl_WorkGroupSize.x*(inoutID+%d)+gl_LocalInvocationID.x] = temp_%d%s;\n", i * stageSize, j + i * logicalRegistersPerThread / stageRadix, stageNormalization);
				}
			}
		}
		else {
			if (((sc.inverse) && (sc.normalize)) || ((sc.convolutionStep) && (stageAngle < 0))) {
				for (uint32_t i = 0; i < logicalRegistersPerThread; i++) {
					sprintf(output + strlen(output), "\
		temp_%d = temp_%d%s;\n", i, i, stageNormalization);
				}
			}
		}
		if (sc.localSize[1] * logicalRegistersPerThread > sc.fftDim)
			sprintf(output + strlen(output), "	}\n");
	}
	static inline void appendRadixShuffle(char* output, VkFFTSpecializationConstantsLayout sc, const char* floatType, const char* uintType, uint32_t stageSize, uint32_t stageSizeSum, double stageAngle, uint32_t stageRadix, uint32_t shuffleType) {
		switch (shuffleType) {
		case 0: case 5: case 6: {
			appendRadixShuffleNonStrided(output, sc, floatType, uintType, stageSize, stageSizeSum, stageAngle, stageRadix);
			//appendBarrierVkFFT(output, 1);
			break;
		}
		case 1: case 2: {
			appendRadixShuffleStrided(output, sc, floatType, uintType, stageSize, stageSizeSum, stageAngle, stageRadix);
			//appendBarrierVkFFT(output, 1);
			break;
		}
		case 3: {
			appendRadixShuffleNonStridedBoost2x(output, sc, floatType, uintType, stageSize, stageSizeSum, stageAngle, stageRadix);
			//appendBarrierVkFFT(output, 1);
			break;
		}
		case 4: {
			appendRadixShuffleNonStridedBoost4x(output, sc, floatType, uintType, stageSize, stageSizeSum, stageAngle, stageRadix);
			//appendBarrierVkFFT(output, 1);
			break;
		}
		}
	}

	static inline void appendCoordinateRegisterStore(char* output, VkFFTSpecializationConstantsLayout sc, uint32_t readType) {
		switch (readType) {
		case 0://single_c2c
		{
			appendBarrierVkFFT(output, 1);
			if (sc.matrixConvolution == 1) {
				sprintf(output + strlen(output), "\
		temp_0 = sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x];\n\
		temp_1 = sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + gl_WorkGroupSize.x];\n\
		temp_2 = sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 2 * gl_WorkGroupSize.x];\n\
		temp_3 = sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 3 * gl_WorkGroupSize.x];\n\
		temp_4 = sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 4 * gl_WorkGroupSize.x];\n\
		temp_5 = sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 5 * gl_WorkGroupSize.x];\n\
		temp_6 = sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 6 * gl_WorkGroupSize.x];\n\
		temp_7 = sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 7 * gl_WorkGroupSize.x];\n");
				//appendBarrierVkFFT(output, 3);
			}
			else {
				sprintf(output + strlen(output), "\
	switch (coordinate) {\n\
	case 0:\n\
		temp_0 = sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x];\n\
		temp_1 = sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + gl_WorkGroupSize.x];\n\
		temp_2 = sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 2 * gl_WorkGroupSize.x];\n\
		temp_3 = sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 3 * gl_WorkGroupSize.x];\n\
		temp_4 = sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 4 * gl_WorkGroupSize.x];\n\
		temp_5 = sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 5 * gl_WorkGroupSize.x];\n\
		temp_6 = sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 6 * gl_WorkGroupSize.x];\n\
		temp_7 = sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 7 * gl_WorkGroupSize.x];\n");
				//appendBarrierVkFFT(output, 3);
				sprintf(output + strlen(output), "			break;\n");
				for (uint32_t i = 1; i < sc.matrixConvolution; i++) {
					sprintf(output + strlen(output), "\
	case %d:\n\
		temp_0_%d = sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x];\n\
		temp_1_%d = sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + gl_WorkGroupSize.x];\n\
		temp_2_%d = sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 2 * gl_WorkGroupSize.x];\n\
		temp_3_%d = sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 3 * gl_WorkGroupSize.x];\n\
		temp_4_%d = sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 4 * gl_WorkGroupSize.x];\n\
		temp_5_%d = sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 5 * gl_WorkGroupSize.x];\n\
		temp_6_%d = sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 6 * gl_WorkGroupSize.x];\n\
		temp_7_%d = sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 7 * gl_WorkGroupSize.x];\n", i, i, i, i, i, i, i, i, i);
					//appendBarrierVkFFT(output, 3);
					sprintf(output + strlen(output), "			break;\n");
				}
				sprintf(output + strlen(output), "		}\n");
			}
			break;
		}
		case 1://grouped_c2c
		{
			appendBarrierVkFFT(output, 1);
			if (sc.matrixConvolution == 1) {
				sprintf(output + strlen(output), "\
		temp_0 = sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y)+gl_LocalInvocationID.x];\n\
		temp_1 = sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+gl_WorkGroupSize.y)+gl_LocalInvocationID.x];\n\
		temp_2 = sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+2*gl_WorkGroupSize.y)+gl_LocalInvocationID.x];\n\
		temp_3 = sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+3*gl_WorkGroupSize.y)+gl_LocalInvocationID.x];\n\
		temp_4 = sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+4*gl_WorkGroupSize.y)+gl_LocalInvocationID.x];\n\
		temp_5 = sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+5*gl_WorkGroupSize.y)+gl_LocalInvocationID.x];\n\
		temp_6 = sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+6*gl_WorkGroupSize.y)+gl_LocalInvocationID.x];\n\
		temp_7 = sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+7*gl_WorkGroupSize.y)+gl_LocalInvocationID.x];\n");
				//appendBarrierVkFFT(output, 3);
			}
			else {
				sprintf(output + strlen(output), "\
	switch (coordinate) {\n\
	case 0:\n\
		temp_0 = sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y)+gl_LocalInvocationID.x];\n\
		temp_1 = sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+gl_WorkGroupSize.y)+gl_LocalInvocationID.x];\n\
		temp_2 = sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+2*gl_WorkGroupSize.y)+gl_LocalInvocationID.x];\n\
		temp_3 = sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+3*gl_WorkGroupSize.y)+gl_LocalInvocationID.x];\n\
		temp_4 = sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+4*gl_WorkGroupSize.y)+gl_LocalInvocationID.x];\n\
		temp_5 = sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+5*gl_WorkGroupSize.y)+gl_LocalInvocationID.x];\n\
		temp_6 = sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+6*gl_WorkGroupSize.y)+gl_LocalInvocationID.x];\n\
		temp_7 = sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+7*gl_WorkGroupSize.y)+gl_LocalInvocationID.x];\n");
				//appendBarrierVkFFT(output, 3);
				sprintf(output + strlen(output), "			break;\n");
				for (uint32_t i = 1; i < sc.matrixConvolution; i++) {
					sprintf(output + strlen(output), "\
	case %d:\n\
		temp_0_%d = sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y)+gl_LocalInvocationID.x];\n\
		temp_1_%d = sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+gl_WorkGroupSize.y)+gl_LocalInvocationID.x];\n\
		temp_2_%d = sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+2*gl_WorkGroupSize.y)+gl_LocalInvocationID.x];\n\
		temp_3_%d = sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+3*gl_WorkGroupSize.y)+gl_LocalInvocationID.x];\n\
		temp_4_%d = sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+4*gl_WorkGroupSize.y)+gl_LocalInvocationID.x];\n\
		temp_5_%d = sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+5*gl_WorkGroupSize.y)+gl_LocalInvocationID.x];\n\
		temp_6_%d = sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+6*gl_WorkGroupSize.y)+gl_LocalInvocationID.x];\n\
		temp_7_%d = sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+7*gl_WorkGroupSize.y)+gl_LocalInvocationID.x];\n", i, i, i, i, i, i, i, i, i);
					//appendBarrierVkFFT(output, 3);
					sprintf(output + strlen(output), "			break;\n");
				}
				sprintf(output + strlen(output), "		}\n");
			}

			break;
		}
		}
	}
	static inline void appendCoordinateRegisterPull(char* output, VkFFTSpecializationConstantsLayout sc, uint32_t readType) {
		switch (readType) {
		case 0://single_c2c
		{
			appendBarrierVkFFT(output, 1);
			if (sc.matrixConvolution == 1) {
				sprintf(output + strlen(output), "\
			sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x] = temp_0;\n\
			sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + gl_WorkGroupSize.x] = temp_1;\n\
			sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 2 * gl_WorkGroupSize.x] = temp_2;\n\
			sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 3 * gl_WorkGroupSize.x] = temp_3;\n\
			sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 4 * gl_WorkGroupSize.x] = temp_4;\n\
			sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 5 * gl_WorkGroupSize.x] = temp_5;\n\
			sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 6 * gl_WorkGroupSize.x] = temp_6;\n\
			sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 7 * gl_WorkGroupSize.x] = temp_7;\n");
				//appendBarrierVkFFT(output, 3);
			}
			else {
				sprintf(output + strlen(output), "\
		switch (coordinate) {\n\
		case 0:\n\
			sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x] = temp_0;\n\
			sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + gl_WorkGroupSize.x] = temp_1;\n\
			sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 2 * gl_WorkGroupSize.x] = temp_2;\n\
			sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 3 * gl_WorkGroupSize.x] = temp_3;\n\
			sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 4 * gl_WorkGroupSize.x] = temp_4;\n\
			sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 5 * gl_WorkGroupSize.x] = temp_5;\n\
			sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 6 * gl_WorkGroupSize.x] = temp_6;\n\
			sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 7 * gl_WorkGroupSize.x] = temp_7;\n");
				//appendBarrierVkFFT(output, 3);
				sprintf(output + strlen(output), "			break;\n");
				for (uint32_t i = 1; i < sc.matrixConvolution; i++) {
					sprintf(output + strlen(output), "\
		case %d:\n\
			sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x] = temp_0_%d;\n\
			sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + gl_WorkGroupSize.x] = temp_1_%d;\n\
			sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 2 * gl_WorkGroupSize.x] = temp_2_%d;\n\
			sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 3 * gl_WorkGroupSize.x] = temp_3_%d;\n\
			sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 4 * gl_WorkGroupSize.x] = temp_4_%d;\n\
			sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 5 * gl_WorkGroupSize.x] = temp_5_%d;\n\
			sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 6 * gl_WorkGroupSize.x] = temp_6_%d;\n\
			sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + 7 * gl_WorkGroupSize.x] = temp_7_%d;\n", i, i, i, i, i, i, i, i, i);
					//appendBarrierVkFFT(output, 3);
					sprintf(output + strlen(output), "			break;\n");
				}
				sprintf(output + strlen(output), "		}\n");
			}

			break;
		}
		case 1://grouped_c2c
		{
			appendBarrierVkFFT(output, 1);
			if (sc.matrixConvolution == 1) {
				sprintf(output + strlen(output), "\
		sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y)+gl_LocalInvocationID.x] = temp_0;\n\
		sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+gl_WorkGroupSize.y)+gl_LocalInvocationID.x] = temp_1;\n\
		sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+2*gl_WorkGroupSize.y)+gl_LocalInvocationID.x] = temp_2;\n\
		sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+3*gl_WorkGroupSize.y)+gl_LocalInvocationID.x] = temp_3;\n\
		sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+4*gl_WorkGroupSize.y)+gl_LocalInvocationID.x] = temp_4;\n\
		sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+5*gl_WorkGroupSize.y)+gl_LocalInvocationID.x] = temp_5;\n\
		sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+6*gl_WorkGroupSize.y)+gl_LocalInvocationID.x] = temp_6;\n\
		sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+7*gl_WorkGroupSize.y)+gl_LocalInvocationID.x] = temp_7;\n");
				//appendBarrierVkFFT(output, 3);
			}
			else {
				sprintf(output + strlen(output), "\
	switch (coordinate) {\n\
	case 0:\n\
		sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y)+gl_LocalInvocationID.x] = temp_0;\n\
		sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+gl_WorkGroupSize.y)+gl_LocalInvocationID.x] = temp_1;\n\
		sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+2*gl_WorkGroupSize.y)+gl_LocalInvocationID.x] = temp_2;\n\
		sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+3*gl_WorkGroupSize.y)+gl_LocalInvocationID.x] = temp_3;\n\
		sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+4*gl_WorkGroupSize.y)+gl_LocalInvocationID.x] = temp_4;\n\
		sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+5*gl_WorkGroupSize.y)+gl_LocalInvocationID.x] = temp_5;\n\
		sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+6*gl_WorkGroupSize.y)+gl_LocalInvocationID.x] = temp_6;\n\
		sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+7*gl_WorkGroupSize.y)+gl_LocalInvocationID.x] = temp_7;\n");
				//appendBarrierVkFFT(output, 3);
				sprintf(output + strlen(output), "			break;\n");
				for (uint32_t i = 1; i < sc.matrixConvolution; i++) {
					sprintf(output + strlen(output), "\
	case %d:\n\
		sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y)+gl_LocalInvocationID.x] = temp_0_%d;\n\
		sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+gl_WorkGroupSize.y)+gl_LocalInvocationID.x] = temp_1_%d;\n\
		sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+2*gl_WorkGroupSize.y)+gl_LocalInvocationID.x] = temp_2_%d;\n\
		sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+3*gl_WorkGroupSize.y)+gl_LocalInvocationID.x] = temp_3_%d;\n\
		sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+4*gl_WorkGroupSize.y)+gl_LocalInvocationID.x] = temp_4_%d;\n\
		sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+5*gl_WorkGroupSize.y)+gl_LocalInvocationID.x] = temp_5_%d;\n\
		sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+6*gl_WorkGroupSize.y)+gl_LocalInvocationID.x] = temp_6_%d;\n\
		sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+7*gl_WorkGroupSize.y)+gl_LocalInvocationID.x] = temp_7_%d;\n", i, i, i, i, i, i, i, i, i);
					//appendBarrierVkFFT(output, 3);
					sprintf(output + strlen(output), "			break;\n");
				}
				sprintf(output + strlen(output), "		}\n");
			}
			break;
		}
		}
	}
	static inline void appendPreparationBatchedKernelConvolution(char* output, VkFFTSpecializationConstantsLayout sc, const char* floatType, const char* floatTypeMemory, const char* uintType, uint32_t dataType) {
		char vecType[10];
		if (!strcmp(floatType, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "dvec2");
		char separateRegisterStore[100] = "_store";
		for (uint32_t i = 0; i < sc.registers_per_thread; i++) {
			sprintf(output + strlen(output), "		%s temp_%d%s;\n", vecType, i, separateRegisterStore);
			for (uint32_t j = 1; j < sc.matrixConvolution; j++) {
				sprintf(output + strlen(output), "		%s temp_%d_%d%s;\n", vecType, i, j, separateRegisterStore);
			}
		}
		for (uint32_t i = 0; i < sc.registers_per_thread; i++) {
			//sprintf(output + strlen(output), "			temp%s[i]=temp[i];\n", separateRegisterStore);
			sprintf(output + strlen(output), "			temp_%d%s=temp_%d;\n", i, separateRegisterStore, i);
			for (uint32_t j = 1; j < sc.matrixConvolution; j++) {
				sprintf(output + strlen(output), "			temp_%d_%d%s=temp_%d_%d;\n", i, j, separateRegisterStore, i, j);
			}
		}
		sprintf(output + strlen(output), "	for (uint batchID=0;  batchID < %d; batchID++){\n", sc.numKernels);
	}
	static inline void appendKernelConvolution(char* output, VkFFTSpecializationConstantsLayout sc, const char* floatType, const char* floatTypeMemory, const char* uintType, uint32_t dataType) {
		char shiftX[100] = "";
		if (sc.performWorkGroupShift[0])
			sprintf(shiftX, " + consts.workGroupShiftX * gl_WorkGroupSize.x ");
		char requestCoordinate[100] = "";
		if (sc.convolutionStep) {
			if (sc.matrixConvolution > 1) {
				sprintf(requestCoordinate, ", 0");
			}
		}
		char requestBatch[100] = "";
		char separateRegisterStore[100] = "";
		if (sc.convolutionStep) {
			if (sc.numKernels > 1) {
				sprintf(requestBatch, ", batchID");
				sprintf(separateRegisterStore, "_store");
			}
		}
		for (uint32_t j = 0; j < sc.matrixConvolution; j++) {
			sprintf(output + strlen(output), "		%s temp_real%d = 0;\n", floatType, j);
			sprintf(output + strlen(output), "		%s temp_imag%d = 0;\n", floatType, j);
		}
		for (uint32_t i = 0; i < sc.registers_per_thread; i++) {
			if (i > 0) {
				for (uint32_t j = 0; j < sc.matrixConvolution; j++) {
					sprintf(output + strlen(output), "		temp_real%d = 0;\n", j);
					sprintf(output + strlen(output), "		temp_imag%d = 0;\n", j);
				}
			}
			switch (dataType) {
			case 0:
			{
				if (sc.fftDim == sc.fft_dim_full) {
					if (sc.localSize[1] == 1)
						sprintf(output + strlen(output), "		combinedId = gl_LocalInvocationID.x + %d;\n", i * sc.localSize[0]);
					else
						sprintf(output + strlen(output), "		combinedId = (gl_LocalInvocationID.x + %d * gl_LocalInvocationID.y) + %d;\n", sc.localSize[0], i * sc.localSize[0] * sc.localSize[1]);

					if (sc.inputStride[0] > 1)
						sprintf(output + strlen(output), "		inoutID = indexInput((combinedId %% %d) * %d + (combinedId / %d) * %d%s%s);\n", sc.fftDim, sc.inputStride[0], sc.fftDim, sc.inputStride[1], requestCoordinate, requestBatch);
					else
						sprintf(output + strlen(output), "		inoutID = indexInput((combinedId %% %d) + (combinedId / %d) * %d%s%s);\n", sc.fftDim, sc.fftDim, sc.inputStride[1], requestCoordinate, requestBatch);
				}
				else
					sprintf(output + strlen(output), "		inoutID = indexInput(gl_LocalInvocationID.x+%d+gl_LocalInvocationID.y * %d + (((gl_WorkGroupID.x%s) %% %d) * %d + ((gl_WorkGroupID.x%s) / %d) * %d)%s%s);\n", i * sc.localSize[0], sc.firstStageStartSize, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.localSize[1] * sc.firstStageStartSize, requestCoordinate, requestBatch);
				break;
			}
			case 1:
			{
				sprintf(output + strlen(output), "		inoutID = indexInput((gl_GlobalInvocationID.x%s) %% (%d), (gl_LocalInvocationID.y+%d)+((gl_GlobalInvocationID.x%s)/%d)%%(%d)+((gl_GlobalInvocationID.x%s)/%d)*(%d)%s%s);\n", shiftX, sc.fft_dim_x, i * sc.localSize[1], shiftX, sc.fft_dim_x, sc.stageStartSize, shiftX, sc.fft_dim_x * sc.stageStartSize, sc.fftDim, requestCoordinate, requestBatch);
				break;
			}
			}
			if (sc.kernelBlockNum == 1) {

				for (uint32_t j = 0; j < sc.matrixConvolution; j++) {
					for (uint32_t l = 0; l < sc.matrixConvolution; l++) {
						uint32_t k = 0;
						if (sc.symmetricKernel) {
							k = (l < j) ? (l * sc.matrixConvolution - l * l + j) : (j * sc.matrixConvolution - j * j + l);
						}
						else {
							k = (j * sc.matrixConvolution + l);
						}
						if (l == 0)
							sprintf(output + strlen(output), "		temp_real%d += kernelBlocks[0].kernel[inoutID+%d].x * temp_%d%s.x - kernelBlocks[0].kernel[inoutID+%d].y * temp_%d%s.y;\n", j, k * sc.inputStride[3], i, separateRegisterStore, k * sc.inputStride[3], i, separateRegisterStore);
						else
							sprintf(output + strlen(output), "		temp_real%d += kernelBlocks[0].kernel[inoutID+%d].x * temp_%d_%d%s.x - kernelBlocks[0].kernel[inoutID+%d].y * temp_%d_%d%s.y;\n", j, k * sc.inputStride[3], i, l, separateRegisterStore, k * sc.inputStride[3], i, l, separateRegisterStore);
					}
					for (uint32_t l = 0; l < sc.matrixConvolution; l++) {
						uint32_t k = 0;
						if (sc.symmetricKernel) {
							k = (l < j) ? (l * sc.matrixConvolution - l * l + j) : (j * sc.matrixConvolution - j * j + l);
						}
						else {
							k = (j * sc.matrixConvolution + l);
						}
						if (l == 0)
							sprintf(output + strlen(output), "		temp_imag%d += kernelBlocks[0].kernel[inoutID+%d].x * temp_%d%s.y + kernelBlocks[0].kernel[inoutID+%d].y * temp_%d%s.x;\n", j, k * sc.inputStride[3], i, separateRegisterStore, k * sc.inputStride[3], i, separateRegisterStore);
						else
							sprintf(output + strlen(output), "		temp_imag%d += kernelBlocks[0].kernel[inoutID+%d].x * temp_%d_%d%s.y + kernelBlocks[0].kernel[inoutID+%d].y * temp_%d_%d%s.x;\n", j, k * sc.inputStride[3], i, l, separateRegisterStore, k * sc.inputStride[3], i, l, separateRegisterStore);

					}
				}
				sprintf(output + strlen(output), "		temp_%d.x = temp_real0;", i);
				sprintf(output + strlen(output), "		temp_%d.y = temp_imag0;", i);
				for (uint32_t l = 1; l < sc.matrixConvolution; l++) {
					sprintf(output + strlen(output), "		temp_%d_%d.x = temp_real%d;", i, l, l);
					sprintf(output + strlen(output), "		temp_%d_%d.y = temp_imag%d;", i, l, l);
				}
			}
			else {
				for (uint32_t j = 0; j < sc.matrixConvolution; j++) {

					sprintf(output + strlen(output), "		%s temp_real%d = 0;\n", floatType, j);
					for (uint32_t l = 0; l < sc.matrixConvolution; l++) {
						uint32_t k = 0;
						if (sc.symmetricKernel) {
							k = (l < j) ? (l * sc.matrixConvolution - l * l + j) : (j * sc.matrixConvolution - j * j + l);
						}
						else {
							k = (j * sc.matrixConvolution + l);
						}
						if (l == 0)
							sprintf(output + strlen(output), "		temp_real%d += kernelBlocks[(inoutID+%d)/%d].kernel[(inoutID+%d) %% %d].x * temp_%d%s.x - kernelBlocks[(inoutID+%d)/%d].kernel[(inoutID+%d) %% %d].y * temp_%d%s.y;\n", j, k * sc.inputStride[3], sc.kernelBlockSize, k * sc.inputStride[3], sc.kernelBlockSize, i, separateRegisterStore, k * sc.inputStride[3], sc.kernelBlockSize, k * sc.inputStride[3], sc.kernelBlockSize, i, separateRegisterStore);
						else
							sprintf(output + strlen(output), "		temp_real%d += kernelBlocks[(inoutID+%d)/%d].kernel[(inoutID+%d) %% %d].x * temp_%d_%d%s.x - kernelBlocks[(inoutID+%d)/%d].kernel[(inoutID+%d) %% %d].y * temp_%d_%d%s.y;\n", j, k * sc.inputStride[3], sc.kernelBlockSize, k * sc.inputStride[3], sc.kernelBlockSize, i, l, separateRegisterStore, k * sc.inputStride[3], sc.kernelBlockSize, k * sc.inputStride[3], sc.kernelBlockSize, i, l, separateRegisterStore);

					}

					sprintf(output + strlen(output), "		%s temp_imag%d = 0;\n", floatType, j);
					for (uint32_t l = 0; l < sc.matrixConvolution; l++) {
						uint32_t k = 0;
						if (sc.symmetricKernel) {
							k = (l < j) ? (l * sc.matrixConvolution - l * l + j) : (j * sc.matrixConvolution - j * j + l);
						}
						else {
							k = (j * sc.matrixConvolution + l);
						}
						if (l == 0)
							sprintf(output + strlen(output), "		temp_imag%d += kernelBlocks[(inoutID+%d)/%d].kernel[(inoutID+%d) %% %d].x * temp_%d%s.y + kernelBlocks[(inoutID+%d)/%d].kernel[(inoutID+%d) %% %d].y * temp_%d%s.x;\n", j, k * sc.inputStride[3], sc.kernelBlockSize, k * sc.inputStride[3], sc.kernelBlockSize, i, separateRegisterStore, k * sc.inputStride[3], sc.kernelBlockSize, k * sc.inputStride[3], sc.kernelBlockSize, i, separateRegisterStore);
						else
							sprintf(output + strlen(output), "		temp_imag%d += kernelBlocks[(inoutID+%d)/%d].kernel[(inoutID+%d) %% %d].x * temp_%d_%d%s.y + kernelBlocks[(inoutID+%d)/%d].kernel[(inoutID+%d) %% %d].y * temp_%d_%d%s.x;\n", j, k * sc.inputStride[3], sc.kernelBlockSize, k * sc.inputStride[3], sc.kernelBlockSize, i, l, separateRegisterStore, k * sc.inputStride[3], sc.kernelBlockSize, k * sc.inputStride[3], sc.kernelBlockSize, i, l, separateRegisterStore);
					}
				}
				sprintf(output + strlen(output), "		temp_%d.x = temp_real0;", i);
				sprintf(output + strlen(output), "		temp_%d.y = temp_imag0;", i);
				for (uint32_t l = 1; l < sc.matrixConvolution; l++) {
					sprintf(output + strlen(output), "		temp_%d_%d.x = temp_real%d;", i, l, l);
					sprintf(output + strlen(output), "		temp_%d_%d.y = temp_imag%d;", i, l, l);
				}
			}
		}
	}
	static inline void appendWriteDataVkFFT(char* output, VkFFTSpecializationConstantsLayout sc, const char* floatType, const char* floatTypeMemory, const char* uintType, uint32_t writeType) {
		char vecType[10];
		if (!strcmp(floatType, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "dvec2");
		char convTypeLeft[10] = "";
		char convTypeRight[10] = "";
		if ((!strcmp(floatTypeMemory, "half")) && (strcmp(floatType, "half"))) {
			if (writeType == 6) {
				sprintf(convTypeLeft, "float16_t(");
				sprintf(convTypeRight, ")");
			}
			else {
				sprintf(convTypeLeft, "f16vec2(");
				sprintf(convTypeRight, ")");
			}
		}
		if ((!strcmp(floatTypeMemory, "float")) && (strcmp(floatType, "float"))) {
			if (writeType == 6) {
				sprintf(convTypeLeft, "float(");
				sprintf(convTypeRight, ")");
			}
			else {
				sprintf(convTypeLeft, "vec2(");
				sprintf(convTypeRight, ")");
			}
		}
		if ((!strcmp(floatTypeMemory, "double")) && (strcmp(floatType, "double"))) {
			if (writeType == 6) {
				sprintf(convTypeLeft, "double(");
				sprintf(convTypeRight, ")");
			}
			else {
				sprintf(convTypeLeft, "dvec2(");
				sprintf(convTypeRight, ")");
			}
		}
		char requestCoordinate[100] = "";
		if (sc.convolutionStep) {
			if (sc.matrixConvolution > 1) {
				sprintf(requestCoordinate, ", coordinate");
			}
		}
		char requestBatch[100] = "";
		if (sc.convolutionStep) {
			if (sc.numKernels > 1) {
				sprintf(requestBatch, ", batchID");//if one buffer - multiple kernel convolution
			}
		}
		switch (writeType) {
		case 0:
		{//single_c2c
			if ((sc.localSize[1] > 1) || (sc.localSize[0] * sc.stageRadix[sc.numStages - 1] * (sc.registers_per_thread / sc.stageRadix[sc.numStages - 1]) > sc.fftDim)) {
				sc.writeFromRegisters = 0;
				appendBarrierVkFFT(output, 1);
			}
			else
				sc.writeFromRegisters = 1;
			char shiftX[100] = "";
			if (sc.performWorkGroupShift[0])
				sprintf(shiftX, " + consts.workGroupShiftX ");
			char shiftY[100] = "";
			if (sc.performWorkGroupShift[1])
				sprintf(shiftY, " + consts.workGroupShiftY*gl_WorkGroupSize.y ");
			char shiftY2[100] = "";
			if (sc.performWorkGroupShift[1])
				sprintf(shiftY, " + consts.workGroupShiftY ");
			if (sc.fftDim < sc.fft_dim_full) {
				//sprintf(output + strlen(output), "		if(gl_LocalInvocationID.y * %d + (((gl_WorkGroupID.x%s) %% %d) * %d + ((gl_WorkGroupID.x%s) / %d) * %d) < %d) {;;\n", sc.firstStageStartSize, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.localSize[1] * sc.firstStageStartSize, sc.size[0]);
				//sprintf(output + strlen(output), "		if ((gl_LocalInvocationID.x + %d * gl_LocalInvocationID.y) %% %d + ((gl_WorkGroupID.x%s) / %d)*%d < %d){\n", sc.localSize[0], sc.localSize[1], shiftX, sc.firstStageStartSize / sc.fftDim, sc.localSize[1], sc.fft_dim_full / sc.fftDim);
				//sprintf(output + strlen(output), "		if ((gl_LocalInvocationID.x + %d * gl_LocalInvocationID.y+432) %% %d + (((gl_LocalInvocationID.x + %d * gl_LocalInvocationID.y+432)/%d) * %d)+ ((gl_WorkGroupID.x%s) / %d)*%d + ((gl_WorkGroupID.x%s) %% %d) * %d< %d){\n", sc.localSize[0], sc.localSize[1], sc.localSize[0], sc.localSize[1], sc.fft_dim_full / sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.localSize[1], shiftX, sc.firstStageStartSize / sc.fftDim, sc.fft_dim_full / sc.firstStageStartSize, sc.fft_dim_full / sc.fftDim);
				sprintf(output + strlen(output), "		if (((gl_LocalInvocationID.x + %d * gl_LocalInvocationID.y) %% %d + ((gl_WorkGroupID.x%s) / %d)*%d < %d)){\n", sc.localSize[0], sc.localSize[1], shiftX, sc.firstStageStartSize / sc.fftDim, sc.localSize[1], sc.fft_dim_full / sc.firstStageStartSize);
			}
			else {
				sprintf(output + strlen(output), "		{ \n");
			}
			if (sc.reorderFourStep) {
				if (sc.zeropad[1]) {
					if (sc.fftDim == sc.fft_dim_full) {
						for (uint32_t i = 0; i < sc.min_registers_per_thread; i++) {

							if (sc.localSize[1] == 1)
								sprintf(output + strlen(output), "		combinedId = gl_LocalInvocationID.x + %d;\n", i * sc.localSize[0]);
							else
								sprintf(output + strlen(output), "		combinedId = (gl_LocalInvocationID.x + %d * gl_LocalInvocationID.y) + %d;\n", sc.localSize[0], i * sc.localSize[0] * sc.localSize[1]);

							if (sc.outputStride[0] > 1)
								sprintf(output + strlen(output), "		inoutID = (combinedId %% %d) * %d + (combinedId / %d) * %d;\n", sc.fftDim, sc.outputStride[0], sc.fftDim, sc.outputStride[1]);
							else
								sprintf(output + strlen(output), "		inoutID = (combinedId %% %d) + (combinedId / %d) * %d;\n", sc.fftDim, sc.fftDim, sc.outputStride[1]);
							if (sc.size[sc.axis_id + 1] % sc.localSize[1] != 0)
								sprintf(output + strlen(output), "		if(combinedId / %d + (gl_WorkGroupID.y%s)*gl_WorkGroupSize.y< %d){", sc.fftDim, shiftY2, sc.size[sc.axis_id + 1]);
							sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d)){\n", sc.fft_dim_full, sc.fft_zeropad_left_write[sc.axis_id], sc.fft_dim_full, sc.fft_zeropad_right_write[sc.axis_id]);
							if (sc.writeFromRegisters) {
								if (sc.outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[indexOutput(inoutID%s%s)] = %stemp_%d%s;\n", requestCoordinate, requestBatch, convTypeLeft, i, convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[indexOutput(inoutID%s%s) / %d].outputs[indexOutput(inoutID%s%s) %% %d] = %stemp_%d%s;\n", requestCoordinate, requestBatch, sc.outputBufferBlockSize, requestCoordinate, requestBatch, sc.outputBufferBlockSize, convTypeLeft, i, convTypeRight);
							}
							else {
								if (sc.outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[indexOutput(inoutID%s%s)] = %ssdata[(combinedId %% %d) + (combinedId / %d) * sharedStride]%s;\n", convTypeLeft, requestCoordinate, requestBatch, sc.fftDim, sc.fftDim, convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[indexOutput(inoutID%s%s) / %d].outputs[indexOutput(inoutID%s%s) %% %d] = %ssdata[(combinedId %% %d) + (combinedId / %d) * sharedStride]%s;\n", requestCoordinate, requestBatch, sc.outputBufferBlockSize, requestCoordinate, requestBatch, sc.outputBufferBlockSize, convTypeLeft, sc.fftDim, sc.fftDim, convTypeRight);
							}
							sprintf(output + strlen(output), "	}\n");
							if (sc.size[sc.axis_id + 1] % sc.localSize[1] != 0)
								sprintf(output + strlen(output), "		}");

						}
					}
					else {
						for (uint32_t i = 0; i < sc.min_registers_per_thread; i++) {

							if (sc.localSize[1] == 1)
								sprintf(output + strlen(output), "		combinedId = gl_LocalInvocationID.x + %d;\n", i * sc.localSize[0]);
							else
								sprintf(output + strlen(output), "		combinedId = (gl_LocalInvocationID.x + %d * gl_LocalInvocationID.y) + %d;\n", sc.localSize[0], i * sc.localSize[0] * sc.localSize[1]);
							if (sc.localSize[1] == 1)
								sprintf(output + strlen(output), "		inoutID = (gl_WorkGroupID.x%s)/%d+ (combinedId * %d)+ ((gl_WorkGroupID.x%s) %% %d) * %d;\n", shiftX, sc.firstStageStartSize / sc.fftDim, sc.fft_dim_full / sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fft_dim_full / sc.firstStageStartSize);
							else
								sprintf(output + strlen(output), "		inoutID = combinedId %% %d + ((gl_WorkGroupID.x%s) / %d)*%d + ((combinedId/%d) * %d)+ ((gl_WorkGroupID.x%s) %% %d) * %d;\n", sc.localSize[1], shiftX, sc.firstStageStartSize / sc.fftDim, sc.localSize[1], sc.localSize[1], sc.fft_dim_full / sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fft_dim_full / sc.firstStageStartSize);

							sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d)){\n", sc.fft_dim_full, sc.fft_zeropad_left_write[sc.axis_id], sc.fft_dim_full, sc.fft_zeropad_right_write[sc.axis_id]);
							if (sc.writeFromRegisters) {
								if (sc.outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[indexOutput(inoutID%s%s)] = %stemp_%d%s;\n", requestCoordinate, requestBatch, convTypeLeft, i, convTypeRight);
								else
									sprintf(output + strlen(output), "			outputBlocks[indexOutput(inoutID%s%s) / %d].outputs[indexOutput(inoutID%s%s) %% %d] = %stemp_%d%s;\n", requestCoordinate, requestBatch, sc.outputBufferBlockSize, requestCoordinate, requestBatch, sc.outputBufferBlockSize, convTypeLeft, i, convTypeRight);
							}
							else {
								if (sc.outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "			outputBlocks[0].outputs[indexOutput(inoutID%s%s)] = %ssdata[(combinedId %% gl_WorkGroupSize.y)*sharedStride+combinedId/gl_WorkGroupSize.y]%s;\n", requestCoordinate, requestBatch, convTypeLeft, convTypeRight);
								else
									sprintf(output + strlen(output), "			outputBlocks[indexOutput(inoutID%s%s) / %d].outputs[indexOutput(inoutID%s%s) %% %d] = %ssdata[(combinedId %% gl_WorkGroupSize.y)*sharedStride+combinedId/gl_WorkGroupSize.y]%s;\n", requestCoordinate, requestBatch, sc.outputBufferBlockSize, requestCoordinate, requestBatch, sc.outputBufferBlockSize, convTypeLeft, convTypeRight);
							}
							/*
							if (sc.outputBufferBlockNum == 1)
								if (sc.localSize[1] == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[indexOutput(inoutID%s%s)] = %stemp_%d%s;\n", requestCoordinate, requestBatch, convTypeLeft, i, convTypeRight);
								else
									sprintf(output + strlen(output), "			outputBlocks[0].outputs[indexOutput(inoutID%s%s)] = %stemp_%d%s;\n", requestCoordinate, requestBatch, convTypeLeft, i, convTypeRight);
							else
								if (sc.localSize[1] == 1)
									sprintf(output + strlen(output), "			outputBlocks[indexOutput(inoutID%s%s) / %d].outputs[indexOutput(inoutID%s%s) %% %d] = %stemp_%d%s;\n", requestCoordinate, requestBatch, sc.outputBufferBlockSize, requestCoordinate, requestBatch, sc.outputBufferBlockSize, convTypeLeft, i, convTypeRight);
								else
									sprintf(output + strlen(output), "			outputBlocks[indexOutput(inoutID%s%s) / %d].outputs[indexOutput(inoutID%s%s) %% %d] = %stemp_%d%s;\n", requestCoordinate, requestBatch, sc.outputBufferBlockSize, requestCoordinate, requestBatch, sc.outputBufferBlockSize, convTypeLeft, i, convTypeRight);
							*/
							sprintf(output + strlen(output), "	}n");
						}
					}
				}
				else {
					if (sc.fftDim == sc.fft_dim_full) {
						for (uint32_t i = 0; i < sc.min_registers_per_thread; i++) {

							if (sc.localSize[1] == 1)
								sprintf(output + strlen(output), "		combinedId = gl_LocalInvocationID.x + %d;\n", i * sc.localSize[0]);
							else
								sprintf(output + strlen(output), "		combinedId = (gl_LocalInvocationID.x + %d * gl_LocalInvocationID.y) + %d;\n", sc.localSize[0], i * sc.localSize[0] * sc.localSize[1]);

							if (sc.outputStride[0] > 1)
								sprintf(output + strlen(output), "		inoutID = indexOutput((combinedId %% %d) * %d + (combinedId / %d) * %d%s%s);\n", sc.fftDim, sc.outputStride[0], sc.fftDim, sc.outputStride[1], requestCoordinate, requestBatch);
							else
								sprintf(output + strlen(output), "		inoutID = indexOutput((combinedId %% %d) + (combinedId / %d) * %d%s%s);\n", sc.fftDim, sc.fftDim, sc.outputStride[1], requestCoordinate, requestBatch);
							if (sc.size[sc.axis_id + 1] % sc.localSize[1] != 0)
								sprintf(output + strlen(output), "		if(combinedId / %d + gl_WorkGroupID.y*gl_WorkGroupSize.y< %d){", sc.fftDim, sc.size[sc.axis_id + 1]);
							if (sc.writeFromRegisters) {
								if (sc.outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %stemp_%d%s;\n", convTypeLeft, i, convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %stemp_%d%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, i, convTypeRight);
							}
							else {
								if (sc.outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %ssdata[(combinedId %% %d) + (combinedId / %d) * sharedStride]%s;\n", convTypeLeft, sc.fftDim, sc.fftDim, convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %ssdata[(combinedId %% %d) + (combinedId / %d) * sharedStride]%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, sc.fftDim, sc.fftDim, convTypeRight);
							}
							if (sc.size[sc.axis_id + 1] % sc.localSize[1] != 0)
								sprintf(output + strlen(output), "		}");

						}
					}
					else {
						for (uint32_t i = 0; i < sc.min_registers_per_thread; i++) {
							if (sc.localSize[1] == 1)
								sprintf(output + strlen(output), "		combinedId = gl_LocalInvocationID.x + %d;\n", i * sc.localSize[0]);
							else
								sprintf(output + strlen(output), "		combinedId = (gl_LocalInvocationID.x + %d * gl_LocalInvocationID.y) + %d;\n", sc.localSize[0], i * sc.localSize[0] * sc.localSize[1]);
							if (sc.localSize[1] == 1)
								sprintf(output + strlen(output), "		inoutID = indexOutput((gl_WorkGroupID.x%s)/%d+ (combinedId * %d)+ ((gl_WorkGroupID.x%s) %% %d) * %d%s%s);\n", shiftX, sc.firstStageStartSize / sc.fftDim, sc.fft_dim_full / sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fft_dim_full / sc.firstStageStartSize, requestCoordinate, requestBatch);
							else
								sprintf(output + strlen(output), "		inoutID = indexOutput(combinedId %% %d + ((gl_WorkGroupID.x%s) / %d)*%d + ((combinedId/%d) * %d)+ ((gl_WorkGroupID.x%s) %% %d) * %d%s%s);\n", sc.localSize[1], shiftX, sc.firstStageStartSize / sc.fftDim, sc.localSize[1], sc.localSize[1], sc.fft_dim_full / sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fft_dim_full / sc.firstStageStartSize, requestCoordinate, requestBatch);
							if (sc.writeFromRegisters) {
								if (sc.outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %stemp_%d%s;\n", convTypeLeft, i, convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %stemp_%d%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, i, convTypeRight);
							}
							else {
								if (sc.outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %ssdata[(combinedId %% gl_WorkGroupSize.y)*sharedStride+combinedId/gl_WorkGroupSize.y]%s;\n", convTypeLeft, convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %ssdata[(combinedId %% gl_WorkGroupSize.y)*sharedStride+combinedId/gl_WorkGroupSize.y]%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, convTypeRight);

							}
							/*if (sc.outputBufferBlockNum == 1)
								if (sc.localSize[1] == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %stemp_%d%s;\n", convTypeLeft, i, convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %stemp_%d%s;\n", convTypeLeft, i, convTypeRight);
							else
								if (sc.localSize[1] == 1)
									sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %stemp_%d%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, i, convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %stemp_%d%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, i, convTypeRight);
							*/
						}
					}
				}
			}
			else {
				if (sc.zeropad[1]) {
					if (sc.fftDim == sc.fft_dim_full) {
						for (uint32_t i = 0; i < sc.min_registers_per_thread; i++) {

							if (sc.localSize[1] == 1)
								sprintf(output + strlen(output), "		combinedId = gl_LocalInvocationID.x + %d;\n", i * sc.localSize[0]);
							else
								sprintf(output + strlen(output), "		combinedId = (gl_LocalInvocationID.x + %d * gl_LocalInvocationID.y) + %d;\n", sc.localSize[0], i * sc.localSize[0] * sc.localSize[1]);

							if (sc.outputStride[0] > 1)
								sprintf(output + strlen(output), "		inoutID = (combinedId %% %d) * %d + (combinedId / %d) * %d;\n", sc.fftDim, sc.outputStride[0], sc.fftDim, sc.outputStride[1]);
							else
								sprintf(output + strlen(output), "		inoutID = (combinedId %% %d) + (combinedId / %d) * %d;\n", sc.fftDim, sc.fftDim, sc.outputStride[1]);
							if (sc.size[sc.axis_id + 1] % sc.localSize[1] != 0)
								sprintf(output + strlen(output), "		if(combinedId / %d + gl_WorkGroupID.y*gl_WorkGroupSize.y< %d){", sc.fftDim, sc.size[sc.axis_id + 1]);

							sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d)){\n", sc.fft_dim_full, sc.fft_zeropad_left_write[sc.axis_id], sc.fft_dim_full, sc.fft_zeropad_right_write[sc.axis_id]);
							if (sc.writeFromRegisters) {
								if (sc.outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[indexOutput(inoutID%s%s)] = %stemp_%d%s;\n", requestCoordinate, requestBatch, convTypeLeft, i, convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[indexOutput(inoutID%s%s) / %d].outputs[indexOutput(inoutID%s%s) %% %d] = %stemp_%d%s;\n", requestCoordinate, requestBatch, sc.outputBufferBlockSize, requestCoordinate, requestBatch, sc.outputBufferBlockSize, convTypeLeft, i, convTypeRight);
							}
							else {
								if (sc.outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[indexOutput(inoutID%s%s)] = %ssdata[(combinedId %% %d) + (combinedId / %d) * sharedStride]%s;\n", requestCoordinate, requestBatch, convTypeLeft, sc.fftDim, sc.fftDim, convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[indexOutput(inoutID%s%s) / %d].outputs[indexOutput(inoutID%s%s) %% %d] = %ssdata[(combinedId %% %d) + (combinedId / %d) * sharedStride]%s;\n", requestCoordinate, requestBatch, sc.outputBufferBlockSize, requestCoordinate, requestBatch, sc.outputBufferBlockSize, convTypeLeft, sc.fftDim, sc.fftDim, convTypeRight);
							}
							sprintf(output + strlen(output), "	}\n");
							if (sc.size[sc.axis_id + 1] % sc.localSize[1] != 0)
								sprintf(output + strlen(output), "		}");

						}
					}
					else {
						for (uint32_t i = 0; i < sc.min_registers_per_thread; i++) {
							if (sc.localSize[1] == 1)
								sprintf(output + strlen(output), "		combinedId = gl_LocalInvocationID.x + %d;\n", i * sc.localSize[0]);
							else
								sprintf(output + strlen(output), "		combinedId = (gl_LocalInvocationID.x + %d * gl_LocalInvocationID.y) + %d;\n", sc.localSize[0], i * sc.localSize[0] * sc.localSize[1]);
							if (sc.localSize[1] == 1)
								sprintf(output + strlen(output), "		inoutID = (gl_WorkGroupID.x%s)/%d+ (combinedId * %d)+ ((gl_WorkGroupID.x%s) %% %d) * %d;\n", shiftX, sc.firstStageStartSize / sc.fftDim, sc.fft_dim_full / sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fft_dim_full / sc.firstStageStartSize);
							else
								sprintf(output + strlen(output), "		inoutID = combinedId %% %d + ((gl_WorkGroupID.x%s) / %d)*%d + ((combinedId/%d) * %d)+ ((gl_WorkGroupID.x%s) %% %d) * %d;\n", sc.localSize[1], shiftX, sc.firstStageStartSize / sc.fftDim, sc.localSize[1], sc.localSize[1], sc.fft_dim_full / sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fft_dim_full / sc.firstStageStartSize);
							sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d)){\n", sc.fft_dim_full, sc.fft_zeropad_left_write[sc.axis_id], sc.fft_dim_full, sc.fft_zeropad_right_write[sc.axis_id]);

							sprintf(output + strlen(output), "		inoutID = indexOutput(gl_LocalInvocationID.x+i*%d+gl_LocalInvocationID.y * %d + (((gl_WorkGroupID.x%s) %% %d) * %d + ((gl_WorkGroupID.x%s) / %d) * %d)%s%s);\n", sc.localSize[0], sc.firstStageStartSize, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.localSize[1] * sc.firstStageStartSize, requestCoordinate, requestBatch);
							if (sc.writeFromRegisters) {
								if (sc.outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID]=%stemp_%d%s;\n", convTypeLeft, i, convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutIDnoShuffle %% %d] = %stemp_%d%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, i, convTypeRight);
							}
							else {
								if (sc.outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID]=%ssdata[sharedStride*gl_LocalInvocationID.y + (gl_LocalInvocationID.x + %d)]%s;\n", convTypeLeft, i * sc.localSize[0], convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutIDnoShuffle %% %d] = %ssdata[sharedStride*gl_LocalInvocationID.y + (gl_LocalInvocationID.x + %d)]%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, i * sc.localSize[0], convTypeRight);
							}
							sprintf(output + strlen(output), "	}\n");
						}
					}
				}
				else {
					if (sc.fftDim == sc.fft_dim_full) {
						for (uint32_t i = 0; i < sc.min_registers_per_thread; i++) {

							if (sc.localSize[1] == 1)
								sprintf(output + strlen(output), "		combinedId = gl_LocalInvocationID.x + %d;\n", i * sc.localSize[0]);
							else
								sprintf(output + strlen(output), "		combinedId = (gl_LocalInvocationID.x + %d * gl_LocalInvocationID.y) + %d;\n", sc.localSize[0], i * sc.localSize[0] * sc.localSize[1]);

							if (sc.outputStride[0] > 1)
								sprintf(output + strlen(output), "		inoutID = indexOutput((combinedId %% %d) * %d + (combinedId / %d) * %d%s%s);\n", sc.fftDim, sc.outputStride[0], sc.fftDim, sc.outputStride[1], requestCoordinate, requestBatch);
							else
								sprintf(output + strlen(output), "		inoutID = indexOutput((combinedId %% %d) + (combinedId / %d) * %d%s%s);\n", sc.fftDim, sc.fftDim, sc.outputStride[1], requestCoordinate, requestBatch);
							if (sc.size[sc.axis_id + 1] % sc.localSize[1] != 0)
								sprintf(output + strlen(output), "		if(combinedId / %d + gl_WorkGroupID.y*gl_WorkGroupSize.y< %d){", sc.fftDim, sc.size[sc.axis_id + 1]);

							if (sc.writeFromRegisters) {
								if (sc.outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %stemp_%d%s;\n", convTypeLeft, i, convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %stemp_%d%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, i, convTypeRight);
							}
							else {
								if (sc.outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %ssdata[(combinedId %% %d) + (combinedId / %d) * sharedStride]%s;\n", convTypeLeft, sc.fftDim, sc.fftDim, convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %ssdata[(combinedId %% %d) + (combinedId / %d) * sharedStride]%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, sc.fftDim, sc.fftDim, convTypeRight);
							}
							if (sc.size[sc.axis_id + 1] % sc.localSize[1] != 0)
								sprintf(output + strlen(output), "		}");

						}

					}
					else {
						for (uint32_t i = 0; i < sc.min_registers_per_thread; i++) {
							sprintf(output + strlen(output), "		inoutID = indexOutput(gl_LocalInvocationID.x+%d+gl_LocalInvocationID.y * %d + (((gl_WorkGroupID.x%s) %% %d) * %d + ((gl_WorkGroupID.x%s) / %d) * %d)%s%s);\n", i * sc.localSize[0], sc.firstStageStartSize, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.localSize[1] * sc.firstStageStartSize, requestCoordinate, requestBatch);
							if (sc.writeFromRegisters) {
								if (sc.outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID]=%stemp_%d%s;\n", convTypeLeft, i, convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %stemp_%d%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, i, convTypeRight);
							}
							else {
								if (sc.outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID]=%ssdata[sharedStride*gl_LocalInvocationID.y + (gl_LocalInvocationID.x + %d)]%s;\n", convTypeLeft, i * sc.localSize[0], convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %ssdata[sharedStride*gl_LocalInvocationID.y + (gl_LocalInvocationID.x + %d)]%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, i * sc.localSize[0], convTypeRight);
							}
						}
					}
				}
			}
			sprintf(output + strlen(output), "	}\n");
			break;
		}
		case 1: {//grouped_c2c
			if (sc.localSize[1] * sc.stageRadix[sc.numStages - 1] * (sc.registers_per_thread / sc.stageRadix[sc.numStages - 1]) > sc.fftDim) {
				sc.writeFromRegisters = 0;
				appendBarrierVkFFT(output, 1);
			}
			else
				sc.writeFromRegisters = 1;
			char shiftX[100] = "";
			if (sc.performWorkGroupShift[0])
				sprintf(shiftX, " + consts.workGroupShiftX * gl_WorkGroupSize.x ");
			sprintf(output + strlen(output), "		if (((gl_GlobalInvocationID.x%s) / %d) %% (%d)+((gl_GlobalInvocationID.x%s) / %d) * (%d) < %d) {;\n", shiftX, sc.fft_dim_x, sc.stageStartSize, shiftX, sc.fft_dim_x* sc.stageStartSize, sc.fftDim* sc.stageStartSize, sc.size[sc.axis_id]);
			if ((sc.reorderFourStep) && (sc.stageStartSize == 1)) {
				if (sc.zeropad[1]) {
					for (uint32_t i = 0; i < sc.min_registers_per_thread; i++) {
						sprintf(output + strlen(output), "		inoutID = (gl_LocalInvocationID.y + %d) * (%d) + (((gl_GlobalInvocationID.x%s) / %d) %% (%d)) * (%d) + ((gl_GlobalInvocationID.x%s) / %d);\n", i * sc.localSize[1], sc.fft_dim_full / sc.fftDim, shiftX, sc.fft_dim_x, sc.firstStageStartSize / sc.fftDim, sc.fft_dim_full / sc.firstStageStartSize, shiftX, sc.fft_dim_x * (sc.firstStageStartSize / sc.fftDim));

						sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d)){\n", sc.fft_dim_full, sc.fft_zeropad_left_write[sc.axis_id], sc.fft_dim_full, sc.fft_zeropad_right_write[sc.axis_id]);
						if (sc.writeFromRegisters) {
							if (sc.outputBufferBlockNum == 1)
								sprintf(output + strlen(output), "			outputBlocks[0].outputs[indexOutput((gl_GlobalInvocationID.x%s) %% (%d), inoutID%s%s)] = %stemp_%d%s;\n", shiftX, sc.fft_dim_x, requestCoordinate, requestBatch, convTypeLeft, i, convTypeRight);
							else
								sprintf(output + strlen(output), "			outputBlocks[indexOutput((gl_GlobalInvocationID.x%s) %% (%d), inoutID%s%s) / %d].outputs[indexOutput((gl_GlobalInvocationID.x%s) %% (%d), inoutID%s%s) %% %d] = %stemp_%d%s;\n", shiftX, sc.fft_dim_x, requestCoordinate, requestBatch, sc.outputBufferBlockSize, shiftX, sc.fft_dim_x, requestCoordinate, requestBatch, sc.outputBufferBlockSize, convTypeLeft, i, convTypeRight);
						}
						else {
							if (sc.outputBufferBlockNum == 1)
								sprintf(output + strlen(output), "			outputBlocks[0].outputs[indexOutput((gl_GlobalInvocationID.x%s) %% (%d), inoutID%s%s)] = %ssdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+%d) + gl_LocalInvocationID.x]%s;\n", shiftX, sc.fft_dim_x, requestCoordinate, requestBatch, convTypeLeft, i * sc.localSize[1], convTypeRight);
							else
								sprintf(output + strlen(output), "			outputBlocks[indexOutput((gl_GlobalInvocationID.x%s) %% (%d), inoutID%s%s) / %d].outputs[indexOutput((gl_GlobalInvocationID.x%s) %% (%d), inoutID%s%s) %% %d] = %ssdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+%d) + gl_LocalInvocationID.x]%s;\n", shiftX, sc.fft_dim_x, requestCoordinate, requestBatch, sc.outputBufferBlockSize, shiftX, sc.fft_dim_x, requestCoordinate, requestBatch, sc.outputBufferBlockSize, convTypeLeft, i * sc.localSize[1], convTypeRight);

						}
						sprintf(output + strlen(output), "	}\n");
					}
				}
				else {
					for (uint32_t i = 0; i < sc.min_registers_per_thread; i++) {
						sprintf(output + strlen(output), "		inoutID = indexOutput((gl_GlobalInvocationID.x%s) %% (%d), (gl_LocalInvocationID.y + %d) * %d + (((gl_GlobalInvocationID.x%s) / %d) %% (%d)) * (%d) + ((gl_GlobalInvocationID.x%s) / %d )%s%s);\n", shiftX, sc.fft_dim_x, i * sc.localSize[1], sc.fft_dim_full / sc.fftDim, shiftX, sc.fft_dim_x, sc.firstStageStartSize / sc.fftDim, sc.fft_dim_full / sc.firstStageStartSize, shiftX, sc.fft_dim_x * (sc.firstStageStartSize / sc.fftDim), requestCoordinate, requestBatch);
						if (sc.writeFromRegisters) {
							if (sc.outputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %stemp_%d%s;\n", convTypeLeft, i, convTypeRight);
							else
								sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %stemp_%d%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, i, convTypeRight);
						}
						else {
							if (sc.outputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %ssdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+%d) + gl_LocalInvocationID.x]%s;\n", convTypeLeft, i * sc.localSize[1], convTypeRight);
							else
								sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %ssdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+%d) + gl_LocalInvocationID.x]%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, i * sc.localSize[1], convTypeRight);
						}
					}
				}
			}
			else {
				if (sc.zeropad[1]) {
					for (uint32_t i = 0; i < sc.min_registers_per_thread; i++) {
						sprintf(output + strlen(output), "		inoutID = (gl_LocalInvocationID.y + %d) * %d + ((gl_GlobalInvocationID.x%s) / %d) %% (%d)+((gl_GlobalInvocationID.x%s) / %d) * (%d);\n", i * sc.localSize[1], sc.stageStartSize, shiftX, sc.fft_dim_x, sc.stageStartSize, shiftX, sc.fft_dim_x * sc.stageStartSize, sc.stageStartSize * sc.fftDim);

						sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d)){\n", sc.fft_dim_full, sc.fft_zeropad_left_write[sc.axis_id], sc.fft_dim_full, sc.fft_zeropad_right_write[sc.axis_id]);
						sprintf(output + strlen(output), "		inoutID = indexOutput((gl_GlobalInvocationID.x%s) %% (%d), %d * (gl_LocalInvocationID.y + %d) + ((gl_GlobalInvocationID.x%s) / %d) %% (%d)+((gl_GlobalInvocationID.x%s) / %d) * (%d)%s%s);\n", shiftX, sc.fft_dim_x, sc.stageStartSize, i * sc.localSize[1], shiftX, sc.fft_dim_x, sc.stageStartSize, shiftX, sc.fft_dim_x * sc.stageStartSize, sc.stageStartSize * sc.fftDim, requestCoordinate, requestBatch);
						if (sc.writeFromRegisters) {
							if (sc.outputBufferBlockNum == 1)
								sprintf(output + strlen(output), "			outputBlocks[0].outputs[inoutID] = %stemp_%d%s;\n", convTypeLeft, i, convTypeRight);
							else
								sprintf(output + strlen(output), "			outputBlocks[inoutID / %d].outputs[inoutID %% %d] =  %stemp_%d%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, i, convTypeRight);
						}
						else {
							if (sc.outputBufferBlockNum == 1)
								sprintf(output + strlen(output), "			outputBlocks[0].outputs[inoutID] = %ssdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+%d) + gl_LocalInvocationID.x]%s;\n", convTypeLeft, i * sc.localSize[1], convTypeRight);
							else
								sprintf(output + strlen(output), "			outputBlocks[inoutID / %d].outputs[inoutID %% %d] =  %ssdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+%d) + gl_LocalInvocationID.x]%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, i * sc.localSize[1], convTypeRight);
						}
						sprintf(output + strlen(output), "	}\n");
					}
				}
				else {
					for (uint32_t i = 0; i < sc.min_registers_per_thread; i++) {
						sprintf(output + strlen(output), "		inoutID = indexOutput((gl_GlobalInvocationID.x%s) %% (%d), %d * (gl_LocalInvocationID.y + %d) + ((gl_GlobalInvocationID.x%s) / %d) %% (%d)+((gl_GlobalInvocationID.x%s) / %d) * (%d)%s%s);\n", shiftX, sc.fft_dim_x, sc.stageStartSize, i * sc.localSize[1], shiftX, sc.fft_dim_x, sc.stageStartSize, shiftX, sc.fft_dim_x * sc.stageStartSize, sc.stageStartSize * sc.fftDim, requestCoordinate, requestBatch);
						if (sc.writeFromRegisters) {
							if (sc.outputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %stemp_%d%s;\n", convTypeLeft, i, convTypeRight);
							else
								sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] =  %stemp_%d%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, i, convTypeRight);
						}
						else {
							if (sc.outputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %ssdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+%d) + gl_LocalInvocationID.x]%s;\n", convTypeLeft, i * sc.localSize[1], convTypeRight);
							else
								sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] =  %ssdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+%d) + gl_LocalInvocationID.x]%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, i * sc.localSize[1], convTypeRight);
						}
					}
				}
			}
			sprintf(output + strlen(output), "	}\n");
			break;

		}
		case 2: {//single_c2c_strided
			if (sc.localSize[1] * sc.stageRadix[sc.numStages - 1] * (sc.registers_per_thread / sc.stageRadix[sc.numStages - 1]) > sc.fftDim) {
				sc.writeFromRegisters = 0;
				appendBarrierVkFFT(output, 1);
			}
			else
				sc.writeFromRegisters = 1;
			char shiftX[100] = "";
			if (sc.performWorkGroupShift[0])
				sprintf(shiftX, " + consts.workGroupShiftX * gl_WorkGroupSize.x ");
			sprintf(output + strlen(output), "		if (((gl_GlobalInvocationID.x%s) / %d) * (%d) < %d) {;\n", shiftX, sc.stageStartSize, sc.stageStartSize* sc.fftDim, sc.fft_dim_full);
			if (sc.zeropad[1]) {
				for (uint32_t i = 0; i < sc.min_registers_per_thread; i++) {
					sprintf(output + strlen(output), "		inoutID = (gl_GlobalInvocationID.x%s) %% (%d) + %d * (gl_LocalInvocationID.y + %d) + ((gl_GlobalInvocationID.x%s) / %d) * (%d);\n", shiftX, sc.stageStartSize, sc.stageStartSize, i * sc.localSize[1], shiftX, sc.stageStartSize, sc.stageStartSize * sc.fftDim);
					sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d)){\n", sc.fft_dim_full, sc.fft_zeropad_left_write[sc.axis_id], sc.fft_dim_full, sc.fft_zeropad_right_write[sc.axis_id]);
					if (sc.writeFromRegisters) {
						if (sc.outputBufferBlockNum == 1)
							sprintf(output + strlen(output), "			outputBlocks[0].outputs[inoutID] = %stemp_%d%s;\n", convTypeLeft, i, convTypeRight);
						else
							sprintf(output + strlen(output), "			outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %stemp_%d%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, i, convTypeRight);
					}
					else {
						if (sc.outputBufferBlockNum == 1)
							sprintf(output + strlen(output), "			outputBlocks[0].outputs[inoutID] = %ssdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+%d) + gl_LocalInvocationID.x]%s;\n", convTypeLeft, i * sc.localSize[1], convTypeRight);
						else
							sprintf(output + strlen(output), "			outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %ssdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+%d) + gl_LocalInvocationID.x]%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, i * sc.localSize[1], convTypeRight);
					}
					sprintf(output + strlen(output), "	}\n");
				}
			}
			else {
				for (uint32_t i = 0; i < sc.min_registers_per_thread; i++) {
					sprintf(output + strlen(output), "		inoutID = indexOutput((gl_GlobalInvocationID.x%s) %% (%d) + %d * (gl_LocalInvocationID.y + %d) + ((gl_GlobalInvocationID.x%s) / %d) * (%d));\n", shiftX, sc.stageStartSize, sc.stageStartSize, i * sc.localSize[1], shiftX, sc.stageStartSize, sc.stageStartSize * sc.fftDim);
					if (sc.writeFromRegisters) {
						if (sc.outputBufferBlockNum == 1)
							sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %stemp_%d%s;\n", convTypeLeft, i, convTypeRight);
						else
							sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %stemp_%d%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, i, convTypeRight);
					}
					else {
						if (sc.outputBufferBlockNum == 1)
							sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %ssdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+%d) + gl_LocalInvocationID.x]%s;\n", convTypeLeft, i * sc.localSize[1], convTypeRight);
						else
							sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %ssdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+%d) + gl_LocalInvocationID.x]%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, i * sc.localSize[1], convTypeRight);
					}
				}
			}
			sprintf(output + strlen(output), "	}\n");
			break;

		}
		case 3:
		{//single_c2c - registerBoost - 2x
			char shiftX[100] = "";
			if (sc.performWorkGroupShift[0])
				sprintf(shiftX, " + consts.workGroupShiftX ");
			if (sc.reorderFourStep) {
				if (sc.zeropad[1]) {
					for (uint32_t j = 0; j < 2; j++) {
						for (uint32_t i = 0; i < sc.registers_per_thread; i++) {
							if (sc.localSize[1] == 1)
								sprintf(output + strlen(output), "		combinedId = gl_LocalInvocationID.x + %d;\n", (i + j * sc.registers_per_thread) * sc.localSize[0]);
							else
								sprintf(output + strlen(output), "		combinedId = (gl_LocalInvocationID.x + %d * gl_LocalInvocationID.y) + %d;\n", sc.localSize[0], (i + j * sc.registers_per_thread) * sc.localSize[0] * sc.localSize[1]);
							if (sc.localSize[1] == 1)
								sprintf(output + strlen(output), "		inoutID = (gl_WorkGroupID.x%s)/%d+ (combinedId * %d)+ ((gl_WorkGroupID.x%s) %% %d) * %d;\n", shiftX, sc.firstStageStartSize / sc.fftDim, sc.fft_dim_full / sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fft_dim_full / sc.firstStageStartSize);
							else
								sprintf(output + strlen(output), "		inoutID = combinedId %% %d + ((gl_WorkGroupID.x%s) / %d)*%d + ((combinedId/%d) * %d)+ ((gl_WorkGroupID.x%s) %% %d) * %d;\n", sc.localSize[1], shiftX, sc.firstStageStartSize / sc.fftDim, sc.localSize[1], sc.localSize[1], sc.fft_dim_full / sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fft_dim_full / sc.firstStageStartSize);
							sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d))\n", sc.fft_dim_full, sc.fft_zeropad_left_write[sc.axis_id], sc.fft_dim_full, sc.fft_zeropad_right_write[sc.axis_id]);

							if (sc.outputBufferBlockNum == 1)
								sprintf(output + strlen(output), "			outputBlocks[0].outputs[indexOutput(inoutID)] = %stemp_%d%s;\n", convTypeLeft, (i + j * sc.registers_per_thread), convTypeRight);
							else
								sprintf(output + strlen(output), "			outputBlocks[indexOutput(inoutID) / %d].outputs[indexOutput(inoutID) %% %d] = %stemp_%d%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, (i + j * sc.registers_per_thread), convTypeRight);

						}
					}
				}
				else {
					if (sc.fftDim == sc.fft_dim_full) {
						for (uint32_t j = 0; j < 2; j++) {
							for (uint32_t i = 0; i < sc.registers_per_thread; i++) {

								if (sc.localSize[1] == 1)
									sprintf(output + strlen(output), "		combinedId = gl_LocalInvocationID.x + %d;\n", (i + j * sc.registers_per_thread) * sc.localSize[0]);
								else
									sprintf(output + strlen(output), "		combinedId = (gl_LocalInvocationID.x + %d * gl_LocalInvocationID.y) + %d;\n", sc.localSize[0], (i + j * sc.registers_per_thread) * sc.localSize[0] * sc.localSize[1]);

								if (sc.outputStride[0] > 1)
									sprintf(output + strlen(output), "		inoutID = indexOutput((combinedId %% %d) * %d + (combinedId / %d) * %d);\n", sc.fftDim, sc.outputStride[0], sc.fftDim, sc.outputStride[1]);
								else
									sprintf(output + strlen(output), "		inoutID = indexOutput((combinedId %% %d) + (combinedId / %d) * %d);\n", sc.fftDim, sc.fftDim, sc.outputStride[1]);

								if (sc.outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %stemp_%d%s;\n", convTypeLeft, (i + j * sc.registers_per_thread), convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %stemp_%d%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, (i + j * sc.registers_per_thread), convTypeRight);

							}
						}
					}
					else {
						for (uint32_t j = 0; j < 2; j++) {
							for (uint32_t i = 0; i < sc.registers_per_thread; i++) {
								if (sc.localSize[1] == 1)
									sprintf(output + strlen(output), "		combinedId = gl_LocalInvocationID.x + %d;\n", (i + j * sc.registers_per_thread) * sc.localSize[0]);
								else
									sprintf(output + strlen(output), "		combinedId = (gl_LocalInvocationID.x + %d * gl_LocalInvocationID.y) + %d;\n", sc.localSize[0], (i + j * sc.registers_per_thread) * sc.localSize[0] * sc.localSize[1]);
								if (sc.localSize[1] == 1)
									sprintf(output + strlen(output), "		inoutID = indexOutput((gl_WorkGroupID.x%s)/%d+ (combinedId * %d)+ ((gl_WorkGroupID.x%s) %% %d) * %d);\n", shiftX, sc.firstStageStartSize / sc.fftDim, sc.fft_dim_full / sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fft_dim_full / sc.firstStageStartSize);
								else
									sprintf(output + strlen(output), "		inoutID = indexOutput(combinedId %% %d + ((gl_WorkGroupID.x%s) / %d)*%d + ((combinedId/%d) * %d)+ ((gl_WorkGroupID.x%s) %% %d) * %d);\n", sc.localSize[1], shiftX, sc.firstStageStartSize / sc.fftDim, sc.localSize[1], sc.localSize[1], sc.fft_dim_full / sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fft_dim_full / sc.firstStageStartSize);
								if (sc.outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %stemp_%d%s;\n", convTypeLeft, (i + j * sc.registers_per_thread), convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %stemp_%d%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, (i + j * sc.registers_per_thread), convTypeRight);

							}
						}
					}
				}
			}
			else {
				if (sc.zeropad[1]) {
					for (uint32_t j = 0; j < 2; j++) {
						for (uint32_t i = 0; i < sc.registers_per_thread; i++) {
							if (sc.localSize[1] == 1)
								sprintf(output + strlen(output), "		combinedId = gl_LocalInvocationID.x + %d;\n", (i + j * sc.registers_per_thread) * sc.localSize[0]);
							else
								sprintf(output + strlen(output), "		combinedId = (gl_LocalInvocationID.x + %d * gl_LocalInvocationID.y) + %d;\n", sc.localSize[0], (i + j * sc.registers_per_thread) * sc.localSize[0] * sc.localSize[1]);
							if (sc.localSize[1] == 1)
								sprintf(output + strlen(output), "		inoutID = (gl_WorkGroupID.x%s)/%d+ (combinedId * %d)+ ((gl_WorkGroupID.x%s) %% %d) * %d;\n", shiftX, sc.firstStageStartSize / sc.fftDim, sc.fft_dim_full / sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fft_dim_full / sc.firstStageStartSize);
							else
								sprintf(output + strlen(output), "		inoutID = combinedId %% %d + ((gl_WorkGroupID.x%s) / %d)*%d + ((combinedId/%d) * %d)+ ((gl_WorkGroupID.x%s) %% %d) * %d;\n", sc.localSize[1], shiftX, sc.firstStageStartSize / sc.fftDim, sc.localSize[1], sc.localSize[1], sc.fft_dim_full / sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fft_dim_full / sc.firstStageStartSize);
							sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d)){\n", sc.fft_dim_full, sc.fft_zeropad_left_write[sc.axis_id], sc.fft_dim_full, sc.fft_zeropad_right_write[sc.axis_id]);

							sprintf(output + strlen(output), "		inoutID = indexOutput(gl_LocalInvocationID.x+%d+gl_LocalInvocationID.y * %d + (((gl_WorkGroupID.x%s) %% %d) * %d + ((gl_WorkGroupID.x%s) / %d) * %d));\n", (i + j * sc.registers_per_thread) * sc.localSize[0], sc.firstStageStartSize, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.localSize[1] * sc.firstStageStartSize);

							if (sc.outputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %stemp_%d%s;\n", convTypeLeft, (i + j * sc.registers_per_thread), convTypeRight);
							else
								sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %stemp_%d%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, (i + j * sc.registers_per_thread), convTypeRight);
							sprintf(output + strlen(output), "	}\n");
						}
					}
				}
				else {
					if (sc.fftDim == sc.fft_dim_full) {
						for (uint32_t j = 0; j < 2; j++) {
							for (uint32_t i = 0; i < sc.registers_per_thread; i++) {

								if (sc.localSize[1] == 1)
									sprintf(output + strlen(output), "		combinedId = gl_LocalInvocationID.x + %d;\n", (i + j * sc.registers_per_thread) * sc.localSize[0]);
								else
									sprintf(output + strlen(output), "		combinedId = (gl_LocalInvocationID.x + %d * gl_LocalInvocationID.y) + %d;\n", sc.localSize[0], (i + j * sc.registers_per_thread) * sc.localSize[0] * sc.localSize[1]);

								if (sc.outputStride[0] > 1)
									sprintf(output + strlen(output), "		inoutID = indexOutput((combinedId %% %d) * %d + (combinedId / %d) * %d);\n", sc.fftDim, sc.outputStride[0], sc.fftDim, sc.outputStride[1]);
								else
									sprintf(output + strlen(output), "		inoutID = indexOutput((combinedId %% %d) + (combinedId / %d) * %d);\n", sc.fftDim, sc.fftDim, sc.outputStride[1]);

								if (sc.outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %stemp_%d%s;\n", convTypeLeft, (i + j * sc.registers_per_thread), convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %stemp_%d%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, (i + j * sc.registers_per_thread), convTypeRight);

							}
						}
					}
					else {
						for (uint32_t j = 0; j < 2; j++) {
							for (uint32_t i = 0; i < sc.registers_per_thread; i++) {
								sprintf(output + strlen(output), "		inoutID = indexOutput(gl_LocalInvocationID.x+%d+gl_LocalInvocationID.y * %d + (((gl_WorkGroupID.x%s) %% %d) * %d + ((gl_WorkGroupID.x%s) / %d) * %d));\n", (i + j * sc.registers_per_thread) * sc.localSize[0], sc.firstStageStartSize, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.localSize[1] * sc.firstStageStartSize);

								if (sc.outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %stemp_%d%s;\n", convTypeLeft, (i + j * sc.registers_per_thread), convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %stemp_%d%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, (i + j * sc.registers_per_thread), convTypeRight);
							}
						}
					}
				}
			}
			break;
		}
		case 4:
		{//single_c2c - registerBoost - 4x
			char shiftX[100] = "";
			if (sc.performWorkGroupShift[0])
				sprintf(shiftX, " + consts.workGroupShiftX ");
			if (sc.reorderFourStep) {
				if (sc.zeropad[1]) {
					for (uint32_t j = 0; j < 4; j++) {
						for (uint32_t i = 0; i < sc.registers_per_thread; i++) {
							if (sc.localSize[1] == 1)
								sprintf(output + strlen(output), "		combinedId = gl_LocalInvocationID.x + %d;\n", (i + j * sc.registers_per_thread) * sc.localSize[0]);
							else
								sprintf(output + strlen(output), "		combinedId = (gl_LocalInvocationID.x + %d * gl_LocalInvocationID.y) + %d;\n", sc.localSize[0], (i + j * sc.registers_per_thread) * sc.localSize[0] * sc.localSize[1]);
							if (sc.localSize[1] == 1)
								sprintf(output + strlen(output), "		inoutID = (gl_WorkGroupID.x%s)/%d+ (combinedId * %d)+ ((gl_WorkGroupID.x%s) %% %d) * %d;\n", shiftX, sc.firstStageStartSize / sc.fftDim, sc.fft_dim_full / sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fft_dim_full / sc.firstStageStartSize);
							else
								sprintf(output + strlen(output), "		inoutID = combinedId %% %d + ((gl_WorkGroupID.x%s) / %d)*%d + ((combinedId/%d) * %d)+ ((gl_WorkGroupID.x%s) %% %d) * %d;\n", sc.localSize[1], shiftX, sc.firstStageStartSize / sc.fftDim, sc.localSize[1], sc.localSize[1], sc.fft_dim_full / sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fft_dim_full / sc.firstStageStartSize);
							sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d))\n", sc.fft_dim_full, sc.fft_zeropad_left_write[sc.axis_id], sc.fft_dim_full, sc.fft_zeropad_right_write[sc.axis_id]);

							if (sc.outputBufferBlockNum == 1)
								sprintf(output + strlen(output), "			outputBlocks[0].outputs[indexOutput(inoutID)] = %stemp_%d%s;\n", convTypeLeft, (i + j * sc.registers_per_thread), convTypeRight);
							else
								sprintf(output + strlen(output), "			outputBlocks[indexOutput(inoutID) / %d].outputs[indexOutput(inoutID) %% %d] = %stemp_%d%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, (i + j * sc.registers_per_thread), convTypeRight);

						}
					}
				}
				else {
					if (sc.fftDim == sc.fft_dim_full) {
						for (uint32_t j = 0; j < 4; j++) {
							for (uint32_t i = 0; i < sc.registers_per_thread; i++) {

								if (sc.localSize[1] == 1)
									sprintf(output + strlen(output), "		combinedId = gl_LocalInvocationID.x + %d;\n", (i + j * sc.registers_per_thread) * sc.localSize[0]);
								else
									sprintf(output + strlen(output), "		combinedId = (gl_LocalInvocationID.x + %d * gl_LocalInvocationID.y) + %d;\n", sc.localSize[0], (i + j * sc.registers_per_thread) * sc.localSize[0] * sc.localSize[1]);

								if (sc.outputStride[0] > 1)
									sprintf(output + strlen(output), "		inoutID = indexOutput((combinedId %% %d) * %d + (combinedId / %d) * %d);\n", sc.fftDim, sc.outputStride[0], sc.fftDim, sc.outputStride[1]);
								else
									sprintf(output + strlen(output), "		inoutID = indexOutput((combinedId %% %d) + (combinedId / %d) * %d);\n", sc.fftDim, sc.fftDim, sc.outputStride[1]);

								if (sc.outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %stemp_%d%s;\n", convTypeLeft, (i + j * sc.registers_per_thread), convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %stemp_%d%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, (i + j * sc.registers_per_thread), convTypeRight);

							}
						}
					}
					else {
						for (uint32_t j = 0; j < 4; j++) {
							for (uint32_t i = 0; i < sc.registers_per_thread; i++) {
								if (sc.localSize[1] == 1)
									sprintf(output + strlen(output), "		combinedId = gl_LocalInvocationID.x + %d;\n", (i + j * sc.registers_per_thread) * sc.localSize[0]);
								else
									sprintf(output + strlen(output), "		combinedId = (gl_LocalInvocationID.x + %d * gl_LocalInvocationID.y) + %d;\n", sc.localSize[0], (i + j * sc.registers_per_thread) * sc.localSize[0] * sc.localSize[1]);
								if (sc.localSize[1] == 1)
									sprintf(output + strlen(output), "		inoutID = indexOutput((gl_WorkGroupID.x%s)/%d+ (combinedId * %d)+ ((gl_WorkGroupID.x%s) %% %d) * %d);\n", shiftX, sc.firstStageStartSize / sc.fftDim, sc.fft_dim_full / sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fft_dim_full / sc.firstStageStartSize);
								else
									sprintf(output + strlen(output), "		inoutID = indexOutput(combinedId %% %d + ((gl_WorkGroupID.x%s) / %d)*%d + ((combinedId/%d) * %d)+ ((gl_WorkGroupID.x%s) %% %d) * %d);\n", sc.localSize[1], shiftX, sc.firstStageStartSize / sc.fftDim, sc.localSize[1], sc.localSize[1], sc.fft_dim_full / sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fft_dim_full / sc.firstStageStartSize);
								if (sc.outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %stemp_%d%s;\n", convTypeLeft, (i + j * sc.registers_per_thread), convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %stemp_%d%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, (i + j * sc.registers_per_thread), convTypeRight);

							}
						}
					}
				}
			}
			else {
				if (sc.zeropad[1]) {
					for (uint32_t j = 0; j < 4; j++) {
						for (uint32_t i = 0; i < sc.registers_per_thread; i++) {
							if (sc.localSize[1] == 1)
								sprintf(output + strlen(output), "		combinedId = gl_LocalInvocationID.x + %d;\n", (i + j * sc.registers_per_thread) * sc.localSize[0]);
							else
								sprintf(output + strlen(output), "		combinedId = (gl_LocalInvocationID.x + %d * gl_LocalInvocationID.y) + %d;\n", sc.localSize[0], (i + j * sc.registers_per_thread) * sc.localSize[0] * sc.localSize[1]);
							if (sc.localSize[1] == 1)
								sprintf(output + strlen(output), "		inoutID = (gl_WorkGroupID.x%s)/%d+ (combinedId * %d)+ ((gl_WorkGroupID.x%s) %% %d) * %d;\n", shiftX, sc.firstStageStartSize / sc.fftDim, sc.fft_dim_full / sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fft_dim_full / sc.firstStageStartSize);
							else
								sprintf(output + strlen(output), "		inoutID = combinedId %% %d + ((gl_WorkGroupID.x%s) / %d)*%d + ((combinedId/%d) * %d)+ ((gl_WorkGroupID.x%s) %% %d) * %d;\n", sc.localSize[1], shiftX, sc.firstStageStartSize / sc.fftDim, sc.localSize[1], sc.localSize[1], sc.fft_dim_full / sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fft_dim_full / sc.firstStageStartSize);
							sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d)){\n", sc.fft_dim_full, sc.fft_zeropad_left_write[sc.axis_id], sc.fft_dim_full, sc.fft_zeropad_right_write[sc.axis_id]);

							sprintf(output + strlen(output), "		inoutID = indexOutput(gl_LocalInvocationID.x+%d+gl_LocalInvocationID.y * %d + (((gl_WorkGroupID.x%s) %% %d) * %d + ((gl_WorkGroupID.x%s) / %d) * %d));\n", (i + j * sc.registers_per_thread) * sc.localSize[0], sc.firstStageStartSize, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.localSize[1] * sc.firstStageStartSize);

							if (sc.outputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %stemp_%d%s;\n", convTypeLeft, (i + j * sc.registers_per_thread), convTypeRight);
							else
								sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %stemp_%d%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, (i + j * sc.registers_per_thread), convTypeRight);
							sprintf(output + strlen(output), "	}\n");
						}
					}
				}
				else {
					if (sc.fftDim == sc.fft_dim_full) {
						for (uint32_t j = 0; j < 4; j++) {
							for (uint32_t i = 0; i < sc.registers_per_thread; i++) {

								if (sc.localSize[1] == 1)
									sprintf(output + strlen(output), "		combinedId = gl_LocalInvocationID.x + %d;\n", (i + j * sc.registers_per_thread) * sc.localSize[0]);
								else
									sprintf(output + strlen(output), "		combinedId = (gl_LocalInvocationID.x + %d * gl_LocalInvocationID.y) + %d;\n", sc.localSize[0], (i + j * sc.registers_per_thread) * sc.localSize[0] * sc.localSize[1]);

								if (sc.outputStride[0] > 1)
									sprintf(output + strlen(output), "		inoutID = indexOutput((combinedId %% %d) * %d + (combinedId / %d) * %d);\n", sc.fftDim, sc.outputStride[0], sc.fftDim, sc.outputStride[1]);
								else
									sprintf(output + strlen(output), "		inoutID = indexOutput((combinedId %% %d) + (combinedId / %d) * %d);\n", sc.fftDim, sc.fftDim, sc.outputStride[1]);

								if (sc.outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %stemp_%d%s;\n", convTypeLeft, (i + j * sc.registers_per_thread), convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %stemp_%d%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, (i + j * sc.registers_per_thread), convTypeRight);

							}
						}
					}
					else {
						for (uint32_t j = 0; j < 4; j++) {
							for (uint32_t i = 0; i < sc.registers_per_thread; i++) {
								sprintf(output + strlen(output), "		inoutID = indexOutput(gl_LocalInvocationID.x+%d+gl_LocalInvocationID.y * %d + (((gl_WorkGroupID.x%s) %% %d) * %d + ((gl_WorkGroupID.x%s) / %d) * %d));\n", (i + j * sc.registers_per_thread) * sc.localSize[0], sc.firstStageStartSize, shiftX, sc.firstStageStartSize / sc.fftDim, sc.fftDim, shiftX, sc.firstStageStartSize / sc.fftDim, sc.localSize[1] * sc.firstStageStartSize);

								if (sc.outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %stemp_%d%s;\n", convTypeLeft, (i + j * sc.registers_per_thread), convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %stemp_%d%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, (i + j * sc.registers_per_thread), convTypeRight);
							}
						}
					}
				}
			}
			break;
		}
		case 5:
		{//single_r2c
			char shiftY[100] = "";
			if (sc.performWorkGroupShift[1])
				sprintf(shiftY, " + consts.workGroupShiftY * %d", sc.localSize[1]);
			char shiftY2[100] = "";
			if (sc.performWorkGroupShift[1])
				sprintf(shiftY, " + consts.workGroupShiftY ");

			if (sc.reorderFourStep) {
				//Not implemented
			}
			else {
				appendBarrierVkFFT(output, 1);
				if (sc.fftDim == sc.fft_dim_full) {
					if ((uint32_t)ceil(sc.size[1] / 2.0) % sc.localSize[1] != 0)
						sprintf(output + strlen(output), "		if(gl_GlobalInvocationID.y%s < %d){", shiftY, (uint32_t)ceil(sc.size[1] / 2.0));
					sprintf(output + strlen(output), "\
	if (gl_LocalInvocationID.x==0)\n\
	{\n\
		temp_0.x = sdata[sharedStride * gl_LocalInvocationID.y].x;\n\
		temp_0.y = 0;\n\
		temp_1.x = sdata[sharedStride * gl_LocalInvocationID.y].y;\n\
		temp_1.y = 0;\n");

					sprintf(output + strlen(output), "		inoutID = indexOutput(2 * (gl_GlobalInvocationID.y%s), %d);\n", shiftY, sc.outputStride[2] / (sc.outputStride[1] + 2));

					if (sc.outputBufferBlockNum == 1)
						sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %stemp_0%s;\n", convTypeLeft, convTypeRight);
					else
						sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %stemp_0%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, convTypeRight);

					sprintf(output + strlen(output), "		inoutID = indexOutput(2 * (gl_GlobalInvocationID.y%s) + 1, %d);\n", shiftY, sc.outputStride[2] / (sc.outputStride[1] + 2));
					if (sc.outputBufferBlockNum == 1)
						sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %stemp_1%s;\n", convTypeLeft, convTypeRight);
					else
						sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %stemp_1%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, convTypeRight);

					sprintf(output + strlen(output), "	}\n");

					for (uint32_t i = 0; i < ceil(sc.min_registers_per_thread / 2.0); i++) {
						if (sc.localSize[1] == 1)
							sprintf(output + strlen(output), "\
		temp_0.x = 0.5 * (sdata[%d + gl_LocalInvocationID.x].x + sdata[%d - gl_LocalInvocationID.x].x);\n\
		temp_0.y = 0.5 * (sdata[%d + gl_LocalInvocationID.x].y - sdata[%d - gl_LocalInvocationID.x].y);\n\
		temp_1.x = 0.5 * (sdata[%d + gl_LocalInvocationID.x].y + sdata[%d - gl_LocalInvocationID.x].y);\n\
		temp_1.y = 0.5 * (-sdata[%d + gl_LocalInvocationID.x].x + sdata[%d - gl_LocalInvocationID.x].x);\n", 1 + i * sc.localSize[0], sc.fftDim - 1 - i * sc.localSize[0], 1 + i * sc.localSize[0], sc.fftDim - 1 - i * sc.localSize[0], 1 + i * sc.localSize[0], sc.fftDim - 1 - i * sc.localSize[0], 1 + i * sc.localSize[0], sc.fftDim - 1 - i * sc.localSize[0]);
						else
							sprintf(output + strlen(output), "\
		temp_0.x = 0.5 * (sdata[sharedStride * gl_LocalInvocationID.y + (%d + gl_LocalInvocationID.x)].x + sdata[sharedStride * gl_LocalInvocationID.y + (%d - gl_LocalInvocationID.x)].x);\n\
		temp_0.y = 0.5 * (sdata[sharedStride * gl_LocalInvocationID.y + (%d + gl_LocalInvocationID.x)].y - sdata[sharedStride * gl_LocalInvocationID.y + (%d - gl_LocalInvocationID.x)].y);\n\
		temp_1.x = 0.5 * (sdata[sharedStride * gl_LocalInvocationID.y + (%d + gl_LocalInvocationID.x)].y + sdata[sharedStride * gl_LocalInvocationID.y + (%d - gl_LocalInvocationID.x)].y);\n\
		temp_1.y = 0.5 * (-sdata[sharedStride * gl_LocalInvocationID.y + (%d + gl_LocalInvocationID.x)].x + sdata[sharedStride * gl_LocalInvocationID.y + (%d - gl_LocalInvocationID.x)].x);\n", 1 + i * sc.localSize[0], sc.fftDim - 1 - i * sc.localSize[0], 1 + i * sc.localSize[0], sc.fftDim - 1 - i * sc.localSize[0], 1 + i * sc.localSize[0], sc.fftDim - 1 - i * sc.localSize[0], 1 + i * sc.localSize[0], sc.fftDim - 1 - i * sc.localSize[0]);
						if (sc.zeropad[1]) {

							sprintf(output + strlen(output), "		inoutID = gl_LocalInvocationID.x+%d;\n", i * sc.localSize[0]);
							sprintf(output + strlen(output), "		if((inoutID < %d)||(inoutID >= %d)){\n", sc.fft_zeropad_left_write[sc.axis_id], sc.fft_zeropad_right_write[sc.axis_id]);

							if (sc.outputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		outputBlocks[0].outputs[indexOutput(inoutID, (gl_GlobalInvocationID.y%s))] = %stemp_0%s;\n", convTypeLeft, shiftY, convTypeRight);
							else
								sprintf(output + strlen(output), "		outputBlocks[indexOutput(inoutID, (gl_GlobalInvocationID.y%s)) / %d].outputs[indexOutput(inoutID, (gl_GlobalInvocationID.y%s)) %% %d] = %stemp_0%s;\n", shiftY, sc.outputBufferBlockSize, shiftY, sc.outputBufferBlockSize, convTypeLeft, convTypeRight);

							sprintf(output + strlen(output), "		inoutID = indexOutput(gl_LocalInvocationID.x+%d, (gl_GlobalInvocationID.y%s));\n", sc.outputStride[1] / 2 + i * sc.localSize[0], shiftY);

							if (sc.outputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %stemp_1%s;\n", convTypeLeft, convTypeRight);
							else
								sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %stemp_1%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, convTypeRight);

							sprintf(output + strlen(output), "	}\n");
						}
						else {
							sprintf(output + strlen(output), "		inoutID = indexOutput(gl_LocalInvocationID.x+%d, (gl_GlobalInvocationID.y%s));\n", i * sc.localSize[0], shiftY);

							if (sc.outputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %stemp_0%s;\n", convTypeLeft, convTypeRight);
							else
								sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %stemp_0%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, convTypeRight);

							sprintf(output + strlen(output), "		inoutID = indexOutput(gl_LocalInvocationID.x+%d, (gl_GlobalInvocationID.y%s));\n", sc.outputStride[1] / 2 + i * sc.localSize[0], shiftY);

							if (sc.outputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %stemp_1%s;\n", convTypeLeft, convTypeRight);
							else
								sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %stemp_1%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, convTypeRight);

						}
					}
					if ((uint32_t)ceil(sc.size[1] / 2.0) % sc.localSize[1] != 0)
						sprintf(output + strlen(output), "		}");
				}
				else {
					//Not implemented
				}
			}
			break;
		}
		case 6: {//single_c2r
			char shiftY[100] = "";
			if (sc.performWorkGroupShift[1])
				sprintf(shiftY, " + consts.workGroupShiftY * %d", sc.localSize[1]);
			if ((sc.localSize[1] > 1) || (sc.localSize[0] * sc.stageRadix[sc.numStages - 1] * (sc.registers_per_thread / sc.stageRadix[sc.numStages - 1]) > sc.fftDim)) {
				sc.writeFromRegisters = 0;
				appendBarrierVkFFT(output, 1);
			}
			else
				sc.writeFromRegisters = 1;
			if (sc.reorderFourStep) {
				//Not implemented
			}
			else {
				if (sc.zeropad[1]) {
					if (sc.fftDim == sc.fft_dim_full) {
						for (uint32_t i = 0; i < sc.min_registers_per_thread; i++) {

							if (sc.localSize[1] == 1)
								sprintf(output + strlen(output), "		combinedId = gl_LocalInvocationID.x + %d;\n", i * sc.localSize[0]);
							else
								sprintf(output + strlen(output), "		combinedId = (gl_LocalInvocationID.x + %d * gl_LocalInvocationID.y) + %d;\n", sc.localSize[0], i * sc.localSize[0] * sc.localSize[1]);

							if (sc.outputStride[0] > 1)
								sprintf(output + strlen(output), "		inoutID = (combinedId %% %d) * %d + (combinedId / %d) * %d;\n", sc.fftDim, sc.outputStride[0], sc.fftDim, 2 * sc.outputStride[1]);
							else
								sprintf(output + strlen(output), "		inoutID = (combinedId %% %d) + (combinedId / %d) * %d;\n", sc.fftDim, sc.fftDim, 2 * sc.outputStride[1]);
							if ((uint32_t)ceil(sc.size[1] / 2.0) % sc.localSize[1] != 0)
								sprintf(output + strlen(output), "		if(combinedId / %d + (gl_WorkGroupID.y%s)*gl_WorkGroupSize.y< %d){", sc.fftDim, shiftY, (uint32_t)ceil(sc.size[1] / 2.0));

							sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d)){\n", sc.fft_dim_full, sc.fft_zeropad_left_write[sc.axis_id], sc.fft_dim_full, sc.fft_zeropad_right_write[sc.axis_id]);
							if (sc.writeFromRegisters) {
								if (sc.outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[indexOutput(inoutID)] = %stemp_%d.x%s;\n", convTypeLeft, i, convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[indexOutput(inoutID) / %d].outputs[indexOutput(inoutID) %% %d] = %stemp_%d.x%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, i, convTypeRight);
								if (sc.outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[indexOutput(inoutID + %d)] = %stemp_%d.y%s;\n", sc.outputStride[1], convTypeLeft, i, convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[indexOutput(inoutID + %d) / %d].outputs[indexOutput(inoutID + %d) %% %d] = %stemp_%d.y%s;\n", sc.outputStride[1], sc.outputBufferBlockSize, sc.outputStride[1], sc.outputBufferBlockSize, convTypeLeft, i, convTypeRight);
							}
							else {
								if (sc.outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[indexOutput(inoutID)] = %ssdata[(combinedId %% %d) + (combinedId / %d) * sharedStride].x%s;\n", convTypeLeft, sc.fftDim, sc.fftDim, convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[indexOutput(inoutID) / %d].outputs[indexOutput(inoutID) %% %d] = %ssdata[(combinedId %% %d) + (combinedId / %d) * sharedStride].x%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, sc.fftDim, sc.fftDim, convTypeRight);
								if (sc.outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[indexOutput(inoutID + %d)] = %ssdata[(combinedId %% %d) + (combinedId / %d) * sharedStride].y%s;\n", sc.outputStride[1], convTypeLeft, sc.fftDim, sc.fftDim, convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[indexOutput(inoutID + %d) / %d].outputs[indexOutput(inoutID + %d) %% %d] = %ssdata[(combinedId %% %d) + (combinedId / %d) * sharedStride].y%s;\n", sc.outputStride[1], sc.outputBufferBlockSize, sc.outputStride[1], sc.outputBufferBlockSize, convTypeLeft, sc.fftDim, sc.fftDim, convTypeRight);

							}
							sprintf(output + strlen(output), "	}\n");
							if ((uint32_t)ceil(sc.size[1] / 2.0) % sc.localSize[1] != 0)
								sprintf(output + strlen(output), "		}");

						}
					}
					else {

					}
				}
				else {
					if (sc.fftDim == sc.fft_dim_full) {
						for (uint32_t i = 0; i < sc.min_registers_per_thread; i++) {

							if (sc.localSize[1] == 1)
								sprintf(output + strlen(output), "		combinedId = gl_LocalInvocationID.x + %d;\n", i * sc.localSize[0]);
							else
								sprintf(output + strlen(output), "		combinedId = (gl_LocalInvocationID.x + %d * gl_LocalInvocationID.y) + %d;\n", sc.localSize[0], i * sc.localSize[0] * sc.localSize[1]);

							if (sc.outputStride[0] > 1)
								sprintf(output + strlen(output), "		inoutID = indexOutput((combinedId %% %d) * %d + (combinedId / %d) * %d);\n", sc.fftDim, sc.outputStride[0], sc.fftDim, 2 * sc.outputStride[1]);
							else
								sprintf(output + strlen(output), "		inoutID = indexOutput((combinedId %% %d) + (combinedId / %d) * %d);\n", sc.fftDim, sc.fftDim, 2 * sc.outputStride[1]);
							if ((uint32_t)ceil(sc.size[1] / 2.0) % sc.localSize[1] != 0)
								sprintf(output + strlen(output), "		if(combinedId / %d + (gl_WorkGroupID.y%s)*gl_WorkGroupSize.y< %d){", sc.fftDim, shiftY, (uint32_t)ceil(sc.size[1] / 2.0));

							if (sc.writeFromRegisters) {
								if (sc.outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %stemp_%d.x%s;\n", convTypeLeft, i, convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %stemp_%d.x%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, i, convTypeRight);
								sprintf(output + strlen(output), "		inoutID += %d;\n", sc.outputStride[1]);
								if (sc.outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %stemp_%d.y%s;\n", convTypeLeft, i, convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %stemp_%d.y%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, i, convTypeRight);
							}
							else {
								if (sc.outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %ssdata[(combinedId %% %d) + (combinedId / %d) * sharedStride].x%s;\n", convTypeLeft, sc.fftDim, sc.fftDim, convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %ssdata[(combinedId %% %d) + (combinedId / %d) * sharedStride].x%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, sc.fftDim, sc.fftDim, convTypeRight);
								sprintf(output + strlen(output), "		inoutID += %d;\n", sc.outputStride[1]);
								if (sc.outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		outputBlocks[0].outputs[inoutID] = %ssdata[(combinedId %% %d) + (combinedId / %d) * sharedStride].y%s;\n", convTypeLeft, sc.fftDim, sc.fftDim, convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[inoutID / %d].outputs[inoutID %% %d] = %ssdata[(combinedId %% %d) + (combinedId / %d) * sharedStride].y%s;\n", sc.outputBufferBlockSize, sc.outputBufferBlockSize, convTypeLeft, sc.fftDim, sc.fftDim, convTypeRight);
							}
							if ((uint32_t)ceil(sc.size[1] / 2.0) % sc.localSize[1] != 0)
								sprintf(output + strlen(output), "		}", sc.fftDim, shiftY, (uint32_t)ceil(sc.size[1] / 2.0));
						}

					}
					else {

					}

				}
			}

			break;
		}
		}
	}

	static inline void shaderGenVkFFT(char* output, VkFFTSpecializationConstantsLayout sc, const char* floatType, const char* floatTypeMemory, const char* uintType, uint32_t type) {
		//appendLicense(output);
		appendVersion(output);
		appendExtensions(output, floatType, floatTypeMemory);
		appendLayoutVkFFT(output, sc);
		appendConstantsVkFFT(output, floatType, uintType);
		if ((!sc.LUT) && (!strcmp(floatType, "double")))
			appendSinCos20(output, floatType, uintType);
		appendPushConstantsVkFFT(output, sc, floatType, uintType);
		uint32_t id = 0;
		appendInputLayoutVkFFT(output, sc, id, floatTypeMemory, type);
		id++;
		appendOutputLayoutVkFFT(output, sc, id, floatTypeMemory, type);
		id++;
		if (sc.convolutionStep) {
			appendKernelLayoutVkFFT(output, sc, id, floatTypeMemory);
			id++;
		}
		if (sc.LUT) {
			appendLUTLayoutVkFFT(output, sc, id, floatType);
			id++;
		}
		appendIndexInputVkFFT(output, sc, uintType, type);
		appendIndexOutputVkFFT(output, sc, uintType, type);
		/*uint32_t appendedRadix[10] = { 0,0,0,0,0,0,0,0,0,0 };
		for (uint32_t i = 0; i < sc.numStages; i++) {
			if (appendedRadix[sc.stageRadix[i]] == 0) {
				appendedRadix[sc.stageRadix[i]] = 1;
				appendRadixKernelVkFFT(output, sc, floatType, uintType, sc.stageRadix[i]);
			}
		}*/
		appendSharedMemoryVkFFT(output, sc, floatType, uintType, type);
		sprintf(output + strlen(output), "void main() {\n");
		appendZeropadReturn(output, sc);
		//if (type==0) sprintf(output + strlen(output), "return;\n");
		appendInitialization(output, sc, floatType, uintType, type);
		if ((sc.convolutionStep) && (sc.matrixConvolution > 1))
			sprintf(output + strlen(output), "	for (uint coordinate=%d; coordinate > 0; coordinate--){\n\
	coordinate--;\n", sc.matrixConvolution);
		appendReadDataVkFFT(output, sc, floatType, floatTypeMemory, uintType, type);
		//appendBarrierVkFFT(output, 1);
		appendReorder4StepRead(output, sc, floatType, uintType, type);
		uint32_t stageSize = 1;
		uint32_t stageSizeSum = 0;
		double PI_const = 3.1415926535897932384626433832795;
		double stageAngle = (sc.inverse) ? -PI_const : PI_const;
		for (uint32_t i = 0; i < sc.numStages; i++) {
			if ((i == sc.numStages - 1) && ((type == 3) || (type == 4))) {
				appendRegisterBoostShuffle(output, sc, floatType, sc.stageRadix[i], type, 0);
				appendRadixStage(output, sc, floatType, uintType, stageSize, stageSizeSum, stageAngle, sc.stageRadix[i], type);
				appendRegisterBoostShuffle(output, sc, floatType, 8 / ((sc.sharedMemSize / sc.complexSize) / (sc.fftDim / sc.stageRadix[i])), type, 1);

			}
			else {
				appendRegisterBoostShuffle(output, sc, floatType, sc.stageRadix[i], type, 0);
				appendRadixStage(output, sc, floatType, uintType, stageSize, stageSizeSum, stageAngle, sc.stageRadix[i], type);
				stageSizeSum += stageSize;
				appendRadixShuffle(output, sc, floatType, uintType, stageSize, stageSizeSum, stageAngle, sc.stageRadix[i], type);
				stageSize *= sc.stageRadix[i];
				stageAngle /= sc.stageRadix[i];
			}
		}

		if (sc.convolutionStep) {
			appendCoordinateRegisterStore(output, sc, type);

			if (sc.matrixConvolution > 1)
				sprintf(output + strlen(output), "	coordinate++;}\n");

			if (sc.numKernels > 1)
				appendPreparationBatchedKernelConvolution(output, sc, floatType, floatTypeMemory, uintType, type);

			appendKernelConvolution(output, sc, floatType, floatTypeMemory, uintType, type);

			if (sc.matrixConvolution > 1)
				sprintf(output + strlen(output), "	for (uint coordinate=0; coordinate < %d; coordinate++){\n", sc.matrixConvolution);

			appendCoordinateRegisterPull(output, sc, type);

			stageSize = 1;
			stageSizeSum = 0;
			stageAngle = -PI_const;
			for (uint32_t i = 0; i < sc.numStages; i++) {
				appendRegisterBoostShuffle(output, sc, floatType, sc.stageRadix[i], type, 0);
				appendRadixStage(output, sc, floatType, uintType, stageSize, stageSizeSum, stageAngle, sc.stageRadix[i], type);
				stageSizeSum += stageSize;
				appendRadixShuffle(output, sc, floatType, uintType, stageSize, stageSizeSum, stageAngle, sc.stageRadix[i], type);
				stageSize *= sc.stageRadix[i];
				stageAngle /= sc.stageRadix[i];
			}

		}
		appendReorder4StepWrite(output, sc, floatType, uintType, type);
		appendWriteDataVkFFT(output, sc, floatType, floatTypeMemory, uintType, type);

		if ((sc.convolutionStep) && (sc.matrixConvolution > 1))
			sprintf(output + strlen(output), "	}\n");
		if ((sc.convolutionStep) && (sc.numKernels > 1))
			sprintf(output + strlen(output), "	}\n");
		sprintf(output + strlen(output), "}\n");
		//printf("%s\n", output);
	}

	static inline uint32_t findMemoryType(VkFFTApplication* app, uint32_t memoryTypeBits, uint32_t memorySize, VkMemoryPropertyFlags properties) {
		VkPhysicalDeviceMemoryProperties memoryProperties = { 0 };

		vkGetPhysicalDeviceMemoryProperties(app->configuration.physicalDevice[0], &memoryProperties);

		for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
			if ((memoryTypeBits & (1 << i)) && ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) && (memoryProperties.memoryHeaps[memoryProperties.memoryTypes[i].heapIndex].size >= memorySize))
				return i;
		}
		return -1;
	}
	static inline void allocateFFTBuffer(VkFFTApplication* app, VkBuffer* buffer, VkDeviceMemory* deviceMemory, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags propertyFlags, VkDeviceSize size) {
		uint32_t queueFamilyIndices;
		VkBufferCreateInfo bufferCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
		bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		bufferCreateInfo.queueFamilyIndexCount = 1;
		bufferCreateInfo.pQueueFamilyIndices = &queueFamilyIndices;
		bufferCreateInfo.size = size;
		bufferCreateInfo.usage = usageFlags;
		vkCreateBuffer(app->configuration.device[0], &bufferCreateInfo, NULL, buffer);
		VkMemoryRequirements memoryRequirements = { 0 };
		vkGetBufferMemoryRequirements(app->configuration.device[0], buffer[0], &memoryRequirements);
		VkMemoryAllocateInfo memoryAllocateInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
		memoryAllocateInfo.allocationSize = memoryRequirements.size;
		memoryAllocateInfo.memoryTypeIndex = findMemoryType(app, memoryRequirements.memoryTypeBits, memoryRequirements.size, propertyFlags);
		vkAllocateMemory(app->configuration.device[0], &memoryAllocateInfo, NULL, deviceMemory);
		vkBindBufferMemory(app->configuration.device[0], buffer[0], deviceMemory[0], 0);
	}
	static inline void transferDataFromCPU(VkFFTApplication* app, void* arr, VkBuffer* buffer, VkDeviceSize bufferSize) {
		VkDeviceSize stagingBufferSize = bufferSize;
		VkBuffer stagingBuffer = { 0 };
		VkDeviceMemory stagingBufferMemory = { 0 };
		allocateFFTBuffer(app, &stagingBuffer, &stagingBufferMemory, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBufferSize);

		void* data;
		vkMapMemory(app->configuration.device[0], stagingBufferMemory, 0, stagingBufferSize, 0, &data);
		memcpy(data, arr, stagingBufferSize);
		vkUnmapMemory(app->configuration.device[0], stagingBufferMemory);
		VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
		commandBufferAllocateInfo.commandPool = app->configuration.commandPool[0];
		commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		commandBufferAllocateInfo.commandBufferCount = 1;
		VkCommandBuffer commandBuffer = { 0 };
		vkAllocateCommandBuffers(app->configuration.device[0], &commandBufferAllocateInfo, &commandBuffer);
		VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
		VkBufferCopy copyRegion = { 0 };
		copyRegion.srcOffset = 0;
		copyRegion.dstOffset = 0;
		copyRegion.size = stagingBufferSize;
		vkCmdCopyBuffer(commandBuffer, stagingBuffer, buffer[0], 1, &copyRegion);
		vkEndCommandBuffer(commandBuffer);
		VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
		vkQueueSubmit(app->configuration.queue[0], 1, &submitInfo, app->configuration.fence[0]);
		vkWaitForFences(app->configuration.device[0], 1, app->configuration.fence, VK_TRUE, 100000000000);
		vkResetFences(app->configuration.device[0], 1, app->configuration.fence);
		vkFreeCommandBuffers(app->configuration.device[0], app->configuration.commandPool[0], 1, &commandBuffer);
		vkDestroyBuffer(app->configuration.device[0], stagingBuffer, NULL);
		vkFreeMemory(app->configuration.device[0], stagingBufferMemory, NULL);
	}
	static inline VkResult VkFFTScheduler(VkFFTApplication* app, VkFFTPlan* FFTPlan, uint32_t axis_id) {
		uint32_t complexSize;
		if (app->configuration.doublePrecision)
			complexSize = (2 * sizeof(double));
		else
			if (app->configuration.halfPrecision)
				complexSize = (2 * sizeof(float));
			else
				complexSize = (2 * sizeof(float));
		uint32_t multipliers[20] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };//split the sequence
		uint32_t isPowOf2 = (pow(2, (uint32_t)log2(app->configuration.size[axis_id])) == app->configuration.size[axis_id]) ? 1 : 0;
		uint32_t tempSequence = app->configuration.size[axis_id];
		for (uint32_t i = 2; i < 8; i++) {
			if (tempSequence % i == 0) {
				tempSequence /= i;
				multipliers[i]++;
				i--;
			}
		}
		if (tempSequence != 1) return VK_ERROR_FORMAT_NOT_SUPPORTED;

		uint32_t registerBoost = app->configuration.registerBoost;
		uint32_t maxSequenceLengthSharedMemory = app->configuration.sharedMemorySize / complexSize;
		uint32_t maxSingleSizeNonStrided = maxSequenceLengthSharedMemory;
		if ((axis_id == 0) && (!app->configuration.performConvolution)) maxSingleSizeNonStrided *= registerBoost;
		uint32_t maxSingleSizeStrided = (app->configuration.coalescedMemory > complexSize) ? app->configuration.sharedMemorySize / (app->configuration.coalescedMemory) : app->configuration.sharedMemorySize / complexSize;
		uint32_t numPasses = 1;
		uint32_t numPassesHalfBandwidth = 1;
		uint32_t temp;
		temp = (axis_id == 0) ? ceil(app->configuration.size[axis_id] / (double)maxSingleSizeNonStrided) : ceil(app->configuration.size[axis_id] / (double)maxSingleSizeStrided);
		if (temp > 1) {//more passes than one
			registerBoost = app->configuration.registerBoost4Step;
			if ((axis_id == 0) && (!app->configuration.performConvolution)) maxSingleSizeNonStrided = maxSequenceLengthSharedMemory * registerBoost;
			temp = ((axis_id == 0) && (!app->configuration.reorderFourStep)) ? app->configuration.size[axis_id] / maxSingleSizeNonStrided : app->configuration.size[axis_id] / maxSingleSizeStrided;
			if (app->configuration.reorderFourStep)
				numPasses = (uint32_t)ceil(log2(app->configuration.size[axis_id]) / log2(maxSingleSizeStrided));
			else
				numPasses += (uint32_t)ceil(log2(temp) / log2(maxSingleSizeStrided));
		}
		uint32_t maxSingleSizeStridedHalfBandwidth = maxSingleSizeStrided;
		if ((app->configuration.performHalfBandwidthBoost) && (!((axis_id == 0) && (numPasses == 2)))) {
			maxSingleSizeStridedHalfBandwidth = (app->configuration.coalescedMemory / 2 > complexSize) ? app->configuration.sharedMemorySize / (app->configuration.coalescedMemory / 2) : app->configuration.sharedMemorySize / complexSize;
			temp = (axis_id == 0) ? app->configuration.size[axis_id] / maxSingleSizeNonStrided : app->configuration.size[axis_id] / maxSingleSizeStridedHalfBandwidth;
			//temp = app->configuration.size[axis_id] / maxSingleSizeNonStrided;
			if (temp > 1) {//more passes than two
				temp = (!app->configuration.reorderFourStep) ? app->configuration.size[axis_id] / maxSingleSizeNonStrided : app->configuration.size[axis_id] / maxSingleSizeStridedHalfBandwidth;
				numPassesHalfBandwidth = 1 + (uint32_t)ceil(log2(temp) / log2(maxSingleSizeStrided));
				/*
				temp = ((axis_id == 0) && (!app->configuration.reorderFourStep)) ? app->configuration.size[axis_id] / maxSingleSizeNonStrided : app->configuration.size[axis_id] / maxSingleSizeStridedHalfBandwidth;

				if (app->configuration.reorderFourStep)
					numPassesHalfBandwidth = (uint32_t)ceil(log2(app->configuration.size[axis_id]) / log2(maxSingleSizeStridedHalfBandwidth));
				else
					numPassesHalfBandwidth = 1 + (uint32_t)ceil(log2(temp) / log2(maxSingleSizeStridedHalfBandwidth));
				if ((numPassesHalfBandwidth == 2)&& (!app->configuration.reorderFourStep)&&(registerBoost>1)) //switch back for two step and don't do half bandwidth on strided accesses if register boost and no 4-step reordering
				*/
			}
			if (numPassesHalfBandwidth < numPasses) numPasses = numPassesHalfBandwidth;
		}
		if (((uint32_t)log2(app->configuration.size[axis_id]) >= app->configuration.swapTo3Stage4Step) && (app->configuration.swapTo3Stage4Step >= 17)) numPasses = 3;//Force set to 3 stage 4 step algorithm
		if (numPasses == 1) {
			FFTPlan->axisSplit[axis_id][0] = app->configuration.size[axis_id];
		}
		if (numPasses == 2) {
			if (isPowOf2) {
				if ((axis_id == 0) && (!app->configuration.reorderFourStep)) {
					uint32_t maxPow8SharedMemory = (uint32_t)pow(8, ((uint32_t)log2(maxSequenceLengthSharedMemory)) / 3);
					//unit stride
					if (app->configuration.size[axis_id] / maxPow8SharedMemory <= maxSingleSizeStrided) {
						FFTPlan->axisSplit[axis_id][0] = maxPow8SharedMemory;
					}
					else {
						if (app->configuration.size[axis_id] / maxSequenceLengthSharedMemory <= maxSingleSizeStrided) {
							FFTPlan->axisSplit[axis_id][0] = maxSequenceLengthSharedMemory;
						}
						else {
							if (app->configuration.size[axis_id] / (maxSequenceLengthSharedMemory * registerBoost) < maxSingleSizeStridedHalfBandwidth) {
								for (uint32_t i = 1; i <= (uint32_t)log2(registerBoost); i++) {
									if (app->configuration.size[axis_id] / (maxSequenceLengthSharedMemory * (uint32_t)pow(2, i)) <= maxSingleSizeStrided) {
										FFTPlan->axisSplit[axis_id][0] = (maxSequenceLengthSharedMemory * (uint32_t)pow(2, i));
										i = (uint32_t)log2(registerBoost) + 1;
									}
								}
							}
							else {
								FFTPlan->axisSplit[axis_id][0] = (maxSequenceLengthSharedMemory * registerBoost);
							}
						}
					}
				}
				else {
					uint32_t maxPow8Strided = (uint32_t)pow(8, ((uint32_t)log2(maxSingleSizeStrided)) / 3);
					//all FFTs are considered as non-unit stride
					if (app->configuration.size[axis_id] / maxPow8Strided <= maxSingleSizeStrided) {
						FFTPlan->axisSplit[axis_id][0] = maxPow8Strided;
					}
					else {
						if (app->configuration.size[axis_id] / maxSingleSizeStrided < maxSingleSizeStridedHalfBandwidth) {
							FFTPlan->axisSplit[axis_id][0] = maxSingleSizeStrided;
						}
						else {
							FFTPlan->axisSplit[axis_id][0] = maxSingleSizeStridedHalfBandwidth;
						}
					}
				}
				FFTPlan->axisSplit[axis_id][1] = app->configuration.size[axis_id] / FFTPlan->axisSplit[axis_id][0];
				if (FFTPlan->axisSplit[axis_id][1] < 64) {
					FFTPlan->axisSplit[axis_id][0] = (FFTPlan->axisSplit[axis_id][1] == 0) ? FFTPlan->axisSplit[axis_id][0] / (64) : FFTPlan->axisSplit[axis_id][0] / (64 / FFTPlan->axisSplit[axis_id][1]);
					FFTPlan->axisSplit[axis_id][1] = 64;
				}
				if (FFTPlan->axisSplit[axis_id][1] > FFTPlan->axisSplit[axis_id][0]) {
					uint32_t swap = FFTPlan->axisSplit[axis_id][0];
					FFTPlan->axisSplit[axis_id][0] = FFTPlan->axisSplit[axis_id][1];
					FFTPlan->axisSplit[axis_id][1] = swap;
				}
			}
			else {
				uint32_t successSplit = 0;
				if ((axis_id == 0) && (!app->configuration.reorderFourStep)) {
					for (uint32_t i = 0; i < maxSequenceLengthSharedMemory; i++) {
						if (app->configuration.size[axis_id] % (maxSequenceLengthSharedMemory - i) == 0) {
							if (((maxSequenceLengthSharedMemory - i) <= maxSequenceLengthSharedMemory) && (app->configuration.size[axis_id] / (maxSequenceLengthSharedMemory - i) <= maxSingleSizeStrided)) {
								FFTPlan->axisSplit[axis_id][0] = (maxSequenceLengthSharedMemory - i);
								FFTPlan->axisSplit[axis_id][1] = app->configuration.size[axis_id] / (maxSequenceLengthSharedMemory - i);
								i = maxSequenceLengthSharedMemory;
								successSplit = 1;
							}
						}
					}
				}
				else {
					uint32_t sqrtSequence = ceil(sqrt(app->configuration.size[axis_id]));
					for (uint32_t i = 0; i < sqrtSequence; i++) {
						if (app->configuration.size[axis_id] % (sqrtSequence - i) == 0) {
							if ((sqrtSequence - i <= maxSingleSizeStrided) && (app->configuration.size[axis_id] / (sqrtSequence - i) <= maxSingleSizeStridedHalfBandwidth)) {
								FFTPlan->axisSplit[axis_id][0] = app->configuration.size[axis_id] / (sqrtSequence - i);
								FFTPlan->axisSplit[axis_id][1] = sqrtSequence - i;
								i = sqrtSequence;
								successSplit = 1;
							}
						}
					}
				}
				if (successSplit == 0)
					numPasses = 3;
			}
		}
		if (numPasses == 3) {
			if (isPowOf2) {
				uint32_t maxPow8Strided = (uint32_t)pow(8, ((uint32_t)log2(maxSingleSizeStrided)) / 3);
				if ((axis_id == 0) && (!app->configuration.reorderFourStep)) {
					//unit stride
					uint32_t maxPow8SharedMemory = (uint32_t)pow(8, ((uint32_t)log2(maxSequenceLengthSharedMemory)) / 3);
					if (app->configuration.size[axis_id] / maxPow8SharedMemory <= maxPow8Strided * maxPow8Strided)
						FFTPlan->axisSplit[axis_id][0] = maxPow8SharedMemory;
					else {
						if (app->configuration.size[axis_id] / maxSequenceLengthSharedMemory <= maxSingleSizeStrided * maxSingleSizeStrided)
							FFTPlan->axisSplit[axis_id][0] = maxSequenceLengthSharedMemory;
						else {
							if (app->configuration.size[axis_id] / (maxSequenceLengthSharedMemory * registerBoost) <= maxSingleSizeStrided * maxSingleSizeStrided) {
								for (uint32_t i = 0; i <= (uint32_t)log2(registerBoost); i++) {
									if (app->configuration.size[axis_id] / (maxSequenceLengthSharedMemory * (uint32_t)pow(2, i)) <= maxSingleSizeStrided * maxSingleSizeStrided) {
										FFTPlan->axisSplit[axis_id][0] = (maxSequenceLengthSharedMemory * (uint32_t)pow(2, i));
										i = (uint32_t)log2(registerBoost) + 1;
									}
								}
							}
							else {
								FFTPlan->axisSplit[axis_id][0] = (maxSequenceLengthSharedMemory * registerBoost);
							}
						}
					}
				}
				else {
					//to account for TLB misses, it is best to coalesce the unit-strided stage to 128 bytes
					/*uint32_t log2axis = (uint32_t)log2(app->configuration.size[axis_id]);
					FFTPlan->axisSplit[axis_id][0] = (uint32_t)pow(2, (uint32_t)log2axis / 3);
					if (log2axis % 3 > 0) FFTPlan->axisSplit[axis_id][0] *= 2;
					FFTPlan->axisSplit[axis_id][1] = (uint32_t)pow(2, (uint32_t)log2axis / 3);
					if (log2axis % 3 > 1) FFTPlan->axisSplit[axis_id][1] *= 2;
					FFTPlan->axisSplit[axis_id][2] = app->configuration.size[axis_id] / FFTPlan->axisSplit[axis_id][0] / FFTPlan->axisSplit[axis_id][1];*/
					uint32_t maxSingleSizeStrided128 = app->configuration.sharedMemorySize / (128);
					uint32_t maxPow8_128 = (uint32_t)pow(8, ((uint32_t)log2(maxSingleSizeStrided128)) / 3);
					//unit stride
					if (app->configuration.size[axis_id] / maxPow8_128 <= maxPow8Strided * maxSingleSizeStrided)
						FFTPlan->axisSplit[axis_id][0] = maxPow8_128;
					//non-unit stride
					else {

						if ((app->configuration.size[axis_id] / (maxPow8_128 * 2) <= maxPow8Strided * maxSingleSizeStrided) && (maxPow8_128 * 2 <= maxSingleSizeStrided128)) {
							FFTPlan->axisSplit[axis_id][0] = maxPow8_128 * 2;
						}
						else {
							if ((app->configuration.size[axis_id] / (maxPow8_128 * 4) <= maxPow8Strided * maxSingleSizeStrided) && (maxPow8_128 * 4 <= maxSingleSizeStrided128)) {
								FFTPlan->axisSplit[axis_id][0] = maxPow8_128 * 4;
							}
							else {
								if (app->configuration.size[axis_id] / maxSingleSizeStrided <= maxSingleSizeStrided * maxSingleSizeStrided) {
									for (uint32_t i = 0; i <= (uint32_t)log2(maxSingleSizeStrided / maxSingleSizeStrided128); i++) {
										if (app->configuration.size[axis_id] / (maxSingleSizeStrided128 * (uint32_t)pow(2, i)) <= maxSingleSizeStrided * maxSingleSizeStrided) {
											FFTPlan->axisSplit[axis_id][0] = (maxSingleSizeStrided128 * (uint32_t)pow(2, i));
											i = (uint32_t)log2(maxSingleSizeStrided / maxSingleSizeStrided128) + 1;
										}
									}
								}
								else
									FFTPlan->axisSplit[axis_id][0] = maxSingleSizeStridedHalfBandwidth;
							}
						}
					}
				}
				if (app->configuration.size[axis_id] / FFTPlan->axisSplit[axis_id][0] / maxPow8Strided <= maxSingleSizeStrided) {
					FFTPlan->axisSplit[axis_id][1] = maxPow8Strided;
					FFTPlan->axisSplit[axis_id][2] = app->configuration.size[axis_id] / FFTPlan->axisSplit[axis_id][1] / FFTPlan->axisSplit[axis_id][0];
				}
				else {
					if (app->configuration.size[axis_id] / FFTPlan->axisSplit[axis_id][0] / maxSingleSizeStrided <= maxSingleSizeStrided) {
						FFTPlan->axisSplit[axis_id][1] = maxSingleSizeStrided;
						FFTPlan->axisSplit[axis_id][2] = app->configuration.size[axis_id] / FFTPlan->axisSplit[axis_id][1] / FFTPlan->axisSplit[axis_id][0];
					}
					else {
						FFTPlan->axisSplit[axis_id][1] = maxSingleSizeStridedHalfBandwidth;
						FFTPlan->axisSplit[axis_id][2] = app->configuration.size[axis_id] / FFTPlan->axisSplit[axis_id][1] / FFTPlan->axisSplit[axis_id][0];
					}
				}
				if (FFTPlan->axisSplit[axis_id][2] < 64) {
					FFTPlan->axisSplit[axis_id][1] = (FFTPlan->axisSplit[axis_id][2] == 0) ? FFTPlan->axisSplit[axis_id][1] / (64) : FFTPlan->axisSplit[axis_id][1] / (64 / FFTPlan->axisSplit[axis_id][2]);
					FFTPlan->axisSplit[axis_id][2] = 64;
				}
				if (FFTPlan->axisSplit[axis_id][2] > FFTPlan->axisSplit[axis_id][1]) {
					uint32_t swap = FFTPlan->axisSplit[axis_id][1];
					FFTPlan->axisSplit[axis_id][1] = FFTPlan->axisSplit[axis_id][2];
					FFTPlan->axisSplit[axis_id][2] = swap;
				}
			}
			else {
				uint32_t successSplit = 0;
				if ((axis_id == 0) && (!app->configuration.reorderFourStep)) {
					for (uint32_t i = 0; i < maxSequenceLengthSharedMemory; i++) {
						if (app->configuration.size[axis_id] % (maxSequenceLengthSharedMemory - i) == 0) {
							uint32_t sqrt3Sequence = ceil(sqrt(app->configuration.size[axis_id] / (maxSequenceLengthSharedMemory - i)));
							for (uint32_t j = 0; j < sqrt3Sequence; j++) {
								if ((app->configuration.size[axis_id] / (maxSequenceLengthSharedMemory - i)) % (sqrt3Sequence - j) == 0) {
									if (((maxSequenceLengthSharedMemory - i) <= maxSequenceLengthSharedMemory) && (sqrt3Sequence - j <= maxSingleSizeStrided) && (app->configuration.size[axis_id] / (maxSequenceLengthSharedMemory - i) / (sqrt3Sequence - j) <= maxSingleSizeStrided)) {
										FFTPlan->axisSplit[axis_id][0] = (maxSequenceLengthSharedMemory - i);
										FFTPlan->axisSplit[axis_id][1] = sqrt3Sequence - j;
										FFTPlan->axisSplit[axis_id][2] = app->configuration.size[axis_id] / (maxSequenceLengthSharedMemory - i) / (sqrt3Sequence - j);
										i = maxSequenceLengthSharedMemory;
										j = sqrt3Sequence;
										successSplit = 1;
									}
								}
							}
						}
					}
				}
				else {
					uint32_t sqrt3Sequence = ceil(pow(app->configuration.size[axis_id], 1.0 / 3.0));
					for (uint32_t i = 0; i < sqrt3Sequence; i++) {
						if (app->configuration.size[axis_id] % (sqrt3Sequence - i) == 0) {
							uint32_t sqrt2Sequence = ceil(sqrt(app->configuration.size[axis_id] / (sqrt3Sequence - i)));
							for (uint32_t j = 0; j < sqrt2Sequence; j++) {
								if ((app->configuration.size[axis_id] / (sqrt3Sequence - i)) % (sqrt2Sequence - j) == 0) {
									if ((sqrt3Sequence - i <= maxSingleSizeStrided) && (sqrt2Sequence - j <= maxSingleSizeStrided) && (app->configuration.size[axis_id] / (sqrt3Sequence - i) / (sqrt2Sequence - j) <= maxSingleSizeStridedHalfBandwidth)) {
										FFTPlan->axisSplit[axis_id][0] = app->configuration.size[axis_id] / (sqrt3Sequence - i) / (sqrt2Sequence - j);
										FFTPlan->axisSplit[axis_id][1] = sqrt3Sequence - i;
										FFTPlan->axisSplit[axis_id][2] = sqrt2Sequence - j;
										i = sqrt3Sequence;
										j = sqrt2Sequence;
										successSplit = 1;
									}
								}
							}
						}
					}
				}
				if (successSplit == 0)
					numPasses = 4;
			}
		}
		if (numPasses > 3) {
			printf("sequence length exceeds boundaries\n");
			return VK_ERROR_FORMAT_NOT_SUPPORTED;
		}
		for (uint32_t i = 0; i < numPasses; i++) {
			if ((FFTPlan->axisSplit[axis_id][0] % 2 != 0) && (FFTPlan->axisSplit[axis_id][i] % 2 == 0)) {
				uint32_t swap = FFTPlan->axisSplit[axis_id][0];
				FFTPlan->axisSplit[axis_id][0] = FFTPlan->axisSplit[axis_id][i];
				FFTPlan->axisSplit[axis_id][i] = swap;
			}
		}
		for (uint32_t i = 0; i < numPasses; i++) {
			if ((FFTPlan->axisSplit[axis_id][0] % 4 != 0) && (FFTPlan->axisSplit[axis_id][i] % 4 == 0)) {
				uint32_t swap = FFTPlan->axisSplit[axis_id][0];
				FFTPlan->axisSplit[axis_id][0] = FFTPlan->axisSplit[axis_id][i];
				FFTPlan->axisSplit[axis_id][i] = swap;
			}
		}
		for (uint32_t i = 0; i < numPasses; i++) {
			if ((FFTPlan->axisSplit[axis_id][0] % 8 != 0) && (FFTPlan->axisSplit[axis_id][i] % 8 == 0)) {
				uint32_t swap = FFTPlan->axisSplit[axis_id][0];
				FFTPlan->axisSplit[axis_id][0] = FFTPlan->axisSplit[axis_id][i];
				FFTPlan->axisSplit[axis_id][i] = swap;
			}
		}
		FFTPlan->numAxisUploads[axis_id] = numPasses;
		for (uint32_t k = 0; k < numPasses; k++) {
			tempSequence = FFTPlan->axisSplit[axis_id][k];
			uint32_t loc_multipliers[20] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };//split the smaller sequence
			for (uint32_t i = 2; i < 8; i++) {
				if (tempSequence % i == 0) {
					tempSequence /= i;
					loc_multipliers[i]++;
					i--;
				}
			}
			uint32_t registers_per_thread = 8;
			uint32_t min_registers_per_thread = 8;
			if (loc_multipliers[2] > 0) {
				if (loc_multipliers[3] > 0) {
					if (loc_multipliers[5] > 0) {
						if ((loc_multipliers[2] == 1) || (app->configuration.doublePrecision)) {
							registers_per_thread = 6;
							min_registers_per_thread = 5;
						}
						else {
							registers_per_thread = 12;
							min_registers_per_thread = 10;
						}
					}
					else
					{
						if ((loc_multipliers[2] == 1) || (app->configuration.doublePrecision)) {
							registers_per_thread = 6;
							min_registers_per_thread = 6;
						}
						else {
							registers_per_thread = 12;
							min_registers_per_thread = 12;
						}
					}
				}
				else {
					if (loc_multipliers[5] > 0) {
						registers_per_thread = 10;
						min_registers_per_thread = 10;
					}
					else
					{

						registers_per_thread = (loc_multipliers[2] > 2) ? 8 : pow(2, loc_multipliers[2]);
						min_registers_per_thread = (loc_multipliers[2] > 2) ? 8 : pow(2, loc_multipliers[2]);
					}
				}
			}
			else {
				if (loc_multipliers[3] > 0) {
					if (loc_multipliers[5] > 0) {
						if (app->configuration.doublePrecision) {
							registers_per_thread = 5;
							min_registers_per_thread = 3;
						}
						else {
							registers_per_thread = 15;
							min_registers_per_thread = 15;
						}
					}
					else
					{
						if ((loc_multipliers[3] == 1) || (app->configuration.doublePrecision)) {
							registers_per_thread = 3;
							min_registers_per_thread = 3;
						}
						else {
							registers_per_thread = 9;
							min_registers_per_thread = 9;
						}
					}
				}
				else {
					if (loc_multipliers[5] > 0) {
						registers_per_thread = 5;
						min_registers_per_thread = 5;
					}
					else
					{
						return VK_ERROR_FORMAT_NOT_SUPPORTED;
					}
				}

			}
			if (registers_per_thread % 8 == 0) {
				loc_multipliers[8] = loc_multipliers[2] / 3;
				loc_multipliers[2] = loc_multipliers[2] - loc_multipliers[8] * 3;
			}
			if (registers_per_thread % 4 == 0) {
				loc_multipliers[4] = loc_multipliers[2] / 2;
				loc_multipliers[2] = loc_multipliers[2] - loc_multipliers[4] * 2;
			}
			uint32_t j = 0;
			FFTPlan->axes[axis_id][k].specializationConstants.registers_per_thread = registers_per_thread;
			FFTPlan->axes[axis_id][k].specializationConstants.min_registers_per_thread = min_registers_per_thread;
			FFTPlan->axes[axis_id][k].specializationConstants.numStages = 0;
			FFTPlan->axes[axis_id][k].specializationConstants.fftDim = FFTPlan->axisSplit[axis_id][k];
			for (uint32_t i = 8; i > 1; i--) {
				if (loc_multipliers[i] > 0) {
					FFTPlan->axes[axis_id][k].specializationConstants.stageRadix[j] = i;
					loc_multipliers[i]--;
					i++;
					j++;
					FFTPlan->axes[axis_id][k].specializationConstants.numStages++;
				}
			}
			if (min_registers_per_thread != registers_per_thread) {
				j = FFTPlan->axes[axis_id][k].specializationConstants.stageRadix[FFTPlan->axes[axis_id][k].specializationConstants.numStages - 1];
				FFTPlan->axes[axis_id][k].specializationConstants.stageRadix[FFTPlan->axes[axis_id][k].specializationConstants.numStages - 1] = FFTPlan->axes[axis_id][k].specializationConstants.stageRadix[0];
				FFTPlan->axes[axis_id][k].specializationConstants.stageRadix[0] = j;
			}
		}
	}
	static inline VkResult VkFFTPlanSupportAxis(VkFFTApplication* app, VkFFTPlan* FFTPlan, uint32_t axis_id, uint32_t axis_upload_id, VkBool32 inverse, uint32_t convolutionInverseStage) {
		//get radix stages
		VkFFTAxis* axis = &FFTPlan->supportAxes[axis_id - 1][axis_upload_id];
		axis->specializationConstants.inverse = inverse;
		axis->specializationConstants.supportAxis = 1;
		axis->specializationConstants.symmetricKernel = app->configuration.symmetricKernel;
		uint32_t complexSize;
		if (app->configuration.doublePrecision)
			complexSize = (2 * sizeof(double));
		else
			if (app->configuration.halfPrecision)
				complexSize = (2 * sizeof(float));
			else
				complexSize = (2 * sizeof(float));
		axis->specializationConstants.complexSize = complexSize;
		uint32_t maxSequenceLengthSharedMemory = app->configuration.sharedMemorySize / complexSize;
		uint32_t maxSequenceLengthSharedMemoryPow2 = app->configuration.sharedMemorySizePow2 / complexSize;
		uint32_t maxSingleSizeStrided = (app->configuration.coalescedMemory > complexSize) ? app->configuration.sharedMemorySize / (app->configuration.coalescedMemory) : app->configuration.sharedMemorySize / complexSize;
		uint32_t maxSingleSizeStridedPow2 = (app->configuration.coalescedMemory > complexSize) ? app->configuration.sharedMemorySizePow2 / (app->configuration.coalescedMemory) : app->configuration.sharedMemorySizePow2 / complexSize;

		axis->specializationConstants.stageStartSize = 1;
		for (uint32_t i = 0; i < axis_upload_id; i++)
			axis->specializationConstants.stageStartSize *= FFTPlan->axisSplit[axis_id][i];

		axis->specializationConstants.firstStageStartSize = app->configuration.size[axis_id] / FFTPlan->axisSplit[axis_id][FFTPlan->numAxisUploads[axis_id] - 1];

		axis->specializationConstants.fft_dim_x = app->configuration.size[1];
		axis->specializationConstants.performR2C = 0;
		axis->specializationConstants.reorderFourStep = (FFTPlan->numSupportAxisUploads[axis_id - 1] > 1) ? app->configuration.reorderFourStep : 0;
		uint32_t passID = FFTPlan->numSupportAxisUploads[axis_id - 1] - 1 - axis_upload_id;
		axis->specializationConstants.fft_dim_full = app->configuration.size[axis_id];
		uint32_t maxBatchCoalesced;
		if (app->configuration.doublePrecision)
			maxBatchCoalesced = app->configuration.coalescedMemory / (2 * sizeof(double));
		else
			if (app->configuration.halfPrecision)
				maxBatchCoalesced = app->configuration.coalescedMemory / (2 * sizeof(float));
			else
				maxBatchCoalesced = app->configuration.coalescedMemory / (2 * sizeof(float));

		axis->groupedBatch = maxBatchCoalesced;
		/*if ((app->configuration.size[0] < 4096) && (app->configuration.size[1] < 512) && (app->configuration.size[2] == 1)) {
			if (app->configuration.sharedMemorySize / axis->specializationConstants.fftDim >= app->configuration.coalescedMemory) {
				if (1024 / axis->specializationConstants.fftDim < maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim) {
					if (1024 / axis->specializationConstants.fftDim > axis->groupedBatch)
						axis->groupedBatch = 1024 / axis->specializationConstants.fftDim;
					else
						axis->groupedBatch = maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim;
				}
			}
		}
		else {
			axis->groupedBatch = (app->configuration.sharedMemorySize / axis->specializationConstants.fftDim >= app->configuration.coalescedMemory) ? maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim : axis->groupedBatch;
		}*/
		//axis->groupedBatch = (app->configuration.sharedMemorySize / axis->specializationConstants.fftDim >= app->configuration.coalescedMemory) ? maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim : axis->groupedBatch;
		if (((FFTPlan->numAxisUploads[axis_id] == 1) && (axis_id == 0)) || ((axis_id == 0) && (!app->configuration.reorderFourStep) && (axis_upload_id == 0))) {
			axis->groupedBatch = (maxSequenceLengthSharedMemoryPow2 / axis->specializationConstants.fftDim > axis->groupedBatch) ? maxSequenceLengthSharedMemoryPow2 / axis->specializationConstants.fftDim : axis->groupedBatch;
		}
		else {
			axis->groupedBatch = (maxSingleSizeStridedPow2 / axis->specializationConstants.fftDim > 1) ? maxSingleSizeStridedPow2 / axis->specializationConstants.fftDim * axis->groupedBatch : axis->groupedBatch;
		}
		if (axis->groupedBatch < maxBatchCoalesced) axis->groupedBatch = maxBatchCoalesced;
		axis->groupedBatch = (axis->groupedBatch / maxBatchCoalesced) * maxBatchCoalesced;
		if (app->configuration.halfThreads)
			axis->groupedBatch = ceil(axis->groupedBatch / 2.0);
		//allocate LUT 
		if (app->configuration.useLUT) {
			double double_PI = 3.1415926535897932384626433832795;
			uint32_t dimMult = 1;
			uint32_t maxStageSum = 0;
			for (uint32_t i = 0; i < axis->specializationConstants.numStages; i++) {
				maxStageSum += dimMult;
				dimMult *= axis->specializationConstants.stageRadix[i];
			}
			axis->specializationConstants.maxStageSumLUT = maxStageSum;
			if (app->configuration.doublePrecision) {
				if (axis_upload_id > 0)
					axis->bufferLUTSize = (3 * maxStageSum + axis->specializationConstants.stageStartSize * axis->specializationConstants.fftDim) * 2 * sizeof(double);
				else
					axis->bufferLUTSize = (3 * maxStageSum) * 2 * sizeof(double);
				double* tempLUT = (double*)malloc(axis->bufferLUTSize);
				uint32_t localStageSize = 1;
				uint32_t localStageSum = 0;
				for (uint32_t i = 0; i < axis->specializationConstants.numStages; i++) {
					for (uint32_t j = 0; j < localStageSize; j++) {
						if (inverse) {
							tempLUT[2 * (j + localStageSum)] = cos(-j * double_PI / localStageSize);
							tempLUT[2 * (j + localStageSum) + 1] = sin(-j * double_PI / localStageSize);
						}
						else {
							tempLUT[2 * (j + localStageSum)] = cos(j * double_PI / localStageSize);
							tempLUT[2 * (j + localStageSum) + 1] = sin(j * double_PI / localStageSize);
						}
					}
					localStageSum += localStageSize;
					localStageSize *= axis->specializationConstants.stageRadix[i];
				}
				localStageSize = 1;
				localStageSum = 0;
				for (uint32_t i = 0; i < axis->specializationConstants.numStages; i++) {
					for (uint32_t j = 0; j < localStageSize; j++) {
						if (inverse) {
							tempLUT[maxStageSum * 2 + 2 * (j + localStageSum)] = cos(-j * double_PI / localStageSize / 2);
							tempLUT[maxStageSum * 2 + 2 * (j + localStageSum) + 1] = sin(-j * double_PI / localStageSize / 2);
						}
						else {
							tempLUT[maxStageSum * 2 + 2 * (j + localStageSum)] = cos(j * double_PI / localStageSize / 2);
							tempLUT[maxStageSum * 2 + 2 * (j + localStageSum) + 1] = sin(j * double_PI / localStageSize / 2);
						}
					}
					localStageSum += localStageSize;
					localStageSize *= axis->specializationConstants.stageRadix[i];
				}
				localStageSize = 1;
				localStageSum = 0;
				for (uint32_t i = 0; i < axis->specializationConstants.numStages; i++) {
					for (uint32_t j = 0; j < localStageSize; j++) {
						if (inverse) {
							tempLUT[2 * maxStageSum * 2 + 2 * (j + localStageSum)] = cos(-j * double_PI / localStageSize / 4);
							tempLUT[2 * maxStageSum * 2 + 2 * (j + localStageSum) + 1] = sin(-j * double_PI / localStageSize / 4);
						}
						else {
							tempLUT[2 * maxStageSum * 2 + 2 * (j + localStageSum)] = cos(j * double_PI / localStageSize / 4);
							tempLUT[2 * maxStageSum * 2 + 2 * (j + localStageSum) + 1] = sin(j * double_PI / localStageSize / 4);
						}
					}
					localStageSum += localStageSize;
					localStageSize *= axis->specializationConstants.stageRadix[i];
				}
				if (axis_upload_id > 0)
					for (uint32_t i = 0; i < axis->specializationConstants.stageStartSize; i++) {
						for (uint32_t j = 0; j < axis->specializationConstants.fftDim; j++) {
							double angle = 2 * double_PI * ((i * j) / (double)(axis->specializationConstants.stageStartSize * axis->specializationConstants.fftDim));
							if (inverse) {
								tempLUT[3 * maxStageSum * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize)] = cos(-angle);
								tempLUT[3 * maxStageSum * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize) + 1] = sin(-angle);
							}
							else {
								tempLUT[3 * maxStageSum * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize)] = cos(angle);
								tempLUT[3 * maxStageSum * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize) + 1] = sin(angle);
							}
						}
					}
				allocateFFTBuffer(app, &axis->bufferLUT, &axis->bufferLUTDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, axis->bufferLUTSize);
				transferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
				free(tempLUT);
			}
			else {
				uint32_t dimMult = 1;
				uint32_t maxStageSum = 0;
				for (uint32_t i = 0; i < axis->specializationConstants.numStages; i++) {
					maxStageSum += dimMult;
					dimMult *= axis->specializationConstants.stageRadix[i];
				}
				axis->specializationConstants.maxStageSumLUT = maxStageSum;
				if (axis_upload_id > 0)
					axis->bufferLUTSize = (3 * maxStageSum + axis->specializationConstants.stageStartSize * axis->specializationConstants.fftDim) * 2 * sizeof(float);
				else
					axis->bufferLUTSize = (3 * maxStageSum) * 2 * sizeof(float);
				float* tempLUT = (float*)malloc(axis->bufferLUTSize);
				uint32_t localStageSize = 1;
				uint32_t localStageSum = 0;
				for (uint32_t i = 0; i < axis->specializationConstants.numStages; i++) {
					for (uint32_t j = 0; j < localStageSize; j++) {
						if (inverse) {
							tempLUT[2 * (j + localStageSum)] = (float)cos(-j * double_PI / localStageSize);
							tempLUT[2 * (j + localStageSum) + 1] = (float)sin(-j * double_PI / localStageSize);
						}
						else {
							tempLUT[2 * (j + localStageSum)] = (float)cos(j * double_PI / localStageSize);
							tempLUT[2 * (j + localStageSum) + 1] = (float)sin(j * double_PI / localStageSize);
						}
					}
					localStageSum += localStageSize;
					localStageSize *= axis->specializationConstants.stageRadix[i];
				}
				localStageSize = 1;
				localStageSum = 0;
				for (uint32_t i = 0; i < axis->specializationConstants.numStages; i++) {
					for (uint32_t j = 0; j < localStageSize; j++) {
						if (inverse) {
							tempLUT[maxStageSum * 2 + 2 * (j + localStageSum)] = (float)cos(-j * double_PI / localStageSize / 2);
							tempLUT[maxStageSum * 2 + 2 * (j + localStageSum) + 1] = (float)sin(-j * double_PI / localStageSize / 2);
						}
						else {
							tempLUT[maxStageSum * 2 + 2 * (j + localStageSum)] = (float)cos(j * double_PI / localStageSize / 2);
							tempLUT[maxStageSum * 2 + 2 * (j + localStageSum) + 1] = (float)sin(j * double_PI / localStageSize / 2);
						}
					}
					localStageSum += localStageSize;
					localStageSize *= axis->specializationConstants.stageRadix[i];
				}
				localStageSize = 1;
				localStageSum = 0;
				for (uint32_t i = 0; i < axis->specializationConstants.numStages; i++) {
					for (uint32_t j = 0; j < localStageSize; j++) {
						if (inverse) {
							tempLUT[2 * maxStageSum * 2 + 2 * (j + localStageSum)] = (float)cos(-j * double_PI / localStageSize / 4);
							tempLUT[2 * maxStageSum * 2 + 2 * (j + localStageSum) + 1] = (float)sin(-j * double_PI / localStageSize / 4);
						}
						else {
							tempLUT[2 * maxStageSum * 2 + 2 * (j + localStageSum)] = (float)cos(j * double_PI / localStageSize / 4);
							tempLUT[2 * maxStageSum * 2 + 2 * (j + localStageSum) + 1] = (float)sin(j * double_PI / localStageSize / 4);
						}
					}
					localStageSum += localStageSize;
					localStageSize *= axis->specializationConstants.stageRadix[i];
				}
				if (axis_upload_id > 0)
					for (uint32_t i = 0; i < axis->specializationConstants.stageStartSize; i++) {
						for (uint32_t j = 0; j < axis->specializationConstants.fftDim; j++) {
							double angle = 2 * double_PI * ((i * j) / (double)(axis->specializationConstants.stageStartSize * axis->specializationConstants.fftDim));
							if (inverse) {
								tempLUT[3 * maxStageSum * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize)] = (float)cos(-angle);
								tempLUT[3 * maxStageSum * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize) + 1] = (float)sin(-angle);
							}
							else {
								tempLUT[3 * maxStageSum * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize)] = (float)cos(angle);
								tempLUT[3 * maxStageSum * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize) + 1] = (float)sin(angle);
							}
						}
					}
				allocateFFTBuffer(app, &axis->bufferLUT, &axis->bufferLUTDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, axis->bufferLUTSize);
				transferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
				free(tempLUT);
			}
		}
		//axis->groupedBatch = ((axis_upload_id>0)&&(axis->groupedBatch > axis->specializationConstants.stageStartSize)) ? axis->specializationConstants.stageStartSize : axis->groupedBatch;
		//configure strides
		//perform r2c
		axis->specializationConstants.inputStride[0] = 1;
		axis->specializationConstants.inputStride[3] = (app->configuration.bufferStride[0] / 2 + 1) * app->configuration.bufferStride[1] * app->configuration.bufferStride[2];

		if (axis_id == 1)
		{

			//don't transpose 0-1
			axis->specializationConstants.inputStride[1] = app->configuration.bufferStride[1];
			axis->specializationConstants.inputStride[2] = (app->configuration.bufferStride[0] / 2 + 1) * app->configuration.bufferStride[1];
			axis->specializationConstants.inputStride[3] = (app->configuration.bufferStride[0] / 2 + 1) * app->configuration.bufferStride[1] * app->configuration.bufferStride[2];
		}
		if (axis_id == 2)
		{

			//don't transpose 0-1, don't transpose 1-2
			axis->specializationConstants.inputStride[1] = (app->configuration.bufferStride[0] / 2 + 1) * app->configuration.bufferStride[1];
			axis->specializationConstants.inputStride[2] = app->configuration.bufferStride[1];

		}

		axis->specializationConstants.inputStride[4] = axis->specializationConstants.inputStride[3] * app->configuration.coordinateFeatures;
		axis->specializationConstants.outputStride[0] = axis->specializationConstants.inputStride[0];
		axis->specializationConstants.outputStride[1] = axis->specializationConstants.inputStride[1];
		axis->specializationConstants.outputStride[2] = axis->specializationConstants.inputStride[2];
		axis->specializationConstants.outputStride[3] = axis->specializationConstants.inputStride[3];
		axis->specializationConstants.outputStride[4] = axis->specializationConstants.inputStride[4];

		/*axis->specializationConstants.inputStride[3] = (app->configuration.coordinateFeatures == 1) ? 0 : axis->specializationConstants.inputStride[3];
		axis->specializationConstants.outputStride[3] = (app->configuration.coordinateFeatures == 1) ? 0 : axis->specializationConstants.outputStride[3];

		axis->specializationConstants.inputStride[4] = ((app->configuration.numberBatches == 1) && (app->configuration.numberKernels == 1)) ? 0 : axis->specializationConstants.inputStride[3] * app->configuration.coordinateFeatures;
		axis->specializationConstants.outputStride[4] = ((app->configuration.numberBatches == 1) && (app->configuration.numberKernels == 1)) ? 0 : axis->specializationConstants.outputStride[3] * app->configuration.coordinateFeatures;
		*/
		axis->specializationConstants.inputOffset = app->configuration.bufferStride[0] * app->configuration.bufferStride[1] / 2;
		axis->specializationConstants.outputOffset = app->configuration.bufferStride[0] * app->configuration.bufferStride[1] / 2;

		VkDescriptorPoolSize descriptorPoolSize = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
		uint32_t storageComplexSize;
		if (app->configuration.doublePrecision)
			storageComplexSize = (2 * sizeof(double));
		else
			if (app->configuration.halfPrecision)
				storageComplexSize = (2 * 2);
			else
				storageComplexSize = (2 * sizeof(float));
		uint32_t initPageSize = 0;
		for (uint32_t i = 0; i < app->configuration.bufferNum; i++) {
			initPageSize += app->configuration.bufferSize[i];
		}
		if (app->configuration.performConvolution) {
			uint32_t initPageSizeKernel = 0;
			for (uint32_t i = 0; i < app->configuration.kernelNum; i++) {
				initPageSizeKernel += app->configuration.kernelSize[i];
			}
			if (initPageSizeKernel > initPageSize) initPageSize = initPageSizeKernel;
		}
		if (axis_id == 1) {
			if ((axis->specializationConstants.inputStride[1] * storageComplexSize > app->configuration.devicePageSize * 1024) && (app->configuration.devicePageSize > 0)) {
				initPageSize = app->configuration.localPageSize * 1024;
			}
		}
		if (axis_id == 2) {
			if ((axis->specializationConstants.inputStride[1] * app->configuration.bufferStride[2] * storageComplexSize > app->configuration.devicePageSize * 1024) && (app->configuration.devicePageSize > 0)) {
				initPageSize = app->configuration.localPageSize * 1024;
			}
		}
		uint32_t locPageSize = initPageSize;
		uint64_t totalSize = 0;
		for (uint32_t i = 0; i < app->configuration.bufferNum; i++) {
			totalSize += app->configuration.bufferSize[i];
			if (app->configuration.bufferSize[i] < locPageSize) locPageSize = app->configuration.bufferSize[i];
		}
		axis->specializationConstants.inputBufferBlockSize = locPageSize / storageComplexSize;
		axis->specializationConstants.inputBufferBlockNum = (uint32_t)ceil(totalSize / (double)(axis->specializationConstants.inputBufferBlockSize * storageComplexSize));
		//if (axis->specializationConstants.inputBufferBlockNum == 1) axis->specializationConstants.inputBufferBlockSize = totalSize / storageComplexSize;
		locPageSize = initPageSize;
		totalSize = 0;
		for (uint32_t i = 0; i < app->configuration.bufferNum; i++) {
			totalSize += app->configuration.bufferSize[i];
			if (app->configuration.bufferSize[i] < locPageSize) locPageSize = app->configuration.bufferSize[i];
		}
		axis->specializationConstants.outputBufferBlockSize = locPageSize / storageComplexSize;
		axis->specializationConstants.outputBufferBlockNum = (uint32_t)ceil(totalSize / (double)(axis->specializationConstants.outputBufferBlockSize * storageComplexSize));
		//if (axis->specializationConstants.outputBufferBlockNum == 1) axis->specializationConstants.outputBufferBlockSize = totalSize / storageComplexSize;

		if (app->configuration.performConvolution) {
			totalSize = 0;
			locPageSize = initPageSize;
			for (uint32_t i = 0; i < app->configuration.kernelNum; i++) {
				totalSize += app->configuration.kernelSize[i];
				if (app->configuration.kernelSize[i] < locPageSize) locPageSize = app->configuration.kernelSize[i];
			}
			axis->specializationConstants.kernelBlockSize = locPageSize / storageComplexSize;
			axis->specializationConstants.kernelBlockNum = (uint32_t)ceil(totalSize / (double)(axis->specializationConstants.kernelBlockSize * storageComplexSize));
			//if (axis->specializationConstants.kernelBlockNum == 1) axis->specializationConstants.kernelBlockSize = totalSize / storageComplexSize;
		}
		else {
			axis->specializationConstants.kernelBlockSize = 0;
			axis->specializationConstants.kernelBlockNum = 0;
		}

		uint32_t numBindings = 2;
		uint32_t numBuffersBound[4] = { axis->specializationConstants.inputBufferBlockNum , axis->specializationConstants.outputBufferBlockNum, 0, 0 };
		descriptorPoolSize.descriptorCount = axis->specializationConstants.inputBufferBlockNum + axis->specializationConstants.outputBufferBlockNum;
		if ((axis_id == 0) && (axis_upload_id == 0) && (app->configuration.FFTdim == 1) && (app->configuration.performConvolution)) {
			numBuffersBound[numBindings] = axis->specializationConstants.kernelBlockNum;
			descriptorPoolSize.descriptorCount += axis->specializationConstants.kernelBlockNum;
			numBindings++;
		}
		if ((axis_id == 1) && (axis_upload_id == 0) && (app->configuration.FFTdim == 2) && (app->configuration.performConvolution)) {
			numBuffersBound[numBindings] = axis->specializationConstants.kernelBlockNum;
			descriptorPoolSize.descriptorCount += axis->specializationConstants.kernelBlockNum;
			numBindings++;
		}
		if ((axis_id == 2) && (axis_upload_id == 0) && (app->configuration.FFTdim == 3) && (app->configuration.performConvolution)) {
			numBuffersBound[numBindings] = axis->specializationConstants.kernelBlockNum;
			descriptorPoolSize.descriptorCount += axis->specializationConstants.kernelBlockNum;
			numBindings++;
		}
		if (app->configuration.useLUT) {
			numBuffersBound[numBindings] = 1;
			descriptorPoolSize.descriptorCount++;
			numBindings++;
		}
		VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
		descriptorPoolCreateInfo.poolSizeCount = 1;
		descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize;
		descriptorPoolCreateInfo.maxSets = 1;
		vkCreateDescriptorPool(app->configuration.device[0], &descriptorPoolCreateInfo, NULL, &axis->descriptorPool);

		const VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		VkDescriptorSetLayoutBinding* descriptorSetLayoutBindings;
		descriptorSetLayoutBindings = (VkDescriptorSetLayoutBinding*)malloc(numBindings * sizeof(VkDescriptorSetLayoutBinding));
		for (uint32_t i = 0; i < numBindings; ++i) {
			descriptorSetLayoutBindings[i].binding = i;
			descriptorSetLayoutBindings[i].descriptorType = descriptorType;
			descriptorSetLayoutBindings[i].descriptorCount = numBuffersBound[i];
			descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		}

		VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
		descriptorSetLayoutCreateInfo.bindingCount = numBindings;
		descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;

		vkCreateDescriptorSetLayout(app->configuration.device[0], &descriptorSetLayoutCreateInfo, NULL, &axis->descriptorSetLayout);
		free(descriptorSetLayoutBindings);
		VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
		descriptorSetAllocateInfo.descriptorPool = axis->descriptorPool;
		descriptorSetAllocateInfo.descriptorSetCount = 1;
		descriptorSetAllocateInfo.pSetLayouts = &axis->descriptorSetLayout;
		vkAllocateDescriptorSets(app->configuration.device[0], &descriptorSetAllocateInfo, &axis->descriptorSet);
		for (uint32_t i = 0; i < numBindings; ++i) {
			for (uint32_t j = 0; j < numBuffersBound[i]; ++j) {
				VkDescriptorBufferInfo descriptorBufferInfo = { 0 };

				if (i == 0) {
					uint32_t bufferId = 0;
					uint32_t offset = j;
					for (uint32_t l = 0; l < app->configuration.bufferNum; ++l) {
						if (offset >= (uint32_t)ceil(app->configuration.bufferSize[l] / (double)(axis->specializationConstants.inputBufferBlockSize * storageComplexSize))) {
							bufferId++;
							offset -= (uint32_t)ceil(app->configuration.bufferSize[l] / (double)(axis->specializationConstants.inputBufferBlockSize * storageComplexSize));
						}
						else {
							l = app->configuration.bufferNum;
						}

					}

					descriptorBufferInfo.buffer = app->configuration.buffer[bufferId];
					descriptorBufferInfo.range = (axis->specializationConstants.inputBufferBlockSize * storageComplexSize);
					descriptorBufferInfo.offset = offset * (axis->specializationConstants.inputBufferBlockSize * storageComplexSize);
				}
				if (i == 1) {
					uint32_t bufferId = 0;
					uint32_t offset = j;
					for (uint32_t l = 0; l < app->configuration.bufferNum; ++l) {
						if (offset >= (uint32_t)ceil(app->configuration.bufferSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize * storageComplexSize))) {
							bufferId++;
							offset -= (uint32_t)ceil(app->configuration.bufferSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize * storageComplexSize));
						}
						else {
							l = app->configuration.bufferNum;
						}

					}

					descriptorBufferInfo.buffer = app->configuration.buffer[bufferId];
					descriptorBufferInfo.range = (axis->specializationConstants.outputBufferBlockSize * storageComplexSize);
					descriptorBufferInfo.offset = offset * (axis->specializationConstants.outputBufferBlockSize * storageComplexSize);
				}
				if ((i == 2) && (app->configuration.performConvolution)) {
					uint32_t bufferId = 0;
					uint32_t offset = j;
					for (uint32_t l = 0; l < app->configuration.kernelNum; ++l) {
						if (offset >= (uint32_t)ceil(app->configuration.kernelSize[l] / (double)(axis->specializationConstants.kernelBlockSize * storageComplexSize))) {
							bufferId++;
							offset -= (uint32_t)ceil(app->configuration.kernelSize[l] / (double)(axis->specializationConstants.kernelBlockSize * storageComplexSize));
						}
						else {
							l = app->configuration.bufferNum;
						}

					}
					descriptorBufferInfo.buffer = app->configuration.kernel[bufferId];
					descriptorBufferInfo.range = (axis->specializationConstants.kernelBlockSize * storageComplexSize);
					descriptorBufferInfo.offset = offset * (axis->specializationConstants.kernelBlockSize * storageComplexSize);
				}
				if ((i == numBindings - 1) && (app->configuration.useLUT)) {
					descriptorBufferInfo.buffer = axis->bufferLUT;
					descriptorBufferInfo.offset = 0;
					descriptorBufferInfo.range = axis->bufferLUTSize;
				}
				VkWriteDescriptorSet writeDescriptorSet = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
				writeDescriptorSet.dstSet = axis->descriptorSet;
				writeDescriptorSet.dstBinding = i;
				writeDescriptorSet.dstArrayElement = j;
				writeDescriptorSet.descriptorType = descriptorType;
				writeDescriptorSet.descriptorCount = 1;
				writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
				vkUpdateDescriptorSets(app->configuration.device[0], 1, &writeDescriptorSet, 0, NULL);
			}
		}

		{
			VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
			pipelineLayoutCreateInfo.setLayoutCount = 1;
			pipelineLayoutCreateInfo.pSetLayouts = &axis->descriptorSetLayout;

			VkPushConstantRange pushConstantRange = { VK_SHADER_STAGE_COMPUTE_BIT };
			pushConstantRange.offset = 0;
			pushConstantRange.size = sizeof(VkFFTPushConstantsLayout);
			// Push constant ranges are part of the pipeline layout
			pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
			pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;


			vkCreatePipelineLayout(app->configuration.device[0], &pipelineLayoutCreateInfo, NULL, &axis->pipelineLayout);
			uint32_t maxThreadNum = maxSequenceLengthSharedMemory / axis->specializationConstants.min_registers_per_thread;
			if (axis_id == 1) {
				if (axis_upload_id == 0) {
					axis->axisBlock[0] = (axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread > 1) ? axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread : 1;
					if (axis->axisBlock[0] > maxThreadNum) axis->axisBlock[0] = maxThreadNum;
					axis->axisBlock[1] = 1;
					axis->axisBlock[2] = 1;
					axis->axisBlock[3] = axis->specializationConstants.fftDim;
				}
				else {
					axis->axisBlock[1] = (axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread > 1) ? axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread : 1;

					axis->axisBlock[0] = (axis->specializationConstants.stageStartSize > axis->groupedBatch) ? axis->groupedBatch : axis->specializationConstants.stageStartSize;

					axis->axisBlock[2] = 1;
					axis->axisBlock[3] = axis->specializationConstants.fftDim;
				}
			}
			if (axis_id == 2) {
				axis->axisBlock[1] = (axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread > 1) ? axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread : 1;

				axis->axisBlock[0] = (app->configuration.size[1] > axis->groupedBatch) ? axis->groupedBatch : app->configuration.size[1];
				/*if (axis->axisBlock[0] * axis->axisBlock[1] < 64)
					if (app->configuration.size[1] > 64 / axis->axisBlock[1])
						axis->axisBlock[0] = 64 / axis->axisBlock[1];
					else
						axis->axisBlock[0] = app->configuration.size[0];*/
				axis->axisBlock[2] = 1;
				axis->axisBlock[3] = axis->specializationConstants.fftDim;
			}
			uint32_t tempSize[3] = { app->configuration.size[0], app->configuration.size[1], app->configuration.size[2] };
			if (axis_id == 1) {
				if (axis_upload_id == 0)
					tempSize[0] = app->configuration.size[1] / axis->specializationConstants.fftDim;
				else
					tempSize[0] = app->configuration.size[1] / axis->specializationConstants.fftDim / axis->axisBlock[0];

				tempSize[1] = 1;
				tempSize[2] = app->configuration.size[2];
				//if (app->configuration.performZeropadding[2]) tempSize[2] = ceil(tempSize[2] / 2.0);

				if (tempSize[0] > app->configuration.maxComputeWorkGroupCount[0]) axis->specializationConstants.performWorkGroupShift[0] = 1;
				else  axis->specializationConstants.performWorkGroupShift[0] = 0;
				if (tempSize[1] > app->configuration.maxComputeWorkGroupCount[1]) axis->specializationConstants.performWorkGroupShift[1] = 1;
				else  axis->specializationConstants.performWorkGroupShift[1] = 0;
				if (tempSize[2] > app->configuration.maxComputeWorkGroupCount[2]) axis->specializationConstants.performWorkGroupShift[2] = 1;
				else  axis->specializationConstants.performWorkGroupShift[2] = 0;

			}
			if (axis_id == 2) {
				tempSize[0] = app->configuration.size[1] / axis->axisBlock[0] * app->configuration.size[2] / axis->specializationConstants.fftDim;
				tempSize[1] = 1;
				tempSize[2] = 1;

				if (tempSize[0] > app->configuration.maxComputeWorkGroupCount[0]) axis->specializationConstants.performWorkGroupShift[0] = 1;
				else  axis->specializationConstants.performWorkGroupShift[0] = 0;
				if (tempSize[1] > app->configuration.maxComputeWorkGroupCount[1]) axis->specializationConstants.performWorkGroupShift[1] = 1;
				else  axis->specializationConstants.performWorkGroupShift[1] = 0;
				if (tempSize[2] > app->configuration.maxComputeWorkGroupCount[2]) axis->specializationConstants.performWorkGroupShift[2] = 1;
				else  axis->specializationConstants.performWorkGroupShift[2] = 0;

			}
			axis->specializationConstants.localSize[0] = axis->axisBlock[0];
			axis->specializationConstants.localSize[1] = axis->axisBlock[1];
			axis->specializationConstants.localSize[2] = axis->axisBlock[2];
			//specializationInfo.pData = &axis->specializationConstants;
			VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };

			VkComputePipelineCreateInfo computePipelineCreateInfo = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };


			pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			uint32_t registerBoost = (FFTPlan->numAxisUploads[axis_id] > 1) ? app->configuration.registerBoost4Step : app->configuration.registerBoost;

			axis->specializationConstants.numCoordinates = (app->configuration.matrixConvolution > 1) ? 1 : app->configuration.coordinateFeatures;
			axis->specializationConstants.matrixConvolution = app->configuration.matrixConvolution;
			if ((app->configuration.FFTdim == 1) && (app->configuration.size[1] == 1) && (app->configuration.numberBatches > 1) && (!app->configuration.performConvolution) && (app->configuration.coordinateFeatures == 1)) {
				app->configuration.size[1] = app->configuration.numberBatches;
				app->configuration.numberBatches = 1;
			}
			axis->specializationConstants.numBatches = app->configuration.numberBatches;
			axis->specializationConstants.numKernels = app->configuration.numberKernels;
			axis->specializationConstants.sharedMemSize = app->configuration.sharedMemorySize;
			axis->specializationConstants.sharedMemSizePow2 = app->configuration.sharedMemorySizePow2;
			axis->specializationConstants.normalize = 1;
			axis->specializationConstants.size[0] = app->configuration.size[0];
			axis->specializationConstants.size[1] = app->configuration.size[1];
			axis->specializationConstants.size[2] = app->configuration.size[2];
			axis->specializationConstants.axis_id = axis_id;
			axis->specializationConstants.axis_upload_id = axis_upload_id;
			if (convolutionInverseStage) {
				for (uint32_t i = 0; i < 3; i++) {
					axis->specializationConstants.performZeropaddingInput[i] = app->configuration.performZeropaddingOutput[i]; // don't read if input is zeropadded (0 - off, 1 - on)
					axis->specializationConstants.performZeropaddingOutput[i] = app->configuration.performZeropaddingInput[i]; // don't write if output is zeropadded (0 - off, 1 - on)
					axis->specializationConstants.fft_zeropad_left_read[i] = app->configuration.fft_zeropad_left_write[i];
					axis->specializationConstants.fft_zeropad_left_write[i] = app->configuration.fft_zeropad_left_read[i];
					axis->specializationConstants.fft_zeropad_right_read[i] = app->configuration.fft_zeropad_right_write[i];
					axis->specializationConstants.fft_zeropad_right_write[i] = app->configuration.fft_zeropad_right_read[i];
				}
			}
			else {
				if ((app->configuration.FFTdim - 1 == axis_id) && (axis_upload_id == 0) && (app->configuration.performConvolution)) {
					for (uint32_t i = 0; i < 3; i++) {
						axis->specializationConstants.performZeropaddingInput[i] = app->configuration.performZeropaddingInput[i];
						axis->specializationConstants.performZeropaddingOutput[i] = app->configuration.performZeropaddingInput[i]; // don't write if output is zeropadded (0 - off, 1 - on)
						axis->specializationConstants.fft_zeropad_left_read[i] = app->configuration.fft_zeropad_left_read[i];
						axis->specializationConstants.fft_zeropad_right_read[i] = app->configuration.fft_zeropad_right_read[i];
						axis->specializationConstants.fft_zeropad_left_write[i] = app->configuration.fft_zeropad_left_read[i];
						axis->specializationConstants.fft_zeropad_right_write[i] = app->configuration.fft_zeropad_right_read[i];
					}
				}
				else {
					for (uint32_t i = 0; i < 3; i++) {
						axis->specializationConstants.performZeropaddingInput[i] = app->configuration.performZeropaddingInput[i]; // don't read if input is zeropadded (0 - off, 1 - on)
						axis->specializationConstants.performZeropaddingOutput[i] = app->configuration.performZeropaddingOutput[i]; // don't write if output is zeropadded (0 - off, 1 - on)
						axis->specializationConstants.fft_zeropad_left_read[i] = app->configuration.fft_zeropad_left_read[i];
						axis->specializationConstants.fft_zeropad_left_write[i] = app->configuration.fft_zeropad_left_write[i];
						axis->specializationConstants.fft_zeropad_right_read[i] = app->configuration.fft_zeropad_right_read[i];
						axis->specializationConstants.fft_zeropad_right_write[i] = app->configuration.fft_zeropad_right_write[i];
					}
				}
			}
			if (inverse) {
				axis->specializationConstants.zeropad[0] = (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) ? axis->specializationConstants.performZeropaddingInput[axis_id] : 0;
				axis->specializationConstants.zeropad[1] = (axis_upload_id == 0) ? axis->specializationConstants.performZeropaddingOutput[axis_id] : 0;
			}
			else {
				axis->specializationConstants.zeropad[0] = (axis_upload_id == 0) ? axis->specializationConstants.performZeropaddingInput[axis_id] : 0;
				axis->specializationConstants.zeropad[1] = (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) ? axis->specializationConstants.performZeropaddingOutput[axis_id] : 0;
			}
			if ((app->configuration.FFTdim - 1 == axis_id) && (axis_upload_id == 0) && (app->configuration.performConvolution)) {
				axis->specializationConstants.convolutionStep = 1;
			}
			else
				axis->specializationConstants.convolutionStep = 0;
			char floatTypeMemory[10];
			char floatType[10];
			axis->specializationConstants.unroll = 1;
			axis->specializationConstants.LUT = app->configuration.useLUT;
			if (app->configuration.doublePrecision) {
				sprintf(floatType, "double");
				sprintf(floatTypeMemory, "double");
				//axis->specializationConstants.unroll = 1;
			}
			else {
				//axis->specializationConstants.unroll = 0;
				if (app->configuration.halfPrecision) {
					sprintf(floatType, "float");
					sprintf(floatTypeMemory, "half");
				}
				else {
					sprintf(floatType, "float");
					sprintf(floatTypeMemory, "float");
				}
			}
			char uintType[10] = "uint";
			uint32_t LUT = app->configuration.useLUT;
			uint32_t type;
			if ((axis_id - 1 == 0) && (axis_upload_id == 0)) type = 0;
			if (axis_id - 1 != 0) type = 1;
			if ((axis_id - 1 == 0) && (axis_upload_id > 0)) type = 2;
			axis->specializationConstants.cacheShuffle = 0;// ((!app->configuration.doublePrecision) && ((type == 0) || (type == 5) || (type == 6))) ? 1 : 0;
			//if ((axis->specializationConstants.fftDim == 2 * maxSequenceLengthSharedMemory) && (app->configuration.registerBoost >= 2)) type = 3;
			//if ((axis->specializationConstants.fftDim == 4 * maxSequenceLengthSharedMemory) && (app->configuration.registerBoost >= 4)) type = 4;
			char* code0 = (char*)malloc(sizeof(char) * 200000);
			shaderGenVkFFT(code0, axis->specializationConstants, floatType, floatTypeMemory, uintType, type);

			const glslang_resource_t default_resource = {
				/* .MaxLights = */ 32,
				/* .MaxClipPlanes = */ 6,
				/* .MaxTextureUnits = */ 32,
				/* .MaxTextureCoords = */ 32,
				/* .MaxVertexAttribs = */ 64,
				/* .MaxVertexUniformComponents = */ 4096,
				/* .MaxVaryingFloats = */ 64,
				/* .MaxVertexTextureImageUnits = */ 32,
				/* .MaxCombinedTextureImageUnits = */ 80,
				/* .MaxTextureImageUnits = */ 32,
				/* .MaxFragmentUniformComponents = */ 4096,
				/* .MaxDrawBuffers = */ 32,
				/* .MaxVertexUniformVectors = */ 128,
				/* .MaxVaryingVectors = */ 8,
				/* .MaxFragmentUniformVectors = */ 16,
				/* .MaxVertexOutputVectors = */ 16,
				/* .MaxFragmentInputVectors = */ 15,
				/* .MinProgramTexelOffset = */ -8,
				/* .MaxProgramTexelOffset = */ 7,
				/* .MaxClipDistances = */ 8,
				/* .MaxComputeWorkGroupCountX = */ 65535,
				/* .MaxComputeWorkGroupCountY = */ 65535,
				/* .MaxComputeWorkGroupCountZ = */ 65535,
				/* .MaxComputeWorkGroupSizeX = */ 1024,
				/* .MaxComputeWorkGroupSizeY = */ 1024,
				/* .MaxComputeWorkGroupSizeZ = */ 64,
				/* .MaxComputeUniformComponents = */ 1024,
				/* .MaxComputeTextureImageUnits = */ 16,
				/* .MaxComputeImageUniforms = */ 8,
				/* .MaxComputeAtomicCounters = */ 8,
				/* .MaxComputeAtomicCounterBuffers = */ 1,
				/* .MaxVaryingComponents = */ 60,
				/* .MaxVertexOutputComponents = */ 64,
				/* .MaxGeometryInputComponents = */ 64,
				/* .MaxGeometryOutputComponents = */ 128,
				/* .MaxFragmentInputComponents = */ 128,
				/* .MaxImageUnits = */ 8,
				/* .MaxCombinedImageUnitsAndFragmentOutputs = */ 8,
				/* .MaxCombinedShaderOutputResources = */ 8,
				/* .MaxImageSamples = */ 0,
				/* .MaxVertexImageUniforms = */ 0,
				/* .MaxTessControlImageUniforms = */ 0,
				/* .MaxTessEvaluationImageUniforms = */ 0,
				/* .MaxGeometryImageUniforms = */ 0,
				/* .MaxFragmentImageUniforms = */ 8,
				/* .MaxCombinedImageUniforms = */ 8,
				/* .MaxGeometryTextureImageUnits = */ 16,
				/* .MaxGeometryOutputVertices = */ 256,
				/* .MaxGeometryTotalOutputComponents = */ 1024,
				/* .MaxGeometryUniformComponents = */ 1024,
				/* .MaxGeometryVaryingComponents = */ 64,
				/* .MaxTessControlInputComponents = */ 128,
				/* .MaxTessControlOutputComponents = */ 128,
				/* .MaxTessControlTextureImageUnits = */ 16,
				/* .MaxTessControlUniformComponents = */ 1024,
				/* .MaxTessControlTotalOutputComponents = */ 4096,
				/* .MaxTessEvaluationInputComponents = */ 128,
				/* .MaxTessEvaluationOutputComponents = */ 128,
				/* .MaxTessEvaluationTextureImageUnits = */ 16,
				/* .MaxTessEvaluationUniformComponents = */ 1024,
				/* .MaxTessPatchComponents = */ 120,
				/* .MaxPatchVertices = */ 32,
				/* .MaxTessGenLevel = */ 64,
				/* .MaxViewports = */ 16,
				/* .MaxVertexAtomicCounters = */ 0,
				/* .MaxTessControlAtomicCounters = */ 0,
				/* .MaxTessEvaluationAtomicCounters = */ 0,
				/* .MaxGeometryAtomicCounters = */ 0,
				/* .MaxFragmentAtomicCounters = */ 8,
				/* .MaxCombinedAtomicCounters = */ 8,
				/* .MaxAtomicCounterBindings = */ 1,
				/* .MaxVertexAtomicCounterBuffers = */ 0,
				/* .MaxTessControlAtomicCounterBuffers = */ 0,
				/* .MaxTessEvaluationAtomicCounterBuffers = */ 0,
				/* .MaxGeometryAtomicCounterBuffers = */ 0,
				/* .MaxFragmentAtomicCounterBuffers = */ 1,
				/* .MaxCombinedAtomicCounterBuffers = */ 1,
				/* .MaxAtomicCounterBufferSize = */ 16384,
				/* .MaxTransformFeedbackBuffers = */ 4,
				/* .MaxTransformFeedbackInterleavedComponents = */ 64,
				/* .MaxCullDistances = */ 8,
				/* .MaxCombinedClipAndCullDistances = */ 8,
				/* .MaxSamples = */ 4,
				/* .maxMeshOutputVerticesNV = */ 256,
				/* .maxMeshOutputPrimitivesNV = */ 512,
				/* .maxMeshWorkGroupSizeX_NV = */ 32,
				/* .maxMeshWorkGroupSizeY_NV = */ 1,
				/* .maxMeshWorkGroupSizeZ_NV = */ 1,
				/* .maxTaskWorkGroupSizeX_NV = */ 32,
				/* .maxTaskWorkGroupSizeY_NV = */ 1,
				/* .maxTaskWorkGroupSizeZ_NV = */ 1,
				/* .maxMeshViewCountNV = */ 4,
				/* .maxDualSourceDrawBuffersEXT = */ 1,

				/* .limits = */ {
					/* .nonInductiveForLoops = */ 1,
					/* .whileLoops = */ 1,
					/* .doWhileLoops = */ 1,
					/* .generalUniformIndexing = */ 1,
					/* .generalAttributeMatrixVectorIndexing = */ 1,
					/* .generalVaryingIndexing = */ 1,
					/* .generalSamplerIndexing = */ 1,
					/* .generalVariableIndexing = */ 1,
					/* .generalConstantMatrixVectorIndexing = */ 1,
				} };
			glslang_target_client_version_t client_version = (app->configuration.halfPrecision) ? GLSLANG_TARGET_VULKAN_1_1 : GLSLANG_TARGET_VULKAN_1_0;
			glslang_target_language_version_t target_language_version = (app->configuration.halfPrecision) ? GLSLANG_TARGET_SPV_1_3 : GLSLANG_TARGET_SPV_1_0;
			const glslang_input_t input =
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
				&default_resource,
			};
			glslang_optimization_level_t optimization = GLSLANG_OPT_NONE;
			glslang_shader_t* shader = glslang_shader_create(&input);
			const char* err;
			//printf("%s\n", code0);
			if (!glslang_shader_preprocess(shader, &input))
			{
				err = glslang_shader_get_info_log(shader);
				printf("%s\n", code0);
				printf("%s\nVkFFT shader type: %d\n", err, type);
				glslang_shader_delete(shader);
				free(code0);
				return VK_ERROR_INITIALIZATION_FAILED;

			}

			if (!glslang_shader_parse(shader, &input))
			{
				err = glslang_shader_get_info_log(shader);
				printf("%s\n", code0);
				printf("%s\nVkFFT shader type: %d\n", err, type);
				glslang_shader_delete(shader);
				free(code0);
				return VK_ERROR_INITIALIZATION_FAILED;

			}
			glslang_program_t* program = glslang_program_create();
			glslang_program_add_shader(program, shader);
			if (!glslang_program_link(program, GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT))
			{
				err = glslang_program_get_info_log(program);
				printf("%s\n", code0);
				printf("%s\nVkFFT shader type: %d\n", err, type);
				glslang_shader_delete(shader);
				free(code0);
				return VK_ERROR_INITIALIZATION_FAILED;

			}

			glslang_program_SPIRV_generate(program, input.stage);

			if (glslang_program_SPIRV_get_messages(program))
			{
				printf("%s", glslang_program_SPIRV_get_messages(program));
			}

			glslang_shader_delete(shader);
			free(code0);
			//shaderGenVkFFT(app->configuration.code0, axis->specializationConstants, floatType, floatTypeMemory, uintType, type);
			//printf("%s\n", app->configuration.code0);
			/*shaderc_compiler_t compiler = shaderc_compiler_initialize();
			shaderc_compile_options_t options = shaderc_compile_options_initialize();
			shaderc_compile_options_set_optimization_level(options, shaderc_optimization_level_performance);*/
			//shaderc_compilation_result_t result = shaderc_compile_into_spv(app->configuration.compiler[0], app->configuration.code0, strlen(app->configuration.code0), shaderc_glsl_default_compute_shader, "file", "main", app->configuration.options[0]);
			//memset(app->configuration.code0, 0, strlen(app->configuration.code0));
			//const char* err = shaderc_result_get_error_message(result);
			//uint32_t* code = (uint32_t*)shaderc_result_get_bytes(result);
			//uint32_t filelength = shaderc_result_get_length(result);
			//if (strcmp(err, "")) printf("%s\n", err);


			VkShaderModuleCreateInfo createInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
			createInfo.pCode = glslang_program_SPIRV_get_ptr(program);
			createInfo.codeSize = glslang_program_SPIRV_get_size(program) * sizeof(uint32_t);
			vkCreateShaderModule(app->configuration.device[0], &createInfo, NULL, &pipelineShaderStageCreateInfo.module);
			//shaderc_result_release(result);
			//free(code0);
			/*shaderc_compile_options_release(options);
			shaderc_compiler_release(compiler);*/

			pipelineShaderStageCreateInfo.pName = "main";
			pipelineShaderStageCreateInfo.pSpecializationInfo = 0;// &specializationInfo;
			computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
			computePipelineCreateInfo.layout = axis->pipelineLayout;



			vkCreateComputePipelines(app->configuration.device[0], VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, &axis->pipeline);
			vkDestroyShaderModule(app->configuration.device[0], pipelineShaderStageCreateInfo.module, NULL);
			glslang_program_delete(program);
		}
		return VK_SUCCESS;

	}
	static inline VkResult VkFFTPlanAxis(VkFFTApplication* app, VkFFTPlan* FFTPlan, uint32_t axis_id, uint32_t axis_upload_id, VkBool32 inverse, uint32_t convolutionInverseStage) {
		//get radix stages
		VkFFTAxis* axis = &FFTPlan->axes[axis_id][axis_upload_id];
		uint32_t complexSize;
		if (app->configuration.doublePrecision)
			complexSize = (2 * sizeof(double));
		else
			if (app->configuration.halfPrecision)
				complexSize = (2 * sizeof(float));
			else
				complexSize = (2 * sizeof(float));
		axis->specializationConstants.complexSize = complexSize;
		axis->specializationConstants.supportAxis = 0;
		axis->specializationConstants.symmetricKernel = app->configuration.symmetricKernel;
		uint32_t maxSequenceLengthSharedMemory = app->configuration.sharedMemorySize / complexSize;
		uint32_t maxSequenceLengthSharedMemoryPow2 = app->configuration.sharedMemorySizePow2 / complexSize;
		uint32_t maxSingleSizeStrided = (app->configuration.coalescedMemory > complexSize) ? app->configuration.sharedMemorySize / (app->configuration.coalescedMemory) : app->configuration.sharedMemorySize / complexSize;
		uint32_t maxSingleSizeStridedPow2 = (app->configuration.coalescedMemory > complexSize) ? app->configuration.sharedMemorySizePow2 / (app->configuration.coalescedMemory) : app->configuration.sharedMemorySizePow2 / complexSize;

		/*
		uint32_t logSize = log2(FFTPlan->axisSplit[axis_id][axis_upload_id]);

		uint32_t stage8 = logSize / 3;
		uint32_t stage4 = 0;
		uint32_t stage2 = 0;
		if (logSize % 3 == 2)
			stage4 = 1;
		if (logSize % 3 == 1)
			stage2 = 1;
		uint32_t totNumStages = stage8 + stage4 + stage2;

		axis->specializationConstants.numStages = stage8;
		axis->specializationConstants.fftDim = pow(8, stage8);
		for (uint32_t i = 0; i < stage8; i++) {
			axis->specializationConstants.stageRadix[i] = 8;
		}
		uint32_t reorganizeLastRadixRegisterBoost = 1;
		if ((axis->specializationConstants.fftDim > maxSequenceLengthSharedMemory) && (logSize > stage8 * 3)) {
			reorganizeLastRadixRegisterBoost = axis->specializationConstants.fftDim / maxSequenceLengthSharedMemory;
			axis->specializationConstants.fftDim = maxSequenceLengthSharedMemory;
			axis->specializationConstants.stageRadix[stage8 - 1] /= reorganizeLastRadixRegisterBoost;
		}
		if (stage4 == 1) {
			axis->specializationConstants.numStages++;
			axis->specializationConstants.stageRadix[stage8] = 4 * reorganizeLastRadixRegisterBoost;
			axis->specializationConstants.fftDim *= 4 * reorganizeLastRadixRegisterBoost;
		}
		if (stage2 == 1) {
			axis->specializationConstants.numStages++;
			axis->specializationConstants.stageRadix[stage8] = 2 * reorganizeLastRadixRegisterBoost;
			axis->specializationConstants.fftDim *= 2 * reorganizeLastRadixRegisterBoost;
		}*/

		axis->specializationConstants.stageStartSize = 1;
		for (uint32_t i = 0; i < axis_upload_id; i++)
			axis->specializationConstants.stageStartSize *= FFTPlan->axisSplit[axis_id][i];


		axis->specializationConstants.firstStageStartSize = app->configuration.size[axis_id] / FFTPlan->axisSplit[axis_id][FFTPlan->numAxisUploads[axis_id] - 1];


		if (axis_id == 0) {
			//configure radix stages
			axis->specializationConstants.fft_dim_x = axis->specializationConstants.stageStartSize;
		}
		else {
			if (app->configuration.performR2C)
				axis->specializationConstants.fft_dim_x = app->configuration.size[0] / 2;
			else
				axis->specializationConstants.fft_dim_x = app->configuration.size[0];
		}
		axis->specializationConstants.performR2C = app->configuration.performR2C;
		axis->specializationConstants.reorderFourStep = (FFTPlan->numAxisUploads[axis_id] > 1) ? app->configuration.reorderFourStep : 0;
		//axis->groupedBatch = (4096 / axis->specializationConstants.fftDim >= app->configuration.coalescedMemory / 8) ? 4096 / axis->specializationConstants.fftDim : app->configuration.coalescedMemory / 8;
		uint32_t passID = FFTPlan->numAxisUploads[axis_id] - 1 - axis_upload_id;
		axis->specializationConstants.fft_dim_full = app->configuration.size[axis_id];
		uint32_t maxBatchCoalesced = app->configuration.coalescedMemory / complexSize;
		axis->groupedBatch = maxBatchCoalesced;
		/*if ((app->configuration.size[0] < 4096) && (app->configuration.size[1] < 512) && (app->configuration.size[2] == 1)) {
			if (app->configuration.sharedMemorySize / axis->specializationConstants.fftDim >= app->configuration.coalescedMemory) {
				if (1024 / axis->specializationConstants.fftDim < maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim) {
					if (1024 / axis->specializationConstants.fftDim > axis->groupedBatch)
						axis->groupedBatch = 1024 / axis->specializationConstants.fftDim;
					else
						axis->groupedBatch = maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim;
				}
			}
		}
		else {
			axis->groupedBatch = (app->configuration.sharedMemorySize / axis->specializationConstants.fftDim >= app->configuration.coalescedMemory) ? maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim : axis->groupedBatch;
		}*/
		//if (axis->groupedBatch * ceil(axis->specializationConstants.fftDim / 8.0) < app->configuration.warpSize) axis->groupedBatch = app->configuration.warpSize / ceil(axis->specializationConstants.fftDim / 8.0);
		//axis->groupedBatch = (app->configuration.sharedMemorySize / axis->specializationConstants.fftDim >= app->configuration.coalescedMemory) ? maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim : axis->groupedBatch;
		if (((FFTPlan->numAxisUploads[axis_id] == 1) && (axis_id == 0)) || ((axis_id == 0) && (!app->configuration.reorderFourStep) && (axis_upload_id == 0))) {
			axis->groupedBatch = (maxSequenceLengthSharedMemoryPow2 / axis->specializationConstants.fftDim > axis->groupedBatch) ? maxSequenceLengthSharedMemoryPow2 / axis->specializationConstants.fftDim : axis->groupedBatch;
		}
		else {
			axis->groupedBatch = (maxSingleSizeStridedPow2 / axis->specializationConstants.fftDim > 1) ? maxSingleSizeStridedPow2 / axis->specializationConstants.fftDim * axis->groupedBatch : axis->groupedBatch;
		}
		//axis->groupedBatch = 8;
		//shared memory bank conflict resolve
		if ((FFTPlan->numAxisUploads[axis_id] == 3) && (axis_upload_id == 0) && (axis->specializationConstants.fftDim < maxSequenceLengthSharedMemory / (2 * complexSize))) {
			axis->groupedBatch = ceil(axis->groupedBatch / 2.0);
		}
		if (axis->groupedBatch < maxBatchCoalesced) axis->groupedBatch = maxBatchCoalesced;
		axis->groupedBatch = (axis->groupedBatch / maxBatchCoalesced) * maxBatchCoalesced;
		//half bandiwdth technique
		if (!((axis_id == 0) && (FFTPlan->numAxisUploads[axis_id] == 1)) && !((axis_id == 0) && (axis_upload_id == 0) && (!app->configuration.reorderFourStep)) && (axis->specializationConstants.fftDim > maxSingleSizeStrided)) {
			axis->groupedBatch = ceil(axis->groupedBatch / 2.0);
		}

		if (app->configuration.halfThreads)
			axis->groupedBatch = ceil(axis->groupedBatch / 2.0);

		//allocate LUT 
		if (app->configuration.useLUT) {
			double double_PI = 3.1415926535897932384626433832795;
			uint32_t dimMult = 1;
			uint32_t maxStageSum = 0;
			for (uint32_t i = 0; i < axis->specializationConstants.numStages; i++) {
				maxStageSum += dimMult;
				dimMult *= axis->specializationConstants.stageRadix[i];
			}
			axis->specializationConstants.maxStageSumLUT = maxStageSum;
			if (app->configuration.doublePrecision) {
				if (axis_upload_id > 0)
					axis->bufferLUTSize = (3 * maxStageSum + axis->specializationConstants.stageStartSize * axis->specializationConstants.fftDim) * 2 * sizeof(double);
				else
					axis->bufferLUTSize = (3 * maxStageSum) * 2 * sizeof(double);
				double* tempLUT = (double*)malloc(axis->bufferLUTSize);
				uint32_t localStageSize = 1;
				uint32_t localStageSum = 0;
				for (uint32_t i = 0; i < axis->specializationConstants.numStages; i++) {
					for (uint32_t j = 0; j < localStageSize; j++) {
						if (inverse) {
							tempLUT[2 * (j + localStageSum)] = cos(-j * double_PI / localStageSize);
							tempLUT[2 * (j + localStageSum) + 1] = sin(-j * double_PI / localStageSize);
						}
						else {
							tempLUT[2 * (j + localStageSum)] = cos(j * double_PI / localStageSize);
							tempLUT[2 * (j + localStageSum) + 1] = sin(j * double_PI / localStageSize);
						}
					}
					localStageSum += localStageSize;
					localStageSize *= axis->specializationConstants.stageRadix[i];
				}
				localStageSize = 1;
				localStageSum = 0;
				for (uint32_t i = 0; i < axis->specializationConstants.numStages; i++) {
					for (uint32_t j = 0; j < localStageSize; j++) {
						if (inverse) {
							tempLUT[maxStageSum * 2 + 2 * (j + localStageSum)] = cos(-j * double_PI / localStageSize / 2);
							tempLUT[maxStageSum * 2 + 2 * (j + localStageSum) + 1] = sin(-j * double_PI / localStageSize / 2);
						}
						else {
							tempLUT[maxStageSum * 2 + 2 * (j + localStageSum)] = cos(j * double_PI / localStageSize / 2);
							tempLUT[maxStageSum * 2 + 2 * (j + localStageSum) + 1] = sin(j * double_PI / localStageSize / 2);
						}
					}
					localStageSum += localStageSize;
					localStageSize *= axis->specializationConstants.stageRadix[i];
				}
				localStageSize = 1;
				localStageSum = 0;
				for (uint32_t i = 0; i < axis->specializationConstants.numStages; i++) {
					for (uint32_t j = 0; j < localStageSize; j++) {
						if (inverse) {
							tempLUT[2 * maxStageSum * 2 + 2 * (j + localStageSum)] = cos(-j * double_PI / localStageSize / 4);
							tempLUT[2 * maxStageSum * 2 + 2 * (j + localStageSum) + 1] = sin(-j * double_PI / localStageSize / 4);
						}
						else {
							tempLUT[2 * maxStageSum * 2 + 2 * (j + localStageSum)] = cos(j * double_PI / localStageSize / 4);
							tempLUT[2 * maxStageSum * 2 + 2 * (j + localStageSum) + 1] = sin(j * double_PI / localStageSize / 4);
						}
					}
					localStageSum += localStageSize;
					localStageSize *= axis->specializationConstants.stageRadix[i];
				}
				if (axis_upload_id > 0)
					for (uint32_t i = 0; i < axis->specializationConstants.stageStartSize; i++) {
						for (uint32_t j = 0; j < axis->specializationConstants.fftDim; j++) {
							double angle = 2 * double_PI * ((i * j) / (double)(axis->specializationConstants.stageStartSize * axis->specializationConstants.fftDim));
							if (inverse) {
								tempLUT[3 * maxStageSum * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize)] = cos(-angle);
								tempLUT[3 * maxStageSum * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize) + 1] = sin(-angle);
							}
							else {
								tempLUT[3 * maxStageSum * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize)] = cos(angle);
								tempLUT[3 * maxStageSum * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize) + 1] = sin(angle);
							}
						}
					}
				allocateFFTBuffer(app, &axis->bufferLUT, &axis->bufferLUTDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, axis->bufferLUTSize);
				transferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
				free(tempLUT);
			}
			else {
				uint32_t dimMult = 1;
				uint32_t maxStageSum = 0;
				for (uint32_t i = 0; i < axis->specializationConstants.numStages; i++) {
					maxStageSum += dimMult;
					dimMult *= axis->specializationConstants.stageRadix[i];
				}
				axis->specializationConstants.maxStageSumLUT = maxStageSum;
				if (axis_upload_id > 0)
					axis->bufferLUTSize = (3 * maxStageSum + axis->specializationConstants.stageStartSize * axis->specializationConstants.fftDim) * 2 * sizeof(float);
				else
					axis->bufferLUTSize = (3 * maxStageSum) * 2 * sizeof(float);
				float* tempLUT = (float*)malloc(axis->bufferLUTSize);
				uint32_t localStageSize = 1;
				uint32_t localStageSum = 0;
				for (uint32_t i = 0; i < axis->specializationConstants.numStages; i++) {
					for (uint32_t j = 0; j < localStageSize; j++) {
						if (inverse) {
							tempLUT[2 * (j + localStageSum)] = (float)cos(-j * double_PI / localStageSize);
							tempLUT[2 * (j + localStageSum) + 1] = (float)sin(-j * double_PI / localStageSize);
						}
						else {
							tempLUT[2 * (j + localStageSum)] = (float)cos(j * double_PI / localStageSize);
							tempLUT[2 * (j + localStageSum) + 1] = (float)sin(j * double_PI / localStageSize);
						}
					}
					localStageSum += localStageSize;
					localStageSize *= axis->specializationConstants.stageRadix[i];
				}
				localStageSize = 1;
				localStageSum = 0;
				for (uint32_t i = 0; i < axis->specializationConstants.numStages; i++) {
					for (uint32_t j = 0; j < localStageSize; j++) {
						if (inverse) {
							tempLUT[maxStageSum * 2 + 2 * (j + localStageSum)] = (float)cos(-j * double_PI / localStageSize / 2);
							tempLUT[maxStageSum * 2 + 2 * (j + localStageSum) + 1] = (float)sin(-j * double_PI / localStageSize / 2);
						}
						else {
							tempLUT[maxStageSum * 2 + 2 * (j + localStageSum)] = (float)cos(j * double_PI / localStageSize / 2);
							tempLUT[maxStageSum * 2 + 2 * (j + localStageSum) + 1] = (float)sin(j * double_PI / localStageSize / 2);
						}
					}
					localStageSum += localStageSize;
					localStageSize *= axis->specializationConstants.stageRadix[i];
				}
				localStageSize = 1;
				localStageSum = 0;
				for (uint32_t i = 0; i < axis->specializationConstants.numStages; i++) {
					for (uint32_t j = 0; j < localStageSize; j++) {
						if (inverse) {
							tempLUT[2 * maxStageSum * 2 + 2 * (j + localStageSum)] = (float)cos(-j * double_PI / localStageSize / 4);
							tempLUT[2 * maxStageSum * 2 + 2 * (j + localStageSum) + 1] = (float)sin(-j * double_PI / localStageSize / 4);
						}
						else {
							tempLUT[2 * maxStageSum * 2 + 2 * (j + localStageSum)] = (float)cos(j * double_PI / localStageSize / 4);
							tempLUT[2 * maxStageSum * 2 + 2 * (j + localStageSum) + 1] = (float)sin(j * double_PI / localStageSize / 4);
						}
					}
					localStageSum += localStageSize;
					localStageSize *= axis->specializationConstants.stageRadix[i];
				}
				if (axis_upload_id > 0)
					for (uint32_t i = 0; i < axis->specializationConstants.stageStartSize; i++) {
						for (uint32_t j = 0; j < axis->specializationConstants.fftDim; j++) {
							double angle = 2 * double_PI * ((i * j) / (double)(axis->specializationConstants.stageStartSize * axis->specializationConstants.fftDim));
							if (inverse) {
								tempLUT[3 * maxStageSum * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize)] = (float)cos(-angle);
								tempLUT[3 * maxStageSum * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize) + 1] = (float)sin(-angle);
							}
							else {
								tempLUT[3 * maxStageSum * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize)] = (float)cos(angle);
								tempLUT[3 * maxStageSum * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize) + 1] = (float)sin(angle);
							}
						}
					}
				allocateFFTBuffer(app, &axis->bufferLUT, &axis->bufferLUTDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, axis->bufferLUTSize);
				transferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
				free(tempLUT);
			}
		}
		//axis->groupedBatch = ((axis_upload_id > 0) && (axis->groupedBatch > axis->specializationConstants.stageStartSize)) ? axis->specializationConstants.stageStartSize : axis->groupedBatch;
		/*if (4096 / app->configuration.size[1] > app->configuration.coalescedMemory / 16) {
			app->configuration.performTranspose[0] = 0;
			FFTPlan->groupedBatch = 4096 / app->configuration.size[1];
		}
		else {
			app->configuration.performTranspose[0] = 1;
		}

		if (4096 / app->configuration.size[2] > app->configuration.coalescedMemory / 16) {
			app->configuration.performTranspose[1] = 0;
			FFTPlan->axes[2].groupedBatch = 4096 / app->configuration.size[2];
		}
		else {
			app->configuration.performTranspose[1] = 1;
		}*/
		//configure strides
		uint32_t* axisStride = axis->specializationConstants.inputStride;
		uint32_t* usedStride = app->configuration.bufferStride;
		if ((!inverse) && (axis_id == 0) && (axis_upload_id == 0) && (app->configuration.isInputFormatted)) usedStride = app->configuration.inputBufferStride;
		if ((inverse) && (axis_id == app->configuration.FFTdim - 1) && (axis_upload_id == app->localFFTPlan.numAxisUploads[axis_id] - 1) && (app->configuration.isInputFormatted)) usedStride = app->configuration.inputBufferStride;
		if (app->configuration.performR2C)
		{
			//perform r2c
			axisStride[0] = 1;
			axisStride[3] = (usedStride[0] / 2 + 1) * usedStride[1] * usedStride[2];

			if (axis_id == 0) {
				axisStride[1] = usedStride[0];
				axisStride[2] = (usedStride[0] / 2 + 1) * usedStride[1];
			}
			if (axis_id == 1)
			{
				axisStride[1] = usedStride[0] / 2;
				axisStride[2] = (usedStride[0] / 2 + 1) * usedStride[1];
			}
			if (axis_id == 2)
			{
				axisStride[1] = (usedStride[0] / 2 + 1) * usedStride[1];
				axisStride[2] = usedStride[0] / 2;
			}
		}
		else {
			//don't perform r2c
			axisStride[0] = 1;
			axisStride[3] = usedStride[0] * usedStride[1] * usedStride[2];

			if (axis_id == 0) {
				axisStride[1] = usedStride[0];
				axisStride[2] = usedStride[0] * usedStride[1];
			}
			if (axis_id == 1)
			{
				axisStride[1] = usedStride[0];
				axisStride[2] = usedStride[0] * usedStride[1];
			}
			if (axis_id == 2)
			{
				axisStride[1] = usedStride[0] * usedStride[1];
				axisStride[2] = usedStride[0];
			}

			/*if (axis_id == 0) {
				if ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (app->configuration.isInputFormatted) && (!inverse)) {
					if (app->configuration.performZeropadding[0])
						axis->specializationConstants.inputStride[1] = app->configuration.inputBufferStride[0] / 2;

					if (app->configuration.performZeropadding[1])
						axis->specializationConstants.inputStride[2] = axis->specializationConstants.inputStride[1] * app->configuration.inputBufferStride[1] / 2;
					else
						axis->specializationConstants.inputStride[2] = axis->specializationConstants.inputStride[1] * app->configuration.inputBufferStride[1];

					if (app->configuration.performZeropadding[2])
						axis->specializationConstants.inputStride[3] = axis->specializationConstants.inputStride[2] * app->configuration.inputBufferStride[2] / 2;
					else
						axis->specializationConstants.inputStride[3] = axis->specializationConstants.inputStride[2] * app->configuration.inputBufferStride[2];
				}
				if ((axis_upload_id == 0) && (app->configuration.isOutputFormatted) && ((inverse) || ((app->configuration.performConvolution) && (app->configuration.FFTdim == 1)))) {
					if (app->configuration.performZeropadding[0])
						axis->specializationConstants.outputStride[1] = app->configuration.outputBufferStride[0] / 2;

					if (app->configuration.performZeropadding[1])
						axis->specializationConstants.outputStride[2] = axis->specializationConstants.outputStride[1] * app->configuration.outputBufferStride[1] / 2;
					else
						axis->specializationConstants.outputStride[2] = axis->specializationConstants.outputStride[1] * app->configuration.outputBufferStride[1];

					if (app->configuration.performZeropadding[2])
						axis->specializationConstants.outputStride[3] = axis->specializationConstants.outputStride[2] * app->configuration.outputBufferStride[2] / 2;
					else
						axis->specializationConstants.outputStride[3] = axis->specializationConstants.outputStride[2] * app->configuration.outputBufferStride[2];
				}
			}*/
		}
		axisStride[4] = axisStride[3] * app->configuration.coordinateFeatures;
		axisStride = axis->specializationConstants.outputStride;
		usedStride = app->configuration.bufferStride;
		if ((!inverse) && (axis_id == app->configuration.FFTdim - 1) && (axis_upload_id == app->localFFTPlan.numAxisUploads[axis_id] - 1) && (app->configuration.isOutputFormatted)) usedStride = app->configuration.outputBufferStride;
		if ((inverse) && (axis_id == 0) && (axis_upload_id == 0) && (app->configuration.isOutputFormatted)) usedStride = app->configuration.outputBufferStride;
		if (app->configuration.performR2C)
		{
			//perform r2c
			axisStride[0] = 1;
			axisStride[3] = (usedStride[0] / 2 + 1) * usedStride[1] * usedStride[2];

			if (axis_id == 0) {
				axisStride[1] = usedStride[0];
				axisStride[2] = (usedStride[0] / 2 + 1) * usedStride[1];
			}
			if (axis_id == 1)
			{
				axisStride[1] = usedStride[0] / 2;
				axisStride[2] = (usedStride[0] / 2 + 1) * usedStride[1];
			}
			if (axis_id == 2)
			{
				axisStride[1] = (usedStride[0] / 2 + 1) * usedStride[1];
				axisStride[2] = usedStride[0] / 2;
			}
		}
		else {
			//don't perform r2c
			axisStride[0] = 1;
			axisStride[3] = usedStride[0] * usedStride[1] * usedStride[2];

			if (axis_id == 0) {
				axisStride[1] = usedStride[0];
				axisStride[2] = usedStride[0] * usedStride[1];
			}
			if (axis_id == 1)
			{
				axisStride[1] = usedStride[0];
				axisStride[2] = usedStride[0] * usedStride[1];
			}
			if (axis_id == 2)
			{
				axisStride[1] = usedStride[0] * usedStride[1];
				axisStride[2] = usedStride[0];
			}

			/*if (axis_id == 0) {
				if ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (app->configuration.isInputFormatted) && (!inverse)) {
					if (app->configuration.performZeropadding[0])
						axis->specializationConstants.inputStride[1] = app->configuration.inputBufferStride[0] / 2;

					if (app->configuration.performZeropadding[1])
						axis->specializationConstants.inputStride[2] = axis->specializationConstants.inputStride[1] * app->configuration.inputBufferStride[1] / 2;
					else
						axis->specializationConstants.inputStride[2] = axis->specializationConstants.inputStride[1] * app->configuration.inputBufferStride[1];

					if (app->configuration.performZeropadding[2])
						axis->specializationConstants.inputStride[3] = axis->specializationConstants.inputStride[2] * app->configuration.inputBufferStride[2] / 2;
					else
						axis->specializationConstants.inputStride[3] = axis->specializationConstants.inputStride[2] * app->configuration.inputBufferStride[2];
				}
				if ((axis_upload_id == 0) && (app->configuration.isOutputFormatted) && ((inverse) || ((app->configuration.performConvolution) && (app->configuration.FFTdim == 1)))) {
					if (app->configuration.performZeropadding[0])
						axis->specializationConstants.outputStride[1] = app->configuration.outputBufferStride[0] / 2;

					if (app->configuration.performZeropadding[1])
						axis->specializationConstants.outputStride[2] = axis->specializationConstants.outputStride[1] * app->configuration.outputBufferStride[1] / 2;
					else
						axis->specializationConstants.outputStride[2] = axis->specializationConstants.outputStride[1] * app->configuration.outputBufferStride[1];

					if (app->configuration.performZeropadding[2])
						axis->specializationConstants.outputStride[3] = axis->specializationConstants.outputStride[2] * app->configuration.outputBufferStride[2] / 2;
					else
						axis->specializationConstants.outputStride[3] = axis->specializationConstants.outputStride[2] * app->configuration.outputBufferStride[2];
				}
			}*/
		}
		axisStride[4] = axisStride[3] * app->configuration.coordinateFeatures;

		/*axis->specializationConstants.inputStride[3] = (app->configuration.coordinateFeatures == 1) ? 0 : axis->specializationConstants.inputStride[3];
		axis->specializationConstants.outputStride[3] = (app->configuration.coordinateFeatures == 1) ? 0 : axis->specializationConstants.outputStride[3];

		axis->specializationConstants.inputStride[4] = ((app->configuration.numberBatches == 1) && (app->configuration.numberKernels == 1)) ? 0 : axis->specializationConstants.inputStride[3] * app->configuration.coordinateFeatures;
		axis->specializationConstants.outputStride[4] = ((app->configuration.numberBatches == 1) && (app->configuration.numberKernels == 1)) ? 0 : axis->specializationConstants.outputStride[3] * app->configuration.coordinateFeatures;
		*/
		axis->specializationConstants.inverse = inverse;


		axis->specializationConstants.inputOffset = 0;
		axis->specializationConstants.outputOffset = 0;

		VkDescriptorPoolSize descriptorPoolSize = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
		uint32_t storageComplexSize;
		if (app->configuration.doublePrecision)
			storageComplexSize = (2 * sizeof(double));
		else
			if (app->configuration.halfPrecision)
				storageComplexSize = (2 * 2);
			else
				storageComplexSize = (2 * sizeof(float));

		uint32_t initPageSize = 0;
		for (uint32_t i = 0; i < app->configuration.bufferNum; i++) {
			initPageSize += app->configuration.bufferSize[i];
		}
		if (app->configuration.performConvolution) {
			uint32_t initPageSizeKernel = 0;
			for (uint32_t i = 0; i < app->configuration.kernelNum; i++) {
				initPageSizeKernel += app->configuration.kernelSize[i];
			}
			if (initPageSizeKernel > initPageSize) initPageSize = initPageSizeKernel;
		}
		if (axis_id == 0) {
			if ((!((!app->configuration.reorderFourStep) && (axis_upload_id == 0))) && (axis->specializationConstants.inputStride[1] * storageComplexSize > app->configuration.devicePageSize * 1024) && (app->configuration.devicePageSize > 0)) {
				initPageSize = app->configuration.localPageSize * 1024;
			}
		}
		if (axis_id == 1) {
			if ((axis->specializationConstants.inputStride[1] * app->configuration.bufferStride[1] * storageComplexSize > app->configuration.devicePageSize * 1024) && (app->configuration.devicePageSize > 0)) {
				initPageSize = app->configuration.localPageSize * 1024;
			}
		}
		if (axis_id == 2) {
			if ((axis->specializationConstants.inputStride[1] * app->configuration.bufferStride[2] * storageComplexSize > app->configuration.devicePageSize * 1024) && (app->configuration.devicePageSize > 0)) {
				initPageSize = app->configuration.localPageSize * 1024;
			}
		}

		if ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (app->configuration.isInputFormatted) && (
			((axis_id == 0) && (!inverse))
			|| ((axis_id == app->configuration.FFTdim - 1) && (inverse) && (!app->configuration.performConvolution)))
			) {
			uint64_t totalSize = 0;
			uint32_t locPageSize = initPageSize;
			for (uint32_t i = 0; i < app->configuration.inputBufferNum; i++) {
				totalSize += app->configuration.inputBufferSize[i];
				if (app->configuration.inputBufferSize[i] < locPageSize) locPageSize = app->configuration.inputBufferSize[i];
			}
			axis->specializationConstants.inputBufferBlockSize = locPageSize / storageComplexSize;
			axis->specializationConstants.inputBufferBlockNum = (uint32_t)ceil(totalSize / (double)(axis->specializationConstants.inputBufferBlockSize * storageComplexSize));
			//if (axis->specializationConstants.inputBufferBlockNum == 1) axis->specializationConstants.inputBufferBlockSize = totalSize / storageComplexSize;

		}
		else {
			if ((axis_upload_id == 0) && (app->configuration.numberKernels > 1) && (inverse) && (!app->configuration.performConvolution)) {
				uint64_t totalSize = 0;
				uint32_t locPageSize = initPageSize;
				for (uint32_t i = 0; i < app->configuration.outputBufferNum; i++) {
					totalSize += app->configuration.outputBufferSize[i];
					if (app->configuration.outputBufferSize[i] < locPageSize) locPageSize = app->configuration.outputBufferSize[i];
				}

				axis->specializationConstants.inputBufferBlockSize = locPageSize / storageComplexSize;
				axis->specializationConstants.inputBufferBlockNum = (uint32_t)ceil(totalSize / (double)(axis->specializationConstants.outputBufferBlockSize * storageComplexSize));
				//if (axis->specializationConstants.inputBufferBlockNum == 1) axis->specializationConstants.outputBufferBlockSize = totalSize / storageComplexSize;

			}
			else {
				uint64_t totalSize = 0;
				uint32_t locPageSize = initPageSize;
				if ((FFTPlan->axes[axis_id]->specializationConstants.reorderFourStep == 1) && (FFTPlan->numAxisUploads[axis_id] > 1))
					if (axis_upload_id > 0) {
						for (uint32_t i = 0; i < app->configuration.bufferNum; i++) {
							totalSize += app->configuration.bufferSize[i];
							if (app->configuration.bufferSize[i] < locPageSize) locPageSize = app->configuration.bufferSize[i];

						}
					}
					else {
						for (uint32_t i = 0; i < app->configuration.tempBufferNum; i++) {
							totalSize += app->configuration.tempBufferSize[i];
							if (app->configuration.tempBufferSize[i] < locPageSize) locPageSize = app->configuration.tempBufferSize[i];

						}
					}
				else {
					for (uint32_t i = 0; i < app->configuration.bufferNum; i++) {
						totalSize += app->configuration.bufferSize[i];
						if (app->configuration.bufferSize[i] < locPageSize) locPageSize = app->configuration.bufferSize[i];

					}
				}

				axis->specializationConstants.inputBufferBlockSize = locPageSize / storageComplexSize;
				axis->specializationConstants.inputBufferBlockNum = (uint32_t)ceil(totalSize / (double)(axis->specializationConstants.inputBufferBlockSize * storageComplexSize));
				//if (axis->specializationConstants.inputBufferBlockNum == 1) axis->specializationConstants.inputBufferBlockSize = totalSize / storageComplexSize;

			}
		}
		if ((axis_upload_id == 0) && (app->configuration.isOutputFormatted && (
			((axis_id == 0) && (inverse))
			|| ((axis_id == app->configuration.FFTdim - 1) && (!inverse) && (!app->configuration.performConvolution))
			|| ((axis_id == 0) && (app->configuration.performConvolution) && (app->configuration.FFTdim == 1)))
			) ||
			((app->configuration.numberKernels > 1) && (
				(inverse)
				|| (axis_id == app->configuration.FFTdim - 1)))
			) {
			uint64_t totalSize = 0;
			uint32_t locPageSize = initPageSize;
			for (uint32_t i = 0; i < app->configuration.outputBufferNum; i++) {
				totalSize += app->configuration.outputBufferSize[i];
				if (app->configuration.outputBufferSize[i] < locPageSize) locPageSize = app->configuration.outputBufferSize[i];
			}

			axis->specializationConstants.outputBufferBlockSize = locPageSize / storageComplexSize;
			axis->specializationConstants.outputBufferBlockNum = (uint32_t)ceil(totalSize / (double)(axis->specializationConstants.outputBufferBlockSize * storageComplexSize));
			//if (axis->specializationConstants.outputBufferBlockNum == 1) axis->specializationConstants.outputBufferBlockSize = totalSize / storageComplexSize;

		}
		else {
			uint64_t totalSize = 0;
			uint32_t locPageSize = initPageSize;
			if ((FFTPlan->axes[axis_id]->specializationConstants.reorderFourStep == 1) && (FFTPlan->numAxisUploads[axis_id] > 1))
				if (axis_upload_id == 1) {
					for (uint32_t i = 0; i < app->configuration.bufferNum; i++) {
						totalSize += app->configuration.bufferSize[i];
						if (app->configuration.bufferSize[i] < locPageSize) locPageSize = app->configuration.bufferSize[i];
					}
				}
				else {
					for (uint32_t i = 0; i < app->configuration.tempBufferNum; i++) {
						totalSize += app->configuration.tempBufferSize[i];
						if (app->configuration.tempBufferSize[i] < locPageSize) locPageSize = app->configuration.tempBufferSize[i];
					}
				}
			else {
				for (uint32_t i = 0; i < app->configuration.bufferNum; i++) {
					totalSize += app->configuration.bufferSize[i];
					if (app->configuration.bufferSize[i] < locPageSize) locPageSize = app->configuration.bufferSize[i];
				}
			}
			axis->specializationConstants.outputBufferBlockSize = locPageSize / storageComplexSize;
			axis->specializationConstants.outputBufferBlockNum = (uint32_t)ceil(totalSize / (double)(axis->specializationConstants.outputBufferBlockSize * storageComplexSize));
			//if (axis->specializationConstants.outputBufferBlockNum == 1) axis->specializationConstants.outputBufferBlockSize = totalSize / storageComplexSize;

		}

		if (app->configuration.performConvolution) {
			uint64_t totalSize = 0;
			uint32_t locPageSize = initPageSize;
			for (uint32_t i = 0; i < app->configuration.kernelNum; i++) {
				totalSize += app->configuration.kernelSize[i];
				if (app->configuration.kernelSize[i] < locPageSize) locPageSize = app->configuration.kernelSize[i];
			}
			axis->specializationConstants.kernelBlockSize = locPageSize / storageComplexSize;
			axis->specializationConstants.kernelBlockNum = (uint32_t)ceil(totalSize / (double)(axis->specializationConstants.kernelBlockSize * storageComplexSize));
			//if (axis->specializationConstants.kernelBlockNum == 1) axis->specializationConstants.inputBufferBlockSize = totalSize / storageComplexSize;
		}
		else {
			axis->specializationConstants.kernelBlockSize = 0;
			axis->specializationConstants.kernelBlockNum = 0;
		}
		uint32_t numBindings = 2;
		uint32_t numBuffersBound[4] = { axis->specializationConstants.inputBufferBlockNum , axis->specializationConstants.outputBufferBlockNum, 0 , 0 };
		descriptorPoolSize.descriptorCount = axis->specializationConstants.inputBufferBlockNum + axis->specializationConstants.outputBufferBlockNum;

		if ((axis_id == 0) && (axis_upload_id == 0) && (app->configuration.FFTdim == 1) && (app->configuration.performConvolution)) {
			numBuffersBound[numBindings] = axis->specializationConstants.kernelBlockNum;
			descriptorPoolSize.descriptorCount += axis->specializationConstants.kernelBlockNum;
			numBindings++;
		}
		if ((axis_id == 1) && (axis_upload_id == 0) && (app->configuration.FFTdim == 2) && (app->configuration.performConvolution)) {
			numBuffersBound[numBindings] = axis->specializationConstants.kernelBlockNum;
			descriptorPoolSize.descriptorCount += axis->specializationConstants.kernelBlockNum;
			numBindings++;
		}
		if ((axis_id == 2) && (axis_upload_id == 0) && (app->configuration.FFTdim == 3) && (app->configuration.performConvolution)) {
			numBuffersBound[numBindings] = axis->specializationConstants.kernelBlockNum;
			descriptorPoolSize.descriptorCount += axis->specializationConstants.kernelBlockNum;
			numBindings++;
		}
		if (app->configuration.useLUT) {
			numBuffersBound[numBindings] = 1;
			descriptorPoolSize.descriptorCount++;
			numBindings++;
		}
		VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
		descriptorPoolCreateInfo.poolSizeCount = 1;
		descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize;
		descriptorPoolCreateInfo.maxSets = 1;
		vkCreateDescriptorPool(app->configuration.device[0], &descriptorPoolCreateInfo, NULL, &axis->descriptorPool);

		const VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		VkDescriptorSetLayoutBinding* descriptorSetLayoutBindings;
		descriptorSetLayoutBindings = (VkDescriptorSetLayoutBinding*)malloc(numBindings * sizeof(VkDescriptorSetLayoutBinding));
		for (uint32_t i = 0; i < numBindings; ++i) {
			descriptorSetLayoutBindings[i].binding = i;
			descriptorSetLayoutBindings[i].descriptorType = descriptorType;
			descriptorSetLayoutBindings[i].descriptorCount = numBuffersBound[i];
			descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		}

		VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
		descriptorSetLayoutCreateInfo.bindingCount = numBindings;
		descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;

		vkCreateDescriptorSetLayout(app->configuration.device[0], &descriptorSetLayoutCreateInfo, NULL, &axis->descriptorSetLayout);
		free(descriptorSetLayoutBindings);
		VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
		descriptorSetAllocateInfo.descriptorPool = axis->descriptorPool;
		descriptorSetAllocateInfo.descriptorSetCount = 1;
		descriptorSetAllocateInfo.pSetLayouts = &axis->descriptorSetLayout;
		vkAllocateDescriptorSets(app->configuration.device[0], &descriptorSetAllocateInfo, &axis->descriptorSet);
		for (uint32_t i = 0; i < numBindings; ++i) {
			for (uint32_t j = 0; j < numBuffersBound[i]; ++j) {
				VkDescriptorBufferInfo descriptorBufferInfo = { 0 };
				if (i == 0) {
					if ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (app->configuration.isInputFormatted) && (
						((axis_id == 0) && (!inverse))
						|| ((axis_id == app->configuration.FFTdim - 1) && (inverse) && (!app->configuration.performConvolution)))
						) {
						uint32_t bufferId = 0;
						uint32_t offset = j;
						for (uint32_t l = 0; l < app->configuration.inputBufferNum; ++l) {
							if (offset >= (uint32_t)ceil(app->configuration.inputBufferSize[l] / (double)(axis->specializationConstants.inputBufferBlockSize * storageComplexSize))) {
								bufferId++;
								offset -= (uint32_t)ceil(app->configuration.inputBufferSize[l] / (double)(axis->specializationConstants.inputBufferBlockSize * storageComplexSize));
							}
							else {
								l = app->configuration.inputBufferNum;
							}

						}

						descriptorBufferInfo.buffer = app->configuration.inputBuffer[bufferId];
						descriptorBufferInfo.range = (axis->specializationConstants.inputBufferBlockSize * storageComplexSize);
						descriptorBufferInfo.offset = offset * (axis->specializationConstants.inputBufferBlockSize * storageComplexSize);

					}
					else {
						if ((axis_upload_id == 0) && (app->configuration.numberKernels > 1) && (inverse) && (!app->configuration.performConvolution)) {
							uint32_t bufferId = 0;
							uint32_t offset = j;
							for (uint32_t l = 0; l < app->configuration.outputBufferNum; ++l) {
								if (offset >= (uint32_t)ceil(app->configuration.outputBufferSize[l] / (double)(axis->specializationConstants.inputBufferBlockSize * storageComplexSize))) {
									bufferId++;
									offset -= (uint32_t)ceil(app->configuration.outputBufferSize[l] / (double)(axis->specializationConstants.inputBufferBlockSize * storageComplexSize));
								}
								else {
									l = app->configuration.outputBufferNum;
								}

							}

							descriptorBufferInfo.buffer = app->configuration.outputBuffer[bufferId];
							descriptorBufferInfo.range = (axis->specializationConstants.inputBufferBlockSize * storageComplexSize);
							descriptorBufferInfo.offset = offset * (axis->specializationConstants.inputBufferBlockSize * storageComplexSize);
						}
						else {
							uint32_t bufferId = 0;
							uint32_t offset = j;
							if ((FFTPlan->axes[axis_id]->specializationConstants.reorderFourStep == 1) && (FFTPlan->numAxisUploads[axis_id] > 1))
								if (axis_upload_id > 0) {
									for (uint32_t l = 0; l < app->configuration.bufferNum; ++l) {
										if (offset >= (uint32_t)ceil(app->configuration.bufferSize[l] / (double)(axis->specializationConstants.inputBufferBlockSize * storageComplexSize))) {
											bufferId++;
											offset -= (uint32_t)ceil(app->configuration.bufferSize[l] / (double)(axis->specializationConstants.inputBufferBlockSize * storageComplexSize));
										}
										else {
											l = app->configuration.bufferNum;
										}

									}

									descriptorBufferInfo.buffer = app->configuration.buffer[bufferId];
								}
								else {
									for (uint32_t l = 0; l < app->configuration.tempBufferNum; ++l) {
										if (offset >= (uint32_t)ceil(app->configuration.tempBufferSize[l] / (double)(axis->specializationConstants.inputBufferBlockSize * storageComplexSize))) {
											bufferId++;
											offset -= (uint32_t)ceil(app->configuration.tempBufferSize[l] / (double)(axis->specializationConstants.inputBufferBlockSize * storageComplexSize));
										}
										else {
											l = app->configuration.tempBufferNum;
										}

									}

									descriptorBufferInfo.buffer = app->configuration.tempBuffer[bufferId];
								}
							else {
								for (uint32_t l = 0; l < app->configuration.bufferNum; ++l) {
									if (offset >= (uint32_t)ceil(app->configuration.bufferSize[l] / (double)(axis->specializationConstants.inputBufferBlockSize * storageComplexSize))) {
										bufferId++;
										offset -= (uint32_t)ceil(app->configuration.bufferSize[l] / (double)(axis->specializationConstants.inputBufferBlockSize * storageComplexSize));
									}
									else {
										l = app->configuration.bufferNum;
									}

								}

								descriptorBufferInfo.buffer = app->configuration.buffer[bufferId];
							}
							descriptorBufferInfo.range = (axis->specializationConstants.inputBufferBlockSize * storageComplexSize);
							descriptorBufferInfo.offset = offset * (axis->specializationConstants.inputBufferBlockSize * storageComplexSize);
						}
					}
					//descriptorBufferInfo.offset = 0;
				}
				if (i == 1) {
					if ((axis_upload_id == 0) && (app->configuration.isOutputFormatted && (
						((axis_id == 0) && (inverse))
						|| ((axis_id == app->configuration.FFTdim - 1) && (!inverse) && (!app->configuration.performConvolution))
						|| ((axis_id == 0) && (app->configuration.performConvolution) && (app->configuration.FFTdim == 1)))
						) ||
						((app->configuration.numberKernels > 1) && (
							(inverse)
							|| (axis_id == app->configuration.FFTdim - 1)))
						) {
						uint32_t bufferId = 0;
						uint32_t offset = j;

						for (uint32_t l = 0; l < app->configuration.outputBufferNum; ++l) {
							if (offset >= (uint32_t)ceil(app->configuration.outputBufferSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize * storageComplexSize))) {
								bufferId++;
								offset -= (uint32_t)ceil(app->configuration.outputBufferSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize * storageComplexSize));
							}
							else {
								l = app->configuration.outputBufferNum;
							}

						}
						descriptorBufferInfo.buffer = app->configuration.outputBuffer[bufferId];
						descriptorBufferInfo.range = (axis->specializationConstants.outputBufferBlockSize * storageComplexSize);
						descriptorBufferInfo.offset = offset * (axis->specializationConstants.outputBufferBlockSize * storageComplexSize);
					}
					else {
						uint32_t bufferId = 0;
						uint32_t offset = j;

						if ((FFTPlan->axes[axis_id]->specializationConstants.reorderFourStep == 1) && (FFTPlan->numAxisUploads[axis_id] > 1))
							if (axis_upload_id == 1) {
								for (uint32_t l = 0; l < app->configuration.tempBufferNum; ++l) {
									if (offset >= (uint32_t)ceil(app->configuration.tempBufferSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize * storageComplexSize))) {
										bufferId++;
										offset -= (uint32_t)ceil(app->configuration.tempBufferSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize * storageComplexSize));
									}
									else {
										l = app->configuration.tempBufferNum;
									}

								}
								descriptorBufferInfo.buffer = app->configuration.tempBuffer[bufferId];
							}
							else {
								for (uint32_t l = 0; l < app->configuration.bufferNum; ++l) {
									if (offset >= (uint32_t)ceil(app->configuration.bufferSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize * storageComplexSize))) {
										bufferId++;
										offset -= (uint32_t)ceil(app->configuration.bufferSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize * storageComplexSize));
									}
									else {
										l = app->configuration.bufferNum;
									}

								}
								descriptorBufferInfo.buffer = app->configuration.buffer[bufferId];
							}

						else {
							for (uint32_t l = 0; l < app->configuration.bufferNum; ++l) {
								if (offset >= (uint32_t)ceil(app->configuration.bufferSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize * storageComplexSize))) {
									bufferId++;
									offset -= (uint32_t)ceil(app->configuration.bufferSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize * storageComplexSize));
								}
								else {
									l = app->configuration.bufferNum;
								}

							}
							descriptorBufferInfo.buffer = app->configuration.buffer[bufferId];
						}

						descriptorBufferInfo.range = (axis->specializationConstants.outputBufferBlockSize * storageComplexSize);
						descriptorBufferInfo.offset = offset * (axis->specializationConstants.outputBufferBlockSize * storageComplexSize);
					}
					//descriptorBufferInfo.offset = 0;
				}
				if ((i == 2) && (app->configuration.performConvolution)) {
					uint32_t bufferId = 0;
					uint32_t offset = j;
					for (uint32_t l = 0; l < app->configuration.kernelNum; ++l) {
						if (offset >= (uint32_t)ceil(app->configuration.kernelSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize * storageComplexSize))) {
							bufferId++;
							offset -= (uint32_t)ceil(app->configuration.kernelSize[l] / (double)(axis->specializationConstants.outputBufferBlockSize * storageComplexSize));
						}
						else {
							l = app->configuration.kernelNum;
						}

					}
					descriptorBufferInfo.buffer = app->configuration.kernel[bufferId];
					descriptorBufferInfo.range = (axis->specializationConstants.kernelBlockSize * storageComplexSize);
					descriptorBufferInfo.offset = offset * (axis->specializationConstants.kernelBlockSize * storageComplexSize);
				}
				if ((i == numBindings - 1) && (app->configuration.useLUT)) {
					descriptorBufferInfo.buffer = axis->bufferLUT;
					descriptorBufferInfo.offset = 0;
					descriptorBufferInfo.range = axis->bufferLUTSize;
				}
				VkWriteDescriptorSet writeDescriptorSet = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
				writeDescriptorSet.dstSet = axis->descriptorSet;
				writeDescriptorSet.dstBinding = i;
				writeDescriptorSet.dstArrayElement = j;
				writeDescriptorSet.descriptorType = descriptorType;
				writeDescriptorSet.descriptorCount = 1;
				writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
				vkUpdateDescriptorSets(app->configuration.device[0], 1, &writeDescriptorSet, 0, NULL);
			}
		}
		{
			VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
			pipelineLayoutCreateInfo.setLayoutCount = 1;
			pipelineLayoutCreateInfo.pSetLayouts = &axis->descriptorSetLayout;

			VkPushConstantRange pushConstantRange = { VK_SHADER_STAGE_COMPUTE_BIT };
			pushConstantRange.offset = 0;
			pushConstantRange.size = sizeof(VkFFTPushConstantsLayout);
			// Push constant ranges are part of the pipeline layout
			pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
			pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;

			vkCreatePipelineLayout(app->configuration.device[0], &pipelineLayoutCreateInfo, NULL, &axis->pipelineLayout);
			uint32_t maxThreadNum = maxSequenceLengthSharedMemory / axis->specializationConstants.min_registers_per_thread;
			if (!inverse) {
				if (axis_id == 0) {

					if (axis_upload_id == 0) {
						axis->axisBlock[0] = (axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread > 1) ? axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread : 1;
						if (axis->axisBlock[0] > maxThreadNum) axis->axisBlock[0] = maxThreadNum;
						if (app->configuration.reorderFourStep && (FFTPlan->numAxisUploads[axis_id] > 1))
							axis->axisBlock[1] = axis->groupedBatch;
						else {
							axis->axisBlock[1] = (axis->axisBlock[0] < app->configuration.warpSize) ? app->configuration.warpSize / axis->axisBlock[0] : 1;
							if (app->configuration.size[1] < axis->axisBlock[1]) axis->axisBlock[1] = app->configuration.size[1];
						}
						axis->axisBlock[2] = 1;
						axis->axisBlock[3] = axis->specializationConstants.fftDim;
					}
					else {
						axis->axisBlock[1] = (axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread > 1) ? axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread : 1;

						axis->axisBlock[0] = (axis->specializationConstants.stageStartSize > axis->groupedBatch) ? axis->groupedBatch : axis->specializationConstants.stageStartSize;

						axis->axisBlock[2] = 1;
						axis->axisBlock[3] = axis->specializationConstants.fftDim;
					}

				}
				if (axis_id == 1) {

					axis->axisBlock[1] = (axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread > 1) ? axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread : 1;

					if (app->configuration.performR2C) {
						if (axis_upload_id == 0) {
							FFTPlan->numSupportAxisUploads[0] = FFTPlan->numAxisUploads[1];
							for (uint32_t i = 0; i < FFTPlan->numSupportAxisUploads[0]; i++) {
								FFTPlan->supportAxes[axis_id - 1][i].specializationConstants.registers_per_thread = FFTPlan->axes[axis_id][i].specializationConstants.registers_per_thread;
								FFTPlan->supportAxes[axis_id - 1][i].specializationConstants.min_registers_per_thread = FFTPlan->axes[axis_id][i].specializationConstants.min_registers_per_thread;
								FFTPlan->supportAxes[axis_id - 1][i].specializationConstants.numStages = FFTPlan->axes[axis_id][i].specializationConstants.numStages;
								FFTPlan->supportAxes[axis_id - 1][i].specializationConstants.fftDim = FFTPlan->axisSplit[axis_id][i];
								for (uint32_t j = 0; j < FFTPlan->supportAxes[axis_id - 1][i].specializationConstants.numStages; j++)
									FFTPlan->supportAxes[axis_id - 1][i].specializationConstants.stageRadix[j] = FFTPlan->axes[axis_id][i].specializationConstants.stageRadix[j];
								VkFFTPlanSupportAxis(app, FFTPlan, 1, i, inverse, convolutionInverseStage);
							}
						}
						axis->axisBlock[0] = (app->configuration.size[0] / 2 > axis->groupedBatch) ? axis->groupedBatch : app->configuration.size[0] / 2;
						/*if (axis->axisBlock[0] * axis->axisBlock[1] < 64)
							if (app->configuration.size[0]/2 > 64 / axis->axisBlock[1])
								axis->axisBlock[0] = 64 / axis->axisBlock[1];
							else
								axis->axisBlock[0] = app->configuration.size[0]/2;*/
					}
					else {
						axis->axisBlock[0] = (app->configuration.size[0] > axis->groupedBatch) ? axis->groupedBatch : app->configuration.size[0];
						/*if (axis->axisBlock[0] * axis->axisBlock[1] < 64)
							if (app->configuration.size[0] > 64 / axis->axisBlock[1])
								axis->axisBlock[0] = 64 / axis->axisBlock[1];
							else
								axis->axisBlock[0] = app->configuration.size[0];*/
					}

					axis->axisBlock[2] = 1;
					axis->axisBlock[3] = axis->specializationConstants.fftDim;

				}
				if (axis_id == 2) {
					axis->axisBlock[1] = (axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread > 1) ? axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread : 1;

					if (app->configuration.performR2C) {
						if (axis_upload_id == 0) {
							FFTPlan->numSupportAxisUploads[1] = FFTPlan->numAxisUploads[2];
							for (uint32_t i = 0; i < FFTPlan->numSupportAxisUploads[1]; i++) {
								FFTPlan->supportAxes[axis_id - 1][i].specializationConstants.registers_per_thread = FFTPlan->axes[axis_id][i].specializationConstants.registers_per_thread;
								FFTPlan->supportAxes[axis_id - 1][i].specializationConstants.min_registers_per_thread = FFTPlan->axes[axis_id][i].specializationConstants.min_registers_per_thread;
								FFTPlan->supportAxes[axis_id - 1][i].specializationConstants.numStages = FFTPlan->axes[axis_id][i].specializationConstants.numStages;
								FFTPlan->supportAxes[axis_id - 1][i].specializationConstants.fftDim = FFTPlan->axisSplit[axis_id][i];
								for (uint32_t j = 0; j < FFTPlan->supportAxes[axis_id - 1][i].specializationConstants.numStages; j++)
									FFTPlan->supportAxes[axis_id - 1][i].specializationConstants.stageRadix[j] = FFTPlan->axes[axis_id][i].specializationConstants.stageRadix[j];
								VkFFTPlanSupportAxis(app, FFTPlan, 2, i, inverse, convolutionInverseStage);
							}
						}
						axis->axisBlock[0] = (app->configuration.size[0] / 2 > axis->groupedBatch) ? axis->groupedBatch : app->configuration.size[0] / 2;
						/*if (axis->axisBlock[0] * axis->axisBlock[1] < 64)
							if (app->configuration.size[0] / 2 > 64 / axis->axisBlock[1])
								axis->axisBlock[0] = 64 / axis->axisBlock[1];
							else
								axis->axisBlock[0] = app->configuration.size[0] / 2;*/
					}
					else {
						axis->axisBlock[0] = (app->configuration.size[0] > axis->groupedBatch) ? axis->groupedBatch : app->configuration.size[0];
						/*if (axis->axisBlock[0] * axis->axisBlock[1] < 64)
							if (app->configuration.size[0] > 64 / axis->axisBlock[1])
								axis->axisBlock[0] = 64 / axis->axisBlock[1];
							else
								axis->axisBlock[0] = app->configuration.size[0];*/
					}
					axis->axisBlock[2] = 1;
					axis->axisBlock[3] = axis->specializationConstants.fftDim;
				}
			}
			else {
				if (axis_id == 0) {
					if (axis_upload_id == 0) {
						axis->axisBlock[0] = (axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread > 1) ? axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread : 1;
						if (axis->axisBlock[0] > maxThreadNum) axis->axisBlock[0] = maxThreadNum;

						if (app->configuration.reorderFourStep && (FFTPlan->numAxisUploads[axis_id] > 1))
							axis->axisBlock[1] = axis->groupedBatch;
						else {
							axis->axisBlock[1] = (axis->axisBlock[0] < app->configuration.warpSize) ? app->configuration.warpSize / axis->axisBlock[0] : 1;
							if (app->configuration.size[1] < axis->axisBlock[1]) axis->axisBlock[1] = app->configuration.size[1];
						}
						axis->axisBlock[2] = 1;
						axis->axisBlock[3] = axis->specializationConstants.fftDim;
					}
					else {
						axis->axisBlock[1] = (axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread > 1) ? axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread : 1;

						axis->axisBlock[0] = (axis->specializationConstants.stageStartSize > axis->groupedBatch) ? axis->groupedBatch : axis->specializationConstants.stageStartSize;

						axis->axisBlock[2] = 1;
						axis->axisBlock[3] = axis->specializationConstants.fftDim;
					}
				}
				if (axis_id == 1) {

					axis->axisBlock[1] = (axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread > 1) ? axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread : 1;

					if (app->configuration.performR2C) {
						if (axis_upload_id == 0) {
							FFTPlan->numSupportAxisUploads[0] = FFTPlan->numAxisUploads[1];
							for (uint32_t i = 0; i < FFTPlan->numSupportAxisUploads[0]; i++) {
								FFTPlan->supportAxes[axis_id - 1][i].specializationConstants.registers_per_thread = FFTPlan->axes[axis_id][i].specializationConstants.registers_per_thread;
								FFTPlan->supportAxes[axis_id - 1][i].specializationConstants.min_registers_per_thread = FFTPlan->axes[axis_id][i].specializationConstants.min_registers_per_thread;
								FFTPlan->supportAxes[axis_id - 1][i].specializationConstants.numStages = FFTPlan->axes[axis_id][i].specializationConstants.numStages;
								FFTPlan->supportAxes[axis_id - 1][i].specializationConstants.fftDim = FFTPlan->axisSplit[axis_id][i];
								for (uint32_t j = 0; j < FFTPlan->supportAxes[axis_id - 1][i].specializationConstants.numStages; j++)
									FFTPlan->supportAxes[axis_id - 1][i].specializationConstants.stageRadix[j] = FFTPlan->axes[axis_id][i].specializationConstants.stageRadix[j];
								VkFFTPlanSupportAxis(app, FFTPlan, 1, i, inverse, convolutionInverseStage);
							}
						}
						axis->axisBlock[0] = (app->configuration.size[0] / 2 > axis->groupedBatch) ? axis->groupedBatch : app->configuration.size[0] / 2;
						/*if (axis->axisBlock[0] * axis->axisBlock[1] < 64)
							if (app->configuration.size[0] / 2 > 64 / axis->axisBlock[1])
								axis->axisBlock[0] = 64 / axis->axisBlock[1];
							else
								axis->axisBlock[0] = app->configuration.size[0] / 2;*/
					}
					else {
						axis->axisBlock[0] = (app->configuration.size[0] > axis->groupedBatch) ? axis->groupedBatch : app->configuration.size[0];
						/*if (axis->axisBlock[0] * axis->axisBlock[1] < 64)
							if (app->configuration.size[0] > 64 / axis->axisBlock[1])
								axis->axisBlock[0] = 64 / axis->axisBlock[1];
							else
								axis->axisBlock[0] = app->configuration.size[0];*/
					}
					axis->axisBlock[2] = 1;
					axis->axisBlock[3] = axis->specializationConstants.fftDim;

				}
				if (axis_id == 2) {

					axis->axisBlock[1] = (axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread > 1) ? axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread : 1;

					if (app->configuration.performR2C) {
						if (axis_upload_id == 0) {
							FFTPlan->numSupportAxisUploads[1] = FFTPlan->numAxisUploads[2];
							for (uint32_t i = 0; i < FFTPlan->numSupportAxisUploads[1]; i++) {
								FFTPlan->supportAxes[axis_id - 1][i].specializationConstants.registers_per_thread = FFTPlan->axes[axis_id][i].specializationConstants.registers_per_thread;
								FFTPlan->supportAxes[axis_id - 1][i].specializationConstants.min_registers_per_thread = FFTPlan->axes[axis_id][i].specializationConstants.min_registers_per_thread;
								FFTPlan->supportAxes[axis_id - 1][i].specializationConstants.numStages = FFTPlan->axes[axis_id][i].specializationConstants.numStages;
								FFTPlan->supportAxes[axis_id - 1][i].specializationConstants.fftDim = FFTPlan->axisSplit[axis_id][i];
								for (uint32_t j = 0; j < FFTPlan->supportAxes[axis_id - 1][i].specializationConstants.numStages; j++)
									FFTPlan->supportAxes[axis_id - 1][i].specializationConstants.stageRadix[j] = FFTPlan->axes[axis_id][i].specializationConstants.stageRadix[j];
								VkFFTPlanSupportAxis(app, FFTPlan, 2, i, inverse, convolutionInverseStage);
							}
						}
						axis->axisBlock[0] = (app->configuration.size[0] / 2 > axis->groupedBatch) ? axis->groupedBatch : app->configuration.size[0] / 2;
						/*if (axis->axisBlock[0] * axis->axisBlock[1] < 64)
							if (app->configuration.size[0] / 2 > 64 / axis->axisBlock[1])
								axis->axisBlock[0] = 64 / axis->axisBlock[1];
							else
								axis->axisBlock[0] = app->configuration.size[0] / 2;*/
					}
					else {
						axis->axisBlock[0] = (app->configuration.size[0] > axis->groupedBatch) ? axis->groupedBatch : app->configuration.size[0];
						/*if (axis->axisBlock[0] * axis->axisBlock[1] < 64)
							if (app->configuration.size[0] > 64 / axis->axisBlock[1])
								axis->axisBlock[0] = 64 / axis->axisBlock[1];
							else
								axis->axisBlock[0] = app->configuration.size[0];*/
					}
					axis->axisBlock[2] = 1;
					axis->axisBlock[3] = axis->specializationConstants.fftDim;

				}

			}


			uint32_t tempSize[3] = { app->configuration.size[0], app->configuration.size[1], app->configuration.size[2] };


			if (axis_id == 0) {
				if (axis_upload_id == 0)
					tempSize[0] = app->configuration.size[0] / axis->specializationConstants.fftDim / axis->axisBlock[1];
				else
					tempSize[0] = app->configuration.size[0] / axis->specializationConstants.fftDim / axis->axisBlock[0];
				if (app->configuration.performR2C == 1) tempSize[1] = ceil(tempSize[1] / 2.0);
				//if (app->configuration.performZeropadding[1]) tempSize[1] = ceil(tempSize[1] / 2.0);
				//if (app->configuration.performZeropadding[2]) tempSize[2] = ceil(tempSize[2] / 2.0);
				if (tempSize[0] > app->configuration.maxComputeWorkGroupCount[0]) axis->specializationConstants.performWorkGroupShift[0] = 1;
				else  axis->specializationConstants.performWorkGroupShift[0] = 0;
				if (tempSize[1] > app->configuration.maxComputeWorkGroupCount[1]) axis->specializationConstants.performWorkGroupShift[1] = 1;
				else  axis->specializationConstants.performWorkGroupShift[1] = 0;
				if (tempSize[2] > app->configuration.maxComputeWorkGroupCount[2]) axis->specializationConstants.performWorkGroupShift[2] = 1;
				else  axis->specializationConstants.performWorkGroupShift[2] = 0;
			}
			if (axis_id == 1) {
				tempSize[0] = app->configuration.size[0] / axis->axisBlock[0] * app->configuration.size[1] / axis->specializationConstants.fftDim;
				tempSize[1] = 1;
				tempSize[2] = app->configuration.size[2];
				if (app->configuration.performR2C == 1) tempSize[0] = ceil(tempSize[0] / 2.0);
				//if (app->configuration.performZeropadding[2]) tempSize[2] = ceil(tempSize[2] / 2.0);

				if (tempSize[0] > app->configuration.maxComputeWorkGroupCount[0]) axis->specializationConstants.performWorkGroupShift[0] = 1;
				else  axis->specializationConstants.performWorkGroupShift[0] = 0;
				if (tempSize[1] > app->configuration.maxComputeWorkGroupCount[1]) axis->specializationConstants.performWorkGroupShift[1] = 1;
				else  axis->specializationConstants.performWorkGroupShift[1] = 0;
				if (tempSize[2] > app->configuration.maxComputeWorkGroupCount[2]) axis->specializationConstants.performWorkGroupShift[2] = 1;
				else  axis->specializationConstants.performWorkGroupShift[2] = 0;

			}
			if (axis_id == 2) {
				tempSize[0] = app->configuration.size[0] / axis->axisBlock[0] * app->configuration.size[2] / axis->specializationConstants.fftDim;
				tempSize[1] = 1;
				tempSize[2] = app->configuration.size[1];
				if (app->configuration.performR2C == 1) tempSize[0] = ceil(tempSize[0] / 2.0);

				if (tempSize[0] > app->configuration.maxComputeWorkGroupCount[0]) axis->specializationConstants.performWorkGroupShift[0] = 1;
				else  axis->specializationConstants.performWorkGroupShift[0] = 0;
				if (tempSize[1] > app->configuration.maxComputeWorkGroupCount[1]) axis->specializationConstants.performWorkGroupShift[1] = 1;
				else  axis->specializationConstants.performWorkGroupShift[1] = 0;
				if (tempSize[2] > app->configuration.maxComputeWorkGroupCount[2]) axis->specializationConstants.performWorkGroupShift[2] = 1;
				else  axis->specializationConstants.performWorkGroupShift[2] = 0;

			}
			/*VkSpecializationMapEntry specializationMapEntries[36] = { {} };
			for (uint32_t i = 0; i < 36; i++) {
				specializationMapEntries[i].constantID = i + 1;
				specializationMapEntries[i].size = sizeof(uint32_t);
				specializationMapEntries[i].offset = i * sizeof(uint32_t);
			}
			VkSpecializationInfo specializationInfo = { 0 };
			specializationInfo.dataSize = 36 * sizeof(uint32_t);
			specializationInfo.mapEntryCount = 36;
			specializationInfo.pMapEntries = specializationMapEntries;*/
			axis->specializationConstants.localSize[0] = axis->axisBlock[0];
			axis->specializationConstants.localSize[1] = axis->axisBlock[1];
			axis->specializationConstants.localSize[2] = axis->axisBlock[2];
			//specializationInfo.pData = &axis->specializationConstants;
			VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };

			VkComputePipelineCreateInfo computePipelineCreateInfo = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };


			pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			uint32_t registerBoost = (FFTPlan->numAxisUploads[axis_id] > 1) ? app->configuration.registerBoost4Step : app->configuration.registerBoost;

			axis->specializationConstants.numCoordinates = (app->configuration.matrixConvolution > 1) ? 1 : app->configuration.coordinateFeatures;
			axis->specializationConstants.matrixConvolution = app->configuration.matrixConvolution;
			if ((app->configuration.FFTdim == 1) && (app->configuration.size[1] == 1) && (app->configuration.numberBatches > 1) && (!app->configuration.performConvolution) && (app->configuration.coordinateFeatures == 1)) {
				app->configuration.size[1] = app->configuration.numberBatches;
				app->configuration.numberBatches = 1;
			}
			axis->specializationConstants.numBatches = app->configuration.numberBatches;
			axis->specializationConstants.numKernels = app->configuration.numberKernels;
			axis->specializationConstants.sharedMemSize = app->configuration.sharedMemorySize;
			axis->specializationConstants.sharedMemSizePow2 = app->configuration.sharedMemorySizePow2;
			axis->specializationConstants.normalize = 1;
			axis->specializationConstants.size[0] = app->configuration.size[0];
			axis->specializationConstants.size[1] = app->configuration.size[1];
			axis->specializationConstants.size[2] = app->configuration.size[2];
			axis->specializationConstants.axis_id = axis_id;
			axis->specializationConstants.axis_upload_id = axis_upload_id;
			if (convolutionInverseStage) {
				for (uint32_t i = 0; i < 3; i++) {
					axis->specializationConstants.performZeropaddingInput[i] = app->configuration.performZeropaddingOutput[i]; // don't read if input is zeropadded (0 - off, 1 - on)
					axis->specializationConstants.performZeropaddingOutput[i] = app->configuration.performZeropaddingInput[i]; // don't write if output is zeropadded (0 - off, 1 - on)
					axis->specializationConstants.fft_zeropad_left_read[i] = app->configuration.fft_zeropad_left_write[i];
					axis->specializationConstants.fft_zeropad_left_write[i] = app->configuration.fft_zeropad_left_read[i];
					axis->specializationConstants.fft_zeropad_right_read[i] = app->configuration.fft_zeropad_right_write[i];
					axis->specializationConstants.fft_zeropad_right_write[i] = app->configuration.fft_zeropad_right_read[i];
				}
			}
			else {
				if ((app->configuration.FFTdim - 1 == axis_id) && (axis_upload_id == 0) && (app->configuration.performConvolution)) {
					for (uint32_t i = 0; i < 3; i++) {
						axis->specializationConstants.performZeropaddingInput[i] = app->configuration.performZeropaddingInput[i];
						axis->specializationConstants.performZeropaddingOutput[i] = app->configuration.performZeropaddingInput[i]; // don't write if output is zeropadded (0 - off, 1 - on)
						axis->specializationConstants.fft_zeropad_left_read[i] = app->configuration.fft_zeropad_left_read[i];
						axis->specializationConstants.fft_zeropad_right_read[i] = app->configuration.fft_zeropad_right_read[i];
						axis->specializationConstants.fft_zeropad_left_write[i] = app->configuration.fft_zeropad_left_read[i];
						axis->specializationConstants.fft_zeropad_right_write[i] = app->configuration.fft_zeropad_right_read[i];
					}
				}
				else {
					for (uint32_t i = 0; i < 3; i++) {
						axis->specializationConstants.performZeropaddingInput[i] = app->configuration.performZeropaddingInput[i]; // don't read if input is zeropadded (0 - off, 1 - on)
						axis->specializationConstants.performZeropaddingOutput[i] = app->configuration.performZeropaddingOutput[i]; // don't write if output is zeropadded (0 - off, 1 - on)
						axis->specializationConstants.fft_zeropad_left_read[i] = app->configuration.fft_zeropad_left_read[i];
						axis->specializationConstants.fft_zeropad_left_write[i] = app->configuration.fft_zeropad_left_write[i];
						axis->specializationConstants.fft_zeropad_right_read[i] = app->configuration.fft_zeropad_right_read[i];
						axis->specializationConstants.fft_zeropad_right_write[i] = app->configuration.fft_zeropad_right_write[i];
					}
				}
			}
			if (inverse) {
				axis->specializationConstants.zeropad[0] = (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) ? axis->specializationConstants.performZeropaddingInput[axis_id] : 0;
				axis->specializationConstants.zeropad[1] = (axis_upload_id == 0) ? axis->specializationConstants.performZeropaddingOutput[axis_id] : 0;
			}
			else {
				axis->specializationConstants.zeropad[0] = (axis_upload_id == 0) ? axis->specializationConstants.performZeropaddingInput[axis_id] : 0;
				axis->specializationConstants.zeropad[1] = (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) ? axis->specializationConstants.performZeropaddingOutput[axis_id] : 0;
			}
			if ((app->configuration.FFTdim - 1 == axis_id) && (axis_upload_id == 0) && (app->configuration.performConvolution)) {
				axis->specializationConstants.convolutionStep = 1;
			}
			else
				axis->specializationConstants.convolutionStep = 0;
			char floatTypeMemory[10];
			char floatType[10];
			axis->specializationConstants.unroll = 1;
			axis->specializationConstants.LUT = app->configuration.useLUT;
			//axis->specializationConstants.registers_per_thread = 8;
			if (app->configuration.doublePrecision) {
				sprintf(floatType, "double");
				sprintf(floatTypeMemory, "double");
				//axis->specializationConstants.unroll = 1;
			}
			else {
				//axis->specializationConstants.unroll = 0;
				if (app->configuration.halfPrecision) {
					sprintf(floatType, "float");
					sprintf(floatTypeMemory, "half");
				}
				else {
					sprintf(floatType, "float");
					sprintf(floatTypeMemory, "float");
				}
			}
			char uintType[10] = "uint";
			uint32_t LUT = app->configuration.useLUT;
			uint32_t type;
			if ((axis_id == 0) && (axis_upload_id == 0)) type = 0;
			if (axis_id != 0) type = 1;
			if ((axis_id == 0) && (axis_upload_id > 0)) type = 2;
			if ((axis->specializationConstants.fftDim == 2 * maxSequenceLengthSharedMemory) && (app->configuration.registerBoost >= 2)) type = 3;
			if ((axis->specializationConstants.fftDim == 4 * maxSequenceLengthSharedMemory) && (app->configuration.registerBoost >= 4)) type = 4;
			if ((axis_id == 0) && (!axis->specializationConstants.inverse) && (app->configuration.performR2C)) type = 5;
			if ((axis_id == 0) && (axis->specializationConstants.inverse) && (app->configuration.performR2C)) type = 6;
			axis->specializationConstants.cacheShuffle = (((axis->specializationConstants.fftDim & (axis->specializationConstants.fftDim - 1)) == 0) && (!app->configuration.doublePrecision) && ((type == 0) || (type == 5) || (type == 6))) ? 1 : 0;
			char* code0 = (char*)malloc(sizeof(char) * 200000);
			shaderGenVkFFT(code0, axis->specializationConstants, floatType, floatTypeMemory, uintType, type);
			const glslang_resource_t default_resource = {
				/* .MaxLights = */ 32,
				/* .MaxClipPlanes = */ 6,
				/* .MaxTextureUnits = */ 32,
				/* .MaxTextureCoords = */ 32,
				/* .MaxVertexAttribs = */ 64,
				/* .MaxVertexUniformComponents = */ 4096,
				/* .MaxVaryingFloats = */ 64,
				/* .MaxVertexTextureImageUnits = */ 32,
				/* .MaxCombinedTextureImageUnits = */ 80,
				/* .MaxTextureImageUnits = */ 32,
				/* .MaxFragmentUniformComponents = */ 4096,
				/* .MaxDrawBuffers = */ 32,
				/* .MaxVertexUniformVectors = */ 128,
				/* .MaxVaryingVectors = */ 8,
				/* .MaxFragmentUniformVectors = */ 16,
				/* .MaxVertexOutputVectors = */ 16,
				/* .MaxFragmentInputVectors = */ 15,
				/* .MinProgramTexelOffset = */ -8,
				/* .MaxProgramTexelOffset = */ 7,
				/* .MaxClipDistances = */ 8,
				/* .MaxComputeWorkGroupCountX = */ 65535,
				/* .MaxComputeWorkGroupCountY = */ 65535,
				/* .MaxComputeWorkGroupCountZ = */ 65535,
				/* .MaxComputeWorkGroupSizeX = */ 1024,
				/* .MaxComputeWorkGroupSizeY = */ 1024,
				/* .MaxComputeWorkGroupSizeZ = */ 64,
				/* .MaxComputeUniformComponents = */ 1024,
				/* .MaxComputeTextureImageUnits = */ 16,
				/* .MaxComputeImageUniforms = */ 8,
				/* .MaxComputeAtomicCounters = */ 8,
				/* .MaxComputeAtomicCounterBuffers = */ 1,
				/* .MaxVaryingComponents = */ 60,
				/* .MaxVertexOutputComponents = */ 64,
				/* .MaxGeometryInputComponents = */ 64,
				/* .MaxGeometryOutputComponents = */ 128,
				/* .MaxFragmentInputComponents = */ 128,
				/* .MaxImageUnits = */ 8,
				/* .MaxCombinedImageUnitsAndFragmentOutputs = */ 8,
				/* .MaxCombinedShaderOutputResources = */ 8,
				/* .MaxImageSamples = */ 0,
				/* .MaxVertexImageUniforms = */ 0,
				/* .MaxTessControlImageUniforms = */ 0,
				/* .MaxTessEvaluationImageUniforms = */ 0,
				/* .MaxGeometryImageUniforms = */ 0,
				/* .MaxFragmentImageUniforms = */ 8,
				/* .MaxCombinedImageUniforms = */ 8,
				/* .MaxGeometryTextureImageUnits = */ 16,
				/* .MaxGeometryOutputVertices = */ 256,
				/* .MaxGeometryTotalOutputComponents = */ 1024,
				/* .MaxGeometryUniformComponents = */ 1024,
				/* .MaxGeometryVaryingComponents = */ 64,
				/* .MaxTessControlInputComponents = */ 128,
				/* .MaxTessControlOutputComponents = */ 128,
				/* .MaxTessControlTextureImageUnits = */ 16,
				/* .MaxTessControlUniformComponents = */ 1024,
				/* .MaxTessControlTotalOutputComponents = */ 4096,
				/* .MaxTessEvaluationInputComponents = */ 128,
				/* .MaxTessEvaluationOutputComponents = */ 128,
				/* .MaxTessEvaluationTextureImageUnits = */ 16,
				/* .MaxTessEvaluationUniformComponents = */ 1024,
				/* .MaxTessPatchComponents = */ 120,
				/* .MaxPatchVertices = */ 32,
				/* .MaxTessGenLevel = */ 64,
				/* .MaxViewports = */ 16,
				/* .MaxVertexAtomicCounters = */ 0,
				/* .MaxTessControlAtomicCounters = */ 0,
				/* .MaxTessEvaluationAtomicCounters = */ 0,
				/* .MaxGeometryAtomicCounters = */ 0,
				/* .MaxFragmentAtomicCounters = */ 8,
				/* .MaxCombinedAtomicCounters = */ 8,
				/* .MaxAtomicCounterBindings = */ 1,
				/* .MaxVertexAtomicCounterBuffers = */ 0,
				/* .MaxTessControlAtomicCounterBuffers = */ 0,
				/* .MaxTessEvaluationAtomicCounterBuffers = */ 0,
				/* .MaxGeometryAtomicCounterBuffers = */ 0,
				/* .MaxFragmentAtomicCounterBuffers = */ 1,
				/* .MaxCombinedAtomicCounterBuffers = */ 1,
				/* .MaxAtomicCounterBufferSize = */ 16384,
				/* .MaxTransformFeedbackBuffers = */ 4,
				/* .MaxTransformFeedbackInterleavedComponents = */ 64,
				/* .MaxCullDistances = */ 8,
				/* .MaxCombinedClipAndCullDistances = */ 8,
				/* .MaxSamples = */ 4,
				/* .maxMeshOutputVerticesNV = */ 256,
				/* .maxMeshOutputPrimitivesNV = */ 512,
				/* .maxMeshWorkGroupSizeX_NV = */ 32,
				/* .maxMeshWorkGroupSizeY_NV = */ 1,
				/* .maxMeshWorkGroupSizeZ_NV = */ 1,
				/* .maxTaskWorkGroupSizeX_NV = */ 32,
				/* .maxTaskWorkGroupSizeY_NV = */ 1,
				/* .maxTaskWorkGroupSizeZ_NV = */ 1,
				/* .maxMeshViewCountNV = */ 4,
				/* .maxDualSourceDrawBuffersEXT = */ 1,

				/* .limits = */ {
					/* .nonInductiveForLoops = */ 1,
					/* .whileLoops = */ 1,
					/* .doWhileLoops = */ 1,
					/* .generalUniformIndexing = */ 1,
					/* .generalAttributeMatrixVectorIndexing = */ 1,
					/* .generalVaryingIndexing = */ 1,
					/* .generalSamplerIndexing = */ 1,
					/* .generalVariableIndexing = */ 1,
					/* .generalConstantMatrixVectorIndexing = */ 1,
				} };
			glslang_target_client_version_t client_version = (app->configuration.halfPrecision) ? GLSLANG_TARGET_VULKAN_1_1 : GLSLANG_TARGET_VULKAN_1_0;
			glslang_target_language_version_t target_language_version = (app->configuration.halfPrecision) ? GLSLANG_TARGET_SPV_1_3 : GLSLANG_TARGET_SPV_1_0;
			const glslang_input_t input =
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
				&default_resource,
			};
			//printf("%s\n", code0);
			glslang_shader_t* shader = glslang_shader_create(&input);
			const char* err;
			if (!glslang_shader_preprocess(shader, &input))
			{
				err = glslang_shader_get_info_log(shader);
				printf("%s\n", code0);
				printf("%s\nVkFFT shader type: %d\n", err, type);
				glslang_shader_delete(shader);
				free(code0);
				return VK_ERROR_INITIALIZATION_FAILED;

			}

			if (!glslang_shader_parse(shader, &input))
			{
				err = glslang_shader_get_info_log(shader);
				printf("%s\n", code0);
				printf("%s\nVkFFT shader type: %d\n", err, type);
				glslang_shader_delete(shader);
				free(code0);
				return VK_ERROR_INITIALIZATION_FAILED;

			}
			glslang_program_t* program = glslang_program_create();
			glslang_program_add_shader(program, shader);
			if (!glslang_program_link(program, GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT))
			{
				err = glslang_program_get_info_log(program);
				printf("%s\n", code0);
				printf("%s\nVkFFT shader type: %d\n", err, type);
				glslang_shader_delete(shader);
				free(code0);
				return VK_ERROR_INITIALIZATION_FAILED;

			}

			glslang_program_SPIRV_generate(program, input.stage);

			if (glslang_program_SPIRV_get_messages(program))
			{
				printf("%s", glslang_program_SPIRV_get_messages(program));
			}

			glslang_shader_delete(shader);
			/*char strr[20] = "out.comp";
			FILE* fname;
			fname = fopen(strr, "w");
			fprintf(fname, "%s", code0);
			fclose(fname);
			char strr2[20] = "out.spv";
			char strrSum[50];
			sprintf(strrSum, "glslangValidator.exe -V %s -o %s", strr, strr2);
			system(strrSum);
			uint32_t filelength;
			uint32_t* code = VkFFTReadShader(&filelength, strr2);*/
			free(code0);
			//shaderGenVkFFT(app->configuration.code0, axis->specializationConstants, floatType, floatTypeMemory, uintType, type);
			//printf("%s\n", app->configuration.code0);
			/*shaderc_compiler_t compiler = shaderc_compiler_initialize();
			shaderc_compile_options_t options = shaderc_compile_options_initialize();
			shaderc_compile_options_set_optimization_level(options, shaderc_optimization_level_performance);*/
			//shaderc_compilation_result_t result = shaderc_compile_into_spv(app->configuration.compiler[0], app->configuration.code0, strlen(app->configuration.code0), shaderc_glsl_default_compute_shader, "file", "main", app->configuration.options[0]);
			//memset(app->configuration.code0, 0, strlen(app->configuration.code0));
			//const char* err = shaderc_result_get_error_message(result);
			//uint32_t* code = (uint32_t*)shaderc_result_get_bytes(result);
			//uint32_t filelength = shaderc_result_get_length(result);
			//if (strcmp(err, "")) printf("%s\n", err);
			//sprintf(strrSum, "glslangValidator.exe -V %s -o %s", strr, strr2);
			//system(strrSum);
			//uint32_t filelength;
			//uint32_t* code = VkFFTReadShader(&filelength, strr2);

			VkShaderModuleCreateInfo createInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
			createInfo.pCode = glslang_program_SPIRV_get_ptr(program);
			createInfo.codeSize = glslang_program_SPIRV_get_size(program) * sizeof(uint32_t);
			vkCreateShaderModule(app->configuration.device[0], &createInfo, NULL, &pipelineShaderStageCreateInfo.module);
			//shaderc_result_release(result);
			//free(code0);
			/*shaderc_compile_options_release(options);
			shaderc_compiler_release(compiler);*/

			pipelineShaderStageCreateInfo.pName = "main";
			pipelineShaderStageCreateInfo.pSpecializationInfo = 0;// &specializationInfo;
			computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
			computePipelineCreateInfo.layout = axis->pipelineLayout;



			vkCreateComputePipelines(app->configuration.device[0], VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, &axis->pipeline);
			vkDestroyShaderModule(app->configuration.device[0], pipelineShaderStageCreateInfo.module, NULL);
			glslang_program_delete(program);
		}

		return VK_SUCCESS;
	}
	static inline void deleteAxis(VkFFTApplication* app, VkFFTAxis* axis) {
		if (app->configuration.useLUT) {
			vkDestroyBuffer(app->configuration.device[0], axis->bufferLUT, NULL);
			vkFreeMemory(app->configuration.device[0], axis->bufferLUTDeviceMemory, NULL);
		}
		vkDestroyDescriptorPool(app->configuration.device[0], axis->descriptorPool, NULL);
		vkDestroyDescriptorSetLayout(app->configuration.device[0], axis->descriptorSetLayout, NULL);
		vkDestroyPipelineLayout(app->configuration.device[0], axis->pipelineLayout, NULL);
		vkDestroyPipeline(app->configuration.device[0], axis->pipeline, NULL);


	}
	static inline VkResult initializeVulkanFFT(VkFFTApplication* app, VkFFTConfiguration inputLaunchConfiguration) {
		VkPhysicalDeviceProperties physicalDeviceProperties = {};
		vkGetPhysicalDeviceProperties(inputLaunchConfiguration.physicalDevice[0], &physicalDeviceProperties);
		app->configuration = inputLaunchConfiguration;
		app->configuration.maxComputeWorkGroupCount[0] = physicalDeviceProperties.limits.maxComputeWorkGroupCount[0];
		app->configuration.maxComputeWorkGroupCount[1] = physicalDeviceProperties.limits.maxComputeWorkGroupCount[1];
		app->configuration.maxComputeWorkGroupCount[2] = physicalDeviceProperties.limits.maxComputeWorkGroupCount[2];
		if ((physicalDeviceProperties.vendorID == 0x8086) && (!app->configuration.doublePrecision)) app->configuration.halfThreads = 1;
		//	if ((physicalDeviceProperties.vendorID == 0x8086) && (!app->configuration.doublePrecision)) app->configuration.sharedMemorySize /= 2;//Temporary measure, until L1 overutilization is enabled
		app->configuration.sharedMemorySize = physicalDeviceProperties.limits.maxComputeSharedMemorySize;
		app->configuration.sharedMemorySizePow2 = (physicalDeviceProperties.limits.maxComputeSharedMemorySize / 32768) * 32768;
		if (app->configuration.matrixConvolution > 1) app->configuration.coordinateFeatures = app->configuration.matrixConvolution;
		//LUT is not implemented for non-pow 2 yet
		if ((app->configuration.size[0] & (app->configuration.size[0] - 1)) != 0) {
			app->configuration.useLUT = 0;
		}
		if ((app->configuration.size[1] & (app->configuration.size[1] - 1)) != 0) {
			app->configuration.useLUT = 0;
		}
		if ((app->configuration.size[2] & (app->configuration.size[2] - 1)) != 0) {
			app->configuration.useLUT = 0;
		}
		app->configuration.registerBoost = 1;
		app->configuration.registerBoost4Step = 1;
		app->configuration.performHalfBandwidthBoost = 0;
		VkResult res = VK_SUCCESS;
		if (!app->configuration.isCompilerInitialized)
			glslang_initialize_process();
		if (app->configuration.performConvolution) {

			app->configuration.inverse = 0;
			for (uint32_t i = 0; i < app->configuration.FFTdim; i++) {
				app->configuration.sharedMemorySize = ((app->configuration.size[i] & (app->configuration.size[i] - 1)) == 0) ? app->configuration.sharedMemorySizePow2 : physicalDeviceProperties.limits.maxComputeSharedMemorySize;
				VkFFTScheduler(app, &app->localFFTPlan_inverse_convolution, i);
				for (uint32_t j = 0; j < app->localFFTPlan_inverse_convolution.numAxisUploads[i]; j++) {
					res = VkFFTPlanAxis(app, &app->localFFTPlan_inverse_convolution, i, j, 1, 1);
					if (res != 0) return VK_ERROR_INITIALIZATION_FAILED;
				}
			}

		}
		for (uint32_t i = 0; i < app->configuration.FFTdim; i++) {
			app->configuration.sharedMemorySize = ((app->configuration.size[i] & (app->configuration.size[i] - 1)) == 0) ? app->configuration.sharedMemorySizePow2 : physicalDeviceProperties.limits.maxComputeSharedMemorySize;
			VkFFTScheduler(app, &app->localFFTPlan, i);
			for (uint32_t j = 0; j < app->localFFTPlan.numAxisUploads[i]; j++) {
				res = VkFFTPlanAxis(app, &app->localFFTPlan, i, j, app->configuration.inverse, 0);
				if (res != 0) return VK_ERROR_INITIALIZATION_FAILED;
				if (!app->configuration.inverse) {
					//printf("%d %d %d %d %d\n", i,j,app->localFFTPlan.axes[i][j].axisBlock[0], app->localFFTPlan.axes[i][j].axisBlock[1], app->localFFTPlan.axes[i][j].axisBlock[2]);
				}
			}
		}

		if (!app->configuration.isCompilerInitialized)
			glslang_finalize_process();
		return res;
	}
	static inline void dispatchEnhanced(VkFFTApplication* app, VkCommandBuffer commandBuffer, VkFFTAxis* axis, uint32_t* dispatchBlock) {
		uint32_t maxBlockPow2Size[3] = { (uint32_t)pow(2,(uint32_t)log2(app->configuration.maxComputeWorkGroupCount[0])),(uint32_t)pow(2,(uint32_t)log2(app->configuration.maxComputeWorkGroupCount[1])),(uint32_t)pow(2,(uint32_t)log2(app->configuration.maxComputeWorkGroupCount[2])) };
		uint32_t blockNumber[3] = { (uint32_t)ceil(dispatchBlock[0] / (float)maxBlockPow2Size[0]),(uint32_t)ceil(dispatchBlock[1] / (float)maxBlockPow2Size[1]),(uint32_t)ceil(dispatchBlock[2] / (float)maxBlockPow2Size[2]) };
		//printf("%d %d %d\n", dispatchBlock[0], dispatchBlock[1], dispatchBlock[2]);
		for (uint32_t i = 0; i < 3; i++)
			if (blockNumber[i] == 1) maxBlockPow2Size[i] = dispatchBlock[i];
		for (uint32_t i = 0; i < blockNumber[0]; i++) {
			for (uint32_t j = 0; j < blockNumber[1]; j++) {
				for (uint32_t k = 0; k < blockNumber[2]; k++) {
					axis->pushConstants.workGroupShift[0] = i * maxBlockPow2Size[0];
					axis->pushConstants.workGroupShift[1] = j * maxBlockPow2Size[1];
					axis->pushConstants.workGroupShift[2] = k * maxBlockPow2Size[2];
					vkCmdPushConstants(commandBuffer, axis->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTPushConstantsLayout), &axis->pushConstants);
					vkCmdDispatch(commandBuffer, maxBlockPow2Size[0], maxBlockPow2Size[1], maxBlockPow2Size[2]);
				}
			}
		}
	}
	static inline void VkFFTAppend(VkFFTApplication* app, VkCommandBuffer commandBuffer) {
		VkMemoryBarrier memory_barrier = {
				VK_STRUCTURE_TYPE_MEMORY_BARRIER,
				0,
				VK_ACCESS_SHADER_WRITE_BIT,
				VK_ACCESS_SHADER_READ_BIT,
		};
		if (!app->configuration.inverse) {
			//FFT axis 0
			for (uint32_t j = 0; j < app->configuration.numberBatches; j++) {
				for (int l = app->localFFTPlan.numAxisUploads[0] - 1; l >= 0; l--) {
					VkFFTAxis* axis = &app->localFFTPlan.axes[0][l];
					axis->pushConstants.batch = j;
					uint32_t maxCoordinate = ((app->configuration.matrixConvolution) > 1 && (app->configuration.performConvolution) && (app->configuration.FFTdim == 1)) ? 1 : app->configuration.coordinateFeatures;
					for (uint32_t i = 0; i < maxCoordinate; i++) {
						axis->pushConstants.coordinate = i;


						vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
						vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
						uint32_t dispatchBlock[3];
						if (l == 0) {
							if (app->localFFTPlan.numAxisUploads[0] > 2) {
								dispatchBlock[0] = ceil(ceil(app->configuration.size[0] / axis->specializationConstants.fftDim / (double)axis->axisBlock[1]) / (double)app->localFFTPlan.axisSplit[0][1]) * app->localFFTPlan.axisSplit[0][1];
								dispatchBlock[1] = app->configuration.size[1];
							}
							else {
								if (app->localFFTPlan.numAxisUploads[0] > 1) {
									dispatchBlock[0] = ceil(ceil(app->configuration.size[0] / axis->specializationConstants.fftDim / (double)axis->axisBlock[1]));
									dispatchBlock[1] = app->configuration.size[1];
								}
								else {
									dispatchBlock[0] = app->configuration.size[0] / axis->specializationConstants.fftDim;
									dispatchBlock[1] = ceil(app->configuration.size[1] / (double)axis->axisBlock[1]);
								}
							}
						}
						else {
							dispatchBlock[0] = ceil(app->configuration.size[0] / axis->specializationConstants.fftDim / (double)axis->axisBlock[0]);
							dispatchBlock[1] = app->configuration.size[1];
						}
						dispatchBlock[2] = app->configuration.size[2];
						if (app->configuration.performR2C == 1) dispatchBlock[1] = ceil(dispatchBlock[1] / 2.0);
						//if (app->configuration.performZeropadding[1]) dispatchBlock[1] = ceil(dispatchBlock[1] / 2.0);
						//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
						dispatchEnhanced(app, commandBuffer, axis, dispatchBlock);
					}
					vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);
				}
			}

			if (app->configuration.FFTdim > 1) {

				//FFT axis 1
				if ((app->configuration.FFTdim == 2) && (app->configuration.performConvolution)) {
					if (app->configuration.performR2C == 1) {
						for (int l = app->localFFTPlan.numSupportAxisUploads[0] - 1; l >= 0; l--) {
							VkFFTAxis* axis = &app->localFFTPlan.supportAxes[0][l];
							uint32_t maxCoordinate = ((app->configuration.matrixConvolution > 1) && (l == 0)) ? 1 : app->configuration.coordinateFeatures;
							for (uint32_t i = 0; i < maxCoordinate; i++) {
								axis->pushConstants.coordinate = i;

								axis->pushConstants.batch = ((l == 0) && (app->configuration.matrixConvolution == 1)) ? app->configuration.numberKernels : 0;


								vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
								vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
								uint32_t dispatchBlock[3];
								if (l == 0)
									dispatchBlock[0] = app->configuration.size[1] / axis->specializationConstants.fftDim;
								else
									dispatchBlock[0] = ceil(app->configuration.size[1] / axis->specializationConstants.fftDim / (double)axis->axisBlock[0]);

								dispatchBlock[1] = 1;
								dispatchBlock[2] = app->configuration.size[2];
								//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
								dispatchEnhanced(app, commandBuffer, axis, dispatchBlock);

							}
							if (l > 0)
								vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

						}

					}

					for (int l = app->localFFTPlan.numAxisUploads[1] - 1; l >= 0; l--) {
						VkFFTAxis* axis = &app->localFFTPlan.axes[1][l];
						uint32_t maxCoordinate = ((app->configuration.matrixConvolution > 1) && (l == 0)) ? 1 : app->configuration.coordinateFeatures;
						for (uint32_t i = 0; i < maxCoordinate; i++) {

							axis->pushConstants.coordinate = i;
							axis->pushConstants.batch = ((l == 0) && (app->configuration.matrixConvolution == 1)) ? app->configuration.numberKernels : 0;

							vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
							vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
							uint32_t dispatchBlock[3];
							dispatchBlock[0] = ceil(app->configuration.size[0] / (double)axis->axisBlock[0] * app->configuration.size[1] / (double)axis->specializationConstants.fftDim);
							dispatchBlock[1] = 1;
							dispatchBlock[2] = app->configuration.size[2];
							if (app->configuration.performR2C == 1) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);
							//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
							dispatchEnhanced(app, commandBuffer, axis, dispatchBlock);
						}
						vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

					}
				}
				else {
					if (app->configuration.performR2C == 1) {
						for (uint32_t j = 0; j < app->configuration.numberBatches; j++) {
							for (int l = app->localFFTPlan.numSupportAxisUploads[0] - 1; l >= 0; l--) {
								VkFFTAxis* axis = &app->localFFTPlan.supportAxes[0][l];
								axis->pushConstants.batch = j;
								for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {
									axis->pushConstants.coordinate = i;

									vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
									vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
									uint32_t dispatchBlock[3];
									if (l == 0)
										dispatchBlock[0] = app->configuration.size[1] / axis->specializationConstants.fftDim;
									else
										dispatchBlock[0] = ceil(app->configuration.size[1] / axis->specializationConstants.fftDim / (double)axis->axisBlock[0]);

									dispatchBlock[1] = 1;
									dispatchBlock[2] = app->configuration.size[2];
									//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
									dispatchEnhanced(app, commandBuffer, axis, dispatchBlock);

								}
								if (l > 0)
									vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

							}
						}
					}
					for (uint32_t j = 0; j < app->configuration.numberBatches; j++) {
						for (int l = app->localFFTPlan.numAxisUploads[1] - 1; l >= 0; l--) {
							VkFFTAxis* axis = &app->localFFTPlan.axes[1][l];
							axis->pushConstants.batch = j;
							for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {
								axis->pushConstants.coordinate = i;

								vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
								vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
								uint32_t dispatchBlock[3];

								dispatchBlock[0] = ceil(app->configuration.size[0] / (double)axis->axisBlock[0] * app->configuration.size[1] / (double)axis->specializationConstants.fftDim);
								dispatchBlock[1] = 1;
								dispatchBlock[2] = app->configuration.size[2];
								if (app->configuration.performR2C == 1) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);
								//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
								dispatchEnhanced(app, commandBuffer, axis, dispatchBlock);
							}
							vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

						}
					}

				}
			}
			//FFT axis 2
			if (app->configuration.FFTdim > 2) {
				if ((app->configuration.FFTdim == 3) && (app->configuration.performConvolution)) {

					if (app->configuration.performR2C == 1) {

						for (int l = app->localFFTPlan.numSupportAxisUploads[1] - 1; l >= 0; l--) {
							VkFFTAxis* axis = &app->localFFTPlan.supportAxes[1][l];
							uint32_t maxCoordinate = ((app->configuration.matrixConvolution > 1) && (l == 0)) ? 1 : app->configuration.coordinateFeatures;
							for (uint32_t i = 0; i < maxCoordinate; i++) {
								axis->pushConstants.coordinate = i;

								axis->pushConstants.batch = ((l == 0) && (app->configuration.matrixConvolution == 1)) ? app->configuration.numberKernels : 0;


								vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
								vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
								uint32_t dispatchBlock[3];
								dispatchBlock[0] = ceil(app->configuration.size[1] / (double)axis->axisBlock[0] * app->configuration.size[2] / (double)axis->specializationConstants.fftDim);
								dispatchBlock[1] = 1;
								dispatchBlock[2] = 1;
								dispatchEnhanced(app, commandBuffer, axis, dispatchBlock);

							}
							if (l > 0)
								vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

						}
					}

					for (int l = app->localFFTPlan.numAxisUploads[2] - 1; l >= 0; l--) {

						VkFFTAxis* axis = &app->localFFTPlan.axes[2][l];
						uint32_t maxCoordinate = ((app->configuration.matrixConvolution > 1) && (l == 0)) ? 1 : app->configuration.coordinateFeatures;
						for (uint32_t i = 0; i < maxCoordinate; i++) {
							axis->pushConstants.coordinate = i;
							axis->pushConstants.batch = ((l == 0) && (app->configuration.matrixConvolution == 1)) ? app->configuration.numberKernels : 0;


							vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
							vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
							uint32_t dispatchBlock[3];
							dispatchBlock[0] = ceil(app->configuration.size[0] / (double)axis->axisBlock[0] * app->configuration.size[2] / (double)axis->specializationConstants.fftDim);
							dispatchBlock[1] = 1;
							dispatchBlock[2] = app->configuration.size[1];
							if (app->configuration.performR2C == 1) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);
							dispatchEnhanced(app, commandBuffer, axis, dispatchBlock);

						}
						vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

					}
				}
				else {
					if (app->configuration.performR2C == 1) {
						for (uint32_t j = 0; j < app->configuration.numberBatches; j++) {
							for (int l = app->localFFTPlan.numSupportAxisUploads[1] - 1; l >= 0; l--) {
								VkFFTAxis* axis = &app->localFFTPlan.supportAxes[1][l];
								axis->pushConstants.batch = j;
								for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {
									axis->pushConstants.coordinate = i;


									vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
									vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
									uint32_t dispatchBlock[3];
									dispatchBlock[0] = ceil(app->configuration.size[1] / (double)axis->axisBlock[0] * app->configuration.size[2] / (double)axis->specializationConstants.fftDim);
									dispatchBlock[1] = 1;
									dispatchBlock[2] = 1;
									dispatchEnhanced(app, commandBuffer, axis, dispatchBlock);

								}
								if (l > 0)
									vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

							}
						}
					}
					for (uint32_t j = 0; j < app->configuration.numberBatches; j++) {
						for (int l = app->localFFTPlan.numAxisUploads[2] - 1; l >= 0; l--) {
							VkFFTAxis* axis = &app->localFFTPlan.axes[2][l];
							axis->pushConstants.batch = j;
							for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {
								axis->pushConstants.coordinate = i;

								vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
								vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
								uint32_t dispatchBlock[3];
								dispatchBlock[0] = ceil(app->configuration.size[0] / (double)axis->axisBlock[0] * app->configuration.size[2] / (double)axis->specializationConstants.fftDim);
								dispatchBlock[1] = 1;
								dispatchBlock[2] = app->configuration.size[1];
								if (app->configuration.performR2C == 1) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);
								dispatchEnhanced(app, commandBuffer, axis, dispatchBlock);
							}
							vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

						}
					}

				}

			}
		}
		if (app->configuration.performConvolution) {
			if (app->configuration.FFTdim > 2) {

				//multiple upload ifft leftovers
				if (app->configuration.FFTdim == 3) {
					if (app->configuration.performR2C == 1) {
						for (uint32_t j = 0; j < app->configuration.numberKernels; j++) {
							for (int l = 1; l < app->localFFTPlan_inverse_convolution.numSupportAxisUploads[1]; l++) {
								VkFFTAxis* axis = &app->localFFTPlan_inverse_convolution.supportAxes[1][l];
								uint32_t maxCoordinate = app->configuration.coordinateFeatures;
								for (uint32_t i = 0; i < maxCoordinate; i++) {
									axis->pushConstants.coordinate = i;
									axis->pushConstants.batch = j;

									vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
									vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
									uint32_t dispatchBlock[3];
									dispatchBlock[0] = ceil(app->configuration.size[1] / (double)axis->axisBlock[0] * app->configuration.size[2] / (double)axis->specializationConstants.fftDim);
									dispatchBlock[1] = 1;
									dispatchBlock[2] = 1;
									dispatchEnhanced(app, commandBuffer, axis, dispatchBlock);

								}
								if (l > 0)
									vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

							}
						}
					}
					for (uint32_t j = 0; j < app->configuration.numberKernels; j++) {
						for (int l = 1; l < app->localFFTPlan_inverse_convolution.numAxisUploads[2]; l++) {
							VkFFTAxis* axis = &app->localFFTPlan_inverse_convolution.axes[2][l];
							uint32_t maxCoordinate = app->configuration.coordinateFeatures;
							for (uint32_t i = 0; i < maxCoordinate; i++) {
								axis->pushConstants.coordinate = i;
								axis->pushConstants.batch = j;

								vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
								vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
								uint32_t dispatchBlock[3];
								dispatchBlock[0] = ceil(app->configuration.size[0] / (double)axis->axisBlock[0] * app->configuration.size[2] / (double)axis->specializationConstants.fftDim);
								dispatchBlock[1] = 1;
								dispatchBlock[2] = app->configuration.size[1];
								if (app->configuration.performR2C == 1) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);
								dispatchEnhanced(app, commandBuffer, axis, dispatchBlock);
							}
							vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

						}
					}
				}
				if (app->configuration.performR2C == 1) {
					for (uint32_t j = 0; j < app->configuration.numberKernels; j++) {
						for (int l = 0; l < app->localFFTPlan_inverse_convolution.numSupportAxisUploads[0]; l++) {
							VkFFTAxis* axis = &app->localFFTPlan_inverse_convolution.supportAxes[0][l];
							axis->pushConstants.batch = j;
							for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {

								axis->pushConstants.coordinate = i;

								vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
								vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
								uint32_t dispatchBlock[3];
								if (l == 0)
									dispatchBlock[0] = app->configuration.size[1] / axis->specializationConstants.fftDim;
								else
									dispatchBlock[0] = ceil(app->configuration.size[1] / axis->specializationConstants.fftDim / (double)axis->axisBlock[0]);
								dispatchBlock[1] = 1;
								dispatchBlock[2] = app->configuration.size[2];
								//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
								dispatchEnhanced(app, commandBuffer, axis, dispatchBlock);
							}
							if (l < app->localFFTPlan_inverse_convolution.numSupportAxisUploads[0] - 1)
								vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

						}
					}
				}
				for (uint32_t j = 0; j < app->configuration.numberKernels; j++) {
					for (int l = 0; l < app->localFFTPlan_inverse_convolution.numAxisUploads[1]; l++) {
						VkFFTAxis* axis = &app->localFFTPlan_inverse_convolution.axes[1][l];
						axis->pushConstants.batch = j;
						for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {
							axis->pushConstants.coordinate = i;

							vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
							vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
							uint32_t dispatchBlock[3];
							dispatchBlock[0] = ceil(app->configuration.size[0] / (double)axis->axisBlock[0] * app->configuration.size[1] / (double)axis->specializationConstants.fftDim);
							dispatchBlock[1] = 1;
							dispatchBlock[2] = app->configuration.size[2];
							if (app->configuration.performR2C == 1) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);
							//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
							dispatchEnhanced(app, commandBuffer, axis, dispatchBlock);

						}
						vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

					}
				}

			}
			if (app->configuration.FFTdim > 1) {
				if (app->configuration.FFTdim == 2) {
					if (app->configuration.performR2C == 1) {
						for (uint32_t j = 0; j < app->configuration.numberKernels; j++) {
							for (int l = 1; l < app->localFFTPlan_inverse_convolution.numSupportAxisUploads[0]; l++) {
								VkFFTAxis* axis = &app->localFFTPlan_inverse_convolution.supportAxes[0][l];
								uint32_t maxCoordinate = app->configuration.coordinateFeatures;
								for (uint32_t i = 0; i < maxCoordinate; i++) {
									axis->pushConstants.coordinate = i;
									axis->pushConstants.batch = j;

									vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
									vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
									uint32_t dispatchBlock[3];
									if (l == 0)
										dispatchBlock[0] = app->configuration.size[1] / axis->specializationConstants.fftDim;
									else
										dispatchBlock[0] = ceil(app->configuration.size[1] / axis->specializationConstants.fftDim / (double)axis->axisBlock[0]);
									dispatchBlock[1] = 1;
									dispatchBlock[2] = app->configuration.size[2];
									//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
									dispatchEnhanced(app, commandBuffer, axis, dispatchBlock);

								}
								if (l > 0)
									vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

							}
						}

					}
					for (uint32_t j = 0; j < app->configuration.numberKernels; j++) {
						for (int l = 1; l < app->localFFTPlan_inverse_convolution.numAxisUploads[1]; l++) {
							VkFFTAxis* axis = &app->localFFTPlan_inverse_convolution.axes[1][l];
							uint32_t maxCoordinate = app->configuration.coordinateFeatures;
							for (uint32_t i = 0; i < maxCoordinate; i++) {

								axis->pushConstants.coordinate = i;
								axis->pushConstants.batch = j;

								vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
								vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
								uint32_t dispatchBlock[3];
								dispatchBlock[0] = ceil(app->configuration.size[0] / (double)axis->axisBlock[0] * app->configuration.size[1] / (double)axis->specializationConstants.fftDim);
								dispatchBlock[1] = 1;
								dispatchBlock[2] = app->configuration.size[2];
								if (app->configuration.performR2C == 1) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);
								//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
								dispatchEnhanced(app, commandBuffer, axis, dispatchBlock);

							}
							vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

						}
					}
				}
				for (uint32_t j = 0; j < app->configuration.numberKernels; j++) {
					for (int l = 0; l < app->localFFTPlan_inverse_convolution.numAxisUploads[0]; l++) {
						VkFFTAxis* axis = &app->localFFTPlan_inverse_convolution.axes[0][l];
						axis->pushConstants.batch = j;
						for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {
							axis->pushConstants.coordinate = i;

							vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
							vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
							uint32_t dispatchBlock[3];
							if (l == 0) {
								if (app->localFFTPlan.numAxisUploads[0] > 2) {
									dispatchBlock[0] = ceil(ceil(app->configuration.size[0] / axis->specializationConstants.fftDim / (double)axis->axisBlock[1]) / (double)app->localFFTPlan.axisSplit[0][1]) * app->localFFTPlan.axisSplit[0][1];
									dispatchBlock[1] = app->configuration.size[1];
								}
								else {
									if (app->localFFTPlan.numAxisUploads[0] > 1) {
										dispatchBlock[0] = ceil(ceil(app->configuration.size[0] / axis->specializationConstants.fftDim / (double)axis->axisBlock[1]));
										dispatchBlock[1] = app->configuration.size[1];
									}
									else {
										dispatchBlock[0] = app->configuration.size[0] / axis->specializationConstants.fftDim;
										dispatchBlock[1] = ceil(app->configuration.size[1] / (double)axis->axisBlock[1]);
									}
								}
							}
							else {
								dispatchBlock[0] = ceil(app->configuration.size[0] / axis->specializationConstants.fftDim / (double)axis->axisBlock[0]);
								dispatchBlock[1] = app->configuration.size[1];
							}
							dispatchBlock[2] = app->configuration.size[2];
							if (app->configuration.performR2C == 1) dispatchBlock[1] = ceil(dispatchBlock[1] / 2.0);
							//if (app->configuration.performZeropadding[1]) dispatchBlock[1] = ceil(dispatchBlock[1] / 2.0);
							//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
							dispatchEnhanced(app, commandBuffer, axis, dispatchBlock);
						}
						vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

					}
				}


			}
			if (app->configuration.FFTdim == 1) {
				for (uint32_t j = 0; j < app->configuration.numberKernels; j++) {
					for (int l = 1; l < app->localFFTPlan_inverse_convolution.numAxisUploads[0]; l++) {
						VkFFTAxis* axis = &app->localFFTPlan_inverse_convolution.axes[0][l];
						uint32_t maxCoordinate = app->configuration.coordinateFeatures;
						for (uint32_t i = 0; i < maxCoordinate; i++) {

							axis->pushConstants.coordinate = i;
							axis->pushConstants.batch = j;

							vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
							vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
							uint32_t dispatchBlock[3];
							dispatchBlock[0] = ceil(app->configuration.size[0] / (double)axis->axisBlock[0] * app->configuration.size[1] / (double)axis->specializationConstants.fftDim);
							dispatchBlock[1] = 1;
							dispatchBlock[2] = app->configuration.size[2];
							if (app->configuration.performR2C == 1) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);
							//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
							dispatchEnhanced(app, commandBuffer, axis, dispatchBlock);

						}
						vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

					}
				}
			}
		}

		if (app->configuration.inverse) {
			//we start from axis 2 and go back to axis 0
			//FFT axis 2
			if (app->configuration.FFTdim > 2) {
				if (app->configuration.performR2C == 1) {
					for (uint32_t j = 0; j < app->configuration.numberBatches; j++) {
						for (int l = app->localFFTPlan.numSupportAxisUploads[1] - 1; l >= 0; l--) {
							if (!app->configuration.reorderFourStep) l = app->localFFTPlan.numSupportAxisUploads[1] - 1 - l;
							VkFFTAxis* axis = &app->localFFTPlan.supportAxes[1][l];
							axis->pushConstants.batch = j;
							for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {
								axis->pushConstants.coordinate = i;


								vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
								vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
								uint32_t dispatchBlock[3];
								dispatchBlock[0] = ceil(app->configuration.size[1] / (double)axis->axisBlock[0] * app->configuration.size[2] / (double)axis->specializationConstants.fftDim);
								dispatchBlock[1] = 1;
								dispatchBlock[2] = 1;
								//if (app->configuration.performZeropaddingInverse[1]) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);

								dispatchEnhanced(app, commandBuffer, axis, dispatchBlock);

							}
							if (!app->configuration.reorderFourStep) l = app->localFFTPlan.numSupportAxisUploads[1] - 1 - l;
							if (l > 0)
								vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

						}
					}
				}

				for (uint32_t j = 0; j < app->configuration.numberBatches; j++) {
					for (int l = app->localFFTPlan.numAxisUploads[2] - 1; l >= 0; l--) {
						if (!app->configuration.reorderFourStep) l = app->localFFTPlan.numAxisUploads[2] - 1 - l;
						VkFFTAxis* axis = &app->localFFTPlan.axes[2][l];
						axis->pushConstants.batch = j;
						for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {
							axis->pushConstants.coordinate = i;

							vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
							vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
							uint32_t dispatchBlock[3];
							dispatchBlock[0] = ceil(app->configuration.size[0] / (double)axis->axisBlock[0] * app->configuration.size[2] / (double)axis->specializationConstants.fftDim);
							dispatchBlock[1] = 1;
							dispatchBlock[2] = app->configuration.size[1];
							//if (app->configuration.performZeropaddingInverse[0]) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);
							//if (app->configuration.performZeropaddingInverse[1]) dispatchBlock[1] = ceil(dispatchBlock[1] / 2.0);

							if (app->configuration.performR2C == 1) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);
							dispatchEnhanced(app, commandBuffer, axis, dispatchBlock);
						}
						vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);
						if (!app->configuration.reorderFourStep) l = app->localFFTPlan.numAxisUploads[2] - 1 - l;
					}
				}

			}
			if (app->configuration.FFTdim > 1) {

				//FFT axis 1
				if (app->configuration.performR2C == 1) {
					for (uint32_t j = 0; j < app->configuration.numberBatches; j++) {
						for (int l = app->localFFTPlan.numSupportAxisUploads[0] - 1; l >= 0; l--) {
							if (!app->configuration.reorderFourStep) l = app->localFFTPlan.numSupportAxisUploads[0] - 1 - l;
							VkFFTAxis* axis = &app->localFFTPlan.supportAxes[0][l];
							axis->pushConstants.batch = j;
							for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {
								axis->pushConstants.coordinate = i;

								vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
								vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
								uint32_t dispatchBlock[3];
								if (l == 0)
									dispatchBlock[0] = app->configuration.size[1] / axis->specializationConstants.fftDim;
								else
									dispatchBlock[0] = ceil(app->configuration.size[1] / axis->specializationConstants.fftDim / (double)axis->axisBlock[0]);
								dispatchBlock[1] = 1;
								dispatchBlock[2] = app->configuration.size[2];
								//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
								dispatchEnhanced(app, commandBuffer, axis, dispatchBlock);

							}
							if (!app->configuration.reorderFourStep) l = app->localFFTPlan.numSupportAxisUploads[0] - 1 - l;
							if (l > 0)
								vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

						}
					}
				}
				for (uint32_t j = 0; j < app->configuration.numberBatches; j++) {
					for (int l = app->localFFTPlan.numAxisUploads[1] - 1; l >= 0; l--) {
						if (!app->configuration.reorderFourStep) l = app->localFFTPlan.numAxisUploads[1] - 1 - l;
						VkFFTAxis* axis = &app->localFFTPlan.axes[1][l];
						axis->pushConstants.batch = j;
						for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {
							axis->pushConstants.coordinate = i;

							vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
							vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
							uint32_t dispatchBlock[3];
							dispatchBlock[0] = ceil(app->configuration.size[0] / (double)axis->axisBlock[0] * app->configuration.size[1] / (double)axis->specializationConstants.fftDim);
							dispatchBlock[1] = 1;
							dispatchBlock[2] = app->configuration.size[2];
							if (app->configuration.performR2C == 1) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);
							//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
							//if (app->configuration.performZeropaddingInverse[0]) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);

							dispatchEnhanced(app, commandBuffer, axis, dispatchBlock);

						}
						if (!app->configuration.reorderFourStep) l = app->localFFTPlan.numAxisUploads[1] - 1 - l;

						vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

					}
				}

			}
			//FFT axis 0
			for (uint32_t j = 0; j < app->configuration.numberBatches; j++) {
				for (int l = app->localFFTPlan.numAxisUploads[0] - 1; l >= 0; l--) {
					if (!app->configuration.reorderFourStep) l = app->localFFTPlan.numAxisUploads[0] - 1 - l;
					VkFFTAxis* axis = &app->localFFTPlan.axes[0][l];
					axis->pushConstants.batch = j;
					uint32_t maxCoordinate = ((app->configuration.matrixConvolution) > 1 && (app->configuration.performConvolution) && (app->configuration.FFTdim == 1)) ? 1 : app->configuration.coordinateFeatures;
					for (uint32_t i = 0; i < maxCoordinate; i++) {
						axis->pushConstants.coordinate = i;

						vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
						vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
						uint32_t dispatchBlock[3];
						if (l == 0) {
							if (app->localFFTPlan.numAxisUploads[0] > 2) {
								dispatchBlock[0] = ceil(ceil(app->configuration.size[0] / axis->specializationConstants.fftDim / (double)axis->axisBlock[1]) / (double)app->localFFTPlan.axisSplit[0][1]) * app->localFFTPlan.axisSplit[0][1];
								dispatchBlock[1] = app->configuration.size[1];
							}
							else {
								if (app->localFFTPlan.numAxisUploads[0] > 1) {
									dispatchBlock[0] = ceil(ceil(app->configuration.size[0] / axis->specializationConstants.fftDim / (double)axis->axisBlock[1]));
									dispatchBlock[1] = app->configuration.size[1];
								}
								else {
									dispatchBlock[0] = app->configuration.size[0] / axis->specializationConstants.fftDim;
									dispatchBlock[1] = ceil(app->configuration.size[1] / (double)axis->axisBlock[1]);
								}
							}
						}
						else {
							dispatchBlock[0] = ceil(app->configuration.size[0] / axis->specializationConstants.fftDim / (double)axis->axisBlock[0]);
							dispatchBlock[1] = app->configuration.size[1];
						}
						dispatchBlock[2] = app->configuration.size[2];
						if (app->configuration.performR2C == 1) dispatchBlock[1] = ceil(dispatchBlock[1] / 2.0);
						//if (app->configuration.performZeropadding[1]) dispatchBlock[1] = ceil(dispatchBlock[1] / 2.0);
						//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
						dispatchEnhanced(app, commandBuffer, axis, dispatchBlock);
					}
					if (!app->configuration.reorderFourStep) l = app->localFFTPlan.numAxisUploads[0] - 1 - l;
					vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

				}
			}


		}
	}
	static inline void deleteVulkanFFT(VkFFTApplication* app) {
		for (uint32_t i = 0; i < app->configuration.FFTdim; i++) {
			for (uint32_t j = 0; j < app->localFFTPlan.numAxisUploads[i]; j++)
				deleteAxis(app, &app->localFFTPlan.axes[i][j]);
		}

		for (uint32_t i = 0; i < app->configuration.FFTdim - 1; i++) {

			if (app->configuration.performR2C) {
				for (uint32_t j = 0; j < app->localFFTPlan.numSupportAxisUploads[i]; j++)
					deleteAxis(app, &app->localFFTPlan.supportAxes[i][j]);
			}
		}
		if (app->configuration.performConvolution) {
			for (uint32_t i = 0; i < app->configuration.FFTdim; i++) {
				for (uint32_t j = 0; j < app->localFFTPlan_inverse_convolution.numAxisUploads[i]; j++)
					deleteAxis(app, &app->localFFTPlan_inverse_convolution.axes[i][j]);
			}
			for (uint32_t i = 0; i < app->configuration.FFTdim - 1; i++) {
				if (app->configuration.performR2C) {
					for (uint32_t j = 0; j < app->localFFTPlan_inverse_convolution.numSupportAxisUploads[i]; j++)
						deleteAxis(app, &app->localFFTPlan_inverse_convolution.supportAxes[i][j]);
				}
			}
		}
	}
#ifdef __cplusplus
}
#endif
#endif