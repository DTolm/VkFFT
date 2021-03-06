// This file is part of VkFFT, a Vulkan Fast Fourier Transform library
//
// Copyright (C) 2021 Dmitrii Tolmachev <dtolm96@gmail.com>
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
#if(VKFFT_BACKEND==0)
#include "vulkan/vulkan.h"
#include "glslang_c_interface.h"
#elif(VKFFT_BACKEND==1)
#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#elif(VKFFT_BACKEND==2)
#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_complex.h>
#endif

	typedef struct {
		//WHDCN layout

		//required parameters:
		uint32_t FFTdim; //FFT dimensionality (1, 2 or 3)
		uint32_t size[3]; // WHD -system dimensions 

#if(VKFFT_BACKEND==0)
		VkPhysicalDevice* physicalDevice;//pointer to Vulkan physical device, obtained from vkEnumeratePhysicalDevices
		VkDevice* device;//pointer to Vulkan device, created with vkCreateDevice
		VkQueue* queue;//pointer to Vulkan queue, created with vkGetDeviceQueue
		VkCommandPool* commandPool;//pointer to Vulkan command pool, created with vkCreateCommandPool
		VkFence* fence;//pointer to Vulkan fence, created with vkCreateFence
		uint32_t isCompilerInitialized;//specify if glslang compiler has been intialized before (0 - off, 1 - on). Default 0
#elif(VKFFT_BACKEND==1)
		CUdevice* device;//pointer to CUDA device, obtained from cuDeviceGet
		//CUcontext* context;//pointer to CUDA context, obtained from cuDeviceGet
		cudaStream_t* stream;//pointer to streams (can be more than 1), where to execute the kernels
		uint32_t num_streams;//try to submit CUDA kernels in multiple streams for asynchronous execution. Default 1
#elif(VKFFT_BACKEND==2)
		hipDevice_t* device;//pointer to HIP device, obtained from hipDeviceGet
		//hipCtx_t* context;//pointer to HIP context, obtained from hipDeviceGet
		hipStream_t* stream;//pointer to streams (can be more than 1), where to execute the kernels
		uint32_t num_streams;//try to submit HIP kernels in multiple streams for asynchronous execution. Default 1
#endif

		//data parameters:
		uint32_t userTempBuffer; //buffer allocated by app automatically if needed to reorder Four step algorithm. Setting to non zero value enables manual user allocation (0 - off, 1 - on)

		uint32_t bufferNum;//multiple buffer sequence storage is Vulkan only. Default 1
		uint32_t tempBufferNum;//multiple buffer sequence storage is Vulkan only. Default 1, buffer allocated by app automatically if needed to reorder Four step algorithm. Setting to non zero value enables manual user allocation
		uint32_t inputBufferNum;//multiple buffer sequence storage is Vulkan only. Default 1, if isInputFormatted is enabled
		uint32_t outputBufferNum;//multiple buffer sequence storage is Vulkan only. Default 1, if isOutputFormatted is enabled
		uint32_t kernelNum;//multiple buffer sequence storage is Vulkan only. Default 1, if performConvolution is enabled

		uint64_t* bufferSize;//array of buffers sizes in bytes
		uint64_t* tempBufferSize;//array of temp buffers sizes in bytes. Default set to bufferSize sum, buffer allocated by app automatically if needed to reorder Four step algorithm. Setting to non zero value enables manual user allocation
		uint64_t* inputBufferSize;//array of input buffers sizes in bytes, if isInputFormatted is enabled
		uint64_t* outputBufferSize;//array of output buffers sizes in bytes, if isOutputFormatted is enabled
		uint64_t* kernelSize;//array of kernel buffers sizes in bytes, if performConvolution is enabled

#if(VKFFT_BACKEND==0)
		VkBuffer* buffer;//pointer to array of buffers (or one buffer) used for computations
		VkBuffer* tempBuffer;//needed if reorderFourStep is enabled to transpose the array. Same sum size or bigger as buffer (can be split in multiple). Default 0. Setting to non zero value enables manual user allocation
		VkBuffer* inputBuffer;//pointer to array of input buffers (or one buffer) used to read data from if isInputFormatted is enabled
		VkBuffer* outputBuffer;//pointer to array of output buffers (or one buffer) used for write data to if isOutputFormatted is enabled
		VkBuffer* kernel;//pointer to array of kernel buffers (or one buffer) used for read kernel data from if performConvolution is enabled
#elif(VKFFT_BACKEND==1)
		void** buffer;//pointer to device buffer used for computations
		void** tempBuffer;//needed if reorderFourStep is enabled to transpose the array. Same size as buffer. Default 0. Setting to non zero value enables manual user allocation
		void** inputBuffer;//pointer to device buffer used to read data from if isInputFormatted is enabled
		void** outputBuffer;//pointer to device buffer used to read data from if isOutputFormatted is enabled
		void** kernel;//pointer to device buffer used to read kernel data from if performConvolution is enabled
#elif(VKFFT_BACKEND==2)
		void** buffer;//pointer to device buffer used for computations
		void** tempBuffer;//needed if reorderFourStep is enabled to transpose the array. Same size as buffer. Default 0. Setting to non zero value enables manual user allocation
		void** inputBuffer;//pointer to device buffer used to read data from if isInputFormatted is enabled
		void** outputBuffer;//pointer to device buffer used to read data from if isOutputFormatted is enabled
		void** kernel;//pointer to device buffer used to read kernel data from if performConvolution is enabled
#endif

		//optional: (default 0 if not stated otherwise)
		uint32_t coalescedMemory;//in bits, for Nvidia and AMD is equal to 32, Intel is equal 64, scaled for half precision. Gonna work regardles, but if specified by user correctly, the performance will be higher. 
		uint32_t aimThreads;//aim at this many threads per block. Default 128
		uint32_t numSharedBanks;//how many banks shared memory has. Default 32

		uint32_t doublePrecision; //perform calculations in double precision (0 - off, 1 - on).
		uint32_t halfPrecision; //perform calculations in half precision (0 - off, 1 - on)
		uint32_t halfPrecisionMemoryOnly; //use half precision only as input/output buffer. Input/Output have to be allocated as half, buffer/tempBuffer have to be allocated as float (out of place mode only). Specify isInputFormatted and isOutputFormatted to use (0 - off, 1 - on)

		uint32_t performR2C; //perform R2C/C2R decomposition (0 - off, 1 - on)
		uint32_t normalize; //normalize inverse transform (0 - off, 1 - on)
		uint32_t disableReorderFourStep; // disables unshuffling of Four step algorithm. Requires tempbuffer allocation (0 - off, 1 - on)
		uint32_t useLUT; //switches from calculating sincos to using precomputed LUT tables (0 - off, 1 - on). Configured by initialization routine

		uint32_t bufferStride[3];//buffer strides - default set to x - x*y - x*y*z values
		uint32_t isInputFormatted; //specify if input buffer is padded - 0 - padded, 1 - not padded. For example if it is not padded for R2C if out-of-place mode is selected (only if numberBatches==1 and numberKernels==1)
		uint32_t isOutputFormatted; //specify if output buffer is padded - 0 - padded, 1 - not padded. For example if it is not padded for R2C if out-of-place mode is selected (only if numberBatches==1 and numberKernels==1)
		uint32_t inputBufferStride[3];//input buffer strides. Used if isInputFormatted is enabled. Default set to bufferStride values
		uint32_t outputBufferStride[3];//output buffer strides. Used if isInputFormatted is enabled. Default set to bufferStride values

		//optional zero padding control parameters: (default 0 if not stated otherwise)
		uint32_t performZeropadding[3]; // don't read some data/perform computations if some input sequences are zeropadded for each axis (0 - off, 1 - on)
		uint32_t fft_zeropad_left[3];//specify start boundary of zero block in the system for each axis
		uint32_t fft_zeropad_right[3];//specify end boundary of zero block in the system for each axis
		uint32_t frequencyZeroPadding; //set to 1 if zeropadding of frequency domain, default 0 - spatial zeropadding

		//optional convolution control parameters: (default 0 if not stated otherwise)
		uint32_t performConvolution; //perform convolution in this application (0 - off, 1 - on). Disables reorderFourStep parameter
		uint32_t coordinateFeatures; // C - coordinate, or dimension of features vector. In matrix convolution - size of vector
		uint32_t matrixConvolution; //if equal to 2 perform 2x2, if equal to 3 perform 3x3 matrix-vector convolution. Overrides coordinateFeatures
		uint32_t symmetricKernel; //specify if kernel in 2x2 or 3x3 matrix convolution is symmetric
		uint32_t numberBatches;// N - used to perform multiple batches of initial data
		uint32_t numberKernels;// N - only used in convolution step - specify how many kernels were initialized before. Expands one input to multiple (batched) output
		uint32_t kernelConvolution;// specify if this application is used to create kernel for convolution, so it has the same properties. performConvolution has to be set to 0 for kernel creation

		//register overutilization (experimental): (default 0 if not stated otherwise)
		uint32_t registerBoost; //specify if register file size is bigger than shared memory and can be used to extend it X times (on Nvidia 256KB register file can be used instead of 32KB of shared memory, set this constant to 4 to emulate 128KB of shared memory). Default 1
		uint32_t registerBoostNonPow2; //specify if register overutilization should be used on non power of 2 sequences (0 - off, 1 - on)
		uint32_t registerBoost4Step; //specify if register file overutilization should be used in big sequences (>2^14), same definition as registerBoost. Default 1

		//not used techniques:
		uint32_t swapTo3Stage4Step; //specify at which power of 2 to switch from 2 upload to 3 upload 4-step FFT, in case if making max sequence size lower than coalesced sequence helps to combat TLB misses. Default 0 - disabled. Must be at least 17
		uint32_t performHalfBandwidthBoost;//try to reduce coalsesced number by a factor of 2 to get bigger sequence in one upload
		uint32_t devicePageSize;//in KB, the size of a page on the GPU. Setting to 0 disables local buffer split in pages
		uint32_t localPageSize;//in KB, the size to split page into if sequence spans multiple devicePageSize pages

		//automatically filled based on device info (still can be reconfigured by user):
		uint32_t maxComputeWorkGroupCount[3]; // maxComputeWorkGroupCount from VkPhysicalDeviceLimits
		uint32_t maxComputeWorkGroupSize[3]; // maxComputeWorkGroupCount from VkPhysicalDeviceLimits
		uint32_t maxThreadsNum; //max number of threads from VkPhysicalDeviceLimits
		uint32_t sharedMemorySizeStatic; //available for static allocation shared memory size, in bytes
		uint32_t sharedMemorySize; //available for allocation shared memory size, in bytes
		uint32_t sharedMemorySizePow2; //power of 2 which is less or equal to sharedMemorySize, in bytes 
		uint32_t warpSize; //number of threads per warp/wavefront.
		uint32_t halfThreads;//Intel fix
		uint32_t allocateTempBuffer; //buffer allocated by app automatically if needed to reorder Four step algorithm. Parameter to check if it has been allocated
		uint32_t reorderFourStep; // unshuffle Four step algorithm. Requires tempbuffer allocation (0 - off, 1 - on). Default 1.

#if(VKFFT_BACKEND==0)
		VkDeviceMemory tempBufferDeviceMemory;//Filled at app creation
		VkCommandBuffer* commandBuffer;//Filled at app execution
		VkMemoryBarrier* memory_barrier;//Filled at app creation
#elif(VKFFT_BACKEND==1)
		cudaEvent_t* stream_event;//Filled at app creation
		uint32_t streamCounter;//Filled at app creation
		uint32_t streamID;//Filled at app creation
#elif(VKFFT_BACKEND==2)
		hipEvent_t* stream_event;//Filled at app creation
		uint32_t streamCounter;//Filled at app creation
		uint32_t streamID;//Filled at app creation
#endif
	} VkFFTConfiguration;
#if(VKFFT_BACKEND==0)
	//static VkFFTConfiguration defaultVkFFTConfiguration = { {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0},{0,0,0}, 0, 0,0,0,0,0,0,{0,0,0},{0,0,0},{0,0,0}, 0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0, 0,0 };
	//static VkFFTConfiguration defaultVkFFTConfiguration = { {1,1,1}, {1,1,1}, {1,1,1}, {1,1,1}, {65535,65535,65535},{1024,1024,64}, 1024, 1,1,1,1,1,0,{0,0,0},{0,0,0},{0,0,0}, 0,0,0,1,0,0,0,0,0,0, 0, 0, 0, 0, 32768,  32768, 32768, 32, 0, 1, 1, 0, 1, 32, 0, 1,1,1,1,1, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0, 0,0 };
#elif(VKFFT_BACKEND==1)
	//static VkFFTConfiguration defaultVkFFTConfiguration = { {1,1,1}, {1,1,1}, {1,1,1}, {1,1,1}, {65535,65535,65535},{1024,1024,64}, 1024, 1,1,1,1,1,0,{0,0,0},{0,0,0},{0,0,0}, 0,0,0,1,0,0,0,0,0,0, 0, 0, 0, 0, 32768,  32768, 32768, 32, 0, 1, 1, 0, 1, 32, 0, 1,1,1,1,1, 0,0,0,0,0, 0,0,0,0,0,0,0, 0,0, 1, 0, 0 };
#elif(VKFFT_BACKEND==2)
	//static VkFFTConfiguration defaultVkFFTConfiguration = { {1,1,1}, {1,1,1}, {1,1,1}, {1,1,1}, {65535,65535,65535},{1024,1024,64}, 1024, 1,1,1,1,1,0,{0,0,0},{0,0,0},{0,0,0}, 0,0,0,1,0,0,0,0,0,0, 0, 0, 0, 0, 32768,  32768, 32768, 32, 0, 1, 1, 0, 1, 32, 0, 1,1,1,1,1, 0,0,0,0,0, 0,0,0,0,0,0,0, 0,0, 1, 0,0 };
#endif
	typedef struct {
		uint32_t size[3];
		uint32_t localSize[3];
		uint32_t fftDim;
		uint32_t inverse;
		uint32_t zeropad[2];
		uint32_t axis_id;
		uint32_t axis_upload_id;
		uint32_t registers_per_thread;
		uint32_t registers_per_thread_per_radix[14];
		uint32_t min_registers_per_thread;
		uint32_t readToRegisters;
		uint32_t writeFromRegisters;
		uint32_t LUT;
		uint32_t performR2C;
		uint32_t frequencyZeropadding;
		uint32_t performZeropaddingFull[3]; // don't do read/write if full sequence is omitted
		uint32_t performZeropaddingInput[3]; // don't read if input is zeropadded (0 - off, 1 - on)
		uint32_t performZeropaddingOutput[3]; // don't write if output is zeropadded (0 - off, 1 - on)
		uint32_t fft_zeropad_left_full[3];
		uint32_t fft_zeropad_left_read[3];
		uint32_t fft_zeropad_left_write[3];
		uint32_t fft_zeropad_right_full[3];
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
		uint32_t reorderFourStep;
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
		uint32_t usedSharedMemory;
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
		uint32_t registerBoost;
		uint32_t warpSize;
		uint32_t numSharedBanks;
		uint32_t resolveBankConflictFirstStages;
		uint32_t sharedStrideBankConflictFirstStages;
		uint32_t sharedStrideReadWriteConflict;
		uint32_t maxSharedStride;

		char** regIDs;
		char* disableThreadsStart;
		char* disableThreadsEnd;
		char sdataID[50];
		char inoutID[50];
		char combinedID[50];
		char gl_LocalInvocationID_x[50];
		char gl_LocalInvocationID_y[50];
		char gl_LocalInvocationID_z[50];
		char gl_GlobalInvocationID_x[200];
		char gl_GlobalInvocationID_y[200];
		char gl_GlobalInvocationID_z[200];
		char tshuffle[50];
		char sharedStride[50];
		char gl_WorkGroupSize_x[50];
		char gl_WorkGroupSize_y[50];
		char gl_WorkGroupSize_z[50];
		char gl_WorkGroupID_x[50];
		char gl_WorkGroupID_y[50];
		char gl_WorkGroupID_z[50];
		char tempReg[50];
		char stageInvocationID[50];
		char blockInvocationID[50];
		char tempStr[200];
		char temp[10];
		char w[10];
		char iw[10];
		char locID[13][40];
		char* output;
		uint32_t currentLen;
	} VkFFTSpecializationConstantsLayout;
#if(VKFFT_BACKEND==0)
	typedef struct {
		uint32_t coordinate;
		uint32_t batch;
		uint32_t workGroupShift[3];
	} VkFFTPushConstantsLayout;
#elif(VKFFT_BACKEND==1)
	typedef struct {
		uint32_t coordinate;
		uint32_t batch;
		uint32_t workGroupShift[3];
	} VkFFTPushConstantsLayout;
#elif(VKFFT_BACKEND==2)
	typedef struct {
		uint32_t coordinate;
		uint32_t batch;
		uint32_t workGroupShift[3];
	} VkFFTPushConstantsLayout;
#endif
	typedef struct {
		uint32_t localSize[3];
		uint32_t inputStride[5];
		uint32_t ratio;
		uint32_t ratioDirection;
	} VkFFTTransposeSpecializationConstantsLayout;
	typedef struct {
		uint32_t numBindings;
		uint32_t axisBlock[4];
		uint32_t groupedBatch;
		VkFFTSpecializationConstantsLayout specializationConstants;
		VkFFTPushConstantsLayout pushConstants;
#if(VKFFT_BACKEND==0)
		VkBuffer* inputBuffer;
		VkBuffer* outputBuffer;
		VkDescriptorPool descriptorPool;
		VkDescriptorSetLayout descriptorSetLayout;
		VkDescriptorSet descriptorSet;
		VkPipelineLayout pipelineLayout;
		VkPipeline pipeline;
		VkDeviceMemory bufferLUTDeviceMemory;
		VkBuffer bufferLUT;
#elif(VKFFT_BACKEND==1)
		void** inputBuffer;
		void** outputBuffer;
		CUmodule VkFFTModule;
		CUfunction VkFFTKernel;
		void* bufferLUT;
		CUdeviceptr consts_addr;
#elif(VKFFT_BACKEND==2)
		void** inputBuffer;
		void** outputBuffer;
		hipModule_t VkFFTModule;
		hipFunction_t VkFFTKernel;
		void* bufferLUT;
		hipDeviceptr_t consts_addr;
#endif
		uint64_t bufferLUTSize;
		uint32_t referenceLUT;
	} VkFFTAxis;

	typedef struct {
		uint32_t numAxisUploads[3];
		uint32_t axisSplit[3][4];
		VkFFTAxis axes[3][4];
	} VkFFTPlan;
	typedef struct {
		VkFFTConfiguration configuration;
		VkFFTPlan* localFFTPlan;
		VkFFTPlan* localFFTPlan_inverse; //additional inverse plan
	} VkFFTApplication;
	static VkFFTApplication defaultVkFFTApplication = { {}, {}, {} };

	static inline void appendLicense(char* output) {
		sprintf(output + strlen(output), "\
// This file is part of VkFFT, a Vulkan Fast Fourier Transform library\n\
//\n\
// Copyright (C) 2021 Dmitrii Tolmachev <dtolm96@gmail.com>\n\
//\n\
// This Source Code Form is subject to the terms of the Mozilla Public\n\
// License, v. 2.0. If a copy of the MPL was not distributed with this\n\
// file, You can obtain one at https://mozilla.org/MPL/2.0/. \n");
	}
	static inline void VkAppendLine(VkFFTSpecializationConstantsLayout* sc, const char* in) {
		sprintf(sc->output + strlen(sc->output), "%s", in);
		//sprintf(sc->output + sc->currentLen, "%s", in);
		//sc->currentLen += strlen(in);
	};
	static inline void VkMovComplex(VkFFTSpecializationConstantsLayout* sc, const char* out, const char* in) {
		sprintf(sc->tempStr, "\
	%s = %s;\n", out, in);
		VkAppendLine(sc, sc->tempStr);
	};
	static inline void VkMovReal(VkFFTSpecializationConstantsLayout* sc, const char* out, const char* in) {
		sprintf(sc->tempStr, "\
	%s = %s;\n", out, in);
		VkAppendLine(sc, sc->tempStr);
	};
	static inline void VkSharedStore(VkFFTSpecializationConstantsLayout* sc, const char* id, const char* in) {
		sprintf(sc->tempStr, "\
	sdata[%s] = %s;\n", id, in);
		VkAppendLine(sc, sc->tempStr);
	};
	static inline void VkSharedLoad(VkFFTSpecializationConstantsLayout* sc, const char* out, const char* id) {
		sprintf(sc->tempStr, "\
	%s = sdata[%s];\n", out, id);
		VkAppendLine(sc, sc->tempStr);
	};
	static inline void VkAddReal(VkFFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_2) {
		sprintf(sc->tempStr, "\
	%s = %s + %s;\n", out, in_1, in_2);
		VkAppendLine(sc, sc->tempStr);
	};
	static inline void VkAddComplex(VkFFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_2) {
		sprintf(sc->tempStr, "\
	%s.x = %s.x + %s.x;\n\
	%s.y = %s.y + %s.y;\n", out, in_1, in_2, out, in_1, in_2);
		VkAppendLine(sc, sc->tempStr);
	};
	static inline void VkAddComplexInv(VkFFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_2) {
		sprintf(sc->tempStr, "\
	%s.x = - %s.x - %s.x;\n\
	%s.y = - %s.y - %s.y;\n", out, in_1, in_2, out, in_1, in_2);
		VkAppendLine(sc, sc->tempStr);
	};
	static inline void VkSubComplex(VkFFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_2) {
		sprintf(sc->tempStr, "\
	%s.x = %s.x - %s.x;\n\
	%s.y = %s.y - %s.y;\n", out, in_1, in_2, out, in_1, in_2);
		VkAppendLine(sc, sc->tempStr);
	};
	static inline void VkSubReal(VkFFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_2) {
		sprintf(sc->tempStr, "\
	%s = %s - %s;\n", out, in_1, in_2);
		VkAppendLine(sc, sc->tempStr);
	};
	static inline void VkFMAComplex(VkFFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_num, const char* in_2) {
		sprintf(sc->tempStr, "\
	%s.x = fma(%s.x, %s, %s.x);\n\
	%s.y = fma(%s.y, %s, %s.y);\n", out, in_1, in_num, in_2, out, in_1, in_num, in_2);
		VkAppendLine(sc, sc->tempStr);
	};
	static inline void VkFMAReal(VkFFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_num, const char* in_2) {
		sprintf(sc->tempStr, "\
	%s = fma(%s, %s, %s);\n", out, in_1, in_num, in_2);
		VkAppendLine(sc, sc->tempStr);
	};
	static inline void VkMulComplex(VkFFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_2, const char* temp) {
		if (strcmp(out, in_1) && strcmp(out, in_2)) {
			sprintf(sc->tempStr, "\
	%s.x = %s.x * %s.x - %s.y * %s.y;\n\
	%s.y = %s.y * %s.x + %s.x * %s.y;\n", out, in_1, in_2, in_1, in_2, out, in_1, in_2, in_1, in_2);
		}
		else {
			sprintf(sc->tempStr, "\
	%s.x = %s.x * %s.x - %s.y * %s.y;\n\
	%s.y = %s.y * %s.x + %s.x * %s.y;\n\
	%s = %s;\n", temp, in_1, in_2, in_1, in_2, temp, in_1, in_2, in_1, in_2, out, temp);
		}
		VkAppendLine(sc, sc->tempStr);
	};
	static inline void VkMulComplexConj(VkFFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_2, const char* temp) {
		if (strcmp(out, in_1) && strcmp(out, in_2)) {
			sprintf(sc->tempStr, "\
	%s.x = %s.x * %s.x + %s.y * %s.y;\n\
	%s.y = %s.y * %s.x - %s.x * %s.y;\n", out, in_1, in_2, in_1, in_2, out, in_1, in_2, in_1, in_2);
		}
		else {
			sprintf(sc->tempStr, "\
	%s.x = %s.x * %s.x + %s.y * %s.y;\n\
	%s.y = %s.y * %s.x - %s.x * %s.y;\n\
	%s = %s;\n", temp, in_1, in_2, in_1, in_2, temp, in_1, in_2, in_1, in_2, out, temp);
		}
		VkAppendLine(sc, sc->tempStr);
	};
	static inline void VkMulComplexNumber(VkFFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_num) {
		sprintf(sc->tempStr, "\
	%s.x = %s.x * %s;\n\
	%s.y = %s.y * %s;\n", out, in_1, in_num, out, in_1, in_num);
		VkAppendLine(sc, sc->tempStr);
	};
	static inline void VkMulComplexNumberImag(VkFFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_num, const char* temp) {
		if (strcmp(out, in_1)) {
			sprintf(sc->tempStr, "\
	%s.x = - %s.y * %s;\n\
	%s.y = %s.x * %s;\n", out, in_1, in_num, out, in_1, in_num);
		}
		else {
			sprintf(sc->tempStr, "\
	%s.x = - %s.y * %s;\n\
	%s.y = %s.x * %s;\n\
	%s = %s;\n", temp, in_1, in_num, temp, in_1, in_num, out, temp);
		}
		VkAppendLine(sc, sc->tempStr);
	};
	static inline void VkDivComplexNumber(VkFFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_num) {
		sprintf(sc->tempStr, "\
	%s.x = %s.x / %s;\n\
	%s.y = %s.y / %s;\n", out, in_1, in_num, out, in_1, in_num);
		VkAppendLine(sc, sc->tempStr);
	};

	static inline void VkMulReal(VkFFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_2) {
		sprintf(sc->tempStr, "\
	%s = %s * %s;\n", out, in_1, in_2);
		VkAppendLine(sc, sc->tempStr);
	};

	static inline void VkShuffleComplex(VkFFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_2, const char* temp) {
		if (strcmp(out, in_2)) {
			sprintf(sc->tempStr, "\
	%s.x = %s.x - %s.y;\n\
	%s.y = %s.y + %s.x;\n", out, in_1, in_2, out, in_1, in_2);
		}
		else {
			sprintf(sc->tempStr, "\
	%s.x = %s.x - %s.y;\n\
	%s.y = %s.x + %s.y;\n\
	%s = %s;\n", temp, in_1, in_2, temp, in_1, in_2, out, temp);
		}
		VkAppendLine(sc, sc->tempStr);
	};
	static inline void VkShuffleComplexInv(VkFFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_2, const char* temp) {
		if (strcmp(out, in_2)) {
			sprintf(sc->tempStr, "\
	%s.x = %s.x + %s.y;\n\
	%s.y = %s.y - %s.x;\n", out, in_1, in_2, out, in_1, in_2);
		}
		else {
			sprintf(sc->tempStr, "\
	%s.x = %s.x + %s.y;\n\
	%s.y = %s.x - %s.y;\n\
	%s = %s;\n", temp, in_1, in_2, temp, in_1, in_2, out, temp);
		}
		VkAppendLine(sc, sc->tempStr);
	};
	static inline void VkModReal(VkFFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_num) {
		sprintf(sc->tempStr, "\
	%s = %s %% %s;\n", out, in_1, in_num);
		VkAppendLine(sc, sc->tempStr);
	};
	static inline void VkDivReal(VkFFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_num) {
		sprintf(sc->tempStr, "\
	%s = %s / %s;\n", out, in_1, in_num);
		VkAppendLine(sc, sc->tempStr);
	};
	static inline void VkPermute(VkFFTSpecializationConstantsLayout* sc, const uint32_t* permute, const uint32_t num_elem, const uint32_t type, char ** regIDs) {
		char temp_ID[13][20];
		if (type == 0) {
			for (uint32_t i = 0; i < num_elem; i++)
				sprintf(temp_ID[i], "%s", sc->locID[i]);
			for (uint32_t i = 0; i < num_elem; i++)
				sprintf(sc->locID[i], "%s", temp_ID[permute[i]]);
		}
		if (type == 1) {
			for (uint32_t i = 0; i < num_elem; i++)
				sprintf(temp_ID[i], "%s", regIDs[i]);
			for (uint32_t i = 0; i < num_elem; i++)
				sprintf(regIDs[i], "%s", temp_ID[permute[i]]);
		}
	};

	static inline void appendVersion(char* output) {
#if(VKFFT_BACKEND==0)
		sprintf(output, "#version 450\n\n");
#elif(VKFFT_BACKEND==1)
		sprintf(output, "");
#elif(VKFFT_BACKEND==2)
		sprintf(output, "");
#endif
	}
	static inline void appendExtensions(char* output, const char* floatType, const char* floatTypeInputMemory, const char* floatTypeOutputMemory, const char* floatTypeKernelMemory) {
#if(VKFFT_BACKEND==0)
		if (!strcmp(floatType, "double"))
			sprintf(output + strlen(output), "\
#extension GL_ARB_gpu_shader_fp64 : enable\n\
#extension GL_ARB_gpu_shader_int64 : enable\n\n");
		if ((!strcmp(floatTypeInputMemory, "half")) || (!strcmp(floatTypeOutputMemory, "half")) || (!strcmp(floatTypeKernelMemory, "half")))
			sprintf(output + strlen(output), "#extension GL_EXT_shader_16bit_storage : require\n\n");
#elif(VKFFT_BACKEND==1)
		//sprintf(output + strlen(output), "\
#include <stdint.h>\n");
#elif(VKFFT_BACKEND==2)
		sprintf(output + strlen(output), "\
#include <hip/hip_runtime.h>\n");

		/*if (!strcmp(floatType, "float")) {
			sprintf(output + strlen(output), "\
typedef struct\n\
{\n\
	float x, y;\n\
} float2;\n");
		}
		if (!strcmp(floatType, "double")) {
			sprintf(output + strlen(output), "\
typedef struct\n\
{\n\
	double x, y;\n\
} double2;\n");
		}*/
#endif
	}
	static inline void appendLayoutVkFFT(char* output, VkFFTSpecializationConstantsLayout* sc) {
#if(VKFFT_BACKEND==0)
		sprintf(output + strlen(output), "layout (local_size_x = %d, local_size_y = %d, local_size_z = %d) in;\n", sc->localSize[0], sc->localSize[1], sc->localSize[2]);
#elif(VKFFT_BACKEND==1)
#elif(VKFFT_BACKEND==2)
#endif
	}
	static inline void appendConstant(char* output, const char* type, const char* name, const char* defaultVal, const char* LFending) {
		sprintf(output + strlen(output), "const %s %s = %s%s;\n", type, name, defaultVal, LFending);
	}
	static inline void appendPushConstant(char* output, const char* type, const char* name) {
		sprintf(output + strlen(output), "	%s %s;\n", type, name);
	}
	static inline void appendBarrierVkFFT(char* output, uint32_t numTab) {
		char tabs[100];
		for (uint32_t i = 0; i < numTab; i++)
			sprintf(tabs, "	");
#if(VKFFT_BACKEND==0)
		sprintf(output + strlen(output), "%sbarrier();\n\n", tabs);
#elif(VKFFT_BACKEND==1)
		sprintf(output + strlen(output), "%s__syncthreads();\n\n", tabs);
#elif(VKFFT_BACKEND==2)
		sprintf(output + strlen(output), "%s__syncthreads();\n\n", tabs);
#endif
	}
	static inline void appendPushConstantsVkFFT(char* output, VkFFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType) {
#if(VKFFT_BACKEND==0)
		sprintf(output + strlen(output), "layout(push_constant) uniform PushConsts\n{\n");
		appendPushConstant(output, uintType, "coordinate");
		appendPushConstant(output, uintType, "batchID");
		appendPushConstant(output, uintType, "workGroupShiftX");
		appendPushConstant(output, uintType, "workGroupShiftY");
		appendPushConstant(output, uintType, "workGroupShiftZ");
		sprintf(output + strlen(output), "} consts;\n\n");
#elif(VKFFT_BACKEND==1)
		VkAppendLine(sc, "	typedef struct {\n");
		appendPushConstant(output, uintType, "coordinate");
		appendPushConstant(output, uintType, "batchID");
		appendPushConstant(output, uintType, "workGroupShiftX");
		appendPushConstant(output, uintType, "workGroupShiftY");
		appendPushConstant(output, uintType, "workGroupShiftZ");
		VkAppendLine(sc, "	}PushConsts;\n");
		VkAppendLine(sc, "	__constant__ PushConsts consts;\n");
#elif(VKFFT_BACKEND==2)
		VkAppendLine(sc, "	typedef struct {\n");
		appendPushConstant(output, uintType, "coordinate");
		appendPushConstant(output, uintType, "batchID");
		appendPushConstant(output, uintType, "workGroupShiftX");
		appendPushConstant(output, uintType, "workGroupShiftY");
		appendPushConstant(output, uintType, "workGroupShiftZ");
		VkAppendLine(sc, "	}PushConsts;\n");
		VkAppendLine(sc, "	__constant__ PushConsts consts;\n");
#endif
	}
	static inline void appendConstantsVkFFT(char* output, const char* floatType, const char* uintType) {
		char LFending[4] = "";
		if (!strcmp(floatType, "float")) sprintf(LFending, "f");
#if(VKFFT_BACKEND==0)
		if (!strcmp(floatType, "double")) sprintf(LFending, "LF");
#elif(VKFFT_BACKEND==1)
		if (!strcmp(floatType, "double")) sprintf(LFending, "l");
#elif(VKFFT_BACKEND==2)
		if (!strcmp(floatType, "double")) sprintf(LFending, "l");
#endif
#if(VKFFT_BACKEND==0)
		appendConstant(output, floatType, "loc_PI", "3.1415926535897932384626433832795", LFending);
		appendConstant(output, floatType, "loc_SQRT1_2", "0.70710678118654752440084436210485", LFending);
#elif(VKFFT_BACKEND==1)
		appendConstant(output, floatType, "loc_PI", "3.1415926535897932384626433832795", LFending);
		appendConstant(output, floatType, "loc_SQRT1_2", "0.70710678118654752440084436210485", LFending);
#elif(VKFFT_BACKEND==2)
		appendConstant(output, floatType, "loc_PI", "3.1415926535897932384626433832795", LFending);
		appendConstant(output, floatType, "loc_SQRT1_2", "0.70710678118654752440084436210485", LFending);
#endif
	}
	static inline void appendSinCos20(char* output, const char* floatType, const char* uintType) {
		char functionDefinitions[100] = "";
		char vecType[30];
		char LFending[4] = "";
		if (!strcmp(floatType, "float")) sprintf(LFending, "f");
#if(VKFFT_BACKEND==0)
		if (!strcmp(floatType, "half")) sprintf(vecType, "f16vec2");
		if (!strcmp(floatType, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "dvec2");
		if (!strcmp(floatType, "double")) sprintf(LFending, "LF");
#elif(VKFFT_BACKEND==1)
		if (!strcmp(floatType, "half")) sprintf(vecType, "f16vec2");
		if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
		if (!strcmp(floatType, "double")) sprintf(LFending, "l");
		sprintf(functionDefinitions, "__device__ static __inline__ ");
#elif(VKFFT_BACKEND==2)
		if (!strcmp(floatType, "half")) sprintf(vecType, "f16vec2");
		if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
		if (!strcmp(floatType, "double")) sprintf(LFending, "l");
		sprintf(functionDefinitions, "__device__ static __inline__ ");
#endif
		appendConstant(output, floatType, "loc_2_PI", "0.63661977236758134307553505349006", LFending);
		appendConstant(output, floatType, "loc_PI_2", "1.5707963267948966192313216916398", LFending);
		appendConstant(output, floatType, "a1", "0.99999999999999999999962122687403772", LFending);
		appendConstant(output, floatType, "a3", "-0.166666666666666666637194166219637268", LFending);
		appendConstant(output, floatType, "a5", "0.00833333333333333295212653322266277182", LFending);
		appendConstant(output, floatType, "a7", "-0.000198412698412696489459896530659927773", LFending);
		appendConstant(output, floatType, "a9", "2.75573192239364018847578909205399262e-6", LFending);
		appendConstant(output, floatType, "a11", "-2.50521083781017605729370231280411712e-8", LFending);
		appendConstant(output, floatType, "a13", "1.60590431721336942356660057796782021e-10", LFending);
		appendConstant(output, floatType, "a15", "-7.64712637907716970380859898835680587e-13", LFending);
		appendConstant(output, floatType, "a17", "2.81018528153898622636194976499656274e-15", LFending);
		appendConstant(output, floatType, "ab", "-7.97989713648499642889739108679114937e-18", LFending);
		sprintf(output + strlen(output), "\
%s%s sincos_20(double x)\n\
{\n\
	//minimax coefs for sin for 0..pi/2 range\n\
	double y = abs(x * loc_2_PI);\n\
	double q = floor(y);\n\
	int quadrant = int(q);\n\
	double t = (quadrant & 1) != 0 ? 1 - y + q : y - q;\n\
	t *= loc_PI_2;\n\
	double t2 = t * t;\n\
	double r = fma(fma(fma(fma(fma(fma(fma(fma(fma(ab, t2, a17), t2, a15), t2, a13), t2, a11), t2, a9), t2, a7), t2, a5), t2, a3), t2 * t, t);\n\
	%s cos_sin;\n\
	cos_sin.x = ((quadrant == 0) || (quadrant == 3)) ? sqrt(1 - r * r) : -sqrt(1 - r * r);\n\
	r = x < 0 ? -r : r;\n\
	cos_sin.y = (quadrant & 2) != 0 ? -r : r;\n\
	return cos_sin;\n\
}\n\n", functionDefinitions, vecType, vecType);
	}
	static inline void appendInputLayoutVkFFT(char* output, VkFFTSpecializationConstantsLayout* sc, uint32_t id, const char* floatTypeMemory, uint32_t inputType) {
		char vecType[30];
		switch (inputType) {
		case 0: case 1: case 2: case 3: case 4: case 6: {
#if(VKFFT_BACKEND==0)
			if (!strcmp(floatTypeMemory, "half")) sprintf(vecType, "f16vec2");
			if (!strcmp(floatTypeMemory, "float")) sprintf(vecType, "vec2");
			if (!strcmp(floatTypeMemory, "double")) sprintf(vecType, "dvec2");
			if (sc->inputBufferBlockNum == 1)
				sprintf(output + strlen(output), "\
layout(std430, binding = %d) buffer DataIn{\n\
	%s inputs[%d];\n\
};\n\n", id, vecType, sc->inputBufferBlockSize);
			else
				sprintf(output + strlen(output), "\
layout(std430, binding = %d) buffer DataIn{\n\
	%s inputs[%d];\n\
} inputBlocks[%d];\n\n", id, vecType, sc->inputBufferBlockSize, sc->inputBufferBlockNum);
#elif(VKFFT_BACKEND==1)
			if (!strcmp(floatTypeMemory, "half")) sprintf(vecType, "f16vec2");
			if (!strcmp(floatTypeMemory, "float")) sprintf(vecType, "float2");
			if (!strcmp(floatTypeMemory, "double")) sprintf(vecType, "double2");
#elif(VKFFT_BACKEND==2)
			if (!strcmp(floatTypeMemory, "half")) sprintf(vecType, "f16vec2");
			if (!strcmp(floatTypeMemory, "float")) sprintf(vecType, "float2");
			if (!strcmp(floatTypeMemory, "double")) sprintf(vecType, "double2");
#endif
			break;
		}
		case 5:
		{
			if (!strcmp(floatTypeMemory, "half")) sprintf(vecType, "float16_t");
			if (!strcmp(floatTypeMemory, "float")) sprintf(vecType, "float");
			if (!strcmp(floatTypeMemory, "double")) sprintf(vecType, "double");
#if(VKFFT_BACKEND==0)
			if (sc->inputBufferBlockNum == 1)
				sprintf(output + strlen(output), "\
layout(std430, binding = %d) buffer DataIn{\n\
	%s inputs[%d];\n\
};\n\n", id, vecType, 2 * sc->inputBufferBlockSize);
			else
				sprintf(output + strlen(output), "\
layout(std430, binding = %d) buffer DataIn{\n\
	%s inputs[%d];\n\
} inputBlocks[%d];\n\n", id, vecType, 2 * sc->inputBufferBlockSize, sc->inputBufferBlockNum);
#endif
			break;
		}
		}


	}
	static inline void appendOutputLayoutVkFFT(char* output, VkFFTSpecializationConstantsLayout* sc, uint32_t id, const char* floatTypeMemory, uint32_t outputType) {
		char vecType[30];
		switch (outputType) {
		case 0: case 1: case 2: case 3: case 4: case 5: {
#if(VKFFT_BACKEND==0)
			if (!strcmp(floatTypeMemory, "half")) sprintf(vecType, "f16vec2");
			if (!strcmp(floatTypeMemory, "float")) sprintf(vecType, "vec2");
			if (!strcmp(floatTypeMemory, "double")) sprintf(vecType, "dvec2");
			if (sc->outputBufferBlockNum == 1)
				sprintf(output + strlen(output), "\
layout(std430, binding = %d) buffer DataOut{\n\
	%s outputs[%d];\n\
};\n\n", id, vecType, sc->outputBufferBlockSize);
			else
				sprintf(output + strlen(output), "\
layout(std430, binding = %d) buffer DataOut{\n\
	%s outputs[%d];\n\
} outputBlocks[%d];\n\n", id, vecType, sc->outputBufferBlockSize, sc->outputBufferBlockNum);
#elif(VKFFT_BACKEND==1)
			if (!strcmp(floatTypeMemory, "half")) sprintf(vecType, "f16vec2");
			if (!strcmp(floatTypeMemory, "float")) sprintf(vecType, "float2");
			if (!strcmp(floatTypeMemory, "double")) sprintf(vecType, "double2");
#elif(VKFFT_BACKEND==2)
			if (!strcmp(floatTypeMemory, "half")) sprintf(vecType, "f16vec2");
			if (!strcmp(floatTypeMemory, "float")) sprintf(vecType, "float2");
			if (!strcmp(floatTypeMemory, "double")) sprintf(vecType, "double2");
#endif
			break;
		}
		case 6:
		{
			if (!strcmp(floatTypeMemory, "half")) sprintf(vecType, "float16_t");
			if (!strcmp(floatTypeMemory, "float")) sprintf(vecType, "float");
			if (!strcmp(floatTypeMemory, "double")) sprintf(vecType, "double");
#if(VKFFT_BACKEND==0)
			if (sc->outputBufferBlockNum == 1)
				sprintf(output + strlen(output), "\
layout(std430, binding = %d) buffer DataOut{\n\
	%s outputs[%d];\n\
};\n\n", id, vecType, 2 * sc->outputBufferBlockSize);
			else
				sprintf(output + strlen(output), "\
layout(std430, binding = %d) buffer DataOut{\n\
	%s outputs[%d];\n\
} outputBlocks[%d];\n\n", id, vecType, 2 * sc->outputBufferBlockSize, sc->outputBufferBlockNum);
#endif
			break;
		}
		}
	}
	static inline void appendKernelLayoutVkFFT(char* output, VkFFTSpecializationConstantsLayout* sc, uint32_t id, const char* floatTypeMemory) {
		char vecType[30];
#if(VKFFT_BACKEND==0)
		if (!strcmp(floatTypeMemory, "half")) sprintf(vecType, "f16vec2");
		if (!strcmp(floatTypeMemory, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatTypeMemory, "double")) sprintf(vecType, "dvec2");
		if (sc->kernelBlockNum == 1)
			sprintf(output + strlen(output), "\
layout(std430, binding = %d) buffer Kernel_FFT{\n\
	%s kernel[%d];\n\
};\n\n", id, vecType, sc->kernelBlockSize);
		else
			sprintf(output + strlen(output), "\
layout(std430, binding = %d) buffer Kernel_FFT{\n\
	%s kernel[%d];\n\
} kernelBlocks[%d];\n\n", id, vecType, sc->kernelBlockSize, sc->kernelBlockNum);
#elif(VKFFT_BACKEND==1)
		if (!strcmp(floatTypeMemory, "half")) sprintf(vecType, "f16vec2");
		if (!strcmp(floatTypeMemory, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatTypeMemory, "double")) sprintf(vecType, "double2");
#elif(VKFFT_BACKEND==2)
		if (!strcmp(floatTypeMemory, "half")) sprintf(vecType, "f16vec2");
		if (!strcmp(floatTypeMemory, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatTypeMemory, "double")) sprintf(vecType, "double2");
#endif


	}
	static inline void appendLUTLayoutVkFFT(char* output, VkFFTSpecializationConstantsLayout* sc, uint32_t id, const char* floatType) {
		char vecType[30];
#if(VKFFT_BACKEND==0)
		if (!strcmp(floatType, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "dvec2");
		sprintf(output + strlen(output), "\
layout(std430, binding = %d) readonly buffer DataLUT {\n\
%s twiddleLUT[];\n\
};\n", id, vecType);
#elif(VKFFT_BACKEND==1)
		if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
#elif(VKFFT_BACKEND==2)
		if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
#endif

	}
	static inline void indexInputVkFFT(char* output, VkFFTSpecializationConstantsLayout* sc, const char* uintType, uint32_t inputType, const char* index_x, const char* index_y, const char* coordinate, const char* batchID) {
		char functionDefinitions[100] = "";
#if(VKFFT_BACKEND==0)
#elif(VKFFT_BACKEND==1)
		sprintf(functionDefinitions, "__device__ static __inline__ ");
#elif(VKFFT_BACKEND==2)
		sprintf(functionDefinitions, "__device__ static __inline__ ");
#endif
		switch (inputType) {
		case 0: case 2: case 3: case 4: {//single_c2c + single_c2c_strided
			char inputOffset[30] = "";
			if (sc->inputOffset > 0)
				sprintf(inputOffset, "%d + ", sc->inputOffset);
			char shiftX[500] = "";
			if (sc->inputStride[0] == 1)
				sprintf(shiftX, "(%s)", index_x);
			else
				sprintf(shiftX, "(%s) * %d", index_x, sc->inputStride[0]);
			char shiftY[500] = "";
			if (sc->size[1] > 1) {
				if (sc->fftDim == sc->fft_dim_full) {
					if (sc->performWorkGroupShift[1])
						sprintf(shiftY, " + (%s + consts.workGroupShiftY) * %d", sc->gl_WorkGroupID_y, sc->localSize[1] * sc->inputStride[1]);
					else
						sprintf(shiftY, " + %s * %d", sc->gl_WorkGroupID_y, sc->localSize[1] * sc->inputStride[1]);
				}
				else {
					if (sc->performWorkGroupShift[1])
						sprintf(shiftY, " + (%s + consts.workGroupShiftY) * %d", sc->gl_WorkGroupID_y, sc->inputStride[1]);
					else
						sprintf(shiftY, " + %s * %d", sc->gl_WorkGroupID_y, sc->inputStride[1]);
				}
			}
			char shiftZ[500] = "";
			if (sc->size[2] > 1) {
				if (sc->performWorkGroupShift[2])
					sprintf(shiftZ, " + (%s + consts.workGroupShiftZ * %s) * %d", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z, sc->inputStride[2]);
				else
					sprintf(shiftZ, " + %s * %d", sc->gl_GlobalInvocationID_z, sc->inputStride[2]);
			}
			char shiftCoordinate[100] = "";
			char requestCoordinate[100] = "";
			if (sc->numCoordinates * sc->matrixConvolution > 1) {
				sprintf(shiftCoordinate, " + consts.coordinate * %d", sc->inputStride[3]);
			}
			if ((sc->matrixConvolution > 1) && (sc->convolutionStep)) {
				sprintf(shiftCoordinate, " + %s * %d", coordinate, sc->inputStride[3]);
				//sprintf(requestCoordinate, ", %s coordinate", uintType);
			}
			char shiftBatch[100] = "";
			char requestBatch[100] = "";
			if ((sc->numBatches > 1) || (sc->numKernels > 1)) {
				if (sc->convolutionStep) {
					sprintf(shiftBatch, " + %s * %d", batchID, sc->inputStride[4]);
					//sprintf(requestBatch, ", %s batchID", uintType);
				}
				else
					sprintf(shiftBatch, " + consts.batchID * %d", sc->inputStride[4]);
			}
			//sprintf(output + strlen(output), "\
%s%s indexInput(%s index%s%s) {\n\
	return %s%s%s%s%s%s;\n\
}\n\n", functionDefinitions, uintType, uintType, requestCoordinate, requestBatch, inputOffset, shiftX, shiftY, shiftZ, shiftCoordinate, shiftBatch);
			sprintf(output + strlen(output), "%s%s%s%s%s%s", inputOffset, shiftX, shiftY, shiftZ, shiftCoordinate, shiftBatch);
			break;
		}
		case 1: {//grouped_c2c
			char inputOffset[30] = "";
			if (sc->inputOffset > 0)
				sprintf(inputOffset, "%d + ", sc->inputOffset);
			char shiftX[500] = "";
			if (sc->inputStride[0] == 1)
				sprintf(shiftX, "(%s)", index_x);
			else
				sprintf(shiftX, "(%s) * %d", index_x, sc->inputStride[0]);

			char shiftY[500] = "";
			sprintf(shiftY, " + (%s) * %d", index_y, sc->inputStride[1]);

			char shiftZ[500] = "";
			if (sc->size[2] > 1) {
				if (sc->performWorkGroupShift[2])
					sprintf(shiftZ, " + (%s + consts.workGroupShiftZ * %s) * %d", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z, sc->inputStride[2]);
				else
					sprintf(shiftZ, " + %s * %d", sc->gl_GlobalInvocationID_z, sc->inputStride[2]);
			}
			char shiftCoordinate[100] = "";
			char requestCoordinate[100] = "";
			if (sc->numCoordinates * sc->matrixConvolution > 1) {
				sprintf(shiftCoordinate, " + consts.coordinate * %d", sc->outputStride[3]);
			}
			if ((sc->matrixConvolution > 1) && (sc->convolutionStep)) {
				sprintf(shiftCoordinate, " + %s * %d", coordinate, sc->inputStride[3]);
				//sprintf(requestCoordinate, ", %s coordinate", uintType);
			}
			char shiftBatch[100] = "";
			char requestBatch[100] = "";
			if ((sc->numBatches > 1) || (sc->numKernels > 1)) {
				if (sc->convolutionStep) {
					sprintf(shiftBatch, " + %s * %d", batchID, sc->inputStride[4]);
					//sprintf(requestBatch, ", %s batchID", uintType);
				}
				else
					sprintf(shiftBatch, " + consts.batchID * %d", sc->inputStride[4]);
			}
			sprintf(output + strlen(output), "%s%s%s%s%s%s", inputOffset, shiftX, shiftY, shiftZ, shiftCoordinate, shiftBatch);
			break;
		}
		case 5: {//single_r2c
			char inputOffset[30] = "";
			if (sc->inputOffset > 0)
				sprintf(inputOffset, "%d + ", sc->inputOffset);
			char shiftX[500] = "";
			if (sc->inputStride[0] == 1)
				sprintf(shiftX, "(%s)", index_x);
			else
				sprintf(shiftX, "(%s) * %d", index_x, sc->inputStride[0]);
			char shiftY[500] = "";
			if (sc->size[1] > 1) {
				if (sc->fftDim == sc->fft_dim_full) {
					if (sc->performWorkGroupShift[1])
						sprintf(shiftY, " + (%s + consts.workGroupShiftY) * %d", sc->gl_WorkGroupID_y, 2 * sc->localSize[1] * sc->inputStride[1]);
					else
						sprintf(shiftY, " + %s * %d", sc->gl_WorkGroupID_y, 2 * sc->localSize[1] * sc->inputStride[1]);
				}
				else {
					if (sc->performWorkGroupShift[1])
						sprintf(shiftY, " + (%s + consts.workGroupShiftY) * %d", sc->gl_WorkGroupID_y, 2 * sc->inputStride[1]);
					else
						sprintf(shiftY, " + %s * %d", sc->gl_WorkGroupID_y, 2 * sc->inputStride[1]);
				}
			}
			char shiftZ[500] = "";
			if (sc->size[2] > 1) {
				if (sc->performWorkGroupShift[2])
					sprintf(shiftZ, " + (%s + consts.workGroupShiftZ * %s) * %d", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z, sc->inputStride[2]);
				else
					sprintf(shiftZ, " + %s * %d", sc->gl_GlobalInvocationID_z,  sc->inputStride[2]);
			}
			char shiftCoordinate[100] = "";
			if (sc->numCoordinates * sc->matrixConvolution > 1) {
				sprintf(shiftCoordinate, " + consts.coordinate * %d", sc->inputStride[3]);
			}
			char shiftBatch[100] = "";
			if ((sc->numBatches > 1) || (sc->numKernels > 1)) {
				sprintf(shiftBatch, " + consts.batchID * %d", sc->inputStride[4]);
			}
			sprintf(output + strlen(output), "%s%s%s%s%s%s", inputOffset, shiftX, shiftY, shiftZ, shiftCoordinate, shiftBatch);
			break;
		}
		case 6: {//single_c2r
			char inputOffset[30] = "";
			if (sc->inputOffset > 0)
				sprintf(inputOffset, "%d + ", sc->inputOffset);
			char shiftX[500] = "";
			if (sc->inputStride[0] == 1)
				sprintf(shiftX, "(%s)", index_x);
			else
				sprintf(shiftX, "(%s) * %d", index_x, sc->inputStride[0]);
			char shiftY[500] = "";
			sprintf(shiftY, " + (%s) * %d", index_y, sc->inputStride[1]);
			char shiftZ[500] = "";
			if (sc->size[2] > 1) {
				if (sc->performWorkGroupShift[2])
					sprintf(shiftZ, " + (%s + consts.workGroupShiftZ * %s) * %d", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z, sc->inputStride[2]);
				else
					sprintf(shiftZ, " + %s * %d", sc->gl_GlobalInvocationID_z, sc->inputStride[2]);
			}
			char shiftCoordinate[100] = "";
			if (sc->numCoordinates * sc->matrixConvolution > 1) {
				sprintf(shiftCoordinate, " + consts.coordinate * %d", sc->inputStride[3]);
			}
			char shiftBatch[100] = "";
			if ((sc->numBatches > 1) || (sc->numKernels > 1)) {
				sprintf(shiftBatch, " + consts.batchID * %d", sc->inputStride[4]);
			}
			sprintf(output + strlen(output), "%s%s%s%s%s%s", inputOffset, shiftX, shiftY, shiftZ, shiftCoordinate, shiftBatch);
			break;
		}
		}
	}
	static inline void indexOutputVkFFT(char* output, VkFFTSpecializationConstantsLayout* sc, const char* uintType, uint32_t outputType, const char* index_x, const char* index_y, const char* coordinate, const char* batchID) {
		char functionDefinitions[100] = "";
#if(VKFFT_BACKEND==0)
#elif(VKFFT_BACKEND==1)
		sprintf(functionDefinitions, "__device__ static __inline__ ");
#elif(VKFFT_BACKEND==2)
		sprintf(functionDefinitions, "__device__ static __inline__ ");
#endif
		switch (outputType) {//single_c2c + single_c2c_strided
		case 0: case 2: case 3: case 4: {
			char outputOffset[30] = "";
			if (sc->outputOffset > 0)
				sprintf(outputOffset, "%d + ", sc->outputOffset);
			char shiftX[500] = "";
			if (sc->fftDim == sc->fft_dim_full)
				sprintf(shiftX, "(%s)", index_x);
			else
				sprintf(shiftX, "(%s) * %d", index_x, sc->outputStride[0]);
			char shiftY[500] = "";
			if (sc->size[1] > 1) {
				if (sc->fftDim == sc->fft_dim_full) {
					if (sc->performWorkGroupShift[1])
						sprintf(shiftY, " + (%s + consts.workGroupShiftY) * %d", sc->gl_WorkGroupID_y, sc->localSize[1] * sc->outputStride[1]);
					else
						sprintf(shiftY, " + %s * %d", sc->gl_WorkGroupID_y, sc->localSize[1] * sc->outputStride[1]);
				}
				else {
					if (sc->performWorkGroupShift[1])
						sprintf(shiftY, " + (%s + consts.workGroupShiftY) * %d", sc->gl_WorkGroupID_y, sc->outputStride[1]);
					else
						sprintf(shiftY, " + %s * %d", sc->gl_WorkGroupID_y, sc->outputStride[1]);
				}
			}
			char shiftZ[500] = "";
			if (sc->size[2] > 1) {
				if (sc->performWorkGroupShift[2])
					sprintf(shiftZ, " + (%s + consts.workGroupShiftZ * %s) * %d", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z, sc->outputStride[2]);
				else
					sprintf(shiftZ, " + %s * %d", sc->gl_GlobalInvocationID_z, sc->outputStride[2]);
			}
			char shiftCoordinate[100] = "";
			char requestCoordinate[100] = "";
			if (sc->numCoordinates * sc->matrixConvolution > 1) {
				sprintf(shiftCoordinate, " + consts.coordinate * %d", sc->outputStride[3]);
			}
			if ((sc->matrixConvolution > 1) && (sc->convolutionStep)) {
				sprintf(shiftCoordinate, " + %s * %d", coordinate, sc->outputStride[3]);
				//sprintf(requestCoordinate, ", %s coordinate", uintType);
			}
			char shiftBatch[100] = "";
			char requestBatch[100] = "";
			if ((sc->numBatches > 1) || (sc->numKernels > 1)) {
				if (sc->convolutionStep) {
					sprintf(shiftBatch, " + %s * %d", batchID, sc->outputStride[4]);
					//sprintf(requestBatch, ", %s batchID", uintType);
				}
				else
					sprintf(shiftBatch, " + consts.batchID * %d", sc->outputStride[4]);
			}
			sprintf(output + strlen(output), "%s%s%s%s%s%s", outputOffset, shiftX, shiftY, shiftZ, shiftCoordinate, shiftBatch);
			break;
		}
		case 1: {//grouped_c2c
			char outputOffset[30] = "";
			if (sc->outputOffset > 0)
				sprintf(outputOffset, "%d + ", sc->outputOffset);
			char shiftX[500] = "";
			if (sc->fftDim == sc->fft_dim_full)
				sprintf(shiftX, "(%s)", index_x);
			else
				sprintf(shiftX, "(%s) * %d", index_x, sc->outputStride[0]);
			char shiftY[500] = "";
			sprintf(shiftY, " + (%s) * %d", index_y, sc->outputStride[1]);
			char shiftZ[500] = "";
			if (sc->size[2] > 1) {
				if (sc->performWorkGroupShift[2])
					sprintf(shiftZ, " + (%s + consts.workGroupShiftZ * %s) * %d", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z, sc->outputStride[2]);
				else
					sprintf(shiftZ, " + %s * %d", sc->gl_GlobalInvocationID_z, sc->outputStride[2]);
			}
			char shiftCoordinate[100] = "";
			char requestCoordinate[100] = "";
			if (sc->numCoordinates * sc->matrixConvolution > 1) {
				sprintf(shiftCoordinate, " + consts.coordinate * %d", sc->outputStride[3]);
			}
			if ((sc->matrixConvolution > 1) && (sc->convolutionStep)) {
				sprintf(shiftCoordinate, " + %s * %d", coordinate, sc->outputStride[3]);
				//sprintf(requestCoordinate, ", %s coordinate", uintType);
			}
			char shiftBatch[100] = "";
			char requestBatch[100] = "";
			if ((sc->numBatches > 1) || (sc->numKernels > 1)) {
				if (sc->convolutionStep) {
					sprintf(shiftBatch, " + %s * %d", batchID, sc->outputStride[4]);
					//sprintf(requestBatch, ", %s batchID", uintType);
				}
				else
					sprintf(shiftBatch, " + consts.batchID * %d", sc->outputStride[4]);
			}
			sprintf(output + strlen(output), "%s%s%s%s%s%s", outputOffset, shiftX, shiftY, shiftZ, shiftCoordinate, shiftBatch);
			break;

		}
		case 5: {//single_r2c
			char outputOffset[30] = "";
			if (sc->outputOffset > 0)
				sprintf(outputOffset, "%d + ", sc->outputOffset);
			char shiftX[500] = "";
			if (sc->outputStride[0] == 1)
				sprintf(shiftX, "(%s)", index_x);
			else
				sprintf(shiftX, "(%s) * %d", index_x, sc->outputStride[0]);
			char shiftY[500] = "";
			sprintf(shiftY, " + (%s) * %d", index_y, sc->outputStride[1]);
			char shiftZ[500] = "";
			if (sc->size[2] > 1) {
				if (sc->performWorkGroupShift[2])
					sprintf(shiftZ, " + (%s + consts.workGroupShiftZ * %s) * %d", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z, sc->outputStride[2]);
				else
					sprintf(shiftZ, " + %s * %d", sc->gl_GlobalInvocationID_z, sc->outputStride[2]);
			}
			char shiftCoordinate[100] = "";
			if (sc->numCoordinates * sc->matrixConvolution > 1) {
				sprintf(shiftCoordinate, " + consts.coordinate * %d", sc->outputStride[3]);
			}
			char shiftBatch[100] = "";
			if ((sc->numBatches > 1) || (sc->numKernels > 1)) {
				sprintf(shiftBatch, " + consts.batchID * %d", sc->outputStride[4]);
			}
			sprintf(output + strlen(output), "%s%s%s%s%s%s", outputOffset, shiftX, shiftY, shiftZ, shiftCoordinate, shiftBatch);
			break;
		}
		case 6: {//single_c2r
			char outputOffset[30] = "";
			if (sc->outputOffset > 0)
				sprintf(outputOffset, "%d + ", sc->outputOffset);
			char shiftX[500] = "";
			if (sc->outputStride[0] == 1)
				sprintf(shiftX, "(%s)", index_x);
			else
				sprintf(shiftX, "(%s) * %d", index_x, sc->outputStride[0]);
			char shiftY[500] = "";
			if (sc->size[1] > 1) {
				if (sc->fftDim == sc->fft_dim_full) {
					if (sc->performWorkGroupShift[1])
						sprintf(shiftY, " + (%s + consts.workGroupShiftY) * %d", sc->gl_WorkGroupID_y, 2 * sc->localSize[1] * sc->outputStride[1]);
					else
						sprintf(shiftY, " + %s * %d", sc->gl_WorkGroupID_y, 2 * sc->localSize[1] * sc->outputStride[1]);
				}
				else {
					if (sc->performWorkGroupShift[1])
						sprintf(shiftY, " + (%s + consts.workGroupShiftY) * %d", sc->gl_WorkGroupID_y, 2 * sc->outputStride[1]);
					else
						sprintf(shiftY, " + %s * %d", sc->gl_WorkGroupID_y, 2 * sc->outputStride[1]);
				}
			}
			char shiftZ[500] = "";
			if (sc->size[2] > 1) {
				if (sc->performWorkGroupShift[2])
					sprintf(shiftZ, " + (%s + consts.workGroupShiftZ * %s) * %d", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z, sc->outputStride[2]);
				else
					sprintf(shiftZ, " + %s * %d", sc->gl_GlobalInvocationID_z, sc->outputStride[2]);
			}
			char shiftCoordinate[100] = "";
			if (sc->numCoordinates * sc->matrixConvolution > 1) {
				sprintf(shiftCoordinate, " + consts.coordinate * %d", sc->outputStride[3]);
			}
			char shiftBatch[100] = "";
			if ((sc->numBatches > 1) || (sc->numKernels > 1)) {
				sprintf(shiftBatch, " + consts.batchID * %d", sc->outputStride[4]);
			}
			sprintf(output + strlen(output), "%s%s%s%s%s%s", outputOffset, shiftX, shiftY, shiftZ, shiftCoordinate, shiftBatch);
			break;
		}
		}
	}

	static inline void inlineRadixKernelVkFFT(char* output, VkFFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint32_t radix, uint32_t stageSize, double stageAngle, char** regID) {
		char vecType[30];
		char LFending[4] = "";
		if (!strcmp(floatType, "float")) sprintf(LFending, "f");
#if(VKFFT_BACKEND==0)
		if (!strcmp(floatType, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "dvec2");
		char cosDef[20] = "cos";
		char sinDef[20] = "sin";
		if (!strcmp(floatType, "double")) sprintf(LFending, "LF");
#elif(VKFFT_BACKEND==1)
		if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
		char cosDef[20] = "__cosf";
		char sinDef[20] = "__sinf";
		if (!strcmp(floatType, "double")) sprintf(LFending, "l");
#elif(VKFFT_BACKEND==2)
		if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
		char cosDef[20] = "__cosf";
		char sinDef[20] = "__sinf";
		if (!strcmp(floatType, "double")) sprintf(LFending, "l");
#endif
		char* temp = sc->temp;
		//sprintf(temp, "loc_0");
		char* w = sc->w;
		//sprintf(w, "w");
		char* iw = sc->iw;
		//sprintf(iw, "iw");
		char convolutionInverse[30] = "";
		if (sc->convolutionStep) sprintf(convolutionInverse, ", %s inverse", uintType);
		switch (radix) {
		case 2: {
			/*if (sc->LUT) {
				sprintf(output + strlen(output), "void radix2(inout %s temp_0, inout %s temp_1, %s LUTId) {\n", vecType, vecType, uintType);
			}
			else {
				sprintf(output + strlen(output), "void radix2(inout %s temp_0, inout %s temp_1, %s angle) {\n", vecType, vecType, floatType);
			}*/
			//VkAppendLine(sc, "	{\n");
			//sprintf(sc->tempStr, "	%s %s;\n", vecType, temp);
			//VkAppendLine(sc, sc->tempStr);
			//sprintf(output + strlen(output), "	{\n\
	%s temp;\n", vecType);
			if (sc->LUT) {
				sprintf(output + strlen(output), "	%s = twiddleLUT[LUTId];\n", w);
				if (!sc->inverse)
					sprintf(output + strlen(output), "	%s.y = -%s.y;\n", w, w);
			}
			else {
				if (!strcmp(floatType, "float")) {
					sprintf(output + strlen(output), "	%s.x = %s(angle);\n", w, cosDef);
					sprintf(output + strlen(output), "	%s.y = %s(angle);\n", w, sinDef);
				}
				if (!strcmp(floatType, "double"))
					sprintf(output + strlen(output), "	%s = sincos_20(angle);\n", w);
			}
			VkMulComplex(sc, temp, regID[1], w, 0);
			VkSubComplex(sc, regID[1], regID[0], temp);
			VkAddComplex(sc, regID[0], regID[0], temp);
			//VkAppendLine(sc, "	}\n");
			//sprintf(output + strlen(output), "\
	temp.x = temp%s.x * w.x - temp%s.y * w.y;\n\
	temp.y = temp%s.y * w.x + temp%s.x * w.y;\n\
	temp%s = temp%s - temp;\n\
	temp%s = temp%s + temp;\n\
}\n", regID[1], regID[1], regID[1], regID[1], regID[1], regID[0], regID[0], regID[0]);
			break;
		}
		case 3: {
			/*	if (sc->LUT) {
					sprintf(output + strlen(output), "void radix3(inout %s temp_0, inout %s temp_1, inout %s temp_2, %s LUTId) {\n", vecType, vecType, vecType, uintType);
				}
				else {
					sprintf(output + strlen(output), "void radix3(inout %s temp_0, inout %s temp_1, inout %s temp_2, %s angle) {\n", vecType, vecType, vecType, floatType);
				}*/
			char* tf[2];
			//VkAppendLine(sc, "	{\n");
			for (uint32_t i = 0; i < 2; i++) {
				tf[i] = (char*)malloc(sizeof(char) * 40);
			}

			sprintf(tf[0], "-0.5%s", LFending);
			sprintf(tf[1], "-0.8660254037844386467637231707529%s", LFending);

			/*for (uint32_t i = 0; i < 3; i++) {
				sc->locID[i] = (char*)malloc(sizeof(char) * 40);
				sprintf(sc->locID[i], "loc_%d", i);
				sprintf(sc->tempStr, "	%s %s;\n", vecType, sc->locID[i]);
				VkAppendLine(sc, sc->tempStr);
			}*/
			if (sc->LUT) {
				sprintf(output + strlen(output), "	%s = twiddleLUT[LUTId];\n", w);
				if (!sc->inverse)
					sprintf(output + strlen(output), "	%s.y = -%s.y;\n", w, w);
			}
			else {
				if (!strcmp(floatType, "float")) {
					sprintf(output + strlen(output), "	%s.x = %s(angle*%.17f);\n", w, cosDef, 4.0 / 3.0);
					sprintf(output + strlen(output), "	%s.y = %s(angle*%.17f);\n", w, sinDef, 4.0 / 3.0);
					//sprintf(output + strlen(output), "	w = %s(cos(angle*%.17f), sin(angle*%.17f));\n\n", vecType, 4.0 / 3.0, 4.0 / 3.0);
				}
				if (!strcmp(floatType, "double"))
					sprintf(output + strlen(output), "	%s = sincos_20(angle*%.17f);\n", w, 4.0 / 3.0);
			}
			VkMulComplex(sc, sc->locID[2], regID[2], w, 0);
			//sprintf(output + strlen(output), "\
	loc_2.x = temp%s.x * w.x - temp%s.y * w.y;\n\
	loc_2.y = temp%s.y * w.x + temp%s.x * w.y;\n", regID[2], regID[2], regID[2], regID[2]);
			if (sc->LUT) {
				sprintf(output + strlen(output), "	%s = twiddleLUT[LUTId+%d];\n", w, stageSize);
				if (!sc->inverse)
					sprintf(output + strlen(output), "	%s.y = -%s.y;\n", w, w);
			}
			else {
				if (!strcmp(floatType, "float")) {
					sprintf(output + strlen(output), "	%s.x = %s(angle*%.17f);\n", w, cosDef, 2.0 / 3.0);
					sprintf(output + strlen(output), "	%s.y = %s(angle*%.17f);\n", w, sinDef, 2.0 / 3.0);
					//sprintf(output + strlen(output), "	w = %s(cos(angle*%.17f), sin(angle*%.17f));\n\n", vecType, 2.0 / 3.0, 2.0 / 3.0);
				}
				if (!strcmp(floatType, "double"))
					sprintf(output + strlen(output), "	%s=sincos_20(angle*%.17f);\n", w, 2.0 / 3.0);
			}
			VkMulComplex(sc, sc->locID[1], regID[1], w, 0);
			//sprintf(output + strlen(output), "\
	loc_1.x = temp%s.x * w.x - temp%s.y * w.y;\n\
	loc_1.y = temp%s.y * w.x + temp%s.x * w.y;\n", regID[1], regID[1], regID[1], regID[1]);
			VkAddComplex(sc, regID[1], sc->locID[1], sc->locID[2]);
			VkSubComplex(sc, regID[2], sc->locID[1], sc->locID[2]);
			//sprintf(output + strlen(output), "\
	temp%s = loc_1 + loc_2;\n\
	temp%s = loc_1 - loc_2;\n", regID[1], regID[2]);
			VkAddComplex(sc, sc->locID[0], regID[0], regID[1]);
			VkFMAComplex(sc, sc->locID[1], regID[1], tf[0], regID[0]);
			VkMulComplexNumber(sc, sc->locID[2], regID[2], tf[1]);
			VkMovComplex(sc, regID[0], sc->locID[0]);
			//sprintf(output + strlen(output), "\
	loc_0 = temp%s + temp%s;\n\
	loc_1 = temp%s - 0.5 * temp%s;\n\
	loc_2 = -0.8660254037844386467637231707529 * temp%s;\n\
	temp%s = loc_0;\n", regID[0], regID[1], regID[0], regID[1], regID[2], regID[0]);

			if (stageAngle < 0)
			{
				VkShuffleComplex(sc, regID[1], sc->locID[1], sc->locID[2], 0);
				VkShuffleComplexInv(sc, regID[2], sc->locID[1], sc->locID[2], 0);
				//sprintf(output + strlen(output), "\
	temp%s.x = loc_1.x - loc_2.y; \n\
	temp%s.y = loc_1.y + loc_2.x; \n\
	temp%s.x = loc_1.x + loc_2.y; \n\
	temp%s.y = loc_1.y - loc_2.x; \n", regID[1], regID[1], regID[2], regID[2]);
			}
			else {
				VkShuffleComplexInv(sc, regID[1], sc->locID[1], sc->locID[2], 0);
				VkShuffleComplex(sc, regID[2], sc->locID[1], sc->locID[2], 0);
				//sprintf(output + strlen(output), "\
	temp%s.x = loc_1.x + loc_2.y; \n\
	temp%s.y = loc_1.y - loc_2.x; \n\
	temp%s.x = loc_1.x - loc_2.y; \n\
	temp%s.y = loc_1.y + loc_2.x; \n", regID[1], regID[1], regID[2], regID[2]);
			}

			//VkAppendLine(sc, "	}\n");
			for (uint32_t i = 0; i < 2; i++) {
				free(tf[i]);
				//free(sc->locID[i]);
			}
			//free(sc->locID[2]);
			break;
		}
		case 4: {
			/*if (sc->LUT)
				sprintf(output + strlen(output), "void radix4(inout %s temp_0, inout %s temp_1, inout %s temp_2, inout %s temp_3, %s LUTId%s) {\n", vecType, vecType, vecType, vecType, uintType, convolutionInverse);
			else
				sprintf(output + strlen(output), "void radix4(inout %s temp_0, inout %s temp_1, inout %s temp_2, inout %s temp_3, %s angle%s) {\n", vecType, vecType, vecType, vecType, floatType, convolutionInverse);
			*/
			//VkAppendLine(sc, "	{\n");
			//sprintf(sc->tempStr, "	%s %s;\n", vecType, temp);
			//VkAppendLine(sc, sc->tempStr);
			if (sc->LUT) {
				sprintf(output + strlen(output), "	%s = twiddleLUT[LUTId];\n", w);
				if (!sc->inverse)
					sprintf(output + strlen(output), "	%s.y = -%s.y;\n", w, w);
			}
			else {
				if (!strcmp(floatType, "float")) {
					sprintf(output + strlen(output), "	%s.x = %s(angle);\n", w, cosDef);
					sprintf(output + strlen(output), "	%s.y = %s(angle);\n", w, sinDef);
				}
				if (!strcmp(floatType, "double"))
					sprintf(output + strlen(output), "	%s = sincos_20(angle);\n", w);
			}
			VkMulComplex(sc, temp, regID[2], w, 0);
			VkSubComplex(sc, regID[2], regID[0], temp);
			VkAddComplex(sc, regID[0], regID[0], temp);
			VkMulComplex(sc, temp, regID[3], w, 0);
			VkSubComplex(sc, regID[3], regID[1], temp);
			VkAddComplex(sc, regID[1], regID[1], temp);
			//sprintf(output + strlen(output), "\
	temp.x=temp%s.x*w.x-temp%s.y*w.y;\n\
	temp.y = temp%s.y * w.x + temp%s.x * w.y;\n\
	temp%s = temp%s - temp;\n\
	temp%s = temp%s + temp;\n\n\
	temp.x=temp%s.x*w.x-temp%s.y*w.y;\n\
	temp.y = temp%s.y * w.x + temp%s.x * w.y;\n\
	temp%s = temp%s - temp;\n\
	temp%s = temp%s + temp;\n\n\
	//DIF 2nd stage with angle\n", regID[2], regID[2], regID[2], regID[2], regID[2], regID[0], regID[0], regID[0], regID[3], regID[3], regID[3], regID[3], regID[3], regID[1], regID[1], regID[1]);
			if (sc->LUT) {
				sprintf(output + strlen(output), "	%s=twiddleLUT[LUTId+%d];\n", w, stageSize);
				if (!sc->inverse)
					sprintf(output + strlen(output), "	%s.y = -%s.y;\n", w, w);
			}
			else {
				if (!strcmp(floatType, "float")) {
					sprintf(output + strlen(output), "	%s.x = %s(0.5*angle);\n", w, cosDef);
					sprintf(output + strlen(output), "	%s.y = %s(0.5*angle);\n", w, sinDef);
				}
				if (!strcmp(floatType, "double"))
					sprintf(output + strlen(output), "	%s=normalize(%s + %s(1.0, 0.0));\n", w, w, vecType);
			}
			VkMulComplex(sc, temp, regID[1], w, 0);
			VkSubComplex(sc, regID[1], regID[0], temp);
			VkAddComplex(sc, regID[0], regID[0], temp);
			//sprintf(output + strlen(output), "\
	temp.x = temp%s.x * w.x - temp%s.y * w.y;\n\
	temp.y = temp%s.y * w.x + temp%s.x * w.y;\n\
	temp%s = temp%s - temp;\n\
	temp%s = temp%s + temp;\n\n", regID[1], regID[1], regID[1], regID[1], regID[1], regID[0], regID[0], regID[0]);
			if (stageAngle < 0) {
				sprintf(output + strlen(output), "	%s.x = %s.x;", temp, w);
				sprintf(output + strlen(output), "	%s.x = %s.y;\n", w, w);
				sprintf(output + strlen(output), "	%s.y = -%s.x;\n", w, temp);
				//sprintf(output + strlen(output), "	w = %s(w.y, -w.x);\n\n", vecType);
			}
			else {
				sprintf(output + strlen(output), "	%s.x = %s.x;", temp, w);
				sprintf(output + strlen(output), "	%s.x = -%s.y;\n", w, w);
				sprintf(output + strlen(output), "	%s.y = %s.x;\n", w, temp);
				//sprintf(output + strlen(output), "	w = %s(-w.y, w.x);\n\n", vecType);
			}
			VkMulComplex(sc, temp, regID[3], w, 0);
			VkSubComplex(sc, regID[3], regID[2], temp);
			VkAddComplex(sc, regID[2], regID[2], temp);
			VkMovComplex(sc, temp, regID[1]);
			VkMovComplex(sc, regID[1], regID[2]);
			VkMovComplex(sc, regID[2], temp);
			//VkAppendLine(sc, "	}\n");
			//sprintf(output + strlen(output), "\
	temp.x = temp%s.x * w.x - temp%s.y * w.y;\n\
	temp.y = temp%s.y * w.x + temp%s.x * w.y;\n\
	temp%s = temp%s - temp;\n\
	temp%s = temp%s + temp;\n\n\
	temp = temp%s;\n\
	temp%s = temp%s;\n\
	temp%s = temp;\n\
}\n", regID[3], regID[3], regID[3], regID[3], regID[3], regID[2], regID[2], regID[2], regID[1], regID[1], regID[2], regID[2]);
			break;
		}
		case 5: {
			/*if (sc->LUT) {
				sprintf(output + strlen(output), "void radix5(inout %s temp_0, inout %s temp_1, inout %s temp_2, inout %s temp_3, inout %s temp_4, %s LUTId) {\n", vecType, vecType, vecType, vecType, vecType, uintType);
			}
			else {
				sprintf(output + strlen(output), "void radix5(inout %s temp_0, inout %s temp_1, inout %s temp_2, inout %s temp_3, inout %s temp_4, %s angle) {\n", vecType, vecType, vecType, vecType, vecType, floatType);
			}*/
			char* tf[5];
			//VkAppendLine(sc, "	{\n");
			for (uint32_t i = 0; i < 5; i++) {
				tf[i] = (char*)malloc(sizeof(char) * 40);
			}
			sprintf(tf[0], "-0.5%s", LFending);
			sprintf(tf[1], "1.538841768587626701285145288018455%s", LFending);
			sprintf(tf[2], "-0.363271264002680442947733378740309%s", LFending);
			sprintf(tf[3], "-0.809016994374947424102293417182819%s", LFending);
			sprintf(tf[4], "-0.587785252292473129168705954639073%s", LFending);

			/*for (uint32_t i = 0; i < 5; i++) {
				sc->locID[i] = (char*)malloc(sizeof(char) * 40);
				sprintf(sc->locID[i], "loc_%d", i);
				sprintf(sc->tempStr, "	%s %s;\n", vecType, sc->locID[i]);
				VkAppendLine(sc, sc->tempStr);
			}*/
			//sprintf(output + strlen(output), "	{\n\
	%s loc_0;\n	%s loc_1;\n	%s loc_2;\n	%s loc_3;\n	%s loc_4;\n", vecType, vecType, vecType, vecType, vecType);
			for (uint32_t i = radix - 1; i > 0; i--) {
				if (i == radix - 1) {
					if (sc->LUT) {
						sprintf(output + strlen(output), "	%s = twiddleLUT[LUTId];\n", w);
						if (!sc->inverse)
							sprintf(output + strlen(output), "	%s.y = -%s.y;\n", w, w);
					}
					else {
						if (!strcmp(floatType, "float")) {
							sprintf(output + strlen(output), "	%s.x = %s(angle*%.17f);\n", w, cosDef, 2.0 * i / radix);
							sprintf(output + strlen(output), "	%s.y = %s(angle*%.17f);\n", w, sinDef, 2.0 * i / radix);
							//sprintf(output + strlen(output), "	w = %s(cos(angle*%.17f), sin(angle*%.17f));\n\n", vecType, 2.0 * i / radix, 2.0 * i / radix);
						}
						if (!strcmp(floatType, "double"))
							sprintf(output + strlen(output), "	%s = sincos_20(angle*%.17f);\n", w, 2.0 * i / radix);
					}
				}
				else {
					if (sc->LUT) {
						sprintf(output + strlen(output), "	%s = twiddleLUT[LUTId+%d];\n", w, (radix - 1 - i) * stageSize);
						if (!sc->inverse)
							sprintf(output + strlen(output), "	%s.y = -%s.y;\n", w, w);
					}
					else {
						if (!strcmp(floatType, "float")) {
							sprintf(output + strlen(output), "	%s.x = %s(angle*%.17f);\n", w, cosDef, 2.0 * i / radix);
							sprintf(output + strlen(output), "	%s.y = %s(angle*%.17f);\n", w, sinDef, 2.0 * i / radix);
							//sprintf(output + strlen(output), "	w = %s(cos(angle*%.17f), sin(angle*%.17f));\n\n", vecType, 2.0 * i / radix, 2.0 * i / radix);
						}
						if (!strcmp(floatType, "double"))
							sprintf(output + strlen(output), "	%s = sincos_20(angle*%.17f);\n", w, 2.0 * i / radix);
					}
				}
				VkMulComplex(sc, sc->locID[i], regID[i], w, 0);
				//sprintf(output + strlen(output), "\
	loc_%d.x = temp%s.x * w.x - temp%s.y * w.y;\n\
	loc_%d.y = temp%s.y * w.x + temp%s.x * w.y;\n", i, regID[i], regID[i], i, regID[i], regID[i]);
			}
			VkAddComplex(sc, regID[1], sc->locID[1], sc->locID[4]);
			VkAddComplex(sc, regID[2], sc->locID[2], sc->locID[3]);
			VkSubComplex(sc, regID[3], sc->locID[2], sc->locID[3]);
			VkSubComplex(sc, regID[4], sc->locID[1], sc->locID[4]);
			VkSubComplex(sc, sc->locID[3], regID[1], regID[2]);
			VkAddComplex(sc, sc->locID[4], regID[3], regID[4]);
			//sprintf(output + strlen(output), "\
	temp%s = loc_1 + loc_4;\n\
	temp%s = loc_2 + loc_3;\n\
	temp%s = loc_2 - loc_3;\n\
	temp%s = loc_1 - loc_4;\n\
	loc_3 = temp%s - temp%s;\n\
	loc_4 = temp%s + temp%s;\n", regID[1], regID[2], regID[3], regID[4], regID[1], regID[2], regID[3], regID[4]);
			VkAddComplex(sc, sc->locID[0], regID[0], regID[1]);
			VkAddComplex(sc, sc->locID[0], sc->locID[0], regID[2]);
			VkFMAComplex(sc, sc->locID[1], regID[1], tf[0], regID[0]);
			VkFMAComplex(sc, sc->locID[2], regID[2], tf[0], regID[0]);
			VkMulComplexNumber(sc, regID[3], regID[3], tf[1]);
			VkMulComplexNumber(sc, regID[4], regID[4], tf[2]);
			VkMulComplexNumber(sc, sc->locID[3], sc->locID[3], tf[3]);
			VkMulComplexNumber(sc, sc->locID[4], sc->locID[4], tf[4]);
			//sprintf(output + strlen(output), "\
	loc_0 = temp%s + temp%s + temp%s;\n\
	loc_1 = temp%s - 0.5 * temp%s;\n\
	loc_2 = temp%s - 0.5 * temp%s;\n\
	temp%s *= 1.538841768587626701285145288018455;\n\
	temp%s *= -0.363271264002680442947733378740309;\n\
	loc_3 *= -0.809016994374947424102293417182819;\n\
	loc_4 *= -0.587785252292473129168705954639073;\n", regID[0], regID[1], regID[2], regID[0], regID[1], regID[0], regID[2], regID[3], regID[4]);
			VkSubComplex(sc, sc->locID[1], sc->locID[1], sc->locID[3]);
			VkAddComplex(sc, sc->locID[2], sc->locID[2], sc->locID[3]);
			VkAddComplex(sc, sc->locID[3], regID[3], sc->locID[4]);
			VkAddComplex(sc, sc->locID[4], sc->locID[4], regID[4]);
			VkMovComplex(sc, regID[0], sc->locID[0]);
			//sprintf(output + strlen(output), "\
	loc_1 -= loc_3;\n\
	loc_2 += loc_3;\n\
	loc_3 = temp%s+loc_4;\n\
	loc_4 += temp%s;\n\
	temp%s = loc_0;\n", regID[3], regID[4], regID[0]);

			if (stageAngle < 0)
			{
				VkShuffleComplex(sc, regID[1], sc->locID[1], sc->locID[4], 0);
				VkShuffleComplex(sc, regID[2], sc->locID[2], sc->locID[3], 0);
				VkShuffleComplexInv(sc, regID[3], sc->locID[2], sc->locID[3], 0);
				VkShuffleComplexInv(sc, regID[4], sc->locID[1], sc->locID[4], 0);
				//sprintf(output + strlen(output), "\
	temp%s.x = loc_1.x - loc_4.y; \n\
	temp%s.y = loc_1.y + loc_4.x; \n\
	temp%s.x = loc_2.x - loc_3.y; \n\
	temp%s.y = loc_2.y + loc_3.x; \n\
	temp%s.x = loc_2.x + loc_3.y; \n\
	temp%s.y = loc_2.y - loc_3.x; \n\
	temp%s.x = loc_1.x + loc_4.y; \n\
	temp%s.y = loc_1.y - loc_4.x; \n", regID[1], regID[1], regID[2], regID[2], regID[3], regID[3], regID[4], regID[4]);
			}
			else {
				VkShuffleComplexInv(sc, regID[1], sc->locID[1], sc->locID[4], 0);
				VkShuffleComplexInv(sc, regID[2], sc->locID[2], sc->locID[3], 0);
				VkShuffleComplex(sc, regID[3], sc->locID[2], sc->locID[3], 0);
				VkShuffleComplex(sc, regID[4], sc->locID[1], sc->locID[4], 0);
				//sprintf(output + strlen(output), "\
	temp%s.x = loc_1.x + loc_4.y; \n\
	temp%s.y = loc_1.y - loc_4.x; \n\
	temp%s.x = loc_2.x + loc_3.y; \n\
	temp%s.y = loc_2.y - loc_3.x; \n\
	temp%s.x = loc_2.x - loc_3.y; \n\
	temp%s.y = loc_2.y + loc_3.x; \n\
	temp%s.x = loc_1.x - loc_4.y; \n\
	temp%s.y = loc_1.y + loc_4.x; \n", regID[1], regID[1], regID[2], regID[2], regID[3], regID[3], regID[4], regID[4]);
			}

			//VkAppendLine(sc, "	}\n");
			for (uint32_t i = 0; i < 5; i++) {
				free(tf[i]);
				//free(sc->locID[i]);
			}
			break;
		}
		case 7: {
			/*if (sc->LUT) {
				sprintf(output + strlen(output), "void radix5(inout %s temp_0, inout %s temp_1, inout %s temp_2, inout %s temp_3, inout %s temp_4, %s LUTId) {\n", vecType, vecType, vecType, vecType, vecType, uintType);
			}
			else {
				sprintf(output + strlen(output), "void radix5(inout %s temp_0, inout %s temp_1, inout %s temp_2, inout %s temp_3, inout %s temp_4, %s angle) {\n", vecType, vecType, vecType, vecType, vecType, floatType);
			}*/
			char* tf[8];

			//VkAppendLine(sc, "	{\n");
			for (uint32_t i = 0; i < 8; i++) {
				tf[i] = (char*)malloc(sizeof(char) * 40);

			}
			sprintf(tf[0], "-1.16666666666666651863693004997913%s", LFending);
			sprintf(tf[1], "0.79015646852540022404554065360571%s", LFending);
			sprintf(tf[2], "0.05585426728964774240049351305970%s", LFending);
			sprintf(tf[3], "0.73430220123575240531721419756650%s", LFending);
			if (stageAngle < 0) {
				sprintf(tf[4], "0.44095855184409837868031445395900%s", LFending);
				sprintf(tf[5], "0.34087293062393136944265847887436%s", LFending);
				sprintf(tf[6], "-0.53396936033772524066165487965918%s", LFending);
				sprintf(tf[7], "0.87484229096165666561546458979137%s", LFending);
			}
			else {
				sprintf(tf[4], "-0.44095855184409837868031445395900%s", LFending);
				sprintf(tf[5], "-0.34087293062393136944265847887436%s", LFending);
				sprintf(tf[6], "0.53396936033772524066165487965918%s", LFending);
				sprintf(tf[7], "-0.87484229096165666561546458979137%s", LFending);
			}
			/*for (uint32_t i = 0; i < 7; i++) {
				sc->locID[i] = (char*)malloc(sizeof(char) * 40);
				sprintf(sc->locID[i], "loc_%d", i);
				sprintf(sc->tempStr, "	%s %s;\n", vecType, sc->locID[i]);
				VkAppendLine(sc, sc->tempStr);
			}*/
			for (uint32_t i = radix - 1; i > 0; i--) {
				if (i == radix - 1) {
					if (sc->LUT) {
						sprintf(output + strlen(output), "	%s = twiddleLUT[LUTId];\n", w);
						if (!sc->inverse)
							sprintf(output + strlen(output), "	%s.y = -%s.y;\n", w, w);
					}
					else {
						if (!strcmp(floatType, "float")) {
							sprintf(output + strlen(output), "	%s.x = %s(angle*%.17f);\n", w, cosDef, 2.0 * i / radix);
							sprintf(output + strlen(output), "	%s.y = %s(angle*%.17f);\n", w, sinDef, 2.0 * i / radix);
							//sprintf(output + strlen(output), "	w = %s(cos(angle*%.17f), sin(angle*%.17f));\n\n", vecType, 2.0 * i / radix, 2.0 * i / radix);
						}
						if (!strcmp(floatType, "double"))
							sprintf(output + strlen(output), "	%s = sincos_20(angle*%.17f);\n", w, 2.0 * i / radix);
					}
				}
				else {
					if (sc->LUT) {
						sprintf(output + strlen(output), "	%s = twiddleLUT[LUTId+%d];\n\n", w, (radix - 1 - i) * stageSize);
						if (!sc->inverse)
							sprintf(output + strlen(output), "	%s.y = -%s.y;\n", w, w);
					}
					else {
						if (!strcmp(floatType, "float")) {
							sprintf(output + strlen(output), "	%s.x = %s(angle*%.17f);\n", w, cosDef, 2.0 * i / radix);
							sprintf(output + strlen(output), "	%s.y = %s(angle*%.17f);\n", w, sinDef, 2.0 * i / radix);
							//sprintf(output + strlen(output), "	w = %s(cos(angle*%.17f), sin(angle*%.17f));\n\n", vecType, 2.0 * i / radix, 2.0 * i / radix);
						}
						if (!strcmp(floatType, "double"))
							sprintf(output + strlen(output), "	%s = sincos_20(angle*%.17f);\n", w, 2.0 * i / radix);
					}
				}
				VkMulComplex(sc, sc->locID[i], regID[i], w, 0);
				//sprintf(output + strlen(output), "\
	loc_%d.x = temp%s.x * w.x - temp%s.y * w.y;\n\
	loc_%d.y = temp%s.y * w.x + temp%s.x * w.y;\n", i, regID[i], regID[i], i, regID[i], regID[i]);
			}
			VkMovComplex(sc, sc->locID[0], regID[0]);
			VkAddComplex(sc, regID[0], sc->locID[1], sc->locID[6]);
			VkSubComplex(sc, regID[1], sc->locID[1], sc->locID[6]);
			VkAddComplex(sc, regID[2], sc->locID[2], sc->locID[5]);
			VkSubComplex(sc, regID[3], sc->locID[2], sc->locID[5]);
			VkAddComplex(sc, regID[4], sc->locID[4], sc->locID[3]);
			VkSubComplex(sc, regID[5], sc->locID[4], sc->locID[3]);
			//sprintf(output + strlen(output), "\
	loc_0 = temp%s;\n\
	temp%s = loc_1 + loc_6;\n\
	temp%s = loc_1 - loc_6;\n\
	temp%s = loc_2 + loc_5;\n\
	temp%s = loc_2 - loc_5;\n\
	temp%s = loc_4 + loc_3;\n\
	temp%s = loc_4 - loc_3;\n", regID[0], regID[0], regID[1], regID[2], regID[3], regID[4], regID[5]);
			VkAddComplex(sc, sc->locID[5], regID[1], regID[3]);
			VkAddComplex(sc, sc->locID[5], sc->locID[5], regID[5]);
			VkAddComplex(sc, sc->locID[1], regID[0], regID[2]);
			VkAddComplex(sc, sc->locID[1], sc->locID[1], regID[4]);
			VkAddComplex(sc, sc->locID[0], sc->locID[0], sc->locID[1]);
			//sprintf(output + strlen(output), "\
	loc_5 = temp%s + temp%s + temp%s;\n\
	loc_1 = temp%s + temp%s + temp%s;\n\
	loc_0 += loc_1;\n", regID[1], regID[3], regID[5], regID[0], regID[2], regID[4]);
			VkSubComplex(sc, sc->locID[2], regID[0], regID[4]);
			VkSubComplex(sc, sc->locID[3], regID[4], regID[2]);
			VkSubComplex(sc, sc->locID[4], regID[2], regID[0]);
			//sprintf(output + strlen(output), "\
	loc_2 = temp%s - temp%s;\n\
	loc_3 = temp%s - temp%s;\n\
	loc_4 = temp%s - temp%s;\n", regID[0], regID[4], regID[4], regID[2], regID[2], regID[0]);
			VkSubComplex(sc, regID[0], regID[1], regID[5]);
			VkSubComplex(sc, regID[2], regID[5], regID[3]);
			VkSubComplex(sc, regID[4], regID[3], regID[1]);
			//sprintf(output + strlen(output), "\
	temp%s = temp%s - temp%s;\n\
	temp%s = temp%s - temp%s;\n\
	temp%s = temp%s - temp%s;\n", regID[0], regID[1], regID[5], regID[2], regID[5], regID[3], regID[4], regID[3], regID[1]);
			
				VkMulComplexNumber(sc, sc->locID[1], sc->locID[1], tf[0]);
				VkMulComplexNumber(sc, sc->locID[2], sc->locID[2], tf[1]);
				VkMulComplexNumber(sc, sc->locID[3], sc->locID[3], tf[2]);
				VkMulComplexNumber(sc, sc->locID[4], sc->locID[4], tf[3]);
			VkMulComplexNumber(sc, sc->locID[5], sc->locID[5], tf[4]);
			VkMulComplexNumber(sc, regID[0], regID[0], tf[5]);
			VkMulComplexNumber(sc, regID[2], regID[2], tf[6]);
			VkMulComplexNumber(sc, regID[4], regID[4], tf[7]);
				//sprintf(output + strlen(output), "\
	loc_1 *= -1.16666666666666651863693004997913;\n\
	loc_2 *= 0.79015646852540022404554065360571;\n\
	loc_3 *= 0.05585426728964774240049351305970;\n\
	loc_4 *= 0.73430220123575240531721419756650;\n\
	loc_5 *= 0.44095855184409837868031445395900;\n\
	temp%s *= 0.34087293062393136944265847887436;\n\
	temp%s *= -0.53396936033772524066165487965918;\n\
	temp%s *= 0.87484229096165666561546458979137;\n", regID[0], regID[2], regID[4]);
			
			VkSubComplex(sc, regID[5], regID[4], regID[2]);
			VkAddComplexInv(sc, regID[6], regID[4], regID[0]);
			VkAddComplex(sc, regID[4], regID[0], regID[2]);
			//sprintf(output + strlen(output), "\
	temp%s = temp%s - temp%s;\n\
	temp%s = - temp%s - temp%s;\n\
	temp%s = temp%s + temp%s;\n", regID[5], regID[4], regID[2], regID[6], regID[4], regID[0], regID[4], regID[0], regID[2]);
			VkAddComplex(sc, regID[0], sc->locID[0], sc->locID[1]);
			VkAddComplex(sc, regID[1], sc->locID[2], sc->locID[3]);
			VkSubComplex(sc, regID[2], sc->locID[4], sc->locID[3]);
			VkAddComplexInv(sc, regID[3], sc->locID[2], sc->locID[4]);
			//sprintf(output + strlen(output), "\
	temp%s = loc_0 + loc_1;\n\
	temp%s = loc_2 + loc_3;\n\
	temp%s = loc_4 - loc_3;\n\
	temp%s = - loc_2 - loc_4;\n", regID[0], regID[1], regID[2], regID[3]);
			VkAddComplex(sc, sc->locID[1], regID[0], regID[1]);
			VkAddComplex(sc, sc->locID[2], regID[0], regID[2]);
			VkAddComplex(sc, sc->locID[3], regID[0], regID[3]);
			VkAddComplex(sc, sc->locID[4], regID[4], sc->locID[5]);
			VkAddComplex(sc, sc->locID[6], regID[6], sc->locID[5]);
			VkAddComplex(sc, sc->locID[5], sc->locID[5], regID[5]);
			VkMovComplex(sc, regID[0], sc->locID[0]);
			//sprintf(output + strlen(output), "\
	loc_1 = temp%s + temp%s;\n\
	loc_2 = temp%s + temp%s;\n\
	loc_3 = temp%s + temp%s;\n\
	loc_4 = temp%s + loc_5;\n\
	loc_6 = temp%s + loc_5;\n\
	loc_5 += temp%s;\n\
	temp%s = loc_0;\n", regID[0], regID[1], regID[0], regID[2], regID[0], regID[3], regID[4], regID[6], regID[5], regID[0]);
			VkShuffleComplexInv(sc, regID[1], sc->locID[1], sc->locID[4], 0);
			VkShuffleComplexInv(sc, regID[2], sc->locID[3], sc->locID[6], 0);
			VkShuffleComplex(sc, regID[3], sc->locID[2], sc->locID[5], 0);
			VkShuffleComplexInv(sc, regID[4], sc->locID[2], sc->locID[5], 0);
			VkShuffleComplex(sc, regID[5], sc->locID[3], sc->locID[6], 0);
			VkShuffleComplex(sc, regID[6], sc->locID[1], sc->locID[4], 0);

			//sprintf(output + strlen(output), "\
	temp%s.x = loc_1.x + loc_4.y; \n\
	temp%s.y = loc_1.y - loc_4.x; \n\
	temp%s.x = loc_3.x + loc_6.y; \n\
	temp%s.y = loc_3.y - loc_6.x; \n\
	temp%s.x = loc_2.x - loc_5.y; \n\
	temp%s.y = loc_2.y + loc_5.x; \n\
	temp%s.x = loc_2.x + loc_5.y; \n\
	temp%s.y = loc_2.y - loc_5.x; \n\
	temp%s.x = loc_3.x - loc_6.y; \n\
	temp%s.y = loc_3.y + loc_6.x; \n\
	temp%s.x = loc_1.x - loc_4.y; \n\
	temp%s.y = loc_1.y + loc_4.x; \n", regID[1], regID[1], regID[2], regID[2], regID[3], regID[3], regID[4], regID[4], regID[5], regID[5], regID[6], regID[6]);
			//VkAppendLine(sc, "	}\n");
			/*for (uint32_t i = 0; i < 7; i++) {
				free(sc->locID[i]);
			}*/
			for (uint32_t i = 0; i < 8; i++) {
				free(tf[i]);
			}
			break;
		}
		case 8: {
			/*if (sc->LUT)
				sprintf(output + strlen(output), "void radix8(inout %s temp_0, inout %s temp_1, inout %s temp_2, inout %s temp_3, inout %s temp_4, inout %s temp_5, inout %s temp_6, inout %s temp_7, %s LUTId%s) {\n", vecType, vecType, vecType, vecType, vecType, vecType, vecType, vecType, uintType, convolutionInverse);
			else
				sprintf(output + strlen(output), "void radix8(inout %s temp_0, inout %s temp_1, inout %s temp_2, inout %s temp_3, inout %s temp_4, inout %s temp_5, inout %s temp_6, inout %s temp_7, %s angle%s) {\n", vecType, vecType, vecType, vecType, vecType, vecType, vecType, vecType, floatType, convolutionInverse);
			*/
			//VkAppendLine(sc, "	{\n");
			/*sprintf(sc->tempStr, "	%s %s;\n", vecType, temp);
			VkAppendLine(sc, sc->tempStr);
			sprintf(sc->tempStr, "	%s %s;\n", vecType, iw);
			VkAppendLine(sc, sc->tempStr);*/
			if (sc->LUT) {
				sprintf(output + strlen(output), "	%s = twiddleLUT[LUTId];\n", w);
				if (!sc->inverse)
					sprintf(output + strlen(output), "	%s.y = -%s.y;\n", w, w);
			}
			else {
				if (!strcmp(floatType, "float")) {
					sprintf(output + strlen(output), "	%s.x = %s(angle);\n", w, cosDef);
					sprintf(output + strlen(output), "	%s.y = %s(angle);\n", w, sinDef);
				}
				if (!strcmp(floatType, "double"))
					sprintf(output + strlen(output), "	%s = sincos_20(angle);\n", w);
			}
			for (uint32_t i = 0; i < 4; i++) {
				VkMulComplex(sc, temp, regID[i + 4], w, 0);
				VkSubComplex(sc, regID[i + 4], regID[i], temp);
				VkAddComplex(sc, regID[i], regID[i], temp);
				//sprintf(output + strlen(output), "\
	temp.x=temp%s.x*w.x-temp%s.y*w.y;\n\
	temp.y = temp%s.y * w.x + temp%s.x * w.y;\n\
	temp%s = temp%s - temp;\n\
	temp%s = temp%s + temp;\n\n", regID[i + 4], regID[i + 4], regID[i + 4], regID[i + 4], regID[i + 4], regID[i + 0], regID[i + 0], regID[i + 0]);
			}
			if (sc->LUT) {
				sprintf(output + strlen(output), "	%s=twiddleLUT[LUTId+%d];\n\n", w, stageSize);
				if (!sc->inverse)
					sprintf(output + strlen(output), "	%s.y = -%s.y;\n", w, w);
			}
			else {
				if (!strcmp(floatType, "float")) {
					sprintf(output + strlen(output), "	%s.x = %s(0.5*angle);\n", w, cosDef);
					sprintf(output + strlen(output), "	%s.y = %s(0.5*angle);\n", w, sinDef);
				}
				if (!strcmp(floatType, "double"))
					sprintf(output + strlen(output), "	%s=normalize(%s + %s(1.0, 0.0));\n", w, w, vecType);
			}
			for (uint32_t i = 0; i < 2; i++) {
				VkMulComplex(sc, temp, regID[i + 2], w, 0);
				VkSubComplex(sc, regID[i + 2], regID[i], temp);
				VkAddComplex(sc, regID[i], regID[i], temp);
				//sprintf(output + strlen(output), "\
	temp.x=temp%s.x*w.x-temp%s.y*w.y;\n\
	temp.y = temp%s.y * w.x + temp%s.x * w.y;\n\
	temp%s = temp%s - temp;\n\
	temp%s = temp%s + temp;\n\n", regID[i + 2], regID[i + 2], regID[i + 2], regID[i + 2], regID[i + 2], regID[i + 0], regID[i + 0], regID[i + 0]);
			}
			if (stageAngle < 0) {
				sprintf(output + strlen(output), "	%s.x = %s.y;\n", iw, w);
				sprintf(output + strlen(output), "	%s.y = -%s.x;\n", iw, w);
				//sprintf(output + strlen(output), "	w = %s(w.y, -w.x);\n\n", vecType);
			}
			else {
				sprintf(output + strlen(output), "	%s.x = -%s.y;\n", iw, w);
				sprintf(output + strlen(output), "	%s.y = %s.x;\n", iw, w);
				//sprintf(output + strlen(output), "	iw = %s(-w.y, w.x);\n\n", vecType);
			}

			for (uint32_t i = 4; i < 6; i++) {
				VkMulComplex(sc, temp, regID[i + 2], iw, 0);
				VkSubComplex(sc, regID[i + 2], regID[i], temp);
				VkAddComplex(sc, regID[i], regID[i], temp);
				//sprintf(output + strlen(output), "\
	temp.x = temp%s.x * iw.x - temp%s.y * iw.y;\n\
	temp.y = temp%s.y * iw.x + temp%s.x * iw.y;\n\
	temp%s = temp%s - temp;\n\
	temp%s = temp%s + temp;\n\n", regID[i + 2], regID[i + 2], regID[i + 2], regID[i + 2], regID[i + 2], regID[i + 0], regID[i + 0], regID[i + 0]);
			}

			if (sc->LUT) {
				sprintf(output + strlen(output), "	%s=twiddleLUT[LUTId+%d];\n\n", w, 2 * stageSize);
				if (!sc->inverse)
					sprintf(output + strlen(output), "	%s.y = -%s.y;\n", w, w);
			}
			else {
				if (!strcmp(floatType, "float")) {
					sprintf(output + strlen(output), "	%s.x = %s(0.25*angle);\n", w, cosDef);
					sprintf(output + strlen(output), "	%s.y = %s(0.25*angle);\n", w, sinDef);
					//sprintf(output + strlen(output), "	w = %s(cos(0.25*angle), sin(0.25*angle));\n\n", vecType);
				}
				if (!strcmp(floatType, "double"))
					sprintf(output + strlen(output), "	%s=normalize(%s + %s(1.0, 0.0));\n", w, w, vecType);
			}
			VkMulComplex(sc, temp, regID[1], w, 0);
			VkSubComplex(sc, regID[1], regID[0], temp);
			VkAddComplex(sc, regID[0], regID[0], temp);
			//sprintf(output + strlen(output), "\
	temp.x=temp%s.x*w.x-temp%s.y*w.y;\n\
	temp.y = temp%s.y * w.x + temp%s.x * w.y;\n\
	temp%s = temp%s - temp;\n\
	temp%s = temp%s + temp;\n\n", regID[1], regID[1], regID[1], regID[1], regID[1], regID[0], regID[0], regID[0]);
			if (stageAngle < 0) {
				sprintf(output + strlen(output), "	%s.x = %s.y;\n", iw, w);
				sprintf(output + strlen(output), "	%s.y = -%s.x;\n", iw, w);
				//sprintf(output + strlen(output), "	w = %s(w.y, -w.x);\n\n", vecType);
			}
			else {
				sprintf(output + strlen(output), "	%s.x = -%s.y;\n", iw, w);
				sprintf(output + strlen(output), "	%s.y = %s.x;\n", iw, w);
				//sprintf(output + strlen(output), "	iw = %s(-w.y, w.x);\n\n", vecType);
			}
			VkMulComplex(sc, temp, regID[3], iw, 0);
			VkSubComplex(sc, regID[3], regID[2], temp);
			VkAddComplex(sc, regID[2], regID[2], temp);
			//sprintf(output + strlen(output), "\
	temp.x = temp%s.x * iw.x - temp%s.y * iw.y;\n\
	temp.y = temp%s.y * iw.x + temp%s.x * iw.y;\n\
	temp%s = temp%s - temp;\n\
	temp%s = temp%s + temp;\n\n", regID[3], regID[3], regID[3], regID[3], regID[3], regID[2], regID[2], regID[2]);
			if (stageAngle < 0) {
				sprintf(output + strlen(output), "	%s.x = %s.x * loc_SQRT1_2 + %s.y * loc_SQRT1_2;\n", iw, w, w);
				sprintf(output + strlen(output), "	%s.y = %s.y * loc_SQRT1_2 - %s.x * loc_SQRT1_2;\n\n", iw, w, w);
			}
			else {
				sprintf(output + strlen(output), "	%s.x = %s.x * loc_SQRT1_2 - %s.y * loc_SQRT1_2;\n", iw, w, w);
				sprintf(output + strlen(output), "	%s.y = %s.y * loc_SQRT1_2 + %s.x * loc_SQRT1_2;\n\n", iw, w, w);
			}
			VkMulComplex(sc, temp, regID[5], iw, 0);
			VkSubComplex(sc, regID[5], regID[4], temp);
			VkAddComplex(sc, regID[4], regID[4], temp);
			//sprintf(output + strlen(output), "\
	temp.x = temp%s.x * iw.x - temp%s.y * iw.y;\n\
	temp.y = temp%s.y * iw.x + temp%s.x * iw.y;\n\
	temp%s = temp%s - temp;\n\
	temp%s = temp%s + temp;\n\n", regID[5], regID[5], regID[5], regID[5], regID[5], regID[4], regID[4], regID[4]);
			if (stageAngle < 0) {
				sprintf(output + strlen(output), "	%s.x = %s.y;\n", w, iw);
				sprintf(output + strlen(output), "	%s.y = -%s.x;\n", w, iw);
				//sprintf(output + strlen(output), "	w = %s(iw.y, -iw.x);\n\n", vecType);
			}
			else {
				sprintf(output + strlen(output), "	%s.x = -%s.y;\n", w, iw);
				sprintf(output + strlen(output), "	%s.y = %s.x;\n", w, iw);
				//sprintf(output + strlen(output), "	w = %s(-iw.y, iw.x);\n\n", vecType);
			}
			VkMulComplex(sc, temp, regID[7], w, 0);
			VkSubComplex(sc, regID[7], regID[6], temp);
			VkAddComplex(sc, regID[6], regID[6], temp);
			VkMovComplex(sc, temp, regID[1]);
			VkMovComplex(sc, regID[1], regID[4]);
			VkMovComplex(sc, regID[4], temp);
			VkMovComplex(sc, temp, regID[3]);
			VkMovComplex(sc, regID[3], regID[6]);
			VkMovComplex(sc, regID[6], temp);
			//sprintf(output + strlen(output), "\
	temp.x = temp%s.x * w.x - temp%s.y * w.y;\n\
	temp.y = temp%s.y * w.x + temp%s.x * w.y;\n\
	temp%s = temp%s - temp;\n\
	temp%s = temp%s + temp;\n\n\
	temp = temp%s;\n\
	temp%s = temp%s;\n\
	temp%s = temp;\n\n\
	temp = temp%s;\n\
	temp%s = temp%s;\n\
	temp%s = temp;\n\
}\n\n", regID[7], regID[7], regID[7], regID[7], regID[7], regID[6], regID[6], regID[6], regID[1], regID[1], regID[4], regID[4], regID[3], regID[3], regID[6], regID[6]);
			//VkAppendLine(sc, "	}\n");

			break;
		}
		case 11: {
			
			char* tf[20];
			//char* tf2[4];
			//char* tf2inv[4];
			//VkAppendLine(sc, "	{\n");
			for (uint32_t i = 0; i < 20; i++) {
				tf[i] = (char*)malloc(sizeof(char) * 40);
				//tf2[i] = (char*)malloc(sizeof(char) * 40);
				//tf2inv[i] = (char*)malloc(sizeof(char) * 40);
			}
			sprintf(tf[0], "-1.100000000000000%s", LFending);
			
			sprintf(tf[2], "0.253097611605959%s", LFending);
			sprintf(tf[3], "-1.288200610773679%s", LFending);
			sprintf(tf[4], "0.304632239669212%s", LFending);
			sprintf(tf[5], "-0.391339615511917%s", LFending);
			sprintf(tf[6], "-2.871022253392850%s", LFending);
			sprintf(tf[7], "1.374907986616384%s", LFending);
			sprintf(tf[8], "0.817178135341212%s", LFending);
			sprintf(tf[9], "1.800746506445679%s", LFending);
			sprintf(tf[10], "-0.859492973614498%s", LFending);

			if (stageAngle < 0) {
				sprintf(tf[1], "0.331662479035540%s", LFending);
				sprintf(tf[11], "-2.373470454748280%s", LFending);
				sprintf(tf[12], "-0.024836393087493%s", LFending);
				sprintf(tf[13], "0.474017017512829%s", LFending);
				sprintf(tf[14], "0.742183927770612%s", LFending);
				sprintf(tf[15], "1.406473309094609%s", LFending);
				sprintf(tf[16], "-1.191364552195948%s", LFending);
				sprintf(tf[17], "0.708088885039503%s", LFending);
				sprintf(tf[18], "0.258908260614168%s", LFending);
				sprintf(tf[19], "-0.049929922194110%s", LFending);
			}
			else {
				sprintf(tf[1], "-0.331662479035540%s", LFending);
				sprintf(tf[11], "2.373470454748280%s", LFending);
				sprintf(tf[12], "0.024836393087493%s", LFending);
				sprintf(tf[13], "-0.474017017512829%s", LFending);
				sprintf(tf[14], "-0.742183927770612%s", LFending);
				sprintf(tf[15], "-1.406473309094609%s", LFending);
				sprintf(tf[16], "1.191364552195948%s", LFending);
				sprintf(tf[17], "-0.708088885039503%s", LFending);
				sprintf(tf[18], "-0.258908260614168%s", LFending);
				sprintf(tf[19], "0.049929922194110%s", LFending);
			}
			for (uint32_t i = radix - 1; i > 0; i--) {
				if (i == radix - 1) {
					if (sc->LUT) {
						sprintf(output + strlen(output), "	%s = twiddleLUT[LUTId];\n", w);
						if (!sc->inverse)
							sprintf(output + strlen(output), "	%s.y = -%s.y;\n", w, w);
					}
					else {
						if (!strcmp(floatType, "float")) {
							sprintf(output + strlen(output), "	%s.x = %s(angle*%.17f);\n", w, cosDef, 2.0 * i / radix);
							sprintf(output + strlen(output), "	%s.y = %s(angle*%.17f);\n", w, sinDef, 2.0 * i / radix);
							//sprintf(output + strlen(output), "	w = %s(cos(angle*%.17f), sin(angle*%.17f));\n\n", vecType, 2.0 * i / radix, 2.0 * i / radix);
						}
						if (!strcmp(floatType, "double"))
							sprintf(output + strlen(output), "	%s = sincos_20(angle*%.17f);\n", w, 2.0 * i / radix);
					}
				}
				else {
					if (sc->LUT) {
						sprintf(output + strlen(output), "	%s = twiddleLUT[LUTId+%d];\n\n", w, (radix - 1 - i) * stageSize);
						if (!sc->inverse)
							sprintf(output + strlen(output), "	%s.y = -%s.y;\n", w, w);
					}
					else {
						if (!strcmp(floatType, "float")) {
							sprintf(output + strlen(output), "	%s.x = %s(angle*%.17f);\n", w, cosDef, 2.0 * i / radix);
							sprintf(output + strlen(output), "	%s.y = %s(angle*%.17f);\n", w, sinDef, 2.0 * i / radix);
							//sprintf(output + strlen(output), "	w = %s(cos(angle*%.17f), sin(angle*%.17f));\n\n", vecType, 2.0 * i / radix, 2.0 * i / radix);
						}
						if (!strcmp(floatType, "double"))
							sprintf(output + strlen(output), "	%s = sincos_20(angle*%.17f);\n", w, 2.0 * i / radix);
					}
				}
				VkMulComplex(sc, sc->locID[i], regID[i], w, 0);
				
			}
			VkMovComplex(sc, sc->locID[0], regID[0]);
			uint32_t permute[11] = { 0,1,9,4,3,5,10,2,7,8,6 };
			VkPermute(sc, permute, 11, 0, 0);
			for (uint32_t i = 0; i < 5; i++) {
				VkAddComplex(sc, regID[i + 1], sc->locID[i+1], sc->locID[i+6]);
				VkSubComplex(sc, regID[i + 6], sc->locID[i + 1], sc->locID[i + 6]);
			}
			VkMovComplex(sc, sc->locID[1], regID[1]);
			for (uint32_t i = 0; i < 4; i++) {
				VkAddComplex(sc, sc->locID[1], sc->locID[1], regID[i + 2]);
				VkSubComplex(sc, sc->locID[i + 3], regID[i + 1], regID[5]);
			}
			VkMovComplex(sc, sc->locID[2], regID[6]);
			for (uint32_t i = 0; i < 4; i++) {
				VkAddComplex(sc, sc->locID[2], sc->locID[2], regID[i + 7]);
				VkSubComplex(sc, sc->locID[i + 7], regID[i + 6], regID[10]);
			}

			VkAddComplex(sc, regID[0], sc->locID[0], sc->locID[1]);
			VkMulComplexNumber(sc, regID[1], sc->locID[1], tf[0]);
			VkMulComplexNumberImag(sc, regID[2], sc->locID[2], tf[1], sc->locID[0]);
			for (uint32_t k = 0; k < 2; k++) {
				VkAddComplex(sc, regID[k*4+3], sc->locID[k*4+3], sc->locID[k*4+5]);
				VkAddComplex(sc, regID[k*4+4], sc->locID[k*4+4], sc->locID[k*4+6]);
				VkAddComplex(sc, regID[k*4+5], sc->locID[k*4+3], sc->locID[k*4+4]);
				VkAddComplex(sc, regID[k*4+6], sc->locID[k*4+5], sc->locID[k*4+6]);
				VkAddComplex(sc, sc->locID[1], regID[k*4+3], regID[k*4+4]);

				if (k == 0) {
					VkMulComplexNumber(sc, sc->locID[k * 4 + 3], sc->locID[k * 4 + 3], tf[k * 9 + 2]);
					VkMulComplexNumber(sc, sc->locID[k * 4 + 4], sc->locID[k * 4 + 4], tf[k * 9 + 3]);
					VkMulComplexNumber(sc, regID[k * 4 + 5], regID[k * 4 + 5], tf[k * 9 + 4]);
					VkMulComplexNumber(sc, sc->locID[k * 4 + 5], sc->locID[k * 4 + 5], tf[k * 9 + 5]);
					VkMulComplexNumber(sc, sc->locID[k * 4 + 6], sc->locID[k * 4 + 6], tf[k * 9 + 6]);
					VkMulComplexNumber(sc, regID[k * 4 + 6], regID[k * 4 + 6], tf[k * 9 + 7]);
					VkMulComplexNumber(sc, regID[k * 4 + 3], regID[k * 4 + 3], tf[k * 9 + 8]);
					VkMulComplexNumber(sc, regID[k * 4 + 4], regID[k * 4 + 4], tf[k * 9 + 9]);
					VkMulComplexNumber(sc, sc->locID[1], sc->locID[1], tf[k * 9 + 10]);
				}
				else {
					VkMulComplexNumberImag(sc, sc->locID[k * 4 + 3], sc->locID[k * 4 + 3], tf[k * 9 + 2], sc->locID[0]);
					VkMulComplexNumberImag(sc, sc->locID[k * 4 + 4], sc->locID[k * 4 + 4], tf[k * 9 + 3], sc->locID[0]);
					VkMulComplexNumberImag(sc, regID[k * 4 + 5], regID[k * 4 + 5], tf[k * 9 + 4], sc->locID[0]);
					VkMulComplexNumberImag(sc, sc->locID[k * 4 + 5], sc->locID[k * 4 + 5], tf[k * 9 + 5], sc->locID[0]);
					VkMulComplexNumberImag(sc, sc->locID[k * 4 + 6], sc->locID[k * 4 + 6], tf[k * 9 + 6], sc->locID[0]);
					VkMulComplexNumberImag(sc, regID[k * 4 + 6], regID[k * 4 + 6], tf[k * 9 + 7], sc->locID[0]);
					VkMulComplexNumberImag(sc, regID[k * 4 + 3], regID[k * 4 + 3], tf[k * 9 + 8], sc->locID[0]);
					VkMulComplexNumberImag(sc, regID[k * 4 + 4], regID[k * 4 + 4], tf[k * 9 + 9], sc->locID[0]);
					VkMulComplexNumberImag(sc, sc->locID[1], sc->locID[1], tf[k * 9 + 10], sc->locID[0]);
				}
				
				VkAddComplex(sc, sc->locID[k*4+3], sc->locID[k*4+3], regID[k*4+3]);
				VkAddComplex(sc, sc->locID[k*4+5], sc->locID[k*4+5], regID[k*4+3]);

				VkAddComplex(sc, sc->locID[k*4+4], sc->locID[k*4+4], regID[k*4+4]);
				VkAddComplex(sc, sc->locID[k*4+6], sc->locID[k*4+6], regID[k*4+4]);

				VkAddComplex(sc, regID[k*4+5], regID[k*4+5], sc->locID[1]);
				VkAddComplex(sc, regID[k*4+6], regID[k*4+6], sc->locID[1]);

				VkAddComplex(sc, regID[k*4+3], sc->locID[k*4+3], regID[k*4+5]);
				VkAddComplex(sc, regID[k*4+4], sc->locID[k*4+4], regID[k*4+5]);

				VkAddComplex(sc, regID[k*4+5], sc->locID[k*4+5], regID[k*4+6]);
				VkAddComplex(sc, regID[k*4+6], sc->locID[k*4+6], regID[k*4+6]);

			}
			VkAddComplex(sc, regID[1], regID[0], regID[1]);

			VkMovComplex(sc, sc->locID[5], regID[1]);
			for (uint32_t i = 0; i < 4; i++) {
				VkAddComplex(sc, sc->locID[i + 1], regID[1], regID[i + 3]);
				VkSubComplex(sc, sc->locID[5], sc->locID[5], regID[i + 3]);
			}
			VkMovComplex(sc, sc->locID[10], regID[2]);
			for (uint32_t i = 0; i < 4; i++) {
				VkAddComplex(sc, sc->locID[i + 6], regID[2], regID[i + 7]);
				VkSubComplex(sc, sc->locID[10], sc->locID[10], regID[i + 7]);
			}
			for (uint32_t i = 0; i < 5; i++) {
				VkAddComplex(sc, regID[i + 1], sc->locID[i + 1], sc->locID[i + 6]);
				VkSubComplex(sc, regID[i + 6], sc->locID[i + 1], sc->locID[i + 6]);
			}
			uint32_t permute2[11] = { 0,10,1,8,7,9,4,2,3,6,5 };
			VkPermute(sc, permute2, 11, 1, regID);

			for (uint32_t i = 0; i < 20; i++) {
				free(tf[i]);
			}
			break;
		}
		case 13: {

			char* tf[20];
			//char* tf2[4];
			//char* tf2inv[4];
			//VkAppendLine(sc, "	{\n");
			for (uint32_t i = 0; i < 20; i++) {
				tf[i] = (char*)malloc(sizeof(char) * 40);
				//tf2[i] = (char*)malloc(sizeof(char) * 40);
				//tf2inv[i] = (char*)malloc(sizeof(char) * 40);
			}
			sprintf(tf[0], "-1.083333333333333%s", LFending);
			sprintf(tf[1], "-0.300462606288666%s", LFending);
			sprintf(tf[5], "1.007074065727533%s", LFending);
			sprintf(tf[6], "0.731245990975348%s", LFending);
			sprintf(tf[7], "-0.579440018900960%s", LFending);
			sprintf(tf[8], "0.531932498429674%s", LFending);
			sprintf(tf[9], "-0.508814921720398%s", LFending);
			sprintf(tf[10], "-0.007705858903092%s", LFending);

			if (stageAngle < 0) {
				sprintf(tf[2], "-0.749279330626139%s", LFending);
				sprintf(tf[3], "0.401002128321867%s", LFending);
				sprintf(tf[4], "0.174138601152136%s", LFending);
				sprintf(tf[11], "-2.511393318389568%s", LFending);
				sprintf(tf[12], "-1.823546408682421%s", LFending);
				sprintf(tf[13], "1.444979909023996%s", LFending);
				sprintf(tf[14], "-1.344056915177370%s", LFending);
				sprintf(tf[15], "-0.975932420775946%s", LFending);
				sprintf(tf[16], "0.773329778651105%s", LFending);
				sprintf(tf[17], "1.927725116783469%s", LFending);
				sprintf(tf[18], "1.399739414729183%s", LFending);
				sprintf(tf[19], "-1.109154843837551%s", LFending);
			}
			else {
				sprintf(tf[2], "0.749279330626139%s", LFending);
				sprintf(tf[3], "-0.401002128321867%s", LFending);
				sprintf(tf[4], "-0.174138601152136%s", LFending);
				sprintf(tf[11], "2.511393318389568%s", LFending);
				sprintf(tf[12], "1.823546408682421%s", LFending);
				sprintf(tf[13], "-1.444979909023996%s", LFending);
				sprintf(tf[14], "1.344056915177370%s", LFending);
				sprintf(tf[15], "0.975932420775946%s", LFending);
				sprintf(tf[16], "-0.773329778651105%s", LFending);
				sprintf(tf[17], "-1.927725116783469%s", LFending);
				sprintf(tf[18], "-1.399739414729183%s", LFending);
				sprintf(tf[19], "1.109154843837551%s", LFending);
			}
			for (uint32_t i = radix - 1; i > 0; i--) {
				if (i == radix - 1) {
					if (sc->LUT) {
						sprintf(output + strlen(output), "	%s = twiddleLUT[LUTId];\n", w);
						if (!sc->inverse)
							sprintf(output + strlen(output), "	%s.y = -%s.y;\n", w, w);
					}
					else {
						if (!strcmp(floatType, "float")) {
							sprintf(output + strlen(output), "	%s.x = %s(angle*%.17f);\n", w, cosDef, 2.0 * i / radix);
							sprintf(output + strlen(output), "	%s.y = %s(angle*%.17f);\n", w, sinDef, 2.0 * i / radix);
							//sprintf(output + strlen(output), "	w = %s(cos(angle*%.17f), sin(angle*%.17f));\n\n", vecType, 2.0 * i / radix, 2.0 * i / radix);
						}
						if (!strcmp(floatType, "double"))
							sprintf(output + strlen(output), "	%s = sincos_20(angle*%.17f);\n", w, 2.0 * i / radix);
					}
				}
				else {
					if (sc->LUT) {
						sprintf(output + strlen(output), "	%s = twiddleLUT[LUTId+%d];\n\n", w, (radix - 1 - i) * stageSize);
						if (!sc->inverse)
							sprintf(output + strlen(output), "	%s.y = -%s.y;\n", w, w);
					}
					else {
						if (!strcmp(floatType, "float")) {
							sprintf(output + strlen(output), "	%s.x = %s(angle*%.17f);\n", w, cosDef, 2.0 * i / radix);
							sprintf(output + strlen(output), "	%s.y = %s(angle*%.17f);\n", w, sinDef, 2.0 * i / radix);
							//sprintf(output + strlen(output), "	w = %s(cos(angle*%.17f), sin(angle*%.17f));\n\n", vecType, 2.0 * i / radix, 2.0 * i / radix);
						}
						if (!strcmp(floatType, "double"))
							sprintf(output + strlen(output), "	%s = sincos_20(angle*%.17f);\n", w, 2.0 * i / radix);
					}
				}
				VkMulComplex(sc, sc->locID[i], regID[i], w, 0);

			}
			VkMovComplex(sc, sc->locID[0], regID[0]);
			uint32_t permute[13] = { 0,1,3,9,5,2,6,12,10,4,8,11,7 };
			VkPermute(sc, permute, 13, 0, 0);
			for (uint32_t i = 0; i < 6; i++) {
				VkSubComplex(sc, regID[i + 7], sc->locID[i + 1], sc->locID[i + 7]);
				VkAddComplex(sc, sc->locID[i + 1], sc->locID[i + 1], sc->locID[i + 7]);
			}
			for (uint32_t i = 0; i < 3; i++) {
				VkAddComplex(sc, regID[i + 1], sc->locID[i + 1], sc->locID[i + 4]);
				VkSubComplex(sc, regID[i + 4], sc->locID[i + 1], sc->locID[i + 4]);
			}
			for (uint32_t i = 0; i < 4; i++) {
				VkAddComplex(sc, sc->locID[i + 1], regID[i*3 + 1], regID[i * 3 + 2]);
				VkSubComplex(sc, sc->locID[i*2 + 5], regID[i * 3 + 1], regID[i * 3 + 3]);
				VkAddComplex(sc, sc->locID[i + 1], sc->locID[i + 1], regID[i * 3 + 3]);
				VkSubComplex(sc, sc->locID[i * 2 + 6], regID[i * 3 + 2], regID[i * 3 + 3]);
			}

			VkAddComplex(sc, regID[0], sc->locID[0], sc->locID[1]);
			VkMulComplexNumber(sc, regID[1], sc->locID[1], tf[0]);
			VkMulComplexNumber(sc, regID[2], sc->locID[2], tf[1]);
			for (uint32_t k = 0; k < 3; k++) {
				VkAddComplex(sc, regID[k * 2 + 4], sc->locID[k * 2 + 3], sc->locID[k * 2 + 4]);
			
				if (k == 0) {
					VkMulComplexNumberImag(sc, sc->locID[k * 2 + 3], sc->locID[k * 2 + 3], tf[k * 3 + 2], sc->locID[0]);
					VkMulComplexNumberImag(sc, sc->locID[k * 2 + 4], sc->locID[k * 2 + 4], tf[k * 3 + 3], sc->locID[0]);
					VkMulComplexNumberImag(sc, regID[k * 2 + 4], regID[k * 2 + 4], tf[k * 3 + 4], sc->locID[0]);
				}
				else {
					VkMulComplexNumber(sc, sc->locID[k * 2 + 3], sc->locID[k * 2 + 3], tf[k * 3 + 2]);
					VkMulComplexNumber(sc, sc->locID[k * 2 + 4], sc->locID[k * 2 + 4], tf[k * 3 + 3]);
					VkMulComplexNumber(sc, regID[k * 2 + 4], regID[k * 2 + 4], tf[k * 3 + 4]);
				}

				VkAddComplex(sc, regID[k * 2 + 3], sc->locID[k * 2 + 3], regID[k * 2 + 4]);
				VkAddComplex(sc, regID[k * 2 + 4], sc->locID[k * 2 + 4], regID[k * 2 + 4]);

			}
			VkAddComplex(sc, regID[9], sc->locID[9], sc->locID[11]);
			VkAddComplex(sc, regID[10], sc->locID[10], sc->locID[12]);
			VkAddComplex(sc, regID[11], sc->locID[9], sc->locID[10]);
			VkAddComplex(sc, regID[12], sc->locID[11], sc->locID[12]);
			VkAddComplex(sc, sc->locID[1], regID[9], regID[10]);

			VkMulComplexNumberImag(sc, sc->locID[9], sc->locID[9], tf[11], sc->locID[0]);
			VkMulComplexNumberImag(sc, sc->locID[10], sc->locID[10], tf[12], sc->locID[0]);
			VkMulComplexNumberImag(sc, regID[11], regID[11], tf[13], sc->locID[0]);
			VkMulComplexNumberImag(sc, sc->locID[11], sc->locID[11], tf[14], sc->locID[0]);
			VkMulComplexNumberImag(sc, sc->locID[12], sc->locID[12], tf[15], sc->locID[0]);
			VkMulComplexNumberImag(sc, regID[12], regID[12], tf[16], sc->locID[0]);
			VkMulComplexNumberImag(sc, regID[9], regID[9], tf[17], sc->locID[0]);
			VkMulComplexNumberImag(sc, regID[10], regID[10], tf[18], sc->locID[0]);
			VkMulComplexNumberImag(sc, sc->locID[1], sc->locID[1], tf[19], sc->locID[0]);

			VkAddComplex(sc, sc->locID[9], sc->locID[9], regID[9]);
			VkAddComplex(sc, sc->locID[11], sc->locID[11], regID[9]);
			VkAddComplex(sc, sc->locID[10], sc->locID[10], regID[10]);
			VkAddComplex(sc, sc->locID[12], sc->locID[12], regID[10]);
			VkAddComplex(sc, regID[11], regID[11], sc->locID[1]);
			VkAddComplex(sc, regID[12], regID[12], sc->locID[1]);

			VkAddComplex(sc, regID[9], sc->locID[9], regID[11]);
			VkAddComplex(sc, regID[10], sc->locID[10], regID[11]);
			VkAddComplex(sc, regID[11], sc->locID[11], regID[12]);
			VkAddComplex(sc, regID[12], sc->locID[12], regID[12]);

			VkAddComplex(sc, regID[1], regID[0], regID[1]);

			for (uint32_t i = 0; i < 4; i++) {
				VkAddComplex(sc, sc->locID[i * 3 + 1], regID[i + 1], regID[i * 2 + 5]);
				VkSubComplex(sc, sc->locID[i * 3 + 3], regID[i + 1], regID[i * 2 + 5]);
				VkAddComplex(sc, sc->locID[i * 3 + 2], regID[i + 1], regID[i * 2 + 6]);
				VkSubComplex(sc, sc->locID[i * 3 + 3], sc->locID[i * 3 + 3], regID[i * 2 + 6]);
			}
			for (uint32_t i = 0; i < 3; i++) {
				VkAddComplex(sc, regID[i + 1], sc->locID[i + 1], sc->locID[i + 4]);
				VkSubComplex(sc, sc->locID[i + 4], sc->locID[i + 1], sc->locID[i + 4]);
				VkMovComplex(sc, sc->locID[i + 1], regID[i + 1]);
			}
			for (uint32_t i = 0; i < 6; i++) {
				VkAddComplex(sc, regID[i + 1], sc->locID[i + 1], sc->locID[i + 7]);
				VkSubComplex(sc, regID[i + 7], sc->locID[i + 1], sc->locID[i + 7]);
			}
			uint32_t permute2[13] = { 0,12,1,10,5,3,2,8,9,11,4,7,6};
			VkPermute(sc, permute2, 13, 1, regID);

			for (uint32_t i = 0; i < 20; i++) {
				free(tf[i]);
			}
			break;
		}
		}
	}
	static inline void appendSharedMemoryVkFFT(char* output, VkFFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint32_t sharedType) {
		char vecType[30];
		char sharedDefinitions[20] = "";
		uint32_t vecSize = 1;
		uint32_t maxSequenceSharedMemory = 0;
		uint32_t maxSequenceSharedMemoryPow2 = 0;
		if (!strcmp(floatType, "float"))
		{
#if(VKFFT_BACKEND==0)
			sprintf(vecType, "vec2");
			sprintf(sharedDefinitions, "shared");
#elif(VKFFT_BACKEND==1)
			sprintf(vecType, "float2");
			sprintf(sharedDefinitions, "__shared__");
			//sprintf(sharedDefinitions, "extern __shared__");
#elif(VKFFT_BACKEND==2)
			sprintf(vecType, "float2");
			sprintf(sharedDefinitions, "__shared__");
			//sprintf(sharedDefinitions, "extern __shared__");
#endif
			vecSize = 8;
		}
		if (!strcmp(floatType, "double")) {
#if(VKFFT_BACKEND==0)
			sprintf(vecType, "dvec2");
			sprintf(sharedDefinitions, "shared");
#elif(VKFFT_BACKEND==1)
			sprintf(vecType, "double2");
			sprintf(sharedDefinitions, "__shared__");
			//sprintf(sharedDefinitions, "extern __shared__");
#elif(VKFFT_BACKEND==2)
			sprintf(vecType, "double2");
			sprintf(sharedDefinitions, "__shared__");
			//sprintf(sharedDefinitions, "extern __shared__");
#endif
			vecSize = 16;
		}
		maxSequenceSharedMemory = sc->sharedMemSize / vecSize;
		maxSequenceSharedMemoryPow2 = sc->sharedMemSizePow2 / vecSize;
		switch (sharedType) {
		case 0: case 5: case 6://single_c2c + single_r2c
		{
			sc->resolveBankConflictFirstStages = 0;
			sc->sharedStrideBankConflictFirstStages = (sc->fftDim > sc->numSharedBanks / 2) ? sc->fftDim / sc->registerBoost * (sc->numSharedBanks / 2 + 1) / (sc->numSharedBanks / 2) : sc->fftDim / sc->registerBoost;
			sc->sharedStrideReadWriteConflict = (sc->numSharedBanks / 2 <= sc->localSize[1]) ? sc->fftDim / sc->registerBoost + 1 : sc->fftDim / sc->registerBoost + (sc->numSharedBanks / 2) / sc->localSize[1];
			sc->maxSharedStride = (sc->sharedStrideBankConflictFirstStages < sc->sharedStrideReadWriteConflict) ? sc->sharedStrideReadWriteConflict : sc->sharedStrideBankConflictFirstStages;
			sc->maxSharedStride = (((maxSequenceSharedMemory - sc->localSize[1] * sc->fftDim / sc->registerBoost) == 0) || ((sc->fftDim & (sc->fftDim - 1)) != 0)) ? sc->fftDim / sc->registerBoost : sc->maxSharedStride;

			sc->sharedStrideBankConflictFirstStages = (sc->maxSharedStride == sc->fftDim / sc->registerBoost) ? sc->fftDim / sc->registerBoost : sc->sharedStrideBankConflictFirstStages;
			sc->sharedStrideReadWriteConflict = (sc->maxSharedStride == sc->fftDim / sc->registerBoost) ? sc->fftDim / sc->registerBoost : sc->sharedStrideReadWriteConflict;

			//printf("%d %d %d %d %d\n", sc->maxSharedStride, sc->sharedStrideBankConflictFirstStages, sc->sharedStrideReadWriteConflict, sc->localSize[1], sc->fftDim);
			sprintf(output + strlen(output), "%s sharedStride = %d; //to avoid bank conflict if we transpose\n", uintType, sc->sharedStrideReadWriteConflict);
#if(VKFFT_BACKEND==0)
			sprintf(output + strlen(output), "%s %s sdata[%d];// sharedStride - fft size,  gl_WorkGroupSize.y - grouped consequential ffts\n\n", sharedDefinitions, vecType, sc->localSize[1] * sc->maxSharedStride);
#elif(VKFFT_BACKEND==1)
			sprintf(output + strlen(output), "%s %s sdata[%d];// sharedStride - fft size,  gl_WorkGroupSize.y - grouped consequential ffts\n\n", sharedDefinitions, vecType, sc->localSize[1] * sc->maxSharedStride);
			//sprintf(output + strlen(output), "%s %s sdata[];// sharedStride - fft size,  gl_WorkGroupSize.y - grouped consequential ffts\n\n", sharedDefinitions, vecType);
#elif(VKFFT_BACKEND==2)
			sprintf(output + strlen(output), "%s %s sdata[%d];// sharedStride - fft size,  gl_WorkGroupSize.y - grouped consequential ffts\n\n", sharedDefinitions, vecType, sc->localSize[1] * sc->maxSharedStride);
			//sprintf(output + strlen(output), "%s %s sdata[];// sharedStride - fft size,  gl_WorkGroupSize.y - grouped consequential ffts\n\n", sharedDefinitions, vecType);
#endif
			sc->usedSharedMemory = vecSize * sc->localSize[1] * sc->maxSharedStride;
			break;
		}
		case 1: case 2://grouped_c2c + single_c2c_strided
		{
#if(VKFFT_BACKEND==0)
			sprintf(output + strlen(output), "%s %s sdata[%d];\n\n", sharedDefinitions, vecType, sc->localSize[0] * sc->fftDim / sc->registerBoost);
#elif(VKFFT_BACKEND==1)
			sprintf(output + strlen(output), "%s %s sdata[%d];\n\n", sharedDefinitions, vecType, sc->localSize[0] * sc->fftDim / sc->registerBoost);
			//sprintf(output + strlen(output), "%s %s sdata[];\n\n", sharedDefinitions, vecType);
#elif(VKFFT_BACKEND==2)
			sprintf(output + strlen(output), "%s %s sdata[%d];\n\n", sharedDefinitions, vecType, sc->localSize[0] * sc->fftDim / sc->registerBoost);
			//sprintf(output + strlen(output), "%s %s sdata[];\n\n", sharedDefinitions, vecType);
#endif
			sc->usedSharedMemory = vecSize * sc->localSize[0] * sc->fftDim / sc->registerBoost;
			break;
		}
		}
	}
	static inline void appendInitialization(char* output, VkFFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint32_t initType) {
		char vecType[30];
#if(VKFFT_BACKEND==0)
		if (!strcmp(floatType, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "dvec2");
#elif(VKFFT_BACKEND==1)
		if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
#elif(VKFFT_BACKEND==2)
		if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
#endif
		switch (initType) {
		case 0: case 1: case 2: case 5: case 6:
		{
			//sprintf(output + strlen(output), "	uint dum=gl_LocalInvocationID.x;\n");
			uint32_t logicalStoragePerThread = sc->registers_per_thread * sc->registerBoost;
			uint32_t logicalRegistersPerThread = sc->registers_per_thread;
			if (sc->convolutionStep) {
				for (uint32_t i = 0; i < sc->registers_per_thread; i++)
					sprintf(output + strlen(output), "	%s temp_%d;\n", vecType, i);
				for (uint32_t j = 1; j < sc->matrixConvolution; j++) {
					for (uint32_t i = 0; i < sc->min_registers_per_thread; i++)
						sprintf(output + strlen(output), "	%s temp_%d_%d;\n", vecType, i, j);
				}
			}
			else {
				for (uint32_t i = 0; i < sc->registers_per_thread; i++)
					sprintf(output + strlen(output), "	%s temp_%d;\n", vecType, i);
			}
			//sprintf(output + strlen(output), "	uint dum=gl_LocalInvocationID.y;//gl_LocalInvocationID.x/gl_WorkGroupSize.x;\n");
			//sprintf(output + strlen(output), "	dum=dum/gl_LocalInvocationID.x-1;\n");
			//sprintf(output + strlen(output), "	dummy=dummy/gl_LocalInvocationID.x-1;\n");
			sc->regIDs = (char**)malloc(sizeof(char*) * logicalStoragePerThread);
			for (uint32_t i = 0; i < logicalStoragePerThread; i++) {
				sc->regIDs[i] = (char*)malloc(sizeof(char) * 40);
				if (i < logicalRegistersPerThread)
					sprintf(sc->regIDs[i], "temp_%d", i);
				else
					sprintf(sc->regIDs[i], "temp_%d", i);
				//sprintf(sc->regIDs[i], "%d[%d]", i / logicalRegistersPerThread, i % logicalRegistersPerThread);
				//sprintf(sc->regIDs[i], "s[%d]", i - logicalRegistersPerThread);

			}
			if (sc->registerBoost > 1) {
				//sprintf(output + strlen(output), "	%s sort0;\n", vecType);
				//sprintf(output + strlen(output), "	%s temps[%d];\n", vecType, (sc->registerBoost -1)* logicalRegistersPerThread);
				for (uint32_t i = 1; i < sc->registerBoost; i++) {
					//sprintf(output + strlen(output), "	%s temp%d[%d];\n", vecType, i, logicalRegistersPerThread);
					for (uint32_t j = 0; j < sc->registers_per_thread; j++)
						sprintf(output + strlen(output), "	%s temp_%d;\n", vecType, j + i * sc->registers_per_thread);
					/*sprintf(output + strlen(output), "\
	for(uint i=0; i<%d; i++)\n\
		temp%d[i]=%s(dum, dum);\n", logicalRegistersPerThread, i, vecType);*/
				}
			}
			sprintf(output + strlen(output), "	%s w;\n", vecType);
			sprintf(sc->w, "w");
			uint32_t maxNonPow2Radix = 1;
			if (sc->fftDim % 3 == 0) maxNonPow2Radix = 3;
			if (sc->fftDim % 5 == 0) maxNonPow2Radix = 5;
			if (sc->fftDim % 7 == 0) maxNonPow2Radix = 7;
			if (sc->fftDim % 11 == 0) maxNonPow2Radix = 11;
			if (sc->fftDim % 13 == 0) maxNonPow2Radix = 13;
			for (uint32_t i = 0; i < maxNonPow2Radix; i++) {
				sprintf(sc->locID[i], "loc_%d", i);
				sprintf(sc->tempStr, "	%s %s;\n", vecType, sc->locID[i]);
				VkAppendLine(sc, sc->tempStr);
			}
			sprintf(sc->temp, "%s", sc->locID[0]);
			uint32_t useRadix8 = 0;
			for (uint32_t i = 0; i < sc->numStages; i++)
				if (sc->stageRadix[i] == 8) useRadix8 = 1;
			if (useRadix8 == 1) {
				if (maxNonPow2Radix > 1) sprintf(sc->iw, sc->locID[1]);
				else {
					sprintf(output + strlen(output), "	%s iw;\n", vecType);
					sprintf(sc->iw, "iw");
				}
			}
			break;
		}
		}
		//sprintf(output + strlen(output), "	%s %s;\n", vecType, sc->tempReg);
		sprintf(output + strlen(output), "	%s %s;\n", uintType, sc->stageInvocationID);
		sprintf(output + strlen(output), "	%s %s;\n", uintType, sc->blockInvocationID);
		sprintf(output + strlen(output), "	%s %s;\n", uintType, sc->sdataID);
		sprintf(output + strlen(output), "	%s %s;\n", uintType, sc->combinedID);
		sprintf(output + strlen(output), "	%s %s;\n", uintType, sc->inoutID);
		if (sc->LUT)
			sprintf(output + strlen(output), "	%s LUTId=0;\n", uintType);
		else
			sprintf(output + strlen(output), "	%s angle=0;\n", floatType);
		if (((sc->stageStartSize > 1) && (!((sc->stageStartSize > 1) && (!sc->reorderFourStep) && (sc->inverse)))) || (((sc->stageStartSize > 1) && (!sc->reorderFourStep) && (sc->inverse)))) {
			sprintf(output + strlen(output), "	%s mult;\n", vecType);
			sprintf(output + strlen(output), "	mult.x = 0;\n");
			sprintf(output + strlen(output), "	mult.y = 0;\n");
		}
		if (sc->cacheShuffle) {
			sprintf(output + strlen(output), "\
	%s tshuffle= ((%s>>1))%%(%d);\n\
	%s shuffle[%d];\n", uintType, sc->gl_LocalInvocationID_x, sc->registers_per_thread, vecType, sc->registers_per_thread);
			for (uint32_t i = 0; i < sc->registers_per_thread; i++) {
				//sprintf(output + strlen(output), "\
	shuffle[%d];\n", i, vecType);
				sprintf(output + strlen(output), "	shuffle[%d].x = 0;\n", i);
				sprintf(output + strlen(output), "	shuffle[%d].y = 0;\n", i);
			}
		}

	}
	static inline void appendZeropadStart(char* output, VkFFTSpecializationConstantsLayout* sc) {
		//return if sequence is full of zeros from the start
		if ((sc->frequencyZeropadding)) {
			switch (sc->axis_id) {
			case 0: {
				break;
			}
			case 1: {
				if (!sc->supportAxis) {
					char idX[100] = "";
					if (sc->performWorkGroupShift[0])
						sprintf(idX, "(%s + consts.workGroupShiftX * %s)", sc->gl_GlobalInvocationID_x, sc->gl_WorkGroupSize_x);
					else
						sprintf(idX, "%s", sc->gl_GlobalInvocationID_x);
					if (sc->performZeropaddingFull[0])
						if (sc->fft_zeropad_left_full[0] < sc->fft_zeropad_right_full[0])
							sprintf(output + strlen(output), "		if(!((%s >= %d)&&(%s < %d))) {\n", idX, sc->fft_zeropad_left_full[0], idX, sc->fft_zeropad_right_full[0]);

				}
				break;
			}
			case 2: {
				if (!sc->supportAxis) {
					char idY[100] = "";
					if (sc->performWorkGroupShift[1])//y axis is along z workgroup here
						sprintf(idY, "(%s + consts.workGroupShiftZ * %s)", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z);
					else
						sprintf(idY, "%s", sc->gl_GlobalInvocationID_z);

					char idX[100] = "";
					if (sc->performWorkGroupShift[0])
						sprintf(idX, "(%s + consts.workGroupShiftX * %s)", sc->gl_GlobalInvocationID_x, sc->gl_WorkGroupSize_x);
					else
						sprintf(idX, "%s", sc->gl_GlobalInvocationID_x);
					if (sc->performZeropaddingFull[0])
						if (sc->fft_zeropad_left_full[0] < sc->fft_zeropad_right_full[0])
							sprintf(output + strlen(output), "		if(!((%s >= %d)&&(%s < %d))) {\n", idX, sc->fft_zeropad_left_full[0], idX, sc->fft_zeropad_right_full[0]);

					if (sc->performZeropaddingFull[1])
						if (sc->fft_zeropad_left_full[1] < sc->fft_zeropad_right_full[1])
							sprintf(output + strlen(output), "		if(!((%s >= %d)&&(%s < %d))) {\n", idY, sc->fft_zeropad_left_full[1], idY, sc->fft_zeropad_right_full[1]);
				}
				else {
					char idY[100] = "";
					if (sc->performWorkGroupShift[1])//for support axes y is along x workgroup
						sprintf(idY, "(%s + consts.workGroupShiftX * %s)", sc->gl_GlobalInvocationID_x, sc->gl_WorkGroupSize_x);
					else
						sprintf(idY, "%s", sc->gl_GlobalInvocationID_x);
					if (sc->performZeropaddingFull[1])
						if (sc->fft_zeropad_left_full[1] < sc->fft_zeropad_right_full[1])
							sprintf(output + strlen(output), "		if(!((%s >= %d)&&(%s < %d))) {\n", idY, sc->fft_zeropad_left_full[1], idY, sc->fft_zeropad_right_full[1]);
				}
				break;
			}
			}
		}
		else {
			switch (sc->axis_id) {
			case 0: {
				char idY[100] = "";
				if (sc->performWorkGroupShift[1])
					sprintf(idY, "(%s + consts.workGroupShiftY * %s)", sc->gl_GlobalInvocationID_y, sc->gl_WorkGroupSize_y);
				else
					sprintf(idY, "%s", sc->gl_GlobalInvocationID_y);

				char idZ[100] = "";
				if (sc->performWorkGroupShift[2])
					sprintf(idZ, "(%s + consts.workGroupShiftZ * %s)", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z);
				else
					sprintf(idZ, "%s", sc->gl_GlobalInvocationID_z);
				if (sc->performZeropaddingFull[1])
					if (sc->fft_zeropad_left_full[1] < sc->fft_zeropad_right_full[1])
						sprintf(output + strlen(output), "		if(!((%s >= %d)&&(%s < %d))) {\n", idY, sc->fft_zeropad_left_full[1], idY, sc->fft_zeropad_right_full[1]);
				if (sc->performZeropaddingFull[2])
					if (sc->fft_zeropad_left_full[2] < sc->fft_zeropad_right_full[2])
						sprintf(output + strlen(output), "		if(!((%s >= %d)&&(%s < %d))) {\n", idZ, sc->fft_zeropad_left_full[2], idZ, sc->fft_zeropad_right_full[2]);

				break;
			}
			case 1: {
				char idZ[100] = "";
				if (sc->performWorkGroupShift[2])
					sprintf(idZ, "(%s + consts.workGroupShiftZ * %s)", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z);
				else
					sprintf(idZ, "%s", sc->gl_GlobalInvocationID_z);
				if (sc->performZeropaddingFull[2])
					if (sc->fft_zeropad_left_full[2] < sc->fft_zeropad_right_full[2])
						sprintf(output + strlen(output), "		if(!((%s >= %d)&&(%s < %d))) {\n", idZ, sc->fft_zeropad_left_full[2], idZ, sc->fft_zeropad_right_full[2]);

				break;
			}
			case 2: {

				break;
			}
			}
		}
	}
	static inline void appendZeropadEnd(char* output, VkFFTSpecializationConstantsLayout* sc) {
		//return if sequence is full of zeros from the start
		if ((sc->frequencyZeropadding)) {
			switch (sc->axis_id) {
			case 0: {
				break;
			}
			case 1: {
				if (!sc->supportAxis) {
					char idX[100] = "";
					if (sc->performWorkGroupShift[0])
						sprintf(idX, "(%s + consts.workGroupShiftX * %s)", sc->gl_GlobalInvocationID_x, sc->gl_WorkGroupSize_x);
					else
						sprintf(idX, "%s", sc->gl_GlobalInvocationID_x);
					if (sc->performZeropaddingFull[0])
						if (sc->fft_zeropad_left_full[0] < sc->fft_zeropad_right_full[0])
							sprintf(output + strlen(output), "		}\n");

				}
				break;
			}
			case 2: {
				if (!sc->supportAxis) {
					char idY[100] = "";
					if (sc->performWorkGroupShift[1])//y axis is along z workgroup here
						sprintf(idY, "(%s + consts.workGroupShiftZ * %s)", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z);
					else
						sprintf(idY, "%s", sc->gl_GlobalInvocationID_z);

					char idX[100] = "";
					if (sc->performWorkGroupShift[0])
						sprintf(idX, "(%s + consts.workGroupShiftX * %s)", sc->gl_GlobalInvocationID_x, sc->gl_WorkGroupSize_x);
					else
						sprintf(idX, "%s", sc->gl_GlobalInvocationID_x);
					if (sc->performZeropaddingFull[0])
						if (sc->fft_zeropad_left_full[0] < sc->fft_zeropad_right_full[0])
							sprintf(output + strlen(output), "		}\n");
					if (sc->performZeropaddingFull[1])
						if (sc->fft_zeropad_left_full[1] < sc->fft_zeropad_right_full[1])
							sprintf(output + strlen(output), "		}\n");
				}
				else {
					char idY[100] = "";
					if (sc->performWorkGroupShift[1])//for support axes y is along x workgroup
						sprintf(idY, "(%s + consts.workGroupShiftX * %s)", sc->gl_GlobalInvocationID_x, sc->gl_WorkGroupSize_x);
					else
						sprintf(idY, "%s", sc->gl_GlobalInvocationID_x);
					if (sc->performZeropaddingFull[1])
						if (sc->fft_zeropad_left_full[1] < sc->fft_zeropad_right_full[1])
							sprintf(output + strlen(output), "		}\n");
				}
				break;
			}
			}
		}
		else {
			switch (sc->axis_id) {
			case 0: {
				char idY[100] = "";
				if (sc->performWorkGroupShift[1])
					sprintf(idY, "(%s + consts.workGroupShiftY * %s)", sc->gl_GlobalInvocationID_y, sc->gl_WorkGroupSize_y);
				else
					sprintf(idY, "%s", sc->gl_GlobalInvocationID_y);

				char idZ[100] = "";
				if (sc->performWorkGroupShift[2])
					sprintf(idZ, "(%s + consts.workGroupShiftZ * %s)", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z);
				else
					sprintf(idZ, "%s", sc->gl_GlobalInvocationID_z);
				if (sc->performZeropaddingFull[1])
					if (sc->fft_zeropad_left_full[1] < sc->fft_zeropad_right_full[1])
						sprintf(output + strlen(output), "		}\n");
				if (sc->performZeropaddingFull[2])
					if (sc->fft_zeropad_left_full[2] < sc->fft_zeropad_right_full[2])
						sprintf(output + strlen(output), "		}\n");
				break;
			}
			case 1: {
				char idZ[100] = "";
				if (sc->performWorkGroupShift[2])
					sprintf(idZ, "(%s + consts.workGroupShiftZ * %s)", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z);
				else
					sprintf(idZ, "%s", sc->gl_GlobalInvocationID_z);
				if (sc->performZeropaddingFull[2])
					if (sc->fft_zeropad_left_full[2] < sc->fft_zeropad_right_full[2])
						sprintf(output + strlen(output), "		}\n");
				break;
			}
			case 2: {

				break;
			}
			}
		}
	}

	static inline void appendReadDataVkFFT(char* output, VkFFTSpecializationConstantsLayout* sc, const char* floatType, const char* floatTypeMemory, const char* uintType, uint32_t readType) {
		char vecType[30];
		char inputsStruct[20] = "";
#if(VKFFT_BACKEND==0)
		if (sc->inputBufferBlockNum == 1)
			sprintf(inputsStruct, "inputs");
		else
			sprintf(inputsStruct, ".inputs");
		if (!strcmp(floatType, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "dvec2");
#elif(VKFFT_BACKEND==1)
		if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
		sprintf(inputsStruct, "inputs");
#elif(VKFFT_BACKEND==2)
		if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
		sprintf(inputsStruct, "inputs");
#endif
		char convTypeLeft[20] = "";
		char convTypeRight[20] = "";
		if ((!strcmp(floatType, "float")) && (strcmp(floatTypeMemory, "float"))) {
			if (readType == 5) {
#if(VKFFT_BACKEND==0)
				sprintf(convTypeLeft, "float(");
				sprintf(convTypeRight, ")");
#elif(VKFFT_BACKEND==1)
				sprintf(convTypeLeft, "(float)");
				sprintf(convTypeRight, "");
#elif(VKFFT_BACKEND==2)
				sprintf(convTypeLeft, "(float)");
				sprintf(convTypeRight, "");
#endif
			}
			else {
#if(VKFFT_BACKEND==0)
				sprintf(convTypeLeft, "vec2(");
				sprintf(convTypeRight, ")");
#elif(VKFFT_BACKEND==1)
				sprintf(convTypeLeft, "(float2)");
				sprintf(convTypeRight, "");
#elif(VKFFT_BACKEND==2)
				sprintf(convTypeLeft, "(float2)");
				sprintf(convTypeRight, "");
#endif
			}
		}
		if ((!strcmp(floatType, "double")) && (strcmp(floatTypeMemory, "double"))) {
			if (readType == 5) {
#if(VKFFT_BACKEND==0)
				sprintf(convTypeLeft, "double(");
				sprintf(convTypeRight, ")");
#elif(VKFFT_BACKEND==1)
				sprintf(convTypeLeft, "(double)");
				sprintf(convTypeRight, "");
#elif(VKFFT_BACKEND==2)
				sprintf(convTypeLeft, "(double)");
				sprintf(convTypeRight, "");
#endif
			}
			else {
#if(VKFFT_BACKEND==0)
				sprintf(convTypeLeft, "dvec2(");
				sprintf(convTypeRight, ")");
#elif(VKFFT_BACKEND==1)
				sprintf(convTypeLeft, "(double2)");
				sprintf(convTypeRight, "");
#elif(VKFFT_BACKEND==2)
				sprintf(convTypeLeft, "(double2)");
				sprintf(convTypeRight, "");
#endif
			}
		}
		char tempNum[50] = "";
		char index_x[500] = "";
		char index_y[500] = "";
		char requestCoordinate[100] = "";
		if (sc->convolutionStep) {
			if (sc->matrixConvolution > 1) {
				sprintf(requestCoordinate, "coordinate");
			}
		}
		char requestBatch[100] = "";
		if (sc->convolutionStep) {
			if (sc->numKernels > 1) {
				sprintf(requestBatch, "0");//if one buffer - multiple kernel convolution
			}
		}
		appendZeropadStart(output, sc);
		switch (readType) {
		case 0://single_c2c
		{
			//sprintf(output + strlen(output), "	return;\n");
			char shiftX[500] = "";
			if (sc->performWorkGroupShift[0])
				sprintf(shiftX, " + consts.workGroupShiftX ");
			char shiftY[500] = "";
			if (sc->performWorkGroupShift[1])
				sprintf(shiftY, " + consts.workGroupShiftY*%s ", sc->gl_WorkGroupSize_y);
			char shiftY2[100] = "";
			if (sc->performWorkGroupShift[1])
				sprintf(shiftY, " + consts.workGroupShiftY ");
			if (sc->fftDim < sc->fft_dim_full) {
				sprintf(sc->disableThreadsStart, "		if(%s * %d + (((%s%s) %% %d) * %d + ((%s%s) / %d) * %d) < %d) {\n", sc->gl_LocalInvocationID_y, sc->firstStageStartSize, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fftDim, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->localSize[1] * sc->firstStageStartSize, sc->fft_dim_full);
				VkAppendLine(sc, sc->disableThreadsStart);
				sprintf(sc->disableThreadsEnd, "}");
			}
			else {
				sprintf(output + strlen(output), "		{ \n");
			}

			if ((sc->localSize[1] > 1) || ((sc->performR2C) && (sc->inverse)) || (sc->localSize[0] * sc->stageRadix[0] * (sc->registers_per_thread_per_radix[sc->stageRadix[0]] / sc->stageRadix[0]) > sc->fftDim))
				sc->readToRegisters = 0;
			else
				sc->readToRegisters = 1;
			if (sc->zeropad[0]) {
				if (sc->fftDim == sc->fft_dim_full) {
					for (uint32_t k = 0; k < sc->registerBoost; k++) {
						for (uint32_t i = 0; i < sc->min_registers_per_thread; i++) {
							/*if (sc->localSize[1] == 1) {
								sprintf(tempNum, "%d", (i + k * sc->min_registers_per_thread) * sc->localSize[0]);
								VkAddReal(sc, sc->combinedID, sc->gl_LocalInvocationID_x, tempNum);
							}
							else {
								sprintf(tempNum, "%d", sc->localSize[0]);
								VkFMAReal(sc, sc->combinedID, tempNum, sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x);
								sprintf(tempNum, "%d", (i + k * sc->min_registers_per_thread) * sc->localSize[0] * sc->localSize[1]);
								VkAddReal(sc, sc->combinedID, sc->combinedID, tempNum);
							}*/
							if (sc->localSize[1] == 1)
								sprintf(output + strlen(output), "		combinedID = %s + %d;\n", sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0]);
							else
								sprintf(output + strlen(output), "		combinedID = (%s + %d * %s) + %d;\n", sc->gl_LocalInvocationID_x, sc->localSize[0], sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[0] * sc->localSize[1]);
							/*
							sprintf(tempNum, "%d", sc->fftDim);
							VkModReal(sc, sc->inoutID, sc->combinedID, tempNum);
							VkDivReal(sc, sc->tempReg, sc->combinedID, tempNum);
							if (sc->inputStride[0] > 1) {
								sprintf(tempNum, "%d", sc->inputStride[0]);
								VkMulReal(sc, sc->inoutID, sc->inoutID, tempNum);
							}
							sprintf(tempNum, "%d", sc->inputStride[1]);
							VkFMAReal(sc, sc->inoutID, sc->tempReg, tempNum, sc->inoutID);
							*/
							if (sc->inputStride[0] > 1)
								sprintf(output + strlen(output), "		inoutID = (combinedID %% %d) * %d + (combinedID / %d) * %d;\n", sc->fftDim, sc->inputStride[0], sc->fftDim, sc->inputStride[1]);
							else
								sprintf(output + strlen(output), "		inoutID = (combinedID %% %d) + (combinedID / %d) * %d;\n", sc->fftDim, sc->fftDim, sc->inputStride[1]);

							if (sc->size[sc->axis_id + 1] % sc->localSize[1] != 0)
								sprintf(output + strlen(output), "		if(combinedID / %d + (%s%s)*%s< %d){", sc->fftDim, sc->gl_WorkGroupID_y, shiftY2, sc->gl_WorkGroupSize_y, sc->size[sc->axis_id + 1]);

							sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d)){\n", sc->fft_dim_full, sc->fft_zeropad_left_read[sc->axis_id], sc->fft_dim_full, sc->fft_zeropad_right_read[sc->axis_id]);
							sprintf(output + strlen(output), "			%s = ", sc->inoutID);
							indexInputVkFFT(output, sc, uintType, readType, sc->inoutID, 0, requestCoordinate, requestBatch);
							sprintf(output + strlen(output), ";\n");
							if (sc->readToRegisters) {
								if (sc->inputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		%s = %s%s[%s]%s;\n", sc->regIDs[i + k * sc->registers_per_thread], convTypeLeft, inputsStruct, sc->inoutID, convTypeRight);
								else
									sprintf(output + strlen(output), "		%s = %sinputBlocks[%s / %d]%s[%s %% %d]%s;\n", sc->regIDs[i + k * sc->registers_per_thread], convTypeLeft, sc->inoutID, sc->inputBufferBlockSize, inputsStruct, sc->inoutID, sc->inputBufferBlockSize, convTypeRight);
							}
							else {
								if (sc->inputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		sdata[(combinedID %% %d) + (combinedID / %d) * sharedStride] = %s%s[%s]%s;\n", sc->fftDim, sc->fftDim, convTypeLeft, inputsStruct, sc->inoutID, convTypeRight);
								else
									sprintf(output + strlen(output), "		sdata[(combinedID %% %d) + (combinedID / %d) * sharedStride] = %sinputBlocks[%s / %d]%s[%s %% %d]%s;\n", sc->fftDim, sc->fftDim, convTypeLeft, sc->inoutID, sc->inputBufferBlockSize, inputsStruct, sc->inoutID, sc->inputBufferBlockSize, convTypeRight);

							}
							sprintf(output + strlen(output), "		}else{\n");
							if (sc->readToRegisters)
								sprintf(output + strlen(output), "			%s.x =0;%s.y = 0;\n", sc->regIDs[i + k * sc->registers_per_thread], sc->regIDs[i + k * sc->registers_per_thread]);
							else {
								sprintf(output + strlen(output), "			sdata[(combinedID %% %d) + (combinedID / %d) * sharedStride].x = 0;\n", sc->fftDim, sc->fftDim);
								sprintf(output + strlen(output), "			sdata[(combinedID %% %d) + (combinedID / %d) * sharedStride].y = 0;\n", sc->fftDim, sc->fftDim);
							}
							sprintf(output + strlen(output), "		}\n");
							if (sc->size[sc->axis_id + 1] % sc->localSize[1] != 0)
								sprintf(output + strlen(output), "		}");

						}
					}
				}
				else {
					for (uint32_t k = 0; k < sc->registerBoost; k++) {
						for (uint32_t i = 0; i < sc->min_registers_per_thread; i++) {
							sprintf(output + strlen(output), "		inoutID = %s+%d+%s * %d + (((%s%s) %% %d) * %d + ((%s%s) / %d) * %d);\n", sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0], sc->gl_LocalInvocationID_y, sc->firstStageStartSize, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fftDim, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->localSize[1] * sc->firstStageStartSize);
							sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d)){\n", sc->fft_dim_full, sc->fft_zeropad_left_read[sc->axis_id], sc->fft_dim_full, sc->fft_zeropad_right_read[sc->axis_id]);
							sprintf(output + strlen(output), "			%s = ", sc->inoutID);
							indexInputVkFFT(output, sc, uintType, readType, sc->inoutID, 0, requestCoordinate, requestBatch);
							sprintf(output + strlen(output), ";\n");
							if (sc->readToRegisters) {
								if (sc->inputBufferBlockNum == 1)
									sprintf(output + strlen(output), "			%s = %s%s[%s]%s;\n", sc->regIDs[i + k * sc->registers_per_thread], convTypeLeft, inputsStruct, sc->inoutID, convTypeRight);
								else
									sprintf(output + strlen(output), "			%s = %sinputBlocks[%s / %d]%s[%s %% %d]%s;\n", sc->regIDs[i + k * sc->registers_per_thread], convTypeLeft, sc->inoutID, sc->inputBufferBlockSize, inputsStruct, sc->inoutID, sc->inputBufferBlockSize, convTypeRight);
							}
							else {
								if (sc->inputBufferBlockNum == 1)
									sprintf(output + strlen(output), "			sdata[sharedStride*%s + (%s + %d)] = %s%s[%s]%s;\n", sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0], convTypeLeft, inputsStruct, sc->inoutID, convTypeRight);
								else
									sprintf(output + strlen(output), "			sdata[sharedStride*%s + (%s + %d)] = %sinputBlocks[%s / %d]%s[%s %% %d]%s;\n", sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0], convTypeLeft, sc->inoutID, sc->inputBufferBlockSize, inputsStruct, sc->inoutID, sc->inputBufferBlockSize, convTypeRight);
							}
							sprintf(output + strlen(output), "		}\n");
							sprintf(output + strlen(output), "		else{\n");
							if (sc->readToRegisters)
								sprintf(output + strlen(output), "			%s.x = 0; %s.y = 0;\n", sc->regIDs[i + k * sc->registers_per_thread], sc->regIDs[i + k * sc->registers_per_thread]);
							else {
								sprintf(output + strlen(output), "			sdata[sharedStride*%s + (%s + %d)].x = 0;\n", sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0]);
								sprintf(output + strlen(output), "			sdata[sharedStride*%s + (%s + %d)].y = 0;\n", sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0]);
							}
							sprintf(output + strlen(output), "		}\n");
						}
					}
				}
			}
			else {
				if (sc->fftDim == sc->fft_dim_full) {
					for (uint32_t k = 0; k < sc->registerBoost; k++) {
						for (uint32_t i = 0; i < sc->min_registers_per_thread; i++) {
							if (sc->localSize[1] == 1)
								sprintf(output + strlen(output), "		combinedID = %s + %d;\n", sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0]);
							else
								sprintf(output + strlen(output), "		combinedID = (%s + %d * %s) + %d;\n", sc->gl_LocalInvocationID_x, sc->localSize[0], sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[0] * sc->localSize[1]);

							if (sc->inputStride[0] > 1) {
								sprintf(output + strlen(output), "			%s = ", sc->inoutID);
								sprintf(index_x, "(combinedID %% %d) * %d + (combinedID / %d) * %d", sc->fftDim, sc->inputStride[0], sc->fftDim, sc->inputStride[1]);
								indexInputVkFFT(output, sc, uintType, readType, index_x, 0, requestCoordinate, requestBatch);
								sprintf(output + strlen(output), ";\n");
								//sprintf(output + strlen(output), "		inoutID = indexInput((combinedID %% %d) * %d + (combinedID / %d) * %d%s%s);\n", sc->fftDim, sc->inputStride[0], sc->fftDim, sc->inputStride[1], requestCoordinate, requestBatch);
							}
							else {
								sprintf(output + strlen(output), "			%s = ", sc->inoutID);
								sprintf(index_x, "(combinedID %% %d) + (combinedID / %d) * %d", sc->fftDim, sc->fftDim, sc->inputStride[1]);
								indexInputVkFFT(output, sc, uintType, readType, index_x, 0, requestCoordinate, requestBatch);
								sprintf(output + strlen(output), ";\n");
								//sprintf(output + strlen(output), "		inoutID = indexInput((combinedID %% %d) + (combinedID / %d) * %d%s%s);\n", sc->fftDim, sc->fftDim, sc->inputStride[1], requestCoordinate, requestBatch);
							}
							if (sc->size[sc->axis_id + 1] % sc->localSize[1] != 0)
								sprintf(output + strlen(output), "		if(combinedID / %d + (%s%s)*%s< %d){", sc->fftDim, sc->gl_WorkGroupID_y, shiftY2, sc->gl_WorkGroupSize_y, sc->size[sc->axis_id + 1]);
							if (sc->readToRegisters) {
								if (sc->inputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		%s = %s%s[inoutID]%s;\n", sc->regIDs[i + k * sc->registers_per_thread], convTypeLeft, inputsStruct, convTypeRight);
								else
									sprintf(output + strlen(output), "		%s = %sinputBlocks[inoutID / %d]%s[inoutID %% %d]%s;\n", sc->regIDs[i + k * sc->registers_per_thread], convTypeLeft, sc->inputBufferBlockSize, inputsStruct, sc->inputBufferBlockSize, convTypeRight);
							}
							else {
								if (sc->inputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		sdata[(combinedID %% %d) + (combinedID / %d) * sharedStride] = %s%s[inoutID]%s;\n", sc->fftDim, sc->fftDim, convTypeLeft, inputsStruct, convTypeRight);
								else
									sprintf(output + strlen(output), "		sdata[(combinedID %% %d) + (combinedID / %d) * sharedStride] = %sinputBlocks[inoutID / %d]%s[inoutID %% %d]%s;\n", sc->fftDim, sc->fftDim, convTypeLeft, sc->inputBufferBlockSize, inputsStruct, sc->inputBufferBlockSize, convTypeRight);
							}
							if (sc->size[sc->axis_id + 1] % sc->localSize[1] != 0)
								sprintf(output + strlen(output), "		}");
						}
					}

				}
				else {
					for (uint32_t k = 0; k < sc->registerBoost; k++) {
						for (uint32_t i = 0; i < sc->min_registers_per_thread; i++) {
							sprintf(output + strlen(output), "			%s = ", sc->inoutID);
							sprintf(index_x, "%s+%d+%s * %d + (((%s%s) %% %d) * %d + ((%s%s) / %d) * %d)", sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0], sc->gl_LocalInvocationID_y, sc->firstStageStartSize, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fftDim, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->localSize[1] * sc->firstStageStartSize);
							indexInputVkFFT(output, sc, uintType, readType, index_x, 0, requestCoordinate, requestBatch);
							sprintf(output + strlen(output), ";\n");
							//sprintf(output + strlen(output), "		inoutID = indexInput(%s+%d+%s * %d + (((%s%s) %% %d) * %d + ((%s%s) / %d) * %d)%s%s);\n", sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0], sc->gl_LocalInvocationID_y, sc->firstStageStartSize, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fftDim, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->localSize[1] * sc->firstStageStartSize, requestCoordinate, requestBatch);
							if (sc->readToRegisters) {
								if (sc->inputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		%s = %s%s[inoutID]%s;\n", sc->regIDs[i + k * sc->registers_per_thread], convTypeLeft, inputsStruct, convTypeRight);
								else
									sprintf(output + strlen(output), "		%s = %sinputBlocks[inoutID / %d]%s[inoutID %% %d]%s;\n", sc->regIDs[i + k * sc->registers_per_thread], convTypeLeft, sc->inputBufferBlockSize, inputsStruct, sc->inputBufferBlockSize, convTypeRight);
							}
							else {
								if (sc->inputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		sdata[sharedStride*%s + (%s + %d)] = %s%s[inoutID]%s;\n", sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0], convTypeLeft, inputsStruct, convTypeRight);
								else
									sprintf(output + strlen(output), "		sdata[sharedStride*%s + (%s + %d)] = %sinputBlocks[inoutID / %d]%s[inoutID %% %d]%s;\n", sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0], convTypeLeft, sc->inputBufferBlockSize, inputsStruct, sc->inputBufferBlockSize, convTypeRight);
							}
						}
					}
				}
			}
			VkAppendLine(sc, "	}\n");
			break;
		}
		case 1://grouped_c2c
		{
			if (sc->localSize[1] * sc->stageRadix[0] * (sc->registers_per_thread_per_radix[sc->stageRadix[0]] / sc->stageRadix[0]) > sc->fftDim)
				sc->readToRegisters = 0;
			else
				sc->readToRegisters = 1;
			char shiftX[500] = "";
			if (sc->performWorkGroupShift[0])
				sprintf(shiftX, " + consts.workGroupShiftX * %s ", sc->gl_WorkGroupSize_x);

			sprintf(sc->disableThreadsStart, "		if (((%s%s) / %d) %% (%d)+((%s%s) / %d) * (%d) < %d) {\n", sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->stageStartSize, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x * sc->stageStartSize, sc->fftDim * sc->stageStartSize, sc->size[sc->axis_id]);
			VkAppendLine(sc, sc->disableThreadsStart);
			sprintf(sc->disableThreadsEnd, "}");

			if (sc->zeropad[0]) {
				for (uint32_t k = 0; k < sc->registerBoost; k++) {
					for (uint32_t i = 0; i < sc->min_registers_per_thread; i++) {
						sprintf(output + strlen(output), "		inoutID = (%d * (%s + %d) + ((%s%s) / %d) %% (%d)+((%s%s) / %d) * (%d));\n", sc->stageStartSize, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->stageStartSize, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x * sc->stageStartSize, sc->fftDim * sc->stageStartSize);

						sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d)){\n", sc->fft_dim_full, sc->fft_zeropad_left_read[sc->axis_id], sc->fft_dim_full, sc->fft_zeropad_right_read[sc->axis_id]);
						sprintf(output + strlen(output), "			%s = ", sc->inoutID);
						sprintf(index_x, "(%s%s) %% (%d)", sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x);
						indexInputVkFFT(output, sc, uintType, readType, index_x, sc->inoutID, requestCoordinate, requestBatch);
						sprintf(output + strlen(output), ";\n");
						if (sc->readToRegisters) {
							if (sc->inputBufferBlockNum == 1)
								sprintf(output + strlen(output), "			%s=%s%s[%s]%s;\n", sc->regIDs[i + k * sc->registers_per_thread], convTypeLeft, inputsStruct, sc->inoutID, convTypeRight);
							else
								sprintf(output + strlen(output), "			%s=%sinputBlocks[%s / %d]%s[%s %% %d]%s;\n", sc->regIDs[i + k * sc->registers_per_thread], convTypeLeft, sc->inoutID, sc->inputBufferBlockSize, inputsStruct, sc->inoutID, sc->inputBufferBlockSize, convTypeRight);
						}
						else {
							if (sc->inputBufferBlockNum == 1)
								sprintf(output + strlen(output), "			sdata[%s*(%s+%d)+%s]=%s%s[%s]%s;\n", sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x, convTypeLeft, inputsStruct, sc->inoutID, convTypeRight);
							else
								sprintf(output + strlen(output), "			sdata[%s*(%s+%d)+%s]=%sinputBlocks[%s / %d]%s[%s %% %d]%s;\n", sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x, convTypeLeft, sc->inoutID, sc->inputBufferBlockSize, inputsStruct, sc->inoutID, sc->inputBufferBlockSize, convTypeRight);
						}
						sprintf(output + strlen(output), "		}\n");
						sprintf(output + strlen(output), "		else{\n");
						if (sc->readToRegisters)
							sprintf(output + strlen(output), "			%s.x = 0; %s.y = 0;\n", sc->regIDs[i + k * sc->registers_per_thread], sc->regIDs[i + k * sc->registers_per_thread]);
						else {
							sprintf(output + strlen(output), "			sdata[%s*(%s+%d)+%s].x=0;\n", sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x);
							sprintf(output + strlen(output), "			sdata[%s*(%s+%d)+%s].y=0;\n", sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x);
						}
						sprintf(output + strlen(output), "		}\n");
					}
				}
			}
			else {
				for (uint32_t k = 0; k < sc->registerBoost; k++) {
					for (uint32_t i = 0; i < sc->min_registers_per_thread; i++) {
						sprintf(output + strlen(output), "			%s = ", sc->inoutID);
						sprintf(index_x, "(%s%s) %% (%d)", sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x);
						sprintf(index_y, "%d * (%s + %d) + ((%s%s) / %d) %% (%d)+((%s%s) / %d) * (%d)", sc->stageStartSize, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->stageStartSize, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x * sc->stageStartSize, sc->fftDim * sc->stageStartSize);
						indexInputVkFFT(output, sc, uintType, readType, index_x, index_y, requestCoordinate, requestBatch);
						sprintf(output + strlen(output), ";\n");
						//sprintf(output + strlen(output), "		inoutID = indexInput((%s%s) %% (%d), %d * (%s + %d) + ((%s%s) / %d) %% (%d)+((%s%s) / %d) * (%d)%s%s);\n", sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->stageStartSize, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->stageStartSize, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x * sc->stageStartSize, sc->fftDim * sc->stageStartSize, requestCoordinate, requestBatch);
						if (sc->readToRegisters) {
							if (sc->inputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		%s = %s%s[inoutID]%s;\n", sc->regIDs[i + k * sc->registers_per_thread], convTypeLeft, inputsStruct, convTypeRight);
							else
								sprintf(output + strlen(output), "		%s = %sinputBlocks[inoutID / %d]%s[inoutID %% %d]%s;\n", sc->regIDs[i + k * sc->registers_per_thread], convTypeLeft, sc->inputBufferBlockSize, inputsStruct, sc->inputBufferBlockSize, convTypeRight);
						}
						else {
							if (sc->inputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		sdata[%s*(%s+%d)+%s] = %s%s[inoutID]%s;\n", sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x, convTypeLeft, inputsStruct, convTypeRight);
							else
								sprintf(output + strlen(output), "		sdata[%s*(%s+%d)+%s] = %sinputBlocks[inoutID / %d]%s[inoutID %% %d]%s;\n", sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x, convTypeLeft, sc->inputBufferBlockSize, inputsStruct, sc->inputBufferBlockSize, convTypeRight);
						}
					}
				}
			}
			VkAppendLine(sc, "	}\n");
			break;
		}
		case 2://single_c2c_strided
		{
			if (sc->localSize[1] * sc->stageRadix[0] * (sc->registers_per_thread_per_radix[sc->stageRadix[0]] / sc->stageRadix[0]) > sc->fftDim)
				sc->readToRegisters = 0;
			else
				sc->readToRegisters = 1;
			char shiftX[500] = "";
			if (sc->performWorkGroupShift[0])
				sprintf(shiftX, " + consts.workGroupShiftX * %s ", sc->gl_WorkGroupSize_x);

			//sprintf(output + strlen(output), "		if(gl_GlobalInvolcationID.x%s >= %d) return; \n", shiftX, sc->size[0] / axis->specializationConstants.fftDim);
			sprintf(sc->disableThreadsStart, "		if (((%s%s) / %d) * (%d) < %d) {\n", sc->gl_GlobalInvocationID_x, shiftX, sc->stageStartSize, sc->stageStartSize * sc->fftDim, sc->fft_dim_full);
			VkAppendLine(sc, sc->disableThreadsStart);
			sprintf(sc->disableThreadsEnd, "}");
			if (sc->zeropad[0]) {
				for (uint32_t k = 0; k < sc->registerBoost; k++) {
					for (uint32_t i = 0; i < sc->min_registers_per_thread; i++) {
						sprintf(output + strlen(output), "		inoutID = (%s%s) %% (%d) + %d * (%s + %d) + ((%s%s) / %d) * (%d);\n", sc->gl_GlobalInvocationID_x, shiftX, sc->stageStartSize, sc->stageStartSize, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_GlobalInvocationID_x, shiftX, sc->stageStartSize, sc->stageStartSize * sc->fftDim);
						sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d)){\n", sc->fft_dim_full, sc->fft_zeropad_left_read[sc->axis_id], sc->fft_dim_full, sc->fft_zeropad_right_read[sc->axis_id]);
						sprintf(output + strlen(output), "			%s = ", sc->inoutID);
						indexInputVkFFT(output, sc, uintType, readType, sc->inoutID, 0, requestCoordinate, requestBatch);
						sprintf(output + strlen(output), ";\n");
						if (sc->readToRegisters) {
							if (sc->inputBufferBlockNum == 1)
								sprintf(output + strlen(output), "			%s=%s%s[%s]%s;\n", sc->regIDs[i + k * sc->registers_per_thread], convTypeLeft, inputsStruct, sc->inoutID, convTypeRight);
							else
								sprintf(output + strlen(output), "			%s=%sinputBlocks[%s / %d]%s[%s %% %d]%s;\n", sc->regIDs[i + k * sc->registers_per_thread], convTypeLeft, sc->inoutID, sc->inputBufferBlockSize, inputsStruct, sc->inoutID, sc->inputBufferBlockSize, convTypeRight);
						}
						else {
							if (sc->inputBufferBlockNum == 1)
								sprintf(output + strlen(output), "			sdata[%s*(%s+%d)+%s]=%s%s[%s]%s;\n", sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x, convTypeLeft, inputsStruct, sc->inoutID, convTypeRight);
							else
								sprintf(output + strlen(output), "			sdata[%s*(%s+%d)+%s]=%sinputBlocks[%s / %d]%s[%s %% %d]%s;\n", sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x, convTypeLeft, sc->inoutID, sc->inputBufferBlockSize, inputsStruct, sc->inoutID, sc->inputBufferBlockSize, convTypeRight);
						}
						sprintf(output + strlen(output), "		}else{\n");
						if (sc->readToRegisters)
							sprintf(output + strlen(output), "			%s.x = 0; %s.y = 0;\n", sc->regIDs[i + k * sc->registers_per_thread], sc->regIDs[i + k * sc->registers_per_thread]);
						else {
							sprintf(output + strlen(output), "			sdata[%s*(%s+%d)+%s].x=0;\n", sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x);
							sprintf(output + strlen(output), "			sdata[%s*(%s+%d)+%s].y=0;\n", sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x);
						}
						sprintf(output + strlen(output), "		}\n");
					}
				}
			}
			else {
				for (uint32_t k = 0; k < sc->registerBoost; k++) {
					for (uint32_t i = 0; i < sc->min_registers_per_thread; i++) {
						sprintf(output + strlen(output), "			%s = ", sc->inoutID);
						sprintf(index_x, "(%s%s) %% (%d) + %d * (%s + %d) + ((%s%s) / %d) * (%d)", sc->gl_GlobalInvocationID_x, shiftX, sc->stageStartSize, sc->stageStartSize, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_GlobalInvocationID_x, shiftX, sc->stageStartSize, sc->stageStartSize * sc->fftDim);
						indexInputVkFFT(output, sc, uintType, readType, index_x, 0, requestCoordinate, requestBatch);
						sprintf(output + strlen(output), ";\n");
						//sprintf(output + strlen(output), "		inoutID = indexInput((%s%s) %% (%d) + %d * (%s + %d) + ((%s%s) / %d) * (%d));\n", sc->gl_GlobalInvocationID_x, shiftX, sc->stageStartSize, sc->stageStartSize, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_GlobalInvocationID_x, shiftX, sc->stageStartSize, sc->stageStartSize * sc->fftDim);

						if (sc->readToRegisters) {
							if (sc->inputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		%s = %s%s[inoutID]%s;\n", sc->regIDs[i + k * sc->registers_per_thread], convTypeLeft, inputsStruct, convTypeRight);
							else
								sprintf(output + strlen(output), "		%s = %sinputBlocks[inoutID / %d]%s[inoutID %% %d]%s;\n", sc->regIDs[i + k * sc->registers_per_thread], convTypeLeft, sc->inputBufferBlockSize, inputsStruct, sc->inputBufferBlockSize, convTypeRight);
						}
						else {
							if (sc->inputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		sdata[%s*(%s+%d)+%s] = %s%s[inoutID]%s;\n", sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x, convTypeLeft, inputsStruct, convTypeRight);
							else
								sprintf(output + strlen(output), "		sdata[%s*(%s+%d)+%s] = %sinputBlocks[inoutID / %d]%s[inoutID %% %d]%s;\n", sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x, convTypeLeft, sc->inputBufferBlockSize, inputsStruct, sc->inputBufferBlockSize, convTypeRight);
						}
					}
				}
			}
			VkAppendLine(sc, "	}\n");
			break;
		}
		case 5://single_r2c
		{
			if ((sc->localSize[1] > 1) || ((sc->performR2C) && (sc->inverse)) || (sc->localSize[0] * sc->stageRadix[0] * (sc->registers_per_thread_per_radix[sc->stageRadix[0]] / sc->stageRadix[0]) > sc->fftDim))
				sc->readToRegisters = 0;
			else
				sc->readToRegisters = 1;
			char shiftX[500] = "";
			if (sc->performWorkGroupShift[0])
				sprintf(shiftX, " + consts.workGroupShiftX ");
			char shiftY[500] = "";
			if (sc->performWorkGroupShift[1])
				sprintf(shiftY, " + consts.workGroupShiftY ");
			if (sc->zeropad[0]) {
				if (sc->fftDim == sc->fft_dim_full) {
					for (uint32_t k = 0; k < sc->registerBoost; k++) {
						for (uint32_t i = 0; i < sc->min_registers_per_thread; i++) {

							if (sc->localSize[1] == 1)
								sprintf(output + strlen(output), "		combinedID = %s + %d;\n", sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0]);
							else
								sprintf(output + strlen(output), "		combinedID = (%s + %d * %s) + %d;\n", sc->gl_LocalInvocationID_x, sc->localSize[0], sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[0] * sc->localSize[1]);

							if (sc->inputStride[0] > 1)
								sprintf(output + strlen(output), "		inoutID = (combinedID %% %d) * %d + (combinedID / %d) * %d;\n", sc->fftDim, sc->inputStride[0], sc->fftDim, 2 * sc->inputStride[1]);
							else
								sprintf(output + strlen(output), "		inoutID = (combinedID %% %d) + (combinedID / %d) * %d;\n", sc->fftDim, sc->fftDim, 2 * sc->inputStride[1]);
							if ((uint32_t)ceil(sc->size[1] / 2.0) % sc->localSize[1] != 0)
								sprintf(output + strlen(output), "		if(combinedID / %d + (%s%s)*%s< %d){", sc->fftDim, sc->gl_WorkGroupID_y, shiftY, sc->gl_WorkGroupSize_y, (uint32_t)ceil(sc->size[1] / 2.0));

							sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d)){\n", sc->fft_dim_full, sc->fft_zeropad_left_read[sc->axis_id], sc->fft_dim_full, sc->fft_zeropad_right_read[sc->axis_id]);

							sprintf(output + strlen(output), "			%s = ", sc->inoutID);
							indexInputVkFFT(output, sc, uintType, readType, sc->inoutID, 0, requestCoordinate, requestBatch);
							sprintf(output + strlen(output), ";\n");

							if (sc->readToRegisters) {
								if (sc->inputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		%s.x = %s%s[%s]%s;\n", sc->regIDs[i + k * sc->registers_per_thread], convTypeLeft, inputsStruct, sc->inoutID, convTypeRight);
								else
									sprintf(output + strlen(output), "		%s.x = %sinputBlocks[%s / %d]%s[%s %% %d]%s;\n", sc->regIDs[i + k * sc->registers_per_thread], convTypeLeft, sc->inoutID, sc->inputBufferBlockSize, inputsStruct, sc->inoutID, sc->inputBufferBlockSize, convTypeRight);
								if (sc->inputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		%s.y = %s%s[(%s + %d)]%s;\n", sc->regIDs[i + k * sc->registers_per_thread], convTypeLeft, inputsStruct, sc->inoutID, sc->inputStride[1], convTypeRight);
								else
									sprintf(output + strlen(output), "		%s.y = %sinputBlocks[(%s + %d)/ %d]%s[(%s + %d) %% %d]%s;\n", sc->regIDs[i + k * sc->registers_per_thread], convTypeLeft, sc->inoutID, sc->inputStride[1], sc->inputBufferBlockSize, inputsStruct, sc->inoutID, sc->inputStride[1], sc->inputBufferBlockSize, convTypeRight);
							}
							else {
								if (sc->inputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		sdata[(combinedID %% %d) + (combinedID / %d) * sharedStride].x = %s%s[%s]%s;\n", sc->fftDim, sc->fftDim, convTypeLeft, inputsStruct, sc->inoutID, convTypeRight);
								else
									sprintf(output + strlen(output), "		sdata[(combinedID %% %d) + (combinedID / %d) * sharedStride].x = %sinputBlocks[%s / %d]%s[%s %% %d]%s;\n", sc->fftDim, sc->fftDim, convTypeLeft, sc->inoutID, sc->inputBufferBlockSize, inputsStruct, sc->inoutID, sc->inputBufferBlockSize, convTypeRight);
								if (sc->inputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		sdata[(combinedID %% %d) + (combinedID / %d) * sharedStride].y = %s%s[(%s + %d)]%s;\n", sc->fftDim, sc->fftDim, convTypeLeft, inputsStruct, sc->inoutID, sc->inputStride[1], convTypeRight);
								else
									sprintf(output + strlen(output), "		sdata[(combinedID %% %d) + (combinedID / %d) * sharedStride].y = %sinputBlocks[(%s + %d)/ %d]%s[(%s + %d) %% %d]%s;\n", sc->fftDim, sc->fftDim, convTypeLeft, sc->inoutID, sc->inputStride[1], sc->inputBufferBlockSize, inputsStruct, sc->inoutID, sc->inputStride[1], sc->inputBufferBlockSize, convTypeRight);

							}
							sprintf(output + strlen(output), "	}else{\n");
							if (sc->readToRegisters)
								sprintf(output + strlen(output), "		%s.x = 0; %s.y = 0;\n", sc->regIDs[i + k * sc->registers_per_thread], sc->regIDs[i + k * sc->registers_per_thread]);
							else {
								sprintf(output + strlen(output), "		sdata[(combinedID %% %d) + (combinedID / %d) * sharedStride].x = 0;\n", sc->fftDim, sc->fftDim);
								sprintf(output + strlen(output), "		sdata[(combinedID %% %d) + (combinedID / %d) * sharedStride].y = 0;\n", sc->fftDim, sc->fftDim);
							}
							VkAppendLine(sc, "	}\n");
							if ((uint32_t)ceil(sc->size[1] / 2.0) % sc->localSize[1] != 0)
								sprintf(output + strlen(output), "		}");
						}
					}
				}
				else {
					//Not implemented
				}
			}
			else {
				if (sc->fftDim == sc->fft_dim_full) {
					for (uint32_t k = 0; k < sc->registerBoost; k++) {
						for (uint32_t i = 0; i < sc->min_registers_per_thread; i++) {

							if (sc->localSize[1] == 1)
								sprintf(output + strlen(output), "		combinedID = %s + %d;\n", sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0]);
							else
								sprintf(output + strlen(output), "		combinedID = (%s + %d * %s) + %d;\n", sc->gl_LocalInvocationID_x, sc->localSize[0], sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[0] * sc->localSize[1]);

							if (sc->inputStride[0] > 1) {
								sprintf(output + strlen(output), "			%s = ", sc->inoutID);
								sprintf(index_x, "(combinedID %% %d) * %d + (combinedID / %d) * %d", sc->fftDim, sc->inputStride[0], sc->fftDim, 2 * sc->inputStride[1]);
								indexInputVkFFT(output, sc, uintType, readType, index_x, 0, requestCoordinate, requestBatch);
								sprintf(output + strlen(output), ";\n");
								//sprintf(output + strlen(output), "		inoutID = indexInput((combinedID %% %d) * %d + (combinedID / %d) * %d);\n", sc->fftDim, sc->inputStride[0], sc->fftDim, 2 * sc->inputStride[1]);
							}
							else {
								sprintf(output + strlen(output), "			%s = ", sc->inoutID);
								sprintf(index_x, "(combinedID %% %d) + (combinedID / %d) * %d", sc->fftDim, sc->fftDim, 2 * sc->inputStride[1]);
								indexInputVkFFT(output, sc, uintType, readType, index_x, 0, requestCoordinate, requestBatch);
								sprintf(output + strlen(output), ";\n");
								//sprintf(output + strlen(output), "		inoutID = indexInput((combinedID %% %d) + (combinedID / %d) * %d);\n", sc->fftDim, sc->fftDim, 2 * sc->inputStride[1]);
							}
							if ((uint32_t)ceil(sc->size[1] / 2.0) % sc->localSize[1] != 0)
								sprintf(output + strlen(output), "		if(combinedID / %d + (%s%s)*%s< %d){", sc->fftDim, sc->gl_WorkGroupID_y, shiftY, sc->gl_WorkGroupSize_y, (uint32_t)ceil(sc->size[sc->axis_id + 1] / 2.0));

							if (sc->readToRegisters) {
								if (sc->inputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		%s.x = %s%s[inoutID]%s;\n", sc->regIDs[i + k * sc->registers_per_thread], convTypeLeft, inputsStruct, convTypeRight);
								else
									sprintf(output + strlen(output), "		%s.x = %sinputBlocks[inoutID / %d]%s[inoutID %% %d]%s;\n", sc->regIDs[i + k * sc->registers_per_thread], convTypeLeft, sc->inputBufferBlockSize, inputsStruct, sc->inputBufferBlockSize, convTypeRight);
								sprintf(output + strlen(output), "		inoutID += %d;\n", sc->inputStride[1]);
								if (sc->inputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		%s.y = %s%s[inoutID]%s;\n", sc->regIDs[i + k * sc->registers_per_thread], convTypeLeft, inputsStruct, convTypeRight);
								else
									sprintf(output + strlen(output), "		%s.y = %sinputBlocks[inoutID / %d]%s[inoutID %% %d]%s;\n", sc->regIDs[i + k * sc->registers_per_thread], convTypeLeft, sc->inputBufferBlockSize, inputsStruct, sc->inputBufferBlockSize, convTypeRight);
							}
							else {
								if (sc->inputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		sdata[(combinedID %% %d) + (combinedID / %d) * sharedStride].x = %s%s[inoutID]%s;\n", sc->fftDim, sc->fftDim, convTypeLeft, inputsStruct, convTypeRight);
								else
									sprintf(output + strlen(output), "		sdata[(combinedID %% %d) + (combinedID / %d) * sharedStride].x = %sinputBlocks[inoutID / %d]%s[inoutID %% %d]%s;\n", sc->fftDim, sc->fftDim, convTypeLeft, sc->inputBufferBlockSize, inputsStruct, sc->inputBufferBlockSize, convTypeRight);
								sprintf(output + strlen(output), "		inoutID += %d;\n", sc->inputStride[1]);
								if (sc->inputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		sdata[(combinedID %% %d) + (combinedID / %d) * sharedStride].y = %s%s[inoutID]%s;\n", sc->fftDim, sc->fftDim, convTypeLeft, inputsStruct, convTypeRight);
								else
									sprintf(output + strlen(output), "		sdata[(combinedID %% %d) + (combinedID / %d) * sharedStride].y = %sinputBlocks[inoutID / %d]%s[inoutID %% %d]%s;\n", sc->fftDim, sc->fftDim, convTypeLeft, sc->inputBufferBlockSize, inputsStruct, sc->inputBufferBlockSize, convTypeRight);

							}
							if ((uint32_t)ceil(sc->size[1] / 2.0) % sc->localSize[1] != 0)
								sprintf(output + strlen(output), "		}");
						}
					}
				}
				else {
					//Not implemented
				}
			}
			break;
		}
		case 6: {//single_c2r
			if ((sc->localSize[1] > 1) || ((sc->performR2C) && (sc->inverse)) || (sc->localSize[0] * sc->stageRadix[0] * (sc->registers_per_thread_per_radix[sc->stageRadix[0]] / sc->stageRadix[0]) > sc->fftDim))
				sc->readToRegisters = 0;
			else
				sc->readToRegisters = 1;
			char shiftY[500] = "";
			if (sc->performWorkGroupShift[1])
				sprintf(shiftY, " + consts.workGroupShiftY * %d", sc->localSize[1]);
			if (sc->zeropad[0]) {
				if (sc->fftDim == sc->fft_dim_full) {
					for (uint32_t i = 0; i < ceil(sc->min_registers_per_thread / 2.0); i++) {
						if ((uint32_t)ceil(sc->size[1] / 2.0) % sc->localSize[1] != 0)
							sprintf(output + strlen(output), "		if(%s%s < %d){", sc->gl_GlobalInvocationID_y, shiftY, (uint32_t)ceil(sc->size[1] / 2.0));
						if ((ceil(sc->min_registers_per_thread / 2.0) != sc->min_registers_per_thread / 2) && (i == (ceil(sc->min_registers_per_thread / 2.0) - 1)))
							sprintf(output + strlen(output), "if (%s < %d){\n", sc->gl_LocalInvocationID_x, sc->fftDim / 2 - i * sc->localSize[0]);

						sprintf(output + strlen(output), "		inoutID = %s+%d;\n", sc->gl_LocalInvocationID_x, i * sc->localSize[0] + 1);

						sprintf(output + strlen(output), "		if((inoutID < %d)||(inoutID >= %d)){\n", sc->fft_zeropad_left_read[sc->axis_id], sc->fft_zeropad_right_read[sc->axis_id]);

						sprintf(output + strlen(output), "			%s = ", sc->inoutID);
						sprintf(index_y, "2*%s%s", sc->gl_GlobalInvocationID_y, shiftY);
						indexInputVkFFT(output, sc, uintType, readType, sc->inoutID, index_y, requestCoordinate, requestBatch);
						sprintf(output + strlen(output), ";\n");

						if (sc->inputBufferBlockNum == 1)
							sprintf(output + strlen(output), "		temp_0 = %s%s[%s]%s;\n", convTypeLeft, inputsStruct, sc->inoutID, convTypeRight);
						else
							sprintf(output + strlen(output), "		temp_0 = %sinputBlocks[%s / %d]%s[%s %% %d]%s;\n", convTypeLeft, sc->inoutID, sc->inputBufferBlockSize, inputsStruct, sc->inoutID, sc->inputBufferBlockSize, convTypeRight);

						sprintf(output + strlen(output), "			%s = ", sc->inoutID);
						sprintf(index_x, "%s+%d", sc->gl_LocalInvocationID_x, sc->inputStride[1] + i * sc->localSize[0] + 1);
						sprintf(index_y, "2*%s%s", sc->gl_GlobalInvocationID_y, shiftY);
						indexInputVkFFT(output, sc, uintType, readType, index_x, index_y, requestCoordinate, requestBatch);
						sprintf(output + strlen(output), ";\n");
						//sprintf(output + strlen(output), "		inoutID = indexInput(%s+%d, (%s%s));\n", sc->gl_LocalInvocationID_x, sc->inputStride[1] / 2 + i * sc->localSize[0], sc->gl_GlobalInvocationID_y, shiftY);

						if (sc->inputBufferBlockNum == 1)
							sprintf(output + strlen(output), "		temp_1 = %s%s[inoutID]%s;\n", convTypeLeft, inputsStruct, convTypeRight);
						else
							sprintf(output + strlen(output), "		temp_1 = %sinputBlocks[inoutID / %d]%s[inoutID %% %d]%s;\n", convTypeLeft, sc->inputBufferBlockSize, inputsStruct, sc->inputBufferBlockSize, convTypeRight);

						sprintf(output + strlen(output), "		}else{\n");
						sprintf(output + strlen(output), "			temp_0.x = 0; temp_0.y = 0;");
						sprintf(output + strlen(output), "			temp_1.x = 0; temp_1.y = 0;");
						sprintf(output + strlen(output), "		}\n");
						if (sc->localSize[1] == 1)
							sprintf(output + strlen(output), "\
		sdata[%s+%d].x=(temp_0.x-temp_1.y);\n\
		sdata[%s + %d].y = (temp_0.y + temp_1.x);\n\
		sdata[%d - %s].x = (temp_0.x + temp_1.y);\n\
		sdata[%d - %s].y = (-temp_0.y + temp_1.x);\n", sc->gl_LocalInvocationID_x, i * sc->localSize[0] + 1, sc->gl_LocalInvocationID_x, i * sc->localSize[0] + 1, sc->fftDim - i * sc->localSize[0] - 1, sc->gl_LocalInvocationID_x, sc->fftDim - i * sc->localSize[0] - 1, sc->gl_LocalInvocationID_x);
						else
							sprintf(output + strlen(output), "\
		sdata[sharedStride * %s + %s + %d].x=(temp_0.x-temp_1.y);\n\
		sdata[sharedStride * %s + %s + %d].y = (temp_0.y + temp_1.x);\n\
		sdata[sharedStride * %s + %d - %s].x = (temp_0.x + temp_1.y);\n\
		sdata[sharedStride * %s + %d - %s].y = (-temp_0.y + temp_1.x);\n", sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x, i * sc->localSize[0] + 1, sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x, i * sc->localSize[0] + 1, sc->gl_LocalInvocationID_y, sc->fftDim - i * sc->localSize[0] - 1, sc->gl_LocalInvocationID_x, sc->gl_LocalInvocationID_y, sc->fftDim - i * sc->localSize[0] - 1, sc->gl_LocalInvocationID_x);
						if ((ceil(sc->min_registers_per_thread / 2.0) != sc->min_registers_per_thread / 2) && (i == (ceil(sc->min_registers_per_thread / 2.0) - 1)))
							sprintf(output + strlen(output), "}\n");
						if ((uint32_t)ceil(sc->size[1] / 2.0) % sc->localSize[1] != 0)
							sprintf(output + strlen(output), "		}");
					}
					sprintf(output + strlen(output), "\
	if (%s==0) \n\
	{\n", sc->gl_LocalInvocationID_x);
					sprintf(output + strlen(output), "			%s = ", sc->inoutID);
					sprintf(index_x, "0");
					sprintf(index_y, "2*%s%s", sc->gl_GlobalInvocationID_y, shiftY);
					indexInputVkFFT(output, sc, uintType, readType, index_x, index_y, requestCoordinate, requestBatch);
					sprintf(output + strlen(output), ";\n");
					//sprintf(output + strlen(output), "		inoutID = indexInput(2 * (%s%s), %d);\n", sc->gl_GlobalInvocationID_y, shiftY, sc->inputStride[2] / (sc->inputStride[1] + 2));
					if (sc->inputBufferBlockNum == 1)
						sprintf(output + strlen(output), "		temp_0 = %s%s[inoutID]%s;\n", convTypeLeft, inputsStruct, convTypeRight);
					else
						sprintf(output + strlen(output), "		temp_0 = %sinputBlocks[inoutID / %d]%s[inoutID %% %d]%s;\n", convTypeLeft, sc->inputBufferBlockSize, inputsStruct, sc->inputBufferBlockSize, convTypeRight);

					sprintf(output + strlen(output), "			%s = ", sc->inoutID);
					sprintf(index_x, "%s", sc->inputStride[1]);
					sprintf(index_y, "2*%s%s", sc->gl_GlobalInvocationID_y, shiftY);
					indexInputVkFFT(output, sc, uintType, readType, index_x, index_y, requestCoordinate, requestBatch);
					sprintf(output + strlen(output), ";\n");
					//sprintf(output + strlen(output), "		inoutID = indexInput(2 * (%s%s) + 1, %d);\n", sc->gl_GlobalInvocationID_y, shiftY, sc->inputStride[2] / (sc->inputStride[1] + 2));
					if (sc->inputBufferBlockNum == 1)
						sprintf(output + strlen(output), "		temp_1 = %s%s[inoutID]%s;\n", convTypeLeft, inputsStruct, convTypeRight);
					else
						sprintf(output + strlen(output), "		temp_1 = %sinputBlocks[inoutID / %d]%s[inoutID %% %d]%s;\n", convTypeLeft, sc->inputBufferBlockSize, inputsStruct, sc->inputBufferBlockSize, convTypeRight);
					if (sc->localSize[1] == 1)
						sprintf(output + strlen(output), "\
		sdata[0].x = (temp_0.x - temp_1.y);\n\
		sdata[0].y = (temp_0.y + temp_1.x);\n");
					else
						sprintf(output + strlen(output), "\
		sdata[sharedStride * %s].x = (temp_0.x - temp_1.y);\n\
		sdata[sharedStride * %s].y = (temp_0.y + temp_1.x);\n", sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_y);
					VkAppendLine(sc, "	}\n");
				}
			}
			else {
				if (sc->fftDim == sc->fft_dim_full) {
					for (uint32_t i = 0; i < ceil(sc->min_registers_per_thread / 2.0); i++) {
						if ((uint32_t)ceil(sc->size[1] / 2.0) % sc->localSize[1] != 0)
							sprintf(output + strlen(output), "		if(%s%s < %d){", sc->gl_GlobalInvocationID_y, shiftY, (uint32_t)ceil(sc->size[1] / 2.0));
						if ((ceil(sc->min_registers_per_thread / 2.0) != sc->min_registers_per_thread / 2) && (i == (ceil(sc->min_registers_per_thread / 2.0) - 1)))
							sprintf(output + strlen(output), "if (%s < %d){\n", sc->gl_LocalInvocationID_x, sc->fftDim / 2 - i * sc->localSize[0]);

						sprintf(output + strlen(output), "			%s = ", sc->inoutID);
						sprintf(index_x, "%s + %d", sc->gl_LocalInvocationID_x, i * sc->localSize[0] + 1);
						sprintf(index_y, "2*%s%s", sc->gl_GlobalInvocationID_y, shiftY);
						indexInputVkFFT(output, sc, uintType, readType, index_x, index_y, requestCoordinate, requestBatch);
						sprintf(output + strlen(output), ";\n");
						//sprintf(output + strlen(output), "		inoutID = indexInput(%s + %d, (%s%s));\n", sc->gl_LocalInvocationID_x, i * sc->localSize[0], sc->gl_GlobalInvocationID_y, shiftY);

						if (sc->inputBufferBlockNum == 1)
							sprintf(output + strlen(output), "		temp_0 = %s%s[inoutID]%s;\n", convTypeLeft, inputsStruct, convTypeRight);
						else
							sprintf(output + strlen(output), "		temp_0 = %sinputBlocks[inoutID / %d]%s[inoutID %% %d]%s;\n", convTypeLeft, sc->inputBufferBlockSize, inputsStruct, sc->inputBufferBlockSize, convTypeRight);

						sprintf(output + strlen(output), "			%s = ", sc->inoutID);
						sprintf(index_x, "%s + %d", sc->gl_LocalInvocationID_x, sc->inputStride[1] + i * sc->localSize[0] + 1);
						sprintf(index_y, "2*%s%s", sc->gl_GlobalInvocationID_y, shiftY);
						indexInputVkFFT(output, sc, uintType, readType, index_x, index_y, requestCoordinate, requestBatch);
						sprintf(output + strlen(output), ";\n");
						//sprintf(output + strlen(output), "		inoutID = indexInput(%s+%d, (%s%s));\n", sc->gl_LocalInvocationID_x, sc->inputStride[1] / 2 + i * sc->localSize[0], sc->gl_GlobalInvocationID_y, shiftY);

						if (sc->inputBufferBlockNum == 1)
							sprintf(output + strlen(output), "		temp_1 = %s%s[inoutID]%s;\n", convTypeLeft, inputsStruct, convTypeRight);
						else
							sprintf(output + strlen(output), "		temp_1 = %sinputBlocks[inoutID / %d]%s[inoutID %% %d]%s;\n", convTypeLeft, sc->inputBufferBlockSize, inputsStruct, sc->inputBufferBlockSize, convTypeRight);
						if (sc->localSize[1] == 1)
							sprintf(output + strlen(output), "\
		sdata[%s+%d].x=(temp_0.x-temp_1.y);\n\
		sdata[%s + %d].y = (temp_0.y + temp_1.x);\n\
		sdata[%d - %s].x = (temp_0.x + temp_1.y);\n\
		sdata[%d - %s].y = (-temp_0.y + temp_1.x);\n", sc->gl_LocalInvocationID_x, i * sc->localSize[0] + 1, sc->gl_LocalInvocationID_x, i * sc->localSize[0] + 1, sc->fftDim - i * sc->localSize[0] - 1, sc->gl_LocalInvocationID_x, sc->fftDim - i * sc->localSize[0] - 1, sc->gl_LocalInvocationID_x);
						else
							sprintf(output + strlen(output), "\
		sdata[sharedStride*%s + %s+%d].x=(temp_0.x-temp_1.y);\n\
		sdata[sharedStride * %s + %s + %d].y = (temp_0.y + temp_1.x);\n\
		sdata[sharedStride * %s + %d - %s].x = (temp_0.x + temp_1.y);\n\
		sdata[sharedStride * %s + %d - %s].y = (-temp_0.y + temp_1.x);\n", sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x, i * sc->localSize[0] + 1, sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x, i * sc->localSize[0] + 1, sc->gl_LocalInvocationID_y, sc->fftDim - i * sc->localSize[0] - 1, sc->gl_LocalInvocationID_x, sc->gl_LocalInvocationID_y, sc->fftDim - i * sc->localSize[0] - 1, sc->gl_LocalInvocationID_x);
						if ((ceil(sc->min_registers_per_thread / 2.0) != sc->min_registers_per_thread / 2) && (i == (ceil(sc->min_registers_per_thread / 2.0) - 1)))
							sprintf(output + strlen(output), "}\n");
						if ((uint32_t)ceil(sc->size[1] / 2.0) % sc->localSize[1] != 0)
							sprintf(output + strlen(output), "		}");
					}
					sprintf(output + strlen(output), "\
	if (%s==0) \n\
	{\n", sc->gl_LocalInvocationID_x);
					sprintf(output + strlen(output), "			%s = ", sc->inoutID);
					sprintf(index_x, "0");
					sprintf(index_y, "2*%s%s", sc->gl_GlobalInvocationID_y, shiftY);
					indexInputVkFFT(output, sc, uintType, readType, index_x, index_y, requestCoordinate, requestBatch);
					sprintf(output + strlen(output), ";\n");
					//sprintf(output + strlen(output), "		inoutID = indexInput(2 * (%s%s), %d);\n", sc->gl_GlobalInvocationID_y, shiftY, sc->inputStride[2] / (sc->inputStride[1] + 2));
					if (sc->inputBufferBlockNum == 1)
						sprintf(output + strlen(output), "		temp_0 = %s%s[inoutID]%s;\n", convTypeLeft, inputsStruct, convTypeRight);
					else
						sprintf(output + strlen(output), "		temp_0 = %sinputBlocks[inoutID / %d]%s[inoutID %% %d]%s;\n", convTypeLeft, sc->inputBufferBlockSize, inputsStruct, sc->inputBufferBlockSize, convTypeRight);

					sprintf(output + strlen(output), "			%s = ", sc->inoutID);
					sprintf(index_x, "%d", sc->inputStride[1]);
					sprintf(index_y, "2*%s%s", sc->gl_GlobalInvocationID_y, shiftY);
					indexInputVkFFT(output, sc, uintType, readType, index_x, index_y, requestCoordinate, requestBatch);
					sprintf(output + strlen(output), ";\n");
					//sprintf(output + strlen(output), "		inoutID = indexInput(2 * (%s%s) + 1, %d);\n", sc->gl_GlobalInvocationID_y, shiftY, sc->inputStride[2] / (sc->inputStride[1] + 2));
					if (sc->inputBufferBlockNum == 1)
						sprintf(output + strlen(output), "		temp_1 = %s%s[inoutID]%s;\n", convTypeLeft, inputsStruct, convTypeRight);
					else
						sprintf(output + strlen(output), "		temp_1 = %sinputBlocks[inoutID / %d]%s[inoutID %% %d]%s;\n", convTypeLeft, sc->inputBufferBlockSize, inputsStruct, sc->inputBufferBlockSize, convTypeRight);
					if (sc->localSize[1] == 1)
						sprintf(output + strlen(output), "\
		sdata[0].x = (temp_0.x - temp_1.y);\n\
		sdata[0].y = (temp_0.y + temp_1.x);\n");
					else
						sprintf(output + strlen(output), "\
		sdata[sharedStride * %s].x = (temp_0.x - temp_1.y);\n\
		sdata[sharedStride * %s].y = (temp_0.y + temp_1.x);\n", sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_y);
					VkAppendLine(sc, "	}\n");
					
				}
				else {
					//Not implemented
				}
			}
			break;
		}
		}
		appendZeropadEnd(output, sc);
	}

	static inline void appendReorder4StepRead(char* output, VkFFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint32_t reorderType) {
		char vecType[30];
		char LFending[4] = "";
		if (!strcmp(floatType, "float")) sprintf(LFending, "f");
#if(VKFFT_BACKEND==0)
		if (!strcmp(floatType, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "dvec2");
		char cosDef[20] = "cos";
		char sinDef[20] = "sin";
		if (!strcmp(floatType, "double")) sprintf(LFending, "LF");
#elif(VKFFT_BACKEND==1)
		if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
		char cosDef[20] = "__cosf";
		char sinDef[20] = "__sinf";
		if (!strcmp(floatType, "double")) sprintf(LFending, "l");
#elif(VKFFT_BACKEND==2)
		if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
		char cosDef[20] = "__cosf";
		char sinDef[20] = "__sinf";
		if (!strcmp(floatType, "double")) sprintf(LFending, "l");
#endif

		uint32_t logicalRegistersPerThread = sc->registers_per_thread_per_radix[sc->stageRadix[0]];// (sc->registers_per_thread % sc->stageRadix[sc->numStages - 1] == 0) ? sc->registers_per_thread : sc->min_registers_per_thread;
		switch (reorderType) {
		case 1: {//grouped_c2c
			char shiftX[500] = "";
			if (sc->performWorkGroupShift[0])
				sprintf(shiftX, " + consts.workGroupShiftX * %s ", sc->gl_WorkGroupSize_x);
			if ((sc->stageStartSize > 1) && (!sc->reorderFourStep) && (sc->inverse)) {
				if (sc->localSize[1] * sc->stageRadix[0] * (sc->registers_per_thread_per_radix[sc->stageRadix[0]] / sc->stageRadix[0]) > sc->fftDim) {
					appendBarrierVkFFT(output, 1);
					sc->readToRegisters = 0;
				}
				else
					sc->readToRegisters = 1;
				appendZeropadStart(output, sc);
				VkAppendLine(sc, sc->disableThreadsStart);
				for (uint32_t i = 0; i < sc->fftDim / sc->localSize[1]; i++) {
					uint32_t id = (i / logicalRegistersPerThread) * sc->registers_per_thread + i % logicalRegistersPerThread;
					if (sc->LUT) {
						sprintf(output + strlen(output), "		mult = twiddleLUT[%d+(((%s%s)/%d) %% (%d))+%d*(%s+%d)];\n", sc->maxStageSumLUT, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->stageStartSize, sc->stageStartSize, sc->gl_LocalInvocationID_y, i * sc->localSize[1]);
						if (!sc->inverse)
							sprintf(output + strlen(output), "	mult.y = -mult.y;\n");
					}
					else {
						sprintf(output + strlen(output), "		angle = 2 * loc_PI * (((((%s%s) / %d) %% (%d)) * (%s + %d)) / %f%s;\n", sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->stageStartSize, sc->gl_LocalInvocationID_y, i * sc->localSize[1], (double)(sc->stageStartSize * sc->fftDim), LFending);
						if (!strcmp(floatType, "float")) {
							sprintf(output + strlen(output), "		mult.x = %s(angle);\n", cosDef);
							sprintf(output + strlen(output), "		mult.y = %s(angle);\n", sinDef);
							//sprintf(output + strlen(output), "		mult = %s(cos(angle), sin(angle));\n", vecType);
						}
						if (!strcmp(floatType, "double"))
							sprintf(output + strlen(output), "		mult = sincos_20(angle);\n");
					}
					if (sc->readToRegisters) {
						sprintf(output + strlen(output), "\
		w.x = %s.x * mult.x - %s.y * mult.y;\n", sc->regIDs[id], sc->regIDs[id]);
						sprintf(output + strlen(output), "\
		%s.y = %s.y * mult.x + %s.x * mult.y;\n", sc->regIDs[id], sc->regIDs[id], sc->regIDs[id]);
						sprintf(output + strlen(output), "\
		%s.x = w.x;\n", sc->regIDs[id]);
					}
					else {
						sprintf(output + strlen(output), "\
		%s = %s*(%d+%s) + %s;\n", sc->inoutID, sc->gl_WorkGroupSize_x, i * sc->localSize[1], sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x);

						sprintf(output + strlen(output), "\
		w.x = sdata[%s].x * mult.x - sdata[%s].y * mult.y;\n", sc->inoutID, sc->inoutID);

						sprintf(output + strlen(output), "\
		sdata[%s].y = sdata[%s].y * mult.x + sdata[%s].x * mult.y;\n", sc->inoutID, sc->inoutID, sc->inoutID);
						sprintf(output + strlen(output), "\
		sdata[%s].x = w.x;\n", sc->inoutID);
					}
				}
				VkAppendLine(sc, sc->disableThreadsEnd);
				appendZeropadEnd(output, sc);
			}

			break;
		}
		case 2: {//single_c2c_strided
			char shiftX[500] = "";
			if (sc->performWorkGroupShift[0])
				sprintf(shiftX, " + consts.workGroupShiftX * %s ", sc->gl_WorkGroupSize_x);
			if ((!sc->reorderFourStep) && (sc->inverse)) {
				if (sc->localSize[1] * sc->stageRadix[0] * (sc->registers_per_thread_per_radix[sc->stageRadix[0]] / sc->stageRadix[0]) > sc->fftDim) {
					appendBarrierVkFFT(output, 1);
					sc->readToRegisters = 0;
				}
				else
					sc->readToRegisters = 1;
				appendZeropadStart(output, sc);
				VkAppendLine(sc, sc->disableThreadsStart);
				for (uint32_t i = 0; i < sc->fftDim / sc->localSize[1]; i++) {
					uint32_t id = (i / logicalRegistersPerThread) * sc->registers_per_thread + i % logicalRegistersPerThread;
					if (sc->LUT) {
						sprintf(output + strlen(output), "		mult = twiddleLUT[%d + ((%s%s) %% (%d)) + (%s + %d) * %d];\n", sc->maxStageSumLUT, sc->gl_GlobalInvocationID_x, shiftX, sc->stageStartSize, sc->gl_LocalInvocationID_y, i * sc->localSize[1], sc->stageStartSize);
						if (!sc->inverse)
							sprintf(output + strlen(output), "	mult.y = -mult.y;\n");
					}
					else {
						sprintf(output + strlen(output), "		angle = 2 * loc_PI * ((((%s%s) %% (%d)) * (%s + %d)) / %f%s);\n", sc->gl_GlobalInvocationID_x, shiftX, sc->stageStartSize, sc->gl_LocalInvocationID_y, i * sc->localSize[1], (double)(sc->stageStartSize * sc->fftDim), LFending);

						if (!strcmp(floatType, "float")) {
							sprintf(output + strlen(output), "		mult.x = %s(angle);\n", cosDef);
							sprintf(output + strlen(output), "		mult.y = %s(angle);\n", sinDef);
							//sprintf(output + strlen(output), "		mult = %s(cos(angle), sin(angle));\n", vecType);
						}
						if (!strcmp(floatType, "double"))
							sprintf(output + strlen(output), "		mult = sincos_20(angle);\n");
					}
					if (sc->readToRegisters) {
						sprintf(output + strlen(output), "\
		w.x = %s.x * mult.x - %s.y * mult.y;\n", sc->regIDs[id], sc->regIDs[id]);
						sprintf(output + strlen(output), "\
		%s.y = %s.y * mult.x + %s.x * mult.y;\n", sc->regIDs[id], sc->regIDs[id], sc->regIDs[id]);
						sprintf(output + strlen(output), "\
		%s.x = w.x;\n", sc->regIDs[id]);
					}
					else {
						sprintf(output + strlen(output), "\
		%s = %s*(%d+%s) + %s;\n", sc->inoutID, sc->gl_WorkGroupSize_x, i * sc->localSize[1], sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x);

						sprintf(output + strlen(output), "\
		w.x = sdata[%s].x * mult.x - sdata[%s].y * mult.y;\n", sc->inoutID, sc->inoutID);

						sprintf(output + strlen(output), "\
		sdata[%s].y = sdata[%s].y * mult.x + sdata[%s].x * mult.y;\n", sc->inoutID, sc->inoutID, sc->inoutID);
						sprintf(output + strlen(output), "\
		sdata[%s].x = w.x;\n", sc->inoutID);
					}
				}
				VkAppendLine(sc, sc->disableThreadsEnd);
				appendZeropadEnd(output, sc);
			}
			//appendBarrierVkFFT(output, 1);
			break;
		}
		}

	}
	static inline void appendReorder4StepWrite(char* output, VkFFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint32_t reorderType) {
		char vecType[30];
		char LFending[4] = "";
		if (!strcmp(floatType, "float")) sprintf(LFending, "f");
#if(VKFFT_BACKEND==0)
		if (!strcmp(floatType, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "dvec2");
		char cosDef[20] = "cos";
		char sinDef[20] = "sin";
		if (!strcmp(floatType, "double")) sprintf(LFending, "LF");
#elif(VKFFT_BACKEND==1)
		if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
		char cosDef[20] = "__cosf";
		char sinDef[20] = "__sinf";
		if (!strcmp(floatType, "double")) sprintf(LFending, "l");
#elif(VKFFT_BACKEND==2)
		if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
		char cosDef[20] = "__cosf";
		char sinDef[20] = "__sinf";
		if (!strcmp(floatType, "double")) sprintf(LFending, "l");
#endif

		uint32_t logicalRegistersPerThread = sc->registers_per_thread_per_radix[sc->stageRadix[sc->numStages - 1]];// (sc->registers_per_thread % sc->stageRadix[sc->numStages - 1] == 0) ? sc->registers_per_thread : sc->min_registers_per_thread;
		switch (reorderType) {
		case 1: {//grouped_c2c
			char shiftX[500] = "";
			if (sc->performWorkGroupShift[0])
				sprintf(shiftX, " + consts.workGroupShiftX * %s ", sc->gl_WorkGroupSize_x);
			if ((sc->stageStartSize > 1) && (!((sc->stageStartSize > 1) && (!sc->reorderFourStep) && (sc->inverse)))) {
				if (sc->localSize[1] * sc->stageRadix[sc->numStages - 1] * (sc->registers_per_thread / sc->stageRadix[sc->numStages - 1]) > sc->fftDim) {
					appendBarrierVkFFT(output, 1);
					sc->writeFromRegisters = 0;
				}
				else
					sc->writeFromRegisters = 1;
				appendZeropadStart(output, sc);
				VkAppendLine(sc, sc->disableThreadsStart);
				for (uint32_t i = 0; i < sc->fftDim / sc->localSize[1]; i++) {
					uint32_t id = (i / logicalRegistersPerThread) * sc->registers_per_thread + i % logicalRegistersPerThread;
					if (sc->LUT) {
						sprintf(output + strlen(output), "		mult = twiddleLUT[%d+(((%s%s)/%d) %% (%d))+%d*(%s+%d)];\n", sc->maxStageSumLUT, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->stageStartSize, sc->stageStartSize, sc->gl_LocalInvocationID_y, i * sc->localSize[1]);
						if (!sc->inverse)
							sprintf(output + strlen(output), "	mult.y = -mult.y;\n");
					}
					else {
						sprintf(output + strlen(output), "		angle = 2 * loc_PI * ((((%s%s) / %d) %% (%d)) * (%s + %d)) / %f%s;\n", sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->stageStartSize, sc->gl_LocalInvocationID_y, i * sc->localSize[1], (double)(sc->stageStartSize * sc->fftDim), LFending);
						if (sc->inverse) {
							if (!strcmp(floatType, "float")) {
								sprintf(output + strlen(output), "		mult.x = %s(angle);\n", cosDef);
								sprintf(output + strlen(output), "		mult.y = %s(angle);\n", sinDef);
								//sprintf(output + strlen(output), "		mult = %s(cos(angle), sin(angle));\n", vecType);
							}
							if (!strcmp(floatType, "double"))
								sprintf(output + strlen(output), "		mult = sincos_20(angle);\n");
						}
						else {
							if (!strcmp(floatType, "float")) {
								sprintf(output + strlen(output), "		mult.x = %s(angle);\n", cosDef);
								sprintf(output + strlen(output), "		mult.y = -%s(angle);\n", sinDef);
								//sprintf(output + strlen(output), "		mult = %s(cos(angle), sin(angle));\n", vecType);
							}
							if (!strcmp(floatType, "double"))
								sprintf(output + strlen(output), "		mult = sincos_20(-angle);\n");
						}
					}
					if (sc->writeFromRegisters) {
						sprintf(output + strlen(output), "\
		w.x = %s.x * mult.x - %s.y * mult.y;\n", sc->regIDs[id], sc->regIDs[id]);
						sprintf(output + strlen(output), "\
		%s.y = %s.y * mult.x + %s.x * mult.y;\n", sc->regIDs[id], sc->regIDs[id], sc->regIDs[id]);
						sprintf(output + strlen(output), "\
		%s.x = w.x;\n", sc->regIDs[id]);
					}
					else {
						sprintf(output + strlen(output), "\
		%s = %s*(%d+%s) + %s;\n", sc->inoutID, sc->gl_WorkGroupSize_x, i * sc->localSize[1], sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x);

						sprintf(output + strlen(output), "\
		w.x = sdata[%s].x * mult.x - sdata[%s].y * mult.y;\n", sc->inoutID, sc->inoutID);

						sprintf(output + strlen(output), "\
		sdata[%s].y = sdata[%s].y * mult.x + sdata[%s].x * mult.y;\n", sc->inoutID, sc->inoutID, sc->inoutID);
						sprintf(output + strlen(output), "\
		sdata[%s].x = w.x;\n", sc->inoutID);
					}
				}
				VkAppendLine(sc, sc->disableThreadsEnd);
				appendZeropadEnd(output, sc);
			}
			break;
		}
		case 2: {//single_c2c_strided
			char shiftX[500] = "";
			if (sc->performWorkGroupShift[0])
				sprintf(shiftX, " + consts.workGroupShiftX * %s ", sc->gl_WorkGroupSize_x);
			if (!((!sc->reorderFourStep) && (sc->inverse))) {
				if (sc->localSize[1] * sc->stageRadix[sc->numStages - 1] * (sc->registers_per_thread_per_radix[sc->stageRadix[sc->numStages - 1]] / sc->stageRadix[sc->numStages - 1]) > sc->fftDim) {
					appendBarrierVkFFT(output, 1);
					sc->writeFromRegisters = 0;
				}
				else
					sc->writeFromRegisters = 1;
				appendZeropadStart(output, sc);
				VkAppendLine(sc, sc->disableThreadsStart);
				for (uint32_t i = 0; i < sc->fftDim / sc->localSize[1]; i++) {
					uint32_t id = (i / logicalRegistersPerThread) * sc->registers_per_thread + i % logicalRegistersPerThread;
					if (sc->LUT) {
						sprintf(output + strlen(output), "		mult = twiddleLUT[%d + ((%s%s) %% (%d)) + (%s + %d) * %d];\n", sc->maxStageSumLUT, sc->gl_GlobalInvocationID_x, shiftX, sc->stageStartSize, sc->gl_LocalInvocationID_y, i * sc->localSize[1], sc->stageStartSize);
						if (!sc->inverse)
							sprintf(output + strlen(output), "	mult.y = -mult.y;\n");
					}
					else {
						sprintf(output + strlen(output), "		angle = 2 * loc_PI * ((((%s%s) %% (%d)) * (%s + %d)) / %f%s);\n", sc->gl_GlobalInvocationID_x, shiftX, sc->stageStartSize, sc->gl_LocalInvocationID_y, i * sc->localSize[1], (double)(sc->stageStartSize * sc->fftDim), LFending);
						if (sc->inverse) {
							if (!strcmp(floatType, "float")) {
								sprintf(output + strlen(output), "		mult.x = %s(angle);\n", cosDef);
								sprintf(output + strlen(output), "		mult.y = %s(angle);\n", sinDef);
								//sprintf(output + strlen(output), "		mult = %s(cos(angle), sin(angle));\n", vecType);
							}
							if (!strcmp(floatType, "double"))
								sprintf(output + strlen(output), "		mult = sincos_20(angle);\n");
						}
						else {
							if (!strcmp(floatType, "float")) {
								sprintf(output + strlen(output), "		mult.x = %s(angle);\n", cosDef);
								sprintf(output + strlen(output), "		mult.y = -%s(angle);\n", sinDef);
								//sprintf(output + strlen(output), "		mult = %s(cos(angle), sin(angle));\n", vecType);
							}
							if (!strcmp(floatType, "double"))
								sprintf(output + strlen(output), "		mult = sincos_20(-angle);\n");
						}
					}
					if (sc->writeFromRegisters) {
						sprintf(output + strlen(output), "\
		w.x = %s.x * mult.x - %s.y * mult.y;\n", sc->regIDs[id], sc->regIDs[id]);
						sprintf(output + strlen(output), "\
		%s.y = %s.y * mult.x + %s.x * mult.y;\n", sc->regIDs[id], sc->regIDs[id], sc->regIDs[id]);
						sprintf(output + strlen(output), "\
		%s.x = w.x;\n", sc->regIDs[id]);
					}
					else {
						sprintf(output + strlen(output), "\
		%s = %s*(%d+%s) + %s;\n", sc->inoutID, sc->gl_WorkGroupSize_x, i * sc->localSize[1], sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x);

						sprintf(output + strlen(output), "\
		w.x = sdata[%s].x * mult.x - sdata[%s].y * mult.y;\n", sc->inoutID, sc->inoutID);

						sprintf(output + strlen(output), "\
		sdata[%s].y = sdata[%s].y * mult.x + sdata[%s].x * mult.y;\n", sc->inoutID, sc->inoutID, sc->inoutID);
						sprintf(output + strlen(output), "\
		sdata[%s].x = w.x;\n", sc->inoutID);
					}
				}
				VkAppendLine(sc, sc->disableThreadsEnd);
				appendZeropadEnd(output, sc);
			}
			//appendBarrierVkFFT(output, 1);
			break;
		}
		}

	}

	static inline void appendRadixStageNonStrided(char* output, VkFFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint32_t stageSize, uint32_t stageSizeSum, double stageAngle, uint32_t stageRadix) {
		char vecType[30];
		char LFending[4] = "";
		if (!strcmp(floatType, "float")) sprintf(LFending, "f");
#if(VKFFT_BACKEND==0)
		if (!strcmp(floatType, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "dvec2");
		if (!strcmp(floatType, "double")) sprintf(LFending, "LF");
#elif(VKFFT_BACKEND==1)
		if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
		if (!strcmp(floatType, "double")) sprintf(LFending, "l");
#elif(VKFFT_BACKEND==2)
		if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
		if (!strcmp(floatType, "double")) sprintf(LFending, "l");
#endif

		char convolutionInverse[10] = "";
		if (sc->convolutionStep) {
			if (stageAngle < 0)
				sprintf(convolutionInverse, ", 0");
			else
				sprintf(convolutionInverse, ", 1");
		}
		uint32_t logicalStoragePerThread = sc->registers_per_thread_per_radix[stageRadix] * sc->registerBoost;// (sc->registers_per_thread % stageRadix == 0) ? sc->registers_per_thread * sc->registerBoost : sc->min_registers_per_thread * sc->registerBoost;
		uint32_t logicalRegistersPerThread = sc->registers_per_thread_per_radix[stageRadix];// (sc->registers_per_thread % stageRadix == 0) ? sc->registers_per_thread : sc->min_registers_per_thread;
		uint32_t logicalGroupSize = sc->fftDim / logicalStoragePerThread;
		if ((sc->localSize[0] * logicalStoragePerThread > sc->fftDim) || (stageSize > 1) || (sc->localSize[1] > 1) || ((sc->performR2C) && (sc->inverse)) || ((sc->convolutionStep) && ((sc->matrixConvolution > 1) || (sc->numKernels > 1)) && (stageAngle > 0)))
			appendBarrierVkFFT(output, 1);
		appendZeropadStart(output, sc);
		VkAppendLine(sc, sc->disableThreadsStart);

		if (sc->localSize[0] * logicalStoragePerThread > sc->fftDim)
			sprintf(output + strlen(output), "\
		if (%s * %d < %d) {\n", sc->gl_LocalInvocationID_x, logicalStoragePerThread, sc->fftDim);
		for (uint32_t k = 0; k < sc->registerBoost; k++) {
			for (uint32_t j = 0; j < logicalRegistersPerThread / stageRadix; j++) {
				sprintf(output + strlen(output), "\
		%s = (%s+ %d) %% (%d);\n", sc->stageInvocationID, sc->gl_LocalInvocationID_x, (j + k * logicalRegistersPerThread / stageRadix) * logicalGroupSize, stageSize);
				if (sc->LUT)
					sprintf(output + strlen(output), "		LUTId = stageInvocationID + %d;\n", stageSizeSum);
				else
					sprintf(output + strlen(output), "		angle = stageInvocationID * %.17f%s;\n", stageAngle, LFending);
				if ((sc->registerBoost == 1) && ((sc->localSize[0] * logicalStoragePerThread > sc->fftDim) || (stageSize > 1) || (sc->localSize[1] > 1) || ((sc->performR2C) && (sc->inverse)) || ((sc->convolutionStep) && ((sc->matrixConvolution > 1) || (sc->numKernels > 1)) && (stageAngle > 0)))) {
					for (uint32_t i = 0; i < stageRadix; i++) {
						uint32_t id = j + i * logicalRegistersPerThread / stageRadix;
						id = (id / logicalRegistersPerThread) * sc->registers_per_thread + id % logicalRegistersPerThread;

						sprintf(output + strlen(output), "\
		%s = %s + %d;\n", sc->sdataID, sc->gl_LocalInvocationID_x, j * logicalGroupSize + i * sc->fftDim / stageRadix);

						if (sc->resolveBankConflictFirstStages == 1) {
							sprintf(output + strlen(output), "\
	%s = (%s / %d) * %d + %s %% %d;", sc->sdataID, sc->sdataID, sc->numSharedBanks / 2, sc->numSharedBanks / 2 + 1, sc->sdataID, sc->numSharedBanks / 2);
						}

						if (sc->localSize[1] > 1)
							sprintf(output + strlen(output), "\
		%s = %s + sharedStride * %s;\n", sc->sdataID, sc->sdataID, sc->gl_LocalInvocationID_y);

							sprintf(output + strlen(output), "\
		%s = sdata[%s];\n", sc->regIDs[id], sc->sdataID);
					}
				}
				char** regID = (char**)malloc(sizeof(char*) * stageRadix);
				for (uint32_t i = 0; i < stageRadix; i++) {
					regID[i] = (char*)malloc(sizeof(char) * 40);
					uint32_t id = j + k * logicalRegistersPerThread / stageRadix + i * logicalStoragePerThread / stageRadix;
					id = (id / logicalRegistersPerThread) * sc->registers_per_thread + id % logicalRegistersPerThread;
					sprintf(regID[i], "%s", sc->regIDs[id]);
					/*if(j + i * logicalStoragePerThread / stageRadix < logicalRegistersPerThread)
						sprintf(regID[i], "%s", sc->regIDs[j + i * logicalStoragePerThread / stageRadix]);
					else
						sprintf(regID[i], "%d[%d]", (j + i * logicalStoragePerThread / stageRadix)/ logicalRegistersPerThread, (j + i * logicalStoragePerThread / stageRadix) % logicalRegistersPerThread);*/

				}
				inlineRadixKernelVkFFT(output, sc, floatType, uintType, stageRadix, stageSize, stageAngle, regID);
				for (uint32_t i = 0; i < stageRadix; i++) {
					uint32_t id = j + k * logicalRegistersPerThread / stageRadix + i * logicalStoragePerThread / stageRadix;
					id = (id / logicalRegistersPerThread) * sc->registers_per_thread + id % logicalRegistersPerThread;
					sprintf(sc->regIDs[id], "%s", regID[i]);
				}
				for (uint32_t i = 0; i < stageRadix; i++)
					free(regID[i]);
				free(regID);
			}
			if ((stageSize == 1) && (sc->cacheShuffle)) {
				for (uint32_t i = 0; i < logicalRegistersPerThread; i++) {
					uint32_t id = i + k * logicalRegistersPerThread;
					id = (id / logicalRegistersPerThread) * sc->registers_per_thread + id % logicalRegistersPerThread;
					sprintf(output + strlen(output), "\
		shuffle[%d]=%s;\n", i, sc->regIDs[id]);
				}
				for (uint32_t i = 0; i < logicalRegistersPerThread; i++) {
					uint32_t id = i + k * logicalRegistersPerThread;
					id = (id / logicalRegistersPerThread) * sc->registers_per_thread + id % logicalRegistersPerThread;
					sprintf(output + strlen(output), "\
		%s=shuffle[(%d+tshuffle)%%(%d)];\n", sc->regIDs[id], i, logicalRegistersPerThread);
				}
			}
		}
		if (sc->localSize[0] * logicalStoragePerThread > sc->fftDim)
			sprintf(output + strlen(output), "		}\n");
		VkAppendLine(sc, sc->disableThreadsEnd);
		appendZeropadEnd(output, sc);

	}
	static inline void appendRadixStageStrided(char* output, VkFFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint32_t stageSize, uint32_t stageSizeSum, double stageAngle, uint32_t stageRadix) {
		char vecType[30];
		char LFending[4] = "";
		if (!strcmp(floatType, "float")) sprintf(LFending, "f");
#if(VKFFT_BACKEND==0)
		if (!strcmp(floatType, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "dvec2");
		if (!strcmp(floatType, "double")) sprintf(LFending, "LF");
#elif(VKFFT_BACKEND==1)
		if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
		if (!strcmp(floatType, "double")) sprintf(LFending, "l");
#elif(VKFFT_BACKEND==2)
		if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
		if (!strcmp(floatType, "double")) sprintf(LFending, "l");
#endif

		char convolutionInverse[10] = "";
		if (sc->convolutionStep) {
			if (stageAngle < 0)
				sprintf(convolutionInverse, ", 0");
			else
				sprintf(convolutionInverse, ", 1");
		}
		uint32_t logicalStoragePerThread = sc->registers_per_thread_per_radix[stageRadix] * sc->registerBoost;// (sc->registers_per_thread % stageRadix == 0) ? sc->registers_per_thread * sc->registerBoost : sc->min_registers_per_thread * sc->registerBoost;
		uint32_t logicalRegistersPerThread = sc->registers_per_thread_per_radix[stageRadix];// (sc->registers_per_thread % stageRadix == 0) ? sc->registers_per_thread : sc->min_registers_per_thread;
		uint32_t logicalGroupSize = sc->fftDim / logicalStoragePerThread;
		if ((sc->localSize[1] * logicalStoragePerThread > sc->fftDim) || (stageSize > 1) || ((sc->convolutionStep) && ((sc->matrixConvolution > 1) || (sc->numKernels > 1)) && (stageAngle > 0)))
			appendBarrierVkFFT(output, 1);
		appendZeropadStart(output, sc);
		VkAppendLine(sc, sc->disableThreadsStart);
		if (sc->localSize[1] * logicalStoragePerThread > sc->fftDim)
			sprintf(output + strlen(output), "\
		if (%s * %d < %d) {\n", sc->gl_LocalInvocationID_y, logicalStoragePerThread, sc->fftDim);
		for (uint32_t k = 0; k < sc->registerBoost; k++) {
			for (uint32_t j = 0; j < logicalRegistersPerThread / stageRadix; j++) {
				sprintf(output + strlen(output), "\
		%s = (%s+ %d) %% (%d);\n", sc->stageInvocationID, sc->gl_LocalInvocationID_y, (j + k * logicalRegistersPerThread / stageRadix) * logicalGroupSize, stageSize);
				if (sc->LUT)
					sprintf(output + strlen(output), "		LUTId = stageInvocationID + %d;\n", stageSizeSum);
				else
					sprintf(output + strlen(output), "		angle = stageInvocationID * %.17f%s;\n", stageAngle, LFending);
				if ((sc->registerBoost == 1) && ((sc->localSize[1] * logicalStoragePerThread > sc->fftDim) || (stageSize > 1) || ((sc->convolutionStep) && ((sc->matrixConvolution > 1) || (sc->numKernels > 1)) && (stageAngle > 0)))) {
					for (uint32_t i = 0; i < stageRadix; i++) {
						uint32_t id = j + i * logicalRegistersPerThread / stageRadix;
						id = (id / logicalRegistersPerThread) * sc->registers_per_thread + id % logicalRegistersPerThread;
						sprintf(output + strlen(output), "\
		%s = sdata[%s*(%s+%d)+%s];\n", sc->regIDs[id], sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, j * logicalGroupSize + i * sc->fftDim / stageRadix, sc->gl_LocalInvocationID_x);
					}
				}

				char** regID = (char**)malloc(sizeof(char*) * stageRadix);
				for (uint32_t i = 0; i < stageRadix; i++) {
					regID[i] = (char*)malloc(sizeof(char) * 40);
					uint32_t id = j + k * logicalRegistersPerThread / stageRadix + i * logicalStoragePerThread / stageRadix;
					id = (id / logicalRegistersPerThread) * sc->registers_per_thread + id % logicalRegistersPerThread;
					sprintf(regID[i], "%s", sc->regIDs[id]);
					/*if (j + i * logicalStoragePerThread / stageRadix < logicalRegistersPerThread)
						sprintf(regID[i], "_%d", j + i * logicalStoragePerThread / stageRadix);
					else
						sprintf(regID[i], "%d[%d]", (j + i * logicalStoragePerThread / stageRadix) / logicalRegistersPerThread, (j + i * logicalStoragePerThread / stageRadix) % logicalRegistersPerThread);*/

				}
				inlineRadixKernelVkFFT(output, sc, floatType, uintType, stageRadix, stageSize, stageAngle, regID);
				for (uint32_t i = 0; i < stageRadix; i++) {
					uint32_t id = j + k * logicalRegistersPerThread / stageRadix + i * logicalStoragePerThread / stageRadix;
					id = (id / logicalRegistersPerThread) * sc->registers_per_thread + id % logicalRegistersPerThread;
					sprintf(sc->regIDs[id], "%s", regID[i]);
				}
				for (uint32_t i = 0; i < stageRadix; i++)
					free(regID[i]);
				free(regID);
			}
		}
		if (sc->localSize[1] * logicalStoragePerThread > sc->fftDim)
			sprintf(output + strlen(output), "		}\n");
		VkAppendLine(sc, sc->disableThreadsEnd);
		appendZeropadEnd(output, sc);
	}
	static inline void appendRadixStage(char* output, VkFFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint32_t stageSize, uint32_t stageSizeSum, double stageAngle, uint32_t stageRadix, uint32_t shuffleType) {
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
		}
	}

	static inline void appendRegisterBoostShuffle(char* output, VkFFTSpecializationConstantsLayout* sc, const char* floatType, uint32_t stageSize, uint32_t stageRadix, uint32_t stageAngle, uint32_t shuffleType, uint32_t lastRadix) {
		char vecType[30];
#if(VKFFT_BACKEND==0)
		if (!strcmp(floatType, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "dvec2");
#elif(VKFFT_BACKEND==1)
		if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
#elif(VKFFT_BACKEND==2)
		if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
#endif
		if (((sc->registerBoost > 1) && (stageSize * stageRadix == sc->fftDim) && (sc->stageRadix[sc->numStages - 1] == sc->registerBoost))) {
			uint32_t logicalStoragePerThread = sc->registers_per_thread_per_radix[stageRadix] * sc->registerBoost;// (sc->registers_per_thread % stageRadix == 0) ? sc->registers_per_thread * sc->registerBoost : sc->min_registers_per_thread * sc->registerBoost;
			uint32_t logicalRegistersPerThread = sc->registers_per_thread_per_radix[stageRadix];// (sc->registers_per_thread % stageRadix == 0) ? sc->registers_per_thread : sc->min_registers_per_thread;
			uint32_t* usedIDs = (uint32_t*)malloc(sizeof(uint32_t) * logicalStoragePerThread);
			/*for (uint32_t i = 1; i < logicalStoragePerThread - 1; i++) {
				usedIDs[i] = 0;
			}
			char stageNormalization[10] = "";
			if ((((sc->inverse) && (sc->normalize)) || ((sc->convolutionStep) && (stageAngle > 0))) && lastRadix) {
				sprintf(stageNormalization, " / %d", stageRadix);
			}
			usedIDs[0] = 1;
			usedIDs[logicalStoragePerThread - 1] = 1;
			uint32_t i = 1;
			char* sort0;
			while (i < logicalStoragePerThread) {
				if (usedIDs[i] == 0) {
					uint32_t j = i;
					usedIDs[i] = 1;
					sort0 = sc->regIDs[(i / logicalRegistersPerThread) * sc->registers_per_thread + i % logicalRegistersPerThread];
					//sprintf(output + strlen(output), "	sort0=temp%s;\n", sc->regIDs[(i / logicalRegistersPerThread) * sc->registers_per_thread + i % logicalRegistersPerThread]);
					uint32_t k;
					if (!lastRadix)
						k = (j % stageRadix) * logicalStoragePerThread / stageRadix + j / stageRadix;
					else
						k = (j % (logicalStoragePerThread / stageRadix)) * stageRadix + j / (logicalStoragePerThread / stageRadix);
					while (k != i) {
						//sprintf(output + strlen(output), "	temp%s=temp%s%s;\n", sc->regIDs[(j/ logicalRegistersPerThread)* sc->registers_per_thread+j% logicalRegistersPerThread], sc->regIDs[(k / logicalRegistersPerThread) * sc->registers_per_thread + k % logicalRegistersPerThread], stageNormalization);
						sc->regIDs[(j / logicalRegistersPerThread) * sc->registers_per_thread + j % logicalRegistersPerThread] = sc->regIDs[(k / logicalRegistersPerThread) * sc->registers_per_thread + k % logicalRegistersPerThread];
						usedIDs[k] = 1;
						j = k;
						if (!lastRadix)
							k = (j % stageRadix) * logicalStoragePerThread / stageRadix + j / stageRadix;
						else
							k = (j % (logicalStoragePerThread / stageRadix)) * stageRadix + j / (logicalStoragePerThread / stageRadix);
					}
					//sprintf(output + strlen(output), "	temp%s=sort0%s;\n", sc->regIDs[(j / logicalRegistersPerThread) * sc->registers_per_thread + j % logicalRegistersPerThread], stageNormalization);
					sc->regIDs[(j / logicalRegistersPerThread) * sc->registers_per_thread + j % logicalRegistersPerThread] = sort0;
				}
				else {
					i++;
				}
			}*/
			char** tempID;
			tempID = (char**)malloc(sizeof(char*) * sc->registers_per_thread * sc->registerBoost);
			char** resID;
			//resID = (char**)malloc(sizeof(char*) * sc->registers_per_thread * sc->registerBoost);
			for (uint32_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
				tempID[i] = (char*)malloc(sizeof(char) * 40);
				//resID[i] = sc->regIDs[i];
			}
			for (uint32_t k = 0; k < sc->registerBoost; ++k) {
				uint32_t t = 0;
				for (uint32_t j = 0; j < logicalRegistersPerThread / stageRadix; j++) {
					for (uint32_t i = 0; i < stageRadix; i++) {
						uint32_t id = j + k * logicalRegistersPerThread / stageRadix + i * logicalStoragePerThread / stageRadix;
						id = (id / logicalRegistersPerThread) * sc->registers_per_thread + id % logicalRegistersPerThread;
						sprintf(tempID[t + k * sc->registers_per_thread], "%s", sc->regIDs[id]);
						t++;
						/*sprintf(output + strlen(output), "\
			sdata[sharedStride * gl_LocalInvocationID.y + inoutID + %d] = temp%s%s;\n", i * stageSize, sc->regIDs[id], stageNormalization);*/
					}
				}

				/*for (uint32_t j = logicalRegistersPerThread; j < sc->registers_per_thread; j++) {
					sprintf(tempID[t + k * sc->registers_per_thread], "%s", sc->regIDs[t + k * sc->registers_per_thread]);
					t++;
				}*/
			}
			for (uint32_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
				//printf("0 - %s\n", resID[i]);
				sprintf(sc->regIDs[i], "%s", tempID[i]);
				//sprintf(resID[i], "%s", tempID[i]);
				//printf("1 - %s\n", resID[i]);
			}
			for (uint32_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++)
				free(tempID[i]);
			free(tempID);
			free(usedIDs);
		}
	}

	static inline void appendRadixShuffleNonStrided(char* output, VkFFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint32_t stageSize, uint32_t stageSizeSum, double stageAngle, uint32_t stageRadix, uint32_t stageRadixNext) {
		char vecType[30];
#if(VKFFT_BACKEND==0)
		if (!strcmp(floatType, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "dvec2");
#elif(VKFFT_BACKEND==1)
		if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
#elif(VKFFT_BACKEND==2)
		if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
#endif
		char stageNormalization[10] = "";
		if (((sc->inverse) && (sc->normalize)) || ((sc->convolutionStep) && (stageAngle > 0)))
			sprintf(stageNormalization, "%d", stageRadix);
		char tempNum[50] = "";

		uint32_t logicalStoragePerThread = sc->registers_per_thread_per_radix[stageRadix] * sc->registerBoost;// (sc->registers_per_thread % stageRadix == 0) ? sc->registers_per_thread * sc->registerBoost : sc->min_registers_per_thread * sc->registerBoost;
		uint32_t logicalStoragePerThreadNext = sc->registers_per_thread_per_radix[stageRadixNext] * sc->registerBoost;// (sc->registers_per_thread % stageRadixNext == 0) ? sc->registers_per_thread * sc->registerBoost : sc->min_registers_per_thread * sc->registerBoost;
		uint32_t logicalRegistersPerThread = sc->registers_per_thread_per_radix[stageRadix];// (sc->registers_per_thread % stageRadix == 0) ? sc->registers_per_thread : sc->min_registers_per_thread;
		uint32_t logicalRegistersPerThreadNext = sc->registers_per_thread_per_radix[stageRadixNext];// (sc->registers_per_thread % stageRadixNext == 0) ? sc->registers_per_thread : sc->min_registers_per_thread;

		uint32_t logicalGroupSize = sc->fftDim / logicalStoragePerThread;
		uint32_t logicalGroupSizeNext = sc->fftDim / logicalStoragePerThreadNext;
		if ((sc->registerBoost == 1) && ((sc->localSize[0] * logicalStoragePerThread > sc->fftDim) || (stageSize < sc->fftDim / stageRadix) || ((sc->reorderFourStep) && (sc->fftDim < sc->fft_dim_full) && (sc->localSize[1] > 1)) || (sc->localSize[1] > 1) || ((sc->performR2C) && (!sc->inverse) && (sc->axis_id == 0)) || ((sc->convolutionStep) && ((sc->matrixConvolution > 1) || (sc->numKernels > 1)) && (stageAngle < 0))))
			appendBarrierVkFFT(output, 1);
		if ((sc->localSize[0] * logicalStoragePerThread > sc->fftDim) || (stageSize < sc->fftDim / stageRadix) || ((sc->reorderFourStep) && (sc->fftDim < sc->fft_dim_full) && (sc->localSize[1] > 1)) || (sc->localSize[1] > 1) || ((sc->performR2C) && (!sc->inverse) && (sc->axis_id == 0)) || ((sc->convolutionStep) && ((sc->matrixConvolution > 1) || (sc->numKernels > 1)) && (stageAngle < 0)) || (sc->registerBoost > 1)) {
			//appendBarrierVkFFT(output, 1);
			if (!((sc->registerBoost > 1) && (stageSize * stageRadix == sc->fftDim / sc->stageRadix[sc->numStages - 1]) && (sc->stageRadix[sc->numStages - 1] == sc->registerBoost))) {
				char** tempID;
				tempID = (char**)malloc(sizeof(char*) * sc->registers_per_thread * sc->registerBoost);
				for (uint32_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
					tempID[i] = (char*)malloc(sizeof(char) * 40);
				}
				appendZeropadStart(output, sc);
				VkAppendLine(sc, sc->disableThreadsStart);
				if (sc->localSize[0] * logicalStoragePerThread > sc->fftDim)
					sprintf(output + strlen(output), "\
	if (%s * %d < %d) {\n", sc->gl_GlobalInvocationID_x, logicalStoragePerThread, sc->fftDim);
				for (uint32_t k = 0; k < sc->registerBoost; ++k) {
					uint32_t t = 0;
					if (k > 0) {
						appendBarrierVkFFT(output, 2);
						appendZeropadStart(output, sc);
						VkAppendLine(sc, sc->disableThreadsStart);
						if (sc->localSize[0] * logicalStoragePerThread > sc->fftDim)
							sprintf(output + strlen(output), "\
	if (%s * %d < %d) {\n", sc->gl_GlobalInvocationID_x, logicalStoragePerThread, sc->fftDim);
					}
					for (uint32_t j = 0; j < logicalRegistersPerThread / stageRadix; j++) {
						sprintf(tempNum, "%d", j * logicalGroupSize);
						VkAddReal(sc, sc->stageInvocationID, sc->gl_LocalInvocationID_x, tempNum);
						VkMovReal(sc, sc->blockInvocationID, sc->stageInvocationID);
						sprintf(tempNum, "%d", stageSize);
						VkModReal(sc, sc->stageInvocationID, sc->stageInvocationID, tempNum);
						VkSubReal(sc, sc->blockInvocationID, sc->blockInvocationID, sc->stageInvocationID);
						sprintf(tempNum, "%d", stageRadix);
						VkMulReal(sc, sc->inoutID, sc->blockInvocationID, tempNum);
						VkAddReal(sc, sc->inoutID, sc->inoutID, sc->stageInvocationID);
						//sprintf(output + strlen(output), "\
		stageInvocationID = (gl_LocalInvocationID.x + %d) %% (%d);\n\
		blockInvocationID = (gl_LocalInvocationID.x + %d) - stageInvocationID;\n\
		inoutID = stageInvocationID + blockInvocationID * %d;\n", j * logicalGroupSize, stageSize, j * logicalGroupSize, stageRadix);
						if ((stageSize == 1) && (sc->cacheShuffle)) {
							for (uint32_t i = 0; i < stageRadix; i++) {
								uint32_t id = j + k * logicalRegistersPerThread / stageRadix + i * logicalStoragePerThread / stageRadix;
								id = (id / logicalRegistersPerThread) * sc->registers_per_thread + id % logicalRegistersPerThread;
								sprintf(tempID[t + k * sc->registers_per_thread], "%s", sc->regIDs[id]);
								t++;
								sprintf(tempNum, "%d", i);
								VkAddReal(sc, sc->sdataID, tempNum, sc->tshuffle);
								sprintf(tempNum, "%d", logicalRegistersPerThread);
								VkModReal(sc, sc->sdataID, sc->sdataID, tempNum);
								sprintf(tempNum, "%d", stageSize);
								VkMulReal(sc, sc->sdataID, sc->sdataID, tempNum);
								if (sc->localSize[1] > 1) {
									VkMulReal(sc, sc->combinedID, sc->gl_LocalInvocationID_y, sc->sharedStride);
									VkAddReal(sc, sc->sdataID, sc->sdataID, sc->combinedID);
								}
								VkAddReal(sc, sc->sdataID, sc->sdataID, sc->inoutID);

								//sprintf(sc->sdataID, "sharedStride * gl_LocalInvocationID.y + inoutID + ((%d+tshuffle) %% (%d))*%d", i, logicalRegistersPerThread, stageSize);
								if (strcmp(stageNormalization, ""))
									VkDivComplexNumber(sc, sc->regIDs[id], sc->regIDs[id], stageNormalization);
								VkSharedStore(sc, sc->sdataID, sc->regIDs[id]);
								//sprintf(output + strlen(output), "\
	sdata[sharedStride * gl_LocalInvocationID.y + inoutID + ((%d+tshuffle) %% (%d))*%d] = temp%s%s;\n", i, logicalRegistersPerThread, stageSize, sc->regIDs[id], stageNormalization);
							}
						}
						else {
							for (uint32_t i = 0; i < stageRadix; i++) {
								uint32_t id = j + k * logicalRegistersPerThread / stageRadix + i * logicalStoragePerThread / stageRadix;
								id = (id / logicalRegistersPerThread) * sc->registers_per_thread + id % logicalRegistersPerThread;
								sprintf(tempID[t + k * sc->registers_per_thread], "%s", sc->regIDs[id]);
								t++;
								sprintf(tempNum, "%d", i * stageSize);
								VkAddReal(sc, sc->sdataID, sc->inoutID, tempNum);
								if ((stageSize <= sc->numSharedBanks / 2) && (sc->fftDim > sc->numSharedBanks / 2) && (sc->sharedStrideBankConflictFirstStages != sc->fftDim / sc->registerBoost) && ((sc->fftDim & (sc->fftDim - 1)) == 0) && (stageSize * stageRadix != sc->fftDim)) {
									if (sc->resolveBankConflictFirstStages == 0) {
										sc->resolveBankConflictFirstStages = 1;
										sprintf(output + strlen(output), "\
	%s = %d;", sc->sharedStride, sc->sharedStrideBankConflictFirstStages);
									}
									sprintf(output + strlen(output), "\
	%s = (%s / %d) * %d + %s %% %d;", sc->sdataID, sc->sdataID, sc->numSharedBanks / 2, sc->numSharedBanks / 2 + 1, sc->sdataID, sc->numSharedBanks / 2);

								}
								else {
									if (sc->resolveBankConflictFirstStages == 1) {
										sc->resolveBankConflictFirstStages = 0;
										sprintf(output + strlen(output), "\
	%s = %d;", sc->sharedStride, sc->sharedStrideReadWriteConflict);
									}
								}
								if (sc->localSize[1] > 1) {
									VkMulReal(sc, sc->combinedID, sc->gl_LocalInvocationID_y, sc->sharedStride);
									VkAddReal(sc, sc->sdataID, sc->sdataID, sc->combinedID);
								}
								//sprintf(sc->sdataID, "sharedStride * gl_LocalInvocationID.y + inoutID + %d", i * stageSize);
								if (strcmp(stageNormalization, ""))
									VkDivComplexNumber(sc, sc->regIDs[id], sc->regIDs[id], stageNormalization);
								VkSharedStore(sc, sc->sdataID, sc->regIDs[id]);
								//sprintf(output + strlen(output), "\
	sdata[sharedStride * gl_LocalInvocationID.y + inoutID + %d] = temp%s%s;\n", i * stageSize, sc->regIDs[id], stageNormalization);
							}
						}
					}
					for (uint32_t j = logicalRegistersPerThread; j < sc->registers_per_thread; j++) {
						sprintf(tempID[t + k * sc->registers_per_thread], "%s", sc->regIDs[t + k * sc->registers_per_thread]);
						t++;
					}
					t = 0;
					if (sc->registerBoost > 1) {
						if (sc->localSize[0] * logicalStoragePerThread > sc->fftDim)
							VkAppendLine(sc, "	}\n");
						VkAppendLine(sc, sc->disableThreadsEnd);
						appendZeropadEnd(output, sc);
						appendBarrierVkFFT(output, 2);
						appendZeropadStart(output, sc);
						VkAppendLine(sc, sc->disableThreadsStart);
						if (sc->localSize[0] * logicalStoragePerThreadNext > sc->fftDim)
							sprintf(output + strlen(output), "\
	if (%s * %d < %d) {\n", sc->gl_GlobalInvocationID_x, logicalStoragePerThreadNext, sc->fftDim);
						for (uint32_t j = 0; j < logicalRegistersPerThreadNext / stageRadixNext; j++) {
							for (uint32_t i = 0; i < stageRadixNext; i++) {
								uint32_t id = j + k * logicalRegistersPerThreadNext / stageRadixNext + i * logicalStoragePerThreadNext / stageRadixNext;
								id = (id / logicalRegistersPerThreadNext) * sc->registers_per_thread + id % logicalRegistersPerThreadNext;
								//resID[t + k * sc->registers_per_thread] = sc->regIDs[id];
								sprintf(tempNum, "%d", t * logicalGroupSizeNext);
								VkAddReal(sc, sc->sdataID, sc->gl_LocalInvocationID_x, tempNum);
								if (sc->localSize[1] > 1) {
									VkMulReal(sc, sc->combinedID, sc->gl_LocalInvocationID_y, sc->sharedStride);
									VkAddReal(sc, sc->sdataID, sc->sdataID, sc->combinedID);
								}
								//sprintf(sc->sdataID, "sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + %d", t * logicalGroupSizeNext);
								VkSharedLoad(sc, tempID[t + k * sc->registers_per_thread], sc->sdataID);
								//sprintf(output + strlen(output), "\
		temp%s = sdata[sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + %d];\n", tempID[t + k * sc->registers_per_thread], t * logicalGroupSizeNext);
								t++;
							}

						}
						if (sc->localSize[0] * logicalStoragePerThreadNext > sc->fftDim)
							VkAppendLine(sc, "	}\n");
						VkAppendLine(sc, sc->disableThreadsEnd);
						appendZeropadEnd(output, sc);
					}
					else {
						if (sc->localSize[0] * logicalStoragePerThread > sc->fftDim)
							VkAppendLine(sc, "	}\n");
						VkAppendLine(sc, sc->disableThreadsEnd);
						appendZeropadEnd(output, sc);
					}
				}
				for (uint32_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
					//printf("0 - %s\n", resID[i]);
					sprintf(sc->regIDs[i], "%s", tempID[i]);
					//sprintf(resID[i], "%s", tempID[i]);
					//printf("1 - %s\n", resID[i]);
				}
				for (uint32_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++)
					free(tempID[i]);
				free(tempID);
			}
			else {
				char** tempID;
				tempID = (char**)malloc(sizeof(char*) * sc->registers_per_thread * sc->registerBoost);
				//resID = (char**)malloc(sizeof(char*) * sc->registers_per_thread * sc->registerBoost);
				for (uint32_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
					tempID[i] = (char*)malloc(sizeof(char) * 40);
				}
				for (uint32_t k = 0; k < sc->registerBoost; ++k) {
					for (uint32_t j = 0; j < logicalRegistersPerThread / stageRadix; j++) {
						for (uint32_t i = 0; i < stageRadix; i++) {
							uint32_t id = j + k * logicalRegistersPerThread / stageRadix + i * logicalStoragePerThread / stageRadix;
							id = (id / logicalRegistersPerThread) * sc->registers_per_thread + id % logicalRegistersPerThread;
							sprintf(tempID[j + i * logicalRegistersPerThread / stageRadix + k * sc->registers_per_thread], "%s", sc->regIDs[id]);
						}
					}
					for (uint32_t j = logicalRegistersPerThread; j < sc->registers_per_thread; j++) {
						sprintf(tempID[j + k * sc->registers_per_thread], "%s", sc->regIDs[j + k * sc->registers_per_thread]);
					}
				}
				for (uint32_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
					sprintf(sc->regIDs[i], "%s", tempID[i]);
				}
				for (uint32_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++)
					free(tempID[i]);
				free(tempID);
			}
		}
		else {
			appendZeropadStart(output, sc);
			VkAppendLine(sc, sc->disableThreadsStart);
			if (sc->localSize[0] * logicalStoragePerThread > sc->fftDim)
				sprintf(output + strlen(output), "\
	if (%s * %d < %d) {\n", sc->gl_GlobalInvocationID_x, logicalStoragePerThread, sc->fftDim);
			if (((sc->inverse) && (sc->normalize)) || ((sc->convolutionStep) && (stageAngle > 0))) {
				for (uint32_t i = 0; i < logicalStoragePerThread; i++) {
					VkDivComplexNumber(sc, sc->regIDs[(i / logicalRegistersPerThread) * sc->registers_per_thread + i % logicalRegistersPerThread], sc->regIDs[(i / logicalRegistersPerThread) * sc->registers_per_thread + i % logicalRegistersPerThread], stageNormalization);
					//sprintf(output + strlen(output), "\
		temp%s = temp%s%s;\n", sc->regIDs[(i / logicalRegistersPerThread) * sc->registers_per_thread + i % logicalRegistersPerThread], sc->regIDs[(i / logicalRegistersPerThread) * sc->registers_per_thread + i % logicalRegistersPerThread], stageNormalization);
				}
			}
			if (sc->localSize[0] * logicalStoragePerThread > sc->fftDim)
				VkAppendLine(sc, "	}\n");
			VkAppendLine(sc, sc->disableThreadsEnd);
			appendZeropadEnd(output, sc);
		}

	}
	static inline void appendRadixShuffleStrided(char* output, VkFFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint32_t stageSize, uint32_t stageSizeSum, double stageAngle, uint32_t stageRadix, uint32_t stageRadixNext) {
		char vecType[30];
#if(VKFFT_BACKEND==0)
		if (!strcmp(floatType, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "dvec2");
#elif(VKFFT_BACKEND==1)
		if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
#elif(VKFFT_BACKEND==2)
		if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
#endif

		char stageNormalization[10] = "";
		char tempNum[50] = "";

		uint32_t logicalStoragePerThread = sc->registers_per_thread_per_radix[stageRadix] * sc->registerBoost;// (sc->registers_per_thread % stageRadix == 0) ? sc->registers_per_thread * sc->registerBoost : sc->min_registers_per_thread * sc->registerBoost;
		uint32_t logicalStoragePerThreadNext = sc->registers_per_thread_per_radix[stageRadixNext] * sc->registerBoost;//(sc->registers_per_thread % stageRadixNext == 0) ? sc->registers_per_thread * sc->registerBoost : sc->min_registers_per_thread * sc->registerBoost;
		uint32_t logicalRegistersPerThread = sc->registers_per_thread_per_radix[stageRadix];//(sc->registers_per_thread % stageRadix == 0) ? sc->registers_per_thread : sc->min_registers_per_thread;
		uint32_t logicalRegistersPerThreadNext = sc->registers_per_thread_per_radix[stageRadixNext];//(sc->registers_per_thread % stageRadixNext == 0) ? sc->registers_per_thread : sc->min_registers_per_thread;

		uint32_t logicalGroupSize = sc->fftDim / logicalStoragePerThread;
		uint32_t logicalGroupSizeNext = sc->fftDim / logicalStoragePerThreadNext;
		if (((sc->inverse) && (sc->normalize)) || ((sc->convolutionStep) && (stageAngle > 0)))
			sprintf(stageNormalization, "%d", stageRadix);
		if ((sc->localSize[1] * logicalStoragePerThread > sc->fftDim) || (stageSize < sc->fftDim / stageRadix) || ((sc->convolutionStep) && ((sc->matrixConvolution > 1) || (sc->numKernels > 1)) && (stageAngle < 0)))
			appendBarrierVkFFT(output, 2);

		if ((sc->localSize[1] * logicalStoragePerThread > sc->fftDim) || (stageSize < sc->fftDim / stageRadix) || ((sc->convolutionStep) && ((sc->matrixConvolution > 1) || (sc->numKernels > 1)) && (stageAngle < 0))) {
			//appendBarrierVkFFT(output, 2);
			if (!((sc->registerBoost > 1) && (stageSize * stageRadix == sc->fftDim / sc->stageRadix[sc->numStages - 1]) && (sc->stageRadix[sc->numStages - 1] == sc->registerBoost))) {
				char** tempID;
				tempID = (char**)malloc(sizeof(char*) * sc->registers_per_thread * sc->registerBoost);
				for (uint32_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
					tempID[i] = (char*)malloc(sizeof(char) * 40);
				}
				appendZeropadStart(output, sc);
				VkAppendLine(sc, sc->disableThreadsStart);
				if (sc->localSize[1] * logicalStoragePerThread > sc->fftDim)
					sprintf(output + strlen(output), "\
	if (%s * %d < %d) {\n", sc->gl_LocalInvocationID_y, logicalStoragePerThread, sc->fftDim);
				for (uint32_t k = 0; k < sc->registerBoost; ++k) {
					uint32_t t = 0;
					if (k > 0) {
						appendBarrierVkFFT(output, 2);
						appendZeropadStart(output, sc);
						VkAppendLine(sc, sc->disableThreadsStart);
						if (sc->localSize[1] * logicalStoragePerThread > sc->fftDim)
							sprintf(output + strlen(output), "\
	if (%s * %d < %d) {\n", sc->gl_LocalInvocationID_y, logicalStoragePerThread, sc->fftDim);
					}
					for (uint32_t j = 0; j < logicalRegistersPerThread / stageRadix; j++) {
						sprintf(tempNum, "%d", j * logicalGroupSize);
						VkAddReal(sc, sc->stageInvocationID, sc->gl_LocalInvocationID_y, tempNum);
						VkMovReal(sc, sc->blockInvocationID, sc->stageInvocationID);
						sprintf(tempNum, "%d", stageSize);
						VkModReal(sc, sc->stageInvocationID, sc->stageInvocationID, tempNum);
						VkSubReal(sc, sc->blockInvocationID, sc->blockInvocationID, sc->stageInvocationID);
						sprintf(tempNum, "%d", stageRadix);
						VkMulReal(sc, sc->inoutID, sc->blockInvocationID, tempNum);
						VkAddReal(sc, sc->inoutID, sc->inoutID, sc->stageInvocationID);
						//sprintf(output + strlen(output), "\
		stageInvocationID = (gl_LocalInvocationID.y + %d) %% (%d);\n\
		blockInvocationID = (gl_LocalInvocationID.y + %d) - stageInvocationID;\n\
		inoutID = stageInvocationID + blockInvocationID * %d;\n", j * logicalGroupSize, stageSize, j * logicalGroupSize, stageRadix);
						for (uint32_t i = 0; i < stageRadix; i++) {
							uint32_t id = j + k * logicalRegistersPerThread / stageRadix + i * logicalStoragePerThread / stageRadix;
							id = (id / logicalRegistersPerThread) * sc->registers_per_thread + id % logicalRegistersPerThread;
							sprintf(tempID[t + k * sc->registers_per_thread], "%s", sc->regIDs[id]);
							t++;
							sprintf(tempNum, "%d", i * stageSize);
							VkAddReal(sc, sc->sdataID, sc->inoutID, tempNum);
							VkMulReal(sc, sc->sdataID, sc->gl_WorkGroupSize_x, sc->sdataID);
							VkAddReal(sc, sc->sdataID, sc->sdataID, sc->gl_LocalInvocationID_x);
							//sprintf(sc->sdataID, "sharedStride * gl_LocalInvocationID.y + inoutID + %d", i * stageSize);
							if (strcmp(stageNormalization, ""))
								VkDivComplexNumber(sc, sc->regIDs[id], sc->regIDs[id], stageNormalization);
							VkSharedStore(sc, sc->sdataID, sc->regIDs[id]);
							//sprintf(output + strlen(output), "\
		sdata[gl_WorkGroupSize.x*(inoutID+%d)+gl_LocalInvocationID.x] = temp%s%s;\n", i * stageSize, sc->regIDs[id], stageNormalization);
						}
					}
					for (uint32_t j = logicalRegistersPerThread; j < sc->registers_per_thread; j++) {
						sprintf(tempID[t + k * sc->registers_per_thread], "%s", sc->regIDs[t + k * sc->registers_per_thread]);
						t++;
					}
					t = 0;
					if (sc->registerBoost > 1) {
						if (sc->localSize[1] * logicalStoragePerThread > sc->fftDim)
							VkAppendLine(sc, "	}\n");
						VkAppendLine(sc, sc->disableThreadsEnd);
						appendZeropadEnd(output, sc);
						appendBarrierVkFFT(output, 2);
						appendZeropadStart(output, sc);
						VkAppendLine(sc, sc->disableThreadsStart);
						if (sc->localSize[1] * logicalStoragePerThreadNext > sc->fftDim)
							sprintf(output + strlen(output), "\
	if (%s * %d < %d) {\n", sc->gl_LocalInvocationID_y, logicalStoragePerThreadNext, sc->fftDim);
						for (uint32_t j = 0; j < logicalRegistersPerThreadNext / stageRadixNext; j++) {
							for (uint32_t i = 0; i < stageRadixNext; i++) {
								uint32_t id = j + k * logicalRegistersPerThreadNext / stageRadixNext + i * logicalRegistersPerThreadNext / stageRadixNext;
								id = (id / logicalRegistersPerThreadNext) * sc->registers_per_thread + id % logicalRegistersPerThreadNext;
								sprintf(tempNum, "%d", t * logicalGroupSizeNext);
								VkAddReal(sc, sc->sdataID, sc->gl_LocalInvocationID_y, tempNum);
								VkMulReal(sc, sc->sdataID, sc->gl_WorkGroupSize_x, sc->sdataID);
								VkAddReal(sc, sc->sdataID, sc->sdataID, sc->gl_LocalInvocationID_x);
								//sprintf(sc->sdataID, "sharedStride * gl_LocalInvocationID.y + gl_LocalInvocationID.x + %d", t * logicalGroupSizeNext);
								VkSharedLoad(sc, tempID[t + k * sc->registers_per_thread], sc->sdataID);
								//sprintf(output + strlen(output), "\
		temp%s = sdata[gl_WorkGroupSize.x*(gl_LocalInvocationID.y+%d)+gl_LocalInvocationID.x];\n", tempID[t + k * sc->registers_per_thread], t * logicalGroupSizeNext);
								t++;
							}
						}
						if (sc->localSize[1] * logicalStoragePerThreadNext > sc->fftDim)
							VkAppendLine(sc, "	}\n");
						VkAppendLine(sc, sc->disableThreadsEnd);
						appendZeropadEnd(output, sc);
					}
					else {
						if (sc->localSize[1] * logicalStoragePerThread > sc->fftDim)
							VkAppendLine(sc, "	}\n");
						VkAppendLine(sc, sc->disableThreadsEnd);
						appendZeropadEnd(output, sc);
					}
				}
				for (uint32_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
					sprintf(sc->regIDs[i], "%s", tempID[i]);
				}
				for (uint32_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++)
					free(tempID[i]);
				free(tempID);
			}
			else {
				char** tempID;
				tempID = (char**)malloc(sizeof(char*) * sc->registers_per_thread * sc->registerBoost);
				//resID = (char**)malloc(sizeof(char*) * sc->registers_per_thread * sc->registerBoost);
				for (uint32_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
					tempID[i] = (char*)malloc(sizeof(char) * 40);
				}
				for (uint32_t k = 0; k < sc->registerBoost; ++k) {
					for (uint32_t j = 0; j < logicalRegistersPerThread / stageRadix; j++) {
						for (uint32_t i = 0; i < stageRadix; i++) {
							uint32_t id = j + k * logicalRegistersPerThread / stageRadix + i * logicalStoragePerThread / stageRadix;
							id = (id / logicalRegistersPerThread) * sc->registers_per_thread + id % logicalRegistersPerThread;
							sprintf(tempID[j + i * logicalRegistersPerThread / stageRadix + k * sc->registers_per_thread], "%s", sc->regIDs[id]);
						}
					}
					for (uint32_t j = logicalRegistersPerThread; j < sc->registers_per_thread; j++) {
						sprintf(tempID[j + k * sc->registers_per_thread], "%s", sc->regIDs[j + k * sc->registers_per_thread]);
					}
				}
				for (uint32_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
					sprintf(sc->regIDs[i], "%s", tempID[i]);
				}
				for (uint32_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++)
					free(tempID[i]);
				free(tempID);
			}
		}
		else {
			appendZeropadStart(output, sc);
			VkAppendLine(sc, sc->disableThreadsStart);
			if (sc->localSize[1] * logicalStoragePerThread > sc->fftDim)
				sprintf(output + strlen(output), "\
	if (%s * %d < %d) {\n", sc->gl_LocalInvocationID_y, logicalStoragePerThread, sc->fftDim);
			if (((sc->inverse) && (sc->normalize)) || ((sc->convolutionStep) && (stageAngle > 0))) {
				for (uint32_t i = 0; i < logicalRegistersPerThread; i++) {
					VkDivComplexNumber(sc, sc->regIDs[(i / logicalRegistersPerThread) * sc->registers_per_thread + i % logicalRegistersPerThread], sc->regIDs[(i / logicalRegistersPerThread) * sc->registers_per_thread + i % logicalRegistersPerThread], stageNormalization);
				}
			}
			if (sc->localSize[1] * logicalRegistersPerThread > sc->fftDim)
				VkAppendLine(sc, "	}\n");
			VkAppendLine(sc, sc->disableThreadsEnd);
			appendZeropadEnd(output, sc);
		}
	}
	static inline void appendRadixShuffle(char* output, VkFFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint32_t stageSize, uint32_t stageSizeSum, double stageAngle, uint32_t stageRadix, uint32_t stageRadixNext, uint32_t shuffleType) {
		switch (shuffleType) {
		case 0: case 5: case 6: {
			appendRadixShuffleNonStrided(output, sc, floatType, uintType, stageSize, stageSizeSum, stageAngle, stageRadix, stageRadixNext);
			//appendBarrierVkFFT(output, 1);
			break;
		}
		case 1: case 2: {
			appendRadixShuffleStrided(output, sc, floatType, uintType, stageSize, stageSizeSum, stageAngle, stageRadix, stageRadixNext);
			//appendBarrierVkFFT(output, 1);
			break;
		}
		}
	}

	static inline void appendBoostThreadDataReorder(char* output, VkFFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint32_t shuffleType, uint32_t start) {
		switch (shuffleType) {
		case 0: case 5: case 6: {
			uint32_t logicalStoragePerThread;
			if (start == 1) {
				logicalStoragePerThread = sc->registers_per_thread_per_radix[sc->stageRadix[0]] * sc->registerBoost;// (sc->registers_per_thread % sc->stageRadix[0] == 0) ? sc->registers_per_thread * sc->registerBoost : sc->min_registers_per_thread * sc->registerBoost;
			}
			else {
				logicalStoragePerThread = sc->registers_per_thread_per_radix[sc->stageRadix[sc->numStages - 1]] * sc->registerBoost;// (sc->registers_per_thread % sc->stageRadix[sc->numStages - 1] == 0) ? sc->registers_per_thread * sc->registerBoost : sc->min_registers_per_thread * sc->registerBoost;
			}
			uint32_t logicalGroupSize = sc->fftDim / logicalStoragePerThread;
			if ((sc->registerBoost > 1) && (logicalStoragePerThread != sc->min_registers_per_thread * sc->registerBoost)) {
				for (uint32_t k = 0; k < sc->registerBoost; k++) {
					if (k > 0)
						appendBarrierVkFFT(output, 2);
					appendZeropadStart(output, sc);
					VkAppendLine(sc, sc->disableThreadsStart);
					if (start == 0) {
						sprintf(output + strlen(output), "\
	if (%s * %d < %d) {\n", sc->gl_GlobalInvocationID_x, logicalStoragePerThread, sc->fftDim);
						for (uint32_t i = 0; i < logicalStoragePerThread/ sc->registerBoost; i++) {
							sprintf(output + strlen(output), "\
	sdata[%s + %d] = %s;\n", sc->gl_LocalInvocationID_x, i * logicalGroupSize, sc->regIDs[i + k * sc->registers_per_thread]);
						}
						VkAppendLine(sc, "	}\n");
					}
					else
					{
						for (uint32_t i = 0; i < sc->min_registers_per_thread; i++) {
							sprintf(output + strlen(output), "\
	sdata[%s + %d] = %s;\n", sc->gl_LocalInvocationID_x, i * sc->localSize[0], sc->regIDs[i + k * sc->registers_per_thread]);
						}
					}
					VkAppendLine(sc, sc->disableThreadsEnd);
					appendZeropadEnd(output, sc);
					appendBarrierVkFFT(output, 2);
					appendZeropadStart(output, sc);
					VkAppendLine(sc, sc->disableThreadsStart);
					if (start == 1) {
						sprintf(output + strlen(output), "\
	if (%s * %d < %d) {\n", sc->gl_GlobalInvocationID_x, logicalStoragePerThread, sc->fftDim);
						for (uint32_t i = 0; i < logicalStoragePerThread / sc->registerBoost; i++) {
							sprintf(output + strlen(output), "\
	%s = sdata[%s + %d];\n", sc->regIDs[i + k * sc->registers_per_thread], sc->gl_LocalInvocationID_x, i * logicalGroupSize);
						}
						VkAppendLine(sc, "	}\n");
					}
					else {
						for (uint32_t i = 0; i < sc->min_registers_per_thread; i++) {
							sprintf(output + strlen(output), "\
	%s = sdata[%s + %d];\n", sc->regIDs[i + k * sc->registers_per_thread], sc->gl_LocalInvocationID_x, i * sc->localSize[0]);
						}
					}
					VkAppendLine(sc, sc->disableThreadsEnd);
					appendZeropadEnd(output, sc);
				}
			}

			break;
		}
		case 1: case 2: {
			uint32_t logicalStoragePerThread;
			if (start == 1) {
				logicalStoragePerThread = sc->registers_per_thread_per_radix[sc->stageRadix[0]] * sc->registerBoost;// (sc->registers_per_thread % sc->stageRadix[0] == 0) ? sc->registers_per_thread * sc->registerBoost : sc->min_registers_per_thread * sc->registerBoost;
			}
			else {
				logicalStoragePerThread = sc->registers_per_thread_per_radix[sc->stageRadix[sc->numStages - 1]] * sc->registerBoost;// (sc->registers_per_thread % sc->stageRadix[sc->numStages - 1] == 0) ? sc->registers_per_thread * sc->registerBoost : sc->min_registers_per_thread * sc->registerBoost;
			}
			uint32_t logicalGroupSize = sc->fftDim / logicalStoragePerThread;
			if ((sc->registerBoost > 1) && (logicalStoragePerThread != sc->min_registers_per_thread * sc->registerBoost)) {
				for (uint32_t k = 0; k < sc->registerBoost; k++) {
					if (k > 0)
						appendBarrierVkFFT(output, 2);
					appendZeropadStart(output, sc);
					VkAppendLine(sc, sc->disableThreadsStart);
					if (start == 0) {
						sprintf(output + strlen(output), "\
	if (%s * %d < %d) {\n", sc->gl_GlobalInvocationID_y, logicalStoragePerThread, sc->fftDim);
						for (uint32_t i = 0; i < logicalStoragePerThread/ sc->registerBoost; i++) {
							sprintf(output + strlen(output), "\
	sdata[%s + %s * (%s + %d)] = %s;\n", sc->gl_LocalInvocationID_x, sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, i * logicalGroupSize, sc->regIDs[i + k * sc->registers_per_thread]);
						}
						VkAppendLine(sc, "	}\n");
					}
					else
					{
						for (uint32_t i = 0; i < sc->min_registers_per_thread; i++) {
							sprintf(output + strlen(output), "\
	sdata[%s + %s * (%s + %d)] = %s;\n", sc->gl_LocalInvocationID_x, sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, i * sc->localSize[1], sc->regIDs[i + k * sc->registers_per_thread]);
						}
					}
					VkAppendLine(sc, sc->disableThreadsEnd);
					appendZeropadEnd(output, sc);
					appendBarrierVkFFT(output, 2);
					appendZeropadStart(output, sc);
					VkAppendLine(sc, sc->disableThreadsStart);
					if (start == 1) {
						sprintf(output + strlen(output), "\
	if (%s * %d < %d) {\n", sc->gl_GlobalInvocationID_y, logicalStoragePerThread, sc->fftDim);
						for (uint32_t i = 0; i < logicalStoragePerThread / sc->registerBoost; i++) {
							sprintf(output + strlen(output), "\
	%s = sdata[%s + %s * (%s + %d)];\n", sc->regIDs[i + k * sc->registers_per_thread], sc->gl_LocalInvocationID_x, sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, i * logicalGroupSize);
						}
						VkAppendLine(sc, "	}\n");
					}
					else {
						for (uint32_t i = 0; i < sc->min_registers_per_thread; i++) {
							sprintf(output + strlen(output), "\
	%s = sdata[%s + %s * (%s + %d)];\n", sc->regIDs[i + k * sc->registers_per_thread], sc->gl_LocalInvocationID_x, sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, i * sc->localSize[1]);
						}
					}
					VkAppendLine(sc, sc->disableThreadsEnd);
					appendZeropadEnd(output, sc);
				}
			}

			break;
		}
		}
	}

	static inline void appendCoordinateRegisterStore(char* output, VkFFTSpecializationConstantsLayout* sc, uint32_t readType) {
		switch (readType) {
		case 0://single_c2c
		{
			appendBarrierVkFFT(output, 1);
			appendZeropadStart(output, sc);
			VkAppendLine(sc, sc->disableThreadsStart);
			if (sc->matrixConvolution == 1) {
				sprintf(output + strlen(output), "\
		%s = sdata[sharedStride * %s + %s];\n", sc->regIDs[0], sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x);
				for (uint32_t i = 1; i < sc->min_registers_per_thread; i++) {
					sprintf(output + strlen(output), "\
		%s = sdata[sharedStride * %s + %s + %d * %s];\n", sc->regIDs[i], sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x, i, sc->gl_WorkGroupSize_x);
				}
				//appendBarrierVkFFT(output, 3);
			}
			else {
				sprintf(output + strlen(output), "\
	switch (coordinate) {\n\
	case 0:\n");
				sprintf(output + strlen(output), "\
		%s = sdata[sharedStride * %s + %s];\n", sc->regIDs[0], sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x);
				for (uint32_t i = 1; i < sc->min_registers_per_thread; i++) {
					sprintf(output + strlen(output), "\
		%s = sdata[sharedStride * %s + %s + %d * %s];\n", sc->regIDs[i], sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x, i, sc->gl_WorkGroupSize_x);
				}
				//appendBarrierVkFFT(output, 3);
				sprintf(output + strlen(output), "			break;\n");
				for (uint32_t i = 1; i < sc->matrixConvolution; i++) {
					sprintf(output + strlen(output), "\
	case %d:\n", i);
					sprintf(output + strlen(output), "\
		%s_%d = sdata[sharedStride * %s + %s];\n", sc->regIDs[0], i, sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x);
					for (uint32_t j = 1; j < sc->min_registers_per_thread; j++) {
						sprintf(output + strlen(output), "\
		%s_%d = sdata[sharedStride * %s + %s + %d * %s];\n", sc->regIDs[j], i, sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x, j, sc->gl_WorkGroupSize_x);
					}
					//appendBarrierVkFFT(output, 3);
					sprintf(output + strlen(output), "			break;\n");
				}
				sprintf(output + strlen(output), "		}\n");
			}
			VkAppendLine(sc, sc->disableThreadsEnd);
			appendZeropadEnd(output, sc);
			break;
		}
		case 1://grouped_c2c
		{
			appendBarrierVkFFT(output, 1);
			appendZeropadStart(output, sc);
			VkAppendLine(sc, sc->disableThreadsStart);
			if (sc->matrixConvolution == 1) {
				sprintf(output + strlen(output), "\
		%s = sdata[%s*(%s)+%s];\n", sc->regIDs[0], sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x);
				for (uint32_t i = 1; i < sc->min_registers_per_thread; i++) {
					sprintf(output + strlen(output), "\
		%s = sdata[%s*(%s+%d*%s)+%s];\n", sc->regIDs[i], sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, i, sc->gl_WorkGroupSize_y, sc->gl_LocalInvocationID_x);
				}
				//appendBarrierVkFFT(output, 3);
			}
			else {
				sprintf(output + strlen(output), "\
	switch (coordinate) {\n\
	case 0:\n");
				sprintf(output + strlen(output), "\
		%s = sdata[%s*(%s)+%s];\n", sc->regIDs[0], sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x);
				for (uint32_t i = 1; i < sc->min_registers_per_thread; i++) {
					sprintf(output + strlen(output), "\
		%s = sdata[%s*(%s+%d*%s)+%s];\n", sc->regIDs[i], sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, i, sc->gl_WorkGroupSize_y, sc->gl_LocalInvocationID_x);
				}
				//appendBarrierVkFFT(output, 3);
				sprintf(output + strlen(output), "			break;\n");
				for (uint32_t i = 1; i < sc->matrixConvolution; i++) {
					sprintf(output + strlen(output), "\
	case %d:\n", i);
					sprintf(output + strlen(output), "\
		%s_%d = sdata[%s*(%s)+%s];\n", sc->regIDs[0], i, sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x);
					for (uint32_t j = 1; j < sc->min_registers_per_thread; j++) {
						sprintf(output + strlen(output), "\
		%s_%d = sdata[%s*(%s+%d*%s)+%s];\n", sc->regIDs[j], i, sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, j, sc->gl_WorkGroupSize_y, sc->gl_LocalInvocationID_x);
					}
					//appendBarrierVkFFT(output, 3);
					sprintf(output + strlen(output), "			break;\n");
				}
				sprintf(output + strlen(output), "		}\n");
			}
			VkAppendLine(sc, sc->disableThreadsEnd);
			appendZeropadEnd(output, sc);
			break;
		}
		}
	}
	static inline void appendCoordinateRegisterPull(char* output, VkFFTSpecializationConstantsLayout* sc, uint32_t readType) {
		switch (readType) {
		case 0://single_c2c
		{
			appendBarrierVkFFT(output, 1);
			appendZeropadStart(output, sc);
			VkAppendLine(sc, sc->disableThreadsStart);
			if (sc->matrixConvolution == 1) {
				sprintf(output + strlen(output), "\
			sdata[sharedStride * %s + %s] = %s;\n", sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x, sc->regIDs[0]);
				for (uint32_t i = 1; i < sc->min_registers_per_thread; i++) {
					sprintf(output + strlen(output), "\
			sdata[sharedStride * %s + %s + %d * %s] = %s;\n", sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x, i, sc->gl_WorkGroupSize_x, sc->regIDs[i]);
				}
				//appendBarrierVkFFT(output, 3);
			}
			else {
				sprintf(output + strlen(output), "\
		switch (coordinate) {\n\
		case 0:\n");
				sprintf(output + strlen(output), "\
			sdata[sharedStride * %s + %s] = %s;\n", sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x, sc->regIDs[0]);
				for (uint32_t i = 1; i < sc->min_registers_per_thread; i++) {
					sprintf(output + strlen(output), "\
			sdata[sharedStride * %s + %s + %d * %s] = %s;\n", sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x, i, sc->gl_WorkGroupSize_x, sc->regIDs[i]);
				}
				//appendBarrierVkFFT(output, 3);
				sprintf(output + strlen(output), "			break;\n");
				for (uint32_t i = 1; i < sc->matrixConvolution; i++) {
					sprintf(output + strlen(output), "\
		case %d:\n", i);
					sprintf(output + strlen(output), "\
			sdata[sharedStride * %s + %s] = %s_%d;\n", sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x, sc->regIDs[0], i);
					for (uint32_t j = 1; j < sc->min_registers_per_thread; j++) {
						sprintf(output + strlen(output), "\
			sdata[sharedStride * %s + %s + %d * %s] = %s_%d;\n", sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x, j, sc->gl_WorkGroupSize_x, sc->regIDs[j], i);
					}
					//appendBarrierVkFFT(output, 3);
					sprintf(output + strlen(output), "			break;\n");
				}
				sprintf(output + strlen(output), "		}\n");
			}
			VkAppendLine(sc, sc->disableThreadsEnd);
			appendZeropadEnd(output, sc);
			break;
		}
		case 1://grouped_c2c
		{
			appendBarrierVkFFT(output, 1);
			appendZeropadStart(output, sc);
			VkAppendLine(sc, sc->disableThreadsStart);
			if (sc->matrixConvolution == 1) {
				sprintf(output + strlen(output), "\
		sdata[%s*(%s)+%s] = %s;\n", sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x, sc->regIDs[0]);
				for (uint32_t i = 1; i < sc->min_registers_per_thread; i++) {
					sprintf(output + strlen(output), "\
		sdata[%s*(%s+%d*%s)+%s] = %s;\n", sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, i, sc->gl_WorkGroupSize_y, sc->gl_LocalInvocationID_x, sc->regIDs[i]);
				}
				//appendBarrierVkFFT(output, 3);
			}
			else {
				sprintf(output + strlen(output), "\
	switch (coordinate) {\n\
	case 0:\n");
				sprintf(output + strlen(output), "\
		sdata[%s*(%s)+%s] = %s;\n", sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x, sc->regIDs[0]);
				for (uint32_t i = 1; i < sc->min_registers_per_thread; i++) {
					sprintf(output + strlen(output), "\
		sdata[%s*(%s+%d*%s)+%s] = %s;\n", sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, i, sc->gl_WorkGroupSize_y, sc->gl_LocalInvocationID_x, sc->regIDs[i]);
				}
				//appendBarrierVkFFT(output, 3);
				sprintf(output + strlen(output), "			break;\n");
				for (uint32_t i = 1; i < sc->matrixConvolution; i++) {
					sprintf(output + strlen(output), "\
	case %d:\n", i);
					sprintf(output + strlen(output), "\
		sdata[%s*(%s)+%s] = %s_%d;\n", sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x, sc->regIDs[0], i);
					for (uint32_t j = 1; j < sc->min_registers_per_thread; j++) {
						sprintf(output + strlen(output), "\
		sdata[%s*(%s+%d*%s)+%s] = %s_%d;\n", sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, j, sc->gl_WorkGroupSize_y, sc->gl_LocalInvocationID_x, sc->regIDs[j], i);
					}
					//appendBarrierVkFFT(output, 3);
					sprintf(output + strlen(output), "			break;\n");
				}
				sprintf(output + strlen(output), "		}\n");
			}
			VkAppendLine(sc, sc->disableThreadsEnd);
			appendZeropadEnd(output, sc);
			break;
		}
		}
	}
	static inline void appendPreparationBatchedKernelConvolution(char* output, VkFFTSpecializationConstantsLayout* sc, const char* floatType, const char* floatTypeMemory, const char* uintType, uint32_t dataType) {
		char vecType[30];
#if(VKFFT_BACKEND==0)
		if (!strcmp(floatType, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "dvec2");
#elif(VKFFT_BACKEND==1)
		if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
#elif(VKFFT_BACKEND==2)
		if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
#endif
		char separateRegisterStore[100] = "_store";

		for (uint32_t i = 0; i < sc->registers_per_thread; i++) {
			sprintf(output + strlen(output), "		%s %s%s;\n", vecType, sc->regIDs[i], separateRegisterStore);
			for (uint32_t j = 1; j < sc->matrixConvolution; j++) {
				sprintf(output + strlen(output), "		%s %s_%d%s;\n", vecType, sc->regIDs[i], j, separateRegisterStore);
			}
		}
		for (uint32_t i = 0; i < sc->registers_per_thread; i++) {
			//sprintf(output + strlen(output), "			temp%s[i]=temp[i];\n", separateRegisterStore);
			sprintf(output + strlen(output), "			%s%s=%s;\n", sc->regIDs[i], separateRegisterStore, sc->regIDs[i]);
			for (uint32_t j = 1; j < sc->matrixConvolution; j++) {
				sprintf(output + strlen(output), "			%s_%d%s=%s_%d;\n", sc->regIDs[i], j, separateRegisterStore, sc->regIDs[i], j);
			}
		}
		sprintf(output + strlen(output), "	for (%s batchID=0;  batchID < %d; batchID++){\n", uintType, sc->numKernels);
	}
	static inline void appendKernelConvolution(char* output, VkFFTSpecializationConstantsLayout* sc, const char* floatType, const char* floatTypeMemory, const char* uintType, uint32_t dataType) {
		char shiftX[500] = "";
		if (sc->performWorkGroupShift[0])
			sprintf(shiftX, " + consts.workGroupShiftX * %s ", sc->gl_WorkGroupSize_x);
		char requestCoordinate[100] = "";
		if (sc->convolutionStep) {
			if (sc->matrixConvolution > 1) {
				sprintf(requestCoordinate, "0");
			}
		}
		char index_x[500] = "";
		char index_y[500] = "";
		char requestBatch[100] = "";
		char separateRegisterStore[100] = "";
		if (sc->convolutionStep) {
			if (sc->numKernels > 1) {
				sprintf(requestBatch, "batchID");
				sprintf(separateRegisterStore, "_store");
			}
		}
		appendZeropadStart(output, sc);
		VkAppendLine(sc, sc->disableThreadsStart);
		for (uint32_t j = 0; j < sc->matrixConvolution; j++) {
			sprintf(output + strlen(output), "		%s temp_real%d = 0;\n", floatType, j);
			sprintf(output + strlen(output), "		%s temp_imag%d = 0;\n", floatType, j);
		}
		for (uint32_t i = 0; i < sc->min_registers_per_thread; i++) {
			if (i > 0) {
				for (uint32_t j = 0; j < sc->matrixConvolution; j++) {
					sprintf(output + strlen(output), "		temp_real%d = 0;\n", j);
					sprintf(output + strlen(output), "		temp_imag%d = 0;\n", j);
				}
			}
			switch (dataType) {
			case 0:
			{
				if (sc->fftDim == sc->fft_dim_full) {
					if (sc->localSize[1] == 1)
						sprintf(output + strlen(output), "		combinedID = %s + %d;\n", sc->gl_LocalInvocationID_x, i * sc->localSize[0]);
					else
						sprintf(output + strlen(output), "		combinedID = (%s + %d * %s) + %d;\n", sc->gl_LocalInvocationID_x, sc->localSize[0], sc->gl_LocalInvocationID_y, i * sc->localSize[0] * sc->localSize[1]);

					if (sc->inputStride[0] > 1) {
						sprintf(output + strlen(output), "			%s = ", sc->inoutID);
						sprintf(index_x, "(combinedID %% %d) * %d + (combinedID / %d) * %d", sc->fftDim, sc->inputStride[0], sc->fftDim, sc->inputStride[1]);
						indexInputVkFFT(output, sc, uintType, dataType, index_x, 0, requestCoordinate, requestBatch);
						sprintf(output + strlen(output), ";\n");
						//sprintf(output + strlen(output), "		inoutID = indexInput((combinedID %% %d) * %d + (combinedID / %d) * %d%s%s);\n", sc->fftDim, sc->inputStride[0], sc->fftDim, sc->inputStride[1], requestCoordinate, requestBatch);
					}
					else {
						sprintf(output + strlen(output), "			%s = ", sc->inoutID);
						sprintf(index_x, "(combinedID %% %d) + (combinedID / %d) * %d", sc->fftDim, sc->fftDim, sc->inputStride[1]);
						indexInputVkFFT(output, sc, uintType, dataType, index_x, 0, requestCoordinate, requestBatch);
						sprintf(output + strlen(output), ";\n");
						//sprintf(output + strlen(output), "		inoutID = indexInput((combinedID %% %d) + (combinedID / %d) * %d%s%s);\n", sc->fftDim, sc->fftDim, sc->inputStride[1], requestCoordinate, requestBatch);
					}
				}
				else {
					sprintf(output + strlen(output), "			%s = ", sc->inoutID);
					sprintf(index_x, "%s+%d+%s * %d + (((%s%s) %% %d) * %d + ((%s%s) / %d) * %d)", sc->gl_LocalInvocationID_x, i * sc->localSize[0], sc->gl_LocalInvocationID_y, sc->firstStageStartSize, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fftDim, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->localSize[1] * sc->firstStageStartSize);
					indexInputVkFFT(output, sc, uintType, dataType, index_x, 0, requestCoordinate, requestBatch);
					sprintf(output + strlen(output), ";\n");
					//sprintf(output + strlen(output), "		inoutID = indexInput(%s+%d+%s * %d + (((%s%s) %% %d) * %d + ((%s%s) / %d) * %d)%s%s);\n", sc->gl_LocalInvocationID_x, i * sc->localSize[0], sc->gl_LocalInvocationID_y, sc->firstStageStartSize, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fftDim, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->localSize[1] * sc->firstStageStartSize, requestCoordinate, requestBatch);
				}
				break;
			}
			case 1:
			{
				sprintf(output + strlen(output), "			%s = ", sc->inoutID);
				sprintf(index_x, "(%s%s) %% (%d)", sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x);
				sprintf(index_y, "(%s+%d)+((%s%s)/%d)%%(%d)+((%s%s)/%d)*(%d)", sc->gl_LocalInvocationID_y, i * sc->localSize[1], sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->stageStartSize, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x * sc->stageStartSize, sc->fftDim);
				indexInputVkFFT(output, sc, uintType, dataType, index_x, index_y, requestCoordinate, requestBatch);
				sprintf(output + strlen(output), ";\n");
				//sprintf(output + strlen(output), "		inoutID = indexInput((%s%s) %% (%d), (%s+%d)+((%s%s)/%d)%%(%d)+((%s%s)/%d)*(%d)%s%s);\n", sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->gl_LocalInvocationID_y, i * sc->localSize[1], sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->stageStartSize, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x * sc->stageStartSize, sc->fftDim, requestCoordinate, requestBatch);
				break;
			}
			}
			if (sc->kernelBlockNum == 1) {

				for (uint32_t j = 0; j < sc->matrixConvolution; j++) {
					for (uint32_t l = 0; l < sc->matrixConvolution; l++) {
						uint32_t k = 0;
						if (sc->symmetricKernel) {
							k = (l < j) ? (l * sc->matrixConvolution - l * l + j) : (j * sc->matrixConvolution - j * j + l);
						}
						else {
							k = (j * sc->matrixConvolution + l);
						}
						if (l == 0)
							sprintf(output + strlen(output), "		temp_real%d += kernel[inoutID+%d].x * %s%s.x - kernel[inoutID+%d].y * %s%s.y;\n", j, k * sc->inputStride[3], sc->regIDs[i], separateRegisterStore, k * sc->inputStride[3], sc->regIDs[i], separateRegisterStore);
						else
							sprintf(output + strlen(output), "		temp_real%d += kernel[inoutID+%d].x * %s_%d%s.x - kernel[inoutID+%d].y * %s_%d%s.y;\n", j, k * sc->inputStride[3], sc->regIDs[i], l, separateRegisterStore, k * sc->inputStride[3], sc->regIDs[i], l, separateRegisterStore);
					}
					for (uint32_t l = 0; l < sc->matrixConvolution; l++) {
						uint32_t k = 0;
						if (sc->symmetricKernel) {
							k = (l < j) ? (l * sc->matrixConvolution - l * l + j) : (j * sc->matrixConvolution - j * j + l);
						}
						else {
							k = (j * sc->matrixConvolution + l);
						}
						if (l == 0)
							sprintf(output + strlen(output), "		temp_imag%d += kernel[inoutID+%d].x * %s%s.y + kernel[inoutID+%d].y * %s%s.x;\n", j, k * sc->inputStride[3], sc->regIDs[i], separateRegisterStore, k * sc->inputStride[3], sc->regIDs[i], separateRegisterStore);
						else
							sprintf(output + strlen(output), "		temp_imag%d += kernel[inoutID+%d].x * %s_%d%s.y + kernel[inoutID+%d].y * %s_%d%s.x;\n", j, k * sc->inputStride[3], sc->regIDs[i], l, separateRegisterStore, k * sc->inputStride[3], sc->regIDs[i], l, separateRegisterStore);

					}
				}
				sprintf(output + strlen(output), "		%s.x = temp_real0;", sc->regIDs[i]);
				sprintf(output + strlen(output), "		%s.y = temp_imag0;", sc->regIDs[i]);
				for (uint32_t l = 1; l < sc->matrixConvolution; l++) {
					sprintf(output + strlen(output), "		%s_%d.x = temp_real%d;", sc->regIDs[i], l, l);
					sprintf(output + strlen(output), "		%s_%d.y = temp_imag%d;", sc->regIDs[i], l, l);
				}
			}
			else {
				for (uint32_t j = 0; j < sc->matrixConvolution; j++) {

					sprintf(output + strlen(output), "		%s temp_real%d = 0;\n", floatType, j);
					for (uint32_t l = 0; l < sc->matrixConvolution; l++) {
						uint32_t k = 0;
						if (sc->symmetricKernel) {
							k = (l < j) ? (l * sc->matrixConvolution - l * l + j) : (j * sc->matrixConvolution - j * j + l);
						}
						else {
							k = (j * sc->matrixConvolution + l);
						}
						if (l == 0)
							sprintf(output + strlen(output), "		temp_real%d += kernelBlocks[(inoutID+%d)/%d].kernel[(inoutID+%d) %% %d].x * %s%s.x - kernelBlocks[(inoutID+%d)/%d].kernel[(inoutID+%d) %% %d].y * %s%s.y;\n", j, k * sc->inputStride[3], sc->kernelBlockSize, k * sc->inputStride[3], sc->kernelBlockSize, sc->regIDs[i], separateRegisterStore, k * sc->inputStride[3], sc->kernelBlockSize, k * sc->inputStride[3], sc->kernelBlockSize, sc->regIDs[i], separateRegisterStore);
						else
							sprintf(output + strlen(output), "		temp_real%d += kernelBlocks[(inoutID+%d)/%d].kernel[(inoutID+%d) %% %d].x * %s_%d%s.x - kernelBlocks[(inoutID+%d)/%d].kernel[(inoutID+%d) %% %d].y * %s_%d%s.y;\n", j, k * sc->inputStride[3], sc->kernelBlockSize, k * sc->inputStride[3], sc->kernelBlockSize, sc->regIDs[i], l, separateRegisterStore, k * sc->inputStride[3], sc->kernelBlockSize, k * sc->inputStride[3], sc->kernelBlockSize, sc->regIDs[i], l, separateRegisterStore);

					}

					sprintf(output + strlen(output), "		%s temp_imag%d = 0;\n", floatType, j);
					for (uint32_t l = 0; l < sc->matrixConvolution; l++) {
						uint32_t k = 0;
						if (sc->symmetricKernel) {
							k = (l < j) ? (l * sc->matrixConvolution - l * l + j) : (j * sc->matrixConvolution - j * j + l);
						}
						else {
							k = (j * sc->matrixConvolution + l);
						}
						if (l == 0)
							sprintf(output + strlen(output), "		temp_imag%d += kernelBlocks[(inoutID+%d)/%d].kernel[(inoutID+%d) %% %d].x * %s%s.y + kernelBlocks[(inoutID+%d)/%d].kernel[(inoutID+%d) %% %d].y * %s%s.x;\n", j, k * sc->inputStride[3], sc->kernelBlockSize, k * sc->inputStride[3], sc->kernelBlockSize, sc->regIDs[i], separateRegisterStore, k * sc->inputStride[3], sc->kernelBlockSize, k * sc->inputStride[3], sc->kernelBlockSize, sc->regIDs[i], separateRegisterStore);
						else
							sprintf(output + strlen(output), "		temp_imag%d += kernelBlocks[(inoutID+%d)/%d].kernel[(inoutID+%d) %% %d].x * %s_%d%s.y + kernelBlocks[(inoutID+%d)/%d].kernel[(inoutID+%d) %% %d].y * %s_%d%s.x;\n", j, k * sc->inputStride[3], sc->kernelBlockSize, k * sc->inputStride[3], sc->kernelBlockSize, sc->regIDs[i], l, separateRegisterStore, k * sc->inputStride[3], sc->kernelBlockSize, k * sc->inputStride[3], sc->kernelBlockSize, sc->regIDs[i], l, separateRegisterStore);
					}
				}
				sprintf(output + strlen(output), "		%s.x = temp_real0;", sc->regIDs[i]);
				sprintf(output + strlen(output), "		%s.y = temp_imag0;", sc->regIDs[i]);
				for (uint32_t l = 1; l < sc->matrixConvolution; l++) {
					sprintf(output + strlen(output), "		%s_%d.x = temp_real%d;", sc->regIDs[i], l, l);
					sprintf(output + strlen(output), "		%s_%d.y = temp_imag%d;", sc->regIDs[i], l, l);
				}
			}
		}
		VkAppendLine(sc, sc->disableThreadsEnd);
		appendZeropadEnd(output, sc);
	}
	static inline void appendWriteDataVkFFT(char* output, VkFFTSpecializationConstantsLayout* sc, const char* floatType, const char* floatTypeMemory, const char* uintType, uint32_t writeType) {
		char vecType[30];
		char outputsStruct[20] = "";
#if(VKFFT_BACKEND==0)
		if (!strcmp(floatType, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "dvec2");
		if (sc->outputBufferBlockNum == 1)
			sprintf(outputsStruct, "outputs");
		else
			sprintf(outputsStruct, ".outputs");
#elif(VKFFT_BACKEND==1)
		if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
		sprintf(outputsStruct, "outputs");
#elif(VKFFT_BACKEND==2)
		if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
		sprintf(outputsStruct, "outputs");
#endif
		char convTypeLeft[20] = "";
		char convTypeRight[20] = "";
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
#if(VKFFT_BACKEND==0)
				sprintf(convTypeLeft, "float(");
				sprintf(convTypeRight, ")");
#elif(VKFFT_BACKEND==1)
				sprintf(convTypeLeft, "(float)");
				sprintf(convTypeRight, "");
#elif(VKFFT_BACKEND==2)
				sprintf(convTypeLeft, "(float)");
				sprintf(convTypeRight, "");
#endif
			}
			else {
#if(VKFFT_BACKEND==0)
				sprintf(convTypeLeft, "vec2(");
				sprintf(convTypeRight, ")");
#elif(VKFFT_BACKEND==1)
				sprintf(convTypeLeft, "(float2)");
				sprintf(convTypeRight, "");
#elif(VKFFT_BACKEND==2)
				sprintf(convTypeLeft, "(float2)");
				sprintf(convTypeRight, "");
#endif
			}
		}
		if ((!strcmp(floatTypeMemory, "double")) && (strcmp(floatType, "double"))) {
			if (writeType == 6) {
#if(VKFFT_BACKEND==0)
				sprintf(convTypeLeft, "double(");
				sprintf(convTypeRight, ")");
#elif(VKFFT_BACKEND==1)
				sprintf(convTypeLeft, "(double)");
				sprintf(convTypeRight, "");
#elif(VKFFT_BACKEND==2)
				sprintf(convTypeLeft, "(double)");
				sprintf(convTypeRight, "");
#endif
			}
			else {
#if(VKFFT_BACKEND==0)
				sprintf(convTypeLeft, "dvec2(");
				sprintf(convTypeRight, ")");
#elif(VKFFT_BACKEND==1)
				sprintf(convTypeLeft, "(double2)");
				sprintf(convTypeRight, "");
#elif(VKFFT_BACKEND==2)
				sprintf(convTypeLeft, "(double2)");
				sprintf(convTypeRight, "");
#endif
			}
		}

		char index_x[500] = "";
		char index_y[500] = "";
		char requestCoordinate[100] = "";
		if (sc->convolutionStep) {
			if (sc->matrixConvolution > 1) {
				sprintf(requestCoordinate, "coordinate");
			}
		}
		char requestBatch[100] = "";
		if (sc->convolutionStep) {
			if (sc->numKernels > 1) {
				sprintf(requestBatch, "batchID");//if one buffer - multiple kernel convolution
			}
		}
		switch (writeType) {
		case 0:
		{//single_c2c
			if ((sc->localSize[1] > 1) || (sc->localSize[0] * sc->stageRadix[sc->numStages - 1] * (sc->registers_per_thread_per_radix[sc->stageRadix[sc->numStages - 1]] / sc->stageRadix[sc->numStages - 1]) > sc->fftDim)) {
				sc->writeFromRegisters = 0;
				appendBarrierVkFFT(output, 1);
			}
			else
				sc->writeFromRegisters = 1;
			appendZeropadStart(output, sc);
			char shiftX[500] = "";
			if (sc->performWorkGroupShift[0])
				sprintf(shiftX, " + consts.workGroupShiftX ");
			char shiftY[500] = "";
			if (sc->performWorkGroupShift[1])
				sprintf(shiftY, " + consts.workGroupShiftY*%s ", sc->gl_WorkGroupSize_y);
			char shiftY2[100] = "";
			if (sc->performWorkGroupShift[1])
				sprintf(shiftY, " + consts.workGroupShiftY ");
			if (sc->fftDim < sc->fft_dim_full) {
				sprintf(output + strlen(output), "		if (((%s + %d * %s) %% %d + ((%s%s) / %d)*%d < %d)){\n", sc->gl_LocalInvocationID_x, sc->localSize[0], sc->gl_LocalInvocationID_y, sc->localSize[1], sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->localSize[1], sc->fft_dim_full / sc->firstStageStartSize);
			}
			else {
				sprintf(output + strlen(output), "		{ \n");
			}
			if (sc->reorderFourStep) {
				if (sc->zeropad[1]) {
					if (sc->fftDim == sc->fft_dim_full) {
						for (uint32_t k = 0; k < sc->registerBoost; k++) {
							for (uint32_t i = 0; i < sc->min_registers_per_thread; i++) {
								if (sc->localSize[1] == 1)
									sprintf(output + strlen(output), "		combinedID = %s + %d;\n", sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0]);
								else
									sprintf(output + strlen(output), "		combinedID = (%s + %d * %s) + %d;\n", sc->gl_LocalInvocationID_x, sc->localSize[0], sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[0] * sc->localSize[1]);

								if (sc->outputStride[0] > 1)
									sprintf(output + strlen(output), "		inoutID = (combinedID %% %d) * %d + (combinedID / %d) * %d;\n", sc->fftDim, sc->outputStride[0], sc->fftDim, sc->outputStride[1]);
								else
									sprintf(output + strlen(output), "		inoutID = (combinedID %% %d) + (combinedID / %d) * %d;\n", sc->fftDim, sc->fftDim, sc->outputStride[1]);
								if (sc->size[sc->axis_id + 1] % sc->localSize[1] != 0)
									sprintf(output + strlen(output), "		if(combinedID / %d + (%s%s)*%s< %d){", sc->fftDim, sc->gl_WorkGroupID_y, shiftY2, sc->gl_WorkGroupSize_y, sc->size[sc->axis_id + 1]);
								sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d)){\n", sc->fft_dim_full, sc->fft_zeropad_left_write[sc->axis_id], sc->fft_dim_full, sc->fft_zeropad_right_write[sc->axis_id]);

								sprintf(output + strlen(output), "			%s = ", sc->inoutID);
								indexOutputVkFFT(output, sc, uintType, writeType, sc->inoutID, 0, requestCoordinate, requestBatch);
								sprintf(output + strlen(output), ";\n");

								if (sc->writeFromRegisters) {
									if (sc->outputBufferBlockNum == 1)
										sprintf(output + strlen(output), "		%s[%s] = %s%s%s;\n", outputsStruct, sc->inoutID, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
									else
										sprintf(output + strlen(output), "		outputBlocks[%s / %d]%s[%s %% %d] = %s%s%s;\n", sc->inoutID, sc->outputBufferBlockSize, outputsStruct, sc->inoutID, sc->outputBufferBlockSize, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
								}
								else {
									if (sc->outputBufferBlockNum == 1)
										sprintf(output + strlen(output), "		%s[%s] = %ssdata[(combinedID %% %d) + (combinedID / %d) * sharedStride]%s;\n", outputsStruct, convTypeLeft, sc->inoutID, sc->fftDim, sc->fftDim, convTypeRight);
									else
										sprintf(output + strlen(output), "		outputBlocks[%s / %d]%s[%s %% %d] = %ssdata[(combinedID %% %d) + (combinedID / %d) * sharedStride]%s;\n", sc->inoutID, sc->outputBufferBlockSize, outputsStruct, sc->inoutID, sc->outputBufferBlockSize, convTypeLeft, sc->fftDim, sc->fftDim, convTypeRight);
								}
								VkAppendLine(sc, "	}\n");
								if (sc->size[sc->axis_id + 1] % sc->localSize[1] != 0)
									sprintf(output + strlen(output), "		}");

							}
						}
					}
					else {
						for (uint32_t k = 0; k < sc->registerBoost; k++) {
							for (uint32_t i = 0; i < sc->min_registers_per_thread; i++) {
								if (sc->localSize[1] == 1)
									sprintf(output + strlen(output), "		combinedID = %s + %d;\n", sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0]);
								else
									sprintf(output + strlen(output), "		combinedID = (%s + %d * %s) + %d;\n", sc->gl_LocalInvocationID_x, sc->localSize[0], sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[0] * sc->localSize[1]);
								if (sc->localSize[1] == 1)
									sprintf(output + strlen(output), "		inoutID = (%s%s)/%d+ (combinedID * %d)+ ((%s%s) %% %d) * %d;\n", sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fft_dim_full / sc->fftDim, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fft_dim_full / sc->firstStageStartSize);
								else
									sprintf(output + strlen(output), "		inoutID = combinedID %% %d + ((%s%s) / %d)*%d + ((combinedID/%d) * %d)+ ((%s%s) %% %d) * %d;\n", sc->localSize[1], sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->localSize[1], sc->localSize[1], sc->fft_dim_full / sc->fftDim, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fft_dim_full / sc->firstStageStartSize);

								sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d)){\n", sc->fft_dim_full, sc->fft_zeropad_left_write[sc->axis_id], sc->fft_dim_full, sc->fft_zeropad_right_write[sc->axis_id]);
								sprintf(output + strlen(output), "			%s = ", sc->inoutID);
								indexOutputVkFFT(output, sc, uintType, writeType, sc->inoutID, 0, requestCoordinate, requestBatch);
								sprintf(output + strlen(output), ";\n");
								if (sc->writeFromRegisters) {
									if (sc->outputBufferBlockNum == 1)
										sprintf(output + strlen(output), "			%s[%s] = %s%s%s;\n", outputsStruct, sc->inoutID, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
									else
										sprintf(output + strlen(output), "			outputBlocks[%s / %d]%s[%s %% %d] = %s%s%s;\n", sc->inoutID, sc->outputBufferBlockSize, outputsStruct, sc->inoutID, sc->outputBufferBlockSize, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
								}
								else {
									if (sc->outputBufferBlockNum == 1)
										sprintf(output + strlen(output), "			%s[%s] = %ssdata[(combinedID %% %s)*sharedStride+combinedID/%s]%s;\n", outputsStruct, sc->inoutID, convTypeLeft, sc->gl_WorkGroupSize_y, sc->gl_WorkGroupSize_y, convTypeRight);
									else
										sprintf(output + strlen(output), "			outputBlocks[%s / %d]%s[%s %% %d] = %ssdata[(combinedID %% %s)*sharedStride+combinedID/%s]%s;\n", sc->inoutID, sc->outputBufferBlockSize, outputsStruct, sc->inoutID, sc->outputBufferBlockSize, convTypeLeft, sc->gl_WorkGroupSize_y, sc->gl_WorkGroupSize_y, convTypeRight);
								}
								/*
								if (sc->outputBufferBlockNum == 1)
									if (sc->localSize[1] == 1)
										sprintf(output + strlen(output), "		%s[indexOutput(inoutID%s%s)] = %stemp_%d%s;\n", requestCoordinate, requestBatch, convTypeLeft, i, convTypeRight);
									else
										sprintf(output + strlen(output), "			%s[indexOutput(inoutID%s%s)] = %stemp_%d%s;\n", requestCoordinate, requestBatch, convTypeLeft, i, convTypeRight);
								else
									if (sc->localSize[1] == 1)
										sprintf(output + strlen(output), "			outputBlocks[indexOutput(inoutID%s%s) / %d]%s[indexOutput(inoutID%s%s) %% %d] = %stemp_%d%s;\n", requestCoordinate, requestBatch, sc->outputBufferBlockSize, outputsStruct, requestCoordinate, requestBatch, sc->outputBufferBlockSize, convTypeLeft, i, convTypeRight);
									else
										sprintf(output + strlen(output), "			outputBlocks[indexOutput(inoutID%s%s) / %d]%s[indexOutput(inoutID%s%s) %% %d] = %stemp_%d%s;\n", requestCoordinate, requestBatch, sc->outputBufferBlockSize, outputsStruct, requestCoordinate, requestBatch, sc->outputBufferBlockSize, convTypeLeft, i, convTypeRight);
								*/
								sprintf(output + strlen(output), "	}");
							}
						}
					}
				}
				else {
					if (sc->fftDim == sc->fft_dim_full) {
						for (uint32_t k = 0; k < sc->registerBoost; k++) {
							for (uint32_t i = 0; i < sc->min_registers_per_thread; i++) {
								if (sc->localSize[1] == 1)
									sprintf(output + strlen(output), "		combinedID = %s + %d;\n", sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0]);
								else
									sprintf(output + strlen(output), "		combinedID = (%s + %d * %s) + %d;\n", sc->gl_LocalInvocationID_x, sc->localSize[0], sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[0] * sc->localSize[1]);

								if (sc->outputStride[0] > 1) {
									sprintf(output + strlen(output), "			%s = ", sc->inoutID);
									sprintf(index_x, "(combinedID %% %d) * %d + (combinedID / %d) * %d", sc->fftDim, sc->outputStride[0], sc->fftDim, sc->outputStride[1]);
									indexOutputVkFFT(output, sc, uintType, writeType, index_x, 0, requestCoordinate, requestBatch);
									sprintf(output + strlen(output), ";\n");
									//sprintf(output + strlen(output), "		inoutID = indexOutput((combinedID %% %d) * %d + (combinedID / %d) * %d%s%s);\n", sc->fftDim, sc->outputStride[0], sc->fftDim, sc->outputStride[1], requestCoordinate, requestBatch);
								}
								else {
									sprintf(output + strlen(output), "			%s = ", sc->inoutID);
									sprintf(index_x, "(combinedID %% %d) + (combinedID / %d) * %d", sc->fftDim, sc->fftDim, sc->outputStride[1]);
									indexOutputVkFFT(output, sc, uintType, writeType, index_x, 0, requestCoordinate, requestBatch);
									sprintf(output + strlen(output), ";\n");
									//sprintf(output + strlen(output), "		inoutID = indexOutput((combinedID %% %d) + (combinedID / %d) * %d%s%s);\n", sc->fftDim, sc->fftDim, sc->outputStride[1], requestCoordinate, requestBatch);
								}
								if (sc->size[sc->axis_id + 1] % sc->localSize[1] != 0)
									sprintf(output + strlen(output), "		if(combinedID / %d + %s*%s< %d){", sc->fftDim, sc->gl_WorkGroupID_y, sc->gl_WorkGroupSize_y, sc->size[sc->axis_id + 1]);
								if (sc->writeFromRegisters) {
									if (sc->outputBufferBlockNum == 1)
										sprintf(output + strlen(output), "		%s[inoutID] = %s%s%s;\n", outputsStruct, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
									else
										sprintf(output + strlen(output), "		outputBlocks[inoutID / %d]%s[inoutID %% %d] = %s%s%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
								}
								else {
									if (sc->outputBufferBlockNum == 1)
										sprintf(output + strlen(output), "		%s[inoutID] = %ssdata[(combinedID %% %d) + (combinedID / %d) * sharedStride]%s;\n", outputsStruct, convTypeLeft, sc->fftDim, sc->fftDim, convTypeRight);
									else
										sprintf(output + strlen(output), "		outputBlocks[inoutID / %d]%s[inoutID %% %d] = %ssdata[(combinedID %% %d) + (combinedID / %d) * sharedStride]%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, sc->fftDim, sc->fftDim, convTypeRight);
								}
								if (sc->size[sc->axis_id + 1] % sc->localSize[1] != 0)
									sprintf(output + strlen(output), "		}");
							}
						}
					}
					else {
						for (uint32_t k = 0; k < sc->registerBoost; k++) {
							for (uint32_t i = 0; i < sc->min_registers_per_thread; i++) {
								if (sc->localSize[1] == 1)
									sprintf(output + strlen(output), "		combinedID = %s + %d;\n", sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0]);
								else
									sprintf(output + strlen(output), "		combinedID = (%s + %d * %s) + %d;\n", sc->gl_LocalInvocationID_x, sc->localSize[0], sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[0] * sc->localSize[1]);
								if (sc->localSize[1] == 1) {
									sprintf(output + strlen(output), "			%s = ", sc->inoutID);
									sprintf(index_x, "(%s%s)/%d+ (combinedID * %d)+ ((%s%s) %% %d) * %d", sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fft_dim_full / sc->fftDim, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fft_dim_full / sc->firstStageStartSize);
									indexOutputVkFFT(output, sc, uintType, writeType, index_x, 0, requestCoordinate, requestBatch);
									sprintf(output + strlen(output), ";\n");
									//sprintf(output + strlen(output), "		inoutID = indexOutput((%s%s)/%d+ (combinedID * %d)+ ((%s%s) %% %d) * %d%s%s);\n", sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fft_dim_full / sc->fftDim, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fft_dim_full / sc->firstStageStartSize, requestCoordinate, requestBatch);
								}
								else {
									sprintf(output + strlen(output), "			%s = ", sc->inoutID);
									sprintf(index_x, "combinedID %% %d + ((%s%s) / %d)*%d + ((combinedID/%d) * %d)+ ((%s%s) %% %d) * %d", sc->localSize[1], sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->localSize[1], sc->localSize[1], sc->fft_dim_full / sc->fftDim, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fft_dim_full / sc->firstStageStartSize);
									indexOutputVkFFT(output, sc, uintType, writeType, index_x, 0, requestCoordinate, requestBatch);
									sprintf(output + strlen(output), ";\n");
									//sprintf(output + strlen(output), "		inoutID = indexOutput(combinedID %% %d + ((%s%s) / %d)*%d + ((combinedID/%d) * %d)+ ((%s%s) %% %d) * %d%s%s);\n", sc->localSize[1], sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->localSize[1], sc->localSize[1], sc->fft_dim_full / sc->fftDim, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fft_dim_full / sc->firstStageStartSize, requestCoordinate, requestBatch);
								}
								if (sc->writeFromRegisters) {
									if (sc->outputBufferBlockNum == 1)
										sprintf(output + strlen(output), "		%s[inoutID] = %s%s%s;\n", outputsStruct, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
									else
										sprintf(output + strlen(output), "		outputBlocks[inoutID / %d]%s[inoutID %% %d] = %s%s%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
								}
								else {
									if (sc->outputBufferBlockNum == 1)
										sprintf(output + strlen(output), "		%s[inoutID] = %ssdata[(combinedID %% %s)*sharedStride+combinedID/%s]%s;\n", outputsStruct, convTypeLeft, sc->gl_WorkGroupSize_y, sc->gl_WorkGroupSize_y, convTypeRight);
									else
										sprintf(output + strlen(output), "		outputBlocks[inoutID / %d]%s[inoutID %% %d] = %ssdata[(combinedID %% %s)*sharedStride+combinedID/%s]%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, sc->gl_WorkGroupSize_y, sc->gl_WorkGroupSize_y, convTypeRight);

								}
							}
							/*if (sc->outputBufferBlockNum == 1)
								if (sc->localSize[1] == 1)
									sprintf(output + strlen(output), "		%s[inoutID] = %stemp_%d%s;\n", convTypeLeft, i, convTypeRight);
								else
									sprintf(output + strlen(output), "		%s[inoutID] = %stemp_%d%s;\n", convTypeLeft, i, convTypeRight);
							else
								if (sc->localSize[1] == 1)
									sprintf(output + strlen(output), "		outputBlocks[inoutID / %d]%s[inoutID %% %d] = %stemp_%d%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, i, convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[inoutID / %d]%s[inoutID %% %d] = %stemp_%d%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, i, convTypeRight);
							*/
						}
					}
				}
			}
			else {
				if (sc->zeropad[1]) {
					if (sc->fftDim == sc->fft_dim_full) {
						for (uint32_t k = 0; k < sc->registerBoost; k++) {
							for (uint32_t i = 0; i < sc->min_registers_per_thread; i++) {
								if (sc->localSize[1] == 1)
									sprintf(output + strlen(output), "		combinedID = %s + %d;\n", sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0]);
								else
									sprintf(output + strlen(output), "		combinedID = (%s + %d * %s) + %d;\n", sc->gl_LocalInvocationID_x, sc->localSize[0], sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[0] * sc->localSize[1]);

								if (sc->outputStride[0] > 1)
									sprintf(output + strlen(output), "		inoutID = (combinedID %% %d) * %d + (combinedID / %d) * %d;\n", sc->fftDim, sc->outputStride[0], sc->fftDim, sc->outputStride[1]);
								else
									sprintf(output + strlen(output), "		inoutID = (combinedID %% %d) + (combinedID / %d) * %d;\n", sc->fftDim, sc->fftDim, sc->outputStride[1]);
								if (sc->size[sc->axis_id + 1] % sc->localSize[1] != 0)
									sprintf(output + strlen(output), "		if(combinedID / %d + %s*%s< %d){", sc->fftDim, sc->gl_WorkGroupID_y, sc->gl_WorkGroupSize_y, sc->size[sc->axis_id + 1]);

								sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d)){\n", sc->fft_dim_full, sc->fft_zeropad_left_write[sc->axis_id], sc->fft_dim_full, sc->fft_zeropad_right_write[sc->axis_id]);
								sprintf(output + strlen(output), "			%s = ", sc->inoutID);
								indexOutputVkFFT(output, sc, uintType, writeType, sc->inoutID, 0, requestCoordinate, requestBatch);
								sprintf(output + strlen(output), ";\n");
								if (sc->writeFromRegisters) {
									if (sc->outputBufferBlockNum == 1)
										sprintf(output + strlen(output), "		%s[%s] = %s%s%s;\n", outputsStruct, sc->inoutID, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
									else
										sprintf(output + strlen(output), "		outputBlocks[%s / %d]%s[%s %% %d] = %s%s%s;\n", sc->inoutID, sc->outputBufferBlockSize, outputsStruct, sc->inoutID, sc->outputBufferBlockSize, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
								}
								else {
									if (sc->outputBufferBlockNum == 1)
										sprintf(output + strlen(output), "		%s[%s] = %ssdata[(combinedID %% %d) + (combinedID / %d) * sharedStride]%s;\n", outputsStruct, sc->inoutID, convTypeLeft, sc->fftDim, sc->fftDim, convTypeRight);
									else
										sprintf(output + strlen(output), "		outputBlocks[%s / %d]%s[%s %% %d] = %ssdata[(combinedID %% %d) + (combinedID / %d) * sharedStride]%s;\n", sc->inoutID, sc->outputBufferBlockSize, outputsStruct, sc->inoutID, sc->outputBufferBlockSize, convTypeLeft, sc->fftDim, sc->fftDim, convTypeRight);
								}
								VkAppendLine(sc, "	}\n");
								if (sc->size[sc->axis_id + 1] % sc->localSize[1] != 0)
									sprintf(output + strlen(output), "		}");
							}
						}
					}
					else {
						for (uint32_t k = 0; k < sc->registerBoost; k++) {
							for (uint32_t i = 0; i < sc->min_registers_per_thread; i++) {
								if (sc->localSize[1] == 1)
									sprintf(output + strlen(output), "		combinedID = %s + %d;\n", sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0]);
								else
									sprintf(output + strlen(output), "		combinedID = (%s + %d * %s) + %d;\n", sc->gl_LocalInvocationID_x, sc->localSize[0], sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[0] * sc->localSize[1]);
								if (sc->localSize[1] == 1)
									sprintf(output + strlen(output), "		inoutID = (%s%s)/%d+ (combinedID * %d)+ ((%s%s) %% %d) * %d;\n", sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fft_dim_full / sc->fftDim, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fft_dim_full / sc->firstStageStartSize);
								else
									sprintf(output + strlen(output), "		inoutID = combinedID %% %d + ((%s%s) / %d)*%d + ((combinedID/%d) * %d)+ ((%s%s) %% %d) * %d;\n", sc->localSize[1], sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->localSize[1], sc->localSize[1], sc->fft_dim_full / sc->fftDim, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fft_dim_full / sc->firstStageStartSize);
								sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d)){\n", sc->fft_dim_full, sc->fft_zeropad_left_write[sc->axis_id], sc->fft_dim_full, sc->fft_zeropad_right_write[sc->axis_id]);

								sprintf(output + strlen(output), "			%s = ", sc->inoutID);
								sprintf(index_x, "%s+%d+%s * %d + (((%s%s) %% %d) * %d + ((%s%s) / %d) * %d)", sc->gl_LocalInvocationID_x, i * sc->localSize[0], sc->gl_LocalInvocationID_y, sc->firstStageStartSize, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fftDim, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->localSize[1] * sc->firstStageStartSize);
								indexOutputVkFFT(output, sc, uintType, writeType, index_x, 0, requestCoordinate, requestBatch);
								sprintf(output + strlen(output), ";\n");
								//sprintf(output + strlen(output), "		inoutID = indexOutput(%s+i*%d+%s * %d + (((%s%s) %% %d) * %d + ((%s%s) / %d) * %d)%s%s);\n", sc->gl_LocalInvocationID_x, sc->localSize[0], sc->gl_LocalInvocationID_y, sc->firstStageStartSize, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fftDim, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->localSize[1] * sc->firstStageStartSize, requestCoordinate, requestBatch);
								if (sc->writeFromRegisters) {
									if (sc->outputBufferBlockNum == 1)
										sprintf(output + strlen(output), "		%s[inoutID]=%s%s%s;\n", outputsStruct, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
									else
										sprintf(output + strlen(output), "		outputBlocks[inoutID / %d]%s[inoutID %% %d] = %s%s%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
								}
								else {
									if (sc->outputBufferBlockNum == 1)
										sprintf(output + strlen(output), "		%s[inoutID]=%ssdata[sharedStride*%s + (%s + %d)]%s;\n", outputsStruct, convTypeLeft, sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0], convTypeRight);
									else
										sprintf(output + strlen(output), "		outputBlocks[inoutID / %d]%s[inoutID %% %d] = %ssdata[sharedStride*%s + (%s + %d)]%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0], convTypeRight);
								}
								VkAppendLine(sc, "	}\n");
							}
						}
					}
				}
				else {
					if (sc->fftDim == sc->fft_dim_full) {
						for (uint32_t k = 0; k < sc->registerBoost; k++) {
							for (uint32_t i = 0; i < sc->min_registers_per_thread; i++) {
								if (sc->localSize[1] == 1)
									sprintf(output + strlen(output), "		combinedID = %s + %d;\n", sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0]);
								else
									sprintf(output + strlen(output), "		combinedID = (%s + %d * %s) + %d;\n", sc->gl_LocalInvocationID_x, sc->localSize[0], sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[0] * sc->localSize[1]);
								if (sc->outputStride[0] > 1) {
									sprintf(output + strlen(output), "			%s = ", sc->inoutID);
									sprintf(index_x, "(combinedID %% %d) * %d + (combinedID / %d) * %d", sc->fftDim, sc->outputStride[0], sc->fftDim, sc->outputStride[1]);
									indexOutputVkFFT(output, sc, uintType, writeType, index_x, 0, requestCoordinate, requestBatch);
									sprintf(output + strlen(output), ";\n");
									//sprintf(output + strlen(output), "		inoutID = indexOutput((combinedID %% %d) * %d + (combinedID / %d) * %d%s%s);\n", sc->fftDim, sc->outputStride[0], sc->fftDim, sc->outputStride[1], requestCoordinate, requestBatch);
								}
								else {
									sprintf(output + strlen(output), "			%s = ", sc->inoutID);
									sprintf(index_x, "(combinedID %% %d) + (combinedID / %d) * %d", sc->fftDim, sc->fftDim, sc->outputStride[1]);
									indexOutputVkFFT(output, sc, uintType, writeType, index_x, 0, requestCoordinate, requestBatch);
									sprintf(output + strlen(output), ";\n");
									//sprintf(output + strlen(output), "		inoutID = indexOutput((combinedID %% %d) + (combinedID / %d) * %d%s%s);\n", sc->fftDim, sc->fftDim, sc->outputStride[1], requestCoordinate, requestBatch);
								}
								if (sc->size[sc->axis_id + 1] % sc->localSize[1] != 0)
									sprintf(output + strlen(output), "		if(combinedID / %d + %s*%s< %d){", sc->fftDim, sc->gl_WorkGroupID_y, sc->gl_WorkGroupSize_y, sc->size[sc->axis_id + 1]);

								if (sc->writeFromRegisters) {
									if (sc->outputBufferBlockNum == 1)
										sprintf(output + strlen(output), "		%s[inoutID] = %s%s%s;\n", outputsStruct, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
									else
										sprintf(output + strlen(output), "		outputBlocks[inoutID / %d]%s[inoutID %% %d] = %s%s%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
								}
								else {
									if (sc->outputBufferBlockNum == 1)
										sprintf(output + strlen(output), "		%s[inoutID] = %ssdata[(combinedID %% %d) + (combinedID / %d) * sharedStride]%s;\n", outputsStruct, convTypeLeft, sc->fftDim, sc->fftDim, convTypeRight);
									else
										sprintf(output + strlen(output), "		outputBlocks[inoutID / %d]%s[inoutID %% %d] = %ssdata[(combinedID %% %d) + (combinedID / %d) * sharedStride]%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, sc->fftDim, sc->fftDim, convTypeRight);
								}
								if (sc->size[sc->axis_id + 1] % sc->localSize[1] != 0)
									sprintf(output + strlen(output), "		}");

							}
						}
					}
					else {
						for (uint32_t k = 0; k < sc->registerBoost; k++) {
							for (uint32_t i = 0; i < sc->min_registers_per_thread; i++) {
								sprintf(output + strlen(output), "			%s = ", sc->inoutID);
								sprintf(index_x, "%s+%d+%s * %d + (((%s%s) %% %d) * %d + ((%s%s) / %d) * %d)", sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0], sc->gl_LocalInvocationID_y, sc->firstStageStartSize, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fftDim, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->localSize[1] * sc->firstStageStartSize);
								indexOutputVkFFT(output, sc, uintType, writeType, index_x, 0, requestCoordinate, requestBatch);
								sprintf(output + strlen(output), ";\n");
								//sprintf(output + strlen(output), "		inoutID = indexOutput(%s+%d+%s * %d + (((%s%s) %% %d) * %d + ((%s%s) / %d) * %d)%s%s);\n", sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0], sc->gl_LocalInvocationID_y, sc->firstStageStartSize, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fftDim, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->localSize[1] * sc->firstStageStartSize, requestCoordinate, requestBatch);
								if (sc->writeFromRegisters) {
									if (sc->outputBufferBlockNum == 1)
										sprintf(output + strlen(output), "		%s[inoutID]=%s%s%s;\n", outputsStruct, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
									else
										sprintf(output + strlen(output), "		outputBlocks[inoutID / %d]%s[inoutID %% %d] = %s%s%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
								}
								else {
									if (sc->outputBufferBlockNum == 1)
										sprintf(output + strlen(output), "		%s[inoutID]=%ssdata[sharedStride*%s + (%s + %d)]%s;\n", outputsStruct, convTypeLeft, sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0], convTypeRight);
									else
										sprintf(output + strlen(output), "		outputBlocks[inoutID / %d]%s[inoutID %% %d] = %ssdata[sharedStride*%s + (%s + %d)]%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0], convTypeRight);
								}
							}
						}
					}
				}
			}
			VkAppendLine(sc, "	}\n");
			break;
		}
		case 1: {//grouped_c2c
			if (sc->localSize[1] * sc->stageRadix[sc->numStages - 1] * (sc->registers_per_thread_per_radix[sc->stageRadix[sc->numStages - 1]] / sc->stageRadix[sc->numStages - 1]) > sc->fftDim) {
				sc->writeFromRegisters = 0;
				appendBarrierVkFFT(output, 1);
			}
			else
				sc->writeFromRegisters = 1;
			appendZeropadStart(output, sc);
			char shiftX[500] = "";
			if (sc->performWorkGroupShift[0])
				sprintf(shiftX, " + consts.workGroupShiftX * %s ", sc->gl_WorkGroupSize_x);
			sprintf(output + strlen(output), "		if (((%s%s) / %d) %% (%d)+((%s%s) / %d) * (%d) < %d) {\n", sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->stageStartSize, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x * sc->stageStartSize, sc->fftDim * sc->stageStartSize, sc->size[sc->axis_id]);
			if ((sc->reorderFourStep) && (sc->stageStartSize == 1)) {
				if (sc->zeropad[1]) {
					for (uint32_t k = 0; k < sc->registerBoost; k++) {
						for (uint32_t i = 0; i < sc->min_registers_per_thread; i++) {
							sprintf(output + strlen(output), "		inoutID = (%s + %d) * (%d) + (((%s%s) / %d) %% (%d)) * (%d) + ((%s%s) / %d);\n", sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->fft_dim_full / sc->fftDim, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->firstStageStartSize / sc->fftDim, sc->fft_dim_full / sc->firstStageStartSize, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x * (sc->firstStageStartSize / sc->fftDim));

							sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d)){\n", sc->fft_dim_full, sc->fft_zeropad_left_write[sc->axis_id], sc->fft_dim_full, sc->fft_zeropad_right_write[sc->axis_id]);
							sprintf(output + strlen(output), "			%s = ", sc->inoutID);
							sprintf(index_x, "(%s%s) %% (%d)", sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x);
							indexOutputVkFFT(output, sc, uintType, writeType, index_x, sc->inoutID, requestCoordinate, requestBatch);
							sprintf(output + strlen(output), ";\n");
							if (sc->writeFromRegisters) {
								if (sc->outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "			%s[%s] = %s%s%s;\n", outputsStruct, sc->inoutID, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
								else
									sprintf(output + strlen(output), "			outputBlocks[%s / %d]%s[%s %% %d] = %s%s%s;\n", sc->inoutID, sc->outputBufferBlockSize, outputsStruct, sc->inoutID, sc->outputBufferBlockSize, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
							}
							else {
								if (sc->outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "			%s[%s] = %ssdata[%s*(%s+%d) + %s]%s;\n", outputsStruct, sc->inoutID, convTypeLeft, sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x, convTypeRight);
								else
									sprintf(output + strlen(output), "			outputBlocks[%s / %d]%s[%s %% %d] = %ssdata[%s*(%s+%d) + %s]%s;\n", sc->inoutID, sc->outputBufferBlockSize, outputsStruct, sc->inoutID, sc->outputBufferBlockSize, convTypeLeft, sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x, convTypeRight);

							}
							VkAppendLine(sc, "	}\n");
						}
					}
				}
				else {
					for (uint32_t k = 0; k < sc->registerBoost; k++) {
						for (uint32_t i = 0; i < sc->min_registers_per_thread; i++) {
							sprintf(output + strlen(output), "			%s = ", sc->inoutID);
							sprintf(index_x, "(%s%s) %% (%d)", sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x);
							sprintf(index_y, "(%s + %d) * %d + (((%s%s) / %d) %% (%d)) * (%d) + ((%s%s) / %d )", sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->fft_dim_full / sc->fftDim, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->firstStageStartSize / sc->fftDim, sc->fft_dim_full / sc->firstStageStartSize, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x * (sc->firstStageStartSize / sc->fftDim));
							indexOutputVkFFT(output, sc, uintType, writeType, index_x, index_y, requestCoordinate, requestBatch);
							sprintf(output + strlen(output), ";\n");
							//sprintf(output + strlen(output), "		inoutID = indexOutput((%s%s) %% (%d), (%s + %d) * %d + (((%s%s) / %d) %% (%d)) * (%d) + ((%s%s) / %d )%s%s);\n", sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->fft_dim_full / sc->fftDim, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->firstStageStartSize / sc->fftDim, sc->fft_dim_full / sc->firstStageStartSize, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x * (sc->firstStageStartSize / sc->fftDim), requestCoordinate, requestBatch);
							if (sc->writeFromRegisters) {
								if (sc->outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		%s[inoutID] = %s%s%s;\n", outputsStruct, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[inoutID / %d]%s[inoutID %% %d] = %s%s%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
							}
							else {
								if (sc->outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		%s[inoutID] = %ssdata[%s*(%s+%d) + %s]%s;\n", outputsStruct, convTypeLeft, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x, convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[inoutID / %d]%s[inoutID %% %d] = %ssdata[%s*(%s+%d) + %s]%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x, convTypeRight);
							}
						}
					}
				}
			}
			else {
				if (sc->zeropad[1]) {
					for (uint32_t k = 0; k < sc->registerBoost; k++) {
						for (uint32_t i = 0; i < sc->min_registers_per_thread; i++) {
							sprintf(output + strlen(output), "		inoutID = (%s + %d) * %d + ((%s%s) / %d) %% (%d)+((%s%s) / %d) * (%d);\n", sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->stageStartSize, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->stageStartSize, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x * sc->stageStartSize, sc->stageStartSize * sc->fftDim);

							sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d)){\n", sc->fft_dim_full, sc->fft_zeropad_left_write[sc->axis_id], sc->fft_dim_full, sc->fft_zeropad_right_write[sc->axis_id]);

							sprintf(output + strlen(output), "			%s = ", sc->inoutID);
							sprintf(index_x, "(%s%s) %% (%d)", sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x);
							sprintf(index_y, "%d * (%s + %d) + ((%s%s) / %d) %% (%d)+((%s%s) / %d) * (%d)", sc->stageStartSize, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->stageStartSize, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x * sc->stageStartSize, sc->stageStartSize * sc->fftDim);
							indexOutputVkFFT(output, sc, uintType, writeType, index_x, index_y, requestCoordinate, requestBatch);
							sprintf(output + strlen(output), ";\n");
							//sprintf(output + strlen(output), "		inoutID = indexOutput((%s%s) %% (%d), %d * (%s + %d) + ((%s%s) / %d) %% (%d)+((%s%s) / %d) * (%d)%s%s);\n", sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->stageStartSize, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->stageStartSize, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x * sc->stageStartSize, sc->stageStartSize * sc->fftDim, requestCoordinate, requestBatch);
							if (sc->writeFromRegisters) {
								if (sc->outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "			%s[inoutID] = %s%s%s;\n", outputsStruct, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
								else
									sprintf(output + strlen(output), "			outputBlocks[inoutID / %d]%s[inoutID %% %d] =  %s%s%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
							}
							else {
								if (sc->outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "			%s[inoutID] = %ssdata[%s*(%s+%d) + %s]%s;\n", outputsStruct, convTypeLeft, sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x, convTypeRight);
								else
									sprintf(output + strlen(output), "			outputBlocks[inoutID / %d]%s[inoutID %% %d] =  %ssdata[%s*(%s+%d) + %s]%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x, convTypeRight);
							}
							VkAppendLine(sc, "	}\n");
						}
					}
				}
				else {
					for (uint32_t k = 0; k < sc->registerBoost; k++) {
						for (uint32_t i = 0; i < sc->min_registers_per_thread; i++) {
							sprintf(output + strlen(output), "			%s = ", sc->inoutID);
							sprintf(index_x, "(%s%s) %% (%d)", sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x);
							sprintf(index_y, "%d * (%s + %d) + ((%s%s) / %d) %% (%d)+((%s%s) / %d) * (%d)", sc->stageStartSize, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->stageStartSize, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x * sc->stageStartSize, sc->stageStartSize * sc->fftDim);
							indexOutputVkFFT(output, sc, uintType, writeType, index_x, index_y, requestCoordinate, requestBatch);
							sprintf(output + strlen(output), ";\n");
							//sprintf(output + strlen(output), "		inoutID = indexOutput((%s%s) %% (%d), %d * (%s + %d) + ((%s%s) / %d) %% (%d)+((%s%s) / %d) * (%d)%s%s);\n", sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->stageStartSize, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->stageStartSize, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x * sc->stageStartSize, sc->stageStartSize * sc->fftDim, requestCoordinate, requestBatch);
							if (sc->writeFromRegisters) {
								if (sc->outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		%s[inoutID] = %s%s%s;\n", outputsStruct, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[inoutID / %d]%s[inoutID %% %d] =  %s%s%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
							}
							else {
								if (sc->outputBufferBlockNum == 1)
									sprintf(output + strlen(output), "		%s[inoutID] = %ssdata[%s*(%s+%d) + %s]%s;\n", outputsStruct, convTypeLeft, sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x, convTypeRight);
								else
									sprintf(output + strlen(output), "		outputBlocks[inoutID / %d]%s[inoutID %% %d] =  %ssdata[%s*(%s+%d) + %s]%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x, convTypeRight);
							}
						}
					}
				}
			}
			VkAppendLine(sc, "	}\n");
			break;

		}
		case 2: {//single_c2c_strided
			if (sc->localSize[1] * sc->stageRadix[sc->numStages - 1] * (sc->registers_per_thread_per_radix[sc->stageRadix[sc->numStages - 1]] / sc->stageRadix[sc->numStages - 1]) > sc->fftDim) {
				sc->writeFromRegisters = 0;
				appendBarrierVkFFT(output, 1);
			}
			else
				sc->writeFromRegisters = 1;
			appendZeropadStart(output, sc);
			char shiftX[500] = "";
			if (sc->performWorkGroupShift[0])
				sprintf(shiftX, " + consts.workGroupShiftX * %s ", sc->gl_WorkGroupSize_x);
			sprintf(output + strlen(output), "		if (((%s%s) / %d) * (%d) < %d) {\n", sc->gl_GlobalInvocationID_x, shiftX, sc->stageStartSize, sc->stageStartSize * sc->fftDim, sc->fft_dim_full);
			if (sc->zeropad[1]) {
				for (uint32_t k = 0; k < sc->registerBoost; k++) {
					for (uint32_t i = 0; i < sc->min_registers_per_thread; i++) {
						sprintf(output + strlen(output), "		inoutID = (%s%s) %% (%d) + %d * (%s + %d) + ((%s%s) / %d) * (%d);\n", sc->gl_GlobalInvocationID_x, shiftX, sc->stageStartSize, sc->stageStartSize, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_GlobalInvocationID_x, shiftX, sc->stageStartSize, sc->stageStartSize * sc->fftDim);
						sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d)){\n", sc->fft_dim_full, sc->fft_zeropad_left_write[sc->axis_id], sc->fft_dim_full, sc->fft_zeropad_right_write[sc->axis_id]);
						if (sc->writeFromRegisters) {
							if (sc->outputBufferBlockNum == 1)
								sprintf(output + strlen(output), "			%s[inoutID] = %s%s%s;\n", outputsStruct, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
							else
								sprintf(output + strlen(output), "			outputBlocks[inoutID / %d]%s[inoutID %% %d] = %s%s%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
						}
						else {
							if (sc->outputBufferBlockNum == 1)
								sprintf(output + strlen(output), "			%s[inoutID] = %ssdata[%s*(%s+%d) + %s]%s;\n", outputsStruct, convTypeLeft, sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x, convTypeRight);
							else
								sprintf(output + strlen(output), "			outputBlocks[inoutID / %d]%s[inoutID %% %d] = %ssdata[%s*(%s+%d) + %s]%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x, convTypeRight);
						}
						VkAppendLine(sc, "	}\n");
					}
				}
			}
			else {
				for (uint32_t k = 0; k < sc->registerBoost; k++) {
					for (uint32_t i = 0; i < sc->min_registers_per_thread; i++) {
						sprintf(output + strlen(output), "			%s = ", sc->inoutID);
						sprintf(index_x, "(%s%s) %% (%d) + %d * (%s + %d) + ((%s%s) / %d) * (%d)", sc->gl_GlobalInvocationID_x, shiftX, sc->stageStartSize, sc->stageStartSize, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_GlobalInvocationID_x, shiftX, sc->stageStartSize, sc->stageStartSize * sc->fftDim);
						indexOutputVkFFT(output, sc, uintType, writeType, index_x, 0, requestCoordinate, requestBatch);
						sprintf(output + strlen(output), ";\n");
						//sprintf(output + strlen(output), "		inoutID = indexOutput((%s%s) %% (%d) + %d * (%s + %d) + ((%s%s) / %d) * (%d));\n", sc->gl_GlobalInvocationID_x, shiftX, sc->stageStartSize, sc->stageStartSize, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_GlobalInvocationID_x, shiftX, sc->stageStartSize, sc->stageStartSize * sc->fftDim);
						if (sc->writeFromRegisters) {
							if (sc->outputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		%s[inoutID] = %s%s%s;\n", outputsStruct, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
							else
								sprintf(output + strlen(output), "		outputBlocks[inoutID / %d]%s[inoutID %% %d] = %s%s%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
						}
						else {
							if (sc->outputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		%s[inoutID] = %ssdata[%s*(%s+%d) + %s]%s;\n", outputsStruct, convTypeLeft, sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x, convTypeRight);
							else
								sprintf(output + strlen(output), "		outputBlocks[inoutID / %d]%s[inoutID %% %d] = %ssdata[%s*(%s+%d) + %s]%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, sc->gl_WorkGroupSize_x, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x, convTypeRight);
						}
					}
				}
			}
			VkAppendLine(sc, "	}\n");
			break;

		}
		case 5:
		{//single_r2c
			char shiftY[500] = "";
			if (sc->performWorkGroupShift[1])
				sprintf(shiftY, " + consts.workGroupShiftY * %d", sc->localSize[1]);
			char shiftY2[100] = "";
			if (sc->performWorkGroupShift[1])
				sprintf(shiftY, " + consts.workGroupShiftY ");

			if (sc->reorderFourStep) {
				//Not implemented
			}
			else {
				appendBarrierVkFFT(output, 1);
				appendZeropadStart(output, sc);
				if (sc->fftDim == sc->fft_dim_full) {
					if ((uint32_t)ceil(sc->size[1] / 2.0) % sc->localSize[1] != 0)
						sprintf(output + strlen(output), "		if(%s%s < %d){", sc->gl_GlobalInvocationID_y, shiftY, (uint32_t)ceil(sc->size[1] / 2.0));
					sprintf(output + strlen(output), "\
	if (%s==0)\n\
	{\n\
		temp_0.x = sdata[sharedStride * %s].x;\n\
		temp_0.y = 0;\n\
		temp_1.x = sdata[sharedStride * %s].y;\n\
		temp_1.y = 0;\n", sc->gl_LocalInvocationID_x, sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_y);
					sprintf(output + strlen(output), "			%s = ", sc->inoutID);
					sprintf(index_x, "0");
					sprintf(index_y, "2*%s%s", sc->gl_GlobalInvocationID_y, shiftY);
					indexOutputVkFFT(output, sc, uintType, writeType, index_x, index_y, requestCoordinate, requestBatch);
					sprintf(output + strlen(output), ";\n");
					//sprintf(output + strlen(output), "		inoutID = indexOutput(2 * (%s%s), %d);\n", sc->gl_GlobalInvocationID_y, shiftY, sc->outputStride[2] / (sc->outputStride[1] + 2));

					if (sc->outputBufferBlockNum == 1)
						sprintf(output + strlen(output), "		%s[inoutID] = %stemp_0%s;\n", outputsStruct, convTypeLeft, convTypeRight);
					else
						sprintf(output + strlen(output), "		outputBlocks[inoutID / %d]%s[inoutID %% %d] = %stemp_0%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, convTypeRight);
					sprintf(output + strlen(output), "			%s = ", sc->inoutID);
					sprintf(index_x, "%d", sc->outputStride[1]);
					sprintf(index_y, "2*%s%s", sc->gl_GlobalInvocationID_y, shiftY);
					indexOutputVkFFT(output, sc, uintType, writeType, index_x, index_y, requestCoordinate, requestBatch);
					sprintf(output + strlen(output), ";\n");
					//sprintf(output + strlen(output), "		inoutID = indexOutput(2 * (%s%s) + 1, %d);\n", sc->gl_GlobalInvocationID_y, shiftY, sc->outputStride[2] / (sc->outputStride[1] + 2));
					if (sc->outputBufferBlockNum == 1)
						sprintf(output + strlen(output), "		%s[inoutID] = %stemp_1%s;\n", outputsStruct, convTypeLeft, convTypeRight);
					else
						sprintf(output + strlen(output), "		outputBlocks[inoutID / %d]%s[inoutID %% %d] = %stemp_1%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, convTypeRight);

					VkAppendLine(sc, "	}\n");

					for (uint32_t i = 0; i < ceil(sc->min_registers_per_thread / 2.0); i++) {
						if ((ceil(sc->min_registers_per_thread / 2.0) != sc->min_registers_per_thread / 2) && (i == (ceil(sc->min_registers_per_thread / 2.0) - 1)))
							sprintf(output + strlen(output), "if (%s < %d){\n", sc->gl_LocalInvocationID_x, sc->fftDim / 2 - i * sc->localSize[0]);

						if (sc->localSize[1] == 1)
							sprintf(output + strlen(output), "\
		temp_0.x = 0.5 * (sdata[%d + %s].x + sdata[%d - %s].x);\n\
		temp_0.y = 0.5 * (sdata[%d + %s].y - sdata[%d - %s].y);\n\
		temp_1.x = 0.5 * (sdata[%d + %s].y + sdata[%d - %s].y);\n\
		temp_1.y = 0.5 * (-sdata[%d + %s].x + sdata[%d - %s].x);\n", 1 + i * sc->localSize[0], sc->gl_LocalInvocationID_x, sc->fftDim - 1 - i * sc->localSize[0], sc->gl_LocalInvocationID_x, 1 + i * sc->localSize[0], sc->gl_LocalInvocationID_x, sc->fftDim - 1 - i * sc->localSize[0], sc->gl_LocalInvocationID_x, 1 + i * sc->localSize[0], sc->gl_LocalInvocationID_x, sc->fftDim - 1 - i * sc->localSize[0], sc->gl_LocalInvocationID_x, 1 + i * sc->localSize[0], sc->gl_LocalInvocationID_x, sc->fftDim - 1 - i * sc->localSize[0], sc->gl_LocalInvocationID_x);
						else
							sprintf(output + strlen(output), "\
		temp_0.x = 0.5 * (sdata[sharedStride * %s + (%d + %s)].x + sdata[sharedStride * %s + (%d - %s)].x);\n\
		temp_0.y = 0.5 * (sdata[sharedStride * %s + (%d + %s)].y - sdata[sharedStride * %s + (%d - %s)].y);\n\
		temp_1.x = 0.5 * (sdata[sharedStride * %s + (%d + %s)].y + sdata[sharedStride * %s + (%d - %s)].y);\n\
		temp_1.y = 0.5 * (-sdata[sharedStride * %s + (%d + %s)].x + sdata[sharedStride * %s + (%d - %s)].x);\n", sc->gl_LocalInvocationID_y, 1 + i * sc->localSize[0], sc->gl_LocalInvocationID_x, sc->gl_LocalInvocationID_y, sc->fftDim - 1 - i * sc->localSize[0], sc->gl_LocalInvocationID_x, sc->gl_LocalInvocationID_y, 1 + i * sc->localSize[0], sc->gl_LocalInvocationID_x, sc->gl_LocalInvocationID_y, sc->fftDim - 1 - i * sc->localSize[0], sc->gl_LocalInvocationID_x, sc->gl_LocalInvocationID_y, 1 + i * sc->localSize[0], sc->gl_LocalInvocationID_x, sc->gl_LocalInvocationID_y, sc->fftDim - 1 - i * sc->localSize[0], sc->gl_LocalInvocationID_x, sc->gl_LocalInvocationID_y, 1 + i * sc->localSize[0], sc->gl_LocalInvocationID_x, sc->gl_LocalInvocationID_y, sc->fftDim - 1 - i * sc->localSize[0], sc->gl_LocalInvocationID_x);
						if (sc->zeropad[1]) {

							sprintf(output + strlen(output), "		inoutID = %s+%d;\n", sc->gl_LocalInvocationID_x, i * sc->localSize[0] + 1);
							sprintf(output + strlen(output), "		if((inoutID < %d)||(inoutID >= %d)){\n", sc->fft_zeropad_left_write[sc->axis_id], sc->fft_zeropad_right_write[sc->axis_id]);

							sprintf(output + strlen(output), "			%s = ", sc->inoutID);
							sprintf(index_y, "2*%s%s", sc->gl_GlobalInvocationID_y, shiftY);
							indexOutputVkFFT(output, sc, uintType, writeType, sc->inoutID, index_y, requestCoordinate, requestBatch);
							sprintf(output + strlen(output), ";\n");
							if (sc->outputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		%s[%s] = %stemp_0%s;\n", outputsStruct, sc->inoutID, convTypeLeft, convTypeRight);
							else
								sprintf(output + strlen(output), "		outputBlocks[%s / %d]%s[%s %% %d] = %stemp_0%s;\n", sc->inoutID, sc->outputBufferBlockSize, outputsStruct, sc->inoutID, sc->outputBufferBlockSize, convTypeLeft, convTypeRight);

							sprintf(output + strlen(output), "			%s = ", sc->inoutID);
							sprintf(index_x, "%s+%d", sc->gl_LocalInvocationID_x, sc->outputStride[1] + i * sc->localSize[0] + 1);
							sprintf(index_y, "2*%s%s", sc->gl_GlobalInvocationID_y, shiftY);
							indexOutputVkFFT(output, sc, uintType, writeType, index_x, index_y, requestCoordinate, requestBatch);
							sprintf(output + strlen(output), ";\n");
							//sprintf(output + strlen(output), "		inoutID = indexOutput(%s+%d, (%s%s));\n", sc->gl_LocalInvocationID_x, sc->outputStride[1] / 2 + i * sc->localSize[0], sc->gl_GlobalInvocationID_y, shiftY);

							if (sc->outputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		%s[inoutID] = %stemp_1%s;\n", outputsStruct, convTypeLeft, convTypeRight);
							else
								sprintf(output + strlen(output), "		outputBlocks[inoutID / %d]%s[inoutID %% %d] = %stemp_1%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, convTypeRight);

							VkAppendLine(sc, "	}\n");
						}
						else {
							sprintf(output + strlen(output), "			%s = ", sc->inoutID);
							sprintf(index_x, "%s+%d", sc->gl_LocalInvocationID_x, i * sc->localSize[0] + 1);
							sprintf(index_y, "2*%s%s", sc->gl_GlobalInvocationID_y, shiftY);
							indexOutputVkFFT(output, sc, uintType, writeType, index_x, index_y, requestCoordinate, requestBatch);
							sprintf(output + strlen(output), ";\n");
							//sprintf(output + strlen(output), "		inoutID = indexOutput(%s+%d, (%s%s));\n", sc->gl_LocalInvocationID_x, i * sc->localSize[0], sc->gl_GlobalInvocationID_y, shiftY);

							if (sc->outputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		%s[inoutID] = %stemp_0%s;\n", outputsStruct, convTypeLeft, convTypeRight);
							else
								sprintf(output + strlen(output), "		outputBlocks[inoutID / %d]%s[inoutID %% %d] = %stemp_0%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, convTypeRight);

							sprintf(output + strlen(output), "			%s = ", sc->inoutID);
							sprintf(index_x, "%s+%d", sc->gl_LocalInvocationID_x, sc->outputStride[1] + i * sc->localSize[0] + 1);
							sprintf(index_y, "2*%s%s", sc->gl_GlobalInvocationID_y, shiftY);
							indexOutputVkFFT(output, sc, uintType, writeType, index_x, index_y, requestCoordinate, requestBatch);
							sprintf(output + strlen(output), ";\n");
							//sprintf(output + strlen(output), "		inoutID = indexOutput(%s+%d, (%s%s));\n", sc->gl_LocalInvocationID_x, sc->outputStride[1] / 2 + i * sc->localSize[0], sc->gl_GlobalInvocationID_y, shiftY);

							if (sc->outputBufferBlockNum == 1)
								sprintf(output + strlen(output), "		%s[inoutID] = %stemp_1%s;\n", outputsStruct, convTypeLeft, convTypeRight);
							else
								sprintf(output + strlen(output), "		outputBlocks[inoutID / %d]%s[inoutID %% %d] = %stemp_1%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, convTypeRight);

						}
						if ((ceil(sc->min_registers_per_thread / 2.0) != sc->min_registers_per_thread / 2) && (i == (ceil(sc->min_registers_per_thread / 2.0) - 1))) {
							sprintf(output + strlen(output), "}\n");
						}
					}
					if ((uint32_t)ceil(sc->size[1] / 2.0) % sc->localSize[1] != 0)
						sprintf(output + strlen(output), "		}");
				}
				else {
					//Not implemented
				}
			}
			break;
		}
		case 6: {//single_c2r
			char shiftY[500] = "";
			if (sc->performWorkGroupShift[1])
				sprintf(shiftY, " + consts.workGroupShiftY * %d", sc->localSize[1]);
			if ((sc->localSize[1] > 1) || (sc->localSize[0] * sc->stageRadix[sc->numStages - 1] * (sc->registers_per_thread / sc->stageRadix[sc->numStages - 1]) > sc->fftDim)) {
				sc->writeFromRegisters = 0;
				appendBarrierVkFFT(output, 1);
			}
			else
				sc->writeFromRegisters = 1;
			appendZeropadStart(output, sc);
			if (sc->reorderFourStep) {
				//Not implemented
			}
			else {
				if (sc->zeropad[1]) {
					if (sc->fftDim == sc->fft_dim_full) {
						for (uint32_t k = 0; k < sc->registerBoost; k++) {
							for (uint32_t i = 0; i < sc->min_registers_per_thread; i++) {
								if (sc->localSize[1] == 1)
									sprintf(output + strlen(output), "		combinedID = %s + %d;\n", sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0]);
								else
									sprintf(output + strlen(output), "		combinedID = (%s + %d * %s) + %d;\n", sc->gl_LocalInvocationID_x, sc->localSize[0], sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[0] * sc->localSize[1]);

								if (sc->outputStride[0] > 1)
									sprintf(output + strlen(output), "		inoutID = (combinedID %% %d) * %d + (combinedID / %d) * %d;\n", sc->fftDim, sc->outputStride[0], sc->fftDim, 2 * sc->outputStride[1]);
								else
									sprintf(output + strlen(output), "		inoutID = (combinedID %% %d) + (combinedID / %d) * %d;\n", sc->fftDim, sc->fftDim, 2 * sc->outputStride[1]);
								if ((uint32_t)ceil(sc->size[1] / 2.0) % sc->localSize[1] != 0)
									sprintf(output + strlen(output), "		if(combinedID / %d + (%s%s)*%s< %d){", sc->fftDim, sc->gl_WorkGroupID_y, shiftY, sc->gl_WorkGroupSize_y, (uint32_t)ceil(sc->size[1] / 2.0));

								sprintf(output + strlen(output), "		if((inoutID %% %d < %d)||(inoutID %% %d >= %d)){\n", sc->fft_dim_full, sc->fft_zeropad_left_write[sc->axis_id], sc->fft_dim_full, sc->fft_zeropad_right_write[sc->axis_id]);
								if (sc->writeFromRegisters) {
									sprintf(output + strlen(output), "			%s = ", sc->inoutID);
									indexOutputVkFFT(output, sc, uintType, writeType, sc->inoutID, 0, requestCoordinate, requestBatch);
									sprintf(output + strlen(output), ";\n");
									if (sc->outputBufferBlockNum == 1)
										sprintf(output + strlen(output), "		%s[%s] = %s%s.x%s;\n", outputsStruct, sc->inoutID, convTypeLeft, sc->regIDs[i], convTypeRight);
									else
										sprintf(output + strlen(output), "		outputBlocks[%s / %d]%s[%s %% %d] = %s%s.x%s;\n", sc->inoutID, sc->outputBufferBlockSize, outputsStruct, sc->inoutID, sc->outputBufferBlockSize, convTypeLeft, sc->regIDs[i], convTypeRight);

									sprintf(output + strlen(output), "			%s = %s + %d;", sc->inoutID, sc->inoutID, sc->outputStride[1]);
									/*sprintf(index_x, "%s+%d", sc->inoutID, sc->outputStride[1]);
									indexOutputVkFFT(output, sc, uintType, writeType, index_x, 0, requestCoordinate, requestBatch);
									sprintf(output + strlen(output), ";\n");*/

									if (sc->outputBufferBlockNum == 1)
										sprintf(output + strlen(output), "		%s[%s] = %s%s.y%s;\n", outputsStruct, sc->inoutID, convTypeLeft, sc->regIDs[i], convTypeRight);
									else
										sprintf(output + strlen(output), "		outputBlocks[%s / %d]%s[%s %% %d] = %s%s.y%s;\n", sc->inoutID, sc->outputBufferBlockSize, outputsStruct, sc->inoutID, sc->outputBufferBlockSize, convTypeLeft, sc->regIDs[i], convTypeRight);
								}
								else {
									sprintf(output + strlen(output), "			%s = ", sc->inoutID);
									indexOutputVkFFT(output, sc, uintType, writeType, sc->inoutID, 0, requestCoordinate, requestBatch);
									sprintf(output + strlen(output), ";\n");
									if (sc->outputBufferBlockNum == 1)
										sprintf(output + strlen(output), "		%s[%s] = %ssdata[(combinedID %% %d) + (combinedID / %d) * sharedStride].x%s;\n", outputsStruct, sc->inoutID, convTypeLeft, sc->fftDim, sc->fftDim, convTypeRight);
									else
										sprintf(output + strlen(output), "		outputBlocks[%s / %d]%s[%s %% %d] = %ssdata[(combinedID %% %d) + (combinedID / %d) * sharedStride].x%s;\n", sc->inoutID, sc->outputBufferBlockSize, outputsStruct, sc->inoutID, sc->outputBufferBlockSize, convTypeLeft, sc->fftDim, sc->fftDim, convTypeRight);

									sprintf(output + strlen(output), "			%s = %s + %d;", sc->inoutID, sc->inoutID, sc->outputStride[1]);
									/*sprintf(index_x, "%s+%d", sc->inoutID, sc->outputStride[1]);
									indexOutputVkFFT(output, sc, uintType, writeType, index_x, 0, requestCoordinate, requestBatch);
									sprintf(output + strlen(output), ";\n");*/

									if (sc->outputBufferBlockNum == 1)
										sprintf(output + strlen(output), "		%s[%s] = %ssdata[(combinedID %% %d) + (combinedID / %d) * sharedStride].y%s;\n", outputsStruct, sc->inoutID, convTypeLeft, sc->fftDim, sc->fftDim, convTypeRight);
									else
										sprintf(output + strlen(output), "		outputBlocks[%s / %d]%s[%s %% %d] = %ssdata[(combinedID %% %d) + (combinedID / %d) * sharedStride].y%s;\n", sc->inoutID, sc->outputBufferBlockSize, outputsStruct, sc->inoutID, sc->outputBufferBlockSize, convTypeLeft, sc->fftDim, sc->fftDim, convTypeRight);

								}
								VkAppendLine(sc, "	}\n");
								if ((uint32_t)ceil(sc->size[1] / 2.0) % sc->localSize[1] != 0)
									sprintf(output + strlen(output), "		}");
							}
						}
					}
					else {

					}
				}
				else {
					if (sc->fftDim == sc->fft_dim_full) {
						for (uint32_t k = 0; k < sc->registerBoost; k++) {
							for (uint32_t i = 0; i < sc->min_registers_per_thread; i++) {
								if (sc->localSize[1] == 1)
									sprintf(output + strlen(output), "		combinedID = %s + %d;\n", sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0]);
								else
									sprintf(output + strlen(output), "		combinedID = (%s + %d * %s) + %d;\n", sc->gl_LocalInvocationID_x, sc->localSize[0], sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[0] * sc->localSize[1]);

								if (sc->outputStride[0] > 1) {
									sprintf(output + strlen(output), "			%s = ", sc->inoutID);
									sprintf(index_x, "(combinedID %% %d) * %d + (combinedID / %d) * %d", sc->fftDim, sc->outputStride[0], sc->fftDim, 2 * sc->outputStride[1]);
									indexOutputVkFFT(output, sc, uintType, writeType, index_x, 0, requestCoordinate, requestBatch);
									sprintf(output + strlen(output), ";\n");

									//sprintf(output + strlen(output), "		inoutID = indexOutput((combinedID %% %d) * %d + (combinedID / %d) * %d);\n", sc->fftDim, sc->outputStride[0], sc->fftDim, 2 * sc->outputStride[1]);
								}
								else {
									sprintf(output + strlen(output), "			%s = ", sc->inoutID);
									sprintf(index_x, "(combinedID %% %d) + (combinedID / %d) * %d", sc->fftDim, sc->fftDim, 2 * sc->outputStride[1]);
									indexOutputVkFFT(output, sc, uintType, writeType, index_x, 0, requestCoordinate, requestBatch);
									sprintf(output + strlen(output), ";\n");

									//sprintf(output + strlen(output), "		inoutID = indexOutput((combinedID %% %d) + (combinedID / %d) * %d);\n", sc->fftDim, sc->fftDim, 2 * sc->outputStride[1]);
								}
								if ((uint32_t)ceil(sc->size[1] / 2.0) % sc->localSize[1] != 0)
									sprintf(output + strlen(output), "		if(combinedID / %d + (%s%s)*%s< %d){", sc->fftDim, sc->gl_WorkGroupID_y, shiftY, sc->gl_WorkGroupSize_y, (uint32_t)ceil(sc->size[1] / 2.0));

								if (sc->writeFromRegisters) {
									if (sc->outputBufferBlockNum == 1)
										sprintf(output + strlen(output), "		%s[inoutID] = %s%s.x%s;\n", outputsStruct, convTypeLeft, sc->regIDs[i], convTypeRight);
									else
										sprintf(output + strlen(output), "		outputBlocks[inoutID / %d]%s[inoutID %% %d] = %s%s.x%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, sc->regIDs[i], convTypeRight);
									sprintf(output + strlen(output), "		inoutID += %d;\n", sc->outputStride[1]);
									if (sc->outputBufferBlockNum == 1)
										sprintf(output + strlen(output), "		%s[inoutID] = %s%s.y%s;\n", outputsStruct, convTypeLeft, sc->regIDs[i], convTypeRight);
									else
										sprintf(output + strlen(output), "		outputBlocks[inoutID / %d]%s[inoutID %% %d] = %s%s.y%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, sc->regIDs[i], convTypeRight);
								}
								else {
									if (sc->outputBufferBlockNum == 1)
										sprintf(output + strlen(output), "		%s[inoutID] = %ssdata[(combinedID %% %d) + (combinedID / %d) * sharedStride].x%s;\n", outputsStruct, convTypeLeft, sc->fftDim, sc->fftDim, convTypeRight);
									else
										sprintf(output + strlen(output), "		outputBlocks[inoutID / %d]%s[inoutID %% %d] = %ssdata[(combinedID %% %d) + (combinedID / %d) * sharedStride].x%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, sc->fftDim, sc->fftDim, convTypeRight);
									sprintf(output + strlen(output), "		inoutID += %d;\n", sc->outputStride[1]);
									if (sc->outputBufferBlockNum == 1)
										sprintf(output + strlen(output), "		%s[inoutID] = %ssdata[(combinedID %% %d) + (combinedID / %d) * sharedStride].y%s;\n", outputsStruct, convTypeLeft, sc->fftDim, sc->fftDim, convTypeRight);
									else
										sprintf(output + strlen(output), "		outputBlocks[inoutID / %d]%s[inoutID %% %d] = %ssdata[(combinedID %% %d) + (combinedID / %d) * sharedStride].y%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, sc->fftDim, sc->fftDim, convTypeRight);
								}
								if ((uint32_t)ceil(sc->size[1] / 2.0) % sc->localSize[1] != 0)
									sprintf(output + strlen(output), "		}");
							}
						}
					}
					else {

					}

				}
			}

			break;
		}
		}
		appendZeropadEnd(output, sc);
	}

	static inline void shaderGenVkFFT(char* output, VkFFTSpecializationConstantsLayout* sc, const char* floatType, const char* floatTypeInputMemory, const char* floatTypeOutputMemory, const char* floatTypeKernelMemory, const char* uintType, uint32_t type) {
		//appendLicense(output);
		sc->output = output;
		char vecType[30];
		char vecTypeInput[30];
		char vecTypeOutput[30];
#if(VKFFT_BACKEND==0)
		if (!strcmp(floatType, "half")) sprintf(vecType, "f16vec2");
		if (!strcmp(floatType, "float")) sprintf(vecType, "vec2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "dvec2");
		if (!strcmp(floatTypeInputMemory, "half")) sprintf(vecTypeInput, "f16vec2");
		if (!strcmp(floatTypeInputMemory, "float")) sprintf(vecTypeInput, "vec2");
		if (!strcmp(floatTypeInputMemory, "double")) sprintf(vecTypeInput, "dvec2");
		if (!strcmp(floatTypeOutputMemory, "half")) sprintf(vecTypeOutput, "f16vec2");
		if (!strcmp(floatTypeOutputMemory, "float")) sprintf(vecTypeOutput, "vec2");
		if (!strcmp(floatTypeOutputMemory, "double")) sprintf(vecTypeOutput, "dvec2");
		sprintf(sc->gl_LocalInvocationID_x, "gl_LocalInvocationID.x");
		sprintf(sc->gl_LocalInvocationID_y, "gl_LocalInvocationID.y");
		sprintf(sc->gl_LocalInvocationID_z, "gl_LocalInvocationID.z");
		sprintf(sc->gl_GlobalInvocationID_x, "gl_GlobalInvocationID.x");
		sprintf(sc->gl_GlobalInvocationID_y, "gl_GlobalInvocationID.y");
		sprintf(sc->gl_GlobalInvocationID_z, "gl_GlobalInvocationID.z");
		sprintf(sc->gl_WorkGroupID_x, "gl_WorkGroupID.x");
		sprintf(sc->gl_WorkGroupID_y, "gl_WorkGroupID.y");
		sprintf(sc->gl_WorkGroupID_z, "gl_WorkGroupID.z");
		sprintf(sc->gl_WorkGroupSize_x, "gl_WorkGroupSize.x");
		sprintf(sc->gl_WorkGroupSize_y, "gl_WorkGroupSize.y");
		sprintf(sc->gl_WorkGroupSize_z, "gl_WorkGroupSize.z");
#elif(VKFFT_BACKEND==1)
		if (!strcmp(floatType, "half")) sprintf(vecType, "f16vec2");
		if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
		if (!strcmp(floatTypeInputMemory, "half")) sprintf(vecTypeInput, "f16vec2");
		if (!strcmp(floatTypeInputMemory, "float")) sprintf(vecTypeInput, "float2");
		if (!strcmp(floatTypeInputMemory, "double")) sprintf(vecTypeInput, "double2");
		if (!strcmp(floatTypeOutputMemory, "half")) sprintf(vecTypeOutput, "f16vec2");
		if (!strcmp(floatTypeOutputMemory, "float")) sprintf(vecTypeOutput, "float2");
		if (!strcmp(floatTypeOutputMemory, "double")) sprintf(vecTypeOutput, "double2");
		sprintf(sc->gl_LocalInvocationID_x, "threadIdx.x");
		sprintf(sc->gl_LocalInvocationID_y, "threadIdx.y");
		sprintf(sc->gl_LocalInvocationID_z, "threadIdx.z");
		sprintf(sc->gl_GlobalInvocationID_x, "(threadIdx.x + blockIdx.x * blockDim.x)");
		sprintf(sc->gl_GlobalInvocationID_y, "(threadIdx.y + blockIdx.y * blockDim.y)");
		sprintf(sc->gl_GlobalInvocationID_z, "(threadIdx.z + blockIdx.z * blockDim.z)");
		sprintf(sc->gl_WorkGroupID_x, "blockIdx.x");
		sprintf(sc->gl_WorkGroupID_y, "blockIdx.y");
		sprintf(sc->gl_WorkGroupID_z, "blockIdx.z");
		sprintf(sc->gl_WorkGroupSize_x, "blockDim.x");
		sprintf(sc->gl_WorkGroupSize_y, "blockDim.y");
		sprintf(sc->gl_WorkGroupSize_z, "blockDim.z");
#elif(VKFFT_BACKEND==2)
		if (!strcmp(floatType, "half")) sprintf(vecType, "f16vec2");
		if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
		if (!strcmp(floatTypeInputMemory, "half")) sprintf(vecTypeInput, "f16vec2");
		if (!strcmp(floatTypeInputMemory, "float")) sprintf(vecTypeInput, "float2");
		if (!strcmp(floatTypeInputMemory, "double")) sprintf(vecTypeInput, "double2");
		if (!strcmp(floatTypeOutputMemory, "half")) sprintf(vecTypeOutput, "f16vec2");
		if (!strcmp(floatTypeOutputMemory, "float")) sprintf(vecTypeOutput, "float2");
		if (!strcmp(floatTypeOutputMemory, "double")) sprintf(vecTypeOutput, "double2");
		sprintf(sc->gl_LocalInvocationID_x, "threadIdx.x");
		sprintf(sc->gl_LocalInvocationID_y, "threadIdx.y");
		sprintf(sc->gl_LocalInvocationID_z, "threadIdx.z");
		sprintf(sc->gl_GlobalInvocationID_x, "(threadIdx.x + blockIdx.x * blockDim.x)");
		sprintf(sc->gl_GlobalInvocationID_y, "(threadIdx.y + blockIdx.y * blockDim.y)");
		sprintf(sc->gl_GlobalInvocationID_z, "(threadIdx.z + blockIdx.z * blockDim.z)");
		sprintf(sc->gl_WorkGroupID_x, "blockIdx.x");
		sprintf(sc->gl_WorkGroupID_y, "blockIdx.y");
		sprintf(sc->gl_WorkGroupID_z, "blockIdx.z");
		sprintf(sc->gl_WorkGroupSize_x, "blockDim.x");
		sprintf(sc->gl_WorkGroupSize_y, "blockDim.y");
		sprintf(sc->gl_WorkGroupSize_z, "blockDim.z");
#endif
		sprintf(sc->stageInvocationID, "stageInvocationID");
		sprintf(sc->blockInvocationID, "blockInvocationID");
		sprintf(sc->tshuffle, "tshuffle");
		sprintf(sc->sharedStride, "sharedStride");
		sprintf(sc->combinedID, "combinedID");
		sprintf(sc->inoutID, "inoutID");
		sprintf(sc->sdataID, "sdataID");
		//sprintf(sc->tempReg, "temp");
		sc->disableThreadsStart = (char*)malloc(sizeof(char) * 200);
		sc->disableThreadsEnd = (char*)malloc(sizeof(char) * 2);
		sprintf(sc->disableThreadsStart, "");
		sprintf(sc->disableThreadsEnd, "");
		appendVersion(output);
		appendExtensions(output, floatType, floatTypeInputMemory, floatTypeOutputMemory, floatTypeKernelMemory);
		appendLayoutVkFFT(output, sc);
		appendConstantsVkFFT(output, floatType, uintType);
		if ((!sc->LUT) && (!strcmp(floatType, "double")))
			appendSinCos20(output, floatType, uintType);
		appendPushConstantsVkFFT(output, sc, floatType, uintType);
		uint32_t id = 0;
		appendInputLayoutVkFFT(output, sc, id, floatTypeInputMemory, type);
		id++;
		appendOutputLayoutVkFFT(output, sc, id, floatTypeOutputMemory, type);
		id++;
		if (sc->convolutionStep) {
			appendKernelLayoutVkFFT(output, sc, id, floatTypeKernelMemory);
			id++;
		}
		if (sc->LUT) {
			appendLUTLayoutVkFFT(output, sc, id, floatType);
			id++;
		}
		//appendIndexInputVkFFT(output, sc, uintType, type);
		//appendIndexOutputVkFFT(output, sc, uintType, type);
		/*uint32_t appendedRadix[10] = { 0,0,0,0,0,0,0,0,0,0 };
		for (uint32_t i = 0; i < sc->numStages; i++) {
			if (appendedRadix[sc->stageRadix[i]] == 0) {
				appendedRadix[sc->stageRadix[i]] = 1;
				appendRadixKernelVkFFT(output, sc, floatType, uintType, sc->stageRadix[i]);
			}
		}*/
#if(VKFFT_BACKEND==0)
		appendSharedMemoryVkFFT(output, sc, floatType, uintType, type);
		sprintf(output + strlen(output), "void main() {\n");
#elif(VKFFT_BACKEND==1)
		if (type == 5)
			sprintf(output + strlen(output), "extern \"C\" __global__ void VkFFT_main (%s* inputs, %s* outputs", floatTypeInputMemory, vecTypeOutput);
		else {
			if (type == 6)
				sprintf(output + strlen(output), "extern \"C\" __global__ void VkFFT_main (%s* inputs, %s* outputs", vecTypeInput, floatTypeOutputMemory);
			else
				sprintf(output + strlen(output), "extern \"C\" __global__ void VkFFT_main (%s* inputs, %s* outputs", vecTypeInput, vecTypeOutput);
		}
		if (sc->convolutionStep) {
			sprintf(output + strlen(output), ", %s* kernel", vecType);
		}
		if (sc->LUT) {
			sprintf(output + strlen(output), ", %s* twiddleLUT", vecType);
		}
		sprintf(output + strlen(output), ") {\n");
		//sprintf(output + strlen(output), ", const PushConsts consts) {\n"); 
		appendSharedMemoryVkFFT(output, sc, floatType, uintType, type);
#elif(VKFFT_BACKEND==2)
		if (type == 5)
			sprintf(output + strlen(output), "extern \"C\" __global__ void VkFFT_main (%s* inputs, %s* outputs", floatTypeInputMemory, vecTypeOutput);
		else {
			if (type == 6)
				sprintf(output + strlen(output), "extern \"C\" __global__ void VkFFT_main (%s* inputs, %s* outputs", vecTypeInput, floatTypeOutputMemory);
			else
				sprintf(output + strlen(output), "extern \"C\" __global__ void VkFFT_main (%s* inputs, %s* outputs", vecTypeInput, vecTypeOutput);
		}
		if (sc->convolutionStep) {
			sprintf(output + strlen(output), ", %s* kernel", vecType);
		}
		if (sc->LUT) {
			sprintf(output + strlen(output), ", %s* twiddleLUT", vecType);
		}
		sprintf(output + strlen(output), ") {\n");
		//sprintf(output + strlen(output), ", const PushConsts consts) {\n"); 
		appendSharedMemoryVkFFT(output, sc, floatType, uintType, type);
#endif
		//if (type==0) sprintf(output + strlen(output), "return;\n");
		appendInitialization(output, sc, floatType, uintType, type);
		if ((sc->convolutionStep) && (sc->matrixConvolution > 1))
			sprintf(output + strlen(output), "	for (%s coordinate=%d; coordinate > 0; coordinate--){\n\
	coordinate--;\n", uintType, sc->matrixConvolution);
		appendReadDataVkFFT(output, sc, floatType, floatTypeInputMemory, uintType, type);
		//appendBarrierVkFFT(output, 1);
		appendReorder4StepRead(output, sc, floatType, uintType, type);
		appendBoostThreadDataReorder(output, sc, floatType, uintType, type, 1);
		uint32_t stageSize = 1;
		uint32_t stageSizeSum = 0;
		double PI_const = 3.1415926535897932384626433832795;
		double stageAngle = (sc->inverse) ? PI_const : -PI_const;
		for (uint32_t i = 0; i < sc->numStages; i++) {
			if ((i == sc->numStages - 1) && (sc->registerBoost > 1)) {
				//appendRegisterBoostShuffle(output, sc, floatType, stageSize, sc->stageRadix[i-1], stageAngle, type, 1);
				//appendRegisterBoostShuffle(output, sc, floatType, stageSize, sc->stageRadix[i], stageAngle, type, 0);
				appendRadixStage(output, sc, floatType, uintType, stageSize, stageSizeSum, stageAngle, sc->stageRadix[i], type);
				//appendRegisterBoostShuffle(output, sc, floatType, stageSize, sc->stageRadix[i], stageAngle, type, 1);
				//appendRegisterBoostShuffle(output, sc, floatType, 8 / ((sc->sharedMemSizePow2 / sc->complexSize) / (sc->fftDim / sc->stageRadix[i])), stageAngle, type, 0);

			}
			else {
				//if (i>0)
					//appendRegisterBoostShuffle(output, sc, floatType, sc->stageRadix[i], stageAngle, type, 0);
				appendRadixStage(output, sc, floatType, uintType, stageSize, stageSizeSum, stageAngle, sc->stageRadix[i], type);
				switch (sc->stageRadix[i]) {
				case 2:
					stageSizeSum += stageSize;
					break;
				case 3:
					stageSizeSum += stageSize * 2;
					break;
				case 4:
					stageSizeSum += stageSize * 2;
					break;
				case 5:
					stageSizeSum += stageSize * 4;
					break;
				case 7:
					stageSizeSum += stageSize * 6;
					break;
				case 8:
					stageSizeSum += stageSize * 3;
					break;
				case 11:
					stageSizeSum += stageSize * 10;
					break;
				case 13:
					stageSizeSum += stageSize * 12;
					break;
				}
				if (i == sc->numStages - 1)
					appendRadixShuffle(output, sc, floatType, uintType, stageSize, stageSizeSum, stageAngle, sc->stageRadix[i], sc->stageRadix[i], type);
				else
					appendRadixShuffle(output, sc, floatType, uintType, stageSize, stageSizeSum, stageAngle, sc->stageRadix[i], sc->stageRadix[i + 1], type);
				stageSize *= sc->stageRadix[i];
				stageAngle /= sc->stageRadix[i];
			}
		}

		if (sc->convolutionStep) {
			appendCoordinateRegisterStore(output, sc, type);

			if (sc->matrixConvolution > 1)
				sprintf(output + strlen(output), "	coordinate++;}\n");

			if (sc->numKernels > 1)
				appendPreparationBatchedKernelConvolution(output, sc, floatType, floatTypeKernelMemory, uintType, type);

			appendKernelConvolution(output, sc, floatType, floatTypeKernelMemory, uintType, type);

			if (sc->matrixConvolution > 1)
				sprintf(output + strlen(output), "	for (%s coordinate=0; coordinate < %d; coordinate++){\n", uintType, sc->matrixConvolution);

			appendCoordinateRegisterPull(output, sc, type);

			stageSize = 1;
			stageSizeSum = 0;
			stageAngle = PI_const;
			for (uint32_t i = 0; i < sc->numStages; i++) {
				//appendRegisterBoostShuffle(output, sc, floatType, sc->stageRadix[i], stageAngle, type, 0);
				appendRadixStage(output, sc, floatType, uintType, stageSize, stageSizeSum, stageAngle, sc->stageRadix[i], type);
				switch (sc->stageRadix[i]) {
				case 2:
					stageSizeSum += stageSize;
					break;
				case 3:
					stageSizeSum += stageSize * 2;
					break;
				case 4:
					stageSizeSum += stageSize * 2;
					break;
				case 5:
					stageSizeSum += stageSize * 4;
					break;
				case 7:
					stageSizeSum += stageSize * 6;
					break;
				case 8:
					stageSizeSum += stageSize * 3;
					break;
				case 11:
					stageSizeSum += stageSize * 10;
					break;
				case 13:
					stageSizeSum += stageSize * 12;
					break;
				}
				if (i == sc->numStages - 1)
					appendRadixShuffle(output, sc, floatType, uintType, stageSize, stageSizeSum, stageAngle, sc->stageRadix[i], sc->stageRadix[i], type);
				else
					appendRadixShuffle(output, sc, floatType, uintType, stageSize, stageSizeSum, stageAngle, sc->stageRadix[i], sc->stageRadix[i + 1], type);
				stageSize *= sc->stageRadix[i];
				stageAngle /= sc->stageRadix[i];
			}

		}
		appendBoostThreadDataReorder(output, sc, floatType, uintType, type, 0);
		appendReorder4StepWrite(output, sc, floatType, uintType, type);
		//appendWriteSharedToRegistersVkFFT(output, sc, floatType, floatTypeOutputMemory, uintType, type);
		appendWriteDataVkFFT(output, sc, floatType, floatTypeOutputMemory, uintType, type);
		if ((sc->convolutionStep) && (sc->matrixConvolution > 1))
			VkAppendLine(sc, "	}\n");
		if ((sc->convolutionStep) && (sc->numKernels > 1))
			VkAppendLine(sc, "	}\n");
		sprintf(output + strlen(output), "}\n");
		free(sc->disableThreadsStart);
		free(sc->disableThreadsEnd);
		for (uint32_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++)
			free(sc->regIDs[i]);
		free(sc->regIDs);
		//printf("%s", output);
	}
#if(VKFFT_BACKEND==0)
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
#endif
	static inline uint32_t VkFFTScheduler(VkFFTApplication* app, VkFFTPlan* FFTPlan, uint32_t axis_id, uint32_t supportAxis) {
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
		for (uint32_t i = 2; i < 14; i++) {
			if (tempSequence % i == 0) {
				tempSequence /= i;
				multipliers[i]++;
				i--;
			}
		}
		if (tempSequence != 1) return 11;
		uint32_t nonStridedAxisId = (supportAxis) ? 1 : 0;
		uint32_t registerBoost = 1;
		for (uint32_t i = 1; i <= app->configuration.registerBoost; i++) {
			if (app->configuration.size[axis_id] % (i * i) == 0)
				registerBoost = i;
		}
		uint32_t maxSequenceLengthSharedMemory = app->configuration.sharedMemorySize / complexSize;
		uint32_t maxSingleSizeNonStrided = maxSequenceLengthSharedMemory;
		if ((axis_id == nonStridedAxisId) && (!app->configuration.performConvolution)) maxSingleSizeNonStrided *= registerBoost;
		uint32_t maxSequenceLengthSharedMemoryStrided = (app->configuration.coalescedMemory > complexSize) ? app->configuration.sharedMemorySize / (app->configuration.coalescedMemory) : app->configuration.sharedMemorySize / complexSize;
		uint32_t maxSingleSizeStrided = (!app->configuration.performConvolution) ? maxSequenceLengthSharedMemoryStrided * registerBoost : maxSequenceLengthSharedMemoryStrided;
		uint32_t numPasses = 1;
		uint32_t numPassesHalfBandwidth = 1;
		uint32_t temp;
		temp = (axis_id == nonStridedAxisId) ? ceil(app->configuration.size[axis_id] / (double)maxSingleSizeNonStrided) : ceil(app->configuration.size[axis_id] / (double)maxSingleSizeStrided);
		if (temp > 1) {//more passes than one
			for (uint32_t i = 1; i <= app->configuration.registerBoost4Step; i++) {
				if (app->configuration.size[axis_id] % (i * i) == 0) {
					registerBoost = i;
				}
			}
			if ((!app->configuration.performConvolution)) maxSingleSizeNonStrided = maxSequenceLengthSharedMemory * registerBoost;
			if ((!app->configuration.performConvolution)) maxSingleSizeStrided = maxSequenceLengthSharedMemoryStrided * registerBoost;
			temp = ((axis_id == nonStridedAxisId) && (!app->configuration.reorderFourStep)) ? app->configuration.size[axis_id] / maxSingleSizeNonStrided : app->configuration.size[axis_id] / maxSingleSizeStrided;
			if (app->configuration.reorderFourStep)
				numPasses = (uint32_t)ceil(log2(app->configuration.size[axis_id]) / log2(maxSingleSizeStrided));
			else
				numPasses += (uint32_t)ceil(log2(temp) / log2(maxSingleSizeStrided));
		}
		registerBoost = ((axis_id == nonStridedAxisId) && ((!app->configuration.reorderFourStep) || (numPasses == 1))) ? ceil(app->configuration.size[axis_id] / (float)(pow(maxSequenceLengthSharedMemoryStrided, numPasses - 1) * maxSequenceLengthSharedMemory)) : ceil(app->configuration.size[axis_id] / (float)pow(maxSequenceLengthSharedMemoryStrided, numPasses));
		uint32_t canBoost = 0;
		for (uint32_t i = registerBoost; i <= app->configuration.registerBoost; i++) {
			if (app->configuration.size[axis_id] % (i * i) == 0) {
				registerBoost = i;
				i = app->configuration.registerBoost + 1;
				canBoost = 1;
			}
		}
		if (((canBoost == 0) || (((app->configuration.size[axis_id] & (app->configuration.size[axis_id] - 1)) != 0) && (!app->configuration.registerBoostNonPow2))) && (registerBoost > 1)) {
			registerBoost = 1;
			numPasses++;
		}
		maxSingleSizeNonStrided = maxSequenceLengthSharedMemory * registerBoost;
		maxSingleSizeStrided = maxSequenceLengthSharedMemoryStrided * registerBoost;
		uint32_t maxSingleSizeStridedHalfBandwidth = maxSingleSizeStrided;
		if ((app->configuration.performHalfBandwidthBoost)) {
			maxSingleSizeStridedHalfBandwidth = (app->configuration.coalescedMemory / 2 > complexSize) ? app->configuration.sharedMemorySizePow2 / (app->configuration.coalescedMemory / 2) : app->configuration.sharedMemorySizePow2 / complexSize;
			temp = (axis_id == nonStridedAxisId) ? ceil(app->configuration.size[axis_id] / (double)maxSingleSizeNonStrided) : ceil(app->configuration.size[axis_id] / (double)maxSingleSizeStridedHalfBandwidth);
			//temp = app->configuration.size[axis_id] / maxSingleSizeNonStrided;
			if (temp > 1) {//more passes than two
				temp = (!app->configuration.reorderFourStep) ? ceil(app->configuration.size[axis_id] / (double)maxSingleSizeNonStrided) : ceil(app->configuration.size[axis_id] / (double)maxSingleSizeStridedHalfBandwidth);
				for (uint32_t i = 0; i < 5; i++) {
					temp = ceil(temp / (double)maxSingleSizeStrided);
					numPassesHalfBandwidth++;
					if (temp == 1) i = 5;
				}
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
			else maxSingleSizeStridedHalfBandwidth = maxSingleSizeStrided;
		}
		if (((uint32_t)log2(app->configuration.size[axis_id]) >= app->configuration.swapTo3Stage4Step) && (app->configuration.swapTo3Stage4Step >= 17)) numPasses = 3;//Force set to 3 stage 4 step algorithm
		uint32_t* locAxisSplit = FFTPlan->axisSplit[axis_id];
		if (numPasses == 1) {
			locAxisSplit[0] = app->configuration.size[axis_id];
		}
		if (numPasses == 2) {
			if (isPowOf2) {
				if ((axis_id == nonStridedAxisId) && (!app->configuration.reorderFourStep)) {
					uint32_t maxPow8SharedMemory = (uint32_t)pow(8, ((uint32_t)log2(maxSequenceLengthSharedMemory)) / 3);
					//unit stride
					if (app->configuration.size[axis_id] / maxPow8SharedMemory <= maxSingleSizeStrided) {
						locAxisSplit[0] = maxPow8SharedMemory;
					}
					else {
						if (app->configuration.size[axis_id] / maxSequenceLengthSharedMemory <= maxSingleSizeStrided) {
							locAxisSplit[0] = maxSequenceLengthSharedMemory;
						}
						else {
							if (app->configuration.size[axis_id] / (maxSequenceLengthSharedMemory * registerBoost) < maxSingleSizeStridedHalfBandwidth) {
								for (uint32_t i = 1; i <= (uint32_t)log2(registerBoost); i++) {
									if (app->configuration.size[axis_id] / (maxSequenceLengthSharedMemory * (uint32_t)pow(2, i)) <= maxSingleSizeStrided) {
										locAxisSplit[0] = (maxSequenceLengthSharedMemory * (uint32_t)pow(2, i));
										i = (uint32_t)log2(registerBoost) + 1;
									}
								}
							}
							else {
								locAxisSplit[0] = (maxSequenceLengthSharedMemory * registerBoost);
							}
						}
					}
				}
				else {
					uint32_t maxPow8Strided = (uint32_t)pow(8, ((uint32_t)log2(maxSingleSizeStrided)) / 3);
					//all FFTs are considered as non-unit stride
					if (app->configuration.size[axis_id] / maxPow8Strided <= maxSingleSizeStrided) {
						locAxisSplit[0] = maxPow8Strided;
					}
					else {
						if (app->configuration.size[axis_id] / maxSingleSizeStrided < maxSingleSizeStridedHalfBandwidth) {
							locAxisSplit[0] = maxSingleSizeStrided;
						}
						else {
							locAxisSplit[0] = maxSingleSizeStridedHalfBandwidth;
						}
					}
				}
				locAxisSplit[1] = app->configuration.size[axis_id] / locAxisSplit[0];
				if (locAxisSplit[1] < 64) {
					locAxisSplit[0] = (locAxisSplit[1] == 0) ? locAxisSplit[0] / (64) : locAxisSplit[0] / (64 / locAxisSplit[1]);
					locAxisSplit[1] = 64;
				}
				if (locAxisSplit[1] > locAxisSplit[0]) {
					uint32_t swap = locAxisSplit[0];
					locAxisSplit[0] = locAxisSplit[1];
					locAxisSplit[1] = swap;
				}
			}
			else {
				uint32_t successSplit = 0;
				if ((axis_id == nonStridedAxisId) && (!app->configuration.reorderFourStep)) {
					for (uint32_t i = 0; i < maxSequenceLengthSharedMemory; i++) {
						if (app->configuration.size[axis_id] % (maxSequenceLengthSharedMemory - i) == 0) {
							if (((maxSequenceLengthSharedMemory - i) <= maxSequenceLengthSharedMemory) && (app->configuration.size[axis_id] / (maxSequenceLengthSharedMemory - i) <= maxSingleSizeStrided)) {
								locAxisSplit[0] = (maxSequenceLengthSharedMemory - i);
								locAxisSplit[1] = app->configuration.size[axis_id] / (maxSequenceLengthSharedMemory - i);
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
								locAxisSplit[0] = app->configuration.size[axis_id] / (sqrtSequence - i);
								locAxisSplit[1] = sqrtSequence - i;
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
				if ((axis_id == nonStridedAxisId) && (!app->configuration.reorderFourStep)) {
					//unit stride
					uint32_t maxPow8SharedMemory = (uint32_t)pow(8, ((uint32_t)log2(maxSequenceLengthSharedMemory)) / 3);
					if (app->configuration.size[axis_id] / maxPow8SharedMemory <= maxPow8Strided * maxPow8Strided)
						locAxisSplit[0] = maxPow8SharedMemory;
					else {
						if (app->configuration.size[axis_id] / maxSequenceLengthSharedMemory <= maxSingleSizeStrided * maxSingleSizeStrided)
							locAxisSplit[0] = maxSequenceLengthSharedMemory;
						else {
							if (app->configuration.size[axis_id] / (maxSequenceLengthSharedMemory * registerBoost) <= maxSingleSizeStrided * maxSingleSizeStrided) {
								for (uint32_t i = 0; i <= (uint32_t)log2(registerBoost); i++) {
									if (app->configuration.size[axis_id] / (maxSequenceLengthSharedMemory * (uint32_t)pow(2, i)) <= maxSingleSizeStrided * maxSingleSizeStrided) {
										locAxisSplit[0] = (maxSequenceLengthSharedMemory * (uint32_t)pow(2, i));
										i = (uint32_t)log2(registerBoost) + 1;
									}
								}
							}
							else {
								locAxisSplit[0] = (maxSequenceLengthSharedMemory * registerBoost);
							}
						}
					}
				}
				else {
					//to account for TLB misses, it is best to coalesce the unit-strided stage to 128 bytes
					/*uint32_t log2axis = (uint32_t)log2(app->configuration.size[axis_id]);
					locAxisSplit[0] = (uint32_t)pow(2, (uint32_t)log2axis / 3);
					if (log2axis % 3 > 0) locAxisSplit[0] *= 2;
					locAxisSplit[1] = (uint32_t)pow(2, (uint32_t)log2axis / 3);
					if (log2axis % 3 > 1) locAxisSplit[1] *= 2;
					locAxisSplit[2] = app->configuration.size[axis_id] / locAxisSplit[0] / locAxisSplit[1];*/
					uint32_t maxSingleSizeStrided128 = app->configuration.sharedMemorySize / (128);
					uint32_t maxPow8_128 = (uint32_t)pow(8, ((uint32_t)log2(maxSingleSizeStrided128)) / 3);
					//unit stride
					if (app->configuration.size[axis_id] / maxPow8_128 <= maxPow8Strided * maxSingleSizeStrided)
						locAxisSplit[0] = maxPow8_128;
					//non-unit stride
					else {

						if ((app->configuration.size[axis_id] / (maxPow8_128 * 2) <= maxPow8Strided * maxSingleSizeStrided) && (maxPow8_128 * 2 <= maxSingleSizeStrided128)) {
							locAxisSplit[0] = maxPow8_128 * 2;
						}
						else {
							if ((app->configuration.size[axis_id] / (maxPow8_128 * 4) <= maxPow8Strided * maxSingleSizeStrided) && (maxPow8_128 * 4 <= maxSingleSizeStrided128)) {
								locAxisSplit[0] = maxPow8_128 * 4;
							}
							else {
								if (app->configuration.size[axis_id] / maxSingleSizeStrided <= maxSingleSizeStrided * maxSingleSizeStrided) {
									for (uint32_t i = 0; i <= (uint32_t)log2(maxSingleSizeStrided / maxSingleSizeStrided128); i++) {
										if (app->configuration.size[axis_id] / (maxSingleSizeStrided128 * (uint32_t)pow(2, i)) <= maxSingleSizeStrided * maxSingleSizeStrided) {
											locAxisSplit[0] = (maxSingleSizeStrided128 * (uint32_t)pow(2, i));
											i = (uint32_t)log2(maxSingleSizeStrided / maxSingleSizeStrided128) + 1;
										}
									}
								}
								else
									locAxisSplit[0] = maxSingleSizeStridedHalfBandwidth;
							}
						}
					}
				}
				if (app->configuration.size[axis_id] / locAxisSplit[0] / maxPow8Strided <= maxSingleSizeStrided) {
					locAxisSplit[1] = maxPow8Strided;
					locAxisSplit[2] = app->configuration.size[axis_id] / locAxisSplit[1] / locAxisSplit[0];
				}
				else {
					if (app->configuration.size[axis_id] / locAxisSplit[0] / maxSingleSizeStrided <= maxSingleSizeStrided) {
						locAxisSplit[1] = maxSingleSizeStrided;
						locAxisSplit[2] = app->configuration.size[axis_id] / locAxisSplit[1] / locAxisSplit[0];
					}
					else {
						locAxisSplit[1] = maxSingleSizeStridedHalfBandwidth;
						locAxisSplit[2] = app->configuration.size[axis_id] / locAxisSplit[1] / locAxisSplit[0];
					}
				}
				if (locAxisSplit[2] < 64) {
					locAxisSplit[1] = (locAxisSplit[2] == 0) ? locAxisSplit[1] / (64) : locAxisSplit[1] / (64 / locAxisSplit[2]);
					locAxisSplit[2] = 64;
				}
				if (locAxisSplit[2] > locAxisSplit[1]) {
					uint32_t swap = locAxisSplit[1];
					locAxisSplit[1] = locAxisSplit[2];
					locAxisSplit[2] = swap;
				}
			}
			else {
				uint32_t successSplit = 0;
				if ((axis_id == nonStridedAxisId) && (!app->configuration.reorderFourStep)) {
					for (uint32_t i = 0; i < maxSequenceLengthSharedMemory; i++) {
						if (app->configuration.size[axis_id] % (maxSequenceLengthSharedMemory - i) == 0) {
							uint32_t sqrt3Sequence = ceil(sqrt(app->configuration.size[axis_id] / (maxSequenceLengthSharedMemory - i)));
							for (uint32_t j = 0; j < sqrt3Sequence; j++) {
								if ((app->configuration.size[axis_id] / (maxSequenceLengthSharedMemory - i)) % (sqrt3Sequence - j) == 0) {
									if (((maxSequenceLengthSharedMemory - i) <= maxSequenceLengthSharedMemory) && (sqrt3Sequence - j <= maxSingleSizeStrided) && (app->configuration.size[axis_id] / (maxSequenceLengthSharedMemory - i) / (sqrt3Sequence - j) <= maxSingleSizeStrided)) {
										locAxisSplit[0] = (maxSequenceLengthSharedMemory - i);
										locAxisSplit[1] = sqrt3Sequence - j;
										locAxisSplit[2] = app->configuration.size[axis_id] / (maxSequenceLengthSharedMemory - i) / (sqrt3Sequence - j);
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
										locAxisSplit[0] = app->configuration.size[axis_id] / (sqrt3Sequence - i) / (sqrt2Sequence - j);
										locAxisSplit[1] = sqrt3Sequence - i;
										locAxisSplit[2] = sqrt2Sequence - j;
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
			return 11;
		}
		for (uint32_t i = 0; i < numPasses; i++) {
			if ((locAxisSplit[0] % 2 != 0) && (locAxisSplit[i] % 2 == 0)) {
				uint32_t swap = locAxisSplit[0];
				locAxisSplit[0] = locAxisSplit[i];
				locAxisSplit[i] = swap;
			}
		}
		for (uint32_t i = 0; i < numPasses; i++) {
			if ((locAxisSplit[0] % 4 != 0) && (locAxisSplit[i] % 4 == 0)) {
				uint32_t swap = locAxisSplit[0];
				locAxisSplit[0] = locAxisSplit[i];
				locAxisSplit[i] = swap;
			}
		}
		for (uint32_t i = 0; i < numPasses; i++) {
			if ((locAxisSplit[0] % 8 != 0) && (locAxisSplit[i] % 8 == 0)) {
				uint32_t swap = locAxisSplit[0];
				locAxisSplit[0] = locAxisSplit[i];
				locAxisSplit[i] = swap;
			}
		}
			FFTPlan->numAxisUploads[axis_id] = numPasses;
		for (uint32_t k = 0; k < numPasses; k++) {
			tempSequence = locAxisSplit[k];
			uint32_t loc_multipliers[20] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };//split the smaller sequence
			for (uint32_t i = 2; i < 14; i++) {
				if (tempSequence % i == 0) {
					tempSequence /= i;
					loc_multipliers[i]++;
					i--;
				}
			}
			uint32_t registers_per_thread = 8;
			uint32_t registers_per_thread_per_radix[14] = {};
			uint32_t min_registers_per_thread = 8;
			if (loc_multipliers[2] > 0) {
				if (loc_multipliers[3] > 0) {
					if (loc_multipliers[5] > 0) {
						if (loc_multipliers[7] > 0) {
							if (loc_multipliers[11] > 0) {
								if (loc_multipliers[13] > 0) {
									registers_per_thread = 15;
									registers_per_thread_per_radix[2] = 14;
									registers_per_thread_per_radix[3] = 15;
									registers_per_thread_per_radix[5] = 15;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 11;
								}
								else {
									registers_per_thread = 15;
									registers_per_thread_per_radix[2] = 14;
									registers_per_thread_per_radix[3] = 15;
									registers_per_thread_per_radix[5] = 15;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 11;
								}
							}
							else {
								if (loc_multipliers[13] > 0) {
									registers_per_thread = 15;
									registers_per_thread_per_radix[2] = 14;
									registers_per_thread_per_radix[3] = 15;
									registers_per_thread_per_radix[5] = 15;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 13;
								}
								else {
									registers_per_thread = 15;
									registers_per_thread_per_radix[2] = 14;
									registers_per_thread_per_radix[3] = 15;
									registers_per_thread_per_radix[5] = 15;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
							min_registers_per_thread = 14;
						}
							}
						}
						else {
							if ((loc_multipliers[2] == 1)) {
								if (loc_multipliers[11] > 0) {
									if (loc_multipliers[13] > 0) {
										registers_per_thread = 15;
										registers_per_thread_per_radix[2] = 10;
										registers_per_thread_per_radix[3] = 15;
										registers_per_thread_per_radix[5] = 10;
										registers_per_thread_per_radix[7] = 0;
										registers_per_thread_per_radix[11] = 11;
										registers_per_thread_per_radix[13] = 13;
										min_registers_per_thread = 10;
									}
									else {
										registers_per_thread = 15;
										registers_per_thread_per_radix[2] = 10;
										registers_per_thread_per_radix[3] = 15;
										registers_per_thread_per_radix[5] = 10;
										registers_per_thread_per_radix[7] = 0;
										registers_per_thread_per_radix[11] = 11;
										registers_per_thread_per_radix[13] = 0;
										min_registers_per_thread = 10;
									}
								}
								else {
									if (loc_multipliers[13] > 0) {
										registers_per_thread = 15;
										registers_per_thread_per_radix[2] = 10;
										registers_per_thread_per_radix[3] = 15;
										registers_per_thread_per_radix[5] = 10;
										registers_per_thread_per_radix[7] = 0;
										registers_per_thread_per_radix[11] = 0;
										registers_per_thread_per_radix[13] = 13;
										min_registers_per_thread = 10;
									}
									else {
								registers_per_thread = 6;
										registers_per_thread_per_radix[2] = 6;
										registers_per_thread_per_radix[3] = 6;
										registers_per_thread_per_radix[5] = 5;
										registers_per_thread_per_radix[7] = 0;
										registers_per_thread_per_radix[11] = 0;
										registers_per_thread_per_radix[13] = 0;
								min_registers_per_thread = 5;
							}
								}
							}
							else {
								if (loc_multipliers[11] > 0) {
									if (loc_multipliers[13] > 0) {
										registers_per_thread = 13;
										registers_per_thread_per_radix[2] = 10;
										registers_per_thread_per_radix[3] = 12;
										registers_per_thread_per_radix[5] = 10;
										registers_per_thread_per_radix[7] = 0;
										registers_per_thread_per_radix[11] = 11;
										registers_per_thread_per_radix[13] = 13;
										min_registers_per_thread = 10;
									}
									else {
								registers_per_thread = 12;
										registers_per_thread_per_radix[2] = 10;
										registers_per_thread_per_radix[3] = 12;
										registers_per_thread_per_radix[5] = 10;
										registers_per_thread_per_radix[7] = 0;
										registers_per_thread_per_radix[11] = 11;
										registers_per_thread_per_radix[13] = 0;
								min_registers_per_thread = 10;
							}
						}
								else {
									if (loc_multipliers[13] > 0) {
										registers_per_thread = 13;
										registers_per_thread_per_radix[2] = 10;
										registers_per_thread_per_radix[3] = 12;
										registers_per_thread_per_radix[5] = 10;
										registers_per_thread_per_radix[7] = 0;
										registers_per_thread_per_radix[11] = 0;
										registers_per_thread_per_radix[13] = 13;
										min_registers_per_thread = 10;
					}
									else {
										registers_per_thread = 12;
										registers_per_thread_per_radix[2] = 10;
										registers_per_thread_per_radix[3] = 12;
										registers_per_thread_per_radix[5] = 10;
										registers_per_thread_per_radix[7] = 0;
										registers_per_thread_per_radix[11] = 0;
										registers_per_thread_per_radix[13] = 0;
										min_registers_per_thread = 10;
									}
								}
							}
						}
					}
					else
					{
						if (loc_multipliers[7] > 0) {
							if ((loc_multipliers[2] == 1)) {
								if (loc_multipliers[11] > 0) {
									if (loc_multipliers[13] > 0) {
										registers_per_thread = 26;
										registers_per_thread_per_radix[2] = 22;
										registers_per_thread_per_radix[3] = 21;
										registers_per_thread_per_radix[5] = 0;
										registers_per_thread_per_radix[7] = 21;
										registers_per_thread_per_radix[11] = 22;
										registers_per_thread_per_radix[13] = 26;
										min_registers_per_thread = 21;
									}
									else {
										registers_per_thread = 22;
										registers_per_thread_per_radix[2] = 22;
										registers_per_thread_per_radix[3] = 21;
										registers_per_thread_per_radix[5] = 0;
										registers_per_thread_per_radix[7] = 21;
										registers_per_thread_per_radix[11] = 22;
										registers_per_thread_per_radix[13] = 0;
										min_registers_per_thread = 21;
									}
								}
								else {
									if (loc_multipliers[13] > 0) {
										registers_per_thread = 26;
										registers_per_thread_per_radix[2] = 26;
										registers_per_thread_per_radix[3] = 21;
										registers_per_thread_per_radix[5] = 0;
										registers_per_thread_per_radix[7] = 21;
										registers_per_thread_per_radix[11] = 0;
										registers_per_thread_per_radix[13] = 26;
										min_registers_per_thread = 21;
									}
									else {
								registers_per_thread = 7;
										registers_per_thread_per_radix[2] = 6;
										registers_per_thread_per_radix[3] = 6;
										registers_per_thread_per_radix[5] = 0;
										registers_per_thread_per_radix[7] = 7;
										registers_per_thread_per_radix[11] = 0;
										registers_per_thread_per_radix[13] = 0;
								min_registers_per_thread = 6;
							}
								}
							}
							else {
								if (loc_multipliers[11] > 0) {
									if (loc_multipliers[13] > 0) {
										registers_per_thread = 14;
										registers_per_thread_per_radix[2] = 12;
										registers_per_thread_per_radix[3] = 12;
										registers_per_thread_per_radix[5] = 0;
										registers_per_thread_per_radix[7] = 14;
										registers_per_thread_per_radix[11] = 11;
										registers_per_thread_per_radix[13] = 13;
										min_registers_per_thread = 11;
									}
									else {
										registers_per_thread = 14;
										registers_per_thread_per_radix[2] = 12;
										registers_per_thread_per_radix[3] = 12;
										registers_per_thread_per_radix[5] = 0;
										registers_per_thread_per_radix[7] = 14;
										registers_per_thread_per_radix[11] = 11;
										registers_per_thread_per_radix[13] = 0;
										min_registers_per_thread = 11;
									}
								}
								else {
									if (loc_multipliers[13] > 0) {
										registers_per_thread = 14;
										registers_per_thread_per_radix[2] = 12;
										registers_per_thread_per_radix[3] = 12;
										registers_per_thread_per_radix[5] = 0;
										registers_per_thread_per_radix[7] = 14;
										registers_per_thread_per_radix[11] = 0;
										registers_per_thread_per_radix[13] = 13;
										min_registers_per_thread = 12;
									}
							else {
								registers_per_thread = 14;
										registers_per_thread_per_radix[2] = 12;
										registers_per_thread_per_radix[3] = 12;
										registers_per_thread_per_radix[5] = 0;
										registers_per_thread_per_radix[7] = 14;
										registers_per_thread_per_radix[11] = 0;
										registers_per_thread_per_radix[13] = 0;
								min_registers_per_thread = 12;
							}
						}
							}
						}
						else {
							if ((loc_multipliers[2] == 1)) {
								if (loc_multipliers[11] > 0) {
									if (loc_multipliers[13] > 0) {
										registers_per_thread = 13;
										registers_per_thread_per_radix[2] = 6;
										registers_per_thread_per_radix[3] = 6;
										registers_per_thread_per_radix[5] = 0;
										registers_per_thread_per_radix[7] = 0;
										registers_per_thread_per_radix[11] = 11;
										registers_per_thread_per_radix[13] = 13;
										min_registers_per_thread = 6;
									}
									else {
										registers_per_thread = 11;
										registers_per_thread_per_radix[2] = 6;
										registers_per_thread_per_radix[3] = 6;
										registers_per_thread_per_radix[5] = 0;
										registers_per_thread_per_radix[7] = 0;
										registers_per_thread_per_radix[11] = 11;
										registers_per_thread_per_radix[13] = 0;
								min_registers_per_thread = 6;
							}
								}
							else {
									if (loc_multipliers[13] > 0) {
										registers_per_thread = 13;
										registers_per_thread_per_radix[2] = 6;
										registers_per_thread_per_radix[3] = 6;
										registers_per_thread_per_radix[5] = 0;
										registers_per_thread_per_radix[7] = 0;
										registers_per_thread_per_radix[11] = 0;
										registers_per_thread_per_radix[13] = 13;
										min_registers_per_thread = 6;
							}
									else {
										registers_per_thread = 6;
										registers_per_thread_per_radix[2] = 6;
										registers_per_thread_per_radix[3] = 6;
										registers_per_thread_per_radix[5] = 0;
										registers_per_thread_per_radix[7] = 0;
										registers_per_thread_per_radix[11] = 0;
										registers_per_thread_per_radix[13] = 0;
										min_registers_per_thread = 6;
						}
					}
				}
				else {
								if (loc_multipliers[11] > 0) {
									if (loc_multipliers[13] > 0) {
										registers_per_thread = 13;
										registers_per_thread_per_radix[2] = 12;
										registers_per_thread_per_radix[3] = 12;
										registers_per_thread_per_radix[5] = 0;
										registers_per_thread_per_radix[7] = 0;
										registers_per_thread_per_radix[11] = 11;
										registers_per_thread_per_radix[13] = 13;
										min_registers_per_thread = 11;
						}
						else {
										registers_per_thread = 12;
										registers_per_thread_per_radix[2] = 12;
										registers_per_thread_per_radix[3] = 12;
										registers_per_thread_per_radix[5] = 0;
										registers_per_thread_per_radix[7] = 0;
										registers_per_thread_per_radix[11] = 11;
										registers_per_thread_per_radix[13] = 0;
										min_registers_per_thread = 11;
						}
					}
								else {
									if (loc_multipliers[13] > 0) {
										registers_per_thread = 13;
										registers_per_thread_per_radix[2] = 12;
										registers_per_thread_per_radix[3] = 12;
										registers_per_thread_per_radix[5] = 0;
										registers_per_thread_per_radix[7] = 0;
										registers_per_thread_per_radix[11] = 0;
										registers_per_thread_per_radix[13] = 13;
										min_registers_per_thread = 12;
						}
						else {
										registers_per_thread = 12;
										registers_per_thread_per_radix[2] = 12;
										registers_per_thread_per_radix[3] = 12;
										registers_per_thread_per_radix[5] = 0;
										registers_per_thread_per_radix[7] = 0;
										registers_per_thread_per_radix[11] = 0;
										registers_per_thread_per_radix[13] = 0;
										min_registers_per_thread = 12;
						}
					}
				}
			}
					}
				}
			else {
					if (loc_multipliers[5] > 0) {
						if (loc_multipliers[7] > 0) {
							if (loc_multipliers[11] > 0) {
								if (loc_multipliers[13] > 0) {
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 10;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 10;
						}
						else {
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 10;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 10;
						}
					}
							else {
								if (loc_multipliers[13] > 0) {
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 10;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 10;
							}
							else {
									registers_per_thread = 10;
									registers_per_thread_per_radix[2] = 10;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 7;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
								min_registers_per_thread = 7;
							}
						}
						}
						else {
							if (loc_multipliers[11] > 0) {
								if (loc_multipliers[13] > 0) {
									registers_per_thread = 13;
									registers_per_thread_per_radix[2] = 10;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 10;
							}
							else {
									registers_per_thread = 11;
									registers_per_thread_per_radix[2] = 10;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 10;
					}
				}
				else {
								if (loc_multipliers[13] > 0) {
									registers_per_thread = 13;
									registers_per_thread_per_radix[2] = 10;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 10;
						}
						else {
									registers_per_thread = 10;
									registers_per_thread_per_radix[2] = 10;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 10;
								}
							}
						}
					}
					else
					{
						if (loc_multipliers[7] > 0) {
							if (loc_multipliers[11] > 0) {
								if (loc_multipliers[13] > 0) {
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 14;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 11;
						}
								else {
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 14;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 11;
					}
				}
							else {
								if (loc_multipliers[13] > 0) {
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 14;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 13;
			}
								else {
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 14;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 14;
			}
			}
			}
						else {
							if (loc_multipliers[11] > 0) {
								if (loc_multipliers[13] > 0) {
									registers_per_thread = 26;
									registers_per_thread_per_radix[2] = 22;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 22;
									registers_per_thread_per_radix[13] = 26;
									min_registers_per_thread = 22;
				}
				else {
									registers_per_thread = 22;
									registers_per_thread_per_radix[2] = 22;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 22;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 22;
				}
			}
							else {
								if (loc_multipliers[13] > 0) {
									registers_per_thread = 26;
									registers_per_thread_per_radix[2] = 26;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 26;
									min_registers_per_thread = 26;
			}
								else {
									registers_per_thread = (loc_multipliers[2] > 2) ? 8 : pow(2, loc_multipliers[2]);
									registers_per_thread_per_radix[2] = (loc_multipliers[2] > 2) ? 8 : pow(2, loc_multipliers[2]);
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = (loc_multipliers[2] > 2) ? 8 : pow(2, loc_multipliers[2]);
					}
				}
					}
				}
				}
			}
			else {
				if (loc_multipliers[3] > 0) {
					if (loc_multipliers[5] > 0) {
						if (loc_multipliers[7] > 0) {
							if (loc_multipliers[11] > 0) {
								if (loc_multipliers[13] > 0) {
									registers_per_thread = 21;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 15;
									registers_per_thread_per_radix[5] = 15;
									registers_per_thread_per_radix[7] = 21;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 11;
				}
				else {
									registers_per_thread = 21;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 15;
									registers_per_thread_per_radix[5] = 15;
									registers_per_thread_per_radix[7] = 21;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 11;
						}
					}
							else {
								if (loc_multipliers[13] > 0) {
									registers_per_thread = 21;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 15;
									registers_per_thread_per_radix[5] = 15;
									registers_per_thread_per_radix[7] = 21;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 13;
				}
								else {
									registers_per_thread = 21;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 15;
									registers_per_thread_per_radix[5] = 15;
									registers_per_thread_per_radix[7] = 21;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 15;
			}
				}
			}
						else {
							if (loc_multipliers[11] > 0) {
								if (loc_multipliers[13] > 0) {
									registers_per_thread = 15;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 15;
									registers_per_thread_per_radix[5] = 15;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 11;
			}
			else {
									registers_per_thread = 15;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 15;
									registers_per_thread_per_radix[5] = 15;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 11;
				}
			}
							else {
								if (loc_multipliers[13] > 0) {
									registers_per_thread = 15;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 15;
									registers_per_thread_per_radix[5] = 15;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 13;
		}
								else {
									registers_per_thread = 15;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 15;
									registers_per_thread_per_radix[5] = 15;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 15;
	}
				}
			}
					}
				else
					{
						if (loc_multipliers[7] > 0) {
							if ((loc_multipliers[3] == 1)) {
								if (loc_multipliers[11] > 0) {
									if (loc_multipliers[13] > 0) {
										registers_per_thread = 21;
										registers_per_thread_per_radix[2] = 0;
										registers_per_thread_per_radix[3] = 21;
										registers_per_thread_per_radix[5] = 0;
										registers_per_thread_per_radix[7] = 21;
										registers_per_thread_per_radix[11] = 11;
										registers_per_thread_per_radix[13] = 13;
										min_registers_per_thread = 11;
							}
									else {
										registers_per_thread = 21;
										registers_per_thread_per_radix[2] = 0;
										registers_per_thread_per_radix[3] = 21;
										registers_per_thread_per_radix[5] = 0;
										registers_per_thread_per_radix[7] = 21;
										registers_per_thread_per_radix[11] = 11;
										registers_per_thread_per_radix[13] = 0;
										min_registers_per_thread = 11;
						}
					}
					else {
									if (loc_multipliers[13] > 0) {
										registers_per_thread = 21;
										registers_per_thread_per_radix[2] = 0;
										registers_per_thread_per_radix[3] = 21;
										registers_per_thread_per_radix[5] = 0;
										registers_per_thread_per_radix[7] = 21;
										registers_per_thread_per_radix[11] = 0;
										registers_per_thread_per_radix[13] = 13;
										min_registers_per_thread = 13;
							}
									else {
										registers_per_thread = 21;
										registers_per_thread_per_radix[2] = 0;
										registers_per_thread_per_radix[3] = 21;
										registers_per_thread_per_radix[5] = 0;
										registers_per_thread_per_radix[7] = 21;
										registers_per_thread_per_radix[11] = 0;
										registers_per_thread_per_radix[13] = 0;
										min_registers_per_thread = 21;
						}
					}
				}
							else {
								if (loc_multipliers[11] > 0) {
									if (loc_multipliers[13] > 0) {
										registers_per_thread = 13;
										registers_per_thread_per_radix[2] = 0;
										registers_per_thread_per_radix[3] = 9;
										registers_per_thread_per_radix[5] = 0;
										registers_per_thread_per_radix[7] = 7;
										registers_per_thread_per_radix[11] = 11;
										registers_per_thread_per_radix[13] = 13;
										min_registers_per_thread = 7;
						}
									else {
										registers_per_thread = 11;
										registers_per_thread_per_radix[2] = 0;
										registers_per_thread_per_radix[3] = 9;
										registers_per_thread_per_radix[5] = 0;
										registers_per_thread_per_radix[7] = 7;
										registers_per_thread_per_radix[11] = 11;
										registers_per_thread_per_radix[13] = 0;
										min_registers_per_thread = 7;
					}
				}
				else {
									if (loc_multipliers[13] > 0) {
										registers_per_thread = 13;
										registers_per_thread_per_radix[2] = 0;
										registers_per_thread_per_radix[3] = 9;
										registers_per_thread_per_radix[5] = 0;
										registers_per_thread_per_radix[7] = 7;
										registers_per_thread_per_radix[11] = 0;
										registers_per_thread_per_radix[13] = 13;
										min_registers_per_thread = 7;
					}
					else {
										registers_per_thread = 9;
										registers_per_thread_per_radix[2] = 0;
										registers_per_thread_per_radix[3] = 9;
										registers_per_thread_per_radix[5] = 0;
										registers_per_thread_per_radix[7] = 7;
										registers_per_thread_per_radix[11] = 0;
										registers_per_thread_per_radix[13] = 0;
										min_registers_per_thread = 7;
					}
				}
			}
						}
			else {
							if ((loc_multipliers[3] == 1)) {
								if (loc_multipliers[11] > 0) {
									if (loc_multipliers[13] > 0) {
										registers_per_thread = 39;
										registers_per_thread_per_radix[2] = 0;
										registers_per_thread_per_radix[3] = 33;
										registers_per_thread_per_radix[5] = 0;
										registers_per_thread_per_radix[7] = 0;
										registers_per_thread_per_radix[11] = 33;
										registers_per_thread_per_radix[13] = 39;
										min_registers_per_thread = 33;
							}
									else {
										registers_per_thread = 33;
										registers_per_thread_per_radix[2] = 0;
										registers_per_thread_per_radix[3] = 33;
										registers_per_thread_per_radix[5] = 0;
										registers_per_thread_per_radix[7] = 0;
										registers_per_thread_per_radix[11] = 33;
										registers_per_thread_per_radix[13] = 0;
										min_registers_per_thread = 33;
						}
					}
					else {
									if (loc_multipliers[13] > 0) {
										registers_per_thread = 39;
										registers_per_thread_per_radix[2] = 0;
										registers_per_thread_per_radix[3] = 39;
										registers_per_thread_per_radix[5] = 0;
										registers_per_thread_per_radix[7] = 0;
										registers_per_thread_per_radix[11] = 0;
										registers_per_thread_per_radix[13] = 39;
										min_registers_per_thread = 39;
							}
									else {
										registers_per_thread = 3;
										registers_per_thread_per_radix[2] = 0;
										registers_per_thread_per_radix[3] = 3;
										registers_per_thread_per_radix[5] = 0;
										registers_per_thread_per_radix[7] = 0;
										registers_per_thread_per_radix[11] = 0;
										registers_per_thread_per_radix[13] = 0;
										min_registers_per_thread = 3;
						}
					}
				}
							else {
								if (loc_multipliers[11] > 0) {
									if (loc_multipliers[13] > 0) {
										registers_per_thread = 13;
										registers_per_thread_per_radix[2] = 0;
										registers_per_thread_per_radix[3] = 9;
										registers_per_thread_per_radix[5] = 0;
										registers_per_thread_per_radix[7] = 0;
										registers_per_thread_per_radix[11] = 11;
										registers_per_thread_per_radix[13] = 13;
										min_registers_per_thread = 9;
						}
									else {
										registers_per_thread = 11;
										registers_per_thread_per_radix[2] = 0;
										registers_per_thread_per_radix[3] = 9;
										registers_per_thread_per_radix[5] = 0;
										registers_per_thread_per_radix[7] = 0;
										registers_per_thread_per_radix[11] = 11;
										registers_per_thread_per_radix[13] = 0;
										min_registers_per_thread = 9;
					}
				}
				else {
									if (loc_multipliers[13] > 0) {
										registers_per_thread = 13;
										registers_per_thread_per_radix[2] = 0;
										registers_per_thread_per_radix[3] = 9;
										registers_per_thread_per_radix[5] = 0;
										registers_per_thread_per_radix[7] = 0;
										registers_per_thread_per_radix[11] = 0;
										registers_per_thread_per_radix[13] = 13;
										min_registers_per_thread = 9;
					}
					else {
										registers_per_thread = 9;
										registers_per_thread_per_radix[2] = 0;
										registers_per_thread_per_radix[3] = 9;
										registers_per_thread_per_radix[5] = 0;
										registers_per_thread_per_radix[7] = 0;
										registers_per_thread_per_radix[11] = 0;
										registers_per_thread_per_radix[13] = 0;
										min_registers_per_thread = 9;
					}
				}
			}
		}
		}
		}
				else {
					if (loc_multipliers[5] > 0) {
						if (loc_multipliers[7] > 0) {
							if (loc_multipliers[11] > 0) {
								if (loc_multipliers[13] > 0) {
									registers_per_thread = 13;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 5;
									registers_per_thread_per_radix[7] = 7;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 5;
		}
								else {
									registers_per_thread = 11;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 5;
									registers_per_thread_per_radix[7] = 7;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 5;
			}
		}
							else {
								if (loc_multipliers[13] > 0) {
									registers_per_thread = 13;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 5;
									registers_per_thread_per_radix[7] = 7;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 5;
			}
								else {
									registers_per_thread = 7;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 5;
									registers_per_thread_per_radix[7] = 7;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 5;
		}
			}
		}
						else {
							if (loc_multipliers[11] > 0) {
								if (loc_multipliers[13] > 0) {
									registers_per_thread = 13;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 5;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 5;
		}
								else {
									registers_per_thread = 11;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 5;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 5;
		}
			}
							else {
								if (loc_multipliers[13] > 0) {
									registers_per_thread = 13;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 5;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 5;
		}
		else {
									registers_per_thread = 5;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 5;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 5;
		}
		}
		}
		}
					else
					{
						if (loc_multipliers[7] > 0) {
							if (loc_multipliers[11] > 0) {
								if (loc_multipliers[13] > 0) {
									registers_per_thread = 13;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 7;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 7;
		}
								else {
									registers_per_thread = 11;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 7;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 7;
		}
						}
						else {
								if (loc_multipliers[13] > 0) {
									registers_per_thread = 13;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 7;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 7;
						}
								else {
									registers_per_thread = 7;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 7;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 7;
					}
				}
						}
						else {
							if (loc_multipliers[11] > 0) {
								if (loc_multipliers[13] > 0) {
									registers_per_thread = 13;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 11;
						}
								else {
									registers_per_thread = 11;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 11;
					}
				}
							else {
								if (loc_multipliers[13] > 0) {
									registers_per_thread = 13;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 13;
						}
						else {
									return 11;
						}
					}
				}
				}
			}

					}
			registers_per_thread_per_radix[8] = registers_per_thread_per_radix[2];
			registers_per_thread_per_radix[4] = registers_per_thread_per_radix[2];
			if ((registerBoost == 4) && (registers_per_thread % 4 != 0)) {
				registers_per_thread *= 2;
				for (uint32_t i = 2; i < 14; i++) {
					registers_per_thread_per_radix[i] *= 2;
				}
				min_registers_per_thread *= 2;
			}
			if (registers_per_thread_per_radix[8] % 8 == 0) {
				loc_multipliers[8] = loc_multipliers[2] / 3;
				loc_multipliers[2] = loc_multipliers[2] - loc_multipliers[8] * 3;
			}
			if (registers_per_thread_per_radix[4] % 4 == 0) {
				loc_multipliers[4] = loc_multipliers[2] / 2;
				loc_multipliers[2] = loc_multipliers[2] - loc_multipliers[4] * 2;
			}
			if ((registerBoost == 2) && (loc_multipliers[2] == 0)) {
				if (loc_multipliers[4] > 0) {
					loc_multipliers[4]--;
					loc_multipliers[2] = 2;
				}
				else {
					loc_multipliers[8]--;
					loc_multipliers[4]++;
					loc_multipliers[2]++;
				}
			}
			if ((registerBoost == 4) && (loc_multipliers[4] == 0)) {
				loc_multipliers[8]--;
				loc_multipliers[4]++;
				loc_multipliers[2]++;
			}
			uint32_t maxBatchCoalesced = ((axis_id == 0) && (((k == 0) && (!app->configuration.reorderFourStep)) || (numPasses == 1))) ? 1 : app->configuration.coalescedMemory / complexSize;
			if (maxBatchCoalesced * locAxisSplit[k] / (min_registers_per_thread * registerBoost) > app->configuration.maxThreadsNum)
			{
				for (uint32_t i = 2; i < 14; i++) {
					if (locAxisSplit[k] / (min_registers_per_thread * registerBoost) % i == 0) {
						min_registers_per_thread *= i;
						i = 14;
			}
			}
				for (uint32_t i = 2; i < 14; i++) {
					for (uint32_t j = 2; j < 14; j++) {
						if (locAxisSplit[k] / (registers_per_thread_per_radix[i] * registerBoost) % j == 0) {
							registers_per_thread_per_radix[i] *= j;
							j = 14;
			}
			}
					registers_per_thread_per_radix[i] *= 2;
				}
				for (uint32_t i = 2; i < 14; i++) {
					if (locAxisSplit[k] / (registers_per_thread * registerBoost) % i == 0) {
						registers_per_thread *= i;
						i = 14;
				}
			}
				if (min_registers_per_thread > registers_per_thread) {
					uint32_t temp = min_registers_per_thread;
					min_registers_per_thread = registers_per_thread;
					registers_per_thread = temp;
				}
				for (uint32_t i = 2; i < 14; i++) {
					if (registers_per_thread_per_radix[i] > registers_per_thread) {
						registers_per_thread = registers_per_thread_per_radix[i];
				}
					if (registers_per_thread_per_radix[i] < min_registers_per_thread) {
						min_registers_per_thread = registers_per_thread_per_radix[i];
			}
			}
			}
			uint32_t j = 0;
			VkFFTAxis* axes = FFTPlan->axes[axis_id];
			axes[k].specializationConstants.registerBoost = registerBoost;
			axes[k].specializationConstants.registers_per_thread = registers_per_thread;
			axes[k].specializationConstants.min_registers_per_thread = min_registers_per_thread;
			for (uint32_t i = 2; i < 14; i++) {
				axes[k].specializationConstants.registers_per_thread_per_radix[i] = registers_per_thread_per_radix[i];
					}
			axes[k].specializationConstants.numStages = 0;
			axes[k].specializationConstants.fftDim = locAxisSplit[k];
			uint32_t tempRegisterBoost = registerBoost;// ((axis_id == nonStridedAxisId) && (!app->configuration.reorderFourStep)) ? ceil(axes[k].specializationConstants.fftDim / (float)maxSingleSizeNonStrided) : ceil(axes[k].specializationConstants.fftDim / (float)maxSingleSizeStrided);
			uint32_t switchRegisterBoost = 0;
			if (tempRegisterBoost > 1) {
				if (loc_multipliers[tempRegisterBoost] > 0) {
					loc_multipliers[tempRegisterBoost]--;
					switchRegisterBoost = tempRegisterBoost;
				}
				else {
					for (uint32_t i = 14; i > 1; i--) {
						if (loc_multipliers[i] > 0) {
							loc_multipliers[i]--;
							switchRegisterBoost = i;
							i = 1;
				}
			}
			}
			}
			for (uint32_t i = 14; i > 1; i--) {
				if (loc_multipliers[i] > 0) {
					axes[k].specializationConstants.stageRadix[j] = i;
					loc_multipliers[i]--;
					i++;
					j++;
					axes[k].specializationConstants.numStages++;
			}
			}
			if (switchRegisterBoost > 0) {
				axes[k].specializationConstants.stageRadix[axes[k].specializationConstants.numStages] = switchRegisterBoost;
				axes[k].specializationConstants.numStages++;
			}
			else {
				if (min_registers_per_thread != registers_per_thread) {
					j = axes[k].specializationConstants.stageRadix[axes[k].specializationConstants.numStages - 1];
					axes[k].specializationConstants.stageRadix[axes[k].specializationConstants.numStages - 1] = axes[k].specializationConstants.stageRadix[0];
					axes[k].specializationConstants.stageRadix[0] = j;
			}
			}
				}
		return 0;
	}
	static inline uint32_t VkFFTPlanAxis(VkFFTApplication* app, VkFFTPlan* FFTPlan, uint32_t axis_id, uint32_t axis_upload_id, uint32_t inverse) {
		//get radix stages
		VkFFTAxis* axis = &FFTPlan->axes[axis_id][axis_upload_id];
		axis->specializationConstants.warpSize = app->configuration.warpSize;
		axis->specializationConstants.numSharedBanks = app->configuration.numSharedBanks;
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
				axis->specializationConstants.fft_dim_x = app->configuration.size[0] / 2 + 1;
			else
				axis->specializationConstants.fft_dim_x = app->configuration.size[0];
		}
		//axis->specializationConstants.registerBoost = 1;
		if ((axis_id == 0) && ((FFTPlan->numAxisUploads[axis_id] == 1) || ((axis_upload_id == 0) && (!app->configuration.reorderFourStep)))) {
			//if ((axis->specializationConstants.fftDim > maxSequenceLengthSharedMemory) && (axis->specializationConstants.fftDim <= 2 * maxSequenceLengthSharedMemory) && (app->configuration.registerBoost >= 2)) axis->specializationConstants.registerBoost = 2;
			//if ((axis->specializationConstants.fftDim > 2 * maxSequenceLengthSharedMemory) && (axis->specializationConstants.fftDim <= 3 * maxSequenceLengthSharedMemory) && (app->configuration.registerBoost >= 3)) axis->specializationConstants.registerBoost = 3;
			//if ((axis->specializationConstants.fftDim > 3 * maxSequenceLengthSharedMemory) && (axis->specializationConstants.fftDim <= 4 * maxSequenceLengthSharedMemory) && (app->configuration.registerBoost >= 4)) axis->specializationConstants.registerBoost = 4;
			maxSequenceLengthSharedMemory *= axis->specializationConstants.registerBoost;
			maxSequenceLengthSharedMemoryPow2 = pow(2, (uint32_t)log2(maxSequenceLengthSharedMemory));
		}
		else {
			/*if ((FFTPlan->numAxisUploads[axis_id] == 1)) {
				if ((axis->specializationConstants.fftDim > maxSingleSizeStrided) && (axis->specializationConstants.fftDim <= 2 * maxSingleSizeStrided) && (app->configuration.registerBoost >= 2)) axis->specializationConstants.registerBoost = 2;
				if ((axis->specializationConstants.fftDim > 2 * maxSingleSizeStrided) && (axis->specializationConstants.fftDim <= 3 * maxSingleSizeStrided) && (app->configuration.registerBoost >= 3)) axis->specializationConstants.registerBoost = 3;
				if ((axis->specializationConstants.fftDim > 3 * maxSingleSizeStrided) && (axis->specializationConstants.fftDim <= 4 * maxSingleSizeStrided) && (app->configuration.registerBoost >= 4)) axis->specializationConstants.registerBoost = 4;
			}
			else {
				if ((axis->specializationConstants.fftDim > maxSingleSizeStrided) && (axis->specializationConstants.fftDim <= 2 * maxSingleSizeStrided) && (app->configuration.registerBoost4Step >= 2)) axis->specializationConstants.registerBoost = 2;
				if ((axis->specializationConstants.fftDim > 2 * maxSingleSizeStrided) && (axis->specializationConstants.fftDim <= 3 * maxSingleSizeStrided) && (app->configuration.registerBoost4Step >= 3)) axis->specializationConstants.registerBoost = 3;
				if ((axis->specializationConstants.fftDim > 3 * maxSingleSizeStrided) && (axis->specializationConstants.fftDim <= 4 * maxSingleSizeStrided) && (app->configuration.registerBoost4Step >= 4)) axis->specializationConstants.registerBoost = 4;
			}*/
			maxSingleSizeStrided *= axis->specializationConstants.registerBoost;
			maxSingleSizeStridedPow2 = pow(2, (uint32_t)log2(maxSingleSizeStrided));
		}
		axis->specializationConstants.performR2C = app->configuration.performR2C;
		axis->specializationConstants.reorderFourStep = (FFTPlan->numAxisUploads[axis_id] > 1) ? app->configuration.reorderFourStep : 0;
		uint32_t passID = FFTPlan->numAxisUploads[axis_id] - 1 - axis_upload_id;
		axis->specializationConstants.fft_dim_full = app->configuration.size[axis_id];
		if ((FFTPlan->numAxisUploads[axis_id] > 1) && (app->configuration.reorderFourStep) && (!app->configuration.userTempBuffer) && (app->configuration.allocateTempBuffer == 0)) {
			app->configuration.allocateTempBuffer = 1;

#if(VKFFT_BACKEND==0)
			app->configuration.tempBuffer = (VkBuffer*)malloc(sizeof(VkBuffer));
			allocateFFTBuffer(app, app->configuration.tempBuffer, &app->configuration.tempBufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, app->configuration.tempBufferSize[0]);
#elif(VKFFT_BACKEND==1)
			app->configuration.tempBuffer = (void**)malloc(sizeof(void*));
			cudaMalloc(app->configuration.tempBuffer, app->configuration.tempBufferSize[0]);
#elif(VKFFT_BACKEND==2)
			app->configuration.tempBuffer = (void**)malloc(sizeof(void*));
			hipMalloc(app->configuration.tempBuffer, app->configuration.tempBufferSize[0]);
#endif
		}
		//allocate LUT 
		if (app->configuration.useLUT) {
			double double_PI = 3.1415926535897932384626433832795;
			uint32_t dimMult = 1;
			uint32_t maxStageSum = 0;
			for (uint32_t i = 0; i < axis->specializationConstants.numStages; i++) {
				switch (axis->specializationConstants.stageRadix[i]) {
				case 2:
					maxStageSum += dimMult;
					break;
				case 3:
					maxStageSum += dimMult * 2;
					break;
				case 4:
					maxStageSum += dimMult * 2;
					break;
				case 5:
					maxStageSum += dimMult * 4;
					break;
				case 7:
					maxStageSum += dimMult * 6;
					break;
				case 8:
					maxStageSum += dimMult * 3;
					break;
				case 11:
					maxStageSum += dimMult * 10;
					break;
				case 13:
					maxStageSum += dimMult * 12;
					break;
				}
				dimMult *= axis->specializationConstants.stageRadix[i];
			}
			axis->specializationConstants.maxStageSumLUT = maxStageSum;
			dimMult = 1;
			if (app->configuration.doublePrecision) {
				if (axis_upload_id > 0)
					axis->bufferLUTSize = (maxStageSum + axis->specializationConstants.stageStartSize * axis->specializationConstants.fftDim) * 2 * sizeof(double);
				else
					axis->bufferLUTSize = (maxStageSum) * 2 * sizeof(double);
				double* tempLUT = (double*)malloc(axis->bufferLUTSize);
				uint32_t localStageSize = 1;
				uint32_t localStageSum = 0;
				for (uint32_t i = 0; i < axis->specializationConstants.numStages; i++) {
					if ((axis->specializationConstants.stageRadix[i] & (axis->specializationConstants.stageRadix[i] - 1)) == 0) {
						for (uint32_t k = 0; k < log2(axis->specializationConstants.stageRadix[i]); k++) {
							for (uint32_t j = 0; j < localStageSize; j++) {
								tempLUT[2 * (j + localStageSum)] = cos(j * double_PI / localStageSize / pow(2, k));
								tempLUT[2 * (j + localStageSum) + 1] = sin(j * double_PI / localStageSize / pow(2, k));
							}
							localStageSum += localStageSize;
						}
						localStageSize *= axis->specializationConstants.stageRadix[i];
					}
					else {
						for (uint32_t k = (axis->specializationConstants.stageRadix[i] - 1); k > 0; k--) {
							for (uint32_t j = 0; j < localStageSize; j++) {
								tempLUT[2 * (j + localStageSum)] = cos(j * 2.0 * k / axis->specializationConstants.stageRadix[i] * double_PI / localStageSize);
								tempLUT[2 * (j + localStageSum) + 1] = sin(j * 2.0 * k / axis->specializationConstants.stageRadix[i] * double_PI / localStageSize);
							}
							localStageSum += localStageSize;
						}
						localStageSize *= axis->specializationConstants.stageRadix[i];
					}
				}

				if (axis_upload_id > 0)
					for (uint32_t i = 0; i < axis->specializationConstants.stageStartSize; i++) {
						for (uint32_t j = 0; j < axis->specializationConstants.fftDim; j++) {
							double angle = 2 * double_PI * ((i * j) / (double)(axis->specializationConstants.stageStartSize * axis->specializationConstants.fftDim));
							tempLUT[maxStageSum * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize)] = cos(angle);
							tempLUT[maxStageSum * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize) + 1] = sin(angle);
						}
					}
				axis->referenceLUT = 0;
				if (!inverse) {
					axis->bufferLUT = app->localFFTPlan_inverse->axes[axis_id][axis_upload_id].bufferLUT;
#if(VKFFT_BACKEND==0)
					axis->bufferLUTDeviceMemory = app->localFFTPlan_inverse->axes[axis_id][axis_upload_id].bufferLUTDeviceMemory;
#endif
					axis->bufferLUTSize = app->localFFTPlan_inverse->axes[axis_id][axis_upload_id].bufferLUTSize;
					axis->referenceLUT = 1;
				}
				else {
					if (((axis_id == 1) || (axis_id == 2)) && (!((!app->configuration.reorderFourStep) && (FFTPlan->numAxisUploads[axis_id] > 1))) && ((axis->specializationConstants.fft_dim_full == FFTPlan->axes[0][0].specializationConstants.fft_dim_full) && (FFTPlan->numAxisUploads[axis_id] == 1) && (axis->specializationConstants.fft_dim_full < maxSingleSizeStrided / axis->specializationConstants.registerBoost))) {
						axis->bufferLUT = FFTPlan->axes[0][axis_upload_id].bufferLUT;
#if(VKFFT_BACKEND==0)
						axis->bufferLUTDeviceMemory = FFTPlan->axes[0][axis_upload_id].bufferLUTDeviceMemory;
#endif
						axis->bufferLUTSize = FFTPlan->axes[0][axis_upload_id].bufferLUTSize;
						axis->referenceLUT = 1;
					}
					else {
						if ((axis_id == 2) && (axis->specializationConstants.fft_dim_full == FFTPlan->axes[1][0].specializationConstants.fft_dim_full)) {
							axis->bufferLUT = FFTPlan->axes[1][axis_upload_id].bufferLUT;
#if(VKFFT_BACKEND==0)
							axis->bufferLUTDeviceMemory = FFTPlan->axes[1][axis_upload_id].bufferLUTDeviceMemory;
#endif
							axis->bufferLUTSize = FFTPlan->axes[1][axis_upload_id].bufferLUTSize;
							axis->referenceLUT = 1;
						}
						else {
#if(VKFFT_BACKEND==0)
							allocateFFTBuffer(app, &axis->bufferLUT, &axis->bufferLUTDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, axis->bufferLUTSize);
							transferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
#elif(VKFFT_BACKEND==1)
							cudaMalloc((void**)&axis->bufferLUT, axis->bufferLUTSize);
							cudaMemcpy(axis->bufferLUT, tempLUT, axis->bufferLUTSize, cudaMemcpyHostToDevice);
#elif(VKFFT_BACKEND==2)
							hipMalloc((void**)&axis->bufferLUT, axis->bufferLUTSize);
							hipMemcpy(axis->bufferLUT, tempLUT, axis->bufferLUTSize, hipMemcpyHostToDevice);
#endif
						}
					}
				}
				free(tempLUT);
			}
			else {
				if (axis_upload_id > 0)
					axis->bufferLUTSize = (maxStageSum + axis->specializationConstants.stageStartSize * axis->specializationConstants.fftDim) * 2 * sizeof(float);
				else
					axis->bufferLUTSize = (maxStageSum) * 2 * sizeof(float);
				float* tempLUT = (float*)malloc(axis->bufferLUTSize);
				uint32_t localStageSize = 1;
				uint32_t localStageSum = 0;
				for (uint32_t i = 0; i < axis->specializationConstants.numStages; i++) {
					if ((axis->specializationConstants.stageRadix[i] & (axis->specializationConstants.stageRadix[i] - 1)) == 0) {
						for (uint32_t k = 0; k < log2(axis->specializationConstants.stageRadix[i]); k++) {
							for (uint32_t j = 0; j < localStageSize; j++) {
								tempLUT[2 * (j + localStageSum)] = (float)cos(j * double_PI / localStageSize / pow(2, k));
								tempLUT[2 * (j + localStageSum) + 1] = (float)sin(j * double_PI / localStageSize / pow(2, k));
							}
							localStageSum += localStageSize;
						}
						localStageSize *= axis->specializationConstants.stageRadix[i];
					}
					else {
						for (uint32_t k = (axis->specializationConstants.stageRadix[i] - 1); k > 0; k--) {
							for (uint32_t j = 0; j < localStageSize; j++) {
								tempLUT[2 * (j + localStageSum)] = (float)cos(j * 2.0 * k / axis->specializationConstants.stageRadix[i] * double_PI / localStageSize);
								tempLUT[2 * (j + localStageSum) + 1] = (float)sin(j * 2.0 * k / axis->specializationConstants.stageRadix[i] * double_PI / localStageSize);
							}
							localStageSum += localStageSize;
						}
						localStageSize *= axis->specializationConstants.stageRadix[i];
					}
				}

				if (axis_upload_id > 0)
					for (uint32_t i = 0; i < axis->specializationConstants.stageStartSize; i++) {
						for (uint32_t j = 0; j < axis->specializationConstants.fftDim; j++) {
							double angle = 2 * double_PI * ((i * j) / (double)(axis->specializationConstants.stageStartSize * axis->specializationConstants.fftDim));
							tempLUT[maxStageSum * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize)] = (float)cos(angle);
							tempLUT[maxStageSum * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize) + 1] = (float)sin(angle);
						}
					}
				axis->referenceLUT = 0;
				if (!inverse) {
					axis->bufferLUT = app->localFFTPlan_inverse->axes[axis_id][axis_upload_id].bufferLUT;
#if(VKFFT_BACKEND==0)
					axis->bufferLUTDeviceMemory = app->localFFTPlan_inverse->axes[axis_id][axis_upload_id].bufferLUTDeviceMemory;
#endif
					axis->bufferLUTSize = app->localFFTPlan_inverse->axes[axis_id][axis_upload_id].bufferLUTSize;
					axis->referenceLUT = 1;
				}
				else {
					if (((axis_id == 1) || (axis_id == 2)) && (!((!app->configuration.reorderFourStep) && (FFTPlan->numAxisUploads[axis_id] > 1))) && ((axis->specializationConstants.fft_dim_full == FFTPlan->axes[0][0].specializationConstants.fft_dim_full) && (FFTPlan->numAxisUploads[axis_id] == 1) && (axis->specializationConstants.fft_dim_full < maxSingleSizeStrided / axis->specializationConstants.registerBoost))) {
						axis->bufferLUT = FFTPlan->axes[0][axis_upload_id].bufferLUT;
#if(VKFFT_BACKEND==0)
						axis->bufferLUTDeviceMemory = FFTPlan->axes[0][axis_upload_id].bufferLUTDeviceMemory;
#endif
						axis->bufferLUTSize = FFTPlan->axes[0][axis_upload_id].bufferLUTSize;
						axis->referenceLUT = 1;
					}
					else {
						if ((axis_id == 2) && (axis->specializationConstants.fft_dim_full == FFTPlan->axes[1][0].specializationConstants.fft_dim_full)) {
							axis->bufferLUT = FFTPlan->axes[1][axis_upload_id].bufferLUT;
#if(VKFFT_BACKEND==0)
							axis->bufferLUTDeviceMemory = FFTPlan->axes[1][axis_upload_id].bufferLUTDeviceMemory;
#endif
							axis->bufferLUTSize = FFTPlan->axes[1][axis_upload_id].bufferLUTSize;
							axis->referenceLUT = 1;
						}
						else {
#if(VKFFT_BACKEND==0)
							allocateFFTBuffer(app, &axis->bufferLUT, &axis->bufferLUTDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, axis->bufferLUTSize);
							transferDataFromCPU(app, tempLUT, &axis->bufferLUT, axis->bufferLUTSize);
#elif(VKFFT_BACKEND==1)
							cudaMalloc((void**)&axis->bufferLUT, axis->bufferLUTSize);
							cudaMemcpy(axis->bufferLUT, tempLUT, axis->bufferLUTSize, cudaMemcpyHostToDevice);
#elif(VKFFT_BACKEND==2)
							hipMalloc((void**)&axis->bufferLUT, axis->bufferLUTSize);
							hipMemcpy(axis->bufferLUT, tempLUT, axis->bufferLUTSize, hipMemcpyHostToDevice);
#endif
						}
					}
				}
				free(tempLUT);
			}
		}

		//configure strides
		uint32_t* axisStride = axis->specializationConstants.inputStride;
		uint32_t* usedStride = app->configuration.bufferStride;
		if ((!inverse) && (axis_id == 0) && (axis_upload_id == 0) && ((app->configuration.isInputFormatted) || (app->configuration.performR2C))) usedStride = app->configuration.inputBufferStride;
		if ((inverse) && (axis_id == app->configuration.FFTdim - 1) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (app->configuration.isOutputFormatted)) usedStride = app->configuration.outputBufferStride;

		axisStride[0] = 1;

		if (axis_id == 0) {
			axisStride[1] = usedStride[0];
			axisStride[2] = usedStride[1];
		}
		if (axis_id == 1)
		{
			axisStride[1] = usedStride[0];
			axisStride[2] = usedStride[1];
		}
		if (axis_id == 2)
		{
			axisStride[1] = usedStride[1];
			axisStride[2] = usedStride[0];
		}

		axisStride[3] = usedStride[2];

		axisStride[4] = axisStride[3] * app->configuration.coordinateFeatures;

		axisStride = axis->specializationConstants.outputStride;
		usedStride = app->configuration.bufferStride;
		if ((!inverse) && (axis_id == app->configuration.FFTdim - 1) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (app->configuration.isOutputFormatted)) usedStride = app->configuration.outputBufferStride;
		if ((inverse) && (axis_id == 0) && (axis_upload_id == 0) && ((app->configuration.isInputFormatted) || (app->configuration.performR2C))) usedStride = app->configuration.inputBufferStride;

		axisStride[0] = 1;

		if (axis_id == 0) {
			axisStride[1] = usedStride[0];
			axisStride[2] = usedStride[1];
		}
		if (axis_id == 1)
		{
			axisStride[1] = usedStride[0];
			axisStride[2] = usedStride[1];
		}
		if (axis_id == 2)
		{
			axisStride[1] = usedStride[1];
			axisStride[2] = usedStride[0];
		}

		axisStride[3] = usedStride[2];

		axisStride[4] = axisStride[3] * app->configuration.coordinateFeatures;

		/*axis->specializationConstants.inputStride[3] = (app->configuration.coordinateFeatures == 1) ? 0 : axis->specializationConstants.inputStride[3];
		axis->specializationConstants.outputStride[3] = (app->configuration.coordinateFeatures == 1) ? 0 : axis->specializationConstants.outputStride[3];

		axis->specializationConstants.inputStride[4] = ((app->configuration.numberBatches == 1) && (app->configuration.numberKernels == 1)) ? 0 : axis->specializationConstants.inputStride[3] * app->configuration.coordinateFeatures;
		axis->specializationConstants.outputStride[4] = ((app->configuration.numberBatches == 1) && (app->configuration.numberKernels == 1)) ? 0 : axis->specializationConstants.outputStride[3] * app->configuration.coordinateFeatures;
		*/
		axis->specializationConstants.inverse = inverse;


		axis->specializationConstants.inputOffset = 0;
		axis->specializationConstants.outputOffset = 0;

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
			if ((app->configuration.bufferStride[1] * storageComplexSize > app->configuration.devicePageSize * 1024) && (app->configuration.devicePageSize > 0)) {
				initPageSize = app->configuration.localPageSize * 1024;
			}
		}
		if (axis_id == 2) {
			if ((app->configuration.bufferStride[2] * storageComplexSize > app->configuration.devicePageSize * 1024) && (app->configuration.devicePageSize > 0)) {
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
		axis->numBindings = 2;
		uint32_t numBuffersBound[4] = { axis->specializationConstants.inputBufferBlockNum , axis->specializationConstants.outputBufferBlockNum, 0 , 0 };
#if(VKFFT_BACKEND==0)
		VkDescriptorPoolSize descriptorPoolSize = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
		descriptorPoolSize.descriptorCount = axis->specializationConstants.inputBufferBlockNum + axis->specializationConstants.outputBufferBlockNum;
#endif
		if ((axis_id == 0) && (axis_upload_id == 0) && (app->configuration.FFTdim == 1) && (app->configuration.performConvolution)) {
			numBuffersBound[axis->numBindings] = axis->specializationConstants.kernelBlockNum;
#if(VKFFT_BACKEND==0)
			descriptorPoolSize.descriptorCount += axis->specializationConstants.kernelBlockNum;
#endif
			axis->numBindings++;
		}
		if ((axis_id == 1) && (axis_upload_id == 0) && (app->configuration.FFTdim == 2) && (app->configuration.performConvolution)) {
			numBuffersBound[axis->numBindings] = axis->specializationConstants.kernelBlockNum;
#if(VKFFT_BACKEND==0)
			descriptorPoolSize.descriptorCount += axis->specializationConstants.kernelBlockNum;
#endif
			axis->numBindings++;
		}
		if ((axis_id == 2) && (axis_upload_id == 0) && (app->configuration.FFTdim == 3) && (app->configuration.performConvolution)) {
			numBuffersBound[axis->numBindings] = axis->specializationConstants.kernelBlockNum;
#if(VKFFT_BACKEND==0)
			descriptorPoolSize.descriptorCount += axis->specializationConstants.kernelBlockNum;
#endif
			axis->numBindings++;
		}
		if (app->configuration.useLUT) {
			numBuffersBound[axis->numBindings] = 1;
#if(VKFFT_BACKEND==0)
			descriptorPoolSize.descriptorCount++;
#endif
			axis->numBindings++;
		}
#if(VKFFT_BACKEND==0)
		VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
		descriptorPoolCreateInfo.poolSizeCount = 1;
		descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize;
		descriptorPoolCreateInfo.maxSets = 1;
		vkCreateDescriptorPool(app->configuration.device[0], &descriptorPoolCreateInfo, NULL, &axis->descriptorPool);

		const VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		VkDescriptorSetLayoutBinding* descriptorSetLayoutBindings;
		descriptorSetLayoutBindings = (VkDescriptorSetLayoutBinding*)malloc(axis->numBindings * sizeof(VkDescriptorSetLayoutBinding));
		for (uint32_t i = 0; i < axis->numBindings; ++i) {
			descriptorSetLayoutBindings[i].binding = i;
			descriptorSetLayoutBindings[i].descriptorType = descriptorType;
			descriptorSetLayoutBindings[i].descriptorCount = numBuffersBound[i];
			descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		}

		VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
		descriptorSetLayoutCreateInfo.bindingCount = axis->numBindings;
		descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;

		vkCreateDescriptorSetLayout(app->configuration.device[0], &descriptorSetLayoutCreateInfo, NULL, &axis->descriptorSetLayout);
		free(descriptorSetLayoutBindings);
		VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
		descriptorSetAllocateInfo.descriptorPool = axis->descriptorPool;
		descriptorSetAllocateInfo.descriptorSetCount = 1;
		descriptorSetAllocateInfo.pSetLayouts = &axis->descriptorSetLayout;
		vkAllocateDescriptorSets(app->configuration.device[0], &descriptorSetAllocateInfo, &axis->descriptorSet);
#endif
		for (uint32_t i = 0; i < axis->numBindings; ++i) {
			for (uint32_t j = 0; j < numBuffersBound[i]; ++j) {
#if(VKFFT_BACKEND==0)
				VkDescriptorBufferInfo descriptorBufferInfo = { 0 };
#endif
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
						axis->inputBuffer = app->configuration.inputBuffer;
#if(VKFFT_BACKEND==0)
						descriptorBufferInfo.buffer = app->configuration.inputBuffer[bufferId];
						descriptorBufferInfo.range = (axis->specializationConstants.inputBufferBlockSize * storageComplexSize);
						descriptorBufferInfo.offset = offset * (axis->specializationConstants.inputBufferBlockSize * storageComplexSize);
#endif

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
							axis->inputBuffer = app->configuration.outputBuffer;
#if(VKFFT_BACKEND==0)
							descriptorBufferInfo.buffer = app->configuration.outputBuffer[bufferId];
							descriptorBufferInfo.range = (axis->specializationConstants.inputBufferBlockSize * storageComplexSize);
							descriptorBufferInfo.offset = offset * (axis->specializationConstants.inputBufferBlockSize * storageComplexSize);
#endif
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
									axis->inputBuffer = app->configuration.buffer;
#if(VKFFT_BACKEND==0)
									descriptorBufferInfo.buffer = app->configuration.buffer[bufferId];
#endif
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
									axis->inputBuffer = app->configuration.tempBuffer;
#if(VKFFT_BACKEND==0)
									descriptorBufferInfo.buffer = app->configuration.tempBuffer[bufferId];
#endif
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
								axis->inputBuffer = app->configuration.buffer;
#if(VKFFT_BACKEND==0)
								descriptorBufferInfo.buffer = app->configuration.buffer[bufferId];
#endif
							}
#if(VKFFT_BACKEND==0)
							descriptorBufferInfo.range = (axis->specializationConstants.inputBufferBlockSize * storageComplexSize);
							descriptorBufferInfo.offset = offset * (axis->specializationConstants.inputBufferBlockSize * storageComplexSize);
#endif
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
						axis->outputBuffer = app->configuration.outputBuffer;
#if(VKFFT_BACKEND==0)
						descriptorBufferInfo.buffer = app->configuration.outputBuffer[bufferId];
						descriptorBufferInfo.range = (axis->specializationConstants.outputBufferBlockSize * storageComplexSize);
						descriptorBufferInfo.offset = offset * (axis->specializationConstants.outputBufferBlockSize * storageComplexSize);
#endif
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
								axis->outputBuffer = app->configuration.tempBuffer;
#if(VKFFT_BACKEND==0)
								descriptorBufferInfo.buffer = app->configuration.tempBuffer[bufferId];
#endif
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
								axis->outputBuffer = app->configuration.buffer;
#if(VKFFT_BACKEND==0)
								descriptorBufferInfo.buffer = app->configuration.buffer[bufferId];
#endif
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
							axis->outputBuffer = app->configuration.buffer;
#if(VKFFT_BACKEND==0)
							descriptorBufferInfo.buffer = app->configuration.buffer[bufferId];
#endif
						}
#if(VKFFT_BACKEND==0)
						descriptorBufferInfo.range = (axis->specializationConstants.outputBufferBlockSize * storageComplexSize);
						descriptorBufferInfo.offset = offset * (axis->specializationConstants.outputBufferBlockSize * storageComplexSize);
#endif
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
#if(VKFFT_BACKEND==0)
					descriptorBufferInfo.buffer = app->configuration.kernel[bufferId];
					descriptorBufferInfo.range = (axis->specializationConstants.kernelBlockSize * storageComplexSize);
					descriptorBufferInfo.offset = offset * (axis->specializationConstants.kernelBlockSize * storageComplexSize);
#endif
				}
				if ((i == axis->numBindings - 1) && (app->configuration.useLUT)) {
#if(VKFFT_BACKEND==0)
					descriptorBufferInfo.buffer = axis->bufferLUT;
					descriptorBufferInfo.offset = 0;
					descriptorBufferInfo.range = axis->bufferLUTSize;
#endif
				}
#if(VKFFT_BACKEND==0)
				VkWriteDescriptorSet writeDescriptorSet = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
				writeDescriptorSet.dstSet = axis->descriptorSet;
				writeDescriptorSet.dstBinding = i;
				writeDescriptorSet.dstArrayElement = j;
				writeDescriptorSet.descriptorType = descriptorType;
				writeDescriptorSet.descriptorCount = 1;
				writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
				vkUpdateDescriptorSets(app->configuration.device[0], 1, &writeDescriptorSet, 0, NULL);
#endif
			}
		}
		{
#if(VKFFT_BACKEND==0)
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
#endif
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
#if(VKFFT_BACKEND!=2)//for some reason, hip doesn't get performance increase from having variable shared memory strides.
			if ((FFTPlan->numAxisUploads[axis_id] == 2) && (axis_upload_id == 0) && (axis->specializationConstants.fftDim * maxBatchCoalesced <= maxSequenceLengthSharedMemory)) {
				axis->groupedBatch = ceil(axis->groupedBatch / 2.0);
			}
#endif
			if ((FFTPlan->numAxisUploads[axis_id] == 3) && (axis_upload_id == 0) && (axis->specializationConstants.fftDim < maxSequenceLengthSharedMemory / (2 * complexSize))) {
				axis->groupedBatch = ceil(axis->groupedBatch / 2.0);
			}
			if (axis->groupedBatch < maxBatchCoalesced) axis->groupedBatch = maxBatchCoalesced;
			axis->groupedBatch = (axis->groupedBatch / maxBatchCoalesced) * maxBatchCoalesced;
			//half bandiwdth technique
			if (!((axis_id == 0) && (FFTPlan->numAxisUploads[axis_id] == 1)) && !((axis_id == 0) && (axis_upload_id == 0) && (!app->configuration.reorderFourStep)) && (axis->specializationConstants.fftDim > maxSingleSizeStrided)) {
				axis->groupedBatch = ceil(axis->groupedBatch / 2.0);
			}

			if ((app->configuration.halfThreads) && (axis->groupedBatch * axis->specializationConstants.fftDim * complexSize >= app->configuration.sharedMemorySize))
				axis->groupedBatch = ceil(axis->groupedBatch / 2.0);
			if (axis->groupedBatch > app->configuration.warpSize) axis->groupedBatch = (axis->groupedBatch / app->configuration.warpSize) * app->configuration.warpSize;
			//if (axis->groupedBatch > maxBatchCoalesced) axis->groupedBatch = (axis->groupedBatch / maxBatchCoalesced) * maxBatchCoalesced;

			uint32_t maxThreadNum = maxSequenceLengthSharedMemory / (axis->specializationConstants.min_registers_per_thread * axis->specializationConstants.registerBoost);
			if (axis_id == 0) {

				if (axis_upload_id == 0) {
					axis->axisBlock[0] = (axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread / axis->specializationConstants.registerBoost > 1) ? axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread / axis->specializationConstants.registerBoost : 1;
					if (axis->axisBlock[0] > maxThreadNum) axis->axisBlock[0] = maxThreadNum;
					if (axis->axisBlock[0] > app->configuration.maxComputeWorkGroupSize[0]) axis->axisBlock[0] = app->configuration.maxComputeWorkGroupSize[0];
					if (app->configuration.reorderFourStep && (FFTPlan->numAxisUploads[axis_id] > 1))
						axis->axisBlock[1] = axis->groupedBatch;
					else {
						axis->axisBlock[1] = (axis->axisBlock[0] < app->configuration.warpSize) ? app->configuration.warpSize / axis->axisBlock[0] : 1;
						if (app->configuration.performR2C) {
							if (app->configuration.size[1]/2 < axis->axisBlock[1]) axis->axisBlock[1] = app->configuration.size[1]/2;
						}
						else {
						if (app->configuration.size[1] < axis->axisBlock[1]) axis->axisBlock[1] = app->configuration.size[1];
					}
					}
					if (axis->axisBlock[1] > app->configuration.maxComputeWorkGroupSize[1]) axis->axisBlock[1] = app->configuration.maxComputeWorkGroupSize[1];
					axis->axisBlock[2] = 1;
					axis->axisBlock[3] = axis->specializationConstants.fftDim;
				}
				else {
					axis->axisBlock[1] = (axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread / axis->specializationConstants.registerBoost > 1) ? axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread / axis->specializationConstants.registerBoost : 1;
					uint32_t scale = app->configuration.aimThreads / axis->axisBlock[1] / axis->groupedBatch;
					if (scale > 1) axis->groupedBatch *= scale;
					axis->axisBlock[0] = (axis->specializationConstants.stageStartSize > axis->groupedBatch) ? axis->groupedBatch : axis->specializationConstants.stageStartSize;
					if (axis->axisBlock[0] > app->configuration.maxComputeWorkGroupSize[0]) axis->axisBlock[0] = app->configuration.maxComputeWorkGroupSize[0];
					axis->axisBlock[2] = 1;
					axis->axisBlock[3] = axis->specializationConstants.fftDim;
				}

			}
			if (axis_id == 1) {

				axis->axisBlock[1] = (axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread / axis->specializationConstants.registerBoost > 1) ? axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread / axis->specializationConstants.registerBoost : 1;

				if (app->configuration.performR2C) {
					/*if (axis_upload_id == 0) {
						VkFFTScheduler(app, FFTPlan, axis_id, 1);
						for (uint32_t i = 0; i < FFTPlan->numSupportAxisUploads[0]; i++) {
							VkFFTPlanSupportAxis(app, FFTPlan, 1, i, inverse);
						}
					}*/
					axis->axisBlock[0] = (app->configuration.size[0] / 2 + 1 > axis->groupedBatch) ? axis->groupedBatch : app->configuration.size[0] / 2 + 1;
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
				if (axis->axisBlock[0] > app->configuration.maxComputeWorkGroupSize[0]) axis->axisBlock[0] = app->configuration.maxComputeWorkGroupSize[0];

				axis->axisBlock[2] = 1;
				axis->axisBlock[3] = axis->specializationConstants.fftDim;

			}
			if (axis_id == 2) {
				axis->axisBlock[1] = (axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread / axis->specializationConstants.registerBoost > 1) ? axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread / axis->specializationConstants.registerBoost : 1;

				if (app->configuration.performR2C) {
					/*if (axis_upload_id == 0) {
						VkFFTScheduler(app, FFTPlan, axis_id, 1);
						//->numSupportAxisUploads[1] = FFTPlan->numAxisUploads[2];
						for (uint32_t i = 0; i < FFTPlan->numSupportAxisUploads[1]; i++) {
							VkFFTPlanSupportAxis(app, FFTPlan, 2, i, inverse);
						}
					}*/
					axis->axisBlock[0] = (app->configuration.size[0] / 2 + 1 > axis->groupedBatch) ? axis->groupedBatch : app->configuration.size[0] / 2 + 1;
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
				if (axis->axisBlock[0] > app->configuration.maxComputeWorkGroupSize[0]) axis->axisBlock[0] = app->configuration.maxComputeWorkGroupSize[0];

				axis->axisBlock[2] = 1;
				axis->axisBlock[3] = axis->specializationConstants.fftDim;
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
				tempSize[0] = (app->configuration.performR2C == 1) ? ceil((app->configuration.size[0]/2+1) / (double)axis->axisBlock[0] * app->configuration.size[1] / (double)axis->specializationConstants.fftDim) : ceil(app->configuration.size[0] / (double)axis->axisBlock[0] * app->configuration.size[1] / (double)axis->specializationConstants.fftDim);
				tempSize[1] = 1;
				tempSize[2] = app->configuration.size[2];
				//if (app->configuration.performR2C == 1) tempSize[0] = ceil(tempSize[0] / 2.0);
				//if (app->configuration.performZeropadding[2]) tempSize[2] = ceil(tempSize[2] / 2.0);

				if (tempSize[0] > app->configuration.maxComputeWorkGroupCount[0]) axis->specializationConstants.performWorkGroupShift[0] = 1;
				else  axis->specializationConstants.performWorkGroupShift[0] = 0;
				if (tempSize[1] > app->configuration.maxComputeWorkGroupCount[1]) axis->specializationConstants.performWorkGroupShift[1] = 1;
				else  axis->specializationConstants.performWorkGroupShift[1] = 0;
				if (tempSize[2] > app->configuration.maxComputeWorkGroupCount[2]) axis->specializationConstants.performWorkGroupShift[2] = 1;
				else  axis->specializationConstants.performWorkGroupShift[2] = 0;

			}
			if (axis_id == 2) {
				tempSize[0] = (app->configuration.performR2C == 1) ? ceil((app->configuration.size[0]/2+1) / (double)axis->axisBlock[0] * app->configuration.size[2] / (double)axis->specializationConstants.fftDim) : ceil(app->configuration.size[0] / (double)axis->axisBlock[0] * app->configuration.size[2] / (double)axis->specializationConstants.fftDim);
				tempSize[1] = 1;
				tempSize[2] = app->configuration.size[1];
				//if (app->configuration.performR2C == 1) tempSize[0] = ceil(tempSize[0] / 2.0);

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
			axis->specializationConstants.normalize = app->configuration.normalize;
			axis->specializationConstants.size[0] = app->configuration.size[0];
			axis->specializationConstants.size[1] = app->configuration.size[1];
			axis->specializationConstants.size[2] = app->configuration.size[2];
			axis->specializationConstants.axis_id = axis_id;
			axis->specializationConstants.axis_upload_id = axis_upload_id;

			for (uint32_t i = 0; i < 3; i++) {
				axis->specializationConstants.frequencyZeropadding = app->configuration.frequencyZeroPadding;
				axis->specializationConstants.performZeropaddingFull[i] = app->configuration.performZeropadding[i]; // don't read if input is zeropadded (0 - off, 1 - on)
				axis->specializationConstants.fft_zeropad_left_full[i] = app->configuration.fft_zeropad_left[i];
				axis->specializationConstants.fft_zeropad_right_full[i] = app->configuration.fft_zeropad_right[i];
			}
			if ((inverse)) {
				if ((app->configuration.frequencyZeroPadding) && ((!app->configuration.reorderFourStep) && (axis_upload_id == 0)) || ((app->configuration.reorderFourStep) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1))) {
					axis->specializationConstants.zeropad[0] = app->configuration.performZeropadding[axis_id];
					axis->specializationConstants.fft_zeropad_left_read[axis_id] = app->configuration.fft_zeropad_left[axis_id];
					axis->specializationConstants.fft_zeropad_right_read[axis_id] = app->configuration.fft_zeropad_right[axis_id];
				}
				else
					axis->specializationConstants.zeropad[0] = 0;
				if ((!app->configuration.frequencyZeroPadding) && (((!app->configuration.reorderFourStep) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1)) || ((app->configuration.reorderFourStep) && (axis_upload_id == 0)))) {
					axis->specializationConstants.zeropad[1] = app->configuration.performZeropadding[axis_id];
					axis->specializationConstants.fft_zeropad_left_write[axis_id] = app->configuration.fft_zeropad_left[axis_id];
					axis->specializationConstants.fft_zeropad_right_write[axis_id] = app->configuration.fft_zeropad_right[axis_id];
				}
				else
					axis->specializationConstants.zeropad[1] = 0;
			}
			else {
				if ((!app->configuration.frequencyZeroPadding) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1)) {
					axis->specializationConstants.zeropad[0] = app->configuration.performZeropadding[axis_id];
					axis->specializationConstants.fft_zeropad_left_read[axis_id] = app->configuration.fft_zeropad_left[axis_id];
					axis->specializationConstants.fft_zeropad_right_read[axis_id] = app->configuration.fft_zeropad_right[axis_id];
				}
				else
					axis->specializationConstants.zeropad[0] = 0;
				if (((app->configuration.frequencyZeroPadding) && (axis_upload_id == 0)) || (((app->configuration.FFTdim - 1 == axis_id) && (axis_upload_id == 0) && (app->configuration.performConvolution)))) {
					axis->specializationConstants.zeropad[1] = app->configuration.performZeropadding[axis_id];
					axis->specializationConstants.fft_zeropad_left_write[axis_id] = app->configuration.fft_zeropad_left[axis_id];
					axis->specializationConstants.fft_zeropad_right_write[axis_id] = app->configuration.fft_zeropad_right[axis_id];
				}
				else
					axis->specializationConstants.zeropad[1] = 0;
			}
			if ((app->configuration.FFTdim - 1 == axis_id) && (axis_upload_id == 0) && (app->configuration.performConvolution)) {
				axis->specializationConstants.convolutionStep = 1;
			}
			else
				axis->specializationConstants.convolutionStep = 0;
			char floatTypeInputMemory[10];
			char floatTypeOutputMemory[10];
			char floatTypeKernelMemory[10];
			char floatType[10];
			axis->specializationConstants.unroll = 1;
			axis->specializationConstants.LUT = app->configuration.useLUT;
			if (app->configuration.doublePrecision) {
				sprintf(floatType, "double");
				sprintf(floatTypeInputMemory, "double");
				sprintf(floatTypeOutputMemory, "double");
				sprintf(floatTypeKernelMemory, "double");
				//axis->specializationConstants.unroll = 1;
			}
			else {
				//axis->specializationConstants.unroll = 0;
				if (app->configuration.halfPrecision) {
					sprintf(floatType, "float");
					if (app->configuration.halfPrecisionMemoryOnly) {
						//only out of place mode, input/output buffer must be different
						sprintf(floatTypeKernelMemory, "float");
						if ((axis_id == 0) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (!axis->specializationConstants.inverse))
							sprintf(floatTypeInputMemory, "half");
						else
							sprintf(floatTypeInputMemory, "float");
						if ((axis_id == 0) && (((!app->configuration.reorderFourStep) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1)) || ((app->configuration.reorderFourStep) && (axis_upload_id == 0))) && (axis->specializationConstants.inverse))
							sprintf(floatTypeOutputMemory, "half");
						else
							sprintf(floatTypeOutputMemory, "float");
					}
					else {
						sprintf(floatTypeInputMemory, "half");
						sprintf(floatTypeOutputMemory, "half");
						sprintf(floatTypeKernelMemory, "half");
					}

				}
				else {
					sprintf(floatType, "float");
					sprintf(floatTypeInputMemory, "float");
					sprintf(floatTypeOutputMemory, "float");
					sprintf(floatTypeKernelMemory, "float");
				}
			}
			char uintType[20] = "";
#if(VKFFT_BACKEND==0)
			sprintf(uintType, "uint");
#elif(VKFFT_BACKEND==1)
			sprintf(uintType, "unsigned int");
#elif(VKFFT_BACKEND==2)
			sprintf(uintType, "unsigned int");
#endif
			uint32_t LUT = app->configuration.useLUT;
			uint32_t type;
			if ((axis_id == 0) && (axis_upload_id == 0)) type = 0;
			if (axis_id != 0) type = 1;
			if ((axis_id == 0) && (axis_upload_id > 0)) type = 2;
			//if ((axis->specializationConstants.fftDim == 8 * maxSequenceLengthSharedMemory) && (app->configuration.registerBoost >= 8)) axis->specializationConstants.registerBoost = 8;
			if ((axis_id == 0) && (!axis->specializationConstants.inverse) && (app->configuration.performR2C)) type = 5;
			if ((axis_id == 0) && (axis->specializationConstants.inverse) && (app->configuration.performR2C)) type = 6;
#if(VKFFT_BACKEND==0)
			axis->specializationConstants.cacheShuffle = ((FFTPlan->numAxisUploads[axis_id] > 1) && ((axis->specializationConstants.fftDim & (axis->specializationConstants.fftDim - 1)) == 0) && (!app->configuration.doublePrecision) && ((type == 0) || (type == 5) || (type == 6))) ? 1 : 0;
#elif(VKFFT_BACKEND==1)
			axis->specializationConstants.cacheShuffle = 0;
#elif(VKFFT_BACKEND==2)
			axis->specializationConstants.cacheShuffle = 0;
#endif

			char* code0 = (char*)malloc(sizeof(char) * 1000000);
			shaderGenVkFFT(code0, &axis->specializationConstants, floatType, floatTypeInputMemory, floatTypeOutputMemory, floatTypeKernelMemory, uintType, type);
#if(VKFFT_BACKEND==0)
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
				return 3;

			}

			if (!glslang_shader_parse(shader, &input))
			{
				err = glslang_shader_get_info_log(shader);
				printf("%s\n", code0);
				printf("%s\nVkFFT shader type: %d\n", err, type);
				glslang_shader_delete(shader);
				free(code0);
				return 3;

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
				return 3;

			}

			glslang_program_SPIRV_generate(program, input.stage);

			if (glslang_program_SPIRV_get_messages(program))
			{
				printf("%s", glslang_program_SPIRV_get_messages(program));
			}

			glslang_shader_delete(shader);
			VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
			VkComputePipelineCreateInfo computePipelineCreateInfo = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
			pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			VkShaderModuleCreateInfo createInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
			createInfo.pCode = glslang_program_SPIRV_get_ptr(program);
			createInfo.codeSize = glslang_program_SPIRV_get_size(program) * sizeof(uint32_t);
			vkCreateShaderModule(app->configuration.device[0], &createInfo, NULL, &pipelineShaderStageCreateInfo.module);
			pipelineShaderStageCreateInfo.pName = "main";
			pipelineShaderStageCreateInfo.pSpecializationInfo = 0;// &specializationInfo;
			computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
			computePipelineCreateInfo.layout = axis->pipelineLayout;
			vkCreateComputePipelines(app->configuration.device[0], VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, &axis->pipeline);
			vkDestroyShaderModule(app->configuration.device[0], pipelineShaderStageCreateInfo.module, NULL);
			glslang_program_delete(program);
#elif(VKFFT_BACKEND==1)
			nvrtcProgram prog;
			/*char* includeNames = (char*)malloc(sizeof(char)*100);
			char* headers = (char*)malloc(sizeof(char) * 100);
			sprintf(headers, "C://Program Files//NVIDIA GPU Computing Toolkit//CUDA//v11.1//include//cuComplex.h");
			sprintf(includeNames, "cuComplex.h");*/
			nvrtcResult result = nvrtcCreateProgram(&prog,         // prog
				code0,         // buffer
				"VkFFT.cu",    // name
				0,             // numHeaders
				0,          // headers
				0);        // includeNames
			//free(includeNames);
			//free(headers);
			if (result != NVRTC_SUCCESS) printf("1 error: %s\n", nvrtcGetErrorString(result));
			//const char opts[20] = "--fmad=false";
			//result = nvrtcAddNameExpression(prog, "&consts");
			//if (result != NVRTC_SUCCESS) printf("1.5 error: %s\n", nvrtcGetErrorString(result));
			result = nvrtcCompileProgram(prog,  // prog
				0,     // numOptions
				NULL); // options
			if (result != NVRTC_SUCCESS) {
				printf("2 error: %s\n", nvrtcGetErrorString(result));
				char* log = (char*)malloc(sizeof(char) * 100000);
				nvrtcGetProgramLog(prog, log);
				printf("%s\n", log);
				free(log);
				printf("%s\n", code0);
			}
			size_t ptxSize;
			result = nvrtcGetPTXSize(prog, &ptxSize);
			if (result != NVRTC_SUCCESS) printf("3 error: %s\n", nvrtcGetErrorString(result));
			char* ptx = (char*)malloc(ptxSize);
			result = nvrtcGetPTX(prog, ptx);
			if (result != NVRTC_SUCCESS) printf("4 error: %s\n", nvrtcGetErrorString(result));
			//printf("%s\n", ptx);
			// Destroy the program.
			result = nvrtcDestroyProgram(&prog);
			if (result != NVRTC_SUCCESS) printf("5 error: %s\n", nvrtcGetErrorString(result));
			axis->VkFFTKernel = {};
			axis->VkFFTModule = {};
			CUresult result2 = cuModuleLoadDataEx(&axis->VkFFTModule, ptx, 0, 0, 0);

			if (result2 != CUDA_SUCCESS) {
				printf("6 error: %d\n", result2);
			}
			result2 = cuModuleGetFunction(&axis->VkFFTKernel, axis->VkFFTModule, "VkFFT_main");
			if (result2 != CUDA_SUCCESS) {
				printf("7 error: %d\n", result2);
			}
			if (axis->specializationConstants.usedSharedMemory > app->configuration.sharedMemorySizeStatic) {
				result2 = cuFuncSetCacheConfig(axis->VkFFTKernel, CU_FUNC_CACHE_PREFER_SHARED);
				if (result2 != CUDA_SUCCESS) {
					printf("7.5 error: %d\n", result2);
				}
			}
			size_t size = sizeof(VkFFTPushConstantsLayout);
			result2 = cuModuleGetGlobal(&axis->consts_addr, &size, axis->VkFFTModule, "consts");
			if (result2 != CUDA_SUCCESS) {
				printf("8 error: %d\n", result2);
			}
			free(ptx);
#elif(VKFFT_BACKEND==2)
			hiprtcProgram prog;
			/*char* includeNames = (char*)malloc(sizeof(char)*100);
			char* headers = (char*)malloc(sizeof(char) * 100);
			sprintf(headers, "C://Program Files//NVIDIA GPU Computing Toolkit//CUDA//v11.1//include//cuComplex.h");
			sprintf(includeNames, "cuComplex.h");*/
			hiprtcResult result = hiprtcCreateProgram(&prog,         // prog
				code0,         // buffer
				"VkFFT.hip",    // name
				0,             // numHeaders
				0,          // headers
				0);        // includeNames
			if (result != HIPRTC_SUCCESS) printf("1 error: %s\n", hiprtcGetErrorString(result));

			result = hiprtcAddNameExpression(prog, "&consts");
			if (result != HIPRTC_SUCCESS) printf("1.5 error: %s\n", hiprtcGetErrorString(result));

			result = hiprtcCompileProgram(prog,  // prog
				0,     // numOptions
				NULL); // options
			if (result != HIPRTC_SUCCESS) {
				printf("2 error: %s\n", hiprtcGetErrorString(result));
				char* log = (char*)malloc(sizeof(char) * 100000);
				hiprtcGetProgramLog(prog, log);
				printf("%s\n", log);
				free(log);
				printf("%s\n", code0);
			}
			size_t codeSize;
			result = hiprtcGetCodeSize(prog, &codeSize);
			if (result != HIPRTC_SUCCESS) printf("3 error: %s\n", hiprtcGetErrorString(result));
			char* code = (char*)malloc(codeSize);
			result = hiprtcGetCode(prog, code);
			if (result != HIPRTC_SUCCESS) printf("4 error: %s\n", hiprtcGetErrorString(result));
			//printf("%s\n", code);
			// Destroy the program.
			result = hiprtcDestroyProgram(&prog);
			if (result != HIPRTC_SUCCESS) printf("5 error: %s\n", hiprtcGetErrorString(result));
			axis->VkFFTKernel = {};
			axis->VkFFTModule = {};
			hipError_t result2 = hipModuleLoadDataEx(&axis->VkFFTModule, code, 0, 0, 0);

			if (result2 != hipSuccess) {
				printf("6 error: %d\n", result2);
			}
			result2 = hipModuleGetFunction(&axis->VkFFTKernel, axis->VkFFTModule, "VkFFT_main");
			if (result2 != hipSuccess) {
				printf("7 error: %d\n", result2);
			}
			if (axis->specializationConstants.usedSharedMemory > app->configuration.sharedMemorySizeStatic) {
				result2 = hipFuncSetCacheConfig(axis->VkFFTKernel, hipFuncCachePreferShared);
				if (result2 != hipSuccess) {
					printf("7.5 error: %d\n", result2);
				}
			}
			size_t size = sizeof(VkFFTPushConstantsLayout);
			result2 = hipModuleGetGlobal(&axis->consts_addr, &size, axis->VkFFTModule, "consts");
			if (result2 != hipSuccess) {
				printf("8 error: %d\n", result2);
			}

			free(code);
#endif
			free(code0);
		}

		return 0;
	}
	static inline void deleteAxis(VkFFTApplication* app, VkFFTAxis* axis) {
#if(VKFFT_BACKEND==0)
		if ((app->configuration.useLUT) && (!axis->referenceLUT)) {
			vkDestroyBuffer(app->configuration.device[0], axis->bufferLUT, NULL);
			vkFreeMemory(app->configuration.device[0], axis->bufferLUTDeviceMemory, NULL);
		}
		vkDestroyDescriptorPool(app->configuration.device[0], axis->descriptorPool, NULL);
		vkDestroyDescriptorSetLayout(app->configuration.device[0], axis->descriptorSetLayout, NULL);
		vkDestroyPipelineLayout(app->configuration.device[0], axis->pipelineLayout, NULL);
		vkDestroyPipeline(app->configuration.device[0], axis->pipeline, NULL);
#elif(VKFFT_BACKEND==1)
		if ((app->configuration.useLUT) && (!axis->referenceLUT)) {
			cudaFree(axis->bufferLUT);
		}
		cuModuleUnload(axis->VkFFTModule);
#elif(VKFFT_BACKEND==2)
		if ((app->configuration.useLUT) && (!axis->referenceLUT)) {
			hipFree(axis->bufferLUT);
		}
		hipModuleUnload(axis->VkFFTModule);
#endif
	}
	static inline uint32_t initializeVkFFT(VkFFTApplication* app, VkFFTConfiguration inputLaunchConfiguration) {
		app->configuration = {};// inputLaunchConfiguration;
		if (inputLaunchConfiguration.doublePrecision != 0)	app->configuration.doublePrecision = inputLaunchConfiguration.doublePrecision;
		if (inputLaunchConfiguration.halfPrecision != 0)	app->configuration.halfPrecision = inputLaunchConfiguration.halfPrecision;
		if (inputLaunchConfiguration.halfPrecisionMemoryOnly != 0)	app->configuration.halfPrecisionMemoryOnly = inputLaunchConfiguration.halfPrecisionMemoryOnly;
		//set device parameters:
#if(VKFFT_BACKEND==0)
		if (inputLaunchConfiguration.physicalDevice == 0) return 1001;
		app->configuration.physicalDevice = inputLaunchConfiguration.physicalDevice;
		if (inputLaunchConfiguration.device == 0) return 1002;
		app->configuration.device = inputLaunchConfiguration.device;
		if (inputLaunchConfiguration.queue == 0) return 1003;
		app->configuration.queue = inputLaunchConfiguration.queue;
		if (inputLaunchConfiguration.commandPool == 0) return 1004;
		app->configuration.commandPool = inputLaunchConfiguration.commandPool;
		if (inputLaunchConfiguration.fence == 0) return 1005;
		app->configuration.fence = inputLaunchConfiguration.fence;

		VkPhysicalDeviceProperties physicalDeviceProperties = {};
		vkGetPhysicalDeviceProperties(app->configuration.physicalDevice[0], &physicalDeviceProperties);
		if (inputLaunchConfiguration.isCompilerInitialized != 0) app->configuration.isCompilerInitialized = inputLaunchConfiguration.isCompilerInitialized;
		if (!app->configuration.isCompilerInitialized)
			glslang_initialize_process();
		app->configuration.maxThreadsNum = physicalDeviceProperties.limits.maxComputeWorkGroupInvocations;
		app->configuration.maxComputeWorkGroupCount[0] = physicalDeviceProperties.limits.maxComputeWorkGroupCount[0];
		app->configuration.maxComputeWorkGroupCount[1] = physicalDeviceProperties.limits.maxComputeWorkGroupCount[1];
		app->configuration.maxComputeWorkGroupCount[2] = physicalDeviceProperties.limits.maxComputeWorkGroupCount[2];
		app->configuration.maxComputeWorkGroupSize[0] = physicalDeviceProperties.limits.maxComputeWorkGroupSize[0];
		app->configuration.maxComputeWorkGroupSize[1] = physicalDeviceProperties.limits.maxComputeWorkGroupSize[1];
		app->configuration.maxComputeWorkGroupSize[2] = physicalDeviceProperties.limits.maxComputeWorkGroupSize[2];
		if ((physicalDeviceProperties.vendorID == 0x8086) && (!app->configuration.doublePrecision)) app->configuration.halfThreads = 1;
		app->configuration.sharedMemorySize = physicalDeviceProperties.limits.maxComputeSharedMemorySize;
		app->configuration.sharedMemorySizePow2 = (uint32_t)pow(2, (uint32_t)log2(physicalDeviceProperties.limits.maxComputeSharedMemorySize));

		switch (physicalDeviceProperties.vendorID) {
		case 0x10DE://NVIDIA
			app->configuration.coalescedMemory = (app->configuration.halfPrecision) ? 64 : 32;//the coalesced memory is equal to 32 bytes between L2 and VRAM. 
			app->configuration.useLUT = (app->configuration.doublePrecision) ? 1 : 0;
			app->configuration.warpSize = 32;
			app->configuration.registerBoostNonPow2 = 0;
			app->configuration.registerBoost = 4;
			app->configuration.registerBoost4Step = 1;
			app->configuration.swapTo3Stage4Step = 0;
			break;
		case 0x8086://INTEL
			app->configuration.coalescedMemory = (app->configuration.halfPrecision) ? 128 : 64;
			app->configuration.useLUT = 1;
			app->configuration.warpSize = 32;
			app->configuration.registerBoostNonPow2 = 0;
			app->configuration.registerBoost = 2;
			app->configuration.registerBoost4Step = 1;
			app->configuration.swapTo3Stage4Step = 0;
			break;
		case 0x1002://AMD
			app->configuration.coalescedMemory = (app->configuration.halfPrecision) ? 64 : 32;
			app->configuration.useLUT = (app->configuration.doublePrecision) ? 1 : 0;
			app->configuration.warpSize = 64;
			app->configuration.registerBoostNonPow2 = 0;
			app->configuration.registerBoost = (physicalDeviceProperties.limits.maxComputeSharedMemorySize >= 65536) ? 2 : 4;
			app->configuration.registerBoost4Step = 1;
			app->configuration.swapTo3Stage4Step = 0;
			break;
		default:
			app->configuration.coalescedMemory = (app->configuration.halfPrecision) ? 128 : 64;
			app->configuration.useLUT = (app->configuration.doublePrecision) ? 1 : 0;
			app->configuration.warpSize = 32;
			app->configuration.registerBoostNonPow2 = 0;
			app->configuration.registerBoost = 1;
			app->configuration.registerBoost4Step = 1;
			app->configuration.swapTo3Stage4Step = 0;
			break;
		}
#elif(VKFFT_BACKEND==1)
		if (inputLaunchConfiguration.device == 0) return 1002;
		app->configuration.device = inputLaunchConfiguration.device;
		if (inputLaunchConfiguration.num_streams != 0)	app->configuration.num_streams = inputLaunchConfiguration.num_streams;
		if (inputLaunchConfiguration.stream != 0)	app->configuration.stream = inputLaunchConfiguration.stream;
		app->configuration.streamID = 0;
		int value = 0;
		cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, app->configuration.device[0]);
		app->configuration.maxThreadsNum = value;
		cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, app->configuration.device[0]);
		app->configuration.maxComputeWorkGroupCount[0] = value;
		cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, app->configuration.device[0]);
		app->configuration.maxComputeWorkGroupCount[1] = value;
		cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, app->configuration.device[0]);
		app->configuration.maxComputeWorkGroupCount[2] = value;
		cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, app->configuration.device[0]);
		app->configuration.maxComputeWorkGroupSize[0] = value;
		cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, app->configuration.device[0]);
		app->configuration.maxComputeWorkGroupSize[1] = value;
		cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, app->configuration.device[0]);
		app->configuration.maxComputeWorkGroupSize[2] = value;
		cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, app->configuration.device[0]);
		app->configuration.sharedMemorySizeStatic = value;
		//cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, app->configuration.device[0]);
		app->configuration.sharedMemorySize = value;
		cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_WARP_SIZE, app->configuration.device[0]);
		app->configuration.warpSize = value;
		app->configuration.sharedMemorySizePow2 = (uint32_t)pow(2, (uint32_t)log2(app->configuration.sharedMemorySize));
		if (app->configuration.num_streams > 1) {
			app->configuration.stream_event = (cudaEvent_t*)malloc(app->configuration.num_streams * sizeof(cudaEvent_t));
			for (uint32_t i = 0; i < app->configuration.num_streams; i++) {
				cudaEventCreate(&app->configuration.stream_event[i]);
			}
		}

		app->configuration.coalescedMemory = (app->configuration.halfPrecision) ? 64 : 32;//the coalesced memory is equal to 32 bytes between L2 and VRAM. 
		app->configuration.useLUT = (app->configuration.doublePrecision) ? 1 : 0;
		app->configuration.registerBoostNonPow2 = 0;
		app->configuration.registerBoost = 2;
		app->configuration.registerBoost4Step = 1;
		app->configuration.swapTo3Stage4Step = 0;

#elif(VKFFT_BACKEND==2)
		if (inputLaunchConfiguration.device == 0) return 1002;
		app->configuration.device = inputLaunchConfiguration.device;
		if (inputLaunchConfiguration.num_streams != 0)	app->configuration.num_streams = inputLaunchConfiguration.num_streams;
		if (inputLaunchConfiguration.stream != 0)	app->configuration.stream = inputLaunchConfiguration.stream;
		app->configuration.streamID = 0;
		int value = 0;
		hipDeviceGetAttribute(&value, hipDeviceAttributeMaxThreadsPerBlock, app->configuration.device[0]);
		app->configuration.maxThreadsNum = value;
		hipDeviceGetAttribute(&value, hipDeviceAttributeMaxGridDimX, app->configuration.device[0]);
		app->configuration.maxComputeWorkGroupCount[0] = value;
		hipDeviceGetAttribute(&value, hipDeviceAttributeMaxGridDimY, app->configuration.device[0]);
		app->configuration.maxComputeWorkGroupCount[1] = value;
		hipDeviceGetAttribute(&value, hipDeviceAttributeMaxGridDimZ, app->configuration.device[0]);
		app->configuration.maxComputeWorkGroupCount[2] = value;
		hipDeviceGetAttribute(&value, hipDeviceAttributeMaxBlockDimX, app->configuration.device[0]);
		app->configuration.maxComputeWorkGroupSize[0] = value;
		hipDeviceGetAttribute(&value, hipDeviceAttributeMaxBlockDimY, app->configuration.device[0]);
		app->configuration.maxComputeWorkGroupSize[1] = value;
		hipDeviceGetAttribute(&value, hipDeviceAttributeMaxBlockDimZ, app->configuration.device[0]);
		app->configuration.maxComputeWorkGroupSize[2] = value;
		hipDeviceGetAttribute(&value, hipDeviceAttributeMaxSharedMemoryPerBlock, app->configuration.device[0]);
		app->configuration.sharedMemorySizeStatic = value;
		//hipDeviceGetAttribute(&value, hipDeviceAttributeMaxSharedMemoryPerBlockOptin, app->configuration.device[0]);
		app->configuration.sharedMemorySize = value;
		hipDeviceGetAttribute(&value, hipDeviceAttributeWarpSize, app->configuration.device[0]);
		app->configuration.warpSize = value;
		app->configuration.sharedMemorySizePow2 = (uint32_t)pow(2, (uint32_t)log2(app->configuration.sharedMemorySize));
		if (app->configuration.num_streams > 1) {
			app->configuration.stream_event = (hipEvent_t*)malloc(app->configuration.num_streams * sizeof(hipEvent_t));
			for (uint32_t i = 0; i < app->configuration.num_streams; i++) {
				hipEventCreate(&app->configuration.stream_event[i]);
			}
		}
		app->configuration.coalescedMemory = (app->configuration.halfPrecision) ? 64 : 32;
		app->configuration.useLUT = (app->configuration.doublePrecision) ? 1 : 0;
		app->configuration.registerBoostNonPow2 = 0;
		app->configuration.registerBoost = 1;
		app->configuration.registerBoost4Step = 1;
		app->configuration.swapTo3Stage4Step = 0;

#endif
		//set main parameters:
		if (inputLaunchConfiguration.FFTdim == 0)
			return 2001;
		app->configuration.FFTdim = inputLaunchConfiguration.FFTdim;
		if (inputLaunchConfiguration.size[0] == 0)
			return 3;

		app->configuration.size[0] = inputLaunchConfiguration.size[0];

		if (inputLaunchConfiguration.bufferStride[0] == 0) {
			if (inputLaunchConfiguration.performR2C)
				app->configuration.bufferStride[0] = app->configuration.size[0] / 2 + 1;
			else
			app->configuration.bufferStride[0] = app->configuration.size[0];
		}
		else
			app->configuration.bufferStride[0] = inputLaunchConfiguration.bufferStride[0];

		if (inputLaunchConfiguration.inputBufferStride[0] == 0) {
			if (inputLaunchConfiguration.performR2C)
				app->configuration.inputBufferStride[0] = app->configuration.size[0] + 2;
			else
			app->configuration.inputBufferStride[0] = app->configuration.size[0];
		}
		else
			app->configuration.inputBufferStride[0] = inputLaunchConfiguration.inputBufferStride[0];

		if (inputLaunchConfiguration.outputBufferStride[0] == 0) {
			if (inputLaunchConfiguration.performR2C)
				app->configuration.outputBufferStride[0] = app->configuration.size[0] + 2;
			else
			app->configuration.outputBufferStride[0] = app->configuration.size[0];
		}
		else
			app->configuration.outputBufferStride[0] = inputLaunchConfiguration.outputBufferStride[0];
		for (uint32_t i = 1; i < 3; i++) {
			if (inputLaunchConfiguration.size[i] == 0)
				return 3;

			app->configuration.size[i] = inputLaunchConfiguration.size[i];

			if (inputLaunchConfiguration.bufferStride[i] == 0)
				app->configuration.bufferStride[i] = app->configuration.bufferStride[i - 1] * app->configuration.size[i];
			else
				app->configuration.bufferStride[i] = inputLaunchConfiguration.bufferStride[i];

			if (inputLaunchConfiguration.inputBufferStride[i] == 0)
				app->configuration.inputBufferStride[i] = app->configuration.inputBufferStride[i - 1] * app->configuration.size[i];
			else
				app->configuration.inputBufferStride[i] = inputLaunchConfiguration.inputBufferStride[i];

			if (inputLaunchConfiguration.outputBufferStride[i] == 0)
				app->configuration.outputBufferStride[i] = app->configuration.outputBufferStride[i - 1] * app->configuration.size[i];
			else
				app->configuration.outputBufferStride[i] = inputLaunchConfiguration.outputBufferStride[i];
		}

		app->configuration.isInputFormatted = inputLaunchConfiguration.isInputFormatted;
		app->configuration.isOutputFormatted = inputLaunchConfiguration.isOutputFormatted;
		app->configuration.performConvolution = inputLaunchConfiguration.performConvolution;

		if (inputLaunchConfiguration.bufferNum == 0)	app->configuration.bufferNum = 1;
		else app->configuration.bufferNum = inputLaunchConfiguration.bufferNum;

		if (inputLaunchConfiguration.bufferSize == 0) return 2001;
		if (inputLaunchConfiguration.buffer == 0) return 2002;
		app->configuration.bufferSize = inputLaunchConfiguration.bufferSize;
		app->configuration.buffer = inputLaunchConfiguration.buffer;

		if (inputLaunchConfiguration.userTempBuffer != 0)	app->configuration.userTempBuffer = inputLaunchConfiguration.userTempBuffer;

		if (app->configuration.userTempBuffer != 0) {
			if (inputLaunchConfiguration.tempBufferNum == 0)	app->configuration.tempBufferNum = 1;
			else app->configuration.tempBufferNum = inputLaunchConfiguration.tempBufferNum;

			if (inputLaunchConfiguration.tempBufferSize == 0) return 2003;
			if (inputLaunchConfiguration.tempBuffer == 0) return 2004;
			app->configuration.tempBufferSize = inputLaunchConfiguration.tempBufferSize;
			app->configuration.tempBuffer = inputLaunchConfiguration.tempBuffer;
		}
		else {
			app->configuration.tempBufferNum = 1;
			app->configuration.tempBufferSize = (uint64_t*)malloc(sizeof(uint64_t));
			app->configuration.tempBufferSize[0] = 0;

			for (uint32_t i = 0; i < app->configuration.bufferNum; i++) {
				app->configuration.tempBufferSize[0] += app->configuration.bufferSize[i];
			}
		}

		if (app->configuration.isInputFormatted) {
			if (inputLaunchConfiguration.inputBufferNum == 0)	app->configuration.inputBufferNum = 1;
			else app->configuration.inputBufferNum = inputLaunchConfiguration.inputBufferNum;

			if (inputLaunchConfiguration.inputBufferSize == 0) return 2005;
			if (inputLaunchConfiguration.inputBuffer == 0) return 2006;
			app->configuration.inputBufferSize = inputLaunchConfiguration.inputBufferSize;
			app->configuration.inputBuffer = inputLaunchConfiguration.inputBuffer;
		}
		else {
			app->configuration.inputBufferNum = app->configuration.bufferNum;

			app->configuration.inputBufferSize = app->configuration.bufferSize;
			app->configuration.inputBuffer = app->configuration.buffer;
		}
		if (app->configuration.isOutputFormatted) {
			if (inputLaunchConfiguration.outputBufferNum == 0)	app->configuration.outputBufferNum = 1;
			else
				app->configuration.outputBufferNum = inputLaunchConfiguration.outputBufferNum;

			if (inputLaunchConfiguration.outputBufferSize == 0) return 2007;
			if (inputLaunchConfiguration.outputBuffer == 0) return 2008;
			app->configuration.outputBufferSize = inputLaunchConfiguration.outputBufferSize;
			app->configuration.outputBuffer = inputLaunchConfiguration.outputBuffer;
		}
		else {
			app->configuration.outputBufferNum = app->configuration.bufferNum;

			app->configuration.outputBufferSize = app->configuration.bufferSize;
			app->configuration.outputBuffer = app->configuration.buffer;
		}
		if (app->configuration.performConvolution) {
			if (inputLaunchConfiguration.kernelNum == 0)	app->configuration.kernelNum = 1;
			else app->configuration.kernelNum = inputLaunchConfiguration.kernelNum;

			if (inputLaunchConfiguration.kernelSize == 0) return 2009;
			if (inputLaunchConfiguration.kernel == 0) return 2010;
			app->configuration.kernelSize = inputLaunchConfiguration.kernelSize;
			app->configuration.kernel = inputLaunchConfiguration.kernel;
		}

		//set optional parameters:
		if (inputLaunchConfiguration.coalescedMemory != 0)	app->configuration.coalescedMemory = inputLaunchConfiguration.coalescedMemory;
		app->configuration.aimThreads = 128;
		if (inputLaunchConfiguration.aimThreads != 0)	app->configuration.aimThreads = inputLaunchConfiguration.aimThreads;
		app->configuration.numSharedBanks = 32;
		if (inputLaunchConfiguration.numSharedBanks != 0)	app->configuration.numSharedBanks = inputLaunchConfiguration.numSharedBanks;

		if (inputLaunchConfiguration.performR2C != 0) {
			app->configuration.performR2C = inputLaunchConfiguration.performR2C;
			/*if (inputLaunchConfiguration.bufferStride[0] == 0)
				app->configuration.bufferStride[0] = app->configuration.size[0];

			if (inputLaunchConfiguration.inputBufferStride[0] == 0)
				app->configuration.inputBufferStride[0] = app->configuration.size[0];

			if (inputLaunchConfiguration.outputBufferStride[0] == 0)
				app->configuration.outputBufferStride[0] = app->configuration.size[0];

			if (inputLaunchConfiguration.bufferStride[1] == 0)
				app->configuration.bufferStride[1] = (app->configuration.bufferStride[0] / 2 + 1) * app->configuration.size[1];

			if (inputLaunchConfiguration.inputBufferStride[1] == 0)
				app->configuration.inputBufferStride[1] = (app->configuration.inputBufferStride[0] / 2 + 1) * app->configuration.size[1];

			if (inputLaunchConfiguration.outputBufferStride[1] == 0)
				app->configuration.outputBufferStride[1] = (app->configuration.outputBufferStride[0] / 2 + 1) * app->configuration.size[1];

			if (inputLaunchConfiguration.bufferStride[2] == 0)
				app->configuration.bufferStride[2] = app->configuration.bufferStride[1] * app->configuration.size[2];

			if (inputLaunchConfiguration.inputBufferStride[2] == 0)
				app->configuration.inputBufferStride[2] = app->configuration.inputBufferStride[1] * app->configuration.size[2];

			if (inputLaunchConfiguration.outputBufferStride[2] == 0)
				app->configuration.outputBufferStride[2] = app->configuration.outputBufferStride[1] * app->configuration.size[2];*/
		}
		app->configuration.normalize = 0;
		if (inputLaunchConfiguration.normalize != 0)	app->configuration.normalize = inputLaunchConfiguration.normalize;
		app->configuration.reorderFourStep = 1;
		if (inputLaunchConfiguration.disableReorderFourStep != 0) app->configuration.reorderFourStep = 0;
		if (inputLaunchConfiguration.frequencyZeroPadding != 0) app->configuration.frequencyZeroPadding = inputLaunchConfiguration.frequencyZeroPadding;
		for (uint32_t i = 0; i < app->configuration.FFTdim; i++) {
			if (inputLaunchConfiguration.performZeropadding[i] != 0) {
				app->configuration.performZeropadding[i] = inputLaunchConfiguration.performZeropadding[i];
				app->configuration.fft_zeropad_left[i] = inputLaunchConfiguration.fft_zeropad_left[i];
				app->configuration.fft_zeropad_right[i] = inputLaunchConfiguration.fft_zeropad_right[i];
			}
		}
		if (inputLaunchConfiguration.registerBoost != 0)	app->configuration.registerBoost = inputLaunchConfiguration.registerBoost;
		if (inputLaunchConfiguration.registerBoostNonPow2 != 0)	app->configuration.registerBoostNonPow2 = inputLaunchConfiguration.registerBoostNonPow2;
		if (inputLaunchConfiguration.registerBoost4Step != 0)	app->configuration.registerBoost4Step = inputLaunchConfiguration.registerBoost4Step;

		if (app->configuration.performR2C != 0) {
			app->configuration.registerBoost = 1;
			app->configuration.registerBoostNonPow2 = 0;
			app->configuration.registerBoost4Step = 1;
		}

		app->configuration.coordinateFeatures = 1;
		app->configuration.matrixConvolution = 1;
		app->configuration.numberBatches = 1;
		app->configuration.numberKernels = 1;

		if (inputLaunchConfiguration.kernelConvolution != 0) {
			if (inputLaunchConfiguration.coordinateFeatures != 0)	app->configuration.coordinateFeatures = inputLaunchConfiguration.coordinateFeatures;
			if (inputLaunchConfiguration.numberBatches != 0)	app->configuration.numberBatches = inputLaunchConfiguration.numberBatches;

			app->configuration.kernelConvolution = inputLaunchConfiguration.kernelConvolution;
			app->configuration.reorderFourStep = 0;
			app->configuration.registerBoost = 1;
			app->configuration.registerBoostNonPow2 = 0;
			app->configuration.registerBoost4Step = 1;
		}

		if (app->configuration.performConvolution) {

			if (inputLaunchConfiguration.coordinateFeatures != 0)	app->configuration.coordinateFeatures = inputLaunchConfiguration.coordinateFeatures;
			if (inputLaunchConfiguration.matrixConvolution != 0)	app->configuration.matrixConvolution = inputLaunchConfiguration.matrixConvolution;
			if (inputLaunchConfiguration.numberBatches != 0)	app->configuration.numberBatches = inputLaunchConfiguration.numberBatches;
			if (inputLaunchConfiguration.numberKernels != 0)	app->configuration.numberKernels = inputLaunchConfiguration.numberKernels;

			if (inputLaunchConfiguration.symmetricKernel != 0)	app->configuration.symmetricKernel = inputLaunchConfiguration.symmetricKernel;

			app->configuration.reorderFourStep = 0;
			app->configuration.registerBoost = 1;
			app->configuration.registerBoostNonPow2 = 0;
			app->configuration.registerBoost4Step = 1;
			if (app->configuration.matrixConvolution > 1) app->configuration.coordinateFeatures = app->configuration.matrixConvolution;
		}

		if (inputLaunchConfiguration.reorderFourStep != 0)	app->configuration.reorderFourStep = inputLaunchConfiguration.reorderFourStep;

		if (inputLaunchConfiguration.halfThreads != 0)	app->configuration.halfThreads = inputLaunchConfiguration.halfThreads;
		if (inputLaunchConfiguration.swapTo3Stage4Step != 0)	app->configuration.swapTo3Stage4Step = inputLaunchConfiguration.swapTo3Stage4Step;
		if (inputLaunchConfiguration.performHalfBandwidthBoost != 0)	app->configuration.performHalfBandwidthBoost = inputLaunchConfiguration.performHalfBandwidthBoost;
		if (inputLaunchConfiguration.devicePageSize != 0)	app->configuration.devicePageSize = inputLaunchConfiguration.devicePageSize;
		if (inputLaunchConfiguration.localPageSize != 0)	app->configuration.localPageSize = inputLaunchConfiguration.localPageSize;

		//temporary set:
		app->configuration.registerBoost4Step = 1;

		uint32_t res = 0;
		uint32_t initSharedMemory = app->configuration.sharedMemorySize;
		app->localFFTPlan_inverse = (VkFFTPlan*)malloc(sizeof(VkFFTPlan));
		for (uint32_t i = 0; i < app->configuration.FFTdim; i++) {
			app->configuration.sharedMemorySize = ((app->configuration.size[i] & (app->configuration.size[i] - 1)) == 0) ? app->configuration.sharedMemorySizePow2 : initSharedMemory;
			VkFFTScheduler(app, app->localFFTPlan_inverse, i, 0);
			for (uint32_t j = 0; j < app->localFFTPlan_inverse->numAxisUploads[i]; j++) {
				res = VkFFTPlanAxis(app, app->localFFTPlan_inverse, i, j, 1);
				if (res != 0) return res;
			}
		}
		app->localFFTPlan = (VkFFTPlan*)malloc(sizeof(VkFFTPlan));
		for (uint32_t i = 0; i < app->configuration.FFTdim; i++) {
			app->configuration.sharedMemorySize = ((app->configuration.size[i] & (app->configuration.size[i] - 1)) == 0) ? app->configuration.sharedMemorySizePow2 : initSharedMemory;
			VkFFTScheduler(app, app->localFFTPlan, i, 0);
			for (uint32_t j = 0; j < app->localFFTPlan->numAxisUploads[i]; j++) {
				res = VkFFTPlanAxis(app, app->localFFTPlan, i, j, 0);
				if (res != 0) return res;
			}
		}
#if(VKFFT_BACKEND==0)
		if (!app->configuration.isCompilerInitialized)
			glslang_finalize_process();
#endif
		return res;
	}
	static inline void dispatchEnhanced(VkFFTApplication* app, VkFFTAxis* axis, uint32_t* dispatchBlock) {
		uint32_t maxBlockSize[3] = { (uint32_t)pow(2,(uint32_t)log2(app->configuration.maxComputeWorkGroupCount[0])),(uint32_t)pow(2,(uint32_t)log2(app->configuration.maxComputeWorkGroupCount[1])),(uint32_t)pow(2,(uint32_t)log2(app->configuration.maxComputeWorkGroupCount[2])) };
		uint32_t blockNumber[3] = { (uint32_t)ceil(dispatchBlock[0] / (float)maxBlockSize[0]),(uint32_t)ceil(dispatchBlock[1] / (float)maxBlockSize[1]),(uint32_t)ceil(dispatchBlock[2] / (float)maxBlockSize[2]) };
		if ((blockNumber[0] > 1) && (blockNumber[0] * maxBlockSize[0] != dispatchBlock[0])) {
			for (uint32_t i = app->configuration.maxComputeWorkGroupCount[0]; i > 0; i--) {
				if (dispatchBlock[0] % i == 0) {
					maxBlockSize[0] = i;
					blockNumber[0] = dispatchBlock[0] / i;
					i = 1;
				}
			}
		}
		if ((blockNumber[1] > 1) && (blockNumber[1] * maxBlockSize[1] != dispatchBlock[1])) {
			for (uint32_t i = app->configuration.maxComputeWorkGroupCount[1]; i > 0; i--) {
				if (dispatchBlock[1] % i == 0) {
					maxBlockSize[1] = i;
					blockNumber[1] = dispatchBlock[1] / i;
					i = 1;
				}
			}
		}
		if ((blockNumber[2] > 1) && (blockNumber[2] * maxBlockSize[2] != dispatchBlock[2])) {
			for (uint32_t i = app->configuration.maxComputeWorkGroupCount[2]; i > 0; i--) {
				if (dispatchBlock[2] % i == 0) {
					maxBlockSize[2] = i;
					blockNumber[2] = dispatchBlock[2] / i;
					i = 1;
				}
			}
		}
		//printf("%d %d %d\n", dispatchBlock[0], dispatchBlock[1], dispatchBlock[2]);
		for (uint32_t i = 0; i < 3; i++)
			if (blockNumber[i] == 1) maxBlockSize[i] = dispatchBlock[i];

		for (uint32_t i = 0; i < blockNumber[0]; i++) {
			for (uint32_t j = 0; j < blockNumber[1]; j++) {
				for (uint32_t k = 0; k < blockNumber[2]; k++) {
					axis->pushConstants.workGroupShift[0] = i * maxBlockSize[0];
					axis->pushConstants.workGroupShift[1] = j * maxBlockSize[1];
					axis->pushConstants.workGroupShift[2] = k * maxBlockSize[2];
#if(VKFFT_BACKEND==0)
					vkCmdPushConstants(app->configuration.commandBuffer[0], axis->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTPushConstantsLayout), &axis->pushConstants);
					vkCmdDispatch(app->configuration.commandBuffer[0], maxBlockSize[0], maxBlockSize[1], maxBlockSize[2]);
#elif(VKFFT_BACKEND==1)
					void* args[5];
					CUresult result;
					args[0] = axis->inputBuffer;
					args[1] = axis->outputBuffer;
					uint32_t args_id = 2;
					if (axis->specializationConstants.convolutionStep) {
						args[args_id] = app->configuration.kernel;
						args_id++;
					}
					if (axis->specializationConstants.LUT) {
						args[args_id] = &axis->bufferLUT;
						args_id++;
					}
					//args[args_id] = &axis->pushConstants;
					if ((i > 0) || (j > 0) || (k > 0) || (axis->pushConstants.coordinate > 0) || (axis->pushConstants.batch > 0)) {
						result = cuMemcpyHtoD(axis->consts_addr, &axis->pushConstants, sizeof(VkFFTPushConstantsLayout));
						if (result != CUDA_SUCCESS) {
							printf("Error: %d\n", result);
						}
					}
					if (app->configuration.num_streams >= 1) {
						result = cuLaunchKernel(axis->VkFFTKernel,
							maxBlockSize[0], maxBlockSize[1], maxBlockSize[2],     // grid dim
							axis->specializationConstants.localSize[0], axis->specializationConstants.localSize[1], axis->specializationConstants.localSize[2],   // block dim
							0, app->configuration.stream[app->configuration.streamID],             // shared mem and stream
							args, 0);
					}
					else {
						result = cuLaunchKernel(axis->VkFFTKernel,
							maxBlockSize[0], maxBlockSize[1], maxBlockSize[2],     // grid dim
							axis->specializationConstants.localSize[0], axis->specializationConstants.localSize[1], axis->specializationConstants.localSize[2],   // block dim
							0, NULL,             // shared mem and stream
							args, 0);
					}
					if (result != CUDA_SUCCESS) {
						printf("Error: %d\n", result);
					}
					if (app->configuration.num_streams > 1) {
						app->configuration.streamID = app->configuration.streamCounter % app->configuration.num_streams;
						if (app->configuration.streamCounter == 0)
							cudaEventRecord(app->configuration.stream_event[app->configuration.streamID], app->configuration.stream[app->configuration.streamID]);
						app->configuration.streamCounter++;
					}
#elif(VKFFT_BACKEND==2)
					hipError_t result;
					void* args[5];
					args[0] = axis->inputBuffer;
					args[1] = axis->outputBuffer;
					uint32_t args_id = 2;
					if (axis->specializationConstants.convolutionStep) {
						args[args_id] = app->configuration.kernel;
						args_id++;
					}
					if (axis->specializationConstants.LUT) {
						args[args_id] = &axis->bufferLUT;
						args_id++;
					}
					//args[args_id] = &axis->pushConstants;
					if ((i > 0) || (j > 0) || (k > 0) || (axis->pushConstants.coordinate > 0) || (axis->pushConstants.batch > 0)) {
						result = hipMemcpyHtoD(axis->consts_addr, &axis->pushConstants, sizeof(VkFFTPushConstantsLayout));
						if (result != hipSuccess) {
							printf("Error: %d\n", result);
						}
					}
					//printf("%d %d %d %d %d %d\n",maxBlockSize[0], maxBlockSize[1], maxBlockSize[2], axis->specializationConstants.localSize[0], axis->specializationConstants.localSize[1], axis->specializationConstants.localSize[2]);
					if (app->configuration.num_streams >= 1) {
						result = hipModuleLaunchKernel(axis->VkFFTKernel,
							maxBlockSize[0], maxBlockSize[1], maxBlockSize[2],     // grid dim
							axis->specializationConstants.localSize[0], axis->specializationConstants.localSize[1], axis->specializationConstants.localSize[2],   // block dim
							0, app->configuration.stream[app->configuration.streamID],             // shared mem and stream
							args, 0);
					}
					else {
						result = hipModuleLaunchKernel(axis->VkFFTKernel,
							maxBlockSize[0], maxBlockSize[1], maxBlockSize[2],     // grid dim
							axis->specializationConstants.localSize[0], axis->specializationConstants.localSize[1], axis->specializationConstants.localSize[2],   // block dim
							0, NULL,             // shared mem and stream
							args, 0);
					}
					if (result != hipSuccess) {
						printf("Error: %d\n", result);
					}
					if (app->configuration.num_streams > 1) {
						app->configuration.streamID = app->configuration.streamCounter % app->configuration.num_streams;
						if (app->configuration.streamCounter == 0)
							hipEventRecord(app->configuration.stream_event[app->configuration.streamID], app->configuration.stream[app->configuration.streamID]);
						app->configuration.streamCounter++;
					}
#endif
				}
			}
		}
	}
	static inline void VkFFTSync(VkFFTApplication* app) {
#if(VKFFT_BACKEND==0)
		vkCmdPipelineBarrier(app->configuration.commandBuffer[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, app->configuration.memory_barrier, 0, NULL, 0, NULL);
#elif(VKFFT_BACKEND==1)
		if (app->configuration.num_streams > 1) {
			for (uint32_t s = 0; s < app->configuration.num_streams; s++)
				cudaEventSynchronize(app->configuration.stream_event[s]);
			app->configuration.streamCounter = 0;
		}
#elif(VKFFT_BACKEND==2)
		if (app->configuration.num_streams > 1) {
			for (uint32_t s = 0; s < app->configuration.num_streams; s++)
				hipEventSynchronize(app->configuration.stream_event[s]);
			app->configuration.streamCounter = 0;
		}
#endif
	}

	static inline void VkFFTAppend(VkFFTApplication* app, int inverse, void* launchParams) {
#if(VKFFT_BACKEND==0)
		app->configuration.commandBuffer = (VkCommandBuffer*)launchParams;
		VkMemoryBarrier memory_barrier = {
				VK_STRUCTURE_TYPE_MEMORY_BARRIER,
				0,
				VK_ACCESS_SHADER_WRITE_BIT,
				VK_ACCESS_SHADER_READ_BIT,
		};
		app->configuration.memory_barrier = &memory_barrier;
#elif(VKFFT_BACKEND==1)
		app->configuration.streamCounter = 0;
#elif(VKFFT_BACKEND==2)
		app->configuration.streamCounter = 0;
#endif
		uint32_t localSize0 = (app->configuration.performR2C == 1) ? app->configuration.size[0]/2+1 : app->configuration.size[0];
		if (inverse != 1) {
			//FFT axis 0
			for (uint32_t j = 0; j < app->configuration.numberBatches; j++) {
				for (int l = app->localFFTPlan->numAxisUploads[0] - 1; l >= 0; l--) {
					VkFFTAxis* axis = &app->localFFTPlan->axes[0][l];
					axis->pushConstants.batch = j;
					uint32_t maxCoordinate = ((app->configuration.matrixConvolution) > 1 && (app->configuration.performConvolution) && (app->configuration.FFTdim == 1)) ? 1 : app->configuration.coordinateFeatures;
					for (uint32_t i = 0; i < maxCoordinate; i++) {
						axis->pushConstants.coordinate = i;

#if(VKFFT_BACKEND==0)
						vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
						vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
#endif
						uint32_t dispatchBlock[3];
						if (l == 0) {
							if (app->localFFTPlan->numAxisUploads[0] > 2) {
								dispatchBlock[0] = ceil(ceil(app->configuration.size[0] / axis->specializationConstants.fftDim / (double)axis->axisBlock[1]) / (double)app->localFFTPlan->axisSplit[0][1]) * app->localFFTPlan->axisSplit[0][1];
								dispatchBlock[1] = app->configuration.size[1];
							}
							else {
								if (app->localFFTPlan->numAxisUploads[0] > 1) {
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
						dispatchEnhanced(app, axis, dispatchBlock);
					}
					VkFFTSync(app);
				}
			}

			if (app->configuration.FFTdim > 1) {

				//FFT axis 1
				if ((app->configuration.FFTdim == 2) && (app->configuration.performConvolution)) {
					/*if (app->configuration.performR2C == 1) {
						for (int l = app->localFFTPlan->numSupportAxisUploads[0] - 1; l >= 0; l--) {
							VkFFTAxis* axis = &app->localFFTPlan->supportAxes[0][l];
							uint32_t maxCoordinate = ((app->configuration.matrixConvolution > 1) && (l == 0)) ? 1 : app->configuration.coordinateFeatures;
							for (uint32_t i = 0; i < maxCoordinate; i++) {
								axis->pushConstants.coordinate = i;

								axis->pushConstants.batch = ((l == 0) && (app->configuration.matrixConvolution == 1)) ? app->configuration.numberKernels : 0;

#if(VKFFT_BACKEND==0)
								vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
								vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
#endif
								uint32_t dispatchBlock[3];
								if (l == 0)
									dispatchBlock[0] = app->configuration.size[1] / axis->specializationConstants.fftDim;
								else
									dispatchBlock[0] = ceil(app->configuration.size[1] / axis->specializationConstants.fftDim / (double)axis->axisBlock[0]);

								dispatchBlock[1] = 1;
								dispatchBlock[2] = app->configuration.size[2];
								//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
								dispatchEnhanced(app, axis, dispatchBlock);

							}
							if (l > 0)
								VkFFTSync(app);
						}

					}*/

					for (int l = app->localFFTPlan->numAxisUploads[1] - 1; l >= 0; l--) {
						VkFFTAxis* axis = &app->localFFTPlan->axes[1][l];
						uint32_t maxCoordinate = ((app->configuration.matrixConvolution > 1) && (l == 0)) ? 1 : app->configuration.coordinateFeatures;
						for (uint32_t i = 0; i < maxCoordinate; i++) {

							axis->pushConstants.coordinate = i;
							axis->pushConstants.batch = ((l == 0) && (app->configuration.matrixConvolution == 1)) ? app->configuration.numberKernels : 0;
#if(VKFFT_BACKEND==0)
							vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
							vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
#endif
							uint32_t dispatchBlock[3];
							dispatchBlock[0] = ceil(localSize0 / (double)axis->axisBlock[0] * app->configuration.size[1] / (double)axis->specializationConstants.fftDim);
							dispatchBlock[1] = 1;
							dispatchBlock[2] = app->configuration.size[2];
							//if (app->configuration.performR2C == 1) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);
							//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
							dispatchEnhanced(app, axis, dispatchBlock);
						}
						VkFFTSync(app);
					}
				}
				else {
					/*if (app->configuration.performR2C == 1) {
						for (uint32_t j = 0; j < app->configuration.numberBatches; j++) {
							for (int l = app->localFFTPlan->numSupportAxisUploads[0] - 1; l >= 0; l--) {
								VkFFTAxis* axis = &app->localFFTPlan->supportAxes[0][l];
								axis->pushConstants.batch = j;
								for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {
									axis->pushConstants.coordinate = i;
#if(VKFFT_BACKEND==0)
									vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
									vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
#endif
									uint32_t dispatchBlock[3];
									if (l == 0)
										dispatchBlock[0] = app->configuration.size[1] / axis->specializationConstants.fftDim;
									else
										dispatchBlock[0] = ceil(app->configuration.size[1] / axis->specializationConstants.fftDim / (double)axis->axisBlock[0]);

									dispatchBlock[1] = 1;
									dispatchBlock[2] = app->configuration.size[2];
									//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
									dispatchEnhanced(app, axis, dispatchBlock);

								}
								if (l > 0)
									VkFFTSync(app);
							}
						}
					}*/
					for (uint32_t j = 0; j < app->configuration.numberBatches; j++) {
						for (int l = app->localFFTPlan->numAxisUploads[1] - 1; l >= 0; l--) {
							VkFFTAxis* axis = &app->localFFTPlan->axes[1][l];
							axis->pushConstants.batch = j;
							for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {
								axis->pushConstants.coordinate = i;
#if(VKFFT_BACKEND==0)
								vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
								vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
#endif
								uint32_t dispatchBlock[3];

								dispatchBlock[0] = ceil(localSize0 / (double)axis->axisBlock[0] * app->configuration.size[1] / (double)axis->specializationConstants.fftDim);
								dispatchBlock[1] = 1;
								dispatchBlock[2] = app->configuration.size[2];
								//if (app->configuration.performR2C == 1) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);
								//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
								dispatchEnhanced(app, axis, dispatchBlock);
							}
							VkFFTSync(app);
						}
					}

				}
			}
			//FFT axis 2
			if (app->configuration.FFTdim > 2) {
				if ((app->configuration.FFTdim == 3) && (app->configuration.performConvolution)) {

					/*if (app->configuration.performR2C == 1) {

						for (int l = app->localFFTPlan->numSupportAxisUploads[1] - 1; l >= 0; l--) {
							VkFFTAxis* axis = &app->localFFTPlan->supportAxes[1][l];
							uint32_t maxCoordinate = ((app->configuration.matrixConvolution > 1) && (l == 0)) ? 1 : app->configuration.coordinateFeatures;
							for (uint32_t i = 0; i < maxCoordinate; i++) {
								axis->pushConstants.coordinate = i;

								axis->pushConstants.batch = ((l == 0) && (app->configuration.matrixConvolution == 1)) ? app->configuration.numberKernels : 0;
#if(VKFFT_BACKEND==0)
								vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
								vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
#endif
								uint32_t dispatchBlock[3];
								dispatchBlock[0] = ceil(app->configuration.size[1] / (double)axis->axisBlock[0] * app->configuration.size[2] / (double)axis->specializationConstants.fftDim);
								dispatchBlock[1] = 1;
								dispatchBlock[2] = 1;
								dispatchEnhanced(app, axis, dispatchBlock);

							}
							if (l > 0)
								VkFFTSync(app);
						}
					}*/

					for (int l = app->localFFTPlan->numAxisUploads[2] - 1; l >= 0; l--) {

						VkFFTAxis* axis = &app->localFFTPlan->axes[2][l];
						uint32_t maxCoordinate = ((app->configuration.matrixConvolution > 1) && (l == 0)) ? 1 : app->configuration.coordinateFeatures;
						for (uint32_t i = 0; i < maxCoordinate; i++) {
							axis->pushConstants.coordinate = i;
							axis->pushConstants.batch = ((l == 0) && (app->configuration.matrixConvolution == 1)) ? app->configuration.numberKernels : 0;
#if(VKFFT_BACKEND==0)
							vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
							vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
#endif
							uint32_t dispatchBlock[3];
							dispatchBlock[0] = ceil(localSize0 / (double)axis->axisBlock[0] * app->configuration.size[2] / (double)axis->specializationConstants.fftDim);
							dispatchBlock[1] = 1;
							dispatchBlock[2] = app->configuration.size[1];
							//if (app->configuration.performR2C == 1) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);
							dispatchEnhanced(app, axis, dispatchBlock);

						}
						VkFFTSync(app);
					}
				}
				else {
					/*if (app->configuration.performR2C == 1) {
						for (uint32_t j = 0; j < app->configuration.numberBatches; j++) {
							for (int l = app->localFFTPlan->numSupportAxisUploads[1] - 1; l >= 0; l--) {
								VkFFTAxis* axis = &app->localFFTPlan->supportAxes[1][l];
								axis->pushConstants.batch = j;
								for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {
									axis->pushConstants.coordinate = i;
#if(VKFFT_BACKEND==0)
									vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
									vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
#endif
									uint32_t dispatchBlock[3];
									dispatchBlock[0] = ceil(app->configuration.size[1] / (double)axis->axisBlock[0] * app->configuration.size[2] / (double)axis->specializationConstants.fftDim);
									dispatchBlock[1] = 1;
									dispatchBlock[2] = 1;
									dispatchEnhanced(app, axis, dispatchBlock);

								}
								if (l > 0)
									VkFFTSync(app);
							}
						}
					}*/
					for (uint32_t j = 0; j < app->configuration.numberBatches; j++) {
						for (int l = app->localFFTPlan->numAxisUploads[2] - 1; l >= 0; l--) {
							VkFFTAxis* axis = &app->localFFTPlan->axes[2][l];
							axis->pushConstants.batch = j;
							for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {
								axis->pushConstants.coordinate = i;
#if(VKFFT_BACKEND==0)
								vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
								vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
#endif
								uint32_t dispatchBlock[3];
								dispatchBlock[0] = ceil(localSize0 / (double)axis->axisBlock[0] * app->configuration.size[2] / (double)axis->specializationConstants.fftDim);
								dispatchBlock[1] = 1;
								dispatchBlock[2] = app->configuration.size[1];
								//if (app->configuration.performR2C == 1) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);
								dispatchEnhanced(app, axis, dispatchBlock);
							}
							VkFFTSync(app);
						}
					}

				}

			}
		}
		if (app->configuration.performConvolution) {
			if (app->configuration.FFTdim > 2) {

				//multiple upload ifft leftovers
				if (app->configuration.FFTdim == 3) {
					/*if (app->configuration.performR2C == 1) {
						for (uint32_t j = 0; j < app->configuration.numberKernels; j++) {
							for (int l = 1; l < app->localFFTPlan_inverse->numSupportAxisUploads[1]; l++) {
								VkFFTAxis* axis = &app->localFFTPlan_inverse->supportAxes[1][l];
								uint32_t maxCoordinate = app->configuration.coordinateFeatures;
								for (uint32_t i = 0; i < maxCoordinate; i++) {
									axis->pushConstants.coordinate = i;
									axis->pushConstants.batch = j;
#if(VKFFT_BACKEND==0)
									vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
									vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
#endif
									uint32_t dispatchBlock[3];
									dispatchBlock[0] = ceil(app->configuration.size[1] / (double)axis->axisBlock[0] * app->configuration.size[2] / (double)axis->specializationConstants.fftDim);
									dispatchBlock[1] = 1;
									dispatchBlock[2] = 1;
									dispatchEnhanced(app, axis, dispatchBlock);

								}
								if (l > 0)
									VkFFTSync(app);
							}
						}
					}*/
					for (uint32_t j = 0; j < app->configuration.numberKernels; j++) {
						for (int l = 1; l < app->localFFTPlan_inverse->numAxisUploads[2]; l++) {
							VkFFTAxis* axis = &app->localFFTPlan_inverse->axes[2][l];
							uint32_t maxCoordinate = app->configuration.coordinateFeatures;
							for (uint32_t i = 0; i < maxCoordinate; i++) {
								axis->pushConstants.coordinate = i;
								axis->pushConstants.batch = j;
#if(VKFFT_BACKEND==0)
								vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
								vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
#endif
								uint32_t dispatchBlock[3];
								dispatchBlock[0] = ceil(localSize0 / (double)axis->axisBlock[0] * app->configuration.size[2] / (double)axis->specializationConstants.fftDim);
								dispatchBlock[1] = 1;
								dispatchBlock[2] = app->configuration.size[1];
								//if (app->configuration.performR2C == 1) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);
								dispatchEnhanced(app, axis, dispatchBlock);
							}
							VkFFTSync(app);
						}
					}
				}
				/*if (app->configuration.performR2C == 1) {
					for (uint32_t j = 0; j < app->configuration.numberKernels; j++) {
						for (int l = 0; l < app->localFFTPlan_inverse->numSupportAxisUploads[0]; l++) {
							VkFFTAxis* axis = &app->localFFTPlan_inverse->supportAxes[0][l];
							axis->pushConstants.batch = j;
							for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {

								axis->pushConstants.coordinate = i;
#if(VKFFT_BACKEND==0)
								vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
								vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
#endif
								uint32_t dispatchBlock[3];
								if (l == 0)
									dispatchBlock[0] = app->configuration.size[1] / axis->specializationConstants.fftDim;
								else
									dispatchBlock[0] = ceil(app->configuration.size[1] / axis->specializationConstants.fftDim / (double)axis->axisBlock[0]);
								dispatchBlock[1] = 1;
								dispatchBlock[2] = app->configuration.size[2];
								//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
								dispatchEnhanced(app, axis, dispatchBlock);
							}
							if (l < app->localFFTPlan_inverse->numSupportAxisUploads[0] - 1)
								VkFFTSync(app);
						}
					}
				}*/
				for (uint32_t j = 0; j < app->configuration.numberKernels; j++) {
					for (int l = 0; l < app->localFFTPlan_inverse->numAxisUploads[1]; l++) {
						VkFFTAxis* axis = &app->localFFTPlan_inverse->axes[1][l];
						axis->pushConstants.batch = j;
						for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {
							axis->pushConstants.coordinate = i;
#if(VKFFT_BACKEND==0)
							vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
							vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
#endif
							uint32_t dispatchBlock[3];
							dispatchBlock[0] = ceil(localSize0 / (double)axis->axisBlock[0] * app->configuration.size[1] / (double)axis->specializationConstants.fftDim);
							dispatchBlock[1] = 1;
							dispatchBlock[2] = app->configuration.size[2];
							//if (app->configuration.performR2C == 1) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);
							//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
							dispatchEnhanced(app, axis, dispatchBlock);

						}
						VkFFTSync(app);
					}
				}

			}
			if (app->configuration.FFTdim > 1) {
				if (app->configuration.FFTdim == 2) {
					/*if (app->configuration.performR2C == 1) {
						for (uint32_t j = 0; j < app->configuration.numberKernels; j++) {
							for (int l = 1; l < app->localFFTPlan_inverse->numSupportAxisUploads[0]; l++) {
								VkFFTAxis* axis = &app->localFFTPlan_inverse->supportAxes[0][l];
								uint32_t maxCoordinate = app->configuration.coordinateFeatures;
								for (uint32_t i = 0; i < maxCoordinate; i++) {
									axis->pushConstants.coordinate = i;
									axis->pushConstants.batch = j;
#if(VKFFT_BACKEND==0)
									vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
									vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
#endif
									uint32_t dispatchBlock[3];
									if (l == 0)
										dispatchBlock[0] = app->configuration.size[1] / axis->specializationConstants.fftDim;
									else
										dispatchBlock[0] = ceil(app->configuration.size[1] / axis->specializationConstants.fftDim / (double)axis->axisBlock[0]);
									dispatchBlock[1] = 1;
									dispatchBlock[2] = app->configuration.size[2];
									//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
									dispatchEnhanced(app, axis, dispatchBlock);

								}
								if (l > 0)
									VkFFTSync(app);
							}
						}

					}*/
					for (uint32_t j = 0; j < app->configuration.numberKernels; j++) {
						for (int l = 1; l < app->localFFTPlan_inverse->numAxisUploads[1]; l++) {
							VkFFTAxis* axis = &app->localFFTPlan_inverse->axes[1][l];
							uint32_t maxCoordinate = app->configuration.coordinateFeatures;
							for (uint32_t i = 0; i < maxCoordinate; i++) {

								axis->pushConstants.coordinate = i;
								axis->pushConstants.batch = j;
#if(VKFFT_BACKEND==0)
								vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
								vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
#endif
								uint32_t dispatchBlock[3];
								dispatchBlock[0] = ceil(localSize0 / (double)axis->axisBlock[0] * app->configuration.size[1] / (double)axis->specializationConstants.fftDim);
								dispatchBlock[1] = 1;
								dispatchBlock[2] = app->configuration.size[2];
								//if (app->configuration.performR2C == 1) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);
								//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
								dispatchEnhanced(app, axis, dispatchBlock);

							}
							VkFFTSync(app);
						}
					}
				}
				for (uint32_t j = 0; j < app->configuration.numberKernels; j++) {
					for (int l = 0; l < app->localFFTPlan_inverse->numAxisUploads[0]; l++) {
						VkFFTAxis* axis = &app->localFFTPlan_inverse->axes[0][l];
						axis->pushConstants.batch = j;
						for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {
							axis->pushConstants.coordinate = i;
#if(VKFFT_BACKEND==0)
							vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
							vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
#endif
							uint32_t dispatchBlock[3];
							if (l == 0) {
								if (app->localFFTPlan->numAxisUploads[0] > 2) {
									dispatchBlock[0] = ceil(ceil(app->configuration.size[0] / axis->specializationConstants.fftDim / (double)axis->axisBlock[1]) / (double)app->localFFTPlan->axisSplit[0][1]) * app->localFFTPlan->axisSplit[0][1];
									dispatchBlock[1] = app->configuration.size[1];
								}
								else {
									if (app->localFFTPlan->numAxisUploads[0] > 1) {
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
							dispatchEnhanced(app, axis, dispatchBlock);
						}
						VkFFTSync(app);
					}
				}


			}
			if (app->configuration.FFTdim == 1) {
				for (uint32_t j = 0; j < app->configuration.numberKernels; j++) {
					for (int l = 1; l < app->localFFTPlan_inverse->numAxisUploads[0]; l++) {
						VkFFTAxis* axis = &app->localFFTPlan_inverse->axes[0][l];
						uint32_t maxCoordinate = app->configuration.coordinateFeatures;
						for (uint32_t i = 0; i < maxCoordinate; i++) {

							axis->pushConstants.coordinate = i;
							axis->pushConstants.batch = j;
#if(VKFFT_BACKEND==0)
							vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
							vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
#endif
							uint32_t dispatchBlock[3];
							dispatchBlock[0] = ceil(localSize0 / (double)axis->axisBlock[0] * app->configuration.size[1] / (double)axis->specializationConstants.fftDim);
							dispatchBlock[1] = 1;
							dispatchBlock[2] = app->configuration.size[2];
							//if (app->configuration.performR2C == 1) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);
							//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
							dispatchEnhanced(app, axis, dispatchBlock);

						}
						VkFFTSync(app);
					}
				}
			}
		}

		if (inverse == 1) {
			//we start from axis 2 and go back to axis 0
			//FFT axis 2
			if (app->configuration.FFTdim > 2) {
				/*if (app->configuration.performR2C == 1) {
					for (uint32_t j = 0; j < app->configuration.numberBatches; j++) {
						for (int l = app->localFFTPlan->numSupportAxisUploads[1] - 1; l >= 0; l--) {
							if (!app->configuration.reorderFourStep) l = app->localFFTPlan->numSupportAxisUploads[1] - 1 - l;
							VkFFTAxis* axis = &app->localFFTPlan->supportAxes[1][l];
							axis->pushConstants.batch = j;
							for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {
								axis->pushConstants.coordinate = i;
#if(VKFFT_BACKEND==0)
								vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
								vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
#endif
								uint32_t dispatchBlock[3];
								dispatchBlock[0] = ceil(app->configuration.size[1] / (double)axis->axisBlock[0] * app->configuration.size[2] / (double)axis->specializationConstants.fftDim);
								dispatchBlock[1] = 1;
								dispatchBlock[2] = 1;
								//if (app->configuration.performZeropaddingInverse[1]) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);

								dispatchEnhanced(app, axis, dispatchBlock);

							}
							if (!app->configuration.reorderFourStep) l = app->localFFTPlan->numSupportAxisUploads[1] - 1 - l;
							if (l > 0)
								VkFFTSync(app);
						}
					}
				}*/

				for (uint32_t j = 0; j < app->configuration.numberBatches; j++) {
					for (int l = app->localFFTPlan_inverse->numAxisUploads[2] - 1; l >= 0; l--) {
						if (!app->configuration.reorderFourStep) l = app->localFFTPlan_inverse->numAxisUploads[2] - 1 - l;
						VkFFTAxis* axis = &app->localFFTPlan_inverse->axes[2][l];
						axis->pushConstants.batch = j;
						for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {
							axis->pushConstants.coordinate = i;
#if(VKFFT_BACKEND==0)
							vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
							vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
#endif
							uint32_t dispatchBlock[3];
							dispatchBlock[0] = ceil(localSize0 / (double)axis->axisBlock[0] * app->configuration.size[2] / (double)axis->specializationConstants.fftDim);
							dispatchBlock[1] = 1;
							dispatchBlock[2] = app->configuration.size[1];
							//if (app->configuration.performZeropaddingInverse[0]) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);
							//if (app->configuration.performZeropaddingInverse[1]) dispatchBlock[1] = ceil(dispatchBlock[1] / 2.0);

							//if (app->configuration.performR2C == 1) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);
							dispatchEnhanced(app, axis, dispatchBlock);
						}
						VkFFTSync(app);
						if (!app->configuration.reorderFourStep) l = app->localFFTPlan_inverse->numAxisUploads[2] - 1 - l;
					}
				}

			}
			if (app->configuration.FFTdim > 1) {

				//FFT axis 1
				/*if (app->configuration.performR2C == 1) {
					for (uint32_t j = 0; j < app->configuration.numberBatches; j++) {
						for (int l = app->localFFTPlan->numSupportAxisUploads[0] - 1; l >= 0; l--) {
							if (!app->configuration.reorderFourStep) l = app->localFFTPlan->numSupportAxisUploads[0] - 1 - l;
							VkFFTAxis* axis = &app->localFFTPlan->supportAxes[0][l];
							axis->pushConstants.batch = j;
							for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {
								axis->pushConstants.coordinate = i;
#if(VKFFT_BACKEND==0)
								vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
								vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
#endif
								uint32_t dispatchBlock[3];
								if (l == 0)
									dispatchBlock[0] = app->configuration.size[1] / axis->specializationConstants.fftDim;
								else
									dispatchBlock[0] = ceil(app->configuration.size[1] / axis->specializationConstants.fftDim / (double)axis->axisBlock[0]);
								dispatchBlock[1] = 1;
								dispatchBlock[2] = app->configuration.size[2];
								//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
								dispatchEnhanced(app, axis, dispatchBlock);

							}
							if (!app->configuration.reorderFourStep) l = app->localFFTPlan->numSupportAxisUploads[0] - 1 - l;
							if (l > 0)
								VkFFTSync(app);
						}
					}
				}*/
				for (uint32_t j = 0; j < app->configuration.numberBatches; j++) {
					for (int l = app->localFFTPlan_inverse->numAxisUploads[1] - 1; l >= 0; l--) {
						if (!app->configuration.reorderFourStep) l = app->localFFTPlan_inverse->numAxisUploads[1] - 1 - l;
						VkFFTAxis* axis = &app->localFFTPlan_inverse->axes[1][l];
						axis->pushConstants.batch = j;
						for (uint32_t i = 0; i < app->configuration.coordinateFeatures; i++) {
							axis->pushConstants.coordinate = i;
#if(VKFFT_BACKEND==0)
							vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
							vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
#endif
							uint32_t dispatchBlock[3];
							dispatchBlock[0] = ceil(localSize0 / (double)axis->axisBlock[0] * app->configuration.size[1] / (double)axis->specializationConstants.fftDim);
							dispatchBlock[1] = 1;
							dispatchBlock[2] = app->configuration.size[2];
							//if (app->configuration.performR2C == 1) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);
							//if (app->configuration.performZeropadding[2]) dispatchBlock[2] = ceil(dispatchBlock[2] / 2.0);
							//if (app->configuration.performZeropaddingInverse[0]) dispatchBlock[0] = ceil(dispatchBlock[0] / 2.0);

							dispatchEnhanced(app, axis, dispatchBlock);

						}
						if (!app->configuration.reorderFourStep) l = app->localFFTPlan_inverse->numAxisUploads[1] - 1 - l;
						VkFFTSync(app);
					}
				}

			}
			//FFT axis 0
			for (uint32_t j = 0; j < app->configuration.numberBatches; j++) {
				for (int l = app->localFFTPlan_inverse->numAxisUploads[0] - 1; l >= 0; l--) {
					if (!app->configuration.reorderFourStep) l = app->localFFTPlan_inverse->numAxisUploads[0] - 1 - l;
					VkFFTAxis* axis = &app->localFFTPlan_inverse->axes[0][l];
					axis->pushConstants.batch = j;
					uint32_t maxCoordinate = ((app->configuration.matrixConvolution) > 1 && (app->configuration.performConvolution) && (app->configuration.FFTdim == 1)) ? 1 : app->configuration.coordinateFeatures;
					for (uint32_t i = 0; i < maxCoordinate; i++) {
						axis->pushConstants.coordinate = i;
#if(VKFFT_BACKEND==0)
						vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
						vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, NULL);
#endif
						uint32_t dispatchBlock[3];
						if (l == 0) {
							if (app->localFFTPlan_inverse->numAxisUploads[0] > 2) {
								dispatchBlock[0] = ceil(ceil(app->configuration.size[0] / axis->specializationConstants.fftDim / (double)axis->axisBlock[1]) / (double)app->localFFTPlan_inverse->axisSplit[0][1]) * app->localFFTPlan_inverse->axisSplit[0][1];
								dispatchBlock[1] = app->configuration.size[1];
							}
							else {
								if (app->localFFTPlan_inverse->numAxisUploads[0] > 1) {
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
						dispatchEnhanced(app, axis, dispatchBlock);
					}
					if (!app->configuration.reorderFourStep) l = app->localFFTPlan_inverse->numAxisUploads[0] - 1 - l;
					VkFFTSync(app);
				}
			}


		}
	}
	static inline void deleteVkFFT(VkFFTApplication* app) {
#if(VKFFT_BACKEND==1)
		if (app->configuration.num_streams > 1) {
			for (uint32_t i = 0; i < app->configuration.num_streams; i++) {
				cudaEventDestroy(app->configuration.stream_event[i]);
			}
			free(app->configuration.stream_event);
		}
#elif(VKFFT_BACKEND==2)
		if (app->configuration.num_streams > 1) {
			for (uint32_t i = 0; i < app->configuration.num_streams; i++) {
				hipEventDestroy(app->configuration.stream_event[i]);
			}
			free(app->configuration.stream_event);
		}
#endif
		if (!app->configuration.userTempBuffer) {
			if (app->configuration.allocateTempBuffer) {
				app->configuration.allocateTempBuffer = 0;
#if(VKFFT_BACKEND==0)
				vkDestroyBuffer(app->configuration.device[0], app->configuration.tempBuffer[0], NULL);
				vkFreeMemory(app->configuration.device[0], app->configuration.tempBufferDeviceMemory, NULL);
#elif(VKFFT_BACKEND==1)
				cudaFree(app->configuration.tempBuffer[0]);
#elif(VKFFT_BACKEND==2)
				hipFree(app->configuration.tempBuffer[0]);
#endif
				free(app->configuration.tempBuffer);
			}
			free(app->configuration.tempBufferSize);
		}
		for (uint32_t i = 0; i < app->configuration.FFTdim; i++) {
			for (uint32_t j = 0; j < app->localFFTPlan->numAxisUploads[i]; j++)
				deleteAxis(app, &app->localFFTPlan->axes[i][j]);
		}

		/*for (uint32_t i = 0; i < app->configuration.FFTdim - 1; i++) {

			if (app->configuration.performR2C) {
				for (uint32_t j = 0; j < app->localFFTPlan->numSupportAxisUploads[i]; j++)
					deleteAxis(app, &app->localFFTPlan->supportAxes[i][j]);
			}
		}*/
		free(app->localFFTPlan);

		for (uint32_t i = 0; i < app->configuration.FFTdim; i++) {
			for (uint32_t j = 0; j < app->localFFTPlan_inverse->numAxisUploads[i]; j++)
				deleteAxis(app, &app->localFFTPlan_inverse->axes[i][j]);
		}
		/*for (uint32_t i = 0; i < app->configuration.FFTdim - 1; i++) {
			if (app->configuration.performR2C) {
				for (uint32_t j = 0; j < app->localFFTPlan_inverse->numSupportAxisUploads[i]; j++)
					deleteAxis(app, &app->localFFTPlan_inverse->supportAxes[i][j]);
			}
		}*/
		free(app->localFFTPlan_inverse);
	}
#ifdef __cplusplus
}
#endif
#endif