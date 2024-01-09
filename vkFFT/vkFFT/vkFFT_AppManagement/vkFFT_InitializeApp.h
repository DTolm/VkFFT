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
#ifndef VKFFT_INITIALIZEAPP_H
#define VKFFT_INITIALIZEAPP_H

#include "vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_Scheduler.h"
#include "vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_RecursiveFFTGenerators.h"

#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT/vkFFT_AppManagement/vkFFT_DeleteApp.h"
#include "vkFFT/vkFFT_PlanManagement/vkFFT_Plans/vkFFT_Plan_FFT.h"
#include "vkFFT/vkFFT_PlanManagement/vkFFT_Plans/vkFFT_Plan_R2C.h"
static inline VkFFTResult initializeBluesteinAutoPadding(VkFFTApplication* app) {
	VkFFTResult resFFT = VKFFT_SUCCESS;
	if (!app->configuration.useCustomBluesteinPaddingPattern) {
		switch (app->configuration.vendorID) {
		case 0x10DE://NVIDIA
			if (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory || app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) {
				app->configuration.autoCustomBluesteinPaddingPattern = 49;
			}
			else {
				app->configuration.autoCustomBluesteinPaddingPattern = 45;
			}
			break;
		default: //have not done a test run for Intel, so everything else uses AMD profile
			if (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory || app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) {
				app->configuration.autoCustomBluesteinPaddingPattern = 54;
			}
			else {
				app->configuration.autoCustomBluesteinPaddingPattern = 29;
			}
			break;
		}
		app->configuration.primeSizes = (pfUINT*)malloc(app->configuration.autoCustomBluesteinPaddingPattern * sizeof(pfUINT));
		if (!app->configuration.primeSizes) return VKFFT_ERROR_MALLOC_FAILED;
		app->configuration.paddedSizes = (pfUINT*)malloc(app->configuration.autoCustomBluesteinPaddingPattern * sizeof(pfUINT));
		if (!app->configuration.paddedSizes) return VKFFT_ERROR_MALLOC_FAILED;
		switch (app->configuration.vendorID) {
		case 0x10DE://Nvidia
			if (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory || app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) {
				app->configuration.primeSizes[0] = 17;
				app->configuration.paddedSizes[0] = 36;
				app->configuration.primeSizes[1] = 19;
				app->configuration.paddedSizes[1] = 40;
				app->configuration.primeSizes[2] = 23;
				app->configuration.paddedSizes[2] = 48;
				app->configuration.primeSizes[3] = 29;
				app->configuration.paddedSizes[3] = 64;
				app->configuration.primeSizes[4] = 34;
				app->configuration.paddedSizes[4] = 70;
				app->configuration.primeSizes[5] = 37;
				app->configuration.paddedSizes[5] = 80;
				app->configuration.primeSizes[6] = 41;
				app->configuration.paddedSizes[6] = 90;
				app->configuration.primeSizes[7] = 46;
				app->configuration.paddedSizes[7] = 96;
				app->configuration.primeSizes[8] = 51;
				app->configuration.paddedSizes[8] = 104;
				app->configuration.primeSizes[9] = 53;
				app->configuration.paddedSizes[9] = 128;
				app->configuration.primeSizes[10] = 67;
				app->configuration.paddedSizes[10] = 144;
				app->configuration.primeSizes[11] = 73;
				app->configuration.paddedSizes[11] = 160;
				app->configuration.primeSizes[12] = 82;
				app->configuration.paddedSizes[12] = 256;
				app->configuration.primeSizes[13] = 129;
				app->configuration.paddedSizes[13] = 288;
				app->configuration.primeSizes[14] = 145;
				app->configuration.paddedSizes[14] = 512;
				app->configuration.primeSizes[15] = 257;
				app->configuration.paddedSizes[15] = 625;
				app->configuration.primeSizes[16] = 314;
				app->configuration.paddedSizes[16] = 750;
				app->configuration.primeSizes[17] = 376;
				app->configuration.paddedSizes[17] = 756;
				app->configuration.primeSizes[18] = 379;
				app->configuration.paddedSizes[18] = 768;
				app->configuration.primeSizes[19] = 386;
				app->configuration.paddedSizes[19] = 1024;
				app->configuration.primeSizes[20] = 513;
				app->configuration.paddedSizes[20] = 1056;
				app->configuration.primeSizes[21] = 529;
				app->configuration.paddedSizes[21] = 1200;
				app->configuration.primeSizes[22] = 601;
				app->configuration.paddedSizes[22] = 1225;
				app->configuration.primeSizes[23] = 614;
				app->configuration.paddedSizes[23] = 1250;
				app->configuration.primeSizes[24] = 626;
				app->configuration.paddedSizes[24] = 1296;
				app->configuration.primeSizes[25] = 649;
				app->configuration.paddedSizes[25] = 1331;
				app->configuration.primeSizes[26] = 667;
				app->configuration.paddedSizes[26] = 1440;
				app->configuration.primeSizes[27] = 721;
				app->configuration.paddedSizes[27] = 1456;
				app->configuration.primeSizes[28] = 730;
				app->configuration.paddedSizes[28] = 1560;
				app->configuration.primeSizes[29] = 781;
				app->configuration.paddedSizes[29] = 2048;
				app->configuration.primeSizes[30] = 1025;
				app->configuration.paddedSizes[30] = 2187;
				app->configuration.primeSizes[31] = 1095;
				app->configuration.paddedSizes[31] = 2304;
				app->configuration.primeSizes[32] = 1153;
				app->configuration.paddedSizes[32] = 2688;
				app->configuration.primeSizes[33] = 1345;
				app->configuration.paddedSizes[33] = 2730;
				app->configuration.primeSizes[34] = 1366;
				app->configuration.paddedSizes[34] = 2925;
				app->configuration.primeSizes[35] = 1464;
				app->configuration.paddedSizes[35] = 3000;
				app->configuration.primeSizes[36] = 1501;
				app->configuration.paddedSizes[36] = 4096;
				app->configuration.primeSizes[37] = 2049;
				app->configuration.paddedSizes[37] = 4368;
				app->configuration.primeSizes[38] = 2185;
				app->configuration.paddedSizes[38] = 4608;
				app->configuration.primeSizes[39] = 2305;
				app->configuration.paddedSizes[39] = 4900;
				app->configuration.primeSizes[40] = 2364;
				app->configuration.paddedSizes[40] = 4900;
				app->configuration.primeSizes[41] = 2451;
				app->configuration.paddedSizes[41] = 5184;
				app->configuration.primeSizes[42] = 2593;
				app->configuration.paddedSizes[42] = 5625;
				app->configuration.primeSizes[43] = 2814;
				app->configuration.paddedSizes[43] = 5760;
				app->configuration.primeSizes[44] = 2881;
				app->configuration.paddedSizes[44] = 6000;
				app->configuration.primeSizes[45] = 3001;
				app->configuration.paddedSizes[45] = 6048;
				app->configuration.primeSizes[46] = 3026;
				app->configuration.paddedSizes[46] = 6144;
				app->configuration.primeSizes[47] = 3073;
				app->configuration.paddedSizes[47] = 6561;
				app->configuration.primeSizes[48] = 3282;
				app->configuration.paddedSizes[48] = 8192;
			}
			else {
				app->configuration.primeSizes[0] = 17;
				app->configuration.paddedSizes[0] = 36;
				app->configuration.primeSizes[1] = 19;
				app->configuration.paddedSizes[1] = 40;
				app->configuration.primeSizes[2] = 23;
				app->configuration.paddedSizes[2] = 48;
				app->configuration.primeSizes[3] = 29;
				app->configuration.paddedSizes[3] = 64;
				app->configuration.primeSizes[4] = 34;
				app->configuration.paddedSizes[4] = 70;
				app->configuration.primeSizes[5] = 37;
				app->configuration.paddedSizes[5] = 80;
				app->configuration.primeSizes[6] = 41;
				app->configuration.paddedSizes[6] = 96;
				app->configuration.primeSizes[7] = 51;
				app->configuration.paddedSizes[7] = 104;
				app->configuration.primeSizes[8] = 53;
				app->configuration.paddedSizes[8] = 112;
				app->configuration.primeSizes[9] = 57;
				app->configuration.paddedSizes[9] = 120;
				app->configuration.primeSizes[10] = 61;
				app->configuration.paddedSizes[10] = 128;
				app->configuration.primeSizes[11] = 67;
				app->configuration.paddedSizes[11] = 144;
				app->configuration.primeSizes[12] = 73;
				app->configuration.paddedSizes[12] = 150;
				app->configuration.primeSizes[13] = 76;
				app->configuration.paddedSizes[13] = 160;
				app->configuration.primeSizes[14] = 82;
				app->configuration.paddedSizes[14] = 256;
				app->configuration.primeSizes[15] = 129;
				app->configuration.paddedSizes[15] = 384;
				app->configuration.primeSizes[16] = 193;
				app->configuration.paddedSizes[16] = 512;
				app->configuration.primeSizes[17] = 257;
				app->configuration.paddedSizes[17] = 567;
				app->configuration.primeSizes[18] = 285;
				app->configuration.paddedSizes[18] = 625;
				app->configuration.primeSizes[19] = 314;
				app->configuration.paddedSizes[19] = 768;
				app->configuration.primeSizes[20] = 386;
				app->configuration.paddedSizes[20] = 832;
				app->configuration.primeSizes[21] = 417;
				app->configuration.paddedSizes[21] = 1024;
				app->configuration.primeSizes[22] = 513;
				app->configuration.paddedSizes[22] = 1152;
				app->configuration.primeSizes[23] = 577;
				app->configuration.paddedSizes[23] = 1200;
				app->configuration.primeSizes[24] = 601;
				app->configuration.paddedSizes[24] = 1296;
				app->configuration.primeSizes[25] = 649;
				app->configuration.paddedSizes[25] = 1536;
				app->configuration.primeSizes[26] = 769;
				app->configuration.paddedSizes[26] = 2048;
				app->configuration.primeSizes[27] = 1025;
				app->configuration.paddedSizes[27] = 2187;
				app->configuration.primeSizes[28] = 1095;
				app->configuration.paddedSizes[28] = 2304;
				app->configuration.primeSizes[29] = 1153;
				app->configuration.paddedSizes[29] = 2500;
				app->configuration.primeSizes[30] = 1251;
				app->configuration.paddedSizes[30] = 2592;
				app->configuration.primeSizes[31] = 1297;
				app->configuration.paddedSizes[31] = 2816;
				app->configuration.primeSizes[32] = 1409;
				app->configuration.paddedSizes[32] = 3072;
				app->configuration.primeSizes[33] = 1537;
				app->configuration.paddedSizes[33] = 4096;
				app->configuration.primeSizes[34] = 2049;
				app->configuration.paddedSizes[34] = 4368;
				app->configuration.primeSizes[35] = 2185;
				app->configuration.paddedSizes[35] = 4563;
				app->configuration.primeSizes[36] = 2283;
				app->configuration.paddedSizes[36] = 4576;
				app->configuration.primeSizes[37] = 2289;
				app->configuration.paddedSizes[37] = 4608;
				app->configuration.primeSizes[38] = 2305;
				app->configuration.paddedSizes[38] = 5184;
				app->configuration.primeSizes[39] = 2593;
				app->configuration.paddedSizes[39] = 5625;
				app->configuration.primeSizes[40] = 2814;
				app->configuration.paddedSizes[40] = 5632;
				app->configuration.primeSizes[41] = 2817;
				app->configuration.paddedSizes[41] = 6000;
				app->configuration.primeSizes[42] = 3001;
				app->configuration.paddedSizes[42] = 6144;
				app->configuration.primeSizes[43] = 3073;
				app->configuration.paddedSizes[43] = 6561;
				app->configuration.primeSizes[44] = 3282;
				app->configuration.paddedSizes[44] = 8192;
			}
			break;
		default: //have not done a test run for Intel, so everything else uses AMD profile
			if (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory || app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) {
				app->configuration.primeSizes[0] = 17;
				app->configuration.paddedSizes[0] = 36;
				app->configuration.primeSizes[1] = 19;
				app->configuration.paddedSizes[1] = 40;
				app->configuration.primeSizes[2] = 23;
				app->configuration.paddedSizes[2] = 56;
				app->configuration.primeSizes[3] = 29;
				app->configuration.paddedSizes[3] = 64;
				app->configuration.primeSizes[4] = 34;
				app->configuration.paddedSizes[4] = 70;
				app->configuration.primeSizes[5] = 37;
				app->configuration.paddedSizes[5] = 78;
				app->configuration.primeSizes[6] = 41;
				app->configuration.paddedSizes[6] = 81;
				app->configuration.primeSizes[7] = 43;
				app->configuration.paddedSizes[7] = 90;
				app->configuration.primeSizes[8] = 46;
				app->configuration.paddedSizes[8] = 125;
				app->configuration.primeSizes[9] = 67;
				app->configuration.paddedSizes[9] = 150;
				app->configuration.primeSizes[10] = 76;
				app->configuration.paddedSizes[10] = 175;
				app->configuration.primeSizes[11] = 89;
				app->configuration.paddedSizes[11] = 189;
				app->configuration.primeSizes[12] = 97;
				app->configuration.paddedSizes[12] = 198;
				app->configuration.primeSizes[13] = 101;
				app->configuration.paddedSizes[13] = 243;
				app->configuration.primeSizes[14] = 123;
				app->configuration.paddedSizes[14] = 256;
				app->configuration.primeSizes[15] = 129;
				app->configuration.paddedSizes[15] = 270;
				app->configuration.primeSizes[16] = 136;
				app->configuration.paddedSizes[16] = 512;
				app->configuration.primeSizes[17] = 257;
				app->configuration.paddedSizes[17] = 625;
				app->configuration.primeSizes[18] = 314;
				app->configuration.paddedSizes[18] = 640;
				app->configuration.primeSizes[19] = 321;
				app->configuration.paddedSizes[19] = 702;
				app->configuration.primeSizes[20] = 353;
				app->configuration.paddedSizes[20] = 750;
				app->configuration.primeSizes[21] = 376;
				app->configuration.paddedSizes[21] = 756;
				app->configuration.primeSizes[22] = 379;
				app->configuration.paddedSizes[22] = 768;
				app->configuration.primeSizes[23] = 386;
				app->configuration.paddedSizes[23] = 875;
				app->configuration.primeSizes[24] = 439;
				app->configuration.paddedSizes[24] = 1024;
				app->configuration.primeSizes[25] = 513;
				app->configuration.paddedSizes[25] = 1296;
				app->configuration.primeSizes[26] = 649;
				app->configuration.paddedSizes[26] = 1300;
				app->configuration.primeSizes[27] = 651;
				app->configuration.paddedSizes[27] = 1323;
				app->configuration.primeSizes[28] = 663;
				app->configuration.paddedSizes[28] = 1344;
				app->configuration.primeSizes[29] = 673;
				app->configuration.paddedSizes[29] = 1512;
				app->configuration.primeSizes[30] = 757;
				app->configuration.paddedSizes[30] = 1792;
				app->configuration.primeSizes[31] = 897;
				app->configuration.paddedSizes[31] = 2016;
				app->configuration.primeSizes[32] = 1009;
				app->configuration.paddedSizes[32] = 2048;
				app->configuration.primeSizes[33] = 1025;
				app->configuration.paddedSizes[33] = 2187;
				app->configuration.primeSizes[34] = 1095;
				app->configuration.paddedSizes[34] = 3136;
				app->configuration.primeSizes[35] = 1569;
				app->configuration.paddedSizes[35] = 3159;
				app->configuration.primeSizes[36] = 1581;
				app->configuration.paddedSizes[36] = 3430;
				app->configuration.primeSizes[37] = 1717;
				app->configuration.paddedSizes[37] = 3584;
				app->configuration.primeSizes[38] = 1793;
				app->configuration.paddedSizes[38] = 4096;
				app->configuration.primeSizes[39] = 2049;
				app->configuration.paddedSizes[39] = 4224;
				app->configuration.primeSizes[40] = 2113;
				app->configuration.paddedSizes[40] = 4375;
				app->configuration.primeSizes[41] = 2189;
				app->configuration.paddedSizes[41] = 4480;
				app->configuration.primeSizes[42] = 2241;
				app->configuration.paddedSizes[42] = 4704;
				app->configuration.primeSizes[43] = 2353;
				app->configuration.paddedSizes[43] = 4928;
				app->configuration.primeSizes[44] = 2465;
				app->configuration.paddedSizes[44] = 4992;
				app->configuration.primeSizes[45] = 2497;
				app->configuration.paddedSizes[45] = 5005;
				app->configuration.primeSizes[46] = 2504;
				app->configuration.paddedSizes[46] = 5103;
				app->configuration.primeSizes[47] = 2553;
				app->configuration.paddedSizes[47] = 5376;
				app->configuration.primeSizes[48] = 2689;
				app->configuration.paddedSizes[48] = 5632;
				app->configuration.primeSizes[49] = 2817;
				app->configuration.paddedSizes[49] = 5824;
				app->configuration.primeSizes[50] = 2913;
				app->configuration.paddedSizes[50] = 6048;
				app->configuration.primeSizes[51] = 3026;
				app->configuration.paddedSizes[51] = 6144;
				app->configuration.primeSizes[52] = 3073;
				app->configuration.paddedSizes[52] = 6875;
				app->configuration.primeSizes[53] = 3439;
				app->configuration.paddedSizes[53] = 8192;
			}
			else {
				app->configuration.primeSizes[0] = 17;
				app->configuration.paddedSizes[0] = 36;
				app->configuration.primeSizes[1] = 19;
				app->configuration.paddedSizes[1] = 42;
				app->configuration.primeSizes[2] = 23;
				app->configuration.paddedSizes[2] = 64;
				app->configuration.primeSizes[3] = 34;
				app->configuration.paddedSizes[3] = 81;
				app->configuration.primeSizes[4] = 43;
				app->configuration.paddedSizes[4] = 88;
				app->configuration.primeSizes[5] = 46;
				app->configuration.paddedSizes[5] = 125;
				app->configuration.primeSizes[6] = 67;
				app->configuration.paddedSizes[6] = 150;
				app->configuration.primeSizes[7] = 76;
				app->configuration.paddedSizes[7] = 162;
				app->configuration.primeSizes[8] = 82;
				app->configuration.paddedSizes[8] = 175;
				app->configuration.primeSizes[9] = 89;
				app->configuration.paddedSizes[9] = 256;
				app->configuration.primeSizes[10] = 129;
				app->configuration.paddedSizes[10] = 512;
				app->configuration.primeSizes[11] = 257;
				app->configuration.paddedSizes[11] = 625;
				app->configuration.primeSizes[12] = 314;
				app->configuration.paddedSizes[12] = 768;
				app->configuration.primeSizes[13] = 386;
				app->configuration.paddedSizes[13] = 1024;
				app->configuration.primeSizes[14] = 513;
				app->configuration.paddedSizes[14] = 1296;
				app->configuration.primeSizes[15] = 649;
				app->configuration.paddedSizes[15] = 2048;
				app->configuration.primeSizes[16] = 1025;
				app->configuration.paddedSizes[16] = 2187;
				app->configuration.primeSizes[17] = 1095;
				app->configuration.paddedSizes[17] = 2304;
				app->configuration.primeSizes[18] = 1153;
				app->configuration.paddedSizes[18] = 2500;
				app->configuration.primeSizes[19] = 1251;
				app->configuration.paddedSizes[19] = 2592;
				app->configuration.primeSizes[20] = 1297;
				app->configuration.paddedSizes[20] = 3072;
				app->configuration.primeSizes[21] = 1537;
				app->configuration.paddedSizes[21] = 3125;
				app->configuration.primeSizes[22] = 1564;
				app->configuration.paddedSizes[22] = 3136;
				app->configuration.primeSizes[23] = 1569;
				app->configuration.paddedSizes[23] = 4096;
				app->configuration.primeSizes[24] = 2049;
				app->configuration.paddedSizes[24] = 4375;
				app->configuration.primeSizes[25] = 2189;
				app->configuration.paddedSizes[25] = 4608;
				app->configuration.primeSizes[26] = 2305;
				app->configuration.paddedSizes[26] = 5184;
				app->configuration.primeSizes[27] = 2593;
				app->configuration.paddedSizes[27] = 6561;
				app->configuration.primeSizes[28] = 3282;
				app->configuration.paddedSizes[28] = 8192;
			}
			break;
		}
	}
	return resFFT;
}
static inline VkFFTResult setConfigurationVkFFT(VkFFTApplication* app, VkFFTConfiguration inputLaunchConfiguration)  {
	VkFFTResult resFFT = VKFFT_SUCCESS;
    //app->configuration = {};// inputLaunchConfiguration;
	if (inputLaunchConfiguration.doublePrecision != 0)	app->configuration.doublePrecision = inputLaunchConfiguration.doublePrecision;
	if (inputLaunchConfiguration.doublePrecisionFloatMemory != 0)	app->configuration.doublePrecisionFloatMemory = inputLaunchConfiguration.doublePrecisionFloatMemory;

	if (inputLaunchConfiguration.quadDoubleDoublePrecision != 0)	app->configuration.quadDoubleDoublePrecision = inputLaunchConfiguration.quadDoubleDoublePrecision;
	if (inputLaunchConfiguration.quadDoubleDoublePrecisionDoubleMemory != 0)	app->configuration.quadDoubleDoublePrecisionDoubleMemory = inputLaunchConfiguration.quadDoubleDoublePrecisionDoubleMemory;
	
	if (inputLaunchConfiguration.halfPrecision != 0)	app->configuration.halfPrecision = inputLaunchConfiguration.halfPrecision;
	if (inputLaunchConfiguration.halfPrecisionMemoryOnly != 0)	app->configuration.halfPrecisionMemoryOnly = inputLaunchConfiguration.halfPrecisionMemoryOnly;
	if (inputLaunchConfiguration.useCustomBluesteinPaddingPattern != 0) {
		app->configuration.useCustomBluesteinPaddingPattern = inputLaunchConfiguration.useCustomBluesteinPaddingPattern;
		app->configuration.primeSizes = inputLaunchConfiguration.primeSizes;
		if (!app->configuration.primeSizes) return VKFFT_ERROR_EMPTY_useCustomBluesteinPaddingPattern_arrays;
		app->configuration.paddedSizes = inputLaunchConfiguration.paddedSizes;
		if (!app->configuration.paddedSizes) return VKFFT_ERROR_EMPTY_useCustomBluesteinPaddingPattern_arrays;
	}
	//set device parameters
#if(VKFFT_BACKEND==0)
	if (!inputLaunchConfiguration.isCompilerInitialized) {
		if (!app->configuration.isCompilerInitialized) {
			int resGlslangInitialize = glslang_initialize_process();
			if (!resGlslangInitialize) return VKFFT_ERROR_FAILED_TO_INITIALIZE;
			app->configuration.isCompilerInitialized = 1;
		}
	}
	if (inputLaunchConfiguration.physicalDevice == 0) {
		deleteVkFFT(app);
		return VKFFT_ERROR_INVALID_PHYSICAL_DEVICE;
	}
	app->configuration.physicalDevice = inputLaunchConfiguration.physicalDevice;
	if (inputLaunchConfiguration.device == 0) {
		deleteVkFFT(app);
		return VKFFT_ERROR_INVALID_DEVICE;
	}
	app->configuration.device = inputLaunchConfiguration.device;
	if (inputLaunchConfiguration.queue == 0) {
		deleteVkFFT(app);
		return VKFFT_ERROR_INVALID_QUEUE;
	}
	app->configuration.queue = inputLaunchConfiguration.queue;
	if (inputLaunchConfiguration.commandPool == 0) {
		deleteVkFFT(app);
		return VKFFT_ERROR_INVALID_COMMAND_POOL;
	}
	app->configuration.commandPool = inputLaunchConfiguration.commandPool;
	if (inputLaunchConfiguration.fence == 0) {
		deleteVkFFT(app);
		return VKFFT_ERROR_INVALID_FENCE;
	}
	app->configuration.fence = inputLaunchConfiguration.fence;

	VkPhysicalDeviceProperties physicalDeviceProperties = { 0 };
	vkGetPhysicalDeviceProperties(app->configuration.physicalDevice[0], &physicalDeviceProperties);
	app->configuration.maxThreadsNum = physicalDeviceProperties.limits.maxComputeWorkGroupInvocations;
	if (physicalDeviceProperties.vendorID == 0x8086) app->configuration.maxThreadsNum = 256; //Intel fix
	app->configuration.maxComputeWorkGroupCount[0] = physicalDeviceProperties.limits.maxComputeWorkGroupCount[0];
	app->configuration.maxComputeWorkGroupCount[1] = physicalDeviceProperties.limits.maxComputeWorkGroupCount[1];
	app->configuration.maxComputeWorkGroupCount[2] = physicalDeviceProperties.limits.maxComputeWorkGroupCount[2];
	app->configuration.maxComputeWorkGroupSize[0] = physicalDeviceProperties.limits.maxComputeWorkGroupSize[0];
	app->configuration.maxComputeWorkGroupSize[1] = physicalDeviceProperties.limits.maxComputeWorkGroupSize[1];
	app->configuration.maxComputeWorkGroupSize[2] = physicalDeviceProperties.limits.maxComputeWorkGroupSize[2];
	//if ((physicalDeviceProperties.vendorID == 0x8086) && (!app->configuration.doublePrecision) && (!app->configuration.doublePrecisionFloatMemory)) app->configuration.halfThreads = 1;
	app->configuration.sharedMemorySize = physicalDeviceProperties.limits.maxComputeSharedMemorySize;
	app->configuration.vendorID = physicalDeviceProperties.vendorID;
	if (inputLaunchConfiguration.pipelineCache != 0)	app->configuration.pipelineCache = inputLaunchConfiguration.pipelineCache;
	app->configuration.useRaderUintLUT = 1;
	switch (physicalDeviceProperties.vendorID) {
	case 0x10DE://NVIDIA
		app->configuration.coalescedMemory = (app->configuration.halfPrecision) ? 64 : 32;//the coalesced memory is equal to 32 bytes between L2 and VRAM.
		app->configuration.useLUT = (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory || app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) ? 1 : -1;
		app->configuration.warpSize = 32;
		app->configuration.registerBoostNonPow2 = 0;
		app->configuration.registerBoost = 4;
		app->configuration.registerBoost4Step = 1;
		app->configuration.swapTo3Stage4Step = (app->configuration.doublePrecision) ? 4194305 : 4194305;
		break;
	case 0x8086://INTEL
		app->configuration.coalescedMemory = (app->configuration.halfPrecision) ? 128 : 64;
		app->configuration.useLUT = 1;
		app->configuration.warpSize = 32;
		app->configuration.registerBoostNonPow2 = 0;
		app->configuration.registerBoost = (physicalDeviceProperties.limits.maxComputeSharedMemorySize >= 65536) ? 1 : 2;
		app->configuration.registerBoost4Step = 1;
		app->configuration.swapTo3Stage4Step = (app->configuration.doublePrecision || app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) ? 262144 : 524288;
		break;
	case 0x1002://AMD
		app->configuration.coalescedMemory = (app->configuration.halfPrecision) ? 64 : 32;
		app->configuration.useLUT = (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory || app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) ? 1 : -1;
		app->configuration.warpSize = 64;
		app->configuration.registerBoostNonPow2 = 0;
		app->configuration.registerBoost = (physicalDeviceProperties.limits.maxComputeSharedMemorySize >= 65536) ? 2 : 4;
		app->configuration.registerBoost4Step = 1;
		app->configuration.swapTo3Stage4Step = (app->configuration.doublePrecision || app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) ? 262144 : 524288;
		break;
	default:
		app->configuration.coalescedMemory = (app->configuration.halfPrecision) ? 128 : 64;
		app->configuration.useLUT = (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory || app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) ? 1 : -1;
		app->configuration.warpSize = 32;
		app->configuration.registerBoostNonPow2 = 0;
		app->configuration.registerBoost = 1;
		app->configuration.registerBoost4Step = 1;
		app->configuration.swapTo3Stage4Step = (app->configuration.doublePrecision || app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) ? 262144 : 524288;
		break;
	}
#elif(VKFFT_BACKEND==1)
	CUresult res = CUDA_SUCCESS;
	cudaError_t res_t = cudaSuccess;
	if (inputLaunchConfiguration.device == 0) {
		deleteVkFFT(app);
		return VKFFT_ERROR_INVALID_DEVICE;
	}
	app->configuration.device = inputLaunchConfiguration.device;
	if (inputLaunchConfiguration.num_streams != 0)	app->configuration.num_streams = inputLaunchConfiguration.num_streams;
	if (inputLaunchConfiguration.stream != 0)	app->configuration.stream = inputLaunchConfiguration.stream;
	app->configuration.streamID = 0;
	int value = 0;
	res = cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, app->configuration.device[0]);
	if (res != CUDA_SUCCESS) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.computeCapabilityMajor = value;

	res = cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, app->configuration.device[0]);
	if (res != CUDA_SUCCESS) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.computeCapabilityMinor = value;

	res = cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, app->configuration.device[0]);
	if (res != CUDA_SUCCESS) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.maxThreadsNum = value;

	res = cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, app->configuration.device[0]);
	if (res != CUDA_SUCCESS) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.maxComputeWorkGroupCount[0] = value;
	res = cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, app->configuration.device[0]);
	if (res != CUDA_SUCCESS) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.maxComputeWorkGroupCount[1] = value;
	res = cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, app->configuration.device[0]);
	if (res != CUDA_SUCCESS) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.maxComputeWorkGroupCount[2] = value;
	res = cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, app->configuration.device[0]);
	if (res != CUDA_SUCCESS) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.maxComputeWorkGroupSize[0] = value;
	res = cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, app->configuration.device[0]);
	if (res != CUDA_SUCCESS) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.maxComputeWorkGroupSize[1] = value;
	res = cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, app->configuration.device[0]);
	if (res != CUDA_SUCCESS) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.maxComputeWorkGroupSize[2] = value;
	res = cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, app->configuration.device[0]);
	if (res != CUDA_SUCCESS) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.sharedMemorySizeStatic = value;
	res = cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, app->configuration.device[0]);
	if (res != CUDA_SUCCESS) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.sharedMemorySize = value;// (value > 65536) ? 65536 : value;
	res = cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_WARP_SIZE, app->configuration.device[0]);
	if (res != CUDA_SUCCESS) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.warpSize = value;
	res = cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO, app->configuration.device[0]);
	if (res != CUDA_SUCCESS) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.useLUT_4step = (value <= 4) ? -1 : 1;
	//we don't need this in CUDA
	app->configuration.useRaderUintLUT = 0;
	if (app->configuration.num_streams > 1) {
		app->configuration.stream_event = (cudaEvent_t*)malloc(app->configuration.num_streams * sizeof(cudaEvent_t));
		if (!app->configuration.stream_event) {
			deleteVkFFT(app);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
		for (pfUINT i = 0; i < app->configuration.num_streams; i++) {
			res_t = cudaEventCreate(&app->configuration.stream_event[i]);
			if (res_t != cudaSuccess) {
				deleteVkFFT(app);
				return VKFFT_ERROR_FAILED_TO_CREATE_EVENT;
			}
		}
	}

	app->configuration.coalescedMemory = (app->configuration.halfPrecision) ? 64 : 32;//the coalesced memory is equal to 32 bytes between L2 and VRAM.
	app->configuration.useLUT = (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory || app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) ? 1 : -1;
	app->configuration.registerBoostNonPow2 = 0;
	app->configuration.registerBoost = 1;
	app->configuration.registerBoost4Step = 1;
	app->configuration.swapTo3Stage4Step = (app->configuration.doublePrecision || app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) ? 4194305 : 4194305;
	app->configuration.vendorID = 0x10DE;
#elif(VKFFT_BACKEND==2)
	hipError_t res = hipSuccess;
	if (inputLaunchConfiguration.device == 0) {
		deleteVkFFT(app);
		return VKFFT_ERROR_INVALID_DEVICE;
	}
	app->configuration.device = inputLaunchConfiguration.device;
	if (inputLaunchConfiguration.num_streams != 0)	app->configuration.num_streams = inputLaunchConfiguration.num_streams;
	if (inputLaunchConfiguration.stream != 0)	app->configuration.stream = inputLaunchConfiguration.stream;
	app->configuration.streamID = 0;
	int value = 0;
	res = hipDeviceGetAttribute(&value, hipDeviceAttributeComputeCapabilityMajor, app->configuration.device[0]);
	if (res != hipSuccess) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.computeCapabilityMajor = value;

	res = hipDeviceGetAttribute(&value, hipDeviceAttributeComputeCapabilityMinor, app->configuration.device[0]);
	if (res != hipSuccess) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.computeCapabilityMinor = value;

	res = hipDeviceGetAttribute(&value, hipDeviceAttributeMaxThreadsPerBlock, app->configuration.device[0]);
	if (res != hipSuccess) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.maxThreadsNum = value;

	res = hipDeviceGetAttribute(&value, hipDeviceAttributeMaxGridDimX, app->configuration.device[0]);
	if (res != hipSuccess) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.maxComputeWorkGroupCount[0] = value;
	res = hipDeviceGetAttribute(&value, hipDeviceAttributeMaxGridDimY, app->configuration.device[0]);
	if (res != hipSuccess) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.maxComputeWorkGroupCount[1] = value;
	res = hipDeviceGetAttribute(&value, hipDeviceAttributeMaxGridDimZ, app->configuration.device[0]);
	if (res != hipSuccess) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.maxComputeWorkGroupCount[2] = value;
	res = hipDeviceGetAttribute(&value, hipDeviceAttributeMaxBlockDimX, app->configuration.device[0]);
	if (res != hipSuccess) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.maxComputeWorkGroupSize[0] = value;
	res = hipDeviceGetAttribute(&value, hipDeviceAttributeMaxBlockDimY, app->configuration.device[0]);
	if (res != hipSuccess) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.maxComputeWorkGroupSize[1] = value;
	res = hipDeviceGetAttribute(&value, hipDeviceAttributeMaxBlockDimZ, app->configuration.device[0]);
	if (res != hipSuccess) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.maxComputeWorkGroupSize[2] = value;
	res = hipDeviceGetAttribute(&value, hipDeviceAttributeMaxSharedMemoryPerBlock, app->configuration.device[0]);
	if (res != hipSuccess) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.sharedMemorySizeStatic = value;
	//hipDeviceGetAttribute(&value, hipDeviceAttributeMaxSharedMemoryPerBlockOptin, app->configuration.device[0]);
	app->configuration.sharedMemorySize = value;// (value > 65536) ? 65536 : value;
	res = hipDeviceGetAttribute(&value, hipDeviceAttributeWarpSize, app->configuration.device[0]);
	if (res != hipSuccess) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.warpSize = value;
	app->configuration.useRaderUintLUT = 0;
	if (app->configuration.num_streams > 1) {
		app->configuration.stream_event = (hipEvent_t*)malloc(app->configuration.num_streams * sizeof(hipEvent_t));
		if (!app->configuration.stream_event) {
			deleteVkFFT(app);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
		for (pfUINT i = 0; i < app->configuration.num_streams; i++) {
			res = hipEventCreate(&app->configuration.stream_event[i]);
			if (res != hipSuccess) {
				deleteVkFFT(app);
				return VKFFT_ERROR_FAILED_TO_CREATE_EVENT;
			}
		}
	}
	app->configuration.coalescedMemory = (app->configuration.halfPrecision) ? 64 : 32;
	app->configuration.useLUT = (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory || app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) ? 1 : -1;
	app->configuration.useLUT_4step = -1;
	app->configuration.registerBoostNonPow2 = 0;
	app->configuration.registerBoost = 1;
	app->configuration.registerBoost4Step = 1;
	app->configuration.swapTo3Stage4Step = (app->configuration.doublePrecision || app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) ? 1048576 : 2097152;
	app->configuration.vendorID = 0x1002;
#elif(VKFFT_BACKEND==3)
	cl_int res = 0;
	if (inputLaunchConfiguration.device == 0) {
		deleteVkFFT(app);
		return VKFFT_ERROR_INVALID_DEVICE;
	}
	app->configuration.device = inputLaunchConfiguration.device;
	if (inputLaunchConfiguration.context == 0) {
		deleteVkFFT(app);
		return VKFFT_ERROR_INVALID_CONTEXT;
	}
	app->configuration.context = inputLaunchConfiguration.context;
	cl_uint vendorID;
	size_t value_int64;
	cl_uint value_cl_uint;
	res = clGetDeviceInfo(app->configuration.device[0], CL_DEVICE_VENDOR_ID, sizeof(cl_int), &vendorID, 0);
	if (res != 0) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	res = clGetDeviceInfo(app->configuration.device[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &value_int64, 0);
	if (res != 0) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.maxThreadsNum = value_int64;

	res = clGetDeviceInfo(app->configuration.device[0], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &value_cl_uint, 0);
	if (res != 0) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	size_t* dims = (size_t*)malloc(sizeof(size_t) * value_cl_uint);
	if (dims) {
		res = clGetDeviceInfo(app->configuration.device[0], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * value_cl_uint, dims, 0);
		if (res != 0) {
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
		}
		app->configuration.maxComputeWorkGroupSize[0] = dims[0];
		app->configuration.maxComputeWorkGroupSize[1] = dims[1];
		app->configuration.maxComputeWorkGroupSize[2] = dims[2];
		free(dims);
		dims = 0;
	}
	else {
		deleteVkFFT(app);
		return VKFFT_ERROR_MALLOC_FAILED;
	}
	app->configuration.maxComputeWorkGroupCount[0] = UINT64_MAX;
	app->configuration.maxComputeWorkGroupCount[1] = UINT64_MAX;
	app->configuration.maxComputeWorkGroupCount[2] = UINT64_MAX;
	//if ((vendorID == 0x8086) && (!app->configuration.doublePrecision) && (!app->configuration.doublePrecisionFloatMemory)) app->configuration.halfThreads = 1;
	cl_ulong sharedMemorySize;
	res = clGetDeviceInfo(app->configuration.device[0], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &sharedMemorySize, 0);
	if (res != 0) {
		deleteVkFFT(app);
		return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.sharedMemorySize = sharedMemorySize;
	app->configuration.vendorID = vendorID;
	app->configuration.useRaderUintLUT = 1;
	switch (vendorID) {
	case 0x10DE://NVIDIA
		app->configuration.coalescedMemory = (app->configuration.halfPrecision) ? 64 : 32;//the coalesced memory is equal to 32 bytes between L2 and VRAM.
		app->configuration.useLUT = (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory || app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) ? 1 : -1;
		app->configuration.warpSize = 32;
		app->configuration.registerBoostNonPow2 = 0;
		app->configuration.registerBoost = 4;
		app->configuration.registerBoost4Step = 1;
		app->configuration.swapTo3Stage4Step = (app->configuration.doublePrecision || app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) ? 4194305 : 4194305;
		app->configuration.sharedMemorySize -= 0x10;//reserved by system
		break;
	case 0x8086://INTEL
		app->configuration.coalescedMemory = (app->configuration.halfPrecision) ? 128 : 64;
		app->configuration.useLUT = 1;
		app->configuration.warpSize = 32;
		app->configuration.registerBoostNonPow2 = 0;
		app->configuration.registerBoost = (sharedMemorySize >= 65536) ? 1 : 2;
		app->configuration.registerBoost4Step = 1;
		app->configuration.swapTo3Stage4Step = (app->configuration.doublePrecision || app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) ? 262144 : 524288;
		break;
	case 0x1002://AMD
		app->configuration.coalescedMemory = (app->configuration.halfPrecision) ? 64 : 32;
		app->configuration.useLUT = (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory || app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) ? 1 : -1;
		app->configuration.warpSize = 64;
		app->configuration.registerBoostNonPow2 = 0;
		app->configuration.registerBoost = (sharedMemorySize >= 65536) ? 2 : 4;
		app->configuration.registerBoost4Step = 1;
		app->configuration.swapTo3Stage4Step = (app->configuration.doublePrecision || app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) ? 262144 : 524288;
		break;
	default:
		app->configuration.coalescedMemory = (app->configuration.halfPrecision) ? 128 : 64;
		app->configuration.useLUT = (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory || app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) ? 1 : -1;
		app->configuration.warpSize = 32;
		app->configuration.registerBoostNonPow2 = 0;
		app->configuration.registerBoost = 1;
		app->configuration.registerBoost4Step = 1;
		app->configuration.swapTo3Stage4Step = (app->configuration.doublePrecision || app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) ? 262144 : 524288;
		break;
	}
#elif(VKFFT_BACKEND==4)
	ze_result_t res = ZE_RESULT_SUCCESS;
	if (inputLaunchConfiguration.device == 0) {
		deleteVkFFT(app);
		return VKFFT_ERROR_INVALID_DEVICE;
	}
	app->configuration.device = inputLaunchConfiguration.device;
	if (inputLaunchConfiguration.context == 0) {
		deleteVkFFT(app);
		return VKFFT_ERROR_INVALID_CONTEXT;
	}
	app->configuration.context = inputLaunchConfiguration.context;
	if (inputLaunchConfiguration.commandQueue == 0) {
		deleteVkFFT(app);
		return VKFFT_ERROR_INVALID_QUEUE;
	}
	app->configuration.commandQueue = inputLaunchConfiguration.commandQueue;
	app->configuration.commandQueueID = inputLaunchConfiguration.commandQueueID;
	ze_device_properties_t device_properties;
	ze_device_compute_properties_t compute_properties;
	res = zeDeviceGetProperties(app->configuration.device[0], &device_properties);
	if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	res = zeDeviceGetComputeProperties(app->configuration.device[0], &compute_properties);
	if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	uint32_t vendorID = device_properties.vendorId;
	app->configuration.maxThreadsNum = compute_properties.maxTotalGroupSize;
	app->configuration.maxComputeWorkGroupSize[0] = compute_properties.maxGroupSizeX;
	app->configuration.maxComputeWorkGroupSize[1] = compute_properties.maxGroupSizeY;
	app->configuration.maxComputeWorkGroupSize[2] = compute_properties.maxGroupSizeZ;

	app->configuration.maxComputeWorkGroupCount[0] = compute_properties.maxGroupCountX;
	app->configuration.maxComputeWorkGroupCount[1] = compute_properties.maxGroupCountY;
	app->configuration.maxComputeWorkGroupCount[2] = compute_properties.maxGroupCountZ;
	//if ((vendorID == 0x8086) && (!app->configuration.doublePrecision) && (!app->configuration.doublePrecisionFloatMemory)) app->configuration.halfThreads = 1;
	app->configuration.sharedMemorySize = compute_properties.maxSharedLocalMemory;

	app->configuration.coalescedMemory = (app->configuration.halfPrecision) ? 128 : 64;
	app->configuration.useLUT = 1;
	app->configuration.warpSize = device_properties.physicalEUSimdWidth;
	app->configuration.registerBoostNonPow2 = 0;
	app->configuration.registerBoost = (app->configuration.sharedMemorySize >= 65536) ? 1 : 2;
	app->configuration.registerBoost4Step = 1;
	app->configuration.swapTo3Stage4Step = (app->configuration.doublePrecision || app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) ? 262144 : 524288;
	app->configuration.vendorID = 0x8086;
	app->configuration.useRaderUintLUT = 1;
#elif(VKFFT_BACKEND==5)
	if (inputLaunchConfiguration.device == 0) {
		deleteVkFFT(app);
		return VKFFT_ERROR_INVALID_DEVICE;
	}
	app->configuration.device = inputLaunchConfiguration.device;

	if (inputLaunchConfiguration.queue == 0) {
		deleteVkFFT(app);
		return VKFFT_ERROR_INVALID_QUEUE;
	}
	app->configuration.queue = inputLaunchConfiguration.queue;

	const char dummy_kernel[50] = "kernel void VkFFT_dummy (){}";
	const char function_name[20] = "VkFFT_dummy";

	NS::Error* error;
	MTL::CompileOptions* compileOptions = MTL::CompileOptions::alloc();
	NS::String* str_code = NS::String::string(dummy_kernel, NS::UTF8StringEncoding);
	MTL::Library* dummy_library = app->configuration.device->newLibrary(str_code, compileOptions, &error);
	NS::String* str_name = NS::String::string(function_name, NS::UTF8StringEncoding);
	MTL::Function* function = dummy_library->newFunction(str_name);
	MTL::ComputePipelineState* dummy_state = app->configuration.device->newComputePipelineState(function, &error);

	MTL::Size size = app->configuration.device->maxThreadsPerThreadgroup();
	app->configuration.maxThreadsNum = dummy_state->maxTotalThreadsPerThreadgroup();

	app->configuration.maxComputeWorkGroupSize[0] = size.width;
	app->configuration.maxComputeWorkGroupSize[1] = size.height;
	app->configuration.maxComputeWorkGroupSize[2] = size.depth;

	if (app->configuration.maxThreadsNum > 256) {
		app->configuration.maxThreadsNum = 256;

		app->configuration.maxComputeWorkGroupSize[0] = 256;
		app->configuration.maxComputeWorkGroupSize[1] = 256;
		app->configuration.maxComputeWorkGroupSize[2] = 256;
		//The dummy kernel approach (above) does not work for some DCT-IV kernels (like 256x256x256). They refuse to have more than 256 threads. I will just force OpenCL thread limits for now.
	}

	app->configuration.maxComputeWorkGroupCount[0] = -1;
	app->configuration.maxComputeWorkGroupCount[1] = -1;
	app->configuration.maxComputeWorkGroupCount[2] = -1;

	app->configuration.sharedMemorySizeStatic = app->configuration.device->maxThreadgroupMemoryLength();
	app->configuration.sharedMemorySize = app->configuration.device->maxThreadgroupMemoryLength();

	app->configuration.warpSize = dummy_state->threadExecutionWidth();

	app->configuration.useRaderUintLUT = 1;

	app->configuration.coalescedMemory = (app->configuration.halfPrecision) ? 128 : 64;//the coalesced memory is equal to 64 bytes between L2 and VRAM.
	app->configuration.useLUT = (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory || app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) ? 1 : -1;
	app->configuration.registerBoostNonPow2 = 0;
	app->configuration.registerBoost = 1;
	app->configuration.registerBoost4Step = 1;
	app->configuration.swapTo3Stage4Step = (app->configuration.doublePrecision || app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) ? 262144 : 524288;
	app->configuration.vendorID = 0x1027f00;

	dummy_state->release();
	function->release();
	str_name->release();
	dummy_library->release();
	str_code->release();
	compileOptions->release();
#endif

	resFFT = initializeBluesteinAutoPadding(app);
	if (resFFT != VKFFT_SUCCESS) {
		deleteVkFFT(app);
		return resFFT;
	}
	//set main parameters:
	if (inputLaunchConfiguration.FFTdim == 0) {
		deleteVkFFT(app);
		return VKFFT_ERROR_EMPTY_FFTdim;
	}
    if (inputLaunchConfiguration.FFTdim > VKFFT_MAX_FFT_DIMENSIONS) {
        deleteVkFFT(app);
        return VKFFT_ERROR_FFTdim_GT_MAX_FFT_DIMENSIONS;
    }
    
	app->configuration.FFTdim = inputLaunchConfiguration.FFTdim;
	if (inputLaunchConfiguration.size[0] == 0) {
		deleteVkFFT(app);
		return VKFFT_ERROR_EMPTY_size;
	}
	app->configuration.isInputFormatted = inputLaunchConfiguration.isInputFormatted;
	app->configuration.isOutputFormatted = inputLaunchConfiguration.isOutputFormatted;
	
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
		if (inputLaunchConfiguration.performR2C && (!app->configuration.isInputFormatted))
			app->configuration.inputBufferStride[0] = app->configuration.size[0] + 2;
		else
			app->configuration.inputBufferStride[0] = app->configuration.size[0];
	}
	else
		app->configuration.inputBufferStride[0] = inputLaunchConfiguration.inputBufferStride[0];

	if (inputLaunchConfiguration.outputBufferStride[0] == 0) {
		if (inputLaunchConfiguration.performR2C && (!app->configuration.isOutputFormatted))
			app->configuration.outputBufferStride[0] = app->configuration.size[0] + 2;
		else
			app->configuration.outputBufferStride[0] = app->configuration.size[0];
	}
	else
		app->configuration.outputBufferStride[0] = inputLaunchConfiguration.outputBufferStride[0];
	for (pfUINT i = 1; i < VKFFT_MAX_FFT_DIMENSIONS; i++) {
		if (inputLaunchConfiguration.size[i] == 0)
			app->configuration.size[i] = 1;
		else
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

	app->configuration.performConvolution = inputLaunchConfiguration.performConvolution;

	if (inputLaunchConfiguration.bufferNum == 0)	app->configuration.bufferNum = 1;
	else app->configuration.bufferNum = inputLaunchConfiguration.bufferNum;
#if(VKFFT_BACKEND==0) 
	if (inputLaunchConfiguration.bufferSize == 0) {
		deleteVkFFT(app);
		return VKFFT_ERROR_EMPTY_bufferSize;
	}
#endif
	app->configuration.bufferSize = inputLaunchConfiguration.bufferSize;
	if (app->configuration.bufferSize != 0) {
		for (pfUINT i = 0; i < app->configuration.bufferNum; i++) {
			if (app->configuration.bufferSize[i] == 0) {
				deleteVkFFT(app);
				return VKFFT_ERROR_EMPTY_bufferSize;
			}
		}
	}
	app->configuration.buffer = inputLaunchConfiguration.buffer;

	if (inputLaunchConfiguration.userTempBuffer != 0)	app->configuration.userTempBuffer = inputLaunchConfiguration.userTempBuffer;

	if (app->configuration.userTempBuffer != 0) {
		if (inputLaunchConfiguration.tempBufferNum == 0)	app->configuration.tempBufferNum = 1;
		else app->configuration.tempBufferNum = inputLaunchConfiguration.tempBufferNum;
#if(VKFFT_BACKEND==0) 
		if (inputLaunchConfiguration.tempBufferSize == 0) {
			deleteVkFFT(app);
			return VKFFT_ERROR_EMPTY_tempBufferSize;
		}
#endif
		app->configuration.tempBufferSize = inputLaunchConfiguration.tempBufferSize;
		if (app->configuration.tempBufferSize != 0) {
			for (pfUINT i = 0; i < app->configuration.tempBufferNum; i++) {
				if (app->configuration.tempBufferSize[i] == 0) {
					deleteVkFFT(app);
					return VKFFT_ERROR_EMPTY_tempBufferSize;
				}
			}
		}
		app->configuration.tempBuffer = inputLaunchConfiguration.tempBuffer;
	}
	else {
		app->configuration.tempBufferNum = 1;
		app->configuration.tempBufferSize = (pfUINT*)malloc(sizeof(pfUINT));
		if (!app->configuration.tempBufferSize) {
			deleteVkFFT(app);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
		app->configuration.tempBufferSize[0] = 0;

	}

	if (app->configuration.isInputFormatted) {
		if (inputLaunchConfiguration.inputBufferNum == 0)	app->configuration.inputBufferNum = 1;
		else app->configuration.inputBufferNum = inputLaunchConfiguration.inputBufferNum;
#if(VKFFT_BACKEND==0) 
		if (inputLaunchConfiguration.inputBufferSize == 0) {
			deleteVkFFT(app);
			return VKFFT_ERROR_EMPTY_inputBufferSize;
		}
#endif
		app->configuration.inputBufferSize = inputLaunchConfiguration.inputBufferSize;
		if (app->configuration.inputBufferSize != 0) {
			for (pfUINT i = 0; i < app->configuration.inputBufferNum; i++) {
				if (app->configuration.inputBufferSize[i] == 0) {
					deleteVkFFT(app);
					return VKFFT_ERROR_EMPTY_inputBufferSize;
				}
			}
		}
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
#if(VKFFT_BACKEND==0) 
		if (inputLaunchConfiguration.outputBufferSize == 0) {
			deleteVkFFT(app);
			return VKFFT_ERROR_EMPTY_outputBufferSize;
		}
#endif
		app->configuration.outputBufferSize = inputLaunchConfiguration.outputBufferSize;
		if (app->configuration.outputBufferSize != 0) {
			for (pfUINT i = 0; i < app->configuration.outputBufferNum; i++) {
				if (app->configuration.outputBufferSize[i] == 0) {
					deleteVkFFT(app);
					return VKFFT_ERROR_EMPTY_outputBufferSize;
				}
			}
		}
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
#if(VKFFT_BACKEND==0) 
		if (inputLaunchConfiguration.kernelSize == 0) {
			deleteVkFFT(app);
			return VKFFT_ERROR_EMPTY_kernelSize;
		}
#endif
		app->configuration.kernelSize = inputLaunchConfiguration.kernelSize;
		if (app->configuration.kernelSize != 0) {
			for (pfUINT i = 0; i < app->configuration.kernelNum; i++) {
				if (app->configuration.kernelSize[i] == 0) {
					deleteVkFFT(app);
					return VKFFT_ERROR_EMPTY_kernelSize;
				}
			}
		}
		app->configuration.kernel = inputLaunchConfiguration.kernel;
	}

	if (inputLaunchConfiguration.bufferOffset != 0)	app->configuration.bufferOffset = inputLaunchConfiguration.bufferOffset;
	if (inputLaunchConfiguration.tempBufferOffset != 0)	app->configuration.tempBufferOffset = inputLaunchConfiguration.tempBufferOffset;
	if (inputLaunchConfiguration.inputBufferOffset != 0)	app->configuration.inputBufferOffset = inputLaunchConfiguration.inputBufferOffset;
	if (inputLaunchConfiguration.outputBufferOffset != 0)	app->configuration.outputBufferOffset = inputLaunchConfiguration.outputBufferOffset;
	if (inputLaunchConfiguration.kernelOffset != 0)	app->configuration.kernelOffset = inputLaunchConfiguration.kernelOffset;
	if (inputLaunchConfiguration.specifyOffsetsAtLaunch != 0)	app->configuration.specifyOffsetsAtLaunch = inputLaunchConfiguration.specifyOffsetsAtLaunch;
	//set optional parameters:
	pfUINT checkBufferSizeFor64BitAddressing = 0;
	for (pfUINT i = 0; i < app->configuration.bufferNum; i++) {
		if (app->configuration.bufferSize)
			checkBufferSizeFor64BitAddressing += app->configuration.bufferSize[i];
		else {
			checkBufferSizeFor64BitAddressing = app->configuration.size[0] * app->configuration.size[1] * app->configuration.size[2] * 8;
			if (app->configuration.coordinateFeatures > 0) checkBufferSizeFor64BitAddressing *= app->configuration.coordinateFeatures;
			if (app->configuration.numberBatches > 0) checkBufferSizeFor64BitAddressing *= app->configuration.numberBatches;
			if (app->configuration.numberKernels > 0) checkBufferSizeFor64BitAddressing *= app->configuration.numberKernels;
			if (app->configuration.doublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) checkBufferSizeFor64BitAddressing *= 2;
			if (app->configuration.quadDoubleDoublePrecision) checkBufferSizeFor64BitAddressing *= 4;
		}
	}
#if(VKFFT_BACKEND==2)
	app->configuration.useStrict32BitAddress = 0;
	if (checkBufferSizeFor64BitAddressing >= (pfUINT)pow((pfUINT)2, (pfUINT)32)) app->configuration.useStrict32BitAddress = -1;
#endif
	if (checkBufferSizeFor64BitAddressing >= (pfUINT)pow((pfUINT)2, (pfUINT)34)) app->configuration.useUint64 = 1;
	checkBufferSizeFor64BitAddressing = 0;
	for (pfUINT i = 0; i < app->configuration.inputBufferNum; i++) {
		if (app->configuration.inputBufferSize)
			checkBufferSizeFor64BitAddressing += app->configuration.inputBufferSize[i];
	}
#if(VKFFT_BACKEND==2)
	if (checkBufferSizeFor64BitAddressing >= (pfUINT)pow((pfUINT)2, (pfUINT)32)) app->configuration.useStrict32BitAddress = -1;
#endif
	if (checkBufferSizeFor64BitAddressing >= (pfUINT)pow((pfUINT)2, (pfUINT)34)) app->configuration.useUint64 = 1;

	checkBufferSizeFor64BitAddressing = 0;
	for (pfUINT i = 0; i < app->configuration.outputBufferNum; i++) {
		if (app->configuration.outputBufferSize)
			checkBufferSizeFor64BitAddressing += app->configuration.outputBufferSize[i];
	}
	if (checkBufferSizeFor64BitAddressing >= (pfUINT)pow((pfUINT)2, (pfUINT)34)) app->configuration.useUint64 = 1;

	checkBufferSizeFor64BitAddressing = 0;
	for (pfUINT i = 0; i < app->configuration.kernelNum; i++) {
		if (app->configuration.kernelSize)
			checkBufferSizeFor64BitAddressing += app->configuration.kernelSize[i];
	}
#if(VKFFT_BACKEND==2)
	if (checkBufferSizeFor64BitAddressing >= (pfUINT)pow((pfUINT)2, (pfUINT)32)) app->configuration.useStrict32BitAddress = -1;
	// No reason was found to disable strict 32 bit addressing, so enable it
	if (app->configuration.useStrict32BitAddress == 0) app->configuration.useStrict32BitAddress = 1;
#endif
	if (checkBufferSizeFor64BitAddressing >= (pfUINT)pow((pfUINT)2, (pfUINT)34)) app->configuration.useUint64 = 1;
	if (inputLaunchConfiguration.useUint64 != 0)	app->configuration.useUint64 = inputLaunchConfiguration.useUint64;
#if(VKFFT_BACKEND==2)
	if (inputLaunchConfiguration.useStrict32BitAddress != 0) app->configuration.useStrict32BitAddress = inputLaunchConfiguration.useStrict32BitAddress;
#endif
	if (inputLaunchConfiguration.maxThreadsNum != 0)	app->configuration.maxThreadsNum = inputLaunchConfiguration.maxThreadsNum;
	if (inputLaunchConfiguration.coalescedMemory != 0)	app->configuration.coalescedMemory = inputLaunchConfiguration.coalescedMemory;
	app->configuration.aimThreads = 128;
	if (inputLaunchConfiguration.aimThreads != 0)	app->configuration.aimThreads = inputLaunchConfiguration.aimThreads;
	app->configuration.numSharedBanks = 32;
	if (inputLaunchConfiguration.numSharedBanks != 0)	app->configuration.numSharedBanks = inputLaunchConfiguration.numSharedBanks;
	if (inputLaunchConfiguration.inverseReturnToInputBuffer != 0)	app->configuration.inverseReturnToInputBuffer = inputLaunchConfiguration.inverseReturnToInputBuffer;

	if (inputLaunchConfiguration.useLUT != 0)	app->configuration.useLUT = inputLaunchConfiguration.useLUT;
	if (inputLaunchConfiguration.useLUT_4step != 0) {
		if (inputLaunchConfiguration.useLUT_4step > 0)
			app->configuration.useLUT = 1;
		app->configuration.useLUT_4step = inputLaunchConfiguration.useLUT_4step;
	}
	else {
		if (app->configuration.useLUT_4step == 0)
			app->configuration.useLUT_4step = app->configuration.useLUT;
	}

	if (app->configuration.useLUT == -1)	app->configuration.useLUT_4step = -1;
	app->configuration.swapTo2Stage4Step = app->configuration.swapTo3Stage4Step;

    if (app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory){
		app->configuration.useLUT_4step = 1;
		app->configuration.useLUT = 1;
		app->configuration.swapTo3Stage4Step = 524288;
	}
	if (inputLaunchConfiguration.fixMaxRadixBluestein != 0) app->configuration.fixMaxRadixBluestein = inputLaunchConfiguration.fixMaxRadixBluestein;
	if (inputLaunchConfiguration.forceBluesteinSequenceSize != 0) app->configuration.forceBluesteinSequenceSize = inputLaunchConfiguration.forceBluesteinSequenceSize;

	if (app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory){
			app->configuration.fixMinRaderPrimeMult = 11;
			app->configuration.fixMaxRaderPrimeMult = 29;
	} else{
			app->configuration.fixMinRaderPrimeMult = 17;
			switch (app->configuration.vendorID) {
			case 0x10DE://NVIDIA
					app->configuration.fixMaxRaderPrimeMult = 89;
					break;
			case 0x1002://AMD profile
					app->configuration.fixMaxRaderPrimeMult = 89;
					break;
			default:
					app->configuration.fixMaxRaderPrimeMult = 17;
					break;
			}
			if (inputLaunchConfiguration.fixMinRaderPrimeMult != 0) app->configuration.fixMinRaderPrimeMult = inputLaunchConfiguration.fixMinRaderPrimeMult;
	}
	if (inputLaunchConfiguration.fixMaxRaderPrimeMult != 0) app->configuration.fixMaxRaderPrimeMult = inputLaunchConfiguration.fixMaxRaderPrimeMult;

	switch (app->configuration.vendorID) {
	case 0x1002://AMD profile
			if (app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory)
					app->configuration.fixMinRaderPrimeFFT = 19;
			else if (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory)
					app->configuration.fixMinRaderPrimeFFT = 29;
			else
					app->configuration.fixMinRaderPrimeFFT = 17;
			break;
	default:
			app->configuration.fixMinRaderPrimeFFT = 17;
			break;
	}
	app->configuration.fixMaxRaderPrimeFFT = 16384;
	if (inputLaunchConfiguration.fixMinRaderPrimeFFT != 0) app->configuration.fixMinRaderPrimeFFT = inputLaunchConfiguration.fixMinRaderPrimeFFT;
	if (inputLaunchConfiguration.fixMaxRaderPrimeFFT != 0) app->configuration.fixMaxRaderPrimeFFT = inputLaunchConfiguration.fixMaxRaderPrimeFFT;
	if (inputLaunchConfiguration.performR2C != 0) {
		app->configuration.performR2C = inputLaunchConfiguration.performR2C;
	}
	if (inputLaunchConfiguration.performDCT != 0) {
		app->configuration.performDCT = inputLaunchConfiguration.performDCT;
	}
	if (inputLaunchConfiguration.performDST != 0) {
		app->configuration.performDST = inputLaunchConfiguration.performDST;
	}
	if (inputLaunchConfiguration.forceCallbackVersionRealTransforms != 0)  app->configuration.forceCallbackVersionRealTransforms = inputLaunchConfiguration.forceCallbackVersionRealTransforms; 
	
	if ((inputLaunchConfiguration.disableMergeSequencesR2C != 0) || app->configuration.forceCallbackVersionRealTransforms) {
		app->configuration.disableMergeSequencesR2C = 1;
	}
	app->configuration.normalize = 0;
	if (inputLaunchConfiguration.normalize != 0)	app->configuration.normalize = inputLaunchConfiguration.normalize;
	if (inputLaunchConfiguration.makeForwardPlanOnly != 0)	app->configuration.makeForwardPlanOnly = inputLaunchConfiguration.makeForwardPlanOnly;
	if (inputLaunchConfiguration.makeInversePlanOnly != 0)	app->configuration.makeInversePlanOnly = inputLaunchConfiguration.makeInversePlanOnly;

	app->configuration.reorderFourStep = 1;
	if (inputLaunchConfiguration.disableReorderFourStep != 0) {
		app->configuration.reorderFourStep = 0;
		//if ((app->configuration.swapTo3Stage4Step < 1048576) && (!app->configuration.quadDoubleDoublePrecision) && (!app->configuration.quadDoubleDoublePrecisionDoubleMemory)) app->configuration.swapTo3Stage4Step = 1048576;
	}
	if (inputLaunchConfiguration.frequencyZeroPadding != 0) app->configuration.frequencyZeroPadding = inputLaunchConfiguration.frequencyZeroPadding;
	for (pfUINT i = 0; i < app->configuration.FFTdim; i++) {
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

	if ((app->configuration.performDCT != 0) || (app->configuration.performDST != 0)) {
		app->configuration.registerBoost = 1;
		app->configuration.registerBoostNonPow2 = 0;
		app->configuration.registerBoost4Step = 1;
		if (app->configuration.sharedMemorySize > 167936) {
			app->configuration.sharedMemorySize = 167936; // H100 fix - register file probably can't keep up with shared memory size 
		}
	}
	if (inputLaunchConfiguration.sharedMemorySize != 0)	app->configuration.sharedMemorySize = inputLaunchConfiguration.sharedMemorySize;
	app->configuration.sharedMemorySizePow2 = (pfUINT)pow(2, (pfUINT)log2(app->configuration.sharedMemorySize));
	
	app->configuration.coordinateFeatures = 1;
	app->configuration.numberBatches = 1;
	if (inputLaunchConfiguration.coordinateFeatures != 0)	app->configuration.coordinateFeatures = inputLaunchConfiguration.coordinateFeatures;
	if (inputLaunchConfiguration.numberBatches != 0)	app->configuration.numberBatches = inputLaunchConfiguration.numberBatches;

	app->configuration.matrixConvolution = 1;
	app->configuration.numberKernels = 1;
	if (inputLaunchConfiguration.kernelConvolution != 0) {
		app->configuration.kernelConvolution = inputLaunchConfiguration.kernelConvolution;
		app->configuration.reorderFourStep = 0;
		app->configuration.registerBoost = 1;
		app->configuration.registerBoostNonPow2 = 0;
		app->configuration.registerBoost4Step = 1;
	}

	if (app->configuration.performConvolution) {

		if (inputLaunchConfiguration.matrixConvolution != 0)	app->configuration.matrixConvolution = inputLaunchConfiguration.matrixConvolution;
		if (inputLaunchConfiguration.numberKernels != 0)	app->configuration.numberKernels = inputLaunchConfiguration.numberKernels;

		if (inputLaunchConfiguration.symmetricKernel != 0)	app->configuration.symmetricKernel = inputLaunchConfiguration.symmetricKernel;
		if (inputLaunchConfiguration.conjugateConvolution != 0)	app->configuration.conjugateConvolution = inputLaunchConfiguration.conjugateConvolution;
		if (inputLaunchConfiguration.crossPowerSpectrumNormalization != 0)	app->configuration.crossPowerSpectrumNormalization = inputLaunchConfiguration.crossPowerSpectrumNormalization;

		app->configuration.reorderFourStep = 0;
		app->configuration.registerBoost = 1;
		app->configuration.registerBoostNonPow2 = 0;
		app->configuration.registerBoost4Step = 1;
		if (app->configuration.matrixConvolution > 1) app->configuration.coordinateFeatures = app->configuration.matrixConvolution;
	}
	app->firstAxis = 0;
	app->lastAxis = app->configuration.FFTdim - 1;
	for (int i = 0; i < app->configuration.FFTdim; i++) {
		app->configuration.omitDimension[i] = inputLaunchConfiguration.omitDimension[i];
		if ((app->configuration.size[i] == 1) && (!(app->configuration.performR2C && (i == 0))) && (!app->configuration.performConvolution)) app->configuration.omitDimension[i] = 1;
	}
	for (int i = (int)app->configuration.FFTdim - 1; i >= 0; i--) {
		if (app->configuration.omitDimension[i] != 0) {
			app->lastAxis--;
			if (app->configuration.performConvolution) {
				deleteVkFFT(app);
				return VKFFT_ERROR_UNSUPPORTED_FFT_OMIT;
			}
		}
		else {
			i = 0;
		}
	}
	for (int i = 0; i < app->configuration.FFTdim; i++) {
		if (app->configuration.omitDimension[i] != 0) {
			app->firstAxis++;
			if (app->configuration.performConvolution) {
				deleteVkFFT(app);
				return VKFFT_ERROR_UNSUPPORTED_FFT_OMIT;
			}
			if ((i == 0) && (app->configuration.performR2C)) {
				deleteVkFFT(app);
				return VKFFT_ERROR_UNSUPPORTED_FFT_OMIT;
			}
		}
		else {
			i = (int)app->configuration.FFTdim;
		}
	}
	
	if (app->firstAxis > app->lastAxis) {
		deleteVkFFT(app);
		return VKFFT_ERROR_UNSUPPORTED_FFT_OMIT;
	}
	if (inputLaunchConfiguration.reorderFourStep != 0)	app->configuration.reorderFourStep = inputLaunchConfiguration.reorderFourStep;
	app->configuration.maxCodeLength = 4000000;
	if (inputLaunchConfiguration.maxCodeLength != 0) app->configuration.maxCodeLength = inputLaunchConfiguration.maxCodeLength;
	app->configuration.maxTempLength = 5000;
	if (inputLaunchConfiguration.maxTempLength != 0) app->configuration.maxTempLength = inputLaunchConfiguration.maxTempLength;

	if (inputLaunchConfiguration.useRaderUintLUT != 0)	app->configuration.useRaderUintLUT = inputLaunchConfiguration.useRaderUintLUT;
	if (inputLaunchConfiguration.halfThreads != 0)	app->configuration.halfThreads = inputLaunchConfiguration.halfThreads;
	if (inputLaunchConfiguration.swapTo2Stage4Step != 0)	app->configuration.swapTo2Stage4Step = inputLaunchConfiguration.swapTo2Stage4Step;
	if (inputLaunchConfiguration.swapTo3Stage4Step != 0)	app->configuration.swapTo3Stage4Step = inputLaunchConfiguration.swapTo3Stage4Step;
	if ((app->configuration.performDCT > 0) || (app->configuration.performDST > 0)) app->configuration.performBandwidthBoost = -1;
	if (inputLaunchConfiguration.performBandwidthBoost != 0)	app->configuration.performBandwidthBoost = inputLaunchConfiguration.performBandwidthBoost;
#if(VKFFT_BACKEND==0)	
	if (inputLaunchConfiguration.stagingBuffer != 0)	app->configuration.stagingBuffer = inputLaunchConfiguration.stagingBuffer;
	if (inputLaunchConfiguration.stagingBufferMemory != 0)	app->configuration.stagingBufferMemory = inputLaunchConfiguration.stagingBufferMemory;
#endif	
    for (pfUINT i = 0; i < app->configuration.FFTdim; i++) {
        if (inputLaunchConfiguration.groupedBatch[i] != 0)	app->configuration.groupedBatch[i] = inputLaunchConfiguration.groupedBatch[i];
    }
	
	if (inputLaunchConfiguration.devicePageSize != 0)	app->configuration.devicePageSize = inputLaunchConfiguration.devicePageSize;
	if (inputLaunchConfiguration.localPageSize != 0)	app->configuration.localPageSize = inputLaunchConfiguration.localPageSize;
	if (inputLaunchConfiguration.keepShaderCode != 0)	app->configuration.keepShaderCode = inputLaunchConfiguration.keepShaderCode;
	if (inputLaunchConfiguration.printMemoryLayout != 0)	app->configuration.printMemoryLayout = inputLaunchConfiguration.printMemoryLayout;
	if (inputLaunchConfiguration.considerAllAxesStrided != 0)	app->configuration.considerAllAxesStrided = inputLaunchConfiguration.considerAllAxesStrided;
#if(VKFFT_BACKEND!=5)
	if (inputLaunchConfiguration.loadApplicationString != 0)	app->configuration.loadApplicationString = inputLaunchConfiguration.loadApplicationString;
	if (inputLaunchConfiguration.saveApplicationToString != 0)	app->configuration.saveApplicationToString = inputLaunchConfiguration.saveApplicationToString;
#endif
	if (inputLaunchConfiguration.disableSetLocale != 0)	app->configuration.disableSetLocale = inputLaunchConfiguration.disableSetLocale;

	if (inputLaunchConfiguration.loadApplicationFromString != 0) {
		app->configuration.loadApplicationFromString = inputLaunchConfiguration.loadApplicationFromString;
		if (app->configuration.saveApplicationToString != 0) {
			deleteVkFFT(app);
			return VKFFT_ERROR_ENABLED_saveApplicationToString;
		}
		if (app->configuration.loadApplicationString == 0) {
			deleteVkFFT(app);
			return VKFFT_ERROR_EMPTY_applicationString;
		}
		memcpy(&app->applicationStringSize, app->configuration.loadApplicationString, sizeof(pfUINT));
		memcpy(&app->applicationStringOffsetRader, (char*)app->configuration.loadApplicationString + 2 * sizeof(pfUINT), sizeof(pfUINT));
		app->currentApplicationStringPos = 5 * sizeof(pfUINT);
	}
	//temporary set:
	app->configuration.registerBoost4Step = 1;
#if(VKFFT_BACKEND==0) 
	app->configuration.useUint64 = 0; //No physical addressing mode in Vulkan shaders. Use multiple-buffer support to achieve emulation of physical addressing.
#endif
	//pfUINT initSharedMemory = app->configuration.sharedMemorySize;
	return resFFT;
}
static inline VkFFTResult initializeVkFFT(VkFFTApplication* app, VkFFTConfiguration inputLaunchConfiguration) {
	
	VkFFTResult resFFT = VKFFT_SUCCESS;
    unsigned char *test = (unsigned char*)app;
    if (app == 0){
    	return VKFFT_ERROR_EMPTY_app;
    }
	if (memcmp(test, test + 1, sizeof(VkFFTApplication) - 1) != 0){
		return VKFFT_ERROR_NONZERO_APP_INITIALIZATION;
	}
	resFFT = setConfigurationVkFFT(app, inputLaunchConfiguration);
	if (resFFT != VKFFT_SUCCESS) {
		deleteVkFFT(app);
		return resFFT;
	}

	if (!app->configuration.makeForwardPlanOnly) {
		app->localFFTPlan_inverse = (VkFFTPlan*)calloc(1, sizeof(VkFFTPlan));
		if (app->localFFTPlan_inverse) {
			for (pfUINT i = 0; i < app->configuration.FFTdim; i++) {
				//app->configuration.sharedMemorySize = ((app->configuration.size[i] & (app->configuration.size[i] - 1)) == 0) ? app->configuration.sharedMemorySizePow2 : initSharedMemory;
				resFFT = VkFFTScheduler(app, app->localFFTPlan_inverse, (int)i);
				if (resFFT == VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH) {
					//try again with Rader disabled - sequences like 89^4 can still be done with Bluestein FFT
					memset(app->localFFTPlan_inverse, 0, sizeof(VkFFTPlan));
					pfUINT temp_fixMaxRaderPrimeFFT = app->configuration.fixMaxRaderPrimeFFT;
					app->configuration.fixMaxRaderPrimeFFT = app->configuration.fixMinRaderPrimeFFT;
					pfUINT temp_fixMaxRaderPrimeMult = app->configuration.fixMaxRaderPrimeMult;
					app->configuration.fixMaxRaderPrimeMult = app->configuration.fixMinRaderPrimeMult;
					resFFT = VkFFTScheduler(app, app->localFFTPlan_inverse, (int)i);
					app->configuration.fixMaxRaderPrimeFFT = temp_fixMaxRaderPrimeFFT;
					app->configuration.fixMaxRaderPrimeMult = temp_fixMaxRaderPrimeMult;
				}
				if (resFFT != VKFFT_SUCCESS) {
					deleteVkFFT(app);
					return resFFT;
				}
				if (app->useBluesteinFFT[i] && (app->localFFTPlan_inverse->numAxisUploads[i] > 1)) {
					for (pfUINT j = 0; j < app->localFFTPlan_inverse->numAxisUploads[i]; j++) {
						app->localFFTPlan_inverse->inverseBluesteinAxes[i][j] = app->localFFTPlan_inverse->axes[i][j];
					}
				}
			}
			for (pfUINT i = 0; i < app->configuration.FFTdim; i++) {
				//app->configuration.sharedMemorySize = ((app->configuration.size[i] & (app->configuration.size[i] - 1)) == 0) ? app->configuration.sharedMemorySizePow2 : initSharedMemory;
				for (pfUINT j = 0; j < app->localFFTPlan_inverse->numAxisUploads[i]; j++) {
					resFFT = VkFFTPlanAxis(app, app->localFFTPlan_inverse, i, j, 1, 0);
					if (resFFT != VKFFT_SUCCESS) {
						deleteVkFFT(app);
						return resFFT;
					}
				}
				if (app->useBluesteinFFT[i] && (app->localFFTPlan_inverse->numAxisUploads[i] > 1)) {
					for (pfUINT j = 1; j < app->localFFTPlan_inverse->numAxisUploads[i]; j++) {
						resFFT = VkFFTPlanAxis(app, app->localFFTPlan_inverse, i, j, 1, 1);
						if (resFFT != VKFFT_SUCCESS) {
							deleteVkFFT(app);
							return resFFT;
						}
					}
				}
				if ((app->localFFTPlan_inverse->bigSequenceEvenR2C) && (i == 0)) {
					resFFT = VkFFTPlanR2CMultiUploadDecomposition(app, app->localFFTPlan_inverse, 1);
					if (resFFT != VKFFT_SUCCESS) {
						deleteVkFFT(app);
						return resFFT;
					}
				}
			}
		}
		else {
			deleteVkFFT(app);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
	}
	if (!app->configuration.makeInversePlanOnly) {
		app->localFFTPlan = (VkFFTPlan*)calloc(1, sizeof(VkFFTPlan));
		if (app->localFFTPlan) {
			for (pfUINT i = 0; i < app->configuration.FFTdim; i++) {
				//app->configuration.sharedMemorySize = ((app->configuration.size[i] & (app->configuration.size[i] - 1)) == 0) ? app->configuration.sharedMemorySizePow2 : initSharedMemory;
				resFFT = VkFFTScheduler(app, app->localFFTPlan, (int)i);
				if (resFFT == VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH) {
					//try again with Rader disabled - sequences like 89^4 can still be done with Bluestein FFT
					memset(app->localFFTPlan, 0, sizeof(VkFFTPlan));
					pfUINT temp_fixMaxRaderPrimeFFT = app->configuration.fixMaxRaderPrimeFFT;
					app->configuration.fixMaxRaderPrimeFFT = app->configuration.fixMinRaderPrimeFFT;
					pfUINT temp_fixMaxRaderPrimeMult = app->configuration.fixMaxRaderPrimeMult;
					app->configuration.fixMaxRaderPrimeMult = app->configuration.fixMinRaderPrimeMult;
					resFFT = VkFFTScheduler(app, app->localFFTPlan, (int)i);
					app->configuration.fixMaxRaderPrimeFFT = temp_fixMaxRaderPrimeFFT;
					app->configuration.fixMaxRaderPrimeMult = temp_fixMaxRaderPrimeMult;
				}
				if (resFFT != VKFFT_SUCCESS) {
					deleteVkFFT(app);
					return resFFT;
				}
				if (app->useBluesteinFFT[i] && (app->localFFTPlan->numAxisUploads[i] > 1)) {
					for (pfUINT j = 0; j < app->localFFTPlan->numAxisUploads[i]; j++) {
						app->localFFTPlan->inverseBluesteinAxes[i][j] = app->localFFTPlan->axes[i][j];
					}
				}
			}
			for (pfUINT i = 0; i < app->configuration.FFTdim; i++) {
				//app->configuration.sharedMemorySize = ((app->configuration.size[i] & (app->configuration.size[i] - 1)) == 0) ? app->configuration.sharedMemorySizePow2 : initSharedMemory;
				for (pfUINT j = 0; j < app->localFFTPlan->numAxisUploads[i]; j++) {
					resFFT = VkFFTPlanAxis(app, app->localFFTPlan, i, j, 0, 0);
					if (resFFT != VKFFT_SUCCESS) {
						deleteVkFFT(app);
						return resFFT;
					}
				}
				if (app->useBluesteinFFT[i] && (app->localFFTPlan->numAxisUploads[i] > 1)) {
					for (pfUINT j = 1; j < app->localFFTPlan->numAxisUploads[i]; j++) {
						resFFT = VkFFTPlanAxis(app, app->localFFTPlan, i, j, 0, 1);
						if (resFFT != VKFFT_SUCCESS) {
							deleteVkFFT(app);
							return resFFT;
						}
					}
				}
				if ((app->localFFTPlan->bigSequenceEvenR2C) && (i == 0)) {
					resFFT = VkFFTPlanR2CMultiUploadDecomposition(app, app->localFFTPlan, 0);
					if (resFFT != VKFFT_SUCCESS) {
						deleteVkFFT(app);
						return resFFT;
					}
				}
			}
		}
		else {
			deleteVkFFT(app);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
	}

	if (app->configuration.allocateTempBuffer && (app->configuration.tempBuffer == 0)) {
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
#if(VKFFT_BACKEND==0)
		app->configuration.tempBuffer = (VkBuffer*)malloc(sizeof(VkBuffer));
		if (!app->configuration.tempBuffer) {
			deleteVkFFT(app);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
		resFFT = allocateBufferVulkan(app, app->configuration.tempBuffer, &app->configuration.tempBufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, app->configuration.tempBufferSize[0]);
		if (resFFT != VKFFT_SUCCESS) {
			deleteVkFFT(app);
			return resFFT;
		}
#elif(VKFFT_BACKEND==1)
		app->configuration.tempBuffer = (void**)malloc(sizeof(void*));
		if (!app->configuration.tempBuffer) {
			deleteVkFFT(app);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
		res = cudaMalloc(app->configuration.tempBuffer, app->configuration.tempBufferSize[0]);
		if (res != cudaSuccess) {
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_TO_ALLOCATE;
		}
#elif(VKFFT_BACKEND==2)
		app->configuration.tempBuffer = (void**)malloc(sizeof(void*));
		if (!app->configuration.tempBuffer) {
			deleteVkFFT(app);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
		res = hipMalloc(app->configuration.tempBuffer, app->configuration.tempBufferSize[0]);
		if (res != hipSuccess) {
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_TO_ALLOCATE;
		}
#elif(VKFFT_BACKEND==3)
		app->configuration.tempBuffer = (cl_mem*)malloc(sizeof(cl_mem));
		if (!app->configuration.tempBuffer) {
			deleteVkFFT(app);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
		app->configuration.tempBuffer[0] = clCreateBuffer(app->configuration.context[0], CL_MEM_READ_WRITE, app->configuration.tempBufferSize[0], 0, &res);
		if (res != CL_SUCCESS) {
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_TO_ALLOCATE;
		}
#elif(VKFFT_BACKEND==4)
		app->configuration.tempBuffer = (void**)malloc(sizeof(void*));
		if (!app->configuration.tempBuffer) {
			deleteVkFFT(app);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
		ze_device_mem_alloc_desc_t device_desc = VKFFT_ZERO_INIT;
		device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
		res = zeMemAllocDevice(app->configuration.context[0], &device_desc, app->configuration.tempBufferSize[0], sizeof(float), app->configuration.device[0], app->configuration.tempBuffer);
		if (res != ZE_RESULT_SUCCESS) {
			deleteVkFFT(app);
			return VKFFT_ERROR_FAILED_TO_ALLOCATE;
		}
#elif(VKFFT_BACKEND==5)
		app->configuration.tempBuffer = (MTL::Buffer**)malloc(sizeof(MTL::Buffer*));
		if (!app->configuration.tempBuffer) {
			deleteVkFFT(app);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
		app->configuration.tempBuffer[0] = app->configuration.device->newBuffer(app->configuration.tempBufferSize[0], MTL::ResourceStorageModePrivate);
#endif

		if (!app->configuration.makeInversePlanOnly) {
			for (pfUINT i = 0; i < app->configuration.FFTdim; i++) {
				for (pfUINT j = 0; j < app->localFFTPlan->numAxisUploads[i]; j++) {
					app->localFFTPlan->axes[i][j].specializationConstants.performBufferSetUpdate = 1;
				}
				if (app->useBluesteinFFT[i] && (app->localFFTPlan->numAxisUploads[i] > 1)) {
					for (pfUINT j = 1; j < app->localFFTPlan->numAxisUploads[i]; j++) {
						app->localFFTPlan->inverseBluesteinAxes[i][j].specializationConstants.performBufferSetUpdate = 1;
					}
				}
			}
			if (app->localFFTPlan->bigSequenceEvenR2C) {
				app->localFFTPlan->R2Cdecomposition.specializationConstants.performBufferSetUpdate = 1;
			}
		}
		if (!app->configuration.makeForwardPlanOnly) {
			for (pfUINT i = 0; i < app->configuration.FFTdim; i++) {
				for (pfUINT j = 0; j < app->localFFTPlan_inverse->numAxisUploads[i]; j++) {
					app->localFFTPlan_inverse->axes[i][j].specializationConstants.performBufferSetUpdate = 1;
				}
				if (app->useBluesteinFFT[i] && (app->localFFTPlan_inverse->numAxisUploads[i] > 1)) {
					for (pfUINT j = 1; j < app->localFFTPlan_inverse->numAxisUploads[i]; j++) {
						app->localFFTPlan_inverse->inverseBluesteinAxes[i][j].specializationConstants.performBufferSetUpdate = 1;
					}
				}
			}
			if (app->localFFTPlan_inverse->bigSequenceEvenR2C) {
				app->localFFTPlan_inverse->R2Cdecomposition.specializationConstants.performBufferSetUpdate = 1;
			}
		}
	}
	for (pfUINT i = 0; i < app->configuration.FFTdim; i++) {
		if (app->useBluesteinFFT[i]) {
			if (!app->configuration.makeInversePlanOnly)
				resFFT = VkFFTGeneratePhaseVectors(app, app->localFFTPlan, i);
			else
				resFFT = VkFFTGeneratePhaseVectors(app, app->localFFTPlan_inverse, i);
			if (resFFT != VKFFT_SUCCESS) {
				deleteVkFFT(app);
				return resFFT;
			}
		}
	}

	if (inputLaunchConfiguration.saveApplicationToString != 0) {
		pfUINT totalBinarySize = 5 * sizeof(pfUINT);
		if (!app->configuration.makeForwardPlanOnly) {
			for (pfUINT i = 0; i < app->configuration.FFTdim; i++) {
				for (pfUINT j = 0; j < app->localFFTPlan_inverse->numAxisUploads[i]; j++) {
					totalBinarySize += app->localFFTPlan_inverse->axes[i][j].binarySize + sizeof(pfUINT);
				}
				if (app->useBluesteinFFT[i] && (app->localFFTPlan_inverse->numAxisUploads[i] > 1)) {
					for (pfUINT j = 1; j < app->localFFTPlan_inverse->numAxisUploads[i]; j++) {
						totalBinarySize += app->localFFTPlan_inverse->inverseBluesteinAxes[i][j].binarySize + sizeof(pfUINT);
					}
				}
				if ((app->localFFTPlan_inverse->bigSequenceEvenR2C) && (i == 0)) {
					totalBinarySize += app->localFFTPlan_inverse->R2Cdecomposition.binarySize + sizeof(pfUINT);
				}
			}
		}
		if (!app->configuration.makeInversePlanOnly) {
			for (pfUINT i = 0; i < app->configuration.FFTdim; i++) {
				for (pfUINT j = 0; j < app->localFFTPlan->numAxisUploads[i]; j++) {
					totalBinarySize += app->localFFTPlan->axes[i][j].binarySize + sizeof(pfUINT);
				}
				if (app->useBluesteinFFT[i] && (app->localFFTPlan->numAxisUploads[i] > 1)) {
					for (pfUINT j = 1; j < app->localFFTPlan->numAxisUploads[i]; j++) {
						totalBinarySize += app->localFFTPlan->inverseBluesteinAxes[i][j].binarySize + sizeof(pfUINT);
					}
				}
				if ((app->localFFTPlan->bigSequenceEvenR2C) && (i == 0)) {
					totalBinarySize += app->localFFTPlan->R2Cdecomposition.binarySize + sizeof(pfUINT);
				}
			}
		}
		for (pfUINT i = 0; i < app->configuration.FFTdim; i++) {
			if (app->useBluesteinFFT[i]) {
				totalBinarySize += app->applicationBluesteinStringSize[i];
			}
		}
		if (app->numRaderFFTPrimes > 0) {
			app->applicationStringOffsetRader = totalBinarySize;
			for (pfUINT i = 0; i < app->numRaderFFTPrimes; i++) {
				totalBinarySize += app->rader_buffer_size[i];
			}
		}
		app->saveApplicationString = calloc(totalBinarySize, 1);
		if (!app->saveApplicationString) {
			deleteVkFFT(app);
			return VKFFT_ERROR_MALLOC_FAILED;
		}
		app->applicationStringSize = totalBinarySize;
		char* localApplicationStringCast = (char*)app->saveApplicationString;
		memcpy(localApplicationStringCast, &totalBinarySize, sizeof(pfUINT));
		memcpy(localApplicationStringCast + 2 * sizeof(pfUINT), &app->applicationStringOffsetRader, sizeof(pfUINT));
		pfUINT currentPos = 5 * sizeof(pfUINT);
		if (!app->configuration.makeForwardPlanOnly) {
			for (pfUINT i = 0; i < app->configuration.FFTdim; i++) {
				for (pfUINT j = 0; j < app->localFFTPlan_inverse->numAxisUploads[i]; j++) {
					memcpy(localApplicationStringCast + currentPos, &app->localFFTPlan_inverse->axes[i][j].binarySize, sizeof(pfUINT));
					currentPos += sizeof(pfUINT);
					memcpy(localApplicationStringCast + currentPos, app->localFFTPlan_inverse->axes[i][j].binary, app->localFFTPlan_inverse->axes[i][j].binarySize);
					currentPos += app->localFFTPlan_inverse->axes[i][j].binarySize;
				}
				if (app->useBluesteinFFT[i] && (app->localFFTPlan_inverse->numAxisUploads[i] > 1)) {
					for (pfUINT j = 1; j < app->localFFTPlan_inverse->numAxisUploads[i]; j++) {
						memcpy(localApplicationStringCast + currentPos, &app->localFFTPlan_inverse->inverseBluesteinAxes[i][j].binarySize, sizeof(pfUINT));
						currentPos += sizeof(pfUINT);
						memcpy(localApplicationStringCast + currentPos, app->localFFTPlan_inverse->inverseBluesteinAxes[i][j].binary, app->localFFTPlan_inverse->inverseBluesteinAxes[i][j].binarySize);
						currentPos += app->localFFTPlan_inverse->inverseBluesteinAxes[i][j].binarySize;
					}
				}
				if ((app->localFFTPlan_inverse->bigSequenceEvenR2C) && (i == 0)) {
					memcpy(localApplicationStringCast + currentPos, &app->localFFTPlan_inverse->R2Cdecomposition.binarySize, sizeof(pfUINT));
					currentPos += sizeof(pfUINT);
					memcpy(localApplicationStringCast + currentPos, app->localFFTPlan_inverse->R2Cdecomposition.binary, app->localFFTPlan_inverse->R2Cdecomposition.binarySize);
					currentPos += app->localFFTPlan_inverse->R2Cdecomposition.binarySize;
				}
			}
		}
		if (!app->configuration.makeInversePlanOnly) {
			for (pfUINT i = 0; i < app->configuration.FFTdim; i++) {
				for (pfUINT j = 0; j < app->localFFTPlan->numAxisUploads[i]; j++) {
					memcpy(localApplicationStringCast + currentPos, &app->localFFTPlan->axes[i][j].binarySize, sizeof(pfUINT));
					currentPos += sizeof(pfUINT);
					memcpy(localApplicationStringCast + currentPos, app->localFFTPlan->axes[i][j].binary, app->localFFTPlan->axes[i][j].binarySize);
					currentPos += app->localFFTPlan->axes[i][j].binarySize;
				}
				if (app->useBluesteinFFT[i] && (app->localFFTPlan->numAxisUploads[i] > 1)) {
					for (pfUINT j = 1; j < app->localFFTPlan->numAxisUploads[i]; j++) {
						memcpy(localApplicationStringCast + currentPos, &app->localFFTPlan->inverseBluesteinAxes[i][j].binarySize, sizeof(pfUINT));
						currentPos += sizeof(pfUINT);
						memcpy(localApplicationStringCast + currentPos, app->localFFTPlan->inverseBluesteinAxes[i][j].binary, app->localFFTPlan->inverseBluesteinAxes[i][j].binarySize);
						currentPos += app->localFFTPlan->inverseBluesteinAxes[i][j].binarySize;
					}
				}
				if ((app->localFFTPlan->bigSequenceEvenR2C) && (i == 0)) {
					memcpy(localApplicationStringCast + currentPos, &app->localFFTPlan->R2Cdecomposition.binarySize, sizeof(pfUINT));
					currentPos += sizeof(pfUINT);
					memcpy(localApplicationStringCast + currentPos, app->localFFTPlan->R2Cdecomposition.binary, app->localFFTPlan->R2Cdecomposition.binarySize);
					currentPos += app->localFFTPlan->R2Cdecomposition.binarySize;
				}
			}
		}
		for (pfUINT i = 0; i < app->configuration.FFTdim; i++) {
			if (app->useBluesteinFFT[i]) {
				memcpy(localApplicationStringCast + currentPos, app->applicationBluesteinString[i], app->applicationBluesteinStringSize[i]);
				currentPos += app->applicationBluesteinStringSize[i];
			}
		}
		if (app->numRaderFFTPrimes > 0) {
			for (pfUINT i = 0; i < app->numRaderFFTPrimes; i++) {
				memcpy(localApplicationStringCast + currentPos, app->raderFFTkernel[i], app->rader_buffer_size[i]);
				currentPos += app->rader_buffer_size[i];
			}
		}
		for (pfUINT i = 0; i < app->configuration.FFTdim; i++) {
			if (app->applicationBluesteinString[i] != 0) {
				free(app->applicationBluesteinString[i]);
				app->applicationBluesteinString[i] = 0;
			}
		}
	}
#if(VKFFT_BACKEND==0)
	if (app->configuration.isCompilerInitialized) {
		glslang_finalize_process();
		app->configuration.isCompilerInitialized = 0;
	}
#endif
	return resFFT;
}

#endif
