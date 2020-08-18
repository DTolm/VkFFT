#include <math.h>
#include <vulkan/vulkan.h>

typedef struct {
	//WHDCN layout
	uint32_t size[3] = { 1,1,1 }; // WHD -system dimensions 
	uint32_t coordinateFeatures = 1; // C - coordinate, or dimension of features vector. In matrix convolution - size of vector
	uint32_t matrixConvolution = 1; //if equal to 2 perform 2x2, if equal to 3 perform 3x3 matrix-vector convolution. Overrides coordinateFeatures

	uint32_t numberBatches = 1;// N - used to perform multiple batches of initial data
	uint32_t numberKernels = 1;// N - only used in convolution step - specify how many kernels were initialized before. Expands one input to multiple (batched) output
	uint32_t FFTdim = 1; //FFT dimensionality (1, 2 or 3)
	uint32_t radix = 8; //FFT radix (2, 4 or 8)
	bool performZeropadding[3] = { false, false, false }; // perform zeropadding (false - off, true - on)
	bool performTranspose[2] = { true, true }; //will be selected automatically
	bool performConvolution = false; //perform convolution in this application (false - off, true - on)
	bool performR2C = false; //perform R2C/C2R decomposition (false - off, true - on)
	bool inverse = false; //perform inverse FFT (false - forward, true - inverse)
	bool symmetricKernel = false; //specify if kernel in 2x2 or 3x3 matrix convolution is symmetric
	bool isInputFormatted = false; //specify if input buffer is not padded for R2C if out-of-place mode is selected (only if numberBatches==1 and numberKernels==1) - false - padded, true - not padded
	bool isOutputFormatted = false; //specify if output buffer is not padded for R2C if out-of-place mode is selected (only if numberBatches==1 and numberKernels==1) - false - padded, true - not padded
	char shaderPath[256] = "shaders/"; //path to shaders, can be selected automatically in CMake
	VkDevice* device;

	VkDeviceSize* bufferSize;
	VkDeviceSize* inputBufferSize;
	VkDeviceSize* outputBufferSize;

	VkBuffer* buffer;
	VkBuffer* inputBuffer;
	VkBuffer* outputBuffer;

	VkDeviceSize* kernelSize;
	VkBuffer* kernel;
} VkFFTConfiguration;

typedef struct {
	VkBool32 inverse;
	VkBool32 zeropad[2];
	uint32_t inputStride[5];
	uint32_t outputStride[5];
	uint32_t radixStride[3];
	uint32_t numStages;
	uint32_t stageRadix[2] = { 0,0 };
	uint32_t ratio[2];
	VkBool32 ratioDirection[2];
	uint32_t inputOffset;
	uint32_t outputOffset;
	uint32_t coordinate;
	uint32_t batch;
} VkFFTPushConstantsLayout;
typedef struct {
	uint32_t inputStride[5];
	uint32_t ratio;
	bool ratioDirection;
	uint32_t coordinate;
	uint32_t batch;
} VkFFTTransposePushConstantsLayout;
typedef struct {
	uint32_t axisBlock[4];
	uint32_t groupedBatch = 16;
	VkFFTPushConstantsLayout pushConstants;
	VkDescriptorPool descriptorPool;
	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorSet descriptorSet;
	VkPipelineLayout pipelineLayout;
	VkPipeline pipeline;
} VkFFTAxis;
typedef struct {
	uint32_t transposeBlock[3];
	VkFFTTransposePushConstantsLayout pushConstants;
	VkDescriptorPool descriptorPool;
	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorSet descriptorSet;
	VkPipelineLayout pipelineLayout;
	VkPipeline pipeline;
} VkFFTTranspose;
typedef struct {

	VkFFTAxis axes[3];
	VkFFTAxis supportAxes[2];//Nx/2+1 for r2c/c2r
	VkFFTTranspose transpose[2];

} VkFFTPlan;
typedef struct VkFFTApplication {
	VkFFTConfiguration configuration = {};
	VkFFTPlan localFFTPlan = {};
	VkFFTPlan localFFTPlan_inverse_convolution = {}; //additional inverse plan for convolution.
	uint32_t* VkFFTReadShader(uint32_t& length, const char* filename) {

		FILE* fp = fopen(filename, "rb");
		if (fp == NULL) {
			printf("Could not find or open file: %s\n", filename);
		}

		// get file size.
		fseek(fp, 0, SEEK_END);
		long filesize = ftell(fp);
		fseek(fp, 0, SEEK_SET);

		long filesizepadded = long(ceil(filesize / 4.0)) * 4;

		char* str = new char[filesizepadded];
		fread(str, filesize, sizeof(char), fp);
		fclose(fp);

		for (long i = filesize; i < filesizepadded; i++) {
			str[i] = 0;
		}

		length = filesizepadded;
		return (uint32_t*)str;
	}
	void VkFFTInitShader(uint32_t shader_id, VkShaderModule* shaderModule) {

		char filename[256];
		switch (shader_id) {
		case 0:
			//printf("vkFFT_single_c2c\n");
			sprintf(filename, "%s%s", configuration.shaderPath, "vkFFT_single_c2c.spv");
			break;
		case 1:
			//printf("vkFFT_single_c2r\n");
			sprintf(filename, "%s%s", configuration.shaderPath, "vkFFT_single_c2r.spv");
			break;
		case 2:
			//printf("vkFFT_single_c2r_zp\n");
			sprintf(filename, "%s%s", configuration.shaderPath, "vkFFT_single_c2r_zp.spv");
			break;
		case 3:
			//printf("vkFFT_single_r2c\n");
			sprintf(filename, "%s%s", configuration.shaderPath, "vkFFT_single_r2c.spv");
			break;
		case 4:
			//printf("vkFFT_single_r2c_zp\n");
			sprintf(filename, "%s%s", configuration.shaderPath, "vkFFT_single_r2c_zp.spv");
			break;
		case 5:
			//printf("vkFFT_single_c2c_afterR2C\n");
			sprintf(filename, "%s%s", configuration.shaderPath, "vkFFT_single_c2c_afterR2C.spv");
			break;
		case 6:
			//printf("vkFFT_single_c2c_beforeC2R\n");
			sprintf(filename, "%s%s", configuration.shaderPath, "vkFFT_single_c2c_beforeC2R.spv");
			break;
		case 7:
			//printf("vkFFT_grouped_c2c\n");
			sprintf(filename, "%s%s", configuration.shaderPath, "vkFFT_grouped_c2c.spv");
			break;
		case 8:
			//printf("vkFFT_grouped_convolution_1x1\n");
			sprintf(filename, "%s%s", configuration.shaderPath, "vkFFT_grouped_convolution_1x1.spv");
			break;
		case 9:
			//printf("vkFFT_single_convolution_1x1\n");
			sprintf(filename, "%s%s", configuration.shaderPath, "vkFFT_single_convolution_1x1.spv");
			break;
		case 10:
			//printf("vkFFT_single_convolution_afterR2C_1x1\n");
			sprintf(filename, "%s%s", configuration.shaderPath, "vkFFT_single_convolution_afterR2C_1x1.spv");
			break;
		case 11:
			//printf("vkFFT_grouped_convolution_symmetric_2x2\n");
			sprintf(filename, "%s%s", configuration.shaderPath, "vkFFT_grouped_convolution_symmetric_2x2.spv");
			break;
		case 12:
			//printf("vkFFT_single_convolution_symmetric_2x2\n");
			sprintf(filename, "%s%s", configuration.shaderPath, "vkFFT_single_convolution_symmetric_2x2.spv");
			break;
		case 13:
			//printf("vkFFT_single_convolution_afterR2C_symmetric_2x2\n");
			sprintf(filename, "%s%s", configuration.shaderPath, "vkFFT_single_convolution_afterR2C_symmetric_2x2.spv");
			break;
		case 14:
			//printf("vkFFT_grouped_convolution_nonsymmetric_2x2\n");
			sprintf(filename, "%s%s", configuration.shaderPath, "vkFFT_grouped_convolution_nonsymmetric_2x2.spv");
			break;
		case 15:
			//printf("vkFFT_single_convolution_nonsymmetric_2x2\n");
			sprintf(filename, "%s%s", configuration.shaderPath, "vkFFT_single_convolution_nonsymmetric_2x2.spv");
			break;
		case 16:
			//printf("vkFFT_single_convolution_afterR2C_nonsymmetric_2x2\n");
			sprintf(filename, "%s%s", configuration.shaderPath, "vkFFT_single_convolution_afterR2C_nonsymmetric_2x2.spv");
			break;
		case 17:
			//printf("vkFFT_grouped_convolution_symmetric_3x3\n");
			sprintf(filename, "%s%s", configuration.shaderPath, "vkFFT_grouped_convolution_symmetric_3x3.spv");
			break;
		case 18:
			//printf("vkFFT_single_convolution_symmetric_3x3\n");
			sprintf(filename, "%s%s", configuration.shaderPath, "vkFFT_single_convolution_symmetric_3x3.spv");
			break;
		case 19:
			//printf("vkFFT_single_convolution_afterR2C_symmetric_3x3\n");
			sprintf(filename, "%s%s", configuration.shaderPath, "vkFFT_single_convolution_afterR2C_symmetric_3x3.spv");
			break;
		case 20:
			//printf("vkFFT_grouped_convolution_nonsymmetric_3x3\n");
			sprintf(filename, "%s%s", configuration.shaderPath, "vkFFT_grouped_convolution_nonsymmetric_3x3.spv");
			break;
		case 21:
			//printf("vkFFT_single_convolution_nonsymmetric_3x3\n");
			sprintf(filename, "%s%s", configuration.shaderPath, "vkFFT_single_convolution_nonsymmetric_3x3.spv");
			break;
		case 22:
			//printf("vkFFT_single_convolution_afterR2C_nonsymmetric_3x3\n");
			sprintf(filename, "%s%s", configuration.shaderPath, "vkFFT_single_convolution_afterR2C_nonsymmetric_3x3.spv");
			break;
		case 23:
			//printf("vkFFT_single_c2r_8192\n");
			sprintf(filename, "%s%s", configuration.shaderPath, "8192/vkFFT_single_c2r_8192.spv");
			break;
		case 24:
			//printf("vkFFT_single_r2c_8192\n");
			sprintf(filename, "%s%s", configuration.shaderPath, "8192/vkFFT_single_r2c_8192.spv");
			break;
		case 25:
			//printf("vkFFT_single_c2c_8192\n");
			sprintf(filename, "%s%s", configuration.shaderPath, "8192/vkFFT_single_c2c_8192.spv");
			break;
		case 26:
			//printf("vkFFT_single_c2r_for_transposition_8192\n");
			sprintf(filename, "%s%s", configuration.shaderPath, "8192/vkFFT_single_c2r_for_transposition_8192.spv");
			break;
		case 27:
			//printf("vkFFT_single_r2c_for_transposition_8192\n");
			sprintf(filename, "%s%s", configuration.shaderPath, "8192/vkFFT_single_r2c_for_transposition_8192.spv");
			break;
		case 28:
			//printf("vkFFT_single_c2c_for_transposition_8192\n");
			sprintf(filename, "%s%s", configuration.shaderPath, "8192/vkFFT_single_c2c_for_transposition_8192.spv");
			break;
		case 29:
			//printf("vkFFT_single_c2c_afterR2C_for_transposition_8192\n");
			sprintf(filename, "%s%s", configuration.shaderPath, "8192/vkFFT_single_c2c_afterR2C_for_transposition_8192.spv");
			break;
		case 30:
			//printf("vkFFT_single_c2c_beforeC2R_for_transposition_8192\n");
			sprintf(filename, "%s%s", configuration.shaderPath, "8192/vkFFT_single_c2c_beforeC2R_for_transposition_8192.spv");
			break;

		}


		uint32_t filelength;
		uint32_t* code = VkFFTReadShader(filelength, filename);
		VkShaderModuleCreateInfo createInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
		createInfo.pCode = code;
		createInfo.codeSize = filelength;
		vkCreateShaderModule(configuration.device[0], &createInfo, NULL, shaderModule);
		delete[] code;

	}
	void VkFFTPlanAxis(VkFFTPlan* FFTPlan, uint32_t axis_id, bool inverse) {
		//get radix stages
		VkFFTAxis* axis = &FFTPlan->axes[axis_id];
		//for (uint32_t i; i<3; i++)
		//	axis->pushConstants.size[i] = configuration.size[i];

		//configure radix stages
		uint32_t logSize = log2(configuration.size[axis_id]);
		switch (configuration.radix) {
		case 8: {
			uint32_t stage8 = logSize / 3;
			uint32_t stage4 = 0;
			uint32_t stage2 = 0;
			if (logSize % 3 == 2)
				stage4 = 1;
			if (logSize % 3 == 1)
				stage2 = 1;
			axis->pushConstants.numStages = stage8 + stage4 + stage2;

			axis->pushConstants.stageRadix[0] = 8;
			axis->pushConstants.stageRadix[1] = 8;
			if (logSize % 3 == 2)
				axis->pushConstants.stageRadix[1] = 4;
			if (logSize % 3 == 1)
				axis->pushConstants.stageRadix[1] = 2;
			break;
		}
		case 4: {
			uint32_t stage4 = logSize / 2;
			uint32_t stage2 = 0;
			if (logSize % 2 == 1)
				stage2 = 1;
			axis->pushConstants.numStages = stage4 + stage2;


			axis->pushConstants.stageRadix[0] = 4;
			axis->pushConstants.stageRadix[1] = 4;
			if (logSize % 2 == 1)
				axis->pushConstants.stageRadix[1] = 2;
			break;
		}
		case 2: {
			uint32_t stage2 = logSize;

			axis->pushConstants.numStages = stage2;


			axis->pushConstants.stageRadix[0] = 2;
			axis->pushConstants.stageRadix[1] = 2;
			break;
		}
		}
		if (4096 / configuration.size[1] > 8) {
			configuration.performTranspose[0] = false;
			FFTPlan->axes[1].groupedBatch = 4096 / configuration.size[1];
		}
		else {
			configuration.performTranspose[0] = true;
		}

		if (4096 / configuration.size[2] > 8) {
			configuration.performTranspose[1] = false;
			FFTPlan->axes[2].groupedBatch = 4096 / configuration.size[2];
		}
		else {
			configuration.performTranspose[1] = true;
		}
		//configure strides
		if (configuration.performR2C)
		{
			//perform r2c
			axis->pushConstants.inputStride[0] = 1;
			axis->pushConstants.inputStride[3] = (configuration.size[0] / 2 + 1) * configuration.size[1] * configuration.size[2];
			if (axis_id == 0) {
				axis->pushConstants.inputStride[1] = configuration.size[0];
				axis->pushConstants.inputStride[2] = (configuration.size[0] / 2 + 1) * configuration.size[1];
			}
			if (axis_id == 1)
			{
				if (configuration.performTranspose[0]) {
					//transpose 0-1
					axis->pushConstants.inputStride[1] = configuration.size[1];
					axis->pushConstants.inputStride[2] = (configuration.size[0] / 2 + 1) * configuration.size[1];
				}
				else {
					//don't transpose
					axis->pushConstants.inputStride[1] = configuration.size[0] / 2;
					axis->pushConstants.inputStride[2] = (configuration.size[0] / 2 + 1) * configuration.size[1];
				}
			}
			if (axis_id == 2)
			{

				if (configuration.performTranspose[1]) {
					//transpose 0-1, transpose 1-2
					axis->pushConstants.inputStride[1] = (configuration.size[0] / 2 + 1) * configuration.size[2];
					axis->pushConstants.inputStride[2] = configuration.size[2];
				}
				else {

					if (configuration.performTranspose[0]) {
						//transpose 0-1, don't transpose 1-2
						axis->pushConstants.inputStride[1] = (configuration.size[0] / 2 + 1) * configuration.size[1];
						axis->pushConstants.inputStride[2] = configuration.size[1];
					}
					else {
						//don't transpose
						axis->pushConstants.inputStride[1] = (configuration.size[0] / 2 + 1) * configuration.size[1];
						axis->pushConstants.inputStride[2] = configuration.size[0] / 2;
					}
				}
			}

			axis->pushConstants.outputStride[0] = axis->pushConstants.inputStride[0];
			axis->pushConstants.outputStride[1] = axis->pushConstants.inputStride[1];
			axis->pushConstants.outputStride[2] = axis->pushConstants.inputStride[2];
			axis->pushConstants.outputStride[3] = axis->pushConstants.inputStride[3];
			if (axis_id == 0) {
				if ((configuration.isInputFormatted) && (!inverse)) {
					if (configuration.performZeropadding[0])
						axis->pushConstants.inputStride[1] = configuration.size[0] / 2;

					if (configuration.performZeropadding[1])
						axis->pushConstants.inputStride[2] = axis->pushConstants.inputStride[1] * configuration.size[1] / 4;
					else
						axis->pushConstants.inputStride[2] = axis->pushConstants.inputStride[1] * configuration.size[1] / 2;

					if (configuration.performZeropadding[2])
						axis->pushConstants.inputStride[3] = axis->pushConstants.inputStride[2] * configuration.size[2] / 2;
					else
						axis->pushConstants.inputStride[3] = axis->pushConstants.inputStride[2] * configuration.size[2];
				}
				if ((configuration.isOutputFormatted) && ((inverse) || ((configuration.performConvolution) && (configuration.FFTdim == 1)))) {
					if (configuration.performZeropadding[0])
						axis->pushConstants.outputStride[1] = configuration.size[0] / 2;

					if (configuration.performZeropadding[1])
						axis->pushConstants.outputStride[2] = axis->pushConstants.outputStride[1] * configuration.size[1] / 4;
					else
						axis->pushConstants.outputStride[2] = axis->pushConstants.outputStride[1] * configuration.size[1] / 2;

					if (configuration.performZeropadding[2])
						axis->pushConstants.outputStride[3] = axis->pushConstants.outputStride[2] * configuration.size[2] / 2;
					else
						axis->pushConstants.outputStride[3] = axis->pushConstants.outputStride[2] * configuration.size[2];
				}
			}
		}
		else {
			//don't perform r2c
			axis->pushConstants.inputStride[0] = 1;
			axis->pushConstants.inputStride[3] = configuration.size[0] * configuration.size[1] * configuration.size[2];
			if (axis_id == 0) {
				axis->pushConstants.inputStride[1] = configuration.size[0];
				axis->pushConstants.inputStride[2] = configuration.size[0] * configuration.size[1];
			}
			if (axis_id == 1)
			{
				if (configuration.performTranspose[0]) {
					//transpose 0-1, no transpose 1-2
					axis->pushConstants.inputStride[1] = configuration.size[1];
					axis->pushConstants.inputStride[2] = configuration.size[0] * configuration.size[1];
				}
				else {
					//no transpose
					axis->pushConstants.inputStride[1] = configuration.size[0];
					axis->pushConstants.inputStride[2] = configuration.size[0] * configuration.size[1];
				}
			}
			if (axis_id == 2)
			{

				if (configuration.performTranspose[1]) {
					//transpose 0-1, transpose 1-2
					axis->pushConstants.inputStride[1] = configuration.size[0] * configuration.size[2];
					axis->pushConstants.inputStride[2] = configuration.size[2];
				}
				else {

					if (configuration.performTranspose[0]) {
						//transpose 0-1, no transpose 1-2
						axis->pushConstants.inputStride[1] = configuration.size[0] * configuration.size[1];
						axis->pushConstants.inputStride[2] = configuration.size[1];
					}
					else {
						//no transpose
						axis->pushConstants.inputStride[1] = configuration.size[0] * configuration.size[1];
						axis->pushConstants.inputStride[2] = configuration.size[0];
					}
				}
			}

			axis->pushConstants.outputStride[0] = axis->pushConstants.inputStride[0];
			axis->pushConstants.outputStride[1] = axis->pushConstants.inputStride[1];
			axis->pushConstants.outputStride[2] = axis->pushConstants.inputStride[2];
			axis->pushConstants.outputStride[3] = axis->pushConstants.inputStride[3];
			if (axis_id == 0) {
				if ((configuration.isInputFormatted) && (!inverse)) {
					if (configuration.performZeropadding[0])
						axis->pushConstants.inputStride[1] = configuration.size[0] / 2;

					if (configuration.performZeropadding[1])
						axis->pushConstants.inputStride[2] = axis->pushConstants.inputStride[1] * configuration.size[1] / 2;
					else
						axis->pushConstants.inputStride[2] = axis->pushConstants.inputStride[1] * configuration.size[1];

					if (configuration.performZeropadding[2])
						axis->pushConstants.inputStride[3] = axis->pushConstants.inputStride[2] * configuration.size[2] / 2;
					else
						axis->pushConstants.inputStride[3] = axis->pushConstants.inputStride[2] * configuration.size[2];
				}
				if ((configuration.isOutputFormatted) && ((inverse) || ((configuration.performConvolution) && (configuration.FFTdim == 1)))) {
					if (configuration.performZeropadding[0])
						axis->pushConstants.outputStride[1] = configuration.size[0] / 2;

					if (configuration.performZeropadding[1])
						axis->pushConstants.outputStride[2] = axis->pushConstants.outputStride[1] * configuration.size[1] / 2;
					else
						axis->pushConstants.outputStride[2] = axis->pushConstants.outputStride[1] * configuration.size[1];

					if (configuration.performZeropadding[2])
						axis->pushConstants.outputStride[3] = axis->pushConstants.outputStride[2] * configuration.size[2] / 2;
					else
						axis->pushConstants.outputStride[3] = axis->pushConstants.outputStride[2] * configuration.size[2];
				}
			}
		}
		axis->pushConstants.inputStride[4] = axis->pushConstants.inputStride[3] * configuration.coordinateFeatures;
		axis->pushConstants.outputStride[4] = axis->pushConstants.outputStride[3] * configuration.coordinateFeatures;
		for (uint32_t i = 0; i < 3; ++i) {
			axis->pushConstants.radixStride[i] = configuration.size[axis_id] / pow(2, i + 1);

		}

		axis->pushConstants.inverse = inverse;
		axis->pushConstants.zeropad[0] = configuration.performZeropadding[axis_id];
		if (axis_id == 0)
			axis->pushConstants.zeropad[1] = configuration.performZeropadding[axis_id + 1];
		else
			axis->pushConstants.zeropad[1] = false;

		if (!inverse) {
			switch (axis_id) {
			case 0:
				axis->pushConstants.ratio[0] = 1;
				axis->pushConstants.ratioDirection[0] = false;
				if (configuration.FFTdim > 1) {
					if (configuration.performR2C) {
						axis->pushConstants.ratio[1] = (configuration.size[0] / configuration.size[1] / 2 >= 1) ? configuration.size[0] / configuration.size[1] / 2 : 2 * configuration.size[1] / configuration.size[0];
						axis->pushConstants.ratioDirection[1] = (configuration.size[0] / configuration.size[1] / 2 >= 1) ? true : false;
					}
					else {
						axis->pushConstants.ratio[1] = (configuration.size[0] / configuration.size[1] >= 1) ? configuration.size[0] / configuration.size[1] : configuration.size[1] / configuration.size[0];
						axis->pushConstants.ratioDirection[1] = (configuration.size[0] / configuration.size[1] >= 1) ? true : false;

					}
				}
				if (!configuration.performTranspose[0]) {
					axis->pushConstants.ratioDirection[0] = false;
					axis->pushConstants.ratioDirection[1] = true;
				}
				break;
			case 1:
				if (configuration.performR2C) {
					if (configuration.size[0] / configuration.size[1] / 2 >= 1) {
						axis->pushConstants.ratio[0] = configuration.size[0] / configuration.size[1] / 2;
						axis->pushConstants.ratioDirection[0] = true;
					}
					else
					{
						axis->pushConstants.ratio[0] = (configuration.size[1] * 2 / configuration.size[0]);
						axis->pushConstants.ratioDirection[0] = false;
					}
				}
				else {
					if (configuration.size[0] / configuration.size[1] >= 1) {
						axis->pushConstants.ratio[0] = configuration.size[0] / configuration.size[1];
						axis->pushConstants.ratioDirection[0] = true;
					}
					else {
						axis->pushConstants.ratio[0] = configuration.size[1] / configuration.size[0];
						axis->pushConstants.ratioDirection[0] = false;
					}
				}
				if ((configuration.performConvolution) && (configuration.FFTdim == 2)) {
					if (configuration.performR2C) {
						if (configuration.size[0] / configuration.size[1] / 2 >= 1) {
							axis->pushConstants.ratio[1] = configuration.size[0] / configuration.size[1] / 2;
							axis->pushConstants.ratioDirection[1] = false;
						}
						else
						{
							axis->pushConstants.ratio[1] = (configuration.size[1] * 2 / configuration.size[0]);
							axis->pushConstants.ratioDirection[1] = true;
						}
					}
					else {
						if (configuration.size[0] / configuration.size[1] >= 1) {
							axis->pushConstants.ratio[1] = configuration.size[0] / configuration.size[1];
							axis->pushConstants.ratioDirection[1] = false;
						}
						else {
							axis->pushConstants.ratio[1] = configuration.size[1] / configuration.size[0];
							axis->pushConstants.ratioDirection[1] = true;
						}
					}
				}
				if (configuration.FFTdim > 2) {
					if (configuration.size[1] / configuration.size[2] >= 1) {
						axis->pushConstants.ratio[1] = configuration.size[1] / configuration.size[2];
						axis->pushConstants.ratioDirection[1] = true;
					}
					else
					{
						axis->pushConstants.ratio[1] = (configuration.size[2] / configuration.size[1]);
						axis->pushConstants.ratioDirection[1] = false;
					}
				}
				if (!configuration.performTranspose[0]) {
					axis->pushConstants.ratioDirection[0] = false;
				}
				if ((!configuration.performTranspose[1]) && (!((configuration.performConvolution) && (configuration.FFTdim == 2)))) {
					axis->pushConstants.ratioDirection[1] = true;
				}
				break;
			case 2:
				if (configuration.size[1] / configuration.size[2] >= 1) {
					axis->pushConstants.ratio[0] = configuration.size[1] / configuration.size[2];
					axis->pushConstants.ratioDirection[0] = true;
				}
				else {
					axis->pushConstants.ratio[0] = configuration.size[2] / configuration.size[1];
					axis->pushConstants.ratioDirection[0] = false;
				}
				axis->pushConstants.ratio[1] = 1;
				axis->pushConstants.ratioDirection[1] = true;
				if ((configuration.performConvolution) && (configuration.FFTdim == 3)) {
					if (configuration.size[1] / configuration.size[2] >= 1) {
						axis->pushConstants.ratio[1] = configuration.size[1] / configuration.size[2];
						axis->pushConstants.ratioDirection[1] = false;
					}
					else {
						axis->pushConstants.ratio[1] = configuration.size[2] / configuration.size[1];
						axis->pushConstants.ratioDirection[1] = true;
					}
				}
				if (!configuration.performTranspose[1]) {
					axis->pushConstants.ratioDirection[0] = false;
					axis->pushConstants.ratioDirection[1] = true;
				}

				break;
			}
		}
		else {
			switch (axis_id) {
			case 0:
				axis->pushConstants.ratio[1] = 1;
				axis->pushConstants.ratioDirection[1] = true;
				if (configuration.FFTdim > 1) {
					if (configuration.performR2C) {
						axis->pushConstants.ratio[0] = (configuration.size[0] / configuration.size[1] / 2 >= 1) ? configuration.size[0] / configuration.size[1] / 2 : 2 * configuration.size[1] / configuration.size[0];
						axis->pushConstants.ratioDirection[0] = (configuration.size[0] / configuration.size[1] / 2 >= 1) ? false : true;
					}
					else
					{
						axis->pushConstants.ratio[0] = (configuration.size[0] / configuration.size[1] >= 1) ? configuration.size[0] / configuration.size[1] : configuration.size[1] / configuration.size[0];
						axis->pushConstants.ratioDirection[0] = (configuration.size[0] / configuration.size[1] >= 1) ? false : true;

					}
				}
				if (!configuration.performTranspose[0]) {
					axis->pushConstants.ratioDirection[0] = false;
					axis->pushConstants.ratioDirection[1] = true;
				}
				break;
			case 1:
				if (configuration.performR2C) {
					if (configuration.size[0] / configuration.size[1] / 2 >= 1) {
						axis->pushConstants.ratio[1] = configuration.size[0] / configuration.size[1] / 2;
						axis->pushConstants.ratioDirection[1] = false;
					}
					else
					{
						axis->pushConstants.ratio[1] = (configuration.size[1] * 2 / configuration.size[0]);
						axis->pushConstants.ratioDirection[1] = true;
					}
				}
				else {
					if (configuration.size[0] / configuration.size[1] >= 1) {
						axis->pushConstants.ratio[1] = configuration.size[0] / configuration.size[1];
						axis->pushConstants.ratioDirection[1] = false;
					}
					else {
						axis->pushConstants.ratio[1] = configuration.size[1] / configuration.size[0];
						axis->pushConstants.ratioDirection[1] = true;
					}
				}
				if (configuration.FFTdim > 2) {
					if (configuration.size[1] / configuration.size[2] >= 1) {
						axis->pushConstants.ratio[0] = configuration.size[1] / configuration.size[2];
						axis->pushConstants.ratioDirection[0] = false;
					}
					else
					{
						axis->pushConstants.ratio[0] = (configuration.size[2] / configuration.size[1]);
						axis->pushConstants.ratioDirection[0] = true;
					}
				}
				if (!configuration.performTranspose[0]) {
					axis->pushConstants.ratioDirection[1] = true;
				}
				if (!configuration.performTranspose[1]) {
					axis->pushConstants.ratioDirection[0] = false;
				}
				break;
			case 2:
				if (configuration.size[1] / configuration.size[2] >= 1) {
					axis->pushConstants.ratio[1] = configuration.size[1] / configuration.size[2];
					axis->pushConstants.ratioDirection[1] = false;
				}
				else {
					axis->pushConstants.ratio[1] = configuration.size[2] / configuration.size[1];
					axis->pushConstants.ratioDirection[1] = true;
				}
				axis->pushConstants.ratio[0] = 1;
				axis->pushConstants.ratioDirection[0] = false;
				if (!configuration.performTranspose[1]) {
					axis->pushConstants.ratioDirection[1] = true;
				}
				break;
			}
		}
		axis->pushConstants.inputOffset = 0;
		axis->pushConstants.outputOffset = 0;

		VkDescriptorPoolSize descriptorPoolSize = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
		descriptorPoolSize.descriptorCount = 2;
		if ((axis_id == 0) && (configuration.FFTdim == 1) && (configuration.performConvolution))
			descriptorPoolSize.descriptorCount = 3;
		if ((axis_id == 1) && (configuration.FFTdim == 2) && (configuration.performConvolution))
			descriptorPoolSize.descriptorCount = 3;
		if ((axis_id == 2) && (configuration.FFTdim == 3) && (configuration.performConvolution))
			descriptorPoolSize.descriptorCount = 3;

		VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
		descriptorPoolCreateInfo.poolSizeCount = 1;
		descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize;
		descriptorPoolCreateInfo.maxSets = 1;
		vkCreateDescriptorPool(configuration.device[0], &descriptorPoolCreateInfo, NULL, &axis->descriptorPool);

		const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
		VkDescriptorSetLayoutBinding* descriptorSetLayoutBindings;
		descriptorSetLayoutBindings = (VkDescriptorSetLayoutBinding*)malloc(descriptorPoolSize.descriptorCount * sizeof(VkDescriptorSetLayoutBinding));
		for (uint32_t i = 0; i < descriptorPoolSize.descriptorCount; ++i) {
			descriptorSetLayoutBindings[i].binding = i;
			descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
			descriptorSetLayoutBindings[i].descriptorCount = 1;
			descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		}

		VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
		descriptorSetLayoutCreateInfo.bindingCount = descriptorPoolSize.descriptorCount;
		descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;

		vkCreateDescriptorSetLayout(configuration.device[0], &descriptorSetLayoutCreateInfo, NULL, &axis->descriptorSetLayout);

		VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
		descriptorSetAllocateInfo.descriptorPool = axis->descriptorPool;
		descriptorSetAllocateInfo.descriptorSetCount = 1;
		descriptorSetAllocateInfo.pSetLayouts = &axis->descriptorSetLayout;
		vkAllocateDescriptorSets(configuration.device[0], &descriptorSetAllocateInfo, &axis->descriptorSet);
		for (uint32_t i = 0; i < descriptorPoolSize.descriptorCount; ++i) {
			VkDescriptorBufferInfo descriptorBufferInfo = {};

			if (i == 0) {
				if (configuration.isInputFormatted && (
					((axis_id == 0) && (!inverse))
					|| ((axis_id == configuration.FFTdim-1) && (inverse)))
					) {
					descriptorBufferInfo.buffer = configuration.inputBuffer[0];
					descriptorBufferInfo.range = configuration.inputBufferSize[0];
				}
				else {
					if ((configuration.numberKernels > 1) && (inverse)) {
						descriptorBufferInfo.buffer = configuration.outputBuffer[0];
						descriptorBufferInfo.range = configuration.outputBufferSize[0];
					}
					else {
						descriptorBufferInfo.buffer = configuration.buffer[0];
						descriptorBufferInfo.range = configuration.bufferSize[0];
					}
				}
				descriptorBufferInfo.offset = 0;
			}
			if (i == 1) {
				if ((configuration.isOutputFormatted && (
					((axis_id == 0) && (inverse))
					|| ((axis_id == configuration.FFTdim-1) && (!inverse) && (!configuration.performConvolution))
					|| ((axis_id == 0) && (configuration.performConvolution) && (configuration.FFTdim == 1)))
					) ||
					((configuration.numberKernels > 1) && (
						(inverse)
						|| (axis_id == configuration.FFTdim-1)))
					) {
					descriptorBufferInfo.buffer = configuration.outputBuffer[0];
					descriptorBufferInfo.range = configuration.outputBufferSize[0];
				}
				else {
					descriptorBufferInfo.buffer = configuration.buffer[0];
					descriptorBufferInfo.range = configuration.bufferSize[0];
				}
				descriptorBufferInfo.offset = 0;
			}
			if (i == 2) {
				descriptorBufferInfo.buffer = configuration.kernel[0];
				descriptorBufferInfo.offset = 0;
				descriptorBufferInfo.range = configuration.kernelSize[0];
			}
			VkWriteDescriptorSet writeDescriptorSet = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
			writeDescriptorSet.dstSet = axis->descriptorSet;
			writeDescriptorSet.dstBinding = i;
			writeDescriptorSet.dstArrayElement = 0;
			writeDescriptorSet.descriptorType = descriptorType[i];
			writeDescriptorSet.descriptorCount = 1;
			writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
			vkUpdateDescriptorSets(configuration.device[0], 1, &writeDescriptorSet, 0, NULL);

		}

		{
			VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
			pipelineLayoutCreateInfo.setLayoutCount = 1;
			pipelineLayoutCreateInfo.pSetLayouts = &axis->descriptorSetLayout;
			VkPushConstantRange pushConstantRange = { VK_SHADER_STAGE_COMPUTE_BIT };
			pushConstantRange.offset = 0;
			pushConstantRange.size = sizeof(axis->pushConstants);
			// Push constant ranges are part of the pipeline layout
			pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
			pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
			vkCreatePipelineLayout(configuration.device[0], &pipelineLayoutCreateInfo, NULL, &axis->pipelineLayout);
			if (!inverse) {
				if (axis_id == 0) {
					FFTPlan->axes[axis_id].axisBlock[0] = (configuration.size[axis_id] / 8 > 1) ? configuration.size[axis_id] / 8 : 1;
					if (FFTPlan->axes[axis_id].axisBlock[0] > 512) FFTPlan->axes[axis_id].axisBlock[0] = 512;
					if (configuration.performR2C)
						FFTPlan->axes[axis_id].axisBlock[1] = (FFTPlan->axes[axis_id].pushConstants.ratioDirection[1]) ? 1 : FFTPlan->axes[axis_id].pushConstants.ratio[1] / 2;
					else
						FFTPlan->axes[axis_id].axisBlock[1] = (FFTPlan->axes[axis_id].pushConstants.ratioDirection[1]) ? 1 : FFTPlan->axes[axis_id].pushConstants.ratio[1];

					FFTPlan->axes[axis_id].axisBlock[2] = 1;
					FFTPlan->axes[axis_id].axisBlock[3] = configuration.size[axis_id];
				}
				if (axis_id == 1) {
					if (configuration.performTranspose[0]) {
						VkFFTPlanTranspose(FFTPlan, 0, inverse);
						FFTPlan->axes[axis_id].axisBlock[0] = (configuration.size[axis_id] / 8 > 1) ? configuration.size[axis_id] / 8 : 1;
						if (FFTPlan->axes[axis_id].axisBlock[0] > 512) FFTPlan->axes[axis_id].axisBlock[0] = 512;
						FFTPlan->axes[axis_id].axisBlock[1] = (FFTPlan->axes[axis_id].pushConstants.ratioDirection[0]) ? FFTPlan->axes[axis_id].pushConstants.ratio[0] : 1;
						FFTPlan->axes[axis_id].axisBlock[2] = 1;
						FFTPlan->axes[axis_id].axisBlock[3] = configuration.size[axis_id];
					}
					else {
						if (configuration.performR2C) {
							VkFFTPlanSupportAxis(FFTPlan, 1, inverse);
							FFTPlan->axes[axis_id].axisBlock[0] = (configuration.size[0] / 2 > FFTPlan->axes[axis_id].groupedBatch) ? FFTPlan->axes[axis_id].groupedBatch : configuration.size[0] / 2;
						}
						else
							FFTPlan->axes[axis_id].axisBlock[0] = (configuration.size[0] > FFTPlan->axes[axis_id].groupedBatch) ? FFTPlan->axes[axis_id].groupedBatch : configuration.size[0];
						FFTPlan->axes[axis_id].axisBlock[1] = (configuration.size[axis_id] / 8 > 1) ? configuration.size[axis_id] / 8 : 1;
						FFTPlan->axes[axis_id].axisBlock[2] = 1;
						FFTPlan->axes[axis_id].axisBlock[3] = configuration.size[axis_id];
					}
				}
				if (axis_id == 2) {
					if (configuration.performTranspose[1]) {
						VkFFTPlanTranspose(FFTPlan, 1, inverse);
						FFTPlan->axes[axis_id].axisBlock[0] = (configuration.size[axis_id] / 8 > 1) ? configuration.size[axis_id] / 8 : 1;
						FFTPlan->axes[axis_id].axisBlock[1] = (FFTPlan->axes[axis_id].pushConstants.ratioDirection[0]) ? FFTPlan->axes[axis_id].pushConstants.ratio[0] : 1;
						FFTPlan->axes[axis_id].axisBlock[2] = 1;
						FFTPlan->axes[axis_id].axisBlock[3] = configuration.size[axis_id];
					}
					else {
						if (configuration.performTranspose[0]) {
							FFTPlan->axes[axis_id].axisBlock[0] = (configuration.size[1] > FFTPlan->axes[axis_id].groupedBatch) ? FFTPlan->axes[axis_id].groupedBatch : configuration.size[1];
							FFTPlan->axes[axis_id].axisBlock[1] = (configuration.size[axis_id] / 8 > 1) ? configuration.size[axis_id] / 8 : 1;
							FFTPlan->axes[axis_id].axisBlock[2] = 1;
							FFTPlan->axes[axis_id].axisBlock[3] = configuration.size[axis_id];
						}
						else {
							if (configuration.performR2C) {
								VkFFTPlanSupportAxis(FFTPlan, 2, inverse);
								FFTPlan->axes[axis_id].axisBlock[0] = (configuration.size[0] / 2 > FFTPlan->axes[axis_id].groupedBatch) ? FFTPlan->axes[axis_id].groupedBatch : configuration.size[0] / 2;
							}
							else
								FFTPlan->axes[axis_id].axisBlock[0] = (configuration.size[0] > FFTPlan->axes[axis_id].groupedBatch) ? FFTPlan->axes[axis_id].groupedBatch : configuration.size[0];
							FFTPlan->axes[axis_id].axisBlock[1] = (configuration.size[axis_id] / 8 > 1) ? configuration.size[axis_id] / 8 : 1;
							FFTPlan->axes[axis_id].axisBlock[2] = 1;
							FFTPlan->axes[axis_id].axisBlock[3] = configuration.size[axis_id];
						}
					}
				}
			}
			else {
				if (axis_id == 0) {
					FFTPlan->axes[axis_id].axisBlock[0] = (configuration.size[axis_id] / 8 > 1) ? configuration.size[axis_id] / 8 : 1;
					if (FFTPlan->axes[axis_id].axisBlock[0] > 512) FFTPlan->axes[axis_id].axisBlock[0] = 512;
					if (configuration.performR2C)
						FFTPlan->axes[axis_id].axisBlock[1] = (FFTPlan->axes[axis_id].pushConstants.ratioDirection[0]) ? FFTPlan->axes[axis_id].pushConstants.ratio[0] / 2 : 1;
					else
						FFTPlan->axes[axis_id].axisBlock[1] = (FFTPlan->axes[axis_id].pushConstants.ratioDirection[0]) ? FFTPlan->axes[axis_id].pushConstants.ratio[0] : 1;
					FFTPlan->axes[axis_id].axisBlock[2] = 1;
					FFTPlan->axes[axis_id].axisBlock[3] = configuration.size[axis_id];
				}
				if (axis_id == 1) {
					if (configuration.performTranspose[0]) {
						VkFFTPlanTranspose(FFTPlan, 0, inverse);
						FFTPlan->axes[axis_id].axisBlock[0] = (configuration.size[axis_id] / 8 > 1) ? configuration.size[axis_id] / 8 : 1;
						if (FFTPlan->axes[axis_id].axisBlock[0] > 512) FFTPlan->axes[axis_id].axisBlock[0] = 512;
						FFTPlan->axes[axis_id].axisBlock[1] = (FFTPlan->axes[axis_id].pushConstants.ratioDirection[1]) ? 1 : FFTPlan->axes[axis_id].pushConstants.ratio[1];
						FFTPlan->axes[axis_id].axisBlock[2] = 1;
						FFTPlan->axes[axis_id].axisBlock[3] = configuration.size[axis_id];
					}
					else {
						if (configuration.performR2C) {
							VkFFTPlanSupportAxis(FFTPlan, 1, inverse);
							FFTPlan->axes[axis_id].axisBlock[0] = (configuration.size[0] / 2 > FFTPlan->axes[axis_id].groupedBatch) ? FFTPlan->axes[axis_id].groupedBatch : configuration.size[0] / 2;
						}
						else
							FFTPlan->axes[axis_id].axisBlock[0] = (configuration.size[0] > FFTPlan->axes[axis_id].groupedBatch) ? FFTPlan->axes[axis_id].groupedBatch : configuration.size[0];
						FFTPlan->axes[axis_id].axisBlock[1] = (configuration.size[axis_id] / 8 > 1) ? configuration.size[axis_id] / 8 : 1;
						FFTPlan->axes[axis_id].axisBlock[2] = 1;
						FFTPlan->axes[axis_id].axisBlock[3] = configuration.size[axis_id];
					}
				}
				if (axis_id == 2) {
					if (configuration.performTranspose[1]) {
						VkFFTPlanTranspose(FFTPlan, 1, inverse);
						FFTPlan->axes[axis_id].axisBlock[0] = (configuration.size[axis_id] / 8 > 1) ? configuration.size[axis_id] / 8 : 1;
						FFTPlan->axes[axis_id].axisBlock[1] = (FFTPlan->axes[axis_id].pushConstants.ratioDirection[1]) ? 1 : FFTPlan->axes[axis_id].pushConstants.ratio[1];
						FFTPlan->axes[axis_id].axisBlock[2] = 1;
						FFTPlan->axes[axis_id].axisBlock[3] = configuration.size[axis_id];
					}
					else {
						if (configuration.performTranspose[0]) {
							FFTPlan->axes[axis_id].axisBlock[0] = (configuration.size[1] > FFTPlan->axes[axis_id].groupedBatch) ? FFTPlan->axes[axis_id].groupedBatch : configuration.size[1];
							FFTPlan->axes[axis_id].axisBlock[1] = (configuration.size[axis_id] / 8 > 1) ? configuration.size[axis_id] / 8 : 1;
							FFTPlan->axes[axis_id].axisBlock[2] = 1;
							FFTPlan->axes[axis_id].axisBlock[3] = configuration.size[axis_id];
						}
						else {
							if (configuration.performR2C) {
								VkFFTPlanSupportAxis(FFTPlan, 2, inverse);
								FFTPlan->axes[axis_id].axisBlock[0] = (configuration.size[0] / 2 > FFTPlan->axes[axis_id].groupedBatch) ? FFTPlan->axes[axis_id].groupedBatch : configuration.size[0] / 2;
							}
							else
								FFTPlan->axes[axis_id].axisBlock[0] = (configuration.size[0] > FFTPlan->axes[axis_id].groupedBatch) ? FFTPlan->axes[axis_id].groupedBatch : configuration.size[0];
							FFTPlan->axes[axis_id].axisBlock[1] = (configuration.size[axis_id] / 8 > 1) ? configuration.size[axis_id] / 8 : 1;
							FFTPlan->axes[axis_id].axisBlock[2] = 1;
							FFTPlan->axes[axis_id].axisBlock[3] = configuration.size[axis_id];
						}
					}
				}

			}
			VkSpecializationMapEntry specializationMapEntries[4] = { {} };
			specializationMapEntries[0].constantID = 1;
			specializationMapEntries[0].size = sizeof(uint32_t);
			specializationMapEntries[0].offset = 0;
			specializationMapEntries[1].constantID = 2;
			specializationMapEntries[1].size = sizeof(uint32_t);
			specializationMapEntries[1].offset = sizeof(uint32_t);
			specializationMapEntries[2].constantID = 3;
			specializationMapEntries[2].size = sizeof(uint32_t);
			specializationMapEntries[2].offset = 2 * sizeof(uint32_t);
			specializationMapEntries[3].constantID = 4;
			specializationMapEntries[3].size = sizeof(uint32_t);
			specializationMapEntries[3].offset = 3 * sizeof(uint32_t);

			VkSpecializationInfo specializationInfo = {};
			specializationInfo.dataSize = 4 * sizeof(uint32_t);
			specializationInfo.mapEntryCount = 4;
			specializationInfo.pMapEntries = specializationMapEntries;
			specializationInfo.pData = &FFTPlan->axes[axis_id].axisBlock;
			VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };

			VkComputePipelineCreateInfo computePipelineCreateInfo = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };


			pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			if (configuration.performR2C) {
				if (axis_id == 0) {
					if (inverse) {
						switch (configuration.size[axis_id]) {
						case 8192:
							VkFFTInitShader(23, &pipelineShaderStageCreateInfo.module);
							break;
						default:
							switch (configuration.size[axis_id+1]) {
							case 8192:
								VkFFTInitShader(26, &pipelineShaderStageCreateInfo.module);
								break;
							default:
								VkFFTInitShader(1, &pipelineShaderStageCreateInfo.module);
								break;
							}
							break;
						}
					}
					else {
						switch (configuration.size[axis_id]) {
						case 8192:
							VkFFTInitShader(24, &pipelineShaderStageCreateInfo.module);
							break;
						default:
							switch (configuration.size[axis_id + 1]) {
							case 8192:
								VkFFTInitShader(27, &pipelineShaderStageCreateInfo.module);
								break;
							default:
								VkFFTInitShader(3, &pipelineShaderStageCreateInfo.module);
								break;
							}
							break;
						}

					}
				}
				if (axis_id == 1) {

					if ((configuration.FFTdim == 2) && (configuration.performConvolution)) {
						if (configuration.performTranspose[0])
							switch (configuration.matrixConvolution) {
							case 1:
								VkFFTInitShader(10, &pipelineShaderStageCreateInfo.module);
								break;
							case 2:
								if (configuration.symmetricKernel)
									VkFFTInitShader(13, &pipelineShaderStageCreateInfo.module);
								else
									VkFFTInitShader(16, &pipelineShaderStageCreateInfo.module);
								break;
							case 3:
								if (configuration.symmetricKernel)
									VkFFTInitShader(19, &pipelineShaderStageCreateInfo.module);
								else
									VkFFTInitShader(22, &pipelineShaderStageCreateInfo.module);
								break;
							}
						else
							switch (configuration.matrixConvolution) {
							case 1:
								VkFFTInitShader(8, &pipelineShaderStageCreateInfo.module);
								break;
							case 2:
								if (configuration.symmetricKernel)
									VkFFTInitShader(11, &pipelineShaderStageCreateInfo.module);
								else
									VkFFTInitShader(14, &pipelineShaderStageCreateInfo.module);
								break;
							case 3:
								if (configuration.symmetricKernel)
									VkFFTInitShader(17, &pipelineShaderStageCreateInfo.module);
								else
									VkFFTInitShader(20, &pipelineShaderStageCreateInfo.module);
								break;
							}

					}
					else {
						if (configuration.performTranspose[0]) {
							switch (configuration.size[axis_id]) {
							case 8192:
								VkFFTInitShader(25, &pipelineShaderStageCreateInfo.module);
								break;
							default:
								if (inverse)
									VkFFTInitShader(6, &pipelineShaderStageCreateInfo.module);
								else
									VkFFTInitShader(5, &pipelineShaderStageCreateInfo.module);
								break;
								/*switch (configuration.size[axis_id - 1]) {
								case 8192:
									if (inverse) 
										VkFFTInitShader(30, &pipelineShaderStageCreateInfo.module);
									else
										VkFFTInitShader(29, &pipelineShaderStageCreateInfo.module);
									break;
								default:
									if (inverse)
										VkFFTInitShader(6, &pipelineShaderStageCreateInfo.module);
									else
										VkFFTInitShader(5, &pipelineShaderStageCreateInfo.module);
									break;
								}
								break;*/
							}

						}
						else {
							VkFFTInitShader(7, &pipelineShaderStageCreateInfo.module);
						}
					}

				}

				if (axis_id == 2) {
					if ((configuration.FFTdim == 3) && (configuration.performConvolution)) {
						if (configuration.performTranspose[1])
							switch (configuration.matrixConvolution) {
							case 1:
								VkFFTInitShader(9, &pipelineShaderStageCreateInfo.module);
								break;
							case 2:
								if (configuration.symmetricKernel)
									VkFFTInitShader(12, &pipelineShaderStageCreateInfo.module);
								else
									VkFFTInitShader(15, &pipelineShaderStageCreateInfo.module);
								break;
							case 3:
								if (configuration.symmetricKernel)
									VkFFTInitShader(18, &pipelineShaderStageCreateInfo.module);
								else
									VkFFTInitShader(21, &pipelineShaderStageCreateInfo.module);
								break;
							}
						else
							switch (configuration.matrixConvolution) {
							case 1:
								VkFFTInitShader(8, &pipelineShaderStageCreateInfo.module);
								break;
							case 2:
								if (configuration.symmetricKernel)
									VkFFTInitShader(11, &pipelineShaderStageCreateInfo.module);
								else
									VkFFTInitShader(14, &pipelineShaderStageCreateInfo.module);
								break;
							case 3:
								if (configuration.symmetricKernel)
									VkFFTInitShader(17, &pipelineShaderStageCreateInfo.module);
								else
									VkFFTInitShader(20, &pipelineShaderStageCreateInfo.module);
								break;
							}

					}
					else {
						if (configuration.performTranspose[1]) {

							switch (configuration.size[axis_id]) {
							case 8192:
								VkFFTInitShader(25, &pipelineShaderStageCreateInfo.module);
								break;
							default:
								switch (configuration.size[axis_id - 1]) {
								case 8192:
									VkFFTInitShader(28, &pipelineShaderStageCreateInfo.module);
									break;
								default:
									VkFFTInitShader(0, &pipelineShaderStageCreateInfo.module);
									break;
								}
								break;
							}
							
						}
						else
							VkFFTInitShader(7, &pipelineShaderStageCreateInfo.module);
					}
				}
			}
			else {
				if (axis_id == 0) {
					if ((configuration.FFTdim == 1) && (configuration.performConvolution)) {

						switch (configuration.matrixConvolution) {
						case 1:
							VkFFTInitShader(9, &pipelineShaderStageCreateInfo.module);
							break;
						case 2:
							if (configuration.symmetricKernel)
								VkFFTInitShader(12, &pipelineShaderStageCreateInfo.module);
							else
								VkFFTInitShader(15, &pipelineShaderStageCreateInfo.module);
							break;
						case 3:
							if (configuration.symmetricKernel)
								VkFFTInitShader(18, &pipelineShaderStageCreateInfo.module);
							else
								VkFFTInitShader(21, &pipelineShaderStageCreateInfo.module);
							break;
						}
					}
					else {
						switch (configuration.size[axis_id]) {
						case 8192:
							VkFFTInitShader(25, &pipelineShaderStageCreateInfo.module);
							break;
						default:
							switch (configuration.size[axis_id + 1]) {
							case 8192:
								VkFFTInitShader(28, &pipelineShaderStageCreateInfo.module);
								break;
							default:
								VkFFTInitShader(0, &pipelineShaderStageCreateInfo.module);
								break;
							}
							break;
						}

					}
				}
				if (axis_id == 1) {

					if ((configuration.FFTdim == 2) && (configuration.performConvolution)) {
						if (configuration.performTranspose[0])
							switch (configuration.matrixConvolution) {
							case 1:
								VkFFTInitShader(10, &pipelineShaderStageCreateInfo.module);
								break;
							case 2:
								if (configuration.symmetricKernel)
									VkFFTInitShader(13, &pipelineShaderStageCreateInfo.module);
								else
									VkFFTInitShader(16, &pipelineShaderStageCreateInfo.module);
								break;
							case 3:
								if (configuration.symmetricKernel)
									VkFFTInitShader(19, &pipelineShaderStageCreateInfo.module);
								else
									VkFFTInitShader(22, &pipelineShaderStageCreateInfo.module);
								break;
							}
						else
							switch (configuration.matrixConvolution) {
							case 1:
								VkFFTInitShader(8, &pipelineShaderStageCreateInfo.module);
								break;
							case 2:
								if (configuration.symmetricKernel)
									VkFFTInitShader(11, &pipelineShaderStageCreateInfo.module);
								else
									VkFFTInitShader(14, &pipelineShaderStageCreateInfo.module);
								break;
							case 3:
								if (configuration.symmetricKernel)
									VkFFTInitShader(17, &pipelineShaderStageCreateInfo.module);
								else
									VkFFTInitShader(20, &pipelineShaderStageCreateInfo.module);
								break;
							}
					}
					else {
						if (configuration.performTranspose[0]) {
							switch (configuration.size[axis_id]) {
							case 8192:
								VkFFTInitShader(25, &pipelineShaderStageCreateInfo.module);
								break;
							default:
								switch (configuration.size[axis_id - 1]) {
								case 8192:
									VkFFTInitShader(28, &pipelineShaderStageCreateInfo.module);
									break;
								default:
									VkFFTInitShader(0, &pipelineShaderStageCreateInfo.module);
									break;
								}
								break;
							}
						}
						else {
							VkFFTInitShader(7, &pipelineShaderStageCreateInfo.module);
						}
					}

				}

				if (axis_id == 2) {
					if ((configuration.FFTdim == 3) && (configuration.performConvolution)) {
						if (configuration.performTranspose[1])
							switch (configuration.matrixConvolution) {
							case 1:
								VkFFTInitShader(9, &pipelineShaderStageCreateInfo.module);
								break;
							case 2:
								if (configuration.symmetricKernel)
									VkFFTInitShader(12, &pipelineShaderStageCreateInfo.module);
								else
									VkFFTInitShader(15, &pipelineShaderStageCreateInfo.module);
								break;
							case 3:
								if (configuration.symmetricKernel)
									VkFFTInitShader(18, &pipelineShaderStageCreateInfo.module);
								else
									VkFFTInitShader(21, &pipelineShaderStageCreateInfo.module);
								break;
							}
						else
							switch (configuration.matrixConvolution) {
							case 1:
								VkFFTInitShader(8, &pipelineShaderStageCreateInfo.module);
								break;
							case 2:
								if (configuration.symmetricKernel)
									VkFFTInitShader(11, &pipelineShaderStageCreateInfo.module);
								else
									VkFFTInitShader(14, &pipelineShaderStageCreateInfo.module);
								break;
							case 3:
								if (configuration.symmetricKernel)
									VkFFTInitShader(17, &pipelineShaderStageCreateInfo.module);
								else
									VkFFTInitShader(20, &pipelineShaderStageCreateInfo.module);
								break;
							}
					}
					else {
						if (configuration.performTranspose[1])
							switch (configuration.size[axis_id]) {
							case 8192:
								VkFFTInitShader(25, &pipelineShaderStageCreateInfo.module);
								break;
							default:
								switch (configuration.size[axis_id - 1]) {
								case 8192:
									VkFFTInitShader(28, &pipelineShaderStageCreateInfo.module);
									break;
								default:
									VkFFTInitShader(0, &pipelineShaderStageCreateInfo.module);
									break;
								}
								break;
							}
						else
							VkFFTInitShader(7, &pipelineShaderStageCreateInfo.module);
					}
				}
			}

			pipelineShaderStageCreateInfo.pName = "main";
			pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
			computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
			computePipelineCreateInfo.layout = axis->pipelineLayout;



			vkCreateComputePipelines(configuration.device[0], VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, &axis->pipeline);
			vkDestroyShaderModule(configuration.device[0], pipelineShaderStageCreateInfo.module, NULL);
		}


	}
	void VkFFTPlanSupportAxis(VkFFTPlan* FFTPlan, uint32_t axis_id, bool inverse) {
		//get radix stages
		VkFFTAxis* axis = &FFTPlan->supportAxes[axis_id - 1];
		//for (uint32_t i; i<3; i++)
		//	axis->pushConstants.size[i] = configuration.size[i];

		//configure radix stages
		uint32_t logSize = log2(configuration.size[axis_id]);

		switch (configuration.radix) {
		case 8: {
			uint32_t stage8 = logSize / 3;
			uint32_t stage4 = 0;
			uint32_t stage2 = 0;
			if (logSize % 3 == 2)
				stage4 = 1;
			if (logSize % 3 == 1)
				stage2 = 1;
			axis->pushConstants.numStages = stage8 + stage4 + stage2;

			axis->pushConstants.stageRadix[0] = 8;
			axis->pushConstants.stageRadix[1] = 8;
			if (logSize % 3 == 2)
				axis->pushConstants.stageRadix[1] = 4;
			if (logSize % 3 == 1)
				axis->pushConstants.stageRadix[1] = 2;
			break;
		}
		case 4: {
			uint32_t stage4 = logSize / 2;
			uint32_t stage2 = 0;
			if (logSize % 2 == 1)
				stage2 = 1;
			axis->pushConstants.numStages = stage4 + stage2;


			axis->pushConstants.stageRadix[0] = 4;
			axis->pushConstants.stageRadix[1] = 4;
			if (logSize % 2 == 1)
				axis->pushConstants.stageRadix[1] = 2;
			break;
		}
		case 2: {
			uint32_t stage2 = logSize;

			axis->pushConstants.numStages = stage2;


			axis->pushConstants.stageRadix[0] = 2;
			axis->pushConstants.stageRadix[1] = 2;
			break;
		}
		}

		//configure strides
		//perform r2c
		axis->pushConstants.inputStride[0] = 1;
		axis->pushConstants.inputStride[3] = (configuration.size[0] / 2 + 1) * configuration.size[1] * configuration.size[2];

		if (axis_id == 1)
		{

			//don't transpose 0-1
			axis->pushConstants.inputStride[1] = configuration.size[1];
			axis->pushConstants.inputStride[2] = (configuration.size[0] / 2 + 1) * configuration.size[1];
			axis->pushConstants.inputStride[3] = (configuration.size[0] / 2 + 1) * configuration.size[1] * configuration.size[2];
		}
		if (axis_id == 2)
		{

			//don't transpose 0-1, don't transpose 1-2
			axis->pushConstants.inputStride[1] = (configuration.size[0] / 2 + 1) * configuration.size[1];
			axis->pushConstants.inputStride[2] = configuration.size[1];

		}

		axis->pushConstants.outputStride[0] = axis->pushConstants.inputStride[0];
		axis->pushConstants.outputStride[1] = axis->pushConstants.inputStride[1];
		axis->pushConstants.outputStride[2] = axis->pushConstants.inputStride[2];
		axis->pushConstants.outputStride[3] = axis->pushConstants.inputStride[3];

		axis->pushConstants.inputStride[4] = axis->pushConstants.inputStride[3] * configuration.coordinateFeatures;
		axis->pushConstants.outputStride[4] = axis->pushConstants.outputStride[3] * configuration.coordinateFeatures;

		for (uint32_t i = 0; i < 3; ++i) {
			axis->pushConstants.radixStride[i] = configuration.size[axis_id] / pow(2, i + 1);

		}

		axis->pushConstants.inverse = inverse;
		axis->pushConstants.zeropad[0] = configuration.performZeropadding[axis_id];
		axis->pushConstants.zeropad[1] = false;
		axis->pushConstants.ratio[0] = configuration.size[axis_id - 1] / configuration.size[axis_id];
		axis->pushConstants.ratio[1] = configuration.size[axis_id - 1] / configuration.size[axis_id];
		axis->pushConstants.ratioDirection[0] = false;
		axis->pushConstants.ratioDirection[1] = true;
		axis->pushConstants.inputOffset = configuration.size[0] * configuration.size[1] / 2;
		axis->pushConstants.outputOffset = configuration.size[0] * configuration.size[1] / 2;

		VkDescriptorPoolSize descriptorPoolSize = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
		descriptorPoolSize.descriptorCount = 2;
		if ((axis_id == 1) && (configuration.FFTdim == 2) && (configuration.performConvolution))
			descriptorPoolSize.descriptorCount = 3;
		if ((axis_id == 2) && (configuration.FFTdim == 3) && (configuration.performConvolution))
			descriptorPoolSize.descriptorCount = 3;

		VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
		descriptorPoolCreateInfo.poolSizeCount = 1;
		descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize;
		descriptorPoolCreateInfo.maxSets = 1;
		vkCreateDescriptorPool(configuration.device[0], &descriptorPoolCreateInfo, NULL, &axis->descriptorPool);

		const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
		VkDescriptorSetLayoutBinding* descriptorSetLayoutBindings;
		descriptorSetLayoutBindings = (VkDescriptorSetLayoutBinding*)malloc(descriptorPoolSize.descriptorCount * sizeof(VkDescriptorSetLayoutBinding));
		for (uint32_t i = 0; i < descriptorPoolSize.descriptorCount; ++i) {
			descriptorSetLayoutBindings[i].binding = i;
			descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
			descriptorSetLayoutBindings[i].descriptorCount = 1;
			descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		}

		VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
		descriptorSetLayoutCreateInfo.bindingCount = descriptorPoolSize.descriptorCount;
		descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;

		vkCreateDescriptorSetLayout(configuration.device[0], &descriptorSetLayoutCreateInfo, NULL, &axis->descriptorSetLayout);

		VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
		descriptorSetAllocateInfo.descriptorPool = axis->descriptorPool;
		descriptorSetAllocateInfo.descriptorSetCount = 1;
		descriptorSetAllocateInfo.pSetLayouts = &axis->descriptorSetLayout;
		vkAllocateDescriptorSets(configuration.device[0], &descriptorSetAllocateInfo, &axis->descriptorSet);
		for (uint32_t i = 0; i < descriptorPoolSize.descriptorCount; ++i) {
			VkDescriptorBufferInfo descriptorBufferInfo = {};

			if (i == 0) {
				if (configuration.isInputFormatted && (
						((axis_id == 0) && (!inverse)) 
						|| ((axis_id == configuration.FFTdim-1) && (inverse)))
					) {
					descriptorBufferInfo.buffer = configuration.inputBuffer[0];
					descriptorBufferInfo.range = configuration.inputBufferSize[0];
				}
				else {
					if ((configuration.numberKernels > 1) && (inverse)) {
						descriptorBufferInfo.buffer = configuration.outputBuffer[0];
						descriptorBufferInfo.range = configuration.outputBufferSize[0];
					}
					else {
						descriptorBufferInfo.buffer = configuration.buffer[0];
						descriptorBufferInfo.range = configuration.bufferSize[0];
					}
				}
				descriptorBufferInfo.offset = 0;
			}
			if (i == 1) {
				if ((configuration.isOutputFormatted && (
						((axis_id == 0) && (inverse))
						|| ((axis_id == configuration.FFTdim-1) && (!inverse) && (!configuration.performConvolution))
						|| ((axis_id == 0) && (configuration.performConvolution) && (configuration.FFTdim == 1)))
					)||
					((configuration.numberKernels>1)&&(
						(inverse)
						||(axis_id== configuration.FFTdim-1)))
					) {
					descriptorBufferInfo.buffer = configuration.outputBuffer[0];
					descriptorBufferInfo.range = configuration.outputBufferSize[0];
				}
				else {
					descriptorBufferInfo.buffer = configuration.buffer[0];
					descriptorBufferInfo.range = configuration.bufferSize[0];
				}
				descriptorBufferInfo.offset = 0;
			}
			if (i == 2) {
				descriptorBufferInfo.buffer = configuration.kernel[0];
				descriptorBufferInfo.offset = 0;
				descriptorBufferInfo.range = configuration.kernelSize[0];
			}
			VkWriteDescriptorSet writeDescriptorSet = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
			writeDescriptorSet.dstSet = axis->descriptorSet;
			writeDescriptorSet.dstBinding = i;
			writeDescriptorSet.dstArrayElement = 0;
			writeDescriptorSet.descriptorType = descriptorType[i];
			writeDescriptorSet.descriptorCount = 1;
			writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
			vkUpdateDescriptorSets(configuration.device[0], 1, &writeDescriptorSet, 0, NULL);

		}

		{
			VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
			pipelineLayoutCreateInfo.setLayoutCount = 1;
			pipelineLayoutCreateInfo.pSetLayouts = &axis->descriptorSetLayout;
			VkPushConstantRange pushConstantRange = { VK_SHADER_STAGE_COMPUTE_BIT };
			pushConstantRange.offset = 0;
			pushConstantRange.size = sizeof(axis->pushConstants);
			// Push constant ranges are part of the pipeline layout
			pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
			pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
			vkCreatePipelineLayout(configuration.device[0], &pipelineLayoutCreateInfo, NULL, &axis->pipelineLayout);
			if (axis_id == 1) {
				FFTPlan->supportAxes[0].axisBlock[0] = (configuration.size[axis_id] / 8 > 1) ? configuration.size[axis_id] / 8 : 1;
				FFTPlan->supportAxes[0].axisBlock[1] = 1;
				FFTPlan->supportAxes[0].axisBlock[2] = 1;
				FFTPlan->supportAxes[0].axisBlock[3] = configuration.size[1];
			}
			if (axis_id == 2) {
				FFTPlan->supportAxes[1].axisBlock[0] = (configuration.size[1] > FFTPlan->supportAxes[1].groupedBatch) ? FFTPlan->supportAxes[1].groupedBatch : configuration.size[1];
				FFTPlan->supportAxes[1].axisBlock[1] = (configuration.size[2] / 8 > 1) ? configuration.size[2] / 8 : 1;
				FFTPlan->supportAxes[1].axisBlock[2] = 1;
				FFTPlan->supportAxes[1].axisBlock[3] = configuration.size[2];
			}
			VkSpecializationMapEntry specializationMapEntries[4] = { {} };
			specializationMapEntries[0].constantID = 1;
			specializationMapEntries[0].size = sizeof(uint32_t);
			specializationMapEntries[0].offset = 0;
			specializationMapEntries[1].constantID = 2;
			specializationMapEntries[1].size = sizeof(uint32_t);
			specializationMapEntries[1].offset = sizeof(uint32_t);
			specializationMapEntries[2].constantID = 3;
			specializationMapEntries[2].size = sizeof(uint32_t);
			specializationMapEntries[2].offset = 2 * sizeof(uint32_t);
			specializationMapEntries[3].constantID = 4;
			specializationMapEntries[3].size = sizeof(uint32_t);
			specializationMapEntries[3].offset = 3 * sizeof(uint32_t);

			VkSpecializationInfo specializationInfo = {};
			specializationInfo.dataSize = 4 * sizeof(uint32_t);
			specializationInfo.mapEntryCount = 4;
			specializationInfo.pMapEntries = specializationMapEntries;
			specializationInfo.pData = &FFTPlan->supportAxes[axis_id - 1].axisBlock;
			VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
			VkComputePipelineCreateInfo computePipelineCreateInfo = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };


			pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;


			if (axis_id == 1) {

				if ((configuration.FFTdim == 2) && (configuration.performConvolution)) {
					switch (configuration.matrixConvolution) {
					case 1:
						VkFFTInitShader(9, &pipelineShaderStageCreateInfo.module);
						break;
					case 2:
						if (configuration.symmetricKernel)
							VkFFTInitShader(12, &pipelineShaderStageCreateInfo.module);
						else
							VkFFTInitShader(15, &pipelineShaderStageCreateInfo.module);
						break;
					case 3:
						if (configuration.symmetricKernel)
							VkFFTInitShader(18, &pipelineShaderStageCreateInfo.module);
						else
							VkFFTInitShader(21, &pipelineShaderStageCreateInfo.module);
						break;
					}

				}
				else {

					VkFFTInitShader(0, &pipelineShaderStageCreateInfo.module);
				}

			}

			if (axis_id == 2) {
				if ((configuration.FFTdim == 3) && (configuration.performConvolution)) {
					switch (configuration.matrixConvolution) {
					case 1:
						VkFFTInitShader(8, &pipelineShaderStageCreateInfo.module);
						break;
					case 2:
						if (configuration.symmetricKernel)
							VkFFTInitShader(11, &pipelineShaderStageCreateInfo.module);
						else
							VkFFTInitShader(14, &pipelineShaderStageCreateInfo.module);
						break;
					case 3:
						if (configuration.symmetricKernel)
							VkFFTInitShader(17, &pipelineShaderStageCreateInfo.module);
						else
							VkFFTInitShader(20, &pipelineShaderStageCreateInfo.module);
						break;
					}
				}
				else {
					VkFFTInitShader(7, &pipelineShaderStageCreateInfo.module);
				}
			}

			pipelineShaderStageCreateInfo.pName = "main";
			pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
			computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
			computePipelineCreateInfo.layout = axis->pipelineLayout;



			vkCreateComputePipelines(configuration.device[0], VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, &axis->pipeline);
			vkDestroyShaderModule(configuration.device[0], pipelineShaderStageCreateInfo.module, NULL);
		}


	}
	void VkFFTPlanTranspose(VkFFTPlan* FFTPlan, uint32_t axis_id, bool inverse) {
		if (axis_id == 0) {
			if (configuration.performR2C) {
				FFTPlan->transpose[0].pushConstants.ratio = (configuration.size[0] / configuration.size[1] / 2 >= 1) ? configuration.size[0] / configuration.size[1] / 2 : 2 * configuration.size[1] / configuration.size[0];
				FFTPlan->transpose[0].pushConstants.ratioDirection = (configuration.size[0] / configuration.size[1] / 2 >= 1) ? true : false;
			}
			else {
				FFTPlan->transpose[0].pushConstants.ratio = (configuration.size[0] / configuration.size[1] >= 1) ? configuration.size[0] / configuration.size[1] : configuration.size[1] / configuration.size[0];
				FFTPlan->transpose[0].pushConstants.ratioDirection = (configuration.size[0] / configuration.size[1] >= 1) ? true : false;

			}
		}
		if (axis_id == 1) {
			FFTPlan->transpose[1].pushConstants.ratio = (configuration.size[1] / configuration.size[2] >= 1) ? configuration.size[1] / configuration.size[2] : configuration.size[2] / configuration.size[1];
			FFTPlan->transpose[1].pushConstants.ratioDirection = (configuration.size[1] / configuration.size[2] >= 1) ? true : false;
		}

		if (axis_id == 0) {
			if (configuration.performR2C) {
				FFTPlan->transpose[axis_id].pushConstants.inputStride[0] = 1;
				FFTPlan->transpose[axis_id].pushConstants.inputStride[1] = (FFTPlan->transpose[0].pushConstants.ratioDirection) ? configuration.size[0] / 2 : configuration.size[1];
				FFTPlan->transpose[axis_id].pushConstants.inputStride[2] = (configuration.size[0] / 2 + 1) * configuration.size[1];
				FFTPlan->transpose[axis_id].pushConstants.inputStride[3] = (configuration.size[0] / 2 + 1) * configuration.size[1] * configuration.size[2];
			}
			else {
				FFTPlan->transpose[axis_id].pushConstants.inputStride[0] = 1;
				FFTPlan->transpose[axis_id].pushConstants.inputStride[1] = (FFTPlan->transpose[0].pushConstants.ratioDirection) ? configuration.size[0] : configuration.size[1];
				FFTPlan->transpose[axis_id].pushConstants.inputStride[2] = configuration.size[0] * configuration.size[1];
				FFTPlan->transpose[axis_id].pushConstants.inputStride[3] = configuration.size[0] * configuration.size[1] * configuration.size[2];
			}
		}
		if (axis_id == 1) {
			if (configuration.performR2C) {
				FFTPlan->transpose[axis_id].pushConstants.inputStride[0] = 1;
				FFTPlan->transpose[axis_id].pushConstants.inputStride[1] = (FFTPlan->transpose[1].pushConstants.ratioDirection) ? (configuration.size[0] / 2 + 1) * configuration.size[1] : (configuration.size[0] / 2 + 1) * configuration.size[2];
				FFTPlan->transpose[axis_id].pushConstants.inputStride[2] = (FFTPlan->transpose[1].pushConstants.ratioDirection) ? configuration.size[1] : configuration.size[2];
				FFTPlan->transpose[axis_id].pushConstants.inputStride[3] = (configuration.size[0] / 2 + 1) * configuration.size[1] * configuration.size[2];
			}
			else {
				FFTPlan->transpose[axis_id].pushConstants.inputStride[0] = 1;
				FFTPlan->transpose[axis_id].pushConstants.inputStride[1] = (FFTPlan->transpose[1].pushConstants.ratioDirection) ? configuration.size[0] * configuration.size[1] : configuration.size[0] * configuration.size[2];
				FFTPlan->transpose[axis_id].pushConstants.inputStride[2] = (FFTPlan->transpose[1].pushConstants.ratioDirection) ? configuration.size[1] : configuration.size[2];
				FFTPlan->transpose[axis_id].pushConstants.inputStride[3] = configuration.size[0] * configuration.size[1] * configuration.size[2];
			}
		}
		FFTPlan->transpose[axis_id].pushConstants.inputStride[4] = FFTPlan->transpose[axis_id].pushConstants.inputStride[3] * configuration.coordinateFeatures;
		VkDescriptorPoolSize descriptorPoolSize = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
		descriptorPoolSize.descriptorCount = 2;
		//collection->descriptorNum = 3;

		VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
		descriptorPoolCreateInfo.poolSizeCount = 1;
		descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize;
		descriptorPoolCreateInfo.maxSets = 1;
		vkCreateDescriptorPool(configuration.device[0], &descriptorPoolCreateInfo, NULL, &FFTPlan->transpose[axis_id].descriptorPool);

		const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
		VkDescriptorSetLayoutBinding* descriptorSetLayoutBindings;
		descriptorSetLayoutBindings = (VkDescriptorSetLayoutBinding*)malloc(descriptorPoolSize.descriptorCount * sizeof(VkDescriptorSetLayoutBinding));
		for (uint32_t i = 0; i < descriptorPoolSize.descriptorCount; ++i) {
			descriptorSetLayoutBindings[i].binding = i;
			descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
			descriptorSetLayoutBindings[i].descriptorCount = 1;
			descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		}

		VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
		descriptorSetLayoutCreateInfo.bindingCount = descriptorPoolSize.descriptorCount;
		descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;

		vkCreateDescriptorSetLayout(configuration.device[0], &descriptorSetLayoutCreateInfo, NULL, &FFTPlan->transpose[axis_id].descriptorSetLayout);

		VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
		descriptorSetAllocateInfo.descriptorPool = FFTPlan->transpose[axis_id].descriptorPool;
		descriptorSetAllocateInfo.descriptorSetCount = 1;
		descriptorSetAllocateInfo.pSetLayouts = &FFTPlan->transpose[axis_id].descriptorSetLayout;
		vkAllocateDescriptorSets(configuration.device[0], &descriptorSetAllocateInfo, &FFTPlan->transpose[axis_id].descriptorSet);
		for (uint32_t i = 0; i < descriptorPoolSize.descriptorCount; ++i) {


			VkDescriptorBufferInfo descriptorBufferInfo = {};
			if (i == 0) {
				if ((configuration.numberKernels > 1) && (inverse)) {
					descriptorBufferInfo.buffer = configuration.outputBuffer[0];
					descriptorBufferInfo.range = configuration.outputBufferSize[0];
				}
				else {
					descriptorBufferInfo.buffer = configuration.buffer[0];
					descriptorBufferInfo.range = configuration.bufferSize[0];
				}
				descriptorBufferInfo.offset = 0;
			}
			if (i == 1) {
				if ((configuration.numberKernels > 1) && (inverse)) {
					descriptorBufferInfo.buffer = configuration.outputBuffer[0];
					descriptorBufferInfo.range = configuration.outputBufferSize[0];
				}
				else {
					descriptorBufferInfo.buffer = configuration.buffer[0];
					descriptorBufferInfo.range = configuration.bufferSize[0];
				}
				descriptorBufferInfo.offset = 0;
			}

			VkWriteDescriptorSet writeDescriptorSet = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
			writeDescriptorSet.dstSet = FFTPlan->transpose[axis_id].descriptorSet;
			writeDescriptorSet.dstBinding = i;
			writeDescriptorSet.dstArrayElement = 0;
			writeDescriptorSet.descriptorType = descriptorType[i];
			writeDescriptorSet.descriptorCount = 1;
			writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
			vkUpdateDescriptorSets(configuration.device[0], 1, &writeDescriptorSet, 0, NULL);
		}



		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
		pipelineLayoutCreateInfo.setLayoutCount = 1;
		pipelineLayoutCreateInfo.pSetLayouts = &FFTPlan->transpose[axis_id].descriptorSetLayout;
		VkPushConstantRange pushConstantRange = { VK_SHADER_STAGE_COMPUTE_BIT };
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(VkFFTTransposePushConstantsLayout);
		// Push constant ranges are part of the pipeline layout
		pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
		pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
		vkCreatePipelineLayout(configuration.device[0], &pipelineLayoutCreateInfo, NULL, &FFTPlan->transpose[axis_id].pipelineLayout);
		VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };

		VkComputePipelineCreateInfo computePipelineCreateInfo = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
		VkSpecializationMapEntry specializationMapEntries[3] = { {} };
		specializationMapEntries[0].constantID = 1;
		specializationMapEntries[0].size = sizeof(uint32_t);
		specializationMapEntries[0].offset = 0;
		specializationMapEntries[1].constantID = 2;
		specializationMapEntries[1].size = sizeof(uint32_t);
		specializationMapEntries[1].offset = sizeof(uint32_t);
		specializationMapEntries[2].constantID = 3;
		specializationMapEntries[2].size = sizeof(uint32_t);
		specializationMapEntries[2].offset = 2 * sizeof(uint32_t);

		uint32_t max_dim = 1;
		if (FFTPlan->axes[axis_id].axisBlock[1] * configuration.size[axis_id] < pow(2, floor(log2(sqrt(1024 * FFTPlan->transpose[axis_id].pushConstants.ratio)))))
			max_dim = FFTPlan->axes[axis_id].axisBlock[1] * configuration.size[axis_id];
		else
			max_dim = pow(2, floor(log2(sqrt(1024 * FFTPlan->transpose[axis_id].pushConstants.ratio))));
		FFTPlan->transpose[axis_id].transposeBlock[0] = max_dim;
		FFTPlan->transpose[axis_id].transposeBlock[1] = max_dim / FFTPlan->transpose[axis_id].pushConstants.ratio;
		FFTPlan->transpose[axis_id].transposeBlock[2] = 1;
		VkSpecializationInfo specializationInfo = {};
		specializationInfo.dataSize = 3 * sizeof(uint32_t);
		specializationInfo.mapEntryCount = 3;
		specializationInfo.pMapEntries = specializationMapEntries;
		specializationInfo.pData = FFTPlan->transpose[axis_id].transposeBlock;

		pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;

		uint32_t filelength;
		//printf("vkFFT_transpose_inplace\n");
		char filename[256];
		sprintf(filename, "%s%s", configuration.shaderPath, "vkFFT_transpose_inplace.spv");

		uint32_t* code = VkFFTReadShader(filelength, filename);
		VkShaderModuleCreateInfo createInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
		createInfo.pCode = code;
		createInfo.codeSize = filelength;
		vkCreateShaderModule(configuration.device[0], &createInfo, NULL, &pipelineShaderStageCreateInfo.module);
		delete[] code;

		pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
		pipelineShaderStageCreateInfo.pName = "main";
		computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
		computePipelineCreateInfo.layout = FFTPlan->transpose[axis_id].pipelineLayout;


		vkCreateComputePipelines(configuration.device[0], VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, &FFTPlan->transpose[axis_id].pipeline);
		vkDestroyShaderModule(configuration.device[0], pipelineShaderStageCreateInfo.module, NULL);

	}
	void deleteAxis(VkFFTAxis* axis) {
		vkDestroyDescriptorPool(configuration.device[0], axis->descriptorPool, NULL);
		vkDestroyDescriptorSetLayout(configuration.device[0], axis->descriptorSetLayout, NULL);
		vkDestroyPipelineLayout(configuration.device[0], axis->pipelineLayout, NULL);
		vkDestroyPipeline(configuration.device[0], axis->pipeline, NULL);


	}
	void deleteTranspose(VkFFTTranspose* transpose) {
		vkDestroyDescriptorPool(configuration.device[0], transpose->descriptorPool, NULL);
		vkDestroyDescriptorSetLayout(configuration.device[0], transpose->descriptorSetLayout, NULL);
		vkDestroyPipelineLayout(configuration.device[0], transpose->pipelineLayout, NULL);
		vkDestroyPipeline(configuration.device[0], transpose->pipeline, NULL);


	}
	void initializeVulkanFFT(VkFFTConfiguration inputLaunchConfiguration) {
		configuration = inputLaunchConfiguration;
		if (configuration.matrixConvolution > 1) configuration.coordinateFeatures = configuration.matrixConvolution;

		if (configuration.performConvolution) {
			configuration.inverse = false;
			for (uint32_t i = 0; i < configuration.FFTdim; i++) {
				VkFFTPlanAxis(&localFFTPlan_inverse_convolution, i, true);
			}
		}
		for (uint32_t i = 0; i < configuration.FFTdim; i++) {
			VkFFTPlanAxis(&localFFTPlan, i, configuration.inverse);
		}

	}
	void VkFFTAppend(VkCommandBuffer commandBuffer) {
		VkMemoryBarrier memory_barrier = {
				VK_STRUCTURE_TYPE_MEMORY_BARRIER,
				nullptr,
				VK_ACCESS_SHADER_WRITE_BIT,
				VK_ACCESS_SHADER_READ_BIT,
		};
		if (!configuration.inverse) {
			//FFT axis 0
			for (uint32_t j = 0; j < configuration.numberBatches; j++) {
				localFFTPlan.axes[0].pushConstants.batch = j;
				uint32_t maxCoordinate = ((configuration.matrixConvolution) > 1 && (configuration.performConvolution) && (configuration.FFTdim == 1)) ? 1 : configuration.coordinateFeatures;
				for (uint32_t i = 0; i < maxCoordinate; i++) {
					localFFTPlan.axes[0].pushConstants.coordinate = i;
					vkCmdPushConstants(commandBuffer, localFFTPlan.axes[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTPushConstantsLayout), &localFFTPlan.axes[0].pushConstants);
					vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[0].pipeline);
					vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[0].pipelineLayout, 0, 1, &localFFTPlan.axes[0].descriptorSet, 0, NULL);
					if (configuration.performZeropadding[1]) {
						if (configuration.performZeropadding[2]) {

							if (configuration.performR2C == true)
								vkCmdDispatch(commandBuffer, 1, ceil(configuration.size[1] / 2.0 / 2.0 / localFFTPlan.axes[0].axisBlock[1]), ceil(configuration.size[2] / 2.0 / localFFTPlan.axes[0].axisBlock[2]));
							else
								vkCmdDispatch(commandBuffer, 1, ceil(configuration.size[1] / 2.0 / localFFTPlan.axes[0].axisBlock[1]), ceil(configuration.size[2] / 2.0 / localFFTPlan.axes[0].axisBlock[2]));
						}
						else {
							if (configuration.performR2C == true)
								vkCmdDispatch(commandBuffer, 1, ceil(configuration.size[1] / 2.0 / 2.0 / localFFTPlan.axes[0].axisBlock[1]), configuration.size[2] / localFFTPlan.axes[0].axisBlock[2]);
							else
								vkCmdDispatch(commandBuffer, 1, ceil(configuration.size[1] / 2.0 / localFFTPlan.axes[0].axisBlock[1]), configuration.size[2] / localFFTPlan.axes[0].axisBlock[2]);
						}
					}
					else {
						if (configuration.performZeropadding[2]) {
							if (configuration.performR2C == true)
								vkCmdDispatch(commandBuffer, 1, configuration.size[1] / 2 / localFFTPlan.axes[0].axisBlock[1], ceil(configuration.size[2] / 2.0 / localFFTPlan.axes[0].axisBlock[2]));
							else
								vkCmdDispatch(commandBuffer, 1, configuration.size[1] / localFFTPlan.axes[0].axisBlock[1], ceil(configuration.size[2] / 2.0 / localFFTPlan.axes[0].axisBlock[2]));
						}
						else {
							if (configuration.performR2C == true)
								vkCmdDispatch(commandBuffer, 1, configuration.size[1] / 2 / localFFTPlan.axes[0].axisBlock[1], configuration.size[2] / localFFTPlan.axes[0].axisBlock[2]);
							else
								vkCmdDispatch(commandBuffer, 1, configuration.size[1] / localFFTPlan.axes[0].axisBlock[1], configuration.size[2] / localFFTPlan.axes[0].axisBlock[2]);
						}
					}

				}
			}
			vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

			if (configuration.FFTdim > 1) {
				//transpose 0-1, if needed
				if (configuration.performTranspose[0]) {
					for (uint32_t j = 0; j < configuration.numberBatches; j++) {
						localFFTPlan.transpose[0].pushConstants.batch = j;
						for (uint32_t i = 0; i < configuration.coordinateFeatures; i++) {
							localFFTPlan.transpose[0].pushConstants.coordinate = i;
							vkCmdPushConstants(commandBuffer, localFFTPlan.transpose[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTTransposePushConstantsLayout), &localFFTPlan.transpose[0].pushConstants);
							vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.transpose[0].pipeline);
							vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.transpose[0].pipelineLayout, 0, 1, &localFFTPlan.transpose[0].descriptorSet, 0, NULL);
							if (configuration.performR2C == true) {
								if (localFFTPlan.transpose[0].pushConstants.ratioDirection)
									vkCmdDispatch(commandBuffer, configuration.size[0] / 2 / localFFTPlan.transpose[0].transposeBlock[0], configuration.size[1] / localFFTPlan.transpose[0].transposeBlock[1], configuration.size[2] / localFFTPlan.transpose[0].transposeBlock[2]);
								else
									vkCmdDispatch(commandBuffer, configuration.size[1] / localFFTPlan.transpose[0].transposeBlock[0], configuration.size[0] / 2 / localFFTPlan.transpose[0].transposeBlock[1], configuration.size[2] / localFFTPlan.transpose[0].transposeBlock[2]);

							}
							else {
								if (localFFTPlan.transpose[0].pushConstants.ratioDirection)
									vkCmdDispatch(commandBuffer, configuration.size[0] / localFFTPlan.transpose[0].transposeBlock[0], configuration.size[1] / localFFTPlan.transpose[0].transposeBlock[1], configuration.size[2] / localFFTPlan.transpose[0].transposeBlock[2]);
								else
									vkCmdDispatch(commandBuffer, configuration.size[1] / localFFTPlan.transpose[0].transposeBlock[0], configuration.size[0] / localFFTPlan.transpose[0].transposeBlock[1], configuration.size[2] / localFFTPlan.transpose[0].transposeBlock[2]);

							}
						}
					}
					vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

				}

				//FFT axis 1
				if ((configuration.FFTdim == 2) && (configuration.performConvolution)) {
					if (configuration.performTranspose[0]) {
						uint32_t maxCoordinate = (configuration.matrixConvolution > 1 ) ? 1 : configuration.coordinateFeatures;
						for (uint32_t i = 0; i < maxCoordinate; i++) {
							localFFTPlan.axes[1].pushConstants.coordinate = i;
							localFFTPlan.axes[1].pushConstants.batch = configuration.numberKernels;
							vkCmdPushConstants(commandBuffer, localFFTPlan.axes[1].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTPushConstantsLayout), &localFFTPlan.axes[1].pushConstants);
							vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[1].pipeline);
							vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[1].pipelineLayout, 0, 1, &localFFTPlan.axes[1].descriptorSet, 0, NULL);
							if (configuration.performZeropadding[2]) {
								if (configuration.performR2C == true)
									vkCmdDispatch(commandBuffer, 1, configuration.size[0] / 2 / localFFTPlan.axes[1].axisBlock[1] + 1, ceil(configuration.size[2] / 2.0 / localFFTPlan.axes[1].axisBlock[2]));
								else
									vkCmdDispatch(commandBuffer, 1, configuration.size[0] / localFFTPlan.axes[1].axisBlock[1], ceil(configuration.size[2] / 2.0 / localFFTPlan.axes[1].axisBlock[2]));
							}
							else {
								if (configuration.performR2C == true)
									vkCmdDispatch(commandBuffer, 1, configuration.size[0] / 2 / localFFTPlan.axes[1].axisBlock[1] + 1, configuration.size[2] / localFFTPlan.axes[1].axisBlock[2]);
								else
									vkCmdDispatch(commandBuffer, 1, configuration.size[0] / localFFTPlan.axes[1].axisBlock[1], configuration.size[2] / localFFTPlan.axes[1].axisBlock[2]);

							}
						}
						vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

					}
					else {
						if (configuration.performR2C == true) {
							uint32_t maxCoordinate = (configuration.matrixConvolution > 1) ? 1 : configuration.coordinateFeatures;
							for (uint32_t i = 0; i < maxCoordinate; i++) {
								localFFTPlan.supportAxes[0].pushConstants.coordinate = i;
								localFFTPlan.supportAxes[0].pushConstants.batch = configuration.numberKernels;
								vkCmdPushConstants(commandBuffer, localFFTPlan.supportAxes[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTPushConstantsLayout), &localFFTPlan.supportAxes[0].pushConstants);
								vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.supportAxes[0].pipeline);
								vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.supportAxes[0].pipelineLayout, 0, 1, &localFFTPlan.supportAxes[0].descriptorSet, 0, NULL);
								if (configuration.performZeropadding[2]) {
									vkCmdDispatch(commandBuffer, 1, 1, ceil(configuration.size[2] / 2.0));
								}
								else {
									vkCmdDispatch(commandBuffer, 1, 1, configuration.size[2]);
								}
							}
						}
						uint32_t maxCoordinate = (configuration.matrixConvolution > 1) ? 1 : configuration.coordinateFeatures;
						for (uint32_t i = 0; i < maxCoordinate; i++) {
							localFFTPlan.axes[1].pushConstants.coordinate = i;
							localFFTPlan.axes[1].pushConstants.batch = configuration.numberKernels;
							vkCmdPushConstants(commandBuffer, localFFTPlan.axes[1].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTPushConstantsLayout), &localFFTPlan.axes[1].pushConstants);
							vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[1].pipeline);
							vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[1].pipelineLayout, 0, 1, &localFFTPlan.axes[1].descriptorSet, 0, NULL);
							if (configuration.performZeropadding[2]) {
								if (configuration.performR2C == true)
									vkCmdDispatch(commandBuffer, configuration.size[0] / 2 / localFFTPlan.axes[1].axisBlock[0], 1, ceil(configuration.size[2] / 2.0 / localFFTPlan.axes[1].axisBlock[2]));
								else
									vkCmdDispatch(commandBuffer, configuration.size[0] / localFFTPlan.axes[1].axisBlock[0], 1, ceil(configuration.size[2] / 2.0 / localFFTPlan.axes[1].axisBlock[2]));
							}
							else {
								if (configuration.performR2C == true)
									vkCmdDispatch(commandBuffer, configuration.size[0] / 2 / localFFTPlan.axes[1].axisBlock[0], 1, configuration.size[2] / localFFTPlan.axes[1].axisBlock[2]);
								else
									vkCmdDispatch(commandBuffer, configuration.size[0] / localFFTPlan.axes[1].axisBlock[0], 1, configuration.size[2] / localFFTPlan.axes[1].axisBlock[2]);

							}
						}
						vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

					}
				}
				else {
					if (configuration.performTranspose[0]) {
						for (uint32_t j = 0; j < configuration.numberBatches; j++) {
							localFFTPlan.axes[1].pushConstants.batch = j;
							for (uint32_t i = 0; i < configuration.coordinateFeatures; i++) {
								localFFTPlan.axes[1].pushConstants.coordinate = i;
								vkCmdPushConstants(commandBuffer, localFFTPlan.axes[1].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTPushConstantsLayout), &localFFTPlan.axes[1].pushConstants);
								vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[1].pipeline);
								vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[1].pipelineLayout, 0, 1, &localFFTPlan.axes[1].descriptorSet, 0, NULL);
								if (configuration.performZeropadding[2]) {
									if (configuration.performR2C == true)
										vkCmdDispatch(commandBuffer, 1, configuration.size[0] / 2 / localFFTPlan.axes[1].axisBlock[1] + 1, ceil(configuration.size[2] / 2.0 / localFFTPlan.axes[1].axisBlock[2]));
									else
										vkCmdDispatch(commandBuffer, 1, configuration.size[0] / localFFTPlan.axes[1].axisBlock[1], ceil(configuration.size[2] / 2.0 / localFFTPlan.axes[1].axisBlock[2]));
								}
								else {
									if (configuration.performR2C == true)
										vkCmdDispatch(commandBuffer, 1, configuration.size[0] / 2 / localFFTPlan.axes[1].axisBlock[1] + 1, configuration.size[2] / localFFTPlan.axes[1].axisBlock[2]);
									else
										vkCmdDispatch(commandBuffer, 1, configuration.size[0] / localFFTPlan.axes[1].axisBlock[1], configuration.size[2] / localFFTPlan.axes[1].axisBlock[2]);

								}

							}
						}
						vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

					}
					else {

						if (configuration.performR2C == true) {
							for (uint32_t j = 0; j < configuration.numberBatches; j++) {
								localFFTPlan.supportAxes[0].pushConstants.batch = j;
								for (uint32_t i = 0; i < configuration.coordinateFeatures; i++) {

									localFFTPlan.supportAxes[0].pushConstants.coordinate = i;
									vkCmdPushConstants(commandBuffer, localFFTPlan.supportAxes[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTPushConstantsLayout), &localFFTPlan.supportAxes[0].pushConstants);
									vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.supportAxes[0].pipeline);
									vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.supportAxes[0].pipelineLayout, 0, 1, &localFFTPlan.supportAxes[0].descriptorSet, 0, NULL);
									if (configuration.performZeropadding[2]) {
										vkCmdDispatch(commandBuffer, 1, 1, ceil(configuration.size[2] / 2.0));
									}
									else {
										vkCmdDispatch(commandBuffer, 1, 1, configuration.size[2]);
									}
								}
							}
						}
						for (uint32_t j = 0; j < configuration.numberBatches; j++) {
							localFFTPlan.axes[1].pushConstants.batch = j;
							for (uint32_t i = 0; i < configuration.coordinateFeatures; i++) {
								localFFTPlan.axes[1].pushConstants.coordinate = i;
								vkCmdPushConstants(commandBuffer, localFFTPlan.axes[1].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTPushConstantsLayout), &localFFTPlan.axes[1].pushConstants);
								vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[1].pipeline);
								vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[1].pipelineLayout, 0, 1, &localFFTPlan.axes[1].descriptorSet, 0, NULL);
								if (configuration.performZeropadding[2]) {
									if (configuration.performR2C == true)
										vkCmdDispatch(commandBuffer, configuration.size[0] / 2 / localFFTPlan.axes[1].axisBlock[0], 1, ceil(configuration.size[2] / 2.0 / localFFTPlan.axes[1].axisBlock[2]));
									else
										vkCmdDispatch(commandBuffer, configuration.size[0] / localFFTPlan.axes[1].axisBlock[0], 1, ceil(configuration.size[2] / 2.0 / localFFTPlan.axes[1].axisBlock[2]));
								}
								else {
									if (configuration.performR2C == true)
										vkCmdDispatch(commandBuffer, configuration.size[0] / 2 / localFFTPlan.axes[1].axisBlock[0], 1, configuration.size[2] / localFFTPlan.axes[1].axisBlock[2]);
									else
										vkCmdDispatch(commandBuffer, configuration.size[0] / localFFTPlan.axes[1].axisBlock[0], 1, configuration.size[2] / localFFTPlan.axes[1].axisBlock[2]);

								}
							}
						}
						vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

					}
				}
			}
			//FFT axis 2
			if (configuration.FFTdim > 2) {
				//transpose 1-2, after 0-1
				if (configuration.performTranspose[1]) {
					for (uint32_t j = 0; j < configuration.numberBatches; j++) {
						localFFTPlan.transpose[1].pushConstants.batch = j;
						for (uint32_t i = 0; i < configuration.coordinateFeatures; i++) {
							localFFTPlan.transpose[1].pushConstants.coordinate = i;
							vkCmdPushConstants(commandBuffer, localFFTPlan.transpose[1].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTTransposePushConstantsLayout), &localFFTPlan.transpose[1].pushConstants);
							vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.transpose[1].pipeline);
							vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.transpose[1].pipelineLayout, 0, 1, &localFFTPlan.transpose[1].descriptorSet, 0, NULL);
							if (configuration.performR2C == true) {
								if (localFFTPlan.transpose[1].pushConstants.ratioDirection)
									vkCmdDispatch(commandBuffer, configuration.size[1] / localFFTPlan.transpose[1].transposeBlock[0], configuration.size[2] / localFFTPlan.transpose[1].transposeBlock[1], (configuration.size[0] / 2 + 1) / localFFTPlan.transpose[1].transposeBlock[2]);
								else
									vkCmdDispatch(commandBuffer, configuration.size[2] / localFFTPlan.transpose[1].transposeBlock[0], configuration.size[1] / localFFTPlan.transpose[1].transposeBlock[1], (configuration.size[0] / 2 + 1) / localFFTPlan.transpose[1].transposeBlock[2]);

							}
							else {
								if (localFFTPlan.transpose[1].pushConstants.ratioDirection)
									vkCmdDispatch(commandBuffer, configuration.size[1] / localFFTPlan.transpose[1].transposeBlock[0], configuration.size[2] / localFFTPlan.transpose[1].transposeBlock[1], configuration.size[0] / localFFTPlan.transpose[1].transposeBlock[2]);
								else
									vkCmdDispatch(commandBuffer, configuration.size[2] / localFFTPlan.transpose[1].transposeBlock[0], configuration.size[1] / localFFTPlan.transpose[1].transposeBlock[1], configuration.size[0] / localFFTPlan.transpose[1].transposeBlock[2]);

							}
						}
					}
					vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

				}

				if ((configuration.FFTdim == 3) && (configuration.performConvolution)) {
					//transposed 1-2, transposed 0-1
					if (configuration.performTranspose[1]) {
						uint32_t maxCoordinate = (configuration.matrixConvolution > 1) ? 1 : configuration.coordinateFeatures;
						for (uint32_t i = 0; i < maxCoordinate; i++) {
							localFFTPlan.axes[2].pushConstants.coordinate = i;
							localFFTPlan.axes[2].pushConstants.batch = configuration.numberKernels;
							vkCmdPushConstants(commandBuffer, localFFTPlan.axes[2].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTPushConstantsLayout), &localFFTPlan.axes[2].pushConstants);
							vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[2].pipeline);
							vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[2].pipelineLayout, 0, 1, &localFFTPlan.axes[2].descriptorSet, 0, NULL);
							if (configuration.performR2C == true)
								vkCmdDispatch(commandBuffer, 1, configuration.size[1] / localFFTPlan.axes[2].axisBlock[2], configuration.size[0] / 2 + 1);
							else
								vkCmdDispatch(commandBuffer, 1, configuration.size[1] / localFFTPlan.axes[2].axisBlock[2], configuration.size[0]);
						}
						vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

					}
					else {
						if (configuration.performTranspose[0]) {
							//transposed 0-1, didn't transpose 1-2
							uint32_t maxCoordinate = (configuration.matrixConvolution > 1) ? 1 : configuration.coordinateFeatures;
							for (uint32_t i = 0; i < maxCoordinate; i++) {
								localFFTPlan.axes[2].pushConstants.coordinate = i;
								localFFTPlan.axes[2].pushConstants.batch = configuration.numberKernels;
								vkCmdPushConstants(commandBuffer, localFFTPlan.axes[2].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTPushConstantsLayout), &localFFTPlan.axes[2].pushConstants);
								vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[2].pipeline);
								vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[2].pipelineLayout, 0, 1, &localFFTPlan.axes[2].descriptorSet, 0, NULL);
								if (configuration.performR2C == true)
									vkCmdDispatch(commandBuffer, configuration.size[1] / localFFTPlan.axes[2].axisBlock[0], 1, configuration.size[0] / 2 + 1);
								else
									vkCmdDispatch(commandBuffer, configuration.size[1] / localFFTPlan.axes[2].axisBlock[0], 1, configuration.size[0]);
							}
							vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);
						}
						else {
							//didn't transpose 0-1, didn't transpose 1-2
							if (configuration.performR2C == true) {
								uint32_t maxCoordinate = (configuration.matrixConvolution > 1) ? 1 : configuration.coordinateFeatures;
								for (uint32_t i = 0; i < maxCoordinate; i++) {
									localFFTPlan.supportAxes[1].pushConstants.coordinate = i;
									localFFTPlan.supportAxes[1].pushConstants.batch = configuration.numberKernels;
									vkCmdPushConstants(commandBuffer, localFFTPlan.supportAxes[1].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTPushConstantsLayout), &localFFTPlan.supportAxes[1].pushConstants);
									vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.supportAxes[1].pipeline);
									vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.supportAxes[1].pipelineLayout, 0, 1, &localFFTPlan.supportAxes[1].descriptorSet, 0, NULL);
									vkCmdDispatch(commandBuffer, configuration.size[1] / localFFTPlan.supportAxes[1].axisBlock[0], 1, 1);

								}
							}
							uint32_t maxCoordinate = (configuration.matrixConvolution > 1) ? 1 : configuration.coordinateFeatures;
							for (uint32_t i = 0; i < maxCoordinate; i++) {
								localFFTPlan.axes[2].pushConstants.coordinate = i;
								localFFTPlan.axes[2].pushConstants.batch = configuration.numberKernels;
								vkCmdPushConstants(commandBuffer, localFFTPlan.axes[2].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTPushConstantsLayout), &localFFTPlan.axes[2].pushConstants);
								vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[2].pipeline);
								vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[2].pipelineLayout, 0, 1, &localFFTPlan.axes[2].descriptorSet, 0, NULL);
								if (configuration.performR2C == true)
									vkCmdDispatch(commandBuffer, configuration.size[0] / 2 / localFFTPlan.axes[2].axisBlock[0], 1, configuration.size[1]);
								else
									vkCmdDispatch(commandBuffer, configuration.size[0] / localFFTPlan.axes[2].axisBlock[0], 1, configuration.size[1]);
							}
							vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);
						}
					}
				}
				else {
					//transposed 1-2, transposed 0-1
					if (configuration.performTranspose[1]) {
						for (uint32_t j = 0; j < configuration.numberBatches; j++) {
							localFFTPlan.axes[2].pushConstants.batch = j;
							for (uint32_t i = 0; i < configuration.coordinateFeatures; i++) {
								localFFTPlan.axes[2].pushConstants.coordinate = i;
								vkCmdPushConstants(commandBuffer, localFFTPlan.axes[2].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTPushConstantsLayout), &localFFTPlan.axes[2].pushConstants);
								vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[2].pipeline);
								vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[2].pipelineLayout, 0, 1, &localFFTPlan.axes[2].descriptorSet, 0, NULL);
								if (configuration.performR2C == true)
									vkCmdDispatch(commandBuffer, 1, configuration.size[1] / localFFTPlan.axes[2].axisBlock[2], configuration.size[0] / 2 + 1);
								else
									vkCmdDispatch(commandBuffer, 1, configuration.size[1] / localFFTPlan.axes[2].axisBlock[2], configuration.size[0]);
							}
						}
						vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

					}
					else {
						if (configuration.performTranspose[0]) {
							//transposed 0-1, didn't transpose 1-2
							for (uint32_t j = 0; j < configuration.numberBatches; j++) {
								localFFTPlan.axes[2].pushConstants.batch = j;
								for (uint32_t i = 0; i < configuration.coordinateFeatures; i++) {
									localFFTPlan.axes[2].pushConstants.coordinate = i;
									vkCmdPushConstants(commandBuffer, localFFTPlan.axes[2].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTPushConstantsLayout), &localFFTPlan.axes[2].pushConstants);
									vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[2].pipeline);
									vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[2].pipelineLayout, 0, 1, &localFFTPlan.axes[2].descriptorSet, 0, NULL);
									if (configuration.performR2C == true)
										vkCmdDispatch(commandBuffer, configuration.size[1] / localFFTPlan.axes[2].axisBlock[0], 1, configuration.size[0] / 2 + 1);
									else
										vkCmdDispatch(commandBuffer, configuration.size[1] / localFFTPlan.axes[2].axisBlock[0], 1, configuration.size[0]);
								}
							}
							vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

						}
						else {
							//didn't transpose 0-1, didn't transpose 1-2
							if (configuration.performR2C == true) {
								for (uint32_t j = 0; j < configuration.numberBatches; j++) {
									localFFTPlan.supportAxes[1].pushConstants.batch = j;
									for (uint32_t i = 0; i < configuration.coordinateFeatures; i++) {
										localFFTPlan.supportAxes[1].pushConstants.coordinate = i;
										vkCmdPushConstants(commandBuffer, localFFTPlan.supportAxes[1].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTPushConstantsLayout), &localFFTPlan.supportAxes[1].pushConstants);
										vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.supportAxes[1].pipeline);
										vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.supportAxes[1].pipelineLayout, 0, 1, &localFFTPlan.supportAxes[1].descriptorSet, 0, NULL);
										vkCmdDispatch(commandBuffer, configuration.size[1] / localFFTPlan.supportAxes[1].axisBlock[0], 1, 1);
									}
								}
							}
							for (uint32_t j = 0; j < configuration.numberBatches; j++) {
								localFFTPlan.axes[2].pushConstants.batch = j;
								for (uint32_t i = 0; i < configuration.coordinateFeatures; i++) {
									localFFTPlan.axes[2].pushConstants.coordinate = i;
									vkCmdPushConstants(commandBuffer, localFFTPlan.axes[2].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTPushConstantsLayout), &localFFTPlan.axes[2].pushConstants);
									vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[2].pipeline);
									vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[2].pipelineLayout, 0, 1, &localFFTPlan.axes[2].descriptorSet, 0, NULL);
									if (configuration.performR2C == true)
										vkCmdDispatch(commandBuffer, configuration.size[0] / 2 / localFFTPlan.axes[2].axisBlock[0], 1, configuration.size[1]);
									else
										vkCmdDispatch(commandBuffer, configuration.size[0] / localFFTPlan.axes[2].axisBlock[0], 1, configuration.size[1]);
								}
							}
							vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

						}
					}
				}

			}
		}
		if (configuration.performConvolution) {
			if (configuration.FFTdim > 2) {
				//transpose 1-2, after 0-1
				if (configuration.performTranspose[1]) {
					for (uint32_t j = 0; j < configuration.numberKernels; j++) {
						localFFTPlan_inverse_convolution.transpose[1].pushConstants.batch = j;
						for (uint32_t i = 0; i < configuration.coordinateFeatures; i++) {
							localFFTPlan_inverse_convolution.transpose[1].pushConstants.coordinate = i;
							vkCmdPushConstants(commandBuffer, localFFTPlan_inverse_convolution.transpose[1].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTTransposePushConstantsLayout), &localFFTPlan_inverse_convolution.transpose[1].pushConstants);
							vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan_inverse_convolution.transpose[1].pipeline);
							vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan_inverse_convolution.transpose[1].pipelineLayout, 0, 1, &localFFTPlan_inverse_convolution.transpose[1].descriptorSet, 0, NULL);
							if (configuration.performR2C == true) {
								if (localFFTPlan_inverse_convolution.transpose[1].pushConstants.ratioDirection)
									vkCmdDispatch(commandBuffer, configuration.size[1] / localFFTPlan_inverse_convolution.transpose[1].transposeBlock[0], configuration.size[2] / localFFTPlan_inverse_convolution.transpose[1].transposeBlock[1], (configuration.size[0] / 2 + 1) / localFFTPlan_inverse_convolution.transpose[1].transposeBlock[2]);
								else
									vkCmdDispatch(commandBuffer, configuration.size[2] / localFFTPlan_inverse_convolution.transpose[1].transposeBlock[0], configuration.size[1] / localFFTPlan_inverse_convolution.transpose[1].transposeBlock[1], (configuration.size[0] / 2 + 1) / localFFTPlan_inverse_convolution.transpose[1].transposeBlock[2]);

							}
							else {
								if (localFFTPlan_inverse_convolution.transpose[1].pushConstants.ratioDirection)
									vkCmdDispatch(commandBuffer, configuration.size[1] / localFFTPlan_inverse_convolution.transpose[1].transposeBlock[0], configuration.size[2] / localFFTPlan_inverse_convolution.transpose[1].transposeBlock[1], configuration.size[0] / localFFTPlan_inverse_convolution.transpose[1].transposeBlock[2]);
								else
									vkCmdDispatch(commandBuffer, configuration.size[2] / localFFTPlan_inverse_convolution.transpose[1].transposeBlock[0], configuration.size[1] / localFFTPlan_inverse_convolution.transpose[1].transposeBlock[1], configuration.size[0] / localFFTPlan_inverse_convolution.transpose[1].transposeBlock[2]);

							}
						}
					}
					vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

				}

				if (configuration.performTranspose[0]) {
					for (uint32_t j = 0; j < configuration.numberKernels; j++) {
						localFFTPlan_inverse_convolution.axes[1].pushConstants.batch = j;
						for (uint32_t i = 0; i < configuration.coordinateFeatures; i++) {
							localFFTPlan_inverse_convolution.axes[1].pushConstants.coordinate = i;
							vkCmdPushConstants(commandBuffer, localFFTPlan_inverse_convolution.axes[1].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTPushConstantsLayout), &localFFTPlan_inverse_convolution.axes[1].pushConstants);
							vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan_inverse_convolution.axes[1].pipeline);
							vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan_inverse_convolution.axes[1].pipelineLayout, 0, 1, &localFFTPlan_inverse_convolution.axes[1].descriptorSet, 0, NULL);
							if (configuration.performZeropadding[2]) {
								if (configuration.performR2C == true)
									vkCmdDispatch(commandBuffer, 1, configuration.size[0] / 2 / localFFTPlan_inverse_convolution.axes[1].axisBlock[1] + 1, ceil(configuration.size[2] / 2.0 / localFFTPlan_inverse_convolution.axes[1].axisBlock[2]));
								else
									vkCmdDispatch(commandBuffer, 1, configuration.size[0] / localFFTPlan_inverse_convolution.axes[1].axisBlock[1], ceil(configuration.size[2] / 2.0 / localFFTPlan_inverse_convolution.axes[1].axisBlock[2]));
							}
							else {
								if (configuration.performR2C == true)
									vkCmdDispatch(commandBuffer, 1, configuration.size[0] / 2 / localFFTPlan_inverse_convolution.axes[1].axisBlock[1] + 1, configuration.size[2] / localFFTPlan_inverse_convolution.axes[1].axisBlock[2]);
								else
									vkCmdDispatch(commandBuffer, 1, configuration.size[0] / localFFTPlan_inverse_convolution.axes[1].axisBlock[1], configuration.size[2] / localFFTPlan_inverse_convolution.axes[1].axisBlock[2]);

							}
						}
					}
					vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);
				}
				else {

					if (configuration.performR2C == true) {
						for (uint32_t j = 0; j < configuration.numberKernels; j++) {
							localFFTPlan_inverse_convolution.supportAxes[0].pushConstants.batch = j;
							for (uint32_t i = 0; i < configuration.coordinateFeatures; i++) {

								localFFTPlan_inverse_convolution.supportAxes[0].pushConstants.coordinate = i;
								vkCmdPushConstants(commandBuffer, localFFTPlan_inverse_convolution.supportAxes[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTPushConstantsLayout), &localFFTPlan_inverse_convolution.supportAxes[0].pushConstants);
								vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan_inverse_convolution.supportAxes[0].pipeline);
								vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan_inverse_convolution.supportAxes[0].pipelineLayout, 0, 1, &localFFTPlan_inverse_convolution.supportAxes[0].descriptorSet, 0, NULL);
								if (configuration.performZeropadding[2]) {
									vkCmdDispatch(commandBuffer, 1, 1, ceil(configuration.size[2] / 2.0));
								}
								else {
									vkCmdDispatch(commandBuffer, 1, 1, configuration.size[2]);
								}
							}
						}
					}
					for (uint32_t j = 0; j < configuration.numberKernels; j++) {
						localFFTPlan_inverse_convolution.axes[1].pushConstants.batch = j;
						for (uint32_t i = 0; i < configuration.coordinateFeatures; i++) {
							localFFTPlan_inverse_convolution.axes[1].pushConstants.coordinate = i;
							vkCmdPushConstants(commandBuffer, localFFTPlan_inverse_convolution.axes[1].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTPushConstantsLayout), &localFFTPlan_inverse_convolution.axes[1].pushConstants);
							vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan_inverse_convolution.axes[1].pipeline);
							vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan_inverse_convolution.axes[1].pipelineLayout, 0, 1, &localFFTPlan_inverse_convolution.axes[1].descriptorSet, 0, NULL);
							if (configuration.performZeropadding[2]) {
								if (configuration.performR2C == true)
									vkCmdDispatch(commandBuffer, configuration.size[0] / 2 / localFFTPlan_inverse_convolution.axes[1].axisBlock[0], 1, ceil(configuration.size[2] / 2.0 / localFFTPlan_inverse_convolution.axes[1].axisBlock[2]));
								else
									vkCmdDispatch(commandBuffer, configuration.size[0] / localFFTPlan_inverse_convolution.axes[1].axisBlock[0], 1, ceil(configuration.size[2] / 2.0 / localFFTPlan_inverse_convolution.axes[1].axisBlock[2]));
							}
							else {
								if (configuration.performR2C == true)
									vkCmdDispatch(commandBuffer, configuration.size[0] / 2 / localFFTPlan_inverse_convolution.axes[1].axisBlock[0], 1, configuration.size[2] / localFFTPlan_inverse_convolution.axes[1].axisBlock[2]);
								else
									vkCmdDispatch(commandBuffer, configuration.size[0] / localFFTPlan_inverse_convolution.axes[1].axisBlock[0], 1, configuration.size[2] / localFFTPlan_inverse_convolution.axes[1].axisBlock[2]);

							}
						}
					}
					vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

				}




			}
			if (configuration.FFTdim > 1) {
				// transpose 0 - 1, if needed
				if (configuration.performTranspose[0]) {
					for (uint32_t j = 0; j < configuration.numberKernels; j++) {
						localFFTPlan_inverse_convolution.transpose[0].pushConstants.batch = j;
						for (uint32_t i = 0; i < configuration.coordinateFeatures; i++) {
							localFFTPlan_inverse_convolution.transpose[0].pushConstants.coordinate = i;
							vkCmdPushConstants(commandBuffer, localFFTPlan_inverse_convolution.transpose[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTTransposePushConstantsLayout), &localFFTPlan_inverse_convolution.transpose[0].pushConstants);
							vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan_inverse_convolution.transpose[0].pipeline);
							vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan_inverse_convolution.transpose[0].pipelineLayout, 0, 1, &localFFTPlan_inverse_convolution.transpose[0].descriptorSet, 0, NULL);
							if (configuration.performR2C == true) {
								if (localFFTPlan_inverse_convolution.transpose[0].pushConstants.ratioDirection)
									vkCmdDispatch(commandBuffer, configuration.size[0] / 2 / localFFTPlan_inverse_convolution.transpose[0].transposeBlock[0], configuration.size[1] / localFFTPlan_inverse_convolution.transpose[0].transposeBlock[1], configuration.size[2] / localFFTPlan_inverse_convolution.transpose[0].transposeBlock[2]);
								else
									vkCmdDispatch(commandBuffer, configuration.size[1] / localFFTPlan_inverse_convolution.transpose[0].transposeBlock[0], configuration.size[0] / 2 / localFFTPlan_inverse_convolution.transpose[0].transposeBlock[1], configuration.size[2] / localFFTPlan_inverse_convolution.transpose[0].transposeBlock[2]);

							}
							else {
								if (localFFTPlan_inverse_convolution.transpose[0].pushConstants.ratioDirection)
									vkCmdDispatch(commandBuffer, configuration.size[0] / localFFTPlan_inverse_convolution.transpose[0].transposeBlock[0], configuration.size[1] / localFFTPlan_inverse_convolution.transpose[0].transposeBlock[1], configuration.size[2] / localFFTPlan_inverse_convolution.transpose[0].transposeBlock[2]);
								else
									vkCmdDispatch(commandBuffer, configuration.size[1] / localFFTPlan_inverse_convolution.transpose[0].transposeBlock[0], configuration.size[0] / localFFTPlan_inverse_convolution.transpose[0].transposeBlock[1], configuration.size[2] / localFFTPlan_inverse_convolution.transpose[0].transposeBlock[2]);

							}
						}
					}
					vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

				}
				for (uint32_t j = 0; j < configuration.numberKernels; j++) {
					localFFTPlan_inverse_convolution.axes[0].pushConstants.batch = j;
					for (uint32_t i = 0; i < configuration.coordinateFeatures; i++) {
						localFFTPlan_inverse_convolution.axes[0].pushConstants.coordinate = i;
						vkCmdPushConstants(commandBuffer, localFFTPlan_inverse_convolution.axes[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTPushConstantsLayout), &localFFTPlan_inverse_convolution.axes[0].pushConstants);
						vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan_inverse_convolution.axes[0].pipeline);
						vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan_inverse_convolution.axes[0].pipelineLayout, 0, 1, &localFFTPlan_inverse_convolution.axes[0].descriptorSet, 0, NULL);
						if (configuration.performZeropadding[1]) {
							if (configuration.performZeropadding[2]) {
								if (configuration.performR2C == true)
									vkCmdDispatch(commandBuffer, 1, ceil(configuration.size[1] / 2.0 / 2.0 / localFFTPlan_inverse_convolution.axes[0].axisBlock[1]), ceil(configuration.size[2] / 2.0 / localFFTPlan_inverse_convolution.axes[0].axisBlock[2]));
								else
									vkCmdDispatch(commandBuffer, 1, ceil(configuration.size[1] / 2.0 / localFFTPlan_inverse_convolution.axes[0].axisBlock[1]), ceil(configuration.size[2] / 2.0 / localFFTPlan_inverse_convolution.axes[0].axisBlock[2]));
							}
							else {
								if (configuration.performR2C == true)
									vkCmdDispatch(commandBuffer, 1, ceil(configuration.size[1] / 2.0 / 2.0 / localFFTPlan_inverse_convolution.axes[0].axisBlock[1]), configuration.size[2] / localFFTPlan_inverse_convolution.axes[0].axisBlock[2]);
								else
									vkCmdDispatch(commandBuffer, 1, ceil(configuration.size[1] / 2.0 / localFFTPlan_inverse_convolution.axes[0].axisBlock[1]), configuration.size[2] / localFFTPlan_inverse_convolution.axes[0].axisBlock[2]);

							}
						}
						else {
							if (configuration.performZeropadding[2]) {
								if (configuration.performR2C == true)
									vkCmdDispatch(commandBuffer, 1, configuration.size[1] / 2 / localFFTPlan_inverse_convolution.axes[0].axisBlock[1], ceil(configuration.size[2] / 2.0 / localFFTPlan_inverse_convolution.axes[0].axisBlock[2]));
								else
									vkCmdDispatch(commandBuffer, 1, configuration.size[1] / localFFTPlan_inverse_convolution.axes[0].axisBlock[1], ceil(configuration.size[2] / 2.0 / localFFTPlan_inverse_convolution.axes[0].axisBlock[2]));
							}
							else {
								if (configuration.performR2C == true)
									vkCmdDispatch(commandBuffer, 1, configuration.size[1] / 2 / localFFTPlan_inverse_convolution.axes[0].axisBlock[1], configuration.size[2] / localFFTPlan_inverse_convolution.axes[0].axisBlock[2]);
								else
									vkCmdDispatch(commandBuffer, 1, configuration.size[1] / localFFTPlan_inverse_convolution.axes[0].axisBlock[1], configuration.size[2] / localFFTPlan_inverse_convolution.axes[0].axisBlock[2]);

							}
						}

					}
				}
				vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);


			}
		}

		if (configuration.inverse) {
			//we start from axis 2 and go back to axis 0
			//FFT axis 2
			if (configuration.FFTdim > 2) {
				//transposed 1-2, transposed 0-1
				if (configuration.performTranspose[1]) {
					for (uint32_t j = 0; j < configuration.numberBatches; j++) {
						localFFTPlan.axes[2].pushConstants.batch = j;
						for (uint32_t i = 0; i < configuration.coordinateFeatures; i++) {
							localFFTPlan.axes[2].pushConstants.coordinate = i;
							vkCmdPushConstants(commandBuffer, localFFTPlan.axes[2].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTPushConstantsLayout), &localFFTPlan.axes[2].pushConstants);
							vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[2].pipeline);
							vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[2].pipelineLayout, 0, 1, &localFFTPlan.axes[2].descriptorSet, 0, NULL);
							if (configuration.performR2C == true)
								vkCmdDispatch(commandBuffer, 1, configuration.size[1] / localFFTPlan.axes[2].axisBlock[2], configuration.size[0] / 2 + 1);
							else
								vkCmdDispatch(commandBuffer, 1, configuration.size[1] / localFFTPlan.axes[2].axisBlock[2], configuration.size[0]);
						}
					}
					vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

				}
				else {
					if (configuration.performTranspose[0]) {
						//transposed 0-1, didn't transpose 1-2
						for (uint32_t j = 0; j < configuration.numberBatches; j++) {
							localFFTPlan.axes[2].pushConstants.batch = j;
							for (uint32_t i = 0; i < configuration.coordinateFeatures; i++) {
								localFFTPlan.axes[2].pushConstants.coordinate = i;
								vkCmdPushConstants(commandBuffer, localFFTPlan.axes[2].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTPushConstantsLayout), &localFFTPlan.axes[2].pushConstants);
								vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[2].pipeline);
								vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[2].pipelineLayout, 0, 1, &localFFTPlan.axes[2].descriptorSet, 0, NULL);
								if (configuration.performR2C == true)
									vkCmdDispatch(commandBuffer, configuration.size[1] / localFFTPlan.axes[2].axisBlock[0], 1, configuration.size[0] / 2 + 1);
								else
									vkCmdDispatch(commandBuffer, configuration.size[1] / localFFTPlan.axes[2].axisBlock[0], 1, configuration.size[0]);
							}
						}
						vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

					}
					else {
						//didn't transpose 0-1, didn't transpose 1-2
						if (configuration.performR2C == true) {
							for (uint32_t j = 0; j < configuration.numberBatches; j++) {
								localFFTPlan.supportAxes[1].pushConstants.batch = j;
								for (uint32_t i = 0; i < configuration.coordinateFeatures; i++) {
									localFFTPlan.supportAxes[1].pushConstants.coordinate = i;
									vkCmdPushConstants(commandBuffer, localFFTPlan.supportAxes[1].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTPushConstantsLayout), &localFFTPlan.supportAxes[1].pushConstants);
									vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.supportAxes[1].pipeline);
									vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.supportAxes[1].pipelineLayout, 0, 1, &localFFTPlan.supportAxes[1].descriptorSet, 0, NULL);
									vkCmdDispatch(commandBuffer, configuration.size[1] / localFFTPlan.supportAxes[1].axisBlock[0], 1, 1);
								}
							}
						}
						for (uint32_t j = 0; j < configuration.numberBatches; j++) {
							localFFTPlan.axes[2].pushConstants.batch = j;
							for (uint32_t i = 0; i < configuration.coordinateFeatures; i++) {
								localFFTPlan.axes[2].pushConstants.coordinate = i;
								vkCmdPushConstants(commandBuffer, localFFTPlan.axes[2].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTPushConstantsLayout), &localFFTPlan.axes[2].pushConstants);
								vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[2].pipeline);
								vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[2].pipelineLayout, 0, 1, &localFFTPlan.axes[2].descriptorSet, 0, NULL);
								if (configuration.performR2C == true)
									vkCmdDispatch(commandBuffer, configuration.size[0] / 2 / localFFTPlan.axes[2].axisBlock[0], 1, configuration.size[1]);
								else
									vkCmdDispatch(commandBuffer, configuration.size[0] / localFFTPlan.axes[2].axisBlock[0], 1, configuration.size[1]);
							}
						}
						vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

					}
				}
				//transpose 1-2, after 0-1
				if (configuration.performTranspose[1]) {
					for (uint32_t j = 0; j < configuration.numberBatches; j++) {
						localFFTPlan.transpose[1].pushConstants.batch = j;
						for (uint32_t i = 0; i < configuration.coordinateFeatures; i++) {
							localFFTPlan.transpose[1].pushConstants.coordinate = i;
							vkCmdPushConstants(commandBuffer, localFFTPlan.transpose[1].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTTransposePushConstantsLayout), &localFFTPlan.transpose[1].pushConstants);
							vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.transpose[1].pipeline);
							vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.transpose[1].pipelineLayout, 0, 1, &localFFTPlan.transpose[1].descriptorSet, 0, NULL);
							if (configuration.performR2C == true) {
								if (localFFTPlan.transpose[1].pushConstants.ratioDirection)
									vkCmdDispatch(commandBuffer, configuration.size[1] / localFFTPlan.transpose[1].transposeBlock[0], configuration.size[2] / localFFTPlan.transpose[1].transposeBlock[1], (configuration.size[0] / 2 + 1) / localFFTPlan.transpose[1].transposeBlock[2]);
								else
									vkCmdDispatch(commandBuffer, configuration.size[2] / localFFTPlan.transpose[1].transposeBlock[0], configuration.size[1] / localFFTPlan.transpose[1].transposeBlock[1], (configuration.size[0] / 2 + 1) / localFFTPlan.transpose[1].transposeBlock[2]);

							}
							else {
								if (localFFTPlan.transpose[1].pushConstants.ratioDirection)
									vkCmdDispatch(commandBuffer, configuration.size[1] / localFFTPlan.transpose[1].transposeBlock[0], configuration.size[2] / localFFTPlan.transpose[1].transposeBlock[1], configuration.size[0] / localFFTPlan.transpose[1].transposeBlock[2]);
								else
									vkCmdDispatch(commandBuffer, configuration.size[2] / localFFTPlan.transpose[1].transposeBlock[0], configuration.size[1] / localFFTPlan.transpose[1].transposeBlock[1], configuration.size[0] / localFFTPlan.transpose[1].transposeBlock[2]);

							}
						}
					}
					vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

				}

			}
			if (configuration.FFTdim > 1) {

				//FFT axis 1
				if (configuration.performTranspose[0]) {
					for (uint32_t j = 0; j < configuration.numberBatches; j++) {
						localFFTPlan.axes[1].pushConstants.batch = j;
						for (uint32_t i = 0; i < configuration.coordinateFeatures; i++) {
							localFFTPlan.axes[1].pushConstants.coordinate = i;
							vkCmdPushConstants(commandBuffer, localFFTPlan.axes[1].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTPushConstantsLayout), &localFFTPlan.axes[1].pushConstants);
							vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[1].pipeline);
							vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[1].pipelineLayout, 0, 1, &localFFTPlan.axes[1].descriptorSet, 0, NULL);
							if (configuration.performR2C == true)
								vkCmdDispatch(commandBuffer, 1, configuration.size[0] / 2 / localFFTPlan.axes[1].axisBlock[1] + 1, configuration.size[2] / localFFTPlan.axes[1].axisBlock[2]);
							else
								vkCmdDispatch(commandBuffer, 1, configuration.size[0] / localFFTPlan.axes[1].axisBlock[1], configuration.size[2] / localFFTPlan.axes[1].axisBlock[2]);

						}
					}
					vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

				}
				else {

					if (configuration.performR2C == true) {
						for (uint32_t j = 0; j < configuration.numberBatches; j++) {
							localFFTPlan.supportAxes[0].pushConstants.batch = j;
							for (uint32_t i = 0; i < configuration.coordinateFeatures; i++) {

								localFFTPlan.supportAxes[0].pushConstants.coordinate = i;
								vkCmdPushConstants(commandBuffer, localFFTPlan.supportAxes[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTPushConstantsLayout), &localFFTPlan.supportAxes[0].pushConstants);
								vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.supportAxes[0].pipeline);
								vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.supportAxes[0].pipelineLayout, 0, 1, &localFFTPlan.supportAxes[0].descriptorSet, 0, NULL);
								if (configuration.performZeropadding[2]) {
									vkCmdDispatch(commandBuffer, 1, 1, ceil(configuration.size[2] / 2.0));
								}
								else {
									vkCmdDispatch(commandBuffer, 1, 1, configuration.size[2]);
								}
							}
						}
					}
					for (uint32_t j = 0; j < configuration.numberBatches; j++) {
						localFFTPlan.axes[1].pushConstants.batch = j;
						for (uint32_t i = 0; i < configuration.coordinateFeatures; i++) {
							localFFTPlan.axes[1].pushConstants.coordinate = i;
							vkCmdPushConstants(commandBuffer, localFFTPlan.axes[1].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTPushConstantsLayout), &localFFTPlan.axes[1].pushConstants);
							vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[1].pipeline);
							vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[1].pipelineLayout, 0, 1, &localFFTPlan.axes[1].descriptorSet, 0, NULL);
							if (configuration.performZeropadding[2]) {
								if (configuration.performR2C == true)
									vkCmdDispatch(commandBuffer, configuration.size[0] / 2 / localFFTPlan.axes[1].axisBlock[0], 1, ceil(configuration.size[2] / 2.0 / localFFTPlan.axes[1].axisBlock[2]));
								else
									vkCmdDispatch(commandBuffer, configuration.size[0] / localFFTPlan.axes[1].axisBlock[0], 1, ceil(configuration.size[2] / 2.0 / localFFTPlan.axes[1].axisBlock[2]));
							}
							else {
								if (configuration.performR2C == true)
									vkCmdDispatch(commandBuffer, configuration.size[0] / 2 / localFFTPlan.axes[1].axisBlock[0], 1, configuration.size[2] / localFFTPlan.axes[1].axisBlock[2]);
								else
									vkCmdDispatch(commandBuffer, configuration.size[0] / localFFTPlan.axes[1].axisBlock[0], 1, configuration.size[2] / localFFTPlan.axes[1].axisBlock[2]);

							}
						}
					}
					vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

				}

				// transpose 0 - 1, if needed
				if (configuration.performTranspose[0]) {
					for (uint32_t j = 0; j < configuration.numberBatches; j++) {
						localFFTPlan.transpose[0].pushConstants.batch = j;
						for (uint32_t i = 0; i < configuration.coordinateFeatures; i++) {
							localFFTPlan.transpose[0].pushConstants.coordinate = i;
							vkCmdPushConstants(commandBuffer, localFFTPlan.transpose[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTTransposePushConstantsLayout), &localFFTPlan.transpose[0].pushConstants);
							vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.transpose[0].pipeline);
							vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.transpose[0].pipelineLayout, 0, 1, &localFFTPlan.transpose[0].descriptorSet, 0, NULL);
							if (configuration.performR2C == true) {
								if (localFFTPlan.transpose[0].pushConstants.ratioDirection)
									vkCmdDispatch(commandBuffer, configuration.size[0] / 2 / localFFTPlan.transpose[0].transposeBlock[0], configuration.size[1] / localFFTPlan.transpose[0].transposeBlock[1], configuration.size[2] / localFFTPlan.transpose[0].transposeBlock[2]);
								else
									vkCmdDispatch(commandBuffer, configuration.size[1] / localFFTPlan.transpose[0].transposeBlock[0], configuration.size[0] / 2 / localFFTPlan.transpose[0].transposeBlock[1], configuration.size[2] / localFFTPlan.transpose[0].transposeBlock[2]);

							}
							else {
								if (localFFTPlan.transpose[0].pushConstants.ratioDirection)
									vkCmdDispatch(commandBuffer, configuration.size[0] / localFFTPlan.transpose[0].transposeBlock[0], configuration.size[1] / localFFTPlan.transpose[0].transposeBlock[1], configuration.size[2] / localFFTPlan.transpose[0].transposeBlock[2]);
								else
									vkCmdDispatch(commandBuffer, configuration.size[1] / localFFTPlan.transpose[0].transposeBlock[0], configuration.size[0] / localFFTPlan.transpose[0].transposeBlock[1], configuration.size[2] / localFFTPlan.transpose[0].transposeBlock[2]);

							}
						}
					}
					vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

				}

			}
			//FFT axis 0
			for (uint32_t j = 0; j < configuration.numberBatches; j++) {
				localFFTPlan.axes[0].pushConstants.batch = j;
				for (uint32_t i = 0; i < configuration.coordinateFeatures; i++) {
					localFFTPlan.axes[0].pushConstants.coordinate = i;
					vkCmdPushConstants(commandBuffer, localFFTPlan.axes[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkFFTPushConstantsLayout), &localFFTPlan.axes[0].pushConstants);
					vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[0].pipeline);
					vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localFFTPlan.axes[0].pipelineLayout, 0, 1, &localFFTPlan.axes[0].descriptorSet, 0, NULL);
					if (configuration.performZeropadding[1]) {
						if (configuration.performZeropadding[2]) {
							if (configuration.performR2C == true)
								vkCmdDispatch(commandBuffer, 1, ceil(configuration.size[1] / 2.0 / 2.0 / localFFTPlan.axes[0].axisBlock[1]), ceil(configuration.size[2] / 2.0 / localFFTPlan.axes[0].axisBlock[2]));
							else
								vkCmdDispatch(commandBuffer, 1, ceil(configuration.size[1] / 2.0 / localFFTPlan.axes[0].axisBlock[1]), ceil(configuration.size[2] / 2.0 / localFFTPlan.axes[0].axisBlock[2]));
						}
						else {
							if (configuration.performR2C == true)
								vkCmdDispatch(commandBuffer, 1, ceil(configuration.size[1] / 2.0 / 2.0 / localFFTPlan.axes[0].axisBlock[1]), configuration.size[2] / localFFTPlan.axes[0].axisBlock[2]);
							else
								vkCmdDispatch(commandBuffer, 1, ceil(configuration.size[1] / 2.0 / localFFTPlan.axes[0].axisBlock[1]), configuration.size[2] / localFFTPlan.axes[0].axisBlock[2]);
						}
					}
					else {
						if (configuration.performZeropadding[2]) {
							if (configuration.performR2C == true)
								vkCmdDispatch(commandBuffer, 1, configuration.size[1] / 2 / localFFTPlan.axes[0].axisBlock[1], ceil(configuration.size[2] / 2.0 / localFFTPlan.axes[0].axisBlock[2]));
							else
								vkCmdDispatch(commandBuffer, 1, configuration.size[1] / localFFTPlan.axes[0].axisBlock[1], ceil(configuration.size[2] / 2.0 / localFFTPlan.axes[0].axisBlock[2]));
						}
						else {
							if (configuration.performR2C == true)
								vkCmdDispatch(commandBuffer, 1, configuration.size[1] / 2 / localFFTPlan.axes[0].axisBlock[1], configuration.size[2] / localFFTPlan.axes[0].axisBlock[2]);
							else
								vkCmdDispatch(commandBuffer, 1, configuration.size[1] / localFFTPlan.axes[0].axisBlock[1], configuration.size[2] / localFFTPlan.axes[0].axisBlock[2]);
						}
					}

				}
			}
			vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);


		}

	}
	void deleteVulkanFFT() {
		for (uint32_t i = 0; i < configuration.FFTdim; i++) {
			deleteAxis(&localFFTPlan.axes[i]);
		}

		for (uint32_t i = 0; i < 2; i++) {
			if (configuration.performTranspose[i])
				deleteTranspose(&localFFTPlan.transpose[i]);
			else
				deleteAxis(&localFFTPlan.supportAxes[i]);
		}
		if (configuration.performConvolution) {
			for (uint32_t i = 0; i < configuration.FFTdim; i++) {
				deleteAxis(&localFFTPlan_inverse_convolution.axes[i]);
			}
			for (uint32_t i = 0; i < configuration.FFTdim - 1; i++) {
				if (configuration.performTranspose[i])
					deleteTranspose(&localFFTPlan_inverse_convolution.transpose[i]);
				else
					deleteAxis(&localFFTPlan_inverse_convolution.supportAxes[i]);
			}
		}
	}
};
