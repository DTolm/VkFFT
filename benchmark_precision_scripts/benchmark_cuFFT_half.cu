//general parts
#include <stdio.h>
#include <vector>
#include <memory>
#include <string.h>
#include <chrono>
#include <thread>
#include <iostream>
#include <assert.h>
//CUDA parts
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>
#include <cuda_fp16.h>
#include <cufftXt.h>
#define GROUP 1


int main()
{

	const int num_benchmark_samples = 26;
	const int num_runs = 7;

	uint32_t benchmark_dimensions[num_benchmark_samples][4] = { {1024, 1024, 1, 2},
		{32,32,1,2},{64,64,1,2},{256,256,1,2},{1024, 256, 1, 2}, {512, 512, 1, 2},  {1024, 1024, 1, 2} , {4096, 1024, 1, 2}, {2048, 2048, 1, 2}, {4096, 4096, 1, 2},
		{64,64,64,3}, {128,128,128, 3}, {256,256,256,3}, {512, 256, 64, 3}, {1024, 1024, 64, 3}, {4096, 256, 32, 3},  {2048, 256, 256, 3},{4096, 4096, 8, 3},
		{(uint32_t)pow(2,15), 64, 1, 2}, {(uint32_t)pow(2,16), 64, 1, 2}, {(uint32_t)pow(2,17), 64, 1, 2}, {(uint32_t)pow(2,18), 64, 1, 2},  {(uint32_t)pow(2,20), 64, 1, 2},  {(uint32_t)pow(2,22), 64, 1, 2},
		{(uint32_t)pow(2,13), (uint32_t)pow(2,13), 1, 2},{(uint32_t)pow(2,14), (uint32_t)pow(2,14), 1, 2},
	};

	double benchmark_result[2] = { 0,0 };//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	half2* inputC = (half2*)malloc((uint64_t)sizeof(half2) * 2 * 4096 * 4096 * 2 * 8);
	for (uint64_t i = 0; i < 2 * 4096 * 4096 * 2 * 8; i++) {
		inputC[i].x = (half) (2 * ((double)rand()) / RAND_MAX - 1.0);
		inputC[i].y = (half) (2 * ((double)rand()) / RAND_MAX - 1.0);
	}
	for (int n = 0; n < num_benchmark_samples; n++) {
		double run_time[num_runs][2];
		for (int r = 0; r < num_runs; r++) {
			cufftHandle planHalf;
			half2* dataC_in;
			half2* dataC_out;

			uint64_t dims[3] = { benchmark_dimensions[n][0] , benchmark_dimensions[n][1] ,benchmark_dimensions[n][2] };

			cudaMalloc((void**)&dataC_in, sizeof(half2) * dims[0] * dims[1] * dims[2]);
			cudaMalloc((void**)&dataC_out, sizeof(half2) * dims[0] * dims[1] * dims[2]);
			cudaMemcpy(dataC_in, inputC, sizeof(half2) * dims[0] * dims[1] * dims[2], cudaMemcpyHostToDevice);
			if (cudaGetLastError() != cudaSuccess) {
				fprintf(stderr, "Cuda error: Failed to allocate\n");
				return;
			}
			uint64_t sizeCUDA;
			cufftResult res = cufftCreate(&planHalf);
			size_t ws = 0;
			long long local_dims[3];
			switch (benchmark_dimensions[n][3]){
			case 1:
				local_dims[0] = (long long)benchmark_dimensions[n][0];
				local_dims[1] = (long long)benchmark_dimensions[n][1];
				local_dims[2] = (long long)benchmark_dimensions[n][2];
				break;
			case 2:
				local_dims[0] = (long long)benchmark_dimensions[n][1];
				local_dims[1] = (long long)benchmark_dimensions[n][0];
				local_dims[2] = (long long)benchmark_dimensions[n][2];
				break;
			case 3:
				local_dims[0] = (long long)benchmark_dimensions[n][2];
				local_dims[1] = (long long)benchmark_dimensions[n][1];
				local_dims[2] = (long long)benchmark_dimensions[n][0];
				break;
			}
			res = cufftXtMakePlanMany(
				planHalf, benchmark_dimensions[n][3], local_dims, NULL, 1, 1, CUDA_C_16F,
				NULL, 1, 1, CUDA_C_16F, 1, &ws, CUDA_C_16F);
			
			assert(res == CUFFT_SUCCESS);

			double totTime = 0;
			uint64_t cuBufferSize = sizeof(half) * 2 * dims[0] * dims[1] * dims[2];
			uint64_t batch = ((4096 * 1024.0 * 1024.0) / cuBufferSize > 1000) ? 1000 : (4096 * 1024.0 * 1024.0) / cuBufferSize ;
			if (batch == 0) batch = 1;
			auto timeSubmit = std::chrono::steady_clock::now();
			for (int i = 0; i < batch; i++) {

				res=cufftXtExec(planHalf, dataC_in, dataC_out, CUFFT_FORWARD);
				assert(res == CUFFT_SUCCESS);
				res=cufftXtExec(planHalf, dataC_out, dataC_in, CUFFT_INVERSE);
				assert(res == CUFFT_SUCCESS);
			}
			cudaDeviceSynchronize();
			auto timeEnd = std::chrono::steady_clock::now();
			totTime = (std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001) / batch;
			run_time[r][0] = totTime;
			if (n > 0) {
				if (r == num_runs - 1) {
					double std_error = 0;
					double avg_time = 0;
					for (uint32_t t = 0; t < num_runs; t++) {
						avg_time += run_time[t][0];
					}
					avg_time /= num_runs;
					for (uint32_t t = 0; t < num_runs; t++) {
						std_error += (run_time[t][0] - avg_time) * (run_time[t][0] - avg_time);
					}
					std_error = sqrt(std_error / num_runs);
					printf("cuFFT System: %dx%dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f batch: %d benchmark: %d\n", benchmark_dimensions[n][0], benchmark_dimensions[n][1], benchmark_dimensions[n][2], cuBufferSize / 1024 / 1024, avg_time, std_error, batch, (int)(((double)cuBufferSize * sizeof(float) / sizeof(half) / 1024) / avg_time));
					benchmark_result[0] += ((double)cuBufferSize * sizeof(float) / sizeof(half) / 1024) / avg_time;
				}

			}
			cufftDestroy(planHalf);
			cudaFree(dataC_out);
			cudaFree(dataC_in);
			cudaDeviceSynchronize();
			//cufftDoubleComplex* output_cuFFT = (cufftDoubleComplex*)(malloc(sizeof(cufftDoubleComplex) * dims[0] * dims[1] * dims[2]));
			//cudaMemcpy(output_cuFFT, dataC, sizeof(cufftDoubleComplex) * dims[0] * dims[1] * dims[2], cudaMemcpyDeviceToHost);
			//cudaDeviceSynchronize();


		}
	}
	free(inputC);
	benchmark_result[0] /= (num_benchmark_samples - 1);
	printf("Benchmark score cuFFT: %d\n", (int)(benchmark_result[0]));

}
