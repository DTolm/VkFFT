//general parts
#include <stdio.h>
#include <vector>
#include <memory>
#include <string.h>
#include <chrono>
#include <thread>
#include <iostream>
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>
#include <assert.h>
//CUDA parts
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>
#include <cuda_fp16.h>
#include <cufftXt.h>
#define GROUP 1


void sample_2_benchmark_cuFFT_half(bool file_output, FILE* output)
{
	const int num_runs = 3;
	if (file_output)
		fprintf(output, "2 - cuFFT FFT + iFFT C2C benchmark 1D batched in half precision\n");
	printf("2 - cuFFT FFT + iFFT C2C benchmark 1D batched in half precision\n");
	double benchmark_result[2] = { 0,0 };//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	half2* inputC = (half2*)malloc((uint64_t)sizeof(half2) * pow(2, 27));
	for (uint64_t i = 0; i < pow(2, 27); i++) {
		inputC[i].x = (half) (2 * ((double)rand()) / RAND_MAX - 1.0);
		inputC[i].y = (half) (2 * ((double)rand()) / RAND_MAX - 1.0);
	}
	for (int n = 0; n < 25; n++) {
		double run_time[num_runs][2];
		for (int r = 0; r < num_runs; r++) {
			cufftHandle planHalf;
			half2* dataC_in;
			half2* dataC_out;

			uint64_t dims[3];
			dims[0] = 4 * pow(2, n); //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			if (n == 0) dims[0] = 4096;
			dims[1] = 64 * 32 * pow(2, 16) / dims[0];
			if (dims[1] == 0) dims[1] = 1;
			//dims[1] = (dims[1] > 32768) ? 32768 : dims[1];
			dims[2] = 1;
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
			switch (1){
			case 1:
				local_dims[0] = (long long)dims[0];
				local_dims[1] = (long long)dims[1];
				local_dims[2] = (long long)dims[2];
				break;
			case 2:
				local_dims[0] = (long long)dims[1];
				local_dims[1] = (long long)dims[0];
				local_dims[2] = (long long)dims[2];
				break;
			case 3:
				local_dims[0] = (long long)dims[2];
				local_dims[1] = (long long)dims[1];
				local_dims[2] = (long long)dims[0];
				break;
			}
			res = cufftXtMakePlanMany(
				planHalf, 1, local_dims, NULL, 1, 1, CUDA_C_16F,
				NULL, 1, 1, CUDA_C_16F, dims[1], &ws, CUDA_C_16F);
			
			assert(res == CUFFT_SUCCESS);

			double totTime = 0;
			uint64_t cuBufferSize = sizeof(half) * 2 * dims[0] * dims[1] * dims[2];
			uint64_t num_iter = ((4096 * 1024.0 * 1024.0) / cuBufferSize > 1000) ? 1000 : (4096 * 1024.0 * 1024.0) / cuBufferSize ;
			if (num_iter == 0) num_iter = 1;
			std::chrono::steady_clock::time_point timeSubmit = std::chrono::steady_clock::now();
			for (int i = 0; i < num_iter; i++) {

				res=cufftXtExec(planHalf, dataC_in, dataC_out, CUFFT_FORWARD);
				//assert(res == CUFFT_SUCCESS);
				res=cufftXtExec(planHalf, dataC_out, dataC_in, CUFFT_INVERSE);
				//assert(res == CUFFT_SUCCESS);
			}
			cudaDeviceSynchronize();
			std::chrono::steady_clock::time_point timeEnd = std::chrono::steady_clock::now();
			totTime = (std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001) / num_iter;
			run_time[r][0] = totTime;
			if (n > 0) {
				if (r == num_runs - 1) {
					double std_error = 0;
					double avg_time = 0;
					for (uint64_t t = 0; t < num_runs; t++) {
						avg_time += run_time[t][0];
					}
					avg_time /= num_runs;
					for (uint64_t t = 0; t < num_runs; t++) {
						std_error += (run_time[t][0] - avg_time) * (run_time[t][0] - avg_time);
					}
					std_error = sqrt(std_error / num_runs);
					if (file_output)
						fprintf(output, "cuFFT System: %" PRIu64 " %" PRIu64 "x%" PRIu64 " Buffer: %" PRIu64 " MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %" PRIu64 " benchmark: %" PRIu64 "\n", (uint64_t)log2(dims[0]), dims[0], dims[1], cuBufferSize / 1024 / 1024, avg_time, std_error, num_iter, (uint64_t)(((double)cuBufferSize * sizeof(float) / sizeof(half) / 1024) / avg_time));

					printf("cuFFT System: %" PRIu64 " %" PRIu64 "x%" PRIu64 " Buffer: %" PRIu64 " MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %" PRIu64 " benchmark: %" PRIu64 "\n", (uint64_t)log2(dims[0]), dims[0], dims[1], cuBufferSize / 1024 / 1024, avg_time, std_error, num_iter, (uint64_t)(((double)cuBufferSize * sizeof(float) / sizeof(half) / 1024) / avg_time));
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
	benchmark_result[0] /= (25 - 1);
	if (file_output)
		fprintf(output, "Benchmark score cuFFT: %" PRIu64 "\n", (uint64_t)(benchmark_result[0]));
	printf("Benchmark score cuFFT: %" PRIu64 "\n", (uint64_t)(benchmark_result[0]));

}
