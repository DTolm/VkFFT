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

//CUDA parts
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>

#define GROUP 1


void sample_0_benchmark_cuFFT_single(bool file_output, FILE* output)
{
	
	const int num_runs = 3;
	if (file_output)
		fprintf(output, "0 - cuFFT FFT + iFFT C2C benchmark 1D batched in single precision\n");
	printf("0 - cuFFT FFT + iFFT C2C benchmark 1D batched in single precision\n");
	double benchmark_result[2] = { 0,0 };//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	cufftComplex* inputC = (cufftComplex*)malloc((uint64_t)sizeof(cufftComplex)*pow(2, 27));
	for (uint64_t i = 0; i < pow(2, 27); i++) {
		inputC[i].x = 2 * ((float)rand()) / RAND_MAX - 1.0;
		inputC[i].y = 2 * ((float)rand()) / RAND_MAX - 1.0;
	}
	for (int n = 0; n < 26; n++) {
		double run_time[num_runs][2];
		for (int r = 0; r < num_runs; r++) {
			cufftHandle planC2C;
			cufftComplex* dataC;

			uint64_t dims[3];
			dims[0] = 4 * pow(2, n); //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			if (n == 0) dims[0] = 4096; 
			dims[1] = 64* 32 * pow(2, 16)/dims[0];
			//dims[1] = (dims[1] > 32768) ? 32768 : dims[1];
			if (dims[1] == 0) dims[1] = 1;
			dims[2] = 1;
			
			cudaMalloc((void**)&dataC, sizeof(cufftComplex) * dims[0] * dims[1] * dims[2]);

			cudaMemcpy(dataC, inputC, sizeof(cufftComplex) * dims[0] * dims[1] * dims[2], cudaMemcpyHostToDevice);
			if (cudaGetLastError() != cudaSuccess) {
				fprintf(stderr, "Cuda error: Failed to allocate\n");
				return;
			}
			uint64_t sizeCUDA;
			switch (1) {
			case 1:
				cufftPlan1d(&planC2C, dims[0], CUFFT_C2C, dims[1]);
				cufftGetSize1d(planC2C, dims[0], CUFFT_C2C, dims[1], (size_t*)&sizeCUDA);
				break;
			case 2:
				cufftPlan2d(&planC2C, dims[1], dims[0], CUFFT_C2C);
				break;
			case 3:
				cufftPlan3d(&planC2C, dims[2], dims[1], dims[0], CUFFT_C2C);
				break;
			}

			float totTime = 0;
			uint64_t cuBufferSize = sizeof(float) * 2 * dims[0] * dims[1] * dims[2];
			uint64_t num_iter = ((3*4096 * 1024.0 * 1024.0) / cuBufferSize > 1000) ? 1000 : (3*4096 * 1024.0 * 1024.0) / cuBufferSize;
			if (num_iter == 0) num_iter = 1;
			
			std::chrono::steady_clock::time_point timeSubmit = std::chrono::steady_clock::now();
			for (int i = 0; i < num_iter; i++) {

				cufftExecC2C(planC2C, dataC, dataC, -1);
				cufftExecC2C(planC2C, dataC, dataC, 1);
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
						fprintf(output, "cuFFT System: %" PRIu64 " %" PRIu64 "x%" PRIu64 " Buffer: %" PRIu64 " MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %" PRIu64 " benchmark: %" PRIu64 "\n", (uint64_t)log2(dims[0]), dims[0], dims[1], cuBufferSize / 1024 / 1024, avg_time, std_error, num_iter, (uint64_t)(((double)cuBufferSize / 1024) / avg_time));

					printf("cuFFT System: %" PRIu64 " %" PRIu64 "x%" PRIu64 " Buffer: %" PRIu64 " MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %" PRIu64 " benchmark: %" PRIu64 "\n", (uint64_t)log2(dims[0]), dims[0], dims[1], cuBufferSize / 1024 / 1024, avg_time, std_error, num_iter, (uint64_t)(((double)cuBufferSize / 1024) / avg_time));
					benchmark_result[0] += ((double)cuBufferSize / 1024) / avg_time;
				}

			}
			cufftDestroy(planC2C);
			cudaFree(dataC);
			cudaDeviceSynchronize();
			//cufftComplex* output_cuFFT = (cufftComplex*)(malloc(sizeof(cufftComplex) * dims[0] * dims[1] * dims[2]));
			//cudaMemcpy(output_cuFFT, dataC, sizeof(cufftComplex) * dims[0] * dims[1] * dims[2], cudaMemcpyDeviceToHost);
			//cudaDeviceSynchronize();
			

		}
	}
	free(inputC);
	benchmark_result[0] /= (26 - 1);
	if (file_output)
		fprintf(output, "Benchmark score cuFFT: %" PRIu64 "\n", (uint64_t)(benchmark_result[0]));
	printf("Benchmark score cuFFT: %" PRIu64 "\n", (uint64_t)(benchmark_result[0]));

}
