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


void sample_3_benchmark_cuFFT_single_3d(bool file_output, FILE* output)
{
	if (file_output)
		fprintf(output, "3 - cuFFT FFT + iFFT C2C multidimensional benchmark in single precision\n");
	printf("3 - cuFFT FFT + iFFT C2C multidimensional benchmark in single precision\n");
	const int num_benchmark_samples = 39;
	const int num_runs = 3;
	uint64_t benchmark_dimensions[num_benchmark_samples][4] = { {1024, 1024, 1, 2},
	{720, 480, 1, 2},{1280, 720, 1, 2},{1920, 1080, 1, 2}, {2560, 1440, 1, 2},{3840, 2160, 1, 2},{7680, 4320, 1, 2},
	{(uint64_t)pow(2,6), (uint64_t)pow(2,6), 1, 2},	{(uint64_t)pow(2,7), (uint64_t)pow(2,6), 1, 2}, {(uint64_t)pow(2,7), (uint64_t)pow(2,7), 1, 2},	{(uint64_t)pow(2,8), (uint64_t)pow(2,7), 1, 2},{(uint64_t)pow(2,8), (uint64_t)pow(2,8), 1, 2},
	{(uint64_t)pow(2,9), (uint64_t)pow(2,8), 1, 2},{(uint64_t)pow(2,9), (uint64_t)pow(2,9), 1, 2},	{(uint64_t)pow(2,10), (uint64_t)pow(2,9), 1, 2},{(uint64_t)pow(2,10), (uint64_t)pow(2,10), 1, 2},	{(uint64_t)pow(2,11), (uint64_t)pow(2,10), 1, 2},{(uint64_t)pow(2,11), (uint64_t)pow(2,11), 1, 2},
	{(uint64_t)pow(2,12), (uint64_t)pow(2,11), 1, 2},{(uint64_t)pow(2,12), (uint64_t)pow(2,12), 1, 2},	{(uint64_t)pow(2,13), (uint64_t)pow(2,12), 1, 2},	{(uint64_t)pow(2,13), (uint64_t)pow(2,13), 1, 2},{(uint64_t)pow(2,14), (uint64_t)pow(2,13), 1, 2},
	{(uint64_t)pow(2,4), (uint64_t)pow(2,4), (uint64_t)pow(2,4), 3} ,{(uint64_t)pow(2,5), (uint64_t)pow(2,4), (uint64_t)pow(2,4), 3} ,{(uint64_t)pow(2,5), (uint64_t)pow(2,5), (uint64_t)pow(2,4), 3} ,{(uint64_t)pow(2,5), (uint64_t)pow(2,5), (uint64_t)pow(2,5), 3},{(uint64_t)pow(2,6), (uint64_t)pow(2,5), (uint64_t)pow(2,5), 3} ,{(uint64_t)pow(2,6), (uint64_t)pow(2,6), (uint64_t)pow(2,5), 3} ,
	{(uint64_t)pow(2,6), (uint64_t)pow(2,6), (uint64_t)pow(2,6), 3},{(uint64_t)pow(2,7), (uint64_t)pow(2,6), (uint64_t)pow(2,6), 3} ,{(uint64_t)pow(2,7), (uint64_t)pow(2,7), (uint64_t)pow(2,6), 3} ,{(uint64_t)pow(2,7), (uint64_t)pow(2,7), (uint64_t)pow(2,7), 3},{(uint64_t)pow(2,8), (uint64_t)pow(2,7), (uint64_t)pow(2,7), 3} , {(uint64_t)pow(2,8), (uint64_t)pow(2,8), (uint64_t)pow(2,7), 3} ,
	{(uint64_t)pow(2,8), (uint64_t)pow(2,8), (uint64_t)pow(2,8), 3},{(uint64_t)pow(2,9), (uint64_t)pow(2,8), (uint64_t)pow(2,8), 3}, {(uint64_t)pow(2,9), (uint64_t)pow(2,9), (uint64_t)pow(2,8), 3},{(uint64_t)pow(2,9), (uint64_t)pow(2,9), (uint64_t)pow(2,9), 3}
	};
	double benchmark_result[2] = { 0,0 };//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	cufftComplex* inputC = (cufftComplex*)malloc((uint64_t)sizeof(cufftComplex)*pow(2, 27));
	for (uint64_t i = 0; i < pow(2, 27); i++) {
		inputC[i].x = 2 * ((float)rand()) / RAND_MAX - 1.0;
		inputC[i].y = 2 * ((float)rand()) / RAND_MAX - 1.0;
	}
	for (int n = 0; n < num_benchmark_samples; n++) {
		double run_time[num_runs][2];
		for (int r = 0; r < num_runs; r++) {
			cufftHandle planC2C;
			cufftComplex* dataC;

			uint64_t dims[3] = { benchmark_dimensions[n][0] , benchmark_dimensions[n][1] ,benchmark_dimensions[n][2] };

			cudaMalloc((void**)&dataC, sizeof(cufftComplex) * dims[0] * dims[1] * dims[2]);

			cudaMemcpy(dataC, inputC, sizeof(cufftComplex) * dims[0] * dims[1] * dims[2], cudaMemcpyHostToDevice);
			if (cudaGetLastError() != cudaSuccess) {
				fprintf(stderr, "Cuda error: Failed to allocate\n");
				return;
			}
			switch (benchmark_dimensions[n][3]) {
			case 1:
				cufftPlan1d(&planC2C, dims[0], CUFFT_C2C, 1);
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
			uint64_t num_iter = ((4096 * 1024.0 * 1024.0) / cuBufferSize > 1000) ? 1000 : (4096 * 1024.0 * 1024.0) / cuBufferSize;
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
						fprintf(output, "cuFFT System: %" PRIu64 "x%" PRIu64 "x%" PRIu64 " Buffer: %" PRIu64 " MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %" PRIu64 " benchmark: %" PRIu64 "\n", benchmark_dimensions[n][0], benchmark_dimensions[n][1], benchmark_dimensions[n][2], cuBufferSize / 1024 / 1024, avg_time, std_error, num_iter, (uint64_t)(((double)cuBufferSize / 1024) / avg_time));

					printf("cuFFT System: %" PRIu64 "x%" PRIu64 "x%" PRIu64 " Buffer: %" PRIu64 " MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %" PRIu64 " benchmark: %" PRIu64 "\n", benchmark_dimensions[n][0], benchmark_dimensions[n][1], benchmark_dimensions[n][2], cuBufferSize / 1024 / 1024, avg_time, std_error, num_iter, (uint64_t)(((double)cuBufferSize / 1024) / avg_time));
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
	benchmark_result[0] /= (num_benchmark_samples - 1);
	if (file_output)
		fprintf(output, "Benchmark score cuFFT: %" PRIu64 "\n", (uint64_t)(benchmark_result[0]));
	printf("Benchmark score cuFFT: %" PRIu64 "\n", (uint64_t)(benchmark_result[0]));

}
