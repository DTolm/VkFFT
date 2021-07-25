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


void sample_8_benchmark_cuFFT_double_Bluestein(bool file_output, FILE* output)
{
	if (file_output)
		fprintf(output, "8 - cuFFT FFT + iFFT C2C big prime benchmark in double precision (similar to VkFFT Bluestein)\n");
	printf("8 - cuFFT FFT + iFFT C2C big prime benchmark in double precision (similar to VkFFT Bluestein)\n");
	const int num_benchmark_samples = 54;
	const int num_runs = 3;
	uint64_t benchmark_dimensions[num_benchmark_samples][4] = { {1024, 1024, 1, 2},
	{17, 17, 1, 2},{19, 19, 1, 2},{23, 23, 1, 2}, {29, 29, 1, 2},{31, 31, 1, 2},{37, 37, 1, 2},{41, 41, 1, 2},{43, 43, 1, 2},{47, 47, 1, 2},{53, 53, 1, 2},{59, 59, 1, 2},{61, 61, 1, 2},{67, 67, 1, 2},{71, 71, 1, 2},{73, 73, 1, 2},{79, 79, 1, 2},{83, 83, 1, 2},{89, 89, 1, 2},{97, 97, 1, 2},
	{17, 17, 17, 3},{19, 19, 19, 3},{23, 23, 23, 3}, {29, 29, 29, 3},{31, 31, 31, 3},{37, 37, 37, 3},{41, 41, 41, 3},{43, 43, 43, 3},{47, 47, 47, 3},{53, 53, 53, 3},{59, 59, 59, 3},{61, 61, 61, 3},{67, 67, 67, 3},{71, 71, 71, 3},{73, 73, 73, 3},{79, 79, 79, 3},{83, 83, 83, 3},{89, 89, 89, 3},{97, 97, 97, 3},
	{179, 179, 1, 2},{283, 283, 1, 2},{419, 419, 1, 2}, {547, 547, 1, 2},{661, 661, 1, 2},{811, 811, 1, 2},{947, 947, 1, 2},{1087, 1087, 1, 2},{1229, 1229, 1, 2},{1381, 1381, 1, 2},{1523, 1523, 1, 2},{2909, 2909, 1, 2},{4241, 4241, 1, 2},{6841, 6841, 1, 2},{7727, 7727, 1, 2}
	};
	double benchmark_result[2] = { 0,0 };//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	cufftDoubleComplex* inputC = (cufftDoubleComplex*)malloc((uint64_t)sizeof(cufftDoubleComplex)*pow(2, 27));
	for (uint64_t i = 0; i < pow(2, 27); i++) {
		inputC[i].x = 2 * ((double)rand()) / RAND_MAX - 1.0;
		inputC[i].y = 2 * ((double)rand()) / RAND_MAX - 1.0;
	}
	for (int n = 0; n < num_benchmark_samples; n++) {
		double run_time[num_runs][2];
		for (int r = 0; r < num_runs; r++) {
			cufftHandle planZ2Z;
			cufftDoubleComplex* dataC;

			uint64_t dims[3] = { benchmark_dimensions[n][0] , benchmark_dimensions[n][1] ,benchmark_dimensions[n][2] };

			cudaMalloc((void**)&dataC, sizeof(cufftDoubleComplex) * dims[0] * dims[1] * dims[2]);

			cudaMemcpy(dataC, inputC, sizeof(cufftDoubleComplex) * dims[0] * dims[1] * dims[2], cudaMemcpyHostToDevice);
			if (cudaGetLastError() != cudaSuccess) {
				fprintf(stderr, "Cuda error: Failed to allocate\n");
				return;
			}
			switch (benchmark_dimensions[n][3]) {
			case 1:
				cufftPlan1d(&planZ2Z, dims[0], CUFFT_Z2Z, 1);
				break;
			case 2:
				cufftPlan2d(&planZ2Z, dims[1], dims[0], CUFFT_Z2Z);
				break;
			case 3:
				cufftPlan3d(&planZ2Z, dims[2], dims[1], dims[0], CUFFT_Z2Z);
				break;
			}

			double totTime = 0;
			uint64_t cuBufferSize = sizeof(double) * 2 * dims[0] * dims[1] * dims[2];
			uint64_t num_iter = ((4096 * 1024.0 * 1024.0) / cuBufferSize > 1000) ? 1000 : (4096 * 1024.0 * 1024.0) / cuBufferSize;
			if (num_iter == 0) num_iter = 1;
			std::chrono::steady_clock::time_point timeSubmit = std::chrono::steady_clock::now();
			for (int i = 0; i < num_iter; i++) {

				cufftExecZ2Z(planZ2Z, dataC, dataC, -1);
				cufftExecZ2Z(planZ2Z, dataC, dataC, 1);
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
			cufftDestroy(planZ2Z);
			cudaFree(dataC);
			cudaDeviceSynchronize();
			//cufftDoubleComplex* output_cuFFT = (cufftDoubleComplex*)(malloc(sizeof(cufftDoubleComplex) * dims[0] * dims[1] * dims[2]));
			//cudaMemcpy(output_cuFFT, dataC, sizeof(cufftDoubleComplex) * dims[0] * dims[1] * dims[2], cudaMemcpyDeviceToHost);
			//cudaDeviceSynchronize();
			

		}
	}
	free(inputC);
	benchmark_result[0] /= (num_benchmark_samples - 1);
	if (file_output)
		fprintf(output, "Benchmark score cuFFT: %" PRIu64 "\n", (uint64_t)(benchmark_result[0]));
	printf("Benchmark score cuFFT: %" PRIu64 "\n", (uint64_t)(benchmark_result[0]));

}
