//general parts
#include <stdio.h>
#include <vector>
#include <memory>
#include <string.h>
#include <chrono>
#include <thread>
#include <iostream>

//CUDA parts
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>

#define GROUP 1


void launch_benchmark_cuFFT_single_2_4096(bool file_output, FILE* output)
{

	const int num_runs = 3;
	if (file_output)
		fprintf(output, "1000 - cuFFT FFT + iFFT C2C benchmark 1D batched in single precision: all supported systems from 2 to 4096\n");
	printf("1000 - FFT + iFFT C2C benchmark 1D batched in single precision: all supported systems from 2 to 4096\n");
	double benchmark_result[2] = { 0,0 };//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	cufftComplex* inputC = (cufftComplex*)malloc((uint64_t)sizeof(cufftComplex) * pow(2, 27));
	for (uint64_t i = 0; i < pow(2, 27); i++) {
		inputC[i].x = 2 * ((float)rand()) / RAND_MAX - 1.0;
		inputC[i].y = 2 * ((float)rand()) / RAND_MAX - 1.0;
	}
	int num_systems = 0;
	for (int n = 1; n < 4097; n++) {
		double run_time[num_runs][2];
		for (int r = 0; r < num_runs; r++) {
			cufftHandle planC2C;
			cufftComplex* dataC;

			uint32_t dims[3];

			dims[0] = n;
			if (n == 1) dims[0] = 4096;
			uint32_t temp = dims[0];

			for (uint32_t j = 2; j < 14; j++)
			{
				if (temp % j == 0) {
					temp /= j;
					j = 1;
				}
			}
			if (temp != 1) break;
			dims[1] = pow(2, (uint32_t)log2(64 * 32 * pow(2, 16) / dims[0]));
			if (dims[1] < 1) dims[1] = 1;
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
				cufftGetSize1d(planC2C, dims[0], CUFFT_C2C, dims[1], &sizeCUDA);
				break;
			case 2:
				cufftPlan2d(&planC2C, dims[1], dims[0], CUFFT_C2C);
				break;
			case 3:
				cufftPlan3d(&planC2C, dims[2], dims[1], dims[0], CUFFT_C2C);
				break;
			}

			float totTime = 0;
			uint32_t cuBufferSize = sizeof(float) * 2 * dims[0] * dims[1] * dims[2];
			uint32_t num_iter = ((3 * 4096 * 1024.0 * 1024.0) / cuBufferSize > 1000) ? 1000 : (3 * 4096 * 1024.0 * 1024.0) / cuBufferSize;
			if (num_iter == 0) num_iter = 1;

			auto timeSubmit = std::chrono::steady_clock::now();
			for (int i = 0; i < num_iter; i++) {

				cufftExecC2C(planC2C, dataC, dataC, -1);
				cufftExecC2C(planC2C, dataC, dataC, 1);
			}
			cudaDeviceSynchronize();
			auto timeEnd = std::chrono::steady_clock::now();
			totTime = (std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001) / num_iter;
			run_time[r][0] = totTime;
			if (n > 1) {
				if (r == num_runs - 1) {
					num_systems++;
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
					if (file_output)
						fprintf(output, "cuFFT System: %d %d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %d benchmark: %d bandwidth: %0.1f\n", dims[0], dims[1], cuBufferSize / 1024 / 1024, avg_time, std_error, num_iter, (int)(((double)cuBufferSize / 1024) / avg_time), cuBufferSize / 1024.0 / 1024.0 / 1.024 * 4 / avg_time);

					printf("cuFFT System: %d %d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %d benchmark: %d bandwidth: %0.1f\n", dims[0], dims[1], cuBufferSize / 1024 / 1024, avg_time, std_error, num_iter, (int)(((double)cuBufferSize / 1024) / avg_time), cuBufferSize / 1024.0 / 1024.0 / 1.024 * 4 / avg_time);
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
	benchmark_result[0] /= (num_systems);
	if (file_output)
		fprintf(output, "Benchmark score cuFFT: %d\n", (int)(benchmark_result[0]));
	printf("Benchmark score cuFFT: %d\n", (int)(benchmark_result[0]));

}
