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


int main()
{
	
	const int num_benchmark_samples = 26;
	const int num_runs = 7;

	int benchmark_dimensions[num_benchmark_samples][4] = { {1024, 1024, 1, 2},
		{32,32,1,2},{64,64,1,2},{256,256,1,2},{1024, 256, 1, 2}, {512, 512, 1, 2},  {1024, 1024, 1, 2} , {4096, 1024, 1, 2}, {2048, 2048, 1, 2}, {4096, 4096, 1, 2},
		{64,64,64,3}, {128,128,128, 3}, {256,256,256,3}, {512, 256, 64, 3}, {1024, 1024, 64, 3}, {4096, 256, 32, 3},  {2048, 256, 256, 3},{4096, 4096, 8, 3},
		{(uint32_t)pow(2,15), 64, 1, 2}, {(uint32_t)pow(2,16), 64, 1, 2}, {(uint32_t)pow(2,17), 64, 1, 2}, {(uint32_t)pow(2,18), 64, 1, 2},  {(uint32_t)pow(2,20), 64, 1, 2},  {(uint32_t)pow(2,22), 64, 1, 2},
		{(uint32_t)pow(2,13), (uint32_t)pow(2,13), 1, 2},{(uint32_t)pow(2,14), (uint32_t)pow(2,14), 1, 2},
	};

	double benchmark_result[2] = { 0,0 };//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	cufftComplex* inputC = (cufftComplex*)malloc((uint64_t)sizeof(cufftComplex)*1024*1024*1024);
	for (uint64_t i = 0; i < 1024 * 1024 * 1024; i++) {
		inputC[i].x = 2 * ((float)rand()) / RAND_MAX - 1.0;
		inputC[i].y = 2 * ((float)rand()) / RAND_MAX - 1.0;
	}
	for (int n = 0; n < num_benchmark_samples; n++) {
		double run_time[num_runs][2];
		for (int r = 0; r < num_runs; r++) {
			cufftHandle planC2C;
			cufftComplex* dataC;

			int dims[3] = { benchmark_dimensions[n][0] , benchmark_dimensions[n][1] ,benchmark_dimensions[n][2] };

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
			uint32_t cuBufferSize = sizeof(float) * 2 * dims[0] * dims[1] * dims[2];
			uint32_t batch = ((4096 * 1024.0 * 1024.0) / cuBufferSize > 1000) ? 1000 : (4096 * 1024.0 * 1024.0) / cuBufferSize;
			if (batch == 0) batch = 1;
			auto timeSubmit = std::chrono::steady_clock::now();
			for (int i = 0; i < batch; i++) {

				cufftExecC2C(planC2C, dataC, dataC, 1);
				cufftExecC2C(planC2C, dataC, dataC, -1);
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
					printf("cuFFT System: %dx%dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f batch: %d benchmark: %d\n", benchmark_dimensions[n][0], benchmark_dimensions[n][1], benchmark_dimensions[n][2], cuBufferSize / 1024 / 1024, avg_time, std_error, batch, (int)(((double)cuBufferSize / 1024) / avg_time));
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
	printf("Benchmark score cuFFT: %d\n", (int)(benchmark_result[0]));

}
