//general parts
#include <stdio.h>
#include <vector>
#include <memory>
#include <string.h>
#include <chrono>
#include <thread>
#include <iostream>

//ROCM parts
#include "hip/hip_runtime.h"
#include <hipfft.h>

#define GROUP 1


void launch_benchmark_rocFFT_single_3d_2_512(bool file_output, FILE* output)
{
	if (file_output)
		fprintf(output, "1003 - rocFFT FFT + iFFT C2C multidimensional benchmark in single precision: all supported cubes from 2 to 512\n");
	printf("1003 - rocFFT FFT + iFFT C2C multidimensional benchmark in single precision: all supported cubes from 2 to 512\n");
const int num_runs = 3;
	
	double benchmark_result[2] = { 0,0 };//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	hipfftComplex* inputC = (hipfftComplex*)malloc((uint64_t)sizeof(hipfftComplex)*pow(2, 27));
	for (uint64_t i = 0; i < pow(2, 27); i++) {
		inputC[i].x = 2 * ((float)rand()) / RAND_MAX - 1.0;
		inputC[i].y = 2 * ((float)rand()) / RAND_MAX - 1.0;
	}
	int num_systems = 0;
	for (int n = 1; n < 513; n++) {
		double run_time[num_runs][2];
		for (int r = 0; r < num_runs; r++) {
			hipfftHandle planC2C;
			hipfftComplex* dataC;

			uint32_t dims[3];

			dims[0] = n;
			if (n == 1) dims[0] = 512;
			uint32_t temp = dims[0];

			for (uint32_t j = 2; j < 14; j++)
			{
				if (temp % j == 0) {
					temp /= j;
					j = 1;
				}
			}
			if (temp != 1) break;
			dims[1] = dims[0];
			dims[2] = dims[0];
			
			hipMalloc((void**)&dataC, sizeof(hipfftComplex) * dims[0] * dims[1] * dims[2]);

			hipMemcpy(dataC, inputC, sizeof(hipfftComplex) * dims[0] * dims[1] * dims[2], hipMemcpyHostToDevice);
			if (hipGetLastError() != hipSuccess) {
				fprintf(stderr, "ROCM error: Failed to allocate\n");
				return;
			}
			switch (3) {
			case 1:
				hipfftPlan1d(&planC2C, dims[0], HIPFFT_C2C, 1);
				break;
			case 2:
				hipfftPlan2d(&planC2C, dims[1], dims[0], HIPFFT_C2C);
				break;
			case 3:
				hipfftPlan3d(&planC2C, dims[2], dims[1], dims[0], HIPFFT_C2C);
				break;
			}

			float totTime = 0;
			uint32_t rocBufferSize = sizeof(float) * 2 * dims[0] * dims[1] * dims[2];
			uint32_t num_iter = ((4096 * 1024.0 * 1024.0) / rocBufferSize > 1000) ? 1000 : (4096 * 1024.0 * 1024.0) / rocBufferSize;
			if (num_iter == 0) num_iter = 1;
			auto timeSubmit = std::chrono::steady_clock::now();
			for (int i = 0; i < num_iter; i++) {

				hipfftExecC2C(planC2C, dataC, dataC, -1);
				hipfftExecC2C(planC2C, dataC, dataC, 1);
			}
			hipDeviceSynchronize();
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
						fprintf(output, "rocFFT System: %d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %d benchmark: %d bandwidth: %0.1f\n", dims[0], rocBufferSize / 1024 / 1024, avg_time, std_error, num_iter, (int)(((double)rocBufferSize / 1024) / avg_time), 3*rocBufferSize / 1024.0 / 1024.0 / 1.024 * 4 / avg_time);

					printf("rocFFT System: %d Buffer: %d MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %d benchmark: %d bandwidth: %0.1f\n", dims[0], rocBufferSize / 1024 / 1024, avg_time, std_error, num_iter, (int)(((double)rocBufferSize / 1024) / avg_time), 3*rocBufferSize / 1024.0 / 1024.0 / 1.024 * 4 / avg_time);
					benchmark_result[0] += ((double)rocBufferSize / 1024) / avg_time;
				}

			}
			hipfftDestroy(planC2C);
			hipFree(dataC);
			hipDeviceSynchronize();
			//hipfftComplex* output_rocFFT = (hipfftComplex*)(malloc(sizeof(hipfftComplex) * dims[0] * dims[1] * dims[2]));
			//hipMemcpy(output_rocFFT, dataC, sizeof(hipfftComplex) * dims[0] * dims[1] * dims[2], hipMemcpyDeviceToHost);
			//hipDeviceSynchronize();
			

		}
	}
	free(inputC);
	benchmark_result[0] /= (num_systems);
	if (file_output)
		fprintf(output, "Benchmark score rocFFT: %d\n", (int)(benchmark_result[0]));
	printf("Benchmark score rocFFT: %d\n", (int)(benchmark_result[0]));

}
