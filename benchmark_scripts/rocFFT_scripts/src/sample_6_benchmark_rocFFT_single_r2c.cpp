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

//ROCM parts
#include "hip/hip_runtime.h"
#include <hipfft.h>

#define GROUP 1


void sample_6_benchmark_rocFFT_single_r2c(bool file_output, FILE* output)
{
	if (file_output)
		fprintf(output, "6 - rocFFT FFT + iFFT R2C/C2R multidimensional benchmark in single precision\n");
	printf("6 - rocFFT FFT + iFFT R2C/C2R multidimensional benchmark in single precision\n");
	const int num_benchmark_samples = 24;
	const int num_runs = 3;
	//printf("First %" PRIu64 " runs are a warmup\n", num_runs);
	uint64_t benchmark_dimensions[num_benchmark_samples][4] = { {1024, 1024, 1, 2}, {64, 64, 1, 2}, {256, 256, 1, 2}, {1024, 256, 1, 2}, {512, 512, 1, 2}, {1024, 1024, 1, 2},  {4096, 256, 1, 2}, {2048, 1024, 1, 2},{4096, 2048, 1, 2}, {4096, 4096, 1, 2}, {720, 480, 1, 2},{1280, 720, 1, 2},{1920, 1080, 1, 2}, {2560, 1440, 1, 2},{3840, 2160, 1, 2},
																{32, 32, 32, 3}, {64, 64, 64, 3}, {256, 256, 32, 3},  {1024, 256, 32, 3},  {256, 256, 256, 3}, {2048, 1024, 8, 3},  {512, 512, 128, 3}, {2048, 256, 256, 3}, {4096, 512, 8, 3} };
	
	double benchmark_result[2] = { 0,0 };//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	hipfftReal* inputC = (hipfftReal*)malloc((uint64_t)sizeof(hipfftReal)*pow(2, 27));
	for (uint64_t i = 0; i < pow(2, 27); i++) {
		inputC[i] = 2 * ((float)rand()) / RAND_MAX - 1.0;
	}
	for (int n = 0; n < num_benchmark_samples; n++) {
		double run_time[num_runs][2];
		for (int r = 0; r < num_runs; r++) {
			hipfftHandle planR2C;
			hipfftHandle planC2R;
			hipfftReal* dataR;
			hipfftComplex* dataC;

			uint64_t dims[3] = { benchmark_dimensions[n][0] , benchmark_dimensions[n][1] ,benchmark_dimensions[n][2] };

			hipMalloc((void**)&dataR, sizeof(hipfftComplex) * (dims[0] / 2 + 1) * dims[1] * dims[2]);
			hipMalloc((void**)&dataC, sizeof(hipfftComplex) * (dims[0] / 2 + 1) * dims[1] * dims[2]);

			hipMemcpy(dataR, inputC, sizeof(hipfftReal) * dims[0] * dims[1] * dims[2], hipMemcpyHostToDevice);
			if (hipGetLastError() != hipSuccess) {
				fprintf(stderr, "ROCM error: Failed to allocate\n");
				return;
			}
			switch (benchmark_dimensions[n][3]) {
			case 1:
				hipfftPlan1d(&planR2C, dims[0], HIPFFT_R2C, 1);
				hipfftPlan1d(&planC2R, dims[0], HIPFFT_C2R, 1);
				break;
			case 2:
				hipfftPlan2d(&planR2C, dims[1], dims[0], HIPFFT_R2C);
				hipfftPlan2d(&planC2R, dims[1], dims[0], HIPFFT_C2R);
				break;
			case 3:
				hipfftPlan3d(&planR2C, dims[2], dims[1], dims[0], HIPFFT_R2C);
				hipfftPlan3d(&planC2R, dims[2], dims[1], dims[0], HIPFFT_C2R);
				break;
			}

			float totTime = 0;
			uint64_t rocBufferSize = sizeof(float) * 2 * (dims[0]/2 + 1) * dims[1] * dims[2];
			uint64_t num_iter = ((4096 * 1024.0 * 1024.0) / rocBufferSize > 1000) ? 1000 : (4096 * 1024.0 * 1024.0) / rocBufferSize;
			if (num_iter == 0) num_iter = 1;
			std::chrono::steady_clock::time_point timeSubmit = std::chrono::steady_clock::now();
			for (int i = 0; i < num_iter; i++) {

				hipfftExecR2C(planR2C, dataR, dataC);
				hipfftExecC2R(planC2R, dataC, dataR);
			}
			hipDeviceSynchronize();
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
						fprintf(output, "rocFFT System: %" PRIu64 "x%" PRIu64 "x%" PRIu64 " Buffer: %" PRIu64 " MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %" PRIu64 " benchmark: %" PRIu64 "\n", benchmark_dimensions[n][0], benchmark_dimensions[n][1], benchmark_dimensions[n][2], rocBufferSize / 1024 / 1024, avg_time, std_error, num_iter, (uint64_t)(((double)rocBufferSize / 1024) / avg_time));

					printf("rocFFT System: %" PRIu64 "x%" PRIu64 "x%" PRIu64 " Buffer: %" PRIu64 " MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %" PRIu64 " benchmark: %" PRIu64 "\n", benchmark_dimensions[n][0], benchmark_dimensions[n][1], benchmark_dimensions[n][2], rocBufferSize / 1024 / 1024, avg_time, std_error, num_iter, (uint64_t)(((double)rocBufferSize / 1024) / avg_time));
					benchmark_result[0] += ((double)rocBufferSize / 1024) / avg_time;
				}

			}
			hipfftDestroy(planR2C);
			hipfftDestroy(planC2R);
			hipFree(dataC);
			hipFree(dataR);
			hipDeviceSynchronize();
			//hipfftComplex* output_rocFFT = (hipfftComplex*)(malloc(sizeof(hipfftComplex) * dims[0] * dims[1] * dims[2]));
			//hipMemcpy(output_rocFFT, dataC, sizeof(hipfftComplex) * dims[0] * dims[1] * dims[2], hipMemcpyDeviceToHost);
			//hipDeviceSynchronize();
			

		}
	}
	free(inputC);
	benchmark_result[0] /= (num_benchmark_samples - 1);
	if (file_output)
		fprintf(output, "Benchmark score rocFFT: %" PRIu64 "\n", (uint64_t)(benchmark_result[0]));
	printf("Benchmark score rocFFT: %" PRIu64 "\n", (uint64_t)(benchmark_result[0]));

}
