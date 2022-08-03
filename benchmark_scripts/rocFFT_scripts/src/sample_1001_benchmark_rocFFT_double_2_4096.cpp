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


void sample_1001_benchmark_rocFFT_double_2_4096(bool file_output, FILE* output, int device_id)
{

	const int num_runs = 5;
	if (file_output)
		fprintf(output, "1001 - rocFFT FFT + iFFT C2C benchmark 1D batched in double precision: all supported systems from 2 to 4096\n");
	printf("1001 - rocFFT FFT + iFFT C2C benchmark 1D batched in double precision: all supported systems from 2 to 4096\n");
	hipSetDevice(device_id);
	double benchmark_result[2] = { 0,0 };//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	hipfftDoubleComplex* inputC = (hipfftDoubleComplex*)malloc((uint64_t)sizeof(hipfftDoubleComplex) *pow(2, 27));
	for (uint64_t i = 0; i <pow(2, 27); i++) {
		inputC[i].x = 2 * ((double)rand()) / RAND_MAX - 1.0;
		inputC[i].y = 2 * ((double)rand()) / RAND_MAX - 1.0;
	}
	int num_systems = 0;
	for (int n = 1; n < 4097; n++) {
		double run_time[num_runs][2];
		for (int r = 0; r < num_runs; r++) {
			hipfftHandle planZ2Z;
			hipfftDoubleComplex* dataC;

			uint64_t dims[3];

			dims[0] = n;
			if (n == 1) dims[0] = 4096;
			uint64_t temp = dims[0];

			/*for (uint64_t j = 2; j < 14; j++)
			{
				if (temp % j == 0) {
					temp /= j;
					j = 1;
				}
			}
			if (temp != 1) break;*/
			dims[1] = pow(2, (uint64_t)log2(64 * 32 * pow(2, 15) / dims[0]));
			if (dims[1] < 1) dims[1] = 1;
			dims[2] = 1;
			
			hipMalloc((void**)&dataC, sizeof(hipfftDoubleComplex) * dims[0] * dims[1] * dims[2]);

			hipMemcpy(dataC, inputC, sizeof(hipfftDoubleComplex) * dims[0] * dims[1] * dims[2], hipMemcpyHostToDevice);
			if (hipGetLastError() != hipSuccess) {
				fprintf(stderr, "ROCM error: Failed to allocate\n");
				return;
			}
			uint64_t sizeROCm;
			switch (1) {
			case 1:
				hipfftPlan1d(&planZ2Z, dims[0], HIPFFT_Z2Z, dims[1]);
				hipfftEstimate1d(dims[0], HIPFFT_Z2Z, 1, (size_t*)&sizeROCm);
				break;
			case 2:
				hipfftPlan2d(&planZ2Z, dims[1], dims[0], HIPFFT_Z2Z);
				hipfftEstimate2d(dims[1], dims[0], HIPFFT_Z2Z, (size_t*)&sizeROCm);
				break;
			case 3:
				hipfftPlan3d(&planZ2Z, dims[2], dims[1], dims[0], HIPFFT_Z2Z);
				hipfftEstimate3d(dims[2], dims[1], dims[0], HIPFFT_Z2Z, (size_t*)&sizeROCm);
				break;
			}

			double totTime = 0;
			uint64_t rocBufferSize = sizeof(double) * 2 * dims[0] * dims[1] * dims[2];
			uint64_t num_iter = ((4096 * 1024.0 * 1024.0) / rocBufferSize > 1000) ? 1000 : (4096 * 1024.0 * 1024.0) / rocBufferSize ;
			if (num_iter == 0) num_iter = 1;
			std::chrono::steady_clock::time_point timeSubmit = std::chrono::steady_clock::now();
			for (int i = 0; i < num_iter; i++) {

				hipfftExecZ2Z(planZ2Z, dataC, dataC, -1);
				hipfftExecZ2Z(planZ2Z, dataC, dataC, 1);
			}
			hipDeviceSynchronize();
			std::chrono::steady_clock::time_point timeEnd = std::chrono::steady_clock::now();
			totTime = (std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001) / num_iter;
			run_time[r][0] = totTime;
			if (n > 1) {
				if (r == num_runs - 1) {
					num_systems++;
					double std_error = 0;
					double avg_time = 0;
					for (uint64_t t = 2; t < num_runs; t++) {
						avg_time += run_time[t][0];
					}
					avg_time /= (num_runs-2);
					for (uint64_t t = 2; t < num_runs; t++) {
						std_error += (run_time[t][0] - avg_time) * (run_time[t][0] - avg_time);
					}
					std_error = sqrt(std_error / (num_runs-2));
					
					if (file_output)
						fprintf(output, "rocFFT System: %" PRIu64 " %" PRIu64 " Buffer: %" PRIu64 " MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %" PRIu64 " benchmark: %" PRIu64 " bandwidth: %0.1f\n", dims[0], dims[1], rocBufferSize / 1024 / 1024, avg_time, std_error, num_iter, (uint64_t)(((double)rocBufferSize * sizeof(float) / sizeof(double) / 1024) / avg_time), rocBufferSize / 1024.0 / 1024.0 / 1.024 * 4 / avg_time);

					printf("rocFFT System: %" PRIu64 " %" PRIu64 " Buffer: %" PRIu64 " MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %" PRIu64 " benchmark: %" PRIu64 " bandwidth: %0.1f\n", dims[0], dims[1], rocBufferSize / 1024 / 1024, avg_time, std_error, num_iter, (uint64_t)(((double)rocBufferSize * sizeof(float) / sizeof(double) / 1024) / avg_time), rocBufferSize / 1024.0 / 1024.0 / 1.024 * 4 / avg_time);
					
					benchmark_result[0] += ((double)rocBufferSize * sizeof(float)/sizeof(double)/ 1024) / avg_time;
				}

			}
			hipfftDestroy(planZ2Z);
			hipFree(dataC);
			hipDeviceSynchronize();
			//hipfftDoubleComplex* output_rocFFT = (hipfftDoubleComplex*)(malloc(sizeof(hipfftDoubleComplex) * dims[0] * dims[1] * dims[2]));
			//hipMemcpy(output_rocFFT, dataC, sizeof(hipfftDoubleComplex) * dims[0] * dims[1] * dims[2], hipMemcpyDeviceToHost);
			//hipDeviceSynchronize();


		}
	}
	free(inputC);
	benchmark_result[0] /= (num_systems);
	if (file_output)
		fprintf(output, "Benchmark score rocFFT: %" PRIu64 "\n", (uint64_t)(benchmark_result[0]));
	printf("Benchmark score rocFFT: %" PRIu64 "\n", (uint64_t)(benchmark_result[0]));

}
