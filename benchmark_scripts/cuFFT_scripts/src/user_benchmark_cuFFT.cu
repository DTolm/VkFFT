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

#include "user_benchmark_cuFFT.h"

void user_benchmark_cuFFT(bool file_output, FILE* output, cuFFTUserSystemParameters* userParams)
{
	
	const int num_runs = 3;
	double benchmark_result[2] = { 0,0 };//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	uint64_t storageComplexSize;
	switch (userParams->P) {
	case 0:
		storageComplexSize = (2 * sizeof(float));
		break;
	case 1:
		storageComplexSize = (2 * sizeof(double));
		break;
	case 2:
		storageComplexSize = (2 * 2);
		break;
	}
	for (int n = 0; n < 2; n++) {
		double run_time[num_runs][2];
		for (int r = 0; r < num_runs; r++) {
			cufftHandle plan;
			cufftHandle plan2;
			void* dataC;

			int dims[3];
			int FFTdim = 1;
			if (userParams->Y > 1) FFTdim++;
			if (userParams->Z > 1) FFTdim++;
			switch (FFTdim) {
			case 1:
				dims[0] = userParams->X;
				dims[1] = 1;
				dims[2] = 1;
				break;
			case 2:
				dims[0] = userParams->Y;
				dims[1] = userParams->X;
				dims[2] = 1;
				break;
			case 3:
				dims[0] = userParams->Z;
				dims[1] = userParams->Y;
				dims[2] = userParams->X;
				break;
			}
			uint64_t bufferSize;
			if (userParams->R2C)
				bufferSize = (uint64_t)(storageComplexSize / 2) * (userParams->X + 2) * userParams->Y * userParams->Z * userParams->B;
			else 
				bufferSize = (uint64_t)storageComplexSize * userParams->X * userParams->Y * userParams->Z * userParams->B;
			
			cudaMalloc((void**)&dataC, bufferSize);
			if (cudaGetLastError() != cudaSuccess) {
				fprintf(stderr, "Cuda error: Failed to allocate\n");
				return;
			}
			//forward + inverse
			int iembed[2][3];
			int istride[2] = { 1, 1 };
			int idist[2] = {bufferSize / userParams->B / storageComplexSize, bufferSize / userParams->B / storageComplexSize};
			if (userParams->R2C) idist[0] *= 2;
			int oembed[2][3];
			int ostride[2] = { 1, 1 };
			int odist[2] = { bufferSize / userParams->B / storageComplexSize, bufferSize / userParams->B / storageComplexSize };
			if (userParams->R2C) odist[1] *= 2;
			switch (FFTdim) {
			case 1:
				iembed[0][0] = (userParams->R2C) ? dims[0] + 2 : dims[0];
				oembed[0][0] = (userParams->R2C) ? (dims[0] + 2) / 2 : dims[0];

				iembed[1][0] = (userParams->R2C) ? (dims[0] + 2) / 2 : dims[0];
				oembed[1][0] = (userParams->R2C) ? dims[0] + 2 : dims[0];
				break;
			case 2:
				iembed[0][0] = dims[0];
				iembed[0][1] = (userParams->R2C) ? dims[1] + 2 : dims[1];
				oembed[0][0] = dims[0];
				oembed[0][1] = (userParams->R2C) ? (dims[1] + 2) / 2 : dims[1];

				iembed[1][0] = dims[0];
				iembed[1][1] = (userParams->R2C) ? (dims[1] + 2) / 2 : dims[1];
				oembed[1][0] = dims[0];
				oembed[1][1] = (userParams->R2C) ? dims[1] + 2 : dims[1];
				break;
			case 3:
				iembed[0][0] = idist[0];
				iembed[0][1] = dims[1];
				iembed[0][2] = (userParams->R2C) ? dims[2] + 2 : dims[2];
				oembed[0][0] = odist[0];
				oembed[0][1] = dims[1];
				oembed[0][2] = (userParams->R2C) ? (dims[2] + 2)/2 : dims[2];

				iembed[1][0] = idist[0];
				iembed[1][1] = dims[1];
				iembed[1][2] = (userParams->R2C) ? (dims[2] + 2)/2 : dims[2];
				oembed[1][0] = odist[0];
				oembed[1][1] = dims[1];
				oembed[1][2] = (userParams->R2C) ? dims[2] + 2 : dims[2];
				break;
			}
			switch (userParams->P) {
			case 0:
				if (userParams->R2C) {
					cufftPlanMany(&plan, FFTdim, dims, iembed[0], istride[0], idist[0], oembed[0], ostride[0], odist[0], CUFFT_R2C, userParams->B);
					cufftPlanMany(&plan2, FFTdim, dims, iembed[1], istride[1], idist[1], oembed[1], ostride[1], odist[1], CUFFT_C2R, userParams->B);
				}
				else {
					cufftPlanMany(&plan, FFTdim, dims, iembed[0], istride[0], idist[0], oembed[0], ostride[0], odist[0], CUFFT_C2C, userParams->B);
				}
				break;
			case 1:
				if (userParams->R2C) {
					cufftPlanMany(&plan, FFTdim, dims, iembed[0], istride[0], idist[0], oembed[0], ostride[0], odist[0], CUFFT_D2Z, userParams->B);
					cufftPlanMany(&plan2, FFTdim, dims, iembed[1], istride[1], idist[1], oembed[1], ostride[1], odist[1], CUFFT_Z2D, userParams->B);
				}
				else
					cufftPlanMany(&plan, FFTdim, dims, iembed[0], istride[0], idist[0], oembed[0], ostride[0], odist[0], CUFFT_Z2Z, userParams->B);
				break;
			}

			float totTime = 0;
			std::chrono::steady_clock::time_point timeSubmit = std::chrono::steady_clock::now();
			for (int i = 0; i < userParams->N; i++) {
				switch (userParams->P) {
				case 0:
					if (userParams->R2C){
						cufftExecR2C(plan, (cufftReal*) dataC, (cufftComplex*) dataC);
						cufftExecC2R(plan2, (cufftComplex*) dataC, (cufftReal*) dataC);
					}else{
						cufftExecC2C(plan, (cufftComplex*) dataC, (cufftComplex*) dataC, -1);
						cufftExecC2C(plan, (cufftComplex*) dataC, (cufftComplex*) dataC, 1);
					}
					break;
				case 1:
					if (userParams->R2C){
						cufftExecD2Z(plan, (cufftDoubleReal*) dataC, (cufftDoubleComplex*) dataC);
						cufftExecZ2D(plan2, (cufftDoubleComplex*) dataC, (cufftDoubleReal*) dataC);
					}else{
						cufftExecZ2Z(plan, (cufftDoubleComplex*) dataC, (cufftDoubleComplex*) dataC, -1);
						cufftExecZ2Z(plan, (cufftDoubleComplex*) dataC, (cufftDoubleComplex*) dataC, 1);
					}
					break;
				}
			}
			cudaDeviceSynchronize();
			std::chrono::steady_clock::time_point timeEnd = std::chrono::steady_clock::now();
			totTime = (std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001) / userParams->N;
				
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
						fprintf(output, "cuFFT System: %" PRIu64 "x%" PRIu64 "x%" PRIu64 " Batch: %" PRIu64 " Buffer: %" PRIu64 " MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %" PRIu64 " benchmark: %" PRIu64 " scaled bandwidth: %0.1f\n",  userParams->X, userParams->Y, userParams->Z, userParams->B, bufferSize / 1024 / 1024, avg_time, std_error, userParams->N, (uint64_t)(((double)bufferSize / 1024) / avg_time), ((double)bufferSize / 1024.0 / 1024.0 / 1.024 * 4 * FFTdim / avg_time));

					printf("cuFFT System: %" PRIu64 "x%" PRIu64 "x%" PRIu64 " Batch: %" PRIu64 " Buffer: %" PRIu64 " MB avg_time_per_step: %0.3f ms std_error: %0.3f num_iter: %" PRIu64 " benchmark: %" PRIu64 " scaled bandwidth: %0.1f\n", userParams->X, userParams->Y, userParams->Z, userParams->B, bufferSize / 1024 / 1024, avg_time, std_error, userParams->N, (uint64_t)(((double)bufferSize / 1024) / avg_time), ((double)bufferSize / 1024.0 / 1024.0 / 1.024 * 4 * FFTdim / avg_time));
					benchmark_result[0] += ((double)bufferSize / 1024) / avg_time;
				}

			}
			cufftDestroy(plan);
			if (userParams->R2C)
				cufftDestroy(plan2);
			cudaFree(dataC);
			cudaDeviceSynchronize();
		}
	}
}
