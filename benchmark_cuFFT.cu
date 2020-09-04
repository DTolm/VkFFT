#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cufft.h>

#define GROUP 1
#include <chrono>

int main()
{
    const int num_benchmark_samples_2D = 10;
    const int num_benchmark_samples_3D = 9;
    const int num_runs = 5;
    //cuFFT works best in when last dimension is the longest in R2C mode
    printf("First %d runs are a warmup\n", num_runs);
    int benchmark_dimensions_2D[num_benchmark_samples_2D][4] = { {1024, 1024, 1, 2},  {64, 64, 1, 2}, {256, 256, 1, 2}, {256, 1024, 1, 2}, {512, 512, 1, 2}, {1024, 1024, 1, 2}, {256, 4096, 1, 2}, {1024, 2048, 1, 2},{2048, 4096, 1, 2}, {4096, 4096, 1, 2} };
    int benchmark_dimensions_3D[num_benchmark_samples_3D][4] = { {32, 32, 32, 3}, {64, 64, 64, 3}, {32, 256, 256, 3}, {32, 256, 1024, 3}, {256, 256, 256, 3},  {8, 1024, 2048, 3},  {128, 512, 512, 3}, {256, 256, 2048, 3}, {8, 4096, 4096, 3}};
   
    //for 8k test
    /*const int num_benchmark_samples_2D = 6;
    const int num_benchmark_samples_3D = 3;
    const int num_runs = 5;
    int benchmark_dimensions_2D[num_benchmark_samples_2D][4] = { {1024, 1024, 1, 2}, {32, 8192, 1, 2}, {256, 8192, 1, 2}, {1024, 8192, 1, 2}, {4096, 8192, 1, 2}, {8192, 8192, 1, 2} };
    int benchmark_dimensions_3D[num_benchmark_samples_3D][4] = { {32, 32, 8192, 3}, {64, 256, 8192, 3}, {8, 1024, 8192, 3} };
    */
    //you can check this with arrays below
    //int benchmark_dimensions_2D[num_benchmark_samples_2D][4] = { {1024, 1024, 1, 2}, {32, 32, 1, 2}, {64, 64, 1, 2}, {256, 32, 1, 2}, {256, 256, 1, 2}, {1024, 256, 1, 2},{1024, 1024, 1, 2}, {4096, 256, 1, 2}, {4096, 2048, 1, 2}, {4096, 4096, 1, 2} };
    //int benchmark_dimensions_3D[num_benchmark_samples_3D][4] = { {32, 32, 32, 3}, {64, 64, 64, 3}, {256, 32, 32, 3}, {256, 256, 32, 3}, {256, 256, 256, 3}, {1024, 256, 32, 3}, {1024, 1024, 8, 3}, {2048, 1024, 8, 3}, {2048, 256, 256, 3}, {4096, 4096, 8, 3}, {4096, 4096, 32, 3} };
    double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples

    for (int n = 0; n < num_benchmark_samples_2D; n++) {

        for (int r = 0; r < num_runs; r++) {
            cufftHandle planR2C;
            cufftHandle planC2R;
            cufftComplex* dataC;
            cufftReal* dataR;

            cufftReal* inputReal;
            int dims[2] = { benchmark_dimensions_2D[n][0] , benchmark_dimensions_2D[n][1] };

            inputReal = (cufftReal*)(malloc(sizeof(cufftReal) * dims[0] * dims[1]));

            for (int j = 0; j < dims[1]; j++) {
                for (int i = 0; i < dims[0]; i++) {
                    inputReal[i + j * dims[0]] = i;
                }
            }
            cudaMalloc((void**)&dataC, sizeof(cufftComplex) * dims[0] * (dims[1] / 2 + 1));
            cudaMalloc((void**)&dataR, sizeof(cufftReal) * dims[0] * dims[1]);
            cudaMemcpy(dataR, inputReal, sizeof(cufftReal) * dims[0] * dims[1], cudaMemcpyHostToDevice);
            if (cudaGetLastError() != cudaSuccess) {
                fprintf(stderr, "Cuda error: Failed to allocate\n");
                return;
            }

            if (cufftPlanMany(&planC2R, 2, dims,
                NULL, 1, 0,
                NULL, 1, 0,
                CUFFT_C2R, GROUP) != CUFFT_SUCCESS) {
                fprintf(stderr, "CUFFT Error: Unable to create C2R plan\n");
                return;
            }
            if (cufftPlanMany(&planR2C, 2, dims,
                NULL, 1, 0,
                NULL, 1, 0,
                CUFFT_R2C, GROUP) != CUFFT_SUCCESS) {
                fprintf(stderr, "CUFFT Error: Unable to create R2C plan\n");
                return;
            }

            double totTime = 0;
            int batch = ((512.0 * 1024.0 * 1024.0) / dims[0] / (dims[1] / 2 + 1) > 1000) ? 1000 : (512.0 * 1024.0 * 1024.0) / dims[0] / (dims[1] / 2 + 1);
            if (batch == 0) batch = 1;
            //batch *= 5;//makes result more smooth, takes longer time
            auto timeSubmit = std::chrono::steady_clock::now();
            cudaDeviceSynchronize();
            for (int i = 0; i < batch; i++) {

                cufftExecR2C(planR2C, dataR, dataC);
                cufftExecC2R(planC2R, dataC, dataR);
            }
            cudaDeviceSynchronize();
            auto timeEnd = std::chrono::steady_clock::now();
            totTime = (std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001) / batch;

            printf("System: %dx%dx%d, run: %d, Buffer: %d MB, time per step: %0.3f ms, batch: %d\n", dims[1], dims[0], 1, r, (sizeof(cufftReal) * dims[0] * dims[1] + sizeof(cufftComplex) * dims[0] * (dims[1] / 2 + 1)) / 1024 / 1024, totTime, batch);
            int bufferSize = sizeof(float) * 2 * (dims[1] / 2 + 1) * dims[0];

            if (n > 0) benchmark_result += ((double)bufferSize / 1024) / totTime;

            /*cufftReal* output = (cufftReal*)(malloc(sizeof(cufftReal) * dims[0] * dims[1]));
            cudaMemcpy(output, dataR, sizeof(cufftReal) * dims[0] * dims[1], cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            for (int i = 0; i < 20; i++) {
                printf("%f %f\n", output[i] / (dims[0] * dims[1]), inputReal[i]);
            }*/
            cufftDestroy(planR2C);
            cudaFree(dataR);
            cufftDestroy(planC2R);
            cudaFree(dataC);
        }
    }
    for (int n = 0; n < num_benchmark_samples_3D; n++) {

        for (int r = 0; r < num_runs; r++) {
            cufftHandle planR2C;
            cufftHandle planC2R;
            cufftComplex* dataC;
            cufftReal* dataR;

            cufftReal* inputReal;
            int dims[3] = { benchmark_dimensions_3D[n][0] , benchmark_dimensions_3D[n][1] , benchmark_dimensions_3D[n][2] };

            inputReal = (cufftReal*)(malloc(sizeof(cufftReal) * dims[0] * dims[1] * dims[2]));
            for (int k = 0; k < dims[2]; k++) {
                for (int j = 0; j < dims[1]; j++) {
                    for (int i = 0; i < dims[0]; i++) {
                        inputReal[i + j * dims[0] + k * dims[0] * dims[1]] = k;
                    }
                }
            }
            cudaMalloc((void**)&dataC, sizeof(cufftComplex) * dims[0] * dims[1] * (dims[2] / 2 + 1));
            cudaMalloc((void**)&dataR, sizeof(cufftReal) * dims[0] * dims[1] * dims[2]);
            cudaMemcpy(dataR, inputReal, sizeof(cufftReal) * dims[0] * dims[1] * dims[2], cudaMemcpyHostToDevice);
            if (cudaGetLastError() != cudaSuccess) {
                fprintf(stderr, "Cuda error: Failed to allocate\n");
                return;
            }

            if (cufftPlanMany(&planC2R, 3, dims,
                NULL, 1, 0,
                NULL, 1, 0,
                CUFFT_C2R, GROUP) != CUFFT_SUCCESS) {
                fprintf(stderr, "CUFFT Error: Unable to create C2R plan\n");
                return;
            }
            if (cufftPlanMany(&planR2C, 3, dims,
                NULL, 1, 0,
                NULL, 1, 0,
                CUFFT_R2C, GROUP) != CUFFT_SUCCESS) {
                fprintf(stderr, "CUFFT Error: Unable to create R2C plan\n");
                return;
            }

            double totTime = 0;
            int batch = ((512.0 * 1024.0 * 1024.0) / dims[0] / dims[1] / (dims[2] / 2 + 1) > 1000) ? 1000 : (512.0 * 1024.0 * 1024.0) / dims[0] / dims[1] / (dims[2] / 2 + 1);
            if (batch == 0) batch = 1;
            //batch *= 5; //makes result more smooth, takes longer time
            auto timeSubmit = std::chrono::steady_clock::now();
            cudaDeviceSynchronize();
            for (int i = 0; i < batch; i++) {

                cufftExecR2C(planR2C, dataR, dataC);
                cufftExecC2R(planC2R, dataC, dataR);

            }
            cudaDeviceSynchronize();
            auto timeEnd = std::chrono::steady_clock::now();
            totTime = (std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001) / batch;

            printf("System: %dx%dx%d, run: %d, Buffer: %d MB, time per step: %0.3f ms, batch: %d\n", dims[2], dims[1], dims[0], r, (sizeof(cufftReal) * dims[0] * dims[1] * dims[2] + sizeof(cufftComplex) * dims[0] * dims[1] * (dims[2] / 2 + 1)) / 1024 / 1024, totTime, batch);
            int bufferSize = sizeof(float) * 2 * (dims[2] / 2 + 1) * dims[1] * dims[0];
            benchmark_result += ((double)bufferSize / 1024) / totTime;

            cufftDestroy(planR2C);
            cudaFree(dataR);
            cufftDestroy(planC2R);
            cudaFree(dataC);
        }
    }
    benchmark_result /= ((num_benchmark_samples_3D + num_benchmark_samples_2D - 1) * num_runs);
    printf("Benchmark score: %d\n", (int)(benchmark_result));
}
