#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cufft.h>

#define GROUP 1
#include <chrono>

int main()
{
    //const int num_benchmark_samples_2D = 10;
    //const int num_benchmark_samples_3D = 9;
    //const int num_runs = 5;
    //cuFFT works best in when last dimension is the longest in R2C mode
    //printf("First %d runs are a warmup\n", num_runs);
    //int benchmark_dimensions_2D[num_benchmark_samples_2D][4] = { {1024, 1024, 1, 2},  {64, 64, 1, 2}, {256, 256, 1, 2}, {256, 1024, 1, 2}, {512, 512, 1, 2}, {1024, 1024, 1, 2}, {256, 4096, 1, 2}, {1024, 2048, 1, 2},{2048, 4096, 1, 2}, {4096, 4096, 1, 2} };
    //int benchmark_dimensions_3D[num_benchmark_samples_3D][4] = { {32, 32, 32, 3}, {64, 64, 64, 3}, {32, 256, 256, 3}, {32, 256, 1024, 3}, {256, 256, 256, 3},  {8, 1024, 2048, 3},  {128, 512, 512, 3}, {256, 256, 2048, 3}, {8, 4096, 4096, 3}};

    //for 8k test
    const int num_benchmark_samples_2D = 11;
    //const int num_benchmark_samples_3D = 3;
    const int num_runs = 7;
    int benchmark_dimensions_2D[num_benchmark_samples_2D][4] = { {1024, 1024, 1, 2}, {pow(2,13), 64, 1, 2}, {pow(2,14), 64, 1, 2}, {pow(2,15), 64, 1, 2}, {pow(2,16), 64, 1, 2}, {pow(2,17), 64, 1, 2}, {pow(2,18), 64, 1, 2},{pow(2,20), 64, 1, 2},{pow(2,22), 64, 1, 2}, {pow(2,13), pow(2,13), 1, 2},{pow(2,14), pow(2,14), 1, 2} };
    //int benchmark_dimensions_3D[num_benchmark_samples_3D][4] = { {32, 32, 8192, 3}, {64, 256, 8192, 3}, {8, 1024, 8192, 3} };

    //you can check this with arrays below
    //int benchmark_dimensions_2D[num_benchmark_samples_2D][4] = { {1024, 1024, 1, 2}, {32, 32, 1, 2}, {64, 64, 1, 2}, {256, 32, 1, 2}, {256, 256, 1, 2}, {1024, 256, 1, 2},{1024, 1024, 1, 2}, {4096, 256, 1, 2}, {4096, 2048, 1, 2}, {4096, 4096, 1, 2} };
    //int benchmark_dimensions_3D[num_benchmark_samples_3D][4] = { {32, 32, 32, 3}, {64, 64, 64, 3}, {256, 32, 32, 3}, {256, 256, 32, 3}, {256, 256, 256, 3}, {1024, 256, 32, 3}, {1024, 1024, 8, 3}, {2048, 1024, 8, 3}, {2048, 256, 256, 3}, {4096, 4096, 8, 3}, {4096, 4096, 32, 3} };
    double benchmark_result = 0;//averaged result = sum(time/num_runs/system size)/num_benchmark_samples

    for (int n = 0; n < num_benchmark_samples_2D; n++) {
        double run_time[num_runs];
        for (int r = 0; r < num_runs; r++) {
            cufftHandle planC2C;
            //cufftHandle planC2R;
            cufftComplex* dataC;
            //cufftReal* dataR;

            cufftComplex* inputComplex;
            int dims[2] = { benchmark_dimensions_2D[n][0] , benchmark_dimensions_2D[n][1] };

            inputComplex = (cufftComplex*)(malloc(sizeof(cufftComplex) * (dims[0]) * dims[1]));


            for (int j = 0; j < dims[1]; j++) {
                for (int i = 0; i < dims[0]; i++) {
                    inputComplex[i + j * dims[0]].x = i % 4 - 1.5;
                    inputComplex[i + j * dims[0]].y = 0;
                }
            }
            cudaMalloc((void**)&dataC, sizeof(cufftComplex) * dims[0] * (dims[1]));
            // cudaMalloc((void**)&dataR, sizeof(cufftReal) * 2 * (dims[0]) * dims[1]);
            cudaMemcpy(dataC, inputComplex, sizeof(cufftComplex) * (dims[0]) * dims[1], cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();

            if (cudaGetLastError() != cudaSuccess) {
                fprintf(stderr, "Cuda error: Failed to allocate \n");
                return;
            }

            if (cufftPlanMany(&planC2C, 2, dims,
                NULL, 1, 0,
                NULL, 1, 0,
                CUFFT_C2C, GROUP) != CUFFT_SUCCESS) {
                fprintf(stderr, "CUFFT Error: Unable to create C2R plan\n");
                return;
            }


            double totTime = 0;
            int batch = ((512.0 * 1024.0 * 1024.0) / dims[0] / (dims[1]) > 1000) ? 1000 : (512.0 * 1024.0 * 1024.0) / dims[0] / (dims[1]);
            if (batch == 0) batch = 1;
            //batch = 1;
            //batch *= 5;//makes result more smooth, takes longer time
            auto timeSubmit = std::chrono::steady_clock::now();
            cudaDeviceSynchronize();
            for (int i = 0; i < batch; i++) {

                cufftExecC2C(planC2C, dataC, (cufftComplex*)dataC, 1);
                cufftExecC2C(planC2C, (cufftComplex*)dataC, dataC, -1);
                //cufftExecR2C(planR2C, dataR, data);
                //cufftExecC2R(planC2R, dataC, dataR);
            }
            cudaDeviceSynchronize();
            auto timeEnd = std::chrono::steady_clock::now();
            totTime = (std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001) / batch;

            //printf("System: %dx%dx%d, run: %d, Buffer: %d MB, time per step: %0.3f ms, batch: %d\n", dims[1], dims[0], 1, r, (sizeof(cufftComplex) * dims[0] * (dims[1])) / 1024 / 1024, totTime, batch);
            uint64_t bufferSize = (sizeof(cufftComplex) * dims[0] * (dims[1])) ;

            run_time[r] = totTime;
            if (n > 0){
                if (r == num_runs - 1) {
                    double std_error = 0;
                    double avg_time = 0;
                    for (uint32_t t = 0; t < num_runs; t++) {
                        avg_time += run_time[t];
                    }
                    avg_time /= num_runs;
                    for (uint32_t t = 0; t < num_runs; t++) {
                        std_error += (run_time[t] - avg_time) * (run_time[t] - avg_time);
                    }
                    std_error = sqrt(std_error / num_runs);
                    printf("System: %dx%dx%d Buffer: %d MB avg_time_per_step: %0.3f ms std_error %0.3f batch: %d\n", dims[0], dims[1], 1, bufferSize/1024/1024, avg_time, std_error, batch);
                }
                benchmark_result += ((double)bufferSize / 1024) / totTime;
            }
            /*cufftComplex* output = (cufftComplex*)(malloc(sizeof(cufftComplex) * dims[0] * dims[1]));
            cudaMemcpy(output, dataC, sizeof(cufftComplex) * dims[0] * dims[1], cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            for (int i = 0; i < 1024; i++) {
                printf("%f %f\n", output[i].x , output[i].y);
            }*/
            free(inputComplex);
            cufftDestroy(planC2C);

            //cufftDestroy(planC2R);
            cudaFree(dataC);
            //cudaFree(dataC);
        }
    }
    benchmark_result /= ((num_benchmark_samples_2D - 1) * num_runs);
    printf("Benchmark score: %d\n", (int)(benchmark_result));
    return;
}
