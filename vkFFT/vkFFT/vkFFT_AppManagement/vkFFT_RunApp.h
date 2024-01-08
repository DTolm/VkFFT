// This file is part of VkFFT
//
// Copyright (C) 2021 - present Dmitrii Tolmachev <dtolm96@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
#ifndef VKFFT_RUNAPP_H
#define VKFFT_RUNAPP_H
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT/vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_DispatchPlan.h"
#include "vkFFT/vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_UpdateBuffers.h"

static inline VkFFTResult VkFFTSync(VkFFTApplication* app) {
#if(VKFFT_BACKEND==0)
    vkCmdPipelineBarrier(app->configuration.commandBuffer[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, app->configuration.memory_barrier, 0, 0, 0, 0);
#elif(VKFFT_BACKEND==1)
    if (app->configuration.num_streams > 1) {
        cudaError_t res = cudaSuccess;
        for (pfUINT s = 0; s < app->configuration.num_streams; s++) {
            res = cudaEventSynchronize(app->configuration.stream_event[s]);
            if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
        }
        app->configuration.streamCounter = 0;
    }
#elif(VKFFT_BACKEND==2)
    if (app->configuration.num_streams > 1) {
        hipError_t res = hipSuccess;
        for (pfUINT s = 0; s < app->configuration.num_streams; s++) {
            res = hipEventSynchronize(app->configuration.stream_event[s]);
            if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
        }
        app->configuration.streamCounter = 0;
    }
#elif(VKFFT_BACKEND==3)
#elif(VKFFT_BACKEND==4)
    ze_result_t res = ZE_RESULT_SUCCESS;
    res = zeCommandListAppendBarrier(app->configuration.commandList[0], nullptr, 0, nullptr);
    if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_SUBMIT_BARRIER;
#elif(VKFFT_BACKEND==5)
#endif
    return VKFFT_SUCCESS;
}
static inline void printDebugInformation(VkFFTApplication* app, VkFFTAxis* axis) {
    if (app->configuration.keepShaderCode) printf("%s\n", axis->specializationConstants.code0);
    if (app->configuration.printMemoryLayout) {
        if ((axis->inputBuffer == app->configuration.inputBuffer) && (app->configuration.inputBuffer != app->configuration.buffer))
            printf("read: inputBuffer\n");
        if (axis->inputBuffer == app->configuration.buffer)
            printf("read: buffer\n");
        if (axis->inputBuffer == app->configuration.tempBuffer)
            printf("read: tempBuffer\n");
        if ((axis->inputBuffer == app->configuration.outputBuffer) && (app->configuration.outputBuffer != app->configuration.buffer))
            printf("read: outputBuffer\n");
        if ((axis->outputBuffer == app->configuration.inputBuffer) && (app->configuration.inputBuffer != app->configuration.buffer))
            printf("write: inputBuffer\n");
        if (axis->outputBuffer == app->configuration.buffer)
            printf("write: buffer\n");
        if (axis->outputBuffer == app->configuration.tempBuffer)
            printf("write: tempBuffer\n");
        if ((axis->outputBuffer == app->configuration.outputBuffer) && (app->configuration.outputBuffer != app->configuration.buffer))
            printf("write: outputBuffer\n");
    }
}
static inline VkFFTResult VkFFTAppend(VkFFTApplication* app, int inverse, VkFFTLaunchParams* launchParams) {
    VkFFTResult resFFT = VKFFT_SUCCESS;
#if(VKFFT_BACKEND==0)
    app->configuration.commandBuffer = launchParams->commandBuffer;
    VkMemoryBarrier memory_barrier = {
            VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            0,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
    };
    app->configuration.memory_barrier = &memory_barrier;
#elif(VKFFT_BACKEND==1)
    app->configuration.streamCounter = 0;
#elif(VKFFT_BACKEND==2)
    app->configuration.streamCounter = 0;
#elif(VKFFT_BACKEND==3)
    app->configuration.commandQueue = launchParams->commandQueue;
#elif(VKFFT_BACKEND==4)
    app->configuration.commandList = launchParams->commandList;
#elif(VKFFT_BACKEND==5)
    app->configuration.commandBuffer = launchParams->commandBuffer;
    app->configuration.commandEncoder = launchParams->commandEncoder;
#endif
    if ((inverse != 1) && (app->configuration.makeInversePlanOnly)) return VKFFT_ERROR_ONLY_INVERSE_FFT_INITIALIZED;
    if ((inverse == 1) && (app->configuration.makeForwardPlanOnly)) return VKFFT_ERROR_ONLY_FORWARD_FFT_INITIALIZED;
    if ((inverse != 1) && (!app->configuration.makeInversePlanOnly) && (!app->localFFTPlan)) return VKFFT_ERROR_PLAN_NOT_INITIALIZED;
    if ((inverse == 1) && (!app->configuration.makeForwardPlanOnly) && (!app->localFFTPlan_inverse)) return VKFFT_ERROR_PLAN_NOT_INITIALIZED;
    
    resFFT = VkFFTCheckUpdateBufferSet(app, 0, 0, launchParams);
    if (resFFT != VKFFT_SUCCESS) {
        return resFFT;
    }
    if (inverse != 1) {
        //FFT axis 0
        if (!app->configuration.omitDimension[0]) {
            for (pfINT l = (pfINT)app->localFFTPlan->numAxisUploads[0] - 1; l >= 0; l--) {
                VkFFTAxis* axis = &app->localFFTPlan->axes[0][l];
                resFFT = VkFFTUpdateBufferSet(app, app->localFFTPlan, axis, 0, l, 0);
                if (resFFT != VKFFT_SUCCESS) return resFFT;
                pfUINT maxCoordinate = ((app->configuration.matrixConvolution > 1) && (app->configuration.performConvolution) && (app->configuration.FFTdim == 1) && (l == 0)) ? 1 : app->configuration.coordinateFeatures;
#if(VKFFT_BACKEND==0)
                vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
                vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
                pfUINT dispatchBlock[3];
                if (l == 0) {
                    if (app->localFFTPlan->numAxisUploads[0] > 2) {
                        dispatchBlock[0] = (pfUINT)pfceil((pfUINT)pfceil(app->localFFTPlan->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)axis->axisBlock[1]) / (double)app->localFFTPlan->axisSplit[0][1]) * app->localFFTPlan->axisSplit[0][1];
                        dispatchBlock[1] = app->localFFTPlan->actualFFTSizePerAxis[0][1];
                    }
                    else {
                        if (app->localFFTPlan->numAxisUploads[0] > 1) {
                            dispatchBlock[0] = (pfUINT)pfceil((pfUINT)pfceil(app->localFFTPlan->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)axis->axisBlock[1]));
                            dispatchBlock[1] = app->localFFTPlan->actualFFTSizePerAxis[0][1];
                        }
                        else {
                            dispatchBlock[0] = app->localFFTPlan->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i;
                            dispatchBlock[1] = (pfUINT)pfceil(app->localFFTPlan->actualFFTSizePerAxis[0][1] / (double)axis->axisBlock[1]);
                        }
                    }
                }
                else {
                    dispatchBlock[0] = (pfUINT)pfceil(app->localFFTPlan->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)axis->axisBlock[0]);
                    dispatchBlock[1] = app->localFFTPlan->actualFFTSizePerAxis[0][1];
                }
                dispatchBlock[2] = maxCoordinate * app->configuration.numberBatches;
                for (int p = 2; p <app->configuration.FFTdim; p++){
                    dispatchBlock[2]*= app->localFFTPlan->actualFFTSizePerAxis[0][p];
                }
                
                if (axis->specializationConstants.mergeSequencesR2C == 1) dispatchBlock[1] = (pfUINT)pfceil(dispatchBlock[1] / 2.0);
                //if (app->configuration.performZeropadding[1]) dispatchBlock[1] = (pfUINT)pfceil(dispatchBlock[1] / 2.0);
                //if (app->configuration.performZeropadding[2]) dispatchBlock[2] = (pfUINT)pfceil(dispatchBlock[2] / 2.0);
                resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
                if (resFFT != VKFFT_SUCCESS) return resFFT;
                printDebugInformation(app, axis);
                resFFT = VkFFTSync(app);
                if (resFFT != VKFFT_SUCCESS) return resFFT;
            }
            if (app->useBluesteinFFT[0] && (app->localFFTPlan->numAxisUploads[0] > 1)) {
                for (pfINT l = 1; l < (pfINT)app->localFFTPlan->numAxisUploads[0]; l++) {
                    VkFFTAxis* axis = &app->localFFTPlan->inverseBluesteinAxes[0][l];
                    resFFT = VkFFTUpdateBufferSet(app, app->localFFTPlan, axis, 0, l, 0);
                    if (resFFT != VKFFT_SUCCESS) return resFFT;
                    pfUINT maxCoordinate = ((app->configuration.matrixConvolution > 1) && (app->configuration.performConvolution) && (app->configuration.FFTdim == 1)) ? 1 : app->configuration.coordinateFeatures;
#if(VKFFT_BACKEND==0)
                    vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
                    vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
                    pfUINT dispatchBlock[3];
                    if (l == 0) {
                        if (app->localFFTPlan->numAxisUploads[0] > 2) {
                            dispatchBlock[0] = (pfUINT)pfceil((pfUINT)pfceil(app->localFFTPlan->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)axis->axisBlock[1]) / (double)app->localFFTPlan->axisSplit[0][1]) * app->localFFTPlan->axisSplit[0][1];
                            dispatchBlock[1] = app->localFFTPlan->actualFFTSizePerAxis[0][1];
                        }
                        else {
                            if (app->localFFTPlan->numAxisUploads[0] > 1) {
                                dispatchBlock[0] = (pfUINT)pfceil((pfUINT)pfceil(app->localFFTPlan->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)axis->axisBlock[1]));
                                dispatchBlock[1] = app->localFFTPlan->actualFFTSizePerAxis[0][1];
                            }
                            else {
                                dispatchBlock[0] = app->localFFTPlan->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i;
                                dispatchBlock[1] = (pfUINT)pfceil(app->localFFTPlan->actualFFTSizePerAxis[0][1] / (double)axis->axisBlock[1]);
                            }
                        }
                    }
                    else {
                        dispatchBlock[0] = (pfUINT)pfceil(app->localFFTPlan->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)axis->axisBlock[0]);
                        dispatchBlock[1] = app->localFFTPlan->actualFFTSizePerAxis[0][1];
                    }
                    
                    dispatchBlock[2] = maxCoordinate * app->configuration.numberBatches;
                    for (int p = 2; p <app->configuration.FFTdim; p++){
                        dispatchBlock[2]*= app->localFFTPlan->actualFFTSizePerAxis[0][p];
                    }
                    
                    if (axis->specializationConstants.mergeSequencesR2C == 1) dispatchBlock[1] = (pfUINT)pfceil(dispatchBlock[1] / 2.0);
                    //if (app->configuration.performZeropadding[1]) dispatchBlock[1] = (pfUINT)pfceil(dispatchBlock[1] / 2.0);
                    //if (app->configuration.performZeropadding[2]) dispatchBlock[2] = (pfUINT)pfceil(dispatchBlock[2] / 2.0);
                    resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
                    if (resFFT != VKFFT_SUCCESS) return resFFT;
                    printDebugInformation(app, axis);
                    resFFT = VkFFTSync(app);
                    if (resFFT != VKFFT_SUCCESS) return resFFT;
                }
            }
            if (app->localFFTPlan->bigSequenceEvenR2C) {
                VkFFTAxis* axis = &app->localFFTPlan->R2Cdecomposition;
                resFFT = VkFFTUpdateBufferSetR2CMultiUploadDecomposition(app, app->localFFTPlan, axis, 0, 0, 0);
                if (resFFT != VKFFT_SUCCESS) return resFFT;
                pfUINT maxCoordinate = ((app->configuration.matrixConvolution > 1) && (app->configuration.performConvolution) && (app->configuration.FFTdim == 1)) ? 1 : app->configuration.coordinateFeatures;
                
#if(VKFFT_BACKEND==0)
                vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
                vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
                pfUINT dispatchBlock[3];
                
                dispatchBlock[0] = (app->configuration.size[0] / 2 + 1);
                for (int p = 1; p <app->configuration.FFTdim; p++){
                    dispatchBlock[0] *= app->configuration.size[p];
                }
                dispatchBlock[0] = (pfUINT)pfceil(dispatchBlock[0] / (double)(2 * axis->axisBlock[0]));
                
                dispatchBlock[1] = 1;
                dispatchBlock[2] = maxCoordinate * axis->specializationConstants.numBatches.data.i;
                resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
                if (resFFT != VKFFT_SUCCESS) return resFFT;
                printDebugInformation(app, axis);
                resFFT = VkFFTSync(app);
                if (resFFT != VKFFT_SUCCESS) return resFFT;
                //app->configuration.size[0] *= 2;
            }
        }
        for (int i = 1; i <app->configuration.FFTdim; i++){
            if (!app->configuration.omitDimension[i]) {
                if ((app->configuration.FFTdim == (i+1)) && (app->configuration.performConvolution)) {
                    
                    for (pfINT l = (pfINT)app->localFFTPlan->numAxisUploads[i] - 1; l >= 0; l--) {
                        VkFFTAxis* axis = &app->localFFTPlan->axes[i][l];
                        resFFT = VkFFTUpdateBufferSet(app, app->localFFTPlan, axis, i, l, 0);
                        if (resFFT != VKFFT_SUCCESS) return resFFT;
                        pfUINT maxCoordinate = ((app->configuration.matrixConvolution > 1) && (l == 0)) ? 1 : app->configuration.coordinateFeatures;
                        
#if(VKFFT_BACKEND==0)
                        vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
                        vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
                        pfUINT dispatchBlock[3];
                        dispatchBlock[0] = (pfUINT)pfceil(app->localFFTPlan->actualFFTSizePerAxis[i][0] / (double)axis->axisBlock[0] * app->localFFTPlan->actualFFTSizePerAxis[i][i] / (double)axis->specializationConstants.fftDim.data.i);
                        dispatchBlock[1] = 1;
                        dispatchBlock[2] = maxCoordinate * app->configuration.numberBatches;
                        for (int p = 1; p <app->configuration.FFTdim; p++){
                            if (p != i)
                                dispatchBlock[2]*= app->localFFTPlan->actualFFTSizePerAxis[i][p];
                        }
                        //if (app->configuration.mergeSequencesR2C == 1) dispatchBlock[0] = (pfUINT)pfceil(dispatchBlock[0] / 2.0);
                        //if (app->configuration.performZeropadding[2]) dispatchBlock[2] = (pfUINT)pfceil(dispatchBlock[2] / 2.0);
                        resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
                        if (resFFT != VKFFT_SUCCESS) return resFFT;
                        printDebugInformation(app, axis);
                        resFFT = VkFFTSync(app);
                        if (resFFT != VKFFT_SUCCESS) return resFFT;
                    }
                }
                else {
                    
                    for (pfINT l = (pfINT)app->localFFTPlan->numAxisUploads[i] - 1; l >= 0; l--) {
                        VkFFTAxis* axis = &app->localFFTPlan->axes[i][l];
                        resFFT = VkFFTUpdateBufferSet(app, app->localFFTPlan, axis, i, l, 0);
                        if (resFFT != VKFFT_SUCCESS) return resFFT;
                        
#if(VKFFT_BACKEND==0)
                        vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
                        vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
                        pfUINT dispatchBlock[3];
                        
                        dispatchBlock[0] = (pfUINT)pfceil(app->localFFTPlan->actualFFTSizePerAxis[i][0] / (double)axis->axisBlock[0] * app->localFFTPlan->actualFFTSizePerAxis[i][i] / (double)axis->specializationConstants.fftDim.data.i);
                        dispatchBlock[1] = 1;
                        dispatchBlock[2] = app->configuration.coordinateFeatures * app->configuration.numberBatches;
                        for (int p = 1; p <app->configuration.FFTdim; p++){
                            if (p != i)
                                dispatchBlock[2]*= app->localFFTPlan->actualFFTSizePerAxis[i][p];
                        }
                        //if (app->configuration.mergeSequencesR2C == 1) dispatchBlock[0] = (pfUINT)pfceil(dispatchBlock[0] / 2.0);
                        //if (app->configuration.performZeropadding[2]) dispatchBlock[2] = (pfUINT)pfceil(dispatchBlock[2] / 2.0);
                        resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
                        if (resFFT != VKFFT_SUCCESS) return resFFT;
                        printDebugInformation(app, axis);
                        
                        resFFT = VkFFTSync(app);
                        if (resFFT != VKFFT_SUCCESS) return resFFT;
                    }
                    if (app->useBluesteinFFT[i] && (app->localFFTPlan->numAxisUploads[i] > 1)) {
                        for (pfINT l = 1; l < (pfINT)app->localFFTPlan->numAxisUploads[i]; l++) {
                            VkFFTAxis* axis = &app->localFFTPlan->inverseBluesteinAxes[i][l];
                            resFFT = VkFFTUpdateBufferSet(app, app->localFFTPlan, axis, i, l, 0);
                            if (resFFT != VKFFT_SUCCESS) return resFFT;
#if(VKFFT_BACKEND==0)
                            vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
                            vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
                            pfUINT dispatchBlock[3];
                            dispatchBlock[0] = (pfUINT)pfceil(app->localFFTPlan->actualFFTSizePerAxis[i][0] / (double)axis->axisBlock[0] * app->localFFTPlan->actualFFTSizePerAxis[i][i] / (double)axis->specializationConstants.fftDim.data.i);
                            dispatchBlock[1] = 1;
                            dispatchBlock[2] = app->configuration.coordinateFeatures * app->configuration.numberBatches;
                            for (int p = 1; p <app->configuration.FFTdim; p++){
                                if (p != i)
                                    dispatchBlock[2]*= app->localFFTPlan->actualFFTSizePerAxis[i][p];
                            }
                            //if (app->configuration.performZeropadding[1]) dispatchBlock[1] = (pfUINT)pfceil(dispatchBlock[1] / 2.0);
                            //if (app->configuration.performZeropadding[2]) dispatchBlock[2] = (pfUINT)pfceil(dispatchBlock[2] / 2.0);
                            resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
                            if (resFFT != VKFFT_SUCCESS) return resFFT;
                            printDebugInformation(app, axis);
                            resFFT = VkFFTSync(app);
                            if (resFFT != VKFFT_SUCCESS) return resFFT;
                        }
                    }
                }
            }
        }
    }
    if (app->configuration.performConvolution) {
        
        for (int i = (int)app->configuration.FFTdim-1; i > 0; i--){

            //multiple upload ifft leftovers
            if (app->configuration.FFTdim == (i+1)) {

                for (pfINT l = (pfINT)1; l < (pfINT)app->localFFTPlan_inverse->numAxisUploads[i]; l++) {
                    VkFFTAxis* axis = &app->localFFTPlan_inverse->axes[i][l];
                    resFFT = VkFFTUpdateBufferSet(app, app->localFFTPlan_inverse, axis, i, l, 1);
                    if (resFFT != VKFFT_SUCCESS) return resFFT;

#if(VKFFT_BACKEND==0)
                    vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
                    vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
                    pfUINT dispatchBlock[3];
                    dispatchBlock[0] = (pfUINT)pfceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[i][0] / (double)axis->axisBlock[0] * app->localFFTPlan_inverse->actualFFTSizePerAxis[i][i] / (double)axis->specializationConstants.fftDim.data.i);
                    dispatchBlock[1] = 1;
                    dispatchBlock[2] = app->configuration.coordinateFeatures * app->configuration.numberKernels;
                    for (int p = 1; p <app->configuration.FFTdim; p++){
                        if (p != i)
                            dispatchBlock[2]*= app->localFFTPlan_inverse->actualFFTSizePerAxis[i][p];
                    }
                    //if (app->configuration.mergeSequencesR2C == 1) dispatchBlock[0] = (pfUINT)pfceil(dispatchBlock[0] / 2.0);
                    resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
                    if (resFFT != VKFFT_SUCCESS) return resFFT;
                    printDebugInformation(app, axis);
                    resFFT = VkFFTSync(app);
                    if (resFFT != VKFFT_SUCCESS) return resFFT;
                }
            }
            if ((app->localFFTPlan_inverse->bigSequenceEvenR2C)&&(i==1)) {
                //app->configuration.size[0] /= 2;
                VkFFTAxis* axis = &app->localFFTPlan_inverse->R2Cdecomposition;
                resFFT = VkFFTUpdateBufferSetR2CMultiUploadDecomposition(app, app->localFFTPlan_inverse, axis, 0, 0, 1);
                if (resFFT != VKFFT_SUCCESS) return resFFT;

#if(VKFFT_BACKEND==0)
                vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
                vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
                pfUINT dispatchBlock[3];

                dispatchBlock[0] = (app->configuration.size[0] / 2 + 1);
                for (int p = 1; p <app->configuration.FFTdim; p++){
                    dispatchBlock[0] *= app->configuration.size[p];
                }
                dispatchBlock[0] = (pfUINT)pfceil(dispatchBlock[0] / (double)(2 * axis->axisBlock[0]));
                
                
                dispatchBlock[1] = 1;
                dispatchBlock[2] = app->configuration.coordinateFeatures * app->configuration.numberKernels;
                resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
                if (resFFT != VKFFT_SUCCESS) return resFFT;
                printDebugInformation(app, axis);

                resFFT = VkFFTSync(app);
                if (resFFT != VKFFT_SUCCESS) return resFFT;
            }
            
            for (pfINT l = 0; l < (pfINT)app->localFFTPlan_inverse->numAxisUploads[i-1]; l++) {
                VkFFTAxis* axis = &app->localFFTPlan_inverse->axes[i-1][l];
                resFFT = VkFFTUpdateBufferSet(app, app->localFFTPlan_inverse, axis, i-1, l, 1);
                if (resFFT != VKFFT_SUCCESS) return resFFT;

#if(VKFFT_BACKEND==0)
                vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
                vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
                pfUINT dispatchBlock[3];
                if (i==1){
                    if (l == 0) {
                        if (app->localFFTPlan_inverse->numAxisUploads[0] > 2) {
                            dispatchBlock[0] = (pfUINT)pfceil((pfUINT)pfceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)axis->axisBlock[1]) / (double)app->localFFTPlan_inverse->axisSplit[0][1]) * app->localFFTPlan_inverse->axisSplit[0][1];
                            dispatchBlock[1] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1];
                        }
                        else {
                            if (app->localFFTPlan_inverse->numAxisUploads[0] > 1) {
                                dispatchBlock[0] = (pfUINT)pfceil((pfUINT)pfceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)axis->axisBlock[1]));
                                dispatchBlock[1] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1];
                            }
                            else {
                                dispatchBlock[0] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i;
                                dispatchBlock[1] = (pfUINT)pfceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1] / (double)axis->axisBlock[1]);
                            }
                        }
                    }
                    else {
                        dispatchBlock[0] = (pfUINT)pfceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)axis->axisBlock[0]);
                        dispatchBlock[1] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1];
                    }
                    dispatchBlock[2] = app->configuration.coordinateFeatures * app->configuration.numberKernels;
                    for (int p = 2; p <app->configuration.FFTdim; p++){
                        dispatchBlock[2]*= app->localFFTPlan_inverse->actualFFTSizePerAxis[i-1][p];
                    }
                    if (axis->specializationConstants.mergeSequencesR2C == 1) dispatchBlock[1] = (pfUINT)pfceil(dispatchBlock[1] / 2.0);
                }else{
                    dispatchBlock[0] = (pfUINT)pfceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[i-1][0] / (double)axis->axisBlock[0] * app->localFFTPlan_inverse->actualFFTSizePerAxis[i-1][i-1] / (double)axis->specializationConstants.fftDim.data.i);
                    dispatchBlock[1] = 1;
                    dispatchBlock[2] = app->configuration.coordinateFeatures * app->configuration.numberKernels;
                    for (int p = 1; p <app->configuration.FFTdim; p++){
                        if (p != (i-1))
                            dispatchBlock[2]*= app->localFFTPlan_inverse->actualFFTSizePerAxis[i-1][p];
                    }
                }
                //if (app->configuration.mergeSequencesR2C == 1) dispatchBlock[0] = (pfUINT)pfceil(dispatchBlock[0] / 2.0);
                //if (app->configuration.performZeropadding[2]) dispatchBlock[2] = (pfUINT)pfceil(dispatchBlock[2] / 2.0);
                resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
                if (resFFT != VKFFT_SUCCESS) return resFFT;
                printDebugInformation(app, axis);
                resFFT = VkFFTSync(app);
                if (resFFT != VKFFT_SUCCESS) return resFFT;
            }

        }
        
        if (app->configuration.FFTdim == 1) {
            for (pfINT l = (pfINT)1; l < (pfINT)app->localFFTPlan_inverse->numAxisUploads[0]; l++) {
                VkFFTAxis* axis = &app->localFFTPlan_inverse->axes[0][l];
                resFFT = VkFFTUpdateBufferSet(app, app->localFFTPlan_inverse, axis, 0, l, 1);
                if (resFFT != VKFFT_SUCCESS) return resFFT;

#if(VKFFT_BACKEND==0)
                vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
                vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
                pfUINT dispatchBlock[3];
                dispatchBlock[0] = (pfUINT)pfceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / (double)axis->axisBlock[0] * app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1] / (double)axis->specializationConstants.fftDim.data.i);
                dispatchBlock[1] = 1;
                dispatchBlock[2] = app->configuration.coordinateFeatures * app->configuration.numberKernels;
                
                //if (app->configuration.mergeSequencesR2C == 1) dispatchBlock[0] = (pfUINT)pfceil(dispatchBlock[0] / 2.0);
                //if (app->configuration.performZeropadding[2]) dispatchBlock[2] = (pfUINT)pfceil(dispatchBlock[2] / 2.0);
                resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
                if (resFFT != VKFFT_SUCCESS) return resFFT;
                printDebugInformation(app, axis);
                resFFT = VkFFTSync(app);
                if (resFFT != VKFFT_SUCCESS) return resFFT;
            }
        }
    }

    if (inverse == 1) {
        //we start from axis N and go back to axis 0
        
        for (int i = (int)app->configuration.FFTdim-1; i > 0; i--){
            if (!app->configuration.omitDimension[i]) {
                for (pfINT l = (pfINT)app->localFFTPlan_inverse->numAxisUploads[i] - 1; l >= 0; l--) {
                    //if ((!app->configuration.reorderFourStep) && (!app->useBluesteinFFT[2])) l = app->localFFTPlan_inverse->numAxisUploads[2] - 1 - l;
                    VkFFTAxis* axis = &app->localFFTPlan_inverse->axes[i][l];
                    resFFT = VkFFTUpdateBufferSet(app, app->localFFTPlan_inverse, axis, i, l, 1);
                    if (resFFT != VKFFT_SUCCESS) return resFFT;

#if(VKFFT_BACKEND==0)
                    vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
                    vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
                    pfUINT dispatchBlock[3];
                    dispatchBlock[0] = (pfUINT)pfceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[i][0]  / (double)axis->axisBlock[0] * app->localFFTPlan_inverse->actualFFTSizePerAxis[i][i] / (double)axis->specializationConstants.fftDim.data.i);
                    dispatchBlock[1] = 1;
                    dispatchBlock[2] = app->configuration.coordinateFeatures * app->configuration.numberBatches;
                    for (int p = 1; p <app->configuration.FFTdim; p++){
                        if (p != i)
                            dispatchBlock[2]*= app->localFFTPlan_inverse->actualFFTSizePerAxis[i][p];
                    }
                    //if (app->configuration.performZeropaddingInverse[0]) dispatchBlock[0] = (pfUINT)pfceil(dispatchBlock[0] / 2.0);
                    //if (app->configuration.performZeropaddingInverse[1]) dispatchBlock[1] = (pfUINT)pfceil(dispatchBlock[1] / 2.0);

                    //if (app->configuration.mergeSequencesR2C == 1) dispatchBlock[0] = (pfUINT)pfceil(dispatchBlock[0] / 2.0);
                    resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
                    if (resFFT != VKFFT_SUCCESS) return resFFT;
                    printDebugInformation(app, axis);
                    resFFT = VkFFTSync(app);
                    if (resFFT != VKFFT_SUCCESS) return resFFT;
                    //if ((!app->configuration.reorderFourStep) && (!app->useBluesteinFFT[2])) l = app->localFFTPlan_inverse->numAxisUploads[2] - 1 - l;
                }
                if (app->useBluesteinFFT[i] && (app->localFFTPlan_inverse->numAxisUploads[i] > 1)) {
                    for (pfINT l = 1; l < (pfINT)app->localFFTPlan_inverse->numAxisUploads[i]; l++) {
                        VkFFTAxis* axis = &app->localFFTPlan_inverse->inverseBluesteinAxes[i][l];
                        resFFT = VkFFTUpdateBufferSet(app, app->localFFTPlan_inverse, axis, i, l, 1);
                        if (resFFT != VKFFT_SUCCESS) return resFFT;
#if(VKFFT_BACKEND==0)
                        vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
                        vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
                        pfUINT dispatchBlock[3];
                        dispatchBlock[0] = (pfUINT)pfceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[i][0]  / (double)axis->axisBlock[0] * app->localFFTPlan_inverse->actualFFTSizePerAxis[i][i] / (double)axis->specializationConstants.fftDim.data.i);
                        dispatchBlock[1] = 1;
                        dispatchBlock[2] = app->configuration.coordinateFeatures * app->configuration.numberBatches;
                        for (int p = 1; p <app->configuration.FFTdim; p++){
                            if (p != i)
                                dispatchBlock[2]*= app->localFFTPlan_inverse->actualFFTSizePerAxis[i][p];
                        }
                        //if (app->configuration.performZeropadding[1]) dispatchBlock[1] = (pfUINT)pfceil(dispatchBlock[1] / 2.0);
                        //if (app->configuration.performZeropadding[2]) dispatchBlock[2] = (pfUINT)pfceil(dispatchBlock[2] / 2.0);
                        resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
                        if (resFFT != VKFFT_SUCCESS) return resFFT;
                        printDebugInformation(app, axis);
                        resFFT = VkFFTSync(app);
                        if (resFFT != VKFFT_SUCCESS) return resFFT;
                    }
                }
            }
        }
        if (!app->configuration.omitDimension[0]) {
            if (app->localFFTPlan_inverse->bigSequenceEvenR2C) {
                //app->configuration.size[0] /= 2;
                VkFFTAxis* axis = &app->localFFTPlan_inverse->R2Cdecomposition;
                resFFT = VkFFTUpdateBufferSetR2CMultiUploadDecomposition(app, app->localFFTPlan_inverse, axis, 0, 0, 1);
                if (resFFT != VKFFT_SUCCESS) return resFFT;

#if(VKFFT_BACKEND==0)
                vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
                vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
                pfUINT dispatchBlock[3];

                dispatchBlock[0] = (app->configuration.size[0] / 2 + 1);
                for (int p = 1; p <app->configuration.FFTdim; p++){
                    dispatchBlock[0] *= app->configuration.size[p];
                }
                dispatchBlock[0] = (pfUINT)pfceil(dispatchBlock[0] / (double)(2 * axis->axisBlock[0]));
                
                
                dispatchBlock[1] = 1;
                dispatchBlock[2] = app->configuration.coordinateFeatures * axis->specializationConstants.numBatches.data.i;
                
                resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
                if (resFFT != VKFFT_SUCCESS) return resFFT;
                printDebugInformation(app, axis);

                resFFT = VkFFTSync(app);
                if (resFFT != VKFFT_SUCCESS) return resFFT;
            }
            //FFT axis 0
            for (pfINT l = (pfINT)app->localFFTPlan_inverse->numAxisUploads[0] - 1; l >= 0; l--) {
                //if ((!app->configuration.reorderFourStep) && (!app->useBluesteinFFT[0])) l = app->localFFTPlan_inverse->numAxisUploads[0] - 1 - l;
                VkFFTAxis* axis = &app->localFFTPlan_inverse->axes[0][l];
                resFFT = VkFFTUpdateBufferSet(app, app->localFFTPlan_inverse, axis, 0, l, 1);
                if (resFFT != VKFFT_SUCCESS) return resFFT;
#if(VKFFT_BACKEND==0)
                vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
                vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
                pfUINT dispatchBlock[3];
                if (l == 0) {
                    if (app->localFFTPlan_inverse->numAxisUploads[0] > 2) {
                        dispatchBlock[0] = (pfUINT)pfceil((pfUINT)pfceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)axis->axisBlock[1]) / (double)app->localFFTPlan_inverse->axisSplit[0][1]) * app->localFFTPlan_inverse->axisSplit[0][1];
                        dispatchBlock[1] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1];
                    }
                    else {
                        if (app->localFFTPlan_inverse->numAxisUploads[0] > 1) {
                            dispatchBlock[0] = (pfUINT)pfceil((pfUINT)pfceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)axis->axisBlock[1]));
                            dispatchBlock[1] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1];
                        }
                        else {
                            dispatchBlock[0] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i;
                            dispatchBlock[1] = (pfUINT)pfceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1] / (double)axis->axisBlock[1]);
                        }
                    }
                }
                else {
                    dispatchBlock[0] = (pfUINT)pfceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)axis->axisBlock[0]);
                    dispatchBlock[1] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1];
                }
                dispatchBlock[2] = app->configuration.coordinateFeatures * app->configuration.numberBatches;
                for (int p = 2; p <app->configuration.FFTdim; p++){
                    dispatchBlock[2]*= app->localFFTPlan_inverse->actualFFTSizePerAxis[0][p];
                }
                if (axis->specializationConstants.mergeSequencesR2C == 1) dispatchBlock[1] = (pfUINT)pfceil(dispatchBlock[1] / 2.0);
                //if (app->configuration.performZeropadding[1]) dispatchBlock[1] = (pfUINT)pfceil(dispatchBlock[1] / 2.0);
                //if (app->configuration.performZeropadding[2]) dispatchBlock[2] = (pfUINT)pfceil(dispatchBlock[2] / 2.0);
                resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
                if (resFFT != VKFFT_SUCCESS) return resFFT;
                printDebugInformation(app, axis);
                //if ((!app->configuration.reorderFourStep) && (!app->useBluesteinFFT[0])) l = app->localFFTPlan_inverse->numAxisUploads[0] - 1 - l;
                resFFT = VkFFTSync(app);
                if (resFFT != VKFFT_SUCCESS) return resFFT;
            }
            if (app->useBluesteinFFT[0] && (app->localFFTPlan_inverse->numAxisUploads[0] > 1)) {
                for (pfINT l = 1; l < (pfINT)app->localFFTPlan_inverse->numAxisUploads[0]; l++) {
                    VkFFTAxis* axis = &app->localFFTPlan_inverse->inverseBluesteinAxes[0][l];
                    resFFT = VkFFTUpdateBufferSet(app, app->localFFTPlan_inverse, axis, 0, l, 1);
                    if (resFFT != VKFFT_SUCCESS) return resFFT;

#if(VKFFT_BACKEND==0)
                    vkCmdBindPipeline(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipeline);
                    vkCmdBindDescriptorSets(app->configuration.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, axis->pipelineLayout, 0, 1, &axis->descriptorSet, 0, 0);
#endif
                    pfUINT dispatchBlock[3];
                    if (l == 0) {
                        if (app->localFFTPlan_inverse->numAxisUploads[0] > 2) {
                            dispatchBlock[0] = (pfUINT)pfceil((pfUINT)pfceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)axis->axisBlock[1]) / (double)app->localFFTPlan_inverse->axisSplit[0][1]) * app->localFFTPlan_inverse->axisSplit[0][1];
                            dispatchBlock[1] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1];
                        }
                        else {
                            if (app->localFFTPlan_inverse->numAxisUploads[0] > 1) {
                                dispatchBlock[0] = (pfUINT)pfceil((pfUINT)pfceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)axis->axisBlock[1]));
                                dispatchBlock[1] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1];
                            }
                            else {
                                dispatchBlock[0] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i;
                                dispatchBlock[1] = (pfUINT)pfceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1] / (double)axis->axisBlock[1]);
                            }
                        }
                    }
                    else {
                        dispatchBlock[0] = (pfUINT)pfceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim.data.i / (double)axis->axisBlock[0]);
                        dispatchBlock[1] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1];
                    }
                    dispatchBlock[2] = app->configuration.coordinateFeatures * app->configuration.numberBatches;
                    for (int p = 2; p <app->configuration.FFTdim; p++){
                        dispatchBlock[2]*= app->localFFTPlan_inverse->actualFFTSizePerAxis[0][p];
                    }
                    if (axis->specializationConstants.mergeSequencesR2C == 1) dispatchBlock[1] = (pfUINT)pfceil(dispatchBlock[1] / 2.0);
                    //if (app->configuration.performZeropadding[1]) dispatchBlock[1] = (pfUINT)pfceil(dispatchBlock[1] / 2.0);
                    //if (app->configuration.performZeropadding[2]) dispatchBlock[2] = (pfUINT)pfceil(dispatchBlock[2] / 2.0);
                    resFFT = VkFFT_DispatchPlan(app, axis, dispatchBlock);
                    if (resFFT != VKFFT_SUCCESS) return resFFT;
                    printDebugInformation(app, axis);
                    resFFT = VkFFTSync(app);
                    if (resFFT != VKFFT_SUCCESS) return resFFT;
                }
            }
        }
        //if (app->localFFTPlan_inverse->bigSequenceEvenR2C) app->configuration.size[0] *= 2;

    }
    return resFFT;
}


#endif
