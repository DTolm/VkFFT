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
#ifndef VKFFT_SCHEDULER_H
#define VKFFT_SCHEDULER_H
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
static inline VkFFTResult VkFFTGetRegistersPerThreadQuad(VkFFTApplication* app, int fft_length, int extraSharedMemoryForPow2, pfUINT max_rhs, int useRader, int* loc_multipliers, int* registers_per_thread_per_radix, int* registers_per_thread, int* min_registers_per_thread, int* isGoodSequence) {
	for (int i = 0; i < 33; i++) {
		registers_per_thread_per_radix[i] = 0;
	}
	registers_per_thread[0] = 0;
	min_registers_per_thread[0] = 10000000;

	if (loc_multipliers[2] > 0) {
		if (loc_multipliers[3] > 0) {
			if (loc_multipliers[5] > 0) {
				if (loc_multipliers[7] > 0) {
					registers_per_thread_per_radix[2] = 6;
					registers_per_thread_per_radix[3] = 6;
					registers_per_thread_per_radix[5] = 5;
					registers_per_thread_per_radix[7] = 7;
					registers_per_thread_per_radix[11] = 0;
					registers_per_thread_per_radix[13] = 0;
				}
				else
				{
					registers_per_thread_per_radix[2] = 6;
					registers_per_thread_per_radix[3] = 6;
					registers_per_thread_per_radix[5] = 5;
					registers_per_thread_per_radix[7] = 0;
					registers_per_thread_per_radix[11] = 0;
					registers_per_thread_per_radix[13] = 0;
				}
			}
			else
			{
				if (loc_multipliers[7] > 0) {
					switch (loc_multipliers[2]) {
					case 1:
						registers_per_thread_per_radix[2] = 6;
						registers_per_thread_per_radix[3] = 6;
						registers_per_thread_per_radix[5] = 0;
						registers_per_thread_per_radix[7] = 7;
						registers_per_thread_per_radix[11] = 0;
						registers_per_thread_per_radix[13] = 0;
						break;
					case 2:
						registers_per_thread_per_radix[2] = 6;
						registers_per_thread_per_radix[3] = 6;
						registers_per_thread_per_radix[5] = 0;
						registers_per_thread_per_radix[7] = 7;
						registers_per_thread_per_radix[11] = 0;
						registers_per_thread_per_radix[13] = 0;
						break;
					default:
						registers_per_thread_per_radix[2] = 8;
						registers_per_thread_per_radix[3] = 6;
						registers_per_thread_per_radix[5] = 0;
						registers_per_thread_per_radix[7] = 7;
						registers_per_thread_per_radix[11] = 0;
						registers_per_thread_per_radix[13] = 0;
						break;
					}
				}
				else
				{
					registers_per_thread_per_radix[2] = 6;
					registers_per_thread_per_radix[3] = 6;
					registers_per_thread_per_radix[5] = 0;
					registers_per_thread_per_radix[7] = 0;
					registers_per_thread_per_radix[11] = 0;
					registers_per_thread_per_radix[13] = 0;
				}
			}
		}
		else {
			if (loc_multipliers[5] > 0) {
				if (loc_multipliers[7] > 0) {
					switch (loc_multipliers[2]) {
					case 1:
						registers_per_thread_per_radix[2] = 6;
						registers_per_thread_per_radix[3] = 0;
						registers_per_thread_per_radix[5] = 5;
						registers_per_thread_per_radix[7] = 7;
						registers_per_thread_per_radix[11] = 0;
						registers_per_thread_per_radix[13] = 0;
						break;
					case 2:
						registers_per_thread_per_radix[2] = 8;
						registers_per_thread_per_radix[3] = 0;
						registers_per_thread_per_radix[5] = 5;
						registers_per_thread_per_radix[7] = 7;
						registers_per_thread_per_radix[11] = 0;
						registers_per_thread_per_radix[13] = 0;
						break;
					default:
						registers_per_thread_per_radix[2] = 8;
						registers_per_thread_per_radix[3] = 0;
						registers_per_thread_per_radix[5] = 5;
						registers_per_thread_per_radix[7] = 7;
						registers_per_thread_per_radix[11] = 0;
						registers_per_thread_per_radix[13] = 0;
						break;
					}
				}
				else
				{
					registers_per_thread_per_radix[2] = 4;
					registers_per_thread_per_radix[3] = 0;
					registers_per_thread_per_radix[5] = 5;
					registers_per_thread_per_radix[7] = 0;
					registers_per_thread_per_radix[11] = 0;
					registers_per_thread_per_radix[13] = 0;
				}
			}
			else
			{
				if (loc_multipliers[7] > 0) {
					registers_per_thread_per_radix[2] = 8;
					registers_per_thread_per_radix[3] = 0;
					registers_per_thread_per_radix[5] = 0;
					registers_per_thread_per_radix[7] = 7;
					registers_per_thread_per_radix[11] = 0;
					registers_per_thread_per_radix[13] = 0;
				}
				else
				{
					int max_loc_multipliers_pow2 = 0;
					pfUINT active_threads_y = max_rhs / 64; //estimate workbalance across CU (assume we have 64 CU)
					if (active_threads_y == 0) active_threads_y = 1;
					int testMinStages = 10000000;
					int maxRadixMinStages = 1;
					int fixMaxCheckRadix2 = 3;

					for (int i = 1; i <= fixMaxCheckRadix2; i++) {
						int numStages = (int)pfceil(log2(fft_length) / ((double)i));
						if (numStages < testMinStages) {
							testMinStages = numStages;
							maxRadixMinStages = i;
						}
					}
					for (int i = maxRadixMinStages; i >= 1; i--) {
						pfUINT active_threads_x = (active_threads_y * fft_length) / ((int)pow(2, i));
						if (active_threads_x >= 128) {
							max_loc_multipliers_pow2 = i;
							i = 1;
						}

					}
					if (max_loc_multipliers_pow2 < 3) max_loc_multipliers_pow2 = 3;

					int final_loc_multipliers_pow2 = 1;
					int num_stages_min = (int)log2(fft_length);
					for (int i = 2; i <= max_loc_multipliers_pow2; i++) {
						int num_stages = (int)pfceil(((int)log2(fft_length)) / (double)i);
						if (num_stages < num_stages_min) {
							final_loc_multipliers_pow2 = i;
							num_stages_min = num_stages;
						}

					}
					registers_per_thread_per_radix[2] = (loc_multipliers[2] > final_loc_multipliers_pow2) ? (int)pow(2, final_loc_multipliers_pow2) : (int)pow(2, loc_multipliers[2]);
					registers_per_thread_per_radix[2] = (loc_multipliers[2] < 3) ? (int)pow(2, loc_multipliers[2]) : registers_per_thread_per_radix[2];
					registers_per_thread_per_radix[3] = 0;
					registers_per_thread_per_radix[5] = 0;
					registers_per_thread_per_radix[7] = 0;
					registers_per_thread_per_radix[11] = 0;
					registers_per_thread_per_radix[13] = 0;
				}
			}
		}
	}
	else {
		if (loc_multipliers[3] > 0) {
			if (loc_multipliers[5] > 0) {
				if (loc_multipliers[7] > 0) {
					registers_per_thread_per_radix[2] = 0;
					registers_per_thread_per_radix[3] = 6;
					registers_per_thread_per_radix[5] = 5;
					registers_per_thread_per_radix[7] = 7;
					registers_per_thread_per_radix[11] = 0;
					registers_per_thread_per_radix[13] = 0;
				}
				else
				{
					registers_per_thread_per_radix[2] = 0;
					registers_per_thread_per_radix[3] = 3;
					registers_per_thread_per_radix[5] = 5;
					registers_per_thread_per_radix[7] = 0;
					registers_per_thread_per_radix[11] = 0;
					registers_per_thread_per_radix[13] = 0;
				}
			}
			else
			{
				if (loc_multipliers[7] > 0) {
					registers_per_thread_per_radix[2] = 0;
					registers_per_thread_per_radix[3] = 6;
					registers_per_thread_per_radix[5] = 0;
					registers_per_thread_per_radix[7] = 7;
					registers_per_thread_per_radix[11] = 0;
					registers_per_thread_per_radix[13] = 0;
				}
				else
				{
					if (loc_multipliers[3] == 1) {
						registers_per_thread_per_radix[2] = 0;
						registers_per_thread_per_radix[3] = 3;
						registers_per_thread_per_radix[5] = 0;
						registers_per_thread_per_radix[7] = 0;
						registers_per_thread_per_radix[11] = 0;
						registers_per_thread_per_radix[13] = 0;
					}
					else {
						registers_per_thread_per_radix[2] = 0;
						registers_per_thread_per_radix[3] = 9;
						registers_per_thread_per_radix[5] = 0;
						registers_per_thread_per_radix[7] = 0;
						registers_per_thread_per_radix[11] = 0;
						registers_per_thread_per_radix[13] = 0;
					}
				}
			}
		}
		else {
			if (loc_multipliers[5] > 0) {
				if (loc_multipliers[7] > 0) {
					registers_per_thread_per_radix[2] = 0;
					registers_per_thread_per_radix[3] = 0;
					registers_per_thread_per_radix[5] = 5;
					registers_per_thread_per_radix[7] = 7;
					registers_per_thread_per_radix[11] = 0;
					registers_per_thread_per_radix[13] = 0;
				}
				else
				{
					registers_per_thread_per_radix[2] = 0;
					registers_per_thread_per_radix[3] = 0;
					registers_per_thread_per_radix[5] = 5;
					registers_per_thread_per_radix[7] = 0;
					registers_per_thread_per_radix[11] = 0;
					registers_per_thread_per_radix[13] = 0;
				}
			}
			else
			{
				if (loc_multipliers[7] > 0) {
					registers_per_thread_per_radix[2] = 0;
					registers_per_thread_per_radix[3] = 0;
					registers_per_thread_per_radix[5] = 0;
					registers_per_thread_per_radix[7] = 7;
					registers_per_thread_per_radix[11] = 0;
					registers_per_thread_per_radix[13] = 0;
				}
				else
				{
					min_registers_per_thread[0] = 2;
					registers_per_thread[0] = 2;
					//Rader-only sequence
					//return VKFFT_ERROR_UNSUPPORTED_RADIX;
				}
			}
		}

	}

	registers_per_thread_per_radix[32] = ((registers_per_thread_per_radix[2] % 32) == 0) ? registers_per_thread_per_radix[2] : 0;
	registers_per_thread_per_radix[16] = ((registers_per_thread_per_radix[2] % 16) == 0) ? registers_per_thread_per_radix[2] : 0;
	registers_per_thread_per_radix[8] = ((registers_per_thread_per_radix[2] % 8) == 0) ? registers_per_thread_per_radix[2] : 0;
	registers_per_thread_per_radix[4] = ((registers_per_thread_per_radix[2] % 4) == 0) ? registers_per_thread_per_radix[2] : 0;
	if ((registers_per_thread_per_radix[2] >= 12) && (registers_per_thread_per_radix[3] >= 12)) {
		registers_per_thread_per_radix[12] = (registers_per_thread_per_radix[2] > registers_per_thread_per_radix[3]) ? registers_per_thread_per_radix[3] : registers_per_thread_per_radix[2];
		if ((registers_per_thread_per_radix[12] % 12) != 0) registers_per_thread_per_radix[12] = 0;
	}
	registers_per_thread_per_radix[6] = (registers_per_thread_per_radix[2] > registers_per_thread_per_radix[3]) ? registers_per_thread_per_radix[3] : registers_per_thread_per_radix[2];
	registers_per_thread_per_radix[9] = ((registers_per_thread_per_radix[3] % 9) == 0) ? registers_per_thread_per_radix[3] : 0;
	registers_per_thread_per_radix[10] = (registers_per_thread_per_radix[2] > registers_per_thread_per_radix[5]) ? registers_per_thread_per_radix[5] : registers_per_thread_per_radix[2];
	registers_per_thread_per_radix[14] = (registers_per_thread_per_radix[2] > registers_per_thread_per_radix[7]) ? registers_per_thread_per_radix[7] : registers_per_thread_per_radix[2];
	registers_per_thread_per_radix[15] = (registers_per_thread_per_radix[3] > registers_per_thread_per_radix[5]) ? registers_per_thread_per_radix[5] : registers_per_thread_per_radix[3];

	for (int i = 0; i < 33; i++) {
		if ((registers_per_thread_per_radix[i] != 0) && (registers_per_thread_per_radix[i] < min_registers_per_thread[0])) min_registers_per_thread[0] = registers_per_thread_per_radix[i];
		if ((registers_per_thread_per_radix[i] != 0) && (registers_per_thread_per_radix[i] > registers_per_thread[0])) registers_per_thread[0] = registers_per_thread_per_radix[i];
	}
	if ((registers_per_thread[0] > 16) || (registers_per_thread[0] >= 2 * min_registers_per_thread[0])) isGoodSequence[0] = 0;
	else isGoodSequence[0] = 1;
	return VKFFT_SUCCESS;
}

static inline VkFFTResult VkFFTGetRegistersPerThread(VkFFTApplication* app, int fft_length, int extraSharedMemoryForPow2, pfUINT max_rhs, int useRader, int* loc_multipliers, int* registers_per_thread_per_radix, int* registers_per_thread, int* min_registers_per_thread, int* isGoodSequence) {
	if (app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) {
		VkFFTGetRegistersPerThreadQuad(app, fft_length, extraSharedMemoryForPow2, max_rhs, useRader, loc_multipliers, registers_per_thread_per_radix, registers_per_thread, min_registers_per_thread, isGoodSequence);
		return VKFFT_SUCCESS;
	}
	for (int i = 0; i < 33; i++) {
		registers_per_thread_per_radix[i] = 0;
	}
	registers_per_thread[0] = 0;
	min_registers_per_thread[0] = 10000000;

	if (loc_multipliers[2] > 0) {
		if (loc_multipliers[3] > 0) {
			if (loc_multipliers[5] > 0) {
				if (loc_multipliers[7] > 0) {
					if (loc_multipliers[11] > 0) {
						if (loc_multipliers[13] > 0) {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 14;
								registers_per_thread_per_radix[3] = 15;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								break;
							case 3:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								break;
							default:
								registers_per_thread_per_radix[2] = 16;
								registers_per_thread_per_radix[3] = 12;
								break;
							}
							registers_per_thread_per_radix[5] = 15;
							registers_per_thread_per_radix[7] = 14;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 14;
								registers_per_thread_per_radix[3] = 15;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								break;
							case 3:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								break;
							default:
								registers_per_thread_per_radix[2] = 16;
								registers_per_thread_per_radix[3] = 12;
								break;
							}
							registers_per_thread_per_radix[5] = 15;
							registers_per_thread_per_radix[7] = 14;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 0;
						}
					}
					else {
						if (loc_multipliers[13] > 0) {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 14;
								registers_per_thread_per_radix[3] = 15;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								break;
							case 3:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								break;
							default:
								registers_per_thread_per_radix[2] = 16;
								registers_per_thread_per_radix[3] = 12;
								break;
							}
							registers_per_thread_per_radix[5] = 15;
							registers_per_thread_per_radix[7] = 14;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 13;

						}
						else {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 14;
								registers_per_thread_per_radix[3] = 15;

								break;
							case 2:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								break;
							case 3:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								break;
							default:
								registers_per_thread_per_radix[2] = 16;
								registers_per_thread_per_radix[3] = 12;
								break;
							}
							registers_per_thread_per_radix[5] = 15;
							registers_per_thread_per_radix[7] = 14;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 0;
						}
					}
				}
				else {
					if (loc_multipliers[11] > 0) {
						if (loc_multipliers[13] > 0) {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 15;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								break;
							default:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								break;
							}
							registers_per_thread_per_radix[5] = 10;
							registers_per_thread_per_radix[7] = 0;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 15;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								break;
							default:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								break;
							}
							registers_per_thread_per_radix[5] = 10;
							registers_per_thread_per_radix[7] = 0;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 0;
						}
					}
					else {
						if (loc_multipliers[13] > 0) {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 15;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								break;
							default:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								break;
							}
							registers_per_thread_per_radix[5] = 10;
							registers_per_thread_per_radix[7] = 0;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 6;
								registers_per_thread_per_radix[3] = 6;
								registers_per_thread_per_radix[5] = 5;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								registers_per_thread_per_radix[5] = 10;
								break;
							default:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								registers_per_thread_per_radix[5] = 10;
								break;
							}
							registers_per_thread_per_radix[7] = 0;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 0;

						}
					}
				}
			}
			else
			{
				if (loc_multipliers[7] > 0) {
					if (loc_multipliers[11] > 0) {
						if (loc_multipliers[13] > 0) {
							registers_per_thread_per_radix[2] = 12;
							registers_per_thread_per_radix[3] = 12;
							registers_per_thread_per_radix[5] = 0;
							registers_per_thread_per_radix[7] = 14;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							registers_per_thread_per_radix[2] = 12;
							registers_per_thread_per_radix[3] = 12;
							registers_per_thread_per_radix[5] = 0;
							registers_per_thread_per_radix[7] = 14;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 0;
						}
					}
					else {
						if (loc_multipliers[13] > 0) {
							registers_per_thread_per_radix[2] = 12;
							registers_per_thread_per_radix[3] = 12;
							registers_per_thread_per_radix[5] = 0;
							registers_per_thread_per_radix[7] = 14;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 6;
								registers_per_thread_per_radix[3] = 6;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 7;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 6;
								registers_per_thread_per_radix[3] = 6;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 7;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
								break;
							default:
								registers_per_thread_per_radix[2] = 8;
								registers_per_thread_per_radix[3] = 6;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 7;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
								break;
							}
						}
					}
				}
				else {
					if (loc_multipliers[11] > 0) {
						if (loc_multipliers[13] > 0) {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 6;
								registers_per_thread_per_radix[3] = 6;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							default:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							}
						}
						else {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 6;
								registers_per_thread_per_radix[3] = 6;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								break;
							default:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								break;
							}
						}
					}
					else {
						if (loc_multipliers[13] > 0) {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 6;
								registers_per_thread_per_radix[3] = 6;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							default:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							}
						}
						else {
							if (loc_multipliers[2] == loc_multipliers[3]) {
								registers_per_thread_per_radix[2] = 6;
								registers_per_thread_per_radix[3] = 6;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
							}
							else {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread_per_radix[2] = 6;
									registers_per_thread_per_radix[3] = 6;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									break;
								case 2:
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									break;
								default:
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									break;
								}
							}
						}
					}
				}
			}
		}
		else {
			if (loc_multipliers[5] > 0) {
				if (loc_multipliers[7] > 0) {
					if (loc_multipliers[11] > 0) {
						if (loc_multipliers[13] > 0) {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							case 3:
								registers_per_thread_per_radix[2] = 8;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							default:
								registers_per_thread_per_radix[2] = 16;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							}
						}
						else {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								break;
							case 3:
								registers_per_thread_per_radix[2] = 8;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								break;
							default:
								registers_per_thread_per_radix[2] = 16;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								break;
							}
						}
					}
					else {
						if (loc_multipliers[13] > 0) {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							case 3:
								registers_per_thread_per_radix[2] = 8;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							default:
								registers_per_thread_per_radix[2] = 16;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							}
						}
						else {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 7;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 7;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
								break;
							default:
								registers_per_thread_per_radix[2] = 8;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 7;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
								break;
							}
						}
					}
				}
				else {
					if (loc_multipliers[11] > 0) {
						if (loc_multipliers[13] > 0) {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							default:
								registers_per_thread_per_radix[2] = 8;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							}
						}
						else {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								break;
							default:
								registers_per_thread_per_radix[2] = 8;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								break;
							}
						}
					}
					else {
						if (loc_multipliers[13] > 0) {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							default:
								registers_per_thread_per_radix[2] = 8;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							}
						}
						else {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
								break;
							default:
								registers_per_thread_per_radix[2] = 8;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
								break;
							}
						}
					}
				}
			}
			else
			{
				if (loc_multipliers[7] > 0) {
					if (loc_multipliers[11] > 0) {
						if (loc_multipliers[13] > 0) {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 14;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 14;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							case 3:
								registers_per_thread_per_radix[2] = 8;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							default:
								registers_per_thread_per_radix[2] = 16;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							}
						}
						else {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 14;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 14;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								break;
							case 3:
								registers_per_thread_per_radix[2] = 8;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								break;
							default:
								registers_per_thread_per_radix[2] = 16;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								break;
							}
						}
					}
					else {
						if (loc_multipliers[13] > 0) {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 14;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 14;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							case 3:
								registers_per_thread_per_radix[2] = 8;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							default:
								registers_per_thread_per_radix[2] = 16;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							}
						}
						else {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 14;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 14;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
								break;
							default:
								registers_per_thread_per_radix[2] = 8;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 7;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
								break;
							}
						}
					}
				}
				else {
					if (loc_multipliers[11] > 0) {
						if (loc_multipliers[13] > 0) {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							default:
								registers_per_thread_per_radix[2] = 8;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							}
						}
						else {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								break;
							default:
								registers_per_thread_per_radix[2] = 8;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								break;
							}
						}
					}
					else {
						if (loc_multipliers[13] > 0) {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							default:
								registers_per_thread_per_radix[2] = 8;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							}
						}
						else {
							int max_loc_multipliers_pow2 = 0;
							pfUINT active_threads_y = max_rhs / 64; //estimate workbalance across CU (assume we have 64 CU)
							if (active_threads_y == 0) active_threads_y = 1;
							int testMinStages = 10000000;
							int maxRadixMinStages = 1;
							int fixMaxCheckRadix2 = 3;
#if(VKFFT_BACKEND==1)
							fixMaxCheckRadix2 = (((fft_length >= 1024) || (fft_length == 256)) && (extraSharedMemoryForPow2) && (!useRader)) ? 5 : 3;
#endif
							for (int i = 1; i <= fixMaxCheckRadix2; i++) {
								int numStages = (int)pfceil(log2(fft_length) / ((double)i));
								if (numStages < testMinStages) {
									testMinStages = numStages;
									maxRadixMinStages = i;
								}
							}
							for (int i = maxRadixMinStages; i >= 1; i--) {
								pfUINT active_threads_x = (active_threads_y * fft_length) / ((int)pow(2, i));
								if (active_threads_x >= 128) {
									max_loc_multipliers_pow2 = i;
									i = 1;
								}

							}
							if (max_loc_multipliers_pow2 < 3) max_loc_multipliers_pow2 = 3;

							int final_loc_multipliers_pow2 = 1;
							int num_stages_min = (int)log2(fft_length);
							for (int i = 2; i <= max_loc_multipliers_pow2; i++) {
								int num_stages = (int)pfceil(((int)log2(fft_length)) / (double)i);
								if (num_stages < num_stages_min) {
									final_loc_multipliers_pow2 = i;
									num_stages_min = num_stages;
								}

							}
							registers_per_thread_per_radix[2] = (loc_multipliers[2] > final_loc_multipliers_pow2) ? (int)pow(2, final_loc_multipliers_pow2) : (int)pow(2, loc_multipliers[2]);
							registers_per_thread_per_radix[2] = (loc_multipliers[2] < 3) ? (int)pow(2, loc_multipliers[2]) : registers_per_thread_per_radix[2];
							registers_per_thread_per_radix[3] = 0;
							registers_per_thread_per_radix[5] = 0;
							registers_per_thread_per_radix[7] = 0;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 0;
						}
					}
				}
			}
		}
	}
	else {
		if (loc_multipliers[3] > 0) {
			if (loc_multipliers[5] > 0) {
				if (loc_multipliers[7] > 0) {
					if (loc_multipliers[11] > 0) {
						if (loc_multipliers[13] > 0) {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 15;
							registers_per_thread_per_radix[5] = 15;
							registers_per_thread_per_radix[7] = 14;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 15;
							registers_per_thread_per_radix[5] = 15;
							registers_per_thread_per_radix[7] = 14;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 0;
						}
					}
					else {
						if (loc_multipliers[13] > 0) {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 15;
							registers_per_thread_per_radix[5] = 15;
							registers_per_thread_per_radix[7] = 14;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 15;
							registers_per_thread_per_radix[5] = 15;
							registers_per_thread_per_radix[7] = 14;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 0;
						}
					}
				}
				else {
					if (loc_multipliers[11] > 0) {
						if (loc_multipliers[13] > 0) {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 15;
							registers_per_thread_per_radix[5] = 15;
							registers_per_thread_per_radix[7] = 0;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 15;
							registers_per_thread_per_radix[5] = 15;
							registers_per_thread_per_radix[7] = 0;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 0;
						}
					}
					else {
						if (loc_multipliers[13] > 0) {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 15;
							registers_per_thread_per_radix[5] = 15;
							registers_per_thread_per_radix[7] = 0;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 15;
							registers_per_thread_per_radix[5] = 15;
							registers_per_thread_per_radix[7] = 0;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 0;
						}
					}
				}
			}
			else
			{
				if (loc_multipliers[7] > 0) {
					if (loc_multipliers[3] == 1) {
						if (loc_multipliers[11] > 0) {
							if (loc_multipliers[13] > 0) {
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 12;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
							}
							else {
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 12;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
							}
						}
						else {
							if (loc_multipliers[13] > 0) {
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 12;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
							}
							else {
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 6;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 7;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
							}
						}
					}
					else {
						if (loc_multipliers[11] > 0) {
							if (loc_multipliers[13] > 0) {
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 9;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 7;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
							}
							else {
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 9;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 7;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
							}
						}
						else {
							if (loc_multipliers[13] > 0) {
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 9;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 7;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
							}
							else {
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 9;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 7;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
							}
						}
					}
				}
				else {
					if (loc_multipliers[3] == 1) {
						if (loc_multipliers[11] > 0) {
							if (loc_multipliers[13] > 0) {
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 12;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
							}
							else {
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 9;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
							}
						}
						else {
							if (loc_multipliers[13] > 0) {
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 12;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
							}
							else {
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 3;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
							}
						}
					}
					else {
						if (loc_multipliers[11] > 0) {
							if (loc_multipliers[13] > 0) {
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 9;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
							}
							else {
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 9;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
							}
						}
						else {
							if (loc_multipliers[13] > 0) {
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 9;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
							}
							else {
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 9;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
							}
						}
					}
				}
			}
		}
		else {
			if (loc_multipliers[5] > 0) {
				if (loc_multipliers[7] > 0) {
					if (loc_multipliers[11] > 0) {
						if (loc_multipliers[13] > 0) {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 0;
							registers_per_thread_per_radix[5] = 5;
							registers_per_thread_per_radix[7] = 7;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 0;
							registers_per_thread_per_radix[5] = 5;
							registers_per_thread_per_radix[7] = 7;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 0;
						}
					}
					else {
						if (loc_multipliers[13] > 0) {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 0;
							registers_per_thread_per_radix[5] = 5;
							registers_per_thread_per_radix[7] = 7;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 0;
							registers_per_thread_per_radix[5] = 5;
							registers_per_thread_per_radix[7] = 7;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 0;
						}
					}
				}
				else {
					if (loc_multipliers[11] > 0) {
						if (loc_multipliers[13] > 0) {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 0;
							registers_per_thread_per_radix[5] = 5;
							registers_per_thread_per_radix[7] = 0;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 0;
							registers_per_thread_per_radix[5] = 5;
							registers_per_thread_per_radix[7] = 0;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 0;
						}
					}
					else {
						if (loc_multipliers[13] > 0) {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 0;
							registers_per_thread_per_radix[5] = 5;
							registers_per_thread_per_radix[7] = 0;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 0;
							registers_per_thread_per_radix[5] = 5;
							registers_per_thread_per_radix[7] = 0;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 0;
						}
					}
				}
			}
			else
			{
				if (loc_multipliers[7] > 0) {
					if (loc_multipliers[11] > 0) {
						if (loc_multipliers[13] > 0) {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 0;
							registers_per_thread_per_radix[5] = 0;
							registers_per_thread_per_radix[7] = 7;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 0;
							registers_per_thread_per_radix[5] = 0;
							registers_per_thread_per_radix[7] = 7;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 0;
						}
					}
					else {
						if (loc_multipliers[13] > 0) {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 0;
							registers_per_thread_per_radix[5] = 0;
							registers_per_thread_per_radix[7] = 7;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 0;
							registers_per_thread_per_radix[5] = 0;
							registers_per_thread_per_radix[7] = 7;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 0;
						}
					}
				}
				else {
					if (loc_multipliers[11] > 0) {
						if (loc_multipliers[13] > 0) {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 0;
							registers_per_thread_per_radix[5] = 0;
							registers_per_thread_per_radix[7] = 0;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 0;
							registers_per_thread_per_radix[5] = 0;
							registers_per_thread_per_radix[7] = 0;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 0;
						}
					}
					else {
						if (loc_multipliers[13] > 0) {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 0;
							registers_per_thread_per_radix[5] = 0;
							registers_per_thread_per_radix[7] = 0;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							min_registers_per_thread[0] = 2;
							registers_per_thread[0] = 2;
							//Rader-only sequence
							//return VKFFT_ERROR_UNSUPPORTED_RADIX;
						}
					}
				}
			}
		}

	}

	registers_per_thread_per_radix[32] = ((registers_per_thread_per_radix[2] % 32) == 0) ? registers_per_thread_per_radix[2] : 0;
	registers_per_thread_per_radix[16] = ((registers_per_thread_per_radix[2] % 16) == 0) ? registers_per_thread_per_radix[2] : 0;
	registers_per_thread_per_radix[8] = ((registers_per_thread_per_radix[2] % 8) == 0) ? registers_per_thread_per_radix[2] : 0;
	registers_per_thread_per_radix[4] = ((registers_per_thread_per_radix[2] % 4) == 0) ? registers_per_thread_per_radix[2] : 0;
	if ((registers_per_thread_per_radix[2] >= 12) && (registers_per_thread_per_radix[3] >= 12)) {
		registers_per_thread_per_radix[12] = (registers_per_thread_per_radix[2] > registers_per_thread_per_radix[3]) ? registers_per_thread_per_radix[3] : registers_per_thread_per_radix[2];
		if ((registers_per_thread_per_radix[12] % 12) != 0) registers_per_thread_per_radix[12] = 0;
	}
	registers_per_thread_per_radix[6] = (registers_per_thread_per_radix[2] > registers_per_thread_per_radix[3]) ? registers_per_thread_per_radix[3] : registers_per_thread_per_radix[2];
	registers_per_thread_per_radix[9] = ((registers_per_thread_per_radix[3] % 9) == 0) ? registers_per_thread_per_radix[3] : 0;
	registers_per_thread_per_radix[10] = (registers_per_thread_per_radix[2] > registers_per_thread_per_radix[5]) ? registers_per_thread_per_radix[5] : registers_per_thread_per_radix[2];
	registers_per_thread_per_radix[14] = (registers_per_thread_per_radix[2] > registers_per_thread_per_radix[7]) ? registers_per_thread_per_radix[7] : registers_per_thread_per_radix[2];
	registers_per_thread_per_radix[15] = (registers_per_thread_per_radix[3] > registers_per_thread_per_radix[5]) ? registers_per_thread_per_radix[5] : registers_per_thread_per_radix[3];

	for (int i = 0; i < 33; i++) {
		if ((registers_per_thread_per_radix[i] != 0) && (registers_per_thread_per_radix[i] < min_registers_per_thread[0])) min_registers_per_thread[0] = registers_per_thread_per_radix[i];
		if ((registers_per_thread_per_radix[i] != 0) && (registers_per_thread_per_radix[i] > registers_per_thread[0])) registers_per_thread[0] = registers_per_thread_per_radix[i];
	}
	if ((registers_per_thread[0] > 16) || (registers_per_thread[0] >= 2 * min_registers_per_thread[0])) isGoodSequence[0] = 0;
	else isGoodSequence[0] = 1;
	return VKFFT_SUCCESS;
}

static inline VkFFTResult VkFFTGetRegistersPerThreadOptimizeShared(int fft_length, int* registers_per_thread_per_radix, int* registers_per_thread, int* min_registers_per_thread) {
	//try to split sequence in supported radix to optimize sm usage
	int numStages = 20;
	int fft_length_copy;
	int stages[20];
	int k = 0;
	for (int i = 0; i < 33; i++) {
		registers_per_thread_per_radix[i] = 0;
	}
	registers_per_thread[0] = 0;
	min_registers_per_thread[0] = 10000000;

	for (int i = 1; i < numStages; i++) {
		fft_length_copy = fft_length;
		int min_comb_radix = (int)floor(pow(fft_length_copy, 1.0 / i));
		if (min_comb_radix <= 16) {
			for (int j = 0; j < 20; j++) {
				stages[j] = 0;
			}
			k = 0;
			for (int j = min_comb_radix; j <= 16; j++) {
				if (k < i) {
					if ((fft_length_copy % j) == 0) {
						fft_length_copy /= j;
						min_comb_radix = (int)floor(pow(fft_length_copy, 1.0 / (i - k - 1)));

						stages[k] = j;
						j = min_comb_radix - 1;
						k++;
					}
				}
			}
			if ((fft_length_copy == 1) && (k == i)) break;
		}
	}
	for (int i = 0; i < k; i++) {
		for (int j = 2; j <= stages[i]; j++) {
			if ((stages[i] % j) == 0) {
				if (registers_per_thread_per_radix[j] < stages[i])
					registers_per_thread_per_radix[j] = stages[i];
			}
		}
	}
	for (int i = 0; i < 33; i++) {
		if ((registers_per_thread_per_radix[i] != 0) && (registers_per_thread_per_radix[i] > registers_per_thread[0])) registers_per_thread[0] = registers_per_thread_per_radix[i];
	}

	for (int i = 0; i < 33; i++) {
		if (registers_per_thread_per_radix[i] != 0) {
			double ratio = (registers_per_thread[0] / (double)registers_per_thread_per_radix[i]);
			int ratio_ceil = (int)pfceil(ratio);
			int ratio_floor = (int)floor(ratio);
			double ratio2 = ((registers_per_thread_per_radix[i] * ratio_ceil) / (double)registers_per_thread[0]);
			double ratio3 = (registers_per_thread[0] / (double)(registers_per_thread_per_radix[i] * ratio_floor));
			if (ratio2 > ratio3) registers_per_thread_per_radix[i] *= ratio_floor;
			else {
				registers_per_thread_per_radix[i] *= ratio_ceil;
			}
		}
	}
	registers_per_thread[0] = 0;
	for (int i = 0; i < 33; i++) {
		if ((registers_per_thread_per_radix[i] != 0) && (registers_per_thread_per_radix[i] < min_registers_per_thread[0])) min_registers_per_thread[0] = registers_per_thread_per_radix[i];
		if ((registers_per_thread_per_radix[i] != 0) && (registers_per_thread_per_radix[i] > registers_per_thread[0])) registers_per_thread[0] = registers_per_thread_per_radix[i];
	}
	return VKFFT_SUCCESS;
}
static inline VkFFTResult VkFFTConstructRaderTree(VkFFTApplication* app, VkFFTRaderContainer** raderContainer_input, pfUINT* tempSequence, int* numRaderPrimes, int fft_radix_part) {
	VkFFTResult res = VKFFT_SUCCESS;
	pfUINT locTempSequence = tempSequence[0];
	pfUINT tempSequence_copy = tempSequence[0];
	pfUINT limit = ((tempSequence[0] + 1) > app->configuration.fixMaxRaderPrimeFFT) ? app->configuration.fixMaxRaderPrimeFFT : (tempSequence[0] + 1);
	for (int i = (int)app->configuration.fixMinRaderPrimeMult; i < limit; i++) {
		if (locTempSequence % i == 0) {
			numRaderPrimes[0]++;
			while (locTempSequence % i == 0) locTempSequence /= i;
		}
	}
	for (int i = (int)app->configuration.fixMinRaderPrimeMult; i < app->configuration.fixMaxRaderPrimeMult; i++) {
		if (locTempSequence % i == 0) {
			numRaderPrimes[0]++;
			while (locTempSequence % i == 0) locTempSequence /= i;
		}
	}

	raderContainer_input[0] = (VkFFTRaderContainer*)calloc(sizeof(VkFFTRaderContainer), numRaderPrimes[0]);
	if (raderContainer_input[0] == 0) return VKFFT_ERROR_MALLOC_FAILED;
	VkFFTRaderContainer* raderContainer = raderContainer_input[0];
	pfUINT tempSequence_temp = 1;
	limit = ((tempSequence[0] + 1) > app->configuration.fixMaxRaderPrimeFFT) ? app->configuration.fixMaxRaderPrimeFFT : (tempSequence[0] + 1);
	for (int i = (int)app->configuration.fixMinRaderPrimeMult; i < limit; i++) {
		if (tempSequence[0] % i == 0) {
			if (i < app->configuration.fixMinRaderPrimeFFT) {
				tempSequence_temp *= i;
				tempSequence[0] /= i;
				i--;
				continue;
			}
			//Sophie Germain safe prime check
			pfUINT tempSequence2 = i - 1;
			for (int j = 2; j < app->configuration.fixMinRaderPrimeMult; j++) {
				if (tempSequence2 % j == 0) {
					tempSequence2 /= j;
					j--;
				}
			}
			if (tempSequence2 != 1) {
				tempSequence_temp *= i;
				tempSequence[0] /= i;
				i--;
				continue;
			}
			tempSequence[0] /= i;
			for (int j = 0; j < numRaderPrimes[0]; j++) {
				if (raderContainer[j].prime == i)
				{
					raderContainer[j].multiplier++;
					j = numRaderPrimes[0];
				}
				else if (raderContainer[j].prime == 0) {
					raderContainer[j].type = 0;
					raderContainer[j].prime = i;
					raderContainer[j].multiplier = 1;
					j = numRaderPrimes[0];
				}
			}
			i--;
		}
	}
	tempSequence[0] *= tempSequence_temp;
	for (int i = (int)app->configuration.fixMinRaderPrimeMult; i < app->configuration.fixMaxRaderPrimeMult; i++) {
		if (tempSequence[0] % i == 0) {
			tempSequence[0] /= i;
			for (int j = 0; j < numRaderPrimes[0]; j++) {
				if (raderContainer[j].prime == i)
				{
					raderContainer[j].multiplier++;
					j = numRaderPrimes[0];
				}
				else if (raderContainer[j].prime == 0) {
					raderContainer[j].type = 1;
					raderContainer[j].prime = i;
					raderContainer[j].multiplier = 1;
					j = numRaderPrimes[0];
				}
			}
			i--;
		}
	}
	//main loop for all primes
	for (int i = 0; i < numRaderPrimes[0]; i++) {
		//generator loop
		for (int r = 2; r < raderContainer[i].prime; r++) {
			int test = r;
			for (int iter = 0; iter < raderContainer[i].prime - 2; iter++) {
				if (test == 1) {
					test = 0;
					iter = raderContainer[i].prime;
				}
				test = ((test * r) % raderContainer[i].prime);
			}
			if (test == 1) {
				raderContainer[i].generator = r;
				r = raderContainer[i].prime;
			}
		}

		//subsplit and information initialization
		if (raderContainer[i].type) {//Multiplication
			raderContainer[i].registers_per_thread = 2;
			raderContainer[i].min_registers_per_thread = 2;
		}
		else {//FFT
			locTempSequence = raderContainer[i].prime - 1;
			raderContainer[i].containerFFTDim = raderContainer[i].prime - 1;
			raderContainer[i].containerFFTNum = fft_radix_part * (int)tempSequence_copy / raderContainer[i].prime;
			int stageID = 0;
			for (int j = 2; j < app->configuration.fixMinRaderPrimeMult; j++) {
				if (locTempSequence % j == 0) {
					locTempSequence /= j;
					raderContainer[i].loc_multipliers[j]++;
					//raderContainer[i].stageRadix[stageID] = j;
					//raderContainer[i].numThreadLaunches[stageID] = fft_radix_part * (tempSequence_copy / raderContainer[i].prime) * ((raderContainer[i].prime-1) / j);
					//stageID++;
					j--;
				}
			}
			//int isGoodSequence;
			//if (raderContainer[i].containerFFTNum<8)
			res = VkFFTGetRegistersPerThreadOptimizeShared(raderContainer[i].prime - 1, raderContainer[i].registers_per_thread_per_radix, &raderContainer[i].registers_per_thread, &raderContainer[i].min_registers_per_thread);
			//else
				//res = VkFFTGetRegistersPerThread(raderContainer[i].prime - 1, 0, 0, 1, raderContainer[i].loc_multipliers, raderContainer[i].registers_per_thread_per_radix, &raderContainer[i].registers_per_thread, &raderContainer[i].min_registers_per_thread, &isGoodSequence);
			if (res != VKFFT_SUCCESS) return res;
			if (locTempSequence != 1) {
				res = VkFFTConstructRaderTree(app, &raderContainer[i].container, &locTempSequence, &raderContainer[i].numSubPrimes, fft_radix_part * (int)tempSequence_copy / raderContainer[i].prime);
				if (res != VKFFT_SUCCESS) return res;
				for (int j = 0; j < raderContainer[i].numSubPrimes; j++) {
					for (int t = 0; t < raderContainer[i].container[j].multiplier; t++) {
						raderContainer[i].stageRadix[stageID] = raderContainer[i].container[j].prime;
						stageID++;
					}
				}
			}
			raderContainer[i].numStages = stageID;
		}
	}
	return res;
}
static inline VkFFTResult VkFFTOptimizeRaderFFTRegisters(VkFFTRaderContainer* raderContainer, int numRaderPrimes, int fftDim, int* min_registers_per_thread, int* registers_per_thread, int* registers_per_thread_per_radix) {
	VkFFTResult res = VKFFT_SUCCESS;
	for (pfINT i = 0; i < (pfINT)numRaderPrimes; i++) {
		if (raderContainer[i].type == 0) {
			if (raderContainer[i].min_registers_per_thread / min_registers_per_thread[0] >= 2) {
				min_registers_per_thread[0] *= (raderContainer[i].min_registers_per_thread / min_registers_per_thread[0]);
				for (int j = 0; j < 33; j++) {
					if ((registers_per_thread_per_radix[j] > 0) && (registers_per_thread_per_radix[j] < min_registers_per_thread[0])) registers_per_thread_per_radix[j] *= (int)pfceil(min_registers_per_thread[0] / (double)registers_per_thread_per_radix[j]);
				}
				for (int j = 0; j < 33; j++) {
					if (registers_per_thread_per_radix[j] > registers_per_thread[0]) registers_per_thread[0] = registers_per_thread_per_radix[j];
				}
			}
			else if (min_registers_per_thread[0] / raderContainer[i].min_registers_per_thread >= 2) {
				raderContainer[i].min_registers_per_thread *= (min_registers_per_thread[0] / raderContainer[i].min_registers_per_thread);
				for (int j = 0; j < 33; j++) {
					if ((raderContainer[i].registers_per_thread_per_radix[j] > 0) && (raderContainer[i].registers_per_thread_per_radix[j] < raderContainer[i].min_registers_per_thread)) raderContainer[i].registers_per_thread_per_radix[j] *= (int)pfceil(raderContainer[i].min_registers_per_thread / (double)raderContainer[i].registers_per_thread_per_radix[j]);
				}
				for (int j = 0; j < 33; j++) {
					if (raderContainer[i].registers_per_thread_per_radix[j] > raderContainer[i].registers_per_thread) raderContainer[i].registers_per_thread = raderContainer[i].registers_per_thread_per_radix[j];
				}
			}

			if (raderContainer[i].min_registers_per_thread < min_registers_per_thread[0]) {
				for (int j = 0; j < 33; j++) {
					if (raderContainer[i].registers_per_thread_per_radix[j] > 0) {
						while (raderContainer[i].registers_per_thread_per_radix[j] < min_registers_per_thread[0])
							raderContainer[i].registers_per_thread_per_radix[j] += j;
						if (raderContainer[i].registers_per_thread_per_radix[j] > raderContainer[i].registers_per_thread)
							raderContainer[i].registers_per_thread = raderContainer[i].registers_per_thread_per_radix[j];
					}
				}
			}

			for (pfINT j = 2; j < 33; j++) {
				if (raderContainer[i].registers_per_thread_per_radix[j] != 0) {
					double scaling = (raderContainer[i].containerFFTDim > raderContainer[i].registers_per_thread_per_radix[j]) ? pfceil(raderContainer[i].containerFFTDim / (double)raderContainer[i].registers_per_thread_per_radix[j]) : 1.0 / floor(raderContainer[i].registers_per_thread_per_radix[j] / (double)raderContainer[i].containerFFTDim);
					while (((int)pfceil(fftDim / (double)min_registers_per_thread[0])) < (raderContainer[i].containerFFTNum * scaling)) {
						raderContainer[i].registers_per_thread_per_radix[j] += (int)j;
						scaling = (raderContainer[i].containerFFTDim > raderContainer[i].registers_per_thread_per_radix[j]) ? pfceil(raderContainer[i].containerFFTDim / (double)raderContainer[i].registers_per_thread_per_radix[j]) : 1.0 / floor(raderContainer[i].registers_per_thread_per_radix[j] / (double)raderContainer[i].containerFFTDim);
					}
					if (raderContainer[i].registers_per_thread_per_radix[j] > raderContainer[i].registers_per_thread) raderContainer[i].registers_per_thread = raderContainer[i].registers_per_thread_per_radix[j];
				}
			}
			if (raderContainer[i].registers_per_thread > registers_per_thread[0]) registers_per_thread[0] = raderContainer[i].registers_per_thread;
		}
	}
	//try to increase registers usage closer to registers_per_thread across all primes
	for (pfINT i = 0; i < (pfINT)numRaderPrimes; i++) {
		if (raderContainer[i].type == 0) {
			for (pfINT j = 2; j < 33; j++) {
				if (raderContainer[i].registers_per_thread_per_radix[j] > 0) {
					while ((raderContainer[i].registers_per_thread_per_radix[j] + j) <= registers_per_thread[0] + 1) {// fix
						raderContainer[i].registers_per_thread_per_radix[j] += (int)j;
					}
				}
			}
			raderContainer[i].registers_per_thread = 0;
			raderContainer[i].min_registers_per_thread = 10000000;
			for (pfINT j = 2; j < 33; j++) {
				if (raderContainer[i].registers_per_thread_per_radix[j] > 0) {
					if (raderContainer[i].registers_per_thread_per_radix[j] < raderContainer[i].min_registers_per_thread) {
						raderContainer[i].min_registers_per_thread = raderContainer[i].registers_per_thread_per_radix[j];
					}
					if (raderContainer[i].registers_per_thread_per_radix[j] > raderContainer[i].registers_per_thread) {
						raderContainer[i].registers_per_thread = raderContainer[i].registers_per_thread_per_radix[j];
					}
				}
			}
		}
	}
	//subprimes optimization
	for (pfINT i = 0; i < (pfINT)numRaderPrimes; i++) {
		if (raderContainer[i].numSubPrimes) {
			res = VkFFTOptimizeRaderFFTRegisters(raderContainer[i].container, raderContainer[i].numSubPrimes, fftDim, min_registers_per_thread, registers_per_thread, registers_per_thread_per_radix);
			if (res != VKFFT_SUCCESS) return res;
		}
	}
	for (pfINT i = 0; i < (pfINT)numRaderPrimes; i++) {
		if (min_registers_per_thread[0] > raderContainer[i].min_registers_per_thread) min_registers_per_thread[0] = raderContainer[i].min_registers_per_thread;
		if (registers_per_thread[0] < raderContainer[i].registers_per_thread) registers_per_thread[0] = raderContainer[i].registers_per_thread;
	}
	return res;
}
static inline VkFFTResult VkFFTOptimizeRadixKernels(int* registers_per_thread_per_radix, int* loc_multipliers, int registerBoost, int* maxNonPow2Radix, int* reqLocRegs, VkFFTRaderContainer* raderContainer, int numRaderPrimes) {
	VkFFTResult res = VKFFT_SUCCESS;
	if (numRaderPrimes) {
		for (int i = 0; i < numRaderPrimes; i++) {
			res = VkFFTOptimizeRadixKernels(raderContainer[i].registers_per_thread_per_radix, raderContainer[i].loc_multipliers, 1, maxNonPow2Radix, reqLocRegs, raderContainer[i].container, raderContainer[i].numSubPrimes);
			if (res != VKFFT_SUCCESS) return res;
		}
	}
	//optimize used radix kernels
	if (((registers_per_thread_per_radix[32] > 0) || ((registers_per_thread_per_radix[2] % 32) == 0)) && ((registers_per_thread_per_radix[32]) % 32 == 0) && (loc_multipliers[2] >= 5)) {
		loc_multipliers[32] = loc_multipliers[2] / 5;
		loc_multipliers[2] = loc_multipliers[2] - loc_multipliers[32] * 5;
		if ((registers_per_thread_per_radix[2] % 32) == 0) registers_per_thread_per_radix[32] = registers_per_thread_per_radix[2];
	}
	if (((registers_per_thread_per_radix[16] > 0) || ((registers_per_thread_per_radix[2] % 16) == 0)) && ((registers_per_thread_per_radix[16]) % 16 == 0) && (loc_multipliers[2] >= 4)) {
		loc_multipliers[16] = loc_multipliers[2] / 4;
		loc_multipliers[2] = loc_multipliers[2] - loc_multipliers[16] * 4;
		if ((registers_per_thread_per_radix[2] % 16) == 0) registers_per_thread_per_radix[16] = registers_per_thread_per_radix[2];//if we got 16 regs, why not use r16 kernel
	}
	if ((registers_per_thread_per_radix[15] > 0) && ((registers_per_thread_per_radix[15]) % 15 == 0) && (loc_multipliers[3] >= 1) && (loc_multipliers[5] >= 1)) {
		loc_multipliers[15] = (loc_multipliers[3] > loc_multipliers[5]) ? loc_multipliers[5] : loc_multipliers[3];
		loc_multipliers[3] = loc_multipliers[3] - loc_multipliers[15];
		loc_multipliers[5] = loc_multipliers[5] - loc_multipliers[15];
	}
	if ((registers_per_thread_per_radix[14] > 0) && ((registers_per_thread_per_radix[14]) % 14 == 0) && (loc_multipliers[2] >= 1) && (loc_multipliers[7] >= 1)) {
		loc_multipliers[14] = (loc_multipliers[2] > loc_multipliers[7]) ? loc_multipliers[7] : loc_multipliers[2];
		loc_multipliers[2] = loc_multipliers[2] - loc_multipliers[14];
		loc_multipliers[7] = loc_multipliers[7] - loc_multipliers[14];
	}
	if ((registers_per_thread_per_radix[12] > 0) && ((registers_per_thread_per_radix[12]) % 12 == 0) && (loc_multipliers[2] >= 2) && (loc_multipliers[3] >= 1)) {
		loc_multipliers[12] = (loc_multipliers[2] > 2 * loc_multipliers[3]) ? loc_multipliers[3] : loc_multipliers[2] / 2;
		loc_multipliers[2] = loc_multipliers[2] - 2 * loc_multipliers[12];
		loc_multipliers[3] = loc_multipliers[3] - loc_multipliers[12];
	}
	if ((registers_per_thread_per_radix[10] > 0) && ((registers_per_thread_per_radix[10]) % 10 == 0) && (loc_multipliers[2] >= 1) && (loc_multipliers[5] >= 1)) {
		loc_multipliers[10] = (loc_multipliers[2] > loc_multipliers[5]) ? loc_multipliers[5] : loc_multipliers[2];
		loc_multipliers[2] = loc_multipliers[2] - loc_multipliers[10];
		loc_multipliers[5] = loc_multipliers[5] - loc_multipliers[10];
	}
	if ((registers_per_thread_per_radix[9] > 0) && ((registers_per_thread_per_radix[9]) % 9 == 0) && (loc_multipliers[3] >= 2)) {
		loc_multipliers[9] = loc_multipliers[3] / 2;
		loc_multipliers[3] = loc_multipliers[3] - loc_multipliers[9] * 2;
	}
	if (((registers_per_thread_per_radix[8] > 0) || ((registers_per_thread_per_radix[2] % 8) == 0)) && ((registers_per_thread_per_radix[8]) % 8 == 0) && (loc_multipliers[2] >= 3)) {
		loc_multipliers[8] = loc_multipliers[2] / 3;
		loc_multipliers[2] = loc_multipliers[2] - loc_multipliers[8] * 3;
		if ((registers_per_thread_per_radix[2] % 8) == 0) registers_per_thread_per_radix[8] = registers_per_thread_per_radix[2];
	}
	if ((registers_per_thread_per_radix[6] > 0) && ((registers_per_thread_per_radix[6]) % 6 == 0) && (loc_multipliers[2] >= 1) && (loc_multipliers[3] >= 1)) {
		loc_multipliers[6] = (loc_multipliers[2] > loc_multipliers[3]) ? loc_multipliers[3] : loc_multipliers[2];
		loc_multipliers[2] = loc_multipliers[2] - loc_multipliers[6];
		loc_multipliers[3] = loc_multipliers[3] - loc_multipliers[6];
	}
	if (((registers_per_thread_per_radix[4] > 0) || ((registers_per_thread_per_radix[2] % 4) == 0)) && ((registers_per_thread_per_radix[4]) % 4 == 0) && (loc_multipliers[2] >= 2)) {
		loc_multipliers[4] = loc_multipliers[2] / 2;
		loc_multipliers[2] = loc_multipliers[2] - loc_multipliers[4] * 2;
		if ((registers_per_thread_per_radix[2] % 4) == 0) registers_per_thread_per_radix[4] = registers_per_thread_per_radix[2];
	}
	if ((registerBoost == 2) && (loc_multipliers[2] == 0)) {
		if (loc_multipliers[4] > 0) {
			loc_multipliers[4]--;
			loc_multipliers[2] = 2;
		}
		else if (loc_multipliers[8] > 0) {
			loc_multipliers[8]--;
			loc_multipliers[4]++;
			loc_multipliers[2]++;
		}
		else if (loc_multipliers[16] > 0) {
			loc_multipliers[16]--;
			loc_multipliers[8]++;
			loc_multipliers[2]++;
		}
		else if (loc_multipliers[32] > 0) {
			loc_multipliers[32]--;
			loc_multipliers[16]++;
			loc_multipliers[2]++;
		}
	}
	if ((registerBoost == 4) && (loc_multipliers[4] == 0)) {
		if (loc_multipliers[8] > 0) {
			loc_multipliers[8]--;
			loc_multipliers[4]++;
			loc_multipliers[2]++;
		}
		else if (loc_multipliers[16] > 0) {
			if (loc_multipliers[2] == 0) {
				loc_multipliers[16]--;
				loc_multipliers[4] = 2;
			}
			else {
				loc_multipliers[16]--;
				loc_multipliers[4]++;
				loc_multipliers[2]--;
				loc_multipliers[8]++;
			}
		}
		else if (loc_multipliers[32] > 0) {
			if (loc_multipliers[2] == 0) {
				loc_multipliers[32]--;
				loc_multipliers[8]++;
				loc_multipliers[4]++;
			}
			else {
				loc_multipliers[32]--;
				loc_multipliers[16]++;
				loc_multipliers[4]++;
				loc_multipliers[2]--;
			}
		}
	}
	for (int i = 2; i < 33; i++) {
		int usedLocRegs = 0;
		if (loc_multipliers[i] > 0) {
			switch (i) {
			case 6:
				usedLocRegs = 3;
				break;
			case 9:
				usedLocRegs = 3;
				break;
			case 10:
				usedLocRegs = 5;
				break;
			case 12:
				usedLocRegs = 3;
				break;
			case 14:
				usedLocRegs = 7;
				break;
			case 15:
				usedLocRegs = 5;
				break;
			default:
				usedLocRegs = i;
				break;
			}
		}
		if ((loc_multipliers[i] > 0) && ((i & (i - 1)) != 0) && (i > maxNonPow2Radix[0])) {
			maxNonPow2Radix[0] = i;
		}
		if ((usedLocRegs > reqLocRegs[0]) && ((i & (i - 1)) != 0)) {
			reqLocRegs[0] = usedLocRegs;
		}
	}
	return res;
}
static inline VkFFTResult VkFFTGetRaderFFTStages(VkFFTRaderContainer* raderContainer, int numRaderPrimes, int* stageid, int* stageRadix, int* stage_rader_generator) {
	VkFFTResult res = VKFFT_SUCCESS;
	for (pfINT i = 0; i < (pfINT)numRaderPrimes; i++) {
		if (raderContainer[i].multiplier > 0) {
			stageRadix[stageid[0]] = raderContainer[i].prime;
			stage_rader_generator[stageid[0]] = raderContainer[i].generator;
			raderContainer[i].multiplier--;
			i--;
			stageid[0]++;
			//axes[k].specializationConstants.numStages++;
			//find primitive root
		}
	}
	for (pfINT i = 0; i < (pfINT)numRaderPrimes; i++) {
		if (raderContainer[i].type == 0) {
			if (raderContainer[i].numSubPrimes > 0) {
				res = VkFFTGetRaderFFTStages(raderContainer[i].container, raderContainer[i].numSubPrimes, &raderContainer[i].numStages, raderContainer[i].stageRadix, raderContainer[i].stage_rader_generator);
				if (res != VKFFT_SUCCESS) return res;
			}
			for (int j = 32; j > 1; j--) {
				if (raderContainer[i].loc_multipliers[j] > 0) {
					raderContainer[i].stageRadix[raderContainer[i].numStages] = j;
					raderContainer[i].loc_multipliers[j]--;
					j++;
					raderContainer[i].numStages++;
				}
			}
			/*//make that convolution step uses min_regs radix - max working threads
			int stage_id_swap = axes[k].specializationConstants.raderContainer[i].numStages - 1;
			int temp_radix = axes[k].specializationConstants.raderContainer[i].stageRadix[axes[k].specializationConstants.raderContainer[i].numStages - 1];
			int temp_regs = axes[k].specializationConstants.raderContainer[i].registers_per_thread_per_radix[axes[k].specializationConstants.raderContainer[i].stageRadix[axes[k].specializationConstants.raderContainer[i].numStages - 1]];

			for (int j = 0; j < axes[k].specializationConstants.raderContainer[i].numStages-1; j++) {
				if (axes[k].specializationConstants.raderContainer[i].registers_per_thread_per_radix[axes[k].specializationConstants.raderContainer[i].stageRadix[j]] < axes[k].specializationConstants.raderContainer[i].registers_per_thread_per_radix[axes[k].specializationConstants.raderContainer[i].stageRadix[stage_id_swap]])
					stage_id_swap = j;
			}
			axes[k].specializationConstants.raderContainer[i].stageRadix[axes[k].specializationConstants.raderContainer[i].numStages - 1] = axes[k].specializationConstants.raderContainer[i].stageRadix[stage_id_swap];
			axes[k].specializationConstants.raderContainer[i].registers_per_thread_per_radix[axes[k].specializationConstants.raderContainer[i].stageRadix[axes[k].specializationConstants.raderContainer[i].numStages - 1]] = axes[k].specializationConstants.raderContainer[i].registers_per_thread_per_radix[axes[k].specializationConstants.raderContainer[i].stageRadix[stage_id_swap]];
			axes[k].specializationConstants.raderContainer[i].stageRadix[stage_id_swap] = temp_radix;
			axes[k].specializationConstants.raderContainer[i].registers_per_thread_per_radix[axes[k].specializationConstants.raderContainer[i].stageRadix[stage_id_swap]] = temp_regs;

			//make that first step uses second to min_regs radix
			stage_id_swap = 0;
			temp_radix = axes[k].specializationConstants.raderContainer[i].stageRadix[0];
			temp_regs = axes[k].specializationConstants.raderContainer[i].registers_per_thread_per_radix[axes[k].specializationConstants.raderContainer[i].stageRadix[0]];

			for (int j = 1; j < axes[k].specializationConstants.raderContainer[i].numStages - 1; j++) {
				if (axes[k].specializationConstants.raderContainer[i].registers_per_thread_per_radix[axes[k].specializationConstants.raderContainer[i].stageRadix[j]] < axes[k].specializationConstants.raderContainer[i].registers_per_thread_per_radix[axes[k].specializationConstants.raderContainer[i].stageRadix[stage_id_swap]])
					stage_id_swap = j;
			}
			axes[k].specializationConstants.raderContainer[i].stageRadix[0] = axes[k].specializationConstants.raderContainer[i].stageRadix[stage_id_swap];
			axes[k].specializationConstants.raderContainer[i].registers_per_thread_per_radix[axes[k].specializationConstants.raderContainer[i].stageRadix[0]] = axes[k].specializationConstants.raderContainer[i].registers_per_thread_per_radix[axes[k].specializationConstants.raderContainer[i].stageRadix[stage_id_swap]];
			axes[k].specializationConstants.raderContainer[i].stageRadix[stage_id_swap] = temp_radix;
			axes[k].specializationConstants.raderContainer[i].registers_per_thread_per_radix[axes[k].specializationConstants.raderContainer[i].stageRadix[stage_id_swap]] = temp_regs;
			*/
		}
	}
	return res;
}
static inline VkFFTResult VkFFTMinMaxRegisterCheck(int numStages, int* stageRadix, int* min_registers_per_thread, int* registers_per_thread, int* registers_per_thread_per_radix, VkFFTRaderContainer* raderContainer, int numRaderPrimes, int* stage_rader_generator) {
	VkFFTResult res = VKFFT_SUCCESS;
	for (pfINT j = 0; j < (pfINT)numStages; j++) {
		if (stage_rader_generator[j] == 0) {
			if (registers_per_thread_per_radix[stageRadix[j]] > 0) {
				if (registers_per_thread_per_radix[stageRadix[j]] < min_registers_per_thread[0]) {
					min_registers_per_thread[0] = registers_per_thread_per_radix[stageRadix[j]];
				}
				if (registers_per_thread_per_radix[stageRadix[j]] > registers_per_thread[0]) {
					registers_per_thread[0] = registers_per_thread_per_radix[stageRadix[j]];
				}
			}
		}
		else {
			for (pfINT i = 0; i < (pfINT)numRaderPrimes; i++) {
				if (raderContainer[i].prime == stageRadix[j]) {
					if (raderContainer[i].type == 0) {
						for (pfINT j2 = 0; j2 < (pfINT)raderContainer[i].numStages; j2++) {
							if (raderContainer[i].stage_rader_generator[j] == 0) {
								if (raderContainer[i].registers_per_thread_per_radix[raderContainer[i].stageRadix[j2]] > 0) {
									if (raderContainer[i].registers_per_thread_per_radix[raderContainer[i].stageRadix[j2]] < min_registers_per_thread[0]) {
										min_registers_per_thread[0] = raderContainer[i].registers_per_thread_per_radix[raderContainer[i].stageRadix[j2]];
									}
									if (raderContainer[i].registers_per_thread_per_radix[raderContainer[i].stageRadix[j2]] > registers_per_thread[0]) {
										registers_per_thread[0] = raderContainer[i].registers_per_thread_per_radix[raderContainer[i].stageRadix[j2]];
									}
								}
							}
							else {
								res = VkFFTMinMaxRegisterCheck(raderContainer[i].numStages, raderContainer[i].stageRadix, min_registers_per_thread, registers_per_thread, raderContainer[i].registers_per_thread_per_radix, raderContainer[i].container, raderContainer[i].numSubPrimes, raderContainer[i].stage_rader_generator);
								if (res != VKFFT_SUCCESS) return res;
							}
						}
					}
				}
			}
		}
	}
	return res;
}
static inline VkFFTResult VkFFTGetRaderFFTThreadsNum(VkFFTRaderContainer* raderContainer, int numRaderPrimes, int* numThreads) {
	VkFFTResult res = VKFFT_SUCCESS;

	for (pfINT i = 0; i < (pfINT)numRaderPrimes; i++) {
		if (raderContainer[i].type == 0) {
			if (raderContainer[i].numSubPrimes > 0) {
				res = VkFFTGetRaderFFTThreadsNum(raderContainer[i].container, raderContainer[i].numSubPrimes, numThreads);
				if (res != VKFFT_SUCCESS) return res;
			}
			for (pfINT j = 0; j < (pfINT)raderContainer[i].numStages; j++) {
				if (raderContainer[i].stage_rader_generator[j] == 0) {
					if (raderContainer[i].containerFFTNum * (int)pfceil(raderContainer[i].containerFFTDim / (double)raderContainer[i].registers_per_thread_per_radix[raderContainer[i].stageRadix[j]]) > numThreads[0]) numThreads[0] = raderContainer[i].containerFFTNum * (int)pfceil(raderContainer[i].containerFFTDim / (double)raderContainer[i].registers_per_thread_per_radix[raderContainer[i].stageRadix[j]]);
				}
			}
		}
	}
	return res;
}

static inline VkFFTResult VkFFTScheduler(VkFFTApplication* app, VkFFTPlan* FFTPlan, int axis_id) {
	VkFFTResult res = VKFFT_SUCCESS;
	VkFFTAxis* axes = FFTPlan->axes[axis_id];

	int complexSize;
	if (app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory)
		complexSize = (4 * sizeof(double));
	else 
	{
		if (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory)
			complexSize = (2 * sizeof(double));
		else
			if (app->configuration.halfPrecision)
				complexSize = (2 * sizeof(float));
			else
				complexSize = (2 * sizeof(float));
	}
	int usedSharedMemory = (((app->configuration.size[axis_id] & (app->configuration.size[axis_id] - 1)) == 0) && (!app->configuration.performDCT) && (!app->configuration.performDST)) ? (int)app->configuration.sharedMemorySizePow2 : (int)app->configuration.sharedMemorySize;
	int maxSequenceLengthSharedMemory = usedSharedMemory / complexSize;
	int maxSingleSizeNonStrided = maxSequenceLengthSharedMemory;

	int nonStridedAxisId = (app->configuration.considerAllAxesStrided) ? -1 : 0;
	pfUINT max_rhs = 1;
	for (int i = 0; i < app->configuration.FFTdim; i++) {
		FFTPlan->actualFFTSizePerAxis[axis_id][i] = app->configuration.size[i];
		if ((FFTPlan->actualFFTSizePerAxis[axis_id][i] > 0)) max_rhs *= FFTPlan->actualFFTSizePerAxis[axis_id][i];
	}
	for (int i = (int)app->configuration.FFTdim; i < VKFFT_MAX_FFT_DIMENSIONS; i++) {
		FFTPlan->actualFFTSizePerAxis[axis_id][i] = 1;
	}
	if (app->configuration.numberBatches > app->actualNumBatches)
		max_rhs *= app->configuration.numberBatches;
	else
		max_rhs *= app->actualNumBatches;
	if (app->configuration.coordinateFeatures > 0) max_rhs *= app->configuration.coordinateFeatures;
	if (app->configuration.numberKernels > 0) max_rhs *= app->configuration.numberKernels;

	FFTPlan->actualPerformR2CPerAxis[axis_id] = (axis_id == 0) ? app->configuration.performR2C : 0;
	if ((axis_id == 0) && (app->configuration.performR2C) && (app->configuration.size[axis_id] > maxSingleSizeNonStrided) && ((app->configuration.size[axis_id] % 2) == 0) && (!app->configuration.forceCallbackVersionRealTransforms)) {
		if (app->configuration.vendorID == 0x1027f00){
			app->configuration.forceCallbackVersionRealTransforms = 1;
		} //fix for m1 r2c issue 
		else{
			FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] = app->configuration.size[axis_id] / 2; // now in actualFFTSize - modified dimension size for R2C/DCT
			FFTPlan->actualPerformR2CPerAxis[axis_id] = 0;
			FFTPlan->bigSequenceEvenR2C = 1;
		}
	}
	if (app->configuration.performDCT == 1) {
		FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] = 2 * app->configuration.size[axis_id] - 2; // now in actualFFTSize - modified dimension size for R2C/DCT
	}
	if (app->configuration.performDST == 1) {
		FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] = 2 * app->configuration.size[axis_id] + 2; // now in actualFFTSize - modified dimension size for R2C/DCT
	}
	if (((app->configuration.performDCT == 4) || (app->configuration.performDST == 4)) && (app->configuration.size[axis_id] % 2 == 0)) {
		FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] = app->configuration.size[axis_id] / 2; // now in actualFFTSize - modified dimension size for R2C/DCT
		//FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] = app->configuration.size[axis_id] * 8; // now in actualFFTSize - modified dimension size for R2C/DCT
	}
	if ((axis_id > 0) && (app->configuration.performR2C)) {
		FFTPlan->actualFFTSizePerAxis[axis_id][0] = FFTPlan->actualFFTSizePerAxis[axis_id][0] / 2 + 1;
	}
	if (axis_id != nonStridedAxisId) {
		if (app->configuration.performBandwidthBoost > 0)
			axes->specializationConstants.performBandwidthBoost = (int)app->configuration.performBandwidthBoost;
	}
	//initial Stockham + Rader check
	int multipliers[33];
	for (int i = 0; i < 33; i++) {
		multipliers[i] = 0;
	}

	pfUINT tempSequence = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id];
	for (int i = 2; i < app->configuration.fixMinRaderPrimeMult; i++) {
		if (tempSequence % i == 0) {
			tempSequence /= i;
			multipliers[i]++;
			i--;
		}
	}
	// verify that we haven't checked for 3 steps being not enougth for Rader before
	int forceRaderTwoUpload = 0; // for sequences like 17*1023 it is better to switch to two uploads for better occupancy. We will switch if one of the Rader primes requests more than 512 threads or maxThreadsNum value.
	if (!app->useBluesteinFFT[axis_id]) {
		int useRaderMult = 0;
		int rader_primes[20];
		int rader_multipliers[20];
		for (int i = 0; i < 20; i++) {
			rader_multipliers[i] = 0;
			rader_primes[i] = 0;
		}
		pfUINT tempSequence_temp = 1;
		int maxSequenceLengthSharedMemoryStrided_temp = (app->configuration.coalescedMemory > complexSize) ? usedSharedMemory / ((int)app->configuration.coalescedMemory) : usedSharedMemory / complexSize;
		int limit_max_rader_prime = ((axis_id == nonStridedAxisId) && (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] <= maxSequenceLengthSharedMemory)) ? maxSequenceLengthSharedMemory : maxSequenceLengthSharedMemoryStrided_temp;
		if (limit_max_rader_prime > app->configuration.fixMaxRaderPrimeFFT) limit_max_rader_prime = (int)app->configuration.fixMaxRaderPrimeFFT;
		for (int i = (int)app->configuration.fixMinRaderPrimeMult; i < limit_max_rader_prime; i++) {
			if (tempSequence % i == 0) {
				if (i < app->configuration.fixMinRaderPrimeFFT) {
					tempSequence_temp *= i;
					tempSequence /= i;
					i--;
					continue;
				}
				//Sophie Germain safe prime check
				pfUINT tempSequence2 = i - 1;
				for (int j = 2; j < app->configuration.fixMinRaderPrimeMult; j++) {
					if (tempSequence2 % j == 0) {
						tempSequence2 /= j;
						j--;
					}
				}
				if (tempSequence2 != 1) {
					maxSequenceLengthSharedMemory = (usedSharedMemory - (i - 1) * complexSize) / complexSize;
					maxSequenceLengthSharedMemoryStrided_temp = (app->configuration.coalescedMemory > complexSize) ? (usedSharedMemory - (i - 1) * complexSize) / ((int)app->configuration.coalescedMemory) : (usedSharedMemory - (i - 1) * complexSize) / complexSize;
					limit_max_rader_prime = ((axis_id == nonStridedAxisId) && (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] <= maxSequenceLengthSharedMemory)) ? maxSequenceLengthSharedMemory : maxSequenceLengthSharedMemoryStrided_temp;
					tempSequence_temp *= i;
					tempSequence /= i;
					i--;
					continue;
				}
				tempSequence /= i;
				if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / i > 512) forceRaderTwoUpload = 1;
				if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / i > app->configuration.maxThreadsNum) forceRaderTwoUpload = 1;
				for (int j = 0; j < 20; j++) {
					if (rader_primes[j] == i)
					{
						rader_multipliers[j]++;
						j = 20;
					}
					else if (rader_primes[j] == 0) {
						rader_primes[j] = i;
						rader_multipliers[j]++;
						j = 20;
					}

				}
				i--;
			}
		}
		tempSequence *= tempSequence_temp;
		int maxRaderPrimeFromThreadNumCoalesced = (int)(app->configuration.maxThreadsNum / (app->configuration.coalescedMemory / complexSize)) * 2 - 1;
		if (maxRaderPrimeFromThreadNumCoalesced < app->configuration.fixMaxRaderPrimeMult) app->configuration.fixMaxRaderPrimeMult = maxRaderPrimeFromThreadNumCoalesced;

		for (int i = (int)app->configuration.fixMinRaderPrimeMult; i < app->configuration.fixMaxRaderPrimeMult; i++) {
			if (tempSequence % i == 0) {
				tempSequence /= i;
				for (int j = 0; j < 20; j++) {
					if (rader_primes[j] == i)
					{
						rader_multipliers[j]++;
						j = 20;
					}
					else if (rader_primes[j] == 0) {
						rader_primes[j] = i;
						rader_multipliers[j]++;
						j = 20;
					}

				}
				useRaderMult = i;
				i--;
			}
		}
		if (tempSequence != 1) {
			useRaderMult = 0;
			forceRaderTwoUpload = 0;
		}
		if (useRaderMult) {
			if (tempSequence == 1) usedSharedMemory -= (useRaderMult - 1) * complexSize; //reserve memory for Rader 
		}
		maxSequenceLengthSharedMemory = usedSharedMemory / complexSize;
		maxSingleSizeNonStrided = maxSequenceLengthSharedMemory;
		//check once again for R2C
		if ((axis_id == 0) && (app->configuration.performR2C) && (tempSequence == 1) && ((app->configuration.size[axis_id] > maxSingleSizeNonStrided) || forceRaderTwoUpload) && ((app->configuration.size[axis_id] % 2) == 0) && (!app->configuration.forceCallbackVersionRealTransforms)) {
			if (app->configuration.vendorID == 0x1027f00){
				app->configuration.forceCallbackVersionRealTransforms = 1;
			} //fix for m1 r2c issue 
			else{
				FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] = app->configuration.size[axis_id] / 2; // now in actualFFTSize - modified dimension size for R2C/DCT
				FFTPlan->actualPerformR2CPerAxis[axis_id] = 0;
				FFTPlan->bigSequenceEvenR2C = 1;
			}
		}
	}
	//initial Bluestein check
	if (tempSequence != 1) {
		app->useBluesteinFFT[axis_id] = 1;
		if (axis_id != nonStridedAxisId) {
			if (app->configuration.performBandwidthBoost == 0)
				axes->specializationConstants.performBandwidthBoost = 1;
		}
		app->configuration.registerBoost = 1;
		tempSequence = 2 * FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - 1;
		int FFTSizeSelected = 0;
		if (((app->configuration.useCustomBluesteinPaddingPattern > 0) || (app->configuration.autoCustomBluesteinPaddingPattern)) && (!app->configuration.fixMaxRadixBluestein)) {
			int arr_limit = (app->configuration.useCustomBluesteinPaddingPattern) ? (int)app->configuration.useCustomBluesteinPaddingPattern : (int)app->configuration.autoCustomBluesteinPaddingPattern;
			for (int i = 0; i < arr_limit; i++) {
				if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] >= app->configuration.primeSizes[i]) {
					if (i != (arr_limit - 1)) {
						if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] < app->configuration.primeSizes[i + 1]) {
							tempSequence = app->configuration.paddedSizes[i];
							FFTSizeSelected = 1;
							i = arr_limit;
						}
					}
					else {
						if ((2 * FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - 1) <= app->configuration.paddedSizes[i]) {
							tempSequence = app->configuration.paddedSizes[i];
							FFTSizeSelected = 1;
							i = arr_limit;
						}
					}
				}
			}
		}
		if (app->configuration.fixMaxRadixBluestein > 0) {
			while (!FFTSizeSelected) {
				int testSequence = (int)tempSequence;
				for (int i = 0; i < 33; i++) {
					multipliers[i] = 0;
				}
				for (int i = 2; i < app->configuration.fixMaxRadixBluestein + 1; i++) {
					if (testSequence % i == 0) {
						testSequence /= i;
						multipliers[i]++;
						i--;
					}
				}
				if (testSequence == 1) FFTSizeSelected = 1;
				else tempSequence++;
			}
		}
		else {
			while (!FFTSizeSelected) {
				if (axis_id == nonStridedAxisId) {
					if ((FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] < 128) || ((((int)pow(2, (int)pfceil(log2(tempSequence))) * 0.75) <= tempSequence) && (((int)pow(2, (int)pfceil(log2(tempSequence))) <= maxSequenceLengthSharedMemory) || ((2 * FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - 1) > maxSequenceLengthSharedMemory))))  tempSequence = (int)pow(2, (int)pfceil(log2(tempSequence)));
				}
				else {
					int maxSequenceLengthSharedMemoryStrided_temp = (app->configuration.coalescedMemory > complexSize) ? usedSharedMemory / ((int)app->configuration.coalescedMemory) : usedSharedMemory / complexSize;
					if ((FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] < 128) || ((((int)pow(2, (int)pfceil(log2(tempSequence))) * 0.75) <= tempSequence) && (((int)pow(2, (int)pfceil(log2(tempSequence))) <= maxSequenceLengthSharedMemoryStrided_temp) || ((2 * FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - 1) > maxSequenceLengthSharedMemoryStrided_temp))))  tempSequence = (int)pow(2, (int)pfceil(log2(tempSequence)));
				}
				pfUINT testSequence = tempSequence;
				for (int i = 0; i < 33; i++) {
					multipliers[i] = 0;
				}
				for (int i = 2; i < 8; i++) {
					if (testSequence % i == 0) {
						testSequence /= i;
						multipliers[i]++;
						i--;
					}
				}
				if (testSequence != 1) tempSequence++;
				else {
					int registers_per_thread_per_radix[33];
					int registers_per_thread = 0;
					int min_registers_per_thread = 10000000;
					int isGoodSequence = 0;
					res = VkFFTGetRegistersPerThread(app, (int)tempSequence, 0, max_rhs / tempSequence, axes->specializationConstants.useRader, multipliers, registers_per_thread_per_radix, &registers_per_thread, &min_registers_per_thread, &isGoodSequence);
					if (res != VKFFT_SUCCESS) return res;
					if (isGoodSequence) FFTSizeSelected = 1;
					else tempSequence++;
				}
			}
		}
		FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] = tempSequence;
		//check if padded system still single upload for r2c - else redo the optimization
		if ((axis_id == 0) && (app->configuration.performR2C) && (!FFTPlan->bigSequenceEvenR2C) && (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] > maxSingleSizeNonStrided) && ((app->configuration.size[axis_id] % 2) == 0) && (!app->configuration.forceCallbackVersionRealTransforms)) {
			if (app->configuration.vendorID == 0x1027f00){
				app->configuration.forceCallbackVersionRealTransforms = 1;
			} //fix for m1 r2c issue 
			else{
				FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] = app->configuration.size[axis_id] / 2; // now in actualFFTSize - modified dimension size for R2C/DCT
				FFTPlan->actualPerformR2CPerAxis[axis_id] = 0;
				FFTPlan->bigSequenceEvenR2C = 1;
				tempSequence = 2 * FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - 1;
				FFTSizeSelected = 0;
				if (((app->configuration.useCustomBluesteinPaddingPattern > 0) || (app->configuration.autoCustomBluesteinPaddingPattern)) && (!app->configuration.fixMaxRadixBluestein)) {
					int arr_limit = (app->configuration.useCustomBluesteinPaddingPattern) ? (int)app->configuration.useCustomBluesteinPaddingPattern : (int)app->configuration.autoCustomBluesteinPaddingPattern;
					for (int i = 0; i < arr_limit; i++) {
						if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] >= app->configuration.primeSizes[i]) {
							if (i != (arr_limit - 1)) {
								if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] < app->configuration.primeSizes[i + 1]) {
									tempSequence = app->configuration.paddedSizes[i];
									FFTSizeSelected = 1;
								}
							}
							else {
								if ((2 * FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - 1) <= app->configuration.paddedSizes[i]) {
									tempSequence = app->configuration.paddedSizes[i];
									FFTSizeSelected = 1;
								}
							}
						}
					}
				}
				if (app->configuration.fixMaxRadixBluestein > 0) {
					while (!FFTSizeSelected) {
						pfUINT testSequence = tempSequence;
						for (int i = 0; i < 33; i++) {
							multipliers[i] = 0;
						}
						for (int i = 2; i < app->configuration.fixMaxRadixBluestein + 1; i++) {
							if (testSequence % i == 0) {
								testSequence /= i;
								multipliers[i]++;
								i--;
							}
						}
						if (testSequence == 1) FFTSizeSelected = 1;
						else tempSequence++;
					}
				}
				else {
					while (!FFTSizeSelected) {
						if (axis_id == nonStridedAxisId) {
							if ((FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] < 128) || ((((int)pow(2, (int)pfceil(log2(tempSequence))) * 0.75) <= tempSequence) && (((int)pow(2, (int)pfceil(log2(tempSequence))) <= maxSequenceLengthSharedMemory) || ((2 * FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - 1) > maxSequenceLengthSharedMemory))))  tempSequence = (int)pow(2, (int)pfceil(log2(tempSequence)));
						}
						else {
							int maxSequenceLengthSharedMemoryStrided_temp = (app->configuration.coalescedMemory > complexSize) ? usedSharedMemory / ((int)app->configuration.coalescedMemory) : usedSharedMemory / complexSize;
							if ((FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] < 128) || ((((int)pow(2, (int)pfceil(log2(tempSequence))) * 0.75) <= tempSequence) && (((int)pow(2, (int)pfceil(log2(tempSequence))) <= maxSequenceLengthSharedMemoryStrided_temp) || ((2 * FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - 1) > maxSequenceLengthSharedMemoryStrided_temp))))  tempSequence = (int)pow(2, (int)pfceil(log2(tempSequence)));
						}
						pfUINT testSequence = tempSequence;
						for (int i = 0; i < 33; i++) {
							multipliers[i] = 0;
						}
						for (int i = 2; i < 8; i++) {
							if (testSequence % i == 0) {
								testSequence /= i;
								multipliers[i]++;
								i--;
							}
						}
						if (testSequence != 1) tempSequence++;
						else {
							int registers_per_thread_per_radix[33];
							int registers_per_thread = 0;
							int min_registers_per_thread = 10000000;
							int isGoodSequence = 0;
							res = VkFFTGetRegistersPerThread(app, (int)tempSequence, 0, max_rhs / tempSequence, axes->specializationConstants.useRader, multipliers, registers_per_thread_per_radix, &registers_per_thread, &min_registers_per_thread, &isGoodSequence);
							if (res != VKFFT_SUCCESS) return res;
							if (isGoodSequence) FFTSizeSelected = 1;
							else tempSequence++;
						}
					}
				}
				FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] = tempSequence;
			}
		}

		if (app->configuration.forceBluesteinSequenceSize) FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] = app->configuration.forceBluesteinSequenceSize;

		if ((FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] & (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - 1)) == 0) {
			usedSharedMemory = (int)app->configuration.sharedMemorySizePow2;
			maxSequenceLengthSharedMemory = usedSharedMemory / complexSize;
			maxSingleSizeNonStrided = maxSequenceLengthSharedMemory;
		}
	}
	int isPowOf2 = (pow(2, (int)log2(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id])) == FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]) ? 1 : 0;
	int locNumBatches = (app->configuration.numberBatches > app->actualNumBatches) ? (int)app->configuration.numberBatches : (int)app->actualNumBatches;
	//return VKFFT_ERROR_UNSUPPORTED_RADIX;
	int registerBoost = 1;
	for (int i = 1; i <= app->configuration.registerBoost; i++) {
		if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] % (i * i) == 0)
			registerBoost = i;
	}
	if ((axis_id == nonStridedAxisId) && (!app->configuration.performConvolution)) maxSingleSizeNonStrided *= registerBoost;
	int maxSequenceLengthSharedMemoryStrided = (app->configuration.coalescedMemory > complexSize) ? usedSharedMemory / ((int)app->configuration.coalescedMemory) : usedSharedMemory / complexSize;
	int maxSingleSizeStrided = (!app->configuration.performConvolution) ? maxSequenceLengthSharedMemoryStrided * registerBoost : maxSequenceLengthSharedMemoryStrided;
	int numPasses = 1;
	int numPassesHalfBandwidth = 1;
	pfUINT temp;
	temp = (axis_id == nonStridedAxisId) ? (pfUINT)pfceil(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (double)maxSingleSizeNonStrided) : (pfUINT)pfceil(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (double)maxSingleSizeStrided);
	if (temp > 1) {//more passes than one
		for (int i = 1; i <= app->configuration.registerBoost4Step; i++) {
			if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] % (i * i) == 0) {
				registerBoost = i;
			}
		}
		if ((!app->configuration.performConvolution)) maxSingleSizeNonStrided = maxSequenceLengthSharedMemory * registerBoost;
		if ((!app->configuration.performConvolution)) maxSingleSizeStrided = maxSequenceLengthSharedMemoryStrided * registerBoost;
		temp = ((axis_id == nonStridedAxisId) && ((!app->configuration.reorderFourStep) || (app->useBluesteinFFT[axis_id]))) ? FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / maxSingleSizeNonStrided : FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / maxSingleSizeStrided;
		if (app->configuration.reorderFourStep && (!app->useBluesteinFFT[axis_id]))
			numPasses = (int)pfceil(log2(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]) / log2(maxSingleSizeStrided));
		else
			numPasses += (int)pfceil(log2(temp) / log2(maxSingleSizeStrided));
	}
	registerBoost = ((axis_id == nonStridedAxisId) && ((app->useBluesteinFFT[axis_id]) || (!app->configuration.reorderFourStep) || (numPasses == 1))) ? (int)pfceil(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (double)(pow(maxSequenceLengthSharedMemoryStrided, numPasses - 1) * maxSequenceLengthSharedMemory)) : (int)pfceil(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (double)pow(maxSequenceLengthSharedMemoryStrided, numPasses));
	int canBoost = 0;
	for (int i = registerBoost; i <= app->configuration.registerBoost; i++) {
		if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] % (i * i) == 0) {
			registerBoost = i;
			i = (int)app->configuration.registerBoost + 1;
			canBoost = 1;
		}
	}
	if (((canBoost == 0) || (((FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] & (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - 1)) != 0) && (!app->configuration.registerBoostNonPow2))) && (registerBoost > 1)) {
		registerBoost = 1;
		numPasses++;
	}
	maxSingleSizeNonStrided = maxSequenceLengthSharedMemory * registerBoost;
	maxSingleSizeStrided = maxSequenceLengthSharedMemoryStrided * registerBoost;
	int maxSingleSizeStridedHalfBandwidth = maxSingleSizeStrided;
	if ((axes->specializationConstants.performBandwidthBoost)) {
		maxSingleSizeStridedHalfBandwidth = (app->configuration.coalescedMemory / axes->specializationConstants.performBandwidthBoost > complexSize) ? usedSharedMemory / ((int)app->configuration.coalescedMemory / axes->specializationConstants.performBandwidthBoost) : usedSharedMemory / complexSize;
		temp = (axis_id == nonStridedAxisId) ? (int)pfceil(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (double)maxSingleSizeNonStrided) : (int)pfceil(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (double)maxSingleSizeStridedHalfBandwidth);
		//temp = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / maxSingleSizeNonStrided;
		if (temp > 1) {//more passes than two
			temp = ((!app->configuration.reorderFourStep) || (app->useBluesteinFFT[axis_id])) ? (int)pfceil(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (double)maxSingleSizeNonStrided) : (int)pfceil(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (double)maxSingleSizeStridedHalfBandwidth);
			for (int i = 0; i < 5; i++) {
				temp = (int)pfceil(temp / (double)maxSingleSizeStrided);
				numPassesHalfBandwidth++;
				if (temp == 1) i = 5;
			}
			/*
			temp = ((axis_id == 0) && (!app->configuration.reorderFourStep)) ? FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / maxSingleSizeNonStrided : FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / maxSingleSizeStridedHalfBandwidth;

			if (app->configuration.reorderFourStep)
				numPassesHalfBandwidth = (int)pfceil(log2(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]) / log2(maxSingleSizeStridedHalfBandwidth));
			else
				numPassesHalfBandwidth = 1 + (int)pfceil(log2(temp) / log2(maxSingleSizeStridedHalfBandwidth));
			if ((numPassesHalfBandwidth == 2)&& (!app->configuration.reorderFourStep)&&(registerBoost>1)) //switch back for two step and don't do half bandwidth on strided accesses if register boost and no 4-step reordering
			*/
		}
		if (numPassesHalfBandwidth < numPasses) numPasses = numPassesHalfBandwidth;
		else maxSingleSizeStridedHalfBandwidth = maxSingleSizeStrided;
	}
	if ((FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] >= app->configuration.swapTo2Stage4Step) && (numPasses < 3)) numPasses = 2;//Force set to 2 stage 4 step algorithm
	if ((FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] >= app->configuration.swapTo3Stage4Step) && (app->configuration.swapTo3Stage4Step >= 65536)) numPasses = 3;//Force set to 3 stage 4 step algorithm
	if (forceRaderTwoUpload && (numPasses == 1)) numPasses = 2;//Force set Rader cases that use more than 512 or maxNumThreads threads per one of Rader primes
	pfUINT* locAxisSplit = FFTPlan->axisSplit[axis_id];
	if (numPasses == 1) {
		locAxisSplit[0] = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id];
	}
	if (numPasses == 2) {
		if (isPowOf2 && (!((app->configuration.vendorID == 0x10DE) && (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] > 262144)))) {
			if ((axis_id == nonStridedAxisId) && ((!app->configuration.reorderFourStep) || (app->useBluesteinFFT[axis_id]))) {
				int maxPow8SharedMemory = (int)pow(8, ((int)log2(maxSequenceLengthSharedMemory)) / 3);
				//unit stride
				if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / maxPow8SharedMemory <= maxSingleSizeStrided) {
					locAxisSplit[0] = maxPow8SharedMemory;
				}
				else {
					if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / maxSequenceLengthSharedMemory <= maxSingleSizeStrided) {
						locAxisSplit[0] = maxSequenceLengthSharedMemory;
					}
					else {
						if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (maxSequenceLengthSharedMemory * registerBoost) < maxSingleSizeStridedHalfBandwidth) {
							for (int i = 1; i <= (int)log2(registerBoost); i++) {
								if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (maxSequenceLengthSharedMemory * (int)pow(2, i)) <= maxSingleSizeStrided) {
									locAxisSplit[0] = (maxSequenceLengthSharedMemory * (int)pow(2, i));
									i = (int)log2(registerBoost) + 1;
								}
							}
						}
						else {
							locAxisSplit[0] = (maxSequenceLengthSharedMemory * registerBoost);
						}
					}
				}
			}
			else {
				int maxPow8Strided = (int)pow(8, ((int)log2(maxSingleSizeStrided)) / 3);
				if (maxPow8Strided > 512) maxPow8Strided = 512;
				//all FFTs are considered as non-unit stride
				if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / maxPow8Strided <= maxSingleSizeStrided) {
					locAxisSplit[0] = maxPow8Strided;
				}
				else {
					if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / maxSingleSizeStrided < maxSingleSizeStridedHalfBandwidth) {
						locAxisSplit[0] = maxSingleSizeStrided;
					}
					else {
						locAxisSplit[0] = maxSingleSizeStridedHalfBandwidth;
					}
				}
			}
			locAxisSplit[1] = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / locAxisSplit[0];
			if (locAxisSplit[1] < 64) {
				locAxisSplit[0] = (locAxisSplit[1] == 0) ? locAxisSplit[0] / (64) : locAxisSplit[0] / (64 / locAxisSplit[1]);
				locAxisSplit[1] = 64;
			}
			if (locAxisSplit[1] > locAxisSplit[0]) {
				int swap = (int)locAxisSplit[0];
				locAxisSplit[0] = locAxisSplit[1];
				locAxisSplit[1] = swap;
			}
		}
		else {
			int successSplit = 0;
			if ((axis_id == nonStridedAxisId) && ((!app->configuration.reorderFourStep) || (app->useBluesteinFFT[axis_id]))) {
				/*for (int i = 0; i < maxSequenceLengthSharedMemory; i++) {
					if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] % (maxSequenceLengthSharedMemory - i) == 0) {
						if (((maxSequenceLengthSharedMemory - i) <= maxSequenceLengthSharedMemory) && (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (maxSequenceLengthSharedMemory - i) <= maxSingleSizeStrided)) {
							locAxisSplit[0] = (maxSequenceLengthSharedMemory - i);
							locAxisSplit[1] = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (maxSequenceLengthSharedMemory - i);
							i = maxSequenceLengthSharedMemory;
							successSplit = 1;
						}
					}
				}*/
				int sqrtSequence = (int)pfceil(sqrt(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]));
				for (int i = 0; i < sqrtSequence; i++) {
					if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] % (sqrtSequence - i) == 0) {
						if ((sqrtSequence - i <= maxSingleSizeStrided) && (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (sqrtSequence - i) <= maxSequenceLengthSharedMemory)) {
							locAxisSplit[0] = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (sqrtSequence - i);
							locAxisSplit[1] = sqrtSequence - i;
							i = sqrtSequence;
							successSplit = 1;
						}
					}
				}
			}
			else {
				int sqrtSequence = (int)pfceil(sqrt(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]));
				for (int i = 0; i < sqrtSequence; i++) {
					if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] % (sqrtSequence - i) == 0) {
						if ((sqrtSequence - i <= maxSingleSizeStrided) && (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (sqrtSequence - i) <= maxSingleSizeStridedHalfBandwidth)) {
							locAxisSplit[0] = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (sqrtSequence - i);
							locAxisSplit[1] = sqrtSequence - i;
							i = sqrtSequence;
							successSplit = 1;
						}
					}
				}
			}
			if (successSplit == 0)
				numPasses = 3;
		}
	}
	if (numPasses == 3) {
		if (isPowOf2 && (!((app->configuration.vendorID == 0x10DE) && (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] > 262144)))) {
			int maxPow8Strided = (int)pow(8, ((int)log2(maxSingleSizeStrided)) / 3);
			if ((axis_id == nonStridedAxisId) && ((!app->configuration.reorderFourStep) || (app->useBluesteinFFT[axis_id]))) {
				//unit stride
				int maxPow8SharedMemory = (int)pow(8, ((int)log2(maxSequenceLengthSharedMemory)) / 3);
				if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / maxPow8SharedMemory <= maxPow8Strided * maxPow8Strided)
					locAxisSplit[0] = maxPow8SharedMemory;
				else {
					if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / maxSequenceLengthSharedMemory <= maxSingleSizeStrided * maxSingleSizeStrided)
						locAxisSplit[0] = maxSequenceLengthSharedMemory;
					else {
						if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (maxSequenceLengthSharedMemory * registerBoost) <= maxSingleSizeStrided * maxSingleSizeStrided) {
							for (int i = 0; i <= (int)log2(registerBoost); i++) {
								if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (maxSequenceLengthSharedMemory * (int)pow(2, i)) <= maxSingleSizeStrided * maxSingleSizeStrided) {
									locAxisSplit[0] = (maxSequenceLengthSharedMemory * (int)pow(2, i));
									i = (int)log2(registerBoost) + 1;
								}
							}
						}
						else {
							locAxisSplit[0] = (maxSequenceLengthSharedMemory * registerBoost);
						}
					}
				}
			}
			else {
				//to account for TLB misses, it is best to coalesce the unit-strided stage to 128 bytes
				/*int log2axis = (int)log2(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]);
				locAxisSplit[0] = (int)pow(2, (int)log2axis / 3);
				if (log2axis % 3 > 0) locAxisSplit[0] *= 2;
				locAxisSplit[1] = (int)pow(2, (int)log2axis / 3);
				if (log2axis % 3 > 1) locAxisSplit[1] *= 2;
				locAxisSplit[2] = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / locAxisSplit[0] / locAxisSplit[1];*/
				int maxSingleSizeStrided128 = usedSharedMemory / (128);
				int maxPow8_128 = (int)pow(8, ((int)log2(maxSingleSizeStrided128)) / 3);
				//unit stride
				if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / maxPow8_128 <= maxPow8Strided * maxSingleSizeStrided)
					locAxisSplit[0] = maxPow8_128;
				//non-unit stride
				else {

					if ((FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (maxPow8_128 * 2) <= maxPow8Strided * maxSingleSizeStrided) && (maxPow8_128 * 2 <= maxSingleSizeStrided128)) {
						locAxisSplit[0] = maxPow8_128 * 2;
					}
					else {
						if ((FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (maxPow8_128 * 4) <= maxPow8Strided * maxSingleSizeStrided) && (maxPow8_128 * 4 <= maxSingleSizeStrided128)) {
							locAxisSplit[0] = maxPow8_128 * 4;
						}
						else {
							if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / maxSingleSizeStrided <= maxSingleSizeStrided * maxSingleSizeStrided) {
								for (int i = 0; i <= (int)log2(maxSingleSizeStrided / maxSingleSizeStrided128); i++) {
									if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (maxSingleSizeStrided128 * (int)pow(2, i)) <= maxSingleSizeStrided * maxSingleSizeStrided) {
										locAxisSplit[0] = (maxSingleSizeStrided128 * (int)pow(2, i));
										i = (int)log2(maxSingleSizeStrided / maxSingleSizeStrided128) + 1;
									}
								}
							}
							else
								locAxisSplit[0] = maxSingleSizeStridedHalfBandwidth;
						}
					}
				}
			}
			if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / locAxisSplit[0] < maxPow8Strided) {
				locAxisSplit[1] = (int)pow(2, (int)(log2(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / locAxisSplit[0]) / 2));
				locAxisSplit[2] = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / locAxisSplit[0] / locAxisSplit[1];
			}
			else {
				if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / locAxisSplit[0] / maxPow8Strided <= maxSingleSizeStrided) {
					locAxisSplit[1] = maxPow8Strided;
					locAxisSplit[2] = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / locAxisSplit[1] / locAxisSplit[0];
				}
				else {
					if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / locAxisSplit[0] / maxSingleSizeStrided <= maxSingleSizeStrided) {
						locAxisSplit[1] = maxSingleSizeStrided;
						locAxisSplit[2] = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / locAxisSplit[1] / locAxisSplit[0];
					}
					else {
						locAxisSplit[1] = maxSingleSizeStridedHalfBandwidth;
						locAxisSplit[2] = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / locAxisSplit[1] / locAxisSplit[0];
					}
				}
				if (locAxisSplit[2] < 64) {
					locAxisSplit[1] = (locAxisSplit[2] == 0) ? locAxisSplit[1] / (64) : locAxisSplit[1] / (64 / locAxisSplit[2]);
					locAxisSplit[2] = 64;
				}
			}
			if (locAxisSplit[2] > locAxisSplit[1]) {
				int swap = (int)locAxisSplit[1];
				locAxisSplit[1] = locAxisSplit[2];
				locAxisSplit[2] = swap;
			}
		}
		else {
			int successSplit = 0;
			if ((axis_id == nonStridedAxisId) && ((!app->configuration.reorderFourStep) || (app->useBluesteinFFT[axis_id]))) {
				for (int i = 0; i < maxSequenceLengthSharedMemory; i++) {
					if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] % (maxSequenceLengthSharedMemory - i) == 0) {
						int sqrt3Sequence = (int)pfceil(sqrt(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (maxSequenceLengthSharedMemory - i)));
						for (int j = 0; j < sqrt3Sequence; j++) {
							if ((FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (maxSequenceLengthSharedMemory - i)) % (sqrt3Sequence - j) == 0) {
								if (((maxSequenceLengthSharedMemory - i) <= maxSequenceLengthSharedMemory) && (sqrt3Sequence - j <= maxSingleSizeStrided) && (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (maxSequenceLengthSharedMemory - i) / (sqrt3Sequence - j) <= maxSingleSizeStrided)) {
									locAxisSplit[0] = (maxSequenceLengthSharedMemory - i);
									locAxisSplit[1] = sqrt3Sequence - j;
									locAxisSplit[2] = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (maxSequenceLengthSharedMemory - i) / (sqrt3Sequence - j);
									i = maxSequenceLengthSharedMemory;
									j = sqrt3Sequence;
									successSplit = 1;
								}
							}
						}
					}
				}
			}
			else {
				int sqrt3Sequence = (int)pfceil(pow(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id], 1.0 / 3.0));
				for (int i = 0; i < sqrt3Sequence; i++) {
					if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] % (sqrt3Sequence - i) == 0) {
						int sqrt2Sequence = (int)pfceil(sqrt(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (sqrt3Sequence - i)));
						for (int j = 0; j < sqrt2Sequence; j++) {
							if ((FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (sqrt3Sequence - i)) % (sqrt2Sequence - j) == 0) {
								if ((sqrt3Sequence - i <= maxSingleSizeStrided) && (sqrt2Sequence - j <= maxSingleSizeStrided) && (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (sqrt3Sequence - i) / (sqrt2Sequence - j) <= maxSingleSizeStridedHalfBandwidth)) {
									locAxisSplit[0] = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (sqrt3Sequence - i) / (sqrt2Sequence - j);
									locAxisSplit[1] = sqrt3Sequence - i;
									locAxisSplit[2] = sqrt2Sequence - j;
									i = sqrt3Sequence;
									j = sqrt2Sequence;
									successSplit = 1;
								}
							}
						}
					}
				}
			}
			if (successSplit == 0)
				numPasses = 4;
		}
	}
	if (numPasses > 3) {
		//printf("sequence length exceeds boundaries\n");
		return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;
	}
	if ((numPasses > 1) && ((app->configuration.performDCT > 0) || (app->configuration.performDST > 0))) {
		//printf("sequence length exceeds boundaries\n");
		//return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH_R2R;
	}
	if ((numPasses > 1) && (app->configuration.performR2C > 0) && (axis_id == 0) && (app->configuration.size[axis_id] % 2 != 0)) {
		//printf("sequence length exceeds boundaries\n");
		//return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH_R2C;
	}
	pfUINT tempBufferSize = 0;
	if ((app->configuration.performR2C) && (axis_id == 0)) {
		if (FFTPlan->bigSequenceEvenR2C) {
			tempBufferSize = (FFTPlan->actualFFTSizePerAxis[axis_id][0] + 1) * app->configuration.coordinateFeatures * locNumBatches * app->configuration.numberKernels * complexSize;
		}
		else {
			tempBufferSize = (FFTPlan->actualFFTSizePerAxis[axis_id][0]) * app->configuration.coordinateFeatures * locNumBatches * app->configuration.numberKernels * complexSize;
		}
		for (int i = 1; i < app->configuration.FFTdim; i++)
			tempBufferSize *= FFTPlan->actualFFTSizePerAxis[axis_id][i];
	}
	else {
		tempBufferSize = FFTPlan->actualFFTSizePerAxis[axis_id][0] * app->configuration.coordinateFeatures * locNumBatches * app->configuration.numberKernels * complexSize;
		for (int i = 1; i < app->configuration.FFTdim; i++)
			tempBufferSize *= FFTPlan->actualFFTSizePerAxis[axis_id][i];
	}
	if (app->useBluesteinFFT[axis_id]) {
		if ((app->configuration.performR2C) && (axis_id == 0)) {
			pfUINT tempSize = 0;
			if (FFTPlan->bigSequenceEvenR2C) {
				tempSize = (FFTPlan->actualFFTSizePerAxis[axis_id][0] + 1) * app->configuration.coordinateFeatures * locNumBatches * app->configuration.numberKernels * complexSize;	
			}
			else {
				tempSize = (FFTPlan->actualFFTSizePerAxis[axis_id][0]) * app->configuration.coordinateFeatures * locNumBatches * app->configuration.numberKernels * complexSize;	
			}
			for (int i = 1; i < app->configuration.FFTdim; i++)
					tempSize *= FFTPlan->actualFFTSizePerAxis[axis_id][i];
		
			if (tempSize > tempBufferSize) tempBufferSize = tempSize;
		}
		else {
			pfUINT tempSize = FFTPlan->actualFFTSizePerAxis[axis_id][0] * app->configuration.coordinateFeatures * locNumBatches * app->configuration.numberKernels * complexSize;
			for (int i = 1; i < app->configuration.FFTdim; i++)
				tempSize *= FFTPlan->actualFFTSizePerAxis[axis_id][i];
		
			if (tempSize > tempBufferSize) tempBufferSize = tempSize;
		}
	}
	if (tempBufferSize > app->configuration.tempBufferSize[0]) {
		if (app->configuration.userTempBuffer)
			return VKFFT_ERROR_INVALID_user_tempBuffer_too_small;
		app->configuration.tempBufferSize[0] = tempBufferSize;
	}
	if (((app->configuration.reorderFourStep) && (!app->useBluesteinFFT[axis_id]))) {
		for (int i = 0; i < numPasses; i++) {
			if ((locAxisSplit[0] % 2 != 0) && (locAxisSplit[i] % 2 == 0)) {
				int swap = (int)locAxisSplit[0];
				locAxisSplit[0] = locAxisSplit[i];
				locAxisSplit[i] = swap;
			}
		}
		for (int i = 0; i < numPasses; i++) {
			if ((locAxisSplit[0] % 4 != 0) && (locAxisSplit[i] % 4 == 0)) {
				int swap = (int)locAxisSplit[0];
				locAxisSplit[0] = locAxisSplit[i];
				locAxisSplit[i] = swap;
			}
		}
		for (int i = 0; i < numPasses; i++) {
			if ((locAxisSplit[0] % 8 != 0) && (locAxisSplit[i] % 8 == 0)) {
				int swap = (int)locAxisSplit[0];
				locAxisSplit[0] = locAxisSplit[i];
				locAxisSplit[i] = swap;
			}
		}
	}
	FFTPlan->numAxisUploads[axis_id] = numPasses;
	for (int k = 0; k < numPasses; k++) {
		tempSequence = locAxisSplit[k];
		int loc_multipliers[33]; //split the smaller sequence
		//split the smaller sequence
		//int rader_multipliers[20]; //split the smaller sequence
		//int* rader_generator = axes[k].specializationConstants.rader_generator_sorted; //split the smaller sequence
		//int* rader_primes = axes[k].specializationConstants.rader_primes;

		for (int i = 0; i < 33; i++) {
			loc_multipliers[i] = 0;
		}

		for (int i = 2; i < app->configuration.fixMinRaderPrimeMult; i++) {
			if (tempSequence % i == 0) {
				tempSequence /= i;
				loc_multipliers[i]++;
				i--;
			}
		}
		axes[k].specializationConstants.useRader = 0;
		axes[k].specializationConstants.useRaderMult = 0;
		axes[k].specializationConstants.useRaderFFT = 0;
		if (tempSequence != 1) {
			res = VkFFTConstructRaderTree(app, &axes[k].specializationConstants.raderContainer, &tempSequence, &axes[k].specializationConstants.numRaderPrimes, (int)(locAxisSplit[k] / tempSequence));
			if (res != VKFFT_SUCCESS) return res;
		}
		
		for (pfINT i = 0; i < (pfINT)axes[k].specializationConstants.numRaderPrimes; i++) {
			if (axes[k].specializationConstants.raderContainer[i].type == 0) {
				if (axes[k].specializationConstants.useRaderFFT < axes[k].specializationConstants.raderContainer[i].prime) axes[k].specializationConstants.useRaderFFT = axes[k].specializationConstants.raderContainer[i].prime;
				if (axes[k].specializationConstants.raderContainer[i].containerFFTNum > app->configuration.maxThreadsNum) return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;
			}
			else {
				if (axes[k].specializationConstants.useRaderMult < axes[k].specializationConstants.raderContainer[i].prime) axes[k].specializationConstants.useRaderMult = axes[k].specializationConstants.raderContainer[i].prime;
			}
		}
		if (axes[k].specializationConstants.useRaderMult) {
			app->configuration.useLUT = 1; // workaround, Mult Rader is better with LUT
		}

		axes[k].specializationConstants.useRader = axes[k].specializationConstants.numRaderPrimes;

		if ((axes[k].specializationConstants.useRader) && (app->configuration.useRaderUintLUT)) {
			app->configuration.useLUT = 1; // useRaderUintLUT forces LUT
		}

		int registers_per_thread_per_radix[33];
		int registers_per_thread = 0;
		int min_registers_per_thread = 10000000;
		int isGoodSequence = 0;
		int extraSharedMemoryForPow2 = ((app->configuration.sharedMemorySizePow2 < app->configuration.sharedMemorySize) || ((locAxisSplit[k] < maxSingleSizeNonStrided) && ((axis_id == nonStridedAxisId))) || ((locAxisSplit[k] < maxSingleSizeStrided) && ((axis_id != nonStridedAxisId)))) ? 1 : 0;

		res = VkFFTGetRegistersPerThread(app, (int)locAxisSplit[k], extraSharedMemoryForPow2, max_rhs / locAxisSplit[k], axes[k].specializationConstants.numRaderPrimes, loc_multipliers, registers_per_thread_per_radix, &registers_per_thread, &min_registers_per_thread, &isGoodSequence);
		if (res != VKFFT_SUCCESS) return res;
		//first optimizer pass
		if (axes[k].specializationConstants.numRaderPrimes) {
			res = VkFFTOptimizeRaderFFTRegisters(axes[k].specializationConstants.raderContainer, axes[k].specializationConstants.numRaderPrimes, (int)locAxisSplit[k], &min_registers_per_thread, &registers_per_thread, registers_per_thread_per_radix);
			if (res != VKFFT_SUCCESS) return res;
		}

		if ((registerBoost == 4) && (registers_per_thread % 4 != 0)) {
			registers_per_thread *= 2;
			for (int i = 2; i < 33; i++) {
				registers_per_thread_per_radix[i] *= 2;
			}
			min_registers_per_thread *= 2;
		}
		int maxBatchCoalesced = ((axis_id == 0) && (((k == 0) && ((!app->configuration.reorderFourStep) || (app->useBluesteinFFT[axis_id]))) || (numPasses == 1))) ? 1 : (int)(app->configuration.coalescedMemory / complexSize);
		int estimate_rader_threadnum = 0;
		int scale_registers_rader = 0;
		int rader_min_registers = min_registers_per_thread;

		if (axes[k].specializationConstants.useRaderMult) {
			for (pfINT i = 0; i < (pfINT)axes[k].specializationConstants.numRaderPrimes; i++) {
				if (axes[k].specializationConstants.raderContainer[i].type == 1) {
					int temp_rader = (int)pfceil((locAxisSplit[k] / (double)((rader_min_registers / 2 + scale_registers_rader) * 2)) / (double)((axes[k].specializationConstants.raderContainer[i].prime + 1) / 2));
					int active_rader = (int)pfceil((locAxisSplit[k] / axes[k].specializationConstants.raderContainer[i].prime) / (double)temp_rader);
					if (active_rader > 1) {
						if ((((double)active_rader - (locAxisSplit[k] / axes[k].specializationConstants.raderContainer[i].prime) / (double)temp_rader) >= 0.5) && ((((int)pfceil((locAxisSplit[k] / axes[k].specializationConstants.raderContainer[i].prime) / (double)(active_rader - 1)) * ((axes[k].specializationConstants.raderContainer[i].prime + 1) / 2)) * maxBatchCoalesced) <= app->configuration.maxThreadsNum)) active_rader--;
					}

					int local_estimate_rader_threadnum = (int)pfceil((locAxisSplit[k] / axes[k].specializationConstants.raderContainer[i].prime) / (double)active_rader) * ((axes[k].specializationConstants.raderContainer[i].prime + 1) / 2) * maxBatchCoalesced;
					if ((maxBatchCoalesced * locAxisSplit[k] / ((rader_min_registers / 2 + scale_registers_rader) * 2 * registerBoost)) > local_estimate_rader_threadnum) local_estimate_rader_threadnum = (maxBatchCoalesced * (int)locAxisSplit[k] / ((rader_min_registers / 2 + scale_registers_rader) * 2 * registerBoost));
					if ((local_estimate_rader_threadnum > app->configuration.maxThreadsNum) || ((((locAxisSplit[k] / min_registers_per_thread) > 256) || (local_estimate_rader_threadnum > 256)) && (((rader_min_registers / 2 + scale_registers_rader) * 2) <= 4))) {
						scale_registers_rader++;
						i = -1;
					}
					else {
						estimate_rader_threadnum = (estimate_rader_threadnum < local_estimate_rader_threadnum) ? local_estimate_rader_threadnum : estimate_rader_threadnum;
					}
				}
			}
			rader_min_registers = (rader_min_registers / 2 + scale_registers_rader) * 2;//min number of registers for Rader (can be more than min_registers_per_thread, but min_registers_per_thread should be at least 4 for Nvidiaif you have >256 threads)
			if (registers_per_thread < rader_min_registers) registers_per_thread = rader_min_registers;
			for (int i = 2; i < 33; i++) {
				if (registers_per_thread_per_radix[i] != 0) {
					if (registers_per_thread / registers_per_thread_per_radix[i] >= 2) {
						registers_per_thread_per_radix[i] *= (registers_per_thread / registers_per_thread_per_radix[i]);
					}
				}
			}

			for (pfINT i = 0; i < (pfINT)axes[k].specializationConstants.numRaderPrimes; i++) {
				if (axes[k].specializationConstants.raderContainer[i].type == 0) {
					for (int j = 2; j < 33; j++) {
						if (axes[k].specializationConstants.raderContainer[i].registers_per_thread_per_radix[j] != 0) {
							if (registers_per_thread / axes[k].specializationConstants.raderContainer[i].registers_per_thread_per_radix[j] >= 2) {
								axes[k].specializationConstants.raderContainer[i].registers_per_thread_per_radix[j] *= (registers_per_thread / axes[k].specializationConstants.raderContainer[i].registers_per_thread_per_radix[j]);
							}
						}
					}
				}
			}
			int new_min_registers = 10000000;
			for (int i = 2; i < 33; i++) {
				if ((registers_per_thread_per_radix[i] > 0) && (registers_per_thread_per_radix[i] < new_min_registers)) new_min_registers = registers_per_thread_per_radix[i];
				if (registers_per_thread_per_radix[i] > registers_per_thread) {
					registers_per_thread = registers_per_thread_per_radix[i];
				}
			}
			for (pfINT i = 0; i < (pfINT)axes[k].specializationConstants.numRaderPrimes; i++) {
				if (axes[k].specializationConstants.raderContainer[i].type == 0) {
					for (int j = 2; j < 33; j++) {
						if ((axes[k].specializationConstants.raderContainer[i].registers_per_thread_per_radix[j] > 0) && (axes[k].specializationConstants.raderContainer[i].registers_per_thread_per_radix[j] < new_min_registers)) new_min_registers = axes[k].specializationConstants.raderContainer[i].registers_per_thread_per_radix[j];
						if (axes[k].specializationConstants.raderContainer[i].registers_per_thread_per_radix[j] > registers_per_thread) {
							registers_per_thread = axes[k].specializationConstants.raderContainer[i].registers_per_thread_per_radix[j];
						}
					}
				}
			}
			min_registers_per_thread = (new_min_registers == 1e7) ? registers_per_thread : new_min_registers;
		}
		if ((int)pfceil((maxBatchCoalesced * locAxisSplit[k] / (double)(min_registers_per_thread * registerBoost)) > app->configuration.maxThreadsNum) || (axes[k].specializationConstants.useRader && (estimate_rader_threadnum > app->configuration.maxThreadsNum)))
		{
			int scaleRegistersNum = 1;
			if ((axis_id == 0) && (k == 0) && (maxBatchCoalesced > 1)) {
				maxBatchCoalesced = (int)(app->configuration.maxThreadsNum * (min_registers_per_thread * registerBoost) / locAxisSplit[k]);
				if (maxBatchCoalesced < 1) maxBatchCoalesced = 1;
			}
			if (((int)pfceil(maxBatchCoalesced * locAxisSplit[k] / (double)(min_registers_per_thread * registerBoost * scaleRegistersNum))) > app->configuration.maxThreadsNum) {
				for (int i = 2; i < locAxisSplit[k]; i++) {
					if ((((int)pfceil(maxBatchCoalesced * locAxisSplit[k] / (double)(min_registers_per_thread * registerBoost * i))) <= app->configuration.maxThreadsNum)) {
						scaleRegistersNum = i;
						i = (int)locAxisSplit[k];
					}
				}
			}
			min_registers_per_thread *= scaleRegistersNum;
			registers_per_thread *= scaleRegistersNum;
			for (int i = 2; i < 33; i++) {
				if (registers_per_thread_per_radix[i] != 0) {
					registers_per_thread_per_radix[i] *= scaleRegistersNum;
				}
			}
			int new_min_registers = 10000000;
			for (int i = 2; i < 33; i++) {
				if ((registers_per_thread_per_radix[i] > 0) && (registers_per_thread_per_radix[i] < new_min_registers)) new_min_registers = registers_per_thread_per_radix[i];
			}
			for (pfINT i = 0; i < (pfINT)axes[k].specializationConstants.numRaderPrimes; i++) {
				if (axes[k].specializationConstants.raderContainer[i].type == 0) {
					for (int j = 2; j < 33; j++) {
						if ((axes[k].specializationConstants.raderContainer[i].registers_per_thread_per_radix[j] > 0) && (axes[k].specializationConstants.raderContainer[i].registers_per_thread_per_radix[j] < new_min_registers)) new_min_registers = axes[k].specializationConstants.raderContainer[i].registers_per_thread_per_radix[j];
					}
				}
			}
			if ((int)pfceil((maxBatchCoalesced * locAxisSplit[k] / (double)(new_min_registers * registerBoost))) > app->configuration.maxThreadsNum) {
				// if we get here, there can be trouble with small primes, as we can have one thread do at max one fftDim. This is only an issue for small primes in sequences close to shared memory limit sizes for extremely big shared memory sizes (>136KB)
				for (pfINT i = 0; i < (pfINT)axes[k].specializationConstants.numRaderPrimes; i++) {
					if (axes[k].specializationConstants.raderContainer[i].type == 0) {
						for (int j = 2; j < 33; j++) {
							if (axes[k].specializationConstants.raderContainer[i].registers_per_thread_per_radix[j] != 0) {
								axes[k].specializationConstants.raderContainer[i].registers_per_thread_per_radix[j] *= scaleRegistersNum;
							}
						}
					}
				}
			}
			else {
				min_registers_per_thread = new_min_registers;
			}
			if (min_registers_per_thread > registers_per_thread) {
				temp = min_registers_per_thread;
				min_registers_per_thread = registers_per_thread;
				registers_per_thread = (int)temp;
			}
			for (int i = 2; i < 33; i++) {
				if (registers_per_thread_per_radix[i] > registers_per_thread) {
					registers_per_thread = registers_per_thread_per_radix[i];
				}
				if ((registers_per_thread_per_radix[i] > 0) && (registers_per_thread_per_radix[i] < min_registers_per_thread)) {
					min_registers_per_thread = registers_per_thread_per_radix[i];
				}
			}
			for (pfINT i = 0; i < (pfINT)axes[k].specializationConstants.numRaderPrimes; i++) {
				if (axes[k].specializationConstants.raderContainer[i].type == 0) {
					for (int j = 2; j < 33; j++) {
						if (axes[k].specializationConstants.raderContainer[i].registers_per_thread_per_radix[j] > registers_per_thread) {
							registers_per_thread = axes[k].specializationConstants.raderContainer[i].registers_per_thread_per_radix[j];
						}
						if ((axes[k].specializationConstants.raderContainer[i].registers_per_thread_per_radix[j] > 0) && (axes[k].specializationConstants.raderContainer[i].registers_per_thread_per_radix[j] < min_registers_per_thread)) {
							min_registers_per_thread = axes[k].specializationConstants.raderContainer[i].registers_per_thread_per_radix[j];
						}
					}
				}
			}
			if ((loc_multipliers[3] >= 2) && (((registers_per_thread / min_registers_per_thread) % 3) == 0) && (axes[k].specializationConstants.numRaderPrimes == 0)) {
				registers_per_thread /= 3;
				for (int i = 2; i < 33; i++) {
					if (registers_per_thread_per_radix[i] % 9 == 0) {
						registers_per_thread_per_radix[i] /= 3;
					}
				}
				for (int i = 2; i < 33; i++) {
					if (registers_per_thread_per_radix[i] > registers_per_thread) {
						registers_per_thread = registers_per_thread_per_radix[i];
					}
					if ((registers_per_thread_per_radix[i] > 0) && (registers_per_thread_per_radix[i] < min_registers_per_thread)) {
						min_registers_per_thread = registers_per_thread_per_radix[i];
					}
				}
			}
		}
		//second optimizer pass
		if (axes[k].specializationConstants.numRaderPrimes) {
			res = VkFFTOptimizeRaderFFTRegisters(axes[k].specializationConstants.raderContainer, axes[k].specializationConstants.numRaderPrimes, (int)locAxisSplit[k], &min_registers_per_thread, &registers_per_thread, registers_per_thread_per_radix);
			if (res != VKFFT_SUCCESS) return res;
		}

		axes[k].specializationConstants.maxNonPow2Radix = 1;
		axes[k].specializationConstants.usedLocRegs = 1;

		res = VkFFTOptimizeRadixKernels(registers_per_thread_per_radix, loc_multipliers, registerBoost, &axes[k].specializationConstants.maxNonPow2Radix, &axes[k].specializationConstants.usedLocRegs, axes[k].specializationConstants.raderContainer, axes[k].specializationConstants.numRaderPrimes);
		if (res != VKFFT_SUCCESS) return res;

		for (int i = 2; i < 33; i++) {
			axes[k].specializationConstants.registers_per_thread_per_radix[i] = registers_per_thread_per_radix[i];
		}
		axes[k].specializationConstants.numStages = 0;
		axes[k].specializationConstants.fftDim.type = 31;
		axes[k].specializationConstants.fftDim.data.i = locAxisSplit[k];
		int tempRegisterBoost = registerBoost;// ((axis_id == nonStridedAxisId) && ((!app->configuration.reorderFourStep)||(app->useBluesteinFFT[axis_id]))) ? (int)pfceil(axes[k].specializationConstants.fftDim / (double)maxSingleSizeNonStrided) : (int)pfceil(axes[k].specializationConstants.fftDim / (double)maxSingleSizeStrided);
		int switchRegisterBoost = 0;
		if (tempRegisterBoost > 1) {
			if (loc_multipliers[tempRegisterBoost] > 0) {
				loc_multipliers[tempRegisterBoost]--;
				switchRegisterBoost = tempRegisterBoost;
			}
			else {
				for (int i = 32; i > 1; i--) {
					if (loc_multipliers[i] > 0) {
						loc_multipliers[i]--;
						switchRegisterBoost = i;
						i = 1;
					}
				}
			}
		}

		res = VkFFTGetRaderFFTStages(axes[k].specializationConstants.raderContainer, axes[k].specializationConstants.numRaderPrimes, &axes[k].specializationConstants.numStages, axes[k].specializationConstants.stageRadix, axes[k].specializationConstants.rader_generator);
		if (res != VKFFT_SUCCESS) return res;

		for (int i = 32; i > 1; i--) {
			if (loc_multipliers[i] > 0) {
				axes[k].specializationConstants.stageRadix[axes[k].specializationConstants.numStages] = i;
				loc_multipliers[i]--;
				i++;
				axes[k].specializationConstants.numStages++;
			}
		}
		
		//add more registers for Rader FFT if needed
		if (axes[k].specializationConstants.useRaderMult) {
			axes[k].specializationConstants.rader_min_registers = rader_min_registers;
			for (int i = 0; i < axes[k].specializationConstants.numRaderPrimes; i++) {
				if (axes[k].specializationConstants.raderContainer[i].type == 1) {
					int temp_rader = (int)pfceil((locAxisSplit[k] / (double)axes[k].specializationConstants.rader_min_registers) / (double)((axes[k].specializationConstants.raderContainer[i].prime + 1) / 2));
					int active_rader = (int)pfceil((locAxisSplit[k] / axes[k].specializationConstants.raderContainer[i].prime) / (double)temp_rader);
					if (active_rader > 1) {
						if ((((double)active_rader - (locAxisSplit[k] / axes[k].specializationConstants.raderContainer[i].prime) / (double)temp_rader) >= 0.5) && ((((int)pfceil((locAxisSplit[k] / axes[k].specializationConstants.raderContainer[i].prime) / (double)(active_rader - 1)) * ((axes[k].specializationConstants.raderContainer[i].prime + 1) / 2)) * maxBatchCoalesced) <= app->configuration.maxThreadsNum)) active_rader--;
					}
					axes[k].specializationConstants.raderRegisters = (active_rader * 2 > axes[k].specializationConstants.raderRegisters) ? active_rader * 2 : axes[k].specializationConstants.raderRegisters;
					if (active_rader * 2 > registers_per_thread) registers_per_thread = active_rader * 2;
				}
			}
			if (axes[k].specializationConstants.raderRegisters < axes[k].specializationConstants.rader_min_registers)	axes[k].specializationConstants.raderRegisters = axes[k].specializationConstants.rader_min_registers;
		}

		//final check up on all registers, increase if bigger
		registers_per_thread = 0;
		min_registers_per_thread = 10000000;
		if (axes[k].specializationConstants.useRaderMult) {
			registers_per_thread = axes[k].specializationConstants.raderRegisters;
			min_registers_per_thread = axes[k].specializationConstants.rader_min_registers;
		}
		res = VkFFTMinMaxRegisterCheck(axes[k].specializationConstants.numStages, axes[k].specializationConstants.stageRadix, &min_registers_per_thread, &registers_per_thread, axes[k].specializationConstants.registers_per_thread_per_radix, axes[k].specializationConstants.raderContainer, axes[k].specializationConstants.numRaderPrimes, axes[k].specializationConstants.rader_generator);;
		if (res != VKFFT_SUCCESS) return res;
		axes[k].specializationConstants.minRaderFFTThreadNum = 0;
		res = VkFFTGetRaderFFTThreadsNum(axes[k].specializationConstants.raderContainer, axes[k].specializationConstants.numRaderPrimes, &axes[k].specializationConstants.minRaderFFTThreadNum);
		if (res != VKFFT_SUCCESS) return res;
		axes[k].specializationConstants.registerBoost = registerBoost;
		axes[k].specializationConstants.registers_per_thread = registers_per_thread;
		axes[k].specializationConstants.min_registers_per_thread = min_registers_per_thread;
		if (axes[k].specializationConstants.registers_per_thread == 0) {
			axes[k].specializationConstants.registers_per_thread = 2;
			axes[k].specializationConstants.min_registers_per_thread = 2;
		}
		if (switchRegisterBoost > 0) {
			axes[k].specializationConstants.stageRadix[axes[k].specializationConstants.numStages] = switchRegisterBoost;
			axes[k].specializationConstants.numStages++;
		}
		else {
			//try to read directly to registers
			if (min_registers_per_thread != registers_per_thread) {
				for (int i = 0; i < axes[k].specializationConstants.numStages; i++) {
					if (axes[k].specializationConstants.registers_per_thread_per_radix[axes[k].specializationConstants.stageRadix[i]] == min_registers_per_thread) {
						int stageid = axes[k].specializationConstants.stageRadix[i];
						axes[k].specializationConstants.stageRadix[i] = axes[k].specializationConstants.stageRadix[0];
						axes[k].specializationConstants.stageRadix[0] = stageid;
						if (axes[k].specializationConstants.useRader) {
							stageid = axes[k].specializationConstants.rader_generator[i];
							axes[k].specializationConstants.rader_generator[i] = axes[k].specializationConstants.rader_generator[0];
							axes[k].specializationConstants.rader_generator[0] = stageid;
						}
						i = axes[k].specializationConstants.numStages;
					}
				}
			}
		}
	}
	return VKFFT_SUCCESS;
}
#endif
