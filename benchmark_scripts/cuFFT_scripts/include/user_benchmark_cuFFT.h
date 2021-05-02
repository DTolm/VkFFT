typedef struct {
	uint64_t X;
	uint64_t Y;
	uint64_t Z;
	uint64_t P;
	uint64_t B;
	uint64_t N;
	uint64_t R2C;
} cuFFTUserSystemParameters;//an example structure used to pass user-defined system for benchmarking

void user_benchmark_cuFFT(bool file_output, FILE* output, cuFFTUserSystemParameters* userParams);