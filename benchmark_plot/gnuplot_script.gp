set title 'VkFFT/cuFFT benchmark'
set xlabel 'system size'
set ylabel 'time, (ms)'

set grid
set key left top
set tics out nomirror
set border 4095 front linetype black linewidth 1.0 dashtype solid

set xtics 1

set style histogram clustered gap 1 title offset character 0, 0, 0
set style data histograms 

set style histogram errorbars linewidth 1 
set errorbars linecolor black

set boxwidth 1 absolute
set style fill solid 5.0 border -1

set terminal png enhanced size 4000, 500 
set output 'benchmark.png'

plot 'cuFFT_benchmark_results.txt' using 8:11:xtic(3) ti "cuFFT" linecolor rgb "#fdc897", ''u 0:8:8 with labels offset -2.5,1 title "", \
	 'VkFFT_benchmark_results.txt' using 8:11:xtic(3) ti "VkFFT" linecolor rgb "#9dd79d", ''u 0:8:8 with labels offset 2.5,1 title ""