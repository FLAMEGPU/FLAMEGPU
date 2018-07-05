# Gnuplot script file for plotting data in file "histogram.dat"
set logscale y
unset label                            # remove any previous labels
#set xtic auto                          # set xtics automatically
#set ytic auto                          # set ytics automatically
set title "Selt-preserving PSD in a batch crystallizer with aggregation only"
set xlabel "crystal size, L"
set ylabel "population, N(L)"
plot    "input.dat" using 1:2 title 'simulation; charge' with points , \
        "histogram_c1.dat" using 1:2 title 'simulation_c1; product' with points

set term png           
set output "MC_cs1.png"
replot
