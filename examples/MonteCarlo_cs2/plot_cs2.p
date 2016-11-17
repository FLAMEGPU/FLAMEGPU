# Gnuplot script file for plotting data in file "histogram.dat"
set logscale y
unset label                            # remove any previous labels
#set xtic auto                          # set xtics automatically
#set ytic auto                          # set ytics automatically
set title "Nucleation and size-dependent growth in an MSMPR crystallizer"
set xlabel "crystal size, L"
set ylabel "population, N(L)"
plot    "input.dat" using 1:2 title 'simulation; charge' with points , \
        "histogram_c2.dat" using 1:2 title 'simulation_c2; product' with points

set term png           
set output "MC_cs2.png"
replot
