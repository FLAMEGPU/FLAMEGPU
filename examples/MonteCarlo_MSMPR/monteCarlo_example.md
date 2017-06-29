# Introduction

Monte Carlo simulation is a method of solving deterministic problems by repeated random sampling. The purpose of this example is to show how multi-agent approach can be applied to Monte Carlo crystallization. We present a general method for implementing Monte Carlo simulation in both batch and continuous systems through two case studies (Batch and MSMPR) implemented within the FLAMEGPU simulation framework. The two case studies have been replicated through parallel implementation of GPU. The examples can be found in _examples/MonteCarlo_MSMPR_ and _examples/MonteCarlo_BATCH_. We used the base concepts similar to Gooch and Hounslow’s paper [^1].

# How to setup, build, and run MonteCarlo examples on Linux

1\. Install all the needed build tools and libraries  

```bash
sudo apt-get install g++ git make libxml2-utils gnuplot
```

Minimum versions:
- g++: 4.8
- cuda: 7.5

3\. Clone the project using Git (it will be stored in the folder "FLAMEGPU"):  

```bash
git clone https://github.com/FLAMEGPU/FLAMEGPU.git
```

Going forward, you will want to pull from the _master_ branch, which will always contain the last known release.

4\. Build the SDK in Release mode (this is the default mode)

- Batch Simulation
```bash
cd FLAMEGPU/examples/MonteCarlo_BATCH
make all
```
- MSMPR Steady State Simulation
```bash
cd FLAMEGPU/examples/MonteCarlo_MSMPR
make all
```

You can build the Debug version by specifying _dbg_ value on the make line instead (`make all dbg=1`). Note that the exacutable is built in Console mode.

5\. To generate charge, simply run the bash script `./generate_charge.sh` in _iterations_ folder. Alternatively, you can manualy compile the code and generate the charge for a different number of population.

```bash
cd iterations/
g++ InputGenerator/inpGen.cpp -o inputGen
./inputGen 0.xml input.dat 10000
```

6\. After building the executable and generating charge, run the example via _make_ and set the iteration number.

Note: If 'arg' is not set, the default value for the number of iterations would be 1. You can simply change this by setting a value. _(e.g: iter=50)_

```bash
cd examples/MonteCarlo_*
make run_console iter='arg'`
```

7\. Plotting the histogram results for the simulation and the input charge.

- Batch Simulation
```bash
cd examples/MonteCarlo_BATCH/iterations
gnuplot plot_cs1.plt
```
- MSMPR Steady State Simulation
```bash
cd examples/MonteCarlo_MSMPR/iterations
gnuplot plot_cs2.plt
```

8\. For the detailed information and the performance results, please view this [document]().

#Problem reports

To report a bug in this documentation or in the software or propose an improvement, please use the FLAMEGPU github issue tracker.
[^1]: John R. van Peborgh Gooch and Michael J. Hounslow. Monte carlo simulation of size-enlargement mechanisms in crystallization.