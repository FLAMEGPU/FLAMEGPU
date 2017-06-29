# FLAME GPU

http://www.flamegpu.com

Current version: `1.5.0` for CUDA 8.0

FLAME GPU (Flexible Large-scale Agent Modelling Environment for Graphics Processing Units) is a high performance Graphics Processing Unit (GPU) extension to the FLAME framework. 
It provides a mapping between a formal agent specifications with C based scripting and optimised CUDA code. 
This includes a number of key ABM building blocks such as multiple agent types, agent communication and birth and death allocation. 
The advantages of our contribution are three fold. 
Firstly Agent Based (AB) modellers are able to focus on specifying agent behaviour and run simulations without explicit understanding of CUDA programming or GPU optimisation strategies. 
Secondly simulation performance is significantly increased in comparison with desktop CPU alternatives. This allows simulation of far larger model sizes with high performance at a fraction of the cost of grid based alternatives. 
Finally massive agent populations can be visualised in real time as agent data is already located on the GPU hardware.

## Documentation

The FLAME GPU technical report and users guide is available at [docs/TechReportAndUserGuide.pdf](https://github.com/FLAMEGPU/FLAMEGPU/blob/master/doc/TechReportAndUserGuide.pdf) and, with the source of this documentation in the [FLAMEGPU/FLAMEGPU_TechnicalReport](https://github.com/FLAMEGPU/FLAMEGPU_TechnicalReport) repository


## Compiling FLAME GPU

### Windows

For windows, Visual Studio 2015 solutions are provided for the example projects, and requires CUDA 8.0.
The relevant example can be built using Visual Studio, with *debug* and *release* configurations available in each solution.

Pre-compiled Windows binaries are available for the example projects in the [FLAME-PGU-SDK](https://github.com/FLAMEGPU/FLAMEGPU/releases), available for each release.

## Linux

Linux compilation is controlled using `make`, with makefiles provided for each example.

# How to setup, build, and run FLAMEGPU examples on Linux

1\. Install [Ubuntu](http://www.ubuntu.com/download) 16.04 or later.  

2\. Install all the needed build tools and libraries  

```bash
sudo apt-get install g++ git make libxml2-utils
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

```bash
cd FLAMEGPU/examples
make all
```

You can build the Debug version by specifying _dbg_ value on the make line instead (`make all dbg=1`).  Moreover, for each example, exacutables can also be built in either Visualisation (`make Visualisation_mode`) or Console (`make Console_mode`) mode.

```bash
cd examples/{folder name}
make XSLTPREP
make Visualisation_mode
# or
make Console_mode
```
_Replace '{folder name}' with the name of the example folder._

5\. After building the executables, you can run the examples by executing the relevant bash script inside the "bin/x64" folder:

- Visualisation mode `./*_vis.sh`

- Console mode `./*_console.sh iter='arg'`

Note: If 'arg' is not set, the default value for the number of iterations would be 1. You can simply change this by setting a value. _(e.g: iter=50)_

Alternatively, run the executables from each example folder (`cd examples/{folder name}`) using _make_.

- Visualisation mode `make run_vis`

- Console mode `make run_console iter='arg'`

6\. Debugging examples:
```bash
cd examples/{folder name}
make Console_mode dbg=1
```
- Debugging with _cuda-gdb_
```bash
cuda-gdb ../../bin/x64/Debug_Console/{folder name}_console
..
(cuda-gdb) run iterations/0.xml 2
...
```
- Debugging with _valgrind_
```bash
valgrind --tool=memcheck ../../bin/x64/Debug_Console/{folder name}_console iterations/0.xml 1
```

6\. For more details on how to build specific targets for each example, run
`make help`


## Usage

FLAME GPU can be executed as either a console application or as an interactive visualisation.
Please see the documentation for further details.

```
# Console mode
executable_console <path/to/0.xml> <iterations> <device_id>

# Interactive visualisation
executable_visualistion <path/to/0.xml> <device_id>
```


## Contributing

To report FLAME GPU bugs or request features, please [file an issue directly using Github](http://github.com/FLAMEGPU/FLAMEGPU/issues).
If you wish to make any contributions, please [issue a Pull Request on Github](https://github.com/FLAMEGPU/FLAMEGPU/pulls).


## Publications

Please cite FLAME GPU using

```
@article{richmond2010high,
  title={High performance cellular level agent-based simulation with FLAME for the GPU},
  author={Richmond, Paul and Walker, Dawn and Coakley, Simon and Romano, Daniela},
  journal={Briefings in bioinformatics},
  volume={11},
  number={3},
  pages={334--347},
  year={2010},
  publisher={Oxford Univ Press}
}
```

**@todo**


## Authors

FLAMNE GPU is developed as an open-source project by the Visual Computing research group in the Department of Computer Science at the University of Sheffield.
The primary author is [Dr Paul Richmond](http://paulrichmond.shef.ac.uk/).

## Acknowledgements

**@todo**

## Copyright and Software Licence

FLAME GPU is copyright the University of Sheffield 2009 - 2017.
The Library, examples and all source code are covered by the Licence in [LICENCE.txt](LICENCE.txt)


## Release Notes

### [1.5.0](https://github.com/FLAMEGPU/FLAMEGPU/releases/tag/v1.5.0)

+ CUDA 8.0 and SM60 support
+ Removed SM20 support
+ Visual Studio 2015
+ **@todo**


### [1.4.3](https://github.com/FLAMEGPU/FLAMEGPU/releases/tag/v1.4.3)

+ Updated Circles Example
+ Purged binaries form history, reducing repository size
+ Updated Visual Studio Project files to 2013
+ Improved Visual Studio build customisation
+ Fixed double precision support within spatial partitioning
+ Compile-time spatial partition config validation

### [1.4.2](https://github.com/FLAMEGPU/FLAMEGPU/releases/tag/v1.4.2)

+ Added support for continuous agents reading discrete messages.

### [1.4.1](https://github.com/FLAMEGPU/FLAMEGPU/releases/tag/v1.4.1)

+ Minor bug fixes and added missing media folder


### [1.4.0](https://github.com/FLAMEGPU/FLAMEGPU/releases/tag/v1.4)

+ FLAME GPU 1.4 for CUDA 7 and Visual Studio 2012

#Problem reports

To report a bug in this documentation or in the software or propose an improvement, please use the FLAMEGPU github issue tracker.