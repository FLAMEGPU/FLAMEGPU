# FLAME GPU

http://www.flamegpu.com

Current version: `1.5.0`

FLAME GPU (Flexible Large-scale Agent Modelling Environment for Graphics Processing Units) is a high performance Graphics Processing Unit (GPU) extension to the FLAME framework.

It provides a mapping between a formal agent specifications with C based scripting and optimised CUDA code.
This includes a number of key ABM building blocks such as multiple agent types, agent communication and birth and death allocation.
The advantages of our contribution are three fold.

1. Agent Based (AB) modellers are able to focus on specifying agent behaviour and run simulations without explicit understanding of CUDA programming or GPU optimisation strategies.
2. Simulation performance is significantly increased in comparison with desktop CPU alternatives. This allows simulation of far larger model sizes with high performance at a fraction of the cost of grid based alternatives.
3. Massive agent populations can be visualised in real time as agent data is already located on the GPU hardware.

## Documentation

The FLAME GPU documentation and user guide can be found at [http://dpcs.flamegpu.com](http://dpcs.flamegpu.com), with source hosted on github at [FLAMEGPU/docs](https://github.com/FLAMEGPU/docs).


## Getting FLAME GPU

Pre-compiled Windows binaries are available for the example projects in the [FLAME-GPU-SDK](https://github.com/FLAMEGPU/FLAMEGPU/releases), available as an archive for each release.

Source is available from github, either as a zip download or via `git`:

```
git clone https://github.com/FLAMEGPU/FLAMEGPU.git
```
Or
```
git clone git@github.com:FLAMEGPU/FLAMEGPU.git
```

## Building FLAME GPU

FLAME GPU can be built for Windows and Linux. MacOS *should* work, but is unsupported.

### Dependancies
+ CUDA 8.0 or later
+ Compute Capability 2.0 or greater GPU (CUDA 8)
    + Compute Capability 3.0 or greater GPU (CUDA 9)
+ Windows
    + Microsoft Visual Studio 2015 or later
    + *Visualisation*: **@todo**
    + *Optional*: make
+ Linux
    + `make`
    + `g++` (which supports the cuda version used)
    + `xsltproc`
    + *Visualistion*: **@todo**
    + *Optional*: `xmllint`


### Windows using Visual Studio

Visual Studio 2015 solutions are provided for the example FLAME GPU projects.
*Release* and *Debug* build configurations are provided, for both *console* mode and (optionally) *visualistion* mode.

An additional solution is provided in the `examples` directory, enabling batch building of all examples.

### `make` for Linux and Windows

`make` can be used to build FLAME GPU under linux and windows (via a windows implementation of `make`).

Makefiles are provided for each example project `examples/project/Makefile`), and for batch building all examples (`examples/Makefile`).

To build a console example in release mode:

```
cd examples/EmptyExample/
make console
```
Or for a visualisation example in release mode:
```
cd examples/EmptyExample/
make Visualisation
```

*Debug* mode executables can be built by specifying *dbg=1* to make, `make all dbg=1`.

For more information on building FLAME GPU via make, run `make help` in an example directory.

## Usage

FLAME GPU can be executed as either a console application or as an interactive visualisation.
Please see the [documentation](http://docs.flamegpu.com) for further details.

```
# Console mode
usage: executable [-h] [--help] input_path num_iterations [cuda_device_id] [XML_output_override]

# Interactive visualisation
usage: executable [-h] [--help] input_path [cuda_device_id]
```

For further details, see the [documentation](http://docs.flamegpu.com) or see `executable --help`.


## How to Contribute

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

For an up to date list of publications related to FLAME GPU and it's use, [visit the flamegpu.com website](http://flamegpu.com).


## Authors

FLAME GPU is developed as an open-source project by the [Visual Computing research group](https://www.sheffield.ac.uk/dcs/research/groups/visual-computing/home) in the [Department of Computer Science](https://www.sheffield.ac.uk/dcs/) at the [University of Sheffield](https://www.sheffield.ac.uk/).
The primary author is [Dr Paul Richmond](http://paulrichmond.shef.ac.uk/).

<!-- ## Acknowledgements

**@todo** -->

## Copyright and Software Licence

FLAME GPU is copyright the University of Sheffield 2009 - 2017.
The Library, examples and all source code are covered by the [LICENSE](LICENSE).


## Release Notes

### [1.5.0](https://github.com/FLAMEGPU/FLAMEGPU/releases/tag/v1.5.0)

+ Supports CUDA 8.0
+ Removed SM20 and SM21
+ Visual Studio 2015
+ Improved Linux Support via makefiles
+ **@todo - all other changes**
+ Adds Step-functions executed between subsequent iterations
+ Adds parallel reduction for use in Init, Step and Exit functions
+ Additional Command line options
+ Documentation now hosted on readthedocs
+ Additional example projects
+ Deprecates the use of `state` as an agent variable to control colour in the default visualisation
+ Adds the use of `colour` agent variable to control agent colour in the default visualisation
+ Adds the abillity to pause the default visualisation, and increment one step at a time
+ Various additional macros defined to improve quality of life
+ Various bugfixes
+ `Template` example has been renamed `EmptyExample`
+ `tools/new_example.py` to quickly create a new example project.
+ Adds instrumentation for simple performance measurement
+ Environmental variables can now be loaded from 0.xml
+ Improved `functions.xslt` output
+ Improved state model diagram generator


### [1.4.3](https://github.com/FLAMEGPU/FLAMEGPU/releases/tag/v1.4.3)

+ Updated Circles Example
+ Purged binaries form history, reducing repository size
+ Updated Visual Studio Project files to 2013
+ Improved Visual Studio build customisation
+ Fixed double precision support within spatial partitioning
+ Compile-time spatial partition configuration validation

### [1.4.2](https://github.com/FLAMEGPU/FLAMEGPU/releases/tag/v1.4.2)

+ Added support for continuous agents reading discrete messages.

### [1.4.1](https://github.com/FLAMEGPU/FLAMEGPU/releases/tag/v1.4.1)

+ Minor bug fixes and added missing media folder


### [1.4.0](https://github.com/FLAMEGPU/FLAMEGPU/releases/tag/v1.4)

+ FLAME GPU 1.4 for CUDA 7 and Visual Studio 2012

##Problem reports

To report a bug in this documentation or in the software or propose an improvement, please use the FLAMEGPU github issue tracker.
