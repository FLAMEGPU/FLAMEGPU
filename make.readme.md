# Building FLAME GPU using `make`

## Prerequisites

`make`, `xsltproc`, `nvcc` and `g++` must be installed and on your `path`.

Optionally, if `xmllint` is installed XML Model Files will be validated against the FLAME GPU XML Schemas.
This can be installed using `sudo apt install libxml2-utils` on Debian based systems.

To build and run visualisation examples, `libglew`, `libglut`, `libglu` and `libgl` must also be installed, in addtion to the relevant `-dev` pacakages. Please use the relevant package manager for your distribution. 


## Building all Examples

All included examples can be built from the `examples` directory. I.e

    cd examples
    make all

## Building individual examples

Individual examples can be built from the respective example directory. I.e. 

    cd examples/GameOfLife
    make all

## Help

Further instructions can be found for both the global and per example make files, using

    cd examples # or cd examples/<example>
    make help

## Targets and Options

Targets other than `all` can be used with these makefiles

+ `validate` - Validates the `XMLModelFile.xml` files against the FLAME GPU Schemas
+ `xslt` - Generates dynamic FLAME GPU code from XSLT templates
+ `console` - Builds the console version of the application
+ `visualisation` - Builds the visualisation version of the application, if enabled
+ `clean` - Removes temporary files from the build directory
+ `clobber` - Removes all generated files, including dynamic XSLT generated files and relevant binaries.

## Creating a new Example with Makefile

To create a new example, with a new makefile, copy an existing Makefile into the new example directory. 
I.e. 

    cd examples
    cp GameOfLife/Makefile NewExample/Makefile

There are 3 key variables which must be changed to configure the new example.

+ `EXAMPLE` must be changed to avoid conflicts with binary files. I.e. for the above example this should be set to `NewExample`.
+ `HAS_VISUALISATION` determines if a visualisation target should be built. `0` for no visualisation and `1` for a visualisation.
+ `CUSTOM_VISUALISATION` determines if the default or a custom visualisation should be used (if a `HAS_VISUALISATION == 1`).
    + `0` for default visualisation.
    + `1` for a custom visualisation, this may required further changes depending on your custom visualisation.


## Contact 

If you have further questions please contact the FLAME GPU authors.

+ Paul Richmond : p.richmond@sheffield.ac.uk
+ Mozhgan K. Chimeh : m.kabiri-chimeh@sheffield.ac.uk
