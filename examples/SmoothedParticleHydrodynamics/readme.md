# FLAME GPU Example: Smoothed Particle Hydrodynamics

This example shows a fluid simulation using [Smoothed Particle Hydrodynamics](https://www.annualreviews.org/doi/abs/10.1146/annurev.aa.30.090192.002551). The method includes a modification to simulate surface tension. The simulation includes both static and dynamic particles. Simulation parameters are found near the top of the functions.c file, along with suggested parameter ranges and things to experiment with.

The example demonstrates the use of the following FlameGPU features:
*Spatial partitioning
*Function conditions

The objects simulated can be modified using the included gen_initial_conditions.py file which can be found in the iterations folder. Usage instructions can be seen by running the script with no arguments.

Author: Matthew Leach

Contact: M.Leach@sheffield.ac.uk