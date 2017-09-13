
----------------- MS07Sep2017 ------------------ (The code is not built yet due to the error code : error MSB3721)

Done by the date: 

- 0.XML generator has been coded by MS to generate 'Three Humps' model(test case) initial condition based on mesh-grid model (Matlab/C code)
	* This code is accessible through 0XMLgenerator folder - number of agents can be changed by decreasing the mesh size.
	* The content is then copied and pasted to ~/itterations/0.xml

- Function.c and XMLModelFile are coded and specified based on the case study model (Three Humps - flood modeling)
	* Finite Volume (FV) method has been used to solve Shallow Water Equation (SWE) in this Flood modelling.


----------------- MS08Sep2017 ------------------ (The project has not been built yet)

- The project is set to CUDA 8.0 and is compatible with VS15
- A few modifications in XMLModelFile and corrections in Functions.c (device function corrections and declaration of some variables) has been taken.

----------------- MS13Sep2017 ------------------ (The project builds now)

- 0.xml has been modified to the correct number of agents and array indicators.
- The project builds, but no result is achieved.