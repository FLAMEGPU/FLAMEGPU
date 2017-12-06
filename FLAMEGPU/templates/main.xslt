<?xml version="1.0" encoding="utf-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" 
                xmlns:xmml="http://www.dcs.shef.ac.uk/~paul/XMML"
                xmlns:gpu="http://www.dcs.shef.ac.uk/~paul/XMMLGPU">
<xsl:output method="text" version="1.0" encoding="UTF-8" indent="yes" />
<xsl:template match="/">
/*
* FLAME GPU v 1.4.0 for CUDA 6
* Copyright 2015 University of Sheffield.
* Author: Dr Paul Richmond
* Contact: p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
*
* University of Sheffield retain all intellectual property and
* proprietary rights in and to this software and related documentation.
* Any use, reproduction, disclosure, or distribution of this software
* and related documentation without an express license agreement from
* University of Sheffield is strictly prohibited.
*
* For terms of licence agreement please attached licence or view licence
* on www.flamegpu.com website.
*
*/
#include &lt;cuda_runtime.h&gt;
#include &lt;stdio.h&gt;
#ifdef VISUALISATION
#include &lt;GL/glew.h&gt;
#include &lt;GL/glut.h&gt;
#endif
#include "header.h"

/* IO Variables*/
char inputfile[100];          /**&lt; Input path char buffer*/
char outputpath[1000];         /**&lt; Output path char buffer*/

// Define the default value indicating if XML output should be produced or not.
#define OUTPUT_TO_XML 1

/** checkUsage
 * Function to check the correct number of arguments
 * @param arc	main argument count
 * @param argv	main argument values
 * @return true if usage is correct, otherwise false
 */
int checkUsage( int argc, char** argv){
	//Check usage
#ifdef VISUALISATION
	printf("FLAMEGPU Visualisation mode\n");
	if(argc &lt; 2)
	{
		printf("Usage: main [XML model data] [Optional CUDA device ID]\n");
		return false;
	}
#else
	printf("FLAMEGPU Console mode\n");
	if(argc &lt; 3)
	{
		printf("Usage: main [XML model data] [Iterations] [Optional CUDA device ID] [Optional XML Output Override]\n");
		return false;
	}
#endif
	return true;
}

const char* getOutputDir(){
  return outputpath;
}


/** setFilePaths
 * Function to set global variables for the input XML file and its directory location
 *@param input input path of model xml file
 */
void setFilePaths(char* input){
	//Copy input file
	strcpy(inputfile, input);
	printf("Initial states: %s\n", inputfile);

	//Calculate the output path from the path of the input file
	int i = 0;
	int lastd = -1;
	while(inputfile[i] != '\0')
	{
		/* For windows directories */
		if(inputfile[i] == '\\') lastd=i;
		/* For unix directories */
		if(inputfile[i] == '/') lastd=i;
		i++;
	}
	strcpy(outputpath, inputfile);
	outputpath[lastd+1] = '\0';
	printf("Output dir: %s\n", outputpath);
}


bool getOutputXML(int argc, char**argv){
	// Initialise to #defined default
	

#ifdef VISUALISATION
	// If visualisation mode is set, we do not output.
	return false;
#else
	// If console mode is set and we have the right number of arguments, use the relevant index.
	if (argc &gt;= 5){
		// Return the value from the argument.
		return (bool)atoi(argv[4]);
	} else {
		// Return the default value.
		return (bool) OUTPUT_TO_XML;
	}
#endif

}

void initCUDA(int argc, char** argv){
	cudaError_t cudaStatus;
	int device;
	int device_count;

	//default device
	device = 0;
	cudaStatus = cudaGetDeviceCount(&amp;device_count);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error finding CUDA devices!  Do you have a CUDA-capable GPU installed?\n");
		exit(EXIT_FAILURE);
	}
	if (device_count == 0){
		fprintf(stderr, "Error no CUDA devices found!\n");
		exit(EXIT_FAILURE);
	}

#ifdef VISUALISATION
	if (argc &gt;= 3){
		device = atoi(argv[2]);
	}
#else
	if (argc &gt;= 4){
		device = atoi(argv[3]);
	}
#endif

	if (device >= device_count){
		fprintf(stderr, "Error selecting CUDA device! Device id '%d' is not found?\n", device);
		exit(EXIT_FAILURE);
	}

	// Select device
	cudaStatus = cudaSetDevice(device);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error setting CUDA device!\n");
		exit(EXIT_FAILURE);
	}
	// Get device properties.
	cudaDeviceProp props;
	cudaStatus = cudaGetDeviceProperties(&amp;props, device);
	if(cudaStatus == cudaSuccess){
	#ifdef _MSC_VER
		const char * driverMode = props.tccDriver ? "TCC" : "WDDM";
	#else
		const char * driverMode = "Linux"
	#endif
		fprintf(stdout, "GPU %d: %s, SM%d%d, %s, pciBusId %d\n", device, props.name, props.major, props.minor, driverMode, props.pciBusID);
	} else {
		fprintf(stderr, "Error Accessing Cuda Device properties for GPU %d\n", device);
	}

}

void runConsoleWithoutXMLOutput(int iterations){
	// Iteratively tun the correct number of iterations.
	for (int i=0; i&lt; iterations; i++)
	{
		printf("Processing Simulation Step %i\n", i+1);
		//single simulation iteration
		singleIteration();
	}
}

void runConsoleWithXMLOutput(int iterations){
	// Iteratively tun the correct number of iterations.
	for (int i=0; i&lt; iterations; i++)
	{
		printf("Processing Simulation Step %i\n", i+1);
		//single simulation iteration
		singleIteration();
		// Save the iteration data to disk
		saveIterationData(outputpath, i+1, <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state">
			<!--<xsl:value-of select="xmml:name"/> state <xsl:value-of select="../../xmml:name"/> agents -->
			get_host_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_agents(), get_device_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_agents(), get_agent_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count()<xsl:choose><xsl:when test="position()=last()">);</xsl:when><xsl:otherwise>,</xsl:otherwise></xsl:choose>
			</xsl:for-each>
			printf("Iteration %i Saved to XML\n", i+1);
	}
}

/**
 * Program main (Handles arguments)
 */
int main( int argc, char** argv) 
{
	cudaError_t cudaStatus;
	//check usage mode
	if (!checkUsage(argc, argv))
		exit(EXIT_FAILURE);

	//get the directory paths
	setFilePaths(argv[1]);

	//determine if we want to output to xml.
	bool outputXML = getOutputXML(argc, argv);

	//initialise CUDA
	initCUDA(argc, argv);

#ifdef VISUALISATION
	//Init visualisation must be done before simulation init
	initVisualisation();
#endif

	//initialise the simulation
	initialise(inputfile);

    
#ifdef VISUALISATION
	runVisualisation();
	exit(EXIT_SUCCESS);
#else	
	//Benchmark simulation
	cudaEvent_t start, stop;
	float milliseconds = 0;
	
	//create timing events
	cudaEventCreate(&amp;start);
	cudaEventCreate(&amp;stop);
	
	//Get the number of iterations
	int iterations = atoi(argv[2]);
	if (iterations &lt;= 0)
	{
		printf("Second argument must be a positive integer (Number of Iterations)\n");
		exit(EXIT_FAILURE);
	}
  
	//start timing
	cudaEventRecord(start);

	// Launch the main loop with / without xml output.
	if(outputXML){
		runConsoleWithXMLOutput(iterations);
	} else {
		runConsoleWithoutXMLOutput(iterations);	
	}
	

	//CUDA stop timing
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&amp;milliseconds, start, stop);
	printf( "Total Processing time: %f (ms)\n", milliseconds);
#endif

	cleanup();
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error resetting the device!\n");
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
</xsl:template>
</xsl:stylesheet>
