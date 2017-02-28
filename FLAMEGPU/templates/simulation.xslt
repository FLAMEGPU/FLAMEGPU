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

//Disable internal thrust warnings about conversions
#pragma warning(push)
#pragma warning (disable : 4267)
#pragma warning (disable : 4244)

// includes
#include &lt;cuda_runtime.h&gt;
#include &lt;device_launch_parameters.h&gt;
#include &lt;stdlib.h&gt;
#include &lt;stdio.h&gt;
#include &lt;string.h&gt;
#include &lt;cmath&gt;
#include &lt;thrust/device_ptr.h&gt;
#include &lt;thrust/scan.h&gt;
#include &lt;thrust/sort.h&gt;
#include &lt;thrust/system/cuda/execution_policy.h&gt;

// include FLAME kernels
#include "FLAMEGPU_kernals.cu"
<!--Compile time error if partitioning radius is not a factor of the partitioning dimensions as this causes partitioning to execute incorrectly-->
<xsl:for-each select="gpu:xmodel/xmml:messages/gpu:message/gpu:partitioningSpatial">
<xsl:if test="(gpu:xmax - gpu:xmin) != (floor((gpu:xmax - gpu:xmin ) div gpu:radius ) * gpu:radius)">
#error "XML model spatial partitioning radius must be a factor of partitioning dimensions. Radius: <xsl:value-of select="gpu:radius"/>, Xmin: <xsl:value-of select="gpu:xmin"/>, Xmax: <xsl:value-of select="gpu:xmax"/>"
</xsl:if><xsl:if test="(gpu:ymax - gpu:ymin) != (floor((gpu:ymax - gpu:ymin ) div gpu:radius ) * gpu:radius)">
#error "XML model spatial partitioning radius must be a factor of partitioning dimensions. Radius: <xsl:value-of select="gpu:radius"/>, Ymin: <xsl:value-of select="gpu:ymin"/>, Ymax: <xsl:value-of select="gpu:ymax"/>"
</xsl:if><xsl:if test="(gpu:zmax - gpu:zmin) != (floor((gpu:zmax - gpu:zmin ) div gpu:radius ) * gpu:radius)">
#error "XML model spatial partitioning radius must be a factor of partitioning dimensions. Radius: <xsl:value-of select="gpu:radius"/>, Zmin: <xsl:value-of select="gpu:zmin"/>, Zmax: <xsl:value-of select="gpu:zmax"/>"
</xsl:if>
</xsl:for-each>

#pragma warning(pop)

/* Error check function for safe CUDA API calling */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* Error check function for post CUDA Kernel calling */
#define gpuErrchkLaunch() { gpuLaunchAssert(__FILE__, __LINE__); }
inline void gpuLaunchAssert(const char *file, int line, bool abort=true)
{
	gpuAssert( cudaPeekAtLastError(), file, line );
#ifdef _DEBUG
	gpuAssert( cudaDeviceSynchronize(), file, line );
#endif
   
}

/* SM padding and offset variables */
int SM_START;
int PADDING;

/* Agent Memory */
<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
/* <xsl:value-of select="xmml:name"/> Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_<xsl:value-of select="xmml:name"/>_list* d_<xsl:value-of select="xmml:name"/>s;      /**&lt; Pointer to agent list (population) on the device*/
xmachine_memory_<xsl:value-of select="xmml:name"/>_list* d_<xsl:value-of select="xmml:name"/>s_swap; /**&lt; Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_<xsl:value-of select="xmml:name"/>_list* d_<xsl:value-of select="xmml:name"/>s_new;  /**&lt; Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_<xsl:value-of select="xmml:name"/>_count;   /**&lt; Agent population size counter */ <xsl:if test="gpu:type='discrete'">
int h_xmachine_memory_<xsl:value-of select="xmml:name"/>_pop_width;   /**&lt; Agent population width */</xsl:if>
uint * d_xmachine_memory_<xsl:value-of select="xmml:name"/>_keys;	  /**&lt; Agent sort identifiers keys*/
uint * d_xmachine_memory_<xsl:value-of select="xmml:name"/>_values;  /**&lt; Agent sort identifiers value */
    <xsl:for-each select="xmml:states/gpu:state">
/* <xsl:value-of select="../../xmml:name"/> state variables */
xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>;      /**&lt; Pointer to agent list (population) on host*/
xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>;      /**&lt; Pointer to agent list (population) on the device*/
int h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count;   /**&lt; Agent population size counter */ 
</xsl:for-each>
</xsl:for-each>

/* Message Memory */
<xsl:for-each select="gpu:xmodel/xmml:messages/gpu:message">
/* <xsl:value-of select="xmml:name"/> Message variables */
xmachine_message_<xsl:value-of select="xmml:name"/>_list* h_<xsl:value-of select="xmml:name"/>s;         /**&lt; Pointer to message list on host*/
xmachine_message_<xsl:value-of select="xmml:name"/>_list* d_<xsl:value-of select="xmml:name"/>s;         /**&lt; Pointer to message list on device*/
xmachine_message_<xsl:value-of select="xmml:name"/>_list* d_<xsl:value-of select="xmml:name"/>s_swap;    /**&lt; Pointer to message swap list on device (used for holding optional messages)*/
<xsl:if test="gpu:partitioningNone or gpu:partitioningSpatial">/* Non partitioned and spatial partitioned message variables  */
int h_message_<xsl:value-of select="xmml:name"/>_count;         /**&lt; message list counter*/
int h_message_<xsl:value-of select="xmml:name"/>_output_type;   /**&lt; message output type (single or optional)*/
</xsl:if><xsl:if test="gpu:partitioningSpatial">/* Spatial Partitioning Variables*/
#ifdef FAST_ATOMIC_SORTING
	uint * d_xmachine_message_<xsl:value-of select="xmml:name"/>_local_bin_index;	  /**&lt; index offset within the assigned bin */
	uint * d_xmachine_message_<xsl:value-of select="xmml:name"/>_unsorted_index;		/**&lt; unsorted index (hash) value for message */
#else
	uint * d_xmachine_message_<xsl:value-of select="xmml:name"/>_keys;	  /**&lt; message sort identifier keys*/
	uint * d_xmachine_message_<xsl:value-of select="xmml:name"/>_values;  /**&lt; message sort identifier values */
#endif
xmachine_message_<xsl:value-of select="xmml:name"/>_PBM * d_<xsl:value-of select="xmml:name"/>_partition_matrix;  /**&lt; Pointer to PCB matrix */
glm::vec3 h_message_<xsl:value-of select="xmml:name"/>_min_bounds;           /**&lt; min bounds (x,y,z) of partitioning environment */
glm::vec3 h_message_<xsl:value-of select="xmml:name"/>_max_bounds;           /**&lt; max bounds (x,y,z) of partitioning environment */
glm::ivec3 h_message_<xsl:value-of select="xmml:name"/>_partitionDim;           /**&lt; partition dimensions (x,y,z) of partitioning environment */
float h_message_<xsl:value-of select="xmml:name"/>_radius;                 /**&lt; partition radius (used to determin the size of the partitions) */
</xsl:if><xsl:if test="gpu:partitioningDiscrete">/* Discrete Partitioning Variables*/
int h_message_<xsl:value-of select="xmml:name"/>_range;     /**&lt; range of the discrete message*/
int h_message_<xsl:value-of select="xmml:name"/>_width;     /**&lt; with of the message grid*/
</xsl:if><xsl:if test="gpu:partitioningDiscrete or gpu:partitioningSpatial">/* Texture offset values for host */<xsl:for-each select="xmml:variables/gpu:variable">
int h_tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_offset;</xsl:for-each>
<xsl:if test="gpu:partitioningSpatial">
int h_tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_start_offset;
int h_tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_end_or_count_offset;
</xsl:if></xsl:if>
</xsl:for-each>
  
/* CUDA Streams for function layers */<xsl:for-each select="gpu:xmodel/xmml:layers/xmml:layer">
<xsl:sort select="count(gpu:layerFunction)" order="descending"/>
<xsl:if test="position() =1"> <!-- Get the layer with most functions -->
<xsl:for-each select="gpu:layerFunction">
cudaStream_t stream<xsl:value-of select="position()"/>;</xsl:for-each>
</xsl:if>
</xsl:for-each>

/*Global condition counts*/<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:functions/gpu:function/gpu:globalCondition">
int h_<xsl:value-of select="../xmml:name"/>_condition_count;
</xsl:for-each>

/* RNG rand48 */
RNG_rand48* h_rand48;    /**&lt; Pointer to RNG_rand48 seed list on host*/
RNG_rand48* d_rand48;    /**&lt; Pointer to RNG_rand48 seed list on device*/

/* CUDA Parallel Primatives variables */
int scan_last_sum;           /**&lt; Indicates if the position (in message list) of last message*/
int scan_last_included;      /**&lt; Indicates if last sum value is included in the total sum count*/

/* Agent function prototypes */
<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:functions/gpu:function">
/** <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>
 * Agent function prototype for <xsl:value-of select="xmml:name"/> function of <xsl:value-of select="../../xmml:name"/> agent
 */
void <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>(cudaStream_t &amp;stream);
</xsl:for-each>
  
void setPaddingAndOffset()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&amp;deviceProp, 0);
	int x64_sys = 0;

	// This function call returns 9999 for both major &amp; minor fields, if no CUDA capable devices are present
	if (deviceProp.major == 9999 &amp;&amp; deviceProp.minor == 9999){
		printf("Error: There is no device supporting CUDA.\n");
		exit(0);
	}
    
    //check if double is used and supported
#ifdef _DOUBLE_SUPPORT_REQUIRED_
	printf("Simulation requires full precision double values\n");
	if ((deviceProp.major &lt; 2)&amp;&amp;(deviceProp.minor &lt; 3)){
		printf("Error: Hardware does not support full precision double values!\n");
		exit(0);
	}
    
#endif

	//check 32 or 64bit
	x64_sys = (sizeof(void*)==8);
	if (x64_sys)
	{
		printf("64Bit System Detected\n");
	}
	else
	{
		printf("32Bit System Detected\n");
	}

	SM_START = 0;
	PADDING = 0;
  
	//copy padding and offset to GPU
	gpuErrchk(cudaMemcpyToSymbol( d_SM_START, &amp;SM_START, sizeof(int)));
	gpuErrchk(cudaMemcpyToSymbol( d_PADDING, &amp;PADDING, sizeof(int)));     
}

int is_sqr_pow2(int x){
	int r = (int)pow(4, ceil(log(x)/log(4)));
	return (r == x);
}

int lowest_sqr_pow2(int x){
	int l;
	
	//escape early if x is square power of 2
	if (is_sqr_pow2(x))
		return x;
	
	//lower bound		
	l = (int)pow(4, floor(log(x)/log(4)));
	
	return l;
}

/* Unary function required for cudaOccupancyMaxPotentialBlockSizeVariableSMem to avoid warnings */
int no_sm(int b){
	return 0;
}

/* Unary function to return shared memory size for reorder message kernels */
int reorder_messages_sm_size(int blockSize)
{
	return sizeof(unsigned int)*(blockSize+1);
}


void initialise(char * inputfile){

	//set the padding and offset values depending on architecture and OS
	setPaddingAndOffset();
  

	printf("Allocating Host and Device memory\n");
  
	/* Agent memory allocation (CPU) */<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
	int xmachine_<xsl:value-of select="xmml:name"/>_SoA_size = sizeof(xmachine_memory_<xsl:value-of select="xmml:name"/>_list);<xsl:for-each select="xmml:states/gpu:state">
	h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/> = (xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list*)malloc(xmachine_<xsl:value-of select="../../xmml:name"/>_SoA_size);</xsl:for-each></xsl:for-each>

	/* Message memory allocation (CPU) */<xsl:for-each select="gpu:xmodel/xmml:messages/gpu:message">
	int message_<xsl:value-of select="xmml:name"/>_SoA_size = sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>_list);
	h_<xsl:value-of select="xmml:name"/>s = (xmachine_message_<xsl:value-of select="xmml:name"/>_list*)malloc(message_<xsl:value-of select="xmml:name"/>_SoA_size);</xsl:for-each>

	//Exit if agent or message buffer sizes are to small for function outputs<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:functions/gpu:function/xmml:xagentOutputs/gpu:xagentOutput">
	<xsl:variable name="xagent_output" select="xmml:xagentName"/><xsl:variable name="xagent_buffer" select="../../../../gpu:bufferSize"/><xsl:if test="../../../../../gpu:xagent[xmml:name=$xagent_output]/gpu:bufferSize&lt;$xagent_buffer">
	printf("ERROR: <xsl:value-of select="$xagent_output"/> agent buffer is too small to be used for output by <xsl:value-of select="../../../../xmml:name"/> agent in <xsl:value-of select="../../xmml:name"/> function!\n");
	exit(0);
	</xsl:if>    
	</xsl:for-each>
    
	<xsl:for-each select="gpu:xmodel/xmml:messages/gpu:message"><xsl:if test="gpu:partitioningDiscrete">
	
	/* Set discrete <xsl:value-of select="xmml:name"/> message variables (range, width)*/
	h_message_<xsl:value-of select="xmml:name"/>_range = <xsl:value-of select="gpu:partitioningDiscrete/gpu:radius"/>; //from xml
	h_message_<xsl:value-of select="xmml:name"/>_width = (int)floor(sqrt((float)xmachine_message_<xsl:value-of select="xmml:name"/>_MAX));
	//check the width
	if (!is_sqr_pow2(xmachine_message_<xsl:value-of select="xmml:name"/>_MAX)){
		printf("ERROR: <xsl:value-of select="xmml:name"/> message max must be a square power of 2 for a 2D discrete message grid!\n");
		exit(0);
	}
	gpuErrchk(cudaMemcpyToSymbol( d_message_<xsl:value-of select="xmml:name"/>_range, &amp;h_message_<xsl:value-of select="xmml:name"/>_range, sizeof(int)));	
	gpuErrchk(cudaMemcpyToSymbol( d_message_<xsl:value-of select="xmml:name"/>_width, &amp;h_message_<xsl:value-of select="xmml:name"/>_width, sizeof(int)));
	</xsl:if><xsl:if test="gpu:partitioningSpatial">
			
	/* Set spatial partitioning <xsl:value-of select="xmml:name"/> message variables (min_bounds, max_bounds)*/
	h_message_<xsl:value-of select="xmml:name"/>_radius = (float)<xsl:value-of select="gpu:partitioningSpatial/gpu:radius"/>;
	gpuErrchk(cudaMemcpyToSymbol( d_message_<xsl:value-of select="xmml:name"/>_radius, &amp;h_message_<xsl:value-of select="xmml:name"/>_radius, sizeof(float)));	
	    h_message_<xsl:value-of select="xmml:name"/>_min_bounds = glm::vec3((float)<xsl:value-of select="gpu:partitioningSpatial/gpu:xmin"/>, (float)<xsl:value-of select="gpu:partitioningSpatial/gpu:ymin"/>, (float)<xsl:value-of select="gpu:partitioningSpatial/gpu:zmin"/>);
	gpuErrchk(cudaMemcpyToSymbol( d_message_<xsl:value-of select="xmml:name"/>_min_bounds, &amp;h_message_<xsl:value-of select="xmml:name"/>_min_bounds, sizeof(glm::vec3)));	
	h_message_<xsl:value-of select="xmml:name"/>_max_bounds = glm::vec3((float)<xsl:value-of select="gpu:partitioningSpatial/gpu:xmax"/>, (float)<xsl:value-of select="gpu:partitioningSpatial/gpu:ymax"/>, (float)<xsl:value-of select="gpu:partitioningSpatial/gpu:zmax"/>);
	gpuErrchk(cudaMemcpyToSymbol( d_message_<xsl:value-of select="xmml:name"/>_max_bounds, &amp;h_message_<xsl:value-of select="xmml:name"/>_max_bounds, sizeof(glm::vec3)));	
	h_message_<xsl:value-of select="xmml:name"/>_partitionDim.x = (int)ceil((h_message_<xsl:value-of select="xmml:name"/>_max_bounds.x - h_message_<xsl:value-of select="xmml:name"/>_min_bounds.x)/h_message_<xsl:value-of select="xmml:name"/>_radius);
	h_message_<xsl:value-of select="xmml:name"/>_partitionDim.y = (int)ceil((h_message_<xsl:value-of select="xmml:name"/>_max_bounds.y - h_message_<xsl:value-of select="xmml:name"/>_min_bounds.y)/h_message_<xsl:value-of select="xmml:name"/>_radius);
	h_message_<xsl:value-of select="xmml:name"/>_partitionDim.z = (int)ceil((h_message_<xsl:value-of select="xmml:name"/>_max_bounds.z - h_message_<xsl:value-of select="xmml:name"/>_min_bounds.z)/h_message_<xsl:value-of select="xmml:name"/>_radius);
	gpuErrchk(cudaMemcpyToSymbol( d_message_<xsl:value-of select="xmml:name"/>_partitionDim, &amp;h_message_<xsl:value-of select="xmml:name"/>_partitionDim, sizeof(glm::ivec3)));	
	</xsl:if></xsl:for-each>
	
	
	<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent"><xsl:if test="gpu:type='discrete'">
	/* Check that population size is a square power of 2*/
	if (!is_sqr_pow2(xmachine_memory_<xsl:value-of select="xmml:name"/>_MAX)){
		printf("ERROR: <xsl:value-of select="xmml:name"/>s agent count must be a square power of 2!\n");
		exit(0);
	}
	h_xmachine_memory_<xsl:value-of select="xmml:name"/>_pop_width = (int)sqrt(xmachine_memory_<xsl:value-of select="xmml:name"/>_MAX);
	</xsl:if></xsl:for-each>

	//read initial states
	readInitialStates(inputfile, <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">h_<xsl:value-of select="xmml:name"/>s_<xsl:value-of select="xmml:states/xmml:initialState"/>, &amp;h_xmachine_memory_<xsl:value-of select="xmml:name"/>_<xsl:value-of select="xmml:states/xmml:initialState"/>_count<xsl:if test="position()!=last()">, </xsl:if></xsl:for-each>);
	
	<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
	/* <xsl:value-of select="xmml:name"/> Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &amp;d_<xsl:value-of select="xmml:name"/>s, xmachine_<xsl:value-of select="xmml:name"/>_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &amp;d_<xsl:value-of select="xmml:name"/>s_swap, xmachine_<xsl:value-of select="xmml:name"/>_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &amp;d_<xsl:value-of select="xmml:name"/>s_new, xmachine_<xsl:value-of select="xmml:name"/>_SoA_size));
    <xsl:if test="gpu:type='continuous'">//continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &amp;d_xmachine_memory_<xsl:value-of select="xmml:name"/>_keys, xmachine_memory_<xsl:value-of select="xmml:name"/>_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &amp;d_xmachine_memory_<xsl:value-of select="xmml:name"/>_values, xmachine_memory_<xsl:value-of select="xmml:name"/>_MAX* sizeof(uint)));</xsl:if>
    <xsl:for-each select="xmml:states/gpu:state">
	/* <xsl:value-of select="xmml:name"/> memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &amp;d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, xmachine_<xsl:value-of select="../../xmml:name"/>_SoA_size));
	gpuErrchk( cudaMemcpy( d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, xmachine_<xsl:value-of select="../../xmml:name"/>_SoA_size, cudaMemcpyHostToDevice));
    </xsl:for-each>
	</xsl:for-each>

	<xsl:for-each select="gpu:xmodel/xmml:messages/gpu:message">
	/* <xsl:value-of select="xmml:name"/> Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &amp;d_<xsl:value-of select="xmml:name"/>s, message_<xsl:value-of select="xmml:name"/>_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &amp;d_<xsl:value-of select="xmml:name"/>s_swap, message_<xsl:value-of select="xmml:name"/>_SoA_size));
	gpuErrchk( cudaMemcpy( d_<xsl:value-of select="xmml:name"/>s, h_<xsl:value-of select="xmml:name"/>s, message_<xsl:value-of select="xmml:name"/>_SoA_size, cudaMemcpyHostToDevice));<xsl:if test="gpu:partitioningSpatial">
	gpuErrchk( cudaMalloc( (void**) &amp;d_<xsl:value-of select="xmml:name"/>_partition_matrix, sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>_PBM)));
#ifdef FAST_ATOMIC_SORTING
	gpuErrchk( cudaMalloc( (void**) &amp;d_xmachine_message_<xsl:value-of select="xmml:name"/>_local_bin_index, xmachine_message_<xsl:value-of select="xmml:name"/>_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &amp;d_xmachine_message_<xsl:value-of select="xmml:name"/>_unsorted_index, xmachine_message_<xsl:value-of select="xmml:name"/>_MAX* sizeof(uint)));
#else
	gpuErrchk( cudaMalloc( (void**) &amp;d_xmachine_message_<xsl:value-of select="xmml:name"/>_keys, xmachine_message_<xsl:value-of select="xmml:name"/>_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &amp;d_xmachine_message_<xsl:value-of select="xmml:name"/>_values, xmachine_message_<xsl:value-of select="xmml:name"/>_MAX* sizeof(uint)));
#endif</xsl:if><xsl:text>
	</xsl:text></xsl:for-each>	

	/*Set global condition counts*/<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:functions/gpu:function/gpu:condition">
	h_<xsl:value-of select="../xmml:name"/>_condition_false_count = 0;
	</xsl:for-each>

	/* RNG rand48 */
	int h_rand48_SoA_size = sizeof(RNG_rand48);
	h_rand48 = (RNG_rand48*)malloc(h_rand48_SoA_size);
	//allocate on GPU
	gpuErrchk( cudaMalloc( (void**) &amp;d_rand48, h_rand48_SoA_size));
	// calculate strided iteration constants
	static const unsigned long long a = 0x5DEECE66DLL, c = 0xB;
	int seed = 123;
	unsigned long long A, C;
	A = 1LL; C = 0LL;
	for (unsigned int i = 0; i &lt; buffer_size_MAX; ++i) {
		C += A*c;
		A *= a;
	}
	h_rand48->A.x = A &amp; 0xFFFFFFLL;
	h_rand48->A.y = (A >> 24) &amp; 0xFFFFFFLL;
	h_rand48->C.x = C &amp; 0xFFFFFFLL;
	h_rand48->C.y = (C >> 24) &amp; 0xFFFFFFLL;
	// prepare first nThreads random numbers from seed
	unsigned long long x = (((unsigned long long)seed) &lt;&lt; 16) | 0x330E;
	for (unsigned int i = 0; i &lt; buffer_size_MAX; ++i) {
		x = a*x + c;
		h_rand48->seeds[i].x = x &amp; 0xFFFFFFLL;
		h_rand48->seeds[i].y = (x >> 24) &amp; 0xFFFFFFLL;
	}
	//copy to device
	gpuErrchk( cudaMemcpy( d_rand48, h_rand48, h_rand48_SoA_size, cudaMemcpyHostToDevice));

	/* Call all init functions */
	<xsl:for-each select="gpu:xmodel/gpu:environment/gpu:initFunctions/gpu:initFunction">
	<xsl:value-of select="gpu:name"/>();<xsl:text>
	</xsl:text></xsl:for-each>
  
  /* Init CUDA Streams for function layers */
  <xsl:for-each select="gpu:xmodel/xmml:layers/xmml:layer">
  <xsl:sort select="count(gpu:layerFunction)" order="descending"/>
  <xsl:if test="position() =1"> <!-- Get the layer with most functions -->
  <xsl:for-each select="gpu:layerFunction">
  gpuErrchk(cudaStreamCreate(&amp;stream<xsl:value-of select="position()"/>));</xsl:for-each>
  </xsl:if>
  </xsl:for-each>
} 

<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent"><xsl:if test="gpu:type='continuous'"> <xsl:for-each select="xmml:states/gpu:state">
void sort_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count); 
	gridSize = (h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs&lt;&lt;&lt;gridSize, blockSize&gt;&gt;&gt;(d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_keys, d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_values, d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_keys),  thrust::device_pointer_cast(d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_keys) + h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count,  thrust::device_pointer_cast(d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, reorder_<xsl:value-of select="../../xmml:name"/>_agents, no_sm, h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count); 
	gridSize = (h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_<xsl:value-of select="../../xmml:name"/>_agents&lt;&lt;&lt;gridSize, blockSize&gt;&gt;&gt;(d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_values, d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, d_<xsl:value-of select="../../xmml:name"/>s_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* d_<xsl:value-of select="../../xmml:name"/>s_temp = d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>;
	d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/> = d_<xsl:value-of select="../../xmml:name"/>s_swap;
	d_<xsl:value-of select="../../xmml:name"/>s_swap = d_<xsl:value-of select="../../xmml:name"/>s_temp;	
}
</xsl:for-each></xsl:if></xsl:for-each>

void cleanup(){

    /* Call all exit functions */
	<xsl:for-each select="gpu:xmodel/gpu:environment/gpu:exitFunctions/gpu:exitFunction">
	<xsl:value-of select="gpu:name"/>();<xsl:text>
	</xsl:text></xsl:for-each>

	/* Agent data free*/
	<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
	/* <xsl:value-of select="xmml:name"/> Agent variables */
	gpuErrchk(cudaFree(d_<xsl:value-of select="xmml:name"/>s));
	gpuErrchk(cudaFree(d_<xsl:value-of select="xmml:name"/>s_swap));
	gpuErrchk(cudaFree(d_<xsl:value-of select="xmml:name"/>s_new));
	<xsl:for-each select="xmml:states/gpu:state">
	free( h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>);
	gpuErrchk(cudaFree(d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>));
	</xsl:for-each>
	</xsl:for-each>

	/* Message data free */
	<xsl:for-each select="gpu:xmodel/xmml:messages/gpu:message">
	/* <xsl:value-of select="xmml:name"/> Message variables */
	free( h_<xsl:value-of select="xmml:name"/>s);
	gpuErrchk(cudaFree(d_<xsl:value-of select="xmml:name"/>s));
	gpuErrchk(cudaFree(d_<xsl:value-of select="xmml:name"/>s_swap));<xsl:if test="gpu:partitioningSpatial">
	gpuErrchk(cudaFree(d_<xsl:value-of select="xmml:name"/>_partition_matrix));
#ifdef FAST_ATOMIC_SORTING
	gpuErrchk(cudaFree(d_xmachine_message_<xsl:value-of select="xmml:name"/>_local_bin_index));
	gpuErrchk(cudaFree(d_xmachine_message_<xsl:value-of select="xmml:name"/>_unsorted_index));
#else
	gpuErrchk(cudaFree(d_xmachine_message_<xsl:value-of select="xmml:name"/>_keys));
	gpuErrchk(cudaFree(d_xmachine_message_<xsl:value-of select="xmml:name"/>_values));
#endif</xsl:if><xsl:text>
	</xsl:text></xsl:for-each>
  
  /* CUDA Streams for function layers */
  <xsl:for-each select="gpu:xmodel/xmml:layers/xmml:layer">
  <xsl:sort select="count(gpu:layerFunction)" order="descending"/>
  <xsl:if test="position() =1"> <!-- Get the layer with most functions -->
  <xsl:for-each select="gpu:layerFunction">
  gpuErrchk(cudaStreamDestroy(stream<xsl:value-of select="position()"/>));</xsl:for-each>
  </xsl:if>
  </xsl:for-each>
}

void singleIteration(){

	/* set all non partitioned and spatial partitioned message counts to 0*/<xsl:for-each select="gpu:xmodel/xmml:messages/gpu:message"><xsl:if test="gpu:partitioningNone or gpu:partitioningSpatial">
	h_message_<xsl:value-of select="xmml:name"/>_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_<xsl:value-of select="xmml:name"/>_count, &amp;h_message_<xsl:value-of select="xmml:name"/>_count, sizeof(int)));
	</xsl:if></xsl:for-each>

	/* Call agent functions in order iterating through the layer functions */
	<xsl:for-each select="gpu:xmodel/xmml:layers/xmml:layer">
	/* Layer <xsl:value-of select="position()"/>*/
	<xsl:for-each select="gpu:layerFunction">
	<xsl:variable name="function" select="xmml:name"/><xsl:variable name="stream_num" select="position()"/><xsl:for-each select="../../../xmml:xagents/gpu:xagent/xmml:functions/gpu:function[xmml:name=$function]">
	<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>(stream<xsl:value-of select="$stream_num"/>);
	</xsl:for-each></xsl:for-each>cudaDeviceSynchronize();
  </xsl:for-each>
    
    /* Call all step functions */
	<xsl:for-each select="gpu:xmodel/gpu:environment/gpu:stepFunctions/gpu:stepFunction">
	<xsl:value-of select="gpu:name"/>();<xsl:text>
	</xsl:text></xsl:for-each>
}

/* Environment functions */
<!--
<xsl:for-each select="gpu:xmodel/gpu:environment/gpu:constants/gpu:variable">
void set_<xsl:value-of select="xmml:name"/>(<xsl:value-of select="xmml:type"/>* h_<xsl:value-of select="xmml:name"/>){
	gpuErrchk(cudaMemcpyToSymbol(<xsl:value-of select="xmml:name"/>, h_<xsl:value-of select="xmml:name"/>, sizeof(<xsl:value-of select="xmml:type"/>)<xsl:if test="xmml:arrayLength">*<xsl:value-of select="xmml:arrayLength"/></xsl:if>));
}
</xsl:for-each>
-->

<!-- -->
//host constant declaration
<xsl:for-each select="gpu:xmodel/gpu:environment/gpu:constants/gpu:variable">
<xsl:value-of select="xmml:type"/><xsl:text> h_env_</xsl:text><xsl:value-of select="xmml:name"/><xsl:if test="xmml:arrayLength">[<xsl:value-of select="xmml:arrayLength"/>]</xsl:if>;
</xsl:for-each>

<xsl:for-each select="gpu:xmodel/gpu:environment/gpu:constants/gpu:variable">

//constant setter
void set_<xsl:value-of select="xmml:name"/>(<xsl:value-of select="xmml:type"/>* h_<xsl:value-of select="xmml:name"/>){
    gpuErrchk(cudaMemcpyToSymbol(<xsl:value-of select="xmml:name"/>, h_<xsl:value-of select="xmml:name"/>, sizeof(<xsl:value-of select="xmml:type"/>)<xsl:if test="xmml:arrayLength">*<xsl:value-of select="xmml:arrayLength"/></xsl:if>));
    memcpy(&amp;h_env_<xsl:value-of select="xmml:name"/>, h_<xsl:value-of select="xmml:name"/>,sizeof(<xsl:value-of select="xmml:type"/>)<xsl:if test="xmml:arrayLength">*<xsl:value-of select="xmml:arrayLength"/></xsl:if>);
}

//constant getter
const <xsl:value-of select="xmml:type"/>* get_<xsl:value-of select="xmml:name"/>(){
    return &amp;h_env_<xsl:value-of select="xmml:name"/>;
}
</xsl:for-each>
<!-- -->


/* Agent data access functions*/
<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
    
int get_agent_<xsl:value-of select="xmml:name"/>_MAX_count(){
    return xmachine_memory_<xsl:value-of select="xmml:name"/>_MAX;
}

<xsl:for-each select="xmml:states/gpu:state">
int get_agent_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count(){
	<xsl:if test="../../gpu:type='continuous'">//continuous agent
	return h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count;
	</xsl:if><xsl:if test="../../gpu:type='discrete'">//discrete agent 
	return xmachine_memory_<xsl:value-of select="../../xmml:name"/>_MAX;</xsl:if>
}

xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* get_device_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_agents(){
	return d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>;
}

xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* get_host_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_agents(){
	return h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>;
}
</xsl:for-each>
<xsl:if test="gpu:type='discrete'">
int get_<xsl:value-of select="xmml:name"/>_population_width(){
  return h_xmachine_memory_<xsl:value-of select="xmml:name"/>_pop_width;
}
</xsl:if>

</xsl:for-each>


/*  Analytics Functions */

<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
  <xsl:variable name="agent_name" select="xmml:name"/>
<xsl:for-each select="xmml:states/gpu:state">
  <xsl:variable name="state" select="xmml:name"/>
<xsl:for-each select="../../xmml:memory/gpu:variable">
<xsl:value-of select="xmml:type"/> reduce_<xsl:value-of select="$agent_name"/>_<xsl:value-of select="$state"/>_<xsl:value-of select="xmml:name"/>_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state"/>-><xsl:value-of select="xmml:name"/>),  thrust::device_pointer_cast(d_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state"/>-><xsl:value-of select="xmml:name"/>) + h_xmachine_memory_<xsl:value-of select="$agent_name"/>_<xsl:value-of select="$state"/>_count);
}

<xsl:if test="xmml:type='int'">
<xsl:value-of select="xmml:type"/> count_<xsl:value-of select="$agent_name"/>_<xsl:value-of select="$state"/>_<xsl:value-of select="xmml:name"/>_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state"/>-><xsl:value-of select="xmml:name"/>),  thrust::device_pointer_cast(d_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state"/>-><xsl:value-of select="xmml:name"/>) + h_xmachine_memory_<xsl:value-of select="$agent_name"/>_<xsl:value-of select="$state"/>_count, count_value);
}
</xsl:if>

</xsl:for-each>
</xsl:for-each>
</xsl:for-each>


/* Agent functions */

<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:functions/gpu:function">
	
/* Shared memory size calculator for agent function */
int <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  <xsl:if test="xmml:inputs/gpu:input"><xsl:variable name="messageName" select="xmml:inputs/gpu:input/xmml:messageName"/>
	<xsl:if test="../../gpu:type='continuous'"><xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messageName]">
	<xsl:if test="gpu:partitioningNone">//Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>));
	</xsl:if><xsl:if test="gpu:partitioningDiscrete">//Continuous agent and message input has discrete partitioning
	//Will be reading using texture lookups so sm size can stay the same but need to hold range and width
	sm_size += (blockSize * sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>));
	</xsl:if><xsl:if test="gpu:partitioningSpatial">//Continuous agent and message input is spatially partitioned
	sm_size += (blockSize * sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>));
	</xsl:if>
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	</xsl:for-each>
	</xsl:if><xsl:if test="../../gpu:type='discrete'">
	<xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messageName]">
	<xsl:if test="gpu:partitioningNone  or gpu:partitioningSpatial">//Discrete agent and continuous message input
	sm_size += (blockSize * sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>));
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	</xsl:if><xsl:if test="gpu:partitioningDiscrete">//Discrete agent and message input has discrete partitioning
	int sm_grid_width = (int)ceil(sqrt(blockSize));
	int sm_grid_size = (int)pow((float)sm_grid_width+(h_message_<xsl:value-of select="xmml:name"/>_range*2), 2);
	sm_size += (sm_grid_size *sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>)); //update sm size
	sm_size += (sm_grid_size * PADDING);  //offset for avoiding conflicts
	</xsl:if></xsl:for-each></xsl:if></xsl:if>
	return sm_size;
}

/** <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>
 * Agent function prototype for <xsl:value-of select="xmml:name"/> function of <xsl:value-of select="../../xmml:name"/> agent
 */
void <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>(cudaStream_t &amp;stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	<xsl:if test="../../gpu:type='continuous'">
	if (h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count == 0)
	{
		return;
	}
	</xsl:if>
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count;

	<xsl:if test="xmml:xagentOutputs/gpu:xagentOutput">
	<xsl:for-each select="xmml:xagentOutputs/gpu:xagentOutput">
	<xsl:variable name="xagent_output" select="xmml:xagentName"/><xsl:if test="../../../../../gpu:xagent[xmml:name=$xagent_output]/gpu:type='continuous'">
	//FOR <xsl:value-of select="xmml:xagentName"/> AGENT OUTPUT, RESET THE AGENT NEW LIST SCAN INPUT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, reset_<xsl:value-of select="xmml:xagentName"/>_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_<xsl:value-of select="xmml:xagentName"/>_scan_input&lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_<xsl:value-of select="xmml:xagentName"/>s_new);
	gpuErrchkLaunch();
	</xsl:if></xsl:for-each></xsl:if>

	//******************************** AGENT FUNCTION CONDITION *********************
	<xsl:choose>
	<xsl:when test="xmml:condition"><xsl:if test="../../gpu:type='continuous'">//CONTINUOUS AGENT FUNCTION AND THERE IS A FUNCTION CONDITION
  	
	//COPY CURRENT STATE COUNT TO WORKING COUNT (host and device)
	h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count = h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, sizeof(int)));	
	
	//RESET SCAN INPUTS
	//reset scan input for currentState
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, reset_<xsl:value-of select="../../xmml:name"/>_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_<xsl:value-of select="../../xmml:name"/>_scan_input&lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>);
	gpuErrchkLaunch();
	//reset scan input for working lists
	reset_<xsl:value-of select="../../xmml:name"/>_scan_input&lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s);
	gpuErrchkLaunch();

	//APPLY FUNCTION FILTER
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, <xsl:value-of select="xmml:name"/>_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	<xsl:value-of select="xmml:name"/>_function_filter&lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>, d_<xsl:value-of select="../../xmml:name"/>s);
	gpuErrchkLaunch();

	//GRID AND BLOCK SIZE FOR COMPACT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, scatter_<xsl:value-of select="../../xmml:name"/>_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	
	//COMPACT CURRENT STATE LIST
	thrust::exclusive_scan(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>->_scan_input), thrust::device_pointer_cast(d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>->_scan_input) + h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, thrust::device_pointer_cast(d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>->_position));
	//reset agent count
	gpuErrchk( cudaMemcpy( &amp;scan_last_sum, &amp;d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>->_position[h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &amp;scan_last_included, &amp;d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>->_scan_input[h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count = scan_last_sum+1;
	else		
		h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count = scan_last_sum;
	//Scatter into swap
	scatter_<xsl:value-of select="../../xmml:name"/>_Agents&lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s_swap, d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>, 0, h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count);
	gpuErrchkLaunch();
	//use a temp pointer change working swap list with current state list
	xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* <xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>_temp = d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>;
	d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/> = d_<xsl:value-of select="../../xmml:name"/>s_swap;
	d_<xsl:value-of select="../../xmml:name"/>s_swap = <xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count, sizeof(int)));	
		
	//COMPACT WORKING STATE LIST
	thrust::exclusive_scan(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_<xsl:value-of select="../../xmml:name"/>s->_scan_input), thrust::device_pointer_cast(d_<xsl:value-of select="../../xmml:name"/>s->_scan_input) + h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, thrust::device_pointer_cast(d_<xsl:value-of select="../../xmml:name"/>s->_position));
	//reset agent count
	gpuErrchk( cudaMemcpy( &amp;scan_last_sum, &amp;d_<xsl:value-of select="../../xmml:name"/>s->_position[h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &amp;scan_last_included, &amp;d_<xsl:value-of select="../../xmml:name"/>s->_scan_input[h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//Scatter into swap
	scatter_<xsl:value-of select="../../xmml:name"/>_Agents&lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s_swap, d_<xsl:value-of select="../../xmml:name"/>s, 0, h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count);
	gpuErrchkLaunch();
	//update working agent count after the scatter
	if (scan_last_included == 1)
		h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count = scan_last_sum+1;
	else		
		h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count = scan_last_sum;
    //use a temp pointer change working swap list with current state list
	xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* <xsl:value-of select="../../xmml:name"/>s_temp = d_<xsl:value-of select="../../xmml:name"/>s;
	d_<xsl:value-of select="../../xmml:name"/>s = d_<xsl:value-of select="../../xmml:name"/>s_swap;
	d_<xsl:value-of select="../../xmml:name"/>s_swap = <xsl:value-of select="../../xmml:name"/>s_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, sizeof(int)));	
	
	//CHECK WORKING LIST COUNT IS NOT EQUAL TO 0
	if (h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count == 0)
	{
		return;
	}
	
	<xsl:if test="../../gpu:type='continuous'">//Update the state list size for occupancy calculations
	state_list_size = h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count;
	</xsl:if>
			
	</xsl:if></xsl:when><xsl:when test="gpu:globalCondition">//THERE IS A GLOBAL CONDITION
	
	//COPY CURRENT STATE COUNT TO WORKING COUNT (host and device)
	h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count = h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, sizeof(int)));	
	
	//RESET SCAN INPUTS
	//reset scan input for currentState
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, reset_<xsl:value-of select="../../xmml:name"/>_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_<xsl:value-of select="../../xmml:name"/>_scan_input&lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>);
	gpuErrchkLaunch();
	
	//APPLY FUNCTION FILTER
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, <xsl:value-of select="xmml:name"/>_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	<xsl:value-of select="xmml:name"/>_function_filter&lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>);
	gpuErrchkLaunch();
	
	//GET CONDTIONS TRUE COUNT FROM CURRENT STATE LIST
    thrust::exclusive_scan(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>->_scan_input),  thrust::device_pointer_cast(d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>->_scan_input) + h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, thrust::device_pointer_cast(d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>->_position));
	//reset agent count
	gpuErrchk( cudaMemcpy( &amp;scan_last_sum, &amp;d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>->_position[h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &amp;scan_last_included, &amp;d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>->_scan_input[h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	int global_conditions_true = 0;
	if (scan_last_included == 1)
		global_conditions_true = scan_last_sum+1;
	else		
		global_conditions_true = scan_last_sum;
	//check if condition is true for all agents or if max condition count is reached
	if ((global_conditions_true <xsl:choose><xsl:when test="gpu:globalCondition/gpu:mustEvaluateTo='true'">!</xsl:when><xsl:otherwise>=</xsl:otherwise></xsl:choose>= h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count)&amp;&amp;(h_<xsl:value-of select="xmml:name"/>_condition_count &lt; <xsl:value-of select="gpu:globalCondition/gpu:maxItterations"/>))
	{
		h_<xsl:value-of select="xmml:name"/>_condition_count ++;
		return;
	}
	if ((h_<xsl:value-of select="xmml:name"/>_condition_count == <xsl:value-of select="gpu:globalCondition/gpu:maxItterations"/>))
	{
		printf("Global agent condition for <xsl:value-of select="xmml:name"/> function reached the maximum number of <xsl:value-of select="gpu:globalCondition/gpu:maxItterations"/> conditions\n");
	}
	
	//RESET THE CONDITION COUNT
	h_<xsl:value-of select="xmml:name"/>_condition_count = 0;
	
	//MAP CURRENT STATE TO WORKING LIST
	xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* <xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>_temp = d_<xsl:value-of select="../../xmml:name"/>s;
	d_<xsl:value-of select="../../xmml:name"/>s = d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>;
	d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/> = <xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>_temp;
	//set current state count to 0
	h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, sizeof(int)));	
	
	
	</xsl:when><xsl:otherwise>//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* <xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>_temp = d_<xsl:value-of select="../../xmml:name"/>s;
	d_<xsl:value-of select="../../xmml:name"/>s = d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>;
	d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/> = <xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>_temp;
	//set working count to current state count
	h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count = h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count, sizeof(int)));	
	</xsl:otherwise>
	</xsl:choose>
 

	//******************************** AGENT FUNCTION *******************************

	<xsl:if test="xmml:outputs/gpu:output"><xsl:if test="../../gpu:type='continuous'">
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_<xsl:value-of select="xmml:outputs/gpu:output/xmml:messageName"/>_count + h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count > xmachine_message_<xsl:value-of select="xmml:outputs/gpu:output/xmml:messageName"/>_MAX){
		printf("Error: Buffer size of <xsl:value-of select="xmml:outputs/gpu:output/xmml:messageName"/> message will be exceeded in function <xsl:value-of select="xmml:name"/>\n");
		exit(0);
	}
	</xsl:if></xsl:if>
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, GPUFLAME_<xsl:value-of select="xmml:name"/>, <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_sm_size, state_list_size);<xsl:if test="../../gpu:type='continuous'">
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	</xsl:if><xsl:if test="../../gpu:type='discrete'">
	blockSize = lowest_sqr_pow2(blockSize); //For discrete agents the block size must be a square power of 2
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = (int)sqrt(blockSize);
	b.y = b.x;
	g.x = (int)sqrt(gridSize);
	g.y = g.x;</xsl:if>
	sm_size = <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_sm_size(blockSize);
	
	
	<xsl:if test="xmml:inputs/gpu:input"><xsl:variable name="messageName" select="xmml:inputs/gpu:input/xmml:messageName"/>
	<xsl:if test="../../gpu:type='discrete'"><xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messageName]"><xsl:if test="gpu:partitioningDiscrete">
	//check that the range is not greater than the square of the block size. If so then there will be too many uncoalesded reads
	if (h_message_<xsl:value-of select="xmml:name"/>_range > (int)blockSize){
		printf("ERROR: Message range is greater than the thread block size. Increase thread block size or reduce the range!");
		exit(0);
	}
	</xsl:if></xsl:for-each></xsl:if></xsl:if>
	
	<xsl:if test="xmml:inputs/gpu:input"><xsl:variable name="messageName" select="xmml:inputs/gpu:input/xmml:messageName"/>
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	<xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messageName]">
	<xsl:if test="gpu:partitioningDiscrete or gpu:partitioningSpatial">//any agent with discrete or partitioned message input uses texture caching
	<xsl:for-each select="xmml:variables/gpu:variable">size_t tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_byte_offset;    
	gpuErrchk( cudaBindTexture(&amp;tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_byte_offset, tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>, d_<xsl:value-of select="../../xmml:name"/>s-><xsl:value-of select="xmml:name"/>, sizeof(<xsl:value-of select="xmml:type"/>)*xmachine_message_<xsl:value-of select="../../xmml:name"/>_MAX));
	h_tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_offset = (int)tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_byte_offset / sizeof(<xsl:value-of select="xmml:type"/>);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_offset, &amp;h_tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_offset, sizeof(int)));
	</xsl:for-each><xsl:if test="gpu:partitioningSpatial">//bind pbm start and end indices to textures
	size_t tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_start_byte_offset;
	size_t tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_end_or_count_byte_offset;
	gpuErrchk( cudaBindTexture(&amp;tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_start_byte_offset, tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_start, d_<xsl:value-of select="xmml:name"/>_partition_matrix->start, sizeof(int)*xmachine_message_<xsl:value-of select="xmml:name"/>_grid_size));
	h_tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_start_offset = (int)tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_start_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_start_offset, &amp;h_tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_start_offset, sizeof(int)));
	gpuErrchk( cudaBindTexture(&amp;tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_end_or_count_byte_offset, tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_end_or_count, d_<xsl:value-of select="xmml:name"/>_partition_matrix->end_or_count, sizeof(int)*xmachine_message_<xsl:value-of select="xmml:name"/>_grid_size));
  h_tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_end_or_count_offset = (int)tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_end_or_count_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_end_or_count_offset, &amp;h_tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_end_or_count_offset, sizeof(int)));

	</xsl:if></xsl:if>
	</xsl:for-each></xsl:if>
	
	<xsl:if test="xmml:outputs/gpu:output"><xsl:variable name="messageName" select="xmml:outputs/gpu:output/xmml:messageName"/><xsl:variable name="outputType" select="xmml:outputs/gpu:output/gpu:type"/>
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	<xsl:if test="../../gpu:type='continuous'"><xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messageName]">
	<xsl:if test="gpu:partitioningNone or gpu:partitioningSpatial">//Set the message_type for non partitioned and spatially partitioned message outputs
	h_message_<xsl:value-of select="xmml:name"/>_output_type = <xsl:value-of select="$outputType"/>;
	gpuErrchk( cudaMemcpyToSymbol( d_message_<xsl:value-of select="xmml:name"/>_output_type, &amp;h_message_<xsl:value-of select="xmml:name"/>_output_type, sizeof(int)));
	<xsl:if test="$outputType='optional_message'">//message is optional so reset the swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, reset_<xsl:value-of select="xmml:name"/>_swaps, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_<xsl:value-of select="xmml:name"/>_swaps&lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_<xsl:value-of select="xmml:name"/>s); <!-- Twin Karmakharm Change - Bug found, need to reset the actual message array and not the swap array -->
	gpuErrchkLaunch();
	</xsl:if></xsl:if></xsl:for-each>
	</xsl:if></xsl:if>
	
	
	<xsl:if test="../../gpu:type='continuous'"><xsl:if test="gpu:reallocate='true'">
	//IF CONTINUOUS AGENT CAN REALLOCATE (process dead agents) THEN RESET AGENT SWAPS	
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, reset_<xsl:value-of select="../../xmml:name"/>_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_<xsl:value-of select="../../xmml:name"/>_scan_input&lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s);
	gpuErrchkLaunch();
	</xsl:if></xsl:if>
	
	//MAIN XMACHINE FUNCTION CALL (<xsl:value-of select="xmml:name"/>)
	//Reallocate   : <xsl:choose><xsl:when test="gpu:reallocate='true'">true</xsl:when><xsl:otherwise>false</xsl:otherwise></xsl:choose>
	//Input        : <xsl:value-of select="xmml:inputs/gpu:input/xmml:messageName"/>
	//Output       : <xsl:value-of select="xmml:outputs/gpu:output/xmml:messageName"/>
	//Agent Output : <xsl:value-of select="xmml:xagentOutputs/gpu:xagentOutput/xmml:xagentName"/>
	GPUFLAME_<xsl:value-of select="xmml:name"/>&lt;&lt;&lt;g, b, sm_size, stream&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s<xsl:if test="xmml:xagentOutputs/gpu:xagentOutput">, d_<xsl:value-of select="xmml:xagentOutputs/gpu:xagentOutput/xmml:xagentName"/>s_new</xsl:if>
		<xsl:if test="xmml:inputs/gpu:input"><xsl:variable name="messagename" select="xmml:inputs/gpu:input/xmml:messageName"/>, d_<xsl:value-of select="xmml:inputs/gpu:input/xmml:messageName"/>s<xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messagename]"><xsl:if test="gpu:partitioningSpatial">, d_<xsl:value-of select="xmml:name"/>_partition_matrix</xsl:if></xsl:for-each></xsl:if>
		<xsl:if test="xmml:outputs/gpu:output">, d_<xsl:value-of select="xmml:outputs/gpu:output/xmml:messageName"/>s<xsl:if test="xmml:outputs/gpu:output/xmml:type='optional_message'">_swap</xsl:if></xsl:if>
		<xsl:if test="gpu:RNG='true'">, d_rand48</xsl:if>);
	gpuErrchkLaunch();
	
	<xsl:if test="xmml:inputs/gpu:input"><xsl:variable name="messageName" select="xmml:inputs/gpu:input/xmml:messageName"/>
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	<xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messageName]">
	<xsl:if test="gpu:partitioningDiscrete or gpu:partitioningSpatial">//any agent with discrete or partitioned message input uses texture caching
	<xsl:for-each select="xmml:variables/gpu:variable">gpuErrchk( cudaUnbindTexture(tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>));
	</xsl:for-each><xsl:if test="gpu:partitioningSpatial">//unbind pbm indices
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_start));
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_end_or_count));
    </xsl:if></xsl:if>
	</xsl:for-each></xsl:if>

	<xsl:if test="xmml:outputs/gpu:output"><xsl:variable name="messageName" select="xmml:outputs/gpu:output/xmml:messageName"/><xsl:variable name="outputType" select="xmml:outputs/gpu:output/gpu:type"/><xsl:variable name="xagentName" select="../../xmml:name"/>
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	<xsl:if test="../../gpu:type='continuous'"><xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messageName]">
	<xsl:if test="gpu:partitioningNone or gpu:partitioningSpatial">
	<xsl:if test="$outputType='optional_message'">//<xsl:value-of select="xmml:name"/> Message Type Prefix Sum
	<!-- Twin Karmakharm bug fix 16/09/2014 - Bug found need to swap the message array so that it gets scanned properly -->
	//swap output
	xmachine_message_<xsl:value-of select="xmml:name"/>_list* d_<xsl:value-of select="xmml:name"/>s_scanswap_temp = d_<xsl:value-of select="xmml:name"/>s;
	d_<xsl:value-of select="xmml:name"/>s = d_<xsl:value-of select="xmml:name"/>s_swap;
	d_<xsl:value-of select="xmml:name"/>s_swap = d_<xsl:value-of select="xmml:name"/>s_scanswap_temp;
	<!-- end bug fix -->
    thrust::exclusive_scan(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_<xsl:value-of select="xmml:name"/>s_swap->_scan_input), thrust::device_pointer_cast(d_<xsl:value-of select="xmml:name"/>s_swap->_scan_input) + h_xmachine_memory_<xsl:value-of select="$xagentName"/>_count, thrust::device_pointer_cast(d_<xsl:value-of select="xmml:name"/>s_swap->_position));
	//Scatter
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, scatter_optional_<xsl:value-of select="xmml:name"/>_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_optional_<xsl:value-of select="xmml:name"/>_messages&lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_<xsl:value-of select="xmml:name"/>s, d_<xsl:value-of select="xmml:name"/>s_swap);
	gpuErrchkLaunch();
	</xsl:if></xsl:if>
	</xsl:for-each></xsl:if>
	</xsl:if>
	
	<xsl:if test="xmml:outputs/gpu:output"><xsl:variable name="messageName" select="xmml:outputs/gpu:output/xmml:messageName"/><xsl:variable name="outputType" select="xmml:outputs/gpu:output/gpu:type"/><xsl:variable name="xagentName" select="../../xmml:name"/>
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT <xsl:if test="../../gpu:type='continuous'">
	<xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messageName]">
	<xsl:if test="gpu:partitioningNone or gpu:partitioningSpatial">
	<xsl:if test="$outputType='optional_message'">
	gpuErrchk( cudaMemcpy( &amp;scan_last_sum, &amp;d_<xsl:value-of select="xmml:name"/>s_swap->_position[h_xmachine_memory_<xsl:value-of select="$xagentName"/>_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &amp;scan_last_included, &amp;d_<xsl:value-of select="xmml:name"/>s_swap->_scan_input[h_xmachine_memory_<xsl:value-of select="$xagentName"/>_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//If last item in prefix sum was 1 then increase its index to get the count
	if (scan_last_included == 1){
		h_message_<xsl:value-of select="xmml:name"/>_count += scan_last_sum+1;
	}else{
		h_message_<xsl:value-of select="xmml:name"/>_count += scan_last_sum;
	}
    </xsl:if><xsl:if test="$outputType='single_message'">
	h_message_<xsl:value-of select="xmml:name"/>_count += h_xmachine_memory_<xsl:value-of select="$xagentName"/>_count;
	</xsl:if>//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_<xsl:value-of select="xmml:name"/>_count, &amp;h_message_<xsl:value-of select="xmml:name"/>_count, sizeof(int)));	
	</xsl:if>
	</xsl:for-each>
	</xsl:if>
	</xsl:if>
	
	<xsl:if test="xmml:xagentOutputs/gpu:xagentOutput">
	<xsl:variable name="xagent_output" select="xmml:xagentOutputs/gpu:xagentOutput/xmml:xagentName"/><xsl:if test="../../../gpu:xagent[xmml:name=$xagent_output]/gpu:type='continuous'">
    //COPY ANY AGENT COUNT BEFORE <xsl:value-of select="../../xmml:name"/> AGENTS ARE KILLED (needed for scatter)
	int <xsl:value-of select="../../xmml:name"/>s_pre_death_count = h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count;
	</xsl:if>
	</xsl:if>
	
	<xsl:if test="../../gpu:type='continuous'"><xsl:if test="gpu:reallocate='true'">
	//FOR CONTINUOUS AGENTS WITH REALLOCATION REMOVE POSSIBLE DEAD AGENTS	
    thrust::exclusive_scan(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_<xsl:value-of select="../../xmml:name"/>s->_scan_input), thrust::device_pointer_cast(d_<xsl:value-of select="../../xmml:name"/>s->_scan_input) + h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, thrust::device_pointer_cast(d_<xsl:value-of select="../../xmml:name"/>s->_position));
	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, scatter_<xsl:value-of select="../../xmml:name"/>_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_<xsl:value-of select="../../xmml:name"/>_Agents&lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s_swap, d_<xsl:value-of select="../../xmml:name"/>s, 0, h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count);
	gpuErrchkLaunch();
	//use a temp pointer to make swap default
	xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* <xsl:value-of select="xmml:name"/>_<xsl:value-of select="../../xmml:name"/>s_temp = d_<xsl:value-of select="../../xmml:name"/>s;
	d_<xsl:value-of select="../../xmml:name"/>s = d_<xsl:value-of select="../../xmml:name"/>s_swap;
	d_<xsl:value-of select="../../xmml:name"/>s_swap = <xsl:value-of select="xmml:name"/>_<xsl:value-of select="../../xmml:name"/>s_temp;
	//reset agent count
	gpuErrchk( cudaMemcpy( &amp;scan_last_sum, &amp;d_<xsl:value-of select="../../xmml:name"/>s_swap->_position[h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &amp;scan_last_included, &amp;d_<xsl:value-of select="../../xmml:name"/>s_swap->_scan_input[h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count = scan_last_sum+1;
	else
		h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count = scan_last_sum;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, sizeof(int)));	
	</xsl:if></xsl:if>

	<xsl:if test="xmml:xagentOutputs/gpu:xagentOutput"><xsl:for-each select="xmml:xagentOutputs/gpu:xagentOutput">
	<xsl:variable name="xagent_output" select="xmml:xagentName"/><xsl:if test="../../../../../gpu:xagent[xmml:name=$xagent_output]/gpu:type='continuous'">
	//FOR <xsl:value-of select="xmml:xagentName"/> AGENT OUTPUT SCATTER AGENTS 
    thrust::exclusive_scan(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_<xsl:value-of select="xmml:xagentName"/>s_new->_scan_input), thrust::device_pointer_cast(d_<xsl:value-of select="xmml:xagentName"/>s_new->_scan_input) + <xsl:value-of select="../../../../xmml:name"/>s_pre_death_count, thrust::device_pointer_cast(d_<xsl:value-of select="xmml:xagentName"/>s_new->_position));
	//reset agent count
	int <xsl:value-of select="xmml:xagentName"/>_after_birth_count;
	gpuErrchk( cudaMemcpy( &amp;scan_last_sum, &amp;d_<xsl:value-of select="xmml:xagentName"/>s_new->_position[<xsl:value-of select="../../../../xmml:name"/>s_pre_death_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &amp;scan_last_included, &amp;d_<xsl:value-of select="xmml:xagentName"/>s_new->_scan_input[<xsl:value-of select="../../../../xmml:name"/>s_pre_death_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		<xsl:value-of select="xmml:xagentName"/>_after_birth_count = h_xmachine_memory_<xsl:value-of select="xmml:xagentName"/>_<xsl:value-of select="xmml:state"/>_count + scan_last_sum+1;
	else
		<xsl:value-of select="xmml:xagentName"/>_after_birth_count = h_xmachine_memory_<xsl:value-of select="xmml:xagentName"/>_<xsl:value-of select="xmml:state"/>_count + scan_last_sum;
	//check buffer is not exceeded
	if (<xsl:value-of select="xmml:xagentName"/>_after_birth_count > xmachine_memory_<xsl:value-of select="xmml:xagentName"/>_MAX){
		printf("Error: Buffer size of <xsl:value-of select="xmml:xagentName"/> agents in state <xsl:value-of select="xmml:state"/> will be exceeded writing new agents in function <xsl:value-of select="../../xmml:name"/>\n");
		exit(0);
	}
	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, scatter_<xsl:value-of select="xmml:xagentName"/>_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_<xsl:value-of select="xmml:xagentName"/>_Agents&lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_<xsl:value-of select="xmml:xagentName"/>s_<xsl:value-of select="xmml:state"/>, d_<xsl:value-of select="xmml:xagentName"/>s_new, h_xmachine_memory_<xsl:value-of select="xmml:xagentName"/>_<xsl:value-of select="xmml:state"/>_count, <xsl:value-of select="../../../../xmml:name"/>s_pre_death_count);
	gpuErrchkLaunch();
	//Copy count to device
	h_xmachine_memory_<xsl:value-of select="xmml:xagentName"/>_<xsl:value-of select="xmml:state"/>_count = <xsl:value-of select="xmml:xagentName"/>_after_birth_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="xmml:xagentName"/>_<xsl:value-of select="xmml:state"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="xmml:xagentName"/>_<xsl:value-of select="xmml:state"/>_count, sizeof(int)));	
	</xsl:if></xsl:for-each>
	</xsl:if>
	
	<xsl:if test="xmml:outputs/gpu:output"><xsl:variable name="messageName" select="xmml:outputs/gpu:output/xmml:messageName"/>
	<xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messageName]">
	<xsl:if test="gpu:partitioningSpatial">
	//reset partition matrix
	gpuErrchk( cudaMemset( (void*) d_<xsl:value-of select="xmml:name"/>_partition_matrix, 0, sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>_PBM)));
    //PR Bug fix: Second fix. This should prevent future problems when multiple agents write the same message as now the message structure is completely rebuilt after an output.
    if (h_message_<xsl:value-of select="xmml:name"/>_count > 0){
#ifdef FAST_ATOMIC_SORTING
      //USE ATOMICS TO BUILD PARTITION BOUNDARY
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, hist_<xsl:value-of select="xmml:name"/>_messages, no_sm, h_message_<xsl:value-of select="xmml:name"/>_count); 
	  gridSize = (h_message_<xsl:value-of select="xmml:name"/>_count + blockSize - 1) / blockSize;
	  hist_<xsl:value-of select="xmml:name"/>_messages&lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_xmachine_message_<xsl:value-of select="xmml:name"/>_local_bin_index, d_xmachine_message_<xsl:value-of select="xmml:name"/>_unsorted_index, d_<xsl:value-of select="xmml:name"/>_partition_matrix->end_or_count, d_<xsl:value-of select="xmml:name"/>s, h_message_<xsl:value-of select="xmml:name"/>_count);
	  gpuErrchkLaunch();
	
	  thrust::device_ptr&lt;int&gt; ptr_count = thrust::device_pointer_cast(d_<xsl:value-of select="xmml:name"/>_partition_matrix->end_or_count);
	  thrust::device_ptr&lt;int&gt; ptr_index = thrust::device_pointer_cast(d_<xsl:value-of select="xmml:name"/>_partition_matrix->start);
	  thrust::exclusive_scan(thrust::cuda::par.on(stream), ptr_count, ptr_count + xmachine_message_<xsl:value-of select="xmml:name"/>_grid_size, ptr_index); // scan
	
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, reorder_<xsl:value-of select="xmml:name"/>_messages, no_sm, h_message_<xsl:value-of select="xmml:name"/>_count); 
	  gridSize = (h_message_<xsl:value-of select="xmml:name"/>_count + blockSize - 1) / blockSize; 	// Round up according to array size 
	  reorder_<xsl:value-of select="xmml:name"/>_messages &lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_xmachine_message_<xsl:value-of select="xmml:name"/>_local_bin_index, d_xmachine_message_<xsl:value-of select="xmml:name"/>_unsorted_index, d_<xsl:value-of select="xmml:name"/>_partition_matrix->start, d_<xsl:value-of select="xmml:name"/>s, d_<xsl:value-of select="xmml:name"/>s_swap, h_message_<xsl:value-of select="xmml:name"/>_count);
	  gpuErrchkLaunch();
#else
	  //HASH, SORT, REORDER AND BUILD PMB FOR SPATIAL PARTITIONING MESSAGE OUTPUTS
	  //Get message hash values for sorting
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, hash_<xsl:value-of select="xmml:name"/>_messages, no_sm, h_message_<xsl:value-of select="xmml:name"/>_count); 
	  gridSize = (h_message_<xsl:value-of select="xmml:name"/>_count + blockSize - 1) / blockSize;
	  hash_<xsl:value-of select="xmml:name"/>_messages&lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_xmachine_message_<xsl:value-of select="xmml:name"/>_keys, d_xmachine_message_<xsl:value-of select="xmml:name"/>_values, d_<xsl:value-of select="xmml:name"/>s);
	  gpuErrchkLaunch();
	  //Sort
	  thrust::sort_by_key(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_xmachine_message_<xsl:value-of select="xmml:name"/>_keys),  thrust::device_pointer_cast(d_xmachine_message_<xsl:value-of select="xmml:name"/>_keys) + h_message_<xsl:value-of select="xmml:name"/>_count,  thrust::device_pointer_cast(d_xmachine_message_<xsl:value-of select="xmml:name"/>_values));
	  gpuErrchkLaunch();
	  //reorder and build pcb
	  gpuErrchk(cudaMemset(d_<xsl:value-of select="xmml:name"/>_partition_matrix->start, 0xffffffff, xmachine_message_<xsl:value-of select="xmml:name"/>_grid_size* sizeof(int)));
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, reorder_<xsl:value-of select="xmml:name"/>_messages, reorder_messages_sm_size, h_message_<xsl:value-of select="xmml:name"/>_count); 
	  gridSize = (h_message_<xsl:value-of select="xmml:name"/>_count + blockSize - 1) / blockSize;
	  int reorder_sm_size = reorder_messages_sm_size(blockSize);
	  reorder_<xsl:value-of select="xmml:name"/>_messages&lt;&lt;&lt;gridSize, blockSize, reorder_sm_size, stream&gt;&gt;&gt;(d_xmachine_message_<xsl:value-of select="xmml:name"/>_keys, d_xmachine_message_<xsl:value-of select="xmml:name"/>_values, d_<xsl:value-of select="xmml:name"/>_partition_matrix, d_<xsl:value-of select="xmml:name"/>s, d_<xsl:value-of select="xmml:name"/>s_swap);
	  gpuErrchkLaunch();
#endif
  }
	//swap ordered list
	xmachine_message_<xsl:value-of select="xmml:name"/>_list* d_<xsl:value-of select="xmml:name"/>s_temp = d_<xsl:value-of select="xmml:name"/>s;
	d_<xsl:value-of select="xmml:name"/>s = d_<xsl:value-of select="xmml:name"/>s_swap;
	d_<xsl:value-of select="xmml:name"/>s_swap = d_<xsl:value-of select="xmml:name"/>s_temp;
	</xsl:if>
	</xsl:for-each>
	</xsl:if>
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    <xsl:choose>
    <xsl:when test="../../gpu:type='continuous'">
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:nextState"/>_count+h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count > xmachine_memory_<xsl:value-of select="../../xmml:name"/>_MAX){
		printf("Error: Buffer size of <xsl:value-of select="xmml:name"/> agents in state <xsl:value-of select="xmml:nextState"/> will be exceeded moving working agents to next state in function <xsl:value-of select="xmml:name"/>\n");
		exit(0);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, append_<xsl:value-of select="../../xmml:name"/>_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_<xsl:value-of select="../../xmml:name"/>_Agents&lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:nextState"/>, d_<xsl:value-of select="../../xmml:name"/>s, h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:nextState"/>_count, h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:nextState"/>_count += h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:nextState"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:nextState"/>_count, sizeof(int)));	
	</xsl:when>
    <xsl:when test="../../gpu:type='discrete'">
    //currentState maps to working list
	<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>_temp = d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>;
	d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/> = d_<xsl:value-of select="../../xmml:name"/>s;
	d_<xsl:value-of select="../../xmml:name"/>s = <xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>_temp;
    //set current state count
	h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count = h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count, sizeof(int)));	
	</xsl:when>
  </xsl:choose>
	
}


</xsl:for-each>
    

<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state"> 
extern void reset_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count()
{
    h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count = 0;
}
</xsl:for-each>
    
</xsl:template>
</xsl:stylesheet>
