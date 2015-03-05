
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
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <vector_operators.h>

// include FLAME kernels
#include "FLAMEGPU_kernals.cu"

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

/* Boid Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_Boid_list* d_Boids;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_Boid_list* d_Boids_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_Boid_list* d_Boids_new;  /**< Pointer to new agent list on the device (used to hold new agents bfore they are appended to the population)*/
int h_xmachine_memory_Boid_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_Boid_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_Boid_values;  /**< Agent sort identifiers value */
    
/* Boid state variables */
xmachine_memory_Boid_list* h_Boids_default;      /**< Pointer to agent list (population) on host*/
xmachine_memory_Boid_list* d_Boids_default;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_Boid_default_count;   /**< Agent population size counter */ 


/* Message Memory */

/* location Message variables */
xmachine_message_location_list* h_locations;         /**< Pointer to message list on host*/
xmachine_message_location_list* d_locations;         /**< Pointer to message list on device*/
xmachine_message_location_list* d_locations_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_location_count;         /**< message list counter*/
int h_message_location_output_type;   /**< message output type (single or optional)*/

  
/* CUDA Streams for function layers */
cudaStream_t stream1;

/*Global condition counts*/

/* RNG rand48 */
RNG_rand48* h_rand48;    /**< Pointer to RNG_rand48 seed list on host*/
RNG_rand48* d_rand48;    /**< Pointer to RNG_rand48 seed list on device*/

/* CUDA Parallel Primatives variables */
int scan_last_sum;           /**< Indicates if the position (in message list) of last message*/
int scan_last_included;      /**< Indicates if last sum value is included in the total sum count*/

/* Agent function prototypes */

/** Boid_outputdata
 * Agent function prototype for outputdata function of Boid agent
 */
void Boid_outputdata(cudaStream_t &stream);

/** Boid_inputdata
 * Agent function prototype for inputdata function of Boid agent
 */
void Boid_inputdata(cudaStream_t &stream);

  
void setPaddingAndOffset()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	int x64_sys = 0;

	// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
	if (deviceProp.major == 9999 && deviceProp.minor == 9999){
		printf("Error: There is no device supporting CUDA.\n");
		exit(0);
	}
    
    //check if double is used and supported
#ifdef _DOUBLE_SUPPORT_REQUIRED_
	printf("Simulation requires full precision double values\n");
	if ((deviceProp.major < 2)&&(deviceProp.minor < 3)){
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
	gpuErrchk(cudaMemcpyToSymbol( d_SM_START, &SM_START, sizeof(int)));
	gpuErrchk(cudaMemcpyToSymbol( d_PADDING, &PADDING, sizeof(int)));     
}

int closest_sqr_pow2(int x){
	int h, h_d;
	int l, l_d;
	
	//higher bound
	h = (int)pow(4, ceil(log(x)/log(4)));
	h_d = h-x;
	
	//escape early if x is square power of 2
	if (h_d == x)
		return x;
	
	//lower bound		
	l = (int)pow(4, floor(log(x)/log(4)));
	l_d = x-l;
	
	//closest bound
	if(h_d < l_d)
		return h;
	else 
		return l;
}

int is_sqr_pow2(int x){
	int r = (int)pow(4, ceil(log(x)/log(4)));
	return (r == x);
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
  

	printf("Allocating Host and Device memeory\n");
  
	/* Agent memory allocation (CPU) */
	int xmachine_Boid_SoA_size = sizeof(xmachine_memory_Boid_list);
	h_Boids_default = (xmachine_memory_Boid_list*)malloc(xmachine_Boid_SoA_size);

	/* Message memory allocation (CPU) */
	int message_location_SoA_size = sizeof(xmachine_message_location_list);
	h_locations = (xmachine_message_location_list*)malloc(message_location_SoA_size);

	//Exit if agent or message buffer sizes are to small for function outpus

	//read initial states
	readInitialStates(inputfile, h_Boids_default, &h_xmachine_memory_Boid_default_count);
	
	
	/* Boid Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Boids, xmachine_Boid_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_Boids_swap, xmachine_Boid_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_Boids_new, xmachine_Boid_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_Boid_keys, xmachine_memory_Boid_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_Boid_values, xmachine_memory_Boid_MAX* sizeof(uint)));
	/* default memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Boids_default, xmachine_Boid_SoA_size));
	gpuErrchk( cudaMemcpy( d_Boids_default, h_Boids_default, xmachine_Boid_SoA_size, cudaMemcpyHostToDevice));
    
	/* location Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_locations, message_location_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_locations_swap, message_location_SoA_size));
	gpuErrchk( cudaMemcpy( d_locations, h_locations, message_location_SoA_size, cudaMemcpyHostToDevice));
		

	/*Set global condition counts*/

	/* RNG rand48 */
	int h_rand48_SoA_size = sizeof(RNG_rand48);
	h_rand48 = (RNG_rand48*)malloc(h_rand48_SoA_size);
	//allocate on GPU
	gpuErrchk( cudaMalloc( (void**) &d_rand48, h_rand48_SoA_size));
	// calculate strided iteration constants
	static const unsigned long long a = 0x5DEECE66DLL, c = 0xB;
	int seed = 123;
	unsigned long long A, C;
	A = 1LL; C = 0LL;
	for (unsigned int i = 0; i < buffer_size_MAX; ++i) {
		C += A*c;
		A *= a;
	}
	h_rand48->A.x = A & 0xFFFFFFLL;
	h_rand48->A.y = (A >> 24) & 0xFFFFFFLL;
	h_rand48->C.x = C & 0xFFFFFFLL;
	h_rand48->C.y = (C >> 24) & 0xFFFFFFLL;
	// prepare first nThreads random numbers from seed
	unsigned long long x = (((unsigned long long)seed) << 16) | 0x330E;
	for (unsigned int i = 0; i < buffer_size_MAX; ++i) {
		x = a*x + c;
		h_rand48->seeds[i].x = x & 0xFFFFFFLL;
		h_rand48->seeds[i].y = (x >> 24) & 0xFFFFFFLL;
	}
	//copy to device
	gpuErrchk( cudaMemcpy( d_rand48, h_rand48, h_rand48_SoA_size, cudaMemcpyHostToDevice));

	/* Call all init functions */
	
  
  /* Init CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamCreate(&stream1));
} 


void sort_Boids_default(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Boid_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_Boid_default_count); 
	gridSize = (h_xmachine_memory_Boid_default_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_Boid_keys, d_xmachine_memory_Boid_values, d_Boids_default);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_Boid_keys),  thrust::device_pointer_cast(d_xmachine_memory_Boid_keys) + h_xmachine_memory_Boid_default_count,  thrust::device_pointer_cast(d_xmachine_memory_Boid_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_Boid_agents, no_sm, h_xmachine_memory_Boid_default_count); 
	gridSize = (h_xmachine_memory_Boid_default_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_Boid_agents<<<gridSize, blockSize>>>(d_xmachine_memory_Boid_values, d_Boids_default, d_Boids_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_Boid_list* d_Boids_temp = d_Boids_default;
	d_Boids_default = d_Boids_swap;
	d_Boids_swap = d_Boids_temp;	
}


void cleanup(){

	/* Agent data free*/
	
	/* Boid Agent variables */
	gpuErrchk(cudaFree(d_Boids));
	gpuErrchk(cudaFree(d_Boids_swap));
	gpuErrchk(cudaFree(d_Boids_new));
	
	free( h_Boids_default);
	gpuErrchk(cudaFree(d_Boids_default));
	

	/* Message data free */
	
	/* location Message variables */
	free( h_locations);
	gpuErrchk(cudaFree(d_locations));
	gpuErrchk(cudaFree(d_locations_swap));
	
  
  /* CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamDestroy(stream1));
}

void singleIteration(){

	/* set all non partitioned and spatial partitionded message counts to 0*/
	h_message_location_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_location_count, &h_message_location_count, sizeof(int)));
	

	/* Call agent functions in order itterating through the layer functions */
	
	/* Layer 1*/
	Boid_outputdata(stream1);
	cudaDeviceSynchronize();
  
	/* Layer 2*/
	Boid_inputdata(stream1);
	cudaDeviceSynchronize();
  
}

/* Environment functions */



/* Agent data access functions*/

    
int get_agent_Boid_MAX_count(){
    return xmachine_memory_Boid_MAX;
}


int get_agent_Boid_default_count(){
	//continuous agent
	return h_xmachine_memory_Boid_default_count;
	
}

xmachine_memory_Boid_list* get_device_Boid_default_agents(){
	return d_Boids_default;
}

xmachine_memory_Boid_list* get_host_Boid_default_agents(){
	return h_Boids_default;
}


/* Agent functions */


	
/* Shared memory size calculator for agent function */
int Boid_outputdata_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** Boid_outputdata
 * Agent function prototype for outputdata function of Boid agent
 */
void Boid_outputdata(cudaStream_t &stream){

	int sm_size;
	int blockSize;
	int minGridSize;
	int gridSize;
	int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Boid_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Boid_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_Boid_list* Boids_default_temp = d_Boids;
	d_Boids = d_Boids_default;
	d_Boids_default = Boids_default_temp;
	//set working count to current state count
	h_xmachine_memory_Boid_count = h_xmachine_memory_Boid_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Boid_count, &h_xmachine_memory_Boid_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_Boid_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Boid_default_count, &h_xmachine_memory_Boid_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_location_count + h_xmachine_memory_Boid_count > xmachine_message_location_MAX){
		printf("Error: Buffer size of location message will be exceeded in function outputdata\n");
		exit(0);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_outputdata, Boid_outputdata_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Boid_outputdata_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned and spatially partitioned message outputs
	h_message_location_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_location_output_type, &h_message_location_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (outputdata)
	//Reallocate   : false
	//Input        : 
	//Output       : location
	//Agent Output : 
	GPUFLAME_outputdata<<<g, b, sm_size, stream>>>(d_Boids, d_locations);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_location_count += h_xmachine_memory_Boid_count;	
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_location_count, &h_message_location_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Boid_default_count+h_xmachine_memory_Boid_count > xmachine_memory_Boid_MAX){
		printf("Error: Buffer size of outputdata agents in state default will be exceeded moving working agents to next state in function outputdata\n");
		exit(0);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_Boid_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_Boid_Agents<<<gridSize, blockSize, 0, stream>>>(d_Boids_default, d_Boids, h_xmachine_memory_Boid_default_count, h_xmachine_memory_Boid_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_Boid_default_count += h_xmachine_memory_Boid_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Boid_default_count, &h_xmachine_memory_Boid_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Boid_inputdata_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_location));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** Boid_inputdata
 * Agent function prototype for inputdata function of Boid agent
 */
void Boid_inputdata(cudaStream_t &stream){

	int sm_size;
	int blockSize;
	int minGridSize;
	int gridSize;
	int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Boid_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Boid_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_Boid_list* Boids_default_temp = d_Boids;
	d_Boids = d_Boids_default;
	d_Boids_default = Boids_default_temp;
	//set working count to current state count
	h_xmachine_memory_Boid_count = h_xmachine_memory_Boid_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Boid_count, &h_xmachine_memory_Boid_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_Boid_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Boid_default_count, &h_xmachine_memory_Boid_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_inputdata, Boid_inputdata_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Boid_inputdata_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	
	//MAIN XMACHINE FUNCTION CALL (inputdata)
	//Reallocate   : false
	//Input        : location
	//Output       : 
	//Agent Output : 
	GPUFLAME_inputdata<<<g, b, sm_size, stream>>>(d_Boids, d_locations);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Boid_default_count+h_xmachine_memory_Boid_count > xmachine_memory_Boid_MAX){
		printf("Error: Buffer size of inputdata agents in state default will be exceeded moving working agents to next state in function inputdata\n");
		exit(0);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_Boid_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_Boid_Agents<<<gridSize, blockSize, 0, stream>>>(d_Boids_default, d_Boids, h_xmachine_memory_Boid_default_count, h_xmachine_memory_Boid_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_Boid_default_count += h_xmachine_memory_Boid_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Boid_default_count, &h_xmachine_memory_Boid_default_count, sizeof(int)));	
	
	
}


 
extern "C" void reset_Boid_default_count()
{
    h_xmachine_memory_Boid_default_count = 0;
}
