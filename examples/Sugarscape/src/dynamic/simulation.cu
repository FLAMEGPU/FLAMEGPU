
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

/* agent Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_agent_list* d_agents;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_agent_list* d_agents_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_agent_list* d_agents_new;  /**< Pointer to new agent list on the device (used to hold new agents bfore they are appended to the population)*/
int h_xmachine_memory_agent_count;   /**< Agent population size counter */ 
int h_xmachine_memory_agent_pop_width;   /**< Agent population width */
uint * d_xmachine_memory_agent_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_agent_values;  /**< Agent sort identifiers value */
    
/* agent state variables */
xmachine_memory_agent_list* h_agents_default;      /**< Pointer to agent list (population) on host*/
xmachine_memory_agent_list* d_agents_default;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_agent_default_count;   /**< Agent population size counter */ 


/* Message Memory */

/* cell_state Message variables */
xmachine_message_cell_state_list* h_cell_states;         /**< Pointer to message list on host*/
xmachine_message_cell_state_list* d_cell_states;         /**< Pointer to message list on device*/
xmachine_message_cell_state_list* d_cell_states_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Discrete Partitioning Variables*/
int h_message_cell_state_range;     /**< range of the discrete message*/
int h_message_cell_state_width;     /**< with of the message grid*/
/* Texture offset values for host */
int h_tex_xmachine_message_cell_state_location_id_offset;
int h_tex_xmachine_message_cell_state_state_offset;
int h_tex_xmachine_message_cell_state_env_sugar_level_offset;
/* movement_request Message variables */
xmachine_message_movement_request_list* h_movement_requests;         /**< Pointer to message list on host*/
xmachine_message_movement_request_list* d_movement_requests;         /**< Pointer to message list on device*/
xmachine_message_movement_request_list* d_movement_requests_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Discrete Partitioning Variables*/
int h_message_movement_request_range;     /**< range of the discrete message*/
int h_message_movement_request_width;     /**< with of the message grid*/
/* Texture offset values for host */
int h_tex_xmachine_message_movement_request_agent_id_offset;
int h_tex_xmachine_message_movement_request_location_id_offset;
int h_tex_xmachine_message_movement_request_sugar_level_offset;
int h_tex_xmachine_message_movement_request_metabolism_offset;
/* movement_response Message variables */
xmachine_message_movement_response_list* h_movement_responses;         /**< Pointer to message list on host*/
xmachine_message_movement_response_list* d_movement_responses;         /**< Pointer to message list on device*/
xmachine_message_movement_response_list* d_movement_responses_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Discrete Partitioning Variables*/
int h_message_movement_response_range;     /**< range of the discrete message*/
int h_message_movement_response_width;     /**< with of the message grid*/
/* Texture offset values for host */
int h_tex_xmachine_message_movement_response_location_id_offset;
int h_tex_xmachine_message_movement_response_agent_id_offset;
  
/* CUDA Streams for function layers */
cudaStream_t stream1;

/*Global condition counts*/
int h_metabolise_and_growback_condition_count;


/* RNG rand48 */
RNG_rand48* h_rand48;    /**< Pointer to RNG_rand48 seed list on host*/
RNG_rand48* d_rand48;    /**< Pointer to RNG_rand48 seed list on device*/

/* CUDA Parallel Primatives variables */
int scan_last_sum;           /**< Indicates if the position (in message list) of last message*/
int scan_last_included;      /**< Indicates if last sum value is included in the total sum count*/

/* Agent function prototypes */

/** agent_metabolise_and_growback
 * Agent function prototype for metabolise_and_growback function of agent agent
 */
void agent_metabolise_and_growback(cudaStream_t &stream);

/** agent_output_cell_state
 * Agent function prototype for output_cell_state function of agent agent
 */
void agent_output_cell_state(cudaStream_t &stream);

/** agent_movement_request
 * Agent function prototype for movement_request function of agent agent
 */
void agent_movement_request(cudaStream_t &stream);

/** agent_movement_response
 * Agent function prototype for movement_response function of agent agent
 */
void agent_movement_response(cudaStream_t &stream);

/** agent_movement_transaction
 * Agent function prototype for movement_transaction function of agent agent
 */
void agent_movement_transaction(cudaStream_t &stream);

  
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
	int xmachine_agent_SoA_size = sizeof(xmachine_memory_agent_list);
	h_agents_default = (xmachine_memory_agent_list*)malloc(xmachine_agent_SoA_size);

	/* Message memory allocation (CPU) */
	int message_cell_state_SoA_size = sizeof(xmachine_message_cell_state_list);
	h_cell_states = (xmachine_message_cell_state_list*)malloc(message_cell_state_SoA_size);
	int message_movement_request_SoA_size = sizeof(xmachine_message_movement_request_list);
	h_movement_requests = (xmachine_message_movement_request_list*)malloc(message_movement_request_SoA_size);
	int message_movement_response_SoA_size = sizeof(xmachine_message_movement_response_list);
	h_movement_responses = (xmachine_message_movement_response_list*)malloc(message_movement_response_SoA_size);

	//Exit if agent or message buffer sizes are to small for function outpus
	
	/* Set discrete cell_state message variables (range, width)*/
	h_message_cell_state_range = 1; //from xml
	h_message_cell_state_width = (int)floor(sqrt((float)xmachine_message_cell_state_MAX));
	//check the width
	if (!is_sqr_pow2(xmachine_message_cell_state_MAX)){
		printf("ERROR: cell_state message max must be a square power of 2 for a 2D discrete message grid!\n");
		exit(0);
	}
	gpuErrchk(cudaMemcpyToSymbol( d_message_cell_state_range, &h_message_cell_state_range, sizeof(int)));	
	gpuErrchk(cudaMemcpyToSymbol( d_message_cell_state_width, &h_message_cell_state_width, sizeof(int)));
	
	
	/* Set discrete movement_request message variables (range, width)*/
	h_message_movement_request_range = 1; //from xml
	h_message_movement_request_width = (int)floor(sqrt((float)xmachine_message_movement_request_MAX));
	//check the width
	if (!is_sqr_pow2(xmachine_message_movement_request_MAX)){
		printf("ERROR: movement_request message max must be a square power of 2 for a 2D discrete message grid!\n");
		exit(0);
	}
	gpuErrchk(cudaMemcpyToSymbol( d_message_movement_request_range, &h_message_movement_request_range, sizeof(int)));	
	gpuErrchk(cudaMemcpyToSymbol( d_message_movement_request_width, &h_message_movement_request_width, sizeof(int)));
	
	
	/* Set discrete movement_response message variables (range, width)*/
	h_message_movement_response_range = 1; //from xml
	h_message_movement_response_width = (int)floor(sqrt((float)xmachine_message_movement_response_MAX));
	//check the width
	if (!is_sqr_pow2(xmachine_message_movement_response_MAX)){
		printf("ERROR: movement_response message max must be a square power of 2 for a 2D discrete message grid!\n");
		exit(0);
	}
	gpuErrchk(cudaMemcpyToSymbol( d_message_movement_response_range, &h_message_movement_response_range, sizeof(int)));	
	gpuErrchk(cudaMemcpyToSymbol( d_message_movement_response_width, &h_message_movement_response_width, sizeof(int)));
	
	/* Check that population size is a square power of 2*/
	if (!is_sqr_pow2(xmachine_memory_agent_MAX)){
		printf("ERROR: agents agent count must be a square power of 2!\n");
		exit(0);
	}
	h_xmachine_memory_agent_pop_width = (int)sqrt(xmachine_memory_agent_MAX);
	

	//read initial states
	readInitialStates(inputfile, h_agents_default, &h_xmachine_memory_agent_default_count);
	
	
	/* agent Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_agents, xmachine_agent_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_agents_swap, xmachine_agent_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_agents_new, xmachine_agent_SoA_size));
    
	/* default memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_agents_default, xmachine_agent_SoA_size));
	gpuErrchk( cudaMemcpy( d_agents_default, h_agents_default, xmachine_agent_SoA_size, cudaMemcpyHostToDevice));
    
	/* cell_state Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_cell_states, message_cell_state_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_cell_states_swap, message_cell_state_SoA_size));
	gpuErrchk( cudaMemcpy( d_cell_states, h_cell_states, message_cell_state_SoA_size, cudaMemcpyHostToDevice));
	
	/* movement_request Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_movement_requests, message_movement_request_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_movement_requests_swap, message_movement_request_SoA_size));
	gpuErrchk( cudaMemcpy( d_movement_requests, h_movement_requests, message_movement_request_SoA_size, cudaMemcpyHostToDevice));
	
	/* movement_response Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_movement_responses, message_movement_response_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_movement_responses_swap, message_movement_response_SoA_size));
	gpuErrchk( cudaMemcpy( d_movement_responses, h_movement_responses, message_movement_response_SoA_size, cudaMemcpyHostToDevice));
		

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



void cleanup(){

	/* Agent data free*/
	
	/* agent Agent variables */
	gpuErrchk(cudaFree(d_agents));
	gpuErrchk(cudaFree(d_agents_swap));
	gpuErrchk(cudaFree(d_agents_new));
	
	free( h_agents_default);
	gpuErrchk(cudaFree(d_agents_default));
	

	/* Message data free */
	
	/* cell_state Message variables */
	free( h_cell_states);
	gpuErrchk(cudaFree(d_cell_states));
	gpuErrchk(cudaFree(d_cell_states_swap));
	
	/* movement_request Message variables */
	free( h_movement_requests);
	gpuErrchk(cudaFree(d_movement_requests));
	gpuErrchk(cudaFree(d_movement_requests_swap));
	
	/* movement_response Message variables */
	free( h_movement_responses);
	gpuErrchk(cudaFree(d_movement_responses));
	gpuErrchk(cudaFree(d_movement_responses_swap));
	
  
  /* CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamDestroy(stream1));
}

void singleIteration(){

	/* set all non partitioned and spatial partitionded message counts to 0*/

	/* Call agent functions in order itterating through the layer functions */
	
	/* Layer 1*/
	agent_metabolise_and_growback(stream1);
	cudaDeviceSynchronize();
  
	/* Layer 2*/
	agent_output_cell_state(stream1);
	cudaDeviceSynchronize();
  
	/* Layer 3*/
	agent_movement_request(stream1);
	cudaDeviceSynchronize();
  
	/* Layer 4*/
	agent_movement_response(stream1);
	cudaDeviceSynchronize();
  
	/* Layer 5*/
	agent_movement_transaction(stream1);
	cudaDeviceSynchronize();
  
}

/* Environment functions */



/* Agent data access functions*/

    
int get_agent_agent_MAX_count(){
    return xmachine_memory_agent_MAX;
}


int get_agent_agent_default_count(){
	//discrete agent 
	return xmachine_memory_agent_MAX;
}

xmachine_memory_agent_list* get_device_agent_default_agents(){
	return d_agents_default;
}

xmachine_memory_agent_list* get_host_agent_default_agents(){
	return h_agents_default;
}

int get_agent_population_width(){
  return h_xmachine_memory_agent_pop_width;
}


/* Agent functions */


	
/* Shared memory size calculator for agent function */
int agent_metabolise_and_growback_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** agent_metabolise_and_growback
 * Agent function prototype for metabolise_and_growback function of agent agent
 */
void agent_metabolise_and_growback(cudaStream_t &stream){

	int sm_size;
	int blockSize;
	int minGridSize;
	int gridSize;
	int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_agent_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS A GLOBAL CONDITION
	
	//COPY CURRENT STATE COUNT TO WORKING COUNT (host and device)
	h_xmachine_memory_agent_count = h_xmachine_memory_agent_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_count, &h_xmachine_memory_agent_count, sizeof(int)));	
	
	//RESET SCAN INPUTS
	//reset scan input for currentState
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_agent_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_agent_scan_input<<<gridSize, blockSize, 0, stream>>>(d_agents_default);
	gpuErrchkLaunch();
	
	//APPLY FUNCTION FILTER
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, metabolise_and_growback_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	metabolise_and_growback_function_filter<<<gridSize, blockSize, 0, stream>>>(d_agents_default);
	gpuErrchkLaunch();
	
	//GET CONDTIONS TRUE COUNT FROM CURRENT STATE LIST
    thrust::exclusive_scan(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_agents_default->_scan_input),  thrust::device_pointer_cast(d_agents_default->_scan_input) + h_xmachine_memory_agent_count, thrust::device_pointer_cast(d_agents_default->_position));
	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_agents_default->_position[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_agents_default->_scan_input[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	int global_conditions_true = 0;
	if (scan_last_included == 1)
		global_conditions_true = scan_last_sum+1;
	else		
		global_conditions_true = scan_last_sum;
	//check if condition is true for all agents or if max condition count is reached
	if ((global_conditions_true != h_xmachine_memory_agent_count)&&(h_metabolise_and_growback_condition_count < 9))
	{
		h_metabolise_and_growback_condition_count ++;
		return;
	}
	if ((h_metabolise_and_growback_condition_count == 9))
	{
		printf("Global agent condition for metabolise_and_growback funtion reached the maximum number of 9 conditions\n");
	}
	
	//RESET THE CONDITION COUNT
	h_metabolise_and_growback_condition_count = 0;
	
	//MAP CURRENT STATE TO WORKING LIST
	xmachine_memory_agent_list* agents_default_temp = d_agents;
	d_agents = d_agents_default;
	d_agents_default = agents_default_temp;
	//set current state count to 0
	h_xmachine_memory_agent_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_count, &h_xmachine_memory_agent_count, sizeof(int)));	
	
	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_metabolise_and_growback, agent_metabolise_and_growback_sm_size, state_list_size);
	blockSize = closest_sqr_pow2(blockSize); //For discrete agents the block size must be a square power of 2
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = (int)sqrt(blockSize);
	b.y = b.x;
	g.x = (int)sqrt(gridSize);
	g.y = g.x;
	sm_size = agent_metabolise_and_growback_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (metabolise_and_growback)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_metabolise_and_growback<<<g, b, sm_size, stream>>>(d_agents);
	gpuErrchkLaunch();
	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
    //currentState maps to working list
	agents_default_temp = d_agents_default;
	d_agents_default = d_agents;
	d_agents = agents_default_temp;
    //set current state count
	h_xmachine_memory_agent_default_count = h_xmachine_memory_agent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int agent_output_cell_state_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** agent_output_cell_state
 * Agent function prototype for output_cell_state function of agent agent
 */
void agent_output_cell_state(cudaStream_t &stream){

	int sm_size;
	int blockSize;
	int minGridSize;
	int gridSize;
	int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_agent_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_agent_list* agents_default_temp = d_agents;
	d_agents = d_agents_default;
	d_agents_default = agents_default_temp;
	//set working count to current state count
	h_xmachine_memory_agent_count = h_xmachine_memory_agent_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_count, &h_xmachine_memory_agent_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_agent_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_output_cell_state, agent_output_cell_state_sm_size, state_list_size);
	blockSize = closest_sqr_pow2(blockSize); //For discrete agents the block size must be a square power of 2
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = (int)sqrt(blockSize);
	b.y = b.x;
	g.x = (int)sqrt(gridSize);
	g.y = g.x;
	sm_size = agent_output_cell_state_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	
	
	//MAIN XMACHINE FUNCTION CALL (output_cell_state)
	//Reallocate   : false
	//Input        : 
	//Output       : cell_state
	//Agent Output : 
	GPUFLAME_output_cell_state<<<g, b, sm_size, stream>>>(d_agents, d_cell_states);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
    //currentState maps to working list
	agents_default_temp = d_agents_default;
	d_agents_default = d_agents;
	d_agents = agents_default_temp;
    //set current state count
	h_xmachine_memory_agent_default_count = h_xmachine_memory_agent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int agent_movement_request_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Discrete agent and message input has discrete partitioning
	int sm_grid_width = (int)ceil(sqrt(blockSize));
	int sm_grid_size = (int)pow((float)sm_grid_width+(h_message_cell_state_range*2), 2);
	sm_size += (sm_grid_size *sizeof(xmachine_message_cell_state)); //update sm size
	sm_size += (sm_grid_size * PADDING);  //offset for avoiding conflicts
	
	return sm_size;
}

/** agent_movement_request
 * Agent function prototype for movement_request function of agent agent
 */
void agent_movement_request(cudaStream_t &stream){

	int sm_size;
	int blockSize;
	int minGridSize;
	int gridSize;
	int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_agent_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_agent_list* agents_default_temp = d_agents;
	d_agents = d_agents_default;
	d_agents_default = agents_default_temp;
	//set working count to current state count
	h_xmachine_memory_agent_count = h_xmachine_memory_agent_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_count, &h_xmachine_memory_agent_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_agent_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_movement_request, agent_movement_request_sm_size, state_list_size);
	blockSize = closest_sqr_pow2(blockSize); //For discrete agents the block size must be a square power of 2
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = (int)sqrt(blockSize);
	b.y = b.x;
	g.x = (int)sqrt(gridSize);
	g.y = g.x;
	sm_size = agent_movement_request_sm_size(blockSize);
	
	
	
	//check that the range is not greater than the square of the block size. If so then there will be too many uncoalesded reads
	if (h_message_cell_state_range > (int)blockSize){
		printf("ERROR: Message range is greater than the thread block size. Increase thread block size or reduce the range!");
		exit(0);
	}
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	
	
	//MAIN XMACHINE FUNCTION CALL (movement_request)
	//Reallocate   : false
	//Input        : cell_state
	//Output       : movement_request
	//Agent Output : 
	GPUFLAME_movement_request<<<g, b, sm_size, stream>>>(d_agents, d_cell_states, d_movement_requests);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
    //currentState maps to working list
	agents_default_temp = d_agents_default;
	d_agents_default = d_agents;
	d_agents = agents_default_temp;
    //set current state count
	h_xmachine_memory_agent_default_count = h_xmachine_memory_agent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int agent_movement_response_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Discrete agent and message input has discrete partitioning
	int sm_grid_width = (int)ceil(sqrt(blockSize));
	int sm_grid_size = (int)pow((float)sm_grid_width+(h_message_movement_request_range*2), 2);
	sm_size += (sm_grid_size *sizeof(xmachine_message_movement_request)); //update sm size
	sm_size += (sm_grid_size * PADDING);  //offset for avoiding conflicts
	
	return sm_size;
}

/** agent_movement_response
 * Agent function prototype for movement_response function of agent agent
 */
void agent_movement_response(cudaStream_t &stream){

	int sm_size;
	int blockSize;
	int minGridSize;
	int gridSize;
	int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_agent_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_agent_list* agents_default_temp = d_agents;
	d_agents = d_agents_default;
	d_agents_default = agents_default_temp;
	//set working count to current state count
	h_xmachine_memory_agent_count = h_xmachine_memory_agent_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_count, &h_xmachine_memory_agent_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_agent_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_movement_response, agent_movement_response_sm_size, state_list_size);
	blockSize = closest_sqr_pow2(blockSize); //For discrete agents the block size must be a square power of 2
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = (int)sqrt(blockSize);
	b.y = b.x;
	g.x = (int)sqrt(gridSize);
	g.y = g.x;
	sm_size = agent_movement_response_sm_size(blockSize);
	
	
	
	//check that the range is not greater than the square of the block size. If so then there will be too many uncoalesded reads
	if (h_message_movement_request_range > (int)blockSize){
		printf("ERROR: Message range is greater than the thread block size. Increase thread block size or reduce the range!");
		exit(0);
	}
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	
	
	//MAIN XMACHINE FUNCTION CALL (movement_response)
	//Reallocate   : false
	//Input        : movement_request
	//Output       : movement_response
	//Agent Output : 
	GPUFLAME_movement_response<<<g, b, sm_size, stream>>>(d_agents, d_movement_requests, d_movement_responses, d_rand48);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
    //currentState maps to working list
	agents_default_temp = d_agents_default;
	d_agents_default = d_agents;
	d_agents = agents_default_temp;
    //set current state count
	h_xmachine_memory_agent_default_count = h_xmachine_memory_agent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int agent_movement_transaction_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Discrete agent and message input has discrete partitioning
	int sm_grid_width = (int)ceil(sqrt(blockSize));
	int sm_grid_size = (int)pow((float)sm_grid_width+(h_message_movement_response_range*2), 2);
	sm_size += (sm_grid_size *sizeof(xmachine_message_movement_response)); //update sm size
	sm_size += (sm_grid_size * PADDING);  //offset for avoiding conflicts
	
	return sm_size;
}

/** agent_movement_transaction
 * Agent function prototype for movement_transaction function of agent agent
 */
void agent_movement_transaction(cudaStream_t &stream){

	int sm_size;
	int blockSize;
	int minGridSize;
	int gridSize;
	int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_agent_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_agent_list* agents_default_temp = d_agents;
	d_agents = d_agents_default;
	d_agents_default = agents_default_temp;
	//set working count to current state count
	h_xmachine_memory_agent_count = h_xmachine_memory_agent_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_count, &h_xmachine_memory_agent_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_agent_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_movement_transaction, agent_movement_transaction_sm_size, state_list_size);
	blockSize = closest_sqr_pow2(blockSize); //For discrete agents the block size must be a square power of 2
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = (int)sqrt(blockSize);
	b.y = b.x;
	g.x = (int)sqrt(gridSize);
	g.y = g.x;
	sm_size = agent_movement_transaction_sm_size(blockSize);
	
	
	
	//check that the range is not greater than the square of the block size. If so then there will be too many uncoalesded reads
	if (h_message_movement_response_range > (int)blockSize){
		printf("ERROR: Message range is greater than the thread block size. Increase thread block size or reduce the range!");
		exit(0);
	}
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	
	//MAIN XMACHINE FUNCTION CALL (movement_transaction)
	//Reallocate   : false
	//Input        : movement_response
	//Output       : 
	//Agent Output : 
	GPUFLAME_movement_transaction<<<g, b, sm_size, stream>>>(d_agents, d_movement_responses);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
    //currentState maps to working list
	agents_default_temp = d_agents_default;
	d_agents_default = d_agents;
	d_agents = agents_default_temp;
    //set current state count
	h_xmachine_memory_agent_default_count = h_xmachine_memory_agent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
	
}


 
extern "C" void reset_agent_default_count()
{
    h_xmachine_memory_agent_default_count = 0;
}
