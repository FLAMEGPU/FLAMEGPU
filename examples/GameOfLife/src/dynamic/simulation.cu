
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

/* cell Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_cell_list* d_cells;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_cell_list* d_cells_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_cell_list* d_cells_new;  /**< Pointer to new agent list on the device (used to hold new agents bfore they are appended to the population)*/
int h_xmachine_memory_cell_count;   /**< Agent population size counter */ 
int h_xmachine_memory_cell_pop_width;   /**< Agent population width */
uint * d_xmachine_memory_cell_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_cell_values;  /**< Agent sort identifiers value */
    
/* cell state variables */
xmachine_memory_cell_list* h_cells_default;      /**< Pointer to agent list (population) on host*/
xmachine_memory_cell_list* d_cells_default;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_cell_default_count;   /**< Agent population size counter */ 


/* Message Memory */

/* state Message variables */
xmachine_message_state_list* h_states;         /**< Pointer to message list on host*/
xmachine_message_state_list* d_states;         /**< Pointer to message list on device*/
xmachine_message_state_list* d_states_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Discrete Partitioning Variables*/
int h_message_state_range;     /**< range of the discrete message*/
int h_message_state_width;     /**< with of the message grid*/
/* Texture offset values for host */
int h_tex_xmachine_message_state_state_offset;
int h_tex_xmachine_message_state_x_offset;
int h_tex_xmachine_message_state_y_offset;
  
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

/** cell_output_state
 * Agent function prototype for output_state function of cell agent
 */
void cell_output_state(cudaStream_t &stream);

/** cell_update_state
 * Agent function prototype for update_state function of cell agent
 */
void cell_update_state(cudaStream_t &stream);

  
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
	int xmachine_cell_SoA_size = sizeof(xmachine_memory_cell_list);
	h_cells_default = (xmachine_memory_cell_list*)malloc(xmachine_cell_SoA_size);

	/* Message memory allocation (CPU) */
	int message_state_SoA_size = sizeof(xmachine_message_state_list);
	h_states = (xmachine_message_state_list*)malloc(message_state_SoA_size);

	//Exit if agent or message buffer sizes are to small for function outpus
	
	/* Set discrete state message variables (range, width)*/
	h_message_state_range = 1; //from xml
	h_message_state_width = (int)floor(sqrt((float)xmachine_message_state_MAX));
	//check the width
	if (!is_sqr_pow2(xmachine_message_state_MAX)){
		printf("ERROR: state message max must be a square power of 2 for a 2D discrete message grid!\n");
		exit(0);
	}
	gpuErrchk(cudaMemcpyToSymbol( d_message_state_range, &h_message_state_range, sizeof(int)));	
	gpuErrchk(cudaMemcpyToSymbol( d_message_state_width, &h_message_state_width, sizeof(int)));
	
	/* Check that population size is a square power of 2*/
	if (!is_sqr_pow2(xmachine_memory_cell_MAX)){
		printf("ERROR: cells agent count must be a square power of 2!\n");
		exit(0);
	}
	h_xmachine_memory_cell_pop_width = (int)sqrt(xmachine_memory_cell_MAX);
	

	//read initial states
	readInitialStates(inputfile, h_cells_default, &h_xmachine_memory_cell_default_count);
	
	
	/* cell Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_cells, xmachine_cell_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_cells_swap, xmachine_cell_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_cells_new, xmachine_cell_SoA_size));
    
	/* default memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_cells_default, xmachine_cell_SoA_size));
	gpuErrchk( cudaMemcpy( d_cells_default, h_cells_default, xmachine_cell_SoA_size, cudaMemcpyHostToDevice));
    
	/* state Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_states, message_state_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_states_swap, message_state_SoA_size));
	gpuErrchk( cudaMemcpy( d_states, h_states, message_state_SoA_size, cudaMemcpyHostToDevice));
		

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
	
	/* cell Agent variables */
	gpuErrchk(cudaFree(d_cells));
	gpuErrchk(cudaFree(d_cells_swap));
	gpuErrchk(cudaFree(d_cells_new));
	
	free( h_cells_default);
	gpuErrchk(cudaFree(d_cells_default));
	

	/* Message data free */
	
	/* state Message variables */
	free( h_states);
	gpuErrchk(cudaFree(d_states));
	gpuErrchk(cudaFree(d_states_swap));
	
  
  /* CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamDestroy(stream1));
}

void singleIteration(){

	/* set all non partitioned and spatial partitionded message counts to 0*/

	/* Call agent functions in order itterating through the layer functions */
	
	/* Layer 1*/
	cell_output_state(stream1);
	cudaDeviceSynchronize();
  
	/* Layer 2*/
	cell_update_state(stream1);
	cudaDeviceSynchronize();
  
}

/* Environment functions */



/* Agent data access functions*/

    
int get_agent_cell_MAX_count(){
    return xmachine_memory_cell_MAX;
}


int get_agent_cell_default_count(){
	//discrete agent 
	return xmachine_memory_cell_MAX;
}

xmachine_memory_cell_list* get_device_cell_default_agents(){
	return d_cells_default;
}

xmachine_memory_cell_list* get_host_cell_default_agents(){
	return h_cells_default;
}

int get_cell_population_width(){
  return h_xmachine_memory_cell_pop_width;
}


/* Agent functions */


	
/* Shared memory size calculator for agent function */
int cell_output_state_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** cell_output_state
 * Agent function prototype for output_state function of cell agent
 */
void cell_output_state(cudaStream_t &stream){

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
	state_list_size = h_xmachine_memory_cell_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_cell_list* cells_default_temp = d_cells;
	d_cells = d_cells_default;
	d_cells_default = cells_default_temp;
	//set working count to current state count
	h_xmachine_memory_cell_count = h_xmachine_memory_cell_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_cell_count, &h_xmachine_memory_cell_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_cell_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_cell_default_count, &h_xmachine_memory_cell_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_output_state, cell_output_state_sm_size, state_list_size);
	blockSize = closest_sqr_pow2(blockSize); //For discrete agents the block size must be a square power of 2
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = (int)sqrt(blockSize);
	b.y = b.x;
	g.x = (int)sqrt(gridSize);
	g.y = g.x;
	sm_size = cell_output_state_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	
	
	//MAIN XMACHINE FUNCTION CALL (output_state)
	//Reallocate   : false
	//Input        : 
	//Output       : state
	//Agent Output : 
	GPUFLAME_output_state<<<g, b, sm_size, stream>>>(d_cells, d_states);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
    //currentState maps to working list
	cells_default_temp = d_cells_default;
	d_cells_default = d_cells;
	d_cells = cells_default_temp;
    //set current state count
	h_xmachine_memory_cell_default_count = h_xmachine_memory_cell_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_cell_default_count, &h_xmachine_memory_cell_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int cell_update_state_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Discrete agent and message input has discrete partitioning
	int sm_grid_width = (int)ceil(sqrt(blockSize));
	int sm_grid_size = (int)pow((float)sm_grid_width+(h_message_state_range*2), 2);
	sm_size += (sm_grid_size *sizeof(xmachine_message_state)); //update sm size
	sm_size += (sm_grid_size * PADDING);  //offset for avoiding conflicts
	
	return sm_size;
}

/** cell_update_state
 * Agent function prototype for update_state function of cell agent
 */
void cell_update_state(cudaStream_t &stream){

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
	state_list_size = h_xmachine_memory_cell_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_cell_list* cells_default_temp = d_cells;
	d_cells = d_cells_default;
	d_cells_default = cells_default_temp;
	//set working count to current state count
	h_xmachine_memory_cell_count = h_xmachine_memory_cell_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_cell_count, &h_xmachine_memory_cell_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_cell_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_cell_default_count, &h_xmachine_memory_cell_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_update_state, cell_update_state_sm_size, state_list_size);
	blockSize = closest_sqr_pow2(blockSize); //For discrete agents the block size must be a square power of 2
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = (int)sqrt(blockSize);
	b.y = b.x;
	g.x = (int)sqrt(gridSize);
	g.y = g.x;
	sm_size = cell_update_state_sm_size(blockSize);
	
	
	
	//check that the range is not greater than the square of the block size. If so then there will be too many uncoalesded reads
	if (h_message_state_range > (int)blockSize){
		printf("ERROR: Message range is greater than the thread block size. Increase thread block size or reduce the range!");
		exit(0);
	}
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	
	//MAIN XMACHINE FUNCTION CALL (update_state)
	//Reallocate   : false
	//Input        : state
	//Output       : 
	//Agent Output : 
	GPUFLAME_update_state<<<g, b, sm_size, stream>>>(d_cells, d_states);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
    //currentState maps to working list
	cells_default_temp = d_cells_default;
	d_cells_default = d_cells;
	d_cells = cells_default_temp;
    //set current state count
	h_xmachine_memory_cell_default_count = h_xmachine_memory_cell_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_cell_default_count, &h_xmachine_memory_cell_default_count, sizeof(int)));	
	
	
}


 
extern "C" void reset_cell_default_count()
{
    h_xmachine_memory_cell_default_count = 0;
}
