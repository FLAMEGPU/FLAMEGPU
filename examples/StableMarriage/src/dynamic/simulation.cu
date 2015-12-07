
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

/* Man Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_Man_list* d_Mans;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_Man_list* d_Mans_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_Man_list* d_Mans_new;  /**< Pointer to new agent list on the device (used to hold new agents bfore they are appended to the population)*/
int h_xmachine_memory_Man_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_Man_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_Man_values;  /**< Agent sort identifiers value */
    
/* Man state variables */
xmachine_memory_Man_list* h_Mans_unengaged;      /**< Pointer to agent list (population) on host*/
xmachine_memory_Man_list* d_Mans_unengaged;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_Man_unengaged_count;   /**< Agent population size counter */ 

/* Man state variables */
xmachine_memory_Man_list* h_Mans_engaged;      /**< Pointer to agent list (population) on host*/
xmachine_memory_Man_list* d_Mans_engaged;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_Man_engaged_count;   /**< Agent population size counter */ 

/* Woman Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_Woman_list* d_Womans;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_Woman_list* d_Womans_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_Woman_list* d_Womans_new;  /**< Pointer to new agent list on the device (used to hold new agents bfore they are appended to the population)*/
int h_xmachine_memory_Woman_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_Woman_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_Woman_values;  /**< Agent sort identifiers value */
    
/* Woman state variables */
xmachine_memory_Woman_list* h_Womans_default;      /**< Pointer to agent list (population) on host*/
xmachine_memory_Woman_list* d_Womans_default;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_Woman_default_count;   /**< Agent population size counter */ 


/* Message Memory */

/* proposal Message variables */
xmachine_message_proposal_list* h_proposals;         /**< Pointer to message list on host*/
xmachine_message_proposal_list* d_proposals;         /**< Pointer to message list on device*/
xmachine_message_proposal_list* d_proposals_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_proposal_count;         /**< message list counter*/
int h_message_proposal_output_type;   /**< message output type (single or optional)*/

/* notification Message variables */
xmachine_message_notification_list* h_notifications;         /**< Pointer to message list on host*/
xmachine_message_notification_list* d_notifications;         /**< Pointer to message list on device*/
xmachine_message_notification_list* d_notifications_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_notification_count;         /**< message list counter*/
int h_message_notification_output_type;   /**< message output type (single or optional)*/

  
/* CUDA Streams for function layers */
cudaStream_t stream1;

/*Global condition counts*/
int h_check_resolved_condition_count;


/* RNG rand48 */
RNG_rand48* h_rand48;    /**< Pointer to RNG_rand48 seed list on host*/
RNG_rand48* d_rand48;    /**< Pointer to RNG_rand48 seed list on device*/

/* CUDA Parallel Primatives variables */
int scan_last_sum;           /**< Indicates if the position (in message list) of last message*/
int scan_last_included;      /**< Indicates if last sum value is included in the total sum count*/

/* Agent function prototypes */

/** Man_make_proposals
 * Agent function prototype for make_proposals function of Man agent
 */
void Man_make_proposals(cudaStream_t &stream);

/** Man_check_notifications
 * Agent function prototype for check_notifications function of Man agent
 */
void Man_check_notifications(cudaStream_t &stream);

/** Man_check_resolved
 * Agent function prototype for check_resolved function of Man agent
 */
void Man_check_resolved(cudaStream_t &stream);

/** Woman_check_proposals
 * Agent function prototype for check_proposals function of Woman agent
 */
void Woman_check_proposals(cudaStream_t &stream);

/** Woman_notify_suitors
 * Agent function prototype for notify_suitors function of Woman agent
 */
void Woman_notify_suitors(cudaStream_t &stream);

  
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
  

	printf("Allocating Host and Device memeory\n");
  
	/* Agent memory allocation (CPU) */
	int xmachine_Man_SoA_size = sizeof(xmachine_memory_Man_list);
	h_Mans_unengaged = (xmachine_memory_Man_list*)malloc(xmachine_Man_SoA_size);
	h_Mans_engaged = (xmachine_memory_Man_list*)malloc(xmachine_Man_SoA_size);
	int xmachine_Woman_SoA_size = sizeof(xmachine_memory_Woman_list);
	h_Womans_default = (xmachine_memory_Woman_list*)malloc(xmachine_Woman_SoA_size);

	/* Message memory allocation (CPU) */
	int message_proposal_SoA_size = sizeof(xmachine_message_proposal_list);
	h_proposals = (xmachine_message_proposal_list*)malloc(message_proposal_SoA_size);
	int message_notification_SoA_size = sizeof(xmachine_message_notification_list);
	h_notifications = (xmachine_message_notification_list*)malloc(message_notification_SoA_size);

	//Exit if agent or message buffer sizes are to small for function outpus

	//read initial states
	readInitialStates(inputfile, h_Mans_unengaged, &h_xmachine_memory_Man_unengaged_count, h_Womans_default, &h_xmachine_memory_Woman_default_count);
	
	
	/* Man Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Mans, xmachine_Man_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_Mans_swap, xmachine_Man_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_Mans_new, xmachine_Man_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_Man_keys, xmachine_memory_Man_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_Man_values, xmachine_memory_Man_MAX* sizeof(uint)));
	/* unengaged memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Mans_unengaged, xmachine_Man_SoA_size));
	gpuErrchk( cudaMemcpy( d_Mans_unengaged, h_Mans_unengaged, xmachine_Man_SoA_size, cudaMemcpyHostToDevice));
    
	/* engaged memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Mans_engaged, xmachine_Man_SoA_size));
	gpuErrchk( cudaMemcpy( d_Mans_engaged, h_Mans_engaged, xmachine_Man_SoA_size, cudaMemcpyHostToDevice));
    
	/* Woman Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Womans, xmachine_Woman_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_Womans_swap, xmachine_Woman_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_Womans_new, xmachine_Woman_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_Woman_keys, xmachine_memory_Woman_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_Woman_values, xmachine_memory_Woman_MAX* sizeof(uint)));
	/* default memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Womans_default, xmachine_Woman_SoA_size));
	gpuErrchk( cudaMemcpy( d_Womans_default, h_Womans_default, xmachine_Woman_SoA_size, cudaMemcpyHostToDevice));
    
	/* proposal Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_proposals, message_proposal_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_proposals_swap, message_proposal_SoA_size));
	gpuErrchk( cudaMemcpy( d_proposals, h_proposals, message_proposal_SoA_size, cudaMemcpyHostToDevice));
	
	/* notification Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_notifications, message_notification_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_notifications_swap, message_notification_SoA_size));
	gpuErrchk( cudaMemcpy( d_notifications, h_notifications, message_notification_SoA_size, cudaMemcpyHostToDevice));
		

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


void sort_Mans_unengaged(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Man_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_Man_unengaged_count); 
	gridSize = (h_xmachine_memory_Man_unengaged_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_Man_keys, d_xmachine_memory_Man_values, d_Mans_unengaged);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_Man_keys),  thrust::device_pointer_cast(d_xmachine_memory_Man_keys) + h_xmachine_memory_Man_unengaged_count,  thrust::device_pointer_cast(d_xmachine_memory_Man_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_Man_agents, no_sm, h_xmachine_memory_Man_unengaged_count); 
	gridSize = (h_xmachine_memory_Man_unengaged_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_Man_agents<<<gridSize, blockSize>>>(d_xmachine_memory_Man_values, d_Mans_unengaged, d_Mans_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_Man_list* d_Mans_temp = d_Mans_unengaged;
	d_Mans_unengaged = d_Mans_swap;
	d_Mans_swap = d_Mans_temp;	
}

void sort_Mans_engaged(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Man_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_Man_engaged_count); 
	gridSize = (h_xmachine_memory_Man_engaged_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_Man_keys, d_xmachine_memory_Man_values, d_Mans_engaged);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_Man_keys),  thrust::device_pointer_cast(d_xmachine_memory_Man_keys) + h_xmachine_memory_Man_engaged_count,  thrust::device_pointer_cast(d_xmachine_memory_Man_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_Man_agents, no_sm, h_xmachine_memory_Man_engaged_count); 
	gridSize = (h_xmachine_memory_Man_engaged_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_Man_agents<<<gridSize, blockSize>>>(d_xmachine_memory_Man_values, d_Mans_engaged, d_Mans_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_Man_list* d_Mans_temp = d_Mans_engaged;
	d_Mans_engaged = d_Mans_swap;
	d_Mans_swap = d_Mans_temp;	
}

void sort_Womans_default(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Woman_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_Woman_default_count); 
	gridSize = (h_xmachine_memory_Woman_default_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_Woman_keys, d_xmachine_memory_Woman_values, d_Womans_default);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_Woman_keys),  thrust::device_pointer_cast(d_xmachine_memory_Woman_keys) + h_xmachine_memory_Woman_default_count,  thrust::device_pointer_cast(d_xmachine_memory_Woman_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_Woman_agents, no_sm, h_xmachine_memory_Woman_default_count); 
	gridSize = (h_xmachine_memory_Woman_default_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_Woman_agents<<<gridSize, blockSize>>>(d_xmachine_memory_Woman_values, d_Womans_default, d_Womans_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_Woman_list* d_Womans_temp = d_Womans_default;
	d_Womans_default = d_Womans_swap;
	d_Womans_swap = d_Womans_temp;	
}


void cleanup(){

	/* Agent data free*/
	
	/* Man Agent variables */
	gpuErrchk(cudaFree(d_Mans));
	gpuErrchk(cudaFree(d_Mans_swap));
	gpuErrchk(cudaFree(d_Mans_new));
	
	free( h_Mans_unengaged);
	gpuErrchk(cudaFree(d_Mans_unengaged));
	
	free( h_Mans_engaged);
	gpuErrchk(cudaFree(d_Mans_engaged));
	
	/* Woman Agent variables */
	gpuErrchk(cudaFree(d_Womans));
	gpuErrchk(cudaFree(d_Womans_swap));
	gpuErrchk(cudaFree(d_Womans_new));
	
	free( h_Womans_default);
	gpuErrchk(cudaFree(d_Womans_default));
	

	/* Message data free */
	
	/* proposal Message variables */
	free( h_proposals);
	gpuErrchk(cudaFree(d_proposals));
	gpuErrchk(cudaFree(d_proposals_swap));
	
	/* notification Message variables */
	free( h_notifications);
	gpuErrchk(cudaFree(d_notifications));
	gpuErrchk(cudaFree(d_notifications_swap));
	
  
  /* CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamDestroy(stream1));
}

void singleIteration(){

	/* set all non partitioned and spatial partitionded message counts to 0*/
	h_message_proposal_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_proposal_count, &h_message_proposal_count, sizeof(int)));
	
	h_message_notification_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_notification_count, &h_message_notification_count, sizeof(int)));
	

	/* Call agent functions in order itterating through the layer functions */
	
	/* Layer 1*/
	Man_make_proposals(stream1);
	cudaDeviceSynchronize();
  
	/* Layer 2*/
	Woman_check_proposals(stream1);
	cudaDeviceSynchronize();
  
	/* Layer 3*/
	Woman_notify_suitors(stream1);
	cudaDeviceSynchronize();
  
	/* Layer 4*/
	Man_check_notifications(stream1);
	cudaDeviceSynchronize();
  
	/* Layer 5*/
	Man_check_resolved(stream1);
	cudaDeviceSynchronize();
  
}

/* Environment functions */



/* Agent data access functions*/

    
int get_agent_Man_MAX_count(){
    return xmachine_memory_Man_MAX;
}


int get_agent_Man_unengaged_count(){
	//continuous agent
	return h_xmachine_memory_Man_unengaged_count;
	
}

xmachine_memory_Man_list* get_device_Man_unengaged_agents(){
	return d_Mans_unengaged;
}

xmachine_memory_Man_list* get_host_Man_unengaged_agents(){
	return h_Mans_unengaged;
}

int get_agent_Man_engaged_count(){
	//continuous agent
	return h_xmachine_memory_Man_engaged_count;
	
}

xmachine_memory_Man_list* get_device_Man_engaged_agents(){
	return d_Mans_engaged;
}

xmachine_memory_Man_list* get_host_Man_engaged_agents(){
	return h_Mans_engaged;
}

    
int get_agent_Woman_MAX_count(){
    return xmachine_memory_Woman_MAX;
}


int get_agent_Woman_default_count(){
	//continuous agent
	return h_xmachine_memory_Woman_default_count;
	
}

xmachine_memory_Woman_list* get_device_Woman_default_agents(){
	return d_Womans_default;
}

xmachine_memory_Woman_list* get_host_Woman_default_agents(){
	return h_Womans_default;
}


/* Agent functions */


	
/* Shared memory size calculator for agent function */
int Man_make_proposals_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** Man_make_proposals
 * Agent function prototype for make_proposals function of Man agent
 */
void Man_make_proposals(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Man_unengaged_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Man_unengaged_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//CONTINUOUS AGENT FUNCTION AND THERE IS A FUNCTION CONDITION
  	
	//COPY CURRENT STATE COUNT TO WORKING COUNT (host and device)
	h_xmachine_memory_Man_count = h_xmachine_memory_Man_unengaged_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Man_count, &h_xmachine_memory_Man_count, sizeof(int)));	
	
	//RESET SCAN INPUTS
	//reset scan input for currentState
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_Man_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_Man_scan_input<<<gridSize, blockSize, 0, stream>>>(d_Mans_unengaged);
	gpuErrchkLaunch();
	//reset scan input for working lists
	reset_Man_scan_input<<<gridSize, blockSize, 0, stream>>>(d_Mans);
	gpuErrchkLaunch();

	//APPLY FUNCTION FILTER
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, make_proposals_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	make_proposals_function_filter<<<gridSize, blockSize, 0, stream>>>(d_Mans_unengaged, d_Mans);
	gpuErrchkLaunch();

	//GRID AND BLOCK SIZE FOR COMPACT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_Man_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	
	//COMPACT CURRENT STATE LIST
	thrust::exclusive_scan(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_Mans_unengaged->_scan_input), thrust::device_pointer_cast(d_Mans_unengaged->_scan_input) + h_xmachine_memory_Man_count, thrust::device_pointer_cast(d_Mans_unengaged->_position));
	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_Mans_unengaged->_position[h_xmachine_memory_Man_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_Mans_unengaged->_scan_input[h_xmachine_memory_Man_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_Man_unengaged_count = scan_last_sum+1;
	else		
		h_xmachine_memory_Man_unengaged_count = scan_last_sum;
	//Scatter into swap
	scatter_Man_Agents<<<gridSize, blockSize, 0, stream>>>(d_Mans_swap, d_Mans_unengaged, 0, h_xmachine_memory_Man_count);
	gpuErrchkLaunch();
	//use a temp pointer change working swap list with current state list
	xmachine_memory_Man_list* Mans_unengaged_temp = d_Mans_unengaged;
	d_Mans_unengaged = d_Mans_swap;
	d_Mans_swap = Mans_unengaged_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Man_unengaged_count, &h_xmachine_memory_Man_unengaged_count, sizeof(int)));	
		
	//COMPACT WORKING STATE LIST
	thrust::exclusive_scan(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_Mans->_scan_input), thrust::device_pointer_cast(d_Mans->_scan_input) + h_xmachine_memory_Man_count, thrust::device_pointer_cast(d_Mans->_position));
	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_Mans->_position[h_xmachine_memory_Man_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_Mans->_scan_input[h_xmachine_memory_Man_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//Scatter into swap
	scatter_Man_Agents<<<gridSize, blockSize, 0, stream>>>(d_Mans_swap, d_Mans, 0, h_xmachine_memory_Man_count);
	gpuErrchkLaunch();
	//update working agent count after the scatter
	if (scan_last_included == 1)
		h_xmachine_memory_Man_count = scan_last_sum+1;
	else		
		h_xmachine_memory_Man_count = scan_last_sum;
    //use a temp pointer change working swap list with current state list
	xmachine_memory_Man_list* Mans_temp = d_Mans;
	d_Mans = d_Mans_swap;
	d_Mans_swap = Mans_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Man_count, &h_xmachine_memory_Man_count, sizeof(int)));	
	
	//CHECK WORKING LIST COUNT IS NOT EQUAL TO 0
	if (h_xmachine_memory_Man_count == 0)
	{
		return;
	}
	
	//Update the state list size for occupancy calculations
	state_list_size = h_xmachine_memory_Man_count;
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_proposal_count + h_xmachine_memory_Man_count > xmachine_message_proposal_MAX){
		printf("Error: Buffer size of proposal message will be exceeded in function make_proposals\n");
		exit(0);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_make_proposals, Man_make_proposals_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Man_make_proposals_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned and spatially partitioned message outputs
	h_message_proposal_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_proposal_output_type, &h_message_proposal_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (make_proposals)
	//Reallocate   : false
	//Input        : 
	//Output       : proposal
	//Agent Output : 
	GPUFLAME_make_proposals<<<g, b, sm_size, stream>>>(d_Mans, d_proposals);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_proposal_count += h_xmachine_memory_Man_count;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_proposal_count, &h_message_proposal_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Man_unengaged_count+h_xmachine_memory_Man_count > xmachine_memory_Man_MAX){
		printf("Error: Buffer size of make_proposals agents in state unengaged will be exceeded moving working agents to next state in function make_proposals\n");
		exit(0);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_Man_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_Man_Agents<<<gridSize, blockSize, 0, stream>>>(d_Mans_unengaged, d_Mans, h_xmachine_memory_Man_unengaged_count, h_xmachine_memory_Man_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_Man_unengaged_count += h_xmachine_memory_Man_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Man_unengaged_count, &h_xmachine_memory_Man_unengaged_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Man_check_notifications_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_notification));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** Man_check_notifications
 * Agent function prototype for check_notifications function of Man agent
 */
void Man_check_notifications(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Man_unengaged_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Man_unengaged_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_Man_list* Mans_unengaged_temp = d_Mans;
	d_Mans = d_Mans_unengaged;
	d_Mans_unengaged = Mans_unengaged_temp;
	//set working count to current state count
	h_xmachine_memory_Man_count = h_xmachine_memory_Man_unengaged_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Man_count, &h_xmachine_memory_Man_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_Man_unengaged_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Man_unengaged_count, &h_xmachine_memory_Man_unengaged_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_check_notifications, Man_check_notifications_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Man_check_notifications_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	
	//MAIN XMACHINE FUNCTION CALL (check_notifications)
	//Reallocate   : false
	//Input        : notification
	//Output       : 
	//Agent Output : 
	GPUFLAME_check_notifications<<<g, b, sm_size, stream>>>(d_Mans, d_notifications);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Man_unengaged_count+h_xmachine_memory_Man_count > xmachine_memory_Man_MAX){
		printf("Error: Buffer size of check_notifications agents in state unengaged will be exceeded moving working agents to next state in function check_notifications\n");
		exit(0);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_Man_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_Man_Agents<<<gridSize, blockSize, 0, stream>>>(d_Mans_unengaged, d_Mans, h_xmachine_memory_Man_unengaged_count, h_xmachine_memory_Man_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_Man_unengaged_count += h_xmachine_memory_Man_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Man_unengaged_count, &h_xmachine_memory_Man_unengaged_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Man_check_resolved_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** Man_check_resolved
 * Agent function prototype for check_resolved function of Man agent
 */
void Man_check_resolved(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Man_unengaged_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Man_unengaged_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS A GLOBAL CONDITION
	
	//COPY CURRENT STATE COUNT TO WORKING COUNT (host and device)
	h_xmachine_memory_Man_count = h_xmachine_memory_Man_unengaged_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Man_count, &h_xmachine_memory_Man_count, sizeof(int)));	
	
	//RESET SCAN INPUTS
	//reset scan input for currentState
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_Man_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_Man_scan_input<<<gridSize, blockSize, 0, stream>>>(d_Mans_unengaged);
	gpuErrchkLaunch();
	
	//APPLY FUNCTION FILTER
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, check_resolved_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	check_resolved_function_filter<<<gridSize, blockSize, 0, stream>>>(d_Mans_unengaged);
	gpuErrchkLaunch();
	
	//GET CONDTIONS TRUE COUNT FROM CURRENT STATE LIST
    thrust::exclusive_scan(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_Mans_unengaged->_scan_input),  thrust::device_pointer_cast(d_Mans_unengaged->_scan_input) + h_xmachine_memory_Man_count, thrust::device_pointer_cast(d_Mans_unengaged->_position));
	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_Mans_unengaged->_position[h_xmachine_memory_Man_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_Mans_unengaged->_scan_input[h_xmachine_memory_Man_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	int global_conditions_true = 0;
	if (scan_last_included == 1)
		global_conditions_true = scan_last_sum+1;
	else		
		global_conditions_true = scan_last_sum;
	//check if condition is true for all agents or if max condition count is reached
	if ((global_conditions_true != h_xmachine_memory_Man_count)&&(h_check_resolved_condition_count < 5000))
	{
		h_check_resolved_condition_count ++;
		return;
	}
	if ((h_check_resolved_condition_count == 5000))
	{
		printf("Global agent condition for check_resolved funtion reached the maximum number of 5000 conditions\n");
	}
	
	//RESET THE CONDITION COUNT
	h_check_resolved_condition_count = 0;
	
	//MAP CURRENT STATE TO WORKING LIST
	xmachine_memory_Man_list* Mans_unengaged_temp = d_Mans;
	d_Mans = d_Mans_unengaged;
	d_Mans_unengaged = Mans_unengaged_temp;
	//set current state count to 0
	h_xmachine_memory_Man_unengaged_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Man_count, &h_xmachine_memory_Man_count, sizeof(int)));	
	
	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_check_resolved, Man_check_resolved_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Man_check_resolved_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (check_resolved)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_check_resolved<<<g, b, sm_size, stream>>>(d_Mans);
	gpuErrchkLaunch();
	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Man_engaged_count+h_xmachine_memory_Man_count > xmachine_memory_Man_MAX){
		printf("Error: Buffer size of check_resolved agents in state engaged will be exceeded moving working agents to next state in function check_resolved\n");
		exit(0);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_Man_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_Man_Agents<<<gridSize, blockSize, 0, stream>>>(d_Mans_engaged, d_Mans, h_xmachine_memory_Man_engaged_count, h_xmachine_memory_Man_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_Man_engaged_count += h_xmachine_memory_Man_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Man_engaged_count, &h_xmachine_memory_Man_engaged_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Woman_check_proposals_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_proposal));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** Woman_check_proposals
 * Agent function prototype for check_proposals function of Woman agent
 */
void Woman_check_proposals(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Woman_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Woman_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_Woman_list* Womans_default_temp = d_Womans;
	d_Womans = d_Womans_default;
	d_Womans_default = Womans_default_temp;
	//set working count to current state count
	h_xmachine_memory_Woman_count = h_xmachine_memory_Woman_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Woman_count, &h_xmachine_memory_Woman_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_Woman_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Woman_default_count, &h_xmachine_memory_Woman_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_check_proposals, Woman_check_proposals_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Woman_check_proposals_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	
	//MAIN XMACHINE FUNCTION CALL (check_proposals)
	//Reallocate   : false
	//Input        : proposal
	//Output       : 
	//Agent Output : 
	GPUFLAME_check_proposals<<<g, b, sm_size, stream>>>(d_Womans, d_proposals);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Woman_default_count+h_xmachine_memory_Woman_count > xmachine_memory_Woman_MAX){
		printf("Error: Buffer size of check_proposals agents in state default will be exceeded moving working agents to next state in function check_proposals\n");
		exit(0);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_Woman_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_Woman_Agents<<<gridSize, blockSize, 0, stream>>>(d_Womans_default, d_Womans, h_xmachine_memory_Woman_default_count, h_xmachine_memory_Woman_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_Woman_default_count += h_xmachine_memory_Woman_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Woman_default_count, &h_xmachine_memory_Woman_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Woman_notify_suitors_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** Woman_notify_suitors
 * Agent function prototype for notify_suitors function of Woman agent
 */
void Woman_notify_suitors(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Woman_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Woman_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//CONTINUOUS AGENT FUNCTION AND THERE IS A FUNCTION CONDITION
  	
	//COPY CURRENT STATE COUNT TO WORKING COUNT (host and device)
	h_xmachine_memory_Woman_count = h_xmachine_memory_Woman_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Woman_count, &h_xmachine_memory_Woman_count, sizeof(int)));	
	
	//RESET SCAN INPUTS
	//reset scan input for currentState
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_Woman_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_Woman_scan_input<<<gridSize, blockSize, 0, stream>>>(d_Womans_default);
	gpuErrchkLaunch();
	//reset scan input for working lists
	reset_Woman_scan_input<<<gridSize, blockSize, 0, stream>>>(d_Womans);
	gpuErrchkLaunch();

	//APPLY FUNCTION FILTER
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, notify_suitors_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	notify_suitors_function_filter<<<gridSize, blockSize, 0, stream>>>(d_Womans_default, d_Womans);
	gpuErrchkLaunch();

	//GRID AND BLOCK SIZE FOR COMPACT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_Woman_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	
	//COMPACT CURRENT STATE LIST
	thrust::exclusive_scan(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_Womans_default->_scan_input), thrust::device_pointer_cast(d_Womans_default->_scan_input) + h_xmachine_memory_Woman_count, thrust::device_pointer_cast(d_Womans_default->_position));
	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_Womans_default->_position[h_xmachine_memory_Woman_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_Womans_default->_scan_input[h_xmachine_memory_Woman_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_Woman_default_count = scan_last_sum+1;
	else		
		h_xmachine_memory_Woman_default_count = scan_last_sum;
	//Scatter into swap
	scatter_Woman_Agents<<<gridSize, blockSize, 0, stream>>>(d_Womans_swap, d_Womans_default, 0, h_xmachine_memory_Woman_count);
	gpuErrchkLaunch();
	//use a temp pointer change working swap list with current state list
	xmachine_memory_Woman_list* Womans_default_temp = d_Womans_default;
	d_Womans_default = d_Womans_swap;
	d_Womans_swap = Womans_default_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Woman_default_count, &h_xmachine_memory_Woman_default_count, sizeof(int)));	
		
	//COMPACT WORKING STATE LIST
	thrust::exclusive_scan(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_Womans->_scan_input), thrust::device_pointer_cast(d_Womans->_scan_input) + h_xmachine_memory_Woman_count, thrust::device_pointer_cast(d_Womans->_position));
	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_Womans->_position[h_xmachine_memory_Woman_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_Womans->_scan_input[h_xmachine_memory_Woman_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//Scatter into swap
	scatter_Woman_Agents<<<gridSize, blockSize, 0, stream>>>(d_Womans_swap, d_Womans, 0, h_xmachine_memory_Woman_count);
	gpuErrchkLaunch();
	//update working agent count after the scatter
	if (scan_last_included == 1)
		h_xmachine_memory_Woman_count = scan_last_sum+1;
	else		
		h_xmachine_memory_Woman_count = scan_last_sum;
    //use a temp pointer change working swap list with current state list
	xmachine_memory_Woman_list* Womans_temp = d_Womans;
	d_Womans = d_Womans_swap;
	d_Womans_swap = Womans_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Woman_count, &h_xmachine_memory_Woman_count, sizeof(int)));	
	
	//CHECK WORKING LIST COUNT IS NOT EQUAL TO 0
	if (h_xmachine_memory_Woman_count == 0)
	{
		return;
	}
	
	//Update the state list size for occupancy calculations
	state_list_size = h_xmachine_memory_Woman_count;
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_notification_count + h_xmachine_memory_Woman_count > xmachine_message_notification_MAX){
		printf("Error: Buffer size of notification message will be exceeded in function notify_suitors\n");
		exit(0);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_notify_suitors, Woman_notify_suitors_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Woman_notify_suitors_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned and spatially partitioned message outputs
	h_message_notification_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_notification_output_type, &h_message_notification_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (notify_suitors)
	//Reallocate   : false
	//Input        : 
	//Output       : notification
	//Agent Output : 
	GPUFLAME_notify_suitors<<<g, b, sm_size, stream>>>(d_Womans, d_notifications);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_notification_count += h_xmachine_memory_Woman_count;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_notification_count, &h_message_notification_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Woman_default_count+h_xmachine_memory_Woman_count > xmachine_memory_Woman_MAX){
		printf("Error: Buffer size of notify_suitors agents in state default will be exceeded moving working agents to next state in function notify_suitors\n");
		exit(0);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_Woman_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_Woman_Agents<<<gridSize, blockSize, 0, stream>>>(d_Womans_default, d_Womans, h_xmachine_memory_Woman_default_count, h_xmachine_memory_Woman_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_Woman_default_count += h_xmachine_memory_Woman_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Woman_default_count, &h_xmachine_memory_Woman_default_count, sizeof(int)));	
	
	
}


 
extern "C" void reset_Man_unengaged_count()
{
    h_xmachine_memory_Man_unengaged_count = 0;
}
 
extern "C" void reset_Man_engaged_count()
{
    h_xmachine_memory_Man_engaged_count = 0;
}
 
extern "C" void reset_Woman_default_count()
{
    h_xmachine_memory_Woman_default_count = 0;
}
