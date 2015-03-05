
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

/* keratinocyte Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_keratinocyte_list* d_keratinocytes;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_keratinocyte_list* d_keratinocytes_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_keratinocyte_list* d_keratinocytes_new;  /**< Pointer to new agent list on the device (used to hold new agents bfore they are appended to the population)*/
int h_xmachine_memory_keratinocyte_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_keratinocyte_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_keratinocyte_values;  /**< Agent sort identifiers value */
    
/* keratinocyte state variables */
xmachine_memory_keratinocyte_list* h_keratinocytes_default;      /**< Pointer to agent list (population) on host*/
xmachine_memory_keratinocyte_list* d_keratinocytes_default;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_keratinocyte_default_count;   /**< Agent population size counter */ 

/* keratinocyte state variables */
xmachine_memory_keratinocyte_list* h_keratinocytes_resolve;      /**< Pointer to agent list (population) on host*/
xmachine_memory_keratinocyte_list* d_keratinocytes_resolve;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_keratinocyte_resolve_count;   /**< Agent population size counter */ 


/* Message Memory */

/* location Message variables */
xmachine_message_location_list* h_locations;         /**< Pointer to message list on host*/
xmachine_message_location_list* d_locations;         /**< Pointer to message list on device*/
xmachine_message_location_list* d_locations_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_location_count;         /**< message list counter*/
int h_message_location_output_type;   /**< message output type (single or optional)*/
/* Spatial Partitioning Variables*/
#ifdef FAST_ATOMIC_SORTING
	uint * d_xmachine_message_location_local_bin_index;	  /**< index offset within the assigned bin */
	uint * d_xmachine_message_location_unsorted_index;		/**< unsorted index (hash) value for message */
#else
	uint * d_xmachine_message_location_keys;	  /**< message sort identifier keys*/
	uint * d_xmachine_message_location_values;  /**< message sort identifier values */
#endif
xmachine_message_location_PBM * d_location_partition_matrix;  /**< Pointer to PCB matrix */
float3 h_message_location_min_bounds;           /**< min bounds (x,y,z) of partitioning environment */
float3 h_message_location_max_bounds;           /**< max bounds (x,y,z) of partitioning environment */
int3 h_message_location_partitionDim;           /**< partition dimensions (x,y,z) of partitioning environment */
float h_message_location_radius;                 /**< partition radius (used to determin the size of the partitions) */
/* Texture offset values for host */
int h_tex_xmachine_message_location_id_offset;
int h_tex_xmachine_message_location_type_offset;
int h_tex_xmachine_message_location_x_offset;
int h_tex_xmachine_message_location_y_offset;
int h_tex_xmachine_message_location_z_offset;
int h_tex_xmachine_message_location_dir_offset;
int h_tex_xmachine_message_location_motility_offset;
int h_tex_xmachine_message_location_range_offset;
int h_tex_xmachine_message_location_iteration_offset;
int h_tex_xmachine_message_location_pbm_start_offset;
int h_tex_xmachine_message_location_pbm_end_or_count_offset;

/* force Message variables */
xmachine_message_force_list* h_forces;         /**< Pointer to message list on host*/
xmachine_message_force_list* d_forces;         /**< Pointer to message list on device*/
xmachine_message_force_list* d_forces_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_force_count;         /**< message list counter*/
int h_message_force_output_type;   /**< message output type (single or optional)*/
/* Spatial Partitioning Variables*/
#ifdef FAST_ATOMIC_SORTING
	uint * d_xmachine_message_force_local_bin_index;	  /**< index offset within the assigned bin */
	uint * d_xmachine_message_force_unsorted_index;		/**< unsorted index (hash) value for message */
#else
	uint * d_xmachine_message_force_keys;	  /**< message sort identifier keys*/
	uint * d_xmachine_message_force_values;  /**< message sort identifier values */
#endif
xmachine_message_force_PBM * d_force_partition_matrix;  /**< Pointer to PCB matrix */
float3 h_message_force_min_bounds;           /**< min bounds (x,y,z) of partitioning environment */
float3 h_message_force_max_bounds;           /**< max bounds (x,y,z) of partitioning environment */
int3 h_message_force_partitionDim;           /**< partition dimensions (x,y,z) of partitioning environment */
float h_message_force_radius;                 /**< partition radius (used to determin the size of the partitions) */
/* Texture offset values for host */
int h_tex_xmachine_message_force_type_offset;
int h_tex_xmachine_message_force_x_offset;
int h_tex_xmachine_message_force_y_offset;
int h_tex_xmachine_message_force_z_offset;
int h_tex_xmachine_message_force_id_offset;
int h_tex_xmachine_message_force_pbm_start_offset;
int h_tex_xmachine_message_force_pbm_end_or_count_offset;

  
/* CUDA Streams for function layers */
cudaStream_t stream1;

/*Global condition counts*/
int h_output_location_condition_count;


/* RNG rand48 */
RNG_rand48* h_rand48;    /**< Pointer to RNG_rand48 seed list on host*/
RNG_rand48* d_rand48;    /**< Pointer to RNG_rand48 seed list on device*/

/* CUDA Parallel Primatives variables */
int scan_last_sum;           /**< Indicates if the position (in message list) of last message*/
int scan_last_included;      /**< Indicates if last sum value is included in the total sum count*/

/* Agent function prototypes */

/** keratinocyte_output_location
 * Agent function prototype for output_location function of keratinocyte agent
 */
void keratinocyte_output_location(cudaStream_t &stream);

/** keratinocyte_cycle
 * Agent function prototype for cycle function of keratinocyte agent
 */
void keratinocyte_cycle(cudaStream_t &stream);

/** keratinocyte_differentiate
 * Agent function prototype for differentiate function of keratinocyte agent
 */
void keratinocyte_differentiate(cudaStream_t &stream);

/** keratinocyte_death_signal
 * Agent function prototype for death_signal function of keratinocyte agent
 */
void keratinocyte_death_signal(cudaStream_t &stream);

/** keratinocyte_migrate
 * Agent function prototype for migrate function of keratinocyte agent
 */
void keratinocyte_migrate(cudaStream_t &stream);

/** keratinocyte_force_resolution_output
 * Agent function prototype for force_resolution_output function of keratinocyte agent
 */
void keratinocyte_force_resolution_output(cudaStream_t &stream);

/** keratinocyte_resolve_forces
 * Agent function prototype for resolve_forces function of keratinocyte agent
 */
void keratinocyte_resolve_forces(cudaStream_t &stream);

  
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
	int xmachine_keratinocyte_SoA_size = sizeof(xmachine_memory_keratinocyte_list);
	h_keratinocytes_default = (xmachine_memory_keratinocyte_list*)malloc(xmachine_keratinocyte_SoA_size);
	h_keratinocytes_resolve = (xmachine_memory_keratinocyte_list*)malloc(xmachine_keratinocyte_SoA_size);

	/* Message memory allocation (CPU) */
	int message_location_SoA_size = sizeof(xmachine_message_location_list);
	h_locations = (xmachine_message_location_list*)malloc(message_location_SoA_size);
	int message_force_SoA_size = sizeof(xmachine_message_force_list);
	h_forces = (xmachine_message_force_list*)malloc(message_force_SoA_size);

	//Exit if agent or message buffer sizes are to small for function outpus
			
	/* Set spatial partitioning location message variables (min_bounds, max_bounds)*/
	h_message_location_radius = (float)125;
	gpuErrchk(cudaMemcpyToSymbol( d_message_location_radius, &h_message_location_radius, sizeof(float)));	
	    h_message_location_min_bounds = make_float3((float)0.0, (float)0.0, (float)0.0);
	gpuErrchk(cudaMemcpyToSymbol( d_message_location_min_bounds, &h_message_location_min_bounds, sizeof(float3)));	
	h_message_location_max_bounds = make_float3((float)500, (float)500, (float)500);
	gpuErrchk(cudaMemcpyToSymbol( d_message_location_max_bounds, &h_message_location_max_bounds, sizeof(float3)));	
	h_message_location_partitionDim.x = (int)ceil((h_message_location_max_bounds.x - h_message_location_min_bounds.x)/h_message_location_radius);
	h_message_location_partitionDim.y = (int)ceil((h_message_location_max_bounds.y - h_message_location_min_bounds.y)/h_message_location_radius);
	h_message_location_partitionDim.z = (int)ceil((h_message_location_max_bounds.z - h_message_location_min_bounds.z)/h_message_location_radius);
	gpuErrchk(cudaMemcpyToSymbol( d_message_location_partitionDim, &h_message_location_partitionDim, sizeof(int3)));	
	
			
	/* Set spatial partitioning force message variables (min_bounds, max_bounds)*/
	h_message_force_radius = (float)15.625;
	gpuErrchk(cudaMemcpyToSymbol( d_message_force_radius, &h_message_force_radius, sizeof(float)));	
	    h_message_force_min_bounds = make_float3((float)0.0, (float)0.0, (float)0.0);
	gpuErrchk(cudaMemcpyToSymbol( d_message_force_min_bounds, &h_message_force_min_bounds, sizeof(float3)));	
	h_message_force_max_bounds = make_float3((float)500, (float)500, (float)500);
	gpuErrchk(cudaMemcpyToSymbol( d_message_force_max_bounds, &h_message_force_max_bounds, sizeof(float3)));	
	h_message_force_partitionDim.x = (int)ceil((h_message_force_max_bounds.x - h_message_force_min_bounds.x)/h_message_force_radius);
	h_message_force_partitionDim.y = (int)ceil((h_message_force_max_bounds.y - h_message_force_min_bounds.y)/h_message_force_radius);
	h_message_force_partitionDim.z = (int)ceil((h_message_force_max_bounds.z - h_message_force_min_bounds.z)/h_message_force_radius);
	gpuErrchk(cudaMemcpyToSymbol( d_message_force_partitionDim, &h_message_force_partitionDim, sizeof(int3)));	
	

	//read initial states
	readInitialStates(inputfile, h_keratinocytes_resolve, &h_xmachine_memory_keratinocyte_resolve_count);
	
	
	/* keratinocyte Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_keratinocytes, xmachine_keratinocyte_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_keratinocytes_swap, xmachine_keratinocyte_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_keratinocytes_new, xmachine_keratinocyte_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_keratinocyte_keys, xmachine_memory_keratinocyte_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_keratinocyte_values, xmachine_memory_keratinocyte_MAX* sizeof(uint)));
	/* default memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_keratinocytes_default, xmachine_keratinocyte_SoA_size));
	gpuErrchk( cudaMemcpy( d_keratinocytes_default, h_keratinocytes_default, xmachine_keratinocyte_SoA_size, cudaMemcpyHostToDevice));
    
	/* resolve memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_keratinocytes_resolve, xmachine_keratinocyte_SoA_size));
	gpuErrchk( cudaMemcpy( d_keratinocytes_resolve, h_keratinocytes_resolve, xmachine_keratinocyte_SoA_size, cudaMemcpyHostToDevice));
    
	/* location Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_locations, message_location_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_locations_swap, message_location_SoA_size));
	gpuErrchk( cudaMemcpy( d_locations, h_locations, message_location_SoA_size, cudaMemcpyHostToDevice));
	gpuErrchk( cudaMalloc( (void**) &d_location_partition_matrix, sizeof(xmachine_message_location_PBM)));
#ifdef FAST_ATOMIC_SORTING
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_location_local_bin_index, xmachine_message_location_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_location_unsorted_index, xmachine_message_location_MAX* sizeof(uint)));
#else
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_location_keys, xmachine_message_location_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_location_values, xmachine_message_location_MAX* sizeof(uint)));
#endif
	
	/* force Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_forces, message_force_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_forces_swap, message_force_SoA_size));
	gpuErrchk( cudaMemcpy( d_forces, h_forces, message_force_SoA_size, cudaMemcpyHostToDevice));
	gpuErrchk( cudaMalloc( (void**) &d_force_partition_matrix, sizeof(xmachine_message_force_PBM)));
#ifdef FAST_ATOMIC_SORTING
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_force_local_bin_index, xmachine_message_force_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_force_unsorted_index, xmachine_message_force_MAX* sizeof(uint)));
#else
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_force_keys, xmachine_message_force_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_force_values, xmachine_message_force_MAX* sizeof(uint)));
#endif
		

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
	setConstants();
	
  
  /* Init CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamCreate(&stream1));
} 


void sort_keratinocytes_default(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_keratinocyte_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_keratinocyte_default_count); 
	gridSize = (h_xmachine_memory_keratinocyte_default_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_keratinocyte_keys, d_xmachine_memory_keratinocyte_values, d_keratinocytes_default);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_keratinocyte_keys),  thrust::device_pointer_cast(d_xmachine_memory_keratinocyte_keys) + h_xmachine_memory_keratinocyte_default_count,  thrust::device_pointer_cast(d_xmachine_memory_keratinocyte_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_keratinocyte_agents, no_sm, h_xmachine_memory_keratinocyte_default_count); 
	gridSize = (h_xmachine_memory_keratinocyte_default_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_keratinocyte_agents<<<gridSize, blockSize>>>(d_xmachine_memory_keratinocyte_values, d_keratinocytes_default, d_keratinocytes_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_keratinocyte_list* d_keratinocytes_temp = d_keratinocytes_default;
	d_keratinocytes_default = d_keratinocytes_swap;
	d_keratinocytes_swap = d_keratinocytes_temp;	
}

void sort_keratinocytes_resolve(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_keratinocyte_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_keratinocyte_resolve_count); 
	gridSize = (h_xmachine_memory_keratinocyte_resolve_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_keratinocyte_keys, d_xmachine_memory_keratinocyte_values, d_keratinocytes_resolve);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_keratinocyte_keys),  thrust::device_pointer_cast(d_xmachine_memory_keratinocyte_keys) + h_xmachine_memory_keratinocyte_resolve_count,  thrust::device_pointer_cast(d_xmachine_memory_keratinocyte_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_keratinocyte_agents, no_sm, h_xmachine_memory_keratinocyte_resolve_count); 
	gridSize = (h_xmachine_memory_keratinocyte_resolve_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_keratinocyte_agents<<<gridSize, blockSize>>>(d_xmachine_memory_keratinocyte_values, d_keratinocytes_resolve, d_keratinocytes_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_keratinocyte_list* d_keratinocytes_temp = d_keratinocytes_resolve;
	d_keratinocytes_resolve = d_keratinocytes_swap;
	d_keratinocytes_swap = d_keratinocytes_temp;	
}


void cleanup(){

	/* Agent data free*/
	
	/* keratinocyte Agent variables */
	gpuErrchk(cudaFree(d_keratinocytes));
	gpuErrchk(cudaFree(d_keratinocytes_swap));
	gpuErrchk(cudaFree(d_keratinocytes_new));
	
	free( h_keratinocytes_default);
	gpuErrchk(cudaFree(d_keratinocytes_default));
	
	free( h_keratinocytes_resolve);
	gpuErrchk(cudaFree(d_keratinocytes_resolve));
	

	/* Message data free */
	
	/* location Message variables */
	free( h_locations);
	gpuErrchk(cudaFree(d_locations));
	gpuErrchk(cudaFree(d_locations_swap));
	gpuErrchk(cudaFree(d_location_partition_matrix));
#ifdef FAST_ATOMIC_SORTING
	gpuErrchk(cudaFree(d_xmachine_message_location_local_bin_index));
	gpuErrchk(cudaFree(d_xmachine_message_location_unsorted_index));
#else
	gpuErrchk(cudaFree(d_xmachine_message_location_keys));
	gpuErrchk(cudaFree(d_xmachine_message_location_values));
#endif
	
	/* force Message variables */
	free( h_forces);
	gpuErrchk(cudaFree(d_forces));
	gpuErrchk(cudaFree(d_forces_swap));
	gpuErrchk(cudaFree(d_force_partition_matrix));
#ifdef FAST_ATOMIC_SORTING
	gpuErrchk(cudaFree(d_xmachine_message_force_local_bin_index));
	gpuErrchk(cudaFree(d_xmachine_message_force_unsorted_index));
#else
	gpuErrchk(cudaFree(d_xmachine_message_force_keys));
	gpuErrchk(cudaFree(d_xmachine_message_force_values));
#endif
	
  
  /* CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamDestroy(stream1));
}

void singleIteration(){

	/* set all non partitioned and spatial partitionded message counts to 0*/
	h_message_location_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_location_count, &h_message_location_count, sizeof(int)));
	
	h_message_force_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_force_count, &h_message_force_count, sizeof(int)));
	

	/* Call agent functions in order itterating through the layer functions */
	
	/* Layer 1*/
	keratinocyte_output_location(stream1);
	cudaDeviceSynchronize();
  
	/* Layer 2*/
	keratinocyte_cycle(stream1);
	cudaDeviceSynchronize();
  
	/* Layer 3*/
	keratinocyte_differentiate(stream1);
	cudaDeviceSynchronize();
  
	/* Layer 4*/
	keratinocyte_death_signal(stream1);
	cudaDeviceSynchronize();
  
	/* Layer 5*/
	keratinocyte_migrate(stream1);
	cudaDeviceSynchronize();
  
	/* Layer 6*/
	keratinocyte_force_resolution_output(stream1);
	cudaDeviceSynchronize();
  
	/* Layer 7*/
	keratinocyte_resolve_forces(stream1);
	cudaDeviceSynchronize();
  
}

/* Environment functions */


void set_calcium_level(float* h_calcium_level){
	gpuErrchk(cudaMemcpyToSymbol(calcium_level, h_calcium_level, sizeof(float)));
}

void set_CYCLE_LENGTH(int* h_CYCLE_LENGTH){
	gpuErrchk(cudaMemcpyToSymbol(CYCLE_LENGTH, h_CYCLE_LENGTH, sizeof(int)*5));
}

void set_SUBSTRATE_FORCE(float* h_SUBSTRATE_FORCE){
	gpuErrchk(cudaMemcpyToSymbol(SUBSTRATE_FORCE, h_SUBSTRATE_FORCE, sizeof(float)*5));
}

void set_DOWNWARD_FORCE(float* h_DOWNWARD_FORCE){
	gpuErrchk(cudaMemcpyToSymbol(DOWNWARD_FORCE, h_DOWNWARD_FORCE, sizeof(float)*5));
}

void set_FORCE_MATRIX(float* h_FORCE_MATRIX){
	gpuErrchk(cudaMemcpyToSymbol(FORCE_MATRIX, h_FORCE_MATRIX, sizeof(float)*25));
}

void set_FORCE_REP(float* h_FORCE_REP){
	gpuErrchk(cudaMemcpyToSymbol(FORCE_REP, h_FORCE_REP, sizeof(float)));
}

void set_FORCE_DAMPENER(float* h_FORCE_DAMPENER){
	gpuErrchk(cudaMemcpyToSymbol(FORCE_DAMPENER, h_FORCE_DAMPENER, sizeof(float)));
}

void set_BASEMENT_MAX_Z(int* h_BASEMENT_MAX_Z){
	gpuErrchk(cudaMemcpyToSymbol(BASEMENT_MAX_Z, h_BASEMENT_MAX_Z, sizeof(int)));
}


/* Agent data access functions*/

    
int get_agent_keratinocyte_MAX_count(){
    return xmachine_memory_keratinocyte_MAX;
}


int get_agent_keratinocyte_default_count(){
	//continuous agent
	return h_xmachine_memory_keratinocyte_default_count;
	
}

xmachine_memory_keratinocyte_list* get_device_keratinocyte_default_agents(){
	return d_keratinocytes_default;
}

xmachine_memory_keratinocyte_list* get_host_keratinocyte_default_agents(){
	return h_keratinocytes_default;
}

int get_agent_keratinocyte_resolve_count(){
	//continuous agent
	return h_xmachine_memory_keratinocyte_resolve_count;
	
}

xmachine_memory_keratinocyte_list* get_device_keratinocyte_resolve_agents(){
	return d_keratinocytes_resolve;
}

xmachine_memory_keratinocyte_list* get_host_keratinocyte_resolve_agents(){
	return h_keratinocytes_resolve;
}


/* Agent functions */


	
/* Shared memory size calculator for agent function */
int keratinocyte_output_location_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** keratinocyte_output_location
 * Agent function prototype for output_location function of keratinocyte agent
 */
void keratinocyte_output_location(cudaStream_t &stream){

	int sm_size;
	int blockSize;
	int minGridSize;
	int gridSize;
	int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_keratinocyte_resolve_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_keratinocyte_resolve_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS A GLOBAL CONDITION
	
	//COPY CURRENT STATE COUNT TO WORKING COUNT (host and device)
	h_xmachine_memory_keratinocyte_count = h_xmachine_memory_keratinocyte_resolve_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_count, &h_xmachine_memory_keratinocyte_count, sizeof(int)));	
	
	//RESET SCAN INPUTS
	//reset scan input for currentState
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_keratinocyte_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_keratinocyte_scan_input<<<gridSize, blockSize, 0, stream>>>(d_keratinocytes_resolve);
	gpuErrchkLaunch();
	
	//APPLY FUNCTION FILTER
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, output_location_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	output_location_function_filter<<<gridSize, blockSize, 0, stream>>>(d_keratinocytes_resolve);
	gpuErrchkLaunch();
	
	//GET CONDTIONS TRUE COUNT FROM CURRENT STATE LIST
    thrust::exclusive_scan(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_keratinocytes_resolve->_scan_input),  thrust::device_pointer_cast(d_keratinocytes_resolve->_scan_input) + h_xmachine_memory_keratinocyte_count, thrust::device_pointer_cast(d_keratinocytes_resolve->_position));
	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_keratinocytes_resolve->_position[h_xmachine_memory_keratinocyte_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_keratinocytes_resolve->_scan_input[h_xmachine_memory_keratinocyte_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	int global_conditions_true = 0;
	if (scan_last_included == 1)
		global_conditions_true = scan_last_sum+1;
	else		
		global_conditions_true = scan_last_sum;
	//check if condition is true for all agents or if max condition count is reached
	if ((global_conditions_true != h_xmachine_memory_keratinocyte_count)&&(h_output_location_condition_count < 200))
	{
		h_output_location_condition_count ++;
		return;
	}
	if ((h_output_location_condition_count == 200))
	{
		printf("Global agent condition for output_location funtion reached the maximum number of 200 conditions\n");
	}
	
	//RESET THE CONDITION COUNT
	h_output_location_condition_count = 0;
	
	//MAP CURRENT STATE TO WORKING LIST
	xmachine_memory_keratinocyte_list* keratinocytes_resolve_temp = d_keratinocytes;
	d_keratinocytes = d_keratinocytes_resolve;
	d_keratinocytes_resolve = keratinocytes_resolve_temp;
	//set current state count to 0
	h_xmachine_memory_keratinocyte_resolve_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_count, &h_xmachine_memory_keratinocyte_count, sizeof(int)));	
	
	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_location_count + h_xmachine_memory_keratinocyte_count > xmachine_message_location_MAX){
		printf("Error: Buffer size of location message will be exceeded in function output_location\n");
		exit(0);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_output_location, keratinocyte_output_location_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = keratinocyte_output_location_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned and spatially partitioned message outputs
	h_message_location_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_location_output_type, &h_message_location_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (output_location)
	//Reallocate   : false
	//Input        : 
	//Output       : location
	//Agent Output : 
	GPUFLAME_output_location<<<g, b, sm_size, stream>>>(d_keratinocytes, d_locations);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_location_count += h_xmachine_memory_keratinocyte_count;	
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_location_count, &h_message_location_count, sizeof(int)));	
	
#ifdef FAST_ATOMIC_SORTING
  //USE ATOMICS TO BUILD PARTITION BOUNDARY
	//reset partition matrix
	gpuErrchk( cudaMemset( (void*) d_location_partition_matrix, 0, sizeof(xmachine_message_location_PBM)));
  //
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, hist_location_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	hist_location_messages<<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_location_local_bin_index, d_xmachine_message_location_unsorted_index, d_location_partition_matrix->end_or_count, d_locations, state_list_size);
	gpuErrchkLaunch();
	
	thrust::device_ptr<int> ptr_count = thrust::device_pointer_cast(d_location_partition_matrix->end_or_count);
	thrust::device_ptr<int> ptr_index = thrust::device_pointer_cast(d_location_partition_matrix->start);
	thrust::exclusive_scan(thrust::cuda::par.on(stream), ptr_count, ptr_count + xmachine_message_location_grid_size, ptr_index); // scan
	
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_location_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize; 	// Round up according to array size 
	reorder_location_messages <<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_location_local_bin_index, d_xmachine_message_location_unsorted_index, d_location_partition_matrix->start, d_locations, d_locations_swap, state_list_size);
	gpuErrchkLaunch();
#else
	//HASH, SORT, REORDER AND BUILD PMB FOR SPATIAL PARTITIONING MESSAGE OUTPUTS
	//Get message hash values for sorting
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, hash_location_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	hash_location_messages<<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_location_keys, d_xmachine_message_location_values, d_locations);
	gpuErrchkLaunch();
	//Sort
	thrust::sort_by_key(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_xmachine_message_location_keys),  thrust::device_pointer_cast(d_xmachine_message_location_keys) + h_message_location_count,  thrust::device_pointer_cast(d_xmachine_message_location_values));
	gpuErrchkLaunch();
	//reorder and build pcb
	gpuErrchk(cudaMemset(d_location_partition_matrix->start, 0xffffffff, xmachine_message_location_grid_size* sizeof(int)));
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_location_messages, reorder_messages_sm_size, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	int reorder_sm_size = reorder_messages_sm_size(blockSize);
	reorder_location_messages<<<gridSize, blockSize, reorder_sm_size, stream>>>(d_xmachine_message_location_keys, d_xmachine_message_location_values, d_location_partition_matrix, d_locations, d_locations_swap);
	gpuErrchkLaunch();
#endif
	//swap ordered list
	xmachine_message_location_list* d_locations_temp = d_locations;
	d_locations = d_locations_swap;
	d_locations_swap = d_locations_temp;
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_keratinocyte_default_count+h_xmachine_memory_keratinocyte_count > xmachine_memory_keratinocyte_MAX){
		printf("Error: Buffer size of output_location agents in state default will be exceeded moving working agents to next state in function output_location\n");
		exit(0);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_keratinocyte_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_keratinocyte_Agents<<<gridSize, blockSize, 0, stream>>>(d_keratinocytes_default, d_keratinocytes, h_xmachine_memory_keratinocyte_default_count, h_xmachine_memory_keratinocyte_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_keratinocyte_default_count += h_xmachine_memory_keratinocyte_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_default_count, &h_xmachine_memory_keratinocyte_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int keratinocyte_cycle_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** keratinocyte_cycle
 * Agent function prototype for cycle function of keratinocyte agent
 */
void keratinocyte_cycle(cudaStream_t &stream){

	int sm_size;
	int blockSize;
	int minGridSize;
	int gridSize;
	int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_keratinocyte_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_keratinocyte_default_count;

	
	//FOR keratinocyte AGENT OUTPUT, RESET THE AGENT NEW LIST SCAN INPUT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_keratinocyte_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_keratinocyte_scan_input<<<gridSize, blockSize, 0, stream>>>(d_keratinocytes_new);
	gpuErrchkLaunch();
	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_keratinocyte_list* keratinocytes_default_temp = d_keratinocytes;
	d_keratinocytes = d_keratinocytes_default;
	d_keratinocytes_default = keratinocytes_default_temp;
	//set working count to current state count
	h_xmachine_memory_keratinocyte_count = h_xmachine_memory_keratinocyte_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_count, &h_xmachine_memory_keratinocyte_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_keratinocyte_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_default_count, &h_xmachine_memory_keratinocyte_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_cycle, keratinocyte_cycle_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = keratinocyte_cycle_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (cycle)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : keratinocyte
	GPUFLAME_cycle<<<g, b, sm_size, stream>>>(d_keratinocytes, d_keratinocytes_new, d_rand48);
	gpuErrchkLaunch();
	
	
    //COPY ANY AGENT COUNT BEFORE keratinocyte AGENTS ARE KILLED (needed for scatter)
	int keratinocytes_pre_death_count = h_xmachine_memory_keratinocyte_count;
	
	//FOR keratinocyte AGENT OUTPUT SCATTER AGENTS 
    thrust::exclusive_scan(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_keratinocytes_new->_scan_input), thrust::device_pointer_cast(d_keratinocytes_new->_scan_input) + keratinocytes_pre_death_count, thrust::device_pointer_cast(d_keratinocytes_new->_position));
	//reset agent count
	int keratinocyte_after_birth_count;
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_keratinocytes_new->_position[keratinocytes_pre_death_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_keratinocytes_new->_scan_input[keratinocytes_pre_death_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		keratinocyte_after_birth_count = h_xmachine_memory_keratinocyte_default_count + scan_last_sum+1;
	else
		keratinocyte_after_birth_count = h_xmachine_memory_keratinocyte_default_count + scan_last_sum;
	//check buffer is not exceeded
	if (keratinocyte_after_birth_count > xmachine_memory_keratinocyte_MAX){
		printf("Error: Buffer size of keratinocyte agents in state default will be exceeded writing new agents in function cycle\n");
		exit(0);
	}
	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_keratinocyte_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_keratinocyte_Agents<<<gridSize, blockSize, 0, stream>>>(d_keratinocytes_default, d_keratinocytes_new, h_xmachine_memory_keratinocyte_default_count, keratinocytes_pre_death_count);
	gpuErrchkLaunch();
	//Copy count to device
	h_xmachine_memory_keratinocyte_default_count = keratinocyte_after_birth_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_default_count, &h_xmachine_memory_keratinocyte_default_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_keratinocyte_default_count+h_xmachine_memory_keratinocyte_count > xmachine_memory_keratinocyte_MAX){
		printf("Error: Buffer size of cycle agents in state default will be exceeded moving working agents to next state in function cycle\n");
		exit(0);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_keratinocyte_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_keratinocyte_Agents<<<gridSize, blockSize, 0, stream>>>(d_keratinocytes_default, d_keratinocytes, h_xmachine_memory_keratinocyte_default_count, h_xmachine_memory_keratinocyte_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_keratinocyte_default_count += h_xmachine_memory_keratinocyte_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_default_count, &h_xmachine_memory_keratinocyte_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int keratinocyte_differentiate_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input is spatially partitioned
	sm_size += (blockSize * sizeof(xmachine_message_location));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** keratinocyte_differentiate
 * Agent function prototype for differentiate function of keratinocyte agent
 */
void keratinocyte_differentiate(cudaStream_t &stream){

	int sm_size;
	int blockSize;
	int minGridSize;
	int gridSize;
	int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_keratinocyte_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_keratinocyte_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_keratinocyte_list* keratinocytes_default_temp = d_keratinocytes;
	d_keratinocytes = d_keratinocytes_default;
	d_keratinocytes_default = keratinocytes_default_temp;
	//set working count to current state count
	h_xmachine_memory_keratinocyte_count = h_xmachine_memory_keratinocyte_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_count, &h_xmachine_memory_keratinocyte_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_keratinocyte_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_default_count, &h_xmachine_memory_keratinocyte_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_differentiate, keratinocyte_differentiate_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = keratinocyte_differentiate_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	//continuous agent with discrete or partitioned message input uses texture caching
	size_t tex_xmachine_message_location_id_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_id_byte_offset, tex_xmachine_message_location_id, d_locations->id, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_id_offset = (int)tex_xmachine_message_location_id_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_id_offset, &h_tex_xmachine_message_location_id_offset, sizeof(int)));
	size_t tex_xmachine_message_location_type_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_type_byte_offset, tex_xmachine_message_location_type, d_locations->type, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_type_offset = (int)tex_xmachine_message_location_type_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_type_offset, &h_tex_xmachine_message_location_type_offset, sizeof(int)));
	size_t tex_xmachine_message_location_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_x_byte_offset, tex_xmachine_message_location_x, d_locations->x, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_x_offset = (int)tex_xmachine_message_location_x_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_x_offset, &h_tex_xmachine_message_location_x_offset, sizeof(int)));
	size_t tex_xmachine_message_location_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_y_byte_offset, tex_xmachine_message_location_y, d_locations->y, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_y_offset = (int)tex_xmachine_message_location_y_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_y_offset, &h_tex_xmachine_message_location_y_offset, sizeof(int)));
	size_t tex_xmachine_message_location_z_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_z_byte_offset, tex_xmachine_message_location_z, d_locations->z, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_z_offset = (int)tex_xmachine_message_location_z_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_z_offset, &h_tex_xmachine_message_location_z_offset, sizeof(int)));
	size_t tex_xmachine_message_location_dir_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_dir_byte_offset, tex_xmachine_message_location_dir, d_locations->dir, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_dir_offset = (int)tex_xmachine_message_location_dir_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_dir_offset, &h_tex_xmachine_message_location_dir_offset, sizeof(int)));
	size_t tex_xmachine_message_location_motility_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_motility_byte_offset, tex_xmachine_message_location_motility, d_locations->motility, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_motility_offset = (int)tex_xmachine_message_location_motility_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_motility_offset, &h_tex_xmachine_message_location_motility_offset, sizeof(int)));
	size_t tex_xmachine_message_location_range_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_range_byte_offset, tex_xmachine_message_location_range, d_locations->range, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_range_offset = (int)tex_xmachine_message_location_range_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_range_offset, &h_tex_xmachine_message_location_range_offset, sizeof(int)));
	size_t tex_xmachine_message_location_iteration_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_iteration_byte_offset, tex_xmachine_message_location_iteration, d_locations->iteration, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_iteration_offset = (int)tex_xmachine_message_location_iteration_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_iteration_offset, &h_tex_xmachine_message_location_iteration_offset, sizeof(int)));
	//bind pbm start and end indices to textures
	size_t tex_xmachine_message_location_pbm_start_byte_offset;
	size_t tex_xmachine_message_location_pbm_end_or_count_byte_offset;
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_pbm_start_byte_offset, tex_xmachine_message_location_pbm_start, d_location_partition_matrix->start, sizeof(int)*xmachine_message_location_grid_size));
	h_tex_xmachine_message_location_pbm_start_offset = (int)tex_xmachine_message_location_pbm_start_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_pbm_start_offset, &h_tex_xmachine_message_location_pbm_start_offset, sizeof(int)));
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_pbm_end_or_count_byte_offset, tex_xmachine_message_location_pbm_end_or_count, d_location_partition_matrix->end_or_count, sizeof(int)*xmachine_message_location_grid_size));
  h_tex_xmachine_message_location_pbm_end_or_count_offset = (int)tex_xmachine_message_location_pbm_end_or_count_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_pbm_end_or_count_offset, &h_tex_xmachine_message_location_pbm_end_or_count_offset, sizeof(int)));

	
	
	//MAIN XMACHINE FUNCTION CALL (differentiate)
	//Reallocate   : false
	//Input        : location
	//Output       : 
	//Agent Output : 
	GPUFLAME_differentiate<<<g, b, sm_size, stream>>>(d_keratinocytes, d_locations, d_location_partition_matrix);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	//continuous agent with discrete or partitioned message input uses texture caching
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_id));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_type));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_y));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_z));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_dir));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_motility));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_range));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_iteration));
	//unbind pbm indices
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_pbm_start));
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_pbm_end_or_count));
    
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_keratinocyte_default_count+h_xmachine_memory_keratinocyte_count > xmachine_memory_keratinocyte_MAX){
		printf("Error: Buffer size of differentiate agents in state default will be exceeded moving working agents to next state in function differentiate\n");
		exit(0);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_keratinocyte_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_keratinocyte_Agents<<<gridSize, blockSize, 0, stream>>>(d_keratinocytes_default, d_keratinocytes, h_xmachine_memory_keratinocyte_default_count, h_xmachine_memory_keratinocyte_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_keratinocyte_default_count += h_xmachine_memory_keratinocyte_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_default_count, &h_xmachine_memory_keratinocyte_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int keratinocyte_death_signal_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input is spatially partitioned
	sm_size += (blockSize * sizeof(xmachine_message_location));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** keratinocyte_death_signal
 * Agent function prototype for death_signal function of keratinocyte agent
 */
void keratinocyte_death_signal(cudaStream_t &stream){

	int sm_size;
	int blockSize;
	int minGridSize;
	int gridSize;
	int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_keratinocyte_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_keratinocyte_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_keratinocyte_list* keratinocytes_default_temp = d_keratinocytes;
	d_keratinocytes = d_keratinocytes_default;
	d_keratinocytes_default = keratinocytes_default_temp;
	//set working count to current state count
	h_xmachine_memory_keratinocyte_count = h_xmachine_memory_keratinocyte_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_count, &h_xmachine_memory_keratinocyte_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_keratinocyte_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_default_count, &h_xmachine_memory_keratinocyte_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_death_signal, keratinocyte_death_signal_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = keratinocyte_death_signal_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	//continuous agent with discrete or partitioned message input uses texture caching
	size_t tex_xmachine_message_location_id_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_id_byte_offset, tex_xmachine_message_location_id, d_locations->id, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_id_offset = (int)tex_xmachine_message_location_id_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_id_offset, &h_tex_xmachine_message_location_id_offset, sizeof(int)));
	size_t tex_xmachine_message_location_type_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_type_byte_offset, tex_xmachine_message_location_type, d_locations->type, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_type_offset = (int)tex_xmachine_message_location_type_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_type_offset, &h_tex_xmachine_message_location_type_offset, sizeof(int)));
	size_t tex_xmachine_message_location_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_x_byte_offset, tex_xmachine_message_location_x, d_locations->x, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_x_offset = (int)tex_xmachine_message_location_x_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_x_offset, &h_tex_xmachine_message_location_x_offset, sizeof(int)));
	size_t tex_xmachine_message_location_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_y_byte_offset, tex_xmachine_message_location_y, d_locations->y, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_y_offset = (int)tex_xmachine_message_location_y_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_y_offset, &h_tex_xmachine_message_location_y_offset, sizeof(int)));
	size_t tex_xmachine_message_location_z_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_z_byte_offset, tex_xmachine_message_location_z, d_locations->z, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_z_offset = (int)tex_xmachine_message_location_z_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_z_offset, &h_tex_xmachine_message_location_z_offset, sizeof(int)));
	size_t tex_xmachine_message_location_dir_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_dir_byte_offset, tex_xmachine_message_location_dir, d_locations->dir, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_dir_offset = (int)tex_xmachine_message_location_dir_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_dir_offset, &h_tex_xmachine_message_location_dir_offset, sizeof(int)));
	size_t tex_xmachine_message_location_motility_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_motility_byte_offset, tex_xmachine_message_location_motility, d_locations->motility, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_motility_offset = (int)tex_xmachine_message_location_motility_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_motility_offset, &h_tex_xmachine_message_location_motility_offset, sizeof(int)));
	size_t tex_xmachine_message_location_range_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_range_byte_offset, tex_xmachine_message_location_range, d_locations->range, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_range_offset = (int)tex_xmachine_message_location_range_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_range_offset, &h_tex_xmachine_message_location_range_offset, sizeof(int)));
	size_t tex_xmachine_message_location_iteration_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_iteration_byte_offset, tex_xmachine_message_location_iteration, d_locations->iteration, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_iteration_offset = (int)tex_xmachine_message_location_iteration_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_iteration_offset, &h_tex_xmachine_message_location_iteration_offset, sizeof(int)));
	//bind pbm start and end indices to textures
	size_t tex_xmachine_message_location_pbm_start_byte_offset;
	size_t tex_xmachine_message_location_pbm_end_or_count_byte_offset;
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_pbm_start_byte_offset, tex_xmachine_message_location_pbm_start, d_location_partition_matrix->start, sizeof(int)*xmachine_message_location_grid_size));
	h_tex_xmachine_message_location_pbm_start_offset = (int)tex_xmachine_message_location_pbm_start_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_pbm_start_offset, &h_tex_xmachine_message_location_pbm_start_offset, sizeof(int)));
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_pbm_end_or_count_byte_offset, tex_xmachine_message_location_pbm_end_or_count, d_location_partition_matrix->end_or_count, sizeof(int)*xmachine_message_location_grid_size));
  h_tex_xmachine_message_location_pbm_end_or_count_offset = (int)tex_xmachine_message_location_pbm_end_or_count_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_pbm_end_or_count_offset, &h_tex_xmachine_message_location_pbm_end_or_count_offset, sizeof(int)));

	
	
	//MAIN XMACHINE FUNCTION CALL (death_signal)
	//Reallocate   : false
	//Input        : location
	//Output       : 
	//Agent Output : 
	GPUFLAME_death_signal<<<g, b, sm_size, stream>>>(d_keratinocytes, d_locations, d_location_partition_matrix, d_rand48);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	//continuous agent with discrete or partitioned message input uses texture caching
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_id));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_type));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_y));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_z));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_dir));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_motility));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_range));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_iteration));
	//unbind pbm indices
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_pbm_start));
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_pbm_end_or_count));
    
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_keratinocyte_default_count+h_xmachine_memory_keratinocyte_count > xmachine_memory_keratinocyte_MAX){
		printf("Error: Buffer size of death_signal agents in state default will be exceeded moving working agents to next state in function death_signal\n");
		exit(0);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_keratinocyte_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_keratinocyte_Agents<<<gridSize, blockSize, 0, stream>>>(d_keratinocytes_default, d_keratinocytes, h_xmachine_memory_keratinocyte_default_count, h_xmachine_memory_keratinocyte_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_keratinocyte_default_count += h_xmachine_memory_keratinocyte_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_default_count, &h_xmachine_memory_keratinocyte_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int keratinocyte_migrate_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input is spatially partitioned
	sm_size += (blockSize * sizeof(xmachine_message_location));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** keratinocyte_migrate
 * Agent function prototype for migrate function of keratinocyte agent
 */
void keratinocyte_migrate(cudaStream_t &stream){

	int sm_size;
	int blockSize;
	int minGridSize;
	int gridSize;
	int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_keratinocyte_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_keratinocyte_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_keratinocyte_list* keratinocytes_default_temp = d_keratinocytes;
	d_keratinocytes = d_keratinocytes_default;
	d_keratinocytes_default = keratinocytes_default_temp;
	//set working count to current state count
	h_xmachine_memory_keratinocyte_count = h_xmachine_memory_keratinocyte_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_count, &h_xmachine_memory_keratinocyte_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_keratinocyte_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_default_count, &h_xmachine_memory_keratinocyte_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_migrate, keratinocyte_migrate_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = keratinocyte_migrate_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	//continuous agent with discrete or partitioned message input uses texture caching
	size_t tex_xmachine_message_location_id_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_id_byte_offset, tex_xmachine_message_location_id, d_locations->id, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_id_offset = (int)tex_xmachine_message_location_id_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_id_offset, &h_tex_xmachine_message_location_id_offset, sizeof(int)));
	size_t tex_xmachine_message_location_type_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_type_byte_offset, tex_xmachine_message_location_type, d_locations->type, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_type_offset = (int)tex_xmachine_message_location_type_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_type_offset, &h_tex_xmachine_message_location_type_offset, sizeof(int)));
	size_t tex_xmachine_message_location_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_x_byte_offset, tex_xmachine_message_location_x, d_locations->x, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_x_offset = (int)tex_xmachine_message_location_x_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_x_offset, &h_tex_xmachine_message_location_x_offset, sizeof(int)));
	size_t tex_xmachine_message_location_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_y_byte_offset, tex_xmachine_message_location_y, d_locations->y, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_y_offset = (int)tex_xmachine_message_location_y_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_y_offset, &h_tex_xmachine_message_location_y_offset, sizeof(int)));
	size_t tex_xmachine_message_location_z_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_z_byte_offset, tex_xmachine_message_location_z, d_locations->z, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_z_offset = (int)tex_xmachine_message_location_z_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_z_offset, &h_tex_xmachine_message_location_z_offset, sizeof(int)));
	size_t tex_xmachine_message_location_dir_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_dir_byte_offset, tex_xmachine_message_location_dir, d_locations->dir, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_dir_offset = (int)tex_xmachine_message_location_dir_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_dir_offset, &h_tex_xmachine_message_location_dir_offset, sizeof(int)));
	size_t tex_xmachine_message_location_motility_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_motility_byte_offset, tex_xmachine_message_location_motility, d_locations->motility, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_motility_offset = (int)tex_xmachine_message_location_motility_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_motility_offset, &h_tex_xmachine_message_location_motility_offset, sizeof(int)));
	size_t tex_xmachine_message_location_range_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_range_byte_offset, tex_xmachine_message_location_range, d_locations->range, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_range_offset = (int)tex_xmachine_message_location_range_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_range_offset, &h_tex_xmachine_message_location_range_offset, sizeof(int)));
	size_t tex_xmachine_message_location_iteration_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_iteration_byte_offset, tex_xmachine_message_location_iteration, d_locations->iteration, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_iteration_offset = (int)tex_xmachine_message_location_iteration_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_iteration_offset, &h_tex_xmachine_message_location_iteration_offset, sizeof(int)));
	//bind pbm start and end indices to textures
	size_t tex_xmachine_message_location_pbm_start_byte_offset;
	size_t tex_xmachine_message_location_pbm_end_or_count_byte_offset;
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_pbm_start_byte_offset, tex_xmachine_message_location_pbm_start, d_location_partition_matrix->start, sizeof(int)*xmachine_message_location_grid_size));
	h_tex_xmachine_message_location_pbm_start_offset = (int)tex_xmachine_message_location_pbm_start_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_pbm_start_offset, &h_tex_xmachine_message_location_pbm_start_offset, sizeof(int)));
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_pbm_end_or_count_byte_offset, tex_xmachine_message_location_pbm_end_or_count, d_location_partition_matrix->end_or_count, sizeof(int)*xmachine_message_location_grid_size));
  h_tex_xmachine_message_location_pbm_end_or_count_offset = (int)tex_xmachine_message_location_pbm_end_or_count_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_pbm_end_or_count_offset, &h_tex_xmachine_message_location_pbm_end_or_count_offset, sizeof(int)));

	
	
	//MAIN XMACHINE FUNCTION CALL (migrate)
	//Reallocate   : false
	//Input        : location
	//Output       : 
	//Agent Output : 
	GPUFLAME_migrate<<<g, b, sm_size, stream>>>(d_keratinocytes, d_locations, d_location_partition_matrix, d_rand48);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	//continuous agent with discrete or partitioned message input uses texture caching
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_id));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_type));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_y));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_z));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_dir));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_motility));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_range));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_iteration));
	//unbind pbm indices
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_pbm_start));
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_pbm_end_or_count));
    
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_keratinocyte_resolve_count+h_xmachine_memory_keratinocyte_count > xmachine_memory_keratinocyte_MAX){
		printf("Error: Buffer size of migrate agents in state resolve will be exceeded moving working agents to next state in function migrate\n");
		exit(0);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_keratinocyte_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_keratinocyte_Agents<<<gridSize, blockSize, 0, stream>>>(d_keratinocytes_resolve, d_keratinocytes, h_xmachine_memory_keratinocyte_resolve_count, h_xmachine_memory_keratinocyte_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_keratinocyte_resolve_count += h_xmachine_memory_keratinocyte_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_resolve_count, &h_xmachine_memory_keratinocyte_resolve_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int keratinocyte_force_resolution_output_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** keratinocyte_force_resolution_output
 * Agent function prototype for force_resolution_output function of keratinocyte agent
 */
void keratinocyte_force_resolution_output(cudaStream_t &stream){

	int sm_size;
	int blockSize;
	int minGridSize;
	int gridSize;
	int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_keratinocyte_resolve_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_keratinocyte_resolve_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_keratinocyte_list* keratinocytes_resolve_temp = d_keratinocytes;
	d_keratinocytes = d_keratinocytes_resolve;
	d_keratinocytes_resolve = keratinocytes_resolve_temp;
	//set working count to current state count
	h_xmachine_memory_keratinocyte_count = h_xmachine_memory_keratinocyte_resolve_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_count, &h_xmachine_memory_keratinocyte_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_keratinocyte_resolve_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_resolve_count, &h_xmachine_memory_keratinocyte_resolve_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_force_count + h_xmachine_memory_keratinocyte_count > xmachine_message_force_MAX){
		printf("Error: Buffer size of force message will be exceeded in function force_resolution_output\n");
		exit(0);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_force_resolution_output, keratinocyte_force_resolution_output_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = keratinocyte_force_resolution_output_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned and spatially partitioned message outputs
	h_message_force_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_force_output_type, &h_message_force_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (force_resolution_output)
	//Reallocate   : false
	//Input        : 
	//Output       : force
	//Agent Output : 
	GPUFLAME_force_resolution_output<<<g, b, sm_size, stream>>>(d_keratinocytes, d_forces);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_force_count += h_xmachine_memory_keratinocyte_count;	
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_force_count, &h_message_force_count, sizeof(int)));	
	
#ifdef FAST_ATOMIC_SORTING
  //USE ATOMICS TO BUILD PARTITION BOUNDARY
	//reset partition matrix
	gpuErrchk( cudaMemset( (void*) d_force_partition_matrix, 0, sizeof(xmachine_message_force_PBM)));
  //
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, hist_force_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	hist_force_messages<<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_force_local_bin_index, d_xmachine_message_force_unsorted_index, d_force_partition_matrix->end_or_count, d_forces, state_list_size);
	gpuErrchkLaunch();
	
	thrust::device_ptr<int> ptr_count = thrust::device_pointer_cast(d_force_partition_matrix->end_or_count);
	thrust::device_ptr<int> ptr_index = thrust::device_pointer_cast(d_force_partition_matrix->start);
	thrust::exclusive_scan(thrust::cuda::par.on(stream), ptr_count, ptr_count + xmachine_message_force_grid_size, ptr_index); // scan
	
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_force_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize; 	// Round up according to array size 
	reorder_force_messages <<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_force_local_bin_index, d_xmachine_message_force_unsorted_index, d_force_partition_matrix->start, d_forces, d_forces_swap, state_list_size);
	gpuErrchkLaunch();
#else
	//HASH, SORT, REORDER AND BUILD PMB FOR SPATIAL PARTITIONING MESSAGE OUTPUTS
	//Get message hash values for sorting
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, hash_force_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	hash_force_messages<<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_force_keys, d_xmachine_message_force_values, d_forces);
	gpuErrchkLaunch();
	//Sort
	thrust::sort_by_key(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_xmachine_message_force_keys),  thrust::device_pointer_cast(d_xmachine_message_force_keys) + h_message_force_count,  thrust::device_pointer_cast(d_xmachine_message_force_values));
	gpuErrchkLaunch();
	//reorder and build pcb
	gpuErrchk(cudaMemset(d_force_partition_matrix->start, 0xffffffff, xmachine_message_force_grid_size* sizeof(int)));
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_force_messages, reorder_messages_sm_size, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	int reorder_sm_size = reorder_messages_sm_size(blockSize);
	reorder_force_messages<<<gridSize, blockSize, reorder_sm_size, stream>>>(d_xmachine_message_force_keys, d_xmachine_message_force_values, d_force_partition_matrix, d_forces, d_forces_swap);
	gpuErrchkLaunch();
#endif
	//swap ordered list
	xmachine_message_force_list* d_forces_temp = d_forces;
	d_forces = d_forces_swap;
	d_forces_swap = d_forces_temp;
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_keratinocyte_resolve_count+h_xmachine_memory_keratinocyte_count > xmachine_memory_keratinocyte_MAX){
		printf("Error: Buffer size of force_resolution_output agents in state resolve will be exceeded moving working agents to next state in function force_resolution_output\n");
		exit(0);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_keratinocyte_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_keratinocyte_Agents<<<gridSize, blockSize, 0, stream>>>(d_keratinocytes_resolve, d_keratinocytes, h_xmachine_memory_keratinocyte_resolve_count, h_xmachine_memory_keratinocyte_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_keratinocyte_resolve_count += h_xmachine_memory_keratinocyte_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_resolve_count, &h_xmachine_memory_keratinocyte_resolve_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int keratinocyte_resolve_forces_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input is spatially partitioned
	sm_size += (blockSize * sizeof(xmachine_message_force));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** keratinocyte_resolve_forces
 * Agent function prototype for resolve_forces function of keratinocyte agent
 */
void keratinocyte_resolve_forces(cudaStream_t &stream){

	int sm_size;
	int blockSize;
	int minGridSize;
	int gridSize;
	int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_keratinocyte_resolve_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_keratinocyte_resolve_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_keratinocyte_list* keratinocytes_resolve_temp = d_keratinocytes;
	d_keratinocytes = d_keratinocytes_resolve;
	d_keratinocytes_resolve = keratinocytes_resolve_temp;
	//set working count to current state count
	h_xmachine_memory_keratinocyte_count = h_xmachine_memory_keratinocyte_resolve_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_count, &h_xmachine_memory_keratinocyte_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_keratinocyte_resolve_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_resolve_count, &h_xmachine_memory_keratinocyte_resolve_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_resolve_forces, keratinocyte_resolve_forces_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = keratinocyte_resolve_forces_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	//continuous agent with discrete or partitioned message input uses texture caching
	size_t tex_xmachine_message_force_type_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_force_type_byte_offset, tex_xmachine_message_force_type, d_forces->type, sizeof(int)*xmachine_message_force_MAX));
	h_tex_xmachine_message_force_type_offset = (int)tex_xmachine_message_force_type_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_force_type_offset, &h_tex_xmachine_message_force_type_offset, sizeof(int)));
	size_t tex_xmachine_message_force_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_force_x_byte_offset, tex_xmachine_message_force_x, d_forces->x, sizeof(int)*xmachine_message_force_MAX));
	h_tex_xmachine_message_force_x_offset = (int)tex_xmachine_message_force_x_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_force_x_offset, &h_tex_xmachine_message_force_x_offset, sizeof(int)));
	size_t tex_xmachine_message_force_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_force_y_byte_offset, tex_xmachine_message_force_y, d_forces->y, sizeof(int)*xmachine_message_force_MAX));
	h_tex_xmachine_message_force_y_offset = (int)tex_xmachine_message_force_y_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_force_y_offset, &h_tex_xmachine_message_force_y_offset, sizeof(int)));
	size_t tex_xmachine_message_force_z_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_force_z_byte_offset, tex_xmachine_message_force_z, d_forces->z, sizeof(int)*xmachine_message_force_MAX));
	h_tex_xmachine_message_force_z_offset = (int)tex_xmachine_message_force_z_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_force_z_offset, &h_tex_xmachine_message_force_z_offset, sizeof(int)));
	size_t tex_xmachine_message_force_id_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_force_id_byte_offset, tex_xmachine_message_force_id, d_forces->id, sizeof(int)*xmachine_message_force_MAX));
	h_tex_xmachine_message_force_id_offset = (int)tex_xmachine_message_force_id_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_force_id_offset, &h_tex_xmachine_message_force_id_offset, sizeof(int)));
	//bind pbm start and end indices to textures
	size_t tex_xmachine_message_force_pbm_start_byte_offset;
	size_t tex_xmachine_message_force_pbm_end_or_count_byte_offset;
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_force_pbm_start_byte_offset, tex_xmachine_message_force_pbm_start, d_force_partition_matrix->start, sizeof(int)*xmachine_message_force_grid_size));
	h_tex_xmachine_message_force_pbm_start_offset = (int)tex_xmachine_message_force_pbm_start_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_force_pbm_start_offset, &h_tex_xmachine_message_force_pbm_start_offset, sizeof(int)));
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_force_pbm_end_or_count_byte_offset, tex_xmachine_message_force_pbm_end_or_count, d_force_partition_matrix->end_or_count, sizeof(int)*xmachine_message_force_grid_size));
  h_tex_xmachine_message_force_pbm_end_or_count_offset = (int)tex_xmachine_message_force_pbm_end_or_count_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_force_pbm_end_or_count_offset, &h_tex_xmachine_message_force_pbm_end_or_count_offset, sizeof(int)));

	
	
	//MAIN XMACHINE FUNCTION CALL (resolve_forces)
	//Reallocate   : false
	//Input        : force
	//Output       : 
	//Agent Output : 
	GPUFLAME_resolve_forces<<<g, b, sm_size, stream>>>(d_keratinocytes, d_forces, d_force_partition_matrix);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	//continuous agent with discrete or partitioned message input uses texture caching
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_force_type));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_force_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_force_y));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_force_z));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_force_id));
	//unbind pbm indices
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_force_pbm_start));
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_force_pbm_end_or_count));
    
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_keratinocyte_resolve_count+h_xmachine_memory_keratinocyte_count > xmachine_memory_keratinocyte_MAX){
		printf("Error: Buffer size of resolve_forces agents in state resolve will be exceeded moving working agents to next state in function resolve_forces\n");
		exit(0);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_keratinocyte_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_keratinocyte_Agents<<<gridSize, blockSize, 0, stream>>>(d_keratinocytes_resolve, d_keratinocytes, h_xmachine_memory_keratinocyte_resolve_count, h_xmachine_memory_keratinocyte_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_keratinocyte_resolve_count += h_xmachine_memory_keratinocyte_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_resolve_count, &h_xmachine_memory_keratinocyte_resolve_count, sizeof(int)));	
	
	
}


 
extern "C" void reset_keratinocyte_default_count()
{
    h_xmachine_memory_keratinocyte_default_count = 0;
}
 
extern "C" void reset_keratinocyte_resolve_count()
{
    h_xmachine_memory_keratinocyte_resolve_count = 0;
}
