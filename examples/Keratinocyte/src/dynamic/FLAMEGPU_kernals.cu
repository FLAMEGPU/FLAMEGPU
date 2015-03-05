

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

#ifndef _FLAMEGPU_KERNELS_H_
#define _FLAMEGPU_KERNELS_H_

#include "header.h"


/* Agent count constants */

__constant__ int d_xmachine_memory_keratinocyte_count;

/* Agent state count constants */

__constant__ int d_xmachine_memory_keratinocyte_default_count;

__constant__ int d_xmachine_memory_keratinocyte_resolve_count;


/* Message constants */

/* location Message variables */
/* Non partitioned and spatial partitioned message variables  */
__constant__ int d_message_location_count;         /**< message list counter*/
__constant__ int d_message_location_output_type;   /**< message output type (single or optional)*/
//Spatial Partitioning Variables
__constant__ float3 d_message_location_min_bounds;           /**< min bounds (x,y,z) of partitioning environment */
__constant__ float3 d_message_location_max_bounds;           /**< max bounds (x,y,z) of partitioning environment */
__constant__ int3 d_message_location_partitionDim;           /**< partition dimensions (x,y,z) of partitioning environment */
__constant__ float d_message_location_radius;                 /**< partition radius (used to determin the size of the partitions) */

/* force Message variables */
/* Non partitioned and spatial partitioned message variables  */
__constant__ int d_message_force_count;         /**< message list counter*/
__constant__ int d_message_force_output_type;   /**< message output type (single or optional)*/
//Spatial Partitioning Variables
__constant__ float3 d_message_force_min_bounds;           /**< min bounds (x,y,z) of partitioning environment */
__constant__ float3 d_message_force_max_bounds;           /**< max bounds (x,y,z) of partitioning environment */
__constant__ int3 d_message_force_partitionDim;           /**< partition dimensions (x,y,z) of partitioning environment */
__constant__ float d_message_force_radius;                 /**< partition radius (used to determin the size of the partitions) */

	
    
//include each function file

#include "functions.c"
    
/* Texture bindings */
/* location Message Bindings */texture<int, 1, cudaReadModeElementType> tex_xmachine_message_location_id;
__constant__ int d_tex_xmachine_message_location_id_offset;texture<int, 1, cudaReadModeElementType> tex_xmachine_message_location_type;
__constant__ int d_tex_xmachine_message_location_type_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_location_x;
__constant__ int d_tex_xmachine_message_location_x_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_location_y;
__constant__ int d_tex_xmachine_message_location_y_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_location_z;
__constant__ int d_tex_xmachine_message_location_z_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_location_dir;
__constant__ int d_tex_xmachine_message_location_dir_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_location_motility;
__constant__ int d_tex_xmachine_message_location_motility_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_location_range;
__constant__ int d_tex_xmachine_message_location_range_offset;texture<int, 1, cudaReadModeElementType> tex_xmachine_message_location_iteration;
__constant__ int d_tex_xmachine_message_location_iteration_offset;
texture<int, 1, cudaReadModeElementType> tex_xmachine_message_location_pbm_start;
__constant__ int d_tex_xmachine_message_location_pbm_start_offset;
texture<int, 1, cudaReadModeElementType> tex_xmachine_message_location_pbm_end_or_count;
__constant__ int d_tex_xmachine_message_location_pbm_end_or_count_offset;


/* force Message Bindings */texture<int, 1, cudaReadModeElementType> tex_xmachine_message_force_type;
__constant__ int d_tex_xmachine_message_force_type_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_force_x;
__constant__ int d_tex_xmachine_message_force_x_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_force_y;
__constant__ int d_tex_xmachine_message_force_y_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_force_z;
__constant__ int d_tex_xmachine_message_force_z_offset;texture<int, 1, cudaReadModeElementType> tex_xmachine_message_force_id;
__constant__ int d_tex_xmachine_message_force_id_offset;
texture<int, 1, cudaReadModeElementType> tex_xmachine_message_force_pbm_start;
__constant__ int d_tex_xmachine_message_force_pbm_start_offset;
texture<int, 1, cudaReadModeElementType> tex_xmachine_message_force_pbm_end_or_count;
__constant__ int d_tex_xmachine_message_force_pbm_end_or_count_offset;


    
#define WRAP(x,m) (((x)<m)?(x):(x%m)) /**< Simple wrap */
#define sWRAP(x,m) (((x)<m)?(((x)<0)?(m+(x)):(x)):(m-(x))) /**<signed integer wrap (no modulus) for negatives where 2m > |x| > m */

//PADDING WILL ONLY AVOID SM CONFLICTS FOR 32BIT
//SM_OFFSET REQUIRED AS FERMI STARTS INDEXING MEMORY FROM LOCATION 0 (i.e. NULL)??
__constant__ int d_SM_START;
__constant__ int d_PADDING;

//SM addressing macro to avoid conflicts (32 bit only)
#define SHARE_INDEX(i, s) (((s + d_PADDING)* i)+d_SM_START) /**<offset struct size by padding to avoid bank conflicts */

//if doubel support is needed then define the following function which requires sm_13 or later
#ifdef _DOUBLE_SUPPORT_REQUIRED_
__inline__ __device__ double tex1DfetchDouble(texture<int2, 1, cudaReadModeElementType> tex, int i)
{
	int2 v = tex1Dfetch(tex, i);
  //IF YOU HAVE AN ERROR HERE THEN YOU ARE USING DOUBLE VALUES IN AGENT MEMORY AND NOT COMPILING FOR DOUBLE SUPPORTED HARDWARE
  //To compile for double supported hardware change the CUDA Build rule property "Use sm_13 Architecture (double support)" on the CUDA-Specific Propert Page of the CUDA Build Rule for simulation.cu
	return __hiloint2double(v.y, v.x);
}
#endif

/* Helper functions */
/** next_cell
 * Function used for finding the next cell when using spatial partitioning
 * Upddates the relative cell variable which can have value of -1, 0 or +1
 * @param relative_cell pointer to the relative cell position
 * @return boolean if there is a next cell. True unless relative_Cell value was 1,1,1
 */
__device__ int next_cell3D(int3* relative_cell)
{
	if (relative_cell->x < 1)
	{
		relative_cell->x++;
		return true;
	}
	relative_cell->x = -1;

	if (relative_cell->y < 1)
	{
		relative_cell->y++;
		return true;
	}
	relative_cell->y = -1;
	
	if (relative_cell->z < 1)
	{
		relative_cell->z++;
		return true;
	}
	relative_cell->z = -1;
	
	return false;
}

/** next_cell2D
 * Function used for finding the next cell when using spatial partitioning. Z component is ignored
 * Upddates the relative cell variable which can have value of -1, 0 or +1
 * @param relative_cell pointer to the relative cell position
 * @return boolean if there is a next cell. True unless relative_Cell value was 1,1
 */
__device__ int next_cell2D(int3* relative_cell)
{
	if (relative_cell->x < 1)
	{
		relative_cell->x++;
		return true;
	}
	relative_cell->x = -1;

	if (relative_cell->y < 1)
	{
		relative_cell->y++;
		return true;
	}
	relative_cell->y = -1;
	
	return false;
}


/** output_location_function_filter
 *	Global condition function. Flags the scan input state to true if the condition is met
 * @param currentState xmachine_memory_keratinocyte_list representing agent i the current state
 */
 __global__ void output_location_function_filter(xmachine_memory_keratinocyte_list* currentState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_keratinocyte_count){
	
		//apply the filter
		if (currentState->movement[index]<0.25)
		{	currentState->_scan_input[index] = 1;
		}
		else
		{
			currentState->_scan_input[index] = 0;
		}
	
	}
 }

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created keratinocyte agent functions */

/** reset_keratinocyte_scan_input
 * keratinocyte agent reset scan input function
 * @param agents The xmachine_memory_keratinocyte_list agent list
 */
__global__ void reset_keratinocyte_scan_input(xmachine_memory_keratinocyte_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_keratinocyte_Agents
 * keratinocyte scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_keratinocyte_list agent list destination
 * @param agents_src xmachine_memory_keratinocyte_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_keratinocyte_Agents(xmachine_memory_keratinocyte_list* agents_dst, xmachine_memory_keratinocyte_list* agents_src, int dst_agent_count, int number_to_scatter){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = agents_src->_scan_input[index];

	//if optional message is to be written. 
	//must check agent is within number to scatter as unused threads may have scan input = 1
	if ((_scan_input == 1)&&(index < number_to_scatter)){
		int output_index = agents_src->_position[index] + dst_agent_count;

		//AoS - xmachine_message_location Un-Coalesced scattered memory write     
        agents_dst->_position[output_index] = output_index;        
		agents_dst->id[output_index] = agents_src->id[index];        
		agents_dst->type[output_index] = agents_src->type[index];        
		agents_dst->x[output_index] = agents_src->x[index];        
		agents_dst->y[output_index] = agents_src->y[index];        
		agents_dst->z[output_index] = agents_src->z[index];        
		agents_dst->force_x[output_index] = agents_src->force_x[index];        
		agents_dst->force_y[output_index] = agents_src->force_y[index];        
		agents_dst->force_z[output_index] = agents_src->force_z[index];        
		agents_dst->num_xy_bonds[output_index] = agents_src->num_xy_bonds[index];        
		agents_dst->num_z_bonds[output_index] = agents_src->num_z_bonds[index];        
		agents_dst->num_stem_bonds[output_index] = agents_src->num_stem_bonds[index];        
		agents_dst->cycle[output_index] = agents_src->cycle[index];        
		agents_dst->diff_noise_factor[output_index] = agents_src->diff_noise_factor[index];        
		agents_dst->dead_ticks[output_index] = agents_src->dead_ticks[index];        
		agents_dst->contact_inhibited_ticks[output_index] = agents_src->contact_inhibited_ticks[index];        
		agents_dst->motility[output_index] = agents_src->motility[index];        
		agents_dst->dir[output_index] = agents_src->dir[index];        
		agents_dst->movement[output_index] = agents_src->movement[index];
	}
}

/** append_keratinocyte_Agents
 * keratinocyte scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_keratinocyte_list agent list destination
 * @param agents_src xmachine_memory_keratinocyte_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_keratinocyte_Agents(xmachine_memory_keratinocyte_list* agents_dst, xmachine_memory_keratinocyte_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->id[output_index] = agents_src->id[index];
	    agents_dst->type[output_index] = agents_src->type[index];
	    agents_dst->x[output_index] = agents_src->x[index];
	    agents_dst->y[output_index] = agents_src->y[index];
	    agents_dst->z[output_index] = agents_src->z[index];
	    agents_dst->force_x[output_index] = agents_src->force_x[index];
	    agents_dst->force_y[output_index] = agents_src->force_y[index];
	    agents_dst->force_z[output_index] = agents_src->force_z[index];
	    agents_dst->num_xy_bonds[output_index] = agents_src->num_xy_bonds[index];
	    agents_dst->num_z_bonds[output_index] = agents_src->num_z_bonds[index];
	    agents_dst->num_stem_bonds[output_index] = agents_src->num_stem_bonds[index];
	    agents_dst->cycle[output_index] = agents_src->cycle[index];
	    agents_dst->diff_noise_factor[output_index] = agents_src->diff_noise_factor[index];
	    agents_dst->dead_ticks[output_index] = agents_src->dead_ticks[index];
	    agents_dst->contact_inhibited_ticks[output_index] = agents_src->contact_inhibited_ticks[index];
	    agents_dst->motility[output_index] = agents_src->motility[index];
	    agents_dst->dir[output_index] = agents_src->dir[index];
	    agents_dst->movement[output_index] = agents_src->movement[index];
    }
}

/** add_keratinocyte_agent
 * Continuous keratinocyte agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_keratinocyte_list to add agents to 
 * @param id agent variable of type int
 * @param type agent variable of type int
 * @param x agent variable of type float
 * @param y agent variable of type float
 * @param z agent variable of type float
 * @param force_x agent variable of type float
 * @param force_y agent variable of type float
 * @param force_z agent variable of type float
 * @param num_xy_bonds agent variable of type int
 * @param num_z_bonds agent variable of type int
 * @param num_stem_bonds agent variable of type int
 * @param cycle agent variable of type int
 * @param diff_noise_factor agent variable of type float
 * @param dead_ticks agent variable of type int
 * @param contact_inhibited_ticks agent variable of type int
 * @param motility agent variable of type float
 * @param dir agent variable of type float
 * @param movement agent variable of type float
 */
template <int AGENT_TYPE>
__device__ void add_keratinocyte_agent(xmachine_memory_keratinocyte_list* agents, int id, int type, float x, float y, float z, float force_x, float force_y, float force_z, int num_xy_bonds, int num_z_bonds, int num_stem_bonds, int cycle, float diff_noise_factor, int dead_ticks, int contact_inhibited_ticks, float motility, float dir, float movement){
	
	int index;
    
    //calculate the agents index in global agent list (depends on agent type)
	if (AGENT_TYPE == DISCRETE_2D){
		int width = (blockDim.x* gridDim.x);
		int2 global_position;
		global_position.x = (blockIdx.x*blockDim.x) + threadIdx.x;
		global_position.y = (blockIdx.y*blockDim.y) + threadIdx.y;
		index = global_position.x + (global_position.y* width);
	}else//AGENT_TYPE == CONTINOUS
		index = threadIdx.x + blockIdx.x*blockDim.x;

	//for prefix sum
	agents->_position[index] = 0;
	agents->_scan_input[index] = 1;

	//write data to new buffer
	agents->id[index] = id;
	agents->type[index] = type;
	agents->x[index] = x;
	agents->y[index] = y;
	agents->z[index] = z;
	agents->force_x[index] = force_x;
	agents->force_y[index] = force_y;
	agents->force_z[index] = force_z;
	agents->num_xy_bonds[index] = num_xy_bonds;
	agents->num_z_bonds[index] = num_z_bonds;
	agents->num_stem_bonds[index] = num_stem_bonds;
	agents->cycle[index] = cycle;
	agents->diff_noise_factor[index] = diff_noise_factor;
	agents->dead_ticks[index] = dead_ticks;
	agents->contact_inhibited_ticks[index] = contact_inhibited_ticks;
	agents->motility[index] = motility;
	agents->dir[index] = dir;
	agents->movement[index] = movement;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_keratinocyte_agent(xmachine_memory_keratinocyte_list* agents, int id, int type, float x, float y, float z, float force_x, float force_y, float force_z, int num_xy_bonds, int num_z_bonds, int num_stem_bonds, int cycle, float diff_noise_factor, int dead_ticks, int contact_inhibited_ticks, float motility, float dir, float movement){
    add_keratinocyte_agent<DISCRETE_2D>(agents, id, type, x, y, z, force_x, force_y, force_z, num_xy_bonds, num_z_bonds, num_stem_bonds, cycle, diff_noise_factor, dead_ticks, contact_inhibited_ticks, motility, dir, movement);
}

/** reorder_keratinocyte_agents
 * Continuous keratinocyte agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_keratinocyte_agents(unsigned int* values, xmachine_memory_keratinocyte_list* unordered_agents, xmachine_memory_keratinocyte_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->id[index] = unordered_agents->id[old_pos];
	ordered_agents->type[index] = unordered_agents->type[old_pos];
	ordered_agents->x[index] = unordered_agents->x[old_pos];
	ordered_agents->y[index] = unordered_agents->y[old_pos];
	ordered_agents->z[index] = unordered_agents->z[old_pos];
	ordered_agents->force_x[index] = unordered_agents->force_x[old_pos];
	ordered_agents->force_y[index] = unordered_agents->force_y[old_pos];
	ordered_agents->force_z[index] = unordered_agents->force_z[old_pos];
	ordered_agents->num_xy_bonds[index] = unordered_agents->num_xy_bonds[old_pos];
	ordered_agents->num_z_bonds[index] = unordered_agents->num_z_bonds[old_pos];
	ordered_agents->num_stem_bonds[index] = unordered_agents->num_stem_bonds[old_pos];
	ordered_agents->cycle[index] = unordered_agents->cycle[old_pos];
	ordered_agents->diff_noise_factor[index] = unordered_agents->diff_noise_factor[old_pos];
	ordered_agents->dead_ticks[index] = unordered_agents->dead_ticks[old_pos];
	ordered_agents->contact_inhibited_ticks[index] = unordered_agents->contact_inhibited_ticks[old_pos];
	ordered_agents->motility[index] = unordered_agents->motility[old_pos];
	ordered_agents->dir[index] = unordered_agents->dir[old_pos];
	ordered_agents->movement[index] = unordered_agents->movement[old_pos];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created location message functions */


/** add_location_message
 * Add non partitioned or spatially partitioned location message
 * @param messages xmachine_message_location_list message list to add too
 * @param id agent variable of type int
 * @param type agent variable of type int
 * @param x agent variable of type float
 * @param y agent variable of type float
 * @param z agent variable of type float
 * @param dir agent variable of type float
 * @param motility agent variable of type float
 * @param range agent variable of type float
 * @param iteration agent variable of type int
 */
__device__ void add_location_message(xmachine_message_location_list* messages, int id, int type, float x, float y, float z, float dir, float motility, float range, int iteration){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_location_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_location_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_location_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_location Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->id[index] = id;
	messages->type[index] = type;
	messages->x[index] = x;
	messages->y[index] = y;
	messages->z[index] = z;
	messages->dir[index] = dir;
	messages->motility[index] = motility;
	messages->range[index] = range;
	messages->iteration[index] = iteration;

}

/**
 * Scatter non partitioned or spatially partitioned location message (for optional messages)
 * @param messages scatter_optional_location_messages Sparse xmachine_message_location_list message list
 * @param message_swap temp xmachine_message_location_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_location_messages(xmachine_message_location_list* messages, xmachine_message_location_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_location_count;

		//AoS - xmachine_message_location Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->id[output_index] = messages_swap->id[index];
		messages->type[output_index] = messages_swap->type[index];
		messages->x[output_index] = messages_swap->x[index];
		messages->y[output_index] = messages_swap->y[index];
		messages->z[output_index] = messages_swap->z[index];
		messages->dir[output_index] = messages_swap->dir[index];
		messages->motility[output_index] = messages_swap->motility[index];
		messages->range[output_index] = messages_swap->range[index];
		messages->iteration[output_index] = messages_swap->iteration[index];				
	}
}

/** reset_location_swaps
 * Reset non partitioned or spatially partitioned location message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_location_swaps(xmachine_message_location_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

/** message_location_grid_position
 * Calculates the grid cell position given an float3 vector
 * @param position float3 vector representing a position
 */
__device__ int3 message_location_grid_position(float3 position)
{
    int3 gridPos;
    gridPos.x = floor((position.x - d_message_location_min_bounds.x) * (float)d_message_location_partitionDim.x / (d_message_location_max_bounds.x - d_message_location_min_bounds.x));
    gridPos.y = floor((position.y - d_message_location_min_bounds.y) * (float)d_message_location_partitionDim.y / (d_message_location_max_bounds.y - d_message_location_min_bounds.y));
    gridPos.z = floor((position.z - d_message_location_min_bounds.z) * (float)d_message_location_partitionDim.z / (d_message_location_max_bounds.z - d_message_location_min_bounds.z));

	//do wrapping or bounding
	

    return gridPos;
}

/** message_location_hash
 * Given the grid position in partition space this function calculates a hash value
 * @param gridPos The position in partition space
 */
__device__ unsigned int message_location_hash(int3 gridPos)
{
	//cheap bounding without mod (within range +- partition dimension)
	gridPos.x = (gridPos.x<0)? d_message_location_partitionDim.x-1: gridPos.x; 
	gridPos.x = (gridPos.x>=d_message_location_partitionDim.x)? 0 : gridPos.x; 
	gridPos.y = (gridPos.y<0)? d_message_location_partitionDim.y-1 : gridPos.y; 
	gridPos.y = (gridPos.y>=d_message_location_partitionDim.y)? 0 : gridPos.y; 
	gridPos.z = (gridPos.z<0)? d_message_location_partitionDim.z-1: gridPos.z; 
	gridPos.z = (gridPos.z>=d_message_location_partitionDim.z)? 0 : gridPos.z; 

	//unique id
	return ((gridPos.z * d_message_location_partitionDim.y) * d_message_location_partitionDim.x) + (gridPos.y * d_message_location_partitionDim.x) + gridPos.x;
}

#ifdef FAST_ATOMIC_SORTING
	/** hist_location_messages
		 * Kernal function for performing a histogram (count) on each partition bin and saving the hash and index of a message within that bin
		 * @param local_bin_index output index of the message within the calculated bin
		 * @param unsorted_index output bin index (hash) value
		 * @param messages the message list used to generate the hash value outputs
		 * @param agent_count the current number of agents outputting messages
		 */
	__global__ void hist_location_messages(uint* local_bin_index, uint* unsorted_index, int* global_bin_count, xmachine_message_location_list* messages, int agent_count)
	{
		unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		if (index >= agent_count)
			return;

		float3 position = make_float3(messages->x[index], messages->y[index], messages->z[index]);
		int3 grid_position = message_location_grid_position(position);
		unsigned int hash = message_location_hash(grid_position);
		unsigned int bin_idx = atomicInc((unsigned int*) &global_bin_count[hash], 0xFFFFFFFF);
		local_bin_index[index] = bin_idx;
		unsorted_index[index] = hash;
	}
	
	/** reorder_location_messages
	 * Reorders the messages accoring to the partition boundary matrix start indices of each bin
	 * @param local_bin_index index of the message within the desired bin
	 * @param unsorted_index bin index (hash) value
	 * @param pbm_start_index the start indices of the partition boundary matrix
	 * @param unordered_messages the original unordered message data
	 * @param ordered_messages buffer used to scatter messages into the correct order
	  @param agent_count the current number of agents outputting messages
	 */
	 __global__ void reorder_location_messages(uint* local_bin_index, uint* unsorted_index, int* pbm_start_index, xmachine_message_location_list* unordered_messages, xmachine_message_location_list* ordered_messages, int agent_count)
	{
		int index = (blockIdx.x *blockDim.x) + threadIdx.x;

		if (index >= agent_count)
			return;

		uint i = unsorted_index[index];
		unsigned int sorted_index = local_bin_index[index] + pbm_start_index[i];

		//finally reorder agent data
		ordered_messages->id[sorted_index] = unordered_messages->id[index];
		ordered_messages->type[sorted_index] = unordered_messages->type[index];
		ordered_messages->x[sorted_index] = unordered_messages->x[index];
		ordered_messages->y[sorted_index] = unordered_messages->y[index];
		ordered_messages->z[sorted_index] = unordered_messages->z[index];
		ordered_messages->dir[sorted_index] = unordered_messages->dir[index];
		ordered_messages->motility[sorted_index] = unordered_messages->motility[index];
		ordered_messages->range[sorted_index] = unordered_messages->range[index];
		ordered_messages->iteration[sorted_index] = unordered_messages->iteration[index];
	}
	 
#else

	/** hash_location_messages
	 * Kernal function for calculating a hash value for each messahe depending on its position
	 * @param keys output for the hash key
	 * @param values output for the index value
	 * @param messages the message list used to generate the hash value outputs
	 */
	__global__ void hash_location_messages(uint* keys, uint* values, xmachine_message_location_list* messages)
	{
		unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		float3 position = make_float3(messages->x[index], messages->y[index], messages->z[index]);
		int3 grid_position = message_location_grid_position(position);
		unsigned int hash = message_location_hash(grid_position);

		keys[index] = hash;
		values[index] = index;
	}

	/** reorder_location_messages
	 * Reorders the messages accoring to the ordered sort identifiers and builds a Partition Boundary Matrix by looking at the previosu threads sort id.
	 * @param keys the sorted hash keys
	 * @param values the sorted index values
	 * @param matrix the PBM
	 * @param unordered_messages the original unordered message data
	 * @param ordered_messages buffer used to scatter messages into the correct order
	 */
	__global__ void reorder_location_messages(uint* keys, uint* values, xmachine_message_location_PBM* matrix, xmachine_message_location_list* unordered_messages, xmachine_message_location_list* ordered_messages)
	{
		extern __shared__ int sm_data [];

		int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		//load threads sort key into sm
		uint key = keys[index];
		uint old_pos = values[index];

		sm_data[threadIdx.x] = key;
		__syncthreads();
	
		unsigned int prev_key;

		//if first thread then no prev sm value so get prev from global memory 
		if (threadIdx.x == 0)
		{
			//first thread has no prev value so ignore
			if (index != 0)
				prev_key = keys[index-1];
		}
		//get previous ident from sm
		else	
		{
			prev_key = sm_data[threadIdx.x-1];
		}

		//TODO: Check key is not out of bounds

		//set partition boundaries
		if (index < d_message_location_count)
		{
			//if first thread then set first partition cell start
			if (index == 0)
			{
				matrix->start[key] = index;
			}

			//if edge of a boundr update start and end of partition
			else if (prev_key != key)
			{
				//set start for key
				matrix->start[key] = index;

				//set end for key -1
				matrix->end_or_count[prev_key] = index;
			}

			//if last thread then set final partition cell end
			if (index == d_message_location_count-1)
			{
				matrix->end_or_count[key] = index+1;
			}
		}
	
		//finally reorder agent data
		ordered_messages->id[index] = unordered_messages->id[old_pos];
		ordered_messages->type[index] = unordered_messages->type[old_pos];
		ordered_messages->x[index] = unordered_messages->x[old_pos];
		ordered_messages->y[index] = unordered_messages->y[old_pos];
		ordered_messages->z[index] = unordered_messages->z[old_pos];
		ordered_messages->dir[index] = unordered_messages->dir[old_pos];
		ordered_messages->motility[index] = unordered_messages->motility[old_pos];
		ordered_messages->range[index] = unordered_messages->range[old_pos];
		ordered_messages->iteration[index] = unordered_messages->iteration[old_pos];
	}

#endif

/** load_next_location_message
 * Used to load the next message data to shared memory
 * Idea is check the current cell index to see if we can simpley get a message from the current cell
 * If we are at the end of the current cell then loop till we find the next cell with messages (this way we ignore cells with no messages)
 * @param messages the message list
 * @param partition_matrix the PBM
 * @param relative_cell the relative partition cell position from the agent position
 * @param cell_index_max the maximum index of the currnt partition cell
 * @param agent_grid_cell the agents partition cell position
 * @param cell_index the current cell index in agent_grid_cell+relative_cell
 * @return true if a messag has been loaded into sm false otherwise
 */
__device__ int load_next_location_message(xmachine_message_location_list* messages, xmachine_message_location_PBM* partition_matrix, int3 relative_cell, int cell_index_max, int3 agent_grid_cell, int cell_index)
{
	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];

	int move_cell = true;
	cell_index ++;

	//see if we need to move to a new partition cell
	if(cell_index < cell_index_max)
		move_cell = false;

	while(move_cell)
	{
		//get the next relative grid position 
        if (next_cell3D(&relative_cell))
		{
			//calculate the next cells grid position and hash
			int3 next_cell_position = agent_grid_cell + relative_cell;
			int next_cell_hash = message_location_hash(next_cell_position);
			//use the hash to calculate the start index
			int cell_index_min = tex1Dfetch(tex_xmachine_message_location_pbm_start, next_cell_hash + d_tex_xmachine_message_location_pbm_start_offset);
			cell_index_max = tex1Dfetch(tex_xmachine_message_location_pbm_end_or_count, next_cell_hash + d_tex_xmachine_message_location_pbm_end_or_count_offset);
			//check for messages in the cell (cell index max is the count for atomic sorting)
#ifdef FAST_ATOMIC_SORTING
			if (cell_index_max > 0)
			{
				//when using fast atomics value represents bin count not last index!
				cell_index_max += cell_index_min; //when using fast atomics value represents bin count not last index!
#else
			if (cell_index_min != 0xffffffff)
			{
#endif
				//start from the cell index min
				cell_index = cell_index_min;
				//exit the loop as we have found a valid cell with message data
				move_cell = false;
			}
		}
		else
		{
			//we have exhausted all the neightbouring cells so there are no more messages
			return false;
		}
	}
	
	//get message data using texture fetch
	xmachine_message_location temp_message;
	temp_message._relative_cell = relative_cell;
	temp_message._cell_index_max = cell_index_max;
	temp_message._cell_index = cell_index;
	temp_message._agent_grid_cell = agent_grid_cell;

	//Using texture cache
  temp_message.id = tex1Dfetch(tex_xmachine_message_location_id, cell_index + d_tex_xmachine_message_location_id_offset); temp_message.type = tex1Dfetch(tex_xmachine_message_location_type, cell_index + d_tex_xmachine_message_location_type_offset); temp_message.x = tex1Dfetch(tex_xmachine_message_location_x, cell_index + d_tex_xmachine_message_location_x_offset); temp_message.y = tex1Dfetch(tex_xmachine_message_location_y, cell_index + d_tex_xmachine_message_location_y_offset); temp_message.z = tex1Dfetch(tex_xmachine_message_location_z, cell_index + d_tex_xmachine_message_location_z_offset); temp_message.dir = tex1Dfetch(tex_xmachine_message_location_dir, cell_index + d_tex_xmachine_message_location_dir_offset); temp_message.motility = tex1Dfetch(tex_xmachine_message_location_motility, cell_index + d_tex_xmachine_message_location_motility_offset); temp_message.range = tex1Dfetch(tex_xmachine_message_location_range, cell_index + d_tex_xmachine_message_location_range_offset); temp_message.iteration = tex1Dfetch(tex_xmachine_message_location_iteration, cell_index + d_tex_xmachine_message_location_iteration_offset); 

	//load it into shared memory (no sync as no sharing between threads)
	int message_index = SHARE_INDEX(threadIdx.x, sizeof(xmachine_message_location));
	xmachine_message_location* sm_message = ((xmachine_message_location*)&message_share[message_index]);
	sm_message[0] = temp_message;

	return true;
}

/*
 * get first non partitioned location message (first batch load into shared memory)
 */
__device__ xmachine_message_location* get_first_location_message(xmachine_message_location_list* messages, xmachine_message_location_PBM* partition_matrix, float x, float y, float z){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];

	int3 relative_cell = make_int3(-2, -1, -1);
	int cell_index_max = 0;
	int cell_index = 0;
	float3 position = make_float3(x, y, z);
	int3 agent_grid_cell = message_location_grid_position(position);
	
	if (load_next_location_message(messages, partition_matrix, relative_cell, cell_index_max, agent_grid_cell, cell_index))
	{
		int message_index = SHARE_INDEX(threadIdx.x, sizeof(xmachine_message_location));
		return ((xmachine_message_location*)&message_share[message_index]);
	}
	else
	{
		return false;
	}
}

/*
 * get next non partitioned location message (either from SM or next batch load)
 */
__device__ xmachine_message_location* get_next_location_message(xmachine_message_location* message, xmachine_message_location_list* messages, xmachine_message_location_PBM* partition_matrix){
	
	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//TODO: check message count
	
	if (load_next_location_message(messages, partition_matrix, message->_relative_cell, message->_cell_index_max, message->_agent_grid_cell, message->_cell_index))
	{
		//get conflict free address of 
		int message_index = SHARE_INDEX(threadIdx.x, sizeof(xmachine_message_location));
		return ((xmachine_message_location*)&message_share[message_index]);
	}
	else
		return false;
	
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created force message functions */


/** add_force_message
 * Add non partitioned or spatially partitioned force message
 * @param messages xmachine_message_force_list message list to add too
 * @param type agent variable of type int
 * @param x agent variable of type float
 * @param y agent variable of type float
 * @param z agent variable of type float
 * @param id agent variable of type int
 */
__device__ void add_force_message(xmachine_message_force_list* messages, int type, float x, float y, float z, int id){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_force_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_force_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_force_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_force Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->type[index] = type;
	messages->x[index] = x;
	messages->y[index] = y;
	messages->z[index] = z;
	messages->id[index] = id;

}

/**
 * Scatter non partitioned or spatially partitioned force message (for optional messages)
 * @param messages scatter_optional_force_messages Sparse xmachine_message_force_list message list
 * @param message_swap temp xmachine_message_force_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_force_messages(xmachine_message_force_list* messages, xmachine_message_force_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_force_count;

		//AoS - xmachine_message_force Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->type[output_index] = messages_swap->type[index];
		messages->x[output_index] = messages_swap->x[index];
		messages->y[output_index] = messages_swap->y[index];
		messages->z[output_index] = messages_swap->z[index];
		messages->id[output_index] = messages_swap->id[index];				
	}
}

/** reset_force_swaps
 * Reset non partitioned or spatially partitioned force message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_force_swaps(xmachine_message_force_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

/** message_force_grid_position
 * Calculates the grid cell position given an float3 vector
 * @param position float3 vector representing a position
 */
__device__ int3 message_force_grid_position(float3 position)
{
    int3 gridPos;
    gridPos.x = floor((position.x - d_message_force_min_bounds.x) * (float)d_message_force_partitionDim.x / (d_message_force_max_bounds.x - d_message_force_min_bounds.x));
    gridPos.y = floor((position.y - d_message_force_min_bounds.y) * (float)d_message_force_partitionDim.y / (d_message_force_max_bounds.y - d_message_force_min_bounds.y));
    gridPos.z = floor((position.z - d_message_force_min_bounds.z) * (float)d_message_force_partitionDim.z / (d_message_force_max_bounds.z - d_message_force_min_bounds.z));

	//do wrapping or bounding
	

    return gridPos;
}

/** message_force_hash
 * Given the grid position in partition space this function calculates a hash value
 * @param gridPos The position in partition space
 */
__device__ unsigned int message_force_hash(int3 gridPos)
{
	//cheap bounding without mod (within range +- partition dimension)
	gridPos.x = (gridPos.x<0)? d_message_force_partitionDim.x-1: gridPos.x; 
	gridPos.x = (gridPos.x>=d_message_force_partitionDim.x)? 0 : gridPos.x; 
	gridPos.y = (gridPos.y<0)? d_message_force_partitionDim.y-1 : gridPos.y; 
	gridPos.y = (gridPos.y>=d_message_force_partitionDim.y)? 0 : gridPos.y; 
	gridPos.z = (gridPos.z<0)? d_message_force_partitionDim.z-1: gridPos.z; 
	gridPos.z = (gridPos.z>=d_message_force_partitionDim.z)? 0 : gridPos.z; 

	//unique id
	return ((gridPos.z * d_message_force_partitionDim.y) * d_message_force_partitionDim.x) + (gridPos.y * d_message_force_partitionDim.x) + gridPos.x;
}

#ifdef FAST_ATOMIC_SORTING
	/** hist_force_messages
		 * Kernal function for performing a histogram (count) on each partition bin and saving the hash and index of a message within that bin
		 * @param local_bin_index output index of the message within the calculated bin
		 * @param unsorted_index output bin index (hash) value
		 * @param messages the message list used to generate the hash value outputs
		 * @param agent_count the current number of agents outputting messages
		 */
	__global__ void hist_force_messages(uint* local_bin_index, uint* unsorted_index, int* global_bin_count, xmachine_message_force_list* messages, int agent_count)
	{
		unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		if (index >= agent_count)
			return;

		float3 position = make_float3(messages->x[index], messages->y[index], messages->z[index]);
		int3 grid_position = message_force_grid_position(position);
		unsigned int hash = message_force_hash(grid_position);
		unsigned int bin_idx = atomicInc((unsigned int*) &global_bin_count[hash], 0xFFFFFFFF);
		local_bin_index[index] = bin_idx;
		unsorted_index[index] = hash;
	}
	
	/** reorder_force_messages
	 * Reorders the messages accoring to the partition boundary matrix start indices of each bin
	 * @param local_bin_index index of the message within the desired bin
	 * @param unsorted_index bin index (hash) value
	 * @param pbm_start_index the start indices of the partition boundary matrix
	 * @param unordered_messages the original unordered message data
	 * @param ordered_messages buffer used to scatter messages into the correct order
	  @param agent_count the current number of agents outputting messages
	 */
	 __global__ void reorder_force_messages(uint* local_bin_index, uint* unsorted_index, int* pbm_start_index, xmachine_message_force_list* unordered_messages, xmachine_message_force_list* ordered_messages, int agent_count)
	{
		int index = (blockIdx.x *blockDim.x) + threadIdx.x;

		if (index >= agent_count)
			return;

		uint i = unsorted_index[index];
		unsigned int sorted_index = local_bin_index[index] + pbm_start_index[i];

		//finally reorder agent data
		ordered_messages->type[sorted_index] = unordered_messages->type[index];
		ordered_messages->x[sorted_index] = unordered_messages->x[index];
		ordered_messages->y[sorted_index] = unordered_messages->y[index];
		ordered_messages->z[sorted_index] = unordered_messages->z[index];
		ordered_messages->id[sorted_index] = unordered_messages->id[index];
	}
	 
#else

	/** hash_force_messages
	 * Kernal function for calculating a hash value for each messahe depending on its position
	 * @param keys output for the hash key
	 * @param values output for the index value
	 * @param messages the message list used to generate the hash value outputs
	 */
	__global__ void hash_force_messages(uint* keys, uint* values, xmachine_message_force_list* messages)
	{
		unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		float3 position = make_float3(messages->x[index], messages->y[index], messages->z[index]);
		int3 grid_position = message_force_grid_position(position);
		unsigned int hash = message_force_hash(grid_position);

		keys[index] = hash;
		values[index] = index;
	}

	/** reorder_force_messages
	 * Reorders the messages accoring to the ordered sort identifiers and builds a Partition Boundary Matrix by looking at the previosu threads sort id.
	 * @param keys the sorted hash keys
	 * @param values the sorted index values
	 * @param matrix the PBM
	 * @param unordered_messages the original unordered message data
	 * @param ordered_messages buffer used to scatter messages into the correct order
	 */
	__global__ void reorder_force_messages(uint* keys, uint* values, xmachine_message_force_PBM* matrix, xmachine_message_force_list* unordered_messages, xmachine_message_force_list* ordered_messages)
	{
		extern __shared__ int sm_data [];

		int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		//load threads sort key into sm
		uint key = keys[index];
		uint old_pos = values[index];

		sm_data[threadIdx.x] = key;
		__syncthreads();
	
		unsigned int prev_key;

		//if first thread then no prev sm value so get prev from global memory 
		if (threadIdx.x == 0)
		{
			//first thread has no prev value so ignore
			if (index != 0)
				prev_key = keys[index-1];
		}
		//get previous ident from sm
		else	
		{
			prev_key = sm_data[threadIdx.x-1];
		}

		//TODO: Check key is not out of bounds

		//set partition boundaries
		if (index < d_message_force_count)
		{
			//if first thread then set first partition cell start
			if (index == 0)
			{
				matrix->start[key] = index;
			}

			//if edge of a boundr update start and end of partition
			else if (prev_key != key)
			{
				//set start for key
				matrix->start[key] = index;

				//set end for key -1
				matrix->end_or_count[prev_key] = index;
			}

			//if last thread then set final partition cell end
			if (index == d_message_force_count-1)
			{
				matrix->end_or_count[key] = index+1;
			}
		}
	
		//finally reorder agent data
		ordered_messages->type[index] = unordered_messages->type[old_pos];
		ordered_messages->x[index] = unordered_messages->x[old_pos];
		ordered_messages->y[index] = unordered_messages->y[old_pos];
		ordered_messages->z[index] = unordered_messages->z[old_pos];
		ordered_messages->id[index] = unordered_messages->id[old_pos];
	}

#endif

/** load_next_force_message
 * Used to load the next message data to shared memory
 * Idea is check the current cell index to see if we can simpley get a message from the current cell
 * If we are at the end of the current cell then loop till we find the next cell with messages (this way we ignore cells with no messages)
 * @param messages the message list
 * @param partition_matrix the PBM
 * @param relative_cell the relative partition cell position from the agent position
 * @param cell_index_max the maximum index of the currnt partition cell
 * @param agent_grid_cell the agents partition cell position
 * @param cell_index the current cell index in agent_grid_cell+relative_cell
 * @return true if a messag has been loaded into sm false otherwise
 */
__device__ int load_next_force_message(xmachine_message_force_list* messages, xmachine_message_force_PBM* partition_matrix, int3 relative_cell, int cell_index_max, int3 agent_grid_cell, int cell_index)
{
	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];

	int move_cell = true;
	cell_index ++;

	//see if we need to move to a new partition cell
	if(cell_index < cell_index_max)
		move_cell = false;

	while(move_cell)
	{
		//get the next relative grid position 
        if (next_cell3D(&relative_cell))
		{
			//calculate the next cells grid position and hash
			int3 next_cell_position = agent_grid_cell + relative_cell;
			int next_cell_hash = message_force_hash(next_cell_position);
			//use the hash to calculate the start index
			int cell_index_min = tex1Dfetch(tex_xmachine_message_force_pbm_start, next_cell_hash + d_tex_xmachine_message_force_pbm_start_offset);
			cell_index_max = tex1Dfetch(tex_xmachine_message_force_pbm_end_or_count, next_cell_hash + d_tex_xmachine_message_force_pbm_end_or_count_offset);
			//check for messages in the cell (cell index max is the count for atomic sorting)
#ifdef FAST_ATOMIC_SORTING
			if (cell_index_max > 0)
			{
				//when using fast atomics value represents bin count not last index!
				cell_index_max += cell_index_min; //when using fast atomics value represents bin count not last index!
#else
			if (cell_index_min != 0xffffffff)
			{
#endif
				//start from the cell index min
				cell_index = cell_index_min;
				//exit the loop as we have found a valid cell with message data
				move_cell = false;
			}
		}
		else
		{
			//we have exhausted all the neightbouring cells so there are no more messages
			return false;
		}
	}
	
	//get message data using texture fetch
	xmachine_message_force temp_message;
	temp_message._relative_cell = relative_cell;
	temp_message._cell_index_max = cell_index_max;
	temp_message._cell_index = cell_index;
	temp_message._agent_grid_cell = agent_grid_cell;

	//Using texture cache
  temp_message.type = tex1Dfetch(tex_xmachine_message_force_type, cell_index + d_tex_xmachine_message_force_type_offset); temp_message.x = tex1Dfetch(tex_xmachine_message_force_x, cell_index + d_tex_xmachine_message_force_x_offset); temp_message.y = tex1Dfetch(tex_xmachine_message_force_y, cell_index + d_tex_xmachine_message_force_y_offset); temp_message.z = tex1Dfetch(tex_xmachine_message_force_z, cell_index + d_tex_xmachine_message_force_z_offset); temp_message.id = tex1Dfetch(tex_xmachine_message_force_id, cell_index + d_tex_xmachine_message_force_id_offset); 

	//load it into shared memory (no sync as no sharing between threads)
	int message_index = SHARE_INDEX(threadIdx.x, sizeof(xmachine_message_force));
	xmachine_message_force* sm_message = ((xmachine_message_force*)&message_share[message_index]);
	sm_message[0] = temp_message;

	return true;
}

/*
 * get first non partitioned force message (first batch load into shared memory)
 */
__device__ xmachine_message_force* get_first_force_message(xmachine_message_force_list* messages, xmachine_message_force_PBM* partition_matrix, float x, float y, float z){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];

	int3 relative_cell = make_int3(-2, -1, -1);
	int cell_index_max = 0;
	int cell_index = 0;
	float3 position = make_float3(x, y, z);
	int3 agent_grid_cell = message_force_grid_position(position);
	
	if (load_next_force_message(messages, partition_matrix, relative_cell, cell_index_max, agent_grid_cell, cell_index))
	{
		int message_index = SHARE_INDEX(threadIdx.x, sizeof(xmachine_message_force));
		return ((xmachine_message_force*)&message_share[message_index]);
	}
	else
	{
		return false;
	}
}

/*
 * get next non partitioned force message (either from SM or next batch load)
 */
__device__ xmachine_message_force* get_next_force_message(xmachine_message_force* message, xmachine_message_force_list* messages, xmachine_message_force_PBM* partition_matrix){
	
	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//TODO: check message count
	
	if (load_next_force_message(messages, partition_matrix, message->_relative_cell, message->_cell_index_max, message->_agent_grid_cell, message->_cell_index))
	{
		//get conflict free address of 
		int message_index = SHARE_INDEX(threadIdx.x, sizeof(xmachine_message_force));
		return ((xmachine_message_force*)&message_share[message_index]);
	}
	else
		return false;
	
}



	
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created GPU kernals  */



/**
 *
 */
__global__ void GPUFLAME_output_location(xmachine_memory_keratinocyte_list* agents, xmachine_message_location_list* location_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_keratinocyte_count)
        return;
    

	//SoA to AoS - xmachine_memory_output_location Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_keratinocyte agent;
	agent.id = agents->id[index];
	agent.type = agents->type[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.force_x = agents->force_x[index];
	agent.force_y = agents->force_y[index];
	agent.force_z = agents->force_z[index];
	agent.num_xy_bonds = agents->num_xy_bonds[index];
	agent.num_z_bonds = agents->num_z_bonds[index];
	agent.num_stem_bonds = agents->num_stem_bonds[index];
	agent.cycle = agents->cycle[index];
	agent.diff_noise_factor = agents->diff_noise_factor[index];
	agent.dead_ticks = agents->dead_ticks[index];
	agent.contact_inhibited_ticks = agents->contact_inhibited_ticks[index];
	agent.motility = agents->motility[index];
	agent.dir = agents->dir[index];
	agent.movement = agents->movement[index];

	//FLAME function call
	int dead = !output_location(&agent, location_messages	);
	
	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_output_location Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->type[index] = agent.type;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->force_x[index] = agent.force_x;
	agents->force_y[index] = agent.force_y;
	agents->force_z[index] = agent.force_z;
	agents->num_xy_bonds[index] = agent.num_xy_bonds;
	agents->num_z_bonds[index] = agent.num_z_bonds;
	agents->num_stem_bonds[index] = agent.num_stem_bonds;
	agents->cycle[index] = agent.cycle;
	agents->diff_noise_factor[index] = agent.diff_noise_factor;
	agents->dead_ticks[index] = agent.dead_ticks;
	agents->contact_inhibited_ticks[index] = agent.contact_inhibited_ticks;
	agents->motility[index] = agent.motility;
	agents->dir[index] = agent.dir;
	agents->movement[index] = agent.movement;
}

/**
 *
 */
__global__ void GPUFLAME_cycle(xmachine_memory_keratinocyte_list* agents, xmachine_memory_keratinocyte_list* keratinocyte_agents, RNG_rand48* rand48){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_keratinocyte_count)
        return;
    

	//SoA to AoS - xmachine_memory_cycle Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_keratinocyte agent;
	agent.id = agents->id[index];
	agent.type = agents->type[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.force_x = agents->force_x[index];
	agent.force_y = agents->force_y[index];
	agent.force_z = agents->force_z[index];
	agent.num_xy_bonds = agents->num_xy_bonds[index];
	agent.num_z_bonds = agents->num_z_bonds[index];
	agent.num_stem_bonds = agents->num_stem_bonds[index];
	agent.cycle = agents->cycle[index];
	agent.diff_noise_factor = agents->diff_noise_factor[index];
	agent.dead_ticks = agents->dead_ticks[index];
	agent.contact_inhibited_ticks = agents->contact_inhibited_ticks[index];
	agent.motility = agents->motility[index];
	agent.dir = agents->dir[index];
	agent.movement = agents->movement[index];

	//FLAME function call
	int dead = !cycle(&agent, keratinocyte_agents, rand48);
	
	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_cycle Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->type[index] = agent.type;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->force_x[index] = agent.force_x;
	agents->force_y[index] = agent.force_y;
	agents->force_z[index] = agent.force_z;
	agents->num_xy_bonds[index] = agent.num_xy_bonds;
	agents->num_z_bonds[index] = agent.num_z_bonds;
	agents->num_stem_bonds[index] = agent.num_stem_bonds;
	agents->cycle[index] = agent.cycle;
	agents->diff_noise_factor[index] = agent.diff_noise_factor;
	agents->dead_ticks[index] = agent.dead_ticks;
	agents->contact_inhibited_ticks[index] = agent.contact_inhibited_ticks;
	agents->motility[index] = agent.motility;
	agents->dir[index] = agent.dir;
	agents->movement[index] = agent.movement;
}

/**
 *
 */
__global__ void GPUFLAME_differentiate(xmachine_memory_keratinocyte_list* agents, xmachine_message_location_list* location_messages, xmachine_message_location_PBM* partition_matrix){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_keratinocyte_count)
        return;
    

	//SoA to AoS - xmachine_memory_differentiate Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_keratinocyte agent;
	agent.id = agents->id[index];
	agent.type = agents->type[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.force_x = agents->force_x[index];
	agent.force_y = agents->force_y[index];
	agent.force_z = agents->force_z[index];
	agent.num_xy_bonds = agents->num_xy_bonds[index];
	agent.num_z_bonds = agents->num_z_bonds[index];
	agent.num_stem_bonds = agents->num_stem_bonds[index];
	agent.cycle = agents->cycle[index];
	agent.diff_noise_factor = agents->diff_noise_factor[index];
	agent.dead_ticks = agents->dead_ticks[index];
	agent.contact_inhibited_ticks = agents->contact_inhibited_ticks[index];
	agent.motility = agents->motility[index];
	agent.dir = agents->dir[index];
	agent.movement = agents->movement[index];

	//FLAME function call
	int dead = !differentiate(&agent, location_messages, partition_matrix);
	
	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_differentiate Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->type[index] = agent.type;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->force_x[index] = agent.force_x;
	agents->force_y[index] = agent.force_y;
	agents->force_z[index] = agent.force_z;
	agents->num_xy_bonds[index] = agent.num_xy_bonds;
	agents->num_z_bonds[index] = agent.num_z_bonds;
	agents->num_stem_bonds[index] = agent.num_stem_bonds;
	agents->cycle[index] = agent.cycle;
	agents->diff_noise_factor[index] = agent.diff_noise_factor;
	agents->dead_ticks[index] = agent.dead_ticks;
	agents->contact_inhibited_ticks[index] = agent.contact_inhibited_ticks;
	agents->motility[index] = agent.motility;
	agents->dir[index] = agent.dir;
	agents->movement[index] = agent.movement;
}

/**
 *
 */
__global__ void GPUFLAME_death_signal(xmachine_memory_keratinocyte_list* agents, xmachine_message_location_list* location_messages, xmachine_message_location_PBM* partition_matrix, RNG_rand48* rand48){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_keratinocyte_count)
        return;
    

	//SoA to AoS - xmachine_memory_death_signal Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_keratinocyte agent;
	agent.id = agents->id[index];
	agent.type = agents->type[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.force_x = agents->force_x[index];
	agent.force_y = agents->force_y[index];
	agent.force_z = agents->force_z[index];
	agent.num_xy_bonds = agents->num_xy_bonds[index];
	agent.num_z_bonds = agents->num_z_bonds[index];
	agent.num_stem_bonds = agents->num_stem_bonds[index];
	agent.cycle = agents->cycle[index];
	agent.diff_noise_factor = agents->diff_noise_factor[index];
	agent.dead_ticks = agents->dead_ticks[index];
	agent.contact_inhibited_ticks = agents->contact_inhibited_ticks[index];
	agent.motility = agents->motility[index];
	agent.dir = agents->dir[index];
	agent.movement = agents->movement[index];

	//FLAME function call
	int dead = !death_signal(&agent, location_messages, partition_matrix, rand48);
	
	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_death_signal Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->type[index] = agent.type;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->force_x[index] = agent.force_x;
	agents->force_y[index] = agent.force_y;
	agents->force_z[index] = agent.force_z;
	agents->num_xy_bonds[index] = agent.num_xy_bonds;
	agents->num_z_bonds[index] = agent.num_z_bonds;
	agents->num_stem_bonds[index] = agent.num_stem_bonds;
	agents->cycle[index] = agent.cycle;
	agents->diff_noise_factor[index] = agent.diff_noise_factor;
	agents->dead_ticks[index] = agent.dead_ticks;
	agents->contact_inhibited_ticks[index] = agent.contact_inhibited_ticks;
	agents->motility[index] = agent.motility;
	agents->dir[index] = agent.dir;
	agents->movement[index] = agent.movement;
}

/**
 *
 */
__global__ void GPUFLAME_migrate(xmachine_memory_keratinocyte_list* agents, xmachine_message_location_list* location_messages, xmachine_message_location_PBM* partition_matrix, RNG_rand48* rand48){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_keratinocyte_count)
        return;
    

	//SoA to AoS - xmachine_memory_migrate Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_keratinocyte agent;
	agent.id = agents->id[index];
	agent.type = agents->type[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.force_x = agents->force_x[index];
	agent.force_y = agents->force_y[index];
	agent.force_z = agents->force_z[index];
	agent.num_xy_bonds = agents->num_xy_bonds[index];
	agent.num_z_bonds = agents->num_z_bonds[index];
	agent.num_stem_bonds = agents->num_stem_bonds[index];
	agent.cycle = agents->cycle[index];
	agent.diff_noise_factor = agents->diff_noise_factor[index];
	agent.dead_ticks = agents->dead_ticks[index];
	agent.contact_inhibited_ticks = agents->contact_inhibited_ticks[index];
	agent.motility = agents->motility[index];
	agent.dir = agents->dir[index];
	agent.movement = agents->movement[index];

	//FLAME function call
	int dead = !migrate(&agent, location_messages, partition_matrix, rand48);
	
	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_migrate Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->type[index] = agent.type;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->force_x[index] = agent.force_x;
	agents->force_y[index] = agent.force_y;
	agents->force_z[index] = agent.force_z;
	agents->num_xy_bonds[index] = agent.num_xy_bonds;
	agents->num_z_bonds[index] = agent.num_z_bonds;
	agents->num_stem_bonds[index] = agent.num_stem_bonds;
	agents->cycle[index] = agent.cycle;
	agents->diff_noise_factor[index] = agent.diff_noise_factor;
	agents->dead_ticks[index] = agent.dead_ticks;
	agents->contact_inhibited_ticks[index] = agent.contact_inhibited_ticks;
	agents->motility[index] = agent.motility;
	agents->dir[index] = agent.dir;
	agents->movement[index] = agent.movement;
}

/**
 *
 */
__global__ void GPUFLAME_force_resolution_output(xmachine_memory_keratinocyte_list* agents, xmachine_message_force_list* force_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_keratinocyte_count)
        return;
    

	//SoA to AoS - xmachine_memory_force_resolution_output Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_keratinocyte agent;
	agent.id = agents->id[index];
	agent.type = agents->type[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.force_x = agents->force_x[index];
	agent.force_y = agents->force_y[index];
	agent.force_z = agents->force_z[index];
	agent.num_xy_bonds = agents->num_xy_bonds[index];
	agent.num_z_bonds = agents->num_z_bonds[index];
	agent.num_stem_bonds = agents->num_stem_bonds[index];
	agent.cycle = agents->cycle[index];
	agent.diff_noise_factor = agents->diff_noise_factor[index];
	agent.dead_ticks = agents->dead_ticks[index];
	agent.contact_inhibited_ticks = agents->contact_inhibited_ticks[index];
	agent.motility = agents->motility[index];
	agent.dir = agents->dir[index];
	agent.movement = agents->movement[index];

	//FLAME function call
	int dead = !force_resolution_output(&agent, force_messages	);
	
	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_force_resolution_output Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->type[index] = agent.type;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->force_x[index] = agent.force_x;
	agents->force_y[index] = agent.force_y;
	agents->force_z[index] = agent.force_z;
	agents->num_xy_bonds[index] = agent.num_xy_bonds;
	agents->num_z_bonds[index] = agent.num_z_bonds;
	agents->num_stem_bonds[index] = agent.num_stem_bonds;
	agents->cycle[index] = agent.cycle;
	agents->diff_noise_factor[index] = agent.diff_noise_factor;
	agents->dead_ticks[index] = agent.dead_ticks;
	agents->contact_inhibited_ticks[index] = agent.contact_inhibited_ticks;
	agents->motility[index] = agent.motility;
	agents->dir[index] = agent.dir;
	agents->movement[index] = agent.movement;
}

/**
 *
 */
__global__ void GPUFLAME_resolve_forces(xmachine_memory_keratinocyte_list* agents, xmachine_message_force_list* force_messages, xmachine_message_force_PBM* partition_matrix){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_keratinocyte_count)
        return;
    

	//SoA to AoS - xmachine_memory_resolve_forces Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_keratinocyte agent;
	agent.id = agents->id[index];
	agent.type = agents->type[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.force_x = agents->force_x[index];
	agent.force_y = agents->force_y[index];
	agent.force_z = agents->force_z[index];
	agent.num_xy_bonds = agents->num_xy_bonds[index];
	agent.num_z_bonds = agents->num_z_bonds[index];
	agent.num_stem_bonds = agents->num_stem_bonds[index];
	agent.cycle = agents->cycle[index];
	agent.diff_noise_factor = agents->diff_noise_factor[index];
	agent.dead_ticks = agents->dead_ticks[index];
	agent.contact_inhibited_ticks = agents->contact_inhibited_ticks[index];
	agent.motility = agents->motility[index];
	agent.dir = agents->dir[index];
	agent.movement = agents->movement[index];

	//FLAME function call
	int dead = !resolve_forces(&agent, force_messages, partition_matrix);
	
	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_resolve_forces Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->type[index] = agent.type;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->force_x[index] = agent.force_x;
	agents->force_y[index] = agent.force_y;
	agents->force_z[index] = agent.force_z;
	agents->num_xy_bonds[index] = agent.num_xy_bonds;
	agents->num_z_bonds[index] = agent.num_z_bonds;
	agents->num_stem_bonds[index] = agent.num_stem_bonds;
	agents->cycle[index] = agent.cycle;
	agents->diff_noise_factor[index] = agent.diff_noise_factor;
	agents->dead_ticks[index] = agent.dead_ticks;
	agents->contact_inhibited_ticks[index] = agent.contact_inhibited_ticks;
	agents->motility[index] = agent.motility;
	agents->dir[index] = agent.dir;
	agents->movement[index] = agent.movement;
}

	
	
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Rand48 functions */

__device__ static uint2 RNG_rand48_iterate_single(uint2 Xn, uint2 A, uint2 C)
{
	unsigned int R0, R1;

	// low 24-bit multiplication
	const unsigned int lo00 = __umul24(Xn.x, A.x);
	const unsigned int hi00 = __umulhi(Xn.x, A.x);

	// 24bit distribution of 32bit multiplication results
	R0 = (lo00 & 0xFFFFFF);
	R1 = (lo00 >> 24) | (hi00 << 8);

	R0 += C.x; R1 += C.y;

	// transfer overflows
	R1 += (R0 >> 24);
	R0 &= 0xFFFFFF;

	// cross-terms, low/hi 24-bit multiplication
	R1 += __umul24(Xn.y, A.x);
	R1 += __umul24(Xn.x, A.y);

	R1 &= 0xFFFFFF;

	return make_uint2(R0, R1);
}

//Templated function
template <int AGENT_TYPE>
__device__ float rnd(RNG_rand48* rand48){

	int index;
	
	//calculate the agents index in global agent list
	if (AGENT_TYPE == DISCRETE_2D){
		int width = (blockDim.x * gridDim.x);
		int2 global_position;
		global_position.x = (blockIdx.x * blockDim.x) + threadIdx.x;
		global_position.y = (blockIdx.y * blockDim.y) + threadIdx.y;
		index = global_position.x + (global_position.y * width);
	}else//AGENT_TYPE == CONTINOUS
		index = threadIdx.x + blockIdx.x*blockDim.x;

	uint2 state = rand48->seeds[index];
	uint2 A = rand48->A;
	uint2 C = rand48->C;

	int rand = ( state.x >> 17 ) | ( state.y << 7);

	// this actually iterates the RNG
	state = RNG_rand48_iterate_single(state, A, C);

	rand48->seeds[index] = state;

	return (float)rand/2147483647;
}

__device__ float rnd(RNG_rand48* rand48){
	return rnd<DISCRETE_2D>(rand48);
}

#endif //_FLAMEGPU_KERNELS_H_
