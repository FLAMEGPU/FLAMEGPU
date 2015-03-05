

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

__constant__ int d_xmachine_memory_agent_count;

__constant__ int d_xmachine_memory_navmap_count;

/* Agent state count constants */

__constant__ int d_xmachine_memory_agent_default_count;

__constant__ int d_xmachine_memory_navmap_static_count;


/* Message constants */

/* pedestrian_location Message variables */
/* Non partitioned and spatial partitioned message variables  */
__constant__ int d_message_pedestrian_location_count;         /**< message list counter*/
__constant__ int d_message_pedestrian_location_output_type;   /**< message output type (single or optional)*/
//Spatial Partitioning Variables
__constant__ float3 d_message_pedestrian_location_min_bounds;           /**< min bounds (x,y,z) of partitioning environment */
__constant__ float3 d_message_pedestrian_location_max_bounds;           /**< max bounds (x,y,z) of partitioning environment */
__constant__ int3 d_message_pedestrian_location_partitionDim;           /**< partition dimensions (x,y,z) of partitioning environment */
__constant__ float d_message_pedestrian_location_radius;                 /**< partition radius (used to determin the size of the partitions) */

/* navmap_cell Message variables */
//Discrete Partitioning Variables
__constant__ int d_message_navmap_cell_range;     /**< range of the discrete message*/
__constant__ int d_message_navmap_cell_width;     /**< with of the message grid*/

	
    
//include each function file

#include "functions.c"
    
/* Texture bindings */
/* pedestrian_location Message Bindings */texture<float, 1, cudaReadModeElementType> tex_xmachine_message_pedestrian_location_x;
__constant__ int d_tex_xmachine_message_pedestrian_location_x_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_pedestrian_location_y;
__constant__ int d_tex_xmachine_message_pedestrian_location_y_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_pedestrian_location_z;
__constant__ int d_tex_xmachine_message_pedestrian_location_z_offset;
texture<int, 1, cudaReadModeElementType> tex_xmachine_message_pedestrian_location_pbm_start;
__constant__ int d_tex_xmachine_message_pedestrian_location_pbm_start_offset;
texture<int, 1, cudaReadModeElementType> tex_xmachine_message_pedestrian_location_pbm_end_or_count;
__constant__ int d_tex_xmachine_message_pedestrian_location_pbm_end_or_count_offset;


/* navmap_cell Message Bindings */texture<int, 1, cudaReadModeElementType> tex_xmachine_message_navmap_cell_x;
__constant__ int d_tex_xmachine_message_navmap_cell_x_offset;texture<int, 1, cudaReadModeElementType> tex_xmachine_message_navmap_cell_y;
__constant__ int d_tex_xmachine_message_navmap_cell_y_offset;texture<int, 1, cudaReadModeElementType> tex_xmachine_message_navmap_cell_exit_no;
__constant__ int d_tex_xmachine_message_navmap_cell_exit_no_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_navmap_cell_height;
__constant__ int d_tex_xmachine_message_navmap_cell_height_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_navmap_cell_collision_x;
__constant__ int d_tex_xmachine_message_navmap_cell_collision_x_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_navmap_cell_collision_y;
__constant__ int d_tex_xmachine_message_navmap_cell_collision_y_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_navmap_cell_exit0_x;
__constant__ int d_tex_xmachine_message_navmap_cell_exit0_x_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_navmap_cell_exit0_y;
__constant__ int d_tex_xmachine_message_navmap_cell_exit0_y_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_navmap_cell_exit1_x;
__constant__ int d_tex_xmachine_message_navmap_cell_exit1_x_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_navmap_cell_exit1_y;
__constant__ int d_tex_xmachine_message_navmap_cell_exit1_y_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_navmap_cell_exit2_x;
__constant__ int d_tex_xmachine_message_navmap_cell_exit2_x_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_navmap_cell_exit2_y;
__constant__ int d_tex_xmachine_message_navmap_cell_exit2_y_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_navmap_cell_exit3_x;
__constant__ int d_tex_xmachine_message_navmap_cell_exit3_x_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_navmap_cell_exit3_y;
__constant__ int d_tex_xmachine_message_navmap_cell_exit3_y_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_navmap_cell_exit4_x;
__constant__ int d_tex_xmachine_message_navmap_cell_exit4_x_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_navmap_cell_exit4_y;
__constant__ int d_tex_xmachine_message_navmap_cell_exit4_y_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_navmap_cell_exit5_x;
__constant__ int d_tex_xmachine_message_navmap_cell_exit5_x_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_navmap_cell_exit5_y;
__constant__ int d_tex_xmachine_message_navmap_cell_exit5_y_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_navmap_cell_exit6_x;
__constant__ int d_tex_xmachine_message_navmap_cell_exit6_x_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_navmap_cell_exit6_y;
__constant__ int d_tex_xmachine_message_navmap_cell_exit6_y_offset;

    
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


////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created agent agent functions */

/** reset_agent_scan_input
 * agent agent reset scan input function
 * @param agents The xmachine_memory_agent_list agent list
 */
__global__ void reset_agent_scan_input(xmachine_memory_agent_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_agent_Agents
 * agent scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_agent_list agent list destination
 * @param agents_src xmachine_memory_agent_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_agent_Agents(xmachine_memory_agent_list* agents_dst, xmachine_memory_agent_list* agents_src, int dst_agent_count, int number_to_scatter){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = agents_src->_scan_input[index];

	//if optional message is to be written. 
	//must check agent is within number to scatter as unused threads may have scan input = 1
	if ((_scan_input == 1)&&(index < number_to_scatter)){
		int output_index = agents_src->_position[index] + dst_agent_count;

		//AoS - xmachine_message_location Un-Coalesced scattered memory write     
        agents_dst->_position[output_index] = output_index;        
		agents_dst->x[output_index] = agents_src->x[index];        
		agents_dst->y[output_index] = agents_src->y[index];        
		agents_dst->velx[output_index] = agents_src->velx[index];        
		agents_dst->vely[output_index] = agents_src->vely[index];        
		agents_dst->steer_x[output_index] = agents_src->steer_x[index];        
		agents_dst->steer_y[output_index] = agents_src->steer_y[index];        
		agents_dst->height[output_index] = agents_src->height[index];        
		agents_dst->exit_no[output_index] = agents_src->exit_no[index];        
		agents_dst->speed[output_index] = agents_src->speed[index];        
		agents_dst->lod[output_index] = agents_src->lod[index];        
		agents_dst->animate[output_index] = agents_src->animate[index];        
		agents_dst->animate_dir[output_index] = agents_src->animate_dir[index];
	}
}

/** append_agent_Agents
 * agent scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_agent_list agent list destination
 * @param agents_src xmachine_memory_agent_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_agent_Agents(xmachine_memory_agent_list* agents_dst, xmachine_memory_agent_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->x[output_index] = agents_src->x[index];
	    agents_dst->y[output_index] = agents_src->y[index];
	    agents_dst->velx[output_index] = agents_src->velx[index];
	    agents_dst->vely[output_index] = agents_src->vely[index];
	    agents_dst->steer_x[output_index] = agents_src->steer_x[index];
	    agents_dst->steer_y[output_index] = agents_src->steer_y[index];
	    agents_dst->height[output_index] = agents_src->height[index];
	    agents_dst->exit_no[output_index] = agents_src->exit_no[index];
	    agents_dst->speed[output_index] = agents_src->speed[index];
	    agents_dst->lod[output_index] = agents_src->lod[index];
	    agents_dst->animate[output_index] = agents_src->animate[index];
	    agents_dst->animate_dir[output_index] = agents_src->animate_dir[index];
    }
}

/** add_agent_agent
 * Continuous agent agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_agent_list to add agents to 
 * @param x agent variable of type float
 * @param y agent variable of type float
 * @param velx agent variable of type float
 * @param vely agent variable of type float
 * @param steer_x agent variable of type float
 * @param steer_y agent variable of type float
 * @param height agent variable of type float
 * @param exit_no agent variable of type int
 * @param speed agent variable of type float
 * @param lod agent variable of type int
 * @param animate agent variable of type float
 * @param animate_dir agent variable of type int
 */
template <int AGENT_TYPE>
__device__ void add_agent_agent(xmachine_memory_agent_list* agents, float x, float y, float velx, float vely, float steer_x, float steer_y, float height, int exit_no, float speed, int lod, float animate, int animate_dir){
	
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
	agents->x[index] = x;
	agents->y[index] = y;
	agents->velx[index] = velx;
	agents->vely[index] = vely;
	agents->steer_x[index] = steer_x;
	agents->steer_y[index] = steer_y;
	agents->height[index] = height;
	agents->exit_no[index] = exit_no;
	agents->speed[index] = speed;
	agents->lod[index] = lod;
	agents->animate[index] = animate;
	agents->animate_dir[index] = animate_dir;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_agent_agent(xmachine_memory_agent_list* agents, float x, float y, float velx, float vely, float steer_x, float steer_y, float height, int exit_no, float speed, int lod, float animate, int animate_dir){
    add_agent_agent<DISCRETE_2D>(agents, x, y, velx, vely, steer_x, steer_y, height, exit_no, speed, lod, animate, animate_dir);
}

/** reorder_agent_agents
 * Continuous agent agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_agent_agents(unsigned int* values, xmachine_memory_agent_list* unordered_agents, xmachine_memory_agent_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->x[index] = unordered_agents->x[old_pos];
	ordered_agents->y[index] = unordered_agents->y[old_pos];
	ordered_agents->velx[index] = unordered_agents->velx[old_pos];
	ordered_agents->vely[index] = unordered_agents->vely[old_pos];
	ordered_agents->steer_x[index] = unordered_agents->steer_x[old_pos];
	ordered_agents->steer_y[index] = unordered_agents->steer_y[old_pos];
	ordered_agents->height[index] = unordered_agents->height[old_pos];
	ordered_agents->exit_no[index] = unordered_agents->exit_no[old_pos];
	ordered_agents->speed[index] = unordered_agents->speed[old_pos];
	ordered_agents->lod[index] = unordered_agents->lod[old_pos];
	ordered_agents->animate[index] = unordered_agents->animate[old_pos];
	ordered_agents->animate_dir[index] = unordered_agents->animate_dir[old_pos];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created navmap agent functions */

/** reset_navmap_scan_input
 * navmap agent reset scan input function
 * @param agents The xmachine_memory_navmap_list agent list
 */
__global__ void reset_navmap_scan_input(xmachine_memory_navmap_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created pedestrian_location message functions */


/** add_pedestrian_location_message
 * Add non partitioned or spatially partitioned pedestrian_location message
 * @param messages xmachine_message_pedestrian_location_list message list to add too
 * @param x agent variable of type float
 * @param y agent variable of type float
 * @param z agent variable of type float
 */
__device__ void add_pedestrian_location_message(xmachine_message_pedestrian_location_list* messages, float x, float y, float z){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_pedestrian_location_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_pedestrian_location_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_pedestrian_location_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_pedestrian_location Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->x[index] = x;
	messages->y[index] = y;
	messages->z[index] = z;

}

/**
 * Scatter non partitioned or spatially partitioned pedestrian_location message (for optional messages)
 * @param messages scatter_optional_pedestrian_location_messages Sparse xmachine_message_pedestrian_location_list message list
 * @param message_swap temp xmachine_message_pedestrian_location_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_pedestrian_location_messages(xmachine_message_pedestrian_location_list* messages, xmachine_message_pedestrian_location_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_pedestrian_location_count;

		//AoS - xmachine_message_pedestrian_location Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->x[output_index] = messages_swap->x[index];
		messages->y[output_index] = messages_swap->y[index];
		messages->z[output_index] = messages_swap->z[index];				
	}
}

/** reset_pedestrian_location_swaps
 * Reset non partitioned or spatially partitioned pedestrian_location message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_pedestrian_location_swaps(xmachine_message_pedestrian_location_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

/** message_pedestrian_location_grid_position
 * Calculates the grid cell position given an float3 vector
 * @param position float3 vector representing a position
 */
__device__ int3 message_pedestrian_location_grid_position(float3 position)
{
    int3 gridPos;
    gridPos.x = floor((position.x - d_message_pedestrian_location_min_bounds.x) * (float)d_message_pedestrian_location_partitionDim.x / (d_message_pedestrian_location_max_bounds.x - d_message_pedestrian_location_min_bounds.x));
    gridPos.y = floor((position.y - d_message_pedestrian_location_min_bounds.y) * (float)d_message_pedestrian_location_partitionDim.y / (d_message_pedestrian_location_max_bounds.y - d_message_pedestrian_location_min_bounds.y));
    gridPos.z = floor((position.z - d_message_pedestrian_location_min_bounds.z) * (float)d_message_pedestrian_location_partitionDim.z / (d_message_pedestrian_location_max_bounds.z - d_message_pedestrian_location_min_bounds.z));

	//do wrapping or bounding
	

    return gridPos;
}

/** message_pedestrian_location_hash
 * Given the grid position in partition space this function calculates a hash value
 * @param gridPos The position in partition space
 */
__device__ unsigned int message_pedestrian_location_hash(int3 gridPos)
{
	//cheap bounding without mod (within range +- partition dimension)
	gridPos.x = (gridPos.x<0)? d_message_pedestrian_location_partitionDim.x-1: gridPos.x; 
	gridPos.x = (gridPos.x>=d_message_pedestrian_location_partitionDim.x)? 0 : gridPos.x; 
	gridPos.y = (gridPos.y<0)? d_message_pedestrian_location_partitionDim.y-1 : gridPos.y; 
	gridPos.y = (gridPos.y>=d_message_pedestrian_location_partitionDim.y)? 0 : gridPos.y; 
	gridPos.z = (gridPos.z<0)? d_message_pedestrian_location_partitionDim.z-1: gridPos.z; 
	gridPos.z = (gridPos.z>=d_message_pedestrian_location_partitionDim.z)? 0 : gridPos.z; 

	//unique id
	return ((gridPos.z * d_message_pedestrian_location_partitionDim.y) * d_message_pedestrian_location_partitionDim.x) + (gridPos.y * d_message_pedestrian_location_partitionDim.x) + gridPos.x;
}

#ifdef FAST_ATOMIC_SORTING
	/** hist_pedestrian_location_messages
		 * Kernal function for performing a histogram (count) on each partition bin and saving the hash and index of a message within that bin
		 * @param local_bin_index output index of the message within the calculated bin
		 * @param unsorted_index output bin index (hash) value
		 * @param messages the message list used to generate the hash value outputs
		 */
	__global__ void hist_pedestrian_location_messages(uint* local_bin_index, uint* unsorted_index, int* global_bin_count, xmachine_message_pedestrian_location_list* messages)
	{
		unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		float3 position = make_float3(messages->x[index], messages->y[index], messages->z[index]);
		int3 grid_position = message_pedestrian_location_grid_position(position);
		unsigned int hash = message_pedestrian_location_hash(grid_position);
		unsigned int bin_idx = atomicInc((unsigned int*) &global_bin_count[hash], 0xFFFFFFFF);
		local_bin_index[index] = bin_idx;
		unsorted_index[index] = hash;
	}
	
	/** reorder_pedestrian_location_messages
	 * Reorders the messages accoring to the partition boundary matrix start indices of each bin
	 * @param local_bin_index index of the message within the desired bin
	 * @param unsorted_index bin index (hash) value
	 * @param pbm_start_index the start indices of the partition boundary matrix
	 * @param unordered_messages the original unordered message data
	 * @param ordered_messages buffer used to scatter messages into the correct order
	 */
	 __global__ void reorder_pedestrian_location_messages(uint* local_bin_index, uint* unsorted_index, int* pbm_start_index, xmachine_message_pedestrian_location_list* unordered_messages, xmachine_message_pedestrian_location_list* ordered_messages)
	{
		int index = (blockIdx.x *blockDim.x) + threadIdx.x;

		uint i = unsorted_index[index];
		unsigned int sorted_index = local_bin_index[index] + pbm_start_index[i];

		//finally reorder agent data
		ordered_messages->x[sorted_index] = unordered_messages->x[index];
		ordered_messages->y[sorted_index] = unordered_messages->y[index];
		ordered_messages->z[sorted_index] = unordered_messages->z[index];
	}
	 
#else

	/** hash_pedestrian_location_messages
	 * Kernal function for calculating a hash value for each messahe depending on its position
	 * @param keys output for the hash key
	 * @param values output for the index value
	 * @param messages the message list used to generate the hash value outputs
	 */
	__global__ void hash_pedestrian_location_messages(uint* keys, uint* values, xmachine_message_pedestrian_location_list* messages)
	{
		unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		float3 position = make_float3(messages->x[index], messages->y[index], messages->z[index]);
		int3 grid_position = message_pedestrian_location_grid_position(position);
		unsigned int hash = message_pedestrian_location_hash(grid_position);

		keys[index] = hash;
		values[index] = index;
	}

	/** reorder_pedestrian_location_messages
	 * Reorders the messages accoring to the ordered sort identifiers and builds a Partition Boundary Matrix by looking at the previosu threads sort id.
	 * @param keys the sorted hash keys
	 * @param values the sorted index values
	 * @param matrix the PBM
	 * @param unordered_messages the original unordered message data
	 * @param ordered_messages buffer used to scatter messages into the correct order
	 */
	__global__ void reorder_pedestrian_location_messages(uint* keys, uint* values, xmachine_message_pedestrian_location_PBM* matrix, xmachine_message_pedestrian_location_list* unordered_messages, xmachine_message_pedestrian_location_list* ordered_messages)
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
		if (index < d_message_pedestrian_location_count)
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
			if (index == d_message_pedestrian_location_count-1)
			{
				matrix->end_or_count[key] = index+1;
			}
		}
	
		//finally reorder agent data
		ordered_messages->x[index] = unordered_messages->x[old_pos];
		ordered_messages->y[index] = unordered_messages->y[old_pos];
		ordered_messages->z[index] = unordered_messages->z[old_pos];
	}

#endif

/** load_next_pedestrian_location_message
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
__device__ int load_next_pedestrian_location_message(xmachine_message_pedestrian_location_list* messages, xmachine_message_pedestrian_location_PBM* partition_matrix, int3 relative_cell, int cell_index_max, int3 agent_grid_cell, int cell_index)
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
        if (next_cell2D(&relative_cell))
		{
			//calculate the next cells grid position and hash
			int3 next_cell_position = agent_grid_cell + relative_cell;
			int next_cell_hash = message_pedestrian_location_hash(next_cell_position);
			//use the hash to calculate the start index
			int cell_index_min = tex1Dfetch(tex_xmachine_message_pedestrian_location_pbm_start, next_cell_hash + d_tex_xmachine_message_pedestrian_location_pbm_start_offset);

			//check for messages in the cell (empty cells with have a start index of baadf00d
			if (cell_index_min != 0xffffffff)
			{
				//if there are messages in the cell then update the cell index max value
				cell_index_max = tex1Dfetch(tex_xmachine_message_pedestrian_location_pbm_end_or_count, next_cell_hash + d_tex_xmachine_message_pedestrian_location_pbm_end_or_count_offset);
#ifdef FAST_ATOMIC_SORTING
				cell_index_max += cell_index_min; //when using fast atomics value represents bin count not last index!
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
	xmachine_message_pedestrian_location temp_message;
	temp_message._relative_cell = relative_cell;
	temp_message._cell_index_max = cell_index_max;
	temp_message._cell_index = cell_index;
	temp_message._agent_grid_cell = agent_grid_cell;

	//Using texture cache
  temp_message.x = tex1Dfetch(tex_xmachine_message_pedestrian_location_x, cell_index + d_tex_xmachine_message_pedestrian_location_x_offset); temp_message.y = tex1Dfetch(tex_xmachine_message_pedestrian_location_y, cell_index + d_tex_xmachine_message_pedestrian_location_y_offset); temp_message.z = tex1Dfetch(tex_xmachine_message_pedestrian_location_z, cell_index + d_tex_xmachine_message_pedestrian_location_z_offset); 

	//load it into shared memory (no sync as no sharing between threads)
	int message_index = SHARE_INDEX(threadIdx.x, sizeof(xmachine_message_pedestrian_location));
	xmachine_message_pedestrian_location* sm_message = ((xmachine_message_pedestrian_location*)&message_share[message_index]);
	sm_message[0] = temp_message;

	return true;
}

/*
 * get first non partitioned pedestrian_location message (first batch load into shared memory)
 */
__device__ xmachine_message_pedestrian_location* get_first_pedestrian_location_message(xmachine_message_pedestrian_location_list* messages, xmachine_message_pedestrian_location_PBM* partition_matrix, float x, float y, float z){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];

	int3 relative_cell = make_int3(-2, -1, -1);
	int cell_index_max = 0;
	int cell_index = 0;
	float3 position = make_float3(x, y, z);
	int3 agent_grid_cell = message_pedestrian_location_grid_position(position);
	
	if (load_next_pedestrian_location_message(messages, partition_matrix, relative_cell, cell_index_max, agent_grid_cell, cell_index))
	{
		int message_index = SHARE_INDEX(threadIdx.x, sizeof(xmachine_message_pedestrian_location));
		return ((xmachine_message_pedestrian_location*)&message_share[message_index]);
	}
	else
	{
		return false;
	}
}

/*
 * get next non partitioned pedestrian_location message (either from SM or next batch load)
 */
__device__ xmachine_message_pedestrian_location* get_next_pedestrian_location_message(xmachine_message_pedestrian_location* message, xmachine_message_pedestrian_location_list* messages, xmachine_message_pedestrian_location_PBM* partition_matrix){
	
	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//TODO: check message count
	
	if (load_next_pedestrian_location_message(messages, partition_matrix, message->_relative_cell, message->_cell_index_max, message->_agent_grid_cell, message->_cell_index))
	{
		//get conflict free address of 
		int message_index = SHARE_INDEX(threadIdx.x, sizeof(xmachine_message_pedestrian_location));
		return ((xmachine_message_pedestrian_location*)&message_share[message_index]);
	}
	else
		return false;
	
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created navmap_cell message functions */


/* Message functions */

template <int AGENT_TYPE>
__device__ void add_navmap_cell_message(xmachine_message_navmap_cell_list* messages, int x, int y, int exit_no, float height, float collision_x, float collision_y, float exit0_x, float exit0_y, float exit1_x, float exit1_y, float exit2_x, float exit2_y, float exit3_x, float exit3_y, float exit4_x, float exit4_y, float exit5_x, float exit5_y, float exit6_x, float exit6_y){
	if (AGENT_TYPE == DISCRETE_2D){
		int width = (blockDim.x * gridDim.x);
		int2 global_position;
		global_position.x = (blockIdx.x * blockDim.x) + threadIdx.x;
		global_position.y = (blockIdx.y * blockDim.y) + threadIdx.y;

		int index = global_position.x + (global_position.y * width);

		
		messages->x[index] = x;			
		messages->y[index] = y;			
		messages->exit_no[index] = exit_no;			
		messages->height[index] = height;			
		messages->collision_x[index] = collision_x;			
		messages->collision_y[index] = collision_y;			
		messages->exit0_x[index] = exit0_x;			
		messages->exit0_y[index] = exit0_y;			
		messages->exit1_x[index] = exit1_x;			
		messages->exit1_y[index] = exit1_y;			
		messages->exit2_x[index] = exit2_x;			
		messages->exit2_y[index] = exit2_y;			
		messages->exit3_x[index] = exit3_x;			
		messages->exit3_y[index] = exit3_y;			
		messages->exit4_x[index] = exit4_x;			
		messages->exit4_y[index] = exit4_y;			
		messages->exit5_x[index] = exit5_x;			
		messages->exit5_y[index] = exit5_y;			
		messages->exit6_x[index] = exit6_x;			
		messages->exit6_y[index] = exit6_y;			
	}
	//else CONTINUOUS agents can not write to discrete space
}

//Used by continuous agents this accesses messages with texture cache. agent_x and agent_y are discrete positions in the message space
__device__ xmachine_message_navmap_cell* get_first_navmap_cell_message_continuous(xmachine_message_navmap_cell_list* messages,  int agent_x, int agent_y){

	//shared memory get from offset dependant on sm usage in function
	extern __shared__ int sm_data [];

	xmachine_message_navmap_cell* message_share = (xmachine_message_navmap_cell*)&sm_data[0];
	
	int range = d_message_navmap_cell_range;
	int width = d_message_navmap_cell_width;
	
	int2 global_position;
	global_position.x = sWRAP(agent_x-range , width);
	global_position.y = sWRAP(agent_y-range , width);
	

	int index = ((global_position.y)* width) + global_position.x;
	
	xmachine_message_navmap_cell temp_message;
	temp_message._position = make_int2(agent_x, agent_y);
	temp_message._relative = make_int2(-range, -range);

	temp_message.x = tex1Dfetch(tex_xmachine_message_navmap_cell_x, index + d_tex_xmachine_message_navmap_cell_x_offset);temp_message.y = tex1Dfetch(tex_xmachine_message_navmap_cell_y, index + d_tex_xmachine_message_navmap_cell_y_offset);temp_message.exit_no = tex1Dfetch(tex_xmachine_message_navmap_cell_exit_no, index + d_tex_xmachine_message_navmap_cell_exit_no_offset);temp_message.height = tex1Dfetch(tex_xmachine_message_navmap_cell_height, index + d_tex_xmachine_message_navmap_cell_height_offset);temp_message.collision_x = tex1Dfetch(tex_xmachine_message_navmap_cell_collision_x, index + d_tex_xmachine_message_navmap_cell_collision_x_offset);temp_message.collision_y = tex1Dfetch(tex_xmachine_message_navmap_cell_collision_y, index + d_tex_xmachine_message_navmap_cell_collision_y_offset);temp_message.exit0_x = tex1Dfetch(tex_xmachine_message_navmap_cell_exit0_x, index + d_tex_xmachine_message_navmap_cell_exit0_x_offset);temp_message.exit0_y = tex1Dfetch(tex_xmachine_message_navmap_cell_exit0_y, index + d_tex_xmachine_message_navmap_cell_exit0_y_offset);temp_message.exit1_x = tex1Dfetch(tex_xmachine_message_navmap_cell_exit1_x, index + d_tex_xmachine_message_navmap_cell_exit1_x_offset);temp_message.exit1_y = tex1Dfetch(tex_xmachine_message_navmap_cell_exit1_y, index + d_tex_xmachine_message_navmap_cell_exit1_y_offset);temp_message.exit2_x = tex1Dfetch(tex_xmachine_message_navmap_cell_exit2_x, index + d_tex_xmachine_message_navmap_cell_exit2_x_offset);temp_message.exit2_y = tex1Dfetch(tex_xmachine_message_navmap_cell_exit2_y, index + d_tex_xmachine_message_navmap_cell_exit2_y_offset);temp_message.exit3_x = tex1Dfetch(tex_xmachine_message_navmap_cell_exit3_x, index + d_tex_xmachine_message_navmap_cell_exit3_x_offset);temp_message.exit3_y = tex1Dfetch(tex_xmachine_message_navmap_cell_exit3_y, index + d_tex_xmachine_message_navmap_cell_exit3_y_offset);temp_message.exit4_x = tex1Dfetch(tex_xmachine_message_navmap_cell_exit4_x, index + d_tex_xmachine_message_navmap_cell_exit4_x_offset);temp_message.exit4_y = tex1Dfetch(tex_xmachine_message_navmap_cell_exit4_y, index + d_tex_xmachine_message_navmap_cell_exit4_y_offset);temp_message.exit5_x = tex1Dfetch(tex_xmachine_message_navmap_cell_exit5_x, index + d_tex_xmachine_message_navmap_cell_exit5_x_offset);temp_message.exit5_y = tex1Dfetch(tex_xmachine_message_navmap_cell_exit5_y, index + d_tex_xmachine_message_navmap_cell_exit5_y_offset);temp_message.exit6_x = tex1Dfetch(tex_xmachine_message_navmap_cell_exit6_x, index + d_tex_xmachine_message_navmap_cell_exit6_x_offset);temp_message.exit6_y = tex1Dfetch(tex_xmachine_message_navmap_cell_exit6_y, index + d_tex_xmachine_message_navmap_cell_exit6_y_offset);
	
	message_share[threadIdx.x] = temp_message;

	//return top left of messages
	return &message_share[threadIdx.x];
}

//Get next navmap_cell message  continuous
//Used by continuous agents this accesses messages with texture cache (agent position in discrete space was set when accessing first message)
__device__ xmachine_message_navmap_cell* get_next_navmap_cell_message_continuous(xmachine_message_navmap_cell* message, xmachine_message_navmap_cell_list* messages){

	//shared memory get from offset dependant on sm usage in function
	extern __shared__ int sm_data [];

	xmachine_message_navmap_cell* message_share = (xmachine_message_navmap_cell*)&sm_data[0];
	
	int range = d_message_navmap_cell_range;
	int width = d_message_navmap_cell_width;

	//Get previous position
	int2 previous_relative = message->_relative;

	//exit if at (range, range)
	if (previous_relative.x == (range))
        if (previous_relative.y == (range))
		    return false;

	//calculate next message relative position
	int2 next_relative = previous_relative;
	next_relative.x += 1;
	if ((next_relative.x)>range){
		next_relative.x = -range;
		next_relative.y = previous_relative.y + 1;
	}

	//skip own message
	if (next_relative.x == 0)
        if (next_relative.y == 0)
		    next_relative.x += 1;

	int2 global_position;
	global_position.x =	sWRAP(message->_position.x + next_relative.x, width);
	global_position.y = sWRAP(message->_position.y + next_relative.y, width);

	int index = ((global_position.y)* width) + (global_position.x);
	
	xmachine_message_navmap_cell temp_message;
	temp_message._position = message->_position;
	temp_message._relative = next_relative;

	temp_message.x = tex1Dfetch(tex_xmachine_message_navmap_cell_x, index + d_tex_xmachine_message_navmap_cell_x_offset);	temp_message.y = tex1Dfetch(tex_xmachine_message_navmap_cell_y, index + d_tex_xmachine_message_navmap_cell_y_offset);	temp_message.exit_no = tex1Dfetch(tex_xmachine_message_navmap_cell_exit_no, index + d_tex_xmachine_message_navmap_cell_exit_no_offset);	temp_message.height = tex1Dfetch(tex_xmachine_message_navmap_cell_height, index + d_tex_xmachine_message_navmap_cell_height_offset);	temp_message.collision_x = tex1Dfetch(tex_xmachine_message_navmap_cell_collision_x, index + d_tex_xmachine_message_navmap_cell_collision_x_offset);	temp_message.collision_y = tex1Dfetch(tex_xmachine_message_navmap_cell_collision_y, index + d_tex_xmachine_message_navmap_cell_collision_y_offset);	temp_message.exit0_x = tex1Dfetch(tex_xmachine_message_navmap_cell_exit0_x, index + d_tex_xmachine_message_navmap_cell_exit0_x_offset);	temp_message.exit0_y = tex1Dfetch(tex_xmachine_message_navmap_cell_exit0_y, index + d_tex_xmachine_message_navmap_cell_exit0_y_offset);	temp_message.exit1_x = tex1Dfetch(tex_xmachine_message_navmap_cell_exit1_x, index + d_tex_xmachine_message_navmap_cell_exit1_x_offset);	temp_message.exit1_y = tex1Dfetch(tex_xmachine_message_navmap_cell_exit1_y, index + d_tex_xmachine_message_navmap_cell_exit1_y_offset);	temp_message.exit2_x = tex1Dfetch(tex_xmachine_message_navmap_cell_exit2_x, index + d_tex_xmachine_message_navmap_cell_exit2_x_offset);	temp_message.exit2_y = tex1Dfetch(tex_xmachine_message_navmap_cell_exit2_y, index + d_tex_xmachine_message_navmap_cell_exit2_y_offset);	temp_message.exit3_x = tex1Dfetch(tex_xmachine_message_navmap_cell_exit3_x, index + d_tex_xmachine_message_navmap_cell_exit3_x_offset);	temp_message.exit3_y = tex1Dfetch(tex_xmachine_message_navmap_cell_exit3_y, index + d_tex_xmachine_message_navmap_cell_exit3_y_offset);	temp_message.exit4_x = tex1Dfetch(tex_xmachine_message_navmap_cell_exit4_x, index + d_tex_xmachine_message_navmap_cell_exit4_x_offset);	temp_message.exit4_y = tex1Dfetch(tex_xmachine_message_navmap_cell_exit4_y, index + d_tex_xmachine_message_navmap_cell_exit4_y_offset);	temp_message.exit5_x = tex1Dfetch(tex_xmachine_message_navmap_cell_exit5_x, index + d_tex_xmachine_message_navmap_cell_exit5_x_offset);	temp_message.exit5_y = tex1Dfetch(tex_xmachine_message_navmap_cell_exit5_y, index + d_tex_xmachine_message_navmap_cell_exit5_y_offset);	temp_message.exit6_x = tex1Dfetch(tex_xmachine_message_navmap_cell_exit6_x, index + d_tex_xmachine_message_navmap_cell_exit6_x_offset);	temp_message.exit6_y = tex1Dfetch(tex_xmachine_message_navmap_cell_exit6_y, index + d_tex_xmachine_message_navmap_cell_exit6_y_offset);	

	message_share[threadIdx.x] = temp_message;

	return &message_share[threadIdx.x];
}

//method used by discrete agents accessing discrete messages to load messages into shared memory
__device__ void navmap_cell_message_to_sm(xmachine_message_navmap_cell_list* messages, char* message_share, int sm_index, int global_index){
		xmachine_message_navmap_cell temp_message;
		
		temp_message.x = messages->x[global_index];		
		temp_message.y = messages->y[global_index];		
		temp_message.exit_no = messages->exit_no[global_index];		
		temp_message.height = messages->height[global_index];		
		temp_message.collision_x = messages->collision_x[global_index];		
		temp_message.collision_y = messages->collision_y[global_index];		
		temp_message.exit0_x = messages->exit0_x[global_index];		
		temp_message.exit0_y = messages->exit0_y[global_index];		
		temp_message.exit1_x = messages->exit1_x[global_index];		
		temp_message.exit1_y = messages->exit1_y[global_index];		
		temp_message.exit2_x = messages->exit2_x[global_index];		
		temp_message.exit2_y = messages->exit2_y[global_index];		
		temp_message.exit3_x = messages->exit3_x[global_index];		
		temp_message.exit3_y = messages->exit3_y[global_index];		
		temp_message.exit4_x = messages->exit4_x[global_index];		
		temp_message.exit4_y = messages->exit4_y[global_index];		
		temp_message.exit5_x = messages->exit5_x[global_index];		
		temp_message.exit5_y = messages->exit5_y[global_index];		
		temp_message.exit6_x = messages->exit6_x[global_index];		
		temp_message.exit6_y = messages->exit6_y[global_index];		

	  int message_index = SHARE_INDEX(sm_index, sizeof(xmachine_message_navmap_cell));
	  xmachine_message_navmap_cell* sm_message = ((xmachine_message_navmap_cell*)&message_share[message_index]);
	  sm_message[0] = temp_message;
}

//Get first navmap_cell message 
//Used by discrete agents this accesses messages with texture cache. Agent position is determined by position in the grid/block
//Possibility of upto 8 thread divergances
__device__ xmachine_message_navmap_cell* get_first_navmap_cell_message_discrete(xmachine_message_navmap_cell_list* messages){

	//shared memory get from offset dependant on sm usage in function
	extern __shared__ int sm_data [];

	char* message_share = (char*)&sm_data[0];
  
	__syncthreads();

	int range = d_message_navmap_cell_range;
	int width = d_message_navmap_cell_width;
	int sm_grid_width = blockDim.x + (range* 2);
	
	
	int2 global_position;
	global_position.x = (blockIdx.x * blockDim.x) + threadIdx.x;
	global_position.y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = global_position.x + (global_position.y * width);
	

	//calculate the position in shared memeory of first load
	int2 sm_pos;
	sm_pos.x = threadIdx.x + range;
	sm_pos.y = threadIdx.y + range;
	int sm_index = (sm_pos.y * sm_grid_width) + sm_pos.x;

	//each thread loads to shared memeory (coalesced read)
	navmap_cell_message_to_sm(messages, message_share, sm_index, index);

	//check for edge conditions
	int left_border = (threadIdx.x < range);
	int right_border = (threadIdx.x >= (blockDim.x-range));
	int top_border = (threadIdx.y < range);
	int bottom_border = (threadIdx.y >= (blockDim.y-range));

	
	int  border_index;
	int  sm_border_index;

	//left
	if (left_border){	
		int2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x - range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = (sm_pos.y * sm_grid_width) + threadIdx.x;
		
		navmap_cell_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//right
	if (right_border){
		int2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x + range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = (sm_pos.y * sm_grid_width) + (sm_pos.x + range);

		navmap_cell_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//top
	if (top_border){
		int2 border_index_2d = global_position;
		border_index_2d.y = sWRAP(border_index_2d.y - range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = (threadIdx.y * sm_grid_width) + sm_pos.x;

		navmap_cell_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//bottom
	if (bottom_border){
		int2 border_index_2d = global_position;
		border_index_2d.y = sWRAP(border_index_2d.y + range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = ((sm_pos.y + range) * sm_grid_width) + sm_pos.x;

		navmap_cell_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//top left
	if ((top_border)&&(left_border)){	
		int2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x - range, width);
		border_index_2d.y = sWRAP(border_index_2d.y - range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = (threadIdx.y * sm_grid_width) + threadIdx.x;
		
		navmap_cell_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//top right
	if ((top_border)&&(right_border)){	
		int2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x + range, width);
		border_index_2d.y = sWRAP(border_index_2d.y - range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = (threadIdx.y * sm_grid_width) + (sm_pos.x + range);
		
		navmap_cell_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//bottom right
	if ((bottom_border)&&(right_border)){	
		int2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x + range, width);
		border_index_2d.y = sWRAP(border_index_2d.y + range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = ((sm_pos.y + range) * sm_grid_width) + (sm_pos.x + range);
		
		navmap_cell_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//bottom left
	if ((bottom_border)&&(left_border)){	
		int2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x - range, width);
		border_index_2d.y = sWRAP(border_index_2d.y + range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = ((sm_pos.y + range) * sm_grid_width) + threadIdx.x;
		
		navmap_cell_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	__syncthreads();
	
  
	//top left of block position sm index
	sm_index = (threadIdx.y * sm_grid_width) + threadIdx.x;
	
	int message_index = SHARE_INDEX(sm_index, sizeof(xmachine_message_navmap_cell));
	xmachine_message_navmap_cell* temp = ((xmachine_message_navmap_cell*)&message_share[message_index]);
	temp->_relative = make_int2(-range, -range); //this is the relative position
	return temp;
}

//Get next navmap_cell message 
//Used by discrete agents this accesses messages through shared memeory which were all loaded on first message retrieval call.
__device__ xmachine_message_navmap_cell* get_next_navmap_cell_message_discrete(xmachine_message_navmap_cell* message, xmachine_message_navmap_cell_list* messages){

	//shared memory get from offset dependant on sm usage in function
	extern __shared__ int sm_data [];

	char* message_share = (char*)&sm_data[0];
  
	__syncthreads();
	
	int range = d_message_navmap_cell_range;
	int sm_grid_width = blockDim.x+(range*2);


	//Get previous position
	int2 previous_relative = message->_relative;

	//exit if at (range, range)
	if (previous_relative.x == range)
        if (previous_relative.y == range)
		    return false;

	//calculate next message relative position
	int2 next_relative = previous_relative;
	next_relative.x += 1;
	if ((next_relative.x)>range){
		next_relative.x = -range;
		next_relative.y = previous_relative.y + 1;
	}

	//skip own message
	if (next_relative.x == 0)
        if (next_relative.y == 0)
		    next_relative.x += 1;


	//calculate the next message position
	int2 next_position;// = block_position+next_relative;
	//offset next position by the sm border size
	next_position.x = threadIdx.x + next_relative.x + range;
	next_position.y = threadIdx.y + next_relative.y + range;

	int sm_index = next_position.x + (next_position.y * sm_grid_width);
	
	__syncthreads();
  
	int message_index = SHARE_INDEX(sm_index, sizeof(xmachine_message_navmap_cell));
	xmachine_message_navmap_cell* temp = ((xmachine_message_navmap_cell*)&message_share[message_index]);
	temp->_relative = next_relative; //this is the relative position
	return temp;
}

//Get first navmap_cell message
template <int AGENT_TYPE>
__device__ xmachine_message_navmap_cell* get_first_navmap_cell_message(xmachine_message_navmap_cell_list* messages, int agent_x, int agent_y){

	if (AGENT_TYPE == DISCRETE_2D)	//use shared memory method
		return get_first_navmap_cell_message_discrete(messages);
	else	//use texture fetching method
		return get_first_navmap_cell_message_continuous(messages, agent_x, agent_y);

}

//Get next navmap_cell message
template <int AGENT_TYPE>
__device__ xmachine_message_navmap_cell* get_next_navmap_cell_message(xmachine_message_navmap_cell* message, xmachine_message_navmap_cell_list* messages){

	if (AGENT_TYPE == DISCRETE_2D)	//use shared memory method
		return get_next_navmap_cell_message_discrete(message, messages);
	else	//use texture fetching method
		return get_next_navmap_cell_message_continuous(message, messages);

}


	
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created GPU kernals  */



/**
 *
 */
__global__ void GPUFLAME_output_pedestrian_location(xmachine_memory_agent_list* agents, xmachine_message_pedestrian_location_list* pedestrian_location_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_agent_count)
        return;
    

	//SoA to AoS - xmachine_memory_output_pedestrian_location Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent agent;
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.velx = agents->velx[index];
	agent.vely = agents->vely[index];
	agent.steer_x = agents->steer_x[index];
	agent.steer_y = agents->steer_y[index];
	agent.height = agents->height[index];
	agent.exit_no = agents->exit_no[index];
	agent.speed = agents->speed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];

	//FLAME function call
	int dead = !output_pedestrian_location(&agent, pedestrian_location_messages	);
	
	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_output_pedestrian_location Coalesced memory write (ignore arrays)
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->velx[index] = agent.velx;
	agents->vely[index] = agent.vely;
	agents->steer_x[index] = agent.steer_x;
	agents->steer_y[index] = agent.steer_y;
	agents->height[index] = agent.height;
	agents->exit_no[index] = agent.exit_no;
	agents->speed[index] = agent.speed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
}

/**
 *
 */
__global__ void GPUFLAME_avoid_pedestrians(xmachine_memory_agent_list* agents, xmachine_message_pedestrian_location_list* pedestrian_location_messages, xmachine_message_pedestrian_location_PBM* partition_matrix, RNG_rand48* rand48){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_agent_count)
        return;
    

	//SoA to AoS - xmachine_memory_avoid_pedestrians Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent agent;
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.velx = agents->velx[index];
	agent.vely = agents->vely[index];
	agent.steer_x = agents->steer_x[index];
	agent.steer_y = agents->steer_y[index];
	agent.height = agents->height[index];
	agent.exit_no = agents->exit_no[index];
	agent.speed = agents->speed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];

	//FLAME function call
	int dead = !avoid_pedestrians(&agent, pedestrian_location_messages, partition_matrix, rand48);
	
	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_avoid_pedestrians Coalesced memory write (ignore arrays)
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->velx[index] = agent.velx;
	agents->vely[index] = agent.vely;
	agents->steer_x[index] = agent.steer_x;
	agents->steer_y[index] = agent.steer_y;
	agents->height[index] = agent.height;
	agents->exit_no[index] = agent.exit_no;
	agents->speed[index] = agent.speed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
}

/**
 *
 */
__global__ void GPUFLAME_force_flow(xmachine_memory_agent_list* agents, xmachine_message_navmap_cell_list* navmap_cell_messages, RNG_rand48* rand48){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_agent_count)
        return;
    

	//SoA to AoS - xmachine_memory_force_flow Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent agent;
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.velx = agents->velx[index];
	agent.vely = agents->vely[index];
	agent.steer_x = agents->steer_x[index];
	agent.steer_y = agents->steer_y[index];
	agent.height = agents->height[index];
	agent.exit_no = agents->exit_no[index];
	agent.speed = agents->speed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];

	//FLAME function call
	int dead = !force_flow(&agent, navmap_cell_messages, rand48);
	
	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_force_flow Coalesced memory write (ignore arrays)
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->velx[index] = agent.velx;
	agents->vely[index] = agent.vely;
	agents->steer_x[index] = agent.steer_x;
	agents->steer_y[index] = agent.steer_y;
	agents->height[index] = agent.height;
	agents->exit_no[index] = agent.exit_no;
	agents->speed[index] = agent.speed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
}

/**
 *
 */
__global__ void GPUFLAME_move(xmachine_memory_agent_list* agents){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_agent_count)
        return;
    

	//SoA to AoS - xmachine_memory_move Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent agent;
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.velx = agents->velx[index];
	agent.vely = agents->vely[index];
	agent.steer_x = agents->steer_x[index];
	agent.steer_y = agents->steer_y[index];
	agent.height = agents->height[index];
	agent.exit_no = agents->exit_no[index];
	agent.speed = agents->speed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];

	//FLAME function call
	int dead = !move(&agent);
	
	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_move Coalesced memory write (ignore arrays)
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->velx[index] = agent.velx;
	agents->vely[index] = agent.vely;
	agents->steer_x[index] = agent.steer_x;
	agents->steer_y[index] = agent.steer_y;
	agents->height[index] = agent.height;
	agents->exit_no[index] = agent.exit_no;
	agents->speed[index] = agent.speed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
}

/**
 *
 */
__global__ void GPUFLAME_output_navmap_cells(xmachine_memory_navmap_list* agents, xmachine_message_navmap_cell_list* navmap_cell_messages){
	
	
	//discrete agent: index is position in 2D agent grid
	int width = (blockDim.x * gridDim.x);
	int2 global_position;
	global_position.x = (blockIdx.x * blockDim.x) + threadIdx.x;
	global_position.y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = global_position.x + (global_position.y * width);
	

	//SoA to AoS - xmachine_memory_output_navmap_cells Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_navmap agent;
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.exit_no = agents->exit_no[index];
	agent.height = agents->height[index];
	agent.collision_x = agents->collision_x[index];
	agent.collision_y = agents->collision_y[index];
	agent.exit0_x = agents->exit0_x[index];
	agent.exit0_y = agents->exit0_y[index];
	agent.exit1_x = agents->exit1_x[index];
	agent.exit1_y = agents->exit1_y[index];
	agent.exit2_x = agents->exit2_x[index];
	agent.exit2_y = agents->exit2_y[index];
	agent.exit3_x = agents->exit3_x[index];
	agent.exit3_y = agents->exit3_y[index];
	agent.exit4_x = agents->exit4_x[index];
	agent.exit4_y = agents->exit4_y[index];
	agent.exit5_x = agents->exit5_x[index];
	agent.exit5_y = agents->exit5_y[index];
	agent.exit6_x = agents->exit6_x[index];
	agent.exit6_y = agents->exit6_y[index];

	//FLAME function call
	output_navmap_cells(&agent, navmap_cell_messages	);
	
	

	//AoS to SoA - xmachine_memory_output_navmap_cells Coalesced memory write (ignore arrays)
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->exit_no[index] = agent.exit_no;
	agents->height[index] = agent.height;
	agents->collision_x[index] = agent.collision_x;
	agents->collision_y[index] = agent.collision_y;
	agents->exit0_x[index] = agent.exit0_x;
	agents->exit0_y[index] = agent.exit0_y;
	agents->exit1_x[index] = agent.exit1_x;
	agents->exit1_y[index] = agent.exit1_y;
	agents->exit2_x[index] = agent.exit2_x;
	agents->exit2_y[index] = agent.exit2_y;
	agents->exit3_x[index] = agent.exit3_x;
	agents->exit3_y[index] = agent.exit3_y;
	agents->exit4_x[index] = agent.exit4_x;
	agents->exit4_y[index] = agent.exit4_y;
	agents->exit5_x[index] = agent.exit5_x;
	agents->exit5_y[index] = agent.exit5_y;
	agents->exit6_x[index] = agent.exit6_x;
	agents->exit6_y[index] = agent.exit6_y;
}

/**
 *
 */
__global__ void GPUFLAME_generate_pedestrians(xmachine_memory_navmap_list* agents, xmachine_memory_agent_list* agent_agents, RNG_rand48* rand48){
	
	
	//discrete agent: index is position in 2D agent grid
	int width = (blockDim.x * gridDim.x);
	int2 global_position;
	global_position.x = (blockIdx.x * blockDim.x) + threadIdx.x;
	global_position.y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = global_position.x + (global_position.y * width);
	

	//SoA to AoS - xmachine_memory_generate_pedestrians Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_navmap agent;
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.exit_no = agents->exit_no[index];
	agent.height = agents->height[index];
	agent.collision_x = agents->collision_x[index];
	agent.collision_y = agents->collision_y[index];
	agent.exit0_x = agents->exit0_x[index];
	agent.exit0_y = agents->exit0_y[index];
	agent.exit1_x = agents->exit1_x[index];
	agent.exit1_y = agents->exit1_y[index];
	agent.exit2_x = agents->exit2_x[index];
	agent.exit2_y = agents->exit2_y[index];
	agent.exit3_x = agents->exit3_x[index];
	agent.exit3_y = agents->exit3_y[index];
	agent.exit4_x = agents->exit4_x[index];
	agent.exit4_y = agents->exit4_y[index];
	agent.exit5_x = agents->exit5_x[index];
	agent.exit5_y = agents->exit5_y[index];
	agent.exit6_x = agents->exit6_x[index];
	agent.exit6_y = agents->exit6_y[index];

	//FLAME function call
	generate_pedestrians(&agent, agent_agents, rand48);
	
	

	//AoS to SoA - xmachine_memory_generate_pedestrians Coalesced memory write (ignore arrays)
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->exit_no[index] = agent.exit_no;
	agents->height[index] = agent.height;
	agents->collision_x[index] = agent.collision_x;
	agents->collision_y[index] = agent.collision_y;
	agents->exit0_x[index] = agent.exit0_x;
	agents->exit0_y[index] = agent.exit0_y;
	agents->exit1_x[index] = agent.exit1_x;
	agents->exit1_y[index] = agent.exit1_y;
	agents->exit2_x[index] = agent.exit2_x;
	agents->exit2_y[index] = agent.exit2_y;
	agents->exit3_x[index] = agent.exit3_x;
	agents->exit3_y[index] = agent.exit3_y;
	agents->exit4_x[index] = agent.exit4_x;
	agents->exit4_y[index] = agent.exit4_y;
	agents->exit5_x[index] = agent.exit5_x;
	agents->exit5_y[index] = agent.exit5_y;
	agents->exit6_x[index] = agent.exit6_x;
	agents->exit6_y[index] = agent.exit6_y;
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
