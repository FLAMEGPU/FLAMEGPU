

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

__constant__ int d_xmachine_memory_Man_count;

__constant__ int d_xmachine_memory_Woman_count;

/* Agent state count constants */

__constant__ int d_xmachine_memory_Man_unengaged_count;

__constant__ int d_xmachine_memory_Man_engaged_count;

__constant__ int d_xmachine_memory_Woman_default_count;


/* Message constants */

/* proposal Message variables */
/* Non partitioned and spatial partitioned message variables  */
__constant__ int d_message_proposal_count;         /**< message list counter*/
__constant__ int d_message_proposal_output_type;   /**< message output type (single or optional)*/

/* notification Message variables */
/* Non partitioned and spatial partitioned message variables  */
__constant__ int d_message_notification_count;         /**< message list counter*/
__constant__ int d_message_notification_output_type;   /**< message output type (single or optional)*/

	
    
//include each function file

#include "functions.c"
    
/* Texture bindings */


    
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


/** make_proposals_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_Man_list representing agent i the current state
 * @param nextState xmachine_memory_Man_list representing agent i the next state
 */
 __global__ void make_proposals_function_filter(xmachine_memory_Man_list* currentState, xmachine_memory_Man_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_Man_count){
	
		//apply the filter
		if (currentState->engaged_to[index]==-1)
		{	//copy agent data to newstate list
			nextState->id[index] = currentState->id[index];
			nextState->round[index] = currentState->round[index];
			nextState->engaged_to[index] = currentState->engaged_to[index];
			nextState->preferred_woman[index] = currentState->preferred_woman[index];
			//set scan input flag to 1
			nextState->_scan_input[index] = 1;
		}
		else
		{
			//set scan input flag of current state to 1 (keep agent)
			currentState->_scan_input[index] = 1;
		}
	
	}
 }

/** notify_suitors_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_Woman_list representing agent i the current state
 * @param nextState xmachine_memory_Woman_list representing agent i the next state
 */
 __global__ void notify_suitors_function_filter(xmachine_memory_Woman_list* currentState, xmachine_memory_Woman_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_Woman_count){
	
		//apply the filter
		if (currentState->current_suitor[index]!=-1)
		{	//copy agent data to newstate list
			nextState->id[index] = currentState->id[index];
			nextState->current_suitor[index] = currentState->current_suitor[index];
			nextState->current_suitor_rank[index] = currentState->current_suitor_rank[index];
			nextState->preferred_man[index] = currentState->preferred_man[index];
			//set scan input flag to 1
			nextState->_scan_input[index] = 1;
		}
		else
		{
			//set scan input flag of current state to 1 (keep agent)
			currentState->_scan_input[index] = 1;
		}
	
	}
 }

/** check_resolved_function_filter
 *	Global condition function. Flags the scan input state to true if the condition is met
 * @param currentState xmachine_memory_Man_list representing agent i the current state
 */
 __global__ void check_resolved_function_filter(xmachine_memory_Man_list* currentState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_Man_count){
	
		//apply the filter
		if (currentState->engaged_to[index]!=-1)
		{	currentState->_scan_input[index] = 1;
		}
		else
		{
			currentState->_scan_input[index] = 0;
		}
	
	}
 }

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created Man agent functions */

/** reset_Man_scan_input
 * Man agent reset scan input function
 * @param agents The xmachine_memory_Man_list agent list
 */
__global__ void reset_Man_scan_input(xmachine_memory_Man_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_Man_Agents
 * Man scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_Man_list agent list destination
 * @param agents_src xmachine_memory_Man_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_Man_Agents(xmachine_memory_Man_list* agents_dst, xmachine_memory_Man_list* agents_src, int dst_agent_count, int number_to_scatter){
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
		agents_dst->round[output_index] = agents_src->round[index];        
		agents_dst->engaged_to[output_index] = agents_src->engaged_to[index];
	    for (int i=0; i<1024; i++){
	      agents_dst->preferred_woman[(i*xmachine_memory_Man_MAX)+output_index] = agents_src->preferred_woman[(i*xmachine_memory_Man_MAX)+index];
	    }
	}
}

/** append_Man_Agents
 * Man scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_Man_list agent list destination
 * @param agents_src xmachine_memory_Man_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_Man_Agents(xmachine_memory_Man_list* agents_dst, xmachine_memory_Man_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->id[output_index] = agents_src->id[index];
	    agents_dst->round[output_index] = agents_src->round[index];
	    agents_dst->engaged_to[output_index] = agents_src->engaged_to[index];
	    for (int i=0; i<1024; i++){
	      agents_dst->preferred_woman[(i*xmachine_memory_Man_MAX)+output_index] = agents_src->preferred_woman[(i*xmachine_memory_Man_MAX)+index];
	    }
    }
}

/** add_Man_agent
 * Continuous Man agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_Man_list to add agents to 
 * @param id agent variable of type int
 * @param round agent variable of type int
 * @param engaged_to agent variable of type int
 * @param preferred_woman agent variable of type int
 */
template <int AGENT_TYPE>
__device__ void add_Man_agent(xmachine_memory_Man_list* agents, int id, int round, int engaged_to){
	
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
	agents->round[index] = round;
	agents->engaged_to[index] = engaged_to;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_Man_agent(xmachine_memory_Man_list* agents, int id, int round, int engaged_to){
    add_Man_agent<DISCRETE_2D>(agents, id, round, engaged_to);
}

/** reorder_Man_agents
 * Continuous Man agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_Man_agents(unsigned int* values, xmachine_memory_Man_list* unordered_agents, xmachine_memory_Man_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->id[index] = unordered_agents->id[old_pos];
	ordered_agents->round[index] = unordered_agents->round[old_pos];
	ordered_agents->engaged_to[index] = unordered_agents->engaged_to[old_pos];
	for (int i=0; i<1024; i++){
	  ordered_agents->preferred_woman[(i*xmachine_memory_Man_MAX)+index] = unordered_agents->preferred_woman[(i*xmachine_memory_Man_MAX)+old_pos];
	}
}

/** get_Man_agent_array_value
 *  Template function for accessing Man agent array memory variables. Assumes array points to the first element of the agents array values (offset by agent index)
 *  @param array Agent memory array
 *  @param index to lookup
 *  @return return value
 */
template<typename T>
__FLAME_GPU_FUNC__ T get_Man_agent_array_value(T *array, uint index){
    return array[index*xmachine_memory_Man_MAX];
}

/** set_Man_agent_array_value
 *  Template function for setting Man agent array memory variables. Assumes array points to the first element of the agents array values (offset by agent index)
 *  @param array Agent memory array
 *  @param index to lookup
 *  @param return value
 */
template<typename T>
__FLAME_GPU_FUNC__ void set_Man_agent_array_value(T *array, uint index, T value){
    array[index*xmachine_memory_Man_MAX] = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created Woman agent functions */

/** reset_Woman_scan_input
 * Woman agent reset scan input function
 * @param agents The xmachine_memory_Woman_list agent list
 */
__global__ void reset_Woman_scan_input(xmachine_memory_Woman_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_Woman_Agents
 * Woman scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_Woman_list agent list destination
 * @param agents_src xmachine_memory_Woman_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_Woman_Agents(xmachine_memory_Woman_list* agents_dst, xmachine_memory_Woman_list* agents_src, int dst_agent_count, int number_to_scatter){
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
		agents_dst->current_suitor[output_index] = agents_src->current_suitor[index];        
		agents_dst->current_suitor_rank[output_index] = agents_src->current_suitor_rank[index];
	    for (int i=0; i<1024; i++){
	      agents_dst->preferred_man[(i*xmachine_memory_Woman_MAX)+output_index] = agents_src->preferred_man[(i*xmachine_memory_Woman_MAX)+index];
	    }
	}
}

/** append_Woman_Agents
 * Woman scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_Woman_list agent list destination
 * @param agents_src xmachine_memory_Woman_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_Woman_Agents(xmachine_memory_Woman_list* agents_dst, xmachine_memory_Woman_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->id[output_index] = agents_src->id[index];
	    agents_dst->current_suitor[output_index] = agents_src->current_suitor[index];
	    agents_dst->current_suitor_rank[output_index] = agents_src->current_suitor_rank[index];
	    for (int i=0; i<1024; i++){
	      agents_dst->preferred_man[(i*xmachine_memory_Woman_MAX)+output_index] = agents_src->preferred_man[(i*xmachine_memory_Woman_MAX)+index];
	    }
    }
}

/** add_Woman_agent
 * Continuous Woman agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_Woman_list to add agents to 
 * @param id agent variable of type int
 * @param current_suitor agent variable of type int
 * @param current_suitor_rank agent variable of type int
 * @param preferred_man agent variable of type int
 */
template <int AGENT_TYPE>
__device__ void add_Woman_agent(xmachine_memory_Woman_list* agents, int id, int current_suitor, int current_suitor_rank){
	
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
	agents->current_suitor[index] = current_suitor;
	agents->current_suitor_rank[index] = current_suitor_rank;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_Woman_agent(xmachine_memory_Woman_list* agents, int id, int current_suitor, int current_suitor_rank){
    add_Woman_agent<DISCRETE_2D>(agents, id, current_suitor, current_suitor_rank);
}

/** reorder_Woman_agents
 * Continuous Woman agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_Woman_agents(unsigned int* values, xmachine_memory_Woman_list* unordered_agents, xmachine_memory_Woman_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->id[index] = unordered_agents->id[old_pos];
	ordered_agents->current_suitor[index] = unordered_agents->current_suitor[old_pos];
	ordered_agents->current_suitor_rank[index] = unordered_agents->current_suitor_rank[old_pos];
	for (int i=0; i<1024; i++){
	  ordered_agents->preferred_man[(i*xmachine_memory_Woman_MAX)+index] = unordered_agents->preferred_man[(i*xmachine_memory_Woman_MAX)+old_pos];
	}
}

/** get_Woman_agent_array_value
 *  Template function for accessing Woman agent array memory variables. Assumes array points to the first element of the agents array values (offset by agent index)
 *  @param array Agent memory array
 *  @param index to lookup
 *  @return return value
 */
template<typename T>
__FLAME_GPU_FUNC__ T get_Woman_agent_array_value(T *array, uint index){
    return array[index*xmachine_memory_Woman_MAX];
}

/** set_Woman_agent_array_value
 *  Template function for setting Woman agent array memory variables. Assumes array points to the first element of the agents array values (offset by agent index)
 *  @param array Agent memory array
 *  @param index to lookup
 *  @param return value
 */
template<typename T>
__FLAME_GPU_FUNC__ void set_Woman_agent_array_value(T *array, uint index, T value){
    array[index*xmachine_memory_Woman_MAX] = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created proposal message functions */


/** add_proposal_message
 * Add non partitioned or spatially partitioned proposal message
 * @param messages xmachine_message_proposal_list message list to add too
 * @param id agent variable of type int
 * @param woman agent variable of type int
 */
__device__ void add_proposal_message(xmachine_message_proposal_list* messages, int id, int woman){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_proposal_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_proposal_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_proposal_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_proposal Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->id[index] = id;
	messages->woman[index] = woman;

}

/**
 * Scatter non partitioned or spatially partitioned proposal message (for optional messages)
 * @param messages scatter_optional_proposal_messages Sparse xmachine_message_proposal_list message list
 * @param message_swap temp xmachine_message_proposal_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_proposal_messages(xmachine_message_proposal_list* messages, xmachine_message_proposal_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_proposal_count;

		//AoS - xmachine_message_proposal Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->id[output_index] = messages_swap->id[index];
		messages->woman[output_index] = messages_swap->woman[index];				
	}
}

/** reset_proposal_swaps
 * Reset non partitioned or spatially partitioned proposal message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_proposal_swaps(xmachine_message_proposal_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_proposal* get_first_proposal_message(xmachine_message_proposal_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_proposal_count/ blockDim.x)* blockDim.x);

	//if no messages then return false
	if (wrap_size == 0)
		return false;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_proposal Coalesced memory read
	xmachine_message_proposal temp_message;
	temp_message._position = messages->_position[index];
	temp_message.id = messages->id[index];
	temp_message.woman = messages->woman[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.x, sizeof(xmachine_message_proposal));
	xmachine_message_proposal* sm_message = ((xmachine_message_proposal*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_proposal*)&message_share[d_SM_START]);
}

__device__ xmachine_message_proposal* get_next_proposal_message(xmachine_message_proposal* message, xmachine_message_proposal_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_proposal_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_proposal_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return false;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we dont change shared memeory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_proposal Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_proposal temp_message;
		temp_message._position = messages->_position[index];
		temp_message.id = messages->id[index];
		temp_message.woman = messages->woman[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.x, sizeof(xmachine_message_proposal));
		xmachine_message_proposal* sm_message = ((xmachine_message_proposal*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we dont start returning messages untill all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_proposal));
	return ((xmachine_message_proposal*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created notification message functions */


/** add_notification_message
 * Add non partitioned or spatially partitioned notification message
 * @param messages xmachine_message_notification_list message list to add too
 * @param id agent variable of type int
 * @param suitor agent variable of type int
 */
__device__ void add_notification_message(xmachine_message_notification_list* messages, int id, int suitor){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_notification_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_notification_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_notification_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_notification Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->id[index] = id;
	messages->suitor[index] = suitor;

}

/**
 * Scatter non partitioned or spatially partitioned notification message (for optional messages)
 * @param messages scatter_optional_notification_messages Sparse xmachine_message_notification_list message list
 * @param message_swap temp xmachine_message_notification_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_notification_messages(xmachine_message_notification_list* messages, xmachine_message_notification_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_notification_count;

		//AoS - xmachine_message_notification Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->id[output_index] = messages_swap->id[index];
		messages->suitor[output_index] = messages_swap->suitor[index];				
	}
}

/** reset_notification_swaps
 * Reset non partitioned or spatially partitioned notification message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_notification_swaps(xmachine_message_notification_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_notification* get_first_notification_message(xmachine_message_notification_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_notification_count/ blockDim.x)* blockDim.x);

	//if no messages then return false
	if (wrap_size == 0)
		return false;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_notification Coalesced memory read
	xmachine_message_notification temp_message;
	temp_message._position = messages->_position[index];
	temp_message.id = messages->id[index];
	temp_message.suitor = messages->suitor[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.x, sizeof(xmachine_message_notification));
	xmachine_message_notification* sm_message = ((xmachine_message_notification*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_notification*)&message_share[d_SM_START]);
}

__device__ xmachine_message_notification* get_next_notification_message(xmachine_message_notification* message, xmachine_message_notification_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_notification_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_notification_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return false;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we dont change shared memeory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_notification Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_notification temp_message;
		temp_message._position = messages->_position[index];
		temp_message.id = messages->id[index];
		temp_message.suitor = messages->suitor[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.x, sizeof(xmachine_message_notification));
		xmachine_message_notification* sm_message = ((xmachine_message_notification*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we dont start returning messages untill all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_notification));
	return ((xmachine_message_notification*)&message_share[message_index]);
}


	
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created GPU kernals  */



/**
 *
 */
__global__ void GPUFLAME_make_proposals(xmachine_memory_Man_list* agents, xmachine_message_proposal_list* proposal_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_Man_count)
        return;
    

	//SoA to AoS - xmachine_memory_make_proposals Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Man agent;
	agent.id = agents->id[index];
	agent.round = agents->round[index];
	agent.engaged_to = agents->engaged_to[index];
    agent.preferred_woman = &(agents->preferred_woman[index]);

	//FLAME function call
	int dead = !make_proposals(&agent, proposal_messages	);
	
	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_make_proposals Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->round[index] = agent.round;
	agents->engaged_to[index] = agent.engaged_to;
}

/**
 *
 */
__global__ void GPUFLAME_check_notifications(xmachine_memory_Man_list* agents, xmachine_message_notification_list* notification_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_check_notifications Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Man agent;
	agent.id = agents->id[index];
	agent.round = agents->round[index];
	agent.engaged_to = agents->engaged_to[index];
    agent.preferred_woman = &(agents->preferred_woman[index]);

	//FLAME function call
	int dead = !check_notifications(&agent, notification_messages);
	
	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_check_notifications Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->round[index] = agent.round;
	agents->engaged_to[index] = agent.engaged_to;
}

/**
 *
 */
__global__ void GPUFLAME_check_resolved(xmachine_memory_Man_list* agents){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_Man_count)
        return;
    

	//SoA to AoS - xmachine_memory_check_resolved Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Man agent;
	agent.id = agents->id[index];
	agent.round = agents->round[index];
	agent.engaged_to = agents->engaged_to[index];
    agent.preferred_woman = &(agents->preferred_woman[index]);

	//FLAME function call
	int dead = !check_resolved(&agent);
	
	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_check_resolved Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->round[index] = agent.round;
	agents->engaged_to[index] = agent.engaged_to;
}

/**
 *
 */
__global__ void GPUFLAME_check_proposals(xmachine_memory_Woman_list* agents, xmachine_message_proposal_list* proposal_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_check_proposals Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Woman agent;
	agent.id = agents->id[index];
	agent.current_suitor = agents->current_suitor[index];
	agent.current_suitor_rank = agents->current_suitor_rank[index];
    agent.preferred_man = &(agents->preferred_man[index]);

	//FLAME function call
	int dead = !check_proposals(&agent, proposal_messages);
	
	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_check_proposals Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->current_suitor[index] = agent.current_suitor;
	agents->current_suitor_rank[index] = agent.current_suitor_rank;
}

/**
 *
 */
__global__ void GPUFLAME_notify_suitors(xmachine_memory_Woman_list* agents, xmachine_message_notification_list* notification_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_Woman_count)
        return;
    

	//SoA to AoS - xmachine_memory_notify_suitors Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Woman agent;
	agent.id = agents->id[index];
	agent.current_suitor = agents->current_suitor[index];
	agent.current_suitor_rank = agents->current_suitor_rank[index];
    agent.preferred_man = &(agents->preferred_man[index]);

	//FLAME function call
	int dead = !notify_suitors(&agent, notification_messages	);
	
	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_notify_suitors Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->current_suitor[index] = agent.current_suitor;
	agents->current_suitor_rank[index] = agent.current_suitor_rank;
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
