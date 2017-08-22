/*
 * Copyright 2017 University of Sheffield.
 * Author: Peter Heywood 
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

#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

#include "header.h"
#include <vector>

// Declare global scope variables for host-based agent creation, so allocation of host data is only performed once.
xmachine_memory_Agent ** h_agent_AoS;
const unsigned int h_agent_AoS_MAX = 32;


/*
 * An Init function to initialise environment variables, seed a simple host RNG and pre-allocate an array of structs for host-based agent creation.
 * AoS allocation is performed here rather than in the relevant step function to avoid the performance penalty of repeated allocations.
 */
__FLAME_GPU_INIT_FUNC__ void initialiseHost() {
	// Initialise host and device constant(s)
	unsigned int max_lifespan = 16;
	set_MAX_LIFESPAN(&max_lifespan);
	printf("Set MAX_LIFESPAN = %u\n", max_lifespan);

	// Seed the host random number generator.
	srand(0);

	/* 
	  Allocate and initialise an array of agent structs to a global pointer.
	  Any number can be specified, however it should be less than the maximum number defined in XMLModelFile.xml
	  To avoid a per-iteration performance penalty the allocation is performed in an INIT function, and deallocation in an EXIT function
	*/
	h_agent_AoS = h_allocate_agent_Agent_array(h_agent_AoS_MAX);

}

/*
 * An example of how to generate a single new agent from the host.
 * In this case an INIT function is used.
 * Memory is allocated and deallocated locally to this method, as it is the only use.
 * When the h_add_agent_Agent_default() function is called, the data from the struct is immediately copied from the host to device.
 * If you wish to create more than one agent of the same type and state is will be much more efficient to use the add_agents_ function.
 */
__FLAME_GPU_INIT_FUNC__ void generateAgentInit(){
	printf("Population from initial states XML file: %u\n", get_agent_Agent_default_count());
	
	// Allocate a single agent struct on the host.
	xmachine_memory_Agent * h_agent = h_allocate_agent_Agent();

	// Set values as required for the single agent.
	h_agent->time_alive = rand() % (*get_MAX_LIFESPAN());
	for (unsigned int i = 0; i < xmachine_memory_Agent_example_array_LENGTH; i++) {
		h_agent->example_array[i] = rand() + i;
	}

	// Copy agent data from the host to the device
	h_add_agent_Agent_default(h_agent);

	// Clear host memory for single struct. The Utility function also deallocates any agent variable arrays.
	h_free_agent_Agent(&h_agent);
	
	printf("Population after init function: %u\n", get_agent_Agent_default_count());
}

/*
 * STEP function to demonstrate the addition of multiple agents from the host.
 * If it is possible to add new agents to the target agent state on the device, upto 32 agents will be created, with randomised initial data
 * The AoS is allocated (and deallocate) outside of this method, to avoid performance penalty.
 * The AoS is converted to a SoA and copied to the device when h_add_agents_Agent_default() is called.
 */
__FLAME_GPU_STEP_FUNC__ void generateAgentStep(){

	// Can create upto h_agent_AoS_MAX agents in a single pass (the number allocated for) but the full amount does not have to be created.
	unsigned int count = 32;

	// It is sensible to check if it is possible to create new agents, and if so how many.
	unsigned int agent_remaining = xmachine_memory_Agent_MAX - get_agent_Agent_default_count();
	if (agent_remaining > 0) {
		if (count > agent_remaining) {
			count = agent_remaining;
		}
		// Populate data as required
		for (unsigned int i = 0; i < count; i++) {
			h_agent_AoS[i]->time_alive = rand() % (*get_MAX_LIFESPAN());
			for (unsigned int j = 0; j < xmachine_memory_Agent_example_array_LENGTH; j++) {
				h_agent_AoS[i]->example_array[j] = rand() + j;
			}
		}
		// Copy the data to the device
		h_add_agents_Agent_default(h_agent_AoS, count);
	}

	printf("Population after step function %u\n", get_agent_Agent_default_count());
}

/*
 * EXIT function which frees global scope, allocated memory for the AoS.
 * The utility function is used to automatically handle agent array variables.
 */
__FLAME_GPU_EXIT_FUNC__ void exitFunction(){
	// Clear host memory for the agent array of structs.
	h_free_agent_Agent_array(&h_agent_AoS, h_agent_AoS_MAX);

	printf("Population for exit function: %u\n", get_agent_Agent_default_count());
}

/**
 * Simple agent function, incrementing the time of life of the agent. 
 * If the agent has exceeded the maximum life span, it dies.
 */
__FLAME_GPU_FUNC__ int update(xmachine_memory_Agent* agent)
{
	// Increment time alive
	agent->time_alive++;
	/*printf(
		"%u {%u [%f %f %f %f]}\n", 
		threadIdx.x + blockDim.x * blockIdx.x, 
		agent->time_alive, 
		get_Agent_agent_array_value<float>(agent->example_array, 0), 
		get_Agent_agent_array_value<float>(agent->example_array, 1), 
		get_Agent_agent_array_value<float>(agent->example_array, 2), 
		get_Agent_agent_array_value<float>(agent->example_array, 3)
	);*/
	// If agent has been alive long enough, kill them.
	if (agent->time_alive > MAX_LIFESPAN){
		return 1;
	}	
	return 0;
}


#endif // #ifndef _FUNCTIONS_H_
