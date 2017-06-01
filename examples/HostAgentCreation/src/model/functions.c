/*
 * Copyright 2011 University of Sheffield.
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

#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

#include "header.h"

__FLAME_GPU_INIT_FUNC__ void generateAgentInit(xmachine_memory_Agent_list * h_Agents_default, unsigned int * new_Agent_default_count, xmachine_memory_Agent * agent){
	// Set the max_lifespan constant
	unsigned int max_lifespan = 16;
	set_MAX_LIFESPAN(&max_lifespan);
	printf("Set MAX_LIFESPAN = %u\n", max_lifespan);

	// seed host rng.
	srand(0);

	unsigned int number_of_new_agents = 32;
	// If there is room in the buffer to create a new agent
	if (get_agent_Agent_default_count() + number_of_new_agents < xmachine_memory_Agent_MAX){
		// Assign values to each local variable, upto 8 times. 
		for (unsigned int count = 0; count < number_of_new_agents; count++){
			// Generate agent data and insert into host array 
			agent->time_alive = rand() % (*get_MAX_LIFESPAN());
			for (unsigned int i = 0; i < 4; i++){
				agent->example_array[i] = get_agent_Agent_default_count()+count;
			}
			// Pass the struct to the local host struct of arrays, which will then be copied to the device
			h_generateAgentAgent(h_Agents_default, agent, count);
			// Update the number of agents which have been created in this step function.
			(*new_Agent_default_count)++;
		}
	}

	printf("Population from initial states XML file: %u\n", get_agent_Agent_default_count());
	printf("Population after init function: %u\n", get_agent_Agent_default_count() + (*new_Agent_default_count));
	
}

__FLAME_GPU_STEP_FUNC__ void generateAgentStep(xmachine_memory_Agent_list * h_Agents_default, unsigned int * new_Agent_default_count, xmachine_memory_Agent * agent){
	unsigned int number_of_new_agents = 32;
	// If there is room in the buffer to create a new agent
	if (get_agent_Agent_default_count() + number_of_new_agents < xmachine_memory_Agent_MAX){
		// Assign values to each local variable, upto 8 times. 
		for (unsigned int count = 0; count < number_of_new_agents; count++){
			// Generate agent data and insert into host array 
			agent->time_alive = rand() % (*get_MAX_LIFESPAN());
			for (unsigned int i = 0; i < 4; i++){
				agent->example_array[i] = get_agent_Agent_default_count() + count;
			}
			// Pass the struct to the local host struct of arrays, which will then be copied to the device
			h_generateAgentAgent(h_Agents_default, agent, count);
			// Update the number of agents which have been created in this step function.
			(*new_Agent_default_count)++;
		}
	}

	printf("Population after step function: %u\n", get_agent_Agent_default_count() + (*new_Agent_default_count));
	
}

__FLAME_GPU_EXIT_FUNC__ void exitFunction(){
	printf("Population for exit function: %u\n", get_agent_Agent_default_count());
}


__FLAME_GPU_FUNC__ int update(xmachine_memory_Agent* agent)
{
	// Increment time alive
	agent->time_alive++;
	//printf("%f %f %f %f\n", get_Agent_agent_array_value<float>(agent->example_array, 0), get_Agent_agent_array_value<float>(agent->example_array, 1), get_Agent_agent_array_value<float>(agent->example_array, 2), get_Agent_agent_array_value<float>(agent->example_array, 3));
	// If agent has been alive long enough, kill them.
	if (agent->time_alive > MAX_LIFESPAN){
		return 1;
	}	
	return 0;
}


#endif // #ifndef _FUNCTIONS_H_
