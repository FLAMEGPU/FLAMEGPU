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

//Agent state variables
#define AGENT_STATE_UNOCCUPIED 0
#define AGENT_STATE_OCCUPIED 1
#define AGENT_STATE_MOVEMENT_REQUESTED 2
#define AGENT_STATE_MOVEMENT_UNRESOLVED 3


//Growback variables
#define SUGAR_GROWBACK_RATE 1
#define SUGAR_MAX_CAPACITY 4

__FLAME_GPU_FUNC__ int get_agent_x(){
	return threadIdx.x + __mul24(blockIdx.x, blockDim.x);
}

__FLAME_GPU_FUNC__ int get_agent_y(){
	return threadIdx.y + __mul24(blockIdx.y, blockDim.y);
}


__FLAME_GPU_FUNC__ int metabolise_and_growback(xmachine_memory_agent* agent){

	//metabolise if occupied
	if (agent->state == AGENT_STATE_OCCUPIED)
	{
		//store sugar
		agent->sugar_level += agent->env_sugar_level;
		agent->env_sugar_level = 0;

		//metabolise
		agent->sugar_level -= agent->metabolism;
		
		//check if agent dies
		if (agent->sugar_level == 0)
		{
			agent->state = AGENT_STATE_UNOCCUPIED;
			agent->agent_id = -1;
			agent->sugar_level = 0;
			agent->metabolism = 0;
		}
	}
	
	//growback if unoccupied
	if (agent->state == AGENT_STATE_UNOCCUPIED)
	{
		if (agent->env_sugar_level < SUGAR_MAX_CAPACITY)
		{
			agent->env_sugar_level += SUGAR_GROWBACK_RATE;
		}
	}

	//set all active agents to unresolved as they may now want to move
	if (agent->state == AGENT_STATE_OCCUPIED)
	{
		agent->state = AGENT_STATE_MOVEMENT_UNRESOLVED;
	}

  
    return 0;
}


__FLAME_GPU_FUNC__ int output_cell_state(xmachine_memory_agent* agent, xmachine_message_cell_state_list* cell_state_messages){

    
    add_cell_state_message<DISCRETE_2D>(cell_state_messages, agent->location_id, agent->state, agent->env_sugar_level);
     
    return 0;
}


__FLAME_GPU_FUNC__ int movement_request(xmachine_memory_agent* agent, xmachine_message_cell_state_list* cell_state_messages, xmachine_message_movement_request_list* movement_request_messages){

	int best_sugar_level = -1;
    int best_location_id = -1;

	//find the best location to move to
    xmachine_message_cell_state* current_message = get_first_cell_state_message<DISCRETE_2D>(cell_state_messages, get_agent_x(), get_agent_y());
    while (current_message)
    {
        //if occupied then look for empty cells
		if (agent->state == AGENT_STATE_MOVEMENT_UNRESOLVED)
		{
			//if location is unoccupied then check for empty locations
			if (current_message->state == AGENT_STATE_UNOCCUPIED)
			{
				//if the sugar level at current location is better than currently stored then update
				if (current_message->env_sugar_level > best_sugar_level)
				{
					best_sugar_level = current_message->env_sugar_level;
					best_location_id = current_message->location_id;
				}
			}
		}
        
		current_message = get_next_cell_state_message<DISCRETE_2D>(current_message, cell_state_messages);
    }

	//if the agent has found a better location to move to then update its state
	if ((agent->state == AGENT_STATE_MOVEMENT_UNRESOLVED))
	{
		//if there is a better location to move to then state indicates a movement request
		if (best_location_id > 0)
		{
			agent->state = AGENT_STATE_MOVEMENT_REQUESTED;
		}
		else
		{
			agent->state = AGENT_STATE_OCCUPIED;
		}
	}

	//add a movement request
	add_movement_request_message<DISCRETE_2D>(movement_request_messages, agent->agent_id, best_location_id, agent->sugar_level, agent->metabolism);
     
   
    return 0;
}


__FLAME_GPU_FUNC__ int movement_response(xmachine_memory_agent* agent, xmachine_message_movement_request_list* movement_request_messages, xmachine_message_movement_response_list* movement_response_messages, RNG_rand48* rand){

	int best_request_id = -1;
	int best_request_priority = -1;
	int best_request_sugar_level = -1;
	int best_request_metabolism = -1;
    
    xmachine_message_movement_request* current_message = get_first_movement_request_message<DISCRETE_2D>(movement_request_messages, get_agent_x(), get_agent_y());
    while (current_message)
    {
		//if the location is unoccupied then check for agents requesting to move here
        if (agent->state == AGENT_STATE_UNOCCUPIED)
		{
			//check if request is to move to this location
			if (current_message->location_id == agent->location_id)
			{
				//check the priority and maintain the best ranked agent
				int message_priority = 0; //rand48(rand);
				if (message_priority > best_request_priority)
				{
					best_request_id = current_message->agent_id;
					best_request_priority = message_priority;

				}
			}
		}
        
        current_message = get_next_movement_request_message<DISCRETE_2D>(current_message, movement_request_messages);
    }
    
	//if the location is unoccupied and an agent wants to move here then do so and send a response
    if ((agent->state == AGENT_STATE_UNOCCUPIED)&&(best_request_id > 0))
	{
		agent->state = AGENT_STATE_OCCUPIED;
		//move the agent to here
		agent->agent_id = best_request_id;
		agent->sugar_level = best_request_sugar_level;
		agent->metabolism = best_request_metabolism;

	}
    
	//add a movement response
    add_movement_response_message<DISCRETE_2D>(movement_response_messages, agent->location_id, best_request_id);
    
    return 0;
}

__FLAME_GPU_FUNC__ int movement_transaction(xmachine_memory_agent* agent, xmachine_message_movement_response_list* movement_response_messages){

    xmachine_message_movement_response* current_message = get_first_movement_response_message<DISCRETE_2D>(movement_response_messages, get_agent_x(), get_agent_y());
    while (current_message)
    {
		//if location contains an agent wanting to move then look for responses allowing relocation
		if (agent->state == AGENT_STATE_MOVEMENT_REQUESTED)
		{
			//if the movement response request came from this location
			if (current_message->agent_id == agent->agent_id)
			{
				//remove the agent and reset agent specific variables as it has now moved
				agent->state = AGENT_STATE_UNOCCUPIED;
				agent->agent_id = -1;
				agent->sugar_level = 0;
				agent->metabolism = 0;
			}
		}
        
        current_message = get_next_movement_response_message<DISCRETE_2D>(current_message, movement_response_messages);
    }

	//if request has not been responded to then agent is unresolved
	if (agent->state == AGENT_STATE_MOVEMENT_REQUESTED)
	{
		agent->state = AGENT_STATE_MOVEMENT_UNRESOLVED;
	}
  
    return 0;
}



#endif // #ifndef _FUNCTIONS_H_
