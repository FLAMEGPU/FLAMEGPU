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

//Environment Variables
#define STATE_ALIVE 1
#define STATE_DEAD 0



//cell Agent Functions

//The following function arguments have been generated automatically by the FLAMEGPU XParser and are dependant on the function input and outputs. If they are changed manually be sure to match any arguments to the XMML specification.
//Input : 
//Output: state 
//Agent Output: 
__FLAME_GPU_FUNC__ int output_state(xmachine_memory_cell* xmemory, xmachine_message_state_list* state_messages) 
{
	add_state_message<DISCRETE_2D>(state_messages, xmemory->state, xmemory->x, xmemory->y);

	return 0;
}

//The following function arguments have been generated automatically by the FLAMEGPU XParser and are dependant on the function input and outputs. If they are changed manually be sure to match any arguments to the XMML specification.
//Input : state 
//Output: 
//Agent Output: 
__FLAME_GPU_FUNC__ int update_state(xmachine_memory_cell* xmemory, xmachine_message_state_list* state_messages) 
{
	int state = xmemory->state;
	int living_neighbours = 0;

	//itterate messages
	int count = 0;
	xmachine_message_state* state_message = get_first_state_message<DISCRETE_2D>(state_messages, xmemory->x, xmemory->y);
	while(state_message){
		count++;
		//Count living neighbours
		int message_state = state_message->state;
		if (message_state == STATE_ALIVE){
			living_neighbours++;
		}

		state_message = get_next_state_message<DISCRETE_2D>(state_message, state_messages);
	}
	
	
	if (state == STATE_ALIVE){
		if (living_neighbours < 2)
			xmemory->state = STATE_DEAD;
		else if(living_neighbours > 3)
			xmemory->state = STATE_DEAD;
		else //exactly 2 or 3 living_neighbours
			xmemory->state = STATE_ALIVE;
	}else{
		if (living_neighbours == 3)
			xmemory->state = STATE_ALIVE;
	}

	return 0;
}



#endif // #ifndef _FUNCTIONS_H_
