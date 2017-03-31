
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

#ifndef _FLAMEGPU_FUNCTIONS
#define _FLAMEGPU_FUNCTIONS

#include <header.h>


/**
 * output_example FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_example_agent. This represents a single agent instance and can be modified directly.
 * @param example_message_messages Pointer to output message list of type xmachine_message_example_message_list. Must be passed as an argument to the add_example_message_message function ??.
 */
__FLAME_GPU_FUNC__ int output_properties(xmachine_memory_pedestrian* agent, xmachine_message_properties_message_list* properties_messages){
	add_properties_message_message(properties_messages, agent->x, agent->y, agent->vx, agent->vy);
	
    return 0;
}

/**
 * input_example FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_example_agent. This represents a single agent instance and can be modified directly.
 * @param example_message_messages  example_message_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_example_message_message and get_next_example_message_message functions.
 */
__FLAME_GPU_FUNC__ int input_properties(xmachine_memory_pedestrian* agent, xmachine_message_properties_message_list* properties_messages){

	glm::vec2 agent_pos = glm::vec2(agent->x, agent->y);
	glm::vec2 agent_vel = glm::vec2(agent->vx, agent->vy);

	xmachine_message_properties_message* current_message = get_first_properties_message_message(properties_messages);
	while (current_message)
	{
		//INSERT MESSAGE PROCESSING CODE HERE
     
		current_message = get_next_properties_message_message(current_message, properties_messages);
	}

     
	agent->x = agent_pos.x + agent_vel.x;
	agent->y = agent_pos.y + agent_vel.y;
	agent->vx = agent_vel.x;
	agent->vy = agent_vel.y;
  
    return 0;
}

  


#endif //_FLAMEGPU_FUNCTIONS
