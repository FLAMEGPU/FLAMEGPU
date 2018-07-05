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

#include "header.h"
#include "CustomVisualisation.h"

#define SCALE_FACTOR 0.03125

#define I_SCALER (SCALE_FACTOR*0.35f)
#define MESSAGE_RADIUS d_message_pedestrian_location_radius
#define MIN_DISTANCE 0.0001f

//#define NUM_EXITS 7

#define PI 3.1415f
#define RADIANS(x) (PI / 180.0f) * x

__FLAME_GPU_FUNC__ int getNewExitLocation(RNG_rand48* rand48){

	int exit1_compare = EXIT1_PROBABILITY;
	int exit2_compare = EXIT2_PROBABILITY + exit1_compare;
	int exit3_compare = EXIT3_PROBABILITY + exit2_compare;
	int exit4_compare = EXIT4_PROBABILITY + exit3_compare;
	int exit5_compare = EXIT5_PROBABILITY + exit4_compare;
	int exit6_compare = EXIT6_PROBABILITY + exit5_compare;

	float range = (float) (EXIT1_PROBABILITY +
				  EXIT2_PROBABILITY +
				  EXIT3_PROBABILITY +
				  EXIT4_PROBABILITY +
				  EXIT5_PROBABILITY +
				  EXIT6_PROBABILITY +
				  EXIT7_PROBABILITY);

	float rand = rnd<DISCRETE_2D>(rand48)*range;

	if (rand<exit1_compare)
		return 1;
	else if (rand<exit2_compare)
		return 2;
	else if (rand<exit3_compare)
		return 3;
	else if (rand<exit4_compare)
		return 4;
	else if (rand<exit5_compare)
		return 5;
	else if (rand<exit6_compare)
		return 6;
	else
		return 7;

}

/**
 * output_location FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param location_messages Pointer to output message list of type xmachine_message_location_list. Must be passed as an argument to the add_location_message function ??.
 */
__FLAME_GPU_FUNC__ int output_pedestrian_location(xmachine_memory_agent* agent, xmachine_message_pedestrian_location_list* pedestrian_location_messages){

    
	add_pedestrian_location_message(pedestrian_location_messages, agent->x, agent->y, 0.0);
  
    return 0;
}

/**
 * output_navmap_cells FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_navmap. This represents a single agent instance and can be modified directly.
 * @param navmap_cell_messages Pointer to output message list of type xmachine_message_navmap_cell_list. Must be passed as an argument to the add_navmap_cell_message function ??.
 */
__FLAME_GPU_FUNC__ int output_navmap_cells(xmachine_memory_navmap* agent, xmachine_message_navmap_cell_list* navmap_cell_messages){
    
	add_navmap_cell_message<DISCRETE_2D>(navmap_cell_messages, 
		agent->x, agent->y, 
		agent->exit_no, 
		agent->height,
		agent->collision_x, agent->collision_y, 
		agent->exit0_x, agent->exit0_y,
		agent->exit1_x, agent->exit1_y,
		agent->exit2_x, agent->exit2_y,
		agent->exit3_x, agent->exit3_y,
		agent->exit4_x, agent->exit4_y,
		agent->exit5_x, agent->exit5_y,
		agent->exit6_x, agent->exit6_y);
       
    return 0;
}



/**
 * move FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param location_messages  location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_location_message and get_next_location_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_location_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an arument to the rand48 function for genertaing random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int avoid_pedestrians(xmachine_memory_agent* agent, xmachine_message_pedestrian_location_list* pedestrian_location_messages, xmachine_message_pedestrian_location_PBM* partition_matrix, RNG_rand48* rand48){

	glm::vec2 agent_pos = glm::vec2(agent->x, agent->y);
	glm::vec2 agent_vel = glm::vec2(agent->velx, agent->vely);

	glm::vec2 navigate_velocity = glm::vec2(0.0f, 0.0f);
	glm::vec2 avoid_velocity = glm::vec2(0.0f, 0.0f);

	xmachine_message_pedestrian_location* current_message = get_first_pedestrian_location_message(pedestrian_location_messages, partition_matrix, agent->x, agent->y, 0.0);
	while (current_message)
	{
		glm::vec2 message_pos = glm::vec2(current_message->x, current_message->y);
		float separation = length(agent_pos - message_pos);
		if ((separation < MESSAGE_RADIUS)&&(separation>MIN_DISTANCE)){
			glm::vec2 to_agent = normalize(agent_pos - message_pos);
			float ang = acosf(dot(agent_vel, to_agent));
			float perception = 45.0f;

			//STEER
			if ((ang < RADIANS(perception)) || (ang > 3.14159265f-RADIANS(perception))){
				glm::vec2 s_velocity = to_agent;
				s_velocity *= powf(I_SCALER/separation, 1.25f)*STEER_WEIGHT;
				navigate_velocity += s_velocity;
			}

			//AVOID
			glm::vec2 a_velocity = to_agent;
			a_velocity *= powf(I_SCALER/separation, 2.00f)*AVOID_WEIGHT;
			avoid_velocity += a_velocity;						

		}
		 current_message = get_next_pedestrian_location_message(current_message, pedestrian_location_messages, partition_matrix);
	}

	//maximum velocity rule
	glm::vec2 steer_velocity = navigate_velocity + avoid_velocity;

	agent->steer_x = steer_velocity.x;
	agent->steer_y = steer_velocity.y;

    return 0;
}

  
/**
 * force_flow FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param navmap_cell_messages  navmap_cell_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_navmap_cell_message and get_next_navmap_cell_message functions.
 */
__FLAME_GPU_FUNC__ int force_flow(xmachine_memory_agent* agent, xmachine_message_navmap_cell_list* navmap_cell_messages, RNG_rand48* rand48){

    //map agent position into 2d grid
	int x = floor(((agent->x+ENV_MAX)/ENV_WIDTH)*d_message_navmap_cell_width);
	int y = floor(((agent->y+ENV_MAX)/ENV_WIDTH)*d_message_navmap_cell_width);

	//lookup single message
    xmachine_message_navmap_cell* current_message = get_first_navmap_cell_message<CONTINUOUS>(navmap_cell_messages, x, y);
  
	glm::vec2 collision_force = glm::vec2(current_message->collision_x, current_message->collision_y);
	collision_force *= COLLISION_WEIGHT;

	//exit location of cell
	int exit_location = current_message->exit_no;

	//agent death flag
	int kill_agent = 0;

	//goal force
	glm::vec2 goal_force;
	if (agent->exit_no == 1)
	{
		goal_force = glm::vec2(current_message->exit0_x, current_message->exit0_y);
		if (exit_location == 1)
		{
			if (EXIT1_STATE)
				kill_agent = 1;
			else
				agent->exit_no = getNewExitLocation(rand48);
		}
	}
	else if (agent->exit_no == 2)
	{
		goal_force = glm::vec2(current_message->exit1_x, current_message->exit1_y);
		if (exit_location == 2)
			if (EXIT2_STATE)
				kill_agent = 1;
			else
				agent->exit_no = getNewExitLocation(rand48);
	}
	else if (agent->exit_no == 3)
	{
		goal_force = glm::vec2(current_message->exit2_x, current_message->exit2_y);
		if (exit_location == 3)
			if (EXIT3_STATE)
				kill_agent = 1;
			else
				agent->exit_no = getNewExitLocation(rand48);
	}
	else if (agent->exit_no == 4)
	{
		goal_force = glm::vec2(current_message->exit3_x, current_message->exit3_y);
		if (exit_location == 4)
			if (EXIT4_STATE)
				kill_agent = 1;
			else
				agent->exit_no = getNewExitLocation(rand48);
	}
	else if (agent->exit_no == 5)
	{
		goal_force = glm::vec2(current_message->exit4_x, current_message->exit4_y);
		if (exit_location == 5)
			if (EXIT5_STATE)
				kill_agent = 1;
			else
				agent->exit_no = getNewExitLocation(rand48);
	}
	else if (agent->exit_no == 6)
	{
		goal_force = glm::vec2(current_message->exit5_x, current_message->exit5_y);
		if (exit_location == 6)
			if (EXIT6_STATE)
				kill_agent = 1;
			else
				agent->exit_no = getNewExitLocation(rand48);
	}
	else if (agent->exit_no == 7)
	{
		goal_force = glm::vec2(current_message->exit6_x, current_message->exit6_y);
		if (exit_location == 7)
			if (EXIT7_STATE)
				kill_agent = 1;
			else
				agent->exit_no = getNewExitLocation(rand48);
	}

	//scale goal force
	goal_force *= GOAL_WEIGHT;

	agent->steer_x += collision_force.x + goal_force.x;
	agent->steer_y += collision_force.y + goal_force.y;

	//update height
	agent->height = current_message->height;

    return kill_agent;
}

/**
 * move FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int move(xmachine_memory_agent* agent){

	glm::vec2 agent_pos = glm::vec2(agent->x, agent->y);
	glm::vec2 agent_vel = glm::vec2(agent->velx, agent->vely);
	glm::vec2 agent_steer = glm::vec2(agent->steer_x, agent->steer_y);

	float current_speed = length(agent_vel)+0.025f;//(powf(length(agent_vel), 1.75f)*0.01f)+0.025f;

    //apply more steer if speed is greater
	agent_vel += current_speed*agent_steer;
	float speed = length(agent_vel);
	//limit speed
	if (speed >= agent->speed){
		agent_vel = normalize(agent_vel)*agent->speed;
		speed = agent->speed;
	}

	//update position
	agent_pos += agent_vel*TIME_SCALER;

    
	//animation
	agent->animate += (agent->animate_dir * powf(speed,2.0f)*TIME_SCALER*100.0f);
	if (agent->animate >= 1)
		agent->animate_dir = -1;
	if (agent->animate <= 0)
		agent->animate_dir = 1;

	//lod
	agent->lod = 1;

	//update
	agent->x = agent_pos.x;
	agent->y = agent_pos.y;
	agent->velx = agent_vel.x;
	agent->vely = agent_vel.y;

	//bound by wrapping
	if (agent->x < -1.0f)
		agent->x+=2.0f;
	if (agent->x > 1.0f)
		agent->x-=2.0f;
	if (agent->y < -1.0f)
		agent->y+=2.0f;
	if (agent->y > 1.0f)
		agent->y-=2.0f;

    return 0;
}



/**
 * generate_pedestrians FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_navmap. This represents a single agent instance and can be modified directly.
 * @param agent_agents Pointer to agent list of type xmachine_memory_agent_list. This must be passed as an argument to the add_agent_agent function to add a new agent.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an arument to the rand48 function for genertaing random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int generate_pedestrians(xmachine_memory_navmap* agent, xmachine_memory_agent_list* agent_agents, RNG_rand48* rand48){

    if (agent->exit_no > 0)
	{
		float random = rnd<DISCRETE_2D>(rand48);
		bool emit_agent = false;

		if ((agent->exit_no == 1)&&((random < EMMISION_RATE_EXIT1*TIME_SCALER)))
			emit_agent = true;
		if ((agent->exit_no == 2)&&((random <EMMISION_RATE_EXIT2*TIME_SCALER)))
			emit_agent = true;
		if ((agent->exit_no == 3)&&((random <EMMISION_RATE_EXIT3*TIME_SCALER)))
			emit_agent = true;
		if ((agent->exit_no == 4)&&((random <EMMISION_RATE_EXIT4*TIME_SCALER)))
			emit_agent = true;
		if ((agent->exit_no == 5)&&((random <EMMISION_RATE_EXIT5*TIME_SCALER)))
			emit_agent = true;
		if ((agent->exit_no == 6)&&((random <EMMISION_RATE_EXIT6*TIME_SCALER)))
			emit_agent = true;
		if ((agent->exit_no == 7)&&((random <EMMISION_RATE_EXIT7*TIME_SCALER)))
			emit_agent = true;

		if (emit_agent){
			float x = ((agent->x+0.5f)/(d_message_navmap_cell_width/ENV_WIDTH))-ENV_MAX;
			float y = ((agent->y+0.5f)/(d_message_navmap_cell_width/ENV_WIDTH))-ENV_MAX;
			int exit = getNewExitLocation(rand48);
			float animate = rnd<DISCRETE_2D>(rand48);
			float speed = (rnd<DISCRETE_2D>(rand48))*0.5f + 1.0f;
			add_agent_agent(agent_agents, x, y, 0.0f, 0.0f, 0.0f, 0.0f, agent->height, exit, speed, 1, animate, 1);
		}
	}


    return 0;
}

#endif //_FLAMEGPU_FUNCTIONS
