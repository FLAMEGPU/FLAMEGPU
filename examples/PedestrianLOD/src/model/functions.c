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

#define LOD1_DISTANCE 0.35f
#define LOD2_DISTANCE 1.00f

//#define NUM_EXITS 7

#define PI 3.1415f
#define RADIANS(x) (PI / 180.0f) * x


inline __FLAME_GPU_FUNC__ float dot(float3 a, float3 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __FLAME_GPU_FUNC__ float length(float3 v)
{
    return sqrtf(dot(v, v));
}
inline __FLAME_GPU_FUNC__ float3 normalize(float3 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
inline __FLAME_GPU_FUNC__ float dot(float2 a, float2 b)
{ 
    return a.x * b.x + a.y * b.y;
}
inline __FLAME_GPU_FUNC__ float length(float2 v)
{
    return sqrtf(dot(v, v));
}
inline __FLAME_GPU_FUNC__ float2 normalize(float2 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
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
 * move FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param location_messages  location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_location_message and get_next_location_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_location_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an arument to the rand48 function for genertaing random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int avoid_pedestrians(xmachine_memory_agent* agent, xmachine_message_pedestrian_location_list* pedestrian_location_messages, xmachine_message_pedestrian_location_PBM* partition_matrix, RNG_rand48* rand48){

    float2 agent_pos = make_float2(agent->x, agent->y);
	float2 agent_vel = make_float2(agent->velx, agent->vely);

	float2 navigate_velocity = make_float2(0.0f, 0.0f);
	float2 avoid_velocity = make_float2(0.0f, 0.0f);

	xmachine_message_pedestrian_location* current_message = get_first_pedestrian_location_message(pedestrian_location_messages, partition_matrix, agent->x, agent->y, 0.0);
	while (current_message)
	{
		float2 message_pos = make_float2(current_message->x, current_message->y);
		float separation = length(agent_pos - message_pos);
		if ((separation < MESSAGE_RADIUS)&&(separation>MIN_DISTANCE)){
			float2 to_agent = normalize(agent_pos-message_pos);	
			float ang = acosf(dot(agent_vel, to_agent));
			float perception = 45.0f;

			//STEER
			if ((ang < RADIANS(perception)) || (ang > 3.14159265f-RADIANS(perception))){
				float2 s_velocity = to_agent;
				s_velocity *= powf(I_SCALER/separation, 1.25f)*STEER_WEIGHT;
				navigate_velocity += s_velocity;
			}

			//AVOID
			float2 a_velocity = to_agent;
			a_velocity *= powf(I_SCALER/separation, 2.00f)*AVOID_WEIGHT;
			avoid_velocity += a_velocity;						

		}
		 current_message = get_next_pedestrian_location_message(current_message, pedestrian_location_messages, partition_matrix);
	}

	//random walk goal
	float2 goal_velocity = make_float2(0.0f, 0.0f);;
	goal_velocity.x += agent->velx * GOAL_WEIGHT;
	goal_velocity.y += agent->vely * GOAL_WEIGHT;

	//maximum velocity rule
	float2 steer_velocity = navigate_velocity + avoid_velocity + goal_velocity;


		
	agent->steer_x = steer_velocity.x;
	agent->steer_y = steer_velocity.y;



    return 0;
}

  
/**
 * move FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int move(xmachine_memory_agent* agent){

	float2 agent_pos = make_float2(agent->x, agent->y);
	float2 agent_vel = make_float2(agent->velx, agent->vely);
	float2 agent_steer = make_float2(agent->steer_x, agent->steer_y);

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
	float3 agent_pos_3d = make_float3(agent->x, agent->y, 0.0);
	float3 eye_pos = make_float3(EYE_X, EYE_Y, EYE_Z);
	float distance = length(eye_pos - agent_pos_3d);
	if (distance < LOD1_DISTANCE)
		agent->lod = 1;
	else if (distance < LOD2_DISTANCE)
		agent->lod = 2;
	else
		agent->lod = 3;

	//update
	agent->x = agent_pos.x;
	agent->y = agent_pos.y;
	agent->velx = agent_vel.x;
	agent->vely = agent_vel.y;

	//bound by wrapping
	if (agent->x <= d_message_pedestrian_location_min_bounds.x)
		agent->x=d_message_pedestrian_location_max_bounds.x;
	if (agent->x > d_message_pedestrian_location_max_bounds.x)
		agent->x=d_message_pedestrian_location_min_bounds.x;
	if (agent->y <= d_message_pedestrian_location_min_bounds.y)
		agent->y=d_message_pedestrian_location_max_bounds.y;
	if (agent->y > d_message_pedestrian_location_max_bounds.y)
		agent->y=d_message_pedestrian_location_min_bounds.y;

    return 0;
}




#endif //_FLAMEGPU_FUNCTIONS
