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

//Environment Bounds
#define MIN_POSITION -0.5f
#define MAX_POSITION +0.5f

//Interaction radius
#define INTERACTION_RADIUS 0.1f
#define SEPARATION_RADIUS 0.005f

//Global Scalers
#define TIME_SCALE 0.0005f
#define GLOBAL_SCALE 0.15f


//Rule scalers
#define STEER_SCALE 0.65f
#define COLLISION_SCALE 0.75f
#define MATCH_SCALE 1.25f

inline __device__ float dot(float3 a, float3 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ float length(float3 v)
{
    return sqrtf(dot(v, v));
}

__FLAME_GPU_FUNC__ float3 boundPosition(float3 agent_position){
	agent_position.x = (agent_position.x < MIN_POSITION)? MAX_POSITION: agent_position.x;
	agent_position.x = (agent_position.x > MAX_POSITION)? MIN_POSITION: agent_position.x;

	agent_position.y = (agent_position.y < MIN_POSITION)? MAX_POSITION: agent_position.y;
	agent_position.y = (agent_position.y > MAX_POSITION)? MIN_POSITION: agent_position.y;

	agent_position.z = (agent_position.z < MIN_POSITION)? MAX_POSITION: agent_position.z;
	agent_position.z = (agent_position.z > MAX_POSITION)? MIN_POSITION: agent_position.z;

	return agent_position;
}


//Boid Agent Functions

//The following function arguments have been generated automatically by the FLAMEGPU XParser and are dependant on the function input and outputs. If they are changed manually be sure to match any arguments to the XMML specification.
//Input : 
//Output: location 
//Agent Output: 
__FLAME_GPU_FUNC__ int outputdata(xmachine_memory_Boid* xmemory, xmachine_message_location_list* location_messages) 
{
	add_location_message(location_messages, xmemory->id, xmemory->x, xmemory->y, xmemory->z, xmemory->fx, xmemory->fy, xmemory->fz);

	return 0;
}

//The following function arguments have been generated automatically by the FLAMEGPU XParser and are dependant on the function input and outputs. If they are changed manually be sure to match any arguments to the XMML specification.
//Input : location 
//Output: 
//Agent Output: 
__FLAME_GPU_FUNC__ int inputdata(xmachine_memory_Boid* xmemory, xmachine_message_location_list* location_messages, xmachine_message_location_PBM* partition_matrix) 
{
	//Agent position vector
	float3 agent_position = make_float3(xmemory->x, xmemory->y, xmemory->z);
	float3 agent_velocity = make_float3(xmemory->fx, xmemory->fy, xmemory->fz);

	//Boids perceived center
	float3 global_centre = make_float3(0.0f, 0.0f, 0.0f);
	int global_centre_count = 0;

	//Boids global velocity matching
	float3 global_velocity = make_float3(0.0f, 0.0f, 0.0f);

	//Boids short range avoidance center
	float3 collision_centre = make_float3(0.0f, 0.0f, 0.0f);
	int collision_count = 0;

	xmachine_message_location* location_message = get_first_location_message(location_messages, partition_matrix, xmemory->x, xmemory->y, xmemory->z);
	int count;
	
    while(location_message)
	{
		count++;
		
		//Message position vector
		float3 message_position = make_float3(location_message->x, location_message->y, location_message->z);

		if (location_message->id != xmemory->id){
			float separation = length(agent_position - message_position);
			if (separation < (INTERACTION_RADIUS)){

				//Update Percieved global center
				global_centre += message_position;
				global_centre_count += 1;

				//Update global velocity matching
				float3 message_velocity = make_float3(location_message->fx, location_message->fy, location_message->fz);
				global_velocity += message_velocity;

				//Update collision center
				if (separation < (SEPARATION_RADIUS)){ //dependant on model size
					collision_centre += message_position;
					collision_count += 1;
				}

			}
		}
		
		
		location_message = get_next_location_message(location_message, location_messages, partition_matrix);
	}
	



	//Total change in velocity
	float3 velocity_change = make_float3(0.0f, 0.0f, 0.0f);


	//Rule 1) Steer towards perceived center of flock
	float3 steer_velocity = make_float3(0.0f, 0.0f, 0.0f);
	if (global_centre_count >0){
		global_centre /= global_centre_count;
		steer_velocity = (global_centre - agent_position)* STEER_SCALE;
	}
	velocity_change += steer_velocity; 

	//Rule 2) Match neighbours speeds
	float3 match_velocity = make_float3(0.0f, 0.0f, 0.0f);
	if (collision_count > 0){
		global_velocity /= collision_count;
		match_velocity = match_velocity* MATCH_SCALE;
	}
	velocity_change += match_velocity; 

	//Rule 3) Avoid close range neighbours
	float3 avoid_velocity = make_float3(0.0f, 0.0f, 0.0f);
	if (collision_count > 0){
		collision_centre /= collision_count;
		avoid_velocity = (agent_position - collision_centre)* COLLISION_SCALE;
	}
	velocity_change += avoid_velocity; 


	//Global scale of velocity change
	velocity_change *= GLOBAL_SCALE;

	//Update agent velocity
	agent_velocity += velocity_change;

	//Bound velocity
	float agent_velocity_scale = length(agent_velocity);
	if (agent_velocity_scale > 1){
		agent_velocity /= agent_velocity_scale; 
	}

	//Apply the velocity
	agent_position += agent_velocity * TIME_SCALE;

	//Bound position
	agent_position = boundPosition(agent_position);
	
	//Update agent structure
	xmemory->x = agent_position.x;
	xmemory->y = agent_position.y;
	xmemory->z = agent_position.z;

	xmemory->fx = agent_velocity.x;
	xmemory->fy = agent_velocity.y;
	xmemory->fz = agent_velocity.z;


	return 0;
}



#endif // #ifndef _FUNCTIONS_H_
