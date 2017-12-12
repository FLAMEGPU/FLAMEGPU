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

inline __device__ float dot(fvec3 a, fvec3 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ float length(fvec3 v)
{
    return sqrtf(dot(v, v));
}

__FLAME_GPU_FUNC__ fvec3 boundPosition(fvec3 agent_position){
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
	add_location_message(location_messages, xmemory->id, xmemory->position.x, xmemory->position.y, xmemory->position.z, xmemory->velocity.x, xmemory->velocity.y, xmemory->velocity.z);

	return 0;
}

//The following function arguments have been generated automatically by the FLAMEGPU XParser and are dependant on the function input and outputs. If they are changed manually be sure to match any arguments to the XMML specification.
//Input : location 
//Output: 
//Agent Output: 
__FLAME_GPU_FUNC__ int inputdata(xmachine_memory_Boid* xmemory, xmachine_message_location_list* location_messages, xmachine_message_location_PBM* partition_matrix) 
{
	//Boids perceived center
	fvec3 global_centre = fvec3(0.0f, 0.0f, 0.0f);
	int global_centre_count = 0;

	//Boids global velocity matching
	fvec3 global_velocity = fvec3(0.0f, 0.0f, 0.0f);

	//Boids short range avoidance center
	fvec3 collision_centre = fvec3(0.0f, 0.0f, 0.0f);
	int collision_count = 0;

	xmachine_message_location* location_message = get_first_location_message(location_messages, partition_matrix, xmemory->position.x, xmemory->position.y, xmemory->position.z);
	int count = 0;
	
    while(location_message)
	{
		count++;
		//create some vector types
		fvec3 message_position = fvec3(location_message->x, location_message->y, location_message->z);
		fvec3 message_velocity = fvec3(location_message->vx, location_message->vy, location_message->vz);

		
		if (location_message->id != xmemory->id){
			float separation = length(xmemory->position - message_position);
			if (separation < (INTERACTION_RADIUS)){

				//Update Perceived global centre

				global_centre += message_position;
				global_centre_count += 1;

				//Update global velocity matching
				global_velocity += message_velocity;

				//Update collision centre
				if (separation < (SEPARATION_RADIUS)){ //dependant on model size
					collision_centre += message_position;
					collision_count += 1;
				}

			}
		}
		
		
		location_message = get_next_location_message(location_message, location_messages, partition_matrix);
	}
	



	//Total change in velocity
	fvec3 velocity_change = fvec3(0.0f, 0.0f, 0.0f);


	//Rule 1) Steer towards perceived center of flock
	fvec3 steer_velocity = fvec3(0.0f, 0.0f, 0.0f);
	if (global_centre_count >0){
		global_centre /= global_centre_count;
		steer_velocity = (global_centre - xmemory->position)* STEER_SCALE;
	}
	velocity_change += steer_velocity; 

	//Rule 2) Match neighbours speeds
	fvec3 match_velocity = fvec3(0.0f, 0.0f, 0.0f);
	if (collision_count > 0){
		global_velocity /= collision_count;
		match_velocity = match_velocity* MATCH_SCALE;
	}
	velocity_change += match_velocity; 

	//Rule 3) Avoid close range neighbours
	fvec3 avoid_velocity = fvec3(0.0f, 0.0f, 0.0f);
	if (collision_count > 0){
		collision_centre /= collision_count;
		avoid_velocity = (xmemory->position - collision_centre)* COLLISION_SCALE;
	}
	velocity_change += avoid_velocity; 


	//Global scale of velocity change
	velocity_change *= GLOBAL_SCALE;

	//Update agent velocity
	xmemory->velocity += velocity_change;

	//Bound velocity
	float velocity_scale = length(xmemory->velocity);
	if (velocity_scale > 1){
		xmemory->velocity /= velocity_scale; 
	}

	//Apply the velocity
	xmemory->position += xmemory->velocity * TIME_SCALE;

	//Bound position
	xmemory->position = boundPosition(xmemory->position);

	return 0;
}



#endif // #ifndef _FUNCTIONS_H_
