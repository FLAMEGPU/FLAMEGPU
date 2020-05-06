
/*
 * Author: Dr Paul Richmond
 * Copyright 2011 University of Sheffield.
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

#include <stdlib.h>
#include <math.h>

#define float3 glm::vec3
#define length(x) glm::length(x)
#define PI 3.1415f
#define RADIANS(x) (PI / 180.0f) * x

#define MIN_BOUND -2.0f
#define MAX_BOUND +2.0f
// P = ceil((MAX_BOUND - MIN_BOUND) / radius)>
// In partitioned messaging: Partition_size: P should be at least 3 in both x and y axes and at least 1 in z-axis.


#define BOUNDS_WIDTH (MAX_BOUND - MIN_BOUND)

// Simulation variables:
#define DELTA_TIME 0.015f //original
#define INTERACTION_RANGE 0.30000f
//#define INTERACTION_RANGE 2.00000f

#define MUTATION_RATE 0.000030f // Original value
//#define MUTATION_RATE 0.030f //----> Higher rate of explosions.Confirmed.

#define TEMP 1.8000f // Temptation. 2.4 | 3.0 | -1
#define REWARD 1.0000f
#define SUCK -1.4000f
#define PUNISH -1.0000f
// 2R > T+S

#define I_RATE 1.000000f // 1.0 | 0.5 | 0.05 | 0.01
//#define I_RATE 0.005000f
// .. 0.5 | 0.1 | 0.05 | 0.001


/***************************************************************************************************
|||																								 |||
|||			COOPERATOR		  ---------------------------------------> 1						 |||
|||			DEFECTOR   			---------------------------------------> 0						 |||
|||	 																							 |||
****************************************************************************************************/
// ------------------------------ ADDITIONAL SUBROUTINES ------------------------------------------------

__FLAME_GPU_FUNC__ float3 boundPosition(float3 position_vector)
{
	position_vector.x = (position_vector.x < MIN_BOUND) ? MAX_BOUND : position_vector.x;
	position_vector.x = (position_vector.x > MAX_BOUND) ? MIN_BOUND : position_vector.x;

	position_vector.y = (position_vector.y < MIN_BOUND) ? MAX_BOUND : position_vector.y;
	position_vector.y = (position_vector.y > MAX_BOUND) ? MIN_BOUND : position_vector.y;

	position_vector.z = (position_vector.z < MIN_BOUND) ? MAX_BOUND : position_vector.z;
	position_vector.z = (position_vector.z > MAX_BOUND) ? MIN_BOUND : position_vector.z;

	return position_vector;
}

__FLAME_GPU_FUNC__ float truncate(float num, int num_after_point)
{
	float exponent = powf(10, 1.0f*num_after_point);
	float normalized_num = num * exponent;
	normalized_num = round(normalized_num);
	int n = (int) normalized_num;
	float result = ((float)n) / exponent;

	return result;
}

__FLAME_GPU_FUNC__ float3 limit(float3 vec, int n)
{
	float3 norm_vect = normalize(vec);
	float x = n * norm_vect.x;
	float y = n * norm_vect.y;
	float z = n * norm_vect.z;

	float3 resulting_vect = float3(x,y,z);
	return resulting_vect;
}

__FLAME_GPU_FUNC__ float3 limit(float3 vec, float f)
{
	float3 norm_vect = normalize(vec);
	float x = f * norm_vect.x;
	float y = f * norm_vect.y;
	float z = f * norm_vect.z;

	float3 resulting_vect = float3(x,y,z);
	return resulting_vect;
}

__FLAME_GPU_FUNC__ float getCurrentPayoff(int agentStrat, int neighStrat, float defaultPayoff)
{
	float payoff = defaultPayoff;//if 's' is neigher 0 nor 1 , pick a random value from payoff matrix. Much unlikely, but we gotta make sure all contengencies are taken care of xD.

	switch(agentStrat)
	{
		case 1://C
		if(neighStrat == 1)
		{
			payoff = REWARD;
		}
		else
		{
			payoff = SUCK;
		}
		break;
		case 0://D
		if(neighStrat == 1) //C
		{
			payoff = TEMP;
		}
		else
		{
			payoff = PUNISH;
		}
		break;
		default:
		  payoff = defaultPayoff;
		  break;
	}

	return payoff;
}

/**
 * navigate FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.

 */
__FLAME_GPU_FUNC__ int navigate(xmachine_memory_agent* agent)
{
  float3 position_vector = float3(agent->x, agent->y, agent->z);
  float3 velocity_vector = float3(agent->vx, agent->vy, agent->vz);
  float3 steer_vector = float3(agent->steer_x, agent->steer_y, agent->steer_z);

  velocity_vector += steer_vector;
	//printf("length: %f - trunc = %f\n", length(velocity_vector), trunc(length(velocity_vector)));


  float current_velocity = truncate(length(velocity_vector), 3);
	if (current_velocity > 1.0f) {
    velocity_vector = normalize(velocity_vector);
		//velocity_vector = limit(velocity_vector, 3.0f);
  }

  position_vector += velocity_vector * DELTA_TIME;

  //Bound coordinates to the environement
  position_vector = boundPosition(position_vector);

  // Update position and velocity vectors
  agent->x = position_vector.x;
  agent->y = position_vector.y;
  agent->z = position_vector.z;

  agent->vx = velocity_vector.x;
  agent->vy = velocity_vector.y;
  agent->vz = velocity_vector.z;

  return 0;
}

/**
 * interact FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param agent_strategy_messages  agent_strategy_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_agent_strategy_message and get_next_agent_strategy_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_agent_strategy_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int interact(xmachine_memory_agent* agent, xmachine_message_agent_strategy_list* agent_strategy_messages, xmachine_message_agent_strategy_PBM* partition_matrix, RNG_rand48* rand48)
{
  int default_payoff = (int)(truncate(rnd<CONTINUOUS>(rand48), 1) * 10) % 4;

	float temp = TEMP;
  float reward = REWARD;
  float suck = SUCK;
  float punish = PUNISH;

	float payoffs[4] = {reward, punish, temp, suck}; // [R,P,T,S]

	float3 position_vector = float3(agent->x, agent->y, agent->z);
	float3 velocity_vector = float3(agent->vx, agent->vy, agent->vz);
	float3 random_steer_vector = float3(agent->steer_x, agent->steer_y, agent->steer_z); //Agent initial steering vector.

	float3 init_steer_vector = float3(0.0f, 0.0f, 0.0f);

	//================== Random steering coefficients ============================
	float3 steer_vector = float3(0.0f, 0.0f, 0.0f);

	//random steering (x, y, z)
	float r1 = truncate(rnd<CONTINUOUS>(rand48), 7);
	float r2 = truncate(rnd<CONTINUOUS>(rand48), 7);
	float r3 = truncate(rnd<CONTINUOUS>(rand48), 7);

	float3 attraction_vector = float3(0.0f, 0.0f, 0.0f);
	float3 avoidance_vector = float3(0.0f, 0.0f, 0.0f);

	float3 attraction_point = float3(0.0f, 0.0f, 0.0f);
	float3 repulsion_point = float3(0.0f, 0.0f, 0.0f);
	int count_neighbors = 0;

	agent->neighbors_score = 0.0f; //reset score after each iteration.

	xmachine_message_agent_strategy* current_message = get_first_agent_strategy_message(agent_strategy_messages,
		partition_matrix,
		agent->x,
		agent->y,
		agent->z);

		float distance = 0.0f;
		float3 message_coordinates = float3(0.0f, 0.0f, 0.0f);
		while(current_message)
		{
			message_coordinates = float3(current_message->x, current_message->y, current_message->z);
			distance = truncate(length(position_vector - message_coordinates), 5);

			if((distance <= INTERACTION_RANGE) && (agent->id != current_message->id) && (distance != 0))
			{
				count_neighbors += 1;
				repulsion_point += message_coordinates;
				attraction_point += message_coordinates;

				//....................................................................................
				// no-memory case: (Default settings)
				default_payoff = (int)(truncate(rnd<CONTINUOUS>(rand48), 1) * 10) % 4;//the un-eventual picked payoff in case (s != {0,1})
				float currentPayoff = getCurrentPayoff(agent->strategy, current_message->strategy, payoffs[default_payoff]);

				agent->neighbors_score += (currentPayoff / distance); // agent-agent relationship is relevant
			}
			current_message = get_next_agent_strategy_message(current_message, agent_strategy_messages, partition_matrix);
		}

		if (count_neighbors != 0)
		{
			if(agent->neighbors_score >= 0) // Attraction
			{
				attraction_point /= count_neighbors;
				attraction_vector = attraction_point - position_vector;// steering (attr.) = (desired - current) ~C. Reynolds.

				steer_vector = float3(attraction_vector.x, attraction_vector.y, attraction_vector.z);
			} else  // Repulsion
			{
				repulsion_point /= count_neighbors;
				avoidance_vector = position_vector - repulsion_point;// steering (repul.) = (current - desired) ~C. Reynolds.

				steer_vector = float3(avoidance_vector.x, avoidance_vector.y, avoidance_vector.z);
			}
		} else
		{ // random movements
			if(r1 >= 0.0005000)
			{
				agent->steer_x = (-1 * agent->steer_x);
			}
			if(r2 >= 0.0005000)
			{
				agent->steer_y = (-1 * agent->steer_y);
			}
			if(r3 >= 0.0005000)
			{
				agent->steer_z = (-1 * agent->steer_z);
			}
			steer_vector = random_steer_vector; // force vector

			//steer_vector = init_steer_vector;
		}

		/*float force_scale = truncate(length(steer_vector), 8);
		if (force_scale >= 0.0300f){ // original: 0.0300f
			steer_vector = limit(steer_vector, 0.0300f);
		}*/

		velocity_vector += steer_vector;

		float velocity_scale = truncate(length(velocity_vector), 5);
		if (velocity_scale > 1.0f){
			velocity_vector = normalize(velocity_vector);
		}

		position_vector += velocity_vector * DELTA_TIME;

		//Bound coordinates to the environement
		position_vector = boundPosition(position_vector);

		// update kinetic parameters
		agent->vx = velocity_vector.x;
		agent->vy = velocity_vector.y;
		agent->vz = velocity_vector.z;

		agent->x = position_vector.x;
		agent->y = position_vector.y;
		agent->z = position_vector.z;
		// Explicit Euler integration

		//agent->neighbors_score = 0.0f; //reset score after each iteration.

		return 0;
}

/**
 * attract FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param agent_strategy_messages  agent_strategy_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_agent_strategy_message and get_next_agent_strategy_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_agent_strategy_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.
 */
__FLAME_GPU_FUNC__ int attract(xmachine_memory_agent* agent, xmachine_message_agent_strategy_list* agent_strategy_messages, xmachine_message_agent_strategy_PBM* partition_matrix)
{
    float3 position_vector = float3(agent->x, agent->y, agent->z);
    float3 velocity_vector = float3(agent->vx, agent->vy, agent->vz);
    float3 steer_vector = float3(agent->steer_x, agent->steer_y, agent->steer_z);

    float3 steering_force = float3(0.0f, 0.0f, 0.0f);

    float3 v1 = float3(1.0f, 0.0f, 0.0f);
    float3 v2 = float3(0.0f, 1.0f,  0.0f);
    float3 v3 = float3(0.0f, 0.0f, 1.0f);
    float3 attraction_point = v1 - v2 + v3;

    xmachine_message_agent_strategy* current_message = get_first_agent_strategy_message(agent_strategy_messages,
      partition_matrix,
      agent->x,
      agent->y,
      agent->z);
    float3 message_coordinates = float3(0.0f, 0.0f, 0.0f);
    while(current_message)
    {
      float3 message_coordinates = float3(current_message->x, current_message->y, current_message->z);
      current_message = get_next_agent_strategy_message(current_message, agent_strategy_messages, partition_matrix);
    }

    steering_force = attraction_point - position_vector;// steering force = (Desired_velocity - current_velocity) ~C. Reynolds.
    steer_vector = float3(steering_force.x, steering_force.y, steering_force.z);

    velocity_vector += steering_force;
    float current_velocity = truncate(length(velocity_vector), 5);

    if (current_velocity > 1.0)
    {
      velocity_vector = normalize(velocity_vector);
    }
    position_vector += velocity_vector * DELTA_TIME;

    position_vector = boundPosition(position_vector);

    agent->x = position_vector.x;
    agent->y = position_vector.y;
    agent->z = position_vector.z;

    agent->vx = velocity_vector.x;
    agent->vy = velocity_vector.y;
    agent->vz = velocity_vector.z;
    // Semi-explicit Euler

    return 0;
}

/**
 * avoid FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param agent_strategy_messages  agent_strategy_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_agent_strategy_message and get_next_agent_strategy_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_agent_strategy_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.
 */
__FLAME_GPU_FUNC__ int avoid(xmachine_memory_agent* agent, xmachine_message_agent_strategy_list* agent_strategy_messages, xmachine_message_agent_strategy_PBM* partition_matrix)
{
    float3 position_vector = float3(agent->x, agent->y, agent->z);
  	float3 velocity_vector = float3(agent->vx, agent->vy, agent->vz);
  	float3 steer_vector = float3(agent->steer_x, agent->steer_y, agent->steer_z);

  	float3 steering_force = float3(0.0f, 0.0f, 0.0f);
  	float3 repulsion_point = float3(0.0f, 0.0f, 0.0f);

  	xmachine_message_agent_strategy* current_message = get_first_agent_strategy_message(agent_strategy_messages,
  		partition_matrix,
  		agent->x,
  		agent->y,
  		agent->z);
  	float3 message_coordinates = float3(0.0f, 0.0f, 0.0f);
  	while(current_message)
  	{
      float3 message_coordinates = float3(current_message->x, current_message->y, current_message->z);
  		current_message = get_next_agent_strategy_message(current_message, agent_strategy_messages, partition_matrix);
  	 }

    steering_force = position_vector - repulsion_point;// steering force = (current_velocity - desired_velocity) ~C. Reynolds.
    steer_vector = float3(steering_force.x, steering_force.y, steering_force.z);

    velocity_vector += steering_force;
    float current_velocity = truncate(length(velocity_vector), 5);

    if(current_velocity > 1.0)
    {
  	   velocity_vector = normalize(velocity_vector);
     }
     position_vector += velocity_vector * DELTA_TIME;

     position_vector = boundPosition(position_vector);

     agent->x = position_vector.x;
     agent->y = position_vector.y;
     agent->z = position_vector.z;

     agent->vx = velocity_vector.x;
     agent->vy = velocity_vector.y;
     agent->vz = velocity_vector.z;
     // Explicit Euler

     return 0;
}

/**
 * align FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param agent_location_messages  agent_location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_agent_location_message and get_next_agent_location_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_agent_location_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.
 */
__FLAME_GPU_FUNC__ int align(xmachine_memory_agent* agent, xmachine_message_agent_location_list* agent_location_messages, xmachine_message_agent_location_PBM* partition_matrix)
{
  float3 position_vector = float3(agent->x, agent->y, agent->z);
  float3 mean_velocity = float3(0.0f, 0.0f, 0.0f);

  float3 match_velocity_vector = float3(0.0f, 0.0f, 0.0f);
  int count_neighbors = 0;

  xmachine_message_agent_location* current_message = get_first_agent_location_message(agent_location_messages,
    partition_matrix,
    agent->x,
    agent->y,
    agent->z);
    float distance = 0.0f;
    float3 message_coordinates = float3(0.0f, 0.0f, 0.0f);
    float3 message_velocity = float3(0.0f, 0.0f, 0.0f); //fetch velocity of neighbors
    while(current_message)
    {
      message_coordinates = float3(current_message->x, current_message->y, current_message->z);
      message_velocity = float3(current_message->vx, current_message->vy, current_message->vz);
      distance = length(position_vector - message_coordinates);

      if((distance <= INTERACTION_RANGE) && (agent->id != current_message->id))
      {
        mean_velocity += message_velocity;
        count_neighbors += 1;
      }
      current_message = get_next_agent_location_message(current_message, agent_location_messages, partition_matrix);
    }

    if (count_neighbors)
    {
      mean_velocity /= count_neighbors;
      match_velocity_vector = mean_velocity;
    }

    agent->steer_x = match_velocity_vector.x;
    agent->steer_y = match_velocity_vector.y;
    agent->steer_z = match_velocity_vector.z;

    return 0;
}

/**
 * agent_output_location FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param agent_location_messages Pointer to output message list of type xmachine_message_agent_location_list. Must be passed as an argument to the add_agent_location_message function ??.
 */
__FLAME_GPU_FUNC__ int agent_output_location(xmachine_memory_agent* agent, xmachine_message_agent_location_list* agent_location_messages)
{
    add_agent_location_message(agent_location_messages, agent->id, agent->x, agent->y, agent->z, agent->vx, agent->vy, agent->vz);
    return 0;
}

/**
 * agent_output_strategy FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param agent_strategy_messages Pointer to output message list of type xmachine_message_agent_strategy_list. Must be passed as an argument to the add_agent_strategy_message function ??.
 */
__FLAME_GPU_FUNC__ int agent_output_strategy(xmachine_memory_agent* agent, xmachine_message_agent_strategy_list* agent_strategy_messages)
{
    add_agent_strategy_message(agent_strategy_messages, agent->id, agent->strategy, agent->x, agent->y, agent->z);
    return 0;
}

/**
 * set_next_strategy FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param agent_strategy_messages  agent_strategy_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_agent_strategy_message and get_next_agent_strategy_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_agent_strategy_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int set_next_strategy(xmachine_memory_agent* agent, xmachine_message_agent_strategy_list* agent_strategy_messages, xmachine_message_agent_strategy_PBM* partition_matrix, RNG_rand48* rand48)
{
  float3 position_vector = float3(agent->x, agent->y, agent->z);

  float3 current_message_coord = float3(0.0f, 0.0f, 0.0f); // vector comprising the exchanged-information between current_agent and other_agents
  int current_message_strategy = -1;

  float distance = 0.0f;
  int num_neigh = 0;
  int num_coop = 0;//num_coop <= num_neigh
  float ratio_of_coop = 0.0f; // ratio_of_coop = num_coop / num_neigh

  float prob = truncate(rnd<CONTINUOUS>(rand48), 6);//probability of mutation

  xmachine_message_agent_strategy* current_message = get_first_agent_strategy_message(agent_strategy_messages,
    partition_matrix,
    agent->x,
    agent->y,
    agent->z);
    while(current_message)
    {
      current_message_coord = float3(current_message->x, current_message->y, current_message->z); // get coord. of neighbor(n)
      current_message_strategy = current_message->strategy;  // get strategy of neighbor(n)
      distance = truncate(length(current_message_coord - position_vector), 5);

      if((distance <= INTERACTION_RANGE) && (distance >= 0.0f) && (agent->id != current_message->id))
      {
        num_neigh +=1;
        if(current_message_strategy == 1)
        {
          num_coop +=1; //Count Num of Cooperators
        }
      }
      current_message = get_next_agent_strategy_message(current_message, agent_strategy_messages, partition_matrix);
    }


    ratio_of_coop = (num_neigh != 0) ? truncate(((float)num_coop / (float)num_neigh ), 6) : 0.000000f;
		// With time, agents learn to approximate the ratio_of_cooperators (cr -> ce) withon proximity, using the following formula:
			/* ce <- ce + L*(cr-ce). */
		agent->ce += (ratio_of_coop - agent->ce) * I_RATE;
		//agent->ct_e += (agent->coop_threshold - agent->ct_e) * I_RATE_TH;

    if(agent->ce != 0) // ratio_of_coop
    {
      if(agent->ce > agent->coop_threshold) //threshold at which cooperation starts. // *** ratio_of_coop.
			//if(agent->ce > agent->ct_e)
      {
        agent->strategy = 1;//C
      } else
      {
        agent->strategy = 0;
      }
    }

		agent->strategy = (prob < MUTATION_RATE) ? ((agent->strategy + 1) % 2) : agent->strategy; // probability of switching the current strategy

    return 0;
}

#endif //_FLAMEGPU_FUNCTIONS
