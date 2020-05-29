
/*
 * FLAME GPU v 1.5.X for CUDA 9
 * Copyright University of Sheffield.
 * Original Author: Dr Paul Richmond (user contributions tracked on https://github.com/FLAMEGPU/FLAMEGPU)
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

/* 
	Smoothed Particle Hydrodynamics Demo
	Author: Matthew Leach
	Contact: M.Leach@sheffield.ac.uk

	Suggested things to try:
	
	Play with the viscosity to try fluids of different thickness - the simulation is
	stable for viscosity values between ~0.1 - 30

	Lower the fluid rest density to TODO see the 'FLAME' explode instead of fall

	Use the python script in the iterations folder to change the initial arrangement of the fluid particles
*/


#ifndef _FLAMEGPU_FUNCTIONS
#define _FLAMEGPU_FUNCTIONS

#include <header.h>

// Mathematic constants
#define PI 3.141593f

// Sim parameters
#define TIMESTEP 0.001f
#define SMOOTHING_LENGTH 0.040f
#define MAX_VELOCITY 4.0f

// Environemental parameters
#define RESTITUTION_COEFFICIENT 0.9f
#define HALF_BOUNDARY_WIDTH 0.5f

// Fluid properties
#define PARTICLE_MASS 0.001953125f
#define PRESSURE_COEFFICIENT 0.9f
#define FLUID_REST_DENSITY 1000.0f
#define MINIMUM_PARTICLE_DENSITY 0.0f
#define VISCOSITY 10.0f

//// Math helpers
inline __device__ float dot(glm::vec3 a, glm::vec3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ float length(glm::vec3 v) {
	return sqrt(dot(v, v));
}

//// Smoothing functions
__device__ float computeW(float distance, float smoothingLength) {
	return distance < smoothingLength ? (315.0f / (64.0f*PI*pow(smoothingLength, 9))) * pow((pow(smoothingLength, 2) - pow(distance, 2)), 3) : 0.0f;
}

__device__ glm::vec3 computeDelW(float distance, glm::vec3 difference, float smoothingLength) {
	return distance < smoothingLength ? -(45.0f / (PI*pow(smoothingLength, 6)))*(pow(smoothingLength - distance, 2))* (1.0f / distance) * difference : glm::vec3(0.0f);
}

__device__ float computeDelSqW(float distance, float smoothingLength) {
	return distance < smoothingLength ? (45.0f / (PI*pow(smoothingLength, 6))) * (smoothingLength - distance) : 0.0f;
}

__device__ float computeSurfaceTension(float distance, float smoothingLength) {
	if (distance > smoothingLength / 2.0f && distance <= smoothingLength)
		return 10 * (32.0f / (PI*pow(smoothingLength, 6))) * pow(smoothingLength - distance, 3) * pow(distance, 3);
	if (distance > 0 && distance <= smoothingLength / 2.0f)
		return (((64.0f / (PI*pow(smoothingLength, 6))) * 2 * pow(smoothingLength - distance, 3) * pow(distance, 3)) - pow(smoothingLength, 4) / 64.0f);
	return 0.0f;
}

/**
 * outputLocationVelocity FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_FluidParticle. This represents a single agent instance and can be modified directly.
 * @param location_messages Pointer to output message list of type xmachine_message_location_list. Must be passed as an argument to the add_location_message function.
 */
__FLAME_GPU_FUNC__ int outputLocation(xmachine_memory_FluidParticle* agent, xmachine_message_location_list* location_messages){
    
    add_location_message(location_messages, agent->id, agent->x, agent->y, agent->z);

    return 0;
}

/**
 * computeDensityPressure FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_FluidParticle. This represents a single agent instance and can be modified directly.
 * @param location_messages  location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_location_message and get_next_location_message functions.* @param density_pressure_messages Pointer to output message list of type xmachine_message_density_pressure_list. Must be passed as an argument to the add_density_pressure_message function.
 */
__FLAME_GPU_FUNC__ int computeDensityPressure(xmachine_memory_FluidParticle* agent, xmachine_message_location_list* location_messages, xmachine_message_location_PBM* partition_matrix, xmachine_message_density_pressure_list* density_pressure_messages){

	// Compute unmodified particle density
	float density = 0.0f;
    
    xmachine_message_location* current_message = get_first_location_message(location_messages, partition_matrix, agent->x, agent->y, agent->z);
	while (current_message)
	{
		// For each agent add weighted density contribution
		glm::vec3 diff = glm::vec3(current_message->x - agent->x, current_message->y - agent->y, current_message->z - agent->z);
		float distance = length(diff);
		float w = computeW(distance, SMOOTHING_LENGTH);
		density += PARTICLE_MASS * w;
		
		current_message = get_next_location_message(current_message, location_messages, partition_matrix);
	}

	// Set density and pressure for the particle - negative pressures are not used
	agent->density = max(density, MINIMUM_PARTICLE_DENSITY);
	agent->pressure = max(PRESSURE_COEFFICIENT * (agent->density - FLUID_REST_DENSITY), 0.0f);

	// Output location/density/pressure
    add_density_pressure_message(density_pressure_messages, agent->id, agent->density, agent->pressure, agent->x, agent->y, agent->z, agent->dx, agent->dy, agent->dz, agent->isStatic);

    return 0;
}

/**
 * computeForce FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_FluidParticle. This represents a single agent instance and can be modified directly.
 * @param density_pressure_messages  density_pressure_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_density_pressure_message and get_next_density_pressure_message functions.* @param force_messages Pointer to output message list of type xmachine_message_force_list. Must be passed as an argument to the add_force_message function.
 */
__FLAME_GPU_FUNC__ int computeForce(xmachine_memory_FluidParticle* agent, xmachine_message_density_pressure_list* density_pressure_messages, xmachine_message_density_pressure_PBM* partition_matrix){
    
	glm::vec3 force = glm::vec3(0.0f);

    xmachine_message_density_pressure* current_message = get_first_density_pressure_message(density_pressure_messages, partition_matrix, agent->x, agent->y, agent->z);
    while (current_message)
    {
		// For each other agent
		if (agent->id != current_message->id)
		{
			// Add weighted pressure contribution
			glm::vec3 diff = glm::vec3(current_message->x - agent->x, current_message->y - agent->y, current_message->z - agent->z);
			float distance = length(diff);			
			float weight = (agent->pressure + current_message->pressure) / (2.0f * current_message->density);
			glm::vec3 del_w = computeDelW(distance, diff, SMOOTHING_LENGTH);
			force += PARTICLE_MASS * weight * del_w;

			// Only add viscosity contribution from non-static particles
			if (current_message->isStatic == false)
			{
				// Add viscosity contribution
				glm::vec3 velocityDiff = glm::vec3(current_message->dx - agent->dx, current_message->dy - agent->dy, current_message->dz - agent->dz);
				glm::vec3 Vij = (VISCOSITY * PARTICLE_MASS / (agent->density * current_message->density)) * velocityDiff;
				float laplacian = computeDelSqW(distance, SMOOTHING_LENGTH);

				force += laplacian * Vij;
			}

			// Add surface tension
			force += computeSurfaceTension(distance, SMOOTHING_LENGTH) * diff;
		}
        
        current_message = get_next_density_pressure_message(current_message, density_pressure_messages, partition_matrix);
    }

	agent->fx = force.x;
	agent->fy = force.y;
	agent->fz = force.z;
 
    return 0;
}

/**
 * integrate FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_FluidParticle. This represents a single agent instance and can be modified directly.
 * @param force_messages  force_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_force_message and get_next_force_message functions.
 */
__FLAME_GPU_FUNC__ int integrate(xmachine_memory_FluidParticle* agent){
    
	// Integrate forces due to pressure and viscosity
	agent->dx += (agent->fx) * TIMESTEP;
	agent->dy += (agent->fy) * TIMESTEP;
	agent->dz += (agent->fz) * TIMESTEP ;

	// Gravity
	agent->dy -= 9.8f * TIMESTEP;

	// Cap velocity
	if (agent->dx < -MAX_VELOCITY)
		agent->dx = -MAX_VELOCITY;
	if (agent->dx > MAX_VELOCITY)
		agent->dx = MAX_VELOCITY;

	if (agent->dy < -MAX_VELOCITY)
		agent->dy = -MAX_VELOCITY;
	if (agent->dy > MAX_VELOCITY)
		agent->dy = MAX_VELOCITY;

	if (agent->dz < -MAX_VELOCITY)
		agent->dz = -MAX_VELOCITY;
	if (agent->dz > MAX_VELOCITY)
		agent->dz = MAX_VELOCITY;

	// Boundary conditions
	if (abs(agent->x + agent->dx*TIMESTEP) > HALF_BOUNDARY_WIDTH)
		agent->dx = -agent->dx*RESTITUTION_COEFFICIENT;
	if (abs(agent->y + agent->dy*TIMESTEP) > HALF_BOUNDARY_WIDTH)
		agent->dy = -agent->dy*RESTITUTION_COEFFICIENT;
	if (abs(agent->z + agent->dz*TIMESTEP) > HALF_BOUNDARY_WIDTH / 2.0f)
		agent->dz = -agent->dz*RESTITUTION_COEFFICIENT;

	// Integrate velocities
	agent->x = agent->x + agent->dx * TIMESTEP;
	agent->y = agent->y + agent->dy * TIMESTEP;
	agent->z = agent->z + agent->dz * TIMESTEP;
    
    return 0;
}



#endif //_FLAMEGPU_FUNCTIONS