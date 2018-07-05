
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


#ifndef _FLAMEGPU_FUNCTIONS
#define _FLAMEGPU_FUNCTIONS

#include <header.h>

#define ALIVE 0
#define DEAD 1
#define NUM_FLAMEGPU_COLOURS 8 

// Random float generation (host)
float randomFloat(const float min, const float max){
	if(min == max){
		return min;
	} else {
		return min + (rand() / (float) RAND_MAX) * (max - min);
	}
}

// Random float generation (device)
float randomUnsigned(const unsigned int min, const unsigned int max){
	if(min == max){
		return min;
	} else {
		return (rand() % (max -  min)) + min;
	}
}

// Prototypes for utility functions.
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ float xPositionFromNetwork(unsigned int edge, float distance);
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ float yPositionFromNetwork(unsigned int edge, float distance);
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ float zPositionFromNetwork(unsigned int edge, float distance);



/**
 * initialiseHost FLAMEGPU Init function
 * Seed the RNG and raise any possible warnings about invalid configurations.
 */
__FLAME_GPU_INIT_FUNC__ void initialiseHost(){
	// Seed the host random number generator.
	srand(*get_SEED());

	bool error = false;
	// Error if bad initial population.
	if(*get_INIT_POPULATION() <= 0 ){
		error = true;
		fprintf(stderr, "Error: Invalid INIT_POPULATION value %u. Must be > 0.\n", *get_INIT_POPULATION());
	}
	
	// Error if bad minimum speed.
	if(*get_PARAM_MIN_SPEED() <= 0.0f){
		error = true;
		fprintf(stderr, "Error: PARAM_MIN_SPEED must be greater than equal to 0.\n");
	}

	// Warn if max speed is less than min, in which case use min.
	if(*get_PARAM_MAX_SPEED() < *get_PARAM_MIN_SPEED()){
		fprintf(stderr, "Warning: PARAM_MAX_SPEED less than PARAM_MIN_SPEED. Using %f\n", *get_PARAM_MIN_SPEED());
		float new_MAX_SPEED = *get_PARAM_MIN_SPEED();
		set_PARAM_MAX_SPEED(&new_MAX_SPEED);
	}

	// Check that a graph has been loaded correctly, error if not. 
	unsigned int numVertices = get_staticGraph_network_vertex_count();
	unsigned int numEdges = get_staticGraph_network_edge_count();

	if(numVertices == 0 || numEdges == 0){
		error = true;
		fprintf(stderr, "Error: 0 vertices or edges in network (%u v, %u e)\n", numVertices, numEdges);
	}

	// Error if the input number of agents means that the graph will be over-capacity.
	unsigned int maximumGraphCapacity = 0;
	unsigned int prevMaximumGraphCapacity = 0;
	for(unsigned int e = 0; e < numEdges; e++){
		prevMaximumGraphCapacity = maximumGraphCapacity;
		maximumGraphCapacity += get_staticGraph_network_edge_capacity(e);
		// If we have overflowed we can break - there is plenty of capacity.
		if(maximumGraphCapacity < prevMaximumGraphCapacity){
			maximumGraphCapacity = UINT_MAX;
			break;
		}
	}

	if(maximumGraphCapacity < *get_INIT_POPULATION()){
		error = true;
		fprintf(stderr, "Error: Initial population is greater than total graph capacity (%u > %u)\n", *get_INIT_POPULATION(), maximumGraphCapacity);
	} else {
		fprintf(stdout, "Maximum capacity for graph = %u\n", maximumGraphCapacity);
	}

	// Flush outputs
	fflush(stdout);
	fflush(stderr);

	// Exit if any errors. This is not a graceful exit (i.e no resetting of device or deallocation of memory)
	if(error){
		fprintf(stdout, "Aborting.\n");
		fflush(stdout);
		exit(EXIT_FAILURE);
	}
}

/**
 * generateAgents FLAMEGPU Init function
 * Generate the initial population of agents based on parameters passed as environmental variables
 */
__FLAME_GPU_INIT_FUNC__ void generateAgents(){

	// If there is not an initial population from the initial states file
	if(get_agent_Agent_default_count() == 0){

		// Allocate memory to create the initial population
		unsigned int initialPopulation = *get_INIT_POPULATION();
		xmachine_memory_Agent ** hostAgentArray = h_allocate_agent_Agent_array(initialPopulation);


		// Prepare values to allow equal distribution of agents to edges, within the contstraints of the network. 
		unsigned int nextEdge = 0;
		unsigned int pass = 0;

		// Create the initial population
		for(unsigned int i = 0; i < initialPopulation; i++){
			// Get a pointer to the relevant part of the host agent data structure
			xmachine_memory_Agent * agent = hostAgentArray[i];

			// Set the agent id.
			agent->id = i;

			// Determine the agents location within the graph, within capacity for the edge.
			unsigned int edge = nextEdge;
			while(pass >= get_staticGraph_network_edge_capacity(edge)){
				edge += 1;
				if(edge >= get_staticGraph_network_edge_count()){
					edge = 0;
					pass++;
				}
			}
			nextEdge = edge + 1;
			if(nextEdge >= get_staticGraph_network_edge_count()){
				nextEdge = 0;
				pass++;
			}
			
			agent->currentEdge = edge;
			// Position is randomly selected between 0.0 and the length of the edge.
			agent->position = randomFloat(0.0f, get_staticGraph_network_edge_length(edge));

			// Determine the agents next edge based on the number of edges leaving the next vertex.
			unsigned int destination = get_staticGraph_network_edge_destination(edge);
			unsigned int edgeCount = get_staticGraph_network_vertex_num_edges(destination);
			if(edgeCount > 0){
				agent->nextEdge = get_staticGraph_network_vertex_first_edge_index(destination) + randomUnsigned(0, edgeCount-1);
			} else {
				agent->nextEdge = get_staticGraph_network_edge_count();
			}

			// Determine the agents speed between the min and max.
			agent->speed = randomFloat(*get_PARAM_MIN_SPEED(), *get_PARAM_MAX_SPEED());

			// Set some default values
			agent->nextEdgeRemainingCapacity = 0;
			agent->hasIntent = false;
			agent->distanceTravelled = 0;
			agent->blockedIterationCount = 0;

			// Store the world location of the agent
			agent->x = xPositionFromNetwork(agent->currentEdge, agent->position);
			agent->y = yPositionFromNetwork(agent->currentEdge, agent->position);
			agent->z = zPositionFromNetwork(agent->currentEdge, agent->position);

			// Set the colour, rotating through the available colours in default visualisation
			agent->colour = i % NUM_FLAMEGPU_COLOURS;
		}

		// Push the initial population to the device in the default state
		h_add_agents_Agent_default(hostAgentArray, initialPopulation);

		// Deallocate memory for creating the initial population
		h_free_agent_Agent_array(&hostAgentArray, initialPopulation);
		hostAgentArray = nullptr;
	}

}

/**
 * exitFunc FLAMEGPU Exit function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_EXIT_FUNC__ void exitFunc(){
	// Get the current population
	unsigned int currentPopulation = get_agent_Agent_default_count();
	// Get the total distance travelled by alive agents and calculate the average
	float totalDistanceTravelled = reduce_Agent_default_distanceTravelled_variable();
	float averageDistanceTravelled = totalDistanceTravelled / (float) currentPopulation;

	// Get the total distance travelled by alive agents and calculate the average
	unsigned int blockedIterationCount = reduce_Agent_default_blockedIterationCount_variable();
	float averageBlockedIterationCount = blockedIterationCount / (float) currentPopulation;

	fprintf(stdout, "Average Distance Travelled      = %f\n", averageDistanceTravelled);
	fprintf(stdout, "Average Blocked Iteration Count = %f\n", averageBlockedIterationCount);
}

/**
 * output_location FLAMEGPU Agent Function
 * Output a location message for each agent, stating where the agent exists within the network.
 */
__FLAME_GPU_FUNC__ int output_location(xmachine_memory_Agent* agent, xmachine_message_location_list* location_messages){
   
   	// Add the location message    
    add_location_message(location_messages, agent->id, agent->currentEdge, agent->position);

    // Reset any per-iteration values (short term memory)
    agent->hasIntent = false;
    agent->nextEdgeRemainingCapacity = 0;
    
    return 0;
}

/**
 * read_locations FLAMEGPU Agent Function
 * Iterate list of location messages to determine 
 */
__FLAME_GPU_FUNC__ int read_locations(xmachine_memory_Agent* agent, xmachine_message_location_list* location_messages, xmachine_message_location_bounds* message_bounds, xmachine_message_intent_list* intent_messages){
    
	// Iterate the list of messages from agents on the target graph element, to count the number of agents there. 
	unsigned int nextEdgePopulation = 0;
    xmachine_message_location* current_message = get_first_location_message(location_messages, message_bounds, agent->nextEdge);
    while (current_message)
    {
    	// Increment the counter
        nextEdgePopulation += 1;
        // Get the next message
        current_message = get_next_location_message(current_message, location_messages, message_bounds);
    }
    
    // Calculate the remaining capacity of the next edge.
    unsigned int nextEdgeCapacity = get_staticGraph_network_edge_capacity(agent->nextEdge);
    agent->nextEdgeRemainingCapacity = nextEdgePopulation <= nextEdgeCapacity ? nextEdgeCapacity - nextEdgePopulation : 0;

    // And calculate if the agent will need to change edge.
    float nextPosition = agent->position + agent->speed;
    bool edgeTransitionRequired = nextPosition > get_staticGraph_network_edge_length(agent->currentEdge);

    // If the agent is going to change edge, and there is capacity, declare intent.
    agent->hasIntent = edgeTransitionRequired && agent->nextEdgeRemainingCapacity > 0;

    if(agent->hasIntent){
	    add_intent_message(intent_messages, agent->id, agent->nextEdge);
    }
    
    return 0;
}

/**
 * resolve_intent FLAMEGPU Agent Function
 * Using the list of intent messages determine if the current individual is allowed to change edge. If an edge change occurs, update the edge and calculate a new edge.
 * If the edge being completed is a final edge and the agent is past the edge, it must terminate.
 */
__FLAME_GPU_FUNC__ int resolve_intent(xmachine_memory_Agent* agent, xmachine_message_intent_list* intent_messages, xmachine_message_intent_bounds* message_bounds, RNG_rand48* rand48){

    // Iterate intent messages for the edge, to determine if the current agent has right of way to move
	unsigned int nextEdgeIntentCount = 0;
	unsigned int nextEdgeLowerIdCount = 0;
	// @future - use time waiting rather than ID / in conjunction with id?

    xmachine_message_intent* current_message = get_first_intent_message(intent_messages, message_bounds, agent->nextEdge);
    while (current_message)
    {
        nextEdgeIntentCount += 1;
        if(current_message->id < agent->id){
        	nextEdgeLowerIdCount += 1;
        }
        
        current_message = get_next_intent_message(current_message, intent_messages, message_bounds);
    }
    

    bool canMove = nextEdgeLowerIdCount < agent->nextEdgeRemainingCapacity;
    // If we could change edge, but there is no next edge the agent needs to leave the simulation.
    if(canMove && agent->nextEdge >= get_staticGraph_network_edge_count()){
    	return DEAD;
    // If the remaining capacity for the target edge is more than the lowerIdCount we can move. 
    } else if(canMove){
    	unsigned int prevEdge = agent->currentEdge;
    	agent->currentEdge = agent->nextEdge;

    	// Set the new next element.
    	unsigned int destination = get_staticGraph_network_edge_destination(agent->currentEdge);
    	unsigned int edgeCount = get_staticGraph_network_vertex_num_edges(destination);
    	if(edgeCount > 0){
	    	unsigned int randomEdge = (unsigned int)round(rnd<CONTINUOUS>(rand48) * (edgeCount - 1));
	    	agent->nextEdge = get_staticGraph_network_vertex_first_edge_index(destination) + randomEdge;
	    } else {
	    	// If the edge is a terminating edge, store the an invalid edge id.
	    	agent->nextEdge = get_staticGraph_network_edge_count();
	    }
    	// Update the position to be a negative value. 
    	agent->position = agent->position - get_staticGraph_network_edge_length(prevEdge);
    }
    
    return ALIVE;
}

/**
 * move FLAMEGPU Agent Function
 * Update the position of the agent on the current edge, and the world coordinates for the agent.
 */
__FLAME_GPU_FUNC__ int move(xmachine_memory_Agent* agent){

	// Move along the current edge, up to the end of the edge. 
	float edgeLength = get_staticGraph_network_edge_length(agent->currentEdge);
	float newPositon = min(agent->position + agent->speed, edgeLength);
	float distanceTravelledThisIteration = newPositon - agent->position;

	// Update the agent position.
	agent->position = newPositon;
	agent->distanceTravelled += distanceTravelledThisIteration;
	if(distanceTravelledThisIteration == 0.0f){
		agent->blockedIterationCount += 1;
	}
	agent->x = xPositionFromNetwork(agent->currentEdge, agent->position);
	agent->y = yPositionFromNetwork(agent->currentEdge, agent->position);
	agent->z = zPositionFromNetwork(agent->currentEdge, agent->position);

    return 0;
}


// Get the X co ordinate for a position along an edge
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ float xPositionFromNetwork(unsigned int edge, float distance){
	// Ensure that the edge index is valid
	if(edge > get_staticGraph_network_edge_count()){
		return 0.0f;
	}
	// Ensure that the distance is positive
	if(distance < 0.0f){
		return 0.0f;
	}

	// Get the X value for the start and end of the edge
	unsigned int v0 = get_staticGraph_network_edge_source(edge);
	float x0 = get_staticGraph_network_vertex_x(v0);
	unsigned int v1 = get_staticGraph_network_edge_destination(edge);
	float x1 = get_staticGraph_network_vertex_x(v1);
	

	// If x0 and x1 are the same, no need to lerp
	if(x0 == x1){
		return x0;
	} else {
		// Calculate how far along the edge we are, based on the distance and length of the edge
		float fractionAlong = distance / get_staticGraph_network_edge_length(edge);

		// LERP to find X value, in world space / units
		return x0 + (fractionAlong * (x1-x0));
	}
}
// Get the Y co ordinate for a position along an edge
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ float yPositionFromNetwork(unsigned int edge, float distance){
	// Ensure that the edge index is valid
	if(edge > get_staticGraph_network_edge_count()){
		return 0.0f;
	}
	// Ensure that the distance is positive
	if(distance < 0.0f){
		return 0.0f;
	}

	// Get the X value for the start and end of the edge
	unsigned int v0 = get_staticGraph_network_edge_source(edge);
	float y0 = get_staticGraph_network_vertex_y(v0);
	unsigned int v1 = get_staticGraph_network_edge_destination(edge);
	float y1 = get_staticGraph_network_vertex_y(v1);
	

	// If y0 and y1 are the same, no need to lerp
	if(y0 == y1){
		return y0;
	} else {
		// Calculate how far along the edge we are, based on the distance and length of the edge
		float fractionAlong = distance / get_staticGraph_network_edge_length(edge);

		// LERP to find X value, in world space / units
		return y0 + (fractionAlong * (y1-y0));
	}
}
// Get the Z co ordinate for a position along an edge
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ float zPositionFromNetwork(unsigned int edge, float distance){
	// Ensure that the edge index is valid
	if(edge > get_staticGraph_network_edge_count()){
		return 0.0f;
	}
	// Ensure that the distance is positive
	if(distance < 0.0f){
		return 0.0f;
	}

	// Get the X value for the start and end of the edge
	unsigned int v0 = get_staticGraph_network_edge_source(edge);
	float z0 = get_staticGraph_network_vertex_z(v0);
	unsigned int v1 = get_staticGraph_network_edge_destination(edge);
	float z1 = get_staticGraph_network_vertex_z(v1);
	

	// If z0 and z1 are the same, no need to lerp
	if(z0 == z1){
		return z0;
	} else {
		// Calculate how far along the edge we are, based on the distance and length of the edge
		float fractionAlong = distance / get_staticGraph_network_edge_length(edge);

		// LERP to find X value, in world space / units
		return z0 + (fractionAlong * (z1-z0));
	}
}

#endif //_FLAMEGPU_FUNCTIONS
