
	/*
	* FLAME GPU v 1.4.0 for CUDA 6
	* Copyright 2015 University of Sheffield.
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

#ifndef __HEADER
#define __HEADER

/* General standard definitions */
//Threads per block (agents per block)
#define THREADS_PER_TILE 64
//Definition for any agent function or helper function
#define __FLAME_GPU_FUNC__ __device__
//Definition for a function used to initialise environment variables
#define __FLAME_GPU_INIT_FUNC__

#define USE_CUDA_STREAMS
#define FAST_ATOMIC_SORTING

typedef unsigned int uint;

	

/* Agent population size definifions must be a multiple of THREADS_PER_TILE (defualt 64) */
//Maximum buffer size (largest agent buffer size)
#define buffer_size_MAX 1048576

//Maximum population size of xmachine_memory_agent
#define xmachine_memory_agent_MAX 1048576
  
  
/* Message poulation size definitions */
//Maximum population size of xmachine_mmessage_cell_state
#define xmachine_message_cell_state_MAX 1048576

//Maximum population size of xmachine_mmessage_movement_request
#define xmachine_message_movement_request_MAX 1048576

//Maximum population size of xmachine_mmessage_movement_response
#define xmachine_message_movement_response_MAX 1048576



/* Spatial partitioning grid size definitions */
  
  
/* enum types */

/**
 * MESSAGE_OUTPUT used for all continuous messaging
 */
enum MESSAGE_OUTPUT{
	single_message,
	optional_message,
};

/**
 * AGENT_TYPE used for templates device message functions
 */
enum AGENT_TYPE{
	CONTINUOUS,
	DISCRETE_2D
};


/* Agent structures */

/** struct xmachine_memory_agent
 * discrete valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_agent
{
    int location_id;    /**< X-machine memory variable location_id of type int.*/
    int agent_id;    /**< X-machine memory variable agent_id of type int.*/
    int state;    /**< X-machine memory variable state of type int.*/
    int sugar_level;    /**< X-machine memory variable sugar_level of type int.*/
    int metabolism;    /**< X-machine memory variable metabolism of type int.*/
    int env_sugar_level;    /**< X-machine memory variable env_sugar_level of type int.*/
};



/* Message structures */

/** struct xmachine_message_cell_state
 * Discrete Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_cell_state
{	
    /* Discrete Partitioning Variables */
    int2 _position;         /**< 2D position of message*/
    int2 _relative;         /**< 2D position of message relative to the agent (range +- radius) */  
      
    int location_id;        /**< Message variable location_id of type int.*/  
    int state;        /**< Message variable state of type int.*/  
    int env_sugar_level;        /**< Message variable env_sugar_level of type int.*/
};

/** struct xmachine_message_movement_request
 * Discrete Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_movement_request
{	
    /* Discrete Partitioning Variables */
    int2 _position;         /**< 2D position of message*/
    int2 _relative;         /**< 2D position of message relative to the agent (range +- radius) */  
      
    int agent_id;        /**< Message variable agent_id of type int.*/  
    int location_id;        /**< Message variable location_id of type int.*/  
    int sugar_level;        /**< Message variable sugar_level of type int.*/  
    int metabolism;        /**< Message variable metabolism of type int.*/
};

/** struct xmachine_message_movement_response
 * Discrete Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_movement_response
{	
    /* Discrete Partitioning Variables */
    int2 _position;         /**< 2D position of message*/
    int2 _relative;         /**< 2D position of message relative to the agent (range +- radius) */  
      
    int location_id;        /**< Message variable location_id of type int.*/  
    int agent_id;        /**< Message variable agent_id of type int.*/
};



/* Agent lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_memory_agent_list
 * discrete valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_agent_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_agent_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_agent_MAX];  /**< Used during parallel prefix sum */
    
    int location_id [xmachine_memory_agent_MAX];    /**< X-machine memory variable list location_id of type int.*/
    int agent_id [xmachine_memory_agent_MAX];    /**< X-machine memory variable list agent_id of type int.*/
    int state [xmachine_memory_agent_MAX];    /**< X-machine memory variable list state of type int.*/
    int sugar_level [xmachine_memory_agent_MAX];    /**< X-machine memory variable list sugar_level of type int.*/
    int metabolism [xmachine_memory_agent_MAX];    /**< X-machine memory variable list metabolism of type int.*/
    int env_sugar_level [xmachine_memory_agent_MAX];    /**< X-machine memory variable list env_sugar_level of type int.*/
};



/* Message lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_message_cell_state_list
 * Discrete Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_cell_state_list
{
    int location_id [xmachine_message_cell_state_MAX];    /**< Message memory variable list location_id of type int.*/
    int state [xmachine_message_cell_state_MAX];    /**< Message memory variable list state of type int.*/
    int env_sugar_level [xmachine_message_cell_state_MAX];    /**< Message memory variable list env_sugar_level of type int.*/
    
};

/** struct xmachine_message_movement_request_list
 * Discrete Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_movement_request_list
{
    int agent_id [xmachine_message_movement_request_MAX];    /**< Message memory variable list agent_id of type int.*/
    int location_id [xmachine_message_movement_request_MAX];    /**< Message memory variable list location_id of type int.*/
    int sugar_level [xmachine_message_movement_request_MAX];    /**< Message memory variable list sugar_level of type int.*/
    int metabolism [xmachine_message_movement_request_MAX];    /**< Message memory variable list metabolism of type int.*/
    
};

/** struct xmachine_message_movement_response_list
 * Discrete Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_movement_response_list
{
    int location_id [xmachine_message_movement_response_MAX];    /**< Message memory variable list location_id of type int.*/
    int agent_id [xmachine_message_movement_response_MAX];    /**< Message memory variable list agent_id of type int.*/
    
};



/* Spatialy Partitioned Message boundary Matrices */



/* Random */
/** struct RNG_rand48 
 *	structure used to hold list seeds
 */
struct RNG_rand48
{
  uint2 A, C;
  uint2 seeds[buffer_size_MAX];
};


/* Random Functions (usable in agent functions) implemented in FLAMEGPU_Kernels */

/**
 * Templated random function using a DISCRETE_2D template calculates the agent index using a 2D block
 * which requires extra processing but will work for CONTINUOUS agents. Using a CONTINUOUS template will
 * not work for DISCRETE_2D agent.
 * @param	rand48	an RNG_rand48 struct which holds the seeds sued to generate a random number on the GPU
 * @return			returns a random float value
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ float rnd(RNG_rand48* rand48);
/**
 * Non templated random function calls the templated version with DISCRETE_2D which will work in either case
 * @param	rand48	an RNG_rand48 struct which holds the seeds sued to generate a random number on the GPU
 * @return			returns a random float value
 */
__FLAME_GPU_FUNC__ float rnd(RNG_rand48* rand48);

/* Agent function prototypes */

/**
 * metabolise_and_growback FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int metabolise_and_growback(xmachine_memory_agent* agent);

/**
 * output_cell_state FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param cell_state_messages Pointer to output message list of type xmachine_message_cell_state_list. Must be passed as an argument to the add_cell_state_message function ??.
 */
__FLAME_GPU_FUNC__ int output_cell_state(xmachine_memory_agent* agent, xmachine_message_cell_state_list* cell_state_messages);

/**
 * movement_request FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param cell_state_messages  cell_state_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_cell_state_message and get_next_cell_state_message functions.* @param movement_request_messages Pointer to output message list of type xmachine_message_movement_request_list. Must be passed as an argument to the add_movement_request_message function ??.
 */
__FLAME_GPU_FUNC__ int movement_request(xmachine_memory_agent* agent, xmachine_message_cell_state_list* cell_state_messages, xmachine_message_movement_request_list* movement_request_messages);

/**
 * movement_response FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param movement_request_messages  movement_request_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_movement_request_message and get_next_movement_request_message functions.* @param movement_response_messages Pointer to output message list of type xmachine_message_movement_response_list. Must be passed as an argument to the add_movement_response_message function ??.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an arument to the rand48 function for genertaing random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int movement_response(xmachine_memory_agent* agent, xmachine_message_movement_request_list* movement_request_messages, xmachine_message_movement_response_list* movement_response_messages, RNG_rand48* rand48);

/**
 * movement_transaction FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param movement_response_messages  movement_response_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_movement_response_message and get_next_movement_response_message functions.
 */
__FLAME_GPU_FUNC__ int movement_transaction(xmachine_memory_agent* agent, xmachine_message_movement_response_list* movement_response_messages);

  
/* Message Function Prototypes for Discrete Partitioned cell_state message implemented in FLAMEGPU_Kernels */

/** add_cell_state_message
 * Function for all types of message partitioning
 * Adds a new cell_state agent to the xmachine_memory_cell_state_list list using a linear mapping
 * @param agents	xmachine_memory_cell_state_list agent list
 * @param location_id	message variable of type int
 * @param state	message variable of type int
 * @param env_sugar_level	message variable of type int
 */
 template <int AGENT_TYPE>
 __FLAME_GPU_FUNC__ void add_cell_state_message(xmachine_message_cell_state_list* cell_state_messages, int location_id, int state, int env_sugar_level);
 
/** get_first_cell_state_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memeory or texture cache implementation depending on AGENT_TYPE
 * @param cell_state_messages message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_cell_state * get_first_cell_state_message(xmachine_message_cell_state_list* cell_state_messages, int agentx, int agent_y);

/** get_next_cell_state_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memeory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param cell_state_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_cell_state * get_next_cell_state_message(xmachine_message_cell_state* current, xmachine_message_cell_state_list* cell_state_messages);

  
/* Message Function Prototypes for Discrete Partitioned movement_request message implemented in FLAMEGPU_Kernels */

/** add_movement_request_message
 * Function for all types of message partitioning
 * Adds a new movement_request agent to the xmachine_memory_movement_request_list list using a linear mapping
 * @param agents	xmachine_memory_movement_request_list agent list
 * @param agent_id	message variable of type int
 * @param location_id	message variable of type int
 * @param sugar_level	message variable of type int
 * @param metabolism	message variable of type int
 */
 template <int AGENT_TYPE>
 __FLAME_GPU_FUNC__ void add_movement_request_message(xmachine_message_movement_request_list* movement_request_messages, int agent_id, int location_id, int sugar_level, int metabolism);
 
/** get_first_movement_request_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memeory or texture cache implementation depending on AGENT_TYPE
 * @param movement_request_messages message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_movement_request * get_first_movement_request_message(xmachine_message_movement_request_list* movement_request_messages, int agentx, int agent_y);

/** get_next_movement_request_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memeory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param movement_request_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_movement_request * get_next_movement_request_message(xmachine_message_movement_request* current, xmachine_message_movement_request_list* movement_request_messages);

  
/* Message Function Prototypes for Discrete Partitioned movement_response message implemented in FLAMEGPU_Kernels */

/** add_movement_response_message
 * Function for all types of message partitioning
 * Adds a new movement_response agent to the xmachine_memory_movement_response_list list using a linear mapping
 * @param agents	xmachine_memory_movement_response_list agent list
 * @param location_id	message variable of type int
 * @param agent_id	message variable of type int
 */
 template <int AGENT_TYPE>
 __FLAME_GPU_FUNC__ void add_movement_response_message(xmachine_message_movement_response_list* movement_response_messages, int location_id, int agent_id);
 
/** get_first_movement_response_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memeory or texture cache implementation depending on AGENT_TYPE
 * @param movement_response_messages message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_movement_response * get_first_movement_response_message(xmachine_message_movement_response_list* movement_response_messages, int agentx, int agent_y);

/** get_next_movement_response_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memeory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param movement_response_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_movement_response * get_next_movement_response_message(xmachine_message_movement_response* current, xmachine_message_movement_response_list* movement_response_messages);
  
  
  
/* Agent Function Prototypes implemented in FLAMEGPU_Kernels */


  
/* Simulation function prototypes implemented in simulation.cu */

/** initialise
 * Initialise the simulation. Allocated host and device memory. Reads the initial agent configuration from XML.
 * @param input	XML file path for agent initial configuration
 */
extern "C" void initialise(char * input);

/** cleanup
 * Function cleans up any memory allocations on the host and device
 */
extern "C" void cleanup();

/** singleIteration
 *	Performs a single itteration of the simulation. I.e. performs each agent function on each function layer in the correct order.
 */
extern "C" void singleIteration();

/** saveIterationData
 * Reads the current agent data fromt he device and saves it to XML
 * @param	outputpath	file path to XML file used for output of agent data
 * @param	itteration_number
 * @param h_agents Pointer to agent list on the host
 * @param d_agents Pointer to agent list on the GPU device
 * @param h_xmachine_memory_agent_count Pointer to agent counter
 */
extern "C" void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_agent_list* h_agents_default, xmachine_memory_agent_list* d_agents_default, int h_xmachine_memory_agent_default_count);


/** readInitialStates
 * Reads the current agent data fromt he device and saves it to XML
 * @param	inputpath	file path to XML file used for input of agent data
 * @param h_agents Pointer to agent list on the host
 * @param h_xmachine_memory_agent_count Pointer to agent counter
 */
extern "C" void readInitialStates(char* inputpath, xmachine_memory_agent_list* h_agents, int* h_xmachine_memory_agent_count);


/* Return functions used by external code to get agent data from device */

    
/** get_agent_agent_MAX_count
 * Gets the max agent count for the agent agent type 
 * @return		the maximum agent agent count
 */
extern "C" int get_agent_agent_MAX_count();



/** get_agent_agent_default_count
 * Gets the agent count for the agent agent type in state default
 * @return		the current agent agent count in state default
 */
extern "C" int get_agent_agent_default_count();

/** reset_default_count
 * Resets the agent count of the agent in state default to 0. This is usefull for interacting with some visualisations.
 */
extern "C" void reset_agent_default_count();

/** get_device_agent_default_agents
 * Gets a pointer to xmachine_memory_agent_list on the GPU device
 * @return		a xmachine_memory_agent_list on the GPU device
 */
extern "C" xmachine_memory_agent_list* get_device_agent_default_agents();

/** get_host_agent_default_agents
 * Gets a pointer to xmachine_memory_agent_list on the CPU host
 * @return		a xmachine_memory_agent_list on the CPU host
 */
extern "C" xmachine_memory_agent_list* get_host_agent_default_agents();


/** get_agent_population_width
 * Gets an int value representing the xmachine_memory_agent population width.
 * @return		xmachine_memory_agent population width
 */
extern "C" int get_agent_population_width();

  
  
/* global constant variables */


/** getMaximumBound
 * Returns the maximum agent positions determined from the initial loading of agents
 * @return 	a three component float indicating the maximum x, y and z positions of all agents
 */
float3 getMaximumBounds();

/** getMinimumBounds
 * Returns the minimum agent positions determined from the initial loading of agents
 * @return 	a three component float indicating the minimum x, y and z positions of all agents
 */
float3 getMinimumBounds();
    
    
#ifdef VISUALISATION
/** initVisualisation
 * Prototype for method which initialises the visualisation. Must be implemented in seperate file
 * @param argc	the argument count from the main function used with GLUT
 * @param argv	the argument values fromt the main function used with GLUT
 */
extern "C" void initVisualisation();

extern "C" void runVisualisation();


#endif

#endif //__HEADER

