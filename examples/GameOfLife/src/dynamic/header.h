
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
#define buffer_size_MAX 65536

//Maximum population size of xmachine_memory_cell
#define xmachine_memory_cell_MAX 65536
  
  
/* Message poulation size definitions */
//Maximum population size of xmachine_mmessage_state
#define xmachine_message_state_MAX 65536



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

/** struct xmachine_memory_cell
 * discrete valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_cell
{
    int state;    /**< X-machine memory variable state of type int.*/
    int x;    /**< X-machine memory variable x of type int.*/
    int y;    /**< X-machine memory variable y of type int.*/
};



/* Message structures */

/** struct xmachine_message_state
 * Discrete Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_state
{	
    /* Discrete Partitioning Variables */
    int2 _position;         /**< 2D position of message*/
    int2 _relative;         /**< 2D position of message relative to the agent (range +- radius) */  
      
    int state;        /**< Message variable state of type int.*/  
    int x;        /**< Message variable x of type int.*/  
    int y;        /**< Message variable y of type int.*/
};



/* Agent lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_memory_cell_list
 * discrete valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_cell_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_cell_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_cell_MAX];  /**< Used during parallel prefix sum */
    
    int state [xmachine_memory_cell_MAX];    /**< X-machine memory variable list state of type int.*/
    int x [xmachine_memory_cell_MAX];    /**< X-machine memory variable list x of type int.*/
    int y [xmachine_memory_cell_MAX];    /**< X-machine memory variable list y of type int.*/
};



/* Message lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_message_state_list
 * Discrete Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_state_list
{
    int state [xmachine_message_state_MAX];    /**< Message memory variable list state of type int.*/
    int x [xmachine_message_state_MAX];    /**< Message memory variable list x of type int.*/
    int y [xmachine_message_state_MAX];    /**< Message memory variable list y of type int.*/
    
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
 * output_state FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_cell. This represents a single agent instance and can be modified directly.
 * @param state_messages Pointer to output message list of type xmachine_message_state_list. Must be passed as an argument to the add_state_message function ??.
 */
__FLAME_GPU_FUNC__ int output_state(xmachine_memory_cell* agent, xmachine_message_state_list* state_messages);

/**
 * update_state FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_cell. This represents a single agent instance and can be modified directly.
 * @param state_messages  state_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_state_message and get_next_state_message functions.
 */
__FLAME_GPU_FUNC__ int update_state(xmachine_memory_cell* agent, xmachine_message_state_list* state_messages);

  
/* Message Function Prototypes for Discrete Partitioned state message implemented in FLAMEGPU_Kernels */

/** add_state_message
 * Function for all types of message partitioning
 * Adds a new state agent to the xmachine_memory_state_list list using a linear mapping
 * @param agents	xmachine_memory_state_list agent list
 * @param state	message variable of type int
 * @param x	message variable of type int
 * @param y	message variable of type int
 */
 template <int AGENT_TYPE>
 __FLAME_GPU_FUNC__ void add_state_message(xmachine_message_state_list* state_messages, int state, int x, int y);
 
/** get_first_state_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memeory or texture cache implementation depending on AGENT_TYPE
 * @param state_messages message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_state * get_first_state_message(xmachine_message_state_list* state_messages, int agentx, int agent_y);

/** get_next_state_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memeory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param state_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_state * get_next_state_message(xmachine_message_state* current, xmachine_message_state_list* state_messages);
  
  
  
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
 * @param h_cells Pointer to agent list on the host
 * @param d_cells Pointer to agent list on the GPU device
 * @param h_xmachine_memory_cell_count Pointer to agent counter
 */
extern "C" void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_cell_list* h_cells_default, xmachine_memory_cell_list* d_cells_default, int h_xmachine_memory_cell_default_count);


/** readInitialStates
 * Reads the current agent data fromt he device and saves it to XML
 * @param	inputpath	file path to XML file used for input of agent data
 * @param h_cells Pointer to agent list on the host
 * @param h_xmachine_memory_cell_count Pointer to agent counter
 */
extern "C" void readInitialStates(char* inputpath, xmachine_memory_cell_list* h_cells, int* h_xmachine_memory_cell_count);


/* Return functions used by external code to get agent data from device */

    
/** get_agent_cell_MAX_count
 * Gets the max agent count for the cell agent type 
 * @return		the maximum cell agent count
 */
extern "C" int get_agent_cell_MAX_count();



/** get_agent_cell_default_count
 * Gets the agent count for the cell agent type in state default
 * @return		the current cell agent count in state default
 */
extern "C" int get_agent_cell_default_count();

/** reset_default_count
 * Resets the agent count of the cell in state default to 0. This is usefull for interacting with some visualisations.
 */
extern "C" void reset_cell_default_count();

/** get_device_cell_default_agents
 * Gets a pointer to xmachine_memory_cell_list on the GPU device
 * @return		a xmachine_memory_cell_list on the GPU device
 */
extern "C" xmachine_memory_cell_list* get_device_cell_default_agents();

/** get_host_cell_default_agents
 * Gets a pointer to xmachine_memory_cell_list on the CPU host
 * @return		a xmachine_memory_cell_list on the CPU host
 */
extern "C" xmachine_memory_cell_list* get_host_cell_default_agents();


/** get_cell_population_width
 * Gets an int value representing the xmachine_memory_cell population width.
 * @return		xmachine_memory_cell population width
 */
extern "C" int get_cell_population_width();

  
  
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

