
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

//Maximum population size of xmachine_memory_agent
#define xmachine_memory_agent_MAX 65536

//Maximum population size of xmachine_memory_navmap
#define xmachine_memory_navmap_MAX 65536
  
  
/* Message poulation size definitions */
//Maximum population size of xmachine_mmessage_pedestrian_location
#define xmachine_message_pedestrian_location_MAX 65536

//Maximum population size of xmachine_mmessage_navmap_cell
#define xmachine_message_navmap_cell_MAX 65536



/* Spatial partitioning grid size definitions */
//xmachine_message_pedestrian_location partition grid size (gridDim.X*gridDim.Y*gridDim.Z)
#define xmachine_message_pedestrian_location_grid_size 6400
  
  
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
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_agent
{
    float x;    /**< X-machine memory variable x of type float.*/
    float y;    /**< X-machine memory variable y of type float.*/
    float velx;    /**< X-machine memory variable velx of type float.*/
    float vely;    /**< X-machine memory variable vely of type float.*/
    float steer_x;    /**< X-machine memory variable steer_x of type float.*/
    float steer_y;    /**< X-machine memory variable steer_y of type float.*/
    float height;    /**< X-machine memory variable height of type float.*/
    int exit_no;    /**< X-machine memory variable exit_no of type int.*/
    float speed;    /**< X-machine memory variable speed of type float.*/
    int lod;    /**< X-machine memory variable lod of type int.*/
    float animate;    /**< X-machine memory variable animate of type float.*/
    int animate_dir;    /**< X-machine memory variable animate_dir of type int.*/
};

/** struct xmachine_memory_navmap
 * discrete valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_navmap
{
    int x;    /**< X-machine memory variable x of type int.*/
    int y;    /**< X-machine memory variable y of type int.*/
    int exit_no;    /**< X-machine memory variable exit_no of type int.*/
    float height;    /**< X-machine memory variable height of type float.*/
    float collision_x;    /**< X-machine memory variable collision_x of type float.*/
    float collision_y;    /**< X-machine memory variable collision_y of type float.*/
    float exit0_x;    /**< X-machine memory variable exit0_x of type float.*/
    float exit0_y;    /**< X-machine memory variable exit0_y of type float.*/
    float exit1_x;    /**< X-machine memory variable exit1_x of type float.*/
    float exit1_y;    /**< X-machine memory variable exit1_y of type float.*/
    float exit2_x;    /**< X-machine memory variable exit2_x of type float.*/
    float exit2_y;    /**< X-machine memory variable exit2_y of type float.*/
    float exit3_x;    /**< X-machine memory variable exit3_x of type float.*/
    float exit3_y;    /**< X-machine memory variable exit3_y of type float.*/
    float exit4_x;    /**< X-machine memory variable exit4_x of type float.*/
    float exit4_y;    /**< X-machine memory variable exit4_y of type float.*/
    float exit5_x;    /**< X-machine memory variable exit5_x of type float.*/
    float exit5_y;    /**< X-machine memory variable exit5_y of type float.*/
    float exit6_x;    /**< X-machine memory variable exit6_x of type float.*/
    float exit6_y;    /**< X-machine memory variable exit6_y of type float.*/
};



/* Message structures */

/** struct xmachine_message_pedestrian_location
 * Spatial Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_pedestrian_location
{	
    /* Spatial Partitioning Variables */
    int3 _relative_cell;    /**< Relative cell position from agent grid cell poistion range -1 to 1 */
    int _cell_index_max;    /**< Max boundary value of current cell */
    int3 _agent_grid_cell;  /**< Agents partition cell position */
    int _cell_index;        /**< Index of position in current cell */  
      
    float x;        /**< Message variable x of type float.*/  
    float y;        /**< Message variable y of type float.*/  
    float z;        /**< Message variable z of type float.*/
};

/** struct xmachine_message_navmap_cell
 * Discrete Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_navmap_cell
{	
    /* Discrete Partitioning Variables */
    int2 _position;         /**< 2D position of message*/
    int2 _relative;         /**< 2D position of message relative to the agent (range +- radius) */  
      
    int x;        /**< Message variable x of type int.*/  
    int y;        /**< Message variable y of type int.*/  
    int exit_no;        /**< Message variable exit_no of type int.*/  
    float height;        /**< Message variable height of type float.*/  
    float collision_x;        /**< Message variable collision_x of type float.*/  
    float collision_y;        /**< Message variable collision_y of type float.*/  
    float exit0_x;        /**< Message variable exit0_x of type float.*/  
    float exit0_y;        /**< Message variable exit0_y of type float.*/  
    float exit1_x;        /**< Message variable exit1_x of type float.*/  
    float exit1_y;        /**< Message variable exit1_y of type float.*/  
    float exit2_x;        /**< Message variable exit2_x of type float.*/  
    float exit2_y;        /**< Message variable exit2_y of type float.*/  
    float exit3_x;        /**< Message variable exit3_x of type float.*/  
    float exit3_y;        /**< Message variable exit3_y of type float.*/  
    float exit4_x;        /**< Message variable exit4_x of type float.*/  
    float exit4_y;        /**< Message variable exit4_y of type float.*/  
    float exit5_x;        /**< Message variable exit5_x of type float.*/  
    float exit5_y;        /**< Message variable exit5_y of type float.*/  
    float exit6_x;        /**< Message variable exit6_x of type float.*/  
    float exit6_y;        /**< Message variable exit6_y of type float.*/
};



/* Agent lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_memory_agent_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_agent_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_agent_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_agent_MAX];  /**< Used during parallel prefix sum */
    
    float x [xmachine_memory_agent_MAX];    /**< X-machine memory variable list x of type float.*/
    float y [xmachine_memory_agent_MAX];    /**< X-machine memory variable list y of type float.*/
    float velx [xmachine_memory_agent_MAX];    /**< X-machine memory variable list velx of type float.*/
    float vely [xmachine_memory_agent_MAX];    /**< X-machine memory variable list vely of type float.*/
    float steer_x [xmachine_memory_agent_MAX];    /**< X-machine memory variable list steer_x of type float.*/
    float steer_y [xmachine_memory_agent_MAX];    /**< X-machine memory variable list steer_y of type float.*/
    float height [xmachine_memory_agent_MAX];    /**< X-machine memory variable list height of type float.*/
    int exit_no [xmachine_memory_agent_MAX];    /**< X-machine memory variable list exit_no of type int.*/
    float speed [xmachine_memory_agent_MAX];    /**< X-machine memory variable list speed of type float.*/
    int lod [xmachine_memory_agent_MAX];    /**< X-machine memory variable list lod of type int.*/
    float animate [xmachine_memory_agent_MAX];    /**< X-machine memory variable list animate of type float.*/
    int animate_dir [xmachine_memory_agent_MAX];    /**< X-machine memory variable list animate_dir of type int.*/
};

/** struct xmachine_memory_navmap_list
 * discrete valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_navmap_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_navmap_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_navmap_MAX];  /**< Used during parallel prefix sum */
    
    int x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list x of type int.*/
    int y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list y of type int.*/
    int exit_no [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit_no of type int.*/
    float height [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list height of type float.*/
    float collision_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list collision_x of type float.*/
    float collision_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list collision_y of type float.*/
    float exit0_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit0_x of type float.*/
    float exit0_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit0_y of type float.*/
    float exit1_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit1_x of type float.*/
    float exit1_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit1_y of type float.*/
    float exit2_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit2_x of type float.*/
    float exit2_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit2_y of type float.*/
    float exit3_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit3_x of type float.*/
    float exit3_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit3_y of type float.*/
    float exit4_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit4_x of type float.*/
    float exit4_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit4_y of type float.*/
    float exit5_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit5_x of type float.*/
    float exit5_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit5_y of type float.*/
    float exit6_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit6_x of type float.*/
    float exit6_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit6_y of type float.*/
};



/* Message lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_message_pedestrian_location_list
 * Spatial Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_pedestrian_location_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_pedestrian_location_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_pedestrian_location_MAX];  /**< Used during parallel prefix sum */
    
    float x [xmachine_message_pedestrian_location_MAX];    /**< Message memory variable list x of type float.*/
    float y [xmachine_message_pedestrian_location_MAX];    /**< Message memory variable list y of type float.*/
    float z [xmachine_message_pedestrian_location_MAX];    /**< Message memory variable list z of type float.*/
    
};

/** struct xmachine_message_navmap_cell_list
 * Discrete Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_navmap_cell_list
{
    int x [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list x of type int.*/
    int y [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list y of type int.*/
    int exit_no [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit_no of type int.*/
    float height [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list height of type float.*/
    float collision_x [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list collision_x of type float.*/
    float collision_y [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list collision_y of type float.*/
    float exit0_x [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit0_x of type float.*/
    float exit0_y [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit0_y of type float.*/
    float exit1_x [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit1_x of type float.*/
    float exit1_y [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit1_y of type float.*/
    float exit2_x [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit2_x of type float.*/
    float exit2_y [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit2_y of type float.*/
    float exit3_x [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit3_x of type float.*/
    float exit3_y [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit3_y of type float.*/
    float exit4_x [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit4_x of type float.*/
    float exit4_y [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit4_y of type float.*/
    float exit5_x [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit5_x of type float.*/
    float exit5_y [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit5_y of type float.*/
    float exit6_x [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit6_x of type float.*/
    float exit6_y [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit6_y of type float.*/
    
};



/* Spatialy Partitioned Message boundary Matrices */

/** struct xmachine_message_pedestrian_location_PBM
 * Partition Boundary Matrix (PBM) for xmachine_message_pedestrian_location 
 */
struct xmachine_message_pedestrian_location_PBM
{
	int start[xmachine_message_pedestrian_location_grid_size];
	int end_or_count[xmachine_message_pedestrian_location_grid_size];
};



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
 * output_pedestrian_location FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param pedestrian_location_messages Pointer to output message list of type xmachine_message_pedestrian_location_list. Must be passed as an argument to the add_pedestrian_location_message function ??.
 */
__FLAME_GPU_FUNC__ int output_pedestrian_location(xmachine_memory_agent* agent, xmachine_message_pedestrian_location_list* pedestrian_location_messages);

/**
 * avoid_pedestrians FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param pedestrian_location_messages  pedestrian_location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_pedestrian_location_message and get_next_pedestrian_location_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_pedestrian_location_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an arument to the rand48 function for genertaing random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int avoid_pedestrians(xmachine_memory_agent* agent, xmachine_message_pedestrian_location_list* pedestrian_location_messages, xmachine_message_pedestrian_location_PBM* partition_matrix, RNG_rand48* rand48);

/**
 * force_flow FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param navmap_cell_messages  navmap_cell_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_navmap_cell_message and get_next_navmap_cell_message functions.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an arument to the rand48 function for genertaing random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int force_flow(xmachine_memory_agent* agent, xmachine_message_navmap_cell_list* navmap_cell_messages, RNG_rand48* rand48);

/**
 * move FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int move(xmachine_memory_agent* agent);

/**
 * output_navmap_cells FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_navmap. This represents a single agent instance and can be modified directly.
 * @param navmap_cell_messages Pointer to output message list of type xmachine_message_navmap_cell_list. Must be passed as an argument to the add_navmap_cell_message function ??.
 */
__FLAME_GPU_FUNC__ int output_navmap_cells(xmachine_memory_navmap* agent, xmachine_message_navmap_cell_list* navmap_cell_messages);

/**
 * generate_pedestrians FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_navmap. This represents a single agent instance and can be modified directly.
 * @param agent_agents Pointer to agent list of type xmachine_memory_agent_list. This must be passed as an argument to the add_agent_agent function to add a new agent.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an arument to the rand48 function for genertaing random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int generate_pedestrians(xmachine_memory_navmap* agent, xmachine_memory_agent_list* agent_agents, RNG_rand48* rand48);

  
/* Message Function Prototypes for Spatially Partitioned pedestrian_location message implemented in FLAMEGPU_Kernels */

/** add_pedestrian_location_message
 * Function for all types of message partitioning
 * Adds a new pedestrian_location agent to the xmachine_memory_pedestrian_location_list list using a linear mapping
 * @param agents	xmachine_memory_pedestrian_location_list agent list
 * @param x	message variable of type float
 * @param y	message variable of type float
 * @param z	message variable of type float
 */
 
 __FLAME_GPU_FUNC__ void add_pedestrian_location_message(xmachine_message_pedestrian_location_list* pedestrian_location_messages, float x, float y, float z);
 
/** get_first_pedestrian_location_message
 * Get first message function for spatially partitioned messages
 * @param pedestrian_location_messages message list
 * @param partition_matrix the boundary partition matrix for the spatially partitioned message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @param agentz z position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_pedestrian_location * get_first_pedestrian_location_message(xmachine_message_pedestrian_location_list* pedestrian_location_messages, xmachine_message_pedestrian_location_PBM* partition_matrix, float x, float y, float z);

/** get_next_pedestrian_location_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memeory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param pedestrian_location_messages message list
 * @param partition_matrix the boundary partition matrix for the spatially partitioned message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_pedestrian_location * get_next_pedestrian_location_message(xmachine_message_pedestrian_location* current, xmachine_message_pedestrian_location_list* pedestrian_location_messages, xmachine_message_pedestrian_location_PBM* partition_matrix);

  
/* Message Function Prototypes for Discrete Partitioned navmap_cell message implemented in FLAMEGPU_Kernels */

/** add_navmap_cell_message
 * Function for all types of message partitioning
 * Adds a new navmap_cell agent to the xmachine_memory_navmap_cell_list list using a linear mapping
 * @param agents	xmachine_memory_navmap_cell_list agent list
 * @param x	message variable of type int
 * @param y	message variable of type int
 * @param exit_no	message variable of type int
 * @param height	message variable of type float
 * @param collision_x	message variable of type float
 * @param collision_y	message variable of type float
 * @param exit0_x	message variable of type float
 * @param exit0_y	message variable of type float
 * @param exit1_x	message variable of type float
 * @param exit1_y	message variable of type float
 * @param exit2_x	message variable of type float
 * @param exit2_y	message variable of type float
 * @param exit3_x	message variable of type float
 * @param exit3_y	message variable of type float
 * @param exit4_x	message variable of type float
 * @param exit4_y	message variable of type float
 * @param exit5_x	message variable of type float
 * @param exit5_y	message variable of type float
 * @param exit6_x	message variable of type float
 * @param exit6_y	message variable of type float
 */
 template <int AGENT_TYPE>
 __FLAME_GPU_FUNC__ void add_navmap_cell_message(xmachine_message_navmap_cell_list* navmap_cell_messages, int x, int y, int exit_no, float height, float collision_x, float collision_y, float exit0_x, float exit0_y, float exit1_x, float exit1_y, float exit2_x, float exit2_y, float exit3_x, float exit3_y, float exit4_x, float exit4_y, float exit5_x, float exit5_y, float exit6_x, float exit6_y);
 
/** get_first_navmap_cell_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memeory or texture cache implementation depending on AGENT_TYPE
 * @param navmap_cell_messages message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_navmap_cell * get_first_navmap_cell_message(xmachine_message_navmap_cell_list* navmap_cell_messages, int agentx, int agent_y);

/** get_next_navmap_cell_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memeory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param navmap_cell_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_navmap_cell * get_next_navmap_cell_message(xmachine_message_navmap_cell* current, xmachine_message_navmap_cell_list* navmap_cell_messages);
  
  
  
/* Agent Function Prototypes implemented in FLAMEGPU_Kernels */

/** add_agent_agent
 * Adds a new continuous valued agent agent to the xmachine_memory_agent_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_agent_list agent list
 * @param x	agent agent variable of type float
 * @param y	agent agent variable of type float
 * @param velx	agent agent variable of type float
 * @param vely	agent agent variable of type float
 * @param steer_x	agent agent variable of type float
 * @param steer_y	agent agent variable of type float
 * @param height	agent agent variable of type float
 * @param exit_no	agent agent variable of type int
 * @param speed	agent agent variable of type float
 * @param lod	agent agent variable of type int
 * @param animate	agent agent variable of type float
 * @param animate_dir	agent agent variable of type int
 */
__FLAME_GPU_FUNC__ void add_agent_agent(xmachine_memory_agent_list* agents, float x, float y, float velx, float vely, float steer_x, float steer_y, float height, int exit_no, float speed, int lod, float animate, int animate_dir);


  
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
 * @param h_navmaps Pointer to agent list on the host
 * @param d_navmaps Pointer to agent list on the GPU device
 * @param h_xmachine_memory_navmap_count Pointer to agent counter
 */
extern "C" void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_agent_list* h_agents_default, xmachine_memory_agent_list* d_agents_default, int h_xmachine_memory_agent_default_count,xmachine_memory_navmap_list* h_navmaps_static, xmachine_memory_navmap_list* d_navmaps_static, int h_xmachine_memory_navmap_static_count);


/** readInitialStates
 * Reads the current agent data fromt he device and saves it to XML
 * @param	inputpath	file path to XML file used for input of agent data
 * @param h_agents Pointer to agent list on the host
 * @param h_xmachine_memory_agent_count Pointer to agent counter
 * @param h_navmaps Pointer to agent list on the host
 * @param h_xmachine_memory_navmap_count Pointer to agent counter
 */
extern "C" void readInitialStates(char* inputpath, xmachine_memory_agent_list* h_agents, int* h_xmachine_memory_agent_count,xmachine_memory_navmap_list* h_navmaps, int* h_xmachine_memory_navmap_count);


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


/** sort_agents_default
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_agents_default(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_agent_list* agents));


    
/** get_agent_navmap_MAX_count
 * Gets the max agent count for the navmap agent type 
 * @return		the maximum navmap agent count
 */
extern "C" int get_agent_navmap_MAX_count();



/** get_agent_navmap_static_count
 * Gets the agent count for the navmap agent type in state static
 * @return		the current navmap agent count in state static
 */
extern "C" int get_agent_navmap_static_count();

/** reset_static_count
 * Resets the agent count of the navmap in state static to 0. This is usefull for interacting with some visualisations.
 */
extern "C" void reset_navmap_static_count();

/** get_device_navmap_static_agents
 * Gets a pointer to xmachine_memory_navmap_list on the GPU device
 * @return		a xmachine_memory_navmap_list on the GPU device
 */
extern "C" xmachine_memory_navmap_list* get_device_navmap_static_agents();

/** get_host_navmap_static_agents
 * Gets a pointer to xmachine_memory_navmap_list on the CPU host
 * @return		a xmachine_memory_navmap_list on the CPU host
 */
extern "C" xmachine_memory_navmap_list* get_host_navmap_static_agents();


/** get_navmap_population_width
 * Gets an int value representing the xmachine_memory_navmap population width.
 * @return		xmachine_memory_navmap population width
 */
extern "C" int get_navmap_population_width();

  
  
/* global constant variables */

__constant__ float EMMISION_RATE_EXIT1;

__constant__ float EMMISION_RATE_EXIT2;

__constant__ float EMMISION_RATE_EXIT3;

__constant__ float EMMISION_RATE_EXIT4;

__constant__ float EMMISION_RATE_EXIT5;

__constant__ float EMMISION_RATE_EXIT6;

__constant__ float EMMISION_RATE_EXIT7;

__constant__ int EXIT1_PROBABILITY;

__constant__ int EXIT2_PROBABILITY;

__constant__ int EXIT3_PROBABILITY;

__constant__ int EXIT4_PROBABILITY;

__constant__ int EXIT5_PROBABILITY;

__constant__ int EXIT6_PROBABILITY;

__constant__ int EXIT7_PROBABILITY;

__constant__ int EXIT1_STATE;

__constant__ int EXIT2_STATE;

__constant__ int EXIT3_STATE;

__constant__ int EXIT4_STATE;

__constant__ int EXIT5_STATE;

__constant__ int EXIT6_STATE;

__constant__ int EXIT7_STATE;

__constant__ float TIME_SCALER;

__constant__ float STEER_WEIGHT;

__constant__ float AVOID_WEIGHT;

__constant__ float COLLISION_WEIGHT;

__constant__ float GOAL_WEIGHT;

/** set_EMMISION_RATE_EXIT1
 * Sets the constant variable EMMISION_RATE_EXIT1 on the device which can then be used in the agent functions.
 * @param h_EMMISION_RATE_EXIT1 value to set the variable
 */
extern "C" void set_EMMISION_RATE_EXIT1(float* h_EMMISION_RATE_EXIT1);

/** set_EMMISION_RATE_EXIT2
 * Sets the constant variable EMMISION_RATE_EXIT2 on the device which can then be used in the agent functions.
 * @param h_EMMISION_RATE_EXIT2 value to set the variable
 */
extern "C" void set_EMMISION_RATE_EXIT2(float* h_EMMISION_RATE_EXIT2);

/** set_EMMISION_RATE_EXIT3
 * Sets the constant variable EMMISION_RATE_EXIT3 on the device which can then be used in the agent functions.
 * @param h_EMMISION_RATE_EXIT3 value to set the variable
 */
extern "C" void set_EMMISION_RATE_EXIT3(float* h_EMMISION_RATE_EXIT3);

/** set_EMMISION_RATE_EXIT4
 * Sets the constant variable EMMISION_RATE_EXIT4 on the device which can then be used in the agent functions.
 * @param h_EMMISION_RATE_EXIT4 value to set the variable
 */
extern "C" void set_EMMISION_RATE_EXIT4(float* h_EMMISION_RATE_EXIT4);

/** set_EMMISION_RATE_EXIT5
 * Sets the constant variable EMMISION_RATE_EXIT5 on the device which can then be used in the agent functions.
 * @param h_EMMISION_RATE_EXIT5 value to set the variable
 */
extern "C" void set_EMMISION_RATE_EXIT5(float* h_EMMISION_RATE_EXIT5);

/** set_EMMISION_RATE_EXIT6
 * Sets the constant variable EMMISION_RATE_EXIT6 on the device which can then be used in the agent functions.
 * @param h_EMMISION_RATE_EXIT6 value to set the variable
 */
extern "C" void set_EMMISION_RATE_EXIT6(float* h_EMMISION_RATE_EXIT6);

/** set_EMMISION_RATE_EXIT7
 * Sets the constant variable EMMISION_RATE_EXIT7 on the device which can then be used in the agent functions.
 * @param h_EMMISION_RATE_EXIT7 value to set the variable
 */
extern "C" void set_EMMISION_RATE_EXIT7(float* h_EMMISION_RATE_EXIT7);

/** set_EXIT1_PROBABILITY
 * Sets the constant variable EXIT1_PROBABILITY on the device which can then be used in the agent functions.
 * @param h_EXIT1_PROBABILITY value to set the variable
 */
extern "C" void set_EXIT1_PROBABILITY(int* h_EXIT1_PROBABILITY);

/** set_EXIT2_PROBABILITY
 * Sets the constant variable EXIT2_PROBABILITY on the device which can then be used in the agent functions.
 * @param h_EXIT2_PROBABILITY value to set the variable
 */
extern "C" void set_EXIT2_PROBABILITY(int* h_EXIT2_PROBABILITY);

/** set_EXIT3_PROBABILITY
 * Sets the constant variable EXIT3_PROBABILITY on the device which can then be used in the agent functions.
 * @param h_EXIT3_PROBABILITY value to set the variable
 */
extern "C" void set_EXIT3_PROBABILITY(int* h_EXIT3_PROBABILITY);

/** set_EXIT4_PROBABILITY
 * Sets the constant variable EXIT4_PROBABILITY on the device which can then be used in the agent functions.
 * @param h_EXIT4_PROBABILITY value to set the variable
 */
extern "C" void set_EXIT4_PROBABILITY(int* h_EXIT4_PROBABILITY);

/** set_EXIT5_PROBABILITY
 * Sets the constant variable EXIT5_PROBABILITY on the device which can then be used in the agent functions.
 * @param h_EXIT5_PROBABILITY value to set the variable
 */
extern "C" void set_EXIT5_PROBABILITY(int* h_EXIT5_PROBABILITY);

/** set_EXIT6_PROBABILITY
 * Sets the constant variable EXIT6_PROBABILITY on the device which can then be used in the agent functions.
 * @param h_EXIT6_PROBABILITY value to set the variable
 */
extern "C" void set_EXIT6_PROBABILITY(int* h_EXIT6_PROBABILITY);

/** set_EXIT7_PROBABILITY
 * Sets the constant variable EXIT7_PROBABILITY on the device which can then be used in the agent functions.
 * @param h_EXIT7_PROBABILITY value to set the variable
 */
extern "C" void set_EXIT7_PROBABILITY(int* h_EXIT7_PROBABILITY);

/** set_EXIT1_STATE
 * Sets the constant variable EXIT1_STATE on the device which can then be used in the agent functions.
 * @param h_EXIT1_STATE value to set the variable
 */
extern "C" void set_EXIT1_STATE(int* h_EXIT1_STATE);

/** set_EXIT2_STATE
 * Sets the constant variable EXIT2_STATE on the device which can then be used in the agent functions.
 * @param h_EXIT2_STATE value to set the variable
 */
extern "C" void set_EXIT2_STATE(int* h_EXIT2_STATE);

/** set_EXIT3_STATE
 * Sets the constant variable EXIT3_STATE on the device which can then be used in the agent functions.
 * @param h_EXIT3_STATE value to set the variable
 */
extern "C" void set_EXIT3_STATE(int* h_EXIT3_STATE);

/** set_EXIT4_STATE
 * Sets the constant variable EXIT4_STATE on the device which can then be used in the agent functions.
 * @param h_EXIT4_STATE value to set the variable
 */
extern "C" void set_EXIT4_STATE(int* h_EXIT4_STATE);

/** set_EXIT5_STATE
 * Sets the constant variable EXIT5_STATE on the device which can then be used in the agent functions.
 * @param h_EXIT5_STATE value to set the variable
 */
extern "C" void set_EXIT5_STATE(int* h_EXIT5_STATE);

/** set_EXIT6_STATE
 * Sets the constant variable EXIT6_STATE on the device which can then be used in the agent functions.
 * @param h_EXIT6_STATE value to set the variable
 */
extern "C" void set_EXIT6_STATE(int* h_EXIT6_STATE);

/** set_EXIT7_STATE
 * Sets the constant variable EXIT7_STATE on the device which can then be used in the agent functions.
 * @param h_EXIT7_STATE value to set the variable
 */
extern "C" void set_EXIT7_STATE(int* h_EXIT7_STATE);

/** set_TIME_SCALER
 * Sets the constant variable TIME_SCALER on the device which can then be used in the agent functions.
 * @param h_TIME_SCALER value to set the variable
 */
extern "C" void set_TIME_SCALER(float* h_TIME_SCALER);

/** set_STEER_WEIGHT
 * Sets the constant variable STEER_WEIGHT on the device which can then be used in the agent functions.
 * @param h_STEER_WEIGHT value to set the variable
 */
extern "C" void set_STEER_WEIGHT(float* h_STEER_WEIGHT);

/** set_AVOID_WEIGHT
 * Sets the constant variable AVOID_WEIGHT on the device which can then be used in the agent functions.
 * @param h_AVOID_WEIGHT value to set the variable
 */
extern "C" void set_AVOID_WEIGHT(float* h_AVOID_WEIGHT);

/** set_COLLISION_WEIGHT
 * Sets the constant variable COLLISION_WEIGHT on the device which can then be used in the agent functions.
 * @param h_COLLISION_WEIGHT value to set the variable
 */
extern "C" void set_COLLISION_WEIGHT(float* h_COLLISION_WEIGHT);

/** set_GOAL_WEIGHT
 * Sets the constant variable GOAL_WEIGHT on the device which can then be used in the agent functions.
 * @param h_GOAL_WEIGHT value to set the variable
 */
extern "C" void set_GOAL_WEIGHT(float* h_GOAL_WEIGHT);


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

