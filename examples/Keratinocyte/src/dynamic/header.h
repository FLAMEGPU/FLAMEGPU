
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
#define buffer_size_MAX 8192

//Maximum population size of xmachine_memory_keratinocyte
#define xmachine_memory_keratinocyte_MAX 8192
  
  
/* Message poulation size definitions */
//Maximum population size of xmachine_mmessage_location
#define xmachine_message_location_MAX 8192

//Maximum population size of xmachine_mmessage_force
#define xmachine_message_force_MAX 8192



/* Spatial partitioning grid size definitions */
//xmachine_message_location partition grid size (gridDim.X*gridDim.Y*gridDim.Z)
#define xmachine_message_location_grid_size 64
//xmachine_message_force partition grid size (gridDim.X*gridDim.Y*gridDim.Z)
#define xmachine_message_force_grid_size 32768
  
  
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

/** struct xmachine_memory_keratinocyte
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_keratinocyte
{
    int id;    /**< X-machine memory variable id of type int.*/
    int type;    /**< X-machine memory variable type of type int.*/
    float x;    /**< X-machine memory variable x of type float.*/
    float y;    /**< X-machine memory variable y of type float.*/
    float z;    /**< X-machine memory variable z of type float.*/
    float force_x;    /**< X-machine memory variable force_x of type float.*/
    float force_y;    /**< X-machine memory variable force_y of type float.*/
    float force_z;    /**< X-machine memory variable force_z of type float.*/
    int num_xy_bonds;    /**< X-machine memory variable num_xy_bonds of type int.*/
    int num_z_bonds;    /**< X-machine memory variable num_z_bonds of type int.*/
    int num_stem_bonds;    /**< X-machine memory variable num_stem_bonds of type int.*/
    int cycle;    /**< X-machine memory variable cycle of type int.*/
    float diff_noise_factor;    /**< X-machine memory variable diff_noise_factor of type float.*/
    int dead_ticks;    /**< X-machine memory variable dead_ticks of type int.*/
    int contact_inhibited_ticks;    /**< X-machine memory variable contact_inhibited_ticks of type int.*/
    float motility;    /**< X-machine memory variable motility of type float.*/
    float dir;    /**< X-machine memory variable dir of type float.*/
    float movement;    /**< X-machine memory variable movement of type float.*/
};



/* Message structures */

/** struct xmachine_message_location
 * Spatial Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_location
{	
    /* Spatial Partitioning Variables */
    int3 _relative_cell;    /**< Relative cell position from agent grid cell poistion range -1 to 1 */
    int _cell_index_max;    /**< Max boundary value of current cell */
    int3 _agent_grid_cell;  /**< Agents partition cell position */
    int _cell_index;        /**< Index of position in current cell */  
      
    int id;        /**< Message variable id of type int.*/  
    int type;        /**< Message variable type of type int.*/  
    float x;        /**< Message variable x of type float.*/  
    float y;        /**< Message variable y of type float.*/  
    float z;        /**< Message variable z of type float.*/  
    float dir;        /**< Message variable dir of type float.*/  
    float motility;        /**< Message variable motility of type float.*/  
    float range;        /**< Message variable range of type float.*/  
    int iteration;        /**< Message variable iteration of type int.*/
};

/** struct xmachine_message_force
 * Spatial Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_force
{	
    /* Spatial Partitioning Variables */
    int3 _relative_cell;    /**< Relative cell position from agent grid cell poistion range -1 to 1 */
    int _cell_index_max;    /**< Max boundary value of current cell */
    int3 _agent_grid_cell;  /**< Agents partition cell position */
    int _cell_index;        /**< Index of position in current cell */  
      
    int type;        /**< Message variable type of type int.*/  
    float x;        /**< Message variable x of type float.*/  
    float y;        /**< Message variable y of type float.*/  
    float z;        /**< Message variable z of type float.*/  
    int id;        /**< Message variable id of type int.*/
};



/* Agent lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_memory_keratinocyte_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_keratinocyte_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_keratinocyte_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_keratinocyte_MAX];  /**< Used during parallel prefix sum */
    
    int id [xmachine_memory_keratinocyte_MAX];    /**< X-machine memory variable list id of type int.*/
    int type [xmachine_memory_keratinocyte_MAX];    /**< X-machine memory variable list type of type int.*/
    float x [xmachine_memory_keratinocyte_MAX];    /**< X-machine memory variable list x of type float.*/
    float y [xmachine_memory_keratinocyte_MAX];    /**< X-machine memory variable list y of type float.*/
    float z [xmachine_memory_keratinocyte_MAX];    /**< X-machine memory variable list z of type float.*/
    float force_x [xmachine_memory_keratinocyte_MAX];    /**< X-machine memory variable list force_x of type float.*/
    float force_y [xmachine_memory_keratinocyte_MAX];    /**< X-machine memory variable list force_y of type float.*/
    float force_z [xmachine_memory_keratinocyte_MAX];    /**< X-machine memory variable list force_z of type float.*/
    int num_xy_bonds [xmachine_memory_keratinocyte_MAX];    /**< X-machine memory variable list num_xy_bonds of type int.*/
    int num_z_bonds [xmachine_memory_keratinocyte_MAX];    /**< X-machine memory variable list num_z_bonds of type int.*/
    int num_stem_bonds [xmachine_memory_keratinocyte_MAX];    /**< X-machine memory variable list num_stem_bonds of type int.*/
    int cycle [xmachine_memory_keratinocyte_MAX];    /**< X-machine memory variable list cycle of type int.*/
    float diff_noise_factor [xmachine_memory_keratinocyte_MAX];    /**< X-machine memory variable list diff_noise_factor of type float.*/
    int dead_ticks [xmachine_memory_keratinocyte_MAX];    /**< X-machine memory variable list dead_ticks of type int.*/
    int contact_inhibited_ticks [xmachine_memory_keratinocyte_MAX];    /**< X-machine memory variable list contact_inhibited_ticks of type int.*/
    float motility [xmachine_memory_keratinocyte_MAX];    /**< X-machine memory variable list motility of type float.*/
    float dir [xmachine_memory_keratinocyte_MAX];    /**< X-machine memory variable list dir of type float.*/
    float movement [xmachine_memory_keratinocyte_MAX];    /**< X-machine memory variable list movement of type float.*/
};



/* Message lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_message_location_list
 * Spatial Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_location_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_location_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_location_MAX];  /**< Used during parallel prefix sum */
    
    int id [xmachine_message_location_MAX];    /**< Message memory variable list id of type int.*/
    int type [xmachine_message_location_MAX];    /**< Message memory variable list type of type int.*/
    float x [xmachine_message_location_MAX];    /**< Message memory variable list x of type float.*/
    float y [xmachine_message_location_MAX];    /**< Message memory variable list y of type float.*/
    float z [xmachine_message_location_MAX];    /**< Message memory variable list z of type float.*/
    float dir [xmachine_message_location_MAX];    /**< Message memory variable list dir of type float.*/
    float motility [xmachine_message_location_MAX];    /**< Message memory variable list motility of type float.*/
    float range [xmachine_message_location_MAX];    /**< Message memory variable list range of type float.*/
    int iteration [xmachine_message_location_MAX];    /**< Message memory variable list iteration of type int.*/
    
};

/** struct xmachine_message_force_list
 * Spatial Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_force_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_force_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_force_MAX];  /**< Used during parallel prefix sum */
    
    int type [xmachine_message_force_MAX];    /**< Message memory variable list type of type int.*/
    float x [xmachine_message_force_MAX];    /**< Message memory variable list x of type float.*/
    float y [xmachine_message_force_MAX];    /**< Message memory variable list y of type float.*/
    float z [xmachine_message_force_MAX];    /**< Message memory variable list z of type float.*/
    int id [xmachine_message_force_MAX];    /**< Message memory variable list id of type int.*/
    
};



/* Spatialy Partitioned Message boundary Matrices */

/** struct xmachine_message_location_PBM
 * Partition Boundary Matrix (PBM) for xmachine_message_location 
 */
struct xmachine_message_location_PBM
{
	int start[xmachine_message_location_grid_size];
	int end_or_count[xmachine_message_location_grid_size];
};

/** struct xmachine_message_force_PBM
 * Partition Boundary Matrix (PBM) for xmachine_message_force 
 */
struct xmachine_message_force_PBM
{
	int start[xmachine_message_force_grid_size];
	int end_or_count[xmachine_message_force_grid_size];
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
 * output_location FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_keratinocyte. This represents a single agent instance and can be modified directly.
 * @param location_messages Pointer to output message list of type xmachine_message_location_list. Must be passed as an argument to the add_location_message function ??.
 */
__FLAME_GPU_FUNC__ int output_location(xmachine_memory_keratinocyte* agent, xmachine_message_location_list* location_messages);

/**
 * cycle FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_keratinocyte. This represents a single agent instance and can be modified directly.
 * @param keratinocyte_agents Pointer to agent list of type xmachine_memory_keratinocyte_list. This must be passed as an argument to the add_keratinocyte_agent function to add a new agent.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an arument to the rand48 function for genertaing random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int cycle(xmachine_memory_keratinocyte* agent, xmachine_memory_keratinocyte_list* keratinocyte_agents, RNG_rand48* rand48);

/**
 * differentiate FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_keratinocyte. This represents a single agent instance and can be modified directly.
 * @param location_messages  location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_location_message and get_next_location_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_location_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.
 */
__FLAME_GPU_FUNC__ int differentiate(xmachine_memory_keratinocyte* agent, xmachine_message_location_list* location_messages, xmachine_message_location_PBM* partition_matrix);

/**
 * death_signal FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_keratinocyte. This represents a single agent instance and can be modified directly.
 * @param location_messages  location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_location_message and get_next_location_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_location_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an arument to the rand48 function for genertaing random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int death_signal(xmachine_memory_keratinocyte* agent, xmachine_message_location_list* location_messages, xmachine_message_location_PBM* partition_matrix, RNG_rand48* rand48);

/**
 * migrate FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_keratinocyte. This represents a single agent instance and can be modified directly.
 * @param location_messages  location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_location_message and get_next_location_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_location_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an arument to the rand48 function for genertaing random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int migrate(xmachine_memory_keratinocyte* agent, xmachine_message_location_list* location_messages, xmachine_message_location_PBM* partition_matrix, RNG_rand48* rand48);

/**
 * force_resolution_output FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_keratinocyte. This represents a single agent instance and can be modified directly.
 * @param force_messages Pointer to output message list of type xmachine_message_force_list. Must be passed as an argument to the add_force_message function ??.
 */
__FLAME_GPU_FUNC__ int force_resolution_output(xmachine_memory_keratinocyte* agent, xmachine_message_force_list* force_messages);

/**
 * resolve_forces FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_keratinocyte. This represents a single agent instance and can be modified directly.
 * @param force_messages  force_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_force_message and get_next_force_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_force_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.
 */
__FLAME_GPU_FUNC__ int resolve_forces(xmachine_memory_keratinocyte* agent, xmachine_message_force_list* force_messages, xmachine_message_force_PBM* partition_matrix);

  
/* Message Function Prototypes for Spatially Partitioned location message implemented in FLAMEGPU_Kernels */

/** add_location_message
 * Function for all types of message partitioning
 * Adds a new location agent to the xmachine_memory_location_list list using a linear mapping
 * @param agents	xmachine_memory_location_list agent list
 * @param id	message variable of type int
 * @param type	message variable of type int
 * @param x	message variable of type float
 * @param y	message variable of type float
 * @param z	message variable of type float
 * @param dir	message variable of type float
 * @param motility	message variable of type float
 * @param range	message variable of type float
 * @param iteration	message variable of type int
 */
 
 __FLAME_GPU_FUNC__ void add_location_message(xmachine_message_location_list* location_messages, int id, int type, float x, float y, float z, float dir, float motility, float range, int iteration);
 
/** get_first_location_message
 * Get first message function for spatially partitioned messages
 * @param location_messages message list
 * @param partition_matrix the boundary partition matrix for the spatially partitioned message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @param agentz z position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_location * get_first_location_message(xmachine_message_location_list* location_messages, xmachine_message_location_PBM* partition_matrix, float x, float y, float z);

/** get_next_location_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memeory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param location_messages message list
 * @param partition_matrix the boundary partition matrix for the spatially partitioned message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_location * get_next_location_message(xmachine_message_location* current, xmachine_message_location_list* location_messages, xmachine_message_location_PBM* partition_matrix);

  
/* Message Function Prototypes for Spatially Partitioned force message implemented in FLAMEGPU_Kernels */

/** add_force_message
 * Function for all types of message partitioning
 * Adds a new force agent to the xmachine_memory_force_list list using a linear mapping
 * @param agents	xmachine_memory_force_list agent list
 * @param type	message variable of type int
 * @param x	message variable of type float
 * @param y	message variable of type float
 * @param z	message variable of type float
 * @param id	message variable of type int
 */
 
 __FLAME_GPU_FUNC__ void add_force_message(xmachine_message_force_list* force_messages, int type, float x, float y, float z, int id);
 
/** get_first_force_message
 * Get first message function for spatially partitioned messages
 * @param force_messages message list
 * @param partition_matrix the boundary partition matrix for the spatially partitioned message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @param agentz z position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_force * get_first_force_message(xmachine_message_force_list* force_messages, xmachine_message_force_PBM* partition_matrix, float x, float y, float z);

/** get_next_force_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memeory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param force_messages message list
 * @param partition_matrix the boundary partition matrix for the spatially partitioned message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_force * get_next_force_message(xmachine_message_force* current, xmachine_message_force_list* force_messages, xmachine_message_force_PBM* partition_matrix);
  
  
  
/* Agent Function Prototypes implemented in FLAMEGPU_Kernels */

/** add_keratinocyte_agent
 * Adds a new continuous valued keratinocyte agent to the xmachine_memory_keratinocyte_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_keratinocyte_list agent list
 * @param id	agent agent variable of type int
 * @param type	agent agent variable of type int
 * @param x	agent agent variable of type float
 * @param y	agent agent variable of type float
 * @param z	agent agent variable of type float
 * @param force_x	agent agent variable of type float
 * @param force_y	agent agent variable of type float
 * @param force_z	agent agent variable of type float
 * @param num_xy_bonds	agent agent variable of type int
 * @param num_z_bonds	agent agent variable of type int
 * @param num_stem_bonds	agent agent variable of type int
 * @param cycle	agent agent variable of type int
 * @param diff_noise_factor	agent agent variable of type float
 * @param dead_ticks	agent agent variable of type int
 * @param contact_inhibited_ticks	agent agent variable of type int
 * @param motility	agent agent variable of type float
 * @param dir	agent agent variable of type float
 * @param movement	agent agent variable of type float
 */
__FLAME_GPU_FUNC__ void add_keratinocyte_agent(xmachine_memory_keratinocyte_list* agents, int id, int type, float x, float y, float z, float force_x, float force_y, float force_z, int num_xy_bonds, int num_z_bonds, int num_stem_bonds, int cycle, float diff_noise_factor, int dead_ticks, int contact_inhibited_ticks, float motility, float dir, float movement);


  
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
 * @param h_keratinocytes Pointer to agent list on the host
 * @param d_keratinocytes Pointer to agent list on the GPU device
 * @param h_xmachine_memory_keratinocyte_count Pointer to agent counter
 */
extern "C" void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_keratinocyte_list* h_keratinocytes_default, xmachine_memory_keratinocyte_list* d_keratinocytes_default, int h_xmachine_memory_keratinocyte_default_count,xmachine_memory_keratinocyte_list* h_keratinocytes_resolve, xmachine_memory_keratinocyte_list* d_keratinocytes_resolve, int h_xmachine_memory_keratinocyte_resolve_count);


/** readInitialStates
 * Reads the current agent data fromt he device and saves it to XML
 * @param	inputpath	file path to XML file used for input of agent data
 * @param h_keratinocytes Pointer to agent list on the host
 * @param h_xmachine_memory_keratinocyte_count Pointer to agent counter
 */
extern "C" void readInitialStates(char* inputpath, xmachine_memory_keratinocyte_list* h_keratinocytes, int* h_xmachine_memory_keratinocyte_count);


/* Return functions used by external code to get agent data from device */

    
/** get_agent_keratinocyte_MAX_count
 * Gets the max agent count for the keratinocyte agent type 
 * @return		the maximum keratinocyte agent count
 */
extern "C" int get_agent_keratinocyte_MAX_count();



/** get_agent_keratinocyte_default_count
 * Gets the agent count for the keratinocyte agent type in state default
 * @return		the current keratinocyte agent count in state default
 */
extern "C" int get_agent_keratinocyte_default_count();

/** reset_default_count
 * Resets the agent count of the keratinocyte in state default to 0. This is usefull for interacting with some visualisations.
 */
extern "C" void reset_keratinocyte_default_count();

/** get_device_keratinocyte_default_agents
 * Gets a pointer to xmachine_memory_keratinocyte_list on the GPU device
 * @return		a xmachine_memory_keratinocyte_list on the GPU device
 */
extern "C" xmachine_memory_keratinocyte_list* get_device_keratinocyte_default_agents();

/** get_host_keratinocyte_default_agents
 * Gets a pointer to xmachine_memory_keratinocyte_list on the CPU host
 * @return		a xmachine_memory_keratinocyte_list on the CPU host
 */
extern "C" xmachine_memory_keratinocyte_list* get_host_keratinocyte_default_agents();


/** sort_keratinocytes_default
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_keratinocytes_default(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_keratinocyte_list* agents));


/** get_agent_keratinocyte_resolve_count
 * Gets the agent count for the keratinocyte agent type in state resolve
 * @return		the current keratinocyte agent count in state resolve
 */
extern "C" int get_agent_keratinocyte_resolve_count();

/** reset_resolve_count
 * Resets the agent count of the keratinocyte in state resolve to 0. This is usefull for interacting with some visualisations.
 */
extern "C" void reset_keratinocyte_resolve_count();

/** get_device_keratinocyte_resolve_agents
 * Gets a pointer to xmachine_memory_keratinocyte_list on the GPU device
 * @return		a xmachine_memory_keratinocyte_list on the GPU device
 */
extern "C" xmachine_memory_keratinocyte_list* get_device_keratinocyte_resolve_agents();

/** get_host_keratinocyte_resolve_agents
 * Gets a pointer to xmachine_memory_keratinocyte_list on the CPU host
 * @return		a xmachine_memory_keratinocyte_list on the CPU host
 */
extern "C" xmachine_memory_keratinocyte_list* get_host_keratinocyte_resolve_agents();


/** sort_keratinocytes_resolve
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_keratinocytes_resolve(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_keratinocyte_list* agents));


  
  
/* global constant variables */

__constant__ float calcium_level;

__constant__ int CYCLE_LENGTH[5];

__constant__ float SUBSTRATE_FORCE[5];

__constant__ float DOWNWARD_FORCE[5];

__constant__ float FORCE_MATRIX[25];

__constant__ float FORCE_REP;

__constant__ float FORCE_DAMPENER;

__constant__ int BASEMENT_MAX_Z;

/** set_calcium_level
 * Sets the constant variable calcium_level on the device which can then be used in the agent functions.
 * @param h_calcium_level value to set the variable
 */
extern "C" void set_calcium_level(float* h_calcium_level);

/** set_CYCLE_LENGTH
 * Sets the constant variable CYCLE_LENGTH on the device which can then be used in the agent functions.
 * @param h_CYCLE_LENGTH value to set the variable
 */
extern "C" void set_CYCLE_LENGTH(int* h_CYCLE_LENGTH);

/** set_SUBSTRATE_FORCE
 * Sets the constant variable SUBSTRATE_FORCE on the device which can then be used in the agent functions.
 * @param h_SUBSTRATE_FORCE value to set the variable
 */
extern "C" void set_SUBSTRATE_FORCE(float* h_SUBSTRATE_FORCE);

/** set_DOWNWARD_FORCE
 * Sets the constant variable DOWNWARD_FORCE on the device which can then be used in the agent functions.
 * @param h_DOWNWARD_FORCE value to set the variable
 */
extern "C" void set_DOWNWARD_FORCE(float* h_DOWNWARD_FORCE);

/** set_FORCE_MATRIX
 * Sets the constant variable FORCE_MATRIX on the device which can then be used in the agent functions.
 * @param h_FORCE_MATRIX value to set the variable
 */
extern "C" void set_FORCE_MATRIX(float* h_FORCE_MATRIX);

/** set_FORCE_REP
 * Sets the constant variable FORCE_REP on the device which can then be used in the agent functions.
 * @param h_FORCE_REP value to set the variable
 */
extern "C" void set_FORCE_REP(float* h_FORCE_REP);

/** set_FORCE_DAMPENER
 * Sets the constant variable FORCE_DAMPENER on the device which can then be used in the agent functions.
 * @param h_FORCE_DAMPENER value to set the variable
 */
extern "C" void set_FORCE_DAMPENER(float* h_FORCE_DAMPENER);

/** set_BASEMENT_MAX_Z
 * Sets the constant variable BASEMENT_MAX_Z on the device which can then be used in the agent functions.
 * @param h_BASEMENT_MAX_Z value to set the variable
 */
extern "C" void set_BASEMENT_MAX_Z(int* h_BASEMENT_MAX_Z);


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

