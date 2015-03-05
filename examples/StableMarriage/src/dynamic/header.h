
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
#define buffer_size_MAX 1024

//Maximum population size of xmachine_memory_Man
#define xmachine_memory_Man_MAX 1024

//Maximum population size of xmachine_memory_Woman
#define xmachine_memory_Woman_MAX 1024
  
  
/* Message poulation size definitions */
//Maximum population size of xmachine_mmessage_proposal
#define xmachine_message_proposal_MAX 1024

//Maximum population size of xmachine_mmessage_notification
#define xmachine_message_notification_MAX 1024



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

/** struct xmachine_memory_Man
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_Man
{
    int id;    /**< X-machine memory variable id of type int.*/
    int round;    /**< X-machine memory variable round of type int.*/
    int engaged_to;    /**< X-machine memory variable engaged_to of type int.*/
    int *preferred_woman;    /**< X-machine memory variable preferred_woman of type int.*/
};

/** struct xmachine_memory_Woman
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_Woman
{
    int id;    /**< X-machine memory variable id of type int.*/
    int current_suitor;    /**< X-machine memory variable current_suitor of type int.*/
    int current_suitor_rank;    /**< X-machine memory variable current_suitor_rank of type int.*/
    int *preferred_man;    /**< X-machine memory variable preferred_man of type int.*/
};



/* Message structures */

/** struct xmachine_message_proposal
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_proposal
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    int id;        /**< Message variable id of type int.*/  
    int woman;        /**< Message variable woman of type int.*/
};

/** struct xmachine_message_notification
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_notification
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    int id;        /**< Message variable id of type int.*/  
    int suitor;        /**< Message variable suitor of type int.*/
};



/* Agent lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_memory_Man_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_Man_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_Man_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_Man_MAX];  /**< Used during parallel prefix sum */
    
    int id [xmachine_memory_Man_MAX];    /**< X-machine memory variable list id of type int.*/
    int round [xmachine_memory_Man_MAX];    /**< X-machine memory variable list round of type int.*/
    int engaged_to [xmachine_memory_Man_MAX];    /**< X-machine memory variable list engaged_to of type int.*/
    int preferred_woman [xmachine_memory_Man_MAX*1024];    /**< X-machine memory variable list preferred_woman of type int.*/
};

/** struct xmachine_memory_Woman_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_Woman_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_Woman_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_Woman_MAX];  /**< Used during parallel prefix sum */
    
    int id [xmachine_memory_Woman_MAX];    /**< X-machine memory variable list id of type int.*/
    int current_suitor [xmachine_memory_Woman_MAX];    /**< X-machine memory variable list current_suitor of type int.*/
    int current_suitor_rank [xmachine_memory_Woman_MAX];    /**< X-machine memory variable list current_suitor_rank of type int.*/
    int preferred_man [xmachine_memory_Woman_MAX*1024];    /**< X-machine memory variable list preferred_man of type int.*/
};



/* Message lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_message_proposal_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_proposal_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_proposal_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_proposal_MAX];  /**< Used during parallel prefix sum */
    
    int id [xmachine_message_proposal_MAX];    /**< Message memory variable list id of type int.*/
    int woman [xmachine_message_proposal_MAX];    /**< Message memory variable list woman of type int.*/
    
};

/** struct xmachine_message_notification_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_notification_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_notification_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_notification_MAX];  /**< Used during parallel prefix sum */
    
    int id [xmachine_message_notification_MAX];    /**< Message memory variable list id of type int.*/
    int suitor [xmachine_message_notification_MAX];    /**< Message memory variable list suitor of type int.*/
    
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
 * make_proposals FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_Man. This represents a single agent instance and can be modified directly.
 * @param proposal_messages Pointer to output message list of type xmachine_message_proposal_list. Must be passed as an argument to the add_proposal_message function ??.
 */
__FLAME_GPU_FUNC__ int make_proposals(xmachine_memory_Man* agent, xmachine_message_proposal_list* proposal_messages);

/**
 * check_notifications FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_Man. This represents a single agent instance and can be modified directly.
 * @param notification_messages  notification_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_notification_message and get_next_notification_message functions.
 */
__FLAME_GPU_FUNC__ int check_notifications(xmachine_memory_Man* agent, xmachine_message_notification_list* notification_messages);

/**
 * check_resolved FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_Man. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int check_resolved(xmachine_memory_Man* agent);

/**
 * check_proposals FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_Woman. This represents a single agent instance and can be modified directly.
 * @param proposal_messages  proposal_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_proposal_message and get_next_proposal_message functions.
 */
__FLAME_GPU_FUNC__ int check_proposals(xmachine_memory_Woman* agent, xmachine_message_proposal_list* proposal_messages);

/**
 * notify_suitors FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_Woman. This represents a single agent instance and can be modified directly.
 * @param notification_messages Pointer to output message list of type xmachine_message_notification_list. Must be passed as an argument to the add_notification_message function ??.
 */
__FLAME_GPU_FUNC__ int notify_suitors(xmachine_memory_Woman* agent, xmachine_message_notification_list* notification_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) proposal message implemented in FLAMEGPU_Kernels */

/** add_proposal_message
 * Function for all types of message partitioning
 * Adds a new proposal agent to the xmachine_memory_proposal_list list using a linear mapping
 * @param agents	xmachine_memory_proposal_list agent list
 * @param id	message variable of type int
 * @param woman	message variable of type int
 */
 
 __FLAME_GPU_FUNC__ void add_proposal_message(xmachine_message_proposal_list* proposal_messages, int id, int woman);
 
/** get_first_proposal_message
 * Get first message function for non partitioned (brute force) messages
 * @param proposal_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_proposal * get_first_proposal_message(xmachine_message_proposal_list* proposal_messages);

/** get_next_proposal_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param proposal_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_proposal * get_next_proposal_message(xmachine_message_proposal* current, xmachine_message_proposal_list* proposal_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) notification message implemented in FLAMEGPU_Kernels */

/** add_notification_message
 * Function for all types of message partitioning
 * Adds a new notification agent to the xmachine_memory_notification_list list using a linear mapping
 * @param agents	xmachine_memory_notification_list agent list
 * @param id	message variable of type int
 * @param suitor	message variable of type int
 */
 
 __FLAME_GPU_FUNC__ void add_notification_message(xmachine_message_notification_list* notification_messages, int id, int suitor);
 
/** get_first_notification_message
 * Get first message function for non partitioned (brute force) messages
 * @param notification_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_notification * get_first_notification_message(xmachine_message_notification_list* notification_messages);

/** get_next_notification_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param notification_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_notification * get_next_notification_message(xmachine_message_notification* current, xmachine_message_notification_list* notification_messages);
  
  
  
/* Agent Function Prototypes implemented in FLAMEGPU_Kernels */

/** add_Man_agent
 * Adds a new continuous valued Man agent to the xmachine_memory_Man_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_Man_list agent list
 * @param id	agent agent variable of type int
 * @param round	agent agent variable of type int
 * @param engaged_to	agent agent variable of type int
 */
__FLAME_GPU_FUNC__ void add_Man_agent(xmachine_memory_Man_list* agents, int id, int round, int engaged_to);

/** get_Man_agent_array_value
 *  Template function for accessing Man agent array memory variables.
 *  @param array Agent memory array
 *  @param index to lookup
 *  @return return value
 */
template<typename T>
__FLAME_GPU_FUNC__ T get_Man_agent_array_value(T *array, unsigned int index);

/** set_Man_agent_array_value
 *  Template function for setting Man agent array memory variables.
 *  @param array Agent memory array
 *  @param index to lookup
 *  @param return value
 */
template<typename T>
__FLAME_GPU_FUNC__ void set_Man_agent_array_value(T *array, unsigned int index, T value);


  

/** add_Woman_agent
 * Adds a new continuous valued Woman agent to the xmachine_memory_Woman_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_Woman_list agent list
 * @param id	agent agent variable of type int
 * @param current_suitor	agent agent variable of type int
 * @param current_suitor_rank	agent agent variable of type int
 */
__FLAME_GPU_FUNC__ void add_Woman_agent(xmachine_memory_Woman_list* agents, int id, int current_suitor, int current_suitor_rank);

/** get_Woman_agent_array_value
 *  Template function for accessing Woman agent array memory variables.
 *  @param array Agent memory array
 *  @param index to lookup
 *  @return return value
 */
template<typename T>
__FLAME_GPU_FUNC__ T get_Woman_agent_array_value(T *array, unsigned int index);

/** set_Woman_agent_array_value
 *  Template function for setting Woman agent array memory variables.
 *  @param array Agent memory array
 *  @param index to lookup
 *  @param return value
 */
template<typename T>
__FLAME_GPU_FUNC__ void set_Woman_agent_array_value(T *array, unsigned int index, T value);


  


  
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
 * @param h_Mans Pointer to agent list on the host
 * @param d_Mans Pointer to agent list on the GPU device
 * @param h_xmachine_memory_Man_count Pointer to agent counter
 * @param h_Womans Pointer to agent list on the host
 * @param d_Womans Pointer to agent list on the GPU device
 * @param h_xmachine_memory_Woman_count Pointer to agent counter
 */
extern "C" void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_Man_list* h_Mans_unengaged, xmachine_memory_Man_list* d_Mans_unengaged, int h_xmachine_memory_Man_unengaged_count,xmachine_memory_Man_list* h_Mans_engaged, xmachine_memory_Man_list* d_Mans_engaged, int h_xmachine_memory_Man_engaged_count,xmachine_memory_Woman_list* h_Womans_default, xmachine_memory_Woman_list* d_Womans_default, int h_xmachine_memory_Woman_default_count);


/** readInitialStates
 * Reads the current agent data fromt he device and saves it to XML
 * @param	inputpath	file path to XML file used for input of agent data
 * @param h_Mans Pointer to agent list on the host
 * @param h_xmachine_memory_Man_count Pointer to agent counter
 * @param h_Womans Pointer to agent list on the host
 * @param h_xmachine_memory_Woman_count Pointer to agent counter
 */
extern "C" void readInitialStates(char* inputpath, xmachine_memory_Man_list* h_Mans, int* h_xmachine_memory_Man_count,xmachine_memory_Woman_list* h_Womans, int* h_xmachine_memory_Woman_count);


/* Return functions used by external code to get agent data from device */

    
/** get_agent_Man_MAX_count
 * Gets the max agent count for the Man agent type 
 * @return		the maximum Man agent count
 */
extern "C" int get_agent_Man_MAX_count();



/** get_agent_Man_unengaged_count
 * Gets the agent count for the Man agent type in state unengaged
 * @return		the current Man agent count in state unengaged
 */
extern "C" int get_agent_Man_unengaged_count();

/** reset_unengaged_count
 * Resets the agent count of the Man in state unengaged to 0. This is usefull for interacting with some visualisations.
 */
extern "C" void reset_Man_unengaged_count();

/** get_device_Man_unengaged_agents
 * Gets a pointer to xmachine_memory_Man_list on the GPU device
 * @return		a xmachine_memory_Man_list on the GPU device
 */
extern "C" xmachine_memory_Man_list* get_device_Man_unengaged_agents();

/** get_host_Man_unengaged_agents
 * Gets a pointer to xmachine_memory_Man_list on the CPU host
 * @return		a xmachine_memory_Man_list on the CPU host
 */
extern "C" xmachine_memory_Man_list* get_host_Man_unengaged_agents();


/** sort_Mans_unengaged
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_Mans_unengaged(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Man_list* agents));


/** get_agent_Man_engaged_count
 * Gets the agent count for the Man agent type in state engaged
 * @return		the current Man agent count in state engaged
 */
extern "C" int get_agent_Man_engaged_count();

/** reset_engaged_count
 * Resets the agent count of the Man in state engaged to 0. This is usefull for interacting with some visualisations.
 */
extern "C" void reset_Man_engaged_count();

/** get_device_Man_engaged_agents
 * Gets a pointer to xmachine_memory_Man_list on the GPU device
 * @return		a xmachine_memory_Man_list on the GPU device
 */
extern "C" xmachine_memory_Man_list* get_device_Man_engaged_agents();

/** get_host_Man_engaged_agents
 * Gets a pointer to xmachine_memory_Man_list on the CPU host
 * @return		a xmachine_memory_Man_list on the CPU host
 */
extern "C" xmachine_memory_Man_list* get_host_Man_engaged_agents();


/** sort_Mans_engaged
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_Mans_engaged(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Man_list* agents));


    
/** get_agent_Woman_MAX_count
 * Gets the max agent count for the Woman agent type 
 * @return		the maximum Woman agent count
 */
extern "C" int get_agent_Woman_MAX_count();



/** get_agent_Woman_default_count
 * Gets the agent count for the Woman agent type in state default
 * @return		the current Woman agent count in state default
 */
extern "C" int get_agent_Woman_default_count();

/** reset_default_count
 * Resets the agent count of the Woman in state default to 0. This is usefull for interacting with some visualisations.
 */
extern "C" void reset_Woman_default_count();

/** get_device_Woman_default_agents
 * Gets a pointer to xmachine_memory_Woman_list on the GPU device
 * @return		a xmachine_memory_Woman_list on the GPU device
 */
extern "C" xmachine_memory_Woman_list* get_device_Woman_default_agents();

/** get_host_Woman_default_agents
 * Gets a pointer to xmachine_memory_Woman_list on the CPU host
 * @return		a xmachine_memory_Woman_list on the CPU host
 */
extern "C" xmachine_memory_Woman_list* get_host_Woman_default_agents();


/** sort_Womans_default
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_Womans_default(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Woman_list* agents));


  
  
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

