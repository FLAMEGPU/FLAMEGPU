<?xml version="1.0" encoding="utf-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" 
                xmlns:xmml="http://www.dcs.shef.ac.uk/~paul/XMML"
                xmlns:gpu="http://www.dcs.shef.ac.uk/~paul/XMMLGPU">
<xsl:output method="text" version="1.0" encoding="UTF-8" indent="yes" />
<xsl:template match="/">
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
#define GLM_FORCE_NO_CTOR_INIT
#include &lt;glm/glm.hpp&gt;

/* General standard definitions */
//Threads per block (agents per block)
#define THREADS_PER_TILE 64
//Definition for any agent function or helper function
#define __FLAME_GPU_FUNC__ __device__
//Definition for a function used to initialise environment variables
#define __FLAME_GPU_INIT_FUNC__
#define __FLAME_GPU_STEP_FUNC__
#define __FLAME_GPU_EXIT_FUNC__

#define USE_CUDA_STREAMS
#define FAST_ATOMIC_SORTING

typedef unsigned int uint;


	<xsl:if test="gpu:xmodel/xmml:messages/gpu:message/xmml:variables/gpu:variable/xmml:type='double' or gpu:xmodel/xmml:xagents/gpu:xagent/xmml:memory/gpu:variable/xmml:type='double'">
//if this is defined then the project must be built with sm_13 or later
#define _DOUBLE_SUPPORT_REQUIRED_</xsl:if>

/* Agent population size definitions must be a multiple of THREADS_PER_TILE (default 64) */
//Maximum buffer size (largest agent buffer size)
<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
      <xsl:sort select="gpu:bufferSize" data-type="number" order="descending" />
      <xsl:if test="position() = 1">#define buffer_size_MAX <xsl:value-of select="gpu:bufferSize" />
      </xsl:if>
</xsl:for-each>
<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">

//Maximum population size of xmachine_memory_<xsl:value-of select="xmml:name"/>
#define xmachine_memory_<xsl:value-of select="xmml:name"/>_MAX <xsl:value-of select="gpu:bufferSize" />
</xsl:for-each>
  
  
/* Message population size definitions */<xsl:for-each select="gpu:xmodel/xmml:messages/gpu:message">
//Maximum population size of xmachine_mmessage_<xsl:value-of select="xmml:name"/>
#define xmachine_message_<xsl:value-of select="xmml:name"/>_MAX <xsl:value-of select="gpu:bufferSize" /><xsl:text>
</xsl:text></xsl:for-each>


/* Spatial partitioning grid size definitions */<xsl:for-each select="gpu:xmodel/xmml:messages/gpu:message"><xsl:if test="gpu:partitioningSpatial">
//xmachine_message_<xsl:value-of select="xmml:name"/> partition grid size (gridDim.X*gridDim.Y*gridDim.Z)<xsl:variable name="x_dim"><xsl:value-of select="ceiling ((gpu:partitioningSpatial/gpu:xmax - gpu:partitioningSpatial/gpu:xmin) div gpu:partitioningSpatial/gpu:radius)"/></xsl:variable>
<xsl:variable name="y_dim"><xsl:value-of select="ceiling ((gpu:partitioningSpatial/gpu:ymax - gpu:partitioningSpatial/gpu:ymin) div gpu:partitioningSpatial/gpu:radius)"/></xsl:variable>
<xsl:variable name="z_dim"><xsl:value-of select="ceiling ((gpu:partitioningSpatial/gpu:zmax - gpu:partitioningSpatial/gpu:zmin) div gpu:partitioningSpatial/gpu:radius)"/></xsl:variable>
#define xmachine_message_<xsl:value-of select="xmml:name"/>_grid_size <xsl:value-of select="$x_dim * $y_dim * $z_dim"/>
</xsl:if></xsl:for-each>
  
  
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
<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
/** struct xmachine_memory_<xsl:value-of select="xmml:name"/>
 * <xsl:value-of select="gpu:type"/> valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_<xsl:value-of select="xmml:name"/>
{<xsl:for-each select="xmml:memory/gpu:variable"><xsl:text>
    </xsl:text><xsl:value-of select="xmml:type"/><xsl:text> </xsl:text><xsl:if test="xmml:arrayLength">*</xsl:if><xsl:value-of select="xmml:name"/>;    /**&lt; X-machine memory variable <xsl:value-of select="xmml:name"/> of type <xsl:value-of select="xmml:type"/>.*/</xsl:for-each>
};
</xsl:for-each>


/* Message structures */
<xsl:for-each select="gpu:xmodel/xmml:messages/gpu:message">
/** struct xmachine_message_<xsl:value-of select="xmml:name"/>
 * <xsl:if test="gpu:partitioningNone">Brute force: No Partitioning</xsl:if><xsl:if test="gpu:partitioningDiscrete">Discrete Partitioning</xsl:if><xsl:if test="gpu:partitioningSpatial">Spatial Partitioning</xsl:if>
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_<xsl:value-of select="xmml:name"/>
{	
    <xsl:if test="gpu:partitioningDiscrete">/* Discrete Partitioning Variables */
    glm::ivec2 _position;         /**&lt; 2D position of message*/
    glm::ivec2 _relative;         /**&lt; 2D position of message relative to the agent (range +- radius) */</xsl:if><xsl:if test="gpu:partitioningNone">/* Brute force Partitioning Variables */
    int _position;          /**&lt; 1D position of message in linear message list */ </xsl:if><xsl:if test="gpu:partitioningSpatial">/* Spatial Partitioning Variables */
    glm::ivec3 _relative_cell;    /**&lt; Relative cell position from agent grid cell position range -1 to 1 */
    int _cell_index_max;    /**&lt; Max boundary value of current cell */
    glm::ivec3 _agent_grid_cell;  /**&lt; Agents partition cell position */
    int _cell_index;        /**&lt; Index of position in current cell */</xsl:if><xsl:text>  
    </xsl:text><xsl:for-each select="xmml:variables/gpu:variable"><xsl:text>  
    </xsl:text><xsl:value-of select="xmml:type"/><xsl:text> </xsl:text><xsl:value-of select="xmml:name"/>;        /**&lt; Message variable <xsl:value-of select="xmml:name"/> of type <xsl:value-of select="xmml:type"/>.*/</xsl:for-each>
};
</xsl:for-each>


/* Agent lists. Structure of Array (SoA) for memory coalescing on GPU */
<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
/** struct xmachine_memory_<xsl:value-of select="xmml:name"/>_list
 * <xsl:value-of select="gpu:type"/> valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_<xsl:value-of select="xmml:name"/>_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_<xsl:value-of select="xmml:name"/>_MAX];    /**&lt; Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_<xsl:value-of select="xmml:name"/>_MAX];  /**&lt; Used during parallel prefix sum */
    <xsl:for-each select="xmml:memory/gpu:variable"><xsl:text>
    </xsl:text><xsl:value-of select="xmml:type"/><xsl:text> </xsl:text><xsl:value-of select="xmml:name"/> [xmachine_memory_<xsl:value-of select="../../xmml:name"/>_MAX<xsl:if test="xmml:arrayLength">*<xsl:value-of select="xmml:arrayLength"/></xsl:if>];    /**&lt; X-machine memory variable list <xsl:value-of select="xmml:name"/> of type <xsl:value-of select="xmml:type"/>.*/</xsl:for-each>
};
</xsl:for-each>


/* Message lists. Structure of Array (SoA) for memory coalescing on GPU */
<xsl:for-each select="gpu:xmodel/xmml:messages/gpu:message">
/** struct xmachine_message_<xsl:value-of select="xmml:name"/>_list
 * <xsl:if test="gpu:partitioningNone">Brute force: No Partitioning</xsl:if><xsl:if test="gpu:partitioningDiscrete">Discrete Partitioning</xsl:if><xsl:if test="gpu:partitioningSpatial">Spatial Partitioning</xsl:if>
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_<xsl:value-of select="xmml:name"/>_list
{
    <xsl:if test="not(gpu:partitioningDiscrete)">/* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_<xsl:value-of select="xmml:name"/>_MAX];    /**&lt; Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_<xsl:value-of select="xmml:name"/>_MAX];  /**&lt; Used during parallel prefix sum */<xsl:text>
    
    </xsl:text></xsl:if><xsl:for-each select="xmml:variables/gpu:variable"><xsl:value-of select="xmml:type"/><xsl:text> </xsl:text><xsl:value-of select="xmml:name"/> [xmachine_message_<xsl:value-of select="../../xmml:name"/>_MAX];    /**&lt; Message memory variable list <xsl:value-of select="xmml:name"/> of type <xsl:value-of select="xmml:type"/>.*/
    </xsl:for-each>
};
</xsl:for-each>


/* Spatially Partitioned Message boundary Matrices */
<xsl:for-each select="gpu:xmodel/xmml:messages/gpu:message"><xsl:if test="gpu:partitioningSpatial">
/** struct xmachine_message_<xsl:value-of select="xmml:name"/>_PBM
 * Partition Boundary Matrix (PBM) for xmachine_message_<xsl:value-of select="xmml:name"/> 
 */
struct xmachine_message_<xsl:value-of select="xmml:name"/>_PBM
{
	int start[xmachine_message_<xsl:value-of select="xmml:name"/>_grid_size];
	int end_or_count[xmachine_message_<xsl:value-of select="xmml:name"/>_grid_size];
};
</xsl:if></xsl:for-each>


  /* Random */
  /** struct RNG_rand48
  *	structure used to hold list seeds
  */
  struct RNG_rand48
  {
  glm::uvec2 A, C;
  glm::uvec2 seeds[buffer_size_MAX];
  };


/** getOutputDir
* Gets the output directory of the simulation. This is the same as the 0.xml input directory.
* @return a const char pointer to string denoting the output directory
*/
const char* getOutputDir();

  /* Random Functions (usable in agent functions) implemented in FLAMEGPU_Kernels */

  /**
  * Templated random function using a DISCRETE_2D template calculates the agent index using a 2D block
  * which requires extra processing but will work for CONTINUOUS agents. Using a CONTINUOUS template will
  * not work for DISCRETE_2D agent.
  * @param	rand48	an RNG_rand48 struct which holds the seeds sued to generate a random number on the GPU
  * @return			returns a random float value
  */
  template &lt;int AGENT_TYPE&gt; __FLAME_GPU_FUNC__ float rnd(RNG_rand48* rand48);
/**
 * Non templated random function calls the templated version with DISCRETE_2D which will work in either case
 * @param	rand48	an RNG_rand48 struct which holds the seeds sued to generate a random number on the GPU
 * @return			returns a random float value
 */
__FLAME_GPU_FUNC__ float rnd(RNG_rand48* rand48);

/* Agent function prototypes */
<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:functions/gpu:function">
/**
 * <xsl:value-of select="xmml:name"/> FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_<xsl:value-of select="../../xmml:name"/>. This represents a single agent instance and can be modified directly.
 <xsl:if test="xmml:xagentOutputs/gpu:xagentOutput">* @param <xsl:value-of select="xmml:xagentOutputs/gpu:xagentOutput/xmml:xagentName"/>_agents Pointer to agent list of type xmachine_memory_<xsl:value-of select="xmml:xagentOutputs/gpu:xagentOutput/xmml:xagentName"/>_list. This must be passed as an argument to the add_<xsl:value-of select="xmml:xagentOutputs/gpu:xagentOutput/xmml:xagentName"/>_agent function to add a new agent.</xsl:if>
 <xsl:if test="xmml:inputs/gpu:input"><xsl:variable name="messagename" select="xmml:inputs/gpu:input/xmml:messageName"/>* @param <xsl:value-of select="$messagename"/>_messages  <xsl:value-of select="xmml:inputs/gpu:input/xmml:messageName"/>_messages Pointer to input message list of type xmachine_message_<xsl:value-of select="xmml:inputs/gpu:inputs/xmml:messageName"/>_list. Must be passed as an argument to the get_first_<xsl:value-of select="xmml:inputs/gpu:input/xmml:messageName"/>_message and get_next_<xsl:value-of select="xmml:inputs/gpu:input/xmml:messageName"/>_message functions.<xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messagename]">
 <xsl:if test="gpu:partitioningSpatial">* @param partition_matrix Pointer to the partition matrix of type xmachine_message_<xsl:value-of select="xmml:name"/>_PBM. Used within the get_first_<xsl:value-of select="xmml:inputs/gpu:input/xmml:messageName"/>_message and get_next_<xsl:value-of select="xmml:inputs/gpu:input/xmml:messageName"/>_message functions for spatially partitioned message access.</xsl:if></xsl:for-each></xsl:if>
 <xsl:if test="xmml:outputs/gpu:output">* @param <xsl:value-of select="xmml:outputs/gpu:output/xmml:messageName"/>_messages Pointer to output message list of type xmachine_message_<xsl:value-of select="xmml:outputs/gpu:output/xmml:messageName"/>_list. Must be passed as an argument to the add_<xsl:value-of select="xmml:outputs/gpu:output/xmml:messageName"/>_message function ??.</xsl:if>
 <xsl:if test="gpu:RNG='true'">* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.</xsl:if>
 */
__FLAME_GPU_FUNC__ int <xsl:value-of select="xmml:name"/>(xmachine_memory_<xsl:value-of select="../../xmml:name"/>* agent<xsl:if test="xmml:xagentOutputs/gpu:xagentOutput">, xmachine_memory_<xsl:value-of select="xmml:xagentOutputs/gpu:xagentOutput/xmml:xagentName"/>_list* <xsl:value-of select="xmml:xagentOutputs/gpu:xagentOutput/xmml:xagentName"/>_agents</xsl:if>
<xsl:if test="xmml:inputs/gpu:input"><xsl:variable name="messagename" select="xmml:inputs/gpu:input/xmml:messageName"/>, xmachine_message_<xsl:value-of select="xmml:inputs/gpu:input/xmml:messageName"/>_list* <xsl:value-of select="xmml:inputs/gpu:input/xmml:messageName"/>_messages<xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messagename]"><xsl:if test="gpu:partitioningSpatial">, xmachine_message_<xsl:value-of select="xmml:name"/>_PBM* partition_matrix</xsl:if></xsl:for-each></xsl:if>
<xsl:if test="xmml:outputs/gpu:output">, xmachine_message_<xsl:value-of select="xmml:outputs/gpu:output/xmml:messageName"/>_list* <xsl:value-of select="xmml:outputs/gpu:output/xmml:messageName"/>_messages</xsl:if>
<xsl:if test="gpu:RNG='true'">, RNG_rand48* rand48</xsl:if>);
</xsl:for-each>

<xsl:for-each select="gpu:xmodel/xmml:messages/gpu:message">
  
/* Message Function Prototypes for <xsl:if test="gpu:partitioningNone">Brute force (No Partitioning) </xsl:if><xsl:if test="gpu:partitioningDiscrete">Discrete Partitioned </xsl:if><xsl:if test="gpu:partitioningSpatial">Spatially Partitioned </xsl:if> <xsl:value-of select="xmml:name"/> message implemented in FLAMEGPU_Kernels */

/** add_<xsl:value-of select="xmml:name"/>_message
 * Function for all types of message partitioning
 * Adds a new <xsl:value-of select="xmml:name"/> agent to the xmachine_memory_<xsl:value-of select="xmml:name"/>_list list using a linear mapping
 * @param agents	xmachine_memory_<xsl:value-of select="xmml:name"/>_list agent list
 <xsl:for-each select="xmml:variables/gpu:variable">* @param <xsl:value-of select="xmml:name"/>	message variable of type <xsl:value-of select="xmml:type"/><xsl:text>
 </xsl:text></xsl:for-each>*/
 <xsl:if test="gpu:partitioningDiscrete">template &lt;int AGENT_TYPE&gt;</xsl:if>
 __FLAME_GPU_FUNC__ void add_<xsl:value-of select="xmml:name"/>_message(xmachine_message_<xsl:value-of select="xmml:name"/>_list* <xsl:value-of select="xmml:name"/>_messages, <xsl:for-each select="xmml:variables/gpu:variable"><xsl:value-of select="xmml:type"/><xsl:text> </xsl:text><xsl:value-of select="xmml:name"/><xsl:if test="position()!=last()">, </xsl:if></xsl:for-each>);
 
<xsl:if test="gpu:partitioningNone">/** get_first_<xsl:value-of select="xmml:name"/>_message
 * Get first message function for non partitioned (brute force) messages
 * @param <xsl:value-of select="xmml:name"/>_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_<xsl:value-of select="xmml:name"/> * get_first_<xsl:value-of select="xmml:name"/>_message(xmachine_message_<xsl:value-of select="xmml:name"/>_list* <xsl:value-of select="xmml:name"/>_messages);

/** get_next_<xsl:value-of select="xmml:name"/>_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param <xsl:value-of select="xmml:name"/>_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_<xsl:value-of select="xmml:name"/> * get_next_<xsl:value-of select="xmml:name"/>_message(xmachine_message_<xsl:value-of select="xmml:name"/>* current, xmachine_message_<xsl:value-of select="xmml:name"/>_list* <xsl:value-of select="xmml:name"/>_messages);
</xsl:if><xsl:if test="gpu:partitioningDiscrete">/** get_first_<xsl:value-of select="xmml:name"/>_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param <xsl:value-of select="xmml:name"/>_messages message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template &lt;int AGENT_TYPE&gt; __FLAME_GPU_FUNC__ xmachine_message_<xsl:value-of select="xmml:name"/> * get_first_<xsl:value-of select="xmml:name"/>_message(xmachine_message_<xsl:value-of select="xmml:name"/>_list* <xsl:value-of select="xmml:name"/>_messages, int agentx, int agent_y);

/** get_next_<xsl:value-of select="xmml:name"/>_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param <xsl:value-of select="xmml:name"/>_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template &lt;int AGENT_TYPE&gt; __FLAME_GPU_FUNC__ xmachine_message_<xsl:value-of select="xmml:name"/> * get_next_<xsl:value-of select="xmml:name"/>_message(xmachine_message_<xsl:value-of select="xmml:name"/>* current, xmachine_message_<xsl:value-of select="xmml:name"/>_list* <xsl:value-of select="xmml:name"/>_messages);
</xsl:if><xsl:if test="gpu:partitioningSpatial">/** get_first_<xsl:value-of select="xmml:name"/>_message
 * Get first message function for spatially partitioned messages
 * @param <xsl:value-of select="xmml:name"/>_messages message list
 * @param partition_matrix the boundary partition matrix for the spatially partitioned message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @param agentz z position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_<xsl:value-of select="xmml:name"/> * get_first_<xsl:value-of select="xmml:name"/>_message(xmachine_message_<xsl:value-of select="xmml:name"/>_list* <xsl:value-of select="xmml:name"/>_messages, xmachine_message_<xsl:value-of select="xmml:name"/>_PBM* partition_matrix, float x, float y, float z);

/** get_next_<xsl:value-of select="xmml:name"/>_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param <xsl:value-of select="xmml:name"/>_messages message list
 * @param partition_matrix the boundary partition matrix for the spatially partitioned message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_<xsl:value-of select="xmml:name"/> * get_next_<xsl:value-of select="xmml:name"/>_message(xmachine_message_<xsl:value-of select="xmml:name"/>* current, xmachine_message_<xsl:value-of select="xmml:name"/>_list* <xsl:value-of select="xmml:name"/>_messages, xmachine_message_<xsl:value-of select="xmml:name"/>_PBM* partition_matrix);
</xsl:if>
</xsl:for-each>  
  
  
/* Agent Function Prototypes implemented in FLAMEGPU_Kernels */
<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent"><xsl:if test="gpu:type='continuous'">
/** add_<xsl:value-of select="xmml:name"/>_agent
 * Adds a new continuous valued <xsl:value-of select="xmml:name"/> agent to the xmachine_memory_<xsl:value-of select="xmml:name"/>_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_<xsl:value-of select="xmml:name"/>_list agent list
 <xsl:for-each select="xmml:memory/gpu:variable"><xsl:if test="not(xmml:arrayLength)">* @param <xsl:value-of select="xmml:name"/>	agent agent variable of type <xsl:value-of select="xmml:type"/><xsl:text>
 </xsl:text></xsl:if></xsl:for-each>*/
__FLAME_GPU_FUNC__ void add_<xsl:value-of select="xmml:name"/>_agent(xmachine_memory_<xsl:value-of select="xmml:name"/>_list* agents, <xsl:for-each select="xmml:memory/gpu:variable[not(xmml:arrayLength)]"><xsl:value-of select="xmml:type"/><xsl:text> </xsl:text><xsl:value-of select="xmml:name"/><xsl:if test="position()!=last()">, </xsl:if></xsl:for-each>);
</xsl:if>

<xsl:if test="xmml:memory/gpu:variable/xmml:arrayLength">
/** get_<xsl:value-of select="xmml:name"/>_agent_array_value
 *  Template function for accessing <xsl:value-of select="xmml:name"/> agent array memory variables.
 *  @param array Agent memory array
 *  @param index to lookup
 *  @return return value
 */
template&lt;typename T&gt;
__FLAME_GPU_FUNC__ T get_<xsl:value-of select="xmml:name"/>_agent_array_value(T *array, unsigned int index);

/** set_<xsl:value-of select="xmml:name"/>_agent_array_value
 *  Template function for setting <xsl:value-of select="xmml:name"/> agent array memory variables.
 *  @param array Agent memory array
 *  @param index to lookup
 *  @param return value
 */
template&lt;typename T&gt;
__FLAME_GPU_FUNC__ void set_<xsl:value-of select="xmml:name"/>_agent_array_value(T *array, unsigned int index, T value);


  
</xsl:if>
    
</xsl:for-each>

  
/* Simulation function prototypes implemented in simulation.cu */

/** initialise
 * Initialise the simulation. Allocated host and device memory. Reads the initial agent configuration from XML.
 * @param input	XML file path for agent initial configuration
 */
extern void initialise(char * input);

/** cleanup
 * Function cleans up any memory allocations on the host and device
 */
extern void cleanup();

/** singleIteration
 *	Performs a single iteration of the simulation. I.e. performs each agent function on each function layer in the correct order.
 */
extern void singleIteration();

/** saveIterationData
 * Reads the current agent data fromt he device and saves it to XML
 * @param	outputpath	file path to XML file used for output of agent data
 * @param	iteration_number
 <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">* @param h_<xsl:value-of select="xmml:name"/>s Pointer to agent list on the host
 * @param d_<xsl:value-of select="xmml:name"/>s Pointer to agent list on the GPU device
 * @param h_xmachine_memory_<xsl:value-of select="xmml:name"/>_count Pointer to agent counter
 </xsl:for-each>*/
extern void saveIterationData(char* outputpath, int iteration_number, <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state">xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, int h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count<xsl:if test="position()!=last()">,</xsl:if></xsl:for-each>);


/** readInitialStates
 * Reads the current agent data from the device and saves it to XML
 * @param	inputpath	file path to XML file used for input of agent data
 <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">* @param h_<xsl:value-of select="xmml:name"/>s Pointer to agent list on the host
 * @param h_xmachine_memory_<xsl:value-of select="xmml:name"/>_count Pointer to agent counter
 </xsl:for-each>*/
extern void readInitialStates(char* inputpath, <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">xmachine_memory_<xsl:value-of select="xmml:name"/>_list* h_<xsl:value-of select="xmml:name"/>s, int* h_xmachine_memory_<xsl:value-of select="xmml:name"/>_count<xsl:if test="position()!=last()">,</xsl:if></xsl:for-each>);


/* Return functions used by external code to get agent data from device */
<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
    
/** get_agent_<xsl:value-of select="xmml:name"/>_MAX_count
 * Gets the max agent count for the <xsl:value-of select="xmml:name"/> agent type 
 * @return		the maximum <xsl:value-of select="xmml:name"/> agent count
 */
extern int get_agent_<xsl:value-of select="xmml:name"/>_MAX_count();


<xsl:for-each select="xmml:states/gpu:state">
/** get_agent_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count
 * Gets the agent count for the <xsl:value-of select="../../xmml:name"/> agent type in state <xsl:value-of select="xmml:name"/>
 * @return		the current <xsl:value-of select="../../xmml:name"/> agent count in state <xsl:value-of select="xmml:name"/>
 */
extern int get_agent_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count();

/** reset_<xsl:value-of select="xmml:name"/>_count
 * Resets the agent count of the <xsl:value-of select="../../xmml:name"/> in state <xsl:value-of select="xmml:name"/> to 0. This is useful for interacting with some visualisations.
 */
extern void reset_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count();

/** get_device_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_agents
 * Gets a pointer to xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list on the GPU device
 * @return		a xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list on the GPU device
 */
extern xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* get_device_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_agents();

/** get_host_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_agents
 * Gets a pointer to xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list on the CPU host
 * @return		a xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list on the CPU host
 */
extern xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* get_host_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_agents();

<xsl:if test="../../gpu:type='continuous'">
/** sort_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* agents));

</xsl:if>
</xsl:for-each>
<xsl:if test="gpu:type='discrete'">
/** get_<xsl:value-of select="xmml:name"/>_population_width
 * Gets an int value representing the xmachine_memory_<xsl:value-of select="xmml:name"/> population width.
 * @return		xmachine_memory_<xsl:value-of select="xmml:name"/> population width
 */
extern int get_<xsl:value-of select="xmml:name"/>_population_width();
</xsl:if>
</xsl:for-each>
  
  
/* Analytics functions for each varible in each state*/
typedef enum {
  REDUCTION_MAX,
  REDUCTION_MIN,
  REDUCTION_SUM
}reduction_operator;

<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
  <xsl:variable name="agent_name" select="xmml:name"/>
<xsl:for-each select="xmml:states/gpu:state">
  <xsl:variable name="state" select="xmml:name"/>
<xsl:for-each select="../../xmml:memory/gpu:variable">
<xsl:if test="not(xmml:arrayLength)"> <!-- Disable agent array reductions -->
/** <xsl:value-of select="xmml:type"/> reduce_<xsl:value-of select="$agent_name"/>_<xsl:value-of select="$state"/>_<xsl:value-of select="xmml:name"/>_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
<xsl:value-of select="xmml:type"/> reduce_<xsl:value-of select="$agent_name"/>_<xsl:value-of select="$state"/>_<xsl:value-of select="xmml:name"/>_variable();
</xsl:if>


<xsl:if test="xmml:type='int'">
<xsl:if test="not(xmml:arrayLength)"> <!-- Disable agent array reductions -->
/** <xsl:value-of select="xmml:type"/> count_<xsl:value-of select="$agent_name"/>_<xsl:value-of select="$state"/>_<xsl:value-of select="xmml:name"/>_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state varaible list
 */
<xsl:value-of select="xmml:type"/> count_<xsl:value-of select="$agent_name"/>_<xsl:value-of select="$state"/>_<xsl:value-of select="xmml:name"/>_variable(int count_value);
</xsl:if>
</xsl:if>

</xsl:for-each>
</xsl:for-each>
</xsl:for-each>

  
/* global constant variables */
<xsl:for-each select="gpu:xmodel/gpu:environment/gpu:constants/gpu:variable">
__constant__ <xsl:value-of select="xmml:type"/><xsl:text> </xsl:text><xsl:value-of select="xmml:name"/><xsl:if test="xmml:arrayLength">[<xsl:value-of select="xmml:arrayLength"/>]</xsl:if>;
</xsl:for-each>
    
<xsl:for-each select="gpu:xmodel/gpu:environment/gpu:constants/gpu:variable">
/** set_<xsl:value-of select="xmml:name"/>
 * Sets the constant variable <xsl:value-of select="xmml:name"/> on the device which can then be used in the agent functions.
 * @param h_<xsl:value-of select="xmml:name"/> value to set the variable
 */
extern void set_<xsl:value-of select="xmml:name"/>(<xsl:value-of select="xmml:type"/>* h_<xsl:value-of select="xmml:name"/>);

extern const <xsl:value-of select="xmml:type"/>* get_<xsl:value-of select="xmml:name"/>();


extern <xsl:value-of select="xmml:type"/><xsl:text> h_env_</xsl:text><xsl:value-of select="xmml:name"/><xsl:if test="xmml:arrayLength">[<xsl:value-of select="xmml:arrayLength"/>]</xsl:if>;
</xsl:for-each>

/** getMaximumBound
 * Returns the maximum agent positions determined from the initial loading of agents
 * @return 	a three component float indicating the maximum x, y and z positions of all agents
 */
glm::vec3 getMaximumBounds();

/** getMinimumBounds
 * Returns the minimum agent positions determined from the initial loading of agents
 * @return 	a three component float indicating the minimum x, y and z positions of all agents
 */
glm::vec3 getMinimumBounds();
    
    
#ifdef VISUALISATION
/** initVisualisation
 * Prototype for method which initialises the visualisation. Must be implemented in separate file
 * @param argc	the argument count from the main function used with GLUT
 * @param argv	the argument values from the main function used with GLUT
 */
extern void initVisualisation();

extern void runVisualisation();


#endif

#endif //__HEADER

</xsl:template>
</xsl:stylesheet>
