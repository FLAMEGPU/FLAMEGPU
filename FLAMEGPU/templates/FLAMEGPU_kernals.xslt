<?xml version="1.0" encoding="utf-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" 
                xmlns:xmml="http://www.dcs.shef.ac.uk/~paul/XMML"
                xmlns:gpu="http://www.dcs.shef.ac.uk/~paul/XMMLGPU">
<xsl:output method="text" version="1.0" encoding="UTF-8" indent="yes" />
<xsl:include href = "./_common_templates.xslt" />
<!--Main template-->
<xsl:template match="/">
<xsl:call-template name="copyrightNotice"></xsl:call-template>

#ifndef _FLAMEGPU_KERNELS_H_
#define _FLAMEGPU_KERNELS_H_

#include "header.h"


/* Agent count constants */
<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
__constant__ int d_xmachine_memory_<xsl:value-of select="xmml:name"/>_count;
</xsl:for-each>
/* Agent state count constants */
<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state">
__constant__ int d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count;
</xsl:for-each>

/* Message constants */
<xsl:for-each select="gpu:xmodel/xmml:messages/gpu:message">
/* <xsl:value-of select="xmml:name"/> Message variables */
<xsl:if test="gpu:partitioningNone or gpu:partitioningSpatial or gpu:partitioningGraphEdge">/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_<xsl:value-of select="xmml:name"/>_count;         /**&lt; message list counter*/
__constant__ int d_message_<xsl:value-of select="xmml:name"/>_output_type;   /**&lt; message output type (single or optional)*/
</xsl:if><xsl:if test="gpu:partitioningSpatial">//Spatial Partitioning Variables
__constant__ glm::vec3 d_message_<xsl:value-of select="xmml:name"/>_min_bounds;           /**&lt; min bounds (x,y,z) of partitioning environment */
__constant__ glm::vec3 d_message_<xsl:value-of select="xmml:name"/>_max_bounds;           /**&lt; max bounds (x,y,z) of partitioning environment */
__constant__ glm::ivec3 d_message_<xsl:value-of select="xmml:name"/>_partitionDim;           /**&lt; partition dimensions (x,y,z) of partitioning environment */
__constant__ float d_message_<xsl:value-of select="xmml:name"/>_radius;                 /**&lt; partition radius (used to determin the size of the partitions) */
</xsl:if><xsl:if test="gpu:partitioningDiscrete">//Discrete Partitioning Variables
__constant__ int d_message_<xsl:value-of select="xmml:name"/>_range;     /**&lt; range of the discrete message*/
__constant__ int d_message_<xsl:value-of select="xmml:name"/>_width;     /**&lt; with of the message grid*/
</xsl:if>
</xsl:for-each>
	

/* Graph Constants */
<xsl:for-each select="gpu:xmodel/gpu:environment/gpu:graphs/gpu:staticGraph">
__constant__ staticGraph_memory_<xsl:value-of select="gpu:name"/>* d_staticGraph_memory_<xsl:value-of select="gpu:name"/>_ptr;
</xsl:for-each>

/* Graph device array pointer(s) */
<xsl:for-each select="gpu:xmodel/gpu:environment/gpu:graphs/gpu:staticGraph">
staticGraph_memory_<xsl:value-of select="gpu:name"/>* d_staticGraph_memory_<xsl:value-of select="gpu:name"/> = nullptr;
</xsl:for-each>

/* Graph host array pointer(s) */
<xsl:for-each select="gpu:xmodel/gpu:environment/gpu:graphs/gpu:staticGraph">
staticGraph_memory_<xsl:value-of select="gpu:name"/>* h_staticGraph_memory_<xsl:value-of select="gpu:name"/> = nullptr;
</xsl:for-each>
    
//include each function file
<xsl:for-each select="gpu:xmodel/gpu:environment/gpu:functionFiles">
#include "<xsl:value-of select="xmml:file"/>"</xsl:for-each>
    
/* Texture bindings */<xsl:for-each select="gpu:xmodel/xmml:messages/gpu:message"><xsl:if test="gpu:partitioningDiscrete or gpu:partitioningSpatial">
/* <xsl:value-of select="xmml:name"/> Message Bindings */<xsl:for-each select="xmml:variables/gpu:variable"><xsl:choose>
<xsl:when test="xmml:type='double'">texture&lt;int2, 1, cudaReadModeElementType&gt; tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>;</xsl:when>
<xsl:otherwise>texture&lt;<xsl:value-of select="xmml:type"/>, 1, cudaReadModeElementType&gt; tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>;</xsl:otherwise></xsl:choose>
__constant__ int d_tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_offset;</xsl:for-each>
<xsl:if test="gpu:partitioningSpatial">
texture&lt;int, 1, cudaReadModeElementType&gt; tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_start;
__constant__ int d_tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_start_offset;
texture&lt;int, 1, cudaReadModeElementType&gt; tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_end_or_count;
__constant__ int d_tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_end_or_count_offset;
</xsl:if></xsl:if><xsl:text>
</xsl:text></xsl:for-each>
    
#define WRAP(x,m) (((x)&lt;m)?(x):(x%m)) /**&lt; Simple wrap */
#define sWRAP(x,m) (((x)&lt;m)?(((x)&lt;0)?(m+(x)):(x)):(m-(x))) /**&lt;signed integer wrap (no modulus) for negatives where 2m > |x| > m */

//PADDING WILL ONLY AVOID SM CONFLICTS FOR 32BIT
//SM_OFFSET REQUIRED AS FERMI STARTS INDEXING MEMORY FROM LOCATION 0 (i.e. NULL)??
__constant__ int d_SM_START;
__constant__ int d_PADDING;

//SM addressing macro to avoid conflicts (32 bit only)
#define SHARE_INDEX(i, s) ((((s) + d_PADDING)* (i))+d_SM_START) /**&lt;offset struct size by padding to avoid bank conflicts */

//if doubel support is needed then define the following function which requires sm_13 or later
#ifdef _DOUBLE_SUPPORT_REQUIRED_
__inline__ __device__ double tex1DfetchDouble(texture&lt;int2, 1, cudaReadModeElementType&gt; tex, int i)
{
	int2 v = tex1Dfetch(tex, i);
  //IF YOU HAVE AN ERROR HERE THEN YOU ARE USING DOUBLE VALUES IN AGENT MEMORY AND NOT COMPILING FOR DOUBLE SUPPORTED HARDWARE
  //To compile for double supported hardware change the CUDA Build rule property "Use sm_13 Architecture (double support)" on the CUDA-Specific Propert Page of the CUDA Build Rule for simulation.cu
	return __hiloint2double(v.y, v.x);
}
#endif

/* Helper functions */
/** next_cell
 * Function used for finding the next cell when using spatial partitioning
 * Upddates the relative cell variable which can have value of -1, 0 or +1
 * @param relative_cell pointer to the relative cell position
 * @return boolean if there is a next cell. True unless relative_Cell value was 1,1,1
 */
__device__ bool next_cell3D(glm::ivec3* relative_cell)
{
	if (relative_cell->x &lt; 1)
	{
		relative_cell->x++;
		return true;
	}
	relative_cell->x = -1;

	if (relative_cell->y &lt; 1)
	{
		relative_cell->y++;
		return true;
	}
	relative_cell->y = -1;
	
	if (relative_cell->z &lt; 1)
	{
		relative_cell->z++;
		return true;
	}
	relative_cell->z = -1;
	
	return false;
}

/** next_cell2D
 * Function used for finding the next cell when using spatial partitioning. Z component is ignored
 * Upddates the relative cell variable which can have value of -1, 0 or +1
 * @param relative_cell pointer to the relative cell position
 * @return boolean if there is a next cell. True unless relative_Cell value was 1,1
 */
__device__ bool next_cell2D(glm::ivec3* relative_cell)
{
	if (relative_cell->x &lt; 1)
	{
		relative_cell->x++;
		return true;
	}
	relative_cell->x = -1;

	if (relative_cell->y &lt; 1)
	{
		relative_cell->y++;
		return true;
	}
	relative_cell->y = -1;
	
	return false;
}

<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:functions/gpu:function/xmml:condition">
/** <xsl:value-of select="../xmml:name"/>_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_<xsl:value-of select="../../../xmml:name"/>_list representing agent i the current state
 * @param nextState xmachine_memory_<xsl:value-of select="../../../xmml:name"/>_list representing agent i the next state
 */
 __global__ void <xsl:value-of select="../xmml:name"/>_function_filter(xmachine_memory_<xsl:value-of select="../../../xmml:name"/>_list* currentState, xmachine_memory_<xsl:value-of select="../../../xmml:name"/>_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index &lt; d_xmachine_memory_<xsl:value-of select="../../../xmml:name"/>_count){
	
		//apply the filter
		if <xsl:apply-templates select="."/>
		{	//copy agent data to newstate list<xsl:for-each select="../../../xmml:memory/gpu:variable">
			nextState-><xsl:value-of select="xmml:name"/>[index] = currentState-><xsl:value-of select="xmml:name"/>[index];</xsl:for-each>
			//set scan input flag to 1
			nextState->_scan_input[index] = 1;
		}
		else
		{
			//set scan input flag of current state to 1 (keep agent)
			currentState->_scan_input[index] = 1;
		}
	
	}
 }
</xsl:for-each>

<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:functions/gpu:function/gpu:globalCondition">
/** <xsl:value-of select="../xmml:name"/>_function_filter
 *	Global condition function. Flags the scan input state to true if the condition is met
 * @param currentState xmachine_memory_<xsl:value-of select="../../../xmml:name"/>_list representing agent i the current state
 */
 __global__ void <xsl:value-of select="../xmml:name"/>_function_filter(xmachine_memory_<xsl:value-of select="../../../xmml:name"/>_list* currentState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index &lt; d_xmachine_memory_<xsl:value-of select="../../../xmml:name"/>_count){
	
		//apply the filter
		if <xsl:apply-templates select="."/>
		{	currentState->_scan_input[index] = 1;
		}
		else
		{
			currentState->_scan_input[index] = 0;
		}
	
	}
 }
</xsl:for-each>


<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created <xsl:value-of select="xmml:name"/> agent functions */

/** reset_<xsl:value-of select="xmml:name"/>_scan_input
 * <xsl:value-of select="xmml:name"/> agent reset scan input function
 * @param agents The xmachine_memory_<xsl:value-of select="xmml:name"/>_list agent list
 */
__global__ void reset_<xsl:value-of select="xmml:name"/>_scan_input(xmachine_memory_<xsl:value-of select="xmml:name"/>_list* agents, int threadCt){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

    if(index &gt; threadCt) return;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}

<xsl:if test="gpu:type='continuous'">

/** scatter_<xsl:value-of select="xmml:name"/>_Agents
 * <xsl:value-of select="xmml:name"/> scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_<xsl:value-of select="xmml:name"/>_list agent list destination
 * @param agents_src xmachine_memory_<xsl:value-of select="xmml:name"/>_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_<xsl:value-of select="xmml:name"/>_Agents(xmachine_memory_<xsl:value-of select="xmml:name"/>_list* agents_dst, xmachine_memory_<xsl:value-of select="xmml:name"/>_list* agents_src, int dst_agent_count, int number_to_scatter){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = agents_src->_scan_input[index];

	//if optional message is to be written. 
	//must check agent is within number to scatter as unused threads may have scan input = 1
	if ((_scan_input == 1)&amp;&amp;(index &lt; number_to_scatter)){
		int output_index = agents_src->_position[index] + dst_agent_count;

		//AoS - xmachine_message_location Un-Coalesced scattered memory write     
        agents_dst->_position[output_index] = output_index;<xsl:for-each select="xmml:memory/gpu:variable"><xsl:choose><xsl:when test="xmml:arrayLength">
	    for (int i=0; i&lt;<xsl:value-of select="xmml:arrayLength"/>; i++){
	      agents_dst-><xsl:value-of select="xmml:name"/>[(i*xmachine_memory_<xsl:value-of select="../../xmml:name"/>_MAX)+output_index] = agents_src-><xsl:value-of select="xmml:name"/>[(i*xmachine_memory_<xsl:value-of select="../../xmml:name"/>_MAX)+index];
	    }</xsl:when><xsl:otherwise>        
		agents_dst-><xsl:value-of select="xmml:name"/>[output_index] = agents_src-><xsl:value-of select="xmml:name"/>[index];</xsl:otherwise></xsl:choose></xsl:for-each>
	}
}

/** append_<xsl:value-of select="xmml:name"/>_Agents
 * <xsl:value-of select="xmml:name"/> scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_<xsl:value-of select="xmml:name"/>_list agent list destination
 * @param agents_src xmachine_memory_<xsl:value-of select="xmml:name"/>_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_<xsl:value-of select="xmml:name"/>_Agents(xmachine_memory_<xsl:value-of select="xmml:name"/>_list* agents_dst, xmachine_memory_<xsl:value-of select="xmml:name"/>_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index &lt; number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;<xsl:for-each select="xmml:memory/gpu:variable"><xsl:choose><xsl:when test="xmml:arrayLength">
	    for (int i=0; i&lt;<xsl:value-of select="xmml:arrayLength"/>; i++){
	      agents_dst-><xsl:value-of select="xmml:name"/>[(i*xmachine_memory_<xsl:value-of select="../../xmml:name"/>_MAX)+output_index] = agents_src-><xsl:value-of select="xmml:name"/>[(i*xmachine_memory_<xsl:value-of select="../../xmml:name"/>_MAX)+index];
	    }</xsl:when><xsl:otherwise>
	    agents_dst-><xsl:value-of select="xmml:name"/>[output_index] = agents_src-><xsl:value-of select="xmml:name"/>[index];</xsl:otherwise></xsl:choose></xsl:for-each>
    }
}

/** add_<xsl:value-of select="xmml:name"/>_agent
 * Continuous <xsl:value-of select="xmml:name"/> agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_<xsl:value-of select="xmml:name"/>_list to add agents to <xsl:for-each select="xmml:memory/gpu:variable">
 * @param <xsl:value-of select="xmml:name"/> agent variable of type <xsl:value-of select="xmml:type"/></xsl:for-each>
 */
template &lt;int AGENT_TYPE&gt;
__device__ void add_<xsl:value-of select="xmml:name"/>_agent(xmachine_memory_<xsl:value-of select="xmml:name"/>_list* agents, <xsl:for-each select="xmml:memory/gpu:variable[not(xmml:arrayLength)]"><xsl:value-of select="xmml:type"/><xsl:text> </xsl:text><xsl:value-of select="xmml:name"/><xsl:if test="position()!=last()">, </xsl:if></xsl:for-each>){
	
	int index;
    
    //calculate the agents index in global agent list (depends on agent type)
	if (AGENT_TYPE == DISCRETE_2D){
		int width = (blockDim.x* gridDim.x);
		glm::ivec2 global_position;
		global_position.x = (blockIdx.x*blockDim.x) + threadIdx.x;
		global_position.y = (blockIdx.y*blockDim.y) + threadIdx.y;
		index = global_position.x + (global_position.y* width);
	}else//AGENT_TYPE == CONTINOUS
		index = threadIdx.x + blockIdx.x*blockDim.x;

	//for prefix sum
	agents->_position[index] = 0;
	agents->_scan_input[index] = 1;

	//write data to new buffer<xsl:for-each select="xmml:memory/gpu:variable"><xsl:if test="not(xmml:arrayLength)">
	agents-><xsl:value-of select="xmml:name"/>[index] = <xsl:value-of select="xmml:name"/>;</xsl:if></xsl:for-each>

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_<xsl:value-of select="xmml:name"/>_agent(xmachine_memory_<xsl:value-of select="xmml:name"/>_list* agents, <xsl:for-each select="xmml:memory/gpu:variable[not(xmml:arrayLength)]"><xsl:value-of select="xmml:type"/><xsl:text> </xsl:text><xsl:value-of select="xmml:name"/><xsl:if test="position()!=last()">, </xsl:if></xsl:for-each>){
    add_<xsl:value-of select="xmml:name"/>_agent&lt;DISCRETE_2D&gt;(agents, <xsl:for-each select="xmml:memory/gpu:variable[not(xmml:arrayLength)]"><xsl:value-of select="xmml:name"/><xsl:if test="position()!=last()">, </xsl:if></xsl:for-each>);
}

/** reorder_<xsl:value-of select="xmml:name"/>_agents
 * Continuous <xsl:value-of select="xmml:name"/> agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_<xsl:value-of select="xmml:name"/>_agents(unsigned int* values, xmachine_memory_<xsl:value-of select="xmml:name"/>_list* unordered_agents, xmachine_memory_<xsl:value-of select="xmml:name"/>_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data<xsl:for-each select="xmml:memory/gpu:variable"><xsl:choose><xsl:when test="xmml:arrayLength">
	for (int i=0; i&lt;<xsl:value-of select="xmml:arrayLength"/>; i++){
	  ordered_agents-><xsl:value-of select="xmml:name"/>[(i*xmachine_memory_<xsl:value-of select="../../xmml:name"/>_MAX)+index] = unordered_agents-><xsl:value-of select="xmml:name"/>[(i*xmachine_memory_<xsl:value-of select="../../xmml:name"/>_MAX)+old_pos];
	}</xsl:when><xsl:otherwise>
	ordered_agents-><xsl:value-of select="xmml:name"/>[index] = unordered_agents-><xsl:value-of select="xmml:name"/>[old_pos];</xsl:otherwise></xsl:choose></xsl:for-each>
}
</xsl:if>
  
<xsl:if test="xmml:memory/gpu:variable/xmml:arrayLength">
/** get_<xsl:value-of select="xmml:name"/>_agent_array_value
 *  Template function for accessing <xsl:value-of select="xmml:name"/> agent array memory variables. Assumes array points to the first element of the agents array values (offset by agent index)
 *  @param array Agent memory array
 *  @param index to lookup
 *  @return return value
 */
template&lt;typename T&gt;
__FLAME_GPU_FUNC__ T get_<xsl:value-of select="xmml:name"/>_agent_array_value(T *array, uint index){
	// Null check for out of bounds agents (brute force communication. )
	if(array != nullptr){
	    return array[index*xmachine_memory_<xsl:value-of select="xmml:name"/>_MAX];
    } else {
    	// Return the default value for this data type 
	    return <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>;
    }
}

/** set_<xsl:value-of select="xmml:name"/>_agent_array_value
 *  Template function for setting <xsl:value-of select="xmml:name"/> agent array memory variables. Assumes array points to the first element of the agents array values (offset by agent index)
 *  @param array Agent memory array
 *  @param index to lookup
 *  @param return value
 */
template&lt;typename T&gt;
__FLAME_GPU_FUNC__ void set_<xsl:value-of select="xmml:name"/>_agent_array_value(T *array, uint index, T value){
	// Null check for out of bounds agents (brute force communication. )
	if(array != nullptr){
	    array[index*xmachine_memory_<xsl:value-of select="xmml:name"/>_MAX] = value;
    }
}
</xsl:if>

</xsl:for-each>

	
<xsl:for-each select="gpu:xmodel/xmml:messages/gpu:message">
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created <xsl:value-of select="xmml:name"/> message functions */

<xsl:if test="gpu:partitioningNone or gpu:partitioningSpatial or gpu:partitioningGraphEdge">
/** add_<xsl:value-of select="xmml:name"/>_message
 * Add non partitioned or spatially partitioned <xsl:value-of select="xmml:name"/> message
 * @param messages xmachine_message_<xsl:value-of select="xmml:name"/>_list message list to add too<xsl:for-each select="xmml:variables/gpu:variable">
 * @param <xsl:value-of select="xmml:name"/> agent variable of type <xsl:value-of select="xmml:type"/></xsl:for-each>
 */
__device__ void add_<xsl:value-of select="xmml:name"/>_message(xmachine_message_<xsl:value-of select="xmml:name"/>_list* messages, <xsl:for-each select="xmml:variables/gpu:variable"><xsl:value-of select="xmml:type"/><xsl:text> </xsl:text><xsl:value-of select="xmml:name"/><xsl:if test="position()!=last()">, </xsl:if></xsl:for-each>){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_<xsl:value-of select="xmml:name"/>_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_<xsl:value-of select="xmml:name"/>_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_<xsl:value-of select="xmml:name"/>_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_<xsl:value-of select="xmml:name"/> Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;<xsl:for-each select="xmml:variables/gpu:variable">
	messages-><xsl:value-of select="xmml:name"/>[index] = <xsl:value-of select="xmml:name"/>;</xsl:for-each>

}

/**
 * Scatter non partitioned or spatially partitioned <xsl:value-of select="xmml:name"/> message (for optional messages)
 * @param messages scatter_optional_<xsl:value-of select="xmml:name"/>_messages Sparse xmachine_message_<xsl:value-of select="xmml:name"/>_list message list
 * @param message_swap temp xmachine_message_<xsl:value-of select="xmml:name"/>_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_<xsl:value-of select="xmml:name"/>_messages(xmachine_message_<xsl:value-of select="xmml:name"/>_list* messages, xmachine_message_<xsl:value-of select="xmml:name"/>_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_<xsl:value-of select="xmml:name"/>_count;

		//AoS - xmachine_message_<xsl:value-of select="xmml:name"/> Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;<xsl:for-each select="xmml:variables/gpu:variable">
		messages-><xsl:value-of select="xmml:name"/>[output_index] = messages_swap-><xsl:value-of select="xmml:name"/>[index];</xsl:for-each>				
	}
}

/** reset_<xsl:value-of select="xmml:name"/>_swaps
 * Reset non partitioned or spatially partitioned <xsl:value-of select="xmml:name"/> message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_<xsl:value-of select="xmml:name"/>_swaps(xmachine_message_<xsl:value-of select="xmml:name"/>_list* messages_swap, int threadCt){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

    if(index &gt; threadCt) return;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}
</xsl:if>

	
<xsl:if test="gpu:partitioningNone">
/* Message functions */

__device__ xmachine_message_<xsl:value-of select="xmml:name"/>* get_first_<xsl:value-of select="xmml:name"/>_message(xmachine_message_<xsl:value-of select="xmml:name"/>_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&amp;sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_<xsl:value-of select="xmml:name"/>_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_<xsl:value-of select="xmml:name"/> Coalesced memory read
	xmachine_message_<xsl:value-of select="xmml:name"/> temp_message;
	temp_message._position = messages->_position[index];<xsl:for-each select="xmml:variables/gpu:variable">
	temp_message.<xsl:value-of select="xmml:name"/> = messages-><xsl:value-of select="xmml:name"/>[index];</xsl:for-each>

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>));
	xmachine_message_<xsl:value-of select="xmml:name"/>* sm_message = ((xmachine_message_<xsl:value-of select="xmml:name"/>*)&amp;message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_<xsl:value-of select="xmml:name"/>*)&amp;message_share[d_SM_START]);
}

__device__ xmachine_message_<xsl:value-of select="xmml:name"/>* get_next_<xsl:value-of select="xmml:name"/>_message(xmachine_message_<xsl:value-of select="xmml:name"/>* message, xmachine_message_<xsl:value-of select="xmml:name"/>_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&amp;sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_<xsl:value-of select="xmml:name"/>_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_<xsl:value-of select="xmml:name"/>_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_<xsl:value-of select="xmml:name"/> Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_<xsl:value-of select="xmml:name"/> temp_message;
		temp_message._position = messages->_position[index];<xsl:for-each select="xmml:variables/gpu:variable">
		temp_message.<xsl:value-of select="xmml:name"/> = messages-><xsl:value-of select="xmml:name"/>[index];</xsl:for-each>

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>));
		xmachine_message_<xsl:value-of select="xmml:name"/>* sm_message = ((xmachine_message_<xsl:value-of select="xmml:name"/>*)&amp;message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>));
	return ((xmachine_message_<xsl:value-of select="xmml:name"/>*)&amp;message_share[message_index]);
}
</xsl:if>
	
	
<xsl:if test="gpu:partitioningDiscrete">
/* Message functions */

template &lt;int AGENT_TYPE&gt;
__device__ void add_<xsl:value-of select="xmml:name"/>_message(xmachine_message_<xsl:value-of select="xmml:name"/>_list* messages, <xsl:for-each select="xmml:variables/gpu:variable"><xsl:value-of select="xmml:type"/><xsl:text> </xsl:text><xsl:value-of select="xmml:name"/><xsl:if test="position()!=last()">, </xsl:if></xsl:for-each>){
	if (AGENT_TYPE == DISCRETE_2D){
		int width = (blockDim.x * gridDim.x);
		glm::ivec2 global_position;
		global_position.x = (blockIdx.x * blockDim.x) + threadIdx.x;
		global_position.y = (blockIdx.y * blockDim.y) + threadIdx.y;

		int index = global_position.x + (global_position.y * width);

		<xsl:for-each select="xmml:variables/gpu:variable">
		messages-><xsl:value-of select="xmml:name"/>[index] = <xsl:value-of select="xmml:name"/>;			</xsl:for-each>
	}
	//else CONTINUOUS agents can not write to discrete space
}

//Used by continuous agents this accesses messages with texture cache. agent_x and agent_y are discrete positions in the message space
__device__ xmachine_message_<xsl:value-of select="xmml:name"/>* get_first_<xsl:value-of select="xmml:name"/>_message_continuous(xmachine_message_<xsl:value-of select="xmml:name"/>_list* messages,  int agent_x, int agent_y){

	//shared memory get from offset dependant on sm usage in function
	extern __shared__ int sm_data [];

	xmachine_message_<xsl:value-of select="xmml:name"/>* message_share = (xmachine_message_<xsl:value-of select="xmml:name"/>*)&amp;sm_data[0];
	
	int range = d_message_<xsl:value-of select="xmml:name"/>_range;
	int width = d_message_<xsl:value-of select="xmml:name"/>_width;
	
	glm::ivec2 global_position;
	global_position.x = sWRAP(agent_x-range , width);
	global_position.y = sWRAP(agent_y-range , width);
	

	int index = ((global_position.y)* width) + global_position.x;
	
	xmachine_message_<xsl:value-of select="xmml:name"/> temp_message;
	temp_message._position = glm::ivec2(agent_x, agent_y);
	temp_message._relative = glm::ivec2(-range, -range);

	<xsl:for-each select="xmml:variables/gpu:variable">
  <xsl:choose>
  <xsl:when test="xmml:type='double'">temp_message.<xsl:value-of select="xmml:name"/> = tex1DfetchDouble(tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>, index + d_tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_offset);
  </xsl:when><xsl:otherwise>temp_message.<xsl:value-of select="xmml:name"/> = tex1Dfetch(tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>, index + d_tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_offset);</xsl:otherwise></xsl:choose>		</xsl:for-each>
	
	message_share[threadIdx.x] = temp_message;

	//return top left of messages
	return &amp;message_share[threadIdx.x];
}

//Get next <xsl:value-of select="xmml:name"/> message  continuous
//Used by continuous agents this accesses messages with texture cache (agent position in discrete space was set when accessing first message)
__device__ xmachine_message_<xsl:value-of select="xmml:name"/>* get_next_<xsl:value-of select="xmml:name"/>_message_continuous(xmachine_message_<xsl:value-of select="xmml:name"/>* message, xmachine_message_<xsl:value-of select="xmml:name"/>_list* messages){

	//shared memory get from offset dependant on sm usage in function
	extern __shared__ int sm_data [];

	xmachine_message_<xsl:value-of select="xmml:name"/>* message_share = (xmachine_message_<xsl:value-of select="xmml:name"/>*)&amp;sm_data[0];
	
	int range = d_message_<xsl:value-of select="xmml:name"/>_range;
	int width = d_message_<xsl:value-of select="xmml:name"/>_width;

	//Get previous position
	glm::ivec2 previous_relative = message->_relative;

	//exit if at (range, range)
	if (previous_relative.x == (range))
        if (previous_relative.y == (range))
		    return nullptr;

	//calculate next message relative position
	glm::ivec2 next_relative = previous_relative;
	next_relative.x += 1;
	if ((next_relative.x)>range){
		next_relative.x = -range;
		next_relative.y = previous_relative.y + 1;
	}

	//skip own message
	if (next_relative.x == 0)
        if (next_relative.y == 0)
		    next_relative.x += 1;

	glm::ivec2 global_position;
	global_position.x =	sWRAP(message->_position.x + next_relative.x, width);
	global_position.y = sWRAP(message->_position.y + next_relative.y, width);

	int index = ((global_position.y)* width) + (global_position.x);
	
	xmachine_message_<xsl:value-of select="xmml:name"/> temp_message;
	temp_message._position = message->_position;
	temp_message._relative = next_relative;

	<xsl:for-each select="xmml:variables/gpu:variable">
  <xsl:choose>
  <xsl:when test="xmml:type='double'">temp_message.<xsl:value-of select="xmml:name"/> = tex1DfetchDouble(tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>, index + d_tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_offset);</xsl:when>
  <xsl:otherwise>temp_message.<xsl:value-of select="xmml:name"/> = tex1Dfetch(tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>, index + d_tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_offset);	</xsl:otherwise></xsl:choose>	</xsl:for-each>

	message_share[threadIdx.x] = temp_message;

	return &amp;message_share[threadIdx.x];
}

//method used by discrete agents accessing discrete messages to load messages into shared memory
__device__ void <xsl:value-of select="xmml:name"/>_message_to_sm(xmachine_message_<xsl:value-of select="xmml:name"/>_list* messages, char* message_share, int sm_index, int global_index){
		xmachine_message_<xsl:value-of select="xmml:name"/> temp_message;
		<xsl:for-each select="xmml:variables/gpu:variable">
		temp_message.<xsl:value-of select="xmml:name"/> = messages-><xsl:value-of select="xmml:name"/>[global_index];		</xsl:for-each>

	  int message_index = SHARE_INDEX(sm_index, sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>));
	  xmachine_message_<xsl:value-of select="xmml:name"/>* sm_message = ((xmachine_message_<xsl:value-of select="xmml:name"/>*)&amp;message_share[message_index]);
	  sm_message[0] = temp_message;
}

//Get first <xsl:value-of select="xmml:name"/> message 
//Used by discrete agents this accesses messages with texture cache. Agent position is determined by position in the grid/block
//Possibility of upto 8 thread divergences
__device__ xmachine_message_<xsl:value-of select="xmml:name"/>* get_first_<xsl:value-of select="xmml:name"/>_message_discrete(xmachine_message_<xsl:value-of select="xmml:name"/>_list* messages){

	//shared memory get from offset dependant on sm usage in function
	extern __shared__ int sm_data [];

	char* message_share = (char*)&amp;sm_data[0];
  
	__syncthreads();

	int range = d_message_<xsl:value-of select="xmml:name"/>_range;
	int width = d_message_<xsl:value-of select="xmml:name"/>_width;
	int sm_grid_width = blockDim.x + (range* 2);
	
	
	glm::ivec2 global_position;
	global_position.x = (blockIdx.x * blockDim.x) + threadIdx.x;
	global_position.y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = global_position.x + (global_position.y * width);
	

	//calculate the position in shared memory of first load
	glm::ivec2 sm_pos;
	sm_pos.x = threadIdx.x + range;
	sm_pos.y = threadIdx.y + range;
	int sm_index = (sm_pos.y * sm_grid_width) + sm_pos.x;

	//each thread loads to shared memory (coalesced read)
	<xsl:value-of select="xmml:name"/>_message_to_sm(messages, message_share, sm_index, index);

	//check for edge conditions
	int left_border = (threadIdx.x &lt; range);
	int right_border = (threadIdx.x &gt;= (blockDim.x-range));
	int top_border = (threadIdx.y &lt; range);
	int bottom_border = (threadIdx.y &gt;= (blockDim.y-range));

	
	int  border_index;
	int  sm_border_index;

	//left
	if (left_border){	
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x - range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = (sm_pos.y * sm_grid_width) + threadIdx.x;
		
		<xsl:value-of select="xmml:name"/>_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//right
	if (right_border){
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x + range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = (sm_pos.y * sm_grid_width) + (sm_pos.x + range);

		<xsl:value-of select="xmml:name"/>_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//top
	if (top_border){
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.y = sWRAP(border_index_2d.y - range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = (threadIdx.y * sm_grid_width) + sm_pos.x;

		<xsl:value-of select="xmml:name"/>_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//bottom
	if (bottom_border){
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.y = sWRAP(border_index_2d.y + range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = ((sm_pos.y + range) * sm_grid_width) + sm_pos.x;

		<xsl:value-of select="xmml:name"/>_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//top left
	if ((top_border)&amp;&amp;(left_border)){	
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x - range, width);
		border_index_2d.y = sWRAP(border_index_2d.y - range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = (threadIdx.y * sm_grid_width) + threadIdx.x;
		
		<xsl:value-of select="xmml:name"/>_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//top right
	if ((top_border)&amp;&amp;(right_border)){	
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x + range, width);
		border_index_2d.y = sWRAP(border_index_2d.y - range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = (threadIdx.y * sm_grid_width) + (sm_pos.x + range);
		
		<xsl:value-of select="xmml:name"/>_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//bottom right
	if ((bottom_border)&amp;&amp;(right_border)){	
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x + range, width);
		border_index_2d.y = sWRAP(border_index_2d.y + range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = ((sm_pos.y + range) * sm_grid_width) + (sm_pos.x + range);
		
		<xsl:value-of select="xmml:name"/>_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//bottom left
	if ((bottom_border)&amp;&amp;(left_border)){	
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x - range, width);
		border_index_2d.y = sWRAP(border_index_2d.y + range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = ((sm_pos.y + range) * sm_grid_width) + threadIdx.x;
		
		<xsl:value-of select="xmml:name"/>_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	__syncthreads();
	
  
	//top left of block position sm index
	sm_index = (threadIdx.y * sm_grid_width) + threadIdx.x;
	
	int message_index = SHARE_INDEX(sm_index, sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>));
	xmachine_message_<xsl:value-of select="xmml:name"/>* temp = ((xmachine_message_<xsl:value-of select="xmml:name"/>*)&amp;message_share[message_index]);
	temp->_relative = glm::ivec2(-range, -range); //this is the relative position
	return temp;
}

//Get next <xsl:value-of select="xmml:name"/> message 
//Used by discrete agents this accesses messages through shared memory which were all loaded on first message retrieval call.
__device__ xmachine_message_<xsl:value-of select="xmml:name"/>* get_next_<xsl:value-of select="xmml:name"/>_message_discrete(xmachine_message_<xsl:value-of select="xmml:name"/>* message, xmachine_message_<xsl:value-of select="xmml:name"/>_list* messages){

	//shared memory get from offset dependant on sm usage in function
	extern __shared__ int sm_data [];

	char* message_share = (char*)&amp;sm_data[0];
  
	__syncthreads();
	
	int range = d_message_<xsl:value-of select="xmml:name"/>_range;
	int sm_grid_width = blockDim.x+(range*2);


	//Get previous position
	glm::ivec2 previous_relative = message->_relative;

	//exit if at (range, range)
	if (previous_relative.x == range)
        if (previous_relative.y == range)
		    return nullptr;

	//calculate next message relative position
	glm::ivec2 next_relative = previous_relative;
	next_relative.x += 1;
	if ((next_relative.x)>range){
		next_relative.x = -range;
		next_relative.y = previous_relative.y + 1;
	}

	//skip own message
	if (next_relative.x == 0)
        if (next_relative.y == 0)
		    next_relative.x += 1;


	//calculate the next message position
	glm::ivec2 next_position;// = block_position+next_relative;
	//offset next position by the sm border size
	next_position.x = threadIdx.x + next_relative.x + range;
	next_position.y = threadIdx.y + next_relative.y + range;

	int sm_index = next_position.x + (next_position.y * sm_grid_width);
	
	__syncthreads();
  
	int message_index = SHARE_INDEX(sm_index, sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>));
	xmachine_message_<xsl:value-of select="xmml:name"/>* temp = ((xmachine_message_<xsl:value-of select="xmml:name"/>*)&amp;message_share[message_index]);
	temp->_relative = next_relative; //this is the relative position
	return temp;
}

//Get first <xsl:value-of select="xmml:name"/> message
template &lt;int AGENT_TYPE&gt;
__device__ xmachine_message_<xsl:value-of select="xmml:name"/>* get_first_<xsl:value-of select="xmml:name"/>_message(xmachine_message_<xsl:value-of select="xmml:name"/>_list* messages, int agent_x, int agent_y){

	if (AGENT_TYPE == DISCRETE_2D)	//use shared memory method
		return get_first_<xsl:value-of select="xmml:name"/>_message_discrete(messages);
	else	//use texture fetching method
		return get_first_<xsl:value-of select="xmml:name"/>_message_continuous(messages, agent_x, agent_y);

}

//Get next <xsl:value-of select="xmml:name"/> message
template &lt;int AGENT_TYPE&gt;
__device__ xmachine_message_<xsl:value-of select="xmml:name"/>* get_next_<xsl:value-of select="xmml:name"/>_message(xmachine_message_<xsl:value-of select="xmml:name"/>* message, xmachine_message_<xsl:value-of select="xmml:name"/>_list* messages){

	if (AGENT_TYPE == DISCRETE_2D)	//use shared memory method
		return get_next_<xsl:value-of select="xmml:name"/>_message_discrete(message, messages);
	else	//use texture fetching method
		return get_next_<xsl:value-of select="xmml:name"/>_message_continuous(message, messages);

}
</xsl:if>
<xsl:if test="gpu:partitioningSpatial">
/* Message functions */

/** message_<xsl:value-of select="xmml:name"/>_grid_position
 * Calculates the grid cell position given an glm::vec3 vector
 * @param position glm::vec3 vector representing a position
 */
__device__ glm::ivec3 message_<xsl:value-of select="xmml:name"/>_grid_position(glm::vec3 position)
{
    glm::ivec3 gridPos;
    gridPos.x = floor((position.x - d_message_<xsl:value-of select="xmml:name"/>_min_bounds.x) * (float)d_message_<xsl:value-of select="xmml:name"/>_partitionDim.x / (d_message_<xsl:value-of select="xmml:name"/>_max_bounds.x - d_message_<xsl:value-of select="xmml:name"/>_min_bounds.x));
    gridPos.y = floor((position.y - d_message_<xsl:value-of select="xmml:name"/>_min_bounds.y) * (float)d_message_<xsl:value-of select="xmml:name"/>_partitionDim.y / (d_message_<xsl:value-of select="xmml:name"/>_max_bounds.y - d_message_<xsl:value-of select="xmml:name"/>_min_bounds.y));
    gridPos.z = floor((position.z - d_message_<xsl:value-of select="xmml:name"/>_min_bounds.z) * (float)d_message_<xsl:value-of select="xmml:name"/>_partitionDim.z / (d_message_<xsl:value-of select="xmml:name"/>_max_bounds.z - d_message_<xsl:value-of select="xmml:name"/>_min_bounds.z));

	//do wrapping or bounding
	

    return gridPos;
}

/** message_<xsl:value-of select="xmml:name"/>_hash
 * Given the grid position in partition space this function calculates a hash value
 * @param gridPos The position in partition space
 */
__device__ unsigned int message_<xsl:value-of select="xmml:name"/>_hash(glm::ivec3 gridPos)
{
	//cheap bounding without mod (within range +- partition dimension)
	gridPos.x = (gridPos.x&lt;0)? d_message_<xsl:value-of select="xmml:name"/>_partitionDim.x-1: gridPos.x; 
	gridPos.x = (gridPos.x>=d_message_<xsl:value-of select="xmml:name"/>_partitionDim.x)? 0 : gridPos.x; 
	gridPos.y = (gridPos.y&lt;0)? d_message_<xsl:value-of select="xmml:name"/>_partitionDim.y-1 : gridPos.y; 
	gridPos.y = (gridPos.y>=d_message_<xsl:value-of select="xmml:name"/>_partitionDim.y)? 0 : gridPos.y; 
	gridPos.z = (gridPos.z&lt;0)? d_message_<xsl:value-of select="xmml:name"/>_partitionDim.z-1: gridPos.z; 
	gridPos.z = (gridPos.z>=d_message_<xsl:value-of select="xmml:name"/>_partitionDim.z)? 0 : gridPos.z; 

	//unique id
	return ((gridPos.z * d_message_<xsl:value-of select="xmml:name"/>_partitionDim.y) * d_message_<xsl:value-of select="xmml:name"/>_partitionDim.x) + (gridPos.y * d_message_<xsl:value-of select="xmml:name"/>_partitionDim.x) + gridPos.x;
}

#ifdef FAST_ATOMIC_SORTING
	/** hist_<xsl:value-of select="xmml:name"/>_messages
		 * Kernal function for performing a histogram (count) on each partition bin and saving the hash and index of a message within that bin
		 * @param local_bin_index output index of the message within the calculated bin
		 * @param unsorted_index output bin index (hash) value
		 * @param messages the message list used to generate the hash value outputs
		 * @param agent_count the current number of agents outputting messages
		 */
	__global__ void hist_<xsl:value-of select="xmml:name"/>_messages(uint* local_bin_index, uint* unsorted_index, int* global_bin_count, xmachine_message_<xsl:value-of select="xmml:name"/>_list* messages, int agent_count)
	{
		unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		if (index >= agent_count)
			return;
        glm::vec3 position = glm::vec3(messages->x[index], messages->y[index], messages->z[index]);
		glm::ivec3 grid_position = message_<xsl:value-of select="xmml:name"/>_grid_position(position);
		unsigned int hash = message_<xsl:value-of select="xmml:name"/>_hash(grid_position);
		unsigned int bin_idx = atomicInc((unsigned int*) &amp;global_bin_count[hash], 0xFFFFFFFF);
		local_bin_index[index] = bin_idx;
		unsorted_index[index] = hash;
	}
	
	/** reorder_<xsl:value-of select="xmml:name"/>_messages
	 * Reorders the messages accoring to the partition boundary matrix start indices of each bin
	 * @param local_bin_index index of the message within the desired bin
	 * @param unsorted_index bin index (hash) value
	 * @param pbm_start_index the start indices of the partition boundary matrix
	 * @param unordered_messages the original unordered message data
	 * @param ordered_messages buffer used to scatter messages into the correct order
	  @param agent_count the current number of agents outputting messages
	 */
	 __global__ void reorder_<xsl:value-of select="xmml:name"/>_messages(uint* local_bin_index, uint* unsorted_index, int* pbm_start_index, xmachine_message_<xsl:value-of select="xmml:name"/>_list* unordered_messages, xmachine_message_<xsl:value-of select="xmml:name"/>_list* ordered_messages, int agent_count)
	{
		int index = (blockIdx.x *blockDim.x) + threadIdx.x;

		if (index >= agent_count)
			return;

		uint i = unsorted_index[index];
		unsigned int sorted_index = local_bin_index[index] + pbm_start_index[i];

		//finally reorder agent data<xsl:for-each select="xmml:variables/gpu:variable">
		ordered_messages-><xsl:value-of select="xmml:name"/>[sorted_index] = unordered_messages-><xsl:value-of select="xmml:name"/>[index];</xsl:for-each>
	}
	 
#else

	/** hash_<xsl:value-of select="xmml:name"/>_messages
	 * Kernal function for calculating a hash value for each messahe depending on its position
	 * @param keys output for the hash key
	 * @param values output for the index value
	 * @param messages the message list used to generate the hash value outputs
	 */
	__global__ void hash_<xsl:value-of select="xmml:name"/>_messages(uint* keys, uint* values, xmachine_message_<xsl:value-of select="xmml:name"/>_list* messages)
	{
		unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        glm::vec3 position = glm::vec3(messages->x[index], messages->y[index], messages->z[index]);
		glm::ivec3 grid_position = message_<xsl:value-of select="xmml:name"/>_grid_position(position);
		unsigned int hash = message_<xsl:value-of select="xmml:name"/>_hash(grid_position);

		keys[index] = hash;
		values[index] = index;
	}

	/** reorder_<xsl:value-of select="xmml:name"/>_messages
	 * Reorders the messages accoring to the ordered sort identifiers and builds a Partition Boundary Matrix by looking at the previosu threads sort id.
	 * @param keys the sorted hash keys
	 * @param values the sorted index values
	 * @param matrix the PBM
	 * @param unordered_messages the original unordered message data
	 * @param ordered_messages buffer used to scatter messages into the correct order
	 */
	__global__ void reorder_<xsl:value-of select="xmml:name"/>_messages(uint* keys, uint* values, xmachine_message_<xsl:value-of select="xmml:name"/>_PBM* matrix, xmachine_message_<xsl:value-of select="xmml:name"/>_list* unordered_messages, xmachine_message_<xsl:value-of select="xmml:name"/>_list* ordered_messages)
	{
		extern __shared__ int sm_data [];

		int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		//load threads sort key into sm
		uint key = keys[index];
		uint old_pos = values[index];

		sm_data[threadIdx.x] = key;
		__syncthreads();
	
		unsigned int prev_key;

		//if first thread then no prev sm value so get prev from global memory 
		if (threadIdx.x == 0)
		{
			//first thread has no prev value so ignore
			if (index != 0)
				prev_key = keys[index-1];
		}
		//get previous ident from sm
		else	
		{
			prev_key = sm_data[threadIdx.x-1];
		}

		//TODO: Check key is not out of bounds

		//set partition boundaries
		if (index &lt; d_message_<xsl:value-of select="xmml:name"/>_count)
		{
			//if first thread then set first partition cell start
			if (index == 0)
			{
				matrix->start[key] = index;
			}

			//if edge of a boundr update start and end of partition
			else if (prev_key != key)
			{
				//set start for key
				matrix->start[key] = index;

				//set end for key -1
				matrix->end_or_count[prev_key] = index;
			}

			//if last thread then set final partition cell end
			if (index == d_message_<xsl:value-of select="xmml:name"/>_count-1)
			{
				matrix->end_or_count[key] = index+1;
			}
		}
	
		//finally reorder agent data<xsl:for-each select="xmml:variables/gpu:variable">
		ordered_messages-><xsl:value-of select="xmml:name"/>[index] = unordered_messages-><xsl:value-of select="xmml:name"/>[old_pos];</xsl:for-each>
	}

#endif

/** load_next_<xsl:value-of select="xmml:name"/>_message
 * Used to load the next message data to shared memory
 * Idea is check the current cell index to see if we can simply get a message from the current cell
 * If we are at the end of the current cell then loop till we find the next cell with messages (this way we ignore cells with no messages)
 * @param messages the message list
 * @param partition_matrix the PBM
 * @param relative_cell the relative partition cell position from the agent position
 * @param cell_index_max the maximum index of the current partition cell
 * @param agent_grid_cell the agents partition cell position
 * @param cell_index the current cell index in agent_grid_cell+relative_cell
 * @return true if a message has been loaded into sm false otherwise
 */
__device__ bool load_next_<xsl:value-of select="xmml:name"/>_message(xmachine_message_<xsl:value-of select="xmml:name"/>_list* messages, xmachine_message_<xsl:value-of select="xmml:name"/>_PBM* partition_matrix, glm::ivec3 relative_cell, int cell_index_max, glm::ivec3 agent_grid_cell, int cell_index)
{
	extern __shared__ int sm_data [];
	char* message_share = (char*)&amp;sm_data[0];

	int move_cell = true;
	cell_index ++;

	//see if we need to move to a new partition cell
	if(cell_index &lt; cell_index_max)
		move_cell = false;

	while(move_cell)
	{
		//get the next relative grid position <!-- check the z component to see if we are operating in 2d or 3d -->
        if (next_cell<xsl:choose><xsl:when test="ceiling((gpu:partitioningSpatial/gpu:zmax - gpu:partitioningSpatial/gpu:zmin) div gpu:partitioningSpatial/gpu:radius) = 1">2D</xsl:when><xsl:otherwise>3D</xsl:otherwise></xsl:choose>(&amp;relative_cell))
		{
			//calculate the next cells grid position and hash
			glm::ivec3 next_cell_position = agent_grid_cell + relative_cell;
			int next_cell_hash = message_<xsl:value-of select="xmml:name"/>_hash(next_cell_position);
			//use the hash to calculate the start index
			int cell_index_min = tex1Dfetch(tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_start, next_cell_hash + d_tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_start_offset);
			cell_index_max = tex1Dfetch(tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_end_or_count, next_cell_hash + d_tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_end_or_count_offset);
			//check for messages in the cell (cell index max is the count for atomic sorting)
#ifdef FAST_ATOMIC_SORTING
			if (cell_index_max > 0)
			{
				//when using fast atomics value represents bin count not last index!
				cell_index_max += cell_index_min; //when using fast atomics value represents bin count not last index!
#else
			if (cell_index_min != 0xffffffff)
			{
#endif
				//start from the cell index min
				cell_index = cell_index_min;
				//exit the loop as we have found a valid cell with message data
				move_cell = false;
			}
		}
		else
		{
			//we have exhausted all the neighbouring cells so there are no more messages
			return false;
		}
	}
	
	//get message data using texture fetch
	xmachine_message_<xsl:value-of select="xmml:name"/> temp_message;
	temp_message._relative_cell = relative_cell;
	temp_message._cell_index_max = cell_index_max;
	temp_message._cell_index = cell_index;
	temp_message._agent_grid_cell = agent_grid_cell;

	//Using texture cache
  <xsl:for-each select="xmml:variables/gpu:variable">
  <xsl:choose>
  <xsl:when test="xmml:type='double'">temp_message.<xsl:value-of select="xmml:name"/> = tex1DfetchDouble(tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>, cell_index + d_tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_offset);</xsl:when>
  <xsl:otherwise>temp_message.<xsl:value-of select="xmml:name"/> = tex1Dfetch(tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>, cell_index + d_tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_offset); </xsl:otherwise></xsl:choose> </xsl:for-each>

	//load it into shared memory (no sync as no sharing between threads)
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>));
	xmachine_message_<xsl:value-of select="xmml:name"/>* sm_message = ((xmachine_message_<xsl:value-of select="xmml:name"/>*)&amp;message_share[message_index]);
	sm_message[0] = temp_message;

	return true;
}

<!-- @note - The following functions check the number of messages prior to attempting to load a message. This resolves an issue realted to spatially partitioned messages being output by conditional functions.
If no agents meet the condition, the PBM is not reset. 
This has been fixed using the computationally cheap soluton of checking the number of messages.
It may be more correct to instead ensure that the PBM is reset. -->
/*
 * get first spatial partitioned <xsl:value-of select="xmml:name"/> message (first batch load into shared memory)
 */
__device__ xmachine_message_<xsl:value-of select="xmml:name"/>* get_first_<xsl:value-of select="xmml:name"/>_message(xmachine_message_<xsl:value-of select="xmml:name"/>_list* messages, xmachine_message_<xsl:value-of select="xmml:name"/>_PBM* partition_matrix, float x, float y, float z){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&amp;sm_data[0];

	// If there are no messages, do not load any messages
	if(d_message_<xsl:value-of select="xmml:name"/>_count == 0){
		return nullptr;
	}

	glm::ivec3 relative_cell = glm::ivec3(-2, -1, -1);
	int cell_index_max = 0;
	int cell_index = 0;
	glm::vec3 position = glm::vec3(x, y, z);
	glm::ivec3 agent_grid_cell = message_<xsl:value-of select="xmml:name"/>_grid_position(position);
	
	if (load_next_<xsl:value-of select="xmml:name"/>_message(messages, partition_matrix, relative_cell, cell_index_max, agent_grid_cell, cell_index))
	{
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>));
		return ((xmachine_message_<xsl:value-of select="xmml:name"/>*)&amp;message_share[message_index]);
	}
	else
	{
		return nullptr;
	}
}

/*
 * get next spatial partitioned <xsl:value-of select="xmml:name"/> message (either from SM or next batch load)
 */
__device__ xmachine_message_<xsl:value-of select="xmml:name"/>* get_next_<xsl:value-of select="xmml:name"/>_message(xmachine_message_<xsl:value-of select="xmml:name"/>* message, xmachine_message_<xsl:value-of select="xmml:name"/>_list* messages, xmachine_message_<xsl:value-of select="xmml:name"/>_PBM* partition_matrix){
	
	extern __shared__ int sm_data [];
	char* message_share = (char*)&amp;sm_data[0];
	
	// If there are no messages, do not load any messages
	if(d_message_<xsl:value-of select="xmml:name"/>_count == 0){
		return nullptr;
	}
	
	if (load_next_<xsl:value-of select="xmml:name"/>_message(messages, partition_matrix, message->_relative_cell, message->_cell_index_max, message->_agent_grid_cell, message->_cell_index))
	{
		//get conflict free address of 
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>));
		return ((xmachine_message_<xsl:value-of select="xmml:name"/>*)&amp;message_share[message_index]);
	}
	else
		return nullptr;
	
}

</xsl:if>

<xsl:if test="gpu:partitioningGraphEdge">
<xsl:variable name="message_name" select="xmml:name" />
<xsl:variable name="edge_variable_name" select="gpu:partitioningGraphEdge/gpu:messageEdgeID"/>
<xsl:variable name="edge_variable_type" select="xmml:variables/gpu:variable[xmml:name=$edge_variable_name]/xmml:type"/>
/* Message functions */

/*
 * Load the next graph edge partitioned <xsl:value-of select="xmml:name"/> messag (either from SM or next batch load)
 * @param messages message list 
 * @param message_bounds edge graph messaging data structure
 * @param <xsl:value-of select="gpu:partitioningGraphEdge/gpu:messageEdgeID"/> target edge index
 * @param messageIndex index of the message
 * @return boolean indicating if a message was loaded or not.
 */
__device__ bool load_<xsl:value-of select="xmml:name"/>_message(xmachine_message_<xsl:value-of select="xmml:name"/>_list* messages, xmachine_message_<xsl:value-of select="xmml:name"/>_bounds* message_bounds, unsigned int <xsl:value-of select="gpu:partitioningGraphEdge/gpu:messageEdgeID"/>, unsigned int messageIndex){
	// Define smem stuff
	extern __shared__ int sm_data[];
	char* message_share = (char*)&amp;sm_data[0];

	// If the taget message is greater than the number of messages return false.
	if (messageIndex >= d_message_<xsl:value-of select="xmml:name"/>_count){
		return false;
	}
	
	// Load the max value from the boundary struct.
	unsigned int firstMessageForEdge = message_bounds->start[<xsl:value-of select="gpu:partitioningGraphEdge/gpu:messageEdgeID"/>];
	
	unsigned int messageForNextEdge = firstMessageForEdge + message_bounds->count[<xsl:value-of select="gpu:partitioningGraphEdge/gpu:messageEdgeID"/>];


	// If there are no other messages return false
	if (messageIndex &lt; firstMessageForEdge || messageIndex &gt;= messageForNextEdge){
		return false;
	}

	// Get the message data for the target message
	xmachine_message_<xsl:value-of select="xmml:name"/> temp_message;
	temp_message._position = messages-&gt;_position[messageIndex];
	<xsl:for-each select="xmml:variables/gpu:variable">temp_message.<xsl:value-of select="xmml:name"/> = messages-&gt;<xsl:value-of select="xmml:name"/>[messageIndex];
	</xsl:for-each>

	// Load the message into shared memory. No sync?
	<!-- @optimisation better use of shared memory - currently each thread pulls the same value into shared memory? -->
	int message_index = SHARE_INDEX(threadIdx.y * blockDim.x + threadIdx.x, sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>));
	xmachine_message_<xsl:value-of select="xmml:name"/>* sm_message = ((xmachine_message_<xsl:value-of select="xmml:name"/>*)&amp;message_share[message_index]);
	sm_message[0] = temp_message;
	
	return true;
}

/**
 * Get the first message from the <xsl:value-of select="xmml:name"/> edge partitioned message list
 * @param messages  the message list
 * @param message_bounds boundary data structure for edge partitioned messages
 * @param <xsl:value-of select="gpu:partitioningGraphEdge/gpu:messageEdgeID"/> target edge for messages
 * @return pointer to the message.
 */
__device__ xmachine_message_<xsl:value-of select="xmml:name"/>* get_first_<xsl:value-of select="xmml:name"/>_message(xmachine_message_<xsl:value-of select="xmml:name"/>_list* messages, xmachine_message_<xsl:value-of select="xmml:name"/>_bounds* message_bounds, unsigned int <xsl:value-of select="gpu:partitioningGraphEdge/gpu:messageEdgeID"/>){

	extern __shared__ int sm_data[];
	char* message_share = (char*)&amp;sm_data[0];

	// Get the first index for the target edge.
	unsigned int firstMessageIndex = message_bounds-&gt;start[<xsl:value-of select="gpu:partitioningGraphEdge/gpu:messageEdgeID"/>];

	if (load_<xsl:value-of select="xmml:name"/>_message(messages, message_bounds, <xsl:value-of select="gpu:partitioningGraphEdge/gpu:messageEdgeID"/>, firstMessageIndex))
	{
		unsigned int message_index = SHARE_INDEX(threadIdx.y*blockDim.x + threadIdx.x, sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>));
		return ((xmachine_message_<xsl:value-of select="xmml:name"/>*)&amp;message_share[message_index]);
	}
	else
	{
		return nullptr;
	}
}

/**
 * Get the next message from the <xsl:value-of select="xmml:name"/> edge partitioned message list
 * @param messages  the message list
 * @param message_bounds boundary data structure for edge partitioned messages
 * @return pointer to the message.
 */
__device__ xmachine_message_<xsl:value-of select="xmml:name"/>* get_next_<xsl:value-of select="xmml:name"/>_message(xmachine_message_<xsl:value-of select="xmml:name"/>* message, xmachine_message_<xsl:value-of select="xmml:name"/>_list* messages, xmachine_message_<xsl:value-of select="xmml:name"/>_bounds* message_bounds){
	extern __shared__ int sm_data[];
	char* message_share = (char*)&amp;sm_data[0];

	if (load_<xsl:value-of select="xmml:name"/>_message(messages, message_bounds, message-&gt;<xsl:value-of select="gpu:partitioningGraphEdge/gpu:messageEdgeID"/>, message-&gt;_position + 1))
	{
		//get conflict free address of 
		unsigned int message_index = SHARE_INDEX(threadIdx.y*blockDim.x + threadIdx.x, sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>));
		return ((xmachine_message_<xsl:value-of select="xmml:name"/>*)&amp;message_share[message_index]);
	}
	else {
		return nullptr;
	}
	
}

/**
 * Generate a histogram of <xsl:value-of select="xmml:name"/> messages per edge index
 * @param local_index
 * @param unsorted_index
 * @param message_counts
 * @param messages
 * @param agent_count
 */
__global__ void hist_<xsl:value-of select="xmml:name"/>_messages(unsigned int* local_index, unsigned int * unsorted_index, unsigned int* message_counts, xmachine_message_<xsl:value-of select="xmml:name"/>_list * messages, unsigned int agent_count){
	unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

	if (index &gt;= agent_count){
		return;
	}
	<xsl:value-of select="$edge_variable_type"/><xsl:text> </xsl:text><xsl:value-of select="$edge_variable_name"/> = messages-&gt;<xsl:value-of select="$edge_variable_name"/>[index];
	unsigned int bin_index = atomicInc((unsigned int*)&amp;message_counts[<xsl:value-of select="$edge_variable_name"/>], 0xFFFFFFFF);
	local_index[index] = bin_index;
	unsorted_index[index] = <xsl:value-of select="$edge_variable_name"/>;
}

/**
 * Reorder <xsl:value-of select="xmml:name"/> messages for edge partitioned communication
 * @param local_index
 * @param unsorted_index
 * @param start_index
 * @param unordered_messages
 * @param ordered_messages
 * @param agent_count
 */
__global__ void reorder_<xsl:value-of select="xmml:name"/>_messages(unsigned int* local_index, unsigned int* unsorted_index, unsigned int* start_index, xmachine_message_<xsl:value-of select="xmml:name"/>_list* unordered_messages, xmachine_message_<xsl:value-of select="xmml:name"/>_list* ordered_messages, unsigned int agent_count){
	unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

	if (index &gt;= agent_count){
		return;
	}

	unsigned int sorted_index = local_index[index] + start_index[unsorted_index[index]];
	
	// The position value should be updated to reflect the new position.
	ordered_messages->_position[sorted_index] = sorted_index;
	ordered_messages->_scan_input[sorted_index] = unordered_messages-&gt;_scan_input[index];

	<xsl:for-each select="xmml:variables/gpu:variable">ordered_messages-&gt;<xsl:value-of select="xmml:name"/>[sorted_index] = unordered_messages-&gt;<xsl:value-of select="xmml:name"/>[index];
	</xsl:for-each>
}

</xsl:if>

</xsl:for-each>
	
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created GPU kernels  */


<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:functions/gpu:function">
/**
 *
 */
__global__ void GPUFLAME_<xsl:value-of select="xmml:name"/>(xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* agents<xsl:if test="xmml:xagentOutputs/gpu:xagentOutput">, xmachine_memory_<xsl:value-of select="xmml:xagentOutputs/gpu:xagentOutput/xmml:xagentName"/>_list* <xsl:value-of select="xmml:xagentOutputs/gpu:xagentOutput/xmml:xagentName"/>_agents</xsl:if>
	<xsl:if test="xmml:inputs/gpu:input"><xsl:variable name="messagename" select="xmml:inputs/gpu:input/xmml:messageName"/>, xmachine_message_<xsl:value-of select="xmml:inputs/gpu:input/xmml:messageName"/>_list* <xsl:value-of select="xmml:inputs/gpu:input/xmml:messageName"/>_messages<xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messagename]"><xsl:if test="gpu:partitioningSpatial">, xmachine_message_<xsl:value-of select="xmml:name"/>_PBM* partition_matrix</xsl:if><xsl:if test="gpu:partitioningGraphEdge">, xmachine_message_<xsl:value-of select="xmml:name"/>_bounds* message_bounds</xsl:if></xsl:for-each></xsl:if>
	<xsl:if test="xmml:outputs/gpu:output">, xmachine_message_<xsl:value-of select="xmml:outputs/gpu:output/xmml:messageName"/>_list* <xsl:value-of select="xmml:outputs/gpu:output/xmml:messageName"/>_messages</xsl:if>
	<xsl:if test="gpu:RNG='true'">, RNG_rand48* rand48</xsl:if>){
	
	<xsl:if test="../../gpu:type='continuous'">//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    <!-- Can only use this for non brute force messaging as brute force messaging requires full thread block sizes -->
    <xsl:variable name="messageName" select="xmml:inputs/gpu:input/xmml:messageName"/>
    <xsl:choose><xsl:when test="../../../../xmml:messages/gpu:message[xmml:name=$messageName]/gpu:partitioningNone">
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    </xsl:when>
    <xsl:otherwise>//For agents not using non partitioned message input check the agent bounds
    if (index &gt;= d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count)
        return;
    </xsl:otherwise></xsl:choose>
	
    
    </xsl:if><xsl:if test="../../gpu:type='discrete'">
	//discrete agent: index is position in 2D agent grid
	int width = (blockDim.x * gridDim.x);
	glm::ivec2 global_position;
	global_position.x = (blockIdx.x * blockDim.x) + threadIdx.x;
	global_position.y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = global_position.x + (global_position.y * width);
	</xsl:if>

	//SoA to AoS - xmachine_memory_<xsl:value-of select="xmml:name"/> Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_<xsl:value-of select="../../xmml:name"/> agent;
    <xsl:variable name="messageName" select="xmml:inputs/gpu:input/xmml:messageName"/>
	<xsl:choose><xsl:when test="../../../../xmml:messages/gpu:message[xmml:name=$messageName]/gpu:partitioningNone">//No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index &lt; d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count){
    </xsl:when><xsl:otherwise>
    // Thread bounds already checked, but the agent function will still execute. load default values?
	</xsl:otherwise></xsl:choose>

	<xsl:for-each select="../../xmml:memory/gpu:variable"><xsl:choose><xsl:when test="xmml:arrayLength">
    agent.<xsl:value-of select="xmml:name"/> = &amp;(agents-&gt;<xsl:value-of select="xmml:name"/>[index]);</xsl:when><xsl:otherwise>
	agent.<xsl:value-of select="xmml:name"/> = agents-&gt;<xsl:value-of select="xmml:name"/>[index];</xsl:otherwise></xsl:choose></xsl:for-each>


	<xsl:choose><xsl:when test="../../../../xmml:messages/gpu:message[xmml:name=$messageName]/gpu:partitioningNone">
	} else {
	<xsl:for-each select="../../xmml:memory/gpu:variable"><xsl:choose><xsl:when test="xmml:arrayLength">
    agent.<xsl:value-of select="xmml:name"/> = nullptr;</xsl:when><xsl:otherwise>
	agent.<xsl:value-of select="xmml:name"/> = <xsl:choose><xsl:when test="xmml:defaultValue"><xsl:value-of select="xmml:defaultValue"/></xsl:when><xsl:otherwise><xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template></xsl:otherwise></xsl:choose>;</xsl:otherwise></xsl:choose></xsl:for-each>
	}</xsl:when><xsl:otherwise></xsl:otherwise></xsl:choose>

	//FLAME function call
	<xsl:if test="../../gpu:type='continuous'">int dead = !</xsl:if><xsl:value-of select="xmml:name"/>(&amp;agent<xsl:if test="xmml:xagentOutputs/gpu:xagentOutput">, <xsl:value-of select="xmml:xagentOutputs/gpu:xagentOutput/xmml:xagentName"/>_agents</xsl:if>
	<xsl:if test="xmml:inputs/gpu:input"><xsl:variable name="messagename" select="xmml:inputs/gpu:input/xmml:messageName"/>, <xsl:value-of select="xmml:inputs/gpu:input/xmml:messageName"/>_messages<xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messagename]"><xsl:if test="gpu:partitioningSpatial">, partition_matrix</xsl:if><xsl:if test="gpu:partitioningGraphEdge">, message_bounds</xsl:if></xsl:for-each></xsl:if>
	<xsl:if test="xmml:outputs/gpu:output">, <xsl:value-of select="xmml:outputs/gpu:output/xmml:messageName"/>_messages	</xsl:if>
	<xsl:if test="gpu:RNG='true'">, rand48</xsl:if>);
	

	<xsl:choose><xsl:when test="../../../../xmml:messages/gpu:message[xmml:name=$messageName]/gpu:partitioningNone">
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index &lt; d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count){
    </xsl:when><xsl:otherwise></xsl:otherwise></xsl:choose>
	<xsl:if test="../../gpu:type='continuous'">//continuous agent: set reallocation flag
	agents-&gt;_scan_input[index]  = dead; </xsl:if>

	//AoS to SoA - xmachine_memory_<xsl:value-of select="xmml:name"/> Coalesced memory write (ignore arrays)<xsl:for-each select="../../xmml:memory/gpu:variable"><xsl:if test="not(xmml:arrayLength)">
	agents-&gt;<xsl:value-of select="xmml:name"/>[index] = agent.<xsl:value-of select="xmml:name"/>;</xsl:if></xsl:for-each>
	<xsl:choose><xsl:when test="../../../../xmml:messages/gpu:message[xmml:name=$messageName]/gpu:partitioningNone">
	}</xsl:when><xsl:otherwise></xsl:otherwise></xsl:choose>
}
</xsl:for-each>
	

/* Agent ID Generation functions implemented in simulation.cu and FLAMEGPU_kernals.cu*/
<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
<xsl:variable name="agent_name" select="xmml:name" />
<xsl:for-each select="xmml:memory/gpu:variable">
<xsl:variable name="variable_name" select="xmml:name" />
<xsl:variable name="variable_type" select="xmml:type" />
<xsl:variable name="type_is_integer"><xsl:call-template name="typeIsInteger"><xsl:with-param name="type" select="$variable_type"/></xsl:call-template></xsl:variable>
<!-- If the agent has a variable name id, of a single integer type -->
<xsl:if test="$variable_name='id' and not(xmml:arrayLength) and $type_is_integer='true'" >
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ <xsl:value-of select="$variable_type"/> generate_<xsl:value-of select="$agent_name"/>_id(){
#if defined(__CUDA_ARCH__)
	// On the device, use atomicAdd to increment the ID, wrapping at overflow. Does not use atomicInc which only supports unsigned
	<xsl:value-of select="$variable_type"/> new_id = atomicAdd(&amp;d_current_value_generate_<xsl:value-of select="$agent_name"/>_id, 1);
	return new_id;	
#else
	// On the host, get the current value to be returned and increment the host value.
	<xsl:value-of select="$variable_type"/> new_id = h_current_value_generate_<xsl:value-of select="$agent_name"/>_id;
	h_current_value_generate_<xsl:value-of select="$agent_name"/>_id++; 
	return new_id;
#endif
}
</xsl:if>
</xsl:for-each>
</xsl:for-each>
	
/* Graph utility functions */
<xsl:for-each select="gpu:xmodel/gpu:environment/gpu:graphs/gpu:staticGraph">
<xsl:variable name="graph_name" select="gpu:name"/>
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ unsigned int get_staticGraph_<xsl:value-of select="$graph_name"/>_vertex_count(){
#if defined(__CUDA_ARCH__)
	return d_staticGraph_memory_<xsl:value-of select="$graph_name"/>_ptr-&gt;vertex.count;
#else
	return h_staticGraph_memory_<xsl:value-of select="$graph_name"/>-&gt;vertex.count;
#endif 
}
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ unsigned int get_staticGraph_<xsl:value-of select="$graph_name"/>_edge_count(){
#if defined(__CUDA_ARCH__)
	return d_staticGraph_memory_<xsl:value-of select="$graph_name"/>_ptr-&gt;edge.count;
	#else
	return h_staticGraph_memory_<xsl:value-of select="$graph_name"/>-&gt;edge.count;
#endif 
}
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ unsigned int get_staticGraph_<xsl:value-of select="$graph_name"/>_vertex_first_edge_index(unsigned int vertexIndex){
	if(vertexIndex &lt;= get_staticGraph_<xsl:value-of select="$graph_name"/>_vertex_count()){
	#if defined(__CUDA_ARCH__)
		return d_staticGraph_memory_<xsl:value-of select="$graph_name"/>_ptr-&gt;vertex.first_edge_index[vertexIndex];
	#else
		return h_staticGraph_memory_<xsl:value-of select="$graph_name"/>-&gt;vertex.first_edge_index[vertexIndex];
	#endif 
	} else {
		// Return the buffer size, i.e. no messages can start here.
		return staticGraph_<xsl:value-of select="$graph_name"/>_edge_bufferSize;
	}
}
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ unsigned int get_staticGraph_<xsl:value-of select="$graph_name"/>_vertex_num_edges(unsigned int vertexIndex){
if(vertexIndex &lt;= get_staticGraph_<xsl:value-of select="$graph_name"/>_vertex_count()){
	#if defined(__CUDA_ARCH__)
		return d_staticGraph_memory_<xsl:value-of select="$graph_name"/>_ptr-&gt;vertex.first_edge_index[vertexIndex + 1] - d_staticGraph_memory_<xsl:value-of select="$graph_name"/>_ptr-&gt;vertex.first_edge_index[vertexIndex];
	#else
		return h_staticGraph_memory_<xsl:value-of select="$graph_name"/>-&gt;vertex.first_edge_index[vertexIndex + 1] - h_staticGraph_memory_<xsl:value-of select="$graph_name"/>-&gt;vertex.first_edge_index[vertexIndex];
	#endif 
	} else {
		return 0;
	}
}


<xsl:for-each select="gpu:vertex/xmml:variables/gpu:variable">__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ <xsl:value-of select="xmml:type" /> get_staticGraph_<xsl:value-of select="$graph_name" />_vertex_<xsl:value-of select="xmml:name" />(unsigned int vertexIndex<xsl:if test="xmml:arrayLength">, unsigned int arrayElement</xsl:if>){
	if(vertexIndex &lt; get_staticGraph_<xsl:value-of select="$graph_name"/>_vertex_count()<xsl:if test="xmml:arrayLength"> &amp;&amp; arrayElement &lt; <xsl:value-of select="xmml:arrayLength" /></xsl:if>){
	#if defined(__CUDA_ARCH__)
		return d_staticGraph_memory_<xsl:value-of select="$graph_name"/>_ptr-&gt;vertex.<xsl:value-of select="xmml:name" />[<xsl:if test="xmml:arrayLength">(arrayElement * staticGraph_<xsl:value-of select="$graph_name"/>_vertex_bufferSize)</xsl:if> + vertexIndex];
	#else
		return h_staticGraph_memory_<xsl:value-of select="$graph_name"/>-&gt;vertex.<xsl:value-of select="xmml:name" />[<xsl:if test="xmml:arrayLength">(arrayElement * staticGraph_<xsl:value-of select="$graph_name"/>_vertex_bufferSize)</xsl:if> + vertexIndex];
	#endif 
	} else {
		return <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/><xsl:with-param name="defaultValue" select="xmml:defaultValue" /></xsl:call-template>;
	}
}
</xsl:for-each>
<xsl:for-each select="gpu:edge/xmml:variables/gpu:variable">__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ <xsl:value-of select="xmml:type" /> get_staticGraph_<xsl:value-of select="$graph_name" />_edge_<xsl:value-of select="xmml:name" />(unsigned int edgeIndex<xsl:if test="xmml:arrayLength">, unsigned int arrayElement</xsl:if>){
	if(edgeIndex &lt; get_staticGraph_<xsl:value-of select="$graph_name"/>_edge_count()<xsl:if test="xmml:arrayLength"> &amp;&amp; arrayElement &lt; <xsl:value-of select="xmml:arrayLength" /></xsl:if>){
	#if defined(__CUDA_ARCH__)
		return d_staticGraph_memory_<xsl:value-of select="$graph_name"/>_ptr-&gt;edge.<xsl:value-of select="xmml:name" />[<xsl:if test="xmml:arrayLength">(arrayElement * staticGraph_<xsl:value-of select="$graph_name"/>_edge_bufferSize)</xsl:if> + edgeIndex];
	#else
		return h_staticGraph_memory_<xsl:value-of select="$graph_name"/>-&gt;edge.<xsl:value-of select="xmml:name" />[<xsl:if test="xmml:arrayLength">(arrayElement * staticGraph_<xsl:value-of select="$graph_name"/>_edge_bufferSize)</xsl:if> + edgeIndex];
	#endif 
	} else {
		return <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/><xsl:with-param name="defaultValue" select="xmml:defaultValue" /></xsl:call-template>;
	}
}
</xsl:for-each>
</xsl:for-each>


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Rand48 functions */

__device__ static glm::uvec2 RNG_rand48_iterate_single(glm::uvec2 Xn, glm::uvec2 A, glm::uvec2 C)
{
	unsigned int R0, R1;

	// low 24-bit multiplication
	const unsigned int lo00 = __umul24(Xn.x, A.x);
	const unsigned int hi00 = __umulhi(Xn.x, A.x);

	// 24bit distribution of 32bit multiplication results
	R0 = (lo00 &amp; 0xFFFFFF);
	R1 = (lo00 &gt;&gt; 24) | (hi00 &lt;&lt; 8);

	R0 += C.x; R1 += C.y;

	// transfer overflows
	R1 += (R0 &gt;&gt; 24);
	R0 &amp;= 0xFFFFFF;

	// cross-terms, low/hi 24-bit multiplication
	R1 += __umul24(Xn.y, A.x);
	R1 += __umul24(Xn.x, A.y);

	R1 &amp;= 0xFFFFFF;

	return glm::uvec2(R0, R1);
}

//Templated function
template &lt;int AGENT_TYPE&gt;
__device__ float rnd(RNG_rand48* rand48){

	int index;
	
	//calculate the agents index in global agent list
	if (AGENT_TYPE == DISCRETE_2D){
		int width = (blockDim.x * gridDim.x);
		glm::ivec2 global_position;
		global_position.x = (blockIdx.x * blockDim.x) + threadIdx.x;
		global_position.y = (blockIdx.y * blockDim.y) + threadIdx.y;
		index = global_position.x + (global_position.y * width);
	}else//AGENT_TYPE == CONTINOUS
		index = threadIdx.x + blockIdx.x*blockDim.x;

	glm::uvec2 state = rand48->seeds[index];
	glm::uvec2 A = rand48->A;
	glm::uvec2 C = rand48->C;

	int rand = ( state.x &gt;&gt; 17 ) | ( state.y &lt;&lt; 7);

	// this actually iterates the RNG
	state = RNG_rand48_iterate_single(state, A, C);

	rand48->seeds[index] = state;

	return (float)rand/2147483647;
}

__device__ float rnd(RNG_rand48* rand48){
	return rnd&lt;DISCRETE_2D&gt;(rand48);
}

#endif //_FLAMEGPU_KERNELS_H_
</xsl:template>


</xsl:stylesheet>
