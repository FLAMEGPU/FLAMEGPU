<?xml version="1.0" encoding="utf-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:xmml="http://www.dcs.shef.ac.uk/~paul/XMML"
                xmlns:gpu="http://www.dcs.shef.ac.uk/~paul/XMMLGPU">
<xsl:output method="text" version="1.0" encoding="UTF-8" indent="yes" />
<xsl:include href = "./_common_templates.xslt" />
<!--Main template-->
<xsl:template match="/">
<xsl:call-template name="copyrightNotice"></xsl:call-template>

#ifndef _FLAMEGPU_FUNCTIONS
#define _FLAMEGPU_FUNCTIONS

#include &lt;header.h&gt;

<!-- Prototypes for Init functions -->
<xsl:for-each select="gpu:xmodel/gpu:environment/gpu:initFunctions/gpu:initFunction">
/**
 * <xsl:value-of select="gpu:name"/> FLAMEGPU Init function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_INIT_FUNC__ void <xsl:value-of select="gpu:name"/>(){

}
</xsl:for-each>
<!-- Prototypes for Step functions -->
<xsl:for-each select="gpu:xmodel/gpu:environment/gpu:stepFunctions/gpu:stepFunction">
/**
 * <xsl:value-of select="gpu:name"/> FLAMEGPU Step function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_STEP_FUNC__ void <xsl:value-of select="gpu:name"/>(){

}
</xsl:for-each>
<!-- Prototypes for Exit functions -->
<xsl:for-each select="gpu:xmodel/gpu:environment/gpu:exitFunctions/gpu:exitFunction">
/**
 * <xsl:value-of select="gpu:name"/> FLAMEGPU Exit function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_EXIT_FUNC__ void <xsl:value-of select="gpu:name"/>(){

}
</xsl:for-each>

<!-- Prototypes for agent functions -->
<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:functions/gpu:function">
/**
 * <xsl:value-of select="xmml:name"/> FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_<xsl:value-of select="../../xmml:name"/>. This represents a single agent instance and can be modified directly.
 <xsl:if test="xmml:xagentOutputs/gpu:xagentOutput">* @param <xsl:value-of select="xmml:xagentOutputs/gpu:xagentOutput/xmml:xagentName"/>_agents Pointer to agent list of type xmachine_memory_<xsl:value-of select="xmml:xagentOutputs/gpu:xagentOutput/xmml:xagentName"/>_list. This must be passed as an argument to the add_<xsl:value-of select="xmml:xagentOutputs/gpu:xagentOutput/xmml:xagentName"/>_agent function to add a new agent.</xsl:if>
 <xsl:if test="xmml:inputs/gpu:input"><xsl:variable name="messagename" select="xmml:inputs/gpu:input/xmml:messageName"/>* @param <xsl:value-of select="$messagename"/>_messages  <xsl:value-of select="xmml:inputs/gpu:input/xmml:messageName"/>_messages Pointer to input message list of type xmachine_message_<xsl:value-of select="xmml:inputs/gpu:inputs/xmml:messageName"/>_list. Must be passed as an argument to the get_first_<xsl:value-of select="xmml:inputs/gpu:input/xmml:messageName"/>_message and get_next_<xsl:value-of select="xmml:inputs/gpu:input/xmml:messageName"/>_message functions.<xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messagename]">
 <xsl:if test="gpu:partitioningSpatial">* @param partition_matrix Pointer to the partition matrix of type xmachine_message_<xsl:value-of select="xmml:name"/>_PBM. Used within the get_first_<xsl:value-of select="xmml:inputs/gpu:input/xmml:messageName"/>_message and get_next_<xsl:value-of select="xmml:inputs/gpu:input/xmml:messageName"/>_message functions for spatially partitioned message access.</xsl:if></xsl:for-each></xsl:if>
 <xsl:if test="xmml:outputs/gpu:output">* @param <xsl:value-of select="xmml:outputs/gpu:output/xmml:messageName"/>_messages Pointer to output message list of type xmachine_message_<xsl:value-of select="xmml:outputs/gpu:output/xmml:messageName"/>_list. Must be passed as an argument to the add_<xsl:value-of select="xmml:outputs/gpu:output/xmml:messageName"/>_message function.</xsl:if>
 <xsl:if test="gpu:RNG='true'">* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.</xsl:if>
 */
__FLAME_GPU_FUNC__ int <xsl:value-of select="xmml:name"/>(xmachine_memory_<xsl:value-of select="../../xmml:name"/>* agent<xsl:if test="xmml:xagentOutputs/gpu:xagentOutput">, xmachine_memory_<xsl:value-of select="xmml:xagentOutputs/gpu:xagentOutput/xmml:xagentName"/>_list* <xsl:value-of select="xmml:xagentOutputs/gpu:xagentOutput/xmml:xagentName"/>_agents</xsl:if>
<xsl:if test="xmml:inputs/gpu:input"><xsl:variable name="messagename" select="xmml:inputs/gpu:input/xmml:messageName"/>, xmachine_message_<xsl:value-of select="xmml:inputs/gpu:input/xmml:messageName"/>_list* <xsl:value-of select="xmml:inputs/gpu:input/xmml:messageName"/>_messages<xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messagename]"><xsl:if test="gpu:partitioningSpatial">, xmachine_message_<xsl:value-of select="xmml:name"/>_PBM* partition_matrix</xsl:if><xsl:if test="gpu:partitioningGraphEdge">, xmachine_message_<xsl:value-of select="xmml:name"/>_bounds* message_bounds</xsl:if></xsl:for-each></xsl:if>
<xsl:if test="xmml:outputs/gpu:output">, xmachine_message_<xsl:value-of select="xmml:outputs/gpu:output/xmml:messageName"/>_list* <xsl:value-of select="xmml:outputs/gpu:output/xmml:messageName"/>_messages</xsl:if>
<xsl:if test="gpu:RNG='true'">, RNG_rand48* rand48</xsl:if>){
    <xsl:variable name="agent_type" select="../../gpu:type" />
    <xsl:if test="xmml:inputs/gpu:input">
    <xsl:variable name="messagename" select="xmml:inputs/gpu:input/xmml:messageName"/>
        <xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messagename]">
    /*<xsl:if test="gpu:partitioningDiscrete">
    // Index of discrete cell
    int agent_x = 0;
    int agent_y = 0;
    </xsl:if><xsl:if test="gpu:partitioningSpatial">
    // Position within space
    float agent_x = 0.0;
    float agent_y = 0.0;
    float agent_z = 0.0;
    </xsl:if>
    <xsl:if test="gpu:partitioningGraphEdge">
    unsigned int edgeIndex = 0;
    </xsl:if>
    //Template for input message iteration
    xmachine_message_<xsl:value-of select="$messagename"/>* current_message = get_first_<xsl:value-of select="$messagename"/>_message<xsl:if test="gpu:partitioningDiscrete">&lt;<xsl:if test="$agent_type='continuous'">CONTINUOUS</xsl:if><xsl:if test="$agent_type='discrete'">DISCRETE_2D</xsl:if>&gt;</xsl:if>(<xsl:value-of select="$messagename"/>_messages<xsl:if test="gpu:partitioningSpatial">, partition_matrix, agent_x, agent_y, agent_z</xsl:if><xsl:if test="gpu:partitioningDiscrete">, agent_x, agent_y</xsl:if><xsl:if test="gpu:partitioningGraphEdge">, message_bounds, edgeIndex</xsl:if>);
    while (current_message)
    {
        //INSERT MESSAGE PROCESSING CODE HERE
        
        current_message = get_next_<xsl:value-of select="$messagename"/>_message<xsl:if test="gpu:partitioningDiscrete">&lt;<xsl:if test="$agent_type='continuous'">CONTINUOUS</xsl:if><xsl:if test="$agent_type='discrete'">DISCRETE_2D</xsl:if>&gt;</xsl:if>(current_message, <xsl:value-of select="$messagename"/>_messages<xsl:if test="gpu:partitioningSpatial">, partition_matrix</xsl:if><xsl:if test="gpu:partitioningGraphEdge">, message_bounds</xsl:if>);
    }
    */
    </xsl:for-each></xsl:if><xsl:if test="xmml:outputs/gpu:output">
    /* 
    //Template for message output function
    <xsl:variable name="messagename" select="xmml:outputs/gpu:output/xmml:messageName"/>
    <xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messagename]/xmml:variables/gpu:variable">
    <xsl:value-of select="xmml:type"/><xsl:text> </xsl:text><xsl:value-of select="xmml:name"/> = <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/><xsl:with-param name="defaultValue" select="xmml:defaultValue"/></xsl:call-template>;
    </xsl:for-each>
    <xsl:variable name="isDiscretePartitioned" select="../../../../xmml:messages/gpu:message[xmml:name=$messagename]/gpu:partitioningDiscrete" />
    add_<xsl:value-of select="$messagename"/>_message<xsl:if test="$isDiscretePartitioned">&lt;<xsl:if test="$agent_type='discrete'">DISCRETE_2D</xsl:if>&gt;</xsl:if>(<xsl:value-of select="xmml:outputs/gpu:output/xmml:messageName"/>_messages, <xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messagename]/xmml:variables/gpu:variable"><xsl:value-of select="xmml:name"/><xsl:if test="position()!=last()">, </xsl:if></xsl:for-each>);
    */     
    </xsl:if><xsl:if test="xmml:xagentOutputs/gpu:xagentOutput">
    /* 
    //Template for agent output functions 
    <xsl:variable name="xagentname" select="xmml:xagentOutputs/gpu:xagentOutput/xmml:xagentName"/>
    <xsl:for-each select="../../../../xmml:xagents/gpu:xagent[xmml:name=$xagentname]/xmml:memory/gpu:variable[not(xmml:arrayLength)]">
    <xsl:value-of select="xmml:type"/><xsl:text> new_</xsl:text><xsl:value-of select="xmml:name"/> = <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/><xsl:with-param name="defaultValue" select="xmml:defaultValue"/></xsl:call-template>;
    </xsl:for-each>
    add_<xsl:value-of select="$xagentname"/>_agent(<xsl:value-of select="$xagentname"/>_agents, <xsl:for-each select="../../../../xmml:xagents/gpu:xagent[xmml:name=$xagentname]/xmml:memory/gpu:variable[not(xmml:arrayLength)]">new_<xsl:value-of select="xmml:name"/><xsl:if test="position()!=last()">, </xsl:if></xsl:for-each>);
    */
    </xsl:if>
    return 0;
}
</xsl:for-each>
  


#endif //_FLAMEGPU_FUNCTIONS
</xsl:template>
</xsl:stylesheet>
