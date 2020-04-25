<?xml version="1.0" encoding="utf-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:xmml="http://www.dcs.shef.ac.uk/~paul/XMML"
                xmlns:gpu="http://www.dcs.shef.ac.uk/~paul/XMMLGPU">
<xsl:output method="text" version="1.0" encoding="UTF-8" indent="yes" />
<!--Main template-->
<xsl:template match="/">
/*
 * Copyright 2011 University of Sheffield.
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

#ifndef _FLAMEGPU_FUNCTIONS
#define _FLAMEGPU_FUNCTIONS

#include &lt;header.h&gt;

<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:functions/gpu:function">
/**
 * <xsl:value-of select="xmml:name"/> FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
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
<xsl:if test="gpu:RNG='true'">, RNG_rand48* rand48</xsl:if>){

    <xsl:if test="xmml:inputs/gpu:input">
    /* //Template for input message iteration
     * <xsl:variable name="messagename" select="xmml:inputs/gpu:input/xmml:messageName"/>
     * xmachine_message_<xsl:value-of select="xmml:inputs/gpu:input/xmml:messageName"/>* current_message = get_first_<xsl:value-of select="$messagename"/>_message(<xsl:value-of select="$messagename"/>_messages<xsl:if test="gpu:partitioningSpatial">, partition_matrix</xsl:if>);
     * while (current_message)
     * {
     *     //INSERT MESSAGE PROCESSING CODE HERE
     *     
     *     current_message = get_next_<xsl:value-of select="$messagename"/>_message(current_message, <xsl:value-of select="$messagename"/>_messages<xsl:if test="gpu:partitioningSpatial">, partition_matrix</xsl:if>);
     * }
     */
     
    </xsl:if><xsl:if test="xmml:outputs/gpu:output">
    /* //Template for message output function use <xsl:variable name="messagename" select="xmml:outputs/gpu:output/xmml:messageName"/>
     * <xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messagename]/xmml:variables/gpu:variable">
     * <xsl:value-of select="xmml:type"/><xsl:text> </xsl:text><xsl:value-of select="xmml:name"/> = 0;</xsl:for-each>
     * add_<xsl:value-of select="xmml:outputs/gpu:output/xmml:messageName"/>_message(<xsl:value-of select="xmml:outputs/gpu:output/xmml:messageName"/>_messages, <xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messagename]/xmml:variables/gpu:variable"><xsl:value-of select="xmml:name"/><xsl:if test="position()!=last()">, </xsl:if></xsl:for-each>);
     */
     
     </xsl:if><xsl:if test="xmml:xagentOutputs/gpu:xagentOutput">
     /* //Template for agent output functions <xsl:variable name="xagentname" select="xmml:xagentOutputs/gpu:xagentOutput/xmml:xagentName"/>
      * <xsl:for-each select="../../../../xmml:xagents/gpu:xagent[xmml:name=$xagentname]/xmml:memory/gpu:variable">
      * <xsl:value-of select="xmml:type"/><xsl:text> </xsl:text><xsl:value-of select="xmml:name"/> = 0;</xsl:for-each>
      * add_<xsl:value-of select="$xagentname"/>_agent(<xsl:value-of select="$xagentname"/>_agents, <xsl:for-each select="../../../../xmml:xagents/gpu:xagent[xmml:name=$xagentname]/xmml:memory/gpu:variable"><xsl:value-of select="xmml:type"/><xsl:text> </xsl:text><xsl:value-of select="xmml:name"/><xsl:if test="position()!=last()">, </xsl:if></xsl:for-each>);
      */
     </xsl:if>
  
    return 0;
}
</xsl:for-each>
  


#endif //_FLAMEGPU_FUNCTIONS
</xsl:template>
</xsl:stylesheet>
