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

#define _CRT_SECURE_NO_WARNINGS
#include &lt;cuda_runtime.h&gt;
#include &lt;stdlib.h&gt;
#include &lt;stdio.h&gt;
#include &lt;string.h&gt;
#include &lt;cmath&gt;
#include &lt;limits.h&gt;
#include &lt;rapidxml/rapidxml.hpp&gt;
#include &lt;fstream&gt;
#include &lt;sstream&gt;
	

// include header
#include "header.h"

glm::vec3 agent_maximum;
glm::vec3 agent_minimum;

void readIntArrayInput(char* buffer, int *array, unsigned int expected_items){
    unsigned int i = 0;
    const char s[2] = ",";
    char * token;

    token = strtok(buffer, s);
    while (token != NULL){
        if (i>=expected_items){
            printf("Error: Agent memory array has too many items, expected %d!\n", expected_items);
            exit(0);
        }
        
        array[i++] = atoi(token);
        
        token = strtok(NULL, s);
    }
    if (i != expected_items){
        printf("Error: Agent memory array has %d items, expected %d!\n", i, expected_items);
        exit(0);
    }
}

void readFloatArrayInput(char* buffer, float *array, unsigned int expected_items){
    unsigned int i = 0;
    const char s[2] = ",";
    char * token;

    token = strtok(buffer, s);
    while (token != NULL){
        if (i>=expected_items){
            printf("Error: Agent memory array has too many items, expected %d!\n", expected_items);
            exit(0);
        }
        
        array[i++] = (float)atof(token);
        
        token = strtok(NULL, s);
    }
    if (i != expected_items){
        printf("Error: Agent memory array has %d items, expected %d!\n", i, expected_items);
        exit(0);
    }
}

void saveIterationData(char* outputpath, int iteration_number, <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state">xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, int h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count<xsl:if test="position()!=last()">,</xsl:if></xsl:for-each>)
{
	cudaError_t cudaStatus;
	
	//Device to host memory transfer
	<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state">
	cudaStatus = cudaMemcpy( h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, sizeof(xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr,"Error Copying <xsl:value-of select="../../xmml:name"/> Agent <xsl:value-of select="xmml:name"/> State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}</xsl:for-each>
	
	/* Pointer to file */
	FILE *file;
	char data[100];

	sprintf(data, "%s%i.xml", outputpath, iteration_number);
	//printf("Writing iteration %i data to %s\n", iteration_number, data);
	file = fopen(data, "w");
	fputs("&lt;states&gt;\n&lt;itno&gt;", file);
	sprintf(data, "%i", iteration_number);
	fputs(data, file);
	fputs("&lt;/itno&gt;\n", file);
	fputs("&lt;environment&gt;\n" , file);
	fputs("&lt;/environment&gt;\n" , file);

	<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state"><xsl:variable name="stateName" select="xmml:name"/>//Write each <xsl:value-of select="../../xmml:name"/> agent to xml
	for (int i=0; i&lt;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count; i++){
		fputs("&lt;xagent&gt;\n" , file);
		fputs("&lt;name&gt;<xsl:value-of select="../../xmml:name"/>&lt;/name&gt;\n", file);
        <xsl:for-each select="../../xmml:memory/gpu:variable">
		fputs("&lt;<xsl:value-of select="xmml:name"/>&gt;", file);
        <xsl:choose><xsl:when test="xmml:arrayLength">for (int j=0;j&lt;<xsl:value-of select="xmml:arrayLength"/>;j++){
            fprintf(file, "%<xsl:choose><xsl:when test="xmml:type='int'">i</xsl:when><xsl:otherwise>f</xsl:otherwise></xsl:choose>", h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="$stateName"/>-><xsl:value-of select="xmml:name"/>[(j*xmachine_memory_<xsl:value-of select="../../xmml:name"/>_MAX)+i]);
            if(j!=(<xsl:value-of select="xmml:arrayLength"/>-1))
                fprintf(file, ",");
        }</xsl:when><xsl:otherwise>sprintf(data, "%<xsl:choose><xsl:when test="xmml:type='int'">i</xsl:when><xsl:otherwise>f</xsl:otherwise></xsl:choose>", h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="$stateName"/>-><xsl:value-of select="xmml:name"/>[i]);
		fputs(data, file);</xsl:otherwise></xsl:choose>
		fputs("&lt;/<xsl:value-of select="xmml:name"/>&gt;\n", file);
        </xsl:for-each>
		fputs("&lt;/xagent&gt;\n", file);
	}
	</xsl:for-each>
	

	fputs("&lt;/states&gt;\n" , file);
	
	/* Close the file */
	fclose(file);
}

//Add &lt;nowarn/&gt; to the root of init files to disable warnings about missing agent properties
void readInitialStates(char* inputpath, <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">xmachine_memory_<xsl:value-of select="xmml:name"/>_list* h_<xsl:value-of select="xmml:name"/>s, int* h_xmachine_memory_<xsl:value-of select="xmml:name"/>_count<xsl:if test="position()!=last()">,</xsl:if></xsl:for-each>)
{

	int temp = 0;
	int* itno = &amp;temp;
	
	/* Initialise variables */
    agent_maximum.x = 0;
    agent_maximum.y = 0;
    agent_maximum.z = 0;
    agent_minimum.x = 0;
    agent_minimum.y = 0;
    agent_minimum.z = 0;

	<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
	//set all <xsl:value-of select="xmml:name"/> values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k&lt;xmachine_memory_<xsl:value-of select="xmml:name"/>_MAX; k++)
	{	<xsl:for-each select="xmml:memory/gpu:variable"><xsl:choose><xsl:when test="xmml:arrayLength">
        for (int i=0;i&lt;<xsl:value-of select="xmml:arrayLength"/>;i++){
            h_<xsl:value-of select="../../xmml:name"/>s-><xsl:value-of select="xmml:name"/>[(i*xmachine_memory_<xsl:value-of select="../../xmml:name"/>_MAX)+k] = 0;
        }</xsl:when><xsl:otherwise>
		h_<xsl:value-of select="../../xmml:name"/>s-><xsl:value-of select="xmml:name"/>[k] = 0;</xsl:otherwise></xsl:choose></xsl:for-each>
	}
	</xsl:for-each>

  /* Open file for parsing */
	rapidxml::xml_document&lt;&gt; doc;    // character type defaults to char
	std::ifstream file(inputpath);
	if (!file.is_open())
	{
		printf("ERROR: Could not open initialisation file %s\n", inputpath);
		exit(0);
	}
	std::stringstream buffer;
	buffer &lt;&lt; file.rdbuf();
	std::string content(buffer.str());
	doc.parse&lt;0&gt;(&amp;content[0]);
  
  //Allocate buffers for array properties
  <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:memory/gpu:variable"><xsl:choose><xsl:when test="xmml:arrayLength"><xsl:text>
    </xsl:text><xsl:value-of select="xmml:type"/><xsl:text> </xsl:text><xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>[<xsl:value-of select="xmml:arrayLength"/>];</xsl:when><xsl:otherwise></xsl:otherwise></xsl:choose></xsl:for-each>

  
  //Allocate nodes
  rapidxml::xml_node&lt;&gt; *root_node = doc.first_node("states");
  rapidxml::xml_node&lt;&gt; *xagent_node = root_node->first_node("xagent");
  rapidxml::xml_node&lt;&gt; *name_node=0, *t_node=0;
  bool noWarn = doc.first_node("nowarn")!=0;
  
  //Iterate all xagent nodes
  while (xagent_node)
  {
    //Extract name of xagent
    name_node = xagent_node->first_node("name");
    <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
    <xsl:if test="position()!=1">else </xsl:if>if (strcmp(name_node-&gt;value(), "<xsl:value-of select="xmml:name"/>") == 0)
    {
      //Too many agents
      if (*h_xmachine_memory_<xsl:value-of select="xmml:name"/>_count &gt; xmachine_memory_<xsl:value-of select="xmml:name"/>_MAX){
        printf("ERROR: MAX Buffer size (%i) for agent <xsl:value-of select="xmml:name"/> exceeded whilst reading data\n", xmachine_memory_<xsl:value-of select="xmml:name"/>_MAX);
        exit(0);
      }
      //Extract the agent properties<xsl:for-each select="xmml:memory/gpu:variable">
      t_node = xagent_node->first_node("<xsl:value-of select="xmml:name"/>");
      if(t_node){<xsl:choose><xsl:when test="xmml:arrayLength">
      read<xsl:choose><xsl:when test="xmml:type='int'">Int</xsl:when><xsl:otherwise>Float</xsl:otherwise></xsl:choose>ArrayInput(t_node->value(), <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>, <xsl:value-of select="xmml:arrayLength"/>);
        for (int k=0;k&lt;<xsl:value-of select="xmml:arrayLength"/>;k++){
          h_<xsl:value-of select="../../xmml:name"/>s-&gt;<xsl:value-of select="xmml:name"/>[(k*xmachine_memory_<xsl:value-of select="../../xmml:name"/>_MAX)+(*h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count)] = <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>[k];    
        }</xsl:when><xsl:otherwise>
        h_<xsl:value-of select="../../xmml:name"/>s-&gt;<xsl:value-of select="xmml:name"/>[*h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count] = (<xsl:value-of select="xmml:type"/>) ato<xsl:choose><xsl:when test="xmml:type='int'">i</xsl:when><xsl:otherwise>f</xsl:otherwise></xsl:choose>(t_node->value());</xsl:otherwise></xsl:choose>
      }else if(!noWarn){printf("WARNING: Agent <xsl:value-of select="../../xmml:name"/>[%d] is missing property '<xsl:value-of select="xmml:name"/>' in init file, 0 used.\n",(*h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count));}
      </xsl:for-each>

      //Calculate agent min/max
      <xsl:for-each select="xmml:memory/gpu:variable"><xsl:if test="xmml:name='x'">//Check maximum x value
      if(agent_maximum.x &lt; h_<xsl:value-of select="../../xmml:name"/>s-&gt;<xsl:value-of select="xmml:name"/>[*h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count])
          agent_maximum.x = (float)h_<xsl:value-of select="../../xmml:name"/>s-&gt;<xsl:value-of select="xmml:name"/>[*h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count];
      </xsl:if>
      <xsl:if test="xmml:name='y'">//Check maximum y value
      if(agent_maximum.y &lt; h_<xsl:value-of select="../../xmml:name"/>s-&gt;<xsl:value-of select="xmml:name"/>[*h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count])
          agent_maximum.y = (float)h_<xsl:value-of select="../../xmml:name"/>s-&gt;<xsl:value-of select="xmml:name"/>[*h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count];
      </xsl:if>
      <xsl:if test="xmml:name='z'">//Check maximum z value
      if(agent_maximum.z &lt; h_<xsl:value-of select="../../xmml:name"/>s-&gt;<xsl:value-of select="xmml:name"/>[*h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count])
          agent_maximum.z = (float)h_<xsl:value-of select="../../xmml:name"/>s-&gt;<xsl:value-of select="xmml:name"/>[*h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count];
      </xsl:if>
      <xsl:if test="xmml:name='x'">//Check minimum x value
      if(agent_minimum.x &gt; h_<xsl:value-of select="../../xmml:name"/>s-&gt;<xsl:value-of select="xmml:name"/>[*h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count])
          agent_minimum.x = (float)h_<xsl:value-of select="../../xmml:name"/>s-&gt;<xsl:value-of select="xmml:name"/>[*h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count];
      </xsl:if>
      <xsl:if test="xmml:name='y'">//Check minimum y value
      if(agent_minimum.y &gt; h_<xsl:value-of select="../../xmml:name"/>s-&gt;<xsl:value-of select="xmml:name"/>[*h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count])
          agent_minimum.y = (float)h_<xsl:value-of select="../../xmml:name"/>s-&gt;<xsl:value-of select="xmml:name"/>[*h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count];
      </xsl:if>
      <xsl:if test="xmml:name='z'">//Check minimum z value
      if(agent_minimum.z &gt; h_<xsl:value-of select="../../xmml:name"/>s-&gt;<xsl:value-of select="xmml:name"/>[*h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count])
          agent_minimum.z = (float)h_<xsl:value-of select="../../xmml:name"/>s-&gt;<xsl:value-of select="xmml:name"/>[*h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count];
      </xsl:if></xsl:for-each>
      //Increment <xsl:value-of select="xmml:name"/> agent count
      (*h_xmachine_memory_<xsl:value-of select="xmml:name"/>_count)++;
    }
    </xsl:for-each>
    //Find next xagent node
    xagent_node = xagent_node->next_sibling("xagent");
	}
}

glm::vec3 getMaximumBounds(){
    return agent_maximum;
}

glm::vec3 getMinimumBounds(){
    return agent_minimum;
}

</xsl:template>
</xsl:stylesheet>
