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

#include &lt;cuda_runtime.h&gt;
#include &lt;stdlib.h&gt;
#include &lt;stdio.h&gt;
#include &lt;string.h&gt;
#include &lt;cmath&gt;
#include &lt;limits.h&gt;


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
            exit(EXIT_FAILURE);
        }

        array[i++] = atoi(token);

        token = strtok(NULL, s);
    }
    if (i != expected_items){
        printf("Error: Agent memory array has %d items, expected %d!\n", i, expected_items);
        exit(EXIT_FAILURE);
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
            exit(EXIT_FAILURE);
        }

        array[i++] = (float)atof(token);

        token = strtok(NULL, s);
    }
    if (i != expected_items){
        printf("Error: Agent memory array has %d items, expected %d!\n", i, expected_items);
        exit(EXIT_FAILURE);
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

void readInitialStates(char* inputpath, <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">xmachine_memory_<xsl:value-of select="xmml:name"/>_list* h_<xsl:value-of select="xmml:name"/>s, int* h_xmachine_memory_<xsl:value-of select="xmml:name"/>_count<xsl:if test="position()!=last()">,</xsl:if></xsl:for-each>)
{

	int temp = 0;
	int* itno = &amp;temp;

	/* Pointer to file */
	FILE *file;
	/* Char and char buffer for reading file to */
	char c = ' ';
	char buffer[10000];
	char agentname[1000];

	/* Pointer to x-memory for initial state data */
	/*xmachine * current_xmachine;*/
	/* Variables for checking tags */
	int reading, i;
	int in_tag, in_itno, in_name;<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:memory/gpu:variable">
    int in_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>;</xsl:for-each>

	/* for continuous agents: set agent count to zero */<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent"><xsl:if test="gpu:type='continuous'">
	*h_xmachine_memory_<xsl:value-of select="xmml:name"/>_count = 0;</xsl:if></xsl:for-each>

	/* Variables for initial state data */<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:memory/gpu:variable"><xsl:choose><xsl:when test="xmml:arrayLength"><xsl:text>
    </xsl:text><xsl:value-of select="xmml:type"/><xsl:text> </xsl:text><xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>[<xsl:value-of select="xmml:arrayLength"/>];</xsl:when><xsl:otherwise><xsl:text>
	</xsl:text><xsl:value-of select="xmml:type"/><xsl:text> </xsl:text><xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>;</xsl:otherwise></xsl:choose></xsl:for-each>


	/* Initialise variables */
  agent_maximum.x = 0;
  agent_maximum.y = 0;
  agent_maximum.z = 0;
  agent_minimum.x = 0;
  agent_minimum.y = 0;
  agent_minimum.z = 0;
	reading = 1;
	in_tag = 0;
	in_itno = 0;
	in_name = 0;<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:memory/gpu:variable">
	in_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/> = 0;</xsl:for-each>

	<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
	//set all <xsl:value-of select="xmml:name"/> values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k&lt;xmachine_memory_<xsl:value-of select="xmml:name"/>_MAX; k++)
	{	<xsl:for-each select="xmml:memory/gpu:variable"><xsl:choose><xsl:when test="xmml:arrayLength">
        for (i=0;i&lt;<xsl:value-of select="xmml:arrayLength"/>;i++){
            h_<xsl:value-of select="../../xmml:name"/>s-><xsl:value-of select="xmml:name"/>[(i*xmachine_memory_<xsl:value-of select="../../xmml:name"/>_MAX)+k] = 0;
        }</xsl:when><xsl:otherwise>
		h_<xsl:value-of select="../../xmml:name"/>s-><xsl:value-of select="xmml:name"/>[k] = 0;</xsl:otherwise></xsl:choose></xsl:for-each>
	}
	</xsl:for-each>

	/* Default variables for memory */<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:memory/gpu:variable"><xsl:choose><xsl:when test="xmml:arrayLength">
    for (i=0;i&lt;<xsl:value-of select="xmml:arrayLength"/>;i++){
        <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>[i] = <xsl:choose><xsl:when test="xmml:defaultValue"><xsl:value-of select="xmml:defaultValue"/></xsl:when><xsl:otherwise>0</xsl:otherwise></xsl:choose>;
    }</xsl:when><xsl:otherwise><xsl:text>
    </xsl:text><xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/> = <xsl:choose><xsl:when test="xmml:defaultValue"><xsl:value-of select="xmml:defaultValue"/></xsl:when><xsl:otherwise>0</xsl:otherwise></xsl:choose>;</xsl:otherwise></xsl:choose></xsl:for-each>

  /* Open config file to read-only */
  /* If inputfile is empty or not found then we initialise only */
	if(inputpath == NULL || (file = fopen(inputpath, "r"))==NULL)
	{
    printf("Initial states file not specifed or does not exist, parameters are initialised to default values\n");
    return;
	}

	/* Read file until end of xml */
    i = 0;
	while(reading==1)
	{
		/* Get the next char from the file */
		c = (char)fgetc(file);

		/* If the end of a tag */
		if(c == '&gt;')
		{
			/* Place 0 at end of buffer to make chars a string */
			buffer[i] = 0;

			if(strcmp(buffer, "states") == 0) reading = 1;
			if(strcmp(buffer, "/states") == 0) reading = 0;
			if(strcmp(buffer, "itno") == 0) in_itno = 1;
			if(strcmp(buffer, "/itno") == 0) in_itno = 0;
			if(strcmp(buffer, "name") == 0) in_name = 1;
			if(strcmp(buffer, "/name") == 0) in_name = 0;
			if(strcmp(buffer, "/xagent") == 0)
			{
				<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
				<xsl:if test="position()!=1">else </xsl:if>if(strcmp(agentname, "<xsl:value-of select="xmml:name"/>") == 0)
				{
					if (*h_xmachine_memory_<xsl:value-of select="xmml:name"/>_count > xmachine_memory_<xsl:value-of select="xmml:name"/>_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent <xsl:value-of select="xmml:name"/> exceeded whilst reading data\n", xmachine_memory_<xsl:value-of select="xmml:name"/>_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    <xsl:for-each select="xmml:memory/gpu:variable"><xsl:choose><xsl:when test="xmml:arrayLength">
                    for (int k=0;k&lt;<xsl:value-of select="xmml:arrayLength"/>;k++){
                        h_<xsl:value-of select="../../xmml:name"/>s-><xsl:value-of select="xmml:name"/>[(k*xmachine_memory_<xsl:value-of select="../../xmml:name"/>_MAX)+(*h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count)] = <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>[k];
                    }</xsl:when><xsl:otherwise>
					h_<xsl:value-of select="../../xmml:name"/>s-><xsl:value-of select="xmml:name"/>[*h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count] = <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>;</xsl:otherwise></xsl:choose>
                    <xsl:if test="xmml:name='x'">//Check maximum x value
                    if(agent_maximum.x &lt; <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>)
                        agent_maximum.x = (float)<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>;
                    </xsl:if>
                    <xsl:if test="xmml:name='y'">//Check maximum y value
                    if(agent_maximum.y &lt; <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>)
                        agent_maximum.y = (float)<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>;
                    </xsl:if>
                    <xsl:if test="xmml:name='z'">//Check maximum z value
                    if(agent_maximum.z &lt; <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>)
                        agent_maximum.z = (float)<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>;
                    </xsl:if>
                    <xsl:if test="xmml:name='x'">//Check minimum x value
                    if(agent_minimum.x &gt; <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>)
                        agent_minimum.x = (float)<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>;
                    </xsl:if>
                    <xsl:if test="xmml:name='y'">//Check minimum y value
                    if(agent_minimum.y &gt; <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>)
                        agent_minimum.y = (float)<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>;
                    </xsl:if>
                    <xsl:if test="xmml:name='z'">//Check minimum z value
                    if(agent_minimum.z &gt; <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>)
                        agent_minimum.z = (float)<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>;
                    </xsl:if></xsl:for-each>
					(*h_xmachine_memory_<xsl:value-of select="xmml:name"/>_count) ++;
				}
				</xsl:for-each>else
				{
					printf("Warning: agent name undefined - '%s'\n", agentname);
				}



				/* Reset xagent variables */<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:memory/gpu:variable"><xsl:choose><xsl:when test="xmml:arrayLength">
                for (i=0;i&lt;<xsl:value-of select="xmml:arrayLength"/>;i++){
                    <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>[i] = <xsl:choose><xsl:when test="xmml:defaultValue"><xsl:value-of select="xmml:defaultValue"/></xsl:when><xsl:otherwise>0</xsl:otherwise></xsl:choose>;
                }</xsl:when><xsl:otherwise><xsl:text>
                </xsl:text><xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/> = <xsl:choose><xsl:when test="xmml:defaultValue"><xsl:value-of select="xmml:defaultValue"/></xsl:when><xsl:otherwise>0</xsl:otherwise></xsl:choose>;</xsl:otherwise></xsl:choose></xsl:for-each>

			}
			<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:memory/gpu:variable">if(strcmp(buffer, "<xsl:value-of select="xmml:name"/>") == 0) in_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/> = 1;
			if(strcmp(buffer, "/<xsl:value-of select="xmml:name"/>") == 0) in_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/> = 0;
			</xsl:for-each>

			/* End of tag and reset buffer */
			in_tag = 0;
			i = 0;
		}
		/* If start of tag */
		else if(c == '&lt;')
		{
			/* Place /0 at end of buffer to end numbers */
			buffer[i] = 0;
			/* Flag in tag */
			in_tag = 1;

			if(in_itno) *itno = atoi(buffer);
			if(in_name) strcpy(agentname, buffer);
			else
			{
				<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:memory/gpu:variable">if(in_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>){
                    <xsl:choose><xsl:when test="xmml:arrayLength">read<xsl:choose><xsl:when test="xmml:type='int'">Int</xsl:when><xsl:otherwise>Float</xsl:otherwise></xsl:choose>ArrayInput(buffer, <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>, <xsl:value-of select="xmml:arrayLength"/>);    </xsl:when>
                    <xsl:otherwise><xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/> = (<xsl:value-of select="xmml:type"/>) ato<xsl:choose><xsl:when test="xmml:type='int'">i</xsl:when><xsl:otherwise>f</xsl:otherwise></xsl:choose>(buffer);    </xsl:otherwise></xsl:choose>
                }
				</xsl:for-each>
			}

			/* Reset buffer */
			i = 0;
		}
		/* If in tag put read char into buffer */
		else if(in_tag)
		{
			buffer[i] = c;
			i++;
		}
		/* If in data read char into buffer */
		else
		{
			buffer[i] = c;
			i++;
		}
	}
	/* Close the file */
	fclose(file);
}

glm::vec3 getMaximumBounds(){
    return agent_maximum;
}

glm::vec3 getMinimumBounds(){
    return agent_minimum;
}

</xsl:template>
</xsl:stylesheet>
