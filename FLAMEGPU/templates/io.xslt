<?xml version="1.0" encoding="utf-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:xmml="http://www.dcs.shef.ac.uk/~paul/XMML"
                xmlns:gpu="http://www.dcs.shef.ac.uk/~paul/XMMLGPU">
<xsl:output method="text" version="1.0" encoding="UTF-8" indent="yes" />
<xsl:include href = "./_common_templates.xslt" />
<xsl:template match="/">
<xsl:call-template name="copyrightNotice"></xsl:call-template>

#include &lt;cuda_runtime.h&gt;
#include &lt;stdlib.h&gt;
#include &lt;stdio.h&gt;
#include &lt;string.h&gt;
#include &lt;cmath&gt;
#include &lt;limits.h&gt;
#include &lt;algorithm&gt;
#include &lt;string&gt;
#include &lt;vector&gt;

<!-- If there are any json graphs, include the appropriate headers and suppress some errors -->
<xsl:if test="//gpu:staticGraph/gpu:loadFromFile/gpu:json">
#if defined __NVCC__
   // Disable the "statement is unreachable" message
   #pragma diag_suppress code_is_unreachable
   // Disable the "dynamic initialization in unreachable code" message
   #pragma diag_suppress initialization_not_reachable
#endif 
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
</xsl:if>

#ifdef _WIN32
#define strtok_r strtok_s
#endif

// include header
#include "header.h"

glm::vec3 agent_maximum;
glm::vec3 agent_minimum;

int fpgu_strtol(const char* str){
    return (int)strtol(str, NULL, 0);
}

unsigned int fpgu_strtoul(const char* str){
    return (unsigned int)strtoul(str, NULL, 0);
}

long long int fpgu_strtoll(const char* str){
    return strtoll(str, NULL, 0);
}

unsigned long long int fpgu_strtoull(const char* str){
    return strtoull(str, NULL, 0);
}

double fpgu_strtod(const char* str){
    return strtod(str, NULL);
}

float fgpu_atof(const char* str){
    return (float)atof(str);
}


//templated class function to read array inputs from supported types
template &lt;class T&gt;
void readArrayInput( T (*parseFunc)(const char*), char* buffer, T *array, unsigned int expected_items, const char * agent_name, const char * variable_name){
    unsigned int i = 0;
    const char s[2] = ",";
    char * token;
    char * end_str;

    token = strtok_r(buffer, s, &amp;end_str);
    while (token != NULL){
        if (i>=expected_items){
            printf("Error: variable array %s-&gt;%s has too many items (%d), expected %d!\n", agent_name, variable_name, i, expected_items);
            exit(EXIT_FAILURE);
        }
        
        array[i++] = (T)parseFunc(token);
        
        token = strtok_r(NULL, s, &amp;end_str);
    }
    if (i != expected_items){
        printf("Error: variable array %s-&gt;%s has %d items, expected %d!\n", agent_name, variable_name, i, expected_items);
        exit(EXIT_FAILURE);
    }
}

//templated class function to read array inputs from supported types
template &lt;class T, class BASE_T, unsigned int D&gt;
void readArrayInputVectorType( BASE_T (*parseFunc)(const char*), char* buffer, T *array, unsigned int expected_items, const char * agent_name, const char * variable_name){
    unsigned int i = 0;
    const char s[2] = "|";
    char * token;
    char * end_str;

    token = strtok_r(buffer, s, &amp;end_str);
    while (token != NULL){
        if (i>=expected_items){
            printf("Error: variable array of vectors %s-&gt;%s has too many items (%d), expected %d!\n", agent_name, variable_name, i, expected_items);
        }
        
        //read vector type as an array
        T vec;
        readArrayInput&lt;BASE_T&gt;(parseFunc, token, (BASE_T*) &amp;vec, D);
        array[i++] = vec;
        
        token = strtok_r(NULL, s, &amp;end_str);
    }
    if (i != expected_items){
        printf("Error: variable array of vectors %s-&gt;%s has %d items, expected %d!\n", agent_name, variable_name, i, expected_items);
        exit(EXIT_FAILURE);
    }
}

void saveIterationData(char* outputpath, int iteration_number, <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state">xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, int h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count<xsl:if test="position()!=last()">,</xsl:if></xsl:for-each>)
{
    PROFILE_SCOPED_RANGE("saveIterationData");
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
    if(file == nullptr){
        printf("Error: Could not open file `%s` for output. Aborting.\n", data);
        exit(EXIT_FAILURE);
    }
    fputs("&lt;states&gt;\n&lt;itno&gt;", file);
    sprintf(data, "%i", iteration_number);
    fputs(data, file);
    fputs("&lt;/itno&gt;\n", file);
    fputs("&lt;environment&gt;\n" , file);
    <xsl:for-each select="gpu:xmodel/gpu:environment/gpu:constants/gpu:variable">
    fputs("\t&lt;<xsl:value-of select="xmml:name"/>&gt;", file);<xsl:choose><xsl:when test="xmml:arrayLength">
    for (int j=0;j&lt;<xsl:value-of select="xmml:arrayLength"/>;j++){
        fprintf(file, "<xsl:call-template name="formatSpecifier"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>", <xsl:call-template name="outputEnvironmentConstantArrayItem"><xsl:with-param name="constant_name" select="xmml:name"/><xsl:with-param name="constant_type" select="xmml:type"/></xsl:call-template>);
        if(j!=(<xsl:value-of select="xmml:arrayLength"/>-1))
            <xsl:choose>
            <xsl:when test="contains(xmml:type, '2')">fprintf(file, "|");</xsl:when> 
            <xsl:when test="contains(xmml:type, '3')">fprintf(file, "|");</xsl:when> 
            <xsl:when test="contains(xmml:type, '4')">fprintf(file, "|");</xsl:when>
            <xsl:otherwise>fprintf(file, ",");</xsl:otherwise>
            </xsl:choose>
    }</xsl:when><xsl:otherwise>
    sprintf(data, "<xsl:call-template name="formatSpecifier"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>", <xsl:call-template name="outputEnvironmentConstant"><xsl:with-param name="constant_name" select="xmml:name"/><xsl:with-param name="constant_type" select="xmml:type"/></xsl:call-template>);
    fputs(data, file);</xsl:otherwise></xsl:choose>
    fputs("&lt;/<xsl:value-of select="xmml:name"/>&gt;\n", file);</xsl:for-each>
	fputs("&lt;/environment&gt;\n" , file);

	<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state"><xsl:variable name="stateName" select="xmml:name"/>//Write each <xsl:value-of select="../../xmml:name"/> agent to xml
	for (int i=0; i&lt;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count; i++){
		fputs("&lt;xagent&gt;\n" , file);
		fputs("&lt;name&gt;<xsl:value-of select="../../xmml:name"/>&lt;/name&gt;\n", file);
        <xsl:for-each select="../../xmml:memory/gpu:variable">
		fputs("&lt;<xsl:value-of select="xmml:name"/>&gt;", file);
        <xsl:choose><xsl:when test="xmml:arrayLength">for (int j=0;j&lt;<xsl:value-of select="xmml:arrayLength"/>;j++){
            fprintf(file, "<xsl:call-template name="formatSpecifier"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>", <xsl:call-template name="outputVariableArrayItem"><xsl:with-param name="agent_name" select="../../xmml:name"/><xsl:with-param name="state_name" select="$stateName"/><xsl:with-param name="variable_name" select="xmml:name"/><xsl:with-param name="variable_type" select="xmml:type"/></xsl:call-template>);
            if(j!=(<xsl:value-of select="xmml:arrayLength"/>-1))
                <xsl:choose>
                <xsl:when test="contains(xmml:type, '2')">fprintf(file, "|");</xsl:when> 
                <xsl:when test="contains(xmml:type, '3')">fprintf(file, "|");</xsl:when> 
                <xsl:when test="contains(xmml:type, '4')">fprintf(file, "|");</xsl:when>
                <xsl:otherwise>fprintf(file, ",");</xsl:otherwise>
                </xsl:choose>
        }</xsl:when><xsl:otherwise>sprintf(data, "<xsl:call-template name="formatSpecifier"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>", <xsl:call-template name="outputVariable"><xsl:with-param name="agent_name" select="../../xmml:name"/><xsl:with-param name="state_name" select="$stateName"/><xsl:with-param name="variable_name" select="xmml:name"/><xsl:with-param name="variable_type" select="xmml:type"/></xsl:call-template>);
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
<xsl:if test="gpu:xmodel/gpu:environment/gpu:constants/gpu:variable/xmml:defaultValue">
void initEnvVars()
{
PROFILE_SCOPED_RANGE("initEnvVars");
<xsl:for-each select="gpu:xmodel/gpu:environment/gpu:constants/gpu:variable">
<xsl:if test="xmml:defaultValue"><!--We cast the variable, so we don't accidentally create 1f or similiar if user omits decimal--><xsl:text>
    </xsl:text><xsl:value-of select="xmml:type"/> t_<xsl:value-of select="xmml:name"/> = (<xsl:value-of select="xmml:type"/>)<xsl:value-of select="xmml:defaultValue"/>;
    set_<xsl:value-of select="xmml:name"/>(&amp;t_<xsl:value-of select="xmml:name"/>);</xsl:if></xsl:for-each>
}
</xsl:if>
void readInitialStates(char* inputpath, <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">xmachine_memory_<xsl:value-of select="xmml:name"/>_list* h_<xsl:value-of select="xmml:name"/>s, int* h_xmachine_memory_<xsl:value-of select="xmml:name"/>_count<xsl:if test="position()!=last()">,</xsl:if></xsl:for-each>)
{
    PROFILE_SCOPED_RANGE("readInitialStates");

	int temp = 0;
	int* itno = &amp;temp;

	/* Pointer to file */
	FILE *file;
	/* Char and char buffer for reading file to */
	char c = ' ';
	const int bufferSize = 10000;
	char buffer[bufferSize];
	char agentname[1000];

	/* Pointer to x-memory for initial state data */
	/*xmachine * current_xmachine;*/
	/* Variables for checking tags */
	int reading, i;
	int in_tag, in_itno, in_xagent, in_name, in_comment;<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:memory/gpu:variable">
    int in_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>;</xsl:for-each>
    
    /* tags for environment global variables */
    int in_env;<xsl:for-each select="gpu:xmodel/gpu:environment/gpu:constants/gpu:variable">
    int in_env_<xsl:value-of select="xmml:name"/>;
    </xsl:for-each>
    
    <!-- initialise the population of all agent types to 0, to avoid launch failures -->
	/* set agent count to zero */<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent"><!--<xsl:if test="gpu:type='continuous'">-->
	*h_xmachine_memory_<xsl:value-of select="xmml:name"/>_count = 0;<!--</xsl:if>--></xsl:for-each>
	
	/* Variables for initial state data */<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:memory/gpu:variable"><xsl:choose><xsl:when test="xmml:arrayLength"><xsl:text>
    </xsl:text><xsl:value-of select="xmml:type"/><xsl:text> </xsl:text><xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>[<xsl:value-of select="xmml:arrayLength"/>];</xsl:when><xsl:otherwise><xsl:text>
	</xsl:text><xsl:value-of select="xmml:type"/><xsl:text> </xsl:text><xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>;</xsl:otherwise></xsl:choose></xsl:for-each>

    /* Variables for environment variables */
    <xsl:for-each select="gpu:xmodel/gpu:environment/gpu:constants/gpu:variable"><xsl:choose><xsl:when test="xmml:arrayLength">
    <xsl:value-of select="xmml:type"/> env_<xsl:value-of select="xmml:name"/>[<xsl:value-of select="xmml:arrayLength"/>];
    </xsl:when><xsl:otherwise>
    <xsl:value-of select="xmml:type"/> env_<xsl:value-of select="xmml:name"/>;
    </xsl:otherwise></xsl:choose></xsl:for-each>


	/* Initialise variables */<xsl:if test="gpu:xmodel/gpu:environment/gpu:constants/gpu:variable/xmml:defaultValue">
    initEnvVars();</xsl:if>
    agent_maximum.x = 0;
    agent_maximum.y = 0;
    agent_maximum.z = 0;
    agent_minimum.x = 0;
    agent_minimum.y = 0;
    agent_minimum.z = 0;
	reading = 1;
    in_comment = 0;
	in_tag = 0;
	in_itno = 0;
    in_env = 0;
    in_xagent = 0;
	in_name = 0;<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:memory/gpu:variable">
	in_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/> = 0;</xsl:for-each>
    <xsl:for-each select="gpu:xmodel/gpu:environment/gpu:constants/gpu:variable">
    in_env_<xsl:value-of select="xmml:name"/> = 0;</xsl:for-each>

	<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
	//set all <xsl:value-of select="xmml:name"/> values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k&lt;xmachine_memory_<xsl:value-of select="xmml:name"/>_MAX; k++)
	{	<xsl:for-each select="xmml:memory/gpu:variable"><xsl:choose><xsl:when test="xmml:arrayLength">
        for (i=0;i&lt;<xsl:value-of select="xmml:arrayLength"/>;i++){
            h_<xsl:value-of select="../../xmml:name"/>s-><xsl:value-of select="xmml:name"/>[(i*xmachine_memory_<xsl:value-of select="../../xmml:name"/>_MAX)+k] = <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/><xsl:with-param name="defaultValue" select="xmml:defaultValue" /></xsl:call-template>;
        }</xsl:when><xsl:otherwise>
		h_<xsl:value-of select="../../xmml:name"/>s-><xsl:value-of select="xmml:name"/>[k] = <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/><xsl:with-param name="defaultValue" select="xmml:defaultValue" /></xsl:call-template>;</xsl:otherwise></xsl:choose></xsl:for-each>
	}
	</xsl:for-each>

	/* Default variables for memory */<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:memory/gpu:variable"><xsl:choose><xsl:when test="xmml:arrayLength">
    for (i=0;i&lt;<xsl:value-of select="xmml:arrayLength"/>;i++){
        <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>[i] = <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/><xsl:with-param name="defaultValue" select="xmml:defaultValue" /></xsl:call-template>;
    }</xsl:when><xsl:otherwise><xsl:text>
    </xsl:text><xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/> = <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/><xsl:with-param name="defaultValue" select="xmml:defaultValue" /></xsl:call-template>;</xsl:otherwise>
    </xsl:choose></xsl:for-each>

    /* Default variables for environment variables */
    <xsl:for-each select="gpu:xmodel/gpu:environment/gpu:constants/gpu:variable"><xsl:choose><xsl:when test="xmml:arrayLength">
    for (i=0;i&lt;<xsl:value-of select="xmml:arrayLength"/>;i++){
        env_<xsl:value-of select="xmml:name"/>[i] = <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/><xsl:with-param name="defaultValue" select="xmml:defaultValue" /></xsl:call-template>;
    }
    </xsl:when><xsl:otherwise>env_<xsl:value-of select="xmml:name"/> = <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/><xsl:with-param name="defaultValue" select="xmml:defaultValue" /></xsl:call-template>;
    </xsl:otherwise></xsl:choose>
    </xsl:for-each>
    
    // If no input path was specified, issue a message and return.
    if(inputpath[0] == '\0'){
        printf("No initial states file specified. Using default values.\n");
        return;
    }
    
    // Otherwise an input path was specified, and we have previously checked that it is (was) not a directory. 
    
	// Attempt to open the non directory path as read only.
	file = fopen(inputpath, "r");
    
    // If the file could not be opened, issue a message and return.
    if(file == nullptr)
    {
      printf("Could not open input file %s. Continuing with default values\n", inputpath);
      return;
    }
    // Otherwise we can iterate the file until the end of XML is reached.
    size_t bytesRead = 0;
    i = 0;
	while(reading==1)
	{
        // If I exceeds our buffer size we must abort
        if(i >= bufferSize){
            fprintf(stderr, "Error: XML Parsing failed Tag name or content too long (> %d characters)\n", bufferSize);
            exit(EXIT_FAILURE);
        }

		/* Get the next char from the file */
		c = (char)fgetc(file);

        // Check if we reached the end of the file.
        if(c == EOF){
            // Break out of the loop. This allows for empty files(which may or may not be)
            break;
        }
        // Increment byte counter.
        bytesRead++;

        /*If in a  comment, look for the end of a comment */
        if(in_comment){

            /* Look for an end tag following two (or more) hyphens.
               To support very long comments, we use the minimal amount of buffer we can. 
               If we see a hyphen, store it and increment i (but don't increment i)
               If we see a &gt; check if we have a correct terminating comment
               If we see any other characters, reset i.
            */

            if(c == '-'){
                buffer[i] = c;
                i++;
            } else if(c == '&gt;' &amp;&amp; i >= 2){
                in_comment = 0;
                i = 0;
            } else {
                i = 0;
            }

            /*// If we see the end tag, check the preceding two characters for a close comment, if enough characters have been read for -->
            if(c == '&gt;' &amp;&amp; i >= 2 &amp;&amp; buffer[i-1] == '-' &amp;&amp; buffer[i-2] == '-'){
                in_comment = 0;
                buffer[0] = 0;
                i = 0;
            } else {
                // Otherwise just store it in the buffer so we can keep checking for close tags
                buffer[i] = c;
                i++;
            }*/
        }
		/* If the end of a tag */
		else if(c == '&gt;')
		{
			/* Place 0 at end of buffer to make chars a string */
			buffer[i] = 0;

			if(strcmp(buffer, "states") == 0) reading = 1;
			if(strcmp(buffer, "/states") == 0) reading = 0;
			if(strcmp(buffer, "itno") == 0) in_itno = 1;
			if(strcmp(buffer, "/itno") == 0) in_itno = 0;
            if(strcmp(buffer, "environment") == 0) in_env = 1;
            if(strcmp(buffer, "/environment") == 0) in_env = 0;
			if(strcmp(buffer, "name") == 0) in_name = 1;
			if(strcmp(buffer, "/name") == 0) in_name = 0;
            if(strcmp(buffer, "xagent") == 0) in_xagent = 1;
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
                    <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>[i] = <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/><xsl:with-param name="defaultValue" select="xmml:defaultValue" /></xsl:call-template>;
                }</xsl:when><xsl:otherwise><xsl:text>
                </xsl:text><xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/> = <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/><xsl:with-param name="defaultValue" select="xmml:defaultValue" /></xsl:call-template>;</xsl:otherwise></xsl:choose></xsl:for-each>
                
                in_xagent = 0;
			}
			<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:memory/gpu:variable">if(strcmp(buffer, "<xsl:value-of select="xmml:name"/>") == 0) in_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/> = 1;
			if(strcmp(buffer, "/<xsl:value-of select="xmml:name"/>") == 0) in_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/> = 0;
			</xsl:for-each>
            /* environment variables */
            <xsl:for-each select="gpu:xmodel/gpu:environment/gpu:constants/gpu:variable">if(strcmp(buffer, "<xsl:value-of select="xmml:name"/>") == 0) in_env_<xsl:value-of select="xmml:name"/> = 1;
            if(strcmp(buffer, "/<xsl:value-of select="xmml:name"/>") == 0) in_env_<xsl:value-of select="xmml:name"/> = 0;
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
			else if (in_xagent)
			{
				<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:memory/gpu:variable">if(in_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>){
                    <xsl:choose>
                      <xsl:when test="xmml:arrayLength">
                        <!-- Specialise input reads for vector types -->
                        <xsl:choose>
                        <xsl:when test="contains(xmml:type, '2')">readArrayInputVectorType&lt;<xsl:value-of select="xmml:type"/>, <xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, 2&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>, <xsl:value-of select="xmml:arrayLength"/>, "<xsl:value-of select="../../xmml:name"/>", "<xsl:value-of select="xmml:name"/>");    </xsl:when>
                        <xsl:when test="contains(xmml:type, '3')">readArrayInputVectorType&lt;<xsl:value-of select="xmml:type"/>, <xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, 3&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>, <xsl:value-of select="xmml:arrayLength"/>, "<xsl:value-of select="../../xmml:name"/>", "<xsl:value-of select="xmml:name"/>");    </xsl:when>
                        <xsl:when test="contains(xmml:type, '4')">readArrayInputVectorType&lt;<xsl:value-of select="xmml:type"/>, <xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, 4&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>, <xsl:value-of select="xmml:arrayLength"/>, "<xsl:value-of select="../../xmml:name"/>", "<xsl:value-of select="xmml:name"/>");    </xsl:when>
                        <xsl:otherwise>readArrayInput&lt;<xsl:value-of select="xmml:type"/>&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>, <xsl:value-of select="xmml:arrayLength"/>, "<xsl:value-of select="../../xmml:name"/>", "<xsl:value-of select="xmml:name"/>");    </xsl:otherwise>
                        </xsl:choose>        
                      </xsl:when>
                      <xsl:otherwise>
                        <!-- Specialise input reads for vector types -->
                        <xsl:choose>
                        <xsl:when test="contains(xmml:type, '2')">
                          readArrayInput&lt;<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, (<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>*)&amp;<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>, 2, "<xsl:value-of select="../../xmml:name"/>", "<xsl:value-of select="xmml:name"/>"); 
                        </xsl:when>
                        <xsl:when test="contains(xmml:type, '3')">
                          readArrayInput&lt;<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, (<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>*)&amp;<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>, 3, "<xsl:value-of select="../../xmml:name"/>", "<xsl:value-of select="xmml:name"/>"); 
                        </xsl:when>
                        <xsl:when test="contains(xmml:type, '4')">
                          readArrayInput&lt;<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, (<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>*)&amp;<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>, 4, "<xsl:value-of select="../../xmml:name"/>", "<xsl:value-of select="xmml:name"/>"); 
                        </xsl:when>
                        <xsl:otherwise><xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/> = (<xsl:value-of select="xmml:type"/>) <xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>(buffer); </xsl:otherwise>
                        </xsl:choose>
                      </xsl:otherwise>
                    </xsl:choose>
                }
				</xsl:for-each>
            }
            else if (in_env){
            <xsl:for-each select="gpu:xmodel/gpu:environment/gpu:constants/gpu:variable">if(in_env_<xsl:value-of select="xmml:name"/>){
              <xsl:choose>
                  <xsl:when test="xmml:arrayLength">
                    <!-- Specialise input reads for vector types -->
                    <xsl:choose>
                    <xsl:when test="contains(xmml:type, '2')">readArrayInputVectorType&lt;<xsl:value-of select="xmml:type"/>, <xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, 2&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, env_<xsl:value-of select="xmml:name"/>, <xsl:value-of select="xmml:arrayLength"/>, "environment", "<xsl:value-of select="xmml:name"/>");</xsl:when>
                    <xsl:when test="contains(xmml:type, '3')">readArrayInputVectorType&lt;<xsl:value-of select="xmml:type"/>, <xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, 3&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, env_<xsl:value-of select="xmml:name"/>, <xsl:value-of select="xmml:arrayLength"/>, "environment", "<xsl:value-of select="xmml:name"/>");</xsl:when>
                    <xsl:when test="contains(xmml:type, '4')">readArrayInputVectorType&lt;<xsl:value-of select="xmml:type"/>, <xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, 4&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, env_<xsl:value-of select="xmml:name"/>, <xsl:value-of select="xmml:arrayLength"/>, "environment", "<xsl:value-of select="xmml:name"/>");</xsl:when>
                    <xsl:otherwise>readArrayInput&lt;<xsl:value-of select="xmml:type"/>&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, env_<xsl:value-of select="xmml:name"/>, <xsl:value-of select="xmml:arrayLength"/>, "environment", "<xsl:value-of select="xmml:name"/>");</xsl:otherwise>
                    </xsl:choose>
                    set_<xsl:value-of select="xmml:name"/>(env_<xsl:value-of select="xmml:name"/>);
                  </xsl:when>
                  <xsl:otherwise>
                    <!-- Specialise input reads for vector types -->
                    <xsl:choose>
                    <xsl:when test="contains(xmml:type, '2')">
                      readArrayInput&lt;<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, (<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>*)&amp;env_<xsl:value-of select="xmml:name"/>, 2, "environment", "<xsl:value-of select="xmml:name"/>"); 
                    </xsl:when>
                    <xsl:when test="contains(xmml:type, '3')">
                      readArrayInput&lt;<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, (<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>*)&amp;env_<xsl:value-of select="xmml:name"/>, 3, "environment", "<xsl:value-of select="xmml:name"/>"); 
                    </xsl:when>
                    <xsl:when test="contains(xmml:type, '4')">
                      readArrayInput&lt;<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, (<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>*)&amp;env_<xsl:value-of select="xmml:name"/>, 4, "environment", "<xsl:value-of select="xmml:name"/>"); 
                    </xsl:when>
                    <xsl:otherwise>
                    env_<xsl:value-of select="xmml:name"/> = (<xsl:value-of select="xmml:type"/>) <xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>(buffer);
                    </xsl:otherwise>
                    </xsl:choose>
                    set_<xsl:value-of select="xmml:name"/>(&amp;env_<xsl:value-of select="xmml:name"/>);
                  </xsl:otherwise>
                </xsl:choose>
              }
            </xsl:for-each>
            }
		/* Reset buffer */
			i = 0;
		}
		/* If in tag put read char into buffer */
		else if(in_tag)
		{
            // Check if we are a comment, when we are in a tag and buffer[0:2] == "!--"
            if(i == 2 &amp;&amp; c == '-' &amp;&amp; buffer[1] == '-' &amp;&amp; buffer[0] == '!'){
                in_comment = 1;
                // Reset the buffer and i.
                buffer[0] = 0;
                i = 0;
            }

            // Store the character and increment the counter
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
    // If no bytes were read, raise a warning.
    if(bytesRead == 0){
        fprintf(stdout, "Warning: %s is an empty file\n", inputpath);
        fflush(stdout);
    }

    // If the in_comment flag is still marked, issue a warning.
    if(in_comment){
        fprintf(stdout, "Warning: Un-terminated comment in %s\n", inputpath);
        fflush(stdout);
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


/* Methods to load static networks from disk */
<xsl:for-each select="gpu:xmodel/gpu:environment/gpu:graphs/gpu:staticGraph">
<xsl:variable name="graph_name" select = "gpu:name"/>

/*
 * bool checkForDuplicates_staticGraph_<xsl:value-of select="$graph_name"/>(staticGraph_memory_<xsl:value-of select="$graph_name"/>* graph)
 * Checks a static graph for duplicate entries, which are not allowed.
 * @param graph pointer to static graph
 * @return boolean indicator of success
 */
bool checkForDuplicates_staticGraph_<xsl:value-of select="$graph_name"/>(staticGraph_memory_<xsl:value-of select="$graph_name"/>* graph){
    // Check for duplicate entries, by parsing all edges which are now sorted.
    if(graph-&gt;edge.count &gt; 1){
        unsigned int prevSource = graph-&gt;edge.source[0];
        unsigned int prevDest = graph-&gt;edge.destination[0];
        for(unsigned int i = 1; i &lt; graph-&gt;edge.count; i++){
            // If 2 sequential edges are the same, there is an error.
            if(prevSource == graph-&gt;edge.source[i] &amp;&amp; prevDest == graph-&gt;edge.destination[i]){
                return true;
            }
            prevSource = graph-&gt;edge.source[i];
            prevDest = graph-&gt;edge.destination[i];
        }
    }
    return false;
}

/*
 * void coo_to_csr_staticGraph_<xsl:value-of select="$graph_name"/>(staticGraph_memory_<xsl:value-of select="$graph_name"/>* coo, staticGraph_memory_<xsl:value-of select="$graph_name"/>* csr)
 * Converts a COO (unsorted) graph into the Compressed Sparse Row (CSR) representation.
 * @param coo graph in unsorted order
 * @param csr graph sorted and stored as CSR
 */
 void coo_to_csr_staticGraph_<xsl:value-of select="$graph_name"/>(staticGraph_memory_<xsl:value-of select="$graph_name"/>* coo, staticGraph_memory_<xsl:value-of select="$graph_name"/>* csr){
    // Copy counts across to the CSR data structure using the new indices.
    csr-&gt;vertex.count = coo-&gt;vertex.count;
    csr-&gt;edge.count = coo-&gt;edge.count;

    // Initialise the csr first edge pointers to 0.
    std::fill(csr-&gt;vertex.first_edge_index, csr-&gt;vertex.first_edge_index + coo-&gt;vertex.count, 0);

    // For each edge, increment the pointer for the source vertex.
    for(unsigned int i = 0; i &lt; coo-&gt;edge.count; i++){
        csr-&gt;vertex.first_edge_index[coo-&gt;edge.source[i]]++;
    }

    // Inclusive prefix sum across these values to get the final value for each vertex
    unsigned int total = 0;
    for(unsigned int i = 0; i &lt; coo-&gt;vertex.count; i++){
        unsigned int old_value = csr-&gt;vertex.first_edge_index[i];
        csr-&gt;vertex.first_edge_index[i] = total;
        total += old_value;
    }
    // Populate the |V| + 1 value
    csr-&gt;vertex.first_edge_index[coo-&gt;vertex.count] = coo-&gt;edge.count;


    // Sort vertices by id. 
    // Create a vector of pairs 
    <xsl:for-each select="gpu:vertex/xmml:variables/gpu:variable[xmml:name='id']">
    std::vector&lt;std::pair&lt;unsigned int,<xsl:value-of select="xmml:type" />&gt;&gt; vertex_indices (coo-&gt;vertex.count);
    // Populate the pairs.
    for(unsigned int i = 0; i &lt; coo-&gt;vertex.count; i++){
        vertex_indices.at(i).first = i;
        vertex_indices.at(i).second = coo-&gt;vertex.id[i] ;
    }
    // sort the vector of indices based on the value of the COO vertex ids.
    std::sort(vertex_indices.begin(), vertex_indices.end(), [](const std::pair&lt;unsigned int,<xsl:value-of select="xmml:type" />&gt; &amp;left, const std::pair&lt;unsigned int,<xsl:value-of select="xmml:type" />&gt; &amp;right) {
        return left.second &lt; right.second;
    });

    </xsl:for-each>
    // Scatter vertices data from coo to csr order
    for(unsigned int coo_index = 0; coo_index &lt; coo-&gt;vertex.count; coo_index++){
        unsigned int csr_index = vertex_indices.at(coo_index).first;
        <!-- unsigned int new_index = csr-&gt;vertex.first_edge_index[source_vertex]; -->
        <xsl:for-each select="gpu:vertex/xmml:variables/gpu:variable"><xsl:choose>
        <xsl:when test="xmml:arrayLength">for(unsigned int i = 0; i &lt; <xsl:value-of select="xmml:arrayLength" />; i++){
            csr-&gt;vertex.<xsl:value-of select="xmml:name"/>[(i*staticGraph_<xsl:value-of select="$graph_name"/>_vertex_bufferSize)+csr_index] = coo-&gt;vertex.<xsl:value-of select="xmml:name"/>[(i*staticGraph_<xsl:value-of select="$graph_name"/>_vertex_bufferSize)+coo_index];
        }
        </xsl:when>
        <xsl:otherwise>csr-&gt;vertex.<xsl:value-of select="xmml:name"/>[csr_index] = coo-&gt;vertex.<xsl:value-of select="xmml:name"/>[coo_index];
        </xsl:otherwise>
        </xsl:choose></xsl:for-each>
    }

    // Scatter values to complete the csr data
    for(unsigned int coo_index = 0; coo_index &lt; coo-&gt;edge.count; coo_index++){
        unsigned int source_vertex = coo-&gt;edge.source[coo_index];
        unsigned int csr_index = csr-&gt;vertex.first_edge_index[source_vertex];
        <xsl:for-each select="gpu:edge/xmml:variables/gpu:variable"><xsl:choose>
        <xsl:when test="xmml:arrayLength">for(unsigned int i = 0; i &lt; <xsl:value-of select="xmml:arrayLength" />; i++){
            csr-&gt;edge.<xsl:value-of select="xmml:name"/>[(i*staticGraph_<xsl:value-of select="$graph_name"/>_edge_bufferSize)+csr_index] = coo-&gt;edge.<xsl:value-of select="xmml:name"/>[(i*staticGraph_<xsl:value-of select="$graph_name"/>_edge_bufferSize)+coo_index];
        }
        </xsl:when>
        <xsl:otherwise>csr-&gt;edge.<xsl:value-of select="xmml:name"/>[csr_index] = coo-&gt;edge.<xsl:value-of select="xmml:name"/>[coo_index];
        </xsl:otherwise>
        </xsl:choose></xsl:for-each>
        csr-&gt;vertex.first_edge_index[source_vertex]++;
    }

    // Fill in any gaps in the CSR
    unsigned int previous_value = 0;
    for (unsigned int i = 0 ; i &lt;= csr-&gt;vertex.count; i++){
        unsigned int old_value = csr-&gt;vertex.first_edge_index[i];
        csr-&gt;vertex.first_edge_index[i] = previous_value;
        previous_value = old_value;
    }
}


<xsl:if test="gpu:loadFromFile/gpu:json">
/* void load_staticGraph_<xsl:value-of select="$graph_name"/>_from_json(const char* file, staticGraph_memory_<xsl:value-of select="$graph_name"/>* h_staticGraph_memory_<xsl:value-of select="$graph_name"/>)
 * Load a static graph from a JSON file on disk.
 * @param file input filename
 * @param h_staticGraph_memory_<xsl:value-of select="$graph_name"/> pointer to graph.
 */
void load_staticGraph_<xsl:value-of select="$graph_name"/>_from_json(const char* file, staticGraph_memory_<xsl:value-of select="$graph_name"/>* h_staticGraph_memory_<xsl:value-of select="$graph_name"/>){
    PROFILE_SCOPED_RANGE("loadGraphFromJSON");
    // Build the path to the file from the working directory by joining the input directory path and the specified file name from XML
    std::string pathToFile(getOutputDir(), strlen(getOutputDir()));
    pathToFile.append("<xsl:value-of select="gpu:loadFromFile/gpu:json"/>");

    FILE *filePointer = fopen(pathToFile.c_str(), "rb");
    // Ensure the File exists
    if (filePointer == nullptr){
        fprintf(stderr, "FATAL ERROR: network file %s could not be opened.\n", pathToFile.c_str());
        exit(EXIT_FAILURE);
    }

    // Print the file being loaded
    fprintf(stdout, "Loading staticGraph <xsl:value-of select="$graph_name"/> from json file %s\n", pathToFile.c_str());

    // Get the length of the file
    fseek(filePointer, 0, SEEK_END);
    long filesize = ftell(filePointer);
    fseek(filePointer, 0, SEEK_SET);

    // Allocate and load the file into memory
    char *string = (char*)malloc(filesize + 1);
    if(string == nullptr){
        fprintf(stderr, "FATAL ERROR: Could not allocate memory to parse %s\n", pathToFile.c_str());
        fclose(filePointer);
        exit(EXIT_FAILURE);
    }
    fread(string, filesize, 1, filePointer);
    fclose(filePointer);
    // terminate the string
    string[filesize] = 0;

    // Use rapidJson to parse the loaded data.
    rapidjson::Document document;
    document.Parse(string);

    <xsl:if test="gpu:vertex/xmml:variables/gpu:variable/xmml:arrayLength or gpu:edge/xmml:variables/gpu:variable/xmml:arrayLength">
    size_t expectedArrayElements = 0;
    </xsl:if>

    // Check Json was valid and contained the required values.
    if (document.IsObject()){
        // Get value references to the relevant json objects
        const rapidjson::Value&amp; vertices = document["vertices"];
        const rapidjson::Value&amp; edges = document["edges"];

        // Get the number of edges and vertices
        unsigned int vertex_count = (vertices.IsArray()) ? vertices.Size() : 0;
        unsigned int edge_count = (edges.IsArray()) ? edges.Size() : 0;

        // If either dimensions is greater than the maximum allowed elements then we must error and exit.
        if(vertex_count &gt; staticGraph_<xsl:value-of select="$graph_name"/>_vertex_bufferSize || edge_count &gt; staticGraph_<xsl:value-of select="$graph_name"/>_edge_bufferSize){
            fprintf(
                stderr,
                "FATAL ERROR: Static Graph <xsl:value-of select="$graph_name"/> (%u vertices, %u edges) exceeds buffer dimensions (%u vertices, %u edges)",
                vertex_count,
                edge_count,
                staticGraph_<xsl:value-of select="$graph_name"/>_vertex_bufferSize,
                staticGraph_<xsl:value-of select="$graph_name"/>_edge_bufferSize 
            );
            exit(EXIT_FAILURE);
        }

        // Allocate a local COO object to load data into from disk.
        staticGraph_memory_<xsl:value-of select="$graph_name"/>* coo = (staticGraph_memory_<xsl:value-of select="$graph_name"/> *) malloc(sizeof(staticGraph_memory_<xsl:value-of select="$graph_name"/>));

        // Ensure it allocated.
        if(coo == nullptr){
            fprintf(stderr, "FATAL ERROR: Could not allocate memory for staticGraph <xsl:value-of select="$graph_name"/> while loading from disk\n");
            exit(EXIT_FAILURE);
        }

        // Store the counts in the COO graph
        coo-&gt;edge.count = edge_count;
        coo-&gt;vertex.count = vertex_count;

        // For each vertex element in the file, load the relevant values into the COO memory, otherwise set defaults.
        for (rapidjson::SizeType i = 0; i &lt; coo-&gt;vertex.count; i++){
            // Set default values for variables.
            <xsl:for-each select="gpu:vertex/xmml:variables/gpu:variable">
            <xsl:choose>
            <xsl:when test="xmml:arrayLength">expectedArrayElements = <xsl:value-of select="xmml:arrayLength"/>;
            // Set each element to the default value. 
            for(size_t arrayElement = 0; arrayElement &lt; expectedArrayElements; arrayElement++){
                coo-&gt;vertex.<xsl:value-of select="xmml:name" />[(arrayElement * staticGraph_<xsl:value-of select="$graph_name"/>_vertex_bufferSize) + i] = <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/><xsl:with-param name="defaultValue" select="xmml:defaultValue" /></xsl:call-template>;
            }
            </xsl:when>
            <xsl:otherwise>coo-&gt;vertex.<xsl:value-of select="xmml:name" />[i] = <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/><xsl:with-param name="defaultValue" select="xmml:defaultValue" /></xsl:call-template>;
            </xsl:otherwise>
            </xsl:choose>
            </xsl:for-each>

            // Attempt to read the correct value from JSON
            <xsl:for-each select="gpu:vertex/xmml:variables/gpu:variable"><xsl:choose>
            <xsl:when test="xmml:arrayLength">expectedArrayElements = <xsl:value-of select="xmml:arrayLength"/>;
            // If an array is present in the json with the correct key, and the array is not empty.
            if (vertices[i].HasMember("<xsl:value-of select="xmml:name"/>") &amp;&amp; vertices[i]["<xsl:value-of select="xmml:name" />"].IsArray() &amp;&amp; vertices[i]["<xsl:value-of select="xmml:name" />"].Size() > 0){
                size_t actualArrayElements = vertices[i]["<xsl:value-of select="xmml:name" />"].Size();
                // If the array is too long, raise a warning. 
                if(actualArrayElements > expectedArrayElements){
                    fprintf(stderr,"Warning: Too many elements for vertex variable array <xsl:value-of select="xmml:name"/>. Expected <xsl:value-of select="xmml:arrayLength"/> found %lu\n", actualArrayElements);
                }
                // For each element in the array. 
                for(size_t arrayElement = 0; arrayElement &lt; std::max(expectedArrayElements, actualArrayElements); arrayElement++ ){
                    <xsl:choose>
                    <xsl:when test="contains(xmml:type,'vec')"><xsl:choose>
                    <xsl:when test="contains(xmml:type, 'vec2')">size_t expectedVecElements = 2;</xsl:when>
                    <xsl:when test="contains(xmml:type, 'vec3')">size_t expectedVecElements = 3;</xsl:when>
                    <xsl:when test="contains(xmml:type, 'vec4')">size_t expectedVecElements = 4;</xsl:when>
                    </xsl:choose>
                    size_t actualVecElements = vertices[i]["<xsl:value-of select="xmml:name" />"][arrayElement].Size();
                    // Warn if too many elements.
                    if(actualVecElements &gt; expectedVecElements){
                        fprintf(stderr, "Warning: Too many vector elements provided for vertex vector type variable <xsl:value-of select="xmml:name" />. Expected %lu found %lu\n", expectedVecElements, actualVecElements);
                    }
                    // For upto the min of the number of elements and actual elements
                    size_t vecElementsToSet = std::min(expectedVecElements, actualVecElements);
                    for(unsigned int vecElement = 0; vecElement &lt; vecElementsToSet; vecElement++ ){
                    // If we have a value to store, place it. 
                        coo-&gt;vertex.<xsl:value-of select="xmml:name" />[(arrayElement * staticGraph_<xsl:value-of select="$graph_name"/>_vertex_bufferSize) + i][vecElement] = vertices[i]["<xsl:value-of select="xmml:name" />"][arrayElement][vecElement].Get&lt;<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>&gt;();
                    }
                    </xsl:when>
                    <xsl:otherwise>if (vertices[i]["<xsl:value-of select="xmml:name" />"][arrayElement].Is&lt;<xsl:value-of select="xmml:type" />&gt;()){
                        coo-&gt;vertex.<xsl:value-of select="xmml:name" />[(arrayElement * staticGraph_<xsl:value-of select="$graph_name"/>_vertex_bufferSize) + i] = vertices[i]["<xsl:value-of select="xmml:name" />"][arrayElement].Get&lt;<xsl:value-of select="xmml:type" />&gt;();
                    }</xsl:otherwise></xsl:choose>
                }
            }
            </xsl:when>
            <xsl:otherwise><xsl:choose>
            <xsl:when test="contains(xmml:type, 'vec')">if (vertices[i].HasMember("<xsl:value-of select="xmml:name"/>") &amp;&amp; vertices[i]["<xsl:value-of select="xmml:name" />"].IsArray() &amp;&amp; vertices[i]["<xsl:value-of select="xmml:name" />"].Size() > 0){
                <xsl:choose><xsl:when test="contains(xmml:type, 'vec2')">size_t expectedVecElements = 2;</xsl:when>
                <xsl:when test="contains(xmml:type, 'vec3')">size_t expectedVecElements = 3;</xsl:when>
                <xsl:when test="contains(xmml:type, 'vec4')">size_t expectedVecElements = 4;</xsl:when>
                </xsl:choose>
                size_t actualVecElements = vertices[i]["<xsl:value-of select="xmml:name" />"].Size();
                // Warn if too many elements.
                if(actualVecElements &gt; expectedVecElements){
                    fprintf(stderr, "Warning: too many vector elements provided for vertex vector type variable <xsl:value-of select="xmml:name" />\n");
                }
                // For upto the min of the number of elements and actual elements
                size_t vecElementsToSet = std::min(expectedVecElements, actualVecElements);
                for(unsigned int vecElement = 0; vecElement &lt; vecElementsToSet; vecElement++ ){
                    // If we have a value to store, place it. 
                    coo-&gt;vertex.<xsl:value-of select="xmml:name" />[i][vecElement] = vertices[i]["<xsl:value-of select="xmml:name" />"][vecElement].Get&lt;<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>&gt;();
                }
            }
            </xsl:when>
            <xsl:otherwise>if (vertices[i].HasMember("<xsl:value-of select="xmml:name"/>") &amp;&amp; vertices[i]["<xsl:value-of select="xmml:name" />"].Is&lt;<xsl:value-of select="xmml:type" />&gt;()){
                coo-&gt;vertex.<xsl:value-of select="xmml:name" />[i] = vertices[i]["<xsl:value-of select="xmml:name" />"].Get&lt;<xsl:value-of select="xmml:type" />&gt;();
            }
            </xsl:otherwise>
            </xsl:choose></xsl:otherwise>
        </xsl:choose>
        </xsl:for-each>
        }

        // For each edge element in the file, load the relevant values into memory, otherwise set defaults.
        for (rapidjson::SizeType i = 0; i &lt; coo-&gt;edge.count; i++){
            // Set default values for variables.
            <xsl:for-each select="gpu:edge/xmml:variables/gpu:variable">
            <xsl:choose>
            <xsl:when test="xmml:arrayLength">expectedArrayElements = <xsl:value-of select="xmml:arrayLength"/>;
            // Set each element to the default value. 
            for(size_t arrayElement = 0; arrayElement &lt; expectedArrayElements; arrayElement++){
                coo-&gt;edge.<xsl:value-of select="xmml:name" />[(arrayElement * staticGraph_<xsl:value-of select="$graph_name"/>_edge_bufferSize) + i] = <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/><xsl:with-param name="defaultValue" select="xmml:defaultValue" /></xsl:call-template>;
            }
            </xsl:when>
            <xsl:otherwise>coo-&gt;edge.<xsl:value-of select="xmml:name" />[i] = <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/><xsl:with-param name="defaultValue" select="xmml:defaultValue" /></xsl:call-template>;
            </xsl:otherwise>
            </xsl:choose>
            </xsl:for-each>

            // Attempt to read the correct value from JSON
            <xsl:for-each select="gpu:edge/xmml:variables/gpu:variable"><xsl:choose>
            <xsl:when test="xmml:arrayLength">expectedArrayElements = <xsl:value-of select="xmml:arrayLength"/>;
            // If an array is present in the json with the correct key, and the array is not empty.
            if (edges[i].HasMember("<xsl:value-of select="xmml:name"/>") &amp;&amp; edges[i]["<xsl:value-of select="xmml:name" />"].IsArray() &amp;&amp; edges[i]["<xsl:value-of select="xmml:name" />"].Size() > 0){
                size_t actualArrayElements = edges[i]["<xsl:value-of select="xmml:name" />"].Size();
                // If the array is too long, raise a warning. 
                if(actualArrayElements > expectedArrayElements){
                    fprintf(stderr,"Warning: Too many elements for edge variable array <xsl:value-of select="xmml:name"/>. Expected <xsl:value-of select="xmml:arrayLength"/> found %lu\n", actualArrayElements);
                }
                // For each element in the array. 
                for(size_t arrayElement = 0; arrayElement &lt; std::max(expectedArrayElements, actualArrayElements); arrayElement++ ){
                    <xsl:choose>
                    <xsl:when test="contains(xmml:type,'vec')"><xsl:choose>
                    <xsl:when test="contains(xmml:type, 'vec2')">size_t expectedVecElements = 2;</xsl:when>
                    <xsl:when test="contains(xmml:type, 'vec3')">size_t expectedVecElements = 3;</xsl:when>
                    <xsl:when test="contains(xmml:type, 'vec4')">size_t expectedVecElements = 4;</xsl:when>
                    </xsl:choose>
                    size_t actualVecElements = edges[i]["<xsl:value-of select="xmml:name" />"][arrayElement].Size();
                    // Warn if too many elements.
                    if(actualVecElements &gt; expectedVecElements){
                        fprintf(stderr, "Warning: Too many vector elements provided for edge vector type variable <xsl:value-of select="xmml:name" />. Expected %lu found %lu\n", expectedVecElements, actualVecElements);
                    }
                    // For upto the min of the number of elements and actual elements
                    size_t vecElementsToSet = std::min(expectedVecElements, actualVecElements);
                    for(unsigned int vecElement = 0; vecElement &lt; vecElementsToSet; vecElement++ ){
                    // If we have a value to store, place it. 
                        coo-&gt;edge.<xsl:value-of select="xmml:name" />[(arrayElement * staticGraph_<xsl:value-of select="$graph_name"/>_edge_bufferSize) + i][vecElement] = edges[i]["<xsl:value-of select="xmml:name" />"][arrayElement][vecElement].Get&lt;<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>&gt;();
                    }
                    </xsl:when>
                    <xsl:otherwise>if (edges[i]["<xsl:value-of select="xmml:name" />"][arrayElement].Is&lt;<xsl:value-of select="xmml:type" />&gt;()){
                        coo-&gt;edge.<xsl:value-of select="xmml:name" />[(arrayElement * staticGraph_<xsl:value-of select="$graph_name"/>_edge_bufferSize) + i] = edges[i]["<xsl:value-of select="xmml:name" />"][arrayElement].Get&lt;<xsl:value-of select="xmml:type" />&gt;();
                    }</xsl:otherwise></xsl:choose>
                }
            }
            </xsl:when>
            <xsl:otherwise><xsl:choose>
            <xsl:when test="contains(xmml:type, 'vec')">if (edges[i].HasMember("<xsl:value-of select="xmml:name"/>") &amp;&amp; edges[i]["<xsl:value-of select="xmml:name" />"].IsArray() &amp;&amp; edges[i]["<xsl:value-of select="xmml:name" />"].Size() > 0){
                <xsl:choose><xsl:when test="contains(xmml:type, 'vec2')">size_t expectedVecElements = 2;</xsl:when>
                <xsl:when test="contains(xmml:type, 'vec3')">size_t expectedVecElements = 3;</xsl:when>
                <xsl:when test="contains(xmml:type, 'vec4')">size_t expectedVecElements = 4;</xsl:when>
                </xsl:choose>
                size_t actualVecElements = edges[i]["<xsl:value-of select="xmml:name" />"].Size();
                // Warn if too many elements.
                if(actualVecElements &gt; expectedVecElements){
                    fprintf(stderr, "Warning: too many vector elements provided for edge vector type variable <xsl:value-of select="xmml:name" />\n");
                }
                // For upto the min of the number of elements and actual elements
                size_t vecElementsToSet = std::min(expectedVecElements, actualVecElements);
                for(unsigned int vecElement = 0; vecElement &lt; vecElementsToSet; vecElement++ ){
                    // If we have a value to store, place it. 
                    coo-&gt;edge.<xsl:value-of select="xmml:name" />[i][vecElement] = edges[i]["<xsl:value-of select="xmml:name" />"][vecElement].Get&lt;<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>&gt;();
                }
            }
            </xsl:when>
            <xsl:otherwise>if (edges[i].HasMember("<xsl:value-of select="xmml:name"/>") &amp;&amp; edges[i]["<xsl:value-of select="xmml:name" />"].Is&lt;<xsl:value-of select="xmml:type" />&gt;()){
                coo-&gt;edge.<xsl:value-of select="xmml:name" />[i] = edges[i]["<xsl:value-of select="xmml:name" />"].Get&lt;<xsl:value-of select="xmml:type" />&gt;();
            }
            </xsl:otherwise>
            </xsl:choose></xsl:otherwise>
        </xsl:choose>
        </xsl:for-each>
        }

        // Construct the CSR representation from COO
        coo_to_csr_staticGraph_<xsl:value-of select="$graph_name"/>(coo, h_staticGraph_memory_<xsl:value-of select="$graph_name"/>);

        // Check for duplicate edges (undefined behaviour)
        bool has_duplicates = checkForDuplicates_staticGraph_<xsl:value-of select="$graph_name"/>( h_staticGraph_memory_<xsl:value-of select="$graph_name"/>);
        if(has_duplicates){
            printf("FATAL ERROR: Duplicate edge found in staticGraph <xsl:value-of select="$graph_name"/>\n");
            free(coo);
            exit(EXIT_FAILURE);
        }

        // Free the COO representation
        free(coo);
        coo = nullptr;

    } else {
        // Otherwise it is not an object and we have failed.
        printf("FATAL ERROR: Network file %s is not a valid JSON file\n", pathToFile.c_str());
        exit(EXIT_FAILURE);
    }

    fprintf(stdout, "Loaded %u vertices, %u edges\n", h_staticGraph_memory_<xsl:value-of select="$graph_name"/>-&gt;vertex.count, h_staticGraph_memory_<xsl:value-of select="$graph_name"/>-&gt;edge.count);

}
</xsl:if>
<xsl:if test="gpu:loadFromFile/gpu:xml">
/* void load_staticGraph_<xsl:value-of select="$graph_name"/>_from_xml(const char* file, staticGraph_memory_<xsl:value-of select="$graph_name"/>* h_staticGraph_memory_<xsl:value-of select="$graph_name"/>)
 * Load a static graph from a JSON file on disk.
 * @param file input filename
 * @param csr pointer to graph in csr format
 */
void load_staticGraph_<xsl:value-of select="$graph_name"/>_from_xml(const char* file, staticGraph_memory_<xsl:value-of select="$graph_name"/>* csr){
    PROFILE_SCOPED_RANGE("loadGraphFromXML");
    
    // Build the path to the file from the working directory by joining the input directory path and the specified file name from XML
    std::string pathToFile(getOutputDir(), strlen(getOutputDir()));
    pathToFile.append("<xsl:value-of select="gpu:loadFromFile/gpu:xml"/>");

    FILE *filePointer = fopen(pathToFile.c_str(), "rb");
    // Ensure the File exists
    if (filePointer == nullptr){
        printf("FATAL ERROR: network file %s could not be opened.\n", pathToFile.c_str());
        exit(EXIT_FAILURE);
    }

    // Print the file being loaded
    fprintf(stdout, "Loading staticGraph <xsl:value-of select="$graph_name"/> from xml file %s\n", pathToFile.c_str());


    // Allocate the COO to load from disk into 
    staticGraph_memory_<xsl:value-of select="$graph_name"/>* coo = (staticGraph_memory_<xsl:value-of select="$graph_name"/> *) malloc(sizeof(staticGraph_memory_<xsl:value-of select="$graph_name"/>));

    // Ensure it allocated.
    if(coo == nullptr){
        printf("FATAL ERROR: Could not allocate memory for staticGraph <xsl:value-of select="$graph_name"/> while loading from disk\n");
        exit(EXIT_FAILURE);
    }

       

    // Get the length of the file
    fseek(filePointer, 0, SEEK_END);
    long filesize = ftell(filePointer);
    fseek(filePointer, 0, SEEK_SET);

    /* Char and char buffer for reading file to */
    char c = ' ';
    const int bufferSize = 10000;
    char buffer[bufferSize];

    /* Variables for checking tags */
    int reading = 1;
    int i = 0;
    int in_tag = 0;
    <!-- int in_graph = 0; -->
    int in_vertices = 0;
    int in_vertex = 0;
    int in_edges = 0;
    int in_edge = 0;
    int in_comment = 0;


    <xsl:for-each select="gpu:vertex/xmml:variables/gpu:variable">
    int in_vertex_<xsl:value-of select="xmml:name"/> = 0;</xsl:for-each>
    <xsl:for-each select="gpu:edge/xmml:variables/gpu:variable">
    int in_edge_<xsl:value-of select="xmml:name"/> = 0;</xsl:for-each>

    // Initialise global memory on the host for the graph data structure to default value. 
    for (unsigned int k=0; k &lt; staticGraph_<xsl:value-of select="$graph_name"/>_vertex_bufferSize; k++){
    <xsl:for-each select="gpu:vertex/xmml:variables/gpu:variable"><xsl:choose><xsl:when test="xmml:arrayLength">
        for (i=0;i&lt;<xsl:value-of select="xmml:arrayLength"/>;i++){
            coo-&gt;vertex.<xsl:value-of select="xmml:name"/>[(i*staticGraph_<xsl:value-of select="$graph_name"/>_vertex_bufferSize)+k] = <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/><xsl:with-param name="defaultValue" select="xmml:defaultValue" /></xsl:call-template>;
        }</xsl:when><xsl:otherwise>
        coo-&gt;vertex.<xsl:value-of select="xmml:name"/>[k] = <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>;</xsl:otherwise></xsl:choose></xsl:for-each>
    }
    for (unsigned int k=0; k &lt; staticGraph_<xsl:value-of select="$graph_name"/>_vertex_bufferSize; k++){
    <xsl:for-each select="gpu:edge/xmml:variables/gpu:variable"><xsl:choose><xsl:when test="xmml:arrayLength">
        for (i=0;i&lt;<xsl:value-of select="xmml:arrayLength"/>;i++){
            coo-&gt;edge.<xsl:value-of select="xmml:name"/>[(i*staticGraph_<xsl:value-of select="$graph_name"/>_edge_bufferSize)+k] = <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/><xsl:with-param name="defaultValue" select="xmml:defaultValue" /></xsl:call-template>;
        }</xsl:when><xsl:otherwise>
        coo-&gt;edge.<xsl:value-of select="xmml:name"/>[k] = <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/><xsl:with-param name="defaultValue" select="xmml:defaultValue" /></xsl:call-template>;</xsl:otherwise></xsl:choose></xsl:for-each>
    }

    /* Variables for initial state data */
    <xsl:for-each select="gpu:vertex/xmml:variables/gpu:variable">
    <xsl:choose><xsl:when test="xmml:arrayLength"><xsl:value-of select="xmml:type"/> vertex_<xsl:value-of select="xmml:name"/>[<xsl:value-of select="xmml:arrayLength"/>];
    for(unsigned int i = 0; i &lt; <xsl:value-of select="xmml:arrayLength" />; i++){
        vertex_<xsl:value-of select="xmml:name"/>[i] = <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/><xsl:with-param name="defaultValue" select="xmml:defaultValue" /></xsl:call-template>;
    }</xsl:when>
    <xsl:otherwise><xsl:value-of select="xmml:type"/> vertex_<xsl:value-of select="xmml:name"/> = <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/><xsl:with-param name="defaultValue" select="xmml:defaultValue" /></xsl:call-template>;</xsl:otherwise>
    </xsl:choose><xsl:text>
    </xsl:text></xsl:for-each>

    <xsl:for-each select="gpu:edge/xmml:variables/gpu:variable">
    <xsl:choose><xsl:when test="xmml:arrayLength"><xsl:value-of select="xmml:type"/> edge_<xsl:value-of select="xmml:name"/>[<xsl:value-of select="xmml:arrayLength"/>];
    for(unsigned int i = 0; i &lt; <xsl:value-of select="xmml:arrayLength" />; i++){
        <xsl:value-of select="xmml:type"/> edge_<xsl:value-of select="xmml:name"/>[i] = <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/><xsl:with-param name="defaultValue" select="xmml:defaultValue" /></xsl:call-template>;
    }</xsl:when>
    <xsl:otherwise><xsl:value-of select="xmml:type"/> edge_<xsl:value-of select="xmml:name"/> = <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/><xsl:with-param name="defaultValue" select="xmml:defaultValue" /></xsl:call-template>;</xsl:otherwise>
    </xsl:choose><xsl:text>
    </xsl:text></xsl:for-each>

    <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:memory/gpu:variable">
    in_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/> = 0;</xsl:for-each>

    // iterate the file until the end of XML is reached.
    size_t bytesRead = 0;
    i = 0;
    while(reading==1)
    {
        // If I exceeds our buffer size we must abort
        if(i >= bufferSize){
            fprintf(stderr, "Error: XML Parsing failed Tag name or content too long (> %d characters)\n", bufferSize);
            fclose(filePointer);
            free(coo);
            exit(EXIT_FAILURE);
        }

        /* Get the next char from the file */
        c = (char)fgetc(filePointer);

        // Check if we reached the end of the file.
        if(c == EOF){
            // Break out of the loop. This allows for empty files(which may or may not be)
            break;
        }
        // Increment byte counter.
        bytesRead++;

        /*If in a  comment, look for the end of a comment */
        if(in_comment){

            /* Look for an end tag following two (or more) hyphens.
               To support very long comments, we use the minimal amount of buffer we can. 
               If we see a hyphen, store it and increment i (but don't increment i)
               If we see a &gt; check if we have a correct terminating comment
               If we see any other characters, reset i.
            */

            if(c == '-'){
                buffer[i] = c;
                i++;
            } else if(c == '&gt;' &amp;&amp; i >= 2){
                in_comment = 0;
                i = 0;
            } else {
                i = 0;
            }

            /*// If we see the end tag, check the preceding two characters for a close comment, if enough characters have been read for -->
            if(c == '&gt;' &amp;&amp; i >= 2 &amp;&amp; buffer[i-1] == '-' &amp;&amp; buffer[i-2] == '-'){
                in_comment = 0;
                buffer[0] = 0;
                i = 0;
            } else {
                // Otherwise just store it in the buffer so we can keep checking for close tags
                buffer[i] = c;
                i++;
            }*/
        }
        /* If the end of a tag */
        else if(c == '&gt;')
        {
            /* Place 0 at end of buffer to make chars a string */
            buffer[i] = 0;

            if(strcmp(buffer, "graph") == 0){
                reading = 1;
                <!-- in_graph = 1; -->
            }
            if(strcmp(buffer, "/graph") == 0){
                reading = 0;
                <!-- in_graph = 0; -->
            }
            
            if(strcmp(buffer, "vertices") == 0) in_vertices = 1;
            if(strcmp(buffer, "/vertices") == 0) in_vertices = 0;
            if(strcmp(buffer, "vertex") == 0) in_vertex = 1;
            if(strcmp(buffer, "edges") == 0) in_edges = 1;
            if(strcmp(buffer, "/edges") == 0) in_edges = 0;
            if(strcmp(buffer, "edge") == 0) in_edge = 1;


            if(in_vertex){
                <xsl:for-each select="gpu:vertex/xmml:variables/gpu:variable">if(strcmp(buffer, "<xsl:value-of select="xmml:name"/>") == 0) in_<xsl:value-of select="../../xmml:name"/>vertex_<xsl:value-of select="xmml:name"/> = 1;
                if(strcmp(buffer, "/<xsl:value-of select="xmml:name"/>") == 0) in_vertex_<xsl:value-of select="xmml:name"/> = 0;
                </xsl:for-each>
            }
            else if(in_edge){
                <xsl:for-each select="gpu:edge/xmml:variables/gpu:variable">if(strcmp(buffer, "<xsl:value-of select="xmml:name"/>") == 0) in_<xsl:value-of select="../../xmml:name"/>edge_<xsl:value-of select="xmml:name"/> = 1;
                if(strcmp(buffer, "/<xsl:value-of select="xmml:name"/>") == 0) in_edge_<xsl:value-of select="xmml:name"/> = 0;
                </xsl:for-each>
            }


            if(strcmp(buffer, "/vertex") == 0){
                // Check bufferSize
                if(coo-&gt;vertex.count > staticGraph_<xsl:value-of select="$graph_name"/>_vertex_bufferSize){ 
                    printf("Error: Max bufferSize(%i) for graph <xsl:value-of select="$graph_name" /> exceeded whilst reading data\n", staticGraph_<xsl:value-of select="$graph_name"/>_vertex_bufferSize);
                    // Close the file and stop reading. 
                    fclose(filePointer);
                    free(coo);
                    exit(EXIT_FAILURE);
                }

                <xsl:for-each select="gpu:vertex/xmml:variables/gpu:variable">
                <xsl:choose>
                <xsl:when test="xmml:arrayLength">for(unsigned int i = 0; i &lt; <xsl:value-of select="xmml:arrayLength"/>; i++){
                    coo-&gt;vertex.<xsl:value-of select="xmml:name"/>[(i * staticGraph_<xsl:value-of select="$graph_name"/>_vertex_bufferSize) + coo-&gt;vertex.count ] = vertex_<xsl:value-of select="xmml:name"/>[i];
                }
                </xsl:when>
                <xsl:otherwise>coo-&gt;vertex.<xsl:value-of select="xmml:name"/>[coo-&gt;vertex.count] = vertex_<xsl:value-of select="xmml:name"/>;
                </xsl:otherwise>
                </xsl:choose>
                </xsl:for-each>

                /* Reset variables */
                <xsl:for-each select="gpu:vertex/xmml:variables/gpu:variable">
                <xsl:choose>
                <xsl:when test="xmml:arrayLength">for(unsigned int k = 0; k &lt; <xsl:value-of select="xmml:arrayLength"/>; k++){
                    vertex_<xsl:value-of select="xmml:name"/>[k] = <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/><xsl:with-param name="defaultValue" select="xmml:defaultValue" /></xsl:call-template>;
                }
                </xsl:when>
                <xsl:otherwise>vertex_<xsl:value-of select="xmml:name"/> = <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/><xsl:with-param name="defaultValue" select="xmml:defaultValue" /></xsl:call-template>;
                </xsl:otherwise>
                </xsl:choose>
                </xsl:for-each>
                // Reset Flag
                in_vertex = 0;


                // Increment the counter
                coo-&gt;vertex.count++;
            }
            if(strcmp(buffer, "/edge") == 0){
                // Check bufferSize
                if(coo-&gt;edge.count > staticGraph_<xsl:value-of select="$graph_name"/>_edge_bufferSize){ 
                    printf("Error: Max bufferSize(%i) for graph <xsl:value-of select="$graph_name" /> exceeded whilst reading data\n", staticGraph_<xsl:value-of select="$graph_name"/>_edge_bufferSize);
                    // Close the file and stop reading. 
                    fclose(filePointer);
                    free(coo);
                    exit(EXIT_FAILURE);
                }

                <xsl:for-each select="gpu:edge/xmml:variables/gpu:variable">
                <xsl:choose>
                <xsl:when test="xmml:arrayLength">for(unsigned int i = 0; i &lt; <xsl:value-of select="xmml:arrayLength"/>; i++){
                    coo-&gt;edge.<xsl:value-of select="xmml:name"/>[(i * staticGraph_<xsl:value-of select="$graph_name"/>_edge_bufferSize) + coo-&gt;edge.count ] = edge_<xsl:value-of select="xmml:name"/>[i];
                }
                </xsl:when>
                <xsl:otherwise>coo-&gt;edge.<xsl:value-of select="xmml:name"/>[coo-&gt;edge.count] = edge_<xsl:value-of select="xmml:name"/>;
                </xsl:otherwise>
                </xsl:choose>
                </xsl:for-each>

                /* Reset variables */
                <xsl:for-each select="gpu:edge/xmml:variables/gpu:variable">
                <xsl:choose>
                <xsl:when test="xmml:arrayLength">for(unsigned int k = 0; k &lt; <xsl:value-of select="xmml:arrayLength"/>; k++){
                    coo-&gt;edge.<xsl:value-of select="xmml:name"/>[coo-&gt;edge.count * <xsl:value-of select="xmml:arrayLength"/> + k ] = edge_<xsl:value-of select="xmml:name"/>[k];
                }
                </xsl:when>
                <xsl:otherwise>edge_<xsl:value-of select="xmml:name"/> = <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/><xsl:with-param name="defaultValue" select="xmml:defaultValue" /></xsl:call-template>;
                </xsl:otherwise>
                </xsl:choose>
                </xsl:for-each>
                // Reset Flag
                in_edge = 0;


                // Increment the counter
                coo-&gt;edge.count++;
            }            

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


            if(in_vertices &amp;&amp; in_vertex){
                <xsl:for-each select="gpu:vertex/xmml:variables/gpu:variable">if(in_vertex_<xsl:value-of select="xmml:name"/>){
                    <xsl:choose>
                        <xsl:when test="xmml:arrayLength">
                            <xsl:choose>
                                <xsl:when test="contains(xmml:type, '2')">readArrayInputVectorType&lt;<xsl:value-of select="xmml:type"/>, <xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, 2&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, vertex_<xsl:value-of select="xmml:name"/>, <xsl:value-of select="xmml:arrayLength"/>, "vertex", "<xsl:value-of select="xmml:name"/>");</xsl:when>
                                <xsl:when test="contains(xmml:type, '3')">readArrayInputVectorType&lt;<xsl:value-of select="xmml:type"/>, <xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, 3&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, vertex_<xsl:value-of select="xmml:name"/>, <xsl:value-of select="xmml:arrayLength"/>, "vertex", "<xsl:value-of select="xmml:name"/>");</xsl:when>
                                <xsl:when test="contains(xmml:type, '4')">readArrayInputVectorType&lt;<xsl:value-of select="xmml:type"/>, <xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, 4&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, vertex_<xsl:value-of select="xmml:name"/>, <xsl:value-of select="xmml:arrayLength"/>, "vertex", "<xsl:value-of select="xmml:name"/>");</xsl:when>
                                <xsl:otherwise>readArrayInput&lt;<xsl:value-of select="xmml:type"/>&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, vertex_<xsl:value-of select="xmml:name"/>, <xsl:value-of select="xmml:arrayLength"/>, "vertex", "<xsl:value-of select="xmml:name"/>");</xsl:otherwise>
                            </xsl:choose>
                        </xsl:when>
                        <xsl:otherwise>
                            <xsl:choose>
                                <xsl:when test="contains(xmml:type, '2')">readArrayInput&lt;<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, (<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>*)&amp;vertex_<xsl:value-of select="xmml:name"/>, 2, "vertex", "<xsl:value-of select="xmml:name"/>"); </xsl:when>
                                <xsl:when test="contains(xmml:type, '3')">readArrayInput&lt;<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, (<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>*)&amp;vertex_<xsl:value-of select="xmml:name"/>, 3, "vertex", "<xsl:value-of select="xmml:name"/>"); </xsl:when>
                                <xsl:when test="contains(xmml:type, '4')">readArrayInput&lt;<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, (<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>*)&amp;vertex_<xsl:value-of select="xmml:name"/>, 4, "vertex", "<xsl:value-of select="xmml:name"/>");</xsl:when>
                                <xsl:otherwise>vertex_<xsl:value-of select="xmml:name"/> = (<xsl:value-of select="xmml:type"/>) <xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>(buffer);</xsl:otherwise>
                            </xsl:choose>
                        </xsl:otherwise>
                    </xsl:choose>
                }
                </xsl:for-each>

            } else if(in_edges &amp;&amp; in_edge){
                <xsl:for-each select="gpu:edge/xmml:variables/gpu:variable">if(in_edge_<xsl:value-of select="xmml:name"/>){
                    <xsl:choose>
                        <xsl:when test="xmml:arrayLength">
                            <xsl:choose>
                                <xsl:when test="contains(xmml:type, '2')">readArrayInputVectorType&lt;<xsl:value-of select="xmml:type"/>, <xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, 2&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, edge_<xsl:value-of select="xmml:name"/>, <xsl:value-of select="xmml:arrayLength"/>, "edge", "<xsl:value-of select="xmml:name"/>");</xsl:when>
                                <xsl:when test="contains(xmml:type, '3')">readArrayInputVectorType&lt;<xsl:value-of select="xmml:type"/>, <xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, 3&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, edge_<xsl:value-of select="xmml:name"/>, <xsl:value-of select="xmml:arrayLength"/>, "edge", "<xsl:value-of select="xmml:name"/>");</xsl:when>
                                <xsl:when test="contains(xmml:type, '4')">readArrayInputVectorType&lt;<xsl:value-of select="xmml:type"/>, <xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, 4&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, edge_<xsl:value-of select="xmml:name"/>, <xsl:value-of select="xmml:arrayLength"/>, "edge", "<xsl:value-of select="xmml:name"/>");</xsl:when>
                                <xsl:otherwise>readArrayInput&lt;<xsl:value-of select="xmml:type"/>&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, edge_<xsl:value-of select="xmml:name"/>, <xsl:value-of select="xmml:arrayLength"/>, "edge", "<xsl:value-of select="xmml:name"/>");</xsl:otherwise>
                            </xsl:choose>
                        </xsl:when>
                        <xsl:otherwise>
                            <xsl:choose>
                                <xsl:when test="contains(xmml:type, '2')">readArrayInput&lt;<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, (<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>*)&amp;edge_<xsl:value-of select="xmml:name"/>, 2, "edge", "<xsl:value-of select="xmml:name"/>"); </xsl:when>
                                <xsl:when test="contains(xmml:type, '3')">readArrayInput&lt;<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, (<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>*)&amp;edge_<xsl:value-of select="xmml:name"/>, 3, "edge", "<xsl:value-of select="xmml:name"/>"); </xsl:when>
                                <xsl:when test="contains(xmml:type, '4')">readArrayInput&lt;<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, (<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>*)&amp;edge_<xsl:value-of select="xmml:name"/>, 4, "edge", "<xsl:value-of select="xmml:name"/>");</xsl:when>
                                <xsl:otherwise>edge_<xsl:value-of select="xmml:name"/> = (<xsl:value-of select="xmml:type"/>) <xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>(buffer);</xsl:otherwise>
                            </xsl:choose>
                        </xsl:otherwise>
                    </xsl:choose>
                }
                </xsl:for-each>
            }
        
        /* Reset buffer */
            i = 0;
        }
        /* If in tag put read char into buffer */
        else if(in_tag)
        {
            // Check if we are a comment, when we are in a tag and buffer[0:2] == "!--"
            if(i == 2 &amp;&amp; c == '-' &amp;&amp; buffer[1] == '-' &amp;&amp; buffer[0] == '!'){
                in_comment = 1;
                // Reset the buffer and i.
                buffer[0] = 0;
                i = 0;
            }

            // Store the character and increment the counter
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

    // Close the file
    fclose(filePointer);

    // If no bytes were read, raise a warning.
    if(bytesRead == 0){
        fprintf(stdout, "Warning: %s is an empty file\n", file);
        fflush(stdout);
    }

    // If the in_comment flag is still marked, issue a warning.
    if(in_comment){
        fprintf(stdout, "Warning: Un-terminated comment in %s\n", file);
        fflush(stdout);
    }    


    // If the COO has any edges or vertices, construct the CSR representation and free the COO
    if(coo-&gt;vertex.count > 0 &amp;&amp; coo-&gt;edge.count > 0) {
        coo_to_csr_staticGraph_<xsl:value-of select="$graph_name"/>(coo, csr);

        // Check for duplicate edges (undefined behaviour)
        bool has_duplicates = checkForDuplicates_staticGraph_<xsl:value-of select="$graph_name"/>(csr);
        if(has_duplicates){
            printf("FATAL ERROR: Duplicate edge found in staticGraph <xsl:value-of select="$graph_name"/>\n");
            free(coo);
            exit(EXIT_FAILURE);
        }
    }

    // Free the COO representation
    free(coo);
    coo = nullptr;

    // Output message
    fprintf(stdout, "Loaded %u vertices, %u edges\n", csr-&gt;vertex.count, csr-&gt;edge.count);
}
</xsl:if>
</xsl:for-each>


</xsl:template>
</xsl:stylesheet>
