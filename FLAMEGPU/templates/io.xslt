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
void readArrayInput( T (*parseFunc)(const char*), char* buffer, T *array, unsigned int expected_items){
    unsigned int i = 0;
    const char s[2] = ",";
    char * token;
    char * end_str;

    token = strtok_r(buffer, s, &amp;end_str);
    while (token != NULL){
        if (i>=expected_items){
            printf("Error: Agent memory array has too many items, expected %d!\n", expected_items);
            exit(EXIT_FAILURE);
        }
        
        array[i++] = (T)parseFunc(token);
        
        token = strtok_r(NULL, s, &amp;end_str);
    }
    if (i != expected_items){
        printf("Error: Agent memory array has %d items, expected %d!\n", i, expected_items);
        exit(EXIT_FAILURE);
    }
}

//templated class function to read array inputs from supported types
template &lt;class T, class BASE_T, unsigned int D&gt;
void readArrayInputVectorType( BASE_T (*parseFunc)(const char*), char* buffer, T *array, unsigned int expected_items){
    unsigned int i = 0;
    const char s[2] = "|";
    char * token;
    char * end_str;

    token = strtok_r(buffer, s, &amp;end_str);
    while (token != NULL){
        if (i>=expected_items){
            printf("Error: Agent memory array has too many items, expected %d!\n", expected_items);
            exit(EXIT_FAILURE);
        }
        
        //read vector type as an array
        T vec;
        readArrayInput&lt;BASE_T&gt;(parseFunc, token, (BASE_T*) &amp;vec, D);
        array[i++] = vec;
        
        token = strtok_r(NULL, s, &amp;end_str);
    }
    if (i != expected_items){
        printf("Error: Agent memory array has %d items, expected %d!\n", i, expected_items);
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
            h_<xsl:value-of select="../../xmml:name"/>s-><xsl:value-of select="xmml:name"/>[(i*xmachine_memory_<xsl:value-of select="../../xmml:name"/>_MAX)+k] = <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>;
        }</xsl:when><xsl:otherwise>
		h_<xsl:value-of select="../../xmml:name"/>s-><xsl:value-of select="xmml:name"/>[k] = <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>;</xsl:otherwise></xsl:choose></xsl:for-each>
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
        env_<xsl:value-of select="xmml:name"/>[i] = <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>;
    }
    </xsl:when><xsl:otherwise>env_<xsl:value-of select="xmml:name"/> = <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>;
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
                        <xsl:when test="contains(xmml:type, '2')">readArrayInputVectorType&lt;<xsl:value-of select="xmml:type"/>, <xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, 2&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>, <xsl:value-of select="xmml:arrayLength"/>);    </xsl:when>
                        <xsl:when test="contains(xmml:type, '3')">readArrayInputVectorType&lt;<xsl:value-of select="xmml:type"/>, <xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, 3&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>, <xsl:value-of select="xmml:arrayLength"/>);    </xsl:when>
                        <xsl:when test="contains(xmml:type, '4')">readArrayInputVectorType&lt;<xsl:value-of select="xmml:type"/>, <xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, 4&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>, <xsl:value-of select="xmml:arrayLength"/>);    </xsl:when>
                        <xsl:otherwise>readArrayInput&lt;<xsl:value-of select="xmml:type"/>&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>, <xsl:value-of select="xmml:arrayLength"/>);    </xsl:otherwise>
                        </xsl:choose>        
                      </xsl:when>
                      <xsl:otherwise>
                        <!-- Specialise input reads for vector types -->
                        <xsl:choose>
                        <xsl:when test="contains(xmml:type, '2')">
                          readArrayInput&lt;<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, (<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>*)&amp;<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>, 2); 
                        </xsl:when>
                        <xsl:when test="contains(xmml:type, '3')">
                          readArrayInput&lt;<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, (<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>*)&amp;<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>, 3); 
                        </xsl:when>
                        <xsl:when test="contains(xmml:type, '4')">
                          readArrayInput&lt;<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, (<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>*)&amp;<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>, 4); 
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
                    <xsl:when test="contains(xmml:type, '2')">readArrayInputVectorType&lt;<xsl:value-of select="xmml:type"/>, <xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, 2&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, env_<xsl:value-of select="xmml:name"/>, <xsl:value-of select="xmml:arrayLength"/>);</xsl:when>
                    <xsl:when test="contains(xmml:type, '3')">readArrayInputVectorType&lt;<xsl:value-of select="xmml:type"/>, <xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, 3&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, env_<xsl:value-of select="xmml:name"/>, <xsl:value-of select="xmml:arrayLength"/>);</xsl:when>
                    <xsl:when test="contains(xmml:type, '4')">readArrayInputVectorType&lt;<xsl:value-of select="xmml:type"/>, <xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, 4&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, env_<xsl:value-of select="xmml:name"/>, <xsl:value-of select="xmml:arrayLength"/>);</xsl:when>
                    <xsl:otherwise>readArrayInput&lt;<xsl:value-of select="xmml:type"/>&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, env_<xsl:value-of select="xmml:name"/>, <xsl:value-of select="xmml:arrayLength"/>);</xsl:otherwise>
                    </xsl:choose>
                    set_<xsl:value-of select="xmml:name"/>(env_<xsl:value-of select="xmml:name"/>);
                  </xsl:when>
                  <xsl:otherwise>
                    <!-- Specialise input reads for vector types -->
                    <xsl:choose>
                    <xsl:when test="contains(xmml:type, '2')">
                      readArrayInput&lt;<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, (<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>*)&amp;env_<xsl:value-of select="xmml:name"/>, 2); 
                    </xsl:when>
                    <xsl:when test="contains(xmml:type, '3')">
                      readArrayInput&lt;<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, (<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>*)&amp;env_<xsl:value-of select="xmml:name"/>, 3); 
                    </xsl:when>
                    <xsl:when test="contains(xmml:type, '4')">
                      readArrayInput&lt;<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>&gt;(&amp;<xsl:call-template name="typeParserFunc"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>, buffer, (<xsl:call-template name="vectorBaseType"><xsl:with-param name="type" select="xmml:type"/></xsl:call-template>*)&amp;env_<xsl:value-of select="xmml:name"/>, 4); 
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

</xsl:template>
</xsl:stylesheet>
