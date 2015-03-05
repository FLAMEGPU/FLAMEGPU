
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

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>
	

// include header
#include "header.h"

float3 agent_maximum;
float3 agent_minimum;

void readIntArrayInput(char* buffer, int *array, unsigned int expected_items){
    unsigned int i = 0;
    const char s[2] = ",";
    char * token;

    token = strtok(buffer, s);
    while (token != NULL){
        if (i>=expected_items){
            printf("Error: Agent Memeory array has too many items, expected %d!\n", expected_items);
            exit(0);
        }
        
        array[i++] = atoi(token);
        
        token = strtok(NULL, s);
    }
    if (i != expected_items){
        printf("Error: Agent Memeory array has %d items, expected %d!\n", i, expected_items);
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
            printf("Error: Agent Memeory array has too many items, expected %d!\n", expected_items);
            exit(0);
        }
        
        array[i++] = (float)atof(token);
        
        token = strtok(NULL, s);
    }
    if (i != expected_items){
        printf("Error: Agent Memeory array has %d items, expected %d!\n", i, expected_items);
        exit(0);
    }
}

void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_Man_list* h_Mans_unengaged, xmachine_memory_Man_list* d_Mans_unengaged, int h_xmachine_memory_Man_unengaged_count,xmachine_memory_Man_list* h_Mans_engaged, xmachine_memory_Man_list* d_Mans_engaged, int h_xmachine_memory_Man_engaged_count,xmachine_memory_Woman_list* h_Womans_default, xmachine_memory_Woman_list* d_Womans_default, int h_xmachine_memory_Woman_default_count)
{
	cudaError_t cudaStatus;
	
	//Device to host memory transfer
	
	cudaStatus = cudaMemcpy( h_Mans_unengaged, d_Mans_unengaged, sizeof(xmachine_memory_Man_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr,"Error Copying Man Agent unengaged State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_Mans_engaged, d_Mans_engaged, sizeof(xmachine_memory_Man_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr,"Error Copying Man Agent engaged State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_Womans_default, d_Womans_default, sizeof(xmachine_memory_Woman_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr,"Error Copying Woman Agent default State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	
	/* Pointer to file */
	FILE *file;
	char data[100];

	sprintf(data, "%s%i.xml", outputpath, iteration_number);
	//printf("Writing itteration %i data to %s\n", iteration_number, data);
	file = fopen(data, "w");
	fputs("<states>\n<itno>", file);
	sprintf(data, "%i", iteration_number);
	fputs(data, file);
	fputs("</itno>\n", file);
	fputs("<environment>\n" , file);
	fputs("</environment>\n" , file);

	//Write each Man agent to xml
	for (int i=0; i<h_xmachine_memory_Man_unengaged_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>Man</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%i", h_Mans_unengaged->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<round>", file);
        sprintf(data, "%i", h_Mans_unengaged->round[i]);
		fputs(data, file);
		fputs("</round>\n", file);
        
		fputs("<engaged_to>", file);
        sprintf(data, "%i", h_Mans_unengaged->engaged_to[i]);
		fputs(data, file);
		fputs("</engaged_to>\n", file);
        
		fputs("<preferred_woman>", file);
        for (int j=0;j<1024;j++){
            fprintf(file, "%i", h_Mans_unengaged->preferred_woman[(j*xmachine_memory_Man_MAX)+i]);
            if(j!=(1024-1))
                fprintf(file, ",");
        }
		fputs("</preferred_woman>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each Man agent to xml
	for (int i=0; i<h_xmachine_memory_Man_engaged_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>Man</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%i", h_Mans_engaged->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<round>", file);
        sprintf(data, "%i", h_Mans_engaged->round[i]);
		fputs(data, file);
		fputs("</round>\n", file);
        
		fputs("<engaged_to>", file);
        sprintf(data, "%i", h_Mans_engaged->engaged_to[i]);
		fputs(data, file);
		fputs("</engaged_to>\n", file);
        
		fputs("<preferred_woman>", file);
        for (int j=0;j<1024;j++){
            fprintf(file, "%i", h_Mans_engaged->preferred_woman[(j*xmachine_memory_Man_MAX)+i]);
            if(j!=(1024-1))
                fprintf(file, ",");
        }
		fputs("</preferred_woman>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each Woman agent to xml
	for (int i=0; i<h_xmachine_memory_Woman_default_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>Woman</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%i", h_Womans_default->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<current_suitor>", file);
        sprintf(data, "%i", h_Womans_default->current_suitor[i]);
		fputs(data, file);
		fputs("</current_suitor>\n", file);
        
		fputs("<current_suitor_rank>", file);
        sprintf(data, "%i", h_Womans_default->current_suitor_rank[i]);
		fputs(data, file);
		fputs("</current_suitor_rank>\n", file);
        
		fputs("<preferred_man>", file);
        for (int j=0;j<1024;j++){
            fprintf(file, "%i", h_Womans_default->preferred_man[(j*xmachine_memory_Woman_MAX)+i]);
            if(j!=(1024-1))
                fprintf(file, ",");
        }
		fputs("</preferred_man>\n", file);
        
		fputs("</xagent>\n", file);
	}
	
	

	fputs("</states>\n" , file);
	
	/* Close the file */
	fclose(file);
}

void readInitialStates(char* inputpath, xmachine_memory_Man_list* h_Mans, int* h_xmachine_memory_Man_count,xmachine_memory_Woman_list* h_Womans, int* h_xmachine_memory_Woman_count)
{

	int temp = 0;
	int* itno = &temp;

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
	int in_tag, in_itno, in_name;
    int in_Man_id;
    int in_Man_round;
    int in_Man_engaged_to;
    int in_Man_preferred_woman;
    int in_Woman_id;
    int in_Woman_current_suitor;
    int in_Woman_current_suitor_rank;
    int in_Woman_preferred_man;

	/* for continuous agents: set agent count to zero */	
	*h_xmachine_memory_Man_count = 0;	
	*h_xmachine_memory_Woman_count = 0;
	
	/* Variables for initial state data */
	int Man_id;
	int Man_round;
	int Man_engaged_to;
    int Man_preferred_woman[1024];
	int Woman_id;
	int Woman_current_suitor;
	int Woman_current_suitor_rank;
    int Woman_preferred_man[1024];
	
	/* Open config file to read-only */
	if((file = fopen(inputpath, "r"))==NULL)
	{
		printf("error opening initial states\n");
		exit(0);
	}
	
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
	in_name = 0;
	in_Man_id = 0;
	in_Man_round = 0;
	in_Man_engaged_to = 0;
	in_Man_preferred_woman = 0;
	in_Woman_id = 0;
	in_Woman_current_suitor = 0;
	in_Woman_current_suitor_rank = 0;
	in_Woman_preferred_man = 0;
	//set all Man values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_Man_MAX; k++)
	{	
		h_Mans->id[k] = 0;
		h_Mans->round[k] = 0;
		h_Mans->engaged_to[k] = 0;
        for (i=0;i<1024;i++){
            h_Mans->preferred_woman[(i*xmachine_memory_Man_MAX)+k] = 0;
        }
	}
	
	//set all Woman values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_Woman_MAX; k++)
	{	
		h_Womans->id[k] = 0;
		h_Womans->current_suitor[k] = 0;
		h_Womans->current_suitor_rank[k] = 0;
        for (i=0;i<1024;i++){
            h_Womans->preferred_man[(i*xmachine_memory_Woman_MAX)+k] = 0;
        }
	}
	

	/* Default variables for memory */
    Man_id = 0;
    Man_round = 0;
    Man_engaged_to = 0;
    for (i=0;i<1024;i++){
        Man_preferred_woman[i] = 0;
    }
    Woman_id = 0;
    Woman_current_suitor = 0;
    Woman_current_suitor_rank = 0;
    for (i=0;i<1024;i++){
        Woman_preferred_man[i] = 0;
    }

	/* Read file until end of xml */
    i = 0;
	while(reading==1)
	{
		/* Get the next char from the file */
		c = (char)fgetc(file);
		
		/* If the end of a tag */
		if(c == '>')
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
				if(strcmp(agentname, "Man") == 0)
				{		
					if (*h_xmachine_memory_Man_count > xmachine_memory_Man_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent Man exceeded whilst reading data\n", xmachine_memory_Man_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(0);
					}
                    
					h_Mans->id[*h_xmachine_memory_Man_count] = Man_id;
					h_Mans->round[*h_xmachine_memory_Man_count] = Man_round;
					h_Mans->engaged_to[*h_xmachine_memory_Man_count] = Man_engaged_to;
                    for (int k=0;k<1024;k++){
                        h_Mans->preferred_woman[(k*xmachine_memory_Man_MAX)+(*h_xmachine_memory_Man_count)] = Man_preferred_woman[k];    
                    }
					(*h_xmachine_memory_Man_count) ++;	
				}
				else if(strcmp(agentname, "Woman") == 0)
				{		
					if (*h_xmachine_memory_Woman_count > xmachine_memory_Woman_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent Woman exceeded whilst reading data\n", xmachine_memory_Woman_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(0);
					}
                    
					h_Womans->id[*h_xmachine_memory_Woman_count] = Woman_id;
					h_Womans->current_suitor[*h_xmachine_memory_Woman_count] = Woman_current_suitor;
					h_Womans->current_suitor_rank[*h_xmachine_memory_Woman_count] = Woman_current_suitor_rank;
                    for (int k=0;k<1024;k++){
                        h_Womans->preferred_man[(k*xmachine_memory_Woman_MAX)+(*h_xmachine_memory_Woman_count)] = Woman_preferred_man[k];    
                    }
					(*h_xmachine_memory_Woman_count) ++;	
				}
				else
				{
					printf("Warning: agent name undefined - '%s'\n", agentname);
				}
				

				
				/* Reset xagent variables */
                Man_id = 0;
                Man_round = 0;
                Man_engaged_to = 0;
                for (i=0;i<1024;i++){
                    Man_preferred_woman[i] = 0;
                }
                Woman_id = 0;
                Woman_current_suitor = 0;
                Woman_current_suitor_rank = 0;
                for (i=0;i<1024;i++){
                    Woman_preferred_man[i] = 0;
                }

			}
			if(strcmp(buffer, "id") == 0) in_Man_id = 1;
			if(strcmp(buffer, "/id") == 0) in_Man_id = 0;
			if(strcmp(buffer, "round") == 0) in_Man_round = 1;
			if(strcmp(buffer, "/round") == 0) in_Man_round = 0;
			if(strcmp(buffer, "engaged_to") == 0) in_Man_engaged_to = 1;
			if(strcmp(buffer, "/engaged_to") == 0) in_Man_engaged_to = 0;
			if(strcmp(buffer, "preferred_woman") == 0) in_Man_preferred_woman = 1;
			if(strcmp(buffer, "/preferred_woman") == 0) in_Man_preferred_woman = 0;
			if(strcmp(buffer, "id") == 0) in_Woman_id = 1;
			if(strcmp(buffer, "/id") == 0) in_Woman_id = 0;
			if(strcmp(buffer, "current_suitor") == 0) in_Woman_current_suitor = 1;
			if(strcmp(buffer, "/current_suitor") == 0) in_Woman_current_suitor = 0;
			if(strcmp(buffer, "current_suitor_rank") == 0) in_Woman_current_suitor_rank = 1;
			if(strcmp(buffer, "/current_suitor_rank") == 0) in_Woman_current_suitor_rank = 0;
			if(strcmp(buffer, "preferred_man") == 0) in_Woman_preferred_man = 1;
			if(strcmp(buffer, "/preferred_man") == 0) in_Woman_preferred_man = 0;
			
			
			/* End of tag and reset buffer */
			in_tag = 0;
			i = 0;
		}
		/* If start of tag */
		else if(c == '<')
		{
			/* Place /0 at end of buffer to end numbers */
			buffer[i] = 0;
			/* Flag in tag */
			in_tag = 1;
			
			if(in_itno) *itno = atoi(buffer);
			if(in_name) strcpy(agentname, buffer);
			else
			{
				if(in_Man_id){ 
                    Man_id = (int) atoi(buffer);    
                }
				if(in_Man_round){ 
                    Man_round = (int) atoi(buffer);    
                }
				if(in_Man_engaged_to){ 
                    Man_engaged_to = (int) atoi(buffer);    
                }
				if(in_Man_preferred_woman){ 
                    readIntArrayInput(buffer, Man_preferred_woman, 1024);    
                }
				if(in_Woman_id){ 
                    Woman_id = (int) atoi(buffer);    
                }
				if(in_Woman_current_suitor){ 
                    Woman_current_suitor = (int) atoi(buffer);    
                }
				if(in_Woman_current_suitor_rank){ 
                    Woman_current_suitor_rank = (int) atoi(buffer);    
                }
				if(in_Woman_preferred_man){ 
                    readIntArrayInput(buffer, Woman_preferred_man, 1024);    
                }
				
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

float3 getMaximumBounds(){
    return agent_maximum;
}

float3 getMinimumBounds(){
    return agent_minimum;
}

