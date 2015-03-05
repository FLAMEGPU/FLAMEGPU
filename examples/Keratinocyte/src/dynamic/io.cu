
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

void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_keratinocyte_list* h_keratinocytes_default, xmachine_memory_keratinocyte_list* d_keratinocytes_default, int h_xmachine_memory_keratinocyte_default_count,xmachine_memory_keratinocyte_list* h_keratinocytes_resolve, xmachine_memory_keratinocyte_list* d_keratinocytes_resolve, int h_xmachine_memory_keratinocyte_resolve_count)
{
	cudaError_t cudaStatus;
	
	//Device to host memory transfer
	
	cudaStatus = cudaMemcpy( h_keratinocytes_default, d_keratinocytes_default, sizeof(xmachine_memory_keratinocyte_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr,"Error Copying keratinocyte Agent default State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_keratinocytes_resolve, d_keratinocytes_resolve, sizeof(xmachine_memory_keratinocyte_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr,"Error Copying keratinocyte Agent resolve State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
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

	//Write each keratinocyte agent to xml
	for (int i=0; i<h_xmachine_memory_keratinocyte_default_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>keratinocyte</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%i", h_keratinocytes_default->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<type>", file);
        sprintf(data, "%i", h_keratinocytes_default->type[i]);
		fputs(data, file);
		fputs("</type>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_keratinocytes_default->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_keratinocytes_default->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<z>", file);
        sprintf(data, "%f", h_keratinocytes_default->z[i]);
		fputs(data, file);
		fputs("</z>\n", file);
        
		fputs("<force_x>", file);
        sprintf(data, "%f", h_keratinocytes_default->force_x[i]);
		fputs(data, file);
		fputs("</force_x>\n", file);
        
		fputs("<force_y>", file);
        sprintf(data, "%f", h_keratinocytes_default->force_y[i]);
		fputs(data, file);
		fputs("</force_y>\n", file);
        
		fputs("<force_z>", file);
        sprintf(data, "%f", h_keratinocytes_default->force_z[i]);
		fputs(data, file);
		fputs("</force_z>\n", file);
        
		fputs("<num_xy_bonds>", file);
        sprintf(data, "%i", h_keratinocytes_default->num_xy_bonds[i]);
		fputs(data, file);
		fputs("</num_xy_bonds>\n", file);
        
		fputs("<num_z_bonds>", file);
        sprintf(data, "%i", h_keratinocytes_default->num_z_bonds[i]);
		fputs(data, file);
		fputs("</num_z_bonds>\n", file);
        
		fputs("<num_stem_bonds>", file);
        sprintf(data, "%i", h_keratinocytes_default->num_stem_bonds[i]);
		fputs(data, file);
		fputs("</num_stem_bonds>\n", file);
        
		fputs("<cycle>", file);
        sprintf(data, "%i", h_keratinocytes_default->cycle[i]);
		fputs(data, file);
		fputs("</cycle>\n", file);
        
		fputs("<diff_noise_factor>", file);
        sprintf(data, "%f", h_keratinocytes_default->diff_noise_factor[i]);
		fputs(data, file);
		fputs("</diff_noise_factor>\n", file);
        
		fputs("<dead_ticks>", file);
        sprintf(data, "%i", h_keratinocytes_default->dead_ticks[i]);
		fputs(data, file);
		fputs("</dead_ticks>\n", file);
        
		fputs("<contact_inhibited_ticks>", file);
        sprintf(data, "%i", h_keratinocytes_default->contact_inhibited_ticks[i]);
		fputs(data, file);
		fputs("</contact_inhibited_ticks>\n", file);
        
		fputs("<motility>", file);
        sprintf(data, "%f", h_keratinocytes_default->motility[i]);
		fputs(data, file);
		fputs("</motility>\n", file);
        
		fputs("<dir>", file);
        sprintf(data, "%f", h_keratinocytes_default->dir[i]);
		fputs(data, file);
		fputs("</dir>\n", file);
        
		fputs("<movement>", file);
        sprintf(data, "%f", h_keratinocytes_default->movement[i]);
		fputs(data, file);
		fputs("</movement>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each keratinocyte agent to xml
	for (int i=0; i<h_xmachine_memory_keratinocyte_resolve_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>keratinocyte</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%i", h_keratinocytes_resolve->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<type>", file);
        sprintf(data, "%i", h_keratinocytes_resolve->type[i]);
		fputs(data, file);
		fputs("</type>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_keratinocytes_resolve->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_keratinocytes_resolve->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<z>", file);
        sprintf(data, "%f", h_keratinocytes_resolve->z[i]);
		fputs(data, file);
		fputs("</z>\n", file);
        
		fputs("<force_x>", file);
        sprintf(data, "%f", h_keratinocytes_resolve->force_x[i]);
		fputs(data, file);
		fputs("</force_x>\n", file);
        
		fputs("<force_y>", file);
        sprintf(data, "%f", h_keratinocytes_resolve->force_y[i]);
		fputs(data, file);
		fputs("</force_y>\n", file);
        
		fputs("<force_z>", file);
        sprintf(data, "%f", h_keratinocytes_resolve->force_z[i]);
		fputs(data, file);
		fputs("</force_z>\n", file);
        
		fputs("<num_xy_bonds>", file);
        sprintf(data, "%i", h_keratinocytes_resolve->num_xy_bonds[i]);
		fputs(data, file);
		fputs("</num_xy_bonds>\n", file);
        
		fputs("<num_z_bonds>", file);
        sprintf(data, "%i", h_keratinocytes_resolve->num_z_bonds[i]);
		fputs(data, file);
		fputs("</num_z_bonds>\n", file);
        
		fputs("<num_stem_bonds>", file);
        sprintf(data, "%i", h_keratinocytes_resolve->num_stem_bonds[i]);
		fputs(data, file);
		fputs("</num_stem_bonds>\n", file);
        
		fputs("<cycle>", file);
        sprintf(data, "%i", h_keratinocytes_resolve->cycle[i]);
		fputs(data, file);
		fputs("</cycle>\n", file);
        
		fputs("<diff_noise_factor>", file);
        sprintf(data, "%f", h_keratinocytes_resolve->diff_noise_factor[i]);
		fputs(data, file);
		fputs("</diff_noise_factor>\n", file);
        
		fputs("<dead_ticks>", file);
        sprintf(data, "%i", h_keratinocytes_resolve->dead_ticks[i]);
		fputs(data, file);
		fputs("</dead_ticks>\n", file);
        
		fputs("<contact_inhibited_ticks>", file);
        sprintf(data, "%i", h_keratinocytes_resolve->contact_inhibited_ticks[i]);
		fputs(data, file);
		fputs("</contact_inhibited_ticks>\n", file);
        
		fputs("<motility>", file);
        sprintf(data, "%f", h_keratinocytes_resolve->motility[i]);
		fputs(data, file);
		fputs("</motility>\n", file);
        
		fputs("<dir>", file);
        sprintf(data, "%f", h_keratinocytes_resolve->dir[i]);
		fputs(data, file);
		fputs("</dir>\n", file);
        
		fputs("<movement>", file);
        sprintf(data, "%f", h_keratinocytes_resolve->movement[i]);
		fputs(data, file);
		fputs("</movement>\n", file);
        
		fputs("</xagent>\n", file);
	}
	
	

	fputs("</states>\n" , file);
	
	/* Close the file */
	fclose(file);
}

void readInitialStates(char* inputpath, xmachine_memory_keratinocyte_list* h_keratinocytes, int* h_xmachine_memory_keratinocyte_count)
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
    int in_keratinocyte_id;
    int in_keratinocyte_type;
    int in_keratinocyte_x;
    int in_keratinocyte_y;
    int in_keratinocyte_z;
    int in_keratinocyte_force_x;
    int in_keratinocyte_force_y;
    int in_keratinocyte_force_z;
    int in_keratinocyte_num_xy_bonds;
    int in_keratinocyte_num_z_bonds;
    int in_keratinocyte_num_stem_bonds;
    int in_keratinocyte_cycle;
    int in_keratinocyte_diff_noise_factor;
    int in_keratinocyte_dead_ticks;
    int in_keratinocyte_contact_inhibited_ticks;
    int in_keratinocyte_motility;
    int in_keratinocyte_dir;
    int in_keratinocyte_movement;

	/* for continuous agents: set agent count to zero */	
	*h_xmachine_memory_keratinocyte_count = 0;
	
	/* Variables for initial state data */
	int keratinocyte_id;
	int keratinocyte_type;
	float keratinocyte_x;
	float keratinocyte_y;
	float keratinocyte_z;
	float keratinocyte_force_x;
	float keratinocyte_force_y;
	float keratinocyte_force_z;
	int keratinocyte_num_xy_bonds;
	int keratinocyte_num_z_bonds;
	int keratinocyte_num_stem_bonds;
	int keratinocyte_cycle;
	float keratinocyte_diff_noise_factor;
	int keratinocyte_dead_ticks;
	int keratinocyte_contact_inhibited_ticks;
	float keratinocyte_motility;
	float keratinocyte_dir;
	float keratinocyte_movement;
	
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
	in_keratinocyte_id = 0;
	in_keratinocyte_type = 0;
	in_keratinocyte_x = 0;
	in_keratinocyte_y = 0;
	in_keratinocyte_z = 0;
	in_keratinocyte_force_x = 0;
	in_keratinocyte_force_y = 0;
	in_keratinocyte_force_z = 0;
	in_keratinocyte_num_xy_bonds = 0;
	in_keratinocyte_num_z_bonds = 0;
	in_keratinocyte_num_stem_bonds = 0;
	in_keratinocyte_cycle = 0;
	in_keratinocyte_diff_noise_factor = 0;
	in_keratinocyte_dead_ticks = 0;
	in_keratinocyte_contact_inhibited_ticks = 0;
	in_keratinocyte_motility = 0;
	in_keratinocyte_dir = 0;
	in_keratinocyte_movement = 0;
	//set all keratinocyte values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_keratinocyte_MAX; k++)
	{	
		h_keratinocytes->id[k] = 0;
		h_keratinocytes->type[k] = 0;
		h_keratinocytes->x[k] = 0;
		h_keratinocytes->y[k] = 0;
		h_keratinocytes->z[k] = 0;
		h_keratinocytes->force_x[k] = 0;
		h_keratinocytes->force_y[k] = 0;
		h_keratinocytes->force_z[k] = 0;
		h_keratinocytes->num_xy_bonds[k] = 0;
		h_keratinocytes->num_z_bonds[k] = 0;
		h_keratinocytes->num_stem_bonds[k] = 0;
		h_keratinocytes->cycle[k] = 0;
		h_keratinocytes->diff_noise_factor[k] = 0;
		h_keratinocytes->dead_ticks[k] = 0;
		h_keratinocytes->contact_inhibited_ticks[k] = 0;
		h_keratinocytes->motility[k] = 0;
		h_keratinocytes->dir[k] = 0;
		h_keratinocytes->movement[k] = 0;
	}
	

	/* Default variables for memory */
    keratinocyte_id = 0;
    keratinocyte_type = 0;
    keratinocyte_x = 0;
    keratinocyte_y = 0;
    keratinocyte_z = 0;
    keratinocyte_force_x = 0;
    keratinocyte_force_y = 0;
    keratinocyte_force_z = 0;
    keratinocyte_num_xy_bonds = 0;
    keratinocyte_num_z_bonds = 0;
    keratinocyte_num_stem_bonds = 0;
    keratinocyte_cycle = 0;
    keratinocyte_diff_noise_factor = 0;
    keratinocyte_dead_ticks = 0;
    keratinocyte_contact_inhibited_ticks = 0;
    keratinocyte_motility = 0;
    keratinocyte_dir = 0;
    keratinocyte_movement = 0;

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
				if(strcmp(agentname, "keratinocyte") == 0)
				{		
					if (*h_xmachine_memory_keratinocyte_count > xmachine_memory_keratinocyte_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent keratinocyte exceeded whilst reading data\n", xmachine_memory_keratinocyte_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(0);
					}
                    
					h_keratinocytes->id[*h_xmachine_memory_keratinocyte_count] = keratinocyte_id;
					h_keratinocytes->type[*h_xmachine_memory_keratinocyte_count] = keratinocyte_type;
					h_keratinocytes->x[*h_xmachine_memory_keratinocyte_count] = keratinocyte_x;//Check maximum x value
                    if(agent_maximum.x < keratinocyte_x)
                        agent_maximum.x = (float)keratinocyte_x;
                    //Check minimum x value
                    if(agent_minimum.x > keratinocyte_x)
                        agent_minimum.x = (float)keratinocyte_x;
                    
					h_keratinocytes->y[*h_xmachine_memory_keratinocyte_count] = keratinocyte_y;//Check maximum y value
                    if(agent_maximum.y < keratinocyte_y)
                        agent_maximum.y = (float)keratinocyte_y;
                    //Check minimum y value
                    if(agent_minimum.y > keratinocyte_y)
                        agent_minimum.y = (float)keratinocyte_y;
                    
					h_keratinocytes->z[*h_xmachine_memory_keratinocyte_count] = keratinocyte_z;//Check maximum z value
                    if(agent_maximum.z < keratinocyte_z)
                        agent_maximum.z = (float)keratinocyte_z;
                    //Check minimum z value
                    if(agent_minimum.z > keratinocyte_z)
                        agent_minimum.z = (float)keratinocyte_z;
                    
					h_keratinocytes->force_x[*h_xmachine_memory_keratinocyte_count] = keratinocyte_force_x;
					h_keratinocytes->force_y[*h_xmachine_memory_keratinocyte_count] = keratinocyte_force_y;
					h_keratinocytes->force_z[*h_xmachine_memory_keratinocyte_count] = keratinocyte_force_z;
					h_keratinocytes->num_xy_bonds[*h_xmachine_memory_keratinocyte_count] = keratinocyte_num_xy_bonds;
					h_keratinocytes->num_z_bonds[*h_xmachine_memory_keratinocyte_count] = keratinocyte_num_z_bonds;
					h_keratinocytes->num_stem_bonds[*h_xmachine_memory_keratinocyte_count] = keratinocyte_num_stem_bonds;
					h_keratinocytes->cycle[*h_xmachine_memory_keratinocyte_count] = keratinocyte_cycle;
					h_keratinocytes->diff_noise_factor[*h_xmachine_memory_keratinocyte_count] = keratinocyte_diff_noise_factor;
					h_keratinocytes->dead_ticks[*h_xmachine_memory_keratinocyte_count] = keratinocyte_dead_ticks;
					h_keratinocytes->contact_inhibited_ticks[*h_xmachine_memory_keratinocyte_count] = keratinocyte_contact_inhibited_ticks;
					h_keratinocytes->motility[*h_xmachine_memory_keratinocyte_count] = keratinocyte_motility;
					h_keratinocytes->dir[*h_xmachine_memory_keratinocyte_count] = keratinocyte_dir;
					h_keratinocytes->movement[*h_xmachine_memory_keratinocyte_count] = keratinocyte_movement;
					(*h_xmachine_memory_keratinocyte_count) ++;	
				}
				else
				{
					printf("Warning: agent name undefined - '%s'\n", agentname);
				}
				

				
				/* Reset xagent variables */
                keratinocyte_id = 0;
                keratinocyte_type = 0;
                keratinocyte_x = 0;
                keratinocyte_y = 0;
                keratinocyte_z = 0;
                keratinocyte_force_x = 0;
                keratinocyte_force_y = 0;
                keratinocyte_force_z = 0;
                keratinocyte_num_xy_bonds = 0;
                keratinocyte_num_z_bonds = 0;
                keratinocyte_num_stem_bonds = 0;
                keratinocyte_cycle = 0;
                keratinocyte_diff_noise_factor = 0;
                keratinocyte_dead_ticks = 0;
                keratinocyte_contact_inhibited_ticks = 0;
                keratinocyte_motility = 0;
                keratinocyte_dir = 0;
                keratinocyte_movement = 0;

			}
			if(strcmp(buffer, "id") == 0) in_keratinocyte_id = 1;
			if(strcmp(buffer, "/id") == 0) in_keratinocyte_id = 0;
			if(strcmp(buffer, "type") == 0) in_keratinocyte_type = 1;
			if(strcmp(buffer, "/type") == 0) in_keratinocyte_type = 0;
			if(strcmp(buffer, "x") == 0) in_keratinocyte_x = 1;
			if(strcmp(buffer, "/x") == 0) in_keratinocyte_x = 0;
			if(strcmp(buffer, "y") == 0) in_keratinocyte_y = 1;
			if(strcmp(buffer, "/y") == 0) in_keratinocyte_y = 0;
			if(strcmp(buffer, "z") == 0) in_keratinocyte_z = 1;
			if(strcmp(buffer, "/z") == 0) in_keratinocyte_z = 0;
			if(strcmp(buffer, "force_x") == 0) in_keratinocyte_force_x = 1;
			if(strcmp(buffer, "/force_x") == 0) in_keratinocyte_force_x = 0;
			if(strcmp(buffer, "force_y") == 0) in_keratinocyte_force_y = 1;
			if(strcmp(buffer, "/force_y") == 0) in_keratinocyte_force_y = 0;
			if(strcmp(buffer, "force_z") == 0) in_keratinocyte_force_z = 1;
			if(strcmp(buffer, "/force_z") == 0) in_keratinocyte_force_z = 0;
			if(strcmp(buffer, "num_xy_bonds") == 0) in_keratinocyte_num_xy_bonds = 1;
			if(strcmp(buffer, "/num_xy_bonds") == 0) in_keratinocyte_num_xy_bonds = 0;
			if(strcmp(buffer, "num_z_bonds") == 0) in_keratinocyte_num_z_bonds = 1;
			if(strcmp(buffer, "/num_z_bonds") == 0) in_keratinocyte_num_z_bonds = 0;
			if(strcmp(buffer, "num_stem_bonds") == 0) in_keratinocyte_num_stem_bonds = 1;
			if(strcmp(buffer, "/num_stem_bonds") == 0) in_keratinocyte_num_stem_bonds = 0;
			if(strcmp(buffer, "cycle") == 0) in_keratinocyte_cycle = 1;
			if(strcmp(buffer, "/cycle") == 0) in_keratinocyte_cycle = 0;
			if(strcmp(buffer, "diff_noise_factor") == 0) in_keratinocyte_diff_noise_factor = 1;
			if(strcmp(buffer, "/diff_noise_factor") == 0) in_keratinocyte_diff_noise_factor = 0;
			if(strcmp(buffer, "dead_ticks") == 0) in_keratinocyte_dead_ticks = 1;
			if(strcmp(buffer, "/dead_ticks") == 0) in_keratinocyte_dead_ticks = 0;
			if(strcmp(buffer, "contact_inhibited_ticks") == 0) in_keratinocyte_contact_inhibited_ticks = 1;
			if(strcmp(buffer, "/contact_inhibited_ticks") == 0) in_keratinocyte_contact_inhibited_ticks = 0;
			if(strcmp(buffer, "motility") == 0) in_keratinocyte_motility = 1;
			if(strcmp(buffer, "/motility") == 0) in_keratinocyte_motility = 0;
			if(strcmp(buffer, "dir") == 0) in_keratinocyte_dir = 1;
			if(strcmp(buffer, "/dir") == 0) in_keratinocyte_dir = 0;
			if(strcmp(buffer, "movement") == 0) in_keratinocyte_movement = 1;
			if(strcmp(buffer, "/movement") == 0) in_keratinocyte_movement = 0;
			
			
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
				if(in_keratinocyte_id){ 
                    keratinocyte_id = (int) atoi(buffer);    
                }
				if(in_keratinocyte_type){ 
                    keratinocyte_type = (int) atoi(buffer);    
                }
				if(in_keratinocyte_x){ 
                    keratinocyte_x = (float) atof(buffer);    
                }
				if(in_keratinocyte_y){ 
                    keratinocyte_y = (float) atof(buffer);    
                }
				if(in_keratinocyte_z){ 
                    keratinocyte_z = (float) atof(buffer);    
                }
				if(in_keratinocyte_force_x){ 
                    keratinocyte_force_x = (float) atof(buffer);    
                }
				if(in_keratinocyte_force_y){ 
                    keratinocyte_force_y = (float) atof(buffer);    
                }
				if(in_keratinocyte_force_z){ 
                    keratinocyte_force_z = (float) atof(buffer);    
                }
				if(in_keratinocyte_num_xy_bonds){ 
                    keratinocyte_num_xy_bonds = (int) atoi(buffer);    
                }
				if(in_keratinocyte_num_z_bonds){ 
                    keratinocyte_num_z_bonds = (int) atoi(buffer);    
                }
				if(in_keratinocyte_num_stem_bonds){ 
                    keratinocyte_num_stem_bonds = (int) atoi(buffer);    
                }
				if(in_keratinocyte_cycle){ 
                    keratinocyte_cycle = (int) atoi(buffer);    
                }
				if(in_keratinocyte_diff_noise_factor){ 
                    keratinocyte_diff_noise_factor = (float) atof(buffer);    
                }
				if(in_keratinocyte_dead_ticks){ 
                    keratinocyte_dead_ticks = (int) atoi(buffer);    
                }
				if(in_keratinocyte_contact_inhibited_ticks){ 
                    keratinocyte_contact_inhibited_ticks = (int) atoi(buffer);    
                }
				if(in_keratinocyte_motility){ 
                    keratinocyte_motility = (float) atof(buffer);    
                }
				if(in_keratinocyte_dir){ 
                    keratinocyte_dir = (float) atof(buffer);    
                }
				if(in_keratinocyte_movement){ 
                    keratinocyte_movement = (float) atof(buffer);    
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

