
/*
 * Copyright 2016 University of Sheffield.
 * Authors: Dr Paul Richmond , Dr Mozhgan Kabiri Chimeh
 * Contact: p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
 *          m.kabiri-chimeh@sheffield.ac.uk (http://www.mkchimeh.staff.shef.ac.uk)
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

#include <header.h>
#include <cuda.h>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

#define L_MAX 2.0
#define NL_MAX 0.4
#define BIN_WIDTH 0.1
#define BIN_COUNT L_MAX/BIN_WIDTH

__device__ float d_dt;
void gpuAssert(cudaError_t code, const char *file, int line, bool abort);
/* Error check function for safe CUDA API calling */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__, true); }


__FLAME_GPU_INIT_FUNC__ void initConstants()
{
	float i_agg = 0.875f; // high index aggregation
	set_I_agg(&i_agg);

	int aggno = 1;
	set_aggNo(&aggno);

//	printf("FLAME GPU Init function. aggno=%f, I_agg=%d\n", aggno, i_agg);
}


// The function calculates the time step needed for CASE STUDY 1 - NOT USING IT !
__FLAME_GPU_STEP_FUNC__ void DELTA_T_func(){

	float beta0 = *get_I_agg(); // it's the aggregation Index number
  int agg_No = *get_aggNo();  // it's the aggregation number
  float popnNo;

	gpuErrchk(cudaMemcpyFromSymbol( &popnNo,d_xmachine_memory_crystal_count, sizeof(float))); // Nc-aggNo

	float dt = ((2*agg_No)/(beta0*popnNo*(1-beta0)*(popnNo-agg_No))); // equation 10 and 11

	gpuErrchk(cudaMemcpyToSymbol(d_dt, &dt, sizeof(float)));

	//printf("FLAME GPU Step function. Delta T is %f\n", dt);
}


__FLAME_GPU_EXIT_FUNC__ void hist_func(){


	printf("FLAME GPU Exit function\n");
  FILE *hist_output = fopen("histogram_c1.dat", "w"); // write only - append

	for (int i=0; i<BIN_COUNT; i++){
		int count = count_crystal_default_bin_variable(i);
	//	printf("bin index=%d, count = %d\n", i, count);
    fprintf(hist_output,"%f %d\n", i*BIN_WIDTH, count);
    //output into same format as initial states
  }
  fprintf(hist_output,"\n\n");
  fclose(hist_output);
}


// we may not be needing this function as we can create a an input .xml file to include all possible inputs
// so, no agents will create another agent
__FLAME_GPU_FUNC__ int create_ranks(xmachine_memory_crystal* agent, xmachine_message_internal_coord_list* internal_coord,RNG_rand48* rand48){


	//output crystal internal coordinates

	//generate a new rank for the iteration
	agent->rank = rnd<CONTINUOUS>(rand48);

	//output ranks and length
    	add_internal_coord_message(internal_coord, agent->rank, agent->l);

	return 0;
}


__FLAME_GPU_FUNC__ int aggregate(xmachine_memory_crystal* agent, xmachine_message_internal_coord_list* internal_coord_messages){


	int lower_rank_count=0;	//count of agents that have a lower rank that the current agent
	float closest_lower_rank = 0;
	float closest_lower_length = 0;
	int destroy = 0;

	//iterate through messages
	xmachine_message_internal_coord* current_message = get_first_internal_coord_message(internal_coord_messages);
    	while (current_message)
    	{
		//check the rank to be less than aggNo
		if (current_message->rank < agent->rank){
			lower_rank_count++;
			if (current_message->rank > closest_lower_rank){
				closest_lower_rank = current_message->rank;
				closest_lower_length = current_message->l;
			}

		}

        	current_message = get_next_internal_coord_message(current_message, internal_coord_messages);
    	}

	//calculate if the agent has a low enough rank to aggregate
	if (lower_rank_count < 2*aggNo){
		//even numbered agents will be destroyed
		if(lower_rank_count % 2 == 0){
			destroy = 1;
		}
		//odd numbered agents will be aggregated
		else{
			//apply the formular to find the aggregate length (equ 8)
			agent->l = powf((powf(agent->l, 3)+ powf(closest_lower_length, 3)), (1/3)); //todo: needs to use powf()
		}



	}

	//calculate the bin
	agent->bin = agent->l / BIN_WIDTH;

    return destroy;
}




#endif //_FLAMEGPU_FUNCTIONS
