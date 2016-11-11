w
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

// #define DELTA_T 0.0005f //time step
// #define B0 2*10^8f
// #define G0 0.00168f
// #define b 0.22615f
// #define y 1f
// #define m0 1 // real number of crystals per unit volume

#define L_MAX 2.0
#define NL_MAX 0.4
#define BIN_WIDTH 0.1
#define BIN_COUNT L_MAX/BIN_WIDTH

__device__ int d_nuclNo;
__device__ int d_exitNo;

void gpuAssert(cudaError_t code, const char *file, int line, bool abort);
/* Error check function for safe CUDA API calling */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__, true); }


__FLAME_GPU_INIT_FUNC__ void initConstants()
{
	float delta_t = 0.875f;
	set_DELTA_T(&delta_t);

	float b0 = 2*pow(10,8);
	set_B0(&b0);

	float g0 = 0.00168f;
	set_G0(&g0);

	float b_temp = 0.22615f;
	set_b(&b_temp);

	float y_temp = 1;
	set_y(&y_temp);

	float m0_temp = 1;
	set_m0(&m0_temp);

	int t_temp = 320;
	set_t(&t_temp);

	printf("FLAME GPU Init function. DELTA_T=%f, B0=%f, G0=%f, b=%f, y=%f, m0= %f\n", delta_t, b0, g0, b_temp, y_temp, m0_temp);
}
/* */
__FLAME_GPU_STEP_FUNC__ void exit_func(){

  float d = *get_DELTA_T();
  int t = *get_t();

 //printf("ct = %d, d = %f, t= %d \n",get_agent_crystal_default_count(), d, t);

	int exit_no = (get_agent_crystal_default_count() * d)/ t;

  gpuErrchk(cudaMemcpyToSymbol(d_exitNo, &exit_no, sizeof(int)));

	//printf("FLAME GPU Step function. exitNo is %d\n", exit_no);
}


// The function calculates the nucleation number needed for CASE STUDY 2
__FLAME_GPU_STEP_FUNC__ void nuclNo_func(){

  float d = *get_DELTA_T();
  float b0 = *get_B0();

	int nc_no = b0*d*(pow(10,-7)); // equation 13, nuclNo = (B0*DELTA_T*popnNo)/m0, m0/Nc is the order of 10^7

  gpuErrchk(cudaMemcpyToSymbol(d_nuclNo, &nc_no, sizeof(int)));
	//printf("FLAME GPU Step function. nuclNo is %d\n", nc_no);
}

__FLAME_GPU_EXIT_FUNC__ void hist_func(){

	printf("FLAME GPU Exit function\n");
  FILE *hist_output = fopen("histogram_c2.dat", "w"); // write only - append

	for (int i=0; i<BIN_COUNT; i++){
		int count = count_crystal_default_bin_variable(i);
		//printf("bin index=%d, count = %d\n", i, count);
		fprintf(hist_output,"%f %d\n", i*BIN_WIDTH, count);
		//output into same format as initial states
	}
		fprintf(hist_output,"\n\n");
	fclose(hist_output);
}

// we may not be needing this function as we can create a an input .xml file to include all possible inputs
// so, no agents will create another agent
__FLAME_GPU_FUNC__ int create_ranks(xmachine_memory_crystal* agent, xmachine_message_internal_coord_list* internal_coord,RNG_rand48* rand48){
	//generate a new rank for the iteration
	agent->rank = rnd<CONTINUOUS>(rand48);

	//output ranks and length
    	add_internal_coord_message(internal_coord, agent->rank, agent->l);

	return 0;
}


// randomly select crystals and set the size to zero
__FLAME_GPU_FUNC__ int nucleate(xmachine_memory_crystal* agent, xmachine_message_internal_coord_list* internal_coord_messages){

	int lower_rank_count=0;	//count of agents that have a lower rank that the current agent
	float closest_lower_rank = 0;

	//iterate through messages
	xmachine_message_internal_coord* current_message = get_first_internal_coord_message(internal_coord_messages);
    	while (current_message)
    	{
		//check the rank to be less than aggNo
		if (current_message->rank < agent->rank){
			lower_rank_count++;
			if (current_message->rank > closest_lower_rank){
				closest_lower_rank = current_message->rank;
			}

		}

        	current_message = get_next_internal_coord_message(current_message, internal_coord_messages);
    	}

	//calculate if the agent has a low enough rank to aggregate
	if (lower_rank_count < (d_exitNo)){
		//even numbered agents will be destroyed
		if(lower_rank_count % 2 == 0){
			agent->l = 0;
		}
	}

	//calculate the bin
	agent->bin = agent->l / BIN_WIDTH;
    return 0;
}


//__FLAME_GPU_FUNC__ int growth(xmachine_memory_crystal* agent, xmachine_message_internal_coord_list * ){
__FLAME_GPU_FUNC__ int growth(xmachine_memory_crystal* agent ){

 	int l_new = agent->l + (DELTA_T*G0*(powf((1+y*agent->l),b))); // equation 20, Lj = Lj + Gj*DELTA_T

 	agent->l = l_new;

 		//calculate the bin
	agent->bin = agent->l / BIN_WIDTH;

 	return 0;
}

#endif //_FLAMEGPU_FUNCTIONS
