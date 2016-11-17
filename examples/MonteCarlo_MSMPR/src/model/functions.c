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

#define L_MAX 8.0
#define NL_MAX 10000
#define BIN_WIDTH 0.1
#define BIN_COUNT L_MAX/BIN_WIDTH


__FLAME_GPU_INIT_FUNC__ void initConstants()
{


    float delta_t = 100; //0.1f;
    set_DELTA_T(&delta_t);

    float b0 = 2*pow(10,8);
    set_B0(&b0);

    float g0 = 0.00168f;
    set_G0(&g0);

    float b_temp = 0.22615f;
    set_b(&b_temp);

    float y_temp = 1;
    set_y(&y_temp);

    int tau = 320;
    set_t(&tau);

	float delta_m0 = 0.0;
	set_delta_m0(&delta_m0);

	//we can calculate m0 from eq (ExitNo and NuclNo) as d_m0 is 0
	float m0 = b0*tau;
	set_m0(&m0);

	//aggno is 0 for case study 2
	int agg_no = (int)((get_agent_crystal_default_count()*delta_m0)/m0);
	set_agg_no(&agg_no);

	//population number is constant so calulcate exit number during initialisation
	int exit_no = (int)((get_agent_crystal_default_count() * delta_t) / tau);
	set_exit_no(&exit_no);

	//steady state with fixed agg_no and exit_no so nuc_no is constant
	int nuc_no = exit_no + agg_no;
	set_nuc_no(&nuc_no);

    printf("FLAME GPU Init function. DELTA_T=%f, B0=%f, G0=%f, b=%f, y=%f, m0= %f\n", delta_t, b0, g0, b_temp, y_temp, m0);
	printf("ExitNo=%d\n", exit_no);
}


__FLAME_GPU_EXIT_FUNC__ void hist_func() {

	char output_file[1024];

	printf("FLAME GPU Exit function\n");

	sprintf(output_file, "%s%s", getOutputDir(), "histogram_c2.dat");
	FILE *hist_output = fopen(output_file, "w");

    for (int i=0; i<BIN_COUNT; i++) {
        int count = count_crystal_default_bin_variable(i);
        //printf("bin index=%d, count = %d\n", i, count);
        fprintf(hist_output,"%f %d\n", i*BIN_WIDTH, count);
    }
    fprintf(hist_output,"\n\n");
    fclose(hist_output);
}

// The function generates a new rank for each crystal and outputs crystal internal coordinates
__FLAME_GPU_FUNC__ int create_ranks(xmachine_memory_crystal* agent, xmachine_message_internal_coord_list* internal_coord,RNG_rand48* rand48) {
    //generate a new rank for the iteration
    agent->rank = rnd<CONTINUOUS>(rand48);

    //output ranks and length
    add_internal_coord_message(internal_coord, agent->rank, agent->l);

    return 0;
}


// randomly select crystals and set the size to zero
__FLAME_GPU_FUNC__ int nucleate(xmachine_memory_crystal* agent, xmachine_message_internal_coord_list* internal_coord_messages) {

    int lower_rank_count=0;	//count of agents that have a lower rank that the current agent
    float closest_lower_rank = 0;
	float closest_lower_length = 0;

    //iterate through messages
    xmachine_message_internal_coord* current_message = get_first_internal_coord_message(internal_coord_messages);
    while (current_message)
    {
        //check the rank to be less than aggNo
        if (current_message->rank < agent->rank) {
            lower_rank_count++;
            if (current_message->rank > closest_lower_rank) {
                closest_lower_rank = current_message->rank;
				closest_lower_length = current_message->l;
            }

        }

        current_message = get_next_internal_coord_message(current_message, internal_coord_messages);
    }

	//perform aggregation
	if (lower_rank_count < 2 * agg_no) {
		//even numbered agents will be destroyed
		if (lower_rank_count % 2 == 0) {
			//this will be a nucleated agent
			agent->l = 0;
		}
		//odd numbered agents will be aggregated
		else {
			//apply the formula to find the aggregate length (equ 8)
			agent->l = powf((powf(agent->l, 3) + powf(closest_lower_length, 3)), (1.0 / 3.0));
		}

	}
	else{

		//select next ranks by removing ranks which have aggregated
		lower_rank_count -= 2 * agg_no;

		//calculate if the agent has a low enough rank to nucleate
		if (lower_rank_count < exit_no) {
			//ranks with low enough values will exit
			//as it is a steady state simulation then seting l to 0 is the same as nucleation
			agent->l = 0;
		}
	}

    return 0;
}


__FLAME_GPU_FUNC__ int growth(xmachine_memory_crystal* agent ) {

    float l_new = agent->l + (DELTA_T*G0*(powf(1+y*agent->l,b))); // equation 20, Lj = Lj + Gj*DELTA_T

    agent->l = l_new;

    //calculate the bin
    agent->bin = agent->l / BIN_WIDTH;

    return 0;
}

#endif //_FLAMEGPU_FUNCTIONS
