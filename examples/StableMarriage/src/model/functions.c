
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

#include <header.h>



__FLAME_GPU_FUNC__ int make_proposals(xmachine_memory_Man* agent, xmachine_message_proposal_list* proposal_messages){

    int woman;
	int round;

	round = agent->round;

	//get next preferable woman
	woman = get_Man_agent_array_value<int>(agent->preferred_woman, round);

	//make a proposal
    add_proposal_message(proposal_messages, agent->id, woman);

	agent->round++;

	return 0;
}


__FLAME_GPU_FUNC__ int check_proposals(xmachine_memory_Woman* agent, xmachine_message_proposal_list* proposal_messages){

	//iterate proposals to find the best suitor so far for this round of proposals
    xmachine_message_proposal* current_message = get_first_proposal_message(proposal_messages);
    while (current_message)
    {
		//if proposal is for the woman
		if (current_message->woman == agent->id){
			//if proposal desirabiloty is higher than current
			int rank = get_Woman_agent_array_value<int>(agent->preferred_man, current_message->id);
			if ((agent->current_suitor_rank == -1)||(rank < agent->current_suitor_rank)){
				agent->current_suitor = current_message->id;
				agent->current_suitor_rank = rank;
			}
		}
        
        current_message = get_next_proposal_message(current_message, proposal_messages);
    }
  
    return 0;
}


__FLAME_GPU_FUNC__ int notify_suitors(xmachine_memory_Woman* agent, xmachine_message_notification_list* notification_messages){

	//function is only called if the woman has been proposed to
    add_notification_message(notification_messages, agent->id, agent->current_suitor);

    return 0;
}


__FLAME_GPU_FUNC__ int check_notifications(xmachine_memory_Man* agent, xmachine_message_notification_list* notification_messages){

    //not engaged
	agent->engaged_to = -1;

    xmachine_message_notification* current_message = get_first_notification_message(notification_messages);
    while (current_message)
    {
		//check any proposal notifications
		if (current_message->suitor == agent->id){
			agent->engaged_to = current_message->id;
		}
        
        current_message = get_next_notification_message(current_message, notification_messages);
    }
   
     
    
  
    return 0;
}

__FLAME_GPU_FUNC__ int check_resolved(xmachine_memory_Man* agent){

    //dummy function
	if (agent->id == 1)
		printf("We are all married!\n");
  
    return 0;
}
  


#endif //_FLAMEGPU_FUNCTIONS
