
#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

#include "header.h"
#include "cutil_math.h"
//#include "math.h"


#define PI 3.1415f
#define RADIANS(x) (PI / 180.0f) * x

//
#if _DEBUG
#define DEBUG_LOG( x, ... ) printf( x, __VA_ARGS__ );
#define DEBUG_LOG_POSITION( name, position ) printf( name ": %f, %f, %f\r\n", position.x, position.y, position.z )
#else
#define DEBUG_LOG( x, ... )
#define DEBUG_LOG_POSITION( name, position )
#endif

#define epsilon 1.0e-3
#define emsmall 1.0e-12
#define GRAVITY 9.80665 
#define GLOBAL_MANNING 0.018500
#define CFL 0.5
#define TOL_H 0.0001
#define TIMESTEP 0.01

#define BIG_NUMBER 800000             // Used in WetDryMessage to skip extra calculations MS05Sep2017


inline __device__ double3 hll_x(double h_L, double h_R, double qx_L, double qx_R, double qy_L, double qy_R);
inline __device__ double3 hll_y(double h_L, double h_R, double qx_L, double qx_R, double qy_L, double qy_R);

inline __device__ double3 Flux_F(double hh, double qx, double qy);
inline __device__ double3 Flux_G(double hh, double qx, double qy);

inline __device__ double3 F_SWE(double hh, double qx, double qy);
inline __device__ double3 G_SWE(double hh, double qx, double qy);


inline __device__ double3 Sb(double hx_mod, double hy_mod, double zprime_x, double zprime_y);

enum ECellDirection { NORTH = 1, EAST = 2, SOUTH = 3, WEST = 4 };

struct __align__(16) AgentFlowData
{
	double z0;
	double h;
	double et;
	double qx;
	double qy;
};


struct __align__(16) LFVResult
{
	__device__ LFVResult(double _h_face, double _et_face, double2 _qFace)
	{
		h_face = _h_face;
		et_face = _et_face;
		qFace = _qFace;
	}

	__device__ LFVResult()
	{
		h_face = 0.0;
		et_face = 0.0;
		qFace = make_double2(0.0, 0.0);

	}

	double h_face;
	double et_face;
	double2 qFace;

};


inline __device__ AgentFlowData GetFlowDataFromAgent(xmachine_memory_FloodCell* agent)
{
	AgentFlowData result;

	result.z0 = agent->z0;
	result.h  = agent->h;
	result.et = agent->z0 + agent->h; // added by MS27Sep2017
	result.qx = agent->qx;
	result.qy = agent->qy;

	return result;

}


// NB. It is assumed that the ghost cell is at the same level as the present cell.
//** Assign the the same topography data for the ghost cell **!
// Boundary condition in ghost cells // 
inline __device__ void centbound(xmachine_memory_FloodCell* agent, const AgentFlowData& FlowData, AgentFlowData& centBoundData)
{

	//Default is a reflective boundary
	centBoundData.z0 = FlowData.z0;

	centBoundData.h  = FlowData.h;

	centBoundData.et = FlowData.et; // added by MS27Sep2017 // I know I do not need it. for now I keep it.

	centBoundData.qx = -FlowData.qx;

	centBoundData.qy = -FlowData.qy;

}

//
//// Conditional function to check whether the Height is less than TOL_H or not
////inline __device__ bool IsDry(double waterHeight)
////{
////	return waterHeight <= TOL_H;
////}

// This function should be called when minh_loc is greater than TOL_H
inline __device__ double2 friction_2D(double dt_loc, double h_loc, double qx_loc, double qy_loc)
{
	//This function add the friction contribution to wet cells.
	//This fucntion should not be called when: minh_loc.LE.TOL_H << check if the function has the criterion to be taken into accont MS05Sep2017

	double2 result;

	result.x = 0.0;
	result.y = 0.0;


	// Local velocities    
	double u_loc = qx_loc / h_loc;
	double v_loc = qy_loc / h_loc;

	// Friction forces are incative as the flow is motionless.
	if ((fabs(u_loc) <= emsmall)
		&& (fabs(v_loc) <= emsmall)
		)
	{
		result.x = qx_loc;
		result.y = qy_loc;
	}
	else
	{
		// The is motional. The FRICTIONS CONTRUBUTION HAS TO BE ADDED SO THAT IT DOESN'T REVERSE THE FLOW.

		double Cf = GRAVITY * pow(GLOBAL_MANNING, 2.0) / pow(h_loc, 1.0 / 3.0);

		double expULoc = pow(u_loc, 2.0);
		double expVLoc = pow(v_loc, 2.0);

		double Sfx = -Cf * u_loc * sqrt(expULoc + expVLoc);
		double Sfy = -Cf * v_loc * sqrt(expULoc + expVLoc);

		double DDx = 1.0 + dt_loc * (Cf / h_loc * (2.0 * expULoc + expVLoc) / sqrt(expULoc + expVLoc));
		double DDy = 1.0 + dt_loc * (Cf / h_loc * (expULoc + 2.0 * expVLoc) / sqrt(expULoc + expVLoc));

		result.x = qx_loc + (dt_loc * (Sfx / DDx));
		result.y = qy_loc + (dt_loc * (Sfy / DDy));

	}

	return result;
}

inline __device__ void Friction_Implicit(xmachine_memory_FloodCell* agent, double dt)
{
	if (GLOBAL_MANNING > 0.0)
	{
		AgentFlowData FlowData = GetFlowDataFromAgent(agent);

		if (FlowData.h <= TOL_H)
		{
			return;
		}

		double2 frict_Q = friction_2D(dt, FlowData.h, FlowData.qx, FlowData.qy);

		agent->qx = frict_Q.x;
		agent->qy = frict_Q.y;


	}

}


__FLAME_GPU_FUNC__ int PrepareWetDry(xmachine_memory_FloodCell* agent, xmachine_message_WetDryMessage_list* WerDryMessage_messages)
{
	if (agent->inDomain==1)
	{

		AgentFlowData FlowData = GetFlowDataFromAgent(agent);
		// added by MS > 
		double hp = FlowData.h; // = agent->h ; MS02OCT2017

		agent->minh_loc = hp;


		add_WetDryMessage_message<DISCRETE_2D>(WerDryMessage_messages, 1, agent->x, agent->y, agent->minh_loc);
	}
	else
	{
		add_WetDryMessage_message<DISCRETE_2D>(WerDryMessage_messages, 0, agent->x, agent->y, BIG_NUMBER);
	}

	return 0;

}

__FLAME_GPU_FUNC__ int ProcessWetDryMessage(xmachine_memory_FloodCell* agent, xmachine_message_WetDryMessage_list* WetDryMessage_messages)
{
	if (agent->inDomain==1)
	{

		//looking up neighbours values for wet/dry tracking "THIS NEEDS TO BE ANALYSED carefully MS05Sep2017"
		xmachine_message_WetDryMessage* msg = get_first_WetDryMessage_message<DISCRETE_2D>(WetDryMessage_messages, agent->x, agent->y);

		double maxHeight = agent->minh_loc;

		while (msg)
		{
			if (msg->inDomain==1)
			{
				agent->minh_loc = min(agent->minh_loc, msg->min_hloc);
			}

			if (msg->min_hloc > maxHeight)
			{
				maxHeight = msg->min_hloc;
			}

			msg = get_next_WetDryMessage_message<DISCRETE_2D>(msg, WetDryMessage_messages);
		}



		if (maxHeight > TOL_H) // if the Height of water is not less than TOL_H => the friction is needed to be taken into account MS05Sep2017
		{
			Friction_Implicit(agent, TIMESTEP); // TIMESTEP has been defined in agents' initial condition MS05Sep2017
		}
		else
		{
			//agent->qx = 0.0; //added by MS28Sep2017
			//agent->qy = 0.0; //added by MS28Sep2017
			//need to go high, so that it won't affect min calculation when it is tested again . Needed to be tested MS05Sep2017 which is now temporary. needs to be corrected somehow
			agent->minh_loc = BIG_NUMBER;
		}

	}

	return 0;
}

__FLAME_GPU_FUNC__ int PrepareLFV(xmachine_memory_FloodCell* agent, xmachine_message_LFVMessage_list* LFVMessage_messages)
{
	if (agent->inDomain==1)
	{

		if (agent->minh_loc > TOL_H)
		{

		AgentFlowData FlowData = GetFlowDataFromAgent(agent);


		//broadcast the to the surrounding cells
			add_LFVMessage_message<DISCRETE_2D>(LFVMessage_messages, 1, agent->x, agent->y, FlowData.z0, FlowData.h, FlowData.qx, FlowData.qy);
		//	add_LFVMessage_message<DISCRETE_2D>(LFVMessage_messages, 1, agent->x, agent->y, agent->z0, agent->h, agent->qx, agent->qy);
	}
		else
		{
			add_LFVMessage_message<DISCRETE_2D>(LFVMessage_messages, 0, agent->x, agent->y, 0.0, 0.0, 0.0, 0.0);
		}
	}
	return 0;
}


inline __device__ void FlowDataFromLFVMessage(AgentFlowData& FlowData, xmachine_message_LFVMessage* msg)
{
	FlowData.z0 = msg->z0;

	FlowData.h  = msg->h;

	FlowData.et = msg->z0 + msg->h; // added by MS27Sep2017

	FlowData.qx = msg->qx;

	FlowData.qy = msg->qy;
}


inline __device__ LFVResult LFV( const AgentFlowData& FlowData)
{

	LFVResult result;


	result.h_face  = FlowData.h;

	result.et_face = FlowData.et;

	result.qFace.x = FlowData.qx;

	result.qFace.y = FlowData.qy;

	return result;
}

__FLAME_GPU_FUNC__ int ProcessLFVMessage(xmachine_memory_FloodCell* agent, xmachine_message_LFVMessage_list* LFVMessage_messages)
{

	if (agent->inDomain == 1
		&& agent->minh_loc > TOL_H)
	{
		AgentFlowData FlowData = GetFlowDataFromAgent(agent);

		//// declaring new variables "MS05Sep2017"
		//AgentFlowData eastNeighbourFlowData;
		//AgentFlowData westNeighbourFlowData;
		//AgentFlowData northNeighbourFlowData;
		//AgentFlowData southNeighbourFlowData;

		////initialise values as transmissive cells  - Boundary condition MS05Sep2017
		//centbound(agent, FlowData, eastNeighbourFlowData);
		//centbound(agent, FlowData, westNeighbourFlowData);
		//centbound(agent, FlowData, northNeighbourFlowData);
		//centbound(agent, FlowData, southNeighbourFlowData);

		//check for neighbour flow data
		xmachine_message_LFVMessage* msg = get_first_LFVMessage_message<DISCRETE_2D>(LFVMessage_messages, agent->x, agent->y);

		while (msg)
		{
			if (msg->inDomain==1)
			{

				LFVResult faceLFV = LFV(FlowData);

				//EAST FACE
				agent->etFace_E = faceLFV.et_face;
				agent->hFace_E  = faceLFV.h_face;
				agent->qxFace_E = faceLFV.qFace.x;
				agent->qyFace_E = faceLFV.qFace.y;

				//WEST FACE
				agent->etFace_W = faceLFV.et_face;
				agent->hFace_W = faceLFV.h_face;
				agent->qxFace_W = faceLFV.qFace.x;
				agent->qyFace_W = faceLFV.qFace.y;

				//NORTH FACE
				agent->etFace_N = faceLFV.et_face;
				agent->hFace_N = faceLFV.h_face;
				agent->qxFace_N = faceLFV.qFace.x;
				agent->qyFace_N = faceLFV.qFace.y;

				//SOUTH FACE
				agent->etFace_S = faceLFV.et_face;
				agent->hFace_S = faceLFV.h_face;
				agent->qxFace_S = faceLFV.qFace.x;
				agent->qyFace_S = faceLFV.qFace.y;

				//if (agent->x - 1 == msg->x
				//	&&   agent->y == msg->y)
				//{
				//	FlowDataFromLFVMessage(westNeighbourFlowData, msg);
				//	/*westNeighbourFlowData.h= msg->h;
				//	westNeighbourFlowData.et = msg->h + msg->z0;
				//	westNeighbourFlowData.qx = msg->qx;
				//	westNeighbourFlowData.qy = msg->qy;*/
				//}
				//else
				//if (agent->x + 1 == msg->x
				//	&&   agent->y == msg->y)
				//{
				//	FlowDataFromLFVMessage(eastNeighbourFlowData, msg);
				//	/*eastNeighbourFlowData.h = msg->h;
				//	eastNeighbourFlowData.et = msg->h + msg->z0;
				//	eastNeighbourFlowData.qx = msg->qx;
				//	eastNeighbourFlowData.qy = msg->qy;*/
				//}
				//else
				//if (agent->x == msg->x
				//	&&   agent->y + 1 == msg->y)
				//{
				//	FlowDataFromLFVMessage(northNeighbourFlowData, msg);
				//	/*northNeighbourFlowData.h = msg->h;
				//	northNeighbourFlowData.et = msg->h + msg->z0;
				//	northNeighbourFlowData.qx = msg->qx;
				//	northNeighbourFlowData.qy = msg->qy;*/
				//}
				//else
				//if (agent->x == msg->x
				//	&&   agent->y - 1 == msg->y)
				//{
				//	FlowDataFromLFVMessage(southNeighbourFlowData, msg);
				//	/*southNeighbourFlowData.h  = msg->h;
				//	southNeighbourFlowData.et = msg->h + msg->z0;
				//	southNeighbourFlowData.qx = msg->qx;
				//	southNeighbourFlowData.qy = msg->qy;*/
				//}
			}

			msg = get_next_LFVMessage_message<DISCRETE_2D>(msg, LFVMessage_messages);
		}
	}

	return 0;
}


__FLAME_GPU_FUNC__ int PrepareSpaceOperator(xmachine_memory_FloodCell* agent, xmachine_message_SpaceOperatorMessage_list* SpaceOperatorMessage_messages)
{
	if (agent->inDomain == 1
		&& agent->minh_loc > TOL_H)
	{
		//broadcast internal LFV values to surrounding cells
		add_SpaceOperatorMessage_message<DISCRETE_2D>(SpaceOperatorMessage_messages,
															1,
															agent->x, agent->y,
															agent->hFace_E, agent->etFace_E, agent->qxFace_E, agent->qyFace_E,
															agent->hFace_W, agent->etFace_W, agent->qxFace_W, agent->qyFace_W,
															agent->hFace_N, agent->etFace_N, agent->qxFace_N, agent->qyFace_N,
															agent->hFace_S, agent->etFace_S, agent->qxFace_S, agent->qyFace_S
															);
	}
	else
	{
		//broadcast internal LFV values to surrounding cells
		add_SpaceOperatorMessage_message<DISCRETE_2D>(SpaceOperatorMessage_messages,
			0,
			agent->x, agent->y,
			0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0
			);
	}

	return 0;
}

inline __device__ void WD(double h_L,
	double h_R,
	double et_L,
	double et_R,
	double qx_L,
	double qx_R,
	double qy_L,
	double qy_R,
	ECellDirection ndir,
	double& z_LR,
	double& h_L_star,
	double& h_R_star,
	double& qx_L_star,
	double& qx_R_star,
	double& qy_L_star,
	double& qy_R_star
	)

{
	// This function provide a non-negative reconstruction of the Riemann-states.

	double z_L = et_L - h_L;
	double z_R = et_R - h_R;

	double u_L = 0.0;
	double v_L = 0.0;
	double u_R = 0.0;
	double v_R = 0.0;

	if (h_L > TOL_H)
	{
		u_L = qx_L / h_L;
		v_L = qy_L / h_L;
	}
	else
	{
		u_L = 0.0;
		v_L = 0.0;
	}


	if (h_R > TOL_H)
	{
		u_R = qx_R / h_R;
		v_R = qy_R / h_R;
	}
	else
	{
		u_R = 0.0;
		v_R = 0.0;
	}

	z_LR = max(z_L, z_R);

	double delta;

	switch (ndir)
	{
	case NORTH:
	case EAST:
	{
				 delta = max(0.0, -(et_L - z_LR));
	}
		break;

	case WEST:
	case SOUTH:
	{
				  delta = max(0.0, -(et_R - z_LR));
	}
		break;

	}

	h_L_star = max(0.0, et_L - z_LR);
	double et_L_star = h_L_star + z_LR;
	qx_L_star = h_L_star * u_L;
	qy_L_star = h_L_star * v_L;

	h_R_star = max(0.0, et_R - z_LR);
	double et_R_star = h_R_star + z_LR;
	qx_R_star = h_R_star * u_R;
	qy_R_star = h_R_star * v_R;

	if (delta > 0.0)
	{
		z_LR = z_LR - delta;
		et_L_star = et_L_star - delta;
		et_R_star = et_R_star - delta;
	}
	else
	{
		z_LR = z_LR;
		et_L_star = et_L_star;
		et_R_star = et_R_star;
	}


	h_L_star = et_L_star - z_LR;
	h_R_star = et_R_star - z_LR;

	// Commented by MS02Oct2017 - changed to 'C' project
	/*if (delta > 0.0)
	{
		z_LR = z_LR - delta;
		et_L_star = et_L_star - delta;
		et_R_star = et_R_star - delta;
	}

	h_L_star = et_L_star - z_LR; 
	h_R_star = et_R_star - z_LR; */
}


inline __device__ double3 hll_x(double h_L, double h_R, double qx_L, double qx_R, double qy_L, double qy_R)
{
	double3 F_face = make_double3(0.0, 0.0, 0.0);

	double u_L = 0.0;
	double v_L = 0.0;
	double u_R = 0.0;
	double v_R = 0.0;

	if ((h_L <= TOL_H) && (h_R <= TOL_H))
	{
		F_face.x = 0.0;
		F_face.y = 0.0;
		F_face.z = 0.0;

		return F_face;
	}
	else
	{

		if (h_L <= TOL_H)
		{
			h_L = 0.0;
			u_L = 0.0;
			v_L = 0.0;
		}
		else
		{

			u_L = qx_L / h_L;
			v_L = qy_L / h_L;
		}


		if (h_R <= TOL_H)
		{
			h_R = 0.0;
			u_R = 0.0;
			v_R = 0.0;
		}
		else
		{
			u_R = qx_R / h_R;
			v_R = qy_R / h_R;
		}

		double a_L = sqrt(GRAVITY * h_L);
		double a_R = sqrt(GRAVITY * h_R);

		double h_star = pow(((a_L + a_R) / 2.0 + (u_L - u_R) / 4.0), 2) / GRAVITY;
		double u_star = (u_L + u_R) / 2.0 + a_L - a_R;
		double a_star = sqrt(GRAVITY * h_star);

		double s_L, s_R;

		if (h_L <= TOL_H)
		{
			s_L = u_R - (2.0 * a_R);
		}
		else
		{
			s_L = min(u_L - a_L, u_star - a_star);
		}



		if (h_R <= TOL_H)
		{
			s_R = u_L + (2.0 * a_L);
		}
		else
		{
			s_R = max(u_R + a_R, u_star + a_star);
		}

		double s_M = ((s_L * h_R * (u_R - s_R)) - (s_R * h_L * (u_L - s_L))) / (h_R * (u_R - s_R) - (h_L * (u_L - s_L)));

		double3 F_L, F_R;

		//FSWE3 F_L = F_SWE((double)h_L, (double)qx_L, (double)qy_L);
		 F_L = F_SWE(h_L, qx_L, qy_L);

		//FSWE3 F_R = F_SWE((double)h_R, (double)qx_R, (double)qy_R);
		 F_R = F_SWE(h_R, qx_R, qy_R);


		if (s_L >= 0.0)
		{
			F_face.x = F_L.x;
			F_face.y = F_L.y;
			F_face.z = F_L.z;

			//return F_L; 
		}

		else if ((s_L < 0.0) && s_R >= 0.0)

		{

			double F1_M = ((s_R * F_L.x) - (s_L * F_R.x) + s_L * s_R * ( h_R - h_L )) / (s_R - s_L);

			double F2_M = ((s_R * F_L.y) - (s_L * F_R.y) + s_L * s_R * (qx_R - qx_L)) / (s_R - s_L);

			//			
			if ((s_L < 0.0) && (s_M >= 0.0)) 
			{
				F_face.x = F1_M;
				F_face.y = F2_M;
				F_face.z = F1_M * v_L;
				//				
			}
			else if ((s_M < 0.0) && (s_R >= 0.0))
			{
				//				
				F_face.x = F1_M;
				F_face.y = F2_M;
				F_face.z = F1_M * v_R;
				//					
			}
		}

		else if (s_R < 0)
		{
			//			
			F_face.x = F_R.x;
			F_face.y = F_R.y;
			F_face.z = F_R.z;
			//	
		}

		return F_face;

	}
	//	

}

inline __device__ double3 hll_y(double h_S, double h_N, double qx_S, double qx_N, double qy_S, double qy_N)
{
	double3 G_face = make_double3(0.0, 0.0, 0.0);
	// This function calculates the interface fluxes in x-direction.
	double u_S = 0.0;
	double v_S = 0.0;
	double u_N = 0.0;
	double v_N = 0.0;

	if ((h_S <= TOL_H) && (h_N <= TOL_H))
	{
		G_face.x = 0.0;
		G_face.y = 0.0;
		G_face.z = 0.0;

		return G_face;
	}
	else
	{

		if (h_S <= TOL_H)
		{
			h_S = 0.0;
			u_S = 0.0;
			v_S = 0.0;
		}
		else
		{

			u_S = qx_S / h_S;
			v_S = qy_S / h_S;
		}


		if (h_N <= TOL_H)
		{
			h_N = 0.0;
			u_N = 0.0;
			v_N = 0.0;
		}
		else
		{
			u_N = qx_N / h_N;
			v_N = qy_N / h_N;
		}

		double a_S = sqrt(GRAVITY * h_S);
		double a_N = sqrt(GRAVITY * h_N);

		double h_star = pow(((a_S + a_N) / 2.0 + (v_S - v_N) / 4.0), 2.0) / GRAVITY;
		double v_star = (v_S + v_N) / 2.0 + a_S - a_N;
		double a_star = sqrt(GRAVITY * h_star);

		double s_S, s_N;

		if (h_S <= TOL_H)
		{
			s_S = v_N - (2.0 * a_N);
		}
		else
		{
			s_S = min(v_S - a_S, v_star - a_star);
		}



		if (h_N <= TOL_H)
		{
			s_N = v_S + (2.0 * a_S);
		}
		else
		{
			s_N = max(v_N + a_N, v_star + a_star);
		}

		double s_M = ((s_S * h_N * (v_N - s_N)) - (s_N * h_S * (v_S - s_S))) / (h_N * (v_N - s_N) - (h_S * (v_S - s_S)));

		
		double3 G_S, G_N;

		 G_S = G_SWE((double)h_S, (double)qx_S, (double)qy_S);

		 G_N = G_SWE((double)h_N, (double)qx_N, (double)qy_N);


		if (s_S >= 0.0)
		{
			G_face.x = G_S.x;
			G_face.y = G_S.y;
			G_face.z = G_S.z;

			//return G_S; the Lewis code

		}

		else if ((s_S < 0.0) && (s_N >= 0.0))

		{

			double G1_M = ((s_N * G_S.x) - (s_S * G_N.x) + s_S * s_N * (h_N - h_S)) / (s_N - s_S);

			double G3_M = ((s_N * G_S.z) - (s_S * G_N.z) + s_S * s_N * (qy_N - qy_S)) / (s_N - s_S);
			//			
			if ((s_S < 0.0) && (s_M >= 0.0))
			{
				G_face.x = G1_M;
				G_face.y = G1_M * u_S;
				G_face.z = G3_M;
				//				
			}
			else if ((s_M < 0.0) && (s_N >= 0.0))
			{
				//				
				G_face.x = G1_M;
				G_face.y = G1_M * u_N;
				G_face.z = G3_M;
				//					
			}
		}

		else if (s_N < 0)
		{
			//			
			G_face.x = G_N.x;
			G_face.y = G_N.y;
			G_face.z = G_N.z;

			//return G_N;
			//	
		}

		return G_face;

	}
	//	
}


__FLAME_GPU_FUNC__ int ProcessSpaceOperatorMessage(xmachine_memory_FloodCell* agent, xmachine_message_SpaceOperatorMessage_list* SpaceOperatorMessage_messages)
{
	double3 FPlus  = make_double3(0.0, 0.0, 0.0);
	double3 FMinus = make_double3(0.0, 0.0, 0.0);
	double3 GPlus  = make_double3(0.0, 0.0, 0.0);
	double3 GMinus = make_double3(0.0, 0.0, 0.0);
	
	// initiating the input/outputs of hll/WD in EAST face
	double z_LR_E = agent->z0;
	double h_L_star_E = 0;
	double h_R_star_E = 0;
	double qx_L_star_E = 0;
	double qx_R_star_E = 0;
	double qy_L_star_E = 0;
	double qy_R_star_E = 0;
	// initiating the input/outputs of hll/WD function in WEST face
	double z_LR_W = agent->z0;
	double h_L_star_W = 0;
	double h_R_star_W = 0;
	double qx_L_star_W = 0;
	double qx_R_star_W = 0;
	double qy_L_star_W = 0;
	double qy_R_star_W = 0;
	// initiating the input/outputs of hll/WD in NORTH face
	double z_LR_N = agent->z0;
	double h_L_star_N = 0;
	double h_R_star_N = 0;
	double qx_L_star_N = 0;
	double qx_R_star_N = 0;
	double qy_L_star_N = 0;
	double qy_R_star_N = 0;
	// initiating the input/outputs of hll/WD in SOUTH face
	double z_LR_S = agent->z0;
	double h_L_star_S = 0;
	double h_R_star_S = 0;
	double qx_L_star_S = 0;
	double qx_R_star_S = 0;
	double qy_L_star_S = 0;
	double qy_R_star_S = 0;

	///////////////////////////////////// Before Reading 
	
	double  h_L_E  = agent->hFace_E;
	double  et_L_E = agent->etFace_E;
	double2 q_L_E  = make_double2(agent->qxFace_E, agent->qyFace_E);


	double  h_R_E  = h_L_E; 
	double  et_R_E = et_L_E; 
	double2 q_R_E  = q_L_E;

		//Wetting and drying "depth-positivity-preserving" reconstructions
	WD(h_L_E, h_R_E, et_L_E, et_R_E, q_L_E.x, q_R_E.x, q_L_E.y, q_R_E.y, EAST, z_LR_E, h_L_star_E, h_R_star_E, qx_L_star_E, qx_R_star_E, qy_L_star_E, qy_R_star_E);

	// Flux across the cell(ic):
	FPlus = hll_x(h_L_star_E, h_R_star_E, qx_L_star_E, qx_R_star_E, qy_L_star_E, qy_R_star_E);




	double h_R_W = agent->hFace_W;
	double et_R_W = agent->etFace_W;
	double2 q_R_W = make_double2(agent->qxFace_W, agent->qyFace_W);

	double  h_L_W = h_R_W;
	double  et_L_W = et_R_W;
	double2 q_L_W = q_R_W;


	//* Wetting and drying "depth-positivity-preserving" reconstructions
	WD(h_L_W, h_R_W, et_L_W, et_R_W, q_L_W.x, q_R_W.x, q_L_W.y, q_R_W.y, WEST, z_LR_W, h_L_star_W, h_R_star_W, qx_L_star_W, qx_R_star_W, qy_L_star_W, qy_R_star_W);

	// Flux across the cell(ic):
	FMinus = hll_x(h_L_star_W, h_R_star_W, qx_L_star_W, qx_R_star_W, qy_L_star_W, qy_R_star_W);


	
	double h_L_N = agent->hFace_N;
	double et_L_N = agent->etFace_N;
	double2 q_L_N = make_double2(agent->qxFace_N, agent->qyFace_N);

	double h_R_N = h_L_N;
	double et_R_N = et_L_N;
	double2 q_R_N = q_L_N;



	//* Wetting and drying "depth-positivity-preserving" reconstructions
	WD(h_L_N, h_R_N, et_L_N, et_R_N, q_L_N.x, q_R_N.x, q_L_N.y, q_R_N.y, NORTH, z_LR_N, h_L_star_N, h_R_star_N, qx_L_star_N, qx_R_star_N, qy_L_star_N, qy_R_star_N);

	// Flux across the cell(ic):
	GPlus = hll_y(h_L_star_N, h_R_star_N, qx_L_star_N, qx_R_star_N, qy_L_star_N, qy_R_star_N);

			
	
	double h_R_S = agent->hFace_S;
	double et_R_S = agent->etFace_S;
	double2 q_R_S = make_double2(agent->qxFace_S, agent->qyFace_S);

	double h_L_S = h_R_S;
	double et_L_S = et_R_S;
	double2 q_L_S = q_R_S;


	//* Wetting and drying "depth-positivity-preserving" reconstructions
	WD(h_L_S, h_R_S, et_L_S, et_R_S, q_L_S.x, q_R_S.x, q_L_S.y, q_R_S.y, SOUTH, z_LR_S, h_L_star_S, h_R_star_S, qx_L_star_S, qx_R_star_S, qy_L_star_S, qy_R_star_S);

	// Flux across the cell(ic):
	GMinus = hll_y(h_L_star_S, h_R_star_S, qx_L_star_S, qx_R_star_S, qy_L_star_S, qy_R_star_S);

	///////////////////////////////////// Before Reading


	
	xmachine_message_SpaceOperatorMessage* msg = get_first_SpaceOperatorMessage_message<DISCRETE_2D>(SpaceOperatorMessage_messages, agent->x, agent->y);
				
	while (msg)

	{
		if (msg->inDomain==1)
		{
			if (msg->x + 1 == agent->x
				&& agent->y == msg->y)
			{
				//WE NEED Local EAST values and Neighbours' WEST Values
				// EAST PART (PLUS in x direction)
				 double& h_R_E  = msg->hFace_W ; //double& changed to double MS02Oct2017
				 double& et_R_E = msg->etFace_W; //double& changed to double MS02Oct2017
				  q_R_E  = make_double2(msg->qFace_X_W, msg->qFace_Y_W);

				  h_L_E  = agent->hFace_E;
				  et_L_E = agent->etFace_E;
				  q_L_E  = make_double2(agent->qxFace_E, agent->qyFace_E);


				//Wetting and drying "depth-positivity-preserving" reconstructions
				WD(h_L_E, h_R_E, et_L_E, et_R_E, q_L_E.x, q_R_E.x, q_L_E.y, q_R_E.y, EAST, z_LR_E, h_L_star_E, h_R_star_E, qx_L_star_E, qx_R_star_E, qy_L_star_E, qy_R_star_E);

				// Flux across the cell(ic):
				FPlus = hll_x(h_L_star_E, h_R_star_E, qx_L_star_E, qx_R_star_E, qy_L_star_E, qy_R_star_E);


			}
			else
			if (msg->x - 1 == agent->x
				&& agent->y == msg->y)
			{
				//Local WEST, Neighbour EAST
				// West PART (Minus x direction)
				double& h_L_W = msg->hFace_E; //double& changed to double MS02Oct2017
				double& et_L_W = msg->etFace_E; //double& changed to double MS02Oct2017
				 q_L_W = make_double2(msg->qFace_X_E, msg->qFace_Y_E);

				 h_R_W = agent->hFace_W;
				 et_R_W = agent->etFace_W;
				 q_R_W = make_double2(agent->qxFace_W, agent->qyFace_W);

				//* Wetting and drying "depth-positivity-preserving" reconstructions
				WD(h_L_W, h_R_W, et_L_W, et_R_W, q_L_W.x, q_R_W.x, q_L_W.y, q_R_W.y, WEST, z_LR_W, h_L_star_W, h_R_star_W, qx_L_star_W, qx_R_star_W, qy_L_star_W, qy_R_star_W);

				// Flux across the cell(ic):
				FMinus = hll_x(h_L_star_W, h_R_star_W, qx_L_star_W, qx_R_star_W, qy_L_star_W, qy_R_star_W);

			}
			else
			if (msg->x == agent->x
				&& agent->y == msg->y - 1)
			{
				//Local NORTH, Neighbour SOUTH
				// North Part (Plus Y direction)
				double& h_R_N = msg->hFace_S; //double& changed to double MS02Oct2017
				double& et_R_N = msg->etFace_S; //double& changed to double MS02Oct2017
				 q_R_N = make_double2(msg->qFace_X_S, msg->qFace_Y_S);


				 h_L_N  = agent->hFace_N;
				 et_L_N = agent->etFace_N;
				 q_L_N = make_double2(agent->qxFace_N, agent->qyFace_N);

				//* Wetting and drying "depth-positivity-preserving" reconstructions
				WD(h_L_N, h_R_N, et_L_N, et_R_N, q_L_N.x, q_R_N.x, q_L_N.y, q_R_N.y, NORTH, z_LR_N, h_L_star_N, h_R_star_N, qx_L_star_N, qx_R_star_N, qy_L_star_N, qy_R_star_N);

				// Flux across the cell(ic):
				GPlus = hll_y(h_L_star_N, h_R_star_N, qx_L_star_N, qx_R_star_N, qy_L_star_N, qy_R_star_N);

			}
			else
			if (msg->x == agent->x
				&& agent->y == msg->y + 1)
			{
				//Local SOUTH, Neighbour NORTH
				// South part (Minus y direction)
				double& h_L_S = msg->hFace_N; //double& changed to double MS02Oct2017
				double& et_L_S = msg->etFace_N; //double& changed to double MS02Oct2017
				 q_L_S = make_double2(msg->qFace_X_N, msg->qFace_Y_N);

				 h_R_S  = agent->hFace_S;
				 et_R_S = agent->etFace_S;
				 q_R_S = make_double2(agent->qxFace_S, agent->qyFace_S);



				//* Wetting and drying "depth-positivity-preserving" reconstructions
				WD(h_L_S, h_R_S, et_L_S, et_R_S, q_L_S.x, q_R_S.x, q_L_S.y, q_R_S.y, SOUTH, z_LR_S, h_L_star_S, h_R_star_S, qx_L_star_S, qx_R_star_S, qy_L_star_S, qy_R_star_S);

				// Flux across the cell(ic):
				GMinus = hll_y(h_L_star_S, h_R_star_S, qx_L_star_S, qx_R_star_S, qy_L_star_S, qy_R_star_S);


			}
		}

		msg = get_next_SpaceOperatorMessage_message<DISCRETE_2D>(msg, SpaceOperatorMessage_messages);
	}

	// Topography slope
	double z1x_bar = (z_LR_E - z_LR_W) / 2.0;
	double z1y_bar = (z_LR_N - z_LR_S) / 2.0;

	// Water height average
	double h0x_bar = (h_R_star_W + h_L_star_E) / 2.0;
	double h0y_bar = (h_R_star_S + h_L_star_N) / 2.0;


	// Evaluating bed slope source term
	double3 Source = Sb(h0x_bar, h0y_bar, 2.0 * z1x_bar / DXL, 2.0 * z1y_bar / DYL);


	agent->h  = agent->h  - (TIMESTEP / DXL) * (FPlus.x - FMinus.x) - (TIMESTEP / DYL) * (GPlus.x - GMinus.x) + TIMESTEP * Source.x;
	agent->qx = agent->qx - (TIMESTEP / DXL) * (FPlus.y - FMinus.y) - (TIMESTEP / DYL) * (GPlus.y - GMinus.y) + TIMESTEP * Source.y;
	agent->qy = agent->qy - (TIMESTEP / DXL) * (FPlus.z - FMinus.z) - (TIMESTEP / DYL) * (GPlus.z - GMinus.z) + TIMESTEP * Source.z;
	//agent->h  = TIMESTEP * agent->h;
	//agent->qx = TIMESTEP * agent->qx;
	//agent->qy = TIMESTEP * agent->qy;
	//agent->h = 0.05 * agent->h;
	//agent->qx = 0.05 * agent->qx;
	//agent->qy = 0.05 * agent->qy;


	// Secure zero velocities at the wet/dry front
	double h0 = agent->h;

	if (h0 <= TOL_H)
	{
		agent->qx = 0.0;
		agent->qy = 0.0;
		//this needs to be set high, so it is ignored in the timestep reduction stage
		agent->timeStep = BIG_NUMBER;
	}
	else
	{
		double up = agent->qx / h0;
		double vp = agent->qy / h0;

		//store for timestep calc
		double xStep = CFL * DXL / (fabs(up) + sqrt(GRAVITY * h0));
		double yStep = CFL * DYL / (fabs(vp) + sqrt(GRAVITY * h0));

		agent->timeStep = min(xStep, yStep);

		//TIMESTEP = agent->timeStep;
	}


	return 0;
}


inline __device__ double3 Sb(double hx_mod, double hy_mod, double zprime_x, double zprime_y)
{
	// This function outputs the bed slope source terms for a specific flow data
	double3 result;

	result.x = 0.0;
	result.y = -GRAVITY * hx_mod * zprime_x;
	result.z = -GRAVITY * hy_mod * zprime_y;

	return result;

}


inline __device__ double3 F_SWE(double hh, double qx, double qy)
{
	//This function evaluates the physical flux in the x-direction

	double3 FF = make_double3(0.0, 0.0, 0.0);

	if (hh <= TOL_H)
	{
		FF.x = 0.0;
		FF.y = 0.0;
		FF.z = 0.0;
	}
	else
	{
		FF.x = qx;
		FF.y = (pow(qx, 2.0) / hh) + ((GRAVITY / 2.0)*pow(hh, 2.0));
		FF.z = qx * qy / hh;
	}

	return FF;

}


inline __device__ double3 G_SWE(double hh, double qx, double qy)
{
	//This function evaluates the physical flux in the y-direction


	double3 GG = make_double3(0.0, 0.0, 0.0);

	if (hh <= TOL_H)
	{
		GG.x = 0.0;
		GG.y = 0.0;
		GG.z = 0.0;
	}
	else
	{
		GG.x = qy ;
		GG.y = qx * qy / hh ;
		GG.z = (pow(qy, 2.0) / hh) + ((GRAVITY / 2.0)*pow(hh, 2.0)) ;
	}

	return GG;

}

#endif 
