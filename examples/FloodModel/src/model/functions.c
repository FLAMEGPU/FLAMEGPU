
#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

#include "header.h"
#include "cutil_math.h"



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

#define BIG_NUMBER 800000             // Used in WetDrayMessage to skip extra calculations MS05Sep2017



inline __device__ double3 Roe_x(double h_L, double h_R, double qx_L, double qx_R, double qy_L, double qy_R);
inline __device__ double3 Roe_y(double h_L, double h_R, double qx_L, double qx_R, double qy_L, double qy_R);
inline __device__ double3 Flux_F(double hh, double qx, double qy);
inline __device__ double3 Flux_G(double hh, double qx, double qy);
inline __device__ double psi_Roe(double a, double d);
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

	centBoundData.h = FlowData.h;

	centBoundData.qx = -FlowData.qx;

	centBoundData.qy = -FlowData.qy;

}

//global functions CHECK IF IT IS NEEDED MS05Sep2017
inline __device__ float GetRandomNumber(const float minNumber, const float maxNumber, RNG_rand48* rand48)
{
	return minNumber + (rnd(rand48)*(maxNumber - minNumber));
}


// Conditional function to check whether the Height is less than TOL_H or not
//inline __device__ bool IsDry(double waterHeight)
//{
//	return waterHeight <= TOL_H;
//}


inline __device__ double2 GetCellPosition(xmachine_memory_FloodCell* agent, double2 offset) // '__inline' changed to 'inline' by MS05Sep2017
{
	double x = (agent->x * DXL) + offset.x;
	double y = (agent->y * DYL) + offset.y;

	return make_double2(x, y);
}


// This function should be called when minh_loc is greater than TOL_H
inline __device__ double2 friction_2D(double dt_loc, double h_loc, double qx_loc, double qy_loc)
{
	//This function add the friction contribution to wet cells.
	//This fucntion should not be called when: minh_loc.LE.TOL_H << check if the function has the criterion to be taken into accont MS05Sep2017

	double2 result;

	result.x = 0.0f;
	result.y = 0.0f;


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

		double expULoc = pow(u_loc, 2);
		double expVLoc = pow(v_loc, 2);

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


//__FLAME_GPU_FUNC__ int PrepareWetDryFrontMessages(xmachine_memory_FloodCell* agent, xmachine_message_WetDryMessage_list* WetDryMessage_messages)
__FLAME_GPU_FUNC__ int PrepareWetDry(xmachine_memory_FloodCell* agent, xmachine_message_WetDryMessage_list* WerDryMessage_messages)
{
	if (agent->inDomain)
	{

		AgentFlowData FlowData = GetFlowDataFromAgent(agent);

		double h0 = FlowData.h; // it recalls the Local h from flow data


		agent->minh_loc = h0;


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
	if (agent->inDomain)
	{

		//looking up neighbours values for wet/dry tracking "THIS NEEDS TO BE ANALYSED carefully MS05Sep2017"
		xmachine_message_WetDryMessage* msg = get_first_WetDryMessage_message<DISCRETE_2D>(WetDryMessage_messages, agent->x, agent->y);

		double maxHeight = agent->minh_loc;

		while (msg)
		{
			if (msg->inDomain)
			{
				agent->minh_loc = min(agent->minh_loc, msg->min_hloc);
			}

			if (msg->min_hloc > maxHeight)
			{
				maxHeight = msg->min_hloc;
			}

			msg = get_next_WetDryMessage_message<DISCRETE_2D>(msg, WetDryMessage_messages);
		}


		//agent->isDry = IsDry(maxHeight); // should be corrected to checkin TOL_H inline here, not calling the function to the minimum hight of water MS05Sep2017

		if (maxHeight > TOL_H) // if the Height of water is not less than TOL_H => the friction is needed to be taken into account MS05Sep2017
		{
			Friction_Implicit(agent, TIMESTEP); // TIMESTEP has been defined in agents' initial constant MS05Sep2017
		}
		else
		{
			//need to go high, so that it won't affect min calculation when it is tested again . Needed to be tested MS05Sep2017 which is now temporary. needs to be corrected somehow
			agent->minh_loc = BIG_NUMBER;
		}

	}

	return 0;
}

__FLAME_GPU_FUNC__ int PrepareLFV(xmachine_memory_FloodCell* agent, xmachine_message_LFVMessage_list* LFVMessage_messages)
{
	if (agent->inDomain)
	{

		//// Needs to be corrected  "MS05Sep2017"
		//agent->isDry == false;
		if (agent->minh_loc > TOL_H)
		{

		AgentFlowData FlowData = GetFlowDataFromAgent(agent);

		//broadcast the to the surrounding cells
		add_LFVMessage_message<DISCRETE_2D>(LFVMessage_messages, 1, agent->x, agent->y, FlowData.z0, FlowData.h, FlowData.qx, FlowData.qy);
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

	FlowData.h = msg->h;

	FlowData.qx = msg->qx;

	FlowData.qy = msg->qy;
}


//inline __device__ LFVResult LFV(
//	const AgentFlowData& FlowData,
//	const double2& CellCentre,  // used for future evaluations MS06Sep2017
//	const double2& facePosition,// used for future evaluations MS06Sep2017
//	double minh_loc				// used for future evaluations MS06Sep2017
//	const AgentFlowData& eastNeighbourFlowData,
//	const AgentFlowData& westNeighbourFlowData,
//	const AgentFlowData& northNeighbourFlowData,
//	const AgentFlowData& southNeighbourFlowData
//	)
//{

inline __device__ LFVResult LFV(
	const AgentFlowData& FlowData
	)
{

	LFVResult result;


	result.h_face = FlowData.h;

	result.et_face = FlowData.et;

	result.qFace.x = FlowData.qx;

	result.qFace.y = FlowData.qy;

	return result;
}


__FLAME_GPU_FUNC__ int ProcessLFVMessages(xmachine_memory_FloodCell* agent, xmachine_message_LFVMessage_list* LFVMessage_messages)
{
	if (agent->inDomain
		&& agent->minh_loc > TOL_H)
	{
		AgentFlowData FlowData = GetFlowDataFromAgent(agent);

		// declaring new variables "MS05Sep2017"
		AgentFlowData eastNeighbourFlowData;
		AgentFlowData westNeighbourFlowData;
		AgentFlowData northNeighbourFlowData;
		AgentFlowData southNeighbourFlowData;

		//initialise values as transmissive cells  - Boundary condition MS05Sep2017
		centbound(agent, FlowData, eastNeighbourFlowData);
		centbound(agent, FlowData, westNeighbourFlowData);
		centbound(agent, FlowData, northNeighbourFlowData);
		centbound(agent, FlowData, southNeighbourFlowData);

		//check for neighbour flow data
		xmachine_message_LFVMessage* msg = get_first_LFVMessage_message<DISCRETE_2D>(LFVMessage_messages, agent->x, agent->y);

		while (msg)
		{
			if (msg->inDomain)
			{
				if (agent->x - 1 == msg->x
					&&   agent->y == msg->y)
				{
					FlowDataFromLFVMessage(westNeighbourFlowData, msg);
				}
				else
				if (agent->x + 1 == msg->x
					&&   agent->y == msg->y)
				{
					FlowDataFromLFVMessage(eastNeighbourFlowData, msg);
				}
				else
				if (agent->x == msg->x
					&&   agent->y + 1 == msg->y)
				{
					FlowDataFromLFVMessage(northNeighbourFlowData, msg);
				}
				else
				if (agent->x == msg->x
					&&   agent->y - 1 == msg->y)
				{
					FlowDataFromLFVMessage(southNeighbourFlowData, msg);
				}
			}

			msg = get_next_LFVMessage_message<DISCRETE_2D>(msg, LFVMessage_messages);

		}

		double2 cellPosition = GetCellPosition(agent, make_double2(0.0, 0.0)); // 2nd term is 'offsets' in x and y direction MS05Sep2017



		//EAST
		double2 facePosition = GetCellPosition(agent, make_double2(DXL * 0.5, 0.0));
		LFVResult faceLFV = LFV(FlowData);

		//store for calculation in the Space_operator stage
		agent->hFace_E = faceLFV.h_face;
		agent->etFace_E = faceLFV.et_face;
		agent->qxFace_E = faceLFV.qFace.x;
		agent->qyFace_E = faceLFV.qFace.y;



		//WEST

		facePosition = GetCellPosition(agent, make_double2(-DXL * 0.5, 0.0));

		faceLFV = LFV(FlowData);

		//store for calculation in the Space_operator stage
		agent->hFace_W = faceLFV.h_face;
		agent->etFace_W = faceLFV.et_face;
		agent->qxFace_W = faceLFV.qFace.x;
		agent->qyFace_W = faceLFV.qFace.y;


		//NORTH
		facePosition = GetCellPosition(agent, make_double2(0.0, DYL * 0.5));

		faceLFV = LFV(FlowData);

		//store for calculation in the Space_operator stage
		agent->hFace_N = faceLFV.h_face;
		agent->etFace_N = faceLFV.et_face;
		agent->qxFace_N = faceLFV.qFace.x;
		agent->qyFace_N = faceLFV.qFace.y;


		//SOUTH 
		facePosition = GetCellPosition(agent, make_double2(0.0, -DYL * 0.5));

		faceLFV = LFV(FlowData);

		//store for calculation in the Space_operator stage

		agent->etFace_S = faceLFV.et_face;
		agent->hFace_S = faceLFV.h_face;
		agent->qxFace_S = faceLFV.qFace.x;
		agent->qyFace_S = faceLFV.qFace.y;

	}

	return 0;
}


__FLAME_GPU_FUNC__ int PrepareSpaceOperator(xmachine_memory_FloodCell* agent, xmachine_message_SpaceOperatorMessage_list* SpaceOperatorMessage_messages)
{
	if (agent->inDomain
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

	h_L_star = et_L_star - z_LR; // added to the code based on the MATLAB three Humps code MS05Sep2017 
	h_R_star = et_R_star - z_LR; // added to the code based on the MATLAB three Humps code MS05Sep2017
}


inline __device__ double3 Roe_x(double h_L, double h_R, double qx_L, double qx_R, double qy_L, double qy_R)
{
	double3 F_face = make_double3(0.0, 0.0, 0.0);
	// This function calculates the interface fluxes in x-direction.

	double u_L = 0.0;
	double u_R = 0.0;
	double v_L = 0.0;
	double v_R = 0.0;

	if ((h_L <= TOL_H) && (h_R <= TOL_H))
	{
		F_face.x = 0.0;
		F_face.y = 0.0;
		F_face.z = 0.0;

		return F_face;
	}

	if (h_L <= TOL_H)
	{
		//h_L = 0.0;  MS:25MAY2017 
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
		//h_R = 0.0;  MS:25MAY2017 
		u_R = 0.0;
		v_R = 0.0;
	}
	else
	{
		u_L = qx_R / h_R;
		v_L = qy_R / h_R;
	}

	double c_L = sqrt(GRAVITY * h_L);
	double a1_L = u_L + c_L;
	double a2_L = u_L;
	double a3_L = u_L - c_L;

	double c_R = sqrt(GRAVITY * h_R);
	double a1_R = u_R + c_R;
	double a2_R = u_R;
	double a3_R = u_R - c_R;

	//Roe Linearisation
	double u_roe = (u_R * sqrt(h_R) + u_L * sqrt(h_L)) / (sqrt(h_R) + sqrt(h_L));
	double v_roe = (v_R * sqrt(h_R) + v_L * sqrt(h_L)) / (sqrt(h_R) + sqrt(h_L));

	double c_mid = sqrt(GRAVITY*(h_L + h_R) / 2.0);

	//Associated eigenvalues
	double a1 = u_roe + c_mid;
	double a2 = u_roe;
	double a3 = u_roe - c_mid;

	// Entropy-fix parameters
	double d1 = min(c_mid, max(0.0, 2.0*(a1_R - a1_L)));
	double d2 = min(c_mid, max(0.0, 2.0*(a2_R - a2_L)));
	double d3 = min(c_mid, max(0.0, 2.0*(a3_R - a3_L)));

	//Eigenvectors (these are vectors)

	double e1[3] = { 1.0, a1, v_roe };
	double e2[3] = { 0, 0, c_mid };
	double e3[3] = { 1.0, a3, v_roe };

	// Waves Strength
	double alpha_1 = 0.5 * (h_R - h_L) + ((h_R*u_R - h_L*u_L) - u_roe*(h_R - h_L)) / (2.0*c_mid);
	double alpha_2 = ((h_R*v_R - h_L*v_L) - v_roe*(h_R - h_L)) / c_mid;
	double alpha_3 = 0.5 * (h_R - h_L) - ((h_R*u_R - h_L*u_L) - u_roe*(h_R - h_L)) / (2.0*c_mid);

	//Physical Left and Right - FLUX EVALUATION -
	double qx_tempL = h_L*u_L; // temporary qx and qy MS:25MAY2017
	double qy_tempL = h_L*v_L;

	double qx_tempR = h_R*u_R;
	double qy_tempR = h_R*v_R;

	double3 F_L = Flux_F(h_L, qx_tempL, qy_tempL); // for each h,qx,qy F_L.x F_L.y and F_L.z
	double3 F_R = Flux_F(h_R, qx_tempR, qy_tempR); // for each h,qx,qy F_R.x F_R.y and F_R.z

	// Entropy-fix for eigenvalues
	double f1 = psi_Roe(a1, d1);
	double f2 = psi_Roe(a2, d2);
	double f3 = psi_Roe(a3, d3);

	// Average Roe Riemann flux
	F_face.x = 0.5 * (F_L.x + F_R.x - alpha_1*f1*e1[0] - alpha_2*f2*e2[0] - alpha_3*f3*e3[0]);
	F_face.y = 0.5 * (F_L.y + F_R.y - alpha_1*f1*e1[1] - alpha_2*f2*e2[1] - alpha_3*f3*e3[1]);
	F_face.z = 0.5 * (F_L.z + F_R.z - alpha_1*f1*e1[2] - alpha_2*f2*e2[2] - alpha_3*f3*e3[2]);

	return F_face;

}

inline __device__ double3 Roe_y(double h_L, double h_R, double qx_L, double qx_R, double qy_L, double qy_R)
{
	double3 G_face = make_double3(0.0, 0.0, 0.0);
	// This function calculates the interface fluxes in x-direction.

	double u_L = 0.0;
	double u_R = 0.0;
	double v_L = 0.0;
	double v_R = 0.0;

	if ((h_L <= TOL_H) && (h_R <= TOL_H))
	{
		G_face.x = 0.0;
		G_face.y = 0.0;
		G_face.z = 0.0;

		return G_face;
	}

	if (h_L <= TOL_H)
	{
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
		u_R = 0.0;
		v_R = 0.0;
	}
	else
	{
		u_L = qx_R / h_R;
		v_L = qy_R / h_R;
	}

	double c_L = sqrt(GRAVITY * h_L);
	double a1_L = v_L + c_L;
	double a2_L = v_L;
	double a3_L = v_L - c_L;

	double c_R = sqrt(GRAVITY * h_R);
	double a1_R = v_R + c_R;
	double a2_R = v_R;
	double a3_R = v_R - c_R;

	//Roe Linearisation
	double u_roe = (u_R * sqrt(h_R) + u_L * sqrt(h_L)) / (sqrt(h_R) + sqrt(h_L));
	double v_roe = (v_R * sqrt(h_R) + v_L * sqrt(h_L)) / (sqrt(h_R) + sqrt(h_L));

	double c_mid = sqrt(GRAVITY*(h_L + h_R) / 2.0);

	//Associated eigenvalues
	double a1 = v_roe + c_mid;
	double a2 = v_roe;
	double a3 = v_roe - c_mid;

	// Entropy-fix parameters
	double d1 = min(c_mid, max(0.0, 2.0*(a1_R - a1_L)));
	double d2 = min(c_mid, max(0.0, 2.0*(a2_R - a2_L)));
	double d3 = min(c_mid, max(0.0, 2.0*(a3_R - a3_L)));

	//Eigenvectors (these are vectors)
	double e1[3] = { 1.0, u_roe, a1 };
	double e2[3] = { 0, -c_mid, 0 };
	double e3[3] = { 1.0, u_roe, a3 };

	// Waves Strength
	double alpha_1 = 0.5 * (h_R - h_L) + ((h_R*v_R - h_L*v_L) - v_roe*(h_R - h_L)) / (2.0*c_mid);
	double alpha_2 = 0.5 * (h_R - h_L) - ((h_R*v_R - h_L*v_L) - v_roe*(h_R - h_L)) / (2.0*c_mid);
	double alpha_3 = -((h_R * u_R - h_L * u_L) - u_roe * (h_R - h_L)) / c_mid;


	//Physical Left and Right - FLUX EVALUATION -
	double qx_tempL = h_L*u_L; // temporary qx and qy MS:25MAY2017
	double qy_tempL = h_L*v_L;

	double qx_tempR = h_R*u_R;
	double qy_tempR = h_R*v_R;

	double3 G_L = Flux_G(h_L, qx_tempL, qy_tempL); // for each h,qx,qy F_L.x F_L.y and F_L.z
	double3 G_R = Flux_G(h_R, qx_tempR, qy_tempR); // for each h,qx,qy F_R.x F_R.y and F_R.z

	// Entropy-fix for eigenvalues
	double f1 = psi_Roe(a1, d1);
	double f2 = psi_Roe(a2, d2);
	double f3 = psi_Roe(a3, d3);

	// Average Roe Riemann flux
	G_face.x = 0.5 * (G_L.x + G_R.x - alpha_1*f1*e1[0] - alpha_2*f2*e2[0] - alpha_3*f3*e3[0]);
	G_face.y = 0.5 * (G_L.y + G_R.y - alpha_1*f1*e1[1] - alpha_2*f2*e2[1] - alpha_3*f3*e3[1]);
	G_face.z = 0.5 * (G_L.z + G_R.z - alpha_1*f1*e1[2] - alpha_2*f2*e2[2] - alpha_3*f3*e3[2]);

	return G_face;

}



__FLAME_GPU_FUNC__ int ProcessSpaceOperatorMessage(xmachine_memory_FloodCell* agent, xmachine_message_SpaceOperatorMessage_list* SpaceOperatorMessage_messages)
{
	double3 FPlus = make_double3(0.0, 0.0, 0.0);
	double3 FMinus = make_double3(0.0, 0.0, 0.0);
	double3 GPlus = make_double3(0.0, 0.0, 0.0);
	double3 GMinus = make_double3(0.0, 0.0, 0.0);
	double z0f_E;
	double z0f_W;
	double z0f_N;
	double z0f_S;
	double hf_E;
	double hf_W;
	double hf_N;
	double hf_S;
	//double qxf_E;
	//double qxf_W;
	//double qxf_N;
	//double qxf_S;
	//double qyf_E;
	//double qyf_W;
	//double qyf_N;
	//double qyf_S;

	xmachine_message_SpaceOperatorMessage* msg = get_first_SpaceOperatorMessage_message<DISCRETE_2D>(SpaceOperatorMessage_messages, agent->x, agent->y);

	while (msg)
	{
		if (msg->inDomain)
		{
			if (msg->x + 1 == agent->x
				&& agent->y == msg->y)
			{
				//Local EAST, Neighbour WEST
				// EAST PART (PLUS x direction)
				double& h_R = msg->hFace_W;
				double& et_R = msg->etFace_W;
				double2 q_R = make_double2(msg->qFace_X_W, msg->qFace_Y_W);

				double h_L = agent->hFace_E;
				double et_L = agent->etFace_E;
				double2 q_L = make_double2(agent->qxFace_E, agent->qyFace_E);

				// initiating the outputs of WD function
				double z0_F = 0.0;
				double h_F_L = 0.0;
				double h_F_R = 0.0;
				double qx_F_L = 0.0;
				double qx_F_R = 0.0;
				double qy_F_L = 0.0;
				double qy_F_R = 0.0;

				//Wetting and drying "depth-positivity-preserving" reconstructions
				WD(h_L, h_R, et_L, et_R, q_L.x, q_R.x, q_L.y, q_R.y, EAST, z0_F, h_F_L, h_F_R, qx_F_L, qx_F_R, qy_F_L, qy_F_R);

				// Flux across the cell(ic):
				FPlus = Roe_x(h_F_L, h_F_R, qx_F_L, qx_F_R, qy_F_L, qy_F_R);

				//** INNER/LOCAL flow data restrictions at the eastern face**!
				z0f_E = z0_F;
				hf_E = h_F_L;
				//qxf_E = qx_F_L;
				//qyf_E = qy_F_L;
			}
			else
			if (msg->x - 1 == agent->x
				&& agent->y == msg->y)
			{
				//Local WEST, Neighbour EAST
				// West PART (Minus x direction)
				double& h_L = msg->hFace_E;
				double& et_L = msg->etFace_E;
				double2 q_L = make_double2(msg->qFace_X_E, msg->qFace_Y_E);

				double h_R = agent->hFace_W;
				double et_R = agent->etFace_W;
				double2 q_R = make_double2(agent->qxFace_W, agent->qyFace_W);

				// initiating the outputs of WD function
				 z0_F = 0.0;
				 h_F_L = 0.0;
				 h_F_R = 0.0;
				 qx_F_L = 0.0;
				 qx_F_R = 0.0;
				 qy_F_L = 0.0;
				 qy_F_R = 0.0;

				//* Wetting and drying "depth-positivity-preserving" reconstructions
				WD(h_L, h_R, et_L, et_R, q_L.x, q_R.x, q_L.y, q_R.y, WEST, z0_F, h_F_L, h_F_R, qx_F_L, qx_F_R, qy_F_L, qy_F_R);

				// Flux across the cell(ic):
				FMinus = Roe_x(h_F_L, h_F_R, qx_F_L, qx_F_R, qy_F_L, qy_F_R);

				//** INNER/LOCAL flow data restrictions at the eastern face**!
				z0f_W = z0_F;
				hf_W = h_F_R;
				//qxf_W = qx_F_R;
				//qyf_W = qy_F_R;

			}
			else
			if (msg->x == agent->x
				&& agent->y == msg->y - 1)
			{
				//Local NORTH, Neighbour SOUTH
				// North Part (Plus Y direction)
				double& h_R = msg->hFace_S;
				double& et_R = msg->etFace_S;
				double2 q_R = make_double2(msg->qFace_X_S, msg->qFace_Y_S);


				double h_L = agent->hFace_N;
				double et_L = agent->etFace_N;
				double2 q_L = make_double2(agent->qxFace_N, agent->qyFace_N);

				// initiating the outputs of WD function
				 z0_F = 0.0;
				 h_F_L = 0.0;
				 h_F_R = 0.0;
				 qx_F_L = 0.0;
				 qx_F_R = 0.0;
				 qy_F_L = 0.0;
				 qy_F_R = 0.0;

				//* Wetting and drying "depth-positivity-preserving" reconstructions
				WD(h_L, h_R, et_L, et_R, q_L.x, q_R.x, q_L.y, q_R.y, NORTH, z0_F, h_F_L, h_F_R, qx_F_L, qx_F_R, qy_F_L, qy_F_R);

				// Flux across the cell(ic):
				GPlus = Roe_y(h_F_L, h_F_R, qx_F_L, qx_F_R, qy_F_L, qy_F_R);

				//printf( "GPlus: %f, %f, %f\r\n", GPlus.x, GPlus.y, GPlus.z );

				//** INNER/LOCAL flow data restrictions at the eastern face**!
				z0f_N = z0_F;
				hf_N = h_F_L;
				//qxf_N = qx_F_L;
				//qyf_N = qy_F_L;

			}
			else
			if (msg->x == agent->x
				&& agent->y == msg->y + 1)
			{
				//Local SOUTH, Neighbour NORTH
				// South part (Minus y direction)
				double& h_L = msg->hFace_N;
				double& et_L = msg->etFace_N;
				double2 q_L = make_double2(msg->qFace_X_N, msg->qFace_Y_N);

				double h_R = agent->hFace_S;
				double et_R = agent->etFace_S;
				double2 q_R = make_double2(agent->qxFace_S, agent->qyFace_S);

				// initiating the outputs of WD function
				 z0_F = 0.0;
				 h_F_L = 0.0;
				 h_F_R = 0.0;
				 qx_F_L = 0.0;
				 qx_F_R = 0.0;
				 qy_F_L = 0.0;
				 qy_F_R = 0.0;


				//* Wetting and drying "depth-positivity-preserving" reconstructions
				WD(h_L, h_R, et_L, et_R, q_L.x, q_R.x, q_L.y, q_R.y, SOUTH, z0_F, h_F_L, h_F_R, qx_F_L, qx_F_R, qy_F_L, qy_F_R);

				// Flux across the cell(ic):
				GMinus = Roe_y(h_F_L, h_F_R, qx_F_L, qx_F_R, qy_F_L, qy_F_R);

				//** INNER/LOCAL flow data restrictions at the eastern face**!
				z0f_S = z0_F;
				hf_S = h_F_R;
				//qxf_S = qx_F_R;
				//qyf_S = qy_F_R;

			}
		}

		msg = get_next_SpaceOperatorMessage_message<DISCRETE_2D>(msg, SpaceOperatorMessage_messages);
	}

	// Topography slope
	double z01x_mod = (z0f_E - z0f_W) * 0.5;
	double z01y_mod = (z0f_N - z0f_S) * 0.5;

	// Water height average
	double hx_mod = (hf_E - hf_W) * 0.5;
	double hy_mod = (hf_N - hf_S) * 0.5;


	// Evaluating bed slope source term
	double3 Source = Sb(hx_mod, hy_mod, 2.0 * z01x_mod / DXL, 2.0 * z01y_mod / DYL);

	// initialising the space operator variable
	double3 L0_loc = make_double3(0.0, 0.0, 0.0);


	L0_loc.x = -(FPlus.x - FMinus.x) / DXL - (GPlus.x - GMinus.x) / DYL + Source.x;
	L0_loc.y = -(FPlus.y - FMinus.y) / DXL - (GPlus.y - GMinus.y) / DYL + Source.y;
	L0_loc.z = -(FPlus.z - FMinus.z) / DXL - (GPlus.z - GMinus.z) / DYL + Source.z;

	/*printf( "agent: %d, %d\r\nFPlus: %f, %f, %f\r\nFMinus: %f, %f, %f\r\nGPlus: %f, %f, %f\r\nGMinus: %f, %f, %f\r\n",  Kept for further testing in C MS05Sep2017
	agent->x, agent->y,
	FPlus.x, FPlus.y, FPlus.z,
	FMinus.x, FMinus.y, FMinus.z,
	GPlus.x, GPlus.y, GPlus.z,
	GMinus.x, GMinus.y, GMinus.z
	); */

	// Updating the variables as the time ellapse // This is to test MS06Sep2017
	agent->h = agent->h + agent->timeStep * L0_loc.x;
	agent->qx = agent->qx + agent->timeStep * L0_loc.y;
	agent->qy = agent->qy + agent->timeStep * L0_loc.z;

	//agent->h = agent->h + TIMESTEP * L0_loc.x;
	//agent->qx = agent->qx + TIMESTEP * L0_loc.y;
	//agent->qy = agent->qy + TIMESTEP * L0_loc.z;


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

inline __device__ double psi_Roe(double a, double d)
{
	double f;

	if (fabs(a) >= d)
	{
		f = fabs(a);
	}
	else
	{
		f = (pow(a, 2)) / (2 * d) + d / 2;
	}

	return f;
}


inline __device__ double3 Flux_F(double hh, double qx, double qy)
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
		FF.y = (pow(qx, 2) / hh) + ((GRAVITY / 2.0)*pow(hh, 2));
		FF.z = qx * qy / hh;
	}

	return FF;

}


inline __device__ double3 Flux_G(double hh, double qx, double qy)
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
		GG.x = qy;
		GG.y = qx * qy / hh;
		GG.z = (pow(qy, 2) / hh) + ((GRAVITY / 2.0)*pow(hh, 2));
	}

	return GG;

}

#endif 
