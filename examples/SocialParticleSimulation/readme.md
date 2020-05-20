The social particle swarm (sps) model is a self-driven particle systems used for the simulation of continuous dynamics of social systems in an abstract space. The model originally assumes agents interacting in a 2D Cartesian space which is an abstraction of the social or psychological space of the agent.

We considered interacting agents in a 3D abstract social space, each with an initial position (Xi) and velocity (Vi), and identical interaction ranges (R). Through kinematics, agents express their social inclination by getting closer to groups offering social gain, or getting away from less fortunate ones, meaning that their positions represent their social relationship against others. Social gain is expressed with a prisoner's dilemma game for (n) neighbors, using a generalized tit-for-tat strategy, each agent calculates the score it receives from its neighbors, weighted by the distance between them. Details on how to compute the total score and an overall overview of the model can be found here:

https://www.mitpressjournals.org/doi/pdfplus/10.1162/isal_a_00092

Agents run three processes, (1)cooperators estimation, (2)strategy update and (3) social relationship change. The first one entails estimating the ratio of neighboring cooperators using a formula inspired partly from models of predictive coding in cognitive neuroscience, which depends on an information update rate about the neighbors (I). The formula is as follows: e(t+1) = e(t) + ( r(t)-e(t) * I ), where e(t) is the estimate at instant (t) and r(t) is the actual ratio of cooperators at (t), I is the update rate (I ∈ [0,1]). If I is large, agents quickly minimize the prediction error about their neighbors, and conversely. In strategy update, e is compared against a cooperation threshold (thresh) (pre-set randomly at the beginning), if e > thresh then the agent i sets its strategy to cooperation, and to defection otherwise. The last process is expressed through steering forces, where an agent steers towards (or away form) the center of mass of its neighbors depending the sign of the TOTAL_SCORE_i. The generated steering force vector is obtained using Reynolds steering formula[1].

Simulation videos can be viewed here:
https://vimeo.com/338902405

Description of the simulation:
The simulation shows interesting phenomena of formation of spherical cooperative clusters followed by their explosion due to overpopulation by defectors. Moreover, when the update rate (I) was minimal (0.01), cooperative clusters were few but consistent over time(has longer lifespan). 

The paper outlining the and motivation behind the model and results can be found here:
https://link.springer.com/article/10.1007/s10015-019-00558-6

Other references:

[1] Reynolds, C. (1999). Steering behaviors for autonomous characters. Proceedings of Game Developers Conference 1999, pages 763-782.

Technical trivia:

(1) This model uses 3D spatially partitioning scheme. 

(2).a Simulations were run on a desktop computer with GeForce GTX 1080 Ti graphics card (up to 100,000 agents)

(2).b Simulations were run on a laptop computer with GeForce 940MX laptop computer graphics card (up to 5,000 agents)


