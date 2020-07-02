import sys
#sys.path.append("/home/taegyun/vector45/mujoco-py") # change directory to mujoco-py in your virtural environment
import mujoco_py
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
import Robotics as r
import Control as c
import numpy as np



# M_SE
M_se = np.array([[1.0, 0.0, 0.0, 0.088],[0.0, -1.0, 0.0, 0.0],[0.0, 0.0, -1.0, 0.823],[0.0, 0.0, 0.0, 1.0]])
# Screws
S = r.Screws()

# initial guess of q0 for inverse kinematics
q_init = np.zeros((7,1))
q_init[:,0]=(r.JointLimit()[1]+r.JointLimit()[0])/2

# joint angles of robot. 
q_t = np.zeros((7,1))

# desired SE3 to grasp an object (Grasp Pose!)
q_t[:,0] = [0.0, np.pi/6, 0.0, -np.pi*12/24, 0.0, np.pi*5/8,0.0]
T_desir =  r.ForwardKinematics(S, q_t, M_se)
print('T_desir', T_desir)

# solve inverse kinmatics of desired SE3
q_desir = r.InvKinematics2(S, q_init, M_se, T_desir)

# adjust inverse kinematics solution to satisfy joint limit
q_desir = r.Joint_Limit_Check(r.JointLimit()[1],r.JointLimit()[0],q_desir)

# raise after grasp
q_raise = np.zeros((7,1))
q_raise[:,0] = [0.0, -np.pi/6,0.0, -np.pi*2/3, 0.0, np.pi/2,0.0]

# error for PID control
e_prev = np.zeros(7)
e_int = np.zeros(7)
start_step = 0

# sim setup
model = load_model_from_path("franka_sim/franka_panda.xml")
sim = MjSim(model)
viewer = MjViewer(sim)

step = 0
time_step = 0.002 # .... does not know how to change it... (Is in franka_sim/assets/assets.xml file)

# check whether reached grasp pose. True if all joint angles reaches desired joint angles
reached_grasp_pose = False

# check whether reached raise pose
reached_raise_pose = False

# True if grasped False else
grasped = False

# thresholds to determine successful grasp.  
time_threshold = 10 # secs
height_threshold = 0.65

sim_state = sim.get_state()
while True:
	sim_state = sim.get_state()
	if not reached_grasp_pose:
		# move each joints to desired joint angles (grasp pose)
		q = sim_state.qpos[0:7]
		e_prev = c.controller(sim,q, q_desir,e_prev,e_int,step,start_step,time_step)
		e_int = e_int + e_prev*time_step
		# open finger
		c.finger_open(sim)	
		reached_grasp_pose = c.check_pose_reached(q,q_desir)
		if reached_grasp_pose:
			e_prev = np.zeros(7)
			e_int = np.zeros(7)
			
		

	if reached_grasp_pose and not c.contact_detect(sim, 'object'):
		c.finger_close(sim)
    
	if reached_grasp_pose and  not grasped and c.contact_detect(sim, 'object'):
		grasped = True
		start_step = step

	if grasped:
		if not reached_raise_pose:

			q = sim_state.qpos[0:7]
			e_prev = c.controller(sim,q, q_raise,e_prev,e_int,step,start_step,time_step)
			e_int = e_int + e_prev*time_step

			reached_raise_pose = c.check_pose_reached(q,q_raise)
			step_raised = step
		else:
			object_height = sim_state.qpos[11]
			if step - step_raised > time_threshold/time_step: 
				if object_height > height_threshold:
					print('grasp success!')
				else:
					print('grasp failed!')
				break;
	step += 1
	sim.step()
	viewer.render()
  
	if os.getenv('TESTING') is not None:
		break
