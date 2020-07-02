#!/usr/bin/env python3
"""
Shows how to toss a capsule to a container.
"""
import sys
sys.path.append("/home/taegyun/vector45/mujoco-py")
import mujoco_py
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
import Robotics as r
import numpy as np
import random

def finger_close(sim):
	left = 7
	right = 8
	sim_state = sim.get_state()
	if sim_state.qpos[left] > 1e-3:
		sim.data.ctrl[left] = -0.1
	else:
		sim.data.ctrl[left] = 0.0

	if sim_state.qpos[right]>1e-3:
		sim.data.ctrl[right] = -0.1
	else:
		sim.data.ctrl[right] = 0.0

def finger_open(sim):
	left = 7
	right = 8
	sim_state = sim.get_state()
	if abs(sim_state.qpos[left]- 0.04) > 1e-3:
		sim.data.ctrl[left] = 0.1
	else:
		sim.data.ctrl[left] = 0.0

	if abs(sim_state.qpos[right]- 0.04) > 1e-3:
		sim.data.ctrl[right] = 0.1
	else:
		sim.data.ctrl[right] = 0.0

def finger_stop(sim):
	left = 7
	right = 8
	sim.data.ctrl[left] = 0.0
	sim.data.ctrl[right] = 0.0


def contact_detect(sim, ob_name):
	left =7
	right = 8	
	contact_l = sim.data.contact[left]
	contact_r = sim.data.contact[right]
	left_appr_cont = 0
	right_appr_cont = 0 
	if sim.model.geom_id2name(contact_l.geom1) == ob_name or sim.model.geom_id2name(contact_l.geom2) == ob_name:
		left_appr_cont = 1
	if sim.model.geom_id2name(contact_r.geom1) == ob_name or sim.model.geom_id2name(contact_r.geom2) == ob_name:
		right_appr_cont = 1
	left_cont = False
	right_cont = False
	if left_appr_cont and contact_l.dist < 1e-3:
		left_cont = True
	if right_appr_cont and contact_r.dist < 1e-3:
		right_cont = True

	return left_cont and right_cont


obj_pos = np.array([0.65, 0.0, 0.65])
obj_dim = np.array([.1 ,.03, .13])
offset = 0.02
pos_desir = np.array([random.uniform(obj_pos[0]-0.5*obj_dim[0]-offset, obj_pos[0]+0.5*obj_dim[0]+offset),random.uniform(obj_pos[1]-0.5*obj_dim[1]-offset, obj_pos[1]+0.5*obj_dim[1]+offset),random.uniform(0.0, obj_pos[2]+0.5*obj_dim[2]+offset)])
so3_desir=r.EulerZYX_to_SO3(random.uniform(0.0,2*np.pi), random.uniform(-0.5*np.pi,0.5*np.pi), random.uniform(0.0,2*np.pi))

T_desir = np.eye(4)
T_desir[0:3,0:3]=so3_desir
T_desir[0:3,3]=pos_desir
print('T_desir', T_desir)


M_se = np.array([[1.0, 0.0, 0.0, 0.088],[0.0, -1.0, 0.0, 0.0],[0.0, 0.0, -1.0, 0.823],[0.0, 0.0, 0.0, 1.0]])
S = r.Screws()
q_init = np.zeros((7,1))
q_init[:,0]=(r.JointLimit()[1]+r.JointLimit()[0])/2

#np.array([[0.0, 0.0, -1.0, 0.4],[0.0, 1.0, 0.0, 0.0],[1.0, 0.0, 0.0, 0.737],[0.0, 0.0, 0.0, 1.0]])
q_desir = r.InvKinematics2(S, q_init, M_se, T_desir)
#q_desir = q_t
q_desir = r.Joint_Limit_Check(r.JointLimit()[1],r.JointLimit()[0],q_desir)
T_sol=r.ForwardKinematics(S,q_desir,M_se)
print('T_sol', T_sol)

T_desir2 = T_desir
T_desir2[2,3] += 0.1
T_desir2[0,3] -= 0.1
print('T_desir2',T_desir2)
q_desir2 = r.InvKinematics2(S, q_init, M_se, T_desir2)
q_desir2 = r.Joint_Limit_Check(r.JointLimit()[1],r.JointLimit()[0],q_desir2)
print('q_desir2', q_desir2)
#q_desir2 = np.zeros((7,1))
#q_desir2[:,0] = [0.0, np.pi/6, 0.0, 0.0, 0.0, np.pi*5/8,0.0]

model = load_model_from_path("franka_sim/franka_panda.xml")
sim = MjSim(model)

viewer = MjViewer(sim)


reached = np.zeros(7)
reached_all = 0

reached2 = np.zeros(7)
reached_all2 = False

grasped = False

sim_state = sim.get_state()
object_height_init = sim_state.qpos[11]

while True:
	for step in range(10000):
		if not reached_all:	
			sim_state = sim.get_state()
			for j in range(np.size(q_desir,axis=0)):
				if abs(sim_state.qpos[j]-q_desir[j,0])<1e-3:
					reached[j] = 1
				else:
					reached[j] = 0
				if not reached[j]:
					sim.data.ctrl[j] = j* (q_desir[j,0]-sim_state.qpos[j])/abs(q_desir[j,0]-sim_state.qpos[j])
				else:
					sim.data.ctrl[j] = 0.0
			reached_all = 1
			for i in range(np.size(q_desir,axis=0)):
				reached_all *= reached[i]

		#print('pose reached',reached_all, ', contact', contact_detect(sim, 'object'),', contact geom', sim.model.geom_id2name(sim.data.contact[7].geom2), ', dist', sim.data.contact[7].dist, ', ctrl', 	sim.data.ctrl[3] )
		
		if not reached_all:
			finger_open(sim)
			#print('open')
		elif not reached_all and not contact_detect(sim, 'object'):
			finger_close(sim)
			#print('close')
		else:
			finger_stop(sim)
			#print('stop')
			grasped = True

		if grasped:
			if not reached_all2:	
				sim_state = sim.get_state()
				for j in range(np.size(q_desir2,axis=0)):
					if abs(sim_state.qpos[j]-q_desir2[j,0])<1e-1:
						reached2[j] = 1
					else:
						reached2[j] = 0

					if not reached2[j]:
						sim.data.ctrl[j] = j* (q_desir2[j,0]-sim_state.qpos[j])/abs(q_desir2[j,0]-sim_state.qpos[j])
					else:
						sim.data.ctrl[j] = 0.0
				if sim_state.qpos[11] - object_height_init > 0.1:
					reached_all2 = True

			else:
				sim.data.ctrl[:]=0.0
		sim_state = sim.get_state()		
		print(sim_state.qpos[0:9])

		sim.step()
		viewer.render()


	if os.getenv('TESTING') is not None:
        	break
