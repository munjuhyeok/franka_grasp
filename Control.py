import numpy as np


def finger_close(sim):
	left = 7
	right = 8
	sim.data.ctrl[left] = 0.0
	sim.data.ctrl[right] = 0.0

def finger_open(sim):
	left = 7
	right = 8
	sim.data.ctrl[left] = 0.04
	sim.data.ctrl[right] = 0.04


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

def check_pose_reached(q, q_desir):
	EPS = 1e-2
	reached_joints=np.zeros(7)
	for i in range(np.size(q_desir,axis=0)):
		if abs(q[i]-q_desir[i,0])<EPS:
			reached_joints[i] = 1
		else:
			reached_joints[i] = 0

	reached_pose = True
	for i in range(np.size(q_desir,axis=0)):
		reached_pose = reached_pose and reached_joints[i]

	return reached_pose

def controller(sim, q, q_des,e_prev,e_int,step,start_step,time_step):
	K_p = 0.001
	K_d = 0.1
	K_i = 1
	e = np.zeros(7)
	for i in range(np.size(q_des,axis=0)):
		e[i] = q_des[i,0]-q[i]		
		if step-start_step < 1:
			sim.data.ctrl[i] = K_p*e[i]
		else: 
			e_d = (e[i]-e_prev[i])/time_step
			e_i = e_int[i] + e[i]*time_step
			sim.data.ctrl[i] = K_p*e[i] + K_d * e_d + K_i * e_i
	return e
