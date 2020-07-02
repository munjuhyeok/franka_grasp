import numpy as np
from numpy import linalg as LA
from scipy.linalg import logm
import snurobotics.Robotics as robo

# Fifth order polynomial

def FifthOrderPolynomial(t_total, timestep):
    # Total step
    n_step = np.int(t_total/timestep)

    # Initialize polynomial
    Poly = np.zeros((1,n_step))
    t = np.linspace(0.0, n_step*timestep, num=n_step)
    Poly = 6 * np.power(t,5) / np.power(t_total,5) - 15 * np.power(t,4) / np.power(t_total,4) + 10 * np.power(t,3) / np.power(t_total,3)
    Poly = Poly.reshape((1,n_step))
    
    # print("Poly.shape:" , Poly.shape)
    return Poly
    
    # for i in range(n_step):
    #     t = (i+1)*timestep
    #     Poly[0,i] = 6 * np.power(t,5) / np.power(t_total,5) - 15 * np.power(t,4) / np.power(t_total,4) + 10 * np.power(t,3) / np.power(t_total,3)
    # return Poly

def dFifthOrderPolynomial(t_total, timestep):
    # Total step
    n_step = np.int(t_total/timestep)

    # Initialize polynomial
    Poly = np.zeros((1,n_step))
    t = np.linspace(0.0, n_step*timestep, num=n_step)
    Poly = 30 * np.power(t,4) / np.power(t_total,5) - 60 * np.power(t,3) / np.power(t_total,4) + 30 * np.power(t,2) / np.power(t_total,3)
    Poly = Poly.reshape((1,n_step))
    
    return Poly
    # for i in range(n_step):
    #     t = (i+1)*timestep
    #     Poly[0,i] = 30 * np.power(t,4) / np.power(t_total,5) - 60 * np.power(t,3) / np.power(t_total,4) + 30 * np.power(t,2) / np.power(t_total,3)
    # return Poly

def ddFifthOrderPolynomial(t_total, timestep):
    # Total step
    n_step = np.int(t_total/timestep)

    # Initialize polynomial
    Poly = np.zeros((1,n_step))
    t = np.linspace(0.0, n_step*timestep, num=n_step)
    Poly = 120 * np.power(t,3) / np.power(t_total,5) - 180 * np.power(t,2) / np.power(t_total,4) + 60 * np.power(t,1) / np.power(t_total,3)
    Poly = Poly.reshape((1,n_step))
    return Poly

    
    # for i in range(n_step):
    #     t = (i)*timestep
    #     Poly[0,i] = 120 * np.power(t,3) / np.power(t_total,5) - 180 * np.power(t,2) / np.power(t_total,4) + 60 * np.power(t,1) / np.power(t_total,3)
    # return Poly

# Previous trajectory planning
def TrajPlanning(T_init, T_final, t_total, timestep):
    # Extract rotation matrix and position
    R_init = T_init[0:3,0:3]
    p_init = T_init[0:3,3]
    R_final = T_final[0:3,0:3]
    p_final = T_final[0:3,3]     
    
    # Set via point and via time
    affordance = np.array([0.0,0.0,0.0])
    t_via = t_total * 2 / 5
    p_via = p_final + affordance

    # Omega for planning rotation matrix
    R = LA.inv(R_init).dot(R_final)
    w = robo.SO3_to_Omega(R)
    w_normal = w / LA.norm(w)

    # Total step & via step
    n_step = np.int(t_total/timestep)
    n_via = np.int(t_via/timestep)

    # Initialize trajctories
    p_plan = np.zeros((3,n_step))
    p_plan_dot = np.zeros((3,n_step))
    p_plan_dotdot = np.zeros((3,n_step))
    R_plan = np.zeros((3,3,n_step))

    # Planning from initial to via
    p_plan[0,0:n_via] = p_init[0] + FifthOrderPolynomial(t_via, timestep).dot(p_via[0]-p_init[0])
    p_plan[1,0:n_via] = p_init[1] + FifthOrderPolynomial(t_via, timestep).dot(p_via[1]-p_init[1])
    p_plan[2,0:n_via] = p_init[2] + FifthOrderPolynomial(t_via, timestep).dot(p_via[2]-p_init[2])
    
    p_plan_dot[0,0:n_via] = dFifthOrderPolynomial(t_via, timestep).dot(p_via[0]-p_init[0])
    p_plan_dot[1,0:n_via] = dFifthOrderPolynomial(t_via, timestep).dot(p_via[1]-p_init[1])
    p_plan_dot[2,0:n_via] = dFifthOrderPolynomial(t_via, timestep).dot(p_via[2]-p_init[2])

    p_plan_dotdot[0,0:n_via] = ddFifthOrderPolynomial(t_via, timestep).dot(p_via[0]-p_init[0])
    p_plan_dotdot[1,0:n_via] = ddFifthOrderPolynomial(t_via, timestep).dot(p_via[1]-p_init[1])
    p_plan_dotdot[2,0:n_via] = ddFifthOrderPolynomial(t_via, timestep).dot(p_via[2]-p_init[2])

    for i in range(n_via) :
        R_plan[:,:,i] = R_init.dot(robo.Omega_to_SO3(w_normal, LA.norm(w) * FifthOrderPolynomial(t_via, timestep)[0,i]))

    # Planning from via to final
    p_plan[0,n_via:n_step] = p_via[0] + FifthOrderPolynomial(t_total-t_via, timestep).dot(p_final[0]-p_via[0])
    p_plan[1,n_via:n_step] = p_via[1] + FifthOrderPolynomial(t_total-t_via, timestep).dot(p_final[1]-p_via[1])
    p_plan[2,n_via:n_step] = p_via[2] + FifthOrderPolynomial(t_total-t_via, timestep).dot(p_final[2]-p_via[2])

    p_plan_dot[0,n_via:n_step] = dFifthOrderPolynomial(t_total-t_via, timestep).dot(p_final[0]-p_via[0])
    p_plan_dot[1,n_via:n_step] = dFifthOrderPolynomial(t_total-t_via, timestep).dot(p_final[1]-p_via[1])
    p_plan_dot[2,n_via:n_step] = dFifthOrderPolynomial(t_total-t_via, timestep).dot(p_final[2]-p_via[2])

    p_plan_dotdot[0,n_via:n_step] = ddFifthOrderPolynomial(t_total-t_via, timestep).dot(p_final[0]-p_via[0])
    p_plan_dotdot[1,n_via:n_step] = ddFifthOrderPolynomial(t_total-t_via, timestep).dot(p_final[1]-p_via[1])
    p_plan_dotdot[2,n_via:n_step] = ddFifthOrderPolynomial(t_total-t_via, timestep).dot(p_final[2]-p_via[2])

    for k in range(n_step-n_via) :
        R_plan[:,:,n_via + k] = R_plan[:,:,n_via-1]

    return [p_plan, p_plan_dot, p_plan_dotdot, R_plan]

# Modified trajectory planning
def TrajPlanning2(T_init, T_final, t_total, timestep):
    # Extract rotation matrix and position
    R_init = T_init[0:3,0:3]
    p_init = T_init[0:3,3]
    R_final = T_final[0:3,0:3]
    p_final = T_final[0:3,3]     
    
    # Set via point and via time
    affordance = np.array([-0.3,0.0,0.0])
    t_via = t_total * 1 / 5
    t_via2 = t_total * 2 / 5
    p_via = p_final + affordance

    # Omega for planning rotation matrix
    R = LA.inv(R_init).dot(R_final)
    w = robo.SO3_to_Omega(R)
    w_normal = w / LA.norm(w)

    # Total step & via step
    n_step = np.int(t_total/timestep)
    n_via = np.int(t_via/timestep)
    n_via2 = np.int(t_via2/timestep)

    # Initialize trajctories
    p_plan = np.zeros((3,n_step))
    p_plan_dot = np.zeros((3,n_step))
    p_plan_dotdot = np.zeros((3,n_step))
    R_plan = np.zeros((3,3,n_step))

    # Planning from initial to via
    p_plan[0,0:n_via] = p_init[0] + FifthOrderPolynomial(t_via, timestep).dot(p_via[0]-p_init[0])
    p_plan[1,0:n_via] = p_init[1] + FifthOrderPolynomial(t_via, timestep).dot(p_via[1]-p_init[1])
    p_plan[2,0:n_via] = p_init[2] + FifthOrderPolynomial(t_via, timestep).dot(p_via[2]-p_init[2])
    
    p_plan_dot[0,0:n_via] = dFifthOrderPolynomial(t_via, timestep).dot(p_via[0]-p_init[0])
    p_plan_dot[1,0:n_via] = dFifthOrderPolynomial(t_via, timestep).dot(p_via[1]-p_init[1])
    p_plan_dot[2,0:n_via] = dFifthOrderPolynomial(t_via, timestep).dot(p_via[2]-p_init[2])

    p_plan_dotdot[0,0:n_via] = ddFifthOrderPolynomial(t_via, timestep).dot(p_via[0]-p_init[0])
    p_plan_dotdot[1,0:n_via] = ddFifthOrderPolynomial(t_via, timestep).dot(p_via[1]-p_init[1])
    p_plan_dotdot[2,0:n_via] = ddFifthOrderPolynomial(t_via, timestep).dot(p_via[2]-p_init[2])

    for i in range(n_via) :
        R_plan[:,:,i] = R_init.dot(robo.Omega_to_SO3(w_normal, LA.norm(w) * FifthOrderPolynomial(t_via, timestep)[0,i]))

    # Planning from via to via2
    p_plan[0,n_via:n_via2] = p_via[0] + FifthOrderPolynomial(t_via2-t_via, timestep).dot(p_final[0]-p_via[0])
    p_plan[1,n_via:n_via2] = p_via[1] + FifthOrderPolynomial(t_via2-t_via, timestep).dot(p_final[1]-p_via[1])
    p_plan[2,n_via:n_via2] = p_via[2] + FifthOrderPolynomial(t_via2-t_via, timestep).dot(p_final[2]-p_via[2])

    p_plan_dot[0,n_via:n_via2] = dFifthOrderPolynomial(t_via2-t_via, timestep).dot(p_final[0]-p_via[0])
    p_plan_dot[1,n_via:n_via2] = dFifthOrderPolynomial(t_via2-t_via, timestep).dot(p_final[1]-p_via[1])
    p_plan_dot[2,n_via:n_via2] = dFifthOrderPolynomial(t_via2-t_via, timestep).dot(p_final[2]-p_via[2])

    p_plan_dotdot[0,n_via:n_via2] = ddFifthOrderPolynomial(t_via2-t_via, timestep).dot(p_final[0]-p_via[0])
    p_plan_dotdot[1,n_via:n_via2] = ddFifthOrderPolynomial(t_via2-t_via, timestep).dot(p_final[1]-p_via[1])
    p_plan_dotdot[2,n_via:n_via2] = ddFifthOrderPolynomial(t_via2-t_via, timestep).dot(p_final[2]-p_via[2])

    for k in range(n_via2-n_via) :
        R_plan[:,:,n_via + k] = R_final

    # Planning from via2 to final
    p_plan[0,n_via2:n_step] = p_plan[0,n_via2-1]
    p_plan[1,n_via2:n_step] = p_plan[1,n_via2-1]
    p_plan[2,n_via2:n_step] = p_plan[2,n_via2-1]

    p_plan_dot[0,n_via2:n_step] = 0
    p_plan_dot[1,n_via2:n_step] = 0
    p_plan_dot[2,n_via2:n_step] = 0

    p_plan_dotdot[0,n_via2:n_step] = 0
    p_plan_dotdot[1,n_via2:n_step] = 0
    p_plan_dotdot[2,n_via2:n_step] = 0

    for k in range(n_step-n_via2) :
        R_plan[:,:,n_via2 + k] = R_plan[:,:,n_via2-1]

    return [p_plan, p_plan_dot, p_plan_dotdot, R_plan]


# Modified trajectory planning 2
def TrajPlanning3(T_init, T_final, t_total, timestep):
    # Extract rotation matrix and position
    R_init = T_init[0:3,0:3]
    p_init = T_init[0:3,3]
    R_final = T_final[0:3,0:3]
    p_final = T_final[0:3,3]     
    
    # Set via point and via time
    affordance = np.array([0.0,0.0,0.04])   # hole depth 0.04
    t_via = t_total * 4 / 10
    t_via2 = t_total * 5 / 10
    t_via3 = t_total * 9 / 10
    p_via = p_final + affordance

    # Omega for planning rotation matrix
    R = LA.inv(R_init).dot(R_final)
    w = robo.SO3_to_Omega(R)
    w_normal = w / LA.norm(w)

    # Total step & via step
    n_step = np.int(t_total/timestep)
    n_via = np.int(t_via/timestep)
    n_via2 = np.int(t_via2/timestep)
    n_via3 = np.int(t_via3/timestep)

    # Initialize trajctories
    p_plan = np.zeros((3,n_step))
    p_plan_dot = np.zeros((3,n_step))
    p_plan_dotdot = np.zeros((3,n_step))
    R_plan = np.zeros((3,3,n_step))

    time_series1 = FifthOrderPolynomial(n_via*timestep, timestep)
    dtime_series1 = dFifthOrderPolynomial(n_via*timestep, timestep)
    ddtime_series1 = ddFifthOrderPolynomial(n_via*timestep, timestep)

    # Planning from initial to via
    p_plan[0,0:n_via] = p_init[0] + time_series1.dot(p_via[0]-p_init[0])
    p_plan[1,0:n_via] = p_init[1] + time_series1.dot(p_via[1]-p_init[1])
    p_plan[2,0:n_via] = p_init[2] + time_series1.dot(p_via[2]-p_init[2])
    
    p_plan_dot[0,0:n_via] = dtime_series1.dot(p_via[0]-p_init[0])
    p_plan_dot[1,0:n_via] = dtime_series1.dot(p_via[1]-p_init[1])
    p_plan_dot[2,0:n_via] = dtime_series1.dot(p_via[2]-p_init[2])

    p_plan_dotdot[0,0:n_via] = ddtime_series1.dot(p_via[0]-p_init[0])
    p_plan_dotdot[1,0:n_via] = ddtime_series1.dot(p_via[1]-p_init[1])
    p_plan_dotdot[2,0:n_via] = ddtime_series1.dot(p_via[2]-p_init[2])

    for i in range(n_via) :
        R_plan[:,:,i] = R_init.dot(robo.Omega_to_SO3(w_normal, LA.norm(w) * time_series1[0,i]))

    # # Planning from initial to via
    # p_plan[0,0:n_via] = p_init[0] + FifthOrderPolynomial(t_via, timestep).dot(p_via[0]-p_init[0])
    # p_plan[1,0:n_via] = p_init[1] + FifthOrderPolynomial(t_via, timestep).dot(p_via[1]-p_init[1])
    # p_plan[2,0:n_via] = p_init[2] + FifthOrderPolynomial(t_via, timestep).dot(p_via[2]-p_init[2])
    
    # p_plan_dot[0,0:n_via] = dFifthOrderPolynomial(t_via, timestep).dot(p_via[0]-p_init[0])
    # p_plan_dot[1,0:n_via] = dFifthOrderPolynomial(t_via, timestep).dot(p_via[1]-p_init[1])
    # p_plan_dot[2,0:n_via] = dFifthOrderPolynomial(t_via, timestep).dot(p_via[2]-p_init[2])

    # p_plan_dotdot[0,0:n_via] = ddFifthOrderPolynomial(t_via, timestep).dot(p_via[0]-p_init[0])
    # p_plan_dotdot[1,0:n_via] = ddFifthOrderPolynomial(t_via, timestep).dot(p_via[1]-p_init[1])
    # p_plan_dotdot[2,0:n_via] = ddFifthOrderPolynomial(t_via, timestep).dot(p_via[2]-p_init[2])

    # for i in range(n_via) :
    #     R_plan[:,:,i] = R_init.dot(robo.Omega_to_SO3(w_normal, LA.norm(w) * FifthOrderPolynomial(t_via, timestep)[0,i]))

    # Planning from via to via2
    p_plan[0,n_via:n_via2] = p_via[0]
    p_plan[1,n_via:n_via2] = p_via[1]
    p_plan[2,n_via:n_via2] = p_via[2]

    p_plan_dot[0,n_via:n_via2] = 0
    p_plan_dot[1,n_via:n_via2] = 0
    p_plan_dot[2,n_via:n_via2] = 0

    p_plan_dotdot[0,n_via:n_via2] = 0
    p_plan_dotdot[1,n_via:n_via2] = 0
    p_plan_dotdot[2,n_via:n_via2] = 0

    for k in range(n_via2-n_via) :
        R_plan[:,:,n_via + k] = R_final

    time_series2 = FifthOrderPolynomial((n_via3-n_via2)*timestep, timestep)
    dtime_series2 = dFifthOrderPolynomial((n_via3-n_via2)*timestep, timestep)
    ddtime_series2 = ddFifthOrderPolynomial((n_via3-n_via2)*timestep, timestep)

    # Planning from via2 to via3
    p_plan[0,n_via2:n_via3] = p_via[0] + time_series2.dot(p_final[0]-p_via[0])
    p_plan[1,n_via2:n_via3] = p_via[1] + time_series2.dot(p_final[1]-p_via[1])
    p_plan[2,n_via2:n_via3] = p_via[2] + time_series2.dot(p_final[2]-p_via[2])

    p_plan_dot[0,n_via2:n_via3] = dtime_series2.dot(p_final[0]-p_via[0])
    p_plan_dot[1,n_via2:n_via3] = dtime_series2.dot(p_final[1]-p_via[1])
    p_plan_dot[2,n_via2:n_via3] = dtime_series2.dot(p_final[2]-p_via[2])

    p_plan_dotdot[0,n_via2:n_via3] = ddtime_series2.dot(p_final[0]-p_via[0])
    p_plan_dotdot[1,n_via2:n_via3] = ddtime_series2.dot(p_final[1]-p_via[1])
    p_plan_dotdot[2,n_via2:n_via3] = ddtime_series2.dot(p_final[2]-p_via[2])


    for k in range(n_via3-n_via2) :
        R_plan[:,:,n_via2 + k] = R_final

    # Planning from via3 to final
    p_plan[0,n_via3:n_step] = p_final[0]
    p_plan[1,n_via3:n_step] = p_final[1]
    p_plan[2,n_via3:n_step] = p_final[2]

    p_plan_dot[0,n_via3:n_step] = 0
    p_plan_dot[1,n_via3:n_step] = 0
    p_plan_dot[2,n_via3:n_step] = 0

    p_plan_dotdot[0,n_via3:n_step] = 0
    p_plan_dotdot[1,n_via3:n_step] = 0
    p_plan_dotdot[2,n_via3:n_step] = 0

    for k in range(n_step-n_via3) :
        R_plan[:,:,n_via3 + k] = R_final

    return [p_plan, p_plan_dot, p_plan_dotdot, R_plan]


# Trajectory planning in configuration space
def TrajPlanningConfig(Screws, q_init, T_final, t_total, M_EF, timestep):
    
    # Extract rotation matrix and position
    p_final = T_final[0:3,3]  

    # Set via point and via time
    affordance = np.array([0.0,0.0,0.04])
    t_via = t_total * 2 / 5
    t_via2 = t_total * 3 / 5
    p_via = p_final + affordance

    # Solve inverse kinematics
    T_via = np.eye((4))
    T_via[0:3,0:3] = T_final[0:3,0:3]
    T_via[0:3,3] = p_via
    # q_0 = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
    q_via = robo.InvKinematics(Screws, q_init, M_EF, T_via)
    q_final = robo.InvKinematics(Screws, q_via, M_EF, T_final)

    # Modifying joint values for trajectory planning
    q_via_modified = robo.ModifyingJointValue(q_init,q_via)
    q_final_modified = robo.ModifyingJointValue(q_via,q_final)

    # Total step & via step
    n_step = np.int(t_total/timestep)
    n_via = np.int(t_via/timestep)
    n_via2 = np.int(t_via2/timestep)

    # Initialize trajctories
    q_plan = np.zeros((6,n_step))
    q_plan_dot = np.zeros((6,n_step))
    q_plan_dotdot = np.zeros((6,n_step))
    
    time_series1 = FifthOrderPolynomial(t_via, timestep)
    dtime_series1 = dFifthOrderPolynomial(t_via, timestep)
    ddtime_series1 = ddFifthOrderPolynomial(t_via, timestep)

    # Planning from initial to via
    for i in range(6):
        q_plan[i,0:n_via] = q_init[i] + time_series1 * (q_via_modified[i]-q_init[i])
        q_plan_dot[i,0:n_via] = dtime_series1 * (q_via_modified[i]-q_init[i])
        q_plan_dotdot[i,0:n_via] = ddtime_series1 * (q_via_modified[i]-q_init[i])

    time_series2 = FifthOrderPolynomial(t_via2-t_via, timestep)
    dtime_series2 = dFifthOrderPolynomial(t_via2-t_via, timestep)
    ddtime_series2 = ddFifthOrderPolynomial(t_via2-t_via, timestep)
    # Planning from via to via2
    for i in range(6):
        q_plan[i,n_via:n_via2] = q_via_modified[i] + time_series2 * (q_final_modified[i]-q_via_modified[i])
        q_plan_dot[i,n_via:n_via2] = dtime_series2 * (q_final_modified[i]-q_via_modified[i])
        q_plan_dotdot[i,n_via:n_via2] = ddtime_series2 * (q_final_modified[i]-q_via_modified[i])

    # Planning from via2 to final
    for i in range(6):
        q_plan[i,n_via2:n_step] = q_final_modified[i]
        q_plan_dot[i,n_via2:n_step] = 0
        q_plan_dotdot[i,n_via2:n_step] = 0

    return [q_plan, q_plan_dot, q_plan_dotdot]


# Previous trajectory planning
def TrajPlanning4(T_init, T_final, t_total, timestep):
    # Extract rotation matrix and position
    R_init = T_init[0:3,0:3]
    p_init = T_init[0:3,3]
    R_final = T_final[0:3,0:3]
    p_final = T_final[0:3,3]     
    
    # Set via point and via time
    affordance = np.array([0.0,0.0,0.05])
    t_via = t_total * 2 / 10
    p_via = p_final + affordance

    # Omega for planning rotation matrix
    R = LA.inv(R_init).dot(R_final)
    w = robo.SO3_to_Omega(R)
    w_normal = w / LA.norm(w)

    # Total step & via step
    n_step = np.int(t_total/timestep)
    n_via = np.int(t_via/timestep)

    # Initialize trajctories
    p_plan = np.zeros((3,n_step))
    p_plan_dot = np.zeros((3,n_step))
    p_plan_dotdot = np.zeros((3,n_step))
    R_plan = np.zeros((3,3,n_step))
    T_plan = np.zeros((4,4,n_step))
    S_plan = np.zeros((6,1,n_step-1))
    A_plan = np.zeros((6,1,n_step-1))

    time_series1 = FifthOrderPolynomial(n_via*timestep, timestep)
    dtime_series1 = dFifthOrderPolynomial(n_via*timestep, timestep)
    ddtime_series1 = ddFifthOrderPolynomial(n_via*timestep, timestep)

    # Planning from initial to via
    p_plan[0,0:n_via] = p_init[0] + time_series1.dot(p_via[0]-p_init[0])
    p_plan[1,0:n_via] = p_init[1] + time_series1.dot(p_via[1]-p_init[1])
    p_plan[2,0:n_via] = p_init[2] + time_series1.dot(p_via[2]-p_init[2])
    
    p_plan_dot[0,0:n_via] = dtime_series1.dot(p_via[0]-p_init[0])
    p_plan_dot[1,0:n_via] = dtime_series1.dot(p_via[1]-p_init[1])
    p_plan_dot[2,0:n_via] = dtime_series1.dot(p_via[2]-p_init[2])

    p_plan_dotdot[0,0:n_via] = ddtime_series1.dot(p_via[0]-p_init[0])
    p_plan_dotdot[1,0:n_via] = ddtime_series1.dot(p_via[1]-p_init[1])
    p_plan_dotdot[2,0:n_via] = ddtime_series1.dot(p_via[2]-p_init[2])

    for i in range(n_via) :
        R_plan[:,:,i] = R_init.dot(robo.Omega_to_SO3(w_normal, LA.norm(w) * FifthOrderPolynomial(t_via, timestep)[0,i]))

    time_series2 = FifthOrderPolynomial((n_step-n_via)*timestep, timestep)
    dtime_series2 = dFifthOrderPolynomial((n_step-n_via)*timestep, timestep)
    ddtime_series2 = ddFifthOrderPolynomial((n_step-n_via)*timestep, timestep)

    # Planning from via to final
    p_plan[0,n_via:n_step] = p_via[0] + time_series2.dot(p_final[0]-p_via[0])
    p_plan[1,n_via:n_step] = p_via[1] + time_series2.dot(p_final[1]-p_via[1])
    p_plan[2,n_via:n_step] = p_via[2] + time_series2.dot(p_final[2]-p_via[2])

    p_plan_dot[0,n_via:n_step] = dtime_series2.dot(p_final[0]-p_via[0])
    p_plan_dot[1,n_via:n_step] = dtime_series2.dot(p_final[1]-p_via[1])
    p_plan_dot[2,n_via:n_step] = dtime_series2.dot(p_final[2]-p_via[2])

    p_plan_dotdot[0,n_via:n_step] = ddtime_series2.dot(p_final[0]-p_via[0])
    p_plan_dotdot[1,n_via:n_step] = ddtime_series2.dot(p_final[1]-p_via[1])
    p_plan_dotdot[2,n_via:n_step] = ddtime_series2.dot(p_final[2]-p_via[2])

    for k in range(n_step-n_via) :
        R_plan[:,:,n_via + k] = R_plan[:,:,n_via-1]

    for i in range(n_step) :
        T_plan[0:3, 0:3, i] = R_plan[:, :, i]
        T_plan[0:3, 3, i] = p_plan[:, i]
        T_plan[3, 3, i] = 1
    
    for i in range(n_step-1) : 
        S_bracket = logm(robo.Tinv(T_plan[:, :, i]).dot(T_plan[:, :, i+1])) / timestep
        S_plan[0, 0, i] = S_bracket[2, 1]
        S_plan[1, 0, i] = S_bracket[0, 2]
        S_plan[2, 0, i] = S_bracket[1, 0]
        S_plan[3:6, 0, i] = S_bracket[0:3, 3]

    A_plan[:, 0, 0] = S_plan[:, 0, 0]
    for i in range(n_step-2) :
        A_plan[:, 0, i+1] = (S_plan[:, 0, i+1] - S_plan[:, 0, i]) / timestep


    return [p_plan, p_plan_dot, p_plan_dotdot, R_plan, S_plan, A_plan]
