from cvxpy import *
import numpy as np
import scipy as sp
from scipy import sparse

# Discrete time model of a quadcopter

Ad = sparse.csc_matrix([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
    ])
Bd = sparse.csc_matrix([
    [0, 0],
    [0, 0],
    [1, 0],
    [0, 1]
    ])
[nx, nu] = Bd.shape

# Constraints
u0 = 1.0


umin = np.array([-10, -10])
umax = np.array([10, 10])
xmin = np.array([-np.inf, -np.inf, -np.inf, -np.inf])
xmax = np.array([np.inf, np.inf, np.inf, np.inf])

# Objective function
Q = sparse.diags([1.0, 1.0, 1.0, 1.0])
QN = Q
R = 0.1*sparse.eye(2)

# Initial and reference states
x0 = np.zeros(4)
xr = np.array([3.0, 3.0, 0.0, 0.0])

# Prediction horizon
N = 10

# Time step
dt = 0.1

# Define problem
u = Variable((nu, N))
x = Variable((nx, N+1))
x_init = Parameter(nx)

def MPC_solver(x_init):
    objective = 0
    constraints = [x[:,0] == x_init]
    for k in range(N):
        objective += quad_form(x[:,k] - xr, Q) + quad_form(u[:,k], R)
        constraints += [x[:,k+1] == (Ad@x[:,k] + Bd@u[:,k]) * dt + x[:,k]]
        # constraints += [xmin <= x[:,k], x[:,k] <= xmax]
        # constraints += [(x[:,k][0]-1.5)**2 + (x[:,k][1]-1.5)**2 >= 1]
        constraints += [umin <= u[:,k], u[:,k] <= umax]
    objective += quad_form(x[:,N] - xr, QN) * 100
    prob = Problem(Minimize(objective), constraints)
    prob.solve(solver=OSQP, warm_start=True)
    # print(u.value)
    return u[:,0].value

# Simulate in closed loop
nsim = 150
Traj = []
for i in range(nsim):
    x_init.value = x0
    u_optimal = MPC_solver(x0)
    x0 = x0 + (Ad.dot(x0) + Bd.dot(u_optimal)) * dt 
    Traj.append(x0[:])

print("Final state: {}".format(x0))
import matplotlib.pyplot as plt
plt.plot([x[0] for x in Traj], [x[1] for x in Traj], 'b-')
plt.show()