from casadi import *

T = 10 # Time horizon
N = 50 # number of control intervals

# Declare model variables
x1 = MX.sym('x1')
x2 = MX.sym('x2')
x3 = MX.sym('x3')
x4 = MX.sym('x4')
x = vertcat(x1, x2, x3, x4)
u1 = MX.sym('u1')
u2 = MX.sym('u2')
u = vertcat(u1, u2)

# Model equations
# xdot = vertcat((1-x2**2)*x1 - x2 + u, x1)
xdot = vertcat(x3, x4, u1, u2)

x1g = 3.0
x2g = 3.0
# Objective term
L = (x1-x1g)**2 + (x2-x2g)**2 + u1**2 + u2**2

# Formulate discrete time dynamics
if False:
   # CVODES from the SUNDIALS suite
   dae = {'x':x, 'p':u, 'ode':xdot, 'quad':L}
   F = integrator('F', 'cvodes', dae, 0, T/N)
else:
   # Fixed step Runge-Kutta 4 integrator
   M = 4 # RK4 steps per interval
   DT = T/N/M
   f = Function('f', [x, u], [xdot, L])
   X0 = MX.sym('X0', 4)
   U = MX.sym('U', 2)
   X = X0
   Q = 0
   for j in range(M):
       k1, k1_q = f(X, U)
       k2, k2_q = f(X + DT/2 * k1, U)
       k3, k3_q = f(X + DT/2 * k2, U)
       k4, k4_q = f(X + DT * k3, U)
       X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
       Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
   F = Function('F', [X0, U], [X, Q],['x0','p'],['xf','qf'])

# Evaluate at a test point
# Fk = F(x0=[0.2,0.3,0.0,0.0],p=[0.4, 0.4])
# print(Fk['xf'])
# print(Fk['qf'])
# # Start with an empty NLP
# w=[]
# w0 = []
# lbw = []
# ubw = []
# J = 0
# g=[]
# lbg = []
# ubg = []



def Optim_solver(x0):
    # Start with an empty NLP
    w=[]
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g=[]
    lbg = []
    ubg = []

    

    # Xk = MX(x0[0])              # MX([0, 0, 0, 0])

    Xk = MX.sym('X_' + str(0), 4)
    w += [Xk]
    lbw += x0
    ubw += x0
    w0 += x0

    for k in range(T):
        Uk = MX.sym('U_' + str(k), 2)
        w += [Uk]
        lbw += [-10, -10]                
        ubw += [10, 10]                     
        w0 += [0, 0]

        # Integrate till the end of the interval
        Fk = F(x0=Xk, p=Uk)
        Xk_end = Fk['xf']
        J = J+Fk['qf']
        
        Xk = MX.sym('X_' + str(k+1), 4)
        w += [Xk]
        w0 += [0, 0, 0, 0]
        lbw += [-inf, -inf, -inf, -inf]
        ubw += [inf, inf, inf, inf]

        # Add inequality constraint
        g += [Xk_end - Xk]
        # print(Xk[0])
        lbg += [0, 0, 0, 0]
        ubg += [0, 0, 0, 0]
    prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
    solver = nlpsol('solver', 'ipopt', prob)

    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol['x']
    return w_opt

# # Formulate the NLP
# Xk = MX([0, 0, 0, 0])
# for k in range(N):
#     # New NLP variable for the control
#     Uk = MX.sym('U_' + str(k))
#     w += [Uk]
#     lbw += [-1]
#     ubw += [1]
#     w0 += [0]

#     # Integrate till the end of the interval
#     Fk = F(x0=Xk, p=Uk)
#     Xk = Fk['xf']
#     J=J+Fk['qf']

#     # Add inequality constraint
#     g += [Xk[0]]
#     lbg += [-.25]
#     ubg += [inf]

# # Create an NLP solver
# prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
# solver = nlpsol('solver', 'ipopt', prob)

# # Solve the NLP
# sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
# w_opt = sol['x']

# Plot the solution
                              # the control input is the optimization variable
x_opt = [[0, 2, 0, 0]]                      # We must start with a feasible state
for k in range(N):                          # Loop over control intervals
    u_opt = Optim_solver(x_opt[-1])             # Solve the optimization problem
    Fk = F(x0=x_opt[-1], p=u_opt[4:6])        # Get the optimal solution
    print(u_opt[4:6])
    x_opt += [Fk['xf'].full()]              # update the state trajectory
x1_opt = vcat([r[0] for r in x_opt])        # state x1
x2_opt = vcat([r[1] for r in x_opt])        # state x2

tgrid = [T/N*k for k in range(N+1)]         # time grid

import matplotlib.pyplot as plt
plt.figure(1)
plt.clf()
plt.plot(tgrid, x1_opt, '--')
plt.plot(tgrid, x2_opt, '-')
# plt.step(tgrid, vertcat(DM.nan(1), u_opt), '-.')
plt.xlabel('t')
plt.legend(['x1','x2','u'])
plt.grid()
plt.show()