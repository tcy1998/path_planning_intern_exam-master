from casadi import *

N = 10 # number of control intervals



# ---- dynamic constraints --------
f = lambda x,u: vertcat(x[2], x[3], u[0], u[1]) # dx/dt = f(x,u)

dt = 0.05 # length of a control interval
def solver_mpc(x_init, y_init, vx_init, vy_init):

    opti = Opti() # Optimization problem

    # ---- decision variables ---------
    X = opti.variable(4,N+1) # state trajectory
    pos_x = X[0,:]
    pos_y = X[1,:]
    vel_x = X[2,:]
    vel_y = X[3,:]

    U = opti.variable(2,N)   # control trajectory (throttle)
    acc_x = U[0,:]
    acc_y = U[1,:]


    # Objective term
    L = sumsqr(X) + sumsqr(U) # sum of QP terms

    # ---- objective          ---------
    opti.minimize(L) # race in minimal time 

    for k in range(N): # loop over control intervals
        # Runge-Kutta 4 integration
        k1 = f(X[:,k],         U[:,k])
        k2 = f(X[:,k]+dt/2*k1, U[:,k])
        k3 = f(X[:,k]+dt/2*k2, U[:,k])
        k4 = f(X[:,k]+dt*k3,   U[:,k])
        x_next = X[:,k] + dt/6*(k1+2*k2+2*k3+k4) 
        opti.subject_to(X[:,k+1]==x_next) # close the gaps

    # ---- path constraints -----------
    limit_upper = lambda pos_x: sin(0.5*pi*pos_x) + 1.0
    limit_lower = lambda pos_x: sin(0.5*pi*pos_x) - 0.5
    opti.subject_to(limit_lower(pos_x)<=pos_y)
    opti.subject_to(pos_y<=limit_upper(pos_x))   # track speed limit
    opti.subject_to(opti.bounded(-10,U,10)) # control is limited

    # ---- boundary conditions --------
    opti.subject_to(pos_x[0]==x_init)
    opti.subject_to(pos_y[0]==y_init)   # start at position (0,0)
    opti.subject_to(vel_x[0]==vx_init)
    opti.subject_to(vel_y[0]==vy_init)   # start from stand-still 
    # opti.subject_to(pos_x[-1]==0) 
    # opti.subject_to(pos_y[-1]==0)  # finish at position (3,3)

    # # ---- initial values for solver ---
    # opti.set_initial(U, 1)
    # opti.set_initial(L, 1)

    # ---- solve NLP              ------
    opti.solver("ipopt") # set numerical backend
    sol = opti.solve()   # actual solve
    return sol.value(pos_x[1]), sol.value(pos_y[1]), sol.value(vel_x[1]), sol.value(vel_y[1])

# ---- post-processing        ------
import matplotlib.pyplot as plt
x_0, y_0, vx_0, vy_0 = -1, -1, 1.0, 0
# x_0, y_0, vx_0, vy_0 = -0.5828317789641562, 0.9186459198637096, 0.6686728841433757, -0.32541632054516134
Epi = 5000
x_log, y_log = [], [] 
for i in range(Epi):
    try:
        x_0, y_0, vx_0, vy_0 = solver_mpc(x_0, y_0, vx_0, vy_0)
        x_log.append(x_0)
        y_log.append(y_0)
        if x_0 ** 2 + y_0 ** 2 < 0.01:
            break
    except RuntimeError:
        break

print(x_0, y_0)

plt.plot(x_log, y_log, 'r-')
plt.plot(0,0,'bo')
plt.xlabel('pos_x')
plt.ylabel('pos_y')
plt.axis([-4.0, 4.0, -4.0, 4.0])

x = np.arange(-4,4,0.01)
y = np.sin(0.5 * pi * x) +1
plt.plot(x, y, 'g-', label='upper limit')
plt.plot(x, y-1.5, 'b-', label='lower limit')
plt.draw()
plt.pause(1)
input("<Hit Enter>")
plt.close()
plt.show()