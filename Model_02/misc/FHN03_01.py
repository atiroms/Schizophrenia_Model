import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

I = 0.33 # external input
def d_u(u, v)
    c = 10
    return c  (-v + u - pow(u,3)3 + I)

def d_v(u, v)
    a = 0.7
    b = 0.8
    return u - b  v + a

def fitzhugh(state, t)
    u, v = state
    deltau = d_u(u,v)
    deltav = d_v(u,v)
    return deltau, deltav

# the initial state
u0 = 2.0
v0 = 1.0

t = np.arange(0.0, 20, 0.01)

y0 = [u0, v0]  # the initial state vector
y = integrate.odeint(fitzhugh, y0, t)
u_vec = y[,0]
v_vec = y[,1]

plt.figure()
plt.plot(t, u_vec, label=u)
plt.plot(t, v_vec, label=v)
plt.title(membrane potential)
plt.grid()
plt.legend()

plt.figure()
plt.plot(u_vec, v_vec)
plt.xlabel(u)
plt.ylabel(v)
plt.title(FitzHugh nullcline)
padding = 0.2
umax, umin = u_vec.max() + padding,  u_vec.min() - padding
vmax, vmin = v_vec.max() + padding,  v_vec.min() - padding
U, V = np.meshgrid(np.arange(umin, umax, 0.1), np.arange(umin, vmax, 0.1))
dU = d_u(U, V)
dV = d_v(U, V)
plt.quiver(U, V, dU, dV)
plt.contour(U, V, dV, levels=[0], colors=green)
plt.contour(U, V, dU, levels=[0], colors=red)
plt.xlim([umin, umax])
plt.ylim([vmin, vmax])
plt.grid()
plt.show()
