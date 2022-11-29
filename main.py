# Zach Pedersen, Rylan Casanova
# This is our work!
# CST-305
# Prof. Citro

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time
import math

# Our ODEs
def dudx1(u, x):
    return [u[1], 2 * x * u[1] - x ** 2 * u[0]]
def dudx2(u, x):
    return [u[1], (x - 2) * u[1] - 2 * u[0]]

# Computes y's using Taylor Polynomial
# Taylor polynomial of degree 4
def taylor1(x):
    return 1 - x - (1 / 3) * x ** 3 - (1 / 12) * x ** 4
# Taylor polynomial of degree 2
def taylor2(x):
    return 6 + (x - 3) - (11 / 2) * (x - 3) ** 2

#Computes Computer Performance model
def dudx3(u, x):
    return [u[1], 0 * u[1] - (1 / (x ** 2 + 4)) * u[0] + x / (x ** 2 + 4)]
def computer(u, x):
    return [u[1], -x * u[1] - x ** 2 * u[0] + x ** 3]

# number of points
r = 101
# start conditions
x = np.linspace(-5, 5, r)
u = [1, -1]

# Calculates with Odeint
uy = odeint(dudx1, u, x)
y = uy[:, 0]

# Plot results (Odeint1)
plt.title('Odeint1')
plt.plot(x, y, label="Odeint1")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# number of points
r = 101
# start conditions
x3 = np.linspace(-5, 5, r)

# Calculates with Taylor Polynomial
y3 = taylor1(x3)

# Plot results (Taylor1)
plt.title('Taylor1')
plt.plot(x3, y3, label="Taylor1", linestyle=":")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# Show Odeint and Taylor together
plt.plot(x, y, label="Odeint1")
plt.plot(x3, y3, label="Taylor1", linestyle=":")
plt.legend()
plt.show()

# number of points
r = 101
# start conditions
x2 = np.linspace(0, 6, r)
u2 = [6, 1]

# Calculates with Odeint
uy2 = odeint(dudx2, u2, x2)
y2 = uy2[:, 0]

# Plot results (Odeint2)
plt.title('Odeint2')
plt.plot(x2, y2, label="Odeint2")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# number of points
r = 101
# start conditions
x4 = np.linspace(0, 6, r)

# Calculates with Taylor Polynomial
y4 = taylor2(x4)

# Plot results (Taylor2)
plt.title('Taylor2')
plt.plot(x4, y4, label="Taylor2", linestyle=":")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# Show Odeint and Taylor together
plt.plot(x2, y2, label="Odeint2")
plt.plot(x4, y4, label="Taylor2", linestyle=":")
plt.legend()
plt.show()

# number of points
r = 101
# start conditions
x3 = np.linspace(0, 10, r)
u3 = [0, 0]

# Calculates with Odeint
uy3 = odeint(dudx3, u3, x3)
y3 = uy3[:, 0]

# Plot results (Odeint2)
plt.title('Odeint3')
plt.plot(x3, y3, label="Odeint3")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# number of points
r = 101
# start conditions
xc = np.linspace(0, 30, r)
uc = [0, 0]

# Calculates with Odeint
uyc = odeint(computer, uc, xc)
yc = uyc[:, 0]

# Plot results (cpu)
plt.title('computer')
plt.plot(xc, yc, label="computer")
plt.xlabel('cost')
plt.ylabel('best performance')
plt.legend()
plt.show()
