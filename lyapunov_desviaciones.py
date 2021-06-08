##Todo esto esta en mi documento en colab que se puede correr en la nube de google.
##https://colab.research.google.com/drive/1dzFv48FO_ERIGLjjAat-HTWWUUWv-IyN?usp=sharing
##https://colab.research.google.com/drive/1iGfMGTkTzrmN_kaJW5etAAkW_Iqkfr4p?usp=sharing
import numpy as np
from numpy import linspace
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class DS: #dynamical system.
  def __init__(self, y0):
    self.y0 = y0  # initial conditions
    self.f = lambda x,t:[x[2],x[3],(3-4*np.sqrt(x[0]**2+(x[1]-1)**2))*x[0]/np.sqrt(x[0]**2+(x[1]-1)**2),(3-4*np.sqrt(x[0]**2+(x[1]-1)**2))*(x[1]-1)/np.sqrt(x[0]**2+(x[1]-1)**2)]
  def sol(self,t):
    return odeint(self.f,self.y0,t)
    
#################

N=30000
t=linspace(0,3000,N)


exam1=DS([15,5,10,1])
exam2=DS([15,5,10,1.12])
sol1=exam1.sol(t)
sol2=exam2.sol(t)
plt.plot(t,np.log(abs(sol1[:,0]-sol2[:,0])))
#plt.plot(t,sol1[:,0])
#plt.plot(t,sol2[:,0])
plt.rcParams['figure.figsize'] = (10.0, 5.0)
#plt.subplot(121); plt.plot(sol[:,0],sol[:,2]);plt.grid();plt.xlabel('x');plt.ylabel('px')
#plt.subplot(122); plt.plot(sol[:,1],sol[:,3]);plt.xlabel('y');plt.ylabel('py')
#plt.xlim(-0.5,0.5)
#plt.ylim(-0.5, 0.5)
plt.grid()
#print(Energia([15,5,10,1]))
#plt.title("")
plt.xlabel("Tiempo")
plt.ylabel("|Desviacion|")


#############################
exam1=DS([0.1,0,0.1,0.2])
exam2=DS([0.101,0.01,0.1,0.2001])
sol1=exam1.sol(t)
sol2=exam2.sol(t)

plt.plot(t,np.log(abs(sol1[:,0]-sol2[:,0])))
#plt.plot(t,sol1[:,0])
#plt.plot(t,sol2[:,0])
plt.rcParams['figure.figsize'] = (10.0, 5.0)
#plt.subplot(121); plt.plot(sol[:,0],sol[:,2]);plt.grid();plt.xlabel('x');plt.ylabel('px')
#plt.subplot(122); plt.plot(sol[:,1],sol[:,3]);plt.xlabel('y');plt.ylabel('py')
#plt.xlim(-0.5,0.5)
#plt.ylim(-0.5, 0.5)
plt.grid()

#plt.title("Sports Watch Data")
plt.xlabel("Tiempo")
plt.ylabel("|Desviacion|")


###########################


!pip install jitcode

from jitcode import jitcode_lyap, y
from scipy.stats import sem
import numpy as np
from symengine import sqrt,sin,cos
from scipy.integrate import odeint
import matplotlib.pyplot as plt
#f = [y(2), y(3)/(y(0)**2), (y(3)**2)/(y(0)**3)+cos(y(1))-y(0)+1, -y(0)*sin(y(1))]

f = [
	y(2),
	y(3),
	3*y(0)/sqrt(y(0)**2+(y(1)-1)**2)-4*y(0),
	3*(y(1)-1)/sqrt(y(0)**2+(y(1)-1)**2)-4*(y(1)-1)
	]

#initial_state = np.array([10.,5.,5.,1.]) 
initial_state = np.array([10.,5.,5.,1.])
n = len(f)
ODE = jitcode_lyap(f, n_lyap=n)
ODE.set_integrator("vode")
ODE.set_initial_value(initial_state,0.0)

times = range(0,20000,5)
#times = range(0,500,5)
lyaps = []
for time in times:
	lyaps.append(ODE.integrate(time)[1])

# converting to NumPy array for easier handling
lyaps = np.vstack(lyaps)

for i in range(n):
	lyap = np.average(lyaps[1000:,i])
	stderr = sem(lyaps[1000:,i]) # Note that this only an estimate
	print("%i. Lyapunov exponent: % .4f ± %.4f" % (i+1,lyap,stderr))














