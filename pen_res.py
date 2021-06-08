from jitcode import jitcode_lyap 
from jitcode import y 
from scipy.stats import sem
import numpy as np
import sympy as sm

 
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os

 
#parametros de entrada del sistema 
m  = 1.0
#m   = 3.0
#k  = 1.0e3
k  = 1.0e4
L0 = 3.0e-2
g  = 9.8
#Si se desea generar la condicion de nu=2 descrita en el articulo , 
#deben elegirse valores convenientes de las variables anteriores


'''

Sistema pendulo-resorte acorde a la pagina:

https://scipython.com/blog/the-spring-pendulum/


thetadot = z1
z1dot = (-g*np.sin(theta) - 2*z1*z2) / L
Ldot = z2
z2dot = (m*L*z1**2 - k*(L-L0) + m*g*np.cos(theta)) 

Lo cual, en nuestro programa es:


Convencion de las variables:

theta         = y(0) 
z1 (dtheta/dt)= y(1)
L             = y(2)
z2 (dL/dt)    = y(3)

Compactificamos el sistema de resorte-pendular como

d(y)/dt = F(y)

Donde y = [ y(0),y(1),y(2),y(3)]

''' 

F = [
    y(1),
    (-g * sm.sin(y(0)) - 2*y(1)*y(3)) / y(2),
    y(3) ,  
    (m * y(2) * y(1)**2 - k*( y(2) - L0) + m*g* sm.cos(y(0))) / m 
    ]
 

#Crear el vector de tiempos
to     = 0
tf     = 10 
t_pasos= 1000
time = np.linspace(to ,tf ,t_pasos) 

#intervalo de tiempo
dt = time[1]-time[0]


#Condicion inicial: theta, dtheta/dt, L, dL/dt
#initial_state     = [ np.pi/40, 0, L0, 0]
initial_state     = [ np.pi/3, 2, L0, 1]

#---------Generando codigo -----------
'''
Generando codigo C
Escogiendo el metodo numerico (Runge-Kutta Felbergs de paso adatativo (RK45) )
'''

ODE = jitcode_lyap(F, n_lyap= len(F) )
ODE.set_integrator("RK45")
ODE.set_initial_value(initial_state,0.0)

  
y       = []   
Lyaps_coeffs = []

for t in time:

    integracion= ODE.integrate(t)


    # el vector solucion de la forma y = [ y(0),y(1),y(2),y(3)]
    y.append(integracion[0])

     
    #Los coefficients de lyapunuv
    Lyaps_coeffs.append(integracion[1])


#Converting to NumPy array for easier handling
y            = np.vstack(y)
Lyaps_coeffs = np.vstack(Lyaps_coeffs)

##############################################################################
#                                                                            #
#  Haciendo la animacion del pendulo                                         #
#                                                                            #
##############################################################################
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np



#expresando la ubucacion de la masa en coordenadas cartesianas 
r      = L0 + y[:,2]
Dr     = y[:,3] # esto es todos los valores en el tiempo de z1

theta  = y[:,0] # esto es todos los valores en el tiempo de theta
Dtheta = y[:,1] # esto es todos los valores en el tiempo de z1


# ubicacion de la masa en coord. cartesianas
x =  r*np.sin(theta)
y = -r*np.cos(theta)




#Creando las figuras
fig = plt.figure(figsize=(12,10))

ax = fig.add_subplot(221, autoscale_on=False, 
                     xlim=(-1.2*r.max(), 1.2*r.max()),
                     ylim=(-1.2*r.max(), 0.2*r.max()), aspect = 1.0)


ax.grid()
ax.set_title('simulacion del sistema')

Masa, = ax.plot([], [], 'bo-', markersize=15 , lw=2)

time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


[resorte] = ax.plot([], [], 'r', lw=2)
[Linea_horizontal] = ax.plot([], [], 'k', lw=8)


ax2 = fig.add_subplot(222,  
                     xlim=(-theta.max()-1, theta.max()+1),
                     ylim=(-Dtheta.max()-5, Dtheta.max()+5) )

ax2.grid()
ax2.set_title('diagrama de fase')
ax2.set_xlabel('theta')
ax2.set_ylabel('dtheta/dt')
Fase, = ax2.plot([], [],  'b-', lw=2)




ax3 = fig.add_subplot(223,  
                     xlim=(r.min()-0.01, r.max()+0.01),
                     ylim=(-Dr.max()-0.1, Dr.max()+0.1) )

ax3.grid()
ax3.set_title('diagrama de fase')
ax3.set_xlabel('r')
ax3.set_ylabel('dr/dt')
Fase2, = ax3.plot([], [], 'g-', lw=2)



"""
ax3 = fig.add_subplot(223 ,   
                     xlim=(0, tf ),
                     ylim=(-1e3 , 1e3 ) )

ax3.grid()
ax3.set_title('Coeficiente de Lyapunov 1')
ax3.set_xlabel('t')
ax3.legend()
ax3.set_yscale('symlog')


ax4 = fig.add_subplot(224 ,   
                     xlim=(0, tf ),
                     ylim=(-1e3 , 1e3 ) )
ax4.grid()
ax4.set_title('Coeficiente de Lyapunov 3')
ax4.set_xlabel('t')
ax4.legend()
ax4.set_yscale('symlog')

Lyapunov1, = ax3.plot([], [],  'b*:', lw=1, label='Coeff relativo a theta')
Lyapunov3, = ax4.plot([], [],  'gs:', lw=1, label='Coeff relativo a l')

"""

#funcion para hacer la animacion del resorte....me la golie de la pagina aquella de donde estan las eqs.
def RESORTE( theta, L):
    """Plot the spring from (0,0) to (x,y) as the projection of a helix."""
    # Spring turn radius, number of turns
    rs, ns = 0.005, 15
    # Number of data points for the helix
    Ns = 1000
    # We don't draw coils all the way to the end of the pendulum:
    # pad a bit from the anchor and from the bob by these number of points
    ipad1, ipad2 = 100, 150
    w = np.linspace(0, L, Ns)
    # Set up the helix along the x-axis ...
    xp = np.zeros(Ns)
    xp[ipad1:-ipad2] = rs * np.sin(2*np.pi * ns * w[ipad1:-ipad2] / L)
    # ... then rotate it to align with  the pendulum and plot.
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    xs, ys = - R @ np.vstack((xp, w))
    #ax.plot(xs, ys, c='b', lw=2)

    return xs, ys


def init():
    Masa.set_data([], [])
    time_text.set_text('')
    return Masa, time_text



xdata, ydata  = [], []
x2data,y2data = [], []


L1data, L2data,L3data,L4data =  [], [], [], []
L1time, L2time,L3time,L4time =  [], [], [], []

def fotogramas(i):
 
    Masa.set_data( x[i], y[i] )

    time_text.set_text(time_template%(i*dt))

    resorte.set_data( RESORTE( theta[i], np.sqrt(x[i]**2 + y[i]**2 )  ) )

    Linea_horizontal.set_data( [ 0, 0] ,  [ 0,  1]   )
 

    xdata.append( theta[i]  )
    ydata.append( Dtheta[i] )
    Fase.set_data( xdata , ydata )


    x2data.append( r[i]  )
    y2data.append( Dr[i] )
    Fase2.set_data( x2data , y2data )
  

    #agregando coefficientes de Lyapunov    
    #L1data.append(Lyaps_coeffs[i,0])        
    #L1time.append(time[i])  
    #Lyapunov1.set_data( L1time, L1data )


    #L3data.append(Lyaps_coeffs[i,2])        
    #L3time.append(time[i])  
    #Lyapunov3.set_data( L3time, L3data )


    return Masa, time_text , resorte , Fase , Fase2, Linea_horizontal



ani = animation.FuncAnimation(fig, fotogramas, np.arange(1, len(y)),
    interval=40 , blit=True, init_func=init)

plt.show()
