import numpy as np
from math import sin, cos, acos, pi, sqrt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import axes3d
import time
import numba

N = 20
m = 20
g1 = 1.0
kbT = 5.0
dt = 0.001
finalT = 0.0
tempsteps = 10000
tempjump = (kbT-finalT)/tempsteps
D  = 1.0
g2 = 1.0

Nx = Ny = Nz = m

dx = N/Nx
dy = N/Ny
dz = N/Nz

@numba.jit(nopython=True)
def torque(nx, ny, nz):
    N = np.shape(nx)[0]

    torques = np.zeros((N, N, N, 3))

    for i in numba.prange(N):
        for j in range(N):
            for k in range(N):
                xplus = x+1
                xminus = x-1
                yplus = y+1
                yminus = y-1
                zplus = z+1
                zminus = z-1
                if (xplus == N):  xplus = 0
                if (xminus == -1):  xminus = N-1
                if (yplus == N):  yplus = 0
                if (yminus == -1):  yminus = N-1
                if (zplus == N):  zplus = 0
                if (zminus == -1):  zminus = 0
                
                # get nearest neighbors
                ntest = np.array([nx[x][y][z], ny[x][y][z], nz[x][y][z]])
                n1 = np.array([nx[xplus][y][z], ny[xplus][y][z], nz[xplus][y][z]])
                n2 = np.array([nx[xminus][y][z], ny[xminus][y][z], nz[xminus][y][z]])
                n3 = np.array([nx[x][yplus][z], ny[x][yplus][z], nz[x][yplus][z]])
                n4 = np.array([nx[x][yminus][z], ny[x][yminus][z], nz[x][yminus][z]])
                n5 = np.array([nx[x][y][zplus], ny[x][y][zplus], nz[x][y][zplus]])
                n6 = np.array([nx[x][y][zminus], ny[x][y][zminus], nz[x][y][zminus]])
                
                dot_xplus = np.dot(ntest,n1)
                dot_xminus = np.dot(ntest,n2)
                dot_yplus = np.dot(ntest,n3)
                dot_yminus = np.dot(ntest,n4)
                dot_zplus = np.dot(ntest,n5)
                dot_zminus = np.dot(ntest,n6)
                
                cross_xplus = np.cross(ntest,n1)
                cross_xminus = np.cross(ntest,n2)
                cross_yplus = np.cross(ntest,n3)
                cross_yminus = np.cross(ntest,n4)
                cross_zplus = np.cross(ntest,n5)
                cross_zminus = np.cross(ntest,n6)
                
                total_torque = dot_xplus*cross_xplus + dot_xminus*cross_xminus + dot_yplus*cross_yplus + dot_yminus*cross_yminus + dot_zplus*cross_zplus + dot_zminus*cross_zminus
                
                np.random.seed(np.random.randint(0,4000000))
                theta2 = acos(2.0 * np.random.rand() - 1.0)
                phi2 = 2.0 * pi * np.random.rand()
                random_direction = [sin(theta2)*cos(phi2), sin(theta2)*sin(phi2), cos(theta2)]
    
                torques[i, j, k] = (1.0/g1)*total_torque + sqrt(2 * kbT / (g1 * dt)) * np.random.normal(0,1) * np.array(random_direction)

    return torques
    
@numba.jit(nopython=True)
def localorder(nx, ny ,nz):
    N = np.shape(nx)[0]

    localorders = np.zeros((N, N, N))

    for x in numba.prange(N):
        for y in range(N):
            for z in range(N):

                xplus = x+1
                xplus2 = x+2
                xminus = x-1
                xminus2 = x-2
                yplus = y+1
                yplus2 = y+2
                yminus = y-1
                yminus2 = y-2
                zplus = z+1
                zplus2 = z+2
                zminus = z-1
                zminus2 = z-2

                if (xplus == N):  xplus = 0
                if (xplus2 == N):  xplus2 = 0
                if (xplus2 == N+1):  xplus2 = 1
                if (xminus == -1):  xminus = N-1
                if (xminus2 == -1):  xminus2 = N-1
                if (xminus2 == -2):  xminus2 = N-2
                if (yplus == N):  yplus = 0
                if (yplus2 == N):  yplus2 = 0
                if (yplus2 == N+1):  yplus2 = 1
                if (yminus == -1):  yminus = N-1
                if (yminus2 == -1):  yminus2 = N-1
                if (yminus2 == -2):  yminus2 == N-2
                if (zplus == N):  zplus = 0
                if (zplus2 == N):  zplus2 = 0
                if (zplus2 == N+1):  zplus2 = 1
                if (zminus == -1):  zminus = N-1
                if (zminus2 == -1):  zminus2 = N-1
                if (zminus2 == -2):  zminus2 = N-2
                
                Qxx = (nx[x][y][z]*nx[x][y][z] + nx[xplus][y][z]*nx[xplus][y][z] + nx[xplus2][y][z]*nx[xplus2][y][z] + nx[xminus][y][z]*nx[xminus][y][z] + nx[xminus2][y][z]*nx[xminus2][y][z] + nx[x][yplus][z]*nx[x][yplus][z] + nx[x][yplus2][z]*nx[x][yplus2][z] + nx[x][yminus][z]*nx[x][yminus][z] + nx[x][yminus2][z]*nx[x][yminus2][z] + nx[x][y][zplus]*nx[x][y][zplus] + nx[x][y][zplus2]*nx[x][y][zplus2] + nx[x][y][zminus]*nx[x][y][zminus] + nx[x][y][zminus2]*nx[x][y][zminus2] - 13.0*0.33333)/13.0
                Qyy = (ny[x][y][z]*ny[x][y][z] + ny[xplus][y][z]*ny[xplus][y][z] + ny[xplus2][y][z]*ny[xplus2][y][z] + ny[xminus][y][z]*ny[xminus][y][z] + ny[xminus2][y][z]*ny[xminus2][y][z] + ny[x][yplus][z]*ny[x][yplus][z] + ny[x][yplus2][z]*ny[x][yplus2][z] + ny[x][yminus][z]*ny[x][yminus][z] + ny[x][yminus2][z]*ny[x][yminus2][z] + ny[x][y][zplus]*ny[x][y][zplus] + ny[x][y][zplus2]*ny[x][y][zplus2] + ny[x][y][zminus]*ny[x][y][zminus] + ny[x][y][zminus2]*ny[x][y][zminus2] - 13.0*0.33333)/13.0
                Qzz = (nz[x][y][z]*nz[x][y][z] + nz[xplus][y][z]*nz[xplus][y][z] + nz[xplus2][y][z]*nz[xplus2][y][z] + nz[xminus][y][z]*nz[xminus][y][z] + nz[xminus2][y][z]*nz[xminus2][y][z] + nz[x][yplus][z]*nz[x][yplus][z] + nz[x][yplus2][z]*nz[x][yplus2][z] + nz[x][yminus][z]*nz[x][yminus][z] + nz[x][yminus2][z]*nz[x][yminus2][z] + nz[x][y][zplus]*nz[x][y][zplus] + nz[x][y][zplus2]*nz[x][y][zplus2] + nz[x][y][zminus]*nz[x][y][zminus] + nz[x][y][zminus2]*nz[x][y][zminus2] - 13.0*0.33333)/13.0    
                Qxy = (nx[x][y][z]*ny[x][y][z] + nx[xplus][y][z]*ny[xplus][y][z] + nx[xplus2][y][z]*ny[xplus2][y][z] + nx[xminus][y][z]*ny[xminus][y][z] + nx[xminus2][y][z]*ny[xminus2][y][z] + nx[x][yplus][z]*ny[x][yplus][z] + nx[x][yplus2][z]*ny[x][yplus2][z] + nx[x][yminus][z]*ny[x][yminus][z] + nx[x][yminus2][z]*ny[x][yminus2][z] + nx[x][y][zplus]*ny[x][y][zplus] + nx[x][y][zplus2]*ny[x][y][zplus2] + nx[x][y][zminus]*ny[x][y][zminus] + nx[x][y][zminus2]*ny[x][y][zminus2])/13.0
                Qxz = (nx[x][y][z]*nz[x][y][z] + nx[xplus][y][z]*nz[xplus][y][z] + nx[xplus2][y][z]*nz[xplus2][y][z] + nx[xminus][y][z]*nz[xminus][y][z] + nx[xminus2][y][z]*nz[xminus2][y][z] + nx[x][yplus][z]*nz[x][yplus][z] + nx[x][yplus2][z]*nz[x][yplus2][z] + nx[x][yminus][z]*nz[x][yminus][z] + nx[x][yminus2][z]*nz[x][yminus2][z] + nx[x][y][zplus]*nz[x][y][zplus] + nx[x][y][zplus2]*nz[x][y][zplus2] + nx[x][y][zminus]*nz[x][y][zminus] + nx[x][y][zminus2]*nz[x][y][zminus2])/13.0
                Qyz = (ny[x][y][z]*nz[x][y][z] + nz[xplus][y][z]*ny[xplus][y][z] + nz[xplus2][y][z]*ny[xplus2][y][z] + nz[xminus][y][z]*ny[xminus][y][z] + nz[xminus2][y][z]*ny[xminus2][y][z] + nz[x][yplus][z]*ny[x][yplus][z] + nz[x][yplus2][z]*ny[x][yplus2][z] + nz[x][yminus][z]*ny[x][yminus][z] + nz[x][yminus2][z]*ny[x][yminus2][z] + nz[x][y][zplus]*ny[x][y][zplus] + nz[x][y][zplus2]*ny[x][y][zplus2] + nz[x][y][zminus]*ny[x][y][zminus] + nz[x][y][zminus2]*ny[x][y][zminus2])/13.0

                Q = np.array([[Qxx, Qxy, Qxz], [Qxy, Qyy, Qyz], [Qxz, Qyz, Qzz]])
                
                eigenvalues = np.linalg.eigvals(Q)
                localorders[x, y, z] =  np.real(1.5 * np.max(eigenvalues.real))
    return localorders

def Laplacian(concentration, dx, dy, dz):
    for x in range(0,N):
        for y in range(0,N):
            for z in range(0,N):
                xplus = x+1
                if (xplus == N):   xplus = 0
                xminus = x-1
                if (xminus == -1):   xminus = N-1
                yplus = y+1
                if (yplus == N):   yplus = 0
                yminus = y-1
                if (yminus == -1):   yminus = N-1
                zplus = z+1
                if (zplus == N):  zplus = 0
                zminus = z-1
                if (zminus == -1):  zminus = N-1
                dc[x][y][z] = (concentration[xplus][y][z] + concentration[xminus][y][z] - 2*concentration[x][y][z])/dx/dx + (concentration[x][yplus][z] + concentration[x][yminus][z] - 2*concentration[x][y][z])/dy/dy + (concentration[x][y][zplus] + concentration[x][y][zminus] - 2*concentration[x][y][z])/dz/dz
    
    return dc

nx = np.zeros((N,N,N))
ny = np.zeros((N,N,N))
nz = np.zeros((N,N,N))
ordermap =  np.zeros((N,N,N))
omega = np.zeros((N,N,N))
n_all = np.zeros((N,N,N),dtype='object')


for x in range(0,N):
    for y in range(0,N):
        for z in range(0,N):
            np.random.seed(np.random.randint(0,4000000))
            theta = acos(2.0 * np.random.rand() - 1.0)
            phi = 2.0 * pi * np.random.rand()
            nx[x][y][z] = sin(theta)*cos(phi)
            ny[x][y][z] = sin(theta)*sin(phi)
            nz[x][y][z] = cos(theta)
            n_all[x,y,z] = np.array([nx[x,y,z], ny[x,y,z], nz[x,y,z]])
            
s = []
count = 0
t0 = 0
t1 = 0
t2 = 0
t = 0


print('Now running nematic liquid crystal dymanic simulation...')
while (t < tempsteps):
    time0 = time.time()  
    
    omega = torque(nx, ny, nz)
#    omega = np.fromfunction(torque, (N, N, N), nx=nx, ny=ny, nz=nz)

    t0 += time.time() - time0

    time0 = time.time()
    n = np.stack([nx, ny, nz], axis=3)
    n = n + np.cross(omega, n, axisa=3, axisb=3)*dt
    norm = np.linalg.norm(n, axis=3)
    #print('test:',n[0,0,0],norm[0,0,0])
    n = n / np.stack([norm]*3, axis=3)
    #print('test:',n[0,0,0], np.linalg.norm(n[0,0,0]))
    nx = n[:,:,:,0]
    ny = n[:,:,:,1]
    nz = n[:,:,:,2]
    t1 += time.time() - time0

    time0 = time.time()
#    ordermap = [[[localorder(i,j,k,nx,ny,nz) for i in range(N)] for j in range(N)] for k in range(N)]
    ordermap = localorder(nx, ny, nz)
    t2 += time.time() - time0

    s.append(np.mean(ordermap))
    kbT -= tempjump
    t += 1
print('Simulation complete.  Now saving director configurations and order parameters.')
#print(t0,t1,t2)
np.save('directors', n)
np.save('order_map', ordermap)
np.save('order_parameter', s)

