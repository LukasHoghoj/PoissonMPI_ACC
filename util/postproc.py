#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 08:43:17 2019

@author: lukas
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.optimize as opt

def am(cores,f):
    return 1/( (1-f) + f/cores )

fnames = "GPUprof"
fnamecmp = "GPU_compare"
fnamegpu = "GPU_serial"
fnameser = "serial"
fnamempi = "MPI"


flopiter = 15

f = open(fnames,'r')
contents = f.readlines()
f.close()
i=0
for line in contents:
    i+=1

# Initiate arrays
GPUdat = np.zeros((i-1,9))
i=0
for line in contents:
    obj = line.split(sep=',')
    if i>0:
        for j in range(6):
            GPUdat[i-1,j] = int(obj[j])
        for j in range(6,9):
            GPUdat[i-1,j] = float(obj[j])
    i+=1

# Compute computations in the problem
flops = ((GPUdat[:,2]-2)*(GPUdat[:,3]-2)*(GPUdat[:,4]-2)*flopiter)*GPUdat[:,5]
# Computations/time
fps = flops/GPUdat[:,6]

GPU2_23 = (GPUdat[:,1]==2)*(GPUdat[:,0]==23)*(GPUdat[:,2]==GPUdat[:,3])*(GPUdat[:,2]==GPUdat[:,4])
GPU4_23 = (GPUdat[:,1]==4)*(GPUdat[:,0]==23)*(GPUdat[:,2]==GPUdat[:,3])*(GPUdat[:,2]==GPUdat[:,4])
GPU6_23 = (GPUdat[:,1]==6)*(GPUdat[:,0]==23)*(GPUdat[:,2]==GPUdat[:,3])*(GPUdat[:,2]==GPUdat[:,4])
GPU2_22 = (GPUdat[:,1]==2)*(GPUdat[:,0]==22)*(GPUdat[:,2]==GPUdat[:,3])*(GPUdat[:,2]==GPUdat[:,4])
GPU4_22 = (GPUdat[:,1]==4)*(GPUdat[:,0]==22)*(GPUdat[:,2]==GPUdat[:,3])*(GPUdat[:,2]==GPUdat[:,4])
GPU6_22 = (GPUdat[:,1]==6)*(GPUdat[:,0]==22)*(GPUdat[:,2]==GPUdat[:,3])*(GPUdat[:,2]==GPUdat[:,4])


plt.figure()
plt.grid()
plt.loglog(flops[GPU2_23],GPUdat[GPU2_23,6],'rx')
plt.loglog(flops[GPU4_23],GPUdat[GPU4_23,6],'bx')
plt.loglog(flops[GPU6_23],GPUdat[GPU6_23,6],'gx')
plt.loglog(flops[GPU2_22],GPUdat[GPU2_22,6],'r+')
plt.loglog(flops[GPU4_22],GPUdat[GPU4_22,6],'b+')
plt.loglog(flops[GPU6_22],GPUdat[GPU6_22,6],'g+')
plt.xlabel('Problem FLOPs')
plt.ylabel('Computational time')

plt.legend(('2 GPUs nonblocking', '4 GPUs nonblocking',\
            '6 GPUs nonblocking', '2 GPUs', '4 GPUs', '6 GPUs'))
plt.savefig('../report/fig/flops_time.eps',format='eps')

msize = GPUdat[:,2]*GPUdat[:,3]*GPUdat[:,4]*8*3*1e-6#/GPUdat[:,1]
#msize=GPUdat[:,2]*GPUdat[:,3]/(GPUdat[:,4]/GPUdat[:,1])
plt.figure()
plt.grid()
plt.semilogx(msize[GPU2_23],fps[GPU2_23],'rx-')
plt.semilogx(msize[GPU4_23],fps[GPU4_23],'bx-')
plt.semilogx(msize[GPU6_23],fps[GPU6_23],'gx-')
plt.xlabel('Memory footprint [MB]')
plt.ylabel('FLOPs/s')
plt.legend(('2 GPUs', '4 GPUs', '6 GPUs'),loc='upper left')
plt.ylim((0, 1e+12))
plt.savefig('../report/fig/m_fps_23.eps',format='eps')

plt.figure()
plt.grid()
plt.semilogx(msize[GPU2_22],fps[GPU2_22],'rx-')
plt.semilogx(msize[GPU4_22],fps[GPU4_22],'bx-')
plt.semilogx(msize[GPU6_22],fps[GPU6_22],'gx-')
plt.xlabel('Memory footprint [MB]')
plt.ylabel('FLOPs/s')
plt.legend(('2 GPUs', '4 GPUs', '6 GPUs'),loc='upper left')
plt.ylim((0, 1e+12))
plt.savefig('../report/fig/m_fps_22.eps',format='eps')


n22_1024 = (GPUdat[:,2]==1024)*(GPUdat[:,0]==22)
n22_1024_0 = n22_1024 * (GPUdat[:,1]==2)
n23_1024 = (GPUdat[:,2]==1024)*(GPUdat[:,0]==23)
n23_1024_0 = n23_1024 * (GPUdat[:,1]==2)

npmax = max(np.hstack((GPUdat[n22_1024,1],GPUdat[n23_1024,1])))

f22, pcov = opt.curve_fit(am, np.hstack((GPUdat[n22_1024_0,6]*2,GPUdat[n22_1024,1])),np.hstack((1,GPUdat[n22_1024_0,6]*2/GPUdat[n22_1024,6])))
f23, pcov = opt.curve_fit(am, np.hstack((GPUdat[n23_1024_0,6]*2,GPUdat[n23_1024,1])),np.hstack((1,GPUdat[n23_1024_0,6]*2/GPUdat[n23_1024,6])))

plt.figure()
plt.grid()
plt.plot(GPUdat[n22_1024,1],GPUdat[n22_1024_0,6]/GPUdat[n22_1024,6],'bx')
plt.plot(GPUdat[n22_1024,1],1/(f22/np.sort(GPUdat[n22_1024,1]) + (1-f22)),'b-.')
plt.plot(GPUdat[n23_1024,1],GPUdat[n23_1024_0,6]/GPUdat[n23_1024,6],'rx')
plt.plot(GPUdat[n23_1024,1],1/(f23/np.sort(GPUdat[n23_1024,1]) + (1-f23)),'r-.')
plt.plot(np.array([0,npmax]),np.array([0,npmax]),'orange')
plt.xlabel('Threads')
plt.ylabel('Speed-up')
plt.legend(('Blocking',
            'Ahmdals law, f=%4.3f'%f22,
            'Non-blocking',
            'Ahmdals law, f=%4.3f'%f23,
            'Theoretical max'))
plt.tight_layout()






##########################3
f = open(fnamecmp,'r')
contents = f.readlines()
f.close()
i=0
for line in contents:
    i+=1

# Initiate arrays
GPUdatcmp = np.zeros((i-1,9))
i=0
for line in contents:
    obj = line.split(sep=',')
    if i>0:
        for j in range(6):
            GPUdatcmp[i-1,j] = int(obj[j])
        for j in range(6,9):
            GPUdatcmp[i-1,j] = float(obj[j])
    i+=1
flopscmp = ((GPUdatcmp[:,2]-2)*(GPUdatcmp[:,3]-2)*(GPUdatcmp[:,4]-2)*flopiter)*GPUdatcmp[:,5]
# Computations/time
fpscmp = flopscmp/GPUdatcmp[:,6]
msizecmp = GPUdatcmp[:,2]*GPUdatcmp[:,3]*GPUdatcmp[:,4]*8*3*1e-6#/GPUdat[:,1]

g20 = GPUdatcmp[:,0]==20
g21 = GPUdatcmp[:,0]==21
g22 = GPUdatcmp[:,0]==22
g23 = GPUdatcmp[:,0]==23

plt.figure()
plt.grid()
plt.semilogx(msizecmp[g20],fpscmp[g20],'rx-')
plt.semilogx(msizecmp[g21],fpscmp[g21],'gx-')
plt.semilogx(msizecmp[g22],fpscmp[g22],'bx-')
plt.semilogx(msizecmp[g23],fpscmp[g23],'mx-')
plt.xlabel('Memory footprint [MB]')
plt.ylabel('FLOPs/s')
plt.legend(('V1','V2','V3','V4'))
plt.savefig('../report/fig/hybtypes.eps',format='eps')

#############################

f = open(fnamegpu,'r')
contents = f.readlines()
f.close()
i=0
for line in contents:
    i+=1

# Initiate arrays
GPUdatgpu = np.zeros((i-1,9))
i=0
for line in contents:
    obj = line.split(sep=',')
    if i>0:
        for j in range(6):
            GPUdatgpu[i-1,j] = int(obj[j])
        for j in range(6,9):
            GPUdatgpu[i-1,j] = float(obj[j])
    i+=1
g1 = GPUdatgpu[:,0]==1
g2 = GPUdatgpu[:,0]==2
flopsgpu = ((GPUdatgpu[:,2]-2)*(GPUdatgpu[:,3]-2)*(GPUdatgpu[:,4]-2)*flopiter)*GPUdatgpu[:,5]
# Computations/time
fpsgpu = flopsgpu/GPUdatgpu[:,6]
msizegpu = GPUdatgpu[:,2]*GPUdatgpu[:,3]*GPUdatgpu[:,4]*8*3*1e-6#/GPUdat[:,1]

plt.figure()
plt.grid()
plt.semilogx(msizegpu[g1],fpsgpu[g1],'rx-')
plt.semilogx(msizegpu[g2],fpsgpu[g2],'gx-')
plt.xlabel('Memory footprint [MB]')
plt.ylabel('FLOPs/s')
plt.legend(('First version','Reduced data movement'))
plt.savefig('../report/fig/gpu.eps',format='eps')


#############################

f = open(fnameser,'r')
contents = f.readlines()
f.close()
i=0
for line in contents:
    i+=1

# Initiate arrays
serdat = np.zeros((i-1,9))
i=0
for line in contents:
    obj = line.split(sep=',')
    if i>0:
        for j in range(6):
            serdat[i-1,j] = int(obj[j])
        for j in range(6,9):
            serdat[i-1,j] = float(obj[j])
    i+=1
    
flopsser = ((serdat[:,2]-2)*(serdat[:,3]-2)*(serdat[:,4]-2)*flopiter)*serdat[:,5]
# Computations/time
fpsser = flopsser/serdat[:,6]
msizeser = serdat[:,2]*serdat[:,3]*serdat[:,4]*8*3*1e-6#/GPUdat[:,1]

plt.figure()
plt.grid()
plt.semilogx(msizeser,fpsser,'rx-')
plt.xlabel('Memory footprint [MB]')
plt.ylabel('FLOPs/s')
plt.savefig('../report/fig/ser.eps',format='eps')

#############################

f = open(fnamempi,'r')
contents = f.readlines()
f.close()
i=0
for line in contents:
    i+=1

# Initiate arrays
mpidat = np.zeros((i-1,9))
i=0
for line in contents:
    obj = line.split(sep=',')
    if i>0:
        for j in range(6):
            mpidat[i-1,j] = int(obj[j])
        for j in range(6,9):
            mpidat[i-1,j] = float(obj[j])
    i+=1
    
flopsmpi = ((mpidat[:,2]-2)*(mpidat[:,3]-2)*(mpidat[:,4]-2)*flopiter)*mpidat[:,5]
# Computations/time
fpsmpi = flopsmpi/mpidat[:,6]
msizempi = mpidat[:,2]*mpidat[:,3]*mpidat[:,4]*8*3*1e-6#/GPUdat[:,1]
nt=np.array([1,2,4,8,16,24,28,36,44,48])
plt.figure()
plt.grid()
for i in range(nt.size):
    ind = (mpidat[:,1]==nt[i])*(mpidat[:,0]==10)
    plt.semilogx(msizempi[ind],fpsmpi[ind],'x-')
plt.xlabel('Memory footprint [MB]')
plt.ylabel('FLOPs/s')
plt.legend(nt,loc='upper right')
plt.ylim((0, 2e11))
plt.xlim((0,2e4))
plt.savefig('../report/fig/mpi1.eps',format='eps')
plt.figure()
plt.grid()
for i in range(nt.size):
    ind = mpidat[:,1]==nt[i]*(mpidat[:,0]==11)
    plt.semilogx(msizempi[ind],fpsmpi[ind],'x-')
plt.xlabel('Memory footprint [MB]')
plt.ylabel('FLOPs/s')
plt.legend(nt,loc='upper right')
plt.ylim((0, 2e11))
plt.xlim((0,2e4))
plt.savefig('../report/fig/mpi2.eps',format='eps')

n11_256 = (mpidat[:,2]==256)*(mpidat[:,0]==11)
n11_256_0 = n11_256 * (mpidat[:,1]==1)
n10_256 = (mpidat[:,2]==256)*(mpidat[:,0]==10)
n10_256_0 = n11_256 * (mpidat[:,1]==1)

npmax = max(np.hstack((mpidat[n10_256,1],mpidat[n10_256,1])))

f10, pcov = opt.curve_fit(am, mpidat[n10_256,1],mpidat[n10_256_0,6]/mpidat[n10_256,6])
f11, pcov = opt.curve_fit(am, mpidat[n11_256,1],mpidat[n11_256_0,6]/mpidat[n11_256,6])
plt.figure()
plt.grid()
plt.plot(mpidat[n10_256,1],mpidat[n10_256_0,6]/mpidat[n10_256,6],'bx')
plt.plot(mpidat[n10_256,1],1/(f10/np.sort(mpidat[n10_256,1]) + (1-f10)),'b-.')
plt.plot(mpidat[n11_256,1],mpidat[n11_256_0,6]/mpidat[n11_256,6],'rx')
plt.plot(mpidat[n11_256,1],1/(f11/np.sort(mpidat[n11_256,1]) + (1-f11)),'r-.')
plt.plot(np.array([0,npmax]),np.array([0,npmax]),'orange')
plt.xlabel('Threads')
plt.ylabel('Speed-up')
plt.legend(('Blocking',
            'Ahmdals law, f=%4.3f'%f10,
            'Non-blocking',
            'Ahmdals law, f=%4.3f'%f11,
            'Theoretical max'))
plt.axis('equal')
plt.axis([0,npmax*1.1, 0,npmax*1.1])
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig('../report/fig/mpiahmdal.eps',format='eps')