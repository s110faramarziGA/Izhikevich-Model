
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 12:40:58 2018

@author: sadegh-110
"""

from pylab import *
import numpy as np

# Change parameters here ####################################
# 1) Initialize simulation parameters

tmax = 12000  # Total simulation time
dt = 1        # Time step size

# 1.2) Input stimulus parameters
lapp = 10  # Input current amplitude
tr = array([200, 700]) / dt  # Stimulation time in discrete steps
 
# 2) Reserve memory for simulation variables
T = int(ceil(tmax / dt))  # Total number of time steps
nn = 6  # Number of neurons

# Initialize neuron state variables as zero matrices
ss = [nn, T]
v = np.zeros(ss)  # Membrane potential
u = np.zeros(ss)  # Recovery variable
phase = np.zeros(ss)  # Phase tracking

# 1.1) Neuron model parameters (Izhikevich model)
a = 0.2  # Parameter a
b = 0.2  # Parameter b
c = -65  # Reset value of membrane potential after spike
d = 8    # Reset value of recovery variable after spike

# 3) Initialize neuron states with random values within a range
vIni = np.random.randint(low=-70, high=-60, size=nn)  # Initial membrane potential
uIni = np.random.randint(low=-14, high=-9, size=nn)   # Initial recovery variable

# 4) Define network connectivity (adjacency matrix)
tau = 1  # Delay in coupling between neurons
aj = [nn, nn]  # Connection matrix size
ajm = np.zeros(aj)  # Initialize adjacency matrix with no connections

# Define specific connections between neurons with small weights
ajm[5][2] = 0.000001
ajm[3][5] = 0.000001
ajm[5][3] = 0.000001
ajm[1][4] = 0.000001   
ajm[4][1] = 0.000001

II = 10  # Constant input current
vsum = 0  # Variable to sum input from connected neurons

# 5) DBS (Deep Brain Stimulation) signals
# Four different DBS signals (can be used to stimulate neuron clusters)
dbs1 = np.zeros(T)  # Initialize DBS signal as zeros
for bb in range(T - 1):
    if bb % 20 == 0 and bb > 100:  # Apply DBS at intervals
        dbs1[bb] = 0

dbs2 = np.zeros(T)
dbs3 = np.zeros(T)
dbs4 = np.zeros(T)

# 6) Phase calculation setup
pcntMax = int(T / 10)  # Maximum number of peaks for a neuron
sss = [nn, pcntMax]
ptime = np.zeros(sss).astype(int)  # Timestamps of neuron spikes (converted to int)
pcnt = np.zeros(nn).astype(int)  # Counter for spikes per neuron

# 7) Synaptic plasticity function (STDP - Spike Timing Dependent Plasticity)
def stdp(td, w):
    if td > 0:
        tauu = 17  # Positive time difference
        w = w + (0.001 * (e**(td / tauu))) * w
    elif td < 0:
        tauu = -35  # Negative time difference
        w = w - (0.001 * 0.5 * (e**(-td / tauu))) * w
    return w

# 8) Simulation loop (time evolution)
for t in arange(T - 1):
    for j in range(nn):
        # Set initial conditions for each neuron
        v[j][0] = vIni[j]  # Resting membrane potential
        u[j][0] = uIni[j]  # Initial recovery variable

        if v[j][t] < 35:  # Check if neuron is below firing threshold
            vsum = 0  # Reset sum of connected neurons' influence
            for i in range(nn):
                vsum += ajm[j][i] * v[i][t - tau]  # Sum weighted inputs
            
            # Izhikevich model membrane potential update equation
            v[j][t + 1] = v[j][t] + 0.5 * (0.04 * v[j][t]**2 + 5 * v[j][t] + 140 - u[j][t] + II) + vsum + dbs1[t + 1]
            u[j][t + 1] = u[j][t] + a * (b * v[j][t] - u[j][t])  # Recovery variable update

        else:
            # 8.1) Neuron spikes!
            v[j][t] = 35  # Cap potential at spike threshold
            v[j][t + 1] = c  # Reset potential
            u[j][t + 1] = u[j][t] + d  # Recovery variable update after spike

            phase[j][t] = 0  # Reset phase
            pcnt[j] += 1  # Increment spike count
            ptime[j][pcnt[j]] = t  # Store spike time

            # Apply STDP updates based on spike timing difference
            for i in range(nn):
                td = ptime[j][pcnt[j]] - ptime[i][pcnt[i]]  # Time difference between spikes
                if ajm[j][i] > 0:
                    ajm[j][i] = stdp(td, ajm[j][i])
                if ajm[i][j] > 0:
                    ajm[i][j] = stdp(-td, ajm[i][j])
        
        # 8.2) Calculate phase difference if past spikes exist
        if ptime[j][pcnt[j]] != 0 and ptime[j][pcnt[j] - 1] != 0:
            phase[j][t] = 2 * 3.14 * ((t - (ptime[j][pcnt[j]])) / (ptime[j][pcnt[j]] - ptime[j][pcnt[j] - 1]))
        else:
            phase[j][t] = 0

# 9) Plot simulation results
tvec = arange(0, tmax, dt)  # Time vector

# Plot membrane potential of neurons for a specific time window
plot(tvec[3000:3100], v[0][3000:3100], 'g')
plot(tvec[3000:3100], v[1][3000:3100], 'b')
plot(tvec[3000:3100], v[2][3000:3100], 'r')
plot(tvec[3000:3100], v[3][3000:3100], 'm')
plot(tvec[3000:3100], v[4][3000:3100], 'c')
plot(tvec[3000:3100], dbs1[3000:3100], 'k')  # Plot DBS signal

print("FIN")  # End of simulation
