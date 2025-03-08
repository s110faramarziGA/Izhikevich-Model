Neural Network Simulation Using the Izhikevich Model

Overview

This script simulates a small network of neurons using the Izhikevich neuron model. It initializes neuron states, defines connectivity, applies deep brain stimulation (DBS), and incorporates spike-timing-dependent plasticity (STDP). The simulation tracks membrane potential, phase, and spike timing, visualizing neuron activity over time.

Features

Implements the Izhikevich neuron model to simulate spiking behavior.

Defines a network of 6 neurons with adjustable connectivity.

Applies deep brain stimulation (DBS) to study its effects on neuron dynamics.

Incorporates spike-timing-dependent plasticity (STDP) to model synaptic changes.

Plots membrane potential of neurons over time for visualization.

Requirements

This script requires Python and the following libraries:

numpy

pylab

matplotlib

To install missing dependencies, run:

pip install numpy matplotlib

Usage

Set simulation parameters such as tmax (total simulation time) and dt (time step size).

Adjust neuron parameters (e.g., a, b, c, d) if needed.

Modify the adjacency matrix (ajm) to define neuron connectivity.

Run the script:

python neural_simulation.py

View the plotted neuron membrane potentials and DBS signals.

Output

Membrane potential plots for different neurons.

DBS signal visualization.

Spike timing and phase tracking.

License

This project is open-source and can be modified or distributed freely.

Author

Originally developed by s110f and sadegh-pc.
