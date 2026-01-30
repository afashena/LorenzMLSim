# Applications of Neural Networks to Physical Modeling

This repo explores various applications of neural networks to physical modeling and simulation, including Neural ODEs and Physics Informed Neural Networks (PINNs).

This is a very exciting area of machine learning which dovetails well with my training in mathematics, and I will be progressing from toy examples to more realistic cases such as the kind seen in industry.

## Toy Example 1: 🍔 Simulating Burgers' 2D viscous flow with PINN

The Burgers' Equation models a simplified fluid flow that captures diffusion and advection effects. This PINN is currently trained on a single value of viscosity, but I will see how well it can model the effects of different viscosities by including it as an input.

![](https://github.com/afashena/LorenzMLSim/blob/main/PINN/toy_example/src/burgers2d_evolution_weights_big.gif)

## Toy Example 2: 🦋 Simulating the Lorenz equations with a Neural ODE

This project trains a model to predict the Lorenz system of differential equations. Being a chaotic system, the Lorenz equations cannot be predicted over a long time horizon with accuracy; however, I am doing this project as an opportunity to learn the approach, as well as to see how far I can take the time horizon before prediction fails.
