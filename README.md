# Machine Learning Engineer Nanodegree
## Capstone Project
Kevin Siswandi  
May 2020

## Software and Libraries

* Deep Learning Framework: TensorFlow 2.0
* Numerics Framework: NodePy, SymPy
* Scientific Computing Stack: SciPy, NumPy

## References

The primary materials I consult are
* [MultiStep Neural Network](https://maziarraissi.github.io/research/7_multistep_neural_networks/)
* [Systems Identification](https://www.mathworks.com/help/ident/gs/about-system-identification.html)

The motivation of this project is derived from
* Villaverde, A. F. & Banga, J. R. Reverse engineering and identification in systems biology: strategies, perspectives and challenges. J. R. Soc. Interface 11, 20130505 (2013).
* [Universal Differential Equations for Scientific Machine Learning](https://arxiv.org/abs/2001.04385)

## Brief Overview

This project is a cross-over between dynamical system and machine learning, with an application to the biosciences. It is very close to the state-of-the-art research currently being conducted in the field. Traditionally, such a method for discovering dynamics was known as *system identification* before machine learning libraries were made open-source commodity. However, systems identification is recognized as a hard problem in the physical sciences. Here, I want to show that a machine learning approach can help accelerate and transform how dynamic modeling is done in the hard sciences.

the objective is to find the function f that best represents the dynamics in the data. This is done by formulating a supervised learning problem embedded in the numerical framework of [linear multistep methods](https://en.wikipedia.org/wiki/Linear_multistep_method). I will build the multistep neural network with TensorFlow 2.0. RMSE will be used as the evaluation metric.

### Datasets and Inputs

I plan to use three set of parameter values for the ordinary differential equations of the dynamical system that will result in different qualitative behaviour of the system dynamics. I expect the multistep neural network to be robust enough to handle these three regimes.

The inputs are the simulated time-series data for yeast glycolysis. There is no further preprocessing because we already generate the data in regular time intervals.

### Solution Statement

The MultiStep Neural Network will take in the time-series data as input and learns the function/derivative that describes the dynamics. Before it can be used to make predictions, the function must be integrated (using `scipy`). The benefit of this method is that it allows a full characterization of how the system will develop in time given only some initial values, which has plenty of use cases in bioengineering.

### Benchmark Model

The classical benchmark model is the [implicit SINDy method](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7809160), which uses LASSO regression to find a sparse representation to a library of candidate functions. Note: it's highly mathematical and assumes prior knowledge about the dynamical systems.

The goal of this project is to demonstrate that Multistep Neural Network can provide better predictive power than implicit SINDy, given more data.


### Project Design

The general workflow is:
1. Generation of training and test data from mathematical models (tools needed: numpy, scipy)
2. Construction of MultiStep Neural Network (tools needed: TensorFlow 2.0, sympy)
3. I will do step 1-2 for three regimes: sustained oscillation, damped oscillation, and state transition in yeast glycolysis.
4. Numerical integration of the dynamics (tools needed: scipy).
5. Determination of influential hyperparameters (step size, number of steps, multistep scheme, etc.)
6. Qualitative comparison in time-course plot and phase space.

I should remark that some terminologies used here are specific to the domain of systems biology and dynamical systems.
