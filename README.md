# Machine Learning Engineer Nanodegree
## Capstone Project
Empirical Dynamic Modeling for [Systems Biology](https://en.wikipedia.org/wiki/Systems_biology) with Machine Learning  
Kevin Siswandi  
May 2020  
https://github.com/Physicist91/ml-engineering-capstone

### Software and Libraries

* Deep Learning Framework: TensorFlow 2.0
* Numerics Framework: NodePy, SymPy
* Scientific Computing Stack: SciPy, NumPy

### References

The primary materials I consult are
* [MultiStep Neural Network](https://maziarraissi.github.io/research/7_multistep_neural_networks/)
* [Systems Identification](https://www.mathworks.com/help/ident/gs/about-system-identification.html)
* [Numerical Analysis of ODE](https://www.mathsim.eu/~gkanscha/notes/ode.pdf)

The motivation of this project is derived from
* Villaverde, A. F. & Banga, J. R. Reverse engineering and identification in systems biology: strategies, perspectives and challenges. J. R. Soc. Interface 11, 20130505 (2013).
* [Universal Differential Equations for Scientific Machine Learning](https://arxiv.org/abs/2001.04385)

### Brief Overview

This project is a cross-over between dynamical system and machine learning, with an application to the biosciences. It is very close to the state-of-the-art research currently being conducted in the field. Traditionally, such a method for discovering dynamics was known as *system identification* before machine learning libraries were made open-source commodity. However, systems identification is recognized as a hard problem in the physical sciences. Here, I want to show that a machine learning approach can help accelerate and transform how dynamic modeling is done in the hard sciences.

### Solution Statement

The MultiStep Neural Network will take in the time-series data as input and learns the function/derivative that describes the dynamics. Before it can be used to make predictions, the function must be integrated (using `scipy`). The benefit of this method is that it allows a full characterization of how the system will develop in time given only some initial values, which has plenty of use cases in bioengineering.



### Project Design

The general workflow is:
1. Generation of training and test data from mathematical models (tools needed: numpy, scipy)
2. Construction of MultiStep Neural Network (tools needed: TensorFlow 2.0, sympy)
3. I will do step 1-2 for three regimes: sustained oscillation, damped oscillation, and state transition in yeast glycolysis.
4. Numerical integration of the dynamics (tools needed: scipy).
5. Determination of influential hyperparameters (step size, number of steps, multistep scheme, etc.)
6. Qualitative comparison in time-course plot and phase space.

I should remark that some terminologies used here are specific to the domain of systems biology and dynamical systems.
