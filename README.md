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

### How It Works

This project is a cross-over between dynamical system and machine learning, with an application to the biosciences. It is very close to the state-of-the-art research currently being conducted in the field. Traditionally, such a method for discovering dynamics was known as *system identification* before machine learning libraries were made open-source commodity. However, systems identification is recognized as a hard problem in the physical sciences. Here, I want to show that a machine learning approach can help accelerate and transform how dynamic modeling is done in the hard sciences.

The general workflow is:
1. Generation of training and test data from mathematical models (tools needed: numpy, scipy)
2. Construction of MultiStep Neural Network (tools needed: TensorFlow 2.0, sympy)
3. three regimes of interest: sustained oscillation, damped oscillation, and state transition in yeast glycolysis.
4. Numerical integration of the dynamics (tools needed: scipy).
5. Determination of influential hyperparameters (step size, number of steps, multistep scheme, etc.)
6. Qualitative comparison in time-course plot and phase space.
