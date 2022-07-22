# Quantifying the Uncertainty and Global Sensitivity of Quantum Computations on Experimental Hardware

This repository provides the implementation of Global Sensitivity Analysis approach utilized in our manuscript "Quantifying the Uncertainty and Global Sensitivity of Quantum Computations on Experimental Hardware" by Guanglei Xu, Kalpana Hanthanan Arachchilage, M.Y. Hussaini, and William S. Oates. 

This folder consists of the Matlab files used in uncertainty quantification analysis presented in the abovementioned paper.

## main.m file
This is the executable file. 

## getModelResponse*.m files
These files estimate the quantum probabilities used for function evaluations. 

## getModelResponseError*.m files
These files estimate the errors corresponding to the measurement and the model response. 

## IBM_full_data_athens.mat file
This file consists of the experimental measurements of the eight quantum state probabilities. 


## output 
The main code has two outputs
1. stat.mat file that consists of all the statistical estimations required for the global sensitivity analysis. 
2. allChain.mat file that consisits of all the Markov chains obtained in the UQ analysis.
