This Matlab code implements the transitional Bayesian Quadrature (TBQ) for Bayesian model inference, with the aim of estimating both model evidence and posterior densities which show complex features such as multimodality, high sharpness and nonlinear dependencies. For the theoretical details, please refer to my paper:
     P. Wei. Bayesian Model Inference with Complex Posteriors: Exponential-Impact-Informed Bayesian Quadrature. Mechanical Systems & Signal Processing, 2025

This code produces results for case 1 of the second example, but reader can also run it for the other examples. The following results will be produced for case 1 of the second example in the paper:

Total number of model calls： 31
Mean estimate of the model evidence： 0.01219139
gamma values:0.0000  0.0198  0.0657  0.1968  1.0000  
Credible intervals of model evidence:0.0119  0.0124  
Reference value of the model evidence： 0.01200019

, and also store the produced MCMC samples of each stage in the file "MCSampCase1.xlsx" and the generated training samples in the file "TtrainCase1.xlsx". It also produces the estimated (intermediate) posterior density of each tempering stage, together with the reference posterior density as well as the absolute error as follows.
![untitled](https://github.com/user-attachments/assets/860cd569-5e55-4840-bf7c-07a7e3578598)
