# Classification and Regression, comparison of linear and logistic regression to feed forward neural networks
This study aims to investigate Feed-Forward Neural Networks (FFNNs) to existing stochastic gradient descent (SGD) regression methods and logistic regression for classification problems. Two datasets were analyzed, one generated using the Franke function, and the Wisconsin breast cancer data set. 
For the regression on the Franke function, we found a theoretical minimum MSE of $2.8 \cdot 10^{-3}$ using OLS. When we used the SGD method RMSprop on the OLS cost function, we achieved an MSE of $9.0 \cdot 10^{-3}$. Using RMSprop on the Ridge cost function with $\lambda = 3.6 \cdot 10^{-6}$ we achieved an MSE of $7.7 \cdot 10^{-3}$, and using an FFNN with the ReLu activation function we achieved an MSE of $2.1 \cdot 10^{-3}$. Further, for the Wisconsin breast cancer data set both logistic regression and FFNNs achieved a maximum accuracy score of $0.993$. Making a rigorous analysis of FFNNs, the study concludes that FFNNs are not necessarily the best-suited method for solving these problems with the dataset studied. 

## Installation 
Code was ran with python 3.8 on Ubuntu 20.04.5 LTS, with a Intel® Core™ i7-9750H CPU @ 2.60GHz × 12, to install packages run command

```Terminal
> pip install -r requirements.txt
```


## Generate data and plots from report
Run commands

```Terminal
> python3 plotter_FFNN.py
> python3 plotter_SGD.py
```