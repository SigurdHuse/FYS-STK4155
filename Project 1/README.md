# Project 1 Regression analysis and resampling methods
This study rigorously examines the performance of Ordinary Least Squares (OLS), Ridge, and Lasso regression methods on synthetic data generated using the Franke function and real-world terrain data. For the synthetic data, all methods performed optimally at $\lambda \approx 10^{-5}$, with OLS yielding the most promising results on real-world data, achieving an MSE of $4.12\cdot10^4$ and an R2-score of $0.54$. Utilizing resampling techniques such as bootstrapping and k-fold cross-validation, we dissect the bias-variance tradeoff and demonstrate the effects of model complexity on generalization to unseen data. The study also explores the limitations of the current models and suggests avenues for future research, including advanced feature selection techniques and hyperparameter optimization for Ridge and Lasso regression.
## Installation 
Code was ran with python 3.8 on Ubuntu 20.04.5 LTS, with a Intel® Core™ i7-9750H CPU @ 2.60GHz × 12, to install packages run command

```Terminal
> pip install -r requirements.txt
```

## Generate data and plots from report
Run commands

```Terminal
> python3 plotter.py
> python3 real_world_data.py
```