# Project 1 Regression analysis and resampling methods
This study aims to investigate various regression methods, focusing predominantly on Ordinary Least Squares
(OLS), Ridge, and Lasso regression, for both synthetic and real digital terrain data. Through the application of
these methods on Franke’s function, a well-known test function, we gain insights into the strengths and weaknesses
of each model. Resampling techniques, such as Bootstrap and Cross-Validation, are employed to provide a robust
evaluation of the models. The analysis reveals that while OLS performs well for simpler models, Ridge and Lasso
regression offers better resistance to overfitting for complex models. The Bias-Variance tradeoff is also examined,
demonstrating the impact of model complexity on prediction accuracy. Lastly, the models are applied to real
digital terrain data, offering promising results despite inherent challenges such as noise and irregularities. The
study concludes with an identification of limitations and suggestions for future research.
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