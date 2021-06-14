# Nonlinear Level Set Learning for Function Approximation on Sparse Data

This is an implementation of the algorithm from:

A. Gruber, M. Gunzburger, L. Ju, Y. Teng, and Z. Wang.  [Nonlinear Level Set Learning for Function Approximation on Sparse Data with Applications to Parametric Differential Equations](https://arxiv.org/pdf/2104.14072.pdf),

which learns an intrinsically one-dimensional representation of a (user or data defined) scalar-valued function of many variables.

The main file is "run_algorithm.py", and "options.py" contains the hyperparameters that will be passed when the program is instantiated.  These should be modified according to your particular needs.

## Installation
The code was written and tested using Python 3.8 on Mac OSX 11.2.3.  The required dependencies are
* [Numpy](https://numpy.org/)
* [Pytorch](https://pytorch.org/)
* [Matplotlib](https://matplotlib.org/)
* [Scipy](https://www.scipy.org/) (for polynomial regression)
* [Scikit-learn](https://scikit-learn.org/stable/) (for nearest neighbors regression)
* [pyDOE](https://pythonhosted.org/pyDOE/) (optional:  for latin hypercube sampling)

## Data
To use your own dataset, you should provide either:
- a scalar-valued function implemented in "functions.py"
- a filepath to an array of length nSamples with feature blocks structured like  [domain_variables, function_outputs, gradients].
Take a look at the function preprocess_data implemented in "trainer.py", as well as the files in the data directory for an example.

## Citation
Please cite [our paper](https://arxiv.org/pdf/2104.14072.pdf) if you use this code in your own work:
```
@article{gruber2021nonlinear,
  title={Nonlinear Level Set Learning for Function Approximation on Sparse Data with Applications to Parametric Differential Equations},
  author={Gruber, Anthony and Gunzburger, Max and Ju, Lili and Teng, Yuankai and Wang, Zhu},
  journal={arXiv preprint arXiv:2104.14072},
  year={2021}
}
