# Implementaion paper "Deep Neural Networks as Gaussian Processes"
This is an informal implementation of [[1711.00165] Deep Neural Networks as Gaussian Processes](https://arxiv.org/abs/1711.00165).

## Usage
```
python3 main.py -mode ["dnn" or "rbf"] -iter_size [int]
```
for example
```
python3 main.py -mode dnn -iter_size 20
```

## Target data
<a href="https://www.codecogs.com/eqnedit.php?latex=5\cdot&space;sin(\pi/15x)&space;\cdot&space;exp(-x/50)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?5\cdot&space;sin(\pi/15x)&space;\cdot&space;exp(-x/50)" title="5\cdot sin(\pi/15x) \cdot exp(-x/50)" /></a>

## Implementerd kernels of gaussian processes
### rbf kernel
![dnn_kernel](https://github.com/onozeam/DNNasGP/blob/image/rbf_kernel.png)

### dnn kernel
![dnn_kernel](https://github.com/onozeam/DNNasGP/blob/image/dnn_kernel_with_relu.png)

## MCMC
To optimize hyper parameters of kernels, We use MCMC sampling. In this code, We adopted `Metropolis-Haistings`.
