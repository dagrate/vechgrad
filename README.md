# VecHGrad: Vector Hessian Gradient product with strong Wolfe's line search to solve accurately complex tensor decomposition and ML/DL applications

<p align="middle">
  <img src="https://github.com/dagrate/vechgrad/blob/master/images/bus_als.png" width="100" />
  <img src="https://github.com/dagrate/vechgrad/blob/master/images/bus_sgd.png" width="100"/>
  <img src="https://github.com/dagrate/vechgrad/blob/master/images/bus_nag.png" width="100"/>
  <img src="https://github.com/dagrate/vechgrad/blob/master/images/bus_adam.png" width="100"/>
  <img src="https://github.com/dagrate/vechgrad/blob/master/images/bus_rmsprop.png" width="100"/>
</p>

<p align="middle">
  <img src="https://github.com/dagrate/vechgrad/blob/master/images/bus_saga.png" width="100" />
  <img src="https://github.com/dagrate/vechgrad/blob/master/images/bus_adagrad.png" width="100"/>
  <img src="https://github.com/dagrate/vechgrad/blob/master/images/bus_ncg.png" width="100"/>
  <img src="https://github.com/dagrate/vechgrad/blob/master/images/bus_bfgs.png" width="100"/>
  <img src="https://github.com/dagrate/vechgrad/blob/master/images/bus_vechgrad.png" width="100"/>
</p>

*From left to right: ALS, SGD, NAG, ADAM, RMSPROP, SAGA, ADAGRAD, NCG, BFGS, VECHGRAD*

VecHGrad, a numerical optimizer, is part of a Julia library that proposes to evaluate the convergence and the strengths of the different numerical optimizers used in machine learning and deep learning in the context of linear algebra and tensors. We assess the accuracy of SGD, NAG, ADAM, RMSPROP, SAGA, ADAGRAD, NCG, BFGS and VecHGrad on popular data sets including CIFAR-10, CIFAR-100, MNIST, COCO, LFW. Since we perform our experiments on linear algebra, we also included the ALS method. The strength of VecHGrad is to include Hessian approximate information and an adaptive line search that relies on the strong Wolfe's line search for faster convergence.

N.B. To be able to reproduce the experiments on (almost) any computer, we have reduced the size of the data sets originally presented in our paper. However, the conclusions and the findings remain identical.

----------------------------

## Dependencies

The library relies on **Julia**. Julia is a powerfull language for all numeric applications, that is compatible multiprocessing and GPUs for accelerated computing. For a GPU implementation, we have to rely on the ArrayFire module. 

```bibtex
@article{bezanson2017julia,
  title={Julia: A fresh approach to numerical computing},
  author={Bezanson, Jeff and Edelman, Alan and Karpinski, Stefan and Shah, Viral B},
  journal={SIAM review},
  volume={59},
  number={1},
  pages={65--98},
  year={2017},
  publisher={SIAM}
}
```

----------------------------

## Citing

If you use the repository, please cite:

```bibtex
To Be Uploaded Soon
```
