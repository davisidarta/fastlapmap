[![Latest PyPI version](https://img.shields.io/pypi/v/fastlapmap.svg)](https://pypi.org/project/fastlapmap/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/DaviSidarta.svg?label=Follow%20%40davisidarta&style=social)](https://twitter.com/davisidarta)
        
# Fast Laplacian Eigenmaps in python

Open-source [Laplacian Eigenmaps](https://www2.imm.dtu.dk/projects/manifold/Papers/Laplacian.pdf) algorithm for dimensionality reduction of large data in python. Comes with an
 wrapper for [NMSlib](https://github.com/nmslib/nmslib) to compute approximate-nearest-neighbors.
Performs several times faster than the default [scikit-learn implementation](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html).    

`fastlapmap` was developed by [Davi Sidarta-Oliveira](https://twitter.com/davisidarta). 

# Installation

You'll need NMSlib for using this package properly. Installing it with no binaries is recommended if your CPU supports
 advanced instructions (it problably does): 

```
pip3 install --no-binary :all: nmslib
# Along with requirements:
pip3 install numpy pandas scipy scikit-learn 
```

Then you can install this package with pip:

```
pip3 install fastlapmap
```
