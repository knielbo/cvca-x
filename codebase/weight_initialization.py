#!/home/knielbo/virtenvs/cv/bin/python
"""
Examples of approaches to weight initialization
"""
import numpy as np
def main():
    m = 64
    n = 32
    # constant initialization
    W = np.zeros((m,n))
    W = np.ones((m,n))
    C = 2
    W = W * C
    # uniform and normal dist initialization
    W = np.random.uniform(low=-.05, high=.05, size=(m,n))
    W = np.random.normal(.0, .05, size=(m,n))
    # LeCun uniform and normal
        # parameter F_{in} "fan in" or number of inputs to the layer
        #           F_{out} "fan out" or number og outputs from the layer
    F_in = 64
    F_out = 32
    limit = np.sqrt(3 / float(F_in))
        # uniform
    W = np.random.uniform(low=-limit, high=limit, size=(F_in, F_out))
        # normal
    limit = np.sqrt(1 / float(F_in))
    W = np.random.normal(.0, limit, size=(F_in, F_out))
    # Glorot/Xavier uniform and normal
        # normal
    limit = np.sqrt(2 / float(F_in + F_out))
    W = np.random.normal(.0, limit, size=(F_in, F_out))
        # uniform
    limit = np.sqrt(6 / float(F_in + F_out))
    W = np.random.uniform(low=-limit, high=limit, size=(F_in, F_out))
    # He et al./Kaiming/MSRA uniform and normal, for very deep models with ReLU-like 
        # activation function
        # uniform
    limit = np.sqrt(6 / float(F_in))
    W = np.random.uniform(low=-limit, high=limit, size=(F_in, F_out))
        # normal
    limit = np.sqrt(2 / float(F_in))
    W = np.random.normal(.0, limit, size=(F_in, F_out))
    
    print(W)

if __name__=="__main__":
    main()


