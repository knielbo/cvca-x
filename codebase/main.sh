#!/usr/bin/env bash

# example of usage
start=`date +%s`
#echo pipeline init
#while true;do echo -n '>';sleep 1;done &

python knn.py --dataset ../datasets/animals
python linear_example.py
python gradient_descent.py
python sgd-py --batch-size 5
python regularization.py --dataset ../datasets/animals
python perceptron_or.py
python perceptron_and.py
python perceptron_xor.py
python nn_xor.py
python nn_mnist.py
python keras_mnist.py --output ../figures/keras_mnist.png
python keras_cifar10.py --output ../figures/keras_cifar10.png
python convolutions.py --image ../datasets/jemma.png

#kill $!; trap 'kill $!' SIGTERM
#echo
#echo ':)'

end=`date +%s`
runtime=$((end-start))
echo $runtime
