#!/usr/bin/env bash

# example of usage 
## suppress warnings on call
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

#kill $!; trap 'kill $!' SIGTERM
#echo
#echo ':)'

end=`date +%s`
runtime=$((end-start))
echo $runtime
