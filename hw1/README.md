# MLDS HW1
## 1-1
### 1-1-1: plot the function and the predicted result
1. Train the model and the checkpoint will automatically save the data for you.
```
python3 train.py --config config_func.json --type [DNN2|DNN5|DNN8]
```
2. Run ``plot_function.py`` to visualize the result
```
cd vis/hw1-1
python3 plot_function.py (裡面路徑路徑要改一下qq 可以把它寫成arg)
```
### 1-1-2: plot the loss and the acc of the training process
1. Train the model and the checkpoint will automatically save the data for you.
```
# func
python3 train.py --config config_func.json --type [DNN2|DNN5|DNN8]
# mnist
python3 train.py --config config_mnist.json --type [CNN1|CNN2|CNN3]
```
2. Run ``plot_loss_acc.py`` to visualize the result
```
cd vis/hw1-1
python3 plot_loss_acc.py (裡面路徑路徑要改一下qq)
```
## 1-2
### 1-2-1: show that different optimization process will converge to the same point
1. train the model
```
# mnist
python3 train.py --config config_mnist.json --type [CNN1|CNN2|CNN3]
python3 train.py --config config_mnist.json --type [CNN1|CNN2|CNN3]
...(做8次)
```
2. Run ``plot_optim_process.py`` to visualize the result
```
cd vis/hw1-2
python3 plot_optim_process.py (裡面路徑路徑要改一下qq)
```
## 1-3