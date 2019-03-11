# MLDS HW1
## hw1(all)
1. Plot model architecture. (要改model路徑跟input大小)
```python
### In directory hw1 ###
python -m vis.model_architecture.vismodel
```
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
python3 plot_loss_acc.py
```
## 1-2

### 1-2-1: visualize the Optimization Process  2/3
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
python3 plot_optim_process.py
```
### 1-2-2: observe gradient norm during training
1. train the model
2. Run ``plot_optim_process.py`` to visualize the result
```
cd vis/hw1-2
python3 plot_grad_loss.py
```
### 1-2-3: what happens when grad_norm = 0
1. train the model
2. Run ``plot_optim_process.py`` to visualize the result
```
cd vis/hw1-2
python3 plot_hessian.py
python3 plot_min_ratio.py
```
## 1-3