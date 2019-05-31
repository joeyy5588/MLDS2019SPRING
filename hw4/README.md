# MLDS HW4

## HW4-1
### 0. Demo
<img src="assets/4-1-demo.gif">

### 1. Model
```python
nn.Linear(80 * 80 * 1, 256),
nn.ReLU(True),
nn.Linear(256, 256),
nn.ReLU(True),
nn.Linear(256, 1),
nn.Sigmoid()
```
### 2. Plot
<img src="assets/4-1-plot.png">

### 3. Inference

```
Run 30 episodes
Mean: 8.133333333333333
```