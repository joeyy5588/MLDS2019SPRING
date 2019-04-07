# Pytorch Simple Template
## File Structure
- Logic
Modules are only used together in root python file like ``train.py``.
They will not import each other.
- Package
1. ``loader``: dataset and dataloader
2. ``models``: networks, loss and metrics
3. ``trainer``: training process, save records(checkpoint, results...)
4. ``utils``: frequently used functions, arguments
- Execution
1. ``train.py``: main train logic
2. ``debug.py``: test each package independently

## Dependency

```bash
torchvision==0.2.1
torch==0.4.1
matplotlib==2.2.2
numpy==1.15.0
```

## Download model

```bash
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1b-42zONA1fsyWp8f6vXfWmCQOUOuKAh3' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1b-42zONA1fsyWp8f6vXfWmCQOUOuKAh3" -O checkpoint.pth && rm -rf /tmp/cookies.txt
```

## Execution
- Reproduce the results
```bash
python3 test.py --checkpoint <path1> --save_path <path2>
# example
python3 test.py --checkpoint a.pth --save_path result.txt
```

## Presentation slides
https://docs.google.com/presentation/d/199j581SGvtr707nRV6UXFT4-9bPza3AOELcRK6ydpwE/edit#slide=id.g559ce4d3fa_0_223

## Reference
https://github.com/victoresque/pytorch-template