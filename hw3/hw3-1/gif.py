import imageio
import os
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='result/exp_4/png')
parser.add_argument('--output', type=str, default='result/exp_4/process.gif')

opt = parser.parse_args()

file_names = sorted(
    (os.path.join(opt.dir ,fn) for fn in os.listdir(opt.dir) if fn.endswith('.png')),
    key = lambda x: int(re.findall(r'\d+', x)[0])
)
images = [imageio.imread(fn) for fn in file_names]
filename = opt.output
imageio.mimsave(filename, images, duration = 0.5)