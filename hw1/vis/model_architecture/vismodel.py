import os
import pydot_ng as pydot
from model.model import *
import torch.nn as nn
from summary import summary
import json

##################################

# pydot and graphviz is required #
# 1. install graphviz.
# 2. install pydot.
# 3. (Not necessary) go to where you install pydot and edit pydot.py
# find function find_graphviz() and change path to your executable file.

##################################

def plot_model(input_shape, model, rankdir='TB', dpi=96):
    try:
        # Attempt to create an image of a blank graph
        # to check the pydot/graphviz installation.
        pydot.Dot.create(pydot.Dot())
    except OSError:
        raise OSError(
            '`pydot` failed to call GraphViz.'
            'Please install GraphViz (https://www.graphviz.org/) '
            'and ensure that its executables are in the $PATH.')
    dot = pydot.Dot()
    dot.set('rankdir', rankdir)
    dot.set('concentrate', True)
    dot.set('dpi', dpi)
    dot.set_node_defaults(shape='record')
    # Create OrderedDict.
    x = summary(input_shape, model)
    # Create graph nodes.
    layer_list = []
    for layer in x:
        layer_list.append(layer)
        layer_id = str(id(layer))
        # Create Table.
        label = '%s\n|{input:|output:}|{{%s}|{%s}}' % (layer, x[layer]['input_shape'], x[layer]['output_shape'])
        node = pydot.Node(layer_id, label=label)
        dot.add_node(node)
    # Create Edge.
    for i, layer in enumerate(x):
        if i == 0:
            continue
        else:
            layer_id = str(id(layer))
            previous_id = str(id(layer_list[i-1]))
            dot.add_edge(pydot.Edge(previous_id, layer_id))
    dot.write_png('%s.png' % ('MnistModel')) 
    return dot
model = MnistModel()
'''
for i, layer in enumerate(model.modules()):
    if i < 2:
        continue
    if i == 1:
        for name, layers in layer.features.named_modules():
            print('{}, type: {}'.format(name, type(layers)))
    elif isinstance(layer, nn.Linear):
        print(i, layer.in_features, layer.out_features)
'''
plot_model([1, 28, 28], model)