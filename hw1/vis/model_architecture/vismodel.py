import os
import pydot_ng as pydot
from model.model import *
import torch.nn as nn
from vis.model_architecture.summary import summary
import json
from collections import Counter

##################################

# pydot and graphviz is required #
# 1. install graphviz.
# 2. install pydot.
# 3. (Not necessary) go to where you install pydot and edit pydot.py
# find function find_graphviz() and change path to your executable file.

##################################

def plot_model(input_shape, model, show_shape = True, rankdir='TB', dpi=96):
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
    # Create layer order
    layer_list = []
    for layer in x:
        layer_list.append(layer)
    layer_list = [layer.split('-')[0] for layer in layer_list]
    count_list = []
    layer_dict = {}
    name_list = []
    for layer in layer_list:
        if layer in layer_dict:
            layer_dict[layer] += 1
            count_list.append(layer_dict[layer])
        else:
            layer_dict[layer] = 1
            count_list.append(1)
    for i, layer in enumerate(layer_list):
        name_list.append(layer + '_' + str(count_list[i])) 
    # Create graph nodes.
    layer_list = []
    for i, layer in enumerate(x):
        layer_list.append(layer)
        layer_id = str(id(layer))
        # Create Table.
        if show_shape:
            label = '%s\n|{input:|output:}|{{%s}|{%s}}' % (name_list[i], x[layer]['input_shape'], x[layer]['output_shape'])
        else:
            label = layer
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
    dot.write_png('%s.png' % ('vis/model_architecture/CNN-3')) 
    return dot
model = CNN3()
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
