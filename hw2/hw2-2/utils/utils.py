import os
import json
import re

def sentence_to_indexs(sen, w2i, pad = 15):
    """
    Usage:
        convert a sentence to index array.
    Example:
        pad = 5
        I am byron. -> [I, am, byron, eos, pad] -> correspondind indexs
    """
    i_list = []
    pad_idx = w2i['<PAD>']
    sen = re.sub('[\n]', '', sen)
    
    for (i, w) in enumerate(sen.split(' ')):
        if i >= pad - 1: break
        i_list.append(w2i[w])

    i_list.append(w2i['<EOS>'])

    while(len(i_list) < pad):
        i_list.append(pad_idx)
    
    return i_list


def ensure_dir(dir):
    os.makedirs(dir, exist_ok=True)
