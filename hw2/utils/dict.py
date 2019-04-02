"""
In this file, 
we will read in training and testing data together to construct a word-to-index dictionary
"""
import os
import json
import re

def build_dict(data_dir):
    """
    Return:
        1. word_to_index dictionary
        2. index_to_word dictionary
    """
    # The order of label.json is same as id_list, so we can directly iterate the json
    pool = set()
    for mode in ['train', 'test']:
        text_file = os.path.join(data_dir, '{}ing_label.json'.format(mode))
        text_data = None
        with open(text_file) as f:
            text_data = json.load(f)
        for e in text_data:
            for s in e['caption']:
                s = re.sub('[.,!]', '', s)
                for w in s.split(' '):
                    pool.add(w)
    # add special word
    w2i, i2w = {}, {}
    w2i['<PAD>'] = 0
    i2w[0] = '<PAD>'
    w2i['<BOS>'] = 1
    i2w[1] = '<BOS>'
    w2i['<EOS>'] = 2
    i2w[2] = '<EOS>'

    # It needs to be sorted!!
    # Otherwise, each training process will have different dictionary...
    for i, w in enumerate(sorted(pool)):
        w2i[w] = i + 3 # start from 3
        i2w[i + 3] = w

    return w2i, i2w

def sentence_to_indexs(sen, w2i, pad = 20):
    
    i_list = []
    pad_idx = 0
    sen = re.sub('[.,!]', '', sen)
    
    for (i, w) in enumerate(sen.split(' ')):
        if i >= pad - 1: break
        i_list.append(w2i[w])

    i_list.append(w2i['<EOS>'])

    while(len(i_list) < pad):
        i_list.append(pad_idx)
    
    return i_list


