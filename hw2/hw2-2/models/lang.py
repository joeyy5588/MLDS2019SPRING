from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import get_tmpfile
import re
import os

class myDict(dict):
    def __init__(self, *arg, **kw):
        """
        Usage:
            Customize 'dict' in order to handle '<UNK>'.
            When it receives an unknown word, it will return the index of '<UNK>'.
        """
        super(myDict, self).__init__(*arg, **kw)
    def __getitem__(self, key):
        if key not in self:
            return super(myDict, self).__getitem__('<UNK>')
        else:
            return super(myDict, self).__getitem__(key)

class Lang():
    def __init__(self, model_path, data_dir = 'data', min_count = 5, pretrain = False):
        """
        Language Model
        Usage:
            1. Map word to index, and also index to word. 
            2. Use gensim to generate word vectors.
        Argument:
            1. data_dir: The path to data folder. ex: data/
            2. model_path: The path to Lang model.
            3. pretrain: Whether the model is pretrained.
               If not pretrained, we will train a new one.
        """
        self.min_count = min_count
        
        corpus = self._read_data(data_dir)
        model = None
        if pretrain == False:
            model = Word2Vec(corpus, size=256, window=5, min_count=min_count, workers=4)
            model = model.wv
            model.save(model_path)
        else:
            model = KeyedVectors.load(model_path, mmap='r')
        self.model = model
        # public member variable, which can be accessed from outside.
        self.w2i = self._build_w2i()
        self.i2w = self._build_i2w()
        self.embed = model.syn0

    def _build_i2w(self):
        i2w = self.model.index2word
        return i2w
    
    def _build_w2i(self):
        w2i = myDict([(k, v.index) for (k, v) in self.model.vocab.items()])
        return w2i
    
    def _read_data(self, data_dir):
        sens = []
        for file in ['clr_conversation.txt', 'test_input.txt']:
            text_file = os.path.join(data_dir, file)
            text_data = None
            with open(text_file) as f:
                text_data = f.readlines()
            for s in text_data:
                s = re.sub('[\n]', '', s)
                s = '<BOS> ' + s + ' <EOS>'
                sens.append(s.split(' '))
        # Add pad and unk to the model.
        sens.append(['<PAD>'] * self.min_count)
        sens.append(['<UNK>'] * self.min_count)
        return sens
