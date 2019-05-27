from loader import Dataset

if __name__ == '__main__':
    D = Dataset('data', n_emb = 120)
    a, b = D.dic_hair, D.dic_eyes
    print(a, b)
    rgb = {'aqua': [0, 255, 255], 'black': [0, 0, 0], 'blonde': [226, 224, 110], 'blue': [0, 0, 255], 'brown': [160, 82, 45], 'gray': [192, 192, 192], 'green': [0, 128, 0], 'orange': [255, 165, 0], 'pink': [255, 192, 203], 'purple': [128, 0, 128], 'red': [255, 0 ,0], 'white': [255, 255, 255], 'yellow': [255, 255, 0]}