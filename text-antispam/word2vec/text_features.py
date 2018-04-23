import numpy as np
import tensorlayer as tl

wv = tl.files.load_npy_to_any(name='./output/model_word2vec_200.npy')
for label in ['pass', 'spam']:
    samples = []
    inp = 'data/msglog/msg' + label + '.log.seg'
    outp = 'output/sample_seq_' + label
    with open(inp) as inp_file:
        lines = inp_file.readlines()
        for line in lines:
            words = line.strip().split(' ')
            text_sequence = []
            for word in words:
                try:
                    text_sequence.append(wv[word])
                except KeyError:
                    text_sequence.append(wv['UNK'])
            samples.append(text_sequence)

        if label is 'spam':
            labels = np.zeros(len(samples))
        elif label is 'pass':
            labels = np.ones(len(samples))

        """
            Save several arrays into a single file in uncompressed ``.npz`` format
        """
        np.savez(file=outp, x=samples, y=labels)
