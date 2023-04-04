from __future__ import print_function
from dataset import Dictionary
import os
import sys
import json
import argparse
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='/data_dir/annotations/')
    parser.add_argument('--dictionary_path', default='/data_dir/data/')
    parser.add_argument('--embed_path', default='/data_dir/embeddings/')
    parser.add_argument('--embed_dim', default=300)
    args = parser.parse_args()
    return args


def create_dictionary(dataroot, files):
    dictionary = Dictionary()
    for path in files:
        question_path = os.path.join(dataroot, path)
        qs = json.load(open(question_path))['questions']
        for q in qs:  # q: {'image_id': 393227, 'question': 'What is this man riding on?', 'question_id': 393227001}
            # q['question']: 'What is this man riding on?'
            dictionary.tokenize(sentence=q['question'], add_word=True)
            # token: [0, 1, 2, 3, 4, 5, 6]
            # ex: What: 0, is: 1, this: 2, ...
    dictionary.tokenize(sentence='wordmask', add_word=True)
    return dictionary


def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        # entry: word, [tensor] (ex: the, [0.3, 0.4, 0.5])
        vals = entry.split(' ')
        word = vals[0]
        vals = np.array(vals[1:], dtype='float64')
        word2emb[word] = vals
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


def get_filenames(dataset):
    if dataset == 'v1':
        return [
            'OpenEnded_mscoco_train2014_questions.json',
            'OpenEnded_mscoco_val2014_questions.json',
            'OpenEnded_mscoco_test2015_questions.json',
            'OpenEnded_mscoco_test-dev2015_questions.json'
        ]
    else:
        return [
            'v2_OpenEnded_mscoco_train2014_questions.json',
            'v2_OpenEnded_mscoco_val2014_questions.json',
            'v2_OpenEnded_mscoco_test2015_questions.json',
            'v2_OpenEnded_mscoco_test-dev2015_questions.json'
        ]


def main():
    args = parse_args()
    files = get_filenames('cpv2')
    d = create_dictionary(args.dataroot, files=files)
    d.dump_to_file(os.path.join(args.dictionary_path, f'dictionary_cpv2.pkl'))

    d = Dictionary.load_from_file(os.path.join(
        args.dictionary_path, f'dictionary_cpv2.pkl'))
    emb_dim = args.embed_dim
    glove_path = args.embed_path
    glove_file = os.path.join(glove_path, f'glove.6B.{emb_dim}d.txt')
    weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)
    glove_np_path = os.path.join(
        glove_path, f'glove6b_init_{emb_dim}d_cpv2.npy')
    np.save(glove_np_path, weights)
    print('saved np embedding file to %s' % glove_np_path)


if __name__ == '__main__':
    main()
