from nltk.tokenize import word_tokenize

import os
import json
import pickle
import copy
from collections import Counter

import numpy as np
import utils
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('-',
                                                                                             ' ').replace('.', '').replace('"', '').replace('n\'t', ' not').replace('$', ' dollar ')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                if w in self.word2idx:
                    tokens.append(self.word2idx[w])
                else:
                    tokens.append(len(self.word2idx))
        return tokens

    def dump_to_file(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        pickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = pickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img_idx, question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id': question['question_id'],
        'image_id': question['image_id'],
        'image_idx': img_idx,
        'question': question['question'],
        'answer': answer
    }
    return entry


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, img_path, data_path,
                 annotations_path, keep_qtype=None,
                 optim_run=False):
        super(VQAFeatureDataset, self).__init__()
        self.name = name
        self.img_path = img_path
        self.data_path = data_path
        self.annotations_path = annotations_path
        self.keep_qtype = keep_qtype
        self.optim_run = optim_run
        self.max_length = 14

        self._load_masks()
        self._load_hintscores()

        assert name in ['train', 'val']

        ans2label_path = os.path.join(
            self.data_path, 'cpv2', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(
            self.data_path, 'cpv2', 'trainval_label2ans.pkl')

        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary
        imgid2idx = None

        self.entries = self._load_dataset(name, imgid2idx, dataset='cpv2')

        self.tokenize()
        self.tensorize()

        self.v_dim = 2048

    def _load_masks(self):
        type_mask = f'cpv2_type_mask.json'
        notype_mask = f'cpv2_notype_mask.json'
        with open(os.path.join(self.data_path, 'cpv2', type_mask), 'r') as f:
            self.type_mask = json.load(f)
        with open(os.path.join(self.data_path, 'cpv2', notype_mask), 'r') as f:
            self.notype_mask = json.load(f)

    def _load_hintscores(self):
        train_file = f'train_cpv2_hintscore.json'
        test_file = f'test_cpv2_hintscore.json'
        with open(os.path.join(self.data_path, 'cpv2', train_file), 'r') as f:
            self.train_hintscore = json.load(f)
        with open(os.path.join(self.data_path, 'cpv2', test_file), 'r') as f:
            self.test_hintsocre = json.load(f)

    def _load_dataset(self, name, img_id2val, dataset):
        """Load entries
        img_id2val: dict {img_id -> val} val can be used to retrieve image or features
        name: 'train', 'val'
        """
        answer_path = os.path.join(
            self.data_path, 'cpv2', '%s_target.pkl' % name)
        name = "train" if name == "train" else "test"
        question_path = os.path.join(
            self.annotations_path, 'vqacp_v2_%s_questions.json' % name)
        with open(question_path) as f:
            questions = json.load(f)

        with open(answer_path, 'rb') as f:
            answers = pickle.load(f)

        questions.sort(key=lambda x: x['question_id'])
        answers.sort(key=lambda x: x['question_id'])

        utils.assert_eq(len(questions), len(answers))
        entries = []
        for question, answer in zip(questions, answers):
            if answer["labels"] is None:
                raise ValueError()
            utils.assert_eq(question['question_id'], answer['question_id'])
            utils.assert_eq(question['image_id'], answer['image_id'])
            img_id = question['image_id']
            img_idx = None
            if img_id2val:
                img_idx = img_id2val[img_id]

            entries.append(_create_entry(img_idx, question, answer))
        return entries

    def tokenize(self):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in tqdm(self.entries, ncols=100, desc=" tokenize"):
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:self.max_length]
            if len(tokens) < self.max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * \
                    (self.max_length - len(tokens))
                padding_mask = [self.dictionary.padding_idx -
                                1] * (self.max_length - len(tokens))
                tokens_mask = padding_mask + tokens
                tokens = padding + tokens

            utils.assert_eq(len(tokens), self.max_length)
            entry['q_token'] = tokens
            entry['q_token_mask'] = tokens_mask

    def tensorize(self):
        for entry in tqdm(self.entries, ncols=100, desc="tensorize"):
            question = torch.from_numpy(np.array(entry['q_token']))
            question_mask = torch.from_numpy(np.array(entry['q_token_mask']))

            entry['q_token'] = question
            entry['q_token_mask'] = question_mask

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def mask_non_stopwords(self, question, q_token_mask, q_type):
        quest = question.lower()
        quest = quest.replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('-',
                                                                                       ' ').replace('.', '').replace('"', '').replace('n\'t', ' not').replace('$', ' dollar ')
        doc_q = word_tokenize(quest)
        token_idx_neg = np.ones(self.max_length)  # relevant words masked
        token_idx_pos = np.ones(self.max_length)  # irrelevant words masked
        q_type = q_type.split(' ')
        for i, word in zip(range(len(doc_q)), doc_q):
            if self.keep_qtype:
                if word in stopwords.words('english'):
                    if word not in q_type:
                        token_idx_pos[(self.max_length-len(doc_q))+i] = 0
                else:
                    if word not in q_type:
                        token_idx_neg[(self.max_length-len(doc_q))+i] = 0
            else:
                if word in q_type:
                    token_idx_neg[(self.max_length-len(doc_q))+i] = 0
                    token_idx_pos[(self.max_length-len(doc_q))+i] = 0
                if not word in stopwords.words('english'):
                    token_idx_neg[(self.max_length-len(doc_q))+i] = 0
                else:
                    token_idx_pos[(self.max_length-len(doc_q))+i] = 0

        # ones = torch.ones(q_token_mask.shape)
        # zeros = torch.zeros(q_token_mask.shape)
        q_neg_mask = copy.deepcopy(q_token_mask)
        token_idx_neg = (token_idx_neg == 0).nonzero()[0]
        token_idx_neg = torch.from_numpy(token_idx_neg).long()
        # using padding_idx (what is use padding_idx-1?)
        q_neg_mask.scatter_(0, token_idx_neg, self.dictionary.padding_idx)
        q_neg_mask[q_neg_mask == self.dictionary.padding_idx -
                   1] = self.dictionary.padding_idx
        # q_neg_mask_type = torch.where(q_neg_mask == self.dictionary.padding_idx, zeros, ones)
        q_pos_mask = copy.deepcopy(q_token_mask)
        token_idx_pos = (token_idx_pos == 0).nonzero()[0]
        token_idx_pos = torch.from_numpy(token_idx_pos).long()
        # using padding_idx (what is use padding_idx-1?)
        q_pos_mask.scatter_(0, token_idx_pos, self.dictionary.padding_idx)
        q_pos_mask[q_pos_mask == self.dictionary.padding_idx -
                   1] = self.dictionary.padding_idx
        # q_pos_mask_type = torch.where(q_pos_mask == self.dictionary.padding_idx, zeros, ones)
        return (q_neg_mask, q_pos_mask)

    def __getitem__(self, index):
        entry = self.entries[index]
        img_id = entry["image_id"]
        try:
            img_name = f"COCO_train2014_{img_id:012d}"
            features = torch.load(
                self.img_path+str(img_name)+'.jpg.pth')['pooled_feat']
        except:
            img_name = f"COCO_val2014_{img_id:012d}"
            features = torch.load(
                self.img_path+str(img_name)+'.jpg.pth')['pooled_feat']

        q_id = entry['question_id']
        ques_mask = entry['q_token_mask']
        if self.optim_run and self.name == 'train':
            ques_mask = self.mask_non_stopwords(
                entry['question'], entry['q_token_mask'], entry['answer']['question_type'])
        ques = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)

        if self.name == 'train':
            train_hint = torch.tensor(self.train_hintscore[str(q_id)])
            type_mask = torch.tensor(self.type_mask[str(q_id)])
            notype_mask = torch.tensor(self.notype_mask[str(q_id)])
            if "bias" in entry:
                return features, ques, target, entry["bias"], train_hint, type_mask, notype_mask, ques_mask

            else:
                return features, ques, target, 0, train_hint
        else:
            test_hint = torch.tensor(self.test_hintsocre[str(q_id)])
            if "bias" in entry:
                return features, ques, target, entry["bias"], q_id, test_hint
            else:
                return features, ques, target, 0, q_id, test_hint

    def __len__(self):
        return len(self.entries)
