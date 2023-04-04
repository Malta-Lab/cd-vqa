import argparse
import click
from collections import defaultdict, Counter
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import os

from dataset import Dictionary, VQAFeatureDataset
from model import base_model
from train import cdvqa_train
import utils


def parse_args():
    parser = argparse.ArgumentParser(
        "Train the BottomUpTopDown model with a de-biasing method")

    # Standard args
    parser.add_argument('-p', "--entropy_penalty", default=0.36, type=float,
                        help="Entropy regularizer weight for the learned_mixin model")

    # CSS args
    parser.add_argument('--topq', type=int, default=1,
                        choices=[1, 2, 3], help="num of words to be masked in question")
    parser.add_argument('--keep_qtype', default=True,
                        action='store_false', help='keep qtype or not')
    parser.add_argument('--topv', type=int, default=1,
                        choices=[1, 3, 5, -1], help="num of object bbox to be masked in image")
    parser.add_argument('--top_hint', type=int, default=9, help="num of hint")
    parser.add_argument('--qvp', type=int, default=0,
                        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], help="ratio of q_bias and v_bias")
    parser.add_argument('--eval_each_epoch', default=True,
                        help="Evaluate every epoch, instead of at the end")
    # Arguments from the original model, we leave this default, except we
    # set --epochs to 30 since the model maxes out its performance on VQA 2.0 well before then
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt',
                        choices=['baseline0_newatt', 'baseline0'])
    parser.add_argument('--output', type=str, default='exp0')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')

    # Arguments added
    parser.add_argument('-dw', "--debias_weight", default=1, type=float,
                        help="Debias weight on loss fn chosen in --debias arg")
    parser.add_argument('--top_n_answ', default=5, type=int,
                        help='top N answers to consider for debiasing')
    parser.add_argument('--model_path', default=None,
                        help='load from this path the pretrained weights')
    parser.add_argument('--gpu_id', '-gpu', type=str, default='-1')
    parser.add_argument('--embed_dim', default=300)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--annotations_path', type=str, required=True)
    parser.add_argument('--embed_path', type=str, required=True)
    parser.add_argument('--img_path', type=str, required=True)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    return args


def get_bias(train_dset, eval_dset, args):
    # Compute the bias:
    # The bias here is just the expected score for each answer/question type
    answer_voc_size = train_dset.num_ans_candidates

    # question_type -> answer -> total score
    question_type_to_probs = defaultdict(Counter)

    # question_type -> num_occurances
    question_type_to_count = Counter()
    for ex in train_dset.entries:
        ans = ex["answer"]
        q_type = ans["question_type"]
        question_type_to_count[q_type] += 1
        if ans["labels"] is not None:
            for label, score in zip(ans["labels"], ans["scores"]):
                question_type_to_probs[q_type][label] += score
    question_type_to_prob_array = {}

    for q_type, count in question_type_to_count.items():
        prob_array = np.zeros(answer_voc_size, np.float32)
        for label, total_score in question_type_to_probs[q_type].items():
            prob_array[label] += total_score
        prob_array /= count
        question_type_to_prob_array[q_type] = prob_array

    for ds in [train_dset, eval_dset]:
        for ex in ds.entries:
            q_type = ex["answer"]["question_type"]
            bias = question_type_to_prob_array[q_type]
            ex["bias"] = bias


def main():
    print('Starting executing main function')
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Executing in {device}')

    args.output = os.path.join('logs', args.output)
    if not os.path.isdir(args.output):
        utils.create_dir(args.output)
    else:
        if click.confirm('Exp directory already exists in {}. Erase?'
                         .format(args.output, default=False)):
            os.system('rm -r ' + args.output)
            utils.create_dir(args.output)
        else:
            os._exit(1)

    dictionary = Dictionary.load_from_file(
        os.path.join(args.data_path, 'dictionary_v2.pkl'))

    print("Building train dataset...")
    optim_run = True

    train_dset = VQAFeatureDataset(
        name='train',
        dictionary=dictionary,
        img_path=args.img_path,
        annotations_path=args.annotations_path,
        data_path=args.data_path,
        keep_qtype=args.keep_qtype,
        optim_run=optim_run
    )

    print("Building test dataset...")
    eval_dset = VQAFeatureDataset(
        name='val',
        dictionary=dictionary,
        img_path=args.img_path,
        data_path=args.data_path,
        annotations_path=args.annotations_path,
        keep_qtype=args.keep_qtype,
        optim_run=optim_run,
    )

    get_bias(train_dset, eval_dset, args)

    # Build the model using the original constructor
    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(
        train_dset, args.num_hid).to(device)

    emb_np_path = os.path.join(
        args.embed_path, f'glove6b_init_{args.embed_dim}d_v2.npy')
    model.w_emb.init_embedding(emb_np_path)
    utils.select_loss_fn(model, args)

    if args.model_path:
        utils.load_pretrained_model(args.model_path, model, device)

    with open(os.path.join(args.data_path, 'cpv2', f'qid2type_cpv2.json'), 'r') as f:
        qid2type = json.load(f)
    model = model.to(device)
    batch_size = args.batch_size

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    train_loader = DataLoader(train_dset, batch_size,
                              shuffle=True, num_workers=32)
    eval_loader = DataLoader(eval_dset, batch_size,
                             shuffle=False, num_workers=32)

    print("Starting training...")
    cdvqa_train(model, device, train_loader, eval_loader, args, qid2type)


if __name__ == '__main__':
    main()
