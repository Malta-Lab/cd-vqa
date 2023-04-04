import os
import json
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from model import base_model
from dataset import Dictionary, VQAFeatureDataset
import utils


def parse_args():
    parser = argparse.ArgumentParser(
        "Eval the BottomUpTopDown model with a de-biasing method")

    parser.add_argument(
        '-p', "--entropy_penalty", default=0.36, type=float,
        help="Entropy regularizer weight for the learned_mixin model")
    # Arguments from the original model, we leave this default, except we
    # set --epochs to 15 since the model maxes out its performance on VQA 2.0 well before then
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt',
                        choices=['qa', 'baseline0_newatt', 'baseline0'])
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--model_state', type=str,
                        default='logs/exp0/model.pth')

    # Arguments added
    parser.add_argument('-dw', "--debias_weight", default=1, type=float,
                        help="Debias weight on loss fn chosen in --debias arg")
    parser.add_argument('--gpu_id', '-gpu', type=str,
                        default='-1', help='-1 for CPU')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--annotations_path', type=str, required=True)
    parser.add_argument('--embed_path', type=str, required=True)
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--embed_dim', default=300)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    return args


def compute_score_with_logits(device, logits, labels):
    logits = torch.argmax(logits, 1)
    one_hots = torch.zeros(*labels.size()).to(device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def evaluate(model, device, dataloader, qid2type):
    score = 0
    upper_bound = 0
    score_yesno = 0
    score_number = 0
    score_other = 0
    total_yesno = 0
    total_number = 0
    total_other = 0
    model.eval()

    for v, q, a, b, qids, _ in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        with torch.no_grad():
            v = v.to(device)
            q = q.to(device)
            pred, _, _ = model(v, q, None, None, None)
            batch_score = compute_score_with_logits(
                device, pred, a.to(device)).cpu().numpy().sum(1)
            score += batch_score.sum()
            upper_bound += (a.max(1)[0]).sum()
            qids = qids.detach().cpu().int().numpy()
            for j in range(len(qids)):
                qid = qids[j]
                typ = qid2type[str(qid)]
                if typ == 'yes/no':
                    score_yesno += batch_score[j]
                    total_yesno += 1
                elif typ == 'other':
                    score_other += batch_score[j]
                    total_other += 1
                elif typ == 'number':
                    score_number += batch_score[j]
                    total_number += 1

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    score_yesno /= total_yesno
    score_other /= total_other
    score_number /= total_number

    results = dict(
        score=score,
        upper_bound=upper_bound,
        score_yesno=score_yesno,
        score_other=score_other,
        score_number=score_number,
    )
    return results


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(args.data_path, 'cpv2', f'qid2type_cpv2.json'), 'r') as f:
        qid2type = json.load(f)

    dictionary = Dictionary.load_from_file(
        os.path.join(args.data_path, 'dictionary_v2.pkl'))

    print("Building test dataset...")
    eval_dset = VQAFeatureDataset(
        'val',
        dictionary=dictionary,
        img_path=args.img_path,
        annotations_path=args.annotations_path,
        data_path=args.data_path)

    # Build the model using the original constructor
    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(
        eval_dset, args.num_hid).to(device)
    utils.select_loss_fn(model, args)

    print(f'Loading model state from {args.model_state}')
    model_state = torch.load(args.model_state)
    model.load_state_dict(model_state)

    model = model.to(device)
    batch_size = args.batch_size

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # The original version uses multiple workers, but that just seems slower on my setup
    eval_loader = DataLoader(eval_dset, batch_size,
                             shuffle=False, num_workers=16)

    print("Starting eval...")
    results = evaluate(model, device, eval_loader, qid2type)
    print('\teval overall score: %.2f' % (100 * results['score']))
    print('\teval up_bound score: %.2f' % (100 * results["upper_bound"]))
    print('\teval y/n score: %.2f' % (100 * results["score_yesno"]))
    print('\teval other score: %.2f' % (100 * results["score_other"]))
    print('\teval number score: %.2f' % (100 * results["score_number"]))


if __name__ == '__main__':
    main()
