import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
from tqdm import tqdm
import random

from eval import evaluate, compute_score_with_logits
from tools.board import Board


def negative_answer_assignment(top_n_answ, label, bias, device):
    '''
    top_n_answ: positive to use top n answers or negative for uniform distribution
    label: ground truth answer
    bias: bias score of each answer
    device: cpu or cuda
    '''
    if top_n_answ == -1:
        label2 = (torch
                  .where(torch.le(label, 0),
                         torch.zeros(
                             bias.shape[0], bias.shape[1], dtype=torch.float)
                         .to(device),
                         1/torch.sum(torch.gt(bias, 0), dim=1, keepdim=True).float())
                  )
        return label2

    w_ind = bias.sort(1, descending=True)[1][:, :top_n_answ]
    false_ans = torch.ones(bias.shape[0], bias.shape[1]).to(device)
    false_ans.scatter_(1, w_ind, 0)
    label2 = label * false_ans
    return label2


def cdvqa_train(model, device, train_loader, eval_loader, args, qid2type):

    num_epochs = args.epochs
    run_eval = args.eval_each_epoch
    output = args.output
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    total_step = 0
    best_eval_score = 0

    # topv = args.topv
    top_hint = args.top_hint
    # topq = args.topq
    # keep_qtype=args.keep_qtype
    qvp = args.qvp

    board = Board(
        name=f'cdvqa_cpv2',
        path=os.path.join(args.output, 'tensorboard'))
    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0

        t = time.time()
        for i, (v, q, a, b, hintscore, type_mask, notype_mask, q_mask) in tqdm(enumerate(train_loader), ncols=100,
                                                                               desc="Epoch %d" % (epoch + 1), total=len(train_loader)):
            total_step += 1

            v = Variable(v, requires_grad=True).to(device)
            q = Variable(q).to(device)
            q_neg_masked = Variable(q_mask[0]).to(device)

            label = Variable(a).to(device)
            bias = Variable(b).to(device)
            hintscore = Variable(hintscore).to(device)

            random_num = random.randint(1, 10)
            if random_num <= qvp:

                naa_label = negative_answer_assignment(
                    args.top_n_answ, label, bias, device)

                questions = torch.cat((q, q_neg_masked), 0)
                labels = torch.cat((label, naa_label), 0)
                images = torch.cat([v, v], 0)
                biases = torch.cat([bias, bias], 0)

                mask_images = None

            else:
                naa_label = negative_answer_assignment(
                    args.top_n_answ, label, bias, device)

                v_all_ones_mask = torch.ones(hintscore.shape).to(device)
                hint_sort, hint_ind = hintscore.sort(1, descending=True)

                v_neg_ind = hint_ind[:, top_hint:]
                v_neg_mask = torch.zeros(hintscore.size(
                    0), hintscore.size(1)).to(device)
                v_neg_mask.scatter_(1, v_neg_ind, 1)

                mask_images = torch.cat([v_all_ones_mask, v_neg_mask], 0)
                labels = torch.cat((label, naa_label), 0)
                images = torch.cat([v, v], 0)
                questions = torch.cat([q, q], 0)
                biases = torch.cat([bias, bias], 0)

            pred, loss, word_emb = model(
                images, questions, labels, biases, mask_images)

            if (loss != loss).any():
                raise ValueError("NaN loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(
                device, pred, labels.data).sum()
            train_score += batch_score
            total_loss += loss.item() * q.size(0)

        total_loss /= len(train_loader.dataset) * 2
        train_score = 100 * train_score / len(train_loader.dataset)

        if run_eval:
            model.train(False)
            results = evaluate(model, device, eval_loader, qid2type)
            results["epoch"] = epoch + 1
            results["step"] = total_step
            results["train_loss"] = total_loss
            results["train_score"] = train_score

            model.train(True)

            eval_score = results["score"]
            bound = results["upper_bound"]
            yn = results['score_yesno']
            other = results['score_other']
            num = results['score_number']

        logger.write('epoch %d, time: %.2f' % (epoch, time.time() - t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' %
                     (total_loss, train_score))
        board.add_scalars(
            prior='Train',
            Loss=total_loss,
            Score=train_score)

        if run_eval:
            logger.write('\teval score: %.2f (%.2f)' %
                         (100 * eval_score, 100 * bound))
            logger.write('\tyn score: %.2f other score: %.2f num score: %.2f' % (
                100 * yn, 100 * other, 100 * num))
            board.add_scalars(
                prior='Valid',
                Score=100*eval_score,
                Bound=100*bound,
                Yn_Score=100*yn,
                Other_Score=100*other,
                Num_Score=100*num)

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score

        board.advance()
    board.close()
