import torch

from models import ConvKB, DisenE, DisenE_Trans, TransE
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from process_data import init_embeddings, build_data
from dataloader import Corpus

import random
import argparse
import os
import sys
import logging
import time


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="./data/", help="data directory")
parser.add_argument("--output_dir", default="./results/", help="Folder name to save the models.")
parser.add_argument("--model_name", default="DisenE", help="")
parser.add_argument("--dataset", default="test_data", help="dataset")
parser.add_argument("--evaluate", type=int, default=0, help="only evaluate")
parser.add_argument("--ckpt", default="None", help="")
parser.add_argument("--load", default="None", help="")

parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument("--pretrained_emb", type=int, default=0, help="Use pretrained embeddings")
parser.add_argument("--embedding_size", type=int, default=100, help="Size of embeddings (if pretrained not used)")
parser.add_argument("--valid_invalid_ratio", type=int, default=40, help="Ratio of valid to invalid triples for training")
parser.add_argument("--seed", type=int, default=42, help="seed")

parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-5, help="L2 reglarization")
parser.add_argument("--step_size", type=int, default=25, help="step_size")
parser.add_argument("--gamma", type=int, default=0.5, help="gamma")

parser.add_argument("--k_factors", type=int, default=4,help="Number of k")
parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")
parser.add_argument("--out_channels", type=int, default=50, help="Number of output channels in conv layer")
parser.add_argument("--do_normalize", type=int, default=0, help="normalize for init embedding")
parser.add_argument("--sample_num", type=int, default=20, help="sample_num")
parser.add_argument("--w1", type=float, default=0.0, help="loss_2 weight: top2 constrain")
parser.add_argument("--w2", type=float, default=0.0, help="loss_3 wight, attention loss")
parser.add_argument("--top_n", type=int, default=2, help="top_n")
parser.add_argument("--margin", type=float, default=5, help="Margin used in hinge loss")

args = parser.parse_args()


def save_model(model, name, folder_name):
    print("Saving Model")
    torch.save(model.state_dict(),
               (os.path.join(folder_name, "trained_" + name + ".pth")))
    print("Done saving Model")


def main():

    args.data_dir = os.path.join(args.data_dir, args.dataset)
    args.output_dir = os.path.join(args.output_dir, args.dataset)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    CUDA = torch.cuda.is_available()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    print("args = ", args)

    train_data, validation_data, test_data, entity2id, relation2id = build_data(args.data_dir)

    if args.pretrained_emb:
        entity_embeddings, relation_embeddings = init_embeddings(os.path.join(args.data_dir, 'entity2vec.txt'),
                                                                 os.path.join(args.data_dir, 'relation2vec.txt'),
                                                                 args.k_factors, args.embedding_size)
        print("Initialised relations and entities from TransE")

    else:
        entity_embeddings = np.random.randn(
            len(entity2id), args.embedding_size * args.k_factors)
        relation_embeddings = np.random.randn(
            len(relation2id), args.embedding_size)
        print("Initialised relations and entities randomly")

    entity_embeddings = torch.FloatTensor(entity_embeddings)
    relation_embeddings = torch.FloatTensor(relation_embeddings)
    print("Initial entity dimensions {} , relation dimensions {}".format(entity_embeddings.size(),
                                                                         relation_embeddings.size()))

    train_loader = Corpus(args, train_data, validation_data, test_data, entity2id, relation2id,
                    args.batch_size, args.valid_invalid_ratio)


    file_name = "model_name_" + str(args.model_name) + "_embedding_size_" + str(args.embedding_size) + "_k_factors_" + str(
        args.k_factors) + "_lr_" + str(args.lr) + "_epochs_" + str(args.epochs) + "_out_channels_" + str(
        args.out_channels) + "_batch_size_" + str(args.batch_size) + "_dropout_" + str(args.dropout) + "_pretrained_emb_" + str(
        args.pretrained_emb) + "_step_size_" + str(args.step_size) + "_gamma_" + str(args.gamma) + "_w1_" + str(
        args.w1) + "_w2_" + str(args.w2) + "_sample_num_" + str(args.sample_num) + "_top_n_" + str(args.top_n)

    model_path = os.path.join(args.output_dir, file_name)
    output_file = os.path.join(args.output_dir, "results_" + file_name + ".txt")

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if args.model_name == 'ConvKB':
        model = ConvKB(entity_embeddings, relation_embeddings, config=args)
    elif args.model_name == 'TransE':
        model = TransE(entity_embeddings, relation_embeddings, config=args)
    elif args.model_name == 'DisenE':
        model = DisenE(entity_embeddings, relation_embeddings, config=args)
    elif args.model_name == 'DisenE_Trans':
        model = DisenE_Trans(entity_embeddings, relation_embeddings, config=args)

    else:
        print("no such model name")

    if args.load != 'None':
        model.load_state_dict(torch.load(args.load))
        print("model loaded")

    if CUDA:
        print("using CUDA")
        model.cuda()

    best_epoch = 0
    if args.evaluate == 0:
        best_epoch = train(args, train_loader, model, CUDA, model_path)
    evaluate(args, model, model_path, train_loader, output_file, best_epoch=best_epoch, best_or_final='best')
    evaluate(args, model, model_path, train_loader, output_file, best_epoch=best_epoch, best_or_final='final')


def cal_atten_loss(batch_atten, batch_triples, batch_labels, iter_num, train_indices):
    att_loss = torch.zeros(1).cuda()
    sample_num = args.sample_num
    tmp_size = args.batch_size
    if (iter_num + 1) * args.batch_size > len(train_indices):
        last_iter_size = len(train_indices) - args.batch_size * iter_num
        tmp_size = last_iter_size

    cnt = 0
    for i in range(tmp_size):
        rel = batch_triples[i, 1]
        att = batch_atten[i, :]
        random_idx = (np.random.choice(batch_triples.shape[0], sample_num, replace=False)).tolist()
        #print("random_idx", random_idx)
        for idx in random_idx:
            if rel == batch_triples[idx, 1] and batch_labels[idx] == 1:
                tmp_att = batch_atten[idx, :]
                #print("att, tmp", att, tmp_att, att_loss)
                att_loss += torch.dist(att, tmp_att, p=2)
                #print("dist", att_loss)
                cnt += 1

    if cnt == 0:
        return att_loss
    return att_loss/cnt



def train(args, train_loader, model, CUDA, model_path):
    print("model training")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma, last_epoch=-1)

    epoch_losses = []   # losses of all epochs
    print("Number of epochs {}".format(args.epochs))

    min_loss = 10000.0
    best_epoch = 0
    start_time = time.time()
    for epoch in range(args.epochs):
        print("\nepoch-> ", epoch)
        random.shuffle(train_loader.train_triples)
        train_loader.train_indices = np.array(list(train_loader.train_triples)).astype(np.int32)

        model.train()  # getting in training mode
        epoch_loss = []

        if len(train_loader.train_indices) % args.batch_size == 0:
            num_iters_per_epoch = len(
                train_loader.train_indices) // args.batch_size
        else:
            num_iters_per_epoch = (
                len(train_loader.train_indices) // args.batch_size) + 1

        for iters in range(num_iters_per_epoch):
            start_time_iter = time.time()
            batch_triples, batch_labels = train_loader.get_iteration_batch(iters)

            if CUDA:
                batch_triples = Variable(torch.LongTensor(batch_triples)).cuda()
                batch_labels = Variable(torch.FloatTensor(batch_labels)).cuda()

            else:
                batch_triples = Variable(torch.LongTensor(batch_triples))
                batch_labels = Variable(torch.FloatTensor(batch_labels))

            pred_loss, batch_atten = model(batch_triples, batch_labels)

            optimizer.zero_grad()

            loss = pred_loss
            top_att_loss_data = 0.0
            att_loss_data = 0.0
            if args.w1 != 0:
                sorted_att, sorted_indices_in = torch.sort(batch_atten, dim=-1, descending=True)
                top_att = sorted_att[:, :args.top_n]
                top_num_att = torch.sum(top_att, 1)
                y2 = torch.ones(int(batch_triples.size(0))).cuda()
                top_att_loss = torch.mean(y2 - top_num_att)
                loss = loss + args.w1 * top_att_loss
                top_att_loss_data = top_att_loss.data.item()
            if args.w2 != 0:
                att_loss = cal_atten_loss(batch_atten, batch_triples, batch_labels.view(-1), iters, train_loader.train_indices)
                loss = loss + args.w2 * att_loss
                att_loss_data = att_loss.data.item()

            end_time_iter = time.time()

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data.item())

            if iters % 50 == 0:
                print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}, Pred_loss {3:.4f}, "
                      "Top_atten_loss {4:.4f}, Atten_diss_loss {5:.4f}".format(
                    iters, end_time_iter - start_time_iter, loss.data.item(), pred_loss.data.item(), top_att_loss_data,
                    att_loss_data))
            

        scheduler.step()
        cur_lr = optimizer.param_groups[0]['lr']
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        print("Epoch {} , average loss {} , tot_time {}, learning rate {}".format(
            epoch, avg_loss, (time.time() - start_time)/60/60, cur_lr))
        epoch_losses.append(avg_loss)

        if avg_loss < min_loss:
            min_loss = avg_loss
            best_epoch = epoch
            save_model(model, "best", model_path)

    save_model(model, "final", model_path)

    return best_epoch


def evaluate(args, model, model_path, train_loader, output_file, best_epoch=0, best_or_final='best'):
    print("model evaluating")
    if best_epoch != 0:
        print("best_epoch", best_epoch)
    if args.ckpt != 'None':
        model_path = args.ckpt
    ckpt_path = os.path.join(model_path, 'trained_' + best_or_final + '.pth')
    print("ckpt_path:", ckpt_path)
    model.load_state_dict(torch.load(ckpt_path))
    print("model loaded")

    model.cuda()
    model.eval()
    if args.model_name == 'DisenE_Trans' or args.model_name == 'TransE':
        model = model.test
    with torch.no_grad():
        MRR, MR, H1, H3, H10 = train_loader.get_validation_pred(args, model)

    with open(output_file, "w") as writer:
        logging.info("***** results *****")
        writer.write('Hits @1: %s\n' % (H1))
        writer.write('Hits @3: %s\n' % (H3))
        writer.write('Hits @10: %s\n' % (H10))
        writer.write('Mean rank: %s\n' % MR)
        writer.write('Mean reciprocal rank: %s\n' % MRR)
        writer.write('Best epoch: %s\n' % str(best_epoch))
        writer.write("%s = %s\n" % ('args', str(args)))


if __name__ == '__main__':
    main()