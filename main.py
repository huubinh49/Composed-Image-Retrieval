# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Main method to train the model."""

# !/usr/bin/python

import argparse
import sys
import gc
import time
import datasets
import img_text_composition_models
import numpy as np
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import test_retrieval
import torch
import torch.utils.data
import torchvision
from tqdm import tqdm as tqdm
from copy import deepcopy
import socket
import os
from datetime import datetime

torch.set_num_threads(3)


def parse_opt():
    """Parses the input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default='')
    parser.add_argument('--comment', type=str)
    parser.add_argument('--dataset', type=str, default="fashionIQ")
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--text_query_path', type=str)
    parser.add_argument('--model', type=str, default='Ours')
    parser.add_argument('--image_embed_dim', type=int, default=512)
    parser.add_argument('--use_bert', type=bool, default=False)
    parser.add_argument('--use_complete_text_query', type=bool, default=False)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--learning_rate_decay_frequency', type=int, default=3500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--category_to_train', type=str, default='all')
    parser.add_argument('--num_iters', type=int, default=120000)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--optimizer', type=int, default='SGD')
    parser.add_argument('--loss', type=str, default='batch_based_classification')
    parser.add_argument('--loader_num_workers', type=int, default=4)
    parser.add_argument('--log_dir', type=str, default='../logs/')
    parser.add_argument('--test_only', type=bool, default=False)
    parser.add_argument('--model_checkpoint', type=str, default='')

    args = parser.parse_args()
    return args


def load_dataset(opt):
    """Loads the input datasets."""
    print('Reading dataset ', opt.dataset)
    if opt.dataset == 'fashion200k':
        trainset = datasets.Fashion200k(
            path=opt.dataset_path,
            split='train',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ]))
        testset = datasets.Fashion200k(
            path=opt.dataset_path,
            split='test',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ]))
    elif opt.dataset == 'mitstates':
        trainset = datasets.MITStates(
            path=opt.dataset_path,
            split='train',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ]))
        testset = datasets.MITStates(
            path=opt.dataset_path,
            split='test',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ]))
    elif opt.dataset == 'fashionIQ':
        trainset = datasets.FashionIQ(
            path=opt.dataset_path,
            cat_type=opt.category_to_train,
            split='train',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ]))
        testset = datasets.FashionIQ(
            path=opt.dataset_path,
            cat_type=opt.category_to_train,
            split='val',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ]))
    else:
        print('Invalid dataset', opt.dataset)
        sys.exit()

    print('trainset size:', len(trainset))
    print('testset size:', len(testset))
    return trainset, testset


def create_model(opt, texts):
    """Builds the model and related optimizer."""
    print("Creating model and optimizer for", opt.model)
    text_embed_dim = 512 if not opt.use_bert else 768

    if opt.model == 'tirg':
        model = img_text_composition_models.TIRG(texts,
                                                 image_embed_dim=opt.image_embed_dim,
                                                 text_embed_dim=text_embed_dim,
                                                 use_bert=opt.use_bert,
                                                 name=opt.model)
    elif opt.model == 'composeAE':
        model = img_text_composition_models.ComposeAE(texts,
                                                      image_embed_dim=opt.image_embed_dim,
                                                      text_embed_dim=text_embed_dim,
                                                      use_bert=opt.use_bert,
                                                      name=opt.model)
    elif opt.model == 'RealSpaceConcatAE':
        model = img_text_composition_models.RealSpaceConcatAE(texts,
                                                              image_embed_dim=opt.image_embed_dim,
                                                              text_embed_dim=text_embed_dim,
                                                              use_bert=opt.use_bert,
                                                              name=opt.model)
    elif opt.model == 'Ours':
        img_text_composition_models.OurRealLinearMapping(texts,
                                                         image_embed_dim=opt.image_embed_dim,
                                                         text_embed_dim=text_embed_dim,
                                                         use_bert=opt.use_bert,
                                                         name=opt.model)
    model = model.cuda()

    return model


def create_optimizer(model, opt):
    parameters = []
    for p in model.parameters():
        p.requires_grad = True
        parameters.append(p)

    for child in model.children():
        for c in child.children():
            for p in c.parameters():
                p.requires_grad = True
                parameters.append(p)
        for parameter in child.parameters():
            parameter.requires_grad = True
            parameters.append(parameter)

    parameters = list(set(parameters))

    for i, p in enumerate(parameters):
        if torch.equal(p, model.a):
            parameters.pop(i)

    params = [
        {
            'params': model.a,
            'lr': opt.learning_rate * 0.01
        },
        {
            'params': parameters,
            'lr': opt.learning_rate
        }
    ]

    for _, p1 in enumerate(params):  # remove duplicated params
        for _, p2 in enumerate(params):
            if p1 is not p2:
                for p11 in p1['params']:
                    for j, p22 in enumerate(p2['params']):
                        if p11 is p22:
                            p2['params'][j] = torch.tensor(0.0, requires_grad=True)

    for p in model.text_model.parameters():
        p.requires_grad = False

    optimizer = None
    if opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params,
                                    lr=opt.learning_rate,
                                    momentum=0.9,
                                    weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.AdamW(params, lr=opt.learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                      weight_decay=opt.weight_decay, amsgrad=False)

    return optimizer


def train_loop(opt, loss_weights, logger, trainset, testset, model, optimizer):
    """Function for train loop"""
    print('Begin training')
    print(len(trainset.test_queries), len(testset.test_queries))
    torch.backends.cudnn.benchmark = True
    losses_tracking = {}
    it = 0
    epoch = 0
    tic = time.time()
    l2_loss = torch.nn.MSELoss().cuda()

    while it < opt.num_iters:
        epoch += 1

        # show/log stats
        print('It', it, 'epoch', epoch, 'Elapsed time', round(time.time() - tic,
                                                              4), opt.comment)
        tic = time.time()
        for loss_name in losses_tracking:
            avg_loss = np.mean(losses_tracking[loss_name][-len(trainloader):])
            print('    Loss', loss_name, round(avg_loss, 4))
            logger.add_scalar(loss_name, avg_loss, it)
        logger.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], it)

        if epoch % 1 == 0:
            gc.collect()

        # test
        if epoch % opt.test_period == 0 and epoch >= opt.warmup_epochs:
            tests = []
            for name, dataset in [('test', testset)]:  # [('train', trainset), ('test', testset)]:
                if opt.dataset == 'fashionIQ':
                    t = test_retrieval.fiq_test(opt, model, dataset)
                else:
                    t = test_retrieval.test(opt, model, dataset)
                tests += [(name + ' ' + metric_name, metric_value)
                          for metric_name, metric_value in t]
            for metric_name, metric_value in tests:
                logger.add_scalar(metric_name, metric_value, it)
                print('    ', metric_name, round(metric_value, 4))

                # save checkpoint
            torch.save({
                'it': it,
                'opt': opt,
                'model_state_dict': model.state_dict(),
                'optimizer': optimizer
            },
                logger.file_writer.get_logdir() + f'/latest_checkpoint{epoch}.pth')

        if epoch < opt.warmup_epochs:
            for p in model.img_model.parameters():
                p.requires_grad = False
        else:
            for p in model.img_model.parameters():
                p.requires_grad = True

                # run training for 1 epoch
        model.train()
        trainloader = trainset.get_loader(
            batch_size=opt.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=opt.loader_num_workers)

        def training_1_iter(data):
            assert type(data) is list
            img1 = np.stack([d['source_img_data'] for d in data])
            img1 = torch.from_numpy(img1).float()
            img1 = torch.autograd.Variable(img1).cuda()

            img2 = np.stack([d['target_img_data'] for d in data])
            img2 = torch.from_numpy(img2).float()
            img2 = torch.autograd.Variable(img2).cuda()

            if opt.use_complete_text_query:
                if opt.dataset == 'mitstates':
                    supp_text = [str(d['noun']) for d in data]
                    mods = [str(d['mod']['str']) for d in data]
                    # text_query here means complete_text_query
                    text_query = [adj + " " + noun for adj, noun in zip(mods, supp_text)]
                else:
                    text_query = [str(d['target_caption']) for d in data]
            else:
                text_query = [str(d['mod']['str']) for d in data]
            # compute loss
            if opt.loss not in ['soft_triplet', 'batch_based_classification']:
                print('Invalid loss function', opt.loss)
                sys.exit()

            losses = []
            if_soft_triplet = True if opt.loss == 'soft_triplet' else False
            loss_value, dct_with_representations = model.compute_loss(img1,
                                                                      text_query,
                                                                      img2,
                                                                      soft_triplet_loss=if_soft_triplet)

            loss_name = opt.loss
            losses += [(loss_name, loss_weights[0], loss_value.cuda())]

            if opt.model == 'composeAE':
                dec_img_loss = l2_loss(dct_with_representations["repr_to_compare_with_source"],
                                       dct_with_representations["img_features"])
                dec_text_loss = l2_loss(dct_with_representations["repr_to_compare_with_mods"],
                                        dct_with_representations["text_features"])

                losses += [("L2_loss", loss_weights[1], dec_img_loss.cuda())]
                losses += [("L2_loss_text", loss_weights[2], dec_text_loss.cuda())]
                losses += [("rot_sym_loss", loss_weights[3], dct_with_representations["rot_sym_loss"].cuda())]
            elif opt.model == 'RealSpaceConcatAE' or opt.model == 'Ours':
                dec_img_loss = l2_loss(dct_with_representations["repr_to_compare_with_source"],
                                       dct_with_representations["img_features"])
                dec_text_loss = l2_loss(dct_with_representations["repr_to_compare_with_mods"],
                                        dct_with_representations["text_features"])

                losses += [("L2_loss", loss_weights[1], dec_img_loss.cuda())]
                losses += [("L2_loss_text", loss_weights[2], dec_text_loss.cuda())]

            total_loss = sum([
                loss_weight * loss_value
                for loss_name, loss_weight, loss_value in losses
            ])
            assert not torch.isnan(total_loss)
            losses += [('total training loss', None, total_loss.item())]

            # track losses
            for loss_name, loss_weight, loss_value in losses:
                if loss_name not in losses_tracking:
                    losses_tracking[loss_name] = []
                losses_tracking[loss_name].append(float(loss_value))

            torch.autograd.set_detect_anomaly(True)

            # gradient descendt
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        for data in tqdm(trainloader, desc='Training for epoch ' + str(epoch)):
            it += 1
            training_1_iter(data)

            # decay learning rate
            if it >= opt.learning_rate_decay_frequency and it % opt.learning_rate_decay_frequency == 0:
                for g in optimizer.param_groups:
                    g['lr'] *= 0.1

    print('Finished training')


def main():
    opt = parse_opt()
    print('Arguments:')
    for k in opt.__dict__.keys():
        print('    ', k, ':', str(opt.__dict__[k]))

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    loss_weights = [1.0, 0.1, 0.1, 0.01]
    logdir = os.path.join(opt.log_dir, current_time + '_' + socket.gethostname() + opt.comment)

    logger = SummaryWriter(logdir)
    print('Log files saved to', logger.file_writer.get_logdir())
    for k in opt.__dict__.keys():
        logger.add_text(k, str(opt.__dict__[k]))

    trainset, testset = load_dataset(opt)
    model = create_model(opt, [t for t in trainset.get_all_texts()])
    optimizer = create_optimizer(model, opt)

    if len(opt.model_checkpoint):
        checkpoint = torch.load(opt.model_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = checkpoint['optimizer']
        it = checkpoint['it']

    if opt.test_only:
        print('Doing test only')
        checkpoint = torch.load(opt.model_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        it = checkpoint['it']
        model.eval()
        tests = []
        it = 0
        for name, dataset in [('train', trainset), ('test', testset)]:
            if opt.dataset == 'fashionIQ':
                t = test_retrieval.fiq_test(opt, model, dataset)
            else:
                t = test_retrieval.test(opt, model, dataset)
            tests += [(name + ' ' + metric_name, metric_value) for metric_name, metric_value in t]
        for metric_name, metric_value in tests:
            logger.add_scalar(metric_name, metric_value, it)
            print('    ', metric_name, round(metric_value, 4))

        return 0
    train_loop(opt, loss_weights, logger, trainset, testset, model, optimizer)
    logger.close()


if __name__ == '__main__':
    main()
