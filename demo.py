# Copyright 2018 Google Inc. All Rights Reserved.
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

"""Evaluates the retrieval model."""
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
import argparse
import torchvision
import PIL
import gc
from sklearn.neighbors import NearestNeighbors
from main import create_model
from datasets import load_dataset

def parse_opt():
    """Parses the input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default='')
    parser.add_argument('--comment', type=str)
    parser.add_argument('--dataset', type=str, default="fashionIQ")
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--text_query_path', type=str)
    parser.add_argument('--query_img_path', type=str)
    parser.add_argument('--query_text', type=str)
    parser.add_argument('--model', type=str, default='Ours')
    parser.add_argument('--image_embed_dim', type=int, default=512)
    parser.add_argument('--use_bert', type=bool, default=False)
    parser.add_argument('--use_complete_text_query', type=bool, default=False)
    parser.add_argument('--loss', type=str, default='batch_based_classification')
    parser.add_argument('--loader_num_workers', type=int, default=4)
    parser.add_argument('--log_dir', type=str, default='../logs/')


    parser.add_argument('--model_checkpoint', type=str, default='')

    args = parser.parse_args()
    return args

def save_embedding(opt):
    trainset, testset = load_dataset(opt)
    model = create_model(opt, [t for t in trainset.get_all_texts()])
    checkpoint = torch.load(opt.model_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    all_img = os.listdir('../input/fashioniq/resized_images/resized_images/')
    all_img = [img for img in all_img if '._' not in img]
    len(all_img)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
    ])
    dict_img = all_img
    batch_size = 64
    all_imgs = []
    imgs = []
    for original_image_id in tqdm(all_img):
        img_path = '../input/fashioniq/resized_images/resized_images/' + original_image_id
        img = None
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')

            img = transform(img)
        imgs += [img]
        torch.cuda.empty_cache()
        if len(imgs) >= batch_size or original_image_id is all_img[-1]:
            if 'torch' not in str(type(imgs[0])):
                imgs = [torch.from_numpy(d).float() for d in imgs]
            imgs = torch.stack(imgs).float()
            imgs = torch.autograd.Variable(imgs).cuda()
            imgs = model.extract_img_feature(imgs.cuda()).data.cpu().numpy()

            all_imgs += [imgs]
            imgs = []
            gc.collect()
            torch.cuda.empty_cache()
    all_imgs = np.concatenate(all_imgs)
    # save embedding space
    # save to npy file
    np.save('embedding.npy', all_imgs)
    # save list of img name
    with open('all_imgs.txt', 'w') as f:
        f.write('\n'.join(all_img))

    tree = NearestNeighbors(
        n_neighbors=10, algorithm='ball_tree',
        metric='l2')
    tree.fit(all_imgs)
    try:
        import cPickle as pickle
    except ImportError:  # Python 3.x
        import pickle

    with open('knnpickle.p', 'wb') as fp:
        pickle.dump(tree, fp)


def open_img(img_path):

    with open(img_path, 'rb') as f:
        img = PIL.Image.open(f)
        img = img.convert('RGB')
    return img


def hide_border(subplot, all=False):
    if all:
        subplot.spines['top'].set_visible(False)
        subplot.spines['right'].set_visible(False)

        # X AXIS -BORDER
        subplot.spines['bottom'].set_visible(False)
        # Y AXIS -BORDER
        subplot.spines['left'].set_visible(False)

    # BLUE
    subplot.set_xticklabels([])
    # RED
    subplot.set_xticks([])
    # RED AND BLUE TOGETHER
    subplot.axes.get_xaxis().set_visible(False)

    # YELLOW
    subplot.set_yticklabels([])
    # GREEN
    subplot.set_yticks([])
    # YELLOW AND GREEN TOGHETHER
    subplot.axes.get_yaxis().set_visible(False)


def fiq_demo(opt, model, query_img, query_text):
    all_imgs = None
    with open('all_imgs.txt', 'w') as f:
        all_imgs = f.readlines().split('\n')

    try:
        import cPickle as pickle
    except ImportError:  # Python 3.x
        import pickle

    # load the model from disk
    neigh = pickle.load(open("knnpickle.p", 'rb'))

    img1 = np.stack([query_img])
    img1 = torch.from_numpy(img1).float()
    img1 = torch.autograd.Variable(img1).cuda()

    repres = model.compose_img_text(img1, query_text)["repres"]

    # dct_with_representations = {
    #     "repres": theta,
    #     "repr_to_compare_with_source": self.decoder(theta_linear),
    #     "repr_to_compare_with_mods": self.txtdecoder(theta_conv),
    #     "img_features": img_features,
    #     "text_features": text_features
    # }
    fig = plt.figure(figsize=(20, 5))

    subplot = fig.add_subplot(1, 8, 1)
    subplot.imshow(query_img)
    hide_border(subplot)

    subplot = fig.add_subplot(1, 6, 2)
    hide_border(subplot, all = True)

    sign = subplot.text(-0.3, 0.45, "+", fontsize = 32)

    txt = subplot.text(0.4, .5, query_text, ha='center', va='center', wrap=True,
          bbox=dict(boxstyle='square', fc='w', ec='r'), fontsize = 15)
    txt._get_wrap_line_width = lambda : 130

    sign2 = subplot.text(0.95, 0.45, "--->", fontsize = 32)

    nbrs = neigh.kneighbors([repres], 5, return_distance=False)
    for i in range(0, 5):
        img_name = all_imgs[nbrs[0][i]]
        img_path = f"{opt.dataset_path}/resized_images/resized_images/{img_name}"
        predict_img = open_img(img_path)
        subplot = fig.add_subplot(1, 8, 4+i)
        hide_border(subplot)
        subplot.imshow(predict_img)


if __name__ == '__main__':
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])

    ])

    opt = parse_opt()
    trainset, testset = load_dataset(opt)
    model = create_model(opt, [t for t in trainset.get_all_texts()])
    checkpoint = torch.load(opt.model_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])

    query_img = transform(open_img(opt.query_img_path))
    query_text = opt.query_text
    fiq_demo(opt, model, query_img, query_text)


