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

"""Models for Text and Image Composition."""

import numpy as np
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
import text_model
import torch_functions
from torch.autograd import Variable
import timm
from sentence_transformers import SentenceTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConCatModule(torch.nn.Module):

    def __init__(self):
        super(ConCatModule, self).__init__()

    def forward(self, x):
        x = torch.cat(x, 1)

        return x


class ImgTextCompositionBase(torch.nn.Module):
    """Base class for image + text composition."""

    def __init__(self):
        super().__init__()
        self.normalization_layer = torch_functions.NormalizationLayer(
            normalize_scale=4.0, learn_scale=True)
        self.soft_triplet_loss = torch_functions.TripletLoss()

    #         self.name = 'model_name'

    def extract_img_feature(self, imgs):
        raise NotImplementedError

    def extract_text_feature(self, text_query, use_bert=True):
        raise NotImplementedError

    def compose_img_text(self, imgs, text_query):
        raise NotImplementedError

    def compute_loss(self,
                     imgs_query,
                     text_query,
                     imgs_target,
                     soft_triplet_loss=True):
        dct_with_representations = self.compose_img_text(imgs_query, text_query)
        composed_source_image = self.normalization_layer(dct_with_representations["repres"])
        target_img_features_non_norm = self.extract_img_feature(imgs_target)
        target_img_features = self.normalization_layer(target_img_features_non_norm)
        assert (composed_source_image.shape[0] == target_img_features.shape[0] and
                composed_source_image.shape[1] == target_img_features.shape[1])
        # Get Rot_Sym_Loss
        if self.name == 'composeAE':
            CONJUGATE = Variable(torch.cuda.FloatTensor(32, 1).fill_(-1.0), requires_grad=False)
            conjugate_representations = self.compose_img_text_features(target_img_features_non_norm,
                                                                       dct_with_representations["text_features"],
                                                                       CONJUGATE)
            composed_target_image = self.normalization_layer(conjugate_representations["repres"])
            source_img_features = self.normalization_layer(dct_with_representations["img_features"])  # img1
            if soft_triplet_loss:
                dct_with_representations["rot_sym_loss"] = \
                    self.compute_soft_triplet_loss_(composed_target_image, source_img_features)
            else:
                dct_with_representations["rot_sym_loss"] = \
                    self.compute_batch_based_classification_loss_(composed_target_image,
                                                                  source_img_features)
        else:  # tirg, RealSpaceConcatAE etc
            dct_with_representations["rot_sym_loss"] = 0

        if soft_triplet_loss:
            return self.compute_soft_triplet_loss_(composed_source_image,
                                                   target_img_features), dct_with_representations
        else:
            return self.compute_batch_based_classification_loss_(composed_source_image,
                                                                 target_img_features), dct_with_representations

    def compute_soft_triplet_loss_(self, mod_img1, img2):
        triplets = []
        labels = list(range(mod_img1.shape[0])) + list(range(img2.shape[0]))
        for i in range(len(labels)):
            triplets_i = []
            for j in range(len(labels)):
                if labels[i] == labels[j] and i != j:
                    for k in range(len(labels)):
                        if labels[i] != labels[k]:
                            triplets_i.append([i, j, k])
            np.random.shuffle(triplets_i)
            triplets += triplets_i[:3]
        assert (triplets and len(triplets) < 2000)
        return self.soft_triplet_loss(torch.cat([mod_img1, img2]), triplets)

    def compute_batch_based_classification_loss_(self, mod_img1, img2):
        x = torch.mm(mod_img1, img2.transpose(0, 1))
        labels = torch.tensor(range(x.shape[0])).long()
        labels = torch.autograd.Variable(labels).cuda()
        return F.cross_entropy(x, labels)


class GlobalAvgPool2d(torch.nn.Module):
    def forward(self, x):
        return F.adaptive_avg_pool2d(x, (1, 1))


class ImgEncoderTextEncoderBase(ImgTextCompositionBase):
    """Base class for image and text encoder."""

    def __init__(self, text_query, image_embed_dim, text_embed_dim, use_bert=True, name=""):
        super().__init__()
        # img model
        img_model = torchvision.models.resnet18(pretrained=True)
        self.img_model = torch.nn.Sequential(
            *[*img_model.children()][:-2])  # timm.create_model('efficientnet_b0', pretrained=True) #
        self.name = name

        self.avgpool = GlobalAvgPool2d()
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(512, image_embed_dim)
        )

        # text model

        self.text_model = SentenceTransformer('nli-roberta-base-v2', device=device)
        for p in self.text_model.parameters():
            p.requires_grad = False

        self.fc_text_output = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(768, text_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(text_embed_dim, text_embed_dim)
        )

    def extract_img_feature(self, imgs):
        return self.fc(self.avgpool(self.img_model(imgs)).squeeze())

    def extract_img_map_feature(self, imgs):
        return self.img_model(imgs)

    def extract_text_feature(self, text_query, use_bert=True):
        if use_bert:
            text_features = self.text_model.encode(text_query, convert_to_numpy=False, show_progress_bar=False)

            text_features = torch.stack(text_features).to(device)
            text_features = self.fc_text_output(text_features)
            return text_features
        return self.text_model(text_query)

    def freeze_pretrained(self):
        for p in self.text_model.parameters():
            p.requires_grad = False
        for p in self.img_model.parameters():
            p.requires_grad = False

    def unfreeze_pretrained(self):
        for p in self.text_model.parameters():
            p.requires_grad = True
        for p in self.img_model.parameters():
            p.requires_grad = True


class TIRG(ImgEncoderTextEncoderBase):
    """The TIRG model.

    The method is described in
    Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays.
    "Composing Text and Image for Image Retrieval - An Empirical Odyssey"
    CVPR 2019. arXiv:1812.07119
    """

    def __init__(self, text_query, image_embed_dim, text_embed_dim, use_bert, name):
        super().__init__(text_query, image_embed_dim, text_embed_dim, use_bert, name)

        self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0]))
        self.use_bert = use_bert

        merged_dim = image_embed_dim + text_embed_dim

        self.gated_feature_composer = torch.nn.Sequential(
            ConCatModule(),
            torch.nn.BatchNorm1d(merged_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(merged_dim, image_embed_dim)
        )

        self.res_info_composer = torch.nn.Sequential(
            ConCatModule(),
            torch.nn.BatchNorm1d(merged_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(merged_dim, merged_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(merged_dim, image_embed_dim)
        )

    def compose_img_text(self, imgs, text_query):
        img_features = self.extract_img_feature(imgs)
        text_features = self.extract_text_feature(text_query, self.use_bert)

        return self.compose_img_text_features(img_features, text_features)

    def compose_img_text_features(self, img_features, text_features):
        f1 = self.gated_feature_composer((img_features, text_features))
        f2 = self.res_info_composer((img_features, text_features))
        f = F.sigmoid(f1) * img_features * self.a[0] + f2 * self.a[1]

        dct_with_representations = {"repres": f}
        return dct_with_representations


class ComplexProjectionModule(torch.nn.Module):

    def __init__(self, image_embed_dim=512, text_embed_dim=768):
        super().__init__()
        self.bert_features = torch.nn.Sequential(
            torch.nn.BatchNorm1d(text_embed_dim),
            torch.nn.Linear(text_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim)
        )
        self.image_features = torch.nn.Sequential(
            torch.nn.BatchNorm1d(image_embed_dim),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
            torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
        )

    def forward(self, x):
        x1 = self.image_features(x[0])
        x2 = self.bert_features(x[1])
        # default value of CONJUGATE is 1. Only for rotationally symmetric loss value is -1.
        # which results in the CONJUGATE of text features in the complex space
        CONJUGATE = x[2]
        num_samples = x[0].shape[0]
        CONJUGATE = CONJUGATE[:num_samples]
        delta = x2  # text as rotation
        re_delta = torch.cos(delta)
        im_delta = CONJUGATE * torch.sin(delta)

        re_score = x1 * re_delta
        im_score = x1 * im_delta

        concat_x = torch.cat([re_score, im_score], 1)
        x0copy = x[0].unsqueeze(1)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        re_score = re_score.unsqueeze(1)
        im_score = im_score.unsqueeze(1)

        return concat_x, x1, x2, x0copy, re_score, im_score


class LinearMapping(torch.nn.Module):
    """
    This is linear mapping to image space. rho(.)
    """

    def __init__(self, image_embed_dim=512):
        super().__init__()
        self.mapping = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2 * image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * image_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim)
        )

    def forward(self, x):
        theta_linear = self.mapping(x[0])
        return theta_linear


class ConvMapping(torch.nn.Module):
    """
    This is convoultional mapping to image space. rho_conv(.)
    """

    def __init__(self, image_embed_dim=512):
        super().__init__()
        self.mapping = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2 * image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * image_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim)
        )
        # in_channels, output channels
        self.conv = torch.nn.Conv1d(5, 64, kernel_size=3, padding=1)
        self.adaptivepooling = torch.nn.AdaptiveMaxPool1d(16)

    def forward(self, x):
        concat_features = torch.cat(x[1:], 1)
        concat_x = self.conv(concat_features)
        concat_x = self.adaptivepooling(concat_x)
        final_vec = concat_x.reshape((concat_x.shape[0], 1024))
        theta_conv = self.mapping(final_vec)
        return theta_conv


class ComposeAE(ImgEncoderTextEncoderBase):
    """The ComposeAE model.

    The method is described in
    Muhammad Umer Anwaar, Egor Labintcev and Martin Kleinsteuber.
    ``Compositional Learning of Image-Text Query for Image Retrieval"
    arXiv:2006.11149
    """

    def __init__(self, text_query, image_embed_dim, text_embed_dim, use_bert, name):
        super().__init__(text_query, image_embed_dim, text_embed_dim, use_bert, name)
        self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0]))
        self.use_bert = use_bert

        # merged_dim = image_embed_dim + text_embed_dim

        self.encoderLinear = torch.nn.Sequential(
            ComplexProjectionModule(),
            LinearMapping()
        )
        self.encoderWithConv = torch.nn.Sequential(
            ComplexProjectionModule(),
            ConvMapping()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim)
        )
        self.txtdecoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, text_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(text_embed_dim, text_embed_dim)
        )

    def compose_img_text(self, imgs, text_query):
        img_features = self.extract_img_feature(imgs)
        text_features = self.extract_text_feature(text_query, self.use_bert)

        return self.compose_img_text_features(img_features, text_features)

    def compose_img_text_features(self, img_features, text_features, CONJUGATE=Variable(torch.cuda.FloatTensor(32, 1).fill_(1.0), requires_grad=False)):
        theta_linear = self.encoderLinear((img_features, text_features, CONJUGATE))
        theta_conv = self.encoderWithConv((img_features, text_features, CONJUGATE))
        theta = theta_linear * self.a[1] + theta_conv * self.a[0]

        dct_with_representations = {"repres": theta,
                                    "repr_to_compare_with_source": self.decoder(theta),
                                    "repr_to_compare_with_mods": self.txtdecoder(theta),
                                    "img_features": img_features,
                                    "text_features": text_features
                                    }

        return dct_with_representations


class RealConCatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        concat_x = torch.cat(x, -1)
        return concat_x


class RealLinearMapping(torch.nn.Module):
    """
    This is linear mapping from real space to image space.
    """

    def __init__(self, image_embed_dim=512, text_embed_dim=768):
        super().__init__()
        self.mapping = torch.nn.Sequential(
            torch.nn.BatchNorm1d(text_embed_dim + image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(text_embed_dim + image_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim)
        )

    def forward(self, x):
        theta_linear = self.mapping(x)
        return theta_linear


class RealConvMapping(torch.nn.Module):
    """
    This is convoultional mapping from Real space to image space.
    """

    def __init__(self, image_embed_dim=512, text_embed_dim=768):
        super().__init__()
        self.mapping = torch.nn.Sequential(
            torch.nn.BatchNorm1d(text_embed_dim + image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(text_embed_dim + image_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim)
        )
        # in_channels, output channels
        self.conv1 = torch.nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.adaptivepooling = torch.nn.AdaptiveMaxPool1d(20)

    def forward(self, x):
        concat_x = self.conv1(x.unsqueeze(1))
        concat_x = self.adaptivepooling(concat_x)
        final_vec = concat_x.reshape((concat_x.shape[0], 1280))
        theta_conv = self.mapping(final_vec)
        return theta_conv


class RealSpaceConcatAE(ImgEncoderTextEncoderBase):
    """The RealSpaceConcatAE model.

    The method  in ablation study Table 5 (Concat in real space)
    Muhammad Umer Anwaar, Egor Labintcev and Martin Kleinsteuber.
    ``Compositional Learning of Image-Text Query for Image Retrieval"
    arXiv:2006.11149
    """

    def __init__(self, text_query, image_embed_dim, text_embed_dim, use_bert, name):
        super().__init__(text_query, image_embed_dim, text_embed_dim, use_bert, name)
        self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0]))
        self.use_bert = use_bert

        # merged_dim = image_embed_dim + text_embed_dim

        self.encoderLinear = torch.nn.Sequential(
            RealConCatModule(),
            RealLinearMapping()
        )
        self.encoderWithConv = torch.nn.Sequential(
            RealConCatModule(),
            RealConvMapping()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim)
        )
        self.txtdecoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, text_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(text_embed_dim, text_embed_dim)
        )

    def compose_img_text(self, imgs, text_query):
        img_features = self.extract_img_feature(imgs)
        text_features = self.extract_text_feature(text_query, self.use_bert)

        return self.compose_img_text_features(img_features, text_features)

    def compose_img_text_features(self, img_features, text_features):
        theta_linear = self.encoderLinear((img_features, text_features))
        theta_conv = self.encoderWithConv((img_features, text_features))
        theta = theta_linear * self.a[1] + theta_conv * self.a[0]

        dct_with_representations = {"repres": theta,
                                    "repr_to_compare_with_source": self.decoder(theta),
                                    "repr_to_compare_with_mods": self.txtdecoder(theta),
                                    "img_features": img_features,
                                    "text_features": text_features
                                    }

        return dct_with_representations


class AttentionNet(nn.Module):
    def __init__(self, input_features=512, output_features=512):
        super(AttentionNet, self).__init__()
        self.input_features = input_features
        self.output_features = output_features  # k

        self.image1 = nn.Linear(input_features, output_features, bias=False)
        self.question1 = nn.Linear(768, output_features)
        self.attention1 = nn.Linear(output_features, output_features)
        nn.init.xavier_uniform_(self.image1.weight)
        nn.init.xavier_uniform_(self.question1.weight)
        nn.init.xavier_uniform_(self.attention1.weight)

        self.image2 = nn.Linear(input_features, output_features, bias=False)
        self.question2 = nn.Linear(output_features, output_features)
        self.attention2 = nn.Linear(output_features, 512)
        nn.init.xavier_uniform_(self.image2.weight)
        nn.init.xavier_uniform_(self.question2.weight)
        nn.init.xavier_uniform_(self.attention2.weight)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, image, question):
        # image_vec = batchx1x512
        # question_vec = batchx768

        irep_1 = self.image1(image)
        qrep_1 = self.question1(question)
        ha_1 = self.tanh(irep_1 + qrep_1)
        ha_1 = self.dropout(ha_1)
        pi_1 = self.softmax(self.attention1(ha_1))

        u_1 = (pi_1 * image) + qrep_1

        irep_2 = self.image2(image)
        qrep_2 = self.question2(u_1)
        ha_2 = self.tanh(irep_2 + qrep_2)
        ha_2 = self.dropout(ha_2)
        pi_2 = self.softmax(self.attention2(ha_2))
        u_2 = (1 - pi_2) * image  # + u_1
        return u_2


class RealConCatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        concat_x = torch.cat(x, -1)
        return concat_x


class OurRealLinearMapping(torch.nn.Module):
    """
    This is linear mapping from real space to image space.
    """

    def __init__(self, image_embed_dim=512, text_embed_dim=768):
        super().__init__()
        self.mapping = torch.nn.Sequential(
            torch.nn.BatchNorm1d(text_embed_dim + image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(text_embed_dim + image_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(image_embed_dim, image_embed_dim)

        )

    def forward(self, x):
        theta_linear = self.mapping(x)
        return theta_linear


def conv(ic, oc, k, s, p):
    return nn.Sequential(
        nn.Conv2d(ic, oc, k, s, p),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(oc),
    )


class FiLMBlock(nn.Module):
    def __init__(self):
        super(FiLMBlock, self).__init__()

    def forward(self, x, gamma, beta):
        beta = beta.view(x.size(0), x.size(1), 1, 1)
        gamma = gamma.view(x.size(0), x.size(1), 1, 1)

        x = gamma * x + beta

        return x


class ResBlock(nn.Module):
    def __init__(self, in_place, out_place):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_place, out_place, 1, 1, 0)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_place, out_place, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(out_place)
        self.film = FiLMBlock()
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, beta, gamma):
        x = self.conv1(x)
        x = self.relu1(x)
        identity = x

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.film(x, beta, gamma)
        x = self.relu2(x)

        x = x + identity

        return x


class FiLM(nn.Module):
    def __init__(self, n_res_blocks, n_channels):
        super(FiLM, self).__init__()

        dim_question = 768
        # Linear에서 나온 결과의 절반은 beta, 절반은 gamma
        # beta, gamma 모두 ResBlock 하나당 n_channels개씩 feed
        self.film_generator = nn.Linear(dim_question, 2 * n_res_blocks * n_channels)
        self.res_blocks = nn.ModuleList()

        for _ in range(n_res_blocks):
            self.res_blocks.append(ResBlock(n_channels + 2, n_channels))

        self.n_res_blocks = n_res_blocks
        self.n_channels = n_channels

    def forward(self, x, question):
        batch_size = x.size(0)

        film_vector = self.film_generator(question).view(
            batch_size, self.n_res_blocks, 2, self.n_channels)

        d = x.size(2)
        coordinate = torch.arange(-1, 1 + 0.00001, 2 / (d - 1)).cuda()
        coordinate_x = coordinate.expand(batch_size, 1, d, d)
        coordinate_y = coordinate.view(d, 1).expand(batch_size, 1, d, d)

        for i, res_block in enumerate(self.res_blocks):
            beta = film_vector[:, i, 0, :]
            gamma = film_vector[:, i, 1, :]

            x = torch.cat([x, coordinate_x, coordinate_y], 1)
            x = res_block(x, beta, gamma)

        # feature = x

        return x  # , feature


class GlobalAvgPool2d(torch.nn.Module):
    def forward(self, x):
        return F.adaptive_avg_pool2d(x, (1, 1))


class RealSpaceConcatAE(ImgEncoderTextEncoderBase):

    def __init__(self, text_query, image_embed_dim, text_embed_dim, use_bert, name):
        super().__init__(text_query, image_embed_dim, text_embed_dim, use_bert, name)
        self.a = torch.nn.Parameter(torch.tensor([0.7, 0.3]))

        self.use_bert = use_bert

        self.film_img_txt = FiLM(n_res_blocks=6, n_channels=512)

        self.maxpool_img = torch.nn.AdaptiveMaxPool2d((1, 1))

        self.encoderLinear = torch.nn.Sequential(
            RealConCatModule(),
            OurRealLinearMapping()
        )

        self.attention_net = AttentionNet()

        self.decoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim)
        )
        self.txtdecoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, text_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(text_embed_dim, text_embed_dim)
        )

    def compose_img_text(self, imgs, text_query):
        text_features = self.extract_text_feature(text_query, self.use_bert)
        img_map_features = self.extract_img_map_feature(imgs)
        img_features = self.extract_img_feature(imgs)
        return self.compose_img_text_features(img_map_features, img_features, text_features)

    def compose_img_text_features(self, img_map_features, img_features, text_features):
        projected_img_features = self.film_img_txt(img_map_features, text_features)
        projected_img_features = self.maxpool_img(projected_img_features).squeeze()
        theta_linear = self.attention_net(img_features.unsqueeze(1), text_features.unsqueeze(1)).squeeze()

        # Nx512 Nx768
        theta_conv = self.encoderLinear((projected_img_features, text_features))
        theta = theta_linear * self.a[0] + theta_conv * self.a[1]

        dct_with_representations = {
            "repres": theta,
            "repr_to_compare_with_source": self.decoder(theta_linear),
            "repr_to_compare_with_mods": self.txtdecoder(theta_conv),
            "img_features": img_features,
            "text_features": text_features
        }

        return dct_with_representations