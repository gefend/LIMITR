import torch
import torch.nn as nn
import cv2
import re
import numpy as np
from sklearn import metrics
import os
import torchvision.transforms as transforms
from PIL import Image
from .. import builder
from .. import loss
from transformers import AutoTokenizer,BertTokenizer
from nltk.tokenize import RegexpTokenizer
import torch.nn.functional as F

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class LIMITR(nn.Module):
    def __init__(self, cfg):
        super(LIMITR, self).__init__()

        self.cfg = cfg
        self.text_encoder = builder.build_text_model(cfg)

        self.img_encoder_f = builder.build_img_model(cfg)
        self.img_encoder_l = builder.build_img_model(cfg)

        self.local_ext_loss = loss.LIMITR_loss.local_ext_loss
        self.global_loss = loss.LIMITR_loss.global_loss

        if cfg.model.limitr.local_int_loss_weight > 0:
            self.local_int_loss = loss.LIMITR_loss.local_int_loss
        self.local_ext_loss_weight = self.cfg.model.limitr.local_ext_loss_weight
        self.global_loss_weight = self.cfg.model.limitr.global_loss_weight

        self.temp1 = self.cfg.model.limitr.temp1
        self.temp2 = self.cfg.model.limitr.temp2
        self.temp3 = self.cfg.model.limitr.temp3
        self.batch_size = self.cfg.train.batch_size

        self.tokenizer = BertTokenizer.from_pretrained(self.cfg.model.text.bert_type)
        self.ixtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

        self.SAF_module = SA(self.cfg.model.text.embedding_dim, 0.4)
        self.sim_eval_w_cap = nn.Linear(self.cfg.model.text.embedding_dim, 1)
        self.sim_eval_w_img = nn.Linear(self.cfg.model.text.embedding_dim, 1)

        self.sigmoid = nn.Sigmoid()

    def text_encoder_forward(self, caption_ids, attention_mask, token_type_ids):
        text_emb_l, text_emb_g, sents = self.text_encoder(
            caption_ids, attention_mask, token_type_ids
        )
        return text_emb_l, text_emb_g, sents

    def image_encoder_forward(self, imgs_f, imgs_l=[]):
        img_feat_g_f, img_emb_l_f = self.img_encoder_f(imgs_f, get_local=True)
        img_emb_g_f, img_emb_l_f = self.img_encoder_f.generate_embeddings(
            img_feat_g_f, img_emb_l_f
        )
        if self.cfg.model.limitr.lateral:
            img_lateral_global = torch.zeros(img_emb_g_f.shape,dtype=img_emb_g_f.dtype).cuda()
            tmp_remove = imgs_l[:, 0, 0, 0]
            ind_lateral = (tmp_remove == 0)
            # feed forward to the feature extractor only cases with lateral images
            imgs_l_continue = imgs_l[~ind_lateral]

            img_feat_g_l, img_emb_l_l = self.img_encoder_l(imgs_l_continue, get_local=True)
            img_emb_g_l, img_emb_l_l = self.img_encoder_l.generate_embeddings(
                img_feat_g_l, img_emb_l_l
            )

            img_lateral_global[~ind_lateral] = img_emb_g_l
            img_emb_g = torch.stack([img_emb_g_f, img_lateral_global])
            img_emb_g = img_emb_g.mean(axis=0)
        else:
            ind_lateral = []
            img_emb_l_l = []
            img_emb_g = img_emb_g_f

        return img_emb_l_l, img_emb_l_f, img_emb_g, ind_lateral

    def attention_fn(self,query, context, temp1,context_img):
        """
        query: batch x ndf x queryL
        context: batch x ndf x ih x iw (sourceL=ihxiw)
        mask: batch_size x sourceL
        """
        if context_img:
            batch_size, queryL = query.size(0), query.size(2)
            sourceL = context.size(2)
            contextT = torch.transpose(context, 1, 2).contiguous()
        else:
            batch_size = query.size(0)
            sourceL = context.size(1)
            queryL = query.size(2)
            contextT = context

        # Get attention
        # (batch x sourceL x ndf)(batch x ndf x queryL)
        # -->batch x sourceL x queryL
        attn = torch.bmm(contextT, query)
        # --> batch*sourceL x queryL
        attn = attn.view(batch_size * sourceL, queryL)
        attn = nn.Softmax(dim=-1)(attn)

        # --> batch x sourceL x queryL
        attn = attn.view(batch_size, sourceL, queryL)
        # --> batch*queryL x sourceL
        attn = torch.transpose(attn, 1, 2).contiguous()
        attn = attn.view(batch_size * queryL, sourceL)

        attn = attn * temp1
        attn = nn.Softmax(dim=-1)(attn)
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attnT = torch.transpose(attn, 1, 2).contiguous()

        # (batch x ndf x sourceL)(batch x sourceL x queryL)
        # --> batch x ndf x queryL
        if context_img:
            weightedContext = torch.bmm(context, attnT)
        else:
            contextT = torch.transpose(context, 1, 2).contiguous()
            weightedContext = torch.bmm(contextT, attnT)

        return weightedContext, attn

    def local_similarities(
            self, img_features_l, img_features_f, ind_lateral, words_emb, cap_lens, sents, temp1=4.0
    ):

        batch_size = img_features_f.shape[0]

        img_features_f = img_features_f.type(words_emb.dtype)
        if self.cfg.model.limitr.lateral:
            img_features_l = img_features_l.type(words_emb.dtype)
            img_feats_lateral_all = torch.zeros(img_features_f.shape, dtype=words_emb.dtype).cuda()
            img_feats_lateral_all[~ind_lateral] = img_features_l
            img_feats_lateral_all[ind_lateral] = 1e-8 * torch.ones((768, 19,19),dtype=words_emb.dtype).cuda()

            ih, iw = img_feats_lateral_all.size(2), img_feats_lateral_all.size(3)
            sourceL = ih * iw
            img_feats_lateral_all = img_feats_lateral_all.view(img_feats_lateral_all.size(0), -1, sourceL)
            img_features_f = img_features_f.view(img_features_f.size(0), -1, sourceL)
            img_features_l = img_features_l.view(img_features_l.size(0), -1, sourceL)

            if self.cfg.model.limitr.pe:
                pe = self.posemb_sincos_2d(num_local_regions=361).cuda()
                img_features_f = img_features_f + pe.T
                img_feats_lateral_all[~ind_lateral] = img_features_l + pe.T
            img_features = torch.cat((img_features_f, img_feats_lateral_all), dim=2)
        else:
            ih, iw = img_features_f.size(2), img_features_f.size(3)
            sourceL = ih * iw
            img_features_f = img_features_f.view(img_features_f.size(0), -1, sourceL)
            img_features = img_features_f
            if self.cfg.model.limitr.pe:
                pe = self.posemb_sincos_2d().cuda()
                img_features = img_features + pe.T

        similarities_cap = []
        similarities_img = []
        att_maps = []
        for i in range(words_emb.shape[0]):
            # Get the i-th text description
            words_num = cap_lens[i]
            word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
            word = word.repeat(batch_size, 1, 1)
            context = img_features
            if self.cfg.model.limitr.cap_attn:
                weiContext, attn = self.attention_fn(
                    word, context, temp1, context_img=True
                )
                att_maps.append(
                    attn[i].unsqueeze(0).contiguous()
                )
                word = word.transpose(1, 2).contiguous()
                weiContext = weiContext.transpose(1, 2).contiguous()

                sim_loc_cap = torch.mul(weiContext, word)
                sim_loc_cap = l2norm(sim_loc_cap, dim=-1)
                sim_ave_cap = torch.mean(sim_loc_cap, 1)
                sim_vec_cap, sim_w_cap = self.SAF_module(sim_loc_cap, sim_ave_cap)
                sim_i_cap = self.sigmoid(self.sim_eval_w_cap(sim_vec_cap))
                similarities_cap.append(sim_i_cap)

            #seperate for frontal and lateral images
            if self.cfg.model.limitr.img_attn:
                weiContext_img, _ = self.attention_fn(
                    context, word, temp1, context_img=False
                )
                weiContext_img_f = weiContext_img[:, :, :361].transpose(1, 2).contiguous()
                if weiContext_img_f.shape != img_features_f.shape:
                    img_features_f = img_features_f.transpose(1, 2).contiguous()

                sim_loc_img_f = torch.mul(weiContext_img_f, img_features_f)
                sim_loc_img_f = l2norm(sim_loc_img_f, dim=-1)

                if self.cfg.model.limitr.lateral:
                    weiContext_img_l = weiContext_img[~ind_lateral, :, 361:].transpose(1, 2).contiguous()
                    if weiContext_img_l.shape != img_features_l.shape:
                        img_features_l = img_features_l.transpose(1, 2).contiguous()

                    sim_loc_img_l = torch.mul(weiContext_img_l, img_features_l)
                    sim_loc_img_l = l2norm(sim_loc_img_l, dim=-1)

                    sim_loc_only_f = sim_loc_img_f[ind_lateral]
                    sim_ave_only_f = torch.mean(sim_loc_only_f, 1)

                    sim_loc_both = torch.cat([sim_loc_img_f[~ind_lateral], sim_loc_img_l], dim=1)
                    sim_ave_both = torch.mean(sim_loc_img_l, 1)

                    sim_vec_only_f, _ = self.SAF_module(sim_loc_only_f, sim_ave_only_f)
                    sim_vec_both, _ = self.SAF_module(sim_loc_both, sim_ave_both)

                    sim_vec_img = torch.zeros_like(sim_vec_cap)
                    sim_vec_img[ind_lateral] = sim_vec_only_f
                    sim_vec_img[~ind_lateral] = sim_vec_both

                    sim_i_img = self.sigmoid(self.sim_eval_w_img(sim_vec_img))
                    similarities_img.append(sim_i_img)
                else:
                    sim_ave_f = torch.mean(sim_loc_img_f, 1)
                    sim_vec_f, sim_w_img = self.SAF_module(sim_loc_img_f, sim_ave_f)
                    sim_i_img = self.sigmoid(self.sim_eval_w_img(sim_vec_f))
                    similarities_img.append(sim_i_img)
        if self.cfg.model.limitr.cap_attn:
            similarities_cap = torch.cat(similarities_cap, 1)
        else:
            similarities_cap = torch.zeros(1)
        if self.cfg.model.limitr.img_attn:
            similarities_img = torch.cat(similarities_img, 1)
        else:
            similarities_img = torch.zeros(1)

        return similarities_cap, similarities_img,att_maps

    def _calc_local_loss(self, img_emb_l_l,img_emb_l_f,ind_lateral, text_emb_l, sents):

        cap_lens = [
            len([w for w in sent if not w.startswith("[")]) + 1 for sent in sents
        ]

        sim_loc_cap, sim_loc_img,attn_maps = self.local_similarities(img_emb_l_l,
            img_emb_l_f,
            ind_lateral,
            text_emb_l,
            cap_lens,
            sents,
            temp1=self.temp1)
        l_loss0_cap, l_loss1_cap,l_loss0_img, l_loss1_img = self.local_ext_loss(
            sim_loc_cap,
            sim_loc_img,
            cfg=self.cfg,
            temp3=self.temp3,
        )
        sim_loc = 0.5 * (sim_loc_cap.cuda() + sim_loc_img.cuda())
        l_loss0 = 0.5 * (l_loss0_cap+l_loss0_img)
        l_loss1 = 0.5 * (l_loss1_cap + l_loss1_img)

        if self.cfg.model.limitr.local_int_loss_weight>0:
            local_int_loss = self.local_int_loss(img_emb_l_l, img_emb_l_f, text_emb_l.permute((0, 2, 1)), ind_lateral,sents,
                                                 self.cfg, self.SAF_module, local_temperature=0.1)
        else:
            local_int_loss = 0

        return l_loss0, l_loss1, local_int_loss, sim_loc, attn_maps

    def _calc_global_loss(self, img_emb_g, text_emb_g):
        g_loss0, g_loss1,sim_glo = self.global_loss(img_emb_g, text_emb_g, temp3=self.temp3)
        return g_loss0, g_loss1,sim_glo

    def calc_loss(self, img_emb_l_l, img_emb_l_f,ind_lateral, img_emb_g, text_emb_l, text_emb_g, sents):
        loss = 0

        if ((self.cfg.model.limitr.local_ext_loss_weight>0) or (self.cfg.model.limitr.local_int_loss_weight>0)):
            l_loss0, l_loss1,local_int_loss,sim_loc,attn_maps = self._calc_local_loss(
                img_emb_l_l,img_emb_l_f,ind_lateral, text_emb_l, sents
            )
            local_ext_loss = (l_loss0 + l_loss1)
            loss += local_ext_loss * self.cfg.model.limitr.local_ext_loss_weight
            loss += local_int_loss* self.cfg.model.limitr.local_int_loss_weight


        if self.cfg.model.limitr.global_loss_weight>0:
            g_loss0, g_loss1,sim_glo = self._calc_global_loss(img_emb_g, text_emb_g)
            global_loss = (g_loss0 + g_loss1)
            loss += global_loss * self.cfg.model.limitr.global_loss_weight
        else:
            sim_glo = torch.zeros_like(sim_loc)
            global_loss = 0

        similarities = torch.stack(
            [self.cfg.model.limitr.local_ext_loss_weight*sim_loc, self.cfg.model.limitr.global_loss_weight*sim_glo]
        )

        similarities = similarities.mean(axis=0)
        # weighted loss
        return loss, local_ext_loss, local_int_loss, global_loss, similarities, sim_loc, sim_glo, attn_maps

    def forward(self, x):
        # img encoder branch
        img_emb_l_l, img_emb_l_f, img_emb_g, ind_lateral = self.image_encoder_forward(x["imgs_F"], x["imgs_L"])
        # text encorder branch
        text_emb_l, text_emb_g, sents = self.text_encoder_forward(
            x["caption_ids"], x["attention_mask"], x["token_type_ids"]
        )

        return img_emb_l_l, img_emb_l_f, ind_lateral, img_emb_g, text_emb_l, text_emb_g, sents

    def posemb_sincos_2d(self,num_local_regions=361, temperature=10000, dtype=torch.float32,img_dim=256):
        h = np.sqrt(num_local_regions)
        dim = 768
        y, x = torch.meshgrid(torch.arange(h), torch.arange(h), indexing='ij')
        assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
        omega = torch.arange(dim // 4) / (dim // 4 - 1)
        omega = 1. / (temperature ** omega)

        y = y.flatten()[:, None] * omega[None, :]
        x = x.flatten()[:, None] * omega[None, :]
        pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
        return pe.type(dtype)

class SA(nn.Module):
    """
    Build global text representations by self-attention.
    Args: - local: local word embeddings, shape: (batch_size, L, 1024)
          - raw_global: raw text by averaging words, shape: (batch_size, 1024)
    Returns: - new_global: final text by self-attention, shape: (batch_size, 1024).
    """

    def __init__(self, embed_dim, dropout_rate):
        super(SA, self).__init__()

        self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                             nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                              nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))
        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of local words and raw global text
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)

        # compute the normalized weights, shape: (batch_size, L)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.embedding_common(common).squeeze(2)
        weights = self.softmax(weights)

        # compute final text, shape: (batch_size, 1024)
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)
        new_global = l2norm(new_global, dim=-1)

        return new_global, weights


