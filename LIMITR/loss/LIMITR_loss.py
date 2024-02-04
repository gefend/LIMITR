"""
Adapted from: https://github.com/mrlibw/ControlGAN
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange
# from LIMITR.models.LIMITR_model import TextSA

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def attention_fn(query, context, temp1,context_img,use_mask,mask):
        """
        query: batch x ndf x queryL
        context: batch x ndf x ih x iw (sourceL=ihxiw)
        mask: batch_size x sourceL
        """
        if context_img:
            sourceL = context.size(2)
            contextT = torch.transpose(context, 1, 2).contiguous()
            query = torch.transpose(query, 1, 2).contiguous()
            batch_size, queryL = query.size(0), query.size(2)
        else:
            batch_size = query.size(0)
            sourceL = context.size(1)
            queryL = query.size(2)
            contextT = context

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
        if use_mask:
            patch_num = queryL
            attn[mask.repeat(patch_num,1)]=float("-inf")
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

def global_loss(cnn_code, rnn_code, eps=1e-8, temp3=10.0):

    batch_size = cnn_code.shape[0]
    labels = Variable(torch.LongTensor(range(batch_size))).to(cnn_code.device)

    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)

    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * temp3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()

    scores1 = scores0.transpose(0, 1)
    loss0 = nn.CrossEntropyLoss()(scores0, labels)
    loss1 = nn.CrossEntropyLoss()(scores1, labels)
    return loss0, loss1, scores0/temp3

def local_ext_loss(
    similarities_cap,similarities_img, cfg, temp3=10.0, agg="sum"
):
    if cfg.model.limitr.cap_attn:
        batch_size = similarities_cap.shape[0]
        similarities_cap = similarities_cap * temp3
        similarities1_cap = similarities_cap.transpose(0, 1)

        labels = Variable(torch.LongTensor(range(batch_size))).to(similarities_cap.device)

        loss0_cap = nn.CrossEntropyLoss()(similarities_cap, labels)
        loss1_cap = nn.CrossEntropyLoss()(similarities1_cap, labels)
    else:
        loss0_cap = 0
        loss1_cap = 0

    if cfg.model.limitr.img_attn:
        batch_size = similarities_img.shape[0]
        similarities_img = similarities_img * temp3
        similarities1_img = similarities_img.transpose(0, 1)

        labels = Variable(torch.LongTensor(range(batch_size))).to(similarities_img.device)

        loss0_img = nn.CrossEntropyLoss()(similarities_img, labels)
        loss1_img = nn.CrossEntropyLoss()(similarities1_img, labels)
    else:
        loss0_img = 0
        loss1_img = 0
    return loss0_cap, loss1_cap,loss0_img, loss1_img


def local_int_loss(img_emb_l_l,img_emb_l_f,text_emb_l,ind_lateral,sents,cfg,SAF_module,local_temperature = 0.1):
    mask = torch.from_numpy(np.array(sents)[:, :] == "[PAD]").type_as(
                img_emb_l_f).bool()

    bz = img_emb_l_f.size(0)
    ih, iw = img_emb_l_f.size(2), img_emb_l_f.size(3)
    sourceL = ih * iw
    img_emb_l_f = img_emb_l_f.view(img_emb_l_f.size(0), -1, sourceL)
    if cfg.model.limitr.lateral:
        img_emb_l_l = img_emb_l_l.view(img_emb_l_l.size(0), -1, sourceL)
        img_emb_l = torch.zeros(img_emb_l_f.shape, dtype=img_emb_l_f.dtype).cuda()
        img_emb_l[~ind_lateral] = img_emb_l_l
        img_emb_l[ind_lateral] = 1e-8 * torch.ones((768, 361), dtype=img_emb_l_f.dtype).cuda()
        img_emb = torch.cat((img_emb_l_f, img_emb_l), dim=2)
    else:
        img_emb = img_emb_l_f

    if cfg.model.limitr.int_w_method == 'uniform':
        cap_mask =~ mask
        y = cap_mask.sum(axis=1)
        norm_cap = cap_mask.T/y
        cap_weights = norm_cap.T
        img_weights = torch.ones(bz, img_emb.size(2)).type_as(img_emb) / img_emb.size(2)

    weiContext, attn = attention_fn(
        text_emb_l, img_emb, 4.0, context_img=True, use_mask=False, mask=0)
    if cfg.model.limitr.int_w_method == 'SAF':
        cap_mask = ~ mask
        y = cap_mask.sum(axis=1)
        norm_cap = cap_mask.T / y
        cap_weights = norm_cap.T

    word_sim = torch.bmm(text_emb_l, weiContext) / local_temperature
    word_num = word_sim.size(1)
    word_sim_1 = rearrange(word_sim, "b n1 n2 -> (b n1) n2")
    targets = torch.arange(word_num).type_as(text_emb_l).long().repeat(bz)
    loss_word_1 = torch.sum(F.cross_entropy(
        word_sim_1, targets, reduction="none") * cap_weights.view(-1)) / bz
    word_sim_2 = rearrange(word_sim, "b n1 n2 -> (b n2) n1")
    loss_word_2 = torch.sum(F.cross_entropy(
        word_sim_2, targets, reduction="none") * cap_weights.view(-1)) / bz

    loss_word = (loss_word_1 + loss_word_2) / 2.

    weiContext_img, _ = attention_fn(
        img_emb,text_emb_l ,4.0, context_img=False,use_mask=True,mask=mask
    )

    if cfg.model.limitr.int_w_method == 'SAF':
        sim_loc_img = torch.mul(weiContext_img,  img_emb)
        sim_loc_img = l2norm(sim_loc_img, dim=-1)
        sim_loc_img = sim_loc_img.transpose(1, 2).contiguous()
        sim_ave_f = torch.mean(sim_loc_img, 1)
        _, img_weights = SAF_module(sim_loc_img, sim_ave_f)

    patch_sim = torch.bmm(img_emb.permute(0, 2, 1), weiContext_img)/ local_temperature #
    patch_num = patch_sim.size(1)
    patch_sim_1 = rearrange(patch_sim, "b n1 n2 -> (b n1) n2")
    targets = torch.arange(patch_num).type_as(
        img_emb).long().repeat(bz)
    # # loss_patch_1 = F.cross_entropy(patch_sim_1, targets)
    loss_patch_1 = torch.sum(F.cross_entropy(
        patch_sim_1, targets, reduction="none") * img_weights.view(-1)) / bz
    #
    patch_sim_2 = rearrange(patch_sim, "b n1 n2 -> (b n2) n1")
    loss_patch_2 = torch.sum(F.cross_entropy(
        patch_sim_2, targets, reduction="none") * img_weights.view(-1)) / bz
    #
    loss_patch = (loss_patch_1 + loss_patch_2) / 2.

    loss_local_int = (loss_patch + loss_word)/2
    return loss_local_int/10

class TextSA(nn.Module):
    """
    Build global text representations by self-attention.
    Args: - local: local word embeddings, shape: (batch_size, L, 1024)
          - raw_global: raw text by averaging words, shape: (batch_size, 1024)
    Returns: - new_global: final text by self-attention, shape: (batch_size, 1024).
    """

    def __init__(self, embed_dim, dropout_rate):
        super(TextSA, self).__init__()

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