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

            # ih, iw = context.size(2), context.size(3)
            sourceL = context.size(2)
            contextT = torch.transpose(context, 1, 2).contiguous()
            query = torch.transpose(query, 1, 2).contiguous()
            batch_size, queryL = query.size(0), query.size(2)
            # context = context.view(batch_size, -1, sourceL)
        else:
            batch_size = query.size(0)
            # ih, iw = query.size(2), query.size(3)
            #sourceL = query.size(2)
            # query = query.view(batch_size,-1,sourceL)
            sourceL = context.size(1)
            queryL = query.size(2)
            contextT = context

        # --> batch x sourceL x ndf
        #context = context.view(batch_size, -1, sourceL)



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
            #mask = torch.transpose(mask, 0, 1).contiguous()
            # attn[mask.unsqueeze(2).repeat(
            #     1,1,patch_num)] = float("-inf")
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

# def attention_fn(query, context, temp1):
#     """
#     query: batch x ndf x queryL
#     context: batch x ndf x ih x iw (sourceL=ihxiw)
#     mask: batch_size x sourceL
#     """
#     batch_size, queryL = query.size(0), query.size(2)
#     ih, iw = context.size(2), context.size(3)
#     sourceL = ih * iw
#
#     # --> batch x sourceL x ndf
#     context = context.view(batch_size, -1, sourceL)
#     contextT = torch.transpose(context, 1, 2).contiguous()
#
#     # Get attention
#     # (batch x sourceL x ndf)(batch x ndf x queryL)
#     # -->batch x sourceL x queryL
#     attn = torch.bmm(contextT, query)
#     # --> batch*sourceL x queryL
#     attn = attn.view(batch_size * sourceL, queryL)
#     attn = nn.Softmax(dim=-1)(attn)
#
#     # --> batch x sourceL x queryL
#     attn = attn.view(batch_size, sourceL, queryL)
#     # --> batch*queryL x sourceL
#     attn = torch.transpose(attn, 1, 2).contiguous()
#     attn = attn.view(batch_size * queryL, sourceL)
#
#     attn = attn * temp1
#     attn = nn.Softmax(dim=-1)(attn)
#     attn = attn.view(batch_size, queryL, sourceL)
#     # --> batch x sourceL x queryL
#     attnT = torch.transpose(attn, 1, 2).contiguous()
#
#     # (batch x ndf x sourceL)(batch x sourceL x queryL)
#     # --> batch x ndf x queryL
#     weightedContext = torch.bmm(context, attnT)
#
#     return weightedContext, attn.view(batch_size, -1, ih, iw)


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
    return loss0, loss1,scores0/temp3

#TODO-img_attn
def local_loss(
    similarities_cap,similarities_img, cfg, temp3=10.0, agg="sum"
):
# def local_loss(
#         similarities_cap, temp3=10.0, agg="sum"):
    #
    # batch_size = img_features.shape[0]
    #
    # att_maps = []
    # similarities = []
    # # cap_lens = cap_lens.data.tolist()
    # for i in range(words_emb.shape[0]):
    #
    #     # Get the i-th text description
    #     words_num = cap_lens[i]  # 25
    #     # TODO: remove [SEP]
    #     # word = words_emb[i, :, 1:words_num+1].unsqueeze(0).contiguous()    # [1, 768, 25]
    #     word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()  # [1, 768, 25]
    #     word = word.repeat(batch_size, 1, 1)  # [48, 768, 25]
    #     context = img_features  # [48, 768, 19, 19]
    #
    #     weiContext, attn = attention_fn(
    #         word, context, temp1
    #     )  # [48, 768, 25], [48, 25, 19, 19]
    #
    #     att_maps.append(
    #         attn[i].unsqueeze(0).contiguous()
    #     )  # add attention for curr index  [25, 19, 19]
    #     word = word.transpose(1, 2).contiguous()  # [48, 25, 768]
    #     weiContext = weiContext.transpose(1, 2).contiguous()  # [48, 25, 768]
    #
    #     sim_loc = torch.mul(weiContext, word)
    #     sim_loc = l2norm(sim_loc, dim=-1)
    #     sim_ave = torch.mean(sim_loc, 1)
    #     sim_vec_cap, sim_weights_cap = self.SAF_module(sim_loc, sim_ave)
    #     sim_i = self.sigmoid(self.sim_eval_w_cap(sim_vec_cap))
    #     similarities.append(sim_i)
    #     #TODO- gloria
    #
    #     # word = word.view(batch_size * words_num, -1)  # [1200, 768]
    #     # weiContext = weiContext.view(batch_size * words_num, -1)  # [1200, 768]
    #     # row_sim = cosine_similarity(word, weiContext)
    #     # row_sim = row_sim.view(batch_size, words_num)  # [48, 25]
    #     #
    #     # row_sim.mul_(temp2).exp_()
    #     # if agg == "sum":
    #     #     row_sim = row_sim.sum(dim=1, keepdim=True)  # [48, 1]
    #     # else:
    #     #     row_sim = row_sim.mean(dim=1, keepdim=True)  # [48, 1]
    #     # row_sim = torch.log(row_sim)
    #     #
    #     # similarities.append(row_sim)
    #     #TODO-gloria
    #
    # similarities = torch.cat(similarities, 1)  #
    if cfg.model.limitr.cap_attn:
        batch_size = similarities_cap.shape[0]
        similarities_cap = similarities_cap * temp3
        similarities1_cap = similarities_cap.transpose(0, 1)  # [48, 48]

        labels = Variable(torch.LongTensor(range(batch_size))).to(similarities_cap.device)

        loss0_cap = nn.CrossEntropyLoss()(similarities_cap, labels)  # labels: arange(batch_size)
        loss1_cap = nn.CrossEntropyLoss()(similarities1_cap, labels)
    else:
        loss0_cap = 0
        loss1_cap = 0
    #TODO- local img attn
    if cfg.model.limitr.img_attn:
        batch_size = similarities_img.shape[0]
        similarities_img = similarities_img * temp3
        similarities1_img = similarities_img.transpose(0, 1)  # [48, 48]

        labels = Variable(torch.LongTensor(range(batch_size))).to(similarities_img.device)

        loss0_img = nn.CrossEntropyLoss()(similarities_img, labels)  # labels: arange(batch_size)
        loss1_img = nn.CrossEntropyLoss()(similarities1_img, labels)
    else:
        loss0_img = 0
        loss1_img = 0
    return loss0_cap, loss1_cap,loss0_img, loss1_img
    #return loss0_cap, loss1_cap

# def local_int_loss(img_emb_l_l,img_emb_l_f,text_emb_l,sents,sim_w_cap,sim_w_img,cfg,local_temperature = 0.1):
#     #TODO add lateral
#     mask = torch.from_numpy(np.array(sents)[:, :] == "[PAD]").type_as(
#             img_emb_l_f).bool()
#     ih, iw = img_emb_l_f.size(2), img_emb_l_f.size(3)
#     sourceL = ih * iw
#     img_emb_l_f = img_emb_l_f.view(img_emb_l_f.size(0), -1, sourceL)
#     bz = img_emb_l_f.size(0)
#
#     if cfg.model.limitr.int_w_method == 'uniform':
#         cap_weights = torch.ones(bz, text_emb_l.size(1)).type_as(text_emb_l)/text_emb_l.size(1)
#         img_weights = torch.ones(bz, img_emb_l_f.size(1)).type_as(img_emb_l_f) / img_emb_l_f.size(1)
#         if cfg.model.limitr.lateral:
#             img_weights_l = torch.ones(bz, img_emb_l_l.size(1)).type_as(img_emb_l_l) / img_emb_l_l.size(1)
#     elif cfg.model.limitr.int_w_method == 'SAF':
#         cap_weights = sim_w_cap
#         img_weights = sim_w_img
#     img_emb_l_f=img_emb_l_f.type_as(text_emb_l)
#     atten_sim = torch.bmm(text_emb_l, img_emb_l_f)#.permute(0, 2, 1)
#     #word_num = text_emb_l.size(1)
#     # atten_sim[mask.unsqueeze(1).repeat(1, word_num, 1)] = float("-inf")
#     atten_scores = F.softmax(atten_sim / local_temperature, dim=-1)  # bz, 196, 111
#     word_atten_output = torch.bmm(atten_scores, img_emb_l_f.permute(0, 2, 1))
#
#     word_atten_output = F.normalize(word_atten_output, dim=-1)
#
#
#
#
#     word_sim = torch.bmm(text_emb_l, word_atten_output.permute(
#         0, 2, 1)) / local_temperature
#     word_num = word_sim.size(1)
#     word_sim_1 = rearrange(word_sim, "b n1 n2 -> (b n1) n2")
#     targets = torch.arange(word_num).type_as(text_emb_l).long().repeat(bz)
#     loss_word_1 = torch.sum(F.cross_entropy(
#         word_sim_1, targets, reduction="none") * cap_weights.view(-1)) / bz
#
#     word_sim_2 = rearrange(word_sim, "b n1 n2 -> (b n2) n1")
#     loss_word_2 = torch.sum(F.cross_entropy(
#         word_sim_2, targets, reduction="none") * cap_weights.view(-1)) / bz
#
#     loss_word = (loss_word_1 + loss_word_2) / 2.
#
#     atten_sim = torch.bmm(img_emb_l_f, text_emb_l.permute(0, 2, 1))
#     patch_num = img_emb_l_f.size(1)
#     atten_sim[mask.unsqueeze(1).repeat(
#         1, patch_num, 1)] = float("-inf")
#     atten_scores = F.softmax(
#         atten_sim / local_temperature, dim=-1)  # bz, 196, 111
#     patch_atten_output = torch.bmm(atten_scores, text_emb_l)
#     #
#     # # patch_atten_output: bz, 196, 128
#     patch_atten_output = F.normalize(patch_atten_output, dim=-1)
#     patch_num = patch_atten_output.size(1)
#     # # patch_atten_weights = torch.ones(
#     # #     bz, patch_num).type_as(batch["imgs"]) / patch_num
#     #
#     patch_sim = torch.bmm(img_emb_l_f, patch_atten_output.permute(
#         0, 2, 1)) / local_temperature
#     patch_num = patch_sim.size(1)
#     patch_sim_1 = rearrange(patch_sim, "b n1 n2 -> (b n1) n2")
#     targets = torch.arange(patch_num).type_as(
#         img_emb_l_f).long().repeat(bz)
#     # # loss_patch_1 = F.cross_entropy(patch_sim_1, targets)
#     loss_patch_1 = torch.sum(F.cross_entropy(
#         patch_sim_1, targets, reduction="none") * img_weights.view(-1)) / bz
#     #
#     patch_sim_2 = rearrange(patch_sim, "b n1 n2 -> (b n2) n1")
#     loss_patch_2 = torch.sum(F.cross_entropy(
#         patch_sim_2, targets, reduction="none") * img_weights.view(-1)) / bz
#     #
#     loss_patch = (loss_patch_1 + loss_patch_2) / 2.
#
#     loss_local_int = (loss_patch + loss_word)/2
#     #loss_local_int = loss_word
#     # word_sim = torch.bmm(
#     #     word_emb_q, word_atten_output.permute(0, 2, 1)) / self.hparams.local_temperature
#
#     # with torch.no_grad():
#     #     atten_weights = word_attn_q.detach()
#     #     word_atten_weights = []
#     #     for i in range(bz):
#     #         atten_weight = atten_weights[i]
#     #         nonzero = atten_weight.nonzero().squeeze()
#     #         low = torch.quantile(atten_weight[nonzero], 0.1)
#     #         high = torch.quantile(atten_weight[nonzero], 0.9)
#     #         atten_weight[nonzero] = atten_weight[nonzero].clip(low, high)
#     #         word_atten_weights.append(atten_weight.clone())
#     #     word_atten_weights = torch.stack(word_atten_weights)
#         # TODO: maybe clip the tensor of 10 percentile and 90 percentile
#
#     #word_atten_weights /= word_atten_weights.sum(dim=1, keepdims=True)
#     return loss_local_int

def local_int_loss(img_emb_l_l,img_emb_l_f,text_emb_l,sents,cfg,SAF_module,local_temperature = 0.1):
    mask = torch.from_numpy(np.array(sents)[:, :] == "[PAD]").type_as(
                img_emb_l_f).bool()
    # mask_new = ~mask
    # mask_n = mask_new.unsqueeze(axis=2)
    # mask_n = mask_n.repeat(1, 1, 768)

    # if cfg.model.limitr.int_w_method == 'SAF':
    #     SAF_module = TextSA(cfg.model.text.embedding_dim, 0.4)

    bz = img_emb_l_f.size(0)
    ih, iw = img_emb_l_f.size(2), img_emb_l_f.size(3)
    sourceL = ih * iw
    img_emb_l_f = img_emb_l_f.view(img_emb_l_f.size(0), -1, sourceL)

    if cfg.model.limitr.int_w_method == 'uniform':
        #cap_weights = torch.ones(bz, text_emb_l.size(1)).type_as(text_emb_l)/text_emb_l.size(1)
        cap_mask =~ mask
        y = cap_mask.sum(axis=1)
        norm_cap = cap_mask.T/y
        cap_weights = norm_cap.T
        img_weights = torch.ones(bz, img_emb_l_f.size(2)).type_as(img_emb_l_f) / img_emb_l_f.size(2)
        if cfg.model.limitr.lateral:
            img_weights_l = torch.ones(bz, img_emb_l_l.size(1)).type_as(img_emb_l_l) / img_emb_l_l.size(1)

    #cap_weights = torch.zeros(bz, text_emb_l.size(1)).type_as(text_emb_l)
    #weiContext = torch.zeros_like(text_emb_l)
    weiContext, attn = attention_fn(
        text_emb_l, img_emb_l_f, 4.0, context_img=True, use_mask=False, mask=0)
    if cfg.model.limitr.int_w_method == 'SAF':
        cap_mask = ~ mask
        y = cap_mask.sum(axis=1)
        norm_cap = cap_mask.T / y
        cap_weights = norm_cap.T
    # for i in range(bz):
    #     n_word = torch.nonzero(text_emb_l).shape[0]
    #     weiContext_i = F.normalize(weiContext[i,:n_word,:], dim=-1)
    #     if cfg.model.limitr.int_w_method == 'SAF':

            # sim_loc_cap = torch.mul(weiContext_i.permute(1, 0), text_emb_l[i, :n_word, :])
            # sim_loc_cap = l2norm(sim_loc_cap, dim=-1)
            # sim_ave_cap = torch.mean(sim_loc_cap,axis=0)
            # #n_word = torch.nonzero(sim_loc_cap).shape[0]
            # _, cap_weights_tmp = SAF_module(sim_loc_cap.unsqueeze(axis=0), sim_ave_cap.unsqueeze(axis=0))
            # cap_weights[i, :n_word] = cap_weights_tmp

    word_sim = torch.bmm(text_emb_l, weiContext) / local_temperature#.permute(0, 2, 1)
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
        img_emb_l_f,text_emb_l ,4.0, context_img=False,use_mask=True,mask=mask
    )

    # if weiContext_img.shape !=  img_emb_l_f.shape:
    #     img_emb_l_f =  img_emb_l_f.transpose(1, 2).contiguous()
    if cfg.model.limitr.int_w_method == 'SAF':
        sim_loc_img = torch.mul(weiContext_img,  img_emb_l_f)
        sim_loc_img = l2norm(sim_loc_img, dim=-1)
        sim_loc_img = sim_loc_img.transpose(1, 2).contiguous()
        sim_ave_f = torch.mean(sim_loc_img, 1)
        _, img_weights = SAF_module(sim_loc_img, sim_ave_f)

    patch_sim = torch.bmm(img_emb_l_f.permute(0, 2, 1), weiContext_img)/ local_temperature #
    patch_num = patch_sim.size(1)
    patch_sim_1 = rearrange(patch_sim, "b n1 n2 -> (b n1) n2")
    targets = torch.arange(patch_num).type_as(
        img_emb_l_f).long().repeat(bz)
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