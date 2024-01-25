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
from .. import utils
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
        # self.img_encoder = builder.build_img_model(cfg)

        self.local_loss = loss.LIMITR_loss.local_loss
        self.global_loss = loss.LIMITR_loss.global_loss

        if cfg.model.limitr.local_int_loss_weight > 0:
            self.local_int_loss = loss.LIMITR_loss.local_int_loss

        self.local_loss_weight = self.cfg.model.limitr.local_loss_weight
        self.global_loss_weight = self.cfg.model.limitr.global_loss_weight

        self.temp1 = self.cfg.model.limitr.temp1
        self.temp2 = self.cfg.model.limitr.temp2
        self.temp3 = self.cfg.model.limitr.temp3
        self.batch_size = self.cfg.train.batch_size

        #self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)
        #self.tokenizer = BertTokenizer.from_pretrained(self.cfg.model.text.bert_type)
        self.tokenizer = BertTokenizer.from_pretrained("/home/gefen/Documents/LIMITR-main/LIMITR/models/bioclinicalbert_tokenizer")
        self.ixtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

        # self.attn_model_cap = nn.MultiheadAttention(self.cfg.model.text.embedding_dim,num_heads=1)
        # self.attn_model_img = nn.MultiheadAttention(self.cfg.model.text.embedding_dim,num_heads=1)
        self.SAF_module = TextSA(self.cfg.model.text.embedding_dim, 0.4)
        self.sim_eval_w_cap = nn.Linear(self.cfg.model.text.embedding_dim, 1)
        self.sim_eval_w_img = nn.Linear(self.cfg.model.text.embedding_dim, 1)

        self.sigmoid = nn.Sigmoid()

    def text_encoder_forward(self, caption_ids, attention_mask, token_type_ids):
        text_emb_l, text_emb_g, sents = self.text_encoder(
            caption_ids, attention_mask, token_type_ids
        )
        return text_emb_l, text_emb_g, sents

    def image_encoder_forward(self, imgs_f,imgs_l=[]):
        img_feat_g_f, img_emb_l_f = self.img_encoder_f(imgs_f, get_local=True)
        # if self.cfg.model.gloria.pe:
        #     pe = self.posemb_sincos_2d(num_local_regions=361).cuda()
        #     pe = pe.permute(1, 0)
        #     pe = pe.view(pe.size(0), 19, 19)
        #     img_emb_l_f = img_emb_l_f + pe
        img_emb_g_f, img_emb_l_f = self.img_encoder_f.generate_embeddings(
            img_feat_g_f, img_emb_l_f
        )
        if self.cfg.model.limitr.lateral:
            img_lateral_global = torch.zeros(img_emb_g_f.shape,dtype=img_emb_g_f.dtype).cuda()
            #img_feats_lateral_all = torch.zeros(img_emb_l_f.shape,dtype=img_emb_l_f.dtype).cuda()
            tmp_remove = imgs_l[:, 0, 0, 0]
            ind_lateral = (tmp_remove == 0)
            # feed forward to the feature extractor only cases with lateral images
            imgs_l_continue = imgs_l[~ind_lateral]
            # img_feats_lateral = self.img_feats_lateral(path_lateral_continue)

            img_feat_g_l, img_emb_l_l = self.img_encoder_l(imgs_l_continue, get_local=True)
            img_emb_g_l, img_emb_l_l = self.img_encoder_l.generate_embeddings(
                img_feat_g_l, img_emb_l_l
            )
        # img_feats_lateral_all[~ind_lateral] = img_emb_l_l
        # img_feats_lateral_all[ind_lateral] = 1e-8 * torch.ones((768, 19,19),dtype=img_emb_l_l.dtype).cuda()
            img_lateral_global[~ind_lateral] = img_emb_g_l
            img_emb_g = torch.stack([img_emb_g_f, img_lateral_global])
            img_emb_g = img_emb_g.mean(axis=0)
        else:
            ind_lateral = []
            img_emb_l_l = []
            img_emb_g = img_emb_g_f
        # ih, iw = img_feats_lateral_all.size(2), img_feats_lateral_all.size(3)
        # sourceL = ih * iw
        # img_feats_lateral_all = img_feats_lateral_all.view(img_feats_lateral_all.size(0), -1, sourceL)
        # img_emb_l_f = img_emb_l_f.view(img_emb_l_f.size(0), -1, sourceL)
        #img_emb_l = torch.cat((img_emb_l_f, img_feats_lateral_all), dim=2)

        return img_emb_l_l,img_emb_l_f, img_emb_g,ind_lateral

    def attention_fn(self,query, context, temp1,context_img):
        """
        query: batch x ndf x queryL
        context: batch x ndf x ih x iw (sourceL=ihxiw)
        mask: batch_size x sourceL
        """
        if context_img:
            batch_size, queryL = query.size(0), query.size(2)
            # ih, iw = context.size(2), context.size(3)
            sourceL = context.size(2)
            contextT = torch.transpose(context, 1, 2).contiguous()
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

        return weightedContext, attn #.view(batch_size, -1, ih, iw)

    def local_similarities(
            self,img_features_l,img_features_f,ind_lateral, words_emb, cap_lens,sents, temp1=4.0
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
            #TODO original pe
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
            # TODO original pe
            if self.cfg.model.limitr.pe:
                pe = self.posemb_sincos_2d().cuda()
                img_features = img_features + pe.T

        # att_maps = []
        similarities_cap = []
        similarities_img = []
        att_maps = []
        # cap_lens = cap_lens.data.tolist()
        for i in range(words_emb.shape[0]):
            # Get the i-th text description
            words_num = cap_lens[i]  # 25
            # TODO: remove [SEP]
            # word = words_emb[i, :, 1:words_num+1].unsqueeze(0).contiguous()    # [1, 768, 25]
            word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()  # [1, 768, 25]
            word = word.repeat(batch_size, 1, 1)  # [48, 768, 25]
            context = img_features  # [48, 768, 19, 19]
            if self.cfg.model.limitr.cap_attn:
                weiContext, attn = self.attention_fn(
                    word, context, temp1, context_img=True
                )  # [48, 768, 25], [48, 25, 19, 19]
                att_maps.append(
                    attn[i].unsqueeze(0).contiguous()
                )
                word = word.transpose(1, 2).contiguous()  # [48, 25, 768]
                weiContext = weiContext.transpose(1, 2).contiguous()  # [48, 25, 768]

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
                # sim_loc_img = torch.cat([sim_loc_img_f, sim_loc_img_l],dim=0)
            # weiContext_img_l, _ = self.attention_fn(
            #     img_features_l, word, temp1, context_img=False
            # )
            # att_maps.append(
            #     attn[i].unsqueeze(0).contiguous()
            # )  # add attention for curr index  [25, 19, 19]
            #context = context.transpose(1, 2).contiguous()
            #weiContext_img = weiContext_img.transpose(1, 2).contiguous()
            # img_features_l = img_features_l.transpose(1, 2).contiguous()
            # img_features_f = img_features_f.transpose(1, 2).contiguous()
            # TODO- gloria
            # word = word.view(batch_size * words_num, -1)  # [1200, 768]
            # weiContext = weiContext.view(batch_size * words_num, -1)  # [1200, 768]
            # row_sim = cosine_similarity(word, weiContext)
            # row_sim = row_sim.view(batch_size, words_num)  # [48, 25]
            #
            # row_sim.mul_(temp2).exp_()
            # if agg == "sum":
            #     row_sim = row_sim.sum(dim=1, keepdim=True)  # [48, 1]
            # else:
            #     row_sim = row_sim.mean(dim=1, keepdim=True)  # [48, 1]
            # row_sim = torch.log(row_sim)
            #
            # similarities.append(row_sim)
            # TODO-gloria

        return similarities_cap, similarities_img,att_maps

        #return similarities_cap

    def global_txt_attention(self,img_features_f, txt_global_emb, temp1=4.0):
        batch_size = img_features_f.shape[0]
        img_features_f = img_features_f.type(txt_global_emb.dtype)
        ih, iw = img_features_f.size(2), img_features_f.size(3)
        sourceL = ih * iw
        img_features_f = img_features_f.view(img_features_f.size(0), -1, sourceL)
        img_features = img_features_f
        #txt_global_emb = txt_global_emb.repeat(batch_size, 1, 1)
        txt_global_emb=txt_global_emb.unsqueeze(2)
        weiContext, attn = self.attention_fn(
            txt_global_emb, img_features, temp1, context_img=True
        )  # [48, 768, 25], [48, 25, 19, 19]
        return attn[0].unsqueeze(0).contiguous()


    def _calc_local_loss(self, img_emb_l_l,img_emb_l_f,ind_lateral, text_emb_l, sents):

        cap_lens = [
            len([w for w in sent if not w.startswith("[")]) + 1 for sent in sents
        ]
        #TODO-local img attn

        sim_loc_cap, sim_loc_img,attn_maps = self.local_similarities(img_emb_l_l,
            img_emb_l_f,
            ind_lateral,
            text_emb_l,
            cap_lens,
            sents,#TODO- qt visualization remove
            temp1=self.temp1)
        l_loss0_cap, l_loss1_cap,l_loss0_img, l_loss1_img = self.local_loss(
            sim_loc_cap,
            sim_loc_img,
            cfg=self.cfg,
            temp3=self.temp3,
        )
        sim_loc = 0.5*(sim_loc_cap.cuda() + sim_loc_img.cuda())
        l_loss0 = 0.5*(l_loss0_cap+l_loss0_img)
        l_loss1 = 0.5 * (l_loss1_cap + l_loss1_img)

        # sim_loc_cap = self.local_similarities(img_emb_l,
        #     #text_emb_l.permute((0, 2, 1)),
        #     text_emb_l,
        #     cap_lens,
        #     temp1=self.temp1)
        # l_loss0, l_loss1 = self.local_loss(
        #     sim_loc_cap,
        #     temp3=self.temp3,
        # )
        # sim_loc = sim_loc_cap

        # img_emb_l = img_emb_l.view(img_emb_l.shape[0], -1, img_emb_l.shape[2])
        # img_emb_l = img_emb_l.permute((0, 2, 1))

        #text_emb_l = text_emb_l.permute((0, 2, 1))
        #img_ave = torch.mean(img_emb_l, 1)
        #txt_ave = torch.mean(text_emb_l, 1)
        # _, img_weights_raw = self.attn_model_img(img_emb_l, img_emb_l,img_emb_l)
        # img_weights = torch.mean(img_weights_raw, axis=1)
        # _, cap_weights_raw = self.attn_model_cap(text_emb_l, text_emb_l,text_emb_l)
        # cap_weights = torch.mean(cap_weights_raw, axis=1)

        # bz = img_emb_l.shape[0]
        # with torch.no_grad():
        #     atten_weights = text_attn.detach()
        #     cap_weights = []
        #     for i in range(bz):
        #         atten_weight = atten_weights[i]
        #         nonzero = atten_weight.nonzero().squeeze()
        #         low = torch.quantile(atten_weight[nonzero], 0.1)
        #         high = torch.quantile(atten_weight[nonzero], 0.9)
        #         atten_weight[nonzero] = atten_weight[nonzero].clip(low, high)
        #         cap_weights.append(atten_weight.clone())
        #     cap_weights = torch.stack(cap_weights)
        # cap_weights /= cap_weights.sum(dim=1, keepdims=True)
        if self.cfg.model.limitr.local_int_loss_weight>0:
            local_int_loss = self.local_int_loss(img_emb_l_l,img_emb_l_f,text_emb_l.permute((0, 2, 1)),sents,self.cfg,self.SAF_module,local_temperature = 0.1)
        else:
            local_int_loss = 0

        return l_loss0, l_loss1,local_int_loss,sim_loc,attn_maps

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
            sim_glo= torch.zeros_like(sim_loc)
            global_loss=0
        #norm = lambda x: (x - x.mean(axis=0)) / (x.std(axis=0))
        #sim_glo = torch.clamp(sim_glo, min=0)
        similarities = torch.stack(
            [self.cfg.model.limitr.local_ext_loss_weight*sim_loc, self.cfg.model.limitr.global_loss_weight*sim_glo]
        )


        # similarities = np.stack([local_similarities, global_similarities])
        similarities = similarities.mean(axis=0)
        # weighted loss
        return loss,local_ext_loss,local_int_loss,global_loss,similarities,sim_loc,sim_glo,attn_maps

    def forward(self, x):

        # img encoder branch
        #img_emb_l, img_emb_g = self.image_encoder_forward(x["imgs_F"],x["imgs_L"])
        img_emb_l_l, img_emb_l_f, img_emb_g, ind_lateral = self.image_encoder_forward(x["imgs_F"],x["imgs_L"])
        # text encorder branch
        text_emb_l, text_emb_g, sents = self.text_encoder_forward(
            x["caption_ids"], x["attention_mask"], x["token_type_ids"]
        )

        return img_emb_l_l,img_emb_l_f,ind_lateral, img_emb_g, text_emb_l, text_emb_g, sents

    def get_global_similarities(self, img_emb_g, text_emb_g):
        img_emb_g = img_emb_g.detach().cpu().numpy()
        text_emb_g = text_emb_g.detach().cpu().numpy()
        global_similarities = metrics.pairwise.cosine_similarity(img_emb_g, text_emb_g)
        global_similarities = torch.Tensor(global_similarities)
        return global_similarities

    def get_local_similarities(self, img_emb_l, text_emb_l, cap_lens):

        batch_size = img_emb_l.shape[0]
        similarities = []

        for i in range(len(text_emb_l)):
            words_num = cap_lens[i]
            word = (
                text_emb_l[i, :, 1 : words_num + 1].unsqueeze(0).contiguous()
            )  # [1, 768, 25]

            word = word.repeat(batch_size, 1, 1)  # [48, 768, 25]
            context = img_emb_l  # [48, 768, 19, 19]

            weiContext, attn = loss.LIMITR_loss.attention_fn(
                word, context, 4.0
            )  # [48, 768, 25], [48, 25, 19, 19]

            word = word.transpose(1, 2).contiguous()  # [48, 25, 768]
            weiContext = weiContext.transpose(1, 2).contiguous()  # [48, 25, 768]

            word = word.view(batch_size * words_num, -1)  # [1200, 768]
            weiContext = weiContext.view(batch_size * words_num, -1)  # [1200, 768]
            #
            row_sim = loss.LIMITR_loss.cosine_similarity(word, weiContext)
            row_sim = row_sim.view(batch_size, words_num)  # [48, 25]

            row_sim.mul_(5.0).exp_()
            row_sim, max_row_idx = torch.max(row_sim, dim=1, keepdim=True)  # [48, 1]

            row_sim = torch.log(row_sim)

            similarities.append(row_sim)

        local_similarities = torch.cat(similarities, 1).detach().cpu()

        return local_similarities

    def get_attn_maps(self, img_emb_l_l,img_emb_l_f,ind_lateral, text_emb_l, sents,text_emb_g):
        #attn_maps_global = self.global_txt_attention(img_emb_l_f, text_emb_g, temp1=4.0)
        _, _,_,_,attn_maps = self._calc_local_loss(img_emb_l_l,img_emb_l_f,ind_lateral, text_emb_l, sents)
        return attn_maps #,attn_maps_global

    def plot_attn_maps(self, attn_maps, imgs, sents, epoch_idx=0, batch_idx=0):

        img_set, _ = utils.build_attention_images(
            imgs,
            attn_maps,
            max_word_num=self.cfg.data.text.word_num,
            nvis=self.cfg.train.nvis,
            rand_vis=self.cfg.train.rand_vis,
            sentences=sents,
        )

        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = (
                f"{self.cfg.output_dir}/"
                f"attention_maps_epoch{epoch_idx}_"
                f"{batch_idx}.png"
            )
            im.save(fullpath)

    def posemb_sincos_2d(self,num_local_regions=361, temperature=500000, dtype=torch.float32,img_dim=256):
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

    def process_text(self, text, device):

        if type(text) == str:
            text = [text]

        processed_text_tensors = []
        for t in text:
            # use space instead of newline
            t = t.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(t)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            all_sents = []

            for t in captions:
                t = t.replace("\ufffd\ufffd", " ")
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(t.lower())

                if len(tokens) <= 1:
                    continue

                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                all_sents.append(" ".join(included_tokens))

            t = " ".join(all_sents)

            text_tensors = self.tokenizer(
                t,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.cfg.data.text.word_num,
            )
            text_tensors["sent"] = [
                self.ixtoword[ix] for ix in text_tensors["input_ids"][0].tolist()
            ]
            processed_text_tensors.append(text_tensors)

        caption_ids = torch.stack([x["input_ids"] for x in processed_text_tensors])
        attention_mask = torch.stack(
            [x["attention_mask"] for x in processed_text_tensors]
        )
        token_type_ids = torch.stack(
            [x["token_type_ids"] for x in processed_text_tensors]
        )

        if len(text) == 1:
            caption_ids = caption_ids.squeeze(0).to(device)
            attention_mask = attention_mask.squeeze(0).to(device)
            token_type_ids = token_type_ids.squeeze(0).to(device)
        else:
            caption_ids = caption_ids.squeeze().to(device)
            attention_mask = attention_mask.squeeze().to(device)
            token_type_ids = token_type_ids.squeeze().to(device)

        cap_lens = []
        for txt in text:
            cap_lens.append(len([w for w in txt if not w.startswith("[")]))

        return {
            "caption_ids": caption_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "cap_lens": cap_lens,
        }

    def process_mask(self,mask):
        transform= transforms.Compose([
            transforms.CenterCrop(224)])
        x = mask
        x = self._resize_img(x, self.cfg.data.image.imsize)
        x = x[np.newaxis,:, : ]
        img = np.concatenate([x, x, x], axis=0)
        img = transform(torch.tensor(img))
        return img

    def process_class_prompts(self, class_prompts, device):

        cls_2_processed_txt = {}
        for k, v in class_prompts.items():
            cls_2_processed_txt[k] = self.process_text(v, device)

        return cls_2_processed_txt

    def process_img(self, paths, device):

        transform = builder.build_transformation(self.cfg, split="test")

        # if type(paths) == str:
        #     paths = [paths]

        all_imgs = []
        for p in paths:
            my_path = '/home/gefen/Documents/Chexpert8X200'
            # chexpert8X200
            path_remove = 'CheXpert-v1.0-small'
            # chexpert5X200
            #path_remove = 'CheXpert-v1.0'
            p_new = os.path.relpath(p, path_remove)
            # my_path = '/home/gefen/Documents/physionet.org/files/mimic-cxr-jpg/2.0.0'
            #joined_path = os.path.join(my_path, p)
            joined_path = os.path.join(my_path, p_new)
            x = cv2.imread(joined_path, 0)

            # tranform images
            x = self._resize_img(x, self.cfg.data.image.imsize)
            img = Image.fromarray(x).convert("RGB")
            img = transform(img)
            all_imgs.append(torch.tensor(img))

        all_imgs = torch.stack(all_imgs).to(device)

        return all_imgs

    def process_img_attn(self, paths, device,mask):
        if mask:
            transform = builder.build_transformation(self.cfg, split="mask")
        else:
            transform = builder.build_transformation(self.cfg, split="test")

        if type(paths) == str:
            paths = [paths]

        all_imgs = []
        for p in paths:
            # my_path = '/home/gefen/Documents/chexpert5X200'
            # path_remove = 'CheXpert-v1.0'
            # p_new = os.path.relpath(p, path_remove)
            # joined_path = os.path.join(my_path, p_new)
            # x = cv2.imread(joined_path, 0)
            # x = self._resize_img(x, self.cfg.data.image.imsize)
            # img = Image.fromarray(x).convert("RGB")
            if not mask:
                my_path = '/home/gefen/Documents/physionet.org/files/mimic-cxr-jpg/2.0.0'
                joined_path = os.path.join(my_path, p)
                x = cv2.imread(joined_path, 0)
                x = self._resize_img(x, self.cfg.data.image.imsize)
                img = Image.fromarray(x).convert("RGB")
            else:
                x = p
                x = self._resize_img(x, self.cfg.data.image.imsize)
                x = x[:, :, np.newaxis]
                img = np.concatenate([x, x, x], axis=2)
            # tranform images

            #
            img = transform(img)
            all_imgs.append(torch.tensor(img))

        all_imgs = torch.stack(all_imgs).to(device)

        return all_imgs

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

        return resized_img

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


