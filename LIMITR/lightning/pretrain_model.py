import torch

from PIL import Image
from .. import builder
import numpy as np
import sys
from .. import loss


from pytorch_lightning.core import LightningModule
from torch.autograd import Variable


class PretrainModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.limitr = builder.build_model(cfg)
        self.lr = cfg.lightning.trainer.lr
        self.dm = None

    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.limitr)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        loss, sents = self.shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        scores,loss = self.shard_attn_scores(batch)
        r, _ = self.i2t(scores, return_ranks=True)
        ri, _ = self.t2i(scores, return_ranks=True)
        r_sum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        self.log("r_sum", r_sum, on_step=False, prog_bar=True,
                 on_epoch=True,)
        self.log("r1_i2t", r[0], on_step=False, prog_bar=True,
                 on_epoch=True)
        self.log("r5_i2t", r[1], on_step=False, prog_bar=True,
                 on_epoch=True)
        self.log("r10_i2t", r[2], on_step=False, prog_bar=True,
                 on_epoch=True)
        self.log("r1_t2i", ri[0], on_step=False, prog_bar=True,
                 on_epoch=True)
        self.log("r5_t2i", ri[1], on_step=False, prog_bar=True,
                 on_epoch=True)
        self.log("r10_t2i", ri[2], on_step=False, prog_bar=True,
                 on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        scores,loss = self.shard_attn_scores(batch)
        r, _ = self.i2t(scores, return_ranks=True)
        ri, _ = self.t2i(scores, return_ranks=True)
        r_sum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        self.log("r_sum_test", r_sum, on_step=False, prog_bar=True,
                 on_epoch=True,)
        self.log("r1_i2t_test", r[0], on_step=False, prog_bar=True,
                 on_epoch=True)
        self.log("r5_i2t_test", r[1], on_step=False, prog_bar=True,
                 on_epoch=True)
        self.log("r10_i2t_test", r[2], on_step=False, prog_bar=True,
                 on_epoch=True)
        self.log("r1_t2i_test", ri[0], on_step=False, prog_bar=True,
                 on_epoch=True)
        self.log("r5_t2i_test", ri[1], on_step=False, prog_bar=True,
                 on_epoch=True)
        self.log("r10_t2i_test", ri[2], on_step=False, prog_bar=True,
                 on_epoch=True)
        return r_sum

    def shard_attn_scores(self,batch, shard_size=100):
        n_examples = len(batch['imgs_F'])
        n_im_shard = (n_examples - 1) // shard_size + 1
        n_cap_shard = (n_examples - 1) // shard_size + 1
        total_loss = 0

        sims = np.zeros((n_examples, n_examples))
        for i in range(n_im_shard):
            im_start, im_end = shard_size * i, min(shard_size * (i + 1), n_examples)
            for j in range(n_cap_shard):
                sys.stdout.write('\r>> shard_attn_scores batch (%d,%d)' % (i, j))
                ca_start, ca_end = shard_size * j, min(shard_size * (j + 1), n_examples)

                with torch.no_grad():
                    new_batch = {
                        "caption_ids": batch["caption_ids"][ca_start:ca_end],
                        "attention_mask": batch["attention_mask"][ca_start:ca_end],
                        "token_type_ids": batch["token_type_ids"][ca_start:ca_end],
                        "imgs_F": batch["imgs_F"][im_start:im_end],
                        "imgs_L": batch["imgs_L"][im_start:im_end]
                    }
                    img_emb_l_l, img_emb_l_f, ind_lateral, img_emb_g, text_emb_l, text_emb_g, sents = self.limitr(new_batch)
                    loss, _,_,_, sim,_,_,attn_maps = self.limitr.calc_loss(
                        img_emb_l_l, img_emb_l_f, ind_lateral, img_emb_g, text_emb_l, text_emb_g, sents
                    )

                sims[im_start:im_end, ca_start:ca_end] = sim.data.cpu().numpy()
                total_loss += loss

        sys.stdout.write('\n')
        return sims, total_loss/shard_size

    @staticmethod
    def i2t(sims, return_ranks=False):
        """
        Images->Text (Image Annotation)
        Images: (N, n_region, d) matrix of images
        Captions: (N, max_n_word, d) matrix of captions
        CapLens: (N) array of caption lengths
        sims: (N, N) matrix of similarity im-cap
        """
        npts = sims.shape[0]
        ranks = np.zeros(npts)
        top1 = np.zeros(npts)
        top5 = np.zeros((npts, 5))
        top10 = np.zeros((npts, 10))

        for index in range(npts):
            inds = np.argsort(-1 * sims[index])
            tmp = np.where(inds == index)[0][0]
            ranks[index] = tmp
            top1[index] = inds[0]
            top5[index] = inds[0:5]
            top10[index] = inds[0:10]

        # Compute metrics
        r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        medr = np.floor(np.median(ranks)) + 1
        meanr = ranks.mean() + 1
        MRR = np.sum(1 / (ranks + 1)) / len(ranks)
        if return_ranks:
            return (r1, r5, r10, medr, meanr, MRR), (ranks, top1, top5, top10)
        else:
            return (r1, r5, r10, medr, meanr, MRR)

    @staticmethod
    def t2i(sims, return_ranks=False):
        """
        Text->Images (Image Search)
        Images: (N, n_region, d) matrix of images
        Captions: (N, max_n_word, d) matrix of captions
        CapLens: (N) array of caption lengths
        sims: (N, N) matrix of similarity im-cap
        """
        npts = sims.shape[0]
        ranks = np.zeros(npts)
        top1 = np.zeros(npts)
        top5 = np.zeros((npts, 5))
        top10 = np.zeros((npts, 10))

        # --> (5N(caption), N(image))
        sims = sims.T

        for index in range(npts):
            inds = np.argsort(-1 * sims[index])
            ranks[index] = np.where(inds == index)[0][0]
            top1[index] = inds[0]
            top5[index] = inds[0:5]
            top10[index] = inds[0:10]

        # Compute metrics
        r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        medr = np.floor(np.median(ranks)) + 1
        meanr = ranks.mean() + 1
        MRR = np.sum(1 / (ranks + 1)) / len(ranks)
        if return_ranks:
            return (r1, r5, r10, medr, meanr, MRR), (ranks, top1, top5, top10)
        else:
            return (r1, r5, r10, medr, meanr, MRR)

    def shared_step(self, batch, split):
        """Similar to traning step"""

        img_emb_l_l, img_emb_l_f, ind_lateral, img_emb_g, text_emb_l, text_emb_g, sents = self.limitr(batch)
        loss,local_ext_loss,local_int_loss,global_loss,_,_,_,_ = self.limitr.calc_loss(
            img_emb_l_l, img_emb_l_f,ind_lateral, img_emb_g, text_emb_l, text_emb_g, sents
        )

        # log training progress
        log_iter_loss = True if split == "train" else False
        self.log(
            f"{split}_loss",
            loss,
            on_epoch=True,
            on_step=log_iter_loss,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "local_int_loss",
            local_int_loss,
            on_epoch=True,
            on_step=log_iter_loss,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "local_ext_loss",
            local_ext_loss,
            on_epoch=True,
            on_step=log_iter_loss,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "global_loss",
            global_loss,
            on_epoch=True,
            on_step=log_iter_loss,
            logger=True,
            prog_bar=True,
        )
        return loss, sents
