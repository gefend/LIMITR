import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer,BertTokenizer

# class BertEncoder(nn.Module):
#     def __init__(self,
#                  tokenizer: BertTokenizer = None,
#                  emb_dim: int = 768,
#                  output_dim: int = 128,
#                  hidden_dim: int = 2048,
#                  freeze_bert: bool = True):
#         super(BertEncoder, self).__init__()
#         self.bert_type = "emilyalsentzer/Bio_ClinicalBERT"
#         self.last_n_layers = 1
#         self.aggregate_method = "sum"
#         self.embedding_dim = emb_dim
#         self.output_dim = output_dim
#         self.freeze_bert = freeze_bert
#         self.agg_tokens = True
#         # self.max_sent_num = 10
#
#         self.config = BertConfig.from_json_file(
#             os.path.join(BASE_DIR, "../../configs/bert_config.json"))
#         self.model = BertModel.from_pretrained(
#             self.bert_type,
#             config=self.config,
#             add_pooling_layer=False,
#         )
#
#         if tokenizer:
#             self.tokenizer = tokenizer
#         else:
#             self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
#
#         self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}
#
#         if self.freeze_bert is True:
#             print("Freezing BERT model")
#             for param in self.model.parameters():
#                 param.requires_grad = False
#
#         # self.global_embed = GlobalEmbedding(
#         #     self.embedding_dim, hidden_dim, self.output_dim)
#         # self.local_embed = LocalEmbedding(
#         #     self.embedding_dim, hidden_dim, self.output_dim)
#
#     def aggregate_tokens(self, embeddings, caption_ids, last_layer_attn):
#         '''
#         :param embeddings: bz, 1, 112, 768
#         :param caption_ids: bz, 112
#         :param last_layer_attn: bz, 111
#         '''
#         _, num_layers, num_words, dim = embeddings.shape
#         embeddings = embeddings.permute(0, 2, 1, 3)
#         agg_embs_batch = []
#         sentences = []
#         last_attns = []
#
#         # loop over batch
#         for embs, caption_id, last_attn in zip(embeddings, caption_ids, last_layer_attn):
#             agg_embs = []
#             token_bank = []
#             words = []
#             word_bank = []
#             attns = []
#             attn_bank = []
#
#             # loop over sentence
#             for word_emb, word_id, attn in zip(embs, caption_id, last_attn):
#                 word = self.idxtoword[word_id.item()]
#                 if word == "[SEP]":
#                     new_emb = torch.stack(token_bank)
#                     new_emb = new_emb.sum(axis=0)
#                     agg_embs.append(new_emb)
#                     words.append("".join(word_bank))
#                     attns.append(sum(attn_bank))
#                     agg_embs.append(word_emb)
#                     words.append(word)
#                     attns.append(attn)
#                     break
#                 # This is because some words are divided into two words.
#                 if not word.startswith("##"):
#                     if len(word_bank) == 0:
#                         token_bank.append(word_emb)
#                         word_bank.append(word)
#                         attn_bank.append(attn)
#                     else:
#                         new_emb = torch.stack(token_bank)
#                         new_emb = new_emb.sum(axis=0)
#                         agg_embs.append(new_emb)
#                         words.append("".join(word_bank))
#                         attns.append(sum(attn_bank))
#
#                         token_bank = [word_emb]
#                         word_bank = [word]
#                         attn_bank = [attn]
#                 else:
#                     token_bank.append(word_emb)
#                     word_bank.append(word[2:])
#                     attn_bank.append(attn)
#             agg_embs = torch.stack(agg_embs)
#             padding_size = num_words - len(agg_embs)
#             paddings = torch.zeros(padding_size, num_layers, dim)
#             paddings = paddings.type_as(agg_embs)
#             words = words + ["[PAD]"] * padding_size
#             last_attns.append(
#                 torch.cat([torch.tensor(attns), torch.zeros(padding_size)], dim=0))
#             agg_embs_batch.append(torch.cat([agg_embs, paddings]))
#             sentences.append(words)
#
#         agg_embs_batch = torch.stack(agg_embs_batch)
#         agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)
#         last_atten_pt = torch.stack(last_attns)
#         last_atten_pt = last_atten_pt.type_as(agg_embs_batch)
#
#         return agg_embs_batch, sentences, last_atten_pt
#
#     def forward(self, ids, attn_mask, token_type, get_local=False):
#         outputs = self.model(ids, attn_mask, token_type,
#                              return_dict=True, mode="text")
#
#         last_layer_attn = outputs.attentions[-1][:, :, 0, 1:].mean(dim=1)
#         all_feat = outputs.last_hidden_state.unsqueeze(1)
#
#         if self.agg_tokens:
#             all_feat, sents, last_atten_pt = self.aggregate_tokens(
#                 all_feat, ids, last_layer_attn)
#             last_atten_pt = last_atten_pt[:, 1:].contiguous()
#         else:
#             sents = [[self.idxtoword[w.item()] for w in sent]
#                      for sent in ids]
#
#         if self.last_n_layers == 1:
#             all_feat = all_feat[:, 0]
#
#         report_feat = all_feat[:, 0].contiguous()
#         word_feat = all_feat[:, 1:].contiguous()
#
#         return report_feat, word_feat, last_atten_pt, sents
#
class BertEncoder(nn.Module):
    def __init__(self, cfg):
        super(BertEncoder, self).__init__()

        #self.bert_type = cfg.model.text.bert_type
        self.last_n_layers = cfg.model.text.last_n_layers
        self.aggregate_method = cfg.model.text.aggregate_method
        self.norm = cfg.model.text.norm
        self.embedding_dim = cfg.model.text.embedding_dim
        self.freeze_bert = cfg.model.text.freeze_bert
        self.agg_tokens = cfg.model.text.agg_tokens

        # self.model = AutoModel.from_pretrained(
        #     self.bert_type, output_hidden_states=True
        # )
        self.model=AutoModel.from_pretrained(
            '/home/gefen/Documents/gloria-main/gloria/models/bioclinicalbert', output_hidden_states=True
        )

        #self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        # self.tokenizer= BertTokenizer.from_pretrained(
        #     "emilyalsentzer/Bio_ClinicalBERT")
        self.tokenizer= BertTokenizer.from_pretrained(
            "/home/gefen/Documents/gloria-main/gloria/models/bioclinicalbert_tokenizer")
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

        self.emb_global, self.emb_local = None, None

        if self.freeze_bert is True:
            print("Freezing BERT model")
            for param in self.model.parameters():
                param.requires_grad = False

    def aggregate_tokens(self, embeddings, caption_ids):

        batch_size, num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3)
        agg_embs_batch = []
        sentences = []

        # loop over batch
        for embs, caption_id in zip(embeddings, caption_ids):

            agg_embs = []
            token_bank = []
            words = []
            word_bank = []

            # loop over sentence
            for word_emb, word_id in zip(embs, caption_id):

                word = self.idxtoword[word_id.item()]

                if word == "[SEP]":
                    new_emb = torch.stack(token_bank)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    words.append("".join(word_bank))

                    agg_embs.append(word_emb)
                    words.append(word)
                    break

                if not word.startswith("##"):
                    if len(word_bank) == 0:
                        token_bank.append(word_emb)
                        word_bank.append(word)
                    else:
                        new_emb = torch.stack(token_bank)
                        new_emb = new_emb.sum(axis=0)
                        agg_embs.append(new_emb)
                        words.append("".join(word_bank))

                        token_bank = [word_emb]
                        word_bank = [word]
                else:
                    if word.startswith("##"):
                        token_bank.append(word_emb)
                        word_bank.append(word[2:])

            agg_embs = torch.stack(agg_embs)
            padding_size = num_words - len(agg_embs)
            paddings = torch.zeros(padding_size, num_layers, dim)
            paddings = paddings.to(agg_embs.device)
            words = words + ["[PAD]"] * padding_size

            agg_embs_batch.append(torch.cat([agg_embs, paddings]))
            sentences.append(words)

        agg_embs_batch = torch.stack(agg_embs_batch)
        agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)
        return agg_embs_batch, sentences

    def forward(self, ids, attn_mask, token_type):

        outputs = self.model(ids, attn_mask, token_type)

        # aggregate intermetidate layers
        if self.last_n_layers > 1:
            all_embeddings = outputs[2]
            embeddings = torch.stack(
                all_embeddings[-self.last_n_layers :]
            )  # layers, batch, sent_len, embedding size

            embeddings = embeddings.permute(1, 0, 2, 3)

            if self.agg_tokens:
                embeddings, sents = self.aggregate_tokens(embeddings, ids)
            else:
                sents = [[self.idxtoword[w.item()] for w in sent] for sent in ids]

            sent_embeddings = embeddings.mean(axis=2)

            if self.aggregate_method == "sum":
                word_embeddings = embeddings.sum(axis=1)
                sent_embeddings = sent_embeddings.sum(axis=1)
            elif self.aggregate_method == "mean":
                word_embeddings = embeddings.mean(axis=1)
                sent_embeddings = sent_embeddings.mean(axis=1)
            else:
                print(self.aggregate_method)
                raise Exception("Aggregation method not implemented")

        # use last layer
        else:
            word_embeddings, sent_embeddings = outputs[0], outputs[1]

        batch_dim, num_words, feat_dim = word_embeddings.shape
        word_embeddings = word_embeddings.view(batch_dim * num_words, feat_dim)
        if self.emb_local is not None:
            word_embeddings = self.emb_local(word_embeddings)
        word_embeddings = word_embeddings.view(batch_dim, num_words, self.embedding_dim)
        word_embeddings = word_embeddings.permute(0, 2, 1)

        if self.emb_global is not None:
            sent_embeddings = self.emb_global(sent_embeddings)

        if self.norm is True:
            word_embeddings = word_embeddings / torch.norm(
                word_embeddings, 2, dim=1, keepdim=True
            ).expand_as(word_embeddings)
            sent_embeddings = sent_embeddings / torch.norm(
                sent_embeddings, 2, dim=1, keepdim=True
            ).expand_as(sent_embeddings)

        return word_embeddings, sent_embeddings, sents
