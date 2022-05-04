import enum
import sys
sys.path+=['./']
import torch
import torch.nn as nn
import transformers
from transformers import LongformerModel, RobertaModel
if int(transformers.__version__[0])<=3:
    from transformers.modeling_roberta import RobertaPreTrainedModel
    from transformers.modeling_longformer import LongformerPreTrainedModel
else:
    from transformers.models.longformer.modeling_longformer import LongformerPreTrainedModel
    from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
import torch.nn.functional as F
from torch.cuda.amp import autocast

class EmbeddingMixin:
    """
    Mixin for common functions in most embedding models. Each model should define its own bert-like backbone and forward.
    We inherit from LongformerModel to use from_pretrained
    """
    def __init__(self, model_argobj):
        if model_argobj is None:
            self.use_mean = False
        else:
            self.use_mean = model_argobj.use_mean
        print("Using mean:", self.use_mean)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d

    def masked_mean_or_first(self, emb_all, mask):
        # emb_all is a tuple from bert - sequence output, pooler
        assert isinstance(emb_all, tuple)
        if self.use_mean:
            return self.masked_mean(emb_all[0], mask)
        else:
            return emb_all[0][:, 0]

    def query_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")

    def body_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")

class BaseModelDot(EmbeddingMixin):
    def _text_encode(self, input_ids, attention_mask):
        # TODO should raise NotImplementedError
        # temporarily do this  
        return None 

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self._text_encode(input_ids=input_ids,
                                attention_mask=attention_mask)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)

    def forward(self, input_ids, attention_mask, is_query, *args):
        assert len(args) == 0
        if is_query:
            return self.query_emb(input_ids, attention_mask)
        else:
            return self.body_emb(input_ids, attention_mask)

class LongformerDot(BaseModelDot, LongformerPreTrainedModel):
    def __init__(self, config, model_argobj=None):
        BaseModelDot.__init__(self, model_argobj)
        LongformerPreTrainedModel.__init__(self, config)
        if int(transformers.__version__[0]) ==4 :
            config.return_dict = False
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        if hasattr(config, "output_embedding_size"):
            self.output_embedding_size = config.output_embedding_size
        else:
            self.output_embedding_size = config.hidden_size
        print("output_embedding_size", self.output_embedding_size)
        self.embeddingHead = nn.Linear(config.hidden_size, self.output_embedding_size)
        self.norm = nn.LayerNorm(self.output_embedding_size)
        self.apply(self._init_weights)

    def _text_encode(self, input_ids, attention_mask):
        outputs1 = self.longformer(input_ids=input_ids,
                                attention_mask=attention_mask)
        return outputs1

class LongformerDot_InBatch(LongformerDot):
    def forward(self, input_query_ids, query_attention_mask,
            input_doc_ids, doc_attention_mask, 
            other_doc_ids=None, other_doc_attention_mask=None,
            rel_pair_mask=None, hard_pair_mask=None):
        return inbatch_train(self.query_emb, self.body_emb,
            input_query_ids, query_attention_mask,
            input_doc_ids, doc_attention_mask, 
            other_doc_ids, other_doc_attention_mask,
            rel_pair_mask, hard_pair_mask)

class LongformerDot_Rand(LongformerDot):
    def forward(self, input_query_ids, query_attention_mask,
            input_doc_ids, doc_attention_mask, 
            other_doc_ids=None, other_doc_attention_mask=None,
            rel_pair_mask=None, hard_pair_mask=None):
        return randneg_train(self.query_emb, self.body_emb,
            input_query_ids, query_attention_mask,
            input_doc_ids, doc_attention_mask, 
            other_doc_ids, other_doc_attention_mask,
            hard_pair_mask)

def inbatch_train(query_encode_func, doc_encode_func,
            input_query_ids, query_attention_mask,
            input_doc_ids, doc_attention_mask, 
            other_doc_ids=None, other_doc_attention_mask=None,
            rel_pair_mask=None, hard_pair_mask=None):
    
    query_embs = query_encode_func(input_query_ids, query_attention_mask)
    doc_embs = doc_encode_func(input_doc_ids, doc_attention_mask)
    batch_size = query_embs.shape[0]
    with autocast(enabled=False):
        batch_scores = torch.matmul(query_embs, doc_embs.T)
        device = batch_scores.get_device()
        rel_pair_mask = rel_pair_mask[:,batch_size * device:batch_size * (device + 1)]
        hard_pair_mask = hard_pair_mask[:, batch_size * device:batch_size * (device+1)]
        #print(rel_pair_mask.shape)
        # print("batch_scores", batch_scores)
        single_positive_scores = torch.diagonal(batch_scores, 0)
        # print("positive_scores", positive_scores)
        positive_scores = single_positive_scores.reshape(-1, 1).repeat(1, batch_size).reshape(-1)
        if rel_pair_mask is None:
            rel_pair_mask = 1 - torch.eye(batch_size, dtype=batch_scores.dtype, device=batch_scores.device)                
        # print("mask", mask)
        batch_scores = batch_scores.reshape(-1)
        logit_matrix = torch.cat([positive_scores.unsqueeze(1),
                                batch_scores.unsqueeze(1)], dim=1)  
        # print(logit_matrix)
        lsm = F.log_softmax(logit_matrix, dim=1)
        #print(lsm[:,0].shape)
        #print(rel_pair_mask.shape)
        loss = -1.0 * lsm[:, 0] * rel_pair_mask.reshape(-1)
        # print(loss)
        # print("\n")
        first_loss, first_num = loss.sum(), rel_pair_mask.sum()

    if other_doc_ids is None:
        return (first_loss/first_num,)

    # other_doc_ids: batch size, per query doc, length
    other_doc_num = other_doc_ids.shape[0] * other_doc_ids.shape[1]
    other_doc_ids = other_doc_ids.reshape(other_doc_num, -1)
    other_doc_attention_mask = other_doc_attention_mask.reshape(other_doc_num, -1)
    other_doc_embs = doc_encode_func(other_doc_ids, other_doc_attention_mask)
    
    with autocast(enabled=False):
        other_batch_scores = torch.matmul(query_embs, other_doc_embs.T)
        other_batch_scores = other_batch_scores.reshape(-1)
        positive_scores = single_positive_scores.reshape(-1, 1).repeat(1, other_doc_num).reshape(-1)
        other_logit_matrix = torch.cat([positive_scores.unsqueeze(1),
                                other_batch_scores.unsqueeze(1)], dim=1)  
        # print(logit_matrix)
        other_lsm = F.log_softmax(other_logit_matrix, dim=1)
        other_loss = -1.0 * other_lsm[:, 0]
        if hard_pair_mask is not None:
            hard_pair_mask = hard_pair_mask.reshape(-1)
            other_loss = other_loss * hard_pair_mask
            second_loss, second_num = other_loss.sum(), hard_pair_mask.sum()
        else:
            second_loss, second_num = other_loss.sum(), len(other_loss)
    
    return ((first_loss+second_loss)/(first_num+second_num),)

class RobertaDot(BaseModelDot, RobertaPreTrainedModel):
    def __init__(self, config, model_argobj=None):
        BaseModelDot.__init__(self, model_argobj)
        RobertaPreTrainedModel.__init__(self, config)
        if int(transformers.__version__[0]) ==4 :
            config.return_dict = False
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        if hasattr(config, "output_embedding_size"):
            self.output_embedding_size = config.output_embedding_size
        else:
            self.output_embedding_size = config.hidden_size
        print("output_embedding_size", self.output_embedding_size)
        self.embeddingHead = nn.Linear(config.hidden_size, self.output_embedding_size)
        self.norm = nn.LayerNorm(self.output_embedding_size)
        self.apply(self._init_weights)
        self.graph = GraphEncoder(config)

    def _text_encode(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        return outputs1

class RobertaDot_Rand(RobertaDot):
    def forward(self, input_query_ids, query_attention_mask,
            input_doc_ids, doc_attention_mask,
            other_doc_ids=None, other_doc_attention_mask=None,
            rel_pair_mask=None, hard_pair_mask=None):
        return randneg_train(self.query_emb, self.body_emb,
            input_query_ids, query_attention_mask,
            input_doc_ids, doc_attention_mask,
            other_doc_ids, other_doc_attention_mask,
            hard_pair_mask)

def randneg_train(query_encode_func, doc_encode_func,
            input_query_ids, query_attention_mask,
            input_doc_ids, doc_attention_mask, 
            other_doc_ids=None, other_doc_attention_mask=None,
            hard_pair_mask=None):

    query_embs = query_encode_func(input_query_ids, query_attention_mask)
    doc_embs = doc_encode_func(input_doc_ids, doc_attention_mask)

    with autocast(enabled=False):
        batch_scores = torch.matmul(query_embs, doc_embs.T)
        single_positive_scores = torch.diagonal(batch_scores, 0)
    # other_doc_ids: batch size, per query doc, length
    other_doc_num = other_doc_ids.shape[0] * other_doc_ids.shape[1]
    other_doc_ids = other_doc_ids.reshape(other_doc_num, -1)
    other_doc_attention_mask = other_doc_attention_mask.reshape(other_doc_num, -1)
    other_doc_embs = doc_encode_func(other_doc_ids, other_doc_attention_mask)
    
    with autocast(enabled=False):
        other_batch_scores = torch.matmul(query_embs, other_doc_embs.T)
        other_batch_scores = other_batch_scores.reshape(-1)
        positive_scores = single_positive_scores.reshape(-1, 1).repeat(1, other_doc_num).reshape(-1)
        other_logit_matrix = torch.cat([positive_scores.unsqueeze(1),
                                other_batch_scores.unsqueeze(1)], dim=1)
        other_lsm = F.log_softmax(other_logit_matrix, dim=1)
        other_loss = -1.0 * other_lsm[:, 0]
        if hard_pair_mask is not None:
            hard_pair_mask = hard_pair_mask.reshape(-1)
            other_loss = other_loss * hard_pair_mask
            second_loss, second_num = other_loss.sum(), hard_pair_mask.sum()
        else:
            second_loss, second_num = other_loss.sum(), len(other_loss)
    return (second_loss/second_num,)

class RobertaDot_InBatch(RobertaDot):
    def forward(self, input_query_ids, query_attention_mask,
            input_doc_ids, doc_attention_mask,
            other_doc_ids=None, other_doc_attention_mask=None,
            rel_pair_mask=None, hard_pair_mask=None):
        return inbatch_train(self.query_emb, self.body_emb,
            input_query_ids, query_attention_mask,
            input_doc_ids, doc_attention_mask,
            other_doc_ids, other_doc_attention_mask,
            rel_pair_mask, hard_pair_mask)

class GraphEncoder(nn.Module):
    def __init__(self, config):
        super(GraphEncoder, self).__init__()
        self.config = config
        if config.RGAT_interaction:
            self.edge_emb = nn.Parameters(torch.FloatTensor(3, config.hidden_size))
            self.use_RGAT_embedding = True
        if config.POS_interaction:
            self.use_pos_embeddings = True
            self.relative_key_embeddings = nn.Embedding(256, self.attention_head_size)
            self.relative_value_embeddings = nn.Embedding(256, self.attention_head_size)
            self.max_relative_position = max_relative_position
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, inputs):
        if self.config.RGAT_interaction:
            hidden_states, input_mask, sentence_doc_mask, sentence_para_mask, para_doc_mask = inputs
        else:
            hidden_state, input_mask = inputs
        batch_size = hidden_states.size(0)

        attention_mask = input_mask.unsqueeze(1).unsqueeze(2)

        attention_mask = attention_mask.to(dtype=hidden_states.dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * -10000.0

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if self.use_pos_embeddings:
            relative_attention_scores = torch.matmul(query_layer,
                                                     self.relative_key_embeddings.weight.transpose(-1, -2))

            batch_indices = self.indices.unsqueeze(0).unsqueeze(1).expand(batch_size, self.num_attention_heads, -1,
                                                                          -1)
            attention_scores = attention_scores + torch.gather(input=relative_attention_scores, dim=3,
                                                               index=batch_indices)

        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)

        if self.use_pos_embeddings:
            relative_attention_probs = torch.zeros_like(relative_attention_scores)
            relative_attention_probs.scatter_add_(dim=3, index=batch_indices, src=attention_probs)
            relative_values = torch.matmul(relative_attention_probs, self.relative_value_embeddings.weight)
            context_layer = context_layer + relative_values
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        node_embeddings = context_layer.view(*new_context_layer_shape)

        return node_embeddings

#raw code of RGAT from https://github.com/shenwzh3/RGAT-ABSA/
class RelationAttention(nn.Module):
    def __init__(self, in_dim = 300, hidden_dim = 64):
        # in_dim: the dimension fo query vector
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, feature, dep_tags_v, dmask):
        '''
        C feature/context [N, L, D]
        Q dep_tags_v          [N, L, L, D]

        mask dmask          [N, L]
        '''
        Q = self.fc1(dep_tags_v)
        Q = self.relu(Q)
        Q = self.fc2(Q)  # (N, L, 1)
        Q = Q.squeeze(2)
        Q = F.softmax(mask_logits(Q, dmask), dim=1)

        Q = Q.unsqueeze(2)
        out = torch.bmm(feature.transpose(1, 2), Q)
        out = out.squeeze(2)
        # out = F.sigmoid(out)
        return out  # ([N, L])

