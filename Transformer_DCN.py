import torch
from fuxictr.utils import not_in_whitelist
from torch import nn
import random
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, CrossNetV2


class Transformer_DCN(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="Transformer_DCN",
                 gpu=-1,
                 task="binary_classification",  # 添加默认的任务类型
                 hidden_activations="ReLU",
                 dcn_cross_layers=3,
                 dcn_hidden_units=[1024, 512, 256],
                 mlp_hidden_units=[64, 32],
                 # 添加 FinalMLP 相关参数
                 mlp1_hidden_units=[64, 64, 64],
                 mlp1_hidden_activations="ReLU",
                 mlp1_dropout=0,
                 mlp1_batch_norm=False,
                 mlp2_hidden_units=[64, 64, 64],
                 mlp2_hidden_activations="ReLU",
                 mlp2_dropout=0,
                 mlp2_batch_norm=False,
                 use_fs=True,
                 fs_hidden_units=[64],
                 fs1_context=[],
                 fs2_context=[],
                 
                 num_heads=1,
                 transformer_layers=2,
                 transformer_dropout=0.2,
                 dim_feedforward=256,
                 learning_rate=5e-4,
                 embedding_dim=64,
                 net_dropout=0.2,
                 first_k_cols=16,
                 batch_norm=False,
                 concat_max_pool=True,
                 accumulation_steps=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super().__init__(feature_map,
                         model_id=model_id,
                         gpu=gpu,
                         embedding_regularizer=embedding_regularizer,
                         net_regularizer=net_regularizer,
                         **kwargs)
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.item_info_dim = 0
        for feat, spec in self.feature_map.features.items():
            if spec.get("source") == "item":
                self.item_info_dim += spec.get("embedding_dim", embedding_dim)

        transformer_in_dim = self.item_info_dim * 2

        self.accumulation_steps = accumulation_steps
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

        self.transformer_encoder = Transformer(
            transformer_in_dim,
            dim_feedforward=dim_feedforward,
            num_heads=num_heads,
            dropout=transformer_dropout,
            transformer_layers=transformer_layers,
            first_k_cols=first_k_cols,
            concat_max_pool=concat_max_pool
        )
        seq_out_dim = (first_k_cols + int(concat_max_pool)) * transformer_in_dim
        
        # 添加 FinalMLP 相关组件
        finalmlp_in_dim = feature_map.sum_emb_out_dim() + seq_out_dim

        # 特征选择模块
        self.fs_module = None
        if use_fs:
            self.fs_module = FeatureSelectionLayer(finalmlp_in_dim, fs_hidden_units)

        # 定义两个流的特征选择
        self.fs1_context = fs1_context  # 流1的上下文特征索引
        self.fs2_context = fs2_context  # 流2的上下文特征索引

        # 两个流的 MLP
        self.mlp1 = MLP_Block(input_dim=finalmlp_in_dim,
                              output_dim=None,
                              hidden_units=mlp1_hidden_units,
                              hidden_activations=mlp1_hidden_activations,
                              output_activation=None,
                              dropout_rates=mlp1_dropout,
                              batch_norm=mlp1_batch_norm)

        self.mlp2 = MLP_Block(input_dim=finalmlp_in_dim,
                              output_dim=None,
                              hidden_units=mlp2_hidden_units,
                              hidden_activations=mlp2_hidden_activations,
                              output_activation=None,
                              dropout_rates=mlp2_dropout,
                              batch_norm=mlp2_batch_norm)

        # 双线性融合和最终输出层
        self.bilinear_fusion = BilinearFusion(mlp1_hidden_units[-1], mlp2_hidden_units[-1], num_heads)
        self.output_layer = nn.Linear(num_heads, 1)
        self.output_activation = self.get_output_activation(task)
        
        
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        emb_list = []
        if batch_dict:  # not empty
            feature_emb = self.embedding_layer(batch_dict, flatten_emb=True)
            emb_list.append(feature_emb)
        feat_emb = torch.cat(emb_list, dim=-1)
        item_feat_emb = self.embedding_layer(item_dict, flatten_emb=True)
        batch_size = mask.shape[0]
        item_feat_emb = item_feat_emb.view(batch_size, -1, self.item_info_dim)

        target_emb = item_feat_emb[:, -1, :]
        sequence_emb = item_feat_emb[:, 0:-1, :]
        transformer_emb = self.transformer_encoder(
            target_emb, sequence_emb, mask=mask
        )
        
        concat_feature = torch.cat([feat_emb, target_emb, transformer_emb], dim=-1)
        # 特征选择
        if self.fs_module is not None:
            stream1_feature = self.fs_module(concat_feature, self.fs1_context)
            stream2_feature = self.fs_module(concat_feature, self.fs2_context)
        else:
            stream1_feature = stream2_feature = concat_feature

        # 双流处理
        stream1_out = self.mlp1(stream1_feature)
        stream2_out = self.mlp2(stream2_feature)

        # 双线性融合
        fusion_out = self.bilinear_fusion(stream1_out, stream2_out)
        y_pred = self.output_layer(fusion_out)

        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)

        return_dict = {"y_pred": y_pred}
        
        return return_dict


    def get_inputs(self, inputs, feature_source=None):
        batch_dict, item_dict, mask = inputs
        X_dict = dict()
        for feature, value in batch_dict.items():
            if feature in self.feature_map.labels:
                continue
            feature_spec = self.feature_map.features[feature]
            if feature_spec["type"] == "meta":
                continue
            if feature_source and not_in_whitelist(feature_spec["source"], feature_source):
                continue
            X_dict[feature] = value.to(self.device)
        for item, value in item_dict.items():
            item_dict[item] = value.to(self.device)
        return X_dict, item_dict, mask.to(self.device)

    def concat_embedding(self, field, feature_emb_dict):
        if type(field) == tuple:
            emb_list = [feature_emb_dict[f] for f in field]
            return torch.cat(emb_list, dim=-1)
        else:
            return feature_emb_dict[field]

    def get_labels(self, inputs):
        labels = self.feature_map.labels
        batch_dict = inputs[0]
        y = batch_dict[labels[0]].to(self.device)
        return y.float().view(-1, 1)

    def get_group_id(self, inputs):
        return inputs[0][self.feature_map.group_id]

    def train_step(self, batch_data):
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)
        loss = self.compute_loss(return_dict, y_true)
        loss = loss / self.accumulation_steps
        loss.backward()
        if (self._batch_index + 1) % self.accumulation_steps == 0:
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss


class Transformer(nn.Module):
    def __init__(self,
                 transformer_in_dim,
                 dim_feedforward=64,
                 num_heads=1,
                 dropout=0,
                 transformer_layers=1,
                 first_k_cols=16,
                 concat_max_pool=True):
        super(Transformer, self).__init__()
        self.concat_max_pool = concat_max_pool
        self.first_k_cols = first_k_cols
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_in_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        if self.concat_max_pool:
            self.out_linear = nn.Linear(transformer_in_dim, transformer_in_dim)

    def forward(self, target_emb, sequence_emb, mask=None):
        # concat action sequence emb with target emb
        seq_len = sequence_emb.size(1)
        concat_seq_emb = torch.cat([sequence_emb,
                                    target_emb.unsqueeze(1).expand(-1, seq_len, -1)], dim=-1)
        # get sequence mask (1's are masked)
        key_padding_mask = self.adjust_mask(mask).bool()  # keep the last dim
        tfmr_out = self.transformer_encoder(src=concat_seq_emb,
                                            src_key_padding_mask=key_padding_mask)
        tfmr_out = tfmr_out.masked_fill(
            key_padding_mask.unsqueeze(-1).repeat(1, 1, tfmr_out.shape[-1]), 0.
        )
        # process the transformer output
        output_concat = []
        output_concat.append(tfmr_out[:, -self.first_k_cols:].flatten(start_dim=1))
        if self.concat_max_pool:
            # Apply max pooling to the transformer output
            tfmr_out = tfmr_out.masked_fill(
                key_padding_mask.unsqueeze(-1).repeat(1, 1, tfmr_out.shape[-1]), -1e9
            )
            pooled_out = self.out_linear(tfmr_out.max(dim=1).values)
            output_concat.append(pooled_out)
        return torch.cat(output_concat, dim=-1)

    def adjust_mask(self, mask):
        # make sure not all actions in the sequence are masked
        fully_masked = mask.all(dim=-1)
        mask[fully_masked, -1] = 0
        return mask

    
class FeatureSelectionLayer(nn.Module):
    def __init__(self, input_dim, hidden_units):
        super(FeatureSelectionLayer, self).__init__()
        self.mlp = MLP_Block(input_dim=input_dim,
                             output_dim=input_dim,
                             hidden_units=hidden_units,
                             hidden_activations="ReLU",
                             output_activation="Sigmoid")
        
    def forward(self, inputs, context_indices=None):
        # 如果有上下文信息，使用上下文特征
        if context_indices and len(context_indices) > 0:
            context_features = inputs[:, context_indices]
            selection_weights = self.mlp(context_features)
        else:
            selection_weights = self.mlp(inputs)
        
        # 应用选择权重
        selected_features = inputs * selection_weights
        return selected_features
    
    
class BilinearFusion(nn.Module):
    def __init__(self, dim1, dim2, num_heads=1):
        super(BilinearFusion, self).__init__()
        self.num_heads = num_heads
        self.weights = nn.Parameter(torch.Tensor(num_heads, dim1, dim2))
        nn.init.xavier_uniform_(self.weights)
        
    def forward(self, x1, x2):
        # x1: [batch_size, dim1]
        # x2: [batch_size, dim2]
        batch_size = x1.size(0)
        
        # 如果使用多头机制，需要分块处理
        if self.num_heads > 1:
            # 实现多头双线性融合逻辑
            results = []
            for i in range(self.num_heads):
                # [batch_size, 1, dim1] x [batch_size, dim1, dim2] x [batch_size, dim2, 1]
                bilinear_out = torch.bmm(
                    torch.bmm(x1.unsqueeze(1), 
                              self.weights[i].unsqueeze(0).expand(batch_size, -1, -1)),
                    x2.unsqueeze(2)
                ).squeeze(2)  # 确保挤压正确的维度
                results.append(bilinear_out)
            # 沿着第1维拼接结果 [batch_size, num_heads]
            return torch.cat(results, dim=1)
        else:
            # 单头情况下的双线性融合
            bilinear_out = torch.bmm(
                torch.bmm(x1.unsqueeze(1), 
                          self.weights[0].unsqueeze(0).expand(batch_size, -1, -1)),
                x2.unsqueeze(2)
            )
            # 确保返回形状为 [batch_size, 1]
            return bilinear_out.squeeze(2)