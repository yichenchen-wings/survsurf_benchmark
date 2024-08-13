from typing import Optional
from torch import nn
import torch
from collections import OrderedDict


class FCNet(nn.Module):
    def __init__(
            self,
            input_dim: int, 
            num_layers: int, 
            h_dim: int, 
            o_dim: int, 
            h_fn_cls: Optional[nn.Module] = None, 
            o_fn_cls: Optional[nn.Module] = None, 
            dropout=0, 
        ):
        """
            GOAL             : Create FC network with different specifications 
            input_dim (int)  : number of input features
            num_layers       : number of layers in FCNet
            h_dim  (int)     : number of hidden units
            o_dim            : output size
            h_fn             : activation function for hidden layers (default: tf.nn.relu)
            o_fn             : activation function for output layers (defalut: None)
            keep_prob        : keep probabilty [0, 1]  (if None, dropout is not employed)
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.h_dim = h_dim

        if h_fn_cls is None:
            self.h_fn_cls = nn.ReLU
        else:
            self.h_fn_cls = h_fn_cls
    
        self.o_dim = o_dim

        if o_fn_cls is None:
            self.o_fn_cls = nn.Identity
        else:
            self.o_fn_cls = o_fn_cls
        self.dropout = dropout

        in_shapes = [self.input_dim] + [self.h_dim]*(self.num_layers-1) 
        assert len(in_shapes) == self.num_layers

        out_shapes = [self.h_dim]*(self.num_layers-1) + [self.o_dim]
        assert len(out_shapes) == self.num_layers

        dropouts = [self.dropout]*(self.num_layers-1) + [0.]
        assert len(dropouts) == self.num_layers

        act_classes = [self.h_fn_cls]*(self.num_layers-1) + [self.o_fn_cls]
        assert len(act_classes) == self.num_layers

        self.layers = nn.ModuleList()
        for in_shape, out_shape, p, act_cls in zip(
            in_shapes, 
            out_shapes, 
            dropouts,
            act_classes
        ):
            fc = nn.Linear(
                in_features=in_shape, 
                out_features=out_shape, 
                bias=True
            )
            act = act_cls()
            drop = nn.Dropout(p=p)
            self.layers.append(fc)
            self.layers.append(act)
            self.layers.append(drop)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
                        
class DeepHit(nn.Module):
    def __init__(
            self,
            n_input_feats,
            k_compete_events,
            t_size,
            h_dim_shared,
            h_dim_cause_spcfc,
            n_layers_shared,
            n_layers_cause_spcfc,
            act_cls,
            dropout=0.
    ):
        super().__init__()
        self.n_input_feats = n_input_feats
        self.k_compete_events = k_compete_events
        self.t_size = t_size
        self.h_dim_shared = h_dim_shared
        self.h_dim_cause_spcfc = h_dim_cause_spcfc
        self.n_layers_shared = n_layers_shared
        self.n_layers_cause_spcfc = n_layers_cause_spcfc
        self.act_cls = act_cls
        self.dropout = dropout

        self.block_shared_base = FCNet(
            input_dim=self.n_input_feats, 
            num_layers=self.n_layers_shared, 
            h_dim=self.h_dim_shared, 
            o_dim=self.h_dim_shared, 
            h_fn_cls=self.act_cls,
            o_fn_cls=self.act_cls,
            dropout=self.dropout
        )

        h_dim_shared_plus = h_dim_shared + n_input_feats
        blocks_cause_spcfc_hidden = OrderedDict()
        for i in range(self.k_compete_events):
            blocks_cause_spcfc_hidden[f'hidden_block_event_{i}'] = FCNet(
                input_dim=h_dim_shared_plus, 
                num_layers=self.n_layers_cause_spcfc, 
                h_dim=self.h_dim_cause_spcfc, 
                o_dim=self.h_dim_cause_spcfc, 
                h_fn_cls=self.act_cls,
                o_fn_cls=self.act_cls,
                dropout=self.dropout
            )
        self.blocks_cause_spcfc_hidden = nn.ModuleDict(blocks_cause_spcfc_hidden)
        self.dropout_cause_spcfc = nn.Dropout(p=dropout)

        head_linear = nn.Linear(
            in_features=self.k_compete_events*self.h_dim_cause_spcfc, 
            out_features=self.k_compete_events*self.t_size, 
            bias=True
        )
        head_act = nn.Softmax(dim=1)
        self.head = nn.ModuleList([head_linear, head_act])

    def _apply_block_shared_base(self, x):
        out = self.block_shared_base(x)
        out = torch.cat([x, out], dim=-1) # -> (batch_size, expanded_feat_len) 
        return out
    
    def _apply_blocks_cause_spcfc(self, x):
        bs, n_feats = x.shape
        out = []
        for k, block in self.blocks_cause_spcfc_hidden.items():
            out.append(block(x))
        out = torch.stack(out, dim=1) # -> (bs, n_events,n_out_feats)
        n_out_feats = out.shape[-1]
        out = out.reshape(bs, self.k_compete_events * n_out_feats)
        out = self.dropout_cause_spcfc(out)
        return out
    
    def _apply_head(self, x):
        for layer in self.head:
            x = layer(x)
        return x

    
    def forward(self, x):
        bs, n_feats = x.shape
        out = self._apply_block_shared_base(x)
        out = self._apply_blocks_cause_spcfc(out)
        out = self._apply_head(out)
        out = out.reshape(bs, self.k_compete_events, self.t_size)
        out = out/out.sum(dim=-1).reshape(bs, self.k_compete_events, 1)
        return out
    
    def forward_cif(self, x):
        f = self.forward(x)
        out = torch.cumsum(f, dim=-1)
        return out




        