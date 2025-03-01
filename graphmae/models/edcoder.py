from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gin import GIN
from .gat import GAT
from .gcn import GCN
from .dot_gat import DotGAT
from .loss_func import sce_loss, semi_loss
from graphmae.utils import create_norm, drop_edge

from graphmae.utils import  calculate_tensor, scale_feats_tensor,extract_indices

#CUDA_LAUNCH_BLOCKING=1


def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "dotgat":
        mod = DotGAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim, 
            num_hidden=num_hidden, 
            out_dim=out_dim, 
            num_layers=num_layers, 
            dropout=dropout, 
            activation=activation, 
            residual=residual, 
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "mlp":
        # * just for decoder 
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError
    
    return mod


class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            tau: float,
            differ:float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            loss_fn: str = "sce",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
            
         ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate

        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden
        
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        self.act_fn = nn.ReLU()
        self.tau = tau
        self.differ = differ

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden 

        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        # build decoder for attribute prediction
        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            #out_dim=enc_num_hidden,
            num_layers=1,       
            nhead=nhead,
            nhead_out=nhead_out,
            #nhead_out=enc_nhead,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )
        
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)
        self.difference = nn.KLDivLoss(reduction="batchmean", log_target=True)
    
    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion
    

    def edge_distribution_high(self, edge_idx, feats, tau):

        src = edge_idx[1][0]
        src = src.to(feats.device)
        dst = edge_idx[1][1]
        dst = dst.to(feats.device)

        feats_abs = torch.abs( feats[src]-feats[dst])
        e_softmax = F.log_softmax(feats_abs / tau, dim=-1)

        return e_softmax
    
    def encoding_mask_noise(self, g, x,recon_infor_low, recon_infor_high, mask_rate=0.3):
        num_nodes = g.num_nodes()


        # random masking
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        #num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        #mask low
        #perm = torch.randperm(num_nodes, device=x.device)
        #num_mask_nodes = int(mask_rate * num_nodes)

        #mask_nodes_low = perm[: num_mask_nodes]
        #mask_nodes_high = perm[num_mask_nodes: ]

        #mask high
        #perm_2 = torch.randperm(num_nodes, device=x.device)
        #num_mask_nodes = int(mask_rate * num_nodes)

        #mask_nodes_high = perm_2[: num_mask_nodes]
        #mask_nodes_high = perm[num_mask_nodes: ]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0
            #out_x_low = recon_infor_low.clone()
            #out_x_high = recon_infor_high.clone()
            #token_nodes_low = mask_nodes_low
            #token_nodes_high = mask_nodes_high
            #out_x_low[token_nodes_low] = 0.0
            #out_x_high[token_nodes_high] = 0.0
            #out_x = out_x_low + out_x_high 
            #out_x[mask_nodes] = 0.0

                                    
        #out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()
        
        #return use_g, out_x, (mask_nodes_low, mask_nodes_high)
        return use_g, out_x, (mask_nodes, keep_nodes)



    def forward(self, g, x,recon_infor_low, recon_infor_high , A):
        # ---- attribute reconstruction ----
        loss = self.mask_attr_prediction(g, x,recon_infor_low, recon_infor_high, A)
        loss_item = {"loss": loss.item()}
        return loss, loss_item
    
    def mask_attr_prediction(self, g, x,recon_infor_low, recon_infor_high, A):

        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x,recon_infor_low, recon_infor_high, self._mask_rate)

        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g
                                    
        enc_rep, all_hidden = self.encoder(use_g, use_x, return_hidden=True)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)
        #rep2 = self.encoder_to_decoder(enc_rep)

        # 首先我们将两个tensor连接在一起
        #concatenated = torch.cat((mask_nodes_low, mask_nodes_high))

        # 然后我们移除重复的元素
        #union = torch.unique(concatenated)
        

        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0
            #rep[union] = 0

        if self._decoder_type in ("mlp", "liear") :
            #recon1 = self.decoder(rep)
            #recon2 = self.decoder(rep)
            recon = self.decoder(rep)
        else:
            #recon1 = self.decoder(pre_use_g, rep)
            #recon2 = self.decoder(pre_use_g, rep)
            recon = self.decoder(pre_use_g, rep)
        
        #reconstruct low and high (different mask ratio)
        #x_init = x[mask_nodes]
        # x_init_low = recon_infor_low[mask_nodes_low]
        # x_init_high = recon_infor_high[mask_nodes_high]
        # x_rec1 = recon1[mask_nodes_low]
        # x_rec2 = recon2[mask_nodes_high]


        # loss1 = self.criterion(x_rec1, x_init_low)
        # loss2 = self.criterion(x_rec2, x_init_high)
        # loss = loss1 + loss2
        # loss = loss1

        #x_init =  recon_infor_low + recon_infor_high
        #x_init = x_init[mask_nodes]
        #recon = recon[mask_nodes]
        #x_rec = 0.5* recon1 + 0.5* recon2
        x_init = x[mask_nodes]
        recon_new = recon[mask_nodes]
        loss = self.criterion(recon_new, x_init)

        #
        #recon_num = recon.cpu().numpy()
        
        #reconstruct low + reconstruct high + reconstruct low+high
        #loss1_low = semi_loss(x_init_low, x_rec1)
        #loss2_high = semi_loss(x_init_high, x_rec2)
       
        #loss12_f = semi_loss(torch.cat((recon_infor_low,recon_infor_high), 1), torch.cat((recon1,recon2), 1))
        #loss12_f = semi_loss(x_init, x_rec)
        #loss_e = loss1_low + loss2_high + loss12_f
    
        #info_loss = 0.5 *loss1_low.mean() + 0.5 *loss2_high.mean()  

        #node - graph
        # x_rec1 = recon1[mask_nodes_low]
        # x_rec2 = recon2[mask_nodes_high]
        # c = self.act_fn(torch.mean(recon_infor_low[mask_nodes_low], dim=0))
        # c_x = c.expand_as(x_rec1).contiguous()

        # c2 = self.act_fn(torch.mean(recon_infor_high[mask_nodes_high], dim=0))
        # c2_x = c.expand_as(x_rec2).contiguous()

        # loss1 = self.criterion(x_rec1, c_x)
        # loss2 = self.criterion(x_rec2, c2_x)
        # loss = loss1 + loss2
        # loss = loss1

        #self.encoder 
        #enc_rep_all, all_hidden_all = self.encoder(g, x, return_hidden=True)

        #x_init_low = enc_rep_all[mask_nodes_low]
        #x_init_high = recon_infor_high[mask_nodes_high]
        #x_rec1 = recon1[mask_nodes_low]
        #x_rec2 = recon2[mask_nodes_high]


        #loss1 = self.criterion(x_rec1, x_init_low)
        #loss2 = self.criterion(x_rec2, x_init_high)
        ##loss = loss1 + loss2
        #loss = loss1

        #先试一试原始重构
        #x_init = x[concatenated]
        #x_rec = recon[concatenated]
        #loss = self.criterion(x_rec, x_init)

        #x_init_all = recon_infor_low * 0.5 + recon_infor_high * 0.5
        #x_init = x_init_all[mask_nodes_low]
        #x_rec = recon[mask_nodes_low]
        #loss = self.criterion(x_rec, x_init)


        #增加一个loss计算邻居pair对之间的距离
        edge_idx = extract_indices(g)
        dif_init = self.edge_distribution_high(edge_idx, x, self.tau)
        dif_recon = self.edge_distribution_high(edge_idx, recon, self.tau)
        #edge_idx = edge_idx.to(x.device)
        loss_s = self.difference(dif_recon,dif_init )
        loss = loss+ self.differ *loss_s

        return loss
                        
    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
