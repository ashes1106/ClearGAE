from typing import Optional
from itertools import chain
from functools import partial

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gin import GIN
from .gat import GAT
from .gcn import GCN
from .dot_gat import DotGAT
from .non_gcn import Tokenizer
from .loss_func import sce_loss, semi_loss
from graphmae.utils import create_norm, drop_edge

from graphmae.utils import  scale_feats_tensor,extract_indices

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
            eps:float,
            norm_enc:float,
            residual: bool,
            norm: Optional[str],
            tokenizer_type: Optional[str],
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
        self.norm_enc=norm_enc
        self.eps=eps

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
        self.std_expander = nn.Sequential(nn.Linear(num_hidden, num_hidden),
                                          nn.PReLU())

        self.std_expander_token = nn.Sequential(nn.Linear(in_dim, num_hidden),
                                          nn.PReLU())
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)
        self.difference = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.tokenizer_nonpara = Tokenizer(in_dim, num_layers, self.eps, JK='last', gnn_type=tokenizer_type,norm=create_norm(norm))


    
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
    
    def encoding_mask_noise(self, g, x, mask_rate=0.3):
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



    def forward(self, g, x, A):
        # ---- attribute reconstruction ----
        loss,loss_s,loss_test,recon = self.mask_attr_prediction(g, x, A)
        loss_item = {"loss": loss.item()}
        return loss, loss_item,loss_s,loss_test,recon
    
    def mask_attr_prediction(self, g, x, A):

        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)

        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g
                                    
        enc_rep, all_hidden = self.encoder(use_g, use_x, return_hidden=True)


        g_tokens = self.tokenizer_nonpara(g,x).detach()
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)
        #rep2 = self.encoder_to_decoder(enc_rep)

        

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

        x_init = x[mask_nodes]
        recon_new = recon[mask_nodes]
        loss = self.criterion(recon_new, x_init)


    
        edge_idx = extract_indices(g)
        dif_init = self.edge_distribution_high(edge_idx, x, self.tau)
        dif_recon = self.edge_distribution_high(edge_idx, recon, self.tau)
        #edge_idx = edge_idx.to(x.device)
        loss_s = self.difference(dif_recon,dif_init )


        #rank

        #covariance loss
        neighbor_mean = self.neighbor_diff(g,enc_rep)
        enc_rep_mean = enc_rep - enc_rep.mean(dim=0)
        neighbor_mean_mean = neighbor_mean - neighbor_mean.mean(dim=0)
    
     
        cov_x = (enc_rep_mean.T @ enc_rep_mean) / (enc_rep_mean.size(0) - 1)
        #cov_y = (neighbor_mean_mean.T @ neighbor_mean_mean) / (neighbor_mean_mean.size(0) - 1)
        con_xy = (enc_rep_mean.T @ neighbor_mean_mean) / (enc_rep_mean.size(0) - 1)
    
 
        mask = ~torch.eye(cov_x.size(0), dtype=torch.bool, device=cov_x.device)
        loss_x = cov_x[mask].pow(2).mean()
        #loss_y = cov_y[mask].pow(2).mean()
        loss_xy = con_xy.pow(2).mean()
        loss_neig = loss_x  + loss_xy


        loss = loss+ self.differ *loss_s +self.norm_enc *loss_neig
        #loss_neig = 0.0
        #loss_s = 0.0
        #loss = loss+ self.differ *loss_s 

        #loss = loss+ self.norm_enc *loss_neig

        return loss,self.differ*loss_s,self.norm_enc *loss_neig,recon
                        
    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])

    def std_loss(self,z,isenc=True):
        if isenc:
            z = self.std_expander(z)
            #z = z
        else:
            z = self.std_expander_token(z)
        # z = F.normalize(z, dim=1)
        # std_z = torch.sqrt(z.var(dim=0) + 1e-4)
        # std_loss = F.relu(1 - std_z)
        #std_loss = z.mean(dim=0).pow(2).mean()
        std_loss = z

        return std_loss

    def neighbor_diff(self,graph,z):
        new_graph = graph
      
        new_graph.ndata['feat'] = z
      
        new_graph = dgl.remove_self_loop(new_graph)

    
        new_graph.update_all(
            dgl.function.copy_u('feat', 'm'),
            dgl.function.mean('m', 'neigh_mean')
        )

   
        degrees = new_graph.out_degrees().float()
        neighbor_means = new_graph.ndata['neigh_mean']
        neighbor_means = torch.where(
            (degrees == 0).unsqueeze(1),
            torch.zeros_like(neighbor_means),
            neighbor_means
        )
        return neighbor_means

    


    def reduce_loss(self, z_a,z_b):
        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD
        # cross-correlation matrix
        c = mm(z_a_norm.T, z_b_norm) / N # DxD
        # loss
        c_diff = (c - eye(D)).pow(2) # DxD
        # multiply off-diagonal elems of c_diff by lambda
        #off_diagonal(c_diff).mul_(lambda)
        loss = c_diff.sum()
        return loss