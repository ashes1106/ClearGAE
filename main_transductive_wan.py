import logging
import numpy as np
from tqdm import tqdm
import torch
import yaml, random

import os
from collections import defaultdict

from graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from graphmae.datasets.data_util import load_dataset
from graphmae.evaluation import node_classification_evaluation
from graphmae.models import build_model
import time
import networkx as nx
from graphmae.utils import get_similarity_neigborhood, get_distance, plot_epoch, get_similarity_difference, calculate
from graphmae.datasets.data_util import scale_feats
from pprint import pprint

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def pretrain(model, graph, feat, A, nei_simi, recon_infor_low, recon_infor_high, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)
    recon_infor_low = recon_infor_low.to(device)
    recon_infor_high = recon_infor_high.to(device)
    # low concat high
    epoch_iter = tqdm(range(max_epoch))

    # train_neig_list = []
    # neigborhood_simi_list = []
    # neigborhood_diff_list = []

    for epoch in epoch_iter:
        model.train()

        loss, loss_dict = model(graph, x, recon_infor_low, recon_infor_high,A)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(
            f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

        # with torch.no_grad():
        #     emb_epoch = model.embed(graph.to(device), x.to(device))
        #     train_neig_simi = get_similarity_neigborhood(emb_epoch,A)

        # dis_com = get_distance(nei_simi,train_neig_simi)
        # #compute the mean similarity between the neigborhood
        # epoch_train_neig = np.mean(train_neig_simi)
        # train_neig_list.append(epoch_train_neig)
        # neigborhood_simi_list.append(dis_com)

        # neigborhood_diff = get_similarity_difference(feat,emb_epoch,A)
        # neigborhood_diff_list.append(neigborhood_diff)

        # if (epoch + 1) % 1 == 0:
        #     node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, mute=True)

    # plot
    #save_neig_name = '/mmu_nlp_ssd/chenge03/graph/KR/bridge_map/imgs/' + \
     #   'neig_'+str(i)+'.png'
    #save_dis_name = '/mmu_nlp_ssd/chenge03/graph/KR/bridge_map/imgs/' + \
     #   'dis_'+str(i)+'.png'
    #save_rank_name = '/mmu_nlp_ssd/chenge03/graph/KR/bridge_map/imgs/' + \
      #  'rank_'+str(i)+'.png'
    # plot_epoch(train_neig_list,save_neig_name)
    # plot_epoch(neigborhood_simi_list,save_dis_name)
    # plot_epoch(neigborhood_diff_list,save_rank_name)
    # return best_model
    return model


def main():

        
    args = build_args()
    #if args.use_cfg:
    args = load_best_configs(args, "./configs.yml")

    assert args.device in range(0, 8)
    torch.cuda.set_device(args.device)
    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    random.seed(args.seed)


    #device = args.device if args.device >= 0 else "cpu"
    #seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate

    optim_type = args.optimizer
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler

    graph, (num_features, num_classes) = load_dataset(dataset_name)
    args.num_features = num_features 
    #
    nx_g = graph.to_networkx()
    A = nx.adjacency_matrix(nx_g).todense()

    acc_list = []
    estp_acc_list = []

    for _ in range(2):
        #print(f"####### Run {i} for seed {seed}")
        #set_random_seed(seed)

        if logs:
            logger = TBLogger(
                name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        model = build_model(args)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")

            def scheduler(epoch): return (
                1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5
            # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
            # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=scheduler)
        else:
            scheduler = None

        x = graph.ndata["feat"]
        #calculate后mask低或者高
        #重构低信息 重构高信息
        low_x, high_x = calculate(A, x)

        x = x.to(device)
        low_x = torch.tensor(low_x)
        low_x = scale_feats(low_x)
        # x = low_x.to(device)
        high_x = torch.tensor(high_x)
        high_x = scale_feats(high_x)

        recon_infor_low = low_x 
        recon_infor_high = high_x 

        #x = torch.cat((low_x, high_x), dim=1)
        nei_simi = get_similarity_neigborhood(x, A)

        # noise = torch.randn(x.shape)
        # noise = noise.to(x.device)
        # x = x + noise * 0.01
        start_time = time.time()
        if not load_model:
            model = pretrain(model, graph, x, A, nei_simi,recon_infor_low, recon_infor_high, optimizer, max_epoch, device, scheduler,num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob,  logger)
            model = model.cpu()
        end_time = time.time()
        print("耗时: {:.2f}秒".format(end_time - start_time))

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")

        model = model.to(device)
        model.eval()

        #new_x = 0.5 * low_x + 0.5 *high_x
        final_acc, estp_acc = node_classification_evaluation(
            model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob)
        acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)

        if logger is not None:
            logger.finish()

    #final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    #estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    #print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    #print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)




    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")




if __name__ == "__main__":
    args = build_args()
    pprint(args)
    if args.debug:        
        print(args)
        main()        
    else:
        curPath = os.path.dirname(os.path.realpath(__file__))        
        if args.task_type ==  "nc":
            yaml_path = os.path.join(curPath, "sweep_nc.yaml")
        elif args.task_type ==  "lp":
            yaml_path = os.path.join(curPath, "sweep_lp.yaml")
        elif args.task_type ==  "clu":
            yaml_path = os.path.join(curPath, "sweep_cluster.yaml")            
        else:
            yaml_path = os.path.join(curPath, "sweep_gc.yaml")

        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = f.read()
        sweep_config = yaml.load(config, Loader=yaml.FullLoader)
        pprint(sweep_config)

        main()

