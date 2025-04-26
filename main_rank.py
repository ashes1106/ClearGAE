import logging
import numpy as np
from tqdm import tqdm
import torch

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
import wandb
import os
import yaml, random
from collections import defaultdict
from pprint import pprint


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def pretrain(model, graph, feat, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)

    epoch_iter = tqdm(range(max_epoch))

    for epoch in epoch_iter:
        model.train()

        loss, loss_dict = model(graph, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            
        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")

        if  epoch % 100 == 0:
            final_acc, estp_acc = node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, mute=True)
            logging.info("Epoch {:05d} |  final_acc {:.4f}|  estp_acc {:.4f}|  Loss {:.4f}"
                         .format(epoch, final_acc, estp_acc, loss.item()))
            
    # return best_model
    return model


def main():
    
    args = build_args()
    #if args.use_cfg:
    args = load_best_configs(args, "/nlp_group/chenge03/graph/graphMoe/GraphMAE_rank/configs.yml")

    assert args.device in range(0, 8)
    torch.cuda.set_device(args.device)
    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if not args.debug:
        wandb.init()
        wandb.config.update(args)

        args.max_epoch = wandb.config.max_epoch
        args.num_hidden = wandb.config.num_hidden
        args.num_layers = wandb.config.num_layers
        args.encoder = wandb.config.encoder
        args.decoder = wandb.config.decoder
        args.optimizer = wandb.config.optimizer
        args.lr = wandb.config.lr
 

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
    use_scheduler = args.scheduler

    graph, (num_features, num_classes) = load_dataset(dataset_name)
    args.num_features = num_features

    acc_list = []
    estp_acc_list = []
    
    
    for _ in range(2):                
        model = build_model(args)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
            # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                    # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None
            
        x = graph.ndata["feat"]        
        # noise = torch.randn(x.shape)
        # noise = noise.to(x.device)
        # x = x + noise * 0.01                   
        start_time = time.time()    
        if not load_model:
            model = pretrain(model, graph, x, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob)
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

        final_acc, estp_acc = node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob)
        acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)


    message = defaultdict(float)
    message['final_acc'] = final_acc
    message['final_acc_std'] = final_acc_std

    message['estp_acc'] = estp_acc
    message['estp_acc_std'] = estp_acc_std

    
    print(message)
    if not args.debug:
        wandb.log(message)


    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")


# Press the green button in the gutter to run the script.
# if __name__ == "__main__":
#     args = build_args()
#     if args.use_cfg:
#         args = load_best_configs(args, "configs.yml")
#     print(args)
#     main(args)



if __name__ == "__main__":
    args = build_args()
    pprint(args)
    if args.debug:        
        print(args)
        main()        
    else:
        wandb.login()
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
        #sweep_id = wandb.sweep(sweep_config,entity="ellepluto", project=f"gmaerank_{args.dataset}")
        #wandb.agent(sweep_id, main)
