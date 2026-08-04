# =============================================================================
# u-stage adapt (逐条登记见 U_COMMITS.md):
#   [1] (B 类) config "seed": 42 -> 24                          —— 四管线种子统一 24
#   [2] (B 类) config "epochs": 10 -> 100                       —— 官方注释 100e for MFCAD2
#       (10 为此前 smoke 配置)
#   [3] (B 类) config "dataset" -> Unified roots\aagnet 训练入口 (原指向 MFCAD2_smoke)
#   [4] (C 类, 官方 bug 修复如实披露) train_dataset 构造显式传
#       num_train_data=10**9 —— dataloader/mfcad2.py 文档明写 "-1 = all",
#       但实现 filelist["train"][:num_train_data] 默认 -1 会 [:-1] 丢 train 末行;
#       官方文件零改动, 调用处显式传大数等价 all(协议 §3 发现 8)
#   [5] (B 类) EarlyStopping: val IoU 连续 50 epoch 无改进则停   —— 早停耐心统一 50
# 其余官方逻辑(EMA/优化器/cosine 调度/augmentation=False/存档与测试口径)零改动。
# =============================================================================
import os
import time
from tqdm import tqdm

import torch
from torch import nn
import numpy as np
from torch_ema import ExponentialMovingAverage
from torchmetrics.classification import (
    MulticlassAccuracy, 
    MulticlassJaccardIndex)
import wandb

from dataloader.mfcad import MFCADDataset
from dataloader.mfcad2 import MFCAD2Dataset
from models.segmentors import AAGNetSegmentor
from utils.misc import seed_torch, init_logger, print_num_params



if __name__ == '__main__':
    torch.set_float32_matmul_precision("high") # may be faster if GPU support TF32
    os.environ["WANDB_API_KEY"] = '##################'
    os.environ["WANDB_MODE"] = "offline"
    
    # start a new wandb run to track this script
    dataset_name = "MFCAD2" # option: MFCAD2 MFCAD
    time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    wandb.init(
            # set the wandb project where this run will be logged
            project="aagnet_" + dataset_name,

            # track hyperparameters and run metadata
            config={
                "edge_attr_dim": 12,
                "node_attr_dim": 10,
                "edge_attr_emb": 64, # recommend: 64
                "node_attr_emb": 64, # recommend: 64
                "edge_grid_dim": 0, 
                "node_grid_dim": 7,
                "edge_grid_emb": 0, 
                "node_grid_emb": 64, # recommend: 64
                "num_layers": 3, # recommend: 3
                "delta": 2, # obsolete
                "mlp_ratio": 2,
                "drop": 0.25, 
                "drop_path": 0.25,
                "head_hidden_dim": 64,
                "conv_on_edge": False,
                "use_uv_gird": True,
                "use_edge_attr": True,
                "use_face_attr": True,

                "seed": 24, # u-stage adapt[1]: 42 -> 24 (B 类)
                "device": 'cuda',
                "architecture": "AAGNetGraphEncoder", # recommend: AAGNetGraphEncoder option: GCN SAGE GIN GAT GATv2 DeeperGCN AAGNetGraphEncoder
                "dataset": dataset_name,
                "dataset": "D:/ShortEssay/Datasets/Unified/roots/aagnet", # u-stage adapt[3]: smoke -> 训练 root (B 类)

                "epochs": 100, # u-stage adapt[2]: 10 -> 100, 官方注释 100e for MFCAD2; 350e for MFCAD (B 类)
                "lr": 1e-2,
                "weight_decay": 1e-2,
                "batch_size": 64,
                "ema_decay_per_epoch": 1. / 2.,
                }
        )
    
    print(wandb.config)
    seed_torch(wandb.config['seed'])
    device = wandb.config['device']
    dataset = wandb.config['dataset']
    if dataset_name == "MFCAD":
        Dataset = MFCADDataset
    elif dataset_name == "MFCAD2":
        Dataset = MFCAD2Dataset
    else:
        assert False, "Not supported dataset"
    n_classes = Dataset.num_classes()

    model = AAGNetSegmentor(num_classes=n_classes,
                            arch=wandb.config['architecture'],
                            edge_attr_dim=wandb.config['edge_attr_dim'], 
                            node_attr_dim=wandb.config['node_attr_dim'], 
                            edge_attr_emb=wandb.config['edge_attr_emb'], 
                            node_attr_emb=wandb.config['node_attr_emb'],
                            edge_grid_dim=wandb.config['edge_grid_dim'], 
                            node_grid_dim=wandb.config['node_grid_dim'], 
                            edge_grid_emb=wandb.config['edge_grid_emb'], 
                            node_grid_emb=wandb.config['node_grid_emb'], 
                            num_layers=wandb.config['num_layers'], 
                            delta=wandb.config['delta'], 
                            mlp_ratio=wandb.config['mlp_ratio'], 
                            drop=wandb.config['drop'], 
                            drop_path=wandb.config['drop_path'], 
                            head_hidden_dim=wandb.config['head_hidden_dim'],
                            conv_on_edge=wandb.config['conv_on_edge'],
                            use_uv_gird=wandb.config['use_uv_gird'],
                            use_edge_attr=wandb.config['use_edge_attr'],
                            use_face_attr=wandb.config['use_face_attr'],)
    model = model.to(device)
    total_params = print_num_params(model)
    wandb.config['total_params'] = total_params

    # model_param = torch.load("E:\\AAGNet\\outpout\\weight_38-epoch.pth", map_location=device)
    # model.load_state_dict(model_param)
    

    train_dataset = Dataset(root_dir=dataset, split='train', 
                            center_and_scale=False, normalize=True, random_rotate=False,
                            num_train_data=10**9, # u-stage adapt[4]: C 类修复, 防官方默认 -1 -> [:-1] 丢末行
                            num_threads=8)
    graphs = train_dataset.graphs() # no need to load graphs again !
    val_dataset = Dataset(root_dir=dataset, graphs=graphs, split='val', 
                          center_and_scale=False, normalize=True,
                          num_threads=8)
    train_loader = train_dataset.get_dataloader(batch_size=wandb.config['batch_size'], pin_memory=True)
    val_loader = val_dataset.get_dataloader(batch_size=wandb.config['batch_size'], shuffle=False, drop_last=False, pin_memory=True)

    seg_loss = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=wandb.config['lr'], weight_decay=wandb.config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=wandb.config['epochs'], eta_min=0)

    train_seg_acc = MulticlassAccuracy(num_classes=n_classes).to(device)
    train_seg_iou = MulticlassJaccardIndex(num_classes=n_classes).to(device)

    val_seg_acc = MulticlassAccuracy(num_classes=n_classes).to(device)
    val_seg_iou = MulticlassJaccardIndex(num_classes=n_classes).to(device)

    iters = len(train_loader)
    ema_decay = wandb.config['ema_decay_per_epoch']**(1/iters)
    print(f'EMA decay: {ema_decay}')
    ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)
    
    best_acc = 0.
    bad_epochs = 0 # u-stage adapt[5]: EarlyStopping 状态
    es_patience = 50 # u-stage adapt[5]: 四管线统一 patience=50 (monitor 随官方 val IoU, mode=max)
    save_path = 'output'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, time_str)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    logger = init_logger(os.path.join(save_path, 'log.txt'))
    for epoch in range(wandb.config['epochs']):
        logger.info(f'------------- Now start epoch {epoch}------------- ')
        model.train()
        # train_per_inst_acc = []
        train_losses = []
        train_bar = tqdm(train_loader)
        for data in train_bar:
            graphs = data["graph"].to(device, non_blocking=True)
            seg_label = graphs.ndata["y"]
            
            # Zero the gradients
            opt.zero_grad()
            
            # Forward pass
            seg_pred = model(graphs)

            loss = seg_loss(seg_pred, seg_label)
            train_losses.append(loss.item())

            lr = opt.param_groups[0]["lr"]
            info = "Epoch:%d LR:%f Loss:%f" % (epoch, lr, loss)
            train_bar.set_description(info)

            # Backward pass
            loss.backward()

            opt.step()
            # Update the moving average with the new parameters from the last optimizer step
            ema.update()
            
            train_seg_acc.update(seg_pred, seg_label)
            train_seg_iou.update(seg_pred, seg_label)
        
        scheduler.step()
        # batch end
        mean_train_loss = np.mean(train_losses).item()
        mean_train_seg_acc = train_seg_acc.compute().item()
        mean_train_seg_iou = train_seg_iou.compute().item()
        
        logger.info(f'train_loss : {mean_train_loss}, \
                      train_seg_acc: {mean_train_seg_acc}, \
                      train_seg_iou: {mean_train_seg_iou}'
                   )
        wandb.log({'epoch': epoch, 
                   'train_loss': mean_train_loss, 
                   'train_seg_acc': mean_train_seg_acc, 
                   'train_seg_iou': mean_train_seg_iou
                    })
        train_seg_acc.reset()
        
        # eval
        with torch.no_grad():
            with ema.average_parameters():
                model.eval()
                # val_per_inst_acc = []
                val_losses = []
                for data in tqdm(val_loader):
                    graphs = data["graph"].to(device)
                    seg_label = graphs.ndata["y"]
                    
                    seg_pred = model(graphs)
                                                          
                    loss = seg_loss(seg_pred, seg_label)
                    val_losses.append(loss.item())

                    val_seg_acc.update(seg_pred, seg_label)
                    val_seg_iou.update(seg_pred, seg_label)
                # val end
                mean_val_loss = np.mean(val_losses).item()
                mean_val_seg_acc = val_seg_acc.compute().item()
                mean_val_seg_iou = val_seg_iou.compute().item()
                                                          
                logger.info(f'val_loss : {mean_val_loss}, \
                              val_seg_acc: {mean_val_seg_acc}, \
                              val_seg_iou: {mean_val_seg_iou}' )
                wandb.log({'epoch': epoch, 
                           'val_loss': mean_val_loss, 
                           'val_seg_acc': mean_val_seg_acc, 
                           'val_seg_iou': mean_val_seg_iou
                            })
                
                val_seg_acc.reset()
                val_seg_iou.reset()

                cur_acc = mean_val_seg_iou
                if cur_acc > best_acc:
                    best_acc = cur_acc
                    bad_epochs = 0 # u-stage adapt[5]
                    logger.info(f'best metric: {cur_acc}, model saved')
                    torch.save(model.state_dict(), os.path.join(save_path, "weight_%d-epoch.pth"%(epoch)))
                else:
                    bad_epochs += 1 # u-stage adapt[5]
                    logger.info(f'no improvement on val IoU: {bad_epochs}/{es_patience}')
          # epoch end
        
        # u-stage adapt[5]: EarlyStopping 判定(val IoU 连续 50 epoch 无改进则停)
        if bad_epochs >= es_patience:
            logger.info(f'EarlyStopping triggered at epoch {epoch}: val IoU no improvement for {es_patience} epochs')
            break
        
    # training end test
    graphs = train_dataset.graphs() # no need to load graphs again !
    test_dataset = Dataset(root_dir=dataset, graphs=graphs, split='test', 
                           center_and_scale=False, normalize=True, random_rotate=False,
                           num_threads=8)
    test_loader = test_dataset.get_dataloader(batch_size=wandb.config['batch_size'], pin_memory=True)

    test_seg_acc = MulticlassAccuracy(num_classes=n_classes).to(device)
    test_seg_iou = MulticlassJaccardIndex(num_classes=n_classes).to(device)


    with torch.no_grad():
        logger.info(f'------------- Now start testing ------------- ')
        model.eval()
        # test_per_inst_acc = []
        test_losses = []
        for data in tqdm(test_loader):
            graphs = data["graph"].to(device, non_blocking=True)
            seg_label = graphs.ndata["y"]
            
            # Forward pass
            seg_pred = model(graphs)

            loss = seg_loss(seg_pred, seg_label)

            test_losses.append(loss.item())
            test_seg_acc.update(seg_pred, seg_label)
            test_seg_iou.update(seg_pred, seg_label)
        
        # batch end
        mean_test_loss = np.mean(test_losses).item()
        mean_test_seg_acc = test_seg_acc.compute().item()
        mean_test_seg_iou = test_seg_iou.compute().item()
        
        logger.info(f'test_loss : {mean_test_loss}, \
                      test_seg_acc: {mean_test_seg_acc}, \
                      test_seg_iou: {mean_test_seg_iou}')
        wandb.log({'test_loss': mean_test_loss, 
                   'test_seg_acc': mean_test_seg_acc, 
                   'test_seg_iou': mean_test_seg_iou, 
                    })