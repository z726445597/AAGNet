# dump_aagnet_test.py
# S6 推理适配器(AAGNet)——本文件为本方新增脚本, 官方仓库零改动。
# v1.0(2026-08-09): 用官方组件拼装全量测试集逐面 dump:
#   模型结构配置 = 用户训练时 seg_trainer.py 的 wandb.config 逐字照抄(U-stage 改版, 协议 §20);
#   权重 = output\2026_08_05_09_16_44\weight_96-epoch.pth(EMA 权重, 与 best val IoU 0.986438 同轮, §26 发现 12);
#   数据 = 官方 MFCAD2Dataset, 参数与训练同款 center_and_scale=False, normalize=True;
#   唯一刻意偏离 = DataLoader(shuffle=False, drop_last=False) 全量 8922
#   (官方 test_loader 默认 shuffle=True + drop_last=True, 随机丢 218 样本, 见协议 §26 发现 10)。
# 输出: <out_dir>/<stem>.pred 与 <stem>.gt, 一行一个类别索引, 面顺序=数据集图节点顺序。
import argparse
import pathlib

import torch
from torch.utils.data import DataLoader

from dataloader.mfcad2 import MFCAD2Dataset
from models.segmentors import AAGNetSegmentor

# 与用户训练 config 逐字一致(seg_trainer.py wandb.config, U-stage 改版)
CONFIG = dict(
    edge_attr_dim=12,
    node_attr_dim=10,
    edge_attr_emb=64,
    node_attr_emb=64,
    edge_grid_dim=0,
    node_grid_dim=7,
    edge_grid_emb=0,
    node_grid_emb=64,
    num_layers=3,
    delta=2,
    mlp_ratio=2,
    drop=0.25,
    drop_path=0.25,
    head_hidden_dim=64,
    conv_on_edge=False,
    use_uv_gird=True,
    use_edge_attr=True,
    use_face_attr=True,
    architecture="AAGNetGraphEncoder",
)


def main():
    parser = argparse.ArgumentParser(description="S6: dump AAGNet per-face predictions on full test split")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_threads", type=int, default=8)
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[S6] device = {device}")

    n_classes = MFCAD2Dataset.num_classes()
    model = AAGNetSegmentor(
        num_classes=n_classes,
        arch=CONFIG["architecture"],
        edge_attr_dim=CONFIG["edge_attr_dim"],
        node_attr_dim=CONFIG["node_attr_dim"],
        edge_attr_emb=CONFIG["edge_attr_emb"],
        node_attr_emb=CONFIG["node_attr_emb"],
        edge_grid_dim=CONFIG["edge_grid_dim"],
        node_grid_dim=CONFIG["node_grid_dim"],
        edge_grid_emb=CONFIG["edge_grid_emb"],
        node_grid_emb=CONFIG["node_grid_emb"],
        num_layers=CONFIG["num_layers"],
        delta=CONFIG["delta"],
        mlp_ratio=CONFIG["mlp_ratio"],
        drop=CONFIG["drop"],
        drop_path=CONFIG["drop_path"],
        head_hidden_dim=CONFIG["head_hidden_dim"],
        conv_on_edge=CONFIG["conv_on_edge"],
        use_uv_gird=CONFIG["use_uv_gird"],
        use_edge_attr=CONFIG["use_edge_attr"],
        use_face_attr=CONFIG["use_face_attr"],
    )
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print(f"[S6] weights loaded: {args.weights}")

    # 与训练同款数据参数(seg_trainer.py:108-110: center_and_scale=False, normalize=True)
    test_dataset = MFCAD2Dataset(
        root_dir=args.dataset_root,
        split="test",
        center_and_scale=False,
        normalize=True,
        random_rotate=False,
        num_threads=args.num_threads,
    )
    print(f"[S6] test samples loaded = {len(test_dataset)}")

    # 唯一偏离点: 全量评测(不丢尾批、不打乱), collate 仍用官方 _collate(携带 filename)
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=test_dataset._collate,
        drop_last=False,
    )

    n_samples = 0
    n_faces = 0
    with torch.no_grad():
        for data in loader:
            filenames = data["filename"]
            graphs = data["graph"].to(device)
            seg_pred = model(graphs)                    # [total_nodes, num_classes] logits
            preds = torch.argmax(seg_pred, dim=-1)
            labels = graphs.ndata["y"]

            counts = graphs.batch_num_nodes().tolist()
            offset = 0
            for fn, n in zip(filenames, counts):
                p = preds[offset:offset + n].cpu().tolist()
                g = labels[offset:offset + n].cpu().tolist()
                offset += n
                with open(out_dir / f"{fn}.pred", "w") as f:
                    f.write("\n".join(str(int(v)) for v in p) + "\n")
                with open(out_dir / f"{fn}.gt", "w") as f:
                    f.write("\n".join(str(int(v)) for v in g) + "\n")
                n_samples += 1
                n_faces += n

    print(f"[S6] AAGNet dump done: samples={n_samples}, faces={n_faces}, out={out_dir}")


if __name__ == "__main__":
    main()
