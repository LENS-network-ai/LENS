"""Dataset class for graph-based survival prediction (time-to-event + censorship)."""

import os
from typing import Any, Dict, List

import torch
from torch.utils import data


class SurvivalGraphDataset(data.Dataset):
    """
    Loads the same per-slide graph files as utils.dataset.GraphDataset
    (features.pt / adj_s.pt under `root/<graph_name>/`), but reads survival
    labels instead of a class label.

    Args:
        root : path to the directory of per-slide graph folders
        ids  : list of lines, each "graph_name\\ty_disc\\tcensorship\\tsurvival_months"
               (as produced by preprocessing/build_survival_labels.py)
    """

    def __init__(self, root: str, ids: List[str]):
        super().__init__()
        self.root = root.strip()
        self.ids = ids

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        info = self.ids[index].strip()
        try:
            parts = info.split('\t')
            if len(parts) != 4:
                raise ValueError(
                    f"Invalid survival id format: {info!r}. "
                    f"Expected 'graph_name\\ty_disc\\tcensorship\\tsurvival_months'"
                )

            graph_name, y_disc, censorship, survival_months = parts

            feature_path = os.path.join(self.root, graph_name, 'features.pt')
            if not os.path.exists(feature_path):
                raise FileNotFoundError(f'features.pt for {graph_name} not found at {feature_path}')
            features = torch.load(feature_path, map_location='cpu')

            adj_s_path = os.path.join(self.root, graph_name, 'adj_s.pt')
            if not os.path.exists(adj_s_path):
                raise FileNotFoundError(f'adj_s.pt for {graph_name} not found at {adj_s_path}')
            adj_s = torch.load(adj_s_path, map_location='cpu')
            if adj_s.is_sparse:
                adj_s = adj_s.to_dense()

            return {
                'id': graph_name,
                'image': features,
                'adj_s': adj_s,
                'y_disc': int(y_disc),
                'censorship': int(float(censorship)),
                'survival_months': float(survival_months),
            }
        except Exception as e:
            print(f"Error processing {info}: {str(e)}")
            raise


def collate_survival(batch):
    """Collate function for SurvivalGraphDataset."""
    return {
        'image': [b['image'] for b in batch],
        'adj_s': [b['adj_s'] for b in batch],
        'id': [b['id'] for b in batch],
        'y_disc': [b['y_disc'] for b in batch],
        'censorship': [b['censorship'] for b in batch],
        'survival_months': [b['survival_months'] for b in batch],
    }


def prepare_survival_batch(batch_graph, batch_adjs, batch_y_disc, batch_censorship,
                            batch_survival_months, n_features: int = 512, device=None):
    """
    Pad a variable-size list of graphs into batched tensors, mirroring
    helper.preparefeatureLabel but for survival targets.

    Returns:
        node_feat : [B, max_N, n_features]
        adjs      : [B, max_N, max_N]
        masks     : [B, max_N]
        y_disc    : [B]  (LongTensor)
        censorship: [B]  (FloatTensor)
        survival_months: [B]  (FloatTensor)
    """
    batch_size = len(batch_graph)
    max_node_num = max(g.shape[0] for g in batch_graph)

    masks = torch.zeros(batch_size, max_node_num)
    adjs = torch.zeros(batch_size, max_node_num, max_node_num)
    node_feat = torch.zeros(batch_size, max_node_num, n_features)

    for i in range(batch_size):
        cur_node_num = batch_graph[i].shape[0]
        node_feat[i, 0:cur_node_num] = batch_graph[i]
        adjs[i, 0:cur_node_num, 0:cur_node_num] = batch_adjs[i]
        masks[i, 0:cur_node_num] = 1

    y_disc = torch.LongTensor(batch_y_disc)
    censorship = torch.FloatTensor(batch_censorship)
    survival_months = torch.FloatTensor(batch_survival_months)

    if device is not None:
        node_feat = node_feat.to(device)
        adjs = adjs.to(device)
        masks = masks.to(device)
        y_disc = y_disc.to(device)
        censorship = censorship.to(device)
        survival_months = survival_months.to(device)

    return node_feat, adjs, masks, y_disc, censorship, survival_months
