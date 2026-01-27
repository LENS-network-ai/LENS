# utils/dataset.py
"""Dataset class for graph classification task"""

import os
from warnings import warn
from typing import Any, Optional, List, Dict

import torch
from torch.utils import data
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class GraphDataset(data.Dataset):
    """Input and label image dataset"""

    def __init__(self,
                 root: str,
                 ids: List[str],
                 site: Optional[str] = 'LUAD',
                 classdict: Optional[Dict[str, int]] = None,
                 target_patch_size: Optional[int] = None,
                 use_refined_adj: bool = False
                 ) -> None:
        """
        Create a GraphDataset
        
        Args:
            root: Path to dataset root directory
            ids: List of ids in format "filename\tlabel"
            site: Canonical tissue site name ('LUAD', 'LSCC', 'NLST', 'TCGA')
            classdict: Dictionary mapping class names to indices
            target_patch_size: Size of patches to extract (not used)
            use_refined_adj: Whether to use refined adjacency matrices
        """
        super(GraphDataset, self).__init__()
        self.root = root
        self.ids = ids
        self.use_refined_adj = use_refined_adj

        if classdict is not None:
            self.classdict = classdict
        else:
            if site is None:
                warn('Neither site nor classdict provided. Assuming class labels are integers.')
                self.classdict = None
            elif site in {'LUAD', 'LSCC'}:
                self.classdict = {'normal': 0, 'luad': 1, 'lscc': 2}
            elif site == 'NLST':
                self.classdict = {'normal': 0, 'tumor': 1}
            elif site == 'TCGA':
                self.classdict = {'Normal': 0, 'TCGA-LUAD': 1, 'TCGA-LUSC': 2}
            else:
                raise ValueError(f'Site {site} not recognized and classdict not provided')
        self.site = site

    def __getitem__(self, index: int) -> Dict[str, Any]:
        info = self.ids[index].replace('\n', '')
        try:
            # Split by tab or multiple spaces
            parts = info.split('\t') if '\t' in info else info.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid id format: {info}. Expected 'filename\tlabel'")
                
            graph_name = parts[0].strip()
            label = parts[1].strip().lower()
            
            graph_path = self.root.strip()
            
            sample = {}
            sample['label'] = self.classdict[label] if (self.classdict is not None) else int(label)
            sample['id'] = graph_name

            # Construct path for features
            feature_path = os.path.join(graph_path, graph_name, 'features.pt')
            if not os.path.exists(feature_path):
                alt_feature_path = os.path.join(graph_path.rstrip(), graph_name, 'features.pt')
                if os.path.exists(alt_feature_path):
                    feature_path = alt_feature_path
                else:
                    raise FileNotFoundError(f'features.pt for {graph_name} not found at {feature_path}')
            
            features = torch.load(feature_path, map_location='cpu')

            # Load adjacency matrix
            if self.use_refined_adj:
                refined_adj_path = os.path.join(graph_path, graph_name, 'refined_adj.pt')
                if os.path.exists(refined_adj_path):
                    adj_s = torch.load(refined_adj_path, map_location='cpu')
                else:
                    # Fall back to original adjacency
                    adj_s_path = os.path.join(graph_path, graph_name, 'adj_s.pt')
                    if not os.path.exists(adj_s_path):
                        alt_adj_path = os.path.join(graph_path.rstrip(), graph_name, 'adj_s.pt')
                        if os.path.exists(alt_adj_path):
                            adj_s_path = alt_adj_path
                        else:
                            raise FileNotFoundError(f'adj_s.pt for {graph_name} not found at {adj_s_path}')
                    
                    adj_s = torch.load(adj_s_path, map_location='cpu')
            else:
                # Use original adjacency only
                adj_s_path = os.path.join(graph_path, graph_name, 'adj_s.pt')
                if not os.path.exists(adj_s_path):
                    alt_adj_path = os.path.join(graph_path.rstrip(), graph_name, 'adj_s.pt')
                    if os.path.exists(alt_adj_path):
                        adj_s_path = alt_adj_path
                    else:
                        raise FileNotFoundError(f'adj_s.pt for {graph_name} not found at {adj_s_path}')
                
                adj_s = torch.load(adj_s_path, map_location='cpu')
            
            # Ensure dense tensor
            if adj_s.is_sparse:
                adj_s = adj_s.to_dense()

            sample['image'] = features
            sample['adj_s'] = adj_s

            return sample
            
        except Exception as e:
            print(f"Error processing {info}: {str(e)}")
            raise

    def __len__(self):
        return len(self.ids)
