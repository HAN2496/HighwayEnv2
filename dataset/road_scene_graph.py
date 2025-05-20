import json
import os
from itertools import product
from typing import Callable, List, Optional, Iterable

import numpy as np
import torch

from torch_geometric.data import (
    Data,
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.utils import remove_self_loops
from ..controller.utils import rotation


class RoadSceneDataset(InMemoryDataset):

    def __init__(
        self,
        root: str,
        episode: Optional[int] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:

        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)

        if episode is None:
            episode = np.random.randint(0, len(self.processed_paths))
        self.load(self.processed_paths[episode])


    @property
    def raw_file_names(self) -> List[str]:
        files = os.listdir(self.root)
        for file in files[::-1]:
            if file.split('.')[-1]!='npy':
                files.pop(-1)

        return files


    @property
    def processed_file_names(self) -> List[str]:
        return [f'processed/episode_{episode}.pt' for episode in range(len(self.raw_file_names))]


    def process(self) -> None:
        import networkx as nx
        from networkx.readwrite import json_graph

        for file, processed_path in zip(self.raw_file_names, self.processed_paths):
            episode = np.load(os.path.join(self.raw_dir, file), allow_pickle=True)

            data_list = []
            for scene in episode:
                data_list.append(self.process_data(scene))
            self.save(data_list, processed_path)


    def extract(self, scene):
        data = Data()
        data.x = torch.from_numpy(np.c_[scene['observation'][:, 1:9], scene['observation'][:, 9:10], scene['observation'][:, 13:14]]).to(torch.float)
        data.y = torch.from_numpy(np.r_[np.reshape([scene['action']], (1, 1)), np.reshape(scene['parameters'][2], (-1, 1))]).to(torch.float)
        row = torch.from_numpy(np.r_[np.zeros(4), np.arange(1, 5)]).to(torch.long)
        col = torch.from_numpy(np.r_[np.arange(1, 5), np.zeros(4)]).to(torch.long)
        data.edge_index = torch.stack([row, col], dim=0)
        return data



class RelativeRoadSceneDataset(RoadSceneDataset):

    def extract(self, scene):
        data = Data()
        R = rotation(scene['observation'][0, 5]).T
        data.x = torch.from_numpy(
            np.c_[
                (scene['observation'][:, 1:3] - scene['observation'][:1, 1:3]) @ R,
                (scene['observation'][:, 3:5] - scene['observation'][:1, 3:5]) @ R,
                scene['observation'][:, 5:6] - scene['observation'][:1, 5:6],
                scene['observation'][:, 6:9], scene['observation'][:, 9:10], scene['observation'][:, 13:14]
            ]
        ).to(torch.float)
        data.y = torch.from_numpy(np.r_[np.reshape([scene['action']], (1, 1)), np.reshape(scene['parameters'][2], (-1, 1))]).to(torch.float)
        row = torch.from_numpy(np.r_[np.zeros(4), np.arange(1, 5)]).to(torch.long)
        col = torch.from_numpy(np.r_[np.arange(1, 5), np.zeros(4)]).to(torch.long)
        data.edge_index = torch.stack([row, col], dim=0)
        return data

    

class HeteroRoadSceneDataset(RoadSceneDataset):

    def extract(self, scene):
        data = HeteroData()
        data['ego_vehicle'].x = torch.from_numpy(scene['observation'][:1, 6:9]).to(torch.float)
        data['ego_vehicle'].y = torch.from_numpy(np.reshape([scene['action']], (1, 1))).to(torch.float)
        data['other_vehicle'].x = torch.from_numpy(scene['observation'][1:, 6:9]).to(torch.float)
        data['other_vehicle'].y = torch.from_numpy(np.reshape(scene['parameters'][2], (-1, 1))).to(torch.float)
        row = torch.from_numpy(np.zeros(4)).to(torch.long)
        col = torch.from_numpy(np.arange(4)).to(torch.long)
        data['ego_vehicle', 'prel', 'other_vehicle'].edge_index = torch.stack([row, col], dim=0)
        data['ego_vehicle', 'prel', 'other_vehicle'].edge_attr = torch.from_numpy(scene['observation'][1:, 1:3] - scene['observation'][:1, 1:3]).to(torch.float)
        data['ego_vehicle', 'vrel', 'other_vehicle'].edge_index = torch.stack([row, col], dim=0)
        data['ego_vehicle', 'vrel', 'other_vehicle'].edge_attr = torch.from_numpy(scene['observation'][1:, 3:5] - scene['observation'][:1, 3:5]).to(torch.float)
        return data