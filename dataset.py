import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(
        self,
        df,
        cfg,
        aug,
    ):
        super().__init__()
        self.cfg = cfg
        self.aug = aug
        self.df = df
        self.epoch_len = self.df.shape[0]

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        img_path = os.path.join(self.cfg.root_dir, f"{sample.patient_id}_{sample.image_id}.png")
        label = np.expand_dims(np.array(sample.cancer, dtype=np.int8), axis=0)

        data = {
            "image": img_path,
            "prediction_id": sample.prediction_id,
            "label": label,
        }

        return self.aug(data)

    def __len__(self):
        return self.epoch_len
