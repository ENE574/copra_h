from data import DataRegister
import pytorch_lightning as pl
import diskcache
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch_geometric.loader import DataLoader as GraphLoader
from data.complex_batch_sampler import ComplexBatchSampler

def get_dataset(data_args:dict=None):
    register = DataRegister()
    dataset_cls = register[data_args.dataset_type]
    return dataset_cls

def get_collate(dataset_type):
    if dataset_type == 'sequence_dataset':
        from data.sequence_dataset import CustomSeqCollate
        return CustomSeqCollate
    if dataset_type in ('structure_dataset', 'structure_dataset_online'):
        from data.structure_dataset import CustomStructCollate
        return CustomStructCollate
    if dataset_type == 'structure_dataset_online_mem':
        from data.structure_dataset_online_mem import CustomStructCollateOnlineMem
        return CustomStructCollateOnlineMem
    if dataset_type == 'pri30k_dataset':
        from data.pri30k_dataset import PRI30kStructCollate
        return PRI30kStructCollate
    raise KeyError(f"Unknown dataset_type for collate: {dataset_type}")

class DataModule(pl.LightningDataModule):
    def __init__(self, 
                 df_path='', 
                 col_group='fold_0', 
                 batch_size=32, 
                 num_workers=0, 
                 pin_memory=True, 
                 cache_dir=None, 
                 strategy='separate',
                 dataset_args=None,
                 group_remap=None,
                 **kwargs):
        super().__init__()
        self.df_path = df_path
        self.col_group=col_group
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.pin_memory=pin_memory
        self.cache_dir=cache_dir
        self.strategy=strategy
        self.dataset_args=dataset_args
        self.group_remap = group_remap or {}
        # print("Dataset Args:", dataset_args)
        
    def setup(self, stage=None):
        if self.cache_dir is None:
            cache = None
        else:
            print("Using diskcache at {}.".format(self.cache_dir))
            cache = diskcache.Cache(directory=self.cache_dir, eviction_policy='none')
        
        df = pd.read_csv(self.df_path)
        if self.group_remap:
            col_vals = df[self.col_group].replace(self.group_remap)
            print(f"group_remap applied: {self.group_remap}")
        else:
            col_vals = df[self.col_group]
        df_train = df[col_vals.isin(['train'])]
        df_val = df[col_vals.isin(['val'])]
        df_test = df[col_vals.isin(['test'])]
        dataset_cls = get_dataset(self.dataset_args)
        self.train_dataset = dataset_cls(df_train, **self.dataset_args, diskcache=cache)
        self.has_val = len(df_val) > 0
        if self.has_val:
            self.val_dataset = dataset_cls(df_val, **self.dataset_args, diskcache=cache)
        else:
            self.val_dataset = None
            print(f"No validation split in {self.col_group}; training on full train set only.")

        print(
            f"Split {self.col_group}: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}"
        )
        if len(df_test) > 0:
            print("Using held-out test split for final evaluation.")
            self.test_dataset = dataset_cls(df_test, **self.dataset_args, diskcache=cache)
        else:
            print(f"Using validation split {self.col_group} for test evaluation.")
            self.test_dataset = dataset_cls(df_val, **self.dataset_args, diskcache=cache)

    def _use_complex_batch_sampler(self) -> bool:
        return bool(getattr(self.dataset_args, "complex_batch_sampler", False))

    def _make_complex_batch_sampler(self, dataset, shuffle: bool):
        return ComplexBatchSampler(
            RandomSampler(dataset) if shuffle else SequentialSampler(dataset),
            batch_size=self.batch_size,
            shuffle=shuffle,
            min_batch_size=int(getattr(self.dataset_args, "complex_batch_min_size", 2)),
            drop_last=bool(getattr(self.dataset_args, "complex_batch_drop_last", False)),
        )

    def train_dataloader(self):
        if self.dataset_args.dataset_type != 'graph_dataset':
            collate = get_collate(self.dataset_args.dataset_type)
            if self._use_complex_batch_sampler():
                return DataLoader(
                    self.train_dataset,
                    batch_sampler=self._make_complex_batch_sampler(self.train_dataset, shuffle=True),
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    persistent_workers=self.num_workers > 0,
                    collate_fn=collate(strategy=self.strategy, dataset_args=self.dataset_args),
                )
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.num_workers > 0,
                collate_fn=collate(strategy=self.strategy, dataset_args=self.dataset_args),
            )
        else:
            return GraphLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.num_workers > 0,
            )

    def val_dataloader(self):
        if not self.has_val or self.val_dataset is None:
            return []
        if self.dataset_args.dataset_type != 'graph_dataset':
            collate = get_collate(self.dataset_args.dataset_type)
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.num_workers > 0,
                collate_fn=collate(strategy=self.strategy, dataset_args=self.dataset_args),
            )
        else:
            return GraphLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.num_workers > 0,
            )


    def test_dataloader(self):
        if self.dataset_args.dataset_type != 'graph_dataset':
            collate = get_collate(self.dataset_args.dataset_type)
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.num_workers > 0,
                collate_fn=collate(strategy=self.strategy, dataset_args=self.dataset_args),
            )
        else:
            return GraphLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.num_workers > 0,
            )
