import torch
from main import Processor
from types import SimpleNamespace

# ---- 1. Define args manually (instead of argparse/YAML)
Args = SimpleNamespace(
    model='model.gig.Model',
    model_args={
        'num_class': 60,
        'num_point': 25,
        'num_person': 2,
        'graph': 'graph.ntu_rgb_d.Graph',
        'graph_args': {'labeling_mode': 'spatial'}
    },
    feeder='data_feeders.feeder_ntu.DatasetFeeder',
    train_feeder_args={
        'data_path': 'processed_data/ntu60/NTU60_CS.npz',
        'split': 'train',
        'debug': False
    },
    test_feeder_args={
        'data_path': 'processed_data/ntu60/NTU60_CS.npz',
        'split': 'test',
        'debug': False
    },
    work_dir='./work_dir/debug',
    device=[0],
    phase='train',
    optimizer='SGD',
    base_lr=0.1,
    weight_decay=0.0004,
    num_epoch=1,
    batch_size=2,
    test_batch_size=2,
    num_worker=0,
    show_topk=[1,5],
    nesterov=True,
    warm_up_epoch=0,
    step=[1],
    seed=1,
    print_log=True,
    save_score=False,
    ignore_weights=[],
    weights=None,
    model_saved_name=''
)


# ---- 2. Init Processor directly (no utils.IO needed)
processor = Processor(Args)

# ---- 3. Start training (1 epoch demo)
processor.start()
