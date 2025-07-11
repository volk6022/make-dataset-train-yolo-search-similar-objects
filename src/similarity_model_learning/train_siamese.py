from sklearn.model_selection import train_test_split
import torch.nn as nn

from fastai.learner import Learner
from fastai.vision.all import *

import network_and_dataloader

# import torch.multiprocessing as mp
# mp.set_start_method('spawn')


def binary_accuracy(inp:Tensor, targ:Tensor, thresh:float=0.5, sigmoid:bool=True):
    "Computes accuracy with `targ` when `pred` is probabilities after sigmoid."
    if sigmoid: inp = inp.sigmoid()
    return ((inp>thresh)==targ.bool()).float().mean()

if __name__ == '__main__':
    similarity_json_path = 'similarity_dino_euclidian_train.json'
    verified_collages_dir = 'pipline_test_collage_dino_euclidian_train'
    objects_preds_json_path = 'predictions_with_embeddings.json'
    formated_dir = 'pipline_test_formated'
    count_verified_labels = 500
    batch_size=128

    ds = network_and_dataloader.SiamesePairedDataset_transform(max_verified_labels=count_verified_labels, model_name='dino')
    ds.load_pairs_from_paths(
        similarity_json_path=similarity_json_path,
        verified_collages_dir=verified_collages_dir,
        objects_preds_json_path=objects_preds_json_path,
        formated_dir=formated_dir
    )
    ds.make_transforms()
    print(ds.num_ftrs)
    train_idx, val_idx = train_test_split(range(len(ds)), test_size=0.2, random_state=0)

    train_pair_obj_names, train_siamese_paire_label = [], []
    train_obj_embeddings = ds.obj_embeddings
    for i in train_idx:
        train_siamese_paire_label.append(ds.siamese_paire_label[i])
        train_pair_obj_names.append(ds.pair_obj_names[i])
    train_ds = network_and_dataloader.SiamesePairedDataset_transform(load_embedding_model=False)
    train_ds._set_data(train_obj_embeddings, train_pair_obj_names, train_siamese_paire_label)

    val_pair_obj_names, val_siamese_paire_label = [], []
    val_obj_embeddings = ds.obj_embeddings
    for i in val_idx:
        val_siamese_paire_label.append(ds.siamese_paire_label[i])
        val_pair_obj_names.append(ds.pair_obj_names[i])
    val_ds = network_and_dataloader.SiamesePairedDataset_transform(load_embedding_model=False)
    val_ds._set_data(val_obj_embeddings, val_pair_obj_names, val_siamese_paire_label)

    del ds.embedding_model, ds.image_processor

    model = network_and_dataloader.SiameseNetwork(ds.num_ftrs)
    loaders = DataLoaders.from_dsets(
        train_ds, 
        val_ds, 
        bs=batch_size,
        valid_bs=batch_size,
        device=ds.device,
        num_workers=0
    )
    criterion = nn.MSELoss()
    learn = Learner(
        loaders, 
        model, 
        loss_func=criterion, 
        path='C:/Users/bhunp/python312/Scripts/local_projects/search_for_not_unique_kerns',
        model_dir='models'
    )

    # # learn.lr_find()
    learn.fit_one_cycle(20, slice(1e-5,1e-3))

    print(learn.get_preds())
    for i in range(val_ds.__len__()):
        (emb_1, emb_2), label = val_ds.__getitem__(i)
        pred_1 = learn.model.predict_single_embedding(emb_1)
        pred_2 = learn.model.predict_single_embedding(emb_2)
        similarity_score = learn.model.calc_two_embeddings_sim(pred_1, pred_2)
        print(similarity_score.item(), label.item())

    learn.save('trained_siamese_500_dino')
    print('learn saved')
    del learn

    model = network_and_dataloader.SiameseNetwork(ds.num_ftrs)
    criterion = nn.MSELoss()
    loaders = DataLoaders.from_dsets(
        network_and_dataloader.SiamesePairedDataset_transform(load_embedding_model=False), 
        network_and_dataloader.SiamesePairedDataset_transform(load_embedding_model=False), 
        bs=batch_size,
        valid_bs=batch_size,
        device=ds.device,
        num_workers=0
    )
    learn = Learner(
        loaders, 
        model, 
        loss_func=criterion, 
        path='C:/Users/bhunp/python312/Scripts/local_projects/search_for_not_unique_kerns', 
        model_dir='models'
    )
    learn.load(file='trained_siamese_500_dino')

    for i in range(val_ds.__len__()):
        (emb_1, emb_2), label = val_ds.__getitem__(i)
        pred_1 = learn.model.predict_single_embedding(emb_1)
        pred_2 = learn.model.predict_single_embedding(emb_2)
        similarity_score = learn.model.calc_two_embeddings_sim(pred_1, pred_2)
        print(similarity_score.item(), label.item())
