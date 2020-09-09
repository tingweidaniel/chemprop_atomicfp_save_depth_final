from typing import List

import torch
import torch.nn as nn
from tqdm import trange

from chemprop.data import MoleculeDataset, StandardScaler


def predict(model: nn.Module,
            data: MoleculeDataset,
            batch_size: int,
            scaler: StandardScaler = None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param batch_size: Batch size.
    :param scaler: A StandardScaler object fit on the training targets.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    model.eval()

    preds = []
    check_fp = []  # wei, for batch problem
    check_fp_d0 = []  # wei, for batch problem
    check_fp_d1 = []  # wei, for batch problem
    check_fp_d2 = []  # wei, for batch problem
    check_fp_final = []  # wei, for batch problem
    check_fp_mol = []  # wei, for batch problem

    num_iters, iter_step = len(data), batch_size

    for i in trange(0, num_iters, iter_step):
        # Prepare batch
        mol_batch = MoleculeDataset(data[i:i + batch_size])
        smiles_batch, features_batch = mol_batch.smiles(), mol_batch.features()

        # Run model
        batch = smiles_batch

        with torch.no_grad():
            batch_preds = model(batch, features_batch)

        batch_preds = batch_preds.data.cpu().numpy()

        # Inverse scale if regression
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)
        
        # wei, for batch problem
        each_fp = model.output_fp.tolist()
        #print('each_fp:', each_fp)
        remain = num_iters % batch_size
        if len(each_fp)/5 == remain:  # /5 for d0, d1, d2, final, mol
            check_fp_d0.extend(each_fp[:remain])
            check_fp_d1.extend(each_fp[remain:remain*2])
            check_fp_d2.extend(each_fp[remain*2:remain*3])
            check_fp_final.extend(each_fp[remain*3:remain*4])
            check_fp_mol.extend(each_fp[remain*4:remain*5])
        else:
            check_fp_d0.extend(each_fp[:batch_size])
            check_fp_d1.extend(each_fp[batch_size:batch_size*2])
            check_fp_d2.extend(each_fp[batch_size*2:batch_size*3])
            check_fp_final.extend(each_fp[batch_size*3:batch_size*4])
            check_fp_mol.extend(each_fp[batch_size*4:batch_size*5])
    check_fp.append(check_fp_d0)
    check_fp.append(check_fp_d1)
    check_fp.append(check_fp_d2)
    check_fp.append(check_fp_final)
    check_fp.append(check_fp_mol)
    
    #print('check_fp_d0:', check_fp_d0)
    #print('check_fp_d1:', check_fp_d1)
    #print('check_fp_d2:', check_fp_d2)
    #print('check_fp_final:', check_fp_final)
    #print('check_fp_mol:', check_fp_mol)
    #print('preds:', preds)
    #print('check_fp:', check_fp)

    return preds, check_fp
