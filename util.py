import json
import numpy as np

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

import db

def build_encoder_decoder(class_name):
    '''build encoder and decoder from a list of classes
    Inputs:
        class_name: list
        
    Return:
        encoder: dict, {'class':class_idx}
        decoder: dict, {class_idx: 'class'}
    '''
    encoder = {v: k for k, v in enumerate(class_name)}
    decoder = {v: k for k, v in encoder.items()}
    return encoder, decoder

def load_encoders():
    res = []
    for fn in db.encoder_json:
        with open(fn, 'r') as f:
            res.append(json.load(f))
    return res

def load_decoders():
    res = []
    for fn in db.decoder_json:
        with open(fn, 'r') as f:
            d = json.load(f)
        res.append(d)
    return res

def encode_label_logits(encoder, label):
    res = encoder.get(label, None)
    if res is None:
        res = encoder['null']
    return res

def encode_all_labels_logits(encoders, label):
    '''encode label as logits
        Inputs:
            encoders: list of encoders
        
        Return:
            list of logits
    '''
    label = format_label(label)
    label_split = label.split('.')
    res = np.zeros(shape=len(encoders))
    for i, e in enumerate(encoders):
        if label == 'null':
            res[i] = e['null']
        else:
            _l = '.'.join(label_split[0:i+1])
            res[i] = e.get(_l, e['null'])
    return res

def encode_all_labels_bits(encoders, label, encoders_sizes=None):
    '''encode label as logits
        Inputs:
            encoders: list of encoders
        
        Return:
            list of logits
    '''
    label = format_label(label)
    label_split = label.split('.')
    res = [None]*len(encoders)
    for i, e in enumerate(encoders):
        if encoders_sizes is None:
            s = encoder_size(e)
        else:
            s = encoders_sizes[i]
        res[i] = np.zeros(shape=(s,))
        if label == 'null':
            res[i][e['null']] = 1
        else:
            _l = '.'.join(label_split[0:i+1])
            res[i][e.get(_l, e['null'])] = 1
    return res

def format_label(label):
    if label.split('.')[0] == '0':
        return 'null'
    if label == 'null':
        return 'null'
    if len(label.split('.')) < 3:
        label = '.'.join(label.split('.') + ['0']*(3-len(label.split('.'))))
    return label

def encoder_size(encoder):
    '''Size of encoder
    Note: encoder index is from 0-N, size is N+1
    '''
    res = 0
    for k, v in encoder.items():
        res = max(res, v)
    return res+1

def get_morgan_fingerprint(smiles, radius=2, length=2048, use_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError('get_morgan_fingerprint(): cannot create rdkit molecule, smiles='+smiles)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=length, useChirality=use_chirality))

def get_rxn_features(rxn_smiles, radius=2, length=2048, use_chirality=False, use_reagents=True):
    reactants, reagents, products = rxn_smiles.split(' ')[0].split('>')
    fp_reactants = get_morgan_fingerprint(reactants, radius=radius, length=length, use_chirality=use_chirality)
    fp_products = get_morgan_fingerprint(products, radius=radius, length=length, use_chirality=use_chirality)
    
    if use_reagents:
        if len(reagents) > 0:
            fp_reagents = get_morgan_fingerprint(reagents, radius=radius, length=length, use_chirality=use_chirality)
        else:
            fp_reagents = np.zeros(shape=(length,))
        fp_rxn = np.concatenate((fp_reactants - fp_products, fp_reagents), axis=-1)
    else:
        fp_rxn = fp_reactants - fp_products
    return fp_rxn

def disable_rdkit_warning():
    from rdkit import RDLogger
    logger = RDLogger.logger()
    logger.setLevel(RDLogger.ERROR)

if __name__ == '__main__':
    assert format_label('null') == 'null'
    assert format_label('3.9') == '3.9.0'
    assert format_label('0') == 'null'
    assert format_label('0.0') == 'null'
    assert format_label('0.1') == 'null'
    assert format_label('1.1.1') == '1.1.1'
    
    encoder = {'null': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '0': 0}
    assert encode_label_logits(encoder, 'null') == 0
    assert encode_label_logits(encoder, '1') == 1
    assert encode_label_logits(encoder, 12) == 0
    
    encoders = load_encoders()
    encoders_sizes = [encoder_size(e) for e in encoders]
    encode_all_labels_logits(encoders, '1.1.2').tolist() == [1.0, 1.0, 1.0]
    
    labels1 = encode_all_labels_bits(encoders, '1.1.1', encoders_sizes=encoders_sizes)
    assert labels1[0][1] == 1
    assert labels1[1][1] == 1
    assert labels1[2][1] == 1
    labels = encode_all_labels_bits(encoders, '1.1.1')
    assert labels[0][1] == 1
    assert labels[1][1] == 1
    assert labels[2][1] == 1
    assert np.all(labels1[0] == labels[0])
    assert np.all(labels1[1] == labels[1])
    assert np.all(labels1[2] == labels[2])
    labels = encode_all_labels_bits(encoders, '0')
    assert labels[0][0] == 1
    assert labels[1][0] == 1
    assert labels[2][0] == 1
    labels = encode_all_labels_bits(encoders, 'null')
    assert labels[0][0] == 1
    assert labels[1][0] == 1
    assert labels[2][0] == 1
    labels = encode_all_labels_bits(encoders, '1.1')
    assert labels[0][1] == 1
    assert labels[1][1] == 1
    assert labels[2][0] == 1
