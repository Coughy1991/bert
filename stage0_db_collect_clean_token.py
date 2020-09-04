#!/usr/bin/env python

# nohup ./stage0_db_collect_clean_token.py >stage0_db_collect_clean_token.stdout 2>stage0_db_collect_clean_token.stderr&

import os
import sys
import socket
import time
import json

import rdkit
from rdkit import Chem
from rdkit import rdBase

import pistachio
import db
import util
import tokenizer_chem

def remove_atom_mapping_mol(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

def remove_atom_mapping(rxn):
    res = []
    rxn_split = rxn.split(' ')[0].split('>')
    for i in rxn_split:
        i_non_mapped = ''
        if len(i) > 0:
            mol = Chem.MolFromSmiles(i)
            remove_atom_mapping_mol(mol)
            i_non_mapped = Chem.MolToSmiles(mol)
        res.append(i_non_mapped)
    return '>'.join(res)

def move_reagents_to_reactants(rxn):
    res = []
    reactants, reagents, products = rxn.split(' ')[0].split('>')
    if len(reagents) > 0:
        reactants += '.' + reagents
    return reactants + '>>' + products

print('PID: ', os.getpid())
print('HOSTNAME: ', socket.gethostname())
print('rdkit version: ', rdBase.rdkitVersion)
sys.stdout.flush()

start_time = time.time()

output_dir = db.stage0_dir
if not os.path.isdir(output_dir):
    print('output dir not exist: '+output_dir)
    raise SystemExit

print('output_dir: '+output_dir)

# db_reactions = pistachio.pistachio_pkl(db.pistachio_pkl_path)
db_reactions = pistachio.pistachio_json(db.pistachio_txt_path)

data = []
token_size_limit = 512

added_smiles = set()
cnt = 0
skip_cnt = 0
for r in db_reactions:
    is_skip = False
    if cnt % 100000 == 0:
        sys.stdout.write("cnt={}\n".format(cnt))
        sys.stdout.flush()
        sys.stderr.flush()
    
    rxn_smiles = r['data'].get('smiles', None)
    rxn_class_num = r['data'].get('namerxn', 'null')
    
    # remove duplicate
    if rxn_smiles in added_smiles:
        continue
    else:
        added_smiles.add(rxn_smiles)
    
    try:
        util.get_rxn_features(rxn_smiles)
        rxn_smiles_non_mapped = remove_atom_mapping(rxn_smiles)
    except:
        continue
    
    try:
        token_smi = tokenizer_chem.smi_tokenizer(move_reagents_to_reactants(rxn_smiles_non_mapped))
    except Exception as e:
        print('smi_tokenizer error:')
        print('rxn_smiles: ', rxn_smiles)
        print('rxn_smiles_non_mapped: ', rxn_smiles_non_mapped)
        print('move_reagents_to_reactants: ', move_reagents_to_reactants(rxn_smiles_non_mapped))
        print(e)
        continue
    if len(token_smi.split(' ')) > token_size_limit:
        is_skip = True
    
    d = {
        'id' : cnt,
        'date' : r['data']['date'],
        'rxn_smiles' : rxn_smiles,
        'rxn_smiles_non_mapped' : rxn_smiles_non_mapped,
        'rxn_class_num' : rxn_class_num,
        'filepath' : r.get('filepath'),
        'filelinenum': r.get('filelinenum'),
        'title': r.get('title'),
        }
    
    if not is_skip:
        if cnt%10000 == 0:
            with open('db_collect_debug.json', 'w') as f:
                f.write(json.dumps(r, indent=4))
                f.write('\n\n')
                f.write('d:\n')
                f.write(json.dumps(d, indent=4))
                f.write('\n\n')
                f.write('token_smi:\n')
                f.write(token_smi)
                f.write('\n\n')
        data.append(d)
        cnt += 1
    else:
        skip_cnt += 1

end_time = time.time()
print('time used for db: ', end_time-start_time, 'sec') # 1200
print('skip_cnt=', skip_cnt)

# save data
with open(output_dir + '/data_token{}.json'.format(token_size_limit), 'w') as f:
    json.dump(data, f)

print('total reactions=', cnt)

print()
end_time = time.time()
print('total time used: ', end_time-start_time, 'sec')
