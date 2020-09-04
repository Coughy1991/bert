# coding=utf-8

import json
import tokenizer_chem
import pandas as pd
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Json input file.")

flags.DEFINE_string(
    "train_output_file", "./data/stage0/train.txt",
    "Output raw text file.")

flags.DEFINE_string(
    "val_output_file", "./data/stage0/val.txt",
    "Output raw text file.")

flags.DEFINE_string(
    "test_output_file", "./data/stage0/test.txt",
    "Output raw text file.")

flags.DEFINE_float(
    "train_ratio", 0.8,
    "Split ratio of training data.")

flags.DEFINE_float(
    "val_ratio", 0.1,
    "Split ratio of validation data.")

flags.DEFINE_string("vocab_file", "./data/stage0/vocab.txt",
                    "The output vocabulary file.")

def move_reagents_to_reactants(rxn):
    res = []
    reactants, reagents, products = rxn.split(' ')[0].split('>')
    if len(reagents) > 0:
        reactants += '.' + reagents
    return reactants + '>>' + products

def create_vocab_and_tokenize(data, vocab, output_fn):
    f_out = open(output_fn, 'w')
    for rxn in data:
        # smiles = move_reagents_to_reactants(rxn["rxn_smiles_non_mapped"])
        smiles = rxn[1]
        smiles_tokenized = tokenizer_chem.smi_tokenizer(smiles)
        # f_out.write(rxn["rxn_class_num"] + " " + smiles_tokenized + "\n")
        f_out.write(str(int(rxn[0]))+' '+smiles_tokenized + "\n")
        tokens = smiles_tokenized.split(" ")
        for i in tokens:
            if i not in vocab:
                vocab.append(i)
    f_out.close()

def save_data_as_json(data, fn):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=0)

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    
    tf.logging.info("*** Reading from input files ***")
    tf.logging.info("input_file: {}".format(FLAGS.input_file))
    # with open(FLAGS.input_file, 'r') as f:
    #     input_data = json.load(f)
    df = pd.read_csv(FLAGS.input_file)
    input_data = list(zip(list(df['label']),list(df['unmapped_smiles'])))
    input_data_length = len(input_data)
    train_data_length = int(input_data_length*FLAGS.train_ratio)
    val_data_length = int(input_data_length*FLAGS.val_ratio)
    tf.logging.info("train_ratio: {}".format(FLAGS.train_ratio))
    tf.logging.info("val_ratio: {}".format(FLAGS.val_ratio))
    assert FLAGS.val_ratio + FLAGS.train_ratio <= 1.0
    tf.logging.info("input_data_length: {}".format(input_data_length))
    tf.logging.info("train_data_length: {}".format(train_data_length))
    tf.logging.info("val_data_length: {}".format(val_data_length))
    
    train_data = input_data[0:train_data_length]
    val_data = input_data[train_data_length:train_data_length+val_data_length]
    test_data = input_data[train_data_length+val_data_length:]
    tf.logging.info("len(train_data): {}".format(len(train_data)))
    tf.logging.info("len(val_data): {}".format(len(val_data)))
    tf.logging.info("len(test_data): {}".format(len(test_data)))
    assert len(train_data) == train_data_length
    assert len(val_data) == val_data_length
    assert len(train_data) + len(val_data) + len(test_data) == input_data_length
    
    save_data_as_json(train_data, "./data/stage0/train.json")
    save_data_as_json(val_data, "./data/stage0/val.json")
    save_data_as_json(test_data, "./data/stage0/test.json")
    
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    tf.logging.info("*** Writing to train output files ***")
    create_vocab_and_tokenize(train_data, vocab, FLAGS.train_output_file)
    tf.logging.info("*** Writing to validation output files ***")
    create_vocab_and_tokenize(val_data, vocab, FLAGS.val_output_file)
    tf.logging.info("*** Writing to test output files ***")
    create_vocab_and_tokenize(test_data, vocab, FLAGS.test_output_file)
    
    tf.logging.info("*** Writing to vocab files ***")
    tf.logging.info("number of tokens: {}".format(len(vocab)))
    with open(FLAGS.vocab_file, 'w') as f:
        for i in vocab:
            f.write(i+'\n')

if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    tf.app.run()
