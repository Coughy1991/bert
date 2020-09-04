# load TF v1
python create_vocab_and_tokenized_smiles.py --input_file=/home/ljn/lab_jensen/rxn_class/data/stage0/data_token512.json &> create_vocab_and_tokenized_smiles.stdout

# if memory issue occurs, run i=1..N with --dupe_factor=1 --random_seed=12345${i} --output_file=./data/stage1/pretrain${i}.tf_record
# python create_pretraining_data.py --vocab_file=./data/stage0/vocab.txt --input_file=./data/stage0/train.txt --dupe_factor=1 --random_seed=12345${i} --output_file=./data/stage1/pretrain${i}.tf_record  &> create_pretraining_data_${i}.stdout
PRETRAIN_DATA=./data/stage1/pretrain0.tf_record,./data/stage1/pretrain1.tf_record,./data/stage1/pretrain2.tf_record,./data/stage1/pretrain3.tf_record,./data/stage1/pretrain4.tf_record,./data/stage1/pretrain5.tf_record,./data/stage1/pretrain6.tf_record,./data/stage1/pretrain7.tf_record,./data/stage1/pretrain8.tf_record,./data/stage1/pretrain9.tf_record
python create_pretraining_data.py --vocab_file=./data/stage0/vocab.txt --input_file=./data/stage0/train.txt --output_file=$PRETRAIN_DATA &> create_pretraining_data.stdout


#  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
#  --num_train_steps=1000000 \
#  --num_warmup_steps=10000 \
#  --train_batch_size=32 \ # 16 for GPU RAM < 11GB
python run_pretraining.py \
  --input_file=$PRETRAIN_DATA \
  --output_dir=./data/stage2 \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=./bert_config_rxn_class.json \
  --train_batch_size=16 \
  --max_seq_length=512 \
  --max_predictions_per_seq=20 \
  --num_train_steps=2000000 \
  --learning_rate=1e-4


PRETRAIN_CHK=./data/stage2/model.ckpt-2000000
FINETUNE_DIR=./data/stage3
mkdir -p $FINETUNE_DIR

python run_classifier_rxn.py \
  --task_name=Pistachio \
  --do_train=true \
  --do_eval=true \
  --data_dir=./data/stage0 \
  --vocab_file=./data/stage0/vocab.txt \
  --bert_config_file=./bert_config_rxn_class.json \
  --init_checkpoint=$PRETRAIN_CHK \
  --max_seq_length=512 \
  --train_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=5.0 \
  --output_dir=$FINETUNE_DIR

