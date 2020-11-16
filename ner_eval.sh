BATCH_SIZE=32
SEED=888
DATA_DIR=cluener_bios/
BERT_PATH=results/bert-base/
MAX_LENGTH=128
LOSS_TYPE=CrossEntropyLoss
python3 token-classification/run_ner.py \
   --data_dir $DATA_DIR \
   --labels $DATA_DIR/labels.txt \
   --model_name_or_path $BERT_PATH \
   --output_dir $BERT_PATH \
   --max_seq_length  $MAX_LENGTH \
   --per_device_eval_batch_size $BATCH_SIZE \
   --seed $SEED \
   --loss_type $LOSS_TYPE \
   --do_predict

