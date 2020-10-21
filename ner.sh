
OUTPUT_DIR=cluener-model
BATCH_SIZE=32
NUM_EPOCHS=3
SAVE_STEPS=750
SEED=888
DATA_DIR=cluener_bios
BERT_PATH=bert-wwm-ext
MAX_LENGTH=128
LEARNING_RATE=3e-5 
NUM_EPOCH=3.0
python3 token-classification/run_ner.py \
   --data_dir $DATA_DIR --learning_rate $LEARNING_RATE \
   --labels $DATA_DIR/labels.txt \
   --model_name_or_path $BERT_PATH \
   --output_dir $OUTPUT_DIR \
   --max_seq_length  $MAX_LENGTH \
   --num_train_epochs $NUM_EPOCHS \
   --per_device_train_batch_size $BATCH_SIZE \
   --save_steps $SAVE_STEPS \
   --seed $SEED \
   --do_train \
   --do_eval \
   --do_predict
