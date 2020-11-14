OUTPUT_DIR=results/roberta-large
BATCH_SIZE=8
NUM_EPOCHS=10
SAVE_STEPS=750
SEED=888
DATA_DIR=cluener_bios/
BERT_PATH=roberta-wwm-ext-large
MAX_LENGTH=128
LEARNING_RATE=3e-5 
LOSS_TYPE=CrossEntropyLoss
LOGGING_STEPS=100
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
   --loss_type $LOSS_TYPE \
   --do_train \
   --do_eval \
   --evaluate_during_training \
   --logging_steps $LOGGING_STEPS \
   --load_best_model_at_end \
   --metric_for_best_model f1 \
   --logging_dir $OUTPUT_DIR"/runs/"

