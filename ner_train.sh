OUTPUT_DIR=results/bert-base
BATCH_SIZE=32
NUM_EPOCHS=10
SAVE_STEPS=750
SEED=888
DATA_DIR=cluener_bios
BERT_PATH=bert-wwm-ext
MAX_LENGTH=128
LEARNING_RATE=3e-5 
LOSS_TYPE=CrossEntropyLoss
LOSS_GAMMA=2
LOGGING_STEPS=50
python3 token-classification/run_ner.py \
   --data_dir $DATA_DIR --learning_rate $LEARNING_RATE \
   --labels $DATA_DIR/labels.txt \
   --model_name_or_path $BERT_PATH \
   --output_dir $OUTPUT_DIR \
   --max_seq_length  $MAX_LENGTH \
   --num_train_epochs $NUM_EPOCHS \
   --per_device_train_batch_size $BATCH_SIZE \
   --per_device_eval_batch_size 32 \
   --save_steps $SAVE_STEPS \
   --seed $SEED \
   --loss_type $LOSS_TYPE \
   --loss_gamma $LOSS_GAMMA \
   --do_train \
   --do_eval \
   --evaluate_during_training \
   --logging_steps $LOGGING_STEPS \
   --load_best_model_at_end \
   --metric_for_best_model f1 \
   --logging_dir $OUTPUT_DIR"/runs/"

