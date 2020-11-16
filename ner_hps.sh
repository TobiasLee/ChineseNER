OUTPUT_DIR=/home/renshuhuai/ChineseNER/results/bert-tune
SEED=888
DATA_DIR=/home/renshuhuai/ChineseNER/cluener_bios/
BERT_PATH=/home/renshuhuai/ChineseNER/bert-wwm-ext
MAX_LENGTH=128
LOSS_TYPE=CrossEntropyLoss
python3 token-classification/tune_ner.py \
   --data_dir $DATA_DIR \
   --labels $DATA_DIR/labels.txt \
   --model_name_or_path $BERT_PATH \
   --output_dir $OUTPUT_DIR \
   --max_seq_length  $MAX_LENGTH \
   --seed $SEED \
   --loss_type $LOSS_TYPE \
   --do_train \
   --do_eval \
   --evaluate_during_training \
   --load_best_model_at_end \
   --metric_for_best_model f1 \

