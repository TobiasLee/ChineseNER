model=$1
case $model in
  bert-base)
    mkdir bert-wwm-ext
    cd bert-wwm-ext
    wget https://s3.amazonaws.com/models.huggingface.co/bert/hfl/chinese-bert-wwm-ext/config.json
    wget https://cdn.huggingface.co/hfl/chinese-bert-wwm-ext/pytorch_model.bin
    wget https://cdn.huggingface.co/hfl/chinese-bert-wwm-ext/special_tokens_map.json
    wget https://cdn.huggingface.co/hfl/chinese-bert-wwm-ext/tokenizer_config.json
    wget https://cdn.huggingface.co/hfl/chinese-bert-wwm-ext/vocab.txt
    wget https://cdn.huggingface.co/hfl/chinese-bert-wwm-ext/added_tokens.json
    ;;
  roberta-base)
    mkdir roberta-wwm-ext
    cd roberta-wwm-ext
    wget https://s3.amazonaws.com/models.huggingface.co/bert/hfl/chinese-roberta-wwm-ext/config.json
    wget https://cdn.huggingface.co/hfl/chinese-roberta-wwm-ext/pytorch_model.bin
    wget https://cdn.huggingface.co/hfl/chinese-roberta-wwm-ext/special_tokens_map.json
    wget https://cdn.huggingface.co/hfl/chinese-roberta-wwm-ext/tokenizer_config.json
    wget https://cdn.huggingface.co/hfl/chinese-roberta-wwm-ext/vocab.txt
    wget https://cdn.huggingface.co/hfl/chinese-roberta-wwm-ext/added_tokens.json
    ;;
  roberta-large)
    mkdir roberta-wwm-ext-large
    cd roberta-wwm-ext-large
    wget https://s3.amazonaws.com/models.huggingface.co/bert/hfl/chinese-roberta-wwm-ext-large/config.json
    wget https://cdn.huggingface.co/hfl/chinese-roberta-wwm-ext-large/pytorch_model.bin
    wget https://cdn.huggingface.co/hfl/chinese-roberta-wwm-ext-large/special_tokens_map.json
    wget https://cdn.huggingface.co/hfl/chinese-roberta-wwm-ext-large/tokenizer_config.json
    wget https://cdn.huggingface.co/hfl/chinese-roberta-wwm-ext-large/vocab.txt
    wget https://cdn.huggingface.co/hfl/chinese-roberta-wwm-ext-large/added_tokens.json
    ;;
  *)
    echo "error"
esac