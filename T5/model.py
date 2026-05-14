add_tokens = ['<', '<=', '<>']

def load_model(model_name, encoder_only=False):
    if not encoder_only:
        from transformers import T5ForConditionalGeneration
        model = T5ForConditionalGeneration
    else:
        from transformers import T5EncoderModel
        model = T5EncoderModel
    return model.from_pretrained(model_name, cache_dir='/nfs_data_storage/huggingface')

def load_tokenizer(model_name):
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir='/nfs_data_storage/huggingface')
    tokenizer.add_tokens(add_tokens)
    return tokenizer
