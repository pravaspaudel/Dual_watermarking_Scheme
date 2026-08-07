from transformers import AutoTokenizer,AutoModelForCausalLM

def load_model(model_name:str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    vocabulary_size = len(tokenizer)

    print(f"model and tokenizer of {model_name} loaded with vocab_size {vocabulary_size}")

    return model,tokenizer,vocabulary_size