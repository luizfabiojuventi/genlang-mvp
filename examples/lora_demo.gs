# Exemplo de treino rápido com LoRA em português

MODEL "pierreguillou/gpt2-small-portuguese"
TOKENIZER AUTO
DATASET CSV "examples/tiny.csv" TEXT_COL input

PROMPT "Pergunta:\n{input}\n\nResposta:"

SPLIT train=80% , eval=20%
SHUFFLE on seed=123

# Ativa LoRA
TRAIN epochs=1 batch=2 lr=1e-4 max_length=128 lora=on r=8 alpha=16 target_modules="c_attn,c_proj"

SAVE "out/gpt2-pt-lora-demo"

GENERATE "Pergunta:\nListe 3 dicas para estudar programação.\n\nResposta:" max_new_tokens=100 temperature=0.5 top_p=0.9

EVAL perplexity
