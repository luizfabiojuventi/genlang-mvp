MODEL "pierreguillou/gpt2-small-portuguese"
TOKENIZER AUTO

DATASET CSV "examples/tiny.csv" TEXT_COL input

PROMPT "Pergunta:\n{input}\n\nResposta:"

TRAIN epochs=0 batch=2 lr=5e-5 max_length=128

SAVE "out/gpt2-pt-demo"

GENERATE "Pergunta:\nListe 3 dicas para estudar programação.\n\nResposta:" max_new_tokens=80 temperature=0.5 top_p=0.9
