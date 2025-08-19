# GenLang — DSL para LLMs (open-core)

[![CI](https://img.shields.io/github/actions/workflow/status/<seu-usuario>/<seu-repo>/ci.yml?branch=main)](https://github.com/<seu-usuario>/<seu-repo>/actions)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE)

**GenLang** é uma DSL minimalista para **treino** e **geração** com modelos da Hugging Face, focada em simplicidade e produtividade.  
Modelo de negócio: **open-core** — recursos básicos abertos, recursos premium (ex.: fine-tuning avançado) opcionais.

## ✨ Recursos

- `MODEL`, `TOKENIZER`, `DATASET CSV`, `PROMPT`
- `TRAIN` (com ou sem **LoRA**)
- `SPLIT` / `SHUFFLE`
- `GENERATE` (amostragem configurável)
- `EVAL perplexity`

## 🚀 Instalação (dev)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# Torch pode variar por SO; em Linux/macOS:
pip install "torch==2.6.0"
```

Requisitos: Python 3.11+. Em macOS Apple Silicon, o backend MPS funciona com Torch 2.6+.

## 🛠️ Estrutura do projeto

genlang/
cli.py # entrypoint CLI (comando: genlang)
runtime.py # parser + intérprete
grammar.lark # gramática Lark
examples/
hello_fast_pt.gs
lora_demo.gs
tiny.csv
out/ # artefatos (gerado em runtime)
pyproject.toml
README.md

## ▶️ Comandos úteis

# Executar um script GenLang

genlang examples/hello_fast_pt.gs

# Rodar um demo com LoRA (se existir lora_demo.gs e 'peft' instalado)

genlang examples/lora_demo.gs
