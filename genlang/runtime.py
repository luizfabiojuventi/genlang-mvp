from __future__ import annotations

import os
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from lark import Lark, Transformer, v_args, Token

GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), "grammar.lark")

# =========================
# AST
# =========================

@dataclass
class Cmd:
    kind: str
    args: Dict[str, Any]


class ASTBuilder(Transformer):
    def start(self, items):
        return items

    def model_stmt(self, items):
        return Cmd("MODEL", {"id": strip_str(items[0])})

    def tokenizer_stmt(self, items):
        # TOKENIZER AUTO | TOKENIZER "hf/tokenizer"
        if not items:
            return Cmd("TOKENIZER", {"id": "AUTO"})
        v = items[0]
        if isinstance(v, Token) and v.type == "STRING":
            return Cmd("TOKENIZER", {"id": strip_str(v)})
        s = strip_str(v)
        if s.upper() == "AUTO":
            s = "AUTO"
        return Cmd("TOKENIZER", {"id": s})

    def dataset_stmt(self, items):
        path = strip_str(items[0])
        text_col = str(items[1])
        return Cmd("DATASET_CSV", {"path": path, "text_col": text_col})

    def prompt_stmt(self, items):
        return Cmd("PROMPT", {"template": strip_str(items[0])})

    def train_stmt(self, items):
        return Cmd("TRAIN", {"params": dict(items)})

    def save_stmt(self, items):
        return Cmd("SAVE", {"dir": strip_str(items[0])})

    def generate_stmt(self, items):
        prompt = strip_str(items[0])
        rest = dict(items[1:]) if len(items) > 1 else {}
        return Cmd("GENERATE", {"text": prompt, "params": rest})

    def set_stmt(self, items):
        name = str(items[0])
        val = items[1]
        return Cmd("SET", {"name": name, "value": val})

    # --- novos statements (SPLIT/SHUFFLE/EVAL) ---
    def split_stmt(self, items):
        # Mantemos parsing simples; resolvemos no runtime
        return Cmd("SPLIT", {"text": " ".join(str(x) for x in items)})

    def shuffle_stmt(self, items):
        vals = [str(x) for x in items]
        return Cmd("SHUFFLE", {"vals": vals})

    def eval_stmt(self, items):
        return Cmd("EVAL", {"kind": "perplexity"})

    @v_args(inline=True)
    def param(self, k, v):
        return (str(k), v)

    def number(self, items):
        x = float(items[0])
        if math.isfinite(x) and abs(x - int(x)) < 1e-12:
            return int(x)
        return x

    def string(self, items):
        return strip_str(items[0])

    def ident(self, items):
        return str(items[0])

    def comment(self, _):
        return None


def strip_str(tok: Token | str) -> str:
    s = str(tok)
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1]
    return s


# =========================
# Contexto de execução
# =========================

@dataclass
class Context:
    model_id: Optional[str] = None
    tokenizer_id: Optional[str] = None
    dataset_cfg: Optional[Dict[str, Any]] = None
    prompt_template: Optional[str] = None
    save_dir: Optional[str] = None

    # dataset ops
    split: Optional[Dict[str, float]] = None  # {"train": 0.9, "eval": 0.1}
    shuffle: bool = True
    seed: int = 42

    # runtime objects
    tokenizer: Any = None
    model: Any = None
    dataset: Any = None


# =========================
# Parser / Runner
# =========================

def parse_script(text: str) -> List[Cmd]:
    with open(GRAMMAR_PATH, "r", encoding="utf-8") as f:
        grammar = f.read()
    parser = Lark(grammar, start="start", parser="lalr")
    tree = parser.parse(text)
    ast = ASTBuilder().transform(tree)
    return [node for node in ast if isinstance(node, Cmd)]


def run_program(cmds: List[Cmd], echo: bool = True):
    ctx = Context()
    for cmd in cmds:
        if cmd.kind == "MODEL":
            ctx.model_id = cmd.args["id"]
            if echo:
                print(f"[GenLang] MODEL = {ctx.model_id}")

        elif cmd.kind == "TOKENIZER":
            ctx.tokenizer_id = cmd.args["id"]
            if echo:
                print(f"[GenLang] TOKENIZER = {ctx.tokenizer_id}")

        elif cmd.kind == "DATASET_CSV":
            ctx.dataset_cfg = cmd.args
            if echo:
                print(f"[GenLang] DATASET CSV = {ctx.dataset_cfg}")

        elif cmd.kind == "PROMPT":
            ctx.prompt_template = cmd.args["template"]
            if echo:
                print("[GenLang] PROMPT template set.")

        elif cmd.kind == "SPLIT":
            # Parse simples de: train=90% , eval=10%
            txt = cmd.args["text"].replace(",", " ").replace("%", "")
            parts = dict(p.split("=") for p in txt.split() if "=" in p)
            tr = float(parts.get("train", "90")) / 100.0
            ev = float(parts.get("eval", str(max(0.0, 1.0 - tr) * 100))) / 100.0
            tr = max(0.0, min(1.0, tr))
            ev = max(0.0, min(1.0, ev))
            ctx.split = {"train": tr, "eval": ev}
            if echo:
                print(f"[GenLang] SPLIT train={tr:.2f} eval={ev:.2f}")

        elif cmd.kind == "SHUFFLE":
            vals = cmd.args["vals"]
            onoff = (vals[0].lower() == "on") if vals else True
            seed = ctx.seed
            for v in vals[1:]:
                if v.startswith("seed"):
                    try:
                        seed = int(v.split("=")[1])
                    except Exception:
                        pass
            ctx.shuffle = onoff
            ctx.seed = seed
            if echo:
                print(f"[GenLang] SHUFFLE {('on' if onoff else 'off')} seed={seed}")

        elif cmd.kind == "TRAIN":
            if echo:
                print("[GenLang] TRAIN start")
            train(ctx, **cmd.args["params"])
            if echo:
                print("[GenLang] TRAIN done")

        elif cmd.kind == "SAVE":
            ctx.save_dir = cmd.args["dir"]
            if echo:
                print(f"[GenLang] SAVE dir = {ctx.save_dir}")
            save(ctx)

        elif cmd.kind == "GENERATE":
            if echo:
                print("[GenLang] GENERATE")
            text = generate(ctx, cmd.args["text"], **cmd.args.get("params", {}))
            print("\n=== Generated ===\n" + text + "\n=================\n")

        elif cmd.kind == "EVAL":
            if echo:
                print("[GenLang] EVAL perplexity")
            ppl = eval_perplexity(ctx)
            print(f"\n=== Perplexity ===\n{ppl:.4f}\n==================\n")

        elif cmd.kind == "SET":
            if echo:
                print(f"[GenLang] SET {cmd.args['name']} = {cmd.args['value']}")

        else:
            raise RuntimeError(f"Unknown command: {cmd.kind}")


# =========================
# ML glue
# =========================

def _import_ml():
    try:
        import datasets
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
        return (
            datasets,
            torch,
            AutoTokenizer,
            AutoModelForCausalLM,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except Exception as e:
        raise RuntimeError(
            "Dependências ML não encontradas. Instale conforme 'requirements.txt'. "
            f"Erro original: {e}"
        )


def _ensure_tokenizer_model(ctx: Context):
    datasets, torch, AutoTokenizer, AutoModelForCausalLM, *_ = _import_ml()
    if not ctx.model_id:
        raise RuntimeError("MODEL não definido.")
    tok_id = ctx.tokenizer_id if (ctx.tokenizer_id and ctx.tokenizer_id != "AUTO") else ctx.model_id
    if ctx.tokenizer is None:
        ctx.tokenizer = AutoTokenizer.from_pretrained(tok_id, use_fast=True)
        # GPT-2 e afins não têm pad_token: use EOS
        if ctx.tokenizer.pad_token is None:
            ctx.tokenizer.pad_token = ctx.tokenizer.eos_token
    if ctx.model is None:
        ctx.model = AutoModelForCausalLM.from_pretrained(ctx.model_id)
        # Ajuste do embedding se alteramos o tokenizer
        ctx.model.resize_token_embeddings(len(ctx.tokenizer))


def _prepare_dataset(ctx: Context, max_length: int = 256):
    datasets, torch, *_ = _import_ml()
    if not ctx.dataset_cfg:
        raise RuntimeError("DATASET não definido.")
    path = ctx.dataset_cfg["path"]
    text_col = ctx.dataset_cfg["text_col"]

    ds = datasets.load_dataset("csv", data_files={"data": path})["data"]
    tmpl = ctx.prompt_template

    def apply_template(batch):
        inputs = batch[text_col]
        if tmpl:
            texts = [tmpl.replace("{input}", str(x)) for x in inputs]
        else:
            texts = [str(x) for x in inputs]
        return {"text": texts}

    ds = ds.map(apply_template, batched=True, remove_columns=ds.column_names)

    if ctx.shuffle:
        ds = ds.shuffle(seed=ctx.seed)

    # Split opcional
    if ctx.split:
        tr = max(0.0, min(1.0, ctx.split.get("train", 0.9)))
        split = ds.train_test_split(test_size=1.0 - tr, seed=ctx.seed)
        train_ds = split["train"]
        eval_ds = split["test"]
    else:
        train_ds = ds
        eval_ds = None

    def tok(batch):
        return ctx.tokenizer(batch["text"], truncation=True, max_length=max_length)

    train_ds = train_ds.map(tok, batched=True)
    if eval_ds is not None:
        eval_ds = eval_ds.map(tok, batched=True)

    return train_ds, eval_ds


def train(
    ctx: Context,
    epochs: int = 1,
    batch: int = 2,
    lr: float = 5e-5,
    max_length: int = 256,
    lora: str = "off",
    r: int = 8,
    alpha: int = 16,
    target_modules: str = "q_proj,v_proj,k_proj,o_proj",
    **kwargs,
):
    (
        datasets,
        torch,
        AutoTokenizer,
        AutoModelForCausalLM,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    ) = _import_ml()
    _ensure_tokenizer_model(ctx)
    train_ds, eval_ds = _prepare_dataset(ctx, max_length=max_length)
    collator = DataCollatorForLanguageModeling(tokenizer=ctx.tokenizer, mlm=False)

    # LoRA (opcional)
    if str(lora).lower() in ("on", "true", "1"):
        try:
            from peft import LoraConfig, get_peft_model
        except Exception as e:
            raise RuntimeError(
                "LoRA solicitado (lora=on), mas o pacote 'peft' não está instalado. "
                "Adicione 'peft>=0.11.0' no requirements.txt e rode 'pip install -r requirements.txt'. "
                f"Erro original: {e}"
            )
        modules = [m.strip() for m in target_modules.split(",") if m.strip()]
        peft_cfg = LoraConfig(
            r=int(r),
            lora_alpha=int(alpha),
            lora_dropout=0.05,
            target_modules=modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        ctx.model = get_peft_model(ctx.model, peft_cfg)
        print(f"[GenLang] LoRA ON r={r} alpha={alpha} targets={modules}")

    outdir = ctx.save_dir or "./out/model"
    args = TrainingArguments(
        output_dir=outdir,
        per_device_train_batch_size=batch,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=5,
        save_steps=50,
        save_total_limit=1,
        report_to="none",
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=False,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=50,
    )
    trainer = Trainer(
        model=ctx.model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    trainer.train()


def save(ctx: Context):
    if not ctx.save_dir:
        raise RuntimeError("SAVE dir não definido.")
    os.makedirs(ctx.save_dir, exist_ok=True)
    if ctx.tokenizer:
        ctx.tokenizer.save_pretrained(ctx.save_dir)
    if ctx.model:
        ctx.model.save_pretrained(ctx.save_dir)


def generate(
    ctx: Context,
    text: str,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_p: float = 0.95,
    no_repeat_ngram_size: int = 4,
    repetition_penalty: float = 1.15,
    top_k: int = 50,
    **kwargs,
) -> str:
    _, torch, *_ = _import_ml()
    _ensure_tokenizer_model(ctx)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx.model.to(device)
    inputs = ctx.tokenizer(text, return_tensors="pt").to(device)

    eos_id = ctx.tokenizer.eos_token_id or ctx.tokenizer.pad_token_id

    with torch.no_grad():
        out = ctx.model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
        )
    s = ctx.tokenizer.decode(out[0], skip_special_tokens=True)

    # Se o template usa "Resposta:", mantenha só o trecho após isso
    if "Resposta:" in s:
        s = s.split("Resposta:", 1)[1]

    # Limpeza básica
    s = s.replace("\\n", "\n").replace("\\t", "\t")
    s = re.sub(r"[^\x09\x0A\x0D\x20-\x7EÀ-ÖØ-öø-ÿ€£¥çÇñÑá-úÁ-Ú]", "", s)
    return s.strip()


def eval_perplexity(ctx: Context, max_length: int = 256) -> float:
    (
        datasets,
        torch,
        AutoTokenizer,
        AutoModelForCausalLM,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    ) = _import_ml()
    _ensure_tokenizer_model(ctx)
    _, eval_ds = _prepare_dataset(ctx, max_length=max_length)
    if eval_ds is None or len(eval_ds) == 0:
        return float("nan")

    collator = DataCollatorForLanguageModeling(tokenizer=ctx.tokenizer, mlm=False)
    args = TrainingArguments(
        output_dir=ctx.save_dir or "./out/eval",
        per_device_eval_batch_size=2,
        report_to="none",
    )
    trainer = Trainer(
        model=ctx.model, args=args, data_collator=collator, eval_dataset=eval_ds
    )
    metrics = trainer.evaluate()
    loss = metrics.get("eval_loss", None)
    if loss is None:
        return float("nan")
    return math.exp(loss)
