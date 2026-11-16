"""Minimal generation probe: print tokenizer + ids type, then generate one answer."""
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "transformers==5.17.0", "peft==0.20.0", "torch==2.13.0", "accelerate==1.15.0")

app = modal.App("gen-probe")
vols = {"/cache": modal.Volume.from_name("ghana-health-understanding-hf-cache", create_if_missing=False),
        "/artifacts": modal.Volume.from_name("ghana-health-understand-train", create_if_missing=False)}
MODEL = "McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1"
REV = "693f12e444a30971ca53889128d7bade9491ecb3"
R6 = "/artifacts/understanding-adaptation-r6/understanding_20261113T152209Z/responses-adapter"


@app.function(gpu="H100", image=image, volumes=vols, timeout=1800)
def probe():
    import torch, traceback
    from transformers import Qwen3_5ForCausalLM, AutoTokenizer, GenerationConfig
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, cache_dir="/cache/hf", padding_side="left")
    print("chat_template repr:", repr(tok.chat_template)[:300])
    print("chat_template type:", type(tok.chat_template).__name__)
    model = Qwen3_5ForCausalLM.from_pretrained(MODEL, revision=REV, cache_dir="/cache/hf",
                                               torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, R6)
    model.eval()
    prompt = "What is 2+2?\nA. 3\nB. 4\nC. 5\nD. 6\nAnswer:"
    try:
        if tok.chat_template:
            ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
            print("via apply_chat_template -> ids type:", type(ids).__name__,
                  "| value[:10]:", repr(ids)[:120])
        else:
            print("NO chat_template; using encode()")
            ids = tok.encode(prompt)
            print("via encode -> ids type:", type(ids).__name__, "| value[:10]:", ids[:10])
        inputs = torch.tensor([ids], dtype=torch.long, device="cuda")
        config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=8,
                                  eos_token_id=248046, pad_token_id=248044, return_dict_in_generate=True)
        with torch.inference_mode():
            out = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                 generation_config=config)
        print("GENERATED:", repr(tok.decode(out.sequences[0, len(ids):], skip_special_tokens=True)))
    except Exception:
        traceback.print_exc()


@app.local_entrypoint()
def main():
    probe.remote()
