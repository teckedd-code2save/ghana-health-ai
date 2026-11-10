"""IFEval-only native generation; never calls or changes the R5 256-token reply."""
import time

from run_understanding_r5_evaluation import digest, require

PROTOCOL = {"version": "r5-ifeval-generation-v1", "context_tokens": 16384, "max_new_tokens": 8192,
            "case_seconds": 240, "do_sample": False, "enable_thinking": False, "num_beams": 1,
            "repetition_penalty": 1.0, "seed": 17, "truncate": False, "system_message": None,
            "messages": "one_original_user_prompt", "prompt_rewriting": False}
POLICIES = ("raw", "product_guarded")
GUARD = "repeat-block-v1"


def messages(prompt):
    require(isinstance(prompt, str) and bool(prompt.strip()), "Empty official prompt")
    return [{"role": "user", "content": prompt}]


def prompt_ids(core, tokenizer, prompt):
    return core.prompt_ids(tokenizer, messages(prompt),
                           {k: PROTOCOL[k] for k in ("context_tokens", "max_new_tokens")})


def verify_reply(engine, core, prompt, variant, ids, result, policy, *, loaded=True):
    require(policy in POLICIES and result["generation_policy"] == policy and result["protocol"] == PROTOCOL
            and result["identity"] == engine.identity(variant)
            and result["messages_sha256"] == digest(messages(prompt))
            and result["prompt_ids_sha256"] == digest(ids), "IFEval generation identity differs")
    expected = core.completion(engine.tokenizer, result["output_ids"], PROTOCOL["max_new_tokens"],
                               result["timed_out"], result["seconds"])
    require(result["diagnostics"] == expected and result["output"] == expected["prediction"], "Native output proof differs")
    stopped = result.get("guard_stopped", False)
    if policy == "product_guarded":
        from response_loop_guard import repeated_passage, VERSION
        require(VERSION == GUARD and result["guard_version"] == GUARD
                and result["guard_trigger"] == (repeated_passage(result["output"]) if stopped else None)
                and (not stopped or result["guard_trigger"] is not None), "Product guard proof differs")
    else:
        require(not stopped and not any(k in result for k in ("guard_version", "guard_trigger")), "Raw result contains guard behavior")
    require(result["finish_reason"] == ("product_guard" if stopped else expected["finish_reason"])
            and result["complete"] == (expected["complete"] and not stopped)
            and result["cap_hit"] == (expected["finish_reason"] == "max_new_tokens"), "Stop/cap proof differs")
    require(not loaded or variant == "base" or variant in engine.loaded, "Adapter tensors not verified")


class Runtime:
    def __init__(self, engine, core):
        self.engine, self.core = engine, core
        self.tokenizer = engine.tokenizer

    def identity(self, variant):
        return self.engine.identity(variant)

    @property
    def loaded(self):
        return self.engine.loaded

    def reply(self, prompt, variant, policy):
        import random
        import numpy as np
        import torch
        from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList
        require(policy in POLICIES, "Unknown IFEval generation policy")
        ids = prompt_ids(self.core, self.tokenizer, prompt)
        detector = None
        if policy == "product_guarded":
            from response_loop_guard import repeated_passage, VERSION
            require(VERSION == GUARD, "Guard implementation differs")
            detector = repeated_passage
        with self.engine.select(variant):
            tick = time.monotonic()
            tokenizer, eos = self.tokenizer, self.core.EOS
            class Stop(StoppingCriteria):
                timed_out = False
                guard_stopped = False

                def __call__(self, input_ids, scores, **kwargs):
                    self.timed_out = time.monotonic() - tick >= PROTOCOL["case_seconds"]
                    if detector:
                        emitted = input_ids[0, len(ids):].tolist()
                        body = emitted[:-1] if emitted and emitted[-1] == eos else emitted
                        self.guard_stopped = bool(detector(tokenizer.decode(body, skip_special_tokens=False,
                                                                          clean_up_tokenization_spaces=False)))
                    return torch.full((input_ids.shape[0],), self.timed_out or self.guard_stopped,
                                      dtype=torch.bool, device=input_ids.device)
            timer = Stop()
            random.seed(PROTOCOL["seed"])
            np.random.seed(PROTOCOL["seed"])
            torch.manual_seed(PROTOCOL["seed"])
            torch.cuda.manual_seed_all(PROTOCOL["seed"])
            inputs = torch.tensor([ids], dtype=torch.long, device="cuda")
            config = GenerationConfig(**{k: PROTOCOL[k] for k in ("do_sample", "num_beams", "repetition_penalty", "max_new_tokens")},
                eos_token_id=self.core.EOS, pad_token_id=self.core.PAD, bos_token_id=None,
                use_cache=True, return_dict_in_generate=True)
            with torch.inference_mode():
                output = self.engine.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                    generation_config=config, stopping_criteria=StoppingCriteriaList([timer]))
            torch.cuda.synchronize()
            generated = output.sequences[0, len(ids):].tolist()
            value = self.core.completion(self.tokenizer, generated, PROTOCOL["max_new_tokens"], timer.timed_out, time.monotonic() - tick)
        result = {"identity": self.identity(variant), "messages_sha256": digest(messages(prompt)),
            "prompt_ids_sha256": digest(ids), "protocol": PROTOCOL, "generation_policy": policy,
            "output_ids": generated, "output": value["prediction"], "seconds": value["seconds"],
            "diagnostics": value, "timed_out": timer.timed_out, "guard_stopped": timer.guard_stopped,
            "finish_reason": "product_guard" if timer.guard_stopped else value["finish_reason"],
            "complete": value["complete"] and not timer.guard_stopped, "cap_hit": value["finish_reason"] == "max_new_tokens"}
        if detector:
            result.update(guard_version=GUARD, guard_trigger=detector(result["output"]) if timer.guard_stopped else None)
        verify_reply(self, self.core, prompt, variant, ids, result, policy)
        return result
