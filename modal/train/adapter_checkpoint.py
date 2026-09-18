"""Save LoRA-only experiments without probing or modifying the base-model cache."""
from pathlib import Path


def save_adapter(model, folder, state_dict=None):
    embeddings = (model.get_input_embeddings(), model.get_output_embeddings())
    if any(parameter.requires_grad for layer in embeddings if layer is not None for parameter in layer.parameters()):
        raise ValueError("Adapter-only checkpointing cannot discard trainable embeddings")
    model.save_pretrained(folder, state_dict=state_dict, save_embedding_layers=False)


def adapter_trainer_class():
    import torch
    from transformers import Trainer

    class AdapterTrainer(Trainer):
        def _save(self, output_dir=None, state_dict=None):
            folder = Path(output_dir or self.args.output_dir)
            folder.mkdir(parents=True, exist_ok=True)
            save_adapter(self.model, folder, state_dict)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(folder)
            torch.save(self.args, folder / "training_args.bin")

    return AdapterTrainer
