from captionizer import caption_from_path
from pytorch_lightning.callbacks import Callback

class CaptionSaverCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        print('Adding training captions to checkpoint [Dataloader]')
        data = trainer.train_dataloader.loaders.sampler.data_source  # type: ignore
        prompts = set([
            caption_from_path(image_path, data.data_root, data.coarse_class_text, data.placeholder_token)
            for image_path in data.image_paths
        ])
        trained_prompts = (list(prompts))
        checkpoint['trained_captions'] = trained_prompts
