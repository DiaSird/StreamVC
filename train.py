import abc
from dataclasses import dataclass
import os
from itertools import islice
from typing import Union

import accelerate.checkpointing
import safetensors as st
import safetensors.torch
import torch
import torch.nn as nn
import accelerate
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger

from streamvc.model import StreamVC
from streamvc.train.discriminator import Discriminator
from streamvc.train.encoder_classifier import EncoderClassifier
from streamvc.train.data import PreprocessedDataset
from streamvc.train.loss import (
    GeneratorLoss,
    DiscriminatorLoss,
    FeatureLoss,
    ReconstructionLoss,
)
import tyro
import config.lr_scheduler as scheduler_config
from config.training_config import ContentEncoderTrainingConfig, DecoderTrainingConfig
from config.utils import get_flattened_config_dict
from typing_extensions import Annotated
import logging

logger = get_logger(__name__)

accelerator = Accelerator(
    log_with="tensorboard",
    project_config=ProjectConfiguration(
        project_dir=os.getcwd(), logging_dir=os.path.join(os.getcwd(), "logs")
    ),
    dataloader_config=DataLoaderConfiguration(split_batches=True),
)

NUM_CLASSES = 100
EMBEDDING_DIMS = 64
SAMPLES_PER_FRAME = 320
DEVICE = accelerator.device

####### cli commands #######

TrainingConfig = Union[
    Annotated[
        ContentEncoderTrainingConfig,
        tyro.conf.subcommand(
            "content-encoder", description="Train the Content Encoder module"
        ),
    ],
    Annotated[
        DecoderTrainingConfig,
        tyro.conf.subcommand(
            "decoder",
            description="Train the Decoder and Traget Speaker modules, requires a trained Content Encoder",
        ),
    ],
]

####### tensorboard logging functions #######


@accelerator.on_main_process
def log_gradients_tensorboard(model, step):
    summary_writer = accelerator.get_tracker("tensorboard").tracker
    for name, param in model.named_parameters():
        if param.grad is not None:
            summary_writer.add_histogram(
                f"gradients/{name}", param.grad, global_step=step
            )


@accelerator.on_main_process
def log_labels_tensorboard(outputs_flat, labels_flat, step):
    _, predicted = torch.max(outputs_flat.data, 1)
    summary_writer = accelerator.get_tracker("tensorboard").tracker
    summary_writer.add_histogram("labels/content_encoder", predicted, global_step=step)
    summary_writer.add_histogram("labels/hubert", labels_flat, global_step=step)


####### LR schedulers #######


def get_lr_Scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    total_steps: int,
    discriminator: bool = False,
):
    scheduler = config.lr_scheduler
    if scheduler is None:
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
    elif isinstance(scheduler, scheduler_config.StepLR):
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler.step, gamma=scheduler.gamma
        )
    elif isinstance(scheduler, scheduler_config.LinearLR):
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=scheduler.start,
            end_factor=scheduler.end,
            total_iters=total_steps,
        )
    elif isinstance(scheduler, scheduler_config.ExponentialLR):
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler.gamma)
    elif isinstance(scheduler, scheduler_config.OneCycleLR):
        max_lr = scheduler.max
        if discriminator:
            assert isinstance(config, DecoderTrainingConfig)
            max_lr = config.lr_discriminator_multiplier * max_lr
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=scheduler.pct_start,
            div_factor=scheduler.div_factor,
            final_div_factor=scheduler.final_div_factor,
        )
    elif isinstance(scheduler, scheduler_config.CosineAnnealingWarmRestarts):
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler.T_0,
            T_mult=1,
            eta_min=scheduler.eta_min,
        )
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")


####### Savable counter #######


@dataclass
class CounterState:
    value: int = 0

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"value": torch.tensor(self.value, dtype=torch.long, device="cpu")}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.value = state_dict["value"].item()


####### Training #######


class TrainerBase(abc.ABC):
    def __init__(self, config: TrainingConfig):
        self.config = config
        config: TrainingConfig = self.config
        self.train_dataset = PreprocessedDataset(config.datasets.train_dataset_path)
        self.train_dataloader = self.train_dataset.get_dataloader(
            config.batch_size, limit_samples=config.limit_batch_samples
        )

        self.steps_per_epoch = max(
            [
                i
                for i in [len(self.train_dataloader), config.limit_num_batches]
                if i is not None
            ]
        )
        self.total_steps = self.steps_per_epoch * config.num_epochs

        self.dev_dataset = PreprocessedDataset(config.datasets.dev_dataset_path)
        self.dev_dataloader = self.dev_dataset.get_dataloader(
            config.batch_size, limit_samples=config.limit_batch_samples
        )
        self.models = {}
        self._prepered_training = False

    @abc.abstractmethod
    def prepare_training(self) -> None:
        assert not self._prepered_training, "prepare_training can only be called once"
        self._prepered_training = True

    @abc.abstractmethod
    def train_step(
        self,
        batch: torch.Tensor,
        lables: torch.Tensor,
        mask: torch.Tensor,
    ) -> Annotated[float, "loss"]: ...

    def save_state(self):
        accelerator.wait_for_everyone()
        accelerator.save_state(
            os.path.join(
                self.config.model_checkpoint_path, f"{self.config.run_name}_state"
            )
        )

    def save_models(self):
        for name, model in self.models.items():
            accelerator.save_model(
                model,
                save_directory=os.path.join(
                    self.config.model_checkpoint_path,
                    f"{self.config.run_name}_{name}",
                ),
            )

    def train(self) -> None:
        config = self.config
        if not self._prepered_training:
            self.prepare_training()

        if not hasattr(self, "global_step"):
            self.global_step = CounterState()
            accelerator.register_for_checkpointing(self.global_step)
        else:
            self.global_step.value = 0

        if config.restore_state_dir is not None:
            accelerator.load_state(config.restore_state_dir)

        start_epoch = self.global_step.value // self.steps_per_epoch
        start_step = self.global_step.value % self.steps_per_epoch

        losses_aggregate = []
        for epoch in range(start_epoch, config.num_epochs):
            logger.info(f"epoch num: {epoch}")

            dataloader = self.train_dataloader
            if start_step != 0:
                dataloader = accelerator.skip_first_batches(
                    self.train_dataloader, start_step
                )

            for step, (batch, labels, mask) in enumerate(
                islice(dataloader, config.limit_num_batches),
                start=start_step,
            ):
                loss = self.train_step(batch, labels, mask)

                losses_aggregate.append(loss)
                if (self.global_step.value + 1) % config.print_interval == 0:
                    logger.info(
                        f"[{epoch}, {step:5}] loss: {torch.tensor(losses_aggregate).mean().item():.4}"
                    )
                    losses_aggregate = []

                if (self.global_step.value + 1) % config.model_checkpoint_interval == 0:
                    self.save_state()

                self.after_train_step()
                self.global_step.value += 1
                start_step = 0

        self.save_state()
        self.save_models()

    def after_train_step(self) -> None:
        pass


class ContentEncoderTrainer(TrainerBase):
    def __init__(self, config: ContentEncoderTrainingConfig):
        super().__init__(config)
        streamvc = StreamVC(gradient_checkpointing=config.gradient_checkpointing)
        content_encoder = streamvc.content_encoder
        wrapped_content_encoder = EncoderClassifier(
            content_encoder, EMBEDDING_DIMS, NUM_CLASSES, dropout=config.dropout
        )
        self.content_encoder = wrapped_content_encoder
        self.models["content_encoder"] = wrapped_content_encoder

    def prepare_training(self) -> None:
        super().prepare_training()
        self.content_encoder.train()
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = torch.optim.AdamW(
            params=self.content_encoder.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay,
        )

        scheduler = get_lr_Scheduler(optimizer, self.config, self.total_steps)

        [
            self.content_encoder,
            self.optimizer,
            self.train_dataloader,
            self.dev_dataloader,
            self.criterion,
            self.scheduler,
        ] = accelerator.prepare(
            self.content_encoder,
            optimizer,
            self.train_dataloader,
            self.dev_dataloader,
            criterion,
            scheduler,
        )

    def train_step(
        self,
        batch: torch.Tensor,
        labels: torch.Tensor,
        _,
    ) -> Annotated[float, "loss"]:
        outputs = self.content_encoder(batch)
        outputs_flat = outputs.view(-1, NUM_CLASSES)
        labels_flat = labels.view(-1)
        loss = self.criterion(outputs_flat, labels_flat)
        accelerator.backward(loss)

        if (
            self.config.log_gradient_interval
            and (self.global_step.value + 1) % self.config.log_gradient_interval == 0
        ):
            log_gradients_tensorboard(self.content_encoder, self.global_step.value)

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step(self.global_step.value)
        accelerator.log(
            {
                "loss/content_encoder": loss.item(),
                "lr/content_encoder": self.scheduler.get_last_lr()[0],
            },
            step=self.global_step.value,
        )

        if (
            self.config.log_labels_interval
            and (self.global_step.value + 1) % self.config.log_labels_interval == 0
        ):
            log_labels_tensorboard(outputs_flat, labels_flat, self.global_step.value)

        return loss.item()

    def after_train_step(self) -> None:
        if (self.global_step.value + 1) % self.config.accuracy_interval == 0:
            accuracy = self.compute_content_encoder_accuracy()
            accuracies = accelerator.gather_for_metrics([accuracy])
            accuracies = torch.tensor(accuracies)
            gathered_accuracy = accuracies.mean().item()
            accelerator.log(
                {"accuracy/content_encoder": gathered_accuracy},
                step=self.global_step.value,
            )
            logger.info(f"accuracy: {accuracy:.2f}%")

    @torch.no_grad()
    def compute_content_encoder_accuracy(self):
        correct = 0
        total = 0
        self.content_encoder.eval()
        for batch, labels, _ in islice(
            self.dev_dataloader, self.config.accuracy_limit_num_batches
        ):
            batch = batch.to(accelerator.device)
            outputs = self.content_encoder(batch)
            outputs_flat = outputs.view(-1, NUM_CLASSES)
            labels_flat = labels.view(-1)
            _, predicted = torch.max(outputs_flat.data, 1)
            total += torch.sum(labels_flat != -1).item()
            correct += (predicted == labels_flat).sum().item()
        self.content_encoder.train()

        return 100 * correct / total


class DecoderTrainer(TrainerBase):
    def __init__(self, config: DecoderTrainingConfig):
        super().__init__(config)
        streamvc = StreamVC(gradient_checkpointing=config.gradient_checkpointing)
        wrapped_encoder_state_dict = st.torch.load_file(
            config.content_encoder_checkpoint
        )
        encoder_state_dict = {
            key[len("encoder.") :]: value
            for key, value in wrapped_encoder_state_dict.items()
            if key.startswith("encoder.")
        }
        streamvc.content_encoder.load_state_dict(encoder_state_dict)

        self.generator = streamvc
        self.discriminator = Discriminator(
            gradient_checkpointing=config.gradient_checkpointing
        )
        self.models["generator"] = self.generator
        self.models["discriminator"] = self.discriminator

    def prepare_training(self) -> None:
        super().prepare_training()
        config = self.config
        for param in self.generator.content_encoder.parameters():
            param.requires_grad = False

        optimizer_generator = torch.optim.AdamW(
            params=[
                param for param in self.generator.parameters() if param.requires_grad
            ],
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )

        lr_discriminator = config.lr_discriminator_multiplier * config.lr
        optimizer_discriminator = torch.optim.AdamW(
            params=self.discriminator.parameters(),
            lr=lr_discriminator,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )

        scheduler_generator = get_lr_Scheduler(
            optimizer_generator, config, self.total_steps
        )
        scheduler_discriminator = get_lr_Scheduler(
            optimizer_discriminator, config, self.total_steps, discriminator=True
        )

        generator_loss_fn = GeneratorLoss()
        discriminator_loss_fn = DiscriminatorLoss()
        feature_loss_fn = FeatureLoss()
        reconstruction_loss_fn = ReconstructionLoss(
            gradient_checkpointing=config.gradient_checkpointing
        )

        [
            self.generator,
            self.discriminator,
            self.optimizer_generator,
            self.optimizer_discriminator,
            self.scheduler_generator,
            self.scheduler_discriminator,
            self.train_dataloader,
            self.generator_loss_fn,
            self.discriminator_loss_fn,
            self.feature_loss_fn,
            self.reconstruction_loss_fn,
        ] = accelerator.prepare(
            self.generator,
            self.discriminator,
            optimizer_generator,
            optimizer_discriminator,
            scheduler_generator,
            scheduler_discriminator,
            self.train_dataloader,
            generator_loss_fn,
            discriminator_loss_fn,
            feature_loss_fn,
            reconstruction_loss_fn,
        )

    def train_step(
        self,
        batch: torch.Tensor,
        _,
        mask: torch.Tensor,
    ) -> Annotated[float, "loss"]:
        config = self.config
        x_pred_t = self.generator(batch, batch)
        # Remove the first 2 frames from the generated audio
        # because we match a output frame t with input frame t-2.
        x_pred_t = x_pred_t[..., SAMPLES_PER_FRAME * 2 :]
        batch = batch[..., : x_pred_t.shape[-1]]

        mask_ratio = mask.sum(dim=-1) / mask.shape[-1]

        # Train Discriminator #
        self.discriminator.zero_grad()

        discriminator_fake_detached = self.discriminator(x_pred_t.detach())
        discriminator_real = self.discriminator(batch)

        discriminator_loss = self.discriminator_loss_fn(
            discriminator_real, discriminator_fake_detached, mask_ratio
        )

        accelerator.backward(discriminator_loss)

        if (
            config.log_gradient_interval
            and (self.global_step.value + 1) % config.log_gradient_interval == 0
        ):
            log_gradients_tensorboard(self.discriminator, self.global_step.value)

        self.optimizer_discriminator.step()
        self.scheduler_discriminator.step(self.global_step.value)

        # Train Generator #

        self.generator.zero_grad()

        discriminator_fake = self.discriminator(x_pred_t)

        adversarial_loss = self.generator_loss_fn(discriminator_fake, mask_ratio)
        feature_loss = self.feature_loss_fn(
            discriminator_real, discriminator_fake, mask_ratio
        )
        reconstruction_loss = self.reconstruction_loss_fn(batch, x_pred_t, mask_ratio)

        losses = (
            config.lambda_adversarial * adversarial_loss
            + config.lambda_feature * feature_loss
            + config.lambda_reconstruction * reconstruction_loss
        )

        accelerator.backward(losses)

        if (
            config.log_gradient_interval
            and (self.global_step.value + 1) % config.log_gradient_interval == 0
        ):
            log_gradients_tensorboard(self.generator, self.global_step.value)

        self.optimizer_generator.step()
        self.scheduler_generator.step(self.global_step.value)

        accelerator.log(
            {
                "loss/discriminator": discriminator_loss.item(),
                "loss/adversarial": adversarial_loss.item(),
                "loss/feature_matching": feature_loss.item(),
                "loss/reconstruction": reconstruction_loss.item(),
                "lr/generator": self.scheduler_generator.get_last_lr()[0],
                "lr/discriminator": self.scheduler_discriminator.get_last_lr()[0],
            },
            step=self.global_step.value,
        )

        return reconstruction_loss.item()


def main(config: TrainingConfig) -> None:
    """Main function for training StreamVC model."""
    logger.debug(f"DEVICE={accelerator.device}")
    hps = get_flattened_config_dict(config)
    hps["num processes"] = accelerator.num_processes
    hps["mixed precision"] = accelerator.mixed_precision

    if accelerator.gradient_accumulation_steps > 1:
        raise ValueError(
            "Gradient accumulation is not supported. Disable gradient accumulation from accelerate"
        )
    logger.debug(f"{hps=}")

    accelerator.init_trackers(config.run_name, config=hps)

    if isinstance(config, ContentEncoderTrainingConfig):
        trainer = ContentEncoderTrainer(config)
    else:
        trainer = DecoderTrainer(config)

    trainer.train()

    accelerator.end_training()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.setLevel(logging.DEBUG)
    training_config = tyro.cli(TrainingConfig)
    main(training_config)
