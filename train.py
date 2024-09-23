import os
import time
from itertools import islice
from typing import Union

import safetensors as st
import safetensors.torch
import torch
import torch.nn as nn
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


def sizeof_fmt(num, suffix="B"):
    for unit in ("", "K", "M", "G", "T", "P"):
        if abs(num) < 1024.0:
            return f"{num:.1f} {unit}{suffix}"
        num /= 1024.0


def print_cuda_memory(s):
    if accelerator.device.type != "cuda":
        logger.debug(s)
        return
    free, total = torch.cuda.mem_get_info()
    curr = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()

    size = {"allocated": curr, "total": total, "free": free, "peak": peak}

    logger.debug(
        " | ".join(map(lambda x: f"{x[0]} {sizeof_fmt(x[1]):8}", size.items()))
        + f" - {s}"
    )


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


def get_lr_Scheduler(optimizer, args, discriminator=False):
    if args.scheduler == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma
        )
    elif args.scheduler == "LinearLR":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=args.scheduler_linear_start,
            end_factor=args.scheduler_linear_end,
            total_iters=args.num_epochs * args.limit_num_batches,
        )
    elif args.scheduler == "ExponentialLR":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.scheduler_gamma
        )
    elif args.scheduler == "OneCycleLR":
        max_lr = args.scheduler_onecycle_max
        if discriminator:
            max_lr = args.lr_discriminator_multiplier * max_lr
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=args.limit_num_batches,
            epochs=args.num_epochs,
            pct_start=args.scheduler_onecycle_pct_start,
            div_factor=args.scheduler_onecycle_div_factor,
            final_div_factor=args.scheduler_onecycle_final_div_factor,
        )
    elif args.scheduler == "CosineAnnealingWarmRestarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.scheduler_step,
            T_mult=1,
            eta_min=args.scheduler_cosine_eta_min,
        )
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")


def train_content_encoder(
    content_encoder: nn.Module, config: ContentEncoderTrainingConfig
) -> nn.Module:
    """
    Train a content encoder as a classifier to predict the same labels as a discrete hubert model.

    :param content_encoder: A content encoder wrapped with a linear layer to
    :param lr: Learning rate.
    :param num_epochs: Number of epochs.
    :return: The trained content encoder wrapped with a linear layer for classification.
    """
    # TODO: add epochs or number of steps when we know how much time it takes to train the model.
    wrapped_content_encoder = EncoderClassifier(
        content_encoder, EMBEDDING_DIMS, NUM_CLASSES, dropout=config.dropout
    ).train()
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.AdamW(
        params=wrapped_content_encoder.parameters(),
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )
    scheduler = get_lr_Scheduler(optimizer, config)

    dataset = PreprocessedDataset(config.datasets.train_dataset_path)
    dataloader = dataset.get_dataloader(
        config.batch_size, limit_samples=config.limit_batch_samples
    )

    dev_dataset = PreprocessedDataset(config.datasets.dev_dataset_path)
    dev_dataloader = dev_dataset.get_dataloader(
        config.batch_size, limit_samples=config.limit_batch_samples
    )

    [
        wrapped_content_encoder,
        optimizer,
        dataloader,
        dev_dataloader,
        criterion,
        scheduler,
    ] = accelerator.prepare(
        wrapped_content_encoder,
        optimizer,
        dataloader,
        dev_dataloader,
        criterion,
        scheduler,
    )

    costs = []
    global_step = 0
    for epoch in range(0, config.num_epochs):
        logger.info(f"epoch num: {epoch}")
        for step, (batch, labels, _) in enumerate(
            islice(dataloader, config.limit_num_batches)
        ):
            with accelerator.accumulate(wrapped_content_encoder):
                outputs = wrapped_content_encoder(batch)
                outputs_flat = outputs.view(-1, NUM_CLASSES)
                labels_flat = labels.view(-1)
                loss = criterion(outputs_flat, labels_flat)
                accelerator.backward(loss)

                if (
                    config.log_gradient_interval
                    and (global_step + 1) % config.log_gradient_interval == 0
                ):
                    log_gradients_tensorboard(wrapped_content_encoder, global_step)

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step(global_step)
                accelerator.log(
                    {
                        "loss/content_encoder": loss.item(),
                        "lr/content_encoder": scheduler.get_last_lr()[0],
                        "allocated_memory": torch.cuda.max_memory_allocated()
                        if accelerator.device.type == "cuda"
                        else 0,
                    },
                    step=global_step,
                )
                costs.append(loss.item())

            # print loss
            if (global_step + 1) % config.print_interval == 0:
                logger.info(
                    f"[{epoch}, {step:5}] loss: {torch.tensor(costs).mean().item():.4}"
                )
                costs = []

            if (
                config.log_labels_interval
                and (global_step + 1) % config.log_labels_interval == 0
            ):
                log_labels_tensorboard(outputs_flat, labels_flat, global_step)

            # save model checkpoints
            if (global_step + 1) % config.model_checkpoint_interval == 0:
                accelerator.save_model(
                    wrapped_content_encoder,
                    save_directory=os.path.join(
                        config.checkpoint_path,
                        f"{config.run_name}_content_encoder_{epoch}_{step}",
                    ),
                )

            if (global_step + 1) % config.accuracy_interval == 0:
                accuracy = compute_content_encoder_accuracy(
                    islice(dev_dataloader, 10), wrapped_content_encoder
                )
                accuracies = accelerator.gather_for_metrics([accuracy])
                accuracies = torch.tensor(accuracies)
                gathered_accuracy = accuracies.mean().item()
                accelerator.log(
                    {"accuracy/content_encoder": gathered_accuracy}, step=global_step
                )
                logger.info(f"accuracy: {accuracy:.2f}%")
            if accelerator.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

            global_step += 1


@torch.no_grad()
def compute_content_encoder_accuracy(dataloader, wrapped_content_encoder: nn.Module):
    correct = 0
    total = 0
    wrapped_content_encoder.to(accelerator.device).eval()
    for batch, labels, _ in dataloader:
        batch = batch.to(accelerator.device)
        outputs = wrapped_content_encoder(batch)
        outputs_flat = outputs.view(-1, NUM_CLASSES)
        labels_flat = labels.view(-1)
        _, predicted = torch.max(outputs_flat.data, 1)
        total += torch.sum(labels_flat != -1).item()
        correct += (predicted == labels_flat).sum().item()
    wrapped_content_encoder.train()

    return 100 * correct / total


def train_streamvc(streamvc_model: StreamVC, config: DecoderTrainingConfig) -> None:
    """
    Trains a StreamVC model.

    :param streamvc_model: The model to train.
    :param args: Hyperparameters for training.
    """
    #######################
    # Load PyTorch Models #
    #######################
    generator = streamvc_model
    discriminator = Discriminator(gradient_checkpointing=config.gradient_checkpointing)

    for param in generator.content_encoder.parameters():
        param.requires_grad = False

    #####################
    # Create optimizers #
    #####################
    optimizer_generator = torch.optim.AdamW(
        params=[param for param in generator.parameters() if param.requires_grad],
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )

    lr_discriminator = config.lr_discriminator_multiplier * config.lr
    optimizer_discriminator = torch.optim.AdamW(
        params=discriminator.parameters(),
        lr=lr_discriminator,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )

    scheduler_generator = get_lr_Scheduler(optimizer_generator, config)
    scheduler_discriminator = get_lr_Scheduler(
        optimizer_discriminator, config, discriminator=True
    )

    dataset = PreprocessedDataset(config.datasets.train_dataset_path)
    dataloader = dataset.get_dataloader(
        config.batch_size, limit_samples=config.limit_batch_samples
    )

    generator_loss_fn = GeneratorLoss()
    discriminator_loss_fn = DiscriminatorLoss()
    feature_loss_fn = FeatureLoss()
    reconstruction_loss_fn = ReconstructionLoss(
        gradient_checkpointing=config.gradient_checkpointing
    )

    [
        generator,
        discriminator,
        optimizer_generator,
        optimizer_discriminator,
        scheduler_generator,
        scheduler_discriminator,
        dataloader,
        generator_loss_fn,
        discriminator_loss_fn,
        feature_loss_fn,
        reconstruction_loss_fn,
    ] = accelerator.prepare(
        generator,
        discriminator,
        optimizer_generator,
        optimizer_discriminator,
        scheduler_generator,
        scheduler_discriminator,
        dataloader,
        generator_loss_fn,
        discriminator_loss_fn,
        feature_loss_fn,
        reconstruction_loss_fn,
    )

    costs = []
    global_step = 0
    for epoch in range(0, config.num_epochs):
        logger.info(f"epoch num: {epoch}")
        for step, (batch, _, mask) in enumerate(
            islice(dataloader, config.limit_num_batches)
        ):
            x_pred_t = generator(batch, batch)
            # Remove the first 2 frames from the generated audio
            # because we match a output frame t with input frame t-2.
            x_pred_t = x_pred_t[..., SAMPLES_PER_FRAME * 2 :]
            batch = batch[..., : x_pred_t.shape[-1]]

            mask_ratio = mask.sum(dim=-1) / mask.shape[-1]

            #######################
            # Train Discriminator #
            #######################

            discriminator.zero_grad()

            discriminator_fake_detached = discriminator(x_pred_t.detach())
            discriminator_real = discriminator(batch)

            discriminator_loss = discriminator_loss_fn(
                discriminator_real, discriminator_fake_detached, mask_ratio
            )

            accelerator.backward(discriminator_loss)

            if (
                config.log_gradient_interval
                and (global_step + 1) % config.log_gradient_interval == 0
            ):
                log_gradients_tensorboard(discriminator, global_step)

            optimizer_discriminator.step()
            scheduler_discriminator.step(global_step)

            ###################
            # Train Generator #
            ###################

            generator.zero_grad()

            discriminator_fake = discriminator(x_pred_t)

            # Compute adversarial loss.
            adversarial_loss = generator_loss_fn(discriminator_fake, mask_ratio)

            # Compute feature loss.
            feature_loss = feature_loss_fn(
                discriminator_real, discriminator_fake, mask_ratio
            )

            # Compute reconstruction loss.
            reconstruction_loss = reconstruction_loss_fn(batch, x_pred_t, mask_ratio)

            losses = (
                config.lambda_adversarial * adversarial_loss
                + config.lambda_feature * feature_loss
                + config.lambda_reconstruction * reconstruction_loss
            )

            accelerator.backward(losses)

            if (
                config.log_gradient_interval
                and (global_step + 1) % config.log_gradient_interval == 0
            ):
                log_gradients_tensorboard(generator, global_step)

            optimizer_generator.step()
            scheduler_generator.step(global_step)

            ######################
            # Update tensorboard #
            ######################
            costs.append(
                [
                    discriminator_loss.item(),
                    adversarial_loss.item(),
                    feature_loss.item(),
                    reconstruction_loss.item(),
                ]
            )

            accelerator.log(
                {
                    "loss/discriminator": discriminator_loss.item(),
                    "loss/adversarial": adversarial_loss.item(),
                    "loss/feature_matching": feature_loss.item(),
                    "loss/reconstruction": reconstruction_loss.item(),
                    "lr/generator": scheduler_generator.get_last_lr()[0],
                    "lr/discriminator": scheduler_discriminator.get_last_lr()[0],
                    "allocated_memory": torch.cuda.max_memory_allocated()
                    if accelerator.device.type == "cuda"
                    else 0,
                },
                step=global_step,
            )

            if (global_step + 1) % config.print_interval == 0:
                logger.info(
                    f"[{epoch}, {step:5}] loss: {torch.tensor(costs).mean().item():.4}"
                )
                costs = []
            if (global_step + 1) % config.model_checkpoint_interval == 0:
                accelerator.save_model(
                    generator,
                    save_directory=os.path.join(
                        config.model_checkpoint_path,
                        f"{config.run_name}_generator_{epoch}_{step}",
                    ),
                )
                accelerator.save_model(
                    discriminator,
                    save_directory=os.path.join(
                        config.model_checkpoint_path,
                        f"{config.run_name}_discriminator_{epoch}_{step}",
                    ),
                )
            if accelerator.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

            global_step += 1


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


def main(config: TrainingConfig) -> None:
    """Main function for training StreamVC model."""
    logger.debug(f"DEVICE={accelerator.device}")
    hps = get_flattened_config_dict(config)
    hps["num processes"] = accelerator.num_processes
    hps["mixed precision"] = accelerator.mixed_precision
    hps["gradient accumulation steps"] = accelerator.gradient_accumulation_steps

    if accelerator.gradient_accumulation_steps > 1 and isinstance(
        config, DecoderTrainingConfig
    ):
        raise ValueError(
            "Gradient accumulation is not supported for the decoder training. Disable gradient accumulation from accelerate"
        )
    logger.debug(f"{hps=}")

    accelerator.init_trackers(config.run_name, config=hps)
    streamvc = StreamVC(gradient_checkpointing=config.gradient_checkpointing)

    if isinstance(config, ContentEncoderTrainingConfig):
        content_encoder = streamvc.content_encoder
        train_content_encoder(content_encoder, config)
    else:
        wrapped_encoder_state_dict = st.torch.load_file(
            config.content_encoder_checkpoint
        )
        encoder_state_dict = {
            key[len("encoder.") :]: value
            for key, value in wrapped_encoder_state_dict.items()
            if key.startswith("encoder.")
        }
        streamvc.content_encoder.load_state_dict(encoder_state_dict)
        train_streamvc(streamvc, config)

    accelerator.end_training()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    config = tyro.cli(TrainingConfig)
    main(config)
