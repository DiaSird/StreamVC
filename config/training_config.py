from dataclasses import dataclass
from .lr_scheduler import LRScheduler
import tyro


@dataclass(frozen=True)
class DatasetsConfig:
    train_dataset_path: str | list[str] = "./dataset/train.clean.100"
    """Path to the preprocessed training dataset"""
    dev_dataset_path: str | list[str] = "./dataset/dev.clean"
    """Path to the preprocessed development dataset"""
    test_dataset_path: str | list[str] = "./dataset/test.clean"
    """Path to the preprocessed test dataset"""


@dataclass(frozen=True)
class TrainingBaseConfig:
    run_name: str
    """Name of the training run for identification purposes"""

    datasets: DatasetsConfig

    batch_size: int = 32
    """Batch size for training"""

    limit_num_batches: int | None = None
    """Limit the number of batches per epoch. Use None for no limit"""

    limit_batch_samples: int | None = 16_000 * 10
    """Limit the number of samples for audio signal in the batch"""

    num_epochs: int = 1
    """Number of epochs for training"""

    lr: float = 4e-3
    """Learning rate for the optimizer"""

    betas: tuple[float, float] = (0.5, 0.9)
    """Beta parameters for the AdamW optimizer"""

    weight_decay: float = 1e-2
    """Weight decay for the AdamW optimizer"""

    gradient_checkpointing: bool = True
    """Enable gradient checkpointing to reduce memory usage"""

    lr_scheduler: LRScheduler | None = None
    """Learning rate scheduler for the optimizer"""

    model_checkpoint_interval: int = 100
    """Interval (in steps) at which to save model checkpoints"""

    model_checkpoint_path: str = "./checkpoints"
    """Path to save model checkpoints"""

    print_interval: int = 20
    """Interval (in steps) at which to print training metrics"""

    log_gradient_interval: int | None = None
    """Interval (in steps) at which to log gradient information. Use None to disable"""

    restore_state_dir: str | None = None
    """Path to restore model (+optimizer, scheduler etc.) state from. Use None to start from scratch"""


@dataclass(frozen=True)
class ContentEncoderTrainingConfig(TrainingBaseConfig):
    """Training configuration for the content encoder module"""

    dropout: float = 0.1
    """Dropout rate"""

    accuracy_interval: int = 100
    """Interval (in steps) at which to compute and log accuracy"""

    accuracy_limit_num_batches: int | None = 20
    """Limit the number of batches to compute accuracy. Use None for no limit"""

    log_labels_interval: int | None = None
    """Interval (in steps) at which to log label information. Use None to disable"""


@dataclass(frozen=True)
class DecoderTrainingConfig(TrainingBaseConfig):
    """Training configuration for the decoder module"""

    content_encoder_checkpoint: str = tyro.MISSING
    """Path to the content encoder checkpoint"""

    lambda_feature: float = 100.0
    """Weight of the feature matching loss"""

    lambda_reconstruction: float = 1.0
    """Weight of the reconstruction loss"""

    lambda_adversarial: float = 1.0
    """Weight of the adversarial loss"""

    lr_discriminator_multiplier: float = 1.0
    """Learning rate multiplier for the discriminator. Use 1.0 for same as generator"""
