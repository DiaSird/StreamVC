SHELL=/bin/bash

# Checkpoints
ENC_CKPT="./checkpoints/best_svc_111_content_encoder/model.safetensors"
# ENC_CKPT="./checkpoints/svc_111_state/model.safetensors"

CKPT=checkpoints/svc_112_state/model.safetensors

# Audio
SRC=test-audio/src.wav
TGT=test-audio/tgt.wav
OUT=test-audio/out.wav

# Dataset Audio Files
tr100="./dataset/train.clean.100"
tr360="./dataset/train.clean.360"
tr500="./dataset/train.other.500"
DATASETS ?= $(tr100)

default: infer

# Preprocessing Dataset
pre-train100:
	python preprocess_dataset.py --split train.clean.100
pre-train360:
	python preprocess_dataset.py --split train.clean.360
pre-train500:
	python preprocess_dataset.py --split train.other.500
pre-dev:
	python preprocess_dataset.py --split dev.clean
	python preprocess_dataset.py --split dev.other
pre-test:
	python preprocess_dataset.py --split test.clean
	python preprocess_dataset.py --split test.other

pre-all: pre-train pre-dev pre-test

# Content Encoder training
enc:
	accelerate launch \
		train.py content-encoder \
		--run-name svc_111 \
		--restore-state-dir ./checkpoints/svc_111_state \
		--batch_size 4 \
		--num-epochs 7 \
		--datasets.train-dataset-path ${DATASETS}  \
		--model-checkpoint-interval 500 \
		--accuracy-interval 200 \
		lr-scheduler:one-cycle-lr \
		--lr-scheduler.div_factor 15 \
		--lr-scheduler.final_div_factor 1000 \
		--lr-scheduler.max 1e-4 \
		--lr-scheduler.pct_start 0.2

# Decoder training, it is required to pass the Content Encoder checkpoint
# --lambda_feature 100 --lambda_reconstruction 1
# --lambda_adversarial 1 --lr_discriminator_multiplier 1.0
dec:
	accelerate launch \
		train.py decoder \
		--run-name svc_112 \
		--batch_size 4 \
		--num-epochs 5 \
		--lr 1e-5 \
		--restore-state-dir ./checkpoints/svc_112_state \
		--lambda_feature 50 --lambda_reconstruction 1 --lambda_adversarial 0.5 \
		--datasets.train-dataset-path ${DATASETS} \
		--model-checkpoint-interval 500 \
		--log-gradient-interval 500 \
		--content-encoder-checkpoint ${ENC_CKPT} \
		lr-scheduler:cosine-annealing-warm-restarts \
		--lr-scheduler.T-0 3000

infer:
	python inference.py -c ${CKPT} -s ${SRC} -t ${TGT} -o ${OUT}

stream:
	python inference.py --stream -c ${CKPT} -s ${SRC} -t ${TGT} -o ${OUT}

log:
	tensorboard --logdir logs
