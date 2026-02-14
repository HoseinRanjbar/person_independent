# Signer-Independent Sign Language Recognition (ISLR) via Contrastive Learning

PyTorch training code for **signer-independent sign language recognition** using a **transformer-like temporal model** with **supervised contrastive learning (SupCon-style)**. The code supports:
- **Joint training**: cross-entropy (CE) + contrastive loss together
- **Two-stage training**: contrastive pretraining → CE fine-tuning

---

## Key idea

To improve **signer-independence**, we encourage embeddings of the **same sign class** (from different signers/instances) to be close while separating different classes using a **supervised contrastive loss**. The model outputs:
- class logits for recognition (CE loss)
- a projection head output for contrastive training (SupCon)

  <img width="924" height="545" alt="contrastive_learning_img" src="https://github.com/user-attachments/assets/85819f90-27da-4022-98a1-5a12825cad04" />


The figure shows contrastive sampling for signer-independent SLR.
Positive group (green): different signers performing the same class (embeddings should be pulled together).
Negative group (red): samples from a different class (embeddings should be pushed apart).

---

## Features

- Transformer-like sequence model for sign video recognition
- Image embedding backbones (e.g., `mb2` / MobileNetV2-style)
- Supervised contrastive loss with temperature scaling
- Two-stage or joint training modes
- Multi-GPU support via `torch.nn.DataParallel`
- Checkpointing + best-model saving
- Learning curve plots saved to the experiment folder

---

## Install dependencies

```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib progressbar2 gputil
```

## Example command (two-stage training)

```bash
python -m train_cl_ga.py \
  --data data/ISLR101-DATASET/Sign_Language_Dataset \
  --dp_keep_prob 0.70 \
  --freeze_cnn False \
  --num_epochs 70 \
  --batch_size 32 \
  --samples_per_class 4 \
  --training_mode two-stage \
  --initial_lr 0.00001 \
  --num_classes 101 \
  --pretrain_epochs 20 \
  --proj_dim 512 \
  --emb_network mb2 \
  --hidden_size 1280 \
  --save_dir ./output/contrastive_learning_2stages
```
