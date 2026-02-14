import argparse
import time
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import datetime as dt
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import progressbar
import matplotlib.pyplot as plt
import GPUtil

from transformer_cl import make_model as TRANSFORMER
from dataloader_cl import loader
from tools.utils import Batch, NoamOpt
from tools.viz import learning_curve_slr
###
# Arg parsing
##############

parser = argparse.ArgumentParser(description='Training the transformer-like network')

parser.add_argument('--data', type=str, default=os.path.join('data','phoenix-2014.v3','phoenix2014-release','phoenix-2014-multisigner'),
                   help='location of the data corpus')

parser.add_argument('--remove_bg_training',type=bool, default= False)

parser.add_argument('--remove_bg_test',type=bool, default= False)

parser.add_argument("--local-rank", type=int, default=0)

parser.add_argument('--fixed_padding', type=int, default=None,
                    help='None/64')

parser.add_argument('--num_classes', type=int)

parser.add_argument('--classifier_hidden_size', type=int, default=512)

# parser.add_argument('--lookup_table', type=str, default=os.path.join('data','slr_lookup.txt'),
#                     help='location of the words lookup table')

parser.add_argument('--rescale', type=int, default=224,
                    help='rescale data images.')

parser.add_argument('--random_drop_probability', type=float, default=0.5,
                    help='probability of frame random drop/0-1 or None')

parser.add_argument('--uniform_drop_probability', type=float, default=None,
                    help='probability of frame random drop/0-1 or None')

#Put to 0 to avoid memory segementation fault
parser.add_argument('--num_workers', type=int, default=10,
                    help='NOTE: put num of workers to 0 to avoid memory saturation.')

parser.add_argument('--show_sample', action='store_true',
                    help='Show a sample a preprocessed data (sequence of image of sign + translation).')

parser.add_argument('--optimizer', type=str, default='ADAM',
                    help='optimization algo to use; SGD, SGD_LR_SCHEDULE, ADAM / NOAM')

parser.add_argument('--scheduler', type=str, default='multi-step',
                    help='Type of scheduler, multi-step or stepLR')

parser.add_argument('--milestones', default="15,30", type=str,
                    help="milestones for MultiStepLR or stepLR")

parser.add_argument('--weight_decay', type= float , default = 5e-5)

parser.add_argument('--batch_size', type=int, default=16,
                    help='size of one minibatch')

parser.add_argument('--samples_per_class', type=int, default=2)

parser.add_argument('--initial_lr', type=float, default=0.0001,
                    help='initial learning rate')

parser.add_argument('--hidden_size', type=int, default=1280,
                    help='size of hidden layers. NOTE: This must be a multiple of n_heads.')

parser.add_argument('--num_layers', type=int, default=2,
                    help='number of transformer blocks')

parser.add_argument('--n_heads', type=int, default=8,
                    help='number of self attention heads')

#Pretrained weights
parser.add_argument('--pretrained', type=bool, default=True,
                    help='embedding layers are pretrained using imagenet')

parser.add_argument('--full_pretrained', type=str, default=None,
                    help='Full frame CNN pretrained')

parser.add_argument('--hand_pretrained', type=str, default=None,
                    help='Hand regions CNN pretrained')

parser.add_argument('--hand_query', action='store_true',
                    help='Set hand as a query for transformer network.')

parser.add_argument('--emb_type', type=str, default='2d',
                    help='Type of image embeddings 2d or 3d.')

parser.add_argument('--emb_network', type=str, default='mb2',
                    help='Image embeddings network: mb2/i3d/m3d')

parser.add_argument('--image_type', type=str, default='rgb',
                    help='Train on rgb/grayscale images')

parser.add_argument('--num_epochs', type=int, default=100,
                    help='number of epochs to stop after')

parser.add_argument('--dp_keep_prob', type=float, default=0.8,
                    help='dropout *keep* probability. drop_prob = 1-dp_keep_prob \
                    (dp_keep_prob=1 means no dropout)')

parser.add_argument('--proj_dim', type=int, default=128,
                    help='Projection dimension for contrastive head.')

parser.add_argument('--training_mode', type=str, default='joint',
                    choices=['joint', 'two-stage'],
                    help='Training strategy: '
                         '"joint" = CE + contrastive together, '
                         '"two-stage" = contrastive pretrain then CE fine-tune.')

parser.add_argument('--pretrain_epochs', type=int, default=10,
                    help='For two-stage: number of epochs for contrastive-only pretraining.')

parser.add_argument('--contrastive_weight', type=float, default=0.5,
                    help='Weight for contrastive loss in joint mode.')

parser.add_argument('--contrastive_temperature', type=float, default=0.07,
                    help='Temperature parameter for supervised contrastive loss.')

parser.add_argument('--valid_steps', type=int, default=2, help='Do validation each valid_step')

parser.add_argument('--save_steps', type=int, default=10, help='Save model after each N epoch')

parser.add_argument('--debug', default=None)

parser.add_argument('--save_dir', type=str, default='EXPERIMENTATIONS',
                    help='path to save the experimental config, logs, model')

parser.add_argument('--evaluate', action='store_true',
                    help="Evaluate dev set using bleu metric each epoch.")

parser.add_argument('--d_ff', type=int,default=2048)

parser.add_argument('--resume', default=False,
                    help="Resume training from a checkpoint.")
                    
parser.add_argument('--checkpoint',type=str, default=None,
                    help="resume training from a previous checkpoint.")

parser.add_argument('--rel_window', type=int, default=None,
                    help="Use local masking window.")

#Training settings
parser.add_argument('--freeze_cnn', default= False,
                    help='freeze the feature extractor (CNN)!')

parser.add_argument('--data_stats', type=str, default=None,
                    help="Normalize images using the dataset stats (mean/std).")

parser.add_argument('--hand_stats', type=str, default=None,
                    help="Normalize images using the dataset stats (mean/std).")


#----------------------------------------------------------------------------------------


## SET EXPERIMENTATION AND SAVE CONFIGURATION

#Same seed for reproducibility)
parser.add_argument('--seed', type=int, default=1111, help='random seed')

#Save folder with the date
start_date = dt.datetime.now().strftime("%Y-%m-%d-%H.%M")
print ("Start Time: "+start_date)

args = parser.parse_args()

#Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

#experiment_path = PureWindowsPath('EXPERIMENTATIONS\\' + start_date)
experiment_path = os.path.join(args.save_dir,f"{start_date}")

# Creates an experimental directory and dumps all the args to a text file
if(os.path.exists(experiment_path)):
    print('Experiment already exists..')
    quit(0)
else:
    os.makedirs(experiment_path)

print ("\nPutting log in EXPERIMENTATIONS/%s"%start_date)

args.save_dir = os.path.join(args.save_dir, start_date)

#Dump all configurations/hyperparameters in txt
with open (os.path.join(experiment_path,'exp_config.txt'), 'w') as f:
    f.write('Experimentation done at: '+ str(start_date)+' with current configurations:\n')
    for arg in vars(args):
        f.write(arg+' : '+str(getattr(args, arg))+'\n')

# ----------------- DEVICE SETUP -----------------

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    print("WARNING: Training on CPU. This may be very slow.")
    device = torch.device("cpu")

n_devices = max(1, torch.cuda.device_count())
print(f"Number of GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Using device: {torch.cuda.get_device_name(0)}")

# ----------------- LOSS: SUPERVISED CONTRASTIVE -----------------

def supervised_contrastive_loss(features, labels, temperature=0.07):
    """
    Supervised contrastive loss (SupCon-style).
    features: [B_global, D] (L2-normalized)
    labels:   [B_global]
    """
    device = features.device
    labels = labels.contiguous().view(-1, 1)  # [B, 1]
    batch_size = features.shape[0]

    if batch_size < 2:
        return torch.tensor(0.0, device=device)

    mask = torch.eq(labels, labels.T).float().to(device)  # [B, B]

    # remove self-contrast
    logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
    mask = mask * logits_mask

    # similarity matrix
    logits = torch.matmul(features, features.T) / temperature  # [B, B]

    # numerical stability
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    pos_counts = mask.sum(1)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (pos_counts + 1e-12)

    valid = pos_counts > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)
    
    if args.debug:  # uses global args
        num_anchors_with_pos = (pos_counts > 0).sum().item()
        print(f"[DEBUG SupCon] batch_size={batch_size}, "
              f"anchors with >=1 positive={num_anchors_with_pos}")

    loss = -mean_log_prob_pos[valid].mean()
    return loss


# ----------------- ONE EPOCH -----------------

def run_epoch(model, data, is_train=False, device=None,
              epoch=0, training_mode='joint',
              pretrain_epochs=0,
              contrastive_weight=1.0,
              contrastive_temperature=0.07):
    
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    phase = 'train' if is_train else 'valid'
    model.train() if is_train else model.eval()
    print("Training.." if is_train else "Evaluating..")

    start_time = time.time()
    total_loss = 0.0
    total_accuracy = 0.0
    count = 0

    ce_loss_fn = nn.CrossEntropyLoss()

    bar = progressbar.ProgressBar(
        maxval=dataset_sizes[phase],
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]
    )
    bar.start()
    processed = 0

    #Loop over minibatches
    for step, (x, x_lengths, y, hand_regions, hand_lengths) in enumerate(data):
        processed += len(x)
        bar.update(processed)

        x = x.to(device)
        y = y.to(device).squeeze(1)

        batch = Batch(
            x_lengths,
            hand_lengths,
            trg=None,
            emb_type=args.emb_type,
            DEVICE=device,
            fixed_padding=args.fixed_padding,
            rel_window=args.rel_window
        )

        if is_train:
            model.zero_grad()

        # Forward
        comb_out, class_logits, output_hand, proj = model(
            x, batch.src_mask, batch.rel_mask, hand_regions
        )

        # CE loss
        ce_loss = ce_loss_fn(class_logits, y)

        # Total loss
        if is_train:
            if training_mode == 'joint':
                con_loss = supervised_contrastive_loss(
                    proj, y, temperature=contrastive_temperature
                )
                if args.debug and step == 0:
                    print(f"[DEBUG] joint mode: CE={ce_loss.item():.4f}, Con={con_loss.item():.4f}")
                loss = ce_loss + contrastive_weight * con_loss

            elif training_mode == 'two-stage':
                if epoch < pretrain_epochs:
                    con_loss = supervised_contrastive_loss(
                        proj, y, temperature=contrastive_temperature
                    )
                    if args.debug and step == 0:
                        print(f"[DEBUG] pretrain (epoch {epoch}): Con={con_loss.item():.4f} (CE ignored)")
                    loss = con_loss
                else:
                    if args.debug and step == 0:
                        print(f"[DEBUG] fine-tune (epoch {epoch}): CE={ce_loss.item():.4f}")
                    loss = ce_loss
            else:
                loss = ce_loss
        else:
            loss = ce_loss  # validation: CE only

        if args.debug and step == 0:
            labels_np = y.detach().cpu().numpy()
            unique, counts = np.unique(labels_np, return_counts=True)
            print("\n[DEBUG] First batch labels:", labels_np.tolist())
            print("[DEBUG] Label counts in first batch:", dict(zip(unique, counts)))

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

        # Accuracy
        _, preds = torch.max(class_logits, 1)
        accuracy = (preds == y.data).float().mean().item()

        total_loss += loss.item()
        total_accuracy += accuracy
        count += 1

    avg_loss = total_loss / count
    avg_accuracy = total_accuracy / count

    if is_train:
        print(f"Average Training Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
    else:
        print(f"Average Validation Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    return avg_loss, avg_accuracy

# ----------------- DATALOADERS -----------------

batch_size =args.batch_size

if args.image_type == 'rgb':
    channels = 3
elif args.image_type == 'grayscale':
    channels = 1
else:
    print('Image type is not supported!')
    quit(0)

train_csv = pd.read_csv(os.path.join(args.data, 'train-dataset.csv'))
test_csv = pd.read_csv(os.path.join(args.data, 'test-dataset.csv'))
val_csv = pd.read_csv(os.path.join(args.data, 'val-dataset.csv'))

with open('./tools/data/lookup_table.json', 'r') as file:
    lookup_table = json.load(file)

if args.data_stats:
    args.data_stats = torch.load(args.data_stats, map_location=torch.device('cpu'))
if args.hand_stats:
    args.hand_stats = torch.load(args.hand_stats, map_location=torch.device('cpu'))

train_dataloader, train_size = loader(
    csv_file=train_csv,
    root_dir=args.data,
    lookup_table=lookup_table,
    local_rank=0,
    remove_bg=args.remove_bg_training,
    rescale=args.rescale,
    batch_size=batch_size,
    samples_per_class=args.samples_per_class,
    num_workers=args.num_workers,
    random_drop=args.random_drop_probability,
    uniform_drop=args.uniform_drop_probability,
    show_sample=args.show_sample,
    debug=args.debug,
    istrain=True,
    fixed_padding=args.fixed_padding,
    hand_dir=None,
    data_stats=args.data_stats,
    hand_stats=args.hand_stats,
    channels=channels
)

valid_dataloader, valid_size = loader(
    csv_file=val_csv,
    root_dir=args.data,
    lookup_table=lookup_table,
    local_rank=0,
    remove_bg=args.remove_bg_test,
    rescale=args.rescale,
    batch_size=args.batch_size,
    samples_per_class=args.samples_per_class,
    num_workers=args.num_workers,
    random_drop=args.random_drop_probability,
    uniform_drop=args.uniform_drop_probability,
    show_sample=args.show_sample,
    istrain=False,
    fixed_padding=args.fixed_padding,
    hand_dir=None,
    data_stats=args.data_stats,
    hand_stats=args.hand_stats,
    channels=channels
)

print('Dataset sizes:')
dataset_sizes = {'train': train_size, 'valid': valid_size}
print(dataset_sizes)

# ----------------- MODEL & OPTIMIZER -----------------

model = TRANSFORMER(
    num_classes=args.num_classes,
    n_stacks=args.num_layers,
    n_units=args.hidden_size,
    n_heads=args.n_heads,
    d_ff=args.d_ff,
    dropout=1.-args.dp_keep_prob,
    image_size=args.rescale,
    pretrained=args.pretrained,
    classifier_hidden_dim=args.classifier_hidden_size,
    emb_type=args.emb_type,
    emb_network=args.emb_network,
    full_pretrained=args.full_pretrained,
    hand_pretrained=args.hand_pretrained,
    freeze_cnn=args.freeze_cnn,
    channels=channels,
    proj_dim=args.proj_dim
)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Model parameters:', trainable_params)

if args.optimizer == 'ADAM':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
elif args.optimizer == 'noam':
    optimizer = NoamOpt(
        args.hidden_size, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    )
else:
    raise ValueError("Unknown optimizer type")

num_epochs = 1 if args.debug else args.num_epochs

# Scheduler & checkpoint resume
if args.checkpoint:
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint['best_accuracy']
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
else:
    start_epoch = 0
    if args.scheduler == 'multi-step':
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    elif args.scheduler == 'stepLR':
        scheduler = StepLR(optimizer, step_size=int(args.milestones), gamma=0.1)
    else:
        scheduler = None
        print('No scheduler!')

model = model.to(device)

if torch.cuda.device_count() > 1:
    print("Using", n_devices, "GPUs with DataParallel!")
    model = nn.DataParallel(model)
else:
    print("Training using 1 device (GPU/CPU)")

# ----------------- MAIN TRAINING LOOP -----------------

train_ppls, train_losses, train_accuracies = [], [], []
val_ppls, val_losses, val_accuracies, times = [], [], [], []
best_accuracy_so_far = 0

for epoch in range(start_epoch, num_epochs):
    start = time.time()
    print('\nEPOCH', epoch, '------------------')
    print("LR", optimizer.param_groups[0]['lr'])

    # TRAIN
    train_loss, train_accuracy = run_epoch(
        model,
        train_dataloader,
        is_train=True,
        device=device,
        epoch=epoch,
        training_mode=args.training_mode,
        pretrain_epochs=args.pretrain_epochs,
        contrastive_weight=args.contrastive_weight,
        contrastive_temperature=args.contrastive_temperature,
    )
    print("After train epoch..")
    print(GPUtil.showUtilization())

    train_ppl = np.exp(train_loss)
    if scheduler is not None:
        scheduler.step()

    # VALIDATION
    if epoch % args.valid_steps == 0:
        with torch.no_grad():
            val_loss, val_accuracy = run_epoch(
                model,
                valid_dataloader,
                is_train=False,
                device=device,
                epoch=epoch,
                training_mode=args.training_mode,
                pretrain_epochs=args.pretrain_epochs,
                contrastive_weight=args.contrastive_weight,
                contrastive_temperature=args.contrastive_temperature,
            )

            if val_accuracy > best_accuracy_so_far:
                best_accuracy_so_far = val_accuracy
                real_model = model.module if hasattr(model, "module") else model

                print("Saving entire model with best params")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': real_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'best_accuracy': best_accuracy_so_far
                }, os.path.join(experiment_path, 'BEST.pt'))

                print("Saving full-frame (CNN) with best params")
                torch.save(real_model.src_emb.state_dict(),
                           os.path.join(experiment_path, 'full_cnn_best_params.pt'))

                if args.hand_query:
                    print("Saving hand regions (CNN) with best params")
                    torch.save(real_model.hand_emb.state_dict(),
                               os.path.join(experiment_path, 'hand_cnn_best_params.pt'))

        val_ppl = np.exp(val_loss)

        # log
        train_ppls.append(train_ppl)
        val_ppls.append(val_ppl)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        times.append(time.time() - start)

        log_str = (
            f"epoch: {epoch}\t"
            f"train ppl: {train_ppl}\t"
            f"val ppl: {val_ppl}\t"
            f"train loss: {train_loss}\t"
            f"val loss: {val_loss}\t"
            f"accuracy: {val_accuracy}\t"
            f"BEST accuracy: {best_accuracy_so_far}\t"
            f"time (s): {times[-1]}"
        )
        print(log_str)
        with open(os.path.join(experiment_path, 'log.txt'), 'a') as f_:
            f_.write(log_str + '\n')

        # save curves & plots
        lc_path = os.path.join(experiment_path, 'learning_curves.npy')
        print('\nDONE\n\nSaving learning curves to', lc_path)
        np.save(lc_path, {
            'train_ppls': train_ppls,
            'val_ppls': val_ppls,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'accuracy': val_accuracies,
        })

        print("Saving plots")
        learning_curve_slr(experiment_path)

        # periodic checkpoint
        if epoch % args.save_steps == 0:
            print("Saving model parameters for epoch:", epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'best_accuracy': best_accuracy_so_far
            }, os.path.join(experiment_path, f'epoch_{epoch}_accuracy_{val_accuracy}.pt'))

        if train_ppl <= 1:
            print("Hello World ;)")
            break