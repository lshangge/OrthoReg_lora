import os
import time

import torch

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp
from src.eval import eval_single_dataset
from src.heads import get_classification_head
from src.loraatt import LoraATTImageEncoder
from src.modeling import ImageClassifier, ImageEncoder
from src.utils import LabelSmoothing, accuracy, cosine_lr


def finetune(rank, args):
    setup_ddp(rank, args.world_size, port=args.port)

    train_dataset = args.train_dataset
    ckpdir = os.path.join(args.save, train_dataset)

    assert args.finetuning_mode in ("loraatt", "loraatt_ortho"), \
        f"Unsupported finetuning mode: {args.finetuning_mode}"

    is_ortho = args.finetuning_mode == "loraatt_ortho"
    mode_prefix = args.finetuning_mode  # "loraatt" or "loraatt_ortho"

    ft_path = os.path.join(ckpdir, f"{mode_prefix}_finetuned.pt")
    # LoRA-ATT saves the PEFT encoder; no separate zeroshot needed (zeroshot is the base model)
    zs_path = os.path.join(ckpdir, f"{mode_prefix}_zeroshot.pt")

    if os.path.exists(zs_path) and os.path.exists(ft_path):
        print(f"Skipping fine-tuning because {ft_path} exists.")
        return zs_path, ft_path

    assert train_dataset is not None, "Please provide a training dataset."

    if args.load is not None and args.load.endswith("pt"):
        image_encoder = LoraATTImageEncoder.load(args.load, args)
    else:
        print("Building LoRA-ATT image encoder.")
        image_encoder = LoraATTImageEncoder(args, keep_lang=False)

    classification_head = get_classification_head(args, train_dataset)
    model = ImageClassifier(image_encoder, classification_head)
    model.freeze_head()
    model = model.cuda()

    preprocess_fn = model.train_preprocess
    print_every = 100

    dataset = get_dataset(
        train_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)
    num_batches = len(dataset.train_loader)

    ddp_loader = distribute_loader(data_loader)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        find_unused_parameters=True,
        output_device=rank,
    )

    loss_fn = LabelSmoothing(args.ls) if args.ls > 0 else torch.nn.CrossEntropyLoss()

    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scheduler = cosine_lr(
        optimizer,
        args.lr,
        args.warmup_length,
        args.epochs * num_batches // args.num_grad_accumulation,
    )

    if args.save is not None and is_main_process():
        os.makedirs(ckpdir, exist_ok=True)
        # Save the initial (unfinetuned) LoRA encoder as zeroshot reference
        ddp_model.module.image_encoder.save(zs_path)

    for epoch in range(args.epochs):
        ddp_model.train()

        for i, batch in enumerate(ddp_loader):
            start_time = time.time()
            step = (
                i // args.num_grad_accumulation
                + epoch * num_batches // args.num_grad_accumulation
            )

            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()
            labels = batch["labels"].cuda()
            data_time = time.time() - start_time

            logits = ddp_model(inputs)
            classification_loss = loss_fn(logits, labels)

            ortho_loss = 0.0
            if is_ortho and args.ortho_lambda > 0:
                ortho_loss = ddp_model.module.image_encoder.calculate_total_orthogonality_loss()

            loss = classification_loss + args.ortho_lambda * ortho_loss

            (acc1,) = accuracy(logits, labels, topk=(1,))
            acc1 /= labels.size(0)

            loss.backward()

            if (i + 1) % args.num_grad_accumulation == 0:
                scheduler(step)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            batch_time = time.time() - start_time

            if (
                args.checkpoint_every > 0
                and step % args.checkpoint_every == 0
                and is_main_process()
            ):
                ckpt_path = os.path.join(ckpdir, f"{mode_prefix}_checkpoint_{step}.pt")
                ddp_model.module.image_encoder.save(ckpt_path)

            if (
                step % print_every == 0
                and ((i + 1) % args.num_grad_accumulation == 0)
                and is_main_process()
            ):
                percent_complete = 100 * i / len(ddp_loader)
                log_msg = (
                    f"Train Epoch: {epoch} [{percent_complete:.0f}%]\t"
                    f"Total Loss: {loss.item():.6f}\t"
                    f"CE Loss: {classification_loss.item():.6f}\t"
                )
                if is_ortho and args.ortho_lambda > 0:
                    log_msg += f"Ortho Loss: {ortho_loss.item():.6f}\t"
                log_msg += f"Acc@1: {100 * acc1:.2f}%\tData (t) {data_time:.3f}"
                print(log_msg, flush=True)

    if is_main_process():
        image_encoder = ddp_model.module.image_encoder
        eval_single_dataset(image_encoder, train_dataset, args)

    if args.save is not None and is_main_process():
        image_encoder.save(ft_path)
        return zs_path, ft_path

    cleanup_ddp()


if __name__ == "__main__":
    train_datasets = [
        "Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN",
    ]
    epochs = {
        "Cars": 35, "DTD": 76, "EuroSAT": 12, "GTSRB": 11,
        "MNIST": 5, "RESISC45": 15, "SUN397": 14, "SVHN": 4,
    }

    for dataset in train_datasets:
        args = parse_arguments()

        args.epochs = epochs[dataset]
        args.train_dataset = dataset + "Val"
        args.batch_size = 64 if args.model == "ViT-L-14" else 128
        args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1

        if "ortho" in args.finetuning_mode:
            args.save = f"checkpoints_{args.seed}/{args.finetuning_mode}_{args.lr}_lambda{args.ortho_lambda}_{args.model}"
        else:
            args.save = f"checkpoints_{args.seed}/{args.finetuning_mode}_{args.lr}_{args.model}"

        print("=" * 100)
        print(f"Finetuning {args.model} on {dataset} [{args.finetuning_mode}]")
        print("=" * 100)
        torch.multiprocessing.spawn(finetune, args=(args,), nprocs=args.world_size)
