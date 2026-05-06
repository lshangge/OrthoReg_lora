import json
import os

from src.args import parse_arguments
from src.eval import eval_single_dataset
from src.loraatt import LoraATTImageEncoder
from src.modeling import ImageEncoder
from src.task_vectors import NonLinearTaskVector, PEFTTaskVector

args = parse_arguments()

if args.finetuning_mode == "none":
    if args.seed is not None:
        args.save = f"checkpoints_{args.seed}/{args.model}"
    else:
        args.save = f"checkpoints/{args.model}"
elif "ortho" in args.finetuning_mode:
    args.save = f"checkpoints_{args.seed}/{args.finetuning_mode}_{args.lr}_lambda{args.ortho_lambda}_{args.model}"
else:
    if args.seed is not None:
        args.save = f"checkpoints_{args.seed}/{args.finetuning_mode}_{args.lr}_{args.model}"
    else:
        args.save = f"checkpoints/{args.finetuning_mode}_{args.lr}_{args.model}"

if args.seed is not None:
    base_model_save_path = f"checkpoints_{args.seed}/{args.model}"
else:
    base_model_save_path = f"checkpoints/{args.model}"

accuracies = {}

mode_labels = {
    "none": "Evaluating pretrained (zero-shot) models.",
    "loraatt": "Evaluating LoRA-ATT models (merged delta_W).",
    "loraatt_ortho": "Evaluating LoRA-ATT + OrthoReg models (merged delta_W).",
}
print("*" * 100)
print(mode_labels.get(args.finetuning_mode, f"Evaluating {args.finetuning_mode} models."))

datasets = ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"]

for dataset in datasets:
    print("*" * 100)
    print(f"Evaluating on {dataset}")

    mode = args.finetuning_mode
    image_encoder = None

    if mode == "none":
        pretrained_checkpoint = f"{base_model_save_path}/{dataset}Val/zeroshot.pt"
        if not os.path.exists(pretrained_checkpoint):
            print(f"Zeroshot checkpoint not found. Creating and saving pretrained model to {pretrained_checkpoint}.")
            image_encoder = ImageEncoder(args)
            os.makedirs(os.path.dirname(pretrained_checkpoint), exist_ok=True)
            image_encoder.save(pretrained_checkpoint)
        else:
            task_vector = NonLinearTaskVector(
                pretrained_checkpoint=pretrained_checkpoint,
                finetuned_checkpoint=pretrained_checkpoint,
            )
            image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.0)

    elif mode in ("loraatt", "loraatt_ortho"):
        finetuned_checkpoint = f"{args.save}/{dataset}Val/{mode}_finetuned.pt"
        base_model_checkpoint = f"{base_model_save_path}/{dataset}Val/zeroshot.pt"

        if not os.path.exists(finetuned_checkpoint):
            print(f"Error: Missing finetuned checkpoint: {finetuned_checkpoint}")
            continue
        if not os.path.exists(base_model_checkpoint):
            print(f"Error: Missing base model checkpoint: {base_model_checkpoint}")
            continue

        task_vector = PEFTTaskVector(
            finetuned_checkpoint=finetuned_checkpoint,
            peft_model_class=LoraATTImageEncoder,
            args=args,
        )
        image_encoder = task_vector.apply_to(base_model_checkpoint, scaling_coef=1.0)

    else:
        print(f"Unknown finetuning mode: {mode}")
        continue

    if image_encoder is None:
        continue

    for split in ["test", "val"]:
        print("=" * 100)
        print(f"Evaluating on {split} split.")
        eval_dataset = dataset if split == "test" else f"{dataset}Val"
        accuracies[eval_dataset] = eval_single_dataset(image_encoder, eval_dataset, args)["top1"]

# Also evaluate on ImageNet for zero-shot mode
if args.finetuning_mode == "none":
    for split in ["ImageNetVal", "ImageNet"]:
        accuracies[split] = eval_single_dataset(image_encoder, split, args)["top1"]

# Save results
save_name_map = {
    "none": "zeroshot_accuracies.json",
    "loraatt": "loraatt_ft_accuracies.json",
    "loraatt_ortho": "loraatt_ortho_ft_accuracies.json",
}
save_path = os.path.join(args.save, save_name_map[args.finetuning_mode])
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, "w") as f:
    json.dump(accuracies, f, indent=4)
print(f"Results saved to {save_path}")
