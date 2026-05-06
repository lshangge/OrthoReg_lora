import json
import os

from utils import find_optimal_coef

from src.args import parse_arguments
from src.eval import evaluate_task_vector, evaluate_task_vector_at_coef
from src.loraatt import LoraATTImageEncoder
from src.task_vectors import PEFTTaskVector

args = parse_arguments()

if "ortho" in args.finetuning_mode:
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

ft_accuracies_name_map = {
    "loraatt": "loraatt_ft_accuracies.json",
    "loraatt_ortho": "loraatt_ortho_ft_accuracies.json",
}

mode_labels = {
    "loraatt": "Evaluating LoRA-ATT models (delta_W merging).",
    "loraatt_ortho": "Evaluating LoRA-ATT + OrthoReg models (delta_W merging).",
}
print("*" * 100)
print(mode_labels.get(args.finetuning_mode, f"Evaluating {args.finetuning_mode} models."))
print("*" * 100)

ft_accuracies_path = os.path.join(args.save, ft_accuracies_name_map[args.finetuning_mode])
with open(ft_accuracies_path) as f:
    args.finetuning_accuracies = json.load(f)

with open(os.path.join(base_model_save_path, "zeroshot_accuracies.json")) as f:
    pretrained_accuracies = json.load(f)

eval_datasets = ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN", "SUN397"]

task_vectors = []
mode = args.finetuning_mode

for dataset in eval_datasets:
    finetuned_checkpoint = f"{args.save}/{dataset}Val/{mode}_finetuned.pt"
    if not os.path.exists(finetuned_checkpoint):
        print(f"Warning: Missing finetuned checkpoint for {dataset}: {finetuned_checkpoint}")
        continue
    task_vectors.append(
        PEFTTaskVector(
            finetuned_checkpoint=finetuned_checkpoint,
            peft_model_class=LoraATTImageEncoder,
            args=args,
        )
    )

if not task_vectors:
    print("No task vectors were created. Exiting.")
    exit()

task_vector = sum(task_vectors)

pretrained_checkpoint = f"{base_model_save_path}/{eval_datasets[0]}Val/zeroshot.pt"
if not os.path.exists(pretrained_checkpoint):
    print(f"Error: Base pretrained checkpoint not found at {pretrained_checkpoint}")
    exit()

args.eval_datasets = [dataset + "Val" for dataset in eval_datasets]
args.control_dataset = None

val_metrics = evaluate_task_vector(
    task_vector,
    pretrained_checkpoint,
    args,
    posthoc_linearization=False,
)

optimal_coef = find_optimal_coef(
    val_metrics,
    metric="avg_normalized_top1",
    minimize=False,
)

args.eval_datasets = eval_datasets
test_metrics = evaluate_task_vector_at_coef(
    task_vector,
    pretrained_checkpoint,
    args,
    float(optimal_coef),
    posthoc_linearization=False,
)

print("=" * 100)
print(f"Optimal Coefficient: {optimal_coef}")
print(f"Test normalized accuracy: {test_metrics['avg_normalized_top1']}")
print(f"Test absolute accuracy: {test_metrics['avg_top1']}")

additive_accuracies = {"test": test_metrics, "val": val_metrics, "optimal_coef": optimal_coef}

save_name_map = {
    "loraatt": "loraatt_additions.json",
    "loraatt_ortho": "loraatt_ortho_additions.json",
}
save_file = os.path.join(args.save, save_name_map[mode])
with open(save_file, "w") as f:
    json.dump(additive_accuracies, f, indent=4)
print(f"Addition results saved to {save_file}")
