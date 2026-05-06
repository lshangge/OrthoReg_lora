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

control_dataset = "ImageNet"
negation_accuracies = {}
eval_datasets = ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"]
mode = args.finetuning_mode

for dataset in eval_datasets:
    finetuned_checkpoint = f"{args.save}/{dataset}Val/{mode}_finetuned.pt"
    base_model_checkpoint = f"{base_model_save_path}/{dataset}Val/zeroshot.pt"

    if not os.path.exists(finetuned_checkpoint):
        print(f"Warning: Missing finetuned checkpoint for {dataset}: {finetuned_checkpoint}")
        continue
    if not os.path.exists(base_model_checkpoint):
        print(f"Warning: Missing base model checkpoint for {dataset}: {base_model_checkpoint}")
        continue

    task_vector = -PEFTTaskVector(
        finetuned_checkpoint=finetuned_checkpoint,
        peft_model_class=LoraATTImageEncoder,
        args=args,
    )

    args.eval_datasets = [dataset + "Val"]
    args.control_dataset = control_dataset + "Val"
    val_metrics = evaluate_task_vector(
        task_vector,
        base_model_checkpoint,
        args,
        posthoc_linearization=False,
    )

    optimal_coef = find_optimal_coef(
        val_metrics,
        metric=f"{dataset}Val:top1",
        minimize=True,
        control_metric=f"{control_dataset}Val:top1",
        control_metric_threshold=args.control_threshold * pretrained_accuracies[control_dataset + "Val"],
    )

    args.eval_datasets = [dataset]
    args.control_dataset = control_dataset
    test_metrics = evaluate_task_vector_at_coef(
        task_vector,
        base_model_checkpoint,
        args,
        optimal_coef,
        posthoc_linearization=False,
    )

    print("=" * 100)
    print(f"Results for dataset: {dataset}")
    print(f"Optimal Coefficient: {optimal_coef}")
    print(f"Test accuracy on {dataset}: {test_metrics.get(f'{dataset}:top1', 'N/A')}")
    print(f"Control accuracy on {control_dataset}: {test_metrics.get(f'{control_dataset}:top1', 'N/A')}")

    negation_accuracies[dataset] = {
        "test": test_metrics.get(f"{dataset}:top1"),
        "test_control": test_metrics.get(f"{control_dataset}:top1"),
        "val": val_metrics,
        "optimal_coef": optimal_coef,
    }

save_name_map = {
    "loraatt": "loraatt_negations.json",
    "loraatt_ortho": "loraatt_ortho_negations.json",
}
save_file = os.path.join(args.save, save_name_map[mode])
with open(save_file, "w") as f:
    json.dump(negation_accuracies, f, indent=4)
print(f"Negation results saved to {save_file}")
