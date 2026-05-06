import os
import pickle

import numpy as np
import torch


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def torch_load_old(save_path, device=None):
    with open(save_path, "rb") as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


def torch_save(model, save_path):
    if os.path.dirname(save_path) != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model, save_path)


def torch_load(save_path, device=None):
    model = torch.load(save_path, map_location="cpu")
    if device is not None:
        model = model.to(device)
    return model


def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, "to"):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)


def get_probs(inputs, classifier):
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)


class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def find_optimal_coef(
    results,
    metric="avg_normalized_top1",
    minimize=False,
    control_metric=None,
    control_metric_threshold=0.0,
):
    best_coef = None
    if minimize:
        best_metric = 1
    else:
        best_metric = 0
    for scaling_coef in results.keys():
        if control_metric is not None:
            if results[scaling_coef][control_metric] < control_metric_threshold:
                print(f"Control metric fell below {control_metric_threshold} threshold")
                continue
        if minimize:
            if results[scaling_coef][metric] < best_metric:
                best_metric = results[scaling_coef][metric]
                best_coef = scaling_coef
        else:
            if results[scaling_coef][metric] > best_metric:
                best_metric = results[scaling_coef][metric]
                best_coef = scaling_coef
    return best_coef


def nonlinear_advantage(nonlinear_acc, linear_acc, num_classes):
    return (nonlinear_acc - linear_acc) / (1.0 - 1.0 / num_classes)


def calculate_linearized_orthogonality_loss(linearized_model):
    """Compute orthogonality loss ||delta_W^T delta_W - I||_F for a LinearizedModel."""
    ortho_loss = 0.0
    for p_finetuned, p_pretrained in zip(linearized_model.params, linearized_model.params0):
        if p_finetuned.requires_grad and p_finetuned.dim() == 2:
            delta_W = p_finetuned - p_pretrained

            rows, cols = delta_W.shape
            if rows < cols:
                mat = delta_W @ delta_W.T
                identity = torch.eye(rows, device=delta_W.device)
            else:
                mat = delta_W.T @ delta_W
                identity = torch.eye(cols, device=delta_W.device)

            ortho_loss += torch.norm(mat - identity, p='fro')

    return ortho_loss


def calculate_standard_orthogonality_loss(model, pretrained_state_dict):
    """Compute orthogonality loss ||delta_W^T delta_W - I||_F for standard/linear-2 finetuning.

    Args:
        model: DDP-wrapped ImageClassifier (ddp_model).
        pretrained_state_dict: snapshot of the pretrained model's inner ViT state_dict.
    """
    ortho_loss = 0.0

    for name, p_finetuned in model.module.image_encoder.model.named_parameters():
        if p_finetuned.requires_grad and p_finetuned.dim() == 2:
            if name in pretrained_state_dict:
                p_pretrained = pretrained_state_dict[name].to(p_finetuned.device)

                delta_W = p_finetuned - p_pretrained

                rows, cols = delta_W.shape
                if rows < cols:
                    mat = delta_W @ delta_W.T
                    identity = torch.eye(rows, device=delta_W.device)
                else:
                    mat = delta_W.T @ delta_W
                    identity = torch.eye(cols, device=delta_W.device)

                ortho_loss += torch.norm(mat - identity, p='fro')

    return ortho_loss
