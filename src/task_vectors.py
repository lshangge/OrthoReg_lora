import abc

import torch

from src.modeling import ImageEncoder


class _TaskVector(abc.ABC):
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                pretrained_state_dict = self._load_checkpoint(pretrained_checkpoint).state_dict()
                finetuned_state_dict = self._load_checkpoint(finetuned_checkpoint).state_dict()
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in (torch.int64, torch.uint8):
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]

    @abc.abstractmethod
    def _load_checkpoint(self, checkpoint):
        raise NotImplementedError

    @abc.abstractmethod
    def _cast_to_same_type(self, other):
        raise NotImplementedError

    def __add__(self, other):
        other = self._cast_to_same_type(other)
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning: key {key} is not present in both task vectors.")
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return self.__class__(vector=new_vector)

    def __sub__(self, other):
        return self.__add__(-other)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        with torch.no_grad():
            new_vector = {key: -v for key, v in self.vector.items()}
        return self.__class__(vector=new_vector)

    def __mul__(self, other):
        with torch.no_grad():
            new_vector = {key: other * v for key, v in self.vector.items()}
        return self.__class__(vector=new_vector)

    def dot(self, other):
        other = self._cast_to_same_type(other)
        with torch.no_grad():
            dot_product = 0.0
            for key in self.vector:
                if key not in other.vector:
                    continue
                dot_product += torch.sum(self.vector[key] * other.vector[key])
        return dot_product

    def norm(self):
        return torch.sqrt(self.dot(self))

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        with torch.no_grad():
            pretrained_model = self._load_checkpoint(pretrained_checkpoint)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f"Warning: key {key} not in task vector")
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        pretrained_model.load_state_dict(new_state_dict)
        return pretrained_model


class NonLinearTaskVector(_TaskVector):
    """Task vector for standard (non-linear) finetuned models."""

    def _load_checkpoint(self, checkpoint):
        model_or_state_dict = torch.load(checkpoint, map_location="cpu")
        if isinstance(model_or_state_dict, torch.nn.Module):
            return model_or_state_dict
        from src.args import parse_arguments
        args = parse_arguments()
        model = ImageEncoder(args)
        model.load_state_dict(model_or_state_dict)
        return model

    def _cast_to_same_type(self, other):
        if isinstance(other, NonLinearTaskVector):
            return other
        raise TypeError(f"Cannot operate between NonLinearTaskVector and {type(other)}")


class PEFTTaskVector(_TaskVector):
    """Task vector for LoRA-ATT models.

    Stores the equivalent dense delta_W extracted from LoRA parameters,
    and applies it to a standard (non-PEFT) pretrained model.
    """

    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None,
                 vector=None, peft_model_class=None, args=None):
        if vector is not None:
            self.vector = vector
            return

        assert finetuned_checkpoint is not None
        assert peft_model_class is not None
        assert args is not None

        peft_model = peft_model_class.load(finetuned_checkpoint, args)

        if hasattr(peft_model, "get_delta_w_dict"):
            self.vector = peft_model.get_delta_w_dict()
        else:
            self.vector = {}
            with torch.no_grad():
                for name, module in peft_model.model.named_modules():
                    if not hasattr(module, "calculate_delta_w"):
                        continue
                    delta_W = module.calculate_delta_w()
                    if "lora_in_proj_layer" in name or "in_proj_layer" in name:
                        attn_pos = name.rfind(".attn.")
                        if attn_pos != -1:
                            base = name[:attn_pos]
                            key = f"model.{base}.attn.in_proj_weight"
                        else:
                            key = f"model.{name}.weight"
                    else:
                        key = f"model.{name}.weight"
                    self.vector[key] = delta_W

    def __add__(self, other):
        other = self._cast_to_same_type(other)
        all_keys = set(self.vector.keys()) | set(other.vector.keys())
        with torch.no_grad():
            new_vector = {}
            for key in all_keys:
                v1 = self.vector.get(key, 0.0)
                v2 = other.vector.get(key, 0.0)
                new_vector[key] = v1 + v2
        return self.__class__(vector=new_vector)

    def __neg__(self):
        with torch.no_grad():
            new_vector = {key: -v for key, v in self.vector.items()}
        return self.__class__(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply merged delta_W to a standard (non-PEFT) pretrained model."""
        from src.args import parse_arguments
        args = parse_arguments()
        base_model = ImageEncoder(args)
        state_dict_or_model = torch.load(pretrained_checkpoint, map_location="cpu")
        if hasattr(state_dict_or_model, "state_dict"):
            base_model.load_state_dict(state_dict_or_model.state_dict())
        else:
            base_model.load_state_dict(state_dict_or_model)

        with torch.no_grad():
            new_state_dict = base_model.state_dict()
            for key, delta_W in self.vector.items():
                if key in new_state_dict:
                    new_state_dict[key] = new_state_dict[key] + scaling_coef * delta_W
                else:
                    print(f"Warning: key {key} from task vector not found in base model.")
        base_model.load_state_dict(new_state_dict)
        return base_model

    def _load_checkpoint(self, checkpoint):
        raise NotImplementedError("PEFTTaskVector uses a custom apply_to method.")

    def _cast_to_same_type(self, other):
        if not isinstance(other, PEFTTaskVector):
            raise TypeError(f"Cannot operate between PEFTTaskVector and {type(other)}")
        return other
