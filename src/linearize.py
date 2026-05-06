import abc
import os

import torch
import torch.nn as nn
from functorch import jvp, make_functional_with_buffers

from src.modeling import ImageEncoder
from src.utils import DotDict


class LinearizedModel(nn.Module):
    """Creates a linearized version of a nn.Module.

    The linearized version of a model is a proper PyTorch model and can be
    trained as any other nn.Module.

    Args:
        model (nn.Module): The model to linearize. The trainable parameters of
            the linearized model will be initialized to the parameters of this
            model.
        init_model (nn.Module): A model of the same type as `model` containing
            the parameters around which the model is initialized. If not
            provided, `model` is used as the initialization model.
    """

    def __init__(self, model: nn.Module, init_model: nn.Module = None) -> None:
        """Initializes the linearized model."""
        super().__init__()
        if init_model is None:
            init_model = model

        func0, params0, self.buffers0 = make_functional_with_buffers(
            init_model.eval(), disable_autograd_tracking=True
        )
        self.func0 = lambda params, x: func0(params, self.buffers0, x)

        _, params, _ = make_functional_with_buffers(
            model, disable_autograd_tracking=True
        )

        self.params = nn.ParameterList(params)
        self.params0 = nn.ParameterList(params0)
        self._model_name = model.__class__.__name__

        # The intial parameters are not trainable.
        for p in self.params0:
            p.requires_grad = False

        # The params are.
        for p in self.params:
            p.requires_grad = True

    def forward(self, x, calculate_ortho_loss=False):
        """
        Computes the linearized model output and optionally the orthogonality loss.
        The method name is changed from __call__ to forward for clarity with DDP.
        """
        dparams = [p - p0 for p, p0 in zip(self.params, self.params0)]
        out, dp = jvp(
            lambda param: self.func0(param, x),
            (tuple(self.params0),),
            (tuple(dparams),),
        )
        output = out + dp

        if not calculate_ortho_loss:
            return output
        
        # --- Integrate Orthogonality Loss Calculation ---
        ortho_loss = 0.0
        for p_finetuned, p_pretrained in zip(self.params, self.params0):
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
        
        return output, ortho_loss

    # for tau_jp
    def dp(self, x) -> torch.Tensor:
        """
        Computes only the change in output (JVP) due to parameter shift.
        Used for the 'Reg' penalty calculation.
        """
        dparams = [p - p0 for p, p0 in zip(self.params, self.params0)]
        _, dp = jvp(
            lambda param: self.func0(param, x),
            (tuple(self.params0),),
            (tuple(dparams),),
        )
        return dp
    def __call__(self, x, calculate_ortho_loss=False):
        return self.forward(x, calculate_ortho_loss)


class LinearizedImageEncoder(abc.ABC, nn.Module):
    """Creates a linearized version of an image encoder."""

    def __init__(
        self, args=None, keep_lang=False, image_encoder=None, init_encoder=None
    ):
        super().__init__()
        if image_encoder is None:
            image_encoder = ImageEncoder(args, keep_lang)
        if init_encoder is None:
            init_encoder = image_encoder

        # Copy the attributes from the image encoder.
        self.train_preprocess = image_encoder.train_preprocess
        self.val_preprocess = image_encoder.val_preprocess
        self.cache_dir = image_encoder.cache_dir

        self._model_name = self._get_name(args.model)
        self.model = LinearizedModel(init_model=init_encoder, model=image_encoder)

    def _get_name(self, model_name):
        if "__pretrained__" in model_name:
            model_name, _ = model_name.split("__pretrained__", "")
        return model_name

    def forward(self, x, calculate_ortho_loss=False, pretrained_state_dict=None):
        # Pass the flag down to the wrapped LinearizedModel
        return self.model(x, calculate_ortho_loss=calculate_ortho_loss)

    def __call__(self, x, calculate_ortho_loss=False, pretrained_state_dict=None):
        return self.forward(x, calculate_ortho_loss, pretrained_state_dict)

    def save(self, filename):
        """Saves the linearized image encoder.

        We save the model name in the state dict so that we can load the
        correct model when loading the linearized image encoder. Directly using
        torch.save would not work becuse func0 is not serializable.

        Args:
            filename (str): The path to save the taylorized image encoder.
        """
        if os.path.dirname(filename) != "":
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        state_dict = self.state_dict()
        state_dict["model_name"] = self._model_name

        torch.save(state_dict, filename)

    @classmethod
    def load(cls, filename):
        """Loads a linearized image encoder.

        It first loads the state dict with the model name and then creates the
        correct model and loads the state dict.

        Args:
            filename (str): The path to the taylorized image encoder.

        Returns:
            LinearizedImageEncoder: The loaded taylorized image encoder.
        """
        print(f"Loading image encoder from {filename}")
        state_dict = torch.load(filename, map_location="cpu")

        # ImageEncoder expects a DotDict
        args = DotDict({"model": state_dict["model_name"]})
        taylorized_encoder = cls(args)

        # Remove the model name from the state dict so that we can load the
        # model.
        state_dict.pop("model_name")
        taylorized_encoder.load_state_dict(state_dict)
        return taylorized_encoder
