import open_clip
import torch

from src import utils


class ImageEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        print(f"Loading {args.model} pre-trained weights.")
        if "__pretrained__" in args.model:
            name, pretrained = args.model.split("__pretrained__")
        elif "__init__" in args.model:
            print("Using random initialization.")
            name, pretrained = args.model.split("__init__")[0], None
        else:
            name = args.model
            pretrained = "openai"
        (
            self.model,
            self.train_preprocess,
            self.val_preprocess,
        ) = open_clip.create_model_and_transforms(
            name, pretrained=pretrained, cache_dir=args.openclip_cachedir
        )

        self.cache_dir = args.cache_dir

        if not keep_lang and hasattr(self.model, "transformer"):
            delattr(self.model, "transformer")

    # def forward(self, images):
    #     assert self.model is not None
    #     return self.model.encode_image(images)

    # def __call__(self, inputs):
    #     return self.forward(inputs)

    def forward(self, images, calculate_ortho_loss=False, pretrained_state_dict=None):
        """
        Extended forward method to optionally compute and return the orthogonal loss.
        """
        # Original forward pass
        features = self.model.encode_image(images)

        # Return features directly if orthogonal loss is not needed
        if not calculate_ortho_loss:
            return features

        # --- Compute orthogonal loss if requested ---
        # This logic is moved here from utils.py
        if pretrained_state_dict is None:
            raise ValueError("pretrained_state_dict must be provided when calculate_ortho_loss is True")

        ortho_loss = 0.0
        # self.model is the open_clip model (e.g. ViT); iterate over its parameters
        for name, p_finetuned in self.model.named_parameters():
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
        
        return features, ortho_loss

    def __call__(self, inputs, calculate_ortho_loss=False, pretrained_state_dict=None):
        # Ensure __call__ forwards all arguments
        return self.forward(inputs, calculate_ortho_loss, pretrained_state_dict)

    def save(self, filename):
        print(f"Saving image encoder to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, model_name, filename):
        print(f"Loading image encoder from {filename}")
        state_dict = torch.load(filename, map_location="cpu")
        return cls.load(model_name, state_dict)

    @classmethod
    def load_from_state_dict(cls, model_name, state_dict):
        (
            self.model,
            self.train_preprocess,
            self.val_preprocess,
        ) = open_clip.create_model_and_transforms(
            name, pretrained=pretrained, cache_dir=args.openclip_cachedir
        )
        self.model.load_from_state_dict(state_dict)


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving classification head to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading classification head from {filename}")
        return utils.torch_load(filename)


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def freeze_head(self):
        self.classification_head.weight.requires_grad_(False)
        self.classification_head.bias.requires_grad_(False)

    # def forward(self, inputs):
    #     features = self.image_encoder(inputs)
    #     outputs = self.classification_head(features)
    #     return outputs

    # def __call__(self, inputs):
    #     return self.forward(inputs)

    def forward(self, inputs, calculate_ortho_loss=False, pretrained_state_dict=None):
        # Forward arguments to image_encoder
        encoder_output = self.image_encoder(inputs, calculate_ortho_loss, pretrained_state_dict)

        if calculate_ortho_loss:
            features, ortho_loss = encoder_output
            outputs = self.classification_head(features)
            return outputs, ortho_loss
        else:
            features = encoder_output
            outputs = self.classification_head(features)
            return outputs

    def __call__(self, inputs, calculate_ortho_loss=False, pretrained_state_dict=None):
        return self.forward(inputs, calculate_ortho_loss, pretrained_state_dict)

    def save(self, filename):
        print(f"Saving image classifier to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading image classifier from {filename}")
        return utils.torch_load(filename)


class MultiHeadImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_heads):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_heads = torch.nn.ModuleList(classification_heads)
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def freeze_head(self):
        for idx in range(len(self.classification_heads)):
            self.classification_heads[idx].weight.requires_grad_(False)
            self.classification_heads[idx].bias.requires_grad_(False)

    def forward(self, inputs, head_idx):
        features = self.image_encoder(inputs)
        outputs = self.classification_heads[head_idx](features)
        return outputs

    def __call__(self, inputs, head_idx):
        return self.forward(inputs, head_idx)

    def save(self, filename):
        print(f"Saving image classifier to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading image classifier from {filename}")
        return utils.torch_load(filename)
