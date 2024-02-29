import torch
from torch import nn


class PatchEmbeddingsLayer(nn.Module):
    """Turns a 2D input image into a 1D sequence of flattened
    2D patch embeddings.

    Args:
        in_channels (int, optional):
            The number of color channels of the input image (default=3).
        patch_size (int, optional):
            The size of the patches to convert the input image into (default=16).
        embed_dim (int, optional):
            The embedding dimension (default=768).
    """

    def __init__(
        self, in_channels: int = 3, patch_size: int = 16, embed_dim: int = 768
    ) -> None:
        super().__init__()

        # Turns an image into a sequence of 2D patch embeddings
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

        # Flattens the 2D patch embeddings
        self.flatten_layer = nn.Flatten(start_dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Accepts an input image with size `(batch_size, color_channels,
        height, width)` and converts it to a sequence of flattened 2D patch
        embeddings with size `(batch_size, number_of_patches, embed_dim)`.

        Args:
            x: torch.Tensor
                The input image.

        Returns:
            torch.Tensor:
                The flattened patch embeddings.
        """
        return self.flatten_layer(self.conv_layer(x)).permute(0, 2, 1)


class EncoderInputEmbeddingsLayer(nn.Module):
    """Creates the input embeddings for the transformer encoder
    of the Vision Transformer architecture.

    Prepends the class token embedding to the patch embeddings
    created from the `PatchEmbeddingsLayer()` class and adds
    the position embeddings to the result.

    Args:
        in_channels (int, optional):
            The number of color channels of the input image (default=3).
        patch_size (int, optional):
            The size of the patches to convert the input image into (default=16).
        embed_dim (int, optional):
            The embedding dimension (default=768).
        img_resolution (int, optional):
            The resolution of the input image (default=224).
        dropout (float, optional):
            The dropout value (default=0.1).
    """

    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        embed_dim: int = 768,
        img_resolution: int = 224,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.patch_size: int = patch_size
        self.img_resolution: int = img_resolution
        self.num_of_patches: int = self._calculate_num_of_patches()

        self.patcher = PatchEmbeddingsLayer(
            in_channels=in_channels, patch_size=patch_size, embed_dim=embed_dim
        )

        self.class_token = nn.Parameter(
            data=torch.randn(size=(1, 1, embed_dim)), requires_grad=True
        )

        self.pos_embeddings = nn.Parameter(
            data=torch.randn(size=(1, self.num_of_patches + 1, embed_dim)),
            requires_grad=True,
        )

        self.dropout_layer = nn.Dropout(p=dropout)

    def _calculate_num_of_patches(self) -> int:
        """Returns the number of patches for a single image."""
        return int(self.img_resolution**2 // self.patch_size**2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        class_token_expanded = self.class_token.expand(size=(batch_size, -1, -1))
        return self.dropout_layer(
            torch.cat(tensors=(class_token_expanded, self.patcher(x)), dim=1)
            + self.pos_embeddings
        )


class MSABlock(nn.Module):
    """Creates a layer normalized multi-head self-attention block.

    Args:
        embed_dim (int, optional):
            The embedding dimension (default=768).
        num_heads (int, optional):
            The number of self-attention heads (default=12).
        dropout (float, optional):
            The dropout value (default=0).
    """

    def __init__(
        self, embed_dim: int = 768, num_heads: int = 12, dropout: float = 0.0
    ) -> None:
        super().__init__()

        self.norm_layer = nn.LayerNorm(normalized_shape=embed_dim)

        self.multihead_attn_layer = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm_layer(x)
        attn_output, _ = self.multihead_attn_layer(
            query=x, key=x, value=x, need_weights=False
        )
        return attn_output


class MLPBlock(nn.Module):
    """Creates a layer normalized multilayer perceptron block.

    Args:
        embed_dim (int, optional):
            The embedding dimension (default=768).
        mlp_dim (int, optional):
            The dimension of the feed forward network model (default=20248).
        dropout (float, optional):
            The dropout value (default=0.1).
    """

    def __init__(
        self, embed_dim: int = 768, mlp_dim: int = 3072, dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.norm_layer = nn.LayerNorm(normalized_shape=embed_dim)

        self.mlp_layer_stack = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=mlp_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_dim, out_features=embed_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_layer_stack(self.norm_layer(x))


class TransformerEncoderBlock(nn.Module):
    """Creates a Transformer Encoder block.

    Args:
        embed_dim (int, optional): The embedding dimension (default=768).
        num_heads (int, optional): The number of self-attention heads
            (default=12).
        mlp_dim (int, optional):
            The dimension of the feed forward network model (default=20248).
        attn_dropout (float, optional):
            The dropout value of the multi-head self-attention network model
            (default=0).
        mlp_dropout (float, optional):
            The dropout value of the feed forward network model (default=0.1).
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        attn_dropout: float = 0,
        mlp_dropout: float = 0,
    ) -> None:
        super().__init__()

        self.msa_block = MSABlock(
            embed_dim=embed_dim, num_heads=num_heads, dropout=attn_dropout
        )

        self.mlp_block = MLPBlock(
            embed_dim=embed_dim, mlp_dim=mlp_dim, dropout=mlp_dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x
        return x


class ViT(nn.Module):
    """Replicates the Vision Transformer (ViT) architecture as described
    in the paper: `AN IMAGE IS WORTH 16X16 WORDS:
    TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE <https://arxiv.org/abs/2010.11929>`_.

    Args:
        in_channels (int, optional):
            The number of color channels of the input image (default=3).
        patch_size (int, optional):
            The size of the patches to convert the input image into (default=16).
        embed_dim (int, optional):
            The embedding dimension (default=768).
        img_resolution (int, optional):
            The resolution of the input image (default=224).
        num_classes (int, optional):
            The number of output classification classes (default=1000).
        num_encoders (int, optional):
            The number of `TransformerEncoderBlock` blocks.
        num_heads (int, optional):
            The number of self-attention heads (default=12).
        mlp_dim (int, optional):
            The dimension of the feed forward network model (default=20248).
        embedding_dropout (float, optional):
            The dropout value of the input patch embeddings (default=0.1).
        attn_dropout (float, optional):
            The dropout value of the multi-head self-attention network model
            (default=0).
        mlp_dropout (float, optional):
            The dropout value of the feed forward network model (default=0.1).
    """

    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        embed_dim: int = 768,
        img_resolution: int = 224,
        num_classes: int = 1000,
        num_encoders: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        embedding_dropout: float = 0.1,
        attn_dropout: float = 0,
        mlp_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Make an assertion that the image size is compatible with the patch size
        assert img_resolution % patch_size == 0, (
            f"Input image size must be divisible by patch size. "
            f"image shape: {img_resolution}, patch_size: {patch_size}"
        )

        self.input_embeddings_layer = EncoderInputEmbeddingsLayer(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            img_resolution=img_resolution,
            dropout=embedding_dropout,
        )

        # self.encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=embed_dim,
        #     nhead=num_heads,
        #     dim_feedforward=mlp_dim,
        #     dropout=mlp_dropout,
        #     activation="gelu",
        #     norm_first=True,
        #     batch_first=True
        # )
        #
        # self.transformer_encoder = nn.TransformerEncoder(
        #     encoder_layer=self.encoder_layer,
        #     num_layers=num_encoders,
        # )

        self.transformer_encoder = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    attn_dropout=attn_dropout,
                    mlp_dropout=mlp_dropout,
                )
                for _ in range(num_encoders)
            ]
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transformer_encoder(self.input_embeddings_layer(x))
        return self.classifier(x[:, 0])


if __name__ == "__main__":
    random_image = torch.randn(size=(32, 3, 224, 224))
    vit_model = ViT(num_classes=3)

    vit_model.eval()
    with torch.inference_mode():
        output = vit_model(random_image)
    pred_probs = torch.softmax(output, dim=1)
    pred_label = torch.argmax(pred_probs, dim=1)
