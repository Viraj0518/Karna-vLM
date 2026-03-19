"""
Karna VLM command-line interface.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def app() -> None:
    """CLI entry point."""
    try:
        import typer
    except ImportError:
        print("Install typer for CLI: pip install typer")
        return

    cli = typer.Typer(name="karna", help="Karna VLM CLI")

    @cli.command()
    def generate(
        image: str = typer.Argument(..., help="Path to image"),
        prompt: str = typer.Option("Describe this image.", help="Text prompt"),
        config: str = typer.Option("configs/model_small.yaml", help="Config file"),
        max_tokens: int = typer.Option(256, help="Max tokens to generate"),
    ) -> None:
        """Generate text from an image."""
        from PIL import Image as PILImage
        from karna_vlm.models.vlm_model import KarnaVLM, KarnaVLMConfig

        cfg = KarnaVLMConfig.from_yaml(config)
        model = KarnaVLM(cfg)
        img = PILImage.open(image).convert("RGB")
        output = model.generate(images=[img], prompt=prompt, max_new_tokens=max_tokens)
        print(output)

    @cli.command()
    def serve(
        config: str = typer.Option("configs/model_small.yaml", help="Config file"),
        host: str = typer.Option("0.0.0.0", help="Server host"),
        port: int = typer.Option(8080, help="Server port"),
    ) -> None:
        """Start the inference API server."""
        import uvicorn
        from karna_vlm.models.vlm_model import KarnaVLM, KarnaVLMConfig
        from karna_vlm.api.server import create_app

        cfg = KarnaVLMConfig.from_yaml(config)
        model = KarnaVLM(cfg)
        api = create_app(model)
        uvicorn.run(api, host=host, port=port)

    @cli.command()
    def info(
        config: str = typer.Option("configs/model_small.yaml", help="Config file"),
    ) -> None:
        """Show model configuration and parameter counts."""
        from karna_vlm.models.vlm_model import KarnaVLMConfig
        from dataclasses import asdict
        import yaml

        cfg = KarnaVLMConfig.from_yaml(config)
        print(yaml.dump(asdict(cfg), default_flow_style=False))

    cli()
