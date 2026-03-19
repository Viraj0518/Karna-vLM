"""
Tests for configuration and YAML round-trip.
"""

import pytest
import tempfile
from pathlib import Path

from karna_vlm.models.vlm_model import KarnaVLMConfig


class TestKarnaVLMConfig:
    def test_default_config(self) -> None:
        config = KarnaVLMConfig()
        assert config.model_family == "small"
        assert config.bridge_type == "qformer_lite"
        assert config.vision_freeze is True
        assert config.decoder_freeze is True

    def test_yaml_round_trip(self) -> None:
        config = KarnaVLMConfig(
            model_name="test-model",
            bridge_type="resampler",
            bridge_num_queries=32,
            max_length=1024,
        )

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            config.to_yaml(f.name)
            loaded = KarnaVLMConfig.from_yaml(f.name)

        assert loaded.model_name == "test-model"
        assert loaded.bridge_type == "resampler"
        assert loaded.bridge_num_queries == 32
        assert loaded.max_length == 1024

    def test_from_yaml_files(self) -> None:
        """Test loading the actual config files."""
        config_dir = Path(__file__).parent.parent / "configs"

        for yaml_file in config_dir.glob("model_*.yaml"):
            config = KarnaVLMConfig.from_yaml(yaml_file)
            assert config.model_name
            assert config.vision_backend in ("siglip", "clip")
            assert config.bridge_type in (
                "linear", "qformer_lite", "resampler",
                "gated", "instruction_conditioned",
            )
