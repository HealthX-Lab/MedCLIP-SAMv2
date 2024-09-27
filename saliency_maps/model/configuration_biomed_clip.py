import os
from typing import *
from transformers.configuration_utils import PretrainedConfig
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig

class BiomedCLIPTextProjectionConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=640,
        projection_dim=512,
        num_hidden_layers=2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from CLIPConfig
        if config_dict.get("model_type") == "clip":
            config_dict = config_dict["text_projection_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)

class BiomedCLIPConfig(CLIPConfig):
    def __init__(
        self, text_config=None, text_projection_config=None, vision_config=None, projection_dim=512, logit_scale_init_value=2.6592, **kwargs
    ):
        # If `_config_dict` exist, we use them for the backward compatibility.
        # We pop out these 2 attributes before calling `super().__init__` to avoid them being saved (which causes a lot
        # of confusion!).
        super().__init__(text_config, vision_config, projection_dim, logit_scale_init_value, **kwargs)
        
        text_projection_config_dict = kwargs.pop("text_projection_config_dict", None)
        if text_projection_config is None:
            if text_projection_config_dict is not None:
                text_projection_config = {}
    
                _text_projection_config_dict = BiomedCLIPTextProjectionConfig(**text_projection_config_dict)
                
                text_projection_config.update(_text_projection_config_dict)
        else:
            text_projection_config = BiomedCLIPTextProjectionConfig(**text_projection_config)
            
        self.text_projection_config = text_projection_config
