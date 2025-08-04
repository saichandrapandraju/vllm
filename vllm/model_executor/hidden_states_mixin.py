# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Mixin class for hidden states extraction functionality."""

from abc import abstractmethod
from typing import Optional, Union
from vllm.logger import init_logger

logger = init_logger(__name__)


class HiddenStatesExtractorMixin:
    """Mixin to add hidden states extraction capability to any model.
    
    This mixin provides a standardized interface for extracting hidden states
    from transformer layers during inference. It supports:
    - Extracting from all layers (hidden_states=True)
    - Extracting from specific layers (hidden_states=[0, 5, 10])
    - Disabling extraction (hidden_states=None/False)
    
    Models using this mixin must implement get_model_layers() to return
    the transformer layers.
    """
    
    def __init_hidden_states__(self):
        """Initialize hidden states infrastructure. 
        
        Call this in model __init__ after creating self.model.
        """
        if not hasattr(self, 'model'):
            raise AttributeError(
                "HiddenStatesExtractorMixin requires 'self.model' attribute. "
                "Ensure model is created before calling __init_hidden_states__."
            )
        
        # Initialize aux_hidden_state_layers attribute if not present
        if not hasattr(self.model, 'aux_hidden_state_layers'):
            self.model.aux_hidden_state_layers = tuple()
        
        logger.debug(f"Initialized hidden states extraction for {self.__class__.__name__}")
    
    def set_aux_hidden_state_layers(self, layers: tuple[int]) -> None:
        """Set which layers to extract hidden states from (internal format).
        
        Args:
            layers: Tuple of layer indices to extract from.
        """
        if not hasattr(self, 'model'):
            raise AttributeError("Model not initialized. Call __init_hidden_states__ first.")
        
        # Validate layer indices
        model_layers = self.get_model_layers()
        if model_layers is not None:
            num_layers = len(model_layers)
            invalid_layers = [layer for layer in layers if layer >= num_layers or layer < 0]
            if invalid_layers:
                raise ValueError(
                    f"Invalid layer indices {invalid_layers} for model with {num_layers} layers. "
                    f"Valid layer indices are 0-{num_layers-1}."
                )
        
        self.model.aux_hidden_state_layers = layers
        logger.debug(f"Set aux_hidden_state_layers to {layers}")
    
    def set_hidden_state_layers(self, hidden_states_param: Optional[Union[bool, list[int]]]) -> None:
        """Set which layers to extract hidden states from (API format).
        
        Args:
            hidden_states_param: 
                - None/False: No extraction
                - True: Extract from all layers
                - list[int]: Extract from specific layer indices
        """
        if hidden_states_param is None or hidden_states_param is False:
            layers = tuple()
        elif hidden_states_param is True:
            # Extract from all layers
            model_layers = self.get_model_layers()
            if model_layers is None:
                raise RuntimeError(
                    f"Cannot determine layer count for {self.__class__.__name__}. "
                    f"Model may not be properly initialized."
                )
            layers = tuple(range(len(model_layers)))
        else:
            # Specific layer indices
            if not isinstance(hidden_states_param, list):
                raise TypeError(
                    f"hidden_states_param must be bool or list[int], got {type(hidden_states_param)}"
                )
            layers = tuple(hidden_states_param)
        
        self.set_aux_hidden_state_layers(layers)
    
    def get_aux_hidden_state_layers(self) -> tuple[int]:
        """Get currently configured hidden state layers.
        
        Returns:
            Tuple of layer indices that will extract hidden states.
        """
        if not hasattr(self, 'model') or not hasattr(self.model, 'aux_hidden_state_layers'):
            return tuple()
        return self.model.aux_hidden_state_layers
    
    def has_hidden_states_enabled(self) -> bool:
        """Check if hidden states extraction is enabled.
        
        Returns:
            True if any layers are configured for extraction.
        """
        return len(self.get_aux_hidden_state_layers()) > 0
    
    @abstractmethod
    def get_model_layers(self):
        """Return the model's transformer layers.
        
        This method must be implemented by each model class using this mixin.
        
        Returns:
            The model's layers (e.g., self.model.layers) or None if unavailable.
        """
        pass
    
    def _validate_model_structure(self) -> None:
        """Validate that the model has the expected structure for hidden states extraction."""
        if not hasattr(self, 'model'):
            raise AttributeError(f"{self.__class__.__name__} must have 'model' attribute")
        
        layers = self.get_model_layers()
        if layers is None:
            logger.warning(
                f"Could not access model layers for {self.__class__.__name__}. "
                f"Hidden states extraction may not work properly."
            )
        else:
            logger.debug(f"Model {self.__class__.__name__} has {len(layers)} layers") 