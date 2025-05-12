import torch.nn as nn
from typing import *
from kans.feature_extractor import EnhancedFeatureExtractor
from kans.fasterkan_layers import FasterKANLayer

class FasterKAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_min: float = -1.,
        grid_max: float = 1.,
        num_grids: int = 5,
        exponent: int = 2,
        inv_denominator: float = 0.5,
        train_grid: bool = False,        
        train_inv_denominator: bool = False,
        #use_base_update: bool = True,
        base_activation = None,
        spline_weight_init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            FasterKANLayer(
                in_dim, out_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                exponent = exponent,
                inv_denominator = inv_denominator,
                train_grid = train_grid ,
                train_inv_denominator = train_inv_denominator,
                #use_base_update=use_base_update,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class FasterKANvolver(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_min: float = -1.2,
        grid_max: float = 0.2,
        num_grids: int = 8,
        exponent: int = 2,
        inv_denominator: float = 0.5,
        train_grid: bool = False,        
        train_inv_denominator: bool = False,
        base_activation = None,
        spline_weight_init_scale: float = 1.0,
        view = [-1, 1, 28, 28],
    ) -> None:
        super(FasterKANvolver, self).__init__()
        
        self.view = view
        self.feature_extractor = EnhancedFeatureExtractor(colors = view[1])

        flat_features = 256 # XX channels, image size reduced to YxY
        
        # Update layers_hidden with the correct input size from conv layers
        layers_hidden = [flat_features] + layers_hidden
        
        # Define the FasterKAN layers
        self.faster_kan_layers = nn.ModuleList([
            FasterKANLayer(
                in_dim, out_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                exponent=exponent,
                inv_denominator = 0.5,
                train_grid = False,        
                train_inv_denominator = False,
                #use_base_update=use_base_update,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])   

    def forward(self, x):
        # Handle different input shapes based on the length of view
        x = x.view(self.view[0], self.view[1], self.view[2], self.view[3])
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        
        # Pass through FasterKAN layers
        for layer in self.faster_kan_layers:
            x = layer(x)        
        return x