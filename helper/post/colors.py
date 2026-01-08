import numpy as np
from matplotlib.colors import LinearSegmentedColormap

__all__ = ['beige_to_black',
           'black_to_beige']


BEIGE = np.array([0.98, 0.96, 0.92, 1.0])  
BLACK = np.array([0, 0, 0, 1.0])            
 
colors = np.linspace(BEIGE, BLACK, 256)
colors_list = [tuple(color) for color in colors]
beige_to_black = LinearSegmentedColormap.from_list('beige_to_black', colors_list)

reversed_colors = np.linspace(BLACK, BEIGE, 256) 
reversed_colors_list = [tuple(color) for color in reversed_colors]
black_to_beige = LinearSegmentedColormap.from_list('black_to_beige', reversed_colors_list)

