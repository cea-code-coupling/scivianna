
import numpy as np

from scivianna.data.data2d import Data2D
from scivianna.utils.color_tools import get_edges_colors

class Data2DWorker:
    """Worker that receives a Data2D object and works with it. """

    def __init__(self, data2d:Data2D):
        """Worker that receives a Data2D object and works with it. 

        Parameters
        ----------
        data2d : Data2D
            Data class containing geometrical properties
        """
        self.data2d = data2d.copy()
        self.data2d_save = data2d.copy()

    def has_changed(self,) -> bool:
        """Tells if the data_2d was changed by the agent

        Returns
        -------
        bool
            data_2d changed
        """
        try:
            np.testing.assert_equal(np.array(self.data2d.cell_colors), np.array(self.data2d_save.cell_colors))
            np.testing.assert_equal(np.array(self.data2d.cell_edge_colors), np.array(self.data2d_save.cell_edge_colors))
            np.testing.assert_equal(np.array(self.data2d.cell_ids), np.array(self.data2d_save.cell_ids))
            np.testing.assert_equal(np.array(self.data2d.cell_values), np.array(self.data2d_save.cell_values))
        except AssertionError:
            return True

        return False

    def check_valid(self,):
        """Check if the Data2D is valid

        Raises
        ------
        AssertionError
            Current Data2D is not valid
        """
        self.data2d.check_valid()

    def get_values(self,) -> np.ndarray:
        """Returns the value per 2D cell of the geometry

        Returns
        -------
        np.ndarray
            Numpy array with the value per cell
        """
        return np.array(self.data2d.cell_values)
    
    def get_colors(self,) -> np.ndarray:
        """Returns the color per 2D cell of the geometry. 
        The returned object is a np.ndarray of shape (cell_count, 4), with values ranging from 0 to 255.

        Returns
        -------
        np.ndarray
            Numpy array with the value per cell
        """
        return np.array(self.data2d.cell_colors)
    
    def set_colors(self, colors:np.ndarray) -> bool:
        """Sets the cells color values, expects a numpy array of integers between 0 and 255 of shape (cell_count, 4). return True if it's ok.

        Parameters
        ----------
        colors : np.ndarray
            np array containing the new per cell colors. expects a numpy array of integers between 0 and 255 of shape (cell_count, 4).

        Returns
        -------
        bool
            True if the given array is ok
        """
        assert type(colors) in (np.ndarray, list), f"A numpy array is expected, type found {type(colors)}."
        colors = np.array(colors)
        assert len(colors.shape) == 2, f"A 2D numpy array is expected, shape found {colors.shape}."
        assert colors.shape == np.array(self.data2d.cell_colors).shape, f"We expect the same number of elements as in self.data2d.cell_colors, received shape {colors.shape} instead of {self.data2d.cell_colors}."
        assert colors.flatten().max() <= 255, f"The values must be lower than 255, found in array {colors.flatten().max()}."
        assert colors.flatten().min() >= 0, f"The values must be greater than 0, found in array {colors.flatten().min()}."

        self.data2d.cell_colors = colors
        self.data2d.cell_edge_colors = get_edges_colors(colors)

        edge_colors = np.array(self.data2d.cell_edge_colors)

        if not isinstance(self.data2d.cell_values[0], str):
            edge_colors[:, 3] = np.where(np.isnan(np.array(self.data2d.cell_values)), 255, edge_colors[:, 3])

        self.data2d.cell_edge_colors = edge_colors

        return True
    
    def set_alphas(self, alphas:np.ndarray) -> bool:
        """
        Sets the cells opacity values, expects a numpy array of integers between 0 and 255. return True if it's ok.

        Parameters
        ----------
        alphas : np.ndarray
            Opacity values     

        Returns
        -------
        bool
            True if the given array is ok            
        """
        assert type(alphas) in (np.ndarray, list), f"A numpy array is expected, type found {type(alphas)}."
        alphas = np.array(alphas)
        assert len(alphas.shape) == 1, f"A 1D numpy array is expected, shape found {alphas.shape}."
        assert alphas.shape[0] == len(self.data2d.cell_colors), f"We expect the same number of elements as in self.data2d.cell_colors, received size {alphas.shape[0]} instead of {len(self.data2d.cell_colors)}."
        assert alphas.max() <= 255, f"The values must be lower than 255, found in array {alphas.max()}."
        assert alphas.min() >= 0, f"The values must be greater than 0, found in array {alphas.min()}."

        colors = np.array(self.data2d.cell_colors)
        colors[:, -1] = alphas.astype(int)
        
        edge_colors = np.array(self.data2d.cell_edge_colors)
        edge_colors[:, -1] = alphas.astype(int)

        self.data2d.cell_colors = colors
        self.data2d.cell_edge_colors = edge_colors

        return True

    def reset(self,):
        """Returns to the data2d provided in the initialization
        """
        self.data2d = self.data2d_save.copy()
    
    def get_numpy(self,):
        """Returns the numpy module
        """
        return np
    