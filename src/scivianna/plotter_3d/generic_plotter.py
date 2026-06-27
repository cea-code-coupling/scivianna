from scivianna.data.data3d import Data3D

class Plotter3D:
    """Unfinished 3D plotter to get the coupling working"""

    def __init__(
        self,
    ):
        pass

    def plot(
        self,
        data: Data3D
    ):
        """Adds a new plot to the figure from a set of polygons

        Parameters
        ----------
        data : Data3D
            Data3D object containing the geometry and data to plot
        """
        raise NotImplementedError()

    def update_plot(
        self,
        data: Data3D
    ):
        """Updates plot to the figure

        Parameters
        ----------
        data : Data3D
            Data3D object containing the geometry and data to plot
        """
        raise NotImplementedError()

    def make_panel(self,):
        raise NotImplementedError()
