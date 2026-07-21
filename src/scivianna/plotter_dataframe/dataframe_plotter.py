from typing import Callable, Optional
import panel as pn
import pandas as pd


class DataframePlotter:
    """Simple plotter for displaying pandas DataFrames."""

    on_mouse_move_callback = None
    """Function to call when the mouse is moved on the plot"""
    on_clic_callback = None
    """Function to call when the mouse is clicked on the plot"""

    def __init__(self):
        self._dataframe_pane = pn.pane.DataFrame(
            sizing_mode="stretch_both",
            styles={"border": "2px solid lightgray"},
        )
        pn.io.push_notebook(self._dataframe_pane)

    def update_data(self, data: pd.DataFrame):
        """Updates the displayed DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame to display
        """
        if data is not None:
            self._dataframe_pane.object = data

    def get_data(self) -> Optional[pd.DataFrame]:
        """Returns the currently displayed DataFrame.

        Returns
        -------
        Optional[pd.DataFrame]
            Current DataFrame, or None
        """
        return self._dataframe_pane.object

    def make_panel(self) -> pn.viewable.Viewable:
        """Makes the Panel viewable displayed in the web app.

        Returns
        -------
        pn.viewable.Viewable
            Displayed viewable
        """
        return self._dataframe_pane

    def provide_on_mouse_move_callback(self, callback: Callable):
        """Stores a function to call when the user moves the mouse on the plot.

        Parameters
        ----------
        callback : Callable
            Function to call.
        """
        self.on_mouse_move_callback = callback

    def provide_on_clic_callback(self, callback: Callable):
        """Stores a function to call when the user clicks on the plot.

        Parameters
        ----------
        callback : Callable
            Function to call.
        """
        self.on_clic_callback = callback

    def enable_highlight(self, enable: bool = True):
        """Enable hover highlight (no-op for dataframe).

        Parameters
        ----------
        enable : bool, optional
            Highlight enabled, by default True
        """
        pass
