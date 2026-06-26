from typing import Any, Dict, Tuple, Union, TYPE_CHECKING
import panel as pn

from scivianna.data.data2d import Data2D
from scivianna.plotter_2d.generic_plotter import Plotter2D

if TYPE_CHECKING:
    from scivianna.panel.visualisation_panel import VisualizationPanel
    from scivianna.slave import ComputeSlave


class Extension:
    """ Definition of a generic scivianna extension. An extension provides a range of tools to customize and interract with the plotting panel and slave.
    """
    title: str
    """Extension title, displayed on top of the extension side bar"""
    description: str = ""
    """Extension short documentation"""
    icon: str
    """Extension icon, displayed on the tabs list"""
    slave: "ComputeSlave"
    """Slave computing the displayed data"""
    plotter: Plotter2D
    """Figure plotter"""
    panel: "VisualizationPanel"
    """Panel to which the extension is attached"""
    iconsize: str = "6em"
    _restoring: bool = False
    """Flag to prevent feedback triggers during from_json restoration"""


    def __init__(
        self,
        title: str,
        icon: str,
        slave: "ComputeSlave",
        plotter: Plotter2D,
        panel: "VisualizationPanel",
    ):
        """Constructor of the extension, saves the slave and the panel

        Parameters
        ----------
        title : str
            Extension title, displayed on top of the extension side bar
        icon : str
            Extension icon, displayed on the tabs list
        slave : ComputeSlave
            Slave computing the displayed data
        plotter : Plotter2D
            Figure plotter
        panel : VisualizationPanel
            Panel to which the extension is attached
        """
        self.title = title
        self.icon = icon
        self.slave = slave
        self.plotter = plotter
        self.panel = panel
        self._restoring = False

    def on_file_load(self, file_path: str, file_key: str):
        """Function called when the user requests a change of field on the GUI

        Parameters
        ----------
        file_path : str
            Path of the loaded file
        file_key : str
            Key associated to the loaded file
        """
        pass

    def on_field_change(self, field_name: str):
        """Function called when the user requests a displayed field change

        Parameters
        ----------
        field_name : str
            Name of the new displayed field
        """
        pass

    def on_updated_data(self, data: Data2D):
        """Function called when the displayed data is being updated. Extension can edit the data on its way to the plotter.

        Parameters
        ----------
        data : Data2D
            Data to display
        """
        pass

    def on_range_change(
        self,
        u_bounds: Tuple[float, float],
        v_bounds: Tuple[float, float],
        w_value: float,
    ):
        """Function called when the viewport range was updated.

        Parameters
        ----------
        y_bounds : Tuple[float, float]
            Bounds along the U axis
        v_bounds : Tuple[float, float]
            Bounds along the V axis
        w_value : Tuple[float, float]
            Coordinate along the W axis
        """
        pass

    def on_frame_change(
        self,
        u_vector: Tuple[float, float, float],
        v_vector: Tuple[float, float, float],
    ):
        """Function called when the 2D viewport updates its u and v vectors

        Parameters
        ----------
        u_vector : Tuple[float, float, float]
            Horizontal vector
        v_vector : Tuple[float, float, float]
            Vertical vector
        """
        pass

    def on_mouse_move(
        self,
        screen_location: Tuple[float, float],
        space_location: Tuple[float, float, float],
        cell_id: Union[str, int],
    ):
        """Function called when the mouse moves on the plot

        Parameters
        ----------
        screen_location : Tuple[float, float]
            Mouse location on screen
        space_location : Tuple[float, float, float]
            Mouse targeted location in space
        cell_id : Union[str, int]
            Currently hovered cell
        """
        pass

    def on_mouse_clic(
        self,
        screen_location: Tuple[float, float],
        space_location: Tuple[float, float, float],
        cell_id: Union[str, int],
    ):
        """Function called when the mouse clics on the plot

        Parameters
        ----------
        screen_location : Tuple[float, float]
            Mouse location on screen
        space_location : Tuple[float, float, float]
            Mouse targeted location in space
        cell_id : Union[str, int]
            Currently hovered cell
        """
        pass

    def on_coupling_update(
        self,
    ):
        """Function called at the end of a coupling time step
        """
        pass

    def on_scale_change(
        self, axis_name: str, value: str
    ):
        """Function called When the scale is changed

        Parameters
        ----------
        axis_name : str
            Name of the axis changing its scale
        value : str
            New scale value
        """
        pass

    def provide_options(self,) -> Dict[str, Any]:
        """Provide a set of options to give to the slave for its update

        Returns
        -------
        Dict[str, Any]
            Options dictionnary
        """
        return {}

    def make_gui(self,) -> pn.viewable.Viewable:
        """Returns a panel viewable to display in the extension tab.

        Returns
        -------
        pn.viewable.Viewable
            Viewable to display in the extension tab
        """
        return None

    def to_json(self) -> dict:
        """Returns a dictionary with the information required to rebuild the extension.

        Returns
        -------
        dict
            Information dictionary
        """
        return {}

    @classmethod
    def from_json(cls, extension: "Extension", info_dict: dict) -> "Extension":
        """Restores the extension from its information dict.

        Parameters
        ----------
        extension : Extension
            Extension instance to restore
        info_dict : dict
            Dictionary containing extension state information

        Returns
        -------
        Extension
            Restored extension
        """
        return extension
