from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import multiprocessing as mp
import numpy as np
import panel as pn
import panel_material_ui as pmui

from scivianna.extension.extension import Extension
from scivianna.interface.generic_interface import Geometry2DGrid
from scivianna.constants import MATERIAL, MESH
from scivianna.panel.panel_2d import Panel2D
from scivianna.plotter_2d.generic_plotter import Plotter2D
from scivianna.slave import ComputeSlave
from scivianna.panel.visualisation_panel import VisualizationPanel
from scivianna.enums import GeometryType, UpdateEvent, VisualizationMode
from scivianna.data.data2d import Data2D
from scivianna.layout.split import SplitDirection, SplitItem, SplitLayout


with open(Path(__file__).parent / "mandelbrot.svg", "r") as f:
    icon_svg = f.read()

class MandelbrotExtension(Extension):
    """Extension to load files and send them to the slave."""

    def __init__(
        self,
        slave: "ComputeSlave",
        plotter: "Plotter2D",
        panel: "VisualizationPanel"
    ):
        """Constructor of the extension, saves the slave and the panel

        Parameters
        ----------
        slave : ComputeSlave
            Slave computing the displayed data
        plotter : Plotter2D
            Figure plotter
        panel : VisualizationPanel
            Panel to which the extension is attached
        """
        super().__init__(
            "Mandelbrot",
            icon_svg,
            slave,
            plotter,
            panel,
        )

        self.description = """
This extension allows defining the medcoupling field display parameters.
"""

        self.iconsize = "1.0em"

        self.u_step_input = pmui.IntInput(
            label = "u_steps",
            value=500,
            description="Horizontal resolution.",
            width=280
        )
        self.v_step_input = pmui.IntInput(
            label = "v_steps",
            value=500,
            description="Vertical resolution.",
            width=280
        )
        self.max_iter_input = pmui.IntInput(
            label = "Max iter",
            value=10,
            description="Maximum mandelbrot iterations.",
            width=280
        )
        

        self.u_step_input.param.watch(self._on_u_steps_change, "value")
        self.v_step_input.param.watch(self._on_v_steps_change, "value")
        self.max_iter_input.param.watch(self._on_max_iter_change, "value")

    def _on_u_steps_change(self, event):
        """Callback for u_steps input change."""
        if self._restoring:
            return
        self.panel.recompute()

    def _on_v_steps_change(self, event):
        """Callback for v_steps input change."""
        if self._restoring:
            return
        self.panel.recompute()

    def _on_max_iter_change(self, event):
        """Callback for max_iter input change."""
        if self._restoring:
            return
        self.panel.recompute()

    def provide_options(self):
        return {
            "u_steps":self.u_step_input.value,
            "v_steps":self.v_step_input.value,
            "Max iter":self.max_iter_input.value,
        }
    
    def make_gui(self,) -> pn.viewable.Viewable:
        """Returns a panel viewable to display in the extension tab.

        Returns
        -------
        pn.viewable.Viewable
            Viewable to display in the extension tab
        """
        return pmui.Column(
            self.u_step_input,
            self.v_step_input,
            self.max_iter_input,
            margin=0
        )

    def to_json(self) -> dict:
        """Returns a dictionary with the information required to rebuild the extension.

        Returns
        -------
        dict
            Information dictionary
        """
        return {
            "u_steps": self.u_step_input.value,
            "v_steps": self.v_step_input.value,
            "max_iter": self.max_iter_input.value,
        }

    @classmethod
    def from_json(cls, extension: "MandelbrotExtension", info_dict: dict) -> "MandelbrotExtension":
        """Restores the extension from its information dict.

        Parameters
        ----------
        extension : MandelbrotExtension
            Extension instance to restore
        info_dict : dict
            Dictionary containing extension state information

        Returns
        -------
        MandelbrotExtension
            Restored extension
        """
        extension._restoring = True

        if info_dict.get("u_steps") is not None:
            extension.u_step_input.value = info_dict["u_steps"]

        if info_dict.get("v_steps") is not None:
            extension.v_step_input.value = info_dict["v_steps"]

        if info_dict.get("max_iter") is not None:
            extension.max_iter_input.value = info_dict["max_iter"]

        extension._restoring = False

        return extension



class MandelBrotInterface(Geometry2DGrid):
    geometry_type: GeometryType = GeometryType._2D
    extensions = [MandelbrotExtension]
    def __init__(
        self,
    ):
        """Antares interface constructor."""
        self.data: Dict[str, Data2D] = {}
        self.last_computed_frame: Dict[str, List[float]] = {}

    def read_file(self, file_path: str, file_label: str):
        """Read a file and store its content in the interface

        Parameters
        ----------
        file_path : str
            File to read
        file_label : str
            Label to define the file type
        """
        pass

    def compute_2D_data(
        self,
        u: Tuple[float, float, float],
        v: Tuple[float, float, float],
        u_min: float,
        u_max: float,
        v_min: float,
        v_max: float,
        w_value: float,
        q_tasks: mp.Queue,
        options: Dict[str, Any],
        caller: str = "API",
    ) -> Tuple[Data2D, bool]:
        """Returns a list of data that defines the geometry in a given frame

        Parameters
        ----------
        u : Tuple[float, float, float]
            Horizontal coordinate director vector
        v : Tuple[float, float, float]
            Vertical coordinate director vector
        u_min : float
            Lower bound value along the u axis
        u_max : float
            Upper bound value along the u axis
        v_min : float
            Lower bound value along the v axis
        v_max : float
            Upper bound value along the v axis
        w_value : float
            Value along the u ^ v axis
        q_tasks : mp.Queue
            Queue from which get orders from the master.
        options : Dict[str, Any]
            Additional options for frame computation.
        caller : str
            Identifier of the caller requesting the computation (default: "API")

        Returns
        -------
        Data2D
            Geometry properties
        bool
            Were the data updated compared to the past call
        """
        print("Computing", 
              u,
                v,
                u_min,
                u_max,
                v_min,
                v_max,
                w_value,
                q_tasks,
                options,
                caller
              )
        last_frame_key = (*u, *v, u_min, u_max, v_min, v_max, w_value, options.get("u_steps", 50), options.get("v_steps", 50), options.get("Max iter", 10))
        if (caller in self.last_computed_frame) and (
            self.last_computed_frame[caller] == last_frame_key
        ) and (caller in self.data):
            print("Skipping polygon computation.")
            return self.data[caller], False

        self.last_computed_frame[caller] = last_frame_key

        # Script taken from:
        # https://gist.github.com/jfpuget/60e07a82dece69b011bb

        maxiter = options["Max iter"] if "Max iter" in options else 10

        def mandelbrot(z, maxiter):
            c = z
            for n in range(maxiter):
                if abs(z) > 2:
                    return n
                z = z * z + c
            return maxiter

        def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, maxiter):
            r1 = np.linspace(xmin, xmax, width)
            r2 = np.linspace(ymin, ymax, height)
            return (
                r1,
                r2,
                [mandelbrot(complex(r, i), maxiter) for r in r1 for i in r2],
            )

        xvalues, yvalues, grid = mandelbrot_set(
            u_min, u_max, v_min, v_max, options["u_steps"], options["v_steps"], maxiter
        )

        self.data[caller] = Data2D.from_grid(
            np.array(grid).reshape((options["v_steps"], options["u_steps"]), order="F"), xvalues, yvalues
        )

        return self.data[caller], True

    def get_value_dict(
        self, value_label: str, cells: List[Union[int, str]], options: Dict[str, Any], caller: str = "API"
    ) -> Dict[Union[int, str], str]:
        """Returns a cell name - field value map for a given field name

        Parameters
        ----------
        value_label : str
            Field name to get values from
        cells : List[Union[int,str]]
            List of cells names
        options : Dict[str, Any]
            Additional options for frame computation.

        Returns
        -------
        Dict[Union[int,str], str]
            Field value for each requested cell names
        """
        if value_label == MATERIAL:
            dict_compo = {v: v for v in cells}
            return dict_compo

        if value_label == MESH:
            dict_compo = {v: np.nan for v in cells}
            return dict_compo

        raise NotImplementedError(
            f"The field {value_label} is not implemented, fields available : {self.get_labels()}"
        )

    def get_labels(
        self,
    ) -> List[str]:
        """Returns a list of fields names displayable with this interface

        Returns
        -------
        List[str]
            List of fields names
        """
        labels = [MATERIAL, MESH]

        return labels

    def get_label_coloring_mode(self, label: str) -> VisualizationMode:
        """Returns wheter the given field is colored based on a string value or a float.

        Parameters
        ----------
        label : str
            Field to color name

        Returns
        -------
        VisualizationMode
            Coloring mode
        """
        if label == MESH:
            return VisualizationMode.NONE
        if label in [MATERIAL]:
            return VisualizationMode.FROM_VALUE

    def get_file_input_list(self) -> List[Tuple[str, str]]:
        """Returns a list of file label and its description for the GUI

        Returns
        -------
        List[Tuple[str, str]]
            List of (file label, description)
        """
        return []


def make_panel(_, return_slaves=False):
    slave = ComputeSlave(MandelBrotInterface)
    panel = Panel2D(slave, name="Mandelbrot polygons")
    panel.update_event = UpdateEvent.RANGE_CHANGE
    panel.set_field(MATERIAL)
    panel.set_colormap("BuRd")

    slave_2 = ComputeSlave(MandelBrotInterface)
    panel_2 = Panel2D(slave_2, name="Mandelbrot grid", display_polygons=False)
    panel_2.update_event = UpdateEvent.RANGE_CHANGE
    panel_2.set_field(MATERIAL)
    panel_2.set_colormap("BuRd")

    layout = SplitLayout(
        SplitItem(panel, panel_2, SplitDirection.VERTICAL),
        additional_interfaces={"Mandelbrot": MandelBrotInterface},
    )

    if return_slaves:
        return layout, [slave, slave_2]
    else:
        return layout


if __name__ == "__main__":
    from scivianna.notebook_tools import _serve_panel

    _serve_panel(get_panel_function=make_panel, title="Mandelbrot set")
