from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import multiprocessing as mp

import panel_material_ui as pmui
import panel as pn
import pyvista as pv
import vtk

import scivianna
from scivianna.extension.extension import Extension
from scivianna.panel.visualisation_panel import VisualizationPanel
from scivianna.plotter_2d.generic_plotter import Plotter2D
from scivianna.slave import ComputeSlave

from scivianna.interface.generic_interface import Geometry2DPolygon
from scivianna.interface import csv_result
from scivianna.utils.polygonize_tools import PolygonCoords, PolygonElement
from scivianna.enums import GeometryType, VisualizationMode
from scivianna.constants import MESH, GEOMETRY, CSV
from scivianna.data.data2d import Data2D


with open(Path(scivianna.__file__).parent / "icon" / "vtk.svg", "r") as f:
    icon_svg = f.read()

class VTKExtension(Extension):
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
            "VTK",
            icon_svg,
            slave,
            plotter,
            panel,
        )

        self.time_values = self.slave.call_custom_function("get_time_values", {})
        if self.time_values == []:
            self.time_values = [0.]

        self.time_slider = pmui.DiscreteSlider(name = "Time", options = self.time_values, width=280, )
        self.recompute_on_change = pmui.Checkbox(name = "Recompute mesh on time change", value=False)

        assert isinstance(plotter, Plotter2D), "VTKExtension can only be used with a Plotter2D"

        self.description = """
This extension allows defining the VTK plot parameters.
"""

        self.iconsize = "1.0em"

        self.time_slider.param.watch(self.recompute, "value_throttled")

    def on_file_load(self, file_path, file_key):
        if file_key == GEOMETRY:
            self.time_values = self.slave.call_custom_function("get_time_values", {})
            if self.time_values == []:
                self.time_values = [0.]

            self.time_slider.options = self.time_values

    def recompute(self, *args, **kwargs):
        self.panel.recompute()

    def provide_options(self):
        return {
            "time": self.time_slider.values[self.time_slider.value_throttled],
            "recompute": self.recompute_on_change.value,
        }

    def make_gui(self,) -> pn.viewable.Viewable:
        """Returns a panel viewable to display in the extension tab.

        Returns
        -------
        pn.viewable.Viewable
            Viewable to display in the extension tab
        """
        return pmui.Column(
            self.time_slider,
            self.recompute_on_change,
            margin=0
        )

def extract_unstructured(dataset):
    if isinstance(dataset, pv.UnstructuredGrid):
        return dataset
    elif isinstance(dataset, pv.MultiBlock):
        for block in dataset:
            if block is not None:
                result = extract_unstructured(block)
                if result is not None:
                    return result
    else:
        try:
            return dataset.cast_to_unstructured_grid()
        except Exception:
            return None
        
class VTKInterface(Geometry2DPolygon):
    
    extensions = [VTKExtension]
    geometry_type = GeometryType._3D_INFINITE

    def __init__(
        self,
    ):
        """VTK interface constructor."""
        self.data: Data2D = None
        self.reader: pv.BaseReader = None
        self.mesh: pv.MultiBlock = None
        self.times: List[float] = []
        self.results: Dict[str, Any] = {}
        self.last_computed_frame: List[float] = []
        self.current_time: float = -1.

    def read_file(self, file_path: str, file_label: str):
        """Read a file and store its content in the interface

        Parameters
        ----------
        file_path : str
            File to read
        file_label : str
            Label to define the file type
        """
        if file_label == GEOMETRY:
            self.reader: pv.PVDReader = pv.get_reader(file_path)
            self.times = self.reader.time_values
            self.load_at_time(self.reader.time_values[-1])

        if file_label == "MULTI_BLOCK":
            self.reader = vtk.vtkXMLMultiBlockDataReader()
            self.reader.SetFileName(file_path) 
            self.reader.Update()

            data = self.reader.GetOutput()

            append = vtk.vtkAppendFilter()

            def add_blocks(block):
                if block is None:
                    return

                if block.IsA("vtkUnstructuredGrid"):
                    append.AddInputData(block)

                elif block.IsA("vtkMultiBlockDataSet"):
                    for i in range(block.GetNumberOfBlocks()):
                        add_blocks(block.GetBlock(i))

                elif block.IsA("vtkMultiPieceDataSet"):
                    for i in range(block.GetNumberOfPieces()):
                        add_blocks(block.GetPiece(i))

            # Traverse everything
            add_blocks(data)

            append.Update()

            self.mesh = pv.wrap(append.GetOutput())
            self.mesh = self.mesh.point_data_to_cell_data()
            self.mesh.cell_data["cell_id"] = list(range(self.mesh.number_of_cells))

        elif file_label == CSV:
            name = Path(file_path).name
            self.results[name] = csv_result.CSVInterface(file_path)

        else:
            raise NotImplementedError(
                f"File label {file_label} not implemented in VTK interface."
            )

    def load_at_time(self, time: float):
        """Loads the data at the provided time

        Parameters
        ----------
        time : float
            Time at which load the data

        Raises
        ------
        ValueError
            Provided time not in data
        """
        if not time in self.times:
            raise ValueError(f"Provided time {time} does not exist.")
        
        self.reader.set_active_time_value(time)
        self.mesh = extract_unstructured(self.reader.read()[0])
        self.mesh = self.mesh.point_data_to_cell_data()
        self.current_time = self.reader.time_values[-1]
        self.mesh.cell_data["cell_id"] = list(range(self.mesh.number_of_cells))

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
    ) -> Tuple[List[PolygonElement], bool]:
        """Returns a list of polygons that defines the geometry in a given frame

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

        Returns
        -------
        List[PolygonElement]
            List of polygons to display
        bool
            Were the polygons updated compared to the past call
        """
        if not "recompute" in options:
            options["recompute"] = True
        if not "time" in options:
            options["time"] = self.current_time

        time_updated = False
        if self.current_time != options["time"]:
            self.load_at_time(options["time"])
            time_updated = True

        if (self.data is not None) and (
            self.last_computed_frame == [*u, *v, w_value]
        ) and ( not (
            options["recompute"] and time_updated
        )):
            print("Skipping polygon computation.")
            return self.data, False
        
        u = np.array(u)/np.linalg.norm(u)
        v = np.array(v)/np.linalg.norm(v)
        w = np.cross(u, v)

        if np.linalg.norm(w) == 0.:
            raise ValueError(f"u and v must be both non zero and non parallel, found {u}, {v}")
        
        w /= np.linalg.norm(w)
        
        origin = u*u_min + v*v_min + w_value*w

        mesh_slice: pv.PolyData = self.mesh.slice(normal = w, origin = origin, generate_triangles = True)

        polygon_elements = []

        if mesh_slice.GetNumberOfCells() == 0:
            raise ValueError(f"Tried slicing at location {origin}, with normal {w}. Mesh has the bounds : {self.mesh.bounds}. No cell was found.")
        
        point_ids = [mesh_slice.get_cell(j).point_ids for j in range(len(mesh_slice.cell_data["cell_id"]))]

        for i, pol in enumerate(point_ids):
            ids = np.array(pol)
            polygon_elements.append(
            PolygonElement(
                PolygonCoords(
                    np.array(mesh_slice.points[ids].dot(u)), 
                    np.array(mesh_slice.points[ids].dot(v))
                ),
                [],
                mesh_slice.cell_data["cell_id"][i],
            ))

        self.data = Data2D.from_polygon_list(polygon_elements)
        self.last_computed_frame = [*u, *v, w_value]
        
        return self.data, True


    def get_value_dict(
        self, value_label: str, cells: List[Union[int, str]], options: Dict[str, Any]
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
        if value_label == MESH:
            dict_compo = {v: np.nan for v in cells}

            return dict_compo

        if value_label in self.mesh.array_names:
            data = self.mesh.cell_data[value_label]
            return dict(zip(cells, data[cells]))

        for res in self.results.values():
            if value_label in res.get_fields():
                results = res.get_values([], cells, [], value_label)
                return {cells[i]: results[i] for i in range(len(cells))}

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
        labels = [MESH]

        labels += [e for e in self.mesh.array_names if not e == "TIME"]

        for res in self.results.values():
            labels += res.get_fields()

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
        else:
            return VisualizationMode.FROM_VALUE

    def get_file_input_list(self) -> List[Tuple[str, str]]:
        """Returns a list of file label and its description for the GUI

        Returns
        -------
        List[Tuple[str, str]]
            List of (file label, description)
        """
        return [
            (GEOMETRY, "VTK .pvd file."),
            (CSV, "CSV result file."),
        ]

    def get_time_values(self, ) -> List[float]:
        """Returns the loaded file time values

        Returns
        -------
        List[float]
            Loaded file time values
        """
        return self.times
    

if __name__ == "__main__":
    file_path = Path("/path/to/file.pvd")
    if True:
        import time 

        st = time.time()
        from scivianna.plotter_2d.api import plot_frame
        from scivianna.constants import Z

        interface = ComputeSlave(VTKInterface)
        interface.read_file(file_path, GEOMETRY)

        fig, ax = plot_frame(
            interface,
            "Temperature",
            w_value = .5,
            edge_width=0.1,
            display_colorbar=True,
            options={"time": 0.5}
        )
        fig.savefig(
            "plot_vtk.png",
            dpi=500
        )
        print("First plot time", time.time() - st)
        
        fig, ax = plot_frame(
            interface,
            "Temperature",
            w_value = .5,
            edge_width=0.1,
            v=Z,
            display_colorbar=True,
            options={"time": 0.5}
        )
        fig.savefig(
            "plot_vtk_xz.png",
            dpi=500
        )
        print("Two plot time", time.time() - st)
    else:
        from scivianna.panel.panel_2d import Panel2D
        slave = ComputeSlave(VTKInterface)
        slave.read_file(file_path, GEOMETRY)

        panel = Panel2D(slave, name="VTK")
        panel.set_coordinates(
            w = .5
        )
        panel.show()