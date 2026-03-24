from logging import warning
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple, Union, TYPE_CHECKING
import numpy as np
import multiprocessing as mp
import panel as pn
import panel_material_ui as pmui
import pickle

from scivianna.extension.save_load_extension import SaveLoadExtension

if TYPE_CHECKING:
    from scivianna.panel.visualisation_panel import VisualizationPanel
    from scivianna.slave import ComputeSlave
    from scivianna.plotter_2d.generic_plotter import Plotter2D

import scivianna
from scivianna.extension.extension import Extension
import scivianna.icon
from scivianna.data.data2d import Data2D
from scivianna.interface.generic_interface import Geometry2DPolygon, IcocoInterface
from scivianna.utils.polygonize_tools import PolygonElement, PolygonCoords
from scivianna.enums import GeometryType, VisualizationMode

import medcoupling

from scivianna.constants import MESH, GEOMETRY, CSV

profile_time = bool(os.environ["VIZ_PROFILE"]) if "VIZ_PROFILE" in os.environ else 0
if profile_time:
    import time

with open(Path(scivianna.icon.__file__).parent / "salome.svg", "r") as f:
    icon_svg = f.read()


class MEDCouplingExtension(Extension):
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
            "MEDCoupling",
            icon_svg,
            slave,
            plotter,
            panel,
        )

        self.description = """
This extension allows defining the medcoupling field display parameters.
"""

        self.iconsize = "1.0em"

        self.field_iterations = None

        self.field_iterations = self.slave.call_custom_function("get_iterations", {})
        self.field_name = list(self.field_iterations.keys())[0]

        self.iteration_input = pmui.IntInput(
            label="Iteration",
            value=self.field_iterations[self.field_name][0][0],
            description="Med field iteration.",
            width=280,
            color="primary",
            sx={}
        )
        self.order_input = pmui.IntInput(
            label="Order",
            value=self.field_iterations[self.field_name][0][1],
            description="Med field order.",
            width=280
        )
        self.slider_w = pmui.FloatSlider(
            label="W coordinate",
            visible=False,
            width=280
        )

        self.valid = True

        self.iteration_input.param.watch(self.recompute, "value")
        self.order_input.param.watch(self.recompute, "value")
        self.slider_w.param.watch(self.on_slider_change, "value")

        self.u = (1, 0, 0)
        self.v = (0, 1, 0)

        self.u_bounds = (0., 1.)
        self.v_bounds = (0., 1.)

        self.w = 0.5

        self.on_file_load(None, None)

    def provide_options(self) -> Dict[str, float]:
        """Provide the medcoupling interface options

        Returns
        -------
        Dict[str, float]
            MED options
        """
        return {
            "Iteration": self.iteration_input.value,
            "Order": self.order_input.value,
        }

    def on_field_change(self, field_name: str):
        """Saves field name and checks order/iteration values

        Parameters
        ----------
        field_name : str
            New displayed field
        """
        self.field_name = field_name
        self.check_int_inputs(force_valid_values=True)

    def on_file_load(self, file_path, file_key):
        """Catches the file load event to update its data
        """
        self.field_iterations = self.slave.call_custom_function("get_iterations", {})
        self.check_int_inputs(force_valid_values=True)
        self.update_slider_range()

    def on_slider_change(self, event):
        """Updates the panel w coordinate on slider change
        """
        if self.slider_w.value != self.w:
            self.panel.set_coordinates(w=self.slider_w.value)

    @pn.io.hold()
    def update_slider_range(self,):
        """Update slider bounds based on the mesh bounding box
        """
        x_range, y_range, z_range = self.slave.call_custom_function("get_bounding_box", {})

        w = np.cross(self.u, self.v).astype(float)
        w /= np.linalg.norm(w).astype(float)

        min_vect = np.array([x_range[0], y_range[0], z_range[0]])
        max_vect = np.array([x_range[1], y_range[1], z_range[1]])

        min_w = np.dot(w, min_vect)
        max_w = np.dot(w, max_vect)

        if self.slider_w.start != min(min_w, max_w):
            self.slider_w.start = min(min_w, max_w)

        if self.slider_w.end != max(min_w, max_w):
            self.slider_w.end = max(min_w, max_w)

        if self.w != self.slider_w.value:
            self.slider_w.value = self.w

        if not self.slider_w.visible:
            self.slider_w.visible = True

    def on_range_change(self, u_bounds: Tuple[float, float], v_bounds: Tuple[float, float], w_value: float):
        """Saves the frame bounds on both coordinates and the normal coordinate

        Parameters
        ----------
        u_bounds : Tuple[float, float]
            Bounds on the horizontal axis
        v_bounds : Tuple[float, float]
            Bounds on the vertical axis
        w_value : float
            Normal axis coordinate
        """
        self.u_bounds = u_bounds
        self.v_bounds = v_bounds
        self.w = w_value

        self.update_slider_range()

    def on_frame_change(self, u_vector, v_vector):
        """Saves new u and v vectors

        Parameters
        ----------
        u_vector : np.ndarray
            Horizontal vector
        v_vector : np.ndarray
            Vertical vector
        """
        self.u = np.array(u_vector)
        self.v = np.array(v_vector)

        self.update_slider_range()

    @pn.io.hold()
    def check_int_inputs(self, force_valid_values: bool = False):
        """Checks iteration and order intinputs values

        Parameters
        ----------
        force_valid_values : bool
            Sets values to valid values if not valid
        """
        if self.field_name is None or (self.field_iterations is None or self.field_name not in self.field_iterations):
            return

        tup = (self.iteration_input.value, self.order_input.value)

        if tup in self.field_iterations[self.field_name]:
            self.iteration_input.color = "primary"
            self.order_input.color = "primary"

            self.iteration_input.sx = {}
            self.order_input.sx = {}

        elif force_valid_values:
            self.iteration_input.value, self.order_input.value = self.field_iterations[self.field_name][0]

        else:
            # https://panel-material-ui.holoviz.org/how_to/customize.html
            pn.state.notifications.error("Iteration/order couple not valid, see console for available values.")
            print(f"Available iteration/order values for field '{self.field_name}': {self.field_iterations[self.field_name]}")

            self.iteration_input.color = "error"
            self.order_input.color = "error"
            self.iteration_input.sx = {
                # Target the notched outline (border) of the MUI OutlinedInput
                "& .MuiOutlinedInput-notchedOutline": {
                    "borderColor": "red",
                }
            }
            self.order_input.sx = {
                # Target the notched outline (border) of the MUI OutlinedInput
                "& .MuiOutlinedInput-notchedOutline": {
                    "borderColor": "red",
                }
            }

    def recompute(self, *args, **kwargs):
        """Recompute event on intinput changes
        """
        self.check_int_inputs()
        if self.valid:
            self.panel.recompute()

    def make_gui(self,) -> pn.viewable.Viewable:
        """Returns a panel viewable to display in the extension tab.

        Returns
        -------
        pn.viewable.Viewable
            Viewable to display in the extension tab
        """
        return pmui.Column(
            self.iteration_input,
            self.order_input,
            pmui.Typography("Coordinate along the normal axis"),
            self.slider_w,
            margin=0
        )


class MEDInterface(Geometry2DPolygon, IcocoInterface):

    polygons: List[PolygonElement]
    """Polygons computed at the previous iteration"""

    file_path: str
    """MEDCoupling .med file path saved to read MedCouplingFields later."""

    meshnames: List[str]
    """Names of the meshes stored in the .med file."""

    mesh: medcoupling.MEDCouplingUMesh
    """Mesh read from the .med file."""

    fieldnames: List[str]
    """Names of the fields stored in the .med file at the selected mesh name."""

    fields_iterations: Dict[str, List[Tuple[int, int]]]
    """List containing for tuples storing the field name, and the associated iteration."""

    fields: Dict[str, np.ndarray]
    """Dictionnary containing the list of per cell value for each read field."""

    field_doubles: Dict[str, medcoupling.MEDCouplingFieldDouble]
    """Dictionnary containing the received MEDCouplingFieldDouble."""

    cell_dict: Dict[int, int]
    """Dictionnary associating the 2D mesh cells to the 3D mesh cells"""

    """ Support mesh
    """
    geometry_type: GeometryType = GeometryType._3D_INFINITE
    extensions = [MEDCouplingExtension, SaveLoadExtension]

    def __init__(self):
        """MEDCoupling interface constructor."""
        self.data: Data2D = None
        """Past computed data"""
        self.file_path = None
        """Read file path saved for lazy load"""
        self.meshnames = []
        """List of mesh names in the file"""
        self.mesh = None
        """Currently loaded mesh"""
        self.fieldnames = []
        """List of fields in the current mesh"""
        self.fields = {}
        """List of fields data"""
        self.field_doubles = {}
        """Dictionnary containing the received MEDCouplingFieldDouble."""
        self.fields_iterations = {}
        """Dictionnary containing the med file available (iter, order) couples"""
        self.cell_dict = {}
        """Dictionnary associating the 2D mesh cells to the 3D mesh cells"""
        self.last_computed_frame = []
        """Parameters of the last computed frame"""

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
            if profile_time:
                start_time = time.time()
            print("File to read", file_path)

            file_path = str(file_path)
            self.file_path = file_path

            if not os.path.isfile(file_path):
                raise ValueError(f"Provided file name does not exist {file_path}")

            self.meshnames = medcoupling.GetMeshNames(file_path)
            self.fieldnames = medcoupling.GetAllFieldNamesOnMesh(
                file_path, self.meshnames[0]
            )

            self.fields_iterations = {}

            for field in self.fieldnames:
                components = medcoupling.GetComponentsNamesOfField(file_path, field)

                iterations = medcoupling.GetFieldIterations(
                    medcoupling.ON_CELLS, file_path, self.meshnames[0], field
                )

                for component in components:
                    for iteration in iterations:
                        self.fields_iterations[
                            (
                                "@".join([field, component[0]])
                                if component[0] != ""
                                else field
                            )
                        ] = [tuple(iteration)]

            self.mesh = medcoupling.ReadMeshFromFile(file_path, 0)

            if profile_time:
                print(f"File reading time {time.time() - start_time}")
        else:
            raise ValueError(f"File label '{file_label}' not implemented")

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
    ) -> Tuple[Data2D, bool]:
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
        Data2D
            Geometry to display
        bool
            Were the polygons updated compared to the past call
        """
        if (self.data is not None) and (
            self.last_computed_frame == [*u, *v, w_value]
        ):
            print("Skipping polygon computation.")
            return self.data, False

        if profile_time:
            start_time = time.time()

        mesh_dimension = self.mesh.getMeshDimension()

        use_cell_id = True

        if mesh_dimension == 2:
            mesh: medcoupling.MEDCouplingUMesh = self.mesh
            cell_ids = list(range(mesh.getNumberOfCells()))
        elif mesh_dimension == 3:
            vec = [float(e) for e in np.cross(u, v)]

            if any([u_min is None, v_min is None, w_value is None]):
                print(f"u_min : {u_min}")
                print(f"v_min : {v_min}")
                print(f"w_value : {w_value}")

            origin = [u_min * u[i] + v_min * v[i] + w_value * vec[i] for i in range(3)]

            try:
                eps = 0.0
                mesh: medcoupling.MEDCouplingUMesh = self.mesh.buildSlice3D(
                    origin, vec, eps
                )[0]

                cells_ids = self.mesh.getCellIdsCrossingPlane(origin, vec, eps)

                cell_ids = [int(c) for c in list(cells_ids)]

            except Exception:
                eps = 1e-7

                mesh: medcoupling.MEDCouplingUMesh = self.mesh.buildSlice3D(
                    origin, vec, eps
                )[0]

                cell_ids = [
                    int(c) for c in self.mesh.getCellIdsCrossingPlane(origin, vec, eps)
                ]

            if len(cell_ids) != mesh.getNumberOfCells():
                use_cell_id = False
        else:
            raise ValueError(
                f"Mesh dimension is {mesh_dimension}, should be either 2 or 3 to be displayed."
            )

        if profile_time:
            print(f"Compute mesh time {time.time() - start_time}")
            start_time = time.time()

        cells_count = mesh.getNumberOfCells()

        self.data = []

        vertices_coords = [list(c) for c in mesh.getCoords()]
        self.cell_dict.clear()

        for cell in range(cells_count):
            x_vals = [
                vertices_coords[cell_id][0] for cell_id in mesh.getNodeIdsOfCell(cell)
            ]
            y_vals = [
                vertices_coords[cell_id][1] for cell_id in mesh.getNodeIdsOfCell(cell)
            ]
            z_vals = [
                vertices_coords[cell_id][2] if mesh_dimension == 3 else 0.0
                for cell_id in mesh.getNodeIdsOfCell(cell)
            ]

            coords = np.array([x_vals, y_vals, z_vals])

            u_vals = np.matmul(coords.T, u)
            v_vals = np.matmul(coords.T, v)

            self.data.append(
                PolygonElement(
                    exterior_polygon=PolygonCoords(x_coords=u_vals, y_coords=v_vals),
                    holes=[],
                    cell_id=str(cell),
                )
            )

            if not use_cell_id:
                self.cell_dict[cell] = self.mesh.getCellContainingPoint(
                    [np.mean(x_vals), np.mean(y_vals), np.mean(z_vals)], eps=0.0
                )

        if use_cell_id:
            self.cell_dict = dict(zip(list(range(cells_count)), cell_ids))

        if profile_time:
            print(
                f"Gathering cells id time: {time.time() - start_time} using cell id {use_cell_id}"
            )

        self.last_computed_frame = [*u, *v, w_value]
        self.data = Data2D.from_polygon_list(self.data)
        return self.data, True

    def get_labels(
        self,
    ) -> List[str]:
        """Returns a list of fields names displayable with this interface

        Returns
        -------
        List[str]
            List of fields names
        """
        labels = list(self.fields_iterations.keys()) + [MESH]
        return labels

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
        if profile_time:
            start_time = time.time()
        if value_label == MESH:
            return {str(v): np.nan for v in cells}

        if "Iteration" not in options:
            print(f"Iteration not found in medcoupling option, setting {self.fields_iterations[value_label][0][0]}")
            options["Iteration"] = self.fields_iterations[value_label][0][0]
        if "Order" not in options:
            print(f"Order not found in medcoupling option, setting {self.fields_iterations[value_label][0][1]}")
            options["Order"] = self.fields_iterations[value_label][0][1]

        field_np_array = None

        if value_label in self.fields:
            field_np_array = self.fields[value_label]
        elif value_label in self.field_doubles:
            field_np_array = self.field_doubles[value_label].getArray().toNumPyArray()
        else:
            print(f"Reading MEDCouplingFieldDouble in {self.file_path}")
            # print("Checking", value_label, "in", self.fields_iterations, field in self.fields_iterations.keys())

            if value_label in self.fields_iterations:
                # if "Iteration" in options and "Order" in options and (options["Iteration"], options["Order"]) in self.fields_iterations[value_label]:
                if True:
                    field_name = value_label.split("@")[0]
                    field: medcoupling.MEDCouplingFieldDouble = medcoupling.ReadField(
                        medcoupling.ON_CELLS,
                        self.file_path,
                        self.meshnames[0],
                        0,
                        field_name,
                        options["Iteration"],
                        options["Order"],
                    )
                    field_array: medcoupling.DataArrayDouble = field.getArray()
                    field_np_array: np.ndarray = field_array.toNumPyArray()

                    if "@" in value_label:
                        components: List[str] = field_array.getInfoOnComponents()
                        field_np_array = field_np_array[
                            :, components.index(value_label.split("@")[1])
                        ]
                    else:
                        field_np_array = field_np_array

            if field_np_array is not None:
                self.fields[value_label] = field_np_array

        if field_np_array is not None:
            indexes = np.array(list(self.cell_dict.values())).astype(int)
            values = field_np_array[indexes[np.array(cells).astype(int)].tolist()]

            value_dict = dict(zip(np.array(cells).astype(str), values))

            if profile_time:
                print(f"Get value dict time: {time.time() - start_time}")
            return value_dict

        raise NotImplementedError(
            f"The field {value_label} is not implemented, fields available : {self.get_labels()}"
        )

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

        return VisualizationMode.FROM_VALUE

    def get_file_input_list(self) -> List[Tuple[str, str]]:
        """Returns a list of file label and its description for the GUI

        Returns
        -------
        List[Tuple[str, str]]
            List of (file label, description)
        """
        return [(GEOMETRY, "MED file."), (CSV, "CSV result file.")]

    def getInputMEDDoubleFieldTemplate(self, field_name: str):
        mcfield = medcoupling.MEDCouplingFieldDouble(
            medcoupling.ON_CELLS, medcoupling.ONE_TIME
        )
        mcfield.setName(field_name)
        mcfield.setTime(0.0, 0, 0)
        mcfield.setMesh(self.mesh)
        array = medcoupling.DataArrayDouble([0.0] * self.mesh.getNumberOfCells())
        mcfield.setArray(array)
        if field_name in self.field_doubles:
            mcfield.setNature(self.field_doubles[field_name].getNature())
        else:
            print(
                field_name,
                f"not found in self.fields, available keys: {list(self.field_doubles.keys())}.",
            )
        return mcfield

    def setInputMEDDoubleField(
        self, field_name: str, field: medcoupling.MEDCouplingFieldDouble
    ):
        if field_name in self.fields:
            del self.fields[field_name]

        self.field_doubles[field_name] = field

    def setTime(self, time_: float):
        pass

    def get_iterations(self,) -> Dict[str, List[Tuple[int, int]]]:
        """Returns the fields iterations

        Returns
        -------
        Dict[str, List[Tuple[int, int]]]
            Iterations/orders per field
        """
        return self.fields_iterations

    def get_bounding_box(self,) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Returns the mesh bounding box

        Returns
        -------
        Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
            Mesh bounding box : ((minx, maxx), (miny, maxy), (minz, maxz))
        """
        return self.mesh.getBoundingBox()

    def save(self, file_path: Path, include_files: bool):
        """Pickle saves the slave content to a file, allows slave state reload.

        Two modes are available:
            -   If **include_files** is at True, all loaded data are saved, the pickled file can be loaded on its own to recover last session.
            -   If **include_files** is at False, only the computed data are loaded, enabling faster first computation allowing a smaller pickle file size.

        Parameters
        ----------
        file_path : Path
            File in which save the file
        include_files : bool
            Included loaded file
        """
        os.makedirs(Path(file_path).parent, exist_ok=True)
        with open(file_path, "wb") as f:
            if include_files:
                data = (
                    scivianna.__version__,
                    medcoupling.__version__,
                    sys.version,
                    include_files,
                    "MEDInterface",
                    self.data,
                    self.file_path,
                    self.meshnames,
                    self.mesh,
                    self.fieldnames,
                    self.fields,
                    self.field_doubles,
                    self.fields_iterations,
                    self.cell_dict,
                    self.last_computed_frame
                )
            else:
                data = (
                    scivianna.__version__,
                    medcoupling.__version__,
                    sys.version,
                    include_files,
                    "MEDInterface",
                    self.last_computed_frame,
                    self.data,
                    self.cell_dict
                )

            pickle.dump(data, f)

    def load(self, file_path: Path, include_files: bool):
        """Pickle loads the slave content to a file, allows slave state reload

        Two modes are available:
            -   If **include_files** is at True, all loaded data are saved, the pickled file can be loaded on its own to recover last session.
            -   If **include_files** is at False, only the computed data are loaded, enabling faster first computation allowing a smaller pickle file size.

        Parameters
        ----------
        file_path : Path
            File from which load the slave
        include_files : bool
            Included loaded file
        """
        if not os.path.isfile(file_path):
            raise ValueError(f"Provided path {file_path} does not exist")

        with open(file_path, "rb") as f:
            data = pickle.load(f)

            assert len(data) > 5, "Loaded data is not meant for MEDInterface"
            version, med_version, python_version, inc_files, interface_name = data[:5]
            if version != scivianna.__version__:
                warning(f"Loading file built with scivianna {version}, current version : {scivianna.__version__}.")
            if med_version != medcoupling.__version__:
                warning(f"Loading file built with medcoupling {med_version}, current version : {medcoupling.__version__}.")
            if python_version != sys.version:
                warning(f"Loading file built with Python {python_version}, current version : {sys.version}.")

            assert inc_files == include_files, f"Loaded file has in include_files at {inc_files}, currently calling with include_files at {include_files}."

            assert interface_name == "MEDInterface", f"Loaded file is built by interface {interface_name}, trying to load with MEDInterface."

            if include_files:
                (
                    self.data,
                    self.file_path,
                    self.meshnames,
                    self.mesh,
                    self.fieldnames,
                    self.fields,
                    self.field_doubles,
                    self.fields_iterations,
                    self.cell_dict,
                    self.last_computed_frame
                ) = data[5:]

            else:
                (
                    self.last_computed_frame,
                    self.data,
                    self.cell_dict
                ) = data[5:]


if __name__ == "__main__":
    from scivianna.notebook_tools import _show_panel

    slave = ComputeSlave(MEDInterface)
    # slave.read_file("/volatile/catA/tmoulignier/Workspace/some_holoviz/jdd/mesh_hexa_3d.med", GEOMETRY)
    slave.read_file(
        "/volatile/catA/tmoulignier/Workspace/some_holoviz/src/scivianna/input_file/INTEGRATED_POWER.med",
        GEOMETRY,
    )

    _show_panel(VisualizationPanel(slave, name="MED visualizer"))
