import functools
from typing import Dict, List, Tuple, Type, Union
import panel as pn
import holoviews as hv

from pathlib import Path

from scivianna.interface.generic_interface import GenericInterface
from scivianna.layout.generic_layout import GenericLayout
from scivianna.panel.visualisation_panel import ComputeSlave, VisualizationPanel
from scivianna.utils.interface_tools import (
    GenericInterfaceEnum,
)
from scivianna.component.gridstack_component import CustomGridStack
from scivianna.utils.serialization import (
    save_gridstack_to_zip,
    load_gridstack_from_zip,
)

pn.extension()
hv.extension("bokeh")

card_style = {}

class GridStackLayout(GenericLayout):
    """Displayable that lets arranging several VisualizationPanel"""

    visualisation_panels: Dict[str, VisualizationPanel]
    """ Name - VisualizationPanel dictionnary
    """
    bounds_x: Dict[str, List[int]]
    """ Name - position in the gridstack along the horizontal axis dictionnary
    """
    bounds_y: Dict[str, List[int]]
    """ Name - position in the gridstack along the vertical axis dictionnary
    """

    main_frame: pn.Column
    """ Main frame : gridstack of different VisualizationPanel main_frame
    """

    available_interfaces: Dict[Union[str, GenericInterfaceEnum], Type[GenericInterface]]
    """ Available interface classes to switch from one to another
    """

    def __init__(
        self,
        visualisation_panels: Dict[str, VisualizationPanel],
        bounds_x: Dict[str, Tuple[int]],
        bounds_y: Dict[str, Tuple[int]],
        additional_interfaces: Dict[
            Union[str, GenericInterfaceEnum], Type[GenericInterface]
        ] = {},
    ):
        """VisualizationGridStack constructor

        Parameters
        ----------
        visualisation_panels : Dict[str, VisualizationPanel]
            Dictionnary containing the VisualizationPanels
        bounds_x : Dict[str, Tuple[int]]
            Dictionnary containing position in the gridstack along the horizontal axis
        bounds_y : Dict[str, Tuple[int]]
            Dictionnary containing position in the gridstack along the vertical axis

        Raises
        ------
        TypeError
            One of the additional interfaces classes does not inherit from GenericInterface
        """
        assert set(bounds_x.keys()) == set(visualisation_panels.keys()), "arguments bounds_x keys are different than visualisation_panels keys"
        assert set(bounds_y.keys()) == set(visualisation_panels.keys()), "arguments bounds_y keys are different than visualisation_panels keys"

        self.bounds_x = bounds_x
        self.bounds_y = bounds_y

        super().__init__(visualisation_panels, additional_interfaces)

        size_x = max(max([x_range for x_range in bounds_x.values()]))
        size_y = max(max([y_range for y_range in bounds_y.values()]))

        size_x = max(1, size_x)
        size_y = max(1, size_y)


        """
            Allow resize frames check box
        """
        self.allow_resize_box = pn.widgets.Checkbox(
            label="Allow frames resize", value=False
        )

        def enable_disable_resize(event):
            self.enable_disable_pan()

        self.allow_resize_box.param.watch(enable_disable_resize, "value")



        """
            Building interface
        """
        self.main_frame = pn.Row(
            self.gui.make_panel(),
            CustomGridStack(sizing_mode="stretch_both", allow_resize=True),
            sizing_mode="stretch_both",
            margin=0,
            scroll=False,
        )

        self.make_grid_stack()
        self.enable_disable_pan()

        self.layout_extension.add_widget(self.allow_resize_box)


    @pn.io.hold()
    def change_code_interface(self, event):
        super().change_code_interface(event)
    
        self.make_grid_stack()
        self.change_current_frame(None)

    @pn.io.hold()
    def change_current_frame(self, *args, **kwargs):
        """Swap the the active panel

        Parameters
        ----------
        event : Any
            Event to make the function linkable to the gridstack
        """
        super().change_current_frame(*args, **kwargs)
        current_frame = self.current_frame
        if self.bounds_x[current_frame][1] - self.bounds_x[current_frame][0] > 1:
            self.layout_extension.duplicate_horizontally_button.disabled = False
        else:
            self.layout_extension.duplicate_horizontally_button.disabled = True

        if self.bounds_y[current_frame][1] - self.bounds_y[current_frame][0] > 1:
            self.layout_extension.duplicate_horizontally_button.disabled = False
        else:
            self.layout_extension.duplicate_horizontally_button.disabled = True

    def get_grid(self) -> CustomGridStack:
        """Returns the gridstack object

        Returns
        -------
        GridStack
            Currently displayed gridstack
        """
        return self.main_frame.objects[-1]

    def enable_disable_pan(self):
        """Enable - disable figure panning"""
        if self.allow_resize_box.value:
            self.disable_figures_pan()
        else:
            self.enable_figures_pan()

        grid = self.get_grid()
        grid.allow_resize = True  # self.allow_resize_box.value
        grid.allow_drag = self.allow_resize_box.value

    def make_grid_stack(self):
        """Updates the displayed gridstack object (called after a panel split/delete)"""

        size_x = max(max([x_range for x_range in self.bounds_x.values()]))
        size_y = max(max([y_range for y_range in self.bounds_y.values()]))

        size_x = max(1, size_x)
        size_y = max(1, size_y)

        self.get_grid().clear_objects()

        for element in self.visualisation_panels:
            #  As disgusting as it looks, sleeping here helps the python to sychronize with the Javascript while splitting and avoir throwing an error
            if len(self.bounds_x[element]) == 0 and len(self.bounds_y[element]) == 0:
                self.get_grid()[:, :] = self.visualisation_panels[element].figure
                self.get_grid().add_object(
                    self.visualisation_panels[element].figure,
                    (0, size_x),
                    (0, size_y),
                )

            elif len(self.bounds_x[element]) == 0:
                self.get_grid()[
                    self.bounds_y[element][0]: self.bounds_y[element][1], :
                ] = self.visualisation_panels[element].figure
                self.get_grid().add_object(
                    self.visualisation_panels[element].figure,
                    (self.bounds_y[element][0], self.bounds_y[element][1]),
                    (0, size_y),
                )

            elif len(self.bounds_y[element]) == 0:
                self.get_grid()[
                    :, self.bounds_x[element][0]: self.bounds_x[element][1]
                ] = self.visualisation_panels[element].figure
                self.get_grid().add_object(
                    self.visualisation_panels[element].figure,
                    (0, size_x),
                    (self.bounds_x[element][0], self.bounds_x[element][1]),
                )

            else:
                self.get_grid()[
                    self.bounds_y[element][0]: self.bounds_y[element][1],
                    self.bounds_x[element][0]: self.bounds_x[element][1],
                ] = self.visualisation_panels[element].figure
                self.get_grid().add_object(
                    self.visualisation_panels[element].figure,
                    (self.bounds_y[element][0], self.bounds_y[element][1]),
                    (self.bounds_x[element][0], self.bounds_x[element][1]),
                )

        self.enable_disable_pan()
    

    def disable_figures_pan(self):
        """Disable all figures pan"""
        for component in self.visualisation_panels:
            self.visualisation_panels[component].plotter._disable_interactions(True)

    def enable_figures_pan(self):
        """Enable all figures pan"""
        for component in self.visualisation_panels:
            self.visualisation_panels[component].plotter._disable_interactions(False)

    @pn.io.hold()
    def duplicate(self, horizontal: bool):
        """Split the panel, the new panel is a copy of the first, all panels are duplicated.

        Parameters
        ----------
        horizontal : bool
            Whether the cut should be horizontal or vertical
        """
        current_frame = self.current_frame
        if horizontal:
            cut_possible = (
                self.bounds_y[current_frame][1] - self.bounds_y[current_frame][0] > 1
            )
        else:
            cut_possible = (
                self.bounds_x[current_frame][1] - self.bounds_x[current_frame][0] > 1
            )

        if cut_possible:
            new_visualisation_panels:Dict[str, VisualizationPanel] = {}

            old_x_min = self.bounds_x[current_frame][0]
            old_x_max = self.bounds_x[current_frame][1]
            old_y_min = self.bounds_y[current_frame][0]
            old_y_max = self.bounds_y[current_frame][1]

            if horizontal:
                cut_coordinate = int(0.5 * (old_y_min + old_y_max))
            else:
                cut_coordinate = int(0.5 * (old_x_min + old_x_max))

            new_frame = self.visualisation_panels[current_frame].duplicate()
            
            new_frame.figure.show_buttons()

            while new_frame.panel_name in self.visualisation_panels:
                new_frame.rename(new_frame.get_new_name())

            new_visualisation_panels[new_frame.panel_name] = new_frame
            self.register_panel(new_frame)

            if horizontal:
                self.bounds_x[new_frame.panel_name] = (old_x_min, old_x_max)

                self.bounds_y[current_frame] = (old_y_min, cut_coordinate)
                self.bounds_y[new_frame.panel_name] = (cut_coordinate, old_y_max)
            else:
                self.bounds_x[current_frame] = (old_x_min, cut_coordinate)
                self.bounds_x[new_frame.panel_name] = (cut_coordinate, old_x_max)

                self.bounds_y[new_frame.panel_name] = (old_y_min, old_y_max)

            for panel_name in self.visualisation_panels:
                new_visualisation_panels[panel_name] = self.visualisation_panels[
                    panel_name
                ].duplicate(keep_name=True)
                
                self.register_panel(new_visualisation_panels[panel_name])

            self.visualisation_panels = new_visualisation_panels

            self.make_grid_stack()

            self.reset_interface()

            
    @pn.io.hold()
    def reset_interface(self,):
        """Rebuilds the interface based on up-to-date SplitItem
        """

        #   We hide the history of objects in self.main_frame and adds a new one
        #   This practice prevents the garbage collector to delete objects that are still to be used
        # self.main_frame.objects = [
        #     self.main_frame.objects[0],
        #     self.get_grid()
        # ]

        self.set_to_frame(self.current_frame)
        self.change_current_frame()

    def save_to_zip(
        self,
        file_path: Union[str, Path],
        include_files: bool = True
    ) -> Path:
        """
        Saves the GridStackLayout configuration and slave data to a zip file.
        
        The zip file contains:
        - layout.json: JSON file describing the layout structure, current frame, 
          bounds, and for each panel: slave info and associated interface name
        - data/: Folder containing serialized data for each slave
        
        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the output zip file
        include_files : bool = True
            If True, includes loaded files in the slave serialization
            
        Returns
        -------
        Path
            Path to the saved zip file
        """
        return save_gridstack_to_zip(self, file_path, include_files=include_files)

    @classmethod
    def restore_from_zip(
        cls,
        file_path: Union[str, Path],
        include_files: bool = True,
        additional_interfaces: Dict[Union[str, GenericInterfaceEnum], Type[GenericInterface]] = {}
    ) -> "GridStackLayout":
        """
        Restores a GridStackLayout from a zip file.
        
        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the input zip file
        include_files : bool = True
            If True, includes loaded files in the slave deserialization
        additional_interfaces : Dict = {}
            Additional interfaces to register
            
        Returns
        -------
        GridStackLayout
            Restored GridStackLayout instance
        """
        return load_gridstack_from_zip(
            file_path,
            include_files=include_files,
            additional_interfaces=additional_interfaces,
        )
            