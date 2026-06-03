import functools
import pickle
from typing import Dict, List, Tuple, Type, Union
import panel as pn
import panel_material_ui as pmui
from enum import Enum
from dataclasses import dataclass

from pathlib import Path

from scivianna.interface.generic_interface import GenericInterface
from scivianna.utils.interface_tools import GenericInterfaceEnum
from scivianna.interface import INTERFACES, register_interface
from scivianna.layout.generic_layout import GenericLayout
from scivianna.panel.visualisation_panel import ComputeSlave, VisualizationPanel
from scivianna.panel.panel_1d import Panel1D
from scivianna.panel.panel_2d import Panel2D
from scivianna.component.splitjs_component import (
    SplitJSVertical,
    SplitJSHorizontal,
)
from scivianna.utils.serialization import (
    save_layout_to_zip,
    load_layout_from_zip,
    _serialize_split_item,
    _deserialize_split_item,
    restore_extensions_state,
)

card_style = {}

class SplitDirection(Enum):
    """
    Enum defining the direction of the split.
    """

    VERTICAL = 0
    """The split is done with a vertical line, the two panels will be side by side"""
    HORIZONTAL = 1
    """The split is done with a horizontal line, the two panels will be on top of each other"""


@dataclass
class SplitItem:
    panel_1: Union[VisualizationPanel, "SplitItem"]
    """First visualization panel or other SplitItem"""
    panel_2: Union[VisualizationPanel, "SplitItem"]
    """First visualization panel or other SplitItem"""

    direction: SplitDirection
    """Direction of the split between the two panels"""


class SplitLayout(GenericLayout):
    """Displayable that lets arranging several VisualizationPanel"""

    split_item: SplitItem
    """ Name - Intricated SplitItem defining the displayed panels
    """

    main_frame: pn.Column
    """ Main frame : gridstack of different VisualizationPanel main_frame
    """

    available_interfaces: Dict[Union[str, GenericInterfaceEnum], Type[GenericInterface]]
    """ Available interface classes to switch from one to another
    """

    def __init__(
        self,
        split_item: SplitItem,
        additional_interfaces: Dict[
            Union[str, GenericInterfaceEnum], Type[GenericInterface]
        ] = {},
        add_run_button: bool = False,
    ):
        """VisualizationGridStack constructor

        Parameters
        ----------
        split_item : SplitItem
            Item defining the panels to display
        add_run_button:bool = False
            Add a run button to add an automatic update of the frames in the case of visualizer coupling.

        Raises
        ------
        TypeError
            One of the additional interfaces classes does not inherit from GenericInterface
        """

        self.split_item = split_item

        super().__init__(self.get_panels_dict(split_item), additional_interfaces, add_run_button)

        """
            Building interface
        """
        self.main_frame = pn.Row(
            self.gui.make_panel(),
            self.build_split_item(split_item),
            height_policy="max",
            width_policy="max",
            margin=0,
            scroll=False,
        )

        self.change_current_frame()


    def change_code_interface(self, event):
        super().change_code_interface(event)
        current_frame = self.current_frame
    
        self.update_interface_in_split_item(current_frame, self.visualisation_panels[current_frame], self.split_item)

        if self.code_interface_to_update:
            self.reset_interface()
            
        self.change_current_frame()

    def current_split_item(
        self, panel_name: str, item: Union[SplitItem, VisualizationPanel]
    ) -> Tuple[Union[SplitItem, VisualizationPanel], int]:
        """Recursively looks for a VisualizationPanel named panel_name in the given item

        Parameters
        ----------
        panel_name : str
            Panel name to find
        item : Union[SplitItem, VisualizationPanel]
            Intricated item in which find the panel

        Raises
        ------
        TypeError
            provided split_item is neither a SplitItem or a VisualizationPanel
        """
        if isinstance(item, SplitItem):
            item_1, index_1 = self.current_split_item(panel_name, item.panel_1)
            item_2, index_2 = self.current_split_item(panel_name, item.panel_2)

            if item_1 is not None:
                if isinstance(item_1, VisualizationPanel):
                    item_1 = item
                if index_1 is None:
                    index_1 = 1
                return item_1, index_1
            if item_2 is not None:
                if isinstance(item_2, VisualizationPanel):
                    item_2 = item
                if index_2 is None:
                    index_2 = 2
                return item_2, index_2

        elif isinstance(item, VisualizationPanel):
            if item.panel_name == panel_name:
                return item, None
        else:
            raise TypeError(
                f"SplitItem or VisualizationPanel expected, found {type(item)}"
            )

        return None, None


    def build_split_item(
        self, split_item: Union[SplitItem, VisualizationPanel],
    ) -> Union[SplitJSVertical, SplitJSHorizontal]:
        """Converts a SplitItem object in its Viewable counterpart

        Parameters
        ----------
        split_item : SplitItem
            SplitItem to convert in its GUI version

        Returns
        -------
        Union[SplitJSVertical, SplitJSHorizontal]
            GUI element to display in self.main_frame

        Raises
        ------
        TypeError
            provided split_item is neither a SplitItem or a VisualizationPanel
        """
        if isinstance(split_item, SplitItem):
            if split_item.direction == SplitDirection.VERTICAL:
                return SplitJSVertical(
                    left=self.build_split_item(split_item.panel_1),
                    right=self.build_split_item(split_item.panel_2),
                    sizing_mode="stretch_both",
                    margin=0,
                )
            else:
                return SplitJSHorizontal(
                    bottom=self.build_split_item(split_item.panel_1),
                    top=self.build_split_item(split_item.panel_2),
                    sizing_mode="stretch_both",
                    margin=0,
                )
            
        elif isinstance(split_item, VisualizationPanel):
            return split_item.figure
        
        else:
            raise TypeError(
                f"SplitItem or VisualizationPanel expected, found {type(split_item)}"
            )

    def update_interface_in_split_item(
        self, panel_name:str, new_panel:VisualizationPanel, split_item:Union[SplitItem, VisualizationPanel]
    ):
        """Replaces a panel in the split item object

        Parameters
        ----------
        panel_name : str
            Panel name
        new_panel : VisualizationPanel
            New visualization panel
        split_item : Union[SplitItem, VisualizationPanel]
            SplitItem in which replace the panel

        Raises
        ------
        TypeError
            provided split_item is neither a SplitItem or a VisualizationPanel
        """
        if isinstance(split_item, SplitItem):
            if isinstance(split_item.panel_1, VisualizationPanel) and split_item.panel_1.panel_name == panel_name:
                split_item.panel_1 = new_panel
            elif isinstance(split_item.panel_2, VisualizationPanel) and split_item.panel_2.panel_name == panel_name:
                split_item.panel_2 = new_panel
            else:
                self.update_interface_in_split_item(panel_name, new_panel, split_item.panel_1)
                self.update_interface_in_split_item(panel_name, new_panel, split_item.panel_2)

        elif isinstance(split_item, VisualizationPanel):
            if split_item.panel_name == panel_name:
                split_item = new_panel
        else:
            raise TypeError(
                f"SplitItem or VisualizationPanel expected, found {type(split_item)}"
            )
        
    def get_panels_dict(self, split_item: SplitItem) -> Dict[str, VisualizationPanel]:
        """Returns all provided VisualizationPanels in a dictionnary

        Parameters
        ----------
        split_item : SplitItem
            SplitItem recursively containing VisualizationPanels

        Returns
        -------
        Dict[str, VisualizationPanel]
            VisualizationPanels dictionnary
        """
        visualisation_panels : Dict[str, VisualizationPanel] = {}
        
        if isinstance(split_item, VisualizationPanel):
            visualisation_panels[split_item.panel_name] = split_item
        
        elif isinstance(split_item, SplitItem):
            if isinstance(split_item.panel_1, SplitItem):
                visualisation_panels = {**visualisation_panels, **self.get_panels_dict(split_item.panel_1)}
            else:
                visualisation_panels[split_item.panel_1.panel_name] = split_item.panel_1
            if isinstance(split_item.panel_2, SplitItem):
                visualisation_panels = {**visualisation_panels, **self.get_panels_dict(split_item.panel_2)}
            else:
                visualisation_panels[split_item.panel_2.panel_name] = split_item.panel_2
                
        return visualisation_panels

    
    @pn.io.hold()
    def reset_interface(self,):
        """Rebuilds the interface based on up-to-date SplitItem
        """

        #   We hide the history of objects in self.main_frame and adds a new one
        #   This practice prevents the garbage collector to delete objects that are still to be used
        self.main_frame.objects = [
            self.main_frame.objects[0],
            self.build_split_item(self.split_item)
        ]

        self.set_to_frame(self.current_frame)
        self.change_current_frame()
            
    @pn.io.hold()
    def duplicate(self, horizontal: bool):
        """Split the panel, the new panel is a copy of the first, all panels are duplicated.

        Parameters
        ----------
        horizontal : bool
            Whether the cut should be horizontal or vertical
        """
        current_frame = self.current_frame

        new_frame = self.visualisation_panels[current_frame].duplicate()
        new_frame.figure.show_buttons()

        while new_frame.panel_name in self.visualisation_panels:
            new_frame.rename(new_frame.get_new_name())

        self.visualisation_panels[new_frame.panel_name] = new_frame
        self.register_panel(new_frame)

        parent_split, split_index = self.current_split_item(
            current_frame, self.split_item
        )

        if horizontal:
            split_item = SplitItem(
                self.visualisation_panels[current_frame],
                self.visualisation_panels[new_frame.panel_name],
                SplitDirection.HORIZONTAL,
            )

        else:
            split_item = SplitItem(
                self.visualisation_panels[current_frame],
                self.visualisation_panels[new_frame.panel_name],
                SplitDirection.VERTICAL,
            )

        if split_index is None:
            # Index is None if there is only one visualizationpanel in self.main_frame
            self.split_item = split_item
        elif split_index == 1:
            parent_split.panel_1 = split_item
        elif split_index == 2:
            parent_split.panel_2 = split_item
        else:
            raise ValueError(f"Unexpected split index {split_index}")

        self.reset_interface()

    #
    #   Save/Restore functionality
    #   These methods now delegate to scivianna.utils.serialization module
    #
    def save_to_zip(self, file_path: Union[str, Path], include_files: bool = True):
        """Saves the layout configuration and slave data to a zip file.
        
        The zip file contains:
        - layout.json: JSON file describing the layout structure, current frame, 
          and for each slave: panel name, slave info, and associated interface name
        - data/: Folder containing serialized data for each slave
        
        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the zip file to create
        include_files : bool = True
            If True, includes loaded files in the slave serialization
            
        Returns
        -------
        Path
            Path to the created zip file
        """
        return save_layout_to_zip(self, file_path, include_files)

    @classmethod
    def restore_from_zip(
        cls, 
        file_path: Union[str, Path], 
        include_files: bool = True,
        additional_interfaces: Dict[Union[str, GenericInterfaceEnum], Type[GenericInterface]] = {},
        add_run_button: bool = False
    ) -> "SplitLayout":
        """Restores a SplitLayout from a zip file.
        
        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the zip file to restore from
        include_files : bool = True
            If True, includes loaded files in the slave deserialization
        additional_interfaces : Dict = {}
            Additional interfaces to register
        add_run_button : bool = False
            Whether to add a run button for coupling simulations
            
        Returns
        -------
        SplitLayout
            Restored SplitLayout instance
        """
        return load_layout_from_zip(
            file_path, 
            include_files, 
            additional_interfaces, 
            add_run_button
        )
