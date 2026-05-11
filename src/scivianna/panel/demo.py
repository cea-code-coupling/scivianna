from pathlib import Path
from typing import Dict, Union
import panel_material_ui as pmui
from panel.reactive import ReactiveHTML
from panel.custom import Child
import param
import panel as pn

import scivianna
from scivianna.layout.generic_layout import GenericLayout
from scivianna.panel.visualisation_panel import VisualizationPanel

class DemoOverlay(ReactiveHTML):
    """This component allow displaying a menu in front of a background."""

    app = Child()
    """App to show as backgorund"""
    menu = Child()
    """Menu to display"""

    distance_from_right: param.String = param.String("10px")
    """Distance between the menu and the right of the background"""
    distance_from_bottom: param.String = param.String("10px")
    """Distance between the menu and the bottom of the background"""

    _template ="""
        
                    <div id="app-container" style="position: relative; width: 100%; height: 100%;">
                        <div id="app" style="width: 100%; height: 100%;">
                            ${app}
                        </div>
                        <div id="menu-container" style="position: absolute; bottom: ${distance_from_bottom}; right: ${distance_from_right};">
                            ${menu}
                        </div>
                    </div>
                """
    """HTML code of the elemet display"""


class Demonstrator:
    """The demonstrator allows having several apps in one and switch between them"""

    menu: pmui.MenuButton
    """Menu button to switch between apps"""

    guis: Dict[str, Union[VisualizationPanel, GenericLayout]]
    """Panel objects for each app"""

    main_frame: DemoOverlay
    """Object to display"""

    def __init__(self, 
        guis: Dict[str, Union[VisualizationPanel, GenericLayout]], 
        icons:Dict[str, str]
    ):
        """Constructor of a scivianna app demonstrator.

        Parameters
        ----------
        guis : Dict[str, Union[VisualizationPanel, GenericLayout]]
            Panel GUI associated to each apps
        icons : Dict[str, str]
            Icons per app name (displayed in the menu)
        """
        self.guis = guis

        for g in range(1, len(self.guis.keys())):
            self.guis[list(self.guis.keys())[g]].visible = False

        for key in guis:
            if not key in icons:
                icons[key] = None

        self.menu = pmui.SpeedDial(
            direction = "left",
            items=[
                {
                    "label": key,
                    "icon": icons[key],
                }
                for key in guis
            ], 
            label='Examples', 
            icon=(Path(scivianna.__file__).parent / "icon" / "apps.svg").read_text().strip(),
        )

        self.menu.param.watch(self.switch_app, "value")

        self.apps = pmui.Row(*[g for g in guis.values()], sizing_mode = "stretch_both")

        self.main_frame = DemoOverlay(
            app = self.apps,
            menu = self.menu, 
            sizing_mode="stretch_both",
            margin=0
        )

    @pn.io.hold()
    def switch_app(self, *args, **kwargs):
        """Function triggered to edit GUIs visibility based on the menu value
        """
        for gui in self.guis:
            self.guis[gui].visible = gui == self.menu.value["label"]

    def __panel__(self,):
        """Returns the displayed object if doing demonstrator.show/serve
        """
        return self.main_frame