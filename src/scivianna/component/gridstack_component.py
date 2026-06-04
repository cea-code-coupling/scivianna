from __future__ import annotations

from typing import Dict, Tuple

import param
import panel as pn
pn.extension("gridstack")
from panel.layout.gridstack import GridStack



class CustomGridStack(GridStack):
    """
    Custom version of the panel.layout.gridstack.GridStack to allow clearing the objects properly.
    """
    object_dict:Dict[str, pn.viewable.Viewable] = {}
    """Dictionnary of viewables that have to be manually added using the function add_object"""

    @param.depends('state', watch=True)
    def _update_objects(self):
        """Updates the range of the objects after a state change triggered by the javascript.
        """
        objects = {}

        for p in self.state:
            objects[(p['y0'], p['x0'], p['y1'], p['x1'])] = self.object_dict[p['id']]
            
        self.objects.clear()
        self.objects.update(objects)

        # self.print_stored_items()
        # self.print_state()

        self._update_sizing()


    @param.depends('objects', watch=True)
    def _update_sizing(self):
        """Updates the objects internal size to match the data stored in the gridstack and in the state attribute
        """
        if self.ncols and self.width:
            width = self.width/self.ncols
        else:
            width = 0

        if self.nrows and self.height:
            height = self.height/self.nrows
        else:
            height = 0

        for (y0, x0, y1, x1), obj in self.objects.items():
            x0 = 0 if x0 is None else x0
            x1 = (self.ncols) if x1 is None else x1
            y0 = 0 if y0 is None else y0
            y1 = (self.nrows) if y1 is None else y1
            h, w = y1-y0, x1-x0

            properties = {}
            if self.sizing_mode in ['fixed', None]:
                if width:
                    properties['width'] = int(w*width)
                if height:
                    properties['height'] = int(h*height)
            else:
                properties['sizing_mode'] = self.sizing_mode
                if 'width' in self.sizing_mode and height:
                    properties['height'] = int(h*height)
                elif 'height' in self.sizing_mode and width:
                    properties['width'] = int(w*width)
            obj.param.update(**{
                k: v for k, v in properties.items()
                if not obj.param[k].readonly
            })

    # def update_render(self,):
    def clear_objects(self,):
        """Clears the objects list and the object dictionnary.
        """
        self.objects.clear()
        self.object_dict = {}
        
    def print_stored_items(self,):
        """Prints the information of the objects stored
        """
        print("Stored objects : ")
        for loc, obj in self.objects.items():
            print(f" - {loc} - {id(obj)}: {obj}")

        print()

    def print_state(self,):
        """Prints the current gridstack state (viewable id to coordinates in the grid).
        """
        print("State : ")
        for p in self.state:
            print(f" - {p['id']} : {p['y0']}, {p['x0']}, {p['y1']}, {p['x1']}")
        print()
        
    def add_object(self, obj:pn.viewable.Viewable, 
                            x_range:Tuple[int, int],
                            y_range:Tuple[int, int]):
        """Adds an object to the gridstack objects dict

        Parameters
        ----------
        obj : pn.viewable.Viewable
            Panel added
        x_range : Tuple[int, int]
            Horizontal range used to update the number of rows and columns
        y_range : Tuple[int, int]
            Vertical range used to update the number of rows and columns
        """
        self.object_dict[str(id(obj))] = obj
