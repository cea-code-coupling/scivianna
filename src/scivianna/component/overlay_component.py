import panel as pn
from panel.reactive import ReactiveHTML
from panel.custom import Child, ReactiveHTML
import param


class Overlay(ReactiveHTML):
    """This component allow displaying buttons over a figure only when the mouse is in the area."""
    key = param.String("")

    figure = Child()
    """Figure : main component"""
    button = Child()
    """First button (displayed on the top right of the figure)"""
    title = Child()
    """Figure title (displayed on the top left of the figure)"""

    distance_from_top: param.String = param.String("10px")
    """Distance between the button and the right of the figure (increased when the axis are displayed)"""
    distance_from_left: param.String = param.String("10px")
    """Distance between the title and the left of the figure (increased when the axis are displayed)"""

    _template = """
<div id="figure-container" 
     onmouseenter="${script('enter')}"
     onmouseleave="${script('leave')}"
     style="position: relative; width: 100%; height: 100%;">
  <!-- Figure -->
  <div id="figure" style="width: 100%; height: 100%;">
    ${figure}
  </div>

  <!-- Container for button and title (overlay, top-aligned, vertically centered) -->
  <div id="overlay-controls" style="position: absolute; top: ${distance_from_top}; left: ${distance_from_left}; display: flex; align-items: center; gap: 0.2em; pointer-events: none;">
    <!-- Button -->
    <div id="button-container_1" style="pointer-events: auto;">
      ${button}
    </div>

    <!-- Title -->
    <div id="title-container" style="white-space: nowrap;">
      ${title}
    </div>
  </div>
</div>
                """
    """HTML code of the elemet display"""

    # Script to listen to key events and update the `key` parameter accordingly
    _scripts = {
    "enter": """
        state.inside = true;

        if (!state.keyHandler) {
            state.keyHandler = (event) => {
                if (state.inside)
                    data.key = event.key;
            };
            window.addEventListener("keydown", state.keyHandler);
        }
    """,

    "leave": """
        state.inside = false;
    """,

    "remove": """
        if (state.keyHandler)
            window.removeEventListener("keydown", state.keyHandler);
    """
}

    @pn.io.hold()
    def hide_buttons(self, *args, **kwargs):
        """Hide all element on top of the figure
        """
        if self.button is not None:
            self.button.visible = False
        if self.title is not None:
            self.title.visible = False

    @pn.io.hold()
    def show_buttons(self, *args, **kwargs):
        """Show all element on top of the figure if the mouse is in the frame"""
        if self.button is not None:
            self.button.visible = True
        if self.title is not None:
            self.title.visible = True
