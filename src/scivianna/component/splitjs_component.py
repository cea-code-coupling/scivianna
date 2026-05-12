import panel as pn
from panel.custom import Child, JSComponent, ReactComponent

split_line_width = 4

CSS_vertical = f"""
.split {{
    display: flex;
    flex-direction: row;
    height: 100%;
    width: 100%;
}}

.gutter {{
    background-color: #eee;
    background-repeat: no-repeat;
    background-position: 50%;
}}

.gutter.gutter-horizontal {{
    cursor: col-resize;
    width: {split_line_width}px !important;
}}
"""

CSS_horizontal = f"""
.split {{
    height: 100%;
    width: 100%;
}}

.gutter {{
    background-color: #eee;
    background-repeat: no-repeat;
    background-position: 50%;
}}

.gutter.gutter-vertical {{
    height: {split_line_width}px !important;
    cursor: row-resize;
}}
"""

class SplitJSVertical(JSComponent):
    """
        Javascript component that hosts two Viewables side by side, and separate them with a draggable vertical line.
    """

    left = Child()
    right = Child()

    _bundle = './dist/splitjs_component.bundle.js'
    _esm = """
    import Split

    export function render({ model }) {
      const splitDiv = document.createElement('div');
      splitDiv.className = 'split';

      const split0 = document.createElement('div');
      splitDiv.appendChild(split0);

      const split1 = document.createElement('div');
      splitDiv.appendChild(split1);

      const split = Split([split0, split1])

      model.on('remove', () => split.destroy())

      split0.append(model.get_child("left"))
      split1.append(model.get_child("right"))
      return splitDiv
    }"""

    _stylesheets = [CSS_vertical]


class SplitJSHorizontal(JSComponent):
    """
        Javascript component that hosts two Viewables on top of one another, and separate them with a draggable horizontal line.
    """


    bottom = Child()
    top = Child()

    _bundle = './dist/splitjs_component.bundle.js'
    _esm = """
    import Split

    export function render({ model }) {
      const splitDiv = document.createElement('div');
      splitDiv.className = 'split';

      const split0 = document.createElement('div');
      splitDiv.appendChild(split0);

      const split1 = document.createElement('div');
      splitDiv.appendChild(split1);

      const split = Split([split0, split1], {
            direction: 'vertical',
        })

      model.on('remove', () => split.destroy())

      split0.append(model.get_child("bottom"))
      split1.append(model.get_child("top"))
      return splitDiv
    }"""

    _stylesheets = [CSS_horizontal]

    # def __repr__(self, *args, **kwargs):
    #     return f"Horizontal split item: ({self.bottom}, {self.top})"

if __name__ == "__main__":
        
    import math
    import pandas as pd
    from bokeh.plotting import figure


    df = pd.DataFrame({
        "sinus": [math.sin(t/10) for t in range(50)],
        "cosinus": [math.cos(t/10) for t in range(50)],
        "atan": [math.atan(t/10) for t in range(50)],
    })

    figure_left = figure(
                            sizing_mode = "stretch_both",
                        )
    figure_left.line(x=df["sinus"], y=df["cosinus"])

    figure_right = figure(
                            sizing_mode = "stretch_both",
                        )
    figure_right.line(x=df["atan"], y=df["cosinus"])

    figure_top = figure(
                            sizing_mode = "stretch_both",
                        )
    figure_top.line(x=df["atan"], y=df["sinus"])

    bokeh_plot_left = pn.pane.Bokeh(figure_left, sizing_mode = "stretch_both")
    bokeh_plot_right = pn.pane.Bokeh(figure_right, sizing_mode = "stretch_both")
    bokeh_plot_top = pn.pane.Bokeh(figure_top, sizing_mode = "stretch_both")

    split_react = SplitJSHorizontal(
        bottom=bokeh_plot_top,
        top=SplitJSVertical(
            left=bokeh_plot_left,
            right=bokeh_plot_right,
            sizing_mode="stretch_both",
        ),
        
        sizing_mode="stretch_both",
    )
    split_react.show()

