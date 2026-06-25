import panel as pn
from panel.custom import Child, JSComponent, ReactComponent

from panel_splitjs import HSplit as SplitJSVertical, VSplit as SplitJSHorizontal


if __name__ == "__main__":
    import panel as pn
    from panel_splitjs import HSplit

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
        bokeh_plot_top,
        SplitJSVertical(
            bokeh_plot_left,
            bokeh_plot_right,
            sizing_mode="stretch_both",
            sizes=(30,70)
        ),

        sizing_mode="stretch_both",
    )
    split_react.show()

