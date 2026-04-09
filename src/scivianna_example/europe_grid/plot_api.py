import matplotlib.pyplot as plt

from scivianna.constants import X, Y
from scivianna.slave import ComputeSlave
from scivianna.plotter_2d.api import plot_frame_in_axes

from europe_grid import EuropeGridInterface


# Field example
slave = ComputeSlave(EuropeGridInterface)

fig, axes = plt.subplots(1, 1, figsize=(8, 4))
plot_frame_in_axes(
    slave,
    coloring_label="Europe",
    display_colorbar=True,
    axes=axes,
    custom_colors={
        "Europe": {
            "European Union (EU)": "#105DC1",
            "EU candidate countries":  "#4D93E3",
            "European Free Trade Association (EFTA)":  "#704783",
            "Other countries":  "#5F6976",
        }
    },
    rename_values={
        # "Europe":{
            # "European Union (EU)": "EU",
            # "EU candidate countries":  "Candidates",
            # "European Free Trade Association (EFTA)":  "EFTA",
            # "Other countries":  "Other",
        # }
    },
    legend_options={
        "bbox_to_anchor":(1.05, 1), 
        "loc":'upper left'
    }
)

fig.subplots_adjust(right=0.7)


axes.set_title("European countries per category")

fig.tight_layout()
fig.savefig("europe_plot.png", dpi = 300)

slave.terminate()
