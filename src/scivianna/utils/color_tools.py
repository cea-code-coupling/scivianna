from enum import Enum
from typing import Dict, List, Tuple
import numpy as np

color_maps = {
    "magma" : ["#000003", "#140d35", "#3b0f6f", "#63197f", "#8c2980", "#b63679", "#dd4968", "#f6705b", "#fd9f6c", "#fdcf92", "#fbfcbf"],
    "inferno" : ["#000003", "#160b39", "#410967", "#6a176e", "#932567", "#bb3754", "#dc5039", "#f37719", "#fba40a", "#f5d745", "#fcfea4"],
    "plasma" : ["#0c0786", "#40039c", "#6a00a7", "#8f0da3", "#b02a8f", "#cb4777", "#e06461", "#f2844b", "#fca635", "#fcce25", "#eff821"],
    "viridis" : ["#440154", "#482374", "#404387", "#345e8d", "#29788e", "#20908c", "#22a784", "#44be70", "#79d151", "#bdde26", "#fde724"],
    "cividis" : ["#00224d", "#083370", "#35456c", "#4e566c", "#666970", "#7c7b78", "#948e77", "#aea271", "#c8b765", "#e4ce51", "#fde737"],
    "Blues" : ["#f7fbff", "#e3eef8", "#cfe1f2", "#b6d4e9", "#93c4de", "#6aadd5", "#4a97c9", "#2d7dbb", "#1764ab", "#084991", "#08306b"],
    "BuGn" : ["#f7fcfd", "#e8f6f9", "#d6efed", "#b8e4da", "#8ed3c1", "#65c1a3", "#48b27f", "#2e9857", "#157e3a", "#006428", "#00441b"],
    "BuPu" : ["#f7fcfd", "#e4eff5", "#ccddeb", "#b2cae1", "#9ab4d6", "#8c95c5", "#8c73b5", "#8951a4", "#852c8f", "#750b71", "#4d004b"],
    "GnBu" : ["#f7fcf0", "#e4f4df", "#d4eecd", "#bee5be", "#9fd9b8", "#7acbc4", "#57b8d0", "#389ac6", "#1d7db6", "#085fa2", "#084081"],
    "Greens" : ["#f7fcf5", "#e8f6e4", "#d3edcc", "#b8e2b1", "#98d493", "#73c375", "#4bb061", "#2e974e", "#157e3a", "#006428", "#00441b"],
    "Greys" : ["#ffffff", "#f3f3f3", "#e2e2e2", "#cecece", "#b5b5b5", "#959595", "#7a7a7a", "#5e5e5e", "#404040", "#1d1d1d", "#000000"],
    "OrRd" : ["#fff7ec", "#feebcf", "#fddcae", "#fdca94", "#fcb17b", "#fb8c58", "#f16d4b", "#e0442e", "#c81c12", "#a70000", "#7f0000"],
    "Oranges" : ["#fff5eb", "#fee9d4", "#fdd8b3", "#fdc28c", "#fda761", "#fc8c3b", "#f3701b", "#e25407", "#c44001", "#9d3203", "#7f2704"],
    "PuBu" : ["#fff7fb", "#f0eaf3", "#dbd9ea", "#bfc9e1", "#9cb9d8", "#73a8ce", "#4294c3", "#177cb6", "#0467a2", "#035281", "#023858"],
    "PuBuGn" : ["#fff7fb", "#f0e6f2", "#dbd7ea", "#bfc9e1", "#99b9d8", "#66a8ce", "#3f94c3", "#15869e", "#017876", "#016351", "#014636"],
    "PuRd" : ["#f7f4f9", "#eae5f1", "#dbc9e2", "#cfaad2", "#cd8ac2", "#df64af", "#e53491", "#d71a69", "#b80a4e", "#8d003b", "#67001f"],
    "Purples" : ["#fcfbfd", "#f1f0f6", "#e2e1ef", "#cecee5", "#b6b6d8", "#9d99c7", "#8682bc", "#7261ab", "#61409b", "#4f1e8b", "#3f007d"],
    "Reds" : ["#fff5f0", "#fee4d8", "#fcc9b4", "#fcab8e", "#fb8a6a", "#fa6949", "#f14432", "#d82522", "#bb1419", "#970b13", "#67000c"],
    "RdPu" : ["#fff7f3", "#fde4e1", "#fccfcb", "#fbb6bb", "#f994b1", "#f667a0", "#e23e99", "#c01487", "#99017b", "#6f0074", "#49006a"],
    "YlGn" : ["#ffffe5", "#f8fcc2", "#e5f4ab", "#c8e89a", "#a2d889", "#77c578", "#4bb062", "#2e924c", "#15783e", "#006033", "#004529"],
    "YlGnBu" : ["#ffffd9", "#f0f9b9", "#d6efb2", "#abdeb6", "#72c8bc", "#40b5c3", "#2498c0", "#2071b1", "#234da0", "#1e2f87", "#081d58"],
    "YlOrBr" : ["#ffffe5", "#fff8c4", "#feeba2", "#fed777", "#febb47", "#fd9828", "#ef7818", "#d85908", "#b74202", "#8e3004", "#662505"],
    "YlOrRd" : ["#ffffcc", "#fff0a9", "#fee186", "#feca65", "#fdaa48", "#fc8c3b", "#fc5a2d", "#ec2d21", "#d30f20", "#af0026", "#800026"],
    "binary" : ["#ffffff", "#e6e6e6", "#cccccc", "#b3b3b3", "#999999", "#7f7f7f", "#666666", "#4c4c4c", "#323232", "#181818", "#000000"],
    "BrBG" : ["#543005", "#8a5009", "#bf812d", "#dec07b", "#f6e8c3", "#f4f4f4", "#c7eae5", "#7ecbc0", "#35978f", "#00655d", "#003c30"],
    "PRGn" : ["#40004b", "#742981", "#9970ab", "#c1a3ce", "#e7d4e8", "#f6f6f6", "#d9f0d3", "#a4da9e", "#5aae61", "#1a7636", "#00441b"],
    "PiYG" : ["#8e0152", "#c31a7c", "#de77ae", "#f0b4d9", "#fde0ef", "#f6f6f6", "#e6f5d0", "#b6e084", "#7fbc41", "#4c9120", "#276419"],
    "PuOr" : ["#7f3b08", "#b15706", "#e08214", "#fcb661", "#fee0b6", "#f6f6f6", "#d8daeb", "#b1a9d1", "#8073ac", "#532686", "#2d004b"],
    "RdBu" : ["#67001f", "#b0172a", "#d6604d", "#f3a380", "#fddbc7", "#f6f6f6", "#d1e5f0", "#90c4dd", "#4393c3", "#2064aa", "#053061"],
    "BuRd" : ["#053061", "#2064aa", "#4393c3", "#90c4dd", "#d1e5f0", "#f7f6f6", "#fddbc7", "#f3a380", "#d6604d", "#b0172a", "#67001f"],
    "RdYlBu" : ["#a50026", "#d62f26", "#f46d43", "#fcac60", "#fee090", "#fefec0", "#e0f3f7", "#a9d8e8", "#74add1", "#4473b3", "#313695"],
    "RdYlGn" : ["#a50026", "#d62f26", "#f46d43", "#fcac60", "#fee08b", "#fefebd", "#d9ef8b", "#a4d869", "#66bd63", "#19974f", "#006837"],
    "Spectral" : ["#9e0142", "#d33c4e", "#f46d43", "#fcac60", "#fee08b", "#fefebe", "#e6f598", "#a9dca4", "#66c2a5", "#3286bc", "#5e4fa2"],
    "copper" : ["#000000", "#1e130c", "#3e2719", "#5d3b25", "#7d4f32", "#9e633f", "#bc774c", "#dd8b59", "#fb9f65", "#ffb372", "#ffc77e"],
    "gray" : ["#000000", "#191919", "#333333", "#4c4c4c", "#666666", "#808080", "#999999", "#b3b3b3", "#cccccc", "#e6e6e6", "#ffffff"],
    "afmhot" : ["#000000", "#320000", "#660000", "#981800", "#cc4c00", "#ff8000", "#ffb232", "#ffe666", "#ffff99", "#ffffcd", "#ffffff"],
    "hot" : ["#0a0000", "#4c0000", "#900000", "#d20000", "#ff1700", "#ff5b00", "#ff9d00", "#ffe100", "#ffff36", "#ffff9c", "#ffffff"],
    "Wistia" : ["#e4ff7a", "#eef554", "#f9ec2d", "#ffdf15", "#ffce0a", "#ffbc00", "#ffb100", "#ffa500", "#fe9900", "#fd8b00", "#fc7f00"],
    "summer" : ["#007f66", "#198c66", "#339966", "#4ca566", "#66b266", "#80bf66", "#99cc66", "#b3d966", "#cce566", "#e6f266", "#ffff66"],
    "winter" : ["#0000ff", "#0019f2", "#0033e5", "#004cd9", "#0066cc", "#0080bf", "#0099b2", "#00b3a5", "#00cc99", "#00e68c", "#00ff7f"],
    "autumn" : ["#ff0000", "#ff1900", "#ff3300", "#ff4c00", "#ff6600", "#ff8000", "#ff9900", "#ffb300", "#ffcc00", "#ffe600", "#ffff00"],
    "hsv" : ["#ff0000", "#ff9300", "#d0ff00", "#3dff00", "#00ff5c", "#00fff5", "#0074ff", "#2500ff", "#b800ff", "#ff00ab", "#ff0017"],
    "jet" : ["#00007f", "#0000f1", "#004cff", "#00b0ff", "#29ffcd", "#7cff79", "#cdff29", "#ffc400", "#ff6700", "#f10700", "#7f0000"],
    "terrain" : ["#333399", "#1175db", "#00b2b2", "#31d56f", "#99ea84", "#fefd98", "#ccbd7d", "#987b61", "#997c76", "#cdbfbb", "#ffffff"],
}

def get_edges_colors(face_colors:np.ndarray, offset: int = -20) -> np.ndarray:
    """Returnds the edge colors from the face colors

    Parameters
    ----------
    face_colors : np.ndarray
        Numpy array containing a list of RBG colors ranging from 0 to 255
    offset : int
        The colors are defined in RGB with colors from 0 to 255, this offset tells how many decrease the edge color

    Returns
    -------
    np.ndarray
        Face colors
    """
    edge_colors:np.ndarray = face_colors.copy()

    # Allowing no data being displayed
    if len(edge_colors.shape) == 1:
        return edge_colors
    
    # Darkening the colors
    edge_colors[:, :3] += offset

    edge_colors = np.where(edge_colors<0, 0, edge_colors)

    return np.where(edge_colors>255, 255, edge_colors)

def interpolate_cmap_at_values(
    cmap_name: str, values: np.ndarray
) -> np.ndarray:
    """Returns a numpy array containing the RGBA 255 colors per value in values

    Parameters
    ----------
    cmap_name : str
        Name of the cmaps to get from scivianna.utils.color_tools.color_maps
    values : np.ndarray
        Values to interpolate

    Returns
    -------
    np.ndarray
        RGBA 255 colors per value in values
    """ 

    cmap = color_maps[cmap_name]
    colors = np.array([(int(c[1:3], 16) , int(c[3:5], 16) , int(c[5:7], 16)) for c in cmap]).astype(float)/255

    r = colors[:, 0]
    g = colors[:, 1]
    b = colors[:, 2]

    xp_colors = np.arange(len(colors)) / (len(colors) - 1)

    r_vals = np.interp(values, xp_colors, r)
    g_vals = np.interp(values, xp_colors, g)
    b_vals = np.interp(values, xp_colors, b)

    float_array = np.stack([r_vals, g_vals, b_vals, np.ones((len(r_vals)))]).T * 255

    # Create a replacement array
    replacement = np.array([0, 0, 0, 0])
    float_array[np.isnan(float_array).any(axis=1)] = replacement

    y_vals = float_array.astype(np.int16)

    return y_vals

beautiful_color_maps = {
    c:[list(e)[:3] for e in interpolate_cmap_at_values(c, np.arange(0, 1, 0.01))] for c in color_maps
}