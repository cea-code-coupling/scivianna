from scivianna.extension.field_selector import FieldSelector
from scivianna.constants import MESH, MATERIAL
import scivianna.utils

from test_interface import make_panel_2d
scivianna.utils._testing = True

def change_field():
    panel, extensions = make_panel_2d()

    slave = panel.slave

    try:
        color_extension: FieldSelector = extensions[FieldSelector]

        data, frame, fiel_path, field_name = slave.call_custom_function("get_info", {})
        assert field_name == MESH
        
        panel.set_field(MATERIAL)
        data, frame, fiel_path, field_name = slave.call_custom_function("get_info", {})
        assert (field_name == MATERIAL), f"Expecting Material, found {field_name}"
            
        panel.set_field("VALUE")
        data, frame, fiel_path, field_name = slave.call_custom_function("get_info", {})
        assert (field_name == "VALUE"), f"Expecting VALUE, found {field_name}"
    except Exception as e:
        raise e
    finally:
        slave.terminate()

if __name__ == "__main__":
    change_field()