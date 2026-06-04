from scivianna.layout.gridstack import GridStackLayout
from scivianna.utils.file_cleaner import mark_for_deletion

from scivianna_example.med.grid_stack_example import get_panel

def get_restored_gridstack():
    panel, slaves = get_panel(None, True)

    panel.save_to_zip("test_gridstack.zip")

    for slave in slaves:
        slave.terminate()

    new_layout = GridStackLayout.restore_from_zip("test_gridstack.zip")
    mark_for_deletion("test_gridstack.zip")
    return new_layout

if __name__ == "__main__":
    new_layout = get_restored_gridstack()
    new_layout.show()