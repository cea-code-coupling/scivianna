from scivianna.layout.gridstack import GridStackLayout

from scivianna_example.med.grid_stack_example import get_panel


panel, slaves = get_panel(None, True)

panel.save_to_zip("test_gridstack.zip")

for slave in slaves:
    slave.terminate()

new_layout = GridStackLayout.restore_from_zip("test_gridstack.zip")
new_layout.show()