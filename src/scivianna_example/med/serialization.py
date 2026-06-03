from scivianna.layout.split import SplitLayout

from scivianna_example.med.split_item_example import get_panel, get_med_panel


panel, slaves = get_panel(None, True)

panel.save_to_zip("test.zip")

for slave in slaves:
    slave.terminate()

new_layout = SplitLayout.restore_from_zip("test.zip")
new_layout.show()