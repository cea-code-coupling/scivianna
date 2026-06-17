from scivianna.layout.split import SplitLayout
from scivianna.utils.file_cleaner import mark_for_deletion

from scivianna_example.med.split_item_example import get_panel


def get_restored_split():
    panel, slaves = get_panel(None, True)

    panel.save_to_zip("test_split.zip")

    for slave in slaves:
        slave.terminate()

    new_layout = SplitLayout.restore_from_zip("test_split.zip")
    mark_for_deletion("test_split.zip")
    return new_layout
   

if __name__ == "__main__":
    new_layout = get_restored_split()
    new_layout.show()
