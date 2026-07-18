import pytest
from scivianna_example.med.single_med import get_panel

@pytest.mark.default
def test_press_z():
    layout, slaves = get_panel(None, return_slaves=True)
    panel = layout.visualisation_panels["MEDCoupling visualizer XY"]
    panel.figure.key = "x"
    panel.figure.key = "z"

    assert panel.u == pytest.approx([1, 0, 0]), f"Expected u to be approximately [1, 0, 0], but got {panel.u}"
    assert panel.v == pytest.approx([0, 1, 0]), f"Expected v to be approximately [0, 1, 0], but got {panel.v}"

    panel.figure.key = "z"

    assert panel.u == pytest.approx([-1, 0, 0]), f"Expected u to be approximately [-1, 0, 0], but got {panel.u}"
    assert panel.v == pytest.approx([0, 1, 0]), f"Expected v to be approximately [0, 1, 0], but got {panel.v}"

    for slave in slaves:
        slave.terminate()

def test_press_x():
    layout, slaves = get_panel(None, return_slaves=True)
    panel = layout.visualisation_panels["MEDCoupling visualizer XY"]
    panel.figure.key = "x"
    
    assert panel.u == pytest.approx([0, 1, 0]), f"Expected u to be approximately [0, 1, 0], but got {panel.u}"
    assert panel.v == pytest.approx([0, 0, 1]), f"Expected v to be approximately [0, 0, 1], but got {panel.v}"
    
    panel.figure.key = "x"

    assert panel.u == pytest.approx([0, -1, 0]), f"Expected u to be approximately [0, -1, 0], but got {panel.u}"
    assert panel.v == pytest.approx([0, 0, 1]), f"Expected v to be approximately [0, 0, 1], but got {panel.v}"

    for slave in slaves:
        slave.terminate()

def test_press_y():
    layout, slaves = get_panel(None, return_slaves=True)
    panel = layout.visualisation_panels["MEDCoupling visualizer XY"]
    panel.figure.key = "y"
    
    assert panel.u == pytest.approx([1, 0, 0]), f"Expected u to be approximately [1, 0, 0], but got {panel.u}"
    assert panel.v == pytest.approx([0, 0, 1]), f"Expected v to be approximately [0, 0, 1], but got {panel.v}"
    
    panel.figure.key = "y"

    assert panel.u == pytest.approx([-1, 0, 0]), f"Expected u to be approximately [-1, 0, 0], but got {panel.u}"
    assert panel.v == pytest.approx([0, 0, 1]), f"Expected v to be approximately [0, 0, 1], but got {panel.v}"

    for slave in slaves:
        slave.terminate()

def test_press_f():
    layout, slaves = get_panel(None, return_slaves=True)
    panel = layout.visualisation_panels["MEDCoupling visualizer XY"]
    panel.figure.key = "f"
    
    assert panel.u == pytest.approx([-1, 0, 0]), f"Expected u to be approximately [-1, 0, 0], but got {panel.u}"
    assert panel.v == pytest.approx([0, 1, 0]), f"Expected v to be approximately [0, 1, 0], but got {panel.v}"
    
    panel.figure.key = "f"

    assert panel.u == pytest.approx([1, 0, 0]), f"Expected u to be approximately [-1, 0, 0], but got {panel.u}"
    assert panel.v == pytest.approx([0, 1, 0]), f"Expected v to be approximately [0, -1, 0], but got {panel.v}"

    for slave in slaves:
        slave.terminate()
