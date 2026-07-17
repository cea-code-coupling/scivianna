from pathlib import Path

import pytest

import scivianna
from scivianna.constants import GEOMETRY, X, Y
from scivianna.interface.med_interface import MEDInterface
from scivianna.slave import ComputeSlave
from scivianna.utils.file_cleaner import mark_for_deletion
import numpy as np


@pytest.mark.default
def test_save_load_med_with_include_files():
    """Simple test to make sure things happen before more tests are actually implemented
    """
    #   First creation of a slave
    slave = ComputeSlave(MEDInterface)
    slave.read_file(
        Path(scivianna.__file__).parent / "input_file" / "power.med",
        GEOMETRY,
    )

    u, v = X, Y
    w = np.cross(u, v)
    origin = np.array(u) * 0.5 + np.array(v) * 0.5 + 0. * w
    size_u = 1. - 0.
    size_v = 1. - 0.
    data, computed = slave.compute_2D_data(u, v, tuple(origin), size_u, size_v, None, "INTEGRATED_POWER", {}, caller="Test")
    assert computed, "First compute_2d_data should have been computed"
    dict1 = slave.get_value_dict("INTEGRATED_POWER", data.cell_ids, {}, caller="Test")

    #   Saving the current save state (loaded file, computed polygons, loaded files...)
    slave.save("med_test.pkl", True)
    mark_for_deletion("med_test.pkl")

    #   Creating a new slave and loading the file
    slave2 = ComputeSlave(MEDInterface)
    slave2.load("med_test.pkl", True)

    #   New compute_2D_data is now instant as the polygons were saved
    data2, computed = slave.compute_2D_data(u, v, tuple(origin), size_u, size_v, None, "INTEGRATED_POWER", {}, caller="Test")
    assert not computed, "Loaded compute_2d_data should have been skipped"
    dict2 = slave2.get_value_dict("INTEGRATED_POWER", data2.cell_ids, {}, caller="Test")

    assert dict1 == dict2, "Returned cell value dictionnary doesn't match the first"

    slave.terminate()
    slave2.terminate()


@pytest.mark.skip("Behavior not implemented yet")
def test_save_load_med_without_include_files():
    """Simple test to make sure things happen before more tests are actually implemented
    """
    #   First creation of a slave
    slave = ComputeSlave(MEDInterface)
    slave.read_file(
        Path(scivianna.__file__).parent / "input_file" / "power.med",
        GEOMETRY,
    )

    u, v = X, Y
    w = np.cross(u, v)
    origin = np.array(u) * 0.5 + np.array(v) * 0.5 + 0. * w
    size_u = 1. - 0.
    size_v = 1. - 0.
    data, computed = slave.compute_2D_data(u, v, tuple(origin), size_u, size_v, None, "INTEGRATED_POWER", {}, caller="Test")
    assert computed, "First compute_2d_data should have been computed"
    dict1 = slave.get_value_dict("INTEGRATED_POWER", data.cell_ids, {}, caller="Test")

    #   Saving the current save state (loaded file, computed polygons, loaded files...)
    slave.save("med_test.pkl", False)
    mark_for_deletion("med_test.pkl")

    #   Creating a new slave and loading the file
    slave2 = ComputeSlave(MEDInterface)
    slave2.read_file(
        Path(scivianna.__file__).parent / "input_file" / "power.med",
        GEOMETRY,
    )
    slave2.load("med_test.pkl", False)

    #   New compute_2D_data is now instant as the polygons were saved
    data2, computed = slave.compute_2D_data(u, v, tuple(origin), size_u, size_v, None, "INTEGRATED_POWER", {}, caller="Test")
    assert not computed, "Loaded compute_2d_data should have been skipped"
    dict2 = slave2.get_value_dict("INTEGRATED_POWER", data2.cell_ids, {}, caller="Test")

    assert dict1 == dict2, "Returned cell value dictionnary doesn't match the first"

    slave.terminate()
    slave2.terminate()