
import pytest
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from scivianna.agent.data_2d_worker import Data2DWorker

from scivianna.data.data2d import Data2D
from scivianna.utils.polygonize_tools import PolygonCoords, PolygonElement


@pytest.fixture
def data2d():
    data2d = Data2D.from_polygon_list(
        [
            PolygonElement(
                exterior_polygon=PolygonCoords(
                    x_coords=[0, 0, 1, 1],
                    y_coords=[i, 1+i, 1+i, i]
                ),
                holes=[],
                cell_id=i
            )
            for i in range(10)
        ]
    )

    data2d.cell_values = np.cos(np.arange(10))
    data2d.cell_colors = (np.ones((10, 4))*255).astype(int)
    data2d.cell_edge_colors = (np.ones((10, 4))*235).astype(int)

    return data2d

@pytest.fixture
def worker(data2d, request):
    """Provide a Data2DWorker instance initialized with a copy of data2d."""
    if request.node.get_closest_marker("default"):
        from scivianna.data.data_2d_worker import Data2DWorker
    elif request.node.get_closest_marker("agent"):
        import os
        os.environ["LLM_MODEL_ID"] = "PYTEST"
        os.environ["LLM_API_BASE"] = "PYTEST"
        os.environ["LLM_API_KEY"] = "PYTEST"
        from scivianna.agent.data_2d_worker import Data2DWorker
    else:
        import os
        os.environ["LLM_MODEL_ID"] = "PYTEST"
        os.environ["LLM_API_BASE"] = "PYTEST"
        os.environ["LLM_API_KEY"] = "PYTEST"
        from scivianna.agent.data_2d_worker import Data2DWorker

    worker = Data2DWorker(data2d)
    return worker

@pytest.mark.default
def test_has_changed_no_change(worker: "Data2DWorker"):
    """Test that has_changed returns False when no change occurred."""
    assert not worker.has_changed()

@pytest.mark.default
def test_has_changed_after_set_colors(worker: "Data2DWorker"):
    """Test that has_changed returns True after modifying cell_colors."""
    original_colors = np.array(worker.data2d.cell_colors)
    new_colors = np.array(original_colors) - 10
    worker.set_colors(new_colors)
    assert worker.has_changed()

@pytest.mark.default
def test_has_changed_after_set_alphas(worker: "Data2DWorker"):
    """Test that has_changed returns True after modifying alphas."""
    original_alphas = np.array([c[-1] for c in worker.data2d.cell_colors])
    new_alphas = original_alphas - 10
    worker.set_alphas(new_alphas)
    assert worker.has_changed()

@pytest.mark.default
def test_has_changed_after_set_values(worker: "Data2DWorker"):
    """Test that has_changed returns True after changing cell_values."""
    original_values = np.array(worker.data2d.cell_values)
    new_values = original_values + 1.0
    worker.data2d.cell_values = new_values
    assert worker.has_changed()

@pytest.mark.default
def test_has_changed_no_change_after_reset(worker: "Data2DWorker"):
    """Test that has_changed returns False after reset."""
    worker.set_colors(np.array(worker.data2d.cell_colors) - 10)
    worker.reset()
    assert not worker.has_changed()

@pytest.mark.default
def test_check_valid(worker: "Data2DWorker"):
    """Test that check_valid passes when valid."""
    worker.check_valid()  # Should not raise

@pytest.mark.default
def test_check_valid_invalid_data2d(worker: "Data2DWorker"):
    """Test that check_valid raises AssertionError when invalid."""
    worker.data2d.cell_values = [1.]
    with pytest.raises(AssertionError):
        worker.check_valid()

@pytest.mark.default
def test_get_values(worker: "Data2DWorker"):
    """Test that get_values returns a numpy array."""
    values = worker.get_values()
    assert isinstance(values, np.ndarray)
    assert values.shape == (10,)
    assert np.allclose(values, np.cos(np.arange(10)))

@pytest.mark.default
def test_get_colors(worker: "Data2DWorker"):
    """Test that get_colors returns correct shape and values."""
    colors = worker.get_colors()
    assert isinstance(colors, np.ndarray)
    assert colors.shape == (10, 4)
    assert np.allclose(colors, 255)

@pytest.mark.default
def test_set_colors_valid(worker: "Data2DWorker"):
    """Test setting valid colors."""
    new_colors = np.ones((10, 4)) * 128
    assert worker.set_colors(new_colors)
    assert np.allclose(worker.data2d.cell_colors, new_colors)

@pytest.mark.default
def test_set_colors_invalid_type(worker: "Data2DWorker"):
    """Test that set_colors raises with invalid type."""
    with pytest.raises(AssertionError, match="A numpy array is expected"):
        worker.set_colors("not an array")

@pytest.mark.default
def test_set_colors_invalid_shape(worker: "Data2DWorker"):
    """Test that set_colors raises with wrong shape."""
    with pytest.raises(AssertionError, match="A 2D numpy array is expected"):
        worker.set_colors(np.array([1, 2, 3]))

@pytest.mark.default
def test_set_colors_wrong_size(worker: "Data2DWorker"):
    """Test that set_colors raises when shape doesn't match."""
    with pytest.raises(AssertionError, match="We expect the same number of elements"):
        worker.set_colors(np.ones((5, 4)))

@pytest.mark.default
def test_set_colors_out_of_bounds(worker: "Data2DWorker"):
    """Test that set_colors raises when values > 255."""
    with pytest.raises(AssertionError, match="The values must be lower than 255"):
        worker.set_colors(np.ones((10, 4)) * 256)

@pytest.mark.default
def test_set_colors_negative_values(worker: "Data2DWorker"):
    """Test that set_colors raises when values < 0."""
    with pytest.raises(AssertionError, match="The values must be greater than 0"):
        worker.set_colors(np.ones((10, 4)) * -1)

@pytest.mark.default
def test_set_alphas_valid(worker: "Data2DWorker"):
    """Test setting valid alphas."""
    alphas = np.arange(10) * 25
    assert worker.set_alphas(alphas)
    updated_colors = np.array(worker.data2d.cell_colors)
    assert np.allclose(updated_colors[:, -1], alphas)

@pytest.mark.default
def test_set_alphas_invalid_type(worker: "Data2DWorker"):
    """Test that set_alphas raises with invalid type."""
    with pytest.raises(AssertionError, match="A numpy array is expected"):
        worker.set_alphas("not an array")

@pytest.mark.default
def test_set_alphas_invalid_shape(worker: "Data2DWorker"):
    """Test that set_alphas raises with wrong shape."""
    with pytest.raises(AssertionError, match="A 1D numpy array is expected"):
        worker.set_alphas(np.ones((10, 4)))

@pytest.mark.default
def test_set_alphas_wrong_size(worker: "Data2DWorker"):
    """Test that set_alphas raises when size doesn't match."""
    with pytest.raises(AssertionError, match="We expect the same number of elements"):
        worker.set_alphas(np.arange(5))

@pytest.mark.default
def test_set_alphas_out_of_bounds(worker: "Data2DWorker"):
    """Test that set_alphas raises when values > 255."""
    with pytest.raises(AssertionError, match="The values must be lower than 255"):
        worker.set_alphas(np.array([256] + [0]*9))

@pytest.mark.default
def test_set_alphas_negative_values(worker: "Data2DWorker"):
    """Test that set_alphas raises when values < 0."""
    with pytest.raises(AssertionError, match="The values must be greater than 0"):
        worker.set_alphas(np.array([-1] + [0]*9))

@pytest.mark.default
def test_reset(worker: "Data2DWorker"):
    """Test that reset restores original data."""
    # Modify data
    worker.data2d.cell_values = [0] * 10
    worker.data2d.cell_colors = [[0, 0, 0, 0]] * 10
    assert worker.has_changed()
    worker.reset()
    assert not worker.has_changed()
    assert np.allclose(worker.data2d.cell_values, np.cos(np.arange(10)))
    assert np.allclose(worker.data2d.cell_colors, 255)

@pytest.mark.default
def test_get_numpy(worker: "Data2DWorker"):
    """Test that get_numpy returns numpy."""
    assert worker.get_numpy() is np

@pytest.mark.agent
def test_execute_code_valid(worker: "Data2DWorker"):
    """Test that execute_code runs valid code and returns success."""
    code = """
np = get_numpy()
values = get_values()
new_colors = np.zeros((10, 4))
set_colors(new_colors)
    """
    success, msg = worker.execute_code(code)
    assert success
    assert "Success" in msg
    assert worker.has_changed()
    # Verify colors were updated
    assert np.allclose(np.array(worker.data2d.cell_colors), 0)

@pytest.mark.agent
def test_execute_code_invalid_code(worker: "Data2DWorker"):
    """Test that execute_code fails when code produces invalid Data2D."""
    code = """
cell_values = None
    """
    success, msg = worker.execute_code(code)
    assert not success

@pytest.mark.agent
def test_execute_code_no_change(worker: "Data2DWorker"):
    """Test that execute_code returns no change if data unchanged."""
    code = "get_values()"
    success, msg = worker.execute_code(code)
    assert not success
    assert "did not change the Data2D" in msg

@pytest.mark.agent
def test_execute_code_with_import(worker: "Data2DWorker"):
    """Test that import numpy is handled correctly."""
    code = """
import numpy as np
values = get_values()
new_colors = np.ones((10, 4)) * 100
set_colors(new_colors)
    """
    success, msg = worker.execute_code(code)
    assert success
    assert np.allclose(np.array(worker.data2d.cell_colors), 100)
