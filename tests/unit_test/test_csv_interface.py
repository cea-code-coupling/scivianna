"""Tests for scivianna.interface.csv_result.CSVInterface."""

import os
import shutil
import tempfile
import uuid
import numpy as np
import pytest
import pandas as pd
from scivianna.interface.csv_result import CSVInterface


def create_csv_file(filename: str, data: dict) -> str:
    """Helper to create a temporary CSV file with a specific filename.
    
    Uses a unique subdirectory to avoid race conditions when tests run in parallel
    (e.g., with pytest-xdist -n 8 in CI).
    
    Parameters
    ----------
    filename : str
        The desired filename (without path)
    data : dict
        Dictionary of column data for the DataFrame
        
    Returns
    -------
    str
        Path to the temporary file
    """
    df = pd.DataFrame(data)
    # Create a unique directory to avoid collisions in parallel test execution
    unique_id = uuid.uuid4().hex[:8]
    unique_dir = tempfile.mkdtemp(prefix=f"csv_test_{unique_id}_")
    temp_path = os.path.join(unique_dir, filename)
    df.to_csv(temp_path, index=False)
    return temp_path


# ---- Fixtures ----

@pytest.fixture
def csv_cell_file():
    """Create a CSV file with cell-indexed data.
    
    File: data.csv with columns 'cell', 'temp', 'pressure'
    get_fields() returns ['temp', 'pressure'] (prefixed)
    get_value()/get_values() use raw column names ('temp', 'pressure')
    """
    path = create_csv_file("data.csv", {
        "cell": ["cell_1", "cell_2", "cell_3"],
        "temp": [10.0, 20.0, 30.0],
        "pressure": [1.0, 2.0, 3.0],
    })
    yield path
    # Clean up the file and its parent directory
    parent_dir = os.path.dirname(path)
    os.unlink(path)
    if os.path.exists(parent_dir):
        os.rmdir(parent_dir)


@pytest.fixture
def csv_cell_file_numeric_cells():
    """Create a CSV file with numeric cell indices."""
    path = create_csv_file("data_numeric_cells.csv", {
        "cell": [1, 2, 3],
        "temp": [10.0, 20.0, 30.0],
        "pressure": [1.0, 2.0, 3.0],
    })
    yield path
    # Clean up the file and its parent directory
    parent_dir = os.path.dirname(path)
    os.unlink(path)
    if os.path.exists(parent_dir):
        os.rmdir(parent_dir)


@pytest.fixture
def csv_custom_basename_file():
    """Create a CSV file with a custom basename."""
    path = create_csv_file("output.csv", {
        "cell": ["c1"],
        "field": [42.0],
    })
    yield path
    # Clean up the file and its parent directory
    parent_dir = os.path.dirname(path)
    os.unlink(path)
    if os.path.exists(parent_dir):
        os.rmdir(parent_dir)


# ---- __init__ Tests ----

@pytest.mark.default
class TestCSVInterfaceInit:
    def test_init_success_cell_file(self, csv_cell_file):
        """Test initialization with a cell-indexed CSV file."""
        interface = CSVInterface(csv_cell_file)
        assert interface.basename == "data"
        assert isinstance(interface.df, pd.DataFrame)
        assert len(interface.df) == 3

    def test_init_nonexistent_file(self):
        """Test initialization raises ValueError for non-existent file."""
        with pytest.raises(ValueError, match="Provided path does not exist"):
            CSVInterface("/nonexistent/path/file.csv")

    def test_init_no_cell_column_raises(self):
        """Test initialization raises ValueError when cell column doesn't exist."""
        data = {
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "value": [10.0, 20.0],
        }
        path = create_csv_file("test.csv", data)
        try:
            with pytest.raises(ValueError, match="Cell column was not found"):
                CSVInterface(path)
        finally:
            os.unlink(path)

    def test_init_with_cell_column(self):
        """Test initialization with cell column present."""
        data = {
            "cell": ["c1", "c2"],
            "temp": [10.0, 20.0],
        }
        path = create_csv_file("test.csv", data)
        try:
            interface = CSVInterface(path)
            assert interface.basename == "test"
        finally:
            os.unlink(path)

    def test_basename_without_extension(self, csv_custom_basename_file):
        """Test that basename uses the CSV filename without extension."""
        interface = CSVInterface(csv_custom_basename_file)
        assert interface.basename == "output"


# ---- get_value Tests ----
# Note: get_value() uses raw DataFrame column names (e.g., 'temp', 'pressure')
# material_name parameter is accepted but ignored

@pytest.mark.default
class TestCSVInterfaceGetValue:
    def test_get_value_cell_index(self, csv_cell_file):
        """Test get_value with cell_index using raw column name."""
        interface = CSVInterface(csv_cell_file)
        result = interface.get_value(
            position=(0.0, 0.0, 0.0),
            cell_index="cell_1",
            material_name="",
            field="temp",  # Raw column name, not "temp"
        )
        assert result == 10.0

    def test_get_value_cell_index_other(self, csv_cell_file):
        """Test get_value with different cell_index."""
        interface = CSVInterface(csv_cell_file)
        result = interface.get_value(
            position=(0.0, 0.0, 0.0),
            cell_index="cell_2",
            material_name="",
            field="temp",
        )
        assert result == 20.0

    def test_get_value_pressure_field(self, csv_cell_file):
        """Test get_value with a different field."""
        interface = CSVInterface(csv_cell_file)
        result = interface.get_value(
            position=(0.0, 0.0, 0.0),
            cell_index="cell_3",
            material_name="",
            field="pressure",
        )
        assert result == 3.0

    def test_get_value_unknown_field_raises(self, csv_cell_file):
        """Test get_value raises ValueError for unknown field."""
        interface = CSVInterface(csv_cell_file)
        with pytest.raises(ValueError, match="Field unknown_field not found"):
            interface.get_value(
                position=(0.0, 0.0, 0.0),
                cell_index="cell_1",
                material_name="",
                field="unknown_field",
            )

    def test_get_value_numeric_cell(self, csv_cell_file_numeric_cells):
        """Test get_value with numeric cell index."""
        interface = CSVInterface(csv_cell_file_numeric_cells)
        result = interface.get_value(
            position=(0.0, 0.0, 0.0),
            cell_index=2,
            material_name="",
            field="temp",
        )
        assert result == 20.0

    def test_get_value_material_name_ignored(self, csv_cell_file):
        """Test that material_name parameter is accepted but ignored."""
        interface = CSVInterface(csv_cell_file)
        result = interface.get_value(
            position=(0.0, 0.0, 0.0),
            cell_index="cell_1",
            material_name="any_material",
            field="temp",
        )
        assert result == 10.0


# ---- get_values Tests ----
# Note: get_values() uses raw DataFrame column names
# material_names parameter is accepted but ignored

@pytest.mark.default
class TestCSVInterfaceGetValues:
    def test_get_values_cell_indexes(self, csv_cell_file):
        """Test get_values with multiple cell indexes."""
        interface = CSVInterface(csv_cell_file)
        results = interface.get_values(
            positions=[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
            cell_indexes=["cell_1", "cell_3"],
            material_names=[],
            field="temp",
        )
        assert results == [10.0, 30.0]

    def test_get_values_single_entry(self, csv_cell_file):
        """Test get_values with a single entry."""
        interface = CSVInterface(csv_cell_file)
        results = interface.get_values(
            positions=[(0.0, 0.0, 0.0)],
            cell_indexes=["cell_2"],
            material_names=[],
            field="pressure",
        )
        assert results == [2.0]

    def test_get_values_unknown_field_raises(self, csv_cell_file):
        """Test get_values raises ValueError for unknown field."""
        interface = CSVInterface(csv_cell_file)
        with pytest.raises(ValueError, match="Field unknown not found"):
            interface.get_values(
                positions=[],
                cell_indexes=["cell_1"],
                material_names=[],
                field="unknown",
            )

    def test_get_values_with_inf_cell(self, csv_cell_file_numeric_cells):
        """Test get_values handles np.inf cell index by replacing with NaN."""
        interface = CSVInterface(csv_cell_file_numeric_cells)
        results = interface.get_values(
            positions=[],
            cell_indexes=[1, np.inf],
            material_names=[],
            field="temp",
        )
        assert results[0] == 10.0
        assert np.isnan(results[1])

    def test_get_values_multiple_fields(self, csv_cell_file):
        """Test get_values with different fields."""
        interface = CSVInterface(csv_cell_file)
        results = interface.get_values(
            positions=[],
            cell_indexes=["cell_1", "cell_2"],
            material_names=[],
            field="pressure",
        )
        assert results == [1.0, 2.0]

    def test_get_values_numeric_cell_indexes(self, csv_cell_file_numeric_cells):
        """Test get_values with numeric cell indexes."""
        interface = CSVInterface(csv_cell_file_numeric_cells)
        results = interface.get_values(
            positions=[],
            cell_indexes=[1, 3],
            material_names=[],
            field="temp",
        )
        assert results == [10.0, 30.0]

    def test_get_values_missing_cell_returns_nan(self, csv_cell_file):
        """Test get_values returns NaN for non-existent cell index."""
        interface = CSVInterface(csv_cell_file)
        results = interface.get_values(
            positions=[],
            cell_indexes=["nonexistent"],
            material_names=[],
            field="temp",
        )
        assert np.isnan(results[0])

    def test_get_values_material_names_ignored(self, csv_cell_file):
        """Test that material_names parameter is accepted but ignored."""
        interface = CSVInterface(csv_cell_file)
        results = interface.get_values(
            positions=[],
            cell_indexes=["cell_1", "cell_2"],
            material_names=["any_material", "another"],
            field="temp",
        )
        assert results == [10.0, 20.0]


# ---- get_fields Tests ----
# Note: get_fields() returns basename-prefixed column names

@pytest.mark.default
class TestCSVInterfaceGetFields:
    def test_get_fields_cell_file(self, csv_cell_file):
        """Test get_fields returns prefixed field names for cell file."""
        interface = CSVInterface(csv_cell_file)
        fields = interface.get_fields()
        assert "temp" in fields
        assert "pressure" in fields
        assert "cell" not in fields

    def test_get_fields_excludes_special_columns(self, csv_cell_file):
        """Test get_fields excludes 'cell' column."""
        interface = CSVInterface(csv_cell_file)
        fields = interface.get_fields()
        for f in fields:
            assert f != "cell"

    def test_get_fields_returns_list(self, csv_cell_file):
        """Test get_fields returns a list."""
        interface = CSVInterface(csv_cell_file)
        fields = interface.get_fields()
        assert isinstance(fields, list)

    def test_get_fields_count(self, csv_cell_file):
        """Test get_fields returns correct number of fields."""
        interface = CSVInterface(csv_cell_file)
        fields = interface.get_fields()
        # 2 data columns: temp, pressure
        assert len(fields) == 2

    def test_get_fields_custom_basename(self, csv_custom_basename_file):
        """Test get_fields with custom basename."""
        interface = CSVInterface(csv_custom_basename_file)
        fields = interface.get_fields()
        # basename is "output", column is "field" -> "output_field"
        assert "field" in fields


# ---- Integration-like Tests ----

@pytest.mark.default
class TestCSVInterfaceIntegration:
    def test_full_workflow(self, csv_cell_file):
        """Test a full workflow with cell-indexed CSV."""
        interface = CSVInterface(csv_cell_file)
        
        # Get available fields (returns prefixed names)
        fields = interface.get_fields()
        assert len(fields) == 2
        assert "temp" in fields
        
        # Get single value (uses raw column name)
        val = interface.get_value(
            position=(0.5, 0.5, 0.5),
            cell_index="cell_2",
            material_name="",
            field="temp",  # Raw column name
        )
        assert val == 20.0
        
        # Get multiple values (uses raw column name)
        vals = interface.get_values(
            positions=[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
            cell_indexes=["cell_1", "cell_3"],
            material_names=[],
            field="temp",
        )
        assert vals == [10.0, 30.0]

    def test_inf_handling_integration(self, csv_cell_file_numeric_cells):
        """Test np.inf handling in get_values."""
        interface = CSVInterface(csv_cell_file_numeric_cells)
        
        results = interface.get_values(
            positions=[],
            cell_indexes=[np.inf],
            material_names=[],
            field="temp",
        )
        assert len(results) == 1
        assert np.isnan(results[0])