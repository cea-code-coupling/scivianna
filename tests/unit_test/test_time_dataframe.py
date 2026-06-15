"""Tests for scivianna.interface.time_dataframe module."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch


@pytest.mark.default
class TestTimeDataFrame:
    """Test suite for TimeDataFrame class."""

    @pytest.fixture
    def time_dataframe(self):
        """Create a TimeDataFrame instance for testing."""
        from scivianna.interface.time_dataframe import TimeDataFrame
        return TimeDataFrame()

    @pytest.fixture
    def time_dataframe_with_data(self):
        """Create a TimeDataFrame instance with pre-populated data."""
        from scivianna.interface.time_dataframe import TimeDataFrame
        td = TimeDataFrame()
        
        # Create a dataframe with some test data
        df_data = {
            'temperature': [300.0, 310.0, 320.0],
            'pressure': [101325.0, 101000.0, 100500.0],
            'velocity': [1.0, 1.5, 2.0],
        }
        td.df = pd.DataFrame(df_data, index=[0.0, 1.0, 2.0])
        td.time = 1.0
        
        return td

    # ---- __init__ tests ----

    def test_init_default_values(self, time_dataframe):
        """Test default initialization values."""
        assert isinstance(time_dataframe.df, pd.DataFrame)
        assert time_dataframe.df.empty
        assert time_dataframe.time == -1.0

    # ---- get_labels tests ----

    def test_get_labels_empty_dataframe(self, time_dataframe):
        """Test get_labels with empty dataframe returns empty list."""
        labels = time_dataframe.get_labels()
        assert labels == []
        assert isinstance(labels, list)

    def test_get_labels_with_data(self, time_dataframe_with_data):
        """Test get_labels returns column names as list."""
        labels = time_dataframe_with_data.get_labels()
        assert set(labels) == {"Time", 'temperature', 'pressure', 'velocity'}
        assert isinstance(labels, list)
        assert all(isinstance(l, str) for l in labels)

    def test_get_labels_without_time(self, time_dataframe_with_data):
        """Test get_labels returns column names as list."""
        time_dataframe_with_data.time = -1
        labels = time_dataframe_with_data.get_labels()
        assert set(labels) == {'temperature', 'pressure', 'velocity'}
        assert isinstance(labels, list)
        assert all(isinstance(l, str) for l in labels)

    def test_get_labels_returns_copy(self, time_dataframe_with_data):
        """Test that get_labels returns a copy, not the internal reference."""
        labels = time_dataframe_with_data.get_labels()
        labels.append('fake_field')
        # Original should be unchanged
        assert 'fake_field' not in time_dataframe_with_data.get_labels()

    # ---- get_1D_value tests ----

    def test_get_1D_value_existing_field(self, time_dataframe_with_data):
        """Test get_1D_value returns correct Series for existing field."""
        result = time_dataframe_with_data.get_1D_value(
            position=(0.0, 0.0, 0.0),
            cell_index="cell1",
            material_name="water",
            field="temperature"
        )
        assert isinstance(result, pd.Series)
        # Check values and index match (name attribute differs: Series has column name, expected has None)
        np.testing.assert_array_equal(result.values, [300.0, 310.0, 320.0])
        pd.testing.assert_index_equal(result.index, pd.Index([0.0, 1.0, 2.0]))

    def test_get_1D_value_all_fields(self, time_dataframe_with_data):
        """Test get_1D_value for all fields in dataframe."""
        for field in ['temperature', 'pressure', 'velocity']:
            result = time_dataframe_with_data.get_1D_value(
                position=(0.0, 0.0, 0.0),
                cell_index="cell1",
                material_name="water",
                field=field
            )
            assert isinstance(result, pd.Series)
            assert len(result) == 3

    def test_get_1D_value_nonexistent_field(self, time_dataframe_with_data):
        """Test get_1D_value raises ValueError for nonexistent field."""
        with pytest.raises(ValueError, match="Field nonexistent not found"):
            time_dataframe_with_data.get_1D_value(
                position=(0.0, 0.0, 0.0),
                cell_index="cell1",
                material_name="water",
                field="nonexistent"
            )

    def test_get_1D_value_position_ignored(self, time_dataframe_with_data):
        """Test that position parameter is accepted but ignored (returns full column)."""
        result1 = time_dataframe_with_data.get_1D_value(
            position=(0.0, 0.0, 0.0),
            cell_index="cell1",
            material_name="water",
            field="temperature"
        )
        result2 = time_dataframe_with_data.get_1D_value(
            position=(100.0, 200.0, 300.0),
            cell_index="other_cell",
            material_name="steel",
            field="temperature"
        )
        pd.testing.assert_series_equal(result1, result2)

    def test_get_1D_value_cell_index_ignored(self, time_dataframe_with_data):
        """Test that cell_index parameter is accepted but ignored."""
        result = time_dataframe_with_data.get_1D_value(
            position=(0.0, 0.0, 0.0),
            cell_index="any_cell",
            material_name="any_material",
            field="pressure"
        )
        assert isinstance(result, pd.Series)
        assert len(result) == 3

    def test_get_1D_value_material_name_ignored(self, time_dataframe_with_data):
        """Test that material_name parameter is accepted but ignored."""
        result = time_dataframe_with_data.get_1D_value(
            position=(0.0, 0.0, 0.0),
            cell_index="cell1",
            material_name="any_material",
            field="velocity"
        )
        assert isinstance(result, pd.Series)

    # ---- set_time tests ----

    def test_set_time_existing(self, time_dataframe_with_data):
        """Test set_time with an existing time index."""
        time_dataframe_with_data.set_time(1.0)
        assert time_dataframe_with_data.time == 1.0
        # Should not add a new row since time exists
        assert len(time_dataframe_with_data.df) == 3

    def test_set_time_new(self, time_dataframe_with_data):
        """Test set_time with a new time index adds a row."""
        time_dataframe_with_data.set_time(3.0)
        assert time_dataframe_with_data.time == 3.0
        # Should add a new row with NaN values
        assert len(time_dataframe_with_data.df) == 4
        assert time_dataframe_with_data.df.loc[3.0].isna().all()

    def test_set_time_from_zero(self, time_dataframe):
        """Test set_time on empty dataframe creates first row."""
        time_dataframe.set_time(5.0)
        assert time_dataframe.time == 5.0
        assert len(time_dataframe.df) == 1
        # Row should have NaN values (no columns yet)
        assert time_dataframe.df.empty or time_dataframe.df.loc[5.0].isna().all()

    def test_set_time_float_precision(self, time_dataframe_with_data):
        """Test set_time handles float precision correctly."""
        time_dataframe_with_data.set_time(1.0000001)
        assert time_dataframe_with_data.time == 1.0000001
        # Should add a new row since 1.0000001 != 1.0 in index
        assert len(time_dataframe_with_data.df) == 4

    def test_set_time_negative(self, time_dataframe_with_data):
        """Test set_time with negative time value."""
        time_dataframe_with_data.set_time(-1.0)
        assert time_dataframe_with_data.time == -1.0
        assert len(time_dataframe_with_data.df) == 4

    # ---- append_data tests ----

    def test_append_data_new_column(self, time_dataframe_with_data):
        """Test append_data creates new column if field doesn't exist."""
        time_dataframe_with_data.append_data("new_field", 42.0)
        assert "new_field" in time_dataframe_with_data.df.columns
        assert time_dataframe_with_data.df.loc[1.0, "new_field"] == 42.0

    def test_append_data_existing_column(self, time_dataframe_with_data):
        """Test append_data updates existing column at current time."""
        # Initial value
        initial_val = time_dataframe_with_data.df.loc[1.0, "temperature"]
        
        time_dataframe_with_data.append_data("temperature", 999.0)
        
        assert time_dataframe_with_data.df.loc[1.0, "temperature"] == 999.0

    def test_append_data_nan_initial(self, time_dataframe_with_data):
        """Test that new columns are initialized with NaN values."""
        time_dataframe_with_data.append_data("computed", 10.0)
        
        # Existing rows should have NaN for the new field (except current time row)
        assert pd.isna(time_dataframe_with_data.df.loc[0.0, "computed"])
        assert time_dataframe_with_data.df.loc[1.0, "computed"] == 10.0
        assert pd.isna(time_dataframe_with_data.df.loc[2.0, "computed"])

    def test_append_data_multiple_times(self, time_dataframe_with_data):
        """Test append_data can be called multiple times."""
        time_dataframe_with_data.append_data("counter", 1)
        assert time_dataframe_with_data.df.loc[1.0, "counter"] == 1
        
        time_dataframe_with_data.append_data("counter", 2)
        assert time_dataframe_with_data.df.loc[1.0, "counter"] == 2

    def test_append_data_zero(self, time_dataframe_with_data):
        """Test append_data with zero value."""
        time_dataframe_with_data.append_data("zero_field", 0.0)
        assert time_dataframe_with_data.df.loc[1.0, "zero_field"] == 0.0

    def test_append_data_negative(self, time_dataframe_with_data):
        """Test append_data with negative value."""
        time_dataframe_with_data.append_data("neg_field", -5.5)
        assert time_dataframe_with_data.df.loc[1.0, "neg_field"] == -5.5

    def test_append_data_large(self, time_dataframe_with_data):
        """Test append_data with very large value."""
        large_val = 1e15
        time_dataframe_with_data.append_data("large_field", large_val)
        assert time_dataframe_with_data.df.loc[1.0, "large_field"] == large_val

    def test_append_data_small(self, time_dataframe_with_data):
        """Test append_data with very small value."""
        small_val = 1e-15
        time_dataframe_with_data.append_data("small_field", small_val)
        assert abs(time_dataframe_with_data.df.loc[1.0, "small_field"] - small_val) < 1e-20

    def test_append_data_nonexistent_field_error(self, time_dataframe_with_data):
        """Test append_data creates column for unknown field (no error)."""
        # This should NOT raise an error - it creates a new column
        time_dataframe_with_data.append_data("unknown_field", 1.0)
        assert "unknown_field" in time_dataframe_with_data.df.columns

    def test_append_data_preserves_index(self, time_dataframe_with_data):
        """Test append_data preserves the dataframe index."""
        original_index = time_dataframe_with_data.df.index.copy()
        time_dataframe_with_data.append_data("new", 1.0)
        pd.testing.assert_index_equal(time_dataframe_with_data.df.index, original_index)

    def test_append_data_multiple_columns(self, time_dataframe_with_data):
        """Test append_data with multiple different fields."""
        time_dataframe_with_data.append_data("field_a", 1.0)
        time_dataframe_with_data.append_data("field_b", 2.0)
        time_dataframe_with_data.append_data("field_c", 3.0)
        
        assert "field_a" in time_dataframe_with_data.df.columns
        assert "field_b" in time_dataframe_with_data.df.columns
        assert "field_c" in time_dataframe_with_data.df.columns
        assert time_dataframe_with_data.df.loc[1.0, "field_a"] == 1.0
        assert time_dataframe_with_data.df.loc[1.0, "field_b"] == 2.0
        assert time_dataframe_with_data.df.loc[1.0, "field_c"] == 3.0

    # ---- Integration tests ----

    def test_set_time_then_append_data(self, time_dataframe):
        """Test workflow: set_time followed by append_data."""
        time_dataframe.set_time(1.0)
        time_dataframe.append_data("temperature", 300.0)
        
        assert time_dataframe.time == 1.0
        assert "temperature" in time_dataframe.df.columns
        assert time_dataframe.df.loc[1.0, "temperature"] == 300.0

    def test_set_time_new_then_append_data(self, time_dataframe):
        """Test workflow: set_time with new value followed by append_data."""
        time_dataframe.set_time(5.0)
        time_dataframe.append_data("pressure", 101325.0)
        
        assert time_dataframe.time == 5.0
        assert "pressure" in time_dataframe.df.columns
        assert time_dataframe.df.loc[5.0, "pressure"] == 101325.0

    def test_multiple_time_updates(self, time_dataframe):
        """Test multiple set_time calls at different times."""
        for t in [0.0, 1.0, 2.0, 3.0]:
            time_dataframe.set_time(t)
            time_dataframe.append_data("temp", float(t * 100))
        
        assert len(time_dataframe.df) == 4
        assert time_dataframe.df.loc[0.0, "temp"] == 0.0
        assert time_dataframe.df.loc[1.0, "temp"] == 100.0
        assert time_dataframe.df.loc[2.0, "temp"] == 200.0
        assert time_dataframe.df.loc[3.0, "temp"] == 300.0

    def test_dataframe_structure_after_operations(self, time_dataframe):
        """Test dataframe maintains correct structure after multiple operations."""
        time_dataframe.set_time(1.0)
        time_dataframe.append_data("a", 1.0)
        time_dataframe.append_data("b", 2.0)
        
        time_dataframe.set_time(2.0)
        time_dataframe.append_data("a", 3.0)
        time_dataframe.append_data("c", 4.0)
        
        assert set(time_dataframe.df.columns) == {'a', 'b', 'c'}
        assert len(time_dataframe.df) == 2
        assert time_dataframe.df.loc[1.0, 'a'] == 1.0
        assert pd.isna(time_dataframe.df.loc[2.0, 'b'])
        assert pd.isna(time_dataframe.df.loc[1.0, 'c'])

    def test_get_labels_after_modifications(self, time_dataframe):
        """Test get_labels reflects changes after append_data."""
        initial_labels = time_dataframe.get_labels()
        assert initial_labels == []
        
        time_dataframe.append_data("new_col", 1.0)
        labels = time_dataframe.get_labels()
        assert labels == ['new_col']

    def test_set_time_with_empty_columns(self, time_dataframe):
        """Test set_time when dataframe has no columns."""
        time_dataframe.set_time(1.0)
        
        assert len(time_dataframe.df) == 1
        # With no columns, the row should still exist but be empty
        assert time_dataframe.time == 1.0


@pytest.mark.default
class TestTimeDataFrameEdgeCases:
    """Edge case tests for TimeDataFrame."""

    def test_empty_dataframe_get_1D_value(self):
        """Test get_1D_value on empty dataframe raises appropriate error."""
        from scivianna.interface.time_dataframe import TimeDataFrame
        td = TimeDataFrame()
        
        with pytest.raises(ValueError, match="not found"):
            td.get_1D_value(
                position=(0.0, 0.0, 0.0),
                cell_index="c1",
                material_name="m1",
                field="any"
            )

    def test_dataframe_with_nan_values(self):
        """Test handling of NaN values in dataframe."""
        from scivianna.interface.time_dataframe import TimeDataFrame
        td = TimeDataFrame()
        
        df_data = {'field': [np.nan, 1.0, np.nan]}
        td.df = pd.DataFrame(df_data, index=[0.0, 1.0, 2.0])
        
        result = td.get_1D_value(
            position=(0.0, 0.0, 0.0),
            cell_index="c1",
            material_name="m1",
            field="field"
        )
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == 1.0
        assert pd.isna(result.iloc[2])

    def test_dataframe_with_single_row(self):
        """Test TimeDataFrame with single row."""
        from scivianna.interface.time_dataframe import TimeDataFrame
        td = TimeDataFrame()
        
        df_data = {'field': [42.0]}
        td.df = pd.DataFrame(df_data, index=[0.0])
        td.time = 0.0
        
        result = td.get_1D_value(
            position=(0.0, 0.0, 0.0),
            cell_index="c1",
            material_name="m1",
            field="field"
        )
        assert len(result) == 1
        assert result.iloc[0] == 42.0

    def test_dataframe_with_many_columns(self):
        """Test TimeDataFrame with many columns."""
        from scivianna.interface.time_dataframe import TimeDataFrame
        td = TimeDataFrame()
        
        num_cols = 50
        df_data = {f"field_{i}": [float(i)] for i in range(num_cols)}
        td.df = pd.DataFrame(df_data, index=[0.0])
        
        labels = td.get_labels()
        assert len(labels) == num_cols
        
        for i in range(num_cols):
            result = td.get_1D_value(
                position=(0.0, 0.0, 0.0),
                cell_index="c1",
                material_name="m1",
                field=f"field_{i}"
            )
            assert len(result) == 1
            assert result.iloc[0] == float(i)

    def test_append_data_with_special_column_names(self):
        """Test append_data with special characters in column names."""
        from scivianna.interface.time_dataframe import TimeDataFrame
        td = TimeDataFrame()
        
        td.df = pd.DataFrame({'existing': [1.0]}, index=[0.0])
        td.time = 0.0
        
        td.append_data("field with spaces", 2.0)
        td.append_data("field-with-dashes", 3.0)
        td.append_data("field.with.dots", 4.0)
        
        assert "field with spaces" in td.df.columns
        assert "field-with-dashes" in td.df.columns
        assert "field.with.dots" in td.df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])