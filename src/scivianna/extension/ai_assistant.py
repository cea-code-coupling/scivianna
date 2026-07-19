
from pathlib import Path
from typing import TYPE_CHECKING
import panel as pn
import panel_material_ui as pmui
import scivianna
from scivianna.icon import get_icon
from scivianna.data.data2d import Data2D
from scivianna.extension.extension import Extension
from scivianna.plotter_2d.generic_plotter import Plotter2D
from scivianna.slave import ComputeSlave

if TYPE_CHECKING:
    from scivianna.panel.visualisation_panel import VisualizationPanel


class AIAssistant(Extension):
    """Extension to load files and send them to the slave."""

    def __init__(
        self,
        slave: ComputeSlave,
        plotter: Plotter2D,
        panel: "VisualizationPanel"
    ):
        """Constructor of the extension, saves the slave and the panel

        Parameters
        ----------
        slave : ComputeSlave
            Slave computing the displayed data
        plotter : Plotter2D
            Figure plotter
        panel : VisualizationPanel
            Panel to which the extension is attached
        """
        super().__init__(
            "AI assistant",
            get_icon("auto_fix_high"),
            slave,
            plotter,
            panel,
        )
        try:
            from scivianna.agent.data_2d_worker import Data2DWorker
            self.description = """
    The AI assistant allows you to edit the plot cells colors based on its values. A code will be generated and executed at each cells updates.

    You can for example request:

    - "Highlight highest value"
    - "Hide zeros"
    - ...
    """

            self.prompt_text_input = pmui.TextInput(
                label="Prompt",
                placeholder='Type your prompt ...',
                enter_pressed=True,
                size="small",
                variant="outlined",
                # sizing_mode="stretch_width",
                margin=1,
                width = 200
            )
            prompt_clear_button = pmui.IconButton(
                label="Clear",
                icon=get_icon('clear'),
                variant="outlined",
                description='Clear the prompt text',
                margin=1,
                )
            prompt_run_button = pmui.IconButton(
                label="Run",
                icon=get_icon('auto_fix_high'),
                variant="contained",
                description='Apply to the geometry',
                margin=1,
            )
            def clear_prompt(*args, **kwargs):
                self.prompt_text_input.value = ""
                self.llm_code = ""
                self.panel.recompute()
                
            def exec_prompt(*args, **kwargs):
                if self.prompt_text_input.value != "":
                    dw = Data2DWorker(self.current_data)
                    valid, llm_code = dw(self.prompt_text_input.value)

                    if valid and isinstance(llm_code, str):
                        self.code_editor.value = llm_code
                        self.dialog.param.update(open=True)
                    else:
                        if self.prompt_text_input.value != "":
                            exec_prompt()

            prompt_clear_button.on_click(clear_prompt)
            prompt_run_button.on_click(exec_prompt)

            self.agent_row = pn.Row(
                self.prompt_text_input, 
                prompt_clear_button, 
                prompt_run_button, 
                align=("start", "center"),
                margin = (10, 5),
            )
        
            self.llm_code = ""
            self.llm_comment = pn.pane.Markdown("")
            self.code_editor = pn.widgets.CodeEditor(value="", language='python', theme='monokai')
            self.code_valid_button = pmui.IconButton(icon=(Path(scivianna.__file__).parent / "icon" / "check.svg").read_text().strip())
            self.code_invalid_button = pmui.IconButton(icon=(Path(scivianna.__file__).parent / "icon" / "clear.svg").read_text().strip())
            self.dialog = pmui.Dialog(
                pn.Column(
                    self.llm_comment, 
                    self.code_editor, 
                    pn.Row(self.code_valid_button, self.code_invalid_button, align="end"), 
                ), 
                open=False, 
                full_screen=False, 
                show_close_button=True, 
                close_on_click=False, 
            )

            def valid_code(e):
                self.llm_code = self.code_editor.value
                self.code_editor.value = ""
                self.panel.recompute()

            def invalid_code(e):
                self.llm_code = ""
                self.code_editor.value = ""

            self.code_valid_button.on_click(valid_code) 
            self.code_invalid_button.on_click(invalid_code) 
            self.code_valid_button.js_on_click(args={'dialog': self.dialog}, code="dialog.data.open = false")
            self.code_invalid_button.js_on_click(args={'dialog': self.dialog}, code="dialog.data.open = false")

            self.has_agent = True

        except Exception as e:
            self.description = f"""
The AI assistant allows you to edit the plot cells colors based on its values. A code will be generated and executed at each cells updates.

Agent not loaded, received error : 
*{e}*
"""
            self.agent_row = pn.Row(
                self.description
            )
            self.has_agent = False


    def make_gui(self,) -> pn.viewable.Viewable:
        """Returns a panel viewable to display in the extension tab.

        Returns
        -------
        pn.viewable.Viewable
            Viewable to display in the extension tab
        """
        col = pmui.Column(
        )
        if self.has_agent:
            col.append(self.dialog)
        return col

    def on_updated_data(self, data: Data2D):
        """Function called when the displayed data is being updated. Extension can edit the data on its way to the plotter.

        Parameters
        ----------
        data : Data2D
            Data to display
        """
        if not self.has_agent:
            return

        from scivianna.agent.data_2d_worker import Data2DWorker

        self.current_data = data
        if self.llm_code != "":
            data_worker = Data2DWorker(self.current_data.copy())
            try:
                data_worker.execute_code(self.llm_code)
                self.current_data = data_worker.data2d.copy()
            except Exception as e:
                print("Execution failed, got error ", e)
                self.current_data = data_worker.data2d_save.copy()

        data.cell_colors = self.current_data.cell_colors
        data.cell_edge_colors = self.current_data.cell_edge_colors
        data.cell_ids = self.current_data.cell_ids
        data.cell_values = self.current_data.cell_values
