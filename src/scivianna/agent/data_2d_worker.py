
import numpy as np
from pathlib import Path
import re

try:
    from smolagents import tool, CodeAgent
except ImportError as e:
    raise ImportError(e, "To install the dependancies to use an agent, install scivianna using the command pip install scivianna[agent]")

from scivianna.data.data2d import Data2D
from scivianna.data.data_2d_worker import Data2DWorker
from scivianna.agent.llm_model import call_llm, ai_server


def extract_python_code(text):
    pattern = r"```python\s*([\s\S]*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None

class Data2DWorker(Data2DWorker):
    """Worker that receives a Data2D object and works with it. """

    def __init__(self, data2d:Data2D):
        """Worker that receives a Data2D object and works with it. 

        Parameters
        ----------
        data2d : Data2D
            Data class containing geometrical properties
        """
        self.data2d = data2d.copy()
        self.data2d_save = data2d.copy()

        @tool
        def check_valid():
            """Check if the Data2D is valid

            Raises
            ------
            AssertionError
                Current Data2D is not valid
            """
            self.check_valid()

        @tool
        def get_values() -> np.ndarray:
            """Returns the value per 2D cell of the geometry

            Returns
            -------
            np.ndarray
                Numpy array with the value per cell
            """
            return self.get_values()
        
        @tool
        def get_ids() -> np.ndarray:
            """Returns the ID per 2D cell of the geometry

            Returns
            -------
            np.ndarray
                Numpy array with the value per cell
            """
            return self.get_ids()
        
        @tool
        def set_alphas(alphas:np.ndarray) -> bool:
            """
            Sets the cells opacity values, expects a numpy array of integers between 0 and 255. return True if it's ok.

            Args:
                alphas : np.ndarray, Opacity values     

            Returns
            -------
            bool
                True if the given array is ok            
            """
            return self.set_alphas(alphas)
        
        @tool
        def get_colors() -> np.ndarray:
            """Returns the color per 2D cell of the geometry. 
            The returned object is a np.ndarray of shape (cell_count, 4), with values ranging from 0 to 255.

            Returns
            -------
            np.ndarray
                Numpy array with the value per cell
            """
            return self.get_colors()
        
        @tool
        def set_colors(colors:np.ndarray) -> bool:
            """Sets the cells color values, expects a numpy array of integers between 0 and 255 of shape (cell_count, 4). return True if it's ok.

            Args:
            colors : np.ndarray, np array containing the new per cell colors. expects a numpy array of integers between 0 and 255 of shape (cell_count, 4).

            Returns
            -------
            bool
                True if the given array is ok
            """
            return self.set_colors(colors)
        
        @tool
        def reset():
            """Returns to the data2d provided in the initialization
            """
            return self.reset()
        
        @tool
        def get_numpy() -> np:
            """Returns the numpy module
            """
            return self.get_numpy()
            
        @tool
        def execute_code(code_to_execute:str) -> bool:
            """Applies a code string to the current object. Used to repeat an already processed prompt.

            Args:
            code_to_execute : Code to execute in the Data2DWorker.

            Returns:
            bool : Code is valid.
            str : comment on why the code is not valid

            """
            return self.execute_code(code_to_execute)

        with open(Path(__file__).parent / "instructions.md", "r") as f:
            instructions = f.read()

        self.smoll_agent = CodeAgent(
                        tools=[execute_code, 
                               check_valid, get_values, set_alphas, get_colors, set_colors, get_ids, reset, get_numpy
                               ],  # List of tools available to the agent
                        # final_answer_checks=[code_is_ok],
                        model=ai_server, 
                        additional_authorized_imports=["numpy"],
                        verbosity_level=2,  # Show detailed agent reasoning
                        instructions=instructions,
                        use_structured_outputs_internally=True,
                        planning_interval=None)
        
        self.smoll_agent.python_executor.send_tools(self.smoll_agent.tools)
        
        self.python_executor = self.smoll_agent.python_executor

    def extract_exectute_code(self, text:str):
        if not "execute_code" in text:
            return
        
        last_exec = 'execute_code("""' + text.split('execute_code("""')[-1]
        last_exec = text.split('""")')[0] + '""")'

        return last_exec
    
    

    def __call__(self, question, reset=False, images=[], max_steps=15, additional_args={}):
        
        print(f"\nLLM used by agent :\n\n- {ai_server.client_kwargs['base_url']}\n- {ai_server.model_id}\n")

        self.executed_code = None

        with open(Path(__file__).parent / "instructions.md", "r") as f:
            instructions = f.read()
        input_messages = question

        step = 0

        while self.executed_code is None and step < max_steps:
            print(f"Executing step {step}")

            msg = call_llm(input_messages, instructions)
            code = extract_python_code(msg)

            if code is None:
                input_messages += 'Returned value not containing a code block respecting the format """python your code here """'
            else:
                try:
                    _, execution_output = self.execute_code(code)
                    self.executed_code = code
                except Exception as e:
                    execution_output = e

                input_messages += f"\n PAST ANSWER \n{execution_output} "
            step += 1

        if self.executed_code is None:
            return False, ""
        
        return True, self.executed_code

    def execute_code(self, code_to_execute:str):
        """Applies a code string to the current object. Used to repeat an already processed prompt.

        Parameters
        ----------
        code_to_execute : str
            Code to execute in the Data2DWorker
        """
        code_to_execute = code_to_execute.replace("import numpy as np", "np = get_numpy()")
        context_string = "\n".join(f"{e} = self.{e}" for e in [
            "check_valid", 
            "get_values", 
            "set_alphas", 
            "get_colors", 
            "set_colors", 
            "reset", 
            "get_numpy"
            ])
        
        self.smoll_agent.python_executor.__call__(code_to_execute)

        try:
            self.check_valid()
        except AssertionError as e:
            return False, f"The code was executed, but the resulting Data2D is not valid. Following problem found {e}."

        if self.has_changed():
            self.executed_code = context_string+"\n"+code_to_execute
            return True, "Success"
        else:
            return False, "The code was executed properly but did not change the Data2D."


if __name__ == "__main__":
    
    from pathlib import Path

    # Field example
    import scivianna
    from scivianna.constants import GEOMETRY, X, Y
    from scivianna.interface.med_interface import MEDInterface
    from scivianna.extension.field_selector import set_colors_list
    from scivianna.data.data2d import Data2D
    from scivianna.agent.data_2d_worker import Data2DWorker
    from scivianna.plotter_2d.polygon.matplotlib import Matplotlib2DPolygonPlotter

    med = MEDInterface()
    med.read_file(
        str(Path(scivianna.__file__).parent / "input_file" / "power.med"),
        GEOMETRY,
    )
    data_2d:Data2D
    data_2d, _ = med.compute_2D_data(
        X,
        Y, 
        0, 1, 0, 1,  # values not used
        0., 
        None,
        {},
        caller="AgentWorker"
    )
    set_colors_list(data_2d, med, "INTEGRATED_POWER", "viridis", False, {})

    dw = Data2DWorker(data_2d)
    code_is_ok, code = dw("highlight the highest value cell, hide zero values, dim the rest")
    plotter = Matplotlib2DPolygonPlotter()
    plotter.plot_2d_frame(dw.data2d)
    plotter.figure.savefig("my_plot.png")
