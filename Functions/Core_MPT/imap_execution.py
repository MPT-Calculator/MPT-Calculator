import gc
from ..Core_MPT.Theta0 import *
from ..Core_MPT.Theta1 import *
from ..Core_MPT.Theta1_Sweep import *
from ..Core_MPT.Theta1_Lower_Sweep import *
from ..Core_MPT.Theta1_Lower_Sweep_Mat_Method import *

def imap_version(args):
    """
    James Elgy 2023:
    Function to convert between the multi argument starmap and the single argument imap function in parallel pool.
    This is so that a progress bar can be drawn effectively.

    Parameters
    ----------
    args, list of tuples where each entry is a tuple containing the input arguments. Last input argument must be
    function name. e.g. 'Theta1_Sweep

    Returns
    -------
    Function evaluation at those input arguments.
    """
    function_name = args[-1]
    args = args[:-1]
    # Since eval allows the user to execute arbitrary code, I've set a list of allowed functions.
    allowed_functions = ['Theta0',
                         'Theta1',
                         'Theta1_Lower_Sweep',
                         'Theta1_Sweep',
                         'Theta1_Lower_Sweep_Mat_Method']

    if function_name not in allowed_functions:
        print(f'Function {function_name} is not trusted')
        return 0
    else:
        output = eval(function_name + '(*args)')
        del args
        gc.collect()
        return output