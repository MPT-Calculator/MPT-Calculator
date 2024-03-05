import os
import numpy as np

from memory_profiler import memory_usage
from matplotlib.ticker import MaxNLocator
from main import main
from time import time, time_ns, sleep
import multiprocessing
from matplotlib import pyplot as plt
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
import matplotlib.text as mtext


class LegendTitle(object):
    """
    This is a small class to enable headings to be added to pyplot legends
    """
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, r'\underline{' + orig_handle + '}', usetex=True, **self.text_props)
        handlebox.add_artist(title)
        return title



def measure_memory_usage(n_cpus=1, order=0):
    """
    James Elgy 2024
    Function to run and time the MPT-Calculator main function. This function currently measures the memory usage using the memory_profiler package
    and relies on that to give accurate estimations of the total memory used my the main function.
    
    Options can be configured using the options_dict dictionary.
    Memory usage and timings are outputted to timings_cpus={n_cpus} and memory_usage_cpus={n_cpus} files.
    
    Currently only supports parallel POD.

    Args:
        n_cpus (int, optional): Number of CPUs to use in the parallel processing. Defaults to 2.
        order (int, optional): Desired order for FEM simulation. Defaults to 3.

    Returns:
         mem (Numpy Array),
         timings (dict): Dictionary of times at different points in the code. 
         function_call_time (int): Time stamp for when main is run
         time_array (Numpy Array): Array of time intervals where memory is queried
         ret (dict): Return dictionary from main function
    """
    
    geo = 'OCC_test_sphere_prism_32.geo'
    options_dict = {'order': order, 'geometry': geo, 'use_OCC': False, 'use_POD': True, 'cpus': n_cpus, 'alpha': 0.01,
                            'use_parallel': True, 'use_iterative_POD': False, 'start_stop':(1,8,40)}
    function_call_time = time()
    mem, ret = memory_usage((main, (), options_dict), interval=0.1, retval=True, include_children=True)
    function_stop_time = time()

    total_time = function_stop_time - function_call_time
    time_array = np.linspace(0, total_time, len(mem))

    timings = np.load('Results/' + ret['SweepName'] + f'/Data/Timings_cpus={n_cpus}.npy', allow_pickle=True).all()
    timings['Function_Call_time'] = function_call_time
    timings['Function_Stop_time'] = function_stop_time

    np.save('Results/' + ret['SweepName'] + f'/Data/Timings_cpus={n_cpus}.npy', timings)
    np.save('Results/' + ret['SweepName'] + f'/Data/ReturnDict.npy', ret)
    np.save('Results/' + ret['SweepName'] + f'/Data/memory_usage_cpus={n_cpus}.npy', [mem, time_array])

    return mem, timings, function_call_time, time_array, ret

def plot_times(Datadir: str, n_cpus: int):
    """
    James Elgy - 2024
    Plotting function to plot time and memory usage for the main function in MPT-Calculator.
    
    Args:
        Datadir (str): path to the data directory for the sweep of interest
        n_cpus (int): number of cpus used in the sweep of interest.
    """
    
    plt.figure()
    cols = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:cyan', 'tab:pink']

    times = np.load(Datadir + f'Timings_cpus={n_cpus}.npy', allow_pickle=True).all()
    
    try:
        mem = np.load(Datadir + f'memory_usage_cpus={n_cpus}.npy', allow_pickle=True)
        mem_array_mat = mem[0,:]
        time_array_mat = mem[1,:]
        plt.plot(time_array_mat, mem_array_mat, label='Memory Usage', color='k')
    except FileNotFoundError:
        pass
    
    # Doing Plot
    timings = times
    
    try:
        function_call_time = timings['Function_Call_time']
    except KeyError:
        print(f'The call time for the main function was not recorded. An estimate for the total time is provided by {timings["Tensors"] - timings["start_time"] = }s')
        function_call_time = timings['start_time']
        
    timing_list = [timings['start_time'] - function_call_time,
                    timings['Theta0'] - function_call_time,
                    timings['Theta1'] - function_call_time,
                    timings['ROM'] - function_call_time,
                    timings['SolvedSmallerSystem'] - function_call_time,
                    timings['BuildSystemMatrices'] - function_call_time,
                    timings['Tensors'] - function_call_time]
    
    b1 = plt.axvspan(0, timing_list[0], label='Computing Mesh', facecolor=cols[0], alpha=0.4)
    b2 = plt.axvspan(timing_list[0], timing_list[1], label=r'Computing $\mathbf{o}_i(\mu_r)$', facecolor=cols[1],
                alpha=0.4)
    b3 = plt.axvspan(timing_list[1], timing_list[2], label=r'Computing $\mathbf{q}_i(\omega_k)$ Snapshots',
                facecolor=cols[2], alpha=0.4)
    b4 = plt.axvspan(timing_list[2], timing_list[3], label=r'TSVD \& Constucting Smaller System', facecolor=cols[3], alpha=0.4)
    b5 = plt.axvspan(timing_list[3], timing_list[4], label=r'Solving Smaller System', facecolor=cols[4], alpha=0.4)
    b6 = plt.axvspan(timing_list[4], timing_list[5], label=r'Building Matrices', facecolor=cols[5], alpha=0.4)
    b7 = plt.axvspan(timing_list[5], timing_list[6], label=r'Computing MPT Spectral Signature', facecolor=cols[6], alpha=0.4)
    plt.legend()
    plt.ylabel('Memory Usage, [MB]')
    plt.xlabel('Time, [sec]')

    
    plt.legend(['Off-line', b1, b2, b3, b4, 'On-line',b5, b6, b7], ['', 'Computing Mesh', 'Computing $\mathbf{o}_i(\mu_r)$', 'Computing $\mathbf{q}_i(\omega_k)$ Snapshots', 'TSVD \& Constucting Smaller System', '', 'Solving Smaller System', 'Building Matrices', 'Computing MPT Spectral Signature' ],
        handler_map={str: LegendTitle({'fontsize': 11})})
    
    plt.ylabel('Memory Usage, [MB]')
    plt.xlabel('Time, [sec]')




if __name__ == '__main__':
    mem, timings, function_call_time, time_array, ret = measure_memory_usage(n_cpus=1, order=0)
    Datadir = 'Results/' + ret['SweepName'] + '/Data/'
    
    # Datadir = r'/home/james/Desktop/MPT-Calculator-Testing-Branch/MPT-Calculator/Results/OCC_Gun_modelv2_30/al_0.01_mu_1,100_sig_3.5e7,4.5e6/1e1-1e8_40_el_32925_ord_5_POD_13_1e-6/Data/'
    plot_times(Datadir, 1)
    plt.show()




