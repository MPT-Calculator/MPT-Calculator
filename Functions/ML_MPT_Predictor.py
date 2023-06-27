"""
James Elgy 2021
module to use mpt eigenvalue data over a range of frequencies and permeabilities to perform curve fitting.
"""

import numpy as np
import pandas as pd
import seaborn as sns
#import matplotlib
# matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from sklearn import neural_network as nn
from sklearn import preprocessing as pp


class DataLoader():

    def __init__(self):
        self.raw_data = None
        self.data = None
        self.scaling_dict = None  # Dictionary containing scaling for each component.

    def load_data(self, filename: str, drop_extra_cols=True) -> pd.DataFrame:
        """
        Loads data from csv and outputs a pandas dataframe
        :param filename: input file name for csv file.
        :return: dataframe.
        """
        data = pd.read_csv(filename)
        data = data.drop(columns='Unnamed: 0')  # Removing index array from dataframe.

        if drop_extra_cols:
            data = data.drop(columns='N0')
            data = data.drop(columns='tensor_coeffs')
        self.raw_data = data
        return data

    def preprocess_data(self):
        """
        For each component in self.raw_data, set mean to 0 and standard deviation to 1.
        :return: data
        """
        data = pd.DataFrame(index=self.raw_data.index)
        scaling_dict = {}
        col_names = [col for col in self.raw_data.columns]
        for component in col_names:
            values = self.raw_data[component].to_numpy()[:, None]
            if component == 'omega':
                values = np.log10(values)
            if component != 'split':
                scalar = pp.StandardScaler()
                values = scalar.fit_transform(values)
                scaling_dict[component] = scalar
            data.insert(0, component, values, allow_duplicates=True)

        self.data = data
        self.scaling_dict = scaling_dict
        return scaling_dict, data

    def postprocess_data(self):
        """
        Function to undo the normalistion performed in preprocess_data.
        :return:
        """

        descaled_data = pd.DataFrame(index=self.data.index)
        col_names = [col for col in self.raw_data.columns]
        for component in col_names:
            values = self.data[component].to_numpy()[:, None]
            if component != 'split':
                scalar = self.scaling_dict[component]
                values = scalar.inverse_transform(values)
            descaled_data.insert(0, component, values, allow_duplicates=True)
        self.data = descaled_data

        return descaled_data

    def train_test_split(self, proportion: float):
        """
        Splits total data into training and testing sets and assigns flags to dataframe
        :param proportion: proportion of total data to be used for testing. 0 < float < 1
        :return:
        """
        if self.data is None:  # Allowing for split to be performed before preprocessing
            total_samples = self.raw_data.shape[0]
        else:
            total_samples = self.data.shape[0]
        N_test_samples = int(np.floor(total_samples * proportion))
        test_indices = np.random.randint(0,
                                         high=total_samples + 1,  # randint goes from low(inclusive) to high(exclusive)
                                         size=N_test_samples)

        flags = ['training'] * total_samples
        flags = ['testing' if ind in test_indices else 'training' for ind in range(total_samples)]
        if self.data is None:
            self.raw_data.insert(0, 'split', flags, allow_duplicates=True)
        else:
            self.data.insert(0, 'split', flags, allow_duplicates=True)

    def plot_data(self, x='omega', y='mur', z=None, use_raw_data=False, fig=None):
        """
        plotting function for data exploration.
        Plots x against y. If z != None, plots in 3d.
        :param x: x component
        :param y: y component
        :param z: z component
        :return:
        """
        # sns.set()
        # sns.set_style("ticks")

        if use_raw_data == True:
            plot_data = self.raw_data
            plot_data['omega'] = np.log10(plot_data['omega'])
        else:
            plot_data = self.data
        if fig == None:
            fig = plt.figure()
        else:
            fig = plt.figure(fig)
        if z is None:
            sns.scatterplot(data=plot_data, x=x, y=y, hue='split', palette=['b', 'g'])
        else:

            plot_data.query('omega<=5 and mur<=50', inplace=True)
            ax = plt.axes(projection='3d')
            s = plot_data['split']
            ax.scatter(plot_data[x][s == 'training'], np.log10(plot_data[y][s == 'training']), plot_data[z][s == 'training'],
                       color='r', label='training')
            ax.scatter(plot_data[x][s != 'training'], np.log10(plot_data[y][s != 'training']), plot_data[z][s != 'training'],
                       color='m', label='testing')
            ax.legend()
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_zlabel(z)

        return fig

class ML():

    def __init__(self, DataLoader, component):
        self.data = DataLoader.data
        self.scaled_data = DataLoader.data
        self.scaling_dict = DataLoader.scaling_dict
        self.component = component

    def perform_fit(self, neurons, activation='tanh', tol=1e-8, alpha=0):
        split = self.data['split']
        x = self.data[['omega', 'mur']][split == 'training'].to_numpy()
        y = np.ravel(self.data[self.component][split == 'training'].to_numpy())

        regressor = nn.MLPRegressor(hidden_layer_sizes=neurons, max_iter=500000, activation=activation,
                                    solver='lbfgs', tol=tol, alpha=alpha, verbose=False, warm_start=False,
                                    random_state=None, n_iter_no_change=1000, max_fun=100000)

        regression = regressor.fit(x, y)
        self.regression = regression
        return regression

    def postprocess_data(self):
        """
        Function to undo the normalistion performed in preprocess_data.
        :return:
        """
        # self.scaled_data = self.data
        descaled_data = pd.DataFrame(index=self.data.index)
        col_names = [col for col in self.data.columns]
        for component in col_names:
            values = self.data[component].to_numpy()[:, None]
            if component != 'split':
                scalar = self.scaling_dict[component]
                values = scalar.inverse_transform(values)
            descaled_data.insert(0, component, values, allow_duplicates=True)
        self.data = descaled_data

        return descaled_data

    def eval_fit(self):
        split = self.data['split']
        x = self.scaled_data[['omega', 'mur']][split == 'testing'].to_numpy()
        y_true = np.ravel(self.data[self.component][split == 'testing'].to_numpy())
        y_pred = self.regression.predict(x)
        y_pred = self.scaling_dict[self.component].inverse_transform(y_pred)

        MSE = np.sum((y_pred - y_true) ** 2) / len(y_true)
        RMSE = np.sqrt(MSE)
        NRMSE = RMSE / (np.sqrt(np.sum(y_true ** 2) / len(y_true)))

        return NRMSE

    def make_prediction(self, omega, mur):
        """
        Function to make final prediction given query values of omega and mur
        :param omega:
        :param mur:
        :param component:
        :return:
        """

        omega = self.scaling_dict['omega'].transform(omega[:,None])
        mur = self.scaling_dict['mur'].transform(mur[:,None])
        x = np.squeeze(np.asarray([omega, mur]).transpose())
        y = self.regression.predict(x)
        y = self.scaling_dict[self.component].inverse_transform(y)
        return y

    def generate_surface(self, omega_array, mur_array, fig=None):
        omega_array = np.log10(omega_array)
        xx, yy = np.meshgrid(omega_array, mur_array)
        x = np.ravel(xx)
        y = np.ravel(yy)
        z = self.make_prediction(x,y)
        zz = np.reshape(z, xx.shape)
        return fig, zz
        if fig == None:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
        else:
            plt.figure(fig)
            ax = fig.axes[0]
        ax.ticklabel_format(style='sci', axis='z')
        ax.plot_wireframe(xx, np.log10(yy), zz, alpha=1, label='ML prediction')
        plt.legend()

        return fig, zz

    def save_all_figures(path, format='png', suffix='', prefix=''):
        """
        Function to save all open figures to disk.
        Files are named as:
        {suffix}{figure_n}{prefix}.{format}
        :param path: path to the desired saving directory.
        :param format: desired file format. pdf, png, jpg, tex
        :param suffix: additional component of the output filename
        :param prefix: additional component of the output filename
        :return:
        """

        if not os.path.isdir(path):
            os.mkdir(path)
        extension = '.' + format
        if format != 'tex':
            for i in plt.get_fignums():
                plt.figure(i)
                filename = prefix + f'figure_{i}' + suffix
                plt.savefig(os.path.join(path, filename) + extension)
        elif format == 'tex':
            for i in plt.get_fignums():
                plt.figure(i)
                filename = prefix + f'figure_{i}' + suffix
                tikzplotlib.save(os.path.join(path, filename) + extension)
        else:
            raise TypeError('Unrecognised file format')


def calc_eddy_current_limit(frequency_array, permeability_array, sigma, alpha):
    """
    James Elgy - 2022:
    function to calculate when the wavelength is smaller than the object size. This is then returned as an
    upper limit on the eddy current approximation.
    :return:
    """

    c = 299792458  # Speed of light m/s

    # wavelength of EM radiation. Assumes conductivity of 0 and relative permittivity of 1. c/(n *f) = lambda
    # converts rad/s to Hz.
    limit_frequency = []
    for mur in permeability_array:
        epsilon = 8.854e-12
        mu = mur * 4 * np.pi * 1e-7
        k = np.sqrt(frequency_array**2 * epsilon * mu + 1j * mu * sigma * frequency_array)
        wavelength = 2*np.pi/k.real
        fudge_factor = 1 # alpha uses a global scaling but the geo file may not define a unit object.
        # wavelength = c / (np.sqrt(1*mur) * (self.frequency_array/(2*np.pi)))
        for lam, freq in zip(wavelength, frequency_array):
            if lam <= alpha*fudge_factor:
                max_frequency = freq
                break
            else:
                max_frequency = np.nan
        limit_frequency += [max_frequency]

    return limit_frequency


if __name__ == '__main__':

    # sns.set()
    # sns.set_style("ticks")

    comp_array = ['eig_1_real']#, 'eig_2_real', 'eig_3_real', 'eig_1_imag', 'eig_2_imag', 'eig_3_imag']
    for q in range(1):  # Using range here because I need to index the list of names, c.
        component = comp_array[q]
        c = ['$re(\lambda_1)$', '$re(\lambda_2)$', '$re(\lambda_3)$', '$im(\lambda_1)$', '$im(\lambda_2)$','$im(\lambda_3)$']
        s = []
        for I in range(1):
            D = DataLoader()
            data = D.load_data(
                r'C:\Users\James\Desktop\MPT_calculator_James\OutputData\hammer_omega_1.00-5.00_mur_0.00-1.70_sigma_1.70e+06_alpha_0.001_len_324_2022-07-31.csv')
            D.train_test_split(0.2)
            scaling, data = D.preprocess_data()


            ml = ML(D, component)
            ml.perform_fit((8,8))
            ml.postprocess_data()
            score = ml.eval_fit()
            print(score)
            s += [score]

        plt.hist(s, bins=20, label=f'component={c[q]}', alpha=0.4)
        plt.xlim(left=-0.0005, right=0.2)
        plt.ylim(bottom=0, top=300)
        plt.ylabel('Frequency')
        plt.xlabel('NRMSE')

    f = D.plot_data(x='omega', y='mur', z=component, use_raw_data=True)
    omega_array = np.logspace(1,5,36)
    mur_array = np.logspace(0,np.log10(50),36)
    z = ml.generate_surface(omega_array, mur_array, fig=f)


    ax = plt.gca()
    import matplotlib.ticker as mticker
    def log_tick_formatter(val, pos=None):
        if val == 0:
            return '$10^0$'
        else:
            return f"$10^{val:.0f}$"

    startx, endx = ax.get_xlim()
    starty, endy = ax.get_ylim()
    ax.xaxis.set_ticks(np.round(np.arange(startx, endx, 1)))
    ax.yaxis.set_ticks(np.round(np.arange(starty, endy, 1)))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))

    # max_omega = calc_eddy_current_limit(np.logspace(1,5,41), np.logspace(0,np.log10(50),41), 1e6, 1e-2)

    z_max = ax.get_zlim()[1]
    z_min = ax.get_zlim()[0]

    yy, zz = np.meshgrid(np.log10(mur_array), np.linspace(z_min, z_max, 50))
    xx = np.ones(yy.shape)

    # for ind, omega in enumerate(max_omega):
    #     xx[:, ind] = np.log10(omega)

    # surf = ax.plot_surface(xx, yy, zz, color='m', alpha=0.4, linewidth=0, label='Eddy current limit')
    # surf._facecolors2d = surf._facecolor3d
    # surf._edgecolors2d = surf._edgecolor3d
    plt.legend()
