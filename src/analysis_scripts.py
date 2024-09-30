import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.mixture import GaussianMixture
from scipy.signal import find_peaks
import os as os

output_folder = 'Repeat_3/python_results'

data_dir = {
    'aBc_acd':'Z:/Chaperone_subgroup/NickM/Collaborations/Josh/FCS_data/240723 FCS Nick data/CSVs/aB-c ACD 240716 repeat.csv',
    'aBc_wt':'Z:/Chaperone_subgroup/NickM/Collaborations/Josh/FCS_data/240716 corrected excel data/CSVs/aB-c WT.csv',
    'hsp27_3d':'Z:/Chaperone_subgroup/NickM/Collaborations/Josh/FCS_data/240723 FCS Nick data/CSVs/Hsp273D 240716 repeat.csv',
    'hsp27_acd':'Z:/Chaperone_subgroup/NickM/Collaborations/Josh/FCS_data/240716 corrected excel data/CSVs/Hsp27 ACD.csv',
    'tdp_and_hsp27_acd':'Z:/Chaperone_subgroup/NickM/Collaborations/Josh/FCS_data/240716 corrected excel data/CSVs/TDP-43 + Hsp27 ACD.csv',
    'tdp_and_hsp27_3d':'Z:/Chaperone_subgroup/NickM/Collaborations/Josh/FCS_data/240716 corrected excel data/CSVs/TDP-43 + Hsp273D.csv',
    'tdp_and_abc_acd':'Z:/Chaperone_subgroup/NickM/Collaborations/Josh/FCS_data/240716 corrected excel data/CSVs/TDP-43 + aB-c ACD.csv',
    'tdp_and_abc_wt':'Z:/Chaperone_subgroup/NickM/Collaborations/Josh/FCS_data/240716 corrected excel data/CSVs/TDP-43 aB-c WT.csv',
    'tdp_only':'Z:/Chaperone_subgroup/NickM/Collaborations/Josh/FCS_data/240716 corrected excel data/CSVs/TDP-43.csv',
    'tdp_GdHCl':'Z:/Chaperone_subgroup/NickM/Collaborations/Josh/FCS_data/240716 corrected excel data/CSVs/TDP-43 6M GdHCl.csv',
    'tdp_gfp':'Z:/Chaperone_subgroup/NickM/Collaborations/Josh/FCS_data/240716 corrected excel data/CSVs/TDP-43 + GFP.csv',
    'gfp':'Z:/Chaperone_subgroup/NickM/Collaborations/Josh/FCS_data/240716 corrected excel data/CSVs/GFP.csv',

}

font = {'weight' : 'normal', 'size'   : 12 }
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
plt.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'


def cleanup_and_import_data(data_dir):
    """import data and clean up the dataframe so that all relevant information is present and organised for downstream 
    processing. Splits the treatment column into two seperate columns, 'repeat' and 'channel'.

    Args:
        data_dir (dict): dictionary containing the key (i.e., sample name) and the value (i.e., the csv file containing
        raw data)

    Returns:
        dataframe: cleaned dataframe
    """
    data_col = []
    for treatment, directory in data_dir.items():
        df = pd.read_csv(f'{directory}', header=None)
        df.iloc[0] = df.iloc[0].fillna(method='ffill')
        df_transposed = df.transpose()
        test = df_transposed.melt(id_vars=[0, 1], var_name='column')
        test_sorted_df = test.sort_values(by=[0, 1, 'column'], ascending=[True, True, True])
        test_sorted_df.columns = ['treatment', 'time_or_counts', 'column', 'value']
        test_sorted_df['treatment'] = test_sorted_df['treatment'].str.replace(' ', '_')
        test_sorted_df['treatment'] = test_sorted_df['treatment'].str.replace('__', '_')
        
        # Splitting the column into two new columns
        test_sorted_df[['repeat', 'channel']] = test_sorted_df['treatment'].str.rsplit('_', 1, expand=True)
        # Removing the '_Count_Rate' part from the 'repeat' column
        test_sorted_df['repeat'] = test_sorted_df['repeat'].str.replace('_Count_Rate_Channel', '')
        test_sorted_df['value'] = test_sorted_df['value'].astype(float)
        test_sorted_df['protein'] = treatment
        data_col.append(test_sorted_df)
    data_col_df = pd.concat(data_col)
    data_col_df['channel'].value_counts()
    return data_col_df

def map_time_to_data(data_col_df):
    """creates a time column so that every row is linked to the acquisition time.

    Args:
        data_col_df (dataframe): dataframe to be processed, comes from 'cleanup_and_import_data' function.

    Returns:
        dataframe: dataframe with time column
    """
    collated_df = []
    for (repeat,channel), df in data_col_df.groupby(['repeat', 'channel']):
        result_dict = df.set_index('column')['value'].to_dict()
        df['time'] = df['column'].map(result_dict)
        df = df[df['time_or_counts']!='Time']
        collated_df.append(df)
    col_df = pd.concat(collated_df)
    col_df.dropna(inplace=True)
    col_df = col_df.reset_index()
    col_df.drop('index', axis=1, inplace=True)
    return col_df

def fit_noise_to_gaussian(col_df):
    """Gets the FCS data and fits the 'background' noise to a gaussian for thresholding purposes. Will find the 
    center of the gaussian and the full-width at half maximum (FWHM), which is then used to define a threshold
    of count values above the noise to identify as an oligomer.

    Args:
        col_df (dataframe): cleaned dataframe from 'map_time_to_data' function.

    Returns:
        dataframe: returns a dataframe containing the mean and FWHM for each channel from each treatment. 
    """
    results = []
    for (protein, channel), df in col_df.groupby(['protein', 'channel']):
        peak_1 = df['value']
        # Generate histogram to visualise
        bins = np.linspace(-20, 2000, 2000)
        counts = pd.DataFrame(pd.cut(peak_1.values, bins=bins, right=True, labels=bins[:-1]).value_counts()).reset_index().sort_index()
        sns.lineplot(data=counts, x='index', y=0)
        plt.xlim(-200, 2000)
        plt.title(f'{protein}_{channel}')
        plt.show()
        # -------------------------------------
        # create gauss model
        gauss_model = GaussianMixture(n_components=1, covariance_type='full',  reg_covar=1e-6)
        gauss_model.fit(peak_1.values.reshape(-1, 1))
        # Extract fitted parameters
        means = gauss_model.means_
        weights = gauss_model.weights_
        covars = gauss_model.covariances_
        # Calculate the standard deviation (sigma)
        std_devs = np.sqrt(covars)
        # Calculate FWHM [full width at half maximum]
        fwhm = 2 * np.sqrt(2 * np.log(2)) * std_devs
        # Store the results in a list
        for mean, width in zip(means, fwhm):
            results.append({
                'protein': protein,
                'channel': channel,
                'mean': mean[0],
                'half_width': width[0][0]  # Store FWHM instead of std_devs
            })
        # ----------------------------------------------------------------------
        # For each model, use fitted parameters to generate sample gaussian line data for plotting fit
        fit_vals = []
        for model in range(len(means)):
            vals = pd.DataFrame([np.linspace(0, 2000, 4000), weights[model]*stats.norm.pdf(np.linspace(0, 2000, 4000), means[model], np.sqrt(covars[model])).ravel()]).T
            vals['model'] = model
            # Optional normalisation to maxiximum height of the sigma
            vals['norm_val'] = vals[1] / vals[1].max()
            fit_vals.append(vals)
        fit_vals = pd.concat(fit_vals)
        fit_vals.columns = ['x_fit', 'y_fit', 'model', 'norm_val']
        fit_vals.reset_index(drop=True, inplace=True)
        # -------------------------------------------------------------------------
        # Visualise fit, overlayed on the original histogram
        # Note, palette is specific for number of components
        palette={0: '#FB8B24', 1: '#820263', 2: '#234E69', 3: '#1AB0B0', 4: '#D90368', 5: '#EA4746'}
        sns.lineplot(data=fit_vals, x='x_fit', y='y_fit', hue='model', palette=palette, linewidth=2)
        sns.distplot(peak_1, kde=False, norm_hist=True, color='black')
        plt.xlim(-200, 2000)
        plt.show()
    results_df = pd.DataFrame(results)
    return results_df

def peak_finding(col_df, results_df, width_multiplier, distance, prominence):
    """identify peaks above noise based on a defined threshold and plot to check.

    Args:
        col_df (dataframe): cleaned data
        results_df (dataframe): from 'fit_noise_to_gaussian'. Contains the center of each gaussian fit and the FWHM used to define threshold for each channel.
        width_multiplier (float): used to set threshold for peak finding. A higher number results in stricter threshold. 
        distance (float): minimum number of frames between peaks. If two peaks are less than distance, the largest peak is kept.
        prominence (float): used by peak_finding algorithm to select based on peak shape.

    Returns:
        dataframe: returns dataframe containing all identified peaks.
    """
    col_peaks_major = []
    for (protein, repeat, channel), df in col_df.groupby(['protein', 'repeat', 'channel']):
        # find peaks and set values to modify
        cell_peak_major, _ = find_peaks(df['value'], height=float(results_df[(results_df['protein']==protein)&(results_df['channel']==channel)]['mean'].values)+float(results_df[(results_df['protein']==protein)&(results_df['channel']==channel)]['half_width'].values*width_multiplier), distance=distance, prominence=prominence)
        if len(cell_peak_major)>0:
            peaks_list_major = pd.DataFrame([df.iloc[index][["value", 'time']] for index in cell_peak_major], index=cell_peak_major)
            peaks_list_major['repeat'] = repeat
            peaks_list_major['channel'] = channel
            peaks_list_major['protein'] = protein
            # plot the peak finding to assess accuracy
            plt.axhline(y=0, linestyle='--', color='grey')
            plt.axhline(y=float(results_df[(results_df['protein']==protein)&(results_df['channel']==channel)]['mean'].values)+float(results_df[(results_df['protein']==protein)&(results_df['channel']==channel)]['half_width'].values*width_multiplier), linestyle='--', color='darkorange')
            sns.lineplot(data=df, x='time', y='value', color='grey')
            sns.scatterplot(data=peaks_list_major, x='time', y='value', color='red')
            plt.title(f'{protein}_{repeat}_{channel}')
            plt.xlabel('Time (s)')
            plt.ylabel('Counts')
            plt.show()
            # collate the peak data
            col_peaks_major.append(peaks_list_major)
        # if no peaks are present, return NaN to retain molecule info for downstream processing
        else:
            empty_df = pd.DataFrame(columns=['value', 'time'])
            empty_df.loc[0] = [np.nan] * 2
            empty_df['repeat'] = repeat
            empty_df['channel'] = channel
            empty_df['protein'] = protein
            col_peaks_major.append(empty_df)
    peaks = pd.concat(col_peaks_major)
    # assign each peak a unique identifier based on metadata and time to match for coincidences
    peaks['unique_id'] = [f'{repeat}_{channel}_{protein}_{time}' for repeat, channel, protein, time in peaks[['repeat', 'channel', 'protein', 'time']].values]
    return peaks

def find_coincident_peaks(peaks, channel_1=1, channel_2=2,threshold=0.02):
    """Look through all peaks in one channel and identify if there is a peak in the other channel within a 
    defined time threshold; if there is, they are labelled as coincident peaks.

    Args:
        peaks (dataframe): dataframe containing all peaks identified in peak_finding function.
        channel_1 (int, optional): labels channel for coincidence. Defaults to 1.
        channel_2 (int, optional): labels the channel that channel 1 is compared to. Defaults to 2.
        threshold (float, optional): time threshold used to determine if two peaks occur at similar times to be coincident. Defaults to 0.02.

    Returns:
        dataframe: returns dataframe in which each peak in channel 1 is coincident (if so, a unique identifier
        of the matched peak from channel 2 is provided in the 'coincident_peak' column); if not, an empty list is
        present, which is converted to a 0.
    """
    coincident_list = []
    for (protein, repeat), df in peaks.groupby(['protein','repeat']):
        df1 = df[df['channel']==f'{channel_1}'].copy()
        df2 = df[df['channel']==f'{channel_2}'].copy()
        peak_list = []
        # for each peak in channel 1 at time X, look for peak in channel 2 at time X +- threshold 
        # if present, label as coincident peak.
        for value, time in df1[['value', 'time']].values:
            upper = time + threshold
            lower = time - threshold
            coincident_peaks = df2[(df2['time']<=upper)&(df2['time']>=lower)].copy()
            if len(coincident_peaks)<1:
                peak_list.append([])
            else:
                peak_list.append(coincident_peaks['unique_id'].to_list())
        df1['coincident_peak'] = peak_list
        coincident_list.append(df1)
    test_df = pd.concat(coincident_list)
    test_df['length'] = [len(row) for row in test_df['coincident_peak']]
    test_df['length'].value_counts()
    return test_df

def plot_and_determine_percent_coincidence(merged_df, col_df_corrected, save_loc):
    """will do two things. (1) will extract information regarding the peaks, including the number of peaks in each 
    channel and the proportion coincidence of peaks. (2) will plot the data of both channels, identifying all peaks
    eith those being non-coincident (blue) or coincident (orange) denoted. Will also collate this data from all
    treatments and export it in a form that is ideal for visualization scripts. 

    Args:
        merged_df (dataframe): from 'find_coincident_peaks' script.
        col_df_corrected (dataframe): used for plotting the corrected traces.
        save_loc (str): where to save plots and data.

    Returns:
        dataframe: returns a dataframe for plotting of peak data and coincidence.
    """
    df_peak_data = []
    for (protein, repeat), df in col_df_corrected.groupby(['protein','repeat']):
        if df['channel'].nunique()>1:
            # determine number of peaks in each channel and number of coincident
            sorted_df = merged_df[(merged_df['protein']==protein)&(merged_df['repeat']==repeat)].copy()
            sorted_df['value_2'] = abs(sorted_df['value_2'])*-1
            channel_1_peak = sorted_df['time_1'].nunique()
            channel_2_peak = sorted_df['time_2'].nunique()
            coincident = sorted_df['is_coincident'].sum()
            # ------------------- plot and save example traces ----------------------------
            sns.lineplot(data=df, x='time', y='value', color='grey', hue='channel', palette='Greys')
            sns.scatterplot(data=sorted_df,x='time_1', y='value_1', hue='is_coincident')
            sns.scatterplot(data=sorted_df,x='time_2', y='value_2', hue='is_coincident', legend=False)
            plt.title(f'{coincident}/{channel_1_peak}_{protein}_{repeat}')
            plt.ylabel('Counts')
            plt.xlabel('Time (s)')
            plot_export = f'{save_loc}/Coincidence_traces'
            if not os.path.exists(plot_export):
                os.makedirs(plot_export)
            plt.savefig(f'{plot_export}/{channel_1_peak}_{protein}_{repeat}.svg', dpi=600)
            plt.show()
            # ----------------- determine percent coincidence for both channels -------------------------
            # create columns with peak data and calculate percent coincidence
            df['#channel_1_peaks'] = channel_1_peak
            df['#channel_2_peaks'] = channel_2_peak
            df['prop_channel_1_peaks_are_coincident'] = (coincident / channel_1_peak) * 100 if channel_1_peak != 0 else np.nan
            df['prop_channel_2_peaks_are_coincident'] = (coincident / channel_2_peak) * 100 if channel_2_peak != 0 else np.nan
            df_peak_data.append(df)
        else:
            sorted_df =  merged_df[(merged_df['protein']==protein)&(merged_df['repeat']==repeat)]
            channel_1_peak = sorted_df['time_1'].nunique()
            df['#channel_1_peaks'] = channel_1_peak
            df_peak_data.append(df)
            continue
    df_peak_data = pd.concat(df_peak_data)
    df_peak_data.to_csv(f'{output_folder}/peak_data.csv')
    return df_peak_data

def FCS_analysis_master(output_folder, data_dir, gaus_width_multiplier=2, gaus_distance=40, gaus_prominence=10, coincidence_time_thresh=0.02):
    data_col_df = cleanup_and_import_data(data_dir)
    col_df = map_time_to_data(data_col_df)
    col_df.to_csv(f'{output_folder}/collated_data.csv')
    results_df = fit_noise_to_gaussian(col_df)
    results_df.to_csv(f'{output_folder}/gaussian_fits.csv')
    peaks = peak_finding(col_df, results_df, width_multiplier=gaus_width_multiplier, distance=gaus_distance, prominence=gaus_prominence)
    peaks.to_csv(f'{output_folder}/peaks.csv')

    test_df = find_coincident_peaks(peaks, channel_1=1, channel_2=2,threshold=coincidence_time_thresh)
    test_df2 = find_coincident_peaks(peaks, channel_1=2, channel_2=1,threshold=coincidence_time_thresh)
    test_df['coincident_peak'] = [item[0] if len(item) > 0 else np.nan for item in test_df['coincident_peak']]
    test_df2['coincident_peak'] = [item[0] if len(item) > 0 else np.nan for item in test_df2['coincident_peak']]
    test_df2.columns = ['value','time','repeat','channel','protein','coincident_peak','unique_id','length']


    ########
    ######## currently errors if there are no coincident peaks identified
    merged_df = pd.merge(test_df, test_df2, on=['unique_id', 'coincident_peak', 'protein', 'repeat'], how='outer', suffixes=('_1', '_2'))
    ########
    ########

    merged_df.drop(['length_1', 'length_2'], axis=1, inplace=True)
    # label each peak if coincident or not
    merged_df['is_coincident'] = [1 if ((type(val_1)==str)&(type(val_2)==str)) else 0 for val_1, val_2 in merged_df[['unique_id', 'coincident_peak']].values]
    merged_df['is_coincident'].value_counts()
    merged_df_uniqueid = merged_df.dropna(subset=['unique_id', 'coincident_peak'])
    # code to make the second channel negative for visualization
    mask = col_df['channel'] == '2'
    col_df_corrected = col_df
    col_df_corrected['value'] = col_df_corrected.apply(lambda row: row['value'] * -1 if row['channel'] == '2' else row['value'], axis=1)
    col_df_corrected.to_csv(f'{output_folder}/collated_data_corrected.csv')



    df_peak_data = plot_and_determine_percent_coincidence(merged_df, col_df_corrected, output_folder)
    return peaks, df_peak_data

# ------------------------------------------ call functions for processing -----------------------------------------

peaks, df_peak_data = FCS_analysis_master(output_folder, 
                            data_dir, 
                            gaus_width_multiplier=2, 
                            gaus_distance=40, 
                            gaus_prominence=10, 
                            coincidence_time_thresh=0.02)

