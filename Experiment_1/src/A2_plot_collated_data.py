import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os as os

# ------------------------------ combining datasets from multiple experiments --------------------

def condense_peak_data(peak_merge, plot_map):
    """generates a dataframe containing peak data for each repeat (i.e., number of peaks and proportion coincidence
    for each repeat).

    Args:
        peak_merge (dataframe): combined peak dataset from all biological repeats.
        plot_map (dict): dictionary containing key information used to classify proteins into relevant groups. This
        is needed in the situation that a particular protein is measured in channel 1 for a treatment and the same
        protein is measured as channel 2 for another treatment (essentially helps with manually grouping channel
        measurements into specific classes)/

    Returns:
        dataframe: returns a dataframe containing a row for each technical replicate, where the number of peaks and
        percent coincidence is provided.
    """
    rows_list = []
    # iterate over groups and collect data
    for (protein, repeat, experiment, channel), group_df in peak_merge.groupby(['protein', 'repeat', 'biological_repeat', 'channel']):
        row = {
        'protein': protein,
        'repeat': repeat,
        'experiment': experiment,
        'channel': channel,
        'peaks': group_df[f'#channel_{channel}_peaks'].iloc[0],
        'proportion_coincident':group_df[f'prop_channel_{channel}_peaks_are_coincident'].iloc[0],
    }
        rows_list.append(row)
    # Create a new DataFrame from the list of rows
    new_df = pd.DataFrame(rows_list)
    # uses plot_map to assign each row as belonging to a class (e.g., client or chaperone protein)
    new_df['new_value'] = [plot_map.get((row['protein'], row['channel']), None) for _, row in new_df.iterrows()]
    return new_df

def plot_violin(plot_export, new_df, label, y_axis, plot_order):
    if y_axis=='proportion_coincident':
        plot_order = [item for item in plot_order if item in new_df[new_df['new_value']==label].dropna(subset=[y_axis])['protein'].unique()]
    sns.violinplot(data=new_df[new_df['new_value']==label], y=y_axis, x='protein', scale='width', palette='Greys', order=plot_order)
    sns.stripplot(data=new_df[new_df['new_value']==label], y=y_axis, x='protein',  hue='experiment', palette='Oranges', alpha=0.8, size=2, order=plot_order)
    plt.xticks(rotation=45)
    plt.xlabel('')
    # plt.legend('')
    plt.ylabel(f'#{label} peaks per repeat')
    plt.title(f'{label}')
    plt.savefig(f'{plot_export}/{label}_{y_axis}.svg', dpi=600)
    plt.show()

def export_data(output_folder, new_df):
    """exports data in a form compatible with plotting in GraphPad

    Args:
        output_folder (str): where to save
        new_df (dataframe): re-organised data. 
    """
    datasets = ['peaks', 'proportion_coincident']
    for datatype in datasets:
        test = new_df.pivot_table(index=['repeat', 'experiment', 'new_value'], columns='protein', values=datatype)

    # Flatten the column hierarchy if needed
        test.columns = [col for col in test.columns]
    # Reset the index to turn 'repeat', 'experiment', and 'channel' back into columns
        test = test.reset_index()
        test_client = test[test['new_value']=='client']
        test_chap = test[test['new_value']=='chap']

        test_client.to_csv(f'{output_folder}/client_{datatype}.csv')
        test_chap.to_csv(f'{output_folder}/chap_{datatype}.csv')

def histogram_of_count_intensity(filt_peaks, plot_export):
    num_treatment = filt_peaks['protein'].nunique()
    fig, axes = plt.subplots(num_treatment, 1, sharex=True, sharey=True)
    for i, protein in enumerate(filt_peaks['protein'].unique()):
        sns.histplot(data=filt_peaks[filt_peaks['protein']==protein], 
                x='value',
                hue='protein', 
                stat='density', 
                common_norm=False, 
                #  element='step', 
                #  fill=False, 
                #  kde=True, 
                log_scale=False, 
                bins=20, 
                binrange=(0, 8000),
                ax=axes[i], 
            # 
                )
        # axes[i].legend(title='')
        plt.xlabel('Intensity')
    plt.savefig(f'{plot_export}/histogram_intensity_of_peaks.svg', dpi=600)
    plt.show()

def violin_plot_count_intensity(filt_peaks, plot_export, palette_violin='Greys', palette_strip='Oranges', log_scale=True):
    fig, ax = plt.subplots()
    sns.violinplot(data=filt_peaks, y='value', x='protein', scale='width', palette=palette_violin, log_scale=log_scale)
    sns.stripplot(data=filt_peaks, y='value', x='protein', palette=palette_strip, hue='biological_repeat', alpha=0.5, size=2)
    plt.xticks(rotation=45)
    plt.xlabel('')
    plt.ylabel('Counts')
    plt.tight_layout()
    plt.savefig(f'{plot_export}/violin_intensity_of_peaks.svg', dpi=600)
    plt.show()

def export_peaksize(output_folder, filt_peaks):
    test = filt_peaks.pivot_table(index=[filt_peaks.index, 'class'], columns='protein', values='value')
    # Flatten the column hierarchy if needed
    test.columns = [col for col in test.columns]
    # Reset the index to turn 'repeat', 'experiment', and 'channel' back into columns
    test = test.reset_index()
    test_client = test[test['class']=='client']
    test_chap = test[test['class']=='chap']

    test_client.to_csv(f'{output_folder}/client_peak_size.csv')
    test_chap.to_csv(f'{output_folder}/chap_peak_size.csv')



if __name__ == "__main__":

    output_folder = 'Experiment_1/python_results'
    plot_export = f'{output_folder}/figures/'
    if not os.path.exists(plot_export):
        os.makedirs(plot_export)

    # order in which to plot data
    chap_order = ['gfp', 'tdp_gfp', 'hsp27_3d', 'hsp27_acd', 'tdp_and_hsp27_3d', 'tdp_and_hsp27_acd','aBc_wt','aBc_acd', 'tdp_and_abc_wt','tdp_and_abc_acd']
    client_order = ['tdp_GdHCl', 'tdp_gfp', 'tdp_only', 'tdp_and_hsp27_3d', 'tdp_and_hsp27_acd', 'tdp_and_abc_wt','tdp_and_abc_acd']

    # dictionary for organizing data. Data is input as ('treatment_name', channel):'protein_group' (i.e., client or chaperone)
    plot_map = {('aBc_acd', 1):'chap',
                ('aBc_wt', 1):'chap',
                ('gfp', 1):'chap',
                ('hsp27_3d', 1):'chap',
                ('hsp27_acd', 1):'chap',
                ('tdp_GdHCl', 1):'client',
                ('aBc_acd', 1):'chap',

                ('tdp_and_abc_acd', 1):'client',
                ('tdp_and_abc_acd', 2):'chap',

                ('tdp_and_abc_wt', 1):'client',
                ('tdp_and_abc_wt', 2):'chap',

                ('tdp_and_hsp27_3d', 2):'chap',
                ('tdp_and_hsp27_3d', 1):'client',

                ('tdp_and_hsp27_acd', 1):'client',
                ('tdp_and_hsp27_acd', 2):'chap',

                ('tdp_gfp', 1):'chap',
                ('tdp_gfp', 2):'client',

                ('tdp_only', 1):'client',
                }


    # ------------------------- plot number of peaks and proportion coincidence ---------------------------------

    # import peak data from each repeat
    repeat_peak1 = pd.read_csv('Experiment_1/python_results/peak_data.csv')
    repeat_peak2 = pd.read_csv('Experiment_2/python_results/peak_data.csv')
    repeat_peak3 = pd.read_csv('Experiment_3/python_results/peak_data.csv')

    # add column to identify which data comes from a particular biological repeat
    repeat_peak1['biological_repeat'] = 1
    repeat_peak2['biological_repeat'] = 2
    repeat_peak3['biological_repeat'] = 3

    # combine datasets
    peak_merge = pd.concat([repeat_peak1, repeat_peak2, repeat_peak3], ignore_index=True)
    new_df = condense_peak_data(peak_merge, plot_map) 

    plot_violin(plot_export, new_df, label='client', y_axis='peaks', plot_order=client_order)
    plot_violin(plot_export, new_df, label='client', y_axis='proportion_coincident', plot_order=client_order)
    plot_violin(plot_export, new_df, label='chap', y_axis='peaks', plot_order=chap_order)
    plot_violin(plot_export, new_df, label='chap', y_axis='proportion_coincident', plot_order=chap_order)

    export_data(plot_export, new_df)


    # ---------------------------- plot intensity of peaks ----------------------------------------------------------

    peaks_repeat1 = pd.read_csv('Experiment_1/python_results/peaks.csv')
    peaks_repeat2 = pd.read_csv('Experiment_2/python_results/peaks.csv')
    peaks_repeat3 = pd.read_csv('Experiment_3/python_results/peaks.csv')

    peaks_repeat1['biological_repeat'] = 1
    peaks_repeat2['biological_repeat'] = 2
    peaks_repeat3['biological_repeat'] = 3

    peaks_for_count = pd.concat([peaks_repeat1, peaks_repeat2, peaks_repeat3], ignore_index=True)
    peaks_for_count['class'] = [plot_map.get((row['protein'], int(row['channel'])), None) for _, row in peaks_for_count.iterrows()]


    # filter for data that you want to keep or exlcude
    filt_peaks = peaks_for_count[peaks_for_count['class']=='client']
    filt_peaks = filt_peaks[~filt_peaks['protein'].str.contains('abc')]
    filt_peaks = filt_peaks[~filt_peaks['protein'].str.contains('Gd')]

    print(filt_peaks.groupby('protein')['value'].mean())
    print(filt_peaks.groupby('protein')['value'].median())
    print(filt_peaks.groupby('protein')['value'].sem())
    print(filt_peaks.groupby('protein')['value'].count())


    histogram_of_count_intensity(filt_peaks, plot_export)
    violin_plot_count_intensity(filt_peaks, plot_export, palette_violin='Greys', palette_strip='Oranges', log_scale=True)
    export_peaksize(output_folder, filt_peaks)


