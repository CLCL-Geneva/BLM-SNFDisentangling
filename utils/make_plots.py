'''
Created on Jan 11, 2023

@author: vivi
'''

import re

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import seaborn as sns

import numpy as np

results_dir = "/home/vivi/work/Projects/BLMs/results/"


def transform(group, x, y, z):
    n_x = len(set(group[x]))
    return np.reshape(np.array(group[z]), (-1, n_x)).T
    
    
def find_vals(results_list, col, val, cols):
   
    values = []
    for results in results_list:
        subset = results.loc[(results[col] == val)]
        values = []
        for c in cols:
            values.extend(list(subset[c]))
        
    return min(values)-0.05, max(values)+0.05
    
    
def get_info(filename):
    m = re.search(r"_train-[_\d\.]+_(.*?)\_\d+-\w+", filename)
    if m:
        return m.group(1)

    m = re.search(r"_test_(.*?)\_\d+-\w+", filename)
    if m:
        return m.group(1)
    
    return filename
    

def make_results_df(list_of_files, target_score, info):
    
    columns = ["model", "train_on", "test_on", target_score]
    results = pd.DataFrame(columns = columns)
        
    for f in list_of_files:
        model_name = get_info(f)
        
        df = pd.read_csv(results_dir + "/" + f, sep="\t")
        print("{} data frame columns: {}".format(model_name, df.columns))
        for train_on, train_group in df.groupby("train_on"):
            #print("\ntrain group {}:\n{}".format(train_on, train_group))
            for test_on, test_group in train_group.groupby("test_on"):
                #print("\ntest group {}:\n{}".format(test_on, test_group))
                scores = []
                for i, row in test_group.iterrows():
                    #print("row: {}".format(row))
                    scores.append(row[target_score])
                
                scores = np.array(scores)
                results = results.append({"model": model_name, "train_on": train_on, "test_on": test_on, target_score: float(np.mean(scores)), "sd": np.std(scores)}, ignore_index=True)
    
    print("averaged results:\n{}".format(results))
    results.to_csv(results_dir + "/averaged_" + info + ".tsv", sep="\t")
    
    return results
    

def make_little_plots(results, target_score):
    
    v_min = min(results[target_score])
    v_max = max(results[target_score])
    
    annot_kws={'fontsize':21}
    
    model_column = results.columns[0]
    models = set(results[model_column])  ## the model is the first column
    
    x_labels = list(set(results["train_on"]))
    y_labels = list(set(results["test_on"]))

    fig,axs = plt.subplots(ncols=len(models), figsize=(20,4))

    i = 0
    for model, group in results.groupby(model_column):
        res_2d = transform(group, "train_on", "test_on", target_score)
        cbar = (i > len(models)-1)
        yticklabels = y_labels if i < 1 else False

        sns.heatmap(res_2d, annot=True, xticklabels = x_labels, yticklabels = yticklabels, cbar=cbar, ax=axs[i], cmap="Blues", linewidth=0.5, vmin=v_min, vmax=v_max, annot_kws=annot_kws)
        axs[i].xaxis.tick_top() # x axis on top
        axs[i].xaxis.set_label_position('top')
        axs[i].set(title=model + " \n", xlabel=" train on \n")
        if i == 0:
            axs[i].set(ylabel="test on")
    
        i += 1

    plt.show()
    

def make_diffs_plots(results_list, labels_list, target_score):

    results = results_list[0]
    
    model_column = results.columns[0]
    models = list(results[model_column].unique())  ## the model is the first column
    print("Models: {}".format(models))
    ref_model = "Baseline-FFNN"  #models[0]  ## this should be the name of the baseline

    x_labels = results["train_on"].unique()
    y_labels = results["test_on"].unique()
        
    inc = 1/len(models)
    
    colors = plt.cm.BuPu(np.linspace(0, 1, len(models)))
            
    fig, axes = plt.subplots(nrows = len(y_labels), ncols = len(x_labels), sharex=True, sharey=True, figsize=(12,4))
    fig.subplots_adjust(top=0.82)
    
    #for ref_dt in data_types:
    for a_i in range(len(x_labels)):
        train_dt = x_labels[a_i]
        
        v_min, v_max = find_vals(results_list, "train_on", train_dt, [target_score])
        print("min max values for training on data type {}: ({}, {})".format(train_dt, v_min, v_max))
        
        for a_j in range(len(y_labels)):
            test_dt = y_labels[a_j]
                
            axes[a_i][a_j].set_title("train on {}, test on {}".format(train_dt, test_dt), fontsize=9)
            axes[a_i][a_j].set_ylim(v_min, v_max)
            axes[a_i][a_j].set_xticks(list(range(1,2*(len(labels_list)),2)))
            axes[a_i][a_j].set_xticklabels(labels_list)
            
            i = 1
            for n in range(len(results_list)):
                results_n = results_list[n]
                results_for_test = results_n.loc[(results_n["test_on"] == test_dt) & (results_n["train_on"] == train_dt)]
                ref_row = results_for_test.loc[results[model_column] == ref_model].iloc[0]
                
                print("\nSelection:\n{}".format(results_for_test))
                #print("reference row:\n{}\n".format(ref_row))
                 
                axes[a_i][a_j].plot([i-0.2,i+1.2],[ref_row[target_score], ref_row[target_score]], color='black')
                    
                x = i
                for ind, row in results_for_test.iterrows():
                    #print("plotting line {}: {}".format(ind, row))
                    #print("\tref type: {}\n\ttest type: {}".format(train_df, row["test on"]))
                    if row[model_column] != ref_model:
                        c = colors[models.index(row[model_column])]
                        axes[a_i][a_j].add_patch(patches.Rectangle((x,ref_row[target_score]), inc, row[target_score]-ref_row[target_score], color=c))
                        #print("added rectangle: {},{} + {},{}".format(x,row[target_score],inc,row[target_score]-ref_row[target_score]))
                    x += inc
                i += 2

    print("Colors: {}".format(colors))
    colors[0] = [0, 0, 0, 1]  ## the reference line is black
    color_patches = [patches.Patch(facecolor=c) for c in colors ]
    fig.legend(handles = color_patches, labels=models,
       loc="lower center",
       borderaxespad=0.1,
       ncol = len(models))

#    plt.suptitle("Model " + m, y=1)                
    plt.show()
    plt.savefig("system_analysis.png")



if __name__ == '__main__':
    
    
    '''
    target_score = "F1"
    #results = pd.read_csv("/home/vivi/work/Projects/BLMs/2D-ing_analysis_dualVAE.tsv", sep="\t")
    results = make_results_df(["results_X_test_Dual-VAE_24x32_12-Jan-2023_12:58.tsv",  
                       "results_X_test_Dual-VAE_48x16_12-Jan-2023_13:46.tsv", 
                       "results_X_test_Dual-VAE_16x48_12-Jan-2023_11:13.tsv",
                       "results_X_test_Dual-VAE_32x24_12-Jan-2023_12:05.tsv"], target_score, "2D-ing analysis")
    make_little_plots(results, target_score)
    '''
    
    target_score = "F1"
    results_sameTrain = make_results_df(
                             ["results_X_test_Dual-VAE_48x16_12-Jan-2023_13:46.tsv", 
                              "results_X_test_Baseline_CNN_48x16_12-Jan-2023_14:36.tsv",
                              "results_X_test_Baseline-FFNN_12-Jan-2023_14:11.tsv", 
                              "results_X_test_VAE_48x16_12-Jan-2023_15:29.tsv"], target_score, "sys_analysis_sameTrain")
      
    results_allTrain = make_results_df(
                             ["results_X_test_train-1.0_1.0_1.0_Dual-VAE_48x16_13-Jan-2023_07:30.tsv",
                              "results_X_test_train-1.0_1.0_1.0_Baseline_CNN_48x16_13-Jan-2023_00:44.tsv",
                              "results_X_test_train-1.0_1.0_1.0_Baseline-FFNN_12-Jan-2023_17:56.tsv",
                              "results_X_test_train-1.0_1.0_1.0_VAE_48x16_13-Jan-2023_03:29.tsv"], target_score, "sys_analysis_allTrain")

    
    make_diffs_plots([results_sameTrain, results_allTrain], ["sameTrain", "allTrain"], target_score)
    