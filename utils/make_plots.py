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


def transform(group, x, y, z):
    n_x = len(set(group[x]))
    return np.reshape(np.array(group[z]), (-1, n_x))
    
    
def find_vals(results_list, col, val, cols):
   
    values = []
    for results in results_list:
        subset = results.loc[(results[col] == val)]
        for c in cols:
            values.extend(list(subset[c]))
            
    print("values for {} = {}:\n{}".format(col, val, values))
        
    return min(values)-0.05, max(values)+0.05
    
    
def get_info(filename):

    m = re.search(r"_train-(\d+)_.*", filename)
    if m:
        return m.group(1)

    '''
    m = re.search(r"_((Baseline|VAE|dual|Dual).*?)_\d+-Jan.*", filename)
    if m:
        return m.group(1)
    '''

    '''
    m = re.search(r"_train-[_\d\.]+(.*?)_\d+-\w+", filename)
    if m:
        return m.group(1)
    '''

    '''
    m = re.search(r"_test_(.*?)\_\d+-\w+", filename)
    if m:
        return m.group(1)
    '''
    
    return filename
    

def make_results_df(list_of_files, target_score, info, results_dir):
    
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


def make_error_results_df(list_of_files, targets, info, results_dir):
    
    columns = ["model", "train_on", "test_on"]
    columns.extend(targets)
    results = pd.DataFrame(columns = columns)
        
    for f in list_of_files:
        model_name = get_info(f)
        
        df = pd.read_csv(results_dir + "/" + f, sep="\t")
        print("{} data frame columns: {}".format(model_name, df.columns))
        
        for train_on, train_group in df.groupby("train_on"):
            #print("\ntrain group {}:\n{}".format(train_on, train_group))
            for test_on, test_group in train_group.groupby("test_on"):
                #print("\ntest group {}:\n{}".format(test_on, test_group))
                results_row = {"model": model_name, "train_on": train_on, "test_on": test_on}
            
                row = test_group.iloc[0]
                N = row["TP"] + row["FP"] + row["FN"] + row["TN"]
                print("N = {}".format(N))

                for t in targets:
                    scores = []
                    for i, row in test_group.iterrows():
                        #print("row: {}".format(row))
                        scores.append(row[t])
                
                    results_row[t] = sum(scores)/len(scores)/N
                    
                results = results.append(results_row, ignore_index=True)
    
    print("averaged results:\n{}".format(results))
    results.to_csv(results_dir + "/averaged_" + info + ".tsv", sep="\t")
    
    return results

    

def make_little_plots(results, target_score, results_dir):
    
    v_min = min(results[target_score])
    v_max = max(results[target_score])
    
    annot_kws={'fontsize':21}
    
    model_column = results.columns[0]
    models = sorted(list(set(results[model_column])))  ## the model is the first column
    #models = ['Baseline-FFNN', 'Baseline_CNN_48x16', 'VAE_48x16', 'Dual-VAE_48x16']
    
    y_labels = sorted(list(set(results["train_on"])))
    x_labels = sorted(list(set(results["test_on"])))

    fig,axs = plt.subplots(ncols=len(models), figsize=(20,4))

    i = 0
    #for model, group in results.groupby(model_column):
    for model in models:
        group = results.loc[results[model_column] == model]
        print("group for model {}:\n{}".format(model, group))
        
        res_2d = transform(group, "train_on", "test_on", target_score)
        cbar = (i > len(models)-1)
        yticklabels = y_labels if i < 1 else False

        sns.heatmap(res_2d, annot=True, xticklabels = x_labels, yticklabels = yticklabels, cbar=cbar, ax=axs[i], cmap="Blues", linewidth=0.5, vmin=v_min, vmax=v_max, annot_kws=annot_kws)
        axs[i].xaxis.tick_top() # x axis on top
        axs[i].xaxis.set_label_position('top')
        #axs[i].set(title=model + " \n", xlabel=" test on \n")
        axs[i].set(title=model, xlabel=" test on \n")
        if i == 0:
            axs[i].set(ylabel="train on")
    
        i += 1

    plt.show()
    


def make_little_plots_2d(results_list, train_type, target_score, results_dir):
    
    v_min = min(get_results(results_list, target_score))
    v_max = max(get_results(results_list, target_score))
    
    annot_kws={'fontsize':21}
    
    model_column = results_list[0].columns[0]
    models = list(set(results_list[0][model_column]))  ## the model is the first column
    #models = ['Baseline-FFNN', 'Baseline_CNN_48x16', 'VAE_48x16', 'Dual-VAE_48x16']
    
    y_labels = sorted(list(set(results_list[0]["train_on"])))
    x_labels = sorted(list(set(results_list[0]["test_on"])))

    fig,axs = plt.subplots(nrows=len(models), ncols=len(train_type), figsize=(10,8))

    i = 0
    #for model, group in results.groupby(model_column):
    #for model in models:
    for i in range(len(models)):
        model = models[i]
        for j in range(len(train_type)):
            group = results_list[j].loc[results_list[j][model_column] == model]
            print("group for model {} {}:\n{}".format(model, train_type[j], group))
            
            res_2d = transform(group, "train_on", "test_on", target_score)
            print("res 2d: {}".format(res_2d))
            
            cbar = (i > len(models)-1)
            yticklabels = y_labels if j == 0 else False
    
            sns.heatmap(res_2d, annot=True, xticklabels = x_labels, yticklabels = yticklabels, cbar=cbar, ax=axs[i][j], cmap="Blues", linewidth=0.5, vmin=v_min, vmax=v_max, annot_kws=annot_kws)
            axs[i][j].xaxis.tick_top() # x axis on top
            axs[i][j].xaxis.set_label_position('top')
            axs[i][j].set(title=get_short_name(model) + "_" + train_type[j] + " \n")
            if j == 0:
                axs[i][j].set(ylabel="train on")
            if i == 0:
                axs[i][j].set(xlabel=" test on \n")
        
    plt.show()



def make_diffs_plots(results_list, labels_list, target_score, results_dir):

    results = results_list[0]
    
    model_column = results.columns[0]
    models = list(results[model_column].unique())  ## the model is the first column
    print("Models: {}".format(models))
    ref_model = "Baseline-FFNN"  #models[0]  ## this should be the name of the baseline
    ## reorder models so the baseline is first
    #models.remove(ref_model)
    #models = ['Baseline-FFNN', 'Baseline_CNN_48x16', 'VAE_48x16', 'Dual-VAE_48x16']
    #models = ['Baseline-FFNN', 'Baseline_CNN_1DxSeq', 'Baseline_CNN_48x16', 'VAE_1DxSeq', 'VAE_48x16', 'dual_VAE_1DxSeq', 'Dual-VAE_48x16']
    models = ['Baseline-FFNN', 'Baseline_CNN_1DxSeq', 'VAE_1DxSeq', 'dual_VAE_1DxSeq', 'Baseline_CNN_48x16', 'VAE_48x16', 'Dual-VAE_48x16']    
    print("Models: {}".format(models))

    models_1d = ['Baseline_CNN_1DxSeq', 'VAE_1DxSeq', 'dual_VAE_1DxSeq']    
    models_2d = ['Baseline_CNN_48x16', 'VAE_48x16', 'Dual-VAE_48x16']    

    #colors = plt.cm.BuPu(np.linspace(0, 1, len(models)))
    colors = make_colors([models_1d, models_2d], ["type_III", "type_I"])
    
    x_labels = sorted(list(results["train_on"].unique()))
    y_labels = sorted(list(results["test_on"].unique()))
        
    inc = 1/(len(models)-1)
                
    fig, axes = plt.subplots(nrows = len(y_labels), ncols = len(x_labels), sharex=True, sharey=True, figsize=(12,4))
    fig.subplots_adjust(top=0.82)
        
    #for ref_dt in data_types:
    for a_i in range(len(x_labels)):
        train_dt = x_labels[a_i]
        
        v_min, v_max = find_vals(results_list, "train_on", train_dt, [target_score])
        print("min max values for training on data type {}: ({}, {})".format(train_dt, v_min, v_max))

        axes[a_i][0].set_ylabel("train on \n" + x_labels[a_i] + "\n\n" + target_score)
        
        for a_j in range(len(y_labels)):
            test_dt = y_labels[a_j]
                
            axes[a_i][a_j].set_ylim([v_min, v_max])
            axes[a_i][a_j].set_xticks(list(range(1,2*(len(labels_list)),2)))
            axes[a_i][a_j].set_xticklabels(labels_list)
            
            print("set y limits: {}, {}".format(v_min, v_max))
            
            i = 1
            for n in range(len(results_list)):
                results_n = results_list[n]
                results_for_test = results_n.loc[(results_n["test_on"] == test_dt) & (results_n["train_on"] == train_dt)]
                ref_row = results_for_test.loc[results_n[model_column] == ref_model].iloc[0]
                
                print("\nSelection:\n{}".format(results_for_test))
                #print("reference row:\n{}\n".format(ref_row))
                 
                axes[a_i][a_j].plot([i-0.2,i+1.2],[ref_row[target_score], ref_row[target_score]], color='black')
                axes[a_i][a_j].text(i-0.3, ref_row[target_score]+0.01, "{:.3f}".format(ref_row[target_score]))
                    
                for ind, row in results_for_test.iterrows():
                    #print("plotting line {}: {}".format(ind, row))
                    #print("\tref type: {}\n\ttest type: {}".format(train_df, row["test on"]))
                    x = i + inc * (models.index(row[model_column])-1)
                    if row[model_column] != ref_model:
                        c = colors[row[model_column]]
                        axes[a_i][a_j].add_patch(patches.Rectangle((x,ref_row[target_score]), inc, row[target_score]-ref_row[target_score], color=c))
                        #print("added rectangle: {},{} + {},{}".format(x,row[target_score],inc,row[target_score]-ref_row[target_score]))
                i += 2
                
    for a_j in range(len(y_labels)):
        axes[0][a_j].set_title("test on {}".format(y_labels[a_j]), fontsize=9)
                
  
    #labels_legend = list(colors.keys())
    labels_legend = []
    for i in range(len(models_1d)):
        labels_legend.extend([models_1d[i], models_2d[i]])
              
    color_patches = [patches.Patch(facecolor=colors[l]) for l in labels_legend]
    labels_legend.append(ref_model)
    color_patches.append(patches.Patch(facecolor=[0,0,0,1]))  # the baseline will appear last and will be black
    fig.legend(handles = color_patches, labels=labels_legend,
       loc="lower center",
       borderaxespad=0.1,
       ncol = int((len(models)+1)/2))
               
#    plt.suptitle("Model " + m, y=1)                
#    plt.show()
    plt.savefig(results_dir + "/system_analysis.png")



def make_error_plots(results, targets, results_dir):
    
    model_column = results.columns[0]
    models = list(results[model_column].unique())  ## the model is the first column
    print("Models: {}".format(models))
    ref_model = "Baseline-FFNN"  #models[0]  ## this should be the name of the baseline
    ## reorder models so the baseline is first
    #models.remove(ref_model)
    #models = ['Baseline-FFNN', 'Baseline_CNN_48x16', 'VAE_48x16', 'Dual-VAE_48x16']
    #models = ['Baseline-FFNN', 'Baseline_CNN_1DxSeq', 'Baseline_CNN_48x16', 'VAE_1DxSeq', 'VAE_48x16', 'dual_VAE_1DxSeq', 'Dual-VAE_48x16']
    models = ['Baseline-FFNN', 'Baseline_CNN_1DxSeq', 'VAE_1DxSeq', 'dual_VAE_1DxSeq', 'Baseline_CNN_48x16', 'VAE_48x16', 'Dual-VAE_48x16']    
    print("Models: {}".format(models))

    models_1d = ['Baseline_CNN_1DxSeq', 'VAE_1DxSeq', 'dual_VAE_1DxSeq']    
    models_2d = ['Baseline_CNN_48x16', 'VAE_48x16', 'Dual-VAE_48x16']    

    #colors = plt.cm.Reds(np.linspace(0, 1, len(models)))
    colors = make_colors([models_1d, models_2d], ["err I", "err II"])

    x_labels = sorted(list(results["train_on"].unique()))
    y_labels = sorted(list(results["test_on"].unique()))
        
    inc = 1/(len(models)-1)

    display_labels = ["AE", "N2 coord N3", "WNA", "WN1", "WN2"] ##make sure they reflect the order of the targets
                
    fig, axes = plt.subplots(nrows = len(y_labels), ncols = len(x_labels), sharex=True, sharey=True, figsize=(12,4))
    fig.subplots_adjust(top=0.82)
        
    #for ref_dt in data_types:
    for a_i in range(len(x_labels)):
        train_dt = x_labels[a_i]
        
        axes[a_i][0].set_ylabel("train on \n" + x_labels[a_i] + "\n\n err. perc.")
                
        for a_j in range(len(y_labels)):
            test_dt = y_labels[a_j]                
            #axes[a_i][a_j].set_ylim([v_min, v_max])
            axes[a_i][a_j].set_xticks(list(range(1,2*(len(targets)),2)))
            axes[a_i][a_j].set_xticklabels(display_labels, rotation=30)
            
            results_for_test = results.loc[(results["test_on"] == test_dt) & (results["train_on"] == train_dt)]
            ref_row = results_for_test.loc[results[model_column] == ref_model].iloc[0]
            
            print("\nSelection:\n{}".format(results_for_test))
            #print("reference row:\n{}\n".format(ref_row))

            i = 1             
            for target_score in targets:
                axes[a_i][a_j].plot([i-0.3,i+1.2],[ref_row[target_score], ref_row[target_score]], color='black')
                axes[a_i][a_j].text(i-0.2, ref_row[target_score]+0.002, "{:.3f}".format(ref_row[target_score]))
                    
                for ind, row in results_for_test.iterrows():
                    #print("plotting line {}: {}".format(ind, row))
                    #print("\tref type: {}\n\ttest type: {}".format(train_df, row["test on"]))
                    x = i + inc * (models.index(row[model_column])-1)
                    if row[model_column] != ref_model:
                        c = colors[row[model_column]]
                        axes[a_i][a_j].add_patch(patches.Rectangle((x,ref_row[target_score]), inc, row[target_score]-ref_row[target_score], color=c))
                        #print("added rectangle: {},{} + {},{}".format(x,row[target_score],inc,row[target_score]-ref_row[target_score]))
                i += 2

    for a_j in range(len(y_labels)):
        axes[0][a_j].set_title("test on {}".format(y_labels[a_j]), fontsize=9)

    #labels_legend = list(colors.keys())
    labels_legend = []
    for i in range(len(models_1d)):
        labels_legend.extend([models_1d[i], models_2d[i]])
              
    color_patches = [patches.Patch(facecolor=colors[l]) for l in labels_legend]
    labels_legend.append(ref_model)
    color_patches.append(patches.Patch(facecolor=[0,0,0,1]))  # the baseline will appear last and will be black
    fig.legend(handles = color_patches, labels=labels_legend,
       loc="lower center",
       borderaxespad=0.1,
       ncol = int((len(models)+1)/2))
        

    #plt.suptitle("Model " + m, y=1)                
    #plt.show()
    plt.savefig("error_analysis.png")
    
    
    
    
def make_train_data_plots(results, target_score, results_dir):
    
    train_labels = sorted(list(results["train_on"].unique()))
    test_labels = sorted(list(results["test_on"].unique()))
    Ns = sorted(list(map(int,list(results["model"].unique()))))
    
    labels = []
    color_patches = []

    for train_on in train_labels:
        colors = get_colors(train_on, len(test_labels))
        print("colors: {}".format(colors))
        for test_on in test_labels:
            color = colors[test_labels.index(test_on)+1]
            labels.append("train_on " + train_on + "/ test_on " + test_on)
            color_patches.append(patches.Patch(facecolor=color))
            scores = get_scores(train_on, test_on, Ns, results, target_score)
            plt.plot(Ns, scores, color=color, marker="o")
            
    plt.legend(handles = color_patches, labels=labels,       
               loc="lower right",
               borderaxespad=0.1,
               ncol = 1)

    plt.ylabel(target_score)
    plt.xlabel("training+validation data (80:20)")
    #plt.show()
    plt.savefig(results_dir + "/train_data_analysis.png")


    
    
    
def make_train_data_plots__(results, target_score, results_dir):
    
    train_labels = sorted(list(results["train_on"].unique()))
    test_labels = sorted(list(results["test_on"].unique()))
    Ns = sorted(list(map(int,list(results["model"].unique()))))
    
    labels = []
    color_patches = []

    for test_on in test_labels:
        colors = get_colors(test_on, len(train_labels))
        print("colors: {}".format(colors))
        for train_on in train_labels:
            color = colors[train_labels.index(train_on)+1]
            labels.append("train_on " + train_on + "/ test_on " + test_on)
            color_patches.append(patches.Patch(facecolor=color))
            scores = get_scores(train_on, test_on, Ns, results, target_score)
            plt.plot(Ns, scores, color=color, marker="o")
            
    plt.legend(handles = color_patches, labels=labels,       
               loc="lower right",
               borderaxespad=0.1,
               ncol = 1)

    plt.ylabel(target_score)
    plt.xlabel("training+validation data (80:20)")
    #plt.show()
    plt.savefig(results_dir + "/train_data_analysis.png")



def get_results(results_list, target_score):
    results = []
    for x in results_list:
        results.extend(list(x[target_score]))

    return results


def get_scores(train_on, test_on, Ns, results, target_score):
    
    scores = []
    sel_results = results[(results['train_on'] == train_on) & (results["test_on"] == test_on)]
    print("selected results: \n{}".format(sel_results))
    for N in Ns:
        scores.append(sel_results[sel_results["model"] == str(N)].iloc[0][target_score])        

    return scores


def get_short_name(model_name):
    if "CNN" in model_name:
        return "CNN"
    if "FFNN" in model_name:
        return "FFNN"
    return model_name



def get_colors(val, N):
    if val == "type_I":
        return plt.cm.Purples(np.linspace(0, 1, N+1))
                
    if val == "type_II":
        return plt.cm.Blues(np.linspace(0, 1, N+1))
    
    if val == "err I":
        return plt.cm.Oranges(np.linspace(0, 1, N+1))

    if val == "err II":
        return plt.cm.Blues(np.linspace(0, 1, N+1))

    return plt.cm.Greens(np.linspace(0, 1, N+1))    



def make_colors(m_lists, col_types):
    colors = {}
    for i in range(len(m_lists)):
        colors.update(dict(zip(m_lists[i], get_colors(col_types[i], len(m_lists[i]))[1:])))
     
    return colors   





if __name__ == '__main__':

    results_dir = "/home/vivi/work/Projects/BLMs/results/agreement/"

      
    '''
    target_score = "F1"
    results = make_results_df(["results_X_test_Dual-VAE_24x32_12-Jan-2023_12:58.tsv",  
                       "results_X_test_Dual-VAE_48x16_12-Jan-2023_13:46.tsv", 
                       "results_X_test_Dual-VAE_16x48_12-Jan-2023_11:13.tsv",
                       "results_X_test_Dual-VAE_32x24_12-Jan-2023_12:05.tsv"], target_score, "2D-ing analysis")
    make_little_plots(results, target_score, results_dir)
    '''
    
    '''    
    target_score = "F1"
    results_sameTrain = make_results_df(
                             ["results_X_test_Baseline-FFNN_12-Jan-2023_14:11.tsv",
                              "results_X_test_Baseline_CNN_48x16_12-Jan-2023_14:36.tsv",
                              "results_X_test_VAE_48x16_12-Jan-2023_15:29.tsv", 
                              "results_X_test_Dual-VAE_48x16_12-Jan-2023_13:46.tsv"], 
                             target_score, "sys_2D-ed_sameTrain")      
    make_little_plots(results_sameTrain, target_score, results_dir)
    '''


    '''
    target_score = "F1"
    results_baselines_sameTrain = make_results_df(
                             ["results_X_test_Baseline-FFNN_12-Jan-2023_14:11.tsv",
                              "results_X_test_train-1.0_2073_2073_Baseline_CNN_1DxSeq_19-Jan-2023_02:29.tsv"], 
                             target_score, "baselines")      
    results_baselines_allTrain = make_results_df(
                             ["results_X_test_train-1.0_1.0_1.0_Baseline-FFNN_12-Jan-2023_17:56.tsv",
                              "results_X_test_train-1.0_1.0_1.0_Baseline_CNN_1DxSeq_19-Jan-2023_05:02.tsv"], 
                             target_score, "baselines")      
    make_little_plots_2d([results_baselines_allTrain, results_baselines_sameTrain], ["allTrain", "sameTrain"], target_score, results_dir)
    '''

    
    '''
    target_scores = ['agreement_error', 'coordination', 'mis_num', 'N1_alter', 'N2_alter']
    results_baselines_sameTrain = make_error_results_df(
                             ["results_X_test_Baseline-FFNN_12-Jan-2023_14:11.tsv",
                              "results_X_test_train-1.0_2073_2073_Baseline_CNN_1DxSeq_19-Jan-2023_02:29.tsv"], 
                             target_scores, "baselines_errors")      
    
    
    make_error_plots(results_baselines_sameTrain, target_scores, results_dir)
    '''

    
    '''
    target_score = "F1"
    results_sameTrain = make_results_df(
                             ["results_X_test_Baseline-FFNN_12-Jan-2023_14:11.tsv",
                              "results_X_test_train-1.0_2073_2073_Baseline_CNN_1DxSeq_19-Jan-2023_02:29.tsv",
                              "results_X_test_Baseline_CNN_48x16_12-Jan-2023_14:36.tsv",
                              "results_X_test_train-1.0_2073_2073_VAE_1DxSeq_18-Jan-2023_21:03.tsv",
                              "results_X_test_VAE_48x16_12-Jan-2023_15:29.tsv", 
                              "results_X_test_train-1.0_2073_2073_dual_VAE_1DxSeq_19-Jan-2023_05:50.tsv",
                              "results_X_test_Dual-VAE_48x16_12-Jan-2023_13:46.tsv"], 
                             target_score, "sys_analysis_sameTrain")      
    results_allTrain = make_results_df(
                             ["results_X_test_train-1.0_1.0_1.0_Baseline-FFNN_12-Jan-2023_17:56.tsv",
                              "results_X_test_train-1.0_1.0_1.0_Baseline_CNN_1DxSeq_19-Jan-2023_05:02.tsv",
                              "results_X_test_train-1.0_1.0_1.0_Baseline_CNN_48x16_13-Jan-2023_00:44.tsv",
                              "results_X_test_train-1.0_1.0_1.0_VAE_1DxSeq_19-Jan-2023_02:07.tsv",
                              "results_X_test_train-1.0_1.0_1.0_VAE_48x16_13-Jan-2023_03:29.tsv",
                              "results_X_test_train-1.0_1.0_1.0_dual_VAE_1DxSeq_19-Jan-2023_08:57.tsv",
                              "results_X_test_train-1.0_1.0_1.0_Dual-VAE_48x16_13-Jan-2023_07:30.tsv"], 
                             target_score, "sys_analysis_allTrain")
    
    make_diffs_plots([results_sameTrain, results_allTrain], ["sameTrain", "allTrain"], target_score, results_dir)
    '''
    

    '''    
    target_score = "F1"
    results_list = ["results_X_test_train-50_50_50_Dual-VAE_48x16_17-Jan-2023_13:30.tsv", 
                    "results_X_test_train-100_100_100_Dual-VAE_48x16_17-Jan-2023_13:43.tsv",
                    "results_X_test_train-200_200_200_Dual-VAE_48x16_17-Jan-2023_13:56.tsv",
                    "results_X_test_train-500_500_500_Dual-VAE_48x16_17-Jan-2023_14:15.tsv",
                    "results_X_test_train-1000_1000_1000_Dual-VAE_48x16_17-Jan-2023_14:42.tsv",
                    "results_X_test_train-1500_1500_1500_Dual-VAE_48x16_17-Jan-2023_15:19.tsv",
                    "results_X_test_train-2073_2073_2073_Dual-VAE_48x16_17-Jan-2023_13:30.tsv"]

    results = make_results_df(results_list, target_score, "train_data_analysis_sameTrain")
    make_train_data_plots(results, target_score, results_dir)
    '''

    '''
    results_list = [
        "results_X_test_train-50_50_50_Baseline-FFNN_08-Feb-2023_16:49.tsv",
        "results_X_test_train-100_100_100_Baseline-FFNN_08-Feb-2023_16:58.tsv",
        "results_X_test_train-200_200_200_Baseline-FFNN_08-Feb-2023_17:08.tsv",
        "results_X_test_train-500_500_500_Baseline-FFNN_08-Feb-2023_17:21.tsv",
        "results_X_test_train-1000_1000_1000_Baseline-FFNN_08-Feb-2023_17:38.tsv",
        "results_X_test_train-1500_1500_1500_Baseline-FFNN_08-Feb-2023_17:55.tsv",
        "results_X_test_train-2000_2000_2000_Baseline-FFNN_08-Feb-2023_18:15.tsv"
        ]
    '''

    '''
    results_list = [
        "results_X_test_train-50_50_50_Dual-VAE_48x16_23-Apr-2023_13:05.tsv",
        "results_X_test_train-100_100_100_Dual-VAE_48x16_23-Apr-2023_13:13.tsv",
        "results_X_test_train-200_200_200_Dual-VAE_48x16_23-Apr-2023_13:23.tsv",
        "results_X_test_train-500_500_500_Dual-VAE_48x16_23-Apr-2023_13:38.tsv",
        "results_X_test_train-1000_1000_1000_Dual-VAE_48x16_23-Apr-2023_14:02.tsv",
        "results_X_test_train-1500_1500_1500_Dual-VAE_48x16_23-Apr-2023_14:34.tsv",
        "results_X_test_train-2000_2000_2000_Dual-VAE_48x16_23-Apr-2023_15:14.tsv",
        "results_X_test_train-2500_2500_2500_Dual-VAE_48x16_23-Apr-2023_16:04.tsv",
        "results_X_test_train-3000_3000_3000_Dual-VAE_48x16_23-Apr-2023_17:02.tsv",
        "results_X_test_train-3375_3375_3375_Dual-VAE_48x16_23-Apr-2023_11:35.tsv"
        ]
    '''
    
    
    
    '''
    results_list_dual_VAE = [
        "results_X_test_train-2000_2000_2000_Dual-VAE_32x24_joint-sampler__latent-cont-size_5__latent-disc-size_4x2_15-May-2023_02:45.tsv",
        "results_X_test_train-1500_1500_1500_Dual-VAE_32x24_joint-sampler__latent-cont-size_5__latent-disc-size_4x2_15-May-2023_01:47.tsv",
        "results_X_test_train-1000_1000_1000_Dual-VAE_32x24_joint-sampler__latent-cont-size_5__latent-disc-size_4x2_15-May-2023_01:02.tsv",
        "results_X_test_train-500_500_500_Dual-VAE_32x24_joint-sampler__latent-cont-size_5__latent-disc-size_4x2_14-May-2023_23:47.tsv",
        "results_X_test_train-200_200_200_Dual-VAE_32x24_joint-sampler__latent-cont-size_5__latent-disc-size_4x2_14-May-2023_23:26.tsv",
        "results_X_test_train-100_100_100_Dual-VAE_32x24_joint-sampler__latent-cont-size_5__latent-disc-size_4x2_14-May-2023_23:12.tsv",
        "results_X_test_train-50_50_50_Dual-VAE_32x24_joint-sampler__latent-cont-size_5__latent-disc-size_4x2_14-May-2023_23:01.tsv"
        ]

    results_list_VAE = [
        "results_X_test_train-2000_2000_2000_VAE_32x24_joint-sampler__latent-cont-size_5__latent-disc-size_4x2_14-May-2023_20:07.tsv",
        "results_X_test_train-1500_1500_1500_VAE_32x24_joint-sampler__latent-cont-size_5__latent-disc-size_4x2_14-May-2023_19:15.tsv",
        "results_X_test_train-1000_1000_1000_VAE_32x24_joint-sampler__latent-cont-size_5__latent-disc-size_4x2_14-May-2023_18:34.tsv",
        "results_X_test_train-500_500_500_VAE_32x24_joint-sampler__latent-cont-size_5__latent-disc-size_4x2_14-May-2023_18:04.tsv",
        "results_X_test_train-200_200_200_VAE_32x24_joint-sampler__latent-cont-size_5__latent-disc-size_4x2_14-May-2023_17:45.tsv",
        "results_X_test_train-100_100_100_VAE_32x24_joint-sampler__latent-cont-size_5__latent-disc-size_4x2_14-May-2023_17:32.tsv",
        "results_X_test_train-50_50_50_VAE_32x24_joint-sampler__latent-cont-size_5__latent-disc-size_4x2_14-May-2023_17:21.tsv"
        ]

        
    target_score = "F1"
    results = make_results_df(results_list_dual_VAE, target_score, "train_data_analysis_sameTrain_dual_VAE")
    make_train_data_plots(results, target_score, results_dir)

    results = make_results_df(results_list_VAE, target_score, "train_data_analysis_sameTrain_VAE")
    make_train_data_plots(results, target_score, results_dir)

    '''


    '''
    target_scores = ['agreement_error', 'coordination', 'mis_num', 'N1_alter', 'N2_alter']
    results_sameTrain = make_error_results_df(
                             ["results_X_test_Baseline-FFNN_12-Jan-2023_14:11.tsv",
                              "results_X_test_train-1.0_2073_2073_Baseline_CNN_1DxSeq_19-Jan-2023_02:29.tsv",
                              "results_X_test_Baseline_CNN_48x16_12-Jan-2023_14:36.tsv",
                              "results_X_test_train-1.0_2073_2073_VAE_1DxSeq_18-Jan-2023_21:03.tsv",
                              "results_X_test_VAE_48x16_12-Jan-2023_15:29.tsv", 
                              "results_X_test_train-1.0_2073_2073_dual_VAE_1DxSeq_19-Jan-2023_05:50.tsv",
                              "results_X_test_Dual-VAE_48x16_12-Jan-2023_13:46.tsv"], target_scores, "error_analysis_sameTrain")
    results_allTrain = make_error_results_df(
                             ["results_X_test_train-1.0_1.0_1.0_Baseline-FFNN_12-Jan-2023_17:56.tsv",
                              "results_X_test_train-1.0_1.0_1.0_Baseline_CNN_1DxSeq_19-Jan-2023_05:02.tsv",
                              "results_X_test_train-1.0_1.0_1.0_Baseline_CNN_48x16_13-Jan-2023_00:44.tsv",
                              "results_X_test_train-1.0_1.0_1.0_VAE_1DxSeq_19-Jan-2023_02:07.tsv",
                              "results_X_test_train-1.0_1.0_1.0_VAE_48x16_13-Jan-2023_03:29.tsv",
                              "results_X_test_train-1.0_1.0_1.0_dual_VAE_1DxSeq_19-Jan-2023_08:57.tsv",
                              "results_X_test_train-1.0_1.0_1.0_Dual-VAE_48x16_13-Jan-2023_07:30.tsv"], target_scores, "error_analysis_allTrain")

    make_error_plots(results_sameTrain, target_scores, results_dir)
    '''
    
    
