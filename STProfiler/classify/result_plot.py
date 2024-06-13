import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc
import seaborn as sns

def ROC_plot(tprs_list, ax, color_list, plot_minor_lines=True, show_AUC=True):
    tprs_num = len(tprs_list)
    
    tprs_len = tprs_list[0][0].shape[0]
    x = np.linspace(0,1,tprs_len)

    if(show_AUC):
        plot_list = []

    for i in range(tprs_num):
        tprs = tprs_list[i]

        if(plot_minor_lines):
            for tpr in tprs:
                ax.plot(x, tpr, color=color_list[i], alpha=0.1)

        if(show_AUC):
            ax.plot(x, np.mean(tprs, axis=0), color='k', lw=5)
            line_plot = ax.plot(x, np.mean(tprs, axis=0), color=color_list[i], lw=3,
                                label='AUC = ' + str(np.round(auc(x, np.mean(tprs, axis=0)), 2)))
            plot_list = plot_list + line_plot
        else:
            ax.plot(x, np.mean(tprs, axis=0), color='k', lw=5)
            ax.plot(x, np.mean(tprs, axis=0), color=color_list[i], lw=3)

    if(show_AUC):
        label_AUC = [p.get_label() for p in plot_list]
        ax.legend(handles=plot_list, labels=label_AUC, loc='lower right')

    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')

    return

def top_features_plot(features_df, 
                      feature_importance, 
                      top_num, 
                      case_name,
                      color_list=None, 
                      feature_title=None,
                      figsize=None):
    
    top_features = features_df.drop(columns=case_name).columns[np.argsort(feature_importance)][(-1*top_num):].to_list()
    top_features.reverse()

    if(feature_title==None):
        feature_title = top_features

    if(figsize==None):
        figsize=(top_num*5,5)

    fig, axs = plt.subplots(1,top_num, figsize=figsize)

    for i in range(len(top_features)):
        sns.violinplot(features_df, 
                       x=case_name, y=top_features[i],
                       ax=axs[i], palette=color_list)
        axs[i].set_xlabel('')
        axs[i].set_ylabel('')
        axs[i].set_title(top_features[i])

    return