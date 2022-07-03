# Problem: Packing
# Problem type: Binary
# Method: Example 3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import output_file_names_packing_binary_small_scale

def read_data(title_Ex3, title_Ex3_Best_Scale):
    df_Ex3_Best_Scale = pd.read_csv(title_Ex3_Best_Scale+'.csv')
    val_gap_bs = df_Ex3_Best_Scale['Approx_Gap'].values.tolist()
    val_time_bs = df_Ex3_Best_Scale['Time'].values.tolist()
    df_Ex3 = pd.read_csv(title_Ex3+'.csv')
    set_N = df_Ex3['kk'].unique().astype(int).tolist()
    set_scale = df_Ex3['scale'].unique().tolist()
    set_scale.insert(0,'bs')
    set_x = df_Ex3['p'].unique().astype(int).tolist()
    y_gap = {}
    y_time = {}
    for i in set_N:
        y_gap[i] = {}
        y_time[i] = {}
        y_gap[i][set_scale[0]] = val_gap_bs
        y_time[i][set_scale[0]] = val_time_bs
        for j in set_scale[1:]:
            val_gap_lst = []
            val_time_lst = []
            for k in set_x:
                val_gap = df_Ex3.loc[(df_Ex3['kk']==i) & (df_Ex3['scale']==j) & (df_Ex3['p']==k)]['Approx_Gap'].values[0]
                val_time = df_Ex3.loc[(df_Ex3['kk']==i) & (df_Ex3['scale']==j) & (df_Ex3['p']==k)]['Time'].values[0]
                val_gap_lst.append(val_gap)
                val_time_lst.append(val_time)
            y_gap[i][j] = val_gap_lst
            y_time[i][j] = val_time_lst
    return set_N, set_scale, set_x, y_gap, y_time

def plot_gap(N, x, y, y_color, y_linestyle, y_marker, y_label, figname):
    plt.figure(dpi=300)
    plt.xticks(x,x)
    plt.yscale('log')
    plt.ylim(top=10)
    plt.ylim(bottom=1e-09) 
    for i in set_scale:
        plt.plot(x,y[i],color=y_color[i],linestyle=y_linestyle[i],marker=y_marker[i],mfc='none',linewidth=1,markersize=6,label=y_label[i])
    plt.xlabel('p')
    plt.ylabel('Gap')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    plt.savefig(figname, bbox_inches='tight')
    #plt.show()

def plot_time(N, x, y, y_color, y_linestyle, y_marker, y_label, figname):
    plt.figure(dpi=300)
    plt.xticks(x,x)
    plt.ylim(top=2) 
    plt.ylim(bottom=0)
    for i in set_scale:
        plt.plot(x,y[i],color=y_color[i],linestyle=y_linestyle[i],marker=y_marker[i],mfc='none',linewidth=1,markersize=6,label=y_label[i])
    plt.xlabel('p')
    plt.ylabel('Running Time (s)')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    plt.savefig(figname, bbox_inches='tight')
    #plt.show()

title_Ex3 = output_file_names_packing_binary_small_scale.output_name_Ex3
title_Ex3_Best_Scale = output_file_names_packing_binary_small_scale.output_name_Ex3_Best_Scale
set_N, set_scale, set_x, y_gap, y_time = read_data(title_Ex3, title_Ex3_Best_Scale)
y_color = ['c', 'm', 'b', 'g', 'r']
y_linestyle = ['-', ':', '--', '-', '-.']
y_marker = ['>', 's', 'o', '^', '*'] 
y_color = dict(zip(set_scale, y_color))
y_linestyle = dict(zip(set_scale, y_linestyle))
y_marker = dict(zip(set_scale, y_marker))
y_label = ['Best Scale', '$\hat{x}$=$2^{-6}$', '$\hat{x}$=$2^{-4}$', '$\hat{x}$=$2^{-2}$', '$\hat{x}$=$2^{0}$']
y_label = dict(zip(set_scale, y_label))

for i in set_N:
    figname = 'Binary_Packing_Ex3_Gap_N'+str(i)
    plot_gap(i, set_x, y_gap[i], y_color, y_linestyle, y_marker, y_label, figname)

for i in set_N:
    figname = 'Binary_Packing_Ex3_Time_N'+str(i)
    plot_time(i, set_x, y_time[i], y_color, y_linestyle, y_marker, y_label, figname)

