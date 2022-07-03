# Problem: Packing
# Problem type: Binary
# Method: Example 1

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import output_file_names_packing_binary_small_scale

def read_data(title_Ex1):
    df_Ex1 = pd.read_csv(title_Ex1+'.csv')
    set_N = df_Ex1['kk'].unique().astype(int).tolist()
    set_a = df_Ex1['a'].unique().tolist()
    set_x = df_Ex1['p'].unique().astype(int).tolist()
    y_gap = {}
    y_time = {}
    for i in set_N:
        y_gap[i] = {}
        y_time[i] = {}
        for j in set_a:
            val_gap_lst = []
            val_time_lst = []
            for k in set_x:
                val_gap = df_Ex1.loc[(df_Ex1['kk']==i) & (df_Ex1['a']==j) & (df_Ex1['p']==k)]['Approx_Gap'].values[0]
                val_time = df_Ex1.loc[(df_Ex1['kk']==i) & (df_Ex1['a']==j) & (df_Ex1['p']==k)]['Time'].values[0]
                val_gap_lst.append(val_gap)
                val_time_lst.append(val_time)
            y_gap[i][j] = val_gap_lst
            y_time[i][j] = val_time_lst
    return set_N, set_a, set_x, y_gap, y_time

def plot_gap(N, x, y, y_color, y_linestyle, y_marker, y_label, figname):
    plt.figure(dpi=300)
    plt.xticks(x,x)
    plt.yscale('log')
    plt.ylim(top=0.01)  
    plt.ylim(bottom=1e-06) 
    for i in set_a:
        plt.plot(x,y[i],color=y_color[i],linestyle=y_linestyle[i],marker=y_marker[i],mfc='none',linewidth=1,markersize=6,label=y_label[i])
    plt.xlabel('p')
    plt.ylabel('Gap')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    plt.savefig(figname, bbox_inches='tight')
    #plt.show()

def plot_time(N, x, y, y_color, y_linestyle, y_marker, y_label, figname):
    plt.figure(dpi=300)
    plt.xticks(x,x)
    plt.ylim(top=2.5) 
    plt.ylim(bottom=0)
    for i in set_a:
        plt.plot(x,y[i],color=y_color[i],linestyle=y_linestyle[i],marker=y_marker[i],mfc='none',linewidth=1,markersize=6,label=y_label[i])
    plt.xlabel('p')
    plt.ylabel('Running Time (s)')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    plt.savefig(figname, bbox_inches='tight')
    #plt.show()

title_Ex1 = output_file_names_packing_binary_small_scale.output_name_Ex1
set_N, set_a, set_x, y_gap, y_time = read_data(title_Ex1)
y_color = ['c', 'm', 'b', 'g', 'r']
y_linestyle = ['-', ':', '--', '-', '-.']
y_marker = ['>', 's', 'o', '^', '*'] 
y_color = dict(zip(set_a, y_color))
y_linestyle = dict(zip(set_a, y_linestyle))
y_marker = dict(zip(set_a, y_marker))
y_label = []
for i in range(len(set_a)):
    val = 'a = ' + str(set_a[i])
    y_label.append(val)
y_label = dict(zip(set_a, y_label))

for i in set_N:
    figname = 'Binary_Packing_Ex1_Gap_N'+str(i)
    plot_gap(i, set_x, y_gap[i], y_color, y_linestyle, y_marker, y_label, figname)

for i in set_N:
    figname = 'Binary_Packing_Ex1_Time_N'+str(i)
    plot_time(i, set_x, y_time[i], y_color, y_linestyle, y_marker, y_label, figname)
    

