# Problem: Packing
# Problem type: Binary
# Method: Example 2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import output_file_names_packing_binary_small_scale

def read_data(title_Ex2):
    df_Ex2 = pd.read_csv(title_Ex2+'.csv')
    set_N = df_Ex2['kk'].unique().astype(int).tolist()
    set_s = df_Ex2['s'].unique().tolist()
    set_x = df_Ex2['p'].unique().astype(int).tolist()
    y_gap = {}
    y_time = {}
    for i in set_N:
        y_gap[i] = {}
        y_time[i] = {}
        for j in set_s:
            val_gap_lst = []
            val_time_lst = []
            for k in set_x:
                val_gap = df_Ex2.loc[(df_Ex2['kk']==i) & (df_Ex2['s']==j) & (df_Ex2['p']==k)]['Approx_Gap'].values[0]
                val_time = df_Ex2.loc[(df_Ex2['kk']==i) & (df_Ex2['s']==j) & (df_Ex2['p']==k)]['Time'].values[0]
                val_gap_lst.append(val_gap)
                val_time_lst.append(val_time)
            y_gap[i][j] = val_gap_lst
            y_time[i][j] = val_time_lst
    return set_N, set_s, set_x, y_gap, y_time

def plot_gap(N, x, y, y_color, y_linestyle, y_marker, y_label, figname):
    plt.figure(dpi=300)
    plt.xticks(x,x)
    plt.yscale('log')
    plt.ylim(top=1)
    plt.ylim(bottom=1e-06) 
    for i in set_s:
        plt.plot(x,y[i],color=y_color[i],linestyle=y_linestyle[i],marker=y_marker[i],mfc='none',linewidth=1,markersize=6,label=y_label[i])
    plt.xlabel('p')
    plt.ylabel('Gap')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    plt.savefig(figname, bbox_inches='tight')
    #plt.show()

def plot_time(N, x, y, y_color, y_linestyle, y_marker, y_label, figname):
    plt.figure(dpi=300)
    plt.xticks(x,x)
    plt.ylim(top=1) 
    plt.ylim(bottom=0)
    for i in set_s:
        plt.plot(x,y[i],color=y_color[i],linestyle=y_linestyle[i],marker=y_marker[i],mfc='none',linewidth=1,markersize=6,label=y_label[i])
    plt.xlabel('p')
    plt.ylabel('Running Time (s)')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    plt.savefig(figname, bbox_inches='tight')
    #plt.show()

title_Ex2 = output_file_names_packing_binary_small_scale.output_name_Ex2
set_N, set_s, set_x, y_gap, y_time = read_data(title_Ex2)
y_color = ['c', 'm', 'b', 'g', 'r']
y_linestyle = ['-', ':', '--', '-', '-.']
y_marker = ['>', 's', 'o', '^', '*'] 
y_color = dict(zip(set_s, y_color))
y_linestyle = dict(zip(set_s, y_linestyle))
y_marker = dict(zip(set_s, y_marker))
y_label = []
for i in range(len(set_s)):
    val = 's = ' + str(set_s[i])
    y_label.append(val)
y_label = dict(zip(set_s, y_label))

for i in set_N:
    figname = 'Binary_Packing_Ex2_Gap_N'+str(i)
    plot_gap(i, set_x, y_gap[i], y_color, y_linestyle, y_marker, y_label, figname)

for i in set_N:
    figname = 'Binary_Packing_Ex2_Time_N'+str(i)
    plot_time(i, set_x, y_time[i], y_color, y_linestyle, y_marker, y_label, figname)
    

