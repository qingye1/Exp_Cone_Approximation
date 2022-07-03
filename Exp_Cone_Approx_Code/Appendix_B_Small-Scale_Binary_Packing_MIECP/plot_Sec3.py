# Problem: Packing
# Problem type: Binary
# Method: Section 3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import output_file_names_packing_binary_small_scale

def read_data(title_Sec31, title_Sec31_Shift, title_Sec32, title_Sec32_Shift):
    Sec3 = {}
    set_method = ['Sec31', 'Sec32', 'Sec31_Shift', 'Sec32_Shift']
    Sec3[set_method[0]] = pd.read_csv(title_Sec31+'.csv')
    Sec3[set_method[1]] = pd.read_csv(title_Sec32+'.csv')
    Sec3[set_method[2]] = pd.read_csv(title_Sec31_Shift+'.csv')
    Sec3[set_method[3]] = pd.read_csv(title_Sec32_Shift+'.csv')
    set_N = Sec3[set_method[0]]['N'].unique().astype(int).tolist()    
    set_x = Sec3[set_method[0]]['p'].unique().astype(int).tolist()
    y_gap = {}
    y_time = {}
    for i in set_N:
        y_gap[i] = {}
        y_time[i] = {}
        for j in set_method:
            val_gap_lst = []
            val_time_lst = []
            for k in set_x:
                df_method = Sec3[j]
                val_gap = df_method.loc[(df_method['N']==i) & (df_method['p']==k)]['Approx_Gap'].values[0]
                if j == 'Sec31' or j == 'Sec32':
                    val_time = df_method.loc[(df_method['N']==i) & (df_method['p']==k)]['Time'].values[0]
                else:
                    val_time = df_method.loc[(df_method['N']==i) & (df_method['p']==k)]['total Time'].values[0]
                val_gap_lst.append(val_gap)
                val_time_lst.append(val_time)
            y_gap[i][j] = val_gap_lst
            y_time[i][j] = val_time_lst
    return set_N, set_method, set_x, y_gap, y_time

def plot_gap(N, x, y, y_color, y_linestyle, y_marker, y_label, figname):
    plt.figure(dpi=300)
    plt.xticks(x,x)
    plt.yscale('log')
    plt.ylim(top=100)
    plt.ylim(bottom=1e-16) 
    for i in set_method:
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
    for i in set_method:
        plt.plot(x,y[i],color=y_color[i],linestyle=y_linestyle[i],marker=y_marker[i],mfc='none',linewidth=1,markersize=6,label=y_label[i])
    plt.xlabel('p')
    plt.ylabel('Running Time (s)')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    plt.savefig(figname, bbox_inches='tight')
    #plt.show()

title_Sec31 = output_file_names_packing_binary_small_scale.output_name_Sec31
title_Sec31_Shift = output_file_names_packing_binary_small_scale.output_name_Sec31_Shift
title_Sec32 = output_file_names_packing_binary_small_scale.output_name_Sec32
title_Sec32_Shift = output_file_names_packing_binary_small_scale.output_name_Sec32_Shift
set_N, set_method, set_x, y_gap, y_time = read_data(title_Sec31, title_Sec31_Shift, title_Sec32, title_Sec32_Shift)
y_color = ['m', 'b', 'g', 'r']
y_linestyle = [':', '--', '-', '-.']
y_marker = ['s', 'o', '^', '>'] 
y_color = dict(zip(set_method, y_color))
y_linestyle = dict(zip(set_method, y_linestyle))
y_marker = dict(zip(set_method, y_marker))
y_label = ['Section 3.1', 'Section 3.2', 'Section 3.1 Shift', 'Section 3.2 Shift']
y_label = dict(zip(set_method, y_label))

for i in set_N:
    figname = 'Binary_Packing_Sec3_Gap_N'+str(i)
    plot_gap(i, set_x, y_gap[i], y_color, y_linestyle, y_marker, y_label, figname)

for i in set_N:
    figname = 'Binary_Packing_Sec3_Time_N'+str(i)
    plot_time(i, set_x, y_time[i], y_color, y_linestyle, y_marker, y_label, figname)

