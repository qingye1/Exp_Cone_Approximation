# Problem: Packing
# Problem type: Continuous
# Method: Mosek, Example 3, Cutting Plane

import pandas as pd
import matplotlib.pyplot as plt
import output_file_names_packing_continuous_large_scale

def read_data(title_Mosek,title_Ex3,title_CP):
    df_Mosek = pd.read_csv(title_Mosek+'.csv')
    df_Ex3 = pd.read_csv(title_Ex3+'.csv')
    df_CP = pd.read_csv(title_CP+'.csv')
    x = df_Mosek['n'].unique().tolist()
    y_Mosek = []
    y_Ex3 = []
    y_CP = []
    for i in x:
        val_Mosek = df_Mosek.loc[df_Mosek['n']==i]['Time'].mean()
        val_Ex3 = df_Ex3.loc[df_Ex3['n']==i]['Time'].mean()
        val_CP = df_CP.loc[df_CP['n']==i]['Time'].mean()    
        y_Mosek.append(val_Mosek)
        y_Ex3.append(val_Ex3)
        y_CP.append(val_CP)        
    return x, y_Mosek, y_Ex3, y_CP

def plot(x, y_Mosek, y_Ex3, y_CP, fig_name):
    plt.figure(dpi=300)
    plt.xticks(x,x)
    plt.ylim(top=30)  
    plt.ylim(bottom=0) 
    plt.plot(x,y_Mosek,color='b',linestyle='--',marker='o',mfc='none',linewidth=1,markersize=6,label='MOSEK')
    plt.plot(x,y_Ex3,color='g',linestyle='-',marker='^',mfc='none',linewidth=1,markersize=6,label='Example 3')
    plt.plot(x,y_CP,color='r',linestyle='-.',marker='s',mfc='none',linewidth=1,markersize=6,label='Cutting Plane')
    plt.xlabel('n')
    plt.ylabel('Running Time (s)')
    plt.legend()
    plt.savefig(fig_name, bbox_inches='tight')
    plt.show()

title_Mosek = output_file_names_packing_continuous_large_scale.output_name_Mosek
title_Ex3 = output_file_names_packing_continuous_large_scale.output_name_Ex3
title_CP = output_file_names_packing_continuous_large_scale.output_name_CP
x, y_Mosek, y_Ex3, y_CP = read_data(title_Mosek,title_Ex3,title_CP)

fig_name = 'packing_continuous_large'
plot(x, y_Mosek, y_Ex3, y_CP, fig_name)
