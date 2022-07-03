# Problem: Comparisons of the proposed methods in Section 3
# Problem type: Two SOC constraints
# Method: Methods in Section 3

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot(x, y_exp, y_sec31, y_sec32, y_sec31_shift, y_sec32_shift, figname):
    fig, ax = plt.subplots(dpi=300)
    plt.xscale('log')
    ax.set_xticks([0.3, 1, 3])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.plot(x,y_exp,color='black',linestyle='-',linewidth=2,label='$\exp(x)$')
    plt.plot(x,y_sec31,color='g',dashes=(3, 1, 1, 1),linewidth=2,label='Section 3.1 with $N=2$')
    plt.plot(x,y_sec32,color='b',linestyle='--',linewidth=2,label='Section 3.2 with $N=1$, $s=1$')
    plt.plot(x,y_sec31_shift,color='c',dashes=(1, 1),linewidth=2,label='Section 3.1 Shift with $N=2$, $\hat{x}=1$')
    plt.plot(x,y_sec32_shift,color='m',dashes=[3, 1, 1, 1, 1, 1],linewidth=2,label='Section 3.2 Shift with $N=1$, $s=1$, $\hat{x}=1$')
    plt.xlabel('x')
    plt.legend()
    plt.savefig(figname, bbox_inches='tight')
    plt.show()

x = np.linspace(0.3, 3) 
y_exp=np.exp(x)
y_sec31=(1+x/4)**4  
y_sec32=(0.5+0.5*(x/2+1)**2)**2  
y_sec31_shift=np.exp(1)*(1+(x-1)/4)**4 
y_sec32_shift=np.exp(1)*(0.5+0.5*(1+(x-1)/2)**2)**2
figname='Comparisons_of_Section_3'

plot(x, y_exp, y_sec31, y_sec32, y_sec31_shift, y_sec32_shift, figname)
