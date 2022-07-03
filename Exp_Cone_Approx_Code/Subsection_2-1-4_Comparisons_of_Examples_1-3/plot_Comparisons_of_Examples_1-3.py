# Problem: Comparisons of the second-order taylor approximation and the proposed methods
# Problem type: Two SOC constraints
# Method: Example 1-Example 3

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot(x, y_log, y_taylor, y_ex1, y_ex2, y_ex3, figname):
    fig, ax = plt.subplots(dpi=300)
    plt.xscale('log')
    ax.set_xticks([0.6, 1, 2, 6])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.plot(x,y_log,color='black',linestyle='-',linewidth=2,label='$\log(x)$')
    plt.plot(x,y_taylor,color='red',linestyle='-.',linewidth=2,label='2nd Order Taylor Approx.')
    plt.plot(x,y_ex1,color='blue',linestyle='--',linewidth=2,label='Example 1 with $N=2$, $a=1$')
    plt.plot(x,y_ex2,color='m',dashes=[3, 1, 1, 1, 1, 1],linewidth=2,label='Example 2 with $N=1$, $s=1$')
    plt.plot(x,y_ex3,color='c',dashes=(1, 1),linewidth=2,label='Example 3 with $N=2$, $\hat{x}=2$')
    plt.xlabel('x')
    plt.legend()
    plt.savefig(figname, bbox_inches='tight')
    plt.show()

x = np.linspace(0.6, 6)
y_log = np.log(x)
y_taylor = np.log(2)+0.5*(x-2)-(x-2)*(x-2)/8  
y_ex1 = (x-1)/(2+(1/(3**0.5)+1)*(x-1))+(x-1)/(2+(-1/(3**0.5)+1)*(x-1)) 
y_ex2 = 4*(x**0.5-1)/(x**0.5+1)  
y_ex3 = np.log(2)+(x/2-1)/(2+(1/(3**0.5)+1)*(x/2-1))+(x/2-1)/(2+(-1/(3**0.5)+1)*(x/2-1)) 
figname = 'Comparisons_of_Examples_1-3'

plot(x, y_log, y_taylor, y_ex1, y_ex2, y_ex3, figname)

