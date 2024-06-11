import matplotlib.pyplot as plt
import numpy as np
name_list = ['acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh', 'cbrt', 'ceil', 'copysign', 'cos', 'cosh', 'cospi', 'cyl_bessel_i1', 'erf', 'erfc', 'erfcinv', 'erfcx', 'erfinv', 'exp', 'exp10', 'exp2', 'expm1', 'fabs', 'fdim', 'floor', 'fmax', 'fmin', 'fmod', 'hypot', 'j0', 'j1', 'lgamma', 'log', 'log10', 'log1p', 'log2', 'logb', 'max', 'min', 'nearbyint', 'nextafter', 'normcdf', 'normcdfinv', 'rcbrt', 'remainder', 'rhypot', 'rint', 'round', 'rsqrt', 'sin', 'sinpi', 'tan', 'tanh', 'tgamma', 'trunc', 'y0', 'y1', 'lulesh', 'cfd', 'backprop', 'leukocyte', 'randlc', 'sp', 'lammps', 'sw4lite', 'examinimd', 'hpccg', 'minife']

def plot_many_boxes(data,name,yname):
    # 创建一些随机数据
    #data = [np.random.normal(0, std, 100) for std in range(1,70)]

    # 创建一个箱线图
    #fig, ax = plt.subplots()
    fig = plt.figure(figsize=(19, 8))
    ax = plt.subplot(111)
    ax.boxplot(data)

    # 设置图表标题和轴标签
    # ax.set_title('Multiple Boxplots')
    # ax.set_xlabel('')
    ax.set_ylabel(yname,fontsize=20)

    # 设置x轴刻度标签
    #xticklabels = [f'Group {i}' for i in range(1, len(data)+1)]
    xticklabels = name_list 
    ax.set_xticklabels(xticklabels, rotation=90,fontsize=16)
    #plt.legend(prop={'size': 23})
    plt.tight_layout()
    plt.grid(zorder=1)
    plt.savefig(name, format="pdf")
    plt.close()
#plot_many_boxes()

def plot_bar_compare(y1,y2,new_name_lst):
    fig = plt.figure(figsize=(19, 8))
    ax = plt.subplot(111)
    index = np.arange(0,len(y1),1)
    # index = range(0,n_groups,2)
    # print(index)
    bar_width = 0.35
    opacity = 0.8

    # plt.boxplot(rpt1_lst)
    # plt.boxplot(rpt2_lst)
    plt.xlim(-0.8, len(y1)) 

    ax.bar(index, y1, bar_width,edgecolor='purple', color='None',hatch="/",label='Xscope')

    ax.bar(index + bar_width, y2, bar_width, label='DFEG')
    #add_value_labels(ax)
    # plt.xlabel('Functions',fontsize=20)
    # plt.ylabel('Repair time ratios',fontsize=20)
    plt.ylabel('Log2 of floating-point exceptions per second',fontsize=24)
    # plt.title('Scores by person')
    # plt.yticks(index+0.5*bar_width, id_lst,rotation=0)
    plt.xticks(index+0.5*bar_width, new_name_lst,rotation=90,fontsize=20)
    # plt.xticks(range(0,int(np.max(rpt1_lst))+1,2), range(0,int(np.max(rpt1_lst))*10+10,20),rotation=30)
    #plt.yticks(list(range(0,int(np.max(y1))+10,20)), list(range(0,int(np.max(rpt1_lst))*10+10,20)),fontsize=16)
    #plt.yticks(x,fontsize=16)
    plt.legend(prop={'size': 19.5})
    plt.tight_layout()
    plt.grid(zorder=1)
    plt.savefig("compareTimes.pdf", format="pdf")
