#-*- coding: UTF-8 -*- 
import pickle
import os
from xlutils.copy import copy
from matplotlib_venn import venn2,venn2_circles,venn3,venn3_circles
import math
import xlwt
import xlrd
import openpyxl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gmean
import struct
from plot_boxes import name_list
from plot_boxes import plot_many_boxes
from matplotlib.legend_handler import HandlerBase
large_fun_name = ['shoelace','nbody','pid','eigenvalue','iterGramSch','insCurrent','matrixDet']
name_list = name_list + large_fun_name

def floatToRawLongBits(value):
	return struct.unpack('Q', struct.pack('d', value))[0]

def longBitsToFloat(bits):
	return struct.unpack('d', struct.pack('Q', bits))[0]
def ini_xls_file(exname):
    new_excel = xlwt.Workbook()
    sheet = new_excel.add_sheet("Xscopevstool")
    sheet.write(0,0,"benchmark")
    sheet.write_merge(0,0,1,2, "INF+")
    sheet.write(1, 1, "Xscope")
    sheet.write(1, 2, "tool")
    sheet.write_merge(0,0,3,4, "INF-")
    sheet.write(1, 3, "RADE")
    sheet.write(1, 4, "tool")
    sheet.write_merge(0,0,5,6, "SUB+")
    sheet.write(1, 5, "Xscope")
    sheet.write(1, 6, "tool")
    sheet.write_merge(0,0,7,8, "SUB-")
    sheet.write(1, 7, "Xscope")
    sheet.write(1, 8, "tool")
    sheet.write_merge(0,0,9,10, "NAN")
    sheet.write(1, 9, "Xscope")
    sheet.write(1, 10, "tool")
    sheet.write_merge(0,0,11,12, "Time")
    sheet.write(1, 11, "Xscope")
    sheet.write(1, 12, "tool")
    sheet.write(0, 13, "SPEEDUP")
    new_excel.save(exname)

class VennLegendHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):
        return [plt.Line2D([width/2], [height/2], marker='o', markersize=8,
                          markerfacecolor=orig_handle.get_facecolor())]
def load_pickle(file_name):
    fp = open(file_name,'rb')
    res = pickle.load(fp)
    fp.close()
    return res
def exceptionNumsDivTime(res):
    times_Defped = []
    for i in res:
        tmp_time = i[1]
        tmp_count = 0
        if i[0]==[]:
            times_Defped.append(0)
            continue
        else:
            for k in i[0][0][1]:
                if k!=0:
                    tmp_count = tmp_count + k
        times_Defped.append(tmp_count/tmp_time)
    #print(times_Defped)
    log2_timesDe = []
    for x in times_Defped:
        if x!=0:
            if x == 1.0:
                log2_timesDe.append(np.log2(x)+1.0)
            else:
                log2_timesDe.append(np.log2(x))
        else:
            log2_timesDe.append(0)
    return log2_timesDe

def file_nums(filepath):
    FileNum = 0
    pklFileNum = 0
    # os.listdir(filePath)会读取出当前文件夹下的文件夹和文件
    for file in os.listdir(filepath): 
        FileNum += 1 # 统计当前文件夹下的文件夹(不包含子文件夹)和文件的总数
        if file.endswith(".pkl"):  
            pklFileNum += 1 # 统计当前文件夹下pkl类型文件的总数
    return pklFileNum


def analysis_dfeg_results_numPersec():
    str_lst = []
    file_num = file_nums("dfeg_res")
    # print(file_num)
    for i in range(0,file_num):
      str_lst.append("dfeg_res/dfeg_res_"+str(i)+".pkl")
    datas = []
    for i in str_lst:
      tmp_res = load_pickle(i)
      datas.append(exceptionNumsDivTime(tmp_res))
    mix_datas=[]
    for i in range(0,len(name_list)):
      mix_datas.append([])
    for i in datas:
      for j in range(0,len(name_list)):
        mix_datas[j].append(i[j])
    cv_lst = []
    print("Calculate CV")
    for i in mix_datas:
        mean = int(np.mean(i))
        std = np.std(i,ddof = 0)
        if mean != 0.0:
            cv_lst.append(std/mean)
        else:
            cv_lst.append(0)
    print("CV lst for each function")
    # print(len(cv_lst))
    print(cv_lst)
    print("Mean value of CV")
    print(np.mean(cv_lst))
    plot_many_boxes(mix_datas,"graph/plot_numsBox.pdf","Log2 of floating-point exceptions per second")

    
#analysis_100_results_numPersec()
    
def extract_basic_results(res1):
    number_excpetions = res1[0][0][0][1]
    time = res1[2]
    return [number_exceptions,time]
def get_means_lst(lsts):
    mean_lsts = [[] for i in range(len(lsts))]
    count = 0
    for j in lsts:
        tmp_mean_lst = []
        for i in range(5):
            tmp_sums = []
            for k in j:
                tmp_sums.append(k[i])
            tmp_mean_lst.append(int(np.mean(tmp_sums)))
        mean_lsts[count]=tmp_mean_lst
        count = count + 1
    return mean_lsts

    
def extract_average_results_mc(res_lsts):
    len_res = len(res_lsts)
    len_funs = len(res_lsts[0])
    num_excps_sums = [[] for i in range(len_funs)]
    time_sums = [[] for i in range(len_funs)]
    # num_excps_sums = []
    # time_sums = []
    # print(len(res_lsts))
    for i in res_lsts:
        count = 0
        for j in i:
            if j[4]!=[]:
                num_excps_sums[count].append(j[4][0][1])
                # num_excps_sums.append(j[0][0][1])
            else:
                num_excps_sums[count].append([0,0,0,0,0])
                # num_excps_sums.append([0,0,0,0,0])
            time_sums[count].append(j[1])
            # time_sums.append(j[1])
            count = count + 1
    # print(len(num_excps_sums))
    mix_datas = get_means_lst(num_excps_sums)
    mean_time = []
    for i in time_sums:
        mean_time.append(np.mean(i))
    return [mix_datas,mean_time]
def rm_duplicate_inputs(inps):
    num_exps = []
    for i in inps:
        if i[1]!=[]:
            unique_list = []
            unique_list = list(set([ tuple(x) for x in i[1]]))
            num_exps.append(len(unique_list))
        else:
            num_exps.append(0)
    return num_exps
      
def extract_average_results(res_lsts):
    len_res = len(res_lsts)
    len_funs = len(res_lsts[0])
    num_excps_sums = [[] for i in range(len_funs)]
    time_sums = [[] for i in range(len_funs)]
    # num_excps_sums = []
    # time_sums = []
    # print(len(res_lsts))
    for i in res_lsts:
        count = 0
        for j in i:
            if j[0]!=[]:
                inps = j[3]
                num_exps = rm_duplicate_inputs(inps)
                num_excps_sums[count].append(num_exps)
                # num_excps_sums[count].append(j[0][0][1])
                # num_excps_sums.append(j[0][0][1])
            else:
                num_excps_sums[count].append([0,0,0,0,0])
                # num_excps_sums.append([0,0,0,0,0])
            time_sums[count].append(j[1])
            # time_sums.append(j[1])
            count = count + 1
    # print(num_excps_sums)
    # print(time_sums)
    #mix_datas=[]
    #for i in range(0,len_funs):
    #  mix_datas.append([])
    #print(mix_datas)
    #count = 0
    #for i in num_excps_sums:
    #    mix_datas[count].append(get_means_lst(i))
    # print("num_excps_sums")
    # print(num_excps_sums)
    # print(len(num_excps_sums))
    mix_datas = get_means_lst(num_excps_sums)
    # print(mix_datas)
    mean_time = []
    for i in time_sums:
        mean_time.append(np.mean(i))
    # print("The sum time is")
    # print(np.sum(mean_time))
    # print(len(mean_time))
    return [mix_datas,mean_time]
        
        

def get_results_dfeg():
    str_lst = []
    file_num = file_nums("dfeg_res")
    # print(file_num)
    for i in range(0,file_num):
      str_lst.append("dfeg_res/dfeg_res_"+str(i)+".pkl")
    datas = []
    for i in str_lst:
      tmp_res = load_pickle(i)
      datas.append(tmp_res)
    return datas
def get_results_dfeg_bo():
    str_lst = []
    file_num = file_nums("dfeg_bo_res")
    # print(file_num)
    for i in range(0,file_num):
      str_lst.append("dfeg_bo_res/dfeg_bo_res.pkl")
    datas = []
    for i in str_lst:
      tmp_res = load_pickle(i)
      datas.append(tmp_res)
    return datas

def get_results_xscope():
    str_lst = []
    file_num = file_nums("xscope_res")
    # print(file_num)
    for i in range(0,file_num):
      str_lst.append("xscope_res/xscope_res_"+str(i)+".pkl")
    datas = []
    for i in str_lst:
      tmp_res = load_pickle(i)
      datas.append(tmp_res)
    return datas

def save_line_list(file_name,l):
    with open(file_name, "wb") as fp:
        pickle.dump(l, fp)
def save_res_table():
    res_lsts = get_results_xscope()
    datas_xscope = extract_average_results(res_lsts)
    res_lsts = get_results_dfeg()
    datas_dfeg = extract_average_results(res_lsts)
    save_line_list("xscope_datas.pkl",datas_xscope)
    save_line_list("dfeg_datas.pkl",datas_dfeg)
    
def save_res_mc_table():
    res_lsts = get_results_dfeg()
    datas_dfeg_mc = extract_average_results_mc(res_lsts)
    save_line_list("dfeg_datas_mc.pkl",datas_dfeg_mc)
def save_res_bo_table():
    res_lsts = get_results_dfeg_bo()
    datas_dfeg_mc = extract_average_results_mc(res_lsts)
    save_line_list("dfeg_datas_bo.pkl",datas_dfeg_mc)
# res_lsts = get_results_3()
# datas_xscope = extract_average_results(res_lsts)

input_nums = [1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 2, 2, 2, 2, 3, 2, 2] 
input_nums_large = [6,6,5,20,9,5,9]
input_nums = input_nums + input_nums_large
       
def generate_table_NOHE(exname,res1,res2):
    old_excel = xlrd.open_workbook(exname, formatting_info=True)
    new_excel = copy(old_excel)
    sheet = new_excel.get_sheet(0)
    k = 2
    total_res_xscope= [0,0,0,0,0,0]
    total_res_dfeg= [0,0,0,0,0,0]
    compare_x = []
    bold_font = xlwt.easyxf('font: bold on')
    # print(res1[0])
    print(len(res1[0]))
    print(len(res2[0]))
    for i,j in zip(res1[0],res2[0]):
        #print(k)
        sheet.write(k,0,name_list[k-2])
        # print(i)
        total_res_xscope[0]= total_res_xscope[0]+i[0]
        total_res_xscope[1]= total_res_xscope[1]+i[1]
        total_res_xscope[2]= total_res_xscope[2]+i[2]
        total_res_xscope[3]= total_res_xscope[3]+i[3]
        total_res_xscope[4]= total_res_xscope[4]+i[4]
        tmp_k = 0
        for idx in range(0,5):
            if i[idx]>j[idx]:
                sheet.write(k,1+tmp_k,str(i[idx]),bold_font)
                sheet.write(k,2+tmp_k,str(j[idx]))
            elif i[idx]<j[idx]:
                sheet.write(k,1+tmp_k,str(i[idx]))
                sheet.write(k,2+tmp_k,str(j[idx]),bold_font)
            else:
                sheet.write(k,1+tmp_k,str(i[idx]))
                sheet.write(k,2+tmp_k,str(j[idx]))
            tmp_k = tmp_k + 2
        total_res_dfeg[0]= total_res_dfeg[0]+j[0]
        total_res_dfeg[1]= total_res_dfeg[1]+j[1]
        total_res_dfeg[2]= total_res_dfeg[2]+j[2]
        total_res_dfeg[3]= total_res_dfeg[3]+j[3]
        total_res_dfeg[4]= total_res_dfeg[4]+j[4]
        t1 = res1[1][k-2]
        t2 = res2[1][k-2]
        sheet.write(k,11,str(t1))
        total_res_xscope[5]= total_res_xscope[5]+t1
        sheet.write(k,12,str(t2))
        total_res_dfeg[5]= total_res_dfeg[5]+t2
        sheet.write(k,13,str(t1/t2))
        compare_x.append(t1/t2)
        sheet.write(k,15,input_nums[k-2])
        k = k + 1
    ct = 0
    ct2 = 1
    ct3 = 2
    for n,m in zip(total_res_xscope,total_res_dfeg):
        sheet.write(k,ct+ct2,str(n))
        ct2 = ct2 + 2
        sheet.write(k,ct+ct3,str(m),bold_font)
        ct3 = ct3 + 2
    print("Mean Seepup")
    print(gmean(compare_x))
    sheet.write(k,13,str(gmean(compare_x)))
    sheet.write(k,14,str(total_res_xscope[5]/total_res_dfeg[5]))
    # print(len(compare_x))
    new_excel.save(exname)

def plot_bar_compare(y1,y2,new_name_lst,fname):
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
    plt.ylabel('Log2 of floating-point exceptions per second',fontsize=20)
    # plt.title('Scores by person')
    # plt.yticks(index+0.5*bar_width, id_lst,rotation=0)
    plt.xticks(index+0.5*bar_width, new_name_lst,rotation=90,fontsize=16)
    # plt.xticks(range(0,int(np.max(rpt1_lst))+1,2), range(0,int(np.max(rpt1_lst))*10+10,20),rotation=30)
    #plt.yticks(list(range(0,int(np.max(y1))+10,20)), list(range(0,int(np.max(rpt1_lst))*10+10,20)),fontsize=16)
    #plt.yticks(x,fontsize=16)
    plt.legend(prop={'size': 19.5})
    plt.tight_layout()
    plt.grid(zorder=1)
    plt.savefig(fname, format="pdf")
    plt.close()

def plot_exceptions_per_seconds(res1,res2,fname):
    xscope_exps_per =[]
    k = 0
    for i in res1[0]:
        xscope_exps_per.append(np.sum(i))
        k = k + 1
    dfeg_exps_per =[]
    k = 0
    for i in res2[0]:
        dfeg_exps_per.append(np.sum(i))
        k = k + 1
    print("Sum of exceptions")
    print("Exceptions detected by Xscope")
    print(np.sum(xscope_exps_per))
    print("Exceptions detected by DFEG")
    print(np.sum(dfeg_exps_per))
    print("Times: DFEG/Xscope")
    print(np.sum(dfeg_exps_per)/np.sum(xscope_exps_per))
    xscope_exps_per =[]
    k = 0
    for i in res1[0]:
        xscope_exps_per.append(np.sum(i)/res1[1][k])
        k = k + 1
    dfeg_exps_per =[]
    k = 0
    for i in res2[0]:
        dfeg_exps_per.append(np.sum(i)/res2[1][k])
        k = k + 1
    print("Xscope: Exceptions per second")
    print(np.mean(xscope_exps_per))
    print("DFEG: Exceptions per second")
    print(np.mean(dfeg_exps_per))
    print("Times: DFEG/Xscope")
    print(np.mean(dfeg_exps_per)/np.mean(xscope_exps_per))
    log2_timesXs = []
    log2_timesDe = []
    for x in dfeg_exps_per:
        if x!=0:
            if x == 1.0:
                log2_timesDe.append(np.log2(x)+1.0)
            else:
                log2_timesDe.append(np.log2(x))
        else:
            log2_timesDe.append(0)
    for x in xscope_exps_per:
        if x!=0:
            if x == 1.0:
                log2_timesXs.append(np.log2(x)+1.0)
            else:
                log2_timesXs.append(np.log2(x))
        else:
            log2_timesXs.append(0)
    plot_bar_compare(log2_timesXs,log2_timesDe,name_list,fname)
def plot_types_fig(y1,y2,fname):
    x = [1, 2, 3, 4,5]

    fig = plt.figure(figsize=(19, 8))
    ax = plt.subplot(111)
    index = np.arange(0,len(y1),1)
    # index = range(0,n_groups,2)
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
    plt.ylabel('Exception types found',fontsize=20)
    # plt.title('Scores by person')
    # plt.yticks(index+0.5*bar_width, id_lst,rotation=0)
    plt.xticks(index+0.5*bar_width, name_list,rotation=90,fontsize=16)
    # plt.xticks(range(0,int(np.max(rpt1_lst))+1,2), range(0,int(np.max(rpt1_lst))*10+10,20),rotation=30)
    #plt.yticks(list(range(0,int(np.max(y1))+10,20)), list(range(0,int(np.max(rpt1_lst))*10+10,20)),fontsize=16)
    plt.yticks(x,fontsize=16)
    plt.legend(prop={'size': 19.5})
    plt.tight_layout()
    plt.grid(zorder=1)
    plt.savefig(fname, format="pdf")
    plt.close()
    cc = 0
    c0c = 0
    for ci,cj,cn,inpn in zip(y1,y2,name_list,input_nums):
        if ci < cj:
    #        print(cn,inpn)
    #        print(ci,cj)
            cc = cc + 1
            if ci==0:
                c0c = c0c + 1
    y1_fd = 0
    for i in y1:
        if i != 0:
            y1_fd = y1_fd + 1
    y2_fd = 0
    for i in y2:
        if i != 0:
            y2_fd = y2_fd + 1
    print("Functions Types")
    print(cc)
    print(c0c)
    print(y1_fd)
    print(y2_fd)
def plot_types(res1,res2,fname):
    y1 = []
    for i in res1[0]:
        count = 0
        for j in i:
            if int(j) != 0:
                count = count + 1
        y1.append(count)
    print("Num of Types")
    print(np.sum(y1))
    y2 = []
    for i in res2[0]:
        count = 0
        for j in i:
            if int(j) != 0:
                count = count + 1
        y2.append(count)
    print(np.sum(y2))
    plot_types_fig(y1,y2,fname)
    
def plot_mc_venn(res1,res2,res3):
    xs_funs_found = []
    count = 0
    for i in res3:
        if i!=[0,0,0,0,0]:
            xs_funs_found.append(count)
        count = count + 1
    print("MCMCM ability")
    mc_funs_found = []
    mc_types_count = []
    rd_funs_found = []
    rd_types_count = []
    funs_found = []
    types_count = []
    count = 0
    for i,j in zip(res1,res2):
        if i!=[0,0,0,0,0]:
            funs_found.append(count)
            count_ori = i 
            if j!=[0,0,0,0,0]:
                mc_funs_found.append(count)
                count_mcmc = j 
                mc_flag = 0
                rd_flag = 0
                tid = 0
                for oi,mc in zip(count_ori,count_mcmc):
                    if oi > 0:
                        types_count.append(count*100+tid)
                    if mc > 0:
                        mc_types_count.append(count*100+tid)
                    # if (oi>0)&(mc==oi):
                    #     #mc_flag = 1
                    #     types_count += 1
                    if (oi>0)&(mc<oi):
                        rd_flag = 1
                        rd_types_count.append(count*100+tid)
                    tid = tid + 1
                if rd_flag == 1:
                    rd_funs_found.append(count)
            else:
                tid = 0
                for oi in count_ori:
                    if oi>0:
                        rd_types_count.append(count*100+tid)
                    tid = tid + 1
                rd_funs_found.append(count)
                # if flag == 1:
                #     print(i[0])
                #     print(i[4])
                #     ability_count += 1
        count = count + 1
    print("funs,types,len(funs)")
    print(mc_funs_found)
    print(len(mc_types_count))
    print(len(mc_funs_found))
    print("funs,types,len(funs)")
    print(rd_funs_found)
    print(len(rd_types_count))
    print(len(rd_funs_found))
    print("funs,types,len(funs)")
    print(funs_found)
    print(len(types_count))
    print(len(funs_found))
    set1 = set(mc_funs_found) 
    set2 = set(rd_funs_found) 
    set3 = set(xs_funs_found) 
    out = venn3([set1, set2,set3], set_labels=('MCMC', 'Random','Xscope'), set_colors=('skyblue', 'lightgreen','orange'), alpha=0.7)
    plt.tight_layout()
    plt.savefig("graph/MCMC_sampling_xs.pdf", format="pdf")
    plt.close()
    set1 = set(mc_types_count)
    set2 = set(rd_types_count)
    plt.rcParams["figure.autolayout"] = True
    out = venn2([set1, set2], set_labels=('MCMC', 'Random'), set_colors=('skyblue', 'lightgreen'), alpha=0.7)
    for text in out.set_labels:
        text.set_fontsize(20)

    for text in out.subset_labels:
        text.set_fontsize(18)
    plt.tight_layout()
    plt.savefig("graph/MCMC_sampling_types.pdf", format="pdf")
    return 0
def partition_double(x):
    if math.isnan(x):
        return 2049.0
    val = np.fabs(x)
    valint = floatToRawLongBits(val)
    # val = valint<<1
    if x<0:
        sign = -1
    else:
        sign = 1
    exponent = (valint>>52)
    significant =val/2.0/pow(2.0,exponent-1023)
    return sign*(exponent+significant)
def plot_2vfunc_domain(res1,res2,id):
    inps_xscope = []
    for i in res1[id][3]:
        inps_xscope = inps_xscope + i[1]
    inps_tool = []
    count = 0
    for i in res2[id][3]:
        inps_tool = inps_tool + i[1]
    inps_mcmc = []
    for i in res2[id][5]:
        print(i)
        inps_mcmc = inps_mcmc + i[1]
    inps_x = []
    inps_y = []
    for inp in inps_tool:
        inps_x.append(partition_double(inp[0]))
        inps_y.append(partition_double(inp[1]))
    fig = plt.figure(figsize=(10, 10))
    fig, ax = plt.subplots(1, figsize=(10, 10))
    #ax = SubplotZero(fig, 111)
    ax.scatter(inps_x,inps_y,label='DFEG')
    inps_x = []
    inps_y = []
    for inp in inps_xscope:
        inps_x.append(partition_double(inp[0]))
        inps_y.append(partition_double(inp[1]))
    ax.scatter(inps_x,inps_y,label='Xscope',color=['red'])
    inps_x = []
    inps_y = []
    for inp in inps_mcmc:
        inps_x.append(partition_double(inp[0]))
        inps_y.append(partition_double(inp[1]))
    ax.scatter(inps_x,inps_y,label='MCMC',color=['green'])
    plt.tick_params(axis='both', labelsize=18)
    plt.xlim((-2150, 2150))
    plt.ylim((-2150, 2150))
    plt.legend(prop={'size': 19.5})
    plt.tight_layout()
    plt.grid(zorder=1)
    plt.savefig("graph/plot_domain"+str(id)+".png", format="png")
    plt.close()
def get_apart_venn_set(res1,res2):
    # print("MCMCM ability")
    mc_funs_found = []
    mc_types_count = []
    rd_funs_found = []
    rd_types_count = []
    funs_found = []
    types_count = []
    count = 0
    for i,j,inpn in zip(res1,res2,input_nums):
        if i!=[0,0,0,0,0]:
            funs_found.append(count)
            count_ori = i 
            if j!=[0,0,0,0,0]:
                # print(count)
                # print(j)
                mc_funs_found.append(count)
                count_mcmc = j 
                mc_flag = 0
                rd_flag = 0
                tid = 0
                for oi,mc in zip(count_ori,count_mcmc):
                    if oi > 0:
                        types_count.append(count*100+tid)
                    if mc > 0:
                        mc_types_count.append(count*100+tid)
                    if (oi>0)&(mc<oi):
                        rd_flag = 1
                        rd_types_count.append(count*100+tid)
                    if (mc>0)&(mc==oi)&(inpn==2):
                        print("******************")
                        print(count)
                        print(sum(i))
                    tid = tid + 1
                if rd_flag == 1:
                    rd_funs_found.append(count)
            else:
                tid = 0
                for oi in count_ori:
                    if oi>0:
                        rd_types_count.append(count*100+tid)
                    tid = tid + 1
                rd_funs_found.append(count)
                # if flag == 1:
                #     print(i[0])
                #     print(i[4])
                #     ability_count += 1
        count = count + 1
    print("MCMC funs,types,len(funs)")
    print(mc_funs_found)
    print(len(mc_types_count))
    print(len(mc_funs_found))
    print("Random funs,types,len(funs)")
    print(rd_funs_found)
    print(len(rd_types_count))
    print(len(rd_funs_found))
    print("ALL funs,types,len(funs)")
    print(funs_found)
    print(len(types_count))
    print(len(funs_found))
    set1 = set(mc_funs_found) 
    set2 = set(rd_funs_found) 
    set3 = set(mc_types_count) 
    set4 = set(rd_types_count)
    return set1,set2,set3,set4

def plot_set_venn(set1,set2,set3,set4,set5,set6):
    # print(set1)
    # print(set2)
    # print(set3)
    # print(set4)
    # print(set5)
    # print(set6)
    plt.rcParams["figure.autolayout"] = True
    out = venn3([set1, set2,set3], set_labels=('BO', 'MCMC','Random'), set_colors=('skyblue', 'lightgreen','orange'), alpha=0.7)
    
    # for text in out.set_labels:
    #     text.set_fontsize(20)

    # for text in out.subset_labels:
    #     text.set_fontsize(18)
    plt.tight_layout()
    plt.savefig("graph/bo_mc_rd_funs_compare.pdf", format="pdf")
    plt.close()
    plt.rcParams["figure.autolayout"] = True
    out = venn3([set4, set5,set6], set_labels=('BO', 'MCMC','Random'), set_colors=('skyblue', 'lightgreen','orange'), alpha=0.7)
    # for text in out.set_labels:
    #     text.set_fontsize(20)

    # for text in out.subset_labels:
    #     text.set_fontsize(18)
    plt.tight_layout()
    plt.savefig("graph/bo_mc_rd_types_compare.pdf", format="pdf")
    plt.close()
    return 0
    
def analysis_results_appendix():
    print("########################")
    print("Compare with Xscope fp many")
    fname = "graph_appendix/compare_typesX_fpmany.pdf"
    datas_xscope = extract_average_results([load_pickle("xscope_res/xscope_res_0.pkl")])
    save_line_list("xscope_datas_fpmany.pkl",datas_xscope)
    res2 = load_pickle("dfeg_datas.pkl")
    res1 = load_pickle("xscope_datas_fpmany.pkl")
    ini_xls_file("res_table_fpmany.xls")
    generate_table_NOHE("res_table_fpmany.xls",res1,res2)
    # get figure 3 in paper
    plot_types(res1,res2,fname)
    # get figure 4 in paper
    fname = "graph_appendix/compareTimes_fpmany.pdf"
    plot_exceptions_per_seconds(res1,res2,fname)
    # get Table 2 in paper
    print("########################")
    print("Compare with Xscope fp two")
    fname = "graph_appendix/compare_typesX_fptwo.pdf"
    datas_xscope = extract_average_results([load_pickle("xscope_res_appendix/xscope_res_3.pkl")])
    save_line_list("xscope_datas_fptwo.pkl",datas_xscope)
    res2 = load_pickle("dfeg_datas.pkl")
    res1 = load_pickle("xscope_datas_fptwo.pkl")
    ini_xls_file("res_table_fptwo.xls")
    generate_table_NOHE("res_table_fptwo.xls",res1,res2)
    # get figure 3 in paper
    plot_types(res1,res2,fname)
    # get figure 4 in paper
    fname = "graph_appendix/compareTimes_fptwo.pdf"
    plot_exceptions_per_seconds(res1,res2,fname)

    print("########################")
    print("Compare with Xscope fp whole")
    fname = "graph_appendix/compare_typesX_fpwhole.pdf"
    datas_xscope = extract_average_results([load_pickle("xscope_res_appendix/xscope_res_4.pkl")])
    save_line_list("xscope_datas_fpwhole.pkl",datas_xscope)
    res2 = load_pickle("dfeg_datas.pkl")
    res1 = load_pickle("xscope_datas_fpwhole.pkl")
    ini_xls_file("res_table_fpwhole.xls")
    generate_table_NOHE("res_table_fpwhole.xls",res1,res2)
    # get figure 3 in paper
    plot_types(res1,res2,fname)
    # get figure 4 in paper
    fname = "graph_appendix/compareTimes_fpwhole.pdf"
    plot_exceptions_per_seconds(res1,res2,fname)

    print("########################")
    print("Compare with Xscope exp two")
    fname = "graph_appendix/compare_typesX_exptwo.pdf"
    datas_xscope = extract_average_results([load_pickle("xscope_res_appendix/xscope_res_5.pkl")])
    save_line_list("xscope_datas_exptwo.pkl",datas_xscope)
    res2 = load_pickle("dfeg_datas.pkl")
    res1 = load_pickle("xscope_datas_exptwo.pkl")
    ini_xls_file("res_table_exptwo.xls")
    generate_table_NOHE("res_table_exptwo.xls",res1,res2)
    # get figure 3 in paper
    plot_types(res1,res2,fname)
    # get figure 4 in paper
    fname = "graph_appendix/compareTimes_exptwo.pdf"
    plot_exceptions_per_seconds(res1,res2,fname)
    
    print("########################")
    print("Compare with Xscope exp whole")
    fname = "graph_appendix/compare_typesX_expwhole.pdf"
    datas_xscope = extract_average_results([load_pickle("xscope_res_appendix/xscope_res_6.pkl")])
    save_line_list("xscope_datas_expwhole.pkl",datas_xscope)
    res2 = load_pickle("dfeg_datas.pkl")
    res1 = load_pickle("xscope_datas_expwhole.pkl")
    ini_xls_file("res_table_expwhole.xls")
    generate_table_NOHE("res_table_expwhole.xls",res1,res2)
    # get figure 3 in paper
    plot_types(res1,res2,fname)
    # get figure 4 in paper
    fname = "graph_appendix/compareTimes_expwhole.pdf"
    plot_exceptions_per_seconds(res1,res2,fname)

    print("########################")
    print("Compare with Xscope exp many")
    fname = "graph_appendix/compare_typesX_expmany.pdf"
    datas_xscope = extract_average_results([load_pickle("xscope_res_appendix/xscope_res_7.pkl")])
    save_line_list("xscope_datas_expmany.pkl",datas_xscope)
    res2 = load_pickle("dfeg_datas.pkl")
    res1 = load_pickle("xscope_datas_expmany.pkl")
    ini_xls_file("res_table_expmany.xls")
    generate_table_NOHE("res_table_expmany.xls",res1,res2)
    # get figure 3 in paper
    plot_types(res1,res2,fname)
    # get figure 4 in paper
    fname = "graph_appendix/compareTimes_expmany.pdf"
    plot_exceptions_per_seconds(res1,res2,fname)
    # get figure_5 in paper
    # analysis_dfeg_results_numPersec()
    # get figure 6 and 7 in paper
    # save_res_bo_table()
    #res1 = load_pickle("dfeg_datas.pkl")
    #res2 = load_pickle("dfeg_datas_bo.pkl")
    #res3 = load_pickle("dfeg_datas_mc.pkl")
    #bo_funs_set,rd_funs_set,bo_types_set,rd_types_set=get_apart_venn_set(res1[0],res2[0])
    #mc_funs_set,rd_funs_set,mc_types_set,rd_types_set=get_apart_venn_set(res1[0],res3[0])
    #plot_set_venn(bo_funs_set,mc_funs_set,rd_funs_set,bo_types_set,mc_types_set,rd_types_set)
def case_study_one(fid):
    res1 = load_pickle("xscope_res/xscope_res_1.pkl")
    res2 = load_pickle("dfeg_res/dfeg_res_1.pkl")
    print(name_list[fid])
    plot_2vfunc_domain(res1,res2,fid)

def case_study(fid):
    res1 = load_pickle("xscope_res/xscope_res_1.pkl")
    res2 = load_pickle("dfeg_res/dfeg_res_1.pkl")
    print(name_list[fid])
    count = 0
    for i,j in zip(name_list,input_nums):
        if j == 2:
            print(i)
            plot_2vfunc_domain(res1,res2,count)
        count = count + 1
    
analysis_results_appendix()
#case_study_one(68)
