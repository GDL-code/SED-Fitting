# -*- coding: utf-8 -*-
#!/usr/bin/python
import numpy as np
from astropy.cosmology import WMAP9 as cosmo
from astropy.cosmology import FlatLambdaCDM
import astropy.constants as con
from scipy.interpolate import interp1d
from astropy.io import fits
import pandas
from scipy import integrate
import os
from extinction import apply,fm07
import emcee
import matplotlib.pyplot as plt
import corner
import time
import multiprocessing
from tqdm import tqdm
import warnings
import sys
sys.path.append('E:\SED fitting')
from draw_picture_v10 import draw_result
from mc_fitting_v10 import *

if __name__=='__main__':
#########Process 1.
    redshift          =0.414
    input_csv_file = 'CW1_FVJ19h04m45.04s_+48d53m08.9s_Phot_v2.csv'
    t_object = pandas.read_csv(input_csv_file)
    flux = np.array(t_object['flux (microJy)'])/1000.
    flux_error = np.array(t_object['flux unc (microJy)']) / 1000.
    filter_list       =np.array(['PS1_g','PS1_r','PS1_i','PS1_z','PS1_y','W1','W2','W3','W4','AKARI_65','AKARI_90','IRAS_12','IRAS_25','IRAS_60','IRAS_100'])
    wave = np.array(t_object['filter wavelength (microns)'])
    up_limit = np.array([])
    sourcename = input_csv_file.split('FV')[1].split('_Phot')[0]
    tem_control       =[0,0,1,1,1,0,1]
    Av_control        =[0,0]
    m=1
    flux = flux[:-1]
    flux_error = flux_error[:-1]
    wave =wave[:-1]
#########Process 2.
    dir_filter_folder = 'E:/Filter/sorted/'
    dir_assef10       = 'E:/SED_templates/Assef_models/'
    save_path         = 'E:/SED fitting/result/'
    initial_step      =500
    walker            =50
    step              =10000
    Grey_Tmin         =5.
    Grey_Tmax         =500.
    Grey_Smin         =0.
    Grey_Smax         =100000
    Grey_beta_min     =0.
    Grey_beta_max     =10.
    Dust_cons_min     =0.
    Dust_cons_max     =1000
    Dust_alpha_min    =-10.
    Dust_alpha_max    =-1.
    Dust_T_min        =3.0*(1+redshift)
    Dust_T_max        =1500.
    Av_1_max          =70
    Av_2_max          =70
    par  =[Grey_Tmin,Grey_Tmax,Grey_Smin,Grey_Smax,Grey_beta_min,Grey_beta_max,Dust_cons_min,Dust_cons_max,Dust_alpha_min,Dust_alpha_max,Av_1_max,Av_2_max,Dust_T_min,Dust_T_max]
    global_dict         ={'dir_filter_folder':dir_filter_folder,'dir_assef10':dir_assef10,'par':par,'step':step,'walker':walker,'initial_step':initial_step}
    source_dict       ={'wave':wave,'flux':flux,'flux_e':flux_error,'redshift':redshift,'filter_list':filter_list,'sourcename':sourcename,'up_limit':up_limit}
    
#########Process 3.
# emcmc(tem_control,Av_control,source_dict, global_dict,0)
    multiprocessing.freeze_support()
    pool = multiprocessing.Pool()
    cpus = multiprocessing.cpu_count()-3
    results = []
    for i in range(0, cpus):
        result = pool.apply_async(emcmc,args=(tem_control,Av_control,source_dict, global_dict,i,))
        results.append(result)
    pool.close()
    pool.join()
    result_sampler    =results[0].get()
    for i in range(1,len(results)):
        samplers      =results[i].get()
        result_sampler= np.vstack((result_sampler,samplers))
############Process 4.
    array             =result_sampler.shape
    theta             =[]
    for i in range(array[1]):
        theta.append(np.median(result_sampler[:,i]))
    chi_suqare,re_chi =emcmc(tem_control,Av_control,source_dict,global_dict,0,chi_square=1,theta=theta)
    source_dict['chi_square']=chi_suqare
    source_dict['re_chi']=re_chi
    draw_result(tem_control,Av_control,result_sampler,save_path,dir_filter_folder,dir_assef10,m,source_dict)


































