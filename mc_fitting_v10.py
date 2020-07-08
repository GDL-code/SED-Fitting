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
# sys.path.append('E:\Code\SED fitting')
# from draw_picture_v10 import draw_result


def cw_filter_loading(bandname):

# This funciton loads filter files from the local copy of the filter files from 
# the SVO archive. 
# Input:
#   bandname:   Name of the band. Similar to the cw_wavelength argument
# Output:
#   bandpass:   python dictionary of the bandpass of the requested band.
#               bandpass.filter_w:  wavelengths of the bandpass (in micron)
#               bandpass.filter_t:  transmittance (in percentage) 
#
# - History
#   2018_0118   Created by Chao-Wei Tsai, UCLA
#   2018_0802   Adding the HST filters -    Chao-Wei Tsai, UCLA
#   2020_0329   Modify Chao-Wei Tsai's code so that it can be run by python -    Guodong Li, NAOC
    file_name         = {'GALAX_FUV' :'GALEX/GALEX.GALEX.FUV.csv'         ,'GALAX_NUV' :'GALEX/GALEX.GALEX.NUV.csv'      ,'SDSS_u'    :'SLOAN/SLOAN.SDSS.u.csv',
                          'SDSS_g'    :'SLOAN/SLOAN.SDSS.g.csv'            ,'SDSS_r'    :'SLOAN/SLOAN.SDSS.r.csv'         ,'SDSS_i'    :'SLOAN/SLOAN.SDSS.i.csv',
                          'SDSS_z'    :'SLOAN/SLOAN.SDSS.z.csv'            ,'u'         :'SLOAN/SLOAN.SDSS.u.csv'         ,'g'         :'SLOAN/SLOAN.SDSS.g.csv',
                          'r'         :'SLOAN/SLOAN.SDSS.r.csv'            ,'i'         :'SLOAN/SLOAN.SDSS.i.csv'         ,'z'         :'SLOAN/SLOAN.SDSS.z.csv',
                          'TWOMASS_J' :'2MASS/2MASS.2MASS.J.csv'           ,'TWOMASS_H' :'2MASS/2MASS.2MASS.H.csv'        ,
                          'TWOMASS_Ks':'2MASS/2MASS.2MASS.Ks.csv'          ,'J'         :'2MASS/2MASS.2MASS.J.csv'        ,'H'         :'2MASS/2MASS.2MASS.H.csv',
                          'Ks'        :'2MASS/2MASS.2MASS.Ks.csv'          ,'TWOMASS_K' :'2MASS/2MASS.2MASS.Ks.csv'       ,'UKIDSS_Y'  :'/UKIRT/UKIRT.UKIDSS.Y.csv',
                          'UKIDSS_Z'  :'/UKIRT/UKIRT.UKIDSS.Z.csv'         ,'UKIDSS_H'  :'/UKIRT/UKIRT.UKIDSS.H.csv'      ,'UKIDSS_J'  :'/UKIRT/UKIRT.UKIDSS.J.csv',
                          'UKIDSS_K'  :'/UKIRT/UKIRT.UKIDSS.K.csv'         ,'WFCAM_Y'   :'/UKIRT/UKIRT.WFCAM.Y.csv'       ,'WFCAM_Z'   :'/UKIRT/UKIRT.WFCAM.Z.csv',
                          'WFCAM_H'   :'/UKIRT/UKIRT.WFCAM.H.csv'          ,'WFCAM_J'   :'/UKIRT/UKIRT.WFCAM.J.csv'       ,'WFCAM_K'   :'/UKIRT/UKIRT.WFCAM.K.csv',
                          'WFCAM_H2'  :'/UKIRT/UKIRT.WFCAM.H2.csv'         ,'WFCAM_Brg' :'/UKIRT/UKIRT.WFCAM.Brg.csv'     ,'WISE1'     :'WISE/WISE.WISE.W1.csv',
                          'WISE2'     :'WISE/WISE.WISE.W2.csv'             ,'WISE3'     :'WISE/WISE.WISE.W3.csv'          ,'WISE4'     :'WISE/WISE.WISE.W4.csv',
                          'W1'        :'WISE/WISE.WISE.W1.csv'             ,'W2'        :'WISE/WISE.WISE.W2.csv'          ,'W3'        :'WISE/WISE.WISE.W3.csv',
                          'W4'        :'WISE/WISE.WISE.W4.csv'             ,'PACS070'   :'Herschel/Herschel.Pacs.blue.csv','PACS100'   :'Herschel/Herschel.Pacs.green.csv',
                          'PACS160'   :'Herschel/Herschel.Pacs.red.csv'    ,'SPIRE250'  :'Herschel/Herschel.SPIRE.PSW.csv','SPIRE350'  :'Herschel/Herschel.SPIRE.PMW.csv',
                          'SPIRE500'  :'Herschel/Herschel.SPIRE.PLW.csv'   ,'PACS70'    :'Herschel/Herschel.Pacs.blue.csv','IRAC1'     :'Spitzer/Spitzer.IRAC.I1.csv',
                          'IRAC2'     :'Spitzer/Spitzer.IRAC.I2.csv'       ,'IRAC3'     :'Spitzer/Spitzer.IRAC.I3.csv'    ,'IRAC4'     :'Spitzer/Spitzer.IRAC.I4.csv',
                          'MIPS25'    :'Spitzer/Spitzer.MIPS.24mu.csv'     ,'MIPS70'    :'Spitzer/Spitzer.MIPS.70mu.csv'  ,'MIPS160'   :'Spitzer/Spitzer.MIPS.160mu.csv',
                          'I1'        :'Spitzer/Spitzer.IRAC.I1.csv'       ,'I2'        :'Spitzer/Spitzer.IRAC.I2.csv'    ,'I3'        :'Spitzer/Spitzer.IRAC.I3.csv',
                          'I4'        :'Spitzer/Spitzer.IRAC.I4.csv'       ,'MIPS025'   :'Spitzer/Spitzer.MIPS.24mu.csv'  ,'MIPS24'    :'Spitzer/Spitzer.MIPS.24mu.csv',
                          'MIPS024'   :'Spitzer/Spitzer.MIPS.24mu.csv'     ,'WFC3_F160W':'HST/HST.WFC3_IR.F160W.csv'      ,'SCUBA2_450':'JCMT_CWT/JCMT.SCUBA2.450.csv',
                          'SCUBA2_850':'JCMT_CWT/JCMT.SCUBA2.850.csv'      ,'PS1_g'     :'PAN-STARRS/PAN-STARRS.PS1.g.csv','PS1_r'     :'PAN-STARRS/PAN-STARRS.PS1.r.csv',
                          'PS1_z'     :'PAN-STARRS/PAN-STARRS.PS1.z.csv'   ,'PS1_i'     :'PAN-STARRS/PAN-STARRS.PS1.i.csv','PS1_w'     :'PAN-STARRS/PAN-STARRS.PS1.w.csv',
                          'PS1_open'  :'PAN-STARRS/PAN-STARRS.PS1.open.csv','PS1_y'     :'PAN-STARRS/PAN-STARRS.PS1.y.csv','GAIA_G'    :'GAIA/GAIA.GAIA0.G.csv',
                          'GAIA_Bp'   :'GAIA/GAIA.GAIA0.Gbp.csv'           ,'GAIA_Rp'   :'GAIA/GAIA.GAIA0.Grp.csv'        ,'ACS_HRC_F220W':'HST/HST.ACS_HRC.F220W.csv',
                          'ACS_HRC_F250W':'HST/HST.ACS_HRC.F250W.csv'      ,'ACS_HRC_F330W':'HST/HST.ACS_HRC.F330W.csv'   ,'ACS_HRC_F344N':'HST/HST.ACS_HRC.F344N.csv',
                          'ACS_HRC_F435W':'HST/HST.ACS_HRC.F435W.csv'      ,'ACS_HRC_F475W':'HST/HST.ACS_HRC.F475W.csv'   ,'ACS_HRC_F502N':'HST/HST.ACS_HRC.F502N.csv',
                          'ACS_HRC_F550M':'HST/HST.ACS_HRC.F550M.csv'      ,'ACS_HRC_F555W':'HST/HST.ACS_HRC.F555W.csv'   ,'ACS_HRC_F606W':'HST/HST.ACS_HRC.F606W.csv',
                          'ACS_HRC_F652W':'HST/HST.ACS_HRC.F652W.csv'      ,'ACS_HRC_F658N':'HST/HST.ACS_HRC.F658N.csv'   ,'ACS_HRC_F660N':'HST/HST.ACS_HRC.F660N.csv',
                          'ACS_HRC_F775W':'HST/HST.ACS_HRC.F775W.csv'      ,'ACS_HRC_F814N':'HST/HST.ACS_HRC.F814N.csv'   ,'ACS_HRC_F850LP':'HST/HST.ACS_HRC.F850LP.csv',
                          'ACS_HRC_F892N':'HST/HST.ACS_HRC.F892N.csv'      ,'ACS_HRC_G800L':'HST/HST.ACS_HRC.G800L.csv'   ,'ACS_SBC_F115LP':'HST/HST.ACS_SBC.F115LP.csv',
                          'ACS_SBC_F122M':'HST/HST.ACS_SBC.F122M.csv'      ,'ACS_SBC_F125LP':'HST/HST.ACS_SBC.F125LP.csv' ,'ACS_SBC_F140LP':'HST/HST.ACS_SBC.F140LP.csv',
                          'ACS_SBC_F150LP':'HST/HST.ACS_SBC.F150LP.csv'    ,'ACS_SBC_F165LP':'HST/HST.ACS_SBC.F165LP.csv' ,'ACS_SBC_PR110L':'HST/HST.ACS_SBC.PR110L.csv',
                          'ACS_SBC_PR130L':'HST/HST.ACS_SBC.PR130L.csv'    ,'ACS_WFC_F435W':'HST/HST.ACS_WFC.F435W_81.csv','ACS_WFC_F475W':'HST/HST.ACS_WFC.F475W_81.csv',
                          'ACS_WFC_F502N':'HST/HST.ACS_WFC.F502N_81.csv'   ,'ACS_WFC_F550M':'HST/HST.ACS_WFC.F550M_81.csv','ACS_WFC_F555W':'HST/HST.ACS_WFC.F555W_81.csv',
                          'ACS_WFC_F606W':'HST/HST.ACS_WFC.F606W_81.csv'   ,'ACS_WFC_F625W':'HST/HST.ACS_WFC.F625W_81.csv','ACS_WFC_F658N':'HST/HST.ACS_WFC.F658N_81.csv',
                          'ACS_WFC_F660N':'HST/HST.ACS_WFC.F660N_81.csv'   ,'ACS_WFC_F775W':'HST/HST.ACS_WFC.F775W_81.csv','ACS_WFC_F814W':'HST/HST.ACS_WFC.F814W_81.csv',
                          'ACS_WFC_F850LP':'HST/HST.ACS_WFC.F850LP_81.csv' ,'ACS_WFC_F892N':'HST/HST.ACS_WFC.F892N_81.csv','NICMOS1_F090M':'HST/HST.NICMOS1.F090M.csv',
                          'NICMOS1_F095N':'HST/HST.NICMOS1.F095N.csv'      ,'NICMOS1_F097N':'HST/HST.NICMOS1.F097N.csv'   ,'NICMOS1_F108N':'HST/HST.NICMOS1.F108N.csv',
                          'NICMOS1_F110M':'HST/HST.NICMOS1.F110M.csv'      ,'NICMOS1_F110W':'HST/HST.NICMOS1.F110W.csv'   ,'NICMOS1_F113N':'HST/HST.NICMOS1.F113N.csv',
                          'NICMOS1_F140W':'HST/HST.NICMOS1.F140W.csv'      ,'NICMOS1_F145M':'HST/HST.NICMOS1.F145M.csv'   ,'NICMOS1_F160W':'HST/HST.NICMOS1.F160W.csv',
                          'NICMOS1_F164N':'HST/HST.NICMOS1.F164N.csv'      ,'NICMOS1_F165M':'HST/HST.NICMOS1.F165M.csv'   ,'NICMOS1_F166N':'HST/HST.NICMOS1.F166N.csv',
                          'NICMOS1_F170M':'HST/HST.NICMOS1.F170M.csv'      ,'NICMOS1_F187N':'HST/HST.NICMOS1.F187N.csv'   ,'NICMOS1_F190N':'HST/HST.NICMOS1.F190N.csv',
                          'NICMOS1_POL0S':'HST/HST.NICMOS1.POL0S.csv'      ,'NICMOS2_F160W':'HST/HST.NICMOS2.F160W.csv'   ,'NICMOS2_F165M':'HST/HST.NICMOS2.F165M.csv',
                          'NICMOS2_F171M':'HST/HST.NICMOS2.F171M.csv'      ,'NICMOS2_F180M':'HST/HST.NICMOS2.F180M.csv'   ,'NICMOS2_F187N':'HST/HST.NICMOS2.F187N.csv',
                          'NICMOS2_F187W':'HST/HST.NICMOS2.F187W.csv'      ,'NICMOS2_F190N':'HST/HST.NICMOS2.F190N.csv'   ,'NICMOS2_F204M':'HST/HST.NICMOS2.F204M.csv',
                          'NICMOS2_F205W':'HST/HST.NICMOS2.F205W.csv'      ,'NICMOS2_F207M':'HST/HST.NICMOS2.F207M.csv'   ,'NICMOS2_F212N':'HST/HST.NICMOS2.F212N.csv',
                          'NICMOS2_F215N':'HST/HST.NICMOS2.F215N.csv'      ,'NICMOS2_F216N':'HST/HST.NICMOS2.F216N.csv'   ,'NICMOS2_F222M':'HST/HST.NICMOS2.F222M.csv',
                          'NICMOS2_F237M':'HST/HST.NICMOS2.F237M.csv'      ,'NICMOS2_POL0L':'HST/HST.NICMOS2.POL0L.csv'   ,'NICMOS2_POL120L':'HST/HST.NICMOS2.POL120L.csv',
                          'NICMOS2_POL240L':'HST/HST.NICMOS2.POL240L.csv'  ,'NICMOS3_F108N':'HST/HST.NICMOS3.F108N.csv'   ,'NICMOS3_F110W':'HST/HST.NICMOS3.F110W.csv',
                          'NICMOS3_F113N':'HST/HST.NICMOS3.F113N.csv'      ,'NICMOS3_F150W':'HST/HST.NICMOS3.F150W.csv'   ,'NICMOS3_F160W':'HST/HST.NICMOS3.F160W.csv',
                          'NICMOS3_F164N':'HST/HST.NICMOS3.F164N.csv'      ,'NICMOS3_F166N':'HST/HST.NICMOS3.F166N.csv'   ,'NICMOS3_F175W':'HST/HST.NICMOS3.F175W.csv',
                          'NICMOS3_F187N':'HST/HST.NICMOS3.F187N.csv'      ,'NICMOS3_F190N':'HST/HST.NICMOS3.F190N.csv'   ,'NICMOS3_F196N':'HST/HST.NICMOS3.F196N.csv',
                          'NICMOS3_F200N':'HST/HST.NICMOS3.F200N.csv'      ,'NICMOS3_F212N':'HST/HST.NICMOS3.F212N.csv'   ,'NICMOS3_F215N':'HST/HST.NICMOS3.F215N.csv',
                          'NICMOS3_F222M':'HST/HST.NICMOS3.F222M.csv'      ,'NICMOS3_F240M':'HST/HST.NICMOS3.F240M.csv'   ,'NICMOS3_G096':'HST/HST.NICMOS3.G096.csv',
                          'NICMOS3_G141' :'HST/HST.NICMOS3.G141.csv'       ,'NICMOS3_G206' :'HST/HST.NICMOS3.G206.csv'    ,'WFC3_IR_F098M':'HST/HST.WFC3_IR.F098M.csv',
                          'WFC3_IR_F105W':'HST/HST.WFC3_IR.F105W.csv'      ,'WFC3_IR_F110W':'HST/HST.WFC3_IR.F110W.csv'   ,'WFC3_IR_F125W':'HST/HST.WFC3_IR.F125W.csv',
                          'WFC3_IR_F126N':'HST/HST.WFC3_IR.F126N.csv'      ,'WFC3_IR_F127M':'HST/HST.WFC3_IR.F127M.csv'   ,'WFC3_IR_F128N':'HST/HST.WFC3_IR.F128N.csv',
                          'WFC3_IR_F130N':'HST/HST.WFC3_IR.F130N.csv'      ,'WFC3_IR_F132N':'HST/HST.WFC3_IR.F132N.csv'   ,'WFC3_IR_F139M':'HST/HST.WFC3_IR.F139M.csv',
                          'WFC3_IR_F140W':'HST/HST.WFC3_IR.F140W.csv'      ,'WFC3_IR_F153M':'HST/HST.WFC3_IR.F153M.csv'   ,'WFC3_IR_F160W':'HST/HST.WFC3_IR.F160W.csv',
                          'WFC3_IR_F164N':'HST/HST.WFC3_IR.F164N.csv'      ,'WFC3_IR_F167N':'HST/HST.WFC3_IR.F167N.csv'   ,'WFC3_IR_G102':'HST/HST.WFC3_IR.G102.csv',
                          'WFC3_IR_G141' :'HST/HST.WFC3_IR.G141.csv'       ,'WFC3_UVIS_F200LP':'HST/HST.WFC3_UVIS2.F200LP.csv','WFC3_UVIS_F218W':'HST/HST.WFC3_UVIS2.F218W.csv',
                          'WFC3_UVIS_F225W':'HST/HST.WFC3_UVIS2.F225W.csv' ,'WFC3_UVIS_F275W':'HST/HST.WFC3_UVIS2.F275W.csv','WFC3_UVIS_F280N':'HST/HST.WFC3_UVIS2.F280N.csv',
                          'WFC3_UVIS_F300X':'HST/HST.WFC3_UVIS2.F300X.csv' ,'WFC3_UVIS_F336W':'HST/HST.WFC3_UVIS2.F336W.csv','WFC3_UVIS_F343N':'HST/HST.WFC3_UVIS2.F343N.csv',
                          'WFC3_UVIS_F350LP':'HST/HST.WFC3_UVIS2.F350LP.csv','WFC3_UVIS_F373N':'HST/HST.WFC3_UVIS2.F373N.csv','WFC3_UVIS_F390M':'HST/HST.WFC3_UVIS2.F390M.csv',
                          'WFC3_UVIS_F390W':'HST/HST.WFC3_UVIS2.F390W.csv' ,'WFC3_UVIS_F395N':'HST/HST.WFC3_UVIS2.F395N.csv','WFC3_UVIS_F410M':'HST/HST.WFC3_UVIS2.F410M.csv',
                          'WFC3_UVIS_F438W':'HST/HST.WFC3_UVIS2.F438W.csv' ,'WFC3_UVIS_F467M':'HST/HST.WFC3_UVIS2.F467M.csv','WFC3_UVIS_F469N':'HST/HST.WFC3_UVIS2.F469N.csv',
                          'WFC3_UVIS_F475W':'HST/HST.WFC3_UVIS2.F475W.csv' ,'WFC3_UVIS_F475X':'HST/HST.WFC3_UVIS2.F475X.csv','WFC3_UVIS_F487N':'HST/HST.WFC3_UVIS2.F487N.csv',
                          'WFC3_UVIS_F502N':'HST/HST.WFC3_UVIS2.F502N.csv' ,'WFC3_UVIS_F547M':'HST/HST.WFC3_UVIS2.F547M.csv','WFC3_UVIS_F555W':'HST/HST.WFC3_UVIS2.F555W.csv',
                          'WFC3_UVIS_F600LP':'HST/HST.WFC3_UVIS2.F600LP.csv','WFC3_UVIS_F606W':'HST/HST.WFC3_UVIS2.F606W.csv','WFC3_UVIS_F621M':'HST/HST.WFC3_UVIS2.F621M.csv',
                          'WFC3_UVIS_F625W':'HST/HST.WFC3_UVIS2.F625W.csv' ,'WFC3_UVIS_F631N':'HST/HST.WFC3_UVIS2.F631N.csv','WFC3_UVIS_F645N':'HST/HST.WFC3_UVIS2.F645N.csv',
                          'WFC3_UVIS_F656N':'HST/HST.WFC3_UVIS2.F656N.csv' ,'WFC3_UVIS_F657N':'HST/HST.WFC3_UVIS2.F657N.csv','WFC3_UVIS_F658N':'HST/HST.WFC3_UVIS2.F658N.csv',
                          'WFC3_UVIS_F665N':'HST/HST.WFC3_UVIS2.F665N.csv' ,'WFC3_UVIS_F673N':'HST/HST.WFC3_UVIS2.F673N.csv','WFC3_UVIS_F680N':'HST/HST.WFC3_UVIS2.F680N.csv',
                          'WFC3_UVIS_F689M':'HST/HST.WFC3_UVIS2.F689M.csv' ,'WFC3_UVIS_F763M':'HST/HST.WFC3_UVIS2.F763M.csv','WFC3_UVIS_F775W':'HST/HST.WFC3_UVIS2.F775W.csv',
                          'WFC3_UVIS_F814W':'HST/HST.WFC3_UVIS2.F814W.csv' ,'WFC3_UVIS_F845M':'HST/HST.WFC3_UVIS2.F845M.csv','WFC3_UVIS_F850LP':'HST/HST.WFC3_UVIS2.F850LP.csv',
                          'WFC3_UVIS_F953N':'HST/HST.WFC3_UVIS2.F953N.csv' ,'WFC3_UVIS_FQ232N':'HST/HST.WFC3_UVIS2.FQ232N.csv','WFC3_UVIS_FQ243N':'HST/HST.WFC3_UVIS2.FQ243N.csv',
                          'WFC3_UVIS_FQ378N':'HST/HST.WFC3_UVIS2.FQ378N.csv','WFC3_UVIS_FQ387N':'HST/HST.WFC3_UVIS2.FQ387N.csv','WFC3_UVIS_FQ422M':'HST/HST.WFC3_UVIS2.FQ422M.csv',
                          'WFC3_UVIS_FQ436N':'HST/HST.WFC3_UVIS2.FQ436N.csv','WFC3_UVIS_FQ437N':'HST/HST.WFC3_UVIS2.FQ437N.csv','WFC3_UVIS_FQ492N':'HST/HST.WFC3_UVIS2.FQ492N.csv',
                          'WFC3_UVIS_FQ508N':'HST/HST.WFC3_UVIS2.FQ508N.csv','WFC3_UVIS_FQ575N':'HST/HST.WFC3_UVIS2.FQ575N.csv','WFC3_UVIS_FQ619N':'HST/HST.WFC3_UVIS2.FQ619N.csv',
                          'WFC3_UVIS_FQ634N':'HST/HST.WFC3_UVIS2.FQ634N.csv','WFC3_UVIS_FQ672N':'HST/HST.WFC3_UVIS2.FQ672N.csv','WFC3_UVIS_FQ674N':'HST/HST.WFC3_UVIS2.FQ674N.csv',
                          'WFC3_UVIS_FQ727N':'HST/HST.WFC3_UVIS2.FQ727N.csv','WFC3_UVIS_FQ750N':'HST/HST.WFC3_UVIS2.FQ750N.csv','WFC3_UVIS_FQ889N':'HST/HST.WFC3_UVIS2.FQ889N.csv',
                          'WFC3_UVIS_FQ906N':'HST/HST.WFC3_UVIS2.FQ906N.csv','WFC3_UVIS_FQ924N':'HST/HST.WFC3_UVIS2.FQ924N.csv','WFC3_UVIS_FQ937N':'HST/HST.WFC3_UVIS2.FQ937N.csv',
                          'WFC3_UVIS_G280'  :'HST/HST.WFC3_UVIS2.G280.csv'  ,'WFPC2_F1042M'    :'HST/HST.WFPC2.F1042M.csv'     ,'WFPC2_F122M'     :'HST/HST.WFPC2.F122M.csv',
                          'WFPC2_F130LP'    :'HST/HST.WFPC2.F130LP.csv'     ,'WFPC2_F157W'     :'HST/HST.WFPC2.F157W.csv'      ,'WFPC2_F160BW'    :'HST/HST.WFPC2.F160BW.csv',
                          'WFPC2_F165LP'    :'HST/HST.WFPC2.F165LP.csv'     ,'WFPC2_F170W'     :'HST/HST.WFPC2.F170W.csv'      ,'WFPC2_F185W'     :'HST/HST.WFPC2.F185W.csv',
                          'WFPC2_F218W'     :'HST/HST.WFPC2.F218W.csv'      ,'WFPC2_F255W'     :'HST/HST.WFPC2.F255W.csv'      ,'WFPC2_F300W'     :'HST/HST.WFPC2.F300W.csv',
                          'WFPC2_F336W'     :'HST/HST.WFPC2.F336W.csv'      ,'WFPC2_F343N'     :'HST/HST.WFPC2.F343N.csv'      ,'WFPC2_F375N'     :'HST/HST.WFPC2.F375N.csv',
                          'WFPC2_F380W'     :'HST/HST.WFPC2.F380W.csv'      ,'WFPC2_F390N'     :'HST/HST.WFPC2.F390N.csv'      ,'WFPC2_F410M'     :'HST/HST.WFPC2.F410M.csv',
                          'WFPC2_F437N'     :'HST/HST.WFPC2.F437N.csv'      ,'WFPC2_F439W'     :'HST/HST.WFPC2.F439W.csv'      ,'WFPC2_F450W'     :'HST/HST.WFPC2.F450W.csv',
                          'WFPC2_F467M'     :'HST/HST.WFPC2.F467M.csv'      ,'WFPC2_F469N'     :'HST/HST.WFPC2.F469N.csv'      ,'WFPC2_F487N'     :'HST/HST.WFPC2.F487N.csv',
                          'WFPC2_F502N'     :'HST/HST.WFPC2.F502N.csv'      ,'WFPC2_F547M'     :'HST/HST.WFPC2.F547M.csv'      ,'WFPC2_F555W'     :'HST/HST.WFPC2.F555W.csv',
                          'WFPC2_F569W'     :'HST/HST.WFPC2.F569W.csv'      ,'WFPC2_F588N'     :'HST/HST.WFPC2.F588N.csv'      ,'WFPC2_F606W'     :'HST/HST.WFPC2.F606W.csv',
                          'WFPC2_F622W'     :'HST/HST.WFPC2.F622W.csv'      ,'WFPC2_F631N'     :'HST/HST.WFPC2.F631N.csv'      ,'WFPC2_F656N'     :'HST/HST.WFPC2.F656N.csv',
                          'WFPC2_F658N'     :'HST/HST.WFPC2.F658N.csv'      ,'WFPC2_F673N'     :'HST/HST.WFPC2.F673N.csv'      ,'WFPC2_F675W'     :'HST/HST.WFPC2.F675W.csv',
                          'WFPC2_F702W'     :'HST/HST.WFPC2.F702W.csv'      ,'WFPC2_F785LP'    :'HST/HST.WFPC2.F785LP.csv'     ,'WFPC2_F791W'     :'HST/HST.WFPC2.F791W.csv',
                          'WFPC2_F814W'     :'HST/HST.WFPC2.F814W.csv'      ,'WFPC2_F850LP'    :'HST/HST.WFPC2.F850LP.csv'     ,'WFPC2_F953N'     :'HST/HST.WFPC2.F953N.csv',
                          'SkyMapper_g'     :'SkyMapper/SkyMapper.SkyMapper.g.csv',
                          'SkyMapper_i'     :'SkyMapper/SkyMapper.SkyMapper.i.csv',
                          'SkyMapper_r'     :'SkyMapper/SkyMapper.SkyMapper.r.csv',
                          'SkyMapper_u'     :'SkyMapper/SkyMapper.SkyMapper.u.csv',
                          'SkyMapper_v'     :'SkyMapper/SkyMapper.SkyMapper.v.csv',
                          'SkyMapper_z'     :'SkyMapper/SkyMapper.SkyMapper.z.csv',
                          'VISTA_Ks'        :'Paranal/Paranal.VISTA.Ks.csv' ,
                          'VISTA_J'         :'Paranal/Paranal.VISTA.J.csv',
                          "Mosaic_g'"       :"KPNO/KPNO.Mosaic.g'.csv",
                          "Mosaic_r'"       :"KPNO/KPNO.Mosaic.r'.csv",
                          'IRAS_12'         :'IRAS/IRAS.IRAS.12mu.csv',
                          'IRAS_25'         :'IRAS/IRAS.IRAS.25mu.csv',
                          'IRAS_60'         :'IRAS/IRAS.IRAS.60mu.csv',
                          'IRAS_100'        :'IRAS/IRAS.IRAS.100mu.csv',
                          'F555W'           :'HST/HST.WFC3_UVIS1.F555W.csv',
                          'F160W'           :'HST/HST.WFC3_IR.F160W.csv',
                          'AKARI_65'        :'AKARI/AKARI.FIS.N60.csv',
                          'AKARI_90'        :'AKARI/AKARI.FIS.WIDE-S.csv'}
    if bandname in file_name.keys():
        file_path=dir_filter_folder+file_name[bandname]
        t = pandas.read_csv(file_path)
        t_dict={'FILTER_W':np.float64(t['filter_w']/1e4),'FILTER_T':np.float64(t['filter_t'])}
        return t_dict
    else:
        key=list(file_name.keys())
        print('--- ERROR ---')
        print("bandname is not supported. This function currently support following bandnames:")
        # print(key[:2])
        # print('SDSS filters : \n',key[2:12])
        # print('2MASS filters: \n',key[12:19])
        # print('UKIRT filters: \n',key[19:31])
        # print('WISE filters: \n',key[31:39])
        # print('Herschel filters: \n',key[39:46])
        # print('Spitzer filters: \n',key[46:60])
        # print('HST WFC3: \n',key[60:61])
        # print('JCMT: \n',key[61:63])
        # print('PAN-STARRS: \n',key[63:70])
        # print('GAIA: \n',key[70:73])
        # print('HST: \n',key[73:287])
        # print('SkyMapper: \n',key[287:293])
        # print('VISTA: \n',key[293:295])
        # print('Mosaic: \n',key[295:297])
        # print('IRAS: \n',key[297:299])
        # print('HST: \n',key[299:301])
        # print('-------------')
        return 0
def cw_fflux_from_spec_bandname(data_w, data_f, bandpass_name):
    band_wave=band_wave_dic[bandpass_name]
    band_tran=band_tran_dic[bandpass_name]
    index  =(data_w>=np.min(band_wave))&(data_w<np.max(band_wave))
    data_w1=data_w[index]
    data_f1=data_f[index]
    # data_w1=np.array(data_w1)
    # data_f1=np.array(data_f1)
    fun=interp1d(band_wave,band_tran, 'linear')
    bp_band_regrid = fun(data_w1)
    bp_band_regrid_pos = (bp_band_regrid + abs(bp_band_regrid)) / 2.
    bp_band_regrid_norm = bp_band_regrid_pos / tsum(data_w1,bp_band_regrid_pos)
    flux_sudo = tsum(data_w1, data_f1 * bp_band_regrid_norm) 
#print, "bandname: " + bandname,  " sum:    ", flux_sudo
    return flux_sudo
def read_filter(filter_list):
    band_wave={}
    band_tran={}
    for i in filter_list:
        if i!='none':
            bp_band = cw_filter_loading(i)
            band_wave[i]=bp_band['FILTER_W']
            band_tran[i]=bp_band['FILTER_T']
        else:
            pass
    return band_wave,band_tran
def cw_sed_load(sed_name):
    '''
    This function return the dictionary which contains 
        sed_name:         The name of the SED template
        sed_ref:          The reference code of the SED template
        sed_description:  The description of the SED template
        sed_wave:         The wavelength of the SED in micron
        sed_flux:         The flux density of the SED in arbitary unit.
        sed_nuLnu:        The nuLnu of the SED in arbitary unit.
    Input:
        sed_name: The name of the desired SED. Currently only following 
              SED are available:
                'Assef10_AGN'
                'Assef10_AGN2'
                'Assef10_E'
                'Assef10_Sbc'
                'Assef10_Im'
            ******* Richards et al. 2006, ApJS
                 Richards06
            ******* SWIRE, Polletta et al. 2006, and other few
                'SWIRE_Arp220'
                'SWIRE_M82'
                'SWIRE_Mrk231'
                'SWIRE_QSO1'
                'SWIRE_QSO2'
                'SWIRE_Torus'
 
    2015 1019       Written by Chao-Wei Tsai, JPL
    2018_0802       Revised to include the units, reference
                    and correct the M82 model
                                    Chao-Wei Tsai, UCLA 
    2019_0215       Add the model "Tsai19_HotDOG"
                                    Chao-Wei Tsai, NAOC
    2020_0226       Modify Chao-Wei Tsai's code so that it can be run by python
                                    LiGuo dong,    NAOC                                
    '''

    sed_name_list = ['Assef10_AGN',  
                     'Assef10_AGN2', 
                     'Assef10_E', 
                     'Assef10_Sbc', 
                     'Assef10_Im', 
                     'Richards06', 
                     'SWIRE_Arp220', 
                     'SWIRE_M82', 
                     'SWIRE_Mrk231', 
                     'SWIRE_QSO1', 
                     'SWIRE_QSO2', 
                     'SWIRE_Torus', 
                     'Tsai19_HotDOG' 
                    ]
# Citations:
    ref_richards06_qso1 = '2006ApJS..166..470R'
    ref_assef10 = '2010ApJ...713..970A'
    ref_swire = '2006ApJ...642..673D,2007ApJ...663...81P'
    ref_tsai19 = '2019_Tsai_To_be_submitted'

# #data directory and filename
    # dir_assef10 =       'E:/SED_template/SED_templates/Assef_models/'
# #dir_richards06 =   '/Users/ctsai/Dropbox/Softwork/IDL/CWT_ASTRO_DATA/SED_templates/'
    # dir_richards06 =    'E:/SED_template/SED_exgal_fits/'
# #dir_swire =        '/Users/ctsai/Dropbox/Softwork/IDL/CWT_ASTRO_DATA/SED_templates/fits_exgal_SED/'
    # dir_swire =         'E:/SED_template/SED_exgal_fits/'
    # dir_grasil =        'E:/SED_template/SED_exgal_fits/'
# #dir_sb99 =         '/Users/ctsai/Dropbox/Softwork/IDL/CWT_ASTRO_DATA/SED_templates/SB99/'
    # dir_tsai19 =        'E:/SED_template/SED_templates/Tsai_models/'

#data filename
    filename_assef10 =      'lrt_templates.dat'
    filename_richards06 =   'QSO1_Richards06.fits'
    filename_arp220 =       'Arp220_SWIRE.fits'
    filename_m82 =          'M82_GRASIL_correct.fits'
    filename_mrk231 =       'Mrk231_SWIRE.fits'
    filename_qso1 =         'QSO1_SWIRE.fits'
    filename_qso2 =         'QSO2_SWIRE.fits'
    filename_torus =        'Torus_SWIRE.fits'
    filename_hotdog =       'HotDOG_SED_LOESS_0p3_5_W2246-0526_z0_3p5e14Lsun.fits'


# Check input, load data and save into a dictionary
    if sed_name in sed_name_list:
        if sed_name=='Assef10_AGN':
            dir = dir_assef10
            filename = filename_assef10
            data=pandas.read_csv(dir + filename,header=None,names=['w_um','Fnu_AGN_m11','Fnu_AGN2_m14','Fnu_E_m16','Fnu_Sbc_m17','Fnu_Im_m15'],skiprows=17,sep='\s+')
            w_um=np.around(np.array(data['w_um']),decimals=4)
            Fnu_mJy=np.around(np.array(data['Fnu_AGN_m11']),decimals=7)*1e-11/1e-26
            sed_str = {'model_name': sed_name, 'wave': w_um, 'wave_unit': "micron", 'Fnu': Fnu_mJy, 'Fnu_unit': "mJy", 'ref': ref_assef10}            
        elif sed_name=='Assef10_AGN2':    
            dir = dir_assef10
            filename = filename_assef10
            data=pandas.read_csv(dir + filename,header=None,names=['w_um','Fnu_AGN_m11','Fnu_AGN2_m14','Fnu_E_m16','Fnu_Sbc_m17','Fnu_Im_m15'],skiprows=17,sep='\s+')
            w_um=np.around(np.array(data['w_um']),decimals=4)
            Fnu_mJy=np.around(np.array(data['Fnu_AGN2_m14']),decimals=7)*1e-14 / 1e-26
            sed_str = {'model_name': sed_name, 'wave': w_um, 'wave_unit': "micron", 'Fnu': Fnu_mJy, 'Fnu_unit': "mJy", 'ref': ref_assef10}            
        elif sed_name=='Assef10_E':
            dir = dir_assef10
            filename = filename_assef10
            data=pandas.read_csv(dir + filename,header=None,names=['w_um','Fnu_AGN_m11','Fnu_AGN2_m14','Fnu_E_m16','Fnu_Sbc_m17','Fnu_Im_m15'],skiprows=17,sep='\s+')
            w_um=np.around(np.array(data['w_um']),decimals=4)
            Fnu_mJy=np.around(np.array(data['Fnu_E_m16']),decimals=7)*1e-16 / 1e-26
            sed_str = {'model_name': sed_name, 'wave': w_um, 'wave_unit': "micron", 'Fnu': Fnu_mJy, 'Fnu_unit': "mJy", 'ref': ref_assef10}                  
        elif sed_name=='Assef10_Sbc':
            dir = dir_assef10
            filename = filename_assef10
            data=pandas.read_csv(dir + filename,header=None,names=['w_um','Fnu_AGN_m11','Fnu_AGN2_m14','Fnu_E_m16','Fnu_Sbc_m17','Fnu_Im_m15'],skiprows=17,sep='\s+')
            w_um=np.around(np.array(data['w_um']),decimals=4)
            Fnu_mJy=np.around(np.array(data['Fnu_Sbc_m17']),decimals=7)*1e-17 / 1e-26
            sed_str = {'model_name': sed_name, 'wave': w_um, 'wave_unit': "micron", 'Fnu': Fnu_mJy, 'Fnu_unit': "mJy", 'ref': ref_assef10}           
        elif sed_name=='Assef10_Im':
            dir = dir_assef10
            filename = filename_assef10
            data=pandas.read_csv(dir + filename,header=None,names=['w_um','Fnu_AGN_m11','Fnu_AGN2_m14','Fnu_E_m16','Fnu_Sbc_m17','Fnu_Im_m15'],skiprows=17,sep='\s+')
            w_um=np.around(np.array(data['w_um']),decimals=4)
            Fnu_mJy=np.around(np.array(data['Fnu_Im_m15']),decimals=7)*1e-15 / 1e-26
            sed_str = {'model_name': sed_name, 'wave': w_um, 'wave_unit': "micron", 'Fnu': Fnu_mJy, 'Fnu_unit': "mJy", 'ref': ref_assef10}            
        elif sed_name=='Richards06':
            dir = dir_richards06
            filename = filename_richards06
            hdu=fits.open(dir + filename)
            hdr=hdu[0].header
            data=hdu[0].data 
            w_um=data[:,0]
            Fnu_mJy =data[:,1]
            sed_str = {'model_name': sed_name, 'wave': w_um, 'wave_unit': "micron", 'Fnu': Fnu_mJy, 'Fnu_unit': "mJy", 'ref': ref_richards06_qso1}            
        elif sed_name=='SWIRE_Arp220':
            dir = dir_swire
            filename = filename_arp220   
            hdu=fits.open(dir + filename) 
            hdr=hdu[0].header
            data=hdu[0].data 
            w_um=data[:,0]
            Fnu_mJy =data[:,1]
            sed_str = {'model_name': sed_name, 'wave': w_um, 'wave_unit': "micron", 'Fnu': Fnu_mJy, 'Fnu_unit': "mJy", 'ref': ref_swire}    
        elif sed_name=='SWIRE_M82':
            dir = dir_swire
            filename = filename_m82        
            hdu=fits.open(dir + filename) 
            hdr=hdu[0].header
            data=hdu[0].data 
            w_um=data[:,0]
            Fnu_mJy =data[:,1]
            sed_str = {'model_name': sed_name, 'wave': w_um, 'wave_unit': "micron", 'Fnu': Fnu_mJy, 'Fnu_unit': "mJy", 'ref': ref_swire} 
        elif sed_name=='SWIRE_Mrk231':
            dir = dir_swire
            filename = filename_mrk231
            hdu=fits.open(dir + filename)
            hdr=hdu[0].header
            data=hdu[0].data 
            w_um=data[:,0]
            Fnu_mJy =data[:,1]
            sed_str = {'model_name': sed_name, 'wave': w_um, 'wave_unit': "micron", 'Fnu': Fnu_mJy, 'Fnu_unit': "mJy", 'ref': ref_swire} 
        elif sed_name=='SWIRE_QSO1':
            dir = dir_swire
            filename = filename_qso1
            hdu=fits.open(dir + filename)
            hdr=hdu[0].header
            data=hdu[0].data 
            w_um=data[:,0]
            Fnu_mJy =data[:,1]
            sed_str = {'model_name': sed_name, 'wave': w_um, 'wave_unit': "micron", 'Fnu': Fnu_mJy, 'Fnu_unit': "mJy", 'ref': ref_swire}
        elif sed_name=='SWIRE_QSO2':
            dir = dir_swire
            filename = filename_qso2
            hdu=fits.open(dir + filename)
            hdr=hdu[0].header
            data=hdu[0].data 
            w_um=data[:,0]
            Fnu_mJy =data[:,1]
            sed_str = {'model_name': sed_name, 'wave': w_um, 'wave_unit': "micron", 'Fnu': Fnu_mJy, 'Fnu_unit': "mJy", 'ref': ref_swire}
        elif sed_name=='SWIRE_Torus':
            dir = dir_swire
            filename = filename_torus
            hdu=fits.open(dir + filename)
            hdr=hdu[0].header
            data=hdu[0].data 
            w_um=data[:,0]
            Fnu_mJy =data[:,1]
            sed_str = {'model_name': sed_name, 'wave': w_um, 'wave_unit': "micron", 'Fnu': Fnu_mJy, 'Fnu_unit': "mJy", 'ref': ref_swire}
        else:
            dir = dir_tsai19
            filename = filename_hotdog
            hdu=fits.open(dir + filename)
            hdr=hdu[0].header
            data=hdu[0].data 
            w_um=data[:,0]
            Fnu_mJy =data[:,1]
            sed_str = {'model_name': sed_name, 'wave': w_um, 'wave_unit': "micron", 'Fnu': Fnu_mJy, 'Fnu_unit': "mJy", 'ref': ref_tsai19}
    else:
        print('--- ERROR ---')
        print('Syntax - return_dictionary = cw_sed_load(SED_NAME)')
        print("Currently supports following templates:")
        print(sed_name_list)
        print('-------------')
        exit(0)
    return sed_str
def tsum(x,y):
    v= integrate.trapz(y,x)
    return v
def none_filter(data_w, data_f,w_width):
    flux_sudo = tsum(data_w, data_f/w_width) 
    return flux_sudo
def planck(wav, T):
    # SI unit, output = W * sr^-1 * m^-2 Hz^-1
    h = 6.626e-34
    c = 3.0e+8
    k_b = 1.38e-23
    a = 2.0*h*c**2
    b = h*c/(wav*k_b*T)
    B_lambda = a/ ( (wav**5) * (np.exp(b) - 1.0) )
    B_nu = B_lambda * wav**2/c
    return B_nu  # in W * sr^-1 * m^-2 Hz^-1
def planck_um_Jy(wav_um,T):
    wav = wav_um * 1e-6         # micron -> meter
    output = planck(wav,T) * 4. * np.pi * 1e26      # in Jy
    return np.float64(output)
def greybody(wav_um, T, beta):
    Qem = 3./1300. * (125./850.)**beta * (850./wav_um)**beta
    greybody = Qem * planck_um_Jy(wav_um,T)
    return np.float64(greybody)    
def greybody_calculate(wav_um, T, beta,l):
    # wave              =np.arange(0.01,1000,1)
    # Bv                =greybody(wave, T, beta)
    Bv_band           =[]
    for i in range(len(filter)):
        if filter[i]=='none':
            w_width   =0.1*wav_um[i]
            wave_filter=np.linspace(0.95*wav_um[i],1.05*wav_um[i],100)
            Bv_filter  =greybody(wave_filter, T, beta)
            Bv_band.append(none_filter(wave_filter*(1+redshift), Bv_filter,w_width))
        else:
            wave_filter=np.linspace(0.7*wav_um[i],1.3*wav_um[i],50)
            Bv_filter  =greybody(wave_filter, T, beta)
            Bv_band.append(cw_fflux_from_spec_bandname(wave_filter*(1+redshift),Bv_filter,filter[i]))
    Bv_band           =np.array(Bv_band)
    vLv               =Bv_band*con.c.value/(wav_um*(1+redshift)*1e-6)
    vLv_sun           =vLv*(10**(-26))/con.L_sun.value
    vLv_sun_norm      =vLv_sun/sum(vLv_sun)
    # Bv                =greybody(wav_um, T, beta)
    # vLv               =Bv*con.c.value/(wav_um*1e-6)
# #    vLv               =(l**2)*Bv*con.c.value/(wav_um*1e-6)
    # vLv_sun           =vLv*(10**(-26))/con.L_sun.value
    # vLv_sun_norm      =vLv_sun/sum(vLv_sun)
    # return np.float64(l*vLv_sun_norm)
    return np.float64(l*vLv_sun_norm)
def cw_SPL_intfun(x, alpha):
    return (x**(-1. * alpha - 2.)/(np.exp(x)-1))
def cw_SPL_int(alpha,Tc, Th,v):
    xc                = con.h.value*v/(con.k_B.value*Tc)
    xh                = con.h.value*v/(con.k_B.value*Th)
    result            = integrate.quad(cw_SPL_intfun,xh,xc,args=(alpha))
    return result[0]
def dust(theta,wave_rest):
    b,alpha,T_min,T_max=theta
    tem_wave          =np.zeros(len(wave_rest)+2)
    tem_wave[1:-1]    =wave_rest
    tem_wave[0]       =0.5*wave_rest[0]
    tem_wave[-1]      =1.5*wave_rest[-1]
    # tem_wave          =wave_rest
    v                 =con.c.value/(tem_wave*1e-6)
    tem_Lv           =[]
    for i in v:
        a             =i**(alpha+5.5)
        S             =cw_SPL_int(alpha,T_min,T_max,i)
        tem_Lv.append(a*S)
    tem_Lv           =np.array(tem_Lv)
    fun               =interp1d(tem_wave,tem_Lv, 'linear')
    tem_Lv_band=[]
    for i in range(len(filter)):
        if filter[i]=='none':
            w_width   =0.1*wave_rest[i]
            wave_filter=np.linspace(0.95*wave_rest[i],1.05*wave_rest[i],100)
            Lv_filter =fun(wave_filter)
            tem_Lv_band.append(none_filter(wave_filter*(1+redshift), Lv_filter,w_width))
        else:
            wave_filter=np.linspace(0.7*wave_rest[i],1.3*wave_rest[i],100)
            Lv_filter =fun(wave_filter)
            tem_Lv_band.append(cw_fflux_from_spec_bandname(wave_filter*(1+redshift),Lv_filter,filter[i]))
    tem_Lv_band=np.array(tem_Lv_band)
    tem_vLv_band=tem_Lv_band*v[1:-1]/(1+redshift)
    tem_vLv_norm      =b*tem_vLv_band/sum(tem_vLv_band)
    return tem_vLv_norm
def Assef10_model(theta,tem_control,Av_control,tem_vLv,wave_rest):
    template_name     =np.array(['Assef10_AGN','Assef10_AGN2','Assef10_E','Assef10_Sbc','Assef10_Im'])
    control_bool      =[bool(i) for i in tem_control[:5]]
    choice_tem        =template_name[control_bool]
    tem_number        =sum(tem_control[:5])
    Av_number         =sum(Av_control)
    if tem_number==1 and Av_number==1:
        a,Av          =theta
        if a>0.0 and Av>0.0 and Av<Av_1_max:
            bdy_value =0.0
        else:
            bdy_value =-np.inf
        model         =a*apply(fm07(wave_rest*(10**4),Av),tem_vLv[choice_tem[0]])
    elif tem_number==1 and Av_number==0:
        a             =theta[0]
        if a>0.0:
            bdy_value =0.0
        else:
            bdy_value =-np.inf
        model         =a*tem_vLv[choice_tem[0]]
    elif tem_number==2 and Av_number==0:
        a,b           =theta
        if a>0.0 and b>0.0:
            bdy_value =0.0
        else:
            bdy_value =-np.inf
        model         =a*tem_vLv[choice_tem[0]]+b*tem_vLv[choice_tem[1]]
    elif tem_number==2 and Av_number==1:
        a,b,Av        =theta
        if a>0.0 and b>0.0 and Av>0.0 and Av<Av_1_max:
            bdy_value =0.0
        else:
            bdy_value =-np.inf
        model         =a*apply(fm07(wave_rest*(10**4),Av),tem_vLv[choice_tem[0]])+b*tem_vLv[choice_tem[1]]
    elif tem_number==2 and Av_number==2:
        a,b,Av1,Av2   =theta
        if a>0.0 and b>0.0 and Av1>0.0 and Av2>0.0:
            bdy_value =0.0
        else:
            bdy_value =-np.inf
        model         =a*apply(fm07(wave_rest*(10**4),Av1),tem_vLv[choice_tem[0]])+b*apply(fm07(wave_rest*(10**4),Av2),tem_vLv[choice_tem[1]])
    elif tem_number==3 and Av_number==0:
        a,b,c         =theta
        if a>0.0 and b>0.0 and c>0.0:
            bdy_value =0.0
        else:
            bdy_value =-np.inf
        model         =a*tem_vLv[choice_tem[0]]+b*tem_vLv[choice_tem[1]]+c*tem_vLv[choice_tem[2]]
    elif tem_number==3 and Av_number==1:
        a,b,c,Av      =theta
        if a>0.0 and b>0.0 and c>0.0 and Av>0.0 and Av<Av_1_max:
            bdy_value =0.0
        else:
            bdy_value =-np.inf
        model         =a*apply(fm07(wave_rest*(10**4),Av),tem_vLv[choice_tem[0]])+b*tem_vLv[choice_tem[1]]+c*tem_vLv[choice_tem[2]]
    elif tem_number==3 and Av_number==2:
        a,b,c,Av1,Av2 =theta
        if a>0.0 and b>0.0 and c>0.0 and Av1>0.0 and Av2>0.0 and Av1<Av_1_max and Av2<Av_2_max:
            bdy_value =0.0
        else:
            bdy_value =-np.inf
        model         =a*apply(fm07(wave_rest*(10**4),Av1),tem_vLv[choice_tem[0]])+b*apply(fm07(wave_rest*(10**4),Av2),tem_vLv[choice_tem[1]])+c*tem_vLv[choice_tem[2]]
    elif tem_number==4 and Av_number==0:
        a,b,c,d       =theta
        if a>0.0 and b>0.0 and c>0.0 and d>0.0:
            bdy_value =0.0
        else:
            bdy_value =-np.inf
        model         =a*tem_vLv[choice_tem[0]]+b*tem_vLv[choice_tem[1]]+c*tem_vLv[choice_tem[2]]+d*tem_vLv[choice_tem[3]]
    elif tem_number==4 and Av_number==1:
        a,b,c,d,Av    =theta
        if a>0.0 and b>0.0 and c>0.0 and d>0.0 and Av>0.0 and Av<Av_1_max:
            bdy_value =0.0
        else:
            bdy_value =-np.inf
        model         =a*apply(fm07(wave_rest*(10**4),Av),tem_vLv[choice_tem[0]])+b*tem_vLv[choice_tem[1]]+c*tem_vLv[choice_tem[2]]+d*tem_vLv[choice_tem[3]]
    elif tem_number==4 and Av_number==2:
        a,b,c,d,e,f   =theta
        if a>0.0 and b>0.0 and c>0.0 and d>0.0 and e>0.0 and f>0.0 and e<Av_1_max and f<Av_2_max:
            bdy_value =0.0
        else:
            bdy_value =-np.inf
        model         =a*apply(fm07(wave_rest*(10**4),e),tem_vLv[choice_tem[0]])+b*apply(fm07(wave_rest*(10**4),f),tem_vLv[choice_tem[1]])+c*tem_vLv[choice_tem[2]]+d*tem_vLv[choice_tem[3]]
    elif tem_number==5:
        a,b,c,d,e,f,h =theta
        if a>0.0 and b>0.0 and c>0.0 and d>0.0 and e>0.0 and f>0.0 and h>0.0 and f<Av_1_max and h<Av_2_max:
            bdy_value =0.0
        else:
            bdy_value =-np.inf
        model         =a*apply(fm07(wave_rest*(10**4),f),tem_vLv[choice_tem[0]])+b*apply(fm07(wave_rest*(10**4),h),tem_vLv[choice_tem[1]])+c*tem_vLv[choice_tem[2]]+d*tem_vLv[choice_tem[3]]+e*tem_vLv[choice_tem[4]]
    else:
        print('The number of Assef template exceed 5.')
        os._exit(0)
    return model,bdy_value
def fitting_model(theta,tem_control,Av_control,tem_vLv,wave_rest):
    if sum(tem_control[:5])!=0 and tem_control[5]==0 and tem_control[6]==0:
        model,bdy_value=Assef10_model(theta,tem_control,Av_control,tem_vLv,wave_rest)
        return model,bdy_value
    elif sum(tem_control[:5])==0 and tem_control[5]==1 and tem_control[6]==0:
        T,beta,l          =theta
        if T>Grey_Tmin and beta>Grey_beta_min and l>Grey_Smin and l<Grey_Smax and beta<Grey_beta_max and T<Grey_Tmax:
            Grey_model    =greybody_calculate(wave_rest, T, beta,l)
            bdy_value     =0.0
        else:
            Grey_model    =-np.inf
            bdy_value     =-np.inf
        return Grey_model,bdy_value
    elif sum(tem_control[:5])==0 and tem_control[5]==0 and tem_control[6]==1:
        a,alpha,T_min,T_max=theta
        if a>Dust_cons_min and a<Dust_cons_max and alpha>Dust_alpha_min and alpha<Dust_alpha_max \
        and T_min>Dust_T_min and T_max<Dust_T_max and T_min<T_max:
            dust_model    =dust(theta,wave_rest)
            bdy_value     =0.0
        else:
            dust_model    =-np.inf
            bdy_value     =-np.inf
        return dust_model,bdy_value
    elif sum(tem_control[:5])!=0 and tem_control[5]==1 and tem_control[6]==0:
        As_model,bdy_value=Assef10_model(theta[:-3],tem_control,Av_control,tem_vLv,wave_rest)
        T,beta,l          =theta[-3:]
        if not np.isfinite(bdy_value):
            return As_model,bdy_value
        else:
            if T>Grey_Tmin and beta>Grey_beta_min and l>Grey_Smin and l<Grey_Smax and beta<Grey_beta_max and T<Grey_Tmax:
                Grey_model=greybody_calculate(wave_rest, T, beta,l)
                index     =wave_rest<=1.0
                Grey_model[index]=0*Grey_model[index]
                model     =Grey_model+As_model
            else:
                bdy_value =-np.inf
                model     =-np.inf
            return model,bdy_value
    elif sum(tem_control[:5])!=0 and tem_control[5]==0 and tem_control[6]==1:
        As_model,bdy_value=Assef10_model(theta[:-4],tem_control,Av_control,tem_vLv,wave_rest)
        a,alpha,T_min,T_max=theta[-4:]
        if not np.isfinite(bdy_value):
            return As_model,bdy_value
        else:
            if a>Dust_cons_min and a<Dust_cons_max and alpha>Dust_alpha_min and alpha<Dust_alpha_max\
               and T_min>Dust_T_min and T_max<Dust_T_max and T_min<T_max:
                dust_model=dust(theta[-4:],wave_rest)
                index     =wave_rest<=1.0
                dust_model[index]=0*dust_model[index]
                model     =dust_model+As_model
            else:
                bdy_value =-np.inf
                model     =-np.inf
            return model,bdy_value
    elif sum(tem_control[:5])==0 and tem_control[5]==1 and tem_control[6]==1:
        T,beta,l,dust_c,alpha,T_min,T_max=theta
        if T>Grey_Tmin and beta>Grey_beta_min and l>Grey_Smin and l<Grey_Smax and beta<Grey_beta_max and \
        T<Grey_Tmax and dust_c>Dust_cons_min and dust_c<Dust_cons_max and alpha<Dust_alpha_max \
        and alpha>Dust_alpha_min and T_min>Dust_T_min and T_max<Dust_T_max and T_min<T_max:
            bdy_value     =0.0
            dust_model    =dust(theta[-4:],wave_rest)
            Grey_model    =greybody_calculate(wave_rest, T, beta,l)
            model         =Grey_model+dust_model
        else:
            bdy_value     =-np.inf
            model         =-np.inf
        return model,bdy_value
    else:
        As_model,bdy_value=Assef10_model(theta[:-7],tem_control,Av_control,tem_vLv,wave_rest)
        T,beta,l,dust_c,alpha,T_min,T_max=theta[-7:]
        if not np.isfinite(bdy_value):
            return As_model,bdy_value
        else:
            if T>Grey_Tmin and beta>Grey_beta_min and l>Grey_Smin and l<Grey_Smax and beta<Grey_beta_max and\
            T<Grey_Tmax and dust_c>Dust_cons_min and dust_c<Dust_cons_max and alpha<Dust_alpha_max and \
            alpha>Dust_alpha_min and T_min>Dust_T_min and T_max<Dust_T_max and T_min<T_max:
                Grey_model=greybody_calculate(wave_rest, T, beta,l)
                dust_model=dust(theta[-4:],wave_rest)
                index     =wave_rest<=1.0
                Grey_model[index]=0*Grey_model[index]
                dust_model[index]=0*dust_model[index]
                model     =Grey_model+As_model+dust_model
            else:
                model     =-np.inf
                bdy_value =-np.inf
            return model,bdy_value
def ln_prior(theta,tem_control,Av_control,vLv_norm,tem_vLv,wave_rest,up_limit):
    model_vLv,b_value =fitting_model(theta,tem_control,Av_control,tem_vLv,wave_rest)
    if not np.isfinite(b_value):
        b_value           =-np.inf
        model_vLv         =-np.inf
        return model_vLv, b_value
    else:
        if len(up_limit)==0:
            b_value       =0.0
            return model_vLv,b_value
        else:
            up_vLv        =vLv_norm[up_limit]
            up_tem        =model_vLv[up_limit]
            bool_value    =(up_tem<=up_vLv)
            if bool_value.all()==False:
                b_value   =-np.inf
                model_vLv =-np.inf
                return model_vLv,b_value
            else:
                b_value   =0.0
                return model_vLv,b_value
def ln_likelihood(vLv_norm,vLv_e_norm,up_limit,vLv_tem):
    if len(up_limit)==0:
        model         =vLv_tem
        sigma2        =vLv_e_norm** 2
        return -0.5 * np.sum(((vLv_norm-model)**2)/sigma2 )
    else:
        model         =vLv_tem
        model_vLv1    =np.delete(model,up_limit)
        vLv_norm1     =np.delete(vLv_norm,up_limit)
        vLv_e_norm1   =np.delete(vLv_e_norm,up_limit)
        sigma2        =vLv_e_norm1** 2
        return -0.5 * np.sum(((vLv_norm1-model_vLv1)**2)/sigma2 )
def ln_probability(theta,tem_control,Av_control,tem_vLv,vLv_norm,vLv_e_norm,wave_rest,up_limit):
    vLv_tem,lp        = ln_prior(theta,tem_control,Av_control,vLv_norm,tem_vLv,wave_rest,up_limit)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + ln_likelihood(vLv_norm,vLv_e_norm,up_limit,vLv_tem)
def source_calculate(source_dict):
# H0=70 km/s/Kpc, OmO=0.3
    redshift          =source_dict['redshift']
    wave              =source_dict['wave']
    flux              =source_dict['flux']
    flux_e            =source_dict['flux_e']
    cosmo             =FlatLambdaCDM(H0=70, Om0=0.3)
    distance          =(10**6)*(cosmo.luminosity_distance(redshift).value)*(con.pc.value)
    vfv               =flux*con.c.value/(wave*1e-6)
    vfv_e             =flux_e*con.c.value/(wave*1e-6)
    if distance==0.0:
        vfv           =vfv*(10**(-3))*(10**(-26))/con.L_sun.value
        vfv_e         =vfv_e*(10**(-3))*(10**(-26))/con.L_sun.value
        source_cdict  ={'redshift':redshift,'vLv':vfv,'vLv_e':vfv_e,'wave':wave}
        return source_cdict
    else:
        vLv           =4*np.pi*distance*distance*vfv*(10**(-3))*(10**(-26))/con.L_sun.value
        vLv_e         =4*np.pi*distance*distance*vfv_e*(10**(-3))*(10**(-26))/con.L_sun.value
        source_cdict  ={'redshift':redshift,'vLv':vLv,'vLv_e':vLv_e,'wave':wave}
        return source_cdict 
def template_cal(source_dict,tem_control):
    template_name     =np.array(['Assef10_AGN','Assef10_AGN2','Assef10_E','Assef10_Sbc','Assef10_Im'])
    control_bool      =[bool(i) for i in tem_control]
    choice_tem        =template_name[control_bool]
    filter_name       =source_dict['filter_list']
    redshift          =source_dict['redshift']
    wave              =source_dict['wave']
    wave_rest         =wave/(1+redshift)
    tem_vLv           ={}
    for i in choice_tem:
        template      =cw_sed_load(i)
        flux          =template['Fnu']
        tem_wave      =template['wave']
        vLv           =flux*con.c.value*(10**(-3))*(10**(-26))/((tem_wave*10**(-6))*con.L_sun.value )
        vLv_band      =[]
        for j in range(len(filter_name)):
            if np.max(tem_wave)<wave_rest[j]:
                vLv_band.append(0)
            else:
                if filter_name[j]=='none':
                    w_width      =0.1*wave_rest[j]
                    index        =(tem_wave>=0.95*wave_rest[j])&(tem_wave<1.05*wave_rest[j])
                    w            =tem_wave[index]
                    f            =flux[index]
                    vLv_band.append(none_filter(w*(1+redshift),f,w_width))
                else:
                    vLv_band.append(cw_fflux_from_spec_bandname(tem_wave*(1+redshift),flux,filter_name[j]))
        vLv_band      =np.array(vLv_band)*con.c.value*(10**(-3))*(10**(-26))/((wave*10**(-6))*con.L_sun.value )
        tem_vLv[i]    =vLv_band/sum(vLv)
    return tem_vLv
def data_for_mc(tem_control,Av_control,source_dict):
    global filter
    global band_wave_dic
    global band_tran_dic
    source_cdict      =source_calculate(source_dict)
    up_limit          =source_dict['up_limit']
    wave              =source_dict['wave']
    redshift          =source_dict['redshift']
    filter_list       =source_dict['filter_list']
    vLv               =source_cdict['vLv']
    vLv_e             =source_cdict['vLv_e']
    wave_rest         =wave/(1+redshift)
    vLv_norm          =vLv/sum(vLv)
    vLv_e_norm        =vLv_e/sum(vLv)
    band_wave_dic,band_tran_dic=read_filter(filter_list)
    if tem_control[5] ==1 and tem_control[6] ==0 and sum(tem_control[:5])!=0:
        tem_vLv       =template_cal(source_dict,tem_control[:5])
        par_number    =sum(tem_control)+sum(Av_control)+2
    elif tem_control[5] ==0 and tem_control[6] ==1 and sum(tem_control[:5])!=0:
        tem_vLv       =template_cal(source_dict,tem_control[:5])
        par_number    =sum(tem_control)+sum(Av_control)+3
    elif tem_control[5] ==1 and tem_control[6] ==1 and sum(tem_control[:5])!=0:
        tem_vLv       =template_cal(source_dict,tem_control[:5])
        par_number    =sum(tem_control)+sum(Av_control)+5
    elif tem_control[5] ==0 and tem_control[6] ==0 and sum(tem_control[:5])!=0:
        index         =wave_rest<=29.6586
        wave_rest     =wave_rest[index]
        vLv_norm      =vLv_norm[index]
        vLv_e_norm    =vLv_e_norm[index]
        up_limit_new  =[]
        for i in up_limit:
            if index[int(i)]==True:
                up_limit_new.append(int(i))
            else:
                pass
        up_limit      =np.array(up_limit_new)
        s_new_dict    ={}
        s_new_dict['wave']       =wave[index]
        filter_list              =source_dict['filter_list']
        s_new_dict['filter_list']=filter_list[index]
        s_new_dict['redshift']   =redshift
        tem_vLv       =template_cal(s_new_dict,tem_control[:5])
        par_number    =sum(tem_control)+sum(Av_control)
        filter_list   =filter_list[index]
    elif tem_control[5] ==1 and tem_control[6] ==0 and sum(tem_control[:5])==0:
        index         =wave_rest>1.0
        wave_rest     =1.0*wave_rest[index]
        tem_vLv       =None
        vLv_norm      =1.0*vLv_norm[index]
        vLv_e_norm    =1.0*vLv_e_norm[index]
        filter_list   =filter_list[index]
        par_number    =3
        l             =len(index)-len(wave_rest)
        up_limit      =np.array([i-l for i in up_limit if (i-1)>=0])
    elif tem_control[5] ==0 and tem_control[6] ==1 and sum(tem_control[:5])==0:
        index         =wave_rest>1.0
        wave_rest     =1.0*wave_rest[index]
        tem_vLv       =None
        vLv_norm      =1.0*vLv_norm[index]
        vLv_e_norm    =1.0*vLv_e_norm[index]
        filter_list   =filter_list[index]
        par_number    =4
        l             =len(index)-len(wave_rest)
        up_limit      =np.array([i-l for i in up_limit if (i-1)>=0])
    elif tem_control[5] ==1 and tem_control[6] ==1 and sum(tem_control[:5])==0:
        index         =wave_rest>1.0
        wave_rest     =1.0*wave_rest[index]
        tem_vLv       =None
        vLv_norm      =1.0*vLv_norm[index]
        vLv_e_norm    =1.0*vLv_e_norm[index]
        filter_list   =filter_list[index]
        par_number    =7
        l             =len(index)-len(wave_rest)
        up_limit      =np.array([i-l for i in up_limit if (i-1)>=0])
    else:
        pass
    filter            =filter_list
    if sum(tem_control)==0:
        print('The number of parameter is 0. Please check the tem_control you input')
        os._exit(0)
    elif (tem_control[0]==0 and Av_control[0]==1) or (tem_control[1]==0 and Av_control[1]==1):
        print("You don't use the tamplate of AGN, but you added extinction to the model.")
        print('Please check the tem_control and Av_control.')
        os._exit(0)
    else:
        pass
    if (len(vLv_norm)-len(up_limit)-par_number)<1:
        print('The points used for fitting are '+ str(len(vLv_norm)-len(up_limit))+'.')
        print('But the number of parameter is '+str(par_number)+'. You should reduce the parameters or expand the data points')
        os._exit(0)
    else:
        pass
    return tem_vLv,vLv_norm,vLv_e_norm,wave_rest,up_limit,par_number
def par_initial(tem_control,up_limit,Av_control,vLv_norm,wave_rest,walkers_num):
    AGN1_c        =np.random.uniform(0,1,walkers_num)
    AGN2_c        =np.random.uniform(0,1,walkers_num)
    E_c           =np.random.uniform(0,1,walkers_num)
    Sbc_c         =np.random.uniform(0,1,walkers_num)
    Im_c          =np.random.uniform(0,1,walkers_num)
    Av_1          =np.random.uniform(3,30,walkers_num)
    Av_2          =np.random.uniform(0,3,walkers_num)
    T_grey        =np.random.uniform(5,300,walkers_num)
    Beta_grey     =np.random.uniform(0.1,3,walkers_num)
    alpha_dust    =np.random.uniform(-8,-4,walkers_num)
    T_min_dust    =np.random.uniform(3*(1+redshift),200,walkers_num)
    T_max_dust    =np.random.uniform(500,1000,walkers_num)
#    S_grey        =np.random.uniform(0,100,walkers_num)
    S_grey        =[]
    if len(up_limit)==0:
        ratio_grey =tsum(wave_rest,vLv_norm)/tsum(wave_rest,greybody_calculate(wave_rest,45,1,1))
        S_grey     =np.random.uniform(0.01*ratio_grey,ratio_grey,walkers_num)
    else:
        ratio_grey =np.min(vLv_norm[up_limit])/np.max(greybody_calculate(wave_rest,45,1,1))
        S_grey     =np.random.uniform(0.01*ratio_grey,ratio_grey,walkers_num)
#    S_grey        =np.random.uniform(ratio_g,100*ratio_g,walkers_num)
    D_dust        =[]
    for i in range(len(alpha_dust)):
        if len(up_limit)==0:
            ratio_sca =tsum(wave_rest,vLv_norm)/tsum(wave_rest,dust([1,alpha_dust[i],T_min_dust[i],T_max_dust[i]],wave_rest))
            D_dust.append(ratio_sca)
        else:
            ratio_sca =np.min(vLv_norm[up_limit])/np.max(dust([1,alpha_dust[i],T_min_dust[i],T_max_dust[i]],wave_rest))
            D_dust.append(ratio_sca)
    D_dust        =np.array(D_dust)
    list_control      =list(tem_control[:5])+list(Av_control)+list(tem_control[5:6])*3+list(tem_control[6:])*4
    control_bool      =[bool(i) for i in list_control]
    pos_initial       =[AGN1_c,AGN2_c,E_c,Sbc_c,Im_c,Av_1,Av_2,T_grey,Beta_grey,S_grey,D_dust,alpha_dust,T_min_dust,T_max_dust]
    pos_1             =tuple([pos_initial[i] for i in range(14) if control_bool[i]==True])
    pos_2             =np.vstack(pos_1)
    pos               =pos_2.T
    return pos
def emcmc(tem_control,Av_control,source_dict,global_dict,index):
    warnings.filterwarnings("ignore")
    global dir_filter_folder
    global dir_assef10
    global Grey_Tmin
    global Grey_Tmax
    global Grey_Smin
    global Grey_Smax
    global Grey_beta_min
    global Grey_beta_max
    global Dust_cons_min
    global Dust_cons_max
    global Dust_alpha_min
    global Dust_alpha_max
    global redshift
    global Av_1_max
    global Av_2_max
    global Dust_T_min
    global Dust_T_max
    Grey_Tmin,Grey_Tmax,Grey_Smin,Grey_Smax,\
    Grey_beta_min,Grey_beta_max,Dust_cons_min,\
    Dust_cons_max,Dust_alpha_min,\
    Dust_alpha_max,Av_1_max,Av_2_max,Dust_T_min,Dust_T_max=global_dict['par']
    dir_filter_folder=global_dict['dir_filter_folder']
    dir_assef10   =global_dict['dir_assef10']
    tem_vLv,vLv_norm,vLv_e_norm,wave_rest,up_limit,\
    par_number    =data_for_mc(tem_control,Av_control,source_dict)
    walkers_num   =int(global_dict['walker'])
    step          =int(global_dict['step'])
    redshift      =source_dict['redshift']
    pos           =par_initial(tem_control,up_limit,Av_control,vLv_norm,wave_rest,walkers_num)
    sampler       =emcee.EnsembleSampler(walkers_num, par_number, ln_probability, args=(tem_control,Av_control,tem_vLv,vLv_norm,vLv_e_norm,wave_rest,up_limit))
    pos,prob,state=sampler.run_mcmc(pos, 500,progress=False)
    sampler.reset()
    if index==0:
        for i in tqdm(sampler.sample(pos, iterations=step),total=step,ascii=True,leave=True):
            pass
    else:
        sampler.run_mcmc(pos, step,progress=False)
    samplers      =sampler.get_chain(flat=True)
    return samplers
if __name__=='__main__':
#########Process 1.
    redshift          =2.452
    flux              =np.array([1.64e-03,2.25e-03,6.60e-03,1.57e-02,1.93e-02,2.52e-02,2.20e+00,1.50e+01,1.00e+02,4.70e+01,\
                                 6.00e+02,7.10e+01,5.10e+01,3.30e+01,3.50e+01,2.00e+00,2.40e+00,1.75e-01,3.50e-01,5.60e-01])
    flux_error        =np.array([7.40e-05,1.10e-04,2.10e-03,3.20e-03,1.90e-03,2.40e-03,1.00e-01,1.00e+00,2.00e+01,3.00e+00,\
                                 1.20e+02,5.00e+00,1.00e+01,9.00e+00,7.00e+00,8.00e-01,4.80e-01,3.50e-02,1.30e-01,7.00e-02])
    filter_list       =np.array(["Mosaic_g'","Mosaic_r'",'J','TWOMASS_K','IRAC1','IRAC2','WISE3','WISE4','IRAS_60','PACS070',\
                                'IRAS_100','PACS160','SPIRE250','none','none','SCUBA2_850','none','none','none','none'])
    wave              =np.array([0.48,0.63,1.25,2.15,3.6,4.5,12.,22.,60,70.,100,160.,250.,350.,450,850.,1100,9700,37800,67700])
    up_limit          =np.array([8,10,14,15,16,17])
    sourcename        ='W1814+3412'
    # redshift          =0.414
    # flux              =np.array([23.40779974,32.24629836,49.14360034,75.95589705,214.832995,546.7606721,2720.026674,54389.02218,157087.248])*0.001
    # flux_error        =np.array([0.506943024,0.669527026,0.631397995,1.423980052,4.116429864,13.24393115,53.71769508,816.3849187,2756.598957])*0.001
    # filter_list       =np.array(['PS1_g','PS1_r','PS1_i','PS1_z','PS1_y','W1','W2','W3','W4'])
    # wave              =np.array([0.49372216,0.621785696,0.750868102,0.86483421,0.955699531,3.338606372,4.533548898,10.36215052,21.60959703])
    # up_limit          =np.array([])
    # sourcename        ='W1904+4853'
    tem_control       =[1,0,1,1,1,0,1]
    Av_control        =[1,0]
    m=10
#########Process 2.
    dir_filter_folder = 'E:/Filter/sorted/'
    dir_assef10       = 'E:/SED_templates/Assef_models/'
    save_path         = 'E:/Code/SED fitting/result/'
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
    global_dict         ={'dir_filter_folder':dir_filter_folder,'dir_assef10':dir_assef10,'par':par,'step':step,'walker':walker}
    source_dict       ={'wave':wave,'flux':flux,'flux_e':flux_error,'redshift':redshift,'filter_list':filter_list,'sourcename':sourcename,'up_limit':up_limit}
#    data_for_mc(tem_control,Av_control,source_dict)
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
    draw_result(tem_control,Av_control,result_sampler,save_path,dir_filter_folder,dir_assef10,m,source_dict)






