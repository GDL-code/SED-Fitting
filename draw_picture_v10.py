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
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


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
# ;
# ;   This function return the sudo-photometry in the bandpass from the given observed spectrum.
# ;   The photometry should have the same unit of the y-axis from the given spectrum.
# ;
# ; - Input:
# ;       data_w:         Wavelength of the data/spectrum. It should be in micron 
# ;                       (in order to match with the wavelength scale of bandpass)
# ;       data_f:         Flux (density) of the data/spectrum.
# ;       bandpass_name:  The nane of the bandpass. **It should have the same unit as the data_w.**
# ;                       For the available bandpass names, check function cw_filter_loading
# ;                       (/Users/ctsai/Softwork/IDL/CWTIDL/cw_filters/cw_filter_loading.pro)
# ;
# ; - Output:
# ;       flux_sudo:      The sudo photometry from the given spectrum (data_w, data_f) within the given 
# ;                       bandpass (bandpass_w, bandpass_t). **The unit of flux_sudo should be the same as 
# ;                       the unit of data_f.**
# ;
# ; - History:
# ;       2018_0523       Created by Chao-Wei Tsai (UCLA)
# ;       2020_0329       Modify Chao-Wei Tsai's code so that it can be run by python - Guodong Li, NAOC
    bp_band = cw_filter_loading(bandpass_name)
    # data_w1=[]
    # data_f1=[]
    # for i in range(len(data_w)):
        # if data_w[i]>=np.min(bp_band['FILTER_W']) and data_w[i]<=np.max(bp_band['FILTER_W']):
            # data_w1.append(data_w[i])
            # data_f1.append(data_f[i])
        # else:
            # pass
    index  =(data_w>=np.min(bp_band['FILTER_W']))&(data_w<np.max(bp_band['FILTER_W']))
    data_w1=data_w[index]
    data_f1=data_f[index]
    # data_w1=np.array(data_w1)
    # data_f1=np.array(data_f1)
    fun=interp1d(bp_band['FILTER_W'],bp_band['FILTER_T'], 'linear')
    bp_band_regrid = fun(data_w1)
    bp_band_regrid_pos = (bp_band_regrid + abs(bp_band_regrid)) / 2.
    bp_band_regrid_norm = bp_band_regrid_pos / tsum(data_w1,bp_band_regrid_pos)
    flux_sudo = tsum(data_w1, data_f1 * bp_band_regrid_norm) 
    return flux_sudo
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
def greybody_calculate(wav_um, T, beta,l,key=0,filter=None):
    if key==0:
        Bv                =greybody(wav_um, T, beta)
        vLv               =Bv*con.c.value/(wav_um*(1+redshift)*1e-6)
        vLv_sun           =vLv*(10**(-26))/con.L_sun.value
        return np.float64(l*vLv_sun)
    else:
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
        return l/sum(vLv_sun)
def cw_SPL_intfun(x, alpha):
    return (x**(-1. * alpha - 2.)/(np.exp(x)-1))
def cw_SPL_int(alpha,Tc, Th,v):
    xc                = con.h.value*v/(con.k_B.value*Tc)
    xh                = con.h.value*v/(con.k_B.value*Th)
    result            = integrate.quad(cw_SPL_intfun,xh,xc,args=(alpha))
    return result[0]
def dust(theta,wave_rest,redshift,key=0,filter=None):
    b,alpha,T_min,T_max   =theta
    if key==0:
        tem_wave          =wave_rest
        v                 =con.c.value/(tem_wave*1e-6)
        tem_vLv           =[]
        for i in v:
            a             =i**(alpha+5.5)
            S             =cw_SPL_int(alpha,T_min,T_max,i)
            tem_vLv.append(a*S)
        tem_vLv           =np.array(tem_vLv)
        tem_vLv           =tem_vLv*v/(1+redshift)
        return b*tem_vLv
    else:
        tem_wave          =np.zeros(len(wave_rest)+2)
        tem_wave[1:-1]    =wave_rest
        tem_wave[0]       =0.5*wave_rest[0]
        tem_wave[-1]      =1.5*wave_rest[-1]
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
        return b/sum(tem_vLv_band)
def tsum(x,y):
    v= integrate.trapz(y,x)
    return v
def none_filter(data_w, data_f,w_width):
    flux_sudo = tsum(data_w, data_f/w_width) 
    return flux_sudo
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
def get_template(source_dict,tem_control):
    template_name     =np.array(['Assef10_AGN','Assef10_AGN2','Assef10_E','Assef10_Sbc','Assef10_Im'])
    control_bool      =[bool(i) for i in tem_control]
    choice_tem        =template_name[control_bool]
    tem_vLv           ={}
    for i in choice_tem:
        template      =cw_sed_load(i)
        flux          =template['Fnu']
        tem_wave      =template['wave']
        vLv           =flux*con.c.value*(10**(-3))*(10**(-26))/((tem_wave*10**(-6))*con.L_sun.value )
        vLv1          =flux*con.c.value*(10**(-3))*(10**(-26))/((tem_wave*(1+redshift)*10**(-6))*con.L_sun.value )
        tem_vLv[i]    =vLv1/sum(vLv)
        Assef_wave    =tem_wave
    return tem_vLv,Assef_wave
def drwa_Assef10(theta,tem_control,Av_control,tem_vLv,wave_rest):
    template_name     =np.array(['Assef10_AGN','Assef10_AGN2','Assef10_E','Assef10_Sbc','Assef10_Im'])
    control_bool      =[bool(i) for i in tem_control[:5]]
    choice_tem        =template_name[control_bool]
    tem_number        =sum(tem_control[:5])
    Av_number         =sum(Av_control)
    single_tem        ={}
    if tem_number==1 and Av_number==1:
        a,Av          =theta
        model         =a*apply(fm07(wave_rest*(10**4),Av),tem_vLv[choice_tem[0]])
        single_tem[choice_tem[0]]=model
    elif tem_number==1 and Av_number==0:
        a             =theta[0]
        model         =a*tem_vLv[choice_tem[0]]
        single_tem[choice_tem[0]]=model
    elif tem_number==2 and Av_number==0:
        a,b           =theta
        model         =a*tem_vLv[choice_tem[0]]+b*tem_vLv[choice_tem[1]]
        single_tem[choice_tem[0]]=a*tem_vLv[choice_tem[0]]
        single_tem[choice_tem[1]]=b*tem_vLv[choice_tem[1]]
    elif tem_number==2 and Av_number==1:
        a,b,Av        =theta
        model         =a*apply(fm07(wave_rest*(10**4),Av),tem_vLv[choice_tem[0]])+b*tem_vLv[choice_tem[1]]
        single_tem[choice_tem[0]]=a*apply(fm07(wave_rest*(10**4),Av),tem_vLv[choice_tem[0]])
        single_tem[choice_tem[1]]=b*tem_vLv[choice_tem[1]]
    elif tem_number==2 and Av_number==2:
        a,b,Av1,Av2   =theta
        model         =a*apply(fm07(wave_rest*(10**4),Av1),tem_vLv[choice_tem[0]])+b*apply(fm07(wave_rest*(10**4),Av2),tem_vLv[choice_tem[1]])
        single_tem[choice_tem[0]]=a*apply(fm07(wave_rest*(10**4),Av1),tem_vLv[choice_tem[0]])
        single_tem[choice_tem[1]]=b*apply(fm07(wave_rest*(10**4),Av2),tem_vLv[choice_tem[1]])
    elif tem_number==3 and Av_number==0:
        a,b,c         =theta
        model         =a*tem_vLv[choice_tem[0]]+b*tem_vLv[choice_tem[1]]+c*tem_vLv[choice_tem[2]]
        single_tem[choice_tem[0]]=a*tem_vLv[choice_tem[0]]
        single_tem[choice_tem[1]]=b*tem_vLv[choice_tem[1]]
        single_tem[choice_tem[2]]=c*tem_vLv[choice_tem[2]]
    elif tem_number==3 and Av_number==1:
        a,b,c,Av      =theta
        model         =a*apply(fm07(wave_rest*(10**4),Av),tem_vLv[choice_tem[0]])+b*tem_vLv[choice_tem[1]]+c*tem_vLv[choice_tem[2]]
        single_tem[choice_tem[0]]=a*apply(fm07(wave_rest*(10**4),Av),tem_vLv[choice_tem[0]])
        single_tem[choice_tem[1]]=b*tem_vLv[choice_tem[1]]
        single_tem[choice_tem[2]]=c*tem_vLv[choice_tem[2]]
    elif tem_number==3 and Av_number==2:
        a,b,c,Av1,Av2 =theta
        model         =a*apply(fm07(wave_rest*(10**4),Av1),tem_vLv[choice_tem[0]])+b*apply(fm07(wave_rest*(10**4),Av2),tem_vLv[choice_tem[1]])+c*tem_vLv[choice_tem[2]]
        single_tem[choice_tem[0]]=a*apply(fm07(wave_rest*(10**4),Av1),tem_vLv[choice_tem[0]])
        single_tem[choice_tem[1]]=b*apply(fm07(wave_rest*(10**4),Av2),tem_vLv[choice_tem[1]])
        single_tem[choice_tem[2]]=c*tem_vLv[choice_tem[2]]
    elif tem_number==4 and Av_number==0:
        a,b,c,d       =theta
        model         =a*tem_vLv[choice_tem[0]]+b*tem_vLv[choice_tem[1]]+c*tem_vLv[choice_tem[2]]+d*tem_vLv[choice_tem[3]]
        single_tem[choice_tem[0]]=a*tem_vLv[choice_tem[0]]
        single_tem[choice_tem[1]]=b*tem_vLv[choice_tem[1]]
        single_tem[choice_tem[2]]=c*tem_vLv[choice_tem[2]]
        single_tem[choice_tem[3]]=d*tem_vLv[choice_tem[3]]
    elif tem_number==4 and Av_number==1:
        a,b,c,d,Av    =theta
        model         =a*apply(fm07(wave_rest*(10**4),Av),tem_vLv[choice_tem[0]])+b*tem_vLv[choice_tem[1]]+c*tem_vLv[choice_tem[2]]+d*tem_vLv[choice_tem[3]]
        single_tem[choice_tem[0]]=a*apply(fm07(wave_rest*(10**4),Av),tem_vLv[choice_tem[0]])
        single_tem[choice_tem[1]]=b*tem_vLv[choice_tem[1]]
        single_tem[choice_tem[2]]=c*tem_vLv[choice_tem[2]]
        single_tem[choice_tem[3]]=d*tem_vLv[choice_tem[3]]
    elif tem_number==4 and Av_number==2:
        a,b,c,d,e,f   =theta
        model         =a*apply(fm07(wave_rest*(10**4),e),tem_vLv[choice_tem[0]])+b*apply(fm07(wave_rest*(10**4),f),tem_vLv[choice_tem[1]])+c*tem_vLv[choice_tem[2]]+d*tem_vLv[choice_tem[3]]
        single_tem[choice_tem[0]]=a*apply(fm07(wave_rest*(10**4),e),tem_vLv[choice_tem[0]])
        single_tem[choice_tem[1]]=b*apply(fm07(wave_rest*(10**4),f),tem_vLv[choice_tem[1]])
        single_tem[choice_tem[2]]=c*tem_vLv[choice_tem[2]]
        single_tem[choice_tem[3]]=d*tem_vLv[choice_tem[3]]
    elif tem_number==5:
        a,b,c,d,e,f,h =theta
        model         =a*apply(fm07(wave_rest*(10**4),f),tem_vLv[choice_tem[0]])+b*apply(fm07(wave_rest*(10**4),h),tem_vLv[choice_tem[1]])+c*tem_vLv[choice_tem[2]]+d*tem_vLv[choice_tem[3]]+e*tem_vLv[choice_tem[4]]
        single_tem[choice_tem[0]]=a*apply(fm07(wave_rest*(10**4),f),tem_vLv[choice_tem[0]])
        single_tem[choice_tem[1]]=b*apply(fm07(wave_rest*(10**4),h),tem_vLv[choice_tem[1]])
        single_tem[choice_tem[2]]=c*tem_vLv[choice_tem[2]]
        single_tem[choice_tem[3]]=d*tem_vLv[choice_tem[3]]
        single_tem[choice_tem[4]]=e*tem_vLv[choice_tem[4]]
    else:
        print('The number of Assef template exceed 5.')
        os._exit(0)
    return model,single_tem
def draw_fitting(source_dict,result_samplers,tem_control,Av_control,f_name):
    array             =result_samplers.shape
    theta             =[]
    for i in range(array[1]):
        theta.append(np.median(result_samplers[:,i]))
    source_cdict      =source_calculate(source_dict)
    up_limit          =source_dict['up_limit']
    redshift          =source_dict['redshift']
    filter_list       =source_dict['filter_list']
    wave              =source_cdict['wave']
    redshift          =source_cdict['redshift']
    vLv               =source_cdict['vLv']
    vLv_e             =source_cdict['vLv_e']
    wave_rest         =wave/(1+redshift)
    vLv_norm          =vLv/sum(vLv)
    vLv_e_norm        =vLv_e/sum(vLv)
    proportion        ={}
    fig               =plt.figure(figsize=[8,6])
    ax                =fig.add_subplot(111)
    t_color           ={'Assef10_AGN':'coral','Assef10_AGN2':'magenta','Assef10_E':'maroon',\
                        'Assef10_Sbc':'green','Assef10_Im':'blue','Fitted curve':'grey','Greybody':'firebrick','Dust':'cyan'}
    if tem_control[5]==0 and tem_control[6]==0:
        tem_vLv,Assef_wave=get_template(source_dict,tem_control[:5])
        model,single  =drwa_Assef10(theta,tem_control,Av_control,tem_vLv,Assef_wave)
        tem_y         =model
        tem_x         =Assef_wave
        for i in single.keys():
            ax.plot(tem_x,sum(vLv)*single[i],color=t_color[i], linewidth=2, linestyle='--',alpha=1,label=i)
            proportion[i]=tsum(tem_x,sum(vLv)*single[i])
        ax.plot(tem_x, sum(vLv)*tem_y,color=t_color['Fitted curve'], linewidth=4, linestyle='-',label='Fitted curve',alpha=0.7)
    elif tem_control[5]==1 and sum(tem_control[:5])==0 and tem_control[6]==0:
        index         =wave_rest>1.0
        fitting_wave  =wave_rest[index]
        filter        =filter_list[index]
        l_norm        =greybody_calculate(fitting_wave, theta[0], theta[1],theta[2],key=1,filter=filter)
        tem_x         =np.arange(0.1,1000,0.1)
        tem_y         =greybody_calculate(tem_x, theta[0], theta[1],l_norm)
        ax.plot(tem_x, sum(vLv)*tem_y,color=t_color['Greybody'], linewidth=2, linestyle='-',label='Greybody',alpha=1)
        proportion['Greybody']=tsum(tem_x, sum(vLv)*tem_y)
    elif tem_control[5]==0 and sum(tem_control[:5])==0 and tem_control[6]==1:
        index         =wave_rest>1.0
        fitting_wave  =wave_rest[index]
        filter        =filter_list[index]
        l_norm        =dust(theta,fitting_wave,redshift,key=1,filter=filter)
        theta[0]      =l_norm
        tem_x         =np.arange(0.1,1000,0.1)
        tem_y         =dust(theta,tem_x,redshift)
        ax.plot(tem_x, sum(vLv)*tem_y,color=t_color['Dust'], linewidth=2, linestyle='-',label='Dust emission',alpha=0.7)
        proportion['Dust']=tsum(tem_x, sum(vLv)*tem_y)
    elif tem_control[5]==1 and tem_control[6]==0 and sum(tem_control[:5])!=0:
        tem_vLv,Assef_wave=get_template(source_dict,tem_control[:5])
        model,single  =drwa_Assef10(theta[:-3],tem_control,Av_control,tem_vLv,Assef_wave)
        breakpoint    =np.max(Assef_wave)
        correct_wave  =np.arange(breakpoint+0.01,1000,0.01)
        model_assf    =np.zeros(len(correct_wave))
        Assef         =np.hstack((model,model_assf))
        tem_x         =np.hstack((Assef_wave,correct_wave))
        index_0       =wave_rest>1.0
        fitting_wave  =wave_rest[index_0]
        filter        =filter_list[index_0]
        l_norm        =greybody_calculate(fitting_wave, theta[-3], theta[-2],theta[-1],key=1,filter=filter)
        greybody      =greybody_calculate(tem_x,theta[-3], theta[-2],l_norm)
        index         =tem_x<=1.0
        greybody_cor  =1.0*greybody
        greybody_cor[index]=0.0*greybody_cor[index]
        tem_y         =Assef+greybody_cor
        for i in single.keys():
            ax.plot(Assef_wave,sum(vLv)*single[i],color=t_color[i], linewidth=2, linestyle='--',alpha=1,label=i)
            proportion[i]=tsum(Assef_wave,sum(vLv)*single[i])
        ax.plot(tem_x, sum(vLv)*tem_y,color=t_color['Fitted curve'], linewidth=4, linestyle='-',label='Fitted curve',alpha=0.7)
        ax.plot(tem_x, sum(vLv)*greybody,color=t_color['Greybody'], linewidth=2, linestyle='-',alpha=1,label='Greybody')
        proportion['Greybody']=tsum(tem_x, sum(vLv)*greybody_cor)
    elif tem_control[5]==0 and tem_control[6]==1 and sum(tem_control[:5])!=0:
        tem_vLv,Assef_wave=get_template(source_dict,tem_control[:5])
        model,single  =drwa_Assef10(theta[:-4],tem_control,Av_control,tem_vLv,Assef_wave)
        breakpoint    =np.max(Assef_wave)
        correct_wave  =np.arange(breakpoint+0.01,1000,0.1)
        model_1       =np.zeros(len(correct_wave))
        Assef         =np.hstack((model,model_1))
        tem_x         =np.hstack((Assef_wave,correct_wave))
        index_0       =wave_rest>1.0
        fitting_wave  =wave_rest[index_0]
        filter        =filter_list[index_0]
        l_norm        =dust(theta[-4:],fitting_wave,redshift,key=1,filter=filter)
        dust_model    =dust([l_norm,theta[-3],theta[-2],theta[-1]],tem_x,redshift)
        index         =tem_x<=1.0
        dust_model1   =1.0*dust_model
        dust_model1[index]=0.0*dust_model1[index]
        tem_y         =Assef+dust_model1
        for i in single.keys():
            ax.plot(Assef_wave,sum(vLv)*single[i],color=t_color[i], linewidth=2, linestyle='--',alpha=1,label=i)
            proportion[i]=tsum(Assef_wave,sum(vLv)*single[i])
        ax.plot(tem_x, sum(vLv)*tem_y,color=t_color['Fitted curve'], linewidth=4, linestyle='-',label='Fitted curve',alpha=0.7)
        ax.plot(tem_x, sum(vLv)*dust_model,color=t_color['Dust'], linewidth=2, linestyle='--',label='Dust emission',alpha=0.6)
        proportion['Dust']=tsum(tem_x, sum(vLv)*dust_model1)
    elif tem_control[5]==1 and tem_control[6]==1 and sum(tem_control[:5])==0:
        index         =wave_rest>1.0
        fitting_wave  =wave_rest[index]
        filter        =filter_list[index]
        l_norm_grey   =greybody_calculate(fitting_wave,theta[0], theta[1],theta[2],key=1,filter=filter)
        l_norm_dust   =dust(theta[-4:],fitting_wave,redshift,key=1,filter=filter)
        wave_tem      =np.arange(0.1,1000,0.1)
        dust_model    =dust([l_norm_dust,theta[-3],theta[-2],theta[-1]],wave_tem,redshift)
        greybody      =greybody_calculate(wave_tem,theta[0], theta[1],l_norm_grey)
        tem_x         =wave_tem
        tem_y         =dust_model+greybody
        ax.plot(tem_x, sum(vLv)*tem_y,color=t_color['Fitted curve'], linewidth=4, linestyle='-',label='Fitted curve',alpha=0.7)
        ax.plot(tem_x, sum(vLv)*dust_model,color=t_color['Dust'], linewidth=2, linestyle='--',label='Dust emission',alpha=1)
        ax.plot(tem_x, sum(vLv)*greybody,color=t_color['Greybody'], linewidth=2, linestyle='--',alpha=1,label='Greybody')
        proportion['Greybody']=tsum(tem_x, sum(vLv)*greybody)
        proportion['Dust']=tsum(tem_x, sum(vLv)*dust_model)
    else:
        tem_vLv,Assef_wave=get_template(source_dict,tem_control[:5])
        model,single  =drwa_Assef10(theta[:-7],tem_control,Av_control,tem_vLv,Assef_wave)
        breakpoint    =np.max(Assef_wave)
        correct_wave  =np.arange(breakpoint+0.01,1000,0.1)
        model1        =np.zeros(len(correct_wave))
        Assef         =np.hstack((model,model_1))
        tem_x         =np.hstack((Assef_wave,correct_wave))
        index_0       =wave_rest>1.0
        fitting_wave  =wave_rest[index_0]
        filter        =filter_list[index_0]
        l_norm_grey   =greybody_calculate(fitting_wave,theta[-7], theta[-6],theta[-5],key=1,filter=filter)
        l_norm_dust   =dust(theta[-4:],fitting_wave,redshift,key=1,filter=filter)
        greybody      =greybody_calculate(tem_x,theta[-7], theta[-6],l_norm_grey)
        dust_model    =dust([l_norm_dust,theta[-3],theta[-2],theta[-1]],tem_x,redshift)
        index         =tem_x<=1.0
        greybody1     =1.0*greybody
        greybody1[index]=0.0*greybody1[index]
        dust_model1   =1.0*dust_model
        dust_model1[index]=0.0*dust_model1[index]
        tem_y         =Assef+greybody1+dust_model1
        for i in single.keys():
            ax.plot(Assef_wave,sum(vLv)*single[i],color=t_color[i], linewidth=2, linestyle='--',alpha=1,label=i)
            proportion[i]=tsum(Assef_wave,sum(vLv)*single[i])
        ax.plot(tem_x,sum(vLv)*tem_y,color=t_color['Fitted curve'], linewidth=4, linestyle='-',label='Fitted curve',alpha=0.7)
        ax.plot(tem_x,sum(vLv)*greybody,color=t_color['Greybody'], linewidth=2, linestyle='-',alpha=1,label='Greybody')
        ax.plot(tem_x,sum(vLv)*dust_model,color=t_color['Dust'], linewidth=2, linestyle='-',alpha=1,label='Dust emission')
        proportion['Greybody']=tsum(tem_x, sum(vLv)*greybody1)
        proportion['Dust']=tsum(tem_x, sum(vLv)*dust_model1)
    if len(up_limit)!=0:
        x                 =np.delete(wave_rest,up_limit)
        y                 =np.delete(vLv, up_limit)
        y_e               =np.delete(vLv_e, up_limit)
        ax.errorbar(x,y,y_e, fmt='ro',linewidth=1,ecolor='k',elinewidth=1,capsize=4,capthick=1,ms=4,label=source_dict['sourcename']+'\n('+str(redshift)+')')
        ax.errorbar(wave_rest[up_limit],vLv[up_limit],vLv_e[up_limit], uplims=True,fmt='ro',ecolor='k',elinewidth=1,capsize=2,ms=4)
    else:
        x                 =wave_rest
        y                 =vLv
        y_e               =vLv_e
        ax.errorbar(x,y,y_e, fmt='ro',linewidth=1,ecolor='k',elinewidth=1,capsize=4,capthick=1,ms=4,label=source_dict['sourcename']+'\n(redshift='+str(redshift)+')')
    
    font = {'family' : 'Microsoft JhengHei','weight' :'normal','size'   :12}
    ax.legend(loc='upper left',fontsize=8,borderaxespad=2)
    ax.set_xlabel(r'Rest-frame Wavelength [$\mu$m]',font,labelpad=8)
    ax.set_ylabel(r'$\nu$$L_{\nu}$ [$L_⊙$]',font)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.01,1000)
    ax.set_ylim(10**(7),10**15)
    xmajorFormatter = FormatStrFormatter('%.1f')
    ax.xaxis.set_major_formatter(xmajorFormatter)
    ax.tick_params(axis='both',which='major',tickdir='in',labelsize=12,pad=6,length=10,bottom=True, top=False, left=True, right=True,labelbottom=True, labeltop=False, labelleft=True)
    ax.tick_params(axis='both',which='minor',tickdir='in',length=5,bottom=True, top=False, left=True, right=True)
    
    ax2=plt.twiny()
    x2=wave
    y=[None]*len(x2)
    y2=np.array(y)
    ax2.plot(x2,y2)
    ax2.set_xlabel(r'Observed Wavelength [$\mu$m]',font,labelpad=10)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(0.01*(1+redshift),1000*(1+redshift))
    ax2.xaxis.set_major_formatter(xmajorFormatter)
    ax2.tick_params(axis='x',which='major',tickdir='in',labelsize=12,pad=6,length=10,top=True,labeltop=True)
    ax2.tick_params(axis='x',which='minor',tickdir='in',length=5,top=True)
    
    plt.savefig(save_path+f_name)
    L_bol                 =tsum(wave_rest,vLv)
    L_fitting             =tsum(tem_x,sum(vLv)*tem_y)
    write_result(source_dict,tem_control,Av_control,L_bol,theta,proportion,L_fitting)
    return 0
def write_result(source_dict,tem_control,Av_control,L_bol,theta,proportion,L_fitting):
    np.set_printoptions(precision=4,suppress=True)
    flux            =np.round(source_dict['flux'],decimals = 4)
    flux            =str(list(flux ))
    flux_error      =np.round(source_dict['flux_e'],decimals = 4)
    flux_error      =str(list(flux_error))
    wave            =np.round(source_dict['wave'],decimals = 4)
    wave            =str(list(wave))
    redshift        =str(source_dict['redshift'])
    sourcename      =source_dict['sourcename']
    filter          =str(list(source_dict['filter_list']))
    upplimit        =[False]*len(source_dict['wave'])
    for i in source_dict['up_limit']:
        upplimit[i] =True
    result_file     =save_path+str(sourcename)+'.dat'
    f               =open(result_file,'a+')
    f.write('Source name'+'\t'+':'+sourcename+'\n')
    f.write('Redshift'+'\t'+':'+redshift+'\n')
    f.write('Wave'+2*'\t'+':'+wave+'\n')
    f.write('Filter'+2*'\t'+':'+filter+'\n')
    f.write('Flux'+2*'\t'+':'+flux+'\n')
    f.write('Flux_error'+'\t'+':'+flux_error+'\n')
    f.write('upplimit'+'\t'+':'+str(upplimit)+'\n')
    f.write('L_bol'+2*'\t'+':'+format(L_bol,'.2e')+' L_sun (power law interpolated)'+'\n')
    model           =['AGN1_c*Assef10_AGN(Av)','AGN2_c*Assef10_AGN2(Av)','E_c*Assef10_E',\
                      'Sbc_c*Assef10_Sbc','Im_c*Assef10_Im','S*Greybody(T,Beta)','D*Dust(alpha)']
    moder_str       =''
    for i in range(len(tem_control)):
        if tem_control[i]==0:
            pass
        else:
            moder_str=moder_str+model[i]+'+'
    moder_str       =moder_str[:-1]
    f.write('model'+2*'\t'+':'+moder_str+'\n')
    f.write('Fitting result'+'\t'+':'+'\n')
    f.write(80*'*'+'\n')
    f.write('Component'+2*'\t'+'Parameter'+2*'\t'+'Value'+2*'\t'+'Proportion'+'\n')
    tem           =np.array(tem_control)*1
    tem[:2]       =tem[:2]+np.array(Av_control)
    tem[5]        =3*tem[5]
    tem[6]        =2*tem[6]
    label           =['Assef10_AGN','Assef10_AGN2','Assef10_E',\
                      'Assef10_Sbc','Assef10_Im','Greybody','Dust']
    parameter       ={'Assef10_AGN':['AGN1_c','Av_1'],'Assef10_AGN2':['AGN2_c','Av_2'],'Assef10_E':['E_c'],\
                      'Assef10_Sbc':['Sbc_c'],'Assef10_Im':['Im_c'],'Greybody':["T","Beta","S"],'Dust':["D","alpha"]}
    par_label       =["AGN1_c","AGN2_c","E_c","Sbc_c","Im_c","Av_1","Av_2","T","Beta","S","D","alpha"]
    choice_par      =list(tem_control[:5])+list(Av_control)+3*list(tem_control[5:6])+2*list(tem_control[6:])
    choice_par_label=[par_label[i] for i in range(12) if choice_par[i]!=0]
    value           ={}
    for i in range(len(choice_par_label)):
        value[choice_par_label[i]]=theta[i]
    for i in range(len(tem)):
        if tem[i]==0:
            pass
        else:
            component   =label[i]
            Par         =parameter[component]
            prop        =proportion[component]
            for j in range(int(tem[i])):
                Par1        =Par[j]
                va          =value[Par1]
                if j==0:
                    prop    =prop/L_fitting
                    f.write(str(component)+2*'\t'+str(Par1)+2*'\t'+format(va,'.2e')+2*'\t'+format(prop,'.3%')+'\n')
                else:
                    f.write(str(component)+2*'\t'+str(Par1)+2*'\t'+format(va,'.2e')+'\n')
    f.write(80*'*'+'\n')
    f.write('L_bol (Fitting)'+'\t'+':'+format(L_fitting,'.2e')+' L_sun (power law interpolated)'+'\n')
    f.write('Chi-Square'+2*'\t'+':'+format(source_dict['chi_square'],'.4e')+'\n')
    f.write('Reduced Chi-Square'+2*'\t'+':'+format(source_dict['re_chi'],'.4e')+'\n')
    f.write('\n')
    f.close()
    return 0
def draw_result(tem_control,Av_control,result_sampler,save_pat,dir_filter_fold,dir_assef,m,source_dict):
    global dir_assef10
    global save_path
    global dir_filter_folder
    global redshift
    dir_assef10       =dir_assef
    save_path         =save_pat
    dir_filter_folder =dir_filter_fold
    redshift          =source_dict['redshift']
    label             =[r"$AGN1_c$",r"$AGN2_c$",r"$E_c$",r"$Sbc_c$",r"$Im_c$",r"$Av_1$",r"$Av_2$",r"$T_{Grey}$",r"$Beta_{Grey}$",r"$L_{Grey}$",r"$D_{Dust}$",r"$alpha_{Dust}$",r"$T_{min}$",r"$T_{max}$"]
    label_control     =list(tem_control[:5])+list(Av_control)+list(tem_control[5:6])*3+list(tem_control[6:])*4
    label_corner      =[label[i] for i in range(14) if label_control[i]==1]
    fig_corner        =corner.corner(result_sampler,labels=label_corner,quantiles=[0.16,0.5,0.84],\
                       smooth=True,show_titles=True,title_kwargs={"fontsize":12},title_fmt='.2f')
    fig_corner.savefig(save_path+'emcmc('+str(m)+').png',bbox_inches='tight')
    f_name            ='fitting'+str(m)+'.png'
    draw_fitting(source_dict,result_sampler,tem_control,Av_control,f_name)
    return 0