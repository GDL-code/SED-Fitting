# SED-Fitting
① The function of cw_filter_loading, greybody, cw_fflux_from_spec_bandname and cw_sed_load come from Tsai.
② The extinction model: Fitzpatrick & Massa (2007) extinction model for R_V = 3.1.
③ H0=70 km/s/Kpc, OmO=0.3
④ When you use the code, you need to modify the process 1 and process 2.
# Process 1: Input data
           ① You should modify the redshift, flux, flux_error, filter_list, wave, up_limit, sourcename, tem_control and Av_control.
           ② You need to ensure the unit of wave is micron and the unit of flux or flux_error is milli Jy
           ③ up_limit: The index of upper_limit points
           ④ The order of template name is:'Assef10_AGN','Assef10_AGN2','Assef10_E','Assef10_Sbc','Assef10_Im','Greybody','Dust emission'
           ⑤ tem_control: The template you want to use in the fitting.
           ⑥ Av_control: To “redden” the AGN1 or AGN2.
# Process 2: Input path
           dir_filter_folder: The path of folder, which include the filter.
           dir_assef10,dir_richards06,dir_richards06,/
           dir_swire,dir_grasil,dir_tsai19: The path of folder, which include the SED template.
# Process 3: SED fitting
           ① Running the program with multiprocess.
           ② Return the sample obtained by MC fitting.
# Process 4: Draw SED fitting diagram
           You can draw images according to your own needs.
