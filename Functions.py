#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# FUNCTIONS

# First off, import the packages we need
import numpy as np  # work horse package for numerical work in python
import matplotlib.pyplot as plt  # plotting library
from dragons import meraxes, munge# DRAGONS modules for reading and dealing with model ouput
import sys
import random
from statistics import mean

def print_units(fname_in):
    units = meraxes.read_units(fname_in)
    print('\nunits\n=====')
    for k, v in units.items():
        if not isinstance(v, dict):
            print(k, ':', v.decode('utf-8'))
    print()

    filtered_galprops_units = {
        key: units[key] 
        for key in units.keys() 
            & {'HIMass', 'HaloID', 'Mvir', 'CentralGal', 'StellarMass', 'Sfr', 'Pos'}
        }

def get_gal_catalogue(snapshot_used):
    #---------------------------------------------
    # SNAPSHOTS AVAILABLE for 'snapshot_used'
    # 100, 115, 134, 158, 173, 192, 216, 250
    # CORRESPONDING REDSHIFTS
    # 5, 4, 3, 2, 1.5, 1, 0.5, 0
    #---------------------------------------------

    # import meraxes and set little h to 0.7
    # h=0.7, converting to a Hubble constant of H0=70 km/s/Mpc for galaxies
    fname_in = "imapping_testrun/meraxes.hdf5"
    h = meraxes.set_little_h(0.7)
    # reads in galaxy catalogue and simulation properties from meraxes function 'read_gals'
    gals, sim_props = meraxes.read_gals(fname_in, snapshot_used, sim_props=True, pandas=True)
    # returns snap list regardless of snap used, just to get all useful variables with one function
    snaplist = meraxes.io.read_snaplist(fname_in, h)    

    return gals, sim_props, snaplist, fname_in
    
def calc_mass_function(mass, volume, bins, range): # kwargs allows additional args to be read in
    # Mass function
    # HImf = "HIMass function" is a number density of galaxies in the universe as a function of their HI mass
    # HIMF (Phi) can be expressed as Phi = Ngal / (V * bin_width)
    #N_gal (number of galaxies), V is volume, bin_width is the HI mass bin width
    
    # If you want to see the values, uncomment the double hashed rows.

    # counts can be number of galaxies from histogram
    N_gals, edges = np.histogram(mass, bins, range) # returns N_gals (counts in each bin) 
    # and bin edges which is the range divided by the number of bins
    ##print('N_gals  : ', N_gals)
    ##print('edges  : ', edges)
    bin_width = edges[1] - edges[0] # get bin width from the edges of two bins next to eachother
    #print('bin width  : ', bin_width)
    bin_center = (edges[:-1] + edges[1:]) / 2 # get bin centers
    #print('bin_center : ', bin_center[:]) # these look like the values of the mass which is a good sign
    
    phi = N_gals / (volume * bin_width) # calculate the mass function
    # convert to correct units (Mpc^-3 h^3)
    #phi = phi / (0.7)**3 ADD IN LATER 
    ##print('phi  : ', phi)
    # Need to return the bin centers corresponding to the HI Mass 
    # and the value of Phi, our Mass Function
    himf=[] # create empty array
    himf.append((bin_center, phi)) # returns as 50 columns and two rows, all bin_centers in [0] and all phi values in [1]
    himf = np.asarray(himf) # convert from list into array (adds extra dimensin by default)
    himf = himf.T.squeeze() # need to transpose the array so it is 2 columns and 50 rows and use squeeze to get rid of extra dimension
    #print('IN LOOP : ', himf)
    return himf

def return_red_from_snap(snaplist, snapshot_used):
    snaplist_index_used = list(snaplist[0]).index(snapshot_used)# convert to list and find the index of the snap we are using
    redshift_used = snaplist[1][snaplist_index_used] # Get the redshift for corresponding snapshot
    redshift_used = round(redshift_used, 1) # round float to nearest whole int
    redshift_used = str(redshift_used)
    return redshift_used

def return_himf_for_Gal_Types(gals, sim_props, range_in):

    Type = gals["Type"] # Get the different types of galaxy
    #gals["HIMass"] = np.log10(gals["HIMass"]*1e10)# Let's convert into appropriate units (log10(M/Msol)) (Msol is in 1e10)
    uniqueTypes = np.unique(Type) # Get the number of different types of galaxy
    #print('uniqueTypes: ', uniqueTypes)
    himf_Types=[]
    
    for i in uniqueTypes:
        # For each type of galaxy create a new mass function CHANGE TO calc_mass_function IN A BIT ONCE TESTED
        phi = calc_mass_function(gals["HIMass"][Type==i], sim_props["Volume"], bins=50, range=range_in)
        himf=[] # create empty array
        himf.append(phi) # returns as 50 columns and two rows, all bin_centers in [0] and all phi values in [1]
        #print('Type = ' + str(i) + "\n[Mass, HIMF]\n", himf)
        himf_Types.append(himf)

    himf_Types = np.asarray(himf_Types) # convert from list into array (adds extra dimensin by default)
    himf_Types = himf_Types.squeeze() # need to transpose the array so it is 2 columns and 50 rows and use squeeze to get rid of extra dimension
    himf_Types.shape
    return himf_Types

    
def Random_Sample_Gals(gals, k): # USED when NOT using PANDAS
    
    # Calculates the number of galaxies in catalogue 'gals'
    gal_num = len(gals)
    #print('gal_num = ', gal_num)

    # Returns a sequence of numbers of galaxy IDs ranging from 0 - number of galaxies
    gal_ids = range(0,gal_num)
    #print('gal_ids = ', gal_ids)

    # Select a random sample of k gal IDs (incides) 
    gal_randsam = random.sample(gal_ids, k)
    #print('gal_randsam = ', gal_randsam)

    # Make new variable containing the selected properties taken from the random sample of k galaxies
    selected_gal_properties= gals[gal_randsam]
    #print('selected_gal_properties = ', selected_gal_properties)

    return selected_gal_properties

def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)*mean(xs)) - mean(xs*xs)))
    
    b = mean(ys) - m*mean(xs)
    
    return m, b

def return_3d_pos(gals):
	# returns (n,d) array of [x,y,z]
	pos_x = gals["Pos_0"]
	pos_y = gals["Pos_1"]
	pos_z = gals["Pos_2"]
	Pos = np.array([pos_x, pos_y, pos_z])
	Pos_new=Pos.T # Transpose the Pos so it is in format (N,D) array
	return pos_x, pos_y, pos_z, Pos, Pos_new

def get_Martin_data():
    # MARTIN 2010 DATA SET FOR 1/ V_max Method
    # Now lets plot the new data on top of the HIMF graph
    # Pull in the data
    file_Martin_VMAX = open("data/HIMF_Martin2010_1-VMAX.data", "r")# from here on Data from Martin 2010 1/VMax method will be referred to as data_MV
    fileread_MV = file_Martin_VMAX.readlines()
    for line in fileread_MV:
        if line.startswith("#"):
            continue # skips that iteration 


    data_MV = np.loadtxt(fileread_MV, delimiter="  ") 
    #print(data_Zwaan)
    log_MHI_MV = data_MV[:,0]
    print('Martin 1/ V_{max} 2010')
    print('log_MHI : ', log_MHI_MV)
    log_HIMF_MV = data_MV[:,1]
    print('log_HIMF : ', log_HIMF_MV)
    HIMF_MV_err = data_MV[:,2]
    return log_MHI_MV, log_HIMF_MV, HIMF_MV_err

def get_Zwaan_data():
    # ZWAAN 2005 DATA SET
    # Now lets plot the new data on top of the HIMF graph
    # Pull in the data
    file_Zwaan = open("data/HIMF_Zwaan2005.data", "r")
    fileread_Zwaan = file_Zwaan.readlines()
    for line in fileread_Zwaan:
        if line.startswith("#"):
            continue # skips that iteration 

    data_Zwaan = np.loadtxt(fileread_Zwaan, delimiter=" ")
    #print(data_Zwaan)
    log_MHI_Zwaan = data_Zwaan[:,0]
    print('Zwaan 2005')
    print('log_MHI : ', log_MHI_Zwaan)
    log_HIMF_Zwaan = data_Zwaan[:,1]
    print('log_HIMF : ', log_HIMF_Zwaan)
    MHI_Zwaan_err = data_Zwaan[:,2]
    HIMF_Zwaan_err = data_Zwaan[:,3]
    return log_MHI_Zwaan, log_HIMF_Zwaan, MHI_Zwaan_err, HIMF_Zwaan_err
