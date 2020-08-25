#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# FUNCTIONS

# First off, import the packages we need
import numpy as np  # work horse package for numerical work in python
import matplotlib.pyplot as plt  # plotting library
from dragons import meraxes, munge
import random
import sys

def get_gal_catalogue(snapshot_used):
    # import meraxes and set little h to 0.7
    fname_in = "imapping_testrun/meraxes.hdf5"
    h = meraxes.set_little_h(0.7)
    gals, sim_props = meraxes.read_gals(fname_in, snapshot_used, sim_props=True)
    snaplist = meraxes.io.read_snaplist(fname_in, h)
    return gals, sim_props, snaplist
    
def ALICE_mass_function(mass, volume, bins, range): # kwargs allows additional args to be read in
    # Mass function
    # HImf = "HIMass function" is a number density of galaxies in the universe as a function of their HI mass
    # HIMF (Phi) can be expressed as Phi = Ngal / (V * bin_width)
    #N_gal (number of galaxies), V is volume, bin_width is the HI mass bin width
    
    # counts can be number of galaxies from histogram
    N_gals, edges = np.histogram(mass, bins, range) # returns N_gals (counts in each bin) 
    # and bin edges which is the range divided by the number of bins
    print('N_gals  : ', N_gals)
    print('edges  : ', edges)
    bin_width = edges[1] - edges[0] # get bin width from the edges of two bins next to eachother
    print('bin width  : ', bin_width)
    bin_center = (edges[:-1] + edges[1:]) / 2 # get bin centers
    print('bin_center : ', bin_center[:]) # these look like the values of the mass which is a good sign
    
    phi = N_gals / (volume * bin_width) # calculate the mass function
    # convert to correct units (Mpc^-3 h^3)
    #phi = phi / (0.7)**3 ADD IN LATER 
    print('phi  : ', phi)
    # Need to return the bin centers corresponding to the HI Mass 
    # and the value of Phi, our Mass Function
    himf=[] # create empty array
    himf.append((bin_center, phi)) # returns as 50 columns and two rows, all bin_centers in [0] and all phi values in [1]
    himf = np.asarray(himf) # convert from list into array (adds extra dimensin by default)
    himf = himf.T.squeeze() # need to transpose the array so it is 2 columns and 50 rows and use squeeze to get rid of extra dimension
    #print('IN LOOP : ', himf)
    return himf


def return_himf_for_Gal_Types(gals, sim_props, range_in):

    Type = gals["Type"] # Get the different types of galaxy
    #gals["HIMass"] = np.log10(gals["HIMass"]*1e10)# Let's convert into appropriate units (log10(M/Msol)) (Msol is in 1e10)
    uniqueTypes = np.unique(Type) # Get the number of different types of galaxy
    #print('uniqueTypes: ', uniqueTypes)
    himf_Types=[]
    
    for i in uniqueTypes:
        # For each type of galaxy create a new mass function CHANGE TO ALICE_mass_function IN A BIT ONCE TESTED
        phi = ALICE_mass_function(gals["HIMass"][Type==i], sim_props["Volume"], bins=50, range=range_in)
        himf=[] # create empty array
        himf.append(phi) # returns as 50 columns and two rows, all bin_centers in [0] and all phi values in [1]
        #print('Type = ' + str(i) + "\n[Mass, HIMF]\n", himf)
        himf_Types.append(himf)

    himf_Types = np.asarray(himf_Types) # convert from list into array (adds extra dimensin by default)
    himf_Types = himf_Types.squeeze() # need to transpose the array so it is 2 columns and 50 rows and use squeeze to get rid of extra dimension
    himf_Types.shape
    return himf_Types

    
def Random_Sample_Gals(gals, k):
    
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
