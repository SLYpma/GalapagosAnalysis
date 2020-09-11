from parcels import AdvectionRK4, Field, FieldSet, JITParticle, ScipyParticle 
from parcels import ParticleFile, ParticleSet, Variable, VectorField, ErrorCode
from parcels.tools.converters import GeographicPolar 
from datetime import timedelta as delta
from os import path
from glob import glob
import numpy as np
import dask
import math
import xarray as xr
from netCDF4 import Dataset
import warnings
import matplotlib.pyplot as plt
import pickle
warnings.simplefilter('ignore', category=xr.SerializationWarning)
from operator import attrgetter

withstokes = True 
withtides = True

data_in = "../../input/modelfields/SMOC"
data_out = "../../input/particles/"
fname = "NorthSea_200911"
NS_domain = [-4.8, 10.5, 48.6, 60.1]

#run details
advection_duration = 1 #unit: days (how long does one particle advect in the fields)
output_frequency = 6 #unit: hours

#Get indices for Galapagos domain to run simulation
def getclosest_ij(lats,lons,latpt,lonpt):    
    """Function to find the index of the closest point to a certain lon/lat value."""
    dist_lat = (lats-latpt)**2                      # find squared distance of every point on grid
    dist_lon = (lons-lonpt)**2
    minindex_lat = dist_lat.argmin()                # 1D index of minimum dist_sq element
    minindex_lon = dist_lon.argmin()
    return minindex_lat, minindex_lon                # Get 2D index for latvals and lonvals arrays from 1D index

dfile = Dataset(data_in+'/SMOC_20200831_R20200901.nc')
lon = dfile.variables['longitude'][:]
lat = dfile.variables['latitude'][:]
iy_min, ix_min = getclosest_ij(lat, lon, NS_domain[2], NS_domain[0])
iy_max, ix_max = getclosest_ij(lat, lon, NS_domain[3], NS_domain[1])

### add fields

files = sorted(glob(data_in + "/SMOC_2020*.nc"))
variables = {'U': 'uo', 'V': 'vo'}
dimensions = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}
indices = {'lon': range(ix_min,ix_max), 'lat': range(iy_min,iy_max)}
fset_currents = FieldSet.from_netcdf(files, variables, dimensions, indices=indices)
fieldset_currents = FieldSet(U=fset_currents.U, V=fset_currents.V)

if withstokes:
    stokesfiles = sorted(glob(data_in + "/SMOC_2020*.nc"))
    stokesdimensions = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}
    stokesvariables = {'U': 'vsdx', 'V': 'vsdy'}
    fset_stokes = FieldSet.from_netcdf(stokesfiles, stokesvariables, stokesdimensions, indices=indices)
    fname += '_stokes'
    
if withtides:
    tidesfiles = sorted(glob(data_in + "/SMOC_2020*.nc"))
    tidesdimensions = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}
    tidesvariables = {'U': 'utide', 'V': 'vtide'}
    fset_tides = FieldSet.from_netcdf(tidesfiles, tidesvariables, tidesdimensions, indices=indices)
    fname += '_tides'

if withstokes and withtides:
    fieldset = FieldSet(U=fset_currents.U + fset_stokes.U + fset_tides.U,
                        V=fset_currents.V + fset_stokes.V + fset_tides.V)
elif withstokes:
    fieldset = FieldSet(U=fset_currents.U + fset_stokes.U, 
                        V=fset_currents.V + fset_stokes.V)
elif withtides:
    fieldset = FieldSet(U=fset_currents.U + fset_tides.U, 
                        V=fset_currents.V + fset_tides.V)
else:
    fieldset = FieldSet(U=fset_currents.U, 
                        V=fset_currents.V)    

### deploy particles
   
fU = fieldset_currents.U
fieldset.computeTimeChunk(fU.grid.time[0], 1)

startlon = []
startlat = []
xmin = 80
xmax = 140
for y in range(0,70,1):
    line = np.array(fU.data[0,y,xmin:xmax])
    I = np.where(line==0)[0]
    if len(I)>0 and len(I)<60:
        startlon.append(fU.grid.lon[I[0]-1+xmin])
        startlat.append(fU.grid.lat[y])

######################## EXECUTE ############################

def DeleteParticle(particle, fieldset, time):
    particle.delete()

pset = ParticleSet(fieldset=fieldset,
                   pclass=JITParticle,
                   lon=startlon,
                   lat=startlat)

fname = path.join(data_out, fname + ".nc") 
outfile = pset.ParticleFile(name=fname, outputdt=delta(hours=output_frequency))

pset.execute(AdvectionRK4,
             runtime=delta(days=advection_duration),
             dt=delta(hours=-1),
             output_file=outfile,
             recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})

outfile.export()
outfile.close()  