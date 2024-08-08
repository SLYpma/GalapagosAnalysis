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
withwind   = 0.01  #Scaling factor 0.01 
seedingEEZ = True  #false is seeding GMR

if seedingEEZ:
    fname = "MITgcm_EEZseeding"
    seeding_frequency = 1     #unit: days (at which frequency do we deploy particles)
else:
    fname = "MITgcm_GMRseeding"
    seeding_frequency = 5     #unit: days (at which frequency do we deploy particles)

data_in = "/data/oceanparcels/input_data"
data_out = "/scratch/SLYpma/output/"
domain = [-97, -84, -6, 6]

#run details
length_simulation = 4*365 #unit: days (for how long do we deploy particles)
advection_duration = 4*30 #unit: days (how long does one particle advect in the fields)
output_frequency = 12     #unit: hours

############ GET INDICES ####################
def getclosest_ij(lats,lons,latpt,lonpt):    
    """Function to find the index of the closest point to a certain lon/lat value."""
    dist_lat = (lats-latpt)**2          # find squared distance of every point on grid
    dist_lon = (lons-lonpt)**2
    minindex_lat = dist_lat.argmin()    # 1D index of minimum dist_sq element
    minindex_lon = dist_lon.argmin()
    return minindex_lat, minindex_lon   # Get 2D index for latvals and lonvals arrays from 1D index

dfile = Dataset(data_in+'/MITgcm4km/RGEMS3_Surf_grid.nc')
lon = dfile.variables['XC'][:]
lat = dfile.variables['YC'][:]
iy_min, ix_min = getclosest_ij(lat, lon, domain[2], domain[0])
iy_max, ix_max = getclosest_ij(lat, lon, domain[3], domain[1])

dfile = Dataset(data_in+'/CMEMS/CERSAT-GLO-BLENDED_WIND_L4_REP-V6-OBS_FULL_TIME_SERIE/2018010100-IFR-L4-EWSB-BlendedWind-GLO-025-6H-REPv6-20190218T162348-fv1.0.nc')
lonw = dfile.variables['lon'][:]
latw = dfile.variables['lat'][:]
wy_min, wx_min = getclosest_ij(latw, lonw, domain[2]-0.5, domain[0]-0.5)
wy_max, wx_max = getclosest_ij(latw, lonw, domain[3]+0.5, domain[1]+0.5)

dfile = Dataset(data_in+'/WaveWatch3data/CFSR/WW3-GLOB-30M_200801_uss.nc')
lons = dfile.variables['longitude'][:]
lats = dfile.variables['latitude'][:]
sy_min, sx_min = getclosest_ij(lats, lons, domain[2]-0.5, domain[0]-0.5)
sy_max, sx_max = getclosest_ij(lats, lons, domain[3]+0.5, domain[1]+0.5)

###########################    ADD HYDRODYNAMIC FIELDS      ###########################

varfiles = sorted(glob(data_in + "/MITgcm4km/RGEMS_20*.nc"))
meshfile = glob(data_in + "/MITgcm4km/RGEMS3_Surf_grid.nc")
files = {'U': {'lon': meshfile, 'lat': meshfile, 'data': varfiles},
         'V': {'lon': meshfile, 'lat': meshfile, 'data': varfiles}}
variables = {'U': 'UVEL', 'V': 'VVEL'}
dimensions = {'lon': 'XG', 'lat': 'YG', 'time': 'time'}
indices = {'lon': range(ix_min,ix_max), 
           'lat': range(iy_min,iy_max)}
fset_eulerian = FieldSet.from_mitgcm(files,
                                     variables,
                                     dimensions,
                                     indices = indices)

def make_fset_windfiles(withwind, wx_min, wx_max, wy_min, wy_max, data_in):
    wind_files = sorted(glob(data_in + "/CMEMS/CERSAT-GLO-BLENDED_WIND_L4_REP-V6-OBS_FULL_TIME_SERIE/20*.nc"))
    wind_indices = {'lon': range(wx_min,wx_max), 
                    'lat': range(wy_min,wy_max)}
    wind_dimensions = {'lon': 'lon', 'lat': 'lat', 'time': 'time'}
    wind_variables = {'U': 'eastward_wind', 'V': 'northward_wind'}
    fset_wind = FieldSet.from_netcdf(wind_files, 
                                     wind_variables, 
                                     wind_dimensions, 
                                     indices=wind_indices)

    fset_wind.U.set_scaling_factor(withwind)
    fset_wind.V.set_scaling_factor(withwind)
    fset_wind.U.units = GeographicPolar()
    fset_wind.V.units = GeographicPolar()
    return fset_wind

def make_fset_stokesfiles(sx_min, sx_max, sy_min, sy_max, data_in):
    stokes_files = sorted(glob(data_in + "/WaveWatch3data/CFSR/WW3-GLOB-30M_200[8-9]*_uss.nc"))
    stokes_files += sorted(glob(data_in + "/WaveWatch3data/CFSR/WW3-GLOB-30M_201[0-2]*_uss.nc"))
    stokes_indices = {'lon': range(sx_min,sx_max), 
                      'lat': range(sy_min,sy_max)}
    stokes_dimensions = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}
    stokes_variables = {'U': 'uuss', 'V': 'vuss'}
    fset_stokes = FieldSet.from_netcdf(stokes_files, 
                                       stokes_variables, 
                                       stokes_dimensions, 
                                       indices=stokes_indices)
    fset_stokes.U.units = GeographicPolar()
    fset_stokes.V.units = GeographicPolar()
    return fset_stokes

if withstokes and withwind:
    fset_wind = make_fset_windfiles(withwind, wx_min, wx_max, wy_min, wy_max, data_in)
    fset_stokes = make_fset_stokesfiles(sx_min, sx_max, sy_min, sy_max, data_in)
    fieldset = FieldSet(U=fset_eulerian.U + fset_stokes.U + fset_wind.U,
                        V=fset_eulerian.V + fset_stokes.V + fset_wind.V)  
    fname += '_total'
else:
    if withstokes:
        fset_stokes = make_fset_stokesfiles(sx_min, sx_max, sy_min, sy_max, data_in)
        fieldset = FieldSet(U=fset_eulerian.U + fset_stokes.U,
                            V=fset_eulerian.V + fset_stokes.V)  
        fname += '_nowind'
    elif withwind:
        fset_wind = make_fset_windfiles(withwind, wx_min, wx_max, wy_min, wy_max, data_in)
        fieldset = FieldSet(U=fset_eulerian.U + fset_wind.U,
                            V=fset_eulerian.V + fset_wind.V)  
        fname += '_nostokes'
    else:
        fieldset = fset_eulerian
        fname += '_eulerian'
        
    
#######################      ADD OTHER FIELDS         #######################

file = open('../input/inputfiles_MITgcm', 'rb')
inputfiles = pickle.load(file)
file.close()

fieldset.add_field(Field('distance',
                         data = inputfiles['distance'],
                         lon = inputfiles['lon_high'],
                         lat = inputfiles['lat_high'],
                         mesh='spherical',
                         interp_method = 'linear'))

fieldset.add_field(Field('island',
                         data = inputfiles['seaborder'],
                         lon = inputfiles['lon_high'],
                         lat = inputfiles['lat_high'],
                         mesh='spherical',
                         interp_method = 'nearest'))

fieldset.add_field(Field('landmask',
                         data = inputfiles['landmask'],
                         lon = inputfiles['lon'],
                         lat = inputfiles['lat'],
                         mesh='spherical',
                         interp_method = 'nearest'))

fieldset.add_field(Field('coastcells',
                         data = inputfiles['coastcells'],
                         lon = inputfiles['lon'],
                         lat = inputfiles['lat'],
                         mesh='spherical',
                         interp_method = 'nearest'))

fieldset.add_constant('advection_duration',advection_duration)    
fieldset.add_constant('lon_max',lon[ix_max-5])
fieldset.add_constant('lon_min',lon[ix_min+5])
fieldset.add_constant('lat_max',lat[iy_max-5])
fieldset.add_constant('lat_min',lat[iy_min+5])

#######################     ADD KERNELS       #############################

def AdvectionRK4(particle, fieldset, time):
    """ Only advect particles that are not out of bounds"""
    if (particle.lon < fieldset.lon_max and
        particle.lon > fieldset.lon_min and
        particle.lat < fieldset.lat_max and
        particle.lat > fieldset.lat_min):
    
        (u1, v1) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
        lon1, lat1 = (particle.lon + u1*.5*particle.dt, particle.lat + v1*.5*particle.dt)
        
        if (lon1 < fieldset.lon_max and
            lon1 > fieldset.lon_min and
            lat1 < fieldset.lat_max and
            lat1 > fieldset.lat_min):

            (u2, v2) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat1, lon1]
            lon2, lat2 = (particle.lon + u2*.5*particle.dt, particle.lat + v2*.5*particle.dt)

            if (lon2 < fieldset.lon_max and
                lon2 > fieldset.lon_min and
                lat2 < fieldset.lat_max and
                lat2 > fieldset.lat_min):

                (u3, v3) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat2, lon2]
                lon3, lat3 = (particle.lon + u3*particle.dt, particle.lat + v3*particle.dt)

                if (lon3 < fieldset.lon_max and
                    lon3 > fieldset.lon_min and
                    lat3 < fieldset.lat_max and
                    lat3 > fieldset.lat_min):

                    (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3]
                    particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
                    particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt

def BeachTesting(particle, fieldset, time):  
    """Check whether particle is on land"""
    if (particle.lon < fieldset.lon_max and
        particle.lon > fieldset.lon_min and
        particle.lat < fieldset.lat_max and
        particle.lat > fieldset.lat_min):

        landcheck = fieldset.landmask[time, particle.depth, particle.lat, particle.lon]
        if landcheck == 1: 
            particle.beached = 1

def WithinCoastalCell(particle, fieldset, time):
    """Check whether particle is in coastal cell and for how long in same cell"""
    if (particle.lon < fieldset.lon_max and
        particle.lon > fieldset.lon_min and
        particle.lat < fieldset.lat_max and
        particle.lat > fieldset.lat_min):
        
        particle.coastcell = fieldset.coastcells[time, particle.depth, particle.lat, particle.lon]
    
        if particle.coastcell>0: #particle in a coastal cell
            if particle.coastcell == particle.prev_coastcell: #particle stayed in the same cell
                particle.dt_coastcell = particle.prev_dt + 1
            else: #particle has moved on to another cell
                particle.dt_coastcell = 0
            if particle.prev_dt == 0 and particle.dt_coastcell == 1: #how often is particle in the same cell
                particle.f_coastcell = particle.prev_f + 1
        else:
            particle.dt_coastcell = 0

        particle.prev_coastcell = particle.coastcell
        particle.prev_dt = particle.dt_coastcell
        particle.prev_f = particle.f_coastcell
        
##

def Age(fieldset, particle, time):
    """ Delete particles when reaching age specified by advection_duration """
    particle.age = particle.age + math.fabs(particle.dt)
    if particle.age > fieldset.advection_duration*86400:
        particle.delete()

def SampleInfo(fieldset, particle, time):
    if (particle.lon < fieldset.lon_max and
        particle.lon > fieldset.lon_min and
        particle.lat < fieldset.lat_max and
        particle.lat > fieldset.lat_min):
        
        particle.distance = fieldset.distance[time, particle.depth, particle.lat, particle.lon]
        particle.island = fieldset.island[time, particle.depth, particle.lat, particle.lon]    

#######################     ADD PARTICLE BEHAVIOR       #############################

if seedingEEZ:
    startlon = inputfiles['lonEEZ'][::7]
    startlat = inputfiles['latEEZ'][::7]
else:
    startlon = inputfiles['lonGMR'][::20]
    startlat = inputfiles['latGMR'][::20]
    
    
class GalapagosParticle(JITParticle):
    age = Variable('age', dtype=np.float32, initial = 0.)
    distance = Variable('distance', dtype=np.float32, initial = 0.)
    island = Variable('island', dtype=np.int32, initial = 0.)
    beached = Variable('beached', dtype=np.int32, initial = 0.) 
    coastcell = Variable('coastcell', dtype=np.float32, initial = 0.)
    dt_coastcell = Variable('dt_coastcell', dtype=np.int32, initial = 0.)
    f_coastcell = Variable('f_coastcell', dtype=np.int32, initial = 0.)
    prev_coastcell = Variable('prev_coastcell', dtype=np.float32, to_write=False, initial = 0.) 
    prev_dt = Variable('prev_dt', dtype=np.int32, to_write=False, initial = 0.) 
    prev_f = Variable('prev_f', dtype=np.int32, to_write=False, initial = 0.)
    
pset = ParticleSet(fieldset=fieldset,
                   pclass=GalapagosParticle,
                   lon=startlon,
                   lat=startlat,
                   repeatdt=delta(days=seeding_frequency))

#######################     EXECUTE       #############################

kernel = (pset.Kernel(AdvectionRK4) + 
          pset.Kernel(BeachTesting) +
          pset.Kernel(WithinCoastalCell) +
          pset.Kernel(Age) + 
          pset.Kernel(SampleInfo))

filename = path.join(data_out, fname + ".nc")
outfile = pset.ParticleFile(name=filename, outputdt=delta(hours=output_frequency))

pset.execute(kernel,
             runtime=delta(days=length_simulation),
             dt=delta(hours=1),
             output_file=outfile)

pset.repeatdt = None

pset.execute(kernel,
             runtime=delta(days=advection_duration),
             dt=delta(hours=1),
             output_file=outfile)

outfile.export()
outfile.close()





