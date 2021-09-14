# conda create -n pyMPAS xarray dask netcdf4 zarr seaborn pandas numpy scipy
# conda install -n pyMPAS -c conda-forge ncl
# conda activate pyMPAS

import xarray as xr
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
import os


def plot_var_mpas(data, var, file, title='nCells'):
    if not os.path.exists(file):
        palette = "cubehelix_r"
        num = data[var].count()
        sns.scatterplot(x='longitude', y='latitude', linewidth=0,
                        hue=var, data=data, palette=palette)
        if title == 'nCells':
            title = 'Cells ' + str(num)
        plt.gca().set_aspect('equal')
        plt.legend()
        plt.title(title)
        plt.savefig(file)
        plt.close()
        
        
file = sys.argv[1]

if not os.path.exists(file + '-reduced'):
    grid = xr.open_dataset(file)
    
    number_cells = grid.dims['nCells']
    print(number_cells)
    
    area = grid['areaCell']
    if 'units' not in area.attrs:
        print('Rescale area to the Earth Sphere')
        area = area * (6371000)**2
        area['units'] = 'm^2'
        
    print(area)
    
    resolution = 2*(xr.apply_ufunc(np.sqrt, area/math.pi))*10**(-3)
    resolution = resolution.rename('resolution')
    resolution.attrs['long_name'] = 'Resolution of the cell (approx)'
    resolution.attrs['units'] = 'km'
    
    print(resolution)
    
    lats = grid['latCell']*180/math.pi
    lats = lats.rename('latitude')
    lats.attrs['long_name'] = 'Latitude Cell'
    lats.attrs['units'] = 'degrees'
    
    lons = grid['lonCell']*180/math.pi
    for i in range(len(lons)):
        if lons[i] > 180.0:
            lons[i] = lons[i] - 360.0
    lons = lons.rename('longitude')
    lons.attrs['long_name'] = 'Longitude Cell'
    lons.attrs['units'] = 'degrees'
    
    reduced = xr.merge((lons, lats, area, resolution))
    reduced.to_netcdf(file + '-reduced')
else:
    reduced = xr.open_dataset(file + '-reduced')

pds = reduced.to_dataframe()
print(pds.describe())

areas = np.sort(pds['areaCell'].values)
print(areas)
plt.plot(areas)
plt.yscale('log')
plt.ylabel('Cell Area')
plt.savefig('hist_areaCell.png')
plt.close()

reso = np.sort(pds['resolution'].values)
print(reso)
plt.plot(reso)
plt.yscale('log')
plt.ylabel('Resolution')
plt.savefig('hist_resolution.png')
plt.close()

exit()
ind = pds[['resolution']].idxmin()
point = [pds.iloc[ind]['longitude'].values[0], pds.iloc[ind]['latitude'].values[0]]
print(point)
point = [0.0, 0.0]

pds['round_res'] = pds['resolution'].round(2)
plot_var_mpas(pds, 'round_res', 'round_res.png')

plot_var_mpas(pds, 'resolution', 'res.png')

for lim in [1.0, 2.0, 5.0, 10.0]:
    pds_small = pds.where((pds['longitude'] < point[0] + lim)
                          & (pds['longitude'] > point[0] - lim)
                          & (pds['latitude'] < point[1] + lim)
                          & (pds['latitude'] > point[1] - lim))
    plot_var_mpas(pds_small, 'round_res', 'centered_margin_' + str(lim) + '.png')


for limit in [15.0, 30.0, 60.0]:
    good = pds.where(pds['round_res'] < limit)
    plot_var_mpas(good, 'round_res', 'smaller_than_' + str(limit) + '.png')

max_res = pds[['resolution']].max().values[0]
print(max_res)
min_res = pds[['resolution']].min().values[0]
print(min_res)
boundary = pds.where(pds['resolution'] > 2.0)
plot_var_mpas(boundary, 'resolution', 'boundary.png')

print('Max Radius in degrees')
max_radi = pds[['longitude', 'latitude']].abs().max().values[0]
print(max_radi)

# lon = pds['longitude'] - 2.0
# lat = pds['latitude'] - 40.0
# max_lon = lon.abs().max()
# max_lat = lat.abs().max()
# print(max(max_lat, max_lon))

print('Total area')
area_total = pds['areaCell'].sum()
print(area_total/(10**6))