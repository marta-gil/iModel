import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import math
import numpy as np
import os
import pandas as pd


def density_function_dists(dists, slope=None, gammas=None, maxdist=None,
                           maxepsilons=None):

    epsilons = gammas /slope
    if epsilons > maxepsilons:
        epsilons = maxepsilons

    # initialize with density = 1
    dens_f = np.ones(np.shape(dists))

    # point in transition zone
    transition_zone = (dists > maxdist) & (dists <= maxdist + epsilons)
    sx = (dists -maxdist ) *slope
    transition_values = 1.0 + sx
    dens_f = np.where(transition_zone, transition_values, dens_f)

    # further points
    far_from_center = (dists > maxdist + epsilons)
    dens_f[far_from_center] += epsilons *slope

    dens_f = 1 / dens_f**2
    return dens_f


def latlon_to_distance_center(lat, lon):
    lat, lon = map(np.radians, [lat, lon])

    haver_formula = np.sin(lat / 2.0) ** 2 + \
                    np.cos(lat) * np.sin(lon / 2.0) ** 2

    dists = 2 * np.arcsin(np.sqrt(haver_formula)) * 6367
    return dists


def density_function(lat, lon, **kwargs):
    dists = latlon_to_distance_center(lat, lon)

    dens_f = density_function_dists(dists, **kwargs)
    return dens_f


def density_to_resolution(dens, area, N=200000):
    sum_dens = dens.sum()
    cells_by_point = dens * N /sum_dens
    area_cells = area /cells_by_point
    res = 2* np.sqrt(area_cells / np.pi)
    return res


def areas(lon, lat, step=0.1):
    area1 = (step * 110) ** 2 * abs(np.cos(np.radians(lat)))
    area = np.tile(area1, (lon.shape[0], 1)).transpose()
    return area


def dens2res(N_cells, step=0.01, plot=True, **kwargs):
    lats0 = np.arange(-90.0, 90.0, step)
    lons0 = np.arange(-180.0, 180.0, step)

    lons, lats = np.meshgrid(lons0, lats0)

    area = areas(lons0, lats0, step=step)

    dens = density_function(lats, lons, **kwargs)

    sum_dens = dens.sum()

    # old method
    neededNcells = (2 * 110 * step) ** 2 / np.pi * sum_dens

    print('Needed cells for lower resolution 1km')
    print(neededNcells)

    resolution_cells = density_to_resolution(dens, area, N=N_cells)

    print('Number smaller than 15km')
    print((resolution_cells < 15.0).sum())

    if plot:
        rangedens = dens[len(lats0) // 2,
                    len(lons0) // 2 - 50:len(lons0) // 2 + 50]
        plt.plot(lons0[len(lons0) // 2 - 50:len(lons0) // 2 + 50], rangedens)
        plt.title("dens")
        plt.show()
        plt.close()

        dists = lons0 * 111

        plt.plot(dists, dens[len(lats0) // 2, :], 'red')
        plt.ylabel('density')
        plt.gca().yaxis.label.set_color('red')
        ax2 = plt.gca().twinx()
        ax2.plot(dists, resolution_cells[len(lats0) // 2, :], 'green')
        ax2.yaxis.label.set_color('green')
        ax2.set_ylabel('resolution')
        plt.show()
        plt.close()

        rangedists = dists[len(lons0) // 2 - 50:len(lons0) // 2 + 50]
        rangeres = resolution_cells[len(lats0) // 2,
                   len(lons0) // 2 - 50:len(lons0) // 2 + 50]

        plt.plot(rangedists, rangeres)
        plt.title("diam")
        plt.show()
        plt.close()

        plt.plot(rangedists, rangedens, 'red')
        plt.gca().yaxis.label.set_color('red')
        plt.ylabel('density')
        ax2 = plt.gca().twinx()
        ax2.plot(rangedists, rangeres, 'green')
        ax2.yaxis.label.set_color('green')
        ax2.set_ylabel('resolution')
        plt.show()
        plt.close()

    return resolution_cells


def read_reduced_grid(file, erase=False):
    if erase:
        os.system('rm -rf ' + file + '-reduced')

    if not os.path.exists(file + '-reduced'):
        grid = xr.open_dataset(file)

        area = grid['areaCell']
        if 'units' not in area.attrs:
            print('Rescale area to the Earth Sphere')
            area = area * (6371000)**2
            area['units'] = 'm^2'
        area.attrs['long_name'] = 'Area of the cell in m^2'
        area = area.rename('area')

        resolution = 2*(xr.apply_ufunc(np.sqrt, area/math.pi))*10**(-3)
        resolution = resolution.rename('resolution')
        resolution.attrs['long_name'] = 'Resolution of the cell (approx)'
        resolution.attrs['units'] = 'km'

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

        distance_to_center = latlon_to_distance_center(lats, lons)
        distance_to_center = distance_to_center.rename('distance')
        distance_to_center.attrs['long_name'] = 'Distance in km to (0, 0)'
        distance_to_center.attrs['units'] = 'km'

        reduced = xr.merge((lons, lats, area, resolution, distance_to_center))
        reduced.to_netcdf(file + '-reduced')
    else:
        reduced = xr.open_dataset(file + '-reduced')
    return reduced


def plot_var_mpas(data, var, outfile=None, title=''):
    if outfile is None or not os.path.exists(outfile):
        palette = "cubehelix_r"
        num = data[var].count()
        sns.scatterplot(x='longitude', y='latitude', linewidth=0,
                        hue=var, data=data, palette=palette)
        title = title  + '\nNumber Cells ' + str(num)
        plt.gca().set_aspect('equal')
        plt.legend()
        plt.title(title)
        if outfile is not None:
            plt.savefig(outfile)
        else:
            plt.show()
        plt.close()


def plot_var_mpas_crss(data, var, file=None, show=False,
                       title='', palette="Spectral_r", ax=None, **kwargs):

    if ax is None:
        ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent([float(data['longitude'].min()),
                   float(data['longitude'].max()),
                   float(data['latitude'].min()),
                   float(data['latitude'].max())])

    ax.coastlines()
    ax.stock_img()

    sns.scatterplot(x='longitude', y='latitude', linewidth=0,
                    hue=var, data=data, legend='brief',
                    palette=palette, ax=ax, **kwargs)

    # We have to set the map's options on all axes
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    ax.set_title(title, fontsize=10)
    if file is not None:
        plt.savefig(file)
    if show:
        plt.show()
    return


def twoplots(ds, var, limres=15, show=True, file=None, figsize=None, hue_norm=None):

    smallds = ds.where(ds['resolution'] < limres)

    try:
        ln = ds[var].attrs['long_name']
        print(ln)
    except:
        ln = ''

    if hue_norm is None:
        hue_norm = (float(ds[var].min()), float(ds[var].max()))

    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': ccrs.PlateCarree()})
    plot_var_mpas_crss(ds, var, ax=axs[0], size='area', hue_norm=hue_norm)
    plot_var_mpas_crss(smallds, var, ax=axs[1], size='area', hue_norm=hue_norm,
                       title='Smaller cells (< 15km res.)')
    if figsize is None:
        figsize = [10, 5]
    fig.set_size_inches(figsize)
    if file is not None:
        plt.savefig(file)
    if show:
        plt.show()
    plt.close()


def study_mesh(file, **kwargs):
    reduced = read_reduced_grid(file, erase=True)

    for v in ['area', 'resolution']:
        vals = np.sort(reduced[v].values)
        plt.plot(vals)
        plt.yscale('log')
        plt.ylabel(v)
        plt.title(reduced[v].attrs['long_name'])
        plt.show()
        plt.close()

    print('Number smaller than 15km: ')
    print(float((reduced['resolution'] < 15.0).sum()))

    twoplots(reduced, 'resolution', limres=15, figsize=[13, 5])

    print('Different Radius')
    for lim in [50, 100, 150, 200]:
        small = reduced.where(reduced['distance'] < lim)
        plot_var_mpas_crss(small, 'resolution', show=True,
                           title='Closer than %.1fkm' % lim)

    print('Different highest resolution')
    for limit in [3.0, 5.0, 10.0, 15.0, 30.0]:
        try:
            good = reduced.where(reduced['resolution'] < limit)
            plot_var_mpas_crss(good, 'resolution', show=True,
                               title='Resolution lower than %.1fkm' % limit)
        except:
            plt.close()
            print('No cells with resolution lower than %.1fkm' % limit)

    plot_resolution_area_sns(reduced.where(reduced['distance'] < 300))


def plot_resolution_area(ds):
    relation = ds.groupby_bins('distance', bins=5000).mean()
    rangedists = relation['distance_bins'].values
    dists = [float(x.mid) for x in rangedists]
    area = relation['area'].values
    resolution_cells = relation['resolution'].values

    plt.plot(dists, area, 'blue')
    plt.ylabel('area')
    plt.gca().yaxis.label.set_color('blue')
    ax2 = plt.gca().twinx()
    ax2.plot(dists, resolution_cells, 'green')
    ax2.yaxis.label.set_color('green')
    ax2.set_ylabel('resolution')
    plt.show()
    plt.close()

    return


def plot_resolution_area_sns(ds, rounded=10):
    pds = ds.to_dataframe().dropna()
    pds['round_dist'] = rounded*((pds['distance']/rounded).astype('int')) + rounded/2
    sns.lineplot(data=pds, x='round_dist', y='resolution')
    plt.show()
    plt.close()

    return


def mpas_center_hex_tr(file, lim=10):
    grid = xr.open_dataset(file)

    ds = grid[['lonCell', 'latCell', 'areaCell']].to_dataframe().dropna()
    ds['lonCell'] *= 180/math.pi
    ds['latCell'] *= 180 / math.pi
    ds['areaCell'] *= ((6371000) ** 2)
    smllds = ds[(ds['lonCell'].abs() < lim) & (ds['latCell'].abs() < lim)]
    sns.scatterplot(data=smllds, x='lonCell', y='latCell', linewidth=0,
                    hue='areaCell', palette='viridis', size='areaCell')

    latv = grid['latVertex'] * 180 / math.pi
    lonv = grid['lonVertex'] * 180 / math.pi
    plt.scatter(lonv.values, latv.values, color='blue', s=3)

    late = grid['latEdge'] * 180 / math.pi
    lone = grid['lonEdge'] * 180 / math.pi
    plt.scatter(lone.values, late.values, color='red', s=3)

    plt.gca().set_aspect('equal')
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.gcf().set_size_inches([7, 7])

    plt.show()
    plt.close()


def read_imodel_meshes(folder, name, describe=False):
    points = {
        'center': folder + name + '_nodes.gmt',
        'vertex': folder + name + '_trcc.gmt',
        'edge_c': folder + name + '_edc.gmt',
    }
    if describe:
        print('Reading iModel meshes (.gtm files) and loading lats&lons')

    listdf = []
    for type, file in points.items():
        if type == 'edge_c':
            skiprows = [0]
        else:
            skiprows = None
        df = pd.read_fwf(file, index_col=None, skipinitialspace=True,
                         names=['longitude', 'latitude'], skiprows=skiprows)
        if describe:
            print(type)
            print('File: ' + file)
            print(df.describe())
        df['type'] = type
        listdf.append(df)
    return pd.concat(listdf, ignore_index=True)


def read_mpas_meshes(file, describe=False):
    grid = xr.open_dataset(file)
    points = {
        'center': 'Cell',
        'vertex': 'Vertex',
        'edge_c': 'Edge',
    }

    if describe:
        print('Reading MPAS mesh and loading lats&lons')
    listdf = []
    for type, keyword in points.items():
        df = grid[['lat' + keyword, 'lon' + keyword]].to_dataframe()
        df['latitude'] = df['lat' + keyword] * 180 / math.pi
        df['longitude'] = df['lon' + keyword] * 180 / math.pi
        if describe:
            print(type)
            print('Keyword: ' + keyword)
            print(df.describe())
        df['type'] = type
        listdf.append(df[['latitude', 'longitude', 'type']])

    return pd.concat(listdf, ignore_index=True)


def center_hex_tr(df=None, folder=None, name=None, ax=None, lim=10):

    if df is None:
        df = read_imodel_meshes(folder, name)
    palette = {
        'center': 'black',
        'vertex': 'blue',
        'edge_c': 'red',
    }
    finish_plot = False
    if ax is None:
        fig, ax = plt.subplots(1)
        finish_plot = True

    sns.scatterplot(data=df, x='longitude', y='latitude', hue='type',
                    palette=palette, ax=ax)

    if finish_plot:
        plt.gca().set_aspect('equal')
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.gcf().set_size_inches([7, 7])

        plt.show()
        plt.close()


def compare_imodel_mpas(mpasfile, imodelfolder, imodelname, lim=1, describe=True):

    dfmpas = read_mpas_meshes(mpasfile, describe=describe)
    dfmpas['source'] = 'mpas'
    dfimod = read_imodel_meshes(imodelfolder, imodelname, describe=describe)
    dfimod['source'] = 'imodel'
    df = pd.concat([dfmpas, dfimod])

    fig, axs = plt.subplots(3, 1)

    palette = {
        'center': 'black',
        'vertex': 'blue',
        'edge_c': 'red',
    }

    center_hex_tr(df=dfimod, ax=axs[0])
    axs[0].set_title('iModel')
    center_hex_tr(df=dfmpas, ax=axs[1])
    axs[1].set_title('MPAS')
    sns.scatterplot(data=df, x='longitude', y='latitude', hue='type',
                    palette=palette, style='source', ax=axs[2])
    axs[2].set_title('Both')
    for ax in axs:
        ax.set_aspect('equal')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    fig.set_size_inches([7, 22])

    plt.show()
    plt.close()


