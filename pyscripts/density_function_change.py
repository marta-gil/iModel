#! /usr/bin/env python3
# ----------------------------------------------------
# This script generates the density function data
# defined in a lat-lon grid
# Output: densf_table.dat (in altitude directory)
# To use this data in iModel grids generator, you must
# set the following options in the mesh.par file:
#
# !Kind of mesh use: icos, octg, read, rand
# icos
# !Position: eqs, pol, ran, ref, readref, readref_andes
# readref
# !Optimization of mesh: nopt, sprg, scvt, salt, hr95
# scvt
# ----------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss

# ----------------------------------
# converts degrees to radians
deg2rad = np.pi / 180.0
# ----------------------------------
# converts degrees to radians
rad2deg = 1.0 / deg2rad


# ----------------------------------

# -----------------------------------------------------------------------
# Transforms geographical coordinates (lat,lon) to Cartesian coordinates.
# -----------------------------------------------------------------------
def sph2cart(lat, lon):
    coslat = np.cos(lat)
    x = coslat * np.cos(lon)
    y = coslat * np.sin(lon)
    z = np.sin(lat)
    return x, y, z


# -----------------------------------------------------------------------
# Compute the density function
# -----------------------------------------------------------------------

def density_function_dists(dists, gammas=20, radkm=100, epsilons=8000):
    # auxilary function
    sx = (dists - radkm) / epsilons

    # Boolean indexes
    # point closer to the center
    closer_to_center = (dists <= radkm)

    # point in transition zone
    transition_zone = (dists <= radkm + epsilons) & (dists > radkm)

    # point far from the center
    far_from_center = (dists > radkm + epsilons)

    # set density
    dens_f = np.zeros(np.shape(dists))
    dens_f[closer_to_center] = gammas ** 4
    dens_f[transition_zone] = ((1.0 - sx[transition_zone]) *
                               gammas + sx[transition_zone]) ** 4
    dens_f[far_from_center] = 1.0

    # normalization - make it in [0,1]
    dens_f = dens_f / gammas ** 4
    return dens_f


def density_function(lat, lon, **kwargs):
    lat, lon = map(np.radians, [lat, lon])

    haver_formula = np.sin(lat / 2.0) ** 2 + \
                    np.cos(lat) * np.sin(lon / 2.0) ** 2

    dists = 2 * np.arcsin(np.sqrt(haver_formula)) * 6367

    dens_f = density_function_dists(dists, **kwargs)
    return dens_f


def density_to_resolution(dens, N=200000):
    sum_dens = dens.sum()
    print(sum_dens)

    res = N * dens / sum_dens
    print(res)
    return res


dists = np.linspace(0.0, 10000.0, 50000)
dens = density_function_dists(dists)
res = density_to_resolution(dens)

plt.plot(dists, res)
plt.show()
plt.close()

