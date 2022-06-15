"""

Created By Manish S. Devana 3/38/22

Tools for loading and using the OSNAP data
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import scipy.interpolate as interper
from scipy.integrate import trapz
from scipy.io import loadmat
import numpy.ma as ma
import scipy.signal as signal
import scipy.stats as stats

# params
defaultMooringGridPath = '/Users/manishdevana/Research/data_storage/osnap_6yr_trimmed.mat'
defaultAdtPath = '/Users/manishdevana/Research/data_storage/adt_icelandBasin_1993_2020.nc'

defaultGebcoPath='/Users/manishdevana/Research/data_storage/GEBCO_2019/GEBCO_2019.nc'

print('loading topography (call loadTopo() to get dataset')
topo = loadmat('/Users/manishdevana/Research/data_storage/OSNAP_data/osnap_section_topo_new.mat')

# data loading modules 

# plot parameters
plotParams = dict(
    bathyBottom=3000,
    bathyColor='k',
    bathyMaskColor='grey',
    mooringLineColor='k',
    mooringLineStye='--',
    mooringGridxlabelDist='[km]',
    mooringGridylabelDepth='[m]'
)

def loadTopo():
    """
    


    Args:
        topo (_type_): _description_
    """
    topo = loadmat('/Users/manishdevana/Research/data_storage/OSNAP_data/osnap_section_topo_new.mat')


    return topo

def loadGriddedMooringData(path=defaultMooringGridPath, applyTopoMasking=True):
    matdata = loadmat(path)
    pt = matdata['ptmp_grid_mask']
    sal = matdata['sal_grid_mask']
    X = np.squeeze(matdata['xplot'])
    Z = np.squeeze(matdata['zplot'])
    # xbathy = np.squeeze(matdata['xfill'])
    # ybathy = np.squeeze(matdata['yfill'])
    dates = pd.to_datetime(matdata['dates'])
    vtran = matdata['vtran_grid']
    sigt = matdata['sigt_grid']
    # dxdy = matdata['dxdy']
    xgrid = matdata['xgrid']

    mooring = xr.Dataset(
            
        {
            'vtrans':(['time', 'z', 'x'], vtran),
            'sal':(['time','z','x'], sal),
            'theta':(['time','z','x'], pt),
            'sig':(['time','z','x'], sigt),
        },
        coords={
            'time':dates,
            'x':X.flatten(),
            'z':Z.flatten(),
    #         'pres':pb.flatten()
        }
        
    )
    if applyTopoMasking:
        mooring = mooring.where(np.isfinite(mooring.sal))
        mooring= mooring.dropna(dim='z', how='all').dropna(dim='x', how='all')

    return mooring


def loadIcelandBasinAdt(path=defaultAdtPath):
    return xr.open_dataset(path)




def isowTransportFromGrid(ds, xbounds=None,
                            v='vtrans',
                            isowLine=27.8, 
                            returnAsDataArray=True):
    """
    Calculate ISOW transport for a given set of xbounds 

    Args:
        ds (_type_): _description_
    """

    # apply xbounds if given
    if xbounds:
        ds = ds.sel(x=slice(xbounds[0], xbounds[1]))

    # calcu new transport
    masked = ds[v].where(ds.sig >= 27.8)
    vtrans = ma.MaskedArray(masked.values, mask = np.isnan(masked.values))
    vint =  trapz(vtrans, x=ds.z.values, axis=1)
    transport = trapz(vint, x=ds.x.values*1e3, axis=1)*1e-6

    if returnAsDataArray:
        transport = xr.DataArray(transport, coords=dict(time=ds.time.values))

        return transport # return as xarray dataarry
    else:
        return transport # returns as just an array



def normalize(x, a=-1, b=1):
    return ((b-a) * ((x - np.nanmin(x))/(np.nanmax(x) - np.nanmin(x)))) + a


def correlateWlags(x,y):
    mask = np.logical_and(
        np.isfinite(x),
        np.isfinite(y)

    )
    xcorr = signal.correlate(x[mask],y[mask],
    mode='full',
    method='direct'
    )
    lags = signal.correlation_lags(len(x[mask]), len(y[mask]),
    )
    fig = plt.figure(dpi=300)
    # xcorr /= len(x) * np.nanstd(x)*np.nanstd(y)
    xcorr /= np.sqrt(np.nansum(x**2)*np.nansum(y**2))
    plt.plot(lags, xcorr)
    
    return xcorr, lags, fig, plt.gca()

def lowpass(x, Wn, fs, N=3):
    """
    Low pass filter

    Args:
        sig (_type_): _description_
        Wn (_type_): _description_
        fs (_type_): _description_
    """

    b,a = signal.butter(N, Wn, btype='low')
    mask = np.isfinite(x)
    lowpassed = x.copy()
    lowpassed[mask] = signal.filtfilt(b, a, x[mask])
    return lowpassed

def lineRegressAndScatter(x, y, dpi=400, figsize=(4,4), c='k', s=1, m='s'):
    plt.figure(dpi=dpi, figsize=figsize)
    ax4plot = plt.axes()
    ax4plot.scatter(x,y, c=c, s=s, marker=m)
    mask = np.logical_and(
        np.isfinite(x),
        np.isfinite(y)
    )
    if np.sum(mask) > 3:
        slope, intercept, r = stats.linregress(x[mask],y[mask])[:3]
    else:
        slope, intercept, r = stats.linregress(x,y)[:3]

    fx = lambda x: slope*x + intercept
    ax4plot.plot(x, fx(x), lw=.4, c='r')
    ax4plot.text(.1, .8, 'r={:.2f}'.format(r), transform=ax4plot.transAxes)

def lineRegressLine(x, y, ax4plot):
    mask = np.logical_and(
        np.isfinite(x),
        np.isfinite(y)
    )
    if np.sum(mask) > 3:
        slope, intercept, r = stats.linregress(x[mask],y[mask])[:3]
    else:
        slope, intercept, r = stats.linregress(x,y)[:3]

    fx = lambda x: slope*x + intercept
    ax4plot.plot(x, fx(x), lw=.4, c='r')
    ax4plot.text(.1, .8, 'r={:.2f}'.format(r), transform=ax4plot.transAxes)

    
def blankOSNAPGridPlot(figsize=(6,3), 
                        dpi=400,
                        xlabel='distance [km]', ylabel='depth [m]',
                        topo=topo,
                        ylim=[3000,1150],
                        xlim=[0, 330],
                        plotParams=plotParams
                        
                        ):

    """
    Makes a blank osnap grid plot to save me from having to do it myself a whole bunch

    """
    fig, ax = plt.subplots(1,1, figsize=figsize, dpi=dpi)

    bottom=plotParams['bathyBottom'],
    ax.fill_between(topo['dist'].flatten(), topo['depth_mask'].flatten()*-1,
    bottom,
    color=plotParams['bathyMaskColor'],
    )

    # osnap topography
    ax.fill_between(topo['dist'].flatten(), topo['depth'].flatten()*-1,
    bottom,
    color=plotParams['bathyColor'],
    zorder=1000
    )

    ax.set_xlabel(plotParams['mooringGridxlabelDist'])
    ax.set_ylabel(plotParams['mooringGridylabelDepth'])

    ax.set_ylim(ylim)
    ax.set_xlim(xlim)


    return fig, ax



def addMooringLines(fig, ax,zorder=2, lc='white', lw=2, alpha=.5, ls='--', topoDict=topo, params=plotParams):
    """
    

    Args:
        fig (_type_): _description_
        ax (_type_): _description_
        topoDict (_type_, optional): _description_. Defaults to topo.
        params (_type_, optional): _description_. Defaults to plotParams.
    """

    for mx in topoDict['moor_dis'].flatten():
        ax.axvline(mx, ls=ls, color=lc, linewidth=lw, alpha=alpha)

    return ax



def xarrayVerticalProfilePlot(x, z, ax, plotStdBand=True, lw=2, c='k', tdim='time', alpha=.5):
    """
    

    Args:
        x (_type_): _description_
        z (_type_): _description_
        ax (_type_): _description_
        plotStdBand (bool, optional): _description_. Defaults to True.
        lw (int, optional): _description_. Defaults to 2.
        c (str, optional): _description_. Defaults to 'k'.
    """

    xmean = x.mean(dim=tdim)
    xstd = x.std(dim='time')

    ll = ax.plot(xmean, z, lw=lw, c=c)

    if plotStdBand:
        ax.fill_betweenx(z, xmean-xstd, xmean+xstd, color=c,alpha=alpha )

    return ll




def loadGebco(path=defaultGebcoPath, lonLims=slice(-40, -10), latlims=slice(50, 70)):
    """
    Load GEBCO dataset cut for iceland basin.


    Args:
        lonLims (_type_, optional): _description_. Defaults to slice().
        latlims (_type_, optional): _description_. Defaults to slice().
    """

    geb = xr.open_dataset(path).sel(lon=lonLims, lat=latlims)

    return geb
