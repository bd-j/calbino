#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script demonstrating the use of the C3K model library to estimate the
calibration vector from a binospec observation of an F-star
"""

import numpy as np
import h5py
from sedpy.observate import load_filters, getSED


lightspeed = 3e18


def choose_model(mags, filters, libwave, libflux, nmod=1):
    """Choose the library model with colors closest to the input colors

    :param mags:
        calibrator star magnitudes, ndarray

    :param filters:
        list of sedpy.Filter objects, same length as `mags`

    :param libwave:

    :param libflux:

    :param nmod: integer, optional, default: 1
        return the `nmod` closest models in color space.  Getting multiple
        models can be useful to explore calibration uncertainty.
    """
    target = mags[1:] - mags[0]
    # Get SED of models
    seds = getSED(libwave, libflux, filters)
    # get colors of models
    colors = seds[:, 1:] - seds[:, 0][:, None]
    # Get distance of models from target colors
    dist = ((target - colors)**2).sum(axis=-1)
    # choose the N-closest models
    order = np.argsort(dist)
    best = order[:nmod]
    return libflux[best, :]


def get_library(libname="data/c3k_v1.3_R5K.Fstars.h5"):
    """Here's the library, after removing stars with very non-solar metallicities

    :returns libwave:
        wavelength vector for the model stars, ndarray shape (nwave,)

    :returns flux:
        flux vectors for the model stars, ndarray of shape (nmodel, nwave)

    :returns params:
        parameter values for the models, structured ndarray of shape (nmodel,)
    """
    # Load the library of stellar models
    with h5py.File(libname, "r") as lib:
        wave = lib["wavelengths"][:]
        params = lib["parameters"][:]
        # restrict to ~solar metallicity
        g = (params["afe"] == 0) & (params["feh"] > -0.1) & (params["feh"] < 0.1)
        spectra = lib["spectra"][g, :]

    # convert to flambda
    flam = spectra * lightspeed / wave**2
    libwave = wave.copy()
    libflux = flam

    return libwave, libflux, params[g]


if __name__ == "__main__":

    # Here you'd get the bino spectrum for the calibrator star.
    blob = np.genfromtxt("star.csv", delimiter=",",
                         unpack=True, skip_header=1)
    data_pix, data_wave, data_flux = blob
    # header units are 1e-19 erg/s/cm^2/AA (though this is wrong!)
    data_flux *= 1e-19
    # Get from header slit_ra?
    ra, dec = None, None

    # --- Data --- 
    # Here is where you put the SDSS mags for the star of interest.
    #
    # This is the filters and (dereddened?) magnitudes for the calibrator star
    filters = load_filters(["sdss_g0", "sdss_r0", "sdss_i0"])
    if ra is not None:
        # Get mags from SDSS directly?
        # This doesn't quite work, plus is not dereddened
        from astroquery.sdss import SDSS
        from astropy import coordinates as coords
        from astropy import units as u
        pos = coords.SkyCoord(ra, dec, unit="deg", frame='icrs')
        flds = ["ra", "dec", "psfMag_g", "psfMag_r", "psfMag_i",
                "probPSF", "run", "rerun", "camcol", "field"]
        phot = SDSS.query_region(pos, radius=1*u.arcsec, spectro=False,
                                 photoobj_fields=flds)

        assert phot[0]["probPSF"] > 0
        star_mags = np.array([phot[0]["psfMag_g"], phot[0]["psfMag_r"],
                              phot[0]["psfMag_i"]])
    else:
        # Put them in by hand
        star_mags = np.array([23., 23.02, 23.1])

    # Get a reasonable set of model spectra
    libwave, libflux, libparams = get_library()

    # choose the model(s) with colors closest to the calibrator
    best_model = choose_model(star_mags, filters, libwave, libflux, nmod=1)

    # Now work out the normalization of the model from the (weighted?)
    # average magnitude offset.
    best_sed = getSED(libwave, best_model, filters)
    dm = np.mean(star_mags - best_sed, axis=-1)
    conv = np.atleast_1d(10**(-0.4 * dm))

    # Here, finally, is the fluxed model (erg/s/cm^2/AA)
    # If you input dereddened magnitudes, then this would be before reddening.
    # Otherwise, the model will be not quite correct.
    fluxed_model = best_model * conv[:, None]

    # Now get the bestest model on the same wavelength vector as the data
    z = 0.0  # redshift of the star, if known.
    a = (1 + z)
    fluxed_model_interp = np.interp(data_wave, libwave * a, fluxed_model[0])
    calibration = fluxed_model_interp / data_flux

    # You probably want to median filter the calibration vector. Perhaps after
    # some sigma clipping.  Differences on small scales could be due to model
    # imperfections (wrong metallicity, wrong gravity, wrong redshift for model,
    # LSF mismatch.) You could also fit the calibration vector with a
    # polynomial, taking into account uncertainties on the spectrum
    from scipy.signal import medfilt
    smoothed_calibration = medfilt(calibration, 101)

    # --- Plot ---
    import matplotlib.pyplot as pl
    fig, axes = pl.subplots(3, 1, sharex=True, figsize=(13, 11))
    ax = axes[0]
    ax.plot(data_wave, calibration, label="raw calibration")
    ax.plot(data_wave, smoothed_calibration, label="smoothed calibration")
    ax.legend()
    ax.text(0.1, 0.9, "Fluxed spectrum = $\mathcal{C}$ * (Instrumental Spectrum)",
            transform=ax.transAxes)
    ax.set_ylabel("$\mathcal{C}$")

    ax = axes[1]
    ax.plot(data_wave, data_flux, label="Bino spectrum")
    ax.plot(libwave, fluxed_model[0], label="Fluxed model")
    ax.set_xlim(data_wave.min(), data_wave.max())
    ax.legend()
    ax.set_ylabel("$F_{\lambda} \, (erg/s/cm^2/\AA)$")

    ax = axes[2]
    [f.display(ax=ax) for f in filters]
    ax.set_ylabel("Transmission")
    ax.set_label("$\lambda (\AA)$")

    fig.savefig("example_calbino.pdf")
