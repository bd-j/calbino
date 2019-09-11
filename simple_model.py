import numpy as np
from numpy.polynomial.chebyshev import chebval, chebvander
from sedpy.observate import load_filters, getSED



def optimal_polynomial(wave, model, obs, unc, order=10, mask=slice(None)):
    """
    :param wave:
        Wavelengths, ndarray of shape (nw,)

    :param model:
        Model spectrum, same shape as `wave`

    :param obs:
        Observed spectrum, same shape as `wave`

    :param unc:
        1sigma uncertainties on the observed spectrum, same shape as `wave`

    :param mask:
        A boolean array with `True` for good pixels and `False` for bad pixels,
        ndarray of same shape as `wave`

    :param order:
        order of the polynomial

    :returns poly:
        The optimal polynomial vector, ndarray of smae shape as `wave`

    :returns coeffs:
        Coefficients of the Chabyshev polynomial.  Note these are only valid
        when used with the wavelength vector transformed into `x`
    """
    # map unmasked wavelengths to the interval -1, 1
    # masked wavelengths may have x>1, x<-1
    x = wave - wave[mask].min()
    x = 2.0 * (x / (x[mask]).max()) - 1.0
    y = (obs / model)[mask]
    yerr = (unc / model)[mask]
    yvar = yerr**2
    A = chebvander(x[mask], order)
    ATA = np.dot(A.T, A / yvar[:, None])
    ATAinv = np.linalg.inv(ATA)
    c = np.dot(ATAinv, np.dot(A.T, y / yvar))
    Afull = chebvander(x, order)
    poly = np.dot(Afull, c)

    return poly, c


def project_filter(wave, spectrum, bands=["sdss_r0"]):
    """
    :param wave:
        Wavelengths in angstroms ndarray of shape (nw,)

    :param spectrum:
        Spectrum, in units of f_lambda (magnitudes will be correct if they are
        units of erg/s/cm^2). same shape as `wave`
    """
    filters = load_filters(bands)
    mags = getSED(wave, spectrum, filterlist=filters)
    return mags


def make_model(obswave, obsspec, obsunc, libwave, libspec, z=0.0, order=10):

    a = 1 + z
    # can replace the following line with a call to prospect.utils.smoothing.smoothspec
    spec = np.interp(obswave, libwave * a, libspec)
    cal, coeffs = optimal_polynomial(obswave, spec, obsspec, obsunc, order=order)

    return cal * spec
