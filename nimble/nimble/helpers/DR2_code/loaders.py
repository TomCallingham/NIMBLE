import bhb_completeness
from astropy.table import Table


basepath = '/global/cfs/cdirs/desi/science/mws/y3kp/'

def load_bhb_data(all_quality_flags=False, get_pobs=True):
    """This function selects the BHBs from the Loa catalogue that are used for the DR2 KP."""
    catpath = basepath + 'field-stars/loa/'
    catfile = catpath + 'loa-bhb-latest.fits'

    RVT = Table.read(catfile, hdu='RVTAB')
    FT = Table.read(catfile, hdu='FIBERMAP')
    GT = Table.read(catfile, hdu='GAIA')

    survey_idx = (RVT['SURVEY'] == 'main') & (RVT['PROGRAM'] == 'bright')
    halo_idx = ~(FT['Sgr_flag'] | FT['GC_flag'] | FT['dwarf_flag'])

    if all_quality_flags: # this selection has a slightly higher quality, and in my BHB paper, I don't use the Redrock criterion
        sel = halo_idx & survey_idx & (RVT['RR_SPECTYPE'] == 'STAR')
    else:
        sel = halo_idx & survey_idx

    if get_pobs:
        ra = FT['TARGET_RA']
        dec = FT['TARGET_DEC']
        p_obs = get_p_obs(ra=ra, dec=dec, tracer='bhb', desi_release='loa', gaia_release='dr3')
        FT['p_obs'] = p_obs

    return RVT[sel], FT[sel], GT[sel]


def load_auridesi_cat(halo, view, tracer='bhb', apply_cuts=True):
    """
    Load AuriDESI mock catalog data for a specified halo, view, and tracer.

    Parameters
    ----------
    halo : int
        The halo identifier number (6, 16, 21, 23, 24, 27).
    view : int
        The viewing angle/orientation number (30, 120, 210, 300).
    tracer : str
        The tracer type to load, 'bhb' or 'rrl'.
    apply_cuts : bool, optional
        If True, apply standard selection cuts. Default is True.
        These cuts remove:
        - Stars that are not observed in Loa (sampling p_obs)
        - Stars that are bound to satellites (pop_id < 2)
        - Stars that are in massive streams (stream_flag_1e8 == False)

    Returns
    -------
    obs : astropy.table.Table
        Table containing error-convolved properties of tracers.
    true : astropy.table.Table
        Table containing true properties of tracers.
    """
    cat_path = basepath + f'field-stars/mocks/auridesi/latest/halo{halo}/'
    cat_file = cat_path + f'halo{halo}-view{view}-{tracer}.fits'

    obs = Table.read(cat_file, hdu='obs')
    true = Table.read(cat_file, hdu='true')

    if apply_cuts:
        sel = obs['in_loa']                 # observed in loa (p_obs sampled)
        sel &= true['pop_id'] < 2           # not bound to satellite
        sel &= ~true['stream_flag_1e8']     # not in massive stream

        obs = obs[sel]
        true = true[sel]

    return obs, true


def get_p_obs(ra, dec, tracer, mag=None, desi_release='loa', gaia_release='dr3'):
    # this is funky but jura == loa for these purposes
    release = 'jura' if desi_release == 'loa' else desi_release

    if tracer == 'bhb':
        bhb_completeness_table_name = f'/global/cfs/cdirs/desi/users/koposov/BHB_all/bhb_completeness/notebooks/bhb_completeness_table_{release}_main_bright.fits'
        bhbfunc = bhb_completeness.get_completness_function(bhb_completeness_table_name)
        p_obs = bhbfunc(ra, dec)
    elif tracer == 'rrl':
        kwargs = {'primsec': 'prim',
                  'surv': 'main',
                  'prog': 'bright',
                  'subset': release,
                  'Gaia': True,
                  'GaiaDR': gaia_release.upper(),
                  'typ': 'ab',
                  'wdist': False,
                  'dist': -99}

        p_obs = compfunc_rrl.ObsComp(ra_in=ra, dec_in=dec, mag=mag, **kwargs)

    return p_obs
