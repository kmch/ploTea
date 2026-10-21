
def test_a_continuous_scheme_builds_its_norm_from_the_data():
    """
    A diverging scheme centres on its neutral point, but only where the data reaches it.

    ``TwoSlopeNorm(vcenter=0)`` raises on values that never cross zero, which a window of one
    island does all the time, so the scheme falls back to a plain ramp instead of failing.

    Examples
    --------
    >>> pytest tests/test_cmaps.py -k builds_its_norm

    """
    import numpy as np
    from matplotlib.colors import SymLogNorm, TwoSlopeNorm

    from plotea.maps.cmaps import population, soil_ph, temperature

    assert isinstance(soil_ph().norm_for(np.ma.array([42.0, 70.0, 77.0])), TwoSlopeNorm)
    assert soil_ph().norm_for(np.ma.array([42.0, 55.0, 60.0])) is None
    assert temperature().norm_for(np.ma.array([6.0, 8.0])) is None
    assert isinstance(temperature().norm_for(np.ma.array([-5.0, 20.0])), TwoSlopeNorm)
    # Counts include zeros, and a plain LogNorm would mask every one of them.
    assert isinstance(population().norm_for(np.ma.array([0.0, 14071.0])), SymLogNorm)
    assert population().norm_for(np.ma.masked_all(3)) is None

def test_every_scheme_names_the_quantity_it_is_for():
    """
    A scheme carries its label and units, so a colorbar needs nothing from the caller.

    Examples
    --------
    >>> pytest tests/test_cmaps.py -k names_the_quantity

    """
    from plotea.maps import cmaps as schemes

    continuous = ['bdod', 'clay', 'sand', 'soil_ph', 'soil_carbon', 'sediment_thickness',
                  'elevation', 'elevation_cm', 'slope', 'ruggedness', 'wetness_index',
                  'temperature', 'temperature_range', 'precipitation', 'precipitation_annual',
                  'precipitation_driest_quarter', 'precipitation_wettest_quarter',
                  'seasonality', 'population', 'livestock']
    for name in continuous:
        scheme = getattr(schemes, name)()
        assert scheme.label, f'{name} has no label'
        assert scheme.cmap is not None
    assert schemes.binary_mask().labels == ['Absent', 'Present']
    assert schemes.esa_worldcover().label == 'Land cover'

    # Quantities that share a ramp must not share a caption: a map of the wettest quarter and
    # one of the driest are opposite halves of the year and would otherwise read as one.
    captions = {name: getattr(schemes, name)().label for name in continuous}
    assert len(set(captions.values())) == len(captions) - 1, captions   # bare precipitation()

def test_a_scheme_can_draw_a_raster_in_a_unit_other_than_the_stored_one():
    """
    Elevation stored in centimetres is drawn, and labelled, in metres.

    Nothing on the band declares the scale, so the hundred lives on the scheme. Without it a
    bar reads 181703 for a mountain 1817 m high.

    Examples
    --------
    >>> pytest tests/test_cmaps.py -k other_than_the_stored_one

    """
    from plotea.maps.cmaps import elevation, elevation_cm

    assert elevation_cm().scale == 0.01 and elevation_cm().unit == 'm'
    assert elevation().scale == 1.0 and elevation().unit == 'm'
