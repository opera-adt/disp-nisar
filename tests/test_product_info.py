from disp_nisar.product_info import DISPLACEMENT_PRODUCTS, ProductInfo


def test_displacement_products_iter():
    """Iterating DisplacementProducts yields ProductInfo instances."""
    products = list(DISPLACEMENT_PRODUCTS)
    assert len(products) > 0
    for p in products:
        assert isinstance(p, ProductInfo)
        assert isinstance(p.name, str)
        assert len(p.name) > 0


def test_displacement_products_names():
    """The names property returns all dataset name strings."""
    names = DISPLACEMENT_PRODUCTS.names
    assert isinstance(names, list)
    assert "displacement" in names
    assert "short_wavelength_displacement" in names
    assert "connected_component_labels" in names
    assert "temporal_coherence" in names
    assert "recommended_mask" in names
    # Ensure count matches iteration
    assert len(names) == len(list(DISPLACEMENT_PRODUCTS))
