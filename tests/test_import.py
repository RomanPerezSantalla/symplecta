def test_package_imports():
    import symplecta

    assert symplecta.__version__
    assert callable(symplecta.solve_symplectic_ivp)
