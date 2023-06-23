def format_float(f):
    """ Float formatting function. Likely unused in this version. """
    return float('{:.3f}'.format(f) if abs(f) >= 0.0005 else '{:.3e}'.format(f))
