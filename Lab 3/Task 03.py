def task3_deshear_barcode(img, shear_factor=-0.5):
    """
    Construct a 2x2 shear matrix that pushes pixels horizontally to undo a
    rightward slant. A positive `shear_factor` slants right, so to CORRECT
    a rightward slant we apply the negative of that factor.
    """
    h, w = img.shape[:2]

    # 1. Manually build the 2x2 horizontal shear matrix:
    #    x' = x + shear_factor * y
    #    y' = y
    Sh = np.array([
        [1.0, shear_factor],
        [0.0, 1.0]
    ], dtype=np.float64)

    # 2. Shearing horizontally shifts columns outward; widen the canvas so
    #    nothing is clipped, and compute the needed x-offset.
    extra_w = int(np.ceil(abs(shear_factor) * h))
    new_w = w + extra_w
    dx = extra_w if shear_factor < 0 else 0

    M = np.array([
        [Sh[0, 0], Sh[0, 1], dx],
        [Sh[1, 0], Sh[1, 1], 0.0]
    ], dtype=np.float64)

    result = cv2.warpAffine(img, M, (new_w, h))
    return result, Sh, M
