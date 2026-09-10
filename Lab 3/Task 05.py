def task5_rigid_align(img, angle_deg=30.0, tx=40, ty=-25):
    """
    A Rigid Transformation = Rotation + Translation only (no scale/shear),
    so shape is perfectly preserved. Combine both into a single 3x3 matrix
    and apply them simultaneously in one warp.
    """
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    R = np.array([
        [cos_t, -sin_t, 0.0],
        [sin_t,  cos_t, 0.0],
        [0.0,    0.0,   1.0]
    ], dtype=np.float64)

    T = np.array([
        [1.0, 0.0, tx],
        [0.0, 1.0, ty],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # Combine: apply rotation first, then translation -> T @ R
    Rigid = T @ R

    h, w = img.shape[:2]
    M = Rigid[:2, :]
    result = cv2.warpAffine(img, M, (w, h))
    return result, Rigid
