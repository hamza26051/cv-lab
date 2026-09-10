def task6_similarity_fix(img, scale=1.8, angle_deg=-20.0, tx=60, ty=30):
    """
    A Similarity Transformation = uniform Scale + Rotation + Translation
    (preserves angles, changes size). Combine all three into one 3x3
    matrix and fix the blueprint in a single operation.
    """
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    S = np.array([
        [scale, 0.0,   0.0],
        [0.0,   scale, 0.0],
        [0.0,   0.0,   1.0]
    ], dtype=np.float64)

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

    # Combine: scale, then rotate, then translate -> T @ R @ S
    Similarity = T @ R @ S

    h, w = img.shape[:2]
    new_w, new_h = int(w * scale) + abs(tx) + 50, int(h * scale) + abs(ty) + 50
    M = Similarity[:2, :]
    result = cv2.warpAffine(img, M, (new_w, new_h))
    return result, Similarity
