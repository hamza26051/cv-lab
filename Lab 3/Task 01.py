def task1_scale_fingerprint(img, sx=3.0, sy=3.0):
    """
    Manually build a 2x2 scaling matrix and enlarge the image by 300% on
    both axes (scale factor = 3), then re-center it so it doesn't just
    grow off the top-left corner.
    """
    h, w = img.shape[:2]

    # 1. Manually build the 2x2 linear scaling matrix
    S = np.array([
        [sx, 0.0],
        [0.0, sy]
    ], dtype=np.float64)

    # 2. A pure 2x2 linear map is anchored at the origin (0,0), so scaling
    #    directly would push the fingerprint off-screen to the bottom-right.
    #    To keep it centered we compute how much bigger the canvas becomes
    #    and shift by half that amount.
    new_w, new_h = int(round(w * sx)), int(round(h * sy))
    dx = (new_w - w) / 2.0
    dy = (new_h - h) / 2.0

    # 3. Embed the 2x2 linear matrix into a 3x3 affine matrix with the
    #    centering translation, then apply it.
    M = np.array([
        [S[0, 0], S[0, 1], dx],
        [S[1, 0], S[1, 1], dy]
    ], dtype=np.float64)

    result = cv2.warpAffine(img, M, (new_w, new_h))
    return result, S, M
