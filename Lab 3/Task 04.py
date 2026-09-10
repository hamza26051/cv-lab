def task4_translate_map(img, shift_left=150, shift_up=80):
    """
    Upgrade to a 3x3 homogeneous matrix and build a pure translation matrix
    that slides the map so the hidden 'X' (currently 150px left and 80px
    up of the visible frame) comes back on screen. Bringing something back
    from "off-screen to the left/up" means sliding the whole map
    RIGHT and DOWN by that amount.
    """
    h, w = img.shape[:2]
    tx, ty = shift_left, shift_up  # move right by 150, down by 80

    # 1. Manually build the 3x3 translation matrix
    T = np.array([
        [1.0, 0.0, tx],
        [0.0, 1.0, ty],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # 2. Apply it (warpAffine wants the top 2x3 rows)
    M = T[:2, :]
    new_w, new_h = w + tx, h + ty
    result = cv2.warpAffine(img, M, (new_w, new_h))
    return result, T
