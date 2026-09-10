def task2_derotate_satellite(img, angle_deg=45.0):
    """
    Formulate a 2x2 rotation matrix from sine/cosine, spin the image back
    to true north (i.e. undo the 45-degree tilt), and manually compute the
    new canvas size so corners aren't clipped.
    """
    h, w = img.shape[:2]

    # We want to UNDO a +45 degree tilt, so we rotate by -angle_deg.
    theta = np.deg2rad(-angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # 1. Manually build the 2x2 rotation matrix
    R = np.array([
        [cos_t, -sin_t],
        [sin_t,  cos_t]
    ], dtype=np.float64)

    # 2. Manually compute the new bounding-box dimensions so the rotated
    #    corners aren't chopped off:
    #    new_w = |w*cos| + |h*sin| ,  new_h = |w*sin| + |h*cos|
    new_w = int(np.ceil(abs(w * cos_t) + abs(h * sin_t)))
    new_h = int(np.ceil(abs(w * sin_t) + abs(h * cos_t)))

    # 3. Rotation about the origin also needs a translation to re-center
    #    the (now larger) canvas, exactly like Task 1.
    cx, cy = w / 2.0, h / 2.0                 # old center
    new_cx, new_cy = new_w / 2.0, new_h / 2.0  # new center

    # tx, ty chosen so that the old center maps to the new center:
    # new_center = R * old_center + t  =>  t = new_center - R * old_center
    old_center = np.array([cx, cy])
    t = np.array([new_cx, new_cy]) - R @ old_center

    M = np.array([
        [R[0, 0], R[0, 1], t[0]],
        [R[1, 0], R[1, 1], t[1]]
    ], dtype=np.float64)

    result = cv2.warpAffin
