from .. import config

def get_rot_regions(unified_script, rot_zero_tol=None):
    if rot_zero_tol is None:
        rot_zero_tol = config.DEFAULT_ROT_ZERO_TOL
    if not unified_script:
        return []
    regions = []
    in_rot = False
    reg_start = -1.0
    last_rot_end = -1.0


    for ev in unified_script:
        t = ev["time"]
        end_t = ev.get("end_time", t)
        rot = ev["mag_rot"]
        is_zero = abs(rot) <= rot_zero_tol
        if not in_rot and not is_zero:
            in_rot = True
            reg_start = t
            last_rot_end = end_t

        elif in_rot:
            if not is_zero:
                last_rot_end = end_t  # don't use event start time
            else:
                in_rot = False
                regions.append([reg_start, end_t])
    if in_rot:
        regions.append([reg_start, last_rot_end])
    return regions

def get_rel_regions(unified_script, desired_speed=None, allowance=None):
    if desired_speed is None:
        desired_speed = config.DEFAULT_REL_REGIONS_SPEED
    if allowance is None:
        allowance = config.DEFAULT_REL_REGIONS_ALLOWANCE
    if not unified_script:
        return []
    regions = []
    reg_start, reg_end, gap = -1.0, -1.0, 0
    for ev in unified_script:
        t, s = ev["time"], ev.get("mag_speed")
        if s == desired_speed:
            gap = 0
            if reg_start < 0:
                reg_start = t
        else:
            if reg_start >= 0:
                if gap < allowance:
                    reg_end = t
                    gap += 1
                else:
                    regions.append([reg_start, reg_end])
                    gap = 0
                    reg_start = reg_end = -1
    if reg_start >= 0:
        reg_end = unified_script[-1]["time"] if reg_end < 0 else reg_end
        regions.append([reg_start, reg_end])
    return regions
