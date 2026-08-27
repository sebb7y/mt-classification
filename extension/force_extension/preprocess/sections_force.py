from .. import config

def get_force_ext_regions(unified_script, desired_speed=None, min_distance=None, use_move_duration=None, extension_only=None, extension_when=None):
    if desired_speed is None:
        desired_speed = config.DEFAULT_FORCE_EXTENSION_SPEED
    if min_distance is None:
        min_distance = config.DEFAULT_FORCE_EXTENSION_MIN_DISTANCE
    if use_move_duration is None:
        use_move_duration = config.DEFAULT_FORCE_EXTENSION_USE_MOVE_DURATION
    if extension_only is None:
        extension_only = config.DEFAULT_FORCE_EXTENSION_EXTENSION_ONLY
    if extension_when is None:
        extension_when = getattr(
            config, "DEFAULT_FORCE_EXTENSION_EXTENSION_WHEN", "mag_pos_increasing"
        )
    extension_increasing = extension_when != "mag_pos_decreasing"

    if not unified_script:
        return []
    regions = []
    prev_pos = 0.0
    n = len(unified_script)
    for i, ev in enumerate(unified_script):
        pos = ev.get("mag_pos", prev_pos)
        speed_ok = ev.get("mag_speed") == desired_speed if not extension_only else True
        if not speed_ok:
            prev_pos = pos
            continue
        if extension_only:
            if extension_increasing:
                if pos <= prev_pos:
                    prev_pos = pos
                    continue
                if prev_pos == 0.0:
                    # usually not a valid extension
                    prev_pos = pos
                    continue
            else:
                if pos >= prev_pos:
                    prev_pos = pos
                    continue
        t0 = ev["time"]
        dist = abs(pos - prev_pos)
        prev_pos = pos
        if min_distance > 0 and dist < min_distance:
            continue
        if use_move_duration:
            t1 = ev.get("end_time", t0)
        else:
            t1 = unified_script[i + 1]["time"] if i + 1 < n else ev.get("end_time", t0)
        regions.append([t0, t1])
    return regions
