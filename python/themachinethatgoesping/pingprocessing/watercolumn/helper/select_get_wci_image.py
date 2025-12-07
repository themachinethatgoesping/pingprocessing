import numpy as np
import themachinethatgoesping.echosounders as echosounders

def apply_pss(ping, pss, apply_pss_to_bottom):
    if apply_pss_to_bottom and ping.has_bottom() and ping.bottom.has_beam_crosstrack_angles():
        if ping.bottom.get_number_of_beams() != ping.watercolumn.get_number_of_beams():
            aw = ping.watercolumn.get_beam_crosstrack_angles()
            ab = ping.bottom.get_beam_crosstrack_angles()

            ad = np.median(aw) - np.median(ab)
            pss_ = pss.copy()

            min_ba = pss.get_min_beam_angle()
            max_ba = pss.get_max_beam_angle()

            if min_ba is not None:
                min_ba += ad

            if max_ba is not None:
                max_ba += ad

            pss_.select_beam_range_by_angles(min_ba, max_ba)

            sel = pss_.apply_selection(ping.watercolumn)
        else:
            sel = pss.apply_selection(ping.bottom)
            if len(sel.get_beam_numbers()) == 0:
                return echosounders.pingtools.BeamSampleSelection()
            pss_ = pss.copy()
            pss_.clear_beam_angle_range()
            pss_.select_beam_range_by_numbers(sel.get_beam_numbers()[0], sel.get_beam_numbers()[-1], pss.get_beam_step())

            sel = pss_.apply_selection(ping.watercolumn)
    else:
        sel = pss.apply_selection(ping.watercolumn)

    return sel

def select_get_wci_image(ping, selection, wci_value, mp_cores = 1):
    if callable(wci_value):
        return wci_value(ping, selection)
    
    if 'vs' in wci_value:
        return select_get_wci_correction(ping, selection, wci_value, mp_cores=mp_cores)

    wci_values = wci_value.split("/")

    # if there is a "/" select the first value that is available
    # if none is available, still the last specified is selected
    for wci_value in wci_values:
        match wci_value:
            case "amp":
                if ping.watercolumn.has_amplitudes():
                    break
            case "av":
                if ping.watercolumn.has_av():
                    break
            case "ap":
                if ping.watercolumn.has_ap():
                    break
            case "power":
                if ping.watercolumn.has_power():
                    break
            case "sp":
                if ping.watercolumn.has_sp():
                    break
            case "sv":
                if ping.watercolumn.has_sv():
                    break
            case "pv":
                if ping.watercolumn.has_pv():
                    break
            case "rv":
                if ping.watercolumn.has_rv():
                    break
            case "rp":
                if ping.watercolumn.has_rp():
                    break
            case "pp":
                if ping.watercolumn.has_pp():
                    break
            case _:
                raise ValueError(
                    f"Invalid value for wci_value: {wci_value}. Choose any of ['amp','power', 'rp', 'rv',  'pp', 'pv',  'ap', 'av',  'sp', 'sv', 'power/amp', 'sp/ap/pp/rp', 'sv/av/pv/rv']."
                )

    return ping.watercolumn.get_wci(selection, wci_value, mp_cores=mp_cores)

    # match wci_value:
    #     case "amp":
    #         wci = ping.watercolumn.get_amplitudes(selection, mp_cores=mp_cores)
    #     case "av":
    #         wci = ping.watercolumn.get_av(selection, mp_cores=mp_cores)
    #     case "ap":
    #         wci = ping.watercolumn.get_ap(selection, mp_cores=mp_cores)
    #     case "power":
    #         wci = ping.watercolumn.get_power(selection, mp_cores=mp_cores)
    #     case "sp":
    #         wci = ping.watercolumn.get_sp(selection, mp_cores=mp_cores)
    #     case "sv":
    #         wci = ping.watercolumn.get_sv(selection, mp_cores=mp_cores)
    #     case "pv":
    #         wci = ping.watercolumn.get_pv(selection, mp_cores=mp_cores)
    #     case "rv":
    #         wci = ping.watercolumn.get_rv(selection, mp_cores=mp_cores)
    #     case "rp":
    #         wci = ping.watercolumn.get_rp(selection, mp_cores=mp_cores)
    #     case "pp":
    #         wci = ping.watercolumn.get_pp(selection, mp_cores=mp_cores)
    #     case _:
    #         raise ValueError(
    #             f"Invalid value for wci_value: {wci_value}. Choose any of ['amp','power', 'rp', 'rv',  'pp', 'pv',  'ap', 'av',  'sp', 'sv', 'power/amp', 'sp/ap/pp/rp', 'sv/av/pv/rv']."
    #         )
    
    # return wci

def select_get_wci_correction(ping, selection, calibration, mp_cores = 1):
    
    wci_values = calibration.split("_vs_")
    
    if len(wci_values) != 2:
        raise ValueError(
            f"Invalid value for calibration: {wci_value}. Must be of format 'calibration_vs_base', e.g., 'sv_vs_av'."
        )
    
    #return select_get_wci_image(ping, selection, wci_values[0], mp_cores=mp_cores) - select_get_wci_image(ping, selection, wci_values[1], mp_cores=mp_cores)
    return ping.watercolumn.get_wci_correction(selection, wci_values[0], wci_values[1], mp_cores=mp_cores)