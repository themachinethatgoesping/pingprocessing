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

def select_get_wci_image(ping, selection, wci_value):
    if callable(wci_value):
        return wci_value(ping, selection)

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

    match wci_value:
        case "amp":
            wci = ping.watercolumn.get_amplitudes(selection)
        case "av":
            wci = ping.watercolumn.get_av(selection)
        case "ap":
            wci = ping.watercolumn.get_ap(selection)
        case "power":
            wci = ping.watercolumn.get_power(selection)
        case "sp":
            wci = ping.watercolumn.get_sp(selection)
        case "sv":
            wci = ping.watercolumn.get_sv(selection)
        case "pv":
            wci = ping.watercolumn.get_pv(selection)
        case "rv":
            wci = ping.watercolumn.get_rv(selection)
        case "rp":
            wci = ping.watercolumn.get_rp(selection)
        case "pp":
            wci = ping.watercolumn.get_pp(selection)
        case _:
            raise ValueError(
                f"Invalid value for wci_value: {wci_value}. Choose any of ['amp','power', 'rp', 'rv',  'pp', 'pv',  'ap', 'av',  'sp', 'sv', 'power/amp', 'sp/ap/pp/rp', 'sv/av/pv/rv']."
            )
    
    return wci
