# Author: Md Fahim Hasan
# PhD Candidate
# Civil and Environmental Engineering
# Colorado State University

# This script simulates FAO56 approach to estimate ETa using dual crop coefficient and simulates soil water balance
# to estimate associated hydrologic fluxes

import pandas as pd


def correct_Kc_for_arid_clim(Kc_table, u2, RHmin, h,  season, target_climate):
    """
    Corrects Kc_mid (and Kc_end if >0.45) values (from Table 8.2 Hoffman 2007) for arid/semi-arid climate
    and greater wind speed. Implements eq. 8.59 in Hoffman 2007.

    Also, applies for Kcb_mid and Kcb_end.

    :param Kc_table: Kc_mid or Kc_end (or Kcb_mid or Kcb_end) from Table 8.2 Hoffman 2007.
    :param u2: Wind speed (unit in m/s).
    :param RHmin: Minimum relative humidity (unit in %).
    :param h: Height of crop (unit in m)
    :param season: The season for which Kcb/Kc is being considered.
                   Have to be within 'initial', 'mid-season', or 'end-season'.
    :param target_climate: Can be anything from ['arid', 'semi-arid', 'humid', 'sub-humid'].
                           But correction will only be applied for 'arid' or 'semi-arid' climate.

    :return: Corrected Kc or Kcb value for 'arid' or 'semi-arid' climate. For other climate types,
             returns the original Kc_table value.
    """
    global Kc_corrected

    if target_climate in ['arid', 'semi-arid']:
        if season == 'initial':
            Kc_corrected = Kc_table

        elif season == 'mid-season':
            Kc_corrected = Kc_table + (0.04 * (u2 - 2) - 0.004 * (RHmin - 45)) * (h / 3) ** 0.3

        elif season == 'end-season' and (Kc_table > 0.45):
            Kc_corrected = Kc_table + (0.04 * (u2 - 2) - 0.004 * (RHmin - 45)) * (h / 3) ** 0.3

        elif season == 'end-season' and (Kc_table <= 0.45):
            Kc_corrected = Kc_table

    else:
        Kc_corrected = Kc_table

    return Kc_corrected


def decide_crop_growth_stage(input_csv_df, emergence_day, full_canopy_day, Lin, Lmid):
    """
    Decides on crop growth stage based on crop emergence day, full canopy day, Length of initial stage, and length of
    mid-season. Implements the timelines of Figure 8.2 of Hoffman 2007.

    :param input_csv_df: Filepath of input csv data. Can also be a dataframe object.
                         Must have a 'Date' column and DOY column.
                         Date column should be in format '%d-%mmm-%Y' (e.g., 15-Jul-2005).
    :param emergence_day: Date of crop emergence in '%d-%mmm-%Y' format.
    :param full_canopy_day: Date of full canopy day in '%d-%mmm-%Y' format.
    :param Lin: Length of initial growth stage. Comes from Table 8.3 Hoffman 2007.
    :param Lmid: Length of mid-season stage (when crops have reached full canopy height).
                 Comes from Table 8.3 Hoffman 2007.

    :return: Returns the dataframe with a column named 'growth_stage' ('ini', 'dev', 'mid', 'late').
    """
    # loading data and converting date to datetime
    if '.csv' in input_csv_df:
        df = pd.read_csv(input_csv_df)

    else:
        df = input_csv_df

    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')

    # converting emergence_day and full_canopy_day to datetime, and getting DOY
    emergence_day = pd.to_datetime(emergence_day, format='%d-%b-%Y')
    full_canopy_day = pd.to_datetime(full_canopy_day, format='%d-%b-%Y')

    Lin_start_DOY = emergence_day.dayofyear
    Lmid_start_DOY = full_canopy_day.dayofyear

    # calculating starting and ending DOY of Lin, Ldev, Lmid, Llate
    Lin_end_DOY = Lin_start_DOY + (Lin - 1)
    Ldev_start_DOY = Lin_end_DOY + 1
    Ldev_end_DOY = Lmid_start_DOY - 1
    Lmid_end_DOY = Lmid_start_DOY + (Lmid - 1)
    Llate_start_DOY = Lmid_end_DOY + 1
    Llate_end_DOY = df['DOY'].iloc[-1:].values[0]

    def crop_growth(DOY, Lin_start_DOY, Lin_end_DOY, Ldev_start_DOY, Ldev_end_DOY,
                    Lmid_start_DOY, Lmid_end_DOY, Llate_start_DOY, Llate_end_DOY):
        """
        Decided crop growth stage based on start and end DOY of different crop growth stage.
        """
        global stage

        if Lin_start_DOY <= DOY <= Lin_end_DOY:
            stage = 'ini'
        elif Ldev_start_DOY <= DOY <= Ldev_end_DOY:
            stage = 'dev'
        elif Lmid_start_DOY <= DOY <= Lmid_end_DOY:
            stage = 'mid'
        elif Llate_start_DOY <= DOY <= Llate_end_DOY:
            stage = 'late'

        return stage

    df['growth_stage'] = df['DOY'].apply(
        lambda doy: crop_growth(doy, Lin_start_DOY, Lin_end_DOY,
                                Ldev_start_DOY, Ldev_end_DOY,
                                Lmid_start_DOY, Lmid_end_DOY,
                                Llate_start_DOY, Llate_end_DOY))

    return df, Ldev_start_DOY, Ldev_end_DOY, Llate_start_DOY, Llate_end_DOY


def calc_Rz(input_df, Rz_ini, Rz_max, doy_col='DOY'):
    """
    Calculate root zone depth for the whole season. Unit depends on units of Rz_ini and Rz_max.

    :param input_df: Input dataframe. Must have 'DOY' column.
    :param Rz_ini: Initial root zone depth.
                   Thumb rule -  depth of seed planting (in) + 2 in
                                 or
                                 depth of seed planting (cm) + 5 cm
                                 or
                                 4-6 times of seed diameter
    :param Rz_max: Maximum root zone depth.
    :param doy_col: Attribute name of DOY column.  Default set to 'DOY'.

    :return: A modified dataframe with root zone (Rz) depth column.
    """
    planting_doy = input_df[input_df['growth_stage'] == 'ini'][doy_col].iloc[0]
    full_cover_doy = input_df[input_df['growth_stage'] == 'mid'][doy_col].iloc[0]

    # function to estimate Rz for each day
    # close to eq. 8.77 of Hoffman 2007
    def Rzi(DOY, Rz_ini, Rz_max, planting_doy, full_cover_doy):
        if DOY <= full_cover_doy:
            Rz_i = Rz_ini + ((DOY - planting_doy) / (full_cover_doy - planting_doy)) * (Rz_max - Rz_ini)  # unit in mm
        else:
            Rz_i = Rz_max

        return Rz_i

    # applying the function to estimate Rzi
    modified_df = input_df.copy()
    modified_df['Rz'] = modified_df.apply(lambda row: Rzi(row[doy_col], Rz_ini, Rz_max,
                                                          planting_doy, full_cover_doy), axis=1)

    return modified_df


def calc_TAW_RAW_dMAD(input_df, MAD_fraction, theta_fc_perc, theta_wp_perc, root_zone_column='Rz'):
    """
    Estimate TAW, RAW, and dMAD.

    :param input_df: Input dataframe. Must have root zone depth column.
    :param MAD_fraction: MAD fraction. Comes from standard tables.
    :param theta_fc_perc: Volumetric soil moisture at field capacity (unit %).
    :param theta_wp_perc: Volumetric soil moisture at wilting point (unit %).
    :param root_zone_column: Attribute name of root zone depth column.  Default set to 'Rz'.

    :return: A modified dataframe with TAW, RAW, and dMAD columns.
    """
    # applying functions to estimate TAW, RAW, and dMAD
    modified_df = input_df.copy()

    modified_df['TAW'] = ((theta_fc_perc - theta_wp_perc) * 1000 / 100) * modified_df[root_zone_column]  # unit mm, considering root zone
    modified_df['MAD'] = MAD_fraction                                  # unit fraction
    modified_df['RAW'] = modified_df['MAD'] * modified_df['TAW']       # unit mm, considers root zone from TAW

    modified_df['dMAD'] = modified_df['RAW']  # the formula is RAW * root zone. Already considered root zone in RAW. SO, dMAD = RAW here

    return modified_df


def estimate_Kcb_all_season(input_df, Kcb_ini, Kcb_mid, Kcb_end,
                            Ldev_start_DOY, Ldev_end_DOY,
                            Llate_start_DOY, Llate_end_DOY,
                            target_climate,
                            windS_col='U', minRH_col='RH min', canopyH_col='hc'):
    """
    Assigns/interpolate daily Kcb values for the whole season.

    :param input_df: Input dataframe processed from decide_crop_growth_stage() function.
                     Must have 'growth_stage', 'DOY', windS_col, minRH_col, canopyH_col.
    :param Kcb_ini: Kcb value for initial growth stage (from Table 8.2-Hoffman 2007).
    :param Kcb_mid: Kcb value for mid-season stage (from Table 8.2-Hoffman 2007)).
    :param Kcb_end: Kcb value for end-season stage (from Table 8.2-Hoffman 2007)).
    :param Ldev_start_DOY: DOY of development stage start day.
    :param Ldev_end_DOY: DOY of development stage end day.
    :param Llate_start_DOY: DOY of end-season start start day.
    :param Llate_end_DOY: DOY of end-season end day.
    :param target_climate: Can be anything from ['arid', 'semi-arid', 'humid', 'sub-humid'].
                       But correction will only be applied for 'arid' or 'semi-arid' climate.
    :param windS_col: Attribute name of wind speed column. Default set to 'U'.
    :param minRH_col: Attribute name of min RH column.  Default set to 'RH min'.
    :param canopyH_col: Attribute name of canopy height column.  Default set to 'hc'.

    :return: A modified dataframe with daily Kcb values populated for the entire season. Also returns modified
             Kcb_ini, Kcb_mid, Kcb_end.
    """

    def daily_Kcb(growth_stage, Kcb_in, Kcb_mid, Kcb_end,
                  current_DOY, Ldev_start_DOY, Ldev_end_DOY,
                  Llate_start_DOY, Llate_end_DOY):
        """
        Assigns/interpolates daily Kcb values based on growth stage and DOY. This function can be applied over the
        dataframe.
        """
        if growth_stage == 'ini':
            Kcb_daily = Kcb_in

        elif growth_stage == 'dev':
            Kcb_daily = Kcb_in + (Kcb_mid - Kcb_in) * (current_DOY - Ldev_start_DOY) / (Ldev_end_DOY - Ldev_start_DOY)

        elif growth_stage == 'mid':
            Kcb_daily = Kcb_mid

        else:  # for growth_stage == 'late'
            Kcb_daily = Kcb_mid + (Kcb_end - Kcb_mid) * (current_DOY - Llate_start_DOY) / (Llate_end_DOY - Llate_start_DOY)

        return Kcb_daily

    # adjusting the Kcb_ini, Kcb_mid, and Kcb_end values obtained from Table 8.2 Hoffman 2007 (for arid/semi-arid climate)
    # for humid/sub-humid climate, no adjustments will be made
    u2_mid_mean = input_df[input_df['growth_stage'] == 'mid'][windS_col].mean()
    rhmin_mid_mean = input_df[input_df['growth_stage'] == 'mid'][minRH_col].mean()
    hc_mid_mean = input_df[input_df['growth_stage'] == 'mid'][canopyH_col].mean()

    u2_end_mean = input_df[input_df['growth_stage'] == 'late'][windS_col].mean()
    rhmin_end_mean = input_df[input_df['growth_stage'] == 'late'][minRH_col].mean()
    hc_end_mean = input_df[input_df['growth_stage'] == 'late'][canopyH_col].mean()

    Kcb_ini_mod = correct_Kc_for_arid_clim(Kc_table=Kcb_ini, u2=None, RHmin=None, h=None, season='initial',
                                           target_climate=target_climate)
    Kcb_mid_mod = correct_Kc_for_arid_clim(Kc_table=Kcb_mid, u2=u2_mid_mean, RHmin=rhmin_mid_mean, h=hc_mid_mean,
                                           season='mid-season', target_climate=target_climate)
    Kcb_end_mod = correct_Kc_for_arid_clim(Kc_table=Kcb_end, u2=u2_end_mean, RHmin=rhmin_end_mean, h=hc_end_mean,
                                           season='end-season', target_climate=target_climate)

    # applying the daily Kcb function on each row of the input dataframe
    modified_df = input_df.copy()

    modified_df['Kcb_daily'] = modified_df.apply(
        lambda row: daily_Kcb(row['growth_stage'], Kcb_ini_mod, Kcb_mid_mod, Kcb_end_mod,
                              row['DOY'], Ldev_start_DOY, Ldev_end_DOY,
                              Llate_start_DOY, Llate_end_DOY), axis=1)

    return modified_df, Kcb_ini_mod, Kcb_mid_mod, Kcb_end_mod


def estimate_Kcmax(input_df, windS_col='U', minRH_col='RH min', canopyH_col='hc'):
    """
    Estimates Kcmax for each row (DOY). Implements eq. 8.61a of Hoffman 2007.

    :param input_df: Input dataframe. Must have columns 'growth_stage', windS_col, minRH_col, canopyH_col.
    :param windS_col: Attribute name of wind speed column. Default set to 'U'.
    :param minRH_col: Attribute name of min RH column.  Default set to 'RH min'.
    :param canopyH_col: Attribute name of canopy height column.  Default set to 'hc'.

    :return: A modified dataframe with Kcmax column.
    """
    # creating a dictionary of growth stage and respectivehc_mean
    hc_ini_mean = input_df[input_df['growth_stage'] == 'ini'][canopyH_col].mean()
    hc_dev_mean = input_df[input_df['growth_stage'] == 'dev'][canopyH_col].mean()
    hc_mid_mean = input_df[input_df['growth_stage'] == 'mid'][canopyH_col].mean()
    hc_end_mean = input_df[input_df['growth_stage'] == 'late'][canopyH_col].mean()

    hcmean_dict = {'ini': hc_ini_mean, 'dev': hc_dev_mean, 'mid': hc_mid_mean, 'late': hc_end_mean}

    # function to calculate Kcmax for each row (each DOY)
    def daily_Kcmax(Kcb_daily, u2, rhmin, hc_mean):
        Kcmax_1 = 1.2 + (0.04 * (u2 - 2) - 0.004 * (rhmin - 45)) * (hc_mean / 3) ** 0.3
        Kcmax_2 = Kcb_daily + 0.05

        Kcmax = max(Kcmax_1, Kcmax_2)

        return Kcmax

    # calculating Kcmax for each DOY
    modified_df = input_df.copy()
    modified_df['Kcmax'] = input_df.apply(
        lambda row: daily_Kcmax(row['Kcb_daily'],  row[windS_col], row[minRH_col],
                                hcmean_dict[row['growth_stage']]), axis=1)

    return modified_df


def assign_TEW_REW(input_df, TEW, REW):
    """
    Assigns TEW and REW across all rows of the dataframe.

    :param input_df:
    :param TEW: Total evaporable water. Comes from Table 8.5 Hoffman 2007 based on soil type info.
    :param REW: Readily evaporable water. Comes from Table 8.5 Hoffman 2007 based on soil type info.

    :return: A modified dataframe with REW and TEW values fora each row.
    """
    input_df['TEW'] = TEW
    input_df['REW'] = REW

    return input_df


def estimate_few(input_df, fw, Kcmin, canopyH_col='hc'):
    """
    Estimates fraction of soil exposed and wetted. Implements eqs. 8.65 and 8.64 of Hoffman 2007.

    :param input_df: Input dataframe. Must have columns 'Kcb_daily', 'Kcmax', and canopyH_col.
    :param fw: fraction wetted value from Table 8.6 Hoffman 2007.
    :param Kcmin: Generally equal to Kcb_ini (source - page 248 of Hoffman 2007).
    :param canopyH_col: Attribute name of canopy height column.  Default set to 'hc'.

    :return: A modified dataframe with fc and few columns.
    """

    # creating function to estimate fraction of canopy cover (fc)
    # eq. 8.65 in Hoffman 2007
    def calc_fc(Kcb_daily, Kcmax_daily, Kcmin, hc):
        # Kcb - Kc_min limited to ≥ 0.01; not explicitly applying it as fc set between 0-0.99 later
        fc = ((Kcb_daily - Kcmin) / (Kcmax_daily - Kcmin)) ** (1 + 0.5 * hc)

        # fc have to be limited within 0 to 0.99
        if fc < 0:
            fc = 0
        elif fc > 0.99:
            fc = 0.99
        else:
            fc = fc

        return fc

    # creating a function to estimate fraction exposed and wetted (few)
    # eq. 8.64 of Hoffman 2007
    def calc_few(fc, fw):
        # 1 – fc and fw, for numerical stability, have limits of 0.01 to 1 (source page 248 Hoffman 2007)
        # this limit has been imposed by calc_fc() function + Table 8.6 of Hoffman 2007
        few = min(1 - fc, fw)

        return few

    # estimating fc
    modified_df = input_df.copy()
    modified_df['fc'] = modified_df.apply(lambda row: calc_fc(row['Kcb_daily'], row['Kcmax'],
                                                              Kcmin, row[canopyH_col]), axis=1)
    # estimating few
    modified_df['few'] = modified_df.apply(lambda row: calc_few(row['fc'], fw), axis=1)

    return modified_df


def decision_irrigate(dMAD, D_i_start):
    """
    Decision of irrigation based on dMAD and D_i_start.

    :param dMAD: dMAD value for the day.
    :param D_i_start: D_i_start value for the day.

    :return: Irrigation decision.
    """
    # Irrigation decision when D_i_start approaches or exceeds dMAD
    if D_i_start >= dMAD:
        irrigate = 'irrigate'
    else:
        irrigate = 'not yet'

    return irrigate


def calc_kr(TEW, REW, D_i_start):
    """
    Estimates evaporation reduction coefficient (Kr). This function goes into soil moisture the water balance equation.
    Implements eqs. 8.63a and 8.63b of Hoffman 2007.

    :param TEW: Total evaporable water. Comes from Table 8.5 Hoffman 2007 based on soil type info.
    :param REW: Readily evaporable water. Comes from Table 8.5 Hoffman 2007 based on soil type info.
    :param D_i_start: Soil water deficit at the beginning of the day. Is equal to the soil moisture deficit
                      at the end of the previous day.

    :return: Kr value for the day.
    """
    if D_i_start <= REW:
        Kr = 1

    elif D_i_start >= TEW:
        Kr = 0

    else:  # D_i_start > REW
        Kr = (TEW - D_i_start) / (TEW - REW)

    if Kr > 1:
        print('Kr is >1. Check!!')

    return Kr


def calc_Ke(Kr, Kcmax_daily, Kcb_daily, few):
    """
    Estimates evaporation coefficient (Ke).
    Implements eq. 8.60 of Hoffman 2007.

    # Ke between 0-1.4 for Kco (based on ETo)
    # Ke between 0-1 for Kcr (based on ETr)
    # these ranges are not set explicitly in this function.
    Further reading and understanding needed before incorporating them.

    :param Kr: Evaporation reduction coefficient (Kr) value for the day.
    :param Kcmax_daily: Kcmax value for the day.
    :param Kcb_daily: Kcb value for the day.
    :param few: few value for the day.

    :return: Ke value for the day.
    """
    few_Kcmax = few * Kcmax_daily

    Ke = Kr * (Kcmax_daily - Kcb_daily)

    if Ke > few_Kcmax:
        Ke = few_Kcmax

    return few_Kcmax, Ke


def calc_Ks(TAW, RAW, D_i_start):
    """
    Estimates soil water stress (Ks).

    :param TAW: TAW value for the entire root zone. Calculated based on given theta_fc and theta_wp.
    :param RAW: RAW value for the entire root zone. Calculated using TAW and MAD.
    :param D_i_start: Soil water deficit at the beginning of the day. Is equal to the soil moisture deficit
                      at the end of the previous day.

    :return: Ks value for the day.
    """
    if D_i_start <= RAW:
        Ks = 1   # crop has enough water available and not under water stress
    else:
        Ks = (TAW - D_i_start) / (TAW - RAW)

    return Ks


def calc_Kco(Kcb_daily, Ks, Ke):
    """
    Estimates daily Kco (KcbKs + Ke) value.

    :param Kcb_daily: Kcb value for the day.
    :param Ks: Ks value for the day.
    :param Ke: Ke value for the day.

    :return: Kco value for the day.
    """
    Kco = Kcb_daily * Ks + Ke

    return Kco


def calc_soil_evaporation(Ke, ETref):
    """
    Calculates evaporation (KeETo).

    :param Ke: Ke for the day.
    :param ETref: ref ET for the day.

    :return: Evaporation. Unit will the unit of ETref.
    """
    KeETref = Ke * ETref

    return  KeETref


def calc_ETa(Kco_daily, ETref):
    """
    Calculates actual ET.

    :param Kco: Kco (KcbKs + Ke) value for the day.
    :param ETref: ref ET for the day.

    :return: ETa value for the day.
    """
    ETa = Kco_daily * ETref

    return ETa


def calc_DP(D_i_start, P, ETa, Irr):
    """
    Calculates deep percolation. Units of all fluxes need to be the same. Preferred unit is 'mm'.

    :param D_i_start: Soil water deficit at the beginning of the day. Is equal to the soil moisture deficit
                      at the end of the previous day.
    :param P: Precipitation of the day.
    :param ETa: Actual evapotranspiration of the day.
    :param Irr: Irrigation of the day.

    :return: Deep percolation value for the day.
    """
    D_interim = D_i_start  + ETa - P - Irr

    # DP calculation
    if D_interim < 0:   # excess water in the system becomes DP
        DP = abs(D_interim)
    else:               # no excess water in the system, DP = 0
        DP = 0

    return DP


def cal_D_i_end(D_i_start, P, ETa, Irr, DP):
    """
    Calculates Soil water deficit of water at the end of the day.
    Units of all fluxes need to be the same. Preferred unit is 'mm'.

    :param D_i_start: Soil water deficit at the beginning of the day. Is equal to the soil moisture deficit
                      at the end of the previous day.
    :param P: Precipitation of the day.
    :param ETa: Actual evapotranspiration of the day.
    :param Irr: Irrigation of the day.
    :param DP: Deep percolation for the day.

    :return: Soil water deficit at the end of the day.
    """
    global D_i_end

    D_interim = D_i_start + ETa - P - Irr

    # D_i_end calculation
    if DP == 0:         # if DP = 0, the water balance of the day becomes deficit at the end of the day
        D_i_end = D_interim
    elif DP > 0:        # means there is excess water in the system which is becoming DP, no deficit at the end of the day
        D_i_end = 0

    return D_i_end


def curate_reorder_P_Irr_data(input_df, precip_col='Precip', Irr_col='Irr'):
    """
    fill Nan values of precip and irrigation column with zero. And bring precip and Irrigation column to the end
    for better visualization.

    :param input_df: Input dataframe. Must have all required columns.
    :param precip_col: Precipitation column in the dataframe. Default set to 'Precip'.
    :param Irr_col: Irrigation column in the dataframe. Default set to 'Irr'.

    :return: A nan-filled 'Precip' and 'Irr' column consisting dataframe.
    """
    # Fill NaN values with zero
    input_df[precip_col].fillna(0, inplace=True)
    input_df[Irr_col].fillna(0, inplace=True)

    # Move the Precip and Irrigation columns to the end
    columns = [col for col in input_df.columns if col not in [precip_col, Irr_col]]
    columns += [precip_col, Irr_col]  # Add 'Precip' and 'Irr' columns to the end

    # Reorder the dataframe
    input_df = input_df[columns]

    return input_df


def run_daily_water_balance(input_df, ETref_col='ETo', precip_col='Precip', Irr_col='Irr',
                            output_csv='SWB_Irr_output.csv'):
    """
    Performs a daily soil water balance calculation, including evapotranspiration, irrigation, precipitation,
    soil water deficit, and deep percolation. The calculations follow the FAO-56 model principles and Hoffman 2007.

    The water balance calculation updates daily soil moisture status, checks irrigation needs,
    calculates actual evapotranspiration (ETa), and estimates deep percolation (DP).

    :param input_df: Input dataframe. Must include the following columns: 'dMAD', 'TEW', 'REW', 'Kcb_daily', 'few',
                     'Kcmax', 'TAW', 'RAW', and the columns specified by ETref_col, precip_col, and Irr_col.
    :param ETref_col: Column name for reference evapotranspiration (ETref). Default set to 'ETo'.
    :param precip_col: Column name for precipitation. Default set to 'Precip'.
    :param Irr_col: Column name for irrigation. Default set to 'Irr'.
    :param output_csv: Output csv filepath.

    :return: The modified dataframe with the results of the daily water balance calculations. The following columns
             are added to the dataframe:

             - 'Di_start': Soil water deficit at the start of the day.
             - 'Kr': Evaporation reduction coefficient for the day.
             - 'Ke': Evaporation coefficient for the day.
             - 'fewKcmax': The minimum between fraction wetted/exposed and Kcmax.
             - 'Ks': Soil water stress coefficient for the day.
             - 'Kco': Crop coefficient, considering evaporation and water stress (Kcb * Ks + Ke).
             - 'KeETo': Evaporation for the day, calculated as Ke * ETref.
             - 'ETa': Actual evapotranspiration for the day.
             - 'DP': Deep percolation (excess water that percolates beyond the root zone).
             - 'Di_end': Soil water deficit at the end of the day.
             - 'irrigate': Decision on whether irrigation is needed for the day (based on dMAD).

    The final dataframe is saved as 'SWB_Irr.csv'.
    """
    print('Running daily soil water balance...')

    swb_df = input_df.copy()

    D_i_start = 0   # no deficit considered at day one (soil at field capacity)

    # initializing empty lists to assign values calculated during water balance
    D_i_start_list = []
    Kr_list = []
    Ke_list = []
    few_Kcmax_list = []
    Ks_list = []
    Kco_list = []
    KeETo_list = []
    ETa_list = []
    DP_list = []
    D_i_end_list = []
    irrigate_decision_list = []

    #  soil moisture water balance for each row
    for idx, row in swb_df.iterrows():
        irrigate = decision_irrigate(row['dMAD'], D_i_start)
        Kr = calc_kr(row['TEW'], row['REW'], D_i_start)
        few_Kcmax, Ke = calc_Ke(Kr, row['Kcmax'], row['Kcb_daily'], row['few'])
        Ks = calc_Ks(row['TAW'], row['RAW'], D_i_start)
        Kco = calc_Kco(row['Kcb_daily'], Ks, Ke)
        KeETo = calc_soil_evaporation(Ke, row[ETref_col])
        ETa = calc_ETa(Kco, row[ETref_col])
        DP = calc_DP(D_i_start, row[precip_col], ETa, row[Irr_col])
        D_i_end = cal_D_i_end(D_i_start, row[precip_col], ETa, row[Irr_col], DP)

        # appending the estimated values on the empty lists
        D_i_start_list.append(D_i_start)
        Kr_list.append(Kr)
        Ke_list.append(Ke)
        few_Kcmax_list.append(few_Kcmax)
        Ks_list.append(Ks)
        Kco_list.append(Kco)
        KeETo_list.append(KeETo)
        ETa_list.append(ETa)
        DP_list.append(DP)
        D_i_end_list.append(D_i_end)
        irrigate_decision_list.append(irrigate)

        # D_i_end becomes D_i_start of next day
        D_i_start = D_i_end

    # adding the estimated values in the dataframe
    swb_df.loc[:, 'Di_start'] = D_i_start_list
    swb_df.loc[:, 'Kr'] = Kr_list
    swb_df.loc[:, 'Ke'] = Ke_list
    swb_df.loc[:, 'fewKcmax'] = few_Kcmax_list
    swb_df.loc[:, 'Ks'] = Ks_list
    swb_df.loc[:, 'Kco'] = Kco_list
    swb_df.loc[:, 'KeETo'] = KeETo_list
    swb_df.loc[:, 'ETa'] = ETa_list
    swb_df.loc[:, 'DP'] = DP_list
    swb_df.loc[:, 'Di_end'] = D_i_end_list
    swb_df.loc[:, 'irrigate'] = irrigate_decision_list

    # saving results
    swb_df.to_csv(output_csv)

    print('Daily soil water balance run completed...')

    return swb_df

