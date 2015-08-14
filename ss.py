import pandas as pd
import numpy as np


def ss_calc(
    contrib_yearly, inv_gwth_rt, num_years, safe_withdrw_rate, start_age=28
):
    """
    inv_gwth_rt is infaltion adjusted.
    contrib_yearly is in first years dollars
    """
    tot_years = max(0, 62 - start_age - num_years) + num_years
    df = pd.DataFrame({
        'contrib_yearly': [contrib_yearly] * num_years + [0.] *
        max(0, (62 - num_years - start_age)),
        'inv_value': [0] * tot_years,
    }, index=range(tot_years))
    for year in range(0, tot_years):
        print year
        multiplier = np.array([
            (1. + inv_gwth_rt) ** (year - y_) for y_ in range(year + 1)])
        print multiplier
        df['inv_value'][year] = np.sum(
            np.array(df['contrib_yearly'][0: year + 1]) * multiplier)
    df['monthly_inv_income'] = safe_withdrw_rate * df['inv_value'] / 12.
    df['monthly_inv_income_w_spouse'] = df['monthly_inv_income'] * 1.5
    return df


if __name__ == "__main__":
    df = ss_calc(15e3, .03, 10, .03)

    ss_benefit_monthly = 939.00
    ss_benefit_w_spouse_monthly = 1.5 * ss_benefit_monthly
