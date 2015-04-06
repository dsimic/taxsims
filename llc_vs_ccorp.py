"""
Script for computing rough after tax profit for LLC vs C-Corp
in absence of state corporate or personal income taxes.

WARNING - These calculations come with no guarantee, and
are likely to contain errors, se at your own risk!

Copyrigth 2015 - David Simic (dpsimic at gmail dot com)
"""
import numpy as np
import pandas as pd

# employer fica
employer_ss = .0620
employer_mc = .0145
# employee fice
employee_ss = 0.0620
employee_mc = 0.0145
# cap
ss_cap = 117000.00
# self employed fica multiplier
se_mul = 0.9235
# target gross income
target_gross_income = 15000.00
target_gross_income = 20000.00
target_gross_income = 50000.00
target_gross_income = 100000.00
target_gross_income = 15000.00
# IRA contibution
personal_ira_contribution_limit = 5500.00

# personal income tax brackets for individual
personal_income_tax_brackets = (
    (9225.00, .100),
    (37450.00, .150),
    (90750.00, .250),
    (189300.00, .280),
    (411501.00, .330),
    (413201.00, .350),
    (np.inf, .396),
)

# corporate income tax brackets
corp_tax_brackets = (
    (5.000e4, .15),
    (7.500e4, .25),
    (1.000e5, .34),
    (3.350e5, .39),
    (1.000e8, .34),
    (1.500e8, .35),
    (1.833e8, .38),
    (np.inf, .35),
)

corp_tax_brackets_florida = (
    (5.000e4, .000),
    (np.inf, .055),
)

corp_tax_brackets_state = corp_tax_brackets_florida

# standard deduction
personal_standard_deduction = 6300.00


def compute_employer_payroll(salary):
    ss = employer_ss * salary.clip(-np.inf, ss_cap)
    mc = employer_mc * salary
    return ss + mc


def compute_employee_payroll(salary):
    ss = employer_ss * salary.clip(-np.inf, ss_cap)
    mc = employer_mc * salary
    return ss + mc


def compute_llc_payroll(salary):
    ss = (employer_ss + employee_ss) * salary.clip(-np.inf, ss_cap)
    mc = (employer_mc + employee_mc) * salary
    return (ss + mc) * se_mul


def compute_personal_income_tax(salary, deduction):
    taxable_income = salary - deduction
    taxable_income = taxable_income.clip(0, np.inf)
    prev_bracket = 0.0
    tax = 0. * taxable_income
    for bracket, rate in personal_income_tax_brackets:
        tax = tax + compute_bracket_tax(
            taxable_income, bracket, prev_bracket, rate)
        prev_bracket = bracket
    return tax


def compute_bracket_tax(taxable_income, bracket, prev_bracket, rate):
    tmp = np.copy(taxable_income)
    tmp[(tmp > bracket)] = bracket
    tmp = tmp - prev_bracket
    tmp[tmp < 0.0] = 0
    return rate * tmp


def compute_corp_income_tax(taxable_income):
    taxable_income = taxable_income.clip(0, np.inf)
    prev_bracket = 0.0
    tax = 0. * taxable_income
    for bracket, rate in corp_tax_brackets:
        tax = tax + compute_bracket_tax(
            taxable_income, bracket, prev_bracket, rate)
        prev_bracket = bracket
    return tax


def compute_llc(gross_profit):
    # llc member payroll
    llc_payroll = compute_llc_payroll(gross_profit)
    # llc member pre income tax net income
    llc_pretax_income = gross_profit - llc_payroll
    # assume standard deduction
    llc_deduction = personal_standard_deduction
    # llc member post income tax net income
    llc_income_tax = compute_personal_income_tax(
        llc_pretax_income, llc_deduction)
    # llc net income
    total_net_income = llc_pretax_income - llc_income_tax
    # total tax
    total_tax = llc_income_tax + llc_payroll

    assert np.allclose(total_tax + total_net_income, gross_profit)

    return {
        'business_fica': llc_payroll,
        'business_pretax_income': llc_pretax_income,
        'business_income_tax': llc_income_tax,
        'business_net_income': total_net_income,
        'personal_deduction': llc_deduction,
        'personal_fica': llc_payroll,
        'personal_income_tax': llc_income_tax,
        'personal_net_income': total_net_income,
        'total_net_income': total_net_income,
        'total_tax': total_tax,
        'net_tax_rate': total_tax / gross_profit,
    }


def compute_corp(gross_profit):
    # gross personal income
    p_gross_income = gross_profit.clip(0, target_gross_income)
    # corporate payroll contribution
    corp_fica = compute_employer_payroll(p_gross_income)
    # corporate net income
    corp_pretax_income = gross_profit - p_gross_income - corp_fica
    # employee fica
    employee_fica = compute_employee_payroll(p_gross_income)
    # employee net income
    employee_pretax_income = p_gross_income - employee_fica
    # personal deduction
    personal_deduction = personal_standard_deduction
    # employee personal income tax
    employee_income_tax = compute_personal_income_tax(
        employee_pretax_income, personal_deduction)
    # employee net income
    employee_net_income = employee_pretax_income - employee_income_tax
    # corporate income tax
    corp_income_tax = compute_corp_income_tax(np.copy(corp_pretax_income))
    # corporate net income
    corp_net_income = corp_pretax_income - corp_income_tax
    # net corp + employee profit
    total_net_income = corp_net_income + employee_net_income
    # total taxes
    total_tax = corp_fica + employee_fica + corp_income_tax + \
        employee_income_tax

    assert np.allclose(total_tax + total_net_income, gross_profit)

    return {
        'business_fica': corp_fica,
        'business_pretax_income': corp_pretax_income,
        'business_income_tax': corp_income_tax,
        'business_net_income': corp_net_income,
        'personal_deduction': personal_deduction,
        'personal_fica': employee_fica,
        'personal_income_tax': employee_income_tax,
        'personal_net_income': employee_net_income,
        'total_net_income': total_net_income,
        'total_tax': total_tax,
        'net_tax_rate': total_tax / gross_profit,
    }


def to_df(dic, pref):
    df = pd.DataFrame(dic)
    df.columns = map(lambda x: pref + x, df.columns.tolist())
    return df


if __name__ == "__main__":
    # gross business profit
    gross_profit = np.linspace(0, 1.e5, 21)
    gross_profit = np.concatenate(
        [gross_profit, np.linspace(1.e5, 1e6, 10)[1:]])
    corp_numbers = compute_corp(gross_profit)
    llc_numbers = compute_llc(gross_profit)
    llc = to_df(llc_numbers, "llc_")
    corp = to_df(corp_numbers, "corp_")
    df = pd.concat(
        [corp, llc], axis=1)
    df['llc_vs_corp'] = \
        df['llc_total_net_income'] - df['corp_total_net_income']
    df['gross_profit'] = gross_profit
    df = df.set_index('gross_profit')
    llc['gross_profit'] = gross_profit
    corp['gross_profit'] = gross_profit
