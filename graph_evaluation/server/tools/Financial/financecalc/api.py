import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def leverage_ratio_lr(total_income: int, total_liabilities: int, total_debts: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Leverage Ratio (LR) is used to calculate the financial leverage of a company or individual to get an idea of the methods of financing or to measure ability to meet financial obligations."
    total_income: total income (eg 20)
        total_liabilities: total liabilities (eg 25)
        total_debts: total debts (eg 10)
        
    """
    url = f"https://financecalc.p.rapidapi.com/leverage-ratio"
    querystring = {'total_income': total_income, 'total_liabilities': total_liabilities, 'total_debts': total_debts, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financecalc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def net_present_value_npv(initial_investment: int, cash_flows: str, rate: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Net Present Value (NPV) compares the money received in the future to an amount of money received today, while accounting for time and interest [through the discount rate]. It's based on the principal of time value of money (TVM), which explains how time affects monetary value."
    initial_investment: initial investment (eg -500000)
        cash_flows: cash flows (eg  200000,300000,200000)
        rate: rate (eg 10)
        
    """
    url = f"https://financecalc.p.rapidapi.com/net-present-value"
    querystring = {'initial_investment': initial_investment, 'cash_flows': cash_flows, 'rate': rate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financecalc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profitability_index_pi(initial_investment: int, cash_flows: str, rate: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Profitability Index (PI) is an index that attempts to identify the relationship between the costs and benefits of a proposed project through the use of a ratio calculated."
    initial_investment: initial investment (eg -40000)
        cash_flows: cash flows (eg 18000,12000,10000,9000,6000)
        rate: rate (eg 10)
        
    """
    url = f"https://financecalc.p.rapidapi.com/profitability-index"
    querystring = {'initial_investment': initial_investment, 'cash_flows': cash_flows, 'rate': rate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financecalc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def return_on_investment_roi(initial_investment: int, earnings: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return on Investment (ROI) is a simple calculation that tells you the bottom line return of any investment."
    initial_investment: initial investment (eg -55000)
        earnings: earnings (eg 60000)
        
    """
    url = f"https://financecalc.p.rapidapi.com/return-on-investment"
    querystring = {'initial_investment': initial_investment, 'earnings': earnings, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financecalc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rule_of_72_r72(rate: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Rule of 72 (R72) is a rule stating that in order to find the number of years required to double your money at a given interest rate, you divide the compound return into 72."
    rate: rate (eg 10)
        
    """
    url = f"https://financecalc.p.rapidapi.com/rule-of-72"
    querystring = {'rate': rate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financecalc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def weighted_average_cost_of_capital_wacc(market_value_of_debt: int, market_value_of_equity: int, tax_rate: int, cost_of_equity: int, cost_of_debt: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Weighted Average Cost of Capital (WACC) is the rate that a company is expected to pay on average to all its security holders to finance its assets."
    market_value_of_debt: market value of debt (eg 400000)
        market_value_of_equity: market value of equity (eg 600000)
        tax_rate: tax rate (eg 35)
        cost_of_equity: cost of equity (eg 6)
        cost_of_debt: cost of debt (eg 5)
        
    """
    url = f"https://financecalc.p.rapidapi.com/weighted-average-cost-of-capital"
    querystring = {'market_value_of_debt': market_value_of_debt, 'market_value_of_equity': market_value_of_equity, 'tax_rate': tax_rate, 'cost_of_equity': cost_of_equity, 'cost_of_debt': cost_of_debt, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financecalc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def inflation_adjusted_return(inflation_rate: int, investment_return: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Measure the return taking into account the time period's inflation rate"
    inflation_rate: inflation rate (eg 0.03)
        investment_return: investment return (eg 0.08)
        
    """
    url = f"https://financecalc.p.rapidapi.com/inflation-adjusted-return"
    querystring = {'inflation_rate': inflation_rate, 'investment_return': investment_return, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financecalc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def future_value_fv(cash_flow: int=None, rate: int=None, number_of_periods: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Future Value (FV) is the value of an asset or cash at a specified date in the future that is equivalent in value to a specified sum today"
    cash_flow: cash flow (eg 1000)
        rate: rate (eg 0.5)
        number_of_periods: number of periods (eg 12)
        
    """
    url = f"https://financecalc.p.rapidapi.com/future-value"
    querystring = {}
    if cash_flow:
        querystring['cash_flow'] = cash_flow
    if rate:
        querystring['rate'] = rate
    if number_of_periods:
        querystring['number_of_periods'] = number_of_periods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financecalc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def internal_rate_of_return_irr(initial_investment: int=None, cash_flows: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Internal Rate of Return (IRR) is the discount rate often used in capital budgeting that makes the net present value of all cash flows from a particular project equal to zero."
    initial_investment: initial investment (eg -500000)
        cash_flows: cash flows (eg 200000,300000,200000)
        
    """
    url = f"https://financecalc.p.rapidapi.com/internal-rate-of-return"
    querystring = {}
    if initial_investment:
        querystring['initial_investment'] = initial_investment
    if cash_flows:
        querystring['cash_flows'] = cash_flows
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financecalc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def compound_annual_growth_rate_cagr(ending_value: int, beginning_value: int, number_of_periods: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Compound Annual Growth Rate (CAGR) is the year-over-year growth rate of an investment over a specified period of time."
    ending_value: ending value (eg 19500)
        beginning_value: beginning value (eg 10000)
        number_of_periods: number of periods (eg 3)
        
    """
    url = f"https://financecalc.p.rapidapi.com/compound-annual-growth-rate"
    querystring = {'ending_value': ending_value, 'beginning_value': beginning_value, 'number_of_periods': number_of_periods, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financecalc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def discount_factor_df(number_of_periods: int, rate: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Discount Factor (DF) is the factor by which a future cash flow must be multiplied in order to obtain the present value."
    number_of_periods: number of periods (eg 6)
        rate: rate (eg 10)
        
    """
    url = f"https://financecalc.p.rapidapi.com/discount-factor"
    querystring = {'number_of_periods': number_of_periods, 'rate': rate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financecalc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def compound_interest_ci(principal: int, compoundings_per_period: int, rate: int, number_of_periods: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Compound Interest is the interest calculated on the initial principal and also on the accumulated interest of previous periods of a deposit or loan."
    principal: principal (eg 1500)
        compoundings_per_period: compoundings per period (eg 4)
        rate: rate (eg 4.3)
        number_of_periods: number of periods (eg 6)
        
    """
    url = f"https://financecalc.p.rapidapi.com/compound-interest"
    querystring = {'principal': principal, 'compoundings_per_period': compoundings_per_period, 'rate': rate, 'number_of_periods': number_of_periods, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financecalc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def amortization(total_number_of_payments: int, rate: int, principal: int, type: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Amortization is the paying off of debt with a fixed repayment schedule in regular installments over a period of time."
    total_number_of_payments: Total number of payments (eg 5)
        rate: rate (eg 7.5)
        principal: principal (eg 20000)
        type: type (eg 0)
        
    """
    url = f"https://financecalc.p.rapidapi.com/amortization"
    querystring = {'total_number_of_payments': total_number_of_payments, 'rate': rate, 'principal': principal, }
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financecalc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def xirr(guess: int, cash_flows: str, dates: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "XIRR is used to determine the Internal Rate of Return when the cash flows are at Irregular intervals."
    guess: guess (eg 0)
        cash_flows: cash flows (eg -1000, -100, 1200)
        dates: dates (eg 2015-11-01,2016-07-01,2016-07-19)
        
    """
    url = f"https://financecalc.p.rapidapi.com/xirr"
    querystring = {'guess': guess, 'cash_flows': cash_flows, 'dates': dates, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financecalc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def payback_period_pp(cash_flows: str, number_of_periods: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Payback Period (PP) is the length of time required to recover the cost of an investment."
    cash_flows: cash flows (eg -105, 25)
        number_of_periods: number of periods (eg 0)
        
    """
    url = f"https://financecalc.p.rapidapi.com/payback-period"
    querystring = {'cash_flows': cash_flows, 'number_of_periods': number_of_periods, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financecalc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def present_value_pv(rate: int, cash_flow: str, number_of_periods: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Present Value (PV) is the current worth of a future sum of money or stream of cash flows given a specified rate of return."
    rate: rate (eg 5)
        cash_flow: cash flow (eg 100)
        number_of_periods: number of periods (eg 1)
        
    """
    url = f"https://financecalc.p.rapidapi.com/present-value"
    querystring = {'rate': rate, 'cash_flow': cash_flow, }
    if number_of_periods:
        querystring['number_of_periods'] = number_of_periods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financecalc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def loan_payment_per_period_pmt(principal: int, number_of_payments: int, fractional_interest_rate: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Payment for a loan based on constant payments and a constant interest rate"
    principal: principal (eg -1000000)
        number_of_payments: number of payments (eg 36)
        fractional_interest_rate: fractional interest rate (eg 0.02)
        
    """
    url = f"https://financecalc.p.rapidapi.com/loan-payment-per-period"
    querystring = {'principal': principal, 'number_of_payments': number_of_payments, 'fractional_interest_rate': fractional_interest_rate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financecalc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

