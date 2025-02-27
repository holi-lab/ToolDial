import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def profitability_index_pi(iinvest: str, cflow: str, rate: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a profitability Index.
		
		What is the Internal Rate of return of an investment that requires $15000 deposit now and then makes the following withdrawals at regular (fixed) intervals: 1500, 2500, 3500, 4500, 6000?. Assuming the ending value is 0.
		
		Required Parameters:
		-> rate
		-> iinvest (initial investment)
		->cflow (cash flows)
		
		// e.g., If rate is 10%, initial investment is -$40,000, cash flows are $18,000, $12,000, $10,000, $9,000, and $6,000, PI is 1.09."
    cflow: (String) Must be separated by a comma. i.e: 18000,12000
        
    """
    url = f"https://financial-calculators.p.rapidapi.com/pi"
    querystring = {'iinvest': iinvest, 'cflow': cflow, 'rate': rate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-calculators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def net_present_value_npv(rate: str, cflow: str, iinvest: str='500000', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the Net Present Value (NPV) compares the money received in the future to an amount of money received today, while accounting for time and interest [through the discount rate]. 
		It's based on the principal of time value of money (TVM), which explains how time affects monetary value.
		
		Required Parameters
		-> rate
		-> cflow (cash flows)
		-> iinvest (initial investment)
		
		// e.g., If discount rate is 10%, initial investment is -$1,000, cash flow in year 1 is $200,000, year 2 is $300,000, and year 3 is $200,000, the NPV is $80,015.03."
    cflow: must be a string separated by a comma. i.e: 200000,300000,200000
        
    """
    url = f"https://financial-calculators.p.rapidapi.com/npv"
    querystring = {'rate': rate, 'cflow': cflow, }
    if iinvest:
        querystring['iinvest'] = iinvest
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-calculators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def compound_interest_ci(cperiod: str, rate: str, nperiods: str, principal: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a Compound Interest is the interest calculated on the initial principal and also on the accumulated interest of previous periods of a deposit or loan.
		
		Required Parameters
		         -> rate
		         ->cperiod (compoundings per period)
		         ->principal
		         ->nperiods (number of periods)
		
		// e.g., If rate is 4.3%, the compoundings per period is 4, the principal is $1,500, and the number of periods is 6, the compound interest is $1,938.84."
    
    """
    url = f"https://financial-calculators.p.rapidapi.com/ci"
    querystring = {'cperiod': cperiod, 'rate': rate, 'nperiods': nperiods, 'principal': principal, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-calculators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def car_loan_calculator_with_yearly_amortization_schedule(cprice: str, rate: str, downpayment: str, loant: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the Home loan with Yearly Amortization
		
		i.e:  How much will be my monthly payment on a car that cost $25000 with 20% as a down payment, with 5% interest rate and term of the loan is 10 years
		
		rate=5
		loant=10 
		cprice=25000 
		downpayment=20"
    
    """
    url = f"https://financial-calculators.p.rapidapi.com/carloanyear"
    querystring = {'cprice': cprice, 'rate': rate, 'downpayment': downpayment, 'loant': loant, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-calculators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def payback_period_pp(nperiods: str, cflow: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Payback Period (PP) is the length of time required to recover the cost of an investment.
		
		Required parameters
		->nperiods (number of periods)
		->cflow (cash flows)
		
		number of periods takes a 0 value for even cash flows;
		for uneven cash flows, number of periods takes any number of projected periods.
		
		[cash flows] takes any number of projected cash flows.
		
		#Uneven Cash Flows
		 // e.g., If number of periods is 5, initial investment is -$50, and the cash flows are $10, $13, $16, $19, and $22 for each year, the payback period is 3.42 years.
		
		 finance.PP(5, -50, 10, 13, 16, 19, 22);
		 => 3.42"
    cflow: [String] values must be separated by a comma. i.e:-50,10,13,16,19,22. The first values also has to be negative has a initial investment.
        
    """
    url = f"https://financial-calculators.p.rapidapi.com/pp"
    querystring = {'nperiods': nperiods, 'cflow': cflow, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-calculators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leverage_ratio_lr(tliability: str, tdebts: str, tincome: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the Leverage Ratio (LR) that is used to calculate the financial leverage of a company or individual to get an idea of the methods of financing or to measure ability to meet financial obligations.
		
		Required Parameters
		->tliability (total liabilities)
		->tdebts (total debts)
		->tincome (total income)
		
		// e.g., If total liabilities are $25, total debts are $10, and total income is $20, the leverage ratio is 1.75."
    
    """
    url = f"https://financial-calculators.p.rapidapi.com/lr"
    querystring = {'tliability': tliability, 'tdebts': tdebts, 'tincome': tincome, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-calculators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def present_value_calculation(fv: str, rate: str, nper: str, pmt: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the Present Value
		
		i.e:
		      -> What is the present value (e.g., the initial investment) of an investment that needs to total $20,000 after 10 years of saving $100 every month? Assume the interest rate is 7% (annually) compounded monthly.
		
		           -> rate=0.07  **(Note that the value must be decimal)**
		           -> nper=10
		           -> pmt=100
		           -> fv=20000"
    
    """
    url = f"https://financial-calculators.p.rapidapi.com/pv"
    querystring = {'fv': fv, 'rate': rate, 'nper': nper, 'pmt': pmt, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-calculators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def car_loan_calculator(cprice: str, loant: str, rate: str, downpayment: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the Home loan with Monthly Amortization
		
		i.e:  How much will be my monthly payment on a car that cost $25000 with 20% as a down payment, with 5% interest rate and term of the loan is 10 years
		
		rate=5
		loant=10 
		cprice=25000 
		downpayment=20"
    
    """
    url = f"https://financial-calculators.p.rapidapi.com/carloan"
    querystring = {'cprice': cprice, 'loant': loant, 'rate': rate, 'downpayment': downpayment, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-calculators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def home_loan_calculator_with_yearly_amortization_schedule(downpayment: str, loant: str, hvalue: str, rate: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the Home loan with Yearly Amortization
		
		i.e:  How much will be my monthly payment on a house that cost $100000 with 20% as a down payment, with 5% interest rate and term of the loan is 30 years
		rate=5
		loant=30
		hvalue=100000
		downpayment=20"
    
    """
    url = f"https://financial-calculators.p.rapidapi.com/homeloanyear"
    querystring = {'downpayment': downpayment, 'loant': loant, 'hvalue': hvalue, 'rate': rate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-calculators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def principal_and_interest_payment_calculation_with_monthly_amortization_schedule(loant: str, lamount: str, rate: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the Principal and Interest payment calculation with Monthly Amortization Schedule.
		
		
		i.e: What is the amortization schedule for a 1 year loan of $5000 at 10% interest per year compounded monthly? And the total interest payments?
		
		            -> rate=10
		            -> loant= 1
		            -> lamount=5000"
    
    """
    url = f"https://financial-calculators.p.rapidapi.com/amortizationcalc"
    querystring = {'loant': loant, 'lamount': lamount, 'rate': rate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-calculators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def principal_and_interest_payment_calculation_with_yearly_amortization_schedule(lamount: str='5000', loant: str='1', rate: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the Principal and Interest payment calculation with yearly Amortization Schedule.
		
		
		i.e: What is the amortization schedule for a 1 year loan of $5000 at 10% interest per year compounded monthly? And the total interest payments?
		
		            -> rate=10
		            -> loant= 1
		            -> lamount=5000"
    
    """
    url = f"https://financial-calculators.p.rapidapi.com/amortizationcalcyear"
    querystring = {}
    if lamount:
        querystring['lamount'] = lamount
    if loant:
        querystring['loant'] = loant
    if rate:
        querystring['rate'] = rate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-calculators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def future_value_calculation(pmt: str, rate: int, pv: str, nper: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the Future Value.
		
		i.e: What is the future value after 30 years of saving $1000 now, with an additional monthly savings of $100. Assume the interest rate is 7% (annually) compounded monthly?
		
		rate= 0.07 (Rate has to be in decimal)
		nper=30
		pmt=-100 (Has to be negative)
		pv=1000"
    
    """
    url = f"https://financial-calculators.p.rapidapi.com/fv"
    querystring = {'pmt': pmt, 'rate': rate, 'pv': pv, 'nper': nper, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-calculators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def payments_calculation(nper: str, pv: str, rate: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the monthly payment on a Loan.
		
		i.e:
		- What is the monthly payment needed to pay off a $100,000 loan in 10 years at an annual interest rate of 2.5%?
		            * **rate=> 0.025 **
		            * nper=> 10
		             *pv: 100000
		result: 942.69"
    
    """
    url = f"https://financial-calculators.p.rapidapi.com/pmt"
    querystring = {'nper': nper, 'pv': pv, 'rate': rate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-calculators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def number_of_period_calculation(rate: str, pmt: str, pv: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the Number of period.
		
		i.e:
		
		
		-> If we only had $500/month to pay towards the loan, how long would it take to pay-off a loan of $10,000 at 3% annual interest?
		         -> rate:=0.03  ** (Note that the rate is has to be converted to decimal, 3%= 0.03)**
		         -> pmt= 500
		         -> pv= 10000
		
		**Result: 20 (Meaning: So, over 20 months would be required to pay off the loan.)**"
    
    """
    url = f"https://financial-calculators.p.rapidapi.com/nper"
    querystring = {'rate': rate, 'pmt': pmt, 'pv': pv, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-calculators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def inflation_adjusted_return(ireturn: str, irate: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint Measure the return taking into account the time period's inflation rate.
		
		Parameters
		-> ireturn (investment Return)
		-> irate (inflation Rate)"
    
    """
    url = f"https://financial-calculators.p.rapidapi.com/iar"
    querystring = {'ireturn': ireturn, 'irate': irate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-calculators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def weighted_average_cost_of_capital_wacc(cequity: str, trate: str, mvalued: str, cdebt: str, mvaluee: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a Weighted Average Cost of Capital (WACC) is the rate that a company is expected to pay on average to all its security holders to finance its assets.
		
		
		Parameters:
		->mvaluee (market value of equity)
		->mvalued (market value of debt)
		->cequity (cost of equity)
		->cdebt (cost of debt)
		->trate (tax rate)
		
		// e.g., If market value of equity is $600,000, market value of debt is $400,000, cost of equity is 6%, cost of debt is 5%, and tax rate is 35%, WACC is 4.9%."
    
    """
    url = f"https://financial-calculators.p.rapidapi.com/wacc"
    querystring = {'cequity': cequity, 'trate': trate, 'mvalued': mvalued, 'cdebt': cdebt, 'mvaluee': mvaluee, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-calculators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rule_of_72_r72(rate: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Rule of 72 (R72) is a rule stating that in order to find the number of years required to double your money at a given 
		This endpoint returns the interest rate, you divide the compound return into 72.
		
		Parameters:
		->rate
		
		// e.g., If annual rate is 10%, rule of 72 is 7.2 years."
    
    """
    url = f"https://financial-calculators.p.rapidapi.com/r72"
    querystring = {'rate': rate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-calculators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def return_on_investment_roi(binvest: str, earnings: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a Return on Investment (ROI) is a simple calculation that tells you the bottom line return of any investment.
		
		
		Parameters
		->binvest (initial investment)
		->earnings
		
		// e.g., If initial investment is -$55,000 and the earnings are $60,000, the return on investment is 9.09%."
    
    """
    url = f"https://financial-calculators.p.rapidapi.com/roi"
    querystring = {'binvest': binvest, 'earnings': earnings, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-calculators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def discount_factor_df(nperiods: str, rate: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Endpoint Returns The Discount Factor (DF) is the factor by which a future cash flow must be multiplied in order to obtain the present value.
		
		Required parameters
		   ->rate
		   ->nperiods
		
		// e.g., If rate is 10% and the number of periods is 6, the result is an array of discount factors: [1, 0.91, 0.827, 0.752, 0.684]."
    
    """
    url = f"https://financial-calculators.p.rapidapi.com/df"
    querystring = {'nperiods': nperiods, 'rate': rate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-calculators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def compound_annual_growth_rate_cagr(evalue: str='19500', nperiods: str='3', bvalue: str='10000', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a Compound Annual Growth Rate (CAGR) is the year-over-year growth rate of an investment over a specified period of time.
		
		
		    ->bvalue (beginning value)
		    ->evalue(ending value)
		    ->nperiods (number of periods)
		
		// e.g., If the beginning value is $10,000, the ending value is $19,500, and the number of periods is 3, the CAGR is 24.93%."
    
    """
    url = f"https://financial-calculators.p.rapidapi.com/cagr"
    querystring = {}
    if evalue:
        querystring['evalue'] = evalue
    if nperiods:
        querystring['nperiods'] = nperiods
    if bvalue:
        querystring['bvalue'] = bvalue
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-calculators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def internal_rate_of_return_calculation(iinvest: str='500,500,500,500', cflow: str='1500', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint Returns the internal rate of return for a series of cash flows. (IRR)"
    iinvest: Values must be separated by a comma. i.e: 500,500,500,500
        
    """
    url = f"https://financial-calculators.p.rapidapi.com/irr"
    querystring = {}
    if iinvest:
        querystring['iinvest'] = iinvest
    if cflow:
        querystring['cflow'] = cflow
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-calculators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def interest_rate_calculation(pmt: str, pv: str, fv: str, nper: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "i.e:
		-> What is the interest rate for a $8,000 loan if the loan term is 5 years and and payments are $152.50 monthly?
		
		        ->pv=8000
		        ->nper=5
		        ->pmt=152.5
		        ->fv=0 (future value)
		        ->"
    
    """
    url = f"https://financial-calculators.p.rapidapi.com/rate"
    querystring = {'pmt': pmt, 'pv': pv, 'fv': fv, 'nper': nper, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-calculators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def home_loan_calculator(hvalue: str, downpayment: str, loant: str, rate: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the Home loan with Monthly Amortization
		
		i.e:  How much will be my monthly payment on a house that cost $100000 with 20% as a down payment, with 5% interest rate and term of the loan is 30 years
		rate=5
		loant=30
		hvalue=100000
		downpayment=20"
    
    """
    url = f"https://financial-calculators.p.rapidapi.com/homeloan"
    querystring = {'hvalue': hvalue, 'downpayment': downpayment, 'loant': loant, 'rate': rate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-calculators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

