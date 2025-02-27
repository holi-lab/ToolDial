import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def barrier_option(strike: int, digital: int, sign: int, arg: str, date: str, time: str, ret: int, h_strike: int, vol: int, fwd: int, rf: int, type_barrier: str, rebate: int, type: str, div: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generic Option to Price a one-barrier Knock-In/Out stype option, with possible Digital Features. Uses only Implied Volatility, and the necessary volatility surface needs to be an external volatility model.
		Parameters as per below:
		
		strike_:  strike of the vanilla option
		 fwd_: the spot of the underlying
		 rf_: risk-free rate. For FX options, this should be the interest-rate differential
		div_: dividend rate for equity options
		 vol_: implied lognormal annualized volatlity
		 type_: 'Call' or 'Put' 
		type_Barrier: type of Barriers, 'UpOut', 'UpIn', 'DownOut', 'DownIn' 
		H_strike: strike of the Barrier feature
		 digital_:  digital = 1 means a digital option rather than vanilla option
		sign_:  set integer = 1 for long the option
		time_: expiry of the option. '1m', , '1w', '6m', '1y' etc, in string format
		date_: option start/pricing date. Usually will be today, but can be variable
		arg_: what data field to return 'NPV' is option premium. Choices are vega, delta, and gamma
		ret_: int =1, return type of value or json."
    
    """
    url = f"https://kiann_options_exotics.p.rapidapi.com/Barrier"
    querystring = {'strike_': strike, 'digital_': digital, 'sign_': sign, 'arg_': arg, 'date_': date, 'time_': time, 'ret_': ret, 'H_strike': h_strike, 'vol_': vol, 'fwd_': fwd, 'rf_': rf, 'type_Barrier': type_barrier, 'rebate_': rebate, 'type_': type, }
    if div:
        querystring['div_'] = div
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kiann_options_exotics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def european_option(sign: str, time: str, date: str, ret: int, arg: str, type: str, vol: int, digital: int, div: int, rf: int, strike: int, fwd: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This function prices a terminal distribution-style options, i.e. of Vanilla Call and Digital Options, single-currency.
		The parameters are as follows :
		strike_ : strike of the option.
		fwd_ : the *** spot *** of the function. The projected forward rate will incorprate the funding/risk-free rf, and dividend rate, div_
		rf_ : the risk-free rate in the BlackScholes model
		div_ : the dividend yield in the BlackScholes model
		vol_ : The lognormal implied volatility, annualized.
		type_ : 'Call' or 'Put'
		sign_ : 'Long' or 'Short' the option
		digital_ : enum, 1 is digital option, 0 is normal option
		arg_ : the type of value returned. NPV (premium),  'delta', 'gamma',  'vega', impliedVolatility'"
    
    """
    url = f"https://kiann_options_exotics.p.rapidapi.com/European"
    querystring = {'sign_': sign, 'time_': time, 'date_': date, 'ret_': ret, 'arg_': arg, 'type_': type, 'vol_': vol, 'digital_': digital, 'div_': div, 'rf_': rf, 'strike_': strike, 'fwd_': fwd, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kiann_options_exotics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

