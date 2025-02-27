import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_calculate_investment(rate: int, principal: int, time: int, compounding: str='yearly', in_currency: str='USD', format: str='yes', to_currency: str='USD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint performs multiple investment calculations in a single request. It accepts a list of investment parameters, such as principal amount, rate of return, time period, and compounding frequency, and returns a list of investment results, including the simple interest, compound interest, investment returns, return on investment, capital gains tax, and annual percentage yield of each investment."
    rate: The interest rate of the investment
        principal: The amount of money invested
        time: The length of the investment in months
        compounding: The frequency of compounding interest. Valid options are **yearly**, **semi-annually**, **quarterly**, and **monthly**. Default is **yearly**
        in_currency: The currency of the investment. Default is USD

Currencies currently supported include:
Currency	Code	Name
NOK	kr	Norwegian Krone
EUR	€	Euro
HKD	$	Hong Kong Dollar
CZK	Kč	Czech Koruna
THB	฿	Thai Baht
CHF	CHF	Swiss Franc
USD	$	US Dollar
ISK	kr	Icelandic Króna
GBP	£	British Pound Sterling
JPY	¥	Japanese Yen
BGN	лв	Bulgarian Lev
CNY	¥	Chinese Yuan
IDR	Rp	Indonesian Rupiah
TRY	₺	Turkish Lira
NZD	$	New Zealand Dollar
SGD	$	Singapore Dollar
DKK	kr	Danish Krone
AUD	$	Australian Dollar
BRL	R$	Brazilian Real
ILS	₪	Israeli Shekel
SEK	kr	Swedish Krona
CAD	$	Canadian Dollar
HUF	Ft	Hungarian Forint
ZAR	R	South African Rand
MYR	RM	Malaysian Ringgit
KRW	₩	South Korean Won
MXN	$	Mexican Peso
INR	₹	Indian Rupee
PHP	₱	Philippine Peso
PLN	zł	Polish Zloty
RON	lei	Romanian Leu
        format: Whether to format the result. Valid options are **yes** and **no**. Default is **yes**
        to_currency: The currency to convert the result to. Default is USD

Currencies currently supported include:
Currency	Code	Name
NOK	kr	Norwegian Krone
EUR	€	Euro
HKD	$	Hong Kong Dollar
CZK	Kč	Czech Koruna
THB	฿	Thai Baht
CHF	CHF	Swiss Franc
USD	$	US Dollar
ISK	kr	Icelandic Króna
GBP	£	British Pound Sterling
JPY	¥	Japanese Yen
BGN	лв	Bulgarian Lev
CNY	¥	Chinese Yuan
IDR	Rp	Indonesian Rupiah
TRY	₺	Turkish Lira
NZD	$	New Zealand Dollar
SGD	$	Singapore Dollar
DKK	kr	Danish Krone
AUD	$	Australian Dollar
BRL	R$	Brazilian Real
ILS	₪	Israeli Shekel
SEK	kr	Swedish Krona
CAD	$	Canadian Dollar
HUF	Ft	Hungarian Forint
ZAR	R	South African Rand
MYR	RM	Malaysian Ringgit
KRW	₩	South Korean Won
MXN	$	Mexican Peso
INR	₹	Indian Rupee
PHP	₱	Philippine Peso
PLN	zł	Polish Zloty
RON	lei	Romanian Leu
        
    """
    url = f"https://calcx-investment-api.p.rapidapi.com/calculate"
    querystring = {'rate': rate, 'principal': principal, 'time': time, }
    if compounding:
        querystring['compounding'] = compounding
    if in_currency:
        querystring['in_currency'] = in_currency
    if format:
        querystring['format'] = format
    if to_currency:
        querystring['to_currency'] = to_currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "calcx-investment-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

