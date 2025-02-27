import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def currency_converter_with_forecast_and_historical_data(currency: str, currency_cross: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Free subscribe plan available! Check it in pricing and subscribe for tests!
		Allows to convert currency and shows prediction of exchange rate for currency pairs.
		
		`country_code` - is now optional. API will try first to get cross rate from European Central Bank. If API will found both currencies in another central bank, you will see from which central bank it was taken.
		
		Example: To get CNY to GBP provided by European Central Bank
		`/api/get_cross?currency=cny&currency_cross=gbp`
		or 
		`/api/get_cross?country_code=eu&currency=cny&currency_cross=gbp`
		or from Kazakhstan's CB:
		`/api/get_cross?country_code=kz&currency=cny&currency_cross=gbp`
		Of course cross rates from KZ and EU will be slightly different.
		
		Example 2: If you want to get historical data, add `&history_yyyy_mm_dd=2022-06-21`
		`/api/get_cross?country_code=eu&currency=cny&currency_cross=gbp&history_yyyy_mm_dd=2022-06-21`
		
		Exotic example: To get CNY to MDL (Yuan to Moldavian Lei) provided by Kazakhstan CB
		`/api/get_cross?currency=cny&currency_cross=mdl`
		
		Parameters:
		1) country_code; 2) currency; 3) currency_cross; 4) optional parameter is `history_yyyy_mm_dd` (example:2022-06-27)' .
		To get available `country_code` - access `/api/info/countries`
		To get all currency abbreviations - access `/api/info/currencies` - there you need `code` value which is having  3 capital letters, like USD or CHF
		
		Example 3: If you want get price of Bitcoin in USD use:
		`/api/get_cross?currency=btc&currency_cross=usd`
		...same in GBP with specified `country_code`:
		`/api/get_cross?country_code=bitcoin&currency=btc&currency_cross=gbp`
		
		Example 4: If you want get price of Bitcoin in USD for a day in the past use:
		`/api/get_cross?country_code=bitcoin&history_yyyy_mm_dd=2022-06-27&currency=btc&currency_cross=usd`"
    
    """
    url = f"https://forecast-crypto-and-fiat-currency-exchange-rates.p.rapidapi.com/api/get_cross?currency=cny&currency_cross=gbp"
    querystring = {'currency': currency, 'currency_cross': currency_cross, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forecast-crypto-and-fiat-currency-exchange-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_banks_in_a_country(country_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Free subscribe plan available! Check it in pricing and subscribe for tests!
		This endpoint gives list of banks in selected country.
		
		Example 1
		` /api/get_all_banks_in_country?country_code=eu`
		
		Example 2
		 `/api/get_all_banks_in_country?country_code=md`
		
		Example 3
		 `/api/get_all_banks_in_country?country_code=bitcoin`
		well... I hope you've got the idea:) The currency must have it's country and bank and for cryptocurrencies it is just imaginable bank "Bitcoin" and imaginable country "Bitcoin". This request will show you JSON
		{... "is_central": 1, "name": "Bitcoin", "name_short": "Bitcoin",...}
		
		To get all available `country_code` use free endpoint `/api/info/countries`"
    
    """
    url = f"https://forecast-crypto-and-fiat-currency-exchange-rates.p.rapidapi.com/api/get_all_banks_in_country"
    querystring = {'country_code': country_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forecast-crypto-and-fiat-currency-exchange-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currency_rates_for_1_bank_historical_data(country_code: str, date_yyyy_mm_dd_from: str, bank_id: str, date_yyyy_mm_dd_till: str, tz: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Free subscribe plan available! Check it in pricing and subscribe for tests!
		**S**o it does - get exchange rates for selected bank on chosen date. Timezone is important!!!
		
		Example 1 - get currency rates provided by ECB(European Central Bank) in Berlin:
		`/api/get_all_currencies_values_for_1bank/?country_code=eu&date_yyyy_mm_dd_from=2021-11-15&date_yyyy_mm_dd_till=2021-11-15&bank_id=15&tz=Europe/Berlin`
		
		Example 2, MAIB (private Moldavian bank): `/api/get_all_currencies_values_for_1bank/?country_code=md&date_yyyy_mm_dd_from=2021-11-12&date_yyyy_mm_dd_till=2021-11-12&bank_id=1&tz=Europe/Chisinau`
		
		Prameters and where to get info:
		`country_code=`   -->> `/api/info/countries`  -here you can get all available country_code's and their's timezones.
		`bank_id=`  -->>  `/api/get_all_banks_in_country?country_code=bitcoin` or `/api/get_all_banks_in_country?country_code=eu`
		`date_yyyy_mm_dd_till=` and `date_yyyy_mm_dd_from=`  -->>  must be equal"
    
    """
    url = f"https://forecast-crypto-and-fiat-currency-exchange-rates.p.rapidapi.com/api/get_all_currencies_values_for_1bank"
    querystring = {'country_code': country_code, 'date_yyyy_mm_dd_from': date_yyyy_mm_dd_from, 'bank_id': bank_id, 'date_yyyy_mm_dd_till': date_yyyy_mm_dd_till, 'tz': tz, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forecast-crypto-and-fiat-currency-exchange-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_info(endpoint_name: str='get_forecast', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Free subscribe plan available! Check it in pricing and subscribe for tests!
		All info about API's of this project are available at `/api/info`     Free of charge.
		For example, if you access  `/api/info/get_forecast`  -  you will get information about  `/api/get_forecast`
		
		The `ENDPOINT_NAME` can be one of these:
		`timezones` , `countries` , `languages` , `currencies` , `author`
		..and it provides response with data, for example all available countries.
		
		Also  `ENDPOINT_NAME` can be one of these:
		`get_forecast` , `get_all_banks_in_country` , `get_best_rates` , `get_all_currencies_values_for_1bank` ,  `faq`
		..but  these requests provides only information about selected endoint.
		
		To get all available `ENDPOINT_NAME` , go to `/api/info/all`"
    
    """
    url = f"https://forecast-crypto-and-fiat-currency-exchange-rates.p.rapidapi.com/api/info/{endpoint_name}"
    querystring = {}
    if endpoint_name:
        querystring['ENDPOINT_NAME'] = endpoint_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forecast-crypto-and-fiat-currency-exchange-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_exchange_rates_forecast(country_code: str, currency: str, predict_days: str='5', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Free subscribe plan available! Check it in pricing and subscribe for tests!
		This API provides convinient way to get forecast for almost all fiat currencies and for bitcoin. Forecast based on Moving Average method with at least 30 historical observations.
		
		Example with fiat- 
		`/api/get_forecast?country_code=eu&currency=usd&predict_days=5`
		List of all available country_code's in `/api/info/countries`
		Info about all currencies: `/api/info/currencies`
		
		Example 1 - 
		`/api/get_forecast?country_code=bitcoin&currency=btc`
		List of all available country_code's in `/api/info/countries`
		Info about all currencies: `/api/info/currencies`
		
		Since 11 April 2022 you can request `/api/get_forecast?country_code=bitcoin&currency=btc&predict_days=10`
		This will provide prediction for 10 days. `&predict_days` accepting integer from 2 to 10
		
		Example 2 - If you want to get forecast for Canadian Dollar in USD:
		`/api/get_forecast?country_code=eu&currency=usd&cross_currency=cad`
		you will get response:
		"`message`":".......... price for 1 CAD in US DOLLAR.... "
		"`forecast_rate`": 0.7896,  -->> means that  forecast for 1 CAD is equal to 0.7896 USD
		"`forecast_date`": "2021-11-21", 
		...
		
		Example 3 - If you want to get forecast of exchange rate for US Dollar in EU (European Union)  provided by European Central Bank:
		`/api/get_forecast?country_code=eu&currency=usd`
		you get:
		"`forecast_rate`": 1.1254,  -->>which means 1 Euro = 1.1254 US Dollar
		"`forecast_date`": "2021-11-21",
		
		Example 4 - get  forecast and exhange rates for 1 USD provided by Moldavian Central Bank in Russian language:
		`/api/get_forecast?country_code=md&lang=ru&currency=eur&cross_currency=usd`
		Info about lang= : `/api/info/languages`
		Info about `cross_currency`= : `/api/info/currencies`"
    
    """
    url = f"https://forecast-crypto-and-fiat-currency-exchange-rates.p.rapidapi.com/api/get_forecast"
    querystring = {'country_code': country_code, 'currency': currency, }
    if predict_days:
        querystring['predict_days'] = predict_days
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forecast-crypto-and-fiat-currency-exchange-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

