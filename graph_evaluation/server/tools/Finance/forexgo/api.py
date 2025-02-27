import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fx(amount: int, is_from: str, to: str, date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Introducing ForexGo API - the powerful currency exchange solution for developers. Effortlessly integrate real-time forex rates and conversion capabilities into your applications with our easy-to-use API. Enhance your projects with accurate and up-to-date exchange data, empowering users across the globe to make informed financial decisions. Get started with ForexGo API today!"
    amount: Defines the value of the amount.
        from: Supported currencies are listed below.

EUR, USD, JPY, BGN, CZK, DKK, GBP, HUF, PLN, RON, SEK, CHF, ISK, NOK, TRY, AUD, BRL, CAD, CNY, HKD, IDR, ILS, INR, KRW, MXN, MYR, NZD, PHP, SGD, THB, ZAR
        to: Supported currencies are listed below.

EUR, USD, JPY, BGN, CZK, DKK, GBP, HUF, PLN, RON, SEK, CHF, ISK, NOK, TRY, AUD, BRL, CAD, CNY, HKD, IDR, ILS, INR, KRW, MXN, MYR, NZD, PHP, SGD, THB, ZAR
        date: ISO Date format:

YYYY-MM-DDTHH:mm:ss.sssZ

Pull data from requested date.
        
    """
    url = f"https://forexgo.p.rapidapi.com/fx"
    querystring = {'amount': amount, 'from': is_from, 'to': to, }
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forexgo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

