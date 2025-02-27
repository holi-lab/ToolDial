import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get(language: str, number: int, authorization: str='YzdiOTg1YmItZWM1Zi00YTQ3LTk3MWItNDExMzEyMWE3NGVlOlc3T09UTXNrbGVnZkhqRm5UcFpBRDllbFdVNXBqSTM0cXREb2NLWUpVNjFUWGw1R2sxajJFRTNDdUFKamxJT1Q=', content_type: str='application/json', accept_charset: str='utf-8', locale: str='USA', units: str='cent', currency: str='dollar', output: str='text', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get equivalent words of this number"
    
    """
    url = f"https://number2words4.p.rapidapi.com/v1/"
    querystring = {'language': language, 'number': number, }
    if authorization:
        querystring['Authorization'] = authorization
    if content_type:
        querystring['Content-Type'] = content_type
    if accept_charset:
        querystring['Accept-Charset'] = accept_charset
    if locale:
        querystring['locale'] = locale
    if units:
        querystring['units'] = units
    if currency:
        querystring['currency'] = currency
    if output:
        querystring['output'] = output
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "number2words4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

