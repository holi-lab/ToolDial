import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def camelizer_get_prices(asin: str, locale: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get latest prices, history highest/lowest prices of a product from Amazon"
    asin: The asin number of a product from Amazon, and must be in the correct format. Ex : B08QB93S6R
        locale: One of the following : au|ca|fr|de|it|es|uk|us
        
    """
    url = f"https://price-tracking-tools.p.rapidapi.com/camelizer/get-prices"
    querystring = {'asin': asin, }
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "price-tracking-tools.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def camelizer_get_price_chart(asin: str, lang: str='en', tp: str='3m', w: int=720, h: int=480, country: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get pricing history chart of a product from Amazon"
    asin: The asin number of a product from Amazon, and must be in the correct format. Ex : B08QB93S6R
        lang: One of the following : en|fr|de|it|es
        tp: Time period. One of the following : 1m|3m|6m|1y|all
        w: The width of the chart
        h: The height of the chart
        country: One of the following : au|ca|fr|de|it|es|uk|us
        
    """
    url = f"https://price-tracking-tools.p.rapidapi.com/camelizer/get-price-chart"
    querystring = {'asin': asin, }
    if lang:
        querystring['lang'] = lang
    if tp:
        querystring['tp'] = tp
    if w:
        querystring['w'] = w
    if h:
        querystring['h'] = h
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "price-tracking-tools.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

