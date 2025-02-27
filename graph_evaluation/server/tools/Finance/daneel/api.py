import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def news(category: str=None, countries: str=None, orderby: str=None, currencies: str=None, data_start: int=None, data_end: int=None, q: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Newsfeed"
    category: The category you want news
        countries: A comma-separated string of iso3166 countries(eg HT,FR,US). Default all
        orderby: The order to sort the news in. Possible option: date, score (score calculated thanks to our machine learning models).
        currencies: A comma-separated string of symbol currencies
        data_start: Date in timestamp for the oldest news
        data_end: Date in timestamp for the newest news
        q: keyword or phrase to search for.
        
    """
    url = f"https://daneel.p.rapidapi.com/v2/news"
    querystring = {}
    if category:
        querystring['category'] = category
    if countries:
        querystring['countries'] = countries
    if orderby:
        querystring['orderBy'] = orderby
    if currencies:
        querystring['currencies'] = currencies
    if data_start:
        querystring['data_start'] = data_start
    if data_end:
        querystring['data_end'] = data_end
    if q:
        querystring['q'] = q
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "daneel.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_information(currencies: str, last_n_hours: int=None, format: str='Object', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get market sentiment"
    currencies: A comma-separated string of symbol currencies
        last_n_hours: Historic data (array or object)
        format: Specifies the returned format (array or object)
        
    """
    url = f"https://daneel.p.rapidapi.com/v2/sentiment"
    querystring = {'currencies': currencies, }
    if last_n_hours:
        querystring['last_n_hours'] = last_n_hours
    if format:
        querystring['Format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "daneel.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

