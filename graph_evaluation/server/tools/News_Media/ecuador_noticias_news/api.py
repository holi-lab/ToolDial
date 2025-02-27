import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_news_by_search_query(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Este endpoint devolverá todas las últimas noticias de Ecuador de 7 diarios que contengan el parámetro de búsqueda.
		This endpoint will return all the latest news from Ecuador from 7 newspapers containing the search parameter."
    
    """
    url = f"https://ecuador-noticias-news.p.rapidapi.com/news/search"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecuador-noticias-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_news_ecuador(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Este endpoint devolverá todas las noticias más recientes del Ecuador de 7 diarios.
		This endpoint will return all the latest news from Ecuador from 7 newspapers."
    
    """
    url = f"https://ecuador-noticias-news.p.rapidapi.com/news"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecuador-noticias-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

