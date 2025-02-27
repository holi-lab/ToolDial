import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_news(language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves news from all the newspapers in the database.
		
		Use parameters "pt-br" for portuguese and "en" for english news."
    language: insert \\\"pt-br\\\" for portuguese and \\\"en\\\" for english
        
    """
    url = f"https://world-cup-api-with-pt-br.p.rapidapi.com/news/{language}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-cup-api-with-pt-br.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_news_by_paper(newspapername: str, language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves news by newspaper.
		
		Use parameters "pt-br" for portuguese and "en" for english news.
		
		Use the name of the paper as a parameter, example:
		"https://api-copa-do-mundo.herokuapp.com//news/pt-br/paper/globo"
		"https://api-copa-do-mundo.herokuapp.com/news/en/paper/foxsports""
    
    """
    url = f"https://world-cup-api-with-pt-br.p.rapidapi.com/news/{language}/paper/{newspapername}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-cup-api-with-pt-br.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

