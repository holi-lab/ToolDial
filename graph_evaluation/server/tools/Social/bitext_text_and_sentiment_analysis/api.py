import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def categories_results(authorization: str, categories_result_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "categories results"
    authorization: Copy your authorization token replacing the string "abcdefghijklmnopqrstvwxyz"
        categories_result_id: Copy the -resultid- code obtained from the previous POST call to categories endpoint, replacing the string "YourPOSTResultIdHere"
        
    """
    url = f"https://bitext-bitext-sentiment-analysis.p.rapidapi.com/categories/{categories_result_id}/"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitext-bitext-sentiment-analysis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def entities_results(authorization: str, entities_result_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "entities results"
    authorization: Copy your authorization token replacing the string "abcdefghijklmnopqrstvwxyz"
        entities_result_id: Copy the -resultid- code obtained from the previous POST call to entities endpoint, replacing the string "YourPOSTResultIdHere"
        
    """
    url = f"https://bitext-bitext-sentiment-analysis.p.rapidapi.com/entities/{entities_result_id}/"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitext-bitext-sentiment-analysis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sentiment_results(authorization: str, sentimentresultiid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "sentiment results"
    authorization: Copy your authorization token replacing the string "abcdefghijklmnopqrstvwxyz"
        sentimentresultiid: Copy the -resultid- code obtained from the previous POST call to sentiment endpoint, replacing the string "YourPOSTResultIdHere"
        
    """
    url = f"https://bitext-bitext-sentiment-analysis.p.rapidapi.com/sentiment/{sentimentresultiid}/"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitext-bitext-sentiment-analysis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def concepts_results(authorization: str, concepts_result_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "concepts results"
    authorization: Copy your authorization token replacing the string "abcdefghijklmnopqrstvwxyz"
        concepts_result_id: Copy the -resultid- code obtained from the previous POST call to concepts endpoint, replacing the string "YourPOSTResultIdHere"
        
    """
    url = f"https://bitext-bitext-sentiment-analysis.p.rapidapi.com/concepts/{concepts_result_id}/"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitext-bitext-sentiment-analysis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

