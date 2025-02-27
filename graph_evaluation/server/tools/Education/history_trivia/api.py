import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def today_in_history(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "he Today In History endpoint is your personalized time capsule. On any given day, it delivers events that occurred on the same day in history, helping you connect the past to the present. The endpoint returns a detailed account of the historical event, related images that add a visual dimension to the story, and helpful links for a deeper dive into the subject matter. Ideal for educational platforms, daily historical updates, or simply enriching your knowledge about our shared past."
    
    """
    url = f"https://history-trivia.p.rapidapi.com/today-in-history"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "history-trivia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_historical_trivia(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Random Historical Trivia endpoint is your ticket to a world of historical surprises. It fetches a random historical fact or event from our extensive database, spanning a diverse range of global occurrences. Each trivia piece returned includes a date, description of the event and one or more links for additional reading. Perfect for trivia apps, random fact generators, or anyone seeking a spontaneous journey into history."
    
    """
    url = f"https://history-trivia.p.rapidapi.com/random-trivia"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "history-trivia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

