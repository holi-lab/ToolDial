import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def shelter(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "dog shelters in Budapest"
    
    """
    url = f"https://petlovers.p.rapidapi.comshelter/id"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "petlovers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def attaboy_street_food_bistro(name: str='fvef', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Located in Síp utca, Attaboy Street Food Bistro is awaiting you and your four-legged friend with special hamburgers, ever-changing limited offers, thin crust pizza, fresh salads and heavenly desserts. The homely eatery delivers the best, high-quality street food to your table reimagined with a special ‘Attaboy’ twist. From American classics to gourmet food creations, you’ll find everything your taste buds desire. Are you vegetarian or you have special dietary needs? Don’t worry, Attaboy supplies all your needs. Meat-free specialties and diet-friendly options are also available. Wait no more, head to Síp utca and put Attaboy’s delicious meals to test! All pups are welcome!"
    name: dvfdv
        
    """
    url = f"https://petlovers.p.rapidapi.com/listbars/"
    querystring = {}
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "petlovers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

