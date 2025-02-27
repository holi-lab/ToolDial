import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_the_chance_of_rejection(me: str, crush: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "When getting this requests, you get back the chance you will not be rejected by your crush..
		
		!! DISCLAMER !! 
		Use at your own risk of mental health, its just an computer generated result. The real world can be different 
		THIS IS NOT DATING ADVISE
		!! ---------------  !!"
    
    """
    url = f"https://love-calculator11.p.rapidapi.com/chances"
    querystring = {'me': me, 'crush': crush, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "love-calculator11.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_percentage_that_your_crush_loves_you_back(crush: str, me: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "When getting this requests, you get back how many percantage your crush loves you back. 
		
		!! DISCLAMER !! 
		Use at your own risk of mental health, its just an computer generated result. The real world can be different 
		THIS IS NOT DATING ADVISE
		!! ---------------  !!"
    
    """
    url = f"https://love-calculator11.p.rapidapi.com/percentage"
    querystring = {'crush': crush, 'me': me, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "love-calculator11.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

