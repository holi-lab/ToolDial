import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def random(category: str='love', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Introducing the "Random" endpoint from DaddyApi's Quotes - the perfect solution to add a touch of inspiration, motivation, and thought-provoking content to your projects or applications. With access to over 1 million quotes spanning thousands of diverse categories, you'll effortlessly tap into the wisdom of renowned figures, historical events, movies, and more. Customize your quote selection using the optional category feature, giving you even greater control over the content you display. Enhance your user experience through seamless integration, user-friendly documentation, and dependable support. Experience the power of the "Random" endpoint by DaddyApi's Quotes and bring wisdom, wit, and wonder to your audience in an instant, tailored to your needs."
    
    """
    url = f"https://quotes-daddy.p.rapidapi.com/random"
    querystring = {}
    if category:
        querystring['category'] = category
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quotes-daddy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

