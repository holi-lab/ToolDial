import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def locksmith_hampton(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All you need for hassle-free locksmith service in London is Speedy Locksmith. It is a reliable and trustworthy locksmith company with advanced skills and modern tools that help to ensure immediate and hassle-free locksmith service in no time.
		They are men of solid experience and have managed to solve different types of issues. 
		
		A safe home and security are the main things that every one of us wants and strive for. However, it is quite difficult to find the right locksmith company around, not because it is a lack of them but because they are a few of them that are trustworthy enough to rely on.
		Speedy Locksmith is one of the companies that do not brag all the time about successful projects but strives to satisfy every customer's need and demand.
		In Hampton, you can freely trust them who will strengthen your safety and security in the nick of the time with quality materials and modern tools. They are the best when the matter omer sit seed, quality and affordable locksmith services.
		Call on 020 3404 3416 or simply visit the[ website ](https://speedylocksmith.co.uk/emergency-locksmith/hampton-tw12/)and make an online booking."
    
    """
    url = f"https://locksmith-hampton.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "locksmith-hampton.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

