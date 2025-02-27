import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fetch_daily_rasipalan(sign: str, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This end points takes a 'GET' request with sign and language string as a parameter and returns a respected daily astrology quotes.
		
		**Sign GET Digits**
		1 - Aries
		2 - Taurus
		3 - Gemini
		4 - Cancer
		5 - Leo
		6 - Virgo
		7 - Libra
		8 - Scorpius
		9 - Sagittarius
		10 - Capricornus
		11 - Aquarius
		12 - Pisces
		
		Using this sign id to fetch respected sign quotes.
		
		**For Language**
		en - English
		ta - Tamil
		ml - Malayalam
		hi - Hindi
		
		and more..."
    
    """
    url = f"https://daily-astrology-rasipalan.p.rapidapi.com/api/rasipalanrapid"
    querystring = {'sign': sign, }
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "daily-astrology-rasipalan.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

