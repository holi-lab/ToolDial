import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_yearly_horoscope_with_name(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Yearly Horoscope With Name
		
		**name** field
		**English Language**
		Aries
		Taurus
		Gemini
		Cancer
		Leo
		Virgo
		Libra
		Scorpio
		Sagittarius
		Capricorn
		Aquarius
		Pisces
		
		**Turkish Language**
		Koç
		Boğa
		İkizler
		Yengeç
		Aslan
		Başak
		Terazi
		Akrep
		Yay
		Oğlak
		Kova
		Balık"
    
    """
    url = f"https://horoscope20.p.rapidapi.com/yearly/{name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horoscope20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_monthly_horoscope_with_name(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Monthly Horoscope With Name
		
		**name** field
		**English Language**
		Aries
		Taurus
		Gemini
		Cancer
		Leo
		Virgo
		Libra
		Scorpio
		Sagittarius
		Capricorn
		Aquarius
		Pisces
		
		**Turkish Language**
		Koç
		Boğa
		İkizler
		Yengeç
		Aslan
		Başak
		Terazi
		Akrep
		Yay
		Oğlak
		Kova
		Balık"
    
    """
    url = f"https://horoscope20.p.rapidapi.com/monthly/{name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horoscope20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_today_horoscope_with_name(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can get **today** horoscope with name
		
		**name** field
		**English Language**
		Aries
		Taurus
		Gemini
		Cancer
		Leo
		Virgo
		Libra
		Scorpio
		Sagittarius
		Capricorn
		Aquarius
		Pisces
		
		**Turkish Language**
		Koç
		Boğa
		İkizler
		Yengeç
		Aslan
		Başak
		Terazi
		Akrep
		Yay
		Oğlak
		Kova
		Balık"
    name: ENGLISH Language
- Aries
- Taurus
- Gemini
- Cancer
- Leo
- Virgo
- Libra
- Scorpio
- Sagittarius
- Capricorn
- Aquarius
- Pisces

TURKISH Language
- Koç
- Boğa
- İkizler
- Yengeç
- Aslan
- Başak
- Terazi
- Akrep
- Yay
- Oğlak
- Kova
- Balık
        
    """
    url = f"https://horoscope20.p.rapidapi.com/daily/{name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horoscope20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

