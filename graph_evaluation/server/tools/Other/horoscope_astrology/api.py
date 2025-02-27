import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def daily_phrase(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a daily phrase"
    
    """
    url = f"https://horoscope-astrology.p.rapidapi.com/dailyphrase"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horoscope-astrology.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def numerology(n: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "."
    
    """
    url = f"https://horoscope-astrology.p.rapidapi.com/numerology"
    querystring = {'n': n, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horoscope-astrology.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sign(s: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Users can access the endpoint by sending a request for a specific sign, and receive a response with in-depth information about the traits, personality, and characteristics associated with that sign. This information can include compatibility with other signs, strengths and weaknesses, and general insights into the individual's nature and tendencies. The endpoint is designed to be easy to use, with a clear and concise format that makes it simple to access and understand the information."
    s: - aries
- taurus
- gemini
- cancer
- leo
- virgo
- libra
- scorpio
- sagittarius
- capricorn
- aquarius
- pisces
        
    """
    url = f"https://horoscope-astrology.p.rapidapi.com/sign"
    querystring = {'s': s, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horoscope-astrology.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def compatibility(sign1: str, sign2: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "."
    sign1: - aries
- taurus
- gemini
- cancer
- leo
- virgo
- libra
- scorpio
- sagittarius
- capricorn
- aquarius
- pisces
        sign2: - aries
- taurus
- gemini
- cancer
- leo
- virgo
- libra
- scorpio
- sagittarius
- capricorn
- aquarius
- pisces
        
    """
    url = f"https://horoscope-astrology.p.rapidapi.com/affinity"
    querystring = {'sign1': sign1, 'sign2': sign2, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horoscope-astrology.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_three_tarot_card(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a randomly selected tarot card from a traditional tarot deck, along with its corresponding interpretation and meaning. The tarot card reading is generated using a randomized algorithm, offering users a unique and personalized tarot experience. The API is designed to be easy to use, allowing developers to integrate tarot card readings into their own applications and websites."
    
    """
    url = f"https://horoscope-astrology.p.rapidapi.com/threetarotcards"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horoscope-astrology.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_tarot_card(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a randomly selected tarot card from a traditional tarot deck, along with its corresponding interpretation and meaning. The tarot card reading is generated using a randomized algorithm, offering users a unique and personalized tarot experience. The API is designed to be easy to use, allowing developers to integrate tarot card readings into their own applications and websites."
    
    """
    url = f"https://horoscope-astrology.p.rapidapi.com/tarotcard"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horoscope-astrology.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def daily_horoscope(sunsign: str, day: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A daily horoscope is a personalized astrological prediction for an individual based on their birth date and zodiac sign. It provides insight and guidance on various aspects of life such as love, career, finances, and personal growth. The predictions take into account the current positions of the planets and other celestial bodies, offering a unique perspective on the individual's current astrological influences. Daily horoscopes are meant to be used as a tool for reflection and can provide helpful insights and advice for navigating life's challenges and opportunities. Whether you're looking to start your day off on the right foot or seeking guidance in a specific area of your life, a daily horoscope can be a valuable resource for gaining new insights and perspective."
    sunsign: - aries
- taurus
- gemini
- cancer
- leo
- virgo
- libra
- scorpio
- sagittarius
- capricorn
- aquarius
- pisces
        day: - Today

- Yesterday

- Tomorrow

- Week

- Month

- Year
        
    """
    url = f"https://horoscope-astrology.p.rapidapi.com/horoscope"
    querystring = {'sunsign': sunsign, 'day': day, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horoscope-astrology.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

