import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def newsfirst_english_sri_lanka(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "NewsFirst English delivers the latest news and insights from Sri Lanka, covering politics, sports, entertainment, and more. Stay informed and connected with up-to-date coverage and analysis tailored for an international audience. Subscribe to NewsFirst English for the top stories from the island nation."
    
    """
    url = f"https://sri-lanka-latest-news.p.rapidapi.com/articles/newsfirst"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sri-lanka-latest-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def newswire_sri_lanka(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Discover the pulse of Sri Lanka with Newswire News. Uncover breaking stories, politics, sports, entertainment, and cultural updates from the vibrant island nation. Stay informed and engaged with up-to-date news tailored for a global audience. Subscribe to Newswire News Sri Lanka today."
    
    """
    url = f"https://sri-lanka-latest-news.p.rapidapi.com/articles/newswire"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sri-lanka-latest-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sirasa_news_sri_lanka(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Sirasa News Sri Lanka delivers reliable, timely, and comprehensive news updates in Sinhala. Stay connected to the heart of Sri Lanka with our SEO-friendly coverage of politics, sports, entertainment, and more. Experience the nation's most trusted news source - subscribe to Sirasa News today!"
    
    """
    url = f"https://sri-lanka-latest-news.p.rapidapi.com/articles/sirasa"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sri-lanka-latest-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def daily_mirror_news_sri_lanka(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Daily Mirror News Sri Lanka provides accurate, timely, and in-depth news on politics, sports, entertainment, and more. Stay informed with our SEO-friendly coverage and experience Sri Lanka's most trusted English-language newspaper. Subscribe now for the latest stories, opinions, and updates!"
    
    """
    url = f"https://sri-lanka-latest-news.p.rapidapi.com/articles/dailymirror"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sri-lanka-latest-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ft_news_sri_lanka(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "FT News Sri Lanka delivers in-depth business, finance, and economic updates tailored to your needs. Stay ahead with our SEO-friendly coverage, expert analysis, and market insights. Embrace success with the nation's most trusted news source. Subscribe to FT News Sri Lanka today!"
    
    """
    url = f"https://sri-lanka-latest-news.p.rapidapi.com/articles/ft"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sri-lanka-latest-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def the_island_news(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Immerse yourself in Sri Lanka's vibrant culture, breathtaking landscapes, and rich heritage with The Island News. Explore travel tips, local events, and lifestyle updates from the Pearl of the Indian Ocean. Subscribe to our SEO-friendly Sri Lanka edition for the latest news and insights."
    
    """
    url = f"https://sri-lanka-latest-news.p.rapidapi.com/articles/island"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sri-lanka-latest-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ada_derana_news(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Ada Derana News brings you the latest and most comprehensive coverage of Sri Lanka's top stories, politics, sports, and entertainment. Stay informed with our SEO-friendly, up-to-the-minute news updates, and experience the best of Sri Lanka from your screen. Subscribe now!"
    
    """
    url = f"https://sri-lanka-latest-news.p.rapidapi.com/articles/adaderana"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sri-lanka-latest-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

