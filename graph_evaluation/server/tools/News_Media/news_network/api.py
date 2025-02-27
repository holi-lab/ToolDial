import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_news(category: str, region: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return list of news by selected region and category.
		Visit our [demo site](https://core-api.net/news/news.html)"
    region: ae: United Arab Emirates
ar: Argentina
at: Austria
au: Australia
be: Belgium
bg: Bulgaria
br: Brazil
ca: Canada
ch: Switzerland
cn: China 
co: Colombia
cu: Cuba
cz:	Czech Republic
de: Germany
eg: Egypt
fr: France
gb: United Kingdom
gr: Greece
hk: Hong Kong
hu: Hungray
id: Indonesia
ie: Ireland
il: Israel
in: India
it: Italy
jp: Japan
kr: South Korea
lt: Lithuania
lv: Latvia
ma: Morocco
mx: Mexico
my: Malaysia
ng: Nigeria
nl: Netherlands
no: Norway
nz: New Zealand
ph: Philippines
pl: Poland
pt: Portugal
ro: Romania
rs: Serbia
ru: Russia
sa: Saudi Arabia
se: Sweden
sg: Singapore
si: Slovenia
sk: Slovakia
th: Thailand
tr: Turkey
tw: Taiwan
ua: Ukraine
us: United States
ve: Venuzuela
za: South Africa
        
    """
    url = f"https://news-network.p.rapidapi.com/get_news"
    querystring = {'category': category, 'region': region, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "news-network.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_news(keyword: str, language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search news by given keyword.
		Visit our [demo site](https://core-api.net/news/news.html)"
    language: 
ar: arabic
de: german
en: english
es: Spanish
fr: French
he: hebrew
it: italian
nl: dutch
no: norwegian
pt: portuguese
ru: russian
se: swedish
zh: chinese
        
    """
    url = f"https://news-network.p.rapidapi.com/search_news"
    querystring = {'keyword': keyword, 'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "news-network.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def available_region_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return list of available regions."
    
    """
    url = f"https://news-network.p.rapidapi.com/get_regions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "news-network.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

