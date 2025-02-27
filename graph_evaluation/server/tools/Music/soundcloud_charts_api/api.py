import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def trending_and_top_charts(genre: str, kind: str, limit: str='50', region: str='soundcloud:regions:US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get trending (New & Hot) and top charts from SoundCloud"
    genre: all-music, ambient, classical, hiphoprap, alternativerock, country, danceedm, dancehall, deephouse, disco, drumbass, dubstep, electronic, folksingersongwriter, hiphoprap, house, indie, jazzblues, latin, metal, piano, pop, reggae, reggaeton, rock, soundtrack, techno, trance, trap, triphop, world .
All in this format: soundcloud:genres:disco
        kind: 'trending' or 'top'
        region: ISO2 Country codes like US, DE, GB. Use in this format:
soundcloud:regions:DE
        
    """
    url = f"https://soundcloud-charts-api.p.rapidapi.com/charts"
    querystring = {'genre': genre, 'kind': kind, }
    if limit:
        querystring['limit'] = limit
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soundcloud-charts-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

