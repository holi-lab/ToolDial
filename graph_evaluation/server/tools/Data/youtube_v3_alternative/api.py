import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def video_details(is_id: str, x_cachebypass: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details of the YouTube video."
    id: Video id
        x_cachebypass: Cache bypass, value must be passed 1
Quota Cost +1
        
    """
    url = f"https://youtube-v3-alternative.p.rapidapi.com/video"
    querystring = {'id': is_id, }
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v3-alternative.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def comments(is_id: str, x_cachebypass: str=None, sort_by: str=None, token: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Comments of the particular video"
    id: Video Id
        x_cachebypass: Cache bypass, value must be passed 1
Quota Cost +1
        sort_by: Available options:
**newest**
**top**  [default]
        token: Pagination token
        
    """
    url = f"https://youtube-v3-alternative.p.rapidapi.com/comments"
    querystring = {'id': is_id, }
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if sort_by:
        querystring['sort_by'] = sort_by
    if token:
        querystring['token'] = token
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v3-alternative.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playlist(is_id: str, x_cachebypass: str=None, token: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get playlist details and videos."
    id: Playlist Id
        x_cachebypass: Cache bypass, value must be passed 1
Quota Cost +1
        token: Pagination Token
        
    """
    url = f"https://youtube-v3-alternative.p.rapidapi.com/playlist"
    querystring = {'id': is_id, }
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if token:
        querystring['token'] = token
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v3-alternative.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(query: str, x_cachebypass: str=None, duration: str=None, lang: str='en', type: str=None, features: str=None, geo: str='US', sort_by: str=None, token: str=None, upload_date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search videos, playlist, channel."
    query: Search term
        x_cachebypass: Cache bypass, value must be passed 1
Quota Cost +1
        duration: Duration filter options:
**short**  - less than 4 min
**medium**  - 4 to 20 min
**long**  - more than 20 min
        lang: Locale/language for request. Like en, gb, hi, etc
        type: Search type filter options:
**video**
**channel**
**playlist**
**movie**
**show**
        features: Video Features options:
**HD**
**subtitles**
**CCommons**
**3D**
**Live**
**Purchased**
**4K**
**360**
**Location**
**HDR**
**VR180**

Multiple features could be joined by ','
For example: HD,subtitles
        geo: ISO 3166-2 country code of the region. Like US (default), UK, CA, IN, etc.
        sort_by: Results Sort options:
**relevance**  [default]
**rating**
**date**
**views**
        token: Pagination Token
        upload_date: Upload Date filter options:
**hour**
**today**
**week**
**month**
**year**
        
    """
    url = f"https://youtube-v3-alternative.p.rapidapi.com/search"
    querystring = {'query': query, }
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if duration:
        querystring['duration'] = duration
    if lang:
        querystring['lang'] = lang
    if type:
        querystring['type'] = type
    if features:
        querystring['features'] = features
    if geo:
        querystring['geo'] = geo
    if sort_by:
        querystring['sort_by'] = sort_by
    if token:
        querystring['token'] = token
    if upload_date:
        querystring['upload_date'] = upload_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v3-alternative.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel(is_id: str, x_cachebypass: str=None, token: str=None, sort_by: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Channel details and videos"
    id: Channel Id
        x_cachebypass: Cache bypass, value must be passed 1
Quota Cost +1
        token: Pagination token
        sort_by: Sorts the channel videos. Available options: 
**newest**  [default]
**oldest**
**popular**
        
    """
    url = f"https://youtube-v3-alternative.p.rapidapi.com/channel"
    querystring = {'id': is_id, }
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if token:
        querystring['token'] = token
    if sort_by:
        querystring['sort_by'] = sort_by
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v3-alternative.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending(geo: str, x_cachebypass: str=None, lang: str='en', type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get trending videos of the particular geo"
    geo: ISO 3166-2 country code of the region for which you want the trending data. Like US (default), UK, CA, IN, etc.
        x_cachebypass: Cache bypass, value must be passed 1
Quota Cost +1
        lang: Locale/language for request. Like en, gb, hi, etc
        type: Trending type:
**now**
**music**
**games**
**movies**

Default is **now**
        
    """
    url = f"https://youtube-v3-alternative.p.rapidapi.com/trending"
    querystring = {'geo': geo, }
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if lang:
        querystring['lang'] = lang
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v3-alternative.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def related_videos(is_id: str, x_cachebypass: str=None, geo: str='US', lang: str='en', token: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the videos related to the provided video."
    id: Video id
        x_cachebypass: Cache bypass, value must be passed 1
Quota Cost +1
        geo: ISO 3166-2 country code of the region. Like US (default), UK, CA, IN, etc.
        lang: Locale/language for request. Like en, gb, hi, etc
        token: Pagination token
        
    """
    url = f"https://youtube-v3-alternative.p.rapidapi.com/related"
    querystring = {'id': is_id, }
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if geo:
        querystring['geo'] = geo
    if lang:
        querystring['lang'] = lang
    if token:
        querystring['token'] = token
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v3-alternative.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

