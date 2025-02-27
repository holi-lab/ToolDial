import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(query: str, hosts: str=None, duration: str=None, sort: str=None, size: str=None, quality: str=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search videos."
    hosts: Filter videos by file-hosts.
The parameter is a comma-separated list of file-hosts codes:
- **DF** (depositfiles)
- **RG** (rapidgator)
- **KS** (keep2share)
- **FB** (fileboom)
- **FJ** (filejoker)
        duration: Filter videos by duration.
The parameter is a range of minimum/maximum values (in seconds).
For example:
- **300..600** will select videos between 300 and 600 seconds
- **300..** will select videos longer than 300 seconds
- **..600** will select videos shorter than 600 seconds
        sort: Sort videos by size/duration/quality.
For example:
- **size** will sort videos by size (ascending)
- **-quality** will sort videos by quality (descending)
- **-quality,duration** will sort videos by quality (descending) then by duration (ascending)
        size: Filter video by size.
The parameter is a range of minimum/maximum values (in MB).
For example:
- **500..1000** will select videos between 500 and 1000 MB
- **500..** will select videos bigger than 500 MB
- **..1000** will select videos smaller than 1000 MB
        quality: Filter video  by quality.
The parameter is either a single value, or a range of minimum/maximum values (in pixels). 
For example:
- **480** will select 480p videos
- **720..1080** will select videos between 720p and 1080p
- **720..** will select 720p videos (or more)
- **..1080** will select 1080p videos (or less)
        page: Page number.
        
    """
    url = f"https://direct-porn.p.rapidapi.com/Search"
    querystring = {'query': query, }
    if hosts:
        querystring['hosts'] = hosts
    if duration:
        querystring['duration'] = duration
    if sort:
        querystring['sort'] = sort
    if size:
        querystring['size'] = size
    if quality:
        querystring['quality'] = quality
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "direct-porn.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

