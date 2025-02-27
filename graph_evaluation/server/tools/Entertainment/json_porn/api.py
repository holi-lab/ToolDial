import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def producers(count: str='5', offset: str='0', producerid: str='5924703508103168', producername: str='Playboy', includeporn: str='True', sort: str='date', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an array of porn producers or series."
    count: The number of entries you want
        offset: The number of entries you want to skip
        producerid: Will return only the entry with a matching producer key ID
        producername: Will return only the entry with a matching producer name
        includeporn: If set to true, the API will also return one porn entry for each producer
        sort: Specifies the sort of the result set. Possible values: "count", "date" and "alphabetical" (default)
        
    """
    url = f"https://steppschuh-json-porn-v1.p.rapidapi.com/producers/"
    querystring = {}
    if count:
        querystring['count'] = count
    if offset:
        querystring['offset'] = offset
    if producerid:
        querystring['producerId'] = producerid
    if producername:
        querystring['producerName'] = producername
    if includeporn:
        querystring['includePorn'] = includeporn
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steppschuh-json-porn-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(q: str, advanced: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoints to search for actors or porn entries. Returns an array of actors, porn entries and producers matching the search query."
    q: The query string
        advanced: If set to true, the query will operate a very long time but also return very good results. It will also look for actor nicknames and entry descriptions. If set to false, the query will be very fast but only find matches with String.startsWith(querry).
        
    """
    url = f"https://steppschuh-json-porn-v1.p.rapidapi.com/search/"
    querystring = {'q': q, }
    if advanced:
        querystring['advanced'] = advanced
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steppschuh-json-porn-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def images(width: str, imagekeyid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Serves images by a given imageKeyId and resizes them as requested. If the requested image width exceeds the original width, the image will be served in its original resolution."
    width: The maximum width of the image
        imagekeyid: The image key id, as provided in actor or porn entries
        
    """
    url = f"https://steppschuh-json-porn-v1.p.rapidapi.com/image/{imagekeyid}/{width}.jpg"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steppschuh-json-porn-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def porn(offset: str='0', pornid: int=5073292679446528, producerid: int=4554967436230656, porntype: int=4, includeimages: bool=None, count: str='5', actorid: int=5681034041491456, genreid: int=5245132710346752, includedownloads: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an array of the latest porn entries."
    offset: The number of entries you want to skip
        pornid: Will return only the entry with the matching key ID
        producerid: Will return only entries that were produced by the producer with the given key ID
        porntype: Will return only entries that are from the given type. Possible types are: Unknown (1), Clip (2), Photos (3) and Movies (4)
        includeimages: Pre-fetches the image entries for each returned porn entry in addition to the imageKeyIds
        count: The number of entries you want
        actorid: Will return only entries in which the actor with the given key ID is starring
        genreid: Will return only entries that contain the genre with the given key ID
        includedownloads: Pre-fetches the available download entries for each porn entry
        
    """
    url = f"https://steppschuh-json-porn-v1.p.rapidapi.com/porn/"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if pornid:
        querystring['pornId'] = pornid
    if producerid:
        querystring['producerId'] = producerid
    if porntype:
        querystring['pornType'] = porntype
    if includeimages:
        querystring['includeImages'] = includeimages
    if count:
        querystring['count'] = count
    if actorid:
        querystring['actorId'] = actorid
    if genreid:
        querystring['genreId'] = genreid
    if includedownloads:
        querystring['includeDownloads'] = includedownloads
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steppschuh-json-porn-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def actors(includeimages: str='false', count: str='5', offset: str='0', actorid: str='5865845276278784', actorname: str='Lisa', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an array of actors matching your request. If you have an actor key ID, use only the actorId parameter. If you are looking for a name, use the actorName param. Keep in mind that the actorName matches using the .startsWith() function. If you want an advanced search (including actor nick names), use the /search/ endpoint. If you don't specify a name or an id, the endpoint will return all available actors in alphabetical order."
    includeimages: Pre-fetches the image entries for each returned actor entry in addition to the imageKeyIds
        count: The number of entries you want
        offset: The number of entries you want to skip
        actorid: The actor key ID of a known actor
        actorname: The name of the actor you are looking for
        
    """
    url = f"https://steppschuh-json-porn-v1.p.rapidapi.com/actors/"
    querystring = {}
    if includeimages:
        querystring['includeImages'] = includeimages
    if count:
        querystring['count'] = count
    if offset:
        querystring['offset'] = offset
    if actorid:
        querystring['actorId'] = actorid
    if actorname:
        querystring['actorName'] = actorname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steppschuh-json-porn-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

