import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def songs_get_details(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get details about song"
    id: The value of 'key' field returned in …/search endpoint

        
    """
    url = f"https://shazam-core7.p.rapidapi.com/songs/get_details"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam-core7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for songs, artists that match input term"
    term: Full name of songs or artists
        
    """
    url = f"https://shazam-core7.p.rapidapi.com/search"
    querystring = {'term': term, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam-core7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def songs_list_recommendations(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Similar songs based on song id"
    id: The value of 'key' field returned in …/search endpoint

        
    """
    url = f"https://shazam-core7.p.rapidapi.com/songs/list-recommendations"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam-core7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def artist_get_top_songs(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the top songs by artist according to Shazam"
    id: The value of 'artist -> adam_id' field returned in …/search endpoint

        
    """
    url = f"https://shazam-core7.p.rapidapi.com/artist/get-top-songs"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam-core7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def artist_get_details(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieving information from an artist profile"
    id: The value of 'artist -> adam_id' field returned in …/search endpoint

        
    """
    url = f"https://shazam-core7.p.rapidapi.com/artist/get-details"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam-core7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts_get_top_songs_in_city(country_code: str, city_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return top songs in city"
    country_code: DZ,BY,CI,SN,TN,AU,AT,AZ,AR,BE,BG,BR,GB,HU,VE,VN,GH,DE,GR,DK,EG,ZM,IL,IN,ID,IE,ES,IT,KZ,CM,CA,KE,CN,CO,CR,MY,MA,MX,MZ,NG,NL,NZ,NO,AE,PE,PL,PT,RU,RO,SA,SG,US,TH,TZ,TR,UG,UZ,UA,UY,PH,FI,FR,HR,CZ,CL,CH,SE,ZA,KR,JP
        city_name: use city name from below 
https://github.com/dotX12/dotX12/blob/main/city.json
        
    """
    url = f"https://shazam-core7.p.rapidapi.com/charts/get-top-songs-in-city"
    querystring = {'country_code': country_code, 'city_name': city_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam-core7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts_get_top_songs_in_country_by_genre(country_code: str, genre: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get top songs in world by genre"
    country_code: DZ,BY,CI,SN,TN,AU,AT,AZ,AR,BE,BG,BR,GB,HU,VE,VN,GH,DE,GR,DK,EG,ZM,IL,IN,ID,IE,ES,IT,KZ,CM,CA,KE,CN,CO,CR,MY,MA,MX,MZ,NG,NL,NZ,NO,AE,PE,PL,PT,RU,RO,SA,SG,US,TH,TZ,TR,UG,UZ,UA,UY,PH,FI,FR,HR,CZ,CL,CH,SE,ZA,KR,JP
        genre: POP,HIP_HOP_RAP,DANCE,ELECTRONIC,SOUL_RNB,ALTERNATIVE,ROCK,LATIN,FILM_TV,COUNTRY,AFRO_BEATS,WORLDWIDE,REGGAE_DANCE_HALL,HOUSE,K_POP,FRENCH_POP,SINGER_SONGWRITER,REG_MEXICO
        
    """
    url = f"https://shazam-core7.p.rapidapi.com/charts/get-top-songs-in_country_by_genre"
    querystring = {'country_code': country_code, 'genre': genre, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam-core7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def songs_get_track_listening_count(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get track listening count"
    id: The value of 'key' field returned in …/search endpoint

        
    """
    url = f"https://shazam-core7.p.rapidapi.com/songs/get-track-listening-count"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam-core7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts_get_top_songs_in_world(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get top songs in world"
    
    """
    url = f"https://shazam-core7.p.rapidapi.com/charts/get-top-songs-in-world"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam-core7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts_get_top_songs_in_world_by_genre(genre: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return top songs in world by genre"
    genre: POP,HIP_HOP_RAP,DANCE,ELECTRONIC,SOUL_RNB,ALTERNATIVE,ROCK,LATIN,FILM_TV,COUNTRY,AFRO_BEATS,WORLDWIDE,REGGAE_DANCE_HALL,HOUSE,K_POP,FRENCH_POP,SINGER_SONGWRITER,REG_MEXICO
        
    """
    url = f"https://shazam-core7.p.rapidapi.com/charts/get-top-songs-in_world_by_genre"
    querystring = {'genre': genre, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam-core7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts_get_top_songs_in_country(country_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get top songs in country"
    country_code: DZ,BY,CI,SN,TN,AU,AT,AZ,AR,BE,BG,BR,GB,HU,VE,VN,GH,DE,GR,DK,EG,ZM,IL,IN,ID,IE,ES,IT,KZ,CM,CA,KE,CN,CO,CR,MY,MA,MX,MZ,NG,NL,NZ,NO,AE,PE,PL,PT,RU,RO,SA,SG,US,TH,TZ,TR,UG,UZ,UA,UY,PH,FI,FR,HR,CZ,CL,CH,SE,ZA,KR,JP
        
    """
    url = f"https://shazam-core7.p.rapidapi.com/charts/get-top-songs-in-country"
    querystring = {'country_code': country_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam-core7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

