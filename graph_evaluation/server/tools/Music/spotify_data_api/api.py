import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_futured_playlists(timestamp: str, country: str, offset: int=0, limit: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get futured playilsts by country from one of the countrys and timestamp date example 2022-10-20 :
		AD, AE, AG, AL, AM, AO, AR, AT, AU, AZ, 
		BA, BB, BD, BE, BF, BG, BH, BI, BJ, BN, 
		BO, BR, BS, BT, BW, BZ, CA, CD, CG, CH, 
		CI, CL, CM, CO, CR, CV, CW, CY, CZ, DE, 
		DJ, DK, DM, DO, DZ, EC, EE, EG, ES, ET, 
		FI, FJ, FM, FR, GA, GB, GD, GE, GH, GM, 
		GN, GQ, GR, GT, GW, GY, HK, HN, HR, HT, 
		HU, ID, IE, IL, IN, IQ, IS, IT, JM, JO, 
		JP, KE, KG, KH, KI, KM, KN, KR, KW, KZ, 
		LA, LB, LC, LI, LK, LR, LS, LT, LU, LV, 
		LY, MA, MC, MD, ME, MG, MH, MK, ML, MN, 
		MO, MR, MT, MU, MV, MW, MX, MY, MZ, NA, 
		NE, NG, NI, NL, NO, NP, NR, NZ, OM, PA, 
		PE, PG, PH, PK, PL, PS, PT, PW, PY, QA, 
		RO, RS, RW, SA, SB, SC, SE, SG, SI, SK, 
		SL, SM, SN, SR, ST, SV, SZ, TD, TG, TH, 
		TJ, TL, TN, TO, TR, TT, TV, TW, TZ, UA, 
		UG, US, UY, UZ, VC, VE, VN, VU, WS, XK,
		ZA, ZM, ZW"
    timestamp: Date of the futured playlists - String in format yyyy-mm-dd
        country: Country one from available :
AD, AE, AG, AL, AM, AO, AR, AT, AU, AZ, 
BA, BB, BD, BE, BF, BG, BH, BI, BJ, BN, 
BO, BR, BS, BT, BW, BZ, CA, CD, CG, CH, 
CI, CL, CM, CO, CR, CV, CW, CY, CZ, DE, 
DJ, DK, DM, DO, DZ, EC, EE, EG, ES, ET, 
FI, FJ, FM, FR, GA, GB, GD, GE, GH, GM, 
GN, GQ, GR, GT, GW, GY, HK, HN, HR, HT, 
HU, ID, IE, IL, IN, IQ, IS, IT, JM, JO, 
JP, KE, KG, KH, KI, KM, KN, KR, KW, KZ, 
LA, LB, LC, LI, LK, LR, LS, LT, LU, LV, 
LY, MA, MC, MD, ME, MG, MH, MK, ML, MN, 
MO, MR, MT, MU, MV, MW, MX, MY, MZ, NA, 
NE, NG, NI, NL, NO, NP, NR, NZ, OM, PA, 
PE, PG, PH, PK, PL, PS, PT, PW, PY, QA, 
RO, RS, RW, SA, SB, SC, SE, SG, SI, SK, 
SL, SM, SN, SR, ST, SV, SZ, TD, TG, TH, 
TJ, TL, TN, TO, TR, TT, TV, TW, TZ, UA, 
UG, US, UY, UZ, VC, VE, VN, VU, WS, XK,
ZA, ZM, ZW
        limit: Limit of the search results default value 20 MAX 50
        
    """
    url = f"https://spotify-data-api1.p.rapidapi.com/playlists/featured-playlists"
    querystring = {'timestamp': timestamp, 'country': country, }
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-data-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_profile_and_public_playlists(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get useer profile and public playlists by user ID."
    
    """
    url = f"https://spotify-data-api1.p.rapidapi.com/users/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-data-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_show_episodes(is_id: str, offset: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get show episodes by show ID."
    offset: offset deafult 0 
        limit: Limit of the search result default value 20 MAX 50
        
    """
    url = f"https://spotify-data-api1.p.rapidapi.com/shows/{is_id}/episodes"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-data-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_show(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get show by show ID."
    
    """
    url = f"https://spotify-data-api1.p.rapidapi.com/shows/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-data-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_episode(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get episode from episode ID."
    
    """
    url = f"https://spotify-data-api1.p.rapidapi.com/episodes/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-data-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_track_audo_futures(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get track audio futures data from track ID."
    
    """
    url = f"https://spotify-data-api1.p.rapidapi.com/audio-data/audio-features/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-data-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tracks(ids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get tracks by tracks IDs separeted by comma MAX 50 tracks at a time."
    ids: Tracks IDs separated by comma
        
    """
    url = f"https://spotify-data-api1.p.rapidapi.com/tracks"
    querystring = {'ids': ids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-data-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_playlist_by_category(type: str, limit: int=20, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get playlist from category, supported categories are :
		pop, hip-hop, rock, latino, dance/electro, mood, indie, workout, country, r&b, k-pop,
		chill, sleep, party, decades, love, metal, jazz, gaming, folk&acoustic, focus, classics,
		punck, ambient, blues, afro, summer"
    type: pop, hip-hop, rock, latino, dance/electro, mood, indie, workout, country, r&b, k-pop,
chill, sleep, party, decades, love, metal, jazz, gaming, folk&acoustic, focus, classics,
punck, ambient, blues, afro, summer
        limit: MAX value 50
        
    """
    url = f"https://spotify-data-api1.p.rapidapi.com/playlists/categories-playlists"
    querystring = {'type': type, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-data-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_playlist_tracks(is_id: str, limit: int=20, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get playlist tracks by playlist ID."
    limit: Optional MAX value 50 default 20
        offset: Optional deafult 0 
        
    """
    url = f"https://spotify-data-api1.p.rapidapi.com/playlists/{is_id}/tracks"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-data-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_playlist(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get playlist by ID."
    
    """
    url = f"https://spotify-data-api1.p.rapidapi.com/playlists/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-data-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_new_releases(country: str, limit: int=20, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get new releases albums from one of the countrys :
		AD, AE, AG, AL, AM, AO, AR, AT, AU, AZ, 
		BA, BB, BD, BE, BF, BG, BH, BI, BJ, BN, 
		BO, BR, BS, BT, BW, BZ, CA, CD, CG, CH, 
		CI, CL, CM, CO, CR, CV, CW, CY, CZ, DE, 
		DJ, DK, DM, DO, DZ, EC, EE, EG, ES, ET, 
		FI, FJ, FM, FR, GA, GB, GD, GE, GH, GM, 
		GN, GQ, GR, GT, GW, GY, HK, HN, HR, HT, 
		HU, ID, IE, IL, IN, IQ, IS, IT, JM, JO, 
		JP, KE, KG, KH, KI, KM, KN, KR, KW, KZ, 
		LA, LB, LC, LI, LK, LR, LS, LT, LU, LV, 
		LY, MA, MC, MD, ME, MG, MH, MK, ML, MN, 
		MO, MR, MT, MU, MV, MW, MX, MY, MZ, NA, 
		NE, NG, NI, NL, NO, NP, NR, NZ, OM, PA, 
		PE, PG, PH, PK, PL, PS, PT, PW, PY, QA, 
		RO, RS, RW, SA, SB, SC, SE, SG, SI, SK, 
		SL, SM, SN, SR, ST, SV, SZ, TD, TG, TH, 
		TJ, TL, TN, TO, TR, TT, TV, TW, TZ, UA, 
		UG, US, UY, UZ, VC, VE, VN, VU, WS, XK,
		ZA, ZM, ZW"
    country: One of: AD, AE, AG, AL, AM, AO, AR, AT, AU, AZ, 
BA, BB, BD, BE, BF, BG, BH, BI, BJ, BN, 
BO, BR, BS, BT, BW, BZ, CA, CD, CG, CH, 
CI, CL, CM, CO, CR, CV, CW, CY, CZ, DE, 
DJ, DK, DM, DO, DZ, EC, EE, EG, ES, ET, 
FI, FJ, FM, FR, GA, GB, GD, GE, GH, GM, 
GN, GQ, GR, GT, GW, GY, HK, HN, HR, HT, 
HU, ID, IE, IL, IN, IQ, IS, IT, JM, JO, 
JP, KE, KG, KH, KI, KM, KN, KR, KW, KZ, 
LA, LB, LC, LI, LK, LR, LS, LT, LU, LV, 
LY, MA, MC, MD, ME, MG, MH, MK, ML, MN, 
MO, MR, MT, MU, MV, MW, MX, MY, MZ, NA, 
NE, NG, NI, NL, NO, NP, NR, NZ, OM, PA, 
PE, PG, PH, PK, PL, PS, PT, PW, PY, QA, 
RO, RS, RW, SA, SB, SC, SE, SG, SI, SK, 
SL, SM, SN, SR, ST, SV, SZ, TD, TG, TH, 
TJ, TL, TN, TO, TR, TT, TV, TW, TZ, UA, 
UG, US, UY, UZ, VC, VE, VN, VU, WS, XK,
ZA, ZM, ZW
        limit: Optional default value 20 MAX value 50.
        offset: Optional default value 0.
        
    """
    url = f"https://spotify-data-api1.p.rapidapi.com/albums/browse/new-releases"
    querystring = {'country': country, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-data-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_album_tracks(is_id: str, limit: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get album tracks by album ID"
    id: album id
        limit: Number of results in range 0-50
        
    """
    url = f"https://spotify-data-api1.p.rapidapi.com/albums/{is_id}/tracks"
    querystring = {'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-data-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_album(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get album by album ID."
    id: album id
        
    """
    url = f"https://spotify-data-api1.p.rapidapi.com/albums/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-data-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_artist_related_artists(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get artist related artist by artis id"
    
    """
    url = f"https://spotify-data-api1.p.rapidapi.com/artists/{is_id}/related-artists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-data-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_artist_top_tracks(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get artist top tracks by artist id"
    
    """
    url = f"https://spotify-data-api1.p.rapidapi.com/artists/{is_id}/top-tracks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-data-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_artist_albums(is_id: str, group: str, limit: int=20, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get artist albums request from one of the groups : album, single, appears_on, compilation"
    id: ID for the artist
        group: One of : album, single, appears_on, compilation
        limit: Limit of the results Optional default 20  MAX 50
        offset: Offset of the results default 0 
        
    """
    url = f"https://spotify-data-api1.p.rapidapi.com/artists/{is_id}/albums"
    querystring = {'group': group, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-data-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_multiple_artists(ids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get multiple artist with one request max 5 at once. Each id separeted by ,"
    ids: Artists IDs separated by comma MAX 5
        
    """
    url = f"https://spotify-data-api1.p.rapidapi.com/artists"
    querystring = {'ids': ids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-data-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_artist(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get artist by ID"
    
    """
    url = f"https://spotify-data-api1.p.rapidapi.com/artists/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-data-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(query: str, limitsearchresults: int, types: str, offset: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for a one of the types "album", "artist", "playlist", "track", "show", "episode""
    query: Query of the search example Pop rock
        limitsearchresults: Number of results per search MAX value 50
        types: Type of the search results one of:  album, artist, playlist, track, show,,episode
        offset: Start of the search point MAX value 1000
        
    """
    url = f"https://spotify-data-api1.p.rapidapi.com/search"
    querystring = {'query': query, 'limitSearchResults': limitsearchresults, 'types': types, 'offset': offset, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-data-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

