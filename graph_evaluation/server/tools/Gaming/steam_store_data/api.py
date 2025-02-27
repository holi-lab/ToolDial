import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def featured_games(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns lists of featured games on Windows and macOS"
    
    """
    url = f"https://steam-store-data.p.rapidapi.com/api/featured/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam-store-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def livestream_info(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of current popular Steam game livestreams"
    
    """
    url = f"https://steam-store-data.p.rapidapi.com/broadcast/ajaxgetpopularpartnerbroadcasts/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam-store-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getplayerachievementsversion1(l: str=None, key: str='XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', steamid: str='76561197972495328', appid: str='440', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetPlayerAchievements (version 1)"
    l: Optional. If possible, returns data in the specified language. Use ISO 639 code to specify the language
        key: Steam Community API key
        steamid: 64-bit SteamID of the user you want achievements for
        appid: AppID of the game you want achievements for
        
    """
    url = f"https://steam-store-data.p.rapidapi.com/ISteamUserStats/GetPlayerAchievements/v0001/"
    querystring = {}
    if l:
        querystring['l'] = l
    if key:
        querystring['key'] = key
    if steamid:
        querystring['steamid'] = steamid
    if appid:
        querystring['appid'] = appid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam-store-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def featured_categories(extra: str=None, trailer: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns lists of games separated by categories: Specials, Coming Soon, Top Sellers, New Releases
		
		It is not known what effect the two optional parameters have"
    extra: Unknown effect (not documented)
        trailer: Unknown effect (not documented)
        
    """
    url = f"https://steam-store-data.p.rapidapi.com/api/featuredcategories/"
    querystring = {}
    if extra:
        querystring['extra'] = extra
    if trailer:
        querystring['trailer'] = trailer
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam-store-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def package_details(packageids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns data on Steam Packages (generally game bundles etc.)
		packageids - Comma-separated list of Package IDs you want data for"
    packageids: Comma-separated list of Package IDs you want data for
        
    """
    url = f"https://steam-store-data.p.rapidapi.com/api/packagedetails/"
    querystring = {'packageids': packageids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam-store-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def app_details(appids: str, filters: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns data for given AppID(s). This data is equivalent to what you would see on the games official Steam store page.
		
		appids - Comma-separated list of AppIDs you want data for
		filters - Filter the keys that get returned. If you list multiple AppIDs for "appids", this MUST be set to "price_overview". "basic" will return a smaller subset of keys"
    appids: Comma-separated list of AppIDs you want data for
        filters: Filter the keys that get returned. If you list multiple AppIDs for \\\\\\\"appids\\\\\\\", this MUST be set to \\\\\\\"price_overview\\\\\\\". \\\\\\\"basic\\\\\\\" will return a smaller subset of keys
        
    """
    url = f"https://steam-store-data.p.rapidapi.com/api/appdetails/"
    querystring = {'appids': appids, }
    if filters:
        querystring['filters'] = filters
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam-store-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getplayersummariesversion2(steamids: str='76561197960435530', key: str='XXXXXXXXXXXXXXXXXXXXXXX', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetPlayerSummaries (version 2)"
    steamids: List of the 64-bit SteamIDs you want profile information for. Separate using commas
        key: Steam Community API key
        
    """
    url = f"https://steam-store-data.p.rapidapi.com/ISteamUser/GetPlayerSummaries/v0002/"
    querystring = {}
    if steamids:
        querystring['steamids'] = steamids
    if key:
        querystring['key'] = key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam-store-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getnewsforappversion2(format: str='json', appid: str='440', count: str='3', maxlength: str='300', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetNewsForApp (version 2)"
    format: The format of the data: JSON (default), XML, VDF
        appid: The AppID of the game you want news for
        count: The number of news articles to return
        maxlength: The maximum length of each news article
        
    """
    url = f"https://steam-store-data.p.rapidapi.com/ISteamNews/GetNewsForApp/v0002/"
    querystring = {}
    if format:
        querystring['format'] = format
    if appid:
        querystring['appid'] = appid
    if count:
        querystring['count'] = count
    if maxlength:
        querystring['maxlength'] = maxlength
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam-store-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getglobalachievementpercentagesforappversion2(format: str='json', gameid: str='440', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetGlobalAchievementPercentagesForApp (version 2)"
    format: The format of the data: JSON (default), XML, VDF
        gameid: The AppID of the game you want data for
        
    """
    url = f"https://steam-store-data.p.rapidapi.com/ISteamUserStats/GetGlobalAchievementPercentagesForApp/v0002/"
    querystring = {}
    if format:
        querystring['format'] = format
    if gameid:
        querystring['gameid'] = gameid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam-store-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getownedgamesversion1(appids_filter: str=None, include_played_free_games: str='true', key: str='XXXXXXXXXXXXXXXXX', include_appinfo: str='true', format: str='json', steamid: str='76561197960434622', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetOwnedGames (version 1)"
    appids_filter: Filter output to a list of AppIDs. Must be defined in a JSON list (e.g.  "appids_filter: [ 440, 500, 550 ]", including quotes)
        include_played_free_games: Set true if you want free games the player has played to be included. These games are owned by all users and exluded by default.
        key: Steam Community API key
        include_appinfo: Set true to return game name and icon. Returns only AppID by default
        format: The format of the data: JSON (default), XML, VDF
        steamid: 64-bit SteamID of the user you want to get the games of
        
    """
    url = f"https://steam-store-data.p.rapidapi.com/IPlayerService/GetOwnedGames/v0001/"
    querystring = {}
    if appids_filter:
        querystring['appids_filter'] = appids_filter
    if include_played_free_games:
        querystring['include_played_free_games'] = include_played_free_games
    if key:
        querystring['key'] = key
    if include_appinfo:
        querystring['include_appinfo'] = include_appinfo
    if format:
        querystring['format'] = format
    if steamid:
        querystring['steamid'] = steamid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam-store-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrecentlyplayedgamesversion1(key: str='XXXXXXXXXXXXXXXXX', format: str='json', steamid: str='76561197960434622', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetRecentlyPlayedGames (version 1)"
    key: Steam Community API key
        format: The format of the data: JSON (default), XML, VDF

        steamid: 64-bit SteamID of the account you want games for
        
    """
    url = f"https://steam-store-data.p.rapidapi.com/IPlayerService/GetRecentlyPlayedGames/v0001/"
    querystring = {}
    if key:
        querystring['key'] = key
    if format:
        querystring['format'] = format
    if steamid:
        querystring['steamid'] = steamid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam-store-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getuserstatsforgameversion2(steamid: str='76561197972495328', appid: str='440', l: str=None, key: str='XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetUserStatsForGame (version 2)"
    steamid: 64-bit SteamID of the user you want achievements for
        appid: AppID of the game you want achievements for 
        l: Optional. If possible, returns data in the specified language. Use ISO 639 code to specify the language
        key: Steam Community API key
        
    """
    url = f"https://steam-store-data.p.rapidapi.com/ISteamUserStats/GetUserStatsForGame/v0002/"
    querystring = {}
    if steamid:
        querystring['steamid'] = steamid
    if appid:
        querystring['appid'] = appid
    if l:
        querystring['l'] = l
    if key:
        querystring['key'] = key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam-store-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfriendlistversion1(steamid: str='76561197960435530', key: str='XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', relationship: str='friend', l: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetFriendList (version 1)"
    steamid: 64-bit SteamID of a friend
        key: Steam Community API key
        relationship: Relationship filter. Possible values: friend, all
        l: Optional. If possible, returns data in the specified language. Use ISO 639 code to specify the language
        
    """
    url = f"https://steam-store-data.p.rapidapi.com/ISteamUser/GetFriendList/v0001/"
    querystring = {}
    if steamid:
        querystring['steamid'] = steamid
    if key:
        querystring['key'] = key
    if relationship:
        querystring['relationship'] = relationship
    if l:
        querystring['l'] = l
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam-store-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

