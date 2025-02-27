import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def matches_get_commentaries(matchid: int, tms: int=None, iid: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match commentaries"
    matchid: The value of matchId field returned in …/matches/list, …/schedules/list, …/series/get-matches, …/teams/get-schedules, …/teams/get-results, …/venues/get-matches endpoints.
        tms: For paging purpose, leave empty to load the first page, or an Epoch timestamp value in milliseconds (Ex : 1640883600000) to load the next page. You are interested in the 'timestamp' field returned right in this endpoint.
        iid: innings Id (Ex : 1)
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/mcenter/v1/{matchid}/comm"
    querystring = {}
    if tms:
        querystring['tms'] = tms
    if iid:
        querystring['iid'] = iid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_scorecard_v2(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match scorecard"
    matchid: The value of matchId field returned in …/matches/list, …/schedules/list, …/series/get-matches, …/teams/get-schedules, …/teams/get-results, …/venues/get-matches endpoints.
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/mcenter/v1/{matchid}/hscard"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_scorecard(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match scorecard"
    matchid: The value of matchId field returned in …/matches/list, …/schedules/list, …/series/get-matches, …/teams/get-schedules, …/teams/get-results, …/venues/get-matches endpoints.
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/mcenter/v1/{matchid}/scard"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_overs(matchid: int, iid: int=None, tms: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match overall"
    matchid: The value of matchId field returned in …/matches/list, …/schedules/list, …/series/get-matches, …/teams/get-schedules, …/teams/get-results, …/venues/get-matches endpoints.
        iid: innings Id (Ex : 1)
        tms: For paging purpose, leave empty to load the first page, or an Epoch timestamp value in milliseconds (Ex : 1640883600000) to load the next page. You are interested in the 'timestamp' field returned right in this endpoint.
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/mcenter/v1/{matchid}/overs"
    querystring = {}
    if iid:
        querystring['iid'] = iid
    if tms:
        querystring['tms'] = tms
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_info(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match info"
    matchid: The value of matchId field returned in …/matches/list, …/schedules/list, …/series/get-matches, …/teams/get-schedules, …/teams/get-results, …/venues/get-matches endpoints.
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/mcenter/v1/{matchid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_team(teamid: int, matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get players attended the match"
    teamid: The value of teamId returned in …/matches/get-info endpoint
        matchid: The value of matchId field returned in …/matches/list, …/schedules/list, …/series/get-matches, …/teams/get-schedules, …/teams/get-results, …/venues/get-matches endpoints.
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/mcenter/v1/{matchid}/team/{teamid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_commentaries_v2(matchid: int, iid: int=None, tms: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match commentaries"
    matchid: The value of matchId field returned in …/matches/list, …/schedules/list, …/series/get-matches, …/teams/get-schedules, …/teams/get-results, …/venues/get-matches endpoints.
        iid: innings Id (Ex : 1)
        tms: For paging purpose, leave empty to load the first page, or an Epoch timestamp value in milliseconds (Ex : 1640883600000) to load the next page. You are interested in the 'timestamp' field returned right in this endpoint.
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/mcenter/v1/{matchid}/hcomm"
    querystring = {}
    if iid:
        querystring['iid'] = iid
    if tms:
        querystring['tms'] = tms
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stats_get_icc_rankings(category: str, formattype: str, iswomen: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get ICC rankings"
    category: 
One of the followings : `batsmen`|`bowlers`|`allrounders`|`teams`
        formattype: One of the followings : test|odi|t20 (if isWomen parameter is 1, there will be no odi)
        iswomen: Set to 1 to get rankings for women
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/stats/v1/rankings/{category}"
    querystring = {'formatType': formattype, }
    if iswomen:
        querystring['isWomen'] = iswomen
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stats_get_records(statstype: str, year: int=None, opponent: int=None, team: int=None, matchtype: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get records"
    statstype: The value of 'value' field returned in …/stats/get-record-filters endpoint
        year: 
Specify year to get records. Ex : 2021
        opponent: The value of teamId field returned right in this endpoint
        team: The value of teamId field returned right in this endpoint
        matchtype: The value of matchTypeId field returned right in this endpoint
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/stats/v1/topstats/0"
    querystring = {'statsType': statstype, }
    if year:
        querystring['year'] = year
    if opponent:
        querystring['opponent'] = opponent
    if team:
        querystring['team'] = team
    if matchtype:
        querystring['matchType'] = matchtype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stats_get_icc_standings(matchtype: int, seasonid: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get ICC standings"
    matchtype: One of the followings : `1`-World test championship|`2`-World cup super league
        seasonid: The value of seasonStandings/id field returned right in this endpoint
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/stats/v1/iccstanding/team/matchtype/{matchtype}"
    querystring = {}
    if seasonid:
        querystring['seasonId'] = seasonid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stats_get_record_filters(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get record filters"
    
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/stats/v1/topstats"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_list_by_category(categoryid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List latest news by category"
    categoryid: Filter news by category, the value of id field returned in …/news/get-categories
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/news/v1/cat/{categoryid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_list_by_topic(topicid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List latest news by topic"
    topicid: Filter news by topic, the value of id field returned in …/news/get-topics
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/news/v1/topics/{topicid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_get_career(playerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player career"
    playerid: The value of id field returned in …/players/list-trending, …/players/search endpoints
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/stats/v1/player/{playerid}/career"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_get_news(playerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news by player"
    playerid: The value of id field returned in …/players/list-trending, …/players/search endpoints
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/news/v1/player/{playerid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def venues_get_stats(venueid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stats by venue"
    venueid: The value of id field returned in …/series/get-venues endpoint
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/stats/v1/venue/{venueid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def venues_get_info(venueid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get venue info"
    venueid: The value of id field returned in …/series/get-venues endpoint
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/venues/v1/{venueid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_get_schedules(teamid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get scheduled matches for a team"
    teamid: The value of teamId field returned in …/teams/list endpoint
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/teams/v1/{teamid}/schedule"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_get_players(teamid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get players by team"
    teamid: The value of teamId field returned in …/teams/list endpoint
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/teams/v1/{teamid}/players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_get_results(teamid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get matched results by team"
    teamid: The value of teamId field returned in …/teams/list endpoint
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/teams/v1/{teamid}/results"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_get_stats_filters(teamid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get supported filters"
    teamid: The value of teamId field returned in …/teams/list endpoint
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/stats/v1/team/{teamid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def series_get_matches(seriesid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get recent and upcoming matches by series"
    seriesid: The value of id field returned in …/series/list or …/series/list-archives endpoints.
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/series/v1/{seriesid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def series_get_squads(seriesid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get squads by series"
    seriesid: The value of id field returned in …/series/list or …/series/list-archives endpoints.
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/series/v1/{seriesid}/squads"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def series_get_points_table(seriesid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get points table by series"
    seriesid: 
The value of id field returned in …/series/list or …/series/list-archives endpoints.
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/stats/v1/series/{seriesid}/points-table"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def series_get_stats_filters(seriesid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get supported filters"
    seriesid: The value of id field returned in …/series/list or …/series/list-archives endpoints.
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/stats/v1/series/{seriesid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_image(imageid: str, d: str=None, p: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to get image by id"
    imageid: Add 'c' in the front of imageId
        d: `high`|`low`
        p: Specify size of image. One of the following : `de`|`det`|`gthumb`|`thumb`
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/img/v1/i1/{imageid}/i.jpg"
    querystring = {}
    if d:
        querystring['d'] = d
    if p:
        querystring['p'] = p
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def photos_get_gallery(galleryid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get photo gallery"
    galleryid: galleryId from photos/list service
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/photos/v1/detail/{galleryid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def photos_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List photo galleries"
    
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/photos/v1/index"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_get_topics(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available topics"
    
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/news/v1/topics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_get_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available categories"
    
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/news/v1/cat"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_detail(newsid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news detail"
    newsid: The value of story/id field returned in …/news/list …/news/list-by-category …/news/list-by-topic endpoint
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/news/v1/detail/{newsid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_list(type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List latest news"
    type: One of the followings : `index`|`premiumIndex`
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/news/v1/{type}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_get_bowling(playerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player's bowling"
    playerid: The value of id field returned in …/players/list-trending, …/players/search endpoints
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/stats/v1/player/{playerid}/bowling"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_get_batting(playerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player's batting"
    playerid: The value of id field returned in …/players/list-trending, …/players/search endpoints
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/stats/v1/player/{playerid}/batting"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_get_info(playerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player info"
    playerid: The value of id field returned in …/players/list-trending, …/players/search endpoints
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/stats/v1/player/{playerid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_search(plrn: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search player by name"
    
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/stats/v1/player/search"
    querystring = {'plrN': plrn, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_list_trending(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List trending players"
    
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/stats/v1/player/trending"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_get_stats(statstype: str, teamid: int, opponent: int=None, team: int=None, year: int=None, matchtype: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stats by team"
    statstype: The value of 'value' field returned in …/teams/get-stats-filter endpoint
        teamid: The value of teamId field returned in …/teams/list endpoint
        opponent: The value of 'teamId' field returned right in this endpoint
        team: The value of 'teamId' field returned right in this endpoint
        year: Specify year to get stats. Ex : 2021
        matchtype: The value of matchTypeId field returned right in this endpoint
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/stats/v1/team/{teamid}"
    querystring = {'statsType': statstype, }
    if opponent:
        querystring['opponent'] = opponent
    if team:
        querystring['team'] = team
    if year:
        querystring['year'] = year
    if matchtype:
        querystring['matchType'] = matchtype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_get_news(teamid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news by team"
    teamid: The value of teamId field returned in …/teams/list endpoint
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/news/v1/team/{teamid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_list(type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List teams"
    type: `international`|`league`|`domestic`|`women`
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/teams/v1/{type}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def series_get_stats(seriesid: int, statstype: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stats"
    seriesid: The value of id field returned in …/series/list or …/series/list-archives endpoints.
        statstype: The value of 'value' field returned in …/series/get-stats-filter endpoint
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/stats/v1/series/{seriesid}"
    querystring = {'statsType': statstype, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def series_get_venues(seriesid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get venues by series"
    seriesid: 
The value of id field returned in …/series/list or …/series/list-archives endpoints.
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/series/v1/{seriesid}/venues"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def series_get_players(squadid: int, seriesid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get players by squad and series"
    squadid: The value of squadId field returned in …/series/get-squads endpoint
        seriesid: The value of id field returned in …/series/list or …/series/list-archives endpoints.
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/series/v1/{seriesid}/squads/{squadid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def series_get_news(seriesid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news by series"
    seriesid: The value of id field returned in …/series/list or …/series/list-archives endpoints.
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/news/v1/series/{seriesid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def series_list_archives(type: str, lastid: int=None, year: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List archived series"
    type: One of the followings : `international`|`league`|`domestic`|`women`
        lastid: For paging purpose, leave empty to load the first page, or the value of id field returned right in this endpoint.
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/series/v1/archives/{type}"
    querystring = {}
    if lastid:
        querystring['lastId'] = lastid
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def series_list(type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List series"
    type: One of the followings : `international`|`league`|`domestic`|`women`
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/series/v1/{type}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_leanback(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match leanback"
    matchid: The value of matchId field returned in …/matches/list, …/schedules/list, …/series/get-matches, …/teams/get-schedules, …/teams/get-results, …/venues/get-matches endpoints.
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/mcenter/v1/{matchid}/leanback"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def schedules_list(type: str, lasttime: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List scheduled matches"
    type: One of the followings : `international`|`league`|`domestic`|`women`
        lasttime: For paging purpose, leave empty to load the first page, or an Epoch timestamp value in milliseconds (Ex : 1640883600000) to load the next page. You are interested in the 'startDt' field returned right in this endpoint.
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/schedule/v1/{type}"
    querystring = {}
    if lasttime:
        querystring['lastTime'] = lasttime
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_list(type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List live, recent, upcoming matches"
    type: One of the followings : `live`|`recent`|`upcoming`
        
    """
    url = f"https://cricbuzz-cricket.p.rapidapi.com/matches/v1/{type}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

