import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def series_get_seasons(seriesid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get seasons related to specific series"
    seriesid: The value of id field returned in .../series/list or .../series/list-archives endpoints.
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/series/get-seasons"
    querystring = {'seriesId': seriesid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
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
    seriesid: The value of id field returned in .../series/list or .../series/list-archives endpoints.
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/series/get-stats-filters"
    querystring = {'seriesId': seriesid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
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
    teamid: The value of teamId returned in .../matches/get-info endpoint
        matchid: The value of matchId field returned in .../matches/list, .../matches/get-schedules, .../series/get-matches, .../teams/get-schedules, .../teams/get-results, .../venues/get-matches endpoints.
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/matches/get-team"
    querystring = {'teamId': teamid, 'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
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
    matchid: The value of matchId field returned in .../matches/list, .../matches/get-schedules, .../series/get-matches, .../teams/get-schedules, .../teams/get-results, .../venues/get-matches endpoints.
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/matches/get-scorecard"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_hscorecard(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Another endpoint to get match scorecard"
    matchid: The value of matchId field returned in .../matches/list, .../matches/get-schedules, .../series/get-matches, .../teams/get-schedules, .../teams/get-results, .../venues/get-matches endpoints.
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/matches/get-hscorecard"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_highlights(matchid: int, htype: int=2, iid: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match highlights"
    matchid: The value of matchId field returned in .../matches/list, .../matches/get-schedules, .../series/get-matches, .../teams/get-schedules, .../teams/get-results, .../venues/get-matches endpoints.
        htype: One of the followings : 2-Fours|4-Sixes|8-Wickets|16-Fifties|32-Hundreds|128-Dropped catches|512-UDRS|1-Others
        iid: innings Id (Ex : 1)
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/matches/get-highlights"
    querystring = {'matchId': matchid, }
    if htype:
        querystring['htype'] = htype
    if iid:
        querystring['iid'] = iid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_overs(matchid: int, tms: int=None, iid: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match overall"
    matchid: The value of matchId field returned in .../matches/list, .../matches/get-schedules, .../series/get-matches, .../teams/get-schedules, .../teams/get-results, .../venues/get-matches endpoints.
        tms: For paging purpose, leave empty to load the first page, or an Epoch timestamp value in milliseconds (Ex : 1640883600000) to load the next page. You are interested in the 'timestamp' field returned right in this endpoint.
        iid: innings Id (Ex : 1)
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/matches/get-overs"
    querystring = {'matchId': matchid, }
    if tms:
        querystring['tms'] = tms
    if iid:
        querystring['iid'] = iid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_commentaries(matchid: int, iid: int=None, tms: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match commentaries"
    matchid: The value of matchId field returned in .../matches/list, .../matches/get-schedules, .../series/get-matches, .../teams/get-schedules, .../teams/get-results, .../venues/get-matches endpoints.
        iid: innings Id (Ex : 1)
        tms: For paging purpose, leave empty to load the first page, or an Epoch timestamp value in milliseconds (Ex : 1640883600000) to load the next page. You are interested in the 'timestamp' field returned right in this endpoint.
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/matches/get-commentaries"
    querystring = {'matchId': matchid, }
    if iid:
        querystring['iid'] = iid
    if tms:
        querystring['tms'] = tms
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
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
    matchid: The value of matchId field returned in .../matches/list, .../matches/get-schedules, .../series/get-matches, .../teams/get-schedules, .../teams/get-results, .../venues/get-matches endpoints.
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/matches/get-info"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_schedules(matchtype: str, lasttime: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get scheduled matches"
    matchtype: One of the followings : international|league|domestic|women
        lasttime: For paging purpose, leave empty to load the first page, or an Epoch timestamp value in milliseconds (Ex : 1640883600000) to load the next page. You are interested in the 'startDt' field returned right in this endpoint.
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/matches/get-schedules"
    querystring = {'matchType': matchtype, }
    if lasttime:
        querystring['lastTime'] = lasttime
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_list(matchstate: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List live, recent, upcoming matches"
    matchstate: One of the followings : live|recent|upcoming
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/matches/list"
    querystring = {'matchState': matchstate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_hcommentaries(matchid: int, iid: int=None, tms: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Another endpoint used to get match commentaries"
    matchid: The value of matchId field returned in .../matches/list, .../matches/get-schedules, .../series/get-matches, .../teams/get-schedules, .../teams/get-results, .../venues/get-matches endpoints.
        iid: innings Id (Ex : 1)
        tms: For paging purpose, leave empty to load the first page, or an Epoch timestamp value in milliseconds (Ex : 1640883600000) to load the next page. You are interested in the 'timestamp' field returned right in this endpoint.
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/matches/get-hcommentaries"
    querystring = {'matchId': matchid, }
    if iid:
        querystring['iid'] = iid
    if tms:
        querystring['tms'] = tms
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_hhighlights(matchid: int, htype: int=2, iid: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Another endpoint used to get match highlights"
    matchid: The value of matchId field returned in .../matches/list, .../matches/get-schedules, .../series/get-matches, .../teams/get-schedules, .../teams/get-results, .../venues/get-matches endpoints.
        htype: One of the followings : 2-Fours|4-Sixes|8-Wickets|16-Fifties|32-Hundreds|128-Dropped catches|512-UDRS|1-Others
        iid: innings Id (Ex : 1)
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/matches/get-hhighlights"
    querystring = {'matchId': matchid, }
    if htype:
        querystring['htype'] = htype
    if iid:
        querystring['iid'] = iid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_image(is_id: int, p: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to get image by id"
    id: The value of imageId, faceImageId, etc... returned in other endpoints.
        p: Specify size of image. One of the following : de|gthumb|thumb
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/get-image"
    querystring = {'id': is_id, }
    if p:
        querystring['p'] = p
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stats_get_records(statstype: str, matchtype: int=None, team: int=None, opponent: int=None, year: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get records"
    statstype: The value of 'value' field returned in .../stats/get-record-filters endpoint
        matchtype: The value of matchTypeId field returned right in this endpoint
        team: The value of teamId field returned right in this endpoint
        opponent: The value of teamId field returned right in this endpoint
        year: Specify year to get records. Ex : 2021
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/stats/get-records"
    querystring = {'statsType': statstype, }
    if matchtype:
        querystring['matchType'] = matchtype
    if team:
        querystring['team'] = team
    if opponent:
        querystring['opponent'] = opponent
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
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
    matchtype: One of the followings : 1-World test championship|2-World cup super league
        seasonid: The value of seasonStandings/id field returned right in this endpoint
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/stats/get-icc-standings"
    querystring = {'matchType': matchtype, }
    if seasonid:
        querystring['seasonId'] = seasonid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
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
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/news/get-categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
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
    playerid: The value of id field returned in .../players/list-trending, .../players/search endpoints
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/players/get-news"
    querystring = {'playerId': playerid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
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
    playerid: The value of id field returned in .../players/list-trending, .../players/search endpoints
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/players/get-batting"
    querystring = {'playerId': playerid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def venues_get_matches(venueid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get scheduled matches by venue"
    venueid: The value of id field returned in .../series/get-venues endpoint
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/venues/get-matches"
    querystring = {'venueId': venueid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
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
    seriesid: The value of id field returned in .../series/list or .../series/list-archives endpoints.
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/series/get-matches"
    querystring = {'seriesId': seriesid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
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
    seriesid: The value of id field returned in .../series/list or .../series/list-archives endpoints.
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/series/get-news"
    querystring = {'seriesId': seriesid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def series_list_archives(matchtype: str, year: int=None, lastid: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List archived series"
    matchtype: One of the followings : international|league|domestic|women
        lastid: For paging purpose, leave empty to load the first page, or the value of id field returned right in this endpoint.
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/series/list-archives"
    querystring = {'matchType': matchtype, }
    if year:
        querystring['year'] = year
    if lastid:
        querystring['lastId'] = lastid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_get_stats(teamid: int, statstype: str, team: int=None, matchtype: int=None, year: int=None, opponent: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stats by team"
    teamid: The value of teamId field returned in .../teams/list endpoint
        team: The value of 'teamId' field returned right in this endpoint
        matchtype: The value of matchTypeId field returned right in this endpoint
        year: Specify year to get stats. Ex : 2021
        opponent: The value of 'teamId' field returned right in this endpoint
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/teams/get-stats"
    querystring = {'teamId': teamid, 'statsType': statstype, }
    if team:
        querystring['team'] = team
    if matchtype:
        querystring['matchType'] = matchtype
    if year:
        querystring['year'] = year
    if opponent:
        querystring['opponent'] = opponent
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
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
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/stats/get-record-filters"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stats_get_icc_rankings(formattype: str, category: str, iswomen: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get ICC rankings"
    formattype: One of the followings : test|odi|t20 (if isWomen parameter is 1, there will be no odi)
        category: One of the followings : batsmen|bowlers|allrounders|teams
        iswomen: Set to 1 to get rankings for women
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/stats/get-icc-rankings"
    querystring = {'formatType': formattype, 'category': category, }
    if iswomen:
        querystring['isWomen'] = iswomen
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
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
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/news/get-topics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_detail(storyid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news detail"
    storyid: The value of story/id field returned in .../news/list endpoint
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/news/detail"
    querystring = {'storyId': storyid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_list(categoryid: int=None, topicid: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List latest news from all categories or topics"
    categoryid: Filter news by category, the value of id field returned in .../news/get-categories
        topicid: Filter news by topic, the value of id field returned in .../news/get-topics
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/news/list"
    querystring = {}
    if categoryid:
        querystring['categoryId'] = categoryid
    if topicid:
        querystring['topicId'] = topicid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
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
    playerid: The value of id field returned in .../players/list-trending, .../players/search endpoints
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/players/get-career"
    querystring = {'playerId': playerid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
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
    playerid: The value of id field returned in .../players/list-trending, .../players/search endpoints
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/players/get-bowling"
    querystring = {'playerId': playerid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
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
    playerid: The value of id field returned in .../players/list-trending, .../players/search endpoints
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/players/get-info"
    querystring = {'playerId': playerid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
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
    plrn: Any terms or phrases that you are familiar with
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/players/search"
    querystring = {'plrN': plrn, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
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
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/players/list-trending"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
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
    venueid: The value of id field returned in .../series/get-venues endpoint
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/venues/get-info"
    querystring = {'venueId': venueid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
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
    teamid: The value of teamId field returned in .../teams/list endpoint
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/teams/get-stats-filters"
    querystring = {'teamId': teamid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
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
    teamid: The value of teamId field returned in .../teams/list endpoint
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/teams/get-players"
    querystring = {'teamId': teamid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
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
    teamid: The value of teamId field returned in .../teams/list endpoint
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/teams/get-news"
    querystring = {'teamId': teamid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
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
    teamid: The value of teamId field returned in .../teams/list endpoint
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/teams/get-results"
    querystring = {'teamId': teamid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
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
    teamid: The value of teamId field returned in .../teams/list endpoint
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/teams/get-schedules"
    querystring = {'teamId': teamid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_list(matchtype: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List teams"
    matchtype: One of the followings : international|league|domestic|women
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/teams/list"
    querystring = {'matchType': matchtype, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
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
    seriesid: The value of id field returned in .../series/list or .../series/list-archives endpoints.
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/series/get-points-table"
    querystring = {'seriesId': seriesid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
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
    seriesid: The value of id field returned in .../series/list or .../series/list-archives endpoints.
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/series/get-venues"
    querystring = {'seriesId': seriesid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
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
    seriesid: The value of id field returned in .../series/list or .../series/list-archives endpoints.
        statstype: The value of 'value' field returned in .../series/get-stats-filter endpoint
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/series/get-stats"
    querystring = {'seriesId': seriesid, 'statsType': statstype, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def series_get_players(seriesid: int, squadid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get players by squad and series"
    seriesid: The value of id field returned in .../series/list or .../series/list-archives endpoints.
        squadid: The value of squadId field returned in .../series/get-squads endpoint
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/series/get-players"
    querystring = {'seriesId': seriesid, 'squadId': squadid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
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
    seriesid: The value of id field returned in .../series/list or .../series/list-archives endpoints.
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/series/get-squads"
    querystring = {'seriesId': seriesid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def series_list(matchtype: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List series"
    matchtype: One of the followings : international|league|domestic|women
        
    """
    url = f"https://unofficial-cricbuzz.p.rapidapi.com/series/list"
    querystring = {'matchType': matchtype, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-cricbuzz.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

