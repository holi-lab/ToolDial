import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def news(sport: int, timezone: str, langid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news."
    sport: The sport.
        timezone: The timezone name. Check the tutorials.
        langid: The language id. Check the tutorials.
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/news"
    querystring = {'sport': sport, 'timezone': timezone, 'langId': langid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stats(langid: int, timezone: str, competition: int, competitorid: int=1303, phasenum: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get competitor or competition stats."
    langid: The language id. Check the tutorials.
        timezone: The timezone name. Check the tutorials.
        competition: The competition id.
        competitorid: The competitor id. If you only need competition stats leave it empty.
        phasenum: The phase number.
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/stats"
    querystring = {'langId': langid, 'timezone': timezone, 'competition': competition, }
    if competitorid:
        querystring['competitorId'] = competitorid
    if phasenum:
        querystring['phaseNum'] = phasenum
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sports(timezone: str, langid: int, withcount: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of sports."
    timezone: The timezone name. Check the tutorials.
        langid: The language id. Check the tutorials.
        withcount: Include events count..
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/sports"
    querystring = {'timezone': timezone, 'langId': langid, }
    if withcount:
        querystring['withCount'] = withcount
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def paging_games(paging: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get games from paging for competitors or competitions."
    paging: The paging string.
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/paging"
    querystring = {'paging': paging, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(query: str, langid: int, timezone: str, filter: str, sport: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search tool."
    query: The query string.
        langid: The language id. Check the tutorials.
        timezone: The timezone name. Check the tutorials.
        filter: The filter.
        sport: The sport.
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/search"
    querystring = {'query': query, 'langId': langid, 'timezone': timezone, 'filter': filter, }
    if sport:
        querystring['sport'] = sport
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def custom_scores(langid: str, timezone: str, enddate: str, startdate: str, competitions: str='11', competitorids: str='5054,5050,132,131,134,139,135', lastupdateid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get custom scores for competitors and competitions."
    langid: The language id. Check the tutorials.
        timezone: The timezone name. Check the tutorials.


        enddate: The last date to get the scores.
        startdate: The first date to get the scores.
        lastupdateid: The last update id. Only fill this parameter if you want updated data
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/custom-scores"
    querystring = {'langId': langid, 'timezone': timezone, 'endDate': enddate, 'startDate': startdate, }
    if competitions:
        querystring['competitions'] = competitions
    if competitorids:
        querystring['competitorIds'] = competitorids
    if lastupdateid:
        querystring['lastUpdateId'] = lastupdateid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_competitors(timezone: str, withseasons: bool, limit: int, langid: int, sport: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top competitors."
    timezone: The timezone name. Check the tutorials.
        withseasons: Include seasons?
        limit: The limit to return.
        langid: The language id. Check the tutorials.
        sport: The sport.
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/top-competitors"
    querystring = {'timezone': timezone, 'withSeasons': withseasons, 'limit': limit, 'langId': langid, 'sport': sport, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def athlete_small_image(imageversion: int, athleteid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get athlete small image. It generates image/png."
    imageversion: The imageVersion value.
        athleteid: The athlete id.
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/img/small/athlete/{athleteid}/version/{imageversion}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competitor_squads(langid: int, timezone: str, competitorid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get competitor squads."
    langid: The language id. Check the tutorials.
        timezone: The timezone name. Check the tutorials.
        competitorid: The competitor id.
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/competitor-squads"
    querystring = {'langId': langid, 'timezone': timezone, 'competitorId': competitorid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competition_large_image(imageversion: int, competitionid: int, countryid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get competition large image. It generates image/png."
    imageversion: The imageVersion value.
        competitionid: The athlete id.
        countryid: The country id.
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/img/large/country/{countryid}/competition/{competitionid}/version/{imageversion}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competition_brackets(timezone: str, langid: int, competitor: int=None, lastupdateid: int=None, competition: int=183, live: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get competition braackets."
    timezone: The timezone name. Check the tutorials.
        langid: The language id. Check the tutorials.
        competitor: The competitor. Leave it empty if you only want competition brackets.
        lastupdateid: The last update id. Only fill this parameter if you want updated data
        competition: The competition id. Leave it empty if you only want competitor brackets.
        live: Return live brackets?
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/competition-brackets"
    querystring = {'timezone': timezone, 'langId': langid, }
    if competitor:
        querystring['competitor'] = competitor
    if lastupdateid:
        querystring['lastUpdateId'] = lastupdateid
    if competition:
        querystring['competition'] = competition
    if live:
        querystring['live'] = live
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competitor_large_image(imageversion: int, competitorid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get competitor large image. It generates image/png."
    imageversion: The imageVersion value.
        competitorid: The athlete id.
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/img/large/competitor/{competitorid}/version/{imageversion}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_competitions(timezone: str, langid: int, sport: int, limit: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top competitions"
    timezone: The timezone name. Check the tutorials.
        langid: The language id. Check the tutorials.
        sport: The sport.
        limit: The limit to return.
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/top-competitions"
    querystring = {'timezone': timezone, 'langId': langid, 'sport': sport, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_athletes(langid: int, timezone: str, limit: int, sporttype: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top athletes."
    langid: The language id. Check the tutorials.
        timezone: The timezone name. Check the tutorials.
        limit: The nummber of players to return.
        sporttype: The sport type.
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/top-athletes"
    querystring = {'langId': langid, 'timezone': timezone, 'limit': limit, 'sportType': sporttype, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competitor_details(competitorid: int, langid: int, withseasons: bool, timezone: str, lastupdateid: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get competitor details."
    competitorid: The competitor id.
        langid: The language id. Check the tutorials.
        withseasons: Include season?.
        timezone: The timezone name. Check the tutorials.
        lastupdateid: The last update id. Only fill this parameter if you want updated data
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/competitor-details"
    querystring = {'competitorId': competitorid, 'langId': langid, 'withSeasons': withseasons, 'timezone': timezone, }
    if lastupdateid:
        querystring['lastUpdateId'] = lastupdateid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def athlete_details(timezone: str, langid: int, fulldetails: bool, athleteid: int, competition: int=103, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get athlete details."
    timezone: The timezone name. Check the tutorials.
        langid: The language id. Check the tutorials.
        fulldetails: Include full details?
        athleteid: The athlete id.
        competition: The competition id.
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/athlete-details"
    querystring = {'timezone': timezone, 'langId': langid, 'fullDetails': fulldetails, 'athleteId': athleteid, }
    if competition:
        querystring['competition'] = competition
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competitor_current_games(competitorid: int, timezone: str, langid: int, lastupdateid: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get competitor current games."
    competitorid: The competitor id.
        timezone: The timezone name. Check the tutorials.
        langid: The language id. Check the tutorials.
        lastupdateid: TThe last update id. Only fill this parameter if you want updated data
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/competitor-current-games"
    querystring = {'competitorId': competitorid, 'timezone': timezone, 'langId': langid, }
    if lastupdateid:
        querystring['lastUpdateId'] = lastupdateid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_predictions(timezone: str, langid: int, sport: int, competitors: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get game predictions."
    timezone: The timezone name. Check the tutorials.
        langid: The language id. Check the tutorials.
        sport: The sport.
        competitors: The competitor id. Only if needed.
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/game-predictions"
    querystring = {'timezone': timezone, 'langId': langid, 'sport': sport, }
    if competitors:
        querystring['competitors'] = competitors
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competition_small_image(imageversion: int, countryid: int, competitionid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get competition small image. It generates image/png."
    imageversion: The imageVersion value.
        countryid: The country id.
        competitionid: The athlete id.
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/img/small/country/{countryid}/competition/{competitionid}/version/{imageversion}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def results(langid: int, timezone: str, lastupdateid: int=None, competition: int=103, competitorid: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get results for competitors or competitions."
    langid: The language id. Check the tutorials.
        timezone: The timezone name. Check the tutorials.
        lastupdateid: The last update id. Only fill this parameter if you want updated data
        competition: The competition id. Only if you want competition's results
        competitorid: The competitor id. Only if you want the competitor's results.
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/results"
    querystring = {'langId': langid, 'timezone': timezone, }
    if lastupdateid:
        querystring['lastUpdateId'] = lastupdateid
    if competition:
        querystring['competition'] = competition
    if competitorid:
        querystring['competitorId'] = competitorid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_scores(startdate: str, langid: int, sport: int, enddate: str, timezone: str, onlymajorgames: bool=None, onlylivegames: bool=None, lastupdateid: int=None, withtop: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get games scores."
    startdate: The start date. Format it like: DD/MM/YYYY.
        langid: The language id. Check the tutorials.
        sport: The sport.
        enddate: The end date. Format it like: DD/MM/YYYY.
        timezone: The timezone name. Check the tutorials.
        onlymajorgames: Only include major games?
        onlylivegames: Only include live games?
        lastupdateid: The last update id. Only fill this parameter if you want updated data
        withtop: Return with top?
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/games-scores"
    querystring = {'startDate': startdate, 'langId': langid, 'sport': sport, 'endDate': enddate, 'timezone': timezone, }
    if onlymajorgames:
        querystring['onlyMajorGames'] = onlymajorgames
    if onlylivegames:
        querystring['onlyLiveGames'] = onlylivegames
    if lastupdateid:
        querystring['lastUpdateId'] = lastupdateid
    if withtop:
        querystring['withTop'] = withtop
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competitor_small_image(imageversion: int, competitorid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get competitor small image. It generates image/png."
    imageversion: The imageVersion value.
        competitorid: The athlete id.
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/img/small/competitor/{competitorid}/version/{imageversion}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def athlete_large_image(athleteid: int, imageversion: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get athlete large image. It generates image/png."
    athleteid: The athlete id.
        imageversion: The imageVersion value.
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/img/large/athlete/{athleteid}/version/{imageversion}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competition_standings(ispreview: bool, seasonnum: int, competition: int, stagenum: int, live: bool, langid: int, timezone: str, lastupdateid: int=None, groupcategory: int=None, type: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get competition standings."
    ispreview: Is it for use in a preview?
        seasonnum: The season number.
        competition: The competition id.
        stagenum: The stage number.
        live: Get live standings.
        langid: The language id. Check the tutorials.
        timezone: The timezone name. Check the tutorials.
        lastupdateid: The last update id. Only fill this parameter if you want updated data
        groupcategory: The group category.
        type: The type.
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/competition-standings"
    querystring = {'isPreview': ispreview, 'seasonNum': seasonnum, 'competition': competition, 'stageNum': stagenum, 'live': live, 'langId': langid, 'timezone': timezone, }
    if lastupdateid:
        querystring['lastUpdateId'] = lastupdateid
    if groupcategory:
        querystring['groupCategory'] = groupcategory
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def athlete_trophy_stats(athleteid: int, competitionid: int, timezone: str, langid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get athlete trophy stats."
    athleteid: The athlete id.
        competitionid: The competition id.
        timezone: The timezone name. Check the tutorials.
        langid: The language id. Check the tutorials.
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/athlete-trophy-stats"
    querystring = {'athleteId': athleteid, 'competitionId': competitionid, 'timezone': timezone, 'langId': langid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_flag(countryid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get country flag. It generates image/png."
    countryid: The country id.
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/img/flag/country/{countryid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_details(timezone: str, gameid: int, langid: int, lastupdateid: int=None, matchupid: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get game details."
    timezone: The timezone name. Check the tutorials.
        gameid: The game id.
        langid: The language id. Check the tutorials.
        lastupdateid: The last update id. Only fill this parameter if you want updated data
        matchupid: The matchup id.
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/game-details"
    querystring = {'timezone': timezone, 'gameId': gameid, 'langId': langid, }
    if lastupdateid:
        querystring['lastUpdateId'] = lastupdateid
    if matchupid:
        querystring['matchupId'] = matchupid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures(langid: int, timezone: str, competitorid: int=None, lastupdateid: int=None, competition: int=103, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get fixtures for competitors or competitions."
    langid: The language id. Check the tutorials.
        timezone: The timezone name. Check the tutorials.
        competitorid: The competitor id. Only if you want the competitor's fixtures.
        lastupdateid: The last update id. Only fill this parameter if you want updated data
        competition: The competition id. Only if you want competition's fixtures
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/fixtures"
    querystring = {'langId': langid, 'timezone': timezone, }
    if competitorid:
        querystring['competitorId'] = competitorid
    if lastupdateid:
        querystring['lastUpdateId'] = lastupdateid
    if competition:
        querystring['competition'] = competition
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def featured_competitions(timezone: str, sport: int, langid: int, type: str, withseasons: bool, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get featured competitions."
    timezone: The timezone name. Check the tutorials.
        sport: The sport.
        langid: The language id. Check the tutorials.
        type: The type.
        withseasons: Include seasons?
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/featured-competitions"
    querystring = {'timezone': timezone, 'sport': sport, 'langId': langid, 'type': type, 'withSeasons': withseasons, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_h2h_matches(gameid: int, langid: int, timezone: str, lastupdateid: int=None, matchupid: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get game h2h matches."
    gameid: The game id.
        langid: The language id. Check the tutorials.
        timezone: The timezone name. Check the tutorials.
        lastupdateid: The last update id. Only fill this parameter if you want updated data
        matchupid: The matchup id.
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/game-h2h"
    querystring = {'gameId': gameid, 'langId': langid, 'timezone': timezone, }
    if lastupdateid:
        querystring['lastUpdateId'] = lastupdateid
    if matchupid:
        querystring['matchupId'] = matchupid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competition_details(competition: int, langid: int, timezone: str, withseasons: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get competition details."
    competition: The competition id.
        langid: The language id. Check the tutorials.
        timezone: The timezone name. Check the tutorials.
        withseasons: Include seasons?
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/competition-details"
    querystring = {'competition': competition, 'langId': langid, 'timezone': timezone, }
    if withseasons:
        querystring['withSeasons'] = withseasons
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def featured_games(langid: int, timezone: str, numberofgames: int, sport: int, lastupdateid: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get featured games."
    langid: The language id. Check the tutorials.
        timezone: The timezone name. Check the tutorials.
        numberofgames: The number of games to include.
        sport: The sport.
        lastupdateid: The last update id. Only fill this parameter if you want updated data
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/featured-games"
    querystring = {'langId': langid, 'timezone': timezone, 'numberOfGames': numberofgames, 'sport': sport, }
    if lastupdateid:
        querystring['lastUpdateId'] = lastupdateid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countries(langid: int, sportid: int, timezone: str, lastupdateid: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of countries."
    langid: The language id. Check the tutorials.
        sportid: The sport id.
        timezone: The timezone name. Check the tutorials.
        lastupdateid: The last update id. Only fill this parameter if you want updated data.
        
    """
    url = f"https://allscores.p.rapidapi.com/api/allscores/countries"
    querystring = {'langId': langid, 'sportId': sportid, 'timezone': timezone, }
    if lastupdateid:
        querystring['lastUpdateId'] = lastupdateid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allscores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

