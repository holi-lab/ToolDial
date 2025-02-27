import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getdraftbyyear(year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    year: The draft year.
        
    """
    url = f"https://nhl-stats-and-live-data.p.rapidapi.com/draft/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nhl-stats-and-live-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdraftprospect(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    id: The prospect ID.
        
    """
    url = f"https://nhl-stats-and-live-data.p.rapidapi.com/draft/prospects/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nhl-stats-and-live-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstattypes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://nhl-stats-and-live-data.p.rapidapi.com/statTypes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nhl-stats-and-live-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getconference(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can use this to also retrieve inactive conferences. For example, the ID for the World Cup of Hockey is `7`."
    id: The ID of the conference.
        
    """
    url = f"https://nhl-stats-and-live-data.p.rapidapi.com/conferences/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nhl-stats-and-live-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getgame(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This contains all data related to a game, from the boxscore, to play data and even on-ice coordinates. Be forewarned that, depending on the game, this endpoint can return a **lot** of data."
    id: The ID of the game. The first 4 digits identify the season of the game (ie. 2017 for the 2017-2018 season). The next 2 digits give the type of game, where 01 = preseason, 02 = regular season, 03 = playoffs, 04 = all-star. The final 4 digits identify the specific game number. For regular season and preseason games, this ranges from 0001 to the number of games played. (1271 for seasons with 31 teams (2017 and onwards) and 1230 for seasons with 30 teams). For playoff games, the 2nd digit of the specific number gives the round of the playoffs, the 3rd digit specifies the matchup, and the 4th digit specifies the game (out of 7).
        
    """
    url = f"https://nhl-stats-and-live-data.p.rapidapi.com/game/{is_id}/feed/live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nhl-stats-and-live-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstandingsbytype(type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    type: Standing types:
  * `byConference` - Standings by Conference
  * `byDivision` - Standings by Division
  * `byLeague` - Standings by League
  * `divisionLeaders` - Division Leader standings
  * `postseason` - Postseason Standings
  * `preseason` - Preseason Standings
  * `regularSeason` - Regular Season Standings
  * `wildCard` - Wild card standings
  * `wildCardWithLeaders` - Wild card standings with Division Leaders

        
    """
    url = f"https://nhl-stats-and-live-data.p.rapidapi.com/standings/{type}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nhl-stats-and-live-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstandings(date: str='2018-01-09', season: str='20032004', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    date: Standings on a specified date.
        season: Standings for a specified season.
        
    """
    url = f"https://nhl-stats-and-live-data.p.rapidapi.com/standings"
    querystring = {}
    if date:
        querystring['date'] = date
    if season:
        querystring['season'] = season
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nhl-stats-and-live-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getgameboxscore(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "If you want detailed play information, you should use `/game/{id}/feed/live` or `/game/{id}/feed/live/diffPatch`."
    id: The ID of the game. The first 4 digits identify the season of the game (ie. 2017 for the 2017-2018 season). The next 2 digits give the type of game, where 01 = preseason, 02 = regular season, 03 = playoffs, 04 = all-star. The final 4 digits identify the specific game number. For regular season and preseason games, this ranges from 0001 to the number of games played. (1271 for seasons with 31 teams (2017 and onwards) and 1230 for seasons with 30 teams). For playoff games, the 2nd digit of the specific number gives the round of the playoffs, the 3rd digit specifies the matchup, and the 4th digit specifies the game (out of 7).
        
    """
    url = f"https://nhl-stats-and-live-data.p.rapidapi.com/game/{is_id}/boxscore"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nhl-stats-and-live-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getconferences(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This only retrieves active conferences. For inactive conferences, use `/conferences/{id}`."
    
    """
    url = f"https://nhl-stats-and-live-data.p.rapidapi.com/conferences"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nhl-stats-and-live-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getplayerstats(is_id: int, stats: str, season: int=20172018, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    id: The ID of the player.
        stats: Stats explanations:
  * `homeAndAway` - Provides a split between home and away games.
  * `byMonth` - Monthly split of stats.
  * `byDayOfWeek` - Split done by day of the week.
  * `goalsByGameSituation` - Shows number on when goals for a player happened like how many in the shootout, how many in each period, etc.
  * `onPaceRegularSeason` - This only works with the current in-progress season and shows projected totals based on current onPaceRegularSeason.
  * `regularSeasonStatRankings` - Returns where someone stands vs the rest of the league for a specific regularSeasonStatRankings
  * `statsSingleSeason` - Obtains single season statistics for a player.
  * `vsConference` - Conference stats split.
  * `vsDivision` - Division stats split.
  * `vsTeam` - Conference stats split.
  * `winLoss` - Very similar to the previous modifier except it provides the W/L/OT split instead of Home and Away.

        season: Return a team's specific season.
        
    """
    url = f"https://nhl-stats-and-live-data.p.rapidapi.com/people/{is_id}/stats"
    querystring = {'stats': stats, }
    if season:
        querystring['season'] = season
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nhl-stats-and-live-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getgamediff(is_id: int, starttimecode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can use this to return a small subset of data relating to game."
    id: The ID of the game. The first 4 digits identify the season of the game (ie. 2017 for the 2017-2018 season). The next 2 digits give the type of game, where 01 = preseason, 02 = regular season, 03 = playoffs, 04 = all-star. The final 4 digits identify the specific game number. For regular season and preseason games, this ranges from 0001 to the number of games played. (1271 for seasons with 31 teams (2017 and onwards) and 1230 for seasons with 30 teams). For playoff games, the 2nd digit of the specific number gives the round of the playoffs, the 3rd digit specifies the matchup, and the 4th digit specifies the game (out of 7).
        starttimecode: The prospect ID.
        
    """
    url = f"https://nhl-stats-and-live-data.p.rapidapi.com/game/{is_id}/feed/live/diffPatch"
    querystring = {'startTimeCode': starttimecode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nhl-stats-and-live-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getschedule(enddate: str='2018-02-18', startdate: str='2018-02-11', teamid: str='28', expand: str='schedule.brodcasts', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    enddate: End date for the search.
        startdate: Start date for the search.
        teamid: Limit results to a specific team. Team ids can be found through the teams endpoint
        expand: Expand explanations:
  * `schedule.brodcasts` - Shows the broadcasts of the game.
  * `schedule.linescore` - Linescore for completed games.
  * `schedule.ticket` - Provides the different places to buy tickets for the upcoming games.
  * `team.schedule.previous` - Same as above but for the last game played.

        
    """
    url = f"https://nhl-stats-and-live-data.p.rapidapi.com/schedule"
    querystring = {}
    if enddate:
        querystring['endDate'] = enddate
    if startdate:
        querystring['startDate'] = startdate
    if teamid:
        querystring['teamId'] = teamid
    if expand:
        querystring['expand'] = expand
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nhl-stats-and-live-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdraftprospects(page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Be forewarned that this endpoint returns a **lot** of data and there does not appear to be a way to paginate results."
    page: Number of page of the results. 500 results are loaded per page.
        
    """
    url = f"https://nhl-stats-and-live-data.p.rapidapi.com/draft/prospects"
    querystring = {'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nhl-stats-and-live-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getteamroster(is_id: int, season: int=20172018, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    id: The ID of the team.
        season: Return a team's specific season.
        
    """
    url = f"https://nhl-stats-and-live-data.p.rapidapi.com/teams/{is_id}/roster"
    querystring = {}
    if season:
        querystring['season'] = season
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nhl-stats-and-live-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getplayer(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    id: The ID of the player.
        
    """
    url = f"https://nhl-stats-and-live-data.p.rapidapi.com/people/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nhl-stats-and-live-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getteams(season: int=20172018, expand: str='team.roster', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    season: Return a team's specific season.
        expand: Expand your response for some additional data.
        
    """
    url = f"https://nhl-stats-and-live-data.p.rapidapi.com/teams"
    querystring = {}
    if season:
        querystring['season'] = season
    if expand:
        querystring['expand'] = expand
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nhl-stats-and-live-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdivision(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can use this to also retrieve inactive divisions. For example, the ID for the old Patrick division is `13`."
    id: The ID of the division.
        
    """
    url = f"https://nhl-stats-and-live-data.p.rapidapi.com/divisions/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nhl-stats-and-live-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getgamecontent(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    id: The ID of the game. The first 4 digits identify the season of the game (ie. 2017 for the 2017-2018 season). The next 2 digits give the type of game, where 01 = preseason, 02 = regular season, 03 = playoffs, 04 = all-star. The final 4 digits identify the specific game number. For regular season and preseason games, this ranges from 0001 to the number of games played. (1271 for seasons with 31 teams (2017 and onwards) and 1230 for seasons with 30 teams). For playoff games, the 2nd digit of the specific number gives the round of the playoffs, the 3rd digit specifies the matchup, and the 4th digit specifies the game (out of 7).
        
    """
    url = f"https://nhl-stats-and-live-data.p.rapidapi.com/game/{is_id}/content"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nhl-stats-and-live-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getteamstats(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    id: The ID of the team.
        
    """
    url = f"https://nhl-stats-and-live-data.p.rapidapi.com/teams/{is_id}/stats"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nhl-stats-and-live-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstandingtypes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://nhl-stats-and-live-data.p.rapidapi.com/standingsTypes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nhl-stats-and-live-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getteam(is_id: int, expand: str='team.roster', season: int=20172018, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    id: The ID of the team.
        expand: Expand your response for some additional data.
        season: Return a team's specific season.
        
    """
    url = f"https://nhl-stats-and-live-data.p.rapidapi.com/teams/{is_id}"
    querystring = {}
    if expand:
        querystring['expand'] = expand
    if season:
        querystring['season'] = season
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nhl-stats-and-live-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdraft(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://nhl-stats-and-live-data.p.rapidapi.com/draft"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nhl-stats-and-live-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdivisions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This only retrieves active divisions. For inactive divisions, use `/divisions/{id}`."
    
    """
    url = f"https://nhl-stats-and-live-data.p.rapidapi.com/divisions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nhl-stats-and-live-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

