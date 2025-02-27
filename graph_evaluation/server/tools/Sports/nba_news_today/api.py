import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_latest_nba_news_by_source(sourceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest NBA news based on its source. 
		You can get from the following sports websites: 
		
		ESPN -> espn
		NBA -> nba 
		SportsIllustrated  -> si
		SBNation -> sbnation
		The Ringer -> theringer
		The Athletic -> theathletic
		RealGM -> realgm
		HoopsHype -> hoopshype"
    
    """
    url = f"https://nba-news-today.p.rapidapi.com/news/{sourceid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-news-today.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_latest_nba_news_for_each_team_and_source(sourceid: str, teamid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest NBA news based on the 30 teams and popular news outlets. 
		Available Team Ids: 
		
		Boston Celtics -> celtics
		Brooklyn Nets -> nets
		NY Knicks -> knicks 
		Philadelphia 76ers -> sixers
		Toronto Raptors -> raptors
		Chicago Buls -> bulls
		Cleveland Cavaliers -> cavs
		Detroit Pistons -> pistons
		Indiana Pacers -> pacers
		Milwaukee Bucks -> bucks
		Atlanta Hawks -> hawks
		Charlotte Hornets -> hornets
		Miami Heat -> heat
		Orlando Magic -> magic
		Washington Wizards -> wizards
		Denver Nuggets -> nuggets
		Minnesota Timberwolves -> wolves
		OKC Thunder -> thunder
		Portland Trailblazers -> blazers
		Utah Jazz -> jazz
		Golden State Warriors -> warriors
		LA Clippers -> clippers
		LA Lakers -> lakers
		Phoenix Suns -> suns
		Sacramento Kings -> kings
		Dallas Mavericks -> mavs
		Houston Rockets -> rockets
		Memphis Grizzlies -> grizzlies
		NO Pelicans -> pelicans
		San Antonio Spurs -> spurs
		
		Source Ids: 
		ESPN -> espn
		bleacherreport -> br
		SBNation -> sbnation"
    
    """
    url = f"https://nba-news-today.p.rapidapi.com/news/teams/{teamid}/{sourceid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-news-today.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_latest_nba_news_by_team(teamid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest NBA news based on the 30 teams. 
		You can use the following Ids to get the latest news from a specific NBA team.
		
		Boston Celtics -> celtics
		Brooklyn Nets -> nets
		NY Knicks -> knicks 
		Philadelphia 76ers -> sixers
		Toronto Raptors -> raptors
		Chicago Buls -> bulls
		Cleveland Cavaliers -> cavs
		Detroit Pistons -> pistons
		Indiana Pacers -> pacers
		Milwaukee Bucks -> bucks
		Atlanta Hawks -> hawks
		Charlotte Hornets -> hornets
		Miami Heat -> heat
		Orlando Magic -> magic
		Washington Wizards -> wizards
		Denver Nuggets -> nuggets
		Minnesota Timberwolves -> wolves
		OKC Thunder -> thunder
		Portland Trailblazers -> blazers
		Utah Jazz -> jazz
		Golden State Warriors -> warriors
		LA Clippers -> clippers
		LA Lakers -> lakers
		Phoenix Suns -> suns
		Sacramento Kings -> kings
		Dallas Mavericks -> mavs
		Houston Rockets -> rockets
		Memphis Grizzlies -> grizzlies
		NO Pelicans -> pelicans
		San Antonio Spurs -> spurs"
    
    """
    url = f"https://nba-news-today.p.rapidapi.com/news/teams/{teamid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-news-today.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_latest_nba_news_from_popular_news_outlets(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back all the latest news about the NBA from all popular NBA news outlets."
    
    """
    url = f"https://nba-news-today.p.rapidapi.com/news"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-news-today.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

