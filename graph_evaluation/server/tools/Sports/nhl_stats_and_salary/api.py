import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_nhl_ahl_prospect_stats_injury_information_and_salary_info_with_filter(league: str, team: str, position: str, playername: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this get request you will get all players stats, injury information and salary information by  these filter :
		**All parameter are required but can be empty**
		-PlayerName
		-Position (ATT,DEF,C,LW,RW,G,RD,LD)
		-Team (bruins,canadiens,panthers etc...)
		-League (NHL,AHL,PROSPECT,RESERVE)
		
		**(Important to know that the id of the player will change every day so i recommended to use the name or make your own id system)**"
    league: Required but can be empty 

possible values : (NHL,AHL,PROSPECT,RESERVE)

        team: Required but can be empty 
possible values : 
capitals
capitals
hurricanes
kraken
goldenknights
devils
stars
bruins
oilers
panthers
flyers
predators
mapleleafs
canadiens
kings
sharks
ducks
islanders
coyotes
wild
avalanche
bluejackets
flames
senators
rangers
jets
lightning
canucks
penguins
redwings
blues
blackhawks
sabres

        position: The parameter is required but can be empty
ATT,DEF,C,LW,RW,G,RD,LD
        playername: The parameter is required but can be empty
        
    """
    url = f"https://nhl-stats-and-salary.p.rapidapi.com/NHLAHLStatsAndSalaryInfo/PlayersWithFilter"
    querystring = {'League': league, 'Team': team, 'Position': position, 'PlayerName': playername, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nhl-stats-and-salary.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

