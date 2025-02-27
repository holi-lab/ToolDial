import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def team_stats(leagueyear: str=None, team: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Over 500 different categorized statistics for each team in the NFL (depending on league year).  Optionally, pass in a league year and/or team to narrow the query results."
    leagueyear: The league year of statistics you want to query.  League years refer to the year of the season when it begins, i.e., the 2022-2023 NFL season's league year is 2022.

Minimum value is 1922 (the first year of the NFL), max value is the current league year. All other queries will return null.

Default value is the current league year.
        team: The team whose statistics you want to query. The controller uses a case-insensitive string matcher, so queries of `phi` or `eag` should return statistics for the Philadelphia Eagles. Please note this query will only return one result, so a query such as `New York` may return an unexpected value.

Default value is empty and the query will return all teams for the league year.
        
    """
    url = f"https://nfl-team-stats1.p.rapidapi.com/teamStats"
    querystring = {}
    if leagueyear:
        querystring['leagueYear'] = leagueyear
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nfl-team-stats1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

