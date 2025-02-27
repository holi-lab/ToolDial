import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getpointstable(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The getPointsTable endpoint returns the points table for Indian Premier League 2023. This endpoint provides the current standings of all teams participating in IPL 2023, including the total number of matches played, wins, losses, and points earned. The points are calculated based on the current tournament format, which awards two points for a win, one point for a tie or no result, and zero points for a loss. The points table is updated after each match, making it a valuable resource for fans, analysts, and broadcasters to track the progress of each team throughout the tournament. Use the data returned by this endpoint to create IPL 2023 leaderboard, tables, and graphs for analysis or display."
    
    """
    url = f"https://ipl-2023-live-scores-match-details-and-points-table-api.p.rapidapi.com/getPointsTable"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ipl-2023-live-scores-match-details-and-points-table-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlivescore(match_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The getLiveScore endpoint returns live scores of a specific match for Indian Premier League 2023. This endpoint takes a match_id parameter, which is the unique ID of the match whose live scores you want to retrieve. The endpoint provides real-time data for the match, including the current score, total overs, and the latest updates on wickets and runs. The data is updated continuously throughout the match, making it ideal for live scorecards, fantasy cricket apps, or sports betting platforms. Use the getMatchIds endpoint to fetch the match_id for the desired match."
    
    """
    url = f"https://ipl-2023-live-scores-match-details-and-points-table-api.p.rapidapi.com/getLiveScore"
    querystring = {'match_id': match_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ipl-2023-live-scores-match-details-and-points-table-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmatchids(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The getMatchIds endpoint returns the details of all matches and their corresponding match IDs for Indian Premier League 2023. This endpoint provides the following information for each match: match ID, match date, teams playing, and venue. The match ID can be used to fetch live scores for a specific match using the getLiveScore endpoint. This endpoint is useful for creating match schedules, generating fixtures, or displaying a list of matches for IPL 2023."
    
    """
    url = f"https://ipl-2023-live-scores-match-details-and-points-table-api.p.rapidapi.com/getMatchIds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ipl-2023-live-scores-match-details-and-points-table-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

