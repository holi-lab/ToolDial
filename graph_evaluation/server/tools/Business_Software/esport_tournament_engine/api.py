import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def retrieve_leaderboard(tournamentid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the tournament leaderboard"
    
    """
    url = f"https://esport-tournament-engine.p.rapidapi.com/getTournamentRankings/{tournamentid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esport-tournament-engine.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_multiple_tournaments(limit: int=None, page: int=None, authorid: str=None, discordguildid: str=None, authorusername: str=None, status: str=None, authoremail: str=None, privacy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all your tournaments"
    
    """
    url = f"https://esport-tournament-engine.p.rapidapi.com/getTournaments"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    if authorid:
        querystring['authorId'] = authorid
    if discordguildid:
        querystring['discordGuildId'] = discordguildid
    if authorusername:
        querystring['authorUsername'] = authorusername
    if status:
        querystring['status'] = status
    if authoremail:
        querystring['authorEmail'] = authoremail
    if privacy:
        querystring['privacy'] = privacy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esport-tournament-engine.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_participant_s_upcoming_matches(participantid: str, tournamentid: str, allfixtures: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve upcoming fixtures of a specific participant"
    
    """
    url = f"https://esport-tournament-engine.p.rapidapi.com/getUpcomingFixtures/{tournamentid}"
    querystring = {'participantId': participantid, }
    if allfixtures:
        querystring['allFixtures'] = allfixtures
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esport-tournament-engine.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_a_tournament(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a tournament by its ID"
    
    """
    url = f"https://esport-tournament-engine.p.rapidapi.com/getTournamentById"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esport-tournament-engine.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

