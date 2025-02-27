import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def tokenstatus(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns with token status, expire data and costs ecc.."
    
    """
    url = f"https://soccer-football-info.p.rapidapi.com/token/status/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-football-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historyteams(i: int, l: str='en_US', f: str=None, w: str='6m', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns with all macthes of a specific team in a lapse of time: 'all' from 2017-01-01, '1y' last year, '6m' last 6 months.
		The return format can be 'json' or 'csv' (use 'f' parameter). The CSV format is MS excel compatible (import with double click)"
    i: Element ID
        l: language code
        f: Format of response can be 'json' or 'csv' (default 'json')
        w: lapse of time can be all 1y 6m
        
    """
    url = f"https://soccer-football-info.p.rapidapi.com/teams/history/"
    querystring = {'i': i, }
    if l:
        querystring['l'] = l
    if f:
        querystring['f'] = f
    if w:
        querystring['w'] = w
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-football-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def viewteams(i: int, l: str='en_US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns with team data: id, name, country, manager data, stadium data, has_image, last 5 matches and last lineup"
    i: Element ID
        l: language code
        
    """
    url = f"https://soccer-football-info.p.rapidapi.com/teams/view/"
    querystring = {'i': i, }
    if l:
        querystring['l'] = l
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-football-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listteams(l: str='en_US', c: str='all', f: str=None, p: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns with all teams and their data: id, name, country and if has image.
		This is a paginate call, if you want to know how it works show FAQ on our documentation (https://info.soccerfootball.info/faq#how-paginate-work)
		The return format can be 'json' or 'csv' (use 'f' parameter). The CSV format is MS excel compatible (import with double click)"
    l: language code
        c: country code
        f: Format of response can be 'json' or 'csv' (default 'json')
        p: Page number
        
    """
    url = f"https://soccer-football-info.p.rapidapi.com/teams/list/"
    querystring = {}
    if l:
        querystring['l'] = l
    if c:
        querystring['c'] = c
    if f:
        querystring['f'] = f
    if p:
        querystring['p'] = p
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-football-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listchampionships(p: int=1, c: str='all', f: str=None, l: str='en_US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns with all championships and their data:id, name, country and if has image.
		This is a paginate call, if you want to know how it works show FAQ on our documentation (https://info.soccerfootball.info/faq#how-paginate-work)
		The return format can be 'json' or 'csv' (use 'f' parameter). The CSV format is MS excel compatible (import with double click)"
    p: Page number
        c: country code
        f: Format of response can be 'json' or 'csv' (default 'json')
        l: language code
        
    """
    url = f"https://soccer-football-info.p.rapidapi.com/championships/list/"
    querystring = {}
    if p:
        querystring['p'] = p
    if c:
        querystring['c'] = c
    if f:
        querystring['f'] = f
    if l:
        querystring['l'] = l
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-football-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listreferees(p: int=1, c: str='all', f: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns with all referees and their data: id, name, country and if has image.
		This is a paginate call, if you want to know how it works show FAQ on our documentation (https://info.soccerfootball.info/faq#how-paginate-work)
		The return format can be 'json' or 'csv' (use 'f' parameter). The CSV format is MS excel compatible (import with double click)"
    p: Page number
        c: country code
        f: Format of response can be 'json' or 'csv' (default 'json')
        
    """
    url = f"https://soccer-football-info.p.rapidapi.com/referees/list/"
    querystring = {}
    if p:
        querystring['p'] = p
    if c:
        querystring['c'] = c
    if f:
        querystring['f'] = f
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-football-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def viewreferees(i: int, l: str='en_US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns with referee data: id, name, country, has_image and last 5 macthes"
    i: Element ID
        l: language code
        
    """
    url = f"https://soccer-football-info.p.rapidapi.com/referees/view/"
    querystring = {'i': i, }
    if l:
        querystring['l'] = l
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-football-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def viewmanagers(i: int, l: str='en_US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns with manager data: id, name, country, has_image ,current team, all past team and last 5 matches"
    i: Element ID
        l: language code
        
    """
    url = f"https://soccer-football-info.p.rapidapi.com/managers/view/"
    querystring = {'i': i, }
    if l:
        querystring['l'] = l
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-football-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def viewstadiums(i: int, l: str='en_US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns with stadium data: id, name, city, country, capacity, geo coordinates, has_image and last 5 matches"
    i: Element ID
        l: language code
        
    """
    url = f"https://soccer-football-info.p.rapidapi.com/stadiums/view/"
    querystring = {'i': i, }
    if l:
        querystring['l'] = l
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-football-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def liststadiums(c: str='all', p: int=1, f: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns with all stadiums and their data: id, name, city, country, capacity, geo coordinates and if has an image.
		This is a paginate call, if you want to know how it works show FAQ on our documentation (https://info.soccerfootball.info/faq#how-paginate-work)
		The return format can be 'json' or 'csv' (use 'f' parameter). The CSV format is MS excel compatible (import with double click)"
    c: country code
        p: Page number
        f: Format of response can be 'json' or 'csv' (default 'json')
        
    """
    url = f"https://soccer-football-info.p.rapidapi.com/stadiums/list/"
    querystring = {}
    if c:
        querystring['c'] = c
    if p:
        querystring['p'] = p
    if f:
        querystring['f'] = f
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-football-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listlanguages(f: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all languages with unique code and status of translate.
		The return format can be 'json' or 'csv' (use 'f' parameter). The CSV format is MS excel compatible (import with double click)"
    f: Format of response can be 'json' or 'csv' (default 'json')
        
    """
    url = f"https://soccer-football-info.p.rapidapi.com/languages/list/"
    querystring = {}
    if f:
        querystring['f'] = f
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-football-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listcountries(f: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all countries with unique code and a count of the releated items.
		The return format can be 'json' or 'csv' (use 'f' parameter). The CSV format is MS excel compatible (import with double click)"
    f: Format of response can be 'json' or 'csv' (default 'json')
        
    """
    url = f"https://soccer-football-info.p.rapidapi.com/countries/list/"
    querystring = {}
    if f:
        querystring['f'] = f
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-football-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def daymatchesbasic(d: int, f: str=None, l: str='en_US', p: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns with all matches of as specific day with: id, date, status, championship data, match stats and match events (only in JSON).
		This call pagine only for today and future day, in past there is all result in one call. If you want to know how it works show FAQ on our documentation (https://info.soccerfootball.info/faq#how-paginate-work)
		The return format can be 'json' or 'csv' (use 'f' parameter). The CSV format is MS excel compatible (import with double click)"
    d: Date ISO format without separator
        f: Format of response can be 'json' or 'csv' (default 'json')
        l: language code
        p: page
        
    """
    url = f"https://soccer-football-info.p.rapidapi.com/matches/day/basic/"
    querystring = {'d': d, }
    if f:
        querystring['f'] = f
    if l:
        querystring['l'] = l
    if p:
        querystring['p'] = p
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-football-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def daymatchesfull(d: int, p: int=1, l: str='en_US', f: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns with all matches of as specific day with: id, date, status, championship data, match stats,match events (only in JSON) and match odds.
		This call pagine only for today and future day, in past there is all result in one call. If you want to know how it works show FAQ on our documentation (https://info.soccerfootball.info/faq#how-paginate-work)
		The return format can be 'json' or 'csv' (use 'f' parameter). The CSV format is MS excel compatible (import with double click)"
    d: Date ISO format without separator
        p: page
        l: language code
        f: Format of response can be 'json' or 'csv' (default 'json')
        
    """
    url = f"https://soccer-football-info.p.rapidapi.com/matches/day/full/"
    querystring = {'d': d, }
    if p:
        querystring['p'] = p
    if l:
        querystring['l'] = l
    if f:
        querystring['f'] = f
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-football-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def macthesodds(i: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns with all match odds (if avaible): 1x2, over_under (goal line), asian handicap, asian corner, 1 half asian handicap, 1 half over under, 1 half asian corner and 1 half result. All odds provided from bet365 and unibet"
    i: Element ID
        
    """
    url = f"https://soccer-football-info.p.rapidapi.com/matches/odds/"
    querystring = {'i': i, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-football-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def viewmatchesbasic(i: int, l: str='en_US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns with match data: id, date, status, timer, championship data, match stats and match events"
    i: Element ID
        l: language code
        
    """
    url = f"https://soccer-football-info.p.rapidapi.com/matches/view/basic/"
    querystring = {'i': i, }
    if l:
        querystring['l'] = l
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-football-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def viewmatchesfull(i: int, l: str='en_US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns with match data: id, date, status, timer, championship data, match stats, match events and match odds"
    i: Element ID
        l: language code
        
    """
    url = f"https://soccer-football-info.p.rapidapi.com/matches/view/full/"
    querystring = {'i': i, }
    if l:
        querystring['l'] = l
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-football-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def viewmatchesprogressive(i: int, f: str=None, l: str='en_US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns with match data (stats and odds) every 30 seconds.
		 This endpoint has data since 2020-01-01
		 Live match can be view only with ULTRA plan
		This call is our killer feature
		The return format can be 'json' or 'csv' (use 'f' parameter). The CSV format is MS excel compatible (import with double click)"
    i: Element ID
        f: Format of response can be 'json' or 'csv' (default 'json')
        l: language code
        
    """
    url = f"https://soccer-football-info.p.rapidapi.com/matches/view/progressive/"
    querystring = {'i': i, }
    if f:
        querystring['f'] = f
    if l:
        querystring['l'] = l
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-football-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def livematchesbasic(l: str='en_US', e: str='no', f: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns with all live mactches with data: id, in_play, status, championship data, teams data and match stats.
		The return format can be 'json' or 'csv' (use 'f' parameter). The CSV format is MS excel compatible (import with double click)"
    l: language code
        f: Format of response can be 'json' or 'csv' (default 'json')
        
    """
    url = f"https://soccer-football-info.p.rapidapi.com/live/basic/"
    querystring = {}
    if l:
        querystring['l'] = l
    if e:
        querystring['e'] = e
    if f:
        querystring['f'] = f
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-football-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def livematchesfull(l: str='en_US', f: str=None, e: str='no', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns with all live mactches with data: id, in_play, status, championship data, teams data, match stats and match odds.
		The return format can be 'json' or 'csv' (use 'f' parameter). The CSV format is MS excel compatible (import with double click)"
    l: language code
        f: Format of response can be 'json' or 'csv' (default 'json')
        
    """
    url = f"https://soccer-football-info.p.rapidapi.com/live/full/"
    querystring = {}
    if l:
        querystring['l'] = l
    if f:
        querystring['f'] = f
    if e:
        querystring['e'] = e
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-football-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listmanagers(f: str=None, p: int=1, c: str='all', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns with all managers and their data: id, name, country and if has image.
		This is a paginate call, if you want to know how it works show FAQ on our documentation (https://info.soccerfootball.info/faq#how-paginate-work)
		The return format can be 'json' or 'csv' (use 'f' parameter). The CSV format is MS excel compatible (import with double click)"
    f: Format of response can be 'json' or 'csv' (default 'json')
        p: Page number
        c: country code
        
    """
    url = f"https://soccer-football-info.p.rapidapi.com/managers/list/"
    querystring = {}
    if f:
        querystring['f'] = f
    if p:
        querystring['p'] = p
    if c:
        querystring['c'] = c
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-football-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listplayers(c: str='all', f: str=None, p: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns with all players and their data: id, name, country and if has image.
		This is a paginate call, if you want to know how it works show FAQ on our documentation (https://info.soccerfootball.info/faq#how-paginate-work)
		The return format can be 'json' or 'csv' (use 'f' parameter). The CSV format is MS excel compatible (import with double click)"
    c: country code
        f: Format of response can be 'json' or 'csv' (default 'json')
        p: Page number
        
    """
    url = f"https://soccer-football-info.p.rapidapi.com/players/list/"
    querystring = {}
    if c:
        querystring['c'] = c
    if f:
        querystring['f'] = f
    if p:
        querystring['p'] = p
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-football-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def versusteams(x: int, y: int, f: str=None, l: str='en_US', w: str='6m', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns with all matches between two teams and the results
		The return format can be 'json' or 'csv' (use 'f' parameter). The CSV format is MS excel compatible (import with double click)"
    x: Team ID of X team
        y: Team ID of Y team
        f: Format of response can be 'json' or 'csv' (default 'json')
        l: language code
        w: lapse of time can be all 1y 6m
        
    """
    url = f"https://soccer-football-info.p.rapidapi.com/teams/versus/"
    querystring = {'x': x, 'y': y, }
    if f:
        querystring['f'] = f
    if l:
        querystring['l'] = l
    if w:
        querystring['w'] = w
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-football-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def viewplayers(i: int, l: str='en_US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns with player data: id, name, country, birth date, prefer foot, height, has_image, last team and all his transfers"
    i: Element ID
        l: language code
        
    """
    url = f"https://soccer-football-info.p.rapidapi.com/players/view/"
    querystring = {'i': i, }
    if l:
        querystring['l'] = l
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-football-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def totalcornermatchschedule(date: str='today_ISO', l: str='en_US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns with past, present or future match with totalcorner schema..
		Why?
		Because of our customers have written software/bots and applications based on the totalcorner scheme. Changing APIs would mean rewriting everything. This way our customers can access our data, and not change their software!
		 See out docs at (https://info.soccerfootball.info/v1/emulated/totalcorner/match-today) to read all differents and improvements of our version."
    date: Date ISO format without separator (default today)
        l: language code
        
    """
    url = f"https://soccer-football-info.p.rapidapi.com/emulation/totalcorner/match/schedule/"
    querystring = {}
    if date:
        querystring['date'] = date
    if l:
        querystring['l'] = l
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-football-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def viewchampionships(i: int, l: str='en_US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns with championship data: id, name, country, has_image and all seasons with their groups and tables"
    i: Element ID
        l: language code
        
    """
    url = f"https://soccer-football-info.p.rapidapi.com/championships/view/"
    querystring = {'i': i, }
    if l:
        querystring['l'] = l
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-football-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def customendpoints(i: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Customized endpoints, tailor-made for you. Write an email to info@soccerfootball.info to request one."
    i: endpoint ID
        
    """
    url = f"https://soccer-football-info.p.rapidapi.com/custom/"
    querystring = {'i': i, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-football-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def totalcornermatchtoday(o: str='no', l: str='en_US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns with all live matches with totalcorner schema (match/today).
		Why?
		Because of our customers have written software/bots and applications based on the totalcorner scheme. Changing APIs would mean rewriting everything. This way our customers can access our data, and not change their software!
		 See out docs at (https://info.soccerfootball.info/v1/emulated/totalcorner/match-today) to read all differents and improvements of our version."
    o: overtime yes or no
        l: language code
        
    """
    url = f"https://soccer-football-info.p.rapidapi.com/emulation/totalcorner/match/today/"
    querystring = {}
    if o:
        querystring['o'] = o
    if l:
        querystring['l'] = l
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-football-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bymatchesbasic(s: int=1, l: str='en_US', m: int=1, p: int=1, c: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns with all matches of as specific filter with basic data.You can filter by: c championship id, m manager id and  s stadium id. If you use more filter will be use AND logic"
    s: Stadium id
        l: language code
        m: Manager id
        p: page
        c: Cahmpionship id
        
    """
    url = f"https://soccer-football-info.p.rapidapi.com/matches/by/basic/"
    querystring = {}
    if s:
        querystring['s'] = s
    if l:
        querystring['l'] = l
    if m:
        querystring['m'] = m
    if p:
        querystring['p'] = p
    if c:
        querystring['c'] = c
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-football-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bymatchesfull(p: int=1, l: str='en_US', c: int=1, s: int=1, m: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns with all matches of as specific filter with full data.You can filter by: c championship id, m manager id and  s stadium id. If you use more filter will be use AND logic"
    p: page
        l: language code
        c: Cahmpionship id
        s: Stadium id
        m: Manager id
        
    """
    url = f"https://soccer-football-info.p.rapidapi.com/matches/by/full/"
    querystring = {}
    if p:
        querystring['p'] = p
    if l:
        querystring['l'] = l
    if c:
        querystring['c'] = c
    if s:
        querystring['s'] = s
    if m:
        querystring['m'] = m
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-football-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

