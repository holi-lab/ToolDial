import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_career_totals_allstar(ft_pct: str=None, ftm: str=None, league_id: str=None, pf: str=None, per_page: int=50, pts: str=None, page: int=1, fg3m: str=None, gs: str=None, fgm: str=None, blk: str=None, fga: str=None, fg_pct: str=None, dreb: str=None, tov: str=None, fg3a: str=None, gp: str=None, fg3_pct: str=None, team_id: str=None, fta: str=None, reb: str=None, oreb: str=None, ast: str=None, stl: str=None, min: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query career totals allstar"
    ft_pct: Filter by ft_pct
        ftm: Filter by ftm
        league_id: Filter by league_id
        pf: Filter by pf
        per_page: Number of resources to return per page for pagination (1 - 500)
        pts: Filter by pts
        page: Page value for pagination
        fg3m: Filter by fg3m
        gs: Filter by gs
        fgm: Filter by fgm
        blk: Filter by blk
        fga: Filter by fga
        fg_pct: Filter by fg_pct
        dreb: Filter by dreb
        tov: Filter by tov
        fg3a: Filter by fg3a
        gp: Filter by gp
        fg3_pct: Filter by fg3_pct
        team_id: Filter by team_id
        fta: Filter by fta
        reb: Filter by reb
        oreb: Filter by oreb
        ast: Filter by ast
        stl: Filter by stl
        min: Filter by min
        
    """
    url = f"https://nba-stats4.p.rapidapi.com/career_totals_allstar/"
    querystring = {}
    if ft_pct:
        querystring['ft_pct'] = ft_pct
    if ftm:
        querystring['ftm'] = ftm
    if league_id:
        querystring['league_id'] = league_id
    if pf:
        querystring['pf'] = pf
    if per_page:
        querystring['per_page'] = per_page
    if pts:
        querystring['pts'] = pts
    if page:
        querystring['page'] = page
    if fg3m:
        querystring['fg3m'] = fg3m
    if gs:
        querystring['gs'] = gs
    if fgm:
        querystring['fgm'] = fgm
    if blk:
        querystring['blk'] = blk
    if fga:
        querystring['fga'] = fga
    if fg_pct:
        querystring['fg_pct'] = fg_pct
    if dreb:
        querystring['dreb'] = dreb
    if tov:
        querystring['tov'] = tov
    if fg3a:
        querystring['fg3a'] = fg3a
    if gp:
        querystring['gp'] = gp
    if fg3_pct:
        querystring['fg3_pct'] = fg3_pct
    if team_id:
        querystring['team_id'] = team_id
    if fta:
        querystring['fta'] = fta
    if reb:
        querystring['reb'] = reb
    if oreb:
        querystring['oreb'] = oreb
    if ast:
        querystring['ast'] = ast
    if stl:
        querystring['stl'] = stl
    if min:
        querystring['min'] = min
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_per36_career_post_season(page: int=1, fta_per36: str=None, oreb_per36: str=None, reb_per36: str=None, fg3a_per36: str=None, min: str=None, ast_per36: str=None, fgm_per36: str=None, per_page: int=50, ftm_per36: str=None, fg3m_per36: str=None, blk_per36: str=None, pts_per36: str=None, dreb_per36: str=None, pf_per36: str=None, stl_per36: str=None, fga_per36: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query per36 career post season"
    page: Page value for pagination
        fta_per36: Filter by fta_per36
        oreb_per36: Filter by oreb_per36
        reb_per36: Filter by reb_per36
        fg3a_per36: Filter by fg3a_per36
        min: Filter by min
        ast_per36: Filter by ast_per36
        fgm_per36: Filter by fgm_per36
        per_page: Number of resources to return per page for pagination (1 - 500)
        ftm_per36: Filter by ftm_per36
        fg3m_per36: Filter by fg3m_per36
        blk_per36: Filter by blk_per36
        pts_per36: Filter by pts_per36
        dreb_per36: Filter by dreb_per36
        pf_per36: Filter by pf_per36
        stl_per36: Filter by stl_per36
        fga_per36: Filter by fga_per36
        
    """
    url = f"https://nba-stats4.p.rapidapi.com/per36_career_post_season/"
    querystring = {}
    if page:
        querystring['page'] = page
    if fta_per36:
        querystring['fta_per36'] = fta_per36
    if oreb_per36:
        querystring['oreb_per36'] = oreb_per36
    if reb_per36:
        querystring['reb_per36'] = reb_per36
    if fg3a_per36:
        querystring['fg3a_per36'] = fg3a_per36
    if min:
        querystring['min'] = min
    if ast_per36:
        querystring['ast_per36'] = ast_per36
    if fgm_per36:
        querystring['fgm_per36'] = fgm_per36
    if per_page:
        querystring['per_page'] = per_page
    if ftm_per36:
        querystring['ftm_per36'] = ftm_per36
    if fg3m_per36:
        querystring['fg3m_per36'] = fg3m_per36
    if blk_per36:
        querystring['blk_per36'] = blk_per36
    if pts_per36:
        querystring['pts_per36'] = pts_per36
    if dreb_per36:
        querystring['dreb_per36'] = dreb_per36
    if pf_per36:
        querystring['pf_per36'] = pf_per36
    if stl_per36:
        querystring['stl_per36'] = stl_per36
    if fga_per36:
        querystring['fga_per36'] = fga_per36
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_per36_career_regular_season_by_id(player_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query per36 career regular season"
    
    """
    url = f"https://nba-stats4.p.rapidapi.com/per36_career_regular_season/{player_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_career_totals_regular_season_by_id(player_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query career totals regular season"
    
    """
    url = f"https://nba-stats4.p.rapidapi.com/career_totals_regular_season/{player_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_per_game_career_regular_season(ast_per_game: str=None, per_page: int=50, gp: str=None, fga_per_game: str=None, fgm_per_game: str=None, blk_per_game: str=None, dreb_per_game: str=None, ftm_per_game: str=None, pf_per_game: str=None, reb_per_game: str=None, fta_per_game: str=None, stl_per_game: str=None, oreb_per_game: str=None, fg3m_per_game: str=None, fg3a_per_game: str=None, pts_per_game: str=None, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query per game career regular season"
    ast_per_game: Filter by ast_per_game
        per_page: Number of resources to return per page for pagination (1 - 500)
        gp: Filter by gp
        fga_per_game: Filter by fga_per_game
        fgm_per_game: Filter by fgm_per_game
        blk_per_game: Filter by blk_per_game
        dreb_per_game: Filter by dreb_per_game
        ftm_per_game: Filter by ftm_per_game
        pf_per_game: Filter by pf_per_game
        reb_per_game: Filter by reb_per_game
        fta_per_game: Filter by fta_per_game
        stl_per_game: Filter by stl_per_game
        oreb_per_game: Filter by oreb_per_game
        fg3m_per_game: Filter by fg3m_per_game
        fg3a_per_game: Filter by fg3a_per_game
        pts_per_game: Filter by pts_per_game
        page: Page value for pagination
        
    """
    url = f"https://nba-stats4.p.rapidapi.com/per_game_career_regular_season/"
    querystring = {}
    if ast_per_game:
        querystring['ast_per_game'] = ast_per_game
    if per_page:
        querystring['per_page'] = per_page
    if gp:
        querystring['gp'] = gp
    if fga_per_game:
        querystring['fga_per_game'] = fga_per_game
    if fgm_per_game:
        querystring['fgm_per_game'] = fgm_per_game
    if blk_per_game:
        querystring['blk_per_game'] = blk_per_game
    if dreb_per_game:
        querystring['dreb_per_game'] = dreb_per_game
    if ftm_per_game:
        querystring['ftm_per_game'] = ftm_per_game
    if pf_per_game:
        querystring['pf_per_game'] = pf_per_game
    if reb_per_game:
        querystring['reb_per_game'] = reb_per_game
    if fta_per_game:
        querystring['fta_per_game'] = fta_per_game
    if stl_per_game:
        querystring['stl_per_game'] = stl_per_game
    if oreb_per_game:
        querystring['oreb_per_game'] = oreb_per_game
    if fg3m_per_game:
        querystring['fg3m_per_game'] = fg3m_per_game
    if fg3a_per_game:
        querystring['fg3a_per_game'] = fg3a_per_game
    if pts_per_game:
        querystring['pts_per_game'] = pts_per_game
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_per36_career_post_season_by_id(player_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query per36 career post season"
    
    """
    url = f"https://nba-stats4.p.rapidapi.com/per36_career_post_season/{player_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_players(first_name: str=None, page: int=1, full_name: str=None, last_name: str=None, is_active: str=None, per_page: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query players"
    first_name: Filter by first_name
        page: Page value for pagination
        full_name: Filter by full_name
        last_name: Filter by last_name
        is_active: Filter by is_active
        per_page: Number of resources to return per page for pagination (1 - 500)
        
    """
    url = f"https://nba-stats4.p.rapidapi.com/players/"
    querystring = {}
    if first_name:
        querystring['first_name'] = first_name
    if page:
        querystring['page'] = page
    if full_name:
        querystring['full_name'] = full_name
    if last_name:
        querystring['last_name'] = last_name
    if is_active:
        querystring['is_active'] = is_active
    if per_page:
        querystring['per_page'] = per_page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_per36_career_regular_season(reb_per36: str=None, fg3m_per36: str=None, per_page: int=50, fta_per36: str=None, min: str=None, fg3a_per36: str=None, oreb_per36: str=None, stl_per36: str=None, pts_per36: str=None, page: int=1, pf_per36: str=None, ast_per36: str=None, fgm_per36: str=None, blk_per36: str=None, ftm_per36: str=None, dreb_per36: str=None, fga_per36: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query per36 career regular season"
    reb_per36: Filter by reb_per36
        fg3m_per36: Filter by fg3m_per36
        per_page: Number of resources to return per page for pagination (1 - 500)
        fta_per36: Filter by fta_per36
        min: Filter by min
        fg3a_per36: Filter by fg3a_per36
        oreb_per36: Filter by oreb_per36
        stl_per36: Filter by stl_per36
        pts_per36: Filter by pts_per36
        page: Page value for pagination
        pf_per36: Filter by pf_per36
        ast_per36: Filter by ast_per36
        fgm_per36: Filter by fgm_per36
        blk_per36: Filter by blk_per36
        ftm_per36: Filter by ftm_per36
        dreb_per36: Filter by dreb_per36
        fga_per36: Filter by fga_per36
        
    """
    url = f"https://nba-stats4.p.rapidapi.com/per36_career_regular_season/"
    querystring = {}
    if reb_per36:
        querystring['reb_per36'] = reb_per36
    if fg3m_per36:
        querystring['fg3m_per36'] = fg3m_per36
    if per_page:
        querystring['per_page'] = per_page
    if fta_per36:
        querystring['fta_per36'] = fta_per36
    if min:
        querystring['min'] = min
    if fg3a_per36:
        querystring['fg3a_per36'] = fg3a_per36
    if oreb_per36:
        querystring['oreb_per36'] = oreb_per36
    if stl_per36:
        querystring['stl_per36'] = stl_per36
    if pts_per36:
        querystring['pts_per36'] = pts_per36
    if page:
        querystring['page'] = page
    if pf_per36:
        querystring['pf_per36'] = pf_per36
    if ast_per36:
        querystring['ast_per36'] = ast_per36
    if fgm_per36:
        querystring['fgm_per36'] = fgm_per36
    if blk_per36:
        querystring['blk_per36'] = blk_per36
    if ftm_per36:
        querystring['ftm_per36'] = ftm_per36
    if dreb_per36:
        querystring['dreb_per36'] = dreb_per36
    if fga_per36:
        querystring['fga_per36'] = fga_per36
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_per_game_career_regular_season_by_id(player_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query per game career regular season"
    
    """
    url = f"https://nba-stats4.p.rapidapi.com/per_game_career_regular_season/{player_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_per_game_career_post_season(per_page: int=50, fg3a_per_game: str=None, page: int=1, fga_per_game: str=None, reb_per_game: str=None, blk_per_game: str=None, stl_per_game: str=None, fta_per_game: str=None, pf_per_game: str=None, fgm_per_game: str=None, fg3m_per_game: str=None, gp: str=None, ftm_per_game: str=None, oreb_per_game: str=None, pts_per_game: str=None, dreb_per_game: str=None, ast_per_game: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query per game career post season"
    per_page: Number of resources to return per page for pagination (1 - 500)
        fg3a_per_game: Filter by fg3a_per_game
        page: Page value for pagination
        fga_per_game: Filter by fga_per_game
        reb_per_game: Filter by reb_per_game
        blk_per_game: Filter by blk_per_game
        stl_per_game: Filter by stl_per_game
        fta_per_game: Filter by fta_per_game
        pf_per_game: Filter by pf_per_game
        fgm_per_game: Filter by fgm_per_game
        fg3m_per_game: Filter by fg3m_per_game
        gp: Filter by gp
        ftm_per_game: Filter by ftm_per_game
        oreb_per_game: Filter by oreb_per_game
        pts_per_game: Filter by pts_per_game
        dreb_per_game: Filter by dreb_per_game
        ast_per_game: Filter by ast_per_game
        
    """
    url = f"https://nba-stats4.p.rapidapi.com/per_game_career_post_season/"
    querystring = {}
    if per_page:
        querystring['per_page'] = per_page
    if fg3a_per_game:
        querystring['fg3a_per_game'] = fg3a_per_game
    if page:
        querystring['page'] = page
    if fga_per_game:
        querystring['fga_per_game'] = fga_per_game
    if reb_per_game:
        querystring['reb_per_game'] = reb_per_game
    if blk_per_game:
        querystring['blk_per_game'] = blk_per_game
    if stl_per_game:
        querystring['stl_per_game'] = stl_per_game
    if fta_per_game:
        querystring['fta_per_game'] = fta_per_game
    if pf_per_game:
        querystring['pf_per_game'] = pf_per_game
    if fgm_per_game:
        querystring['fgm_per_game'] = fgm_per_game
    if fg3m_per_game:
        querystring['fg3m_per_game'] = fg3m_per_game
    if gp:
        querystring['gp'] = gp
    if ftm_per_game:
        querystring['ftm_per_game'] = ftm_per_game
    if oreb_per_game:
        querystring['oreb_per_game'] = oreb_per_game
    if pts_per_game:
        querystring['pts_per_game'] = pts_per_game
    if dreb_per_game:
        querystring['dreb_per_game'] = dreb_per_game
    if ast_per_game:
        querystring['ast_per_game'] = ast_per_game
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_career_totals_post_season(fg3m: str=None, fgm: str=None, page: int=1, dreb: str=None, tov: str=None, ft_pct: str=None, blk: str=None, ftm: str=None, team_id: str=None, min: str=None, pf: str=None, gp: str=None, fta: str=None, fga: str=None, league_id: str=None, per_page: int=50, fg_pct: str=None, pts: str=None, ast: str=None, fg3_pct: str=None, gs: str=None, stl: str=None, fg3a: str=None, reb: str=None, oreb: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query career totals post season"
    fg3m: Filter by fg3m
        fgm: Filter by fgm
        page: Page value for pagination
        dreb: Filter by dreb
        tov: Filter by tov
        ft_pct: Filter by ft_pct
        blk: Filter by blk
        ftm: Filter by ftm
        team_id: Filter by team_id
        min: Filter by min
        pf: Filter by pf
        gp: Filter by gp
        fta: Filter by fta
        fga: Filter by fga
        league_id: Filter by league_id
        per_page: Number of resources to return per page for pagination (1 - 500)
        fg_pct: Filter by fg_pct
        pts: Filter by pts
        ast: Filter by ast
        fg3_pct: Filter by fg3_pct
        gs: Filter by gs
        stl: Filter by stl
        fg3a: Filter by fg3a
        reb: Filter by reb
        oreb: Filter by oreb
        
    """
    url = f"https://nba-stats4.p.rapidapi.com/career_totals_post_season/"
    querystring = {}
    if fg3m:
        querystring['fg3m'] = fg3m
    if fgm:
        querystring['fgm'] = fgm
    if page:
        querystring['page'] = page
    if dreb:
        querystring['dreb'] = dreb
    if tov:
        querystring['tov'] = tov
    if ft_pct:
        querystring['ft_pct'] = ft_pct
    if blk:
        querystring['blk'] = blk
    if ftm:
        querystring['ftm'] = ftm
    if team_id:
        querystring['team_id'] = team_id
    if min:
        querystring['min'] = min
    if pf:
        querystring['pf'] = pf
    if gp:
        querystring['gp'] = gp
    if fta:
        querystring['fta'] = fta
    if fga:
        querystring['fga'] = fga
    if league_id:
        querystring['league_id'] = league_id
    if per_page:
        querystring['per_page'] = per_page
    if fg_pct:
        querystring['fg_pct'] = fg_pct
    if pts:
        querystring['pts'] = pts
    if ast:
        querystring['ast'] = ast
    if fg3_pct:
        querystring['fg3_pct'] = fg3_pct
    if gs:
        querystring['gs'] = gs
    if stl:
        querystring['stl'] = stl
    if fg3a:
        querystring['fg3a'] = fg3a
    if reb:
        querystring['reb'] = reb
    if oreb:
        querystring['oreb'] = oreb
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_per36_regular_season(pts_per36: str=None, blk_per36: str=None, stl_per36: str=None, reb_per36: str=None, fta_per36: str=None, per_page: int=50, dreb_per36: str=None, pf_per36: str=None, ast_per36: str=None, fg3a_per36: str=None, fga_per36: str=None, fg3m_per36: str=None, page: int=1, fgm_per36: str=None, min: str=None, ftm_per36: str=None, oreb_per36: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query per36 regular season"
    pts_per36: Filter by pts_per36
        blk_per36: Filter by blk_per36
        stl_per36: Filter by stl_per36
        reb_per36: Filter by reb_per36
        fta_per36: Filter by fta_per36
        per_page: Number of resources to return per page for pagination (1 - 500)
        dreb_per36: Filter by dreb_per36
        pf_per36: Filter by pf_per36
        ast_per36: Filter by ast_per36
        fg3a_per36: Filter by fg3a_per36
        fga_per36: Filter by fga_per36
        fg3m_per36: Filter by fg3m_per36
        page: Page value for pagination
        fgm_per36: Filter by fgm_per36
        min: Filter by min
        ftm_per36: Filter by ftm_per36
        oreb_per36: Filter by oreb_per36
        
    """
    url = f"https://nba-stats4.p.rapidapi.com/per36_regular_season/"
    querystring = {}
    if pts_per36:
        querystring['pts_per36'] = pts_per36
    if blk_per36:
        querystring['blk_per36'] = blk_per36
    if stl_per36:
        querystring['stl_per36'] = stl_per36
    if reb_per36:
        querystring['reb_per36'] = reb_per36
    if fta_per36:
        querystring['fta_per36'] = fta_per36
    if per_page:
        querystring['per_page'] = per_page
    if dreb_per36:
        querystring['dreb_per36'] = dreb_per36
    if pf_per36:
        querystring['pf_per36'] = pf_per36
    if ast_per36:
        querystring['ast_per36'] = ast_per36
    if fg3a_per36:
        querystring['fg3a_per36'] = fg3a_per36
    if fga_per36:
        querystring['fga_per36'] = fga_per36
    if fg3m_per36:
        querystring['fg3m_per36'] = fg3m_per36
    if page:
        querystring['page'] = page
    if fgm_per36:
        querystring['fgm_per36'] = fgm_per36
    if min:
        querystring['min'] = min
    if ftm_per36:
        querystring['ftm_per36'] = ftm_per36
    if oreb_per36:
        querystring['oreb_per36'] = oreb_per36
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_per_game_post_season_by_id(player_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query per game post season"
    
    """
    url = f"https://nba-stats4.p.rapidapi.com/per_game_post_season/{player_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_per_game_regular_season(pf_per_game: str=None, fg3a_per_game: str=None, blk_per_game: str=None, ftm_per_game: str=None, fgm_per_game: str=None, fg3m_per_game: str=None, pts_per_game: str=None, per_page: int=50, fta_per_game: str=None, stl_per_game: str=None, ast_per_game: str=None, page: int=1, dreb_per_game: str=None, oreb_per_game: str=None, fga_per_game: str=None, gp: str=None, reb_per_game: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query per game regular season"
    pf_per_game: Filter by pf_per_game
        fg3a_per_game: Filter by fg3a_per_game
        blk_per_game: Filter by blk_per_game
        ftm_per_game: Filter by ftm_per_game
        fgm_per_game: Filter by fgm_per_game
        fg3m_per_game: Filter by fg3m_per_game
        pts_per_game: Filter by pts_per_game
        per_page: Number of resources to return per page for pagination (1 - 500)
        fta_per_game: Filter by fta_per_game
        stl_per_game: Filter by stl_per_game
        ast_per_game: Filter by ast_per_game
        page: Page value for pagination
        dreb_per_game: Filter by dreb_per_game
        oreb_per_game: Filter by oreb_per_game
        fga_per_game: Filter by fga_per_game
        gp: Filter by gp
        reb_per_game: Filter by reb_per_game
        
    """
    url = f"https://nba-stats4.p.rapidapi.com/per_game_regular_season/"
    querystring = {}
    if pf_per_game:
        querystring['pf_per_game'] = pf_per_game
    if fg3a_per_game:
        querystring['fg3a_per_game'] = fg3a_per_game
    if blk_per_game:
        querystring['blk_per_game'] = blk_per_game
    if ftm_per_game:
        querystring['ftm_per_game'] = ftm_per_game
    if fgm_per_game:
        querystring['fgm_per_game'] = fgm_per_game
    if fg3m_per_game:
        querystring['fg3m_per_game'] = fg3m_per_game
    if pts_per_game:
        querystring['pts_per_game'] = pts_per_game
    if per_page:
        querystring['per_page'] = per_page
    if fta_per_game:
        querystring['fta_per_game'] = fta_per_game
    if stl_per_game:
        querystring['stl_per_game'] = stl_per_game
    if ast_per_game:
        querystring['ast_per_game'] = ast_per_game
    if page:
        querystring['page'] = page
    if dreb_per_game:
        querystring['dreb_per_game'] = dreb_per_game
    if oreb_per_game:
        querystring['oreb_per_game'] = oreb_per_game
    if fga_per_game:
        querystring['fga_per_game'] = fga_per_game
    if gp:
        querystring['gp'] = gp
    if reb_per_game:
        querystring['reb_per_game'] = reb_per_game
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_per36_post_season_by_id(player_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query per36 post season"
    
    """
    url = f"https://nba-stats4.p.rapidapi.com/per36_post_season/{player_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_rankings_post_season_by_id(player_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query rankings post season"
    
    """
    url = f"https://nba-stats4.p.rapidapi.com/rankings_post_season/{player_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_rankings_regular_season_by_id(player_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query rankings regular season"
    
    """
    url = f"https://nba-stats4.p.rapidapi.com/rankings_regular_season/{player_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_season_totals_post_season(player_age: str=None, stl: str=None, ast: str=None, team_abbreviation: str=None, reb: str=None, ftm: str=None, fg3m: str=None, ft_pct: str=None, team_id: str=None, league_id: str=None, gs: str=None, fga: str=None, page: int=1, pts: str=None, blk: str=None, per_page: int=50, gp: str=None, fgm: str=None, dreb: str=None, fg3a: str=None, oreb: str=None, fg_pct: str=None, pf: str=None, min: str=None, fta: str=None, fg3_pct: str=None, tov: str=None, season_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query season totals post season"
    player_age: Filter by player_age
        stl: Filter by stl
        ast: Filter by ast
        team_abbreviation: Filter by team_abbreviation
        reb: Filter by reb
        ftm: Filter by ftm
        fg3m: Filter by fg3m
        ft_pct: Filter by ft_pct
        team_id: Filter by team_id
        league_id: Filter by league_id
        gs: Filter by gs
        fga: Filter by fga
        page: Page value for pagination
        pts: Filter by pts
        blk: Filter by blk
        per_page: Number of resources to return per page for pagination (1 - 500)
        gp: Filter by gp
        fgm: Filter by fgm
        dreb: Filter by dreb
        fg3a: Filter by fg3a
        oreb: Filter by oreb
        fg_pct: Filter by fg_pct
        pf: Filter by pf
        min: Filter by min
        fta: Filter by fta
        fg3_pct: Filter by fg3_pct
        tov: Filter by tov
        season_id: Filter by season_id
        
    """
    url = f"https://nba-stats4.p.rapidapi.com/season_totals_post_season/"
    querystring = {}
    if player_age:
        querystring['player_age'] = player_age
    if stl:
        querystring['stl'] = stl
    if ast:
        querystring['ast'] = ast
    if team_abbreviation:
        querystring['team_abbreviation'] = team_abbreviation
    if reb:
        querystring['reb'] = reb
    if ftm:
        querystring['ftm'] = ftm
    if fg3m:
        querystring['fg3m'] = fg3m
    if ft_pct:
        querystring['ft_pct'] = ft_pct
    if team_id:
        querystring['team_id'] = team_id
    if league_id:
        querystring['league_id'] = league_id
    if gs:
        querystring['gs'] = gs
    if fga:
        querystring['fga'] = fga
    if page:
        querystring['page'] = page
    if pts:
        querystring['pts'] = pts
    if blk:
        querystring['blk'] = blk
    if per_page:
        querystring['per_page'] = per_page
    if gp:
        querystring['gp'] = gp
    if fgm:
        querystring['fgm'] = fgm
    if dreb:
        querystring['dreb'] = dreb
    if fg3a:
        querystring['fg3a'] = fg3a
    if oreb:
        querystring['oreb'] = oreb
    if fg_pct:
        querystring['fg_pct'] = fg_pct
    if pf:
        querystring['pf'] = pf
    if min:
        querystring['min'] = min
    if fta:
        querystring['fta'] = fta
    if fg3_pct:
        querystring['fg3_pct'] = fg3_pct
    if tov:
        querystring['tov'] = tov
    if season_id:
        querystring['season_id'] = season_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_per36_post_season(dreb_per36: str=None, per_page: int=50, stl_per36: str=None, fgm_per36: str=None, page: int=1, ftm_per36: str=None, pts_per36: str=None, reb_per36: str=None, fta_per36: str=None, fg3a_per36: str=None, fg3m_per36: str=None, fga_per36: str=None, ast_per36: str=None, min: str=None, pf_per36: str=None, oreb_per36: str=None, blk_per36: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query per36 post season"
    dreb_per36: Filter by dreb_per36
        per_page: Number of resources to return per page for pagination (1 - 500)
        stl_per36: Filter by stl_per36
        fgm_per36: Filter by fgm_per36
        page: Page value for pagination
        ftm_per36: Filter by ftm_per36
        pts_per36: Filter by pts_per36
        reb_per36: Filter by reb_per36
        fta_per36: Filter by fta_per36
        fg3a_per36: Filter by fg3a_per36
        fg3m_per36: Filter by fg3m_per36
        fga_per36: Filter by fga_per36
        ast_per36: Filter by ast_per36
        min: Filter by min
        pf_per36: Filter by pf_per36
        oreb_per36: Filter by oreb_per36
        blk_per36: Filter by blk_per36
        
    """
    url = f"https://nba-stats4.p.rapidapi.com/per36_post_season/"
    querystring = {}
    if dreb_per36:
        querystring['dreb_per36'] = dreb_per36
    if per_page:
        querystring['per_page'] = per_page
    if stl_per36:
        querystring['stl_per36'] = stl_per36
    if fgm_per36:
        querystring['fgm_per36'] = fgm_per36
    if page:
        querystring['page'] = page
    if ftm_per36:
        querystring['ftm_per36'] = ftm_per36
    if pts_per36:
        querystring['pts_per36'] = pts_per36
    if reb_per36:
        querystring['reb_per36'] = reb_per36
    if fta_per36:
        querystring['fta_per36'] = fta_per36
    if fg3a_per36:
        querystring['fg3a_per36'] = fg3a_per36
    if fg3m_per36:
        querystring['fg3m_per36'] = fg3m_per36
    if fga_per36:
        querystring['fga_per36'] = fga_per36
    if ast_per36:
        querystring['ast_per36'] = ast_per36
    if min:
        querystring['min'] = min
    if pf_per36:
        querystring['pf_per36'] = pf_per36
    if oreb_per36:
        querystring['oreb_per36'] = oreb_per36
    if blk_per36:
        querystring['blk_per36'] = blk_per36
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_players_by_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query players"
    
    """
    url = f"https://nba-stats4.p.rapidapi.com/players/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_per_game_regular_season_by_id(player_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query per game regular season"
    
    """
    url = f"https://nba-stats4.p.rapidapi.com/per_game_regular_season/{player_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_rankings_regular_season(rank_fg3_pct: str=None, rank_blk: str=None, rank_ast: str=None, rank_min: str=None, rank_reb: str=None, rank_pts: str=None, gp: str=None, rank_fg3a: str=None, page: int=1, team_id: str=None, rank_eff: str=None, rank_dreb: str=None, team_abbreviation: str=None, season_id: str=None, player_age: str=None, rank_fta: str=None, rank_fg_pct: str=None, rank_fg3m: str=None, rank_oreb: str=None, rank_fga: str=None, per_page: int=50, gs: str=None, league_id: str=None, rank_stl: str=None, rank_fgm: str=None, rank_ftm: str=None, rank_tov: str=None, rank_ft_pct: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query rankings regular season"
    rank_fg3_pct: Filter by rank_fg3_pct
        rank_blk: Filter by rank_blk
        rank_ast: Filter by rank_ast
        rank_min: Filter by rank_min
        rank_reb: Filter by rank_reb
        rank_pts: Filter by rank_pts
        gp: Filter by gp
        rank_fg3a: Filter by rank_fg3a
        page: Page value for pagination
        team_id: Filter by team_id
        rank_eff: Filter by rank_eff
        rank_dreb: Filter by rank_dreb
        team_abbreviation: Filter by team_abbreviation
        season_id: Filter by season_id
        player_age: Filter by player_age
        rank_fta: Filter by rank_fta
        rank_fg_pct: Filter by rank_fg_pct
        rank_fg3m: Filter by rank_fg3m
        rank_oreb: Filter by rank_oreb
        rank_fga: Filter by rank_fga
        per_page: Number of resources to return per page for pagination (1 - 500)
        gs: Filter by gs
        league_id: Filter by league_id
        rank_stl: Filter by rank_stl
        rank_fgm: Filter by rank_fgm
        rank_ftm: Filter by rank_ftm
        rank_tov: Filter by rank_tov
        rank_ft_pct: Filter by rank_ft_pct
        
    """
    url = f"https://nba-stats4.p.rapidapi.com/rankings_regular_season/"
    querystring = {}
    if rank_fg3_pct:
        querystring['rank_fg3_pct'] = rank_fg3_pct
    if rank_blk:
        querystring['rank_blk'] = rank_blk
    if rank_ast:
        querystring['rank_ast'] = rank_ast
    if rank_min:
        querystring['rank_min'] = rank_min
    if rank_reb:
        querystring['rank_reb'] = rank_reb
    if rank_pts:
        querystring['rank_pts'] = rank_pts
    if gp:
        querystring['gp'] = gp
    if rank_fg3a:
        querystring['rank_fg3a'] = rank_fg3a
    if page:
        querystring['page'] = page
    if team_id:
        querystring['team_id'] = team_id
    if rank_eff:
        querystring['rank_eff'] = rank_eff
    if rank_dreb:
        querystring['rank_dreb'] = rank_dreb
    if team_abbreviation:
        querystring['team_abbreviation'] = team_abbreviation
    if season_id:
        querystring['season_id'] = season_id
    if player_age:
        querystring['player_age'] = player_age
    if rank_fta:
        querystring['rank_fta'] = rank_fta
    if rank_fg_pct:
        querystring['rank_fg_pct'] = rank_fg_pct
    if rank_fg3m:
        querystring['rank_fg3m'] = rank_fg3m
    if rank_oreb:
        querystring['rank_oreb'] = rank_oreb
    if rank_fga:
        querystring['rank_fga'] = rank_fga
    if per_page:
        querystring['per_page'] = per_page
    if gs:
        querystring['gs'] = gs
    if league_id:
        querystring['league_id'] = league_id
    if rank_stl:
        querystring['rank_stl'] = rank_stl
    if rank_fgm:
        querystring['rank_fgm'] = rank_fgm
    if rank_ftm:
        querystring['rank_ftm'] = rank_ftm
    if rank_tov:
        querystring['rank_tov'] = rank_tov
    if rank_ft_pct:
        querystring['rank_ft_pct'] = rank_ft_pct
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_career_totals_allstar_by_id(player_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query career totals allstar"
    
    """
    url = f"https://nba-stats4.p.rapidapi.com/career_totals_allstar/{player_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_per_game_career_post_season_by_id(player_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query per game career post season"
    
    """
    url = f"https://nba-stats4.p.rapidapi.com/per_game_career_post_season/{player_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_per36_regular_season_by_id(player_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query per36 regular season"
    
    """
    url = f"https://nba-stats4.p.rapidapi.com/per36_regular_season/{player_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_season_totals_allstar_by_id(player_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query season totals allstar"
    
    """
    url = f"https://nba-stats4.p.rapidapi.com/season_totals_allstar/{player_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_rankings_post_season(rank_fgm: str=None, rank_oreb: str=None, rank_fg_pct: str=None, rank_pts: str=None, rank_fg3_pct: str=None, gs: str=None, rank_blk: str=None, rank_ast: str=None, rank_fg3a: str=None, rank_reb: str=None, rank_ft_pct: str=None, rank_stl: str=None, per_page: int=50, rank_ftm: str=None, rank_eff: str=None, gp: str=None, rank_fga: str=None, team_id: str=None, season_id: str=None, player_age: str=None, team_abbreviation: str=None, rank_tov: str=None, page: int=1, rank_fg3m: str=None, rank_dreb: str=None, league_id: str=None, rank_fta: str=None, rank_min: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query rankings post season"
    rank_fgm: Filter by rank_fgm
        rank_oreb: Filter by rank_oreb
        rank_fg_pct: Filter by rank_fg_pct
        rank_pts: Filter by rank_pts
        rank_fg3_pct: Filter by rank_fg3_pct
        gs: Filter by gs
        rank_blk: Filter by rank_blk
        rank_ast: Filter by rank_ast
        rank_fg3a: Filter by rank_fg3a
        rank_reb: Filter by rank_reb
        rank_ft_pct: Filter by rank_ft_pct
        rank_stl: Filter by rank_stl
        per_page: Number of resources to return per page for pagination (1 - 500)
        rank_ftm: Filter by rank_ftm
        rank_eff: Filter by rank_eff
        gp: Filter by gp
        rank_fga: Filter by rank_fga
        team_id: Filter by team_id
        season_id: Filter by season_id
        player_age: Filter by player_age
        team_abbreviation: Filter by team_abbreviation
        rank_tov: Filter by rank_tov
        page: Page value for pagination
        rank_fg3m: Filter by rank_fg3m
        rank_dreb: Filter by rank_dreb
        league_id: Filter by league_id
        rank_fta: Filter by rank_fta
        rank_min: Filter by rank_min
        
    """
    url = f"https://nba-stats4.p.rapidapi.com/rankings_post_season/"
    querystring = {}
    if rank_fgm:
        querystring['rank_fgm'] = rank_fgm
    if rank_oreb:
        querystring['rank_oreb'] = rank_oreb
    if rank_fg_pct:
        querystring['rank_fg_pct'] = rank_fg_pct
    if rank_pts:
        querystring['rank_pts'] = rank_pts
    if rank_fg3_pct:
        querystring['rank_fg3_pct'] = rank_fg3_pct
    if gs:
        querystring['gs'] = gs
    if rank_blk:
        querystring['rank_blk'] = rank_blk
    if rank_ast:
        querystring['rank_ast'] = rank_ast
    if rank_fg3a:
        querystring['rank_fg3a'] = rank_fg3a
    if rank_reb:
        querystring['rank_reb'] = rank_reb
    if rank_ft_pct:
        querystring['rank_ft_pct'] = rank_ft_pct
    if rank_stl:
        querystring['rank_stl'] = rank_stl
    if per_page:
        querystring['per_page'] = per_page
    if rank_ftm:
        querystring['rank_ftm'] = rank_ftm
    if rank_eff:
        querystring['rank_eff'] = rank_eff
    if gp:
        querystring['gp'] = gp
    if rank_fga:
        querystring['rank_fga'] = rank_fga
    if team_id:
        querystring['team_id'] = team_id
    if season_id:
        querystring['season_id'] = season_id
    if player_age:
        querystring['player_age'] = player_age
    if team_abbreviation:
        querystring['team_abbreviation'] = team_abbreviation
    if rank_tov:
        querystring['rank_tov'] = rank_tov
    if page:
        querystring['page'] = page
    if rank_fg3m:
        querystring['rank_fg3m'] = rank_fg3m
    if rank_dreb:
        querystring['rank_dreb'] = rank_dreb
    if league_id:
        querystring['league_id'] = league_id
    if rank_fta:
        querystring['rank_fta'] = rank_fta
    if rank_min:
        querystring['rank_min'] = rank_min
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_season_totals_regular_season(ft_pct: str=None, gp: str=None, gs: str=None, team_id: str=None, fg3a: str=None, fg3_pct: str=None, reb: str=None, tov: str=None, ast: str=None, league_id: str=None, fg3m: str=None, per_page: int=50, fta: str=None, fg_pct: str=None, stl: str=None, pf: str=None, page: int=1, oreb: str=None, ftm: str=None, player_age: str=None, min: str=None, fga: str=None, team_abbreviation: str=None, fgm: str=None, pts: str=None, blk: str=None, dreb: str=None, season_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query season totals regular season"
    ft_pct: Filter by ft_pct
        gp: Filter by gp
        gs: Filter by gs
        team_id: Filter by team_id
        fg3a: Filter by fg3a
        fg3_pct: Filter by fg3_pct
        reb: Filter by reb
        tov: Filter by tov
        ast: Filter by ast
        league_id: Filter by league_id
        fg3m: Filter by fg3m
        per_page: Number of resources to return per page for pagination (1 - 500)
        fta: Filter by fta
        fg_pct: Filter by fg_pct
        stl: Filter by stl
        pf: Filter by pf
        page: Page value for pagination
        oreb: Filter by oreb
        ftm: Filter by ftm
        player_age: Filter by player_age
        min: Filter by min
        fga: Filter by fga
        team_abbreviation: Filter by team_abbreviation
        fgm: Filter by fgm
        pts: Filter by pts
        blk: Filter by blk
        dreb: Filter by dreb
        season_id: Filter by season_id
        
    """
    url = f"https://nba-stats4.p.rapidapi.com/season_totals_regular_season/"
    querystring = {}
    if ft_pct:
        querystring['ft_pct'] = ft_pct
    if gp:
        querystring['gp'] = gp
    if gs:
        querystring['gs'] = gs
    if team_id:
        querystring['team_id'] = team_id
    if fg3a:
        querystring['fg3a'] = fg3a
    if fg3_pct:
        querystring['fg3_pct'] = fg3_pct
    if reb:
        querystring['reb'] = reb
    if tov:
        querystring['tov'] = tov
    if ast:
        querystring['ast'] = ast
    if league_id:
        querystring['league_id'] = league_id
    if fg3m:
        querystring['fg3m'] = fg3m
    if per_page:
        querystring['per_page'] = per_page
    if fta:
        querystring['fta'] = fta
    if fg_pct:
        querystring['fg_pct'] = fg_pct
    if stl:
        querystring['stl'] = stl
    if pf:
        querystring['pf'] = pf
    if page:
        querystring['page'] = page
    if oreb:
        querystring['oreb'] = oreb
    if ftm:
        querystring['ftm'] = ftm
    if player_age:
        querystring['player_age'] = player_age
    if min:
        querystring['min'] = min
    if fga:
        querystring['fga'] = fga
    if team_abbreviation:
        querystring['team_abbreviation'] = team_abbreviation
    if fgm:
        querystring['fgm'] = fgm
    if pts:
        querystring['pts'] = pts
    if blk:
        querystring['blk'] = blk
    if dreb:
        querystring['dreb'] = dreb
    if season_id:
        querystring['season_id'] = season_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_per_game_post_season(reb_per_game: str=None, ftm_per_game: str=None, fg3m_per_game: str=None, pts_per_game: str=None, fga_per_game: str=None, per_page: int=50, oreb_per_game: str=None, gp: str=None, blk_per_game: str=None, fgm_per_game: str=None, page: int=1, fg3a_per_game: str=None, stl_per_game: str=None, pf_per_game: str=None, dreb_per_game: str=None, fta_per_game: str=None, ast_per_game: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query per game post season"
    reb_per_game: Filter by reb_per_game
        ftm_per_game: Filter by ftm_per_game
        fg3m_per_game: Filter by fg3m_per_game
        pts_per_game: Filter by pts_per_game
        fga_per_game: Filter by fga_per_game
        per_page: Number of resources to return per page for pagination (1 - 500)
        oreb_per_game: Filter by oreb_per_game
        gp: Filter by gp
        blk_per_game: Filter by blk_per_game
        fgm_per_game: Filter by fgm_per_game
        page: Page value for pagination
        fg3a_per_game: Filter by fg3a_per_game
        stl_per_game: Filter by stl_per_game
        pf_per_game: Filter by pf_per_game
        dreb_per_game: Filter by dreb_per_game
        fta_per_game: Filter by fta_per_game
        ast_per_game: Filter by ast_per_game
        
    """
    url = f"https://nba-stats4.p.rapidapi.com/per_game_post_season/"
    querystring = {}
    if reb_per_game:
        querystring['reb_per_game'] = reb_per_game
    if ftm_per_game:
        querystring['ftm_per_game'] = ftm_per_game
    if fg3m_per_game:
        querystring['fg3m_per_game'] = fg3m_per_game
    if pts_per_game:
        querystring['pts_per_game'] = pts_per_game
    if fga_per_game:
        querystring['fga_per_game'] = fga_per_game
    if per_page:
        querystring['per_page'] = per_page
    if oreb_per_game:
        querystring['oreb_per_game'] = oreb_per_game
    if gp:
        querystring['gp'] = gp
    if blk_per_game:
        querystring['blk_per_game'] = blk_per_game
    if fgm_per_game:
        querystring['fgm_per_game'] = fgm_per_game
    if page:
        querystring['page'] = page
    if fg3a_per_game:
        querystring['fg3a_per_game'] = fg3a_per_game
    if stl_per_game:
        querystring['stl_per_game'] = stl_per_game
    if pf_per_game:
        querystring['pf_per_game'] = pf_per_game
    if dreb_per_game:
        querystring['dreb_per_game'] = dreb_per_game
    if fta_per_game:
        querystring['fta_per_game'] = fta_per_game
    if ast_per_game:
        querystring['ast_per_game'] = ast_per_game
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_career_totals_post_season_by_id(player_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query career totals post season"
    
    """
    url = f"https://nba-stats4.p.rapidapi.com/career_totals_post_season/{player_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_teams(per_page: int=50, state: str=None, full_name: str=None, page: int=1, year_founded: str=None, abbreviation: str=None, city: str=None, nickname: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query teams"
    per_page: Number of resources to return per page for pagination (1 - 500)
        state: Filter by state
        full_name: Filter by full_name
        page: Page value for pagination
        year_founded: Filter by year_founded
        abbreviation: Filter by abbreviation
        city: Filter by city
        nickname: Filter by nickname
        
    """
    url = f"https://nba-stats4.p.rapidapi.com/teams/"
    querystring = {}
    if per_page:
        querystring['per_page'] = per_page
    if state:
        querystring['state'] = state
    if full_name:
        querystring['full_name'] = full_name
    if page:
        querystring['page'] = page
    if year_founded:
        querystring['year_founded'] = year_founded
    if abbreviation:
        querystring['abbreviation'] = abbreviation
    if city:
        querystring['city'] = city
    if nickname:
        querystring['nickname'] = nickname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_season_totals_allstar(season_id: str=None, page: int=1, pts: str=None, team_abbreviation: str=None, fg3_pct: str=None, gs: str=None, dreb: str=None, gp: str=None, ftm: str=None, fga: str=None, fg_pct: str=None, stl: str=None, blk: str=None, league_id: str=None, pf: str=None, tov: str=None, player_age: str=None, reb: str=None, fgm: str=None, fg3m: str=None, team_id: str=None, oreb: str=None, fg3a: str=None, ft_pct: str=None, min: str=None, per_page: int=50, fta: str=None, ast: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query season totals allstar"
    season_id: Filter by season_id
        page: Page value for pagination
        pts: Filter by pts
        team_abbreviation: Filter by team_abbreviation
        fg3_pct: Filter by fg3_pct
        gs: Filter by gs
        dreb: Filter by dreb
        gp: Filter by gp
        ftm: Filter by ftm
        fga: Filter by fga
        fg_pct: Filter by fg_pct
        stl: Filter by stl
        blk: Filter by blk
        league_id: Filter by league_id
        pf: Filter by pf
        tov: Filter by tov
        player_age: Filter by player_age
        reb: Filter by reb
        fgm: Filter by fgm
        fg3m: Filter by fg3m
        team_id: Filter by team_id
        oreb: Filter by oreb
        fg3a: Filter by fg3a
        ft_pct: Filter by ft_pct
        min: Filter by min
        per_page: Number of resources to return per page for pagination (1 - 500)
        fta: Filter by fta
        ast: Filter by ast
        
    """
    url = f"https://nba-stats4.p.rapidapi.com/season_totals_allstar/"
    querystring = {}
    if season_id:
        querystring['season_id'] = season_id
    if page:
        querystring['page'] = page
    if pts:
        querystring['pts'] = pts
    if team_abbreviation:
        querystring['team_abbreviation'] = team_abbreviation
    if fg3_pct:
        querystring['fg3_pct'] = fg3_pct
    if gs:
        querystring['gs'] = gs
    if dreb:
        querystring['dreb'] = dreb
    if gp:
        querystring['gp'] = gp
    if ftm:
        querystring['ftm'] = ftm
    if fga:
        querystring['fga'] = fga
    if fg_pct:
        querystring['fg_pct'] = fg_pct
    if stl:
        querystring['stl'] = stl
    if blk:
        querystring['blk'] = blk
    if league_id:
        querystring['league_id'] = league_id
    if pf:
        querystring['pf'] = pf
    if tov:
        querystring['tov'] = tov
    if player_age:
        querystring['player_age'] = player_age
    if reb:
        querystring['reb'] = reb
    if fgm:
        querystring['fgm'] = fgm
    if fg3m:
        querystring['fg3m'] = fg3m
    if team_id:
        querystring['team_id'] = team_id
    if oreb:
        querystring['oreb'] = oreb
    if fg3a:
        querystring['fg3a'] = fg3a
    if ft_pct:
        querystring['ft_pct'] = ft_pct
    if min:
        querystring['min'] = min
    if per_page:
        querystring['per_page'] = per_page
    if fta:
        querystring['fta'] = fta
    if ast:
        querystring['ast'] = ast
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_season_totals_regular_season_by_id(player_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query season totals regular season"
    
    """
    url = f"https://nba-stats4.p.rapidapi.com/season_totals_regular_season/{player_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_teams_by_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query teams"
    
    """
    url = f"https://nba-stats4.p.rapidapi.com/teams/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_season_totals_post_season_by_id(player_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query season totals post season"
    
    """
    url = f"https://nba-stats4.p.rapidapi.com/season_totals_post_season/{player_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_career_totals_regular_season(ft_pct: str=None, ftm: str=None, fta: str=None, fga: str=None, fg3m: str=None, fgm: str=None, gs: str=None, team_id: str=None, stl: str=None, dreb: str=None, fg_pct: str=None, reb: str=None, page: int=1, gp: str=None, min: str=None, pts: str=None, oreb: str=None, per_page: int=50, fg3a: str=None, pf: str=None, league_id: str=None, fg3_pct: str=None, blk: str=None, tov: str=None, ast: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to query career totals regular season"
    ft_pct: Filter by ft_pct
        ftm: Filter by ftm
        fta: Filter by fta
        fga: Filter by fga
        fg3m: Filter by fg3m
        fgm: Filter by fgm
        gs: Filter by gs
        team_id: Filter by team_id
        stl: Filter by stl
        dreb: Filter by dreb
        fg_pct: Filter by fg_pct
        reb: Filter by reb
        page: Page value for pagination
        gp: Filter by gp
        min: Filter by min
        pts: Filter by pts
        oreb: Filter by oreb
        per_page: Number of resources to return per page for pagination (1 - 500)
        fg3a: Filter by fg3a
        pf: Filter by pf
        league_id: Filter by league_id
        fg3_pct: Filter by fg3_pct
        blk: Filter by blk
        tov: Filter by tov
        ast: Filter by ast
        
    """
    url = f"https://nba-stats4.p.rapidapi.com/career_totals_regular_season/"
    querystring = {}
    if ft_pct:
        querystring['ft_pct'] = ft_pct
    if ftm:
        querystring['ftm'] = ftm
    if fta:
        querystring['fta'] = fta
    if fga:
        querystring['fga'] = fga
    if fg3m:
        querystring['fg3m'] = fg3m
    if fgm:
        querystring['fgm'] = fgm
    if gs:
        querystring['gs'] = gs
    if team_id:
        querystring['team_id'] = team_id
    if stl:
        querystring['stl'] = stl
    if dreb:
        querystring['dreb'] = dreb
    if fg_pct:
        querystring['fg_pct'] = fg_pct
    if reb:
        querystring['reb'] = reb
    if page:
        querystring['page'] = page
    if gp:
        querystring['gp'] = gp
    if min:
        querystring['min'] = min
    if pts:
        querystring['pts'] = pts
    if oreb:
        querystring['oreb'] = oreb
    if per_page:
        querystring['per_page'] = per_page
    if fg3a:
        querystring['fg3a'] = fg3a
    if pf:
        querystring['pf'] = pf
    if league_id:
        querystring['league_id'] = league_id
    if fg3_pct:
        querystring['fg3_pct'] = fg3_pct
    if blk:
        querystring['blk'] = blk
    if tov:
        querystring['tov'] = tov
    if ast:
        querystring['ast'] = ast
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-stats4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

