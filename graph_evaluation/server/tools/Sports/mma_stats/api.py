import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ufc_fight_night_dawson_vs_green_october_07_2023(offset: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**UFC Fight Night: Dawson vs. Green**.    
		  .Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/October_07_2023"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ufc_fight_night_yusuff_vs_barboza_october_14_2023(offset: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**UFC Fight Night: Yusuff vs. Barboza**.    
		  .Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/October_14_2023"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ufc_294_makhachev_vs_oliveira_2_october_21_2023(offset: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**UFC 294: Makhachev vs. Oliveira 2**.    
		  .Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/October_21_2023"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ufc_fight_night_blaydes_vs_almeida_november_04_2023(limit: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**UFC Fight Night: Blaydes vs. Almeida**.    
		  .Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/November_04_2023"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ufc_295_jones_vs_miocic_november_11_2023(offset: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**UFC 295: Jones vs. Miocic**.    
		  .Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/November_11_2023"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ufc_fight_night_allen_vs_craig_november_18_2023(limit: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**UFC Fight Night: Allen vs. Craig**.    
		  .Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/November_18_2023"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ufc_296_edwards_vs_covington_december_16_2023(limit: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**UFC 296: Edwards vs. Covington.**.    
		  .Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/December_16_2023"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bruna_brasil_vs_shauna_bannon(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Bruna Brasil  vs.  Shauna Bannon**                                                                                            
		  Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/July_22_2023/matchup/13"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def chris_duncan_vs_yanal_ashmouz(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Chris Duncan  vs.  Yanal Ashmouz**                                                                                            
		  Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/July_22_2023/matchup/13"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def jafel_filho_vs_daniel_barez(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Jafel Filho  vs.  Daniel Barez**                                                                                            
		  Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/July_22_2023/matchup/14"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ketlen_vieira_vs_pannie_kianzad(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Ketlen Vieira  vs.  Pannie Kianzad                                                                                                      
		  Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/July_22_2023/matchup/12"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bryan_barberena_vs_makhmud_muradov(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Bryan Barberena  vs.  Makhmud Muradov                                                                                                            
		  Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/July_22_2023/matchup/11"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mick_parkin_vs_jamal_pogues(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Mick Parkin  vs.  Jamal Pogues**                                                                                                              
		  Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/July_22_2023/matchup/10"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def marc_diakiese_vs_joel_alvarez(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Marc Diakiese  vs.  Joel Alvarez**                                                                                                              
		  Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/July_22_2023/matchup/9"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def danny_roberts_vs_jonny_parsons(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Danny Roberts  vs.  Jonny Parsons**                                                                                                              
		  Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/July_22_2023/matchup/8"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def davey_grant_vs_daniel_marcos(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Davey Grant  vs.  Daniel Marcos**                                                                                                              
		  Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/July_22_2023/matchup/7"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lerone_murphy_vs_josh_culibao(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Lerone Murphy  vs.  Josh Culibao**                                                                                                              
		  Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/July_22_2023/matchup/6"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def jai_herbert_vs_fares_ziam(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Jai Herbert  vs.  Fares Ziam**                                                                                                              
		  Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/July_22_2023/matchup/5"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def andr_muniz_vs_paul_craig(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**André Muniz  vs.  Paul Craig**
		Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/July_22_2023/matchup/4"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def molly_mccann_vs_j_stoliarenko(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Molly McCann  vs.  J. Stoliarenko                                                                                                 
		  Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/July_22_2023/matchup/2"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nathaniel_wood_vs_andre_fili(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Nathaniel Wood  vs.  Andre Fili**
		Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/July_22_2023/matchup/3"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tom_aspinall_vs_marcin_tybura(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Tom Aspinall vs Marcin Tybura
		Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/July_22_2023/matchup/1"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ufc_fight_night_aspinall_vs_tybura_july_22_2023(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**UFC Fight Night: Aspinall vs. Tybura (July 22, 2023)**
		Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/July_22_2023"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ufc_fight_night_fiziev_vs_gamrot(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "** UFC Fight Night: Fiziev vs. Gamrot**
		Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/September_23_2023"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ufc_fight_night_grasso_vs_shevchenko_2(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "** UFC Fight Night: Grasso vs. Shevchenko 2**
		Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/September_16_2023"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ufc_293_adesanya_vs_strickland(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "** UFC 293: Adesanya vs. Strickland**
		Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/September_09_2023"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ufc_fight_night_gane_vs_spivac(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "** UFC Fight Night: Gane vs. Spivac**
		Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/September_02_2023"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ufc_fight_night_luque_vs_dos_anjos(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**UFC Fight Night: Luque vs. Dos Anjos**
		Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/August_12_2023"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ufc_fight_night_holloway_vs_the_korean_zombie(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**UFC Fight Night: Holloway vs. The Korean Zombie**
		Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/August_26_2023/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ufc_fight_night_holm_vs_bueno_silva(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**UFC Fight Night: Holm vs. Bueno Silva**.         
		Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/July_15_2023"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ufc_292_sterling_vs_o_malley(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**UFC 292: Sterling vs. O'Malley**
		Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/August_19_2023/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sterling_vs_o_malley(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "** Sterling vs. O'Malley**
		Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/August_19_2023/matchup/1"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ufc_290_volkanovski_vs_rodriguez_july_08_2023(offset: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**UFC 290: Volkanovski vs. Rodriguez**.                                                          
		 Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/July_08_2023"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ufc_fight_night_strickland_vs_magomedov_july_01_2023(limit: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**UFC Fight Night: Strickland vs. Magomedov **.                                                          
		 Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/July_01_2023"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ufc_fight_night_rozenstruik_vs_almeida_may_13_2023(offset: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Get details to UFC Fight Night: Rozenstruik vs. Almeida**.                                                          
		 Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/May_13_2023"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ufc_fight_night_sandhagen_vs_font(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**UFC Fight Night: Sandhagen vs. Font**
		Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/August_05_2023"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ufc_291_poirier_vs_gaethje_2(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**UFC 291: Poirier vs. Gaethje 2**
		Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/July_29_2023"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ufc_fight_night_emmett_vs_topuriar_june_24_2023(limit: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**UFC Fight Night: Emmett vs. Topuria**.                                                          
		 Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/June_24_2023"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ufc_fight_night_vettori_vs_cannonier_june_17_2023(offset: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**UFC Fight Night: Vettori vs. Cannonier**.                                                          
		 Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/June_17_2023"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ufc_289_nunes_vs_aldana_june_10_2023(offset: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Get details to  UFC 289: Nunes vs. Aldana**.                                                          
		 Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/June_10_2023"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ufc_fight_night_dern_vs_hill_may_20_2023(offset: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Get details to UFC Fight Night: Dern vs. Hill**.                                                          
		 Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/May_20_2023"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ufc_288_sterling_vs_cejudo_may_06_2023(offset: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Get details to  UFC 288: Sterling vs. Cejudo**.                                                          
		 Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/May_06_2023"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ufc_fight_night_song_vs_simon_april_28_2023(offset: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Get details to UFC Fight Night: Song vs. Simon**.                                                          
		 Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/April_28_2023"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ufc_fight_night_pavlovich_vs_blaydes_april_22_2023(offset: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Get details to UFC Fight Night: Pavlovich vs. Blaydes**.                                                          
		 Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/April_22_2023"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ufc_fight_night_holloway_vs_allen_april_15_2023(limit: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Get details to UFC Fight Night: Holloway vs. Allen**.                                                          
		 Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/April_15_2023"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ufc_287_pereira_vs_adesanya_2_april_08_2023(limit: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Get details to UFC 287: Pereira vs. Adesanya 2.**.    
		  .Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/April_08_2023"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_fighter_stats(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The search functionality provided helps you find fighter statistics and UFC/MMA history based on their names. It compares your search criteria with the fighters' information and returns a list of fighters that match all the specified criteria, including their statistics, averages, titles, wins, losses, and more."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/search"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_fighter_stats_by_age(age: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The search functionality provided helps you find fighter statistics and UFC/MMA history based on their age. It compares your search criteria with the fighters' information and returns a list of fighters that match all the specified criteria, including their statistics, averages, titles, wins, losses, and more."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/search"
    querystring = {'age': age, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ufc_fight_night_kara_france_vs_albazi_june_03_2023(offset: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Get details to UUFC Fight Night: Kara-France vs. Albazi**.                                                          
		 Access a range of information about each fighter, including their win-loss record, height, weight, reach, and age. results of a particular fight or a fighter's win-loss record."
    
    """
    url = f"https://mma-stats.p.rapidapi.com/June_03_2023"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mma-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

