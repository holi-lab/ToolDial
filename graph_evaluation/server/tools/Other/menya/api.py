import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def exercicebybodypart_endpoint(body: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "cette Endpoint vous retourner des exercices de musculation suivant le parti du corps"
    
    """
    url = f"https://menya.p.rapidapi.com/api/exercices/part/{body}"
    querystring = {}
    if body:
        querystring['body'] = body
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "menya.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exercicesbymuscle_endpoint(muscle: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "cet Endpoint retourne des données des exercices de musculation par leur muscle cible"
    
    """
    url = f"https://menya.p.rapidapi.com/api/exercices/muscle/{muscle}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "menya.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exercices_byid(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "cet Endpoint retourne des exercices de musculation par leur id"
    
    """
    url = f"https://menya.p.rapidapi.com/api/exercices/id/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "menya.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exercicesbyequipement_endpoint(equipement: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "cet Endpoint vous retourner des données des exercices de musculation par un équipement"
    
    """
    url = f"https://menya.p.rapidapi.com/api/exercices/eq/{equipement}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "menya.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bodypart_endpoint(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "cette Endpoint retourner un tableau des parties du corps pour les exercices de musculation"
    
    """
    url = f"https://menya.p.rapidapi.com/api/exercices/bodyPart"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "menya.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def equipement_endpoint(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "cet Endpoint va retourner des données des équipements utilise dans des exercices de musculation"
    
    """
    url = f"https://menya.p.rapidapi.com/api/exercices/equipement"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "menya.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exercices_endpoint(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "il vous retourner l'identifier de l'exercice, le nom, l'image anime, le muscle cible..."
    
    """
    url = f"https://menya.p.rapidapi.com/api/exercices"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "menya.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def utujajuro_endpoint(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "cet Endpoint vous permettez de recevoir les données de divers récitations en langue kirundi (utujajuro), cela inclus "
		
		- l'identifiant
		- akajajuro"
    
    """
    url = f"https://menya.p.rapidapi.com/api/jajura"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "menya.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sokwe_endpoint(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "cet Endpoint vous permettez de recevoir des données de divers énigme en langue kirundi(sokwe , niruze)"
    
    """
    url = f"https://menya.p.rapidapi.com/api/sokwe"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "menya.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def universit_endpoint(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "cet Endpoint vous permettez de recevoir les données des universités opérant au Burundi . cela inclus :
		- le nom de l'université
		- les facultés qu'ils offrent
		- le lien du site web
		 - tec"
    
    """
    url = f"https://menya.p.rapidapi.com/api/universite"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "menya.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def banks_endpoint(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "cet Endpoint vous permettez de recevoir les informations concernant divers Banks opérant au Burundi ainsi  que certains microfinance. cela inclus :
		
		- le nom de la bank
		- le lien du site web
		- les services qu'il offre
		- tec"
    
    """
    url = f"https://menya.p.rapidapi.com/api/banks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "menya.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def companies_endpoint(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "cet endpoint vous permettez de recevoir des informations sur les grands entreprise operant au burundi , cela inclus :
		- le nom de l'entreprise
		- les services qu'il offre
		- le lien du site web
		- etc"
    
    """
    url = f"https://menya.p.rapidapi.com/api/companies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "menya.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hotels_endpiont(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "cet endpoint vous permettez de recevoir toute les informations concernant certains grands Hotels du pays, il inclus :
		- le nom de l'hôtels
		- le lien de son site web
		- ses Etoiles
		-etc"
    
    """
    url = f"https://menya.p.rapidapi.com/api/Hotels"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "menya.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def touristique_site_endpoint(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "cet Endpoint vous permettez de recevoir tout les sites touristiques du burundi..il inclus :
		- le nom du site
		-sa location
		- description
		-etc"
    
    """
    url = f"https://menya.p.rapidapi.com/api/touriste-site"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "menya.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def imyibusta_endpoint(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "cet Endpoint vous permettez de vous fournir des citations burundaise(umyibusta) que pour le moment on a presque 700 citations en kirundi et prochainement on va plus y ajouter d'autres."
    
    """
    url = f"https://menya.p.rapidapi.com/api/imyibusta"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "menya.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

