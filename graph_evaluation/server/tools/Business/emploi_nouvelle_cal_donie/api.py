import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def n_derni_res_offres(nbannonces: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Renvoie les n (n < 100) dernières offres"
    nbannonces: Le nombre (< 100) d'annonces souhaitées.
        
    """
    url = f"https://emploi-nouvelle-caledonie.p.rapidapi.com/emploi/latest/{nbannonces}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emploi-nouvelle-caledonie.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def statistiques(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Statistiques sur les emplois"
    
    """
    url = f"https://emploi-nouvelle-caledonie.p.rapidapi.com/stats"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emploi-nouvelle-caledonie.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tous_les_emloyeurs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Liste de tous les employeurs"
    
    """
    url = f"https://emploi-nouvelle-caledonie.p.rapidapi.com/employeurs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emploi-nouvelle-caledonie.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def derni_res_offres(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Renvoi les 25 offres d'emploi les plus récentes"
    
    """
    url = f"https://emploi-nouvelle-caledonie.p.rapidapi.com/emploi/latest"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emploi-nouvelle-caledonie.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def employeur_par_nom_exact(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Renvoie un employeur précis selon son nom exact"
    
    """
    url = f"https://emploi-nouvelle-caledonie.p.rapidapi.com/employeurs/{name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emploi-nouvelle-caledonie.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def employeur_d_une_offre(numero: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Renvoie l'employeur d'une offre d'emploi précise"
    
    """
    url = f"https://emploi-nouvelle-caledonie.p.rapidapi.com/emploi/{numero}/employeur"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emploi-nouvelle-caledonie.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def offre_d_emploi_par_num_ro(numero: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Renvoie l'offre d'emploi selon son numéro emploi.gouv.nc"
    numero: Numéro de l'offre d'emploi
        
    """
    url = f"https://emploi-nouvelle-caledonie.p.rapidapi.com/emploi/{numero}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emploi-nouvelle-caledonie.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recherche_d_offres(datefin: str, contrat: str, commune: str, nombremaxoffres: int, motscles: str, datedebut: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Rechercher des offres d'emploi à l'aide de critères tel que le nombre max, les dates, etc..."
    datefin: Date de fin au format DDMMYYYY
        contrat: Type de contrat (CDD, CDI, ...)
        commune: Lieu de l'emploi
        nombremaxoffres: Le nombre max d'offres que je veux
        motscles: Même fonctionnement que celui de emploi.gouv.nc : recherche full text (Si plusieurs mots clès les séparés par &, par exemple (chef&projet).)
        datedebut: Date de début au format DDMMYYYY
        
    """
    url = f"https://emploi-nouvelle-caledonie.p.rapidapi.com/search/{nombremaxoffres}/{motscles}/{commune}/{contrat}/{datedebut}/{datefin}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emploi-nouvelle-caledonie.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def csv(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Export les offres au format csv"
    
    """
    url = f"https://emploi-nouvelle-caledonie.p.rapidapi.com/csv"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emploi-nouvelle-caledonie.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def documentation(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API documentation"
    
    """
    url = f"https://emploi-nouvelle-caledonie.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emploi-nouvelle-caledonie.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

