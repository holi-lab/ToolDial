import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api_v1_annonces(since: str=None, from_index: str=None, categorie: str=None, departement: str='75', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Annonces Immobilieres (téléphone compris). Retourne 10 annonces, ordonnées comme sur le site leboncoin.
		
		La réponse contient la clé `next_index`,  dont la valeur est à renseigner dans la prochaine requête dans le paramètre `from_index` afin de récupérer les 10 prochaines annonces.
		
		L'appel n'est pas facturé si pas de résultat disponible (A partir de l'abonnement PRO)."
    since: Date limite des annonces à rechercher.

Seule les annonces plus récentes que cette date seront retournées
        from_index: Index de pagination, permet de récupérer les annonce suivantes. 

L'index est indiqué dans le résultat de la requête précédente dans la valeur `next_index`
        categorie: Catégorie des annonces: `vente` ou `location`
        departement: Département des annonces (deux chiffres)
        
    """
    url = f"https://immobilier-leboncoin.p.rapidapi.com/api/v1/annonces"
    querystring = {}
    if since:
        querystring['since'] = since
    if from_index:
        querystring['from_index'] = from_index
    if categorie:
        querystring['categorie'] = categorie
    if departement:
        querystring['departement'] = departement
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "immobilier-leboncoin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_connection(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API de connexion (Make - Integromat uniquement)"
    
    """
    url = f"https://immobilier-leboncoin.p.rapidapi.com/api/v1/connection"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "immobilier-leboncoin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_annonces_list_id(list_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Renvoie les donnée d'une annonce, incluant le numéro de téléphone, si disponible.
		
		L'appel n'est pas facturé si pas de résultat disponible (A partir de l'abonnement PRO).
		
		Les annonces ne sont pas disponibles en temps réel. Il est conseillé de retenter ultérieurement en cas d' absence de résultat."
    list_id: id de l'annonce.

Example: pour l'annonces https://www.leboncoin.fr/locations/2064259858.htm, l'id de l'annonce est 2064259858
        
    """
    url = f"https://immobilier-leboncoin.p.rapidapi.com/api/v1/annonces/{list_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "immobilier-leboncoin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

