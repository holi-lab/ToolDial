import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def catecismo_capitulo(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Catecismo, catequesis, comunion, formación, recursos, catecismo catolico, catecismo de la iglesia, iglesia apostolica romana. La fe, doctrina y moral de la Iglesia católica"
    
    """
    url = f"https://creemos.p.rapidapi.com/api/catecismo/{slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "creemos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def catecismo_cat_lico(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Catecismo, catequesis, comunion, formación, recursos, catecismo catolico, catecismo de la iglesia, iglesia apostolica romana. La fe, doctrina y moral de la Iglesia católica"
    
    """
    url = f"https://creemos.p.rapidapi.com/api/catecismo"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "creemos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def santo(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Biografia completa de cada santo.
		Conjunto de las personas veneradas en la Iglesia católica como santos o beatos"
    
    """
    url = f"https://creemos.p.rapidapi.com/api/santo/{slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "creemos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def podcast(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "¡Sumate a esta nueva propuesta de comunicacion! Los Podcasts catolicos son voces jovenes (y otras no tanto jeje) que nos evangelizan con su testimonio y sus dones.
		Podcasts, podcast, programa de radio, radio, show, episodios, catolicos, spotify"
    
    """
    url = f"https://creemos.p.rapidapi.com/api/podcast/{slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "creemos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def podcasts_cat_licos(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "¡Sumate a esta nueva propuesta de comunicacion! Los Podcasts catolicos son voces jovenes (y otras no tanto jeje) que nos evangelizan con su testimonio y sus dones.
		Podcasts, podcast, programa de radio, radio, show, episodios, catolicos, spotify"
    
    """
    url = f"https://creemos.p.rapidapi.com/api/podcasts"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "creemos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movimiento_o_carisma(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Agentes de una nueva Evangelización"
    
    """
    url = f"https://creemos.p.rapidapi.com/api/movimiento/{slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "creemos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movimientos_y_carismas_cat_licos(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Agentes de una nueva Evangelización.
		Los movimientos son realidades laicales que desarrollan su vida en la Iglesia, para llevar Evangelizar con sus distintos carismas. Conocélos!"
    
    """
    url = f"https://creemos.p.rapidapi.com/api/movimientos"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "creemos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def oraci_n(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Oración Para pedir, agradecer, adorar y alabar"
    
    """
    url = f"https://creemos.p.rapidapi.com/api/oracion/{slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "creemos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cancionero_cat_lico(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Cancionero católico con mas de 845 recursos de letras y acordes para que cantes, ores y alabes a Dios!
		Pues aquel que canta alabanzas, también alaba con alegría!"
    
    """
    url = f"https://creemos.p.rapidapi.com/api/cancionero"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "creemos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def artistas_cat_licos(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "¡Conocé a los artistas católicos más populares! Más de 100 Musicos, Dibujantes, Actores, Escritores y compositores Católicos. Athenas Vénica, Cristobal Fones, Eduardo Meana, Pascua Joven, Maxi Larghi, Pablo Martinez y muchos mas hermanos que con su arte nos acercan un poco más a Dios"
    
    """
    url = f"https://creemos.p.rapidapi.com/api/artistas"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "creemos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def influencer(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Evangelizando a las nuevas generaciones
		Influencer Catolicos! Hermanos que con su carisma particular deciden evangelizar en el mundo del Instagram, TikTok, Facebook, Twitter, Youtube y Twitch!"
    
    """
    url = f"https://creemos.p.rapidapi.com/api/influencer/{slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "creemos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def artista(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "¡Conocé a los artistas católicos más populares! Más de 100 Musicos, Dibujantes, Actores, Escritores y compositores Católicos. Athenas Vénica, Cristobal Fones, Eduardo Meana, Pascua Joven, Maxi Larghi, Pablo Martinez y muchos mas hermanos que con su arte nos acercan un poco más a Dios"
    
    """
    url = f"https://creemos.p.rapidapi.com/api/artista/{slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "creemos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def influencers_catolicos(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Evangelizando a las nuevas generaciones
		Influencer Catolicos! Hermanos que con su carisma particular deciden evangelizar en el mundo del Instagram, TikTok, Facebook, Twitter, Youtube y Twitch!"
    
    """
    url = f"https://creemos.p.rapidapi.com/api/influencers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "creemos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def santos(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Biografia completa de cada santo.
		Conjunto de las personas veneradas en la Iglesia católica como santos o beatos"
    
    """
    url = f"https://creemos.p.rapidapi.com/api/santos"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "creemos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cancion(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtene los acordes , la letra , categoria, spotify y youtube de la cancion.
		Pues aquel que canta alabanzas, también alaba con alegría!"
    
    """
    url = f"https://creemos.p.rapidapi.com/api/cancion/{slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "creemos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def oraciones(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Compendio de 592 oraciones para que puedas seguir creciendo en comunión. El Credo, Oraciones, alabanzas, fe, Para rezar, Para pedir, agradecer, adorar y alabar."
    
    """
    url = f"https://creemos.p.rapidapi.com/api/oraciones"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "creemos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def frase(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Una frase inspiradora, oracion o cita biblica aleatoria (Hay más de 3000)"
    
    """
    url = f"https://creemos.p.rapidapi.com/api/frase"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "creemos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

