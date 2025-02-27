import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def top_coins_by_market_cap(limit: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top coins by mCap"
    
    """
    url = f"https://all-in-one-crypto-swiss-knife.p.rapidapi.com/coins/toplist"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-crypto-swiss-knife.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news(page: str='1', limit: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get crypto news"
    
    """
    url = f"https://all-in-one-crypto-swiss-knife.p.rapidapi.com/news"
    querystring = {}
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-crypto-swiss-knife.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def podcasts(limit: str='10', page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get crypto podcasts"
    
    """
    url = f"https://all-in-one-crypto-swiss-knife.p.rapidapi.com/podcasts"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-crypto-swiss-knife.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gas_price(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get current ETH gas price"
    
    """
    url = f"https://all-in-one-crypto-swiss-knife.p.rapidapi.com/gas"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-crypto-swiss-knife.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def summary(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get play 2 earn  market summary"
    
    """
    url = f"https://all-in-one-crypto-swiss-knife.p.rapidapi.com/gaming/summary"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-crypto-swiss-knife.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def free_games(page: str='1', pagesize: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get list of free games"
    
    """
    url = f"https://all-in-one-crypto-swiss-knife.p.rapidapi.com/gaming/free"
    querystring = {}
    if page:
        querystring['page'] = page
    if pagesize:
        querystring['pageSize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-crypto-swiss-knife.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def upcoming(pagesize: str='10', page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get upcoming play 2 earn games"
    
    """
    url = f"https://all-in-one-crypto-swiss-knife.p.rapidapi.com/gaming/upcoming"
    querystring = {}
    if pagesize:
        querystring['pageSize'] = pagesize
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-crypto-swiss-knife.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending_games(pagesize: str='20', page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get trending play 2 earn games"
    
    """
    url = f"https://all-in-one-crypto-swiss-knife.p.rapidapi.com/gaming/trending"
    querystring = {}
    if pagesize:
        querystring['pageSize'] = pagesize
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-crypto-swiss-knife.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games(page: str='1', pagesize: str='10', platform: str=None, genre: str=None, blockchain: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get play 2 earn games"
    platform: PLATFORMS: 
Android
iOS
Windows
Browser
Mac
Linux
PC
Mobile

        genre: GENRES:
Action
Adventure
AR
Arcade
Auto Battler
Base Building
Battle Royale
Brawler
Breeding
Card Game
Casual
City Building
Collectible
Combat
DeFi
Fantasy
Fighting
Football Game
FPS
Frontier defense
Horror Game
Idle Game
Metaverse
Minigame
Mining
MMO
MMORPG
MOBA
Move To Earn
Puzzle
PVP
Racing
RNG
RPG
Sci-Fi
Shooter
Simulation
Space Game
Sports
Strategy
Survival
Tower Defense
Turn-based Strategy
Virtual-World
VR
Other
        blockchain: BLOCKCHAINS:
Hive
WAX
EOS
BNB Chain
Polygon
Ethereum
Immutable-X
Solana
TRON
NEO
Flow
Enjin
NEAR
Avalanche
Waves
Vulcan Forged
Harmony
OKExChain
IOST
HECO
Gala Games
Terra
Aurora
Cardano
Elrond
Celo
VeChain
Fantom
Qtum
Klaytn
Algorand
Cronos
DEAP Coin
Moonriver
Oasis Network
Telos
Polkadot
Xaya
Phantasma
WEMIX
Myria
Huobi ECO Chain Mainnet
Wanchain
Arbitrum One
BNB Sidechain
Dogechain
MEVerse
Aptos
Other
        
    """
    url = f"https://all-in-one-crypto-swiss-knife.p.rapidapi.com/gaming/games"
    querystring = {}
    if page:
        querystring['page'] = page
    if pagesize:
        querystring['pageSize'] = pagesize
    if platform:
        querystring['platform'] = platform
    if genre:
        querystring['genre'] = genre
    if blockchain:
        querystring['blockchain'] = blockchain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-crypto-swiss-knife.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reddit(limit: str='10', name: str='dogecoin', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top latest crypto posts from Reddit from any subreddit.
		name= subreddit name from reddit url
		Example: https://www.reddit.com/r/dogecoin/"
    name: Subreddit name from url. Example: https://www.reddit.com/r/dogecoin/
        
    """
    url = f"https://all-in-one-crypto-swiss-knife.p.rapidapi.com/social/reddit"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-crypto-swiss-knife.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def twitter(page: str='1', limit: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get crypto tweets by curated authors"
    
    """
    url = f"https://all-in-one-crypto-swiss-knife.p.rapidapi.com/social/twitter"
    querystring = {}
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-crypto-swiss-knife.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def youtube_videos(q: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get videos.
		By default returns videos with 'crypto' keyword but  you can search for any keyword"
    
    """
    url = f"https://all-in-one-crypto-swiss-knife.p.rapidapi.com/social/youtube"
    querystring = {}
    if q:
        querystring['q'] = q
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-crypto-swiss-knife.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def whale_activities(limit: str='15', start: str='0', order_by: str='-time_at', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get latest whale activities."
    order_by: [\"-time_at\", \"time_at\", \"-usd_value\", \"usd_value\"];
        
    """
    url = f"https://all-in-one-crypto-swiss-knife.p.rapidapi.com/whales/activities"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if start:
        querystring['start'] = start
    if order_by:
        querystring['order_by'] = order_by
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-crypto-swiss-knife.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def whales_rank(start: str='0', limit: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get biggest whale addresses"
    
    """
    url = f"https://all-in-one-crypto-swiss-knife.p.rapidapi.com/whales"
    querystring = {}
    if start:
        querystring['start'] = start
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-crypto-swiss-knife.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events(limit: str='10', page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get crypto upcoming  events grouped by dates"
    
    """
    url = f"https://all-in-one-crypto-swiss-knife.p.rapidapi.com/events"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-crypto-swiss-knife.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def projects_staking_other_activities(address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get activities that address is involved in"
    
    """
    url = f"https://all-in-one-crypto-swiss-knife.p.rapidapi.com/portfolio/projects"
    querystring = {'address': address, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-crypto-swiss-knife.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nfts_on_address(address: str, chain: str='eth', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of NFTs that address holds."
    chain:     \"eth, bsc, matic, avax, okt, hmy, heco, klay, op, arb, ftm, xdai, cro, mobm, celo, aurora, fuse, evmos, cfx, kava\";

        
    """
    url = f"https://all-in-one-crypto-swiss-knife.p.rapidapi.com/portfolio/nfts"
    querystring = {'address': address, }
    if chain:
        querystring['chain'] = chain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-crypto-swiss-knife.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def net_worth_history_sparklines(address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get sparklines with address net worth for chart rendering.
		Returns array of objects with timestamp and USD value"
    
    """
    url = f"https://all-in-one-crypto-swiss-knife.p.rapidapi.com/portfolio/net-curve"
    querystring = {'address': address, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-crypto-swiss-knife.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def address_transaction_history(address: str, start_time: str='0', chain: str=None, page_count: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get transaction history of any address.
		Supported chains:     "eth, bsc, matic, avax, okt, hmy, heco, klay, op, arb, ftm, xdai, cro, mobm, celo, aurora, fuse, evmos, cfx, kava";"
    start_time: timestamp
        chain: Supported chains:     \"eth, bsc, matic, avax, okt, hmy, heco, klay, op, arb, ftm, xdai, cro, mobm, celo, aurora, fuse, evmos, cfx, kava\";

        page_count: How many elements to fetch
        
    """
    url = f"https://all-in-one-crypto-swiss-knife.p.rapidapi.com/portfolio/history"
    querystring = {'address': address, }
    if start_time:
        querystring['start_time'] = start_time
    if chain:
        querystring['chain'] = chain
    if page_count:
        querystring['page_count'] = page_count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-crypto-swiss-knife.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tokens_balances(address: str, chain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get balance of any address.
		supportedChains: eth, bsc, matic, avax, okt, hmy, heco, klay, op, arb, ftm, xdai, cro, mobm, celo, aurora, fuse, evmos, cfx, kava"
    
    """
    url = f"https://all-in-one-crypto-swiss-knife.p.rapidapi.com/portfolio/balance"
    querystring = {'address': address, 'chain': chain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-crypto-swiss-knife.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_holders(is_id: str, start: str='0', limit: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top holders in each protocol"
    
    """
    url = f"https://all-in-one-crypto-swiss-knife.p.rapidapi.com/protocols/top-holders"
    querystring = {'id': is_id, }
    if start:
        querystring['start'] = start
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-crypto-swiss-knife.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def protocols(q: str=None, chain_id: str='fuse', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get protocols.
		You can search by adding 'q' query param.
		
		Supported chains:  "eth, bsc, matic, avax, okt, hmy, heco, klay, op, arb, ftm, xdai, cro, mobm, celo, aurora, fuse, evmos, cfx, kava";"
    q: Search parameter
        chain_id: Supported chains:  \"eth, bsc, matic, avax, okt, hmy, heco, klay, op, arb, ftm, xdai, cro, mobm, celo, aurora, fuse, evmos, cfx, kava\";

Leave empty for all protocols 
        
    """
    url = f"https://all-in-one-crypto-swiss-knife.p.rapidapi.com/protocols"
    querystring = {}
    if q:
        querystring['q'] = q
    if chain_id:
        querystring['chain_id'] = chain_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-crypto-swiss-knife.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def protocol(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get protocol"
    
    """
    url = f"https://all-in-one-crypto-swiss-knife.p.rapidapi.com/protocols/protocol"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-crypto-swiss-knife.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pools(is_id: str, start: str='0', limit: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get DeFi pools
		You can get id from /protocols endpoint."
    
    """
    url = f"https://all-in-one-crypto-swiss-knife.p.rapidapi.com/protocols/pools"
    querystring = {'id': is_id, }
    if start:
        querystring['start'] = start
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-crypto-swiss-knife.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def total_value_locked(limit: str='20', page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get TVL  in each chain"
    
    """
    url = f"https://all-in-one-crypto-swiss-knife.p.rapidapi.com/protocols/tvl"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-crypto-swiss-knife.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def potential_upcoming(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of tokenless protocols that may airdrop soon"
    
    """
    url = f"https://all-in-one-crypto-swiss-knife.p.rapidapi.com/airdrops/potential"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-crypto-swiss-knife.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def single_coin(interval: str, slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get single coin info and history sparklines for rendering chart"
    interval: Intervals supported: 1D, 1W, 1M, 3M, YTD, 1Y, 3Y, ALL (case insensitive)


        
    """
    url = f"https://all-in-one-crypto-swiss-knife.p.rapidapi.com/coins/history"
    querystring = {'interval': interval, 'slug': slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-crypto-swiss-knife.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

