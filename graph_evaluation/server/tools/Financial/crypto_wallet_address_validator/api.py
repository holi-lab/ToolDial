import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def validate(address: str, currency: str='btc', network: str='prod', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allow you to validate a wallet address."
    currency: Currency name or symbol, 'bitcoin' (default), 'litecoin' or 'LTC'.

Supported:
Auroracoin/AUR, 'auroracoin' or 'AUR'
Bankex/BKX, 'bankex' or 'BKX'
BeaverCoin/BVC, 'beavercoin' or 'BVC'
Biocoin/BIO, 'biocoin' or 'BIO'
Bitcoin/BTC, 'bitcoin' or 'BTC'
BitcoinCash/BCH, 'bitcoincash' or 'BCH'
BitcoinGold/BTG, 'bitcoingold' or 'BTG'
BitcoinPrivate/BTCP, 'bitcoinprivate' or 'BTCP'
BitcoinZ/BTCZ, 'bitcoinz' or 'BTCZ'
Callisto/CLO, 'callisto' or 'CLO'
Dash, 'dash' or 'DASH'
Decred/DCR, 'decred' or 'DCR'
Digibyte/DGB, 'digibyte' or 'DGB'
Dogecoin/DOGE, 'dogecoin' or 'DOGE'
Ethereum/ETH, 'ethereum' or 'ETH'
EthereumClassic/ETH, 'ethereumclassic' or 'ETC'
EthereumZero/ETZ, 'etherzero' or 'ETZ'
Freicoin/FRC, 'freicoin' or 'FRC'
Garlicoin/GRLC, 'garlicoin' or 'GRLC'
Hush/HUSH, 'hush' or 'HUSH'
Komodo/KMD, 'komodo' or 'KMD'
Litecoin/LTC, 'litecoin' or 'LTC'
Megacoin/MEC, 'megacoin' or 'MEC'
Monero/XMR, 'monero' or 'XMR'
Namecoin/NMC, 'namecoin' or 'NMC'
Nano/NANO, 'nano' or 'NANO'
NEO/NEO, 'NEO' or 'NEO'
NeoGas/GAS, 'neogas' or 'GAS'
Peercoin/PPCoin/PPC, 'peercoin' or 'PPC'
Primecoin/XPM, 'primecoin' or 'XPM'
Protoshares/PTS, 'protoshares' or 'PTS'
Qtum/QTUM, 'qtum' or 'QTUM'
Raiblocks/XRB, 'raiblocks' or 'XRB'
Ripple/XRP, 'ripple' or 'XRP'
Snowgem/SNG, 'snowgem' or 'SNG'
Vertcoin/VTC, 'vertcoin' or 'VTC'
Votecoin/VTC, 'votecoin' or 'VOT'
Zcash/ZEC, 'zcash' or 'ZEC'
Zclassic/ZCL, 'zclassic' or 'ZCL'
ZenCash/ZEN, 'zencash' or 'ZEN'
        network: Blockhain network,  'prod' (default) to enforce standard address, 'testnet' to enforce testnet address and 'both' to enforce nothing.

        
    """
    url = f"https://crypto-wallet-address-validator.p.rapidapi.com/validate/{address}"
    querystring = {}
    if currency:
        querystring['currency'] = currency
    if network:
        querystring['network'] = network
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-wallet-address-validator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

