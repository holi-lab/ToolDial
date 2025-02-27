import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def decryptstring(cryptalgorithm: str, secretkey: str, encryptedtext: str, ciphermode: str='CBC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Decrypt encrypted-text (base64-format) using input secret-key according to selected crypt-algorithm and cipher-mode (optional with CBC as default value)."
    cryptalgorithm: Decryption Algorithm (AES [default], DES, TripleDES, RC2, Rijndael).
        secretkey: Secret-Key string which will be used on decryption process, and it should be same secret-key which used on encryption process.
        encryptedtext: Encrypted text on base64-string format.
        ciphermode: Decryption Cipher-Mode (CBC [default], ECB, CFB).
        
    """
    url = f"https://encryption-api1.p.rapidapi.com/api/Cryptor/decryptstring"
    querystring = {'cryptAlgorithm': cryptalgorithm, 'secretKey': secretkey, 'encryptedText': encryptedtext, }
    if ciphermode:
        querystring['cipherMode'] = ciphermode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "encryption-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def decryptstring(cryptalgorithm: str, secretkey: str, encryptedtext: str, ciphermode: str='CBC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Decrypt encrypted-text (base64-format) using input secret-key according to selected crypt-algorithm and cipher-mode (optional with CBC as default value)."
    cryptalgorithm: Decryption Algorithm (AES [default], DES, TripleDES, RC2, Rijndael).
        secretkey: Secret-Key string which will be used on decryption process, and it should be same secret-key which used on encryption process.
        encryptedtext: Encrypted text on base64-string format.
        ciphermode: Decryption Cipher-Mode (CBC [default], ECB, CFB).
        
    """
    url = f"https://encryption-api1.p.rapidapi.com/api/Cryptor/decryptstring"
    querystring = {'cryptAlgorithm': cryptalgorithm, 'secretKey': secretkey, 'encryptedText': encryptedtext, }
    if ciphermode:
        querystring['cipherMode'] = ciphermode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "encryption-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def encryptstring(secretkey: str, plaintext: str, cryptalgorithm: str, ciphermode: str='CBC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Encrypt plain-text using input secret-key according to selected crypt-algorithm and cipher-mode (optional with CBC as default value)."
    secretkey: Secret-Key string which will be used on encryption process.
        plaintext: Plain text which you want to encrypt.
        cryptalgorithm: Encryption Algorithm (AES [default], DES, TripleDES, RC2, Rijndael).
        ciphermode: Encryption Cipher-Mode (CBC [default], ECB, CFB).
        
    """
    url = f"https://encryption-api1.p.rapidapi.com/api/Cryptor/encryptstring"
    querystring = {'secretKey': secretkey, 'plainText': plaintext, 'cryptAlgorithm': cryptalgorithm, }
    if ciphermode:
        querystring['cipherMode'] = ciphermode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "encryption-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def encryptstring(secretkey: str, plaintext: str, cryptalgorithm: str, ciphermode: str='CBC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Encrypt plain-text using input secret-key according to selected crypt-algorithm and cipher-mode (optional with CBC as default value)."
    secretkey: Secret-Key string which will be used on encryption process.
        plaintext: Plain text which you want to encrypt.
        cryptalgorithm: Encryption Algorithm (AES [default], DES, TripleDES, RC2, Rijndael).
        ciphermode: Encryption Cipher-Mode (CBC [default], ECB, CFB).
        
    """
    url = f"https://encryption-api1.p.rapidapi.com/api/Cryptor/encryptstring"
    querystring = {'secretKey': secretkey, 'plainText': plaintext, 'cryptAlgorithm': cryptalgorithm, }
    if ciphermode:
        querystring['cipherMode'] = ciphermode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "encryption-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

