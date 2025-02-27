import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def on_line(bot_name: str, channel_token: str, user_name: str, user_msg_text: str, save_only_positive_info: bool=None, use_change_topic: bool=None, use_detect_user_info: bool=None, load_only_positive_info: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "LINEボット専用の雑談APIです。
		
		以下はon_lineのレスポンス例と、その意味です。
		```
		{
		"response":{
		    "res":"おはよーございます"         # 生成された返答
		    "user_score":-2.8724350929260254   # ユーザーの発話のスコア
		    "lm":0.7096909880638123            # 文章の妥当性スコア
		    "mc":-1.5187406539916992           # 返答としての妥当性スコア
		    "score":-0.3534274697303772        # 総合スコア
		    "history_text":"<speaker2>Hello"   # 発話の生成に使われた発話履歴
		    "res_score_list":[                 # 採用しなかった返答のリスト
		        0:[
		            0:"おはよーございます"       # 返答の内容
		            1:[                          # スコア（文章の妥当性,返答としての妥当性,総合スコア）
		                0:0.7096909880638123
		                1:-1.5187406539916992
		                2:-0.3534274697303772
		            ]
		        ]
		        ...     # 中略
		    ]
		    }
		}
		```"
    bot_name: このAPIを利用するボットの名前を指定してください。
※API内部では[bot_name - user_name]間の会話履歴を元に、「流れのある自然な会話」を成立させます。
        channel_token: チャンネルアクセストークンを指定してください。
        user_name: ボットの話し相手であるユーザーの名前を指定してください。
        user_msg_text: ユーザーから話しかけられた内容（テキスト）を指定してください。
        save_only_positive_info: use_detect_user_info の情報抽出の対象をポジティブな事柄のみにする場合true、すべての事柄を抽出する場合はfalseを指定してください。
        use_change_topic: trueならば話題変換をします。（detect_user_info=falseの時は動作しません）
        use_detect_user_info: ユーザーのメッセージから情報を抽出する場合はtrueを、情報抽出をしない場合はfalseを指定してください。
        load_only_positive_info: trueの時、話題変換時にはポジティブな単語のみを使用します。

        
    """
    url = f"https://generaltalker.p.rapidapi.com/on_line/"
    querystring = {'bot_name': bot_name, 'channel_token': channel_token, 'user_name': user_name, 'user_msg_text': user_msg_text, }
    if save_only_positive_info:
        querystring['save_only_positive_info'] = save_only_positive_info
    if use_change_topic:
        querystring['use_change_topic'] = use_change_topic
    if use_detect_user_info:
        querystring['use_detect_user_info'] = use_detect_user_info
    if load_only_positive_info:
        querystring['load_only_positive_info'] = load_only_positive_info
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "generaltalker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def on_twitter(user_msg_text: str, user_name: str, reply_to_id: int, bot_name: str, load_only_positive_info: bool=None, use_detect_user_info: bool=None, use_change_topic: bool=None, save_only_positive_info: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "ツイッターボット専用の雑談APIです。reply_to_id（リプライ先のID）を渡すことで、過去の会話履歴をふまえた会話ができます。
		
		以下はレスポンスの例と、その意味です。
		
		```
		{
		"response":{
		    "res":"おはよーございます"         # 生成された返答
		    "user_score":-2.8724350929260254   # ユーザーの発話のスコア
		    "lm":0.7096909880638123            # 文章の妥当性スコア
		    "mc":-1.5187406539916992           # 返答としての妥当性スコア
		    "score":-0.3534274697303772        # 総合スコア
		    "history_text":"<speaker2>Hello"   # 発話の生成に使われた発話履歴
		    "res_score_list":[                 # 採用しなかった返答のリスト
		        0:[
		            0:"おはよーございます"       # 返答の内容
		            1:[                          # スコア（文章の妥当性,返答としての妥当性,総合スコア）
		                0:0.7096909880638123
		                1:-1.5187406539916992
		                2:-0.3534274697303772
		            ]
		        ]
		        ...     # 中略
		    ]
		    "user_utt_id":1355             # ユーザーの発話のid
		    "bot_utt_id":1356              # 生成した返答のid
		    }
		}
		```"
    user_msg_text: ユーザーから話しかけられた内容（テキスト）を指定してください。
        user_name: ボットの話し相手であるユーザーの名前を指定してください。
        reply_to_id: リプライ先のid番号
        bot_name: このAPIを利用するボットの名前を指定してください。
※API内部では[bot_name - user_name]間の会話履歴を元に、「流れのある自然な会話」を成立させます。
        load_only_positive_info: trueの時、話題変換時にはポジティブな単語のみを使用します。

        use_detect_user_info: ユーザーのメッセージから情報を抽出する場合はtrueを、情報抽出をしない場合はfalseを指定してください。
        use_change_topic: trueならば話題変換をします。（detect_user_info=falseの時は動作しません）
        save_only_positive_info: use_detect_user_info の情報抽出の対象をポジティブな事柄のみにする場合true、すべての事柄を抽出する場合はfalseを指定してください。
        
    """
    url = f"https://generaltalker.p.rapidapi.com/on_twitter/"
    querystring = {'user_msg_text': user_msg_text, 'user_name': user_name, 'reply_to_id': reply_to_id, 'bot_name': bot_name, }
    if load_only_positive_info:
        querystring['load_only_positive_info'] = load_only_positive_info
    if use_detect_user_info:
        querystring['use_detect_user_info'] = use_detect_user_info
    if use_change_topic:
        querystring['use_change_topic'] = use_change_topic
    if save_only_positive_info:
        querystring['save_only_positive_info'] = save_only_positive_info
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "generaltalker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def on_slack(user_name: str, user_msg_text: str, channel_token: str, bot_name: str, save_only_positive_info: bool=None, use_detect_user_info: bool=None, load_only_positive_info: bool=None, use_change_topic: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "slackボット専用の雑談APIです。
		
		以下はon_slackのレスポンス例と、その意味です。
		```
		{
		"response":{
		    "res":"おはよーございます"         # 生成された返答
		    "user_score":-2.8724350929260254   # ユーザーの発話のスコア
		    "lm":0.7096909880638123            # 文章の妥当性スコア
		    "mc":-1.5187406539916992           # 返答としての妥当性スコア
		    "score":-0.3534274697303772        # 総合スコア
		    "history_text":"<speaker2>Hello"   # 発話の生成に使われた発話履歴
		    "res_score_list":[                 # 採用しなかった返答のリスト
		        0:[
		            0:"おはよーございます"       # 返答の内容
		            1:[                          # スコア（文章の妥当性,返答としての妥当性,総合スコア）
		                0:0.7096909880638123
		                1:-1.5187406539916992
		                2:-0.3534274697303772
		            ]
		        ]
		        ...     # 中略
		    ]
		    }
		}
		```"
    user_name: ボットの話し相手であるユーザーの名前を指定してください。
        user_msg_text: ユーザーから話しかけられた内容（テキスト）を指定してください。
        channel_token: Slackのチャンネルごとのトークンを指定してください。
例 'D01E65TPZ2M'
        bot_name: このAPIを利用するボットの名前を指定してください。
※API内部では[bot_name - user_name]間の会話履歴を元に、「流れのある自然な会話」を成立させます。
        save_only_positive_info: use_detect_user_info の情報抽出の対象をポジティブな事柄のみにする場合true、すべての事柄を抽出する場合はfalseを指定してください。
        use_detect_user_info: ユーザーのメッセージから情報を抽出する場合はtrueを、情報抽出をしない場合はfalseを指定してください。
        load_only_positive_info: trueの時、話題変換時にはポジティブな単語のみを使用します。

        use_change_topic: trueならば話題変換をします。（detect_user_info=falseの時は動作しません）
        
    """
    url = f"https://generaltalker.p.rapidapi.com/on_slack/"
    querystring = {'user_name': user_name, 'user_msg_text': user_msg_text, 'channel_token': channel_token, 'bot_name': bot_name, }
    if save_only_positive_info:
        querystring['save_only_positive_info'] = save_only_positive_info
    if use_detect_user_info:
        querystring['use_detect_user_info'] = use_detect_user_info
    if load_only_positive_info:
        querystring['load_only_positive_info'] = load_only_positive_info
    if use_change_topic:
        querystring['use_change_topic'] = use_change_topic
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "generaltalker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def on_general(user_name: str, user_msg_text: str, bot_name: str, save_only_positive_info: bool=None, load_only_positive_info: bool=None, use_detect_user_info: bool=None, use_change_topic: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "いろいろなボットでご利用いただける雑談APIです。
		
		以下はon_generalのレスポンス例と、その意味です。
		```
		{
		"response":{
		    "res":"おはよーございます"         # 生成された返答
		    "user_score":-2.8724350929260254   # ユーザーの発話のスコア
		    "lm":0.7096909880638123            # 文章の妥当性スコア
		    "mc":-1.5187406539916992           # 返答としての妥当性スコア
		    "score":-0.3534274697303772        # 総合スコア
		    "history_text":"<speaker2>Hello"   # 発話の生成に使われた発話履歴
		    "res_score_list":[                 # 採用しなかった返答のリスト
		        0:[
		            0:"おはよーございます"       # 返答の内容
		            1:[                          # スコア（文章の妥当性,返答としての妥当性,総合スコア）
		                0:0.7096909880638123
		                1:-1.5187406539916992
		                2:-0.3534274697303772
		            ]
		        ]
		        ...     # 中略
		    ]
		    }
		}
		```"
    user_name: ボットの話し相手であるユーザーの名前を指定してください。
        user_msg_text: ユーザーから話しかけられた内容（テキスト）を指定してください。
        bot_name: このAPIを利用するボットの名前を指定してください。
※API内部では[bot_name - user_name]間の会話履歴を元に、「流れのある自然な会話」を成立させます。
        save_only_positive_info: use_detect_user_info の情報抽出の対象をポジティブな事柄のみにする場合true、すべての事柄を抽出する場合はfalseを指定してください。
        load_only_positive_info: trueの時、話題変換時にはポジティブな単語のみを使用します。

        use_detect_user_info: ユーザーのメッセージから情報を抽出する場合はtrueを、情報抽出をしない場合はfalseを指定してください。
        use_change_topic: trueならば話題変換をします。（detect_user_info=falseの時は動作しません）
        
    """
    url = f"https://generaltalker.p.rapidapi.com/on_general/"
    querystring = {'user_name': user_name, 'user_msg_text': user_msg_text, 'bot_name': bot_name, }
    if save_only_positive_info:
        querystring['save_only_positive_info'] = save_only_positive_info
    if load_only_positive_info:
        querystring['load_only_positive_info'] = load_only_positive_info
    if use_detect_user_info:
        querystring['use_detect_user_info'] = use_detect_user_info
    if use_change_topic:
        querystring['use_change_topic'] = use_change_topic
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "generaltalker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

