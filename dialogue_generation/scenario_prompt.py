import random
from copy import deepcopy

aug_ment = {
    "inform": [
        "Sure! ~~"
        "Ok ~",
        "Certainly! ~~",
    ],
    "affirm": [
        "Yes, that works.",
        "That would be great."
        "Sure, that sounds good.",
        "Yes, please proceed."
    ],
    "negate": [
        "No, that’s not what I meant.",
        "I'm good. Thank you though.",
        "Umm... that's not what I want.."
    ],
    "request": [
        "To call ~"
        "I need~",
        "May I ask for ~",
        "Please tell me ~",
    ],
    "clarify":[
        "Could you please provide more ~",
        "I’m not sure I understand. Can you clarify ~",
        "Could you explain that in more ~",
        "Can you clarify your ~"
    ]
}


backward_common_instruction = """<Instruction>
You are an intelligent scenario writer.
Your mission is to generate multi call reasoning step of API call to solve the given user's query based on the given informations.
Every data in required_parameters and optional_parameters of api_request and output_components of api_response must be the same as given data in <API documentation with mock data>, so you should generate dialogue based on <API documentation with mock data>.
Do not hallucinate values for APIs, use <API documentation with mock data>.
System's thought should be generated based on the content of the situation of each turns.
"""

backward_prompt = """
{common_instruction}

<Relation>
{relation}

<Dialogue scenario>
{scenario}

- During the generation of thought of system, refer the given situation of each turn of the system. And don't let the conversation participants talk too monotonous.
- For retriever_status, paste the Retriever status of the turn in the dialog scenario exactly as in <Dialogue format>.

<API documentation with mock data>
{data}

<Dialogue format>
```json
[
    {{
        User: {{"message": <User's request>, "action":<user action>}}
    }},
    {{
        System: {{
            "thought": "Based on the user's request, I need to call the retriever to search the proper API",
            "action": "retriever_call",
            "retriever_status":{{'retriever_call': 'true', 'retrieved_api': {{'API2': 0.74, 'API3': 0.62, 'API1': 0.6, 'API6: 0.54, 'API8': 0.28608537}}}}.
            "api_request":{{}},
            "api_response":{{}},
            "message": ""
        }}
    }},
    {{
        System: {{
            "thought": "Based on results of the retriever, I need to call API2. To call it, I need to ask 'param1' and 'param2' to user.",
            "action": <system action>,
            "retriever_status":{{'retriever_call': 'false', 'retrieved_api': 'none'}},
            "api_request":{{}},
            "api_response":{{}},
            "message": <System ask the 'param1' and 'param2' to user>
        }}
    }},
    {{
        User: {{"message": <User provide 'param1' as '123' and 'param2' as '456' to system>, "action":"inform"}}
    }},
    {{
        System: {{
            "thought": "Based on the user's response, now I can call the API2",
            "action": "call",
            "retriever_status":{{'retriever_call': 'false', 'retrieved_api': 'none'}},
            "api_request":{{
                "api_name":"API2",
                "required_parameters":{{"param1":"123"}},
                "optional_parameters":{{"param2":"456"}}
            }},
            "api_response":{{
                "api_name":"API2",
                "result":{{"ouput|comp1":"target_value", "output|comp2":"val6"}}
            }},
            "message": ""
        }},
    }}
    {{
        System: {{
            "thought": "Based on the results of API2, I can respond to user.",
            "action": <system action>,
            "retriever_status":{{'retriever_call': 'false', 'retrieved_api': 'none'}},
            "api_request":{{
                "api_name":"API2",
                "required_parameters":{{"param1":"123"}},
                "optional_parameters":{{"param2":"456"}}
            }},
            "api_response":{{
                "api_name":"API2",
                "result":{{"ouput|comp1":"target_value", "output|comp2":"val6"}}
            }},
            "message": "<System respond to user based on the result of the API2>"
        }},
    }}
]
```
"""

req_back_prompt = "To call {api} system asks {input_param} as an required input parameter to user."