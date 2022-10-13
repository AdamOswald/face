import requests
import json
import urllib.request
import time


class Request:
    def __init__(self):
        self.tasks_url = "https://paint.api.wombo.ai/api/tasks/"
        self.headers = {
            "accept-encoding": "gzip",
            "authorization": "Bearer " + get_token(),
            "connection": "keep-alive",
            "content-type": "application/json; charset=utf-8",
            "host": "paint.api.wombo.ai",
            "user-agent": "okhttp/3.14.9",
        }

    def start_task(self, prompt, style):
        styles = {
            "Synthwave": 1,
            "Ukiyoe": 2,
            "No Style": 3,
            "Steampunk": 4,
            "Fantasy Art": 5,
            "Vibrant": 6,
            "HD": 7,
            "Pastel": 8,
            "Psychic": 9,
            "Dark Fantasy": 10,
            "Mystical": 11,
            "Festive": 12,
            "Baroque": 13,
            "Etching": 14
        }

        tasks_payload = {
            "premium": False,
        }

        first_request_payload = {
            "input_spec": {
                "prompt": prompt,
                "style": styles[style],
            }
        }

        # get id of task
        tasks_response = requests.post(self.tasks_url, json=tasks_payload, headers=self.headers)
        tasks_response_data = json.loads(tasks_response.text)
        task_id = tasks_response_data["id"]

        # use the id to pass in the prompt and style
        # state is pending
        requests.put(url=self.tasks_url + task_id, headers=self.headers,
                     json=first_request_payload)
        time.sleep(1)

        # state is generating
        second_request = requests.get(url="https://paint.api.wombo.ai/api/tasks/" + task_id, headers=self.headers)
        task_state = json.loads(second_request.text)["state"]

        return task_id, task_state

    def get_task_state(self, task_id):

        # repeat second request until state is completed
        state_request = requests.get(url="https://paint.api.wombo.ai/api/tasks/" + task_id, headers=self.headers)
        data = json.loads(state_request.text)
        task_state = data["state"]
        photo_url_list = data["photo_url_list"]
        return task_state, photo_url_list

    def get_final_result(self, task_id):

        final_request = requests.get(url="https://paint.api.wombo.ai/api/tasks/" + task_id, headers=self.headers)
        final_result = json.loads(final_request.text)["result"]["final"]

        return final_result


def get_token():
    x_goog_api_key = "AIzaSyDxCoSRCFvdsYcJalNfBQQfGl0-YycRkdE"
    url = f"https://www.googleapis.com/identitytoolkit/v3/relyingparty/signupNewUser?key={x_goog_api_key}"
    headers = {
        "accept-encoding": "gzip",
        "accept-language": "en-DE, en-US",
        "connection": "Keep-Alive",
        "content-encoding": "gzip",
        "content-type": "application/json",
        "host": "www.googleapis.com",
        "user-agent": "Dalvik/2.1.0 (Linux; U; Android 12; POCO F1 Build/SD1A.210817.036)",
        "x-android-package": "com.womboai.wombodream",
        "x-android-cert": "659AA1EACE253B8667AA28414BF5E21ACD798A4D",
        "x-client-version": "Android/Fallback/X21000001/FirebaseCore-Android"
    }
    payload = {}

    response = requests.post(url, headers=headers, json=payload)

    data = json.loads(response.text)
    token = data['idToken']
    # print(response.text)
    return token