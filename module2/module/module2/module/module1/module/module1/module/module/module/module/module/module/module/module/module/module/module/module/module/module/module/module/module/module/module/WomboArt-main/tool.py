import requests, os
from colorama import Fore, init

init(autoreset=True)

def generate(title: str):
    os.system("cls")
    print(f"{Fore.LIGHTBLUE_EX}Retrieving Auth Token.")

    getAuthToken = requests.post("https://identitytoolkit.googleapis.com/v1/accounts:signUp?key=AIzaSyDCvp5MTJLUdtBYEKYWXJrlLzu1zuKM6Xw", headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:96.0) Gecko/20100101 Firefox/96.0",
        "Accept": "*/*",
        "Accept-Language": "fr,fr-FR;q=0.8,en-US;q=0.5,en;q=0.3",
        "Accept-Encoding": "gzip, deflate, br",
        "content-type": "application/json",
        "x-client-version": "Firefox/JsCore/9.1.2/FirebaseCore-web",
        "Origin": "https://app.wombo.art",
        "DNT": "1",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "cross-site",
        "TE": "trailers"
    }, json={
        "returnSecureToken": "true"
    }).json()

    authToken = getAuthToken["idToken"]

    os.system("cls")
    print(f"{Fore.LIGHTBLUE_EX}Initiating task.")

    initTask = requests.post("https://app.wombo.art/api/tasks", headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:96.0) Gecko/20100101 Firefox/96.0",
        "Accept": "*/*",
        "Accept-Language": "fr,fr-FR;q=0.8,en-US;q=0.5,en;q=0.3",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://app.wombo.art/",
        "Authorization": f"bearer {authToken}",
        "Content-Type": "text/plain;charset=UTF-8",
        "Origin": "https://app.wombo.art",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "TE": "trailers"
    }, json={
        "premium": False
    }).json()

    taskID = initTask["id"]

    os.system("cls")
    print(f"{Fore.LIGHTBLUE_EX}Initiating art creation.")

    initCreateTask = requests.put(f"https://app.wombo.art/api/tasks/{taskID}", headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:96.0) Gecko/20100101 Firefox/96.0",
        "Accept": "*/*",
        "Accept-Language": "fr,fr-FR;q=0.8,en-US;q=0.5,en;q=0.3",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://app.wombo.art/",
        "Authorization": f"bearer {authToken}",
        "Content-Type": "text/plain;charset=UTF-8",
        "Origin": "https://app.wombo.art",
        "DNT": "1",
        "Connection": "keep-alive",
        "Cookie": "_ga_BRH9PT4RKM=GS1.1.1644347760.1.0.1644347820.0; _ga=GA1.1.1610806426.1644347761",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "TE": "trailers"
    }, json={
        "input_spec": {
            "prompt": title,
            "style": 18,
            "display_freq": 10
        }
    })

    os.system("cls")
    print(f"{Fore.LIGHTBLUE_EX}Starting generating art.")

    createArt = {}
    createArt["state"] = "generating"
    while createArt["state"] == "generating":
        createArt = requests.get(f"https://app.wombo.art/api/tasks/{taskID}", headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:96.0) Gecko/20100101 Firefox/96.0",
            "Accept": "*/*",
            "Accept-Language": "fr,fr-FR;q=0.8,en-US;q=0.5,en;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://app.wombo.art/",
            "Authorization": f"bearer {authToken}",
            "DNT": "1",
            "Connection": "keep-alive",
            "Cookie": "_ga_BRH9PT4RKM=GS1.1.1644347760.1.0.1644347820.0; _ga=GA1.1.1610806426.1644347761",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "TE": "trailers"
        }).json()

        os.system("cls")
        print(f"{Fore.LIGHTYELLOW_EX}Art is still generating.")

    artURL = createArt["result"]["final"]

    if not os.path.isdir("results/"):
        os.makedirs("results/")
    
    with open(f"results/{taskID}.png", "wb") as file:
        file.write(requests.get(artURL).content)
        file.close()

    os.system(f"explorer {os.getcwd()}\\results\{taskID}.png")

    os.system("cls")
    print(f"{Fore.LIGHTGREEN_EX}Art generated with success! {artURL}")

    input()

if __name__ == "__main__":
    try:
        os.system("cls")
        title = input(f"{Fore.LIGHTYELLOW_EX}Please enter your art title.\n\n{Fore.RESET}~# ")

        generate(title)
    except KeyboardInterrupt:
        exit()