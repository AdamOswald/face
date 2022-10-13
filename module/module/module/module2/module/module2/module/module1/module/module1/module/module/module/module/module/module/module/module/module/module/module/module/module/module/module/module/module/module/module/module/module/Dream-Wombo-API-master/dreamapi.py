import requests

class PuttedData():
    def __init__(self,json):
        self.id = json['id']
        self.uid = json['user_id']
        self.input_prompt = json['input_spec']['prompt']
        self.input_style = json['input_spec']['style']
        self.state = json['state']
        self.created_date = json['created_at']
        self.updated_date = json['updated_at']

class CheckData():
    def __init__(self,json):
        self.id = json['id']
        self.state = json['state']
        self.resulturl = json['result']
        if self.resulturl != None:
            self.resulturl=json['result']['final']
        self.process_urls = json['photo_url_list']

class PostData():
    """
    RUS: Возвращает информацию об запросе
    ENG: Return info about request
    """
    def __init__(self,json):
        self.id = json['id']
        self.uid = json['user_id']
        self.state = json['state']

class DreamAPI():
    def __init__(self):
        self.headers = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36','content-type':'application/json'}
        self.url = 'https://paint.api.wombo.ai/api/tasks/'
        self.gen_token()

    def see_all_styles(self, print:bool = False):
        """
        RUS: Выводит все возможные стили и возвращает словарь
        ENG: Print all styles and return dict
        :param print (boolean) - Выводить все стили в консоль | Print all styles in console
        """
        styles={44:'Meme',32:'Realistic',35:'Throwback',3:'No Style',22:'Ghibli',28:'Melancholic',17:'Provenance',34:'Arcane',27:'Radioactive',7:'HD',20:'Blacklight',18:'Rose gold',16:'Wuhtercuhler',15:'S.Dali',14:'Etching',13:'Baroque',11:'Mystical',10:'Dark Fantasy',9:'Psychic',6:'Vibrant',5:'Fantasy Art',4:'Steampunk',21:'Psychedelic',1:'Synthwave',2:'Ukiyoe'}
        if print:
            for x in styles.keys():
                print(f'{str(x)} - {styles[x]}')
        return styles

    def post(self):
        """
        RUS: Сделать запрос на заявку для генерации
        ENG: Make a request for an application for generation
        """
        r = requests.post(self.url,headers=self.headers,json={'premium':False})
        return PostData(json=r.json())

    def put_prompt(self,id:str,prompt:str,style:int = 3):
        r = requests.put(self.url+id,headers=self.headers,json={'input_spec':{'prompt': prompt, 'style': style, 'display_freq': 10}})
        return PuttedData(r.json())

    def gen_token(self, returntoken:bool = False):
        """
        RUS: Генерирует токен для аутентификации
        ENG: Generate token for auth
        :param returntoken (boolean) - Return token|Вернуть токен - def:False
        """
        r = requests.post('https://identitytoolkit.googleapis.com/v1/accounts:signUp?key=AIzaSyDCvp5MTJLUdtBYEKYWXJrlLzu1zuKM6Xw', json={'returnSecureToken': True})
        token = r.json()['idToken']
        self.headers.update({'authorization':'bearer ' + token})
        if returntoken: r.json()['idToken']
        else:   return True

    def check(self,id:str):
        """
        RUS: Проверяет, сгенерировалась ли изображение
        ENG: Checks if the image has been generated
        """
        r = requests.get(self.url+id,headers=self.headers)
        return CheckData(r.json())


if __name__ == '__main__':
    from time import sleep
    api = DreamAPI()
    x = api.post()
    id = x.id
    print('id = '+id)
    api.put_prompt(id=id,prompt='Wombo Dream API')
    while True:
        sleep(10)
        temp = api.check(id=id)
        if temp != None:
            print(temp.resulturl)
            break
        
