
# Dream Wombo API
Библиотека для генерации изображений при помощи [Wombo Dream](https://app.wombo.art/  "Сайт")  
Library for generate pictures by [Wombo Dream](https://app.wombo.art/  "Сайт")
## Документация

Пример использования|Usage example:
```python
import dreamapi
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
```
***
`see_all_styles(print:bool = False)`  
RUS: Выводит все возможные стили и возвращает словарь  
ENG: Print all styles and return dict  
:param print (boolean) - Выводить все стили в консоль Print all styles in console
***
`post()`  
RUS: Сделать запрос на заявку для генерации  
ENG: Make a request for an application for generation
***
`put_prompt(id:str, prompt:str, style:int = 3)`  
RUS: Вкладывает по айди номер стиля и сам запрос  
ENG: Put prompt and style number by id
***
`check(id:str)`  
RUS: Проверяет, сгенерировалась ли изображение  
ENG: Checks if the image has been generated  
:param id (str) - id запроса | Request ID

