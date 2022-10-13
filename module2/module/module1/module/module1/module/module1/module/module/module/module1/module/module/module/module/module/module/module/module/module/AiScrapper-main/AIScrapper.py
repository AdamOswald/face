from typing import List
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

import threading
import urllib.request
import time
import os
from PIL import Image


def getImage(category,inputText,iteration):

    browserOptions = Options()
    browserOptions.add_argument("--window-size=1920,1080")
    browserOptions.add_argument("--start-maximized")
    browserOptions.add_argument("--headless")

    driver = webdriver.Chrome(executable_path='chromedriver',options=browserOptions)

    driver.get("https://app.wombo.art/")


    textfield = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH ,"//input[@class=\"TextInput__Input-sc-1qnfwgf-1 bDjNPR PromptConfig__StyledInput-sc-1p3eskz-0 iVLOGq\"]")))
    textfield.send_keys(inputText)

    categoryBox = WebDriverWait(driver,20).until(EC.element_to_be_clickable((By.XPATH,f"//img[@class=\"Thumbnail__StyledThumbnail-sc-p7nt3c-0 gVABqX\" and @alt=\"{category}\"]")))
    categoryBox.click()

    time.sleep(1)

    startButton = WebDriverWait(driver,20).until(EC.element_to_be_clickable((By.XPATH,"//button[@class=\"Button-sc-1fhcnov-2 bZmxDe\"]")))
    startButton.click()

    resultImage = WebDriverWait(driver,100).until(EC.element_to_be_clickable((By.XPATH,"//img[@class=\"ArtCard__CardImage-sc-bttd39-1 fHqXjT\"]")))
    resultImageSrc = resultImage.get_attribute('src')

    inputText = inputText.replace(" ", "_")
    category = category.replace(" ", "_")

    urllib.request.urlretrieve(resultImageSrc, f"{inputText}/{str(iteration) + inputText + category}.png")

    time.sleep(1)

    im = Image.open(f"{inputText}/{str(iteration) + inputText + category}.png")
    im = im.crop((65, 165, 950, 1510))
    im.save(f"{inputText}/{str(iteration) + inputText + category}.png")




categories = ["Mystical","Festive","Dark Fantasy","Psychic","Pastel","HD","Vibrant","Fantasy Art","Steampunk","Ukiyoe","Synthwave"]
threads = []

inputText = input("Input: ")
iterations = int(input("Iterations: "))

os.mkdir(inputText.replace(" ", "_"))

for i in categories:
    for j in range(iterations):
        threads.append(threading.Thread(target=getImage, kwargs={'category':i,'inputText':inputText,'iteration':j}))

for i in threads:
    try:
        i.start()
    except:
        print("Error: unable to start thread")

for i in threads:
    i.join()
