import tkinter as tk
from PIL import Image, ImageTk
import utils
import os
import urllib.request
import threading

# If "result" and "replacement" folder don't exist, create it and download the images from the GitHub repository
if not os.path.exists("result"):
    os.makedirs("result")
if not os.path.exists("replacement"):
    os.makedirs("replacement")
    for i in range(16):
        urllib.request.urlretrieve(f"https://raw.githubusercontent.com/wizpizz/DREAMpAI/master/replacement/{i+1}.png", f"./replacement/{i+1}.png")
    urllib.request.urlretrieve("https://raw.githubusercontent.com/wizpizz/DREAMpAI/master/replacement/r.png", "./replacement/r.png")


# Resite value for the generated images
resize = (int(320 / 3), int(522 / 3))

root = tk.Tk()
Request = utils.Request()

# Set window title to Wombo Dream
root.title("DREAMpAI")

# Create a label widget that says "Generated Images"
generated_label = tk.Label(root, text="Generated Images")
generated_label.grid(row=0, column=0, columnspan=4)

# Declaring generated image labels
_1_image_label = tk.Label(root)
_2_image_label = tk.Label(root)
_3_image_label = tk.Label(root)
_4_image_label = tk.Label(root)
_5_image_label = tk.Label(root)
_6_image_label = tk.Label(root)
_7_image_label = tk.Label(root)
_8_image_label = tk.Label(root)
_9_image_label = tk.Label(root)
_10_image_label = tk.Label(root)
_11_image_label = tk.Label(root)
_12_image_label = tk.Label(root)
_13_image_label = tk.Label(root)
_14_image_label = tk.Label(root)
_15_image_label = tk.Label(root)
_16_image_label = tk.Label(root)

functions_dict = {
    1: _1_image_label,
    2: _2_image_label,
    3: _3_image_label,
    4: _4_image_label,
    5: _5_image_label,
    6: _6_image_label,
    7: _7_image_label,
    8: _8_image_label,
    9: _9_image_label,
    10: _10_image_label,
    11: _11_image_label,
    12: _12_image_label,
    13: _13_image_label,
    14: _14_image_label,
    15: _15_image_label,
    16: _16_image_label
}


def generated_images_on_launch(is_first_time=True):
    # Either use replacement images or latest generated images
    if is_first_time:
        if os.path.exists("result/final.jpg"):
            image_slot = 1
            for file in os.listdir("./result"):
                if not file.startswith("final"):
                    temp_img = Image.open(f"./result/{file}")
                    temp_img = temp_img.resize(resize)
                    temp_img2 = ImageTk.PhotoImage(temp_img)
                    functions_dict[image_slot].configure(image=temp_img2)
                    functions_dict[image_slot].image = temp_img2
                    image_slot += 1
            while image_slot <= 16:
                temp_img = Image.open(f"./replacement/{image_slot}.png")
                temp_img = temp_img.resize(resize)
                temp_img2 = ImageTk.PhotoImage(temp_img)
                functions_dict[image_slot].configure(image=temp_img2)
                functions_dict[image_slot].image = temp_img2
                image_slot += 1
        else:
            image_slot = 1
            while image_slot <= 16:
                temp_img = Image.open(f"./replacement/{image_slot}.png")
                temp_img = temp_img.resize(resize)
                temp_img2 = ImageTk.PhotoImage(temp_img)
                functions_dict[image_slot].configure(image=temp_img2)
                functions_dict[image_slot].image = temp_img2
                image_slot += 1

    else:
        image_slot = 1
        while image_slot <= 16:
            temp_img = Image.open(f"./replacement/{image_slot}.png")
            temp_img = temp_img.resize(resize)
            temp_img2 = ImageTk.PhotoImage(temp_img)
            functions_dict[image_slot].configure(image=temp_img2)
            functions_dict[image_slot].image = temp_img2
            image_slot += 1

    _1_image_label.grid(row=1, column=0)
    _2_image_label.grid(row=1, column=1)
    _3_image_label.grid(row=1, column=2)
    _4_image_label.grid(row=1, column=3)
    _5_image_label.grid(row=2, column=0)
    _6_image_label.grid(row=2, column=1)
    _7_image_label.grid(row=2, column=2)
    _8_image_label.grid(row=2, column=3)
    _9_image_label.grid(row=3, column=0)
    _10_image_label.grid(row=3, column=1)
    _11_image_label.grid(row=3, column=2)
    _12_image_label.grid(row=3, column=3)
    _13_image_label.grid(row=4, column=0)
    _14_image_label.grid(row=4, column=1)
    _15_image_label.grid(row=4, column=2)
    _16_image_label.grid(row=4, column=3)


# Get final.jpg's size and resize it by 50%
if os.path.exists("./result/final.jpg"):
    img = Image.open("./result/final.jpg")
    img = img.resize((int(img.size[0] / 3), int(img.size[1] / 3)))
else:
    img = Image.open("./replacement/r.png")
    # img = img.resize((int(img.size[0] / 1.5), int(img.size[1] / 1.5)))

# Display the image
result_img = ImageTk.PhotoImage(img)
final_image_label = tk.Label(root, image=result_img)
final_image_label.grid(row=0, rowspan=5, column=4, padx=50, pady=50)

# Create a label widget that says "Latest Final Result"
latest_label = tk.Label(root, text="Latest Final Result")
latest_label.grid(row=1, column=4, sticky=tk.N, pady=60)

# Create an input field and make a function to get the input
prompt_input = tk.Entry(root, width=35)
prompt_input.grid(row=4, column=4, sticky=tk.E, padx=50)

# Create an option menu that has a list of options
style_option = tk.StringVar(root)
style_option.set("No Style")
style_option_menu = tk.OptionMenu(root, style_option, "No Style", "Etching", "Mystical", "Dark Fantasy", "Pastel",
                                  "Vibrant",
                                  "Steampunk", "Synthwave", "Baroque", "Festive", "Psychic", "HD", "Fantasy Art",
                                  "Ukiyoe")
style_option_menu.grid(row=4, column=4, sticky=tk.W, padx=50)


def start_make_request():
    threading.Thread(target=make_request()).start()


def make_request():
    generated_images_on_launch(is_first_time=False)

    for file in os.listdir("./result/"):
        os.remove("./result/" + file)

    task_id, task_state = Request.start_task(prompt=prompt_input.get(), style=style_option.get())

    latest_displayed = 0
    latest_updated = 0

    while task_state != "completed":
        if task_state == "pending":
            break
        task_state, photo_url_list = Request.get_task_state(task_id)
        latest_generated = photo_url_list[-1]
        latest_generated_number = latest_generated.split("/")[-1].split(".")[0]
        if latest_generated_number != latest_updated:
            latest_updated = latest_generated_number
            latest_displayed += 1
            urllib.request.urlretrieve(latest_generated, f"./result/{latest_generated_number}.jpg")
            root.after(500, update_generated_image(latest_displayed, latest_generated_number))

    # After the task is completed, get the final result and save it to a file
    urllib.request.urlretrieve(Request.get_final_result(task_id), "./result/final.jpg")

    # Change final_image_label's image to the new image "final.jpg" and display it
    result_image = Image.open("./result/final.jpg")
    result_image = result_image.resize((int(result_image.size[0] / 3), int(result_image.size[1] / 3)))
    result_img2 = ImageTk.PhotoImage(result_image)
    final_image_label.configure(image=result_img2)
    latest_label.configure(text=f'Final Result for "{prompt_input.get()}" [{style_option.get()}]')
    final_image_label.image = result_img2


# Create a button that sends the data to the API and displays the result
request_button = tk.Button(root, text="Request", height=2, width=10, command=start_make_request)
request_button.grid(row=4, column=4, sticky=tk.S, pady=20)


def update_generated_image(latest_displayed, latest_generated_number):
    latest_generated_image = Image.open(f"./result/{latest_generated_number}.jpg")
    latest_generated_image = latest_generated_image.resize(resize)
    latest_generated_image_tk = ImageTk.PhotoImage(latest_generated_image)
    that_image = functions_dict[int(latest_displayed)]
    that_image.update_idletasks()
    that_image.configure(image=latest_generated_image_tk)
    that_image.image = latest_generated_image_tk


# Stuff to run before mainloop
generated_images_on_launch(is_first_time=True)

root.mainloop()
