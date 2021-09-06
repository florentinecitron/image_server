


from logging import disable


def run_experiment_1():
    # list of images and their categories
    # all captions generated

    # random sample category
    # random sample 2 images
    # select caption of one image

    # show 2 images, caption, 2 buttons

    # generate as excel
    pass


def run_experiment_2():
    import random
    from collections import defaultdict
    import os.path
    import PySimpleGUI as sg
    import io
    from PIL import Image
    import json
    import requests

    #
    # ./ngrok http --region=us --hostname=eyebot.ngrok.io 8042

    image_base_path = "/home/nrg/datasets"
    vqa_url = "http://65e6-141-89-97-67.ngrok.io"
    image_list_file = "/home/nrg/projects/vqa-test/image_server/ADE_image_list.txt"
    with open(image_list_file) as i:
        images = [line.strip() for line in i]

    category_to_images = defaultdict(list)
    for i in images:
        image_split = i.split("/")
        image_id = image_split[-1]
        image_category = tuple(image_split[3:-1])
        category_to_images[image_category].append({"id": image_id, "image": i})

    category_to_images_gt2 = {k: v for k, v in category_to_images.items() if len(v) > 2}

    categories = list(category_to_images_gt2.keys())

    with open("dialogues-ex2.jsonl", "a") as o:
        # start a round
        category = random.choice(categories)
        images = image_pos, image_neg = random.sample(category_to_images_gt2[category], k=2)
        show_pos_first = random.random() > .5

        image_pos_path, image_neg_path = os.path.join(image_base_path, image_pos["image"]), os.path.join(image_base_path, image_neg["image"])

        round_data = {
            "images": {"positive": image_pos["image"], "negative": image_neg["image"]},
            "show_pos_first": show_pos_first,
            "dialogue_history": []
        }

        sg.theme('GreenTan') # give our window a spiffy set of colors

        ic = [Image.open(i) for i in ([image_pos_path, image_neg_path] if show_pos_first else [image_neg_path, image_pos_path])]

        for image in ic:
            image.thumbnail((400, 400), Image.ANTIALIAS)

        image_bytes_left = io.BytesIO()
        ic[0].save(image_bytes_left, format="PNG")
        image_bytes_right = io.BytesIO()
        ic[1].save(image_bytes_right, format="PNG")

        caption_pos = requests.get(f"{vqa_url}/get_caption", params={"image": image_pos["id"].split(".", 1)[0]}).text
        #caption_neg = requests.get(f"{vqa_url}/get_caption", params={"image": image_neg["id"].split(".", 1)[0]}).text

        #caption_left, caption_right = [caption_pos, caption_neg] if show_pos_first else [caption_neg, caption_pos]

        layout = [
            [
                sg.Image(key="-IMAGE-LEFT-", data=image_bytes_left.getvalue()),
                sg.Image(key="-IMAGE-RIGHT-", data=image_bytes_right.getvalue()),
            ], # Show pictures
            [
                sg.Text(text=caption_pos, justification='center', )
            ],
            [
                sg.Button('GUESS LEFT IMAGE', button_color=(sg.YELLOWS[0], sg.BLUES[0]), bind_return_key=True),
                sg.Button('GUESS RIGHT IMAGE', button_color=(sg.YELLOWS[0], sg.GREENS[0])),
                sg.Button('EXIT', button_color=(sg.YELLOWS[0], sg.GREENS[0]))
            ]
        ]

        window = sg.Window('Chat window', [[sg.Column(layout, element_justification='center')]], font=('Helvetica', ' 13'), default_button_element_size=(8,2), use_default_focus=False)

        while True:     # The Event Loop
            event, value = window.read()
            if event in (sg.WIN_CLOSED, 'EXIT'):  # quit if exit button or X
                break
            if event == 'GUESS LEFT IMAGE' or event == 'GUESS RIGHT IMAGE':
                user_guessed = "left" if event == 'GUESS LEFT IMAGE' else "right"
                round_data["user_guessed"] = user_guessed
                correct = user_guessed == "left" if show_pos_first else user_guessed == "right"
                print(json.dumps(round_data), file=o)
                break

        window.close()


def run_experiment_3():
    import random
    from collections import defaultdict
    import os.path
    import PySimpleGUI as sg
    import io
    from PIL import Image
    import json
    import requests

    #
    # ./ngrok http --region=us --hostname=eyebot.ngrok.io 8042

    image_base_path = "/home/nrg/datasets"
    vqa_url = "http://2f8f-141-89-97-67.ngrok.io"
    image_list_file = "/home/nrg/projects/vqa-test/image_server/ADE_image_list.txt"
    with open(image_list_file) as i:
        images = [line.strip() for line in i]

    category_to_images = defaultdict(list)
    for i in images:
        image_split = i.split("/")
        image_id = image_split[-1]
        image_category = tuple(image_split[3:-1])
        category_to_images[image_category].append({"id": image_id, "image": i})

    category_to_images_gt2 = {k: v for k, v in category_to_images.items() if len(v) > 2}

    categories = list(category_to_images_gt2.keys())

    with open("dialogues.jsonl", "a") as o:
        # start a round
        category = random.choice(categories)
        images = image_pos, image_neg = random.sample(category_to_images_gt2[category], k=2)
        show_pos_first = random.random() > .5

        image_pos_path, image_neg_path = os.path.join(image_base_path, image_pos["image"]), os.path.join(image_base_path, image_neg["image"])

        round_data = {
            "images": {"positive": image_pos["image"], "negative": image_neg["image"]},
            "show_pos_first": show_pos_first,
            "dialogue_history": []
        }

        sg.theme('GreenTan') # give our window a spiffy set of colors

        ic = [Image.open(i) for i in ([image_pos_path, image_neg_path] if show_pos_first else [image_neg_path, image_pos_path])]

        for image in ic:
            image.thumbnail((400, 400), Image.ANTIALIAS)

        image_bytes_left = io.BytesIO()
        ic[0].save(image_bytes_left, format="PNG")
        image_bytes_right = io.BytesIO()
        ic[1].save(image_bytes_right, format="PNG")

        layout = [
            [
                sg.Image(key="-IMAGE-LEFT-", data=image_bytes_left.getvalue()),
                sg.Image(key="-IMAGE-RIGHT-", data=image_bytes_right.getvalue())
            ], # Show pictures
            [
                sg.Multiline(size=(110, 20), font=('Helvetica 10'), disabled=True, autoscroll=True, key="-OUTPUT-")
            ],
            [
                sg.Multiline(size=(70, 5), enter_submits=False, key='-QUERY-', do_not_clear=False),
                sg.Button('ASK', button_color=(sg.YELLOWS[0], sg.BLUES[0]), bind_return_key=True),
                sg.Button('EXIT', button_color=(sg.YELLOWS[0], sg.GREENS[0]))
            ],
            [
                sg.Button('GUESS LEFT IMAGE', button_color=(sg.YELLOWS[0], sg.BLUES[0]), bind_return_key=True),
                sg.Button('GUESS RIGHT IMAGE', button_color=(sg.YELLOWS[0], sg.GREENS[0])),
            ]
        ]

        window = sg.Window('Chat window', layout, font=('Helvetica', ' 13'), default_button_element_size=(8,2), use_default_focus=False)

        while True:     # The Event Loop
            event, value = window.read()
            if event in (sg.WIN_CLOSED, 'EXIT'):  # quit if exit button or X
                break
            if event == 'ASK':
                query = value['-QUERY-'].rstrip()
                # EXECUTE YOUR COMMAND HERE

                window['-OUTPUT-'].update(f'Q: {query}\n', append=True)

                answer = requests.get(f"{vqa_url}/answer_question", params={"image": image_pos["id"].split(".", 1)[0], "question": query}).text

                # curl 'localhost:5000/answer_question?image=ADE_train_00001505&question=what+color+is+the+building?'

                window['-OUTPUT-'].update(f'A: {answer}\n', append=True)

                round_data["dialogue_history"].append({"question": query, "answer": answer})
            if event == 'GUESS LEFT IMAGE' or event == 'GUESS RIGHT IMAGE':
                user_guessed = "left" if event == 'GUESS LEFT IMAGE' else "right"
                round_data["user_guessed"] = user_guessed
                correct = user_guessed == "left" if show_pos_first else user_guessed == "right"
                print(json.dumps(round_data), file=o)
                break

        window.close()