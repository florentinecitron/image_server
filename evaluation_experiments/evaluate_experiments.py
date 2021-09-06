import json
import tabulate
from collections import Counter
import random
from pathlib import Path

image_base_path = Path("/home/nrg/datasets")

def eval_experiment_captions_1():
    with open("evaluation_data/dialogues-caption-based.jsonl") as i:
        data = [json.loads(l) for l in i]
    for el in data:
        el["is_correct"] = el["user_guessed"] == "left" if el["show_pos_first"] else el["user_guessed"] == "right"
    n = len(data)
    correct = sum([el["user_guessed"] == "left" if el["show_pos_first"] else el["user_guessed"] == "right" for el in data])
    accuracy = correct / n

    import matplotlib.pylab as plt
    random_incorrect = random.sample([el for el in data if not el["is_correct"]], k=2)
    fig, axes = plt.subplots(2,2)
    for idx, image_pair in enumerate(random_incorrect):
        im_left = plt.imread(image_base_path/image_pair["images"]["positive"])
        im_right = plt.imread(image_base_path/image_pair["images"]["negative"])
        fig_left = axes[idx][0].imshow(im_left)
        fig_right = axes[idx][1].imshow(im_right)
        #fig_left.axes.get_xaxis().set_visible(False)
        #fig_left.axes.get_yaxis().set_visible(False)
        if idx == 0:
            caption = 'a large group of steps in a garden.'
        else:
            caption = 'a couple of metal stairs on the side of a brick building.'
        fig_left.axes.set_xlabel(r'\begin{center}\textit{\small{' + caption + r'}}\end{center}')
        for subfig in [fig_left, fig_right]:
            ax = subfig.axes
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        #fig_right.axes.get_xaxis().set_visible(False)
        #fig_right.axes.get_yaxis().set_visible(False)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
    plt.show()

    # show

def eval_experiment_vqa():
    with open("evaluation_data/dialogues-vqa-based.jsonl") as i:
        data = [json.loads(l) for l in i]
    for el in data:
        el["is_correct"] = el["user_guessed"] == "left" if el["show_pos_first"] else el["user_guessed"] == "right"

    n = len(data)
    correct = sum([el["is_correct"] for el in data])
    accuracy = correct / n

    # create histograms of dialogue lengths
    dialogue_length_histogram = Counter(len(el["dialogue_history"]) for el in data)
    dialogue_length_histogram_correct = Counter(len(el["dialogue_history"]) for el in data if el["is_correct"])
    dialogue_length_histogram_incorrect = Counter(len(el["dialogue_history"]) for el in data if not el["is_correct"])

    # collect stats about questions asked
    n_color = 0 #"color"
    n_person = 0
    total_qs = 0
    n_binary_answer = 0
    n_how_many = 0
    import nltk
    count_words = Counter()
    for el in data:
        for q in el["dialogue_history"]:
            if "color" in q["question"] or "colour" in q["question"]:
                n_color += 1
            if "person" in q["question"] or "people" in q["question"]:
                n_person += 1
            if q["answer"] == "yes" or q["answer"] == "no":
                n_binary_answer += 1
            total_qs += 1
            count_words.update(nltk.wordpunct_tokenize(q["question"].lower()))

    # create image
    random_incorrect = random.sample([el for el in data if not el["is_correct"]], k=2)
    fig, axes = plt.subplots(2,3)
    for idx, image_pair in enumerate(random_incorrect):
        im_left = plt.imread(image_base_path/image_pair["images"]["positive"])
        im_right = plt.imread(image_base_path/image_pair["images"]["negative"])
        fig_left = axes[idx][0].imshow(im_left)
        fig_center = axes[idx][1].imshow(im_right)

        dialogue = "\n".join(f"Q: {turn['question']}\nA: {turn['answer']}" for turn in image_pair["dialogue_history"])
        fig_right = axes[idx][2].text(0, .5, dialogue, horizontalalignment='left', verticalalignment='center') #, transform=ax.transAxes)

        for subfig in [fig_left, fig_center, fig_right]:
            ax = subfig.axes
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()

def eval_experiment_mapworld():
    game_avatar_file = "evaluation_data/game_avatar.out"

    dialogues = []
    dialogue = []

    with open(game_avatar_file) as i:
        for line in i:
            _, line = line.strip().split(" ", 1)
            if not line.startswith("[socket.io packet"):
                continue
            a, b, c, rest = line.split(" ", 3)

            if not rest.startswith("b'2[\"text_message\""):
                continue

            data = json.loads(rest[3:-1].replace("\\", ""))
            message = data[1]

            if isinstance(message["msg"], str) and message["msg"].startswith("You are a rescue bot."):
                #if isinstance(message["msg"], dict) and "situation" in message["msg"]["observation"]:
                dialogues.append(dialogue)
                dialogue = [message]
            else:
                dialogue.append(message)
        dialogues.append(dialogue)


    dialogue_info = []
    for dialogue in dialogues:
        n_questions_user = len([d["msg"] for d in dialogue if d["user"]["id"] == 3])
        player_won = any(isinstance(d["msg"], str) and d["msg"].startswith("The player ended the game and was lucky.") for d in dialogue)
        cnt_directions = 0
        cnt_describe = 0
        cnt_go_direction = 0
        cnt_vqa = 0

        player_questions = [d["msg"]  for d in dialogue if d["user"]["id"] == 3]
        for q in [q.lower() for q in player_questions]:
            if "what" in q and "see" in q:
                cnt_describe += 1
            elif "where" in q and "go" in q:
                cnt_directions += 1
            elif "go" in q:
                cnt_go_direction += 1
            else:
                cnt_vqa += 1
        dialogue_info.append({
            "n_questions_user": n_questions_user,
            "player_won": player_won,
            "cnt_directions": cnt_directions,
            "cnt_describe": cnt_describe,
            "cnt_go_direction": cnt_go_direction,
            "cnt_vqa": cnt_vqa
        })
    df = pd.DataFrame(dialogue_info)