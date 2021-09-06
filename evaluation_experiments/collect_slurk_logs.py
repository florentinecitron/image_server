game_master_file = "evaluation_data/game_master.out"
game_avatar_file = "evaluation_data/game_avatar.out"

# socket.io packet received  ... ["text_message"]
# socket.io packet sent
import json

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