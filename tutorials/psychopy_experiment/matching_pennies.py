# ------------- Setup -------------
# import packages
import os

import pandas as pd
from psychopy import core, event, gui, visual

import tomsup as ts

# Set path to location of the file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# ------------- Getting participant information -------------


# Create data folder if it doesn't exist
if not os.path.exists("data"):
    os.mkdir("data")

# Get out names of data in the datafolder
l = os.listdir("data")

# If there is data already present
if l:
    # Find the max ID and set it 1 higher
    ID = max([int(i.split("_")[-1].split(".")[0]) for i in l if i.startswith("ID")])
    ID = ID + 1
else:
    # Otherwise start at 1
    ID = 1


# Pop-up asking for participant info
popup = gui.Dlg(title="Matching Pennies")
popup.addField("ID: ", ID)
popup.addField("Age: ", 21)
popup.addField("Gender", choices=["Male", "Female", "Other"])
popup.addField("Number of trials", 2)
popup.addField("Game type", choices=["penny_competitive", "penny_cooperative"])
popup.addField(
    "Opponent Strategy",
    choices=["RB", "WSLS", "TFT", "QL", "0-TOM", "1-TOM", "2-TOM", "3-TOM", "4-TOM"],
)
popup.addField("Opponent parameters", "{}")
popup.addField("Save opponent internal states", choices=["False", "True"])
popup.show()

if popup.OK:
    ID = popup.data[0]
    age = popup.data[1]
    gender = popup.data[2]
    n_trials = popup.data[3]
    game_type = popup.data[4]
    opponent_strategy = popup.data[5]
    opponent_params_str = popup.data[6]
    save_history_str = popup.data[7]

elif popup.Cancel:
    core.quit()

if save_history_str == "True":
    save_history = True
else:
    save_history = False

exec(f"opponent_params = {opponent_params_str}")

# ------------- create agent and payoff matrix -------------
opponent_params["save_history"] = save_history
tom = ts.create_agents(agents=opponent_strategy, start_params=opponent_params)
penny = ts.PayoffMatrix(name=game_type)

# ------------- Defining Variables and function -------------
introtext = """Dear participant

Thank you for playing against tomsup!
Here we will make you play against simulated agents in simple decision-making games.

We ask you for some basic demographic information. Apart from that, only performance in the game is collected.
If you at any time wish to do so, you are free to stop the experiment and ask for any generated data to be deleted.
If you have read the above and wish to proceed, press ENTER."""

rulestext_pennycompetitive = f"""
You will now play a game of competitive matching pennies.

You will see the two hands of your opponent, one on the left, the other on the right.
One of the hands hides a penny. Your goal is to figure out which of the two hands contain the penny.
If you guess the correct hand, you get a point and your opponent loses a point.
If you guess incorrectly, you lose a point and your opponent gains a point.
The game will last for {n_trials} trials.

By pressing the "right arrow" on your keyboard, you guess "right".
By pressing the "left arrow" on your keyboard, you guess "left".
After guessing, press ENTER to continue.
To quit the game, press ESCAPE.
When you have read and understood the above, press ENTER to continue."""

rulestext_pennycooperattive = f"""
You will now play a game of cooperative matching pennies.

You will see the two hands of your opponent, one on the left, the other on the right.
One of the hands hides a penny. Your goal is to figure out which of the two hands contain the penny.
If you guess the correct hand, you and your opponent both get get a point.
If you guess incorrectly, you and your opponent both lose a point.
The game will last for {n_trials} trials.

By pressing the "right arrow" on your keyboard, you guess "right".
By pressing the "left arrow" on your keyboard, you guess "left".
After guessing, press ENTER to continue.
To quit the game, press ESCAPE.
When you have read and understood the above, press ENTER to continue."""

# Set rulestext to fit the specified game
if game_type == "penny_competitive":
    rulestext = rulestext_pennycompetitive
elif game_type == "penny_cooperative":
    rulestext = rulestext_pennycooperattive


# Show_text for normal text
def show_text(txt):
    msg = visual.TextStim(win, text=txt, height=0.05)
    msg.draw()
    win.flip()
    k = event.waitKeys(
        keyList=["return", "escape"]
    )  # press enter to move on or escape to quit
    if k[0] in ["escape"]:
        core.quit()


# setting window and reading images
win = visual.Window(fullscr=False)
stopwatch = core.Clock()
RH_closed = "images/RH_closed.png"
LH_closed = "images/LH_closed.png"
LH_open = "images/LH_open.png"
RH_open = "images/RH_open.png"
LH_coin = "images/LH_coin.png"
RH_coin = "images/RH_coin.png"

# ---------- Preparing dataframe for CSV -------------
trial_list = []
for trial in range(n_trials):
    trial_list += [
        {
            "ID": ID,
            "age": age,
            "gender": gender,
            "opponent_strategy": opponent_strategy,
            "trialnr": trial,
            "Response_participant": "",
            "Response_tom": "",
            "payoff_participant": "",
            "payoff_tom": "",
            "RT": "",
        }
    ]

# ------------- Running the experiment -------------

# run intro
show_text(introtext)
show_text(rulestext)
op_choice = None  # setting opponent choice to none for the first round
current_score_part = 0
current_score_tom = 0

img_pos1 = [-0.50, 0.0]
img_pos2 = [0.50, 0.0]
img_size = [0.9, 0.9]

wait_time = 2

for trial in trial_list:
    picture1 = visual.ImageStim(
        win, image=RH_closed, pos=img_pos1, units="norm", size=img_size
    )
    picture2 = visual.ImageStim(
        win, image=LH_closed, pos=img_pos2, units="norm", size=img_size
    )
    picture1.draw()
    picture2.draw()
    stopwatch.reset()
    win.flip()

    resp_part = None  # setting participant response to none for the first round

    # get ToM response
    resp_tom = tom.compete(p_matrix=penny, op_choice=resp_part, agent=0)

    k = event.waitKeys(keyList=["escape", "left", "right"])

    # get participant response
    if k[0] == "escape":
        core.quit()
    elif k[0] == "left":
        resp_part = 0
    elif k[0] == "right":
        resp_part = 1
    trial["RT"] = stopwatch.getTime()

    #
    if resp_tom == 0:  # left hand
        rl_tom = "left"
        picture1 = visual.ImageStim(
            win,
            image=RH_coin,  # agent point of view
            pos=img_pos1,
            units="norm",
            size=img_size,
        )
        picture2 = visual.ImageStim(
            win, image=LH_open, pos=img_pos2, units="norm", size=img_size
        )
    elif resp_tom == 1:  # right hand
        rl_tom = "right"
        picture1 = visual.ImageStim(
            win, image=RH_open, pos=img_pos1, units="norm", size=img_size
        )
        picture2 = visual.ImageStim(
            win, image=LH_coin, pos=img_pos2, units="norm", size=img_size
        )
    picture1.draw()
    picture2.draw()
    win.flip()
    event.waitKeys()

    # get payoff
    payoff_part = penny.payoff(
        choice_agent0=resp_part, choice_agent1=resp_tom, agent=1
    )  # agent0 is seeker e.g. the participant
    payoff_tom = penny.payoff(choice_agent0=resp_part, choice_agent1=resp_tom, agent=0)

    # Give response text
    current_score_part += payoff_part
    current_score_tom += payoff_tom
    show_text(
        f"You chose {k[0]} and the penny was in the {rl_tom} hand. This gives you {payoff_part} points while your opponent gets {payoff_tom} points.\n\n"
        + f"Your current score is: {current_score_part} \n Your opponent's current score is: {current_score_tom}. \nPress ENTER to continue."
    )

    # Save data
    trial["Response_participant"] = resp_part
    trial["Response_tom"] = resp_tom
    trial["payoff_participant"] = payoff_part
    trial["payoff_tom"] = payoff_tom
    if save_history:
        trial["tom_internal_states"] = tom.get_internal_states()

    # write data (writes at each trial, so that even if the program crashes there should be an issue)
    pd.DataFrame(trial_list).to_csv("data/ID_" + str(ID) + ".csv")

# write data
pd.DataFrame(trial_list).to_csv("data/ID_" + str(ID) + ".csv")

show_text(
    """
This concludes the game!
Thank you playing!

Press ENTER to quit.
"""
)

event.waitKeys()

# Close psychopy
core.quit()
# Close psychopy
core.quit()
