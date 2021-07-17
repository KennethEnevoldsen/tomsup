# import packages
from psychopy import core, gui, event, visual
import os
import pandas as pd

# add tomsup to path
import sys

sys.path.insert(1, "/Users/au561649/Desktop/Github/tomsup/python package")
import tomsup as ts

# ------------- Getting participant information ---------

# check if ID exists:
if os.path.exists("data"):
    l = os.listdir("data")
    ID = max([int(i.split("_")[-1].split(".")[0]) for i in l])
    ID = ID + 1


# Pop-up asking for participant info
popup = gui.Dlg(title="Matching Pennies")
popup.addField("ID: ", ID)
popup.addField("Age: ", 21)
popup.addField("Gender", choices=["Male", "Female", "Other"])
popup.addField("ToM level", choices=["0", "1", "2", "3", "4"])
popup.addField("Number of trials", 2)
popup.show()

if popup.OK:
    ID = popup.data[0]
    age = popup.data[1]
    gender = popup.data[2]
    k = popup.data[3]
    n_trials = popup.data[4]
elif popup.Cancel:
    core.quit()

print(f"this is {k}")

# ------------- create agent and payoff matrix ---------
tom = ts.create_agents(agents="RB")  # this need to be changed
penny = ts.PayoffMatrix(name="penny_competitive")

# ------------- Defining Variables and function ---------
intro0 = f"""Dear participant

In the following experiment you will compete against another person in the matching pennies game for {n_trials}.

Before starting the experiment, we would like to inform you that no personal information is collected, and that beside the given information no personal information will be recorded. If at any time you should feel uncomfortable, you are free to stop the experiment and ask for any generated data to be deleted.
If you have read the above and agree to proceed, press ENTER."""

intro1 = f"""We will now briefly explain the rules of the game.

You will see two closed hands. Your opponent will have hidden a penny in either one of them. Yours goal is to figure out which of the two hands contain the penny.
If you guess right, you get a point, if not, your opponnent gains a point.
If you have read the above and understand the rules, press ENTER."""


def show_text(txt):  # Show_text for normal text
    msg = visual.TextStim(win, text=txt, height=0.05)
    msg.draw()
    win.flip()
    k = event.waitKeys(
        keyList=["return", "escape"]
    )  # press enter to move on or ecape to quit
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

# ---------- Preparing dataframe for CSV ----------
trial_list = []
for trial in range(n_trials):
    trial_list += [
        {
            "ID": ID,
            "age": age,
            "gender": gender,
            "tom_level": k,
            "trialnr": trial,
            "Response_participant": "",
            "Response_tom": "",
            "payoff_participant": "",
            "payoff_tom": "",
            "RT": "",
        }
    ]

# ------------- Running the experiment ---------

# run intro
show_text(intro0)
show_text(intro1)
op_choice = None  # setting opponenent choice to none for the first round
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

    k = event.waitKeys(keyList=["escape", "left", "right"])
    if k[0] == "escape":
        core.quit()
    if k[0] == "left":
        resp_part = 0
        trial["RT"] = stopwatch.getTime()
        picture1 = visual.ImageStim(
            win, image=RH_open, pos=img_pos1, units="norm", size=img_size
        )
        picture2 = visual.ImageStim(
            win, image=LH_coin, pos=img_pos2, units="norm", size=img_size
        )
        picture1.draw()
        picture2.draw()
        win.flip()
        core.wait(wait_time)
    elif k[0] == "right":
        resp_part = 1
        trial["RT"] = stopwatch.getTime()
        picture1 = visual.ImageStim(
            win, image=RH_coin, pos=img_pos1, units="norm", size=img_size
        )
        picture2 = visual.ImageStim(
            win, image=LH_open, pos=img_pos2, units="norm", size=img_size
        )
        picture1.draw()
        picture2.draw()
        win.flip()
        core.wait(wait_time)

    # get ToM response
    resp_tom = tom.compete(p_matrix=penny, op_choice=op_choice, agent=1)

    # get payoff
    payoff_part = penny.payoff(
        action_agent0=resp_part, action_agent1=resp_tom, agent=0
    )  # agent0 is seeker e.g. the participant
    payoff_tom = penny.payoff(action_agent0=resp_part, action_agent1=resp_tom, agent=1)

    # Give response text
    rl_tom = "left" if resp_tom == 0 else "right"
    current_score_part += payoff_part
    current_score_tom += payoff_tom
    show_text(
        f"You choose {k[0]} and penny was in the {rl_tom} hand. This results in a payoff of {payoff_part}. This mean that \n\n your current score is: "
        + f"{current_score_part} \n your opponents score is: {current_score_tom}. Press enter to continue."
    )

    # Save data
    trial["Response_participant"] = resp_part
    trial["Response_tom"] = resp_tom
    trial["payoff_participant"] = payoff_part
    trial["payoff_tom"] = payoff_tom

    # write data (writes at each trial, so that even if the program crashes there should be an issue)
    pd.DataFrame(trial_list).to_csv("data/ID_" + str(ID) + ".csv")

if not os.path.exists("data"):
    os.makedirs("data")

# write data
pd.DataFrame(trial_list).to_csv("data/ID_" + str(ID) + ".csv")