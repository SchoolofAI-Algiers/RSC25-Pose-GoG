import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation , PillowWriter
import random
from data_feeders.bone_pairs import ntu_pairs   # your file
from project_setup import NTU_FINAL_PROCESSED_DATA

# Convert 1-based to 0-based indices for Python
EDGES = [(i-1, j-1) for (i, j) in ntu_pairs if i > 0 and j > 0]

# Action class names for NTU RGB+D 60 dataset
NTU_ACTIONS = [
    "drink water", "eat meal/snack", "brushing teeth", "brushing hair", "drop", "pickup", "throw", "sitting down",
    "standing up", "clapping", "reading", "writing", "tear up paper", "wear jacket", "take off jacket", "wear shoe",
    "take off shoe", "wear on glasses", "take off glasses", "put on a hat/cap", "take off a hat/cap", "cheer up",
    "hand waving", "kicking something", "reach into pocket", "hopping", "jump up", "make a phone call", "play with phone/tablet",
    "typing on a keyboard", "pointing to something", "taking a selfie", "check time (watch)", "rub two hands",
    "nod head/bow", "shake head", "wipe face", "salute", "put palms together", "cross hands in front", "sneeze/cough",
    "staggering", "falling", "touch head (headache)", "touch chest (stomachache/heart pain)", "touch back (backache)",
    "touch neck (neckache)", "nausea/vomiting", "use fan (with hand or paper)/feeling warm", "punching/slapping other person",
    "kicking other person", "pushing other person", "pat on back of other person", "point finger at other person",
    "hugging other person", "giving something to other person", "touch other person's pocket", "handshaking",
    "walking towards each other", "walking apart from each other"
]

def plot_frame(joints, ax, color="blue"):
    """Draw skeleton frame for one person"""
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=color, s=20)
    for e in EDGES:
        ax.plot([joints[e[0], 0], joints[e[1], 0]],
                [joints[e[0], 1], joints[e[1], 1]],
                [joints[e[0], 2], joints[e[1], 2]], c=color)
        


def main():
    # Pick random dataset file
    fname = random.choice([NTU_FINAL_PROCESSED_DATA / "NTU60_CS.npz", NTU_FINAL_PROCESSED_DATA / "NTU60_CV.npz"])
    data = np.load(fname)

    # Pick random sequence
    X, Y = data["x_train"], data["y_train"]
    idx = random.randint(0, len(X) - 1)
    seq = X[idx]   # (300, 150)
    label_idx = np.argmax(Y[idx])
    action_name = NTU_ACTIONS[label_idx]

    # Person 1 and Person 2
    person1 = seq[:, :75].reshape(-1, 25, 3)
    person2 = seq[:, 75:].reshape(-1, 25, 3)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.cla()
        ax.set_title(f"Action: {action_name}", fontsize=12)
        ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])

        if (person1[frame] != 0).any():
            plot_frame(person1[frame], ax, color="blue")
        if (person2[frame] != 0).any():
            plot_frame(person2[frame], ax, color="red")

    anim = FuncAnimation(fig, update, frames=seq.shape[0], interval=100)
    # save
    # anim.save("skeleton_animation.gif", writer=PillowWriter(fps=10))

    plt.show()

if __name__ == "__main__":
    main()
