import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def cost_to_cost_rate(array):
    global acc, i, tmp
    acc = 0
    for i in range(len(array[2])):
        tmp = array[2][i]
        array[2][i] += acc
        array[2][i] /= (i+1)
        acc += tmp
"""
def smooth(x):
    # calculates the mean over a window of 50 data points (timestamps)
    n = x.shape[0]
    b = np.zeros((n,))
    for i in range(n):
        b[i] = x[max(0, i-50):min(n, i+50)].mean()
    return b

"""

def smooth(x):
    return x

def create_plots(files, keys, base_path_save):
    save_path_plots = base_path_save + '/stats_plots.png'
    save_path_df = base_path_save + '/stats_table.csv'
    scores_for_table = {}
    for i, file in enumerate(files):
        rewards, costs = file[1], file[2]
        avg_rewards = round(sum(rewards) / len(rewards), 5)
        avg_costs = round(sum(costs) / len(costs), 5)
        scores_for_table[keys[i]] = (avg_rewards, avg_costs)
        # Create a DataFrame from the dictionary
        df = pd.DataFrame(scores_for_table.items(), columns=['Parameter Value', 'Average Rewards and Costs'])
        # Split the tuple into two columns
        df[['Average Rewards', 'Average Costs']] = pd.DataFrame(df['Average Rewards and Costs'].tolist(), index=df.index)
        df = df.drop(columns=['Average Rewards and Costs'])
        df.to_csv(save_path_df, index=False)
        for file in files:
            cost_to_cost_rate(file)

        # Create a figure with 6 subplots (2 rows and 3 columns)
        fig, axs = plt.subplots(2, len(files), figsize=(30, 16))
        # Define colors for each experiment
        colors = ['r', 'b', 'g', 'purple', 'orange', 'cyan']
     #   colors = ['r','b']
        # Loop through rows (experiments)
        for i, data in enumerate(files):
            # Plot return in the first column
            axs[0, i].set_title(f'Return - {keys[i]}')
            axs[0, i].set_xlabel("Steps")
            axs[0, i].set_ylabel("Return")
            axs[0, i].plot(data[0][1:], np.array(data[1][1:]), label=keys[i], color=colors[i])
            # Plot mistake rate in the second column
            axs[1, i].set_title(f'Mistake Rate - {keys[i]}')
            axs[1, i].set_xlabel("Steps")
            axs[1, i].set_ylabel("Mistake Rate")
            axs[1, i].plot(data[0][1:], np.array(data[2][1:]), label=keys[i], color=colors[i])
            plt.tight_layout()
            # save figure
            plt.savefig(save_path_plots, dpi=100)

def save_collision_log(collision_log_path, base_path_save):
    save_df = base_path_save + '/collision_log_df.csv'
    collision_log = torch.load(collision_log_path,  map_location=torch.device('cpu'))
    print("collision_log is")
    print(collision_log['Safety Scores (Shield)'])
    collision_log.to_csv(save_df)
    using_shield_columns = 0
    count_no_action_to_choose = 0
    # Iterate through the DataFrame rows
    for index, row in collision_log.iterrows():
        safety_score = row['Safety Scores (Shield)']
        no_safe_action = row['No Safe Action']
        if safety_score is not 'No shield masking yet':
            using_shield_columns +=1
            if no_safe_action == True:
                count_no_action_to_choose +=1
    print("count_no_action_to_choose:", count_no_action_to_choose)
    print("using_shield_columns:", using_shield_columns)
    print("no_action_to_choose rate is", round(count_no_action_to_choose/using_shield_columns,4))


      #  # Check if the value is not 'No shield masking yet'
       # if safety_score != 'No shield masking yet':
        #    masking_columns += 1




# ------------------------------------------------------------------------------------------------------------------------------------


files = [torch.load("models/safety_tresholds/tresh=0.01/stats.log", map_location=torch.device('cpu')),torch.load("models/safety_tresholds/tresh=0.05/stats.log",map_location=torch.device('cpu')), torch.load("models/safety_tresholds/tresh=0.1/stats.log", map_location=torch.device('cpu')), torch.load("models/safety_tresholds/tresh=0.25/stats.log", map_location=torch.device('cpu'))]

keys = ["tresh=0.01","tresh=0.05", "tresh=0.1", "tresh=0.25"]
base_path_save = "models\safety_tresholds\plots&tables"

create_plots(files, keys, base_path_save)


#save_collision_log("models/PIAPUA_2609/collision_info_log.log", "models/PIAPUA_2609")



def show_additional_stats(file_path):
    file = torch.load(file_path)
    time_steps, rewards, costs, tasks, total_training_time, episodes_len, amount_of_done = file
    print(episodes_len)
    print("Average episodes length is", sum(episodes_len)/len(episodes_len))
    print("Total Training Time is", total_training_time)
    print("Amount of done out of all epochs", amount_of_done/len(rewards))


#show_additional_stats('models/PIAPUA_2609/stats.log')


