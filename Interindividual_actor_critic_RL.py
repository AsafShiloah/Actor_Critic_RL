import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.special import softmax

def interindividual_actor_critic(HP, group_assignment, reward_mat, thetas_init='uniform', return_values=False, V_init='default'):
    n = HP['n_agents']  # מספר הסוכנים
    n_rounds = int(HP['nt'] / n)  # מספר סבבים

    if HP['nt'] % n != 0:
        raise ValueError('number of trials must be divisible by number of agents')

    # הגדרת מערכים לשמירת הנתונים
    policy_all = np.zeros((HP['trials'], n_rounds, n, HP['n_actions']))
    V_all = np.zeros((HP['trials'], HP['nt'], n, HP['n_states']))

    if return_values:
        advantage_all = np.zeros((HP['trials'], HP['nt'], n, HP['n_actions'], HP['n_states']))

    action_arr = np.arange(HP['n_actions'])

    for t in range(HP['trials']):
        if return_values:
            adv = np.zeros((HP['nt'], n, HP['n_actions'], HP['n_states']))  # מערך לשמירת ערכי יתרון
            adv[0, :, :, :] = np.nan  # ערך יתרון התחלתי מוגדר כ-nan כדי למנוע השפעה של ערכים התחלתיים של 0
            V = np.zeros((HP['nt'], n, HP['n_states']))  # הערכת ערכי מצב
        else:
            V = np.zeros((n, HP['n_states']))

        c = np.zeros((n, HP['n_actions']))  # משתנה לשליטה
        thetas = np.zeros((n, HP['n_actions']))  # משתנה למדיניות ראשונית ללא הטיה
        policy = np.ones((n_rounds, n, HP['n_actions'])) / HP['n_actions']  # אתחול מדיניות
        state_mat = np.eye(n).astype(int)  # מטריצת מצב

        if V_init != 'default':
            V[0, :, :] = np.copy(V_init)

        if thetas_init != 'uniform':
            thetas = np.copy(thetas_init)
            policy[0, :, :] = np.exp(HP['dr'] * thetas) / np.sum(np.exp(HP['dr'] * thetas), axis=1, keepdims=True)

        i = 1  # התחלה מהצעד השני לצורך חישוב יתרון
        adj_mat = create_new_adj_matrix(group_assignment, HP) # יצירת מטריצת שכנויות חדשה בין ניסויים עם אותה חלוקה לקבוצות
        while i < HP['nt'] - n:
            agents_idx = np.arange(n)
            round = int(i / n) + 1

            # if round % 100 == 0:
            #     print('finished_round ' + str(round))
            #     print(adj_mat, '\n')

            for j in np.arange(n):
                # בחירת פעולה
                a = np.random.choice(action_arr, size=1, p=policy[round - 1, j, :])
                a = a.astype(int)



                # קבלת תגמול
                reward = np.squeeze(reward_mat[j, :, a])

                if return_values:
                    pe = reward - V[i - 1, agents_idx, state_mat[j]]
                    pe = np.random.binomial(1,adj_mat[j]) * pe #  לפי ברנולי של כניסה של המטריצה להכפיל במשתנה בינארי
                    pe = pe * (1 - np.sign(pe) * HP['la'])
                    pe[j] = reward[j] + HP['ratio'] * np.sum(pe) - V[i - 1, j, 1]
                    pe[j] = pe[j] * (1 - np.sign(pe[j]) * HP['la'])
                    V[i] = np.copy(V[i - 1])
                    V[i, np.arange(n), state_mat[j]] = V[i - 1, np.arange(n), state_mat[j]] + HP['lrs'] * pe
                    adv[i] = np.copy(adv[i - 1])
                    adv[i, agents_idx, a, state_mat[j]] = pe

                else:
                    pe = reward - V[agents_idx, state_mat[j]]

                    pe = np.random.binomial(1,adj_mat[j]) * pe
                    pe = pe * (1 - np.sign(pe) * HP['la'])
                    pe[j] = reward[j] + HP['ratio'] * np.sum(pe) - V[j, 1]
                    pe[j] = pe[j] * (1 - np.sign(pe[j]) * HP['la'])
                    V[np.arange(n), state_mat[j]] = V[np.arange(n), state_mat[j]] + HP['lrs'] * pe


                # עדכון מדיניות
                c[j, a] = HP['dr'] * policy[round - 1, j, a] * (1 - policy[round - 1, j, a])
                thetas[j, a] = thetas[j, a] + HP['lrp'] * c[j, a] * pe[j]

                # עדכון softmax של המדיניות - סוכן פועל
                policy[round, j, :] = softmax(thetas[j])
                connected_agents = np.nonzero(adj_mat[j])[0]

                for k in connected_agents:
                    # עדכון מטריצת קשרים בהתבסס על תגמול
                    adj_mat = update_adj_matrix(adj_mat, pe, HP[lrm])
                    # להכפיל בכיוון השגיאה או בשגיאה?
                adj_mat = binarize_adj_matrix(adj_mat, HP) # בינאריזציה של מטריצת הקשרים מעל או מתחת לסף

                i += 1
        policy_all[t, :, :, :] = np.copy(policy)
        if return_values:
            V_all[t, :, :, :] = np.copy(V)
            advantage_all[t, :, :, :, :] = np.copy(adv)

    if return_values:
        return policy_all, V_all, advantage_all
    else:
        return policy_all, V




def create_reward_mat(HP, group_assignment):
    n_agents = HP['n_agents']
    n_actions = HP['n_actions']
    reward_mat = np.zeros((n_agents, n_agents, n_actions))

    for i in range(n_agents):
        for j in range(n_agents):
            if group_assignment[i] == group_assignment[j]:
                # תגמול חיובי עבור שניהם כשהם באותה קבוצה ובוחרים את הפעולה של הקבוצה שלהם
                reward_mat[i, j, 0] = HP['rewards']['group1'][0]  # תגמול חיובי עבור פעולה 0 (קבוצה 1)
                reward_mat[i, j, 1] = HP['rewards']['group2'][0]  # תגמול חיובי עבור פעולה 1 (קבוצה 2)
            else:
                # תגמול חיובי לסוכן הפועל ותגמול שלילי לסוכן השני אם הם בקבוצות שונות
                reward_mat[i, j, 0] = HP['rewards']['group1'][0] if group_assignment[i] == 0 else HP['rewards']['group1'][1]
                reward_mat[i, j, 1] = HP['rewards']['group2'][0] if group_assignment[i] == 1 else HP['rewards']['group2'][1]

    np.fill_diagonal(reward_mat[:, :, 0], 0)
    np.fill_diagonal(reward_mat[:, :, 1], 0)

    return reward_mat





def create_grouped_adj_matrix(HP):
    total_nodes = HP['n_agents']
    adj_matrix = np.zeros((total_nodes, total_nodes))

    nodes = np.arange(total_nodes)
    np.random.shuffle(nodes)
    groups = np.array_split(nodes, HP['d'])

    group_assignment = np.zeros(total_nodes, dtype=int)

    for group_idx, group in enumerate(groups):
        for i in group:
            group_assignment[i] = group_idx
            for j in group:
                if i != j:
                    adj_matrix[i, j] = round(np.random.uniform(HP['adj_mat_group_conn']['in_group'], 1),4)


    for i in range(total_nodes):
        for j in range(total_nodes):
            if i != j and adj_matrix[i, j] == 0:
                adj_matrix[i, j] = round(np.random.uniform(0, HP['adj_mat_group_conn']['out_group']),4)

    np.fill_diagonal(adj_matrix, 0)

    return adj_matrix, group_assignment

def create_new_adj_matrix(group_assignment, HP):
    total_nodes = HP['n_agents']
    adj_matrix = np.zeros((total_nodes, total_nodes))

    groups = [np.where(group_assignment == i)[0] for i in range(HP['d'])]

    for group in groups:
        for i in group:
            for j in group:
                if i != j:
                    adj_matrix[i, j] = round(np.random.uniform(HP['adj_mat_group_conn']['in_group'], 1),4)

    for i in range(total_nodes):
        for j in range(total_nodes):
            if i != j and adj_matrix[i, j] == 0:
                adj_matrix[i, j] = round(np.random.uniform(0, HP['adj_mat_group_conn']['out_group']),4)

    np.fill_diagonal(adj_matrix, 0)

    return adj_matrix


def binarize_adj_matrix(adj_matrix,HP):
    # Create a copy of the adjacency matrix to avoid modifying the original
    binarized_matrix = np.copy(adj_matrix)

    # Apply thresholds
    binarized_matrix[binarized_matrix >= HP['adj_mat_threshold']['high_t']] = 1
    binarized_matrix[binarized_matrix <= HP['adj_mat_threshold']['low_t']] = 0

    return binarized_matrix


def update_adj_matrix(adj_mat, pe, learning_rate=0.001):
    for j in range(adj_mat.shape[0]):
        for k in range(adj_mat.shape[1]):
            if j != k:  # לא לעדכן את האלכסון
                adj_mat[j, k] = np.clip(adj_mat[j, k] + learning_rate * pe[j], 0, 1)
    return adj_mat


# HP = {'trials': 1, 'nt':10000, 'n_actions':2, 'lrs':0.2,'lrp':0.2,'lrm':0.1,
#       'ratio':0.5, 'dr':1, 'n_states':2, 'la':0, 'n_agents':10, 'd':2, # laמשנה את התוצאות לבדוק
#       'actions':['group1','group2'],
#       'rewards':{'group1':[1,-1],'group2':[1,-1]},
#       'reward_noise_var':{'selfish':[0,0],'coop':[0,0]}, # [self, other] variance of the noise
#       'adj_mat_group_conn':{'in_group':0.8,'out_group':0.2},
#       'adj_mat_threshold':{'high_t':0.90,'low_t':0.1}
#       }
#
#
#
# # יצירת הקצאת הקבוצות
# adj_matrix, group_assignment = create_grouped_adj_matrix(HP)
# # adj_matrix = binarize_adj_matrix(adj_matrix,HP)
#
# print("Adj matrix:")
# print(adj_matrix)
#
# # יצירת מטריצת התגמולים
# reward_mat = create_reward_mat(HP, group_assignment)
#
# print("Group Assignment:", group_assignment)
# print("Reward Matrix for Action 0 (group1):")
# print(reward_mat[:, :, 0])
# print("Reward Matrix for Action 1 (group2):")
# print(reward_mat[:, :, 1])
# # הרצת הפונקציה interindividual_actor_critic עם הפרמטרים הנתונים
# policy_all, V_all,A_all = interindividual_actor_critic(HP, group_assignment, reward_mat, thetas_init='uniform', return_values=True)
# # print(V_all[-1])
#
#
# # ניתוח תוצאות
# group1_action_preference = np.mean(policy_all[:, :, group_assignment == 0, 0], axis=(0, 1))
# group2_action_preference = np.mean(policy_all[:, :, group_assignment == 1, 1], axis=(0, 1))
#
# print("Group 1 Action Preference for 'group1' Action: ", group1_action_preference)
# print("Group 2 Action Preference for 'group2' Action: ", group2_action_preference)



def plot_policy_and_value_all_agents(HP, policy_all, V_all, group_assignment):

    n_rounds = int(HP['nt'] / HP['n_agents'])
    n_agents = HP['n_agents']
    n_actions = HP['n_actions']

    # Set up the plotting grid
    fig, axs = plt.subplots(n_agents, 2, figsize=(14, n_agents * 3))

    for agent_index in range(n_agents):
        group = group_assignment[agent_index]

        # Plot the policy for the specified agent
        for action in range(n_actions):
            axs[agent_index, 0].plot(policy_all[0, :, agent_index, action], label=f'Action {action}')
        axs[agent_index, 0].set_xlabel('Round')
        axs[agent_index, 0].set_ylabel('Policy Probability')
        axs[agent_index, 0].set_title(f'Policy for Agent {agent_index+1} (Group {group+1})')
        axs[agent_index, 0].legend()
        axs[agent_index, 0].grid(True)


        if V_all.ndim == 4:
            for state in range(V_all.shape[3]):
                axs[agent_index, 1].plot(V_all[0, :, agent_index, state], label=f'State {state}')
        elif V_all.ndim == 2:
            for state in range(V_all.shape[1]):
                axs[agent_index, 1].plot(np.arange(V_all.shape[0]), V_all[:, state], label=f'State {state}')
        axs[agent_index, 1].set_xlabel('Round')
        axs[agent_index, 1].set_ylabel('Value')
        axs[agent_index, 1].set_title(f'Value Function for Agent {agent_index+1} (Group {group+1})')
        axs[agent_index, 1].legend()
        axs[agent_index, 1].grid(True)

    plt.tight_layout()
    plt.show()





