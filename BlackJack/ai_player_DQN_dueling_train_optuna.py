import argparse
import copy
import optuna
from optuna.trial import Trial
import numpy as np
import torch
from tqdm import tqdm
import os

# Import the original training module (must be in the same folder)
import ai_player_DQN_dueling_train_original as orig

# We'll reuse many functions/classes from orig: Agent, Player, Action, Strategy, RemainingCards, get_action_name, get_current_hands, act, get_state


def train_episode(agent: orig.Agent, n_games: int, args, early_stopping=None):
    """Run n_games using functions from the original module and perform learning.
    Returns final win rate computed over the most recent 1000 games (or fewer if not enough played).
    """
    global_orig = orig

    # prepare log files if requested
    action_logfile = None
    game_logfile = None
    if getattr(args, 'history', ''):
        if getattr(args, 'log_mode', 'per_game') == 'per_action':
            action_logfile = open(args.history, 'w')
            print('score,hand_length,p_hand_is_nbj,p_hand_is_busted,d_hand_1,remaining_2_6,remaining_7_9,remaining_10_K,remaining_A,money_bin,action,reward,result', file=action_logfile)
        else:
            game_logfile = open(args.history, 'w')
            print('game_id,result,reward', file=game_logfile)

    win_count = 0
    total_count = 0
    recent_wins = []  # use list and cap manually to avoid importing deque here

    for n in tqdm(range(1, n_games)):
        remaining_cards = orig.RemainingCards(orig.N_DECKS)
        orig.game_start(n, remaining_cards=remaining_cards)
        state = orig.get_state(remaining_cards=remaining_cards)
        game_reward = 0

        while True:
            if args.testmode:
                action = orig.select_action(state, orig.Strategy.QMAX, agent)
            else:
                action = orig.select_action(state, orig.Strategy.E_GREEDY, agent)
            if orig.g_retry_counter >= orig.RETRY_MAX and action == orig.Action.RETRY:
                action = np.random.choice([orig.Action.HIT, orig.Action.STAND, orig.Action.DOUBLE_DOWN, orig.Action.SURRENDER])

            reward, done, status = orig.act(action)
            if action == orig.Action.RETRY:
                orig.g_retry_counter += 1

            prev_state = state
            state = orig.get_state(remaining_cards=remaining_cards)

            if not args.testmode:
                # Use the original's simple intermediate reward (as in ai_player_DQN_dueling_train_original)
                score_diff = (state[0] - prev_state[0]) * 21.0
                bonus = 0.0
                if score_diff > 0:
                    bonus = 0.05 * score_diff
                total_reward = reward + bonus

                a_idx = orig.action_to_index(action)
                agent.add_experience(prev_state, a_idx, total_reward, state, done)
                agent.update()
                agent.decay_epsilon_step()

            if action_logfile:
                try:
                    score_out = prev_state[0]
                    hand_len_out = prev_state[1]
                    p_nbj = int(prev_state[2])
                    p_bst = int(prev_state[3])
                    d_hand_1_out = int(round(prev_state[4]*12))
                    rem2 = int(round(prev_state[5]* (5*4*orig.N_DECKS)))
                    rem7 = int(round(prev_state[6]* (3*4*orig.N_DECKS)))
                    rem10 = int(round(prev_state[7]* (4*4*orig.N_DECKS)))
                    remA = int(round(prev_state[8]* (1*4*orig.N_DECKS)))
                    money_out = int(round(prev_state[9]*4))
                except Exception:
                    score_out = hand_len_out = p_nbj = p_bst = d_hand_1_out = 0
                    rem2 = rem7 = rem10 = remA = money_out = 0
                print('{},{},{},{},{},{},{},{},{},{},{},{}'.format(
                    score_out, hand_len_out, p_nbj, p_bst, d_hand_1_out,
                    rem2, rem7, rem10, remA, money_out,
                    orig.get_action_name(action), reward
                ), file=action_logfile)

            if done and game_logfile:
                status_l = str(status).lower() if status is not None else ''
                if 'unsettled' not in status_l and 'retry' not in status_l:
                    try:
                        r = float(reward)
                    except Exception:
                        r = 0.0
                    if r > 0.0:
                        res = 'win'
                        recent_wins.append(1)
                    elif r < 0.0:
                        res = 'lose'
                        recent_wins.append(0)
                    else:
                        res = None
                    if res is not None:
                        print(f"{n},{res},{r}", file=game_logfile)

            if done:
                game_reward = reward
                break

        if game_reward > 0:
            win_count += 1
        total_count += 1

        # cap recent_wins to 1000
        if len(recent_wins) > 1000:
            recent_wins = recent_wins[-1000:]

    if game_logfile:
        game_logfile.close()
    if action_logfile:
        action_logfile.close()

    if len(recent_wins) > 0:
        final_win_rate = sum(recent_wins) / len(recent_wins)
    else:
        final_win_rate = win_count / total_count if total_count > 0 else 0.0
    return final_win_rate


def objective(trial: Trial) -> float:
    # hyperparameter search space
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    discount_factor = trial.suggest_float('discount_factor', 0.95, 0.99)
    epsilon = trial.suggest_float('epsilon', 0.05, 0.5)
    buffer_size = trial.suggest_int('buffer_size', 2000, 20000, step=2000)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    target_update = trial.suggest_int('target_update_freq', 100, 500, step=100)

    # prepare player and agent
    orig.player = orig.Player(initial_money=orig.INITIAL_MONEY, basic_bet=orig.BET)
    class Args: pass
    args = Args()
    args.testmode = False
    args.history = ''
    args.log_mode = 'per_game'

    agent = orig.Agent(orig.STATE_SIZE, orig.ACTION_SIZE, learning_rate, discount_factor, epsilon, buffer_size=buffer_size)

    # run a smaller training to evaluate
    win_rate = train_episode(agent, 10001, args)
    return win_rate


def main():
    parser = argparse.ArgumentParser(description='Dueling DQN Optuna runner (based on original)')
    parser.add_argument('--trials', type=int, default=20, help='Optuna trials')
    parser.add_argument('--games', type=int, default=orig.TOTAL_GAMES, help='final training games')
    parser.add_argument('--save', type=str, default='dqn_model_optuna.pth', help='save path for best model')
    parser.add_argument('--history', type=str, default=os.path.join('play_log_optuna.csv'), help='CSV path to save per-game summaries for plotting')
    parser.add_argument('--log-mode', type=str, choices=('per_action','per_game'), default='per_game', help='logging mode to use when saving history CSV')
    args = parser.parse_args()

    study = optuna.create_study(direction='maximize', study_name='dqn_dueling_original_optuna')
    study.optimize(objective, n_trials=args.trials)

    print('\nBest params:', study.best_params)
    print('Best value:', study.best_value)

    # Train final model with best params
    best = study.best_params
    orig.player = orig.Player(initial_money=orig.INITIAL_MONEY, basic_bet=orig.BET)
    agent = orig.Agent(orig.STATE_SIZE, orig.ACTION_SIZE, best.get('learning_rate', orig.LEARNING_RATE), best.get('discount_factor', orig.DISCOUNT_FACTOR), best.get('epsilon', orig.EPS), buffer_size=best.get('buffer_size', orig.BUFFER_SIZE))

    # Prepare args for final training and CSV logging
    class Args: pass
    args2 = Args()
    args2.testmode = False
    args2.history = args.history
    args2.log_mode = args.log_mode

    win_rate = train_episode(agent, args.games+1, args2)
    print(f"Final win rate (last 1000 games): {win_rate:.4f}")

    agent.save_model(args.save)
    print(f"Saved best model to {args.save}")
    if args.history:
        print(f"Per-game CSV saved to: {args.history}")


if __name__ == '__main__':
    main()
