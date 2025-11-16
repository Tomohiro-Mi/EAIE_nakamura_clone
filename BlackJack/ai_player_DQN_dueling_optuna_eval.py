import argparse
import copy
import optuna
from optuna.trial import Trial
import numpy as np
import torch
from tqdm import tqdm
import os
import shutil

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


def evaluate_agent_money(agent: orig.Agent, n_games: int, verbose=False):
    """Run n_games in testmode using the provided agent and return the final player's money.
    Assumes orig.player will be (re)initialized before each evaluation run.
    """
    for n in range(1, n_games+1):
        remaining_cards = orig.RemainingCards(orig.N_DECKS)
        orig.game_start(n, remaining_cards=remaining_cards)
        state = orig.get_state(remaining_cards=remaining_cards)

        while True:
            action = orig.select_action(state, orig.Strategy.QMAX, agent)
            if orig.g_retry_counter >= orig.RETRY_MAX and action == orig.Action.RETRY:
                action = np.random.choice([orig.Action.HIT, orig.Action.STAND, orig.Action.DOUBLE_DOWN, orig.Action.SURRENDER])

            reward, done, status = orig.act(action)
            if action == orig.Action.RETRY:
                orig.g_retry_counter += 1

            state = orig.get_state(remaining_cards=remaining_cards)

            if done:
                break

    try:
        money = float(orig.player.get_money())
    except Exception:
        money = float(getattr(orig.player, 'money', 0))
    if verbose:
        print(f"Evaluation final money: {money}")
    return money


def objective(trial: Trial, args) -> float:
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
    args_train = Args()
    args_train.testmode = False
    args_train.history = ''
    args_train.log_mode = 'per_game'

    agent = orig.Agent(orig.STATE_SIZE, orig.ACTION_SIZE, learning_rate, discount_factor, epsilon, buffer_size=buffer_size)

    # run training for the configured number of games
    _ = train_episode(agent, args.train_games+1, args_train)

    # After training, evaluate the learned policy: run `eval_runs` repeats of `eval_games` in testmode
    eval_results = []
    for r in range(args.eval_runs):
        # reinitialize player for each evaluation run
        orig.player = orig.Player(initial_money=orig.INITIAL_MONEY, basic_bet=orig.BET)
        money = evaluate_agent_money(agent, args.eval_games)
        eval_results.append(money)

    avg_money = float(np.mean(eval_results)) if len(eval_results) > 0 else 0.0
    # Save model for this trial so the callback can copy the best one later.
    try:
        trial_model_path = f"trial_model_{trial.number}.pth"
        agent.save_model(trial_model_path)
    except Exception as e:
        print(f"Warning: failed to save trial model for trial {trial.number}: {e}")

    # We want to maximize average final money
    return avg_money


def main():
    parser = argparse.ArgumentParser(description='Dueling DQN Optuna evaluator (train then evaluate average money)')
    parser.add_argument('--trials', type=int, default=20, help='Optuna trials')
    parser.add_argument('--train-games', type=int, default=orig.TOTAL_GAMES, help='training games used inside each trial')
    parser.add_argument('--final-train-games', type=int, default=orig.TOTAL_GAMES, help='final training games for best model')
    parser.add_argument('--eval-games', type=int, default=1000, help='number of games per evaluation run')
    parser.add_argument('--eval-runs', type=int, default=10, help='number of evaluation repeats (each repeats eval-games)')
    parser.add_argument('--save', type=str, default='dqn_model_optuna_best.pth', help='save path for best model')
    parser.add_argument('--history', type=str, default=os.path.join('play_log_optuna_eval.csv'), help='CSV path to save per-game summaries for plotting')
    parser.add_argument('--log-mode', type=str, choices=('per_action','per_game'), default='per_game', help='logging mode to use when saving history CSV')
    args = parser.parse_args()

    # attach eval params onto args so objective can use them
    args.eval_games = args.eval_games
    args.eval_runs = args.eval_runs

    study = optuna.create_study(direction='maximize', study_name='dqn_dueling_eval_optuna')

    # Callback: if this trial becomes the new best, copy its saved model to the desired path.
    def _save_best_model_callback(study_obj, trial_obj):
        try:
            # study_obj.best_trial is updated by Optuna before calling callbacks
            if study_obj.best_trial.number == trial_obj.number:
                src = f"trial_model_{trial_obj.number}.pth"
                dst = args.save
                if os.path.exists(src):
                    shutil.copyfile(src, dst)
                    print(f"Saved new best model from trial {trial_obj.number} -> {dst}")
                else:
                    print(f"Best trial {trial_obj.number} model file not found: {src}")
        except Exception as e:
            print(f"Warning in save_best_model_callback: {e}")

    study.optimize(lambda t: objective(t, args), n_trials=args.trials, callbacks=[_save_best_model_callback])

    print('\nBest params:', study.best_params)
    print('Best value (avg final money):', study.best_value)
    if os.path.exists(args.save):
        print(f"Best model saved to: {args.save}")
    else:
        print("No best model file found. Ensure trials completed and models were saved per trial.")


if __name__ == '__main__':
    main()
