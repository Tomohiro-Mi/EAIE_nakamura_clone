import argparse
import os
import shutil
import torch
import numpy as np
from tqdm import tqdm

# import the original training module (contains Agent, Player, Action, Strategy, RemainingCards, get_state, select_action, act)
import ai_player_DQN_dueling_train_original as orig


def train_episode(agent: orig.Agent, n_games: int, args):
    """Train agent by playing n_games (same logic as original training loop).
    Returns nothing; agent is updated in-place.
    """
    for n in tqdm(range(1, n_games)):
        remaining_cards = orig.RemainingCards(orig.N_DECKS)
        orig.game_start(n, remaining_cards=remaining_cards)
        state = orig.get_state(remaining_cards=remaining_cards)

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
                # intermediate bonus as in original
                score_diff = (state[0] - prev_state[0]) * 21.0
                bonus = 0.0
                if score_diff > 0:
                    bonus = 0.05 * score_diff
                total_reward = reward + bonus

                a_idx = orig.action_to_index(action)
                agent.add_experience(prev_state, a_idx, total_reward, state, done)
                agent.update()
                agent.decay_epsilon_step()

            if done:
                break


def evaluate_agent_money(agent: orig.Agent, n_games: int, verbose=False):
    """Run n_games in testmode (greedy) and return final player's money."""
    for n in range(1, n_games + 1):
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


def main():
    parser = argparse.ArgumentParser(description='Train fixed-hyper DQN multiple times and save best model')
    parser.add_argument('--runs', type=int, default=10, help='how many independent training runs to perform')
    parser.add_argument('--train-games', type=int, default=orig.TOTAL_GAMES, help='number of training games per run')
    parser.add_argument('--eval-games', type=int, default=1000, help='number of games for each evaluation run')
    parser.add_argument('--eval-runs', type=int, default=15, help='how many repeated evaluation runs to average')
    parser.add_argument('--save', type=str, default='best_model_fixed.pth', help='path to save best model')
    parser.add_argument('--per-run-save-dir', type=str, default='', help='optional directory to save per-run models (kept even if not best)')
    parser.add_argument('--base-model', type=str, default='', help='optional path to pretrained weights to load before training')
    parser.add_argument('--testmode', action='store_true', help='run in testmode (no learning)')
    args = parser.parse_args()

    best_value = -float('inf')
    best_run = None

    # ensure per-run dir exists if provided
    if args.per_run_save_dir:
        os.makedirs(args.per_run_save_dir, exist_ok=True)

    for run in range(1, args.runs + 1):
        print(f"Starting run {run}/{args.runs}")

        # reset player
        orig.player = orig.Player(initial_money=orig.INITIAL_MONEY, basic_bet=orig.BET)

        # construct agent with the fixed hyperparameters from orig (or user-provided constants)
        agent = orig.Agent(orig.STATE_SIZE, orig.ACTION_SIZE, orig.LEARNING_RATE, orig.DISCOUNT_FACTOR, orig.EPS, buffer_size=orig.BUFFER_SIZE)

        # load base/pretrained weights if provided
        if args.base_model:
            try:
                if os.path.exists(args.base_model):
                    state_dict = torch.load(args.base_model, map_location=agent.device)
                    try:
                        agent.original_q_network.load_state_dict(state_dict)
                        agent.sync_net()
                        print(f"Loaded base model weights from {args.base_model}")
                    except Exception:
                        # maybe saved full checkpoint dict, try to extract key
                        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                            agent.original_q_network.load_state_dict(state_dict['state_dict'])
                            agent.sync_net()
                            print(f"Loaded base model (checkpoint['state_dict']) from {args.base_model}")
                        else:
                            print(f"Warning: could not load weights from {args.base_model} into network")
                else:
                    print(f"Warning: base model file not found: {args.base_model}")
            except Exception as e:
                print(f"Warning while loading base model {args.base_model}: {e}")

        # optional load (none by default)
        # train
        train_episode(agent, args.train_games + 1, args)

        # save per-run model
        per_run_path = ''
        try:
            per_run_path = os.path.join(args.per_run_save_dir, f"run_{run}_model.pth") if args.per_run_save_dir else f"run_model_{run}.pth"
            agent.save_model(per_run_path)
        except Exception as e:
            print(f"Warning: failed to save per-run model for run {run}: {e}")

        # evaluate agent: do several eval_runs and average final money
        eval_results = []
        for r in range(args.eval_runs):
            # reinitialize player for each evaluation run
            orig.player = orig.Player(initial_money=orig.INITIAL_MONEY, basic_bet=orig.BET)
            money = evaluate_agent_money(agent, args.eval_games)
            eval_results.append(money)

        avg_money = float(np.mean(eval_results)) if len(eval_results) > 0 else 0.0
        print(f"Run {run} average final money over {args.eval_runs} eval runs: {avg_money}")

        # if best, copy model to args.save
        try:
            if avg_money > best_value:
                best_value = avg_money
                best_run = run
                # copy per-run model file if exists, else save directly from agent
                if per_run_path and os.path.exists(per_run_path):
                    shutil.copyfile(per_run_path, args.save)
                else:
                    agent.save_model(args.save)
                print(f"New best model from run {run} saved to {args.save} (avg_money={avg_money})")
        except Exception as e:
            print(f"Warning when saving best model for run {run}: {e}")

    print("All runs finished.")
    if best_run is not None:
        print(f"Best run: {best_run} with avg_money={best_value}")
    else:
        print("No best run recorded.")


if __name__ == '__main__':
    main()
