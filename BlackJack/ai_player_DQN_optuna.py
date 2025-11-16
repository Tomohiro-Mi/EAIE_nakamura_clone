import copy
import socket
import argparse
import numpy as np
from classes import Action, Strategy, RemainingCards, Player, get_card_info, get_action_name
from config import PORT, BET, INITIAL_MONEY, N_DECKS
import torch
from typing import NamedTuple, Union, Tuple
import random
from collections import deque
from tqdm import tqdm

try:
    import optuna
except Exception as e:
    optuna = None

# 1ゲームあたりのRETRY回数の上限(変更不可)
RETRY_MAX = 10

### グローバル変数 ###

# ゲームごとのRETRY回数のカウンター
g_retry_counter = 0

# プレイヤークラスのインスタンスを作成
player = Player(initial_money=INITIAL_MONEY, basic_bet=BET)

# ディーラーとの通信用ソケット
soc = None

# 残りカード追跡（各アクションで更新するため）
g_remaining_cards: RemainingCards | None = None

# デフォルトハイパーパラメータ（Optuna で上書き）
TOTAL_GAMES = 1000 # 総ゲーム数
BUFFER_SIZE = 1000 # 経験再生バッファのサイズ
BATCH_SIZE = 32 # ミニバッチサイズ
STATE_SIZE = 10 # 状態の次元数（score,length,nbj,busted,d_up,rem4(4),money_bin）
ACTION_SIZE = 5 # 行動の種類数
TARGET_UPDATE_FREQ = 500 # ターゲットネットワークの更新頻度（ステップ単位）
EPS = 0.3 # ε-greedyにおけるε
LEARNING_RATE = 0.001 # 学習率（DQN用に小さめ）
DISCOUNT_FACTOR = 0.99 # 割引率

# 所持金ビン化
def money_to_bin(money: int) -> int:
    if money < 0:
        return 0
    if money < BET:
        return 1
    if money < INITIAL_MONEY * 0.8:
        return 2
    if money < INITIAL_MONEY:
        return 3
    return 4

def remaining_counts_4cats(counts: list[int]) -> list[int]:
    two_to_six = sum(counts[1:6])     # 2..6
    seven_to_nine = sum(counts[6:9])  # 7..9
    ten_to_king = sum(counts[9:13])   # 10,J,Q,K
    aces = counts[0]                  # A
    return [two_to_six, seven_to_nine, ten_to_king, aces]

def score_to_bin(score: int) -> int:
    if score <= 8:
        return 0
    if score <= 11:
        return 1
    if score <= 16:
        return 2
    return 3

def index_to_action(i: int) -> Action:
    mapping = [Action.HIT, Action.STAND, Action.DOUBLE_DOWN, Action.SURRENDER, Action.RETRY]
    return mapping[i]

def action_to_index(a: Action) -> int:
    mapping = {
        Action.HIT: 0,
        Action.STAND: 1,
        Action.DOUBLE_DOWN: 2,
        Action.SURRENDER: 3,
        Action.RETRY: 4,
    }
    return mapping[a]

class TorchTensor(NamedTuple):
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    done: torch.Tensor


class ExperienceReplayBuffer:
    def __init__(self, buffer_size: int=10000, batch_size: int = 64):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def add(
        self,
        state: Union[np.ndarray, Tuple],
        action: int,
        reward: Union[int, float],
        next_state: Union[np.ndarray, Tuple],
        done: bool,
        ) -> None:
        self.buffer.append((state, action, reward, next_state, done))
        
    def get(self) -> TorchTensor:
        data = random.sample(self.buffer, self.batch_size)
        
        batch_data = (
            np.stack([x[0] for x in data]).astype(np.float32), # state
            np.array([x[1] for x in data]).astype(np.int32), # action
            np.array([x[2] for x in data]).astype(np.float32), # reward
            np.stack([x[3] for x in data]).astype(np.float32), # next_state
            np.array([x[4] for x in data]).astype(np.int32), # done
        )
        
        return TorchTensor(*tuple(map(self.to_torch, batch_data)))
        
    def to_torch(self, array: np.ndarray) -> torch.Tensor:
        return torch.tensor(array, dtype=torch.float32, device=self.device)   


class QNetwork(torch.nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(QNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)
    

class Agent:
    def __init__(self, state_size: int, action_size: int, learning_rate: float, discount_factor: float, epsilon: float, epsilon_min: float = 0.01, buffer_size: int = 10000, batch_size: int = 32, target_update_freq: int = 500):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        # εはステップ単位で減衰させる（ゲームごとではなく行動ごと）
        self.epsilon_decay = (epsilon - epsilon_min) / (TOTAL_GAMES * 20)

        self.replay = ExperienceReplayBuffer(buffer_size=buffer_size, batch_size=batch_size)
        self.data = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Qネットワーク（オンライン／ターゲット）
        self.original_q_network = QNetwork(state_size, action_size).to(self.device)
        self.target_q_network = QNetwork(state_size, action_size).to(self.device)
        self.sync_net()  # 初期同期

        # 追加属性（他メソッドで参照されるもの）
        self.gamma = discount_factor
        self.batch_size = batch_size
        self.learn_steps = 0
        self.total_steps = 0
        self.target_update_freq = target_update_freq
        # オンラインネットワークを最適化対象に設定
        self.optimizer = torch.optim.Adam(self.original_q_network.parameters(), lr=learning_rate)

    def get_best_action(self, state: np.ndarray) -> int:
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_c = self.original_q_network(state)
        return q_c.detach().argmax().item()
    
    def update(self) -> None:
        """経験再生バッファからサンプルして1ステップ学習を行う"""
        if len(self.replay.buffer) < self.batch_size:
            return

        self.data = self.replay.get()

        q_values = self.original_q_network(self.data.state)
        actions = self.data.action.long()  # gather用に整数化
        q_sa = q_values[torch.arange(self.batch_size), actions]

        with torch.no_grad():
            next_q_values = self.target_q_network(self.data.next_state)
            next_q_max = next_q_values.max(1)[0]
            target = self.data.reward + (1 - self.data.done) * self.gamma * next_q_max

        loss_function = torch.nn.MSELoss()
        loss = loss_function(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ターゲットネットの定期同期
        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            self.sync_net()

    def decay_epsilon_step(self) -> None:
        # 行動ごとにεを減衰
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

    def add_experience(
        self,
        state: Union[np.ndarray, Tuple],
        action: int,
        reward: Union[int, float],
        next_state: Union[np.ndarray, Tuple],
        done: bool,
        ) -> None:
        self.replay.add(state, action, reward, next_state, done)

    def sync_net(self) -> None:
        """ターゲットネットワークをオンラインネットワークで同期"""
        self.target_q_network.load_state_dict(self.original_q_network.state_dict())

    def set_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

    def save_model(self, path: str = 'dqn_model_optuna.pth') -> None:
        torch.save(self.original_q_network.state_dict(), path)


### 環境インタフェース ###

def game_start(game_ID=0, remaining_cards: RemainingCards | None = None):
    global g_retry_counter, player, soc, g_remaining_cards

    g_retry_counter = 0
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.connect((socket.gethostname(), PORT))
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    bet, money = player.set_bet()

    cardset_shuffled = player.receive_card_shuffle_status(soc)
    if cardset_shuffled and remaining_cards is not None:
        remaining_cards.shuffle()

    dc, pc1, pc2 = player.receive_init_cards(soc)
    g_remaining_cards = remaining_cards
    if g_remaining_cards is not None:
        g_remaining_cards.draw(dc)
        g_remaining_cards.draw(pc1)
        g_remaining_cards.draw(pc2)

def get_current_hands():
    return copy.deepcopy(player.player_hand), copy.deepcopy(player.dealer_hand)

def hit():
    global player, soc, g_remaining_cards
    player.send_message(soc, 'hit')
    pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True)
    if g_remaining_cards is not None:
        g_remaining_cards.draw(pc)
    if status == 'bust':
        if g_remaining_cards is not None:
            for c in dc:
                g_remaining_cards.draw(c)
        soc.close()
        reward = player.update_money(rate=rate)
        return reward, True, status
    else:
        return 0, False, status

def stand():
    global player, soc, g_remaining_cards
    player.send_message(soc, 'stand')
    score, status, rate, dc = player.receive_message(dsoc=soc, get_dealer_cards=True)
    if g_remaining_cards is not None:
        for c in dc:
            g_remaining_cards.draw(c)
    soc.close()
    reward = player.update_money(rate=rate)
    return reward, True, status

def double_down():
    global player, soc, g_remaining_cards
    bet, money = player.double_bet()
    player.send_message(soc, 'double_down')
    pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True)
    if g_remaining_cards is not None:
        g_remaining_cards.draw(pc)
    if g_remaining_cards is not None:
        for c in dc:
            g_remaining_cards.draw(c)
    soc.close()
    reward = player.update_money(rate=rate)
    return reward, True, status

def surrender():
    global player, soc, g_remaining_cards
    player.send_message(soc, 'surrender')
    score, status, rate, dc = player.receive_message(dsoc=soc, get_dealer_cards=True)
    if g_remaining_cards is not None:
        for c in dc:
            g_remaining_cards.draw(c)
    soc.close()
    reward = player.update_money(rate=rate)
    return reward, True, status

def retry():
    global player, soc, g_remaining_cards
    penalty = player.current_bet // 4
    player.consume_money(penalty)
    player.send_message(soc, 'retry')
    pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True, retry_mode=True)
    if g_remaining_cards is not None:
        g_remaining_cards.draw(pc)
    if status == 'bust':
        soc.close()
        reward = player.update_money(rate=rate)
        return reward-penalty, True, status
    else:
        return -penalty, False, status

def act(action: Action):
    if action == Action.HIT:
        return hit()
    elif action == Action.STAND:
        return stand()
    elif action == Action.DOUBLE_DOWN:
        return double_down()
    elif action == Action.SURRENDER:
        return surrender()
    elif action == Action.RETRY:
        return retry()
    else:
        exit()

def get_state(remaining_cards: RemainingCards | None = None):
    p_hand, d_hand = get_current_hands()
    raw_score = p_hand.get_score()
    length = p_hand.length()
    p_hand_is_nbj = p_hand.is_nbj()
    p_hand_is_busted = p_hand.is_busted()
    d_hand_1 = d_hand[0]
    money_bin = money_to_bin(player.get_money())
    ref_cards = remaining_cards if remaining_cards is not None else g_remaining_cards
    full_counts = ref_cards.get_remaining_card_counts() if ref_cards is not None else [0]*13
    rem4 = remaining_counts_4cats(full_counts)
    score_norm = float(raw_score) / 21.0
    length_norm = float(length) / 11.0
    dealer_norm = float(d_hand_1) / 12.0
    max_two_to_six = 5 * 4 * N_DECKS
    max_seven_to_nine = 3 * 4 * N_DECKS
    max_ten_to_king = 4 * 4 * N_DECKS
    max_aces = 1 * 4 * N_DECKS
    rem4_norm = [
        rem4[0] / max_two_to_six if max_two_to_six>0 else 0.0,
        rem4[1] / max_seven_to_nine if max_seven_to_nine>0 else 0.0,
        rem4[2] / max_ten_to_king if max_ten_to_king>0 else 0.0,
        rem4[3] / max_aces if max_aces>0 else 0.0,
    ]
    money_norm = float(money_bin) / 4.0
    flat_state = np.array([
        score_norm, length_norm,
        float(int(p_hand_is_nbj)), float(int(p_hand_is_busted)),
        dealer_norm,
        *rem4_norm,
        money_norm
    ], dtype=np.float32)

    return flat_state

def select_action(state, strategy: Strategy, agent: Agent=None):
    if strategy == Strategy.QMAX:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        q_vals = agent.original_q_network(state_tensor).detach().cpu().numpy()[0]
        mask = get_valid_action_mask_from_state(state)
        q_vals_masked = np.copy(q_vals)
        q_vals_masked[~mask] = -1e9
        return index_to_action(int(np.argmax(q_vals_masked)))
    elif strategy == Strategy.E_GREEDY:
        if np.random.rand() < agent.epsilon:
            mask = get_valid_action_mask_from_state(state)
            valid_indices = np.where(mask)[0]
            chosen = np.random.choice(valid_indices)
            return index_to_action(int(chosen))
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            q_vals = agent.original_q_network(state_tensor).detach().cpu().numpy()[0]
            mask = get_valid_action_mask_from_state(state)
            q_vals[~mask] = -1e9
            return index_to_action(int(np.argmax(q_vals)))
    else:
        mask = get_valid_action_mask_from_state(state)
        valid_indices = np.where(mask)[0]
        chosen = np.random.choice(valid_indices)
        return index_to_action(int(chosen))

def get_valid_action_mask_from_state(state: np.ndarray) -> np.ndarray:
    score_norm = state[0]
    length_norm = state[1]
    length = int(round(length_norm * 11))
    mask = np.ones(ACTION_SIZE, dtype=bool)
    if length != 2:
        mask[action_to_index(Action.DOUBLE_DOWN)] = False
    if g_retry_counter >= RETRY_MAX:
        mask[action_to_index(Action.RETRY)] = False
    return mask


### Optuna 用のラッパー / objective ###

def run_games_with_agent(agent: Agent, n_games: int, window: int = 500, testmode: bool=False, verbose: bool=False):
    """
    Run n_games and return win rate over the last `window` finished games and the number of finished games.
    A finished game is counted when its result is settled (not 'unsettled' or 'retry').
    """
    global g_retry_counter
    outcomes: list[int] = []  # 1 for win, 0 for not-win
    for n in tqdm(range(1, n_games+1)):
        remaining_cards = RemainingCards(N_DECKS)
        game_start(n, remaining_cards=remaining_cards)
        state = get_state(remaining_cards=remaining_cards)
        while True:
            if testmode:
                action = select_action(state, Strategy.QMAX, agent)
            else:
                action = select_action(state, Strategy.E_GREEDY, agent)
            if g_retry_counter >= RETRY_MAX and action == Action.RETRY:
                action = np.random.choice([
                    Action.HIT, Action.STAND, Action.DOUBLE_DOWN, Action.SURRENDER
                ])
            reward, done, status = act(action)
            if action == Action.RETRY:
                # RETRYはグローバルでカウント
                g_retry_counter += 1
            prev_state = state
            state = get_state(remaining_cards=remaining_cards)
            if not testmode:
                score_diff = (state[0] - prev_state[0]) * 21.0
                bonus = 0.0
                if score_diff > 0:
                    bonus = 0.05 * score_diff
                total_reward = reward + bonus
                a_idx = action_to_index(action)
                agent.add_experience(prev_state, a_idx, total_reward, state, done)
                agent.update()
                agent.decay_epsilon_step()
            if done:
                # unsettled や retry の場合は記録を飛ばす
                status_l = str(status).lower() if status is not None else ''
                if ('unsettled' in status_l) or ('retry' in status_l):
                    pass
                else:
                    try:
                        r = float(reward)
                    except Exception:
                        r = 0.0
                    outcomes.append(1 if r > 0.0 else 0)
                break

    n_finished = len(outcomes)
    if n_finished == 0:
        return 0.0, 0

    last_slice = outcomes[-min(window, n_finished):]
    win_rate = float(sum(last_slice)) / len(last_slice)
    return win_rate, n_finished


def objective(trial, args):
    # サーチ空間
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float('gamma', 0.8, 0.999)
    eps = trial.suggest_float('eps', 0.1, 0.5)
    buffer_size = trial.suggest_categorical('buffer_size', [500, 1000, 5000, 10000])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    target_update = trial.suggest_categorical('target_update', [100, 500, 1000])

    # エージェント初期化
    agent = Agent(STATE_SIZE, ACTION_SIZE, lr, gamma, eps, buffer_size=buffer_size, batch_size=batch_size, target_update_freq=target_update)

    # ゲーム実行（trial 毎に指定数を回す）
    win_rate, n_finished = run_games_with_agent(agent, args.games_per_trial, window=args.window, testmode=False)

    trial.set_user_attr('win_rate', win_rate)
    trial.set_user_attr('n_finished', n_finished)
    # 目的は最後 window 回の勝率の最大化
    return win_rate


def main():
    if optuna is None:
        print('optuna がインストールされていません。pip install optuna で導入してください。')
        return

    parser = argparse.ArgumentParser(description='AI Black Jack Player (DQN with Optuna)')
    parser.add_argument('--trials', type=int, default=10, help='number of optuna trials')
    parser.add_argument('--games-per-trial', type=int, default=50, help='games to play per trial (keeps optuna fast)')
    parser.add_argument('--window', type=int, default=500, help='window size (last N finished games) to compute win rate')
    parser.add_argument('--study-name', type=str, default='bj_dqn_optuna', help='optuna study name')
    parser.add_argument('--storage', type=str, default='', help='optuna storage url (optional)')
    parser.add_argument('--save-best', type=str, default='dqn_model_optuna_best.pth', help='path to save best model (after rerun)')
    parser.add_argument('--load', type=str, default='', help='optional pretrained model to load before training')
    args = parser.parse_args()

    # パラメータを渡すために args をラップ
    class _Args:
        pass
    _args = _Args()
    _args.games_per_trial = args.games_per_trial
    _args.window = args.window

    # Optuna study
    study = optuna.create_study(direction='maximize', study_name=args.study_name, storage=args.storage or None)
    study.optimize(lambda t: objective(t, _args), n_trials=args.trials)

    print('Best trial:')
    best = study.best_trial
    print(f'  Value: {best.value}')
    print('  Params:')
    for k, v in best.params.items():
        print(f'    {k}: {v}')

    # ユーザーに促す: ベストのハイパーパラメータで再学習してモデルを保存
    print('\nTo save a model with the best hyperparameters, run this script in training mode by passing the best params as arguments.\n')


if __name__ == '__main__':
    main()
