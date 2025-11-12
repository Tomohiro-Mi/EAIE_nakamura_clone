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

# ハイパーパラメータ
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
# 0: 借金 (<0), 1: 破産寸前 (<BET), 2: 通常 (<INITIAL_MONEY), 3: 潤沢 (>=INITIAL_MONEY)
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

# 残り枚数を4分類（2-6, 7-9, 10-K, A）に集約
def remaining_counts_4cats(counts: list[int]) -> list[int]:
    # counts: 0:A,1:2,2:3,3:4,4:5,5:6,6:7,7:8,8:9,9:10,10:J,11:Q,12:K
    two_to_six = sum(counts[1:6])     # 2..6
    seven_to_nine = sum(counts[6:9])  # 7..9
    ten_to_king = sum(counts[9:13])   # 10,J,Q,K
    aces = counts[0]                  # A
    return [two_to_six, seven_to_nine, ten_to_king, aces]

# スコア4段階ビン化
def score_to_bin(score: int) -> int:
    # 旧来のビン化は使わない（互換性維持のため関数は残す）
    if score <= 8:
        return 0
    if score <= 11:
        return 1
    if score <= 16:
        return 2
    return 3

# DQNの行動インデックスをAction列挙に変換
def index_to_action(i: int) -> Action:
    mapping = [Action.HIT, Action.STAND, Action.DOUBLE_DOWN, Action.SURRENDER, Action.RETRY]
    return mapping[i]

# Action列挙からDQNの行動インデックスへ変換
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
        # 深めのネットワークに変更
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
    def __init__(self, state_size: int, action_size: int, learning_rate: float, discount_factor: float, epsilon: float, epsilon_min: float = 0.01, buffer_size: int = 10000):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        # εはステップ単位で減衰させる（ゲームごとではなく行動ごと）
        self.epsilon_decay = (epsilon - epsilon_min) / (TOTAL_GAMES * 20)

        self.replay = ExperienceReplayBuffer(buffer_size=buffer_size, batch_size=BATCH_SIZE)
        self.data = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Qネットワーク（オンライン／ターゲット）
        self.original_q_network = QNetwork(state_size, action_size).to(self.device)
        self.target_q_network = QNetwork(state_size, action_size).to(self.device)
        self.sync_net()  # 初期同期

        # 追加属性（他メソッドで参照されるもの）
        self.gamma = discount_factor
        self.batch_size = BATCH_SIZE
        self.learn_steps = 0
        self.total_steps = 0
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

        # Q(s, a) の抽出
        q_values = self.original_q_network(self.data.state)
        actions = self.data.action.long()  # gather用に整数化
        q_sa = q_values[torch.arange(self.batch_size), actions]

        # ターゲット値の計算
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
        if self.learn_steps % TARGET_UPDATE_FREQ == 0:
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

    def save_model(self, path: str = 'dqn_model.pth') -> None:
        torch.save(self.original_q_network.state_dict(), path)


### 環境インタフェース ###

# ゲームを開始する
def game_start(game_ID=0, remaining_cards: RemainingCards | None = None):
    global g_retry_counter, player, soc, g_remaining_cards

    # RETRY回数カウンターの初期化
    g_retry_counter = 0

    # ディーラープログラムに接続する
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.connect((socket.gethostname(), PORT))
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # ベット
    bet, money = player.set_bet()

    # ディーラーから「カードシャッフルを行ったか否か」の情報を取得
    cardset_shuffled = player.receive_card_shuffle_status(soc)
    if cardset_shuffled and remaining_cards is not None:
        remaining_cards.shuffle()

    # ディーラーから初期カード情報を受信
    dc, pc1, pc2 = player.receive_init_cards(soc)
    # 残りカード管理を保持
    g_remaining_cards = remaining_cards
    if g_remaining_cards is not None:
        g_remaining_cards.draw(dc)
        g_remaining_cards.draw(pc1)
        g_remaining_cards.draw(pc2)

# 現時点での手札情報（ディーラー手札は見えているもののみ）を取得
def get_current_hands():
    return copy.deepcopy(player.player_hand), copy.deepcopy(player.dealer_hand)

# HITを実行する
def hit():
    global player, soc, g_remaining_cards

    # ディーラーにメッセージを送信
    player.send_message(soc, 'hit')

    # ディーラーから情報を受信
    pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True)

    # 引いたカードを残り枚数から減算
    if g_remaining_cards is not None:
        g_remaining_cards.draw(pc)

    # バーストした場合はゲーム終了
    if status == 'bust':
        # ディーラー公開カードを減算
        if g_remaining_cards is not None:
            for c in dc:
                g_remaining_cards.draw(c)
        soc.close()
        reward = player.update_money(rate=rate)
        return reward, True, status

    # バーストしなかった場合は続行
    else:
        return 0, False, status

# STANDを実行する
def stand():
    global player, soc, g_remaining_cards

    # ディーラーにメッセージを送信
    player.send_message(soc, 'stand')

    # ディーラーから情報を受信
    score, status, rate, dc = player.receive_message(dsoc=soc, get_dealer_cards=True)

    # ディーラー公開カードを減算
    if g_remaining_cards is not None:
        for c in dc:
            g_remaining_cards.draw(c)

    # ゲーム終了
    soc.close()

    # 所持金額を更新
    reward = player.update_money(rate=rate)

    return reward, True, status

# DOUBLE_DOWNを実行する
def double_down():
    global player, soc, g_remaining_cards

    # 今回のみベットを倍にする
    bet, money = player.double_bet()

    # ディーラーにメッセージを送信
    player.send_message(soc, 'double_down')

    # ディーラーから情報を受信
    pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True)

    # 引いたカードを減算
    if g_remaining_cards is not None:
        g_remaining_cards.draw(pc)

    # ディーラー公開カードを減算
    if g_remaining_cards is not None:
        for c in dc:
            g_remaining_cards.draw(c)

    # ゲーム終了
    soc.close()

    # 所持金額を更新
    reward = player.update_money(rate=rate)

    return reward, True, status

# SURRENDERを実行する
def surrender():
    global player, soc, g_remaining_cards

    # ディーラーにメッセージを送信
    player.send_message(soc, 'surrender')

    # ディーラーから情報を受信
    score, status, rate, dc = player.receive_message(dsoc=soc, get_dealer_cards=True)

    # ディーラー公開カードを減算
    if g_remaining_cards is not None:
        for c in dc:
            g_remaining_cards.draw(c)

    # ゲーム終了
    soc.close()

    # 所持金額を更新
    reward = player.update_money(rate=rate)

    return reward, True, status

# RETRYを実行する
def retry():
    global player, soc, g_remaining_cards

    # ベット額の 1/4 を消費
    penalty = player.current_bet // 4
    player.consume_money(penalty)

    # ディーラーにメッセージを送信
    player.send_message(soc, 'retry')

    # ディーラーから情報を受信
    pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True, retry_mode=True)

    # 引いたカードを減算
    if g_remaining_cards is not None:
        g_remaining_cards.draw(pc)

    # バーストした場合はゲーム終了
    if status == 'bust':
        soc.close()
        reward = player.update_money(rate=rate)
        return reward-penalty, True, status

    # バーストしなかった場合は続行
    else:
        return -penalty, False, status

# 行動の実行
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


### これ以降の関数が重要 ###

# 現在の状態の取得
def get_state(remaining_cards: RemainingCards | None = None):
    # 現在の手札情報を取得
    #   - p_hand: プレイヤー手札
    #   - d_hand: ディーラー手札（見えているもののみ）
    p_hand, d_hand = get_current_hands()

    # 生のスコア（0-21）
    raw_score = p_hand.get_score()
    length = p_hand.length()    # プレイヤー手札の枚数
    p_hand_is_nbj = p_hand.is_nbj()
    p_hand_is_busted = p_hand.is_busted()
    d_hand_1 = d_hand[0]

    # money をビン化
    money_bin = money_to_bin(player.get_money())

    # 残りカード 4分類
    ref_cards = remaining_cards if remaining_cards is not None else g_remaining_cards
    full_counts = ref_cards.get_remaining_card_counts() if ref_cards is not None else [0]*13
    rem4 = remaining_counts_4cats(full_counts)

    # 正規化を行う（すべて 0-1 にスケーリング）
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

    # フラットな固定長ベクトル（10次元）
    flat_state = np.array([
        score_norm, length_norm,
        float(int(p_hand_is_nbj)), float(int(p_hand_is_busted)),
        dealer_norm,
        *rem4_norm,
        money_norm
    ], dtype=np.float32)

    return flat_state

# 行動戦略
def select_action(state, strategy: Strategy, agent: Agent=None):

    # Q値最大行動を選択する戦略
    if strategy == Strategy.QMAX:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        q_vals = agent.original_q_network(state_tensor).detach().cpu().numpy()[0]
        mask = get_valid_action_mask_from_state(state)
        q_vals_masked = np.copy(q_vals)
        q_vals_masked[~mask] = -1e9
        return index_to_action(int(np.argmax(q_vals_masked)))

    # ε-greedy
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

    # ランダム戦略: 有効な行動のみ
    else:
        mask = get_valid_action_mask_from_state(state)
        valid_indices = np.where(mask)[0]
        chosen = np.random.choice(valid_indices)
        return index_to_action(int(chosen))


def get_valid_action_mask_from_state(state: np.ndarray) -> np.ndarray:
    """stateは正規化されたベクトル。返り値は長さ ACTION_SIZE の bool 配列で、各行動の有効性を示す。"""
    score_norm = state[0]
    length_norm = state[1]
    # 正規化を戻して判定
    length = int(round(length_norm * 11))
    mask = np.ones(ACTION_SIZE, dtype=bool)
    # DOUBLE_DOWN は手札が2枚のときのみ有効と仮定
    if length != 2:
        mask[action_to_index(Action.DOUBLE_DOWN)] = False
    # RETRY はグローバルの g_retry_counter を参照
    if g_retry_counter >= RETRY_MAX:
        mask[action_to_index(Action.RETRY)] = False
    return mask


### ここから処理開始 ###

def main():
    global g_retry_counter, player, soc

    parser = argparse.ArgumentParser(description='AI Black Jack Player (DQN train)')
    parser.add_argument('--games', type=int, default=TOTAL_GAMES, help='num. of games to play')
    parser.add_argument('--history', type=str, default='', help='(optional) CSV file to save transitions or game summaries')
    parser.add_argument('--log-mode', type=str, choices=('per_action', 'per_game'), default='per_game', help='logging mode: per_action (old) or per_game (summary)')
    parser.add_argument('--load', type=str, default='', help='filename of model to be loaded before learning')
    parser.add_argument('--save', type=str, default='dqn_model.pth', help='filename to save trained model')
    parser.add_argument('--testmode', action='store_true', help='run without learning (greedy)')
    args = parser.parse_args()

    n_games = (TOTAL_GAMES if args.games == -1 else args.games) + 1

    # エージェント構築
    agent = Agent(STATE_SIZE, ACTION_SIZE, LEARNING_RATE, DISCOUNT_FACTOR, EPS, buffer_size=BUFFER_SIZE)

    # 学習済みモデルの読み込み
    if args.load:
        agent.original_q_network.load_state_dict(torch.load(args.load))
        agent.sync_net()

    # オプションのCSV: per_action（従来）/ per_game（1行=1ゲーム）を切替
    action_logfile = None
    game_logfile = None
    if args.history:
        if args.log_mode == 'per_action':
            action_logfile = open(args.history, 'w')
            # 従来の詳細ログ（ヘッダは旧フォーマットに合わせる）
            print('score,hand_length,p_hand_is_nbj,p_hand_is_busted,d_hand_1,'
                  'remaining_2_6,remaining_7_9,remaining_10_K,remaining_A,'
                  'money_bin,action,reward,result', file=action_logfile)
        else:
            game_logfile = open(args.history, 'w')
            print('game_id,result,reward', file=game_logfile)

    # n_games回ゲームを実行（進捗tqdm）
    for n in tqdm(range(1, n_games)):
        # 残りカード数
        remaining_cards = RemainingCards(N_DECKS)

        # nゲーム目を開始
        game_start(n, remaining_cards=remaining_cards)

        # 「現在の状態」を取得
        state = get_state(remaining_cards=remaining_cards)

        while True:

            # 次に実行する行動を選択
            if args.testmode:
                action = select_action(state, Strategy.QMAX, agent)
            else:
                action = select_action(state, Strategy.E_GREEDY, agent)
            if g_retry_counter >= RETRY_MAX and action == Action.RETRY:
                # RETRY回数が上限に達しているにもかかわらずRETRYが選択された場合，他の行動をランダムに選択
                action = np.random.choice([
                    Action.HIT, Action.STAND, Action.DOUBLE_DOWN, Action.SURRENDER
                ])

            # 行動実行
            reward, done, status = act(action)

            # RETRY回数カウント
            if action == Action.RETRY:
                g_retry_counter += 1

            # 次状態
            prev_state = state
            state = get_state(remaining_cards=remaining_cards)

            # 学習（経験の追加と更新）
            if not args.testmode:
                # 中間報酬: スコアの改善があれば小さな正の報酬を与える
                # state のスコアは正規化済み (0-1), 実スコア差は *21 で復元
                score_diff = (state[0] - prev_state[0]) * 21.0
                bonus = 0.0
                if score_diff > 0:
                    bonus = 0.05 * score_diff
                total_reward = reward + bonus

                a_idx = action_to_index(action)
                agent.add_experience(prev_state, a_idx, total_reward, state, done)
                agent.update()
                # εを行動単位で減衰
                agent.decay_epsilon_step()

            # per_action ログ（行動ごと）
            if action_logfile:
                # 出力は旧フォーマットに合わせて値を落とす（必要に応じて拡張可）
                try:
                    score_out = prev_state[0]
                    hand_len_out = prev_state[1]
                    p_nbj = int(prev_state[2])
                    p_bst = int(prev_state[3])
                    d_hand_1_out = int(round(prev_state[4]*12))
                    rem2 = int(round(prev_state[5]* (5*4*N_DECKS)))
                    rem7 = int(round(prev_state[6]* (3*4*N_DECKS)))
                    rem10 = int(round(prev_state[7]* (4*4*N_DECKS)))
                    remA = int(round(prev_state[8]* (1*4*N_DECKS)))
                    money_out = int(round(prev_state[9]*4))
                except Exception:
                    # フォールバック: 出力不可なら空欄で出す
                    score_out = hand_len_out = p_nbj = p_bst = d_hand_1_out = 0
                    rem2 = rem7 = rem10 = remA = money_out = 0
                print('{},{},{},{},{},{},{},{},{},{},{},{}'.format(
                    score_out, hand_len_out, p_nbj, p_bst, d_hand_1_out,
                    rem2, rem7, rem10, remA, money_out,
                    get_action_name(action), reward
                ), file=action_logfile)

            # per_game ログ（終了時のみ）
            if done and game_logfile:
                status_l = str(status).lower() if status is not None else ''
                if 'unsettled' in status_l or 'retry' in status_l:
                    pass
                else:
                    try:
                        r = float(reward)
                    except Exception:
                        r = 0.0
                    if r > 0.0:
                        res = 'win'
                    elif r < 0.0:
                        res = 'lose'
                    else:
                        res = None
                    if res is not None:
                        print(f"{n},{res},{r}", file=game_logfile)

            if done:
                break

    # εは行動単位で減衰しているため，ここでは何もしない
    pass

    if game_logfile:
        game_logfile.close()
    if action_logfile:
        action_logfile.close()

    # 学習済みモデルを保存
    if not args.testmode and args.save:
        agent.save_model(args.save)


if __name__ == '__main__':
    main()
