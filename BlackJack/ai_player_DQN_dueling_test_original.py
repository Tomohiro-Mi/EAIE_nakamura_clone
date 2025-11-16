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

# ハイパーパラメータ
TOTAL_GAMES = 1000 # 総ゲーム数
BUFFER_SIZE = 1000 # 経験再生バッファのサイズ
BATCH_SIZE = 32 # ミニバッチサイズ
STATE_SIZE = 10
ACTION_SIZE = 5 # 行動の種類数
TARGET_UPDATE_FREQ = 500 # ターゲットネットワークの更新頻度（ステップ単位）
EPS = 0.1 # ε-greedyにおけるε
LEARNING_RATE = 0.001 # 学習率（DQN用に小さめ）
DISCOUNT_FACTOR = 0.99 # 割引率

# ログ制御: edit は詳細出力あり
VERBOSE = True

def log(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

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


class DuelingQNetwork(torch.nn.Module):
    """Dueling DQN implementation for edit script (prints enabled)."""
    def __init__(self, state_size: int, action_size: int):
        super(DuelingQNetwork, self).__init__()
        hid = 256
        self.shared1 = torch.nn.Linear(state_size, hid)
        self.shared2 = torch.nn.Linear(hid, hid)

        # value stream
        self.value_fc = torch.nn.Linear(hid, 128)
        self.value_out = torch.nn.Linear(128, 1)

        # advantage stream
        self.adv_fc = torch.nn.Linear(hid, 128)
        self.adv_out = torch.nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.shared1(x))
        x = torch.relu(self.shared2(x))

        v = torch.relu(self.value_fc(x))
        v = self.value_out(v)

        a = torch.relu(self.adv_fc(x))
        a = self.adv_out(a)

        return v + (a - a.mean(dim=1, keepdim=True))
    

class Agent:
    def __init__(self, state_size: int, action_size: int, learning_rate: float, discount_factor: float, epsilon: float, epsilon_min: float = 0.01, buffer_size: int = 10000):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = (epsilon - epsilon_min) / TOTAL_GAMES

        self.replay = ExperienceReplayBuffer(buffer_size=buffer_size, batch_size=BATCH_SIZE)
        self.data = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Qネットワーク（オンライン／ターゲット） — Dueling
        self.original_q_network = DuelingQNetwork(state_size, action_size).to(self.device)
        self.target_q_network = DuelingQNetwork(state_size, action_size).to(self.device)
        self.sync_net()  # 初期同期

        # 追加属性（他メソッドで参照されるもの）
        self.gamma = discount_factor
        self.batch_size = BATCH_SIZE
        self.learn_steps = 0
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
        self.epsilon -= self.epsilon_decay

    def save_model(self) -> None:
        torch.save(self.original_q_network.state_dict(), 'dqn_model.pth')


### 関数 ###

# ゲームを開始する
def game_start(game_ID=0, remaining_cards=None):
    global g_retry_counter, player, soc

    log('Game {0} start.'.format(game_ID))
    log('  money: ', player.get_money(), '$')

    # RETRY回数カウンターの初期化
    g_retry_counter = 0

    # ディーラープログラムに接続する
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.connect((socket.gethostname(), PORT))
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # ベット
    bet, money = player.set_bet()
    log('Action: BET')
    log('  money: ', money, '$')
    log('  bet: ', bet, '$')

    # ディーラーから「カードシャッフルを行ったか否か」の情報を取得
    # シャッフルが行われた場合は True が, 行われなかった場合は False が，変数 cardset_shuffled にセットされる
    # なお，本サンプルコードではここで取得した情報は使用していない
    cardset_shuffled = player.receive_card_shuffle_status(soc)
    if cardset_shuffled:
        log('Dealer said: Card set has been shuffled before this game.')
        remaining_cards.shuffle()
    # ディーラーから初期カード情報を受信
    dc, pc1, pc2 = player.receive_init_cards(soc)
    remaining_cards.draw(dc)
    remaining_cards.draw(pc1)
    remaining_cards.draw(pc2)

    log('Delaer gave cards.')
    log('  dealer-card: ', get_card_info(dc))
    log('  player-card 1: ', get_card_info(pc1))
    log('  player-card 2: ', get_card_info(pc2))
    log('  current score: ', player.get_score())

# 現時点での手札情報（ディーラー手札は見えているもののみ）を取得
def get_current_hands():
    return copy.deepcopy(player.player_hand), copy.deepcopy(player.dealer_hand)

# HITを実行する
def hit():
    global player, soc
    log('Action: HIT')

    # ディーラーにメッセージを送信
    player.send_message(soc, 'hit')

    # ディーラーから情報を受信
    pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True)
    log('  player-card {0}: '.format(player.get_num_player_cards()), get_card_info(pc))
    log('  current score: ', score)

    # バーストした場合はゲーム終了
    if status == 'bust':
        for i in range(len(dc)):
            log('  dealer-card {0}: '.format(i+2), get_card_info(dc[i]))
        log("  dealer's score: ", player.get_dealer_score())
        soc.close() # ディーラーとの通信をカット
        reward = player.update_money(rate=rate) # 所持金額を更新
        log('Game finished.')
        log('  result: bust')
        log('  money: ', player.get_money(), '$')
        return reward, True, status

    # バーストしなかった場合は続行
    else:
        return 0, False, status

# STANDを実行する
def stand():
    global player, soc
    log('Action: STAND')

    # ディーラーにメッセージを送信
    player.send_message(soc, 'stand')

    # ディーラーから情報を受信
    score, status, rate, dc = player.receive_message(dsoc=soc, get_dealer_cards=True)
    log('  current score: ', score)
    for i in range(len(dc)):
        log('  dealer-card {0}: '.format(i+2), get_card_info(dc[i]))
    log("  dealer's score: ", player.get_dealer_score())

    # ゲーム終了，ディーラーとの通信をカット
    soc.close()

    # 所持金額を更新
    reward = player.update_money(rate=rate)
    log('Game finished.')
    log('  result: ', status)
    log('  money: ', player.get_money(), '$')
    return reward, True, status

# DOUBLE_DOWNを実行する
def double_down():
    global player, soc
    log('Action: DOUBLE DOWN')

    # 今回のみベットを倍にする
    bet, money = player.double_bet()
    log('  money: ', money, '$')
    log('  bet: ', bet, '$')

    # ディーラーにメッセージを送信
    player.send_message(soc, 'double_down')

    # ディーラーから情報を受信
    pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True)
    log('  player-card {0}: '.format(player.get_num_player_cards()), get_card_info(pc))
    log('  current score: ', score)
    for i in range(len(dc)):
        log('  dealer-card {0}: '.format(i+2), get_card_info(dc[i]))
    log("  dealer's score: ", player.get_dealer_score())

    # ゲーム終了，ディーラーとの通信をカット
    soc.close()

    # 所持金額を更新
    reward = player.update_money(rate=rate)
    log('Game finished.')
    log('  result: ', status)
    log('  money: ', player.get_money(), '$')
    return reward, True, status

# SURRENDERを実行する
def surrender():
    global player, soc
    log('Action: SURRENDER')

    # ディーラーにメッセージを送信
    player.send_message(soc, 'surrender')

    # ディーラーから情報を受信
    score, status, rate, dc = player.receive_message(dsoc=soc, get_dealer_cards=True)
    log('  current score: ', score)
    for i in range(len(dc)):
        log('  dealer-card {0}: '.format(i+2), get_card_info(dc[i]))
    log("  dealer's score: ", player.get_dealer_score())

    # ゲーム終了，ディーラーとの通信をカット
    soc.close()

    # 所持金額を更新
    reward = player.update_money(rate=rate)
    log('Game finished.')
    log('  result: ', status)
    log('  money: ', player.get_money(), '$')
    return reward, True, status

# RETRYを実行する
def retry():
    global player, soc
    log('Action: RETRY')

    # ベット額の 1/4 を消費
    penalty = player.current_bet // 4
    player.consume_money(penalty)
    log('  player-card {0} has been removed.'.format(player.get_num_player_cards()))
    log('  money: ', player.get_money(), '$')

    # ディーラーにメッセージを送信
    player.send_message(soc, 'retry')

    # ディーラーから情報を受信
    pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True, retry_mode=True)
    log('  player-card {0}: '.format(player.get_num_player_cards()), get_card_info(pc))
    log('  current score: ', score)

    # バーストした場合はゲーム終了
    if status == 'bust':
        for i in range(len(dc)):
            log('  dealer-card {0}: '.format(i+2), get_card_info(dc[i]))
        log("  dealer's score: ", player.get_dealer_score())
        soc.close() # ディーラーとの通信をカット
        reward = player.update_money(rate=rate) # 所持金額を更新
        log('Game finished.')
        log('  result: bust')
        log('  money: ', player.get_money(), '$')
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
def get_state(remainng_cards: RemainingCards=None):
    # 現在の手札情報を取得
    p_hand, d_hand = get_current_hands()

    # 生のスコア（0-21）
    raw_score = p_hand.get_score()
    length = p_hand.length()    # プレイヤー手札の枚数
    p_hand_is_nbj = p_hand.is_nbj()
    p_hand_is_busted = p_hand.is_busted()
    d_hand_1 = d_hand[0]

    # money をビン化（簡易: 0-4 の範囲に丸める）
    # train 側の実装と互換を持たせるために簡易的なビン化を採用
    if player.get_money() < 0:
        money_bin = 0
    elif player.get_money() < BET:
        money_bin = 1
    elif player.get_money() < INITIAL_MONEY * 0.8:
        money_bin = 2
    elif player.get_money() < INITIAL_MONEY:
        money_bin = 3
    else:
        money_bin = 4

    # 残りカード 4分類 (2-6,7-9,10-K,A)
    full_counts = remainng_cards.get_remaining_card_counts() if remainng_cards is not None else [0]*13
    two_to_six = sum(full_counts[1:6])     # 2..6
    seven_to_nine = sum(full_counts[6:9])  # 7..9
    ten_to_king = sum(full_counts[9:13])   # 10,J,Q,K
    aces = full_counts[0]

    # 正規化を行う（すべて 0-1 にスケーリング）
    score_norm = float(raw_score) / 21.0
    length_norm = float(length) / 11.0
    dealer_norm = float(d_hand_1) / 12.0
    max_two_to_six = 5 * 4 * N_DECKS
    max_seven_to_nine = 3 * 4 * N_DECKS
    max_ten_to_king = 4 * 4 * N_DECKS
    max_aces = 1 * 4 * N_DECKS
    rem4_norm = [
        two_to_six / max_two_to_six if max_two_to_six>0 else 0.0,
        seven_to_nine / max_seven_to_nine if max_seven_to_nine>0 else 0.0,
        ten_to_king / max_ten_to_king if max_ten_to_king>0 else 0.0,
        aces / max_aces if max_aces>0 else 0.0,
    ]
    money_norm = float(money_bin) / 4.0

    # Dueling の学習時と互換のある 10 次元の正規化ベクトルを返す
    flat_state = np.array([
        score_norm, length_norm,
        float(int(p_hand_is_nbj)), float(int(p_hand_is_busted)),
        dealer_norm,
        *rem4_norm,
        money_norm
    ], dtype=np.float32)

    return flat_state


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

# 行動戦略
def select_action(state, strategy: Strategy, agent: Agent=None):

    # Q値最大行動を選択する戦略
    if strategy == Strategy.QMAX:
        # Use the network outputs and apply the same valid-action mask used in training
        state_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        q_vals = agent.original_q_network(state_tensor).detach().cpu().numpy()[0]
        mask = get_valid_action_mask_from_state(state)
        q_vals_masked = np.copy(q_vals)
        # invalidate illegal actions by setting very low value
        q_vals_masked[~mask] = -np.inf
        idx = int(np.nanargmax(q_vals_masked))
        return index_to_action(idx)

    # ε-greedy
    elif strategy == Strategy.E_GREEDY:
        if np.random.rand() < agent.epsilon:
            return select_action(state, strategy=Strategy.RANDOM)
        else:
            idx = agent.get_best_action(state)
            return index_to_action(idx)

    # ランダム戦略
    else:
        z = np.random.randint(0, 5)
        if z == 0:
            return Action.HIT
        elif z == 1:
            return Action.STAND
        elif z == 2:
            return Action.DOUBLE_DOWN
        elif z == 3:
            return Action.SURRENDER
        else: # z == 4 のとき
            return Action.RETRY


### ここから処理開始 ###

def main():
    global g_retry_counter, player, soc

    parser = argparse.ArgumentParser(description='AI Black Jack Player (Q-learning)')
    parser.add_argument('--games', type=int, default=1000, help='num. of games to play')
    parser.add_argument('--history', type=str, default='play_log.csv', help='filename where game history will be saved')
    parser.add_argument('--log-mode', type=str, choices=('per_action','per_game'), default='per_game', help='logging mode: per_action (detailed) or per_game (summary)')
    parser.add_argument('--load', type=str, default='', help='filename of Q table to be loaded before learning')
    parser.add_argument('--save', type=str, default='', help='filename to save trained model')
    parser.add_argument('--testmode', help='this option runs the program without learning', action='store_true')
    args = parser.parse_args()

    if args.games == -1:
        n_games = TOTAL_GAMES
    else:
        n_games = args.games + 1

    # 学習済みモデルの読み込み
    if args.load != '':
        print('Loading model from {}...'.format(args.load))
        # モデルのstate_dictを先にロードして、state_sizeを動的に取得
        checkpoint = torch.load(args.load)
        loaded_state_size = checkpoint['shared1.weight'].shape[1]
        agent = Agent(loaded_state_size, ACTION_SIZE, LEARNING_RATE, DISCOUNT_FACTOR, EPS, buffer_size=BUFFER_SIZE)
        agent.original_q_network.load_state_dict(checkpoint)
        agent.sync_net()
        print('Model loaded.')

    # ログファイル: per_action (詳細) / per_game (サマリ)
    action_logfile = None
    game_logfile = None
    if args.history:
        if args.log_mode == 'per_action':
            action_logfile = open(args.history, 'w')
            print('score,hand_length,p_hand_1,p_hand_2,p_hand_is_nbj,p_hand_is_busted,d_hand_1,d_hand_length,' + \
                'remaining_A,remaining_2,remaining_3,remaining_4,remaining_5,remaining_6,remaining_7,remaining_8,remaining_9,remaining_10,remaining_J,remaining_Q,remaining_K,' + \
                'money,action,reward,status', file=action_logfile)
        else:
            game_logfile = open(args.history, 'w')
            print('game_id,result,reward', file=game_logfile)

    # n_games回ゲームを実行
    for n in tqdm(range(1, n_games)):

        # 残りカード数
        remaining_cards = RemainingCards(N_DECKS)

        # nゲーム目を開始
        game_start(n, remaining_cards=remaining_cards)

        # 「現在の状態」を取得
        state = get_state(remainng_cards=remaining_cards)

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
            action_name = get_action_name(action) # 行動名を表す文字列を取得

            # 選択した行動を実際に実行
            # 戻り値:
            #   - done: 終了フラグ．今回の行動によりゲームが終了したか否か（終了した場合はTrue, 続行中ならFalse）
            #   - reward: 獲得金額（ゲーム続行中の場合は 0 , ただし RETRY を実行した場合は1回につき -BET/4 ）
            #   - status: 行動実行後のプレイヤーステータス（バーストしたか否か，勝ちか負けか，などの状態を表す文字列）
            reward, done, status = act(action)

            # 実行した行動がRETRYだった場合はRETRY回数カウンターを1増やす
            if action == Action.RETRY:
                g_retry_counter += 1

            # 「現在の状態」を再取得
            prev_state = state # 行動前の状態を別変数に退避
            prev_score = prev_state[0] # 行動前のプレイヤー手札のスコア（prev_state の一つ目の要素）
            state = get_state(remainng_cards=remaining_cards)
            score = state[0] # 行動後のプレイヤー手札のスコア（state の一つ目の要素）

            # 学習（経験の追加と更新）
            if not args.testmode:
                try:
                    a_idx = action_to_index(action)
                    agent.add_experience(prev_state, a_idx, reward, state, done)
                    agent.update()
                except Exception as e:
                    # 学習エラーはログに流しつつゲーム続行
                    print(f"[LEARN-ERROR] {e}")

            # per_action ログ
            if action_logfile:
                print('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21},{22},{23},{24}'.format(
                    prev_state[0], prev_state[1], prev_state[2], prev_state[3], prev_state[4], prev_state[5],
                    prev_state[6], prev_state[7],
                    prev_state[8], prev_state[9], prev_state[10], prev_state[11], prev_state[12],
                    prev_state[13], prev_state[14], prev_state[15], prev_state[16], prev_state[17],
                    prev_state[18], prev_state[19], prev_state[20],
                    prev_state[21],
                    action_name,
                    reward,
                    status
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
            # 終了フラグが立った場合はnゲーム目を終了
            if done == True:
                break

        print('')

    # エピソード終了ごとにεを減衰
        if not args.testmode:
            agent.set_epsilon()

        print('')

    # ログファイルを閉じる
    if action_logfile:
        action_logfile.close()
    if game_logfile:
        game_logfile.close()

    # 学習済みモデルを保存
    if not args.testmode:
        if args.save != '':
            agent.save_model()


if __name__ == '__main__':
    main()
