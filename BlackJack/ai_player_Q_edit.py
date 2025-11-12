import copy
import socket
import argparse
import numpy as np
import sys
import os
import contextlib
from classes import Action, Strategy, QTable, Player, RemainingCards, get_card_info, get_action_name
from config import PORT, BET, INITIAL_MONEY, N_DECKS
from tqdm import tqdm


# 1ゲームあたりのRETRY回数の上限
RETRY_MAX = 10


### グローバル変数 ###

# ゲームごとのRETRY回数のカウンター
g_retry_counter = 0

# プレイヤークラスのインスタンスを作成
player = Player(initial_money=INITIAL_MONEY, basic_bet=BET)

# ディーラーとの通信用ソケット
soc = None

# Q学習用のQテーブル
q_table = QTable(action_class=Action, default_value=0)

# Q学習の設定値
EPS = 0.3 # ε-greedyにおけるε
LEARNING_RATE = 0.1 # 学習率
DISCOUNT_FACTOR = 0.9 # 割引率

# 残りカードのグローバル参照（各アクションで更新するため）
g_remaining_cards: RemainingCards | None = None

# 所持金を大雑把なビンに変換
# 0: 借金 (<0), 1: 破産寸前 (<BET), 2: 通常 (<INITIAL_MONEY), 3: 潤沢 (>=INITIAL_MONEY)
def money_to_bin(money: int) -> int:
    if money < 0:
        return 0
    if money < BET:
        return 1
    if money < INITIAL_MONEY:
        return 2
    return 3

# 残り枚数を4分類（2-6, 7-9, 10-K, A）に集約
def remaining_counts_4cats(counts: list[int]) -> list[int]:
    # インデックス対応: 0:A, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:8, 8:9, 9:10, 10:J, 11:Q, 12:K
    two_to_six = sum(counts[1:6])     # 2..6 -> idx 1..5
    seven_to_nine = sum(counts[6:9])  # 7..9 -> idx 6..8
    ten_to_king = sum(counts[9:13])   # 10,J,Q,K -> idx 9..12
    aces = counts[0]                  # A -> idx 0
    return [two_to_six, seven_to_nine, ten_to_king, aces]

# スコアを4段階にビン化
# 0: <=8, 1: 9-11, 2: 12-16, 3: >=17
def score_to_bin(score: int) -> int:
    if score <= 8:
        return 0
    if score <= 11:
        return 1
    if score <= 16:
        return 2
    return 3

# ゲームを開始する
def game_start(game_ID=0, remaining_cards: RemainingCards | None = None):
    global g_retry_counter, player, soc, g_remaining_cards

    print('Game {0} start.'.format(game_ID))
    print('  money: ', player.get_money(), '$')

    # RETRY回数カウンターの初期化
    g_retry_counter = 0

    # ディーラープログラムに接続する
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.connect((socket.gethostname(), PORT))
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # ベット
    bet, money = player.set_bet()
    print('Action: BET')
    print('  money: ', money, '$')
    print('  bet: ', bet, '$')

    # ディーラーから「カードシャッフルを行ったか否か」の情報を取得
    # シャッフルが行われた場合は True が, 行われなかった場合は False が，変数 cardset_shuffled にセットされる
    # なお，本サンプルコードではここで取得した情報は使用していない
    cardset_shuffled = player.receive_card_shuffle_status(soc)
    if cardset_shuffled:
        print('Dealer said: Card set has been shuffled before this game.')
        if remaining_cards is not None:
            remaining_cards.shuffle()

    # ディーラーから初期カード情報を受信
    dc, pc1, pc2 = player.receive_init_cards(soc)
    # 残りカード管理をグローバルに保持
    g_remaining_cards = remaining_cards
    if g_remaining_cards is not None:
        g_remaining_cards.draw(dc)
        g_remaining_cards.draw(pc1)
        g_remaining_cards.draw(pc2)
    print('Delaer gave cards.')
    print('  dealer-card: ', get_card_info(dc))
    print('  player-card 1: ', get_card_info(pc1))
    print('  player-card 2: ', get_card_info(pc2))
    print('  current score: ', player.get_score())

# 現時点での手札情報（ディーラー手札は見えているもののみ）を取得
def get_current_hands():
    return copy.deepcopy(player.player_hand), copy.deepcopy(player.dealer_hand)

# HITを実行する
def hit():
    global player, soc, g_remaining_cards

    print('Action: HIT')

    # ディーラーにメッセージを送信
    player.send_message(soc, 'hit')

    # ディーラーから情報を受信
    pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True)
    print('  player-card {0}: '.format(player.get_num_player_cards()), get_card_info(pc))
    print('  current score: ', score)

    # 引いたカードを残り枚数から減算
    if g_remaining_cards is not None:
        g_remaining_cards.draw(pc)

    # バーストした場合はゲーム終了
    if status == 'bust':
        for i in range(len(dc)):
            print('  dealer-card {0}: '.format(i+2), get_card_info(dc[i]))
        # ディーラー公開カードも減算
        if g_remaining_cards is not None:
            for c in dc:
                g_remaining_cards.draw(c)
        print("  dealer's score: ", player.get_dealer_score())
        soc.close() # ディーラーとの通信をカット
        reward = player.update_money(rate=rate) # 所持金額を更新
        print('Game finished.')
        print('  result: bust')
        print('  money: ', player.get_money(), '$')
        return reward, True, status

    # バーストしなかった場合は続行
    else:
        return 0, False, status

# STANDを実行する
def stand():
    global player, soc, g_remaining_cards

    print('Action: STAND')

    # ディーラーにメッセージを送信
    player.send_message(soc, 'stand')

    # ディーラーから情報を受信
    score, status, rate, dc = player.receive_message(dsoc=soc, get_dealer_cards=True)
    print('  current score: ', score)
    for i in range(len(dc)):
        print('  dealer-card {0}: '.format(i+2), get_card_info(dc[i]))
    print("  dealer's score: ", player.get_dealer_score())

    # ディーラー公開カードを減算
    if g_remaining_cards is not None:
        for c in dc:
            g_remaining_cards.draw(c)

    # ゲーム終了，ディーラーとの通信をカット
    soc.close()

    # 所持金額を更新
    reward = player.update_money(rate=rate)
    print('Game finished.')
    print('  result: ', status)
    print('  money: ', player.get_money(), '$')
    return reward, True, status

# DOUBLE_DOWNを実行する
def double_down():
    global player, soc, g_remaining_cards

    print('Action: DOUBLE DOWN')

    # 今回のみベットを倍にする
    bet, money = player.double_bet()
    print('  money: ', money, '$')
    print('  bet: ', bet, '$')

    # ディーラーにメッセージを送信
    player.send_message(soc, 'double_down')

    # ディーラーから情報を受信
    pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True)
    print('  player-card {0}: '.format(player.get_num_player_cards()), get_card_info(pc))
    print('  current score: ', score)
    # 引いたカードを減算
    if g_remaining_cards is not None:
        g_remaining_cards.draw(pc)
    for i in range(len(dc)):
        print('  dealer-card {0}: '.format(i+2), get_card_info(dc[i]))
    print("  dealer's score: ", player.get_dealer_score())

    # ディーラー公開カードを減算
    if g_remaining_cards is not None:
        for c in dc:
            g_remaining_cards.draw(c)

    # ゲーム終了，ディーラーとの通信をカット
    soc.close()

    # 所持金額を更新
    reward = player.update_money(rate=rate)
    print('Game finished.')
    print('  result: ', status)
    print('  money: ', player.get_money(), '$')
    return reward, True, status

# SURRENDERを実行する
def surrender():
    global player, soc, g_remaining_cards

    print('Action: SURRENDER')

    # ディーラーにメッセージを送信
    player.send_message(soc, 'surrender')

    # ディーラーから情報を受信
    score, status, rate, dc = player.receive_message(dsoc=soc, get_dealer_cards=True)
    print('  current score: ', score)
    for i in range(len(dc)):
        print('  dealer-card {0}: '.format(i+2), get_card_info(dc[i]))
    print("  dealer's score: ", player.get_dealer_score())

    # ディーラー公開カードを減算
    if g_remaining_cards is not None:
        for c in dc:
            g_remaining_cards.draw(c)

    # ゲーム終了，ディーラーとの通信をカット
    soc.close()

    # 所持金額を更新
    reward = player.update_money(rate=rate)
    print('Game finished.')
    print('  result: ', status)
    print('  money: ', player.get_money(), '$')
    return reward, True, status

# RETRYを実行する
def retry():
    global player, soc, g_remaining_cards

    print('Action: RETRY')

    # ベット額の 1/4 を消費
    penalty = player.current_bet // 4
    player.consume_money(penalty)
    print('  player-card {0} has been removed.'.format(player.get_num_player_cards()))
    print('  money: ', player.get_money(), '$')

    # ディーラーにメッセージを送信
    player.send_message(soc, 'retry')

    # ディーラーから情報を受信
    pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True, retry_mode=True)
    print('  player-card {0}: '.format(player.get_num_player_cards()), get_card_info(pc))
    print('  current score: ', score)

    # 引いたカードを減算
    if g_remaining_cards is not None:
        g_remaining_cards.draw(pc)

    # バーストした場合はゲーム終了
    if status == 'bust':
        for i in range(len(dc)):
            print('  dealer-card {0}: '.format(i+2), get_card_info(dc[i]))
        print("  dealer's score: ", player.get_dealer_score())
        soc.close() # ディーラーとの通信をカット
        reward = player.update_money(rate=rate) # 所持金額を更新
        print('Game finished.')
        print('  result: bust')
        print('  money: ', player.get_money(), '$')
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

    # 「現在の状態」を設定
    # ここでは例として，プレイヤー手札のスコアとプレイヤー手札の枚数の組を「現在の状態」とする
    raw_score = p_hand.get_score() # プレイヤー手札のスコア
    score_bin = score_to_bin(raw_score)
    length = p_hand.length() # プレイヤー手札の枚数
    p_hand_is_nbj = p_hand.is_nbj()
    p_hand_is_busted = p_hand.is_busted()
    d_hand_1 = d_hand[0]
    # money をビン化
    money_bin = money_to_bin(player.get_money())

    # 残りカード数（13要素）を状態に追加
    # 参照優先度: 引数 > グローバル
    ref_cards = remaining_cards if remaining_cards is not None else g_remaining_cards
    full_counts = ref_cards.get_remaining_card_counts() if ref_cards is not None else [0]*13
    remaining_counts = remaining_counts_4cats(full_counts)

    # 状態簡略化: 個別カード(p_hand_1, p_hand_2)とディーラー手札枚数(d_hand_length)を除外
    # 新しい並び: score, length, p_hand_is_nbj, p_hand_is_busted, d_hand_1,
    #             remaining_2_6, remaining_7_9, remaining_10_K, remaining_A, money_bin
    state = (
        score_bin, length,
        p_hand_is_nbj, p_hand_is_busted,
        d_hand_1,
        *remaining_counts,
        money_bin
    )

    return state

# 行動戦略
def select_action(state, strategy: Strategy):
    global q_table

    # Q値最大行動を選択する戦略
    if strategy == Strategy.QMAX:
        return q_table.get_best_action(state)

    # ε-greedy
    elif strategy == Strategy.E_GREEDY:
        if np.random.rand() < EPS:
            return select_action(state, strategy=Strategy.RANDOM)
        else:
            return q_table.get_best_action(state)

    # ランダム戦略
    else:
        z = np.random.randint(0, 4)
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
    global g_retry_counter, player, soc, q_table

    parser = argparse.ArgumentParser(description='AI Black Jack Player (Q-learning)')
    parser.add_argument('--games', type=int, default=1000, help='num. of games to play')
    parser.add_argument('--history', type=str, default='play_log_Q.csv', help='filename where game history will be saved')
    parser.add_argument('--load', type=str, default='', help='filename of Q table to be loaded before learning')
    parser.add_argument('--save', type=str, default='', help='filename where Q table will be saved after learning')
    parser.add_argument('--testmode', help='this option runs the program without learning', action='store_true')
    args = parser.parse_args()

    n_games = args.games + 1

    # Qテーブルをロード
    if args.load != '':
        q_table.load(args.load)

    # ログファイルを開く
    logfile = open(args.history, 'w')
    # ログファイルにヘッダ行（項目名の行）を出力（残りカード数13列を追加）
    print('score_bin,hand_length,action,result,reward,'
        'p_hand_is_nbj,p_hand_is_busted,d_hand_1,'
        'remaining_2_6,remaining_7_9,remaining_10_K,remaining_A,'
        'money_bin', file=logfile)
    # n_games回ゲームを実行
    for n in tqdm(range(1, n_games)):
        # nゲーム目を開始（残りカード管理を追加）
        remaining_cards = RemainingCards(N_DECKS)
        # グローバルに設定
        global g_remaining_cards
        g_remaining_cards = remaining_cards
        game_start(n, remaining_cards=remaining_cards)

        # 「現在の状態」を取得
        state = get_state(remaining_cards=remaining_cards)

        while True:

            # 次に実行する行動を選択
            if args.testmode:
                action = select_action(state, Strategy.QMAX)
            else:
                action = select_action(state, Strategy.E_GREEDY)
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
            state = get_state(remaining_cards=remaining_cards)
            score = state[0] # 行動後のプレイヤー手札のスコア（state の一つ目の要素）

            # Qテーブルを更新
            if not args.testmode:
                _, V = q_table.get_best_action(state, with_value=True)
                Q = q_table.get_Q_value(prev_state, action) # 現在のQ値
                Q = (1 - LEARNING_RATE) * Q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * V) # 新しいQ値
                q_table.set_Q_value(prev_state, action, Q) # 新しいQ値を登録

            # ログファイルに「行動前の状態」「行動の種類」「行動結果」「獲得金額」などの情報を記録
            # 出力順: score,hand_length,action,result,reward,p_hand_1,p_hand_2,p_hand_is_nbj,p_hand_is_busted,d_hand_1,d_hand_length,remaining_A..K,money
            print('{},{},{},{},{},'  # score,hand_length,action,result,reward
                  '{},{},{},'  # p_hand_is_nbj,p_hand_is_busted,d_hand_1
                  '{},{},{},'  # remaining counts 4列 (2-6,7-9,10-K,A)
                  '{}'  # money_bin
                  .format(
                      prev_state[0], prev_state[1], action_name, status, reward,
                      prev_state[2], prev_state[3], prev_state[4],
                      prev_state[5], prev_state[6], prev_state[7], prev_state[8],
                      prev_state[9]
                  ), file=logfile)
            # 終了フラグが立った場合はnゲーム目を終了
            if done == True:
                break

        print('')

    # ログファイルを閉じる
    logfile.close()

    # Qテーブルをセーブ
    if args.save != '':
        q_table.save(args.save)


if __name__ == '__main__':
    main()
