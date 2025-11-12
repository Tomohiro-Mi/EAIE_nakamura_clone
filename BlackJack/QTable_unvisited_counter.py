import argparse
from classes import Action, QTable
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='Count unvisited Q-values (within observed states) in a saved Q-table')
    parser.add_argument('--file', type=str, required=True, help='Path to saved Q-table file (pickle)')
    parser.add_argument('--detail', action='store_true', help='Print per-state missing action counts')
    args = parser.parse_args()

    # Load table
    q_table = QTable(action_class=Action)
    q_table.load(args.file)

    # Collect visited (state, action)
    entries = q_table.table
    actions = [a for a in Action if a != Action.UNDEFINED]

    # Derive set of observed states
    states = set(state for (state, _action) in entries.keys())

    # Count visited per state
    visited_pairs = len(entries)
    total_pairs_in_observed = len(states) * len(actions)

    missing_total = 0
    if args.detail:
        print('state,missing_actions_count')

    for s in tqdm(states):
        present = {a for (_s, a) in entries.keys() if _s == s}
        missing = [a for a in actions if a not in present]
        cnt = len(missing)
        missing_total += cnt
        if args.detail:
            print(f'{s},{cnt}')

    coverage = 0.0
    if total_pairs_in_observed > 0:
        coverage = visited_pairs / total_pairs_in_observed

    print('--- Summary (within observed states) ---')
    print(f'Unique states observed     : {len(states)}')
    print(f'Num actions per state      : {len(actions)}')
    print(f'Visited state-action pairs : {visited_pairs}')
    print(f'Unvisited pairs (observed) : {missing_total}')
    print(f'Coverage ratio             : {coverage:.4f}')
    print('Note: This counts only within states that appear in the table.\n'
          '      The true number of unvisited pairs in the full state space is much larger.')


if __name__ == '__main__':
    main()
