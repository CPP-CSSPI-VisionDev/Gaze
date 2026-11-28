"""Score a prediction CSV after manually filling its label column."""
import argparse
import csv
import json
from collections import Counter

STATES = ('FACING CAMERA', 'NOT FACING', 'UNKNOWN')


def score(rows):
    matrix = Counter()
    labeled = 0
    for row in rows:
        truth, predicted = row['label'].strip(), row['prediction'].strip()
        if not truth:
            continue
        if truth not in STATES or predicted not in STATES:
            raise ValueError('Labels and predictions must be FACING CAMERA, NOT FACING, or UNKNOWN.')
        matrix[truth,predicted] += 1
        labeled += 1
    tp = matrix[STATES[0],STATES[0]]
    fp = matrix[STATES[1],STATES[0]] + matrix[STATES[2],STATES[0]]
    fn = matrix[STATES[0],STATES[1]] + matrix[STATES[0],STATES[2]]
    return {'labeled_frames':labeled, 'true_positives':tp, 'false_positives':fp, 'false_negatives':fn,
            'precision':tp/(tp+fp) if tp+fp else None, 'recall':tp/(tp+fn) if tp+fn else None,
            'unknown_rate':sum(matrix[s,STATES[2]] for s in STATES)/labeled if labeled else None,
            'confusion_matrix':{s:{p:matrix[s,p] for p in STATES} for s in STATES}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv_file')
    a = p.parse_args()
    with open(a.csv_file, newline='') as f:
        result = score(csv.DictReader(f))
    if result['labeled_frames'] == 0:
        p.error('No labeled frames. Fill the label column from the original video first.')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
