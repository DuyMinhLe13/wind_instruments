import json
import os
import numpy as np
from extract_features import extract_features
def similar(y1, y2, index_sort1, index_sort2):
    y1 = np.array([y1])
    y1[0] = y1[0][index_sort1]
    y2 = np.array([y2])
    y2[0] = y2[0][index_sort2]
    ratio1 = y1 / y2
    ratio2 = y2 / y1
    ratio = np.concatenate((ratio1, ratio2), axis=0).min(axis=0)
    return ratio.mean()
def cbr(y, database_dir, 
        zero_crossing_rates_weight=1, avg_energies_weight=1, bandwidths_weight=1, 
        spectral_centroids_weight=1, pitch_frequencies_weight=1):
    features = extract_features(y)
    index_sort_input = np.argsort(np.array(features[4]))
    scores = dict()
    for file_name in os.listdir(database_dir):
        with open(database_dir + '/' + file_name) as json_file:
            document = json.load(json_file)
        index_sort = np.argsort(np.array(document['pitch_frequencies']))
        score = 0
        score += zero_crossing_rates_weight * similar(document['zero_crossing_rates'], features[0], index_sort, index_sort_input)
        score += avg_energies_weight * similar(document['avg_energies'], features[1], index_sort, index_sort_input)
        score += bandwidths_weight * similar(document['bandwidths'], features[2], index_sort, index_sort_input)
        score += spectral_centroids_weight * similar(document['spectral_centroids'], features[3], index_sort, index_sort_input)
        score += pitch_frequencies_weight * similar(document['pitch_frequencies'], features[4], index_sort, index_sort_input)
        score /= (zero_crossing_rates_weight + avg_energies_weight + bandwidths_weight + 
                  spectral_centroids_weight + pitch_frequencies_weight)
        scores[document['class'] + '_' + str(document['id'])] = score
    return sorted(list(scores.items()), key=lambda x: -x[1])
