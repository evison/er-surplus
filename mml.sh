#!/bin/bash
# mymedialite commands


#### training

echo "BiasedMatrixFactorization"
rating_prediction --training-file="data/mml/train.csv" --save-model --test-no-ratings --test-file="data/mml/test_no_rating.csv"  --prediction-file="data/mml/test_nr_prediction.csv" --recommender="BiasedMatrixFactorization" --recommender-options num_factors=20 


echo "global average model"
rating_prediction --training-file="data/train.csv" --test-file="data/test.csv" --recommender="GlobalAverage" 