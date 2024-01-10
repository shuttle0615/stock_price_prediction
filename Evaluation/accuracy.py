from Evaluation import *

def stat_computation(a, b):
    positive_actual = 0
    neutral_actual = 0
    negative_actual = 0

    positive_prediction = 0

    prediction_hit = 0
    prediction_nuetral = 0
    prediction_miss = 0

    for i, (out, lab) in enumerate(zip(a,b)):
        #actual positive reaction
        if lab == [1,0,0]:
            positive_actual += 1
        elif lab == [0,1,0]:
            neutral_actual += 1
        elif lab == [0,0,1]:
            negative_actual += 1

        #positive predicted
        if out == [1,0,0]:
            positive_prediction += 1
            if lab == [1,0,0]:
                prediction_hit += 1
            elif lab == [0,1,0]:
                prediction_nuetral += 1
            elif lab == [0,0,1]:
                prediction_miss += 1

    return positive_actual, neutral_actual, negative_actual, positive_prediction, prediction_hit, prediction_nuetral, prediction_miss