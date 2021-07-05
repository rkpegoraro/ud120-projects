#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    for a, n, p in zip(ages, net_worths, predictions) :
        #print a[0], " - ", n[0], " - ", p[0]
        cleaned_data.append((a[0], n[0], abs(p[0] - n[0])))

    cleaned_data = sorted(cleaned_data, key=lambda x: x[2])[:81]
    return cleaned_data

