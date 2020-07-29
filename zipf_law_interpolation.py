""" Given a vacabulary file, interploate the zipf \alpha """
import numpy as np
import sys

def powerLaw(y, x):
    """
    'When the frequency of an event varies as power of some attribute of that
    event the frequency is said to follow a power law.' (wikipedia)
    This is represented by the following equation, where c and alpha are
    constants:
    y = c . x ^ alpha
    Args
    --------
    y: array with frequency of events >0
    x: numpy array with attribute of events >0
    Output
    --------
    (c, alpha)
    c: the maximum frequency of any event
    alpha: defined by (Newman, 2005 for details):
        alpha = 1 + n * sum(ln( xi / xmin )) ^ -1
    """
    c = 0
    alpha = .0

    if len(y) and len(y)==len(x):
        c = max(y)
        xmin = float(min(x))
        alpha = 1 + len(x) * pow(sum(np.log(x/xmin)),-1)

    return (c, alpha)

def read_word_frequency(fname):
    freq = {}
    with open(fname, 'r') as f:
        for l in f:
            entry = l.split()
            if len(entry) != 2: # some edge cases like ' ' or just empty token.
                continue
            freq[entry[0]] = int(entry[1])
    return freq

if __name__=="__main__":
    # vocab.txt
    freq = read_word_frequency(sys.argv[1])
    y = sorted(freq.values(),reverse=True)
    x = np.array(range(1,len(y)+1))
    c, alpha = powerLaw(y, x)
    print(c, alpha)

    """
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(rank[30:-100],freq[30:-100])
    print(slope, intercept, std_err)
    """
