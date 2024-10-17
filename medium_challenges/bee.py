from nltk.corpus import words
# from english_words import english_words_lower_set
import pandas as pd
import numpy as np
import re

# Need to only run this once
# import nltk
# nltk.download('words')
def main():
    """
    Main Function
    :return:
    """
    # Fixed letter
    fixed_letter = 'u'

    # The 6 possible letters
    possible_letters = ['r', 'n', 'y', 'l', 'a', 't']

    # word_list = words.words()
    wordlist = words.words()
    wordlist = [x.lower() for x in wordlist]

    # df = pd.DataFrame(english_words_lower_set)
    df = pd.DataFrame(wordlist)

    # Get the length of words
    mylen = np.vectorize(len)
    words_len = mylen(df[0].values)

    # Remove from the dataframe words with length less than 4
    short_ind = np.where(words_len < 4)[0]
    df.drop(df.index[short_ind], inplace=True)
    df = df.reset_index(drop=True)

    # Find the letters that will not be used in the matching
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    alphabet.remove(fixed_letter)
    for val in possible_letters:
        alphabet.remove(val)
    unused_letters = alphabet

    # Remove words from the dictionary which include the unused letters
    unused_words_ind = []
    for wordind in np.arange(len(df[0].values)):
        word = df[0].values[wordind]
        if any([characters in unused_letters for characters in word]):
            unused_words_ind.append(wordind)

    df.drop(df.index[unused_words_ind], inplace=True)
    df = df.reset_index(drop=True)

    # Lastly, remove all the words that do not contain the fixed letter
    dropind = []
    for wordind in np.arange(len(df[0].values)):
        word = df[0].values[wordind]
        if not fixed_letter in word:
            dropind.append(wordind)

    df.drop(df.index[dropind], inplace=True)
    df = df.reset_index(drop=True)

    df = df.sort_values([0])
    df = df.reset_index(drop=True)

    # We can output the name here
    # outfolder = '/home/francesco/Desktop/'
    # df.to_csv(f'{outfolder}wordlist.txt', index=False, header=None)

    # Print out the wordlist
    print(df)


    return


# Main
main()
