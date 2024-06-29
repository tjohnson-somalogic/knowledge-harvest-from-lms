from nltk.corpus import stopwords

stopword_list = stopwords.words('english')
stopword_list.extend(
    [
        'everything',
        'everybody',
        'everyone',
        'anything',
        'anybody',
        'anyone',
        'something',
        'somebody',
        'someone',
        'nothing',
        'nobody',
        'one',
        'neither',
        'either',
        'many',
        'us',
        'first',
        'second',
        'next',
        'following',
        'last',
        'new',
        'main',
        'also',
    ]
)
