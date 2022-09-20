from CountVectorizer import CountVectorizer


def main():
    corpus = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste'
    ]

    expected_matrix = [
        [1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    ]

    expected_feature_names = [
        'crock', 'pot', 'pasta', 'never', 'boil', 'again',
        'pomodoro', 'fresh', 'ingredients', 'parmesan', 'to', 'taste'
    ]

    cv = CountVectorizer()

    assert cv.fit_transform(corpus) == expected_matrix
    assert cv.get_feature_names() == expected_feature_names


if __name__ == '__main__':
    main()
