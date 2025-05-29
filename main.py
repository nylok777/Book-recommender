from book import Books, Ratings

books = Books("Book reviews/BX-Books.csv")
ratings = Ratings("Book reviews/BX-Book-Ratings.csv")

recs = books.recommend_svd(500, ratings)
for r in recs:
    print(books.books[books.books['ISBN'] == r[1]]['Book-Title'].values, r[3])