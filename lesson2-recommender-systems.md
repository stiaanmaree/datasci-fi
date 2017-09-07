Lesson 2 // Recommender systems
================

In this lesson we'll:

1.  introduce recommender systems based on collaborative filtering
2.  build recommender systems based on various kinds of collaborative filtering
    -   user-based collaborative filtering
    -   item-based collaborative filtering
    -   matrix factorization

3.  introduce L2 regularization and bias terms, two ways of improving recommender systems based on matrix factorization.
4.  use these approaches to build a system for recommending movies to users based on their past viewing habits.

This notebook is based quite closely on the following sources:

-   Chapter 22 of Joel Grus' ["Data Science from Scratch: First Principles with Python"](http://shop.oreilly.com/product/0636920033400.do). The (Python) code from the book is [here](https://github.com/joelgrus/data-science-from-scratch).
-   Part of [Lesson 4](http://course.fast.ai/lessons/lesson4.html) of the fast.ai course "Practical Deep Learning for Coders". There's a timeline of the lesson [here](http://wiki.fast.ai/index.php/Lesson_4_Timeline). Code (also in Python) is [here](https://github.com/fastai/courses/tree/master/deeplearning1).

### Load required packages and the dataset we created last lesson

``` r
library(tidyverse)
```

    ## Loading tidyverse: ggplot2
    ## Loading tidyverse: tibble
    ## Loading tidyverse: tidyr
    ## Loading tidyverse: readr
    ## Loading tidyverse: purrr
    ## Loading tidyverse: dplyr

    ## Conflicts with tidy packages ----------------------------------------------

    ## filter(): dplyr, stats
    ## lag():    dplyr, stats

``` r
load("output/recommender.RData")
```

We need to convert the data to a matrix form or else some the later functions we use will give an error (see what happens if you don't make the change)

``` r
sorted_my_users <- as.character(unlist(viewed_movies[,1]))
viewed_movies <- as.matrix(viewed_movies[,-1])
row.names(viewed_movies) <- sorted_my_users
```

User-based collaborative filtering
----------------------------------

### The basic idea behind user-based collaborative filtering

A really simple recommender system would just recommend the most popular movies (that a user hasn't seen before). This information is obtained by summing the values of each column of *viewed movies*

``` r
sort(apply(viewed_movies,2,sum),decreasing = T)
```

    ##                                Beautiful Mind, A (2001) 
    ##                                                       9 
    ##                                     American Pie (1999) 
    ##                                                       8 
    ##                                       Armageddon (1998) 
    ##                                                       8 
    ##                                Kill Bill: Vol. 1 (2003) 
    ##                                                       8 
    ##                                        Inception (2010) 
    ##                                                       7 
    ##                                Wizard of Oz, The (1939) 
    ##                                                       7 
    ##                                   Apocalypse Now (1979) 
    ##                                                       6 
    ##                              Breakfast Club, The (1985) 
    ##                                                       6 
    ##                               Fifth Element, The (1997) 
    ##                                                       6 
    ##                                  Minority Report (2002) 
    ##                                                       6 
    ##                                         Rain Man (1988) 
    ##                                                       6 
    ## Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) 
    ##                                                       5 
    ##            Austin Powers: The Spy Who Shagged Me (1999) 
    ##                                                       4 
    ##                                      Stand by Me (1986) 
    ##                                                       4 
    ##                                         Outbreak (1995) 
    ##                                                       3 
    ##                           Star Trek: Generations (1994) 
    ##                                                       3 
    ##                                      Taxi Driver (1976) 
    ##                                                       3 
    ##                                       Waterworld (1995) 
    ##                                                       3 
    ##                                       Casablanca (1942) 
    ##                                                       2 
    ##                         Clear and Present Danger (1994) 
    ##                                                       2

This approach has an intuitive appeal but is pretty unsophisticated (everyone gets the same recommendations, barring the filtering out of seen movies!) In other words, everyone's vote counts the same.

User-based CF extends the approach by changing how much each person's vote counts. Specifically, when recommending what I should watch next, a user-based CF system will upweight the votes of people that are "more similar" to me. In this context "similar" means "has seen many of the same movies as me". You can think of this as replacing the 1's in the *viewed\_movies* matrix will a number that increases with similarity to the user we're trying to recommend a movie to.

There are lots of different similarity measures. The one we'll use is called cosine similarity and is widely used, but search online for others and try them out.

``` r
# function calculating cosine similarity
cosine_sim <- function(a,b){crossprod(a,b)/sqrt(crossprod(a)*crossprod(b))}
```

Cosine similarity lies between 0 and 1 inclusive and increases with similarity. Here are a few test cases to get a feel for it:

``` r
# maximally similar
x1 <- c(1,1,1,0,0)
x2 <- c(1,1,1,0,0)
cosine_sim(x1,x2)
```

    ##      [,1]
    ## [1,]    1

``` r
# maximally dissimilar
x1 <- c(1,1,1,0,0)
x2 <- c(0,0,0,1,1)
cosine_sim(x1,x2)
```

    ##      [,1]
    ## [1,]    0

``` r
# but also
x1 <- c(1,1,0,0,0)
x2 <- c(0,0,0,1,1)
cosine_sim(x1,x2)
```

    ##      [,1]
    ## [1,]    0

``` r
# try an example from our data
as.numeric(viewed_movies[1,]) # user 1's viewing history
```

    ##  [1] 1 0 0 0 0 1 0 0 0 1 1 0 0 0 0 1 0 0 0 0

``` r
as.numeric(viewed_movies[2,]) # user 2's viewing history
```

    ##  [1] 0 0 1 0 0 1 0 1 0 1 0 0 0 1 0 1 1 1 0 0

``` r
cosine_sim(viewed_movies[1,], viewed_movies[2,])
```

    ##           [,1]
    ## [1,] 0.4743416

Let's get similarities between user pairs. We'll do this with a loop below, because its easier to see what's going, but this will be inefficient and very slow for bigger datasets. As an exercise, see if you can do the same without loops.

``` r
user_similarities = matrix(0, nrow = 15, ncol = 15)
for(i in 1:14){
  for(j in (i+1):15){
    user_similarities[i,j] <- cosine_sim(viewed_movies[i,], viewed_movies[j,])
  }
}
user_similarities <- user_similarities + t(user_similarities)
diag(user_similarities) <- 0
row.names(user_similarities) <- row.names(viewed_movies)
colnames(user_similarities) <- row.names(viewed_movies)
```

``` r
# who are the most similar users to user 149?
user_similarities["149",]
```

    ##       149       177       200       236       240       270       287 
    ## 0.0000000 0.4743416 0.5477226 0.0000000 0.0000000 0.2000000 0.1825742 
    ##       295       303       408       426       442       500       522 
    ## 0.1414214 0.7453560 0.4743416 0.3651484 0.2696799 0.3651484 0.3380617 
    ##       562 
    ## 0.1490712

Let's see if this makes sense from the viewing histories. Below we show user 149's history, together with the user who is most similar to user 149 (user 303) and another user who is very dissimilar (user 236).

``` r
viewed_movies[c("149","303","236"),]
```

    ##     American Pie (1999) Apocalypse Now (1979) Armageddon (1998)
    ## 149                   1                     0                 0
    ## 303                   1                     0                 0
    ## 236                   0                     1                 0
    ##     Austin Powers: The Spy Who Shagged Me (1999) Beautiful Mind, A (2001)
    ## 149                                            0                        0
    ## 303                                            0                        1
    ## 236                                            0                        0
    ##     Breakfast Club, The (1985) Casablanca (1942)
    ## 149                          1                 0
    ## 303                          1                 0
    ## 236                          0                 1
    ##     Clear and Present Danger (1994)
    ## 149                               0
    ## 303                               0
    ## 236                               0
    ##     Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000)
    ## 149                                                       0
    ## 303                                                       0
    ## 236                                                       1
    ##     Fifth Element, The (1997) Inception (2010) Kill Bill: Vol. 1 (2003)
    ## 149                         1                1                        0
    ## 303                         1                1                        1
    ## 236                         0                0                        0
    ##     Minority Report (2002) Outbreak (1995) Rain Man (1988)
    ## 149                      0               0               0
    ## 303                      0               0               1
    ## 236                      0               0               0
    ##     Stand by Me (1986) Star Trek: Generations (1994) Taxi Driver (1976)
    ## 149                  1                             0                  0
    ## 303                  1                             0                  0
    ## 236                  0                             0                  1
    ##     Waterworld (1995) Wizard of Oz, The (1939)
    ## 149                 0                        0
    ## 303                 0                        1
    ## 236                 0                        0

### Recommending movies for a single user

As an example, let's consider the process of recommending a movie to one user, say user 149. How would we do this with a user-based collaborative filtering system?

First, we need to know what movies have they already seen (so we don't recommend these).

``` r
viewed_movies["149",]
```

    ##                                     American Pie (1999) 
    ##                                                       1 
    ##                                   Apocalypse Now (1979) 
    ##                                                       0 
    ##                                       Armageddon (1998) 
    ##                                                       0 
    ##            Austin Powers: The Spy Who Shagged Me (1999) 
    ##                                                       0 
    ##                                Beautiful Mind, A (2001) 
    ##                                                       0 
    ##                              Breakfast Club, The (1985) 
    ##                                                       1 
    ##                                       Casablanca (1942) 
    ##                                                       0 
    ##                         Clear and Present Danger (1994) 
    ##                                                       0 
    ## Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) 
    ##                                                       0 
    ##                               Fifth Element, The (1997) 
    ##                                                       1 
    ##                                        Inception (2010) 
    ##                                                       1 
    ##                                Kill Bill: Vol. 1 (2003) 
    ##                                                       0 
    ##                                  Minority Report (2002) 
    ##                                                       0 
    ##                                         Outbreak (1995) 
    ##                                                       0 
    ##                                         Rain Man (1988) 
    ##                                                       0 
    ##                                      Stand by Me (1986) 
    ##                                                       1 
    ##                           Star Trek: Generations (1994) 
    ##                                                       0 
    ##                                      Taxi Driver (1976) 
    ##                                                       0 
    ##                                       Waterworld (1995) 
    ##                                                       0 
    ##                                Wizard of Oz, The (1939) 
    ##                                                       0

The basic idea is now to recommend what's popular by adding up the number of users that have seen each movie, but *to weight each user by their similarity to user 149*.

Let's work through the calculations for one movie, say Apocalypse Now (movie 2). The table below shows who's seen Apocalypse Now, and how similar each person is to user 149.

``` r
seen_movie <- viewed_movies[,"Apocalypse Now (1979)"]
sim_to_user <- user_similarities["149",]
cbind(seen_movie,sim_to_user)
```

    ##     seen_movie sim_to_user
    ## 149          0   0.0000000
    ## 177          0   0.4743416
    ## 200          0   0.5477226
    ## 236          1   0.0000000
    ## 240          0   0.0000000
    ## 270          1   0.2000000
    ## 287          0   0.1825742
    ## 295          0   0.1414214
    ## 303          0   0.7453560
    ## 408          1   0.4743416
    ## 426          0   0.3651484
    ## 442          1   0.2696799
    ## 500          0   0.3651484
    ## 522          1   0.3380617
    ## 562          1   0.1490712

The basic idea in user-based collaborative filtering is that user 236's vote counts less than user 408's, because user 408 is more similar to user 149 (in terms of viewing history).

Note that this only means user 236 counts more in the context of making recommendations to user 149. When recommending to users *other than user 149*, user 408 may carry more weight.

We can now work out an overall recommendation score for Apocalypse Now by multiplying together the two elements in each row of the table above, and summing these products (taking the dot product):

``` r
# overall score for Apocalypse now
crossprod(viewed_movies[,"Apocalypse Now (1979)"],user_similarities["149",])
```

    ##          [,1]
    ## [1,] 1.431154

Note this score will increase with (a) the number of people who've seen the movie (more 1's in the first column above) and (b) if the people who've seen it are similar to user 1

Let's repeat this calculation for all movies and compare recommendation scores:

``` r
user_similarities["149",] %*% viewed_movies
```

    ##      American Pie (1999) Apocalypse Now (1979) Armageddon (1998)
    ## [1,]            2.780188              1.431154          2.800941
    ##      Austin Powers: The Spy Who Shagged Me (1999) Beautiful Mind, A (2001)
    ## [1,]                                     1.258241                 2.972538
    ##      Breakfast Club, The (1985) Casablanca (1942)
    ## [1,]                   2.328868         0.1414214
    ##      Clear and Present Danger (1994)
    ## [1,]                       0.7440216
    ##      Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000)
    ## [1,]                                               0.4730667
    ##      Fifth Element, The (1997) Inception (2010) Kill Bill: Vol. 1 (2003)
    ## [1,]                  2.383183         2.378863                 2.208739
    ##      Minority Report (2002) Outbreak (1995) Rain Man (1988)
    ## [1,]               1.220789       0.7648342        2.334009
    ##      Stand by Me (1986) Star Trek: Generations (1994) Taxi Driver (1976)
    ## [1,]           1.694039                      1.022064          0.4743416
    ##      Waterworld (1995) Wizard of Oz, The (1939)
    ## [1,]         0.5601725                 2.178522

To come up with a final recommendation, we just need to remember to remove movies user 149 has already seen, and sort the remaining movies in descending order of recommendation score.

We do that below, after tidying up the results a bit by putting them in a data frame.

``` r
user_scores <- data.frame(title = colnames(viewed_movies), 
                          score = as.vector(user_similarities["149",] %*% viewed_movies), 
                          seen = viewed_movies["149",])
user_scores %>% filter(seen == 0) %>% arrange(desc(score)) 
```

    ##                                                      title     score seen
    ## 1                                 Beautiful Mind, A (2001) 2.9725383    0
    ## 2                                        Armageddon (1998) 2.8009413    0
    ## 3                                          Rain Man (1988) 2.3340090    0
    ## 4                                 Kill Bill: Vol. 1 (2003) 2.2087386    0
    ## 5                                 Wizard of Oz, The (1939) 2.1785215    0
    ## 6                                    Apocalypse Now (1979) 1.4311545    0
    ## 7             Austin Powers: The Spy Who Shagged Me (1999) 1.2582412    0
    ## 8                                   Minority Report (2002) 1.2207893    0
    ## 9                            Star Trek: Generations (1994) 1.0220642    0
    ## 10                                         Outbreak (1995) 0.7648342    0
    ## 11                         Clear and Present Danger (1994) 0.7440216    0
    ## 12                                       Waterworld (1995) 0.5601725    0
    ## 13                                      Taxi Driver (1976) 0.4743416    0
    ## 14 Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) 0.4730667    0
    ## 15                                       Casablanca (1942) 0.1414214    0

Now that we've understood the calculations, let's get recommendations for one more user, 236:

``` r
# recommendations for user 236
user_scores <- data.frame(title = colnames(viewed_movies), 
                          score = as.vector(user_similarities["236",] %*% viewed_movies), 
                          seen = viewed_movies["236",])
user_scores %>% filter(seen == 0) %>% arrange(desc(score)) 
```

    ##                                           title     score seen
    ## 1                      Kill Bill: Vol. 1 (2003) 1.6211541    0
    ## 2                        Minority Report (2002) 1.4855403    0
    ## 3                      Beautiful Mind, A (2001) 1.2878208    0
    ## 4                      Wizard of Oz, The (1939) 1.2561326    0
    ## 5                             Armageddon (1998) 1.2307488    0
    ## 6                               Rain Man (1988) 0.8327424    0
    ## 7                               Outbreak (1995) 0.8263378    0
    ## 8                             Waterworld (1995) 0.8003168    0
    ## 9                           American Pie (1999) 0.6730712    0
    ## 10                    Fifth Element, The (1997) 0.6697812    0
    ## 11 Austin Powers: The Spy Who Shagged Me (1999) 0.6608657    0
    ## 12                             Inception (2010) 0.6167132    0
    ## 13                   Breakfast Club, The (1985) 0.5043091    0
    ## 14                Star Trek: Generations (1994) 0.3809008    0
    ## 15                           Stand by Me (1986) 0.3535534    0
    ## 16              Clear and Present Danger (1994) 0.3275324    0

### A simple function to generate a user-based CF recommendation for any user

``` r
# a function to generate a recommendation for any user
user_based_recommendations <- function(user, user_similarities, viewed_movies){
  
  # turn into character if not already
  user <- ifelse(is.character(user),user,as.character(user))
  
  # get scores
  user_scores <- data.frame(title = colnames(viewed_movies), 
                            score = as.vector(user_similarities[user,] %*% viewed_movies), 
                            seen = viewed_movies[user,])
  
  # sort unseen movies by score and remove the 'seen' column
  user_recom <- user_scores %>% 
    filter(seen == 0) %>% 
    arrange(desc(score)) %>% 
    select(-seen) 
  
  return(user_recom)
  
}
```

Let's check the function is working by running it on a user we've used before:

``` r
user_based_recommendations(user = 149, user_similarities = user_similarities, viewed_movies = viewed_movies)
```

    ##                                                      title     score
    ## 1                                 Beautiful Mind, A (2001) 2.9725383
    ## 2                                        Armageddon (1998) 2.8009413
    ## 3                                          Rain Man (1988) 2.3340090
    ## 4                                 Kill Bill: Vol. 1 (2003) 2.2087386
    ## 5                                 Wizard of Oz, The (1939) 2.1785215
    ## 6                                    Apocalypse Now (1979) 1.4311545
    ## 7             Austin Powers: The Spy Who Shagged Me (1999) 1.2582412
    ## 8                                   Minority Report (2002) 1.2207893
    ## 9                            Star Trek: Generations (1994) 1.0220642
    ## 10                                         Outbreak (1995) 0.7648342
    ## 11                         Clear and Present Danger (1994) 0.7440216
    ## 12                                       Waterworld (1995) 0.5601725
    ## 13                                      Taxi Driver (1976) 0.4743416
    ## 14 Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) 0.4730667
    ## 15                                       Casablanca (1942) 0.1414214

Now do it for all users with `lapply`

``` r
lapply(sorted_my_users, user_based_recommendations, user_similarities, viewed_movies)
```

    ## [[1]]
    ##                                                      title     score
    ## 1                                 Beautiful Mind, A (2001) 2.9725383
    ## 2                                        Armageddon (1998) 2.8009413
    ## 3                                          Rain Man (1988) 2.3340090
    ## 4                                 Kill Bill: Vol. 1 (2003) 2.2087386
    ## 5                                 Wizard of Oz, The (1939) 2.1785215
    ## 6                                    Apocalypse Now (1979) 1.4311545
    ## 7             Austin Powers: The Spy Who Shagged Me (1999) 1.2582412
    ## 8                                   Minority Report (2002) 1.2207893
    ## 9                            Star Trek: Generations (1994) 1.0220642
    ## 10                                         Outbreak (1995) 0.7648342
    ## 11                         Clear and Present Danger (1994) 0.7440216
    ## 12                                       Waterworld (1995) 0.5601725
    ## 13                                      Taxi Driver (1976) 0.4743416
    ## 14 Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) 0.4730667
    ## 15                                       Casablanca (1942) 0.1414214
    ## 
    ## [[2]]
    ##                                                      title     score
    ## 1                                      American Pie (1999) 2.2387168
    ## 2                                 Wizard of Oz, The (1939) 2.1186491
    ## 3                                 Beautiful Mind, A (2001) 1.8966173
    ## 4                                         Inception (2010) 1.6832135
    ## 5                                          Rain Man (1988) 1.6749295
    ## 6                                 Kill Bill: Vol. 1 (2003) 1.5549693
    ## 7                                    Apocalypse Now (1979) 1.3659107
    ## 8             Austin Powers: The Spy Who Shagged Me (1999) 1.3441785
    ## 9                                   Minority Report (2002) 1.1809969
    ## 10 Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) 1.0690985
    ## 11                                       Waterworld (1995) 0.7791101
    ## 12                                       Casablanca (1942) 0.4003835
    ## 
    ## [[3]]
    ##                                                      title     score
    ## 1                                 Kill Bill: Vol. 1 (2003) 3.8740881
    ## 2                                          Rain Man (1988) 2.8734591
    ## 3                                 Wizard of Oz, The (1939) 2.7562457
    ## 4                               Breakfast Club, The (1985) 2.3720117
    ## 5                                    Apocalypse Now (1979) 2.2311339
    ## 6                                       Stand by Me (1986) 1.6694039
    ## 7  Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) 1.6288800
    ## 8             Austin Powers: The Spy Who Shagged Me (1999) 1.3995312
    ## 9                                        Waterworld (1995) 1.1648211
    ## 10                           Star Trek: Generations (1994) 1.1220085
    ## 11                                         Outbreak (1995) 1.0842218
    ## 12                         Clear and Present Danger (1994) 0.6579496
    ## 13                                      Taxi Driver (1976) 0.6220085
    ## 14                                       Casablanca (1942) 0.3872983
    ## 
    ## [[4]]
    ##                                           title     score
    ## 1                      Kill Bill: Vol. 1 (2003) 1.6211541
    ## 2                        Minority Report (2002) 1.4855403
    ## 3                      Beautiful Mind, A (2001) 1.2878208
    ## 4                      Wizard of Oz, The (1939) 1.2561326
    ## 5                             Armageddon (1998) 1.2307488
    ## 6                               Rain Man (1988) 0.8327424
    ## 7                               Outbreak (1995) 0.8263378
    ## 8                             Waterworld (1995) 0.8003168
    ## 9                           American Pie (1999) 0.6730712
    ## 10                    Fifth Element, The (1997) 0.6697812
    ## 11 Austin Powers: The Spy Who Shagged Me (1999) 0.6608657
    ## 12                             Inception (2010) 0.6167132
    ## 13                   Breakfast Club, The (1985) 0.5043091
    ## 14                Star Trek: Generations (1994) 0.3809008
    ## 15                           Stand by Me (1986) 0.3535534
    ## 16              Clear and Present Danger (1994) 0.3275324
    ## 
    ## [[5]]
    ##                                           title     score
    ## 1                             Armageddon (1998) 2.5414713
    ## 2                           American Pie (1999) 2.4943778
    ## 3                              Inception (2010) 2.4312442
    ## 4                               Rain Man (1988) 2.2092976
    ## 5                         Apocalypse Now (1979) 2.1864379
    ## 6                     Fifth Element, The (1997) 1.6757540
    ## 7                             Waterworld (1995) 1.4230200
    ## 8                    Breakfast Club, The (1985) 1.3995312
    ## 9  Austin Powers: The Spy Who Shagged Me (1999) 1.2551937
    ## 10                              Outbreak (1995) 1.1980831
    ## 11                            Casablanca (1942) 1.0537455
    ## 12                Star Trek: Generations (1994) 0.9776709
    ## 13                           Stand by Me (1986) 0.6969234
    ## 14              Clear and Present Danger (1994) 0.5136120
    ## 
    ## [[6]]
    ##                                                      title     score
    ## 1                                      American Pie (1999) 3.4530898
    ## 2                                        Armageddon (1998) 3.1465643
    ## 3                                          Rain Man (1988) 2.6579574
    ## 4                                 Wizard of Oz, The (1939) 2.5295566
    ## 5  Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) 2.0079554
    ## 6                                Fifth Element, The (1997) 1.7773141
    ## 7                               Breakfast Club, The (1985) 1.3924216
    ## 8                                        Waterworld (1995) 1.2759976
    ## 9             Austin Powers: The Spy Who Shagged Me (1999) 1.1924216
    ## 10                                         Outbreak (1995) 0.8714777
    ## 11                                      Stand by Me (1986) 0.8053275
    ## 12                                      Taxi Driver (1976) 0.7713294
    ## 13                                       Casablanca (1942) 0.6478709
    ## 14                           Star Trek: Generations (1994) 0.5477226
    ## 15                         Clear and Present Danger (1994) 0.4045199
    ## 
    ## [[7]]
    ##                                           title     score
    ## 1                      Beautiful Mind, A (2001) 3.2460686
    ## 2                      Kill Bill: Vol. 1 (2003) 2.8209835
    ## 3                           American Pie (1999) 2.5844444
    ## 4                     Fifth Element, The (1997) 1.9193883
    ## 5                               Rain Man (1988) 1.8362620
    ## 6                         Apocalypse Now (1979) 1.8209856
    ## 7                    Breakfast Club, The (1985) 1.6116063
    ## 8  Austin Powers: The Spy Who Shagged Me (1999) 1.2764397
    ## 9                               Outbreak (1995) 1.0842218
    ## 10                            Waterworld (1995) 1.0417296
    ## 11                           Stand by Me (1986) 1.0320900
    ## 12                           Taxi Driver (1976) 0.9927993
    ## 13                            Casablanca (1942) 0.5914225
    ## 14              Clear and Present Danger (1994) 0.5348581
    ## 
    ## [[8]]
    ##                                           title     score
    ## 1                             Armageddon (1998) 3.0832582
    ## 2                           American Pie (1999) 3.0636093
    ## 3                              Inception (2010) 2.6131953
    ## 4                         Apocalypse Now (1979) 2.4382482
    ## 5                    Breakfast Club, The (1985) 1.9624148
    ## 6  Austin Powers: The Spy Who Shagged Me (1999) 1.5973867
    ## 7                            Stand by Me (1986) 1.2274846
    ## 8                            Taxi Driver (1976) 1.1853318
    ## 9                 Star Trek: Generations (1994) 0.8691040
    ## 10              Clear and Present Danger (1994) 0.7003381
    ## 
    ## [[9]]
    ##                                                      title     score
    ## 1                                        Armageddon (1998) 3.8949051
    ## 2                                    Apocalypse Now (1979) 2.4916549
    ## 3                                   Minority Report (2002) 2.4212270
    ## 4             Austin Powers: The Spy Who Shagged Me (1999) 1.9588316
    ## 5  Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) 1.4296823
    ## 6                                        Waterworld (1995) 1.3522912
    ## 7                            Star Trek: Generations (1994) 1.1700500
    ## 8                                          Outbreak (1995) 1.1028219
    ## 9                          Clear and Present Danger (1994) 0.9565761
    ## 10                                      Taxi Driver (1976) 0.7618017
    ## 11                                       Casablanca (1942) 0.5270463
    ## 
    ## [[10]]
    ##                                                      title     score
    ## 1                                      American Pie (1999) 3.4680077
    ## 2                                 Beautiful Mind, A (2001) 3.2779743
    ## 3                                 Kill Bill: Vol. 1 (2003) 2.9098398
    ## 4                                         Inception (2010) 2.4886284
    ## 5                                   Minority Report (2002) 1.5687653
    ## 6                                        Waterworld (1995) 1.3285657
    ## 7  Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) 1.2987530
    ## 8                            Star Trek: Generations (1994) 1.2216878
    ## 9                                          Outbreak (1995) 1.1889636
    ## 10                         Clear and Present Danger (1994) 1.1396021
    ## 11                                      Taxi Driver (1976) 0.8211143
    ## 12                                       Casablanca (1942) 0.5121869
    ## 
    ## [[11]]
    ##                                                      title     score
    ## 1                                 Wizard of Oz, The (1939) 2.9718447
    ## 2                                    Apocalypse Now (1979) 2.7859235
    ## 3                                   Minority Report (2002) 2.6766025
    ## 4                                Fifth Element, The (1997) 2.5325399
    ## 5                               Breakfast Club, The (1985) 2.4273657
    ## 6             Austin Powers: The Spy Who Shagged Me (1999) 1.6457142
    ## 7                                       Stand by Me (1986) 1.4785749
    ## 8  Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) 1.4622133
    ## 9                                        Waterworld (1995) 1.4110041
    ## 10                                         Outbreak (1995) 0.9398842
    ## 11                           Star Trek: Generations (1994) 0.8110042
    ## 12                         Clear and Present Danger (1994) 0.7597950
    ## 13                                      Taxi Driver (1976) 0.4776709
    ## 14                                       Casablanca (1942) 0.3872983
    ## 
    ## [[12]]
    ##                                                     title     score
    ## 1                                        Inception (2010) 3.1919009
    ## 2                               Fifth Element, The (1997) 2.6781116
    ## 3                                  Minority Report (2002) 2.4690058
    ## 4 Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) 1.8459671
    ## 5                                      Stand by Me (1986) 1.8321059
    ## 6                                         Outbreak (1995) 1.3995551
    ## 7                           Star Trek: Generations (1994) 1.1814415
    ## 8                                      Taxi Driver (1976) 0.8398312
    ## 9                                       Casablanca (1942) 0.6274870
    ## 
    ## [[13]]
    ##                                                      title     score
    ## 1                                        Armageddon (1998) 2.9179175
    ## 2                                 Kill Bill: Vol. 1 (2003) 2.8480005
    ## 3                                          Rain Man (1988) 2.4929401
    ## 4                                         Inception (2010) 2.4006603
    ## 5                                Fifth Element, The (1997) 2.2226995
    ## 6                                    Apocalypse Now (1979) 1.8118166
    ## 7                                   Minority Report (2002) 1.7129386
    ## 8                                       Stand by Me (1986) 1.6311673
    ## 9  Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) 1.1970311
    ## 10                                       Waterworld (1995) 1.1458219
    ## 11                         Clear and Present Danger (1994) 0.9041326
    ## 12                                         Outbreak (1995) 0.8190396
    ## 13                                      Taxi Driver (1976) 0.6220085
    ## 14                                       Casablanca (1942) 0.2581989
    ## 
    ## [[14]]
    ##                                                      title     score
    ## 1                                 Wizard of Oz, The (1939) 2.9989848
    ## 2                                   Minority Report (2002) 2.7730714
    ## 3                               Breakfast Club, The (1985) 2.4948951
    ## 4                                Fifth Element, The (1997) 2.4783070
    ## 5             Austin Powers: The Spy Who Shagged Me (1999) 1.8972147
    ## 6  Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) 1.6687168
    ## 7                                        Waterworld (1995) 1.5462847
    ## 8                                       Stand by Me (1986) 1.5025250
    ## 9                                          Outbreak (1995) 0.9961518
    ## 10                         Clear and Present Danger (1994) 0.8173941
    ## 11                           Star Trek: Generations (1994) 0.7508440
    ## 12                                      Taxi Driver (1976) 0.6312196
    ## 13                                       Casablanca (1942) 0.5475508
    ## 
    ## [[15]]
    ##                              title     score
    ## 1         Beautiful Mind, A (2001) 3.8003678
    ## 2         Wizard of Oz, The (1939) 2.7945067
    ## 3                  Rain Man (1988) 2.6180455
    ## 4                 Inception (2010) 2.5472045
    ## 5        Fifth Element, The (1997) 1.8958436
    ## 6       Breakfast Club, The (1985) 1.8357373
    ## 7               Taxi Driver (1976) 0.9772839
    ## 8               Stand by Me (1986) 0.9605491
    ## 9    Star Trek: Generations (1994) 0.9161161
    ## 10               Casablanca (1942) 0.8603796
    ## 11 Clear and Present Danger (1994) 0.8387249

A variant on the above is a *k-nearest-neighbours* approach that bases recommendations *only on k most similar users*. This is faster when there are many users. Try implement this as an exercise.

Item-based collaborative filtering
----------------------------------

### The basic idea behind item-based collaborative filtering

Item-based collaborative filtering works very similarly to user-based counterpart, but is a tiny bit less intuitive (in my opinion). It is also based on similarities, but similarities between *movies* rather than *users*.

There are two main conceptual parts to item-based collaborative filtering:

1.  One movie is similar to another if many of the same users have seen both movies.
2.  When deciding what movie to recommend to a particular user, movies are evaluated on how similar they are to movies *that the user has already seen*.

Let's start by computing the similarities between all pairs of movies. We can reuse the same code we used to compute user similarities, if we first transpose the *viewed\_movies* matrix.

``` r
# transpose the viewed_movies matrix
movies_user <- t(viewed_movies)

# get all similarities between MOVIES
movie_similarities = matrix(0, nrow = 20, ncol = 20)
for(i in 1:19){
  for(j in (i+1):20){
    movie_similarities[i,j] <- cosine_sim(viewed_movies[,i], viewed_movies[,j])
  }
}
movie_similarities <- movie_similarities + t(movie_similarities)
diag(movie_similarities) <- 0
row.names(movie_similarities) <- colnames(viewed_movies)
colnames(movie_similarities) <- colnames(viewed_movies)
```

We can use the result to see, for example, what movies are most similar to "Apocalypse Now":

``` r
movie_similarities[,"Apocalypse Now (1979)"]
```

    ##                                     American Pie (1999) 
    ##                                               0.4330127 
    ##                                   Apocalypse Now (1979) 
    ##                                               0.0000000 
    ##                                       Armageddon (1998) 
    ##                                               0.5773503 
    ##            Austin Powers: The Spy Who Shagged Me (1999) 
    ##                                               0.6123724 
    ##                                Beautiful Mind, A (2001) 
    ##                                               0.4082483 
    ##                              Breakfast Club, The (1985) 
    ##                                               0.3333333 
    ##                                       Casablanca (1942) 
    ##                                               0.2886751 
    ##                         Clear and Present Danger (1994) 
    ##                                               0.2886751 
    ## Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) 
    ##                                               0.3651484 
    ##                               Fifth Element, The (1997) 
    ##                                               0.1666667 
    ##                                        Inception (2010) 
    ##                                               0.3086067 
    ##                                Kill Bill: Vol. 1 (2003) 
    ##                                               0.5773503 
    ##                                  Minority Report (2002) 
    ##                                               0.3333333 
    ##                                         Outbreak (1995) 
    ##                                               0.2357023 
    ##                                         Rain Man (1988) 
    ##                                               0.5000000 
    ##                                      Stand by Me (1986) 
    ##                                               0.2041241 
    ##                           Star Trek: Generations (1994) 
    ##                                               0.0000000 
    ##                                      Taxi Driver (1976) 
    ##                                               0.2357023 
    ##                                       Waterworld (1995) 
    ##                                               0.4714045 
    ##                                Wizard of Oz, The (1939) 
    ##                                               0.3086067

### Recommending movies for a single user

Let's again look at a concrete example of recommending a movie to a particular user, say user 236.

User 236 has seen the following movies:

``` r
viewed_movies["236",]
```

    ##                                     American Pie (1999) 
    ##                                                       0 
    ##                                   Apocalypse Now (1979) 
    ##                                                       1 
    ##                                       Armageddon (1998) 
    ##                                                       0 
    ##            Austin Powers: The Spy Who Shagged Me (1999) 
    ##                                                       0 
    ##                                Beautiful Mind, A (2001) 
    ##                                                       0 
    ##                              Breakfast Club, The (1985) 
    ##                                                       0 
    ##                                       Casablanca (1942) 
    ##                                                       1 
    ##                         Clear and Present Danger (1994) 
    ##                                                       0 
    ## Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) 
    ##                                                       1 
    ##                               Fifth Element, The (1997) 
    ##                                                       0 
    ##                                        Inception (2010) 
    ##                                                       0 
    ##                                Kill Bill: Vol. 1 (2003) 
    ##                                                       0 
    ##                                  Minority Report (2002) 
    ##                                                       0 
    ##                                         Outbreak (1995) 
    ##                                                       0 
    ##                                         Rain Man (1988) 
    ##                                                       0 
    ##                                      Stand by Me (1986) 
    ##                                                       0 
    ##                           Star Trek: Generations (1994) 
    ##                                                       0 
    ##                                      Taxi Driver (1976) 
    ##                                                       1 
    ##                                       Waterworld (1995) 
    ##                                                       0 
    ##                                Wizard of Oz, The (1939) 
    ##                                                       0

Another way of doing the same thing:

``` r
ratings_red %>% filter(userId == 236) %>% select(userId, title)
```

    ## # A tibble: 4 x 2
    ##   userId                                                   title
    ##    <int>                                                  <fctr>
    ## 1    236                                      Taxi Driver (1976)
    ## 2    236                                       Casablanca (1942)
    ## 3    236                                   Apocalypse Now (1979)
    ## 4    236 Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000)

We now implement the main idea behind item-based filtering. For each movie, we find out its similarity between that movie and each of the four movies user 236 has seen, and sum up those similarities. The resulting sum is that movie's "recommendation score".

We start by identifying the movies the user has seen:

``` r
user_seen <- ratings_red %>% 
        filter(userId == 236) %>% 
        select(title) %>% 
        unlist() %>% 
        as.character()
```

We then compute the similarities between all movies and these "seen" movies. For example, similarities the first seen movie, *Taxi Driver* are:

``` r
user_seen[1]
```

    ## [1] "Taxi Driver (1976)"

``` r
movie_similarities[,user_seen[1]]
```

    ##                                     American Pie (1999) 
    ##                                               0.0000000 
    ##                                   Apocalypse Now (1979) 
    ##                                               0.2357023 
    ##                                       Armageddon (1998) 
    ##                                               0.2041241 
    ##            Austin Powers: The Spy Who Shagged Me (1999) 
    ##                                               0.0000000 
    ##                                Beautiful Mind, A (2001) 
    ##                                               0.1924501 
    ##                              Breakfast Club, The (1985) 
    ##                                               0.2357023 
    ##                                       Casablanca (1942) 
    ##                                               0.4082483 
    ##                         Clear and Present Danger (1994) 
    ##                                               0.4082483 
    ## Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) 
    ##                                               0.5163978 
    ##                               Fifth Element, The (1997) 
    ##                                               0.2357023 
    ##                                        Inception (2010) 
    ##                                               0.0000000 
    ##                                Kill Bill: Vol. 1 (2003) 
    ##                                               0.2041241 
    ##                                  Minority Report (2002) 
    ##                                               0.2357023 
    ##                                         Outbreak (1995) 
    ##                                               0.3333333 
    ##                                         Rain Man (1988) 
    ##                                               0.0000000 
    ##                                      Stand by Me (1986) 
    ##                                               0.2886751 
    ##                           Star Trek: Generations (1994) 
    ##                                               0.3333333 
    ##                                      Taxi Driver (1976) 
    ##                                               0.0000000 
    ##                                       Waterworld (1995) 
    ##                                               0.0000000 
    ##                                Wizard of Oz, The (1939) 
    ##                                               0.2182179

We can do the same for each of the four seen movies or, more simply, do all four at once:

``` r
movie_similarities[,user_seen]
```

    ##                                                         Taxi Driver (1976)
    ## American Pie (1999)                                              0.0000000
    ## Apocalypse Now (1979)                                            0.2357023
    ## Armageddon (1998)                                                0.2041241
    ## Austin Powers: The Spy Who Shagged Me (1999)                     0.0000000
    ## Beautiful Mind, A (2001)                                         0.1924501
    ## Breakfast Club, The (1985)                                       0.2357023
    ## Casablanca (1942)                                                0.4082483
    ## Clear and Present Danger (1994)                                  0.4082483
    ## Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000)          0.5163978
    ## Fifth Element, The (1997)                                        0.2357023
    ## Inception (2010)                                                 0.0000000
    ## Kill Bill: Vol. 1 (2003)                                         0.2041241
    ## Minority Report (2002)                                           0.2357023
    ## Outbreak (1995)                                                  0.3333333
    ## Rain Man (1988)                                                  0.0000000
    ## Stand by Me (1986)                                               0.2886751
    ## Star Trek: Generations (1994)                                    0.3333333
    ## Taxi Driver (1976)                                               0.0000000
    ## Waterworld (1995)                                                0.0000000
    ## Wizard of Oz, The (1939)                                         0.2182179
    ##                                                         Casablanca (1942)
    ## American Pie (1999)                                             0.0000000
    ## Apocalypse Now (1979)                                           0.2886751
    ## Armageddon (1998)                                               0.0000000
    ## Austin Powers: The Spy Who Shagged Me (1999)                    0.0000000
    ## Beautiful Mind, A (2001)                                        0.2357023
    ## Breakfast Club, The (1985)                                      0.0000000
    ## Casablanca (1942)                                               0.0000000
    ## Clear and Present Danger (1994)                                 0.0000000
    ## Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000)         0.6324555
    ## Fifth Element, The (1997)                                       0.2886751
    ## Inception (2010)                                                0.0000000
    ## Kill Bill: Vol. 1 (2003)                                        0.2500000
    ## Minority Report (2002)                                          0.2886751
    ## Outbreak (1995)                                                 0.4082483
    ## Rain Man (1988)                                                 0.2886751
    ## Stand by Me (1986)                                              0.0000000
    ## Star Trek: Generations (1994)                                   0.0000000
    ## Taxi Driver (1976)                                              0.4082483
    ## Waterworld (1995)                                               0.4082483
    ## Wizard of Oz, The (1939)                                        0.2672612
    ##                                                         Apocalypse Now (1979)
    ## American Pie (1999)                                                 0.4330127
    ## Apocalypse Now (1979)                                               0.0000000
    ## Armageddon (1998)                                                   0.5773503
    ## Austin Powers: The Spy Who Shagged Me (1999)                        0.6123724
    ## Beautiful Mind, A (2001)                                            0.4082483
    ## Breakfast Club, The (1985)                                          0.3333333
    ## Casablanca (1942)                                                   0.2886751
    ## Clear and Present Danger (1994)                                     0.2886751
    ## Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000)             0.3651484
    ## Fifth Element, The (1997)                                           0.1666667
    ## Inception (2010)                                                    0.3086067
    ## Kill Bill: Vol. 1 (2003)                                            0.5773503
    ## Minority Report (2002)                                              0.3333333
    ## Outbreak (1995)                                                     0.2357023
    ## Rain Man (1988)                                                     0.5000000
    ## Stand by Me (1986)                                                  0.2041241
    ## Star Trek: Generations (1994)                                       0.0000000
    ## Taxi Driver (1976)                                                  0.2357023
    ## Waterworld (1995)                                                   0.4714045
    ## Wizard of Oz, The (1939)                                            0.3086067
    ##                                                         Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000)
    ## American Pie (1999)                                                                                   0.1581139
    ## Apocalypse Now (1979)                                                                                 0.3651484
    ## Armageddon (1998)                                                                                     0.3162278
    ## Austin Powers: The Spy Who Shagged Me (1999)                                                          0.2236068
    ## Beautiful Mind, A (2001)                                                                              0.2981424
    ## Breakfast Club, The (1985)                                                                            0.0000000
    ## Casablanca (1942)                                                                                     0.6324555
    ## Clear and Present Danger (1994)                                                                       0.0000000
    ## Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000)                                               0.0000000
    ## Fifth Element, The (1997)                                                                             0.1825742
    ## Inception (2010)                                                                                      0.1690309
    ## Kill Bill: Vol. 1 (2003)                                                                              0.4743416
    ## Minority Report (2002)                                                                                0.7302967
    ## Outbreak (1995)                                                                                       0.5163978
    ## Rain Man (1988)                                                                                       0.1825742
    ## Stand by Me (1986)                                                                                    0.0000000
    ## Star Trek: Generations (1994)                                                                         0.2581989
    ## Taxi Driver (1976)                                                                                    0.5163978
    ## Waterworld (1995)                                                                                     0.5163978
    ## Wizard of Oz, The (1939)                                                                              0.5070926

Each movie's recommendation score is obtained by summing across columns, each column representing a seen movie:

``` r
apply(movie_similarities[,user_seen],1,sum)
```

    ##                                     American Pie (1999) 
    ##                                               0.5911266 
    ##                                   Apocalypse Now (1979) 
    ##                                               0.8895258 
    ##                                       Armageddon (1998) 
    ##                                               1.0977022 
    ##            Austin Powers: The Spy Who Shagged Me (1999) 
    ##                                               0.8359792 
    ##                                Beautiful Mind, A (2001) 
    ##                                               1.1345430 
    ##                              Breakfast Club, The (1985) 
    ##                                               0.5690356 
    ##                                       Casablanca (1942) 
    ##                                               1.3293790 
    ##                         Clear and Present Danger (1994) 
    ##                                               0.6969234 
    ## Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) 
    ##                                               1.5140017 
    ##                               Fifth Element, The (1997) 
    ##                                               0.8736182 
    ##                                        Inception (2010) 
    ##                                               0.4776376 
    ##                                Kill Bill: Vol. 1 (2003) 
    ##                                               1.5058161 
    ##                                  Minority Report (2002) 
    ##                                               1.5880075 
    ##                                         Outbreak (1995) 
    ##                                               1.4936817 
    ##                                         Rain Man (1988) 
    ##                                               0.9712493 
    ##                                      Stand by Me (1986) 
    ##                                               0.4927993 
    ##                           Star Trek: Generations (1994) 
    ##                                               0.5915322 
    ##                                      Taxi Driver (1976) 
    ##                                               1.1603483 
    ##                                       Waterworld (1995) 
    ##                                               1.3960506 
    ##                                Wizard of Oz, The (1939) 
    ##                                               1.3011784

The preceding explanation hopefully makes the details of the calculations clear, but it is quite unwieldy. We can do all the calculations more neatly as:

``` r
user_scores <- tibble(title = row.names(movie_similarities), 
                      score = apply(movie_similarities[,user_seen],1,sum),
                      seen = viewed_movies["236",])
user_scores %>% filter(seen == 0) %>% arrange(desc(score))
```

    ## # A tibble: 16 x 3
    ##                                           title     score  seen
    ##                                           <chr>     <dbl> <dbl>
    ##  1                       Minority Report (2002) 1.5880075     0
    ##  2                     Kill Bill: Vol. 1 (2003) 1.5058161     0
    ##  3                              Outbreak (1995) 1.4936817     0
    ##  4                            Waterworld (1995) 1.3960506     0
    ##  5                     Wizard of Oz, The (1939) 1.3011784     0
    ##  6                     Beautiful Mind, A (2001) 1.1345430     0
    ##  7                            Armageddon (1998) 1.0977022     0
    ##  8                              Rain Man (1988) 0.9712493     0
    ##  9                    Fifth Element, The (1997) 0.8736182     0
    ## 10 Austin Powers: The Spy Who Shagged Me (1999) 0.8359792     0
    ## 11              Clear and Present Danger (1994) 0.6969234     0
    ## 12                Star Trek: Generations (1994) 0.5915322     0
    ## 13                          American Pie (1999) 0.5911266     0
    ## 14                   Breakfast Club, The (1985) 0.5690356     0
    ## 15                           Stand by Me (1986) 0.4927993     0
    ## 16                             Inception (2010) 0.4776376     0

So we'd end up recommending "Minority Report" to this particular user.

Let's repeat the process to generate a recommendation for one more user, user 149:

``` r
# do for user 149
user <- "149"
user_seen <- ratings_red %>% filter(userId == user) %>% select(title) %>% unlist() %>% as.character()
user_scores <- tibble(title = row.names(movie_similarities), 
                      score = apply(movie_similarities[,user_seen],1,sum),
                      seen = viewed_movies[user,])
user_scores %>% filter(seen == 0) %>% arrange(desc(score))
```

    ## # A tibble: 15 x 3
    ##                                                      title     score  seen
    ##                                                      <chr>     <dbl> <dbl>
    ##  1                                         Rain Man (1988) 2.4485086     0
    ##  2                                       Armageddon (1998) 2.3791013     0
    ##  3                                Beautiful Mind, A (2001) 2.3202108     0
    ##  4                                Wizard of Oz, The (1939) 2.1446941     0
    ##  5                                Kill Bill: Vol. 1 (2003) 1.9136494     0
    ##  6            Austin Powers: The Spy Who Shagged Me (1999) 1.5968267     0
    ##  7                         Clear and Present Danger (1994) 1.4695788     0
    ##  8                                   Apocalypse Now (1979) 1.4457435     0
    ##  9                           Star Trek: Generations (1994) 1.4181240     0
    ## 10                                         Outbreak (1995) 1.1999061     0
    ## 11                                  Minority Report (2002) 1.0849185     0
    ## 12                                       Waterworld (1995) 0.8796528     0
    ## 13                                      Taxi Driver (1976) 0.7600797     0
    ## 14 Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) 0.5097189     0
    ## 15                                       Casablanca (1942) 0.2886751     0

### A simple function to generate an item-based CF recommendation for any user

``` r
# a function to generate an item-based recommendation for any user
item_based_recommendations <- function(user, movie_similarities, viewed_movies){
  
  # turn into character if not already
  user <- ifelse(is.character(user),user,as.character(user))
  
  # get scores
  user_seen <- row.names(movie_similarities)[viewed_movies[user,] == TRUE]
  user_scores <- tibble(title = row.names(movie_similarities), 
                        score = apply(movie_similarities[,user_seen],1,sum),
                        seen = viewed_movies[user,])
  # sort unseen movies by score and remove the 'seen' column
  user_recom <- user_scores %>% filter(seen == 0) %>% arrange(desc(score)) %>% select(-seen)

  return(user_recom)
  
}
```

Let's check that its working with a user we've seen before, user 236:

``` r
item_based_recommendations(user = 236, movie_similarities = movie_similarities, viewed_movies = viewed_movies)
```

    ## # A tibble: 16 x 2
    ##                                           title     score
    ##                                           <chr>     <dbl>
    ##  1                       Minority Report (2002) 1.5880075
    ##  2                     Kill Bill: Vol. 1 (2003) 1.5058161
    ##  3                              Outbreak (1995) 1.4936817
    ##  4                            Waterworld (1995) 1.3960506
    ##  5                     Wizard of Oz, The (1939) 1.3011784
    ##  6                     Beautiful Mind, A (2001) 1.1345430
    ##  7                            Armageddon (1998) 1.0977022
    ##  8                              Rain Man (1988) 0.9712493
    ##  9                    Fifth Element, The (1997) 0.8736182
    ## 10 Austin Powers: The Spy Who Shagged Me (1999) 0.8359792
    ## 11              Clear and Present Danger (1994) 0.6969234
    ## 12                Star Trek: Generations (1994) 0.5915322
    ## 13                          American Pie (1999) 0.5911266
    ## 14                   Breakfast Club, The (1985) 0.5690356
    ## 15                           Stand by Me (1986) 0.4927993
    ## 16                             Inception (2010) 0.4776376

And now do it for all users with \`lapply'

``` r
lapply(sorted_my_users, item_based_recommendations, movie_similarities, viewed_movies)
```

    ## [[1]]
    ## # A tibble: 15 x 2
    ##                                                      title     score
    ##                                                      <chr>     <dbl>
    ##  1                                         Rain Man (1988) 2.4485086
    ##  2                                       Armageddon (1998) 2.3791013
    ##  3                                Beautiful Mind, A (2001) 2.3202108
    ##  4                                Wizard of Oz, The (1939) 2.1446941
    ##  5                                Kill Bill: Vol. 1 (2003) 1.9136494
    ##  6            Austin Powers: The Spy Who Shagged Me (1999) 1.5968267
    ##  7                         Clear and Present Danger (1994) 1.4695788
    ##  8                                   Apocalypse Now (1979) 1.4457435
    ##  9                           Star Trek: Generations (1994) 1.4181240
    ## 10                                         Outbreak (1995) 1.1999061
    ## 11                                  Minority Report (2002) 1.0849185
    ## 12                                       Waterworld (1995) 0.8796528
    ## 13                                      Taxi Driver (1976) 0.7600797
    ## 14 Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) 0.5097189
    ## 15                                       Casablanca (1942) 0.2886751
    ## 
    ## [[2]]
    ## # A tibble: 12 x 2
    ##                                                      title    score
    ##                                                      <chr>    <dbl>
    ##  1                                Wizard of Oz, The (1939) 2.999113
    ##  2                                     American Pie (1999) 2.647165
    ##  3            Austin Powers: The Spy Who Shagged Me (1999) 2.527730
    ##  4                                         Rain Man (1988) 2.509976
    ##  5                                Beautiful Mind, A (2001) 2.267620
    ##  6                                Kill Bill: Vol. 1 (2003) 2.116499
    ##  7                                   Apocalypse Now (1979) 2.041554
    ##  8                                       Waterworld (1995) 1.954568
    ##  9                                        Inception (2010) 1.902222
    ## 10 Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) 1.789796
    ## 11                                  Minority Report (2002) 1.709155
    ## 12                                       Casablanca (1942) 1.105172
    ## 
    ## [[3]]
    ## # A tibble: 14 x 2
    ##                                                      title     score
    ##                                                      <chr>     <dbl>
    ##  1                                Kill Bill: Vol. 1 (2003) 3.3505058
    ##  2                                         Rain Man (1988) 2.9646911
    ##  3                                Wizard of Oz, The (1939) 2.6432589
    ##  4                              Breakfast Club, The (1985) 2.3938846
    ##  5                                   Apocalypse Now (1979) 2.2272180
    ##  6                                      Stand by Me (1986) 2.0682345
    ##  7                                       Waterworld (1995) 1.9085035
    ##  8 Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) 1.8543858
    ##  9            Austin Powers: The Spy Who Shagged Me (1999) 1.8022418
    ## 10                                         Outbreak (1995) 1.7476316
    ## 11                           Star Trek: Generations (1994) 1.4944449
    ## 12                         Clear and Present Danger (1994) 1.2743774
    ## 13                                      Taxi Driver (1976) 0.8679788
    ## 14                                       Casablanca (1942) 0.8130525
    ## 
    ## [[4]]
    ## # A tibble: 16 x 2
    ##                                           title     score
    ##                                           <chr>     <dbl>
    ##  1                       Minority Report (2002) 1.5880075
    ##  2                     Kill Bill: Vol. 1 (2003) 1.5058161
    ##  3                              Outbreak (1995) 1.4936817
    ##  4                            Waterworld (1995) 1.3960506
    ##  5                     Wizard of Oz, The (1939) 1.3011784
    ##  6                     Beautiful Mind, A (2001) 1.1345430
    ##  7                            Armageddon (1998) 1.0977022
    ##  8                              Rain Man (1988) 0.9712493
    ##  9                    Fifth Element, The (1997) 0.8736182
    ## 10 Austin Powers: The Spy Who Shagged Me (1999) 0.8359792
    ## 11              Clear and Present Danger (1994) 0.6969234
    ## 12                Star Trek: Generations (1994) 0.5915322
    ## 13                          American Pie (1999) 0.5911266
    ## 14                   Breakfast Club, The (1985) 0.5690356
    ## 15                           Stand by Me (1986) 0.4927993
    ## 16                             Inception (2010) 0.4776376
    ## 
    ## [[5]]
    ## # A tibble: 14 x 2
    ##                                           title    score
    ##                                           <chr>    <dbl>
    ##  1                            Waterworld (1995) 2.421511
    ##  2                              Rain Man (1988) 2.368556
    ##  3                            Armageddon (1998) 2.325661
    ##  4                        Apocalypse Now (1979) 2.228389
    ##  5                          American Pie (1999) 2.179788
    ##  6                              Outbreak (1995) 2.140052
    ##  7                            Casablanca (1942) 2.082342
    ##  8                             Inception (2010) 2.082118
    ##  9                    Fifth Element, The (1997) 1.911443
    ## 10 Austin Powers: The Spy Who Shagged Me (1999) 1.681564
    ## 11                   Breakfast Club, The (1985) 1.549839
    ## 12                Star Trek: Generations (1994) 1.456120
    ## 13              Clear and Present Danger (1994) 1.161212
    ## 14                           Stand by Me (1986) 1.010083
    ## 
    ## [[6]]
    ## # A tibble: 15 x 2
    ##                                                      title     score
    ##                                                      <chr>     <dbl>
    ##  1                                     American Pie (1999) 2.7219477
    ##  2                                         Rain Man (1988) 2.5316784
    ##  3                                       Armageddon (1998) 2.5162900
    ##  4                                Wizard of Oz, The (1939) 2.2216943
    ##  5 Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) 2.0369600
    ##  6                                       Waterworld (1995) 1.9400817
    ##  7                               Fifth Element, The (1997) 1.6598335
    ##  8            Austin Powers: The Spy Who Shagged Me (1999) 1.5033833
    ##  9                              Breakfast Club, The (1985) 1.3388635
    ## 10                                         Outbreak (1995) 1.3078052
    ## 11                                       Casablanca (1942) 1.0630525
    ## 12                                      Stand by Me (1986) 0.9255320
    ## 13                                      Taxi Driver (1976) 0.8679788
    ## 14                         Clear and Present Danger (1994) 0.7743774
    ## 15                           Star Trek: Generations (1994) 0.6463702
    ## 
    ## [[7]]
    ## # A tibble: 14 x 2
    ##                                           title    score
    ##                                           <chr>    <dbl>
    ##  1                     Beautiful Mind, A (2001) 2.766210
    ##  2                     Kill Bill: Vol. 1 (2003) 2.620737
    ##  3                          American Pie (1999) 2.344958
    ##  4                    Fifth Element, The (1997) 2.110443
    ##  5                              Rain Man (1988) 2.006715
    ##  6                              Outbreak (1995) 1.947602
    ##  7                        Apocalypse Now (1979) 1.893045
    ##  8                            Waterworld (1995) 1.832486
    ##  9                   Breakfast Club, The (1985) 1.830237
    ## 10 Austin Powers: The Spy Who Shagged Me (1999) 1.813683
    ## 11                           Taxi Driver (1976) 1.507775
    ## 12                           Stand by Me (1986) 1.398157
    ## 13                            Casablanca (1942) 1.188392
    ## 14              Clear and Present Danger (1994) 1.175510
    ## 
    ## [[8]]
    ## # A tibble: 10 x 2
    ##                                           title    score
    ##                                           <chr>    <dbl>
    ##  1                            Armageddon (1998) 3.948396
    ##  2                          American Pie (1999) 3.802523
    ##  3                        Apocalypse Now (1979) 3.655136
    ##  4 Austin Powers: The Spy Who Shagged Me (1999) 3.159962
    ##  5                             Inception (2010) 3.007939
    ##  6                   Breakfast Club, The (1985) 2.952208
    ##  7                           Taxi Driver (1976) 2.344176
    ##  8                           Stand by Me (1986) 2.234828
    ##  9              Clear and Present Danger (1994) 2.146810
    ## 10                Star Trek: Generations (1994) 1.691823
    ## 
    ## [[9]]
    ## # A tibble: 11 x 2
    ##                                                      title    score
    ##                                                      <chr>    <dbl>
    ##  1                                       Armageddon (1998) 4.328748
    ##  2            Austin Powers: The Spy Who Shagged Me (1999) 3.258908
    ##  3                                   Apocalypse Now (1979) 3.239949
    ##  4                                  Minority Report (2002) 2.836177
    ##  5                                       Waterworld (1995) 2.784766
    ##  6                         Clear and Present Danger (1994) 2.511217
    ##  7                                         Outbreak (1995) 2.254525
    ##  8                           Star Trek: Generations (1994) 2.047010
    ##  9 Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) 1.971870
    ## 10                                      Taxi Driver (1976) 1.374872
    ## 11                                       Casablanca (1942) 1.330314
    ## 
    ## [[10]]
    ## # A tibble: 12 x 2
    ##                                                      title    score
    ##                                                      <chr>    <dbl>
    ##  1                                     American Pie (1999) 3.930501
    ##  2                                Beautiful Mind, A (2001) 3.506504
    ##  3                                Kill Bill: Vol. 1 (2003) 3.441241
    ##  4                         Clear and Present Danger (1994) 2.917744
    ##  5                                       Waterworld (1995) 2.836248
    ##  6                                        Inception (2010) 2.741235
    ##  7                                         Outbreak (1995) 2.382328
    ##  8                           Star Trek: Generations (1994) 2.129141
    ##  9                                  Minority Report (2002) 1.933380
    ## 10 Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) 1.777224
    ## 11                                      Taxi Driver (1976) 1.418124
    ## 12                                       Casablanca (1942) 1.133287
    ## 
    ## [[11]]
    ## # A tibble: 14 x 2
    ##                                                      title     score
    ##                                                      <chr>     <dbl>
    ##  1                                Wizard of Oz, The (1939) 2.8691747
    ##  2                                   Apocalypse Now (1979) 2.8045682
    ##  3                               Fifth Element, The (1997) 2.5258589
    ##  4                              Breakfast Club, The (1985) 2.5158931
    ##  5                                  Minority Report (2002) 2.4729459
    ##  6                                       Waterworld (1995) 2.2851737
    ##  7            Austin Powers: The Spy Who Shagged Me (1999) 2.1557952
    ##  8                                      Stand by Me (1986) 1.8367629
    ##  9 Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) 1.5984307
    ## 10                         Clear and Present Danger (1994) 1.5243774
    ## 11                                         Outbreak (1995) 1.4487731
    ## 12                           Star Trek: Generations (1994) 1.0230404
    ## 13                                       Casablanca (1942) 0.7743774
    ## 14                                      Taxi Driver (1976) 0.6006984
    ## 
    ## [[12]]
    ## # A tibble: 9 x 2
    ##                                                     title    score
    ##                                                     <chr>    <dbl>
    ## 1                               Fifth Element, The (1997) 4.087694
    ## 2                                        Inception (2010) 3.732977
    ## 3                                         Outbreak (1995) 3.501986
    ## 4                                  Minority Report (2002) 3.481808
    ## 5                                      Stand by Me (1986) 3.460937
    ## 6 Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) 3.041645
    ## 7                           Star Trek: Generations (1994) 2.409586
    ## 8                                       Casablanca (1942) 1.738562
    ## 9                                      Taxi Driver (1976) 1.698569
    ## 
    ## [[13]]
    ## # A tibble: 14 x 2
    ##                                                      title     score
    ##                                                      <chr>     <dbl>
    ##  1                                       Armageddon (1998) 2.8688875
    ##  2                                         Rain Man (1988) 2.7832258
    ##  3                                Kill Bill: Vol. 1 (2003) 2.6267089
    ##  4                               Fifth Element, The (1997) 2.4106641
    ##  5                                      Stand by Me (1986) 2.2533562
    ##  6                                        Inception (2010) 2.1106328
    ##  7                                   Apocalypse Now (1979) 2.0955735
    ##  8                         Clear and Present Danger (1994) 2.0921155
    ##  9                                       Waterworld (1995) 2.0426368
    ## 10                                  Minority Report (2002) 1.7357426
    ## 11                                         Outbreak (1995) 1.4725029
    ## 12 Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) 1.4451545
    ## 13                                      Taxi Driver (1976) 0.9797036
    ## 14                                       Casablanca (1942) 0.5029635
    ## 
    ## [[14]]
    ## # A tibble: 13 x 2
    ##                                                      title     score
    ##                                                      <chr>     <dbl>
    ##  1                                Wizard of Oz, The (1939) 3.1777814
    ##  2                              Breakfast Club, The (1985) 2.8492264
    ##  3                                  Minority Report (2002) 2.8062792
    ##  4            Austin Powers: The Spy Who Shagged Me (1999) 2.7681676
    ##  5                                       Waterworld (1995) 2.7565782
    ##  6                               Fifth Element, The (1997) 2.6925255
    ##  7                                      Stand by Me (1986) 2.0408871
    ##  8 Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) 1.9635791
    ##  9                         Clear and Present Danger (1994) 1.8130525
    ## 10                                         Outbreak (1995) 1.6844753
    ## 11                                       Casablanca (1942) 1.0630525
    ## 12                           Star Trek: Generations (1994) 1.0230404
    ## 13                                      Taxi Driver (1976) 0.8364006
    ## 
    ## [[15]]
    ## # A tibble: 11 x 2
    ##                              title    score
    ##                              <chr>    <dbl>
    ##  1        Beautiful Mind, A (2001) 4.164875
    ##  2                 Rain Man (1988) 3.840984
    ##  3        Wizard of Oz, The (1939) 3.836516
    ##  4       Fifth Element, The (1997) 2.748506
    ##  5      Breakfast Club, The (1985) 2.716148
    ##  6                Inception (2010) 2.677746
    ##  7 Clear and Present Danger (1994) 2.458725
    ##  8               Casablanca (1942) 2.276302
    ##  9              Taxi Driver (1976) 1.729384
    ## 10   Star Trek: Generations (1994) 1.728282
    ## 11              Stand by Me (1986) 1.626683

Collaborative filtering with matrix factorization
-------------------------------------------------

In this section we're going to look at a different way of doing collaborative filtering, one based on the idea of *matrix factorization*, a topic from linear algebra.

Matrix factorization, also called matrix decomposition, takes a matrix and represents it as a product of other (usually two) matrices. There are many ways to do matrix factorization, and different problems tend to use different methods.

In recommendation systems, matrix factorization is used to decompose the ratings matrix into the product of two matrices. This is done in such a way that the known ratings are matched as closely as possible.

The key feature of matrix factorization for recommendation systems is that while the ratings matrix is incomplete (i.e. some entries are blank), the two matrices the ratings matrix is decomposed into are *complete* (no blank entries). This gives a straightforward way of filling in blank spaces in the original ratings matrix, as we'll see.

Its actually easier to see the underlying logic and calculations in a spreadsheet setting, so we'll first save the ratings matrix as a .csv file and then jump over to Excel for a bit, before returning to work in R again.

``` r
# get ratings in wide format
ratings_wide <- ratings_red %>% 
  select(userId,title,rating) %>% 
  complete(userId, title) %>% 
  spread(key = title, value = rating)

# convert data to matrix form 
sorted_my_users <- as.character(unlist(ratings_wide[,1]))
ratings_wide <- as.matrix(ratings_wide[,-1])
row.names(ratings_wide) <- sorted_my_users

# save as csv for Excel demo
write.csv(ratings_wide,"output/ratings_for_excel_example.csv")
```

Now let's set up the same computations in R, which will be faster and easier to generalise beyond a particular size dataset. We start by defining a function that will compute the sum of squared differences between the observed movie ratings and any other set of predicted ratings (for example, ones predicted by matrix factorization). Note the we only count movies that have already been rated in the accuracy calculation.

``` r
recommender_accuracy <- function(x, observed_ratings){
    
  # extract user and movie factors from parameter vector (note x is defined such that 
  # the first 75 elements are latent factors for users and rest are for movies)
  user_factors <- matrix(x[1:75],15,5)
  movie_factors <- matrix(x[76:175],5,20)
  
  # get predictions from dot products of respective user and movie factor
  predicted_ratings <- user_factors %*% movie_factors
  
  # model accuracy is sum of squared errors over all rated movies
  errors <- (observed_ratings - predicted_ratings)^2 
  accuracy <- sqrt(mean(errors[!is.na(observed_ratings)]))   # only use rated movies
  
  return(accuracy)
}
```

> **Exercise**: This function isn't general, because it refers specifically to a ratings matrix with 15 users, 20 movies, and 5 latent factors. Make the function general.

We'll now optimize the values in the user and movie latent factors, choosing them so that the root mean square error (the square root of the average squared difference between observed and predicted ratings) is a minimum. I've done this using R's inbuilt numerical optimizer `optim()`, with the default "Nelder-Mead" method. There are better ways to do this - experiment! Always check whether the optimizer has converged (although you can't always trust this), see `help(optim)` for details.

``` r
set.seed(10)
# optimization step
rec1 <- optim(par=runif(175), recommender_accuracy, 
            observed_ratings = ratings_wide, control=list(maxit=100000))
rec1$convergence
```

    ## [1] 1

``` r
rec1$value
```

    ## [1] 0.2587128

The best value of the objective function found by `optim()` after 100000 iterations is 0.258, but note that it hasn't converged yet, so we should really run for longer or try another optimizer! Ignoring this for now, we can extract the optimal user and movie factors. With a bit of work, these can be interpreted and often give useful information. Unfortunately we don't have time to look at this further (although it is similar to the interpretation of principal components, if you are familiar with that).

``` r
# extract optimal user factors
user_factors <- matrix(rec1$par[1:75],15,5)
head(user_factors)
```

    ##            [,1]       [,2]       [,3]       [,4]       [,5]
    ## [1,]  0.3657830  2.7268681  0.6118837  0.5915031  0.5388406
    ## [2,] -0.3584511  1.4483009  0.1445300  3.9665257  0.4519185
    ## [3,] -1.1586760  1.1801953 -0.3340436  3.3474468  0.5945029
    ## [4,]  2.3956245  1.4171811  0.6028387 -0.4774059 -2.9170233
    ## [5,]  3.8781238 -0.4556751  0.2479186  1.5770055  0.8684524
    ## [6,]  2.2149462  0.7954067  0.7399249  0.1134245 -1.1141201

``` r
# extract optimal movie factors
movie_factors <- matrix(rec1$par[76:175],5,20)
head(movie_factors)
```

    ##           [,1]       [,2]       [,3]       [,4]       [,5]       [,6]
    ## [1,] 0.8897293  1.0935242 -0.9464366  1.4006422  0.6064796  0.9428172
    ## [2,] 0.6096810  1.1843201  1.6947961  0.3717246  0.3912175  0.9700681
    ## [3,] 1.6166807  1.5763010 -0.3074508 -1.5712743  0.4193076 -0.5207672
    ## [4,] 0.6136185 -1.1419063  0.4361951 -0.2146516  1.9620240  0.9729611
    ## [5,] 0.2804760  0.2288244 -2.9590191 -5.0187050 -2.4163176  0.6632936
    ##            [,7]       [,8]       [,9]      [,10]      [,11]      [,12]
    ## [1,] -0.7147964  0.4015494  0.3123863  0.6102844  0.3513919  0.9861259
    ## [2,]  0.8112541  1.1702875 -1.0225496  1.4569461  1.6938504  0.5535347
    ## [3,]  1.1674692 -1.8203013  0.2476967  0.6434199  0.8325815  0.4092764
    ## [4,] -4.2578095  0.6896335  2.4218357  0.8215509  0.9052081  0.5272057
    ## [5,] -0.5934876 -0.2747528 -1.7981541 -1.4443502 -1.4804007 -1.7033162
    ##          [,13]      [,14]      [,15]     [,16]      [,17]      [,18]
    ## [1,] 0.7456028  0.3259996  0.7134509  1.427887  0.3648229  1.3502739
    ## [2,] 2.3285074  0.0461059  0.9337365  1.096742 -0.2743746  2.3285977
    ## [3,] 1.7934327  0.2166642  1.0682245  1.135006  1.4934491 -0.5324783
    ## [4,] 0.7493992  1.2393460  1.1172589  1.170068  1.1219603  0.3021888
    ## [5,] 1.1223406 -1.8454364 -0.9872299 -1.794029 -0.1636815  0.5653309
    ##           [,19]     [,20]
    ## [1,]  0.7810916 0.7038951
    ## [2,]  0.4156057 0.8435226
    ## [3,]  1.0687913 1.4344043
    ## [4,] -0.1618643 1.0733206
    ## [5,]  0.2941909 0.5923625

Most importantly, we can get **predicted movie ratings** for any user, by taking the appropriate dot product of user and movie factors. Here we show the predictions for user 1:

``` r
# check predictions for one user
predicted_ratings <- user_factors %*% movie_factors
rbind(round(predicted_ratings[1,],1), as.numeric(ratings_wide[1,]))
```

    ##      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10] [,11] [,12] [,13]
    ## [1,]  3.5    4  2.8 -2.3  1.4  3.6 -0.2  2.5 -2.1   4.3     5   1.5   8.8
    ## [2,]  3.0   NA   NA   NA   NA  4.0   NA   NA   NA   4.0     5    NA    NA
    ##      [,14] [,15] [,16] [,17] [,18] [,19] [,20]
    ## [1,]   0.1   3.6   3.9   0.9     7   2.1   4.4
    ## [2,]    NA    NA   4.0    NA    NA    NA    NA

### Adding L2 regularization

One trick that can improve the performance of matrix factorization collaborative filtering is to add L2 regularization. L2 regularization adds a penalty term to the function that we're trying to minimize, equal to the sum of the L2 norms over all user and movie factors. This penalizes large parameter values.

We first rewrite the *evaluate\_fit* function to make use of L2 regularization:

``` r
## adds L2 regularization, often improves accuracy

evaluate_fit_l2 <- function(x, observed_ratings, lambda){
  
  # extract user and movie factors from parameter vector
  user_factors <- matrix(x[1:75],15,5)
  movie_factors <- matrix(x[76:175],5,20)
  
  # get predictions from dot products
  predicted_ratings <- user_factors %*% movie_factors
  
  errors <- (observed_ratings - predicted_ratings)^2 
  
  # L2 norm penalizes large parameter values
  penalty <- sum(sqrt(apply(user_factors^2,1,sum))) + sum(sqrt(apply(movie_factors^2,2,sum)))
  
  # model accuracy contains an error term and a weighted penalty 
  accuracy <- sqrt(mean(errors[!is.na(observed_ratings)])) + lambda * penalty
  
  return(accuracy)
}
```

We now rerun the optimization with this new evaluation function:

``` r
set.seed(10)
# optimization step
rec2 <- optim(par=runif(175), evaluate_fit_l2, 
            lambda = 3e-3, observed_ratings = ratings_wide, control=list(maxit=100000))
rec2$convergence
```

    ## [1] 1

``` r
rec2$value
```

    ## [1] 0.5293048

The best value found is **worse** than before, but remember that we changed the objective function to include the L2 penalty term, so the numbers are not comparable. We need to extract just the RMSE that we're interested in. To do that we first need to extract the optimal parameter values (user and movie factors), and multiply these matrices together to get predicted ratings. From there, its easy to calculate the errors.

``` r
# extract optimal user and movie factors
user_factors <- matrix(rec2$par[1:75],15,5)
movie_factors <- matrix(rec2$par[76:175],5,20)

# get predicted ratings
predicted_ratings <- user_factors %*% movie_factors

# check accuracy
errors <- (ratings_wide - predicted_ratings)^2 
sqrt(mean(errors[!is.na(ratings_wide)]))
```

    ## [1] 0.2480996

Compare this with what we achieved without L2 regularization: did it work? As before, we can extract user and movie factors, and get predictions for any user.

``` r
# check predictions for one user
rbind(round(predicted_ratings[1,],1), as.numeric(ratings_wide[1,]))
```

    ##      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10] [,11] [,12] [,13]
    ## [1,]  3.1  1.8  2.4  6.8  5.3    4 -1.1  1.4  5.8   3.7     5   0.8   7.3
    ## [2,]  3.0   NA   NA   NA   NA    4   NA   NA   NA   4.0     5    NA    NA
    ##      [,14] [,15] [,16] [,17] [,18] [,19] [,20]
    ## [1,]   4.3   6.1   4.1   0.5  -2.1  -0.2     5
    ## [2,]    NA    NA   4.0    NA    NA    NA    NA

### Adding bias terms

We've already seen bias terms in the Excel example. Bias terms are additive factors that model the fact that some users are more generous than others (and so will give higher ratings, on average) and some movies are better than others (and so will get higher ratings, on average).

Let's adapt our evaluation function further to include a bias terms for both users and movies:

``` r
## add an additive bias term for each user and movie

evaluate_fit_l2_bias <- function(x, observed_ratings, lambda){
  # extract user and movie factors and bias terms from parameter vector
  user_factors <- matrix(x[1:75],15,5)
  movie_factors <- matrix(x[76:175],5,20)
  # the bias vectors are repeated to make the later matrix calculations easier 
  user_bias <- matrix(x[176:190],nrow=15,ncol=20)
  movie_bias <- t(matrix(x[191:210],nrow=20,ncol=15))
  
  # get predictions from dot products + bias terms
  predicted_ratings <- user_factors %*% movie_factors + user_bias + movie_bias
  
  errors <- (observed_ratings - predicted_ratings)^2 
  
  # L2 norm penalizes large parameter values (note not applied to bias terms)
  penalty <- sum(sqrt(apply(user_factors^2,1,sum))) + sum(sqrt(apply(movie_factors^2,2,sum)))
  
  # model accuracy contains an error term and a weighted penalty 
  accuracy <- sqrt(mean(errors[!is.na(observed_ratings)])) + lambda * penalty
  
  return(accuracy)
}
```

Again, rerun the optimization:

``` r
set.seed(10)
# optimization step (note longer parameter vector to include bias)
rec3 <- optim(par=runif(220),evaluate_fit_l2_bias,
              observed_ratings = ratings_wide, lambda = 3e-3, control=list(maxit=100000))
rec3$convergence
```

    ## [1] 1

``` r
rec3$value
```

    ## [1] 0.3566958

This value isn't comparable to either of the previous values, for the same reason as before: the objective function has changed to include bias terms. Extracting just the RMSE:

``` r
# extract optimal user and movie factors and bias terms
user_factors <- matrix(rec3$par[1:75],15,5)
movie_factors <- matrix(rec3$par[76:175],5,20)
user_bias <- matrix(rec3$par[176:190],nrow=15,ncol=20)
movie_bias <- t(matrix(rec3$par[191:210],nrow=20,ncol=15))

# get predicted ratings
predicted_ratings <- user_factors %*% movie_factors + user_bias + movie_bias

# check accuracy
errors <- (ratings_wide - predicted_ratings)^2 
sqrt(mean(errors[!is.na(ratings_wide)]))
```

    ## [1] 0.1684927

This is indeed an improvement over what we've seen before (at least, for the parameter settings above!).

We can examine and interpret the user or movie latent factors, or bias terms, if we want to. Below we show the movie bias terms, which give a reasonable reflection of movie quality (with some notable exceptions!)

``` r
data.frame(movies = colnames(viewed_movies), bias = movie_bias[1,]) %>% arrange(desc(bias))
```

    ##                                                     movies       bias
    ## 1                               Breakfast Club, The (1985)  2.3733467
    ## 2                                       Stand by Me (1986)  2.2084965
    ## 3                                 Beautiful Mind, A (2001)  1.8421056
    ## 4                                       Taxi Driver (1976)  1.6469531
    ## 5                                        Armageddon (1998)  1.5461443
    ## 6                                Fifth Element, The (1997)  1.4482570
    ## 7                                 Wizard of Oz, The (1939)  1.1822792
    ## 8  Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000)  1.1459319
    ## 9                                         Inception (2010)  1.0463059
    ## 10                                  Minority Report (2002)  0.9356814
    ## 11                                Kill Bill: Vol. 1 (2003)  0.9259268
    ## 12            Austin Powers: The Spy Who Shagged Me (1999)  0.8259566
    ## 13                                   Apocalypse Now (1979)  0.7815287
    ## 14                           Star Trek: Generations (1994)  0.6191147
    ## 15                                       Casablanca (1942)  0.5886945
    ## 16                         Clear and Present Danger (1994)  0.3475975
    ## 17                                         Rain Man (1988)  0.3262127
    ## 18                                     American Pie (1999)  0.2632694
    ## 19                                       Waterworld (1995)  0.1802591
    ## 20                                         Outbreak (1995) -0.3320020

Finally, we again get predicted ratings for one user:

``` r
# check predictions for one user
rbind(round(predicted_ratings[1,],1), as.numeric(ratings_wide[1,]))
```

    ##      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10] [,11] [,12] [,13]
    ## [1,]    3  2.1  3.1  0.3  2.7  3.9  5.4  4.4    3   3.9     5   1.9   1.8
    ## [2,]    3   NA   NA   NA   NA  4.0   NA   NA   NA   4.0     5    NA    NA
    ##      [,14] [,15] [,16] [,17] [,18] [,19] [,20]
    ## [1,]   3.7   2.8   4.2   1.9     4   1.1   4.8
    ## [2,]    NA    NA   4.0    NA    NA    NA    NA

Exercises
---------

There are a few places in the notebook where an exercise is indicated. Specifically:

1.  Adapt the pairwise similarity function so that it doesn't use loops.
2.  Implement a k-nearest-neighbours version of item-based collaborative filtering.
3.  Adapt the `recommender_accuracy()` function so that it can be used with an arbitrary number of users and movies.
4.  Experiment with the optimizers used in the matrix factorization collaborative filter.
