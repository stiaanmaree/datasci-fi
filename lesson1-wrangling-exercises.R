library(dplyr)

data <- read.csv("data/ml-latest-small/ratings.csv")

View(data)

#filter
#arrange
#mutate
#select
#summarise
#group_by


#inner_join()


head(data)


data %>% filter(userId == 353)  %>% arrange(desc(rating), movieId) %>% mutate(liked = ifelse(rating >= 3, "yes", "no"))


data %>% filter(userId == 353)

data %>% arrange(desc(rating), movieId)

data %>% mutate(liked = ifelse(rating >= 3, "yes", "no"))

data %>% select(movieId, rating)

data %>% summarise(ave_rating = mean(rating))

data %>% group_by(movieId) %>% summarise(count = n(), ave_rating = mean(rating))

data %>% group_by(movieId) %>% mutate(good_movie = ifelse(mean(rating) >= 4, "yes", "no"))

data %>% group_by(movieId) %>% filter(n() > 24)

