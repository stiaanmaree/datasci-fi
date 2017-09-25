library(devtools)
has_devel()


library(jsonlite)

tweets <- data.frame()    
for(i in 2009:2017){
  x <- fromJSON(txt=paste0("data/condensed_",i,".json"),simplifyDataFrame = T)
  tweets <- rbind.data.frame(tweets, x)
}
rm(x)

save(tweets,file="data/trump-tweets.RData")

