library(tidyverse)
library(tidytext)
library(stringr)
library(lubridate)
library(topicmodels)
library(ggplot2)
library(shiny)

set.seed(1234)

# Define UI for Complaints application 
ui <- fluidPage(
   
   # Application title
   titlePanel("US Consumer Financial Protection Bureau Complaints"),
   
   # Sidebar with input for:
   #    product
   #    consumer compensated
   #    number of topics
   #    input for a new complaint
   sidebarLayout(
      sidebarPanel(
        
         selectInput("product", "Product:", 
                     choices = c("All", 
                                 "Mortgage",
                                 "Credit card",
                                 "Debt collection",
                                 "Credit reporting",
                                 "Bank account or service"),
                     selected = "All"
                     #selected = "Bank account or service"
                     ),
         
         selectInput("consumer_compensated", "Consumer Compensated:", 
                     choices = c("All", "FALSE", "TRUE"),
                     selected = "All"
                     #selected = "TRUE"
                     ),
         
         sliderInput("nr_topics",
                     "Number of Topics:",
                     min = 2,
                     max = 5,
                     value = 2),
         
         # Prepopulated a comment
         textAreaInput("new_complaint", "New Complaint:", 
                       "On XXXX/XXXX/15, someone created fraudulent checks and went to XXXX different Comerica Bank locations to cash checks from our accounts. In total XXXX checks were cashed on the same day. The bank branch suspected fraud and called us but could not get through to us immediately. We were able to speak to the bank branch that day, instructed the bank that it was fraudulent, to stop cashing those checks and to call the police. The police were able to arrive and arrest one individual while he was trying to cash a fake check. The police noted the other assailants fled the scene.",
                       rows = 5),
         
         actionButton("submit_button","Submit")
         
      ),

      # Create the three output areas
      mainPanel(

        plotOutput("sentimentHist"),
        plotOutput("topicsPlot"),
        verbatimTextOutput("nText")
      )
   )
)

# Define server logic
server <- function(input, output) {

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #
  # We create four reactive functions to generate and store the 
  # data and output.  Reactive functions only update when needed, 
  # so this will not result in for instance caluclating the data
  # for the sentiment again when changing the number of topics.
  #
  # The four functions:
  #
  #     get_tidy_complaints_reactive
  #     compute_sentiment_reactive
  #     compute_topics_reactive
  #     process_new_complaint_reactive
  #
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  # This function loads the data from the RData file computes the tidy complaints.
  # It also computes the bing sentiment
  get_tidy_complaints_reactive <- reactive({
    
      selected_product <- input$product
      selected_consumer_compensated <- input$consumer_compensated
      
      #load("D:/UCT/2017/Work/DSciFi/Project2/complaints.RData")
      load("complaints.RData")
      
      # parse the date
      complaints <- complaints %>% 
        mutate(date = parse_datetime(complaints$date_received, "%m/%d/%Y"))  
      # %>% mutate(month = make_date(year(date), month(date)))
      
      # filter complaints if the selections were not "All"
      if (selected_product != "All")
      {   complaints <- complaints %>% filter(str_detect(product, selected_product)) }
      if (selected_consumer_compensated != "All")
      { complaints <- complaints %>% filter(consumer_compensated == selected_consumer_compensated) }
      
      # clean complaints 
      replace_reg <- "XXXX|XX|'s|n't"
      unnest_reg <- "([^A-Za-z_\\d#@']|'(?![A-Za-z_\\d#@]))"
      
      tidy_complaints <- complaints %>% 
        mutate(consumer_complaint_narrative = str_replace_all(consumer_complaint_narrative, replace_reg, "")) %>% # remove stuff we don't want like XXXX
        unnest_tokens(word, consumer_complaint_narrative, token = "regex", pattern = unnest_reg) %>% # tokenize
        filter(!word %in% stop_words$word, str_detect(word, "[a-z]")) %>% # remove stop words
        select(product, consumer_compensated, id, date, word) # choose the variables we need
      
      # get bing sentiment score per word
      tidy_complaints <- tidy_complaints %>% 
        left_join(get_sentiments("bing")) %>% # add sentiments (pos or neg)
        select(word,sentiment,everything()) %>%
        mutate(sentiment = ifelse(is.na(sentiment), "neutral", sentiment))

      tidy_complaints
      
    })  

  # This function uses the output of get_tidy_complaints_reactive group the sentiment by complaints
  compute_sentiment_reactive  <- reactive({

    # summarize sentiment per complaint
    get_tidy_complaints_reactive() %>%
      group_by(id) %>%
      summarize(net_sentiment = (sum(sentiment == "positive") - sum(sentiment == "negative"))) %>% 
      select(net_sentiment)
  })

  # This functions computes the LDA topics, using the get_tidy_complaints_reactive function
  compute_topics_reactive <- reactive({

    selected_nr_of_topics <- input$nr_topics
    tidy_complaints <- get_tidy_complaints_reactive()
    tidy_complaints_topic <- tidy_complaints[,c(5,1)]

    # count the number of times words appear per complaint
    complaints_tdf <- tidy_complaints_topic %>%
      group_by(id,word) %>%
      count() %>%  
      ungroup() 
    
    # document term matrix
    dtm_complaints <- complaints_tdf %>% 
      cast_dtm(id, word, n)
    
    # computes LDA topics
    complaints_lda <- LDA(dtm_complaints, k = selected_nr_of_topics)
    
    term <- as.character(complaints_lda@terms)
    
    # Caters for 2,3,4 & 5 topics
    if (selected_nr_of_topics == 5) {
      
      topic1 <- complaints_lda@beta[1,]
      topic2 <- complaints_lda@beta[2,]
      topic3 <- complaints_lda@beta[3,]
      topic4 <- complaints_lda@beta[4,]
      topic5 <- complaints_lda@beta[5,]
      complaints_topics <- tibble(term = term, 
                                  topic1 = topic1, 
                                  topic2 = topic2, 
                                  topic3 = topic3, 
                                  topic4 = topic4, 
                                  topic5 = topic5)
      
      complaints_topics <- complaints_topics %>% 
        gather(topic1, topic2, topic3, topic4, topic5, key = "topic", value = "beta") %>%
        mutate(beta = exp(beta)) 
      
    } else if (selected_nr_of_topics == 4) {
      
      topic1 <- complaints_lda@beta[1,]
      topic2 <- complaints_lda@beta[2,]
      topic3 <- complaints_lda@beta[3,]
      topic4 <- complaints_lda@beta[4,]
      complaints_topics <- tibble(term = term, 
                                  topic1 = topic1, 
                                  topic2 = topic2, 
                                  topic3 = topic3, 
                                  topic4 = topic4)
      
      complaints_topics <- complaints_topics %>% 
        gather(topic1, topic2, topic3, topic4, key = "topic", value = "beta") %>%
        mutate(beta = exp(beta))
      
    } else if (selected_nr_of_topics == 3) {
      
      topic1 <- complaints_lda@beta[1,]
      topic2 <- complaints_lda@beta[2,]
      topic3 <- complaints_lda@beta[3,]
      complaints_topics <- tibble(term = term, 
                                  topic1 = topic1, 
                                  topic2 = topic2, 
                                  topic3 = topic3)
      
      complaints_topics <- complaints_topics %>% 
        gather(topic1, topic2, topic3, key = "topic", value = "beta") %>%
        mutate(beta = exp(beta)) 
      
    } else {
      
      topic1 <- complaints_lda@beta[1,]
      topic2 <- complaints_lda@beta[2,]
      complaints_topics <- tibble(term = term, topic1 = topic1, topic2 = topic2)
      
      complaints_topics <- complaints_topics %>% 
        gather(topic1, topic2, key = "topic", value = "beta") %>%
        mutate(beta = exp(beta)) 
      
    }
    
    complaints_topics <- tidy(complaints_lda, matrix = "beta")
    
    # get top 15 terms per topic
    complaints_topics %>%
      group_by(topic) %>%
      top_n(15, beta) %>%
      ungroup() %>%
      arrange(topic, -beta)
    
    })

    # This functio computes output for the new complaint
    process_new_complaint_reactive <- reactive({ 
      
      selected_new_complaint <- input$new_complaint
      selected_product <- input$product
      selected_consumer_compensated <-input$consumer_compensated
      selected_nr_of_topics <- input$nr_topics

      # to prevent an emppty textbox to generate an error
      if (selected_new_complaint == "")
      { selected_new_complaint = "NA" }

      # The result needs to cater for three pieces of information.
      result <- c("","","")
      
      #~~~~
      # Sentiment
      #~~~~
      
      # we work out the sentiment the same ways as for all the complaints of the current selection
      replace_reg <- "XXXX|XX|'s|n't"
      unnest_reg <- "([^A-Za-z_\\d#@']|'(?![A-Za-z_\\d#@]))"
      
      selected_new_complaint_df <- data.frame(id = 100000, consumer_complaint_narrative = selected_new_complaint)
      
      tidy_selected_new_complaint <- selected_new_complaint_df %>% 
        mutate(consumer_complaint_narrative = str_replace_all(consumer_complaint_narrative, replace_reg, "")) %>% # remove stuff we don't want like links
        unnest_tokens(word, consumer_complaint_narrative, token = "regex", pattern = unnest_reg) %>% # tokenize
        filter(!word %in% stop_words$word, str_detect(word, "[a-z]")) #%>% # remove stop words
      
      # get bing sentiment score per word
      tidy_selected_new_complaint  <- tidy_selected_new_complaint %>% 
        left_join(get_sentiments("bing")) %>% # add sentiments (pos or neg)
        select(word,sentiment,everything()) %>%
        mutate(sentiment = ifelse(is.na(sentiment), "neutral", sentiment))
      
      # summarize sentiment per complaint
      sentiments_selected_new_complaint <- tidy_selected_new_complaint %>%
        group_by(id) %>%
        summarize(net_sentiment = (sum(sentiment == "positive") - sum(sentiment == "negative")))

      # the sentiment score
      result[1] <-  sentiments_selected_new_complaint$net_sentiment
      
      # the quantile of the sentiment score 
      sentiments <- unlist(compute_sentiment_reactive())
      result[2] <-  ecdf(sentiments)(sentiments_selected_new_complaint$net_sentiment)

      #~~~~
      # LDA
      #~~~~
      
      # For the LDA, we need to add the new complaint to the current selection
      # then work out the gammas, and join the coplaint back to the gammas to get the probabilities for each topics.
      
      # Get Original Complaints
      tidy_complaints <- get_tidy_complaints_reactive()
      tidy_complaints_topic <- tidy_complaints[,c(5,1)]
      
      # Add New Complaint
      tidy_selected_new_complaint_lda <- tidy_selected_new_complaint[c(3,1)]
      tidy_complaints_topic_combo <- rbind(tidy_complaints_topic, tidy_selected_new_complaint_lda)

      # the rest of the topics is worked out the same as for the current selection      
      complaints_combo_tdf <- tidy_complaints_topic_combo %>%
        group_by(id,word) %>%
        count() %>%  
        ungroup() 
      
      dtm_complaints_combo <- complaints_combo_tdf %>% 
        cast_dtm(id, word, n)
      
      complaints_combo_lda <- LDA(dtm_complaints_combo, k = selected_nr_of_topics)

      # we need to join the new complaint back to the gamma values
      # to find get the probabilities of the different topics
      complaints_gamma <- selected_new_complaint_df %>% 
        left_join(tidy(complaints_combo_lda, matrix = "gamma") %>% 
                    mutate(id = as.numeric(document)) %>% # some cleaning to make key variable (reviewId) usable
                    select(-document) %>%
                    spread(key = topic, value = gamma, sep = "_"))
      
      # probabilities of topics
      if (selected_nr_of_topics == 5) {
        
        result[3] <- paste("topic1: ", complaints_gamma[3],
                           "topic2: ", complaints_gamma[4],
                           "topic3: ", complaints_gamma[5],
                           "topic4: ", complaints_gamma[6],
                           "topic5: ", complaints_gamma[7])
        
      } else if (selected_nr_of_topics == 4) {
        
        result[3] <- paste("topic1: ", complaints_gamma[3],
                           "topic2: ", complaints_gamma[4],
                           "topic3: ", complaints_gamma[5],
                           "topic4: ", complaints_gamma[6])
        
      } else if (selected_nr_of_topics == 3) {
        
        result[3] <- paste("topic1: ", complaints_gamma[3],
                           "topic2: ", complaints_gamma[4],
                           "topic3: ", complaints_gamma[5])
        
      } else {
        
        result[3] <- paste("topic1: ", complaints_gamma[3],
                           "topic2: ", complaints_gamma[4])
        
      }
      
      resultText <- paste("a. Sentiment Score: ", result[1], "\nb. Quantile of the sentiment score: ", result[2], "\nc. Topic probabilities: ", result[3])
      
      resultText

      })
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    # The rest of the server functions use the reactive functions to 
    # create the output
    #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
   # the compute_sentiment_reactive function is used for the historgram data
   output$sentimentHist <- renderPlot({

     withProgress(message = 'Processing Sentiment', value = 0, {
       incProgress(5/10)
       sentiments <- unlist(compute_sentiment_reactive())
       incProgress(10/10)
     })
     
     hist(sentiments, breaks = 15, col = 'darkgray', border = 'white')
     
   })
   
   # the compute_topics_reactive function is used for the topics plot
   output$topicsPlot <- renderPlot({

     withProgress(message = 'Processing Topics', value = 0, {
       incProgress(5/10)
        top_terms <- compute_topics_reactive()
        incProgress(10/10)
     })
      
     top_terms %>%
       mutate(term = reorder(term, beta)) %>%
       ggplot(aes(term, beta, fill = factor(topic))) +
       geom_col(show.legend = FALSE) +
       facet_wrap(~ topic, scales = "free") +
       coord_flip()
   })
   
   # Create an event to pick up the actionButton click event for the new complaint
   # It assigns the new complaint's computed values to one text variable
   ntext <- eventReactive(input$submit_button, {
     withProgress(message = 'Processing New Complaint', value = 0, {
        incProgress(5/10)
        process_new_complaint_reactive()
     })
   })
   
   # Upon changes in the computed text, the new complaint's output is displayed
   output$nText <- renderText({
     ntext()
   })
   
}

# Run the application 
shinyApp(ui = ui, server = server)
