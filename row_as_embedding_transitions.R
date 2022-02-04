# Load packages -----
library(tidyverse)
library(lme4)
library(lmerTest)

# Load data -------
rowdf <- read.csv('C:\\Users\\User\\Drive USP\\école\\2a\\p2\\modal\\python\\row_as_embedding_transitions.csv',
                 sep='\t', fileEncoding='utf8')

# Standardize data for analysis -------

c.length <- rowdf$length - 0.5                # between -0.5 and 0.5
c.interv <- rowdf$interv - mean(rowdf$interv)  # in seconds
c.pause <- rowdf$pause - mean(rowdf$pause)     # in seconds
c.speed <- rowdf$speed - mean(rowdf$speed)     # in seconds^-1, positive
c.age <- (rowdf$age - mean(rowdf$age[!is.na(rowdf$age)])) # in years
c.ADHD <- (rowdf$ADHD - mean(rowdf$ADHD))/5   # in units of 5 ADHD points
c.ADHD.first <- (rowdf$ADHD.first - mean(rowdf$ADHD.first))/5
c.ADHD.last <- (rowdf$ADHD.last - mean(rowdf$ADHD.last))/5
c.MEWS <- (rowdf$MEWS - mean(rowdf$MEWS))/5 # units of 5 MEWS points

crowdf = data.frame(suj = rowdf$suj,
                   bloc = rowdf$bloc,
                   prob = rowdf$prob,
                   length = c.length,
                   interv = c.interv,
                   pause = c.pause,
                   speed = c.speed,
                   age = c.age,
                   genre = rowdf$genre,
                   exp = rowdf$exp,
                   level = rowdf$level,
                   topic = rowdf$topic,
                   ADHD = c.ADHD,
                   ADHD.first = c.ADHD.first,
                   ADHD.last = c.ADHD.last,
                   MEWS = c.MEWS)

# Fixed effects models ------

fe01 <- lm(speed ~ ADHD, data=crowdf)
fe02 <- lm(length ~ ADHD, data=crowdf)
fe03 <- lm(interv ~ ADHD, data=crowdf)

# Mixed effects models ------

# Total ADHD, no MEWS
me01 <- lmer(speed ~ ADHD + (1|suj), data=crowdf)
me02 <- lmer(length ~ ADHD + (1|suj), data=crowdf)
me03 <- lmer(interv ~ ADHD + (1|suj), data=crowdf)
me01c <- lmer(speed ~ ADHD + genre + age + (1|suj), data=crowdf)
me02c <- lmer(length ~ ADHD + genre + age + (1|suj), data=crowdf)
me03c <- lmer(interv ~ ADHD + genre + age + (1|suj), data=crowdf)

me04 <- lmer(ADHD ~ length + interv + (1|suj), data=crowdf) # does not converge

# ADHD divided into first 9 and last 9 questions, no MEWS
me05 <- lmer(speed ~ ADHD.first + ADHD.last + (1|suj), data=crowdf)
me06 <- lmer(length ~ ADHD.first + ADHD.last + (1|suj), data=crowdf) #!!!
me07 <- lmer(interv ~ ADHD.first + ADHD.last + (1|suj), data=crowdf)
me05c <- lmer(speed ~ ADHD.first + ADHD.last + genre + age + (1|suj), data=crowdf)
me06c <- lmer(length ~ ADHD.first + ADHD.last + genre + age + (1|suj), data=crowdf)
me07c <- lmer(interv ~ ADHD.first + ADHD.last + genre + age + (1|suj), data=crowdf)

# Pause predictions. We only seem to be able to do it by masking out
# the 15% of transitions which have null pauses
me08 <- lmer(pause ~ ADHD.first + ADHD.last + (1|suj), data=crowdf)
me09 <- lmer(pause ~ ADHD + (1|suj), data=crowdf)
crowdf_masked <- filter(crowdf, pause > 0)
me10 <- lmer(pause ~ ADHD.first + ADHD.last + (1|suj), data=crowdf_masked)
me11 <- lmer(pause ~ ADHD.first + ADHD.last + genre + age + (1|suj), data=crowdf_masked)

# MEWS does not seem to predict anything
me12 <- lmer(length ~ ADHD + MEWS + (1|suj), data=crowdf)
me13 <- lmer(interv ~ ADHD + MEWS + (1|suj), data=crowdf)
me14 <- lmer(speed ~ ADHD + MEWS + (1|suj), data=crowdf)

# -------------
# We want to:
# 1. "Keep it maximal", i.e. fit the most complex model consistent with
# experimental design that does not result in a singular fit
# 2. Eventually compare models through some criterion like AIC or BIC
