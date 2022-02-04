# Load packages -----
library(tidyverse)
library(lme4)
library(lmerTest)

# Load data -------
probdf <- read.csv('C:\\Users\\User\\Drive USP\\école\\2a\\p2\\modal\\python\\probe_as_embedding_transitions.csv',
                 sep='\t', fileEncoding='utf8')

# Standardize data for analysis -------

c.length <- probdf$length - 0.5                # between -0.5 and 0.5
c.age <- (probdf$age - mean(probdf$age[!is.na(probdf$age)])) # in years
c.ADHD <- (probdf$ADHD - mean(probdf$ADHD))/5   # in units of 5 ADHD points
c.ADHD.first <- (probdf$ADHD.first - mean(probdf$ADHD.first))/5
c.ADHD.last <- (probdf$ADHD.last - mean(probdf$ADHD.last))/5
c.MEWS <- (probdf$MEWS - mean(probdf$MEWS))/5 # units of 5 MEWS points

cprobdf = data.frame(suj = probdf$suj,
                   bloc = probdf$bloc,
                   length = c.length,
                   age = c.age,
                   genre = probdf$genre,
                   exp = probdf$exp,
                   level = probdf$level,
                   topic = probdf$topic,
                   ADHD = c.ADHD,
                   ADHD.first = c.ADHD.first,
                   ADHD.last = c.ADHD.last,
                   MEWS = c.MEWS)

# Fixed effects models ------

fe01 <- lm(length ~ ADHD, data=cprobdf)

# Mixed effects models ------

me01 <- lmer(length ~ ADHD + (1|suj), data=cprobdf)
me01c <- lmer(length ~ ADHD + genre + age + (1|suj), data=cprobdf)

me02 <- lmer(ADHD ~ length + (1|suj), data=cprobdf) # does not converge

me03 <- lmer(length ~ ADHD.first + ADHD.last + (1|suj), data=cprobdf)
me03c <- lmer(length ~ ADHD.first + ADHD.last + genre + age + (1|suj), data=cprobdf)

# MEWS here actually does predict something !
me04 <- lmer(length ~ ADHD + MEWS + (1|suj), data=cprobdf)
me04c <- lmer(length ~ ADHD + MEWS + age + genre + (1|suj), data=cprobdf)
me05 <- lmer(length ~ ADHD.first + ADHD.last + MEWS + (1|suj), data=cprobdf)
me05c <- lmer(length ~ ADHD.first + ADHD.last + MEWS + age + genre + (1|suj), data=cprobdf)

# ----------
# We want to:
# 1. "Keep it maximal", i.e. fit the most complex model consistent with
# experimental design that does not result in a singular fit
# 2. Eventually compare models through some criterion like AIC or BIC
