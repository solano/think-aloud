# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 13:35:28 2021

Cleans up oral markers, then spellchecks data_stream_text_time.csv,
producing the files text_rows and text_probes.csv.

@author: Solano Felicio
"""

import pandas as pd
import re
import pyparsing as pp
import language_tool_python
import string

# %% Import data

srcfile = "data_stream_text_time.csv"
data = pd.read_csv(srcfile, sep='\t')
# Note: there are some NaN lines.

# %% To skip processing: import clean data

outfile_rows = "text_rows.csv"
outfile_subrows = "text_subrows.csv"
outfile_probes = "text_probes.csv"
data2 = pd.read_csv(outfile_rows, sep="\t")
data4 = pd.read_csv(outfile_subrows, sep="\t")
data3 = pd.read_csv(outfile_probes, sep="\t")

# %% Transform NAs into emtpy strings
cleanNAs = lambda s: "" if pd.isna(s) else s
data.SPEECH = data.SPEECH.apply(cleanNAs)

# %% Sort data

probes = data.groupby(['suj','bloc','prob'], group_keys=False)
data = probes.apply(lambda g: g.sort_values(by='start_time'))
data = data.reset_index(drop=True)
probes = data.groupby(['suj','bloc','prob'], group_keys=False)

# %% Define tools to treat oral markers

# --- The oral markers ---
#
# -- The ones we treat with regular expressions --
# Prolonged syllables: "et après: j'ai pensé" -> suppress ":"
# Hesitation, fillers: "&-euh &=bah &-mmh" -> suppress all these
# Vowel deletion: "j(e)veux c(e) qu(e)" -> suppress parentheses, add whitespace
# Other parentheses: -> suppress parentheses
#
# -- The ones we treat with a context-free grammar defined below --
# Mistakes: "<je veux>[//] je voudrais" -> suppress mistake
# Repetitions: "et puis <j'ai>[x3] fini" -> suppress marker, leave word
# Short pauses: Alors (.) après tout ça -> suppress marker or cut up string,
#                                          depending on level of segmentation

# This function takes a strings and returns a "clean" version,
# without markers of orality.
# If split is True, it returns a list of clean strings, separated by
# the small pauses marked by (.).
# Examples:
# cleanup("&-euh et p(u)is, (.) j(e)me suis dit:")
# -> 'et puis, (.) je me suis dit'
# cleanup("(en)fin avec <<le:> [x2]>[//] <la>[x2] polenta")
# -> 'enfin avec la polenta'
# cleanup("&-bah")
# -> ''
def cleanup(s, split=False):
    s = s.replace(":","")               # prolonged syllables
    s = re.sub(r"\&[-=]\w+", "", s)     # hesitation, fillers
    
    # The split parameter decides whether this function returns
    # a simple clean string or a list of clean strings, separated on
    # the short pauses marked by (.).
    # We do this by simple str.split().
    # This is a naive procedure, but it does work on the data we have.
    # It could have broken if there were short pauses inside structures
    # parsed by the grammar. We could also in principle get multiple levels
    # of recursion, but it didn't happen.
    if split:
        s_tuple = s.split("(.)")
        s_tuple = [cleanup(si) for si in s_tuple] # divide and conquer
        return s_tuple
    else:
        s = s.replace("(.)", "")            # remove small pause
    
        # clean up parentheses. First, all those (e) without
        # the necessary whitespace in front i.e. j(e)veux -> je veux
        s = re.sub(r"\b(j|c|m|d|qu)\(e\)[ ]?", r"\1e ", s, flags=re.IGNORECASE)
        s = s.replace("(", "")
        s = s.replace(")", "") # all remaining parentheses
        
        # Word repetitions and mistakes (<...>[..] expressions)
        # are parsed by the grammar defined below (expr) because
        # they may be nested.
        parsed = expr.searchString(s).as_list()
        if parsed:
            s = parsed[0][0]
        else:
            s = ""
        return s

# (Arbitrarily) nested expressions cannot be parsed with regexp.
# They need context-free grammars.

# Here I define a very simple one to deal with:


# In the 
# pyparsing also nicely deals with whitespace so I don't have to worry
# about it.

langle = pp.Suppress("<")
rangle = pp.Suppress(">")
mistakebracket = pp.Literal("[//]")
repbracket = pp.Suppress("[") + pp.Word("x"+pp.nums) + pp.Suppress("]")
pword = pp.Word(pp.alphanums+pp.alphas8bit+",.;'!?") # word with punctuation

expr = pp.Forward()

mistake = langle + expr + rangle + mistakebracket
#mistake.setParseAction(lambda s, l, t : print("Mistake:", t[0])) #debug
mistake = pp.Suppress(mistake) # suppress mistake from results

repetition = langle + expr + rangle + repbracket
#repetition.setParseAction(lambda s, l, t: print(f"Repetition {t[1]}:", t[0])) #debug
repetition.setParseAction(lambda t: t[0]) # remove repetition marker

expr <<= (mistake | repetition | pword)[...]
expr.setParseAction(lambda t: " ".join(t))


# %% Clean up oral markers

corrected = data.SPEECH.map(cleanup)
corrected_split = data.SPEECH.map(lambda s: cleanup(s, split=True))

# TODO: Make each string take one line of the dataframe
# solution for now: same thing but not taking other columns into account

split_list = []
for index, phrases in corrected_split.iteritems():
    for phrase in phrases:
        split_list.append((index, phrase))
corrected_split = pd.DataFrame(data = split_list,
                               columns = ['ind','phrase'])


# %% Spellcheck row-level phrases

# This took 32 minutes to run.
# After spellchecking, some mistakes remain, and many others are
# generated by the spellchecker itself. However, most of the
# data seems to be improved by doing this.

tool = language_tool_python.LanguageTool('fr-FR')
corrected2 = corrected.map(tool.correct)

# Undo first-letter uppercasing done by Language Tool
temp = []
for s1, s2 in zip(corrected, corrected2):
    s = ''
    if s1 != '' and s1[0] not in string.ascii_uppercase:
        s = s2[0].lower() + s2[1:]
    else:
        s = s2
    temp.append(s)
corrected2 = pd.Series(temp)

# %% Spellcheck subrow-level phrases

corrected_splitphrase2 = corrected_split.phrase.map(tool.correct)
# Takes about 40 minutes

# Undo first-letter uppercasing done by Language Tool
temp = []
for s1, s2 in zip(corrected_split.SPEECH, corrected_splitphrase2):
    s = ''
    if s1 != '' and s1[0] not in string.ascii_uppercase:
        s = s2[0].lower() + s2[1:]
    else:
        s = s2
    temp.append(s)
corrected_splitphrase2 = pd.Series(temp)

corrected_split2 = corrected_split
corrected_split2.phrase = corrected_splitphrase2

data2 = data
data2.SPEECH = corrected2

# %% Prepare subrow data for statistical analysis

data4 = data.merge(corrected_split2, how="right", left_index=True, right_on="ind")
data4 = data4[['suj','bloc','prob','start_time','SPEECH','phrase']]

# Will be unecessary once we sort the original data before spellchecking
probes4 = data4.groupby(['suj','bloc','prob'], group_keys=False)
data4 = probes4.apply(lambda g: g.sort_values(by='start_time'))

#%%
# Eliminate unnecessary info, rename column for uniformity of code
data4 = data4[['suj','bloc','prob','phrase']]
data4 = data4.rename(columns={'phrase': 'SPEECH'})

# %% Write row and subrow segmented text to CSV

#outfile_rows = "text_rows.csv"
#data2.to_csv(outfile_rows, sep="\t", index=False)

outfile_subrows = "text_subrows.csv"
data4.to_csv(outfile_subrows, sep="\t", index=False)

# %% Lump phrases together to get probe-level segmentation

# Use this line to avoid rerunning the row-level processing above
data2 = pd.read_csv(outfile_rows, sep="\t")

probes = data2.groupby(['suj','bloc','prob'], group_keys=False)

# Joins probe.SPEECH into a single string
# Assumes probe.SPEECH only contains non-null phrases
def probe_to_text(probe):
    # Only nonempty rows
    indexes = (probe.SPEECH.isna() == False)
    phrases = probe.SPEECH[indexes]
    return " ".join(phrases)

probe_text = probes.apply(probe_to_text)

# TODO: include start and end times for probe?

# %% Write probe-segemented text to CSV

data3 = probe_text.reset_index()
data3 = data3.rename(columns={0: "SPEECH"})
outfile_probes = "text_probes.csv"
data3.to_csv(outfile_probes, sep="\t", index=False)

# %% Find oral markers on str_speech
# This is useful to find sentences containing complicated notation
# which can be used to test the cleanup function.
# After performing cleanup all these become empty.

str_series = corrected_split.SPEECH

dd = str_series[[str(st).find("<")!=-1 for st in str_series]]
ddr = str_series[[str(st).find(">")!=-1 for st in str_series]]
dde = str_series[[str(st).find("[//]")!=-1 for st in str_series]]
dds = str_series[[str(st).find("&")!=-1 for st in str_series]]
ddss = str_series[[str(st).find("&")!=-1 and str(st).find("&-euh")==-1 for st in str_series]]
ddp = str_series[[str(st).find("(")!=-1 for st in str_series]]
ddpr = str_series[[str(st).find(")")!=-1 for st in str_series]]
ddpp = str_series[[re.search("\([^.]*\)", str(st))!=None for st in str_series]]
ddpe = str_series[[str(st).find("(e)")!=-1 for st in str_series]]
ddpne = str_series[[re.search(r"\((?![e.])\w+\)", str(st))!=None for st in str_series]]
ddc = str_series[[str(st).find("[")!=-1 for st in str_series]]
ddcr = str_series[[str(st).find("]")!=-1 for st in str_series]]
ddcc = str_series[[str(st).find("[x")!=-1 for st in str_series]]