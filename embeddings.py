# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 13:39:14 2022

Uses preprocessed data to produce trajectories in embedding space. Each
row of text is made into an embedding. Analyses speed profile of such trajectories.

@author: Solano Felicio
"""

# %% Import modules

from scipy import spatial
from laserembeddings import Laser
import pandas as pd
#import matplotlib.pyplot as plt

# %% Create LASER object

laser = Laser()

# %% Read data

# --- Experience data in original segmentation level ---
srcfile = "text_rows.csv"
df_rows = pd.read_csv(srcfile, sep="\t")

# Group row-level data into probes
probes = df_rows.groupby(['suj','bloc','prob'], group_keys=False)

# --- Experience data segmented at probe level ---

probesrcfile = "text_probes.csv"
df_probes = pd.read_csv(probesrcfile, sep="\t")

# Group probe-level data into blocks
blocks = df_probes.groupby(['suj','bloc'], group_keys=False)

# --- Experience data segmented at subrow level ---
subrowsrcfile = "text_subrows.csv"
df_subrows = pd.read_csv(subrowsrcfile, sep="\t")

# Group subrow-level data into probes
probes_sub = df_subrows.groupby(['suj','bloc','prob'], group_keys=False)

# --- Subject data ---
subj_srcfile = "info_participants.csv"
subj_data = pd.read_csv(subj_srcfile, sep="\t")

# Correct some entries which are lowercase, replace 'H' -> 'M', simplify names
subj_data.genre = subj_data.genre.str.upper()
subj_data.genre = subj_data.genre.apply(lambda s: 'M' if s=='H' else s)
subj_data = subj_data.rename(columns={"total-MEWS":"MEWS"})

# %% How many rows per subject?
# Includes empty rows

#sujs = df_rows.groupby(['suj'], group_keys=False)
#nb_rows_per_subj = sujs.apply(len)
#plt.hist(nb_rows_per_subj)
#plt.title("Number of rows per subject")
#plt.show()
#print(nb_rows_per_subj.describe())


# %% Statistics of probe length
# Includes empty rows

#probelength = probes.apply(len)
#plt.hist(probelength, bins=10, range=(0,10))
#plt.title("Probe length")
#plt.show()
#nb_probes = len(probes)
#nb_large_probes = len(probelength[probelength > 1])
#print(f"There are {nb_large_probes} probes with more than a row, out of {nb_probes} total")

# %% Define useful functions

# Cosine distance between vectors or arrays of vectors
dist = spatial.distance.cosine

# Computes speed based on spatialtemporal trajectory
# vecs : array of vectors in embedding-space, size N > 1
# dt : array of time intervals, size N-1
# Returns array of size N-1 representing speed.
# If times are not specified, assumes Δt = 1
def trajectory_speed(vecs, dt=1):
    N = len(vecs)
    if N <= 1: return None
        
    dx = pd.array([dist(vecs[i], vecs[i+1]) for i in range(N-1)])
        
    return dx/dt

# %% Embedding at row level

row_embeddings = laser.embed_sentences(df_rows.SPEECH, lang="fr")
# Runtime: 1 min 20 s

# %% Embedding at probe level

probe_embeddings = laser.embed_sentences(df_probes.SPEECH, lang="fr")
# Runtime: 1 min 30 s

# %% Embedding at subrow level

subrow_embeddings = laser.embed_sentences(df_subrows.SPEECH, lang="fr")
# Runtime: 1 min

# %% Analysis of transitions at row level

# Here each nonempty row of text is considered a different phrase
# to be embedded. "Speed" is defined as the quotient of distance
# between consecutive rows and the time elapsed between them.
# Note that we don't know the time between probes, so there is
# one trajectory per probe (thus, multiple trajectories per subject).

# For each probe, compute trajectory jump lengths, intervals and speeds
def probe_to_trajectory(probe):
    # Only nonempty rows
    indexes = (probe.SPEECH.isna() == False)
    
    vecs = row_embeddings[probe.index][indexes]
    
    # Choose time for each row as average of start and end times
    start_time = probe.start_time[indexes].reset_index(drop=True)
    end_time = probe.end_time[indexes].reset_index(drop=True)
    time = (start_time + end_time)/2
    
    interv = pd.array([time[i+1]-time[i] for i in range(len(time)-1)])
    pause = pd.array([start_time[i+1]-end_time[i] for i in range(len(time)-1)])
    
    jumps = trajectory_speed(vecs)
    speed = trajectory_speed(vecs, interv)
    
    if jumps is None:
        return None # null trajectory
    
    return list(zip(jumps, interv, pause, speed))

trajectories_rows = probes.apply(probe_to_trajectory)

# Delete null trajectories
trajectories_rows = trajectories_rows[trajectories_rows.isnull()==False]

#nb_trajectories_rows = len(trajectories_rows)
#print(f"There are {nb_trajectories_rows} non-null trajectories out of {nb_probes}")


# Separate transitions as units

transitions_rows = []

for index,traj in trajectories_rows.iteritems():
    for datatraj in traj:
        transitions_rows.append((*index, *datatraj))
        
transitions_rows = pd.DataFrame(data = transitions_rows,
                           columns=['suj','bloc','prob','length','interv',
                                    'pause','speed'])

#nb_transitions_rows = len(transitions_rows)
#print(f"There are {nb_transitions_rows} transitions at row level")

# %% Prepare and export row-level transitions for statistical analysis in R

df = transitions_rows.merge(subj_data, how='left', left_on='suj', right_on='sujet')

first_questions = df[[f"ADHD-{n}" for n in range(1,10)]].astype('int').sum(1)
last_questions = df[[f"ADHD-{n}" for n in range(10,19)]].astype('int').sum(1)

df = df[['suj','bloc','prob','length','interv','pause','speed',
         'age','genre','exp','level','topic','ADHD','MEWS']]
df.insert(len(df.columns)-1, 'ADHD-first', first_questions)
df.insert(len(df.columns)-1, 'ADHD-last', last_questions)


# Sum of 9 first ADHD questions = impulsivity
# last 9 = inattention (soit ça soit l'inverse)

df.to_csv("row_as_embedding_transitions.csv", sep='\t', index=False)

# %% Trajectory length statistics

#trajectory_length = trajectories.map(len)
#plt.hist(trajectory_length, bins=25)
#plt.title("Trajectory length at row level")
#plt.show()

#print(trajectory_length.describe())

# %% Analysis of transitions at probe level

# Here each probe gives a phrase. There is no temporal data,
# so we compute only jump lengths between probes. Each block gives
# a trajectory in embedding space.

# For each block, compute trajectory jump lengths
def block_to_trajectory(block):
    # Only nonempty phrases
    indexes = (block.SPEECH.isna() == False)
    vecs = probe_embeddings[block.index][indexes]
        
    jumps = trajectory_speed(vecs)
    
    if jumps is None:
        return None # null trajectory
    
    return jumps

trajectories_prob = blocks.apply(block_to_trajectory)

# Delete null trajectories
trajectories_prob = trajectories_prob[trajectories_prob.isnull()==False]

# Separate transitions as units

transitions_prob = []

for index,traj in trajectories_prob.iteritems():
    for jumplength in traj:
        transitions_prob.append((*index, jumplength))
        
transitions_prob = pd.DataFrame(data = transitions_prob,
                           columns=['suj','bloc','length'])

# %% Prepare and export probe-level transitions for statistical analysis in R

df2 = transitions_prob.merge(subj_data, how='left', left_on='suj', right_on='sujet')

first_questions = df2[[f"ADHD-{n}" for n in range(1,10)]].astype('int').sum(1)
last_questions = df2[[f"ADHD-{n}" for n in range(10,19)]].astype('int').sum(1)

df2 = df2[['suj','bloc','length',
         'age','genre','exp','level','topic','ADHD','MEWS']]
df2.insert(len(df2.columns)-1, 'ADHD-first', first_questions)
df2.insert(len(df2.columns)-1, 'ADHD-last', last_questions)


# Sum of 9 first ADHD questions = impulsivity
# last 9 = inattention (soit ça soit l'inverse)

df2.to_csv("probe_as_embedding_transitions.csv", sep='\t', index=False)

# %% Analysis of transitions at subrow level

# Here each row of original data gives multiple phrases. There is no
# temporal data, so we compute only jump lengths. Each probe gives
# a trajectory in embedding space.

# For each probe, compute trajectory jump lengths
def probe_sub_to_trajectory(probe):
    # Only nonempty phrases
    indexes = (probe.SPEECH.isna() == False)
    vecs = subrow_embeddings[probe.index][indexes]
        
    jumps = trajectory_speed(vecs)
    
    if jumps is None:
        return None # null trajectory
    
    return jumps

trajectories_sub = probes_sub.apply(probe_sub_to_trajectory)

# Delete null trajectories
trajectories_sub = trajectories_sub[trajectories_sub.isnull()==False]

# Separate transitions as units

transitions_sub = []

for index,traj in trajectories_sub.iteritems():
    for jumplength in traj:
        transitions_sub.append((*index, jumplength))
        
transitions_sub = pd.DataFrame(data = transitions_sub,
                           columns=['suj','bloc','prob','length'])

# %% Prepare and export subrow-level transitions for statistical analysis in R

df3 = transitions_sub.merge(subj_data, how='left', left_on='suj', right_on='sujet')

first_questions = df3[[f"ADHD-{n}" for n in range(1,10)]].astype('int').sum(1)
last_questions = df3[[f"ADHD-{n}" for n in range(10,19)]].astype('int').sum(1)

df3 = df3[['suj','bloc','prob','length',
         'age','genre','exp','level','topic','ADHD','MEWS']]
df3.insert(len(df3.columns)-1, 'ADHD-first', first_questions)
df3.insert(len(df3.columns)-1, 'ADHD-last', last_questions)


# Sum of 9 first ADHD questions = impulsivity
# last 9 = inattention (soit ça soit l'inverse)

df3.to_csv("subrow_as_embedding_transitions.csv", sep='\t', index=False)