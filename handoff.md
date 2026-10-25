# Project Approach

## Goal

Develop a deep-learning system that analyzes the defensive positioning of a football back four and predicts whether a dangerous attacking situation is likely to occur within the next three seconds.

The initial goal is **risk prediction**, not finding a universally perfect defensive formation.

## Approach

1. **Select a data source**
   - Start with available football tracking data containing player and ball coordinates over time.
   - Check data access, licensing, timestamps, coordinate systems, and whether player roles can be identified.
   - Prefer tracking data over raw video for the first prototype to keep the scope manageable.

2. **Prepare the data**
   - Standardize pitch coordinates and attacking direction.
   - Identify the four defenders and relevant opposing attackers.
   - Handle missing or invalid observations.
   - Calculate useful spatial features, such as distances between defenders, defensive-line width and depth, player velocities, and distances to the ball and goal.

3. **Create training examples and labels**
   - Use a short history window (initially around three seconds) as the model input.
   - Define a measurable dangerous event, such as an attacker entering a predefined space behind the defensive line.
   - Label each example `1` if the event occurs within the following three seconds and `0` otherwise.
   - Review a sample of labels with football knowledge. Do not assume that a large gap automatically means a defensive mistake.

4. **Train and compare models**
   - Establish a baseline with a small multilayer perceptron (MLP) using engineered spatial features.
   - Train a GRU or LSTM to learn movement patterns over time.
   - If the data and project time allow, implement a spatiotemporal graph neural network (GNN + GRU), representing players and the ball as nodes and their relationships as edges.
   - Compare the models on the same data splits to determine whether the more complex model adds value.

5. **Evaluate carefully**
   - Split data by complete matches, not randomly by individual frames, to reduce data leakage.
   - Report precision, recall, F1-score, and PR-AUC; consider the Brier score if probability quality is important.
   - Inspect false positives and false negatives and visualize representative examples.
   - With only a few matches, describe results as exploratory and avoid claiming that the model generalizes to all teams or competitions.

6. **Visualize results**
   - Display the defensive line, attackers, ball, and predicted risk on a top-down pitch.
   - Explain that the prediction indicates risk based on learned patterns; it is not proof that a defender made a mistake or that a particular position is optimal.

## Initial Scope

Prioritize a working pipeline from tracking data to labeled examples, an MLP baseline, a temporal model, and an evaluation. Treat the GNN and interactive platform as extensions after the baseline works.

## Key Principle

The model may only use information available at prediction time. Future observations can be used to create training labels, but must never be included in the model input.
