# Mapping on Beluga Cluster

This is a simple example for training a landcover mapping model on the beluga
cluster. It only uses the first 10 training and validation examples, just so you
can see what the overall workflow looks like.

# How to Run

Clone this repository into your space on the computer cluster. Modify the comet
username in `src/exper.py`, and update the paths to the cloned repo / logging
directories in `cluster/mapping.sbatch`. At that point, you should be able to
run `sbatch mapping.sbatch` from inside the cluster folder, and you should end
up with a couple of saved models after a few minutes.
