# Federated runs (FedAvg and FedProx checkpoints)

## Setup

After standard training, the Dataset A notebook ran a "federated" experiment: 5 clients, 12 rounds, 3 local epochs per round, learning rate 3e-4, and FedProx proximal term mu = 0.01. The 5 clients are contiguous slices of the 640-image training split (128 images per client), so they are random shards of the same data, not different hospitals or cameras.

## Important caveat: not true federated averaging

The five FLClient objects are created with the same model object (FLClient(global_model, ...)). Copying the global weights into a client therefore copies the model onto itself, the clients train the same model one after another, and the FedAvg step averages five identical copies, which changes nothing. In practice each round is sequential training over the five shards. For FedProx, the proximal anchor is taken from the model's current weights at the start of each client's turn, so it anchors to the previous client's result, not to the weights at the start of the round.

The FedAvg and FedProx checkpoints should therefore be described as "sequential shard training labelled FedAvg/FedProx". Differences between them are not evidence that one federated algorithm is better. Fixing this needs one deep copy of the global model per client and a snapshot of the global weights at the start of each round, then re-running and re-saving.

## Results reported in the notebook

Dice after round 12: FedAvg validation 0.9226, test 0.9163. FedProx validation 0.9167, test 0.9096. For comparison, the standard (centralised) final-epoch model has test Dice 0.9295. Both federated variants are below standard training.
