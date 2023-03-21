import torch
import pygad.torchga

# Define the PyTorch model.
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=128, embedding_dim=10)
        self.rnn = torch.nn.RNN(input_size=10, hidden_size=20, num_layers=1, batch_first=True)
        self.linear = torch.nn.Linear(in_features=20, out_features=128)

    def forward(self, inputs):
        x = self.embedding(inputs)
        x, _ = self.rnn(x)
        x = self.linear(x)
        return x

    @property
    def num_weights(self):
        return sum(p.numel() for p in self.parameters())

# Define the fitness function.
def fitness_func(solution, sol_idx):
    # Convert the solution to PyTorch tensors.
    weights = torch.tensor(solution, dtype=torch.float32)

    # Set the weights of the model.
    model.set_weights(weights)

    # Compute the loss.
    inputs, targets = dataset
    outputs = model(inputs)
    loss = torch.nn.functional.cross_entropy(outputs.view(-1, 128), targets.view(-1))

    # Return the negative loss (since PyGAD minimizes the fitness function).
    return -loss.item()

# Define the dataset.
with open("names.txt", "r") as f:
    names = f.read().splitlines()
inputs = [[ord(c) for c in name[:-1]] for name in names]
targets = [[ord(c) for c in name[1:]] for name in names]
inputs = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in inputs], batch_first=True)
targets = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in targets], batch_first=True)

# Create the PyTorch model.
model = Model()

# Create the PyGAD optimizer.
num_solutions = 10
num_weights = model.num_weights
optimizer = pygad.torchga.TorchGA(model=model,
                                  num_solutions=num_solutions,
                                  num_parents_mating=5,
                                  initial_population_range=[-1, 1],
                                  mutation_probability=0.01,
                                  fitness_func=fitness_func)

# Optimize the model.
num_generations = 100
for i in range(num_generations):
    optimizer.run_iteration()

# Get the best solution.
best_solution, best_solution_fitness = optimizer.best_solution()

# Set the weights of the model to the best solution.
model.set_weights(torch.tensor(best_solution, dtype=torch.float32))

# Generate some text.
with torch.no_grad():
    inputs = torch.tensor([[ord("a"), ord("b"), ord("c")]], dtype=torch.long)
    for i in range(10):
        outputs = model(inputs)
        _, predicted = torch.max(outputs, dim=2)
        print(chr(predicted.item()), end="")
        inputs = torch.cat([inputs[:, 1:], predicted], dim=1)
