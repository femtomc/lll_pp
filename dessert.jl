module PP_Dessert

# A dish with a gooey deep learning center.

#####
##### MNIST data loader
#####

import Random
Random.seed!(1)

import MLDatasets
train_x, train_y = MLDatasets.MNIST.traindata()

mutable struct DataLoader
    cur_id::Int
    order::Vector{Int}
end

DataLoader() = DataLoader(1, Random.shuffle(1 : 60000))

function next_batch(loader::DataLoader, batch_size)
    x = zeros(Float64, 28, 28, 1, batch_size)
    y = Vector{Int}(undef, batch_size)
    for i = 1 : batch_size
        x[:, :, 1, i] = train_x[:, :, loader.cur_id]
        y[i] = train_y[loader.cur_id] + 1
        loader.cur_id = (loader.cur_id % 60000) + 1
    end
    x, y
end

function load_test_set()
    test_x, test_y = MLDatasets.MNIST.testdata()
    N = length(test_y)
    x = zeros(Float64, 28, 28, 1, N)
    y = Vector{Int}(undef, N)
    for i = 1 : N
        x[:, :, 1, i] = test_x[:, :, i]
        y[i] = test_y[i] + 1
    end
    x, y
end

const loader = DataLoader()

test_x, test_y = load_test_set()

#####
##### Model (hybrid PP + deep)
#####

using Gen
using Flux
using GenFlux
using Random

g = @genflux Chain(Conv((5, 5), 1 => 10),
                   MaxPool((2, 2)),
                   x -> relu.(x),
                   Conv((5, 5), 10 => 20),
                   x -> relu.(x),
                   MaxPool((2, 2)),
                   x -> flatten(x),
                   Dense(320, 50),
                   Dense(50, 10),
                   softmax) |> Flux.f64
display(g)

# Probabilities drawn from deep network, used to parametrized a categorical distribution on labels directly in Gen.
@gen function f(xs::Vector{Float64})
    probs ~ g(xs)
    [{:y => i} ~ categorical(p |> collect) for (i, p) in enumerate(eachcol(probs))]
end

#####
##### Learning
#####

update = ParamUpdate(Flux.ADAM(1e-4, (0.9, 0.999)), g)
for i = 1 : 3500
    # Create trace from data
    (xs, ys) = next_batch(loader, 100)
    constraints = choicemap([(:y => i) => y for (i, y) in enumerate(ys)]...)
    (trace, weight) = generate(f, (xs,), constraints)

    # Increment gradient accumulators
    accumulate_param_gradients!(trace)

    # Perform ADAM update and then resets gradient accumulators
    apply!(update)
    println("i: $i, weight: $weight")
end

#####
##### Sample inference (just run model forward)
#####

using Statistics: mean
test_accuracy = mean(f(test_x) .== test_y)
println("Test set accuracy: $test_accuracy")

end # module
