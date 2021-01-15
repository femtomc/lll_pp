module PP_HorsDoeuvres

# A nice sample.

using Gen
using Jaynes

#####
##### Syntax breakdown
#####

# A ~ B : denotes a random choice. 
# This desugars to: A = trace(:A, B) which is a special call that interacts with PP system (it _carries_ the PP semantics if you're familiar with effects systems).
# (DefaultPipeline, etc) : denotes compilation and typing options.

#####
##### Example - burglary
#####

model = @jaynes function burglary_model()::Nothing
    burglary ~ Bernoulli(0.05)
    disabled ~ Bernoulli(0.03)
    !disabled ? alarm ~ Bernoulli(burglary ? 0.94 : 0.01) : alarm ~ Bernoulli(0.01)
    call ~ Bernoulli(alarm ? 0.70 : 0.05)
    return nothing
end (DefaultPipeline) # you get a concrete representation of this model when you compile it as a PP.

display(model) # TODO: Ask/explore: what does this metadata mean?

# Now what can we do?

# 1. We can sample from the model.
tr = simulate(model, ()) # TODO: Ask/explore: what is a trace?
display(get_choices(tr))

# 2. We can sample from the model _with constraints_.
chm = choicemap(:disabled => true)
tr, _ = generate(model, (), chm)
display(get_choices(tr))

# 3. We can do inference given some set of constraints.
chm = choicemap((:call, true)) # constraint.

trs, lnws, lmle = begin
    @time importance_sampling(model, (), chm, 5000) # do inference.
end

est = sum(map(zip(lnws, trs)) do (lnw, tr)
              (tr[:burglary] ? 1 : 0) * exp(lnw)
          end) # compute an estimator.
println(est) # Probability that a burglary occurred.

end # module
