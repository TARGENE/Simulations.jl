function make_number(nt)
    number = 0.
    for n in keys(nt)
        number += nt[n] + rand()
    end
    return number
end

function bad_loop(variants)
    numbers = Vector{Float64}(undef, size(variants, 1))
    for (index, v) in enumerate(variants)
        nt = NamedTuple{(v,)}([rand()])
        numbers[index] = make_number(nt)
    end
    return numbers
end

function bad_loop_2(variants)
    numbers = Vector{Float64}(undef, size(variants, 1))
    nts = [NamedTuple{(v,)}([rand()]) for v in variants]
    for (index, nt) in enumerate(nts)
        numbers[index] = make_number(nt)
    end
    return numbers
end

variants = [Symbol("rs", i) for i in rand(Int, 100)]
@time bad_loop_2(variants)

function better_loop(variants)
    numbers = Vector{Float64}(undef, size(variants, 1))
    for (index, v) in enumerate(variants)
        nt = Dict(v => rand())
        numbers[index] = make_number(nt)
    end
    return numbers
end

variants = [Symbol("rs", i) for i in rand(Int, 100)]
@time bad_loop(variants)
@time better_loop(variants)