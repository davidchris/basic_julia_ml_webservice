include("./Clustering/src/kmeans.jl")

using DataFrames
using Distributions
using Plots
gr()
using LinearAlgebra

function makeData()
    # groupOne = rand(MvNormal([10.0, 10.0], 10.0 + I), 100)
    # groupTwo = rand(MvNormal([0.0, 0.0], 10 + I), 100)
    # groupThree = rand(MvNormal([15.0, 0.0], 10.0 + I), 100)
    groupOne = rand(MvNormal([10.0, 10.0], fill(10.0, (2, 2)) + I), 100)
    groupTwo = rand(MvNormal([0.0, 0.0], fill(10.0, (2, 2)) + I), 100)
    groupThree = rand(MvNormal([15.0, 0.0], fill(10.0, (2, 2)) + I), 100)
    return hcat(groupOne, groupTwo, groupThree)'
end

data = makeData()

scatter(data[1:100, 1], data[1:100, 2], color = "blue")
scatter!(data[101:200, 1], data[101:200, 2], color = "red")
scatter!(data[201:300, 1], data[201:300, 2], color = "green")
# png("scatterPlot")

result = kMeans(DataFrame(data), 3)

propertynames(result)

println(result.estimatedClass)

print(result.centroids[end])

one = rand(MvNormal([10.0, 10.0], fill(10.0, (2, 2)) + I), 1)
two = rand(MvNormal([0.0, 0.0], fill(10.0, (2, 2)) + I), 1)
three = rand(MvNormal([15.0, 0.0], fill(10.0, (2, 2)) + I), 1)

@show(one)
@show(two)
@show(three)