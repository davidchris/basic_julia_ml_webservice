using HTTP
include("./Clustering/src/kmeans.jl")

centroids = [[-1.24429, -1.26819], [10.3804, 10.2753], [15.1108, 0.029277]]

function findNearestCentroid(centroids, dataPoint)
    distances = []
    for centroid in centroids
        push!(distances, calcDist(centroid, dataPoint))
    end
    return argmin(distances)
end    

# twoTimes = function(x)
#     return 2 * x
# end

HTTP.listen() do request::HTTP.Request 
    body = parse.(Float64, split(String(request.body), ";"))
    try 
        return HTTP.Response(string(findNearestCentroid(centroids, body)))
    catch e 
        return HTTP.Response(404, "Error: $e")
    end
end
