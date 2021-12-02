module JsonUtils

    using JSON

    function df2json(df)
        dictvector = Vector{Dict}()
        ctype = eltype.(eachcol(df));
        for row in eachrow(df)
            dict = Dict();
            for i in 1:length(names(df))
                if(ctype[i] === "String")
                    push!(dict, names(df)[i] => string(row[i]))
                else
                    push!(dict, names(df)[i] => row[i])   
                end
            end
            push!(dictvector, dict)
        end
        return JSON.json(dictvector)
    end

end