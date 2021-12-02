module DfUtils

    using DataFrames, CSV, StatsBase, Statistics, MLJ, Lathe, GLM;
    using Lathe.preprocess: TrainTestSplit;

    function generatedataframe(filename::String, numrows::Int64, missingcolname)
        if(numrows == -1)
            df = generatedataframe(filename);
        else
            df = generaterandomdataframe(filename, numrows);
        end

        fixcolumnnames(df);

        if(missingcolname !== nothing)
            dropamissingcolumn(df, missingcolname);
        end
        return df;
    end

    function generatedataframe(fname)
        df = DataFrame(CSV.File(fname));
    end

    function generaterandomdataframe(fname, n)
        df = generatedataframe(fname);
        return df[sample(axes(df, 1), n; replace = false, ordered = true), :];
    end    

    function generaterandomdatafromdataframe(df, n)
        return df[sample(axes(df, 1), n; replace = false, ordered = true), :];
    end

    function fixcolumnnames(df)
        colnames = Symbol[]
        for i in string.(names(df))
            push!(colnames,Symbol(replace(replace(replace(strip(i)," " => "_"),"-" => "_"), "/" => "_")))
        end
        rename!(df, colnames);
    end

    function dropallmissingcolumns(df)
        dropmissing!(df)
    end

    function dropamissingcolumn(df, c)
        dropmissing!(df, c)
    end

    function getcolumnnames(df)
        return names(df)
    end

    function getcolumntypes(df)
        return eltype.(eachcol(df))
    end    

    function getcolumnameswithtypes(df::DataFrame)
        ctypes = getcolumntypes(df)
        cnames = getcolumnnames(df)
        dict = Dict()
        for i in 1:length(ctypes)
            push!(dict, cnames[i] => ctypes[i])
        end    
        return dict
    end    

    function getacolumntype(df, cname) 
        ctypes = getcolumntypes(df)
        cnames = getcolumnnames(df)
        idx = findfirst(isequal(names), cnames)
        return ctypes[i]
    end

    function removeoutliers(df, freq)
        lower_limit = percentile(df.Life_expectancy, freq::Int64);
        iqr_value = iqr(df.Life_expectancy);
        df = df[df.Life_expectancy .>  (lower_limit - 1.5*iqr_value),:];
        return df;
    end

    function getcorrelation(col1, col2)
        return cor(col1, col2)
    end

    function preprocessdata(df, col)
        enc = Lathe.preprocess.OneHotEncoder();
        return enc.predict(df, col);
    end

    function splittraintestdata(df, p::Float64)
        train, test = TrainTestSplit(df, p)
        return train, test
    end    

end