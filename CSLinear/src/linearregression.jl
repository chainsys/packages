module LinearRegression

    include("utils/dfutils.jl")
    include("utils/modelutils.jl")
    

    using DataFrames, CSV, Lathe, GLM, Statistics, MLBase;

    dfu = DfUtils;
    mu = ModelUtils;

    function removeoutliers(df::DataFrame, thershold::Int64)
        df = dfu.removeoutliers(df, thershold);
        return df;
    end    
    
    function getcorrelation(dfcolx, dfcoly)
        return dfu.getcorrelation(dfcolx, dfcoly);
    end

    function preprocessdata(df, ycolname)
        df = dfu.preprocessdata(df, ycolname);
        return df;
    end

    function buildmodel(traindata, x, y)
        fm = (StatsModels.Term(Symbol(y)) ~ StatsModels.Term(Symbol(x)))
        linearregressor = lm(fm, traindata);    
        return linearregressor;
    end

    function optimizemodel(df, percentile, name, x, y, xlabel, ylabel)
        obj = []; 
        mu = ModelUtils;
        
        data = dfu.splittraintestdata(df, .75);
        traindata = data[1]
        testdata = data[2] 
        @async mu.writedataset(name, traindata, testdata);
        
        linearregressor = buildmodel(traindata, x, y);
        push!(obj, linearregressor);
        rsq = r2(linearregressor);
        push!(obj, rsq);
        
        linearfit = predict(linearregressor);
        push!(obj, linearfit);
        
        ypredicted_train = predict(linearregressor, traindata);
        push!(obj, ypredicted_train);
        
        ypredicted_test = predict(linearregressor, testdata);
        push!(obj, ypredicted_test);
        
        push!(obj,traindata);
        push!(obj,testdata);
        
        return obj;
    end

    function predictdata(traindata, testdata, x, y)
        linearregressor = buildmodel(traindata, x, y);
        return predict(linearregressor, testdata);
    end

end