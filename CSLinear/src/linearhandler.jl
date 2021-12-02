module LinearHandler

    include("linearregression.jl");
    include("utils/modelutils.jl");
    include("utils/dfutils.jl");
    include("utils/globalvars.jl");

    using DataFrames, Statistics, StatsBase, StatsModels, Gaston;
    
    mdict = Dict();
    mu = ModelUtils;
    dfu = DfUtils;
    
    function linear(linearst)
        lr = LinearRegression;
        modelname = linearst.modelname;
        fname = linearst.fname;
        pval = linearst.pval;
        rval = linearst.rval;

        rowcount = linearst.rowcount;
        fname = Global.model_path * modelname * "/raw/" * fname; 
        df = dfu.generatedataframe(fname, rowcount, rval);
        push!(mdict, "rows" => size(df));
        
        df = lr.removeoutliers(df, 25);
        cor = lr.getcorrelation(df[!,Symbol(pval)], df[!,Symbol(pval)]);
        push!(mdict, "correlation" => cor);
        
        df = lr.preprocessdata(df, Symbol(rval));
        
        obj = lr.optimizemodel(df, percentile, modelname, pval, rval, rval, pval);
        
        info = replace(string(obj[1]), "\\n" => "</br>");
        push!(mdict, "modelinfo" => info);
        push!(mdict, "rsq" => obj[2]);
        push!(mdict, "ypredicted_train" => string(obj[4]));
        push!(mdict, "ypredicted_test" => string(obj[5]));
        
        ypredicted_train = obj[4];
        ypredicted_test = obj[5];
        
        fn = Global.model_path * modelname * "/plots"
        mkpath(fn);
        
        train = obj[6];
        checkperformance(obj, df, ypredicted_train, ypredicted_test, pval, rval, mu, fn);
        fm = (StatsModels.Term(Symbol(rval)) ~ StatsModels.Term(Symbol(pval)));
        val = mu.crossvalidation(train, 10, fm);
        push!(mdict, "crossvalidation" => val);
        
        mu.savemodelinfo(modelname, mdict);

        return mdict;
    end

    function checkperformance(obj, df, ypredicted_train, ypredicted_test, pval, rval, mu, fn)
        linearfit = obj[3];
        train = obj[6];
        test = obj[7];

        performance_testdf = DataFrame(y_actual = test[!,Symbol(rval)], y_predicted = ypredicted_test);
        performance_testdf.error = performance_testdf[!,:y_actual] - performance_testdf[!,:y_predicted];
        performance_testdf.error_sq = performance_testdf.error.*performance_testdf.error;

        performance_traindf = DataFrame(y_actual = train[!,Symbol(rval)], y_predicted = ypredicted_train);
        performance_traindf.error = performance_traindf[!,:y_actual] - performance_traindf[!,:y_predicted];
        performance_traindf.error_sq = performance_traindf.error.*performance_traindf.error;

        push!(mdict, "Mean Absolute Test Error" => mean(abs.(performance_testdf.error)));
        push!(mdict, "Mean Aboslute Percentage Test Error" => mu.mape(performance_testdf));
        push!(mdict, "Root Mean Square Test Error" => mu.rmse(performance_testdf));
        push!(mdict, "Mean Square Test Error" => mean(performance_testdf.error_sq));

        push!(mdict, "Mean Train Error" => mean(abs.(performance_traindf.error)));
        push!(mdict, "Mean Absolute Percentage Train Error" => mu.mape(performance_traindf));
        push!(mdict, "Root Mean Square Train Error" => mu.rmse(performance_traindf));
        push!(mdict, "Mean Square Train Error" => mean(performance_traindf.error_sq));

        @async doplots(df, train, performance_traindf, performance_testdf, linearfit, pval, rval, fn);
    end

    function doplots(df, train, performance_traindf, performance_testdf, linearfit, pval, rval, fn)
        mu.linearfitplot(train, linearfit, pval, rval, fn);
        mu.plotoutlieranalysis(df[:, rval], "Box Plot - " * rval, rval, false, fn)
        mu.densityplot(df[:, rval], "Density Plot - " * rval, rval, "Frequency", false, fn);
        mu.scatterplot(df[:, pval], df[:, rval], "Scatter Plot - " * rval, pval, rval, false, fn);

        h = histogram(performance_testdf.error, bins = 50, title = "Test Error Analysis", ylabel = "Frequency", xlabel = "Error",legend = false);
        mu.plot2file(h, fn * "/testerrorhis.png");

        h = histogram(performance_traindf.error, bins = 50, title = "Training Error Analysis", ylabel = "Frequency", xlabel = "Error",legend = false);
        mu.plot2file(h, fn * "/trainerrorhis.png");

        test_plot = scatter(performance_testdf[!,:y_actual],performance_testdf[!,:y_predicted], title = "Predicted value vs Actual value on Test Data", ylabel = "Predicted value", xlabel = "Actual value", legend = false);
        mu.plot2file(test_plot, fn * "/testprevsact.png");

        train_plot = scatter(performance_traindf[!,:y_actual],performance_traindf[!,:y_predicted], title = "Predicted value vs Actual value on Train Data", ylabel = "Predicted value", xlabel = "Actual value",legend = false);
        mu.plot2file(train_plot, fn * "/trainprevsact.png");
    end

    function predictlinear(modelname, pdata, pval, rval)
        dfu = DfUtils;
        lr = LinearRegression;
        pdata = DataFrame(c1 = pdata);
        rename!(pdata, :c1 => Symbol(pval));
        traindata = dfu.generatedataframe(Global.model_path * modelname * "/data/" * modelname * "_traindata.csv");
        return lr.predictdata(traindata, pdata, pval, rval);
    end

end