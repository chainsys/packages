module ModelUtils
    
    using DataFrames, Base64, CSV, JSON, Statistics, Images, StatsPlots, MLBase, GLM;

    include("globalvars.jl");
    
    function plottobase64(p) 
        io = IOBuffer();
        iob64_encode = Base64EncodePipe(io);
        show(iob64_encode, MIME("image/png"), p);
        close(iob64_encode);
        str = String(take!(io));
        return str;
    end

    function plotoutlieranalysis(col, titlep, ylabelp, legendp, fn)
        p = StatsPlots.boxplot(col, title = titlep, ylabel = ylabelp, legend = legendp);
        fn = fn * "/outlier.png";
        plot2file(p, fn);
    end

    function densityplot(col, titlep, xlabelp, ylabelp, legend, fn)
        p = StatsPlots.density(col, title = titlep, ylabel = ylabelp, xlabel = xlabelp, legend = legend);
        fn = fn * "/density.png";
        plot2file(p, fn);
    end

    function scatterplot(col1, col2, titlep, xlabelp, ylabelp, legend, fn)
        p = StatsPlots.scatter(col1, col2, title = titlep, ylabel = ylabelp, xlabel = xlabelp,legend = legend);
        fn = fn * "/scatter.png";
        plot2file(p, fn);
    end

    function linearfitplot(traindata, linearfit, x, y, fn)
        plot(traindata[:,x],linearfit)
        s = StatsPlots.scatter!(traindata[:,x],traindata[:,y], xaxis=x, yaxis=y);
        fn = fn * "/filplot.png";
        plot2file(s, fn);
    end

    function writedataset(modelname, traindata, testdata)
        path = mkpath(Global.model_path * modelname * "/data");
        path = path * "/";
        CSV.write(path * modelname * "_traindata.csv", traindata);
        CSV.write(path * modelname * "_testdata.csv", testdata);  
    end 

    function plot2file(plot, fn)
        savefig(plot, fn);
    end

    function mape(performance_df)
        mape = mean(abs.(performance_df.error./performance_df.y_actual))
        return mape
    end

    function rmse(performance_df)
        rmse = sqrt(mean(performance_df.error.*performance_df.error))
        return rmse
    end

    function crossvalidation(train,k, fm)
        val = [];
        a = collect(Kfold(size(train)[1], k))
        for i in 1:k
            row = a[i];
            temp_train = train[row,:];
            temp_test = train[setdiff(1:end, row),:];
            linearregressor = lm(fm, temp_train);
            performance_testdf = DataFrame(y_actual = temp_test[!,:Life_expectancy], y_predicted = predict(linearregressor, temp_test));
            performance_testdf.error = performance_testdf[!,:y_actual] - performance_testdf[!,:y_predicted];
            v = mean(abs.(performance_testdf.error));
            push!(val, "Mean error for set $i is $v");
        end
        return val;
    end

    function savemodelinfo(modelname, mdict::Dict)
        path = mkpath(Global.model_path * modelname * "/info");
        path = path * "/modelinfo.txt";
        js = JSON.json(mdict);
        open(path, "w+") do file
            write(file, js);
        end
        return js;
    end

    function getmodelinfo(modelname)
        path = mkpath(Global.model_path * modelname * "/info");
        path = path * "/modelinfo.txt";
        s = open(f->read(f, String), path);
        return s;
    end

    function getplots(modelname)
        d = Global.model_path * modelname * "/plots";
        idict = Dict();
        foreach(readdir(d)) do f
            p = load(d*"/"*f)
            io = IOBuffer();
            iob64_encode = Base64EncodePipe(io);
            show(iob64_encode, MIME("image/png"), p);
            close(iob64_encode);
            str = String(take!(io));
            push!(idict, f => str);
            close(io);
            close(iob64_encode)
        end
        return idict;
    end

end