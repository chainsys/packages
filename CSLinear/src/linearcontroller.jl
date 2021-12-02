module LinearController

    include("linear.jl");
    include("linearhandler.jl");
    include("utils/modelutils.jl");

    using JSON;
    
    function analyzelinearregression(pdata)
        linearst = LinearStructs.LinearSt(pdata["modelname"], pdata["datafilename"], pdata["pval"], pdata["rval"], pdata["rownum"]);
        mdict = LinearHandler.linear(linearst);
        js = JSON.json(mdict);
        return js;
    end

    function predictlinearregression(pdata)
        @time data = LinearHandler.predictlinear(pdata["modelname"], pdata["pdata"], pdata["pval"], pdata["rval"]);
        data = JSON.json(data);
        return data;
    end  
    
    function getmodelinfo(modelname)
        mu = ModelUtils;
        info = mu.getmodelinfo(modelname);
        return info;
    end

    function getplots(modelname) 
        mu = ModelUtils;
        plots = mu.getplots(modelname);
        plots = JSON.json(plots);
        return plots;
    end

end

