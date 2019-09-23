using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--gpu"
            help = "use gpu"
            action = :store_true
        "--iter"
            help = "number of iterations"
            arg_type = Int
            default = 1000
        "--step"
            help = "interval of checking monitored values"
            arg_type = Int
            default = 100
        "--seed"
            help = "seed"
            arg_type = Int
            default = 777
        "--Float32"
            help = "use Float32 instead of Float64"
            action = :store_true
        "--set_zero_subnormals"
            help = "set subnormals to zero"
            action = :store_true
        "--init_from_master"
            help = "use centralized random initialization (costly)"
            action = :store_true
        "--eval_obj"
            help = "evaluate objective function (costly). Maximum difference from the previous iteration is printed otherwise."
            action = :store_true
    end

    return s
end

function parse_commandline_nmf()
    s = parse_commandline()
    @add_arg_table s begin
        "--rows"
            help = "number of rows"
            arg_type = Int
            default = 10_000
        "--cols"
            help = "number of cols"
            arg_type = Int
            default = 10_000
        "--r"
            help = "intermediate size"
            arg_type = Int
            default = 20
    end
    return parse_args(s)
end

function parse_commandline_mds()
    parse_commandline_nmf()
end

function parse_commandline_cox()
    s = parse_commandline()
    @add_arg_table s begin
        "--rows"
            help = "number of rows"
            arg_type = Int
            default = 10_000
        "--cols"
            help = "number of cols"
            arg_type = Int
            default = 10_000
        "--censor_rate"
            help = "rate of censored subjects"
            arg_type = Float64
            default=0.5
        "--lambda"
            help = "regularization parameter"
            arg_type = Float64
            default=0.0
    end
    return parse_args(s)
end

function parse_commandline_pet()
    s = parse_commandline()
    @add_arg_table s begin
        "--data"
            help = "path to datafile"
            arg_type = String
        "--reg"
            help = "regularization parameter"
            arg_type = Float64
            default=0.0
    end
    return parse_args(s)
end