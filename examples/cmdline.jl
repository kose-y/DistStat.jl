using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--gpu"
            help = "use gpu"
            action = :store_true
        "--rows"
            help = "number of rows"
            arg_type = Int
            default = 10_000
        "--cols"
            help = "number of cols"
            arg_type = Int
            default = 10_000
        "--iter"
            help = "number of iterations"
            arg_type = Int
            default = 1000
        "--step"
            help = "interval of checking monitored values"
            arg_type = Int
            default = 100
        "--r"
            help = "intermediate size"
            arg_type = Int
            default = 20
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
    end

    return parse_args(s)
end
