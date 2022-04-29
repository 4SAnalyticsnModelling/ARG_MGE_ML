# This script predicts abundance of ARG and MGE from tuned and trained best models among the trained RF (Random Forest), XGBOOST(Extreme Gradient Boost), and NNET(Artificial Neural Network) regression models
# Install required packages
using Pkg;
Pkg.add(["DataFrames", "CSV", "Statistics", "BSON", "MLJ", "MLJXGBoostInterface", "MLJDecisionTreeInterface", "Flux", "Flux.Zygote"]);
# Import the required packages
using DataFrames, CSV, Statistics, BSON, MLJ, MLJXGBoostInterface, MLJDecisionTreeInterface, Flux, Flux.Zygote;
# Set current directory as working directory
cd(@__DIR__);
# Import custom function files
include("../utility_functions.jl");
include("../cross_validation.jl");
include("../train_flux_regression.jl");
include("../train_mlj_models_regression.jl");
# Predefined functions
function jitter(a, width_jit)
    a = convert.(Float64, a)
    a .+= rand(size(a)...) .* width_jit .- width_jit .* 0.5
    return a
end
function normalize(x)
    return (x .- extrema(x)[1])/(extrema(x)[2] - extrema(x)[1])
end
# Import datasets used as features for prediction of ARG abundance
pred_df = DataFrame(CSV.File("../../assets/features_for_prediction_df.csv", normalizenames = true));
pred_df[!, :operation] .= ifelse.(pred_df.operation .== "Batch", 0.0, 1.0);
pred_df[!, :additives_or_chemicals] .= ifelse.(pred_df.additives_or_chemicals .== "No", 0.0, 1.0);
pred_df[!, :pre_treatment] .= ifelse.(pred_df.pre_treatment .== "No", 0.0, 1.0);
list_cols = [:feedstock_and_co_substrate]
for items in list_cols
    pred_df[!, items] .= MLJ.categorical(pred_df[!, items])
end
hot = MLJ.OneHotEncoder(ordered_factor = true);
mach = MLJ.fit!(machine(hot, pred_df), verbosity = 0);
pred_df1 = MLJ.transform(mach, pred_df);
show(first(pred_df, 6), allcols = true)

# Import tuned and trained NNET Model and performed prediction on ARG abundance
params_of_interest = "ARG";
model_name = "nnet"
J, N = size(pred_df1);
mutable struct nnet_mod_builder
    n1 :: Int
    n2 :: Int
    n3 :: Int
end
function nnet_build(nn :: nnet_mod_builder, n_in, n_out)
    return Flux.Chain(Dense(n_in, nn.n1, relu, init = Flux.kaiming_normal),
                 Dense(nn.n1, nn.n2, relu, init = Flux.kaiming_normal),
                 Dense(nn.n2, nn.n3, relu, init = Flux.kaiming_normal),
                 Dense(nn.n3, n_out, init = Flux.kaiming_normal))
end
nnet_model = nnet_build(nnet_mod_builder(N * 6, N * 4, N * 2), N, 1);
target_location = params_of_interest * "_" * model_name * "_trained_model";
trained_model = readdir(target_location);
trained_model = trained_model[contains.(trained_model, ".bson")][2];
nnet_trained_model_weights = BSON.load(target_location .* "/" .* trained_model);
Flux.loadparams!(nnet_model, nnet_trained_model_weights[:weights]);
trained_scaler = readdir(target_location);
trained_scaler = trained_scaler[contains.(trained_scaler, ".bson")][1];
trained_scaler = BSON.load(target_location .* "/" .* trained_scaler);
pred_df_conv = scale_transform(trained_scaler[:x_scaler], Matrix(pred_df1));
pred_df[!, :predicted_values] = vec(nnet_model(Matrix(pred_df_conv')));
feed_flag_df = DataFrame(feedstock_and_co_substrate = unique(pred_df.feedstock_and_co_substrate),
feed_flag = 1:length(unique(pred_df.feedstock_and_co_substrate)));
pred_df = innerjoin(feed_flag_df, pred_df, on = :feedstock_and_co_substrate);

# Import datasets used as features for prediction of MGE abundance
pred_df = DataFrame(CSV.File("../../assets/features_for_prediction_df.csv", normalizenames = true));
pred_df[!, :operation] .= ifelse.(pred_df.operation .== "Batch", 0.0, 1.0);
pred_df[!, :additives_or_chemicals] .= ifelse.(pred_df.additives_or_chemicals .== "No", 0.0, 1.0);
pred_df[!, :pre_treatment] .= ifelse.(pred_df.pre_treatment .== "No", 0.0, 1.0);
list_cols = [:feedstock_and_co_substrate]
for items in list_cols
    pred_df[!, items] .= MLJ.categorical(pred_df[!, items])
end
hot = MLJ.OneHotEncoder(ordered_factor = true);
mach = MLJ.fit!(machine(hot, pred_df), verbosity = 0);
pred_df1 = MLJ.transform(mach, pred_df);
show(first(pred_df, 6), allcols = true)

# Import tuned and trained NNET Model and performed prediction on MGE abundance
params_of_interest = "MGE";
model_name = "nnet"
J, N = size(pred_df1);
mutable struct nnet_mod_builder
    n1 :: Int
    n2 :: Int
    n3 :: Int
end
function nnet_build(nn :: nnet_mod_builder, n_in, n_out)
    return Flux.Chain(Dense(n_in, nn.n1, relu, init = Flux.kaiming_normal),
                 Dense(nn.n1, nn.n2, relu, init = Flux.kaiming_normal),
                 Dense(nn.n2, nn.n3, relu, init = Flux.kaiming_normal),
                 Dense(nn.n3, n_out, init = Flux.kaiming_normal))
end
nnet_model = nnet_build(nnet_mod_builder(N * 6, N * 4, N * 2), N, 1);
target_location = params_of_interest * "_" * model_name * "_trained_model";
trained_model = readdir(target_location);
trained_model = trained_model[contains.(trained_model, ".bson")][2];
nnet_trained_model_weights = BSON.load(target_location .* "/" .* trained_model);
Flux.loadparams!(nnet_model, nnet_trained_model_weights[:weights]);
trained_scaler = readdir(target_location);
trained_scaler = trained_scaler[contains.(trained_scaler, ".bson")][1];
trained_scaler = BSON.load(target_location .* "/" .* trained_scaler);
pred_df_conv = scale_transform(trained_scaler[:x_scaler], Matrix(pred_df1));
pred_df[!, :predicted_values] = vec(nnet_model(Matrix(pred_df_conv')));
feed_flag_df = DataFrame(feedstock_and_co_substrate = unique(pred_df.feedstock_and_co_substrate),
feed_flag = 1:length(unique(pred_df.feedstock_and_co_substrate)));
pred_df = innerjoin(feed_flag_df, pred_df, on = :feedstock_and_co_substrate);
