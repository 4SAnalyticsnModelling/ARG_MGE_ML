# This script trains RF (Random Forest), XGBOOST(Extreme Gradient Boost), and NNET(Artificial Neural Network) regression models to predict abundance of ARG
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
# Import the dataset for model training
arg_mge_df = DataFrame(CSV.File("../../assets/arg_mge_data_for_ML.csv", normalizenames = true));
# Process the dataset for training
arg_df0 = arg_mge_df[!, [:temperature, :residence_time, :operation, :feedstock_and_co_substrate, :additives_or_chemicals, :pre_treatment, :arg_removal_efficiency]];
arg_df0[!, :operation] .= ifelse.(arg_df0.operation .== "Batch", 0.0, 1.0);
arg_df0[!, :additives_or_chemicals] .= ifelse.(arg_df0.additives_or_chemicals .== "No", 0.0, 1.0);
arg_df0[!, :pre_treatment] .= ifelse.(arg_df0.pre_treatment .== "No", 0.0, 1.0);
list_cols = [:feedstock_and_co_substrate]
for items in list_cols
    arg_df0[!, items] .= MLJ.categorical(arg_df0[!, items])
end
# One hot encoding
hot = MLJ.OneHotEncoder(ordered_factor = true);
mach = MLJ.fit!(machine(hot, arg_df0), verbosity = 0);
arg_df = MLJ.transform(mach, arg_df0);
show(first(arg_df, 6), allcols = true)

# ML training for predicting abundance in ARG
x = arg_df[:, 1:13];
y = arg_df[:, 14];
param_of_interest = "ARG";

# Random Forest (RF) model training and cross-validation
model_name = "random_forest";
# Build the model
forest_model = MLJDecisionTreeInterface.RandomForestRegressor();
# 20% holdout cross-validation
mod_perform_df = mlj_mod_eval(forest_model, x, y, param_of_interest * "_" * model_name * "_training_performance_holdout", forest_model.n_trees, :ntree, 155:1:155, Holdout(eachindex(y), 0.8, true, 500));
# 5-fold cross-validation
mod_perform_df = mlj_mod_eval(forest_model, x, y, param_of_interest * "_" * model_name * "_training_performance_kfold", forest_model.n_trees, :ntree, 155:1:155, KFold(eachindex(y), 5, true, 100));
# Final trained model with tuned hyperparameters
mod_perform_df = mlj_mod_eval(forest_model, x, y, param_of_interest * "_" * model_name * "_trained_model", forest_model.n_trees, :ntree, 155:1:155);

# Extreme Gradient Boost (XGBOOST) model training and cross-validation
model_name = "xgboost";
# Build the model
xgboost_model = MLJXGBoostInterface.XGBoostRegressor();
# 20% holdout cross-validation
mod_perform_df = mlj_mod_eval(xgboost_model, x, y, param_of_interest * "_" * model_name * "_training_performance_holdout", xgboost_model.eta, :etas, 0.1:0.1:0.1, Holdout(eachindex(y), 0.8, true, 500));
# 5-fold cross-validation
mod_perform_df = mlj_mod_eval(xgboost_model, x, y, param_of_interest * "_" * model_name * "_training_performance_kfold", xgboost_model.eta, :etas, 0.1:0.1:0.1, KFold(eachindex(y), 5, true, 100));
# Final trained model with tuned hyperparameters
mod_perform_df = mlj_mod_eval(xgboost_model, x, y, param_of_interest * "_" * model_name * "_trained_model", xgboost_model.eta, :etas, 0.1:0.1:0.1);

# Artficial Neural Network (NNET) model training and cross-validation
model_name = "nnet"
J, N = size(x);
# Build the model
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
# 20% holdout cross-validation
flux_mod_eval(nnet_build(nnet_mod_builder(N * 6, N * 4, N * 2), N, 1), x, y, param_of_interest * "_" * model_name * "_training_performance_holdout", Holdout(eachindex(y), 0.8, true, 500), 100, true, standard_scaler(), nothing, 5, trunc(Int64, round(J * 0.8)), 3, 2, Flux.Losses.mse, Flux.Optimise.ADAM(0.001));
# 5-fold cross-validation
flux_mod_eval(nnet_build(nnet_mod_builder(N * 6, N * 4, N * 2), N, 1), x, y, param_of_interest * "_" * model_name * "_trained_model", nothing, 500, false, standard_scaler(), nothing, 10, length(y), 3, 2, Flux.Losses.mse, Flux.Optimise.ADAM(0.001));
# Final trained model with tuned hyperparameters
flux_mod_eval(nnet_build(nnet_mod_builder(N * 6, N * 4, N * 2), N, 1), x, y, param_of_interest * "_" * model_name * "_training_performance_kfold", KFold(eachindex(y), 5, true, 100), 100, true, standard_scaler(), nothing, 5, div(J, 5) * 4, 3,
2, Flux.Losses.mse, Flux.Optimise.ADAM(0.001));
